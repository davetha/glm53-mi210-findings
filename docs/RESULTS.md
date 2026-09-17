# Results: GLM-5.3-Flash on 2x MI210

## Standing configuration

    MODEL_DIR=/mnt/llm-storage/glm53-w4a16-mtp     # 182 GB, W4A16 int4
    TP=2  --enable-expert-parallel
    UTIL=0.97  MAXLEN=8192  MAX_NUM_SEQS=1
    OFFLOAD_GB=45          # 26 of 42 MoE layers on host
    SLOTS=52               # 52 of 144 EP-local experts resident per armed layer
    POLICY=lfu  DECAY=256  SPEC=off
    --max-num-batched-tokens 4096
    --kv-cache-memory=644245094   # 0.6 GiB -> 15,753 tokens (MAXLEN 8192)
    GLM53_FASTPATH_PARTS=gate,topk,nodoublegate
    wide scratch ON

Full recipe with every flag and patch explained: `docs/CONFIGURATION.md`.

`fp_launch.sh` carries these as defaults.

## Where it started and where it ended

| metric            | session start | final  | change |
|-------------------|---------------|--------|--------|
| decode tok/s      | 10.96         | ~13.1  | +20%   |
| prefill @500      | 138.9         | 233.0  | +68%   |
| prefill @1000     | 216.4         | 413.7  | +91%   |
| prefill @2000     | 316.8         | 699.5  | +121%  |
| prefill @4000     | 264.9         | 675.1  | +155%  |
| prefill @7000     | 183.7         | 603.3  | +228%  |

Decode is measured over 3 interleaved container restarts per arm (see
`docs/METHODOLOGY.md` #1). Prefill uses unique prompts to defeat the prefix cache.

## What produced the gains

### 1. Resident-layer decline — a bug, not a tuning miss

The expert cache armed every MoE layer, including those whose experts never left
the GPU. vLLM's UVA offloader is whole-layer and budget-limited, so at
`--cpu-offload-gb 28` only ~17 of 42 layers were on the host. The other 25 carried
an 8-slot buffer they could never miss into, plus a manager and a gather every step
copying VRAM to itself.

    offloaded per rank       29.07 GiB
    one MoE layer of experts  1.74 GiB/rank (144 EP-local experts x 12.38 MiB)
    => 16.7 offloaded, 25.3 resident -- confirmed exactly by the fix:
       "active on 17 MoE layer(s) (25 declined)"
    dead slot VRAM           25 x 8 x 12.38 MiB = 2.42 GiB/rank

The offloader leaves `.device` reading as the accelerator (an offloaded parameter
is a device-addressable VIEW of pinned host memory), so `_vllm_is_uva_offloaded` is
the only reliable signal.

### 2. Operating point: more offload, far more slots

Freeing that VRAM let the slot count rise. Hit rate vs slots, from 12,600 real
routing refreshes (`cache_sim.py` on a `GLM53_TRACE_DUMP` trace):

    S    best online hit   OPT hit   misses/layer-step   PCIe MiB/step
    8         33.4%         47.2%        2.704               569
   16         45.6%         61.5%        2.207               464
   32         59.2%         75.5%        1.655               348
   48         68.8%         82.9%        1.267               266
   96         87.3%         93.0%        0.518               109

At ~24 GB/s the old 569 MiB/step was ~24 ms of a ~91 ms step.

VRAM ceiling found empirically: expert VRAM (resident layers + slot buffers) tops
out near 47 GiB. 43.5+3.29 boots; 43.5+4.93 arms fine then dies in cudagraph
capture. So raising slots REQUIRES converting resident layers into offloaded ones.

### 3. Prefill chunk 4096

Cost tracks the DISTINCT experts each chunk routes to, so larger chunks help by
shrinking the tail chunk. 8192 does not fit at SLOTS=56 (OOM, needs 1.27 GiB more
than available) — chunk size and slot count are two positions on the same VRAM dial.

### 4. Wide-step VRAM scratch — the prefill 3x

Prefill bypasses the cache on any step wider than SLOTS, so the fused MoE GEMM read
expert weights directly from pinned host memory. Measured cost, same 2070-token
prompt at three offload levels: 6,535 / 9,089 / 11,174 ms at 17 / 22 / 26 offloaded
layers = **515 ms per offloaded layer**, or ~6.8 GB/s. The gather kernel sustains
21-25 GB/s on the same link.

Fix: stage the whole local expert set into VRAM with `expert_cache_gather_k`, then
compute from there. The scratch is full-size and indexed BY LOCAL EXPERT ID exactly
as the source is, so expert_map, global_num_experts and the routing ids all keep
their meaning — nothing is remapped. No kernel change needed: the gather already
reads its work list as (expert, slot) pairs, so an identity list copies expert e to
row e. One 1.74 GiB allocation is shared by all layers (they run in order on one
stream, so a refill cannot overtake a read).

Cost: 1.74 GiB of VRAM, paid by dropping SLOTS 56 -> 46, about 2.4 ms of decode.

## Closed by measurement — do not re-litigate

Cache policy / decay / algorithms (within 1.5% of the best online policy;
~13pp below Belady regardless) · gather geometry (flat 86.6-90.6 ms) · lossless
compression, both PCIe and VRAM-resident (Huffman decode cost 17x the transfer it
saved; structural, not an implementation bug) · block dedup (zero duplicates) ·
vision-tower offload (HSA memory fault) · torch.compile (lost) · shared-expert
overlap (lost) · MoE autotuning (6% worse) · uint8 layout + int4 GEMV · PCIe link
health (Gen4 x16 confirmed) · kernel fusion of the small-kernel block (removed 126
kernels and ~1.13 ms of GPU busy time per step; wall clock unmoved, see
METHODOLOGY #7).

`expert_cache_fused_k` is built and exported but unused: it targets a hot/cold GEMM
split that does not exist in this branch, and its prize (42 moe_align calls,
437.8 us) is inside the noise floor.


---

## 5. Expert cache slots: 46 -> 52 (2026-09-17)

**This section corrects an earlier claim.** "Slots" used to appear in the closed-by-
measurement list above. That verdict was reached at a different operating point
(OFFLOAD 76 / 42 layers armed) and does not hold at the current one:

    config                  median   min     max     IQR    vs baseline
    SLOTS=46                 79.50   78.64   83.13   1.30   --
    SLOTS=52                 76.38   75.74   77.54   1.30   -3.13 ms
    SLOTS=52 + KV cap        76.14   75.61   77.04   0.97   -3.36 ms  <- standing
    SLOTS=58 + KV cap        80.08   74.58   85.97   5.16   UNSTABLE
    SLOTS=60                 LAUNCH FAILED
    SLOTS=64 + KV cap        ILLEGAL MEMORY ACCESS at engine init

The 46 and 52 samples do not overlap (46's min 78.64 > 52's max 77.54), so this is not
drift. Mechanism is the expected one: the gather is 15.43 ms/step driven by a 1.29
misses/layer-step rate, and the gather is already at the PCIe wall (28.1 GB/s, verified
three independent ways), so the only remaining lever is missing less often.

The VRAM for those 6 slots comes from `--kv-cache-memory=644245094`: vLLM was holding
2.52 GiB of KV where 0.6 GiB gives 15,753 tokens against a MAXLEN of 8192. Verified with
a 4021-token prompt, not only short-answer gates.

**The cache degrades above ~52 slots, and not gracefully.** 58 serves and is numerically
correct but its timings are SCATTERED rather than drifting (81.77 76.36 76.91 80.20 77.30
85.97 79.96 82.07 80.45 82.78 75.67 74.58). 60 will not launch. 64 dies with an illegal
memory access, which is not what a clean OOM looks like. Isolated: SLOTS=52 with the
identical KV cap is the tightest sample of the session (IQR 0.97, monotonic), so the KV
cap is innocent and the slot count is the variable. Possible bug in vllm-expert-cache at
high slot counts, worth chasing independently of performance.

## 6. hostar: a host-staged all-reduce for PCIe-only GPUs (2026-09-17)

Optional, off by default (`VLLM_HOSTAR=1`), worth a further ~3.5 ms/step.

A GPU-to-GPU flag handshake in PEER VRAM does not work between these cards: writes land
once a kernel ends, but a concurrently-spinning kernel stalls intermittently. Six
store/load memory-ordering combinations all fail, and disabling PCIe ACS redirect changes
nothing. The same handshake through PINNED HOST MEMORY runs 2000 round trips clean at
4.5 us -- which is also what NCCL does for PCIe-only peers.

Measured in-model (CUDA-graph trace, 19,928 hostar_k calls, 4/4 correctness):
all-reduce **14.45 -> 10.94 ms/step**.

Not the ~16 ms the microbenchmark implied, and the gap is the finding: per-call p50 is
13.6 us but p90 is 405 us, and that tail is one rank spinning while the other catches up.
Under expert parallelism the ranks route to different experts and do different amounts of
gather work per layer. NCCL pays the same wait inside its 185.68 us average. **The
remaining all-reduce time is load imbalance, not communication.**

See `docs/HOSTAR-INTEGRATION.md` and `hostar/`.

## Still open, ranked

1. **mHC: 4.2 ms/step on ONE workgroup.** `dim3 grid(m_blocks)` in AITER's
   `mhc_pre_big_fuse_rmsnorm` parallelises over tokens, and decode has exactly one, so
   4.2 ms of work runs on 1 of 104 CUs. Needs a batch-1 variant parallelising over hidden
   or residual streams. Largest clearly-wasteful item left.
2. **Rank skew.** Now visible three ways: hostar's 405 us p90, 2.60 ms/step of
   post-all-reduce idle, and the residual after replacing NCCL. Caused by data-dependent
   expert routing, so not fixable by faster communication.
3. Prefill numerical instability at temperature 0 (pre-existing, not from this work).


## Speculative decoding (MTP): closed 2026-09-17

Layer 45 IS a complete MTP draft layer -- but it shipped BF16 (14.87 GB, 7.43 GB/rank)
while the model is W4A16 int4, so it cannot be resident and must be offloaded at ~400 MB
of PCIe per draft step.

It OOMs at the standing config, dies in CUDA graph capture at OFFLOAD=62/SLOTS=36, and in
eager mode it SERVES but generates degenerate looping text (correctness 0/3). A control at
the identical config with SPEC=off is clean, so MTP is the variable, not the cache or the
offload level.

Measured acceptance: mean length 1.17-1.77 (typically ~1.3), avg rate 17-77%. Break-even
needs the MTP step under ~1.3x the base; the offloaded BF16 draft alone adds ~18%. Even
with the verification bug fixed and the draft quantised to int4 (~1.95 GB/rank, would fit),
this is break-even at best.

Full detail, including what would have to be true to revisit: docs/SPECULATIVE-DECODING.md
