# Results: GLM-5.3-Flash on 2x MI210

## Standing configuration

    MODEL_DIR=/mnt/llm-storage/glm53-w4a16-mtp     # 182 GB, W4A16 int4
    TP=2  --enable-expert-parallel
    UTIL=0.97  MAXLEN=8192  MAX_NUM_SEQS=1
    OFFLOAD_GB=45          # 26 of 42 MoE layers on host
    SLOTS=46               # 46 of 144 EP-local experts resident per armed layer
    POLICY=lfu  DECAY=256  SPEC=off
    --max-num-batched-tokens 4096
    GLM53_FASTPATH_PARTS=gate,topk,nodoublegate
    wide scratch ON

`fp_launch.sh` carries these as defaults.

## Where it started and where it ended

| metric            | session start | final  | change |
|-------------------|---------------|--------|--------|
| decode tok/s      | 10.96         | ~11.8  | +7%    |
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

Cache policy / decay / slots / algorithms (within 1.5% of the best online policy;
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
