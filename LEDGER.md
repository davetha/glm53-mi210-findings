# VALIDATED RESULT (clean host, sequential, 5 runs, empty queue)

    TP=2  --enable-expert-parallel  --enforce-eager
    OFFLOAD_GB=33  UTIL=0.95  MAXLEN=8192  MAX_NUM_SEQS=1
    SLOTS=8  POLICY=lfu  SPEC=off

    124.87 126.83 128.68 128.45 127.70 ms/step  ->  7.77-8.01 tok/s
    correctness 3/3 on the chat-templated arithmetic probe

Session start was 2.2 tok/s (and before that, no correct output at all).
That is 3.6x. An earlier 8.36 figure was a best-of-4 and should not be quoted.

## TWO WAYS I CORRUPTED MEASUREMENTS TONIGHT (both produced plausible numbers)
1. Overlapping benchmark loops -> requests queued behind each other, 201-303 ms
   readings that were pure queueing. Check "Waiting: N reqs" in the engine log.
2. Host paging -> offloaded weights live in host RAM; when swap filled, UVA reads
   faulted to disk and runs went bimodal (132 then 242 ms). Check free -g / vmstat.
The engine's own "Avg generation throughput" is immune to both and is the
cross-check to trust.

# GLM-5.3-Flash on 2x MI210 -- overnight optimisation ledger

Config unless stated: TP=2, --enable-expert-parallel, --enforce-eager, W4A16,
MAXLEN 8192, MAX_NUM_SEQS 1, SPEC off, chat-template correctness gate.

## Measured (ms/step, best of 3, 256-token generations)

| # | change | offload | slots | ms/step | tok/s | note |
|---|--------|---------|-------|---------|-------|------|
| 0 | plain TP=2 | 88 | off | - | - | WRONG OUTPUT (MoE kernel OOB) |
| 1 | +EP (correct) | 88 | off | ~455 | 2.2 | first correct 2-card run |
| 2 | offload 26 | 26 | off | 201.3 | 4.97 | stale offload value was starving VRAM |
| 3 | offload 45 | 45 | off | 322.8 | 3.10 | control: offload costs 6.4 ms/GiB |
| 4 | +expert cache (EP support written) | 45 | 32 | 141.1 | 7.09 | |
| 5 | +remove per-layer sync | 45 | 32 | 132.8 | 7.53 | 42 syncs/step removed |
| 6 | +gather chunks=256 | 45 | 32 | 128.8 | 7.76 | occupancy: only 3% |
| 7 | more slots | 62 | 64 | 144.0 | 6.94 | residency flat, no gain |
| 8 | cudagraphs (FULL_DECODE_ONLY) | 28 | off | 217.9 | 4.59 | slower AND less correct |
| 9 | DMA gather (clean A/B vs kernel 131) | 45 | 32 | 134-137 | 7.3-7.5 | NO GAIN. copy is not the bottleneck |

| 10 | NO-GATHER diagnostic (wrong output, timing only) | 45 | 32 | 120.7 | 8.28 | gather is only ~8 ms of 128.8 |

## THE BIG ONE (found 02:xx via cache-repo audit)
vllm-expert-cache README:159-163 -- "Graph capture is worth about 4x, so staying
capturable dominates every other design consideration." docs/results.md: a variant
whose only difference is a per-step D2H sync stays at ~14 tok/s where the capturable
one reaches 54.3.

WE HAVE BEEN RUNNING --enforce-eager ALL SESSION.

Blocker: capture dies in fp8_paged_mqa_logits_torch (the DSA torch fallback we
switched to for correctness) because it calls .item() per batch element.
  -> vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:505
Fixing that to be capture-safe is now the top item.

Also: _gather_dma() adds a per-layer stream.synchronize(), which breaks the same
invariant (cache.py:13). The pushed DMA commit must stay default-off.

| 11 | cudagraphs + cache + EP (AITER DSA) | 45 | 32 | 142-228 | 4.4-7.0 | SLOWER, variable, 2/4 correct |

## Why cudagraphs do not help THIS model
vLLM's splitting_ops excludes every attention op (unified_mla_attention_with_output,
sparse_attn_indexer, linear_attention, mamba_mixer...). GLM-5.3 has 45 hybrid
attention layers, so the graph breaks 45x per forward and only the fragments
between get captured. The 4x in the cache README was Qwen3.8-Flash-Next on R9700 --
different architecture, does not transfer.

## STEP BUDGET (measured, 5600 layer-calls, both ranks)
refresh 0.61 ms/call, moe 0.375 ms/call, over 42 MoE layers:
  cache refresh (manage+gather)  ~25.6 ms   (of which gather ~8, manager ~17.6)
  MoE proper                     ~15.8 ms
  EVERYTHING ELSE (45 attn layers, norms, router, shared expert) ~87 ms = 68%
=> attention dominates at ~1.9 ms/layer; the manager is the biggest single
   actionable item at 14% of the step.
NOTE: repo docs/results.md measures the manager at 11.3 us/layer; we see ~420 us.
   37x discrepancy -- unexplained, worth chasing.

## THE RESIDENCY/SLOTS TRADE (the night's real result)
| offload | slots | resident experts | ms/step | tok/s |
|---------|-------|------------------|---------|-------|
| 62 | 64 | ~0    | 144.0 | 6.94 |
| 45 | 32 | 37.6G | 128.8 | 7.76 |
| 31 |  8 | 51.6G | 118.8 | 8.41 |
| 29 |  8 | 53.6G | 134-241 UNSTABLE | 4.2-7.4 |
| 26 |  8 | 56.6G | OOM | - |
| 33 |  8 | 49.6G | 119.6-125.2 | 8.36 | <-- RECOMMENDED (4.09 GiB KV headroom, 3/3 correct)
| 37 | 16 | 45.6G | (measuring, equal-VRAM slots test) | |

There is a cliff between offload 31 and 29: below ~2 GiB of KV headroom the run
goes bimodal (134 ms once, 240 ms twice) rather than degrading smoothly. 33 is
the safe operating point; 31 is marginally faster but halves the headroom.

Monotone: fewer slots + less offload wins. The cache is NOT valuable as
residency -- it is valuable as a STAGING BUFFER that turns fine-grained UVA
reads into bulk copies into VRAM. Proof: offload 26 with NO cache = 201.3 ms,
offload 31 with 8 slots = 118.8 ms. Less resident, far faster.
Every slot beyond the per-layer working set (~4 owned experts under EP)
displaces a resident expert for no benefit.

NOTE this inverts the sizing guidance in the cache README, which was derived on
Qwen3.8-Flash-Next/R9700 where the model FITS and slots are nearly free. For a
model that overflows VRAM the two cases pull in opposite directions.

## MEASUREMENT VALIDITY WARNING (found 07:00)
Host swap is EXHAUSTED (7/7 GB) and vmstat shows active paging (si=75 so=139).
  glm53 container   146.2 GiB
  q38fn-lru (PROD)  136.9 GiB
  page cache        369 GiB
Offloaded expert weights are read from host pages via UVA. Under reclaim those
reads fault to DISK. That is the bimodal pattern seen in two arms: first run
~132 ms, later runs ~242 ms.

Affected: offload 29 (134/241/235) and offload 37+16slots (132/243/243).
Both were read as config effects; they are more likely host paging.
The offload-33 arm was stable (119.6-125.2, spread 5.5 ms) so it is probably
clean, but the sweep as a whole is not trustworthy point-by-point.

ALSO: these runs share the box with the user's production q38fn-lru service.

## Falsified hypotheses
- launch overhead dominates -> cudagraphs gave nothing
- cache miss rate dominates -> doubling slots gave nothing (residency was flat, my error)
- gather bandwidth dominates -> 4.7x faster copies gave nothing
- gather occupancy -> chunks 16->256 gave 3%

## Established
- Hit rate 62% (38,800 layer-steps, both ranks)
- Isolated transfer: kernel 5.8 GB/s, DMA 27.6 GB/s -- but end-to-end insensitive
- Residency ceiling ~54%: 61.6 GiB experts/rank vs ~36.8 GiB spare VRAM
- Plain TP=2 faults in fused_moe_kernel_gptq_awq (OOB); EP avoids it
- Correctness needs the chat template; raw completions mislead

## Open
- Where do the 128.8 ms actually go? (no-gather diagnostic in flight)

## VALIDATED: indexer page size was CUDA-shaped on ROCm (2026-09-14)

`Glm5NextIndexerCache.get_kv_cache_spec()` — vllm/model_executor/models/glm5next/nvidia/attention.py:124-156.
`glm5next/__init__.py:9` imports the *nvidia* class on every non-XPU platform; there is no amd/attention.py.

The function picks the indexer's KV page from DeepGEMM's `PAGED_MQA_PAGE_SIZES = (32, 64)`,
a CUDA kernel constraint. On ROCm the paged-MQA work runs through AITER, whose only
constraint is a 16-entry preshuffle tile.

GLM-5.3 forces attention block_size=2176 (KDA/mamba page match). index_kpool=4
=> storage_block_size = 2176/4 = 544. 544 % 64 != 0 and 544 % 32 != 0 with the
max-page rule, so it fell to page_size=32 => indexer block 128 vs model block 2176
=> reconciliation factor 17 in `block_table[:, ::factor] // factor` (indexer.py:1059).
Unbounded stride -> OOB in _kpool_softmax_rotate_write_cache_kernel above ~700 prompt tokens.

FIX (patched_glm5next_attention.py): on ROCm, if storage_block_size % 16 == 0,
take page_size = storage_block_size. Factor becomes 1; reconciliation is a no-op.

RESULT: prompt ceiling removed entirely.
  before: 839 tok faulted both GPUs
  after:  839/965/1105/1513/2213/3013/4013/6013/7813 all OK
  prefill peaks ~951 tok/s (3013 tok)
Cost: mamba page padding 1.49% -> 2.64%.

This is the 4th CUDA-shaped-code-on-ROCm bug of the session (AITER gates, fp8 tl.dot,
is_cuda() capability tests, this).

## VALIDATED: the MoE path is only 38% of the decode step (2026-09-14)

Instrumented run (glm53-stats, EXPERT_CACHE_STATS=200 EXPERT_CACHE_TIMING=200),
same validated config: TP=2 EP, eager, offload 33, slots 8, lfu, maxlen 8192.

  refresh 0.700-0.723 ms/call | moe 0.373-0.403 ms/call | remap 0.021 ms/call
  => ~1.10-1.15 ms per MoE layer-call x 42 MoE layers = ~47 ms
  hit rate: TP0 34.2%, TP1 54.8%   <- ranks differ; the slow rank gates the step

Decode step is 125 ms. So:
  MoE path (gather + kernel + remap) = ~47 ms  (38%)
  EVERYTHING ELSE                    = ~78 ms  (62%)

This retires the expert cache as the primary target and explains why six
bandwidth/residency/DMA/occupancy hypotheses all returned null: each was aimed
at 38% of the step. ~78 ms over 45 attention layers is ~1.7 ms/layer for a
SINGLE token, where the KDA recurrent state update is microseconds of real
arithmetic -> suspect launch/dispatch overhead under forced-eager, not compute.

Next: patched_glm5next_model.py adds per-layer CUDA-event timing bucketed by
attention kind (GLM53_LAYER_TIMING=N). Inert unless the env var is set.

## ESTABLISHED: every MoE kernel on gfx90a runs an untuned fallback
  vLLM Triton fused_moe configs:      0 of 333 target gfx90a/MI210
  AITER tuned_fmoe.csv:               0 of 617 (cu_num=80 only)
  AITER tuned_grouped_fmoe.csv:       0 of 111 (gfx1250 only)
vLLM's autotuner exists at /opt/vllm/benchmarks/kernels/benchmark_moe.py.
Shape to tune: E=144 (288/2 under EP2), N=2048, K=4096, W4A16 compressed-tensors.

## CORRECTIONS to earlier claims in this ledger
- "there is no amd/attention.py": an amd/ package DOES exist, but holds only
  ops/ (kpool_compress + third_party). No attention.py/model.py. __init__.py
  imports .nvidia.model unconditionally for non-XPU. Substance unchanged.
- This build's module path is vllm/models/glm5next/..., NOT
  vllm/model_executor/models/glm5next/... The launcher mount is correct;
  patch verified live at attention.py:148 (is_rocm carve-out).
- KDA IS correctly ROCm-aware: nvidia/kda.py:42 imports amd.ops.third_party.kda
  under current_platform.is_rocm(). Not a CUDA-shaped bug.
- vLLM's torch profiler is absent in this build: VLLM_TORCH_PROFILER_DIR is an
  unknown env var and /start_profile returns 404. Use in-process CUDA events.

## OPERATIONAL
Each relaunch pays a ~10-15 min AITER JIT rebuild (ninja -j24, FMHA variants),
because the aiter jit cache lives in the container fs, not a mounted volume.
Mounting it to /cache would make the tune loop much faster. Not yet done.

## VALIDATED: the decode step, fully localized (2026-09-14)

patched_glm5next_model.py wraps each decoder layer in CUDA events bucketed by
attention kind (GLM53_LAYER_TIMING=N). NOTE ON READING IT: the counters
accumulate from step 1, so the PRINTED mean is dominated by ~40 s of first-call
Triton warmup. Take the DIFFERENCE of two successive reports to get the
steady-state window. Two windows, independently:

  window 40->80      window 240->280      layers   ms/layer
  85.9 ms            83.9 ms              31       2.71   KDA + MoE
  42.6 ms            40.9 ms              11       3.72   MLA sparse + MoE
   3.9 ms             3.9 ms               3       1.31   KDA + dense
 -------            -------
 132.4 ms           128.7 ms   <- reconciles with the 125-129 ms benchmark

Netting out the independently measured MoE cost (1.12 ms/call x 42 = 47 ms):

  KDA linear attention   ~49 ms  (39%)   1.59 ms/layer over 34
  MoE                    ~47 ms  (38%)   1.12 ms/layer over 42
  MLA sparse attention   ~29 ms  (23%)   2.60 ms/layer over 11

ATTENTION IS 61% OF THE DECODE STEP. The expert cache, which absorbed nearly
all prior effort, is 38%. This is why six cache hypotheses returned null.

KDA is the anomaly: its recurrent state is HV*K*V = 32*128*128 fp32 = 2 MB per
layer, i.e. ~2.5 us of HBM traffic at 1.6 TB/s. 1.59 ms/layer is ~600x that, so
the cost is NOT the state update. Candidates: the W4A16 projections (in_proj is
a merged qkvbfg_a GEMM; Marlin is unavailable on ROCm, so a dequant-to-bf16
fallback would move ~100 MB/layer for a single token), or eager dispatch.

Config confirmed from the log: CompressedTensorsWNA16MoEMethod -> TRITON WNA16
backend -> TritonWNA16Experts; linear_backend='auto'; cudagraph_mode NONE.

### Lead: num_warps=1 hardcoded in the AMD KDA decode kernel
amd/ops/third_party/kda/fused_recurrent.py:473 sets num_warps=1, num_stages=3
for fused_recurrent_gated_delta_rule_packed_decode_kernel.
At our shape (B=1, HV=32/rank, K=V=128): BK=128, BV=32, NV=4,
grid=(4, 32) = 128 workgroups x 1 wavefront (64 threads) across 104 CUs.
A CUDA-shaped default (warp=32) sitting in the AMD-specific file.
One-line experiment -- but only worth running if the sub-step timing shows the
core kernel is actually a meaningful slice of the 1.59 ms.

Next measurement: patched_glm5next_kda.py (GLM53_KDA_TIMING=N) splits the KDA
forward into in_proj / small_proj / core / o_norm / o_proj.

## CORRECTION + REFINEMENT: KDA attention is 15.6 ms, not 49 ms (2026-09-14)

The "KDA ~49 ms" figure above was WRONG. It came from taking the layer-level
bucket (KDA+MoE, 2.71 ms/layer) and subtracting only the expert-cache cost
(1.12 ms/layer). That leaves the layer's OTHER work -- RMSNorms, mHC
hyper-connections, MoE router, shared expert -- silently attributed to KDA.

patched_glm5next_kda.py (GLM53_KDA_TIMING) measures the attention module
directly. Steady-state window (TP1, calls 1700->2040, differenced):

  core (conv + recurrent)  0.140 ms/call   30%
  in_proj                  0.117 ms/call   25%
  o_proj                   0.098 ms/call   21%
  o_norm                   0.062 ms/call   14%
  small_proj               0.042 ms/call    9%
  total                    0.459 ms/call
  => 34 layers x 0.459 = 15.6 ms/step

NOTE: read TP1, not TP0. TP0's o_proj shows 0.39 ms because o_proj is
RowParallelLinear with reduce_results -- it carries the TP all-reduce and
therefore the wait for the peer rank. TP1 is the clean read.

CONSEQUENCE: no W4A16 projection disaster (in_proj+o_proj = 0.215 ms/call for
~41 MB of int4 weights, which is reasonable), and no single dominant sub-step.
The num_warps=1 lead can only address the 0.140 ms core, so it is SHELVED --
not worth a 15 min relaunch for a <=0.1 ms/layer target.
patched_kda_fused_recurrent.py exists (KDA_DECODE_WARPS/STAGES/BV env knobs)
if that ever becomes worth testing.

### Revised decode budget (128.7 ms step)
  MoE experts        47.0 ms   37%   (measured, expert-cache events)
  MLA sparse attn    28.6 ms   22%   (layer bucket minus MoE)
  KDA attention      15.6 ms   12%   (measured, sub-step events)
  UNLOCATED          37.5 ms   29%

The 37.5 ms is per-layer work outside attention and experts: RMSNorms, the MoE
router (noaux_tc topk over 288), the shared expert (1 per layer, 12.6 MB int4),
and mHC. PRIME SUSPECT: hc_sinkhorn_iters=20 with hc_mult=4 -- 20 Sinkhorn
normalization iterations per layer x 45 layers ~ 900 tiny kernel launches per
token, with cudagraphs forcibly disabled (hybrid attention breaks capture).

Measuring now: patched_glm5next_model.py gained _timed_mhc_forward
(GLM53_HC_TIMING=N) splitting the mHC layer into
hc_attn_pre / attn / hc_ffn_post_pre / mlp / hc_tail.

## FALSIFIED: mHC is not the problem (2026-09-14)
GLM53_HC_TIMING window (450 calls, TP0), per mHC layer:
  mlp               2.031 ms  68%
  attn              0.703 ms  24%
  hc_ffn_post_pre   0.122 ms   4%
  hc_attn_pre       0.118 ms   4%
  hc_tail           0.009 ms
  total             2.983 ms   (x45 = 134 ms, reconciles with the step)
mHC total is 0.24 ms/layer = ~11 ms/step (8%). The hc_sinkhorn_iters=20
launch-overhead theory is WRONG.

Why it is fine: mhc dispatch is forward_hip -> aiter if supported, ELSE
tilelang, ELSE torch. Runtime flags on gfx90a (measured in-container):
  has_tilelang()                 = True
  is_aiter_found_and_supported() = False   <- docstring: "CDNA 3 or better"
  HAS_TILELANG_MHC               = True
  HAS_AITER_MHC                  = False
So AITER mHC IS excluded on CDNA2, but it lands on tilelang (compiled, fused),
not the eager-torch fallback. The CDNA-3 gate costs little here.

## VALIDATED: full decode budget, every line measured (2026-09-14)
GLM53_MOE_TIMING window (420 calls, TP0), per MoE layer:
  gate              0.096 ms
  self.experts(..)  1.798 ms
Expert-cache counters inside that call: refresh 0.866 + moe 0.338 + remap 0.016
= 1.220 ms. So 0.578 ms/call (32% of the experts call) is grouped-topk
(noaux_tc over 288) + shared expert + prepare/finalize -- never instrumented
before.

  expert gather (refresh)     36.4 ms  28%   0.866 x 42
  attention (KDA + MLA)       31.6 ms  25%   0.703 x 45
  topk + shared + finalize    24.3 ms  19%   0.578 x 42
  MoE kernel proper           14.2 ms  11%   0.338 x 42
  mHC                         10.8 ms   8%   0.240 x 45
  router gate                  4.0 ms   3%   0.096 x 42
  ----------------------------------------
  ~121 ms of the 128.7 ms step

## THE GATHER IS COPYING VRAM->VRAM (explains every null result)
refresh 0.866 ms/call moves ~52 MB (4 missed experts x 12.97 MB at ~50% hit)
=> ~61 GB/s. That is ABOVE PCIe and above the measured 26.7 GB/s host-copy
rate, so most "misses" cannot be coming from host memory.

--cpu-offload-gb 33 is per rank, against ~91 GB of weights per rank, so only
~36% of experts are host-resident. The other ~64% already live in VRAM and the
cache copies them into slots anyway: ~2.2 GB/token of VRAM->VRAM traffic that
buys nothing.

cache.py's docstring assumes otherwise ("The model's full expert tensors stay
where the offloader put them -- host RAM, mapped so the GPU can read them").
True for the Qwen/R9700 setup it was designed against; NOT true here.

THIS IS WHY 4.7x faster copies gave nothing, why more slots gave nothing, and
why "minimum slots + maximum residency" won: the copy is the waste, not its
speed. PROPOSED FIX: stage only host-resident experts; let the MoE kernel
address VRAM-resident experts in place via expert_map. Not yet implemented.

## RETRACTION of the section immediately above (2026-09-14)
"THE GATHER IS COPYING VRAM->VRAM (explains every null result)" is WRONG.
It came from dividing 52 MB by 0.866 ms to get one blended ~61 GB/s rate and
concluding the traffic could not be host-sourced. That conflates two very
different rates. Splitting the mix properly:

  host share  ~36% of 52 MB = 18.7 MB @ ~25 GB/s  = 0.75 ms
  VRAM share  ~64% of 52 MB = 33.3 MB @ ~500 GB/s = 0.07 ms
  total                                            ~0.82 ms  vs 0.866 measured

So the gather IS PCIe/host-bound, as originally believed. The VRAM->VRAM copies
cost ~0.07 ms/call (~3 ms/step) -- real but minor, NOT the explanation for
anything. Do not build the "stage only host-resident experts" redesign on this;
its upside is ~3 ms, not ~36 ms.

STILL UNEXPLAINED: why 4.7x faster copies produced no end-to-end gain when the
gather is 36 ms of a 128 ms step. Candidates: (a) the DMA A/B did not actually
accelerate the host-sourced leg, only the VRAM leg; (b) the sweep was
confounded by host paging -- this ledger already flags that sweep as
untrustworthy point-by-point. Needs a clean re-run before any conclusion.

Note the structural constraint on hiding this cost: MoE routing for layer N+1
depends on layer N's output, so the gather cannot be prefetched ahead in the
general case. Only cross-TOKEN locality is exploitable, which is exactly what
the LFU policy already does (hit rate 34-55%).

## RESOLVED: the DMA paradox -- faster copies are WORSE (2026-09-14)
Clean A/B, single variable, measured on refresh ms/call (NOT end-to-end step
time, which is what made the earlier attempt untrustworthy). Both arms
converged (identical to 3 dp across 2940/3360/3780 calls):

  kernel gather (EXPERT_CACHE_DMA=0)   refresh 0.866 (TP0) / 0.838 (TP1)
  DMA copy engine (EXPERT_CACHE_DMA=1) refresh 0.943 (TP0) / 0.908 (TP1)

DMA is ~9% SLOWER in situ despite being 4.7x faster in isolation
(27.6 vs 5.8 GB/s).

CAUSE, from cache.py's own comment at _gather_dma: "The miss list lives on the
device, so issuing per-copy from the host..." -- the path does
_h_nmiss.copy_(n_miss); _h_miss.copy_(miss); then int(self._h_nmiss[0]),
which SYNCS device->host once per layer-call, i.e. 42 syncs per token. The
sync costs more than the faster copy saves.

CONSEQUENCE: "4.7x faster copies gave nothing" was NOT a host-paging artifact.
Faster copies are actively worse. Keep EXPERT_CACHE_DMA=0 (the default).
The retraction already in docs/DMA_GATHER.md stands and is now explained.

ALSO: both isolated bandwidth numbers are useless as predictors. In situ the
kernel gather moves ~52 MB in 0.866 ms = ~60 GB/s, 10x its own 5.8 GB/s
microbenchmark. Do not reason about this path from microbenchmarks again.

## What this leaves as the real lever on the 36 ms gather
It is PCIe-bound on the ~36% host-resident share, cannot be prefetched (layer
N+1 routing depends on layer N output), and neither copy mechanism helps.
The only remaining lever is shrinking the host-resident share = requantization.

Projection from 182 GB @ ~4 bits, 91 GB/rank, ~60.8 GB usable VRAM/rank:
  3.5 bits -> 159 GB -> offload ~19 GB -> gather ~24 ms   (step ~117, 8.5 t/s)
  3.0 bits -> 137 GB -> offload  ~7 GB -> gather ~10 ms   (step ~103, 9.7 t/s)
  2.5 bits -> 114 GB -> fits entirely  -> gather  ~0 ms   (step  ~92, 10.9 t/s)
Full residency is ~38% faster than today, NOT multiples. The larger remaining
blocks are attention (31.6 ms) and topk+shared+finalize (24.3 ms).

## METHODOLOGICAL LIMIT: nested CUDA-event timing perturbs and double-counts
Adding GLM53_MK_TIMING (wrapping FusedMoEKernelModularImpl._prepare /
_maybe_apply_shared_experts / _fused_experts / _finalize) CHANGED a number it
was not supposed to touch:

  same config, expert-cache refresh ms/call
    without mk wrappers:  0.866
    with mk wrappers:     1.891 -> 1.846 -> 1.803 (still converging down)

And the nested readings are internally impossible: the expert cache hooks
fused_experts.apply, which is INSIDE _fused_experts, yet reported
refresh+moe = 1.204 ms against _fused_experts = 0.285 ms. An inner region
cannot exceed its enclosing one.

CAUSE: torch.cuda.Event windows on a busy eager stream measure elapsed GPU
TIMELINE, not the kernel's own occupancy. A window that opens after a gap
absorbs whatever was still queued. Adding Python wrappers also changes launch
pacing. So nested measurements both perturb and double-count stalls.

CONSEQUENCE -- what is trustworthy and what is not:
  TRUSTWORTHY
    - Layer-level timing: mlp 2.031 / attn 0.703 / mHC 0.24 ms per layer.
      It RECONCILES with the wall-clock step (x45 = 134 vs 125-129 ms measured),
      which is the check that matters.
    - The DMA A/B: same metric, same instrumentation, single variable changed.
    - KDA sub-steps: internally consistent and summed to their own total.
  NOT TRUSTWORTHY
    - Every sub-MoE attribution derived by mixing the expert-cache counters with
      the modular-kernel wrappers, i.e. the claimed split
      "gather 36.4 / topk+shared+finalize 24.3 / MoE kernel 14.2 ms".
      RETRACTED. The gather may be materially smaller than 36 ms.

To split the MoE module properly, use ONE instrumentation layer at a time and
re-validate the total against wall clock on every arm. Do not nest.

## bench2.sh's correctness gate is UNRELIABLE -- harness bug, not a regression
On the clean relaunch the gate read 1/3 ("4", "2+2", ""), which looks like a
correctness regression. It is not.

  bench2.sh asks for max_tokens=256 for a one-digit answer, so the reasoning
  model keeps generating past the answer; the script then truncates content to
  40 chars and pattern-matches, seeing trailing trace text instead of the answer.

Direct probes at max_tokens=48: '4', '4', '4' -- 3/3 clean.
A 64-token probe ("reply with the digit 7") returned content starting "7" with
coherent reasoning, but with trace text leaking into `content` AFTER the answer:
  '7\n"`#7. responding to user\n7. analyzing the prompt, the user wants me...'
The separate `reasoning` field was clean. So the reasoning-parser split is
messy, the MODEL is fine.

This also explains the earlier odd reading "2+2 = 3 (in base 3)" -- same cause.
FIX THE HARNESS before trusting any correctness gate: use a small max_tokens
for short-answer probes, and match on the parsed answer, not content[:40].
Do not read a low gate score as model degradation without a direct probe.

## MEASUREMENT VALIDITY: decode speed depends on expert-cache WARMTH
A fresh container reads ~10% slow and converges downward with use.

  cold (just launched, 5 runs):  137.2 133.8 135.9 146.9 135.4  (median 135.9)
  after a 752-token warmup:      138.3 132.6 134.3 130.4 126.7  (monotone down)

The last warm run, 126.67 ms/step / 7.89 tok/s, reproduces the original
125.2-129.1 ms baseline exactly. So the 142 ms reading earlier was a COLD CACHE,
not a regression, and not the inert diagnostic mounts (those are ~87 integer
branch checks per step, ~4 us).

Verified not contention: only the glm53 container held renderD128/renderD131,
and the queue was Running: 1 / Waiting: 0 throughout.

RULE: warm with several hundred tokens before benchmarking, and prefer the
LAST run of a series, not the best or the first. This also retroactively
explains part of the 118.8-144 ms spread across arms in the residency sweep --
those arms sat at different cache warmth and were never comparable point-by-point.

## VALIDATED: routing and finalize are NEGLIGIBLE; it is all the expert kernel
Disciplined retry of the split that failed before: ONE instrumentation layer
(EXPERT_CACHE_TIMING off), whole and parts in the SAME event stream, warmed
900 tokens first, and a parts-sum-vs-whole check printed every report.

  whole (_forward_impl)   1.770 ms/call
  _apply_quant_method     1.657 ms   94%
  _maybe_combine          0.0059 ms   0.3%
  _maybe_dispatch         0.0041 ms   0.2%
  parts sum = 94% of whole  <- self-consistent this time (missing 6% = gate+glue)

TWO HYPOTHESES KILLED:
 1. The "topk + shared + finalize = 24.3 ms" block DOES NOT EXIST. Routing and
    finalize together are ~10 us/call, ~0.4 ms/step.
 2. The TP all-reduce theory is dead: _maybe_combine is 6 us/call, so the
    "42 NCCL all-reduces per token over PCIe" concern was unfounded.

Essentially the whole MoE module is the expert kernel path
(cache gather + WNA16 Triton kernel + shared expert).

CAVEAT ON THE ABSOLUTES: this arm ran 139-143 ms/step vs the 126-127 ms clean
baseline, despite the warmup -- so the instrumentation itself costs ~10%
(168 extra CUDA events/step + Python wrappers). Treat the per-call absolutes as
INFLATED UPPER BOUNDS. The RATIO is safe: 1.657 vs 0.010 ms is 165x, which
~0.08 ms of overhead across four event pairs cannot manufacture.

### Where this leaves the decode step (127 ms, warm)
  expert kernel path (gather + WNA16 kernel + shared expert)   ~55-66%
  attention (KDA ~15.6 ms measured + MLA)                       ~25%
  mHC (tilelang)                                                 ~8%
  router gate + glue                                             ~3%

THE single lever worth pulling next, and it needs no quality trade:
the WNA16 Triton MoE kernel is UNTUNED for gfx90a (0 of 333 vLLM configs,
0 of 728 AITER configs). It sits inside the largest block in the step.
Autotune shape: E=144 (288/2 under EP2), N=2048, K=4096, W4A16 group-128.
Tuner is on the box at /opt/vllm/benchmarks/kernels/benchmark_moe.py.

## IN FLIGHT: WNA16 MoE kernel autotune for gfx90a (2026-09-14 ~10:07)
The one lever that needs no quality trade. Target: the WNA16 Triton MoE kernel,
which sits inside the largest block of the decode step and has ZERO tuned
configs for this arch (0/333 vLLM, 0/728 AITER).

Command (container moe-tune3, image local/vllm-mi210:rocm10-mi210.7-aiter):
  python3 benchmark_moe_glm.py --model /models/glm53-w4a16-mtp \
    --trust-remote-code --tp-size 2 --enable-expert-parallel \
    --dtype int4_w4a16 --batch-size 1 --tune --save-dir /models/moe_configs

TWO FIXES NEEDED TO GET IT RUNNING:
 1. `--group-add render` fails on this host ("Unable to find group render").
    The launcher uses `--group-add video --cap-add SYS_PTRACE` instead.
 2. benchmark_moe.py's get_model_params() did not know GLM5Next. It expects
    Mixtral-style config.num_local_experts; GLM-5.3 nests MoE params in
    text_config and spells the count n_routed_experts. Patched copy lives at
    /home/dave/glm53-bringup/benchmark_moe_glm.py (adds a Glm5NextForCausalLM /
    Glm5NextForConditionalGeneration branch). Worth upstreaming.

Status: "Start tuning over 8000 configurations...". Ray up, BenchmarkW at 184%
CPU with GPUs at 0% -> it is in the Triton JIT COMPILE phase. 8000 compiles at
~0.5-2 s each makes this a MULTI-HOUR job, not minutes.

Output will be /mnt/llm-storage/moe_configs/E=144,N=2048,device_name=...json
(E=144 because EP2 halves 288). To use it: mount into the container's
vllm/model_executor/layers/fused_moe/configs/ and A/B decode against the
126-127 ms warm baseline. REMEMBER to warm before measuring.

### Box state while the tune runs
The MODEL SERVER IS DOWN: container moe-tune3 holds both MI210s. The tuner is
disposable -- kill it and relaunch the validated server with:

  docker rm -f moe-tune3
  cd /home/dave/glm53-bringup && NAME=glm53 PORT=8145 TP=2 UTIL=0.95 \
    MAXLEN=8192 MAX_NUM_SEQS=1 OFFLOAD_GB=33 SLOTS=8 POLICY=lfu SPEC=off \
    EXPECT_NO_CACHE=1 EXTRA_VLLM_ARGS="--enable-expert-parallel --enforce-eager" \
    ./launch_glm53.sh

Validated config: 126-127 ms/step, 7.89 tok/s warm, correctness 3/3,
prompt ceiling >= 7813 tokens. User's q38fn-lru / qwen35 / litellm services were
never touched (protected list, separate devices).

A/B harness for the tuned config is staged at
/home/dave/glm53-bringup/ab_tuned_moe.sh -- it warms 900 tokens, reports the
LAST of 6 runs, and uses max_tokens=48 for correctness probes (bench2.sh's 256
lets the reasoning trace corrupt the gate).

## FALSIFIED: the autotuned MoE config is ~6% SLOWER (2026-09-14)
The tune completed (~13 min, not hours -- the 8000-config sweep prunes fast).
Output: E=144,N=2048,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json
  M=1 only: BLOCK_SIZE_M 16, BLOCK_SIZE_N 16, BLOCK_SIZE_K 64,
            GROUP_SIZE_M 1, num_warps 1, num_stages 2, waves_per_eu 0, SPLIT_K 1

A/B, both warmed, idle queue, 12 clean runs:
  untuned (vLLM runtime heuristic)   126-127 ms/step   7.89 tok/s
  autotuned M=1                      133-140 ms/step   7.16-7.49 tok/s
=> the tuned config is ~6% WORSE. Do NOT mount it.

WHY: benchmark_moe.py times the MoE kernel in ISOLATION, against freshly
allocated weights. In the real model that kernel runs immediately after the
expert cache has staged its weights into slots, and the isolated benchmark
cannot see that interaction. Same trap as the DMA gather: a microbenchmark that
does not predict in-situ behaviour. vLLM's runtime fallback heuristic was
already the better choice for this shape.

MEASUREMENT NOTE: the first readings on this arm were 175-920 ms because the
A/B harness was still running against the same port while I probed it
separately -- the exact queueing error this ledger already warns about.
Always confirm "Waiting: 0 reqs" AND that no harness is live before measuring.

CONSEQUENCE: kernel autotuning is now a CLOSED avenue on gfx90a for this shape,
alongside Marlin/FlashInfer/AITER-W4A16/graph-capture/DMA. The remaining levers
all cost something: requantization (quality), or a redesign of the gather that
does not copy VRAM-resident experts (~3 ms upside, small).

## CORRECTED: marginal cost of offload is 0.33 ms/GiB, NOT 0.81 (2026-09-14)
Three clean warm arms, same config otherwise:
  OFFLOAD_GB=30   VRAM 55.90 GiB/rank   ~129.6 ms/step   7.69 tok/s
  OFFLOAD_GB=33   VRAM 53.59 GiB/rank   ~127.0 ms/step   7.89 tok/s  (baseline)
  OFFLOAD_GB=48   VRAM 38.43 GiB/rank   ~141.0 ms/step   7.09 tok/s

30 -> 48 moves 17.47 GiB/rank = 34.9 GiB total and costs 11.4 ms
  => 0.327 ms per GiB offloaded.
The old 0.81 ms/GiB came from the residency sweep this ledger already flagged as
confounded by host paging. It was ~2.5x too high.

REQUANT BUSINESS CASE, CORRECTED:
  current offload ~56.7 GiB (model 163.9 GiB excl. MTP, 107.2 GiB resident)
  eliminating ALL offload = 56.7 x 0.327 = ~18.5 ms
  => 127 ms -> ~108 ms -> ~9.2 tok/s, i.e. +17% (NOT the +38% / 10.9 tok/s
     previously projected off the bad rate).
A 3-bit requant, which does not reach full residency, buys proportionally less.

WHY THE RATE IS LOW: the LFU expert cache absorbs most of the offload penalty --
hot experts occupy VRAM slots regardless of where their master copy lives, so
host traffic is far below the naive "offloaded fraction x 4.36 GB/token".
The cache is working; that is precisely why shrinking the model pays so little.

## Also checked and CLOSED this round
- HUMMING WNA16 MoE backend: exists and is selectable via moe_backend='humming',
  but _supports_current_device() requires platform.is_cuda() AND an external
  `humming` package. Genuinely CUDA-only, not an artificial ROCm gate.
- PCIe link: both MI210s negotiate Gen4 x16 (LnkSta 16GT/s x16 == LnkCap).
  No degraded link. Ruled out.
- MTP layer 45 (13.84 GiB, entirely BF16, unquantized in the checkpoint) is
  NOT loaded into VRAM: model.py:799 `if spec_layer is not None: continue`.
  Dead weight on disk only. No win available.
- KV cache is NOT over-provisioned waste: it is the slack. OFFLOAD_GB=28 fails
  with "0.31 GiB KV cache is needed, larger than available". Genuine headroom
  was only ~3.8 GiB/rank, and spending it (offload 30) gained nothing measurable.

## Checkpoint composition (what is actually loaded, 163.9 GiB)
  experts int4 (I32 packed)   141.75 GiB
  expert scales BF16            4.43 GiB
  attn/norm/gate BF16          12.58 GiB   <- NOT quantized
  embed/lm_head BF16            2.36 GiB   <- NOT quantized
  vision tower BF16             1.05 GiB   <- unused, we serve text-only
int8 on the 16 GiB of BF16 non-expert weights would free ~8 GiB => ~2.6 ms at
the corrected rate. Marginal. int8 on the EXPERTS would be catastrophic:
141.75 -> 283 GiB, adding ~140 GiB of offload.

## VALIDATED: the expert cache is worth ~125 ms/step (2x decode) (2026-09-14)
Same validated config, ONE variable (EXPERT_CACHE_DISABLE=1), both warmed:

  cache ON    126-130 ms/step   7.89 tok/s
  cache OFF   251.8-252.7 ms    3.96 tok/s   (5 runs, spread 0.9 ms)

The cache nearly DOUBLES decode. Reading offloaded experts directly from UVA
host memory, as vLLM does natively, costs ~125 ms/step more than staging them
into VRAM slots with LFU reuse.

This closes the question of whether the cache earns its keep: decisively yes.
It also kills my earlier speculation that the gather's VRAM->slot copying is
net waste -- whatever inefficiency remains inside it is dwarfed by what it saves.

## CORRECTION to "the expert path is ~100% PCIe-bound"
That claim was WRONG. It assumed host traffic scales linearly with the
offloaded fraction, which the LFU cache invalidates. The measured marginal
rate (0.33 ms/GiB x 56.7 GiB) puts the offload cost at only ~18.5 ms of the
127 ms step -- ~15%, not ~50%. Most of the expert path is VRAM-side work that
the cache has already optimised down from the 252 ms no-cache case.

## Standing summary of where decode time goes (127 ms, warm, trustworthy only)
  MoE module      ~66%   (cache ON; would be ~250 ms of it with cache OFF)
  attention       ~25%   (KDA 15.6 ms measured + MLA) <- LEAST EXPLORED
  mHC             ~8%
Offload accounts for ~18.5 ms of the whole step.

## ATTENTION INVESTIGATED (2026-09-14) -- 25% of the step, previously untouched

### MLA sparse layer split (GLM53_MLA_TIMING, whole-and-parts, parts=50% of whole)
  whole MLA layer   1.672 ms/call
    mla_attn        0.469   28%
    indexer         0.363   22%
    UNACCOUNTED     0.840   50%   <- projections + rmsnorm + rope glue
x11 layers => MLA ~18.4 ms/step, of which ~9.2 ms is the projection path.

NOTE: the first attempt read "parts = 438% of whole" -- impossible. Cause was my
own double-wrapping: I set a guard flag on the INSTANCE but replaced
type(idx).forward globally, so all 11 MLA layers re-wrapped the shared class
method and nested the timings 11 deep (7.2196/0.6421 = 11.2, exactly the layer
count). Wrap ONCE at class level. The parts-vs-whole check is what caught it.

### The quantization recipe deliberately skipped attention
config.quantization_config.ignore has 2303 entries:
  attention linears      1386   -> stay BF16
  shared-expert linears   504   -> stay BF16
  vision                  248
  mlp/dense               162
Only the ROUTED experts were quantized. BF16 breakdown of what is loaded:
  attention        11.31 GiB
  shared experts    1.97 GiB
  quant scales      4.43 GiB (required)
  embed/lm_head     2.36 GiB
  norms/gates       0.97 GiB
  (MTP 13.84 GiB and vision 1.05 GiB are not loaded / unused)
QUANTIZABLE = 13.28 GiB -> int8 frees 6.6 GiB, int4 frees 10.0 GiB.

### But the payoff is limited, because the two halves differ
  KDA projections  ~135 MB/rank/call, roofline 0.084 ms, measured 0.215  = 2.5x off
                   -> genuinely bandwidth-bound; halving bytes helps.
  MLA projections   ~44 MB/rank/call, roofline 0.027 ms, measured 0.840  = 30x off
                   -> OVERHEAD-bound (many small eager ops: fused_qkv_a_proj,
                      fused_q_kv_rmsnorm, rope split/cat, q_b_proj, plus an
                      fp32 torch.mm(hidden_states.float(), _wp_fp32) at
                      attention.py:337). Bit-width changes nothing here.

ESTIMATE for quantizing attention + shared experts:
  int8: ~2.2 ms (offload, at the measured 0.33 ms/GiB) + ~2 ms (KDA read traffic)
        = ~4 ms  -> 127 -> ~123 ms -> ~8.1 tok/s  (+3%)
  int4: ~7 ms    -> 127 -> ~120 ms -> ~8.3 tok/s  (+5%)
Real but small, and int4 on attention carries the most quality risk of anywhere.

### The bigger attention prize is NOT quantization
~9.2 ms/step sits in the MLA projection path at 30x off roofline, i.e. pure
eager launch overhead. The structural fix is graph capture, which is disabled
because 45 hybrid-attention layers break capture by design (already in this
ledger). Targeted fusion of the MLA pre-attention glue is the tractable version.

## CORRECTION: the MLA "0.84 ms projections" was actually o_proj (2026-09-14)
Finer inline marks in mla.py (GLM53_MLAP_TIMING) give the real per-call split.
Windowed (calls 8580->8800), BOTH ranks:

            TP0      TP1
  o_proj    0.792    0.228
  mla_attn  0.773    1.192    (span includes rope + indexer + mla_attn)
  q_proj    0.033    0.049
  fused_qkv 0.029    0.045
  rmsnorm   0.014    0.045
  TOTAL     1.645    1.583

The earlier "0.840 ms unaccounted" was NOT the projections. o_proj runs AFTER
mla_attn in the wrapper, so the previous instrumentation (which wrapped
mla_attn and indexer only) dumped it into the unaccounted bucket.
REAL projections = fused_qkv_a_proj + q_proj + rmsnorm = ~0.076 ms/call. FAST.

### o_proj is mostly INTER-RANK SKEW, not communication
o_proj is RowParallelLinear with reduce_results=True, so it carries the TP
all-reduce. TP0 spends 0.792 there while TP1 spends 0.228 -- but TP1 spends
1.192 in mla_attn vs TP0's 0.773. The per-call TOTALS match (1.645 vs 1.583),
so the ranks are equally loaded overall; the WAIT simply lands at a different
sync point on each. CUDA events on a collective absorb the peer's arrival time.

P2P is NOT the problem: torch.cuda.can_device_access_peer(0,1) and (1,0) are
both True. Link type is PCIE (no XGMI bridge), cards at 86:00.0 and c3:00.0,
both Gen4 x16. Custom all-reduce stays unavailable (gfx94/gfx95 only), so
PYNCCL over P2P is what runs.

### CONSEQUENCE for the int8-attention idea
MLA projections are only 0.076 ms/call, so quantizing them buys ~nothing.
The quantizable win is almost entirely KDA's projections (0.215 ms/call x 34).
Revised estimate for int8 on attention + shared experts:
  ~3.6 ms (KDA read traffic) + ~2.2 ms (offload, at 0.33 ms/GiB) = ~5.8 ms
  -> 127 -> ~121 ms -> ~8.3 tok/s (+5%)
And the "fuse the MLA glue" idea is now much weaker: there is no 9 ms of glue,
only ~0.8 ms/call of real projection work across 11 layers (~0.9 ms/step).

## *** BREAKTHROUGH: cudagraphs unblocked -> 9.84 tok/s (+25%) (2026-09-14) ***

  baseline (eager)     126-130 ms/step   7.89 tok/s
  cudagraph FULL       100.0-102.1 ms    9.84 tok/s   <- 6 runs, spread 2.1 ms
Correctness verified: short answers 3/3 reach 4; 991-token retrieval returns
PURPLE-7734. Capture: "Capturing CUDA graphs (FULL): 100%", 1-2 s, 0.14 GiB.

TWO THINGS WERE REQUIRED.

1. STOP PASSING --enforce-eager. The launcher already sets
   --compilation-config {"cudagraph_mode": "FULL_DECODE_ONLY"} and carries a
   hard-fail guard ("FATAL: cudagraph silently downgraded to PIECEWISE -- costs
   ~60% decode") plus TWO patches written specifically to make capture survive
   on gfx90a (the fp8->bf16 upcast for tl.dot, and the _fold_seqlen_indptr
   device-side zero_). All of that machinery was dead because every run in this
   session passed --enforce-eager, inherited from the validated config and
   never questioned. My earlier ledger line "graph capture: 45 hybrid-attention
   layers break it by design" was NOT something I tested -- it was assumed.

2. MAKE THE SPARSE INDEXER CAPTURE-SAFE.
   rocm_aiter_mla_sparse.py fp8_paged_mqa_logits_torch is a DeepGEMM UNIT-TEST
   reference ("Taken from .../tests/test_attention.py#L156") that becomes the
   PRODUCTION sparse-indexer path on gfx90a, because the AITER fp8 gate is
   correctly False on CDNA2 (no fp8 MFMA). It ran:
       seq_len = int(context_lens[i].item())   <- device->host sync per MLA layer
       pages   = block_tables[i, :num_pages]   <- data-dependent shape
       logits[i, :seq_len] = score[:seq_len]   <- data-dependent shape
   Any one of those invalidates capture (hipErrorStreamCaptureInvalidated), and
   the sync alone drained the pipeline 11x per decode step.

   FIX (patched_rocm_aiter_mla_sparse.py): take every page the block table can
   hold (block_tables.shape[1], a static bound), clamp the page ids in bounds
   (unused slots hold NULL_BLOCK_ID/stale), and mask the out-of-context tail on
   device with `pos < context_lens[i]` instead of slicing by a synced scalar.
   Same result, no sync, static shapes. Costs the max-context page count on
   short sequences -- a sub-millisecond gather at these sizes.
   NOTE: use clamp(), not clamp_() -- the in-place form mutates vLLM's block
   table. I wrote that bug and caught it before shipping.

### New standing config (supersedes the old baseline)
  TP=2 --enable-expert-parallel  (NO --enforce-eager)
  UTIL=0.95 MAXLEN=8192 MAX_NUM_SEQS=1 OFFLOAD_GB=33 SLOTS=8 POLICY=lfu SPEC=off
  => 101 ms/step, 9.84 tok/s, correctness verified.

### What this does to the other levers
The remaining budget shrinks by 26 ms, so everything measured against the old
127 ms step is now a smaller share. int8 on attention+shared experts (~5.8 ms)
is still real. The expert requant case is weaker still.

## FALSIFIED: lowering MAXLEN does not help (2026-09-14, under cudagraphs)
My capture-safe indexer rewrite processes every page block_tables can hold, so
the bound scales with MAXLEN. Halving MAXLEN should have halved that work.

  MAXLEN=8192   100.0-102.1 ms   9.84 tok/s
  MAXLEN=4096   113.3-114.8 ms   8.71-8.82 tok/s   <- WORSE by ~13 ms

Five flat runs each, both warmed, so it is real and not warmth. Counterintuitive
and unexplained -- something else about the smaller KV/block geometry costs more
than the indexer bound saves. KEEP MAXLEN=8192. Do not "optimise" it downward.

## Also: torch.compile was off the whole session too
Dropping --enforce-eager did not just enable cudagraphs. The compilation mode
went from CompilationMode.NONE to CompilationMode.VLLM_COMPILE (inductor, with
the hybrid-attention ops in splitting_ops). So the +25% is compile AND capture
together, not capture alone.

## Retest queue under cudagraphs (all prior nulls were measured in EAGER)
Launch overhead used to dominate, so kernel-level effects were masked.
 - tuned MoE config (was 6% worse in eager)   <- testing now
 - expert-cache slots / policy
 - DMA gather: now DEFINITIVELY out, its per-call device->host sync is exactly
   what stream capture forbids.

## KEY: the offload marginal rate DOUBLES under cudagraphs (2026-09-14)
Ablation, cudagraphs on, both warm:
  OFFLOAD_GB=33   100.0-102.1 ms   9.84 tok/s   (spread 2.1 ms)
  OFFLOAD_GB=48   117.2-134.8 ms   ~8.0 tok/s   (spread 18 ms, noisier)
  delta ~24 ms for 34.9 GiB  =>  0.69 ms/GiB   (eager was 0.33)

Removing launch overhead left less compute to hide PCIe behind, so the gather's
share went UP. Applied to the ~57 GiB currently offloaded:
  expert gather ~= 39 ms of the 101 ms step  (~39%) -- the dominant block.

CONSEQUENCES:
 - Every GiB of VRAM freed is now worth ~0.69 ms, 2x my earlier estimate.
 - int8 on attention + shared experts (frees 6.6 GiB) = ~4.6 ms, not ~2.2.
 - Full residency would be ~39 ms => ~62 ms/step => ~16 tok/s. That is a much
   larger prize than the +17% computed from the eager rate. (Expert requant to
   reach it is ruled out by the user; noting the number for completeness.)
 - The KV-slack retest (offload 33->30, frees 4.6 GiB) is now worth ~3 ms,
   where in eager it measured as nothing.

## PROFILING LIMITATION under cudagraphs
Neither profiler can see inside a replayed graph on this build:
 - torch.profiler (CPU+CUDA activities): 68 events, device-time sum 0.0 us.
 - rocprofv3 --kernel-trace --attach <pid>: attaches, finalizes, emits no records.
Kernels dispatched by graph replay are invisible to both. Use ABLATION
(change one knob, diff the step time) to attribute time in this configuration.

## *** NEW STANDING CONFIG: 10.42 tok/s (+32% from 7.89) (2026-09-14) ***
  TP=2 --enable-expert-parallel      <- and NO --enforce-eager
  UTIL=0.97 MAXLEN=8192 MAX_NUM_SEQS=1 OFFLOAD_GB=28 SLOTS=8 POLICY=lfu SPEC=off
  => 96.0-97.0 ms/step, 10.42 tok/s. Correctness verified (2+2 and a 991-token
     PURPLE-7734 retrieval, both clean at adequate max_tokens).

  step   tok/s   change
  127    7.89    session baseline (eager)
  101    9.84    cudagraphs + torch.compile + capture-safe indexer
   99   10.09    offload 33 -> 30 (KV slack; measured NULL in eager)
   96   10.42    UTIL 0.95 -> 0.97 and offload 30 -> 28
Model VRAM went 53.59 -> 57.64 GiB/rank across those last two steps.

## WHAT IS THE BOTTLENECK (answered from ablation)
  PCIe bandwidth  ~36 ms of 96  (37%)  <- THE bottleneck
  CPU / launch    was ~26 ms, removed by cudagraphs
  HBM             ~5 ms  (5%)          <- NOT limiting

PCIe arithmetic: 0.69 ms/GiB x ~52 GiB offloaded = ~36 ms. Per token the gather
moves ~0.77 GiB from host (4.36 GiB experts touched x ~32% host-resident x ~55%
miss). 0.77 GiB / 36 ms = ~21 GB/s = ~85% of practical Gen4 x16. Link confirmed
Gen4 x16 both cards, P2P enabled. We are near the link ceiling.

HBM arithmetic: per rank per token ~6.6 GiB attention weights + ~1.5 GiB
VRAM-resident experts = 8.1 GiB at 1.6 TB/s = ~5 ms.

This is why the MoE kernel autotune could not help (the kernel is not what waits)
and why every byte-reduction lever got more valuable once launch overhead went.

## CAUTION when correctness-probing this model
max_tokens=48 is TOO FEW: the reasoning trace can consume the budget and return
an EMPTY content field, which looks like a regression and is not. Use >=200 for
short-answer probes. (bench2.sh's 256 fails the opposite way -- it lets the
trace run past the answer and then truncates content to 40 chars.)

## *** THE GATHER COSTS 32.3 ms -- 34% OF THE STEP (2026-09-14) ***
Direct ablation, EXPERT_CACHE_NOGATHER=1, cudagraphs, both warm:
  gather ON    96.0-97.0 ms   10.42 tok/s
  gather OFF   63.6-63.9 ms   15.70 tok/s   (garbage output; timing only)
  => the gather costs 32.3 ms. Five runs spanning 0.3 ms, very precise.

Independently confirms the offload ablation (which implied ~36 ms). CEILING:
eliminating gather traffic entirely gives ~15.7 tok/s.

So the 96 ms step is:
  gather (PCIe + its VRAM-side copies)   32.3 ms   34%
  everything else                        63.7 ms   66%
and the "everything else" is where the remaining ~58 ms of compute lives
(HBM reads are only ~5 ms of it).

## REOPENED: hit rate is a lever that needs no requant
The gather's cost scales with MISSES, and the hit rate is only 34-55% at
SLOTS=8 -- with top-8 routing into 8 slots, nearly everything is evicted each
step. This ledger's earlier "doubling slots gave nothing" was measured in EAGER,
where the gather sat behind launch overhead. It is now 34% of the step.

Trade to watch: slots cost VRAM. 8 slots = 12.97 MB x 8 x 42 layers = ~4.4
GiB/rank. Doubling to 16 costs ~4.4 GiB/rank more = ~8.7 GiB total, which at
0.75 ms/GiB adds ~6.5 ms of offload. It only wins if the hit-rate gain cuts
more than that from the 32.3 ms gather.

## CLOSED (with a mechanism): the slots lever
Two attempts, both fail at the current VRAM ceiling (UTIL=0.97, OFFLOAD=28):
  SLOTS=16  -> model 61.72 GiB/rank (vs 57.64 at SLOTS=8), CUDA OOM
  SLOTS=12  -> ValueError: No available memory for the cache blocks

WHY MORE SLOTS CANNOT WIN HERE:
slots and resident weights compete for the same VRAM, and residency is strictly
more efficient per GiB --
  1 GiB resident weights = 1 GiB never fetched over PCIe. Unconditional.
  1 GiB of slots        = ~2 extra experts/layer, and only pays on a cache HIT.
So at the ceiling you want the MINIMUM legal slot count. 8 IS the minimum: the
launcher enforces SLOTS >= MAX_NUM_SEQS x TOP_K = 8 or the cache is bypassed.
This supplies the mechanism behind the old ledger note "minimum slots + maximum
residency wins", which had been recorded as an observation without a reason.

ALSO: the POLICY lever is weak at SLOTS=8. With 8 slots and exactly 8 routed
experts per step, every slot is needed every step -- there is nothing for
LFU vs LRU to decide. The ~45% hit rate is just whatever overlap consecutive
tokens happen to have. Do not bother sweeping POLICY/DECAY at S=8.

NOTE: hit-rate stats CANNOT be read under cudagraphs -- _report_stats lives in
the Python apply wrapper, which never runs during graph replay (same blind spot
as the profilers). Measure hit rate in eager; it transfers, being a property of
routing + policy.

### So the gather (32.3 ms) is at its floor for this VRAM budget
Its two terms are now both pinned:
  bytes per miss  -> fixed by quantization (experts ruled out by the user)
  miss count      -> slots capped by VRAM, policy inert at S=8
The only way left to shrink it is to free VRAM so MORE weights stay resident.

## AITER int8 IS AVAILABLE ON gfx90a (2026-09-14) -- different gate
Verified at runtime in-container:
  is_rocm: True
  compute_capability: DeviceCapability(major=9, minor=0)   -> 90
  aiter linear enabled: True
  AiterInt8ScaledMMLinearKernel.is_supported(): (True, None)

AiterInt8ScaledMMLinearKernel gates on `compute_capability >= 90`, NOT on
is_aiter_found_and_supported() (which demands "CDNA 3 or better" and is what
excludes gfx90a from AITER's MoE and mHC paths). So the W8A8 int8 scaled-MM
path IS open on CDNA2 even though those others are not. Do not assume "AITER is
gated off on gfx90a" as a blanket statement -- it is per-kernel.

## Kernel options for quantizing the 13.28 GiB of BF16 weights
_POSSIBLE_INT8_KERNELS[ROCM] = [AiterInt8ScaledMM, TritonInt8ScaledMM]   (W8A8)
_POSSIBLE_KERNELS[ROCM]      = [RDNA3W4A16, RDNAHybridW4A16, TritonW4A16,
                                Conch, Exllama]                          (weight-only)
  - TritonW4A16 gates on on_gfx90a() -- written for this chip, but 4-bit ONLY
    (TRITON_W4A16_SUPPORTED_QUANT_TYPES = [uint4b8]).
  - Conch supports uint8b128 (int8 weight-only) and IS installed, but sits
    BELOW the 4-bit kernels in priority and is a generic Triton fallback.
  - exllamav2 is NOT installed.

  W8A8 int8   -> AITER gfx90a-native, frees 6.6 GiB, AND uses int8 MFMA
                 (CDNA2 has int8 matrix cores at ~2x fp16 rate). BEST OPTION.
  int4 w-only -> frees 10.0 GiB, gfx90a Triton kernel, but bf16 activations and
                 much more aggressive on attention quality.
  int8 w-only -> frees 6.6 GiB, Conch fallback. Dominated by W8A8.

CAVEAT: W8A8 quantizes ACTIVATIONS, so it needs calibration data and carries
more quality risk than weight-only. int8 activations are well-behaved, but this
is not free. The original recipe deliberately left these 1890 linears alone.

## BLOCKED (not abandoned): int8 attention requant (2026-09-14)
The conversion itself WORKED. The checkpoint fails to LOAD, for a structural
reason in how compressed-tensors resolves fused modules in this model.

DONE AND REUSABLE:
  /home/dave/glm53-bringup/requant_int8.py  -- converter, phased by --targets
  /mnt/llm-storage/glm53-w4a8-int8          -- 173G (vs 182G), 394 attention
     tensors converted 11.26 GiB BF16 -> int8, saving 5.62 GiB. config.json has
     a group_1 with the W8A8 dynamic-token scheme. Original checkpoint untouched.
  No calibration was needed: weights per-channel symmetric int8 computed
  directly; activations are dynamic per-token at runtime.

THE BLOCKER:
  KeyError: 'layers.0.self_attn.in_proj_qkvbfg_a.weight_scale'
  GLM5Next declares NO packed_modules_mapping (grep found none in
  models/glm5next/nvidia/*.py). in_proj_qkvbfg_a is a
  _Glm5NextMergedColumnParallelLinear fusing SIX projections
  (q,k,v,b,f_a,g_a) with replicated_shard_ids=(4,5). The checkpoint stores the
  six components separately; the loader maps q_proj.weight_scale onto the fused
  module's weight_scale, finds the module unquantized, and throws.

  find_matched_target tries, in order:
    1. _find_first_match(layer_name, targets)
    2. _match_fused_layer(layer_name, targets, fused_mapping)   <- dead, no mapping
    3. _find_first_match(module.__class__.__name__, targets, check_contains=True)
  Step 3 with check_contains is why group_0's target "Linear" matches every
  *ParallelLinear class by substring. Explicit per-module targets failed (step 2
  is dead); a `re:.*\.self_attn\..*` target also failed, so the module is ending
  up unquantized -- most likely group ordering or should_ignore_layer's handling
  of a fused layer whose components are partially still in `ignore`.

TO FINISH THIS someone needs to determine, in vLLM's compressed_tensors.py,
how get_scheme orders config_groups and what should_ignore_layer does for a
fused module with no packed_modules_mapping. Then either fix the targets/ignore
lists, or add a packed_modules_mapping for GLM5Next upstream.

CHEAPER ALTERNATIVE: convert only NON-FUSED linears (o_proj, f_b_proj, g_b_proj,
vision, shared experts, dense MLP) and leave the fused in_proj/qkv_a alone. Less
saving, no fused-module problem. requant_int8.py already supports this via
--targets; it would need a name filter to exclude the fused components.

## int8 attention: THREE failed load attempts, same KeyError -- STOPPED
  attempt 1: explicit per-module targets  -> KeyError in_proj_qkvbfg_a.weight_scale
  attempt 2: regex target re:.*\.self_attn\..*  -> same
  attempt 3: int8 group reordered FIRST in config_groups -> same

OFFLINE SIMULATION of the matcher (worth repeating before any further attempt):
  should_ignore_layer(name, ignore, fused_mapping={})  -> False   (not ignored)
  find_matched_target(group_0 ["Linear"])              -> "Linear"
  find_matched_target(group_1 [regex])                 -> regex
  So BOTH groups match and the layer is NOT ignored -- yet at load it has no
  weight_scale parameter, i.e. it was built UNQUANTIZED. The reason is still
  unidentified; it is NOT simply target matching or group ordering.

  IMPORTANT: adding a packed_modules_mapping would make this WORSE, not better.
  With fused_mapping supplied, should_ignore_layer(name) flips to True (the
  fused layer inherits its components' ignore status), so the layer would be
  explicitly skipped. Do not "fix" it that way.

NEXT DIAGNOSTIC if resumed: instrument CompressedTensorsConfig.get_quant_method
to log, for prefix ".*in_proj_qkvbfg_a", what it returns and which branch it
takes. That is the only remaining unknown. One patch + one relaunch.

NOT WORTH IT AS A FALLBACK: converting only the non-fused attention linears
(o_proj, f_b_proj, g_b_proj) saves ~1.5-2 GiB = ~1.2 ms = ~1%. The conversion
cycle costs more than the gain.

ARTIFACTS KEPT (reusable, original checkpoint untouched):
  /home/dave/glm53-bringup/requant_int8.py
  /mnt/llm-storage/glm53-w4a8-int8   (173G, 394 tensors converted, config has
      the int8 group first, ordering already applied)

## FALSIFIED: gather-kernel launch geometry (2026-09-14, under cudagraphs)
  baseline  CHUNKS=16  LANES=64    96.0-97.0 ms   10.42 tok/s
  wide grid CHUNKS=128 LANES=64    96.8-97.4 ms   10.31 tok/s
  wide lane CHUNKS=16  LANES=256   ~97.0 ms       10.31 tok/s
8x the grid and 4x the lane width both change NOTHING. The gather is not
parallelism-limited. Retested under cudagraphs specifically because the earlier
eager-mode test ("chunks 16->256 gave 3%") was taken while launch overhead
masked the gather; it is now 34% of the step and still does not respond.

## CORRECTION: we are at ~50% of PCIe, not ~85%
My earlier "~85% of practical Gen4 x16" wrongly pooled both links. Each GPU has
its OWN x16 link and each EP rank fetches only its own experts, so the correct
denominator is PER RANK:
  ~4 of 8 experts per rank per layer x 42 layers = 168 expert-uses
  x ~55% miss                                    = ~92 misses
  x ~34% of that rank's experts host-resident    = ~31 host fetches
  x 12.97 MB                                     = ~402 MB / rank / token
  402 MB / 32.3 ms                               = ~12.4 GB/s per rank
against ~25 GB/s practical => ~50% link utilisation, i.e. ~2x headroom exists.

So the gather is MECHANISM-limited, not link-limited. It is a KERNEL reading
pinned host memory over UVA; kernel-initiated PCIe reads issue at wavefront
granularity with few outstanding requests and do not reach link bandwidth no
matter how the grid is shaped. This matches the isolated numbers already in this
ledger: kernel 5.8 GB/s vs DMA 27.6 GB/s.

THE ONLY FIX WOULD BE A DMA-BASED TRANSFER, which cudagraphs forbid: the DMA
path reads the device-side miss list back to host once per layer-call, and a
device->host sync invalidates stream capture. Choosing DMA means giving up the
+25% that cudagraphs bought -- a bad trade (DMA was also 9% slower end-to-end
even in eager, for the same sync reason).

=> The gather's ~32 ms is structural under this design:
     bytes    pinned by quantization (expert requant ruled out by the user)
     misses   pinned by the VRAM ceiling (SLOTS=12 and 16 both OOM)
     rate     pinned by the UVA read path (geometry does not help, DMA is
              incompatible with capture)
A fix would need a gather that saturates PCIe WITHOUT a per-call host sync --
e.g. a device-side-enqueued copy, or persistent buffers with the miss list
consumed on-device. That is a real engineering project, not a knob.

## STILL WORTH AUDITING: the per-rank hit-rate asymmetry
TP0 34.2% vs TP1 54.8% with near-balanced routed-expert counts (38 vs 42).
Same policy, same slots, 21-point gap. Expert placement is "linear" (0-143 ->
rank 0, 144-287 -> rank 1), so if routing favours certain index ranges one rank
gets worse locality -- and the SLOWER rank gates every step. Fixing the
asymmetry would cut misses with no extra VRAM. Needs eager mode to read the
stats (they are invisible under graph replay).

## ROUTING AUDIT: the misses are COMPULSORY (2026-09-14) -- avenue CLOSED
GLM53_ROUTE_AUDIT hooks RoutedExperts.forward_modular (eager only; it never runs
under graph replay). 56000 calls, both ranks:

  consecutive-step overlap   26.4%
  distinct experts           288 (ids 0..287)  -- FULL coverage
  top-10% of experts take    14.8% of routes   (uniform would be 10%)
  measured cache hit rate    27.9% / 36.3%

THREE CONCLUSIONS:
1. BOTH RANKS SEE IDENTICAL ROUTING. topk_ids are GLOBAL (0..287) and the two
   ranks report identical statistics -- routing is deterministic from the same
   input. So the earlier TP0-34% / TP1-55% gap is NOT a routing or placement
   effect. The "linear expert placement causes rank skew" hypothesis is DEAD.
2. THE HIT RATE IS THE OVERLAP. Only ~26% of routed experts repeat between
   consecutive tokens, and the hit rate tracks that. The cache is retaining
   essentially everything it can; it is not underperforming.
3. THERE ARE NO HOT EXPERTS. 14.8% vs 10% uniform is mild skew, and all 288
   experts are used. Nothing for a smarter policy or placement to exploit.

STRUCTURAL CONSEQUENCE: with near-uniform top-8 routing over 288 experts, the
working set IS all 288 experts. A high hit rate would need ~288 slots/layer =
288 x 12.97 MB = ~3.7 GB PER LAYER. Impossible. This is a STREAMING problem,
not a caching problem. No audit, eviction policy, or placement change can
reduce the miss count.

=> Combined with the geometry result above, the gather's ~32 ms is now closed on
   all three terms:
     bytes  -- quantization (expert requant ruled out by the user)
     misses -- COMPULSORY, proven by the routing audit
     rate   -- UVA mechanism, not geometry; DMA incompatible with capture
   Further gains must come from moving fewer bytes (quantization) or from a
   gather that saturates PCIe without a per-call host sync.

## ROOT CAUSE FOUND: attention is unquantizable BY MODEL DESIGN (2026-09-14)
After four failed load attempts, a get_quant_method trace fired ZERO times for
any attention module -- because that function is never reached. GLM5Next
deliberately suppresses quantization for these submodules:

  kda.py:161-167
      # KDA projections remain BF16 because fp8 checkpoints omit their scales.
      saved_quant_config = vllm_config.quant_config
      try:
          vllm_config.quant_config = None      <-- suppressed for KDA attention
          super().__init__(config, vllm_config, prefix)
      finally:
          vllm_config.quant_config = saved_quant_config

  model.py:325   quant_config=None,  # MLA projections are BF16 in checkpoint
  model.py:1049  quant_config=None,  (vision tower; comment warns that
                 inheriting the global quant_config yields NaN image features)

This is why EVERY config-level fix failed identically (explicit targets, regex
targets, group reordering): the config is never consulted for these layers.
No config change can ever work.

TO ACTUALLY DO IT one must patch all three sites to thread the real
quant_config through, THEN verify per-channel int8 scales shard correctly across
the SIX fused components of in_proj_qkvbfg_a -- two of which are REPLICATED
(replicated_shard_ids=(4,5)). Hours of work, real chance of failure, for +6.5%.
RECOMMENDATION: do not. Prefer MTP or an expert requant, where the model
already expects quantized weights and none of this applies.

ARTIFACTS KEPT: requant_int8.py (works, phased by --targets) and
/mnt/llm-storage/glm53-w4a8-int8 (173G, attention converted). Both reusable if
the model-code patch is ever done. Original checkpoint untouched.

## CONFIG SPACE EXHAUSTED (2026-09-15)
Final sweeps, all under cudagraphs:
  VLLM_ROCM_USE_AITER_MOE=1  -> INERT. Still "Using 'TRITON' WNA16 MoE backend".
      The WNA16 MoE oracle has NO AITER entry at all (MARLIN, BATCHED_MARLIN,
      HUMMING, FLASHINFER_TRTLLM, TRITON, XPU, CPU, EMULATION). AITER's MoE
      covers fp8/int8 schemes, not int4 weight-only.
  AITER tuned GEMM tables have ZERO gfx90a rows:
      bf16_tuned_gemm.csv  232 rows: gfx1250, gfx950
      a8w8_tuned_gemm.csv  579 rows: gfx942, gfx950
      (matches tuned_fmoe.csv 0/617 and vLLM fused_moe configs 0/333)
      AITER linear IS enabled, but runs untuned heuristics on this arch.
  VRAM ceiling reached: at UTIL=0.97 OFFLOAD=28 only 0.91 GiB KV is available
      against 0.31 needed, i.e. ~0.6 GiB/rank slack = <1% if spent.

EVERY quality-neutral lever is now closed by measurement:
  cudagraphs+compile        DONE (+25%)
  offload / UTIL tuning     DONE (+7%)
  MoE kernel autotune       worse in both eager and cudagraph modes
  AITER MoE                 inert (no WNA16 support)
  AITER GEMM tuning         no gfx90a configs exist
  gather launch geometry    flat (8x grid, 4x lanes)
  DMA gather                slower; incompatible with graph capture
  cache slots / policy      VRAM-capped; misses proven COMPULSORY
  expert placement          both ranks see identical routing
  int8 attention            blocked by deliberate model-level quant suppression
  MAXLEN reduction          worse

REMAINING (both need a user decision):
  MTP / spec decode  -- amortises the 32.3 ms per-STEP streaming cost across
                        multiple tokens per step. Targets the one term nothing
                        else touches: steps per token. Weights already in the
                        checkpoint; run_nocache.sh has the invocation.
  Expert requant 3.0-3.5 bit -- experts are 86% of the model and ARE quantizable
                        by design. ~+20% at 3.5-bit, ~+43% at 3.0-bit, ~15.7
                        tok/s at full residency (measured no-gather floor).
                        User vetoed 2.5-bit.

## TP INVESTIGATION (2026-09-15): collective LATENCY is fine; the cost is IMBALANCE
User hypothesis: "something is missing -- expert grouping to cards, TP not
working correctly, or latency somewhere." Measured:

RCCL all-reduce, 2x MI210 over PCIe (no XGMI), torchrun, 400 iters:
    8 KB   41.6 us/op     <- the decode-time o_proj vector (hidden=4096 bf16)
   32 KB   46.6 us/op
  256 KB   52.9 us/op
    4 MB  248.2 us/op   (16.9 GB/s)
~45-90 collectives/step x 42 us = only 2-4 ms. Raw communication is NOT the
problem, and custom all-reduce (gated to gfx94/95) could recover at most
~2-3 ms of it.

WHAT THE ~20 ms OF MEASURED WAIT ACTUALLY IS: rank imbalance. Under TP the
step is the SLOWER rank's time. The two ranks' expert-cache hit rates differ
(34% vs 55% measured earlier): rank 0 misses ~60% more, does ~60% more gather,
and is the critical path every step while rank 1 idles at the collective.

WHY THE EARLIER ROUTING AUDIT MISSED IT: it measured GLOBAL consecutive-step
overlap (26.4%, identical on both ranks). But each rank's cache holds only ITS
OWN experts. Placement is "linear" (0-143 -> rank 0, 144-287 -> rank 1), so if
temporal locality is uneven across index ranges, one rank inherits worse reuse
even though the global routing stream is identical. The audit needed to split
overlap by owned subset; it did not.

THE TEST: --expert-placement-strategy round_robin (even ids -> rank 0, odd ->
rank 1) should spread locality evenly across ranks. This is exactly the user's
"grouping experts to certain cards" intuition. Running now.

## FALSIFIED: round_robin expert placement (2026-09-15)
  linear       96-97 ms   10.42 tok/s   (standing config)
  round_robin  107-109 ms  9.2 tok/s    (6 flat runs after 800-token warmup)
~11% WORSE. Placement DOES affect the cache (so the mechanism is real), but
linear was already the better choice: nearby expert ids evidently share
temporal locality, and linear keeps that within one rank's cache while
round_robin splits it across both. KEEP linear.

The per-rank hit-rate asymmetry is also NOT a stable structural effect: it
read 34%/55% in one run and 36%/28% (reversed) in another. It is dynamic --
LFU state evolving differently -- not a rank-0 penalty to engineer away.

## THE REAL STRUCTURAL GAP: batch-1 OCCUPANCY (hypothesis, testing)
no-gather floor          63.7 ms
HBM roofline for the bytes touched  ~6 ms
collectives (measured)   ~4 ms
=> ~54 ms unexplained. NOT sync (42 us/all-reduce), NOT placement (linear is
best). At batch 1 every kernel is a matvec launching a handful of workgroups
on a 104-CU GPU, so each reads its weights at a small fraction of peak
bandwidth. Decode is LATENCY-bound (GPU mostly idle per kernel), not
bandwidth-bound. This is the standard single-stream decode problem and it is
what MTP addresses on the compute side too (M=2-4 rows per step).

TEST: MAX_NUM_SEQS=2 (needs SLOTS=16, hence OFFLOAD=33 to fit). Compare
ms/step for 1 vs 2 concurrent streams ON THE SAME SERVER. If batch 2 costs
~the same per step as batch 1, the capacity was idle -- and that number is a
direct estimate of MTP's compute-side upside.

## USER CLARIFICATION (2026-09-15): "batch processing is fine as long as it's
## a single prompt stream" -- that IS speculative decoding. MTP is now in scope.

## MTP CONFIG, from the launcher (not guessed)
  SPEC=${SPEC:-on}     <- launcher DEFAULTS to on. Every run this session forced
                          SPEC=off. The original designer intended MTP.
  SPEC_N=${SPEC_N:-1}  <- model ships exactly ONE MTP draft layer; SPEC_N>1 is
                          unverified (launcher comment line 50).
  SPEC_METHOD          <- glm5_next_mtp, written to speculative-config.json and
                          passed as --speculative-config.
  cache-bypass check   <- MAX_NUM_SEQS x (1+SPEC_N) x TOP_K = 1 x 2 x 8 = 16,
                          so SLOTS MUST BE >= 16 or the launcher hard-fails
                          (ALLOW_CACHE_BYPASS=1 would run the 252 ms no-cache
                          path -- never do that).

VRAM COST: SLOTS 8->16 (+4.4 GiB/rank, needed OFFLOAD=33 in the batch-2 test)
plus the MTP layer itself (~13.84 GiB BF16 total, ~6.9 GiB/rank under TP=2).
Expect OFFLOAD ~40. That is ~24 GiB more offloaded = ~+17 ms of gather that
acceptance must earn back.

PROJECTION (1 spec token, tokens/step = 1 + alpha, step ~96+17 = ~113 ms):
  alpha 0.4 -> 1.4 tok/step -> ~81 ms/token -> ~12.4 tok/s
  alpha 0.7 -> 1.7 tok/step -> ~66 ms/token -> ~15.0 tok/s
Net positive unless acceptance is very poor. MTP helps BOTH measured problems:
it amortises the per-STEP gather AND raises kernel occupancy (M=2 rows).

PLAN: MODEL_DIR default, SPEC=on SPEC_N=1 SLOTS=16 OFFLOAD=40 UTIL=0.97,
cudagraphs on. Read the acceptance rate from vLLM's spec-decode metrics, and
measure tok/s the same warm way. Compare to 10.42.

## MTP launch checklist (verified, not assumed)
  method name  "glm5_next_mtp" is valid: config/speculative.py:64 and :1028
  acceptance   vllm/v1/spec_decode/metrics.py:128 logs
               "Avg Draft acceptance rate: %.1f%%" -- grep the server log
               for "Draft acceptance" after a warm run. Underlying counters:
               num_accepted_tokens, num_draft_tokens, accepted_tokens_per_pos.
  MTP weights  layer 45 lives in model-mtp-00001.safetensors inside the SAME
               checkpoint dir (MODEL_DIR default). The main model skips it
               (model.py:799); Glm5NextMTP loads it when SPEC=on. No copy needed.
  launch       MODEL_DIR default, SPEC=on SPEC_N=1 SLOTS=16 OFFLOAD=40
               UTIL=0.97 MAXLEN=8192 MAX_NUM_SEQS=1, EXTRA_VLLM_ARGS=
               "--enable-expert-parallel" (NO --enforce-eager).
               If OFFLOAD=40 OOMs, step to 42/44; if it leaves slack, step down.
  measure      warm 800 tokens; 6 runs, take the last; compare to 10.42 tok/s.
               ALSO read "Draft acceptance" -- a low alpha with a small speedup
               means the +17 ms of added gather ate the gain, and SPEC_N=1 is
               the only width the model ships.
Blocked only on the GPUs: the batch-2 occupancy test (glm53-b2) holds them
until it finishes. Its result predicts MTP's compute-side upside.

## RETRACTED: "round_robin is 11% worse" -- the flag never applied
The glm53-rr server log reads "Expert placement strategy: linear". The
--expert-placement-strategy round_robin arg did not take effect, so that run
was LINEAR placement at 107-109 ms. Round-robin is UNTESTED, not falsified.

Which exposes something worse: an identical-config run sat 11 ms slower than
the standing 96-97 ms. That is outside noise. NEW HYPOTHESIS: NUMA placement
of the PINNED HOST MEMORY holding the offloaded experts. GPUs are at 86:00.0
and c3:00.0 (probably different sockets). If a rank's expert buffers land on
the far node, every gather crosses the inter-socket link. This would explain
run-to-run variance AND the per-rank hit-rate/latency asymmetry that FLIPS
between runs (whichever rank got unlucky placement). Checking topology.

## USER DIRECTIVE (2026-09-15): NO MTP yet -- raw performance only. Dropped.

## NUMA ruled out
lscpu: 1 NUMA node (node0 = CPUs 0-47). Both GPUs report numa_node=-1.
Single-socket box; there is no inter-socket link for the gather to cross.
The pinned-host-memory NUMA hypothesis is DEAD. The 107-109 vs 96-97 ms
same-config discrepancy remains UNEXPLAINED -> label it run-to-run variance
across container launches until re-measured. IMPORTANT: this means the
standing 96-97 ms may itself be one sample of a wider distribution. Re-measure
the baseline across >=2 fresh launches before trusting any <5% delta.

## Placement flag did not apply -- arg name IS correct (arg_utils.py:1209)
So the launcher is overriding or dropping it. Investigating.

## CLOSED: expert placement -- round_robin is structurally unavailable
expert_map_manager.py:123-136: round_robin is supported only when
  num_expert_group > 1  AND  num_redundant_experts == 0  AND  not enable_eplb
GLM-5.3 has n_group = 1, so vLLM silently falls back to linear (the launcher
passed the flag correctly; vLLM rejected it). Round-robin is designed for
grouped-expert models (DeepSeek-style). LINEAR IS THE ONLY PLACEMENT THIS MODEL
CAN HAVE. The "grouping experts to certain cards" lever does not exist here.

Remaining open question from this thread: the 107-109 ms same-config run vs
the 96-97 ms standing config. NUMA ruled out, placement ruled out. Must be
re-measured across fresh launches -- if the true baseline is ~100 ms, several
sub-5% deltas in this ledger are within launch-to-launch variance.

## OCCUPANCY TEST: the "11x idle GPU" hypothesis is WRONG (2026-09-15)
MAX_NUM_SEQS=2, SLOTS=16, OFFLOAD=33, cudagraphs, same server for both arms:
  batch 1 (one stream)        ~105 ms/step   9.5 tok/s   (2 clean runs;
                              161/153 ms outliers = launch/run variance)
  batch 2 (two concurrent)    ~166 ms/step  11.9 tok/s aggregate
Doubling rows per step cost +58%, NOT ~0%. If kernels were mostly idle at
batch 1, batch 2 would have been nearly free. It was not. The compute portion
IS reasonably occupied; the +61 ms is the GATHER scaling -- a second token
touches its own ~8 experts per layer, so expert streaming roughly doubles.

STRUCTURAL CONCLUSION: expert streaming scales PER TOKEN, not per step.
Any scheme that adds rows to a step (batching, spec decode) pays for the extra
experts those rows touch. This KILLS the "MTP amortises the gather" story:
at alpha=0.7, ~1.7 tokens/step for ~1.5x the time is ~+13%, not the 1.5-1.8x
I projected. The user's decision to skip MTP is vindicated by measurement.

For a SINGLE stream the per-token gather is therefore the floor, and the
compute side is already reasonably efficient at batch 1. Nothing in this
thread -- TP latency (42 us), placement (structurally linear), NUMA (single
node), occupancy (busy) -- was "the missing thing". The missing thing is that
expert streaming is inherent to running a 164 GiB model in 128 GiB of VRAM.

VARIANCE: 161/153 ms outliers here, 107-109 ms in an earlier same-config
launch, vs 96-97 standing. Launch-to-launch variance is real and sizeable.
Re-measuring the standing config on a fresh launch to get its true spread.

## VARIANCE: the standing 96-97 ms was a FAVORABLE launch (2026-09-15)
Fresh launch of the identical standing config, 900-token warm, 16 runs:
  runs 1-8:   110.7 109.2 108.5 108.2 106.5 102.0 102.1 100.7  (descending)
  runs 9-16:  101.5 102.0 103.5 109.1 105.2 105.5 109.8 109.1  (drifts BACK UP)
Never reached 96-97. Steady state across launches is ~100-110 ms, ~9.1-10.4
tok/s. The +32% headline carries ~+/-7% launch-to-launch variance, so any
delta under ~5% elsewhere in this ledger should be read as "within variance",
not as a result. This includes: offload 33->30 (+2.5%), UTIL/offload 28 (+3%),
and the gather-geometry and AITER_MOE "nulls" (which at least were flat).
The two large results stand: cudagraphs+compile (+25%) and the 2x cache effect.

The UPWARD drift after run 10 is not warmth (warmth only descends) and not
LFU decay (policy is provably inert at S=8). Candidates: thermal throttling,
or a neighbour service (q38fn-lru / qwen35) stealing CPU/PCIe. Checking.

## *** THE "LATENCY SOMEWHERE": HOST CONTENTION FROM q38fn-lru (2026-09-15) ***
At the moment of the drifting-upward runs:
  q38fn-lru   407% CPU   138 GiB RAM     <- user's production Qwen service, ACTIVE
  glm53         4.5% CPU  114.5 GiB RAM
  load avg    6.17, 10.59, 13.80           <- was much busier minutes earlier
  GPU temps   52-64 C on both cards        <- NO thermal throttling
GLM streams ~48 GiB of experts from PINNED HOST DRAM over PCIe. A neighbour
working 138 GiB of host RAM at 4+ cores competes for DRAM bandwidth and for
the cores the vLLM scheduler/API server need. Single NUMA node, so there is
nowhere to isolate DRAM bandwidth -- but CPU contention IS separable.

This explains, in one mechanism:
  - launch-to-launch variance (depends on whether q38fn-lru is busy)
  - mid-run UPWARD drift (q38fn-lru took a request)
  - the 96-97 ms "favorable launch" (a quiet period on the box)
  - the per-rank asymmetry that FLIPS between runs
  - NOT TP latency, NOT placement, NOT NUMA, NOT occupancy -- all correctly
    ruled out above; the cause was outside the GLM container entirely.

The launcher has NO cpu pinning (no taskset/cpuset/numactl). Lever:
--cpuset-cpus on the GLM container to keep it off q38fn-lru's cores.
Cannot fix DRAM-bandwidth sharing on one NUMA node; can only measure it by
benchmarking during quiet vs busy periods of the neighbour.
  Follow-up read: q38fn-lru at 575% CPU, load 10.43 -- actively busy now.
  Neither container is pinned (CpusetCpus=[] on both), so they collide freely
  across all 48 cores. Benchmarking GLM concurrently to correlate.

## CONFIRMED BY CORRELATION: neighbour contention costs ~8-10% (2026-09-15)
Same server, same warm state, per-run q38fn-lru CPU sampled alongside:
  103.2-107.4 ms/step  at q38fn 399-510% CPU, load 14.3
  96-97 ms/step        during the earlier quiet period
=> ~8-10 ms of decode time is host contention from the production neighbour.
This is the "latency somewhere". It is NOT in GLM, TP, placement, NUMA, or
the GPUs (temps 52-64 C). Every "favourable launch" was a quiet neighbour.

NEXT: which resource? Pin GLM to cores 40-47 (--cpuset-cpus) and re-measure
while q38fn is busy. Recovery toward 96 => CPU cores (fixable). No recovery
=> DRAM bandwidth (single NUMA node; not fixable by pinning). q38fn-lru is
the user's production service -- do NOT pin or touch it without permission.

## Pinning test in flight: GLM on --cpuset-cpus 40-47 (2026-09-15)
Caveat checked: glm53-pin has 8 procs / 43 threads, but only the two vLLM
processes are heavy (22 and 15 threads); the rest idle at steady state.
8 cores is sufficient for decode. (AITER JIT compile is slower on 8 cores --
startup only, does not touch the benchmark.) So: a SLOWER pinned result is a
real signal, not a core-starvation artifact.
Read the result as:
  <= ~99 ms with q38fn > 400%  => contention is CPU cores -> pinning is the fix
  >= ~103 ms with q38fn > 400% => contention is DRAM bandwidth -> not fixable
                                  on one NUMA node; only scheduling around it

## Pinning approach revised: pin at RUNTIME, not container level (2026-09-15)
--cpuset-cpus throttles the AITER JIT compile at startup along with everything
else (ninja -j24 -> 8 cores). Fix: launch unpinned, then after readiness
  for p in $(docker top <name> -eo pid | tail -n +2); do taskset -acp 40-47 $p; done
Compile gets the whole box; decode gets isolation. If pinning proves out, the
launcher should do this itself post-readiness (PIN_CPUS var), not via cpuset.

## ROOT CAUSE: AITER JIT recompiles every launch (~15 min) -- cache never worked
/mnt/llm-storage/cache/aiter/ is mounted as /cache/aiter and the container env
sets AITER_ROOT_DIR=/cache/aiter -- but it is EMPTY (0 bytes, only an empty
build/ from Sep 13). aiter/jit/core.py:85 IGNORES the env var:
    AITER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
and uses it only for configs/*.csv paths, not the JIT output. The compiled
.so files land in aiter/jit/ inside the container's WRITABLE LAYER and are
destroyed with it. A fresh `docker run --rm` of the image shows no .so and no
build/ dir. Every launch this session paid the full rebuild.

A bind mount is the wrong fix: the .so files sit BESIDE __init__.py/core.py in
aiter/jit/, so mounting that dir would shadow the package.

THE FIX: docker commit a fully-warmed container to a new tag, e.g.
    docker commit glm53-pin2 local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm
then IMAGE=that tag in the launcher. AITER reuses an existing .so at import
(rank 1 already does this via "baton release" when rank 0 builds first), so
the committed image never compiles. Bind mounts (patches, /models, the cache
repo) are NOT captured by commit -- only the writable layer -- which is what
we want. Disk: ~149 GB free on /; the layer is a few GB.
TIMING: commit only after the benchmark finishes -- docker commit PAUSES the
container and would corrupt an in-flight timing run.

## USER DIRECTIVE (2026-09-15): "I wouldn't pin them." -- CPU pinning DROPPED.
The runtime-taskset chain was stopped before it pinned anything. The neighbour
contention finding STANDS (~8-10% when q38fn-lru is busy); the user chooses not
to partition the box. Consequence for all future measurement: expect ~+/-8%
launch/run variance from the neighbour, sample q38fn CPU alongside every
benchmark, and compare configs only within quiet windows or with matched load.
The JIT-warm image (docker commit) is unaffected and proceeds.

## Fresh launch #3 (unpinned), prefill, and the JIT-warm image (2026-09-15)
DECODE with q38fn-lru CPU sampled per run:
   97.9 ms @ q38fn 243%   |  104-107 ms @ q38fn 400-493%
Correlation holds a third time. STANDING ESTIMATE: ~96-98 ms (10.2-10.4 tok/s)
when the neighbour is quiet, ~104-107 ms (9.3-9.6) when busy. Report both.

PREFILL under cudagraphs (FULL_DECODE_ONLY, so prefill is not captured):
   572 tok  97.8 tok/s | 1118 tok 246 | 2226 tok 300 | 4026 tok 670
Same as eager within noise. No regression; page-size fix intact to 4026+.

JIT-WARM IMAGE: docker commit glm53-pin2 ->
   local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm   (93.1 GB total image)
Bind mounts (patches, /models, cache repo) are not in the layer -- by design.
Verifying it skips the ~15 min AITER compile by relaunching from it and timing
to ready. Making it the launcher default (backup kept) if it does.

## Disk after the jitwarm commit: 138 GB free on / (was 149). Layer ~11 GB.
Images 250 GB total. 13.4 GB of stopped-container layers reclaimable via
`docker container prune` -- not urgent, not done (user said purge is OK
earlier in the session, but it is destructive; leave it for them).
Warm image verified to contain the 3 AITER modules that were rebuilt every
launch: module_aiter_core.so, mha_varlen_fwd_bf16_*.so, module_rmsnorm_quant.so.
Launcher IMAGE default now -> rocm10-mi210.7-aiter-jitwarm
(backup: launch_glm53.sh.bak-prejitwarm-*). Timing the first relaunch from it.

## *** JIT-WARM IMAGE VERIFIED: ~35 s to ready, 0 compiles (2026-09-15) ***
First relaunch from local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm as the
standing server (NAME=glm53 PORT=8145):
  READY after 20 s past T0 (~35 s from launch)   vs 600-900 s before
  AITER "start build" lines: 0                     vs 3 modules every launch
  correctness: '4', '4'
Two effects stack: the image carries the compiled .so files (no ~15 min AITER
rebuild), and the 164 GiB of weights were in the host page cache from the
previous launch (499 GiB RAM box), so loading ran at RAM speed rather than
NVMe. A cold-page-cache relaunch will be ~2-4 min, still without the compile.
The launcher default IMAGE now points at this tag (backup kept). Every future
relaunch -- including the user's own -- skips the compile.

## CORRECTION to the entry above: "~35 s to ready" and "page cache" are WRONG
vLLM's log for the same launch: "Model loading took 57.64 GiB memory and
155.4 seconds", "Graph capturing finished in 8 secs". My readiness loop set T0
AFTER the launch ssh returned, and that ssh BLOCKED until the launcher's own
readiness wait finished -- so the first check found the server already up.
(Same reason several launch commands earlier hit the 240 s tool timeout.)
The page-cache theory is also wrong: 164 GiB / 155 s ~ 1 GB/s is NVMe speed;
q38fn-lru's 138 GiB working set evicts the cache.

WHAT IS TRUE, confirmed two ways:
  AITER "start build" lines: 0   (was 3 modules, every launch)
  "[aiter] import [module_aiter_core]" / "[module_rmsnorm_quant]" -- the
  baked .so files are being REUSED, not rebuilt.
REAL time-to-ready: ~3-4 min (155 s load + 8 s capture + init)
                    vs ~17-18 min before (same load + ~15 min compile).
The warm image saves ~13-15 min PER LAUNCH. That is the deliverable for
"make the JIT compile work": it no longer runs.
LESSON (again): time things from the process's own log, not from a wrapper
whose start time depends on what a blocking ssh did first.

## USER (2026-09-15): "Keep trying without MTP. I feel like there could be a
## fast path. Also is there anything still bf16?"

### BF16 inventory (safetensors headers, excluding MTP layer 45 and norms)
Everything except the ROUTED experts is BF16 -- the quantizer's ignore list
(2303 entries) explicitly excludes attention, shared experts, dense MLP 0-2,
router gates, lm_head, and vision. Decode-path BF16 = 19.86 GiB total,
~9.9 GiB per rank per step under TP=2:
  KDA q/k/v/o_proj        ~9.9 GiB   (34 layers, [8192,4096] x3 + o_proj)
  lm_head + embed          2.36
  shared experts           1.97      (42 x 3 x [2048,4096])
  MLA q_a/q_b/kv_a/kv_b    1.03
  dense MLP layers 0-2     0.84
  vision tower             2.2       (replicated per rank, idle at decode)
  int4 expert SCALES       4.43 GiB  (BF16 [.,16]/[.,32] per expert; only the
                                     active 8/layer are read per step)
Roofline value of int8-ing all of it: ~9.9 GiB/rank -> ~5 GiB = ~4 ms/step
(~4%) at ~1.2 TB/s; KDA matvecs already run ~2.5x off roofline so the real
gain would be less. And vLLM hard-codes quant_config=None for KDA/MLA/vision
(kda.py:161-167, model.py:325/:1049) -- established earlier, still true.

### torch.compile is NOT ACTIVE for this model (server log, every launch):
  "`torch.compile` is turned on, but the model /models/glm53-w4a16-mtp does
   not support it."  (config/vllm.py:3027 -- no @support_torch_compile)
In this build only models/qwen4_exp/amd/model.py carries the decorator.
So the +25% "cudagraphs+compile" step was CUDAGRAPHS ONLY; inductor never
fused anything. The per-layer glue (norms, router math, gating, mHC glue,
KDA gate math) runs as separate eager kernels inside the captured graph.
The model does carry LEAF @torch.compile on small pieces (attention.py:53/61/70
indexer leaves, kda.py:116). Whole-model compile would need custom-op
wrapping for KDA (GatedDeltaNetAttention/PluggableLayer), mHC (tilelang),
the sparse indexer, and the expert-cache gather -- non-trivial, upside
unknown until the kernel profile says how much is small-kernel glue.

### Async scheduling: default-ON in this build (config/vllm.py:1273-1279:
None -> enabled unless pooling / non-eagle spec). SPEC=off, so it is on.
No lever there.

### Sampling: greedy vs top_p=0.95 (what real requests use via
generation_config.json: temperature 1.0, top_p 0.95), standing server,
300-token essay, warm:
  greedy 87 / 87 ms/tok   top_p 0.95: 89   default(genconfig): 89
~2 ms (2%) for the top-p sort over the 154,880 vocab. Within variance; not a lever.
(Also note: 87 ms/tok wall on this launch -- a favourable launch, vs the
100-110 steady-state spread recorded earlier.)

### NEXT: eager-mode kernel profile. Profilers were only ever tried under graph
replay (blind). Under --enforce-eager torch.profiler sees every kernel, and
per-kernel DEVICE time is the same as under replay (graphs only remove launch
gaps). This gives: kernel count per step, top kernels by device time, and
(device-time sum vs 63.7 ms no-gather floor) how much of the floor is
inter-kernel gap even inside the graph. Script: profile_eager.sh (port 8146,
GLM53_PROF=300, 12 steps). Requires stopping the standing server.

## *** KERNEL-LEVEL DECODE PROFILE OBTAINED (2026-09-15) ***
How: torch.profiler is blind on this build even in eager (0 device time), and
rocprofv3 --attach is refused in the container (rocattach status 1). Working
recipe (profile_rocprof.sh + rp/mi210-entrypoint.wrap mounted over
/usr/local/bin/mi210-entrypoint, because the launcher pins --entrypoint bash
AFTER EXTRA_DOCKER_ARGS so --entrypoint there is silently overridden):
  ROCP_TOOL_LIBRARIES=<sdk>/rocprofiler-sdk/librocprofiler-sdk-tool.so
  ROCPROF_OUTPUT_PATH=/rp/out ROCPROF_OUTPUT_FORMAT=csv ROCPROF_KERNEL_TRACE=1
  ROCPROF_STATS=1 ROCPROF_LOG_LEVEL=error       (NO LD_PRELOAD, NO
  ROCPROFILER_REGISTER_LIBRARY: LD_PRELOAD's ctor printed to stdout in every
  child and broke pip's pyproject hooks and AITER's hipconfig parse; the
  REGISTER var conflicts with _rocm_sdk_core's copy and is fatal.)
The tool flushes only on normal worker exit: GLM53_DIE_AT_STEP=<n> in the
patched model raises SystemExit in execute_model (RuntimeError is swallowed by
the busy loop and the worker is then SIGKILLed with nothing written).
Works in --enforce-eager. Under cudagraphs the worker did not flush (exit hung
-> killed); graph-mode trace still unavailable. Analyzers: rp_analyze.py
(window), rp_step.py (one exact step), rp_seq.py (kernel sequence with gaps).

ONE EXACT DECODE STEP, eager, TP0 (rp_step.py):
  2781 kernels/step, busy 70.3 ms  (wall 220 ms eager+tracer; graphs: ~96-105)
  gather      24.4 ms  (84 expert_cache_gather_k, ~295 us avg)
  moe_kernel  10.1 ms  (84 fused_moe_kernel_gptq_awq, 121 us avg: ~6x off roofline)
  other        8.7 ms  (1606 tiny kernels, ~5 us each)
  wvSplitK     7.9 ms  (282 GEMVs, ~29 us: the BF16 weights, AT roofline ~1.3 TB/s)
  hipBLASLt    5.5 ms  (84 Cijk BSS MT128x32x64, 45-58 us each)  <- see below
  mHC          4.3 ms
  topk/sort    3.7 ms  (84 gatherTopK + bitonic + warpMerge)
  nccl         2.9 ms  this step; 6-s average 49 ms/step (skew-inflated in eager)
  mla 2.0, kda 0.8
=> Under graphs (no launch gaps) the ~96 ms step is ~65 ms kernels + ~2700
   inter-kernel gaps (~4-5 us each on MI210 replay = ~12 ms) + NCCL skew +
   host. KERNEL COUNT is a first-class cost. "Fast path" = fewer, better kernels.

THE hipBLASLt BSS KERNEL: microbench (rp/gemm_micro.py under rocprofv3)
attributes it exactly to torch.mm(x_bf16, W_bf16.T, out_dtype=float32) =
GateLinear's `allow_cublas_router_gemm` path (gate_linear.py:212; the
"_router_gemm_cublas_capable" gate is is_cuda() OR is_rocm()). hipBLASLt picks
an MT128x32x64 tile for [1,4096]x[4096,288]: 45 us for a 2.4 MB read (~50 GB/s).
wvSplitK does the same GEMV in 3.2 us. AND it runs TWICE per MoE layer: the
model calls self.gate() and MoERunner (moe_runner.py:902 `if self.gate is not
None: router_logits = self.gate(hidden_states)`) recomputes it, discarding the
model's result. 84 calls x ~52 us = 4.4 ms/step for a 0.3 ms job.

THE ROUTER: grouped_topk with n_group=1 still runs the whole group machinery
(topk(2).sum -> topk(1) group select -> mask -> topk(8)): ~15 kernels/layer,
two sbtopk::gatherTopK at 38+27 us. n_group=1 makes group selection a no-op.

SHARED EXPERT ACT: SiluAndMulWithClamp is forward_native on ROCm
(activation.py:232: is_rocm -> native) = 7 elementwise kernels per layer.

## FAST PATH v1 (glm53_fastpath.py, mounted; imported+installed by the patched
## model at module import; GLM53_FASTPATH=0 disables; GLM53_FASTPATH_PARTS=
## gate,topk,act bisects). All plain Triton, static shapes, graph-capturable.
  gate : fp32-accumulate bf16 GEMV (M<=32) replacing torch.mm out_dtype=fp32.
         maxabs 3e-6 vs cuBLAS-fp32 reference.
  topk : one kernel: sigmoid -> +bias -> 8x argmax -> gather unbiased -> renorm
         -> x routed_scaling. ids match torch path exactly, weights to 6e-8.
         Patched at grouped_topk_router.grouped_topk (GroupedTopk binds the
         module global at __init__). Only when n_group<=1, sigmoid, bias, M<=64.
  act  : fused clamp/sigmoid/mul (1 ulp bf16 vs native's 7-kernel rounding).
  Microbench wall (incl. launch): gate 45->28 us, topk 194->32 us, act 58->34.
FIRST LAUNCH FAULTED during graph capture (HSA hardware exception, hang
analysis grid=[128] = a 2-warp kernel, i.e. NOT one of mine): diagnosis =
capture-time dummy inputs push NaN logits through the router; tl.argmax over
NaN returned an index in the masked pad region (>=288), and the downstream
expert-cache/gather kernels indexed out of bounds. torch.topk never returns
an id >= N. Fix: treat NaN scores as -inf and clamp idx to N-1. Relaunching.

## FAST PATH v1: launches clean after the NaN guard; timing + a correctness flag
ab_tuned_moe.sh (warm 900, 6 x 96 tok, take last): 93.2 95.2 98.1 92.4 88.3 88.4
ms/step -> 88.4 ms = 11.3 tok/s (standing config's launch spread: 87-110).
Not conclusive by itself (launch variance); kernel-count/busy deltas from an
eager trace will be the variance-free proof.
CORRECTNESS FLAG: qprobe.sh battery (greedy, 5 prompts):
  "What is 2+2? Answer with just the number."  -> reasoning coherent, content '2'  <- WRONG
  17*23 -> 391 ok | primes ok | sea sentence ok | capital of France -> 400 tok of
  reasoning, no content (rambled).
  (ab_tuned_moe's own 2+2 gate said '4' x3 on the same server.)
Running the identical battery with GLM53_FASTPATH=0 on a fresh launch to tell
a real regression from greedy-decoding chaos; bisect by GLM53_FASTPATH_PARTS if it is real.

## BISECT (fp_launch.sh, one part per launch, same 5-prompt greedy battery)
  baseline (off) : 4 | 391 | Paris | sentence | primes      crisp, 27-69 tok
  gate only      : 4 | 391 | Paris | ...                    correct
  topk only      : 4 | 391 | Paris! | ...                   correct
  act only       : '2+2' | garbled arithmetic then 391 | Paris. | verbose  <- BROKEN
=> the fused clamped-SwiGLU is the regression. Yet through the REAL class
(act_test2.py, swiglu_limit=10.0, alpha 1.0, beta 0.0, shapes [1,2048],
[1,12288], [7,2048]) it matches forward_native to 1 bf16 ulp (0.25 @ |y|~44-74).
So the mismatch is in what the MODEL feeds it, not the math. Launched eager
act-only with GLM53_FASTPATH_ACT_CHECK=400: compares mine vs native on the
live inputs and logs shape/stride/contiguity/stream per call.

## FAST PATH v1 RESULT: gate+topk shipped (default), act parked
act under cudagraphs fails 3/3 launches (garbled "2+2" answers) while eager
act-only is correct and the kernel matches forward_native to 1 ulp on the live
inputs (GLM53_FASTPATH_ACT_CHECK). The shared-expert side stream is NOT the
cause: is_multistream_safe() is False on ROCm for unquantized activations
(moe_runner.py:295), so shared experts already run on the main stream.
Root cause unknown; act is opt-in only (GLM53_FASTPATH_PARTS=gate,topk,act).

gate+topk under cudagraphs: battery = baseline-crisp (4 | 391 | Paris | ...).
Standing config timing: 97.7 99.7 89.8 89.7 100.8 101.0 ms -> inside the
87-110 launch-variance band; wall clock cannot resolve an ~8 ms delta.
VARIANCE-FREE PROOF (eager kernel trace, one exact step, TP0):
                      before      after (gate+topk)
  kernels/step        2781        2487    (-294)
  hipBLASLt BSS       84 / 5.5ms  0 / 0   (1.0 ms hipBLASLt left = MLA GEMMs)
  gatherTopK          84 / 3.7ms  0 / 1.0 ms topk-related
  other               8.7 ms      9.2 ms  (+126 tiny triton launches of mine)
=> ~6.7 ms/step less kernel time + ~294 fewer launches (~1.2-1.5 ms of
   replay gaps) = ~8 ms/step (~8%) at the same numerics (answers unchanged).
   (This step's nccl 94.7 / gather 36.9 ms are eager skew noise; ignore.)
Launcher now mounts glm53_fastpath.py; patched model installs it at import;
GLM53_FASTPATH=0 disables. Backup: launch_glm53.sh.bak-prefastpath-*.

NEXT candidates from the same trace, ranked by ms/step:
  fused_moe_kernel_gptq_awq  10.5 ms, ~6x off roofline at M=1  (kernel/config)
  "other" 1522 tiny kernels  ~9 ms busy + ~6 ms replay gaps     (fusion, laborious)
  act fusion (270 kernels)   ~2.7 ms  (blocked by the cudagraph mystery)
  shared-expert overlap      ~2.5 ms  (is_multistream_safe is False on ROCm; risky)
  custom all-reduce          ~3 ms    (gated off gfx90a upstream; silent-corruption risk)

## TUNED MoE CONFIG WAS NEVER APPLIED (2026-09-15)
vLLM looks for E=144,N=1024,...json (N = w2.shape[1]/intermediate per rank),
the tuner wrote E=144,N=2048. The earlier "tuned config is 6% worse" e2e
result was the DEFAULT config measured under launch noise. Copied the file to
the N=1024 name; kernel-tracing it (variance-free) -- default
fused_moe_kernel_gptq_awq avg = 126.7 us (gate_up ~138, down ~52).
Scoping an M=1 int4 GEMV replacement: B is [E, N, K/2] uint8 (K packed, low
nibble first: shifter=(k%2)*4), scales [E, N, K/128] bf16, no zero-points
(symmetric). Roofline per layer per rank ~42 us vs ~190 us now -> ~6 ms/step.

## TUNED MoE CONFIG: applied properly (N=1024 filename), kernel-traced: WORSE
  fused_moe_kernel_gptq_awq avg: default 126.7 us -> tuned 134.1 us. Dead end,
  now closed on a variance-free measurement (the tuner's search space cannot
  fix a kernel that tiles 16 MFMA rows around one real token).

## FAST PATH v2: M=1 int4 MoE GEMV (glm53_fastpath.py part "moe", default ON)
Intercepts the module-level `fused_moe_kernel_gptq_awq[grid](...)` launch (a
_GptqAwqDispatch object; __getitem__ returns a launcher) and, only when
num_valid_tokens <= top_k (M == 1), int4, no zero-points, K-packed uint8
layout, launches `_moe_int4_gemv_m1`: one program per (expert block, 32 output
rows), streams B in 512-wide K chunks (4 quant groups), dequantises in
registers (nibble - 8), fp32 FMA against the token row, per-group scale
applied to the 4 partial sums. Same sorted_token_ids / expert_ids / C layout
and routed-weight semantics as the original; everything else falls through.
Self-test vs the original kernel (E=8, EP-style -1 ids): rel err 3e-3 of
absmax (bf16 output rounding). Sweep at the real 4-expert-per-rank shape,
event-timed (rp/moe_gemv_tune2.py):
  gate_up [4096x4096]: orig ~138 us -> v1 (BK=128,BN=32,nw=4) 84 -> v2 (BK=512,BN=32,nw=2) 67 us
  down    [4096x2048]: orig  ~52 us -> 45 -> 36 us
  => ~87 us/layer x 42 = ~3.7 ms/step. Still only ~500 GB/s (latency-bound
  loop; roofline ~1.2 TB/s) -- more possible with software pipelining.
gate+topk+moe(v1) under cudagraphs: battery identical to baseline; timing
89.5 89.6 92.0 89.6 89.5 89.5 ms/step = 11.17 tok/s, with a Chrome at 1077%
CPU on the host. Most stable run of the session.
Now: v2 ported, self-test, eager trace, relaunch + battery + timing (chain running).

## MoE GEMV: the layout question (2026-09-15)
The live expert weights are NOT the K-packed uint8 [E, N, K/2] the checkpoint
ships: this build's compressed_tensors_moe_wna16.py:596-640 repacks them on
ROCm to N-packed int32 [E, K, N/8] (+ scales transposed to [E, K/gs, N]) so
fused_moe_kernel_gptq_awq takes its tl.interleave path -- an earlier-session
change, justified there by 1.45-4.8x on the ORIGINAL kernel at M=1..128.
(The mounted /home/dave/vllm-patch/moe_wna16_utils.py is the chunked repack.)
My GEMV (v2) wants K-packed uint8: 67/36 us vs original 138/52 us. On the
int32 N-packed layout the GEMV is no better than the original:
  v3 (tl.interleave unpack): best 188 us gate_up / 100 us down (orig 193/100 synthetic)
  v4 (8 shift/mask passes on the word tile): best 186 / 97 us
Diagnosis: with k as the tile's ROW axis the per-group reductions cross
threads (LDS traffic); v2 reduces along the last axis in registers.
Tried and abandoned: converting back to uint8 after vLLM's repack (my
_repack_npacked_to_kpacked) -> OOM at load (974 MiB free when the 1.12 GiB
int32 temp is allocated; fragmentation on top of the 0.97 budget).
Options: (a) keep uint8 by skipping vLLM's repack -> decode -3.7 ms/step but
prefill's MoE kernel loses its 1.45-4.8x; (b) v5: tl.trans the packed tile so
k is the last axis, keep int32 for prefill -- sweeping now.
"moe" is opt-in again (GLM53_FASTPATH_PARTS); standing server = gate,topk.

## uint8 repack path: v5 (tl.trans) also dead (255+ us); going with option (a)
Repack-back-to-uint8 launches now (alias leak fixed: vLLM aliases
layer.w13_weight -> w13_weight_packed before my replace; re-pointing it stops
the 1.12 GiB/tensor/layer leak). New failure: "Model loading took 59.86 GiB"
(was 57.64) -> no KV memory left. +2.2 GiB/rank ~ one layer of expert weights.
Suspects: allocator fragmentation from the 1.12 GiB churn (expandable
segments; try empty_cache per layer), or interaction with the UVA offloader
(offloaded params live in pinned host memory; a replace_parameter after the
offloader ran would re-materialise them on GPU). Reading offloader/uva.py.

## uint8 layout: garbage root-caused and fixed (2026-09-15)
Layout-only arm (skip vLLM's int32 repack, original kernel on uint8) first
produced pure "-----" output. Kernel-pair test (rp/uint8_vs_int32.py):
original uint8 path == original int32 path BIT-EXACT, GEMV within 4e-3 (bf16
rounding) -- so the kernels were fine and the PLUMBING was wrong:
compressed_tensors_moe_wna16.py permutes the scales [E,N,K/gs] -> [E,K/gs,N]
right before _setup_kernel(layer), and the kernel object captures the layer's
tensors THEN. Undoing the permute after process_weights_after_loading
returned replaced the layer attribute but the kernel kept the permuted
scales -> garbage. Fix: wrap cls._setup_kernel and put the scales back BEFORE
delegating. Layout-only battery now identical to baseline (4 | 391 | Paris).
Also learned: identity repack + replace_parameter(w.data) costs +1.74 GiB
GPU per offloaded layer transiently (oracle copies to device; re-offloaded at
device_loading_context exit) -- same as the int32 flow.
Open: "Model loading took 59.86 GiB" (+2.2 GiB vs 57.64) with the uint8
layout at OFFLOAD_GB=28 -> no KV room; running at OFFLOAD_GB=31 for now
(~+2 ms/step of gather) until the source of the 2.2 GiB is found.
Measuring now: layout-only decode+prefill, then GEMV arm decode+prefill.

## uint8 layout + GEMV: measured, and it LOSES on this box (2026-09-15)
  config (all OFFLOAD_GB=31, cudagraphs)      decode ms/step   prefill tok/s (279/1105/2211/4423)
  layout-only (orig kernel, uint8)            115.7-118.6      31 / 86 / 79 / 189
  layout + M=1 GEMV                           107.3            35 / 95 / 92 / 194
  int32 baseline (gate,topk) @28              89.5             98 / 246 / 300 / 670 (earlier)
Eager trace of layout+GEMV: _moe_int4_gemv_m1 54 us avg (84/step = 4.5 ms,
down from 10.5-11.6) BUT expert_cache_gather_k 364 us avg (was 295-302):
+21% per call, only ~11% explained by the extra 3 GiB offloaded. The
original kernel's uint8 (scalar-shift) branch is ~3x slower for prefill.
Host side: workers' RSS 55.8 -> 59.9 GB (the loader's re-offload allocates
new pinned buffers; the offloader's originals stay alive) and ksmd at 92%
CPU during these runs. Running int32@31 (apples to apples) and uint8@28.

## VERDICT on the uint8 MoE GEMV (2026-09-15): CLOSED, opt-in only
  int32 (gate,topk) @28   89.5 ms/step   <- STANDING
  int32 (gate,topk) @31   92.2           (+3 GiB offload = +2.7 ms, as modelled)
  uint8 + GEMV      @28  103.6           (-6 ms MoE kernel, +~20 ms elsewhere)
  uint8 + GEMV      @31  107.3
  uint8, orig kernel @31 115.7-118.6
Plus prefill ~3x slower on uint8. Kept in glm53_fastpath.py as
GLM53_FASTPATH_PARTS=gate,topk,layout,moe for anyone who wants to chase the
gather regression; default is gate,topk.

## STANDING SERVER (final for this pass): :8145, gate+topk fast path, OFFLOAD 28
  89.8 ms/step x3 (11.14 tok/s), battery 4 | 391 | Paris.
  Session total: 7.89 -> 11.1 tok/s (+41%): cudagraphs (+25%), memory knobs
  (+7%), gate GEMV + fused top-k (~8% by kernel trace: -6.7 ms kernels,
  -294 launches/step). Launch-to-launch spread is ~+/-7%; 89.5-89.8 has now
  been reproduced on 4 separate launches of this config.

## Item 3 (VRAM trims), 2026-09-15
vision_offload (UVA views of the 2.2 GiB tower): GPU MEMORY_FAULT at the
profiling run -- hipBLASLt/wvSplitK cannot read host-mapped weights the way
the expert-cache gather kernel does. Dropped (opt-in code left, marked broken).
Vision on the standing server verified working (vision_probe.sh: "Red square").
Trial A2 = --kv-cache-memory 0.4 GiB + --max-num-batched-tokens 1024, OFFLOAD 26.

## Trial A2 (item 3): --kv-cache-memory 0.4 GiB + --max-num-batched-tokens 1024, OFFLOAD 26
  "Model loading took 60.53 GiB" (weights up 2.9 GiB = 2 GiB less offload + slack)
  KV 10,082 tokens (>8192 needed for one 8k request)
  decode 86.85 ms/step x2 (11.5 tok/s)   vs 89.5-89.8 at OFFLOAD 28  -> -3 ms
  prefill 107 / 217 / 239 / 471 tok/s    vs 98 / 246 / 300 / 670     -> the 1024 chunk
                                            costs ~30% on long prompts
  battery: 4 | 391 | Paris (unchanged)
  => keep the KV cap; the chunk size is a decode-vs-prefill trade the user
     should pick (KV cap alone frees ~0.5 GiB -> OFFLOAD 27, ~-1 ms).
Trial B = same settings + shared-expert aux-stream overlap (forced past the
two ROCm gates in shared_experts.py / moe_runner.py:295).

## Trial B/C (item 2, shared-expert aux-stream overlap): LOSES
  B: overlap @ OFFLOAD 26 + kv cap + 1024 chunk -> hipErrorOutOfMemory at start
     (the aux stream's pool needs headroom we no longer have).
  C: overlap @ OFFLOAD 27 + kv cap, default chunk -> answers correct, but
     decode 94.7-97.2 ms/step vs ~88 expected for the same memory settings
     without it: the cross-stream event/sync costs more on MI210 than the
     ~50 us shared expert it hides. Left opt-in (GLM53_FASTPATH_PARTS=...,overlap).
Trial D = kv cap only @27, default chunk (candidate final); then item 1 (compile).

## Trial D (item 3, final form): --kv-cache-memory 0.4 GiB, OFFLOAD 27, default chunk
  decode 88.1-88.2 ms/step x4 (11.35 tok/s)   vs 89.5-89.8 @28  -> -1.5 ms
  vision OK ("Red square", 73 tok)  |  prefill 103 / 282 / 267 / 566 tok/s
  battery unchanged. The 1024-token chunk (A2) is what broke the image
  request (500 tok of reasoning, no answer) and cost ~30% long-prompt
  prefill -- not shipped; A2's extra -1.5 ms is available if prefill/vision
  do not matter to the user.
Trial E = D + GLM53_COMPILE=1 (item 1: @support_torch_compile on
Glm5NextModel with KDA/MLA/MoE wrapped as vllm custom ops).

## Item 1 (torch.compile) RESULT: works, gains nothing (2026-09-15)
GLM53_COMPILE=1: @support_torch_compile on Glm5NextModel with KDA / MLA /
MoE-block wrapped as vllm custom ops (glmfastpath.<tag>.<n> keys; lessons:
custom-op funcs need full type annotations; keys must be deterministic
because the AOT-compiled graph is cached on disk with them baked in; and
dot-separated with exactly one integer for extract_layer_index()).
Dynamo traces the whole model (bytecode 1.9 s, inductor 4-5 s -- a small
graph), capture OK, answers correct, vision OK.
  decode 88.5 ms/step  vs 88.1 without compile (same D settings)  -> 0
  prefill 80 / 280 / 264 / 563 tok/s -> same within noise
Reason: the ~1,600 tiny kernels live INSIDE the MoE block (moe_align, expert
map/index copies, shared-expert act, cache manage/remap), which is opaque
to the model-level graph; only norms/residual/mHC glue got fused. Left opt-in.

## STANDING CONFIG after items 3/2/1: D = kv cap + OFFLOAD 27 (88.1 ms, 11.35 tok/s)
  NAME=glm53 PORT=8145 TP=2 UTIL=0.97 MAXLEN=8192 MAX_NUM_SEQS=1 OFFLOAD_GB=27
  SLOTS=8 POLICY=lfu SPEC=off EXTRA_VLLM_ARGS="--enable-expert-parallel
  --kv-cache-memory 429496729" ./launch_glm53.sh

## *** PREFILL: every earlier number in this ledger is CONTAMINATED (2026-09-15) ***
prefill_bench.sh sends its prompts in ascending length, built from the same
filler, so each request shares a prefix with the previous one -- and prefix
caching is ON whenever chunked prefill is on. The long-prompt figures
(4423 tok -> 566-670 tok/s) were partly served from cache.
prefill_bench2.py: fresh random-syllable prompts (no shared prefix), repeats,
max_tokens=1. First honest numbers, standing config (chunk 2048, OFFLOAD 27):
   892 tok  3971 ms   225 tok/s
  1806 tok  5388 ms   335
  3594 tok 13229 ms   272
  7213 tok 28696 ms   251
So real prefill is ~250-340 tok/s, NOT 566. (2211-tok agreement between the
two benches is why this went unnoticed: only the longest prompt was cached.)

## The cost model, fitted to those four points
Per-chunk time is not constant; fit time = n_chunks*fixed + tokens*per_token:
  892 vs 1806 (both 1 chunk): +914 tok -> +1417 ms  => 1.55 ms/token
  extrapolate to 0 tokens                          => ~2.6 s fixed per chunk
The 2.6 s/chunk matches the predicted PCIe stream of the 27 GiB offloaded
expert set (27 GiB / ~12 GB/s = 2.25 s). CONFIRMED MECHANISM: cache.py's
fits() is `topk_ids.numel() <= SLOTS(8)`, so any prefill chunk reads experts
through UVA instead of the slot cache -- one full stream per chunk.
BUT per-token compute (1.55 ms/tok = 645 tok/s, ~21% of bf16 peak) dominates
at long prompts, so bigger chunks are worth ~27% at 7k tokens, not 2-3x.

## CHUNK-SIZE HYPOTHESIS FALSIFIED (2026-09-15)
Prediction was "bigger chunks amortise the per-chunk expert stream". Measured
(prefill_bench2.py, best of 2, honest prompts; decode = ab_tuned_moe last runs):
  chunk/offload   528tok  2010   4090   7099  | decode ms
  2048 / 27        158     343    288    250  |  88.1     <- STANDING
  4096 / 30        139     303    300    181  |  91.7
  8192 / 33        126     272    258    234  |  95-101
Bigger is worse everywhere, and worse for decode too. The confound is the
point, not a flaw: a bigger chunk needs a bigger activation reservation
(2.82 GiB at 2048, ~2x per doubling), which on this box can only be paid for
by offloading MORE experts -- and offloaded bytes are streamed once per chunk
at prefill AND missed more often at decode. The knobs are not independent
here; chunk size buys nothing it does not immediately give back.
Also measured: 27 -> 33 GiB offload costs decode 88 -> 95-101 ms, i.e.
1.3-2.2 ms/GiB, worse than the 0.7 ms/GiB modelled from earlier deltas.
=> 2048 (the vLLM default) is already right on this box. The real prefill
lever is the same as the decode lever: FREE VRAM so less is offloaded.

## Standing server restored and re-verified (2026-09-15)
chunk 2048 / OFFLOAD 27 / kv cap 0.4 GiB / fastpath gate,topk:
  88.15 88.23 88.19 88.19 88.15 88.21 ms  (6/6, tightest spread of the session)
  battery 4 | 391 -- and this was WITH a busy host (Chrome 1107% CPU, a second
  vLLM at 44 GiB RES, ksmd 57%), so the config is robust to the neighbour.
The chunk-1024/OFFLOAD-26 arm failed to start (container exited during
startup, unrelated to memory -- A2 ran that combination earlier). Not chased.

## Prefill: what is actually left
Two terms, both measured: ~2.45 s per chunk of expert streaming (unavoidable
while SLOTS=8 < any prefill chunk's routed set) + ~1.67 ms/token of compute
(= a 600 tok/s ceiling even at zero streaming).
UNTRIED and the natural next step: the MoE autotune that failed earlier was
run with --batch-size 1, i.e. tuned for DECODE. Prefill runs the same kernel
at M~2048 where tiling actually matters. benchmark_moe_glm.py --batch-size
2048 is a different experiment and is where the 1.67 ms/token would move.

## DeepSeek-V4.1-Flash assessed against this box (2026-09-15)
Model card: 552B backbone, 8B active/token prefill & 16B decode, 40 layers
(20 causal encoder + 20 decoder), 384 routed experts top-6 + 1 shared,
Engram conditional memory 196B "sparsely accessed via token-based lookup",
FP4 KV at 890 B/token, DSpark spec decode, DeepSeek-ViT.
The user's instinct is mechanically right: a lookup-addressed table (Engram,
like Gemma-3n PLE) is the one parameter class that offloads for ~free, because
a lookup reads rows per token while an expert GEMM reads whole matrices.
Three checks against the box, all negative:
 1. `engram` appears 0 times in this vLLM build. models/deepseek_v4/ exists
    (and its config expectations match the card: 384/top-6/shared 1), but the
    Engram has no implementation here.
 2. Size is worse: even granting the 196B Engram lives on host, the GEMM-
    carrying remainder ~356B ~= 200 GiB int4 vs GLM-5.3-Flash's 164 GiB, so
    ~40 GiB/rank offloaded instead of 27. (Card is ambiguous on whether the
    552B includes the Engram; if not, far worse.)
 3. deepseek_v4/amd/model.py:207 _heterogeneous_shared_expert_enabled needs
    gfx950 AND TP=8 AND expert-parallel OFF AND AITER MoE AND fp4 experts --
    this box fails all five. dspark.py header is gfx950 too. Same fallbacks
    as GLM, with more parameters.

## Vision-tower offload via the SUPPORTED path: works, but the prize was 4x
## smaller than I estimated -- net neutral (2026-09-15)
vLLM has tower offload: UVAOffloader.supports_tower_offload = True, and
interfaces.py:386 wraps multimodal towers at construction (weights never
allocated on device). GLM5Next registers its tower via _mark_language_model,
so `--cpu-offload-params experts visual` reaches it.
  offloader log: "Total CPU offloaded parameters: 0.54" (tower) then 27.87
  Model loading took 58.85 GiB (vs 59.39 baseline) -- 0.54 GiB/rank freed
  VISION STILL WORKS ("Red square") -- the supported path does NOT fault,
  unlike my hand-rolled post-load UVA views (HSA MEMORY_FAULT).
CORRECTION to my earlier estimate: the tower is NOT 2.2 GiB/rank. That figure
came from summing BF16 tensors only; the checkpoint's ignore list exempts just
visual.blocks.0-3 + merger, so most of the tower IS int4. Real cost 0.54 GiB.
Net effect is neutral-to-negative: the budget is a TOTAL, so adding visual
offloads 0.54 vision + 27.33 experts (= slightly MORE expert offload) and
leaves 0.54 GiB of VRAM unused. Spending it needs OFFLOAD 26, which does not
fit (launch failed). At 0.54 GiB x ~1.3-2.2 ms/GiB the ceiling was ~1 ms
anyway -- under this box's noise floor. NOT ADOPTED.

## *** CORRECTION: Engram IS upstream, with CPU offload ON BY DEFAULT and an
## *** AMD path. The user was right; my "0 occurrences" was about OUR IMAGE only.
gh code search, vllm-project/vllm (20 hits):
  vllm/config/engram.py            <- EngramConfig
  vllm/models/deepseek_v41/{common,nvidia,amd}/engram.py, model.py, rocm.py,
                                   dspark.py, vl_model.py   <- amd/ EXISTS
  vllm/engine/arg_utils.py, distributed/parallel_state.py, config/vllm.py
  vllm/models/qwen4_exp/nvidia/ngram_embedding.py
EngramConfig:
  cpu_offload: bool = VLLM_PLE_CPU_OFFLOAD   (DEFAULT ENABLED)
    """Store embedding weights in pinned CPU memory for UVA lookup."""
  embedding_across_dp, dp_shared_memory (share host copy between DP replicas)
  _NGRAM_LAYER_FIELDS = {DeepseekV41ForCausalLM: engram_layer_ids,
                         Qwen4Exp*: ple_layer_ids}
common/engram.py:588 `_engram_lookup_kernel`, and its own comment at :609:
  "`weight`/`scales` may address pinned host memory through UVA."
=> it is a TRITON GATHER from pinned host memory. That op class is PROVEN on
   gfx90a -- expert_cache_gather_k does exactly this every decode step. It is
   the GEMM-from-UVA case that faults (my vision-tower attempt), not this.

THE ONE GATE: config/engram.py:77 `or not current_platform.is_cuda()` inside
verify_model_config -> raises "EngramConfig requires ... CUDA" on ROCm.
is_cuda() is False on ROCm (is_cuda_alike() is the both-platforms predicate).
Likely a one-word patch, and this box already carries an mi210 patch set.

HONEST REMAINING UNCERTAINTY (the part Engram does NOT fix):
  V4.1-Flash 552B backbone, of which Engram 196B -> host for ~free.
  GPU-resident remainder ~356B ~= 182 GiB int4 -> ~91 GiB/rank -> ~36 GiB/rank
  offloaded, vs GLM-5.3-Flash's 27 GiB/rank today. Expert side is still
  somewhat worse on size; offsets are FP4 KV (890 B/token) and 16B active/token
  at decode. Genuinely close -- could go either way, not a slam dunk.
  Also unknown: whether deepseek_v41/amd/ is gfx950-gated like deepseek_v4's
  _heterogeneous_shared_expert_enabled (gfx950 + TP=8 + EP off + AITER + fp4).
NEXT STEP IS A BUILD, not a config change: our image is 0.28.1rc0+mi210.7 and
predates all of this.

## VERDICT: DeepSeek-V4.1 on gfx90a -- Engram YES, attention NO (hardware)
Checked upstream main via gh, before spending a build.
GREEN, the Engram half (the user's actual thesis):
  - deepseek_v41/amd/model.py has NO gfx950/gfx942 gate (unlike deepseek_v4's
    _heterogeneous_shared_expert_enabled, which needs gfx950+TP8+EP-off+AITER+fp4)
  - EngramConfig.cpu_offload defaults ON; common/engram.py:588
    _engram_lookup_kernel is TRITON reading pinned host memory via UVA
  - expert_dtype is parameterised (fp4 is only the default; the weights mapper
    is rebuilt per-instance when expert_dtype != "fp4"), so int4 is conceivable
RED, the attention half:
  - deepseek_v41/amd/model.py:139 RAISES for any backend that is not
    AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4. One backend, no fallback.
  - that backend (deepseek_v4/amd/rocm.py DeepseekV4ROCMAiterMLASparseBackend)
    is built on AITER FP8: gemm_a8w8_blockscale_bpreshuffle, group_fp8_quant,
    kv_cache_dtype "fp8_ds_mla", rocm_aiter_mla_sparse, _ON_GFX950 helpers
  - gfx90a/CDNA2 has NO FP8 MFMA (FP8 arrives with CDNA3/MI300). Not a software
    gate -- missing silicon.
WHY THE R9700 BUILDS WORK: RDNA4 (gfx1201) HAS FP8. The user's batched code is
fine there; it cannot transfer to MI210 for a hardware reason.
=> A build would give a working Engram and a model that refuses to start on
   attention. Do not spend it for V4.1 on these cards.

## THE TRANSFERABLE PART: Qwen4Exp carries the SAME PLE/Engram offload machinery
EngramConfig._NGRAM_LAYER_FIELDS maps BOTH DeepseekV41ForCausalLM
(engram_layer_ids) AND Qwen4Exp{ForCausalLM,ForConditionalGeneration}
(ple_layer_ids) -- same cpu_offload=UVA-lookup path.
qwen4_exp/amd/ in OUR CURRENT IMAGE has no gfx950/fp8 gate at all:
  indexer_qsa.py:96 "Qwen4Exp QSA currently requires BF16"
  model.py:86 without_modelopt_fp4(...)   <- actively handles the non-fp4 case
  attention via MambaAttentionBackendEnum GDN_ATTN / SHORT_CONV (Triton, the
  same family as GLM's KDA, proven on this box)
and the PLE machinery is already present: common/ple.py PLEVocabParallelEmbedding,
amd/ple_layer.py, custom ops qwen4_exp_amd_ple_ngram_embedding /
qwen4_exp_ple_short_conv. (ple_layer_ids is not yet in this image's
transformers_utils config -- that part is newer than 0.28.1rc0+mi210.7.)
=> If a Qwen4Exp-family PLE model exists at a size that fits, the Engram-style
   "big lookup table on host for free" win IS reachable on gfx90a. That is the
   build worth considering, not V4.1.

## *** CORRECTION to the V4.1 verdict: the FP8 blocker is NOT real (2026-09-15)
I said "missing silicon, not gating". Wrong. gfx90a indeed has no FP8 MFMA,
but the V4.1 ROCm attention path does not USE FP8 MFMA -- it uses FP8 as a
STORAGE format unpacked inside Triton. Its own assertion text says so:
  rocm_sparse_attn_decode: "ROCm Triton sparse decode expects uint8
                            fp8_ds_mla SWA cache / extra cache"
i.e. uint8 bytes + a Triton kernel, not fp8 tensors + matrix units.

PROVEN ON THE BOX (rp/fp8_triton_gfx90a.py, gfx90a, Triton 3.8.0):
  float8e4b8    OK  256/256 finite   <- the ROCm fnuz e4m3 (what ds_mla uses)
  float8e5b16   OK  256/256 finite
  float8e4nv    OK  253/256 finite   (3 NaN = random bytes hitting NaN codes)
  float8e5      OK  248/256 finite   (same)
  dot/float8e4b8 OK finite, absmax 2.75e4   <- widened -> bf16 -> tl.dot -> MFMA
So uint8->fp8 bitcast -> bf16/fp32 widen -> MFMA works on CDNA2 today.
(And GLM already relies on this: its indexer stores fp8 KV and runs
_fwht_quant_kernel / indexer_k_quant_and_cache_triton every step.)

The other three FP8 uses all have escapes, checked in dsv41 amd/rocm.py:
  gemm_a8w8_blockscale_bpreshuffle (wq_b) -> _wq_b_uses_aiter_block_scaled is
     a cached predicate with an explicit "otherwise fall back" branch
  prepare_attn_preshuffle -> `if not rocm_aiter_ops.is_enabled(): return`
  _trust_dsv4_extra_cache_nan_free -> gated on _ON_GFX950, pure optimisation
=> ANSWER TO "convert the fp8 to int8?": you should not need to. The format
   that matters is storage, and this card already reads it.

STILL OPEN before a build is worth it:
  1. sizing: ~356B GPU-resident -> ~36 GiB/rank offloaded vs 27 today
  2. the FP8-compute fallbacks are written for "AITER present but not
     applicable", not specifically for CDNA2 -- untested there
  3. perf: software fp8->bf16 widening per element is slower than native FP8
     MFMA (GLM pays this today and still does 88 ms/step, so viable not free)
  4. unaudited: mega-MoE, DSpark, compressor, SWA may carry other CDNA3 assumptions

## AUDIT COMPLETE: no hard gfx950 gate anywhere in the V4.1 tree (2026-09-15)
Fetched upstream deepseek_v41/{amd/model,amd/rocm,amd/dspark,amd/vl_model,
attention,sparse_mla,common/engram}.py (4,800 lines). Arch references: FIVE,
all in amd/rocm.py, none of them a requirement:
  :25  import _ON_GFX950
  :51  _trust_dsv4_extra_cache_nan_free -> returns a BOOL (False here = take
       the safe path). Optimisation.
  :180,:183  comments
  :387 `if _ON_GFX950: ragged_out = ragged_out[:max(n,1)]` -- graph-stable
       base-pointer slicing for a sync-free selector. Skipping it is correct.
common/engram.py:211 `@triton.jit` -- the engram lookup is Triton. Portable.
Hard raises that matter:
  amd/model.py:141  backend must be ROCM_FLASHMLA_SPARSE_DSV4  (that IS the
                    one we want -- satisfiable, not a blocker)
  amd/model.py:385  mega-MoE requires --enable-expert-parallel (we use EP)
  amd/model.py:567  engram needs lookback_token_ids from the runner; the
                    DBO/ubatch wrapper drops model kwargs -> do not enable DBO
  config/engram.py:77  `not current_platform.is_cuda()` -> RAISES on ROCm
                    <-- THE ONLY REAL GATE, and it is one word.

## PLAN: DeepSeek-V4.1-Flash on 2x MI210, supported paths only
Principle: no monkeypatching. One upstreamable source change; everything else
is flags, checkpoint choice, and the existing mi210 build pipeline.

P0 -- DERISK WITHOUT A BUILD (~2 h, no GPU monopoly)
  0.1 Checkpoint. expert_dtype defaults to "fp4"; MXFP4 MoE is dead on this
      box (mxfp4.py cannot import triton_kernels). We need a W4A16 /
      compressed-tensors int4 expert quant. lvkaokao/DeepSeek-V4.1-Flash-
      W4A16-Engram-AutoRound is exactly that -- confirm from its config.json:
        expert_dtype, engram_layer_ids non-empty, engram_num_embeddings,
        quantization_config (group size, symmetric, ignore list),
        total on-disk size, and whether engram tables are quantised too.
  0.2 Size the split: GPU-resident = total - engram. Kill if
      (total - engram)/2 - 64 GiB > ~36 GiB per rank.
  0.3 Confirm amd/model.py's non-fp4 weights-mapper branch covers the
      checkpoint's tensor names (_make_deepseek_v4_weights_mapper).

P1 -- THE ONE SOURCE CHANGE (minutes)
  config/engram.py:77  current_platform.is_cuda() -> is_cuda_alike()
  Justification for upstream: EngramConfig gates only a Triton lookup kernel
  over UVA host memory; deepseek_v41/amd/ exists; nothing in the engram path
  is CUDA-specific. File it as a PR -- it is the kind of one-liner that lands.

P2 -- BUILD (their existing pipeline)
  upstream main pinned to a commit containing deepseek_v41, + P1 patch.
  Reuse the AITER JIT-warm commit trick (saves 13-15 min/launch).
  Keep the mi210 patch set that is already carried.

P3 -- BRING-UP, cheapest failure first (each step gates the next)
  a. load only: --load-format dummy, TP=2, EP on -> catches weight-mapper and
     config plumbing without paying 550B of I/O
  b. real load, engram offload ON (default): confirm in the log that engram
     landed in pinned host memory and that "Model loading took" matches the
     P0.2 arithmetic. This is the whole thesis -- verify it explicitly.
  c. correctness: 2+2 / retrieval battery at max_tokens>=200, eager first
     (--enforce-eager) to keep capture out of the picture
  d. attention fallbacks: the FP8-compute escapes (_wq_b_uses_aiter_block_
     scaled, prepare_attn_preshuffle) are written for "AITER present but not
     applicable", never exercised on CDNA2. Expect to fix one or two.
  e. cudagraph capture. GLM precedent: data-dependent .item()/slicing breaks
     capture (I rewrote fp8_paged_mqa_logits for exactly this). Budget for it.
  f. perf: ab_tuned_moe.sh + prefill_bench2.py, same protocol as GLM.
  DO NOT enable DBO/ubatching (amd/model.py:567 drops engram kwargs).

P4 -- DECISION GATE
  Ship only if decode < 88 ms/step AND prefill >= ~340 tok/s at 2k with
  correct output. Otherwise record and stay on GLM-5.3-Flash.

REUSABLE FROM THIS SESSION
  JIT-warm image recipe; rocprofv3 recipe (ROCP_TOOL_LIBRARIES, no LD_PRELOAD,
  GLM53_DIE_AT_STEP to flush); rp_step.py / rp_analyze.py; prefill_bench2.py
  (unique prompts -- the prefix cache WILL lie to you); ab_tuned_moe.sh;
  the expert cache on its expert-parallel-support branch.

BIGGEST RISK, restated: P0.2. Engram removes 196B from the VRAM budget, which
is the win -- but the remaining ~356B is still larger than GLM's 321B, so the
expert-side offload gets worse before the engram saving is counted. The whole
bet is that (a) engram really is free on host and (b) top-6-of-384 with 16B
active gathers fewer bytes/token than GLM's top-8-of-288. (b) is UNMEASURED.

## P0 DONE: lvkaokao/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound inspected
config.json / HF API:
  DeepseekV41ForCausalLM, model_type deepseek_v41, dtype bfloat16
  384 routed experts, top-6, 1 shared | hidden 5120, moe_intermediate 2304,
  40 layers | head_dim 512, qk_rope 64, index_head_dim 128, q_lora 1280,
  o_lora 1024, sliding_window 128, vocab 129280
  engram_layer_ids [1, 14]        engram_num_embeddings [384006168, 384016682]
  engram_n_heads 8  engram_head_dim 256  engram_vocab_size 16,000,000
  engram_max_ngram_size 4  engram_compressed_vocab_size 99,092
  quant: auto-round, 4 bit, GROUP_SIZE 32, sym, int,
         packing "auto_round:auto_gptq"
  48 shards, 301.4 GB of weights (repo 451.7 GB incl. the tech-report PDF)

SIZING, and it reconciles exactly:
  engram table = 384,006,168 rows x 256 dims = 98.3B params PER LAYER,
  x2 layers = 196.6B  <- this IS the card's "Engram 196B"
  at int4+group32 (~4.5 bits): 196.6e9 x 0.5625 = 110.6 GB
  and shards 47+48 are ~55.6 GB each = 111.2 GB. The engram tables ARE
  int4-quantised and ARE the last two shards. Nothing to convert there.
  => GPU-resident backbone = 301.4 - 111 = ~190 GB = 177 GiB
     per rank TP=2: 88.4 GiB; usable VRAM/rank ~58-60 after act+KV+graphs
     => ~28-30 GiB/rank offloaded  vs GLM's 27 GiB/rank TODAY.
  P4 kill gate was ">36 GiB/rank": PASSES, but it is a wash, not a win.

WHAT STILL NEEDS CONVERTING / WATCHING (the actual question)
  1. FP8: NOTHING. The checkpoint is bf16 + int4 auto-round. The fp8 concern
     was only the runtime KV dtype, already proven fine on gfx90a.
  2. PACKING: "auto_round:auto_gptq" -- NOT compressed-tensors like GLM. A
     different vLLM loader. Possibly FAVOURABLE: gptq packs int32 N-first,
     which is exactly what this build's ROCm path repacks compressed-tensors
     INTO (compressed_tensors_moe_wna16.py:596-640). Could skip that repack.
     Must verify deepseek_v41 + auto_round MoE actually loads on ROCm.
  3. GROUP_SIZE 32 (GLM uses 128) -> 4x the scale bytes per expert. Scales
     ride along in the gather, so this directly inflates the 24 ms that
     gather costs on GLM. Real, measurable, unavoidable without requant.
  4. bf16 exclusion list is LARGE and is the same shape as GLM's problem:
     all vision blocks, all 40 layers' attn (wkv / compressor / indexer),
     shared_experts w1/w2/w3, alignment layers, output head, MTP.
     Literal keys: "layers.3.attn.wkv", "layers.2.attn.compressor.wgate",
     "layers.20.attn.indexer.weights_proj", "mtp.0.ffn.shared_experts.w1",
     "layers.1.engram.wkv" (the engram PROJECTION is bf16; the TABLE is int4)

*** HARD BLOCKER RIGHT NOW: DISK ***
  /mnt/llm-storage: 238 GB free, checkpoint needs 301 GB. Short by ~63 GB.
  Largest items: q38fn-heretic2-bf16 336G | glm53-w4a16-mtp 182G (IN USE) |
  glm53-w4a8-int8 173G  <- the ABANDONED int8 conversion from this session
  (it never loaded: attention is unquantizable by model design). Deleting it
  frees 173 GB -> 411 GB free -> fits. USER DECISION, not mine to make.
  Host RAM: 220 GiB available vs ~170 GiB needed (111 engram + ~58 offload).
  Fits, but tight alongside q38fn-lru's 138 GiB working set.

## ANSWERED: are the experts loaded twice under TP=2? NO (2026-09-15)
tp_duplication.py sums the safetensors headers and compares to what the ranks
actually hold (59.39 device + 27.87 offloaded = 87.26 GiB/rank):
  routed experts (EP-sharded)  146.18 GiB   <- 288 experts, 144 per rank
  MTP (skipped at load)         13.84
  attention (TP)                11.28
  embed / lm_head (vocab-par)    2.36
  shared experts (TP)            1.97
  vision tower                   1.05        <- REPLICATED per rank
  dense MLP (TP)                 0.84
  MoE router gate                0.09        <- REPLICATED (ReplicatedLinear)
  checkpoint 177.69 | loaded excl. MTP 163.85
  perfect sharding per rank 81.92 vs observed 87.26 -> 5.34 GiB "extra"
Of that 5.34, only ~1.2 GiB is replicated WEIGHTS (vision 1.05 + gate 0.09 +
misc 0.07). The remaining ~4.1 GiB is runtime, not tensors: "Model loading
took" is a memory DELTA, so it includes the topk_indices_buffer (sized by
max_num_batched_tokens), quant scratch, tile padding, and the expert-cache
slot table (8 x ~12.4 MB).
=> Experts are sharded, not duplicated. The offloaded 27 GiB/rank is also
   disjoint per rank (each rank offloads its own 144). Nothing to reclaim.
   vLLM has no cross-process VRAM sharing for TP ranks (separate HIP
   contexts); the only shared-memory feature is EngramConfig.dp_shared_memory,
   which shares HOST copies between DP replicas, not VRAM between TP ranks.
FOR V4.1: common/engram.py:563 _engram_head_shard_weight_loader -> the engram
   is HEAD-SHARDED across TP, so 111 GB becomes ~55.5 GB/rank on host, not
   111 per rank. Confirms the host-RAM arithmetic in the plan.

## *** P4 GATE FIRES: the W4A16-Engram checkpoint does NOT fit. STOP. ***
## (and a CORRECTION: I had the checkpoint size wrong by 150 GB)
I quoted 301.4 GB from a WebFetch page summary instead of summing the API
myself. Summing siblings[].size directly: 48 safetensors = 451.7 GB.
Per-shard distribution settles everything:
  shards 01-46  ~340.5 GB   (40 of them are 8.19 GB each)
  shards 47-48  55.61 GB each = 111.2 GB   <- the two engram layers
ENGRAM IS int4 (my second reading was right): 55.61 GB / 98.3e9 params
  = 0.566 bytes/param = 4 bits + a bf16 scale per 32 = 4.5 bits. And
  384,006,168 rows x 256 dims x 2 layers = 196.6B params = the card's "196B".
BACKBONE = 451.7 - 111.2 = 340.5 GB for ~355B params = 0.96 bytes/param
  = ~7.6 bits average. That is NOT int4: AutoRound quantised only the ROUTED
  experts. config.json's extra_config keeps bf16 for every layer's attn
  (wkv / compressor / indexer), shared_experts w1/w2/w3, dense FFN, alignment
  layers, output head, all 32 vision blocks, and MTP.

THE ARITHMETIC THAT KILLS IT
  GPU-resident backbone 340.5 GB = 317 GiB
  per rank at TP=2: 158.5 GiB;  usable VRAM/rank ~58 GiB
  => ~101 GiB/rank OFFLOADED   vs GLM's 27 GiB/rank today  (3.7x worse)
  P4 kill gate was 36 GiB/rank. Fails by a factor of ~3.
  Secondary: 451.7 GB does not fit the 381 GB free either (after the 173 GB
  reclaim), so it cannot even be downloaded as-is.
Engram doing its job (196B on host for ~free) is NOT enough, because the part
Engram does not cover is itself larger than the whole GLM checkpoint.

WHAT WOULD MAKE IT VIABLE
  Backbone must come down from 340 GB to ~180 GB to match GLM's current
  offload pressure -- i.e. the bf16 remainder (attention, shared experts,
  dense FFN, head, vision) needs to be int4/int8. That is a requant project
  with real quality risk, and this session already showed GLM's equivalent
  was blocked by vLLM hardcoding quant_config=None for those modules
  (kda.py:161-167, model.py:325/:1049) -- deepseek_v41 may or may not differ.
  Cheaper: wait for / find a quant that compresses the non-expert weights.
DOWNLOAD ABORTED after 17 MB (configs only). Nothing else spent.
Disk reclaimed from glm53-w4a8-int8 (173 GB, user-approved) stands: 381 GB free.

## Requant feasibility for deepseek_v41: STRUCTURALLY FINE, but IRRELEVANT
Checked the thing I said needed checking before anyone spends days requanting.
UNLIKE GLM, deepseek_v41 does NOT refuse to quantize its attention:
  attention.py:320,329,339,352,416,1093  quant_config=quant_config
  amd/model.py:428                        quant_config=quant_config
Only TWO hardcoded quant_config=None, both trivially small indexer pieces:
  attention.py:1126 weights_proj  (hidden_size -> n_head = 5120 -> 8)
  attention.py:1126 wk            (-> head_dim 128; the comment even says
                                   "checkpoint stores it in bf16 with no
                                    quantization scales")
So a requant that compressed attention WOULD load. Compare GLM, where
kda.py:161-167 nulls quant_config and model.py:325/:1049 hardcode None.

BUT IT BUYS ALMOST NOTHING, because the bf16 remainder is small. Shard
arithmetic, and it reconciles to the byte:
  per expert = 3 x 2304 x 5120 = 35.4M params
             x 384 experts     = 13.59B params per MoE layer
             at int4+group32 (0.5625 B/param) = 7.64 GB per MoE layer
  and there are exactly 40 shards of 8.19 GB  (7.64 GB experts + ~0.55 GB of
  bf16 attention/norms per layer). 40 x 8.19 = 327.6 GB, plus shards
  1,2,43-46 (12.75 GB) = 340.4 GB = the backbone. Exact match.
  => routed experts ~306 GB ALREADY int4
     bf16 remainder  ~34.5 GB  -> requanting it saves ~26 GB
     backbone would go 340.5 -> ~314 GB, against the ~180 GB needed.

THE REAL REASON IT CANNOT FIT, stated properly:
  GLM-5.3-Flash  288 experts x 42 layers, 12.97 MB/expert  = 146 GiB experts
  V4.1-Flash     384 experts x 40 layers, 19.9  MB/expert  = 306 GB experts
  V4.1-Flash carries ~2.1x GLM's expert bulk AT THE SAME BIT WIDTH.
  Engram (196B on host, free) is a real win and it is not nearly enough.
  Only sub-int4 experts would close it -- explicitly ruled out by the user.
CLOSED. No build, no requant, no download. GLM-5.3-Flash stays.

## *** THE BEST LEAD OF THE SESSION: REAP-pruned GLM (2026-09-15) ***
Searched GLM-5.3-Flash variants (200 repos). REAP-pruned ones exist:
  patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2  172.4 GB  144 experts, top-8
  patrickbdevaney/GLM-5.3-Flash-REAP50-FP8     168.5 GB  (v1)
  OpenMOSE/GLM-5.3-Flash-REAP-250B-A18B        264.9 GB  228 experts, fp8
  + NVFP4 / GGUF / MLX variants (all unusable here)
README of the FP8-v2: base zai-org/GLM-5.3-Flash (FP8 E4M3), 288->144 experts,
top-8 routing unchanged, "pruning is deleting whole tensors - lossless on
every retained weight", healed: yes, MTP excluded, EVALUATION STATUS: NONE.

RUNNING REAP50-FP8 DIRECTLY IS A WASH -- do not bother:
  loaded 160.6 GiB vs our current 163.85 GiB  -> still ~27 GiB/rank offloaded
  and fp8 experts are 1.78x bigger per expert than int4, so every gather MISS
  moves 23 MB instead of 12.97 MB. Better hit rate (72 candidates/rank instead
  of 144) roughly cancels it. Net: neutral to worse.

THE PRIZE IS REAP50 AT INT4, AND IT DOES NOT EXIST ON HF:
  expert params 288E int4 = 146.18 GiB  -> 278.9B params
  REAP50 -> 139.5B params -> at int4 (0.5625 B/param) = 73 GiB
  + non-expert 17.67 GiB (unchanged) = ~91 GiB loaded
  -> 45.5 GiB/rank  -> FITS ENTIRELY IN VRAM (58 GiB usable/rank)
  -> ZERO offload -> NO GATHER -> the measured no-gather floor 63.7 ms/step
  -> 15.7 tok/s, i.e. +38% over today's 88.2 ms / 11.3 tok/s.

HOW TO GET IT WITHOUT A REQUANT: apply REAP's mask to OUR int4 checkpoint.
  reap_metadata.json is published but is SUMMARY ONLY (857 bytes: sparsity
  0.5, experts_kept 144, saliency_mass_retained 0.711, routing_mass_retained
  0.497, domain_retention general 0.49 / code 0.73 / agentic 0.75 /
  vision 0.68, calib 5.5M tokens). NO per-layer indices.
  But the mask is RECOVERABLE: the router gate is [n_experts, hidden] and the
  pruned model's 144 gate rows are a SUBSET of the base's 288. Match rows
  (nearest-neighbour; 4096 dims makes it unambiguous even after healing) to
  recover per-layer kept indices. Only the gate tensors are needed --
  ~2.4 MB x 42 layers ~ 100 MB, fetchable by HTTP range from the shards using
  the pruned model's safetensors headers. No need to pull 172 GB.
  Then slice OUR int4 checkpoint: expert weight_packed + scales, gate rows,
  e_score_correction_bias. Pure tensor slicing, no requant, no dequant.

RISK, and it is the real one: QUALITY. REAP50 discards half the experts;
routing_mass_retained is 0.497 and the repo ships NO evaluation. Our copy
would also be UNHEALED (their fp8 release was healed; we would apply the mask
to unhealed int4 weights). This is a different axis from the user's "no lower
than int4" rule and is plausibly more damaging than sub-int4 quantisation.
Needs a real eval gate (the retrieval battery + something harder) before it
could replace the standing server.

## CHECKPOINT AXIS FOR GLM-5.3-FLASH: EXHAUSTED (2026-09-15)
Every published int4 GLM-5.3-Flash, weights only:
  canada-quant W4A16-MTP   190.8 GB (mtp 14.9) -> 175.9 ex-MTP  <- smallest
  Intel W4A16-AutoRound    181.5 GB (no mtp)
  voska W4A16              190.8    (no mtp)
  wtdcode AWQ-W4A16        190.8    (no mtp)
  JANGQ-AI W4A16           194.7    (no mtp)
  Justvugg colibri-int4-g64 194.7   (fp8 metadata)
  cyankiwi AWQ-INT4        197.2    (no mtp)
  OURS (local)             195.1 GB (mtp 14.9) -> 180.2 ex-MTP
Spread is 176-197 GB because they ALL quantise only the routed experts and
leave the same ~17.67 GiB bf16 remainder. Best alternative is ~4.3 GB smaller
than ours ex-MTP = ~2 GiB/rank less offload = ~2-4 ms. Not worth 176 GB of
download. Ours is fine; stop shopping for checkpoints.

WHAT IS ACTUALLY LEFT ON GLM-5.3-FLASH, ranked
 1. REAP50 mask -> our int4 weights: 91 GiB loaded, ZERO offload, 63.7 ms
    (+38%). Quality cost is real and unmeasured. See the entry above.
 2. Patch KDA to accept a quant_config, then requant attention to int8.
    Attention is 11.28 GiB of the 17.67 GiB bf16 remainder; int8 halves it
    (-5.6 GiB -> ~2.8 GiB/rank less offload) AND halves its GEMV read traffic
    (wvSplitK is 7.9 ms/step at ~78% of roofline). Estimate ~8-10 ms total,
    88 -> ~78-80 ms (~10%). Multi-day: this session ALREADY hit the wall here
    (KeyError in_proj_qkvbfg_a.weight_scale) because KDA fuses q/k/v/b/f_a/g_a
    into one projection and kda.py:161-167 nulls quant_config around it. We
    already mount a patched kda.py, so the plumbing is reachable -- it is the
    fused-projection scale handling that needs writing.
 3. Requant experts at group 256 instead of 128: scales 4.43 -> 2.2 GiB,
    saves ~1.1 GiB/rank = ~1.5-2.5 ms. Large job for ~2%. Not worth it.

## CLOSED: does V4.1-Flash touch FEWER expert bytes per token than GLM? NO.
This was the last open item in the V4.1 evaluation ("(b) is UNMEASURED").
  GLM-5.3-Flash          expert  25.2M params = 12.98 MB (group 128)
                         top-8 x 42 layers ->  103.8 MB/layer,  4.36 GB/token
  DeepSeek-V4.1-Flash    expert  35.4M params = 19.91 MB (group  32)
                         top-6 x 40 layers ->  119.4 MB/layer,  4.78 GB/token

  ratio DeepSeek / GLM = 1.10x  -> V4.1 reads 10% MORE per token
  decomposed: fewer experts 6/8 = 0.75x
              bigger experts     = 1.53x
              fewer layers 40/42 = 0.95x
  If V4.1 used group-128 like GLM its expert would be 18.25 MB
  and the ratio would be 1.00x -- so the group-32
  choice alone accounts for essentially the whole regression.
  SECOND STRIKE: the slot cache would also hit LESS often. V4.1 has 384 experts
  (192/rank) competing for the same 8 slots vs GLM's 288 (144/rank).
  => V4.1 is worse on BOTH gather terms: more bytes per miss AND more misses.

## MEASURED AT LAST: the expert-cache hit rate, and the policy does NOT matter
New probe (glm53_fastpath GLM53_CACHE_STATS=N) accumulates the cache's own
device-side n_miss counter and reports once per N refreshes. Eager, the
standing memory config (SLOTS=8, OFFLOAD=27), 300-token essay, both ranks:
  policy=lfu  12600 refreshes | 100800 requested | 30569 / 30360 misses
              -> HIT RATE 69.7% / 69.9%
  policy=lru  12600 refreshes | 100800 requested | 30812 / 31026 misses
              -> HIT RATE 69.4% / 69.2%
CORRECTION: I had been quoting 62% all session. That figure came from
SLOTS=32 / OFFLOAD=45 / eager and was stale. The real number at the shipped
config is ~69.8%.
LFU beats LRU by 0.5 percentage points -- i.e. the REPLACEMENT POLICY IS NOT
THE LEVER. With SLOTS=8 and top_k=8 the cache has no headroom for a policy to
be clever with, and the ledger's earlier note stands: under EP each rank owns
only ~4 of the 8 routed experts, so 8 slots already exceed the per-step
working set. The residual 30% are COMPULSORY misses -- new experts each step,
which is what near-uniform routing (routing_mass_retained 0.497) predicts.
=> Nothing to win by tuning policy or decay. The cache is doing its job.
   The only way to remove the remaining gather is to stop offloading at all,
   which is exactly what the REAP-144E-int4 idea would achieve.

## *** CORRECTION: the cache hit rate is ~35%, NOT 69.8% (and not 62%) ***
My GLM53_CACHE_STATS hook was WRONG and I have removed it. Under expert
parallelism generic.py does:
    local = emap[topk_ids]            # global -> local, -1 where not owned
    cache.refresh(local.clamp(min=0)) # -1 CLAMPED TO 0, not masked
so of the 8 ids the manager sees, only ~4 are experts this rank owns; the
other ~4 are collapsed onto local expert 0. My hook used topk_ids.numel()=8
as the denominator, counting those ~4 phantoms as guaranteed hits.
The repo ALREADY ships the correct reporter (EXPERT_CACHE_STATS=<interval>),
which counts `(owned_ids >= 0).sum()`. Measured, eager, standing config:
    rank 0: 136 routed, 93 misses -> 31.6% hit
    rank 1: 104 routed, 64 misses -> 38.5% hit
    combined 240 routed / 157 misses -> ~35% HIT, ~65% MISS
This reconciles with the raw counter: 2.43 misses per refresh / ~4 owned = 61%
miss. Every "62%" and "69.8%" earlier in this ledger is retracted.

## CONSEQUENCE: the clamp-to-0 is a real defect worth fixing
Feeding ~4 phantom requests for local expert 0 on EVERY refresh means:
  * expert 0 is permanently the hottest entry under LFU -> it pins one of the
    8 slots forever, leaving 7 for ~4 real experts;
  * LFU's frequency counts are polluted -- one entry accrues ~4 counts per
    refresh while genuine experts accrue ~1, so the ranking LFU sorts on is
    mostly noise. That is why lfu beat lru by only 0.5 pp: neither policy has
    usable statistics to work with.
The repo's own comment acknowledges it ("Clamping costs a little LFU bias
toward local expert 0") but the cost is larger than "a little" at top_k=8 with
half the experts non-owned.
FIX (in the user's repo, kernels/): let the manager skip negative ids instead
of clamping -- pass `local` through unchanged and add an `if (id < 0) continue;`
in expert_cache_manage, or clamp to >= E and bounds-check there.
UPSIDE: 8 clean slots for ~4 real experts is 2x headroom. If the hit rate moved
35% -> 50%, misses drop ~30%, gather 24 ms -> ~17 ms, step 88 -> ~81 ms (~8%).
Not guaranteed -- near-uniform routing may cap it -- but this is a defect with
a measurable fix, which is better than anything else left on the cache side.

## CLAMP FIX: shipped, correct, and MUCH smaller than I predicted
THE KERNEL NEVER NEEDED THE CLAMP. expert_cache.hip:192 documents its ids as
"<0 = padding" and every read is guarded (`if (e >= 0 && e < E)`, lines 224,
464, 562). So the fix is one line of Python, no kernel rebuild:
  backends/generic.py:  cache.refresh(local.clamp(min=0))  ->  cache.refresh(local)
(+ the comment above it rewritten to explain why). Backup: /tmp/generic.py.bak.

MEASURED, clean matched A/B -- same prompt, greedy, eager, EXPERT_CACHE_STATS=100,
and the routing is deterministic so both arms sampled exactly 1008 routed experts:
  clamp (before) : 582 misses / 1008 -> 42.3% hit
  no clamp (after): 566 misses / 1008 -> 43.8% hit
  = 16 fewer misses, 2.7% of all misses, +1.6 pp hit rate
  projected step: gather 24 -> 23.3 ms, 88.1 -> ~87.4 ms (~0.7 ms)
That is BELOW this box's noise floor; it cannot be confirmed in wall clock.

WHY MY 8% ESTIMATE WAS WRONG: the phantom requests all collapse onto local
expert 0, so they pin exactly ONE slot. Removing them takes usable slots from
7 to 8 (+14% capacity), not from "polluted" to "clean". With ~4 owned experts
per step and near-uniform routing, that marginal slot is rarely reused.
ALSO CORRECTING MYSELF AGAIN: the baseline hit rate is ~42%, not the 35% I
reported last message -- that came from a 240-request sample. 1008 requests
says 42.3%. The session's hit-rate figures, in order of increasing trust:
62% (stale config) -> 69.8% (my broken denominator) -> 35% (undersampled)
-> 42.3% (clean). Use 42%.

KEPT ANYWAY: strictly better, costs nothing, no rebuild, and it makes the LFU
counts honest (which is what made lfu-vs-lru a fair test in the first place).
It is hygiene, not a speedup. Correctness verified under cudagraphs: the full
battery and the vision probe are byte-identical to the pre-fix standing server.
UNCOMMITTED, per the standing rule.

## CACHE HIT-RATE SWEEP (2026-09-15). Greedy + deterministic routing, so every
## arm sampled exactly 1008 routed experts -- differences are purely the cache.
Eager, EXPERT_CACHE_STATS=100, same 300-token essay, clamp fix in place.
  SLOTS=8  OFFLOAD=27:
    lfu decay 256 (base)   566 misses   43.8% hit
    lfu decay 0            569          43.6%
    lfu decay 64           567          43.8%
    lfu decay 1024         569          43.6%
    lru decay 256          560          44.4%
  slots (offload raised to pay for the slot VRAM, 521 MB/rank/slot):
    S12 lfu off29          507          49.7%
    S16 lfu off31          496          50.8%

(2) DECAY IS A DEAD KNOB ON REAL ROUTING. 566/569/567/569 across a 16x range.
    This closes the launcher's own open question at line 127: DECAY=256 was
    tuned on a SYNTHETIC ZIPF TRACE and flagged "provisional until checked
    against GLM's real routing". Checked: it does not matter. With 8 slots
    turning over every ~2 steps an LFU count never grows enough to age.
(3) POLICY REVERSED once the counts became honest: lru 560 vs lfu 566 (+0.6pp
    for LRU). Pre-clamp-fix it was lfu ahead by 0.5pp -- i.e. that earlier
    result was an artefact of the phantom counts. Recency beats frequency here,
    which is what a 2-step retention window predicts. Still small.
(1) SLOTS IS THE ONLY REAL LEVER: 43.8 -> 49.7 -> 50.8% for S8/S12/S16,
    misses 566 -> 507 -> 496 (-12.4% at S16). CONFIRMS the temporal-locality
    reading: retention window, not policy, is what the cache is short of.
    AND IT CONFIRMS ROUTING IS *NOT* UNIFORM within a generation -- uniform
    routing over 144 experts would predict ~5% hit at a 2-step window; we
    measure 44%. The REAP metadata's routing_mass_retained=0.497 is an
    AGGREGATE statistic and says nothing about temporal locality. My earlier
    "misses are compulsory" conclusion was wrong.
THE CATCH: slots are copies and compete 1:1 with residency. Modelling PCIe
misses as (misses x offload_fraction): S8 566x0.370=209, S12 507x0.397=201,
S16 496x0.425=211 -- a wash. Wall-clock arms running to settle it.

## *** CACHE POLICY: CLOSED BY SIMULATION AGAINST BELADY (2026-09-15) ***
Captured GLM's REAL per-layer routing (GLM53_TRACE_DUMP, 12,600 refreshes,
42 layers, 50,594 owned-expert requests, per-rank files) and replayed every
policy offline (cache_sim.py), including Belady's clairvoyant optimum, which
bounds what ANY online policy could do on this trace.
   S      LRU     LFU     ARC    LIRS    LRFU     OPT    best   OPT    gap
   8    31445   29110   31152   31039   28675   22213   43.3%  56.1%  12.8pp
  16    24520   23597   23854   23990   22548   15576   55.4%  69.2%  13.8pp
  32    17174   16961   16837   16830   15977    9976   68.4%  80.3%  11.9pp
FINDINGS
 * LRFU (recency/frequency hybrid, lam=0.05) is the best online policy at every
   size -- but beats the LFU we already run by 435 of 29,110 misses = 1.5%,
   worth ~0.4 ms/step. Under the noise floor. NOT worth an on-device kernel.
 * ARC and LIRS are NO BETTER THAN LRU (31,152 / 31,039 vs 31,445). LIRS is the
   textbook answer to the LRU-OPT gap and does nothing here: this workload is
   not the scan-resistant hot/cold pattern it targets.
 * The gap to Belady is ~13 pp at EVERY cache size and for EVERY policy. The
   online policies cluster within 2% of each other and all sit ~30% above
   optimal. That gap needs the future, and in MoE decode step t+1's routing
   does not exist until step t's hidden state is computed. Only an MTP draft
   layer would give lookahead -- and MTP is off the table.
 * SIMULATOR VALIDATED: simulated LFU at S=8 gives 42.5% hit vs 43.8% measured
   on hardware -- the small delta is my eviction tie-break vs the kernel's exact
   victim rule.
 * CORRECTION (again): the hardware A/B that showed "LRU beats LFU" was a
   1008-request sample. On all 50,594 requests LFU beats LRU by 4.7 pp. LFU is
   the correct default; POLICY=lfu stays.
=> The cache is within 1.5% of the best achievable online policy. This whole
   line (policy, decay, slots, algorithms) is now closed by measurement.

## LOSSLESS COMPRESSION OF THE GATHER STREAM -- what is proven, what is not
QUESTION: can we ship fewer PCIe bytes losslessly and get tok/s back?

PROVEN (offline, on the real checkpoint):
  nibble entropy 3.4823 of 4 bits -> 12.94% ceiling for ANY lossless coder
  static Huffman, whole stream    -> 12.29% (95% of ceiling)
  block-parallel, offsets counted -> 64B:5.36% 256B:10.56% 1KB:11.86% 4KB:12.18%
  1 KB blocks give 11.86% net with 12,666 independent decode streams per expert,
  which is ample GPU parallelism. 16 symbols = a single LDS LUT, not rANS.

PROVEN (isolated, rp/uva_scaling.py): host->device read time IS byte-proportional.
  64 DISTINCT 12.97 MB slabs from an 830 MB pinned source (so L2 cannot serve):
     1.00 -> 0.627 ms  20.69 GB/s   100.0% of full time
     0.88 -> 0.512 ms  22.31 GB/s    81.6%
     0.75 -> 0.458 ms  21.22 GB/s    73.1%
     0.50 -> 0.304 ms  21.30 GB/s    48.6%
  Flat GB/s => ~12% fewer bytes ~= ~12% less gather ~= ~3 ms/step.
TWO FAILED ATTEMPTS AT THIS, both my design error, both recorded so nobody repeats them:
  1. EXPERT_CACHE_GATHER_FRAC (copy a fraction of each slab in-model): INVALID.
     Corrupting weights changes the OUTPUT, which changes the ROUTING, which
     changes the MISS COUNT. frac=0.88 "gained" 47% -- that is routing collapse
     (degenerate repetitive output -> few distinct experts -> cache hits
     everything), not bandwidth. Reverted; cache.py restored from /tmp/cache.py.bak.
  2. First uva_scaling: re-read ONE buffer 20x, so MI210's 8 MB L2 served the
     smaller copies. GB/s appeared to RISE as the copy shrank (14.4 -> 24.9).
     Fixed by reading 64 distinct slabs.
=> No in-model shortcut exists. Any test that ships fewer bytes corrupts weights
   and perturbs routing. Compression can only be confirmed by BUILDING it.

## GATHER GEOMETRY: FLAT. My occupancy theory FALSIFIED (2026-09-15)
Hypothesis: grid is (CHUNKS, LANES) and LANES indexes MISSES, so at ~2.4 misses
per call ~62 of 64 lane-blocks idle -> under-occupied -> that explains the
5.8 GB/s in docs/DMA_GATHER.md vs the 20.7 GB/s an isolated copy achieves.
Measured, cudagraphs, tok/s directly:
  chunks=16   lanes=64   87.44 ms  11.44 tok/s   <- default
  chunks=64   lanes=64   86.66     11.54
  chunks=256  lanes=64   88.79     11.27
  chunks=256  lanes=8    90.61     11.04
  chunks=1024 lanes=4    86.63     11.54
  chunks=16   lanes=64   87.16     11.47         <- repeat baseline, drift 0.3 ms
Spread 86.6-90.6 ms with the baseline repeating to 0.3 ms: NO geometry effect.
WHY THE THEORY WAS WRONG: the idle lane-blocks cost nothing -- they compare their
miss index against n_miss and retire immediately. And adding chunks does not help
either, so the limit is not thread count; it is PCIe request concurrency in
hardware, which more blocks cannot raise.
This independently CONFIRMS the ledger's earlier "CHUNKS/LANES flat" result,
which I had doubted because it was measured at a different config.
KEEP THE DEFAULTS (chunks=16, lanes=64). Nothing to win here.

## *** HUFFMAN GATHER DECODER: BUILT, BIT-EXACT, AND IT DOES NOT PAY ***
Built the whole thing rather than argue about it: rp/huff_gather.py (canonical
Huffman over the 16 nibble symbols, MAXLEN=11 so the decode LUT is 2048 entries,
1 KB independently-decodable blocks + int32 offset table, Triton kernel where one
program decodes LANES blocks with one block per lane) and rp/huff_test.py.

CORRECTNESS: BIT-EXACT round trip on real expert weights, both single-phase and
two-phase. Ratio 0.8781 = 12.19% saving, +0.39% offsets -> 11.8% net, matching
the offline prediction exactly. The idea is sound; the economics are not.

MEASURED (2.1 MB of layers.0 down_proj.weight_packed):
  plain copy (today)                0.083 ms  25.14 GB/s   100.0%
  phase 1: copy compressed (PCIe)   0.085 ms  24.59 GB/s   102.2%
  phase 2: decode from VRAM l=64    1.417 ms   1.48 GB/s  1800.7% total
                            l=256   2.097                 2616.1%
                            l=1024 10.949                13227.1%
DECODE COSTS 1.417 ms TO SAVE AT MOST 0.010 ms OF TRANSFER. Off by ~140x.

WHY IT IS STRUCTURAL, NOT AN IMPLEMENTATION BUG:
  copying 13 MB          = ~800K vectorised 16-byte ops
  Huffman-decoding 13 MB = ~26M SERIAL symbol decodes -- each symbol's bit
                           position depends on the previous symbol's LENGTH, so
                           there is a dependent chain per stream that parallelism
                           cannot remove. 30x more work items, against a transfer
                           already running at 21-25 GB/s.
  My kernel is admittedly unoptimised (each lane does scattered per-symbol byte
  loads instead of staging its block in LDS first). Fixing that might recover
  5-10x. It needs 140x.

TWO SECONDARY BLOCKERS, recorded so they are not rediscovered:
  * phase 1 measured 102.2% (no gain) because a 2.1 MB buffer re-read 30x lives
    in MI210's 8 MB L2 -- the SAME contamination as the first uva_scaling run.
    The trustworthy transfer numbers remain the 12.97 MB / 64-distinct-slab ones.
  * the host encoder runs at 1.3 MB/s in pure Python; 73 GiB/rank would take
    ~16 hours without a vectorised rewrite.

=> CLOSED. The 12.94% of redundancy is real and is not worth what it costs to
   exploit on this hardware. The gather is already fast enough that entropy
   decoding costs more than the bytes it saves.
   Artifacts kept under rp/ (huff_gather.py, huff_test.py) -- they are correct
   and reusable if a future card has a hardware decompressor in the DMA path.

================================================================================
KERNEL FUSION OF THE SMALL-KERNEL BLOCK  ("#1")           2026-09-16
================================================================================
Premise from the eager rocprofv3 trace of the standing config (fastpath gate,topk),
rank 387, one warm steady-state step, analyser rp_step2.py:

  2,445 kernels/step
  1,726 of them under 20 us each, 11,108.6 us busy = 6.8% of eager step busy
  _gate_gemv_kernel           84 calls    632.2 us   <- 84 for 42 layers
  _topk_sigmoid_bias_kernel   42 calls  1,045.3 us   <- 24.9 us each
  index_elementwise          129 calls    552.8 us
  vectorized_elementwise     548 calls  3,020.0 us
  elementwise_manual_unroll  234 calls  1,627.1 us
  moe_align_block_size        42 calls    432.0 us
  expert_cache (mgr+gather)  126 calls 25,113.0 us   <- the PCIe cost, untouched here

THREE FUSIONS IMPLEMENTED (all gated, so they can be A/B'd in one build):

 1. duplicate gate           GLM53_FASTPATH_PARTS=...,nodoublegate
    The model computes router_logits and passes them to the MoE runner; the runner
    then calls the SAME gate module on the SAME hidden states and discards the value
    it was handed (moe_runner.py `if self.gate is not None: router_logits = self.gate(...)`).
    42 redundant GEMVs per step. Fixed by nulling the runner's gate for the call
    whenever the caller supplied logits of the matching shape.

 2. top-k register store     GLM53_TOPK_REGSTORE=1 (default)
    _topk_sigmoid_bias_kernel stored each selected id inside the 8-iteration
    static_range loop: 8 dependent scalar stores per call. Ids now accumulate in
    registers like the weights already did, and store once.

 3. fused EP remap           EXPERT_CACHE_FUSED_REMAP=1 (default)
    vllm_expert_cache/fused_remap.py (new) + backends/generic.py.
    The EP path ran four torch ops per layer per step on 8- and 288-element tensors:
    cast + gather for local ids, index_select + masked_fill to compose the slot table
    into expert_map. Now two Triton kernels into persistent buffers. 168 -> 84 launches.

CORRECTNESS: qprobe greedy battery passes on every arm (4 / 391 / Paris / the sea
sentence / 2,3,5,7,11). Nothing committed or pushed.

MEASUREMENT PROBLEM FOUND, and it invalidates single-container A/B at this scale:
  Every arm needs its own container, and the 27 GB host offload buffer lands in
  different physical memory on each start. Repeating the SAME arm across restarts:

      arm=on,  3 containers, per-container means:  91.80  97.71  92.81 ms/step
      arm=off, 3 containers, per-container means:  93.53  94.33  89.75 ms/step

  Between-container spread is 5.9 ms (on) and 4.6 ms (off). Within one container
  the spread can be as low as 0.13 ms, which is what made a single container look
  conclusive. It is not. RETRACTED: an earlier single-container pair read
  "87.45 on vs 93.5 off = 6 ms win". That was container placement, not the change.
  Any future decode claim below ~6 ms needs interleaved restarts (fusion_ab.sh).

WALL-CLOCK VERDICT (3 interleaved restarts per arm, 12 samples each):
      fusions ON  mean 94.11 ms  (88.46-99.67, stdev 3.69)
      fusions OFF mean 92.54 ms  (89.69-96.91, stdev 2.44)
      difference -1.57 ms in favour of OFF, pooled stdev 3.13 ms
  => NOT RESOLVABLE. No measurable gain, and no measurable regression either.

KERNEL VERDICT (eager rocprofv3, same analyser, same step delimitation; the
per-kernel call counts are exact multiples of the 42 MoE layers in both traces,
so the step boundaries are right in both):

     kernel                      before            after          delta
     total kernels/step           2,445            2,319           -126
     _gate_gemv_kernel          84 /  632.2 us   42 / 346.7 us   -42, -285 us
     _topk_sigmoid_bias         42 / 1045.3      42 / 945.6       -100 us
     index_elementwise         129 /  552.8      87 / 344.8      -42, -208 us
     vectorized_elementwise    548 / 3020.0     506 / 2740.8     -42, -279 us
     elementwise_manual_unroll 234 / 1627.1     192 / 1369.6     -42, -258 us
     moe_align_block_size       42 /  432.0      42 / 437.8       unchanged
     expert_cache (mgr+gather) 126 /25113.0     126 /26638.5      unchanged (PCIe)

  The code does exactly what it was written to do: 126 fewer kernels and ~1.13 ms
  less GPU busy time per step. Do NOT quote the traces' total busy figures against
  each other (163.6 vs 73.6 ms) -- that difference is the eager MoE GEMM varying
  with cache warmth between runs, not the fusion.

CORRECTNESS: remap_test.py proves the two new Triton kernels bit-match the torch
  expressions they replace on decode M=1, prefill M=64, nothing-owned, all-owned
  with an empty table, and a fully resident table. The last two are the cases that
  would corrupt SILENTLY (wrong slot -> plausible text from the wrong weights).
  qprobe greedy battery passed on all six A/B containers.

WHY THE "~7 ms" PROJECTION WAS WRONG -- the estimate assumed halving ~1,600 small
kernels and recovering ~5 ms of cudagraph replay gaps as well. Both legs were bad:
  * only 126 of those kernels were removable without rewriting the model's own
    pointwise chains (mHC, norms, residuals, the sparse indexer). The remaining
    ~500 vectorized_elementwise + ~190 manual_unroll are model-level ops, and
    torch.compile -- the tool built for exactly that fusion -- was already measured
    and LOST on this model.
  * the eager trace shows only 110 us of inter-kernel gaps in a whole step. There
    is no dispatch headroom on this path to recover; the GPU is not launch-bound.

=> #1 CLOSED. Implemented, correct, less work done per step, no measurable speed-up.
   Kept on by default (strictly less work, verified equal output). Turn off with
   GLM53_TOPK_REGSTORE=0, EXPERT_CACHE_FUSED_REMAP=0, and dropping `nodoublegate`
   from GLM53_FASTPATH_PARTS.

NOT PURSUED, recorded so it is not rediscovered: expert_cache_fused_k is built and
exported (kernels/build.sh:36) and folds both moe_align_block_size outputs into the
manager, 6 launches -> 2 per layer. It is unused because it targets a hot/cold GEMM
split (`_gemm_split`) that does not exist in this branch; the repo's own porting docs
say to skip it. Wiring it means implementing that split, and its prize is the 42
moe_align calls at 437.8 us -- i.e. inside the noise floor measured above.

================================================================================
EXPERT-CACHE OPERATING POINT: 10.96 -> 12.16 tok/s (+10.9%)     2026-09-16
================================================================================
THE BUG THAT MADE THIS POSSIBLE. The cache armed every MoE layer, including the
ones whose experts never left the GPU. vLLM's UVA offloader is whole-layer and
budget-limited, so at --cpu-offload-gb 28 only ~17 of 42 layers were on the host;
the other 25 were fully GPU-resident and carried an 8-slot buffer they could never
miss into, plus a manager and a gather every step copying VRAM to itself.

  offloaded per rank      29.07 GiB
  one MoE layer of experts 1.74 GiB/rank (144 EP-local experts x 12.38 MiB)
  => 16.7 layers offloaded, 25.3 resident   -- confirmed exactly by the fix:
     "active on 17 MoE layer(s) (25 declined)"
  dead slot VRAM          25 x 8 x 12.38 MiB = 2.42 GiB/rank

FIX: backends/generic.py arm() now declines a layer unless its expert weights are
really on the host. vLLM's UVA path leaves `.device` reading as the accelerator
(the parameter becomes a device-addressable VIEW of pinned host memory), so the
`_vllm_is_uva_offloaded` marker is the only reliable signal; device is consulted
only for the non-UVA path. The decline is logged, not silent, because which layers
the offloader reached depends on --cpu-offload-gb.

HIT RATE vs SLOTS, measured on 12,600 real refreshes over 17 layers (51,114
owned-expert requests), rp/route.rank382.json via cache_sim.py:

     S     best online hit    OPT hit    misses/layer-step   PCIe MiB/step
     8          33.4%          47.2%          2.704           569  <- was here
    16          45.6%          61.5%          2.207           464
    24          53.2%          69.8%          1.900           400
    32          59.2%          75.5%          1.655           348
    48          68.8%          82.9%          1.267           266
    64          76.1%          87.7%          0.971           204
    96          87.3%          93.0%          0.518           109

At ~24 GB/s the old 569 MiB/step was ~24 ms of a ~91 ms step. That is the cost
the slot count actually buys down.

VRAM CEILING FOUND EMPIRICALLY (expert VRAM = resident layers + slot buffers):
     43.5 + 3.29 = 46.8 GiB  boots        (off 28, S=16)
     43.5 + 4.93 = 48.4 GiB  DIES         (off 28, S=24, and S=32) -- arms fine,
                                           then dies later in cudagraph capture
  => ceiling ~47 GiB. So raising slots REQUIRES converting resident layers into
     offloaded ones; at a fixed offload budget the slot count is capped near 16.

NEW STANDING CONFIG: OFFLOAD_GB=45, SLOTS=56 -> 26 layers armed, 16 declined,
56/144 experts resident per armed layer. Interleaved A/B, 3 container restarts
per arm, arms alternated:

     arm                       n    mean     min     max    tok/s
     off 45, 56 slots         12   82.23   77.98   87.78   12.16
     off 28,  8 slots         12   91.22   84.58   95.12   10.96
     per-container means big : 80.60  78.88  87.22
     per-container means base: 84.76  94.81  94.09
     => big wins all 3 paired restarts; mean -8.99 ms/step, +10.9% tok/s.
     (The two distributions touch at the edges, which is why the PAIRED result
      matters -- 3/3 -- rather than the raw min/max.)

WHY NOT PUSH FURTHER. Bytes/step = L x m(S) x 12.38 MiB under
(42-L) x 1.74 + L x S x 0.012088 <= 47 GiB. Offloading more layers raises the
affordable S but also multiplies the miss count by more layers, so it flattens:
     L=26, S<=61  ->  328 MiB/step   (pinned host  90 GiB)
     L=34, S<=80  ->  307 MiB/step   (pinned host 120 GiB)
     L=42, S<=92  ->  294 MiB/step   (pinned host 152 GiB)
A 10% further cut in PCIe bytes costs 62 GiB more PINNED (unreclaimable) host
memory on a box that already has 12 GB free, 6 GB in swap, and the PROTECTED
q38fn-lru holding 138 GiB. Not taken.

CUDAGRAPH FACT worth remembering: under FULL_DECODE_ONLY the Python body of the
cache's apply() runs only at CAPTURE, not per step. That is why EXPERT_CACHE_STATS
prints nothing in a graph run (its .item() never re-executes), and it is the reason
the kernel-fusion work above moved nothing -- host-side savings are free already,
and only GPU busy time counts.

THE COST: PREFILL. Prefill bypasses the cache entirely on any step wider than
SLOTS (tokens x top_k >> any feasible slot count), so every offloaded layer is
read from host at full width. More offloaded layers therefore means slower
prefill, and the two directions trade directly. prefill_bench2.py, unique prompts:

     prompt tok   off28/S8   off37/S36   off45/S56
        500        138.9       110.2        81.0
       1000        216.4       144.3       125.1
       2000        316.8       227.8       185.2
       4000        264.9       213.2       182.8
       7000        183.7       190.5       156.7
     decode tok/s   10.96      ~11.1*      12.16     (* one container only)

BREAK-EVEN, off45/S56 versus off28/S8. Decode saves 0.0090 s per output token;
prefill costs the difference above. The new point wins on TOTAL latency once the
output exceeds:
     prompt  500 -> ~292 output tokens      prompt 4000 -> ~777
     prompt 1000 -> ~375                    prompt 7000 -> ~745
     prompt 2000 -> ~516
A reasoning model emitting chain-of-thought clears that on most turns, which is
why this was the recommendation.

DECISION (owner, 2026-09-16): run off45/S56. fp_launch.sh defaults updated to
OFFLOAD_GB=45 SLOTS=56. The middle point off37/S36 was measured and sits ON the
line between the other two rather than beating both, so there is no free lunch
here -- it is a straight decode/prefill dial.

The 60 GiB corner was NOT taken: projected decode gain is inside the noise floor
and it would add 30 GiB of unreclaimable pinned memory to a host where the
PROTECTED q38fn-lru holds 138 GiB and 12 GB is free with 6 GB already swapped.

VERIFIED on the final config: greedy battery (4 / 391 / Paris / sea sentence /
2,3,5,7,11), vision probe (red square), health 200, 26 layers armed / 16 declined
/ 56 of 144 resident. Nothing committed, nothing pushed.

================================================================================
PREFILL DIAGNOSIS: the bypass path reads host memory ~3.5x slower than the
gather kernel does                                              2026-09-16
================================================================================
Prefill bypasses the expert cache on any step wider than SLOTS (cache.py:115,
`topk_ids.numel() <= self.S`). A 2048-token chunk at top_k=8 is 16,384 routing
entries, so every prefill step bypasses, and the fused MoE GEMM reads expert
weights DIRECTLY from the UVA host view over PCIe.

COST IS LINEAR IN OFFLOADED LAYERS. Same 2070-token prompt, three offload levels:
     17 offloaded layers   6,535 ms
     22 offloaded layers   9,089 ms
     26 offloaded layers  11,174 ms
     => 515 ms per offloaded layer, consistent across both intervals
        (2554/5 = 510.8 and 2085/4 = 521.3)

WHAT THAT RATE IMPLIES. A 2070-token prompt is 2 chunks at the default chunk
size, and a chunk that wide routes to essentially all 144 EP-local experts, so
each offloaded layer ships 2 x 144 x 12.38 MiB = 3.48 GiB.
     3.48 GiB / 0.515 s = ~6.8 GB/s
The SAME link, read by expert_cache_gather_k, measures 21-25 GB/s
(rp/uva_scaling.py, 64 distinct slabs so nothing is served from the 8 MB L2).
=> the bypass path is running at roughly 3.5x below what this PCIe link gives a
   kernel that streams it properly. That gap, not the link, is the prefill wall.

WHY: the fused MoE GEMM's access pattern is built for HBM -- tiled, with reuse
across a tile -- and it is reading host memory through a device-addressable view.
The gather kernel instead does wide contiguous 16-byte-per-lane copies, which is
what a PCIe link wants.

TWO LEVERS, one cheap and one structural:
  (a) CHUNK SIZE. Each chunk pays a full expert sweep regardless of how full it
      is, so cost scales with ceil(N / chunk). Raising --max-num-batched-tokens
      cuts the number of sweeps directly. This is why short prompts have the
      WORST tok/s: a 512-token prefill pays the same ~45 GiB sweep across 26
      layers that a 2048-token one does, amortised over a quarter of the tokens.
  (b) DO NOT BYPASS. For a wide step, walk the experts in blocks of SLOTS:
      gather a block with the fast kernel, run the MoE restricted to that block,
      accumulate. 144 experts at S=56 is 3 blocks per layer. That converts the
      expert read from ~7 GB/s to ~24 GB/s -- the full 3.5x -- at the price of
      implementing blocked MoE execution in the plugin.

SLOTS AND CHUNK COMPETE FOR THE SAME VRAM. --max-num-batched-tokens 8192 with
SLOTS=56 dies: "CUDA out of memory. Tried to allocate 1.27 GiB ... 670.00 MiB is
free". The slot buffers took the activation headroom. So (a) is not free; it is
another position on the same VRAM dial that slots sit on.

LEVER (a) MEASURED: --max-num-batched-tokens 4096 on off45/S56.
Each chunk pays a full expert sweep regardless of occupancy, so halving the
chunk count nearly halves the sweeps. Prefill, unique prompts (prefill_bench2):

     prompt   chunk default   chunk 4096   change
        500        81.0          99.6       +23%
       1000       125.1         152.2       +22%
       2000       185.2         221.2       +19%
       4000       182.8         220.2       +20%
       7000       156.7         206.2       +32%
     decode        12.16         12.86*     (* one container; chunk size cannot
                                              affect a 1-token decode step, so
                                              read this as "not worse")

At 7000 tokens chunk 4096 now BEATS the old off28/S8 config (206.2 vs 183.7),
while decode stays at the new, higher level. Below ~5000 tokens the old config
is still ahead on prefill.

--max-num-batched-tokens 8192 does NOT fit at SLOTS=56: "CUDA out of memory.
Tried to allocate 1.27 GiB ... 670.00 MiB is free." Chunk size and slot count are
two positions on the SAME VRAM dial.

CHUNK 6144 (needed SLOTS trimmed 56 -> 50 to pay for activation memory):
     prompt   4096/S56   6144/S50
        500      99.6       99.0
       1000     152.2      152.0
       2000     221.2      222.2
       4000     220.2      220.3
       7000     206.2      229.5   <- the only prompt where 6144 helps (+11%)
     decode      12.86      12.53   (-2.6%, consistent with 6 fewer slots)

NOTE the 4000 case: 4122 tokens is 2 chunks at 4096 and 1 chunk at 6144, yet the
time is identical (18723 vs 18710 ms). So "each chunk pays a full expert sweep"
is too crude. The real cost tracks the DISTINCT experts each chunk routes to, and
a 26-token tail chunk touches far fewer than a 4096-token one. Bigger chunks help
only by shrinking the TAIL, which is why the gain shows up at 7151 tokens (tail
3055 -> 1007) and nowhere else.

=> STANDING: --max-num-batched-tokens 4096, OFFLOAD_GB=45, SLOTS=56. Best decode,
   and prefill ties 6144 everywhere except a 7k prompt. fp_launch.sh defaults
   updated. If long prompts ever dominate the workload, 6144/S50 trades 2.6% of
   decode for 11% of prefill at 7k.

WHERE PREFILL STANDS versus where this session started (off28/S8, default chunk):
     prompt   was     now    delta
        500   138.9    99.6   -28%
       1000   216.4   152.2   -30%
       2000   316.8   221.2   -30%
       4000   264.9   220.2   -17%
       7000   183.7   206.2   +12%
     decode    10.96   12.86  +17%
Prefill is still down on short and mid prompts. The remaining 3.5x is lever (b),
NOT YET IMPLEMENTED and awaiting owner go-ahead.

LEVER (b) IS SIMPLER THAN FIRST THOUGHT. expert_cache_gather_k already takes the
miss list as (expert, slot) PAIRS (kernels/expert_cache.hip: `e = miss[2*j]`,
`sl = miss[2*j+1]`). So a full-size [144, ...] scratch buffer with an IDENTITY
pair list needs no kernel change, and because the scratch is indexed by local
expert id exactly as the original tensor is, expert_map and global_num_experts
stay untouched. That makes it a plain copy-to-VRAM-first, not the blocked
partial-sum scheme -- which removes most of the silent-corruption risk.
  cost:  1.74 GiB/rank of scratch (one buffer, reused across all 26 layers)
         => SLOTS 56 -> ~46, about -0.35 tok/s of decode
  prize: the prefill expert read moves from ~7 GB/s to ~24 GB/s
  size:  ~80 lines in backends/generic.py, in the single bypass branch at :341

================================================================================
LEVER (b) IMPLEMENTED: wide-step VRAM scratch -- prefill 3x       2026-09-16
================================================================================
WHAT IT DOES. On a step too wide to serve from slots (prefill), stage the whole
local expert set into a VRAM buffer with expert_cache_gather_k, then run the MoE
from there, instead of letting the GEMM pick at pinned host memory.

WHY IT IS SAFE. The scratch is full-size [E, ...] and indexed BY LOCAL EXPERT ID,
exactly as the source tensor is. So expert_map, global_num_experts and the routing
ids all keep their existing meaning and NOTHING is remapped -- it is purely a
change of where the weights live for the duration of the call. This is what makes
it different from (and much safer than) the blocked partial-sum scheme originally
sketched.

NO KERNEL CHANGE NEEDED. expert_cache_gather_k already reads its work list as
(expert, slot) pairs (`e = miss[2*j]`, `sl = miss[2*j+1]`), so an identity list
[0,0, 1,1, 2,2, ...] copies expert e to row e.

ONE ALLOCATION FOR ALL LAYERS. Layers run in order on one stream, so layer N's
GEMM is enqueued before layer N+1's fill and the refill cannot overtake a read.
144 experts x 12.38 MiB = 1.74 GiB once, not per layer.

FILES (uncommitted):
  cache.py            + class WideScratch (acquire/fill, shared by geometry)
  backends/generic.py + _bound_to(), scratch + wide_cfg built at arm time,
                        bypass branch at :341 now fills-binds-computes-restores.
                        The _leaked_full_tensors check is deliberately NOT run
                        against wide_cfg: full-size storage is CORRECT there.
  config.py           + EXPERT_CACHE_WIDE_SCRATCH (default on)

RESULT, off45 / SLOTS=46 (46 not 56: the scratch costs 1.74 GiB of the same VRAM):
     prompt   chunk4096/S56   +scratch/S46    vs session start (off28/S8)
        500        99.6           233.0        138.9   -> +68%
       1000       152.2           413.7        216.4   -> +91%
       2000       221.2           699.5        316.8   -> +121%
       4000       220.2           675.1        264.9   -> +155%
       7000       206.2           603.3        183.7   -> +228%
     decode       12.86           11.74*       10.96   -> +7%
  (* one container; S46 vs S56 explains the direction, size not yet separated
     from the +-0.9 tok/s container noise)

Prefill is now 2.3x-3.2x the previous config and 1.7x-3.3x the session start, and
it is FASTER than the session start at every prompt length rather than only at 7k.

--------------------------------------------------------------------------------
METHODOLOGY: this stack is NOT deterministic at temperature 0 on long generations
--------------------------------------------------------------------------------
Found while trying to validate the wide-step scratch. ONE unchanged server, the
same ~1100-token prompt, temperature 0, three consecutive requests:

     run1  2592 chars  556 tokens  "A person is checking my general abilities..."
     run2   753 chars  145 tokens  "The notes concern distributed storage..."
     run3  1737 chars  345 tokens  "The user is asking me to identify the topic..."

Same process, same weights, same prompt. This is PRE-EXISTING and has nothing to
do with the scratch -- the run above was on WIDE_SCRATCH=0. The likely source is
float non-associativity in the MoE reduction (moe_sum over top_k, plus vllm's
moe_align_block_size using global atomicAdd cursors), which perturbs a logit
enough to flip a near-tied token, after which a long greedy chain diverges
completely.

CONSEQUENCES, both learned by getting them wrong first:
  * Comparing generated TEXT between two configurations is not a correctness test
    on long outputs. My first equivalence run reported "long DIFFERS" and I nearly
    read it as a scratch bug; it was comparing against a moving target.
  * Comparing generated text is fine for SHORT outputs -- short and medium prompts
    were byte-identical across arms -- but short outputs cannot reveal a subtle
    expert misindex.
  => Correctness for this kind of change must be judged on the FIRST generated
     token's logprobs, before anything compounds, and against the WITHIN-server
     spread as the noise floor rather than against zero.

ALSO RETRACTED: the first equivalence test "passed" with both arms at 1 byte. This
model returns its tokens under `reasoning` until that block closes, so a small
max_tokens yields EMPTY content and two empty strings compare equal. Any output
comparison here must assert non-empty BEFORE believing an equality result.

--------------------------------------------------------------------------------
WIDE SCRATCH: CORRECTNESS VERDICT, and a PRE-EXISTING instability it uncovered
--------------------------------------------------------------------------------
Three test designs failed before one worked. Recording all three, because each
failure mode is a trap this stack sets:

  1. "IDENTICAL" on two 1-byte outputs. This model returns tokens under `reasoning`
     until that block closes; at max_tokens=300 both arms returned EMPTY content and
     two empty strings compare equal. A comparison must assert non-empty FIRST.
  2. "long DIFFERS" against a nondeterministic baseline. Text comparison is not a
     correctness test here at all (see below).
  3. A verdict line that passed because I had written a 10x slack factor into the
     criterion. The gap was 1.26 against a 0.47 noise floor; that is not a pass.

WHAT ACTUALLY WORKS: the first generated token's logprob for a FIXED token, with
the WITHIN-server spread as the noise floor. Note the subtlety in #4 below -- the
top-1 token's logprob is the wrong thing to track.

FINAL VERDICT (7 samples/arm, token 'The', off45/S46):
     length    spread on   spread off   between-arm mean gap
     tiny       0.000000     0.000000        0.000000
     medium     3.287136     1.548775        0.189942
     large      1.003758     2.904902        1.041144
  At every length the arms differ by less than one server differs from itself, and
  at `tiny` -- where the stack is perfectly reproducible, so there is nowhere to
  hide -- they are bit-identical. Short and medium generated TEXT was also
  byte-identical. => the scratch reads the same weights. PASS.

--------------------------------------------------------------------------------
PRE-EXISTING: prefill is numerically unstable, and it is NOT from this session
--------------------------------------------------------------------------------
Same server, same prompt, temperature 0, max_tokens=1 (ONE forward pass), 7 samples:
the fixed token 'The' varies by up to 3.29 nats -- a ~27x swing in probability.

CONTROL RUN, with every change from this session disabled
(EXPERT_CACHE_WIDE_SCRATCH=0, EXPERT_CACHE_FUSED_REMAP=0,
 EXPERT_CACHE_DECLINE_RESIDENT=0, GLM53_TOPK_REGSTORE=0, off28/S8, 42 layers
 armed / 0 declined -- i.e. the plugin as it was):
     tiny    spread 0.000000
     small   spread 1.761063   top1 flips between 'Present' and 'The'
     medium  spread 2.047323   top1 flips between ','        and 'The'
     large   spread 2.241498   top1 flips between 'All'      and 'The'
  Same magnitude as every configuration built today. NOT introduced by this work.

  4. MEASUREMENT TRAP worth its own line: the first sweep tracked the TOP-1 token's
     logprob. When top-1 changes identity between samples, that compares two
     DIFFERENT tokens and reports their gap as "noise". Track a fixed token.

WHERE TO LOOK NEXT. The instability is EXACTLY 0.000000 at ~13 tokens and only
appears as prompts grow. That length threshold fits the sparse attention indexer,
which only engages once the sequence exceeds its budget: unstable tie-breaking in
its top-k selection would give precisely this signature (perfect determinism when
short, large swings when long). Reduction order in the MoE sum -- my first guess --
does not explain the length dependence nearly as well. See
patched_sparse_attn_indexer_kpool.py. NOT INVESTIGATED; separate from this work.

=> STANDING CONFIG: OFFLOAD_GB=45, SLOTS=46, --max-num-batched-tokens 4096,
   fastpath gate,topk,nodoublegate, wide scratch ON. fp_launch.sh defaults updated.
   New gates, all default-on, all reversible without editing code:
     EXPERT_CACHE_WIDE_SCRATCH, EXPERT_CACHE_FUSED_REMAP,
     EXPERT_CACHE_DECLINE_RESIDENT, GLM53_TOPK_REGSTORE
