# What a decode step is actually made of (2× MI210, GLM-5.3-Flash W4A16, TP=2)

**Verdict: the step is fully accounted for, and every bucket above 5 ms is closed by
measurement. ~12.4 tok/s is near the practical floor on this hardware.**

Counterpart to `P2P-PEER-HANDSHAKE.md`. That one covers the all-reduce; this covers
everything else, and the methodology, because most of the work here was discovering that
earlier numbers — including several of my own — were measuring the wrong thing.

Hardware: 2× MI210 (gfx90a/CDNA2), PCIe Gen4 x16, **no Infinity Fabric between the cards**.
Config: `OFFLOAD_GB=45 SLOTS=46 --max-num-batched-tokens 4096`, fastpath
`gate,topk,nodoublegate`, CUDA graphs FULL_DECODE_ONLY, `--enable-expert-parallel`.

---

## 1. The budget

Single decode step, CUDA graphs (**not** eager — see §2), prefill-free window,
213.6 steps:

| bucket | ms/step | % | calls/step | status |
|---|---|---|---|---|
| expert gather (PCIe) | 15.43 | 23.2% | 50.9 | **closed** — at the PCIe wall |
| all-reduce (NCCL) | 14.45 | 21.7% | 90.0 | **closed** — PCIe, no fabric |
| MoE expert GEMM | 10.14 | 15.2% | 82.2 | **closed** — replacement loses net |
| dense GEMM (`wvSplitK`) | 8.93 | 13.4% | 351.2 | **closed** — ~60% of HBM peak |
| elementwise | 5.59 | 8.4% | 1105.5 | **closed** — no launch gaps to recover |
| mHC hyper-connections | 4.22 | 6.3% | 176.1 | unexamined |
| idle | 3.61 | 5.1% | — | rare stalls, uninvestigated |
| other | 3.16 | 4.7% | 442.2 | small |
| tensor copies | 1.73 | 2.6% | 275.9 | small |
| sparse attn + indexer | 1.54 | 2.3% | 21.5 | small |
| routing / top-k | 0.80 | 1.2% | 41.1 | small |
| reductions / norms | 0.65 | 1.0% | 91.0 | small |
| **step** | **70.22** | | **2727.5** | GPU **94.9% busy** |

Two facts that kill whole categories of optimisation before anyone starts:

* **The GPU is 94.9% busy.** The step is not dispatch-bound despite 2,727 kernel
  launches, and not stalling on host round-trips. Fusing the 1,105 elementwise launches
  would chase gaps that are not there — 84% of the little idle there is sits in ~57 rare
  multi-ms stalls, not between kernels.
* **45% of the step is PCIe physics** (gather + all-reduce), not software. Both would
  largely vanish given a fabric-connected card pair or enough VRAM to stop offloading.

On `70.22` vs the benchmarked `~83 ms/step`: both are correct and measure different
things. The trace is the pure kernel timeline in steady decode; the benchmark divides
request wall-time by tokens, so it carries amortised prefill, detokenisation and HTTP.
The budget is internally consistent: 66.6 ms busy + 3.6 idle = 70.2.

---

## 2. Methodology, and four traps that produced wrong answers first

**Profile under CUDA graphs, not eager.** Every earlier profile in this project used
`--enforce-eager`, which distributes time completely differently (an eager step measured
73.6 ms busy against an ~80 ms graph step with a different composition). The 66% of the
step that is not gather had never been looked at in the mode actually served.

**Do not use the whole-run `kernel_stats.csv`.** Its top entry, `direct_copy_kernel_cuda`,
has min 5.3 µs and **max 1.8 seconds** — those are model weight loads. The aggregate
describes startup, not serving, and reports that one kernel as 68.8% of GPU time.

**TRAP 1 — window edges counted as idle.** Computing wall time as the requested window
rather than the span the kernels occupy reported **17.5 ms/step idle** where the true
figure is 3.6. If decode does not fill the window, the empty edges become fake idle.
Always take wall from `max(end) - min(start)` of the dispatches themselves.

**TRAP 2 — one prefill contaminating the window.** 26 gather calls of ~68 ms each,
clustered in a single 1.9 s burst (one call per armed layer, each refilling the whole cold
cache), accounted for **1,783 ms of 5,107 ms** of gather time. Including them inflated the
gather bucket from 15.43 to 23.75 ms/step and produced an impossible-looking bandwidth.
Check for bulk calls before trusting any per-step average.

**TRAP 3 — reading one dimension of a 2D grid.** `Grid_Size_X` alone said the gather ran
**16 workgroups**; the grid is `dim3(chunks, lanes)` and it actually runs 1024. This
produced a confident and completely wrong "the gather is occupancy-starved" claim.

**TRAP 4 — `EXPERT_CACHE_STATS` is blind under CUDA graphs.** `_report_stats` is Python in
the MoE apply path, and under FULL_DECODE_ONLY the Python body runs **only at capture**.
The counters therefore only ever see the eager warm-up passes: three attempts returned
~100 layer-steps regardless of how long decode ran, and a cold-start hit rate implying
42.6 GB/s — above PCIe Gen4 x16's 31.5 GB/s ceiling, which is how the error was caught.
**Any cache hit rate read from this mechanism in graph mode is a cold-start artifact.**
That matters beyond this document: the cache-policy work reads hit rates.
Workaround: run `--enforce-eager` with `EXPERT_CACHE_STATS=100`. The miss rate is a
property of routing and transfers between modes; only wall time differs. Setting
`STATS=1` does **not** work — the `.item()` sync lands inside graph capture and kills it
with `hipErrorStreamCaptureInvalidated`.

**The noise floor.** End-to-end decode timing on this box has a within-arm spread of
**8.91 ms** (paired A/B, 12 kept samples after 8 warm-up: median 83.50, min 79.46, max
88.37, drifting ±4 ms mid-run). **Nothing under ~9 ms can be demonstrated by tok/s at any
sample size.** Kernel-level tracing is the only usable instrument for changes below that.

---

## 3. Why each bucket is closed

### Expert gather — 15.43 ms — AT THE PCIe WALL

The decisive evidence is that gather durations **quantise into integer multiples of
462 µs**, which is exactly one expert (12.375 MiB) at 28.1 GB/s:

```
65.3%  hits (<50 us)          15.7%  1 expert  (~462 us)
10.9%  2 experts (~924 us)     5.3%  3 experts (~1385 us)
 1.9%  4 experts (~1847 us)
```

Smeared buckets would mean a bandwidth shortfall; integer buckets mean every copy runs at
link speed. Corroborated from two independent directions:

* **miss rate 1.29/layer-step** from the trace (decode-only expert-copies ÷ steps)
* **miss rate 1.28/layer-step** from `EXPERT_CACHE_STATS` in eager mode — a different run,
  a different prompt, 403 samples, both ranks agreeing (67.6% / 68.4% hit rate), and
  routed-experts/layer-step 4.05 against the ~4.2 predicted for top-8 across 2 EP ranks.

This also vindicates the old 1.34 figure the ledger had flagged as untrustworthy.

Everything else about the gather was eliminated directly, with a standalone harness around
the **verbatim production kernel** (`gather_geom.hip`):

| hypothesis | result |
|---|---|
| launch geometry | flat: 25–28 GB/s from chunks 16→512; default already optimal |
| scattered vs sequential reads | **no penalty**: 23.94 vs 23.83 GB/s (`scatter_bw.py`) |
| pinned-region size / IOMMU reach | flat: 28.1 GB/s at 0.6, 4, 8 and 14.5 GiB |

The ledger's earlier "geometry is flat" conclusion was **right**, despite having been
measured end-to-end where the noise floor should have hidden the effect. It got lucky, but
the kernel-level sweep confirms it independently.

### MoE expert GEMM — 10.14 ms — replacement exists and loses

`fused_moe_kernel_gptq_awq` runs ~6× off roofline at M=1 (independently corroborated here:
~234 GB/s from tensor-index bytes ÷ traced time, against `wvSplitK`'s ~1,000 GB/s in the
same step). Three attacks, all closed:

* **autotune** — the tuned config is *worse*, kernel-traced variance-free: 126.7 → 134.1 µs.
  The search space cannot fix a kernel that tiles 16 MFMA rows around one real token.
* **AITER MoE** — inert; no WNA16 support (`VLLM_ROCM_USE_AITER_MOE=1` still logs
  "Using TRITON WNA16 MoE backend").
* **`_moe_int4_gemv_m1`** — a hand-written M=1 int4 GEMV that *works*: gate_up 138→67 µs,
  down 52→36 µs, ~10.5 → ~4.5 ms/step. But it needs K-packed uint8, and moving off vLLM's
  int32 N-packed repack makes the gather **21% slower per call** and prefill **~3× slower**:
  89.5 → 103.6 ms/step. **Net loss ~14 ms.** Retained as
  `GLM53_FASTPATH_PARTS=...,layout,moe` for anyone wanting to chase the gather regression.
  At today's operating point it would be *worse* still, since the gather is a larger share.

### Dense GEMM — 8.93 ms — already at ~60% of peak

Non-expert weights are **19.34 GB total, 9.67 GB per rank** at TP=2 (summed exactly from
the safetensors index), essentially all read once per step. 9.67 GB / 8.93 ms ≈
**1,000 GB/s** against MI210's 1,638 GB/s peak.

`wvSplitK` launches **104 workgroups on 104 CUs** — one per CU, pinned there by a full
64 KB LDS allocation, so one wavefront per CU with no latency hiding. That *looks*
pathological and is not: it is vLLM's skinny-GEMM design trading occupancy for ILP, and at
60% of peak it is working. Occupancy is not evidence of a problem by itself.

### Elementwise — 5.59 ms across 1,105 launches — nothing to fuse

Fusion would recover inter-kernel gaps. At 94.9% GPU busy those gaps do not exist.

---

## 4. What is actually left

* **mHC hyper-connections, 4.22 ms, 176 calls/step.** GLM-specific, never examined.
* **3.61 ms of idle**, 84% of it in ~57 rare multi-ms stalls across 213 steps. Worth
  identifying what serialises there, though the ceiling is small.
* **Requantisation below int4** — ruled out by the user, and correctly: it is the only
  remaining lever that would meaningfully cut the 45% of the step that is PCIe traffic.

Realistically, single-stream decode here is near its floor. The two dominant buckets are
both PCIe physics on a card pair with no fabric between them.

## Files

Under the bring-up directory (`glm53-bringup/`):

* `decode_budget.py` — per-step budget from a rocprofv3 trace, windowed to steady state
* `gap_profile.py` — idle-time distribution; separates dispatch overhead from real stalls
* `gather_geom.hip` — standalone harness around the verbatim production gather kernel
* `rp/scatter_bw.py` — sequential vs scattered pinned-host read bandwidth
* `miss_eager.sh` — miss rate via `EXPERT_CACHE_STATS` in eager mode (the only mode it works in)
* `profile_graph.sh` — rocprofv3 trace of a CUDA-graph decode run
