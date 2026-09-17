# The working configuration: exact flags, every patch, and why each one is there

**This is the reproduction recipe.** GLM-5.3-Flash (W4A16 int4, 182 GB) on 2× MI210
(gfx90a/CDNA2, 64 GB each, PCIe Gen4 x16, **no XGMI between the cards**), vLLM
0.28.1rc0+mi210.7. Single-stream decode: **~13.1 tok/s** at 76.1 ms/step.

Every value below is load-bearing and was established by measurement. Where a knob looks
arbitrary, the reason it is not is given. Numbers in `DECODE-COMPUTE-BUDGET.md`.

---

## 1. The launch command

```
vllm serve /models/glm53-w4a16-mtp \
  --served-model-name glm53 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --cpu-offload-gb 45 \
  --cpu-offload-params experts \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45 \
  --enable-expert-parallel \
  --max-num-batched-tokens 4096 \
  --kv-cache-memory=644245094
```

| flag | why |
|---|---|
| `--tensor-parallel-size 2` | The model does not fit on one card. PP=2 was analysed and is worse: it halves nothing and doubles per-layer latency at batch 1. |
| `--gpu-memory-utilization 0.97` | Higher fails to load. The residual ~3% is vLLM's fragmentation/workspace margin — it is **not** reclaimable headroom (tested: raising slots into it fails). |
| `--max-num-seqs 1` | Single-stream target. Also keeps `tokens × top_k` under the cache's slot count. |
| `--cpu-offload-gb 45` + `--cpu-offload-params experts` | 182 GB of weights against 128 GB of VRAM. Only experts are offloaded; everything else stays resident. |
| `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` | Worth ~25%. PIECEWISE silently costs ~60% of decode — **check the log for `Capturing CUDA graphs (FULL)`**, the launcher gates on it. |
| `--enable-expert-parallel` | Plain TP=2 faults in `fused_moe_kernel_gptq_awq` (OOB). EP avoids it. Also shards experts across ranks. |
| `--max-num-batched-tokens 4096` | Decode/prefill trade. 1024 costs ~30% on long prompts; 4096 is the measured balance. |
| `--kv-cache-memory=644245094` | **0.6 GiB → 15,753 tokens**, against `MAXLEN` 8192. vLLM otherwise holds 2.52 GiB. Frees VRAM for expert slots. Verified with a 4021-token prompt, not just short gates. |

## 2. Environment

| var | value | why |
|---|---|---|
| `EXPERT_CACHE_SLOTS` | `52` | **Measured optimum.** 46→52 is −3.13 ms; 58 is unstable; 60 will not launch; 64 dies on an illegal memory access. See §5. |
| `EXPERT_CACHE_POLICY` | `lfu` | LFU beats LRU; Belady's optimal bounds any online policy at only 12–16 pp above it, so policy work is closed. |
| `EXPERT_CACHE_DECAY` | `256` | LFU counter decay. |
| `GLM53_FASTPATH_PARTS` | `gate,topk,nodoublegate` | ~8% by kernel trace (−6.7 ms, −294 launches/step). **Not** `moe` — see §5. |
| `VLLM_ROCM_USE_AITER` | `1` | The image carries the AITER re-port for gfx90a. |
| `VLLM_ROCM_USE_AITER_MOE` | `0` | AITER's MoE path is inert here (no WNA16 support); leaving it on changes nothing but is misleading. |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Required by this image on CDNA2. |
| `PYTORCH_ROCM_ARCH` | `gfx90a` | |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | |

**Deliberately NOT set:**

* `HSA_FORCE_FINE_GRAIN_PCIE` — RCCL warns when it is missing, but it is measurably
  irrelevant here (`PCIE-FINE-GRAIN-TRACE.md`): indistinguishable at 0 and 1, both
  end-to-end and in a kernel microbenchmark.
* `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` — forcing NCCL from `P2P/IPC` to `SHM` moved
  nothing measurable (−1.44 ms against a 7.93 ms spread).
* `VLLM_ROCM_USE_AITER_CUSTOM_AR` / `VLLM_ALLOW_CUSTOM_AR_GFX90A` — the custom all-reduce
  path. Do not enable: see `ALL-REDUCE-CDNA2.md` and `P2P-PEER-HANDSHAKE.md`.

## 3. Device exposure — the one that silently costs 2.4×

```
--device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD131
```

**Expose only the MI210 render nodes.** On this box `renderD129`/`renderD130` are R9700s.
Exposing them makes vLLM's amdsmi arch probe read an RDNA4 card and silently disable every
gfx9 kernel path: **19.9 → 48.0 tok/s on a tiny model, with no error of any kind**.
`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` do **not** fix this — only render-node
exposure does.

## 4. Patches

### `patches/standing/` — mounted by the working server, all load-bearing

| file | mounts over | purpose |
|---|---|---|
| `patched_glm5next_model.py` | `vllm/models/glm5next/nvidia/model.py` | model wiring; installs the fast path at import |
| `patched_glm5next_attention.py` | `.../nvidia/attention.py` | attention for this arch |
| `patched_glm5next_kda.py` | `.../nvidia/kda.py` | KDA linear-attention layers |
| `patched_mla.py` | `vllm/model_executor/layers/mla.py` | MLA sparse attention |
| `patched_rocm_aiter_mla_sparse.py` | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | ROCm MLA sparse op |
| `patched_sparse_attn_indexer_kpool.py` | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | sparse indexer; **carries the page-size fix that lifted the prompt ceiling from ~700 to 7813+ tokens** |
| `patched_aiter_mla.py` | `aiter/mla.py` | AITER MLA for gfx90a |
| `moe_wna16_utils.py` | `vllm/.../quantization/utils/moe_wna16_utils.py` | chunked int4 repack; avoids an OOM at load |
| `glm53_fastpath.py` | `.../nvidia/glm53_fastpath.py` | gate GEMV + fused top-k + no-double-gate |

Plus the **`vllm-expert-cache`** plugin (separate repo, branch `offload-aware-caching`) and
optionally **`hostar/`** (this repo) for the all-reduce.

### `patches/experimental/` — kept for the record, NOT in the standing config

| file | status |
|---|---|
| `patched_rocm_ar_merged.py` | widens `use_custom_allreduce()` to gfx90a behind an env gate |
| `patched_aiter_ops_cdna2.py` | narrow exemption for `is_custom_all_reduce_enabled()` |
| `patched_opus.hpp` | guards FP8 helpers so the AR kernel JIT-builds on CDNA2 |
| `patched_custom_all_reduce.cuh` | the memory-ordering fixes; **correct but not faster than NCCL** |
| `patched_rocm_platform.py`, `patched_pa_mqa_logits.py` | earlier bring-up experiments |

**Do not enable the custom all-reduce chain.** It can be made numerically correct on
gfx90a (the MI300-only allowlist guards a real ordering bug, fixable in three lines), but
it does not beat NCCL, and the peer-VRAM handshake it depends on is unreliable between
these cards. Full reasoning in `P2P-PEER-HANDSHAKE.md`.

## 5. Things that look like wins and are not

| idea | verdict |
|---|---|
| `GLM53_FASTPATH_PARTS=...,layout,moe` | The M=1 int4 MoE GEMV **works** (10.5 → 4.5 ms) but needs a uint8 layout that makes the gather 21% slower and prefill ~3× slower. **Net −14 ms.** |
| More expert slots | 52 is the ceiling. 58 is unstable, 60 fails, 64 faults. |
| Tuned MoE config | Kernel-traced: 126.7 → 134.1 µs. Worse. |
| `VLLM_ROCM_USE_AITER_MOE=1` | Inert — no WNA16 support. |
| Custom all-reduce | See above. |
| Disabling PCIe ACS | No measurable effect, and does not fix peer-VRAM handshakes. |
| Gather geometry (`CHUNKS`/`LANES`) | Flat 25–28 GB/s from 16→512 chunks. Defaults are optimal. |

## 6. Verifying a launch is actually good

The launcher gates on these; check them by hand if you launch differently:

1. `Capturing CUDA graphs (FULL)` — PIECEWISE costs ~60% of decode, silently.
2. `expert-cache <ver> active on N MoE layer(s)` — otherwise you are on the stock,
   uncached MoE path and every benchmark is invalid.
3. `policy=lfu` in that same line.
4. `GPU KV cache size: 15,753 tokens` — must exceed `--max-model-len`.
5. Decode ≈ **76 ms/step**. If it is ~96 ms, the fast path or cudagraphs did not engage.

**Correctness gate:** this model streams into `reasoning` until that block closes, so a
small `max_tokens` returns an EMPTY `content` that compares equal to anything. Always
assert non-empty before believing a pass, and use `max_tokens >= 3000`.

## 7. Measurement warnings

* **End-to-end decode timing cannot resolve anything under ~9 ms.** Within-arm spread is
  up to 8.9 ms and the box drifts several ms mid-run, in both directions. Use kernel traces
  for smaller effects.
* **Compare arms within one script, launched back to back.** Launch-to-launch variance
  alone spans 79–96 ms for identical configs.
* **`EXPERT_CACHE_STATS` is blind under CUDA graphs** — the counters are Python and only
  run at capture, so any hit rate read in graph mode is a cold-start artifact. Use
  `--enforce-eager` to measure miss rate.
* **Check a trace is of the configuration you think it is** before reading anything off it.
* **The box thermally throttles under sustained load, and IQR will not reveal it.** A
  monotonically RISING sample series is a thermal signal, not noise: a single card driven
  hard goes 79 -> 92 C junction and 238 -> 189 W, doubling decode time, then recovers on
  idle. Check for monotonic rise before reporting a median; if present, report cold and
  sustained separately. Cool ~60 s between arms. GLM-5.3 masks this because its step is
  ~20% idle on PCIe so power stays low -- a flat GLM run does NOT mean the box is immune.
  See docs/GEMMA4-31B.md.
