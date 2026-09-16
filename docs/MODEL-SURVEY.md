# Model survey: is there anything better to run on 2x MI210?

Four search passes, September 2026. **Conclusion: no.** Nothing is both close to
GLM-5.3-Flash in capability and small enough to remove the PCIe offload.

## Hard filters this hardware imposes

  * 128 GB VRAM total (gfx90a / CDNA2). GLM at 182 GB exceeds it by 54 GB, which is
    the entire reason offload dominates decode.
  * **No native FP8, no native FP4.** FP8-only, MXFP4-only and NVFP4-only
    checkpoints are unusable. W4A4 is also out — it needs 4-bit ACTIVATION kernels.
  * Proven path: compressed-tensors pack-quantized, 4-bit int, group 128, symmetric.
  * There is an open vLLM issue reporting AWQ failing on MI210 specifically, so
    prefer GPTQ / compressed-tensors over AWQ.
  * A model fully resident under ~110 GB is worth far more here than its benchmark
    delta suggests, because it removes the offload entirely.

## Benchmark baseline — get the index version right

Artificial Analysis Intelligence Index **v4.3 is current**. GLM-5.3-Flash's
widely-quoted **57 is v4.1.1**; **42 is the v4.3 rebase of the same model**. Both
are real. Sanity check: GLM-5.3 (max) is 60 on the old index and 45 on v4.3 — the
same ~0.75 rescale. **Compare on 42.**

Comparing a v4.3 score against a v4.1.1 score is the easiest mistake to make here.

## The bands

| band | best candidate | AA v4.3 | verdict |
|---|---|---|---|
| 80-200B | Ling-3.0-flash-VL (124B/5.5B) | 25 | far below |
| 200-340B | DeepSeek-V4-Flash (284B/13B) | ~35 | below, and text-only |
| **current** | **GLM-5.3-Flash (321B/18B)** | **42** | — |
| 400B+ | MiniMax M3, Kimi K3 | — | too large to serve |

### 80-200B — everything is far weaker

  * **Ling-3.0-flash-VL** 124B/5.5B, MIT, image+video, first-party int4 at 76.1 GB
    in compressed-tensors. Model card claims **42**; AA measured **25** on v4.3 and
    inclusionAI conceded the card is stale. AA-Omniscience 14% accuracy / 22%
    hallucination (thin recall). Terminal-Bench v4.0 **0%**. Verbose: ~50k output
    tokens per task vs ~30k for peers, which is a direct latency multiplier.
    The `vllm-ling-v3` fork appears to be STALE, not required — recipes.vllm.ai
    lists an official recipe needing only a nightly. (Two research passes disagreed
    on this; the one citing the recipe page is more likely right.)
  * **Qwen3.5-122B-A10B** 122B/10B, Apache 2.0, image+video, AA v4.3 **16**.
    Community AWQ 77 GB; no GPTQ, no first-party int4.
  * **Mistral-Small-4-119B-2603** 119B/6.5B, Apache 2.0, text+image (no video).
    AA **27 on v4.1.1** — NOT comparable to the v4.3 numbers above. First-party
    quant is NVFP4 only. Base ships FP8, so a W4A16 needs a dequant pass first.
    One ROCm report hits an AITER MLA assert (expects 16 or 128 heads; has 32).
  * Mainline vLLM support is the one thing Mistral clearly wins on.

### 200-340B — closer, but the checkpoint is always wrong

  * **DeepSeek-V4-Flash** 284B/13B, MIT, **text-only**, AA v4.3 ~35. The one case
    with a correct-format int4: `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`,
    **155.6 GB verified by summing the HF API**. Built from the APRIL base
    (`deepseek-ai/DeepSeek-V4-Flash`, created 2026-04-22), NOT the 0731 refresh,
    which is a separate repo with no usable int4 (its only 4-bit is MXFP4).
  * **DeepSeek-V4-Flash-Vision-Exp** ~305B. Same backbone: 43 layers, 256 experts,
    top-6. Official release is FP8. The only community 4-bit is **mixed int4+int8
    at 184.6 GB** — its size is the QUANTIZATION, not the model. A true W4A16 would
    land near 156 GB, i.e. ~26 GB smaller than GLM, with vision intact. No such
    checkpoint exists yet.
  * **Command A Plus** 218B/25B, Apache 2.0, multimodal, smallest in band — but its
    only 4-bit release is W4A4. Disqualified.
  * **Step-3.7-Flash** 198B multimodal — NVFP4 only. Disqualified.
  * **MiniMax M2.5 / M2.7** ~229B, text-only, AA v4.3 23 for M2.7. Far below.
  * **Hunyuan Hy3** 295B/21B, Apache 2.0, community AWQ-INT4, below GLM.

### Note on the DeepSeek family's native format

Its FP8 base is 166.9 GB (0731) / 159.6 GB (April) against a 155.6 GB int4 of the
same model — only ~7% apart. A genuinely 8-bit base would roughly halve when taken
to 4-bit. It does not, because the model is quantization-aware-trained at FP4+FP8
and **no BF16 master was released**. So converting buys EXECUTABILITY on gfx90a,
not size, and it is a lossy round trip from a source already at four bits.

### Larger, and better, but unrunnable

GLM-5.2 (744B, text-only, strongest open coder) · DeepSeek V4-Pro (1.6T, text-only)
· Qwen3.8-2.4T-A95B (explicitly drops vision) · Nemotron 3 Ultra (550B, text-only)
· Kimi K3 (2.8T/104B, natively multimodal, top-5 open weights — but 1.56 TB of
weights, more than the free disk).

"DeepSeek V4-Flash-Max" is **not a model** — it is AA running V4-Flash at maximum
reasoning effort.

## What is already on this box

    glm53-w4a16-mtp            182 GB   GLM-5.3-Flash, current
    q38fn-heretic2-mxfp4-fp8   118 GB   Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration,
                                        512 experts top-10, ngram+PLE keys) - AA v4.3 40,
                                        essentially at the GLM bar, and already in production
    dsv4-reap145-gguf           78 GB   DSV4-Flash-Vision-Exp REAP-145B, MXFP4/Q8 GGUF
    qwen36-35b-a3b              39 GB   Qwen3.6-35B-A3B + mmproj

Qwen3.8-Flash-Next deserves a second look: its large offloadable component is an
n-gram / per-layer-embedding table, which is a SPARSE GATHER. Only the rows for
actual tokens cross PCIe, unlike GLM's dense expert weights where every routed
expert's full matrix must come across. That is a structurally better offload
profile for this exact hardware.

No official REAP has ever been applied to a multimodal model, and REAP quality loss
swings enormously by architecture (99.8% retention on one model, collapse to 17% on
another).
