# How to measure anything on this box

Every rule here was learned by getting it wrong first. The retractions are kept
deliberately: each one is a trap this specific stack sets, and none of them is
obvious until it has cost you a conclusion.

## 1. One container start is worth ±0.9 tok/s

The single most important rule. The 27 GB pinned host offload buffer lands in
different physical memory on every container start, and the gather reads it over
PCIe, so between-container variance is confounded with whatever you are testing.

Repeating the SAME configuration across three restarts:

    arm=on,  per-container means:  91.80  97.71  92.81 ms/step
    arm=off, per-container means:  93.53  94.33  89.75 ms/step

Within a single container the spread can be as low as 0.13 ms, which is exactly
what makes one container look conclusive. It is not.

**Rule:** any decode claim below ~6 ms / ~0.9 tok/s needs interleaved restarts,
three per arm minimum, arms alternated so drift hits both equally. Use
`tools/bench/fusion_ab.sh` as the template. Compare PAIRED per-restart means, not
raw min/max — two distributions can overlap at the edges while one wins every pair.

**Retracted because of this:** an early single-container pair read "87.45 on vs
93.5 off = 6 ms win". That was memory placement, not the change.

## 2. This stack is NOT deterministic at temperature 0

Same server, same prompt, temperature 0, `max_tokens=1` — ONE forward pass — the
fixed token 'The' varies by up to 3.29 nats. That is a ~27x swing in probability.

Confirmed PRE-EXISTING with every session change disabled
(`EXPERT_CACHE_WIDE_SCRATCH=0 EXPERT_CACHE_FUSED_REMAP=0
EXPERT_CACHE_DECLINE_RESIDENT=0 GLM53_TOPK_REGSTORE=0`, off28/S8, 42 layers armed):

    tiny   (~13 tok)  spread 0.000000
    small            spread 1.761063   top1 flips 'Present' <-> 'The'
    medium           spread 2.047323   top1 flips ','       <-> 'The'
    large  (~1100)   spread 2.241498   top1 flips 'All'     <-> 'The'

**Rule:** comparing generated TEXT between two configurations is not a correctness
test on long outputs. It is fine on short ones. For anything longer, compare the
FIRST generated token's logprob, against the WITHIN-server spread as the noise
floor — never against zero.

## 3. Track a FIXED token, not the top-1 token

The first length sweep recorded the top-1 token's logprob. When the top-1 token
changes identity between samples — which it does here — that compares two DIFFERENT
tokens and reports their gap as "noise", inflating it badly.

`tools/correctness/fixedtok.py` tracks one token present in every sample.

## 4. Assert non-empty BEFORE believing an equality result

The first equivalence test reported IDENTICAL on two 1-byte outputs. This model
returns its tokens under `reasoning` until that block closes, so at `max_tokens=300`
both arms returned EMPTY content — and two empty strings compare equal.

**Rule:** any output comparison must assert non-empty and fail loudly otherwise.

## 5. No slack factors in a verdict

A pass criterion written as `gap <= noise * 10` reported PASS on a 1.26 gap against
a 0.47 noise floor. If the bar needs a fudge factor to clear, it did not clear.

## 6. Sum the file sizes; never quote a page summary

A checkpoint was quoted at 301.4 GB from a page summary. Summing the HF API sizes
gave 451.7 GB. That 150 GB error invalidated an entire sizing analysis.
`tools/hf/hfinfo.py` sums `siblings[].size` directly.

Related: comparing `du -sb` on a directory against a sum of API sizes produced a
phantom "4.3 GB smaller" difference. They were identical.

## 7. Under CUDA graphs, Python in the hot path runs only at CAPTURE

With `FULL_DECODE_ONLY`, the expert cache's `apply()` body executes during capture,
not per step. Consequences:

  * `EXPERT_CACHE_STATS` prints nothing in a graph run — its `.item()` never
    re-executes. Use eager mode for it.
  * Host-side savings are already free. Only GPU busy time counts. This is why a
    kernel-fusion pass that removed 126 kernels/step moved the wall clock zero.

## 8. Don't corrupt weights to measure bandwidth

An in-model test that degraded gather fidelity to measure PCIe cost was INVALID:
corrupting weights changes the output, which changes routing, which changes the
miss count. A "47% gain" was routing collapse.

Measure bandwidth outside the model — `rp/uva_scaling.py` uses 64 distinct slabs
from an 830 MB source so nothing is served from MI210's 8 MB L2. An earlier version
re-read one buffer 20x and showed GB/s *rising* as the copy shrank.

## 9. Vendor support matrices under-report this box — three times

  * vLLM's quantization x hardware matrix marks AWQ, GPTQ and compressed-tensors
    as unsupported on AMD. The production server runs compressed-tensors int4,
    4-bit, group 128, symmetric, on gfx90a, daily.
  * GLM-5.3-Flash's own ROCm matrix lists gfx950 only. It serves on gfx90a.
  * A research pass called Qwen3.8-Flash-Next "Blackwell-required, hard no". It had
    been running on this box for three days.

**Rule:** direct observation beats a support matrix. Test before believing a "no".
