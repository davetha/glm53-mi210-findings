# GLM-5.3-Flash on 2x AMD MI210 — findings

Performance work on GLM-5.3-Flash (321B MoE, multimodal, W4A16 int4, 182 GB) served
by vLLM on two MI210s (gfx90a / CDNA2, 128 GB VRAM total). The model is 54 GB larger
than VRAM, so expert weights are offloaded to pinned host memory and read over PCIe
Gen4 x16. **Everything here follows from that one fact.**

## Headline

| metric        | before | after | change |
|---------------|--------|-------|--------|
| decode tok/s  | 10.96  | ~11.8 | +7%    |
| prefill @2000 | 316.8  | 699.5 | +121%  |
| prefill @7000 | 183.7  | 603.3 | +228%  |

## Read this first

**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — how to measure anything on this box
without fooling yourself. Nine rules, every one learned by getting it wrong. If you
read nothing else, read rules 1 and 2:

  1. One container start is worth ±0.9 tok/s. Single-container comparisons below
     ~6 ms are meaningless.
  2. This stack is not deterministic at temperature 0. Comparing generated text
     between configurations is not a correctness test on long outputs.

## Contents

    docs/METHODOLOGY.md    measurement rules and the retractions behind them
    docs/RESULTS.md        final config, what produced each gain, what is closed
    docs/MODEL-SURVEY.md   is there a better model to run? (short answer: no)
    docs/OPEN-ISSUES.md    ranked, with the biggest one first
    docs/CACHE-POLICY-CEILING.md      Belady optimal: why the policy question is closed
    docs/EXPERT-OFFLOAD-PLACEMENT.md  which layers to offload (unconfirmed, with retraction)
    docs/DECODE-COMPUTE-BUDGET.md     what a decode step is made of; every bucket, closed
    docs/ALL-REDUCE-CDNA2.md          custom all-reduce on gfx90a: it corrupts (superseded in part)
    docs/P2P-PEER-HANDSHAKE.md        why the custom all-reduce cannot pay; GPU peer handshakes
    LEDGER.md              the full experiment log, ~3000 lines, including dead ends

    patches/               code changes, all behind default-on env gates
    tools/bench/           interleaved A/B harnesses (use these, not single runs)
    tools/trace/           rocprofv3 per-step kernel analysis
    tools/correctness/     equivalence, determinism, logprob comparison
    tools/hf/              HuggingFace inspection (sum file sizes, read quant configs)

## The three changes that mattered

1. **Resident-layer decline** — a real bug. The expert cache armed all 42 MoE layers
   including the ~25 whose experts never left the GPU, wasting 2.42 GiB/rank on slot
   buffers that could never miss, and running a gather every step that copied VRAM
   to itself.

2. **Operating point** — the freed VRAM plus more offload took slots from 8 to 46 of
   144, cutting PCIe traffic per step from 569 MiB to roughly 360.

3. **Wide-step VRAM scratch** — prefill bypasses the cache, so the MoE GEMM was
   reading host memory at ~6.8 GB/s where the gather kernel sustains 21-25 GB/s on
   the same link. Staging experts into VRAM first tripled prefill.

## The biggest open item is not performance

Prefill is numerically unstable and was before any of this work: the same prompt at
temperature 0 produces materially different answers. See docs/OPEN-ISSUES.md #1.

## Caveat on the model survey

The survey rests on web research that was demonstrably wrong at least twice — it
called a model "hard no, Blackwell-required" that had been running on this box for
three days, and it missed an int4 checkpoint that existed. Treat its numbers as a
starting point, and prefer testing to trusting a support matrix.
