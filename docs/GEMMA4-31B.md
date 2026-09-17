# Gemma-4-31B W8A8-INT8 on 2× MI210: 45.9 tok/s, and what it teaches about GLM

**Runs first try, hits every fast path, and is 3.5× GLM-5.3's decode and ~5× its prefill.**
The gap is structural, not tuning — and measuring it corrected two beliefs carried over from
the GLM work.

Model: `valoomba/Gemma-4-31B-it-uncensored-heretic-Quark-W8A8-INT8`, 33.3 GB.

## Results

| | GLM-5.3-Flash (321B MoE, W4A16) | **Gemma-4-31B (dense, W8A8 INT8)** |
|---|---|---|
| decode, cold | 13.1 tok/s | **45.9 tok/s** (21.8 ms/step) |
| decode, sustained | 13.1 tok/s | **35.7 tok/s** (~28 ms/step) |
| prefill @1k | 456 tok/s | **2617 tok/s** |
| prefill @6.5k | 632 tok/s | **2290 tok/s** |
| weights | 182 GB, 45 GB/rank offloaded | 33 GB, **fits entirely** |
| PCIe expert gather | 15.43 ms/step (20%) | **none** |

GLM does not throttle and Gemma-4 does, so GLM's single figure is both cold and sustained.

## Configuration

```
--tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 8192
--max-num-seqs 1 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--trust-remote-code
```
Devices: `--device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD131`.
Env: `VLLM_ROCM_USE_AITER=1 HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**No expert cache, no offload, no fastpath patches.** All of the GLM apparatus is dead
weight here — the model is dense and fits.

Engagement checks, all of which have silent failure modes:

```
FULL cudagraph captures: 3      PIECEWISE costs ~60% of decode and logs no error
36 int8 / 1 quark               INT8 engaged, NOT silently dequantised to bf16
gfx90a                          correct arch (an R9700 render node -> RDNA4 probe -> gfx9 paths off)
16.46 GiB/rank                  exactly half of 31.47
GPU KV cache: 38,870 tokens     sliding window working (50 of 60 layers cap at 1024)
correctness 3/3
```

## TP=1 vs TP=2: I predicted wrong, twice over

The model fits on one 63 GiB card, and this box has no XGMI, so TP=1 avoiding all-reduce
looked like the obvious win. It is not:

```
TP=1: 33.43 33.82 36.81 40.47 44.30 48.22 52.45 56.41 60.64 63.69 64.33 63.36
      median 50.34 ms -> 19.87 tok/s   IQR 22.89
TP=2: 21.79 21.78 21.78 21.78 21.79 21.78 21.79 22.32 22.68 22.83 23.04 23.16
      median 21.79 ms -> 45.89 tok/s   IQR  1.05
```

TP=2 beats TP=1 even at TP=1's *cold* best (21.8 vs 33.4). Two reasons:

### 1. All-reduce here costs ~44 µs/exchange, not 116-185 µs

I predicted TP=2 would pay ~19 ms of all-reduce, extrapolating GLM's measured 116-185 µs
per exchange across 60 layers × 2. But TP=2's entire step is 21.79 ms, and 16.46 GB/rank at
the ~1000 GB/s this hardware delivers already accounts for ~16.5 ms — leaving ~5 ms for 120
exchanges, about **44 µs each**.

**GLM's expensive all-reduce was substantially PCIe contention, not protocol latency.** GLM
streams expert weights over the same link continuously (15.43 ms/step of gather), so its
all-reduces compete with saturating traffic. On an uncontended link the same two cards
exchange 3-4× faster.

Consequence for the GLM work: `hostar`'s realistic ceiling there was lower than the raw
116 µs-vs-6.7 µs comparison suggested, because much of NCCL's cost was contention that a
faster barrier cannot remove either.

### 2. The box thermally throttles, and TP=1 throttles much harder

```
TP=1 sustained (single card):  junction 79 -> 92 °C, power 238 -> 189 W, decode 33 -> 64 ms
TP=2 sustained (both cards):   junction ~86-87 °C, power 163-175 W each, decode 21.5 -> ~28 ms
```

TP=1 degrades **92%**; TP=2 degrades **30%** and plateaus. Splitting the work halves
per-card power, so two cards at 170 W sustain far better than one at 238 W.

## Methodology: throttling invalidates the usual noise handling

`CONFIGURATION.md` says to judge runs by IQR rather than max−min. **That is insufficient
here.** A monotonically rising sample series is a thermal *signal*, not noise to be
averaged:

```
33.43 33.82 36.81 40.47 44.30 48.22 52.45 56.41 60.64 63.69 64.33 63.36
```

Taking the median of that (50.34) describes the throttle curve, not the model. Rules:

* **Check for monotonic rise before reporting any median.** If present, report cold and
  sustained separately — they are different numbers and both are real.
* **Cool the cards between arms** (~60 s idle) or the second arm inherits the first's heat.
* GLM masked this entirely: its step is ~20% idle waiting on PCIe, so power stays low and
  its samples are genuinely flat (IQR ~1.3). Do not assume a flat GLM run means the box
  does not throttle.

## Why this model suits the hardware and GLM does not

GLM-5.3 is 321B MoE at 182 GB against 128 GB of VRAM, so 45 GB/rank lives in host RAM and
every step streams experts over PCIe. That single fact produces both of its dominant costs:
the 15.43 ms gather, and the inflated all-reduce that the gather causes by saturating the
link. Together roughly 40% of its step, and neither is recoverable in software.

Gemma-4-31B is dense and fits. It pays neither. Nothing clever was required — it ran
correctly on the first launch with stock flags plus the device-exposure and cudagraph
lessons from the GLM work.

## Files

* `tools/gemma4/g4_launch.sh` — launcher + engagement checks + decode benchmark, TP as argv
