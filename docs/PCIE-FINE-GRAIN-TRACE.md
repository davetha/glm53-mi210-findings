# Task 1.1 — HSA_FORCE_FINE_GRAIN_PCIE trace results

**Verdict: no measurable effect. Leave it UNSET (default). There is no "optimal setting" to
lock in — the two settings are indistinguishable, and inventing a preference would be
fabricating a result.**

## What was measured

The 60-line host-staged all-reduce (`hostring.hip`) in isolation: 2 ranks, 8 KB payload
(the real per-layer decode payload at batch 1), 100 warm iterations discarded, then 2000
measured. Per-iteration timings recorded on rank 0 via `s_memrealtime`, with the counter
rate self-calibrated against the HIP event timer (so no assumption about 25 vs 100 MHz).
3 repetitions per setting, fresh container each time.

## Results

| setting | mean (3 reps, µs) | p50 | p90 | p99 | max | tail p99/p50 |
|---|---|---|---|---|---|---|
| `HSA_FORCE_FINE_GRAIN_PCIE=0` | 5.03 / 5.04 / 5.25 | 4.86–5.03 | 5.35–6.16 | 6.00–6.65 | 9.9–118.8 | 1.23–1.32× |
| `HSA_FORCE_FINE_GRAIN_PCIE=1` | 5.09 / 5.26 / 4.97 | 4.86–4.87 | 5.19–6.00 | 6.00–6.65 | 10.9–118.6 | 1.23–1.37× |

The between-setting difference is smaller than the between-repetition spread *within* each
setting. Correctness passed in all 6 runs (`all elements == in0 + in1`, both ranks agree).

## Why the flag does nothing here

`hostring.hip` allocates its staging arena with `hipHostMalloc(..., hipHostMallocCoherent)`
— explicitly fine-grained host memory. `HSA_FORCE_FINE_GRAIN_PCIE=1` forces globally what
this code already requests per-allocation, so it is redundant for this kernel.

RCCL warns at startup when the flag is missing ("can lead to low RCCL performance, system
instability or hang"), and that warning may well be load-bearing for RCCL's *own*
allocations. But an end-to-end decode A/B of the flag against a paired baseline also came
back inside noise (−3.12 ms against an 11.35 ms within-arm spread), so there is no evidence
for setting it in this deployment either.

## The number that actually matters

Independent of the flag, the host-staged all-reduce is **stable**, which is the property
the peer-VRAM path lacked entirely:

```
p50 4.86 us   p90 ~5.6 us   p99 ~6.3 us   tail ratio 1.23-1.37x
```

A tail ratio near 1.3× is steady latency, not stall-prone behaviour. Rare ~118 µs outliers
occur at roughly 1-in-2000 iterations (~0.05%) and do not move p99.

Projected against the traced decode step:

| | per exchange | × 90 per step |
|---|---|---|
| NCCL (measured in-model) | 116 µs | 14.45 ms |
| host-staged, p50 | 4.86 µs | 0.44 ms |
| host-staged, p99 | 6.65 µs | 0.60 ms |

Even costing p99 on every single exchange, this is **~13.9 ms/step** below NCCL.

## Caveat carried into Task 1.2

This harness loops *inside* one kernel launch, so it amortises launch cost across 2000
iterations. In the model each all-reduce is a separate node in a CUDA graph; at the
ledger's measured ~2.4 µs per graph node that adds to every exchange, putting the realistic
in-model figure nearer 7–9 µs. Still ~13 ms/step better than NCCL, but the isolated 4.86 µs
is a floor, not a prediction.

## Reproduce

```bash
docker run --rm --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD131 \
  --group-add video --ipc host -e HSA_FORCE_FINE_GRAIN_PCIE=<0|1> \
  -v <bringup>:/w -w /w --entrypoint /w/hostring <image>
```
