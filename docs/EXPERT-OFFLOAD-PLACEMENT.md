# Does it matter WHICH MoE layers you offload?

*An unconfirmed result, a solid observation, and a retraction. Written for someone
working on PCIe-bound inference.*

**TL;DR** — Serving a 321B MoE that does not fit in VRAM, ~26 of 42 expert layers live in
pinned host RAM and cross PCIe on every token. vLLM picks *which* layers those are by
module construction order, i.e. by accident. Layers are not interchangeable: measured
per-layer cache hit rate spans **46% to 89%** at an identical slot budget — that part is
solid. Offloading the *cacheable* layers instead of the *uncacheable* ones measured
**+3.3% decode (2.66 ms/step) over 3 interleaved restarts per arm, winning 2 of 3 pairs,
with the distributions OVERLAPPING**. That is below this machine's measurement floor, so
the idea is **NOT CONFIRMED** — promising mechanism, inconclusive result. Read §6 and §8
before acting on it.

---

## 1. Setup

| | |
|---|---|
| Model | GLM-5.3-Flash, 321B total / 18B active MoE, 45 layers (42 with experts), 288 experts, top-8 |
| Quant | W4A16 int4, compressed-tensors pack-quantized, group 128, symmetric → 182 GB |
| GPUs | 2× AMD Instinct MI210 (gfx90a / CDNA2), 64 GiB each, 128 GB total |
| Serving | vLLM on ROCm, TP=2 + expert parallelism, `max_num_seqs=1` (single-stream decode) |
| Offload | vLLM UVA offloader, `--cpu-offload-gb 45` per rank → ~45 GiB/rank in pinned host RAM |

The model exceeds VRAM by ~54 GB. Offloaded expert weights are read **in place over PCIe**
via a device-addressable view of pinned host memory (UVA) — they are not staged, the GEMM
reads them where they live.

### Measured link characteristics

Both MI210s are **PCIe Gen4 x16** (16 GT/s), confirmed at every hop from root port to die.
Topology is one MI210 + one R9700 per IO-die quadrant on an EPYC 74F3 (Milan), so the two
TP ranks sit on **separate root complexes** and do not contend for host bandwidth.

| path | GB/s |
|---|---|
| Gen4 x16 theoretical | 31.5 |
| host→device, pinned, large | 26.4 (84%) |
| expert gather kernel, real workload | 21–25 (73%) |
| P2P GPU↔GPU (crosses quadrant fabric) | 23.7 |
| P2P 4 KB one-way latency | 20.5 µs |

IOMMU is `iommu=pt` (passthrough), so DMA is not paying address remapping. NPS1, so host
memory is interleaved across all 8 channels (2933 MT/s, ~187 GB/s aggregate) — memory has
~6× the headroom of the link. **The link is the constraint, not memory and not topology.**

---

## 2. The expert cache

An LFU slot cache sits in front of the offloaded experts. Per armed layer it holds `S`
experts resident in VRAM; a routed expert that is resident is free, a miss is a
**12.375 MiB** PCIe transfer (measured: 97% packed weights, 3% scales — no metadata waste).

Current operating point: `S = 46` of the 144 EP-local experts per layer.

---

## 3. The observation: layers are not interchangeable

Dumped a real routing trace (12,600 cache refreshes, 51,114 owned-expert requests) and
replayed LFU per layer at a fixed `S = 46`:

```
layer    requests     misses     hit %
L15          2917        326     88.8%     <- deep
L16          2637        439     83.4%
L13          3196        635     80.1%
L14          3473        763     78.0%
...
L00          2909       1475     49.3%
L03          3114       1603     48.5%
L02          2809       1479     47.3%
L01          2989       1610     46.1%     <- shallow

overall 65.0%   spread 42.7 pp
misses in the 8 MOST cacheable layers : 5,229
misses in the 9 LEAST cacheable layers: 12,652
```

Clean monotonic gradient with depth. **Shallow layers route near-uniformly across experts**,
so a cache cannot help them and they drag most of their working set across the link every
step. **Deep layers concentrate on a stable favourite set**, so a cache serves most requests
from VRAM. A shallow layer on the host costs ~5× the PCIe traffic of a deep one.

This is consistent with the general finding that early transformer layers do more generic
feature extraction while later layers specialise, but here it is measured, not assumed.

---

## 4. The bug: placement is chosen by accident

vLLM's offloader (`vllm/model_executor/offloader/uva.py`) works like this:

```python
def wrap_modules(self, modules_generator, prefix=""):
    modules = [self._maybe_offload_to_cpu(module, prefix)
               for module in modules_generator]
    ...

def _maybe_offload_to_cpu(self, module, prefix=""):
    if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
        return module                     # budget exhausted, keep on GPU
    ...                                   # else offload this module
```

It walks modules in **construction order** and offloads until a byte budget runs out. So it
offloads layers 0..25 and keeps 26..44 resident. That is the worst available choice: the
flat-routing layers, which caching cannot rescue, are put where caching is the only defence,
while VRAM is spent on the layers that would have survived being offloaded.

```
BEFORE   layer 0 ................... 25 | 26 ......... 44
                └── host, crosses PCIe ─┘ └── VRAM ────┘
                   (flat routing, 46% hit)   (cacheable, wasted here)

AFTER    layer 0 ......... 18 | 19 ................... 44
                └── VRAM ────┘ └── host, crosses PCIe ─┘
                 (flat, never                (cacheable,
                  touches PCIe)              ~85% hit)
```

---

## 5. The fix

Do **not** reorder by materialising the generator:

```python
mods = list(modules_generator)     # WRONG
```

`modules_generator` *constructs* each layer as it is pulled. The stock comprehension builds
one layer, offloads it (freeing its VRAM), then builds the next. Calling `list()` first
allocates all 42 layers on the GPU simultaneously — we hit **63.23 GiB then OOM** partway
through creating expert weights. Placement must be decided per module as it arrives:

```python
def wrap_modules(self, modules_generator, prefix=""):
    pfx = f"{prefix}." if prefix else ""
    out = []
    for module in modules_generator:        # stays lazy
        i = state["n"]; state["n"] += 1
        if i < SKIP:
            out.append(module)              # held resident on purpose
        else:
            out.append(self._maybe_offload_to_cpu(module, pfx))
    return out
```

`SKIP=19` was chosen so the offload boundary lands such that the **same number of layers**
(26) ends up on host, making the comparison clean. It is not tuned.

The returned list must keep original positions — the caller builds the layer stack from it
positionally.

---

## 6. Results — inconclusive

Both arms verified identical: 26 layers armed / 16 declined, 46 of 144 experts resident,
45.36 GiB offloaded. Only the *identity* of the offloaded layers differs.

Three interleaved container restarts per arm (arms alternated, because this box moves
±0.9 tok/s between starts):

```
arm                        n     mean      min      max    tok/s
skip=19 (deep offloaded)  12    77.75    74.68    82.87    12.86
   per-restart means: 75.09  76.01  82.15
skip=0  (vLLM default)    12    80.41    79.99    80.94    12.44
   per-restart means: 80.35  80.59  80.30

paired by rep:
  rep1  75.09 vs 80.35  -> deep    by 5.26 ms
  rep2  76.01 vs 80.59  -> deep    by 4.59 ms
  rep3  82.15 vs 80.30  -> default by 1.86 ms

overall -2.66 ms/step | 12.44 -> 12.86 tok/s (+3.3%) | 2 of 3 paired wins
worst deep restart 82.15 vs best default restart 80.30 -> OVERLAPS
```

**Verdict: not confirmed.** +3.3% is below the ±6 ms / ±0.9 tok/s floor established for this
machine, and the arms overlap.

### A retraction, and why it matters methodologically

An earlier draft of this document claimed the gain was ~7 ms (~9%) and cited the fact that
"the miss-rate arithmetic predicted 7.5 ms and measurement gave 7.3 ms" as independent
corroboration. **Both claims were wrong and are withdrawn.**

* The 7.3 ms came from a SINGLE container. It did not replicate: the paired mean is 2.66 ms.
* The 7.5 ms "prediction" was derived *after* seeing that measurement, then presented as
  a-priori. The genuine prior was only directional ("should cut traffic toward half").
* The estimate has three soft inputs — an assumed halving of misses, a bandwidth figure with
  a 21–25 GB/s range, and a baseline traffic number. Agreement to 3% from that should have
  been read as a warning sign, not as confirmation.

### The unexplained asymmetry — the interesting part

```
default arm: 80.35  80.59  80.30    spread 0.29 ms
deep arm:    75.09  76.01  82.15    spread 7.06 ms
```

If container-start variance explained the result, BOTH arms would scatter. One is tight and
the other is bimodal — near 75 twice, near 82 once. Three samples cannot distinguish a real
bimodality in the deep configuration from the default arm drawing three similar containers
by chance.

Worth checking: `_maybe_offload_to_cpu` breaks out of its per-parameter loop mid-module when
the budget is exhausted, so moving the boundary can leave a layer PARTIALLY offloaded. That
would be deterministic rather than run-to-run, so it does not obviously explain bimodality,
but the boundary behaviour differs between the arms and has not been inspected.

**Before anyone builds on this, run more restarts (5+ per arm) and find out whether the deep
arm is genuinely bimodal.** That question is more interesting than the 3.3%.

## 7. What is worth taking from this

Stated honestly, in two parts.

**The observation, which is solid and measured:**

> At a fixed cache budget, per-layer expert-routing concentration varies several-fold with
> depth — here 46% to 89% hit rate across layers. And vLLM (like most frameworks) decides
> which layers to offload by module construction order, not by any property of the layer.
> Those two facts together mean the placement decision is being made by accident.

**The hypothesis, which is NOT confirmed:**

> That choosing placement by measured hit rate is worth meaningful throughput. Tested here
> at +3.3% with overlapping distributions and 2 of 3 paired wins — promising, not proven.
> See §6.

Cheap to evaluate on your own stack regardless: dump per-layer routing ids for a few thousand
steps, replay your cache policy offline per layer, and rank by hit rate. That costs one
instrumented run and tells you whether the spread exists on your model. If it does not, the
idea is dead for you. If it does, the cost of exploiting it is a few lines at the offloader —
but budget for a proper A/B, because the effect size here did not clear the noise.

**The trap worth copying even if nothing else transfers:** do not reorder the module
generator to control placement (§5). It constructs layers as it is pulled, so materialising
it allocates the whole model on the GPU at once.

---

## 8. Caveats / not yet verified

* **The headline result is NOT confirmed** (+3.3%, overlapping distributions, 2 of 3
  paired wins). The per-layer hit-rate spread that motivates it IS solid; the
  performance gain is not.
* **Measurement floor**: this box moves ±0.9 tok/s between container starts (pinned buffer
  lands in different physical memory each time). Any claim below ~6 ms needs interleaved
  restarts, 3+ per arm, arms alternated. Single-container comparisons are worthless here.
* The routing trace covers the 17 layers armed under an *earlier* configuration. The
  gradient for layers 17–41 is **extrapolated**, not measured. A full 42-layer trace should
  be taken before tuning `SKIP` further.
* GLM alternates linear-attention (KDA) and sparse-attention layers, so a depth-ordered
  choice is not a clean random sample — some of the effect could track attention type rather
  than depth. Untested.
* `SKIP=19` is unswept. The gradient is steep enough that a better boundary likely exists.
* Related but much smaller: the cache gives every layer the *same* slot count, which is also
  not optimal. Greedy reallocation of the same total slots models at only **2.6%** fewer
  misses — below this box's measurement floor — and it pulls the *opposite* direction
  (shallow layers want more slots, since deep layers saturate). Not implemented.
