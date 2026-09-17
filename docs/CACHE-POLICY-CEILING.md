# Belady's optimal, and why it closes the expert-cache policy question

Written because a proposed "future-use predictor" design raised the question of how much
a smarter cache policy could possibly be worth here. Belady answers that exactly.

---

## 1. The algorithm

From László Bélády's 1966 IBM paper on virtual-storage replacement. Also called **MIN** or
**OPT**. The rule is one line:

> **On a miss, evict the item whose next use is furthest in the future.**

The rule is trivial. The catch is that applying it requires knowing the future, which is
why it is called *clairvoyant* and cannot be implemented in a live cache. It is computed
offline, by replaying a recorded access trace.

## 2. Why it is provably optimal

An exchange argument. Let A be any optimal algorithm, and find the first moment A disagrees
with OPT. A evicts item `x`; OPT evicts item `y`, the one whose next use is furthest ahead.

Construct A' that evicts `y` at that moment, then imitates A from there. Because `y` is not
needed for at least as long as `x` is, A' can always match A's later choices, and it incurs
no more misses than A. So A' is no worse than A, and it agrees with OPT one step longer.

Repeat at each divergence. A is transformed into OPT without the miss count ever rising.
Therefore nothing beats OPT. ∎

**The consequence that matters:** OPT *is* a perfect future-use predictor. So the gap
between your online policy and OPT is the entire headroom available to ANY predictor,
model, or heuristic, no matter how sophisticated. It is not an estimate — it is a bound.

## 3. What the policies actually differ on

| policy | evicts by | direction |
|---|---|---|
| LRU | least recently used | backward |
| LFU | least frequently used | backward |
| LRFU | recency/frequency blend | backward |
| **OPT** | **furthest next use** | **forward, exact** |

LRU and LFU both guess the future from the past. LFU does respectably on MoE expert routing
because expert popularity is genuinely skewed and reasonably stable, so a frequency count is
a decent proxy.

What frequency cannot represent is **timing**. LFU keeps a high-count expert even when it
will not be touched for hundreds of steps, because a counter has no concept of *when*.
Frequency answers "how often." The question that decides a cache is "how soon."

## 4. Measured on GLM-5.3-Flash (12,600 refreshes, 51,114 requests, 17 layers, S=46)

```
   S      LRU     LFU     ARC    LIRS    LRFU     OPT   best hit  OPT hit    gap
  46    17819   16881   17451   17602   16512    9113      67.7%    82.2%  14.5pp
```

Every online policy lands within ~1.5% of every other. The interesting distance is not
between them, it is the **14.5 pp to OPT**.

### Where that 14.5 pp comes from: reuse distance

Requests between two uses of the same expert, over 48,784 reuses:

| percentile | distance |
|---|---|
| p10 | 4 |
| p25 | 8 |
| p50 | 35 |
| p75 | 116 |
| p90 | 278 |
| p95 | 453 |
| p99 | 1,007 |
| max | 3,129 |

**~30× spread between median and p99.** A frequency counter collapses that whole range into
one integer. LFU cannot tell an expert due again in 4 requests from one due in 1,007 when
their counts are similar, so it keeps both. OPT evicts the distant one immediately and
spends the slot on something imminent. That is the gap, in one sentence.

(For scale: ~93% of reuses fall within the rough reach of a 46-slot cache, yet even OPT
only reaches 82.2% — because more experts want residency inside their reuse window than
there are slots. That reach figure is a crude LRU-flavoured approximation.)

## 5. What this is worth in wall-clock

Closing the *entire* gap, with a flawless predictor:

```
misses/layer-step   1.3  ->  0.7
PCIe per step       ~360 MiB -> ~200 MiB
at the 24 GB/s this link achieves   ~6.5 ms of an ~80 ms step
                                    =  ~8% ceiling
```

A real predictor captures a fraction of that.

## 6. Evaluating a future-use predictor design

A proposed architecture: routing history → predictor emitting {probability,
horizon/deadline, confidence} → feeding CACHE (eviction, admission, pinning, reservation,
early evict), PCIe (DMA priority, BW budget, prefetch depth, batching), and POLICY
(confidence, fallback).

**The CACHE branch is bounded by §5: ~8% ceiling, realistically far less.**

**The PCIe branch is a different axis** — prefetch hides latency rather than reducing bytes,
so it is not bounded by OPT. But it depends on the misses being anticipatable, and they are
not:

```
21.3%  of a step's experts were also used in the previous step
 5.6%  of MISSES were experts used in the previous step
```

The misses are, nearly by definition, what recent history failed to anticipate. Worse,
routing is decided layer-by-layer *during* the step, so for layer 30 there is no signal at
all until layer 30's router runs. There is no lead time to prefetch into.

**The POLICY branch** is sound engineering, but confidence-gated fallback only matters once
the predictor is earning something.

**Verdict for this workload:** well-conceived design, wrong workload. It chases at most 8%
using a signal that is 5.6% informative about the events that cost anything. On a model with
higher temporal locality in routing it could pay — which is worth measuring first, cheaply,
with the two numbers above.

## 7. Belady's anomaly (different result, same author)

For FIFO, *increasing* cache size can *increase* misses. OPT and LRU are immune because they
are **stack algorithms**: the contents of an n-slot cache are always a subset of an
(n+1)-slot cache. FIFO has no such property. Worth knowing if anyone proposes a FIFO-ish
admission scheme and reports a surprising slot sweep.

## 8. Why compute an unimplementable algorithm

To know when to stop. The 14.5 pp bound is what closed the policy question in this project:
ARC, LIRS, LRFU and tuned-decay LFU all landed within 1.5% of each other and nowhere near
the ceiling, which says the remaining gap is not about cleverness but about information
nobody online possesses.

Any time a policy change is proposed, compute OPT on a real trace first. If the gap is
small, the work is capped before it starts.

**Reproduce:** `cache_sim.py <trace.json> <slots>` replays LRU/LFU/ARC/LIRS/LRFU/OPT on a
`GLM53_TRACE_DUMP` routing trace. OPT is O(n²) in the naive form, so sample long traces.
