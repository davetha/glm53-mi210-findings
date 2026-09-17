# GPU-side peer handshakes on 2× MI210 (gfx90a): why the custom all-reduce cannot pay

**Verdict: abandoned. Zero tok/s gained. The negative result is the deliverable.**

This continues `ALL-REDUCE-CDNA2.md`, which ended at "it compiles, it runs, and it
CORRUPTS". That corruption was subsequently **fixed** — the kernel was made numerically
correct on gfx90a — and it still did not beat NCCL. This document covers everything after
that point, including three of my own bugs, because each one is a trap the next person
will hit.

Hardware: 2× MI210 (gfx90a/CDNA2), PCIe Gen4 x16, **no XGMI/Infinity Fabric between the
cards**, on separate root complexes. vLLM 0.28.1rc0+mi210.7, AITER, GLM-5.3-Flash W4A16,
TP=2 + `--enable-expert-parallel`, single-stream decode.

---

## 1. Why this looked worth doing

Decode step ≈ 80 ms. Measured split:

| component | ms | share |
|---|---|---|
| PCIe expert gather | 27 | 34% |
| **all-reduce (NCCL)** | **10.46** | **13%** |
| everything else (compute) | ~42 | 53% |

Removing the all-reduce entirely would be 12.44 → ~14.3 tok/s. Upstream restricts the
AITER custom all-reduce to an `["gfx94", "gfx95"]` allowlist, i.e. MI300+. The question
was whether that allowlist guards a *hardware* limitation or merely an untested path.

**Answer: it guards both a real memory-ordering bug and a topology assumption.** Neither
is fixable into a win on this hardware.

---

## 2. The arc, in order

1. `use_custom_allreduce()` widened for gfx90a behind `VLLM_ALLOW_CUSTOM_AR_GFX90A`.
2. `is_custom_all_reduce_enabled()` exempted narrowly, bypassing `@if_aiter_supported`.
3. FP8 helpers in `opus.hpp` guarded so the JIT builds on CDNA2 (`__builtin_trap()` stubs).
4. Peer handshake given release/acquire at system scope → **numerically correct**.
5. Acquire-per-poll → ~91 ms/step, **11 ms worse than NCCL**.
6. Relaxed poll + one acquire fence → 82.1–82.5 ms… **plus a 131.69 ms outlier in 4 runs**.
7. `glc slc` per-access bypass → **numerically broken, 0/3** (§5).
8. Paired A/B → **noise floor 8.9 ms**, four times the effect being chased (§6).
9. P2P microbenchmark → **no working GPU-side handshake at all** (§8).

---

## 3. The decisive ISA facts (reusable; get these right before writing anything)

Compiled for `gfx90a`, `-O3 -S`. **`USE_ROCM` must be defined** or you silently compile the
CUDA branch and measure nothing — this cost me one full round trip.

| source construct | emitted on gfx90a |
|---|---|
| `__scoped_atomic_load_n(RELAXED, SYSTEM)` | `global_load_dword … glc` |
| `__scoped_atomic_load_n(ACQUIRE, SYSTEM)` | `global_load_dword … glc` + `buffer_invl2` + `buffer_wbinvl1_vol` |
| `__scoped_atomic_store_n(RELEASE, SYSTEM)` | store preceded by `buffer_wbl2` |
| `__builtin_nontemporal_store` | `global_store_dword … glc slc` |
| `__builtin_nontemporal_load` | `global_load_dword … glc slc` |
| inline asm `global_load_dword … glc slc` | as written |

Three consequences:

* **`ACQUIRE` inside a spin loop invalidates the entire 8 MB L2 on every poll iteration.**
  That is where the 11 ms went. The all-reduce sits between layers, so each invalidate also
  evicts the activations and resident expert weights the next layer reads — the aftermath
  costs more than the instruction.
* **`RELAXED` emits `glc` but not `slc`, so it polls L2**, which on CDNA2 is not coherent
  with a peer's PCIe write. The variant that passed correctness was **correct by luck**.
* **`__builtin_nontemporal_load` is not volatile.** In a spin loop the compiler hoists it
  clean out and the loop never re-reads. A polling loop *must* use inline asm. The
  nontemporal builtin is fine for straight-line payload reads.

Also worth recording: the builtin only emits `glc slc` when the address is thread-varying.
With a uniform address the compiler uses `s_load` and you see neither bit — an easy way to
convince yourself a probe "proved" something it did not.

---

## 4. What the release-side writeback is for (do not remove it)

The producer's data is dirty in **its own** L2, and the peer reads its **HBM** over PCIe.
So `buffer_wbl2` on the release store is load-bearing: without it the peer reads stale
memory. Only the **acquire-side `buffer_invl2` is removable**, and it is the destructive
one — invalidate drops clean lines, writeback does not.

---

## 5. BUG (mine): removing a fence from a barrier shared by 46 call sites

The `glc slc` patch did exactly what was intended, verified on the instantiated
`cross_device_reduce_1stage<bf16,2,false>`:

| | `buffer_invl2` | `buffer_wbinvl1` | `buffer_wbl2` | `glc slc` loads |
|---|---|---|---|---|
| before | 1 | 1 | 1 | 0 |
| after | **0** | **0** | 1 (kept) | **2** |

**And it produced garbage: 0/3 on the correctness gate, emitting `0 and 0. 0. 0. 0.0…`.**

Cause: `start_sync` / `end_sync` are shared by **46 call sites across ~16 kernels** —
`cross_device_reduce_{1,2}stage{,_naive}`, every `allgather_*`, every
`reduce_scatter_*`, `allreduce_mhc_post_large_m_kernel`, `allReduceQuantFp8`. I removed the
acquire fence from the shared barrier but only converted the payload reads in the **one**
kernel I happened to be reading. Every other consumer — including the allgather and
reduce-scatter paths that `--enable-expert-parallel` uses for MoE dispatch/combine — then
read peer data through a stale L2.

> **Trap:** these barriers are shared infrastructure. Removing their fence is only sound if
> *every* consumer is converted to uncached reads. That is ~16 kernels, not a header patch.

Reverted. Broken version kept as `patched_custom_all_reduce.cuh.slc_BROKEN`.

---

## 6. The measurement was never capable of resolving this

Paired A/B, both arms launched back to back in one script, 8 warm-up generations discarded,
12 kept:

```
nccl   80.12 79.92 79.46 79.63 82.48 88.37 83.57 86.01 84.03 84.67 83.43 84.03
       median 83.50   min 79.46   max 88.37   drift +4.0 ms  (NOT CONVERGED)
```

**Within-arm spread: 8.91 ms.** The fence work was chasing ~2 ms.

Two consequences:

* The **80.41 ms "NCCL baseline"** used for most of this investigation was a favourable
  sample from a distribution spanning 79.5–88.4. It is not a number. (The ledger had
  already recorded this exact trap once, for the 96–97 ms standing config.)
* An earlier unpaired run descended monotonically 96.9 → 82.7 and was **still falling** at
  the last sample. Its median described the warm-up, not the server.

> **Trap:** on this box, any all-reduce change worth under ~9 ms **cannot be demonstrated by
> end-to-end tok/s**, at any sample size. It needs a direct all-reduce microbenchmark.

---

## 7. The geometry: 370× off bandwidth

`hidden_size` 4096, 45 layers, batch 1 → **90 all-reduces per decode step**, payload
4096 × bf16 = **8 KB** each, **720 KB per step total**.

| | |
|---|---|
| 720 KB at the link's measured 26 GB/s | **28 µs** |
| NCCL actually spends | **10,460 µs** |
| ratio | **≈ 370×** |
| per exchange | **116 µs for 8 KB** |

The all-reduce uses ~0.3% of the link. It is **not partly** latency-bound, it is *entirely*
latency-bound. Every fence variant landed at NCCL's number because cache maintenance is a
few µs inside a 116 µs protocol cost.

This also reframes an earlier note in `ALL-REDUCE-CDNA2.md` ("P2P 4 KB one-way is 20.5 µs;
NCCL 113.7 µs; ~90 µs on the table"). That 20.5 µs was a **`hipMemcpyPeer` / copy-engine**
figure. Copy-engine transfers work fine. What does not work is a **kernel-resident spin
handshake**, which is what any custom all-reduce requires — see below.

---

## 8. The P2P microbenchmark: the primitive itself does not hold

Purpose: bound the floor before writing a push-based kernel (posted writes instead of
AITER's non-posted peer *reads*, one barrier instead of two). Source: `p2p_floor.hip`.

**What works:**

```
peer capable: 0->1 1   1->0 1
peer access enabled both ways
--- test 0: does a peer write land? ---
  dev0 -> dev1 : wrote 0xABCD, peer reads 0xABCD  OK
  dev1 -> dev0 : wrote 0xABCE, peer reads 0xABCE  OK
```

A peer write lands and reads back correctly **when the writing kernel has ended**.

**What does not work — the protocol matrix.** Can one GPU observe the other's write while
**both kernels are running**? Every combination, 50 round trips, ~1–4 s budget each:

```
  store=glc+slc  load=glc+slc   NOT VISIBLE (stalled)
  store=glc+slc  load=glc       NOT VISIBLE (stalled)
  store=glc+slc  load=acquire   NOT VISIBLE (stalled)
  store=release  load=glc+slc   NOT VISIBLE (stalled)
  store=release  load=glc       NOT VISIBLE (stalled)
  store=release  load=acquire   NOT VISIBLE (stalled)
```

**6/6 stall. There is no memory-ordering flavour that fixes it.**

And critically, it is **intermittent, not absent**:

```
  warmup       BAILED at iter 1/100
  rt_flag      BAILED at iter 11/2000
  rt_push_8k   BAILED at iter 1/2000
  rt_pull_8k   BAILED at iter 390/2000
```

389 successful round trips, then stall. Writes become visible mid-kernel for a while, then
stop.

**This independently corroborates the live-model result:** the AITER custom all-reduce,
once numerically correct, threw a **131.69 ms outlier** in four runs. Same signature —
works, then stalls — observed once in a microbenchmark and once in the real model.

That is almost certainly also why NCCL costs 116 µs: it is *not* naively spinning on peer
flags over PCIe, because that does not hold up without a fabric.

---

## 9. Conclusion

The push rewrite rests entirely on a reliable GPU-side peer flag handshake. **That
primitive cannot be demonstrated to work on this hardware in 40 lines**, so building a
kernel on it is a bad bet. The all-reduce direction is closed.

**Caveat, stated plainly:** I cannot fully separate "my benchmark is buggy" from "the
hardware will not sustain this" — there were three bugs in that file (block-buffered stdout
hiding a hang; a 2e9-poll watchdog sized for cached loads, ~2000 s in practice; a
count-based cap that could not outlast a cold code-object load, fixed with an
`s_memrealtime` time bound). But the 131 ms model-level outlier is independent evidence
pointing the same way.

**What would be needed to reopen it**, in order:
1. A GPU-side peer handshake that sustains ≥10⁵ round trips without stalling. Until that
   exists, nothing downstream matters.
2. A direct all-reduce microbenchmark, because the 8.9 ms end-to-end noise floor cannot
   resolve anything smaller.
3. Only then a push kernel — which additionally needs a staging buffer added to the
   `Signal` struct (it currently holds only `start`/`end`/`_flag`), i.e. host-side
   allocation changes, not a header patch.

---

## 10. Traps, collected

| trap | cost |
|---|---|
| Compiling the header without `-DUSE_ROCM` silently builds the CUDA branch | one full round trip; before/after looked identical |
| `__builtin_nontemporal_load` is not volatile — hoisted out of spin loops | a spin that never re-reads |
| The builtin only emits `glc slc` for thread-varying addresses | a probe that "proves" nothing |
| `start_sync`/`end_sync` are shared by 46 call sites | silent garbage output |
| Block-buffered stdout in a program that can hang | zero output, no diagnosis |
| Count-based spin watchdog on uncached PCIe reads | ~2000 s "watchdog"; use `s_memrealtime` |
| Comparing arms across launches | 8.9 ms spread swamps a 2 ms effect |
| Correctness gate with small `max_tokens` | this model emits into `reasoning` until it closes; `content` is `''` and two empty strings compare equal. **Assert non-empty.** |

## Files

All under the bring-up directory (`glm53-bringup/`):

* `p2p_floor.hip` — P2P handshake probe: peer-write visibility, protocol matrix, timed ping-pongs
* `isa_probe.hip` — what each spin construct emits on gfx90a
* `patched_custom_all_reduce.cuh.slc_BROKEN` — the `glc slc` patch (0/3 correctness)
* `patched_custom_all_reduce.cuh.prefence` — the correct relaxed-spin version, restored
* `ar_paired.sh` — paired A/B harness with warm-up discard and convergence check
