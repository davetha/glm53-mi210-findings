# Task 1.2 — host-staged all-reduce in vLLM: integration log

**Result: shipped and working. ~3.5 ms/step (~5%), not the ~16 ms the microbenchmark
implied. The gap is the finding: in-model, the all-reduce protocol was never the dominant
cost — RANK SKEW is, and no protocol change can recover that.**

## What was built

| piece | status |
|---|---|
| POSIX shm arena (`shm_open` + `mmap(MAP_SHARED)` + `hipHostRegister`) | works across two processes |
| communicator patch via `vllm.general_plugins` entry point | loads in every TP worker |
| threshold routing (<=64 KB custom, else NCCL) | intercepts decode `(1,4096)` bf16 |
| CUDA-graph capture compatibility | verified: 19,928 `hostar_k` calls in a FULL_DECODE_ONLY trace |
| numerical correctness | 4/4 exact-answer gate in-model |

Files: `hostar.hip` → `libhostar.so`, `hostar.py`, `hostar_dist/` (entry point),
`hostar_run.sh` (A/B), `hostar_shm.hip` (two-process standalone proof).

## The measurement that matters

Kernel trace of a CUDA-graph decode, hostar active (confirmed by `hostar: USED` in the
traced container's own log — see trap 3):

```
hostar_k   19928 calls   2422.0 ms   121.54 us/call     (decode all-reduces)
nccl         455 calls    305.4 ms   671.15 us/call     (tensors over the 64 KB threshold)

221.4 decode steps  ->  hostar all-reduce = 10.94 ms/step
per-call:  p50 13.60 us   p90 404.80 us   p99 1584.17 us   max 5596.85 us
```

Against NCCL measured the same way (`DECODE-COMPUTE-BUDGET.md`, prefill-free window):
**14.45 ms/step → 10.94 ms/step, saving ~3.5 ms (~5% of a 70 ms step).**

## Why it is 3.5 ms and not 16

The protocol is as fast as advertised: **p50 13.60 µs** in-model, against 6.73 µs standalone
(the difference is contention and dispatch in a 2,700-kernel step).

But p90 is 405 µs and p99 is 1584 µs — 30× and 116× the median. That tail is **not** the
barrier being slow. It is the kernel spinning while the OTHER rank catches up. Under expert
parallelism the two ranks route to different experts and therefore do different amounts of
PCIe gather work per layer, so they arrive at each layer's all-reduce at different times.
The early rank waits, and the trace bills that wait to `hostar_k`.

NCCL pays exactly the same wait — it is buried inside its 185.68 µs average. So replacing
the protocol recovers the protocol's share and nothing else:

| | ms/step |
|---|---|
| NCCL all-reduce, measured | 14.45 |
| hostar all-reduce, measured | 10.94 |
| **recovered (protocol)** | **3.51** |
| irrecoverable (rank skew) | the remainder |

**Consequence for anyone continuing this:** the remaining all-reduce time is a LOAD
BALANCING problem, not a communication one. Reducing per-rank gather variance (expert
placement, slot allocation) would cut it; a faster barrier cannot.

## Traps, each of which produced a confidently wrong conclusion

**Trap 1 — `.pth` + meta-path finder never fired.** `find_spec` called
`importlib.util.find_spec(fullname)`, which imports the PARENT package while already inside
an import; that raises, and a broad `except` turned it into a silent fall-through. Symptom:
everything looks healthy, patch never installs. Fix: `PathFinder.find_spec(fullname, path)`
using the `path` already supplied — or better, use the `vllm.general_plugins` entry point,
which vLLM calls from `worker_base.py` *inside each worker*. A parent-side import never
reaches the workers; they are separate processes.

**Trap 2 — `set -o pipefail` + `grep -q` inverts the test.** `docker logs | grep -q X` makes
grep exit on first match and close the pipe; `docker logs` then dies on SIGPIPE, and
pipefail propagates that, so a SUCCESSFUL match reports failure. My install gate aborted a
good run. Count matches into a variable instead.

**Trap 3 — tracing a container that was not the configuration under test.**
`profile_graph.sh` hardcodes `EXTRA_DOCKER_ARGS`, silently dropping anything passed as
`EXTRA_DOCKER`. The resulting trace showed `hostar_k` absent, and I concluded the kernel
never ran — when in fact the traced container simply had no hostar in it. **Always confirm
the feature is live in the traced container from that container's own log before reading
anything off the trace.**

**Trap 4 — gating on a signal that does not imply what you want.** I gated on "plugin
installed" + "arena built". Both were true while the kernel ran zero times, because the
arena is constructed BEFORE `should_use()` is consulted. The only signal that ever meant
"the kernel ran" was `hostar_k` in a kernel trace, or an explicit use counter.

**Trap 5 — logging the first N decisions.** The first calls are all from the startup
profiling run (64.9 MB tensors), so a small cap never shows what DECODE passes. Histogram
every decision instead.

## Design notes worth keeping

* **Sequence counter lives in device memory**, incremented by the kernel. Under
  FULL_DECODE_ONLY the kernel arguments are baked at capture, so a host-side counter would
  replay one value forever and every exchange after the first would see an
  already-satisfied flag.
* **Data and flag must travel the same path.** An earlier design put data in peer VRAM and
  the flag in host memory; different PCIe routes, nothing orders them, so the peer can
  observe the flag and read stale staging.
* **Timeout traps rather than returning.** No Python runs per step under graphs, so nothing
  can check a status flag; a timed-out all-reduce returning quietly would feed stale
  staging into the next layer as a plausible activation.
* **`hipHostRegister` accepts an shm-backed mapping**, and both processes' device pointers
  alias the same pages — verified by arithmetic (rank 0 contributes 1.0, rank 1 contributes
  2.0, both read 3.0) before any vLLM code was written.

## Enabling

```
VLLM_HOSTAR=1                      # off by default; everything falls back to NCCL
VLLM_HOSTAR_MAX_BYTES=65536        # above this, NCCL
VLLM_HOSTAR_SPIN_SEC=10            # spin budget before the kernel traps
```
Requires `hostar.py`, `libhostar.so` and `hostar_dist/` (as `hostar-0.1.0.dist-info`) on
the Python path / in site-packages.
