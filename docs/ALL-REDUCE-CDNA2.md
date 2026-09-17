# Custom all-reduce on gfx90a (MI210): it compiles, it runs, and it CORRUPTS

> **SUPERSEDED IN PART — read `P2P-PEER-HANDSHAKE.md` next.**
> The corruption described below was later FIXED (the kernel was made numerically correct
> on gfx90a), and it still did not beat NCCL. The closing estimate here — "~90 us per call
> on the table, ~8 ms/step" — did **not** hold: that 20.5 us figure is a `hipMemcpyPeer`
> copy-engine number, and a kernel-resident spin handshake, which is what a custom
> all-reduce actually needs, could not be made to work at all. Final verdict: abandoned.

**Do not enable custom all-reduce on CDNA2.** Not vLLM's, not AITER's. The upstream
MI300-only allowlist is protecting against real silent data corruption, not being
conservative. This documents the full chain so nobody repeats it.

## Why anyone would try

On 2x MI210 with no XGMI bridge, the cards sit on separate IO-die quadrants, so every
tensor-parallel reduction crosses PCIe. Measured from a CUDA-graph rocprofv3 trace of a
real decode step:

```
ncclDevKernel_Generic_4   92 calls   10.46 ms   17.0% of an 80 ms step
```

92 calls at 113.7 us each. The tensors are ~8 KB, so this is ~70 MB/s -- pure latency, not
bandwidth. A one-shot peer-write reduction should beat it comfortably. That is the prize:
~10.5 ms, larger than anything else left on this machine.

## Every gate in the way, and how to open it

Four independent blocks, all excluding CDNA2:

| backend | gate |
|---|---|
| vLLM `CUSTOM` | `RocmPlatform.use_custom_allreduce()` returns `any(gfx in _GCN_ARCH for gfx in ["gfx94","gfx95"])` |
| `QUICK_REDUCE` | same allowlist, **and** `_QR_MIN_SIZE` is 2 MB for bf16/world=2 -- decode reductions are 8 KB |
| `AITER_CUSTOM` | `is_aiter_found_and_supported()` -> `get_cdna_version() > 2`; gfx90a is CDNA **2** |
| `NCCL` | none; this is what you run |

Note `use_aiter_allreduce = use_custom_allreduce and is_custom_all_reduce_enabled()`, so the
MI300 allowlist gates the AITER path too. Opening one alone does nothing.

To reach AITER's kernel you need all of:
1. `use_custom_allreduce()` widened to include gfx90a
2. `is_aiter_found_and_supported()` -> `get_cdna_version() > 2 or on_gfx90a()`
3. `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_CUSTOM_AR=1`
4. FP8 conversion helpers in `aiter_meta/csrc/include/opus/opus.hpp` guarded (below)

## The build blocker, and why it is not the real problem

AITER JIT-compiles `module_custom_all_reduce` at first use. On gfx90a it fails:

```
opus.hpp:1311: error: '__builtin_amdgcn_cvt_pk_fp8_f32' needs target feature fp8-conversion-insts
opus.hpp:1322: error: '__builtin_amdgcn_cvt_pk_f32_fp8' needs target feature fp8-conversion-insts
```

Four errors, none in the reduction. They are FP8 conversion templates in a transitively
included header; clang validates builtin target features while analysing a `__device__`
function even for templates nothing instantiates. CDNA2 has no FP8 ALU -- per the
aiter-cdna2 port matrix, `v_cvt_pk_fp8_f32` does not assemble for gfx90a at all.

Guarding the block (lines 1304-1324) behind `#if defined(__gfx942__) || defined(__gfx950__)
|| defined(__gfx1250__)` with trapping stubs makes it compile cleanly: exit 0, 0 errors,
3 MB object. **This works.** It is also a trap, because what comes next is worse.

## The actual result: silent, non-deterministic corruption

Tiny MoE (422 MB, w8a16), TP=1 as ground truth since one GPU performs no reduction at all.
16 tokens, temperature 0, two runs per arm:

```
tp1          run1: '겅' 'ayı' ' glossy' ',args' '尺度' '_timing' ...
             run2: identical
nccl         run1: '겅' 'ayı' ' glossy' ',args' ' الإسلامية' 'categories' ...
             run2: identical
vllm CUSTOM  run1: (same as nccl)
             run2: identical
AITER_CUSTOM run1: ' states' 'ItemImage' ' consultation' ' sorts' 'apo' ...
             run2: ' setLoading' 'ATURE' 'MOVED' '-LAST' '藩' ...        <-- DIFFERENT
```

Two separate failures:

* **Diverges at token 1**, not token 5. TP=2 arms normally agree with TP=1 for several
  tokens before reduction-order drift flips one; this is wrong immediately.
* **run1 != run2.** Every other arm repeats exactly. Non-determinism is the signature of a
  synchronisation race -- peers reading each other's buffers before the writes are visible.

That is an Infinity Fabric ordering assumption failing over PCIe peer-to-peer. MI300 has
the fabric; MI210 does not.

**The failure is silent.** No crash, no error, no warning -- fluent plausible tokens that
are simply wrong, and differently wrong each run. This is the most dangerous possible
failure mode and it is why the allowlist exists.

## The two kernels fail differently

* **vLLM's** refuses to launch: `custom_all_reduce_hip.cuh:167 'invalid argument'` during
  CUDA graph capture, on both workers. Its init path uses IPC memory handles
  (`hipIpcGetMemHandle`/`hipIpcOpenMemHandle`), which is a different capability from the
  peer access that `can_device_access_peer` reports -- vLLM runs TP workers as separate
  processes. Failing loudly is the safe outcome.
* **AITER's** runs and corrupts. Strictly worse.

## If someone still wants this

The hardware primitive is there: peer access is bidirectional at 23.7 GB/s, measured. What
is missing is the ordering guarantee a spin-wait reduction needs. Any attempt must

1. treat TP=1 as ground truth and compare token sequences, not just check for crashes;
2. run each arm at least twice -- this bug is invisible in a single run;
3. assume silent corruption until proven otherwise, because that is what it does.

Measured facts for anyone sizing the work: P2P 4 KB one-way is 20.5 us; NCCL's reduction of
the same payload is 113.7 us. So roughly 90 us per call is on the table, ~8 ms/step -- but
only with correct synchronisation, which is the entire difficulty.
