"""Is 18.2 GB/s the real ceiling for the gather's access pattern, or is it leaving ~6.7 ms?

MEASURED: the in-model gather moves 433 MB/step in 23.75 ms = 18.2 GB/s. An isolated
UVA copy on this box reached 26.9 GB/s. That 30% gap is ~6.7 ms/step -- the largest
remaining item in the decode budget.

Geometry is NOT the explanation: CHUNKS/LANES have been swept twice (16..1024 chunks,
4..64 lanes) and are flat within drift.

The untested difference is the ACCESS PATTERN. The isolated benchmark streams
sequentially through a pinned buffer. The real gather reads ~8 MiB expert blocks from
SCATTERED offsets across a ~45 GiB pinned region -- a different IOMMU/TLB workload
entirely. If scattered reads also land near 18 GB/s, the gather is already at its wall
and the lever is closed; if they hit 26, the gap is real and worth chasing.

Controls, in order of resemblance to the real thing:
  sequential      read consecutive slabs           (what the old benchmark did)
  scattered       read slabs at random offsets     (what the gather actually does)
  scattered_2x    two blocks per miss (w13 + w2)   (the real gather's shape)
"""
import random
import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@triton.jit
def _copy(src, dst, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(dst + o, tl.load(src + o, mask=m, other=0), mask=m)


# One expert's w13 is ~8.25 MiB and w2 ~4.125 MiB at int4; the pinned region the real
# offloader owns is tens of GiB, far past any TLB reach.
MIB = 1024 * 1024
W13 = int(8.25 * MIB) // 4096 * 4096
REGION_GB = 8                      # big enough to defeat TLB caching, small enough to pin
N_SLABS = REGION_GB * 1024 * MIB // W13

print(f"pinning {REGION_GB} GiB ({N_SLABS} slabs of {W13/MIB:.2f} MiB)...", flush=True)
host = torch.empty(REGION_GB * 1024 * MIB // 4, dtype=torch.int32, pin_memory=True)
uva = get_accelerator_view_from_cpu_tensor(host)
dev = torch.empty(W13 // 4, dtype=torch.int32, device="cuda")
print("pinned\n")

BLOCK = 1024
words = W13 // 4
grid = (triton.cdiv(words, BLOCK),)


def bench(offsets, iters=None):
    iters = iters or len(offsets)
    for i in range(8):
        base = offsets[i % len(offsets)]
        _copy[grid](uva[base:base + words], dev, words, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters):
        base = offsets[i % len(offsets)]
        _copy[grid](uva[base:base + words], dev, words, BLOCK=BLOCK, num_warps=4)
    b.record()
    torch.cuda.synchronize()
    ms = a.elapsed_time(b) / iters
    return ms, W13 / 1e9 / (ms / 1e3)


seq = [i * words for i in range(N_SLABS)]
rng = random.Random(0)
scat = [rng.randrange(N_SLABS) * words for _ in range(N_SLABS)]

print(f"{'pattern':>16} {'ms/slab':>9} {'GB/s':>8}   vs in-model 18.2")
for name, offs in (("sequential", seq), ("scattered", scat)):
    ms, gbs = bench(offs)
    print(f"{name:>16} {ms:>9.3f} {gbs:>8.2f}   {'<- matches gather' if abs(gbs-18.2) < 2 else ''}")

print()
print("If scattered ~= 18 GB/s, the gather is AT ITS WALL and the 6.7 ms is not real.")
print("If scattered ~= 26 GB/s, the gap is in the kernel and worth chasing.")
