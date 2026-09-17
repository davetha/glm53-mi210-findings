"""Per-step decode budget from a CUDA-GRAPH trace, windowed to steady state.

Two things this answers that the whole-run kernel_stats.csv cannot:

 1. WHERE the ~80 ms step goes, in the mode actually served (every prior profile in this
    project was --enforce-eager, which distributes time completely differently).
 2. Whether the GPU is BUSY. Summing kernel durations gives occupancy, not wall time --
    if busy << wall there are bubbles, and bubbles are a scheduling problem, not a kernel
    problem. That distinction decides what kind of optimisation is even applicable.

Step clock = all-reduce count / 90. The model does exactly 90 all-reduces per decode step
(45 layers x 2: attention out-proj and MoE down-proj), which is a far more reliable step
marker than wall-time guessing.

Note on overlap: kernels can overlap across streams, so summed duration can exceed wall
time. Busy is therefore computed as the union of occupied intervals, not the sum.
"""
import collections
import csv
import sys

path = sys.argv[1]
lo_s = float(sys.argv[2])
hi_s = float(sys.argv[3])
AR = "ncclDevKernel"


def bucket(n):
    if AR in n:                                   return "all-reduce (NCCL)"
    if "expert_cache_gather" in n:                return "expert gather (PCIe)"
    if "fused_moe_kernel" in n:                   return "MoE expert GEMM"
    if "wvSplitK" in n or "Cijk" in n:            return "dense GEMM"
    if "mhc_" in n:                               return "mHC hyper-connections"
    if "sparse_attn" in n or "_fwht" in n:        return "sparse attn + indexer"
    if "kda" in n.lower() or "chunk_" in n:       return "KDA linear attention"
    if "topk" in n or "sigmoid_bias" in n:        return "routing / top-k"
    if "direct_copy" in n or "copyBuffer" in n or "CatArray" in n:
        return "tensor copies"
    if "elementwise" in n or "vectorized_elementwise" in n:
        return "elementwise"
    if "reduce_kernel" in n or "norm" in n.lower():
        return "reductions / norms"
    return "other"


rows = []
with open(path, newline="") as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
t0 = rows[0][0]
lo = t0 + int(lo_s * 1e9)
hi = t0 + int(hi_s * 1e9)
win = [(s, e, n) for s, e, n in rows if s >= lo and e <= hi]

steps = sum(1 for _, _, n in win if AR in n) / 90.0
# Wall must be the span the kernels actually OCCUPY, not the requested window: if decode
# does not fill the window, the empty edges are counted as idle and the step time is
# inflated. (This bug reported 17.5 ms/step idle where the true figure is ~5.3.)
wall_ms = (max(e for _, e, _ in win) - min(s_ for s_, _, _ in win)) / 1e6
print(f"window {lo_s:.1f}-{hi_s:.1f}s   {len(win)} dispatches   "
      f"{steps:.1f} decode steps   wall {wall_ms:.0f} ms   "
      f"=> {wall_ms/steps:.2f} ms/step ({1000*steps/wall_ms:.2f} tok/s)\n")

# union of occupied intervals = real busy time (kernels can overlap across streams)
iv = sorted((s, e) for s, e, _ in win)
busy = 0
cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        busy += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
busy_ms = busy / 1e6
print(f"GPU busy {busy_ms:.0f} ms of {wall_ms:.0f} ms wall = {100*busy_ms/wall_ms:.1f}%   "
      f"=> {(wall_ms-busy_ms)/steps:.2f} ms/step IDLE\n")

agg = collections.Counter()
cnt = collections.Counter()
for s, e, n in win:
    agg[bucket(n)] += e - s
    cnt[bucket(n)] += 1
tot = sum(agg.values())
print(f"{'ms/step':>9} {'%busy':>7} {'calls/step':>11}  bucket")
for b, ns in agg.most_common():
    print(f"{ns/1e6/steps:>9.2f} {100*ns/tot:>6.1f}% {cnt[b]/steps:>11.1f}  {b}")
print(f"{tot/1e6/steps:>9.2f} {'100.0%':>7} {sum(cnt.values())/steps:>11.1f}  TOTAL (summed, overlaps counted)")

print("\ntop individual kernels in window:")
k = collections.Counter()
kc = collections.Counter()
for s, e, n in win:
    k[n[:96]] += e - s
    kc[n[:96]] += 1
for n, ns in k.most_common(15):
    print(f"{ns/1e6/steps:>9.3f} ms/step {kc[n]/steps:>8.1f} calls/step  {n[:80]}")
