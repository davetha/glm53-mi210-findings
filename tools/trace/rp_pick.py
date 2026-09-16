"""Per-step call count and busy time for a named set of kernels, from a rocprof trace.

Same step delimitation as rp_step2.py, but reports chosen kernels regardless of how long
each takes, so a kernel that sits above the small-kernel threshold is still comparable
across an A/B.
"""
import csv, collections, sys

path, pats = sys.argv[1], sys.argv[2].split(",")
MARKER = "moe_align_block_size"
rows = []
with open(path, newline="") as fh:
    for r in csv.DictReader(fh):
        if r["Kind"] == "KERNEL_DISPATCH":
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
marks = [i for i, (_, _, n) in enumerate(rows) if MARKER in n]
gaps = [(rows[marks[i + 1]][0] - rows[marks[i]][1], i) for i in range(len(marks) - 1)]
cut = sorted(g for g, _ in gaps)[int(len(gaps) * 0.97)]
bounds = [marks[i + 1] for g, i in gaps if g >= cut]
k = int(len(bounds) * 0.75)
step = rows[bounds[k]:bounds[k + 1]]
busy = sum(t1 - t0 for t0, t1, _ in step) / 1000.0
print("step: {:,} kernels, {:,.1f} us busy".format(len(step), busy))
for p in pats:
    c = 0
    u = 0.0
    for t0, t1, n in step:
        if p in n:
            c += 1
            u += (t1 - t0) / 1000.0
    print("{:>6} calls {:>9.1f} us  {}".format(c, u, p))
