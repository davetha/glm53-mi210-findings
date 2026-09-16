"""One steady-state decode step out of a rocprofv3 kernel trace, bucketed by kernel.

Step boundaries come from a marker kernel that fires a fixed number of times per step
rather than from wall-clock windows, so a host stall mid-step does not split one step in
two. A step from the last quarter of the run is used: the expert cache is warm and JIT is
done by then, and a cold step reads ~10% slow.

Prints the step's kernel count and busy time, then the small-kernel block (mean duration
under the threshold, default 20 us) ranked by total busy time. That block is what kernel
fusion targets.
"""
import csv, collections, sys

path = sys.argv[1]
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
MARKER = "moe_align_block_size"          # once per MoE layer per step

rows = []
with open(path, newline="") as fh:
    for r in csv.DictReader(fh):
        if r["Kind"] != "KERNEL_DISPATCH":
            continue
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
print("{:,} dispatches in trace".format(len(rows)))

marks = [i for i, (_, _, n) in enumerate(rows) if MARKER in n]
if len(marks) < 90:
    sys.exit("only {} marker dispatches; cannot delimit steps".format(len(marks)))

gaps = [(rows[marks[i + 1]][0] - rows[marks[i]][1], i) for i in range(len(marks) - 1)]
cut = sorted(g for g, _ in gaps)[int(len(gaps) * 0.97)]
bounds = [marks[i + 1] for g, i in gaps if g >= cut]
if len(bounds) < 3:
    sys.exit("could not find step boundaries")
k = int(len(bounds) * 0.75)
step = rows[bounds[k]:bounds[k + 1]]

busy = sum(t1 - t0 for t0, t1, _ in step) / 1000.0
span = (step[-1][1] - step[0][0]) / 1000.0
print("one step: {:,} kernels, busy {:,.1f} us, span {:,.1f} us, gaps {:,.1f} us".format(
    len(step), busy, span, span - busy))

agg = collections.defaultdict(lambda: [0, 0.0])
for t0, t1, n in step:
    a = agg[n]
    a[0] += 1
    a[1] += (t1 - t0) / 1000.0

small = {n: v for n, v in agg.items() if v[1] / v[0] < THRESH}
sn = sum(v[0] for v in small.values())
su = sum(v[1] for v in small.values())
print("\n{:,} kernels under {:.0f} us each, {:,.1f} us busy ({:.1f}% of step busy)".format(
    sn, THRESH, su, 100 * su / busy))
print("{:>6} {:>10} {:>8}  kernel".format("calls", "total us", "us each"))
for n, (c, u) in sorted(small.items(), key=lambda kv: -kv[1][1])[:18]:
    print("{:>6} {:>10.1f} {:>8.2f}  {}".format(c, u, u / c, n[:72]))
