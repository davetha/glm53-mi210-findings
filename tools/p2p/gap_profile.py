"""Is the 17.5 ms/step of GPU idle UNIFORM dispatch overhead, or a few big stalls?

These need opposite fixes and the distinction is not visible in a kernel list:

  * uniform (~microseconds between every kernel)  -> the step is DISPATCH-bound. 2730
    kernels/step is the problem; the fix is fusing small kernels to cut launch count.
  * concentrated (a handful of multi-ms stalls)   -> something is SERIALISING, e.g. a host
    round-trip or a dependency wait. The fix is finding those specific points; fusing
    elementwise ops would buy nothing.

Also reports which kernel PRECEDES the biggest gaps, since that names the stall site.
"""
import collections
import csv
import sys

path, lo_s, hi_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
AR = "ncclDevKernel"

rows = []
with open(path, newline="") as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
t0 = rows[0][0]
lo, hi = t0 + int(lo_s * 1e9), t0 + int(hi_s * 1e9)
win = [(s, e, n) for s, e, n in rows if s >= lo and e <= hi]
steps = sum(1 for _, _, n in win if AR in n) / 90.0

# Walk the union frontier; a gap is any time no kernel is resident.
gaps = []
cur_end = win[0][1]
prev_name = win[0][2]
for s, e, n in win[1:]:
    if s > cur_end:
        gaps.append((s - cur_end, prev_name))
    if e > cur_end:
        cur_end, prev_name = e, n
gaps.sort(reverse=True)

tot_gap_ms = sum(g for g, _ in gaps) / 1e6
print(f"{len(gaps)} gaps totalling {tot_gap_ms:.0f} ms over {steps:.1f} steps "
      f"= {tot_gap_ms/steps:.2f} ms/step idle, {len(gaps)/steps:.0f} gaps/step\n")

buckets = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 25), (25, 100), (100, 1000), (1000, 10**9)]
print(f"{'gap size':>14} {'count':>9} {'ms/step':>9} {'% of idle':>10}")
for lo_us, hi_us in buckets:
    sel = [g for g, _ in gaps if lo_us * 1000 <= g < hi_us * 1000]
    if not sel:
        continue
    ms = sum(sel) / 1e6
    lbl = f"{lo_us}-{hi_us} us" if hi_us < 10**9 else f">{lo_us} us"
    print(f"{lbl:>14} {len(sel):>9} {ms/steps:>9.2f} {100*ms/tot_gap_ms:>9.1f}%")

print("\nkernels preceding the most idle time (where the GPU goes quiet):")
by_prev = collections.Counter()
cnt_prev = collections.Counter()
for g, n in gaps:
    by_prev[n[:70]] += g
    cnt_prev[n[:70]] += 1
for n, ns in by_prev.most_common(12):
    print(f"  {ns/1e6/steps:>7.2f} ms/step  {cnt_prev[n]/steps:>7.1f} gaps/step  "
          f"avg {ns/cnt_prev[n]/1e3:>7.1f} us  after: {n[:58]}")
