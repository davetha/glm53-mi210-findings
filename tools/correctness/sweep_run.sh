#!/usr/bin/env bash
set -uo pipefail
cd /home/dave/glm53-bringup
for arm in on off; do
  [ "$arm" = on ] && V=1 || V=0
  EXTRA_DOCKER="-e EXPERT_CACHE_WIDE_SCRATCH=$V" SLOTS=46 ./fp_launch.sh gate,topk,nodoublegate >/tmp/sw_$arm.log 2>&1
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8145/health)" = 200 ] || { echo "$arm LAUNCH FAILED"; exit 1; }
  echo "arm=$arm (WIDE_SCRATCH=$V)"
  python3 lensweep.py "$arm" || exit 1
done
python3 - <<'PY'
import json, statistics
a = json.load(open("/tmp/sweep_on.json")); b = json.load(open("/tmp/sweep_off.json"))
print(f"\n{'length':<8} {'top1 same':<10} {'within on':>10} {'within off':>11} {'mean gap':>10}  verdict")
bad = []
for k in a:
    same = a[k]["tokens"] == b[k]["tokens"]
    noise = max(a[k]["spread"], b[k]["spread"])
    gap = abs(a[k]["mean"] - b[k]["mean"])
    # honest bar: the arms must agree on the token, and the difference of MEANS must not
    # exceed the spread one server already shows on its own.
    ok = same and gap <= max(noise, 1e-9)
    if not ok:
        bad.append(k)
    print(f"{k:<8} {str(same):<10} {a[k]['spread']:>10.6f} {b[k]['spread']:>11.6f} "
          f"{gap:>10.6f}  {'ok' if ok else 'EXCEEDS OWN NOISE'}")
print("\nVERDICT: " + ("PASS - at every length the arms pick the same token and differ by "
      "no more than one server differs from itself"
      if not bad else f"INCONCLUSIVE/FAIL at: {', '.join(bad)}"))
PY
