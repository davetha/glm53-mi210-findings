#!/usr/bin/env bash
set -uo pipefail
cd /home/dave/glm53-bringup
for arm in on off; do
  [ "$arm" = on ] && V=1 || V=0
  EXTRA_DOCKER="-e EXPERT_CACHE_WIDE_SCRATCH=$V" SLOTS=46 ./fp_launch.sh gate,topk,nodoublegate >/tmp/ft_$arm.log 2>&1
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8145/health)" = 200 ] || { echo "$arm LAUNCH FAILED"; exit 1; }
  echo "arm=$arm (WIDE_SCRATCH=$V)"
  python3 fixedtok.py "$arm" || exit 1
done
python3 - <<'PY'
import json
a = json.load(open("/tmp/ft_on.json")); b = json.load(open("/tmp/ft_off.json"))
print(f"\n{'length':<8} {'spread on':>11} {'spread off':>12} {'mean gap':>11}  verdict")
bad = []
for k in sorted(set(a) & set(b)):
    noise = max(a[k]["spread"], b[k]["spread"])
    gap = abs(a[k]["mean"] - b[k]["mean"])
    ok = gap <= max(noise, 1e-9)
    if not ok:
        bad.append(k)
    print(f"{k:<8} {a[k]['spread']:>11.6f} {b[k]['spread']:>12.6f} {gap:>11.6f}  "
          f"{'ok' if ok else 'EXCEEDS OWN NOISE'}")
print("\nVERDICT: " + ("PASS - on a fixed token the arms differ by no more than one "
      "server differs from itself, at every length"
      if not bad else f"FAIL at: {', '.join(bad)}"))
PY
