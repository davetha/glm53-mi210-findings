#!/usr/bin/env bash
set -uo pipefail
cd /home/dave/glm53-bringup
for arm in on off; do
  [ "$arm" = on ] && V=1 || V=0
  EXTRA_DOCKER="-e EXPERT_CACHE_WIDE_SCRATCH=$V" SLOTS=46 ./fp_launch.sh gate,topk,nodoublegate >/tmp/eq2_$arm.log 2>&1
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8145/health)" = 200 ] || { echo "$arm LAUNCH FAILED"; exit 1; }
  echo "arm=$arm (EXPERT_CACHE_WIDE_SCRATCH=$V)"
  python3 scratch_equiv2.py "$arm" || exit 1
done
python3 - <<'PY'
import json
a = json.load(open("/tmp/equiv_on.json"))
b = json.load(open("/tmp/equiv_off.json"))
bad = 0
for k in a:
    if a[k] == b[k]:
        print(f"  {k:<7} IDENTICAL ({len(a[k])} chars)")
    else:
        bad += 1
        i = next((j for j in range(min(len(a[k]), len(b[k]))) if a[k][j] != b[k][j]), min(len(a[k]), len(b[k])))
        print(f"  {k:<7} DIFFERS at char {i}")
        print(f"      off: ...{b[k][max(0,i-40):i+40]!r}")
        print(f"      on : ...{a[k][max(0,i-40):i+40]!r}")
print("VERDICT:", "PASS - the scratch does not change the output" if not bad else f"FAIL - {bad} prompt(s) differ")
PY
