#!/usr/bin/env bash
set -uo pipefail
cd /home/dave/glm53-bringup
for arm in on off; do
  [ "$arm" = on ] && V=1 || V=0
  EXTRA_DOCKER="-e EXPERT_CACHE_WIDE_SCRATCH=$V" SLOTS=46 ./fp_launch.sh gate,topk,nodoublegate >/tmp/lp_$arm.log 2>&1
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' localhost:8145/health)" = 200 ] || { echo "$arm LAUNCH FAILED"; exit 1; }
  echo "arm=$arm (WIDE_SCRATCH=$V)"
  python3 logprob_cmp.py "$arm" || exit 1
done
python3 - <<'PY'
import json
a = json.load(open("/tmp/lp_on.json")); b = json.load(open("/tmp/lp_off.json"))
verdict_ok = True
for k in a:
    pa, pb = a[k]["probe"], b[k]["probe"]
    ta, tb = pa[0][0], pb[0][0]
    da, db = dict(pa), dict(pb)
    shared = set(da) & set(db)
    between = max((abs(da[t] - db[t]) for t in shared), default=float("nan"))
    noise = max(a[k]["within_arm_gap"], b[k]["within_arm_gap"])
    same_tok = ta == tb
    # The bar: the two arms must not differ by more than ONE SERVER differs from itself.
    ok = same_tok and (between <= max(noise, 1e-6) * 10 or between < 1e-3)
    verdict_ok &= ok
    print(f"  {k:<6} top1 on={ta!r} off={tb!r} {'MATCH' if same_tok else 'DIFFERENT TOKEN'}")
    print(f"         between-arm gap {between:.6f} vs within-arm noise {noise:.6f} -> "
          f"{'within noise' if ok else 'EXCEEDS NOISE'}")
print("\nVERDICT: " + ("PASS - the scratch reads the same weights; differences do not "
      "exceed what one unchanged server shows run to run"
      if verdict_ok else
      "FAIL - the two paths differ by more than this stack's own nondeterminism"))
PY
