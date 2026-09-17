#!/usr/bin/env bash
# Task 1.2 validation: does the host-staged all-reduce work inside vLLM, and is it faster?
#
# Correctness first and hard. This replaces the reduction that every layer depends on, so a
# subtle error would not crash -- it would produce fluent, plausible, wrong text. The gate
# below demands exact answers on arithmetic prompts AND asserts non-empty content (this
# model streams into `reasoning` until that block closes, so a small max_tokens yields an
# empty string that compares equal to anything).
set -uo pipefail
cd /home/dave/glm53-bringup
PORT=8145
SP=/opt/python/lib/python3.14/site-packages
B=/home/dave/glm53-bringup
WARM=8; KEEP=12

MOUNTS="\
-v $B/hostar.py:$SP/hostar.py:ro \
-v $B/hostar_dist:$SP/hostar-0.1.0.dist-info:ro \
-v $B/libhostar.so:$SP/libhostar.so:ro \
-e HOSTAR_LIB=$SP/libhostar.so"

ask() { curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm53\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0}"; }
content() { python3 -c "import json,sys; d=json.load(sys.stdin)['choices'][0]['message']; print((d.get('content') or '').strip()[:70])"; }

arm() {  # arm <label> <extra env>
  local label=$1 extra=$2
  echo "############ $label ############"
  EXTRA_DOCKER="$MOUNTS $extra" ./fp_launch.sh "gate,topk,nodoublegate" > /tmp/hostar_${label}.log 2>&1
  curl -sf -m 5 localhost:$PORT/health >/dev/null || { echo "$label LAUNCH FAILED"; tail -25 /tmp/hostar_${label}.log; return 1; }

  echo "--- hostar status ---"
  docker logs glm53 2>&1 | grep -iE "hostar" | sed 's/.*\] //' | sort -u | head -6
  # Hard gate: the previous run silently compared stock-vs-stock for 25 minutes because the
  # patch failed to install and nothing checked. An arm that should be patched and is not is
  # a failed run, not a data point.
  if [ "$label" = "hostar" ]; then
    local lg n_inst n_arena
    lg=$(docker logs glm53 2>&1)
    n_inst=$(printf '%s' "$lg" | grep -ci "hostar: installed" || true)
    n_arena=$(printf '%s' "$lg" | grep -ci "hostar: arena" || true)
    if [ "${n_inst:-0}" -eq 0 ]; then
      echo "  *** hostar NOT installed -- aborting, this would compare stock vs stock ***"
      printf '%s' "$lg" | grep -iE "hostar" | head -10
      return 1
    fi
    if [ "${n_arena:-0}" -lt 2 ]; then
      echo "  *** only $n_arena of 2 ranks built the arena -- aborting ***"
      printf '%s' "$lg" | grep -iE "hostar" | head -10
      return 1
    fi
    echo "  hostar ACTIVE ($n_inst install lines, $n_arena/2 ranks attached to the arena)"
  fi
  docker logs glm53 2>&1 | grep -oE "Using \[[^]]*\] all-reduce backends" | head -1

  echo "--- correctness (exact answers required) ---"
  local ok=0 tot=0
  for qa in "What is 2+2? Answer with just the number.|4" \
            "What is 17 times 23? Answer with just the number.|391" \
            "What is the capital of France? Answer with one word.|Paris" \
            "List the first five prime numbers.|11"; do
    local q=${qa%|*} want=${qa#*|} got
    got=$(ask "$q" 3000 | content); tot=$((tot+1))
    if [ -z "$got" ]; then echo "  EMPTY -- reasoning never closed, correctness NOT established"
    elif grep -qiF "$want" <<<"$got"; then ok=$((ok+1)); echo "  ok: '$got'"
    else echo "  WRONG: got '$got' want '$want'"; fi
  done
  echo "  correctness $ok/$tot"
  if [ "$ok" -lt "$tot" ]; then
    echo "  *** $label CORRECTNESS FAILED -- timings below are meaningless ***"
  fi

  for i in $(seq 1 $WARM); do ask "Write a detailed paragraph about ocean currents." 96 >/dev/null; done
  rm -f /tmp/hostar_${label}.times
  for i in $(seq 1 $KEEP); do
    local s e n
    s=$(date +%s.%N)
    n=$(ask "Write a detailed paragraph about ocean currents." 96 | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])")
    e=$(date +%s.%N)
    python3 -c "print(f'{($e-$s)*1000/$n:.2f}')" >> /tmp/hostar_${label}.times
  done
  echo -n "  kept: "; tr '\n' ' ' < /tmp/hostar_${label}.times; echo
}

arm nccl   "-e VLLM_HOSTAR=0" || exit 1
arm hostar "-e VLLM_HOSTAR=1" || exit 1

python3 - <<'PY'
import statistics as st
r={}
for lab in ("nccl","hostar"):
    v=[float(x) for x in open(f"/tmp/hostar_{lab}.times") if x.strip()]
    med=st.median(v)
    out=[x for x in v if x > 2*med]
    if out: print(f"  ({lab}: discarding outlier(s) {[f'{x:.0f}' for x in out]})")
    v=[x for x in v if x <= 2*med]
    h=len(v)//2
    r[lab]=(st.median(v),min(v),max(v),st.median(v[h:])-st.median(v[:h]))
    print(f"{lab:7s} median={r[lab][0]:7.2f}  min={r[lab][1]:7.2f}  max={r[lab][2]:7.2f}  drift={r[lab][3]:+.1f}")
n,hh=r["nccl"][0],r["hostar"][0]
spread=max(r["nccl"][2]-r["nccl"][1], r["hostar"][2]-r["hostar"][1])
print(f"\nhostar - nccl = {hh-n:+.2f} ms   (widest within-arm spread {spread:.2f} ms)")
print(f"tok/s: nccl {1000/n:.2f} -> hostar {1000/hh:.2f}   ({100*(n/hh-1):+.1f}%)")
print("VERDICT:", "inside noise -- not demonstrated" if abs(hh-n)<=spread
      else ("HOST-STAGED FASTER" if hh<n else "HOST-STAGED SLOWER"))
print("\nPredicted from microbenchmark: 90 x (116 - 6.73) us = -9.8 ms/step")
PY
