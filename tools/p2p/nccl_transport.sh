#!/usr/bin/env bash
# Which NCCL transport is in use, and is it the slow one?
#
# NCCL/RCCL choose between:
#   P2P transport  -- head/tail flags in PEER DEVICE MEMORY, direct GPU-GPU visibility
#   SHM transport  -- head/tail flags in HOST SHARED MEMORY
#
# Measured on this box: a peer-VRAM flag handshake between these two MI210s STALLS
# intermittently (six memory-ordering combinations, all fail), while the same handshake
# through pinned host memory runs 2000 round trips clean at 4.49 us. NCCL issue #2079
# reports the identical split on NVIDIA hardware over a PCIe host bridge: P2P transport
# deadlocks, peer data copies are fine.
#
# NCCL currently costs 116 us per 8 KB exchange here (14.45 ms of a 70 ms decode step).
# If it selected the P2P transport on this topology, forcing it to SHM may be much faster
# -- and that is one environment variable, no code.
#
# Paired, because the e2e noise floor on this box is 8.91 ms: both arms launched back to
# back, 8 warm-up generations discarded, 12 kept, drift checked.
set -uo pipefail
cd /home/dave/glm53-bringup
PORT=8145
WARM=8; KEEP=12

ask() { curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm53\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0}"; }

arm() {  # arm <label> <extra docker env>
  local label=$1 extra=$2
  echo "############ $label ############"
  EXTRA_DOCKER="$extra -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,GRAPH" \
    ./fp_launch.sh "gate,topk,nodoublegate" > /tmp/ncclt_${label}.log 2>&1
  curl -sf -m 5 localhost:$PORT/health >/dev/null || { echo "$label LAUNCH FAILED"; tail -15 /tmp/ncclt_${label}.log; return 1; }

  # Which transport did it actually pick for the TP pair?
  echo "--- transport selection ---"
  docker logs glm53 2>&1 | grep -iE "via (P2P|SHM|direct|NET)|P2P is (enabled|disabled)|Channel .* via" \
    | sed 's/.*NCCL INFO //' | sort -u | head -6
  docker logs glm53 2>&1 | grep -oiE "P2P/(IPC|direct|CUMEM)|SHM/direct|via SHM|via P2P" | sort | uniq -c | head -5

  # correctness, with the non-empty assertion this model requires
  local ok=0
  for qa in "What is 2+2? Answer with just the number.|4" \
            "What is 17 times 23? Answer with just the number.|391"; do
    local got
    got=$(ask "${qa%|*}" 3000 | python3 -c "import json,sys; d=json.load(sys.stdin)['choices'][0]['message']; print((d.get('content') or '').strip()[:50])")
    [ -n "$got" ] && grep -qiF "${qa#*|}" <<<"$got" && ok=$((ok+1))
    echo "  '$got'"
  done
  echo "  correctness $ok/2"

  for i in $(seq 1 $WARM); do ask "Write a detailed paragraph about ocean currents." 96 >/dev/null; done
  rm -f /tmp/ncclt_${label}.times
  for i in $(seq 1 $KEEP); do
    local s e n
    s=$(date +%s.%N)
    n=$(ask "Write a detailed paragraph about ocean currents." 96 | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])")
    e=$(date +%s.%N)
    python3 -c "print(f'{($e-$s)*1000/$n:.2f}')" >> /tmp/ncclt_${label}.times
  done
  echo -n "  kept: "; tr '\n' ' ' < /tmp/ncclt_${label}.times; echo
}

arm baseline ""                                   || exit 1
arm p2poff   "-e NCCL_P2P_DISABLE=1"              || exit 1
arm finegrain "-e HSA_FORCE_FINE_GRAIN_PCIE=1"    || exit 1

python3 - <<'PY'
import statistics as st
r={}
for lab in ("baseline","p2poff","finegrain"):
    v=[float(x) for x in open(f"/tmp/ncclt_{lab}.times") if x.strip()]
    out=[x for x in v if x > 2*st.median(v)]
    if out: print(f"  ({lab}: discarding {len(out)} outlier(s) {[f'{x:.0f}' for x in out]})")
    v=[x for x in v if x <= 2*st.median(v)]
    h=len(v)//2
    r[lab]=(st.median(v),min(v),max(v),st.median(v[h:])-st.median(v[:h]))
    print(f"{lab:9s} median={r[lab][0]:7.2f}  min={r[lab][1]:7.2f}  max={r[lab][2]:7.2f}  drift={r[lab][3]:+.1f}")
b=r["baseline"][0]
print()
for lab in ("p2poff","finegrain"):
    d=r[lab][0]-b
    spread=max(r["baseline"][2]-r["baseline"][1], r[lab][2]-r[lab][1])
    verdict=("inside noise -- no effect" if abs(d)<=spread else ("FASTER" if d<0 else "SLOWER"))
    print(f"{lab:10s} - baseline = {d:+7.2f} ms   (widest within-arm spread {spread:5.2f} ms)  {verdict}")
print("\nNote: the box cannot resolve effects under ~9 ms end-to-end.")
PY
