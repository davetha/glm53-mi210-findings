#!/usr/bin/env bash
# Does MTP speculative decoding pay, given expert offload?
#
# THE WEIGHTS ARE THERE: layer 45 is a full MTP draft layer (eh_proj, enorm/hnorm, 288-expert
# MoE). But it shipped in BF16 while the rest of the model is W4A16 int4:
#   layer 45 = 14.87 GB  ->  7.43 GB/rank at TP=2, against ~3.3 GB free VRAM
# So it cannot be resident. It will be offloaded, and each draft step routes 8 of its
# experts -- at BF16 those are ~51.6 MB each vs 12.375 MiB for the int4 layers, so ~400 MB
# per draft step, ~14 ms at the measured 28 GB/s.
#
# THE QUESTION: does the acceptance rate buy more than that ~14 ms costs? ~33 ms of the
# 76 ms step is fixed per-step cost (all-reduce, dense GEMM, elementwise, mHC) that a
# verified draft token gets for free, so the bar is reachable -- but not obviously.
#
# If this loses, the follow-up is to quantise layer 45 to int4 (14.87 -> ~3.9 GB, 1.95
# GB/rank, fits resident, gathers nothing). This run decides whether that work is worth it.
set -uo pipefail
cd /home/dave/glm53-bringup
PORT=8145
OFF=${OFF:-45}
SL=${SL:-52}

docker stop -t 30 glm53 >/dev/null 2>&1; docker rm -f glm53 >/dev/null 2>&1

NAME=glm53 PORT=$PORT TP=2 UTIL=0.97 MAXLEN=8192 MAX_NUM_SEQS=1 \
  OFFLOAD_GB=$OFF SLOTS=$SL POLICY=lfu SPEC=on SPEC_N=1 \
  EXTRA_VLLM_ARGS="--enable-expert-parallel --max-num-batched-tokens 4096 --kv-cache-memory=644245094 ${EAGER:+--enforce-eager}" \
  EXTRA_DOCKER_ARGS="-e GLM53_FASTPATH_PARTS=gate,topk,nodoublegate" \
  ./launch_glm53.sh > /tmp/mtp_launch.log 2>&1
echo "launcher rc=$? (rc=1 can be the DECLINED-layers gate, not fatal)"

if ! curl -sf -m 5 localhost:$PORT/health >/dev/null; then
  echo "=== DID NOT COME UP ==="
  docker logs glm53 2>&1 | grep -iE "out of memory|OutOfMemory|No available memory|Free memory on device|illegal memory|RuntimeError|ValueError|not supported|speculative" | tail -12
  exit 1
fi
echo "SERVING with MTP"
docker logs glm53 2>&1 | grep -oE "expert-cache [0-9.]+ active on [0-9]+ MoE layer\(s\) \([0-9]+ declined\)|Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens" | head -3
docker logs glm53 2>&1 | grep -iE "speculative|draft|num_speculative" | grep -viE "error|warn" | head -4

ask() { curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm53\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0}"; }

echo "=== correctness ==="
ok=0
for qa in "What is 2+2? Answer with just the number.|4" \
          "What is 17 times 23? Answer with just the number.|391" \
          "What is the capital of France? Answer with one word.|Paris"; do
  g=$(ask "${qa%|*}" 3000 | python3 -c "import json,sys; d=json.load(sys.stdin)['choices'][0]['message']; print((d.get('content') or '').strip()[:50])")
  if [ -n "$g" ] && grep -qiF "${qa#*|}" <<<"$g"; then ok=$((ok+1)); echo "  ok: $g"; else echo "  BAD: '$g'"; fi
done
echo "  correctness $ok/3"

for i in 1 2 3 4 5 6; do ask "Write a detailed paragraph about ocean currents." 96 >/dev/null; done
rm -f /tmp/mtp.times
for i in $(seq 1 12); do
  s=$(date +%s.%N)
  n=$(ask "Write a detailed paragraph about ocean currents." 96 | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])")
  e=$(date +%s.%N)
  python3 -c "print(f'{($e-$s)*1000/$n:.2f}')" >> /tmp/mtp.times
done
echo -n "  kept: "; tr '\n' ' ' < /tmp/mtp.times; echo

echo "=== acceptance rate (the number that decides it) ==="
docker logs glm53 2>&1 | grep -iE "acceptance|accepted|num_accepted|drafted|spec_decode|Speculative" | tail -10

python3 - <<'PY'
import statistics as st, os
base=[float(x) for x in open("/tmp/slots_52kv.times")] if os.path.exists("/tmp/slots_52kv.times") else None
v=[float(x) for x in open("/tmp/mtp.times")]
q=sorted(v)
print(f"\nMTP       median={st.median(v):7.2f} min={min(v):7.2f} max={max(v):7.2f} IQR={q[int(len(q)*.75)]-q[int(len(q)*.25)]:5.2f}")
if base:
    qb=sorted(base)
    print(f"no-spec   median={st.median(base):7.2f} min={min(base):7.2f} max={max(base):7.2f} IQR={qb[int(len(qb)*.75)]-qb[int(len(qb)*.25)]:5.2f}")
    d=st.median(v)-st.median(base)
    print(f"\ndelta = {d:+.2f} ms/token   tok/s {1000/st.median(base):.2f} -> {1000/st.median(v):.2f}")
    print("VERDICT:", "MTP WINS" if d < -2 else ("MTP LOSES" if d > 2 else "inside noise"))
PY
