#!/usr/bin/env bash
# Launch Gemma-4-31B W8A8-INT8 and benchmark it. TP is the argument.
#
# WHY THIS IS A DIFFERENT PROBLEM FROM GLM-5.3
#   dense (enable_moe_block: false), 33 GB at INT8, fits on ONE 63 GiB MI210.
#   So: no expert offload, no PCIe gather (that was 20% of GLM's step), no expert cache,
#   no slot tuning, no wide scratch. All of that apparatus is dead weight here.
#
# THE CENTRAL QUESTION: TP=1 or TP=2?
#   All-reduce on this box costs 116-185 us per exchange -- two MI210s with no XGMI, and
#   peer-VRAM handshakes do not work at all (see P2P-PEER-HANDSHAKE.md). At 60 layers TP=2
#   means ~120 all-reduces/step, roughly 19 ms of pure communication tax.
#   TP=1 pays none of it but halves memory bandwidth. Arithmetic says TP=1, so measure it.
#
# HARDWARE LESSONS THAT DO CARRY OVER
#   * Expose ONLY the MI210 render nodes. Including an R9700 makes vLLM's amdsmi arch probe
#     read RDNA4 and silently disable every gfx9 path: 19.9 -> 48.0 tok/s, no error.
#   * FULL_DECODE_ONLY cudagraphs. PIECEWISE costs ~60% of decode, silently.
#   * Correctness: assert non-empty output before believing a pass.
set -uo pipefail
TP=${1:-1}
PORT=${PORT:-8146}
NAME=${NAME:-gemma4}
MODEL=/models/gemma4-31b-w8a8
MAXLEN=${MAXLEN:-8192}
UTIL=${UTIL:-0.90}

# renderD128 + renderD131 are the MI210s. renderD129/130 are the R9700s -- never expose.
DEV=(--device /dev/kfd --device /dev/dri/renderD128)
[ "$TP" -ge 2 ] && DEV+=(--device /dev/dri/renderD131)

docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" "${DEV[@]}" --group-add video --ipc host -p ${PORT}:8000 \
  -v /mnt/llm-storage:/models:ro -v /mnt/llm-storage/cache:/cache \
  -e VLLM_USE_V1=1 -e HSA_NO_SCRATCH_RECLAIM=1 -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ROCM_USE_AITER=1 -e HSA_ENABLE_COREDUMP=0 --ulimit core=0 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton \
  --entrypoint /usr/local/bin/mi210-entrypoint local/vllm-mi210:rocm10-mi210.7-aiter-jitwarm \
  serve "$MODEL" --served-model-name gemma4 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 1 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  ${EXTRA_VLLM:-} >/dev/null 2>&1

echo "launched TP=$TP on :$PORT, waiting..."
for i in $(seq 1 180); do
  curl -sf -m 3 localhost:$PORT/health >/dev/null 2>&1 && { echo "ready after ~$((i*10))s"; break; }
  docker ps -q --filter name=^${NAME}$ | grep -q . || { echo "CONTAINER DIED"; docker logs "$NAME" 2>&1 | tail -25; exit 1; }
  sleep 10
done
curl -sf -m 3 localhost:$PORT/health >/dev/null 2>&1 || { echo "TIMED OUT"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }

echo "=== engagement checks (a silent fallback here makes every number meaningless) ==="
docker logs "$NAME" 2>&1 | grep -c "Capturing CUDA graphs (FULL)" | sed 's/^/  FULL cudagraph captures: /'
docker logs "$NAME" 2>&1 | grep -oiE "quark|int8|w8a8|compressed-tensors" | sort | uniq -c | head -5
docker logs "$NAME" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens|Model loading took [0-9.]+ GiB|Using \[[^]]*\] all-reduce backends" | head -3
echo "  gfx arch seen by vLLM:"; docker logs "$NAME" 2>&1 | grep -oiE "gfx[0-9a-z]+" | sort -u | head -3 | sed 's/^/    /'

ask() { curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"gemma4\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0}"; }

echo "=== correctness ==="
ok=0
for qa in "What is 2+2? Answer with just the number.|4" \
          "What is 17 times 23? Answer with just the number.|391" \
          "What is the capital of France? Answer with one word.|Paris"; do
  g=$(ask "${qa%|*}" 200 | python3 -c "import json,sys
try:
    d=json.load(sys.stdin)['choices'][0]['message']; print((d.get('content') or '').strip()[:60])
except Exception as e: print('PARSE_FAIL')")
  if [ -n "$g" ] && [ "$g" != "PARSE_FAIL" ] && grep -qiF "${qa#*|}" <<<"$g"; then ok=$((ok+1)); echo "  ok: $g"; else echo "  BAD: '$g'"; fi
done
echo "  correctness $ok/3"

for i in 1 2 3 4 5 6; do ask "Write a detailed paragraph about ocean currents." 128 >/dev/null; done
rm -f /tmp/g4_tp${TP}.times
for i in $(seq 1 12); do
  s=$(date +%s.%N)
  n=$(ask "Write a detailed paragraph about ocean currents." 128 | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['completion_tokens'])")
  e=$(date +%s.%N)
  python3 -c "print(f'{($e-$s)*1000/$n:.2f}')" >> /tmp/g4_tp${TP}.times
done
python3 -c "
import statistics as st
v=[float(x) for x in open('/tmp/g4_tp${TP}.times')]; q=sorted(v)
print(f'  TP=${TP} decode: median={st.median(v):.2f} ms/step  ({1000/st.median(v):.2f} tok/s)  min={min(v):.2f} max={max(v):.2f} IQR={q[int(len(q)*.75)]-q[int(len(q)*.25)]:.2f}')
"
