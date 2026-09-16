#!/usr/bin/env bash
# Interleaved A/B of the expert-cache slot count, repeated across container restarts.
# Both arms carry the resident-layer decline, so the only variable is slots per layer.
# Alternating the arms cancels the ~6 ms of container-start drift measured earlier.
set -uo pipefail
cd /home/dave/glm53-bringup
for rep in 1 2 3; do
  for s in 32 8; do
    SLOTS=$s ./fp_launch.sh gate,topk,nodoublegate >/tmp/sab_${s}_${rep}.log 2>&1
    ok=$(grep -c "'4'" /tmp/sab_${s}_${rep}.log)
    armed=$(docker logs glm53 2>&1 | grep -oE "active on [0-9]+ MoE layer" | tail -1 | grep -oE "[0-9]+")
    res=$(docker logs glm53 2>&1 | grep -oE "[0-9]+/144 experts resident" | tail -1)
    ms=$(./ab_tuned_moe.sh 8145 2>&1 | grep -oE "run[3-6]: +96 tok +[0-9.]+" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "rep$rep slots=$s armed=$armed $res ok=$ok ms/step: $ms"
  done
done
