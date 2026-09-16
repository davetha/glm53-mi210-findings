#!/usr/bin/env bash
# Interleaved A/B of the expert-cache operating point, across container restarts.
#   big  = offload 45 GiB (26 layers on host), 56 of 144 experts resident per layer
#   base = offload 28 GiB (17 layers on host),  8 of 144 experts resident per layer
# Both carry the resident-layer decline, so declined layers never allocate slots.
# Arms alternate so the ~6 ms of container-start drift hits both equally.
set -uo pipefail
cd /home/dave/glm53-bringup
for rep in 1 2 3; do
  for arm in big base; do
    if [ "$arm" = big ]; then O=45; S=56; else O=28; S=8; fi
    OFFLOAD_GB=$O SLOTS=$S ./fp_launch.sh gate,topk,nodoublegate >/tmp/cab_${arm}_${rep}.log 2>&1
    h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:8145/health)
    ok=$(grep -c "content: .4." /tmp/cab_${arm}_${rep}.log)
    res=$(docker logs glm53 2>&1 | grep -oE "active on [0-9]+ MoE layer\(s\) \([0-9]+ declined\): [0-9]+/144" | tail -1)
    if [ "$h" = 200 ]; then
      ms=$(./ab_tuned_moe.sh 8145 2>&1 | grep -oE "run[3-6]: +96 tok +[0-9.]+" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    else
      ms="LAUNCH_FAILED"
    fi
    echo "rep$rep arm=$arm off=$O slots=$S health=$h ok=$ok [$res] ms/step: $ms"
  done
done
