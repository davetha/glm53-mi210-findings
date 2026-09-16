#!/usr/bin/env bash
# Interleaved A/B of the kernel-fusion set, repeated across container restarts.
#
# One container per arm is not enough to resolve a few ms: the 27 GB host offload buffer
# lands in different physical memory on every start, and the gather reads it over PCIe,
# so between-container variance is confounded with the change under test. Arms are
# therefore alternated ON/OFF/ON/OFF so any drift over the run hits both equally, and
# each arm is measured three times.
set -uo pipefail
cd /home/dave/glm53-bringup
REPS=${REPS:-3}
for rep in $(seq 1 "$REPS"); do
  for arm in on off; do
    if [ "$arm" = on ]; then
      D="-e GLM53_TOPK_REGSTORE=1 -e EXPERT_CACHE_FUSED_REMAP=1"; P=gate,topk,nodoublegate
    else
      D="-e GLM53_TOPK_REGSTORE=0 -e EXPERT_CACHE_FUSED_REMAP=0"; P=gate,topk
    fi
    EXTRA_DOCKER="$D" ./fp_launch.sh "$P" >/tmp/fab_${arm}_${rep}.log 2>&1
    ok=$(grep -c "'4'" /tmp/fab_${arm}_${rep}.log)
    ms=$(./ab_tuned_moe.sh 8145 2>&1 | grep -oE "run[3-6]: +96 tok +[0-9.]+" | grep -oE "[0-9.]+$" | tr '\n' ' ')
    echo "rep$rep arm=$arm correctness_ok=$ok ms/step: $ms"
  done
done
