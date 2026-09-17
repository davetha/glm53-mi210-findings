#!/usr/bin/env bash
# Build libhostar.so. Mirrors vllm-expert-cache/kernels/build.sh.
#   ./build.sh            # gfx90a (MI210)
#   ./build.sh gfx942     # MI300
set -euo pipefail
cd "$(dirname "$0")"
ARCHS=("$@"); [ ${#ARCHS[@]} -eq 0 ] && ARCHS=(gfx90a)
OUT=${OUT:-libhostar.so}
HIPCC=${HIPCC:-}
if [ -z "$HIPCC" ]; then
  for c in /opt/rocm/bin/hipcc "$(command -v hipcc || true)" \
           /opt/python/lib/python*/site-packages/_rocm_sdk_devel/bin/hipcc; do
    [ -n "$c" ] && [ -x "$c" ] && { HIPCC=$c; break; }
  done
fi
[ -n "$HIPCC" ] || { echo "hipcc not found; set HIPCC=" >&2; exit 1; }
FLAGS=(-O3 -std=c++17 -fPIC -shared)
for a in "${ARCHS[@]}"; do FLAGS+=(--offload-arch="$a"); done
DEVLIB=$(ls -d /opt/python/lib/python*/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode 2>/dev/null | head -1 || true)
[ -n "$DEVLIB" ] && FLAGS+=(--rocm-device-lib-path="$DEVLIB")
"$HIPCC" "${FLAGS[@]}" hostar.hip -o "$OUT"
for sym in hostar_allreduce hostar_reset; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym" >&2; exit 1; }
done
echo "built $OUT"; ls -l "$OUT"
