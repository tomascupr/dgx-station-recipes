#!/usr/bin/env bash
# Compile-only build of the unvalidated Inkling peer objects.
#   peer_inkling_gb300_sm103.so   nvcc -arch=sm_103
#   peer_inkling_rtx_sm120.so     nvcc -arch=sm_120
# Does not run the result, does not call nvidia-smi, and does not start a
# container. If nvcc is missing this exits 2 unless --dry-run, which prints
# the commands and exits 0. A cloud agent is typically x86 and cannot produce
# the aarch64 objects the DGX loads; compile those on the Station with
# CUDA_VISIBLE_DEVICES unset so the compiler does not need a device.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${INKLING2_KERNEL_OUT:-$HERE/../peer/prebuilt}
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; fi
unset CUDA_VISIBLE_DEVICES
mkdir -p "$OUT"
compile() {
  local src=$1 arch=$2 so=$3
  echo "nvcc -shared -O2 -std=c++17 -arch=${arch} -I${HERE}/csrc ${src} -o ${so}"
  if [ "$DRY" = 1 ]; then return 0; fi
  nvcc -shared -O2 -std=c++17 -arch="${arch}" -I"${HERE}/csrc" "${src}" -o "${so}"
}
if [ "$DRY" = 0 ] && ! command -v nvcc >/dev/null 2>&1; then
  echo "UNVALIDATED: nvcc is not on PATH. Nothing was compiled. Re-run on the DGX, or pass --dry-run." >&2
  exit 2
fi
compile "$HERE/csrc/peer_inkling_gb300.cu" sm_103 "$OUT/peer_inkling_gb300_sm103.so"
compile "$HERE/csrc/peer_inkling_rtx.cu" sm_120 "$OUT/peer_inkling_rtx_sm120.so"
if [ "$DRY" = 1 ]; then
  echo "dry-run only; no .so written"
  exit 0
fi
echo "UNVALIDATED: objects written to $OUT. They have not been executed."
