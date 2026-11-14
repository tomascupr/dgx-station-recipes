#!/usr/bin/env bash
# Compile-only build of the Inkling peer objects.
#   peer_inkling_gb300_sm103.so   nvcc -arch=sm_103a
#   peer_inkling_rtx_sm120.so     nvcc -arch=sm_120a
# Block-scaled FP4/FP8 need the 'a' suffix. Does not run the result, does not
# call nvidia-smi, and does not start a container. CUDA_VISIBLE_DEVICES is
# left alone. CUDA_HOME, if set, supplies nvcc (the Station keeps it at
# /usr/local/cuda-13.2/bin, which is not on PATH).
#
# CUTLASS b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd is required. Set CUTLASS_ROOT, or
# the script clones that commit into kernels/third_party/cutlass. --dry-run
# only prints the nvcc lines and does not download.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${INKLING2_KERNEL_OUT:-$HERE/../peer/prebuilt}
PIN=b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
DRY=0
if [ "${1:-}" = --dry-run ]; then DRY=1; fi

if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  NVCC=${NVCC:-${CUDA_HOME}/bin/nvcc}
else
  NVCC=${NVCC:-nvcc}
fi
CUTLASS_ROOT=${CUTLASS_ROOT:-${INKLING2_CUTLASS_ROOT:-$HERE/third_party/cutlass}}

have_cutlass() {
  [ -f "$CUTLASS_ROOT/include/cutlass/gemm/device/gemm_universal_adapter.h" ]
}

if ! have_cutlass; then
  if [ "$DRY" = 1 ]; then
    echo "CUTLASS $PIN is not at $CUTLASS_ROOT; dry-run does not download it"
  else
    echo "fetching CUTLASS $PIN into $CUTLASS_ROOT"
    rm -rf "$CUTLASS_ROOT"
    git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
    git -C "$CUTLASS_ROOT" fetch --depth 1 origin "$PIN"
    git -C "$CUTLASS_ROOT" checkout --detach FETCH_HEAD
    have_cutlass || { echo "CUTLASS checkout has no headers" >&2; exit 2; }
  fi
fi

mkdir -p "$OUT"
compile() {
  local src=$1 arch=$2 so=$3
  # -arch=sm_103a also emits forward-compat PTX for compute_103, and ptxas
  # rejects tcgen05 block-scale there. Bind the virtual arch to compute_${arch}.
  local virt=compute_${arch#sm_}
  echo "$NVCC" -shared -O3 -std=c++17 -Xcompiler -fPIC --expt-relaxed-constexpr \
    -diag-suppress 20012 \
    -gencode "arch=${virt},code=${arch}" \
    -I"$HERE/csrc" -I"$CUTLASS_ROOT/include" -I"$CUTLASS_ROOT/tools/util/include" \
    "$src" -o "$so"
  if [ "$DRY" = 1 ]; then return 0; fi
  "$NVCC" -shared -O3 -std=c++17 -Xcompiler -fPIC --expt-relaxed-constexpr \
    -diag-suppress 20012 \
    -gencode "arch=${virt},code=${arch}" \
    -I"$HERE/csrc" -I"$CUTLASS_ROOT/include" -I"$CUTLASS_ROOT/tools/util/include" \
    "$src" -o "$so"
}
if [ "$DRY" = 0 ] && ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "nvcc is not available ($NVCC). Set CUDA_HOME or pass --dry-run." >&2
  exit 2
fi
compile "$HERE/csrc/peer_inkling_gb300.cu" sm_103a "$OUT/peer_inkling_gb300_sm103.so"
compile "$HERE/csrc/peer_inkling_rtx.cu" sm_120a "$OUT/peer_inkling_rtx_sm120.so"
if [ "$DRY" = 1 ]; then
  echo "dry-run only; no .so written"
  exit 0
fi
echo "objects written to $OUT. They have not been executed."
