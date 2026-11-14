// GB300 half, compiled with -arch=sm_103a. The FP64 symbols are the reference.
// inkling_gb300_expert_* is the CUTLASS serving path (NVFP4 ultra, MXFP8 via
// the SM100 MXF8 schedule). Do not load this into a running dsv41 process.
#define INKLING_ARCH 103
#include "inkling_expert.cuh"
#include "inkling_gemm.cuh"

extern "C" int inkling_gb300_nvfp4(
    const double* x, const int* ids, const double* route, const uint8_t* w13, const uint8_t* s13,
    const double* amax13, const uint8_t* w2, const uint8_t* s2, const double* amax2, double denom,
    int tokens, int hidden, int inter, int topk, double* scratch, double* y) {
  return inkling::launch_nvfp4(x, ids, route, w13, s13, amax13, w2, s2, amax2, denom, tokens, hidden,
                               inter, topk, scratch, y);
}

extern "C" int inkling_gb300_mxfp8(
    const double* x, const int* ids, const double* route, const uint8_t* w13, const uint8_t* s13,
    const uint8_t* w2, const uint8_t* s2, int tokens, int hidden, int inter, int topk, double* scratch,
    double* y) {
  return inkling::launch_mxfp8(x, ids, route, w13, s13, w2, s2, tokens, hidden, inter, topk, scratch, y);
}

extern "C" int inkling_gb300_expert_nvfp4(
    int tokens, int topk, int experts, int hidden, int inter, const __nv_bfloat16* x, const int* ids,
    const float* route, const uint8_t* w13, const uint8_t* s13, const float* amax13, const uint8_t* w2,
    const uint8_t* s2, const float* amax2, float denom, __nv_bfloat16* y, cudaStream_t stream) {
  return inkling::expert::forward<inkling::arch::Nvfp4Gemm, inkling::arch::Nvfp4A, inkling::arch::Nvfp4A,
                                  inkling::arch::Nvfp4Scale, 16, true>(
      tokens, topk, experts, hidden, inter, x, ids, route, w13, s13, amax13, w2, s2, amax2, denom, y, stream);
}

extern "C" int inkling_gb300_expert_nvfp4_each(
    int tokens, int topk, int experts, int hidden, int inter, const __nv_bfloat16* x, const int* ids,
    const float* route, const uint8_t* const* w13, const uint8_t* const* s13, const float* amax13,
    const uint8_t* const* w2, const uint8_t* const* s2, const float* amax2, float denom, __nv_bfloat16* y,
    cudaStream_t stream) {
  return inkling::expert::forward<inkling::arch::Nvfp4Gemm, inkling::arch::Nvfp4A, inkling::arch::Nvfp4A,
                                  inkling::arch::Nvfp4Scale, 16, true>(
      tokens, topk, experts, hidden, inter, x, ids, route, nullptr, nullptr, amax13, nullptr, nullptr, amax2,
      denom, y, stream, w13, s13, w2, s2);
}

extern "C" int inkling_gb300_expert_mxfp8_each(
    int tokens, int topk, int experts, int hidden, int inter, const __nv_bfloat16* x, const int* ids,
    const float* route, const uint8_t* const* w13, const uint8_t* const* s13, const uint8_t* const* w2,
    const uint8_t* const* s2, __nv_bfloat16* y, cudaStream_t stream) {
  return inkling::expert::forward<inkling::arch::Mxfp8Gemm, inkling::arch::Mxfp8A, inkling::arch::Mxfp8A,
                                  inkling::arch::Mxfp8Scale, 32, false>(
      tokens, topk, experts, hidden, inter, x, ids, route, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      1.f, y, stream, w13, s13, w2, s2);
}

extern "C" int inkling_gb300_expert_mxfp8(
    int tokens, int topk, int experts, int hidden, int inter, const __nv_bfloat16* x, const int* ids,
    const float* route, const uint8_t* w13, const uint8_t* s13, const uint8_t* w2, const uint8_t* s2,
    __nv_bfloat16* y, cudaStream_t stream) {
  return inkling::expert::forward<inkling::arch::Mxfp8Gemm, inkling::arch::Mxfp8A, inkling::arch::Mxfp8A,
                                  inkling::arch::Mxfp8Scale, 32, false>(
      tokens, topk, experts, hidden, inter, x, ids, route, w13, s13, nullptr, w2, s2, nullptr, 1.f, y, stream);
}
