// UNVALIDATED sm_120 half. Same math as the GB300 object; a separate
// translation unit so the two .so files are built with different -arch flags.
#include "inkling_gemm.cuh"

extern "C" int inkling_rtx_nvfp4(
    const double* x, const int* ids, const double* route, const uint8_t* w13, const uint8_t* s13,
    const double* amax13, const uint8_t* w2, const uint8_t* s2, const double* amax2, double denom,
    int tokens, int hidden, int inter, int topk, double* scratch, double* y) {
  return inkling::launch_nvfp4(x, ids, route, w13, s13, amax13, w2, s2, amax2, denom, tokens, hidden,
                               inter, topk, scratch, y);
}

extern "C" int inkling_rtx_mxfp8(
    const double* x, const int* ids, const double* route, const uint8_t* w13, const uint8_t* s13,
    const uint8_t* w2, const uint8_t* s2, int tokens, int hidden, int inter, int topk, double* scratch,
    double* y) {
  return inkling::launch_mxfp8(x, ids, route, w13, s13, w2, s2, tokens, hidden, inter, topk, scratch, y);
}
