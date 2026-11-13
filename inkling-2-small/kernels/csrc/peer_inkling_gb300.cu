// UNVALIDATED sm_103 half. Build with kernels/build.sh. Do not load this into
// a running dsv41 process. The Python sidecar does not call these symbols.
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
