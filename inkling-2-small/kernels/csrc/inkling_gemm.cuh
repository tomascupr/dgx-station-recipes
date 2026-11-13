// UNVALIDATED. Not compiled and not run on sm_103 or sm_120.
// Correctness kernel only: one thread per token, fp64 accumulation, the same
// formulas as kernels/reference/quant.py. It is not a serving GEMM. Do not
// quote its speed. The sidecar does not call it.
#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace inkling {

__device__ __forceinline__ double e2m1(unsigned code) {
  const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  double v = mag[code & 7u];
  return (code & 8u) ? -v : v;
}

__device__ __forceinline__ double e4m3(unsigned bits) {
  unsigned exp = (bits >> 3) & 0xFu;
  unsigned man = bits & 7u;
  if ((bits & 0x7Fu) == 0x7Fu) return nan("");
  double val = (exp == 0) ? man * 0.001953125
                           : (1.0 + man * 0.125) * exp2(static_cast<double>(static_cast<int>(exp) - 7));
  return (bits & 0x80u) ? -val : val;
}

__device__ __forceinline__ double nvfp4(const uint8_t* packed, const uint8_t* scale, int k,
                                        double amax, double denom) {
  unsigned byte = packed[k >> 1];
  unsigned nib = (k & 1) ? (byte >> 4) : (byte & 0xFu);
  return e2m1(nib) * e4m3(scale[k >> 4]) * (amax / denom);
}

__device__ __forceinline__ double mxfp8(const uint8_t* values, const uint8_t* scale, int k) {
  return e4m3(values[k]) * exp2(static_cast<double>(static_cast<int>(scale[k >> 5]) - 127));
}

__device__ __forceinline__ double swiglu(double gate, double up, double route) {
  double gate_c = 4.0 * tanh(gate / 4.0);
  double up_c = 25.0 * tanh(up / 25.0);
  double sig = 1.0 / (1.0 + exp(-gate));
  return gate_c * sig * up_c * route;
}

// x_dq is the activation after fouroversix (NVFP4) or MXFP8 dequant, one row per token.
// Weights are still packed. y is fp64 and must be zeroed by the caller.
__global__ void grouped_nvfp4(const double* x, const int* ids, const double* route,
                              const uint8_t* w13, const uint8_t* s13, const double* amax13,
                              const uint8_t* w2, const uint8_t* s2, const double* amax2,
                              double denom, int tokens, int hidden, int inter, int topk,
                              double* scratch, double* y) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens) return;
  const int two_i = inter * 2;
  double* gate_up = scratch + static_cast<long long>(t) * two_i;
  double* act = scratch + static_cast<long long>(tokens) * two_i + static_cast<long long>(t) * inter;
  const long long w13_row = static_cast<long long>(two_i) * (hidden / 2);
  const long long s13_row = static_cast<long long>(two_i) * (hidden / 16);
  const long long w2_row = static_cast<long long>(hidden) * (inter / 2);
  const long long s2_row = static_cast<long long>(hidden) * (inter / 16);
  for (int s = 0; s < topk; ++s) {
    int e = ids[t * topk + s];
    if (e < 0) continue;
    const uint8_t* pw = w13 + e * w13_row;
    const uint8_t* ps = s13 + e * s13_row;
    double a13 = amax13[e];
    for (int n = 0; n < two_i; ++n) {
      double acc = 0.0;
      const uint8_t* row = pw + static_cast<long long>(n) * (hidden / 2);
      const uint8_t* sc = ps + static_cast<long long>(n) * (hidden / 16);
      for (int k = 0; k < hidden; ++k) acc += nvfp4(row, sc, k, a13, denom) * x[t * hidden + k];
      gate_up[n] = acc;
    }
    double rw = route[t * topk + s];
    for (int i = 0; i < inter; ++i) act[i] = swiglu(gate_up[2 * i], gate_up[2 * i + 1], rw);
    const uint8_t* dw = w2 + e * w2_row;
    const uint8_t* ds = s2 + e * s2_row;
    double a2 = amax2[e];
    for (int h = 0; h < hidden; ++h) {
      double acc = 0.0;
      const uint8_t* row = dw + static_cast<long long>(h) * (inter / 2);
      const uint8_t* sc = ds + static_cast<long long>(h) * (inter / 16);
      for (int k = 0; k < inter; ++k) acc += nvfp4(row, sc, k, a2, denom) * act[k];
      y[t * hidden + h] += acc;
    }
  }
}

__global__ void grouped_mxfp8(const double* x, const int* ids, const double* route,
                              const uint8_t* w13, const uint8_t* s13, const uint8_t* w2, const uint8_t* s2,
                              int tokens, int hidden, int inter, int topk, double* scratch, double* y) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens) return;
  const int two_i = inter * 2;
  double* gate_up = scratch + static_cast<long long>(t) * two_i;
  double* act = scratch + static_cast<long long>(tokens) * two_i + static_cast<long long>(t) * inter;
  const long long w13_row = static_cast<long long>(two_i) * hidden;
  const long long s13_row = static_cast<long long>(two_i) * (hidden / 32);
  const long long w2_row = static_cast<long long>(hidden) * inter;
  const long long s2_row = static_cast<long long>(hidden) * (inter / 32);
  for (int s = 0; s < topk; ++s) {
    int e = ids[t * topk + s];
    if (e < 0) continue;
    const uint8_t* pw = w13 + e * w13_row;
    const uint8_t* ps = s13 + e * s13_row;
    for (int n = 0; n < two_i; ++n) {
      double acc = 0.0;
      const uint8_t* row = pw + static_cast<long long>(n) * hidden;
      const uint8_t* sc = ps + static_cast<long long>(n) * (hidden / 32);
      for (int k = 0; k < hidden; ++k) acc += mxfp8(row, sc, k) * x[t * hidden + k];
      gate_up[n] = acc;
    }
    double rw = route[t * topk + s];
    for (int i = 0; i < inter; ++i) act[i] = swiglu(gate_up[2 * i], gate_up[2 * i + 1], rw);
    const uint8_t* dw = w2 + e * w2_row;
    const uint8_t* ds = s2 + e * s2_row;
    for (int h = 0; h < hidden; ++h) {
      double acc = 0.0;
      const uint8_t* row = dw + static_cast<long long>(h) * inter;
      const uint8_t* sc = ds + static_cast<long long>(h) * (inter / 32);
      for (int k = 0; k < inter; ++k) acc += mxfp8(row, sc, k) * act[k];
      y[t * hidden + h] += acc;
    }
  }
}

inline int launch_nvfp4(const double* x, const int* ids, const double* route, const uint8_t* w13,
                        const uint8_t* s13, const double* amax13, const uint8_t* w2, const uint8_t* s2,
                        const double* amax2, double denom, int tokens, int hidden, int inter, int topk,
                        double* scratch, double* y) {
  grouped_nvfp4<<<tokens, 1>>>(x, ids, route, w13, s13, amax13, w2, s2, amax2, denom, tokens, hidden,
                               inter, topk, scratch, y);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

inline int launch_mxfp8(const double* x, const int* ids, const double* route, const uint8_t* w13,
                        const uint8_t* s13, const uint8_t* w2, const uint8_t* s2, int tokens, int hidden,
                        int inter, int topk, double* scratch, double* y) {
  grouped_mxfp8<<<tokens, 1>>>(x, ids, route, w13, s13, w2, s2, tokens, hidden, inter, topk, scratch, y);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // namespace inkling
