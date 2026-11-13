// SiTU between the two grouped GEMMs, plus the activation quant the down
// projection consumes. Not a tensor-core kernel. The sigmoid input is the
// unclamped gate: vLLM d664cd1e vllm/models/inkling2/ops/activation.py:75-78.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inkling {
namespace situ {

__device__ __forceinline__ float e4m3_f(unsigned bits) {
  unsigned exp = (bits >> 3) & 0xFu;
  unsigned man = bits & 7u;
  if ((bits & 0x7Fu) == 0x7Fu) return nanf("");
  float val = (exp == 0) ? man * 0.001953125f
                          : (1.f + man * 0.125f) * exp2f(static_cast<float>(static_cast<int>(exp) - 7));
  return (bits & 0x80u) ? -val : val;
}

__device__ __forceinline__ unsigned nearest_e4m3(float x) {
  unsigned best = 0;
  float best_d = fabsf(x);
  for (unsigned c = 1; c < 256; ++c) {
    if ((c & 0x7Fu) == 0x7Fu) continue;
    float d = fabsf(x - e4m3_f(c));
    if (d < best_d) {
      best_d = d;
      best = c;
    }
  }
  return best;
}

__device__ __forceinline__ float round_e2m1(float x) {
  float ax = fabsf(x);
  float mag = 0.f;
  if (ax > 0.25f && ax < 0.75f) mag = 0.5f;
  else if (ax >= 0.75f && ax <= 1.25f) mag = 1.f;
  else if (ax > 1.25f && ax < 1.75f) mag = 1.5f;
  else if (ax >= 1.75f && ax <= 2.5f) mag = 2.f;
  else if (ax > 2.5f && ax < 3.5f) mag = 3.f;
  else if (ax >= 3.5f && ax <= 5.f) mag = 4.f;
  else if (ax > 5.f) mag = 6.f;
  return copysignf(mag, x);
}

__device__ __forceinline__ unsigned e2m1_code(float x) {
  float r = round_e2m1(x);
  float ax = fabsf(r);
  unsigned code = 0;
  if (ax == 0.5f) code = 1;
  else if (ax == 1.f) code = 2;
  else if (ax == 1.5f) code = 3;
  else if (ax == 2.f) code = 4;
  else if (ax == 3.f) code = 5;
  else if (ax == 4.f) code = 6;
  else if (ax == 6.f) code = 7;
  if (r < 0.f) code |= 8u;
  return code;
}

__global__ void gather_x(const __nv_bfloat16* __restrict__ x, const int* __restrict__ token,
                         __nv_bfloat16* __restrict__ dst, int rows, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= rows || i >= hidden) return;
  dst[row * hidden + i] = x[token[row] * hidden + i];
}

// gate_up is interleaved (gate, up) per column, contiguous per row.
// w_scale is the NVFP4 weight global scale (amax/denominator), 1 for MXFP8.
__global__ void situ_kernel(const __nv_bfloat16* __restrict__ gate_up, const float* __restrict__ route,
                            const float* __restrict__ w_scale, __nv_bfloat16* __restrict__ act, int rows,
                            int inter) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= rows || i >= inter) return;
  const float scale = w_scale[row];
  float gate = __bfloat162float(gate_up[row * (2 * inter) + 2 * i]) * scale;
  float up = __bfloat162float(gate_up[row * (2 * inter) + 2 * i + 1]) * scale;
  const float sig = 1.f / (1.f + expf(-gate));
  const float gate_c = 4.f * tanhf(gate / 4.f);
  const float up_c = 25.f * tanhf(up / 25.f);
  act[row * inter + i] = __float2bfloat16(gate_c * sig * up_c * route[row]);
}

__global__ void quant_nvfp4_kernel(const __nv_bfloat16* __restrict__ act, uint8_t* __restrict__ packed,
                                   uint8_t* __restrict__ scales, float* __restrict__ amax, int rows, int k) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  float row_max = 0.f;
  for (int i = 0; i < k; ++i) row_max = fmaxf(row_max, fabsf(__bfloat162float(act[row * k + i])));
  amax[row] = row_max;
  const float outer = row_max == 0.f ? 0.f : row_max / 1536.f;
  const int blocks = k / 16;
  for (int b = 0; b < blocks; ++b) {
    float vals[16];
    float bmax = 0.f;
    for (int j = 0; j < 16; ++j) {
      vals[j] = __bfloat162float(act[row * k + b * 16 + j]);
      bmax = fmaxf(bmax, fabsf(vals[j]));
    }
    auto score = [&](float scale_code_f) {
      float err = 0.f;
      const float scale = e4m3_f(nearest_e4m3(scale_code_f));
      for (int j = 0; j < 16; ++j) {
        float q = (scale == 0.f || outer == 0.f) ? 0.f : vals[j] / (scale * outer);
        err += fabsf(round_e2m1(q) * scale * outer - vals[j]);
      }
      return err;
    };
    const float raw6 = (row_max == 0.f) ? 0.f : fminf(bmax * (1536.f / row_max) / 6.f, 448.f);
    const float raw4 = (row_max == 0.f) ? 0.f : fminf(bmax * (1536.f / row_max) / 4.f, 448.f);
    const bool pick4 = score(raw4) < score(raw6);
    const unsigned sc = nearest_e4m3(pick4 ? raw4 : raw6);
    const float scale = e4m3_f(sc);
    scales[row * blocks + b] = static_cast<uint8_t>(sc);
    for (int j = 0; j < 16; j += 2) {
      float q0 = (scale == 0.f || outer == 0.f) ? 0.f : vals[j] / (scale * outer);
      float q1 = (scale == 0.f || outer == 0.f) ? 0.f : vals[j + 1] / (scale * outer);
      packed[row * (k / 2) + (b * 16 + j) / 2] =
          static_cast<uint8_t>(e2m1_code(q0) | (e2m1_code(q1) << 4));
    }
  }
}

__global__ void quant_mxfp8_kernel(const __nv_bfloat16* __restrict__ act, uint8_t* __restrict__ values,
                                   uint8_t* __restrict__ scales, int rows, int k) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  const int blocks = k / 32;
  for (int b = 0; b < blocks; ++b) {
    float vals[32];
    float bmax = 0.f;
    for (int j = 0; j < 32; ++j) {
      vals[j] = __bfloat162float(act[row * k + b * 32 + j]);
      bmax = fmaxf(bmax, fabsf(vals[j]));
    }
    int byte = 0;
    if (bmax != 0.f) {
      float exp = ceilf(log2f(bmax / 448.f));
      byte = static_cast<int>(exp) + 127;
      if (byte < 0) byte = 0;
      if (byte > 254) byte = 254;
    }
    scales[row * blocks + b] = static_cast<uint8_t>(byte);
    const float scale = (byte == 0) ? 0.f : exp2f(static_cast<float>(byte - 127));
    for (int j = 0; j < 32; ++j) {
      float q = scale == 0.f ? 0.f : vals[j] / scale;
      values[row * k + b * 32 + j] = static_cast<uint8_t>(nearest_e4m3(q));
    }
  }
}

__global__ void scatter_kernel(const __nv_bfloat16* __restrict__ grouped, const int* __restrict__ token,
                               const float* __restrict__ scale, float* __restrict__ acc, int rows, int hidden) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= rows || h >= hidden) return;
  atomicAdd(acc + token[row] * hidden + h, __bfloat162float(grouped[row * hidden + h]) * scale[row]);
}

__global__ void store_bf16(const float* __restrict__ acc, __nv_bfloat16* __restrict__ y, int tokens, int hidden) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (t >= tokens || h >= hidden) return;
  y[t * hidden + h] = __float2bfloat16(__bfloat162float(y[t * hidden + h]) + acc[t * hidden + h]);
}

}  // namespace situ
}  // namespace inkling
