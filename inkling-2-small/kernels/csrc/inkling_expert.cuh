// One expert forward: group by expert, CUTLASS w13, SiTU, CUTLASS w2, scatter.
// Weights are expert-major in the stored packed layout. Scales are stored
// row-major (rows, K/block) and packed into the CUTLASS layout for the call.
#pragma once

#include <cuda_bf16.h>

#include <vector>

#include "inkling_arch.cuh"
#include "inkling_situ.cuh"

namespace inkling {
namespace expert {

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) {
    if (bytes && cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr;
  }
  ~Buf() { cudaFree(p); }
  template <class T>
  T* as() const { return static_cast<T*>(p); }
};

template <class Gemm, class ElementA, class ElementB, class Scale, int Block, bool IsNvfp4>
// w13_each, when set, is a host array of device pointers, one per expert.
// The contiguous bases are ignored for any tensor whose each-pointer is set.
// Hot experts in HBM and spill experts staged from Grace then share one launch.
int forward(int tokens, int topk, int experts, int hidden, int inter, const __nv_bfloat16* x,
            const int* ids, const float* route, const uint8_t* w13, const uint8_t* s13, const float* amax13,
            const uint8_t* w2, const uint8_t* s2, const float* amax2, float denom, __nv_bfloat16* y,
            cudaStream_t stream, const uint8_t* const* w13_each = nullptr,
            const uint8_t* const* s13_each = nullptr, const uint8_t* const* w2_each = nullptr,
            const uint8_t* const* s2_each = nullptr) {
  if (tokens <= 0) return 0;
  if (hidden % 128 || inter % 128) return 2;
  const int tile_k = IsNvfp4 ? arch::kNvfp4TileK : arch::kMxfp8TileK;
  if (hidden % tile_k || inter % tile_k) return 2;

  std::vector<int> ids_h(static_cast<size_t>(tokens) * topk);
  std::vector<float> route_h(ids_h.size());
  if (cudaMemcpy(ids_h.data(), ids, ids_h.size() * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(route_h.data(), route, route_h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 3;
  }
  std::vector<int> counts(experts);
  int total = 0;
  for (int i = 0; i < tokens * topk; ++i) {
    const int e = ids_h[i];
    if (e < 0) continue;
    if (e >= experts) return 2;
    counts[e] += 1;
    total += 1;
  }
  if (total == 0) return 0;

  std::vector<float> amax13_h(experts, 1.f), amax2_h(experts, 1.f);
  if (IsNvfp4) {
    if (cudaMemcpy(amax13_h.data(), amax13, experts * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(amax2_h.data(), amax2, experts * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 3;
    }
  }

  std::vector<int> base(experts), used, used_n;
  int cursor = 0;
  for (int e = 0; e < experts; ++e) {
    base[e] = cursor;
    if (counts[e]) {
      used.push_back(e);
      used_n.push_back(counts[e]);
      cursor += counts[e];
    }
  }
  std::vector<int> fill = base;
  std::vector<int> token_of(total);
  std::vector<float> route_of(total), w13_scale(total), w2_scale(total);
  for (int t = 0; t < tokens; ++t) {
    for (int s = 0; s < topk; ++s) {
      const int e = ids_h[t * topk + s];
      if (e < 0) continue;
      const int row = fill[e]++;
      token_of[row] = t;
      route_of[row] = route_h[t * topk + s];
      w13_scale[row] = IsNvfp4 ? amax13_h[e] / denom : 1.f;
      w2_scale[row] = IsNvfp4 ? amax2_h[e] / denom : 1.f;
    }
  }

  const int groups = static_cast<int>(used.size());
  const int pack = IsNvfp4 ? 2 : 1;
  Buf token_d(total * sizeof(int)), route_d(total * sizeof(float)), w13s_d(total * sizeof(float));
  Buf w2s_d(total * sizeof(float));
  Buf x_g(static_cast<size_t>(total) * hidden * sizeof(__nv_bfloat16));
  Buf gate(static_cast<size_t>(total) * (2 * inter) * sizeof(__nv_bfloat16));
  Buf act(static_cast<size_t>(total) * inter * sizeof(__nv_bfloat16));
  Buf down(static_cast<size_t>(total) * hidden * sizeof(__nv_bfloat16));
  Buf acc(static_cast<size_t>(tokens) * hidden * sizeof(float));
  if (!token_d.p || !x_g.p || !gate.p || !act.p || !down.p || !acc.p) return 3;
  cudaMemcpy(token_d.p, token_of.data(), total * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(route_d.p, route_of.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(w13s_d.p, w13_scale.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(w2s_d.p, w2_scale.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(acc.p, 0, static_cast<size_t>(tokens) * hidden * sizeof(float));

  situ::gather_x<<<dim3((hidden + 127) / 128, total), 128, 0, stream>>>(
      x, token_d.as<int>(), x_g.as<__nv_bfloat16>(), total, hidden);

  const int x_blocks = hidden / Block;
  const int act_blocks = inter / Block;
  Buf x_q(static_cast<size_t>(total) * (hidden / pack));
  Buf x_sf(static_cast<size_t>(total) * x_blocks);
  Buf x_sf_sw(static_cast<size_t>(total) * x_blocks * 4);
  Buf act_q(static_cast<size_t>(total) * (inter / pack));
  Buf act_sf(static_cast<size_t>(total) * act_blocks);
  Buf act_sf_sw(static_cast<size_t>(total) * act_blocks * 4);
  Buf act_amax(total * sizeof(float));
  if (!x_q.p || !act_q.p) return 3;

  if (IsNvfp4) {
    situ::quant_nvfp4_kernel<<<total, 32, 0, stream>>>(x_g.as<__nv_bfloat16>(), x_q.as<uint8_t>(),
                                                       x_sf.as<uint8_t>(), act_amax.as<float>(), total, hidden);
  } else {
    situ::quant_mxfp8_kernel<<<total, 32, 0, stream>>>(x_g.as<__nv_bfloat16>(), x_q.as<uint8_t>(),
                                                       x_sf.as<uint8_t>(), total, hidden);
  }

  auto group_ptrs = [&](const uint8_t* base, const uint8_t* const* each, int bytes_per_expert,
                        std::vector<const void*>& out) {
    out.resize(groups);
    for (int g = 0; g < groups; ++g) {
      const int e = used[g];
      out[g] = each ? static_cast<const void*>(each[e])
                    : static_cast<const void*>(base + static_cast<size_t>(e) * bytes_per_expert);
    }
  };
  std::vector<const void*> a13, s13p, b13, sb13, a2, s2p, b2, sb2;
  std::vector<void*> d13, d2;
  const int w13_bytes = (2 * inter) * (hidden / pack);
  const int s13_bytes = (2 * inter) * (hidden / Block);
  const int w2_bytes = hidden * (inter / pack);
  const int s2_bytes = hidden * (inter / Block);
  group_ptrs(w13, w13_each, w13_bytes, a13);
  group_ptrs(s13, s13_each, s13_bytes, s13p);
  group_ptrs(w2, w2_each, w2_bytes, a2);
  group_ptrs(s2, s2_each, s2_bytes, s2p);

  // Swizzle each group's stored scales. Weight scales do not depend on the
  // token count for their values, but the CUTLASS layout is a function of N,
  // so the pack uses that group's token count.
  Buf s13_sw(static_cast<size_t>(groups) * s13_bytes * 4);
  Buf s2_sw(static_cast<size_t>(groups) * s2_bytes * 4);
  if (!s13_sw.p || !s2_sw.p) return 3;
  std::vector<const void*> s13_sw_p(groups), s2_sw_p(groups);
  int row0 = 0;
  for (int g = 0; g < groups; ++g) {
    const int n = used_n[g];
    uint8_t* dst13 = s13_sw.as<uint8_t>() + static_cast<size_t>(g) * s13_bytes * 4;
    uint8_t* dst2 = s2_sw.as<uint8_t>() + static_cast<size_t>(g) * s2_bytes * 4;
    if (grouped::pack_weight_scales<Gemm>(2 * inter, hidden / Block, n, hidden,
                                          static_cast<const uint8_t*>(s13p[g]), dst13, stream) ||
        grouped::pack_weight_scales<Gemm>(hidden, inter / Block, n, inter,
                                          static_cast<const uint8_t*>(s2p[g]), dst2, stream)) {
      return 1;
    }
    s13_sw_p[g] = dst13;
    s2_sw_p[g] = dst2;
    (void)row0;
  }

  std::vector<const void*> b_ptr(groups), bsf_ptr(groups);
  std::vector<void*> d_ptr(groups);
  int off = 0;
  Buf xsf_sw(static_cast<size_t>(total) * x_blocks * 4);
  if (!xsf_sw.p) return 3;
  for (int g = 0; g < groups; ++g) {
    const int n = used_n[g];
    b_ptr[g] = x_q.as<uint8_t>() + static_cast<size_t>(off) * (hidden / pack);
    uint8_t* sf = xsf_sw.as<uint8_t>() + static_cast<size_t>(off) * x_blocks * 4;
    if (grouped::pack_act_scales<Gemm>(n, x_blocks, 2 * inter, hidden,
                                       x_sf.as<uint8_t>() + static_cast<size_t>(off) * x_blocks, sf, stream)) {
      return 1;
    }
    bsf_ptr[g] = sf;
    d_ptr[g] = gate.as<__nv_bfloat16>() + static_cast<size_t>(off) * (2 * inter);
    off += n;
  }

  grouped::Launch up{};
  up.groups = groups;
  up.M = 2 * inter;
  up.K = hidden;
  up.tokens = used_n.data();
  up.A = a13.data();
  up.B = b_ptr.data();
  up.SFA = s13_sw_p.data();
  up.SFB = bsf_ptr.data();
  up.D = d_ptr.data();
  up.stream = stream;
  if (int rc = grouped::launch_grouped<Gemm, ElementA, ElementB, Scale>(up)) return rc;

  situ::situ_kernel<<<dim3((inter + 127) / 128, total), 128, 0, stream>>>(
      gate.as<__nv_bfloat16>(), route_d.as<float>(), w13s_d.as<float>(), act.as<__nv_bfloat16>(), total, inter);

  Buf act_row_amax(total * sizeof(float));
  if (IsNvfp4) {
    situ::quant_nvfp4_kernel<<<total, 32, 0, stream>>>(act.as<__nv_bfloat16>(), act_q.as<uint8_t>(),
                                                       act_sf.as<uint8_t>(), act_row_amax.as<float>(), total, inter);
  } else {
    situ::quant_mxfp8_kernel<<<total, 32, 0, stream>>>(act.as<__nv_bfloat16>(), act_q.as<uint8_t>(),
                                                       act_sf.as<uint8_t>(), total, inter);
  }

  off = 0;
  Buf asf_sw(static_cast<size_t>(total) * act_blocks * 4);
  if (!asf_sw.p) return 3;
  for (int g = 0; g < groups; ++g) {
    const int n = used_n[g];
    b_ptr[g] = act_q.as<uint8_t>() + static_cast<size_t>(off) * (inter / pack);
    uint8_t* sf = asf_sw.as<uint8_t>() + static_cast<size_t>(off) * act_blocks * 4;
    if (grouped::pack_act_scales<Gemm>(n, act_blocks, hidden, inter,
                                       act_sf.as<uint8_t>() + static_cast<size_t>(off) * act_blocks, sf, stream)) {
      return 1;
    }
    bsf_ptr[g] = sf;
    d_ptr[g] = down.as<__nv_bfloat16>() + static_cast<size_t>(off) * hidden;
    off += n;
  }
  grouped::Launch down_l{};
  down_l.groups = groups;
  down_l.M = hidden;
  down_l.K = inter;
  down_l.tokens = used_n.data();
  down_l.A = a2.data();
  down_l.B = b_ptr.data();
  down_l.SFA = s2_sw_p.data();
  down_l.SFB = bsf_ptr.data();
  down_l.D = d_ptr.data();
  down_l.stream = stream;
  if (int rc = grouped::launch_grouped<Gemm, ElementA, ElementB, Scale>(down_l)) return rc;

  // NVFP4 down-proj scale is (activation amax / 1536) * (weight amax / denominator).
  // MXFP8 has no tensor-level scale; the UE8M0 block scale is inside the MMA.
  std::vector<float> out_scale(total, 1.f);
  if (IsNvfp4) {
    std::vector<float> row_amax(total);
    cudaMemcpy(row_amax.data(), act_row_amax.as<float>(), total * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < total; ++i) out_scale[i] = (row_amax[i] / 1536.f) * w2_scale[i];
  }
  Buf scale_d(total * sizeof(float));
  cudaMemcpy(scale_d.p, out_scale.data(), total * sizeof(float), cudaMemcpyHostToDevice);
  situ::scatter_kernel<<<dim3((hidden + 127) / 128, total), 128, 0, stream>>>(
      down.as<__nv_bfloat16>(), token_d.as<int>(), scale_d.as<float>(), acc.as<float>(), total, hidden);
  // Same stream, so the atomics are visible to the store. Adds into y.
  situ::store_bf16<<<dim3((hidden + 127) / 128, tokens), 128, 0, stream>>>(
      acc.as<float>(), y, tokens, hidden);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // namespace expert
}  // namespace inkling
