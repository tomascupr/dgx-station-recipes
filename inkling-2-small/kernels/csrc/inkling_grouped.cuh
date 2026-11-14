// Serving GEMMs. CUTLASS b2f249acd203.
//
// Layout matches examples 90 (SM103 NVFP4 grouped), 79d (SM120 NVFP4 grouped)
// and 92 (MoE: the short side is N). Each expert is one group,
//   M = weight rows, N = tokens that hit that expert, K = contraction.
// N tile 64 covers a decode expert of 1..64 tokens in one tile. SM103's
// builder only accepts e2m1, so MXFP8 on the GB300 uses the SM100 MXF8
// grouped schedule (tcgen05 MXF8 is enabled for sm_103a). Tile N there is
// 128, which still covers 1..64 tokens in one predicated tile.
//
// Scales passed to launch_grouped are already in CUTLASS block-scaled
// layout. pack_scales() writes that layout from the checkpoint's stored
// shape: one scale per row, 16 (NVFP4) or 32 (MXFP8) along K.
//
// Not executed on a device yet. The FP64 kernels in inkling_gemm.cuh are the
// numeric reference.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/detail/collective/moe_stride_utils.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

namespace inkling {
namespace grouped {

struct Launch {
  int groups = 0;
  int M = 0;
  int K = 0;
  // Host, length groups. N of that group (tokens). 0 is skipped by the caller.
  const int* tokens = nullptr;
  // Device pointer arrays, length groups. Null entries are not allowed;
  // the caller omits empty experts.
  const void* const* A = nullptr;
  const void* const* B = nullptr;
  const void* const* SFA = nullptr;
  const void* const* SFB = nullptr;
  void* const* D = nullptr;
  const float* alpha = nullptr;  // host, length groups, or null for 1
  cudaStream_t stream = nullptr;
};

template <class T, class = void>
struct storage_of { using type = T; };
template <class T>
struct storage_of<T, std::void_t<typename T::DataType>> { using type = typename T::DataType; };
template <class A, class B>
struct storage_of<cute::tuple<A, B>> { using type = typename storage_of<A>::type; };

template <class Gemm, class ElementA, class ElementB, class Scale>
int launch_grouped(const Launch& in) {
  using namespace cute;
  using Problem = cute::Shape<int, int, int>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Blk = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using AStore = typename storage_of<ElementA>::type;
  using BStore = typename storage_of<ElementB>::type;
  using SFAStore = typename storage_of<Scale>::type;

  if (in.groups <= 0) return 0;
  std::vector<Problem> problems(in.groups);
  std::vector<StrideA> stride_a(in.groups);
  std::vector<StrideB> stride_b(in.groups);
  std::vector<StrideC> stride_c(in.groups);
  std::vector<StrideD> stride_d(in.groups);
  std::vector<LayoutSFA> layout_sfa(in.groups);
  std::vector<LayoutSFB> layout_sfb(in.groups);
  std::vector<const AStore*> ptr_a(in.groups);
  std::vector<const BStore*> ptr_b(in.groups);
  std::vector<const SFAStore*> ptr_sfa(in.groups);
  std::vector<const SFAStore*> ptr_sfb(in.groups);
  std::vector<cutlass::bfloat16_t*> ptr_d(in.groups);
  std::vector<float> alpha(in.groups, 1.f);
  for (int g = 0; g < in.groups; ++g) {
    const int N = in.tokens[g];
    if (N <= 0) return 2;
    problems[g] = make_shape(in.M, N, in.K);
    stride_a[g] = cutlass::make_cute_packed_stride(StrideA{}, make_shape(in.M, in.K, 1));
    stride_b[g] = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, in.K, 1));
    stride_c[g] = cutlass::make_cute_packed_stride(StrideC{}, make_shape(in.M, N, 1));
    stride_d[g] = stride_c[g];
    // Rank-4 problem shape matches InternalLayoutSFA/B, which include the
    // batch mode. tile_atom_to_shape on a rank-3 shape drops that mode.
    layout_sfa[g] = Blk::tile_atom_to_shape_SFA(make_shape(in.M, N, in.K, 1));
    layout_sfb[g] = Blk::tile_atom_to_shape_SFB(make_shape(in.M, N, in.K, 1));
    ptr_a[g] = static_cast<const AStore*>(in.A[g]);
    ptr_b[g] = static_cast<const BStore*>(in.B[g]);
    ptr_sfa[g] = static_cast<const SFAStore*>(in.SFA[g]);
    ptr_sfb[g] = static_cast<const SFAStore*>(in.SFB[g]);
    ptr_d[g] = static_cast<cutlass::bfloat16_t*>(in.D[g]);
    if (in.alpha) alpha[g] = in.alpha[g];
  }

  auto upload = [](const void* src, size_t bytes, void** dst) -> int {
    if (cudaMalloc(dst, bytes) != cudaSuccess) return 3;
    if (cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    return 0;
  };
  void *d_prob = nullptr, *d_sa = nullptr, *d_sb = nullptr, *d_sc = nullptr, *d_sd = nullptr;
  void *d_lsfa = nullptr, *d_lsfb = nullptr;
  void *d_pa = nullptr, *d_pb = nullptr, *d_psfa = nullptr, *d_psfb = nullptr, *d_pd = nullptr;
  void *d_alpha = nullptr;
  int rc = 0;
  auto step = [&](const void* src, size_t bytes, void** dst) {
    if (rc) return;
    rc = upload(src, bytes, dst);
  };
  step(problems.data(), sizeof(Problem) * in.groups, &d_prob);
  step(stride_a.data(), sizeof(StrideA) * in.groups, &d_sa);
  step(stride_b.data(), sizeof(StrideB) * in.groups, &d_sb);
  step(stride_c.data(), sizeof(StrideC) * in.groups, &d_sc);
  step(stride_d.data(), sizeof(StrideD) * in.groups, &d_sd);
  step(layout_sfa.data(), sizeof(LayoutSFA) * in.groups, &d_lsfa);
  step(layout_sfb.data(), sizeof(LayoutSFB) * in.groups, &d_lsfb);
  step(ptr_a.data(), sizeof(void*) * in.groups, &d_pa);
  step(ptr_b.data(), sizeof(void*) * in.groups, &d_pb);
  step(ptr_sfa.data(), sizeof(void*) * in.groups, &d_psfa);
  step(ptr_sfb.data(), sizeof(void*) * in.groups, &d_psfb);
  step(ptr_d.data(), sizeof(void*) * in.groups, &d_pd);
  if (rc) return 3;

  cutlass::KernelHardwareInfo hw{};
  hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);

  typename Gemm::Arguments arguments{};
  auto& fusion = arguments.epilogue.thread;
  // One alpha for every group. Per-expert NVFP4 scales are applied in the
  // SiTU and the scatter, not through alpha_ptr_array (that field is
  // const float* const*, one pointer per group).
  fusion.alpha = 1.f;
  fusion.beta = 0.f;
  fusion.alpha_ptr = nullptr;
  fusion.alpha_ptr_array = nullptr;
  fusion.beta_ptr = nullptr;
  fusion.beta_ptr_array = nullptr;
  (void)alpha;

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};
  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {in.groups, static_cast<Problem*>(d_prob), problems.data()},
      {static_cast<const AStore**>(d_pa), static_cast<StrideA*>(d_sa),
       static_cast<const BStore**>(d_pb), static_cast<StrideB*>(d_sb),
       static_cast<const SFAStore**>(d_psfa), static_cast<LayoutSFA*>(d_lsfa),
       static_cast<const SFAStore**>(d_psfb), static_cast<LayoutSFB*>(d_lsfb)},
      {fusion, static_cast<const cutlass::bfloat16_t**>(d_pd), static_cast<StrideC*>(d_sc),
       static_cast<cutlass::bfloat16_t**>(d_pd), static_cast<StrideD*>(d_sd)},
      hw, scheduler};

  Gemm gemm;
  const auto can = gemm.can_implement(arguments);
  void* workspace = nullptr;
  const size_t bytes = Gemm::get_workspace_size(arguments);
  if (can == cutlass::Status::kSuccess && bytes && cudaMalloc(&workspace, bytes) != cudaSuccess) {
    rc = 3;
  }
  if (rc == 0 && can == cutlass::Status::kSuccess) {
    if (gemm.initialize(arguments, workspace) != cutlass::Status::kSuccess ||
        gemm.run(in.stream) != cutlass::Status::kSuccess) {
      rc = 1;
    }
  } else if (can != cutlass::Status::kSuccess) {
    rc = 1;
  }
  cudaFree(workspace);
  cudaFree(d_prob); cudaFree(d_sa); cudaFree(d_sb); cudaFree(d_sc); cudaFree(d_sd);
  cudaFree(d_lsfa); cudaFree(d_lsfb);
  cudaFree(d_pa); cudaFree(d_pb); cudaFree(d_psfa); cudaFree(d_psfb); cudaFree(d_pd);
  cudaFree(d_alpha);
  return rc;
}

// Stored scale is row-major (rows, K/block): one scale per row, block along K.
template <class Layout>
__global__ void pack_sf_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                               Layout layout, int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  // Layout is rank 3: (row, k-block, batch). Batch is 1.
  dst[layout(cute::make_coord(idx / cols, idx % cols, 0))] = src[idx];
}

template <class Gemm>
int pack_weight_scales(int rows, int k_blocks, int tokens, int K, const uint8_t* src, uint8_t* dst,
                       cudaStream_t stream) {
  using Blk = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout = Blk::tile_atom_to_shape_SFA(cute::make_shape(rows, tokens, K, 1));
  const int n = rows * k_blocks;
  pack_sf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, layout, rows, k_blocks);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

template <class Gemm>
int pack_act_scales(int tokens, int k_blocks, int rows, int K, const uint8_t* src, uint8_t* dst,
                    cudaStream_t stream) {
  using Blk = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout = Blk::tile_atom_to_shape_SFB(cute::make_shape(rows, tokens, K, 1));
  const int n = tokens * k_blocks;
  pack_sf_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, layout, tokens, k_blocks);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // namespace grouped
}  // namespace inkling
