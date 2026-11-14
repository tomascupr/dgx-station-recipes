// GEMM types for one target. The .cu sets INKLING_ARCH to 103 or 120
// before including this. CUTLASS b2f249acd203.
#pragma once

#include "inkling_grouped.cuh"

namespace inkling {
namespace arch {

template <class Arch, class ElementA, class ElementB, class Tile, class Cluster, class MainSchedule,
          class EpiSchedule, int AlignA, int AlignC>
struct MakeGemm {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, cutlass::bfloat16_t,
      cutlass::layout::RowMajor*, AlignC, cutlass::bfloat16_t, cutlass::layout::RowMajor*, AlignC,
      EpiSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, cutlass::layout::RowMajor*, AlignA,
      ElementB, cutlass::layout::ColumnMajor*, AlignA, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainSchedule>::CollectiveOp;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>, CollectiveMainloop,
      CollectiveEpilogue>>;
};

#if INKLING_ARCH == 103

// CTA N must be 128, 192, or 256
// (sm103_blockscaled_mma_array_warpspecialized.hpp:140). The SM103 builder
// also only allows M of 128 or 256 (sm103_blockscaled_umma_builder.inl), so
// putting tokens on M (TRT-LLM's swap_ab, moe_gemm_template_dispatch.h:668)
// is the same 128-wide predicate, not a smaller tile. Tile N = 128 is the
// smallest legal CTA N: a decode expert of 1..128 tokens is one predicated
// tile. K = 768 is required by that builder and divides 4608 and 2304.
// vLLM's inkling2 grouped_gemm.py masks a 64-row Triton tile; that kernel is
// BF16 and is not this MMA.
using Nvfp4 = MakeGemm<
    cutlass::arch::Sm103, cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue4m3_t>,
    cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue4m3_t>,
    cute::Shape<cute::_128, cute::_128, cute::Int<768>>, cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103,
    cutlass::epilogue::PtrArrayNoSmemWarpSpecialized1Sm, 32, 8>;
using Nvfp4Gemm = Nvfp4::Gemm;
using Nvfp4A = cutlass::float_e2m1_t;
using Nvfp4Scale = cutlass::float_ue4m3_t;
inline constexpr int kNvfp4Block = 16;
inline constexpr int kNvfp4TileK = 768;

// Sm103's builder is FP4-only (sm103_make_blockscaled_1sm_tiled_mma). MXFP8
// uses the SM100 grouped MXF8 schedule; cute enables tcgen05 MXF8 for sm_103a.
using Mxfp8 = MakeGemm<
    cutlass::arch::Sm100, cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::mx_float8_t<cutlass::float_e4m3_t>, cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm, 16, 8>;
using Mxfp8Gemm = Mxfp8::Gemm;
using Mxfp8A = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Mxfp8Scale = cutlass::float_ue8m0_t;
inline constexpr int kMxfp8Block = 32;
inline constexpr int kMxfp8TileK = 128;

#elif INKLING_ARCH == 120

// 79d's element types and KernelScheduleAuto, 79a's bf16 epilogue.
// Tile N is 64, not 79d's 128: N=128 spilled about 80 B at 168 registers
// and ptxas would not use more. N=64 compiles on sm_120a (the builder
// allows N >= 8) and ptxas reports 0 spills at 118-119 registers.
// There is no separate stage-count knob on KernelScheduleAuto.
// KernelPtrArrayTmaWarpSpecializedPingpongBlockScaledSm120
// (dispatch_policy.hpp:640) is a template; 79d does not instantiate it.
using Nvfp4 = MakeGemm<
    cutlass::arch::Sm120, cutlass::nv_float4_t<cutlass::float_e2m1_t>,
    cutlass::nv_float4_t<cutlass::float_e2m1_t>,     cute::Shape<cute::_128, cute::_64, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto, 32, 8>;
using Nvfp4Gemm = Nvfp4::Gemm;
using Nvfp4A = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Nvfp4Scale = cutlass::float_ue4m3_t;
inline constexpr int kNvfp4Block = 16;
inline constexpr int kNvfp4TileK = 128;

// Same 128x64x128 tile as NVFP4 on this target, for the same spill reason.
// Element type and alignment 16 follow 79c's MXFP8 operand.
using Mxfp8 = MakeGemm<
    cutlass::arch::Sm120, cutlass::mx_float8_t<cutlass::float_e4m3_t>,
    cutlass::mx_float8_t<cutlass::float_e4m3_t>,     cute::Shape<cute::_128, cute::_64, cute::_128>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueScheduleAuto, 16, 8>;
using Mxfp8Gemm = Mxfp8::Gemm;
using Mxfp8A = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Mxfp8Scale = cutlass::float_ue8m0_t;
inline constexpr int kMxfp8Block = 32;
inline constexpr int kMxfp8TileK = 128;

#else
#error "set INKLING_ARCH to 103 or 120"
#endif

}  // namespace arch
}  // namespace inkling
