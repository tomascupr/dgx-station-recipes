# SPDX-License-Identifier: Apache-2.0
# experts_fp8 is adapted from vLLM's fused_experts_impl (vllm/model_executor/layers/fused_moe/fused_moe.py at
# a2bcb5657a881a002e79d0adab3c6b5e39436121), Copyright contributors to the vLLM project, Apache-2.0
# (peer/LICENSE.vllm). Changed: the clamped SwiGLU activation, the expert map, a tile override; see NOTICE.
"""Expert computation for the peer tier: GLM-5.3-Flash's FP8 experts on whichever GPU holds them.

experts_fp8() is vLLM's fused_experts_impl (Triton, FP8 weights with 128x128 block scales, FP8 activations with 1x128
scales) with two changes GLM needs: its clamped SwiGLU, silu(min(gate, L)) * clamp(up, -L, L) with L = swiglu_limit,
which the functional wrapper does not apply; and an expert map, so a GPU holding only some experts computes only the
routes that land on them. The same function runs on the GB300 (hot experts) and the RTX PRO 6000 (cold experts).
"""
import functools

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig, MoEActivation, apply_moe_activation)
from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment, dispatch_fused_moe_kernel, try_get_optimal_moe_config)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.triton_utils import tl

BLOCK = [128, 128]


def experts_fp8(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w1_scale: torch.Tensor, w2_scale: torch.Tensor,
                topk_weights: torch.Tensor, topk_ids: torch.Tensor, expert_map: torch.Tensor,
                clamp_limit: float, config: dict | None = None) -> torch.Tensor:
    """Sum over the routes this GPU owns of weight * expert(x). Routes whose expert maps to -1 contribute zero.

    x [M, H] bf16; w1 [E_local, 2I, H] and w2 [E_local, H, I] float8_e4m3fn with float32 block scales;
    topk_* [M, k] in global expert ids; expert_map [E_global] int32, global id -> local slot or -1.
    """
    M, k = topk_ids.shape
    _, N, _ = w1.shape
    K = w2.size(1)
    config = config or functools.partial(try_get_optimal_moe_config, w1.size(), w2.size(), k,
                               _get_config_dtype_str(use_fp8_w8a8=True, use_int8_w8a16=False,
                                                     use_int4_w4a16=False, dtype=x.dtype),
                               block_shape=BLOCK)(M)
    cache13 = torch.empty(M * k * max(N, K), device=x.device, dtype=x.dtype)
    cache1 = cache13[: M * k * N].view(M, k, N)
    cache3 = cache13[: M * k * K].view(M, k, K)
    cache2 = torch.empty((M * k, N // 2), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    qx, qx_scale = moe_kernel_quantize_input(A=x, A_scale=None, quant_dtype=torch.float8_e4m3fn,
                                             per_act_token_quant=False, block_shape=BLOCK)
    sorted_ids, expert_ids, n_post_pad = _prepare_expert_assignment(
        topk_ids, config, M, k, expert_map.numel(), expert_map, block_shape=BLOCK, ignore_invalid_experts=True)
    dispatch_fused_moe_kernel(qx, w1, cache1, qx_scale, w1_scale, None, topk_weights, sorted_ids, expert_ids,
                              n_post_pad, False, k, config, compute_type=tl.bfloat16, use_fp8_w8a8=True,
                              use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                              per_channel_quant=False, block_shape=BLOCK, B_bias=None)
    apply_moe_activation(MoEActivation.SILU, cache2, cache1.view(-1, N),
                         activation_config=ApplyMoEActivationConfig(clamp_limit=clamp_limit))
    qc2, qc2_scale = moe_kernel_quantize_input(A=cache2, A_scale=None, quant_dtype=torch.float8_e4m3fn,
                                               per_act_token_quant=False, block_shape=BLOCK)
    cache3.zero_()
    dispatch_fused_moe_kernel(qc2, w2, cache3, qc2_scale, w2_scale, None, topk_weights, sorted_ids, expert_ids,
                              n_post_pad, True, 1, config, compute_type=tl.bfloat16, use_fp8_w8a8=True,
                              use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                              per_channel_quant=False, block_shape=BLOCK, B_bias=None)
    ops.moe_sum(cache3, out)
    return out
