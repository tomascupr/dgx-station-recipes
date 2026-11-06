"""The GB300's hot experts on FlashInfer's TRT-LLM FP8 block-scale MoE (routed variant), as vLLM's own TRT-LLM path does.

prepare() takes one layer's hot experts in the checkpoint layout ([gate; up] rows, 128x128 block scales) and returns
what the kernel wants: [up; gate] rows and scales, weights shuffled into the BlockMajorK layout. Experts are renumbered
so the hot ones are 0..n_hot-1 and the cold ones follow; the kernel is told its local experts are [0, n_hot), so
routes to cold experts contribute nothing here (the RTX computes them). GLM's SwiGLU clamp goes in as
gemm1_clamp_limit, which is how vLLM serves this model on the TRT-LLM backend.
"""
import torch
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import prepare_deepseek_fp8_x_sf
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    _shuffle_deepseek_fp8_moe_weights, swap_w13_to_w31)

E, I_SIZE = 288, 2048


def prepare(hot: list, w13: torch.Tensor, s13: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor, clamp: float):
    """w13 [n_hot, 2I, H], s13, w2 [n_hot, H, I], s2 on the GPU, rows in the order of `hot`. Returns a state dict."""
    dev = w13.device
    renum = torch.empty(E, dtype=torch.int32)
    cold = [e for e in range(E) if e not in set(hot)]
    renum[torch.tensor(hot)] = torch.arange(len(hot), dtype=torch.int32)
    renum[torch.tensor(cold)] = torch.arange(len(hot), E, dtype=torch.int32)
    w31, s31 = swap_w13_to_w31(w13), swap_w13_to_w31(s13)
    w31, w2b = _shuffle_deepseek_fp8_moe_weights(w31, w2)
    return dict(renum=renum.to(dev), w1=w31, s1=s31.contiguous(), w2=w2b, s2=s2.contiguous(), n_hot=len(hot),
                clamp=torch.full((len(hot),), clamp, dtype=torch.float32, device=dev))


def hot_experts(st: dict, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """Sum over hot routes of weight * expert(x); topk ids are global (0..287), weights already scaled."""
    import flashinfer
    from flashinfer.fused_moe import Fp8QuantizationType, WeightLayout
    qx, qs = moe_kernel_quantize_input(A=x, A_scale=None, quant_dtype=torch.float8_e4m3fn, per_act_token_quant=False,
                                       block_shape=[128, 128])
    ids = st["renum"][topk_ids.long()]
    out = torch.empty_like(x)
    flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe(
        topk_ids=(ids, topk_weights), routing_bias=None, hidden_states=qx,
        hidden_states_scale=prepare_deepseek_fp8_x_sf(qx, qs),
        gemm1_weights=st["w1"], gemm1_weights_scale=st["s1"], gemm1_clamp_limit=st["clamp"],
        gemm2_weights=st["w2"], gemm2_weights_scale=st["s2"], num_experts=E, top_k=topk_ids.size(1),
        n_group=None, topk_group=None, intermediate_size=I_SIZE, local_expert_offset=0,
        local_num_experts=st["n_hot"], routed_scaling_factor=None, routing_method_type=1,
        use_shuffled_weight=True, weight_layout=WeightLayout.BlockMajorK,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8, do_finalize=True, output=out,
        tune_max_num_tokens=8192)
    return out
