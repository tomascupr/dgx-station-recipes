"""CPU reference for Inkling 2 Small expert math.

Numerics follow the day-0 vLLM module ``vllm.models.inkling2`` (PR 67325, head
d664cd1e): NVFP4 is E2M1, two values per byte, low nibble first, one E4M3
scale per 16 K-elements, weight decode scale ``amax / denominator`` (the
checkpoint stores 2688), activation decode scale ``amax / 1536``. MXFP8 is
E4M3 with a UE8M0 scale per 32 K-elements. w13 rows are interleaved
gate, up, gate, up. The MoE activation is
``4*tanh(gate/4) * sigmoid(gate) * 25*tanh(up/25) * route_weight``.
The router is sigmoid(logits) + bias, top-k, then softmax(logsigmoid) * 8.
"""
from __future__ import annotations

import numpy as np

HIDDEN = 4608
INTER = 2304
N_EXPERTS = 384
TOPK = 8
N_LAYERS = 48
NVFP4_LAYERS = 40
GATE_CAP = 4.0
UP_CAP = 25.0
ROUTE_SCALE = 8.0
WEIGHT_DENOMINATOR = 2688.0
ACT_DENOMINATOR = 1536.0
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
E4M3_MAX = 448.0

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
_E2M1_CODE = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7}


def e4m3fn_to_float(codes: np.ndarray) -> np.ndarray:
    bits = np.asarray(codes, dtype=np.uint8)
    exp = (bits >> 3) & 0x0F
    man = (bits & 0x07).astype(np.float64)
    normal = (1.0 + man * 0.125) * np.exp2(exp.astype(np.int16) - 7)
    val = np.where(exp == 0, man * 0.001953125, normal)
    val = np.where((bits & 0x7F) == 0x7F, np.nan, val)
    return np.where((bits & 0x80) != 0, -val, val)


def _e4m3_table() -> tuple[np.ndarray, np.ndarray]:
    codes = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=np.uint8)
    return codes, e4m3fn_to_float(codes)


_E4M3_CODES, _E4M3_VALS = _e4m3_table()


def quantize_e4m3(values: np.ndarray) -> np.ndarray:
    """Nearest finite E4M3 code. Ties take the lower code."""
    x = np.asarray(values, dtype=np.float64)
    flat = np.nan_to_num(x, nan=np.inf).reshape(-1)
    # Broadcast against the 254 finite codes. Fine for the reference shapes.
    idx = np.abs(_E4M3_VALS - flat[:, None]).argmin(axis=1)
    return _E4M3_CODES[idx].reshape(x.shape)


def e2m1_from_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    mag = E2M1_MAG[codes & 0x07]
    return np.where((codes & 0x08) != 0, -mag, mag)


def round_e2m1(values: np.ndarray) -> np.ndarray:
    """vLLM ``cast_to_fp4`` thresholds."""
    x = np.asarray(values, dtype=np.float64)
    ax = np.abs(x)
    out = np.zeros_like(ax)
    out = np.where((ax > 0.25) & (ax < 0.75), 0.5, out)
    out = np.where((ax >= 0.75) & (ax <= 1.25), 1.0, out)
    out = np.where((ax > 1.25) & (ax < 1.75), 1.5, out)
    out = np.where((ax >= 1.75) & (ax <= 2.5), 2.0, out)
    out = np.where((ax > 2.5) & (ax < 3.5), 3.0, out)
    out = np.where((ax >= 3.5) & (ax <= 5.0), 4.0, out)
    out = np.where(ax > 5.0, 6.0, out)
    return out * np.where(x < 0.0, -1.0, 1.0)


def e2m1_to_codes(values: np.ndarray) -> np.ndarray:
    rounded = round_e2m1(values)
    codes = np.zeros(rounded.shape, dtype=np.uint8)
    for mag, code in _E2M1_CODE.items():
        codes = np.where(np.abs(rounded) == mag, code, codes)
    codes = np.where(rounded < 0.0, codes | 0x08, codes)
    return codes


def unpack_nvfp4(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    low = e2m1_from_codes(packed & 0x0F)
    high = e2m1_from_codes(packed >> 4)
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float64)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def pack_nvfp4(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2:
        raise ValueError("NVFP4 codes must be even along K")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


def dequant_nvfp4(packed: np.ndarray, scales: np.ndarray, amax, denominator: float) -> np.ndarray:
    """``e2m1 * e4m3(scale) * (amax / denominator)``, scale shared by 16 K elements."""
    vals = unpack_nvfp4(packed)
    scale = e4m3fn_to_float(scales)
    if scale.shape[-1] * NVFP4_BLOCK != vals.shape[-1]:
        raise ValueError(
            f"NVFP4 scale count {scale.shape[-1]} does not cover K={vals.shape[-1]}")
    scale = np.repeat(scale, NVFP4_BLOCK, axis=-1)
    factor = np.asarray(amax, dtype=np.float64) / float(denominator)
    if factor.ndim == 1 and vals.ndim >= 2 and factor.shape[0] == vals.shape[0]:
        factor = factor.reshape((factor.shape[0],) + (1,) * (vals.ndim - 1))
    return vals * scale * factor


def dequant_mxfp8(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``e4m3 * 2^(ue8m0 - 127)``, scale shared by 32 K elements."""
    vals = e4m3fn_to_float(values)
    scale = np.exp2(np.asarray(scales, dtype=np.int16) - 127)
    return vals * np.repeat(scale, MXFP8_BLOCK, axis=-1)


def _row_amax(x: np.ndarray) -> np.ndarray:
    return np.max(np.abs(x), axis=-1)


def fouroversix_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NVFP4 activation quant, ``fouroversix_mae``.

    Each block of 16 tries decode scales that map the block max through FP4
    6 and through FP4 4 (the 4-candidate is 1.5x the 6-candidate). The lower
    sum of absolute error wins. A tie keeps map-to-6. Returns packed values,
    E4M3 scales and the per-row amax.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] % NVFP4_BLOCK:
        raise ValueError("K must be a multiple of 16")
    amax = _row_amax(x)
    blocks = x.reshape(*x.shape[:-1], -1, NVFP4_BLOCK)
    block_max = np.max(np.abs(blocks), axis=-1)
    safe = np.where(amax == 0.0, 1.0, amax)
    cands = []
    for divisor in (6.0, 4.0):
        raw = block_max * (ACT_DENOMINATOR / safe[..., None]) / divisor
        raw = np.minimum(raw, E4M3_MAX)
        raw = np.where(amax[..., None] == 0.0, 0.0, raw)
        cands.append(e4m3fn_to_float(quantize_e4m3(raw)))
    outer = (amax / ACT_DENOMINATOR)[..., None, None]
    errors = []
    rounded = []
    for scale in cands:
        level = scale[..., None]
        q = np.where(level == 0.0, 0.0, blocks / (level * np.where(outer == 0.0, 1.0, outer)))
        rnd = round_e2m1(q)
        recon = rnd * level * outer
        errors.append(np.sum(np.abs(recon - blocks), axis=-1))
        rounded.append(rnd)
    # Tie -> map-to-6, which is candidate 0.
    pick4 = errors[1] < errors[0]
    chosen_scale = np.where(pick4, cands[1], cands[0])
    chosen_vals = np.where(pick4[..., None], rounded[1], rounded[0])
    flat = chosen_vals.reshape(*x.shape[:-1], -1)
    return pack_nvfp4(e2m1_to_codes(flat)), quantize_e4m3(chosen_scale), amax


def mxfp8_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round each block of 32 up to an E4M3 scale of ``448 * 2^n``."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] % MXFP8_BLOCK:
        raise ValueError("K must be a multiple of 32")
    blocks = x.reshape(*x.shape[:-1], -1, MXFP8_BLOCK)
    amax = np.max(np.abs(blocks), axis=-1)
    with np.errstate(divide="ignore"):
        exp = np.ceil(np.log2(np.maximum(amax, 1e-30) / E4M3_MAX))
    byte = np.where(amax == 0.0, 0, exp + 127)
    byte = np.clip(byte, 0, 254).astype(np.uint8)
    scale = np.exp2(byte.astype(np.int16) - 127)
    q = np.where(scale[..., None] == 0.0, 0.0, blocks / scale[..., None])
    codes = quantize_e4m3(q).reshape(x.shape)
    return codes, byte


def swiglu(gate_up: np.ndarray, route_weight: np.ndarray) -> np.ndarray:
    """Interleaved gate/up rows in, activated up-projection input out."""
    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    gate_c = GATE_CAP * np.tanh(gate / GATE_CAP)
    up_c = UP_CAP * np.tanh(up / UP_CAP)
    sig = 1.0 / (1.0 + np.exp(-gate))
    return gate_c * sig * up_c * np.asarray(route_weight, dtype=np.float64)[..., None]


def log_sigmoid(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))


def select_experts(logits: np.ndarray, bias: np.ndarray, topk: int = TOPK,
                   route_scale: float = ROUTE_SCALE) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(weights, ids)``, each ``(..., topk)``. Ties keep the lower id."""
    logits = np.asarray(logits, dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits)) + np.asarray(bias, dtype=np.float64)
    order = np.argsort(-scores, axis=-1, kind="stable")
    ids = order[..., :topk]
    chosen = np.take_along_axis(logits, ids, axis=-1)
    shifted = log_sigmoid(chosen)
    shifted = shifted - np.max(shifted, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights, axis=-1, keepdims=True) * route_scale
    return weights, ids


def project(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``x @ weight.T`` for one matrix ``weight`` of shape (N, K)."""
    return weight @ x


def expert_forward(x: np.ndarray, ids: np.ndarray, weights: np.ndarray,
                   w13: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Sum of routed experts. ``w13[e]`` is (2I, H), ``w2[e]`` is (H, I), both fp64.

    ``ids`` may be -1. Experts that are not in this tier are simply absent
    from ``w13``: pass a local id map and the local tensors, or pass global
    tensors and global ids. This function indexes ``w13[e]`` directly, so the
    caller remaps ids first.
    """
    x = np.asarray(x, dtype=np.float64)
    ids = np.asarray(ids)
    weights = np.asarray(weights, dtype=np.float64)
    y = np.zeros_like(x)
    tokens, _slots = ids.shape
    for t in range(tokens):
        for s in range(ids.shape[1]):
            e = int(ids[t, s])
            if e < 0:
                continue
            gate_up = w13[e] @ x[t]
            act = swiglu(gate_up, weights[t, s])
            y[t] += w2[e] @ act
    return y


def quantize_activation(x: np.ndarray, kind: str) -> np.ndarray:
    """Dequantized activation, the tensor the expert GEMM multiplies."""
    if kind == "nvfp4":
        packed, scales, amax = fouroversix_quantize(x)
        return dequant_nvfp4(packed, scales, amax[..., None], ACT_DENOMINATOR)
    if kind == "mxfp8":
        values, scales = mxfp8_quantize(x)
        return dequant_mxfp8(values, scales)
    raise ValueError(kind)


def dequant_expert(kind: str, packed_w13, scale_w13, packed_w2, scale_w2,
                   amax_w13=None, amax_w2=None, denominator: float = WEIGHT_DENOMINATOR):
    if kind == "nvfp4":
        w13 = dequant_nvfp4(packed_w13, scale_w13, amax_w13, denominator)
        w2 = dequant_nvfp4(packed_w2, scale_w2, amax_w2, denominator)
        return w13, w2
    if kind == "mxfp8":
        return dequant_mxfp8(packed_w13, scale_w13), dequant_mxfp8(packed_w2, scale_w2)
    raise ValueError(kind)
