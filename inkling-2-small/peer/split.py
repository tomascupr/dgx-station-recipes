"""Split one MoE layer across hot, cold and spill tiers.

Each tier holds a subset of the experts and sees the whole top-k. An id that
is not in the tier contributes nothing. The sum of the three tiers is the
full layer when the tiers partition the expert ids.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "reference"))
from quant import expert_forward, quantize_activation  # noqa: E402

TIERS = ("hot", "cold", "spill")


def local_table(members: list[int], n_experts: int) -> np.ndarray:
    table = np.full(n_experts, -1, dtype=np.int32)
    for local, expert in enumerate(members):
        table[expert] = local
    return table


def remap_ids(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    out = np.full(ids.shape, -1, dtype=np.int64)
    valid = ids >= 0
    out[valid] = table[ids[valid]]
    return out


def apply_tier(x: np.ndarray, ids: np.ndarray, weights: np.ndarray, members: list[int],
               w13: np.ndarray, w2: np.ndarray, kind: str, n_experts: int) -> np.ndarray:
    if not members:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    local = remap_ids(ids, local_table(members, n_experts))
    xq = quantize_activation(np.asarray(x, dtype=np.float64), kind)
    return expert_forward(xq, local, weights, w13, w2)


def forward_split(x, ids, weights, placement_layer: dict, banks: dict, kind: str,
                  n_experts: int) -> np.ndarray:
    y = np.zeros_like(np.asarray(x, dtype=np.float64))
    for tier in TIERS:
        bank = banks.get(tier)
        if bank is None:
            continue
        y += apply_tier(x, ids, weights, placement_layer[tier], bank["w13"], bank["w2"], kind, n_experts)
    return y
