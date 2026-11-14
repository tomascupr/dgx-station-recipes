"""Grace spill. The plan keeps these experts on the CPU, not on a third GPU
and not in a host-mapped GEMM. Packed bytes stay on the host; a hit expert
is dequantized into the fp64 reference and cached.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "reference"))
from quant import dequant_expert, expert_forward, quantize_activation  # noqa: E402

from bank import HIDDEN, INTER, local_ids  # noqa: E402


def _view(flat: np.ndarray, n: int, rows: int, cols: int) -> np.ndarray:
    return np.ascontiguousarray(flat).reshape(n, rows, cols)


def views(bank: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = bank["n"]
    if bank["kind"] == "nvfp4":
        w13 = _view(bank["w13"], n, INTER * 2, HIDDEN // 2)
        s13 = _view(bank["s13"], n, INTER * 2, HIDDEN // 16)
        w2 = _view(bank["w2"], n, HIDDEN, INTER // 2)
        s2 = _view(bank["s2"], n, HIDDEN, INTER // 16)
    else:
        w13 = _view(bank["w13"], n, INTER * 2, HIDDEN)
        s13 = _view(bank["s13"], n, INTER * 2, HIDDEN // 32)
        w2 = _view(bank["w2"], n, HIDDEN, INTER)
        s2 = _view(bank["s2"], n, HIDDEN, INTER // 32)
    return w13, s13, w2, s2


class CpuTier:
    def __init__(self, bank: dict):
        self.bank = bank
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if bank["n"]:
            self._w13, self._s13, self._w2, self._s2 = views(bank)

    def _weights(self, local: int) -> tuple[np.ndarray, np.ndarray]:
        hit = self.cache.get(local)
        if hit is not None:
            return hit
        kind = self.bank["kind"]
        if kind == "nvfp4":
            w13, w2 = dequant_expert(
                kind, self._w13[local], self._s13[local], self._w2[local], self._s2[local],
                self.bank["amax13"][local], self.bank["amax2"][local], self.bank["denom13"])
            # w2 uses its own denominator. dequant_expert applies one value to both.
            if self.bank["denom2"] != self.bank["denom13"]:
                w2 = dequant_expert(
                    kind, self._w13[local], self._s13[local], self._w2[local], self._s2[local],
                    self.bank["amax13"][local], self.bank["amax2"][local], self.bank["denom2"])[1]
        else:
            w13, w2 = dequant_expert(kind, self._w13[local], self._s13[local], self._w2[local], self._s2[local])
        self.cache[local] = (w13, w2)
        return w13, w2

    def forward(self, x: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if not self.bank["members"]:
            return np.zeros((x.shape[0], x.shape[-1]), dtype=np.float64)
        local = local_ids(ids, self.bank["members"])
        xq = quantize_activation(np.asarray(x, dtype=np.float64), self.bank["kind"])
        # Stack only the experts this call touches, and remap again into that stack.
        used = sorted({int(e) for e in local.reshape(-1) if int(e) >= 0})
        if not used:
            return np.zeros_like(xq)
        table = {old: i for i, old in enumerate(used)}
        remapped = np.full(local.shape, -1, dtype=np.int64)
        for index, value in np.ndenumerate(local):
            if int(value) >= 0:
                remapped[index] = table[int(value)]
        w13 = np.stack([self._weights(i)[0] for i in used])
        w2 = np.stack([self._weights(i)[1] for i in used])
        return expert_forward(xq, remapped, weights, w13, w2)
