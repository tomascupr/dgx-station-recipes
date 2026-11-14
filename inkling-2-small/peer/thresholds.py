"""Pass bar for the serving GEMM against the fp64 reference.

The serving kernel writes bf16. One bf16 ulp is 2^-7 = 1/128 of the value.
Gate/up, SiTU and the down projection are each stored as bf16 before the
next step, so three storage roundings plus an fp32 accumulation of K=4608
against an fp64 sum are in the result. The bar is 2.5% of max(|reference|, 1):
about three bf16 ulps on a result of magnitude 1 or larger, and 0.025 absolute
when the reference is smaller than 1. A wrong scale swizzle misses this by
a wide margin; it is not a claim that the kernel matches bf16 rounding.
"""
from __future__ import annotations

import numpy as np

REL = 0.025


def limits(reference: np.ndarray) -> tuple[float, float, float]:
    """Returns peak |ref|, the absolute limit, and max |got-ref| is compared to it."""
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    limit = REL * max(peak, 1.0)
    return peak, limit, REL


def judge(reference: np.ndarray, got: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float64)
    out = np.asarray(got, dtype=np.float64)
    if ref.shape != out.shape:
        raise ValueError(f"shape {out.shape} != reference {ref.shape}")
    err = np.abs(out - ref)
    max_abs = float(err.max()) if err.size else 0.0
    peak, limit, rel = limits(ref)
    return {
        "max_abs": max_abs,
        "peak": peak,
        "max_rel_vs_peak": max_abs / max(peak, 1e-6),
        "limit_abs": limit,
        "rel_bar": rel,
        "pass": bool(max_abs <= limit),
    }
