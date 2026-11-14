"""ctypes bindings for the compiled peer objects.

The GB300 object is called in-process. The RTX object is called by the
sidecar. Both export the same suffix: expert_nvfp4, expert_mxfp8, and the
fp64 reference kernels nvfp4 and mxfp8. The GB300 object also exports
``expert_*_each``, one device pointer per expert, so a staged Grace expert
and a hot expert share one launch. ``INKLING2_SPILL_CPU=1`` keeps the fp64
host fallback.
"""
from __future__ import annotations

import ctypes
import os

C_INT = ctypes.c_int
C_FLOAT = ctypes.c_float
C_DOUBLE = ctypes.c_double
C_VOID = ctypes.c_void_p


def _bind(lib: ctypes.CDLL, name: str, argtypes: list) -> ctypes._FuncPtr:
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = C_INT
    return fn


class PeerSo:
    def __init__(self, path: str, prefix: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.prefix = prefix
        self.lib = ctypes.CDLL(path)
        expert_nv = [
            C_INT, C_INT, C_INT, C_INT, C_INT, C_VOID, C_VOID, C_VOID,
            C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_FLOAT, C_VOID, C_VOID,
        ]
        expert_mx = [
            C_INT, C_INT, C_INT, C_INT, C_INT, C_VOID, C_VOID, C_VOID,
            C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID,
        ]
        ref_nv = [
            C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID,
            C_DOUBLE, C_INT, C_INT, C_INT, C_INT, C_VOID, C_VOID,
        ]
        ref_mx = [
            C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID,
            C_INT, C_INT, C_INT, C_INT, C_VOID, C_VOID,
        ]
        self.expert_nvfp4 = _bind(self.lib, f"{prefix}_expert_nvfp4", expert_nv)
        self.expert_mxfp8 = _bind(self.lib, f"{prefix}_expert_mxfp8", expert_mx)
        self.expert_nvfp4_each = None
        self.expert_mxfp8_each = None
        if prefix == "inkling_gb300":
            each_nv = [
                C_INT, C_INT, C_INT, C_INT, C_INT, C_VOID, C_VOID, C_VOID,
                C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_FLOAT, C_VOID, C_VOID,
            ]
            each_mx = [
                C_INT, C_INT, C_INT, C_INT, C_INT, C_VOID, C_VOID, C_VOID,
                C_VOID, C_VOID, C_VOID, C_VOID, C_VOID, C_VOID,
            ]
            self.expert_nvfp4_each = _bind(self.lib, f"{prefix}_expert_nvfp4_each", each_nv)
            self.expert_mxfp8_each = _bind(self.lib, f"{prefix}_expert_mxfp8_each", each_mx)
        self.ref_nvfp4 = _bind(self.lib, f"{prefix}_nvfp4", ref_nv)
        self.ref_mxfp8 = _bind(self.lib, f"{prefix}_mxfp8", ref_mx)

    def expert(self, kind: str, tokens: int, topk: int, experts: int, hidden: int, inter: int,
               x, ids, route, w13, s13, amax13, w2, s2, amax2, denom: float, y, stream=None) -> None:
        stream_p = C_VOID(stream)
        if kind == "nvfp4":
            rc = self.expert_nvfp4(
                tokens, topk, experts, hidden, inter, C_VOID(x), C_VOID(ids), C_VOID(route),
                C_VOID(w13), C_VOID(s13), C_VOID(amax13), C_VOID(w2), C_VOID(s2), C_VOID(amax2),
                C_FLOAT(denom), C_VOID(y), stream_p)
        elif kind == "mxfp8":
            rc = self.expert_mxfp8(
                tokens, topk, experts, hidden, inter, C_VOID(x), C_VOID(ids), C_VOID(route),
                C_VOID(w13), C_VOID(s13), C_VOID(w2), C_VOID(s2), C_VOID(y), stream_p)
        else:
            raise ValueError(kind)
        if rc:
            raise RuntimeError(f"{self.prefix} expert {kind} returned {rc}")

    def expert_each(self, kind: str, tokens: int, topk: int, experts: int, hidden: int, inter: int,
                    x, ids, route, w13, s13, amax13, w2, s2, amax2, denom: float, y, stream=None) -> None:
        """``w13`` and the other weight arguments are pointers to pointer arrays."""
        stream_p = C_VOID(stream)
        if kind == "nvfp4":
            rc = self.expert_nvfp4_each(
                tokens, topk, experts, hidden, inter, C_VOID(x), C_VOID(ids), C_VOID(route),
                C_VOID(w13), C_VOID(s13), C_VOID(amax13), C_VOID(w2), C_VOID(s2), C_VOID(amax2),
                C_FLOAT(denom), C_VOID(y), stream_p)
        elif kind == "mxfp8":
            rc = self.expert_mxfp8_each(
                tokens, topk, experts, hidden, inter, C_VOID(x), C_VOID(ids), C_VOID(route),
                C_VOID(w13), C_VOID(s13), C_VOID(w2), C_VOID(s2), C_VOID(y), stream_p)
        else:
            raise ValueError(kind)
        if rc:
            raise RuntimeError(f"{self.prefix} expert_each {kind} returned {rc}")

    def reference(self, kind: str, tokens: int, topk: int, hidden: int, inter: int,
                  x, ids, route, w13, s13, amax13, w2, s2, amax2, denom: float, scratch, y) -> None:
        if kind == "nvfp4":
            rc = self.ref_nvfp4(
                C_VOID(x), C_VOID(ids), C_VOID(route), C_VOID(w13), C_VOID(s13), C_VOID(amax13),
                C_VOID(w2), C_VOID(s2), C_VOID(amax2), C_DOUBLE(denom), tokens, hidden, inter, topk,
                C_VOID(scratch), C_VOID(y))
        elif kind == "mxfp8":
            rc = self.ref_mxfp8(
                C_VOID(x), C_VOID(ids), C_VOID(route), C_VOID(w13), C_VOID(s13),
                C_VOID(w2), C_VOID(s2), tokens, hidden, inter, topk, C_VOID(scratch), C_VOID(y))
        else:
            raise ValueError(kind)
        if rc:
            raise RuntimeError(f"{self.prefix} fp64 {kind} returned {rc}")


def open_so(path: str) -> PeerSo:
    base = os.path.basename(path)
    if "gb300" in base or "sm103" in base:
        prefix = "inkling_gb300"
    elif "rtx" in base or "sm120" in base:
        prefix = "inkling_rtx"
    else:
        raise ValueError(f"cannot tell which symbols {path} exports")
    return PeerSo(path, prefix)
