"""Grace spill staging.

The sm_103a GEMM loads weights with TMA (`cp.async.bulk.tensor`, descriptor
from `cuTensorMapEncodeTiled` in CUDA 13.2 cuda.h, built in
cute/atom/copy_traits_sm90_tma.hpp). That instruction copies tensor data in
the global-memory state space. The bulk path NVIDIA documents for Grace CPU
memory is the DMA copy engine, not TMA: "NVLink-C2C with Address Translation
Services (ATS) leverages the NVIDIA Hopper Direct Memory Access (DMA) copy
engines for accelerating bulk transfers of pageable memory across host and
device."
https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/

So spill bytes stay in Grace memory and a copy-engine prefetch fills an LRU
in GB300 HBM. The grouped GEMM then sees device pointers. The prefetch for a
layer is issued at attention entry from the previous token's spill hits, so
the copy overlaps that layer's attention. The MoE launch waits and copies
any miss. INKLING2_SPILL_CPU=1 keeps the fp64 host fallback.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

PAGEABLE = 88
HOST_TABLES = 100


class Lru:
    def __init__(self, slots: int):
        if slots < 1:
            raise ValueError("spill pool needs at least one slot")
        self.slots = slots
        self.order: list = []
        self.where: dict = {}

    def assign(self, key):
        """Returns (slot, miss)."""
        if key in self.where:
            self.order.remove(key)
            self.order.append(key)
            return self.where[key], False
        if len(self.where) < self.slots:
            slot = len(self.where)
        else:
            old = self.order.pop(0)
            slot = self.where.pop(old)
        self.where[key] = slot
        self.order.append(key)
        return slot, True


def host_path(device: int = 0) -> str:
    """ATS pageable memory when the driver says so, else pinned host memory."""
    import cudart
    access = ctypes.c_int()
    tables = ctypes.c_int()
    attr = cudart.lib().cudaDeviceGetAttribute
    attr.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    attr.restype = ctypes.c_int
    rc1 = attr(ctypes.byref(access), PAGEABLE, device)
    rc2 = attr(ctypes.byref(tables), HOST_TABLES, device)
    if rc1 == 0 and rc2 == 0 and access.value and tables.value:
        return "ats"
    return "pinned"


class SpillPool:
    def __init__(self, bank_of, pool_bytes: int, device: int = 0):
        """``bank_of(layer)`` returns the host bank for that layer's spill ids."""
        import cudart
        self.cudart = cudart
        self.bank_of = bank_of
        self.device = device
        self.mode = host_path(device)
        self.predicted: dict[int, list[int]] = {}
        self._host: dict[tuple[int, int], dict] = {}
        # Slot size is one expert. The first non-empty bank fixes it.
        self.slot_bytes = 0
        self.slots: list[dict] = []
        self.lru: Lru | None = None
        self.pool_bytes = pool_bytes
        self._stream = ctypes.c_void_p()
        self._event = ctypes.c_void_p()
        cudart.set_device(device)
        cudart.lib().cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cudart.lib().cudaStreamCreate.restype = ctypes.c_int
        cudart.lib().cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cudart.lib().cudaEventCreate.restype = ctypes.c_int
        cudart.lib().cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cudart.lib().cudaEventRecord.restype = ctypes.c_int
        cudart.lib().cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        cudart.lib().cudaStreamWaitEvent.restype = ctypes.c_int
        cudart.lib().cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        cudart.lib().cudaMemcpyAsync.restype = ctypes.c_int
        cudart.check(cudart.lib().cudaStreamCreate(ctypes.byref(self._stream)), "cudaStreamCreate")
        cudart.check(cudart.lib().cudaEventCreate(ctypes.byref(self._event)), "cudaEventCreate")
        self._ready = False

    def _ensure_slots(self, bank: dict) -> None:
        n = bank["n"]
        spans = {k: bank[k].nbytes // n for k in ("w13", "s13", "w2", "s2")}
        need = sum(spans.values())
        if self.lru is not None:
            if need != self.slot_bytes:
                raise RuntimeError(
                    f"spill expert is {need} bytes and the pool slots are {self.slot_bytes}")
            return
        self._span = spans
        self.slot_bytes = need
        count = max(1, self.pool_bytes // self.slot_bytes)
        self.lru = Lru(count)
        for _ in range(count):
            base = self.cudart.alloc(self.slot_bytes)
            off = 0
            view = {}
            for key in ("w13", "s13", "w2", "s2"):
                view[key] = base + off
                off += spans[key]
            view["base"] = base
            self.slots.append(view)

    def _host_expert(self, layer: int, expert: int, bank: dict) -> dict:
        key = (layer, expert)
        got = self._host.get(key)
        if got is not None:
            return got
        local = bank["members"].index(expert)
        spans = {k: bank[k].nbytes // bank["n"] for k in ("w13", "s13", "w2", "s2")}
        pieces = {}
        for name, span in spans.items():
            raw = np.ascontiguousarray(bank[name][local * span:(local + 1) * span])
            if self.mode == "pinned":
                ptr = ctypes.c_void_p()
                self.cudart.lib().cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
                self.cudart.lib().cudaMallocHost.restype = ctypes.c_int
                self.cudart.check(self.cudart.lib().cudaMallocHost(ctypes.byref(ptr), span), "cudaMallocHost")
                ctypes.memmove(ptr.value, raw.ctypes.data, span)
                pieces[name] = (ptr.value, span, True)
            else:
                pieces[name] = (raw.ctypes.data, span, False)
                pieces[name + "_keep"] = raw
        self._host[key] = pieces
        return pieces

    def prefetch(self, layer: int, experts: list[int], bank: dict) -> int:
        """Queue copies for cache misses. Returns bytes queued. Does not wait."""
        if not experts:
            return 0
        self._ensure_slots(bank)
        queued = 0
        for expert in experts:
            slot, miss = self.lru.assign((layer, expert))
            if not miss:
                continue
            host = self._host_expert(layer, expert, bank)
            view = self.slots[slot]
            for name in ("w13", "s13", "w2", "s2"):
                src, span, _pinned = host[name]
                rc = self.cudart.lib().cudaMemcpyAsync(
                    ctypes.c_void_p(view[name]), ctypes.c_void_p(src), span, 1, self._stream)
                self.cudart.check(rc, "cudaMemcpyAsync")
                queued += span
        self.cudart.check(
            self.cudart.lib().cudaEventRecord(self._event, self._stream), "cudaEventRecord")
        return queued

    def wait(self) -> None:
        """The default stream, which launches the GEMM, waits for the copies."""
        self.cudart.check(
            self.cudart.lib().cudaStreamWaitEvent(None, self._event, 0), "cudaStreamWaitEvent")

    def sync_copy(self) -> None:
        """Block the host until the queued copies finish. Used by validate."""
        fn = self.cudart.lib().cudaEventSynchronize
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        self.cudart.check(fn(self._event), "cudaEventSynchronize")

    def pointers(self, layer: int, expert: int) -> dict:
        slot = self.lru.where[(layer, expert)]
        return self.slots[slot]

    def note(self, layer: int, experts: list[int]) -> None:
        self.predicted[layer] = list(experts)
