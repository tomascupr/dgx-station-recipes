"""Minimal CUDA runtime calls for the validate script.

The vLLM hook uses torch tensors instead. This module is only the host
driver for a process that has the toolkit and a device.
"""
from __future__ import annotations

import ctypes

_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            _LIB = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _LIB is None:
        raise RuntimeError("libcudart is not installed; this step runs on the DGX")
    _LIB.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _LIB.cudaMalloc.restype = ctypes.c_int
    _LIB.cudaFree.argtypes = [ctypes.c_void_p]
    _LIB.cudaFree.restype = ctypes.c_int
    _LIB.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _LIB.cudaMemcpy.restype = ctypes.c_int
    _LIB.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    _LIB.cudaMemset.restype = ctypes.c_int
    _LIB.cudaSetDevice.argtypes = [ctypes.c_int]
    _LIB.cudaSetDevice.restype = ctypes.c_int
    _LIB.cudaDeviceSynchronize.restype = ctypes.c_int
    _LIB.cudaGetErrorString.argtypes = [ctypes.c_int]
    _LIB.cudaGetErrorString.restype = ctypes.c_char_p
    _LIB.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _LIB.cudaEventCreate.restype = ctypes.c_int
    _LIB.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _LIB.cudaEventRecord.restype = ctypes.c_int
    _LIB.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    _LIB.cudaEventSynchronize.restype = ctypes.c_int
    _LIB.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    _LIB.cudaEventElapsedTime.restype = ctypes.c_int
    _LIB.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    _LIB.cudaEventDestroy.restype = ctypes.c_int
    return _LIB


def check(rc: int, what: str) -> None:
    if rc == 0:
        return
    msg = lib().cudaGetErrorString(rc)
    text = msg.decode() if msg else str(rc)
    raise RuntimeError(f"{what}: {text}")


HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


def set_device(index: int) -> None:
    check(lib().cudaSetDevice(index), "cudaSetDevice")


def alloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    check(lib().cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr.value


def free(ptr: int) -> None:
    if ptr:
        check(lib().cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def memcpy(dst: int, src: int, nbytes: int, kind: int) -> None:
    check(lib().cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, kind), "cudaMemcpy")


def h2d(host: bytes | memoryview, ptr: int) -> None:
    buf = (ctypes.c_char * len(host)).from_buffer_copy(bytes(host))
    check(lib().cudaMemcpy(ctypes.c_void_p(ptr), ctypes.cast(buf, ctypes.c_void_p), len(host), HOST_TO_DEVICE),
          "cudaMemcpy h2d")


def d2h(ptr: int, nbytes: int) -> bytes:
    buf = ctypes.create_string_buffer(nbytes)
    check(lib().cudaMemcpy(ctypes.cast(buf, ctypes.c_void_p), ctypes.c_void_p(ptr), nbytes, DEVICE_TO_HOST),
          "cudaMemcpy d2h")
    return buf.raw


def memset0(ptr: int, nbytes: int) -> None:
    check(lib().cudaMemset(ctypes.c_void_p(ptr), 0, nbytes), "cudaMemset")


def sync() -> None:
    check(lib().cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class Event:
    def __init__(self):
        self.ptr = ctypes.c_void_p()
        check(lib().cudaEventCreate(ctypes.byref(self.ptr)), "cudaEventCreate")

    def record(self) -> None:
        check(lib().cudaEventRecord(self.ptr, None), "cudaEventRecord")

    def sync(self) -> None:
        check(lib().cudaEventSynchronize(self.ptr), "cudaEventSynchronize")

    def close(self) -> None:
        if self.ptr:
            lib().cudaEventDestroy(self.ptr)
            self.ptr = ctypes.c_void_p()


def elapsed_ms(start: Event, end: Event) -> float:
    ms = ctypes.c_float()
    check(lib().cudaEventElapsedTime(ctypes.byref(ms), start.ptr, end.ptr), "cudaEventElapsedTime")
    return float(ms.value)
