"""Host segment shared by the GB300 and the RTX sidecar.

One request in flight. The client publishes x, ids and route weights, then
a release sequence number. The sidecar writes y and the same sequence.
Tests use a file or a bytearray; the DGX uses /dev/shm/inkling2-peer.
"""
from __future__ import annotations

import mmap
import os
import struct

import numpy as np

MAGIC = struct.unpack("<Q", b"INK2PEER")[0]
VERSION = 1
HIDDEN = 4608
TOPK = 8
MAX_TOKENS = 64
QUANT_NVFP4 = 0
QUANT_MXFP8 = 1


def _align(n: int, page: int = 4096) -> int:
    return (n + page - 1) // page * page


class Layout:
    def __init__(self, hidden: int = HIDDEN, max_tokens: int = MAX_TOKENS, topk: int = TOPK):
        self.hidden = hidden
        self.max_tokens = max_tokens
        self.topk = topk
        self.off_magic = 0
        self.off_version = 8
        self.off_req = 16
        self.off_resp = 24
        self.off_layer = 32
        self.off_quant = 36
        self.off_t = 40
        self.off_err = 44
        self.off_x = 4096
        self.off_ids = self.off_x + max_tokens * hidden * 2
        self.off_w = _align(self.off_ids + max_tokens * topk * 4)
        self.off_y = _align(self.off_w + max_tokens * topk * 4)
        self.nbytes = _align(self.off_y + max_tokens * hidden * 2)


def production_layout() -> Layout:
    return Layout()


class Segment:
    def __init__(self, path: str | None, layout: Layout, create: bool):
        self.layout = layout
        self.path = path
        self._buf = None
        self._mm = None
        if path is None:
            self._buf = bytearray(layout.nbytes)
            self._mv = memoryview(self._buf)
        else:
            flags = os.O_RDWR | (os.O_CREAT if create else 0)
            fd = os.open(path, flags, 0o600)
            try:
                if create:
                    os.ftruncate(fd, layout.nbytes)
                self._mm = mmap.mmap(fd, layout.nbytes)
            finally:
                os.close(fd)
            self._mv = memoryview(self._mm)
        if create:
            self._u64(self.layout.off_magic, MAGIC)
            self._u64(self.layout.off_version, VERSION)
            self._u64(self.layout.off_req, 0)
            self._u64(self.layout.off_resp, 0)

    def close(self) -> None:
        self._mv = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None

    def _u64(self, off: int, value: int | None = None) -> int:
        if value is None:
            return struct.unpack_from("<Q", self._mv, off)[0]
        struct.pack_into("<Q", self._mv, off, value)
        return value

    def _u32(self, off: int, value: int | None = None) -> int:
        if value is None:
            return struct.unpack_from("<I", self._mv, off)[0]
        struct.pack_into("<I", self._mv, off, value)
        return value

    def check_magic(self) -> None:
        if self._u64(self.layout.off_magic) != MAGIC or self._u64(self.layout.off_version) != VERSION:
            raise RuntimeError("inkling peer segment magic or version does not match this build")

    def publish(self, seq: int, layer: int, quant: int, x: np.ndarray,
                ids: np.ndarray, weights: np.ndarray) -> None:
        self.check_magic()
        t = int(x.shape[0])
        if t > self.layout.max_tokens:
            raise ValueError(f"{t} tokens exceeds the segment ({self.layout.max_tokens})")
        if x.shape != (t, self.layout.hidden) or ids.shape != (t, self.layout.topk):
            raise ValueError("publish shapes do not match the segment")
        raw = _f32_to_bf16(np.asarray(x, dtype=np.float32))
        self._mv[self.layout.off_x:self.layout.off_x + raw.nbytes] = raw.tobytes()
        ids_b = np.asarray(ids, dtype=np.int32).tobytes()
        w_b = np.asarray(weights, dtype=np.float32).tobytes()
        self._mv[self.layout.off_ids:self.layout.off_ids + len(ids_b)] = ids_b
        self._mv[self.layout.off_w:self.layout.off_w + len(w_b)] = w_b
        self._u32(self.layout.off_layer, layer)
        self._u32(self.layout.off_quant, quant)
        self._u32(self.layout.off_t, t)
        self._u32(self.layout.off_err, 0)
        self._u64(self.layout.off_req, seq)

    def read_request(self) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
        self.check_magic()
        seq = self._u64(self.layout.off_req)
        layer = self._u32(self.layout.off_layer)
        quant = self._u32(self.layout.off_quant)
        t = self._u32(self.layout.off_t)
        x = _bf16_to_f32(self._take(self.layout.off_x, t * self.layout.hidden * 2)).reshape(t, self.layout.hidden)
        ids = np.frombuffer(self._take(self.layout.off_ids, t * self.layout.topk * 4), dtype=np.int32).reshape(t, self.layout.topk)
        w = np.frombuffer(self._take(self.layout.off_w, t * self.layout.topk * 4), dtype=np.float32).reshape(t, self.layout.topk)
        return seq, layer, quant, x, ids.copy(), w.copy()

    def publish_y(self, seq: int, y: np.ndarray) -> None:
        raw = _f32_to_bf16(np.asarray(y, dtype=np.float32))
        self._mv[self.layout.off_y:self.layout.off_y + raw.nbytes] = raw.tobytes()
        self._u64(self.layout.off_resp, seq)

    def read_y(self, t: int) -> np.ndarray:
        raw = self._take(self.layout.off_y, t * self.layout.hidden * 2)
        return _bf16_to_f32(raw).reshape(t, self.layout.hidden)

    def req(self) -> int:
        return self._u64(self.layout.off_req)

    def resp(self) -> int:
        return self._u64(self.layout.off_resp)

    def _take(self, off: int, n: int) -> bytes:
        return bytes(self._mv[off:off + n])


def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16, returned as uint16."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = u + (0x7FFF + lsb)
    return (u >> 16).astype(np.uint16)


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).copy()
