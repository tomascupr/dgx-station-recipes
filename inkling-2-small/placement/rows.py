"""Read one expert row out of a packed safetensors tensor.

The checkpoint stores experts along axis 0, contiguous, so a cold expert is a
byte range. This does not dequantize and does not read any other expert.
"""
from __future__ import annotations

import json
import os
import struct

from headers import header_from_file, tensor_nbytes
from slice import file_offsets


def _pad8(n: int) -> int:
    return (n + 7) // 8 * 8


def write_safetensors(path: str, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    """``tensors`` is (name, dtype, shape, raw_bytes) in order."""
    header = {}
    blobs = []
    cursor = 0
    for name, dtype, shape, raw in tensors:
        cursor = _pad8(cursor)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + len(raw)]}
        blobs.append((cursor, raw))
        cursor += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        pos = 0
        for start, raw in blobs:
            if start > pos:
                f.write(b"\0" * (start - pos))
            f.write(raw)
            pos = start + len(raw)


def read_expert_row(path: str, name: str, expert_id: int, header: dict | None = None,
                    n_experts: int = 384) -> bytes:
    header = header if header is not None else header_from_file(path)
    meta = header[name]
    expect = tensor_nbytes(meta)
    start, end = meta["data_offsets"]
    if end - start != expect:
        raise ValueError(f"{name} data_offsets span {end - start} != {expect} bytes")
    lo, hi = file_offsets(meta, expert_id, n_experts)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + lo)
        raw = f.read(hi - lo)
    if len(raw) != hi - lo:
        raise ValueError(f"short read for {name} expert {expert_id}")
    return raw


def read_expert_rows(model_dir: str, names_and_experts: list[tuple[str, int]],
                     n_experts: int = 384) -> dict[tuple[str, int], bytes]:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    files: dict[str, dict] = {}
    out = {}
    for name, expert in names_and_experts:
        shard = index[name]
        path = os.path.join(model_dir, shard)
        if shard not in files:
            files[shard] = header_from_file(path)
        out[(name, expert)] = read_expert_row(path, name, expert, files[shard], n_experts)
    return out
