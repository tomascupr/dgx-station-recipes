"""Load a tier's packed experts from the checkpoint.

One expert is one contiguous slice of a 384-wide tensor. The CUTLASS launch
wants those slices concatenated in member order, which is also how
``w13[local_id]`` is indexed by the fp64 reference.
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "placement"))
from headers import header_from_file  # noqa: E402
from rows import read_expert_row  # noqa: E402

HIDDEN = 4608
INTER = 2304


def tensor_name(layer: int, stem: str) -> str:
    return f"model.llm.layers.{layer}.mlp.experts.{stem}"


def _index(model_dir: str) -> dict:
    import json
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _read_full(model_dir: str, index: dict, name: str) -> tuple[dict, bytes]:
    shard = index[name]
    path = os.path.join(model_dir, shard)
    header = header_from_file(path)
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + start)
        raw = f.read(end - start)
    if len(raw) != end - start:
        raise ValueError(f"short read {name}")
    return meta, raw


def load_members(model_dir: str, layer: int, members: list[int], kind: str) -> dict:
    """Packed arrays in member order. ``kind`` is nvfp4 or mxfp8."""
    if not members:
        return {"kind": kind, "members": [], "n": 0}
    index = _index(model_dir)
    stems = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]
    if kind == "nvfp4":
        stems += ["w13_weight_amax", "w2_weight_amax"]
    columns: dict[str, list[bytes]] = {stem: [] for stem in stems}
    for expert in members:
        for stem in stems:
            name = tensor_name(layer, stem)
            shard = index[name]
            path = os.path.join(model_dir, shard)
            columns[stem].append(read_expert_row(path, name, expert))
    bank = {
        "kind": kind,
        "members": list(members),
        "n": len(members),
        "w13": np.concatenate([np.frombuffer(b, dtype=np.uint8) for b in columns["w13_weight"]]),
        "s13": np.concatenate([np.frombuffer(b, dtype=np.uint8) for b in columns["w13_weight_scale"]]),
        "w2": np.concatenate([np.frombuffer(b, dtype=np.uint8) for b in columns["w2_weight"]]),
        "s2": np.concatenate([np.frombuffer(b, dtype=np.uint8) for b in columns["w2_weight_scale"]]),
    }
    for key in ("w13", "s13", "w2", "s2"):
        bank[key] = np.ascontiguousarray(bank[key])
    if kind == "nvfp4":
        bank["amax13"] = np.concatenate([
            np.frombuffer(b, dtype=np.float32) for b in columns["w13_weight_amax"]]).astype(np.float32, copy=False)
        bank["amax2"] = np.concatenate([
            np.frombuffer(b, dtype=np.float32) for b in columns["w2_weight_amax"]]).astype(np.float32, copy=False)
        _meta, raw13 = _read_full(model_dir, index, tensor_name(layer, "w13_weight_outer_scale_denominator"))
        _meta, raw2 = _read_full(model_dir, index, tensor_name(layer, "w2_weight_outer_scale_denominator"))
        bank["denom13"] = float(np.frombuffer(raw13, dtype=np.float32)[0])
        bank["denom2"] = float(np.frombuffer(raw2, dtype=np.float32)[0])
    _check_shapes(bank)
    return bank


def _check_shapes(bank: dict) -> None:
    """Stored scales are one per row, 16 (NVFP4) or 32 (MXFP8) along K."""
    n = bank["n"]
    if n == 0:
        return
    two_i = INTER * 2
    if bank["kind"] == "nvfp4":
        w13 = n * two_i * (HIDDEN // 2)
        s13 = n * two_i * (HIDDEN // 16)
        w2 = n * HIDDEN * (INTER // 2)
        s2 = n * HIDDEN * (INTER // 16)
    elif bank["kind"] == "mxfp8":
        w13 = n * two_i * HIDDEN
        s13 = n * two_i * (HIDDEN // 32)
        w2 = n * HIDDEN * INTER
        s2 = n * HIDDEN * (INTER // 32)
    else:
        raise ValueError(bank["kind"])
    got = (bank["w13"].size, bank["s13"].size, bank["w2"].size, bank["s2"].size)
    expect = (w13, s13, w2, s2)
    if got != expect:
        raise ValueError(f"stored layout {got} != expert-major packed {expect}")


def local_ids(ids: np.ndarray, members: list[int]) -> np.ndarray:
    table = {expert: i for i, expert in enumerate(members)}
    out = np.full(ids.shape, -1, dtype=np.int32)
    for index, expert in np.ndenumerate(np.asarray(ids)):
        if int(expert) < 0:
            continue
        local = table.get(int(expert))
        if local is not None:
            out[index] = local
    return out
