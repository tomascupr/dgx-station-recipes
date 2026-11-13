"""Expert-hit profile the placement planner consumes.

The counts themselves come from a GPU forward. This module only accumulates
ids and writes the JSON. ``-1`` (no expert) is ignored.
"""
from __future__ import annotations

import json
import os

import numpy as np

SCHEMA = "inkling2-expert-profile-v1"


def empty_counts(n_layers: int, n_experts: int) -> np.ndarray:
    return np.zeros((n_layers, n_experts), dtype=np.int64)


def add_hits(counts: np.ndarray, layer: int, ids: np.ndarray) -> None:
    flat = np.asarray(ids, dtype=np.int64).reshape(-1)
    flat = flat[flat >= 0]
    if flat.size and (flat.max() >= counts.shape[1] or flat.min() < 0):
        raise ValueError(f"expert id outside 0..{counts.shape[1] - 1}")
    np.add.at(counts[layer], flat, 1)


def dump_profile(path: str, counts: np.ndarray, meta: dict | None = None) -> None:
    body = {
        "schema": SCHEMA,
        "n_layers": int(counts.shape[0]),
        "n_experts": int(counts.shape[1]),
        "counts": {str(i): counts[i].tolist() for i in range(counts.shape[0])},
    }
    if meta:
        body["meta"] = meta
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(body, f)
    os.replace(tmp, path)


def load_counts(path: str) -> np.ndarray:
    with open(path) as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not {SCHEMA}")
    n_layers, n_experts = int(raw["n_layers"]), int(raw["n_experts"])
    counts = empty_counts(n_layers, n_experts)
    for key, row in raw["counts"].items():
        counts[int(key)] = row
    return counts
