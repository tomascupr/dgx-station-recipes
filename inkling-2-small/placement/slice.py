"""Views of one expert inside a packed 384-wide tensor, and the GB300 load filter."""
from __future__ import annotations

from headers import tensor_nbytes

N_EXPERTS = 384


def expert_span(meta: dict, n_experts: int = N_EXPERTS) -> int:
    nbytes = tensor_nbytes(meta)
    if not meta["shape"] or int(meta["shape"][0]) != n_experts:
        raise ValueError(f"tensor shape {meta['shape']} is not packed {n_experts}-wide")
    if nbytes % n_experts:
        raise ValueError(f"{nbytes} bytes do not divide into {n_experts} experts")
    return nbytes // n_experts


def expert_offsets(meta: dict, expert_id: int, n_experts: int = N_EXPERTS) -> tuple[int, int]:
    """Byte offsets relative to the tensor's data_offsets[0]."""
    if not 0 <= expert_id < n_experts:
        raise ValueError(f"expert {expert_id} is outside 0..{n_experts - 1}")
    span = expert_span(meta, n_experts)
    return expert_id * span, (expert_id + 1) * span


def file_offsets(meta: dict, expert_id: int, n_experts: int = N_EXPERTS) -> tuple[int, int]:
    """Absolute offsets inside the safetensors payload region (after the header)."""
    start, _end = meta["data_offsets"]
    lo, hi = expert_offsets(meta, expert_id, n_experts)
    return start + lo, start + hi


def coldskip(placement: dict, layer: int, expert_id: int) -> bool:
    """True when the GB300 loader must not copy this expert into HBM."""
    row = placement["layers"][str(layer)]
    return expert_id not in row["hot"]


def hot_rows(placement: dict, layer: int) -> list[int]:
    return list(placement["layers"][str(layer)]["hot"])
