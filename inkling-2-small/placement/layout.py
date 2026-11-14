"""Per-layer expert bytes from safetensors headers."""
from __future__ import annotations

import re

from headers import tensor_nbytes

EXPERT_RE = re.compile(
    r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)"
    r"(_scale|_amax|_outer_scale_denominator)?$"
)
N_EXPERTS = 384


class LayerLayout:
    def __init__(self, index: int):
        self.index = index
        self.tensors: dict[str, dict] = {}

    @property
    def quant(self) -> str:
        w13 = self.tensors.get("w13_weight")
        if w13 is None:
            raise ValueError(f"layer {self.index} has no w13_weight")
        if "w13_weight_amax" in self.tensors:
            if w13["dtype"] != "U8":
                raise ValueError(f"layer {self.index}: amax with dtype {w13['dtype']}")
            return "nvfp4"
        if w13["dtype"] == "F8_E4M3":
            return "mxfp8"
        raise ValueError(f"layer {self.index}: unrecognised expert dtype {w13['dtype']}")

    def per_expert_bytes(self) -> int:
        """Bytes of one expert, excluding the layer-shared denominator scalars."""
        total = 0
        for name, meta in self.tensors.items():
            if name.endswith("_outer_scale_denominator"):
                continue
            nbytes = tensor_nbytes(meta)
            if not meta["shape"] or int(meta["shape"][0]) != N_EXPERTS:
                raise ValueError(f"layer {self.index} {name} is not packed {N_EXPERTS}-wide")
            if nbytes % N_EXPERTS:
                raise ValueError(f"layer {self.index} {name} is not divisible by {N_EXPERTS}")
            total += nbytes // N_EXPERTS
        return total

    def shared_bytes(self) -> int:
        return sum(tensor_nbytes(meta) for name, meta in self.tensors.items()
                   if name.endswith("_outer_scale_denominator"))


def classify(headers: dict[str, dict]) -> tuple[dict[int, LayerLayout], int, int]:
    """Returns layers, expert bytes (including shared denominators), other bytes.

    ``headers`` maps a shard name to its parsed safetensors header.
    """
    layers: dict[int, LayerLayout] = {}
    expert_bytes = 0
    other_bytes = 0
    for shard, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            match = EXPERT_RE.match(name)
            if match is None:
                other_bytes += tensor_nbytes(meta)
                continue
            layer = int(match.group(1))
            key = "w" + name.split(".experts.w", 1)[1]
            slot = layers.setdefault(layer, LayerLayout(layer))
            if key in slot.tensors:
                raise ValueError(f"{name} appears in more than one shard")
            slot.tensors[key] = meta
            expert_bytes += tensor_nbytes(meta)
    return layers, expert_bytes, other_bytes


def assert_uniform(layers: dict[int, LayerLayout]) -> dict[str, int]:
    """One per-expert size per quant group. Raises if a layer breaks the group."""
    sizes: dict[str, int] = {}
    for layer in sorted(layers):
        layout = layers[layer]
        kind = layout.quant
        size = layout.per_expert_bytes()
        if kind not in sizes:
            sizes[kind] = size
        elif sizes[kind] != size:
            raise ValueError(
                f"layer {layer} {kind} expert is {size} bytes, group is {sizes[kind]}")
    return sizes
