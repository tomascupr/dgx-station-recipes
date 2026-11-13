"""vLLM hook: cold experts leave the GB300 and come back through the sidecar.

The day-0 class is ``vllm.models.inkling2.moe.Inkling2MoE`` (PR 67325, head
d664cd1e). ``route`` only understands a contiguous expert range, so the hook
replaces ``forward``. It selects experts with the reference rule, publishes
the token to ``/dev/shm/inkling2-peer``, and adds the sidecar's cold rows.

The hot and spill GEMMs are the numpy reference when INKLING2_REFERENCE=1
(the CPU tests). Otherwise they raise: the grouped kernels in kernels/csrc
are UNVALIDATED and are not called from this hook. Attention stays in vLLM.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys

TARGET = "vllm.models.inkling2.moe"
RUNNERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "kernels", "reference")


def _layer_of(name: str) -> int:
    parts = name.split(".")
    i = parts.index("layers")
    return int(parts[i + 1])


def install() -> None:
    with open(os.environ["INKLING2_PLACEMENT"]) as f:
        placement = json.load(f)
    segment_path = os.environ.get("INKLING2_SEGMENT", "/dev/shm/inkling2-peer")
    reference = os.environ.get("INKLING2_REFERENCE") == "1"
    wait_s = float(os.environ.get("INKLING2_PEER_WAIT_S", "120"))

    def _patch(module) -> None:
        import time

        import numpy as np

        sys.path.insert(0, HERE)
        sys.path.insert(0, REF)
        from quant import select_experts
        from segment import Layout, Segment
        from split import apply_tier

        cls = module.Inkling2MoE
        if getattr(cls, "_inkling_peer", False):
            return
        seg = Segment(segment_path, Layout(), create=False)
        seq = {"n": 0}
        quant_code = {"nvfp4": 0, "mxfp8": 1}

        def _banks(self):
            banks = getattr(self, "_inkling_banks", None)
            if banks is None:
                raise RuntimeError(
                    "INKLING2_PEER: hot and spill GEMMs are UNVALIDATED. "
                    "Set INKLING2_REFERENCE=1 only for a tiny reference bank.")
            return banks

        def forward(self, x, reduce=True, push=None, lora=None):
            if lora is not None or push is not None or not reduce:
                raise RuntimeError("INKLING2_PEER covers one process that reduces")
            layer = getattr(self, "_inkling_layer", None)
            if layer is None:
                raise RuntimeError("INKLING2_PEER: layer index was not bound")
            flat = x.detach().float().reshape(-1, x.shape[-1]).cpu().numpy()
            logits = flat @ self.gate.weight.detach().float().cpu().numpy().T
            bias = self.gate.bias.detach().float().cpu().numpy()
            weights, ids = select_experts(
                logits, bias, topk=int(self.gate.topk), route_scale=float(self.gate.route_scale))
            row = placement["layers"][str(layer)]
            kind = row["quant"]
            n_experts = int(placement["meta"]["n_experts"])
            if reference:
                banks = _banks(self)
                y = apply_tier(flat, ids, weights, row["hot"], banks["hot"]["w13"], banks["hot"]["w2"],
                               kind, n_experts)
            else:
                raise RuntimeError(
                    "INKLING2_PEER: the GB300 grouped GEMM is UNVALIDATED "
                    "(kernels/csrc, kernels/build.sh). The day-0 forward is still available "
                    "with INKLING2_PEER unset.")
            seq["n"] = seq["n"] % 0xFFFF + 1
            seg.publish(seq["n"], layer, quant_code[kind], flat, ids, weights)
            deadline = time.monotonic() + wait_s
            while seg.resp() != seq["n"]:
                if time.monotonic() > deadline:
                    raise RuntimeError("INKLING2_PEER: sidecar did not answer")
                time.sleep(0.0002)
            y = y + seg.read_y(flat.shape[0])
            if row["spill"] and reference:
                y = y + apply_tier(
                    flat, ids, weights, row["spill"], banks["spill"]["w13"], banks["spill"]["w2"],
                    kind, n_experts)
            out = x.new_empty(x.shape)
            return out.copy_(__import__("torch").from_numpy(np.asarray(y, dtype=np.float32)))

        cls.forward = forward
        cls._inkling_peer = True
        print(f"INKLING2_PEER ACTIVE on {TARGET} ({'reference' if reference else 'kernel unset'})", flush=True)

    def _bind(model) -> None:
        named = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Inkling2MoE"]
        if not named:
            raise RuntimeError("INKLING2_PEER: no Inkling2MoE modules")
        for name, mod in named:
            mod._inkling_layer = _layer_of(name)

    def _patch_runner(module) -> None:
        orig = module.GPUModelRunner.load_model

        def load_model(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            _bind(self.model)
            return result

        module.GPUModelRunner.load_model = load_model

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self, names, patch):
            self.names = set(names)
            self.patch = patch

        def find_spec(self, name, path, target=None):
            if name not in self.names:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            exec_module = spec.loader.exec_module

            def wrapped(module, _exec=exec_module, _patch=self.patch):
                _exec(module)
                _patch(module)

            spec.loader.exec_module = wrapped
            return spec

    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder((TARGET,), _patch))
    for name in RUNNERS:
        if name in sys.modules:
            _patch_runner(sys.modules[name])
    sys.meta_path.insert(0, _Finder(RUNNERS, _patch_runner))
