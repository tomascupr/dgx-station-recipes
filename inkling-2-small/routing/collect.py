"""Record per-layer expert hits during a GPU forward.

Wraps ``vllm.models.inkling2.moe.Inkling2MoE.forward`` after the model loads.
The wrapper repeats the gate as a dense GEMM and the top-8 rule in
``kernels/reference/quant.py``, writes the hit, then calls the original
forward. vLLM's router is what serves; this pass only counts. It refuses to
start unless the engine is eager, because a CUDA graph would count the
capture instead of the real tokens.

Set INKLING2_ROUTE_OUT to the profile path. The vLLM class names are PR
67325 head d664cd1e and are unvalidated until the DGX run.
"""
from __future__ import annotations

import os
import sys

OUT = os.environ.get("INKLING2_ROUTE_OUT")
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "kernels", "reference")
RUNNERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")


def _layer_of(name: str) -> int:
    parts = name.split(".")
    if "layers" not in parts:
        raise RuntimeError(f"INKLING2_ROUTE: no layer index in {name}")
    i = parts.index("layers")
    if i + 1 >= len(parts) or not parts[i + 1].isdigit():
        raise RuntimeError(f"INKLING2_ROUTE: no layer index in {name}")
    return int(parts[i + 1])


def install() -> None:
    if not OUT:
        return
    import atexit
    import importlib.abc
    import importlib.util

    sys.path.insert(0, HERE)
    sys.path.insert(0, REF)
    import numpy as np
    from profile import add_hits, dump_profile, empty_counts
    from quant import select_experts

    state = {"counts": None}

    def _dump() -> None:
        if state["counts"] is not None:
            dump_profile(OUT, state["counts"], {"source": "inkling2-route-collector"})

    atexit.register(_dump)

    def _wrap(mod, layer: int) -> None:
        import torch
        orig = mod.forward

        def forward(x, *args, **kwargs):
            if not getattr(x, "is_cuda", False):
                raise RuntimeError("INKLING2_ROUTE: the collector runs on the GPU forward")
            flat = x.reshape(-1, x.shape[-1]).float()
            logits = torch.mm(flat, mod.gate.weight.float().T)
            _weights, ids = select_experts(
                logits.detach().cpu().numpy(), mod.gate.bias.detach().float().cpu().numpy())
            add_hits(state["counts"], layer, np.asarray(ids))
            return orig(x, *args, **kwargs)

        mod.forward = forward

    def _bind(model) -> None:
        named = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Inkling2MoE"]
        if not named:
            raise RuntimeError("INKLING2_ROUTE: no Inkling2MoE modules")
        layers = [_layer_of(name) for name, _mod in named]
        n_exp = int(named[0][1].gate.weight.shape[0])
        state["counts"] = empty_counts(max(layers) + 1, n_exp)
        for name, mod in named:
            _wrap(mod, _layer_of(name))
        print(f"INKLING2_ROUTE ACTIVE: layers {min(layers)}-{max(layers)} -> {OUT}", flush=True)

    def _patch_runner(module) -> None:
        orig = module.GPUModelRunner.load_model

        def load_model(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            if not self.vllm_config.model_config.enforce_eager:
                raise RuntimeError("INKLING2_ROUTE needs --enforce-eager")
            _bind(self.model)
            return result

        module.GPUModelRunner.load_model = load_model

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in RUNNERS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            exec_module = spec.loader.exec_module

            def wrapped(module, _exec=exec_module):
                _exec(module)
                _patch_runner(module)

            spec.loader.exec_module = wrapped
            return spec

    for name in RUNNERS:
        if name in sys.modules:
            _patch_runner(sys.modules[name])
    sys.meta_path.insert(0, _Finder())


if OUT:
    install()
