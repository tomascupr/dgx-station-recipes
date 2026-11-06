"""Count routed-expert hits per MoE layer, for the rowmap.

Loaded into every Python process of the vLLM container through PYTHONPATH=/routing; it acts only where a GPU model
runner is imported (vLLM has two, V1 and V2, and the engine picks one). After the model and its MTP drafter load, a
counter is bound to every MoE layer through vLLM's own routed-experts binding (bind_routed_experts_capturer), which
covers both router-based and monolithic MoE kernels. Each call adds its top-k expert ids to
counts[bucket, layer, expert], where bucket is the call's token count:
  0: 1-8 tokens   1: 9-64   2: 65-512   3: 513+
Calls of up to DECODE_MAX tokens are decode and verify steps (the peer tier's fused path); larger ones are prefill.
Column E counts ids of -1 (no expert).

Eager mode only (--enforce-eager): under CUDA graphs this Python would run at capture time only, and graph padding
tokens would be counted as traffic. It refuses to start otherwise. TP rank 0 writes $GLM53_ROUTE_OUT (JSON) every
EVERY_S seconds and at exit; counts are cumulative from engine start, so a workload phase is the difference of two
snapshots.
"""
import os

OUT = os.environ.get("GLM53_ROUTE_OUT")

if OUT:
    import atexit
    import importlib.abc
    import importlib.util
    import json
    import sys
    import time

    TARGETS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")
    EDGES = (8, 64, 512)
    DECODE_MAX = 64   # Garner's peer-tier fused-path threshold
    EVERY_S = 5

    class _Counter:
        def __init__(self, n_layers: int, n_experts: int, device, rank0: bool, target: list, drafter: list):
            import torch
            self.E, self.rank0, self.target, self.drafter = n_experts, rank0, target, drafter
            self.counts = torch.zeros((len(EDGES) + 1, n_layers, n_experts + 1), dtype=torch.int64, device=device)
            self.tokens = [[0] * n_layers for _ in range(len(EDGES) + 1)]
            self.ones = torch.ones(1 << 20, dtype=torch.int64, device=device)
            self.last = 0.0

        def capture(self, layer_id: int, topk_ids) -> None:
            n = int(topk_ids.shape[0])
            b = sum(n > e for e in EDGES)
            ids = topk_ids.reshape(-1).long() % (self.E + 1)   # -1 (no expert) lands in column E
            if ids.numel() > self.ones.numel():
                raise RuntimeError(f"GLM53_ROUTE: {ids.numel()} ids in one call exceed the counter's buffer")
            self.counts[b, layer_id].index_add_(0, ids, self.ones[: ids.numel()])
            self.tokens[b][layer_id] += n
            now = time.monotonic()
            if self.rank0 and now - self.last >= EVERY_S:
                self.last = now
                self.dump()

        def dump(self) -> None:
            rec = dict(schema="glm53-route-counts-v2", time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       bucket_edges=list(EDGES), decode_max_tokens=DECODE_MAX, n_experts=self.E,
                       target_layers=self.target, drafter_layers=self.drafter,
                       tokens=self.tokens, counts=self.counts.cpu().tolist())
            with open(OUT + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(OUT + ".tmp", OUT)

    def _bind(runner) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.fused_moe.layer import MoERunner
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            RoutedExpertsCaptureSource, bind_routed_experts_capturer)
        vc = runner.vllm_config
        if not vc.model_config.enforce_eager:
            raise RuntimeError("GLM53_ROUTE needs --enforce-eager (see routing/sitecustomize.py)")

        def moe_layers(model) -> list:
            return sorted({m.layer_id for m in model.modules() if isinstance(m, (MoERunner, RoutedExpertsCaptureSource))})

        spec = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
        drafter = getattr(spec, "model", None)
        target = moe_layers(runner.model)
        draft = moe_layers(drafter) if drafter is not None else []
        if not target:
            raise RuntimeError("GLM53_ROUTE: the target model has no MoE layers to count")
        rank0 = get_tensor_model_parallel_rank() == 0
        counter = _Counter(max(target + draft) + 1, vc.model_config.hf_text_config.n_routed_experts, runner.device,
                           rank0, target, draft)
        bind_routed_experts_capturer(runner.model, counter)
        if drafter is not None:
            bind_routed_experts_capturer(drafter, counter)
        if rank0:
            atexit.register(counter.dump)
            print(f"GLM53_ROUTE ACTIVE: counting target layers {target[0]}-{target[-1]} ({len(target)}), "
                  f"drafter layers {draft} -> {OUT}", flush=True)

    def _patch(module) -> None:
        cls = module.GPUModelRunner
        orig = cls.load_model

        def load_model(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            _bind(self)
            return result

        cls.load_model = load_model

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"GLM53_ROUTE: cannot find {name}")
            exec_module = spec.loader.exec_module

            def wrapped(module):
                exec_module(module)
                _patch(module)

            spec.loader.exec_module = wrapped
            return spec

    for _name in TARGETS:
        if _name in sys.modules:
            _patch(sys.modules[_name])
    sys.meta_path.insert(0, _Hook())
