"""vLLM hook: cold experts leave the GB300 and come back through the sidecar.

The day-0 class is ``vllm.models.inkling2.moe.Inkling2MoE`` (PR 67325, head
d664cd1e). ``route`` only understands a contiguous expert range, so the hook
replaces ``forward``. It selects experts with the reference rule, publishes
the token to ``/dev/shm/inkling2-peer``, and adds the sidecar's cold rows.

Hot experts call the sm_103a GEMMs in-process. Cold experts are published
to the sidecar, which calls the sm_120a GEMMs. Spill experts stay on the
CPU: the plan's Grace tier is a host dequant plus the fp64 reference, not a
third device kernel. All three stay behind INKLING2_ALLOW_UNVALIDATED=1.
INKLING2_REFERENCE=1 keeps the numpy path the CPU tests use. Attention
stays in vLLM.
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

        from bank import load_members, local_ids
        from cpu_tier import CpuTier
        from so import open_so

        cls = module.Inkling2MoE
        if getattr(cls, "_inkling_peer", False):
            return
        seg = Segment(segment_path, Layout(), create=False)
        seq = {"n": 0}
        quant_code = {"nvfp4": 0, "mxfp8": 1}
        allow = os.environ.get("INKLING2_ALLOW_UNVALIDATED") == "1"
        spill_cpu = os.environ.get("INKLING2_SPILL_CPU") == "1"
        gpu = {"so": None, "pool": None, "spill_banks": {}}

        def _banks(self):
            banks = getattr(self, "_inkling_banks", None)
            if banks is None:
                raise RuntimeError(
                    "INKLING2_PEER: hot and spill GEMMs are UNVALIDATED. "
                    "Set INKLING2_REFERENCE=1 only for a tiny reference bank.")
            return banks

        def _ensure_gpu(self, layer: int, row: dict):
            if getattr(self, "_inkling_ready", False):
                return
            model_dir = os.environ["INKLING2_MODEL_DIR"]
            so_path = os.environ.get(
                "INKLING2_GB300_SO",
                os.path.join(HERE, "prebuilt", "peer_inkling_gb300_sm103.so"))
            if gpu["so"] is None:
                gpu["so"] = open_so(so_path)
            import torch
            from gpus import cuda_ordinal, find
            chosen = find("gb300")
            torch.cuda.set_device(cuda_ordinal(chosen["pci"]))
            visible = torch.cuda.get_device_name()
            if "GB300" not in visible:
                raise RuntimeError(f"INKLING2_PEER is on {visible}, not the GB300 ({chosen['uuid']})")
            print(f"INKLING2_PEER GPU {visible} {chosen['uuid']}", flush=True)
            hot = load_members(model_dir, layer, row["hot"], row["quant"])
            self._inkling_hot = {k: (torch.as_tensor(np.ascontiguousarray(v), device="cuda")
                                     if isinstance(v, np.ndarray) else v)
                                 for k, v in hot.items()}
            self._inkling_hot["members"] = hot["members"]
            self._inkling_hot["spans"] = {
                key: hot[key].nbytes // hot["n"] for key in ("w13", "s13", "w2", "s2")
            } if hot["n"] else {}
            if spill_cpu and row["spill"]:
                self._inkling_spill = CpuTier(load_members(model_dir, layer, row["spill"], row["quant"]))
            else:
                self._inkling_spill = None
            self._inkling_ready = True

        def _spill_bank(layer: int, row: dict):
            bank = gpu["spill_banks"].get(layer)
            if bank is None:
                bank = load_members(os.environ["INKLING2_MODEL_DIR"], layer, row["spill"], row["quant"])
                gpu["spill_banks"][layer] = bank
            return bank

        def _pool():
            if gpu["pool"] is None:
                from spill_pool import SpillPool
                gib = float(os.environ.get("INKLING2_SPILL_POOL_GIB", "4"))
                gpu["pool"] = SpillPool(lambda _layer: None, int(gib * (1 << 30)))
                print(f"INKLING2_SPILL pool {gib} GiB path {gpu['pool'].mode}", flush=True)
            return gpu["pool"]

        def _launch_hot(self, kind: str, flat_bf16, ids_local, route, hot) -> "np.ndarray":
            import torch
            tokens, topk = ids_local.shape
            y = torch.zeros(tokens, flat_bf16.shape[-1], dtype=torch.bfloat16, device=flat_bf16.device)
            denom = float(hot.get("denom13", 1.0))
            gpu["so"].expert(
                kind, tokens, topk, int(hot["n"]), int(flat_bf16.shape[-1]), 2304,
                flat_bf16.data_ptr(), ids_local.data_ptr(), route.data_ptr(),
                hot["w13"].data_ptr(), hot["s13"].data_ptr(),
                hot["amax13"].data_ptr() if kind == "nvfp4" else 0,
                hot["w2"].data_ptr(), hot["s2"].data_ptr(),
                hot["amax2"].data_ptr() if kind == "nvfp4" else 0,
                denom, y.data_ptr(), None)
            return y.float().reshape(-1).cpu().numpy().reshape(tokens, -1)

        def _cold(self, layer: int, kind: str, flat, ids, weights) -> np.ndarray:
            y = np.zeros((flat.shape[0], flat.shape[1]), dtype=np.float64)
            cap = seg.layout.max_tokens
            for start in range(0, flat.shape[0], cap):
                stop = min(start + cap, flat.shape[0])
                seq["n"] = seq["n"] % 0xFFFF + 1
                seg.publish(seq["n"], layer, quant_code[kind], flat[start:stop],
                            ids[start:stop], weights[start:stop])
                deadline = time.monotonic() + wait_s
                while seg.resp() != seq["n"]:
                    if time.monotonic() > deadline:
                        raise RuntimeError("INKLING2_PEER: sidecar did not answer")
                    time.sleep(0.0002)
                y[start:stop] = seg.read_y(stop - start)
            return y

        def forward(self, x, reduce=True, push=None, lora=None):
            if lora is not None or push is not None or not reduce:
                raise RuntimeError("INKLING2_PEER covers one process that reduces")
            layer = getattr(self, "_inkling_layer", None)
            if layer is None:
                raise RuntimeError("INKLING2_PEER: layer index was not bound")
            import torch
            flat_t = x.detach().reshape(-1, x.shape[-1])
            flat = flat_t.float().cpu().numpy()
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
                if row["cold"]:
                    y = y + _cold(layer, kind, flat, ids, weights)
                if row["spill"]:
                    y = y + apply_tier(
                        flat, ids, weights, row["spill"], banks["spill"]["w13"], banks["spill"]["w2"],
                        kind, n_experts)
            elif not allow:
                raise RuntimeError(
                    "INKLING2_PEER: grouped GEMMs are UNVALIDATED. "
                    "Set INKLING2_ALLOW_UNVALIDATED=1 only after ./inkling2 validate passes. "
                    "Day-0 forward is still available with INKLING2_PEER unset.")
            else:
                _ensure_gpu(self, layer, row)
                route = torch.as_tensor(np.ascontiguousarray(weights, dtype=np.float32), device=flat_t.device)
                bf16 = flat_t if flat_t.dtype == torch.bfloat16 else flat_t.to(torch.bfloat16)
                spill_hits = []
                if row["spill"] and not spill_cpu:
                    spill_set = set(row["spill"])
                    spill_hits = [int(e) for e in dict.fromkeys(int(v) for v in ids.reshape(-1) if int(v) in spill_set)]
                if spill_hits:
                    bank = _spill_bank(layer, row)
                    pool = _pool()
                    pool.prefetch(layer, spill_hits, bank)
                    pool.wait()
                    pool.note(layer, spill_hits)
                    y = _launch_mixed(kind, bf16, ids, route, self._inkling_hot, bank, pool, layer, spill_hits)
                else:
                    ids_local = torch.as_tensor(local_ids(ids, row["hot"]), device=flat_t.device)
                    y = _launch_hot(kind, bf16, ids_local, route, self._inkling_hot)
                if row["cold"]:
                    y = y + _cold(layer, kind, flat, ids, weights)
                if self._inkling_spill is not None:
                    y = y + self._inkling_spill.forward(flat, ids, weights)

        def _launch_mixed(kind, flat_bf16, ids, route, hot, bank, pool, layer, spill_hits):
            import torch
            members = list(hot["members"]) + list(spill_hits)
            local = local_ids(ids, members)
            ids_t = torch.as_tensor(local, device=flat_bf16.device)
            spans = hot["spans"]
            w13, s13, w2, s2 = [], [], [], []
            for i in range(len(hot["members"])):
                w13.append(hot["w13"].data_ptr() + i * spans["w13"])
                s13.append(hot["s13"].data_ptr() + i * spans["s13"])
                w2.append(hot["w2"].data_ptr() + i * spans["w2"])
                s2.append(hot["s2"].data_ptr() + i * spans["s2"])
            for expert in spill_hits:
                view = pool.pointers(layer, expert)
                w13.append(view["w13"])
                s13.append(view["s13"])
                w2.append(view["w2"])
                s2.append(view["s2"])
            def pack(values):
                arr = np.ascontiguousarray(np.array(values, dtype=np.uint64))
                return arr, arr.ctypes.data
            w13a, w13p = pack(w13)
            s13a, s13p = pack(s13)
            w2a, w2p = pack(w2)
            s2a, s2p = pack(s2)
            tokens = ids_t.shape[0]
            y = torch.zeros(tokens, flat_bf16.shape[-1], dtype=torch.bfloat16, device=flat_bf16.device)
            amax13 = amax2 = 0
            if kind == "nvfp4":
                hot_a13 = (hot["amax13"].detach().float().cpu().numpy() if hot.get("n")
                           else np.zeros(0, dtype=np.float32))
                hot_a2 = (hot["amax2"].detach().float().cpu().numpy() if hot.get("n")
                          else np.zeros(0, dtype=np.float32))
                a13 = np.concatenate([
                    hot_a13,
                    np.array([bank["amax13"][bank["members"].index(e)] for e in spill_hits], dtype=np.float32),
                ])
                a2 = np.concatenate([
                    hot_a2,
                    np.array([bank["amax2"][bank["members"].index(e)] for e in spill_hits], dtype=np.float32),
                ])
                t13 = torch.as_tensor(a13, device=flat_bf16.device)
                t2 = torch.as_tensor(a2, device=flat_bf16.device)
                amax13, amax2 = t13.data_ptr(), t2.data_ptr()
            denom = float(hot.get("denom13", bank.get("denom13", 1.0)))
            gpu["so"].expert_each(
                kind, tokens, ids_t.shape[1], len(members), int(flat_bf16.shape[-1]), 2304,
                flat_bf16.data_ptr(), ids_t.data_ptr(), route.data_ptr(),
                w13p, s13p, amax13, w2p, s2p, amax2, denom, y.data_ptr(), None)
            del w13a, s13a, w2a, s2a
            return y.float().cpu().numpy()

        cls.forward = forward
        cls._inkling_peer = True
        mode = "reference" if reference else ("unvalidated kernels" if allow else "raised until allow flag")
        print(f"INKLING2_PEER ACTIVE on {TARGET} ({mode})", flush=True)

    def _bind(model) -> None:
        named = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "Inkling2MoE"]
        if not named:
            raise RuntimeError("INKLING2_PEER: no Inkling2MoE modules")
        for name, mod in named:
            mod._inkling_layer = _layer_of(name)
        if allow and not spill_cpu:
            for name, mod in model.named_modules():
                if "layers" not in name or not type(mod).__name__.endswith("Attention"):
                    continue
                layer = _layer_of(name)
                row = placement["layers"].get(str(layer))
                if not row or not row["spill"]:
                    continue
                orig = mod.forward

                def attn_forward(*args, _orig=orig, _layer=layer, _row=row, **kwargs):
                    pred = gpu["pool"].predicted.get(_layer, []) if gpu["pool"] is not None else []
                    if pred:
                        _pool().prefetch(_layer, pred, _spill_bank(_layer, _row))
                    return _orig(*args, **kwargs)

                mod.forward = attn_forward

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
