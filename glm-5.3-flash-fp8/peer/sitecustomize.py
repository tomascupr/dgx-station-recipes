"""The GB300 side of the peer tier, loaded into vLLM through PYTHONPATH=/w when GLM53_PEER_CHANNEL is set.

vLLM loads every routed expert into pinned Grace memory (UVA offload of w13_weight and w2_weight). After loading, this
hook, for each target layer of the rowmap, copies the hot experts into HBM, releases the layer's host copy, and routes
the layer through glm53_peer_moe; the MTP layer's experts are copied into HBM whole. glm53_peer_moe computes the
layer's top-k once (sigmoid with the correction bias, renormalized, scaled by routed_scaling_factor), sends the rows to
the RTX sidecar, runs the hot experts and the shared expert here, then waits for the sidecar's cold sum and adds it.
Everything it launches records into CUDA graphs. Apply or raise: any mismatch with the expected model, kernel layout
or devices stops the engine with one line; a sidecar that stops answering is caught by a watchdog that ends the
process. One ACTIVE line when bound.
"""
import os

CHANNEL = os.environ.get("GLM53_PEER_CHANNEL")

if CHANNEL:
    import importlib.abc
    import importlib.util
    import json
    import re
    import sys
    import threading
    import time

    import torch

    TARGETS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")
    ROWMAP = os.environ.get("GLM53_PEER_ROWMAP", "/w/rowmap-mix-v1-h202.json")
    MODEL = os.environ.get("GLM53_PEER_MODEL", "/model")
    REG: dict = {}
    STATE: dict = {}

    def _peer_moe(hidden_states: torch.Tensor, layer: int) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
        from peer_moe import experts_fp8
        st, ch = REG[layer], STATE["channel"]
        mod, cfg = st["mod"], STATE["cfg"]
        x = hidden_states.contiguous()
        logits, _ = mod.gate(x)
        tw, ti = fused_topk_bias(hidden_states=x, gating_output=logits, scoring_func=cfg.scoring_func,
                                 e_score_correction_bias=mod.gate.e_score_correction_bias.data,
                                 topk=cfg.num_experts_per_tok, renormalize=cfg.norm_topk_prob,
                                 indices_type=torch.int32, routed_scaling_factor=cfg.routed_scaling_factor)
        tw, ti = tw.to(torch.float32).contiguous(), ti.contiguous()
        ch.send(x, ti, tw, layer)
        out = experts_fp8(x, st["w1"], st["w2"], st["s1"], st["s2"], tw, ti, st["map"], cfg.swiglu_limit)
        out = out + mod.shared_experts(x)
        ch.wait_add(out)
        return out

    def _peer_moe_fake(hidden_states: torch.Tensor, layer: int) -> torch.Tensor:
        return torch.empty_like(hidden_states)

    def _checkpoint_tensor(name: str):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(MODEL, idx[name]), "pt") as f:
            return f.get_tensor(name)

    def _bind(runner) -> None:
        from vllm.models.glm5next.common import model as gm
        from vllm.utils.torch_utils import direct_register_custom_op
        from channel import ERR, READY, READY_MAGIC, SIZE, GB300Side

        dev = runner.device
        if torch.cuda.get_device_capability(dev) != (10, 3):
            raise RuntimeError(f"GLM53_PEER: vLLM is on {torch.cuda.get_device_name(dev)}, expected the GB300 (10.3)")
        cfg = runner.vllm_config.model_config.hf_text_config
        if (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.scoring_func) != (288, 8, "sigmoid"):
            raise RuntimeError("GLM53_PEER: not the GLM-5.3-Flash MoE this rowmap was built for")
        rowmap = {int(k): v for k, v in json.load(open(ROWMAP))["layers"].items()}
        mods = {}
        for name, m in runner.model.named_modules():
            if isinstance(m, gm.Glm5NextMoE):
                mods[int(re.search(r"layers\.(\d+)\.", name + ".")[1])] = m
        if set(rowmap) - set(mods):
            raise RuntimeError(f"GLM53_PEER: rowmap layers {sorted(set(rowmap) - set(mods))} are not MoE layers")
        t0 = time.time()
        for layer, split in sorted(rowmap.items()):
            m = mods[layer]
            if m.use_mega_moe or m.shared_experts is None or m.is_fused_shared_expert_enabled:
                raise RuntimeError(f"GLM53_PEER: layer {layer} is not the unfused Triton MoE layout")
            p = {n.rsplit(".", 1)[-1]: q for n, q in m.experts.named_parameters()}
            w13, w2 = p["w13_weight"], p["w2_weight"]
            s13 = p.get("w13_weight_scale_inv", p.get("w13_weight_scale"))
            s2 = p.get("w2_weight_scale_inv", p.get("w2_weight_scale"))
            if w13.shape != (288, 4096, 4096) or w2.shape != (288, 4096, 2048) or s13 is None or s2 is None:
                raise RuntimeError(f"GLM53_PEER: layer {layer} expert tensors {tuple(w13.shape)} / {tuple(w2.shape)}")
            hot = torch.tensor(split["hot"], device=dev)
            if layer == min(rowmap):
                e, pre = split["hot"][0], f"model.language_model.layers.{layer}.mlp.experts.{split['hot'][0]}"
                gate = _checkpoint_tensor(f"{pre}.gate_proj.weight").to(dev)
                sgate = _checkpoint_tensor(f"{pre}.gate_proj.weight_scale_inv").to(dev)
                if not (torch.equal(w13[e, :2048].view(torch.uint8), gate.view(torch.uint8))
                        and torch.equal(s13[e, :16], sgate)):
                    raise RuntimeError("GLM53_PEER: vLLM's expert tensors are not in the checkpoint layout; "
                                       "serve with --moe-backend triton")
            emap = torch.full((288,), -1, dtype=torch.int32, device=dev)
            emap[hot] = torch.arange(len(split["hot"]), dtype=torch.int32, device=dev)
            REG[layer] = dict(mod=m, map=emap, w1=w13.index_select(0, hot), w2=w2.index_select(0, hot),
                              s1=s13.index_select(0, hot), s2=s2.index_select(0, hot))
            for q in (w13, w2):
                q.data = torch.empty(0, dtype=q.dtype, device=dev)
            m._glm53_layer = layer
        moved = 0
        spec = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
        for model in (runner.model, getattr(spec, "model", None)):
            for n, q in (model.named_parameters() if model is not None else ()):
                if getattr(q, "_vllm_is_uva_offloaded", False) and q.numel():
                    q.data = q.data.clone()
                    q._vllm_is_uva_offloaded = False
                    moved += q.numel() * q.element_size()
        hot_gib = sum(t.numel() * t.element_size() for r in REG.values() for k, t in r.items()
                      if k in ("w1", "w2", "s1", "s2")) / 2**30
        STATE["cfg"] = cfg
        direct_register_custom_op("glm53_peer_moe", _peer_moe, mutates_args=[], fake_impl=_peer_moe_fake)
        orig = gm.Glm5NextMoE.forward

        def forward(self, hidden_states, already_sequence_parallel=False):
            layer = getattr(self, "_glm53_layer", None)
            if layer is None:
                return orig(self, hidden_states, already_sequence_parallel)
            return torch.ops.vllm.glm53_peer_moe(hidden_states, layer)

        gm.Glm5NextMoE.forward = forward
        deadline = time.time() + 1800
        while not (os.path.exists(CHANNEL) and os.path.getsize(CHANNEL) == SIZE):
            if time.time() > deadline:
                raise RuntimeError(f"GLM53_PEER: no sidecar channel at {CHANNEL} after 30 min")
            time.sleep(2)
        ch = GB300Side(CHANNEL, dev)
        while int(ch.cpu["hdr"][READY]) != READY_MAGIC:
            if time.time() > deadline:
                raise RuntimeError("GLM53_PEER: the sidecar never marked the channel ready")
            time.sleep(2)
        STATE["channel"] = ch

        def watchdog():
            while True:
                bad = int(ch.cpu["hdr"][ERR])
                if bad:
                    print(f"GLM53_PEER FATAL: the sidecar did not answer request {bad}; stopping", flush=True)
                    os._exit(1)
                time.sleep(1)

        threading.Thread(target=watchdog, daemon=True, name="glm53-peer-watchdog").start()
        print(f"GLM53_PEER ACTIVE: {len(REG)} layers split, hot experts {hot_gib:.1f} GiB in HBM in "
              f"{time.time() - t0:.0f} s, {moved / 2**30:.2f} GiB of MTP experts moved to HBM, sidecar on {CHANNEL}",
              flush=True)

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
                raise RuntimeError(f"GLM53_PEER: cannot find {name}")
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
