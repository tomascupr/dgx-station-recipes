"""Kimi K3.1 Pro's routed experts from packed 2-bit files, inside stock vLLM. Loaded through PYTHONPATH when
KIMI_PACK is set.

vLLM loads a checkpoint without routed experts (kstrip.py). This hook
  1. builds every MoE layer with a small stand-in instead of vLLM's expert tensors (which would be 1.3 TiB) and gives
     the layer the forward of vLLM's "routed experts as one call" branch: router -> experts(latent, weights, ids) ->
     shared experts, latent norm and up-projection, each of them vLLM's own module;
  2. after the model has loaded, places each layer's experts by usage rank (KIMI_RANK, most used first): the
     KIMI_PEER experts from rank KIMI_PEER_FIRST on go to the RTX sidecar (not held here); of the others the first
     KIMI_HOT go to HBM, the last KIMI_PRUNE nowhere (the router can no longer choose them), the rest to pinned
     Grace memory that the GPU reads in place;
  3. computes the experts with kmoe.packed_experts, one launch reading HBM and Grace memory through one address
     table; with KIMI_PEER > 0 the layer's rows, ids and weights first go to the sidecar (ksidecar.py) through the
     shared-memory channel KIMI_CHANNEL (kchannel.py), and the sidecar's sum is added when this GPU's part is done.
Everything it launches records into vLLM's CUDA graphs. Apply or raise: a model or file that is not what this was
built for stops the engine with one line. One ACTIVE line when bound.
"""
import os

PACK = os.environ.get("KIMI_PACK")

if PACK:
    import importlib.abc
    import importlib.util
    import json
    import mmap
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    MODEL = "vllm.models.kimi_k3.nvidia.model"
    RUNNERS = ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner")
    RANK = os.environ.get("KIMI_RANK", "/w/rank-native.json")
    HOT, PEER, PRUNE = (int(os.environ.get(k, d)) for k, d in (("KIMI_HOT", "150"), ("KIMI_PEER", "0"),
                                                                ("KIMI_PRUNE", "0")))
    CHANNEL = os.environ.get("KIMI_CHANNEL")
    NOOP = os.environ.get("KIMI_PACK_NOOP") == "1"      # measure the step time without routed experts (wrong output)
    GROUPED = True if os.environ.get("KIMI_PACK_GROUPED") == "1" else None   # checks: the prefill kernel for every call
    PEER_FIRST = int(os.environ.get("KIMI_PEER_FIRST", HOT))     # rank of the first expert the sidecar holds
    REG: dict = {}
    STATE: dict = {}

    class PackedExperts(torch.nn.Module):
        """What KimiMoE calls for its routed experts: latent rows, top-k weights and ids in, latent sum out."""

        def __init__(self, top_k: int):
            super().__init__()
            self.top_k = top_k
            self.layer = -1

        def forward(self, hidden_states, topk_weights, topk_ids):
            return torch.ops.vllm.kimi_packed_moe(hidden_states, topk_weights, topk_ids, self.layer)

    def _packed_moe(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, layer: int) -> torch.Tensor:
        from kmoe import packed_experts
        st = REG[layer]
        if NOOP:                 # timing only: everything but the routed experts
            return torch.zeros_like(x)
        x, ids, w = x.contiguous(), ids.to(torch.int32).contiguous(), w.to(torch.float32).contiguous()
        ch = STATE.get("channel")
        if ch is not None:
            ch.send(x, ids, w, layer)        # the RTX computes its experts while this GPU computes the rest
        y = packed_experts(x, ids, w, st["hot"], st["table"], st["own"], STATE["lay"], STATE["beta"],
                           STATE["linear_beta"], grouped=GROUPED)
        if ch is not None:
            ch.wait_add(y)
        return y.to(x.dtype)

    def _packed_moe_fake(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, layer: int) -> torch.Tensor:
        return torch.empty_like(x)

    def _forward(self, hidden_states):
        # vLLM's own steps for "routed experts as one call" (its MegaMoE branch), with the packed experts
        mod = STATE["mod"]
        num_tokens, hidden_size = hidden_states.shape
        latent, _ = self.routed_expert_down_proj(hidden_states)
        logits, _ = self.gate(hidden_states)
        weights, ids = mod.fused_grouped_topk(
            hidden_states=hidden_states, gating_output=logits, topk=self.experts.top_k,
            renormalize=self.moe_renormalize, e_score_correction_bias=self.gate.e_score_correction_bias.data,
            num_expert_group=self.num_expert_group, topk_group=self.topk_group,
            scoring_func=self.moe_router_activation_func, routed_scaling_factor=self.routed_scaling_factor)
        routed = self.experts(latent, weights, ids)
        out = self.routed_output_transform(routed, residual=self.shared_experts(hidden_states))
        return out.view(num_tokens, hidden_size)

    def _patch_model(mod) -> None:
        cls = mod.KimiMoE
        init = cls.__init__

        def __init__(self, config, *args, **kwargs):
            if (config.num_experts, config.num_experts_per_token, config.hidden_act) != (896, 16, "situ"):
                raise RuntimeError("KIMI_PACK: not the Kimi K3.1 Pro MoE the packed experts were made from")
            factory = mod.FusedMoEFactory
            mod.FusedMoEFactory = lambda **kw: PackedExperts(kw["top_k"])
            try:
                init(self, config, *args, **kwargs)
            finally:
                mod.FusedMoEFactory = factory
            if not isinstance(self.experts, PackedExperts) or self.routed_output_transform is None:
                raise RuntimeError("KIMI_PACK: KimiMoE did not take the latent, single-GPU layout this hook expects")
            self.use_mega_moe = True          # while loading: the weight loader then expects no expert tensors

        STATE["mod"] = mod
        cls.__init__ = __init__
        cls.forward = _forward

    @torch.no_grad()
    def _selfcheck(m, layer, src, dev) -> None:
        """The first MoE layer, as it will run (vLLM's router and tail, the packed kernel, this process's HBM and
        Grace addresses), against the reference model's formulas on float32 experts unpacked from the file."""
        from kmoe import UNIT, packed_experts, reference
        st, lay = REG[layer], STATE["lay"]
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn((4, m.gate.weight.shape[1]), generator=g).to(dev, torch.bfloat16)
        scores = torch.nn.functional.linear(x.float(), m.gate.weight.float()).sigmoid()
        ids = (scores + m.gate.e_score_correction_bias.float()).topk(m.experts.top_k, dim=-1)[1]
        w = scores.gather(1, ids)
        w = w / w.sum(-1, keepdim=True) * m.routed_scaling_factor
        latent, _ = m.routed_expert_down_proj(x)
        blob_of = lambda e: torch.from_numpy(np.array(src[e])).to(dev)      # noqa: E731
        want = reference(latent, ids, w, blob_of, st["own"], lay, STATE["beta"], STATE["linear_beta"])
        got = packed_experts(latent.contiguous(), ids.int(), w.float(), st["hot"], st["table"], st["own"], lay,
                             STATE["beta"], STATE["linear_beta"])
        rel = ((got - want).norm() / want.norm()).item()
        entry = st["table"][ids.long()] * UNIT
        tiers = [int((st["own"][ids] >= 0).sum()),
                 int((((entry >= st["hot"].numel()) | (entry < 0)) & (st["own"][ids] >= 0)).sum())]
        if PEER:                 # the whole block also has the sidecar's experts in it
            want = reference(latent, ids, w, blob_of, st["all"], lay, STATE["beta"], STATE["linear_beta"])
        want_out = (m.routed_output_transform(want.to(x.dtype)) + m.shared_experts(x)).float()
        got_out = _forward(m, x).float()
        rel_out = ((got_out - want_out).norm() / want_out.norm()).item()
        print(f"KIMI_PACK selfcheck, layer {layer}: experts from packed bytes vs unpacked float32 {rel:.2e} "
              f"({tiers[0]} of {ids.numel()} routes held here, {tiers[1]} of them in Grace memory); whole MoE block with "
              f"vLLM's router vs the reference formulas {rel_out:.2e}", flush=True)
        if rel > 2e-2 or rel_out > 2e-2:
            raise RuntimeError("KIMI_PACK: selfcheck failed")

    def _connect(dev) -> None:
        """Wait for the sidecar's channel, map it, and watch it: a sidecar that stops answering ends this process."""
        import threading
        from kchannel import ERR, READY, READY_MAGIC, SIZE, GB300Side
        if not CHANNEL:
            raise RuntimeError("KIMI_PACK: KIMI_PEER is set but KIMI_CHANNEL is not")
        deadline = time.time() + 1800
        while not (os.path.exists(CHANNEL) and os.path.getsize(CHANNEL) == SIZE):
            if time.time() > deadline:
                raise RuntimeError(f"KIMI_PACK: no sidecar channel at {CHANNEL} after 30 min")
            time.sleep(2)
        ch = GB300Side(CHANNEL, dev)
        while int(ch.cpu["hdr"][READY]) != READY_MAGIC:
            if time.time() > deadline:
                raise RuntimeError("KIMI_PACK: the sidecar never marked the channel ready")
            time.sleep(2)
        STATE["channel"] = ch

        def watchdog():
            while True:
                bad = int(ch.cpu["hdr"][ERR])
                if bad:
                    print(f"KIMI_PACK FATAL: the sidecar did not answer request {bad}; stopping", flush=True)
                    os._exit(1)
                time.sleep(1)

        threading.Thread(target=watchdog, daemon=True, name="kimi-peer-watchdog").start()

    def _bind(runner) -> None:
        from kpack import GROUP, layout, mmap_experts
        from kmoe import FAR, table_entries
        from vllm.utils.torch_utils import direct_register_custom_op, get_accelerator_view_from_cpu_tensor

        mod = sys.modules[MODEL]
        dev = runner.device
        if torch.cuda.get_device_capability(dev) != (10, 3):
            raise RuntimeError(f"KIMI_PACK: vLLM is on {torch.cuda.get_device_name(dev)}, expected the GB300 (10.3)")
        mods = {m.layer_idx: m for m in runner.model.modules() if isinstance(m, mod.KimiMoE)}
        if not mods:
            raise RuntimeError("KIMI_PACK: the loaded model has no KimiMoE layers")
        rank = {int(k): v for k, v in json.load(open(RANK))["layers"].items()}
        cfg = runner.vllm_config.model_config.hf_text_config
        E = cfg.num_experts
        n_host = E - HOT - PEER - PRUNE
        if min(HOT, PEER, PRUNE, n_host) < 0:
            raise RuntimeError(f"KIMI_PACK: KIMI_HOT + KIMI_PEER + KIMI_PRUNE exceed the layer's {E} experts")
        dirs = PACK.split(",")
        if PEER:
            _connect(dev)
        pool = ThreadPoolExecutor(16)
        direct_register_custom_op("kimi_packed_moe", _packed_moe, mutates_args=[], fake_impl=_packed_moe_fake)
        t0 = time.time()
        lay = None
        for layer, m in sorted(mods.items()):
            path = next((p for p in (os.path.join(d, f"layer-{layer:02d}.safetensors") for d in dirs)
                         if os.path.exists(p)), None)
            if path is None:
                raise RuntimeError(f"KIMI_PACK: no layer-{layer:02d}.safetensors in {PACK}")
            src, md = mmap_experts(path)
            this = layout(int(md["inter"]), int(md["latent"]))
            if int(md["group"]) != GROUP or src.shape != (E, this["size"]) or (lay and this != lay):
                raise RuntimeError(f"KIMI_PACK: {path} is not {E} experts in this hook's packed layout")
            if (this["w1"]["out"], this["w1"]["inn"]) != (cfg.moe_intermediate_size, cfg.routed_expert_hidden_size):
                raise RuntimeError(f"KIMI_PACK: {path} was packed for another expert shape")
            lay = this
            order = rank[layer]
            if sorted(order) != list(range(E)):
                raise RuntimeError(f"KIMI_PACK: {RANK} layer {layer} is not a ranking of {E} experts")
            peer_ids = order[PEER_FIRST:PEER_FIRST + PEER]
            rest = order[:PEER_FIRST] + order[PEER_FIRST + PEER:]
            hot_ids = sorted(rest[:HOT])
            host_ids = sorted(rest[HOT:len(rest) - PRUNE])
            pruned = rest[len(rest) - PRUNE:] if PRUNE else []
            hot_cpu = torch.empty((max(len(hot_ids), 1), lay["size"]), dtype=torch.uint8)
            # anonymous memory registered with CUDA after it is filled: torch's pinned allocator would round each
            # layer's block up to a power of two
            host_buf = mmap.mmap(-1, max(len(host_ids), 1) * lay["size"])
            host_cpu = torch.frombuffer(host_buf, dtype=torch.uint8).view(-1, lay["size"])

            def copy(job):
                dst, row, e = job
                np.copyto(dst[row], src[e])

            jobs = [(hot_cpu.numpy(), i, e) for i, e in enumerate(hot_ids)]
            jobs += [(host_cpu.numpy(), i, e) for i, e in enumerate(host_ids)]
            list(pool.map(copy, jobs))
            hot = hot_cpu.to(dev)
            del hot_cpu
            if int(torch.cuda.cudart().cudaHostRegister(host_cpu.data_ptr(), host_cpu.numel(), 0x3)) != 0:
                raise RuntimeError(f"KIMI_PACK: could not pin layer {layer}'s {host_cpu.numel() / 2**30:.1f} GiB "
                                   f"of experts in host memory")
            host = get_accelerator_view_from_cpu_tensor(host_cpu)
            table = torch.full((E,), FAR, dtype=torch.int64)
            own = torch.full((E,), -1, dtype=torch.int32)
            if hot_ids:
                table[hot_ids] = table_entries(torch.arange(len(hot_ids)) * lay["size"])
            if host_ids:
                table[host_ids] = table_entries(host.data_ptr() - hot.data_ptr()
                                                + torch.arange(len(host_ids)) * lay["size"])
            mine = hot_ids + host_ids
            own[mine] = torch.tensor(mine, dtype=torch.int32)
            everyone = own.clone()
            everyone[peer_ids] = 1
            if pruned:
                m.gate.e_score_correction_bias.data[pruned] = -1e9
            REG[layer] = dict(hot=hot, host=host, host_buf=host_buf, table=table.to(dev), own=own.to(dev),
                              all=everyone.to(dev))
            m.experts.layer = layer
            if layer == min(mods):
                STATE.update(lay=lay, beta=float(cfg.activation_situ_beta),
                             linear_beta=float(cfg.activation_situ_linear_beta))
                _selfcheck(m, layer, src, dev)
            m.use_mega_moe = False            # loaded; vLLM's MegaMoE warm-up must not look for its kernels here
            del src
            if layer % 10 == 0:
                print(f"KIMI_PACK: layer {layer} placed, {time.time() - t0:.0f} s", flush=True)
        pool.shutdown()
        gib = len(REG) * lay["size"] / 2**30
        print(f"KIMI_PACK ACTIVE: {len(REG)} MoE layers, per layer {HOT} experts in HBM ({HOT * gib:.1f} GiB), "
              f"{n_host} in Grace memory ({n_host * gib:.1f} GiB), {PEER} on the sidecar (ranks from {PEER_FIRST}), "
              f"{PRUNE} pruned; "
              f"placed in {time.time() - t0:.0f} s", flush=True)

    def _patch_runner(module) -> None:
        cls = module.GPUModelRunner
        orig = cls.load_model

        def load_model(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            _bind(self)
            return result

        cls.load_model = load_model

    PATCHES = {MODEL: _patch_model, **{r: _patch_runner for r in RUNNERS}}

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in PATCHES:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"KIMI_PACK: cannot find {name}")
            exec_module = spec.loader.exec_module

            def wrapped(module):
                exec_module(module)
                PATCHES[name](module)

            spec.loader.exec_module = wrapped
            return spec

    for _name, _fn in PATCHES.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _Hook())
