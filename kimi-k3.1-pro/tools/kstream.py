#!/usr/bin/env python3
"""Run Kimi K3.1 Pro's text model one layer at a time, from disk, on one GPU.

The checkpoint (1,454 GiB) is larger than the Station's memory, so no process can hold the model. One decoder layer is
~16 GiB, though, and a layer only needs the previous layer's output. This script keeps the hidden states of a fixed
set of token sequences in host memory and passes them through layer 0, then layer 1, and so on, loading each layer
once. Several "streams" share that one pass over the disk: stream 0 is the model as published, the others are
compressed variants of it (experts pruned or re-quantized, see variants.py). The result is, per stream:

  * counts[layer, expert]      how many tokens were routed to each expert
  * wsum[layer, expert]        the sum of their routing weights
  * final hidden states        from which metrics.py computes the divergence from stream 0 at the logits

Everything except the routed experts runs through Moonshot's reference modeling code (modeling_kimi_linear.py, the
pinned Kimi K3 reference): sequences here are at most 2,048 tokens, where K3.1's sparse attention (top 2,048 tokens)
selects every token and equals the dense MLA that code implements. The routed experts are held as stacked tensors,
decoded from the checkpoint's packed MXFP4. `--selftest` checks this file against the reference's own full forward.

    kstream.py MODEL_DIR TOKENS.pt OUT_DIR --ref REF_DIR [--variants SPEC.json] [--layers N] [--batch 8]
"""
import argparse
import importlib
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
PREFIX = "language_model.model."


def decode_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """packed [out, in/2] uint8 (low nibble = even column), scale [out, in/32] uint8 E8M0 -> weight [out, in]."""
    lut = FP4.to(packed.device, dtype)
    p = packed.long()
    w = torch.stack((lut[p & 15], lut[p >> 4]), dim=-1).flatten(-2)
    s = torch.pow(2.0, scale.float() - 127).to(dtype)
    return (w.view(*scale.shape, 32) * s.unsqueeze(-1)).view(w.shape)


class Checkpoint:
    """Tensor access by name over a sharded (or single-file) safetensors checkpoint."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self.by_layer = {}
        for k in self.map:
            m = re.match(re.escape(PREFIX) + r"layers\.(\d+)\.", k)
            if m:
                self.by_layer.setdefault(int(m.group(1)), []).append(k)

    def get(self, names, device) -> dict:
        out, files = {}, {}
        for n in names:
            files.setdefault(self.map[n], []).append(n)
        for fn, ns in files.items():
            with safe_open(os.path.join(self.dir, fn), "pt", device=str(device)) as f:
                for n in ns:
                    out[n] = f.get_tensor(n)
        return out


class StreamMoE(torch.nn.Module):
    """The reference KimiSparseMoeBlock with the routed experts as stacked tensors set from outside."""

    def __init__(self, config, ref):
        super().__init__()
        self.top_k, self.E = config.num_experts_per_token, config.num_experts
        self.scaling, self.renorm = config.routed_scaling_factor, config.moe_renormalize
        assert config.moe_router_activation_func == "sigmoid" and getattr(config, "num_expert_group", 1) == 1
        self.gate = ref.KimiMoEGate(config)
        self.shared_experts = ref.KimiMLP(config=config,
                                          intermediate_size=config.moe_intermediate_size * config.num_shared_experts)
        d = config.routed_expert_hidden_size
        self.routed_expert_down_proj = torch.nn.Linear(config.hidden_size, d, bias=False)
        self.routed_expert_up_proj = torch.nn.Linear(d, config.hidden_size, bias=False)
        assert config.latent_moe_use_norm
        self.routed_expert_norm = ref.KimiRMSNorm(d, eps=config.rms_norm_eps)
        beta, linear_beta = ref._get_situ_activation_params(config)
        self.act = ref.SituAndMul(beta=beta, linear_beta=linear_beta)
        self.sets = None                        # treatment -> (w1 [E, I, D], w2 [E, D, I], w3 [E, I, D])
        self.treat = None                       # per expert: the treatment whose weights it uses
        self.dead = None                        # bool [E]: experts removed from routing
        self.counts = self.wsum = None
        self.trace = None                       # list collecting each batch's routing choice, int16 [tokens, top_k]

    def route(self, x):
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        choice = scores + self.gate.e_score_correction_bias.float().unsqueeze(0)
        if self.dead is not None:
            choice = choice.masked_fill(self.dead.unsqueeze(0), float("-inf"))
        idx = torch.topk(choice, k=self.top_k, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if self.top_k > 1 and self.renorm:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w * self.scaling

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        idx, w = self.route(x)
        if self.trace is not None:
            self.trace.append(idx.to(torch.int16).cpu())
        self.counts.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.int64, device=x.device))
        self.wsum.index_add_(0, idx.reshape(-1), w.reshape(-1).double())
        h = self.routed_expert_down_proj(x)
        flat = idx.reshape(-1)
        order = flat.argsort()
        tok = order // self.top_k
        bounds = torch.searchsorted(flat[order], torch.arange(self.E + 1, device=x.device)).tolist()
        y = torch.zeros(h.shape, dtype=torch.float32, device=x.device)
        wflat = w.reshape(-1)
        for e in range(self.E):
            a, b = bounds[e], bounds[e + 1]
            if a == b:
                continue
            t = tok[a:b]
            z = h[t]
            w1, w2, w3 = self.sets[self.treat[e]]
            z = self.act(torch.cat((z @ w1[e].T, z @ w3[e].T), dim=-1)) @ w2[e].T
            y.index_add_(0, t, z.float() * wflat[order[a:b]].unsqueeze(-1))
        y = self.routed_expert_up_proj(self.routed_expert_norm(y.to(h.dtype)))
        return y.view(shape) + self.shared_experts(hidden_states)


def build_layer(config, ref, i: int):
    moe = i >= config.first_k_dense_replace
    orig = ref.KimiSparseMoeBlock
    ref.KimiSparseMoeBlock = lambda c: StreamMoE(c, ref)
    try:
        layer = ref.KimiDecoderLayer(config, i)
    finally:
        ref.KimiSparseMoeBlock = orig
    assert moe == hasattr(layer, "block_sparse_moe")
    return layer.eval()


def load_layer(layer, ckpt: Checkpoint, i: int, device, dtype):
    """Load layer i. Returns the expert stacks (w1, w2, w3) or None for the dense layer."""
    pre = f"{PREFIX}layers.{i}."
    names = ckpt.by_layer[i]
    exp = [n for n in names if ".block_sparse_moe.experts." in n]
    rest = [n for n in names if n not in set(exp) and ".self_attn.indexer." not in n]
    sd = {n[len(pre):]: t for n, t in ckpt.get(rest, device).items()}
    want = dict(layer.named_parameters())
    missing = sorted(set(want) - set(sd))
    extra = sorted(set(sd) - set(want))
    if missing or extra:
        raise RuntimeError(f"layer {i}: missing {missing[:5]} unexpected {extra[:5]}")
    for k, p in want.items():
        t = sd[k]
        if t.shape != p.shape:
            raise RuntimeError(f"layer {i}: {k} is {tuple(t.shape)}, expected {tuple(p.shape)}")
        p.data = t.to(dtype)   # as the reference is loaded (from_pretrained(dtype=...)); it calls .float() where needed
    if not exp:
        return None
    E = layer.block_sparse_moe.E
    stacks = []
    for wname in ("w1", "w2", "w3"):
        per = []
        for lo in range(0, E, 64):      # bound the transient uint8/long memory of the decode
            ns = [f"{pre}block_sparse_moe.experts.{e}.{wname}.weight{sfx}"
                  for e in range(lo, min(E, lo + 64)) for sfx in ("_packed", "_scale")]
            t = ckpt.get(ns, device)
            for e in range(lo, min(E, lo + 64)):
                b = f"{pre}block_sparse_moe.experts.{e}.{wname}.weight"
                per.append(decode_mxfp4(t[b + "_packed"], t[b + "_scale"], dtype))
        stacks.append(torch.stack(per))
    return tuple(stacks)


def fp8_copy(layer):
    """A copy of the layer with every weight matrix outside the routed experts and the router rounded to FP8
    (E4M3, one scale per tensor, as an online FP8 conversion would store it)."""
    import copy
    q = copy.deepcopy(layer)
    for n, p in q.named_parameters():
        if p.ndim == 2 and ".gate." not in n:
            s = p.data.float().abs().max() / 448
            p.data = ((p.data.float() / s).to(torch.float8_e4m3fn).float() * s).to(p.dtype)
    return q


def causal_mask(T: int, device, dtype):
    return torch.full((T, T), torch.finfo(dtype).min, device=device, dtype=dtype).triu(1)[None, None]


@torch.no_grad()
def run(model_dir, tokens, out_dir, ref_dir, variants=None, n_layers=None, batch=8, device="cuda",
        dtype=torch.bfloat16, keep=False, trace=False):
    """tokens: int64 [N, T]. Returns the final normed hidden states per stream, [S, N, T, H] on the CPU."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(ref_dir)))
    pkg = os.path.basename(os.path.abspath(ref_dir))
    ref = importlib.import_module(pkg + ".modeling_kimi_linear")
    cfgmod = importlib.import_module(pkg + ".configuration_kimi_k3")
    config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(model_dir, "config.json")))["text_config"])
    config._attn_implementation = "eager"
    ckpt = Checkpoint(model_dir)
    L = n_layers or config.num_hidden_layers
    N, T = tokens.shape
    H, E = config.hidden_size, config.num_experts
    assert T <= config.dsa_config["index_topk"], "sparse attention would no longer equal dense MLA"
    from variants import make_variants, requant        # noqa: E402  (beside this file)
    streams = make_variants(variants, config)     # [0] is the published model
    S = len(streams)
    os.makedirs(out_dir, exist_ok=True)

    emb = ckpt.get([PREFIX + "embed_tokens.weight"], "cpu")[PREFIX + "embed_tokens.weight"]
    h0 = emb[tokens].to(dtype)
    del emb
    hidden = [h0.clone() for _ in range(S)]          # CPU [N, T, H]
    blocks = [[] for _ in range(S)]                  # per stream: list of CPU [N, T, H]
    counts = torch.zeros((S, L, E), dtype=torch.int64)
    wsum = torch.zeros((S, L, E), dtype=torch.float64)
    route = torch.zeros((L, N * T, config.num_experts_per_token), dtype=torch.int16) if trace else None
    drift = torch.zeros((S, L))                      # relative L2 distance of each stream's output from stream 0
    mask = causal_mask(T, device, dtype)
    t00 = time.time()
    for i in range(L):
        t0 = time.time()
        layer = build_layer(config, ref, i).to(device)
        native = load_layer(layer, ckpt, i, device, dtype)
        t_load = time.time() - t0
        is_block = i % config.attn_res_block_size == 0
        layer_q = fp8_copy(layer) if any(v.nonexpert == "fp8" for v in streams) else None   # before the experts attach
        if native is not None:
            moe = layer.block_sparse_moe
            plans = [var.treatments(i, E) for var in streams]
            moe.sets = {"native": native}
            for name in sorted({t for p in plans for t in p} - {"native", "prune"}):
                moe.sets[name] = tuple(requant(w, name) for w in native)
        t_q = time.time() - t0 - t_load
        base_layer, base_moe = layer, (layer.block_sparse_moe if native is not None else None)
        for s, var in enumerate(streams):
            layer = layer_q if var.nonexpert == "fp8" else base_layer
            if native is not None:
                moe = layer.block_sparse_moe
                moe.sets = base_moe.sets
                moe.treat = plans[s]
                dead = torch.tensor([t == "prune" for t in plans[s]], device=device)
                moe.dead = dead if bool(dead.any()) else None
                moe.counts = torch.zeros(E, dtype=torch.int64, device=device)
                moe.wsum = torch.zeros(E, dtype=torch.float64, device=device)
                moe.trace = [] if trace and s == 0 else None
            new_h = torch.empty_like(hidden[s])
            new_block = torch.empty_like(hidden[s]) if is_block else None
            for a in range(0, N, batch):
                x = hidden[s][a:a + batch].to(device)
                B = x.shape[0]
                if blocks[s]:
                    br = torch.stack([b[a:a + batch].to(device).view(B * T, H) for b in blocks[s]], dim=1)
                else:
                    br = x.new_zeros(B * T, 0, H)
                y, br2 = layer(x, attention_mask=None if layer.is_linear_attn else mask, block_residual=br)
                new_h[a:a + batch] = y.cpu()
                if is_block:
                    new_block[a:a + batch] = br2[:, -1].view(B, T, H).cpu()
            hidden[s] = new_h
            if is_block:
                blocks[s].append(new_block)
            if native is not None:
                counts[s, i], wsum[s, i] = moe.counts.cpu(), moe.wsum.cpu()
                if moe.trace is not None:
                    route[i] = torch.cat(moe.trace)
                    moe.trace = None
            if s:
                drift[s, i] = ((hidden[s].float() - hidden[0].float()).norm() / hidden[0].float().norm()).item()
        if native is not None:
            base_moe.sets = moe.sets = None
        del layer, base_layer, layer_q, native
        torch.cuda.empty_cache()
        print(f"layer {i:2d} load {t_load:5.1f}s requant {t_q:5.1f}s total {time.time() - t0:6.1f}s "
              f"drift {' '.join(f'{d:.4f}' for d in drift[1:, i].tolist())}", flush=True)
        torch.save(dict(layer=i, counts=counts, wsum=wsum, drift=drift, names=[v.name for v in streams],
                        tokens=N * T), os.path.join(out_dir, "stats.pt"))

    final = None
    if L == config.num_hidden_layers:
        g = ckpt.get([PREFIX + n for n in ("output_attn_res_norm.weight", "output_attn_res_proj.weight",
                                           "norm.weight")], device)
        res_norm = ref.KimiRMSNorm(H, eps=config.rms_norm_eps).to(device)
        res_norm.weight.data = g[PREFIX + "output_attn_res_norm.weight"].to(dtype)
        proj = torch.nn.Linear(H, 1, bias=False).to(device)
        proj.weight.data = g[PREFIX + "output_attn_res_proj.weight"].to(dtype)
        norm = ref.KimiRMSNorm(H, eps=config.rms_norm_eps).to(device)
        norm.weight.data = g[PREFIX + "norm.weight"].to(dtype)
        final = torch.empty((S, N, T, H), dtype=dtype)
        for s in range(S):
            for a in range(0, N, batch):
                x = hidden[s][a:a + batch].to(device)
                B = x.shape[0]
                br = torch.stack([b[a:a + batch].to(device).view(B * T, H) for b in blocks[s]], dim=1)
                y = ref._apply_attn_res(x.view(B * T, H), br, proj, res_norm).view(B, T, H)
                final[s, a:a + batch] = norm(y).cpu()
        torch.save(dict(final=final, tokens=tokens, names=[v.name for v in streams]),
                   os.path.join(out_dir, "final.pt"))
    if trace:       # the published model's routing, token by token: [L, N*T, top_k] expert ids (layer 0 is dense: zeros)
        torch.save(dict(route=route, shape=(N, T)), os.path.join(out_dir, "route.pt"))
    print(f"done: {L} layers, {S} streams, {N * T} tokens in {time.time() - t00:.0f}s", flush=True)
    return final, counts


@torch.no_grad()
def selftest(model_dir, ref_dir, device="cuda", dtype=torch.bfloat16):
    """Compare this file's layer-at-a-time pass with the reference model's own forward, on a tiny checkpoint."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(ref_dir)))
    pkg = os.path.basename(os.path.abspath(ref_dir))
    ref = importlib.import_module(pkg + ".modeling_kimi_linear")
    cfgmod = importlib.import_module(pkg + ".configuration_kimi_k3")
    config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(model_dir, "config.json")))["text_config"])
    config._attn_implementation = "eager"
    ckpt = Checkpoint(model_dir)
    raw = ckpt.get([k for k in ckpt.map if k.startswith("language_model.")], device)
    sd = {}
    for k, t in raw.items():
        k = k[len("language_model."):]
        if ".self_attn.indexer." in k or k.endswith(".weight_scale"):
            continue
        if k.endswith(".weight_packed"):
            base = "language_model." + k[: -len("_packed")]
            sd[k[: -len("_packed")]] = decode_mxfp4(t, raw[base + "_scale"], dtype)
        else:
            sd[k] = t.to(dtype)
    model = ref.KimiLinearForCausalLM(config)
    res = model.load_state_dict(sd, strict=False, assign=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    model = model.to(device).eval()
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, config.vocab_size, (6, 96), generator=g)
    want = model.model(input_ids=tokens.to(device), use_cache=False).last_hidden_state.float().cpu()
    got, counts = run(model_dir, tokens, "/tmp/kstream-selftest", ref_dir, batch=4, device=device, dtype=dtype)
    got = got[0].float()
    rel = ((got - want).norm() / want.norm()).item()
    lw, lg = model.lm_head(want.to(device, dtype)).float(), model.lm_head(got.to(device, dtype)).float()
    agree = (lw.argmax(-1) == lg.argmax(-1)).float().mean().item()
    print(f"selftest: relative L2 difference of the final hidden states {rel:.2e}, top-1 agreement {agree:.4f}, "
          f"routed {int(counts[0].sum())} = {tokens.numel()} x {config.num_experts_per_token} x "
          f"{config.num_hidden_layers - config.first_k_dense_replace}")
    assert int(counts[0].sum()) == tokens.numel() * config.num_experts_per_token * (
        config.num_hidden_layers - config.first_k_dense_replace)
    # decode check: the packing in the tiny model's own creation script, inverted
    w = torch.randn(8, 64) * 3
    blk = w.view(8, 2, 32)
    ex = torch.round(torch.log2(blk.abs().amax(-1) / 6)).clamp(-126, 127)
    nrm = (blk / torch.pow(2.0, ex).unsqueeze(-1)).clamp(-6, 6)
    codes = torch.bucketize(nrm.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])) + nrm.signbit() * 8
    codes = codes.view(8, 64)
    dec = decode_mxfp4((codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8), (ex + 127).to(torch.uint8),
                       torch.float32)
    q = (FP4[codes.view(8, 2, 32)] * torch.pow(2.0, ex).unsqueeze(-1)).view(8, 64)
    assert torch.equal(dec, q), "MXFP4 decode does not invert the packing"
    assert rel < 2e-2 and agree > 0.97, "layer-at-a-time pass disagrees with the reference forward"
    print("selftest: PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir")
    p.add_argument("tokens", nargs="?")
    p.add_argument("out_dir", nargs="?")
    p.add_argument("--ref", required=True)
    p.add_argument("--variants")
    p.add_argument("--layers", type=int)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--trace", action="store_true", help="also save route.pt: the published model's routing per token")
    a = p.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.selftest:
        selftest(a.model_dir, a.ref)
    else:
        run(a.model_dir, torch.load(a.tokens), a.out_dir, a.ref,
            json.load(open(a.variants)) if a.variants else None, a.layers, a.batch, trace=a.trace)
