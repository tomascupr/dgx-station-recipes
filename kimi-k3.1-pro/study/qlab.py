#!/usr/bin/env python3
"""Compare expert quantizers on one layer of Kimi K3.1 Pro with the layer's real inputs.

    qlab.py MODEL_DIR TOKENS.pt --ref REF_DIR --layer I [--rows 16] [--experts 32]

Runs the published model up to layer I on the first ROWS sequences (as kstream.py does), takes layer I's routing
and the tokens each expert receives, and for a sample of experts (spread over the usage ranking) compares each
candidate in quant2.py with the published expert:
  werr   relative squared-error norm of the weights, ||q - w|| / ||w||, over the expert's three matrices
  oerr   the same for the expert's output, ||f_q(h) - f(h)|| / ||f(h)||, on its tokens in the second half of the
         sequences; anything a candidate calibrates (Hessians, input importance) comes from the first half
  s/exp  seconds to quantize one expert (all three matrices) on this GPU
"""
import argparse
import importlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kstream import PREFIX, Checkpoint, build_layer, causal_mask, decode_mxfp4, load_layer  # noqa: E402
from quant2 import CANDIDATES, gptq, levels2  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("model_dir"); p.add_argument("tokens"); p.add_argument("--ref", required=True)
p.add_argument("--layer", type=int, required=True); p.add_argument("--rows", type=int, default=16)
p.add_argument("--experts", type=int, default=32); p.add_argument("--only", default="")
a = p.parse_args()
device, dtype = "cuda", torch.bfloat16
sys.path.insert(0, os.path.dirname(os.path.abspath(a.ref)))
pkg = os.path.basename(os.path.abspath(a.ref))
ref = importlib.import_module(pkg + ".modeling_kimi_linear")
cfgmod = importlib.import_module(pkg + ".configuration_kimi_k3")
config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"])
config._attn_implementation = "eager"
ckpt = Checkpoint(a.model_dir)
tokens = torch.load(a.tokens)[: a.rows]
N, T = tokens.shape
H, E, B = config.hidden_size, config.num_experts, 4

with torch.no_grad():
    emb = ckpt.get([PREFIX + "embed_tokens.weight"], "cpu")[PREFIX + "embed_tokens.weight"]
    hidden = emb[tokens].to(dtype)
    del emb
    blocks, mask, grabbed = [], causal_mask(T, device, dtype), []
    for i in range(a.layer + 1):
        layer = build_layer(config, ref, i).to(device)
        native = load_layer(layer, ckpt, i, device, dtype)
        if native is not None:
            moe = layer.block_sparse_moe
            moe.sets, moe.treat, moe.dead = {"native": native}, ["native"] * E, None
            moe.counts = torch.zeros(E, dtype=torch.int64, device=device)
            moe.wsum = torch.zeros(E, dtype=torch.float64, device=device)
        if i == a.layer:
            hook = moe.register_forward_pre_hook(lambda m, args: grabbed.append(args[0].reshape(-1, H)))
        is_block = i % config.attn_res_block_size == 0
        new_h = torch.empty_like(hidden)
        new_block = torch.empty_like(hidden) if is_block else None
        for s in range(0, N, B):
            x = hidden[s:s + B].to(device)
            n = x.shape[0]
            br = (torch.stack([b[s:s + B].to(device).view(n * T, H) for b in blocks], dim=1) if blocks
                  else x.new_zeros(n * T, 0, H))
            y, br2 = layer(x, attention_mask=None if layer.is_linear_attn else mask, block_residual=br)
            new_h[s:s + B] = y.cpu()
            if is_block:
                new_block[s:s + B] = br2[:, -1].view(n, T, H).cpu()
        hidden = new_h
        if is_block:
            blocks.append(new_block)
        if i < a.layer:
            if native is not None:
                moe.sets = None
                del moe
            del layer, native
            torch.cuda.empty_cache()
        print(f"layer {i} done", flush=True)

    x = torch.cat(grabbed)                                   # [N*T, H] input of layer I's MoE block
    idx, _ = moe.route(x)
    h = moe.routed_expert_down_proj(x).float()
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    order = counts.argsort(descending=True)
    sample = order[torch.linspace(0, E - 1, a.experts).long()].tolist()
    w1s, w2s, w3s = native

    def f(z, w1, w2, w3):
        return moe.act(torch.cat((z @ w1.T, z @ w3.T), dim=-1)).float() @ w2.T

    # Hessians for GPTQ: of the latent input h over all tokens of the layer (shared by every expert's w1 and w3),
    # and per expert of its own tokens; w2's input is the expert's activation, so its Hessian is per expert,
    # blended with the pool over the sampled experts because one expert sees few tokens.
    cal = torch.arange(h.shape[0], device=h.device) < h.shape[0] // 2      # calibration tokens; the rest score
    Hh = (h[cal].T @ h[cal]) / int(cal.sum())
    mids = {}
    for e in sample:
        t = ((idx == e).any(-1) & cal).nonzero()[:, 0]
        if t.numel():
            z = h[t]
            mids[e] = moe.act(torch.cat((z @ w1s[e].float().T, z @ w3s[e].float().T), dim=-1)).float()
    Hm = sum(m.T @ m for m in mids.values()) / sum(m.shape[0] for m in mids.values())

    def gp(how, mix, g=32):
        def fn(w, c, k, z, mid):
            if k == 1:
                He = mid.T @ mid / mid.shape[0]
                H = mix * He + (1 - mix) * Hm * (He.diagonal().mean() / Hm.diagonal().mean())
            else:
                He = z.T @ z / z.shape[0]
                H = mix * He + (1 - mix) * Hh * (He.diagonal().mean() / Hh.diagonal().mean())
            return gptq(w, H, *levels2(w, g, how), g=g)
        return fn

    EXTRA = {"q2-gptq-pooled": (2.25, gp("uniform", 0.0)), "q2-gptq-blend": (2.25, gp("uniform", 0.5)),
             "q2-gptq-own": (2.25, gp("uniform", 1.0)), "q2-gptq-lloyd-blend": (2.50, gp("lloyd", 0.5))}
    ALL = {n: (b, (lambda w, c, k, z, mid, f_=f_: f_(w, c))) for n, (b, f_) in CANDIDATES.items()}
    ALL.update(EXTRA)
    names = [n for n in ALL if not a.only or n in a.only.split(",")]
    acc = {n: dict(we=0.0, wn=0.0, oe=0.0, on=0.0, t=0.0) for n in names}
    for e in sample:
        t = ((idx == e).any(-1) & cal).nonzero()[:, 0]
        ts = ((idx == e).any(-1) & ~cal).nonzero()[:, 0]
        if t.numel() == 0 or ts.numel() == 0:
            continue
        z, zs = h[t], h[ts]
        w = [w1s[e].float(), w2s[e].float(), w3s[e].float()]
        mid = moe.act(torch.cat((z @ w[0].T, z @ w[2].T), dim=-1)).float()
        yn = f(zs, *w)
        colw = [z.pow(2).mean(0), mid.pow(2).mean(0), z.pow(2).mean(0)]
        for n in names:
            fn = ALL[n][1]
            torch.cuda.synchronize(); t0 = time.time()
            q = [fn(w[k], colw[k], k, z, mid) for k in range(3)]
            torch.cuda.synchronize(); acc[n]["t"] += time.time() - t0
            yq = f(zs, q[0], q[1], q[2])
            acc[n]["we"] += sum((q[k] - w[k]).pow(2).sum().item() for k in range(3))
            acc[n]["wn"] += sum(w[k].pow(2).sum().item() for k in range(3))
            acc[n]["oe"] += (yq - yn).pow(2).sum().item()
            acc[n]["on"] += yn.pow(2).sum().item()
    print(f"\nlayer {a.layer}: {len(sample)} experts sampled over the usage ranking, {N * T} tokens")
    print(f"{'candidate':22s} {'bits':>5s} {'werr':>7s} {'oerr':>7s} {'s/exp':>7s}")
    for n in names:
        r = acc[n]
        print(f"{n:22s} {ALL[n][0]:5.2f} {(r['we'] / r['wn']) ** .5:7.4f} {(r['oe'] / r['on']) ** .5:7.4f} "
              f"{r['t'] / len(sample):7.3f}")
