#!/usr/bin/env python3
"""Quantize every routed expert of Kimi K3.1 Pro to 2 bits with output-aware rounding (GPTQ), layer by layer,
and carry the quantized model's hidden states along to score it.

    kgptq.py MODEL_DIR TOKENS.pt OUT_DIR --ref REF_DIR [--cal-rows N/2] [--group 64] [--layers N]

As kstream.py: one pass over the checkpoint, one layer in memory at a time, two streams of hidden states (0 the
published model, 1 the quantized one). At each MoE layer:
  1. stream 0 runs through the published layer;
  2. stream 1's calibration sequences (the first CAL-ROWS) run up to the MoE block, giving the inputs and the
     routing the quantized model really has at this depth;
  3. each expert's three matrices are rounded to the codebook {-3s, -s, +s, +3s}/2 with one scale s per GROUP
     inputs (2 + 8/GROUP bits per weight), columns left to right, each column's error pushed onto the columns still
     to come through the inverse Hessian of the expert's own inputs (blended half and half with the layer-wide
     Hessian, since one expert sees few tokens); w2 is rounded against the activations of the already rounded
     w1 and w3;
  4. stream 1 runs through the layer with the rounded experts.
With --save the rounded experts are also written in kpack.py's packed format, one file per layer.
OUT_DIR gets final.pt and stats.pt in kstream.py's format; score with `metrics.py MODEL_DIR OUT_DIR --score-from
CAL-ROWS`, i.e. on the sequences the rounding never saw.
"""
import argparse
import importlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kstream import PREFIX, Checkpoint, build_layer, causal_mask, load_layer  # noqa: E402
from kpack import GROUP, pack, scale_to_u8, to_blob, u8_to_scale  # noqa: E402
from variants import _requant  # noqa: E402


class _Stop(Exception):
    pass


def gptq_batch(W, H, g, block=128, damp=0.01):
    """W [B, out, in] float32, H [B, in, in]: GPTQ rounding to 4 uniform levels, scale per `g` inputs.

    Returns Q and the scales as kpack.py stores them (uint8 [B, out, in/g]); Q holds exactly those levels."""
    B, R, n = W.shape
    q0 = _requant(W.reshape(B * R, n), 4, g).abs().reshape(B, R, n // g, g)
    a_u8 = scale_to_u8(q0.amin(-1))                  # the smaller magnitude; the larger is 3a
    a = u8_to_scale(a_u8)
    del q0
    W = W.clone()
    H = H.clone()
    d = H.diagonal(dim1=1, dim2=2)
    dead = d == 0
    d[dead] = 1
    W.masked_fill_(dead.unsqueeze(1), 0)
    d.add_(damp * d.mean(-1, keepdim=True))
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    del H
    Q = torch.empty_like(W)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Err = torch.empty((B, R, i1 - i0), device=W.device)
        for j in range(i0, i1):
            x = W[:, :, j]
            aj = a[:, :, j // g]
            q = torch.where(x.abs() > 2 * aj, 3 * aj, aj) * torch.where(x < 0, -1.0, 1.0)
            Q[:, :, j] = q
            e = (x - q) / U[:, j, j].unsqueeze(1)
            W[:, :, j + 1:i1] -= e.unsqueeze(2) * U[:, j, j + 1:i1].unsqueeze(1)
            Err[:, :, j - i0] = e
        W[:, :, i1:] -= Err @ U[:, i0:i1, i1:]
    return Q, a_u8


def blend(He, pool):
    """Half the expert's own Hessian, half the pooled one scaled to the same trace."""
    tr = He.diagonal(dim1=1, dim2=2).mean(-1)
    tp = pool.diagonal().mean()
    scale = torch.where(tr > 0, tr / tp, torch.ones_like(tr))
    mix = torch.where(tr > 0, 0.5, 0.0).view(-1, 1, 1)
    return mix * He + (1 - mix) * pool.unsqueeze(0) * scale.view(-1, 1, 1)


@torch.no_grad()
def quantize_layer(moe, native, x, g, ebatch=16, save=None):
    """Round all experts of one layer in place. x [tokens, H]: the MoE block's inputs on the calibration tokens."""
    w1s, w2s, w3s = native
    E = w1s.shape[0]
    idx, _ = moe.route(x)
    h = moe.routed_expert_down_proj(x).float()
    Hh = h.T @ h / h.shape[0]
    member = torch.zeros((h.shape[0], E), dtype=torch.bool, device=x.device).scatter_(1, idx, True)
    err = nrm = 0.0
    blobs = []
    for lo in range(0, E, ebatch):
        es = list(range(lo, min(E, lo + ebatch)))
        zs = [h[member[:, e].nonzero()[:, 0]] for e in es]
        He = torch.stack([z.T @ z / max(1, z.shape[0]) for z in zs])
        Hb = blend(He, Hh)
        del He
        q1, a1 = gptq_batch(w1s[lo:lo + ebatch].float(), Hb, g)
        q3, a3 = gptq_batch(w3s[lo:lo + ebatch].float(), Hb, g)
        del Hb
        mids = [moe.act(torch.cat((z @ q1[k].T, z @ q3[k].T), dim=-1)).float() for k, z in enumerate(zs)]
        Hm = torch.stack([m.T @ m / max(1, m.shape[0]) for m in mids])
        pool = sum(m.T @ m for m in mids) / max(1, sum(m.shape[0] for m in mids))
        q2, a2 = gptq_batch(w2s[lo:lo + ebatch].float(), blend(Hm, pool), g)
        if save:
            blobs.append(to_blob([(pack(q, au), au) for q, au in ((q1, a1), (q2, a2), (q3, a3))]).cpu())
        del Hm
        for k, e in enumerate(es):                  # in-sample output error, for the log
            if zs[k].shape[0]:
                yn = moe.act(torch.cat((zs[k] @ w1s[e].float().T, zs[k] @ w3s[e].float().T), dim=-1)).float() @ w2s[e].float().T
                err += (mids[k] @ q2[k].T - yn).pow(2).sum().item()
                nrm += yn.pow(2).sum().item()
        w1s[lo:lo + ebatch], w2s[lo:lo + ebatch], w3s[lo:lo + ebatch] = q1.to(w1s.dtype), q2.to(w2s.dtype), q3.to(w3s.dtype)
        del q1, q2, q3, mids, zs
    if save:
        from safetensors.torch import save_file
        save_file({"experts": torch.cat(blobs)}, save + ".tmp",
                  metadata={"group": str(g), "inter": str(w1s.shape[1]), "latent": str(w1s.shape[2])})
        os.replace(save + ".tmp", save)
    return (err / max(nrm, 1e-30)) ** 0.5


@torch.no_grad()
def run(model_dir, tokens, out_dir, ref_dir, cal_rows=None, group=64, n_layers=None, batch=4, device="cuda",
        dtype=torch.bfloat16, save=None, split=None, ebatch=32):
    sys.path.insert(0, os.path.dirname(os.path.abspath(ref_dir)))
    pkg = os.path.basename(os.path.abspath(ref_dir))
    ref = importlib.import_module(pkg + ".modeling_kimi_linear")
    cfgmod = importlib.import_module(pkg + ".configuration_kimi_k3")
    config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(model_dir, "config.json")))["text_config"])
    config._attn_implementation = "eager"
    ckpt = Checkpoint(model_dir)
    L = n_layers or config.num_hidden_layers
    N, T = tokens.shape
    H, E, S = config.hidden_size, config.num_experts, 2
    cal_rows = cal_rows or N // 2
    assert not save or group == GROUP, f"kpack.py's format has GROUP = {GROUP}"
    names = ["native", f"gptq-q2-g{group}"]
    os.makedirs(out_dir, exist_ok=True)
    emb = ckpt.get([PREFIX + "embed_tokens.weight"], "cpu")[PREFIX + "embed_tokens.weight"]
    h0 = emb[tokens].to(dtype)
    del emb
    hidden = [h0.clone() for _ in range(S)]
    blocks = [[] for _ in range(S)]
    counts = torch.zeros((S, L, E), dtype=torch.int64)
    wsum = torch.zeros((S, L, E), dtype=torch.float64)
    drift, oerr = torch.zeros((S, L)), torch.zeros(L)
    mask = causal_mask(T, device, dtype)
    t00 = time.time()

    def forward(layer, s, rows, store):
        new_h = torch.empty_like(hidden[s]) if store else None
        new_block = torch.empty_like(hidden[s]) if store and store_block else None
        for a in range(0, rows, batch):
            b = min(a + batch, rows)
            x = hidden[s][a:b].to(device)
            n = x.shape[0]
            br = (torch.stack([blk[a:b].to(device).view(n * T, H) for blk in blocks[s]], dim=1) if blocks[s]
                  else x.new_zeros(n * T, 0, H))
            try:
                y, br2 = layer(x, attention_mask=None if layer.is_linear_attn else mask, block_residual=br)
            except _Stop:
                continue
            if store:
                new_h[a:b] = y.cpu()
                if store_block:
                    new_block[a:b] = br2[:, -1].view(n, T, H).cpu()
        if store:
            hidden[s] = new_h
            if store_block:
                blocks[s].append(new_block)

    for i in range(L):
        t0 = time.time()
        layer = build_layer(config, ref, i).to(device)
        native = load_layer(layer, ckpt, i, device, dtype)
        store_block = i % config.attn_res_block_size == 0
        t_q = 0.0
        if native is not None:
            moe = layer.block_sparse_moe
            moe.sets, moe.treat, moe.dead = {"native": native}, ["native"] * E, None
        for s in range(S):
            if native is not None:
                if s == 1:
                    grabbed = []

                    def grab(m, args):
                        grabbed.append(args[0].reshape(-1, H))
                        raise _Stop

                    hook = moe.register_forward_pre_hook(grab)
                    forward(layer, 1, cal_rows, store=False)
                    hook.remove()
                    tq = time.time()
                    dest = None
                    if save:        # layers below `split` go to the first directory, the rest to the second
                        dirs = save.split(",")
                        dest = os.path.join(dirs[0] if split is None or i < split else dirs[1], f"layer-{i:02d}.safetensors")
                    oerr[i] = quantize_layer(moe, native, torch.cat(grabbed), group, ebatch, save=dest)
                    del grabbed
                    t_q = time.time() - tq
                moe.counts = torch.zeros(E, dtype=torch.int64, device=device)
                moe.wsum = torch.zeros(E, dtype=torch.float64, device=device)
            forward(layer, s, N, store=True)
            if native is not None:
                counts[s, i], wsum[s, i] = moe.counts.cpu(), moe.wsum.cpu()
        ev = slice(cal_rows, N)
        drift[1, i] = ((hidden[1][ev].float() - hidden[0][ev].float()).norm() / hidden[0][ev].float().norm()).item()
        if native is not None:
            moe.sets = None
            del moe
        del layer, native
        torch.cuda.empty_cache()
        print(f"layer {i:2d} quantize {t_q:6.1f}s total {time.time() - t0:6.1f}s in-sample expert output error "
              f"{oerr[i]:.4f} held-out drift {drift[1, i]:.4f}", flush=True)
        torch.save(dict(layer=i, counts=counts, wsum=wsum, drift=drift, oerr=oerr, names=names, tokens=N * T,
                        cal_rows=cal_rows), os.path.join(out_dir, "stats.pt"))

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
        torch.save(dict(final=final, tokens=tokens, names=names, cal_rows=cal_rows), os.path.join(out_dir, "final.pt"))
    print(f"done: {L} layers in {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir"); p.add_argument("tokens"); p.add_argument("out_dir")
    p.add_argument("--ref", required=True)
    p.add_argument("--cal-rows", type=int)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--layers", type=int)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--ebatch", type=int, default=32, help="experts rounded at once (memory against speed)")
    p.add_argument("--save", help="directory (or two, comma-separated) for the packed experts, one file per layer")
    p.add_argument("--split", type=int, help="with two --save directories: the first layer that goes to the second")
    a = p.parse_args()
    run(a.model_dir, torch.load(a.tokens), a.out_dir, a.ref, a.cal_rows, a.group, a.layers, a.batch, save=a.save,
        split=a.split, ebatch=a.ebatch)
