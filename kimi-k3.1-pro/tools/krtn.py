#!/usr/bin/env python3
"""Round every routed expert of Kimi K3.1 Pro to kpack.py's packed 2-bit format, no calibration data needed.

    krtn.py MODEL_DIR OUT_DIR[,OUT_DIR2] [--split N] [--first A] [--last B] [--ebatch 32]

Each group of GROUP inputs of each output row gets the scale (among the format's 256 log-spaced values) that
minimises the group's squared rounding error; the weights round to the nearest of {-3a, -a, +a, +3a}. One file per
MoE layer, layer-NN.safetensors with tensor `experts` uint8 [E, BLOB] (as kgptq.py --save). Layers that already have
a file are skipped, so an interrupted run resumes.
"""
import argparse
import os
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kstream import PREFIX, Checkpoint, decode_mxfp4  # noqa: E402
from kpack import GROUP, scale_to_u8, to_blob, u8_to_scale  # noqa: E402

SHRINK = (1.0, 0.88, 0.77, 0.68, 0.6, 0.52, 0.45, 0.38)


def rtn(W: torch.Tensor, pow2: bool = False):
    """W [B, out, in] float32 -> (codes uint8 [B, out, in/4], scale uint8 [B, out, in/GROUP], squared error, norm).
    pow2: only scales that are powers of two, so the rounded weights are also exact MXFP4 values (kab.py)."""
    B, R, n = W.shape
    x = W.reshape(-1, GROUP)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    best_c = torch.zeros(x.shape, dtype=torch.uint8, device=W.device)
    best_u = torch.zeros((x.shape[0], 1), dtype=torch.uint8, device=W.device)
    best_e = torch.full((x.shape[0], 1), float("inf"), device=W.device)
    for sh in SHRINK:
        u = scale_to_u8(amax * (sh / 3))
        if pow2:
            u = ((u.float() / 8).round() * 8).clamp(0, 248).to(torch.uint8)
        a = u8_to_scale(u)
        c = ((x / a + 3) * 0.5).round_().clamp_(0, 3)
        e = ((2 * c - 3) * a - x).pow_(2).sum(-1, keepdim=True)
        take = e < best_e
        best_e = torch.where(take, e, best_e)
        best_u = torch.where(take, u, best_u)
        best_c = torch.where(take, c.to(torch.uint8), best_c)
    c = best_c.reshape(B, R, n // 4, 4)
    codes = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return codes, best_u.reshape(B, R, n // GROUP), best_e.sum().item(), x.pow(2).sum().item()


@torch.no_grad()
def run(model_dir, outs, split, first, last, ebatch, device="cuda"):
    ckpt = Checkpoint(model_dir)
    layers = sorted(i for i, ns in ckpt.by_layer.items() if any(".block_sparse_moe.experts." in n for n in ns))
    for i in layers:
        if i < first or (last is not None and i > last):
            continue
        out = os.path.join(outs[1] if len(outs) > 1 and split is not None and i >= split else outs[0],
                           f"layer-{i:02d}.safetensors")
        if os.path.exists(out):
            continue
        t0 = time.time()
        pre = f"{PREFIX}layers.{i}.block_sparse_moe.experts."
        E = 1 + max(int(n[len(pre):].split(".")[0]) for n in ckpt.by_layer[i] if n.startswith(pre))
        blobs, err, nrm, shape = [], 0.0, 0.0, None
        for lo in range(0, E, ebatch):
            es = range(lo, min(E, lo + ebatch))
            t = ckpt.get([f"{pre}{e}.{w}.weight{s}" for e in es for w in ("w1", "w2", "w3")
                          for s in ("_packed", "_scale")], device)
            parts = []
            for w in ("w1", "w2", "w3"):
                W = torch.stack([decode_mxfp4(t[f"{pre}{e}.{w}.weight_packed"], t[f"{pre}{e}.{w}.weight_scale"],
                                              torch.float32) for e in es])
                if w == "w1":
                    shape = W.shape[1:]
                codes, u, e2, n2 = rtn(W)
                err += e2; nrm += n2
                parts.append((codes, u))
            blobs.append(to_blob(parts).cpu())
            del t, parts
        save_file({"experts": torch.cat(blobs)}, out + ".tmp",
                  metadata={"group": str(GROUP), "inter": str(shape[0]), "latent": str(shape[1]), "method": "rtn"})
        os.replace(out + ".tmp", out)
        print(f"layer {i:2d}: {E} experts, relative weight error {(err / nrm) ** 0.5:.4f}, {time.time() - t0:.0f}s",
              flush=True)
    print("RTN DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir"); p.add_argument("out")
    p.add_argument("--split", type=int); p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int); p.add_argument("--ebatch", type=int, default=32)
    a = p.parse_args()
    run(a.model_dir, a.out.split(","), a.split, a.first, a.last, a.ebatch)
