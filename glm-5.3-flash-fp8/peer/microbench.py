#!/usr/bin/env python3
"""Stage-5 go/no-go on one real MoE layer, no vLLM server: how fast each GPU computes its share, and is it exact?

    microbench.py MODEL_DIR ROWMAP --side hot|cold|all [--layer 20]

One GPU per process, as the Station's driver requires (a process sees the GB300 or the RTX, never both), so the
GB300 and the RTX exchange tokens through host memory. Run inside the vLLM image with one GPU visible. Measures
experts_fp8 on this side's experts at 6 and 60 tokens, the bf16 activation copy to and from pinned host memory,
and the error against a float32 dequantized reference over this side's routes.
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peer_moe import experts_fp8  # noqa: E402

E, TOPK, H, I, CLAMP, SCALE = 288, 8, 4096, 2048, 10.0, 2.5


def timed(fn, devs, n=100):
    for _ in range(10):
        fn()
    for d in devs:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    for d in devs:
        torch.cuda.synchronize(d)
    return (time.perf_counter() - t0) / n * 1e6


def load_layer(model: str, layer: int):
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts"
    files: dict = {}

    def get(name):
        f = idx[name]
        if f not in files:
            files[f] = safe_open(os.path.join(model, f), "pt")
        return files[f].get_tensor(name)

    w1, s1, w2, s2 = [], [], [], []
    for e in range(E):
        p = f"{pre}.{e}"
        w1.append(torch.cat([get(f"{p}.gate_proj.weight"), get(f"{p}.up_proj.weight")]))
        s1.append(torch.cat([get(f"{p}.gate_proj.weight_scale_inv"), get(f"{p}.up_proj.weight_scale_inv")]))
        w2.append(get(f"{p}.down_proj.weight"))
        s2.append(get(f"{p}.down_proj.weight_scale_inv"))
    return torch.stack(w1), torch.stack(s1), torch.stack(w2), torch.stack(s2)


def dequant(w, s):
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]


def reference(x, w1, s1, w2, s2, tw, ti):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for m in range(x.shape[0]):
        for j in range(ti.shape[1]):
            e = int(ti[m, j])
            h = dequant(w1[e], s1[e]) @ x[m].float()
            g, u = h[:I].clamp(max=CLAMP), h[I:].clamp(-CLAMP, CLAMP)
            out[m] += tw[m, j] * (dequant(w2[e], s2[e]) @ (torch.nn.functional.silu(g) * u))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("rowmap")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--side", choices=("hot", "cold", "all"), required=True,
                   help="which experts this process computes; one GPU per process (the Station's driver gives a "
                        "process either the GB300 or the RTX, not both)")
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    print(f"{a.side} on {torch.cuda.get_device_name(dev)} {torch.cuda.get_device_capability(dev)}")
    split = json.load(open(a.rowmap))["layers"][str(a.layer)]
    ids = split[a.side] if a.side != "all" else list(range(E))
    t0 = time.time()
    w1, s1, w2, s2 = load_layer(a.model, a.layer)
    m = torch.full((E,), -1, dtype=torch.int32)
    m[torch.tensor(ids)] = torch.arange(len(ids), dtype=torch.int32)
    sel = torch.tensor(ids)
    W = [t_[sel].contiguous().to(dev) for t_ in (w1, s1, w2, s2)] + [m.to(dev)]
    print(f"layer {a.layer}: {len(ids)} experts loaded in {time.time() - t0:.0f} s")
    owned = set(ids)
    for M in (6, 60):
        torch.manual_seed(M)
        x = torch.randn(M, H, dtype=torch.bfloat16, device=dev) * 0.5
        ti = torch.stack([torch.randperm(E)[:TOPK] for _ in range(M)]).to(dev, torch.int32)
        tw = torch.rand(M, TOPK, device=dev)
        tw = tw / tw.sum(1, keepdim=True) * SCALE
        frac = sum(int(e) in owned for e in ti.flatten().tolist()) / ti.numel()
        hb = torch.empty((M, H), dtype=torch.bfloat16, pin_memory=True)
        us = timed(lambda: experts_fp8(x, W[0], W[2], W[1], W[3], tw, ti, W[4], CLAMP), [dev])
        d2h = timed(lambda: hb.copy_(x, non_blocking=True), [dev])
        h2d = timed(lambda: x.copy_(hb, non_blocking=True), [dev])
        print(f"M={M}: {100 * frac:.0f}% of uniform-random routes owned | experts {us:.0f} us | "
              f"bf16 [M,4096] to pinned host {d2h:.0f} us, back {h2d:.0f} us")
        if M == 6:
            out = experts_fp8(x, W[0], W[2], W[1], W[3], tw, ti, W[4], CLAMP).float()
            mask = torch.tensor([[int(e) in owned for e in row] for row in ti.tolist()], device=dev)
            ref = reference(x, w1.to(dev), s1.to(dev), w2.to(dev), s2.to(dev), tw * mask, ti)
            print(f"M={M}: |out - fp32 reference| / |reference| = {float((out - ref).norm() / ref.norm()):.2e}")


if __name__ == "__main__":
    main()
