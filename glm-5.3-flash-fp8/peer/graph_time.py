#!/usr/bin/env python3
"""GPU time of experts_fp8 for one side of one real layer, as a replayed CUDA graph (no Python launch cost).

    graph_time.py MODEL_DIR ROWMAP --side hot|cold [--layer 20] [--owned 0.83]

Routes are drawn so this side owns about --owned of them (0.83 hot / 0.17 cold matches the rowmap's measured share).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microbench import CLAMP, E, H, SCALE, TOPK, load_layer  # noqa: E402
from peer_moe import experts_fp8  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("rowmap")
    p.add_argument("--side", choices=("hot", "cold"), required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--owned", type=float, default=None)
    a = p.parse_args()
    owned_share = a.owned if a.owned is not None else (0.83 if a.side == "hot" else 0.17)
    dev = torch.device("cuda", 0)
    split = json.load(open(a.rowmap))["layers"][str(a.layer)]
    ids, other = split[a.side], split["cold" if a.side == "hot" else "hot"]
    w1, s1, w2, s2 = load_layer(a.model, a.layer)
    sel = torch.tensor(ids)
    W = [t[sel].contiguous().to(dev) for t in (w1, s1, w2, s2)]
    emap = torch.full((E,), -1, dtype=torch.int32)
    emap[sel] = torch.arange(len(ids), dtype=torch.int32)
    emap = emap.to(dev)
    g_ = torch.Generator().manual_seed(0)
    for m in (1, 6, 12, 24, 48, 96, 192):
        rows = []
        for _ in range(m):
            k_own = int(torch.binomial(torch.tensor(float(TOPK)), torch.tensor(owned_share), generator=g_))
            pick = [ids[i] for i in torch.randperm(len(ids), generator=g_)[:k_own]] + \
                   [other[i] for i in torch.randperm(len(other), generator=g_)[:TOPK - k_own]]
            rows.append(pick)
        ti = torch.tensor(rows, dtype=torch.int32, device=dev)
        x = (torch.randn(m, H, device=dev) * 0.5).to(torch.bfloat16)
        tw = torch.rand(m, TOPK, device=dev)
        tw = tw / tw.sum(1, keepdim=True) * SCALE
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                experts_fp8(x, W[0], W[2], W[1], W[3], tw, ti, emap, CLAMP)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            experts_fp8(x, W[0], W[2], W[1], W[3], tw, ti, emap, CLAMP)
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 200 * 1e6
        distinct = len({int(e) for e in ti.flatten().tolist()} & set(ids))
        mb = distinct * 25_171_968 / 1e6
        print(f"{a.side} M={m}: {us:.0f} us per call as a graph; {distinct} distinct experts owned, {mb:.0f} MB of "
              f"weights -> {mb / us / 1e3:.2f} TB/s effective", flush=True)


if __name__ == "__main__":
    main()
