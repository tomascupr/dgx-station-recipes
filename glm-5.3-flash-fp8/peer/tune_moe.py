#!/usr/bin/env python3
"""Tune experts_fp8's Triton tiles for one GPU and write the config file vLLM looks up by name.

    tune_moe.py MODEL_DIR ROWMAP --side hot|cold OUT_DIR [--layer 20]

For each token count vLLM captures, times every candidate tile (BLOCK_SIZE_M x N x K=128, GROUP_SIZE_M, warps,
stages) on one real layer's experts for this side, each as a replayed CUDA graph (serving runs in graphs, so Python
launch cost must not pick the tile), with routes drawn so this side owns its measured share (hot 83 %, cold 17 %),
and keeps the fastest. Writes OUT_DIR/E=<n>,N=2048,device_name=<gpu>,dtype=fp8_w8a8,block_shape=[128,128].json,
where <n> is this side's expert count per layer, which is what its kernel sees in serving. Point
VLLM_TUNED_CONFIG_FOLDER at OUT_DIR to use it.
"""
import argparse
import itertools
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microbench import CLAMP, E, H, SCALE, TOPK, load_layer  # noqa: E402
from peer_moe import experts_fp8  # noqa: E402

MS = (1, 2, 4, 6, 8, 12, 16, 18, 24, 30, 36, 48, 60, 72, 96, 120, 144, 192, 256, 512, 1024, 2048, 4096, 8192)
TILES = [dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128, GROUP_SIZE_M=g, num_warps=w, num_stages=s)
         for bm, bn, g, w, s in itertools.product((16, 32, 64), (64, 128), (1, 16, 64), (4, 8), (2, 3, 4))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("rowmap")
    p.add_argument("--side", choices=("hot", "cold"), required=True)
    p.add_argument("out")
    p.add_argument("--layer", type=int, default=20)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    rm = json.load(open(a.rowmap))["layers"]
    ids = rm[str(a.layer)][a.side]
    served_e = len(ids)
    w1, s1, w2, s2 = load_layer(a.model, a.layer)
    sel = torch.tensor(ids)
    W = [t[sel].contiguous().to(dev) for t in (w1, s1, w2, s2)]
    emap = torch.full((E,), -1, dtype=torch.int32)
    emap[sel] = torch.arange(len(ids), dtype=torch.int32)
    emap = emap.to(dev)
    best = {}
    for m in MS:
        torch.manual_seed(m)
        x = (torch.randn(m, H, device=dev) * 0.5).to(torch.bfloat16)
        other = [e for e in range(E) if e not in set(ids)]
        share = 0.83 if a.side == "hot" else 0.17
        rows = []
        for _ in range(m):
            k = int(torch.binomial(torch.tensor(float(TOPK)), torch.tensor(share)))
            rows.append([ids[i] for i in torch.randperm(len(ids))[:k]] + [other[i] for i in torch.randperm(len(other))[:TOPK - k]])
        ti = torch.tensor(rows, dtype=torch.int32, device=dev)
        tw = torch.rand(m, TOPK, device=dev)
        tw = tw / tw.sum(1, keepdim=True) * SCALE
        n = 50 if m <= 256 else 10

        def graph_us(cfg):
            fn = lambda: experts_fp8(x, W[0], W[2], W[1], W[3], tw, ti, emap, CLAMP, config=cfg)  # noqa: E731
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            g.replay()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                g.replay()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e6

        results = []
        for cfg in TILES:
            try:
                results.append((graph_us(cfg), cfg))
            except Exception as exc:  # a tile that does not compile or fit is skipped, and said so once per M
                results.append((float("inf"), dict(cfg, error=str(exc)[:80])))
        results.sort(key=lambda r: r[0])
        best[str(m)] = {k: v for k, v in results[0][1].items() if k != "error"}
        dflt = graph_us(None)  # vLLM's own choice (or the file already in VLLM_TUNED_CONFIG_FOLDER), for comparison
        print(f"M={m}: best {results[0][0]:.0f} us {best[str(m)]} | default {dflt:.0f} us | "
              f"{sum(r[0] == float('inf') for r in results)} tiles failed", flush=True)
    name = (f"E={served_e},N=2048,device_name={torch.cuda.get_device_name(dev).replace(' ', '_')},"
            f"dtype=fp8_w8a8,block_shape=[128,128].json")
    os.makedirs(a.out, exist_ok=True)
    json.dump(best, open(os.path.join(a.out, name), "w"), indent=1)
    print("wrote", os.path.join(a.out, name))


if __name__ == "__main__":
    main()
