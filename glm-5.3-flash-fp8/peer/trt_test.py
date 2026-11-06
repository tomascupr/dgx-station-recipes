#!/usr/bin/env python3
"""TRT-LLM hot experts against the Triton path on one real layer: same sum, and GPU time of each as a CUDA graph.

    trt_test.py MODEL_DIR ROWMAP [--layer 20]
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
from trtllm_hot import hot_experts, prepare  # noqa: E402


def graph_us(fn, n=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("rowmap")
    p.add_argument("--layer", type=int, default=20)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    split = json.load(open(a.rowmap))["layers"][str(a.layer)]
    hot, cold = split["hot"], split["cold"]
    w1, s1, w2, s2 = load_layer(a.model, a.layer)
    sel = torch.tensor(hot)
    W = [t[sel].contiguous().to(dev) for t in (w1, s1, w2, s2)]
    hmap = torch.full((E,), -1, dtype=torch.int32)
    hmap[sel] = torch.arange(len(hot), dtype=torch.int32)
    hmap = hmap.to(dev)
    st = prepare(hot, W[0], W[1], W[2], W[3], CLAMP)
    g = torch.Generator().manual_seed(0)
    for m in (1, 6, 12, 24, 48, 96, 192):
        rows = []
        for _ in range(m):
            k = int(torch.binomial(torch.tensor(float(TOPK)), torch.tensor(0.83), generator=g))
            rows.append([hot[i] for i in torch.randperm(len(hot), generator=g)[:k]] +
                        [cold[i] for i in torch.randperm(len(cold), generator=g)[:TOPK - k]])
        ti = torch.tensor(rows, dtype=torch.int32, device=dev)
        x = (torch.randn(m, H, device=dev) * 0.5).to(torch.bfloat16)
        tw = torch.rand(m, TOPK, device=dev)
        tw = (tw / tw.sum(1, keepdim=True) * SCALE).contiguous()
        t_tri, y_tri = graph_us(lambda: experts_fp8(x, W[0], W[2], W[1], W[3], tw, ti, hmap, CLAMP))
        t_trt, y_trt = graph_us(lambda: hot_experts(st, x, tw, ti))
        err = float((y_trt.float() - y_tri.float()).norm() / y_tri.float().norm())
        print(f"M={m}: Triton {t_tri:.0f} us, TRT-LLM {t_trt:.0f} us; |TRT - Triton| / |Triton| = {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
