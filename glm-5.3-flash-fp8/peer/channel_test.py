#!/usr/bin/env python3
"""Check the channel against a running sidecar, from the GB300, without vLLM: exact, fast, and graph-capturable.

    channel_test.py MODEL_DIR ROWMAP CHANNEL [--layer 20]

For 6 and 60 tokens: hot experts here + the sidecar's cold sum must equal all 288 experts computed here; time one
split layer eagerly and as a replayed CUDA graph.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import READY, READY_MAGIC, GB300Side  # noqa: E402
from microbench import CLAMP, E, H, SCALE, TOPK, load_layer  # noqa: E402
from peer_moe import experts_fp8  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("rowmap")
    p.add_argument("channel")
    p.add_argument("--layer", type=int, default=20)
    a = p.parse_args()
    dev = torch.device("cuda", 0)
    split = json.load(open(a.rowmap))["layers"][str(a.layer)]
    w1, s1, w2, s2 = (t.to(dev) for t in load_layer(a.model, a.layer))
    hot = torch.tensor(split["hot"], device=dev)
    hmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
    hmap[hot] = torch.arange(len(split["hot"]), dtype=torch.int32, device=dev)
    amap = torch.arange(E, dtype=torch.int32, device=dev)
    hw = [t.index_select(0, hot) for t in (w1, s1, w2, s2)]
    ch = GB300Side(a.channel, dev)
    while int(ch.cpu["hdr"][READY]) != READY_MAGIC:
        time.sleep(1)
    for m in (6, 60):
        torch.manual_seed(m)
        x = (torch.randn(m, H, device=dev) * 0.5).to(torch.bfloat16)
        ti = torch.stack([torch.randperm(E)[:TOPK] for _ in range(m)]).to(dev, torch.int32)
        tw = torch.rand(m, TOPK, device=dev)
        tw = (tw / tw.sum(1, keepdim=True) * SCALE).contiguous()

        def layer():
            ch.send(x, ti, tw, a.layer)
            out = experts_fp8(x, hw[0], hw[2], hw[1], hw[3], tw, ti, hmap, CLAMP)
            ch.wait_add(out)
            return out

        whole = experts_fp8(x, w1, w2, s1, s2, tw, ti, amap, CLAMP).float()
        got = layer().float()
        torch.cuda.synchronize()
        err = float((got - whole).norm() / whole.norm())
        for _ in range(20):
            layer()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            layer()
        torch.cuda.synchronize()
        eager_us = (time.perf_counter() - t0) / 200 * 1e6
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            layer()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            gout = layer()
        g.replay()
        torch.cuda.synchronize()
        gerr = float((gout.float() - whole).norm() / whole.norm())
        t0 = time.perf_counter()
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
        graph_us = (time.perf_counter() - t0) / 200 * 1e6
        print(f"M={m}: split vs all-288 rel err {err:.2e} (graph replay {gerr:.2e}); split layer {eager_us:.0f} us "
              f"eager, {graph_us:.0f} us as a CUDA graph; error flag {int(ch.cpu['hdr'][48])}", flush=True)


if __name__ == "__main__":
    main()
