#!/usr/bin/env python3
"""The RTX sidecar: computes the cold experts of every target layer for the GB300's vLLM, through the channel.

    sidecar.py MODEL_DIR ROWMAP CHANNEL

Loads the rowmap's cold experts of every target layer from the checkpoint into one stacked set of tensors on the
RTX (layer-major), creates the channel file, marks itself ready, then serves requests in order. A request is the
channel's rows, ids and weights for one layer; the sidecar runs experts_fp8 on that layer's view of the stack (its
86 cold experts, with a 288-entry map: hot experts map to -1 and compute nothing; vLLM's alignment kernel takes
fewer than 1,024 experts, so the stack cannot be addressed as one) and copies the sum back. For the token counts vLLM
captures (GRAPH_M) the whole request (copies in, experts, copy out) is one CUDA graph keyed by (layer, M), captured on
first use; other counts (prefill chunks) run eagerly.
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import ERR, LAYER, M_, MAXM, READY, READY_MAGIC, REQ, RESP, H, K, open_channel  # noqa: E402
from peer_moe import experts_fp8  # noqa: E402

E, CLAMP = 288, 10.0
GRAPH_M = (1, 2, 4, 6, 8, 12, 16, 18, 24, 30, 36, 48, 60, 72, 96, 120, 144, 192)


def load_cold(model: str, rowmap: dict, dev: torch.device):
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    files: dict = {}

    def get(name):
        f = idx[name]
        if f not in files:
            files[f] = safe_open(os.path.join(model, f), "pt")
        return files[f].get_tensor(name)

    layers = sorted(int(k) for k in rowmap["layers"])
    n = {len(rowmap["layers"][str(li)]["cold"]) for li in layers}
    if len(n) != 1:
        raise SystemExit(f"GLM53_SIDECAR: the rowmap's layers have different cold counts {sorted(n)}")
    n = n.pop()
    total = len(layers) * n
    W1 = torch.empty(total, 2 * 2048, H, dtype=torch.float8_e4m3fn, device=dev)
    S1 = torch.empty(total, 2 * 2048 // 128, H // 128, dtype=torch.float32, device=dev)
    W2 = torch.empty(total, H, 2048, dtype=torch.float8_e4m3fn, device=dev)
    S2 = torch.empty(total, H // 128, 2048 // 128, dtype=torch.float32, device=dev)
    for li, layer in enumerate(layers):
        pre = f"model.language_model.layers.{layer}.mlp.experts"
        for j, e in enumerate(rowmap["layers"][str(layer)]["cold"]):
            s = li * n + j
            W1[s, :2048].copy_(get(f"{pre}.{e}.gate_proj.weight"))
            W1[s, 2048:].copy_(get(f"{pre}.{e}.up_proj.weight"))
            S1[s, :16].copy_(get(f"{pre}.{e}.gate_proj.weight_scale_inv"))
            S1[s, 16:].copy_(get(f"{pre}.{e}.up_proj.weight_scale_inv"))
            W2[s].copy_(get(f"{pre}.{e}.down_proj.weight"))
            S2[s].copy_(get(f"{pre}.{e}.down_proj.weight_scale_inv"))
    views = {}
    for li, layer in enumerate(layers):
        m = torch.full((E,), -1, dtype=torch.int32)
        m[torch.tensor(rowmap["layers"][str(layer)]["cold"])] = torch.arange(n, dtype=torch.int32)
        views[layer] = [t[li * n:(li + 1) * n] for t in (W1, S1, W2, S2)] + [m.to(dev)]
    return views, sum(t.numel() * t.element_size() for t in (W1, S1, W2, S2)) / 2**30


def main():
    model, rowmap_path, path = sys.argv[1:4]
    dev = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(dev) != (12, 0):
        raise SystemExit(f"GLM53_SIDECAR: expected the RTX PRO 6000 (12.0), got {torch.cuda.get_device_name(dev)}")
    t0 = time.time()
    views, gib = load_cold(model, json.load(open(rowmap_path)), dev)
    print(f"GLM53_SIDECAR loaded {len(views)} layers of cold experts, {gib:.1f} GiB, in {time.time() - t0:.0f} s",
          flush=True)
    ch = open_channel(path, create=True)
    hdr, x_h, ids_h, w_h, y_h = ch["hdr"], ch["x"], ch["ids"], ch["w"], ch["y"]
    x_d = torch.empty(MAXM, H, dtype=torch.bfloat16, device=dev)
    ids_d = torch.empty(MAXM, K, dtype=torch.int32, device=dev)
    w_d = torch.empty(MAXM, K, dtype=torch.float32, device=dev)

    def request(layer: int, m: int) -> None:
        x_d[:m].copy_(x_h[:m], non_blocking=True)
        ids_d[:m].copy_(ids_h[:m], non_blocking=True)
        w_d[:m].copy_(w_h[:m], non_blocking=True)
        w1, s1, w2, s2, emap = views[layer]
        y = experts_fp8(x_d[:m], w1, w2, s1, s2, w_d[:m], ids_d[:m], emap, CLAMP)
        y_h[:m].copy_(y, non_blocking=True)

    graphs: dict = {}
    pool = torch.cuda.graph_pool_handle()  # one pool for all graphs: requests replay one at a time, nothing outlives one
    stream = torch.cuda.Stream(dev)
    torch.cuda.set_stream(stream)

    def serve(layer: int, m: int) -> None:
        if m not in GRAPH_M:
            request(layer, m)
            return
        g = graphs.get((layer, m))
        if g is None:
            request(layer, m)  # warm-up outside capture (Triton compiles, allocator settles)
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream, pool=pool):
                request(layer, m)
            graphs[(layer, m)] = g
        g.replay()

    last = int(hdr[REQ])
    hdr[READY] = READY_MAGIC
    print(f"GLM53_SIDECAR ACTIVE: serving {path} on {torch.cuda.get_device_name(dev)}, CUDA graphs for M in {GRAPH_M}",
          flush=True)
    served = 0
    stats = dict(busy=0.0, idle=0.0, n=0)  # decode-sized requests: time serving vs time waiting for the next
    t_done = time.perf_counter()
    while True:
        s = int(hdr[REQ])
        if s == last:
            continue
        t_seen = time.perf_counter()
        layer, m = int(hdr[LAYER]), int(hdr[M_])
        if layer not in views or not 0 < m <= MAXM:
            hdr[ERR] = s
            raise SystemExit(f"GLM53_SIDECAR: bad request seq {s}: layer {layer}, {m} tokens")
        serve(layer, m)
        stream.synchronize()
        hdr[RESP] = s
        last = s
        served += 1
        now = time.perf_counter()
        if m <= 64:
            stats["busy"] += now - t_seen
            stats["idle"] += t_seen - t_done
            stats["n"] += 1
        t_done = now
        if served % 50000 == 0:
            n = max(stats["n"], 1)
            print(f"GLM53_SIDECAR served {served} requests, {len(graphs)} graphs; decode-sized requests: "
                  f"{1e6 * stats['busy'] / n:.0f} us serving, {1e6 * stats['idle'] / n:.0f} us waiting for the next "
                  f"(n={stats['n']})", flush=True)
            stats = dict(busy=0.0, idle=0.0, n=0)


if __name__ == "__main__":
    main()
