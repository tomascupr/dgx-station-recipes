#!/usr/bin/env python3
"""The RTX sidecar: computes each layer's second tier of experts for the GB300's vLLM, through the channel.

    ksidecar.py PACK_DIRS RANK.json PEER_FIRST PEER CHANNEL

Loads, for every MoE layer that has a packed file, the experts ranked PEER_FIRST .. PEER_FIRST+PEER-1 (the ones the
vLLM hook leaves out of HBM and Grace memory) into one block of RTX memory, creates the channel file, marks itself ready, then
serves requests in order. A request is one layer's latent rows, top-k ids and weights; the answer is the weighted sum
of this GPU's experts for each row (kmoe.packed_experts with a table that knows only these experts). For decode-sized
token counts (GRAPH_M) a request is one CUDA graph keyed by (layer, M), captured on first use.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hook"))
from kchannel import ERR, LAYER, M_, MAXM, READY, READY_MAGIC, REQ, RESP, H, K, open_channel  # noqa: E402
from kmoe import FAR, packed_experts, table_entries  # noqa: E402
from kpack import GROUP, layout, mmap_experts  # noqa: E402

GRAPH_M = (1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 48)


def load(dirs, rank, hot, peer, dev):
    lay, per = None, {}
    for layer, order in sorted(rank.items()):
        path = next((p for p in (os.path.join(d, f"layer-{layer:02d}.safetensors") for d in dirs)
                     if os.path.exists(p)), None)
        if path is None:
            continue
        src, md = mmap_experts(path)
        lay = lay or layout(int(md["inter"]), int(md["latent"]))
        if int(md["group"]) != GROUP or src.shape[1] != lay["size"] or lay["w1"]["inn"] != H:
            raise SystemExit(f"KIMI_SIDECAR: {path} is not in the expected packed layout")
        per[layer] = (src, sorted(order[hot:hot + peer]))
    store = torch.empty((len(per) * peer, lay["size"]), dtype=torch.uint8, device=dev)
    views = {}
    for li, (layer, (src, ids)) in enumerate(sorted(per.items())):
        for j, e in enumerate(ids):
            store[li * peer + j].copy_(torch.from_numpy(src[e].copy()))
        E = src.shape[0]
        table = torch.full((E,), FAR, dtype=torch.int64)
        own = torch.full((E,), -1, dtype=torch.int32)
        table[ids] = table_entries(torch.arange(peer) * lay["size"])
        own[ids] = torch.tensor(ids, dtype=torch.int32)
        views[layer] = (store[li * peer:(li + 1) * peer], table.to(dev), own.to(dev))
    return views, lay, store.numel() / 2**30


def main():
    dirs, rank_path, hot, peer, path = sys.argv[1].split(","), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    dev = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(dev) != (12, 0):
        raise SystemExit(f"KIMI_SIDECAR: expected the RTX PRO 6000 (12.0), got {torch.cuda.get_device_name(dev)}")
    t0 = time.time()
    rank = {int(k): v for k, v in json.load(open(rank_path))["layers"].items()}
    views, lay, gib = load(dirs, rank, hot, peer, dev)
    print(f"KIMI_SIDECAR loaded {peer} experts for each of {len(views)} layers, {gib:.1f} GiB, in "
          f"{time.time() - t0:.0f} s", flush=True)
    ch = open_channel(path, create=True)
    hdr, x_h, ids_h, w_h, y_h = ch["hdr"], ch["x"], ch["ids"], ch["w"], ch["y"]
    x_d = torch.empty(MAXM, H, dtype=torch.bfloat16, device=dev)
    ids_d = torch.empty(MAXM, K, dtype=torch.int32, device=dev)
    w_d = torch.empty(MAXM, K, dtype=torch.float32, device=dev)

    def request(layer: int, m: int) -> None:
        x_d[:m].copy_(x_h[:m], non_blocking=True)
        ids_d[:m].copy_(ids_h[:m], non_blocking=True)
        w_d[:m].copy_(w_h[:m], non_blocking=True)
        store, table, own = views[layer]
        y = packed_experts(x_d[:m], ids_d[:m], w_d[:m], store, table, own, lay)
        y_h[:m].copy_(y.to(torch.bfloat16), non_blocking=True)

    graphs: dict = {}
    pool = torch.cuda.graph_pool_handle()   # one pool for all graphs: requests replay one at a time
    stream = torch.cuda.Stream(dev)
    torch.cuda.set_stream(stream)

    def serve(layer: int, m: int) -> None:
        if m not in GRAPH_M:
            request(layer, m)
            return
        g = graphs.get((layer, m))
        if g is None:
            request(layer, m)                # warm-up outside capture (Triton compiles, allocator settles)
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream, pool=pool):
                request(layer, m)
            graphs[(layer, m)] = g
        g.replay()

    last = int(hdr[REQ])
    hdr[READY] = READY_MAGIC
    print(f"KIMI_SIDECAR ACTIVE: serving {path} on {torch.cuda.get_device_name(dev)}, CUDA graphs for M in {GRAPH_M}",
          flush=True)
    served = 0
    stats = dict(busy=0.0, idle=0.0, n=0)    # decode-sized requests: time serving vs time waiting for the next
    t_done = time.perf_counter()
    with torch.no_grad():
        while True:
            s = int(hdr[REQ])
            if s == last:
                continue
            t_seen = time.perf_counter()
            layer, m = int(hdr[LAYER]), int(hdr[M_])
            if layer not in views or not 0 < m <= MAXM:
                hdr[ERR] = s
                raise SystemExit(f"KIMI_SIDECAR: bad request seq {s}: layer {layer}, {m} tokens")
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
            if served % 20000 == 0:
                n = max(stats["n"], 1)
                print(f"KIMI_SIDECAR served {served} requests, {len(graphs)} graphs; decode-sized requests: "
                      f"{1e6 * stats['busy'] / n:.0f} us serving, {1e6 * stats['idle'] / n:.0f} us waiting for the "
                      f"next (n={stats['n']})", flush=True)
                stats = dict(busy=0.0, idle=0.0, n=0)


if __name__ == "__main__":
    main()
