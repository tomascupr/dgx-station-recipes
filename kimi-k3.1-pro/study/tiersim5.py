#!/usr/bin/env python3
"""A copy-only cache in HBM (swapping costs twice the link traffic): which split pays? Replay of a routing trace.

    tiersim5.py ROUTE.pt [--peer 110] [--slots 200] [--grace 538] [--rows 16]

Per layer, by usage rank: the RTX holds the top --peer; of the HBM's --slots, S are a fixed tier (the next S of the
ranking, held only there) and C = slots - S are a cache holding copies of experts that also stay in Grace memory;
Grace holds --grace experts, so slots - S more of the ranking's tail must be dropped than today. For each S: the share
of calls that hit dropped experts, and the experts fetched from Grace per step (all layers) at 1 and 3 tokens per step,
a miss being one copy into the cache (least recently used out).
"""
import argparse

import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("route"); p.add_argument("--peer", type=int, default=110); p.add_argument("--slots", type=int, default=200)
p.add_argument("--grace", type=int, default=538); p.add_argument("--rows", type=int, default=16)
a = p.parse_args()
d = torch.load(a.route)
route = d["route"].long()
N, T = d["shape"]
route = route[route.sum((1, 2)) > 0]
L, n, K = route.shape
E = int(route.max()) + 1
half = (N // 2) * T
blob = 8773632
cnt = torch.zeros(L, E, dtype=torch.long).scatter_add_(1, route[:, :half].reshape(L, -1), torch.ones(L, half * K, dtype=torch.long))
rank = torch.empty_like(cnt).scatter_(1, cnt.argsort(1, descending=True), torch.arange(E).expand(L, E))
test = route[:, half:half + a.rows * T]
for S in (200, 150, 100, 50, 0):
    C = a.slots - S
    fixed = (rank >= a.peer) & (rank < a.peer + S)
    grace = (rank >= a.peer + S) & (rank < a.peer + S + a.grace)
    dropped = rank >= a.peer + S + a.grace
    line = (f"fixed {S:3d} + cache {C:3d}, dropped {int(dropped[0].sum()):3d} per layer "
            f"({dropped.gather(1, test.reshape(L, -1)).float().mean().item():.3f} of calls):")
    for W in (1, 3):
        steps = test.shape[1] // W
        cached = torch.zeros(L, E, dtype=torch.bool)
        if C:                                           # start with the most used Grace experts cached
            cached = grace & (rank < a.peer + S + C)
        last = torch.zeros(L, E, dtype=torch.long)
        fetched = 0
        for s in range(steps):
            ids = test[:, s * W:(s + 1) * W].reshape(L, -1)
            need = torch.zeros(L, E, dtype=torch.bool).scatter_(1, ids, True) & grace
            miss = need & ~cached
            fetched += int(miss.sum())
            if C:
                m = miss.sum(1, keepdim=True)
                last[need & cached] = s + 1
                score = torch.where(cached & ~need, last, torch.full((), 1 << 60, dtype=torch.long))
                evict = torch.zeros(L, E, dtype=torch.bool).scatter_(1, score.argsort(1), torch.arange(E).expand(L, E) < m)
                cached = (cached & ~evict) | miss
                last[miss] = s + 1
        e = fetched / steps
        line += f"  {W} token(s)/step {e:7.1f} experts = {e * blob / 2**30 / 320 * 1e3:5.1f} ms;"
    print(line, flush=True)
