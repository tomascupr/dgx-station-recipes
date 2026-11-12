#!/usr/bin/env python3
"""Would a recency-based HBM tier beat the usage-ranked one? Replay of a routing trace (kstream.py --trace).

    tiersim4.py ROUTE.pt [--peer 110] [--hot 200] [--prune 48] [--rows 24]

As served: each layer's experts are ranked by use (first half of the sequences); the RTX holds the top --peer, the
bottom --prune are dropped (their calls go to other experts in reality; here they are just not counted), and of the
rest --hot are in HBM and the others in Grace memory. Replayed on sequences of the second half, in steps of 1 and 3
tokens, counting the experts a step must fetch from Grace memory, per step and summed over the layers:
  static, per route   HBM = the next --hot of the ranking; every route to a Grace expert reads it (today's decode path)
  static, deduped     the same, each distinct expert read once per step
  LRU                 HBM = the --hot most recently used; a missing expert is copied in once, replacing the least
                      recently used one that this step does not need
"""
import argparse

import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("route"); p.add_argument("--peer", type=int, default=110); p.add_argument("--hot", type=int, default=200)
p.add_argument("--prune", type=int, default=48); p.add_argument("--rows", type=int, default=24)
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
local = (rank >= a.peer) & (rank < E - a.prune)                      # [L, E] experts the GB300 process serves
static_hot = local & (rank < a.peer + a.hot)
test = route[:, half:half + a.rows * T]
share = [(x.gather(1, test.reshape(L, -1)).float().mean().item()) for x in (rank < a.peer, static_hot, local & ~static_hot, rank >= E - a.prune)]
print(f"shares of calls on the test rows: RTX {share[0]:.3f}, HBM (static) {share[1]:.3f}, Grace {share[2]:.3f}, pruned {share[3]:.3f}")
for W in (1, 3):
    steps = test.shape[1] // W
    in_hbm = static_hot.clone()
    last = torch.where(in_hbm, torch.zeros((), dtype=torch.long), torch.full((), -1, dtype=torch.long)).expand(L, E).clone()
    tot = dict(route=0, dedup=0, lru=0)
    for s in range(steps):
        ids = test[:, s * W:(s + 1) * W].reshape(L, -1)                 # [L, W*K]
        need = torch.zeros(L, E, dtype=torch.bool).scatter_(1, ids, True) & local
        tot["route"] += int((local & ~static_hot).gather(1, ids).sum())
        tot["dedup"] += int((need & ~static_hot).sum())
        miss = need & ~in_hbm
        m = miss.sum(1, keepdim=True)                                   # [L, 1] experts to bring in
        tot["lru"] += int(m.sum())
        last[need & in_hbm] = s + 1
        score = torch.where(in_hbm & ~need, last, torch.full((), 1 << 60, dtype=torch.long))
        order = score.argsort(1)
        evict = torch.zeros(L, E, dtype=torch.bool).scatter_(1, order, torch.arange(E).expand(L, E) < m)
        in_hbm = (in_hbm & ~evict) | miss
        last[miss] = s + 1
    print(f"steps of {W} token(s), {steps} steps: Grace experts fetched per step (all layers), GiB, ms at 320 GiB/s:")
    for k, name in (("route", "static, per route"), ("dedup", "static, deduped"), ("lru", "LRU")):
        e = tot[k] / steps
        print(f"  {name:18s} {e:7.1f} experts  {e * blob / 2**30:6.2f} GiB  {e * blob / 2**30 / 320 * 1e3:6.1f} ms")
