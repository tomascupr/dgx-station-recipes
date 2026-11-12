#!/usr/bin/env python3
"""From a routing trace (kstream.py --trace), how often would a decode step need an expert that is not on the GPU?

    tiersim.py ROUTE.pt [--bits 2.0] [--copy-gibs 335]

The trace is replayed as one decode stream: token after token, sequence after sequence. Two policies, each with room
for k experts per layer on the GPU:
  static   the k most used experts of the layer (ranked on the first half of the sequences, scored on the second)
  lru      the k most recently used experts of the layer (scored on the second half too, after warming on the first)
A miss means the expert's packed weights are read from host memory. With `bits` per weight an expert is
33.03M weights x bits / 8 bytes; misses x that size is the host traffic per token, and `copy-gibs` (the measured
host-to-GPU copy rate) turns it into a ceiling on single-stream tokens/s from this traffic alone.
"""
import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("route"); p.add_argument("--bits", type=float, default=2.0); p.add_argument("--copy-gibs", type=float, default=335)
a = p.parse_args()
d = torch.load(a.route)
route = d["route"].long()                      # [L, tokens, K]
N, T = d["shape"]
live = (route.sum((1, 2)) > 0)                 # the dense layer's row is zeros
route = route[live]
L, n, K = route.shape
E = int(route.max()) + 1
half = (N // 2) * T
dev = "cuda" if torch.cuda.is_available() else "cpu"
route = route.to(dev)

# static: rank on the first half, score on the second
cnt = torch.zeros(L, E, dtype=torch.long, device=dev)
cnt.scatter_add_(1, route[:, :half].reshape(L, -1), torch.ones(L, half * K, dtype=torch.long, device=dev))
rank = torch.empty_like(cnt)
rank.scatter_(1, cnt.argsort(1, descending=True), torch.arange(E, device=dev).expand(L, E))
r2 = rank.gather(1, route[:, half:].reshape(L, -1))          # rank of every call in the second half

# lru: stack distance of every call = distinct experts of that layer used since this expert's last use
last = torch.full((L, E), -1, dtype=torch.long, device=dev)
dist = torch.empty((L, n, K), dtype=torch.long, device=dev)
for t in range(n):
    c = route[:, t]                                           # [L, K]
    lu = last.gather(1, c)                                    # [L, K]
    dd = (last.unsqueeze(1) > lu.unsqueeze(2)).sum(2)         # [L, K]
    dist[:, t] = torch.where(lu < 0, torch.full_like(dd, E), dd)
    last.scatter_(1, c, t)
d2 = dist[:, half:].reshape(L, -1)

size = 33.03e6 * a.bits / 8
calls = L * K
print(f"{L} MoE layers, {E} experts, {n} tokens; scored on the last {n - half}. Expert at {a.bits} bits: {size / 2**20:.2f} MiB; "
      f"all experts {L * E * size / 2**30:.0f} GiB")
print(f"{'k/layer':>8s} {'GPU GiB':>8s} | {'static miss':>11s} {'GiB/tok':>8s} {'tok/s<=':>8s} | {'lru miss':>9s} {'GiB/tok':>8s} {'tok/s<=':>8s}")
for k in (56, 112, 168, 224, 280, 336, 448, 560, 672):
    ms, ml = (r2 >= k).float().mean().item(), (d2 >= k).float().mean().item()
    gs, gl = calls * ms * size / 2**30, calls * ml * size / 2**30
    print(f"{k:8d} {L * k * size / 2**30:8.0f} | {ms:11.4f} {gs:8.3f} {a.copy_gibs / gs:8.1f} | {ml:9.4f} {gl:8.3f} {a.copy_gibs / max(gl, 1e-9):8.1f}")
# how concentrated one sequence is: distinct experts per layer within a sequence of T tokens
dis = torch.stack([(torch.zeros(L, E, device=dev).scatter_(1, route[:, r * T:(r + 1) * T].reshape(L, -1), 1.0) > 0).sum(1).float().mean() for r in range(N)])
print(f"distinct experts per layer used within one {T}-token sequence: mean {dis.mean():.0f}, min {dis.min():.0f}, max {dis.max():.0f} of {E}")
torch.save(dict(static_rank=r2.cpu().to(torch.int16), lru_dist=d2.cpu().to(torch.int16)), a.route.replace("route.pt", "tiersim.pt"))
