#!/usr/bin/env python3
"""kmoe_test.py [LAYER_FILE]   packed_experts against the unpacked reference, on hot + host-resident experts, eagerly
and as a CUDA graph, then timings per token count. With a layer file (krtn.py / kgptq.py output) the experts are real."""
import sys
import time

import torch

from kmoe import FAR, packed_experts, reference, table_entries
from kpack import GROUP, from_blob, layout, scale_to_u8

torch.manual_seed(0)
dev = "cuda"
INTER, LATENT, K = 3072, 3584, 16
lay = layout(INTER, LATENT)
if len(sys.argv) > 1:
    from safetensors import safe_open
    with safe_open(sys.argv[1], "pt") as f:
        allb = f.get_tensor("experts")
    E = allb.shape[0]
else:
    E = 96
    allb = torch.randint(0, 256, (E, lay["size"]), dtype=torch.uint8)
    for n in ("w1", "w2", "w3"):
        from_blob(allb, lay, n)[1].copy_(scale_to_u8(torch.rand(E, lay[n]["out"], lay[n]["inn"] // GROUP) * 0.02 + 0.001))
print(f"{E} experts of {lay['size']} bytes", flush=True)

# a third of the experts in HBM, a third in pinned host memory seen through UVA, a third held elsewhere
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
perm = torch.randperm(E)
hot_ids, host_ids = perm[: E // 3], perm[E // 3: 2 * E // 3]
hot = allb[hot_ids].to(dev)
host_cpu = allb[host_ids].pin_memory()
host = get_accelerator_view_from_cpu_tensor(host_cpu)
table = torch.full((E,), FAR, dtype=torch.int64)
table[hot_ids] = table_entries(torch.arange(len(hot_ids)) * lay["size"])
table[host_ids] = table_entries(host.data_ptr() - hot.data_ptr() + torch.arange(len(host_ids)) * lay["size"])
own = torch.full((E,), -1, dtype=torch.int32)
own[hot_ids] = hot_ids.int(); own[host_ids] = host_ids.int()
table, own = table.to(dev), own.to(dev)


def inputs(M):
    x = (torch.randn(M, LATENT, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).int()
    w = torch.rand(M, K, device=dev)
    return x, ids, w / w.sum(-1, keepdim=True)


for M in (1, 6, 40):
    x, ids, w = inputs(M)
    got = packed_experts(x, ids, w, hot, table, own, lay)
    want = reference(x, ids, w, lambda e: allb[e].to(dev), own, lay)
    rel = ((got - want).norm() / want.norm()).item()
    print(f"M {M:3d}: relative difference from the unpacked reference {rel:.2e}", flush=True)
    assert rel < 2e-2, rel

g_, u_ = torch.randn(5000, device=dev).to(torch.bfloat16) * 6, torch.randn(5000, device=dev).to(torch.bfloat16) * 30
from kmoe import situ
want_s = (4.0 * torch.tanh(g_.float() / 4.0) * torch.sigmoid(g_.float()) * 25.0 * torch.tanh(u_.float() / 25.0))
rel = ((situ(g_, u_, 4.0, 25.0).float() - want_s).norm() / want_s.norm()).item()
print(f"situ kernel vs torch: {rel:.2e}", flush=True)
assert rel < 5e-3, rel

for M in (1, 6, 40):                      # both paths against each other at every size
    x, ids, w = inputs(M)
    y1 = packed_experts(x, ids, w, hot, table, own, lay, grouped=True)
    y2 = packed_experts(x, ids, w, hot, table, own, lay, grouped=False)
    rel = ((y1 - y2).norm() / y1.norm()).item()
    print(f"M {M:3d}: per-route path vs grouped path {rel:.2e}", flush=True)
    assert rel < 2e-2, rel

for sel, label in ((hot_ids, "HBM only"), (host_ids, "host only"), (None, "mixed")):
    for M in (1, 3, 6, 16, 64, 512) + ((8192,) if sel is None else ()):
        x, ids, w = inputs(M)
        if sel is not None:
            pool = sel.to(dev)
            ids = torch.stack([pool[torch.randperm(len(pool), device=dev)[:K]] for _ in range(M)]).int()
        uniq = ids.unique().numel()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        for grouped in ((False, True) if M <= 64 else (True,)):
            g = torch.cuda.CUDAGraph()
            packed_experts(x, ids, w, hot, table, own, lay, grouped=grouped)
            with torch.cuda.graph(g):
                y = packed_experts(x, ids, w, hot, table, own, lay, grouped=grouped)
            g.replay(); torch.cuda.synchronize()
            n = 30 if M <= 64 else 5
            t0 = time.time()
            for _ in range(n):
                g.replay()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n
            print(f"{label:9s} M {M:4d} {'grouped  ' if grouped else 'per route'}: {dt * 1e3:8.3f} ms per layer as a "
                  f"CUDA graph, {uniq:3d} distinct experts, workspace peak "
                  f"{(torch.cuda.max_memory_allocated() - before) / 2**30:.2f} GiB", flush=True)
            del g, y
print("kmoe: PASS")
