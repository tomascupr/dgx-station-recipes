#!/usr/bin/env python3
"""ktune.py LAYER_FILE   time variants of the packed GEMM kernel as CUDA graphs (HBM and host tiers, M = 1, 6, 64)."""
import sys
import time

import torch
import triton
import triton.language as tl

from kmoe import FAR, packed_experts, situ
from kpack import GROUP, layout


@triton.jit
def _v2(a_ptr, base_ptr, table_ptr, c_ptr, sorted_ptr, eid_ptr, npost_ptr, w_ptr,
        N, n_valid, stride_a, stride_c, coff, soff,
        K: tl.constexpr, TOPK: tl.constexpr, MUL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
        G: tl.constexpr, T: tl.constexpr, HALF: tl.constexpr):
    nn = tl.cdiv(N, BN)
    pm = tl.program_id(0) // nn
    pn = tl.program_id(0) % nn
    if pm * BM >= tl.load(npost_ptr):
        return
    route = tl.load(sorted_ptr + pm * BM + tl.arange(0, BM)).to(tl.int64)
    live = route < n_valid
    rows = pn * BN + tl.arange(0, BN)
    rok = rows < N
    off = tl.load(table_ptr + tl.load(eid_ptr + pm).to(tl.int64))
    blob = base_ptr + off
    cbase = blob + coff + rows.to(tl.int64) * (K // 4)
    sbase = blob + soff + rows.to(tl.int64) * (K // G)
    abase = a_ptr + (route // TOPK) * stride_a
    byte = tl.arange(0, T // 4)
    grp = tl.arange(0, T // G)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for t in range(0, K // T):
        b = tl.load(cbase[:, None] + t * (T // 4) + byte[None, :], mask=rok[:, None], other=0).to(tl.int32)
        u = tl.load(sbase[:, None] + t * (T // G) + grp[None, :], mask=rok[:, None], other=0).to(tl.float32)
        s = tl.exp2((u - 200.0) * 0.125)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, T // G, G // 4)), (BN, T // 4))
        if HALF:
            s = s.to(tl.float16)
        for k in tl.static_range(4):
            a = tl.load(abase[:, None] + (t * T + 4 * byte + k)[None, :], mask=live[:, None], other=0.0)
            if HALF:
                w = (2 * ((b >> (2 * k)) & 3) - 3).to(tl.float16) * s
                acc = tl.dot(a.to(tl.float16), tl.trans(w), acc)
            else:
                w = ((2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32) * s).to(a.dtype)
                acc = tl.dot(a, tl.trans(w), acc)
    if MUL:
        acc = acc * tl.load(w_ptr + route, mask=live, other=0.0)[:, None]
    tl.store(c_ptr + route[:, None] * stride_c + rows[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=live[:, None] & rok[None, :])


def _go(a, base, table, c, sorted_ids, expert_ids, npost, w, lay, name, topk, mul, n_valid, bm, bn, T, half, warps):
    f = lay[name]
    grid = (triton.cdiv(sorted_ids.shape[0], bm) * triton.cdiv(f["out"], bn),)
    _v2[grid](a, base, table, c, sorted_ids, expert_ids, npost, w, f["out"], n_valid, a.stride(0), c.stride(0),
              f["codes"], f["scale"], K=f["inn"], TOPK=topk, MUL=mul, BM=bm, BN=bn, G=GROUP, T=T, HALF=half,
              num_warps=warps)


def v2(x, topk_ids, topk_weights, base, table, own, lay, bm=16, bn=64, T=256, half=False, warps=4):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    M, k = topk_ids.shape
    R = M * k
    inter, latent = lay["w1"]["out"], lay["w1"]["inn"]
    sorted_ids, expert_ids, npost = moe_align_block_size(topk_ids, bm, own.numel(), own, ignore_invalid_experts=True)
    h = torch.zeros((2, R, inter), dtype=x.dtype, device=x.device)
    args = (bm, bn, T, half, warps)
    _go(x, base, table, h[0], sorted_ids, expert_ids, npost, topk_weights, lay, "w1", k, False, R, *args)
    _go(x, base, table, h[1], sorted_ids, expert_ids, npost, topk_weights, lay, "w3", k, False, R, *args)
    mid = situ(h[0], h[1], 4.0, 25.0)
    out = torch.zeros((R, latent), dtype=torch.float32, device=x.device)
    _go(mid, base, table, out, sorted_ids, expert_ids, npost, topk_weights.reshape(-1), lay, "w2", 1, True, R, *args)
    return out.view(M, k, latent).sum(1)


if __name__ == "__main__":
    from safetensors import safe_open
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    torch.manual_seed(0)
    dev = "cuda"
    lay = layout(3072, 3584)
    with safe_open(sys.argv[1], "pt") as f:
        allb = f.get_tensor("experts")
    E, K = allb.shape[0], 16
    hot_ids, host_ids = torch.arange(0, E // 2), torch.arange(E // 2, E)
    hot = allb[hot_ids].to(dev)
    host_cpu = allb[host_ids].pin_memory()
    host = get_accelerator_view_from_cpu_tensor(host_cpu)
    table = torch.full((E,), FAR, dtype=torch.int64)
    table[hot_ids] = torch.arange(len(hot_ids)) * lay["size"]
    table[host_ids] = host.data_ptr() - hot.data_ptr() + torch.arange(len(host_ids)) * lay["size"]
    own = torch.arange(E, dtype=torch.int32).to(dev)
    table = table.to(dev)
    variants = [("v1 bn64", lambda *a: packed_experts(*a, bm=16, bn=64))]
    for half in (False, True):
        for T in (256, 512):
            for bn in (32, 64, 128):
                for warps in (4, 8):
                    variants.append((f"v2 {'fp16' if half else 'bf16'} T{T} bn{bn} w{warps}",
                                     lambda *a, bn=bn, T=T, half=half, warps=warps: v2(*a, 16, bn, T, half, warps)))
    for sel, label in ((hot_ids, "HBM"), (host_ids, "host")):
        pool = sel.to(dev)
        for M in (1, 6, 64):
            x = (torch.randn(M, 3584, device=dev) * 0.5).to(torch.bfloat16)
            ids = torch.stack([pool[torch.randperm(len(pool), device=dev)[:K]] for _ in range(M)]).int()
            w = torch.rand(M, K, device=dev); w = w / w.sum(-1, keepdim=True)
            args = (x, ids, w, hot, table, own, lay)
            want = packed_experts(*args, bm=16)
            uniq = ids.unique().numel()
            res = []
            for name, fn in variants:
                try:
                    y = fn(*args)
                    rel = ((y - want).norm() / want.norm()).item()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        y = fn(*args)
                    g.replay(); torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(20):
                        g.replay()
                    torch.cuda.synchronize()
                    res.append(((time.time() - t0) / 20, name, rel))
                except Exception as ex:                      # a variant this Triton cannot compile
                    print(f"{name}: {type(ex).__name__}: {str(ex)[:150]}", flush=True)
            res.sort()
            for dt, name, rel in res[:6] + [r for r in res if r[1].startswith("v1")]:
                print(f"{label:4s} M {M:2d} {name:24s}: {dt * 1e3:7.3f} ms per layer, rel diff {rel:.1e}, "
                      f"{uniq * lay['size'] / dt / 2**30:7.1f} GiB/s", flush=True)
