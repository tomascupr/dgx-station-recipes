#!/usr/bin/env python3
"""kprof2.py LAYER_FILE   decode-sized variants without tensor cores: R routes per expert block, each a row sum."""
import sys
import time

import torch
import triton
import triton.language as tl

from kpack import GROUP, from_blob, layout, unpack


@triton.jit
def _vec(a_ptr, blob_ptr, c_ptr, slot_ptr, N, stride_a, stride_c, stride_e, coff, soff,
         K: tl.constexpr, R: tl.constexpr, BN: tl.constexpr, G: tl.constexpr, T: tl.constexpr):
    nn = tl.cdiv(N, BN)
    pm = tl.program_id(0) // nn
    pn = tl.program_id(0) % nn
    rows = pn * BN + tl.arange(0, BN)
    rok = rows < N
    blob = blob_ptr + tl.load(slot_ptr + pm).to(tl.int64) * stride_e
    cbase = blob + coff + rows.to(tl.int64) * (K // 4)
    sbase = blob + soff + rows.to(tl.int64) * (K // G)
    byte = tl.arange(0, T // 4)
    grp = tl.arange(0, T // G)
    acc0 = tl.zeros((BN,), dtype=tl.float32)
    acc1 = tl.zeros((BN,), dtype=tl.float32)
    acc2 = tl.zeros((BN,), dtype=tl.float32)
    acc3 = tl.zeros((BN,), dtype=tl.float32)
    a0 = a_ptr + (pm * R).to(tl.int64) * stride_a
    for t in range(0, K // T):
        b = tl.load(cbase[:, None] + t * (T // 4) + byte[None, :], mask=rok[:, None], other=0).to(tl.int32)
        u = tl.load(sbase[:, None] + t * (T // G) + grp[None, :], mask=rok[:, None], other=0).to(tl.float32)
        s = tl.exp2((u - 200.0) * 0.125)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, T // G, G // 4)), (BN, T // 4))
        for k in tl.static_range(4):
            w = (2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32) * s
            x = tl.load(a0 + t * T + 4 * byte + k).to(tl.float32)
            acc0 += tl.sum(w * x[None, :], axis=1)
            if R > 1:
                x = tl.load(a0 + stride_a + t * T + 4 * byte + k).to(tl.float32)
                acc1 += tl.sum(w * x[None, :], axis=1)
            if R > 2:
                x = tl.load(a0 + 2 * stride_a + t * T + 4 * byte + k).to(tl.float32)
                acc2 += tl.sum(w * x[None, :], axis=1)
                x = tl.load(a0 + 3 * stride_a + t * T + 4 * byte + k).to(tl.float32)
                acc3 += tl.sum(w * x[None, :], axis=1)
    c0 = c_ptr + (pm * R).to(tl.int64) * stride_c + rows
    tl.store(c0, acc0.to(c_ptr.dtype.element_ty), mask=rok)
    if R > 1:
        tl.store(c0 + stride_c, acc1.to(c_ptr.dtype.element_ty), mask=rok)
    if R > 2:
        tl.store(c0 + 2 * stride_c, acc2.to(c_ptr.dtype.element_ty), mask=rok)
        tl.store(c0 + 3 * stride_c, acc3.to(c_ptr.dtype.element_ty), mask=rok)


if __name__ == "__main__":
    from safetensors import safe_open
    torch.manual_seed(0)
    dev = "cuda"
    lay = layout(3072, 3584)
    with safe_open(sys.argv[1], "pt") as f:
        hot = f.get_tensor("experts")[:256].to(dev)
    f1 = lay["w1"]
    for nexp in (16, 96):
        slot = torch.randperm(256, device=dev)[:nexp].int()
        for R in (1, 2, 4):
            a = (torch.randn(nexp * R, 3584, device=dev) * 0.5).to(torch.bfloat16)
            W = unpack(*from_blob(hot[slot[0].item()], lay, "w1"))
            want = a[:R].float() @ W.T
            for BN, T, warps in ((32, 512, 4), (64, 256, 4), (16, 512, 4), (16, 1024, 4), (32, 256, 4), (32, 512, 8),
                                 (64, 512, 8), (32, 1024, 8), (8, 2048, 4), (8, 1024, 4), (16, 2048, 8)):
                if 3584 % T:
                    continue
                c = torch.empty((nexp * R, 3072), dtype=torch.float32, device=dev)

                def run():
                    _vec[(nexp * triton.cdiv(3072, BN),)](a, hot, c, slot, 3072, a.stride(0), c.stride(0), lay["size"],
                                                          f1["codes"], f1["scale"], K=3584, R=R, BN=BN, G=GROUP, T=T,
                                                          num_warps=warps)
                run()
                rel = ((c[:R] - want).norm() / want.norm()).item()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run()
                g.replay(); torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(50):
                    g.replay()
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 50
                gib = nexp * 3072 * 3584 / 4 / dt / 2**30
                print(f"{nexp:2d} experts x {R} routes, w1: BN {BN:3d} T {T:4d} warps {warps}: {dt * 1e6:7.1f} us  "
                      f"{gib:7.1f} GiB/s of code bytes, rel diff {rel:.1e}", flush=True)
