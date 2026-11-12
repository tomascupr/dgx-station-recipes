#!/usr/bin/env python3
"""kprof.py LAYER_FILE   where the packed GEMM kernel's time goes: variants that leave out one part each
(results are wrong on purpose; only the timings mean something). HBM only, M = 1 and 6, as CUDA graphs."""
import sys
import time

import torch
import triton
import triton.language as tl

from kpack import GROUP, layout


@triton.jit
def _k(a_ptr, blob_ptr, c_ptr, slot_ptr, N, stride_a, stride_c, stride_e, coff, soff,
       K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr, T: tl.constexpr, MODE: tl.constexpr):
    nn = tl.cdiv(N, BN)
    pm = tl.program_id(0) // nn
    pn = tl.program_id(0) % nn
    rows = pn * BN + tl.arange(0, BN)
    rok = rows < N
    blob = blob_ptr + tl.load(slot_ptr + pm).to(tl.int64) * stride_e
    cbase = blob + coff + rows.to(tl.int64) * (K // 4)
    sbase = blob + soff + rows.to(tl.int64) * (K // G)
    abase = a_ptr + (pm * BM + tl.arange(0, BM)).to(tl.int64) * stride_a
    byte = tl.arange(0, T // 4)
    grp = tl.arange(0, T // G)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for t in range(0, K // T):
        b = tl.load(cbase[:, None] + t * (T // 4) + byte[None, :], mask=rok[:, None], other=0).to(tl.int32)
        if MODE == 1:                       # loads only
            acc += tl.sum(b, axis=1).to(tl.float32)[None, :]
        else:
            u = tl.load(sbase[:, None] + t * (T // G) + grp[None, :], mask=rok[:, None], other=0).to(tl.float32)
            s = tl.exp2((u - 200.0) * 0.125)
            s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, T // G, G // 4)), (BN, T // 4))
            for k in tl.static_range(4):
                if MODE == 2:               # loads + unpack, no dot
                    w = ((2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32) * s).to(tl.bfloat16)
                    acc += tl.sum(w, axis=1).to(tl.float32)[None, :]
                elif MODE == 3:             # loads + dot on the raw bytes, no unpack
                    a = tl.load(abase[:, None] + (t * T + 4 * byte + k)[None, :])
                    acc = tl.dot(a, tl.trans(b.to(tl.bfloat16)), acc)
                elif MODE == 4:             # unpack without the scale
                    a = tl.load(abase[:, None] + (t * T + 4 * byte + k)[None, :])
                    w = (2 * ((b >> (2 * k)) & 3) - 3).to(tl.bfloat16)
                    acc = tl.dot(a, tl.trans(w), acc)
                else:                       # everything
                    a = tl.load(abase[:, None] + (t * T + 4 * byte + k)[None, :])
                    w = ((2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32) * s).to(a.dtype)
                    acc = tl.dot(a, tl.trans(w), acc)
    tl.store(c_ptr + (pm * BM + tl.arange(0, BM)).to(tl.int64)[:, None] * stride_c + rows[None, :],
             acc.to(c_ptr.dtype.element_ty), mask=rok[None, :])


if __name__ == "__main__":
    from safetensors import safe_open
    torch.manual_seed(0)
    dev = "cuda"
    lay = layout(3072, 3584)
    with safe_open(sys.argv[1], "pt") as f:
        hot = f.get_tensor("experts")[:256].to(dev)
    f1 = lay["w1"]
    names = {0: "everything", 1: "code loads only", 2: "loads + unpack, no dot", 3: "loads + dot, no unpack",
             4: "loads + unpack without scale + dot"}
    for nexp in (16, 96):
        slot = torch.randperm(256, device=dev)[:nexp].int()
        for BM, BN, T in ((16, 32, 512), (16, 64, 256), (16, 128, 512)):
            a = (torch.randn(nexp * BM, 3584, device=dev) * 0.5).to(torch.bfloat16)
            c = torch.empty((nexp * BM, 3072), dtype=torch.bfloat16, device=dev)
            for mode in (0, 1, 2, 3, 4):
                def run():
                    _k[(nexp * triton.cdiv(3072, BN),)](a, hot, c, slot, 3072, a.stride(0), c.stride(0), lay["size"],
                                                        f1["codes"], f1["scale"], K=3584, BM=BM, BN=BN, G=GROUP, T=T,
                                                        MODE=mode)
                run()
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
                print(f"{nexp:2d} experts, w1 only, BN {BN:3d} T {T:4d}: {names[mode]:36s} {dt * 1e6:7.1f} us  "
                      f"{gib:7.1f} GiB/s of code bytes", flush=True)
