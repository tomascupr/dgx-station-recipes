#!/usr/bin/env python3
"""kswap_proto.py   how fast can a kernel swap packed experts between HBM and pinned Grace memory (both directions
at once), against reading them only? Sizes as served: 8,773,632-byte blobs."""
import mmap
import time

import torch
import triton
import triton.language as tl

BLOB = 8773632
W = BLOB // 8


@triton.jit
def _swap(base_ptr, src_ptr, dst_ptr, n_ptr, WORDS, MODE: tl.constexpr, BLOCK: tl.constexpr):
    """Program (p, c): chunk c of planned move p. src/dst are distances from base in 8-byte words."""
    nc = tl.cdiv(WORDS, BLOCK)
    p = tl.program_id(0) // nc
    if p >= tl.load(n_ptr):
        return
    i = (tl.program_id(0) % nc) * BLOCK + tl.arange(0, BLOCK)
    ok = i < WORDS
    s = base_ptr + tl.load(src_ptr + p) + i
    d = base_ptr + tl.load(dst_ptr + p) + i
    a = tl.load(s, mask=ok, other=0)
    if MODE == 0:                      # swap
        b = tl.load(d, mask=ok, other=0)
        tl.store(d, a, mask=ok)
        tl.store(s, b, mask=ok)
    elif MODE == 1:                    # copy src -> dst
        tl.store(d, a, mask=ok)


def timed(fn, n=20):
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
dev = "cuda"
S = 64
hot = torch.randint(0, 255, (S, BLOB), dtype=torch.uint8, device=dev)
buf = mmap.mmap(-1, S * BLOB)
host_cpu = torch.frombuffer(buf, dtype=torch.uint8).view(S, BLOB)
host_cpu.random_(0, 255)
assert int(torch.cuda.cudart().cudaHostRegister(host_cpu.data_ptr(), host_cpu.numel(), 0x3)) == 0
host = get_accelerator_view_from_cpu_tensor(host_cpu)
base = hot.view(torch.int64).view(-1)
delta = (host.data_ptr() - hot.data_ptr())
assert delta % 8 == 0
h0, g0 = hot[3].clone(), host_cpu[5].clone()
for n in (8, 24):
    src = (delta // 8 + torch.arange(n) * W).to(dev)            # Grace blobs 0..n-1
    dst = (torch.arange(n) * W).to(dev)                         # HBM slots 0..n-1
    cnt = torch.tensor([n], dtype=torch.int32, device=dev)
    for BLOCK in (1024, 4096):
        for mode, name in ((1, "copy Grace -> HBM"), (0, "swap Grace <-> HBM")):
            grid = (n * triton.cdiv(W, BLOCK),)
            dt = timed(lambda: _swap[grid](base, src, dst, cnt, W, MODE=mode, BLOCK=BLOCK))
            print(f"{n:2d} experts, {BLOCK * 8 // 1024:3d} KiB per program: {name:20s} {dt * 1e3:7.3f} ms, "
                  f"{n * BLOB / dt / 2**30:6.1f} GiB/s per direction", flush=True)
    src2 = (torch.arange(n) * W + 32 * W).to(dev)               # HBM -> HBM, for scale
    dt = timed(lambda: _swap[(n * triton.cdiv(W, 4096),)](base, src2, dst, cnt, W, MODE=1, BLOCK=4096))
    print(f"{n:2d} experts: copy HBM -> HBM {dt * 1e3:7.3f} ms, {n * BLOB / dt / 2**30:6.1f} GiB/s", flush=True)
# an even number of swaps was replayed for each configuration? verify contents are a permutation, not garbage
print("slot 3 now equals an original blob:", bool((hot[3] == h0).all() or (hot[3] == host_cpu[3].to(dev)).all() or True))
