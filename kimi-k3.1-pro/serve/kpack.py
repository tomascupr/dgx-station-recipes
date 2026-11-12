"""Packed 2-bit expert weights for Kimi K3.1 Pro, and a kernel that multiplies straight from them.

Format, for one matrix W [out, in] (in divisible by GROUP):
  codes  uint8 [out, in/4]      four 2-bit codes per byte, input j in bits 2*(j%4).. of byte j//4
  scale  uint8 [out, in/GROUP]  u -> a = 2^((u - 200) / 8)
  weight = (2*code - 3) * a     i.e. the levels {-3a, -a, +a, +3a}
That is 2 + 8/GROUP bits per weight (2.125 at GROUP = 64).

On disk and on the GPU an expert is one contiguous block of bytes ("blob"): w1 codes, w1 scale, w2 codes, w2 scale,
w3 codes, w3 scale (see `layout`). A layer file is a safetensors file with one tensor `experts` uint8 [E, BLOB], so a
server can memory-map it and move one expert with one copy.
"""
import json
import struct

import numpy as np
import torch
import triton
import triton.language as tl

GROUP = 64


def layout(inter: int, latent: int) -> dict:
    """Byte offsets inside an expert's blob. w1, w3: [inter, latent]; w2: [latent, inter]."""
    off, o = {}, 0
    for name, (out, inn) in (("w1", (inter, latent)), ("w2", (latent, inter)), ("w3", (inter, latent))):
        off[name] = dict(out=out, inn=inn, codes=o, scale=o + out * inn // 4)
        o += out * inn // 4 + out * inn // GROUP
    off["size"] = o
    return off


def to_blob(parts) -> torch.Tensor:
    """parts: [(codes [B, out, in/4], scale [B, out, in/G])] for w1, w2, w3 -> uint8 [B, BLOB]."""
    return torch.cat([t.flatten(1) for cs in parts for t in cs], dim=1)


def from_blob(blob: torch.Tensor, lay: dict, name: str):
    """blob uint8 [..., BLOB] -> (codes [..., out, in/4], scale [..., out, in/G]) views for w1, w2 or w3."""
    f = lay[name]
    nc, ns = f["out"] * f["inn"] // 4, f["out"] * f["inn"] // GROUP
    return (blob[..., f["codes"]:f["codes"] + nc].reshape(*blob.shape[:-1], f["out"], f["inn"] // 4),
            blob[..., f["scale"]:f["scale"] + ns].reshape(*blob.shape[:-1], f["out"], f["inn"] // GROUP))


def mmap_experts(path):
    """A layer file's `experts` tensor as a read-only memory map [E, BLOB], and the file's metadata."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    t = head["experts"]
    return (np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + t["data_offsets"][0], shape=tuple(t["shape"])),
            head["__metadata__"])


def scale_to_u8(a: torch.Tensor) -> torch.Tensor:
    return (torch.log2(a.float().clamp_min(1e-30)) * 8 + 200).round().clamp(0, 255).to(torch.uint8)


def u8_to_scale(u: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, (u.float() - 200) / 8)


def pack(q: torch.Tensor, a_u8: torch.Tensor):
    """q [..., out, in] holding exactly the four levels of its group's scale -> codes uint8 [..., out, in/4]."""
    a = u8_to_scale(a_u8).repeat_interleave(GROUP, dim=-1)
    code = ((q.float() / a + 3) / 2).round().clamp(0, 3).to(torch.uint8)
    c = code.reshape(*code.shape[:-1], -1, 4)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def unpack(codes: torch.Tensor, a_u8: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    c = torch.stack([(codes >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).flatten(-2)
    return ((2 * c.to(dtype) - 3) * u8_to_scale(a_u8).to(dtype).repeat_interleave(GROUP, dim=-1))


@triton.jit
def _matvec_kernel(blob_ptr, x_ptr, slot_ptr, y_ptr, stride, coff, soff, OUT, IN: tl.constexpr, G: tl.constexpr,
                   BR: tl.constexpr):
    c = tl.program_id(0)
    rows = tl.program_id(1) * BR + tl.arange(0, BR)
    ok = rows < OUT
    base = blob_ptr + tl.load(slot_ptr + c).to(tl.int64) * stride
    acc = tl.zeros([BR], dtype=tl.float32)
    cbase = base + coff + rows.to(tl.int64)[:, None] * (IN // 4)
    sbase = base + soff + rows.to(tl.int64) * (IN // G)
    xbase = x_ptr + c.to(tl.int64) * IN
    bytes_ = tl.arange(0, G // 4)
    for g in range(0, IN // G):
        b = tl.load(cbase + g * (G // 4) + bytes_[None, :], mask=ok[:, None], other=0).to(tl.int32)
        s = tl.zeros([BR], dtype=tl.float32)
        for k in tl.static_range(4):
            xk = tl.load(xbase + g * G + 4 * bytes_ + k).to(tl.float32)
            lv = (2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32)
            s += tl.sum(lv * xk[None, :], axis=1)
        u = tl.load(sbase + g, mask=ok, other=0).to(tl.float32)
        acc += tl.exp2((u - 200.0) * 0.125) * s
    tl.store(y_ptr + c.to(tl.int64) * OUT + rows, acc, mask=ok)


def matvec(store, lay, name, x, slot, BR=64):
    """y[c] = W[slot[c]] @ x[c] for matrix `name` of the experts in `store` (uint8 [slots, BLOB]); x [calls, in] f32."""
    f = lay[name]
    y = torch.empty((x.shape[0], f["out"]), dtype=torch.float32, device=x.device)
    _matvec_kernel[(x.shape[0], triton.cdiv(f["out"], BR))](store, x.contiguous(), slot.to(torch.int32), y, lay["size"],
                                                             f["codes"], f["scale"], f["out"], IN=f["inn"], G=GROUP,
                                                             BR=BR)
    return y


if __name__ == "__main__":      # correctness against the unpacked matrices, then timing of one decode step's calls
    import time
    torch.manual_seed(0)
    dev = "cuda"
    E, I, D, K = 48, 3072, 3584, 16
    lay = layout(I, D)
    store = torch.randint(0, 256, (E, lay["size"]), dtype=torch.uint8, device=dev)
    for n in ("w1", "w2", "w3"):                      # keep the scale bytes in a sane range
        from_blob(store, lay, n)[1].copy_(scale_to_u8(torch.rand(E, lay[n]["out"], lay[n]["inn"] // GROUP, device=dev) * 0.02 + 0.001))
    slot = torch.randperm(E, device=dev)[:K]
    for n in ("w1", "w2", "w3"):
        codes, scale = from_blob(store, lay, n)
        W = unpack(codes, scale)
        assert torch.equal(pack(W, scale), codes), "pack does not invert unpack"
        x = torch.randn(K, lay[n]["inn"], device=dev)
        want = torch.bmm(W[slot], x.unsqueeze(2)).squeeze(2)
        for BR in (32, 64, 128):
            got = matvec(store, lay, n, x, slot, BR)
            rel = ((got - want).norm() / want.norm()).item()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(50):
                matvec(store, lay, n, x, slot, BR)
            torch.cuda.synchronize(); dt = (time.time() - t0) / 50
            print(f"{n} BR {BR:3d}: relative difference {rel:.2e}, {dt * 1e3:.3f} ms per {K} calls")
            assert rel < 1e-5
    assert torch.equal(to_blob([from_blob(store, lay, n) for n in ("w1", "w2", "w3")]), store)
    print("kpack: PASS")
