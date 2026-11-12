"""The shared host-memory channel between the GB300's vLLM and the RTX sidecar.

The Station's driver gives a process either the GB300 or the RTX, so the two talk through one file in /dev/shm,
registered as pinned memory in both processes. One request is in flight at a time (MoE layers run in order):

  hdr int64[64]: [0] request seq, [1] layer, [2] tokens M, [16] response seq, [32] sidecar ready, [48] error seq
  x   bf16 [MAXM, 3584]   the layer's latent rows          (GB300 -> RTX)
  ids int32 [MAXM, 16]    top-k expert ids, global        (GB300 -> RTX)
  w   fp32 [MAXM, 16]     top-k weights, already scaled   (GB300 -> RTX)
  y   bf16 [MAXM, 3584]   the sidecar experts' sum per row (RTX -> GB300)

The GB300 side is three Triton launches with no host synchronization, so it records into CUDA graphs: send (copy
rows, ids and weights into the channel, then a system-scope release), publish (bump a device-side sequence number,
write layer and M, release the request seq), and wait (spin on the response seq with a system-scope acquire, up to
MAX_SPINS; on timeout it writes the seq to hdr[48] instead of returning wrong rows silently). The sidecar side is
plain host code.
"""
import mmap
import os

import torch
import triton
import triton.language as tl

H, K, MAXM = 3584, 16, 8192
HDR = 64 * 8
OFF_X = 4096
OFF_IDS = OFF_X + MAXM * H * 2
OFF_W = OFF_IDS + MAXM * K * 4
OFF_Y = OFF_W + MAXM * K * 4
SIZE = OFF_Y + MAXM * H * 2
REQ, LAYER, M_, RESP, READY, ERR = 0, 1, 2, 16, 32, 48
READY_MAGIC = 0x5EED
MAX_SPINS = 1 << 26


def _cpu_views(buf):
    def view(off, n, dtype):
        return torch.frombuffer(buf, dtype=dtype, count=n, offset=off)
    return dict(hdr=view(0, 64, torch.int64), x=view(OFF_X, MAXM * H, torch.bfloat16).view(MAXM, H),
                ids=view(OFF_IDS, MAXM * K, torch.int32).view(MAXM, K),
                w=view(OFF_W, MAXM * K, torch.float32).view(MAXM, K),
                y=view(OFF_Y, MAXM * H, torch.bfloat16).view(MAXM, H))


def open_channel(path: str, create: bool):
    """Map the channel file, register it as pinned memory for this process's GPU, return CPU tensor views."""
    fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
    if create:
        os.ftruncate(fd, SIZE)
    elif os.fstat(fd).st_size != SIZE:
        raise RuntimeError(f"KIMI_PEER: {path} has size {os.fstat(fd).st_size}, expected {SIZE}")
    buf = mmap.mmap(fd, SIZE)
    os.close(fd)
    cpu = _cpu_views(buf)
    rc = torch.cuda.cudart().cudaHostRegister(cpu["hdr"].data_ptr(), SIZE, 0x3)  # portable | mapped
    if int(rc) != 0:
        raise RuntimeError(f"KIMI_PEER: cudaHostRegister of {path} failed ({rc})")
    cpu["_mmap"] = buf
    return cpu


@triton.jit
def _send_kernel(x_ptr, ids_ptr, w_ptr, sx_ptr, sids_ptr, sw_ptr, fence_ptr,
                 H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    for h0 in tl.range(0, H, BLOCK):
        tl.store(sx_ptr + row * H + h0 + offs, tl.load(x_ptr + row * H + h0 + offs))
    k = tl.arange(0, K)
    tl.store(sids_ptr + row * K + k, tl.load(ids_ptr + row * K + k))
    tl.store(sw_ptr + row * K + k, tl.load(w_ptr + row * K + k))
    tl.atomic_add(fence_ptr, 0, sem="release", scope="sys")


@triton.jit
def _publish_kernel(hdr_ptr, seq_ptr, layer, m):
    s = tl.load(seq_ptr) + 1
    tl.store(seq_ptr, s)
    tl.store(hdr_ptr + 1, layer)  # scalars cast to the pointer's int64; m == 1 arrives as a constexpr, so no .to()
    tl.store(hdr_ptr + 2, m)
    tl.atomic_xchg(hdr_ptr, s, sem="release", scope="sys")


@triton.jit
def _wait_kernel(hdr_ptr, seq_ptr, max_spins):
    s = tl.load(seq_ptr)
    v = tl.atomic_add(hdr_ptr + 16, 0, sem="acquire", scope="sys")
    n = 0
    while (v != s) & (n < max_spins):
        v = tl.atomic_add(hdr_ptr + 16, 0, sem="acquire", scope="sys")
        n += 1
    if v != s:
        tl.store(hdr_ptr + 48, s)


class GB300Side:
    """The vLLM side: device views of the channel and the three launches."""

    def __init__(self, path: str, device: torch.device):
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as uva
        self.cpu = open_channel(path, create=False)
        self.dev = {k: uva(v) for k, v in self.cpu.items() if not k.startswith("_")}
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)
        self.fence = torch.zeros(1, dtype=torch.int32, device=device)
        for i in (REQ, RESP, ERR):
            self.cpu["hdr"][i] = 0

    def send(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, layer: int) -> None:
        m = x.shape[0]
        if m > MAXM:
            raise RuntimeError(f"KIMI_PEER: {m} tokens in one MoE call exceed the channel's {MAXM}")
        d = self.dev
        _send_kernel[(m,)](x, ids, w, d["x"], d["ids"], d["w"], self.fence, H=H, K=K, BLOCK=512)
        _publish_kernel[(1,)](d["hdr"], self.seq, layer, m)

    def wait_add(self, out: torch.Tensor) -> None:
        _wait_kernel[(1,)](self.dev["hdr"], self.seq, MAX_SPINS)
        out.add_(self.dev["y"][: out.shape[0]])
