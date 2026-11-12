"""Kimi K3.1 Pro's routed experts computed straight from kpack.py's packed 2-bit blobs, for vLLM.

packed_experts() is one MoE layer's routed part in the latent space: for every token the weighted sum of its top-k
experts' w2 @ situ(w1 @ x, w3 @ x). A kernel launch reads every expert through one address table:
`table[e]` is the distance from `base` to expert e's blob in units of UNIT bytes, wherever that blob lives (HBM, or
pinned host memory mapped into the GPU's address space), or FAR for an expert this GPU does not hold (its routes
contribute zero).
Nothing is unpacked, nothing depends on values at Python level, so the whole call records into a CUDA graph.
"""
import torch
import triton
import triton.language as tl

from kpack import GROUP

UNIT = 16               # table entries count UNIT bytes: Triton only merges byte loads into wide ones when it can
                        # prove the address aligned, which it can for "entry * UNIT" but not for a loaded number
FAR = -(1 << 57)        # table entry of an expert held elsewhere


def table_entries(byte_offsets: torch.Tensor) -> torch.Tensor:
    """Byte distances from `base` -> table entries. Blobs must sit at multiples of UNIT from each other."""
    assert bool((byte_offsets % UNIT == 0).all()), "expert blobs are not UNIT-aligned to each other"
    return byte_offsets // UNIT
BLOCK_M, BLOCK_N = 16, 32   # routes and output rows per program
TILE = 8 * GROUP            # inputs per inner step (tuned on the GB300 with ktune.py)
VEC_MAX_M = 16              # up to this many tokens every route is computed on its own (_pvec)
VEC_N, VEC_WARPS = 16, 4    # output rows per program and warps of _pvec


@triton.jit
def _pgemm(a_ptr, base_ptr, table_ptr, c_ptr, sorted_ptr, eid_ptr, npost_ptr, w_ptr,
           N, n_valid, stride_a, stride_c, coff, soff,
           K: tl.constexpr, TOPK: tl.constexpr, MUL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           G: tl.constexpr, T: tl.constexpr, U: tl.constexpr):
    """C[route] = A[route // TOPK] @ W[expert of the block].T for one block of BM routes and BN output rows; with MUL
    the result is also multiplied by the route's weight. Each step unpacks T inputs of BN rows (T/4 code bytes and
    T/G scale bytes per row) into four BN x T/4 tiles, one per position of a code inside its byte."""
    nn = tl.cdiv(N, BN)
    pm = tl.program_id(0) // nn
    pn = tl.program_id(0) % nn
    if pm * BM >= tl.load(npost_ptr):
        return
    route = tl.load(sorted_ptr + pm * BM + tl.arange(0, BM)).to(tl.int64)
    live = route < n_valid
    rows = pn * BN + tl.arange(0, BN)
    rok = rows < N
    blob = base_ptr + tl.load(table_ptr + tl.load(eid_ptr + pm).to(tl.int64)) * U
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
        for k in tl.static_range(4):
            a = tl.load(abase[:, None] + (t * T + 4 * byte + k)[None, :], mask=live[:, None], other=0.0)
            w = ((2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32) * s).to(a.dtype)
            acc = tl.dot(a, tl.trans(w), acc)
    if MUL:
        acc = acc * tl.load(w_ptr + route, mask=live, other=0.0)[:, None]
    tl.store(c_ptr + route[:, None] * stride_c + rows[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=live[:, None] & rok[None, :])


def _launch(a, base, table, c, sorted_ids, expert_ids, npost, w, lay, name, topk, mul, n_valid):
    f = lay[name]
    assert f["inn"] % TILE == 0
    grid = (triton.cdiv(sorted_ids.shape[0], BLOCK_M) * triton.cdiv(f["out"], BLOCK_N),)
    _pgemm[grid](a, base, table, c, sorted_ids, expert_ids, npost, w, f["out"], n_valid, a.stride(0), c.stride(0),
                 f["codes"], f["scale"], K=f["inn"], TOPK=topk, MUL=mul, BM=BLOCK_M, BN=BLOCK_N, G=GROUP, T=TILE,
                 U=UNIT)


@triton.jit
def _pvec(a_ptr, base_ptr, off_ptr, c_ptr, w_ptr, N, stride_a, stride_c, coff, soff,
          K: tl.constexpr, TOPK: tl.constexpr, MUL: tl.constexpr, BN: tl.constexpr, G: tl.constexpr,
          T: tl.constexpr, U: tl.constexpr):
    """C[route] = W[expert of the route] @ A[route // TOPK] for one route and BN output rows, by multiply-adds on
    the unpacked tile (no tensor cores: at one row per expert their operand shuffling costs more than they save).
    off_ptr[route] is the route's table entry: the distance from base to its expert in units of U bytes, or FAR
    for a route this GPU does not serve (its row of C stays as the caller left it: zero)."""
    nn = tl.cdiv(N, BN)
    route = tl.program_id(0) // nn
    off = tl.load(off_ptr + route)
    if off < -(1 << 56):                  # a route served elsewhere: nothing to do here
        return
    rows = (tl.program_id(0) % nn) * BN + tl.arange(0, BN)
    rok = rows < N
    blob = base_ptr + off * U
    cbase = blob + coff + rows.to(tl.int64) * (K // 4)
    sbase = blob + soff + rows.to(tl.int64) * (K // G)
    abase = a_ptr + (route // TOPK).to(tl.int64) * stride_a
    byte = tl.arange(0, T // 4)
    grp = tl.arange(0, T // G)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for t in range(0, K // T):
        b = tl.load(cbase[:, None] + t * (T // 4) + byte[None, :], mask=rok[:, None], other=0).to(tl.int32)
        u = tl.load(sbase[:, None] + t * (T // G) + grp[None, :], mask=rok[:, None], other=0).to(tl.float32)
        s = tl.exp2((u - 200.0) * 0.125)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, T // G, G // 4)), (BN, T // 4))
        for k in tl.static_range(4):
            x = tl.load(abase + t * T + 4 * byte + k).to(tl.float32)
            w = (2 * ((b >> (2 * k)) & 3) - 3).to(tl.float32) * s
            acc += tl.sum(w * x[None, :], axis=1)
    if MUL:
        acc = acc * tl.load(w_ptr + route)
    tl.store(c_ptr + route.to(tl.int64) * stride_c + rows, acc.to(c_ptr.dtype.element_ty), mask=rok)


def _launch_vec(a, base, off, c, w, lay, name, topk, mul):
    f = lay[name]
    assert f["inn"] % TILE == 0
    _pvec[(off.shape[0] * triton.cdiv(f["out"], VEC_N),)](a, base, off, c, w, f["out"], a.stride(0), c.stride(0),
                                                           f["codes"], f["scale"], K=f["inn"], TOPK=topk, MUL=mul,
                                                           BN=VEC_N, G=GROUP, T=TILE, U=UNIT, num_warps=VEC_WARPS)


@triton.jit
def _situ(g_ptr, u_ptr, o_ptr, n, beta, linear_beta, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n
    g = tl.load(g_ptr + i, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + i, mask=ok, other=0.0).to(tl.float32)
    tg = 2.0 * tl.sigmoid(2.0 * g / beta) - 1.0                 # tanh(g / beta)
    tu = 2.0 * tl.sigmoid(2.0 * u / linear_beta) - 1.0          # tanh(u / linear_beta)
    tl.store(o_ptr + i, (beta * tg * tl.sigmoid(g) * linear_beta * tu).to(o_ptr.dtype.element_ty), mask=ok)


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Kimi's SiTU: beta tanh(gate / beta) sigmoid(gate) * linear_beta tanh(up / linear_beta), computed in float32,
    in one pass without temporaries (a long prompt has 131,072 routes x 3,072 values here)."""
    out = torch.empty_like(gate)
    if gate.is_cuda:
        _situ[(triton.cdiv(gate.numel(), 4096),)](gate, up, out, gate.numel(), beta, linear_beta, BLOCK=4096)
    else:
        g, u = gate.float(), up.float()
        out.copy_(beta * torch.tanh(g / beta) * torch.sigmoid(g) * (linear_beta * torch.tanh(u / linear_beta)))
    return out


def packed_experts(x, topk_ids, topk_weights, base, table, own, lay, beta=4.0, linear_beta=25.0, grouped=None):
    """x [M, latent] bf16; topk_ids int32 [M, k]; topk_weights float32 [M, k]; base uint8 tensor; table int64 [E];
    own int32 [E] (e if this GPU holds expert e, else -1). Returns float32 [M, latent].

    Decode-sized calls (M <= VEC_MAX_M) compute every route on its own; larger ones group the routes by expert
    (vLLM's moe_align_block_size) and use tensor cores, so an expert's bytes are read once per BLOCK_M tokens."""
    M, k = topk_ids.shape
    R = M * k
    inter, latent = lay["w1"]["out"], lay["w1"]["inn"]
    h = torch.zeros((2, R, inter), dtype=x.dtype, device=x.device)
    out = torch.zeros((R, latent), dtype=torch.float32, device=x.device)
    wflat = topk_weights.reshape(-1)
    if grouped is None:
        grouped = M > VEC_MAX_M
    if grouped:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        sorted_ids, expert_ids, npost = moe_align_block_size(topk_ids, BLOCK_M, own.numel(), own,
                                                             ignore_invalid_experts=True)
        _launch(x, base, table, h[0], sorted_ids, expert_ids, npost, wflat, lay, "w1", k, False, R)
        _launch(x, base, table, h[1], sorted_ids, expert_ids, npost, wflat, lay, "w3", k, False, R)
        mid = situ(h[0], h[1], beta, linear_beta)
        _launch(mid, base, table, out, sorted_ids, expert_ids, npost, wflat, lay, "w2", 1, True, R)
    else:
        off = table[topk_ids.reshape(-1).long()]
        _launch_vec(x, base, off, h[0], wflat, lay, "w1", k, False)
        _launch_vec(x, base, off, h[1], wflat, lay, "w3", k, False)
        mid = situ(h[0], h[1], beta, linear_beta)
        _launch_vec(mid, base, off, out, wflat, lay, "w2", 1, True)
    return out.view(M, k, latent).sum(1)


def reference(x, topk_ids, topk_weights, blob_of, own, lay, beta=4.0, linear_beta=25.0):
    """The same sum from unpacked float32 matrices, one route at a time (slow; for checks). blob_of(e) -> uint8
    [BLOB] on x's device."""
    from kpack import from_blob, unpack
    out = torch.zeros((x.shape[0], lay["w1"]["inn"]), dtype=torch.float32, device=x.device)
    cache = {}
    for m in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[m, j])
            if int(own[e]) < 0:
                continue
            if e not in cache:
                cache[e] = [unpack(*from_blob(blob_of(e), lay, n)) for n in ("w1", "w2", "w3")]
            w1, w2, w3 = cache[e]
            z = x[m].float()
            out[m] += topk_weights[m, j] * (w2 @ situ(w1 @ z, w3 @ z, beta, linear_beta))
    return out
