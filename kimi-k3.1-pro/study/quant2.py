"""Candidate low-bit quantizers for Kimi K3.1 Pro's routed experts, for qlab.py.

Each takes one expert matrix w [out, in] (float32, decoded MXFP4) and returns its de-quantized approximation.
Scales are per group of `g` consecutive inputs within a row, as in the published MXFP4 (g = 32).
BITS gives the storage cost in bits per weight, counting 8 bits per stored scale.
"""
import torch

from variants import _requant


def uniform(w, levels, g=32):
    return _requant(w, levels, g)


def lloyd(w, mags, g=32, iters=8, colw=None):
    """Symmetric codebook with `mags` free magnitudes per group (plus zero when `mags` is odd-coded, see below).

    mags=2: levels {-b, -a, +a, +b} (2 bits). mags=1 with zero: {-a, 0, +a} is `lloyd(w, 1)` -> ternary (1.58 bits).
    mags=3 with zero: {0, +-a, +-b, +-c} would be 7 levels; not used here.
    colw [in]: optional importance of each input (mean squared activation); centroids become weighted means.
    """
    x = w.float().reshape(-1, g)
    m = x.abs()
    cw = None if colw is None else colw.float().reshape(1, -1).expand(w.shape[0], -1).reshape(-1, g).clamp_min(1e-12)
    srt = m.sort(-1).values
    if mags == 1:                                   # ternary: threshold t, magnitude a
        a = srt[:, g // 2:].mean(-1, keepdim=True)
        for _ in range(iters):
            on = m > a / 2
            if cw is None:
                a = (m * on).sum(-1, keepdim=True) / on.sum(-1, keepdim=True).clamp_min(1)
            else:
                a = (m * on * cw).sum(-1, keepdim=True) / (on * cw).sum(-1, keepdim=True).clamp_min(1e-12)
        q = torch.where(m > a / 2, a, torch.zeros_like(a))
    else:
        a = srt[:, : g // 2].mean(-1, keepdim=True)
        b = srt[:, g // 2:].mean(-1, keepdim=True)
        for _ in range(iters):
            hi = m > (a + b) / 2
            lo = ~hi
            if cw is None:
                b = torch.where(hi.any(-1, keepdim=True), (m * hi).sum(-1, keepdim=True) / hi.sum(-1, keepdim=True).clamp_min(1), b)
                a = torch.where(lo.any(-1, keepdim=True), (m * lo).sum(-1, keepdim=True) / lo.sum(-1, keepdim=True).clamp_min(1), a)
            else:
                b = torch.where(hi.any(-1, keepdim=True), (m * hi * cw).sum(-1, keepdim=True) / (hi * cw).sum(-1, keepdim=True).clamp_min(1e-12), b)
                a = torch.where(lo.any(-1, keepdim=True), (m * lo * cw).sum(-1, keepdim=True) / (lo * cw).sum(-1, keepdim=True).clamp_min(1e-12), a)
        q = torch.where(m > (a + b) / 2, b, a)
    return (q * x.sign()).reshape(w.shape)


def row_gain(w, q):
    """Rescale each output row of q by the least-squares gain towards w (one extra scale per row: free)."""
    g = (w.float() * q).sum(-1, keepdim=True) / (q * q).sum(-1, keepdim=True).clamp_min(1e-30)
    return q * g


# name -> (bits per weight, function(w, colw) -> q)
CANDIDATES = {
    "q3-uniform-g32":   (3.25, lambda w, c: uniform(w, 8)),
    "q2-uniform-g32":   (2.25, lambda w, c: uniform(w, 4)),
    "q2-uniform-gain":  (2.25, lambda w, c: row_gain(w, uniform(w, 4))),
    "q2-lloyd-g32":     (2.50, lambda w, c: lloyd(w, 2, 32)),
    "q2-lloyd-g64":     (2.25, lambda w, c: lloyd(w, 2, 64)),
    "q2-lloyd-g128":    (2.125, lambda w, c: lloyd(w, 2, 128)),
    "q2-lloyd-g64-act": (2.25, lambda w, c: lloyd(w, 2, 64, colw=c)),
    "t-uniform-g32":    (1.83, lambda w, c: uniform(w, 3)),
    "t-lloyd-g32":      (1.83, lambda w, c: lloyd(w, 1, 32)),
    "t-lloyd-g64":      (1.71, lambda w, c: lloyd(w, 1, 64)),
    "t-lloyd-g64-act":  (1.71, lambda w, c: lloyd(w, 1, 64, colw=c)),
}


def levels2(w, g=32, how="uniform"):
    """Per group the two magnitudes (a < b) of a 2-bit codebook {-b, -a, +a, +b}: [rows, in/g] each."""
    q = (uniform(w, 4, g) if how == "uniform" else lloyd(w, 2, g)).abs().reshape(w.shape[0], -1, g)
    a, b = q.amin(-1), q.amax(-1)
    return a, torch.where(b > a, b, 3 * a)


def gptq(w, H, a, b, g=32, block=128, damp=0.01):
    """GPTQ/OBQ rounding of w [out, in] to the 2-bit codebooks (a, b per group) under the input Hessian H [in, in].

    Columns are rounded left to right; each column's rounding error is pushed onto the columns not yet rounded,
    weighted by the inverse Hessian, so that the layer's output (not the weights) stays close.
    """
    W = w.float().clone()
    n = W.shape[1]
    H = H.float().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.empty_like(W)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        Err = torch.empty((W.shape[0], i1 - i0), device=W.device)
        for j in range(i0, i1):
            x = W[:, j]
            aj, bj = a[:, j // g], b[:, j // g]
            q = torch.where(x.abs() > (aj + bj) / 2, bj, aj) * torch.where(x < 0, -1.0, 1.0)
            Q[:, j] = q
            e = (x - q) / U[j, j]
            W[:, j + 1:i1] -= e.unsqueeze(1) * U[j, j + 1:i1].unsqueeze(0)
            Err[:, j - i0] = e
        W[:, i1:] -= Err @ U[i0:i1, i1:]
    return Q
