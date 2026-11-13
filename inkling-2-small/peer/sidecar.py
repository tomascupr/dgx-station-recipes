#!/usr/bin/env python3
"""RTX sidecar for Inkling 2 Small cold experts.

Polls /dev/shm/inkling2-peer. Computes with the numpy reference. If
peer_inkling_rtx_sm120.so is present it is reported and not called: that
kernel is UNVALIDATED and is not on the serving path until a DGX smoke
compares it with kernels/reference/quant.py.

    sidecar.py --segment PATH --placement MAP.json --mock-request REQ.json --out Y.npy

--mock-request serves one published segment and exits. It does not open a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import QUANT_MXFP8, QUANT_NVFP4, Layout, Segment  # noqa: E402
from split import apply_tier  # noqa: E402

KIND = {QUANT_NVFP4: "nvfp4", QUANT_MXFP8: "mxfp8"}


def answer(placement: dict, banks: dict, n_experts: int, layer: int, quant: int,
           x: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    kind = KIND.get(quant)
    if kind is None:
        raise RuntimeError(f"unknown quant code {quant}")
    row = placement["layers"][str(layer)]
    bank = banks[layer]
    return apply_tier(x, ids, weights, row["cold"], bank["w13"], bank["w2"], kind, n_experts)


def serve_once(seg: Segment, placement: dict, banks: dict, n_experts: int) -> int:
    seq, layer, quant, x, ids, weights = seg.read_request()
    if seq == 0 or seq == seg.resp():
        return 0
    y = answer(placement, banks, n_experts, layer, quant, x, ids, weights)
    seg.publish_y(seq, y)
    return seq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", required=True)
    p.add_argument("--placement", required=True)
    p.add_argument("--banks")
    p.add_argument("--hidden", type=int, default=4608)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--n-experts", type=int, default=384)
    p.add_argument("--mock-request", action="store_true",
                   help="serve the one request already in the segment, then exit")
    p.add_argument("--so", default="")
    a = p.parse_args()
    so = a.so or os.environ.get("INKLING2_RTX_SO", "")
    if so and os.path.exists(so):
        print(f"UNVALIDATED kernel present at {so}; the sidecar keeps the numpy reference", flush=True)
    placement = json.load(open(a.placement))
    banks = np.load(a.banks, allow_pickle=True).item() if a.banks else {}
    layout = Layout(a.hidden, a.max_tokens, a.topk)
    seg = Segment(a.segment, layout, create=False)
    if a.mock_request:
        seq = serve_once(seg, placement, banks, a.n_experts)
        print(f"served seq {seq}", flush=True)
        seg.close()
        return
    print("INKLING2_SIDECAR polling; numpy reference; no GPU kernel", flush=True)
    while True:
        if seg.req() != seg.resp():
            serve_once(seg, placement, banks, a.n_experts)
        time.sleep(0.0005)


if __name__ == "__main__":
    main()
