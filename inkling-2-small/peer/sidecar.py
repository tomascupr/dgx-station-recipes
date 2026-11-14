#!/usr/bin/env python3
"""RTX sidecar for Inkling 2 Small cold experts.

Polls /dev/shm/inkling2-peer. Cold experts run on the sm_120a GEMM only
when INKLING2_ALLOW_UNVALIDATED=1 and the .so is present. Otherwise the
numpy reference answers, which is what the CPU tests use.

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


def answer_gpu(so, placement: dict, device_banks: dict, layer: int, quant: int,
               x: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    import torch

    from bank import local_ids

    kind = KIND.get(quant)
    if kind is None:
        raise RuntimeError(f"unknown quant code {quant}")
    row = placement["layers"][str(layer)]
    bank = device_banks[layer]
    local = local_ids(ids, row["cold"])
    tokens = x.shape[0]
    x_t = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device="cuda").to(torch.bfloat16)
    ids_t = torch.as_tensor(np.ascontiguousarray(local), device="cuda")
    route = torch.as_tensor(np.ascontiguousarray(weights, dtype=np.float32), device="cuda")
    y = torch.zeros(tokens, x.shape[1], dtype=torch.bfloat16, device="cuda")
    if bank["n"]:
        so.expert(
            kind, tokens, local.shape[1], int(bank["n"]), int(x.shape[1]), 2304,
            x_t.data_ptr(), ids_t.data_ptr(), route.data_ptr(),
            bank["w13"].data_ptr(), bank["s13"].data_ptr(),
            bank["amax13"].data_ptr() if kind == "nvfp4" else 0,
            bank["w2"].data_ptr(), bank["s2"].data_ptr(),
            bank["amax2"].data_ptr() if kind == "nvfp4" else 0,
            float(bank.get("denom13", 1.0)), y.data_ptr(), None)
    return y.float().cpu().numpy()


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
    p.add_argument("--model-dir", default=os.environ.get("INKLING2_MODEL_DIR", ""))
    a = p.parse_args()
    so_path = a.so or os.environ.get("INKLING2_RTX_SO", "")
    allow = os.environ.get("INKLING2_ALLOW_UNVALIDATED") == "1"
    placement = json.load(open(a.placement))
    layout = Layout(a.hidden, a.max_tokens, a.topk)
    seg = Segment(a.segment, layout, create=False)
    use_gpu = bool(allow and so_path and os.path.exists(so_path))
    if so_path and os.path.exists(so_path) and not allow:
        print(f"UNVALIDATED kernel present at {so_path}; the sidecar keeps the numpy reference", flush=True)
    if use_gpu:
        from gpus import find
        chosen = find("rtx")
        # Before the runtime initializes. Ordinal 0 on this host is only the
        # RTX when this mask is the RTX UUID.
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen["uuid"]
        import torch
        visible = torch.cuda.get_device_name(0)
        if "RTX PRO 6000" not in visible:
            raise SystemExit(f"sidecar is on {visible}, not {chosen['name']} {chosen['uuid']}")
        print(f"sidecar GPU {visible} {chosen['uuid']}", flush=True)
        from bank import load_members
        from so import open_so
        peer = open_so(so_path)
        device_banks = {}
        for layer, row in placement["layers"].items():
            if not row["cold"]:
                device_banks[int(layer)] = {"n": 0, "kind": row["quant"]}
                continue
            host = load_members(a.model_dir, int(layer), row["cold"], row["quant"])
            device_banks[int(layer)] = {
                k: (torch.as_tensor(np.ascontiguousarray(v), device="cuda") if isinstance(v, np.ndarray) else v)
                for k, v in host.items()
            }
            print(f"RTX cold layer {layer}: {host['n']} experts", flush=True)

        def serve_gpu() -> int:
            seq, layer, quant, x, ids, weights = seg.read_request()
            if seq == 0 or seq == seg.resp():
                return 0
            y = answer_gpu(peer, placement, device_banks, layer, quant, x, ids, weights)
            seg.publish_y(seq, y)
            return seq

        print("INKLING2_SIDECAR polling; sm_120a GEMM; UNVALIDATED", flush=True)
        if a.mock_request:
            print(f"served seq {serve_gpu()}", flush=True)
            seg.close()
            return
        while True:
            if seg.req() != seg.resp():
                serve_gpu()
            time.sleep(0.0005)
    banks = np.load(a.banks, allow_pickle=True).item() if a.banks else {}
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
