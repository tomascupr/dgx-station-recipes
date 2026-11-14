"""Device diff of the serving GEMMs against the fp64 kernels in the same .so.

Runs on the DGX. There is no GPU in the authoring environment, so
``--thresholds`` only prints the pass bar and exits.

For each of GB300 and RTX, both formats, and 1, 8, 64 and 128 tokens all
routed to one expert: load experts 0 and 1 from a real NVFP4 layer and a
real MXFP8 layer, run the serving kernel and the fp64 reference, and judge
with peer.thresholds. 128 tokens is a direct launch. The peer segment itself
holds 64.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "kernels", "reference"))

from bank import HIDDEN, INTER, load_members  # noqa: E402
from quant import quantize_activation  # noqa: E402
from segment import _bf16_to_f32, _f32_to_bf16  # noqa: E402
from thresholds import judge  # noqa: E402

LAYERS = (("nvfp4", 0), ("mxfp8", 40))
COUNTS = (1, 8, 64, 128)
EXPERTS = (0, 1)
TARGET = 1  # every token hits the second loaded expert, so the stride is real


def f32_to_bf16_f32(x: np.ndarray) -> np.ndarray:
    raw = _f32_to_bf16(np.asarray(x, dtype=np.float32))
    return _bf16_to_f32(raw.tobytes())


def gpus() -> list[tuple[int, str]]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], text=True)
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        index, name = line.split(", ", 1)
        if "GB300" in name or "RTX" in name:
            out.append((int(index), name.strip()))
    return out


def so_for(name: str, root: str) -> tuple[str, str]:
    if "GB300" in name:
        return os.path.join(root, "peer", "prebuilt", "peer_inkling_gb300_sm103.so"), "sm_103a"
    return os.path.join(root, "peer", "prebuilt", "peer_inkling_rtx_sm120.so"), "sm_120a"


def touched_bytes(bank: dict, tokens: int) -> int:
    one = bank["n"]
    return (bank["w13"].nbytes + bank["s13"].nbytes + bank["w2"].nbytes + bank["s2"].nbytes) // one \
        + tokens * HIDDEN * 2 * 2


def run_case(peer, bank: dict, tokens: int) -> dict:
    import cudart
    kind = bank["kind"]
    rng = np.random.default_rng(tokens + (0 if kind == "nvfp4" else 1000))
    x = f32_to_bf16_f32(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
    ids = np.full((tokens, 1), TARGET, dtype=np.int32)
    route = np.full((tokens, 1), 0.5, dtype=np.float64)
    xq = quantize_activation(x, kind)
    cudart.set_device(int(os.environ.get("INKLING2_VALIDATE_DEVICE", "0")))
    x_b = _f32_to_bf16(x.astype(np.float32)).tobytes()
    dx = cudart.alloc(len(x_b))
    dids = cudart.alloc(ids.nbytes)
    droute = cudart.alloc(tokens * 4)
    dw13 = cudart.alloc(bank["w13"].nbytes)
    ds13 = cudart.alloc(bank["s13"].nbytes)
    dw2 = cudart.alloc(bank["w2"].nbytes)
    ds2 = cudart.alloc(bank["s2"].nbytes)
    dy = cudart.alloc(tokens * HIDDEN * 2)
    drefx = cudart.alloc(tokens * HIDDEN * 8)
    dref_ids = cudart.alloc(ids.nbytes)
    dref_route = cudart.alloc(route.nbytes)
    dscratch = cudart.alloc(tokens * (INTER * 2 + INTER) * 8)
    drefy = cudart.alloc(tokens * HIDDEN * 8)
    da13 = da2 = 0
    try:
        cudart.h2d(x_b, dx)
        cudart.h2d(ids.tobytes(), dids)
        cudart.h2d(np.ascontiguousarray(route, dtype=np.float32).tobytes(), droute)
        cudart.h2d(np.ascontiguousarray(xq, dtype=np.float64).tobytes(), drefx)
        cudart.h2d(ids.tobytes(), dref_ids)
        cudart.h2d(np.ascontiguousarray(route, dtype=np.float64).tobytes(), dref_route)
        cudart.h2d(bank["w13"].tobytes(), dw13)
        cudart.h2d(bank["s13"].tobytes(), ds13)
        cudart.h2d(bank["w2"].tobytes(), dw2)
        cudart.h2d(bank["s2"].tobytes(), ds2)
        if kind == "nvfp4":
            da13 = cudart.alloc(bank["amax13"].nbytes)
            da2 = cudart.alloc(bank["amax2"].nbytes)
            cudart.h2d(np.ascontiguousarray(bank["amax13"]).tobytes(), da13)
            cudart.h2d(np.ascontiguousarray(bank["amax2"]).tobytes(), da2)
            denom = float(bank["denom13"])
        else:
            denom = 1.0
        cudart.memset0(dy, tokens * HIDDEN * 2)
        cudart.memset0(drefy, tokens * HIDDEN * 8)
        start, end = cudart.Event(), cudart.Event()
        start.record()
        peer.expert(kind, tokens, 1, bank["n"], HIDDEN, INTER, dx, dids, droute,
                    dw13, ds13, da13, dw2, ds2, da2, denom, dy, None)
        end.record()
        end.sync()
        ms = cudart.elapsed_ms(start, end)
        start.close()
        end.close()
        peer.reference(kind, tokens, 1, HIDDEN, INTER, drefx, dref_ids, dref_route,
                       dw13, ds13, da13, dw2, ds2, da2, denom, dscratch, drefy)
        cudart.sync()
        got = _bf16_to_f32(cudart.d2h(dy, tokens * HIDDEN * 2)).reshape(tokens, HIDDEN)
        ref = np.frombuffer(cudart.d2h(drefy, tokens * HIDDEN * 8), dtype=np.float64).reshape(tokens, HIDDEN).copy()
    finally:
        for ptr in (dx, dids, droute, dw13, ds13, dw2, ds2, dy, drefx, dref_ids, dref_route,
                    dscratch, drefy, da13, da2):
            cudart.free(ptr)
    result = judge(ref, got)
    seconds = ms / 1e3
    result["us"] = ms * 1e3
    result["tb_s"] = (touched_bytes(bank, tokens) / seconds / 1e12) if seconds else 0.0
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=os.environ.get("INKLING2_MODEL_DIR", "/data/models/Inkling-2-Small"))
    p.add_argument("--thresholds", action="store_true", help="print the pass bar and exit")
    a = p.parse_args()
    if a.thresholds:
        print("pass if max_abs <= 0.025 * max(peak|ref|, 1)")
        print("bf16 ulp is 1/128; three bf16 stores plus fp32-vs-fp64 accumulation of K=4608")
        return
    root = os.path.abspath(os.path.join(HERE, ".."))
    devices = gpus()
    if not devices:
        raise SystemExit("no GB300 or RTX visible")
    failed = 0
    for index, name in devices:
        os.environ["INKLING2_VALIDATE_DEVICE"] = str(index)
        path, arch = so_for(name, root)
        from so import open_so
        peer = open_so(path)
        import cudart
        cudart.set_device(index)
        print(f"device {index} {name} {arch} {path}", flush=True)
        for kind, layer in LAYERS:
            bank = load_members(a.model_dir, layer, list(EXPERTS), kind)
            print(f"  stored scales {kind} layer {layer}: "
                  f"s13 {bank['s13'].size} bytes, s2 {bank['s2'].size} bytes "
                  f"(one scale per row, {16 if kind == 'nvfp4' else 32} along K)", flush=True)
            for tokens in COUNTS:
                row = run_case(peer, bank, tokens)
                status = "PASS" if row["pass"] else "FAIL"
                if not row["pass"]:
                    failed += 1
                print(
                    f"  {status} {kind} layer {layer} tokens/expert {tokens} "
                    f"max_abs {row['max_abs']:.3e} limit {row['limit_abs']:.3e} "
                    f"max_rel_vs_peak {row['max_rel_vs_peak']:.3e} "
                    f"measured {row['us']:.1f} us  {row['tb_s']:.4f} TB/s "
                    f"(bytes: one expert's packed weights and scales, plus bf16 x and y; "
                    f"time includes the launch's workspace alloc)",
                    flush=True)
    if failed:
        raise SystemExit(f"{failed} cases failed")
    print("validate: all cases passed", flush=True)


if __name__ == "__main__":
    main()
