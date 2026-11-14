"""Device diff of the serving GEMMs against the fp64 kernels in the same .so.

Runs on the DGX. There is no GPU in the authoring environment, so
``--thresholds`` only prints the pass bar and exits.

For each of GB300 and RTX, both formats, and 1, 8, 64 and 128 tokens all
routed to one expert: load experts 0 and 1 from a real NVFP4 layer and a
real MXFP8 layer, run the serving kernel and the fp64 reference, and judge
with peer.thresholds. On the GB300 the same cases also run one expert whose
packed bytes started in host memory: the copy engine stages it, the GEMM
reads the staged device pointer, and the log prints that copy's TB/s.
128 tokens is a direct launch. The peer segment itself holds 64.
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


def _spill_cases(peer, model_dir: str) -> None:
    """One host-resident expert, copied by the DMA engine, then the GEMM."""
    import time

    from spill_pool import SpillPool
    for kind, layer in LAYERS:
        bank = load_members(model_dir, layer, list(EXPERTS), kind)
        one = _one_expert_bank(bank, EXPERTS[TARGET])
        for tokens in COUNTS:
            pool = SpillPool(lambda _layer: None, max(one["w13"].nbytes * 4, 1 << 20))
            t0 = time.perf_counter()
            queued = pool.prefetch(layer, [EXPERTS[TARGET]], one)
            pool.sync_copy()
            copy_s = time.perf_counter() - t0
            view = pool.pointers(layer, EXPERTS[TARGET])
            row = _run_ptrs(peer, one, tokens, view)
            status = "PASS" if row["pass"] else "FAIL"
            tb = (queued / copy_s / 1e12) if copy_s else 0.0
            print(
                f"  {status} spill {pool.mode} {kind} layer {layer} tokens/expert {tokens} "
                f"max_abs {row['max_abs']:.3e} limit {row['limit_abs']:.3e} "
                f"measured copy {copy_s * 1e6:.1f} us  {tb:.4f} TB/s "
                f"({queued} bytes from Grace, including the pin on a miss; "
                f"plan figure 0.355 TB/s is the plan's measurement)",
                flush=True)
            if not row["pass"]:
                raise SystemExit("spill path failed the fp64 diff")


def _run_ptrs(peer, bank: dict, tokens: int, view: dict) -> dict:
    """GEMM and fp64 reference both read the staged device expert."""
    import cudart
    kind = bank["kind"]
    rng = np.random.default_rng(10_000 + tokens)
    x = f32_to_bf16_f32(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
    ids = np.zeros((tokens, 1), dtype=np.int32)
    route = np.full((tokens, 1), 0.5, dtype=np.float64)
    xq = quantize_activation(x, kind)
    x_b = _f32_to_bf16(x.astype(np.float32)).tobytes()
    dx = cudart.alloc(len(x_b))
    dids = cudart.alloc(ids.nbytes)
    droute = cudart.alloc(tokens * 4)
    dy = cudart.alloc(tokens * HIDDEN * 2)
    drefx = cudart.alloc(tokens * HIDDEN * 8)
    dref_route = cudart.alloc(route.nbytes)
    dscratch = cudart.alloc(tokens * (INTER * 2 + INTER) * 8)
    drefy = cudart.alloc(tokens * HIDDEN * 8)
    da13 = da2 = 0
    ptrs = {k: np.array([view[k]], dtype=np.uint64) for k in ("w13", "s13", "w2", "s2")}
    try:
        cudart.h2d(x_b, dx)
        cudart.h2d(ids.tobytes(), dids)
        cudart.h2d(np.ascontiguousarray(route, dtype=np.float32).tobytes(), droute)
        cudart.h2d(np.ascontiguousarray(xq, dtype=np.float64).tobytes(), drefx)
        cudart.h2d(np.ascontiguousarray(route, dtype=np.float64).tobytes(), dref_route)
        cudart.memset0(dy, tokens * HIDDEN * 2)
        cudart.memset0(drefy, tokens * HIDDEN * 8)
        if kind == "nvfp4":
            da13 = cudart.alloc(4)
            da2 = cudart.alloc(4)
            cudart.h2d(np.ascontiguousarray(bank["amax13"][:1]).tobytes(), da13)
            cudart.h2d(np.ascontiguousarray(bank["amax2"][:1]).tobytes(), da2)
            denom = float(bank["denom13"])
        else:
            denom = 1.0
        peer.expert_each(
            kind, tokens, 1, 1, HIDDEN, INTER, dx, dids, droute,
            ptrs["w13"].ctypes.data, ptrs["s13"].ctypes.data, da13,
            ptrs["w2"].ctypes.data, ptrs["s2"].ctypes.data, da2, denom, dy, None)
        cudart.sync()
        peer.reference(
            kind, tokens, 1, HIDDEN, INTER, drefx, dids, dref_route,
            view["w13"], view["s13"], da13, view["w2"], view["s2"], da2, denom, dscratch, drefy)
        cudart.sync()
        got = _bf16_to_f32(cudart.d2h(dy, tokens * HIDDEN * 2)).reshape(tokens, HIDDEN)
        ref = np.frombuffer(cudart.d2h(drefy, tokens * HIDDEN * 8), dtype=np.float64).reshape(tokens, HIDDEN).copy()
    finally:
        for ptr in (dx, dids, droute, dy, drefx, dref_route, dscratch, drefy, da13, da2):
            cudart.free(ptr)
    return judge(ref, got)


def _one_expert_bank(bank: dict, expert: int) -> dict:
    local = bank["members"].index(expert)
    out = {"kind": bank["kind"], "members": [expert], "n": 1}
    for key in ("w13", "s13", "w2", "s2"):
        span = bank[key].nbytes // bank["n"]
        out[key] = np.ascontiguousarray(bank[key][local * span:(local + 1) * span])
    if bank["kind"] == "nvfp4":
        out["amax13"] = bank["amax13"][local:local + 1]
        out["amax2"] = bank["amax2"][local:local + 1]
        out["denom13"] = bank["denom13"]
        out["denom2"] = bank["denom2"]
    return out


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
        if "GB300" in name:
            _spill_cases(peer, a.model_dir)
    if failed:
        raise SystemExit(f"{failed} cases failed")
    print("validate: all cases passed", flush=True)


if __name__ == "__main__":
    main()
