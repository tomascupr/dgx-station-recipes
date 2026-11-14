"""Place each expert on the GB300 (hot), the RTX (cold) or Grace (spill).

Default prior, from the 14 Nov plan: every MXFP8 expert stays on the GB300,
and NVFP4 expert ids are split uniformly (lowest ids hottest) until a
frequency profile says otherwise. A profile reorders NVFP4 only. MXFP8 is
never moved to the RTX or to Grace.

    plan_experts.py --from-hf --out placement.json
    plan_experts.py --index INDEX --headers-dir DIR --out placement.json [--profile counts.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from headers import fetch_header, header_from_file, list_safetensors, resolve_url  # noqa: E402
from layout import N_EXPERTS, assert_uniform, classify  # noqa: E402

GIB = 1 << 30
REPO = "thinkingmachines/Inkling-2-Small"
REV = "8361e7a31ae7e9966a49a91d3994fde618936c2b"
# Preferred prior inside the plan's 200-201 GiB GB300 band. 200.81 GiB divides
# to 160 NVFP4 ids, not the 158 the plan writes down; 199.425 GiB is the byte
# total of all MXFP8 plus 158 NVFP4 ids. RTX 87 GiB divides to 130.
DEFAULT_GB300_GIB = 199.425
DEFAULT_RTX_GIB = 87.0


def load_profile(path: str | None) -> dict[int, list[int]] | None:
    if path is None:
        return None
    raw = json.load(open(path))
    counts = raw["counts"] if isinstance(raw, dict) and "counts" in raw else raw
    out = {}
    for key, row in counts.items():
        layer = int(key)
        if len(row) != N_EXPERTS:
            raise SystemExit(f"profile layer {layer} has {len(row)} counts, expected {N_EXPERTS}")
        out[layer] = [int(v) for v in row]
    return out


def _fit(budget: int, column: int) -> int:
    if column <= 0:
        raise SystemExit("per-expert column is empty")
    if budget < 0:
        raise SystemExit("byte budget is negative")
    return budget // column


def place(layers: dict, sizes: dict[str, int], gb300_bytes: int, rtx_bytes: int,
          profile: dict[int, list[int]] | None) -> dict:
    nv = sizes.get("nvfp4")
    mx = sizes.get("mxfp8")
    if nv is None or mx is None:
        raise SystemExit(f"need both quant groups, got {sorted(sizes)}")
    nv_layers = [i for i in sorted(layers) if layers[i].quant == "nvfp4"]
    mx_layers = [i for i in sorted(layers) if layers[i].quant == "mxfp8"]
    mx_bytes = mx * N_EXPERTS * len(mx_layers)
    if mx_bytes > gb300_bytes:
        raise SystemExit(
            f"MXFP8 experts are {mx_bytes} bytes and do not fit the GB300 budget "
            f"{gb300_bytes}. This recipe does not spill MXFP8.")
    nv_column = nv * len(nv_layers)
    n_hot = _fit(gb300_bytes - mx_bytes, nv_column)
    n_cold = _fit(rtx_bytes, nv_column)
    if n_hot > N_EXPERTS:
        n_hot = N_EXPERTS
    if n_hot + n_cold > N_EXPERTS:
        n_cold = N_EXPERTS - n_hot
    n_spill = N_EXPERTS - n_hot - n_cold

    def order_for(layer: int) -> list[int]:
        if profile is None:
            return list(range(N_EXPERTS))
        if layer not in profile:
            raise SystemExit(f"profile has no counts for NVFP4 layer {layer}")
        counts = profile[layer]
        return sorted(range(N_EXPERTS), key=lambda e: (-counts[e], e))

    out_layers = {}
    for layer in nv_layers:
        order = order_for(layer)
        hot = sorted(order[:n_hot])
        cold = sorted(order[n_hot:n_hot + n_cold])
        spill = sorted(order[n_hot + n_cold:])
        _check_partition(layer, hot, cold, spill)
        out_layers[str(layer)] = {"quant": "nvfp4", "hot": hot, "cold": cold, "spill": spill}
    for layer in mx_layers:
        out_layers[str(layer)] = {
            "quant": "mxfp8", "hot": list(range(N_EXPERTS)), "cold": [], "spill": [],
        }
    gb300 = mx_bytes + n_hot * nv_column
    rtx = n_cold * nv_column
    grace = n_spill * nv_column
    return {
        "schema": "inkling2-placement-v1",
        "layers": {k: out_layers[k] for k in sorted(out_layers, key=int)},
        "meta": {
            "model": f"{REPO}@{REV}",
            "nvfp4_layers": [nv_layers[0], nv_layers[-1]] if nv_layers else [],
            "mxfp8_layers": [mx_layers[0], mx_layers[-1]] if mx_layers else [],
            "n_experts": N_EXPERTS,
            "per_expert_bytes": {"nvfp4": nv, "mxfp8": mx},
            "counts": {
                "gb300_nvfp4": n_hot, "rtx_nvfp4": n_cold, "grace_nvfp4": n_spill,
                "gb300_mxfp8": N_EXPERTS,
            },
            "bytes": {"gb300": gb300, "rtx": rtx, "grace": grace, "mxfp8": mx_bytes},
            "budgets": {"gb300": gb300_bytes, "rtx": rtx_bytes},
            "profile": "uniform-id-prior" if profile is None else "frequency",
            "mxfp8_policy": "all on GB300",
        },
    }


def _check_partition(layer: int, hot: list[int], cold: list[int], spill: list[int]) -> None:
    got = hot + cold + spill
    if sorted(got) != list(range(N_EXPERTS)) or len(got) != N_EXPERTS:
        raise SystemExit(f"layer {layer} placement is not a partition of 0..{N_EXPERTS - 1}")


def load_local(index_path: str, headers_dir: str) -> dict[str, dict]:
    index = json.load(open(index_path))
    weight_map = index["weight_map"] if "weight_map" in index else index
    names = sorted(set(weight_map.values()))
    headers = {}
    for name in names:
        base = os.path.basename(name)
        json_path = os.path.join(headers_dir, base + ".header.json")
        raw_path = os.path.join(headers_dir, base)
        if os.path.exists(json_path):
            headers[name] = json.load(open(json_path))
        elif os.path.exists(raw_path):
            headers[name] = header_from_file(raw_path)
        else:
            raise SystemExit(f"no header for {name} under {headers_dir}")
    extra = os.path.join(headers_dir, "mtp.safetensors.header.json")
    if os.path.exists(extra):
        headers["mtp.safetensors"] = json.load(open(extra))
    return headers


def load_hf(repo: str, rev: str, cache: str | None) -> dict[str, dict]:
    names = list_safetensors(repo, rev)
    headers = {}
    for name in names:
        url = resolve_url(repo, rev, name)
        header = fetch_header(url)
        headers[name] = header
        if cache:
            os.makedirs(cache, exist_ok=True)
            with open(os.path.join(cache, os.path.basename(name) + ".header.json"), "w") as f:
                json.dump(header, f)
    return headers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-hf", action="store_true")
    p.add_argument("--repo", default=REPO)
    p.add_argument("--revision", default=REV)
    p.add_argument("--index")
    p.add_argument("--headers-dir")
    p.add_argument("--profile")
    p.add_argument("--gb300-expert-gib", type=float, default=DEFAULT_GB300_GIB)
    p.add_argument("--rtx-expert-gib", type=float, default=DEFAULT_RTX_GIB)
    p.add_argument("--cache")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.from_hf:
        headers = load_hf(a.repo, a.revision, a.cache)
    else:
        if not a.index or not a.headers_dir:
            raise SystemExit("need --from-hf or both --index and --headers-dir")
        headers = load_local(a.index, a.headers_dir)
    layers, expert_bytes, other_bytes = classify(headers)
    sizes = assert_uniform(layers)
    placed = place(
        layers, sizes, round(a.gb300_expert_gib * GIB), round(a.rtx_expert_gib * GIB),
        load_profile(a.profile),
    )
    placed["meta"]["expert_bytes"] = expert_bytes
    placed["meta"]["other_bytes"] = other_bytes
    placed["meta"]["header_tensors"] = sum(
        1 for h in headers.values() for k in h if k != "__metadata__")
    with open(a.out, "w") as f:
        json.dump(placed, f)
    meta = placed["meta"]
    c = meta["counts"]
    b = meta["bytes"]
    print(
        f"NVFP4 {c['gb300_nvfp4']}/{N_EXPERTS} GB300, {c['rtx_nvfp4']} RTX, "
        f"{c['grace_nvfp4']} Grace; MXFP8 {c['gb300_mxfp8']} on GB300"
    )
    print(
        f"bytes GB300 {b['gb300']} ({b['gb300'] / GIB:.3f} GiB), "
        f"RTX {b['rtx']} ({b['rtx'] / GIB:.3f} GiB), "
        f"Grace {b['grace']} ({b['grace'] / GIB:.3f} GiB)"
    )
    print(f"expert payload {expert_bytes} bytes, other {other_bytes} bytes")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
