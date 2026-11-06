#!/usr/bin/env python3
"""Build a hot/cold rowmap from route counts and report what share of routed tokens it sends to the RTX.

    rowmap.py BEFORE.json AFTER.json OUT.json [--cold-per-layer N] [--decode-weight 0.6]
              [--rtx-gib 85] [--gb300-gib 250.7] [--util 0.95]

BEFORE/AFTER are sitecustomize.py snapshots taken around the workload; their difference is the calibration.
Each target MoE layer ranks its experts by
    score = w * decode share + (1 - w) * prefill share
(decode = calls of up to the snapshot's decode_max_tokens, prefill = larger calls; shares are per layer), and its N
lowest-scoring experts go cold, i.e. to the RTX. N is the same for every layer, as in Garner's rowmap, so each
layer's RTX work is alike. The drafter's (MTP) layers stay whole on the GB300: its sequential single-token passes
cannot wait on the RTX.
Without --cold-per-layer, N is the most cold experts that fit in --rtx-gib. Output is Garner's format,
{"layers": {"<layer>": {"hot": [...], "cold": [...]}}, "meta": {...}}, keyed by the model's layer index.
"""
import argparse
import json

# From the safetensors headers of zai-org/GLM-5.3-Flash at 46a050be; another revision needs them re-measured.
EXPERT_BYTES = 25_171_968          # one routed expert, FP8 weights + scales
NON_EXPERT_GIB = 15.43             # attention, embeddings, LM head, vision, shared/dense MLPs, MTP non-expert
MTP_EXPERTS_GIB = 6.75             # the MTP layer's 288 experts, kept on the GB300
GIB = 2 ** 30


def delta(before: dict, after: dict) -> list:
    """counts[bucket][layer][expert] of AFTER minus BEFORE."""
    return [[[a - b for a, b in zip(la, lb)] for la, lb in zip(ba, bb)]
            for ba, bb in zip(after["counts"], before["counts"])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("out")
    p.add_argument("--cold-per-layer", type=int)
    p.add_argument("--decode-weight", type=float, default=0.6)
    p.add_argument("--rtx-gib", type=float, default=85.0)
    p.add_argument("--gb300-gib", type=float, default=250.7)
    p.add_argument("--util", type=float, default=0.95)
    a = p.parse_args()
    before, after = json.load(open(a.before)), json.load(open(a.after))
    edges, E = after["bucket_edges"], after["n_experts"]
    split = edges.index(after["decode_max_tokens"]) + 1   # buckets [0, split) are decode, the rest prefill
    c = delta(before, after)
    layers = after["target_layers"]
    dec_tok = sum(after["tokens"][b][layers[0]] - before["tokens"][b][layers[0]] for b in range(split))
    all_tok = sum(after["tokens"][b][layers[0]] - before["tokens"][b][layers[0]] for b in range(len(c)))
    if dec_tok < 0.2 * all_tok:
        raise SystemExit(f"only {dec_tok} of {all_tok} counted tokens were in decode-sized calls: with MTP each "
                         f"stream sends 1 + drafts tokens per step, so keep drive.py's concurrency x that "
                         f"<= {after['decode_max_tokens']}")
    if any(c[b][layer][E] for b in range(len(c)) for layer in layers):
        raise SystemExit("invalid (negative) expert ids were counted; the calibration is not usable")

    def hits(layer: int, buckets) -> list:
        return [sum(c[b][layer][e] for b in buckets) for e in range(E)]

    n_cold = a.cold_per_layer or int(a.rtx_gib * GIB // EXPERT_BYTES) // len(layers)
    out_layers, rows = {}, []
    tot = dict(dec=0, dec_cold=0, pre=0, pre_cold=0)
    for layer in layers:
        dec, pre = hits(layer, range(split)), hits(layer, range(split, len(c)))
        sd, sp = sum(dec) or 1, sum(pre) or 1
        score = [a.decode_weight * dec[e] / sd + (1 - a.decode_weight) * pre[e] / sp for e in range(E)]
        order = sorted(range(E), key=lambda e: (score[e], e))
        cold, hot = sorted(order[:n_cold]), sorted(order[n_cold:])
        out_layers[str(layer)] = {"hot": hot, "cold": cold}
        dc, pc = sum(dec[e] for e in cold), sum(pre[e] for e in cold)
        tot["dec"] += sum(dec)
        tot["dec_cold"] += dc
        tot["pre"] += sum(pre)
        tot["pre_cold"] += pc
        rows.append((layer, dc / sd, pc / sp))

    cold_gib = n_cold * len(layers) * EXPERT_BYTES / GIB
    hot_gib = (E - n_cold) * len(layers) * EXPERT_BYTES / GIB
    gb300_weights = hot_gib + NON_EXPERT_GIB + MTP_EXPERTS_GIB
    budget = a.gb300_gib * a.util
    dec_share = tot["dec_cold"] / max(tot["dec"], 1)
    pre_share = tot["pre_cold"] / max(tot["pre"], 1)
    meta = dict(calibration=f"{a.decode_weight} decode (calls <= {after['decode_max_tokens']} tokens) + "
                            f"{1 - a.decode_weight:.1f} prefill; counts {a.before} -> {a.after}",
                model="zai-org/GLM-5.3-Flash@46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
                layers=f"{layers[0]}-{layers[-1]}",
                n_hot=E - n_cold, n_cold=n_cold, cold_gib=round(cold_gib, 2), gb300_weights_gib=round(gb300_weights, 2),
                decode_tokens=dec_tok, prefill_tokens=all_tok - dec_tok,
                rtx_share_decode=round(dec_share, 4), rtx_share_prefill=round(pre_share, 4))
    json.dump({"layers": out_layers, "meta": meta}, open(a.out, "w"))
    print(f"cold {n_cold}/layer x {len(layers)} layers = {cold_gib:.1f} GiB on the RTX (budget {a.rtx_gib})")
    print(f"GB300 weights {gb300_weights:.1f} GiB of {budget:.1f} at util {a.util}: "
          f"{budget - gb300_weights:.1f} GiB left for KV, activations and CUDA graphs")
    print(f"routed tokens to the RTX: decode {100 * dec_share:.1f} %, prefill {100 * pre_share:.1f} % "
          f"(uniform would be {100 * n_cold / E:.1f} %)")
    worst = sorted(rows, key=lambda r: -r[1])[:5]
    print("layers with the largest decode share on the RTX:",
          ", ".join(f"{layer}: {100 * d:.1f} %" for layer, d, _ in worst))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
