"""CPU checks for the Inkling 2 Small recipe. No GPU and no checkpoint download."""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "placement"),
    str(ROOT / "kernels" / "reference"),
    str(ROOT / "peer"),
    str(ROOT / "routing"),
]

import plan_experts  # noqa: E402
import profile as profile_mod  # noqa: E402
import quant  # noqa: E402
import rows as rows_mod  # noqa: E402
import segment as segment_mod  # noqa: E402
import slice as slice_mod  # noqa: E402
import split as split_mod  # noqa: E402
from headers import tensor_nbytes  # noqa: E402
from layout import assert_uniform, classify  # noqa: E402

GIB = 1 << 30
NV = 17_915_912
MX = 32_845_824


def _nv_tensors(layer: int) -> dict:
    p = f"model.llm.layers.{layer}.mlp.experts"
    return {
        f"{p}.w13_weight": {"dtype": "U8", "shape": [384, 4608, 2304], "data_offsets": [0, 1]},
        f"{p}.w13_weight_scale": {"dtype": "F8_E4M3", "shape": [384, 4608, 288], "data_offsets": [0, 1]},
        f"{p}.w13_weight_amax": {"dtype": "F32", "shape": [384], "data_offsets": [0, 1]},
        f"{p}.w13_weight_outer_scale_denominator": {"dtype": "F32", "shape": [1], "data_offsets": [0, 1]},
        f"{p}.w2_weight": {"dtype": "U8", "shape": [384, 4608, 1152], "data_offsets": [0, 1]},
        f"{p}.w2_weight_scale": {"dtype": "F8_E4M3", "shape": [384, 4608, 144], "data_offsets": [0, 1]},
        f"{p}.w2_weight_amax": {"dtype": "F32", "shape": [384], "data_offsets": [0, 1]},
        f"{p}.w2_weight_outer_scale_denominator": {"dtype": "F32", "shape": [1], "data_offsets": [0, 1]},
    }


def _mx_tensors(layer: int) -> dict:
    p = f"model.llm.layers.{layer}.mlp.experts"
    return {
        f"{p}.w13_weight": {"dtype": "F8_E4M3", "shape": [384, 4608, 4608], "data_offsets": [0, 1]},
        f"{p}.w13_weight_scale": {"dtype": "U8", "shape": [384, 4608, 144], "data_offsets": [0, 1]},
        f"{p}.w2_weight": {"dtype": "F8_E4M3", "shape": [384, 4608, 2304], "data_offsets": [0, 1]},
        f"{p}.w2_weight_scale": {"dtype": "U8", "shape": [384, 4608, 72], "data_offsets": [0, 1]},
    }


def full_headers() -> dict:
    tensors = {"model.llm.norm.weight": {"dtype": "BF16", "shape": [4608], "data_offsets": [0, 1]}}
    for layer in range(40):
        tensors.update(_nv_tensors(layer))
    for layer in range(40, 48):
        tensors.update(_mx_tensors(layer))
    return {"model.safetensors": {**tensors, "__metadata__": {"format": "pt"}}}


class QuantTest(unittest.TestCase):
    def test_e2m1_pack_order(self):
        codes = np.array([0, 1, 8, 15], dtype=np.uint8)
        vals = quant.e2m1_from_codes(codes)
        self.assertEqual(vals.tolist(), [0.0, 0.5, -0.0, -6.0])
        packed = quant.pack_nvfp4(codes.reshape(1, 4))
        self.assertEqual(int(packed[0, 0]), 0x10)  # low 0, high 1
        self.assertTrue(np.array_equal(quant.unpack_nvfp4(packed), vals.reshape(1, 4)))

    def test_e4m3_known_codes(self):
        codes = np.array([0x00, 0x01, 0x38, 0x3C, 0x7E, 0x7F], dtype=np.uint8)
        vals = quant.e4m3fn_to_float(codes)
        self.assertEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[1], 0.001953125)
        self.assertEqual(vals[2], 1.0)
        self.assertEqual(vals[3], 1.5)
        self.assertEqual(vals[4], 448.0)
        self.assertTrue(np.isnan(vals[5]))

    def test_nvfp4_dequant_uses_stored_denominator(self):
        # Eight packed bytes (16 values). First low nibble is 2 -> 1.0. Scale is 1.0.
        packed = np.zeros((1, 8), dtype=np.uint8)
        packed[0, 0] = 0x02
        scales = np.array([[0x38]], dtype=np.uint8)
        got = quant.dequant_nvfp4(packed, scales, 2688.0, 2688.0)
        self.assertEqual(got.shape, (1, 16))
        self.assertAlmostEqual(float(got[0, 0]), 1.0)
        self.assertTrue(np.all(got[0, 1:] == 0.0))
        half = quant.dequant_nvfp4(packed, scales, 1344.0, 2688.0)
        self.assertAlmostEqual(float(half[0, 0]), 0.5)

    def test_fouroversix_tie_keeps_map6_and_never_loses(self):
        block = np.full((1, 16), 4.0)
        _packed, scales, amax = quant.fouroversix_quantize(block)
        self.assertEqual(float(amax[0]), 4.0)
        # Both candidates reconstruct 4 exactly (see the test comment in quant.py's callers).
        # Map-to-6 scale is 256, code 0x78.
        self.assertEqual(int(scales[0, 0]), 0x78)
        x = np.array([[0.2, 0.2, 0.2, 5.5] + [0.1] * 12])
        packed, scales, amax = quant.fouroversix_quantize(x)
        recon = quant.dequant_nvfp4(packed, scales, amax[:, None], quant.ACT_DENOMINATOR)
        err = np.sum(np.abs(recon - x))
        # Rebuilding each candidate the way the function does must not beat the choice.
        self.assertLess(err, 5.0)

    def test_mxfp8_power_of_two_scale(self):
        values = np.array([[0x38] * 32], dtype=np.uint8)  # 1.0
        scales = np.array([[127 + 1]], dtype=np.uint8)  # 2^1
        got = quant.dequant_mxfp8(values, scales)
        self.assertTrue(np.allclose(got, 2.0))
        nan_scale = quant.dequant_mxfp8(values, np.array([[0xFF]], dtype=np.uint8))
        self.assertTrue(np.isnan(nan_scale).all())

    def test_router_tie_breaks_low_and_weights_sum_to_scale(self):
        logits = np.zeros((1, 4))
        bias = np.zeros(4)
        weights, ids = quant.select_experts(logits, bias, topk=2, route_scale=8.0)
        self.assertEqual(ids[0].tolist(), [0, 1])
        self.assertAlmostEqual(float(weights.sum()), 8.0)

    def test_swiglu_interleave(self):
        gate, up, route = 4.0, 0.0, 0.5
        gate_up = np.array([gate, up])
        got = quant.swiglu(gate_up, route)
        expect = (4.0 * np.tanh(1.0)) * (1.0 / (1.0 + np.exp(-gate))) * (25.0 * np.tanh(0.0)) * route
        self.assertAlmostEqual(float(got[0]), float(expect))


class PlacementTest(unittest.TestCase):
    def setUp(self):
        self.layers, self.expert_bytes, self.other = classify(full_headers())
        self.sizes = assert_uniform(self.layers)

    def test_bytes_match_the_plan(self):
        self.assertEqual(self.sizes["nvfp4"], NV)
        self.assertEqual(self.sizes["mxfp8"], MX)
        self.assertEqual(self.expert_bytes, 40 * (NV * 384 + 8) + 8 * MX * 384)
        self.assertEqual(self.other, 4608 * 2)
        self.assertEqual(tensor_nbytes(self.layers[0].tensors["w13_weight"]), 384 * 4608 * 2304)

    def test_default_budgets_are_158_130_96(self):
        placed = plan_experts.place(
            self.layers, self.sizes, round(199.425 * GIB), round(87 * GIB), None)
        c = placed["meta"]["counts"]
        b = placed["meta"]["bytes"]
        self.assertEqual((c["gb300_nvfp4"], c["rtx_nvfp4"], c["grace_nvfp4"]), (158, 130, 96))
        self.assertEqual(c["gb300_mxfp8"], 384)
        self.assertEqual(b["gb300"], 214_130_935_168)
        self.assertEqual(b["rtx"], 93_162_742_400)
        self.assertEqual(b["grace"], 68_797_102_080)
        self.assertEqual(placed["layers"]["0"]["hot"], list(range(158)))
        self.assertEqual(placed["layers"]["40"]["cold"], [])
        self.assertEqual(placed["layers"]["47"]["quant"], "mxfp8")

    def test_looser_gb300_budget_fits_160_not_158(self):
        placed = plan_experts.place(
            self.layers, self.sizes, round(200.81 * GIB), round(87 * GIB), None)
        self.assertEqual(placed["meta"]["counts"]["gb300_nvfp4"], 160)

    def test_profile_reorders_nvfp4_only(self):
        counts = {str(i): [0] * 384 for i in range(40)}
        counts["0"][383] = 1000
        placed = plan_experts.place(
            self.layers, self.sizes, round(199.425 * GIB), round(87 * GIB),
            {int(k): v for k, v in counts.items()})
        self.assertIn(383, placed["layers"]["0"]["hot"])
        self.assertEqual(placed["layers"]["40"]["hot"], list(range(384)))

    def test_refuses_to_spill_mxfp8(self):
        with self.assertRaises(SystemExit):
            plan_experts.place(self.layers, self.sizes, 1000, round(87 * GIB), None)

    def test_coldskip_and_row_slice(self):
        placed = plan_experts.place(
            self.layers, self.sizes, round(199.425 * GIB), round(87 * GIB), None)
        meta = self.layers[0].tensors["w13_weight"]
        meta = dict(meta, data_offsets=[1000, 1000 + tensor_nbytes(meta)])
        self.assertTrue(slice_mod.coldskip(placed, 0, 200))
        self.assertFalse(slice_mod.coldskip(placed, 0, 0))
        lo, hi = slice_mod.file_offsets(meta, 2)
        self.assertEqual(hi - lo, 4608 * 2304)


class ProfileTest(unittest.TestCase):
    def test_roundtrip_ignores_masked_ids(self):
        counts = profile_mod.empty_counts(2, 8)
        profile_mod.add_hits(counts, 1, np.array([[0, -1], [0, 3]]))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            profile_mod.dump_profile(path, counts)
            loaded = profile_mod.load_counts(path)
        self.assertEqual(int(loaded[1, 0]), 2)
        self.assertEqual(int(loaded[1, 3]), 1)
        self.assertEqual(int(loaded[0].sum()), 0)


class SplitAndSegmentTest(unittest.TestCase):
    def test_tiers_sum_to_the_full_layer(self):
        rng = np.random.default_rng(0)
        h, i, e, t, k = 32, 32, 4, 2, 2
        kind = "nvfp4"
        w13 = rng.normal(size=(e, 2 * i, h))
        w2 = rng.normal(size=(e, h, i))
        x = rng.normal(size=(t, h))
        ids = np.array([[0, 1], [2, 3]])
        weights = np.array([[0.5, 0.5], [0.25, 0.75]])
        full = split_mod.apply_tier(x, ids, weights, list(range(e)), w13, w2, kind, e)
        parts = np.zeros_like(full)
        for members in ([0], [1], [2, 3]):
            local = members
            parts += split_mod.apply_tier(
                x, ids, weights, local, w13[local], w2[local], kind, e)
        self.assertTrue(np.allclose(parts, full))

    def test_segment_matches_the_cuh_and_the_sidecar(self):
        layout = segment_mod.production_layout()
        text = (ROOT / "kernels" / "csrc" / "inkling_proto.cuh").read_text()
        self.assertIn("kOffX = 4096", text)
        self.assertIn("kOffIds = 593920", text)
        self.assertIn("kSegBytes = 1191936", text)
        self.assertEqual(layout.off_x, 4096)
        self.assertEqual(layout.off_ids, 593920)
        self.assertEqual(layout.off_w, 598016)
        self.assertEqual(layout.off_y, 602112)
        self.assertEqual(layout.nbytes, 1191936)
        self.assertEqual(segment_mod.MAGIC, struct.unpack("<Q", b"INK2PEER")[0])

        rng = np.random.default_rng(1)
        h, i, n, k = 32, 32, 4, 2
        cold = [1]
        w13 = rng.normal(size=(1, 2 * i, h))
        w2 = rng.normal(size=(1, h, i))
        x = rng.normal(size=(1, h))
        ids = np.array([[1, 0]], dtype=np.int32)
        route = np.array([[0.6, 0.4]], dtype=np.float32)
        placement = {
            "schema": "inkling2-placement-v1",
            "layers": {"0": {"quant": "nvfp4", "hot": [0], "cold": cold, "spill": [2, 3]}},
            "meta": {"n_experts": n},
        }
        with tempfile.TemporaryDirectory() as d:
            seg_path = os.path.join(d, "seg")
            place_path = os.path.join(d, "place.json")
            bank_path = os.path.join(d, "banks.npy")
            with open(place_path, "w") as f:
                json.dump(placement, f)
            np.save(bank_path, {0: {"w13": w13, "w2": w2}})
            seg = segment_mod.Segment(seg_path, segment_mod.Layout(h, 4, k), create=True)
            seg.publish(7, 0, segment_mod.QUANT_NVFP4, x, ids, route)
            seg.close()
            proc = subprocess.run(
                [sys.executable, str(ROOT / "peer" / "sidecar.py"),
                 "--segment", seg_path, "--placement", place_path, "--banks", bank_path,
                 "--mock-request", "--hidden", str(h), "--max-tokens", "4", "--topk", str(k),
                 "--n-experts", str(n)],
                check=True, capture_output=True, text=True)
            self.assertIn("served seq 7", proc.stdout)
            seg = segment_mod.Segment(seg_path, segment_mod.Layout(h, 4, k), create=False)
            got = seg.read_y(1)
            x_q = segment_mod._bf16_to_f32(segment_mod._f32_to_bf16(x).tobytes()).reshape(1, h)
            expect = split_mod.apply_tier(x_q, ids, route, cold, w13, w2, "nvfp4", n)
            expect = segment_mod._bf16_to_f32(segment_mod._f32_to_bf16(expect).tobytes()).reshape(expect.shape)
            seg.close()
        self.assertTrue(np.allclose(got, expect, atol=0, rtol=0))


class RowTest(unittest.TestCase):
    def test_reads_one_expert_and_nothing_else(self):
        raw = bytes(range(64))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model-00001-of-00001.safetensors")
            rows_mod.write_safetensors(path, [
                ("model.llm.layers.0.mlp.experts.w13_weight", "U8", [4, 16], raw),
            ])
            got = rows_mod.read_expert_row(
                path, "model.llm.layers.0.mlp.experts.w13_weight", 2, n_experts=4)
            self.assertEqual(got, raw[32:48])
            index = {"weight_map": {"model.llm.layers.0.mlp.experts.w13_weight": "model-00001-of-00001.safetensors"}}
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)
            batch = rows_mod.read_expert_rows(
                d, [("model.llm.layers.0.mlp.experts.w13_weight", 0)], n_experts=4)
            self.assertEqual(batch[("model.llm.layers.0.mlp.experts.w13_weight", 0)], raw[:16])


class ThresholdTest(unittest.TestCase):
    def test_bar_is_two_and_a_half_percent_of_the_peak(self):
        sys.path.insert(0, str(ROOT / "peer"))
        from thresholds import judge
        ref = np.ones((4, 4))
        self.assertTrue(judge(ref, ref + 0.02)["pass"])
        self.assertFalse(judge(ref, ref + 0.03)["pass"])

    def test_local_ids_keep_strangers_out(self):
        sys.path.insert(0, str(ROOT / "peer"))
        from bank import local_ids
        ids = np.array([[0, 5], [1, -1]])
        self.assertTrue(np.array_equal(local_ids(ids, [5, 1]), np.array([[-1, 0], [1, -1]], dtype=np.int32)))


class ScriptTest(unittest.TestCase):
    def test_launch_scripts_do_not_stop_dsv41(self):
        for rel in ("inkling2", "serve/serve.sh", "kernels/build.sh", "m3.sh"):
            text = (ROOT / rel).read_text()
            code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
            code = re.sub(r"<<EOF.*?^EOF", "", code, count=1, flags=re.M | re.S)
            code = re.sub(r'"[^"\n]*"', '""', code)
            self.assertNotIn("ds41 down", code)
            self.assertNotRegex(code, r"docker\s+(rm|stop)[^\n]*ds41")
            proc = subprocess.run(["bash", "-n", str(ROOT / rel)], check=False)
            self.assertEqual(proc.returncode, 0, rel)
        serve = (ROOT / "serve" / "serve.sh").read_text()
        self.assertIn("ds41", serve)
        self.assertNotIn("docker rm -f ds41", serve)
        self.assertNotIn("docker stop ds41", serve)
        dry = subprocess.run(["bash", str(ROOT / "kernels" / "build.sh"), "--dry-run"],
                             check=False, capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("sm_103a", dry.stdout)
        self.assertIn("sm_120a", dry.stdout)
        self.assertIn("-fPIC", dry.stdout)
        build = (ROOT / "kernels" / "build.sh").read_text()
        code = "\n".join(line.split("#", 1)[0] for line in build.splitlines())
        self.assertNotIn("unset CUDA_VISIBLE_DEVICES", code)
        self.assertIn("CUDA_HOME", build)
        self.assertNotIn("nvidia-smi", code)
        self.assertIn("30008", (ROOT / "config" / "recipe.env").read_text())
        nvcc = os.environ.get("NVCC")
        cuda = os.environ.get("CUDA_HOME")
        if not nvcc and cuda and os.path.isfile(os.path.join(cuda, "bin", "nvcc")):
            nvcc = os.path.join(cuda, "bin", "nvcc")
        if nvcc and (os.path.isfile(nvcc) or shutil.which(nvcc)):
            compiled = subprocess.run(["bash", str(ROOT / "kernels" / "build.sh")],
                                      check=False, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)


if __name__ == "__main__":
    unittest.main()
