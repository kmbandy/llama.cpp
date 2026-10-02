"""CPU-only checks for the wp-forge convert path (rules.py, ggml_quant.py, convert.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "calibration"))

import gguf  # noqa: E402

from conversion.wp_forge import ggml_quant  # noqa: E402
from conversion.wp_forge.convert import Ml8Opts, convert  # noqa: E402
from conversion.wp_forge.rules import QuantRules, RulesError  # noqa: E402

GQT = gguf.GGMLQuantizationType
needs_lib = pytest.mark.skipif(not ggml_quant.lib_path().is_file(), reason="no build-forge libggml-base")


def test_rules_parse_and_decide() -> None:
    r = QuantRules.parse({"default": "ml8_4", "rules": [
        {"match": r"^output\.weight$", "type": "drop"},
        {"match": r"token_embd", "type": "q8_0"}]})
    assert r.decide("output.weight", 2, False) == ("drop", r"^output\.weight$")
    assert r.decide("token_embd.weight", 2, False)[0] == "q8_0"
    assert r.decide("blk.0.ffn_up.weight", 2, False) == ("ml8_4", None)
    assert r.decide("blk.0.attn_norm.weight", 1, True)[0] == "keep"
    assert r.decide("blk.0.ssm_conv1d.weight", 2, True)[0] == "keep"  # converter forced F32
    for bad in ({"default": "q9_9"}, {"default": "drop"}, {"rules": [{"match": "(", "type": "q8_0"}]},
                {"rules": [{"type": "q8_0"}]}, {"defualt": "q8_0"}):
        with pytest.raises(RulesError):
            QuantRules.parse(bad)


@needs_lib
def test_ggml_quant_matches_gguf_py() -> None:
    x = np.random.default_rng(0).standard_normal((64, 512)).astype(np.float32)
    for t in ("q8_0", "q4_0"):
        assert np.array_equal(ggml_quant.quantize(x, t, workers=4, chunk_rows=7 * 1),
                              gguf.quantize(x, GQT[t.upper()]))
    q = ggml_quant.quantize(x, "q4_k")
    d = gguf.dequantize(q, GQT.Q4_K)
    assert np.linalg.norm(d - x) / np.linalg.norm(x) < 0.1


def _bf16(a: np.ndarray) -> np.ndarray:
    return gguf.quantize(a.astype(np.float32), GQT.BF16)


def _source(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    w = {
        "token_embd.weight": rng.standard_normal((96, 256)),
        "output.weight": rng.standard_normal((96, 256)),
        "blk.0.attn_qkv.weight": rng.standard_normal((128, 256)),
        "blk.0.attn_gate.weight": rng.standard_normal((64, 256)),
        "blk.0.ssm_out.weight": rng.standard_normal((256, 256)),
        "blk.0.ssm_alpha.weight": rng.standard_normal((8, 256)),
        "blk.0.ffn_gate_exps.weight": rng.standard_normal((4, 64, 256)),
        "blk.0.ffn_down_exps.weight": rng.standard_normal((4, 256, 64)),
    }
    w = {k: v.astype(np.float32) for k, v in w.items()}
    wr = gguf.GGUFWriter(str(path), arch="qwen35")
    wr.add_block_count(1)
    for k, v in w.items():
        wr.add_tensor(k, _bf16(v), raw_dtype=GQT.BF16)
    norm = rng.standard_normal(256).astype(np.float32)
    wr.add_tensor("blk.0.attn_norm.weight", norm)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    w["blk.0.attn_norm.weight"] = norm
    # bf16-rounded reference values
    return {k: (gguf.dequantize(_bf16(v), GQT.BF16) if v.ndim > 1 else v) for k, v in w.items()}


def _decode_ml8_4(packed: np.ndarray, cent: np.ndarray, k: int) -> torch.Tensor:
    from gguf_state import decode_centroids_fp8, unpack_ml8_blocks
    n = packed.shape[0]
    idx, scl = unpack_ml8_blocks(packed.reshape(n, -1), n, k)
    c = decode_centroids_fp8(cent.reshape(k // 64, 16))
    g = torch.arange(k) // 64
    return c[g, idx.long()] * scl[:, g]


@needs_lib
def test_gguf_requant_ml8_rules_and_sidecars(tmp_path: Path) -> None:
    from convert_fp8_rotated import _group_seed, role_group_key
    from kronecker_rotation import BlockHadamardRotation, KroneckerRotation, factor_for_dim, random_orthogonal
    from conversion.wp_forge import ml8 as ml8mod

    src = tmp_path / "src.gguf"
    ref = _source(src)
    out = tmp_path / "out.gguf"
    rules = QuantRules.parse({"default": "ml8_4", "rules": [
        {"match": r"^output\.weight$", "type": "drop"},
        {"match": r"^token_embd\.weight$", "type": "q4_k"}]})
    events: list[dict] = []
    res = convert(f"gguf:{src}", out, rules, Ml8Opts(fit_rows=4096), events=events.append)
    r = gguf.GGUFReader(str(out))
    t = {x.name: x for x in r.tensors}

    assert "output.weight" not in t
    assert t["token_embd.weight"].tensor_type == GQT.Q4_K
    assert t["blk.0.ssm_alpha.weight"].tensor_type == GQT.Q8_0  # no ml8 role -> fallback
    assert any(e["kind"] == "fallback" and e["tensor"] == "blk.0.ssm_alpha.weight" for e in events)
    assert np.array_equal(np.asarray(t["blk.0.attn_norm.weight"].data), ref["blk.0.attn_norm.weight"])
    assert res.bytes == out.stat().st_size

    # kronecker roles: h_a/meta exactly what convert_fp8_rotated would write; qkv and gate share one h_a
    a, b = factor_for_dim(256, max_b=1024)
    h_a = random_orthogonal(a, seed=_group_seed(0, 0, role_group_key("attn_qkv")))
    for name in ("blk.0.attn_qkv", "blk.0.attn_gate"):
        assert t[f"{name}.weight"].tensor_type == GQT.ML8_4
        assert np.array_equal(np.asarray(t[f"{name}.rotation_h_a"].data).reshape(a, a), h_a.numpy())
        assert list(np.asarray(t[f"{name}.rotation_meta"].data)) == [a, b, 256, 1]
    w = _decode_ml8_4(np.asarray(t["blk.0.attn_qkv.weight"].data), np.asarray(t["blk.0.attn_qkv.centroids"].data), 256)
    back = KroneckerRotation(h_a=h_a, b_dim=b).inverse(w).numpy()
    src_w = ref["blk.0.attn_qkv.weight"]
    assert np.linalg.norm(back - src_w) / np.linalg.norm(src_w) < 0.15

    # block-Hadamard role: meta only, b = local_b
    assert "blk.0.ssm_out.rotation_h_a" not in t
    assert list(np.asarray(t["blk.0.ssm_out.rotation_meta"].data)) == [2, 128, 256, 2]
    w = _decode_ml8_4(np.asarray(t["blk.0.ssm_out.weight"].data), np.asarray(t["blk.0.ssm_out.centroids"].data), 256)
    back = BlockHadamardRotation(in_features=256, b_dim=128).inverse(w).numpy()
    src_w = ref["blk.0.ssm_out.weight"]
    assert np.linalg.norm(back - src_w) / np.linalg.norm(src_w) < 0.15

    # routed experts: per-expert centroids, ExpertStage's rotation
    g = t["blk.0.ffn_gate_exps.weight"]
    assert g.tensor_type == GQT.ML8_4
    assert np.asarray(t["blk.0.ffn_gate_exps.centroids"].data).size == 4 * (256 // 64) * 16
    rot = ml8mod.build_rotation(256, "kronecker", 0, 1024, 0, "gate_up")[0]
    assert np.array_equal(np.asarray(t["blk.0.ffn_gate_exps.rotation_h_a"].data).reshape(rot.h_a.shape),
                          ml8mod.rotation_h_a_array(rot))
    packed = np.asarray(g.data).reshape(4, 64, -1)
    cents = np.asarray(t["blk.0.ffn_gate_exps.centroids"].data).reshape(4, 4, 16)
    w = _decode_ml8_4(packed[2], cents[2], 256)
    back = rot.inverse(w).numpy()
    src_w = ref["blk.0.ffn_gate_exps.weight"][2]
    assert np.linalg.norm(back - src_w) / np.linalg.norm(src_w) < 0.15


@needs_lib
def test_ml8_3_uses_eight_levels(tmp_path: Path) -> None:
    src = tmp_path / "src.gguf"
    _source(src)
    out = tmp_path / "out.gguf"
    convert(f"gguf:{src}", out, QuantRules.parse({"default": "ml8_3"}), Ml8Opts(fit_rows=4096))
    t = {x.name: x for x in gguf.GGUFReader(str(out)).tensors}
    from gguf_state import unpack_ml8_blocks
    idx, _ = unpack_ml8_blocks(np.asarray(t["blk.0.attn_qkv.weight"].data).reshape(128, -1), 128, 256)
    assert int(idx.max()) <= 7


@needs_lib
def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "src.gguf"
    _source(src)
    out = tmp_path / "out.gguf"
    res = convert(f"gguf:{src}", out, QuantRules.parse({"default": "q8_0"}), dry_run=True)
    assert not out.exists()
    assert res.bytes > 0 and all(d.type in ("q8_0", "keep") for d in res.decisions)
