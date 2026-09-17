#!/usr/bin/env python3
"""Tests for convert_fp8_rotated.py — data-free BF16 -> ML8_FP8-rotated GGUF."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402

from convert_fp8_rotated import (  # noqa: E402
    _sidecar_base,
    build_plan, classify_tensor, convert, tensor_role, _quantize_q8_0_gpu,
    _CHUNK_ROWS, _row_chunk_bounds,
)
from gguf.quants import quantize as gguf_quantize  # noqa: E402
from kronecker_rotation import (  # noqa: E402
    KroneckerRotation, BlockHadamardRotation, random_orthogonal, factor_for_dim,
    KRONECKER_ORTH_SYLVESTER_KIND_ID, BLOCK_HADAMARD_KIND_ID,
)

torch.manual_seed(0)

# K values chosen so:
#   4864 is divisible by 128 (local_b) but NOT by 1024 -> exercises block_hadamard
#   2048 is divisible by 1024 -> exercises kronecker with max_b=1024
K_HADAMARD = 4864
K_KRON = 2048
N_OUT = 96
VOCAB = 40
D_MODEL = K_KRON  # hidden size == the kronecker-rotated attn/ffn in_features
FFN_INNER = K_HADAMARD  # ffn_down's in_features (the FFN intermediate size)


def _add_bf16(writer, name, t: torch.Tensor):
    data = np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint8).numpy())
    writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.BF16)


def _make_synthetic_gguf(path: Path) -> dict:
    """Tiny synthetic bf16 GGUF with tensors named/shaped like the real model,
    covering every allowlist action. Returns the dict of {name: torch weight}
    (torch layout, i.e. [N, K] — reversed from GGUF ne) used to build it, so
    tests can recompute the expected math independently of the converter."""
    w = gguf.GGUFWriter(str(path), arch="qwen35")
    w.add_uint32("qwen35.embedding_length", D_MODEL)
    w.add_uint32("qwen35.block_count", 1)

    weights = {}

    def add(name, N, K):
        t = torch.randn(N, K, dtype=torch.float32) * 0.5
        weights[name] = t
        _add_bf16(w, name, t)

    add("token_embd.weight", VOCAB, D_MODEL)          # q8_0, no rotation

    # 1D norm/bias tensors -> always copied verbatim regardless of role table.
    norm = torch.ones(D_MODEL) + 0.05 * torch.randn(D_MODEL)
    weights["blk.0.attn_norm.weight"] = norm
    _add_bf16(w, "blk.0.attn_norm.weight", norm)

    add("blk.0.attn_qkv.weight", N_OUT, D_MODEL)      # kronecker (N-split)
    add("blk.0.attn_output.weight", D_MODEL, K_HADAMARD)  # block_hadamard (K-split)
    add("blk.0.ffn_gate.weight", FFN_INNER, D_MODEL)  # kronecker
    add("blk.0.ffn_up.weight", FFN_INNER, D_MODEL)    # kronecker
    add("blk.0.ffn_down.weight", D_MODEL, K_HADAMARD)  # block_hadamard
    add("blk.0.ssm_out.weight", D_MODEL, K_HADAMARD)  # block_hadamard
    add("blk.0.ssm_alpha.weight", 48, D_MODEL)        # q8_0
    add("blk.0.ssm_beta.weight", 48, D_MODEL)         # q8_0

    conv = torch.randn(4, D_MODEL)
    weights["blk.0.ssm_conv1d.weight"] = conv
    _add_bf16(w, "blk.0.ssm_conv1d.weight", conv)     # copy (not '.weight' role match... it IS
                                                       # a "ssm_conv1d" role, not in any allowlist)

    add("output.weight", VOCAB, D_MODEL)              # kronecker (untied lm head)
    onorm = torch.ones(D_MODEL)
    weights["output_norm.weight"] = onorm
    _add_bf16(w, "output_norm.weight", onorm)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return weights


@pytest.fixture()
def synthetic_gguf(tmp_path):
    src = tmp_path / "tiny_bf16.gguf"
    weights = _make_synthetic_gguf(src)
    return src, weights


def _dequant_ml8_fp8(reader, name: str) -> np.ndarray:
    """Dequantize an ML8_FP8 tensor back to fp32 [N, K] (numpy)."""
    t = next(x for x in reader.tensors if x.name == name)
    assert t.tensor_type == GGMLQuantizationType.ML8_FP8
    block_size, block_bytes = gguf.constants.GGML_QUANT_SIZES[GGMLQuantizationType.ML8_FP8]
    raw = np.ascontiguousarray(t.data)  # [N, n_blocks * block_bytes] uint8
    N = raw.shape[0]
    n_blocks = raw.shape[1] // block_bytes
    raw = raw.reshape(N, n_blocks, block_bytes)
    scale_bytes = raw[:, :, :2].reshape(N, n_blocks, 2)
    scale = scale_bytes.view(np.float16).astype(np.float32).reshape(N, n_blocks)
    e4m3_bytes = raw[:, :, 2:].reshape(N, n_blocks * block_size)
    e4m3 = torch.from_numpy(e4m3_bytes.copy()).view(torch.float8_e4m3fn).to(torch.float32).numpy()
    e4m3 = e4m3.reshape(N, n_blocks, block_size)
    out = (e4m3 * scale[:, :, None]).reshape(N, n_blocks * block_size)
    return out


def _read_rotation_meta(reader, weight_name: str) -> np.ndarray:
    base = _sidecar_base(weight_name)
    t = next(x for x in reader.tensors if x.name == base + ".rotation_meta")
    return np.ascontiguousarray(t.data).view(np.int32).copy()


def test_classify_tensor_allowlist():
    assert classify_tensor("blk.3.attn_output.weight", (4864, 5120))[0] == "rotate_hadamard"
    assert classify_tensor("blk.3.ffn_down.weight", (4864, 5120))[0] == "rotate_hadamard"
    assert classify_tensor("blk.3.ssm_out.weight", (4864, 5120))[0] == "rotate_hadamard"
    for role in ["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "ffn_gate", "ffn_up"]:
        assert classify_tensor(f"blk.0.{role}.weight", (2048, 96))[0] == "rotate_kronecker"
    assert classify_tensor("output.weight", (2048, 40))[0] == "rotate_kronecker"
    assert classify_tensor("token_embd.weight", (2048, 40))[0] == "q8_0"
    assert classify_tensor("blk.0.ssm_alpha.weight", (2048, 48))[0] == "q8_0"
    assert classify_tensor("blk.0.ssm_beta.weight", (2048, 48))[0] == "q8_0"
    assert classify_tensor("blk.0.ssm_conv1d.weight", (4, 2048))[0] == "copy"
    assert classify_tensor("blk.0.attn_norm.weight", (2048,))[0] == "copy"
    print("  PASS test_classify_tensor_allowlist")


def test_tensor_role_parsing():
    assert tensor_role("blk.3.attn_output.weight") == ("attn_output", True)
    assert tensor_role("output.weight") == ("output", True)
    assert tensor_role("token_embd.weight") == ("token_embd", True)
    assert tensor_role("blk.0.attn_norm.weight") == ("attn_norm", True)
    assert tensor_role("blk.0.ssm_a") == (None, False)
    print("  PASS test_tensor_role_parsing")


def test_quantize_q8_0_gpu_matches_reference():
    """_quantize_q8_0_gpu (CPU-device torch, standing in for the GPU path in
    this CPU-only test run) must be bit-exact with gguf.quants.quantize(...,
    Q8_0) — it exists only to avoid materializing giant fp32 host arrays for
    token_embd/output.weight, not to change the numerics."""
    torch.manual_seed(3)
    for N, K in [(1, 32), (7, 64), (40, 2048)]:
        w = torch.randn(N, K, dtype=torch.float32) * 3.0
        actual = _quantize_q8_0_gpu(w)
        expected = gguf_quantize(w.numpy(), GGMLQuantizationType.Q8_0)
        np.testing.assert_array_equal(actual, expected)
    print("  PASS test_quantize_q8_0_gpu_matches_reference")


def test_dry_run_plan(synthetic_gguf):
    src, _ = synthetic_gguf
    reader = gguf.GGUFReader(src)
    plan = build_plan(reader, rotation_seed=0, local_b=128, max_b=1024)
    by_name = {e["name"]: e for e in plan}

    assert by_name["blk.0.attn_output.weight"]["action"] == "rotate_hadamard"
    assert by_name["blk.0.attn_output.weight"]["kind"] == "block_hadamard"
    assert by_name["blk.0.attn_output.weight"]["a"] == K_HADAMARD // 128
    assert by_name["blk.0.attn_output.weight"]["b"] == 128

    assert by_name["blk.0.attn_qkv.weight"]["action"] == "rotate_kronecker"
    a, b = factor_for_dim(K_KRON, max_b=1024)
    assert by_name["blk.0.attn_qkv.weight"]["a"] == a
    assert by_name["blk.0.attn_qkv.weight"]["b"] == b

    assert by_name["token_embd.weight"]["action"] == "q8_0"
    assert by_name["blk.0.ssm_alpha.weight"]["action"] == "q8_0"
    assert by_name["blk.0.ssm_conv1d.weight"]["action"] == "copy"
    assert by_name["blk.0.attn_norm.weight"]["action"] == "copy"
    print("  PASS test_dry_run_plan")


def test_convert_end_to_end(synthetic_gguf, tmp_path):
    src, weights = synthetic_gguf
    out = tmp_path / "tiny_fp8rot.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024)
    assert out.exists()

    reader = gguf.GGUFReader(out)
    names = {t.name: t for t in reader.tensors}

    # ── Types per the allowlist ──────────────────────────────────────────
    assert names["blk.0.attn_output.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["blk.0.ffn_down.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["blk.0.ssm_out.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["blk.0.attn_qkv.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["blk.0.ffn_gate.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["blk.0.ffn_up.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["output.weight"].tensor_type == GGMLQuantizationType.ML8_FP8
    assert names["token_embd.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["blk.0.ssm_alpha.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["blk.0.ssm_beta.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["blk.0.ssm_conv1d.weight"].tensor_type == GGMLQuantizationType.BF16
    assert names["blk.0.attn_norm.weight"].tensor_type == GGMLQuantizationType.BF16

    # ── Sidecars ──────────────────────────────────────────────────────────
    assert "blk.0.attn_output.rotation_h_a" not in names   # block_hadamard: no h_a
    assert "blk.0.attn_output.rotation_meta" in names
    assert "blk.0.attn_qkv.rotation_h_a" in names          # kronecker: has h_a
    assert "blk.0.attn_qkv.rotation_meta" in names
    assert "token_embd.rotation_meta" not in names         # no rotation for q8_0 tier

    meta_hada = _read_rotation_meta(reader, "blk.0.attn_output.weight")
    assert meta_hada.tolist() == [K_HADAMARD // 128, 128, K_HADAMARD, BLOCK_HADAMARD_KIND_ID]

    meta_kron = _read_rotation_meta(reader, "blk.0.attn_qkv.weight")
    a, b = factor_for_dim(K_KRON, max_b=1024)
    assert meta_kron.tolist() == [a, b, K_KRON, KRONECKER_ORTH_SYLVESTER_KIND_ID]

    # ── Copied tensors are bit-identical ────────────────────────────────
    conv_orig = weights["blk.0.ssm_conv1d.weight"].to(torch.bfloat16).to(torch.float32).numpy()
    conv_out = next(t for t in reader.tensors if t.name == "blk.0.ssm_conv1d.weight")
    # F32 in the source writer? no — it was written as BF16, and passthrough keeps
    # BF16 unchanged; compare bit-for-bit via the same widen trick.
    assert conv_out.tensor_type == GGMLQuantizationType.BF16
    u16 = np.ascontiguousarray(conv_out.data).view(np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(f32, conv_orig)

    print("  PASS test_convert_end_to_end (types/sidecars/copy)")


def _nmse(actual: np.ndarray, expected: np.ndarray) -> float:
    num = np.sum((actual - expected) ** 2)
    den = np.sum(expected ** 2)
    return float(num / den) if den > 0 else float(num)


def test_convert_rotation_math_matches_unrotated_gemm(synthetic_gguf, tmp_path):
    """dequant(W_rot) . rotate.forward(x) ~= W . x for random x — within fp8 noise."""
    src, weights = synthetic_gguf
    out = tmp_path / "tiny_fp8rot.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024)
    reader = gguf.GGUFReader(out)

    cases = [
        ("blk.0.attn_output.weight", K_HADAMARD, "block_hadamard"),
        ("blk.0.attn_qkv.weight", K_KRON, "kronecker_orth_sylvester"),
        ("blk.0.ffn_down.weight", K_HADAMARD, "block_hadamard"),
        ("blk.0.ffn_gate.weight", K_KRON, "kronecker_orth_sylvester"),
        ("output.weight", K_KRON, "kronecker_orth_sylvester"),
    ]
    torch.manual_seed(7)
    for name, K, kind in cases:
        base = _sidecar_base(name)
        meta = _read_rotation_meta(reader, name)
        a, b, in_features, kind_id = [int(v) for v in meta]
        assert in_features == K

        if kind == "block_hadamard":
            rot = BlockHadamardRotation(in_features=K, b_dim=b)
        else:
            h_a_t = next(t for t in reader.tensors if t.name == base + ".rotation_h_a")
            h_a = torch.from_numpy(np.ascontiguousarray(h_a_t.data).copy())
            rot = KroneckerRotation(h_a=h_a, b_dim=b)

        W = weights[name]  # [N, K] fp32, the ORIGINAL (unrotated) weight
        N = W.shape[0]
        x = torch.randn(5, K, dtype=torch.float32)

        y_expected = x @ W.T  # [5, N]

        W_rot_dequant = torch.from_numpy(_dequant_ml8_fp8(reader, name).copy())  # [N, K]
        x_rot = rot.forward(x)
        y_actual = x_rot @ W_rot_dequant.T

        err = _nmse(y_actual.numpy(), y_expected.numpy())
        assert err < 5e-3, f"{name}: NMSE {err:.3e} too high"
    print("  PASS test_convert_rotation_math_matches_unrotated_gemm")


def test_seed_determinism(synthetic_gguf, tmp_path):
    """Two runs with the same --rotation-seed give bit-identical rotation sidecars
    and ML8_FP8 bytes."""
    src, _ = synthetic_gguf
    out1 = tmp_path / "run1.gguf"
    out2 = tmp_path / "run2.gguf"
    convert(src, out1, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024)
    convert(src, out2, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024)

    r1 = gguf.GGUFReader(out1)
    r2 = gguf.GGUFReader(out2)
    names1 = {t.name: t for t in r1.tensors}
    names2 = {t.name: t for t in r2.tensors}
    assert set(names1) == set(names2)
    for name in names1:
        b1 = np.ascontiguousarray(names1[name].data)
        b2 = np.ascontiguousarray(names2[name].data)
        np.testing.assert_array_equal(b1, b2, err_msg=f"{name}: bytes differ across runs")
    print("  PASS test_seed_determinism")


def test_row_chunk_bounds():
    # N=10, chunk_rows=3 -> naive bounds [(0,3),(3,6),(6,9),(9,10)] has a lone
    # trailing row -> merged into [(0,3),(3,6),(6,10)].
    assert _row_chunk_bounds(10, 3) == [(0, 3), (3, 6), (6, 10)]
    assert _row_chunk_bounds(9, 3) == [(0, 3), (3, 6), (6, 9)]
    assert _row_chunk_bounds(1, 3) == [(0, 1)]
    # N=4, chunk_rows=3 -> naive bounds [(0,3),(3,4)] has a lone trailing row
    # -> merged into a single (0,4) chunk.
    assert _row_chunk_bounds(4, 3) == [(0, 4)]
    # N=7, chunk_rows=3 -> naive bounds would be [(0,3),(3,6),(6,7)] (lone
    # trailing row) -> merged into [(0,3),(3,7)].
    assert _row_chunk_bounds(7, 3) == [(0, 3), (3, 7)]
    assert _row_chunk_bounds(100, 8192) == [(0, 100)]
    with pytest.raises(ValueError):
        _row_chunk_bounds(10, 0)
    print("  PASS test_row_chunk_bounds")


def test_convert_chunked_matches_unchunked(synthetic_gguf, tmp_path):
    """--chunk-rows forces every weight (rotate_kronecker, rotate_hadamard, and
    q8_0 alike) through multiple row-chunks — every tensor here has more rows
    than chunk_rows=3 (VOCAB=40, D_MODEL=2048, FFN_INNER=4864, etc.). Since
    rotation only mixes within a row's K and per-block scaling is per-row,
    chunking along N must be bit-exact with the single-chunk (whole-tensor)
    path. This is the regression test for the OOM fix: row-chunking is what
    keeps peak GPU memory bounded for huge tensors like token_embd/output.weight."""
    src, _ = synthetic_gguf
    out_whole = tmp_path / "whole.gguf"
    out_chunked = tmp_path / "chunked.gguf"
    convert(src, out_whole, rotation_seed=99, device_str="cpu", local_b=128, max_b=1024,
           chunk_rows=10_000)   # bigger than every tensor's row count -> single chunk
    assert _CHUNK_ROWS > 3   # sanity: the module default isn't itself tiny
    convert(src, out_chunked, rotation_seed=99, device_str="cpu", local_b=128, max_b=1024,
           chunk_rows=3)      # forces multiple chunks for every tensor in the fixture

    r1 = gguf.GGUFReader(out_whole)
    r2 = gguf.GGUFReader(out_chunked)
    names1 = {t.name: t for t in r1.tensors}
    names2 = {t.name: t for t in r2.tensors}
    assert set(names1) == set(names2)
    for name in names1:
        b1 = np.ascontiguousarray(names1[name].data)
        b2 = np.ascontiguousarray(names2[name].data)
        np.testing.assert_array_equal(b1, b2, err_msg=f"{name}: chunked vs unchunked bytes differ")
    print("  PASS test_convert_chunked_matches_unchunked")


def test_different_seed_changes_kronecker_rotation(synthetic_gguf, tmp_path):
    """Sanity check the determinism test isn't vacuous: a different seed must
    actually change the kronecker h_a sidecar bytes."""
    src, _ = synthetic_gguf
    out1 = tmp_path / "runA.gguf"
    out2 = tmp_path / "runB.gguf"
    convert(src, out1, rotation_seed=1, device_str="cpu", local_b=128, max_b=1024)
    convert(src, out2, rotation_seed=2, device_str="cpu", local_b=128, max_b=1024)
    r1 = gguf.GGUFReader(out1)
    r2 = gguf.GGUFReader(out2)
    h1 = next(t for t in r1.tensors if t.name == "blk.0.attn_qkv.rotation_h_a").data
    h2 = next(t for t in r2.tensors if t.name == "blk.0.attn_qkv.rotation_h_a").data
    assert not np.array_equal(np.ascontiguousarray(h1), np.ascontiguousarray(h2))
    print("  PASS test_different_seed_changes_kronecker_rotation")


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
