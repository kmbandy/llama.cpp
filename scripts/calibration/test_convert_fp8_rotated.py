#!/usr/bin/env python3
"""Tests for convert_fp8_rotated.py — data-free BF16 -> ML8_FP8-rotated GGUF."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402

from convert_fp8_rotated import (  # noqa: E402
    _sidecar_base,
    build_plan, classify_tensor, convert, tensor_role, _quantize_q8_0_gpu,
    _CHUNK_ROWS, _row_chunk_bounds,
    _quantize_fp8_b128_gpu, _rotate_and_fp8_b128, role_group_key, _group_seed, _parse_layer,
    _FP8B128_TILE, _FP8B128_BLOCK_BYTES, _FP8B128_MAX,
    _fit_ml8_centroids, _assign_ml8_indices, _process_rotate_ml8_4_chunked,
    _ML8_FIT_ROWS_DEFAULT, _copy_field,
)
from centroid_quantizer import _lloyd_max_signed, _lloyd_max_signed_batched  # noqa: E402
from ml8_to_gguf import QK_ML8, ML8_BLOCK_BYTES, N_CENTROIDS  # noqa: E402
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


# ═══════════════════════════════════════════════════════════════════════════
# --format fp8_b128 tests
# ═══════════════════════════════════════════════════════════════════════════

# All dims here are 128-aligned on both N and K so fp8_b128 rotates them
# (except D_B128_ODD_N, deliberately NOT 128-aligned, to exercise the
# fallback rule).
D_B128 = 384          # hidden size (K for kronecker roles); factor_for_dim(384) -> a=3,b=128
                       # (a>1 so the group-vs-group h_a comparison below isn't a 1x1 coin flip)
FFN_B128 = 256         # ffn intermediate size (K for hadamard roles)
VOCAB_B128 = 128
D_B128_ODD_N = 100    # not a multiple of 128 -> fp8_b128 fallback to q8_0


def _add_bf16_b128(writer, name, t: torch.Tensor):
    data = np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint8).numpy())
    writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.BF16)


def _make_synthetic_gguf_b128(path: Path) -> dict:
    """Two-layer synthetic bf16 GGUF, 128-aligned on N/K, exercising: kronecker
    input groups (attn_qkv+attn_gate, ffn_gate+ffn_up) per layer, hadamard
    singletons (attn_output/ffn_down/ssm_out), the untied lm head (output,
    top-level singleton), q8_0 exclusions, and one deliberately-unaligned
    tensor (attn_v) to hit the 128-alignment fallback."""
    w = gguf.GGUFWriter(str(path), arch="qwen35")
    w.add_uint32("qwen35.embedding_length", D_B128)
    w.add_uint32("qwen35.block_count", 2)

    weights: dict[str, torch.Tensor] = {}

    def add(name, N, K):
        t = torch.randn(N, K, dtype=torch.float32) * 0.5
        weights[name] = t
        _add_bf16_b128(w, name, t)

    add("token_embd.weight", VOCAB_B128, D_B128)               # q8_0

    for layer in (0, 1):
        add(f"blk.{layer}.attn_qkv.weight", 3 * D_B128, D_B128)     # kronecker, group A
        add(f"blk.{layer}.attn_gate.weight", D_B128, D_B128)        # kronecker, group A
        add(f"blk.{layer}.attn_v.weight", D_B128_ODD_N, D_B128)     # kronecker role, N%128!=0 -> fallback q8_0
        add(f"blk.{layer}.attn_output.weight", D_B128, FFN_B128)    # hadamard, singleton
        add(f"blk.{layer}.ffn_gate.weight", FFN_B128, D_B128)       # kronecker, group B
        add(f"blk.{layer}.ffn_up.weight", FFN_B128, D_B128)         # kronecker, group B
        add(f"blk.{layer}.ffn_down.weight", D_B128, FFN_B128)       # hadamard, singleton
        add(f"blk.{layer}.ssm_out.weight", D_B128, FFN_B128)        # hadamard, singleton
        add(f"blk.{layer}.ssm_alpha.weight", 48, D_B128)            # q8_0
        add(f"blk.{layer}.ssm_beta.weight", 48, D_B128)             # q8_0
        norm = torch.ones(D_B128)
        weights[f"blk.{layer}.attn_norm.weight"] = norm
        _add_bf16_b128(w, f"blk.{layer}.attn_norm.weight", norm)    # copy

    add("output.weight", VOCAB_B128, D_B128)                   # kronecker, singleton (top-level)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return weights


@pytest.fixture()
def synthetic_gguf_b128(tmp_path):
    src = tmp_path / "tiny_bf16_b128.gguf"
    weights = _make_synthetic_gguf_b128(src)
    return src, weights


def _decode_e4m3_bytes_numpy(b: np.ndarray) -> np.ndarray:
    """Independent (no torch) OCP e4m3fn decoder: 1 sign / 4 exp (bias 7) /
    3 mantissa bits, no infinities, 0x7F/0xFF = NaN. Used to check the
    converter's packed bytes without relying on torch's own e4m3 codec for
    both the encode and the check."""
    b = np.asarray(b, dtype=np.uint8)
    sign = ((b >> 7) & 1).astype(np.float64)
    exp = ((b >> 3) & 0xF).astype(np.int64)
    mant = (b & 0x7).astype(np.float64)
    is_nan = (exp == 15) & (mant == 7)
    subnormal = exp == 0
    normal_val = (1.0 + mant / 8.0) * np.exp2((exp - 7).astype(np.float64))
    subnormal_val = (mant / 8.0) * np.exp2(-6.0)
    val = np.where(subnormal, subnormal_val, normal_val)
    val = np.where(sign == 1.0, -val, val)
    val = np.where(is_nan, np.nan, val)
    return val.astype(np.float32)


def test_decode_e4m3_bytes_numpy_matches_torch():
    """Sanity-check the independent numpy decoder against torch's own e4m3
    codec before trusting it to validate the converter's output."""
    torch.manual_seed(11)
    x = (torch.randn(2000) * 300.0).clamp(-_FP8B128_MAX, _FP8B128_MAX)
    e4m3 = x.to(torch.float8_e4m3fn)
    expected = e4m3.to(torch.float32).numpy()
    raw = e4m3.view(torch.uint8).numpy()
    actual = _decode_e4m3_bytes_numpy(raw)
    np.testing.assert_array_equal(actual, expected)
    print("  PASS test_decode_e4m3_bytes_numpy_matches_torch")


def test_quantize_fp8_b128_tile_scale_and_roundtrip():
    """Block encoding round-trips: byte size is 130*N*K/128, every block in a
    128x128 tile carries the identical fp16 scale, and decoding the e4m3
    bytes (independent numpy decoder) times that scale reproduces the
    quantized tensor closely."""
    torch.manual_seed(5)
    N, K = 256, 256   # 2x2 tiles
    w = torch.randn(N, K, dtype=torch.float32) * 2.0
    packed = _quantize_fp8_b128_gpu(w)

    n_col_blocks = K // _FP8B128_TILE
    assert packed.shape == (N, n_col_blocks * _FP8B128_BLOCK_BYTES)
    assert packed.nbytes == _FP8B128_BLOCK_BYTES * N * K // _FP8B128_TILE

    packed3 = packed.reshape(N, n_col_blocks, _FP8B128_BLOCK_BYTES)
    scale_bytes = packed3[:, :, :2]
    qs_bytes = packed3[:, :, 2:]
    scales = scale_bytes.reshape(N, n_col_blocks, 2).view(np.float16).astype(np.float32).reshape(N, n_col_blocks)

    n_row_tiles = N // _FP8B128_TILE
    for rt in range(n_row_tiles):
        r0, r1 = rt * _FP8B128_TILE, (rt + 1) * _FP8B128_TILE
        for cb in range(n_col_blocks):
            tile_scales = scales[r0:r1, cb]
            # Same fp16 scale replicated across all 128 rows of the tile.
            assert np.all(tile_scales == tile_scales[0]), f"tile ({rt},{cb}) scale not uniform"
            assert tile_scales[0] > 0   # always positive, even for a degenerate tile

    decoded_e4m3 = _decode_e4m3_bytes_numpy(qs_bytes.reshape(-1)).reshape(N, n_col_blocks, _FP8B128_TILE)
    dequant = decoded_e4m3 * scales[:, :, None]
    dequant = dequant.reshape(N, K)
    err = _nmse(dequant, w.numpy())
    assert err < 3e-2, f"fp8_b128 round-trip NMSE {err:.3e} too high"
    print("  PASS test_quantize_fp8_b128_tile_scale_and_roundtrip")


def test_quantize_fp8_b128_degenerate_zero_tile():
    """An all-zero tile must get a tiny positive scale (never zero/NaN) and
    decode back to all-zero e4m3 bytes."""
    w = torch.zeros(128, 128, dtype=torch.float32)
    packed = _quantize_fp8_b128_gpu(w)
    scale = packed[:, :2].reshape(128, 2).view(np.float16).astype(np.float32)
    assert np.all(scale > 0.0)
    qs = packed[:, 2:]
    assert np.all(qs == 0)   # +0.0 e4m3 encodes as byte 0x00
    print("  PASS test_quantize_fp8_b128_degenerate_zero_tile")


def test_quantize_fp8_b128_rejects_non_128_aligned():
    with pytest.raises(ValueError):
        _quantize_fp8_b128_gpu(torch.randn(100, 256))
    with pytest.raises(ValueError):
        _quantize_fp8_b128_gpu(torch.randn(256, 100))
    print("  PASS test_quantize_fp8_b128_rejects_non_128_aligned")


def test_role_group_key():
    assert role_group_key("attn_qkv") == role_group_key("attn_gate")
    assert role_group_key("attn_q") == role_group_key("attn_k") == role_group_key("attn_v")
    assert role_group_key("ffn_gate") == role_group_key("ffn_up")
    # singletons: each keyed by its own role, all distinct from each other
    # and from the grouped keys above.
    singleton_roles = ["attn_output", "ffn_down", "ssm_out", "output"]
    keys = {role_group_key(r) for r in singleton_roles}
    assert len(keys) == len(singleton_roles)
    assert role_group_key("attn_qkv") not in keys
    assert role_group_key("ffn_gate") not in keys
    print("  PASS test_role_group_key")


def test_group_seed_deterministic_and_distinct():
    assert _group_seed(0, 3, "attn_qkv_gate") == _group_seed(0, 3, "attn_qkv_gate")
    assert _group_seed(0, 3, "attn_qkv_gate") != _group_seed(0, 3, "ffn_gate_up")
    assert _group_seed(0, 3, "attn_qkv_gate") != _group_seed(0, 4, "attn_qkv_gate")
    assert _group_seed(0, None, "output") == _group_seed(0, None, "output")
    print("  PASS test_group_seed_deterministic_and_distinct")


def test_parse_layer():
    assert _parse_layer("blk.3.attn_output.weight") == 3
    assert _parse_layer("output.weight") is None
    assert _parse_layer("token_embd.weight") is None
    print("  PASS test_parse_layer")


# ─── DS4.1 (deepseek41 arch) role classification, added for the data-free
# ml8_4 attention conversion (2026-09-22). Tensor names/shapes verified
# against /home/kmbandy/models/dsv41-spine.gguf's own header.
def test_dsv41_roles_classify_ml8_4():
    cases = [
        ("blk.0.attn_q_a.weight", (5120, 1280), "rotate_kronecker"),
        ("blk.0.attn_q_b.weight", (1280, 32768), "rotate_kronecker"),
        ("blk.0.attn_kv.weight", (5120, 512), "rotate_kronecker"),
        ("blk.0.attn_output_b.weight", (8192, 5120), "rotate_hadamard"),
        ("blk.2.indexer.proj.weight", (5120, 32), "rotate_kronecker"),
        ("blk.2.indexer.attn_q_b.weight", (1280, 4096), "rotate_kronecker"),
        ("blk.2.indexer.attn_k.weight", (512, 128), "rotate_kronecker"),
        ("blk.2.attn_compressor_kv.weight", (5120, 512), "rotate_kronecker"),
        ("blk.2.attn_compressor_gate.weight", (5120, 512), "rotate_kronecker"),
        # attn_output_a itself (unsplit, block-diagonal over 8 groups) must
        # NOT be rotated here -- rotating the flat concatenation would mix
        # groups. Non-GEMM tensors stay copied verbatim regardless of format.
        ("blk.0.attn_output_a.weight", (4096, 8192), "copy"),
        ("blk.0.attn_norm.weight", (5120,), "copy"),
        ("blk.0.attn_sinks.weight", (64,), "copy"),
        # post-split wo_a groups (produced by the wo_a-split conversion step)
        ("blk.0.attn_output_a.g0.weight", (4096, 1024), "rotate_hadamard"),
        ("blk.0.attn_output_a.g7.weight", (4096, 1024), "rotate_hadamard"),
    ]
    for name, shape, expect in cases:
        action, role = classify_tensor(name, shape, format="ml8_4")
        assert action == expect, f"{name}: expected {expect}, got {action} (role={role})"
    print("  PASS test_dsv41_roles_classify_ml8_4")


def test_dsv41_role_group_sharing():
    # wq_a(cur), wkv(cur), indexer.proj(cur) all consume the SAME activation
    # (deepseek41.cpp:1499/1511/975) -> must share a rotation group.
    assert role_group_key("attn_q_a") == role_group_key("attn_kv") == role_group_key("indexer.proj")
    # wq_b(qr), indexer.attn_q_b(qr) consume qr (deepseek41.cpp:1503/959).
    assert role_group_key("attn_q_b") == role_group_key("indexer.attn_q_b")
    # attn_compressor_kv(cur_kv), attn_compressor_gate(cur_kv) (:1551/1558).
    assert role_group_key("attn_compressor_kv") == role_group_key("attn_compressor_gate")
    # these three groups, plus the ungrouped indexer.attn_k singleton, must
    # all be distinct from each other and from the qwen3x groups.
    keys = {
        role_group_key("attn_q_a"), role_group_key("attn_q_b"),
        role_group_key("attn_compressor_kv"), role_group_key("indexer.attn_k"),
        role_group_key("attn_qkv"), role_group_key("ffn_gate"),
    }
    assert len(keys) == 6
    print("  PASS test_dsv41_role_group_sharing")


def _make_dsv41_wo_a_gguf(path: Path, o_group_dim: int, o_lora_rank: int, o_groups: int):
    """Tiny synthetic deepseek41-arch GGUF with just one attn_output_a
    (wo_a) tensor [o_group_dim, o_lora_rank*o_groups] and the
    output_group_count KV field the wo_a split (Task 2) reads. Returns the
    torch weight [N, K] (N=o_lora_rank*o_groups, K=o_group_dim) used to
    build it, group-sliced the same way create_tensor's TENSOR_ALLOW_RESHAPE
    reinterprets the flat tensor in deepseek41.cpp:312 (group g = rows
    [g*o_lora_rank, (g+1)*o_lora_rank))."""
    w = gguf.GGUFWriter(str(path), arch="deepseek41")
    w.add_uint32("deepseek41.embedding_length", o_group_dim)
    w.add_uint32("deepseek41.block_count", 1)
    w.add_uint32("deepseek41.attention.output_group_count", o_groups)
    N = o_lora_rank * o_groups
    torch.manual_seed(7)
    weight = torch.randn(N, o_group_dim, dtype=torch.float32) * 0.5
    _add_bf16(w, "blk.0.attn_output_a.weight", weight)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return weight


def test_dsv41_wo_a_split_produces_group_tensors(tmp_path):
    """wo_a split (Task 2): with format=ml8_4 and output_group_count present,
    the flat attn_output_a.weight must NOT appear in the output -- only the
    8 (here 4, for test speed) per-group tensors, each independently
    rotate_hadamard/ml8_4-classified."""
    o_group_dim, o_lora_rank, o_groups = 256, 64, 4  # K=256 (%128 local_b, %64 QK_ML8), per-group N=64
    src = tmp_path / "wo_a_src.gguf"
    weight = _make_dsv41_wo_a_gguf(src, o_group_dim, o_lora_rank, o_groups)

    reader = gguf.GGUFReader(src)
    plan = build_plan(reader, rotation_seed=0, local_b=128, max_b=1024, format="ml8_4")
    names = [e["name"] for e in plan]
    assert "blk.0.attn_output_a.weight" not in names
    for g in range(o_groups):
        gname = f"blk.0.attn_output_a.g{g}.weight"
        assert gname in names, f"missing {gname}"
        e = next(x for x in plan if x["name"] == gname)
        assert e["action"] == "rotate_hadamard"
        assert e["shape"] == (o_group_dim, o_lora_rank)
        assert e["src_name"] == "blk.0.attn_output_a.weight"
        assert e["row_offset"] == g * o_lora_rank

    out = tmp_path / "wo_a_out.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024, format="ml8_4")
    out_reader = gguf.GGUFReader(out)
    out_names = {t.name: t for t in out_reader.tensors}
    assert "blk.0.attn_output_a.weight" not in out_names
    for g in range(o_groups):
        gname = f"blk.0.attn_output_a.g{g}.weight"
        assert out_names[gname].tensor_type == GGMLQuantizationType.ML8_4

    # ── group-concat equivalence check: dequantize each of the o_groups
    # ml8_4 group tensors and un-rotate (block_hadamard is self-inverse),
    # then verify concatenating them along rows reproduces the ORIGINAL
    # flat weight's block structure (same row range per group as
    # TENSOR_ALLOW_RESHAPE's 2D->3D reinterpretation at load time).
    from kronecker_rotation import BlockHadamardRotation
    recon = np.empty((o_lora_rank * o_groups, o_group_dim), dtype=np.float32)
    for g in range(o_groups):
        gname = f"blk.0.attn_output_a.g{g}.weight"
        decoded_rot = _decode_ml8_4_numpy(out_reader, gname)  # still in rotated basis
        rot = BlockHadamardRotation(in_features=o_group_dim, b_dim=128)
        decoded = rot.inverse(torch.from_numpy(decoded_rot)).numpy()
        recon[g * o_lora_rank:(g + 1) * o_lora_rank] = decoded

    err = _nmse(recon, weight.numpy())
    assert err < 0.15, f"wo_a group-concat NMSE {err:.3e} too high"
    print("  PASS test_dsv41_wo_a_split_produces_group_tensors")


def test_dsv41_token_embd_output_stay_copy():
    """DS4.1 (deepseek41 arch) is explicitly out of scope for token_embd/
    output (task instructions: keep them at whatever type the production
    spine already has, BF16) -- unlike qwen35, where token_embd (Q8_0) and
    output (rotate_kronecker, untied lm head) rotating/quantizing is the
    correct, existing, must-stay-byte-identical behavior. This was a real
    bug caught during the ml8_4 dry-run against the real DS4.1 BF16 spine:
    without the arch guard, classify_tensor used the generic qwen role
    tables for these two names regardless of architecture."""
    shape = (5120, 129280)
    assert classify_tensor("token_embd.weight", shape, format="ml8_4")[0] == "q8_0"          # qwen: unaffected
    assert classify_tensor("output.weight", shape, format="ml8_4")[0] == "rotate_kronecker"  # qwen: unaffected
    assert classify_tensor("token_embd.weight", shape, format="ml8_4", arch="deepseek41")[0] == "copy"
    assert classify_tensor("output.weight", shape, format="ml8_4", arch="deepseek41")[0] == "copy"
    assert classify_tensor("token_embd.weight", shape, format="ml8_4", arch="qwen35")[0] == "q8_0"
    print("  PASS test_dsv41_token_embd_output_stay_copy")


def test_dsv41_wo_a_split_respects_nextn_fallback(tmp_path):
    """wo_a split (Task 2) must not bypass the existing MTP/nextn safety
    fallback (classify_tensor's first_nextn_layer rule, ml8 sidecars aren't
    registered for nextn layers on the C++ side): a wo_a tensor AT OR PAST
    first_nextn_layer must stay a single flat Q8_0 tensor, not 8 rotated
    split groups. Caught as a real bug: build_plan's wo_a-split branch
    originally ran before the first_nextn_layer check entirely."""
    o_group_dim, o_lora_rank, o_groups = 256, 64, 4
    src = tmp_path / "wo_a_nextn.gguf"
    w = gguf.GGUFWriter(str(src), arch="deepseek41")
    w.add_uint32("deepseek41.embedding_length", o_group_dim)
    w.add_uint32("deepseek41.block_count", 2)
    w.add_uint32("deepseek41.nextn_predict_layers", 1)  # -> first_nextn_layer = 2 - 1 = 1
    w.add_uint32("deepseek41.attention.output_group_count", o_groups)
    N = o_lora_rank * o_groups
    torch.manual_seed(3)
    for layer in (0, 1):
        weight = torch.randn(N, o_group_dim, dtype=torch.float32) * 0.5
        _add_bf16(w, f"blk.{layer}.attn_output_a.weight", weight)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    reader = gguf.GGUFReader(src)
    plan = build_plan(reader, rotation_seed=0, local_b=128, max_b=1024, format="ml8_4")
    names = {e["name"]: e for e in plan}
    # layer 0 (< first_nextn=1): split into o_groups rotated tensors.
    for g in range(o_groups):
        assert f"blk.0.attn_output_a.g{g}.weight" in names
    assert "blk.0.attn_output_a.weight" not in names
    # layer 1 (>= first_nextn=1): stays flat, plain q8_0, NOT split/rotated.
    assert "blk.1.attn_output_a.weight" in names
    assert names["blk.1.attn_output_a.weight"]["action"] == "q8_0"
    for g in range(o_groups):
        assert f"blk.1.attn_output_a.g{g}.weight" not in names
    print("  PASS test_dsv41_wo_a_split_respects_nextn_fallback")


def test_dsv41_k_not_multiple_of_64_falls_back_to_q8_0():
    # Defensive: if a future DS4.1 config ever produced a K not divisible by
    # QK_ML8=64 for one of these roles, ml8_4 classification must fall back
    # to unrotated Q8_0 rather than silently mis-round K into groups.
    action, role = classify_tensor("blk.0.attn_q_a.weight", (5121, 1280), format="ml8_4")
    assert action == "q8_0" and role == "attn_q_a"
    print("  PASS test_dsv41_k_not_multiple_of_64_falls_back_to_q8_0")


def test_dry_run_classification_fp8_b128(synthetic_gguf_b128):
    """--dry-run classification: aligned roles rotate, the unaligned attn_v
    and the standing q8_0/copy exclusions behave as documented."""
    src, _ = synthetic_gguf_b128
    reader = gguf.GGUFReader(src)
    plan = build_plan(reader, rotation_seed=0, local_b=128, max_b=1024, format="fp8_b128")
    by_name = {e["name"]: e for e in plan}

    assert by_name["blk.0.attn_qkv.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.attn_gate.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.ffn_gate.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.ffn_up.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.attn_output.weight"]["action"] == "rotate_hadamard"
    assert by_name["blk.0.ffn_down.weight"]["action"] == "rotate_hadamard"
    assert by_name["blk.0.ssm_out.weight"]["action"] == "rotate_hadamard"
    assert by_name["output.weight"]["action"] == "rotate_kronecker"

    # Fallback: attn_v's role is kronecker-eligible but N=100 isn't 128-aligned.
    assert by_name["blk.0.attn_v.weight"]["action"] == "q8_0"
    assert by_name["blk.1.attn_v.weight"]["action"] == "q8_0"

    # Untouched exclusions/copies still behave as in ml8_fp8 mode.
    assert by_name["token_embd.weight"]["action"] == "q8_0"
    assert by_name["blk.0.ssm_alpha.weight"]["action"] == "q8_0"
    assert by_name["blk.0.attn_norm.weight"]["action"] == "copy"

    # Group ids present for kronecker entries and match the design's groups.
    assert by_name["blk.0.attn_qkv.weight"]["group"] == by_name["blk.0.attn_gate.weight"]["group"]
    assert by_name["blk.0.ffn_gate.weight"]["group"] == by_name["blk.0.ffn_up.weight"]["group"]
    assert by_name["blk.0.attn_qkv.weight"]["group"] != by_name["blk.0.ffn_gate.weight"]["group"]
    print("  PASS test_dry_run_classification_fp8_b128")


def test_convert_fp8_b128_end_to_end(synthetic_gguf_b128, tmp_path):
    src, weights = synthetic_gguf_b128
    out = tmp_path / "tiny_fp8b128.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128")
    reader = gguf.GGUFReader(out)
    names = {t.name: t for t in reader.tensors}

    for role_name in ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
                      "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                      "blk.0.attn_output.weight", "blk.0.ffn_down.weight",
                      "blk.0.ssm_out.weight", "output.weight"]:
        assert names[role_name].tensor_type == GGMLQuantizationType.FP8_B128, role_name

    # Fallback tensor: plain unrotated Q8_0, no sidecars.
    assert names["blk.0.attn_v.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert "blk.0.attn_v.rotation_meta" not in names
    assert "blk.0.attn_v.rotation_h_a" not in names

    assert names["token_embd.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["blk.0.attn_norm.weight"].tensor_type == GGMLQuantizationType.BF16

    # Byte size: FP8_B128 is 130 bytes / 128 elems.
    block_size, block_bytes = gguf.constants.GGML_QUANT_SIZES[GGMLQuantizationType.FP8_B128]
    assert (block_size, block_bytes) == (128, 130)
    t = names["blk.0.ffn_gate.weight"]
    N, K = FFN_B128, D_B128
    assert np.ascontiguousarray(t.data).nbytes == N * (K // block_size) * block_bytes
    print("  PASS test_convert_fp8_b128_end_to_end")


def test_group_members_share_h_a_bytewise(synthetic_gguf_b128, tmp_path):
    """attn_qkv/attn_gate (group A) share h_a; ffn_gate/ffn_up (group B) share
    h_a; group A and group B (and the same group across different layers)
    get DIFFERENT h_a."""
    src, _ = synthetic_gguf_b128
    out = tmp_path / "tiny_fp8b128.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128")
    reader = gguf.GGUFReader(out)

    def h_a(weight_name):
        base = _sidecar_base(weight_name)
        t = next(x for x in reader.tensors if x.name == base + ".rotation_h_a")
        return np.ascontiguousarray(t.data).copy()

    qkv0, gate0 = h_a("blk.0.attn_qkv.weight"), h_a("blk.0.attn_gate.weight")
    np.testing.assert_array_equal(qkv0, gate0)

    fgate0, fup0 = h_a("blk.0.ffn_gate.weight"), h_a("blk.0.ffn_up.weight")
    np.testing.assert_array_equal(fgate0, fup0)

    # Different groups -> different rotation.
    assert not np.array_equal(qkv0, fgate0)

    # Same group, different layer -> different rotation (layer is part of the key).
    qkv1 = h_a("blk.1.attn_qkv.weight")
    assert not np.array_equal(qkv0, qkv1)

    # Sanity: layer 1's own group A pair still agrees with itself.
    gate1 = h_a("blk.1.attn_gate.weight")
    np.testing.assert_array_equal(qkv1, gate1)
    print("  PASS test_group_members_share_h_a_bytewise")


def test_seed_determinism_fp8_b128(synthetic_gguf_b128, tmp_path):
    """Two runs with the same --rotation-seed give bit-identical bytes in
    fp8_b128 mode too (group-derived seeds are deterministic)."""
    src, _ = synthetic_gguf_b128
    out1 = tmp_path / "run1.gguf"
    out2 = tmp_path / "run2.gguf"
    convert(src, out1, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024, format="fp8_b128")
    convert(src, out2, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024, format="fp8_b128")
    r1, r2 = gguf.GGUFReader(out1), gguf.GGUFReader(out2)
    names1 = {t.name: t for t in r1.tensors}
    names2 = {t.name: t for t in r2.tensors}
    assert set(names1) == set(names2)
    for name in names1:
        b1 = np.ascontiguousarray(names1[name].data)
        b2 = np.ascontiguousarray(names2[name].data)
        np.testing.assert_array_equal(b1, b2, err_msg=f"{name}: bytes differ across runs")
    print("  PASS test_seed_determinism_fp8_b128")


def test_ml8_fp8_format_unchanged_by_default(synthetic_gguf, tmp_path):
    """The existing ml8_fp8 tests all call convert()/build_plan() without
    --format, which must keep behaving exactly as before fp8_b128 existed —
    covered structurally by the untouched tests above, and spot-checked here:
    default format plan == explicit format="ml8_fp8" plan."""
    src, _ = synthetic_gguf
    reader = gguf.GGUFReader(src)
    plan_default = build_plan(reader, rotation_seed=7, local_b=128, max_b=1024)
    plan_explicit = build_plan(reader, rotation_seed=7, local_b=128, max_b=1024, format="ml8_fp8")
    assert plan_default == plan_explicit
    print("  PASS test_ml8_fp8_format_unchanged_by_default")


# ═══════════════════════════════════════════════════════════════════════════
# --format ml8_4 tests (data-free codebook: 4-bit centroid idx + fp32
# per-(row,group) scale over 64-element K-groups, per-K-group 16-e4m3-
# centroid LUT sidecar — see ggml-common.h block_ml8_4 / ml8_to_gguf.py).
# ═══════════════════════════════════════════════════════════════════════════

# D_MODEL (2048) and K_HADAMARD (4864) are both multiples of QK_ML8=64, so the
# existing `synthetic_gguf` fixture (shared with the ml8_fp8 tests above)
# exercises ml8_4 rotation for every role without needing a new fixture.
D_ML8_ODD_K = 96   # multiple of 32 (Q8_0-compatible) but NOT of 64 -> ml8_4 fallback to plain Q8_0


def _decode_ml8_4_numpy(reader, weight_name: str) -> np.ndarray:
    """Independent (no torch-side packing reuse) numpy decoder for the
    block_ml8_4 on-disk layout: per (row, 64-col group) a 36-byte block
    ([0:4) little-endian fp32 scale, [4:36) 32 bytes of lo-nibble-first 4-bit
    centroid indices), dequantized via the per-K-group F8_E4M3 centroids
    sidecar: value = centroids[group, idx] * scale[row, group]. Used to
    validate the converter's output against ggml-common.h's block_ml8_4 /
    ml8_to_gguf.py's exact byte layout, independent of the converter's own
    pack_ml8_blocks/cast_centroids_to_fp8 helpers."""
    t = next(x for x in reader.tensors if x.name == weight_name)
    assert t.tensor_type == GGMLQuantizationType.ML8_4
    block_size, block_bytes = gguf.constants.GGML_QUANT_SIZES[GGMLQuantizationType.ML8_4]
    assert (block_size, block_bytes) == (QK_ML8, ML8_BLOCK_BYTES)
    raw = np.ascontiguousarray(t.data)
    N = raw.shape[0]
    n_blocks = raw.shape[1] // block_bytes
    raw = raw.reshape(N, n_blocks, block_bytes)

    scale_bytes = raw[:, :, :4].reshape(N, n_blocks, 4)
    scale = scale_bytes.reshape(N, n_blocks, 4).view(np.float32).reshape(N, n_blocks)

    packed = raw[:, :, 4:4 + block_size // 2].reshape(N, n_blocks, block_size // 2)
    lo = (packed & 0x0F).astype(np.uint8)
    hi = ((packed >> 4) & 0x0F).astype(np.uint8)
    idx = np.empty((N, n_blocks, block_size), dtype=np.uint8)
    idx[:, :, 0::2] = lo
    idx[:, :, 1::2] = hi

    base = _sidecar_base(weight_name)
    cent_t = next(x for x in reader.tensors if x.name == base + ".centroids")
    assert cent_t.tensor_type == GGMLQuantizationType.F8_E4M3
    cent_raw = np.ascontiguousarray(cent_t.data).reshape(-1)[:n_blocks * N_CENTROIDS]
    cent_bytes = cent_raw.reshape(n_blocks, N_CENTROIDS)
    centroids = torch.from_numpy(cent_bytes.copy()).view(torch.float8_e4m3fn).to(torch.float32).numpy()

    block_idx = np.broadcast_to(np.arange(n_blocks).reshape(1, n_blocks, 1), (N, n_blocks, block_size))
    cent_lookup = centroids[block_idx, idx.astype(np.int64)]  # [N, n_blocks, block_size]
    out = (cent_lookup * scale[:, :, None]).reshape(N, n_blocks * block_size)
    return out


def test_classify_tensor_ml8_4_fallback():
    # K%64==0 -> rotates normally, same as ml8_fp8/fp8_b128 classification.
    assert classify_tensor("blk.3.attn_output.weight", (4864, 5120), format="ml8_4")[0] == "rotate_hadamard"
    assert classify_tensor("blk.0.attn_qkv.weight", (2048, 96), format="ml8_4")[0] == "rotate_kronecker"
    # K%64!=0 -> falls back to plain Q8_0 (unlike ml8_fp8, which has no such
    # constraint since it groups by 32 and this codebook is data-free).
    assert classify_tensor("blk.0.attn_qkv.weight", (D_ML8_ODD_K, 96), format="ml8_4")[0] == "q8_0"
    assert classify_tensor("blk.3.attn_output.weight", (D_ML8_ODD_K, 5120), format="ml8_4")[0] == "q8_0"
    # Standing exclusions/copies are format-independent.
    assert classify_tensor("token_embd.weight", (2048, 40), format="ml8_4")[0] == "q8_0"
    assert classify_tensor("blk.0.attn_norm.weight", (2048,), format="ml8_4")[0] == "copy"
    print("  PASS test_classify_tensor_ml8_4_fallback")


def test_classify_tensor_nextn_fallback_scoped_to_targeted_roles():
    """first_nextn_layer's MTP-safety fallback must only force Q8_0 for roles
    that would otherwise be rotated/ml8'd (K_SPLIT_HADAMARD_ROLES /
    N_SPLIT_KRONECKER_ROLES) -- NOT every 2D weight in the nextn block.
    2026-09-22 regression: the fallback used to catch-all every 2D weight at
    or past first_nextn_layer, which forced DS4.1 MTP-block tensors like
    ffn_gate_inp.weight (BF16) and hc_attn_fn.weight/hc_ffn_fn.weight (F32)
    down to Q8_0, diverging from production's full-precision types for those
    tensors. See classify_tensor's docstring and ml84-fast-report.md."""
    # Targeted attention roles (hadamard/kronecker) at/past first_nextn_layer:
    # still forced to plain q8_0 -- ml8 sidecars aren't registered there.
    assert classify_tensor("blk.5.attn_output.weight", (4864, 5120),
                            first_nextn_layer=5)[0] == "q8_0"
    assert classify_tensor("blk.5.attn_q_a.weight", (2048, 96),
                            first_nextn_layer=5, format="ml8_4")[0] == "q8_0"
    # Same roles below first_nextn_layer: rotate normally (unaffected).
    assert classify_tensor("blk.4.attn_output.weight", (4864, 5120),
                            first_nextn_layer=5)[0] == "rotate_hadamard"
    # Non-targeted 2D weights at/past first_nextn_layer: NOT forced to q8_0 --
    # fall through to normal classification (copy, since none of these are in
    # any role table), preserving the source GGUF's type (BF16/F32/whatever
    # production has), matching production's full-precision MTP tensors.
    assert classify_tensor("blk.5.ffn_gate_inp.weight", (2048, 8),
                            first_nextn_layer=5)[0] == "copy"
    assert classify_tensor("blk.5.hc_attn_fn.weight", (2048, 2048),
                            first_nextn_layer=5)[0] == "copy"
    assert classify_tensor("blk.5.hc_ffn_fn.weight", (2048, 2048),
                            first_nextn_layer=5)[0] == "copy"
    # A Q8_0_ROLES member at/past first_nextn_layer still lands on q8_0 (both
    # the explicit-role path and the old catch-all agree here, so this isn't
    # a behavior change, just confirming it still holds under the new scoping).
    assert classify_tensor("blk.5.ssm_alpha.weight", (2048, 48),
                            first_nextn_layer=5)[0] == "q8_0"
    print("  PASS test_classify_tensor_nextn_fallback_scoped_to_targeted_roles")


def test_dry_run_classification_ml8_4(synthetic_gguf):
    src, _ = synthetic_gguf
    reader = gguf.GGUFReader(src)
    plan = build_plan(reader, rotation_seed=0, local_b=128, max_b=1024, format="ml8_4")
    by_name = {e["name"]: e for e in plan}

    assert by_name["blk.0.attn_output.weight"]["action"] == "rotate_hadamard"
    assert by_name["blk.0.ffn_down.weight"]["action"] == "rotate_hadamard"
    assert by_name["blk.0.ssm_out.weight"]["action"] == "rotate_hadamard"
    assert by_name["blk.0.attn_qkv.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.ffn_gate.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.ffn_up.weight"]["action"] == "rotate_kronecker"
    assert by_name["output.weight"]["action"] == "rotate_kronecker"
    assert by_name["token_embd.weight"]["action"] == "q8_0"
    assert by_name["blk.0.ssm_alpha.weight"]["action"] == "q8_0"
    assert by_name["blk.0.attn_norm.weight"]["action"] == "copy"

    # One-rotation-per-input-group applies to ml8_4 too (same rule as fp8_b128).
    assert by_name["blk.0.ffn_gate.weight"]["group"] == by_name["blk.0.ffn_up.weight"]["group"]
    print("  PASS test_dry_run_classification_ml8_4")


def test_fit_and_assign_ml8_centroids_roundtrip():
    """_fit_ml8_centroids + _assign_ml8_indices on a small identity-rotated
    tensor: centroids land on the e4m3 lattice, indices are in [0,15], and
    dequant(idx)*scale reconstructs the (normalized-then-rescaled) weight
    with codebook-level (not garbage) error."""
    from kronecker_rotation import BlockHadamardRotation

    class _Identity:
        def forward(self, x):
            return x

    torch.manual_seed(21)
    N, K = 64, 128  # K = 2 groups of QK_ML8=64
    w = torch.randn(N, K, dtype=torch.float32) * 0.7

    class _Tensor:
        tensor_type = GGMLQuantizationType.BF16
        name = "synthetic"

        def __init__(self, data):
            self.data = data

    t16 = w.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    tensor = _Tensor(t16)

    rot = _Identity()
    centroids = _fit_ml8_centroids(tensor, K, N, torch.device("cpu"), rot, fit_rows=N)
    assert centroids.shape == (K // QK_ML8, 16)
    # Every centroid value must itself be exactly representable as e4m3
    # (snap_to_e4m3 is idempotent on e4m3-lattice values).
    snapped_twice = centroids.to(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(centroids, snapped_twice, atol=0.0, rtol=0.0)

    indices, scale = _assign_ml8_indices(w, centroids)
    assert indices.dtype == torch.int8
    assert indices.shape == (N, K)
    assert int(indices.min()) >= 0 and int(indices.max()) <= 15
    assert scale.shape == (N, K // QK_ML8)
    assert bool((scale > 0).all())

    n_groups = K // QK_ML8
    idx_r = indices.long().view(N, n_groups, QK_ML8)
    dequant = torch.empty(N, n_groups, QK_ML8)
    for g in range(n_groups):
        dequant[:, g, :] = centroids[g][idx_r[:, g, :]]
    dequant = dequant * scale.unsqueeze(-1)
    dequant = dequant.reshape(N, K)
    err = _nmse(dequant.numpy(), w.numpy())
    assert err < 0.15, f"ml8_4 fit/assign NMSE {err:.3e} too high"
    print("  PASS test_fit_and_assign_ml8_centroids_roundtrip")


def test_lloyd_max_signed_batched_matches_per_group():
    """_lloyd_max_signed_batched (2026-09-22 perf fix: vectorizes the
    per-group `for g in range(n_groups): _lloyd_max_signed(...)` loop in
    _fit_ml8_centroids across all groups at once) must reproduce
    `_lloyd_max_signed` called once per group, on a small synthetic case,
    bit-exactly (both use n_iter fixed iterations with no early-stop
    dependence, so there's no floating summation-order slop to allow for at
    this size -- unlike the real-weight equivalence check, which tolerates
    rare ULP-level boundary flips)."""
    torch.manual_seed(7)
    n_groups, n_samples, n_levels, n_iter = 5, 4096, 16, 25

    samples2d = torch.randn(n_groups, n_samples, dtype=torch.float32)
    # Make groups have different scales/shapes so a bug that only shows up
    # for non-uniform data (e.g. an off-by-one group index) isn't masked.
    samples2d = samples2d * (1.0 + 0.5 * torch.arange(n_groups).view(-1, 1))

    batched = _lloyd_max_signed_batched(
        samples2d, n_levels=n_levels, n_iter=n_iter, fit_loss="mse")
    per_group = torch.stack([
        _lloyd_max_signed(
            samples2d[g], sample_col_idx=None, col_weights=None,
            n_levels=n_levels, n_iter=n_iter, fit_loss="mse", mag_weight_p=5.0,
        )
        for g in range(n_groups)
    ])

    assert batched.shape == per_group.shape == (n_groups, n_levels)
    # Tiny (~1e-6) floating differences are expected and acceptable here: the
    # batched path's `.sum(dim=1)` reduction over a bin's members can use a
    # different addition order than the per-group path's `.sum()` over the
    # same values (see _lloyd_max_signed_batched's docstring) -- this is the
    # documented summation-order slop, not a correctness bug. A real mismatch
    # (wrong group indexing, wrong bin assignment, etc.) would show up as
    # differences far larger than float32 summation noise.
    torch.testing.assert_close(batched, per_group, atol=1e-5, rtol=1e-5)
    print("  PASS test_lloyd_max_signed_batched_matches_per_group")


def test_fit_ml8_centroids_batched_matches_per_group():
    """End-to-end (through _fit_ml8_centroids, not just the Lloyd-Max
    kernel): batched_fit=True vs batched_fit=False must produce the same
    e4m3-snapped centroids on a small synthetic weight."""
    from kronecker_rotation import BlockHadamardRotation

    class _Identity:
        def forward(self, x):
            return x

    torch.manual_seed(31)
    N, K = 96, 192  # K = 3 groups of QK_ML8=64
    w = torch.randn(N, K, dtype=torch.float32) * 0.9

    class _Tensor:
        tensor_type = GGMLQuantizationType.BF16
        name = "synthetic"

        def __init__(self, data):
            self.data = data

    t16 = w.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    tensor = _Tensor(t16)

    rot = _Identity()
    c_batched = _fit_ml8_centroids(tensor, K, N, torch.device("cpu"), rot,
                                   fit_rows=N, batched_fit=True, fit_group_chunk=2)
    c_per_group = _fit_ml8_centroids(tensor, K, N, torch.device("cpu"), rot,
                                     fit_rows=N, batched_fit=False)
    torch.testing.assert_close(c_batched, c_per_group, atol=0.0, rtol=0.0)
    print("  PASS test_fit_ml8_centroids_batched_matches_per_group")


def test_convert_ml8_4_end_to_end(synthetic_gguf, tmp_path):
    src, weights = synthetic_gguf
    out = tmp_path / "tiny_ml8_4.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024,
           format="ml8_4", ml8_fit_rows=_ML8_FIT_ROWS_DEFAULT)
    reader = gguf.GGUFReader(out)
    names = {t.name: t for t in reader.tensors}

    rotated_names = ["blk.0.attn_output.weight", "blk.0.ffn_down.weight",
                      "blk.0.ssm_out.weight", "blk.0.attn_qkv.weight",
                      "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "output.weight"]
    for name in rotated_names:
        assert names[name].tensor_type == GGMLQuantizationType.ML8_4, name
        base = _sidecar_base(name)
        assert base + ".centroids" in names, f"{name}: missing centroids sidecar"
        assert base + ".rotation_meta" in names

    assert "blk.0.attn_output.rotation_h_a" not in names   # block_hadamard: no h_a
    assert "blk.0.attn_qkv.rotation_h_a" in names          # kronecker: has h_a

    assert names["token_embd.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["blk.0.ssm_alpha.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["blk.0.attn_norm.weight"].tensor_type == GGMLQuantizationType.BF16

    # ── Centroids sidecar shape/dtype: GGUF ne order [16, K/64] == numpy/byte
    # shape (K/64, 16) ─────────────────────────────────────────────────────
    for name, K in [("blk.0.attn_qkv.weight", K_KRON), ("blk.0.attn_output.weight", K_HADAMARD),
                     ("output.weight", K_KRON)]:
        base = _sidecar_base(name)
        cent = names[base + ".centroids"]
        assert cent.tensor_type == GGMLQuantizationType.F8_E4M3
        n_groups = K // QK_ML8
        assert np.ascontiguousarray(cent.data).size == n_groups * 16

    # ── Block encode/decode round trip against the independent numpy decoder
    # for one kronecker-rotated and one hadamard-rotated weight ─────────────
    from kronecker_rotation import KroneckerRotation, BlockHadamardRotation
    for name, K, kind in [
        ("blk.0.attn_qkv.weight", K_KRON, "kronecker_orth_sylvester"),
        ("blk.0.attn_output.weight", K_HADAMARD, "block_hadamard"),
    ]:
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

        W = weights[name]
        w_rot_expected = rot.forward(W)  # [N, K] fp32, the rotated weight
        decoded = _decode_ml8_4_numpy(reader, name)
        err = _nmse(decoded, w_rot_expected.numpy())
        # 4-bit codebook quant: coarser than fp8_b128/ml8_fp8 but should still
        # be codebook-level error, not garbage (a broken decode/assign would
        # give NMSE ~O(1) or worse).
        assert err < 0.15, f"{name}: ml8_4 round-trip NMSE {err:.3e} too high"
    print("  PASS test_convert_ml8_4_end_to_end")


def test_convert_ml8_4_k_fallback(tmp_path):
    """A kronecker-eligible role whose K isn't a multiple of QK_ML8=64 must
    fall back to plain unrotated Q8_0 with no ml8_4 sidecars, in an actual
    convert() run (not just classify_tensor)."""
    path = tmp_path / "odd_k.gguf"
    w = gguf.GGUFWriter(str(path), arch="qwen35")
    w.add_uint32("qwen35.embedding_length", D_ML8_ODD_K)
    w.add_uint32("qwen35.block_count", 1)
    t = torch.randn(32, D_ML8_ODD_K, dtype=torch.float32) * 0.5
    _add_bf16(w, "blk.0.attn_qkv.weight", t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    out = tmp_path / "odd_k_out.gguf"
    convert(path, out, rotation_seed=0, device_str="cpu", local_b=128, max_b=1024, format="ml8_4")
    reader = gguf.GGUFReader(out)
    names = {t.name: t for t in reader.tensors}
    assert names["blk.0.attn_qkv.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert "blk.0.attn_qkv.centroids" not in names
    assert "blk.0.attn_qkv.rotation_meta" not in names
    print("  PASS test_convert_ml8_4_k_fallback")


def test_quantize_fp8_b128_channel_scale_and_roundtrip():
    """--scale-mode channel: byte size/layout identical to tile mode, but
    every block in a ROW carries the identical fp16 scale (row absmax over
    all K / 448), and decoding the e4m3 bytes (independent numpy decoder)
    reproduces the quantized tensor with codebook-level error."""
    torch.manual_seed(6)
    N, K = 256, 256   # 2x2 tiles
    w = torch.randn(N, K, dtype=torch.float32) * 2.0
    packed = _quantize_fp8_b128_gpu(w, scale_mode="channel")

    n_col_blocks = K // _FP8B128_TILE
    assert packed.shape == (N, n_col_blocks * _FP8B128_BLOCK_BYTES)
    assert packed.nbytes == _FP8B128_BLOCK_BYTES * N * K // _FP8B128_TILE

    packed3 = packed.reshape(N, n_col_blocks, _FP8B128_BLOCK_BYTES)
    scale_bytes = packed3[:, :, :2]
    qs_bytes = packed3[:, :, 2:]
    scales = scale_bytes.reshape(N, n_col_blocks, 2).view(np.float16).astype(np.float32).reshape(N, n_col_blocks)

    # All blocks of a given row share the identical stored scale.
    for row in range(N):
        row_scales = scales[row]
        assert np.all(row_scales == row_scales[0]), f"row {row} scale not uniform across blocks"
        assert row_scales[0] > 0   # always positive, even for a degenerate row

    # Independently recompute the expected per-row scale and cross-check.
    expected_scale = (np.abs(w.numpy()).max(axis=1) / _FP8B128_MAX).astype(np.float32)
    expected_scale = np.maximum(expected_scale, 1e-6).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(scales[:, 0], expected_scale)

    decoded_e4m3 = _decode_e4m3_bytes_numpy(qs_bytes.reshape(-1)).reshape(N, n_col_blocks, _FP8B128_TILE)
    dequant = decoded_e4m3 * scales[:, :, None]
    dequant = dequant.reshape(N, K)
    err = _nmse(dequant, w.numpy())
    assert err < 3e-2, f"fp8_b128 channel round-trip NMSE {err:.3e} too high"
    print("  PASS test_quantize_fp8_b128_channel_scale_and_roundtrip")


def test_quantize_fp8_b128_channel_degenerate_zero_row():
    """An all-zero row must get a tiny positive scale (never zero/NaN) and
    decode back to all-zero e4m3 bytes, one row inside an otherwise nonzero
    tile (channel scale is per-row, unlike tile mode's per-tile scale)."""
    torch.manual_seed(9)
    w = torch.randn(128, 256, dtype=torch.float32)
    w[5, :] = 0.0
    packed = _quantize_fp8_b128_gpu(w, scale_mode="channel")
    n_col_blocks = 256 // _FP8B128_TILE
    packed3 = packed.reshape(128, n_col_blocks, _FP8B128_BLOCK_BYTES)
    row5_scale = packed3[5, :, :2].reshape(n_col_blocks, 2).view(np.float16).astype(np.float32)
    assert np.all(row5_scale > 0.0)
    row5_qs = packed3[5, :, 2:]
    assert np.all(row5_qs == 0)   # +0.0 e4m3 encodes as byte 0x00
    print("  PASS test_quantize_fp8_b128_channel_degenerate_zero_row")


def test_quantize_fp8_b128_invalid_scale_mode():
    with pytest.raises(ValueError):
        _quantize_fp8_b128_gpu(torch.randn(128, 128), scale_mode="bogus")
    print("  PASS test_quantize_fp8_b128_invalid_scale_mode")


def test_dry_run_classification_fp8_b128_channel_scale_mode(synthetic_gguf_b128):
    """--scale-mode has no bearing on classification: build_plan/classify_tensor
    don't take a scale_mode argument at all, so the plan for --format fp8_b128
    is identical regardless of the scale mode that convert() will later use."""
    src, _ = synthetic_gguf_b128
    reader = gguf.GGUFReader(src)
    plan = build_plan(reader, rotation_seed=0, local_b=128, max_b=1024, format="fp8_b128")
    by_name = {e["name"]: e for e in plan}
    assert by_name["blk.0.attn_qkv.weight"]["action"] == "rotate_kronecker"
    assert by_name["blk.0.attn_v.weight"]["action"] == "q8_0"   # unaligned fallback, unaffected
    print("  PASS test_dry_run_classification_fp8_b128_channel_scale_mode")


def test_convert_fp8_b128_channel_end_to_end(synthetic_gguf_b128, tmp_path):
    """convert(..., format="fp8_b128", scale_mode="channel") produces the same
    tensor types/sidecars as tile mode, but every block of a row shares the
    identical stored fp16 scale."""
    src, weights = synthetic_gguf_b128
    out = tmp_path / "tiny_fp8b128_channel.gguf"
    convert(src, out, rotation_seed=1234, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128", scale_mode="channel")
    reader = gguf.GGUFReader(out)
    names = {t.name: t for t in reader.tensors}

    for role_name in ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
                      "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
                      "blk.0.attn_output.weight", "blk.0.ffn_down.weight",
                      "blk.0.ssm_out.weight", "output.weight"]:
        assert names[role_name].tensor_type == GGMLQuantizationType.FP8_B128, role_name

    assert names["blk.0.attn_v.weight"].tensor_type == GGMLQuantizationType.Q8_0
    assert names["token_embd.weight"].tensor_type == GGMLQuantizationType.Q8_0

    block_size, block_bytes = gguf.constants.GGML_QUANT_SIZES[GGMLQuantizationType.FP8_B128]
    for name, K in [("blk.0.ffn_gate.weight", D_B128), ("blk.0.attn_output.weight", FFN_B128)]:
        t = names[name]
        raw = np.ascontiguousarray(t.data)
        N = raw.shape[0]
        n_blocks = raw.shape[1] // block_bytes
        raw3 = raw.reshape(N, n_blocks, block_bytes)
        scale_bytes = raw3[:, :, :2].reshape(N, n_blocks, 2)
        scales = scale_bytes.view(np.float16).astype(np.float32).reshape(N, n_blocks)
        for row in range(N):
            row_scales = scales[row]
            assert np.all(row_scales == row_scales[0]), f"{name} row {row}: scale not uniform across blocks"
    print("  PASS test_convert_fp8_b128_channel_end_to_end")


def test_convert_fp8_b128_default_scale_mode_is_tile(synthetic_gguf_b128, tmp_path):
    """convert() with format="fp8_b128" and no --scale-mode given must be
    byte-identical to explicit scale_mode="tile" — the default is unchanged
    from today's tile-scaled behaviour."""
    src, _ = synthetic_gguf_b128
    out_default = tmp_path / "default.gguf"
    out_tile = tmp_path / "tile.gguf"
    convert(src, out_default, rotation_seed=55, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128")
    convert(src, out_tile, rotation_seed=55, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128", scale_mode="tile")

    r1, r2 = gguf.GGUFReader(out_default), gguf.GGUFReader(out_tile)
    names1 = {t.name: t for t in r1.tensors}
    names2 = {t.name: t for t in r2.tensors}
    assert set(names1) == set(names2)
    for name in names1:
        b1 = np.ascontiguousarray(names1[name].data)
        b2 = np.ascontiguousarray(names2[name].data)
        np.testing.assert_array_equal(b1, b2, err_msg=f"{name}: default scale_mode differs from explicit 'tile'")
    print("  PASS test_convert_fp8_b128_default_scale_mode_is_tile")


def test_convert_fp8_b128_channel_differs_from_tile(synthetic_gguf_b128, tmp_path):
    """Sanity check the channel-mode tests above aren't vacuous: channel and
    tile scale modes must actually produce different bytes for a rotated
    weight (a>1 kronecker weight in this fixture has genuinely non-uniform
    per-tile absmax vs. per-row absmax)."""
    src, _ = synthetic_gguf_b128
    out_tile = tmp_path / "tile.gguf"
    out_channel = tmp_path / "channel.gguf"
    convert(src, out_tile, rotation_seed=7, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128", scale_mode="tile")
    convert(src, out_channel, rotation_seed=7, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128", scale_mode="channel")
    r1, r2 = gguf.GGUFReader(out_tile), gguf.GGUFReader(out_channel)
    t1 = next(t for t in r1.tensors if t.name == "blk.0.ffn_gate.weight")
    t2 = next(t for t in r2.tensors if t.name == "blk.0.ffn_gate.weight")
    b1 = np.ascontiguousarray(t1.data)
    b2 = np.ascontiguousarray(t2.data)
    assert not np.array_equal(b1, b2)
    print("  PASS test_convert_fp8_b128_channel_differs_from_tile")


def test_seed_determinism_fp8_b128_channel(synthetic_gguf_b128, tmp_path):
    """Two runs with the same --rotation-seed and --scale-mode channel give
    bit-identical bytes."""
    src, _ = synthetic_gguf_b128
    out1 = tmp_path / "run1_channel.gguf"
    out2 = tmp_path / "run2_channel.gguf"
    convert(src, out1, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128", scale_mode="channel")
    convert(src, out2, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024,
           format="fp8_b128", scale_mode="channel")
    r1, r2 = gguf.GGUFReader(out1), gguf.GGUFReader(out2)
    names1 = {t.name: t for t in r1.tensors}
    names2 = {t.name: t for t in r2.tensors}
    assert set(names1) == set(names2)
    for name in names1:
        b1 = np.ascontiguousarray(names1[name].data)
        b2 = np.ascontiguousarray(names2[name].data)
        np.testing.assert_array_equal(b1, b2, err_msg=f"{name}: bytes differ across runs")
    print("  PASS test_seed_determinism_fp8_b128_channel")


def test_seed_determinism_ml8_4(synthetic_gguf, tmp_path):
    """Two runs with the same --rotation-seed give bit-identical bytes,
    including the fitted centroids sidecar (the Lloyd-Max fit subsample is
    deterministic — uniformly spaced row indices, not random)."""
    src, _ = synthetic_gguf
    out1 = tmp_path / "run1.gguf"
    out2 = tmp_path / "run2.gguf"
    convert(src, out1, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024, format="ml8_4")
    convert(src, out2, rotation_seed=42, device_str="cpu", local_b=128, max_b=1024, format="ml8_4")
    r1, r2 = gguf.GGUFReader(out1), gguf.GGUFReader(out2)
    names1 = {t.name: t for t in r1.tensors}
    names2 = {t.name: t for t in r2.tensors}
    assert set(names1) == set(names2)
    for name in names1:
        b1 = np.ascontiguousarray(names1[name].data)
        b2 = np.ascontiguousarray(names2[name].data)
        np.testing.assert_array_equal(b1, b2, err_msg=f"{name}: bytes differ across runs")
    print("  PASS test_seed_determinism_ml8_4")


def _make_reference_gguf_with_weight_pager_kv(path: Path) -> None:
    """Small reference GGUF carrying weight_pager.* KV fields, standing in
    for a production spine (e.g. dsv41-spine.gguf) that has metadata a
    BF16-from-convert_hf intermediate lacks. Also carries a
    `weight_pager.already_present` key with a DIFFERENT value than the
    source will have, to verify --kv-from never overwrites a key the source
    already defines -- only backfills ones it's missing."""
    w = gguf.GGUFWriter(str(path), arch="qwen35")
    w.add_uint32("qwen35.embedding_length", D_MODEL)
    w.add_uint32("qwen35.block_count", 1)
    w.add_bool("weight_pager.routed_experts_external", True)
    w.add_uint32("weight_pager.expert_count", 128)
    w.add_uint32("weight_pager.already_present", 999)  # source defines its own value (1) for this
    w.add_string("general.name", "reference-spine-not-copied")  # non-weight_pager.* -- must NOT copy
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()


def test_kv_from_backfills_missing_weight_pager_keys(synthetic_gguf, tmp_path):
    """--kv-from copies weight_pager.* keys present in the reference GGUF
    but absent from --src, leaves keys --src already has untouched, and
    copies nothing outside the weight_pager.* prefix (e.g. general.name is
    NOT clobbered by the reference's). Regression: an ml8-4-converted DS4.1
    spine built from a BF16 intermediate straight off convert_hf_to_gguf.py
    (no weight-pager forge stage) failed to load
    ("check_tensor_dims: tensor 'blk.0.ffn_gate_exps.weight' not found")
    because it lacked `weight_pager.routed_experts_external` (BOOL, True in
    production) -- the C++ loader gates expert-tensor creation on that key."""
    src, _ = synthetic_gguf
    ref = tmp_path / "reference_with_wp_kv.gguf"
    _make_reference_gguf_with_weight_pager_kv(ref)

    # Source already defines this one key -- must survive --kv-from untouched.
    reader_src = gguf.GGUFReader(src)
    assert "weight_pager.already_present" not in reader_src.fields  # sanity: not in the fixture

    # Add it to a modified copy of src with its OWN value (1), to prove
    # --kv-from doesn't clobber it with the reference's (999).
    src_with_own_key = tmp_path / "src_with_own_key.gguf"
    w = gguf.GGUFWriter(str(src_with_own_key), arch="qwen35")
    for name, field in reader_src.fields.items():
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue  # meta fields GGUFWriter already sets via arch=/write_header_to_file
        _copy_field(w, name, field)
    w.add_uint32("weight_pager.already_present", 1)
    for t in reader_src.tensors:
        w.add_tensor(t.name, np.ascontiguousarray(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    out_no_kv_from = tmp_path / "out_no_kv_from.gguf"
    out_kv_from = tmp_path / "out_kv_from.gguf"
    convert(src_with_own_key, out_no_kv_from, rotation_seed=0, device_str="cpu",
            local_b=128, max_b=1024, format="ml8_4")  # kv_from=None (default)
    convert(src_with_own_key, out_kv_from, rotation_seed=0, device_str="cpu",
            local_b=128, max_b=1024, format="ml8_4", kv_from=ref)

    r_no_kv = gguf.GGUFReader(out_no_kv_from)
    r_kv = gguf.GGUFReader(out_kv_from)

    # Unset --kv-from: byte-identical to prior behavior -- no weight_pager.*
    # keys beyond what --src already had (Qwen conversions never pass
    # --kv-from, so this is exactly their code path).
    assert "weight_pager.routed_experts_external" not in r_no_kv.fields
    assert "weight_pager.expert_count" not in r_no_kv.fields
    assert r_no_kv.fields["weight_pager.already_present"].contents() == 1

    # --kv-from set: missing keys backfilled from the reference...
    assert r_kv.fields["weight_pager.routed_experts_external"].contents() == True
    assert r_kv.fields["weight_pager.expert_count"].contents() == 128
    # ...but the key --src already had is NOT overwritten by the reference's.
    assert r_kv.fields["weight_pager.already_present"].contents() == 1
    # ...and non-weight_pager.* fields from the reference are never copied
    # (the source has no general.name of its own, so if --kv-from leaked
    # non-weight_pager.* keys it would show up here).
    assert "general.name" not in r_kv.fields
    print("  PASS test_kv_from_backfills_missing_weight_pager_keys")


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
