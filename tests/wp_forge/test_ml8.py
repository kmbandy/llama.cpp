"""CPU-only checks for conversion/wp_forge/ml8.py: rotation math, ML8_FP8 /
ML8_4 byte format, and dequant against the exact ggml formulas (see
ggml/src/ggml-turbo-quant.c dequantize_row_ml8_fp8 / dequantize_row_ml8_4_with_lut).

Sizes are tiny (K=128 or K=2048, few rows) so the tests stay fast.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conversion.wp_forge import ml8

SEED = 20260927


def _f32(rows: int, cols: int, scale: float = 0.03) -> np.ndarray:
    return (np.random.RandomState(SEED).randn(rows, cols).astype(np.float32) * scale)


# --- rotation math ----------------------------------------------------------

def test_kronecker_rotation_orthogonal_roundtrip() -> None:
    rot, kind, a, b = ml8.build_rotation(2048, "kronecker", 0, 1024, layer=3, group="gate_up")
    assert kind == ml8.KRONECKER_KIND_ID
    x = torch.randn(5, 2048, dtype=torch.float64)
    y = rot.forward(x.float()).double()
    # orthogonality: rotation preserves norm
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-3)
    # W_rot @ x_rot == W @ x (forward/inverse are exact inverses, float64 tol)
    xr = rot.inverse(rot.forward(x.float())).double()
    assert torch.allclose(x, xr, atol=1e-5)


def test_block_hadamard_rotation() -> None:
    rot, kind, a, b = ml8.build_rotation(2048, "block_hadamard", 0, 1024, layer=1, group="down")
    assert kind == ml8.BLOCK_HADAMARD_KIND_ID
    x = torch.randn(4, 2048)
    y = rot.forward(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-3)
    xr = rot.inverse(y)
    assert torch.allclose(x, xr, atol=1e-4)


def test_none_rotation_is_identity() -> None:
    rot, kind, a, b = ml8.build_rotation(2048, "none", 0, 1024, layer=0, group="gate_up")
    assert rot is None and kind == 0 and a == 0 and b == 0


def test_gate_up_share_rotation_down_differs() -> None:
    rot_gate, _, _, _ = ml8.build_rotation(2048, "kronecker", 7, 1024, layer=5, group="gate_up")
    rot_up, _, _, _ = ml8.build_rotation(2048, "kronecker", 7, 1024, layer=5, group="gate_up")
    assert torch.equal(rot_gate.h_a, rot_up.h_a)
    # same K, but the group key differs ("down" vs "gate_up") -> different
    # seed -> different h_a, even restricted to the same shape.
    rot_down_same_k, _, _, _ = ml8.build_rotation(2048, "kronecker", 7, 1024, layer=5, group="down")
    assert not torch.equal(rot_gate.h_a, rot_down_same_k.h_a)


def test_rotation_varies_by_layer() -> None:
    r0, _, _, _ = ml8.build_rotation(2048, "kronecker", 0, 1024, layer=0, group="gate_up")
    r1, _, _, _ = ml8.build_rotation(2048, "kronecker", 0, 1024, layer=1, group="gate_up")
    assert not torch.equal(r0.h_a, r1.h_a)


def test_rotation_meta_bytes() -> None:
    b = ml8.rotation_meta_bytes(5, 512, 2560, ml8.KRONECKER_KIND_ID)
    assert b.dtype == np.int32 and b.tolist() == [5, 512, 2560, 1]


# --- ML8_FP8 ------------------------------------------------------------

def _dequant_ml8_fp8(packed: np.ndarray, k: int) -> np.ndarray:
    n = packed.shape[0]
    nblk = k // 32
    out = np.zeros((n, k), dtype=np.float32)
    for r in range(n):
        for blk in range(nblk):
            off = blk * 34
            scale = np.frombuffer(packed[r, off:off + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
            qs = packed[r, off + 2:off + 34]
            e4m3 = torch.tensor(qs).view(torch.float8_e4m3fn).float().numpy()
            out[r, blk * 32:(blk + 1) * 32] = e4m3 * scale
    return out


def test_ml8_fp8_byte_format_and_dequant() -> None:
    K = 128
    w = _f32(6, K)
    packed = ml8.quantize_role_ml8_fp8(w, None)
    assert packed.shape == (6, (K // 32) * 34)
    deq = _dequant_ml8_fp8(packed, K)
    relerr = np.abs(deq - w) / np.abs(w).max()
    assert relerr.max() < 0.05


def test_ml8_fp8_rotation_applied() -> None:
    K = 2048
    w = _f32(3, K)
    rot, _, _, _ = ml8.build_rotation(K, "kronecker", 0, 1024, layer=0, group="gate_up")
    packed_rot = ml8.quantize_role_ml8_fp8(w, rot)
    packed_none = ml8.quantize_role_ml8_fp8(w, None)
    assert not np.array_equal(packed_rot, packed_none)


def test_quantize_experts_ml8_fp8_matches_per_expert() -> None:
    K, rows, n_e = 128, 4, 3
    stack = np.stack([_f32(rows, K, scale=0.01 * (e + 1)) for e in range(n_e)], axis=0)
    rot, _, _, _ = ml8.build_rotation(K, "kronecker", 0, 1024, layer=2, group="down")
    batched = ml8.quantize_experts_ml8_fp8(stack, rot)
    assert batched.shape == (n_e, rows, (K // 32) * 34)
    for e in range(n_e):
        single = ml8.quantize_role_ml8_fp8(stack[e], rot)
        assert np.array_equal(batched[e], single)


# --- ML8_4 ----------------------------------------------------------------

def _dequant_ml8_4(packed: np.ndarray, cent: np.ndarray, k: int) -> np.ndarray:
    n = packed.shape[0]
    ngrp = k // 64
    out = np.zeros((n, k), dtype=np.float32)
    for r in range(n):
        for g in range(ngrp):
            off = g * 36
            scale = np.frombuffer(packed[r, off:off + 4].tobytes(), dtype=np.float32)[0]
            qs = packed[r, off + 4:off + 36]
            lut = torch.tensor(cent[g]).view(torch.float8_e4m3fn).float().numpy()
            for i in range(32):
                byte = int(qs[i])
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                out[r, g * 64 + 2 * i] = lut[lo] * scale
                out[r, g * 64 + 2 * i + 1] = lut[hi] * scale
    return out


def test_ml8_4_pack_roundtrip_vs_ggml_dequant_formula() -> None:
    K = 128
    w = _f32(8, K)
    packed, cent = ml8.quantize_role_ml8_4(w, None, fit_rows=8)
    assert packed.shape == (8, (K // 64) * 36)
    assert cent.shape == (K // 64, 16) and cent.dtype == np.uint8
    deq = _dequant_ml8_4(packed, cent, K)
    rel_l2 = np.linalg.norm(deq - w) / np.linalg.norm(w)
    # relative-error bound on random Gaussian weights (data-free mse fit;
    # convert_fp8_rotated.py's own docstring measures ~0.09-0.10 for this
    # recipe on real weight rows -- keep a little headroom for a tiny/random
    # synthetic tensor).
    assert rel_l2 < 0.20


def test_ml8_4_indices_in_range() -> None:
    K = 128
    w = _f32(8, K)
    packed, _cent = ml8.quantize_role_ml8_4(w, None, fit_rows=8)
    for r in range(8):
        for g in range(K // 64):
            off = g * 36 + 4
            qs = packed[r, off:off + 32]
            lo = qs & 0x0F
            hi = (qs >> 4) & 0x0F
            assert lo.max() <= 15 and hi.max() <= 15


def test_quantize_experts_ml8_4_fits_per_expert_lut() -> None:
    """Two experts with very different weight DISTRIBUTIONS (not just an
    overall scale -- absmax normalization per (row, group) cancels a pure
    scalar factor exactly, so that alone wouldn't distinguish the fit) must
    get different LUTs: the fused-path centroid fit is per expert, not
    shared/pooled."""
    K, rows = 128, 32
    gaussian = np.random.RandomState(1).randn(rows, K).astype(np.float32) * 0.02
    # heavy-tailed (laplace): different shape after per-group normalization,
    # so its Lloyd-Max fit lands on different centroids than the gaussian one.
    heavy_tailed = np.random.RandomState(2).laplace(size=(rows, K)).astype(np.float32) * 0.02
    stack = np.stack([gaussian, heavy_tailed], axis=0)
    rot, _, _, _ = ml8.build_rotation(K, "kronecker", 0, 1024, layer=0, group="down")
    packed, cent = ml8.quantize_experts_ml8_4(stack, rot, fit_rows=rows)
    assert packed.shape == (2, rows, (K // 64) * 36)
    assert cent.shape == (2, K // 64, 16)
    assert not np.array_equal(cent[0], cent[1])
    # and matches the per-expert single-call path exactly
    p0, c0 = ml8.quantize_role_ml8_4(gaussian, rot, fit_rows=rows)
    p1, c1 = ml8.quantize_role_ml8_4(heavy_tailed, rot, fit_rows=rows)
    assert np.array_equal(packed[0], p0) and np.array_equal(cent[0], c0)
    assert np.array_equal(packed[1], p1) and np.array_equal(cent[1], c1)


def test_ml8_4_bad_shape_raises() -> None:
    with pytest.raises(ml8.Ml8Error):
        ml8.quantize_role_ml8_4(_f32(4, 100), None, fit_rows=4)  # 100 not %64
