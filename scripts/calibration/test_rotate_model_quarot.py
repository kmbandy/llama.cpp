#!/usr/bin/env python3
"""Tests for rotate_model_quarot.py — Role enum, classify_tensor."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rotate_model_quarot import Role, classify_tensor


def _assert_eq(actual, expected, label):
    assert actual == expected, f"{label}: got {actual!r}, expected {expected!r}"


def test_classify_qwen36_tensors():
    """Known Qwen3.6 MoE tensor names map to expected roles."""
    cases = [
        ("token_embd.weight",                 Role.EMBED),
        ("blk.0.attn_norm.weight",            Role.NORM_PRE_ATTN),
        ("blk.0.attn_q.weight",               Role.ATTN_Q),
        ("blk.0.attn_k.weight",               Role.ATTN_K),
        ("blk.0.attn_v.weight",               Role.ATTN_V),
        ("blk.0.attn_output.weight",          Role.ATTN_O),
        ("blk.0.ffn_norm.weight",             Role.NORM_PRE_FFN),
        ("blk.0.ffn_gate_inp.weight",         Role.FFN_GATE_INP),
        ("blk.0.ffn_gate_exps.weight",        Role.FFN_GATE_EXPS),
        ("blk.0.ffn_up_exps.weight",          Role.FFN_UP_EXPS),
        ("blk.0.ffn_down_exps.weight",        Role.FFN_DOWN_EXPS),
        ("blk.0.ssm_norm.weight",             Role.NORM_PRE_SSM),
        ("blk.0.ssm_in.weight",               Role.MAMBA_IN),
        ("blk.0.ssm_out.weight",              Role.MAMBA_OUT),
        ("output_norm.weight",                Role.NORM_OUT),
        ("output.weight",                     Role.LM_HEAD),
        ("rope_freqs.weight",                 Role.PASSTHROUGH),
    ]
    for name, expected in cases:
        actual = classify_tensor(name, arch="qwen36moe")
        _assert_eq(actual, expected, f"qwen36moe / {name}")
    print(f"  PASS test_classify_qwen36_tensors")


def test_classify_unknown_raises():
    """Unknown tensor name on a known arch raises with the name in the message."""
    try:
        classify_tensor("blk.0.fictional.weight", arch="qwen36moe")
    except ValueError as e:
        msg = str(e)
        assert "blk.0.fictional.weight" in msg, f"name missing from error: {msg}"
        print(f"  PASS test_classify_unknown_raises")
        return
    raise AssertionError("expected ValueError, got none")


from rotate_model_quarot import build_R_resid


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, label: str):
    diff = (actual - expected).abs().max().item()
    assert diff <= tol, f"{label}: max abs diff {diff:.3e} > tol {tol:.3e}"


def test_R_resid_orthogonal_pow2():
    """R @ R.T == I for power-of-2 d_model (pure Sylvester)."""
    R = build_R_resid(d_model=2048, seed=42, device=torch.device("cpu"))
    assert R.shape == (2048, 2048), f"shape {R.shape}"
    assert R.dtype == torch.float32, f"dtype {R.dtype}"
    I = torch.eye(2048, dtype=torch.float32)
    _assert_close(R @ R.T, I, tol=1e-5, label="R @ R.T")
    _assert_close(R.T @ R, I, tol=1e-5, label="R.T @ R")
    print(f"  PASS test_R_resid_orthogonal_pow2")


def test_R_resid_orthogonal_kronecker():
    """R @ R.T == I for non-power-of-2 d_model via Kronecker H_a ⊗ H_b."""
    R = build_R_resid(d_model=2560, seed=42, device=torch.device("cpu"))
    assert R.shape == (2560, 2560), f"shape {R.shape}"
    I = torch.eye(2560, dtype=torch.float32)
    _assert_close(R @ R.T, I, tol=1e-4, label="R @ R.T (kronecker)")
    print(f"  PASS test_R_resid_orthogonal_kronecker")


def test_R_resid_seed_determinism():
    """Same seed → same R; different seed → different R."""
    R1 = build_R_resid(d_model=64, seed=42, device=torch.device("cpu"))
    R2 = build_R_resid(d_model=64, seed=42, device=torch.device("cpu"))
    R3 = build_R_resid(d_model=64, seed=43, device=torch.device("cpu"))
    _assert_close(R1, R2, tol=0.0, label="same-seed determinism")
    diff = (R1 - R3).abs().max().item()
    assert diff > 1e-3, f"different-seed difference too small: {diff:.3e}"
    print(f"  PASS test_R_resid_seed_determinism")


if __name__ == "__main__":
    test_classify_qwen36_tensors()
    test_classify_unknown_raises()
    test_R_resid_orthogonal_pow2()
    test_R_resid_orthogonal_kronecker()
    test_R_resid_seed_determinism()
    print("\nALL TESTS PASSED")
