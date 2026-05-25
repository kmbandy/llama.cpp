#!/usr/bin/env python3
"""Tests for AWQ rescaling — Activation-aware Weight Quantization for ml8."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from awq import compute_awq_scale, apply_awq_to_weight, absorb_awq_in_reconstruction  # noqa: E402


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, label: str):
    diff = (actual - expected).abs().max().item()
    assert diff <= tol, f"{label}: max abs diff {diff:.3e} > tol {tol:.3e}"


def test_compute_awq_scale_higher_for_salient_channels():
    """s_i = mean(|x_i|)^alpha grows monotonically with channel magnitude."""
    # Build activations where channel 5 is the most salient
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 0.5  # batch 64, 128 channels
    x[:, 5] *= 10.0  # make channel 5 huge
    s = compute_awq_scale(x, alpha=0.5)
    assert s.shape == (128,), f"wrong shape {s.shape}"
    assert s[5] > s.mean() * 2, f"salient channel didn't get high scale: s[5]={s[5]:.3f}, mean={s.mean():.3f}"
    print(f"  PASS test_compute_awq_scale_higher_for_salient_channels")


def test_compute_awq_scale_alpha_zero_is_uniform():
    """alpha=0 → all scales == 1 (no AWQ effect)."""
    torch.manual_seed(1)
    x = torch.randn(32, 64) * 0.5
    x[:, 10] *= 100  # huge outlier channel
    s = compute_awq_scale(x, alpha=0.0)
    # alpha=0 makes |x|^0 = 1 everywhere, so mean = 1, so s = 1
    _assert_close(s, torch.ones_like(s), tol=1e-6, label="alpha=0 → uniform scale")
    print(f"  PASS test_compute_awq_scale_alpha_zero_is_uniform")


def test_apply_awq_to_weight_divides_columns_by_s():
    """W_awq[:, c] == W[:, c] / s[c]."""
    torch.manual_seed(2)
    W = torch.randn(16, 32)
    s = torch.rand(32) + 0.1  # avoid div by zero
    W_awq = apply_awq_to_weight(W, s)
    expected = W / s.unsqueeze(0)  # broadcast across rows
    _assert_close(W_awq, expected, tol=1e-6, label="AWQ column rescale")
    print(f"  PASS test_apply_awq_to_weight_divides_columns_by_s")


def test_absorb_awq_in_reconstruction_inverse():
    """absorb_awq(W_awq, s) * s_inv-applied-to-columns recovers original W."""
    torch.manual_seed(3)
    W = torch.randn(16, 32)
    s = torch.rand(32) + 0.1
    W_awq = apply_awq_to_weight(W, s)            # W / s per column
    W_recovered = absorb_awq_in_reconstruction(W_awq, s)  # W_awq * s per column → back to W
    _assert_close(W_recovered, W, tol=1e-5, label="AWQ apply+absorb round-trip")
    print(f"  PASS test_absorb_awq_in_reconstruction_inverse")


def test_awq_forward_pass_invariance():
    """Mathematically: y = x @ W.T == (x * s) @ (W / s).T for any positive s.

    This is the core identity AWQ relies on for absorption — verified at fp32."""
    torch.manual_seed(4)
    x = torch.randn(8, 32)
    W = torch.randn(16, 32)
    s = torch.rand(32) + 0.1
    y_orig = x @ W.T
    x_scaled = x * s.unsqueeze(0)
    W_scaled = apply_awq_to_weight(W, s)  # W / s
    y_awq = x_scaled @ W_scaled.T
    _assert_close(y_awq, y_orig, tol=1e-4, label="AWQ math invariance")
    print(f"  PASS test_awq_forward_pass_invariance")


if __name__ == "__main__":
    test_compute_awq_scale_higher_for_salient_channels()
    test_compute_awq_scale_alpha_zero_is_uniform()
    test_apply_awq_to_weight_divides_columns_by_s()
    test_absorb_awq_in_reconstruction_inverse()
    test_awq_forward_pass_invariance()
    print("\nALL TESTS PASSED")
