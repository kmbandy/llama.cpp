#!/usr/bin/env python3
"""Tests for kronecker_rotation.py — KroneckerRotation, sylvester, random_orthogonal."""

import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kronecker_rotation import (  # noqa: E402
    sylvester, random_orthogonal, KroneckerRotation,
    factor_for_dim, rotate_hessian, fwht_raw,
)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, label: str):
    diff = (actual - expected).abs().max().item()
    assert diff <= tol, f"{label}: max abs diff {diff:.3e} > tol {tol:.3e}"


def test_fwht_raw_matches_dense_hadamard():
    """fwht_raw(x)/sqrt(n) == x @ sylvester(n) (the normalized Sylvester right-multiply).

    This is the fast butterfly that replaces the dense H_b matmul in the rotation:
    same math, O(n log n) instead of O(n^2), and it's what the deployed ml8 kernel
    does for the H_b leg (ml8.cu fused rot+quant)."""
    torch.manual_seed(0)
    for n in [2, 4, 16, 256, 512, 1024]:
        H = sylvester(n)                                   # normalized: H_raw / sqrt(n)
        x = torch.randn(7, n)
        expected = x @ H                                   # dense normalized right-mul
        actual = fwht_raw(x) / math.sqrt(n)                # fast butterfly + normalize
        _assert_close(actual, expected, tol=1e-4, label=f"fwht_raw n={n}")
    # works on a batched (..., a, n) tensor along the last dim
    x = torch.randn(4, 5, 512)
    expected = x @ sylvester(512)
    actual = fwht_raw(x) / math.sqrt(512)
    _assert_close(actual, expected, tol=1e-4, label="fwht_raw batched")
    # gradient flows (used inside the autograd-tracked rotation forward)
    xg = torch.randn(3, 256, requires_grad=True)
    fwht_raw(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    print("  PASS test_fwht_raw_matches_dense_hadamard")


def test_sylvester_orthonormal():
    """Sylvester Hadamard at b in {2,4,8,...,1024} satisfies H @ H.T == I."""
    for b in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        H = sylvester(b)
        assert H.shape == (b, b), f"b={b}: wrong shape {H.shape}"
        identity = torch.eye(b, dtype=H.dtype)
        _assert_close(H @ H.T, identity, tol=1e-5, label=f"sylvester({b}) orthonormality")
    print(f"  PASS test_sylvester_orthonormal")


def test_random_orthogonal():
    """random_orthogonal(a, seed) returns (a, a) orthonormal: Q @ Q.T == I."""
    for a in [2, 3, 5, 7, 9]:
        Q = random_orthogonal(a, seed=42)
        assert Q.shape == (a, a), f"a={a}: wrong shape {Q.shape}"
        identity = torch.eye(a, dtype=Q.dtype)
        _assert_close(Q @ Q.T, identity, tol=1e-5, label=f"random_orthogonal({a}) orthonormality")
    # Same seed → same matrix
    Q1 = random_orthogonal(5, seed=42)
    Q2 = random_orthogonal(5, seed=42)
    _assert_close(Q1, Q2, tol=0.0, label="random_orthogonal seed determinism")
    # Different seed → different matrix
    Q3 = random_orthogonal(5, seed=43)
    diff = (Q1 - Q3).abs().max().item()
    assert diff > 1e-3, f"different seeds produced same matrix (diff {diff:.3e})"
    print(f"  PASS test_random_orthogonal")


def test_kronecker_round_trip():
    """KroneckerRotation: inverse(forward(x)) == x to fp32 tolerance for representative dims."""
    # Cover the actual Qwen3.5-4B layer dims plus a small case.
    cases = [
        (2, 4),     # small smoke
        (5, 512),   # gate_proj / up_proj / attn_qkv in_features = 2560
        (9, 1024),  # down_proj in_features = 9216
        (1, 4096),  # attn_o in_features = 4096 (pure Sylvester, a=1)
    ]
    for a, b in cases:
        d = a * b
        rot = KroneckerRotation(h_a=random_orthogonal(a, seed=7), b_dim=b)
        # Single vector
        x = torch.randn(d, dtype=torch.float32)
        x_back = rot.inverse(rot.forward(x))
        _assert_close(x_back, x, tol=1e-4, label=f"K({a},{b}) round-trip vector")
        # Batched (mimics how nn.Linear gets called with batch * seq inputs)
        X = torch.randn(3, 7, d, dtype=torch.float32)
        X_back = rot.inverse(rot.forward(X))
        _assert_close(X_back, X, tol=1e-4, label=f"K({a},{b}) round-trip batched")
    print(f"  PASS test_kronecker_round_trip")


def test_kronecker_matches_dense_right_mul():
    """forward(x) == x @ Q where Q = kron(H_a, H_b) explicit dense matrix.

    PyTorch nn.Linear convention is `y = x @ W.T` (x is row-vector); to insert
    rotation we want `x_rot = x @ Q` and `W_rot = W @ Q` so that
    `(x @ Q) @ (W @ Q).T == x @ W.T`. This test pins down that forward
    implements the row-vector right-multiplication, not the column-vector left-mul.
    """
    torch.manual_seed(0)
    for a, b in [(2, 4), (3, 8), (5, 16)]:
        d = a * b
        h_a = random_orthogonal(a, seed=11)
        rot = KroneckerRotation(h_a=h_a, b_dim=b)
        Q_dense = torch.kron(h_a.contiguous(), sylvester(b).contiguous())  # (d, d) explicit Kronecker product
        x = torch.randn(d, dtype=torch.float32)
        expected = x @ Q_dense                    # row-vector right-mul
        actual = rot.forward(x)
        _assert_close(actual, expected, tol=1e-4, label=f"K({a},{b}) forward matches x @ Q")
        # And inverse must match x @ Q.T
        expected_inv = x @ Q_dense.T
        actual_inv = rot.inverse(x)
        _assert_close(actual_inv, expected_inv, tol=1e-4, label=f"K({a},{b}) inverse matches x @ Q.T")
    print(f"  PASS test_kronecker_matches_dense_right_mul")


def test_kronecker_serialize_round_trip():
    """to_dict produces blob with required schema; from_dict reconstructs an equivalent rotation.

    Equivalent means: rot.forward(x) == rot_reloaded.forward(x) bit-exact (no math drift on reload).
    Required schema keys: kind, h_a, a_dim, b_dim, in_features.
    """
    rot = KroneckerRotation(h_a=random_orthogonal(5, seed=99), b_dim=512)
    blob = rot.to_dict()
    required = {"kind", "h_a", "a_dim", "b_dim", "in_features"}
    missing = required - set(blob.keys())
    assert not missing, f"to_dict missing keys: {missing}"
    assert blob["kind"] == "kronecker_orth_sylvester", f"unexpected kind {blob['kind']!r}"
    assert blob["a_dim"] == 5 and blob["b_dim"] == 512 and blob["in_features"] == 2560

    rot_reloaded = KroneckerRotation.from_dict(blob)
    x = torch.randn(2560, dtype=torch.float32)
    _assert_close(rot.forward(x), rot_reloaded.forward(x), tol=0.0,
                  label="serialized rotation forward equality")
    _assert_close(rot.inverse(x), rot_reloaded.inverse(x), tol=0.0,
                  label="serialized rotation inverse equality")
    print(f"  PASS test_kronecker_serialize_round_trip")


def test_factor_for_dim_qwen35_layers():
    """factor_for_dim picks the right (a, b) for Qwen3.5-4B MLP layer dims (max_b=1024)."""
    cases = {
        2560: (5, 512),    # gate_proj / up_proj / attn_qkv in_features
        9216: (9, 1024),   # down_proj in_features
        4096: (4, 1024),   # attn_o in_features — capped at max_b=1024
        256:  (1, 256),    # head_dim
        2048: (2, 1024),   # capped
        1024: (1, 1024),   # power of 2
        1:    (1, 1),      # degenerate
        7:    (7, 1),      # pure-odd, no Hadamard factor available
    }
    for d, (a_expected, b_expected) in cases.items():
        a, b = factor_for_dim(d, max_b=1024)
        assert (a, b) == (a_expected, b_expected), \
            f"factor_for_dim({d}, max_b=1024) = ({a}, {b}), expected ({a_expected}, {b_expected})"
        assert a * b == d, f"factor_for_dim({d}) factors don't multiply: {a}*{b} != {d}"
    print(f"  PASS test_factor_for_dim_qwen35_layers")


def test_rotate_hessian_matches_dense():
    """rotate_hessian(H, rot) == Q.T @ H @ Q where Q is the dense Kronecker product."""
    torch.manual_seed(1)
    for a, b in [(2, 4), (3, 8), (5, 16)]:
        d = a * b
        h_a = random_orthogonal(a, seed=17)
        rot = KroneckerRotation(h_a=h_a, b_dim=b)
        Q_dense = torch.kron(h_a.contiguous(), sylvester(b).contiguous())
        H = torch.randn(d, d, dtype=torch.float32)
        H = (H + H.T) / 2  # symmetrize (real Hessians are symmetric)
        expected = Q_dense.T @ H @ Q_dense
        actual = rotate_hessian(H, rot)
        _assert_close(actual, expected, tol=1e-4, label=f"K({a},{b}) rotate_hessian")
    print(f"  PASS test_rotate_hessian_matches_dense")


if __name__ == "__main__":
    test_sylvester_orthonormal()
    test_random_orthogonal()
    test_kronecker_round_trip()
    test_kronecker_matches_dense_right_mul()
    test_kronecker_serialize_round_trip()
    test_factor_for_dim_qwen35_layers()
    test_rotate_hessian_matches_dense()
    print("\nALL TESTS PASSED")
