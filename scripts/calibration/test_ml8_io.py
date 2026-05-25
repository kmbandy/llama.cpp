#!/usr/bin/env python3
"""Tests for ml8_io rotation-field handling (backward-compat + happy path)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ml8_io import (  # noqa: E402
    get_rotation, get_awq, reconstruct_weight, reconstruct_inference_weight,
)
from kronecker_rotation import KroneckerRotation, random_orthogonal  # noqa: E402
from awq import apply_awq_to_weight, absorb_awq_in_reconstruction  # noqa: E402


def _synthetic_quant_blob(rows: int, in_features: int, group_size: int = 4,
                          n_centroids: int = 16, seed: int = 0) -> dict:
    """Build a deterministic blob carrying valid indices/centroids/scales."""
    n_groups = in_features // group_size
    torch.manual_seed(seed)
    centroids = torch.linspace(-1, 1, n_centroids).unsqueeze(0).expand(n_groups, -1).clone()
    scales = torch.rand(rows, n_groups) + 0.1
    indices = torch.randint(0, n_centroids, (rows, in_features), dtype=torch.int8)
    return {
        "name": "test",
        "shape": [rows, in_features],
        "group_size": group_size,
        "n_centroids": n_centroids,
        "indices": indices,
        "centroids_per_group": centroids,
        "scale_per_group": scales,
    }


def test_get_rotation_returns_none_for_legacy_blob():
    """Saturday's blobs have no 'rotation' key — must return None, not crash."""
    legacy_blob = {
        "name": "model.layers.0.mlp.gate_proj",
        "shape": [2560, 2560],
        "group_size": 128,
        "n_centroids": 16,
    }
    assert get_rotation(legacy_blob) is None
    print(f"  PASS test_get_rotation_returns_none_for_legacy_blob")


def test_get_rotation_returns_none_for_explicit_identity():
    """An explicit 'identity' rotation is the same as no rotation."""
    blob = {"rotation": {"kind": "identity"}}
    assert get_rotation(blob) is None
    print(f"  PASS test_get_rotation_returns_none_for_explicit_identity")


def test_get_rotation_round_trips_kronecker():
    """A blob with a serialized KroneckerRotation reconstructs to an equivalent rotation."""
    rot = KroneckerRotation(h_a=random_orthogonal(5, seed=33), b_dim=512)
    blob = {"rotation": rot.to_dict()}
    reloaded = get_rotation(blob)
    assert reloaded is not None
    x = torch.randn(2560, dtype=torch.float32)
    diff = (rot.forward(x) - reloaded.forward(x)).abs().max().item()
    assert diff == 0.0, f"rotation forward differs after round-trip: max abs diff {diff:.3e}"
    print(f"  PASS test_get_rotation_round_trips_kronecker")


def test_reconstruct_inference_weight_no_rotation():
    """When blob has no rotation, reconstruct_inference_weight == reconstruct_weight bit-exact."""
    blob = _synthetic_quant_blob(rows=8, in_features=16)
    plain = reconstruct_weight(blob)
    inference = reconstruct_inference_weight(blob)
    diff = (plain - inference).abs().max().item()
    assert diff == 0.0, f"no-rotation path should be bit-exact, max abs diff {diff:.3e}"
    print(f"  PASS test_reconstruct_inference_weight_no_rotation")


def test_reconstruct_inference_weight_absorbs_rotation():
    """When blob has rotation, reconstruct_inference_weight == reconstruct_weight @ Q.T."""
    rows = 8
    a, b = 2, 4
    in_features = a * b
    blob = _synthetic_quant_blob(rows=rows, in_features=in_features, group_size=4)
    rot = KroneckerRotation(h_a=random_orthogonal(a, seed=55), b_dim=b)
    blob["rotation"] = rot.to_dict()

    W_rot = reconstruct_weight(blob)
    expected = rot.inverse(W_rot)   # W_rot @ Q.T
    actual = reconstruct_inference_weight(blob)
    diff = (actual - expected).abs().max().item()
    assert diff == 0.0, f"absorption mismatch: max abs diff {diff:.3e}"
    print(f"  PASS test_reconstruct_inference_weight_absorbs_rotation")


def test_get_awq_returns_none_for_legacy_or_explicit_none():
    """Missing 'awq' key, None spec, or {'kind': 'none'} → None (backward-compat)."""
    assert get_awq({}) is None
    assert get_awq({"awq": None}) is None
    assert get_awq({"awq": {"kind": "none"}}) is None
    print(f"  PASS test_get_awq_returns_none_for_legacy_or_explicit_none")


def test_get_awq_returns_scale_tensor_when_present():
    """When awq spec has kind='mean' and s tensor, returns the s tensor."""
    s = torch.rand(64) + 0.1
    blob = {"awq": {"kind": "mean", "alpha": 0.5, "s": s}}
    got = get_awq(blob)
    assert got is not None
    assert torch.equal(got, s), "get_awq should return the stored s tensor unchanged"
    print(f"  PASS test_get_awq_returns_scale_tensor_when_present")


def test_reconstruct_inference_weight_absorbs_awq_only():
    """Blob with awq but no rotation: reconstruct = dequant * s per col (= absorb_awq)."""
    rows, in_features = 8, 16
    blob = _synthetic_quant_blob(rows=rows, in_features=in_features, group_size=4)
    s = torch.rand(in_features) + 0.1
    blob["awq"] = {"kind": "mean", "alpha": 0.5, "s": s}

    W_dequant = reconstruct_weight(blob)
    expected = absorb_awq_in_reconstruction(W_dequant, s)
    actual = reconstruct_inference_weight(blob)
    diff = (actual - expected).abs().max().item()
    assert diff == 0.0, f"AWQ-only absorption mismatch: {diff:.3e}"
    print(f"  PASS test_reconstruct_inference_weight_absorbs_awq_only")


def test_reconstruct_inference_weight_absorbs_rotation_then_awq():
    """Both rotation and awq: dequant → rotation.inverse → absorb_awq."""
    rows = 8
    a, b = 2, 4
    in_features = a * b
    blob = _synthetic_quant_blob(rows=rows, in_features=in_features, group_size=4)
    rot = KroneckerRotation(h_a=random_orthogonal(a, seed=77), b_dim=b)
    s = torch.rand(in_features) + 0.1
    blob["rotation"] = rot.to_dict()
    blob["awq"] = {"kind": "mean", "alpha": 0.5, "s": s}

    W_dequant = reconstruct_weight(blob)
    after_rot = rot.inverse(W_dequant)
    expected = absorb_awq_in_reconstruction(after_rot, s)
    actual = reconstruct_inference_weight(blob)
    diff = (actual - expected).abs().max().item()
    assert diff == 0.0, f"rotation+AWQ absorption mismatch: {diff:.3e}"
    print(f"  PASS test_reconstruct_inference_weight_absorbs_rotation_then_awq")


if __name__ == "__main__":
    test_get_rotation_returns_none_for_legacy_blob()
    test_get_rotation_returns_none_for_explicit_identity()
    test_get_rotation_round_trips_kronecker()
    test_reconstruct_inference_weight_no_rotation()
    test_reconstruct_inference_weight_absorbs_rotation()
    test_get_awq_returns_none_for_legacy_or_explicit_none()
    test_get_awq_returns_scale_tensor_when_present()
    test_reconstruct_inference_weight_absorbs_awq_only()
    test_reconstruct_inference_weight_absorbs_rotation_then_awq()
    print("\nALL TESTS PASSED")
