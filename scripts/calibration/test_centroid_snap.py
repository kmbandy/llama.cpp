#!/usr/bin/env python3
"""Tests for E4M3 centroid snap — the bridge that makes ml8 actually FP8-WMMA-ready."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from centroid_quantizer import snap_to_e4m3, CentroidQuantizer  # noqa: E402


def test_snap_idempotent():
    """Snapping a snapped tensor is a no-op."""
    torch.manual_seed(0)
    x = torch.randn(100) * 0.5  # centroid-range values
    s1 = snap_to_e4m3(x)
    s2 = snap_to_e4m3(s1)
    diff = (s1 - s2).abs().max().item()
    assert diff == 0.0, f"snap not idempotent: max abs diff {diff:.3e}"
    print(f"  PASS test_snap_idempotent")


def test_snap_on_lattice_values():
    """Known E4M3-representable values snap to themselves."""
    # E4M3 includes 0, ±0.5, ±1.0, ±1.5, ±2.0 exactly
    on_lattice = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0], dtype=torch.float32)
    snapped = snap_to_e4m3(on_lattice)
    diff = (snapped - on_lattice).abs().max().item()
    assert diff == 0.0, f"on-lattice values changed by snap: {diff:.3e}"
    print(f"  PASS test_snap_on_lattice_values")


def test_snap_preserves_dtype():
    """Snap returns same dtype as input."""
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.tensor([0.5, 1.5, 2.0], dtype=dtype)
        s = snap_to_e4m3(x)
        assert s.dtype == dtype, f"dtype changed: {x.dtype} → {s.dtype}"
    print(f"  PASS test_snap_preserves_dtype")


def test_centroid_quantizer_snaps_centroids_when_configured():
    """When configure(snap_centroids='e4m3'), find_params yields centroids on E4M3 lattice."""
    torch.manual_seed(1)
    q = CentroidQuantizer(n_centroids=16, n_iter=10)
    q.configure(bits=4, sym=True, fit_loss="mse", mag_weight_p=2.0, snap_centroids="e4m3")
    # Centroid-range data (weights typically have this scale post-normalize)
    x = torch.randn(64, 128) * 0.5
    q.find_params(x)
    # The "centroids" the quantizer holds (after fit) should be E4M3-representable
    cent = q.centroids if hasattr(q, "centroids") else None
    assert cent is not None, "CentroidQuantizer must expose .centroids after find_params"
    # Re-snap and verify zero diff (idempotency at the quantizer level)
    re_snapped = snap_to_e4m3(cent)
    diff = (cent - re_snapped).abs().max().item()
    assert diff == 0.0, f"centroids not on E4M3 lattice after configured snap: max diff {diff:.3e}"
    print(f"  PASS test_centroid_quantizer_snaps_centroids_when_configured")


def test_centroid_quantizer_default_is_no_snap():
    """configure() without snap_centroids leaves centroids unconstrained (backward compat)."""
    torch.manual_seed(2)
    q = CentroidQuantizer(n_centroids=16, n_iter=10)
    q.configure(bits=4, sym=True, fit_loss="mse", mag_weight_p=2.0)  # no snap_centroids
    x = torch.randn(64, 128) * 0.5
    q.find_params(x)
    cent = q.centroids
    re_snapped = snap_to_e4m3(cent)
    diff = (cent - re_snapped).abs().max().item()
    # Without snap, centroids almost certainly fall off E4M3 lattice
    assert diff > 0.0, f"unsnapped centroids should differ from E4M3-snapped (got {diff:.3e})"
    print(f"  PASS test_centroid_quantizer_default_is_no_snap")


if __name__ == "__main__":
    test_snap_idempotent()
    test_snap_on_lattice_values()
    test_snap_preserves_dtype()
    test_centroid_quantizer_snaps_centroids_when_configured()
    test_centroid_quantizer_default_is_no_snap()
    print("\nALL TESTS PASSED")
