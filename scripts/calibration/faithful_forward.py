# scripts/calibration/faithful_forward.py
"""Deployment-faithful (W4A8) calibration forward: rotation precompute + the
activation-e4m3 pre-hook. See docs/superpowers/specs/2026-05-31-w4a8-faithful-calibration-design.md.
"""
import torch
from kronecker_rotation import (KroneckerRotation, random_orthogonal,
                                factor_for_dim)

def build_rotations(dims: dict, seeds: dict, max_b: int = 1024) -> dict:
    """dims/seeds keyed by (layer_key, kind) -> rotation. Built from dims+seeds
    only (never from H values), so it can run before Hessian collection."""
    rots = {}
    for key, K in dims.items():
        a, b = factor_for_dim(K, max_b=max_b)
        rots[key] = KroneckerRotation(h_a=random_orthogonal(a, seed=int(seeds[key])), b_dim=b)
    return rots
