"""ml8_io — load and reconstruct saved per-layer ml8-4 quantized weights.

Companion to scripts/calibration/calibrate_ml8.py. Each .pt file produced by
the driver contains:

    {
        "name": str,                  # tensor name (e.g. "model.layers.0.mlp.up_proj")
        "shape": [rows, in_features],
        "group_size": int,            # e.g. 128
        "n_centroids": int,           # 16 for ml8-4
        "indices": int8 [rows, in_features],          # values 0..n_centroids-1
        "centroids_per_group": fp32 [n_groups, n_centroids],
        "scale_per_group": fp32 [rows, n_groups],
        "mse": float,
        "w_snr_db": float,
        "y_snr_db": float,
        "rel_err": float,
        # Optional (added MAD-223 Phase B.3 for the calibration sweep). Missing key,
        # None, or {"kind": "identity"} all mean no input rotation is applied — fully
        # backward-compatible with Saturday's blobs. When present and non-identity,
        # reconstruct_model.py wraps the target Linear so input is rotated before matmul.
        "rotation": {
            "kind": "kronecker_orth_sylvester",
            "h_a": fp32 [a, a],       # only h_a stored; H_b regenerated from b_dim
            "a_dim": int,
            "b_dim": int,             # power of 2; H_b is sylvester(b_dim)
            "in_features": int,       # == a_dim * b_dim, must match shape[1]
        },
    }

Reconstruction formula:
    g = c // group_size                      # which group column c belongs to
    W[r, c] = centroids_per_group[g][indices[r, c]] * scale_per_group[r, g]

Usage:
    from ml8_io import load_ml8_layer, reconstruct_weight
    blob = load_ml8_layer("/path/to/layer.pt")
    W = reconstruct_weight(blob)
    assert tuple(W.shape) == tuple(blob["shape"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def get_awq(blob: dict[str, Any]):
    """Reconstruct the layer's AWQ per-input-channel scale, if any.

    Returns None for legacy blobs, blob['awq'] is None, or kind=='none'.
    Otherwise returns the s tensor (shape (in_features,)).
    """
    spec = blob.get("awq")
    if spec is None:
        return None
    kind = spec.get("kind")
    if kind in (None, "none"):
        return None
    s = spec.get("s")
    if s is None:
        raise ValueError(f"awq spec missing 's' tensor: {spec}")
    return s


def reconstruct_inference_weight(blob: dict[str, Any]) -> torch.Tensor:
    """Dequantize and absorb both rotation and AWQ scale into the weight.

    Pipeline (matches calibrate_ml8.py order: AWQ → rotation → quant):
      1. dequant gives W in rotated, AWQ-rescaled basis
      2. rotation.inverse undoes the rotation (W @ Q.T)
      3. absorb_awq_in_reconstruction multiplies each col by s (W * diag(s))

    No-rotation, no-awq blob: returns plain `reconstruct_weight(blob)` unchanged.
    Backward-compatible with all earlier blob formats.
    """
    W = reconstruct_weight(blob)
    rotation = get_rotation(blob)
    if rotation is not None:
        W = rotation.inverse(W)
    awq_s = get_awq(blob)
    if awq_s is not None:
        # Lazy import to keep ml8_io a leaf in dependency order for legacy callers.
        from awq import absorb_awq_in_reconstruction
        W = absorb_awq_in_reconstruction(W, awq_s.to(device=W.device, dtype=W.dtype))
    return W


def get_rotation(blob: dict[str, Any]):
    """Reconstruct the layer's input rotation, if any. Returns None for legacy blobs.

    Blob schema (backward-compatible):
      - missing "rotation" key                       → None (Saturday's blobs)
      - blob["rotation"] is None                     → None
      - blob["rotation"]["kind"] == "identity"       → None
      - blob["rotation"]["kind"] == "kronecker_orth_sylvester"
                                                     → KroneckerRotation.from_dict(...)
    """
    spec = blob.get("rotation")
    if spec is None:
        return None
    kind = spec.get("kind")
    if kind in (None, "identity"):
        return None
    # Import lazily so legacy callers that never touch rotation don't pay the import.
    from kronecker_rotation import KroneckerRotation
    return KroneckerRotation.from_dict(spec)


def load_ml8_layer(path: str | Path) -> dict[str, Any]:
    """Load a single layer's ml8-4 blob."""
    return torch.load(path, map_location="cpu", weights_only=True)


def reconstruct_weight(blob: dict[str, Any]) -> torch.Tensor:
    """Dequantize: produce the float reconstruction of the original weight."""
    indices = blob["indices"]                       # [rows, in_features] int
    centroids = blob["centroids_per_group"]         # [n_groups, n_centroids] fp
    scales = blob["scale_per_group"]                # [rows, n_groups] fp
    group_size = int(blob["group_size"])
    rows, in_features = blob["shape"]
    n_groups = centroids.shape[0]

    # Sanity
    assert indices.shape == (rows, in_features), \
        f"indices shape {tuple(indices.shape)} != {(rows, in_features)}"
    assert scales.shape == (rows, n_groups), \
        f"scales shape {tuple(scales.shape)} != {(rows, n_groups)}"

    # For each column c, group index g = c // group_size.
    # Vectorized: build column->group mapping and gather.
    dev = indices.device
    centroids = centroids.to(dev).float()
    scales = scales.to(dev).float()
    indices = indices.long()  # for gather

    col_idx = torch.arange(in_features, device=dev)
    group_idx = col_idx // group_size                 # [in_features]

    # For each (r, c): val = centroids[group_idx[c], indices[r, c]] * scales[r, group_idx[c]]
    # Build per-column centroid LUT broadcast: [in_features, n_centroids] = centroids[group_idx]
    cent_cols = centroids[group_idx]                  # [in_features, n_centroids]
    # Gather centroid value per (r, c): need centroids[group_idx[c], indices[r, c]]
    # Equivalent: cent_cols[c, indices[r, c]]
    # Use advanced indexing.
    # Expand cent_cols to [1, in_features, n_centroids], gather along last dim by indices [rows, in_features, 1]
    centroid_vals = torch.gather(
        cent_cols.unsqueeze(0).expand(rows, -1, -1),  # [rows, in_features, n_centroids]
        2,
        indices.unsqueeze(-1)                         # [rows, in_features, 1]
    ).squeeze(-1)                                     # [rows, in_features]

    # Per-column scale lookup: scales[r, group_idx[c]] → [rows, in_features]
    scale_cols = scales[:, group_idx]                 # [rows, in_features]

    return centroid_vals * scale_cols


def bits_per_value(blob: dict[str, Any], scale_dtype_bits: int = 16) -> float:
    """Compute effective bits-per-value for this layer's ml8 encoding.

    indices: ceil(log2(n_centroids)) bits per value (typically 4)
    centroids: n_centroids * lut_bits per group (amortized over rows*group_size values)
    scales: scale_dtype_bits per (row, group) = (rows * n_groups * scale_bits) / numel

    For ml8-4 (n_centroids=16, group_size=128, scales fp16):
        idx_bits = 4
        centroid overhead per value = 16 * 32 / (rows * 128)   (negligible for rows ≫ 1)
        scale overhead per value = 16 / 128 = 0.125
        Total ≈ 4.125 bpv (excluding small centroid overhead)
    """
    rows, in_features = blob["shape"]
    group_size = int(blob["group_size"])
    n_groups = (in_features + group_size - 1) // group_size
    n_centroids = int(blob["n_centroids"])

    idx_bits = (n_centroids - 1).bit_length()        # ceil(log2(n_centroids))
    numel = rows * in_features
    idx_total = numel * idx_bits
    centroid_total = n_groups * n_centroids * 32     # centroids stored fp32
    scale_total = rows * n_groups * scale_dtype_bits

    total_bits = idx_total + centroid_total + scale_total
    return total_bits / numel


# Tiny self-test ─────────────────────────────────────────────────────────────

def _self_test() -> bool:
    """Synthetic round-trip: build a fake blob, reconstruct, verify shape + values."""
    rows, in_features, group_size, n_centroids = 8, 16, 4, 16
    n_groups = in_features // group_size
    # Random centroids in [-1, 1]
    centroids = torch.linspace(-1, 1, n_centroids).unsqueeze(0).expand(n_groups, -1).clone()
    # Random per-row scales
    scales = torch.rand(rows, n_groups) + 0.1
    # Random indices
    indices = torch.randint(0, n_centroids, (rows, in_features), dtype=torch.int8)

    blob = {
        "name": "test",
        "shape": [rows, in_features],
        "group_size": group_size,
        "n_centroids": n_centroids,
        "indices": indices,
        "centroids_per_group": centroids,
        "scale_per_group": scales,
    }

    W = reconstruct_weight(blob)
    assert tuple(W.shape) == (rows, in_features), W.shape

    # Spot-check: a few (r, c) values
    for r in range(rows):
        for c in range(in_features):
            g = c // group_size
            expected = centroids[g, indices[r, c].long()] * scales[r, g]
            got = W[r, c]
            assert torch.allclose(got, expected, atol=1e-6), \
                f"mismatch at ({r}, {c}): expected {expected} got {got}"

    bpv = bits_per_value(blob)
    print(f"  self-test PASSED. {rows}x{in_features} group={group_size} "
          f"n_centroids={n_centroids} → {bpv:.3f} bpv")
    return True


if __name__ == "__main__":
    _self_test()
