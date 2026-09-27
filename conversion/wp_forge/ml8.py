"""Data-free ROTATED ML8_FP8 / ML8_4 routed-expert quantization for wp-forge.

Reuses scripts/calibration modules for the exact on-disk format and
normalization recipe:
  - kronecker_rotation: factor_for_dim, random_orthogonal, KroneckerRotation,
    BlockHadamardRotation, the kronecker/block_hadamard kind ids.
  - centroid_quantizer: _lloyd_max_signed_batched (fit_loss="mse") + snap_to_e4m3.
  - scaled_fp8: quantize_scaled_fp8 (ML8_FP8's per-32-block scale+e4m3 cast).
  - ml8_to_gguf: pack_ml8_blocks (AOS layout), cast_centroids_to_fp8,
    pack_scaled_fp8_blocks, QK_ML8, N_CENTROIDS.
  - convert_fp8_rotated: _group_seed (deterministic per (rotation_seed, layer,
    group) seed so gate/up share one rotation) and _assign_ml8_indices.

convert_fp8_rotated._fit_ml8_centroids is NOT reused directly: it streams rows
straight out of a GGUFReader's mmap (row-index gather + bf16 widen), which is
the right move for its multi-GB whole-checkpoint streaming use case but is
awkward here -- wp-forge's ExpertStage._gather already fully materializes one
expert's f32 gate/up/down arrays in RAM before quantizing. `_fit_centroids`
below is the identical math (per-K-group Lloyd-Max mse fit over a
uniformly-subsampled row set, e4m3-snapped, absmax floor 1e-8) against an
in-memory array instead of a mmapped tensor reader.

All functions here import torch lazily (inside the function body, not at
module scope) so importing this module -- e.g. transitively from
quant.forge_quant_types() -- stays cheap when no ml8 quant is in play.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
_CALIB_DIR = _REPO / "scripts" / "calibration"

# rotation_meta kind ids -- mirrors kronecker_rotation.KRONECKER_ORTH_SYLVESTER_KIND_ID
# / BLOCK_HADAMARD_KIND_ID (duplicated here as plain ints so callers that only
# need the kind id, e.g. stages.py's sidecar dispatch, don't have to import
# torch just to read a constant; build_rotation() below imports the real
# module and returns these exact values).
KRONECKER_KIND_ID = 1
BLOCK_HADAMARD_KIND_ID = 2

ML8_QUANT_TYPES = ("ml8_fp8", "ml8_4")


class Ml8Error(ValueError):
    """ml8-specific quantization failure (shape/rotation/centroid mismatch)."""


def _ensure_paths() -> None:
    for p in (str(_CALIB_DIR), str(_REPO / "gguf-py"), str(_REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_rotation(k: int, rotation: str, rotation_seed: int, max_b: int,
                    layer: int | None, group: str):
    """Build the rotation for one (layer, role-group) pair.

    Returns (rotation_obj_or_None, kind_id, a, b). kind_id/a/b are 0 when
    rotation == "none" (no rotation_meta sidecar is written for that role).
    """
    _ensure_paths()
    from kronecker_rotation import (  # noqa: E402
        KroneckerRotation, BlockHadamardRotation, factor_for_dim, random_orthogonal,
        KRONECKER_ORTH_SYLVESTER_KIND_ID, BLOCK_HADAMARD_KIND_ID as _BH_ID,
    )
    from convert_fp8_rotated import _group_seed  # noqa: E402

    if rotation == "none":
        return None, 0, 0, 0
    a, b = factor_for_dim(k, max_b)
    if rotation == "kronecker":
        seed = _group_seed(rotation_seed, layer, group)
        h_a = random_orthogonal(a, seed)
        return KroneckerRotation(h_a, b), KRONECKER_ORTH_SYLVESTER_KIND_ID, a, b
    if rotation == "block_hadamard":
        return BlockHadamardRotation(k, b), _BH_ID, a, b
    raise Ml8Error(f"unknown rotation kind {rotation!r}")


def rotation_meta_bytes(a: int, b: int, k: int, kind_id: int) -> np.ndarray:
    return np.array([a, b, k, kind_id], dtype=np.int32)


def rotation_h_a_array(rotation) -> np.ndarray:
    """rotation.h_a (KroneckerRotation) as row-major float32 numpy."""
    return rotation.h_a.detach().cpu().float().contiguous().numpy()


# --- ML8_FP8 --------------------------------------------------------------

def quantize_role_ml8_fp8(f32: np.ndarray, rotation) -> np.ndarray:
    """f32: [rows, K]. Returns packed ML8_FP8 bytes [rows, (K/32)*34] uint8."""
    _ensure_paths()
    import torch
    from scaled_fp8 import quantize_scaled_fp8  # noqa: E402
    from ml8_to_gguf import pack_scaled_fp8_blocks  # noqa: E402

    if f32.ndim != 2:
        raise Ml8Error(f"ml8_fp8: expected 2D (rows, cols), got {f32.shape}")
    w = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32))
    w_rot = rotation.forward(w) if rotation is not None else w
    q = quantize_scaled_fp8(w_rot, group_size=32)
    return pack_scaled_fp8_blocks(q["e4m3"], q["scale"])


def quantize_experts_ml8_fp8(stack: np.ndarray, rotation) -> np.ndarray:
    """stack: [n_expert, rows, K] f32, one role of a fused gather. The SAME
    rotation applies to every expert (rotation is a function of (layer,
    role-group) only, not of the expert id). Returns [n_expert, rows,
    (K/32)*34] uint8."""
    if stack.ndim != 3:
        raise Ml8Error(f"ml8_fp8: expected 3D (n_expert, rows, cols), got {stack.shape}")
    n_e, rows, k = stack.shape
    flat = quantize_role_ml8_fp8(stack.reshape(n_e * rows, k), rotation)
    return np.ascontiguousarray(flat.reshape(n_e, rows, flat.shape[-1]))


# --- ML8_4 ------------------------------------------------------------------

def _fit_centroids(w_rot, group_size: int, n_centroids: int, fit_rows: int,
                    n_iter: int = 25) -> "object":
    """Per-K-group Lloyd-Max mse fit (see module docstring for why this is a
    reimplementation of convert_fp8_rotated._fit_ml8_centroids rather than an
    import of it). w_rot: torch.Tensor [rows, K], already rotated.

    Returns e4m3-snapped centroids [K/group_size, n_centroids] torch.Tensor.
    """
    _ensure_paths()
    from centroid_quantizer import _lloyd_max_signed_batched, snap_to_e4m3  # noqa: E402

    rows, k = w_rot.shape
    n_groups = k // group_size
    n_sample = min(fit_rows, rows)
    # Uniformly spaced (not random) row subsample -> deterministic given
    # (rows, fit_rows), matching convert_fp8_rotated._fit_ml8_centroids.
    idx = np.unique(np.linspace(0, rows - 1, num=n_sample, dtype=np.int64))
    pooled = w_rot[idx].reshape(len(idx), n_groups, group_size)
    scale = pooled.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    x_norm = pooled / scale
    # [n_groups, n_sample * group_size]: one Lloyd-Max row per K-group, fit
    # together in a single _lloyd_max_signed_batched call (the required
    # "batch the Lloyd-Max over groups" perf path).
    samples = x_norm.permute(1, 0, 2).reshape(n_groups, -1).contiguous()
    centroids = _lloyd_max_signed_batched(
        samples, n_levels=n_centroids, n_iter=n_iter, fit_loss="mse"
    )
    return snap_to_e4m3(centroids)


def quantize_role_ml8_4(f32: np.ndarray, rotation, fit_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """f32: [rows, K], ONE expert's weight for one role. Returns
    (packed [rows, (K/64)*36] uint8, centroids_bytes [K/64, 16] uint8 F8_E4M3).
    """
    _ensure_paths()
    import torch
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8, QK_ML8, N_CENTROIDS  # noqa: E402
    from convert_fp8_rotated import _assign_ml8_indices  # noqa: E402

    if f32.ndim != 2:
        raise Ml8Error(f"ml8_4: expected 2D (rows, cols), got {f32.shape}")
    if f32.shape[1] % QK_ML8 != 0:
        raise Ml8Error(f"ml8_4: cols {f32.shape[1]} not a multiple of QK_ML8={QK_ML8}")
    w = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32))
    w_rot = rotation.forward(w) if rotation is not None else w
    centroids = _fit_centroids(w_rot, QK_ML8, N_CENTROIDS, fit_rows)
    indices, scale = _assign_ml8_indices(w_rot, centroids, group_size=QK_ML8)
    packed = pack_ml8_blocks(indices, scale)
    centroids_bytes = cast_centroids_to_fp8(centroids)
    return packed, centroids_bytes


def quantize_experts_ml8_4(stack: np.ndarray, rotation, fit_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """stack: [n_expert, rows, K] f32, one role of a fused gather. Centroids
    are fit PER EXPERT (required even in the fused path -- two experts with
    very different scales must not share a LUT). Returns
    (packed [n_expert, rows, (K/64)*36] uint8,
     centroids [n_expert, K/64, 16] uint8 F8_E4M3)."""
    if stack.ndim != 3:
        raise Ml8Error(f"ml8_4: expected 3D (n_expert, rows, cols), got {stack.shape}")
    n_e = stack.shape[0]
    packed = []
    cents = []
    for e in range(n_e):
        p, c = quantize_role_ml8_4(stack[e], rotation, fit_rows)
        packed.append(p)
        cents.append(c)
    return np.ascontiguousarray(np.stack(packed, axis=0)), np.ascontiguousarray(np.stack(cents, axis=0))
