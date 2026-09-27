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

_ML8_FIT_EPS = 1e-8  # matches convert_fp8_rotated._ML8_FIT_EPS


def _lloyd_max_signed_batched_fast(samples, *, n_levels: int, n_iter: int):
    """Numerically-equivalent-up-to-summation-order fast path for
    centroid_quantizer._lloyd_max_signed_batched(fit_loss="mse").

    Exploits that Lloyd-Max on 1-D data has an exact O(log M) per-iteration
    form: sort each group's pooled samples ONCE; a Lloyd-Max iteration's bin
    membership is a monotonic function of sample value (bin index = number
    of decision boundaries below the sample), so on the sorted array each
    bin is a contiguous run and its (count, sum) come from two
    torch.searchsorted lookups (boundary -> split position) plus a
    float64 prefix-sum difference, instead of an O(M) 16-way masked
    reduction. Same init (torch.quantile per row), same per-iteration rule
    (decision boundaries = midpoints of adjacent centroids; empty bins keep
    the previous iteration's centroid), same fixed iteration count with no
    early-stop, same final `torch.sort` -- see
    centroid_quantizer._lloyd_max_signed_batched's docstring for why those
    choices are load-bearing (this is a drop-in replacement, not a new
    algorithm).

    samples: [G, M] float tensor (one Lloyd-Max fit per row of G groups).
    Returns centroids [G, n_levels] float, sorted ascending per row.

    Only fit_loss="mse" (uniform per-sample weight) is supported here --
    the "mag_weighted" path used by the GPTQ calibration pipeline isn't on
    this data-free wp-forge path and doesn't have a comparable prefix-sum
    trick (weights depend on the sample value itself, but a *weighted*
    prefix sum of `w_i * s_i` and `w_i` works the same way -- left for a
    future caller that needs it).
    """
    import torch

    assert samples.dim() == 2, f"expected [G, M], got shape {tuple(samples.shape)}"
    G, M = samples.shape
    device = samples.device
    dtype = samples.dtype

    sorted_samples, _ = torch.sort(samples, dim=1)  # [G, M]

    # Same init as the reference (torch.quantile(samples, q, dim=1),
    # interpolation="linear", the torch.quantile default), computed from
    # the sort we already have instead of a second, redundant internal
    # sort inside torch.quantile itself: pos = q*(M-1), linearly
    # interpolate between the two neighboring sorted samples. Verified
    # bit-for-bit equivalent (max abs diff ~2e-7, float32 rounding) to
    # torch.quantile(samples, q, dim=1) across random inputs.
    q = torch.linspace(0.0, 1.0, n_levels, device=device, dtype=dtype)
    pos = q * (M - 1)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    frac = (pos - lo.to(dtype)).view(1, n_levels)
    lo_v = sorted_samples[:, lo]  # [G, n_levels]
    hi_v = sorted_samples[:, hi]  # [G, n_levels]
    centroids = (lo_v * (1.0 - frac) + hi_v * frac).contiguous()  # [G, n_levels]
    # float64 prefix sums -- pooled sample counts run into the 100k-1M range
    # per group (rows * group_size), and float32 cumsum over that many
    # terms measurably loses precision vs. a masked float32 .sum(dim=1).
    cumsum = torch.zeros((G, M + 1), dtype=torch.float64, device=device)
    torch.cumsum(sorted_samples.double(), dim=1, out=cumsum[:, 1:])
    # Preallocated once and reused every iteration (only the middle
    # n_levels-1 columns change) instead of a fresh torch.cat per
    # iteration -- the first/last columns (0 and M, the "everything before
    # the first edge" / "everything after the last edge" bounds) are
    # loop-invariant.
    bounds = torch.empty((G, n_levels + 1), dtype=torch.long, device=device)
    bounds[:, 0] = 0
    bounds[:, -1] = M

    for _ in range(n_iter):
        edges = (centroids[:, :-1] + centroids[:, 1:]) / 2.0  # [G, n_levels-1]
        # split[:, j] = count of samples <= edges[:, j] in the sorted array
        # (right=True / side='right' -- matches torch.searchsorted(edges,
        # samples)'s default right=False bin-index convention: bin(x) =
        # count(edges < x), so bin(x) <= j  <=>  x <= edges[j]).
        torch.searchsorted(sorted_samples, edges.contiguous(), right=True, out=bounds[:, 1:-1])
        counts = bounds[:, 1:] - bounds[:, :-1]  # [G, n_levels], long
        sums = (torch.gather(cumsum, 1, bounds[:, 1:])
                - torch.gather(cumsum, 1, bounds[:, :-1]))  # [G, n_levels], float64
        nonzero = counts > 0
        means = (sums / counts.clamp_min(1).to(torch.float64)).to(dtype)
        # Empty bins keep the previous iteration's centroid (matches the
        # reference's `new = centroids.clone()` + skip-if-mask-empty).
        centroids = torch.where(nonzero, means, centroids)

    return torch.sort(centroids, dim=1).values


def _assign_ml8_indices_fast(w_rot, centroids, group_size: int = 64):
    """Fast nearest-centroid assignment: replaces the 16-way distance
    argmin in convert_fp8_rotated._assign_ml8_indices with one
    torch.searchsorted against the (sorted centroids') midpoints --
    O(log 16) instead of O(16) per element, and vectorized across groups.

    Tie-break equivalence: the reference iterates k=0..15 in order and
    only replaces `best` on a STRICT `<` distance improvement, so an exact
    tie (x_norm equidistant between centroid k and k+1) resolves to the
    LOWER index k. torch.searchsorted(midpoints, x, right=False) (side
    'left') returns the count of midpoints strictly less than x; at x
    exactly on a midpoint, that midpoint is not counted, so the result is
    still k -- same tie rule, for free.

    w_rot: [rows, K] torch.Tensor (already rotated). centroids: [n_groups,
    n_centroids] (sorted ascending, e4m3-snapped). Returns (indices int8
    [rows, K] in [0, n_centroids-1], scale fp32 [rows, n_groups]) --
    identical contract to convert_fp8_rotated._assign_ml8_indices.
    """
    import torch

    rows, K = w_rot.shape
    n_groups, n_centroids = centroids.shape
    x = w_rot.view(rows, n_groups, group_size)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(_ML8_FIT_EPS)  # [rows, n_groups, 1]
    x_norm = x / scale

    if n_centroids <= 1:
        idx = torch.zeros((rows, n_groups, group_size), dtype=torch.int64, device=w_rot.device)
    else:
        midpoints = (centroids[:, :-1] + centroids[:, 1:]) / 2.0  # [n_groups, n_centroids-1]
        # torch.searchsorted needs its batch (leading) dims to match, so put
        # n_groups first: [n_groups, rows * group_size].
        x_t = x_norm.permute(1, 0, 2).reshape(n_groups, rows * group_size).contiguous()
        idx_t = torch.searchsorted(midpoints.contiguous(), x_t, right=False)
        idx = idx_t.view(n_groups, rows, group_size).permute(1, 0, 2)

    indices = idx.to(torch.int8).reshape(rows, K)
    scale_out = scale.view(rows, n_groups)
    return indices, scale_out


def _fit_centroids(w_rot, group_size: int, n_centroids: int, fit_rows: int,
                    n_iter: int = 25, fast: bool = True) -> "object":
    """Per-K-group Lloyd-Max mse fit (see module docstring for why this is a
    reimplementation of convert_fp8_rotated._fit_ml8_centroids rather than an
    import of it). w_rot: torch.Tensor [rows, K], already rotated.

    Returns e4m3-snapped centroids [K/group_size, n_centroids] torch.Tensor.

    `fast=True` (default) uses `_lloyd_max_signed_batched_fast` (sorted
    samples + prefix sums, see its docstring); `fast=False` uses
    centroid_quantizer._lloyd_max_signed_batched (the reference O(G*M*16)
    per-iteration path), kept reachable for tests / equivalence checks.
    """
    _ensure_paths()
    from centroid_quantizer import snap_to_e4m3  # noqa: E402

    rows, k = w_rot.shape
    n_groups = k // group_size
    n_sample = min(fit_rows, rows)
    # Uniformly spaced (not random) row subsample -> deterministic given
    # (rows, fit_rows), matching convert_fp8_rotated._fit_ml8_centroids.
    idx = np.unique(np.linspace(0, rows - 1, num=n_sample, dtype=np.int64))
    pooled = w_rot[idx].reshape(len(idx), n_groups, group_size)
    scale = pooled.abs().amax(dim=-1, keepdim=True).clamp_min(_ML8_FIT_EPS)
    x_norm = pooled / scale
    # [n_groups, n_sample * group_size]: one Lloyd-Max row per K-group, fit
    # together in a single batched call (the required "batch the Lloyd-Max
    # over groups" perf path).
    samples = x_norm.permute(1, 0, 2).reshape(n_groups, -1).contiguous()
    if fast:
        centroids = _lloyd_max_signed_batched_fast(samples, n_levels=n_centroids, n_iter=n_iter)
    else:
        from centroid_quantizer import _lloyd_max_signed_batched  # noqa: E402
        centroids = _lloyd_max_signed_batched(
            samples, n_levels=n_centroids, n_iter=n_iter, fit_loss="mse"
        )
    return snap_to_e4m3(centroids)


def quantize_role_ml8_4(f32: np.ndarray, rotation, fit_rows: int, fast: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """f32: [rows, K], ONE expert's weight for one role. Returns
    (packed [rows, (K/64)*36] uint8, centroids_bytes [K/64, 16] uint8 F8_E4M3).

    `fast=True` (default) uses the sorted-samples Lloyd-Max fit and the
    searchsorted index assignment; `fast=False` uses the original
    O(G*M*16)-per-iteration fit and the O(16)-per-element distance argmin
    (kept reachable for equivalence tests).
    """
    _ensure_paths()
    import torch
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8, QK_ML8, N_CENTROIDS  # noqa: E402

    if f32.ndim != 2:
        raise Ml8Error(f"ml8_4: expected 2D (rows, cols), got {f32.shape}")
    if f32.shape[1] % QK_ML8 != 0:
        raise Ml8Error(f"ml8_4: cols {f32.shape[1]} not a multiple of QK_ML8={QK_ML8}")
    w = torch.from_numpy(np.ascontiguousarray(f32, dtype=np.float32))
    w_rot = rotation.forward(w) if rotation is not None else w
    centroids = _fit_centroids(w_rot, QK_ML8, N_CENTROIDS, fit_rows, fast=fast)
    if fast:
        indices, scale = _assign_ml8_indices_fast(w_rot, centroids, group_size=QK_ML8)
    else:
        from convert_fp8_rotated import _assign_ml8_indices  # noqa: E402
        indices, scale = _assign_ml8_indices(w_rot, centroids, group_size=QK_ML8)
    packed = pack_ml8_blocks(indices, scale)
    centroids_bytes = cast_centroids_to_fp8(centroids)
    return packed, centroids_bytes


def _fit_centroids_multi(w_rot_multi, group_size: int, n_centroids: int, fit_rows: int,
                          n_iter: int = 25):
    """Batched-over-experts version of `_fit_centroids`: fits one LUT PER
    (expert, K-group) but as a single `_lloyd_max_signed_batched_fast` call
    over a combined (expert, group) batch axis instead of a Python loop
    over experts. w_rot_multi: [n_e, rows, K] (already rotated). Returns
    e4m3-snapped centroids [n_e, K/group_size, n_centroids]."""
    _ensure_paths()
    from centroid_quantizer import snap_to_e4m3  # noqa: E402

    n_e, rows, k = w_rot_multi.shape
    n_groups = k // group_size
    n_sample = min(fit_rows, rows)
    idx = np.unique(np.linspace(0, rows - 1, num=n_sample, dtype=np.int64))
    pooled = w_rot_multi[:, idx, :].reshape(n_e, len(idx), n_groups, group_size)
    scale = pooled.abs().amax(dim=-1, keepdim=True).clamp_min(_ML8_FIT_EPS)
    x_norm = pooled / scale
    # Combine (expert, group) into one batch axis, expert-major -- matches
    # centroids.reshape(n_e, n_groups, ...) on the way back out.
    samples = x_norm.permute(0, 2, 1, 3).reshape(n_e * n_groups, -1).contiguous()
    centroids = _lloyd_max_signed_batched_fast(samples, n_levels=n_centroids, n_iter=n_iter)
    centroids = snap_to_e4m3(centroids)
    return centroids.reshape(n_e, n_groups, n_centroids)


def _assign_ml8_indices_multi(w_rot_multi, centroids_multi, group_size: int = 64):
    """Batched-over-experts version of `_assign_ml8_indices_fast`.
    w_rot_multi: [n_e, rows, K] (already rotated, SAME row count for every
    expert -- always true here, one role's weight shape is shared across
    all experts of a layer). centroids_multi: [n_e, n_groups, n_centroids].
    Returns (indices int8 [n_e, rows, K], scale fp32 [n_e, rows, n_groups])."""
    import torch

    n_e, rows, K = w_rot_multi.shape
    _n_e2, n_groups, n_centroids = centroids_multi.shape
    x = w_rot_multi.view(n_e, rows, n_groups, group_size)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(_ML8_FIT_EPS)  # [n_e, rows, n_groups, 1]
    x_norm = x / scale

    if n_centroids <= 1:
        idx = torch.zeros((n_e, rows, n_groups, group_size), dtype=torch.int64, device=w_rot_multi.device)
    else:
        centroids_flat = centroids_multi.reshape(n_e * n_groups, n_centroids)
        midpoints = (centroids_flat[:, :-1] + centroids_flat[:, 1:]) / 2.0  # [n_e*n_groups, n_centroids-1]
        # rows is shared across experts -> combine (expert, group) into the
        # searchsorted batch axis while keeping rows in the per-batch-row
        # values dimension: [n_e, rows, n_groups, gs] -> [n_e*n_groups, rows*gs].
        x_t = (x_norm.permute(0, 2, 1, 3)
               .reshape(n_e * n_groups, rows * group_size).contiguous())
        idx_t = torch.searchsorted(midpoints.contiguous(), x_t, right=False)
        idx = (idx_t.view(n_e, n_groups, rows, group_size)
               .permute(0, 2, 1, 3))

    indices = idx.to(torch.int8).reshape(n_e, rows, K)
    scale_out = scale.view(n_e, rows, n_groups)
    return indices, scale_out


def quantize_experts_ml8_4(stack: np.ndarray, rotation, fit_rows: int,
                           fast: bool = True, expert_chunk: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """stack: [n_expert, rows, K] f32, one role of a fused gather. Centroids
    are fit PER EXPERT (required even in the fused path -- two experts with
    very different scales must not share a LUT). Returns
    (packed [n_expert, rows, (K/64)*36] uint8,
     centroids [n_expert, K/64, 16] uint8 F8_E4M3).

    `fast=True` (default): rotates the whole stack in one pass, then fits
    + assigns `expert_chunk` experts at a time via `_fit_centroids_multi` /
    `_assign_ml8_indices_multi` (one batched Lloyd-Max / searchsorted call
    per chunk instead of one Python-level call per expert) -- bounds peak
    memory to O(expert_chunk * rows * K) rather than O(n_expert * rows * K)
    while still cutting per-expert kernel-launch overhead by expert_chunk.
    `fast=False` falls back to the original per-expert Python loop (kept
    reachable for equivalence tests)."""
    _ensure_paths()
    if stack.ndim != 3:
        raise Ml8Error(f"ml8_4: expected 3D (n_expert, rows, cols), got {stack.shape}")
    n_e = stack.shape[0]

    if not fast:
        packed = []
        cents = []
        for e in range(n_e):
            p, c = quantize_role_ml8_4(stack[e], rotation, fit_rows, fast=False)
            packed.append(p)
            cents.append(c)
        return np.ascontiguousarray(np.stack(packed, axis=0)), np.ascontiguousarray(np.stack(cents, axis=0))

    import torch
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8, QK_ML8, N_CENTROIDS  # noqa: E402

    _n_e, rows, k = stack.shape
    if k % QK_ML8 != 0:
        raise Ml8Error(f"ml8_4: cols {k} not a multiple of QK_ML8={QK_ML8}")
    w = torch.from_numpy(np.ascontiguousarray(stack.reshape(n_e * rows, k), dtype=np.float32))
    w_rot = (rotation.forward(w) if rotation is not None else w).view(n_e, rows, k)

    packed = []
    cents = []
    for start in range(0, n_e, max(1, expert_chunk)):
        end = min(start + max(1, expert_chunk), n_e)
        chunk = w_rot[start:end].contiguous()
        n_chunk = end - start
        centroids = _fit_centroids_multi(chunk, QK_ML8, N_CENTROIDS, fit_rows)
        indices, scale = _assign_ml8_indices_multi(chunk, centroids, group_size=QK_ML8)
        flat_packed = pack_ml8_blocks(indices.reshape(n_chunk * rows, k),
                                       scale.reshape(n_chunk * rows, k // QK_ML8))
        packed.append(flat_packed.reshape(n_chunk, rows, flat_packed.shape[-1]))
        flat_cent = cast_centroids_to_fp8(centroids.reshape(n_chunk * (k // QK_ML8), N_CENTROIDS))
        cents.append(flat_cent.reshape(n_chunk, k // QK_ML8, N_CENTROIDS))

    return np.ascontiguousarray(np.concatenate(packed, axis=0)), np.ascontiguousarray(np.concatenate(cents, axis=0))


def _quantize_experts_ml8_4_reference(stack: np.ndarray, rotation, fit_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Original per-expert Python-loop implementation, kept for tests."""
    if stack.ndim != 3:
        raise Ml8Error(f"ml8_4: expected 3D (n_expert, rows, cols), got {stack.shape}")
    n_e = stack.shape[0]
    packed = []
    cents = []
    for e in range(n_e):
        p, c = quantize_role_ml8_4(stack[e], rotation, fit_rows, fast=False)
        packed.append(p)
        cents.append(c)
    return np.ascontiguousarray(np.stack(packed, axis=0)), np.ascontiguousarray(np.stack(cents, axis=0))
