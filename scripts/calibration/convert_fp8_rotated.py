#!/usr/bin/env python3
"""convert_fp8_rotated — data-free BF16 GGUF -> GGUF with ML8_FP8 weights in a
rotated basis, for tensor-parallel inference (design "ML8_FP8 + rotation under
tensor parallel", 2026-09-17).

No calibration data, no GPTQ. Every 2D GEMM weight in the role allowlist is
cast to GGML_TYPE_ML8_FP8 (per-32-element block: fp16 scale + 32 OCP e4m3fn
bytes, scale = absmax/448 — see scaled_fp8.py) in the ROTATED basis, plus
rotation sidecars. Everything else (norms, biases, ssm_conv1d, ...) is copied
verbatim.

Rotation convention (matches ml8_runtime.py's Ml8Linear and calibrate_ml8_paged.py):
    W_rot[n, :] = rotation.forward(W[n, :])          (applied to the GEMM weight, at convert time)
    x_rot       = rotation.forward(x)                (applied to the activation, at inference time)
    y = W_rot . x_rot = W . x                        (exact, by orthogonality of `rotation`)
The C++ ml8 op already implements this forward/inverse convention for
calibrated ml8 blobs, so no C++ change is needed to consume these tensors.

Role allowlist (by GGUF tensor name, e.g. blk.3.attn_output.weight):
  K-split (TP)  -> block_hadamard(b=local_b): attn_output, ffn_down, ssm_out
  N-split (TP)  -> kronecker_orth_sylvester:  attn_q, attn_k, attn_v, attn_qkv,
                                               attn_gate, ffn_gate, ffn_up, output
  no rotation, Q8_0: token_embd, ssm_alpha, ssm_beta
  everything else (norms, biases, ssm_conv1d, ssm_a, ssm_dt.bias, nextn.*):
      copied verbatim, type unchanged.

The GGUF tensor is [N rows, K cols] with K = ne[0] (GGUFReader.shape is ne
order, i.e. shape[0] == K). Rotation acts along K.

Streams one tensor at a time from an mmapped GGUFReader; RAM stays bounded
regardless of checkpoint size (a 27B-class bf16 GGUF is ~55 GB, host RAM here
is 15 GB). Rotation matmuls run on --device (default cuda:0) in fp32.

--format fp8_b128 (design "FP8_B128 phase 2", 2026-09-17): same rotation,
but the quantized tier is GGML_TYPE_FP8_B128 instead of ML8_FP8 — one fp16
scale per aligned 128x128 tile (tile_absmax/448), replicated into every
row's 130-byte block for that tile. Weights not 128-aligned on both N and K
fall back to unrotated Q8_0 (see classify_tensor). Also enforces one
rotation per input group (role_group_key/_group_seed): weights that consume
the same activation (e.g. attn_qkv+attn_gate, ffn_gate+ffn_up) get the
identical h_a. --format ml8_fp8 (the default) is unaffected and remains
byte-identical to this script's behaviour before --format existed.

--scale-mode {tile,channel} (--format fp8_b128 only, default tile = today's
behaviour): "channel" replaces the per-128x128-tile fp16 scale with a single
per-output-row fp16 scale (row absmax over all K / 448), replicated into
every block_fp8_b128 of that row — the on-disk layout is unchanged, and the
tile invariant (every block in a tile shares one scale) holds trivially
since it's now shared row-wide. See _quantize_fp8_b128_gpu.
"""
from __future__ import annotations

import argparse
import hashlib
import mmap as _mmap_mod
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

from kronecker_rotation import (  # noqa: E402
    KroneckerRotation, BlockHadamardRotation, factor_for_dim, random_orthogonal,
    KRONECKER_ORTH_SYLVESTER_KIND_ID, BLOCK_HADAMARD_KIND_ID,
)
from scaled_fp8 import quantize_scaled_fp8  # noqa: E402
from ml8_to_gguf import (  # noqa: E402
    pack_scaled_fp8_blocks, _FP8_GROUP_SIZE, _FP8_BLOCK_BYTES,
    pack_ml8_blocks, cast_centroids_to_fp8, QK_ML8, ML8_BLOCK_BYTES, N_CENTROIDS,
)
from centroid_quantizer import snap_to_e4m3, _lloyd_max_signed  # noqa: E402


# ─── FP8_B128 (block-128 tile-scaled fp8) constants ────────────────────────
# block_fp8_b128 { fp16 d; uint8_t qs[128]; } — 130 bytes / 128 elems. The
# scale is shared by every block in an aligned 128x128 tile (replicated
# across the tile's 128 rows) — see gguf.GGML_QUANT_SIZES[FP8_B128] and the
# fp8b128-design.md CONVERTER INVARIANT.
_FP8B128_TILE = 128
_FP8B128_BLOCK_BYTES = 130
_FP8B128_MAX = 448.0     # OCP e4m3fn max representable magnitude
# Zero/degenerate tiles get this tiny positive scale. Must round to a
# non-zero fp16 value (the on-disk dtype for the scale) — fp16's smallest
# positive subnormal is ~5.96e-8, so 1e-12 (used elsewhere in this codebase
# for fp32-only clamps) would silently underflow to 0.0 here; 1e-6 rounds to
# a representable fp16 subnormal.
_FP8B128_EPS = 1e-6
_bs_b128, _ts_b128 = GGML_QUANT_SIZES[GGMLQuantizationType.FP8_B128]
assert (_bs_b128, _ts_b128) == (_FP8B128_TILE, _FP8B128_BLOCK_BYTES), (
    f"FP8_B128 GGML_QUANT_SIZES mismatch: expected ({_FP8B128_TILE}, "
    f"{_FP8B128_BLOCK_BYTES}), got ({_bs_b128}, {_ts_b128})"
)

# ─── ML8_4 (data-free codebook) constants ───────────────────────────────────
# block_ml8_4 { fp32 scale; uint8_t qs[32]; } — 36 bytes / 64 elems, per
# ggml-common.h QK_ML8/block_ml8_4 (imported from ml8_to_gguf, which is the
# authority for the exact on-disk layout — see pack_ml8_blocks). A shared
# 16-centroid e4m3 LUT per K-group is the ".centroids" sidecar, exactly as
# ml8_to_gguf.py's calibrated pipeline writes it, except here the LUT is fit
# on the weight alone (no GPTQ Hessian, no calibration corpus).
_bs_ml8_4, _ts_ml8_4 = GGML_QUANT_SIZES[GGMLQuantizationType.ML8_4]
assert (_bs_ml8_4, _ts_ml8_4) == (QK_ML8, ML8_BLOCK_BYTES), (
    f"ML8_4 GGML_QUANT_SIZES mismatch: expected ({QK_ML8}, {ML8_BLOCK_BYTES}), "
    f"got ({_bs_ml8_4}, {_ts_ml8_4})"
)
# Per-row-per-group absmax floor before normalizing — matches
# CentroidQuantizer.find_params's own scale = ...clamp_min(1e-8) exactly (the
# ml8_4 scale is stored as fp32, unlike fp8_b128's fp16 tile scale, so there's
# no fp16-underflow concern that would push this any higher).
_ML8_FIT_EPS = 1e-8
# Default row-subsample size for the data-free Lloyd-Max centroid fit (see
# --ml8-fit-rows): pooling every row of a 248320-row output.weight would be
# both slow and unnecessary — a uniform subsample of rows is enough to fit a
# stable per-K-group LUT, and every row is still individually assigned+scaled
# in the (memory-bounded, chunked) second pass.
_ML8_FIT_ROWS_DEFAULT = 65536


def _rotate_dtype_params(fmt: str):
    """(group_size, block_bytes, raw_dtype) for a rotated (non-q8_0) weight
    blob, keyed by --format. q8_0 fallback/exclusion tensors always use
    Q8_0's own (32, 34) regardless of format."""
    if fmt == "fp8_b128":
        return _FP8B128_TILE, _FP8B128_BLOCK_BYTES, GGMLQuantizationType.FP8_B128
    if fmt == "ml8_4":
        return QK_ML8, ML8_BLOCK_BYTES, GGMLQuantizationType.ML8_4
    return _FP8_GROUP_SIZE, _FP8_BLOCK_BYTES, GGMLQuantizationType.ML8_FP8


# ─── Role allowlist ─────────────────────────────────────────────────────────
# Roles are the tensor name with the leading "blk.{L}." (if any) stripped and
# the trailing ".weight" stripped, e.g. "blk.3.attn_output.weight" -> "attn_output",
# "output.weight" -> "output", "token_embd.weight" -> "token_embd".
K_SPLIT_HADAMARD_ROLES = {"attn_output", "ffn_down", "ssm_out"}
N_SPLIT_KRONECKER_ROLES = {
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate",
    "ffn_gate", "ffn_up", "output",
}
Q8_0_ROLES = {"token_embd", "ssm_alpha", "ssm_beta"}

# ─── One-rotation-per-input-group (fp8_b128 only, see module docstring) ────
# Roles that consume the SAME activation tensor must share a rotation (same
# h_a) within a layer: {attn_qkv, attn_gate}, {ffn_gate, ffn_up}, and
# {attn_q, attn_k, attn_v} for models that split qkv. Anything not listed
# here is its own singleton group (attn_output/ssm_out/ffn_down/output — the
# first two kinds are block_hadamard and don't have an h_a/seed at all, but
# they still get a stable per-(layer,role) group identity for uniformity).
_GROUPED_ROLES = {
    "attn_qkv": "attn_qkv_gate", "attn_gate": "attn_qkv_gate",
    "attn_q": "attn_qkv_split", "attn_k": "attn_qkv_split", "attn_v": "attn_qkv_split",
    "ffn_gate": "ffn_gate_up", "ffn_up": "ffn_gate_up",
}

_SKIP_FIELDS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


def role_group_key(role: str) -> str:
    """Group name for the one-rotation-per-input-group rule. Roles that share
    an activation map to the same key; anything else is a singleton keyed by
    its own role name."""
    return _GROUPED_ROLES.get(role, role)


def _parse_layer(name: str) -> int | None:
    """blk.{L}.... -> L, else None (top-level tensors like output.weight)."""
    return int(name.split(".", 2)[1]) if name.startswith("blk.") else None


def _group_seed(rotation_seed: int, layer: int | None, group_key: str) -> int:
    """Deterministic seed for a (layer, group) pair, so weights that consume
    the same activation get the identical h_a. Uses sha256 rather than
    Python's built-in str hash (which is randomized per-process by
    PYTHONHASHSEED) so the mapping is stable across runs/processes — required
    by test_seed_determinism_fp8_b128 and the group-sharing test."""
    tag = f"{int(rotation_seed)}:{layer if layer is not None else '-'}:{group_key}"
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)



def _sidecar_base(weight_name: str) -> str:
    """Sidecar tensors are named `<weight name without .weight>.<suffix>`, e.g.
    `blk.3.attn_output.rotation_meta` — the C++ loader builds the name with
    tn(tensor_id, "rotation_meta", il) (llama-arch.cpp LLM_TN_IMPL::str), which
    replaces the ".weight" suffix rather than appending to it; ml8_to_gguf.py
    follows the same convention."""
    return weight_name[:-len(".weight")] if weight_name.endswith(".weight") else weight_name

def tensor_role(name: str) -> tuple[str | None, bool]:
    """Return (role, is_weight). role is None for non-'.weight' tensors
    (norms/biases/etc — always copied verbatim regardless of role table)."""
    rest = name.split(".", 2)[2] if name.startswith("blk.") else name
    if not rest.endswith(".weight"):
        return None, False
    return rest[: -len(".weight")], True


def classify_tensor(name: str, shape: tuple, first_nextn_layer: int | None = None,
                    format: str = "ml8_fp8") -> tuple[str, str | None]:
    """Return (action, role). action in {"rotate_hadamard", "rotate_kronecker",
    "q8_0", "copy"}. Falls back to "copy" for anything not 2D even if the role
    would otherwise match (defensive — every allowlisted role in this model is
    a 2D GEMM weight, but this keeps the converter honest if that ever changes).

    Layers at or past `first_nextn_layer` (the MTP / nextn draft block, e.g.
    blk.64 when block_count=65 and nextn_predict_layers=1) are loaded by the
    C++ side without the ml8 sidecar registration, so a rotated weight there
    would leave its sidecars unconsumed ("wrong number of tensors"). Their 2D
    GEMM weights become plain Q8_0 instead.

    format="fp8_b128" adds one more fallback: FP8_B128 tiles are 128x128, so
    any would-be-rotated tensor whose row count (N) or column count (K) isn't
    a multiple of 128 can't be tiled — it drops to plain unrotated Q8_0
    instead (same as the nextn/MTP case). format="ml8_fp8" (the default)
    leaves classification byte-identical to before this fallback existed.

    format="ml8_4" adds the analogous fallback for the codebook format:
    block_ml8_4 groups are 64 columns wide (QK_ML8), so any would-be-rotated
    tensor whose K isn't a multiple of 64 drops to plain unrotated Q8_0
    (no row-count constraint — ml8_4 blocks are per-row, unlike fp8_b128's
    128x128 tiles)."""
    role, is_weight = tensor_role(name)
    if not is_weight or len(shape) != 2:
        return "copy", role
    if first_nextn_layer is not None and name.startswith("blk."):
        if int(name.split(".", 2)[1]) >= first_nextn_layer:
            return "q8_0", role
    if role in K_SPLIT_HADAMARD_ROLES:
        action = "rotate_hadamard"
    elif role in N_SPLIT_KRONECKER_ROLES:
        action = "rotate_kronecker"
    elif role in Q8_0_ROLES:
        return "q8_0", role
    else:
        return "copy", role
    if format == "fp8_b128":
        K, N = shape
        if K % 128 != 0 or N % 128 != 0:
            return "q8_0", role
    elif format == "ml8_4":
        K, N = shape
        if K % QK_ML8 != 0:
            return "q8_0", role
    return action, role


def _advise_dontneed(fd: int, offset: int, length: int) -> None:
    if not hasattr(os, "posix_fadvise") or fd < 0:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


# Default row-chunk size for GPU processing. A GEMM weight's rotation and
# quantization are both purely row-local (rotation.forward only mixes within
# a row's K elements; per-block FP8/Q8_0 scaling is per-row-per-group), so
# chunking along N (rows) is exact — see test_convert_chunked_matches_unchunked.
# Bounds peak GPU (and transient host) memory to ~chunk_rows * K * 4 bytes
# regardless of total tensor size: at 8192 rows the biggest K in this model
# (17408) is ~571 MB fp32, comfortably inside the R9700's 32 GB, vs. the ~4.9 GB
# single-shot fp32 materialization of token_embd/output.weight (1.27B elements)
# that OOM'd the first attempt at this (the GPU was also nearly full from
# unreleased prior-tensor allocations — torch.cuda.empty_cache() per chunk
# below addresses that side of it too).
_CHUNK_ROWS = 8192


def _bf16_rows_to_fp32_gpu(tensor, device: torch.device, row_start: int, row_end: int) -> torch.Tensor:
    """Widen rows [row_start:row_end) of a GGUFReader BF16 tensor to fp32 on `device`.

    `tensor.data` is a uint8 memmap view of the on-disk bytes; `.view(np.uint16)`
    is a zero-copy reinterpretation (no host allocation), and slicing rows out
    of that view is also zero-copy — only `np.ascontiguousarray` on the row
    slice actually materializes bytes, and only for this chunk.
    """
    if tensor.tensor_type != GGMLQuantizationType.BF16:
        raise ValueError(f"{tensor.name}: expected BF16 source, got {tensor.tensor_type.name}")
    u16_all = tensor.data.view(np.uint16)               # [N, K] zero-copy view of the mmap
    # np.array(..., copy=True) (not ascontiguousarray, which no-ops — and
    # leaves a read-only view — when the slice is already contiguous, as a
    # plain row-slice of a C-contiguous 2D array always is) to guarantee a
    # writable owned copy of just this chunk; torch.from_numpy requires that.
    chunk = np.array(u16_all[row_start:row_end], copy=True)
    t16 = torch.from_numpy(chunk).to(device=device)
    t32 = t16.to(torch.int32) << 16
    return t32.view(torch.float32).contiguous()


def _row_chunk_bounds(N: int, chunk_rows: int) -> list[tuple[int, int]]:
    """(start, end) row ranges covering [0, N), each of size chunk_rows except
    possibly the last — merged into the previous chunk if it would otherwise
    be a lone single row.

    That single-row case isn't just an aesthetic nit: ml8_to_gguf.pack_scaled_
    fp8_blocks (reused by _rotate_and_fp8) does
    `scale_fp16[:, b].detach().cpu().to(torch.float16).contiguous().view(torch.uint8)`,
    and PyTorch's `.contiguous()` is a documented no-op for a size-1 dim (it's
    trivially "contiguous" regardless of its stride) — so the stale non-unit
    stride from the `[:, b]` column-slice survives, and `.view(torch.uint8)`
    then rejects it (it checks the literal stride, not logical contiguity).
    Never handing that function a 1-row chunk sidesteps the bug entirely.
    """
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    bounds: list[list[int]] = []
    start = 0
    while start < N:
        end = min(start + chunk_rows, N)
        bounds.append([start, end])
        start = end
    if len(bounds) >= 2 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2][1] = bounds[-1][1]
        bounds.pop()
    return [(s, e) for s, e in bounds]


def _process_rotate_chunked(tensor, e: dict, device: torch.device, rotation,
                            chunk_rows: int = _CHUNK_ROWS, fmt: str = "ml8_fp8",
                            scale_mode: str = "tile") -> np.ndarray:
    """Rotate + quantize (ML8_FP8 scaled-fp8 or FP8_B128 tile/channel-scaled
    fp8, per `fmt`/`scale_mode`) one GEMM weight, one row-chunk of the GPU at
    a time. Returns the fully assembled packed bytes (CPU numpy). `scale_mode`
    is only consulted for fmt="fp8_b128" ("tile" = one fp16 scale per aligned
    128x128 tile, "channel" = one fp16 scale per output row, replicated into
    every block of that row — see _quantize_fp8_b128_gpu).

    For fmt="fp8_b128", chunk_rows must be a multiple of 128: a tile's scale
    spans all 128 rows of a row-tile, so a chunk boundary must never split a
    tile. Since N is already guaranteed %128==0 for any fp8_b128 rotate
    action (classify_tensor's fallback rule), and chunk_rows%128==0 here,
    every _row_chunk_bounds chunk (including the last) is itself a multiple
    of 128 rows — the remainder of two multiples of 128 is a multiple of 128,
    so the "merge lone trailing row" special case in _row_chunk_bounds can
    never fire in this mode. (channel mode's per-row scale doesn't strictly
    need this — a row's scale never spans a chunk boundary — but the same
    128-aligned chunking is kept for both scale modes for uniformity.)"""
    K, N = e["shape"]
    group_size, block_bytes, _raw_dtype = _rotate_dtype_params(fmt)
    if fmt == "fp8_b128" and chunk_rows % 128 != 0:
        raise ValueError(f"--chunk-rows={chunk_rows} must be a multiple of 128 for --format fp8_b128")
    n_blocks = K // group_size
    out = np.empty((N, n_blocks * block_bytes), dtype=np.uint8)
    for start, end in _row_chunk_bounds(N, chunk_rows):
        w = _bf16_rows_to_fp32_gpu(tensor, device, start, end)
        if fmt == "fp8_b128":
            out[start:end] = _rotate_and_fp8_b128(w, rotation, scale_mode=scale_mode)
        else:
            out[start:end] = _rotate_and_fp8(w, rotation)
        del w
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _process_q8_0_chunked(tensor, e: dict, device: torch.device,
                          chunk_rows: int = _CHUNK_ROWS) -> np.ndarray:
    """Q8_0-quantize one weight, one row-chunk of the GPU at a time."""
    K, N = e["shape"]
    n_blocks = K // _FP8_GROUP_SIZE
    out = np.empty((N, n_blocks * _FP8_BLOCK_BYTES), dtype=np.uint8)
    for start, end in _row_chunk_bounds(N, chunk_rows):
        w = _bf16_rows_to_fp32_gpu(tensor, device, start, end)
        out[start:end] = _quantize_q8_0_gpu(w)
        del w
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _bf16_rows_to_fp32_gpu_indices(tensor, device: torch.device,
                                   row_indices: np.ndarray) -> torch.Tensor:
    """Widen an arbitrary (non-contiguous) set of rows of a GGUFReader BF16
    tensor to fp32 on `device`. Like _bf16_rows_to_fp32_gpu but for a
    fancy-indexed row set — used to build the bounded-size Lloyd-Max fit
    subsample (see _fit_ml8_centroids/--ml8-fit-rows) without ever
    materializing the full tensor on the host."""
    if tensor.tensor_type != GGMLQuantizationType.BF16:
        raise ValueError(f"{tensor.name}: expected BF16 source, got {tensor.tensor_type.name}")
    u16_all = tensor.data.view(np.uint16)
    chunk = np.array(u16_all[row_indices], copy=True)
    t16 = torch.from_numpy(chunk).to(device=device)
    t32 = t16.to(torch.int32) << 16
    return t32.view(torch.float32).contiguous()


def _fit_ml8_centroids(tensor, K: int, N: int, device: torch.device, rotation,
                       fit_rows: int, group_size: int = QK_ML8,
                       n_centroids: int = N_CENTROIDS, n_iter: int = 25,
                       fit_loss: str = "mse", mag_weight_p: float = 5.0,
                       chunk_rows: int = _CHUNK_ROWS) -> torch.Tensor:
    """Fit one shared 16-centroid LUT per K-group (64 columns), pooling the
    scale-normalised ROTATED values of a uniformly-subsampled set of rows (up
    to `fit_rows`, see --ml8-fit-rows) across ALL N rows of the weight.

    Matches CentroidQuantizer.find_params's convention (per-row absmax
    scale over the group, floor 1e-8, signed Lloyd-Max fit on the normalised
    values, col_weights=None — i.e. no Hessian, since a data-free conversion
    has no calibration corpus) with TWO deliberate departures. (1) fit_loss
    defaults to plain "mse", NOT the KV cache's "mag_weighted" p=5: the
    magnitude-weighted fit puts every centroid at |c| >= 0.25*absmax (measured
    on Qwen3.8-27B blk.0.ffn_gate: no level below 0.25), so the small values
    that make up most of a weight row all round to +-0.25*absmax -- relL2
    0.28 vs the rotated source, against 0.09 for the mse fit and 0.10 for a
    Q4_0-style absmax quant of the same rows. The calibrated (GPTQ) pipeline
    could afford mag-weighting because its error feedback re-absorbed that
    bias into later columns; a data-free conversion has no such correction
    and the 2026-09-17 mag_weighted file produced token-salad output.
    (2) find_params pools every row of the group while this pools a bounded
    row subsample — the largest
    weight in this model (output.weight, 248320x5120) can't have its rotated
    fp32 form materialized in full within the 15 GB host RAM budget, and a
    uniform subsample is enough to fit a stable per-group LUT (every row is
    still individually scaled+assigned in the chunked second pass).

    Returns centroids [n_groups, n_centroids] float32 on `device`, already
    snapped to the e4m3 lattice (matches cast_centroids_to_fp8's input
    contract)."""
    n_groups = K // group_size
    n_sample = min(fit_rows, N)
    # Uniformly spaced (not random) sample row indices -> deterministic given
    # (N, fit_rows), independent of any RNG/seed state.
    sample_idx = np.unique(np.linspace(0, N - 1, num=n_sample, dtype=np.int64))

    pooled = torch.empty((len(sample_idx), K), dtype=torch.float32, device=device)
    for start, end in _row_chunk_bounds(len(sample_idx), chunk_rows):
        idx_chunk = sample_idx[start:end]
        w = _bf16_rows_to_fp32_gpu_indices(tensor, device, idx_chunk)
        pooled[start:end] = rotation.forward(w)
        del w
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled = pooled.view(len(sample_idx), n_groups, group_size)
    scale = pooled.abs().amax(dim=-1, keepdim=True).clamp_min(_ML8_FIT_EPS)  # [rows, n_groups, 1]
    x_norm = pooled / scale

    centroids = torch.empty((n_groups, n_centroids), dtype=torch.float32, device=device)
    for g in range(n_groups):
        samples = x_norm[:, g, :].reshape(-1)
        c = _lloyd_max_signed(
            samples, sample_col_idx=None, col_weights=None,
            n_levels=n_centroids, n_iter=n_iter,
            fit_loss=fit_loss, mag_weight_p=mag_weight_p,
        )
        centroids[g] = c.to(device=device, dtype=torch.float32)
    return snap_to_e4m3(centroids)


def _assign_ml8_indices(w_rot: torch.Tensor, centroids: torch.Tensor,
                        group_size: int = QK_ML8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(row,group) absmax-normalize `w_rot` [rows, K] and assign each
    element the index of its nearest centroid in `centroids` [n_groups, 16]
    (already e4m3-snapped) — the CentroidQuantizer.quantize() convention
    (dequant = centroids[idx] * scale). Iterates over the 16 centroids
    (rather than materializing a [rows, n_groups, group_size, 16] distance
    tensor) to keep peak memory at O(rows * K) instead of O(rows * K * 16).

    Returns (indices int8 [rows, K] in [0, 15], scale fp32 [rows, n_groups])."""
    rows, K = w_rot.shape
    n_groups = centroids.shape[0]
    x = w_rot.view(rows, n_groups, group_size)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(_ML8_FIT_EPS)  # [rows, n_groups, 1]
    x_norm = x / scale

    best_dist = None
    best_idx = torch.zeros((rows, n_groups, group_size), dtype=torch.int64, device=w_rot.device)
    for k in range(centroids.shape[1]):
        c = centroids[:, k].view(1, n_groups, 1)
        d = (x_norm - c).abs()
        if best_dist is None:
            best_dist = d
        else:
            mask = d < best_dist
            best_dist = torch.where(mask, d, best_dist)
            best_idx = torch.where(mask, torch.full_like(best_idx, k), best_idx)
    indices = best_idx.to(torch.int8).view(rows, K)
    scale_out = scale.view(rows, n_groups)
    return indices, scale_out


def _process_rotate_ml8_4_chunked(tensor, e: dict, device: torch.device, rotation,
                                  chunk_rows: int = _CHUNK_ROWS,
                                  fit_rows: int = _ML8_FIT_ROWS_DEFAULT
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Full data-free ml8_4 pipeline for one GEMM weight [N, K]: fit a
    per-K-group 16-centroid e4m3 LUT (pooled across a uniformly-subsampled
    `fit_rows` rows via _fit_ml8_centroids), then assign+pack ALL N rows
    chunk-wise on the GPU (memory-bounded the same way as the other formats —
    see _process_rotate_chunked).

    Returns (packed_block_bytes [N, n_groups*36] uint8 — block_ml8_4 layout
    via ml8_to_gguf.pack_ml8_blocks, centroids_bytes [n_groups, 16] uint8 —
    F8_E4M3 via ml8_to_gguf.cast_centroids_to_fp8)."""
    K, N = e["shape"]
    if K % QK_ML8 != 0:
        raise ValueError(f"{tensor.name}: K={K} not divisible by QK_ML8={QK_ML8}")
    n_groups = K // QK_ML8

    centroids = _fit_ml8_centroids(tensor, K, N, device, rotation, fit_rows,
                                   chunk_rows=chunk_rows)

    out = np.empty((N, n_groups * ML8_BLOCK_BYTES), dtype=np.uint8)
    for start, end in _row_chunk_bounds(N, chunk_rows):
        w = _bf16_rows_to_fp32_gpu(tensor, device, start, end)
        w_rot = rotation.forward(w)
        indices, scale = _assign_ml8_indices(w_rot, centroids)
        out[start:end] = pack_ml8_blocks(indices, scale)
        del w, w_rot, indices, scale
        if device.type == "cuda":
            torch.cuda.empty_cache()

    centroids_bytes = cast_centroids_to_fp8(centroids)
    return out, centroids_bytes


def _copy_field(writer: "gguf.GGUFWriter", name: str, field) -> None:
    types = field.types
    value = field.contents()
    primary = types[0]
    if primary == gguf.GGUFValueType.ARRAY:
        if len(types) < 2:
            raise ValueError(f"field {name!r}: ARRAY type without sub-type")
        writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub_type=types[1])
    else:
        writer.add_key_value(name, value, primary)


def _rotate_and_fp8(w: torch.Tensor, rotation, group_size: int = _FP8_GROUP_SIZE) -> np.ndarray:
    """Apply `rotation.forward` along K (last dim), quantize to scaled-FP8, pack."""
    w_rot = rotation.forward(w)
    q = quantize_scaled_fp8(w_rot, group_size=group_size)
    return pack_scaled_fp8_blocks(q["e4m3"], q["scale"])


def _quantize_fp8_b128_gpu(w: torch.Tensor, scale_mode: str = "tile") -> np.ndarray:
    """Quantize w [N, K] (N, K both %128==0) into packed FP8_B128 bytes,
    e4m3 = torch.float8_e4m3fn round-to-nearest of v/scale clamped to +-448.
    Returns [N, (K/128)*130] uint8, matching gguf.GGML_QUANT_SIZES[FP8_B128]
    byte layout (fp16 d then 128 e4m3 bytes per block). Stays entirely on
    `w`'s device except for the final .cpu() — only the packed bytes reach
    the host, same discipline as _quantize_q8_0_gpu.

    scale_mode="tile" (default): one fp16 scale per aligned 128x128 tile
    (tile_absmax/448), replicated into every row's block for that tile.

    scale_mode="channel": one fp16 scale per output ROW n (row_absmax over
    all K / 448), replicated into EVERY block of that row (so every block in
    a row shares the identical stored `d` — the 128x128 tile invariant, that
    every block within a tile shares one scale, holds trivially since it's
    now shared row-wide). Still requires N%128==0/K%128==0 like tile mode —
    channel mode reuses the same on-disk block_fp8_b128 layout and the same
    classify_tensor 128-alignment fallback rule, it just computes the scale
    per row instead of per tile.

    Either way, degenerate (all-zero) rows/tiles get the tiny positive
    _FP8B128_EPS floor rather than a zero scale."""
    N, K = w.shape
    if N % _FP8B128_TILE != 0 or K % _FP8B128_TILE != 0:
        raise ValueError(f"fp8_b128 requires N%128==0 and K%128==0, got N={N} K={K}")
    n_row_tiles = N // _FP8B128_TILE
    n_col_blocks = K // _FP8B128_TILE

    if scale_mode == "channel":
        row_absmax = w.abs().amax(dim=1, keepdim=True)            # [N, 1]
        scale_fp32 = (row_absmax / _FP8B128_MAX).clamp_min(_FP8B128_EPS)
        scale_fp16 = scale_fp32.to(torch.float16)                 # the value actually stored
        # Quantize with the *fp16-rounded* scale (not the fp32 pre-round
        # value) so decode(qs) * stored_scale reproduces v up to e4m3
        # rounding only — same discipline as tile mode.
        scale_bcast = scale_fp16.to(torch.float32)                # [N, 1]
        v = (w / scale_bcast).clamp(-_FP8B128_MAX, _FP8B128_MAX)
        e4m3 = v.to(torch.float8_e4m3fn).reshape(N, n_col_blocks, _FP8B128_TILE)
        scale_per_row = scale_fp16.repeat(1, n_col_blocks)        # [N, col_blocks] — same d in every block of the row
    elif scale_mode == "tile":
        wt = w.reshape(n_row_tiles, _FP8B128_TILE, n_col_blocks, _FP8B128_TILE)
        tile_absmax = wt.abs().amax(dim=(1, 3))                       # [row_tiles, col_blocks]
        scale_fp32 = (tile_absmax / _FP8B128_MAX).clamp_min(_FP8B128_EPS)
        scale_fp16 = scale_fp32.to(torch.float16)                     # the value actually stored
        scale_bcast = scale_fp16.to(torch.float32).reshape(n_row_tiles, 1, n_col_blocks, 1)
        v = (wt / scale_bcast).clamp(-_FP8B128_MAX, _FP8B128_MAX)
        e4m3 = v.to(torch.float8_e4m3fn).reshape(N, n_col_blocks, _FP8B128_TILE)
        scale_per_row = scale_fp16.repeat_interleave(_FP8B128_TILE, dim=0)  # [N, col_blocks]
    else:
        raise ValueError(f"scale_mode must be 'tile' or 'channel', got {scale_mode!r}")

    scale_bytes = scale_per_row.contiguous().cpu().numpy().view(np.uint8).reshape(
        N, n_col_blocks, 2)
    qs_bytes = e4m3.contiguous().cpu().view(torch.uint8).numpy().reshape(
        N, n_col_blocks, _FP8B128_TILE)
    return np.concatenate([scale_bytes, qs_bytes], axis=-1).reshape(
        N, n_col_blocks * _FP8B128_BLOCK_BYTES)


def _rotate_and_fp8_b128(w: torch.Tensor, rotation, scale_mode: str = "tile") -> np.ndarray:
    """Apply `rotation.forward` along K (last dim), quantize to FP8_B128."""
    w_rot = rotation.forward(w)
    return _quantize_fp8_b128_gpu(w_rot, scale_mode=scale_mode)


def _round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Round-half-away-from-zero (matches gguf.quants.np_roundf), on whatever
    device `x` lives on. torch.round is round-half-to-even, which would only
    ever disagree at an exact .5 tie — never observed for real fp32 weight
    data, but we match the reference convention exactly rather than rely on
    that being true."""
    a = x.abs()
    floored = torch.floor(a)
    b = floored + torch.floor(2 * (a - floored))
    return torch.sign(x) * b


def _quantize_q8_0_gpu(w: torch.Tensor) -> np.ndarray:
    """Bit-exact-with-gguf.quants Q8_0 quantizer that keeps the (potentially
    huge, e.g. token_embd/output.weight = 1.27B elements) fp32 activation on
    `w`'s device the whole time — only the packed uint8 bytes (8.5 bits/elem)
    ever reach the host. See test_convert_fp8_rotated.py for the bit-exact
    comparison against gguf.quants.quantize(..., Q8_0)."""
    N, K = w.shape
    assert K % _FP8_GROUP_SIZE == 0, f"K={K} not divisible by {_FP8_GROUP_SIZE}"
    n_blocks = K // _FP8_GROUP_SIZE
    blocks = w.reshape(N, n_blocks, _FP8_GROUP_SIZE)
    d = blocks.abs().amax(dim=-1, keepdim=True) / 127.0
    inv_d = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    qs = _round_away_from_zero(blocks * inv_d).to(torch.int8)
    d16 = d.squeeze(-1).to(torch.float16)  # [N, n_blocks]

    d_bytes = d16.contiguous().cpu().numpy().view(np.uint8).reshape(N, n_blocks, 2)
    qs_bytes = qs.contiguous().cpu().numpy().view(np.uint8).reshape(N, n_blocks, _FP8_GROUP_SIZE)
    return np.concatenate([d_bytes, qs_bytes], axis=-1).reshape(N, n_blocks * _FP8_BLOCK_BYTES)


def _emission_specs(tensor, e: dict) -> list[dict]:
    """Return the list of {name, byte_shape, dtype, nbytes, raw_dtype} blobs
    this GGUF tensor expands to (main weight + optional sidecars), computed
    purely from shapes/metadata — no tensor data is read or computed. Used to
    register tensor_info (pass 1) and must be produced identically (same
    order, same nbytes) as the actual data written in pass 2."""
    action = e["action"]
    if action in ("rotate_kronecker", "rotate_hadamard"):
        K, N = e["shape"]
        fmt = e.get("format", "ml8_fp8")
        group_size, block_bytes, raw_dtype = _rotate_dtype_params(fmt)
        n_blocks = K // group_size
        specs = [{
            "name": tensor.name,
            "byte_shape": (N, n_blocks * block_bytes),
            "dtype": np.uint8,
            "nbytes": N * n_blocks * block_bytes,
            "raw_dtype": raw_dtype,
        }]
        if fmt == "ml8_4":
            # Centroids sidecar: one shared 16-e4m3-centroid LUT per K-group,
            # exactly as ml8_to_gguf.cast_centroids_to_fp8 writes it — GGUF
            # ne-order shape [16, n_blocks] == numpy/byte_shape (n_blocks, 16).
            specs.append({
                "name": _sidecar_base(tensor.name) + ".centroids",
                "byte_shape": (n_blocks, N_CENTROIDS),
                "dtype": np.uint8,
                "nbytes": n_blocks * N_CENTROIDS,
                "raw_dtype": GGMLQuantizationType.F8_E4M3,
            })
        if action == "rotate_kronecker":
            a = e["a"]
            specs.append({
                "name": _sidecar_base(tensor.name) + ".rotation_h_a",
                "byte_shape": (a, a), "dtype": np.float32,
                "nbytes": a * a * 4, "raw_dtype": None,
            })
        specs.append({
            "name": _sidecar_base(tensor.name) + ".rotation_meta",
            "byte_shape": (4,), "dtype": np.int32,
            "nbytes": 16, "raw_dtype": None,
        })
        return specs
    if action == "q8_0":
        K, N = e["shape"]
        n_blocks = K // _FP8_GROUP_SIZE
        return [{
            "name": tensor.name,
            "byte_shape": (N, n_blocks * _FP8_BLOCK_BYTES),
            "dtype": np.uint8,
            "nbytes": N * n_blocks * _FP8_BLOCK_BYTES,
            "raw_dtype": GGMLQuantizationType.Q8_0,
        }]
    # copy: byte-identical to the source tensor's own on-disk layout.
    return [{
        "name": tensor.name,
        "byte_shape": tensor.data.shape,
        "dtype": tensor.data.dtype,
        "nbytes": tensor.n_bytes,
        "raw_dtype": tensor.tensor_type,
    }]


def _first_nextn_layer(reader: "gguf.GGUFReader") -> int | None:
    """block_count - nextn_predict_layers from the GGUF KV (any arch prefix), or
    None when the model has no nextn block."""
    block_count = None
    nextn = 0
    for key, field in reader.fields.items():
        if key.endswith(".block_count"):
            block_count = int(field.parts[field.data[0]][0])
        elif key.endswith(".nextn_predict_layers"):
            nextn = int(field.parts[field.data[0]][0])
    if block_count is None or nextn <= 0:
        return None
    return block_count - nextn


def build_plan(reader: "gguf.GGUFReader", rotation_seed: int, local_b: int, max_b: int,
              format: str = "ml8_fp8") -> list[dict]:
    """Compute the per-tensor conversion plan (name, shape, action, rotation
    kind/a/b) without doing any tensor math.

    format="ml8_fp8" (default, byte-identical to before this option existed):
    kronecker seeds are assigned by a stable index over the sorted names of
    kronecker-rotated tensors, so the plan (and the resulting rotation)
    doesn't depend on GGUF tensor order.

    format="fp8_b128": kronecker seeds are instead derived per (layer, input
    group) via _group_seed — weights that consume the same activation
    (role_group_key) get the identical seed, hence the identical h_a, per the
    one-rotation-per-input-group requirement."""
    first_nextn = _first_nextn_layer(reader)
    kron_names = sorted(
        t.name for t in reader.tensors
        if classify_tensor(t.name, tuple(int(s) for s in t.shape), first_nextn, format)[0] == "rotate_kronecker"
    )
    kron_index = {name: i for i, name in enumerate(kron_names)}

    plan = []
    for t in reader.tensors:
        shape = tuple(int(s) for s in t.shape)  # ne order: shape[0] == K for 2D
        action, role = classify_tensor(t.name, shape, first_nextn, format)
        entry = {"name": t.name, "shape": shape, "action": action, "role": role, "format": format}
        if action == "rotate_kronecker":
            K = shape[0]
            a, b = factor_for_dim(K, max_b=max_b)
            if format in ("fp8_b128", "ml8_4"):
                layer = _parse_layer(t.name)
                group_key = role_group_key(role)
                seed = _group_seed(rotation_seed, layer, group_key)
                entry["group"] = f"{layer if layer is not None else 'top'}.{group_key}"
            else:
                seed = rotation_seed + kron_index[t.name]
            entry.update(kind="kronecker_orth_sylvester", a=a, b=b, seed=seed)
        elif action == "rotate_hadamard":
            K = shape[0]
            if K % local_b != 0:
                raise ValueError(
                    f"{t.name}: K={K} not divisible by --local-b={local_b} "
                    f"(block_hadamard requires this)")
            entry.update(kind="block_hadamard", a=K // local_b, b=local_b)
        plan.append(entry)
    return plan


def print_plan(plan: list[dict]) -> None:
    counts: dict[str, int] = {}
    for e in plan:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    print(f"[dry-run] {len(plan)} tensors total")
    for action, n in sorted(counts.items()):
        print(f"  {action}: {n}")
    print("[dry-run] per-tensor plan:")
    for e in plan:
        if e["action"] in ("rotate_kronecker", "rotate_hadamard"):
            print(f"  {e['name']:45s} shape={e['shape']} action={e['action']:16s} "
                  f"kind={e['kind']} a={e['a']} b={e['b']}"
                  + (f" seed={e['seed']}" if "seed" in e else "")
                  + (f" group={e['group']}" if "group" in e else ""))
        else:
            print(f"  {e['name']:45s} shape={e['shape']} action={e['action']}")


def convert(src: Path, out: Path, rotation_seed: int, device_str: str,
           local_b: int, max_b: int, chunk_rows: int = _CHUNK_ROWS,
           format: str = "ml8_fp8", ml8_fit_rows: int = _ML8_FIT_ROWS_DEFAULT,
           scale_mode: str = "tile") -> dict:
    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    print(f"[base] {src}  arch={arch!r}  fields={len(reader.fields)}  tensors={len(reader.tensors)}")

    plan = build_plan(reader, rotation_seed, local_b, max_b, format=format)
    plan_by_name = {e["name"]: e for e in plan}

    device = torch.device(device_str)

    # Memory-pressure mitigation: sequential madvise on the reader's mmap +
    # per-tensor POSIX_FADV_DONTNEED, same trick as ml8_to_gguf.py — keeps
    # the source GGUF's pages from piling up in cache on a 15 GB host.
    base_fd = -1
    try:
        underlying = reader.data._mmap
        if hasattr(underlying, "madvise"):
            try:
                underlying.madvise(_mmap_mod.MADV_SEQUENTIAL)
            except (OSError, ValueError):
                pass
        try:
            base_fd = os.open(str(src), os.O_RDONLY)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(base_fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except OSError:
            base_fd = -1
    except AttributeError:
        pass

    # use_temp_file=False (default): with the two-pass registration below we
    # write each tensor's bytes straight to the output file via
    # write_tensor_data as it is computed, instead of spooling every
    # converted tensor (main weight + sidecars, ~29 GB total) through a
    # temp-file/page-cache buffer ahead of a single final copy — that spool
    # is what blew the 10 GB cgroup cap on a 15 GB host on the first attempt.
    writer = gguf.GGUFWriter(str(out), arch=arch)

    for name, field in reader.fields.items():
        if name in _SKIP_FIELDS:
            continue
        _copy_field(writer, name, field)
    _format_version = {"fp8_b128": 2, "ml8_4": 3}.get(format, 1)
    writer.add_key_value("fp8rot.format_version", _format_version, gguf.GGUFValueType.UINT32)
    writer.add_key_value("fp8rot.rotation_seed", int(rotation_seed), gguf.GGUFValueType.INT32)
    writer.add_key_value("fp8rot.local_b", int(local_b), gguf.GGUFValueType.INT32)

    # ── Pass 1: register every tensor's (name, shape, dtype, nbytes) — no
    # tensor data is read or computed here, so this is a cheap metadata-only
    # walk of the plan. add_tensor_info requires this to happen before the
    # output file is opened (write_header_to_file), and write_tensor_data
    # (pass 2) consumes these registrations strictly in insertion order.
    n_fields = sum(1 for name in reader.fields if name not in _SKIP_FIELDS) + 3
    emission_order: list[dict] = []
    for tensor in reader.tensors:
        e = plan_by_name[tensor.name]
        for spec in _emission_specs(tensor, e):
            writer.add_tensor_info(
                spec["name"], spec["byte_shape"], np.dtype(spec["dtype"]),
                spec["nbytes"], raw_dtype=spec["raw_dtype"])
            emission_order.append(spec)
    print(f"[fields] copied {n_fields - 3} + 3 fp8rot markers; "
          f"[pass1] registered {len(emission_order)} tensor blobs "
          f"({len(reader.tensors)} source tensors)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ── Pass 2: stream actual bytes, one GGUF tensor's worth of GPU work at a
    # time, straight into the file via write_tensor_data (no spooling).
    counts = {"rotate_hadamard": 0, "rotate_kronecker": 0, "q8_0": 0, "copy": 0}
    t_start = time.time()
    for i, tensor in enumerate(reader.tensors):
        e = plan_by_name[tensor.name]
        action = e["action"]
        if action == "rotate_kronecker":
            K, N = e["shape"]
            h_a = random_orthogonal(e["a"], seed=e["seed"]).to(device=device, dtype=torch.float32)
            rot = KroneckerRotation(h_a=h_a, b_dim=e["b"])
            if format == "ml8_4":
                packed, centroids_bytes = _process_rotate_ml8_4_chunked(
                    tensor, e, device, rot, chunk_rows=chunk_rows, fit_rows=ml8_fit_rows)
                writer.write_tensor_data(packed)
                writer.write_tensor_data(centroids_bytes)
            else:
                packed = _process_rotate_chunked(tensor, e, device, rot, chunk_rows=chunk_rows,
                                                 fmt=format, scale_mode=scale_mode)
                writer.write_tensor_data(packed)
            meta = np.array([e["a"], e["b"], K, KRONECKER_ORTH_SYLVESTER_KIND_ID], dtype=np.int32)
            writer.write_tensor_data(h_a.detach().cpu().contiguous().numpy())
            writer.write_tensor_data(meta)
            del h_a
            if device.type == "cuda":
                torch.cuda.empty_cache()
            counts["rotate_kronecker"] += 1
        elif action == "rotate_hadamard":
            K, N = e["shape"]
            rot = BlockHadamardRotation(in_features=K, b_dim=e["b"])
            if format == "ml8_4":
                packed, centroids_bytes = _process_rotate_ml8_4_chunked(
                    tensor, e, device, rot, chunk_rows=chunk_rows, fit_rows=ml8_fit_rows)
                writer.write_tensor_data(packed)
                writer.write_tensor_data(centroids_bytes)
            else:
                packed = _process_rotate_chunked(tensor, e, device, rot, chunk_rows=chunk_rows,
                                                 fmt=format, scale_mode=scale_mode)
                writer.write_tensor_data(packed)
            meta = np.array([e["a"], e["b"], K, BLOCK_HADAMARD_KIND_ID], dtype=np.int32)
            writer.write_tensor_data(meta)
            counts["rotate_hadamard"] += 1
        elif action == "q8_0":
            packed = _process_q8_0_chunked(tensor, e, device, chunk_rows=chunk_rows)
            writer.write_tensor_data(packed)
            counts["q8_0"] += 1
        else:
            cloned = np.ascontiguousarray(tensor.data)
            writer.write_tensor_data(cloned)
            del cloned
            counts["copy"] += 1

        if base_fd >= 0:
            _advise_dontneed(base_fd, tensor.data_offset, tensor.n_bytes)

        if (i + 1) % 25 == 0 or (i + 1) == len(reader.tensors):
            elapsed = time.time() - t_start
            print(f"[progress] {i+1}/{len(reader.tensors)} tensors "
                  f"({elapsed:.1f}s elapsed) — {tensor.name} -> {action}")

    writer.close()

    out_size = out.stat().st_size
    print(
        f"[done] wrote {out} ({out_size/1e9:.2f} GB) — "
        f"rotate_hadamard={counts['rotate_hadamard']} "
        f"rotate_kronecker={counts['rotate_kronecker']} "
        f"q8_0={counts['q8_0']} copy={counts['copy']} "
        f"({time.time()-t_start:.1f}s)"
    )
    return {"out_path": str(out), "out_size": out_size, **counts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True, help="Source BF16 GGUF")
    p.add_argument("--out", type=Path, required=True, help="Output GGUF path")
    p.add_argument("--rotation-seed", type=int, default=0,
                   help="Base seed for kronecker h_a matrices (default 0)")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for rotation matmuls (default cuda:0)")
    p.add_argument("--local-b", type=int, default=128,
                   help="block_hadamard block size along K (default 128)")
    p.add_argument("--max-b", type=int, default=1024,
                   help="kronecker factor_for_dim max_b (default 1024)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-tensor plan and exit without converting")
    p.add_argument("--chunk-rows", type=int, default=_CHUNK_ROWS,
                   help="Row-chunk size for GPU rotation/quantization work "
                        f"(default {_CHUNK_ROWS}) — bounds peak GPU memory "
                        "regardless of tensor size (e.g. the 1.27B-element "
                        "token_embd/output.weight); chunking along rows is "
                        "exact since rotation only mixes within a row's K")
    p.add_argument("--format", type=str, default="ml8_fp8",
                   choices=["ml8_fp8", "fp8_b128", "ml8_4"],
                   help="Output weight format (default ml8_fp8, byte-identical to "
                        "this script's original behaviour). fp8_b128: one fp16 "
                        "scale per aligned 128x128 tile (GGML_TYPE_FP8_B128); "
                        "weights not 128-aligned on both dims fall back to "
                        "unrotated Q8_0, and rotated weights that share an "
                        "input activation share one rotation (see "
                        "role_group_key/_group_seed). ml8_4: data-free "
                        "codebook quant (GGML_TYPE_ML8_4) — 4-bit centroid "
                        "index per weight + fp32 per-(row,group) scale over "
                        "64-element K-groups, with a per-K-group 16-centroid "
                        "e4m3 LUT (sidecar) fit on the rotated weight alone "
                        "(no calibration data); K not a multiple of 64 falls "
                        "back to unrotated Q8_0; same input-group rotation "
                        "sharing as fp8_b128.")
    p.add_argument("--scale-mode", type=str, default="tile",
                   choices=["tile", "channel"],
                   help="--format fp8_b128 only: how the per-block fp16 scale "
                        "is computed (default tile, today's behaviour: one "
                        "scale per aligned 128x128 tile). channel: one fp16 "
                        "scale per output row (row absmax over all K / 448), "
                        "replicated into every block of that row — same "
                        "on-disk block_fp8_b128 layout, just row-wide instead "
                        "of tile-wide scaling. Ignored for --format ml8_fp8/ml8_4.")
    p.add_argument("--ml8-fit-rows", type=int, default=_ML8_FIT_ROWS_DEFAULT,
                   help="--format ml8_4 only: number of rows to uniformly "
                        f"subsample per weight for the Lloyd-Max centroid fit "
                        f"(default {_ML8_FIT_ROWS_DEFAULT}) — bounds the fit "
                        "cost for huge weights (e.g. 248320-row output.weight); "
                        "every row is still individually scaled+assigned in "
                        "the full (chunked) second pass regardless of this.")
    args = p.parse_args()

    reader = gguf.GGUFReader(args.src)
    plan = build_plan(reader, args.rotation_seed, args.local_b, args.max_b, format=args.format)
    print_plan(plan)
    if args.dry_run:
        return

    convert(args.src, args.out, args.rotation_seed, args.device, args.local_b, args.max_b,
           chunk_rows=args.chunk_rows, format=args.format, ml8_fit_rows=args.ml8_fit_rows,
           scale_mode=args.scale_mode)


if __name__ == "__main__":
    main()
