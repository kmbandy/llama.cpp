#!/usr/bin/env python3
"""
MAD-214 Option F: fit turbo4-FP8 centroid LUTs from K_cur/V_cur dumps.

Inputs:
  --dump-dir DIR         dir containing l<N>_k.fp16 / l<N>_v.fp16 raw dumps
                         written by the MT_TURBO_FP8_DUMP_DIR hook in
                         mt_pagedattn_aiter.cu
  --head-size N          per-head dimension (256 for Qwen3.6 family)
  --n-kv-heads N         number of KV heads (2 for Qwen3.6-35B-A3B)
  --out-dir DIR          where to write l<N>_<k|v>.bin files (16 E4M3 bytes
                         each). Typically ~/.cache/llama.cpp/turbo-fp8/<fp>/
  --block-size N         BS=256 (one block per (token, kv_head) row)
  --hadamard             apply Walsh-Hadamard rotation along head_dim before
                         fitting (matches Phase 0; helps ~12% MSE on Qwen K)
  --n-iter N             Lloyd-Max iterations (default 30)

For each (layer, dir):
  1. Read raw fp16 dump → reshape to (n_tokens, n_kv_heads, head_size)
  2. Optionally apply Hadamard along head_size axis
  3. Block by (token, kv_head) → 256-element rows
  4. Per row: max-abs scale, normalize, take |abs|
  5. Pool all normalized magnitudes across all rows
  6. Lloyd-Max fit 16 centroids on the pooled distribution
  7. Snap each centroid to the nearest E4M3 byte (one of 128 valid magnitudes)
  8. Sort ascending, write 16 bytes to <out-dir>/l<layer>_<dir>.bin
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# ── E4M3 byte table (the 128 valid non-sign magnitudes, sorted ascending) ──
def e4m3_byte_to_float(b: int) -> float:
    """Decode an E4M3 byte (sign|4e|3m) to fp32. b=255 is NaN; we exclude it."""
    sign = (b >> 7) & 1
    e    = (b >> 3) & 0xF
    m    = b & 0x7
    if e == 15 and m == 7:
        return float("nan")
    if e == 0:
        v = (1.0 / 64.0) * (m / 8.0)
    else:
        v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -v if sign else v


def build_e4m3_magnitude_table() -> tuple[np.ndarray, np.ndarray]:
    """Returns (magnitudes, bytes) sorted ascending by magnitude. Only the
    positive-sign-bit half (bytes 0..127), since we encode signs separately."""
    bytes_arr = []
    mags_arr  = []
    for b in range(128):  # sign=0 → positive magnitudes
        v = e4m3_byte_to_float(b)
        if not np.isnan(v) and not np.isinf(v):
            bytes_arr.append(b)
            mags_arr.append(v)
    order = np.argsort(mags_arr)
    return np.array(mags_arr)[order], np.array(bytes_arr, dtype=np.uint8)[order]


def hadamard_matrix(n: int) -> np.ndarray:
    """Walsh-Hadamard matrix of size n (n must be power of 2). Normalized."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be power of 2"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def lloyd_max_unsigned(samples: np.ndarray, n_levels: int, n_iter: int = 30) -> np.ndarray:
    """Fit n_levels centroids on the magnitude distribution of `samples`.
    Returns sorted centroids in [0, max(samples)]."""
    mags = np.abs(samples).astype(np.float64)
    # Filter NaN/Inf (some K_cur rows have NaN padding from upstream)
    mags = mags[np.isfinite(mags)]
    mags = mags[mags > 0]  # drop zeros
    if len(mags) == 0:
        return np.linspace(0, 1, n_levels)
    # Initialize centroids at equal-quantile points
    centroids = np.quantile(mags, np.linspace(0.0, 1.0, n_levels))
    for _ in range(n_iter):
        # Assign each sample to nearest centroid
        edges = (centroids[:-1] + centroids[1:]) / 2.0
        bins = np.searchsorted(edges, mags)  # 0..n_levels-1
        # Move each centroid to mean of its assigned samples
        new = np.zeros_like(centroids)
        for k in range(n_levels):
            mask = bins == k
            new[k] = mags[mask].mean() if mask.any() else centroids[k]
        if np.allclose(new, centroids, rtol=1e-6):
            centroids = new
            break
        centroids = new
    return np.sort(centroids)


def snap_to_e4m3(centroids: np.ndarray, e4m3_mags: np.ndarray, e4m3_bytes: np.ndarray) -> np.ndarray:
    """For each centroid, find the nearest E4M3 magnitude byte, enforcing
    distinct bytes (no duplicates). For peaked distributions, Lloyd-Max
    will pile centroids near zero; without distinctness, ~6/16 collapse
    to 0x00 (=0.0) and we lose encoding capacity. With distinctness, the
    excess "zero-prone" centroids get pushed to the next-available
    non-zero E4M3 bytes (0x01, 0x02, …) so each index encodes a unique
    magnitude.

    Algorithm: assign in nearest-first order, marking used bytes; when
    a centroid's first choice is taken, walk outward to the next unused
    E4M3 byte.
    """
    n_lvls = len(centroids)
    used = np.zeros(len(e4m3_mags), dtype=bool)
    out = np.zeros(n_lvls, dtype=np.uint8)
    # Sort indices by how decisive the centroid's "nearest" choice is
    # (centroids far from anything get first pick to minimize displacement).
    centroid_distances = np.abs(centroids[:, None] - e4m3_mags[None, :])
    nearest_dist = np.min(centroid_distances, axis=1)
    order = np.argsort(-nearest_dist)  # most decisive first
    for i in order:
        # Walk the E4M3 magnitudes ordered by proximity, take the first unused.
        proximity_order = np.argsort(centroid_distances[i])
        for j in proximity_order:
            if not used[j]:
                out[i] = e4m3_bytes[j]
                used[j] = True
                break
    # Sort the final LUT ascending so the kernel decode pattern is consistent.
    out_mags = np.array([float(e4m3_mags[np.where(e4m3_bytes == b)[0][0]]) for b in out])
    return out[np.argsort(out_mags)]


def fit_one(samples: np.ndarray, head_size: int, n_kv_heads: int,
            apply_hadamard: bool, H: np.ndarray | None, n_iter: int,
            e4m3_mags: np.ndarray, e4m3_bytes: np.ndarray) -> tuple[np.ndarray, dict]:
    """samples: 1D fp32 array. Reshape into (n_tokens, n_kv_heads, head_size),
    optionally rotate by Hadamard along head_size, then block-normalize each
    (token, kv_head) row by max-abs, pool all magnitudes, fit 16 centroids,
    snap to E4M3."""
    elts_per_row = head_size  # one block per (token, kv_head) for BS=256
    total = samples.size
    assert total % (n_kv_heads * head_size) == 0, \
        f"sample count {total} not divisible by n_kv_heads*head_size={n_kv_heads*head_size}"
    n_tokens = total // (n_kv_heads * head_size)
    x = samples.reshape(n_tokens, n_kv_heads, head_size).astype(np.float32)

    if apply_hadamard:
        # Rotate each (token, kv_head) row by H along head_size axis
        x = np.einsum("ij,...j->...i", H, x)

    # Replace NaN/Inf with 0 (some rows have NaN padding from upstream).
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    # Block-normalize each (token, kv_head) row by max-abs (matches kernel scale).
    abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
    abs_max = np.where(abs_max > 0, abs_max, 1.0)
    normalized = x / abs_max  # in [-1, 1]
    mags = np.abs(normalized).flatten()

    centroids = lloyd_max_unsigned(mags, n_levels=16, n_iter=n_iter)
    lut_bytes = snap_to_e4m3(centroids, e4m3_mags, e4m3_bytes)

    return lut_bytes, {
        "n_tokens":          int(n_tokens),
        "centroids":         centroids.tolist(),
        "snapped_mags":      [float(e4m3_mags[np.where(e4m3_bytes == b)[0][0]]) for b in lut_bytes],
        "snap_mse_pct":      float(np.mean((centroids - np.array(
            [e4m3_mags[np.where(e4m3_bytes == b)[0][0]] for b in lut_bytes]))**2)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir",   required=True, type=Path)
    ap.add_argument("--head-size",  required=True, type=int)
    ap.add_argument("--n-kv-heads", required=True, type=int)
    ap.add_argument("--out-dir",    required=True, type=Path)
    ap.add_argument("--block-size", default=256, type=int)
    ap.add_argument("--hadamard",   action="store_true")
    ap.add_argument("--n-iter",     default=30, type=int)
    args = ap.parse_args()

    assert args.block_size == 256, "only BS=256 wired in this script"
    assert args.block_size == args.head_size, "BS=256 → head_size must be 256"

    e4m3_mags, e4m3_bytes = build_e4m3_magnitude_table()
    print(f"E4M3 magnitude table: {len(e4m3_mags)} unique non-NaN values "
          f"in [{e4m3_mags.min():.4g}, {e4m3_mags.max():.4g}]", file=sys.stderr)

    H = hadamard_matrix(args.head_size) if args.hadamard else None
    if H is not None:
        print(f"Hadamard rotation enabled (head_size={args.head_size})", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Find all l<N>_<k|v>.fp16 files
    dumps = sorted(args.dump_dir.glob("l*_*.fp16"))
    if not dumps:
        print(f"ERROR: no l*_*.fp16 dumps found in {args.dump_dir}", file=sys.stderr)
        return 1

    fitted = 0
    for path in dumps:
        # Parse layer + dir from filename
        stem = path.stem  # e.g. "l3_k"
        layer_part, dir_part = stem.split("_")
        layer = int(layer_part[1:])
        dir_letter = dir_part  # "k" or "v"

        raw = np.fromfile(path, dtype=np.float16)
        if raw.size == 0:
            print(f"  l{layer:2d}_{dir_letter}: SKIP (empty dump)", file=sys.stderr)
            continue

        lut_bytes, stats = fit_one(
            raw, args.head_size, args.n_kv_heads,
            args.hadamard, H, args.n_iter,
            e4m3_mags, e4m3_bytes,
        )
        out_path = args.out_dir / f"l{layer}_{dir_letter}.bin"
        out_path.write_bytes(lut_bytes.tobytes())

        lut_hex = " ".join(f"{b:02x}" for b in lut_bytes)
        print(f"  l{layer:2d}_{dir_letter}: {stats['n_tokens']:5d} tokens fit → "
              f"{out_path.name} ({lut_hex})", file=sys.stderr)
        fitted += 1

    print(f"\nWrote {fitted} LUT files to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
