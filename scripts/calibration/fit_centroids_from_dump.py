#!/usr/bin/env python3
"""
MAD-214 Option F + calibration matrix sweep: fit turbo4-FP8 centroid LUTs
from K_cur/V_cur dumps.

Supports the three orthogonal levers from TURBO_FP8_CALIBRATION_DESIGN.md:
  --fit-loss       {mse, mag_weighted, log_space}                       (lever #2)
  --granularity    {per_layer_dir, per_dir, global}                     (lever #4)
  --snap-strategy  {first_fit, distinct, greedy_coverage, forced_anchors} (lever #7)

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
  --fit-loss MODE        mse (default) | mag_weighted | log_space
  --mag-weight-p P       exponent for mag_weighted loss (default 1.0)
  --granularity MODE     per_layer_dir (default) | per_dir | global
  --snap-strategy MODE   distinct (default) | first_fit | greedy_coverage | forced_anchors
  --forced-anchors HEXS  comma-separated E4M3 bytes always included when
                         --snap-strategy=forced_anchors (default "0x00,0x38"
                         = true-zero + 1.0; remaining 14 fitted in between)

Output: one 16-byte E4M3 LUT per (layer, dir) at <out-dir>/l<layer>_<dir>.bin.
Under per_dir granularity the same bytes are written for every layer of that
direction; under global every file gets the same bytes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ── E4M3 byte table (128 valid non-sign magnitudes, sorted ascending) ──
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
    """Returns (magnitudes, bytes) sorted ascending by magnitude. Positive
    sign-bit half only (bytes 0..127), since signs are encoded separately."""
    bytes_arr, mags_arr = [], []
    for b in range(128):
        v = e4m3_byte_to_float(b)
        if not np.isnan(v) and not np.isinf(v):
            bytes_arr.append(b)
            mags_arr.append(v)
    order = np.argsort(mags_arr)
    return np.array(mags_arr)[order], np.array(bytes_arr, dtype=np.uint8)[order]


def hadamard_matrix(n: int) -> np.ndarray:
    """Walsh-Hadamard matrix of size n (must be power of 2). Normalized."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be power of 2"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


# ─────────────────────────────── Lloyd-Max ───────────────────────────────
def lloyd_max_unsigned(samples: np.ndarray, n_levels: int, n_iter: int,
                       fit_loss: str, mag_weight_p: float) -> np.ndarray:
    """Fit n_levels centroids on the magnitude distribution of `samples`.

    fit_loss:
      "mse"          — classical Lloyd-Max; centroid = mean of assigned samples.
      "mag_weighted" — weighted mean using w_i = |x_i|^mag_weight_p. Biases
                       centroids toward larger magnitudes (where attention
                       actually has signal).
      "log_space"    — Lloyd-Max in log domain; geometric centroid spacing.
                       Naturally covers orders of magnitude evenly.
    """
    mags = np.abs(samples).astype(np.float64)
    mags = mags[np.isfinite(mags)]
    mags = mags[mags > 0]
    if len(mags) == 0:
        return np.linspace(0, 1, n_levels)

    if fit_loss == "log_space":
        log_mags = np.log(mags)
        centroids = np.quantile(log_mags, np.linspace(0.0, 1.0, n_levels))
        for _ in range(n_iter):
            edges = (centroids[:-1] + centroids[1:]) / 2.0
            bins = np.searchsorted(edges, log_mags)
            new = np.zeros_like(centroids)
            for k in range(n_levels):
                mask = bins == k
                new[k] = log_mags[mask].mean() if mask.any() else centroids[k]
            if np.allclose(new, centroids, rtol=1e-6):
                centroids = new
                break
            centroids = new
        return np.sort(np.exp(centroids))

    # mse or mag_weighted: linear-domain Lloyd-Max with optional weights
    centroids = np.quantile(mags, np.linspace(0.0, 1.0, n_levels))
    weights = (mags ** mag_weight_p) if fit_loss == "mag_weighted" else None
    for _ in range(n_iter):
        edges = (centroids[:-1] + centroids[1:]) / 2.0
        bins = np.searchsorted(edges, mags)
        new = np.zeros_like(centroids)
        for k in range(n_levels):
            mask = bins == k
            if not mask.any():
                new[k] = centroids[k]
                continue
            if weights is None:
                new[k] = mags[mask].mean()
            else:
                w = weights[mask]
                new[k] = (w * mags[mask]).sum() / max(w.sum(), 1e-30)
        if np.allclose(new, centroids, rtol=1e-6):
            centroids = new
            break
        centroids = new
    return np.sort(centroids)


# ───────────────────────────── E4M3 snap strategies ─────────────────────────────
def _snap_first_fit(centroids, e4m3_mags, e4m3_bytes):
    """Original: each centroid → nearest E4M3 byte. Duplicates allowed (and
    common for peaked distributions, where ~6/16 collapse to 0x00)."""
    idx = np.argmin(np.abs(centroids[:, None] - e4m3_mags[None, :]), axis=1)
    return e4m3_bytes[idx]


def _snap_distinct(centroids, e4m3_mags, e4m3_bytes):
    """Force all 16 bytes distinct. Assign most-decisive centroid first, walk
    outward when its first choice is already taken."""
    n_lvls = len(centroids)
    used = np.zeros(len(e4m3_mags), dtype=bool)
    out  = np.zeros(n_lvls, dtype=np.uint8)
    dist = np.abs(centroids[:, None] - e4m3_mags[None, :])
    order = np.argsort(-np.min(dist, axis=1))  # most decisive first
    for i in order:
        for j in np.argsort(dist[i]):
            if not used[j]:
                out[i] = e4m3_bytes[j]
                used[j] = True
                break
    return out


def _snap_greedy_coverage(centroids, e4m3_mags, e4m3_bytes):
    """Maximize magnitude range covered: sort centroids, snap to evenly-spaced
    E4M3 bytes spanning [c_min, c_max]. Gives up local precision in dense
    regions to ensure the magnitude tail is encodable."""
    n_lvls = len(centroids)
    c_sorted = np.sort(centroids)
    targets = np.linspace(c_sorted[0], c_sorted[-1], n_lvls)
    used = np.zeros(len(e4m3_mags), dtype=bool)
    out  = np.zeros(n_lvls, dtype=np.uint8)
    for i, t in enumerate(targets):
        for j in np.argsort(np.abs(t - e4m3_mags)):
            if not used[j]:
                out[i] = e4m3_bytes[j]
                used[j] = True
                break
    return out


def _snap_forced_anchors(centroids, e4m3_mags, e4m3_bytes, anchor_bytes):
    """Always include the anchor bytes (e.g. 0x00 = true zero, 0x38 = 1.0);
    distinct-snap the remaining n_levels - len(anchors) centroids around them.
    Useful when known structural values (zero, unit-max) should be encodable
    exactly regardless of how the fitter places its centroids."""
    n_lvls = len(centroids)
    n_anchors = len(anchor_bytes)
    assert n_anchors < n_lvls, f"need < {n_lvls} anchors, got {n_anchors}"

    used = np.zeros(len(e4m3_mags), dtype=bool)
    for ab in anchor_bytes:
        idx = np.where(e4m3_bytes == ab)[0]
        if len(idx) == 0:
            raise ValueError(f"anchor byte 0x{ab:02x} not in E4M3 table")
        used[idx[0]] = True

    out = np.zeros(n_lvls, dtype=np.uint8)
    for i, ab in enumerate(anchor_bytes):
        out[i] = ab

    # Demote centroids that are "covered" by an anchor; pick the rest as the
    # most-novel ones to distinct-snap into the remaining E4M3 slots.
    anchor_mags = np.array([e4m3_byte_to_float(int(ab)) for ab in anchor_bytes])
    novelty = np.min(np.abs(centroids[:, None] - anchor_mags[None, :]), axis=1)
    pick_idx = np.argsort(-novelty)[: n_lvls - n_anchors]
    picked = centroids[pick_idx]

    dist = np.abs(picked[:, None] - e4m3_mags[None, :])
    order = np.argsort(-np.min(dist, axis=1))
    write_idx = n_anchors
    for i in order:
        for j in np.argsort(dist[i]):
            if not used[j]:
                out[write_idx] = e4m3_bytes[j]
                used[j] = True
                write_idx += 1
                break
    return out


def snap_to_e4m3(centroids: np.ndarray, e4m3_mags: np.ndarray,
                 e4m3_bytes: np.ndarray, strategy: str,
                 forced_anchors: list[int] | None) -> np.ndarray:
    if strategy == "first_fit":
        out = _snap_first_fit(centroids, e4m3_mags, e4m3_bytes)
    elif strategy == "distinct":
        out = _snap_distinct(centroids, e4m3_mags, e4m3_bytes)
    elif strategy == "greedy_coverage":
        out = _snap_greedy_coverage(centroids, e4m3_mags, e4m3_bytes)
    elif strategy == "forced_anchors":
        anchors = forced_anchors if forced_anchors is not None else [0x00, 0x38]
        out = _snap_forced_anchors(centroids, e4m3_mags, e4m3_bytes, anchors)
    else:
        raise ValueError(f"unknown snap strategy: {strategy}")
    # Sort ascending so the kernel decode pattern is consistent across cells.
    out_mags = np.array([float(e4m3_mags[np.where(e4m3_bytes == b)[0][0]]) for b in out])
    return out[np.argsort(out_mags)]


# ─────────────────────────── Sample preprocessing ───────────────────────────
def preprocess_samples(samples: np.ndarray, head_size: int, n_kv_heads: int,
                       apply_hadamard: bool, H: np.ndarray | None
                       ) -> tuple[np.ndarray, int]:
    """1D fp32 → flat normalized magnitudes in [0,1] from per-(token, kv_head)
    max-abs scaling, with NaN/Inf filtered. Returns (mags, n_tokens)."""
    total = samples.size
    assert total % (n_kv_heads * head_size) == 0, \
        f"sample count {total} not divisible by n_kv_heads*head_size={n_kv_heads*head_size}"
    n_tokens = total // (n_kv_heads * head_size)
    x = samples.reshape(n_tokens, n_kv_heads, head_size).astype(np.float32)
    if apply_hadamard:
        x = np.einsum("ij,...j->...i", H, x)
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
    abs_max = np.where(abs_max > 0, abs_max, 1.0)
    normalized = x / abs_max
    return np.abs(normalized).flatten(), int(n_tokens)


def fit_pool(mags: np.ndarray, n_iter: int, fit_loss: str, mag_weight_p: float,
             snap_strategy: str, forced_anchors: list[int] | None,
             e4m3_mags: np.ndarray, e4m3_bytes: np.ndarray) -> np.ndarray:
    centroids = lloyd_max_unsigned(mags, n_levels=16, n_iter=n_iter,
                                   fit_loss=fit_loss, mag_weight_p=mag_weight_p)
    return snap_to_e4m3(centroids, e4m3_mags, e4m3_bytes,
                        strategy=snap_strategy, forced_anchors=forced_anchors)


def parse_anchors(s: str) -> list[int]:
    return [int(x, 0) for x in s.split(",")]


# ─────────────────────────────── Main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dump-dir",       required=True, type=Path)
    ap.add_argument("--head-size",      required=True, type=int)
    ap.add_argument("--n-kv-heads",     required=True, type=int)
    ap.add_argument("--out-dir",        required=True, type=Path)
    ap.add_argument("--block-size",     default=256, type=int)
    ap.add_argument("--hadamard",       action="store_true")
    ap.add_argument("--n-iter",         default=30, type=int)
    ap.add_argument("--fit-loss",       default="mse",
                    choices=["mse", "mag_weighted", "log_space"])
    ap.add_argument("--mag-weight-p",   default=1.0, type=float,
                    help="exponent p in w_i = |x_i|^p (mag_weighted loss only)")
    ap.add_argument("--granularity",    default="per_layer_dir",
                    choices=["per_layer_dir", "per_dir", "global"])
    ap.add_argument("--snap-strategy",  default="distinct",
                    choices=["first_fit", "distinct", "greedy_coverage", "forced_anchors"])
    ap.add_argument("--forced-anchors", default="0x00,0x38", type=parse_anchors,
                    help='comma-separated hex bytes, e.g. "0x00,0x38" '
                         '(used only when --snap-strategy=forced_anchors)')
    args = ap.parse_args()

    assert args.block_size == 256, "only BS=256 wired in this script"
    assert args.block_size == args.head_size, "BS=256 → head_size must be 256"

    e4m3_mags, e4m3_bytes = build_e4m3_magnitude_table()
    print(f"E4M3 table: {len(e4m3_mags)} non-NaN magnitudes in "
          f"[{e4m3_mags.min():.4g}, {e4m3_mags.max():.4g}]", file=sys.stderr)

    cfg = f"fit_loss={args.fit_loss}"
    if args.fit_loss == "mag_weighted":
        cfg += f" (p={args.mag_weight_p})"
    cfg += f", granularity={args.granularity}, snap={args.snap_strategy}"
    if args.snap_strategy == "forced_anchors":
        anchor_hex = ",".join(f"0x{b:02x}" for b in args.forced_anchors)
        cfg += f" (anchors={anchor_hex})"
    print(f"Config: {cfg}", file=sys.stderr)

    H = hadamard_matrix(args.head_size) if args.hadamard else None
    if H is not None:
        print(f"Hadamard rotation enabled (head_size={args.head_size})", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dumps = sorted(args.dump_dir.glob("l*_*.fp16"))
    if not dumps:
        print(f"ERROR: no l*_*.fp16 dumps found in {args.dump_dir}", file=sys.stderr)
        return 1

    # Pass 1: preprocess each dump into normalized magnitudes, keyed by (layer, dir).
    per_pair_mags: dict[tuple[int, str], np.ndarray] = {}
    for path in dumps:
        layer_part, dir_letter = path.stem.split("_")
        layer = int(layer_part[1:])
        raw = np.fromfile(path, dtype=np.float16)
        if raw.size == 0:
            print(f"  l{layer:2d}_{dir_letter}: SKIP (empty dump)", file=sys.stderr)
            continue
        mags, n_tokens = preprocess_samples(
            raw, args.head_size, args.n_kv_heads, args.hadamard, H
        )
        per_pair_mags[(layer, dir_letter)] = mags
        print(f"  l{layer:2d}_{dir_letter}: {n_tokens:5d} tokens", file=sys.stderr)

    # Pool according to granularity.
    if args.granularity == "per_layer_dir":
        pools = dict(per_pair_mags)
    elif args.granularity == "per_dir":
        bucket: dict[tuple[str, str], list[np.ndarray]] = {}
        for (_, d), m in per_pair_mags.items():
            bucket.setdefault(("*", d), []).append(m)
        pools = {k: np.concatenate(v) for k, v in bucket.items()}
    else:  # global
        pools = {("*", "*"): np.concatenate(list(per_pair_mags.values()))}

    # Fit one LUT per pool.
    fitted: dict[tuple, np.ndarray] = {}
    for key, mags in pools.items():
        lut = fit_pool(mags, args.n_iter, args.fit_loss, args.mag_weight_p,
                       args.snap_strategy, args.forced_anchors,
                       e4m3_mags, e4m3_bytes)
        fitted[key] = lut
        print(f"  pool {key}: {' '.join(f'{b:02x}' for b in lut)}", file=sys.stderr)

    # Expand pools out to per-(layer, dir) files.
    written = 0
    for (layer, dir_letter) in per_pair_mags.keys():
        if args.granularity == "per_layer_dir":
            lut = fitted[(layer, dir_letter)]
        elif args.granularity == "per_dir":
            lut = fitted[("*", dir_letter)]
        else:
            lut = fitted[("*", "*")]
        (args.out_dir / f"l{layer}_{dir_letter}.bin").write_bytes(lut.tobytes())
        written += 1

    print(f"\nWrote {written} LUT files to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
