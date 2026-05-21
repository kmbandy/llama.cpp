#!/usr/bin/env python3
"""
MAD-214 Phase 0: Fit FP8-constrained Lloyd-Max centroids to extracted KV samples.

Consumes the .npz files produced by extract_kv.py and computes the central
go/no-go metric: how well can N centroids constrained to E4M3-representable
values fit the actual KV distribution.

Variants evaluated (MAD-214 design):
  turbo3-FP8:  8 centroid magnitudes + sign bit, per-block FP32 scale (~5.0 bpv @ block=32)
  turbo4-FP8: 16 centroid magnitudes + sign bit, per-block FP32 scale (~6.0 bpv @ block=32)
  turbo5-FP8: 32 centroid magnitudes + sign bit, per-block FP32 scale (~7.0 bpv @ block=32)

Note: BPV math uses FP32 scale (32 bits / block_size = 1.0 bpv overhead at block=32).
Original MAD-214 ticket assumed FP8 scale (~0.25 bpv overhead), but per hipfire production
practice, scale must stay FP32 and multiply post-WMMA to avoid E4M3 saturation. See README.

Baselines for comparison:
  fp8_raw   : round-to-nearest in E4M3 lattice (the upper bound for FP8 KV)
  int4      : uniform 4-bit quant with per-block FP16 scale (matches q4_0)
  turbo3    : 8 unconstrained Lloyd-Max centroids in FP32 (existing turbo3)
  turbo4    : 16 unconstrained Lloyd-Max centroids in FP32 (existing turbo4)

Per-block scale model: blocks of 32 elements, scale = max(|values|) within block.
Normalized values v / scale ∈ [-1, 1], quantized via signed-magnitude codebook.

Output: JSON report with per-variant MSE (averaged across K and V, all layers,
all samples), plus per-layer breakdown. The go/no-go gate: turbo4-FP8 MSE
within 1.5x of fp8_raw MSE → proceed to Phase 1. >3x → revisit design.

Usage:
  python3 fit_centroids.py \\
      --input-dir /tmp/mad214_kv \\
      --output-report /tmp/mad214_fit_report.json \\
      --block-size 32 \\
      --subsample-tokens 4096
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# E4M3 lattice generation
# ---------------------------------------------------------------------------

def enumerate_e4m3_finite() -> np.ndarray:
    """
    Enumerate all finite E4M3 (1 sign, 4 exp, 3 mantissa) representable values.
    Bias=7. e=15,m=7 is reserved as NaN. Returns sorted unique fp32 values.
    """
    vals = []
    for s in (0, 1):
        for e in range(16):
            for m in range(8):
                if e == 15 and m == 7:
                    continue  # NaN
                if e == 0:
                    val = (-1) ** s * (2 ** -6) * (m / 8.0)  # subnormal
                else:
                    val = (-1) ** s * (2 ** (e - 7)) * (1 + m / 8.0)
                vals.append(val)
    arr = np.array(sorted(set(vals)), dtype=np.float64)
    return arr


def positive_e4m3_finite() -> np.ndarray:
    """Just the non-negative half of E4M3 — used for unsigned magnitude codebooks."""
    full = enumerate_e4m3_finite()
    return np.unique(full[full >= 0])


# ---------------------------------------------------------------------------
# Per-block scaling
# ---------------------------------------------------------------------------

def block_normalize(x: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape x to (..., n_blocks, block_size), compute per-block scale =
    max(|values|), and return (normalized, scale). Last dimension of x must be
    divisible by block_size.
    """
    last = x.shape[-1]
    assert last % block_size == 0, f"last dim {last} not divisible by block_size {block_size}"
    n_blocks = last // block_size
    blocked = x.reshape(*x.shape[:-1], n_blocks, block_size)
    scale = np.max(np.abs(blocked), axis=-1, keepdims=True)
    # Avoid div-by-zero
    safe = np.where(scale > 0, scale, 1.0)
    normalized = blocked / safe
    return normalized, scale  # both still blocked shape; squeeze later


def block_dequant(normalized: np.ndarray, scale: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Inverse of block_normalize — multiply by scale, reshape back."""
    blocked = normalized * scale
    return blocked.reshape(original_shape)


# ---------------------------------------------------------------------------
# Variant implementations
# ---------------------------------------------------------------------------

@dataclass
class QuantResult:
    name: str
    mse: float          # total mean squared error
    bits_per_value: float
    extra: dict


def quant_fp8_raw(x: np.ndarray, e4m3_lattice: np.ndarray) -> QuantResult:
    """Round each value to nearest E4M3 representable value. No block scale."""
    # Vectorized snap: for each x, argmin |x - lattice|
    # Reshape lattice for broadcasting
    flat = x.ravel().astype(np.float64)
    # Find nearest via searchsorted on sorted lattice
    idx = np.searchsorted(e4m3_lattice, flat)
    idx = np.clip(idx, 1, len(e4m3_lattice) - 1)
    left = e4m3_lattice[idx - 1]
    right = e4m3_lattice[idx]
    snapped = np.where(np.abs(flat - left) < np.abs(flat - right), left, right)
    snapped = snapped.reshape(x.shape)
    mse = float(np.mean((x - snapped) ** 2))
    return QuantResult("fp8_raw", mse, 8.0, {})


def quant_int4_blockwise(x: np.ndarray, block_size: int) -> QuantResult:
    """Uniform INT4 with per-block FP16 scale (matches q4_0)."""
    normalized, scale = block_normalize(x, block_size)
    # 4-bit signed: 16 levels in [-1, 1]. Centers at i/7.5 - 1 + 1/15 for i in 0..15
    # Simpler: levels at (2*i - 15) / 15 for i in 0..15
    levels = (2 * np.arange(16) - 15) / 15.0
    # Snap normalized to nearest level
    flat = normalized.ravel()
    idx = np.searchsorted(levels, flat)
    idx = np.clip(idx, 1, len(levels) - 1)
    left = levels[idx - 1]
    right = levels[idx]
    snapped = np.where(np.abs(flat - left) < np.abs(flat - right), left, right)
    snapped = snapped.reshape(normalized.shape)
    # Dequant: snapped * scale, reshape back
    deq = (snapped * scale).reshape(x.shape)
    mse = float(np.mean((x - deq) ** 2))
    # Bits per value: 4 (index) + 16 (scale) / block_size
    bpv = 4.0 + 16.0 / block_size
    return QuantResult("int4_q4_0", mse, bpv, {})


def fit_lloyd_max_signed(samples: np.ndarray, n_levels: int, n_iter: int = 50) -> np.ndarray:
    """
    Fit n_levels Lloyd-Max centroids over signed normalized values in [-1, 1].
    Returns sorted centroids (1D array of length n_levels).
    """
    # Init: uniformly spaced over [-1, 1]
    centroids = np.linspace(-1, 1, n_levels)
    for _ in range(n_iter):
        # Assign each sample to nearest centroid
        # Use searchsorted on sorted centroids midpoints
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        assign = np.searchsorted(midpoints, samples)
        # Update each centroid to mean of assigned samples
        new_centroids = centroids.copy()
        for k in range(n_levels):
            mask = assign == k
            if np.any(mask):
                new_centroids[k] = samples[mask].mean()
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = np.sort(new_centroids)
    return centroids


def fit_lloyd_max_unsigned_magnitudes(samples: np.ndarray, n_levels: int, n_iter: int = 50) -> np.ndarray:
    """
    Fit n_levels Lloyd-Max centroids over UNSIGNED magnitudes (folded from sign).
    Returns sorted positive centroids (1D, length n_levels).
    """
    mags = np.abs(samples)
    centroids = np.linspace(0, 1, n_levels + 1)[1:]  # avoid 0
    for _ in range(n_iter):
        midpoints = 0.5 * (centroids[:-1] + centroids[1:])
        assign = np.searchsorted(midpoints, mags)
        new_centroids = centroids.copy()
        for k in range(n_levels):
            mask = assign == k
            if np.any(mask):
                new_centroids[k] = mags[mask].mean()
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = np.sort(new_centroids)
    return centroids


def snap_to_lattice(values: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """For each value, find nearest in sorted lattice."""
    lattice = np.sort(lattice)
    idx = np.searchsorted(lattice, values)
    idx = np.clip(idx, 1, len(lattice) - 1)
    left = lattice[idx - 1]
    right = lattice[idx]
    return np.where(np.abs(values - left) < np.abs(values - right), left, right)


def quant_turbo_unconstrained(
    x: np.ndarray, n_magnitudes: int, block_size: int, name: str
) -> QuantResult:
    """
    Classical turbo3/turbo4: unsigned magnitudes via Lloyd-Max + sign bit +
    per-block FP16 scale. Centroids in FP32 (no E4M3 constraint).
    """
    normalized, scale = block_normalize(x, block_size)
    flat = normalized.ravel()
    # Sample for k-means speed
    samp_size = min(200_000, len(flat))
    samp = np.random.default_rng(0).choice(flat, size=samp_size, replace=False)
    mags_centroids = fit_lloyd_max_unsigned_magnitudes(samp, n_magnitudes)
    # Quantize: take abs, snap to nearest magnitude centroid, restore sign
    signs = np.sign(flat)
    mags = np.abs(flat)
    snapped_mags = snap_to_lattice(mags, mags_centroids)
    deq_flat = signs * snapped_mags
    deq = (deq_flat.reshape(normalized.shape) * scale).reshape(x.shape)
    mse = float(np.mean((x - deq) ** 2))
    # Bits: log2(n) magnitude index + 1 sign + 16 scale / block_size
    bpv = float(np.log2(n_magnitudes) + 1 + 16.0 / block_size)
    return QuantResult(name, mse, bpv, {"centroids": mags_centroids.tolist()})


def quant_turbo_fp8(
    x: np.ndarray, n_magnitudes: int, block_size: int, pos_lattice: np.ndarray, name: str
) -> QuantResult:
    """
    turbo3/4/5-FP8: unsigned magnitude centroids constrained to positive E4M3
    lattice + sign bit + per-block FP32 scale.

    Two-step optimization:
      1. Fit unconstrained Lloyd-Max magnitude centroids
      2. Snap each centroid to nearest positive E4M3 representable value
    Then quantize all values via the constrained centroids.
    """
    normalized, scale = block_normalize(x, block_size)
    flat = normalized.ravel()
    samp_size = min(200_000, len(flat))
    samp = np.random.default_rng(0).choice(flat, size=samp_size, replace=False)
    unconstrained = fit_lloyd_max_unsigned_magnitudes(samp, n_magnitudes)
    # Snap each unconstrained magnitude centroid to nearest positive E4M3 value
    pos_in_unit = pos_lattice[pos_lattice <= 1.0]  # only values in [0, 1] make sense for normalized
    constrained = np.array([snap_to_lattice(np.array([c]), pos_in_unit)[0] for c in unconstrained])
    constrained = np.unique(constrained)  # collapse duplicates
    # If snapping collapsed centroids (e.g., 16 centroids all map to a few unique E4M3 values),
    # we lose some quantization resolution. Track this in extra.
    n_unique_after_snap = len(constrained)

    signs = np.sign(flat)
    mags = np.abs(flat)
    snapped_mags = snap_to_lattice(mags, constrained)
    deq_flat = signs * snapped_mags
    deq = (deq_flat.reshape(normalized.shape) * scale).reshape(x.shape)
    mse = float(np.mean((x - deq) ** 2))
    # Bits: log2(n) magnitude index + 1 sign + 32 scale / block_size
    # (per-block scale stored as FP32 since it's post-WMMA multiplied, not folded into FP8)
    bpv = float(np.log2(n_magnitudes) + 1 + 32.0 / block_size)
    return QuantResult(
        name,
        mse,
        bpv,
        {
            "n_magnitudes_requested": n_magnitudes,
            "n_magnitudes_after_e4m3_snap": int(n_unique_after_snap),
            "centroids_unconstrained": unconstrained.tolist(),
            "centroids_constrained": constrained.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_kv_samples(input_dir: Path, subsample_tokens: int | None) -> dict:
    """
    Load all .npz files. Returns dict keyed by ('k' or 'v', layer_idx) → list of
    np arrays of shape (n_kv_heads, seq_len, head_dim).
    """
    manifest = json.loads((input_dir / "manifest.json").read_text())
    samples: dict[tuple[str, int], list[np.ndarray]] = {}
    for prompt_meta in manifest["prompts"]:
        prompt_dir = input_dir / prompt_meta["prompt_id"]
        for layer_meta in prompt_meta["layers"]:
            L = layer_meta["layer"]
            k_path = prompt_dir / f"layer_{L:03d}_k.npz"
            v_path = prompt_dir / f"layer_{L:03d}_v.npz"
            k = np.load(k_path)["k"]
            v = np.load(v_path)["v"]
            # k, v shape: (n_kv_heads, seq_len, head_dim)
            samples.setdefault(("k", L), []).append(k)
            samples.setdefault(("v", L), []).append(v)
    # Subsample to limit memory: keep at most subsample_tokens per (kv, layer)
    if subsample_tokens:
        rng = np.random.default_rng(0)
        out = {}
        for key, arrs in samples.items():
            cat = np.concatenate(arrs, axis=1)  # along token axis
            n_kv_heads, seq, head_dim = cat.shape
            total_tokens = seq
            if total_tokens > subsample_tokens:
                idx = rng.choice(total_tokens, size=subsample_tokens, replace=False)
                cat = cat[:, idx, :]
            out[key] = cat
        return out
    else:
        return {k: np.concatenate(v, axis=1) for k, v in samples.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--subsample-tokens", type=int, default=4096, help="Per (kv, layer) token cap")
    p.add_argument("--per-layer", action="store_true", help="Report per-layer MSE in addition to global")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    print(f"[fit_centroids] Loading KV samples from {in_dir}", flush=True)
    samples = load_kv_samples(in_dir, args.subsample_tokens)
    print(f"[fit_centroids] Loaded {len(samples)} (kv, layer) keys", flush=True)

    e4m3 = enumerate_e4m3_finite()
    pos_e4m3 = positive_e4m3_finite()
    print(
        f"[fit_centroids] E4M3 lattice: {len(e4m3)} signed values, {len(pos_e4m3)} positive",
        flush=True,
    )

    # Aggregate samples across all (kv, layer) for global fit
    all_x = np.concatenate([arr.ravel() for arr in samples.values()])
    print(f"[fit_centroids] Total elements for global fit: {all_x.size:,}", flush=True)

    # Reshape so the last dim is divisible by block_size
    # all_x is already flat; pad to multiple of block_size
    pad = (-all_x.size) % args.block_size
    if pad:
        all_x = np.concatenate([all_x, np.zeros(pad)])
    all_x = all_x.reshape(-1, args.block_size)

    variants: list[QuantResult] = []
    print("[fit_centroids] Running quantization variants...", flush=True)
    variants.append(quant_fp8_raw(all_x.astype(np.float32), e4m3))
    variants.append(quant_int4_blockwise(all_x, args.block_size))
    variants.append(quant_turbo_unconstrained(all_x, 8, args.block_size, "turbo3_fp16_scale"))
    variants.append(quant_turbo_unconstrained(all_x, 16, args.block_size, "turbo4_fp16_scale"))
    variants.append(quant_turbo_fp8(all_x, 8, args.block_size, pos_e4m3, "turbo3_fp8"))
    variants.append(quant_turbo_fp8(all_x, 16, args.block_size, pos_e4m3, "turbo4_fp8"))
    variants.append(quant_turbo_fp8(all_x, 32, args.block_size, pos_e4m3, "turbo5_fp8"))

    fp8_raw_mse = next(v.mse for v in variants if v.name == "fp8_raw")

    print("\n[fit_centroids] === RESULTS ===", flush=True)
    print(f"{'variant':<25} {'bpv':>6}  {'MSE':>14}  {'MSE / fp8_raw':>15}", flush=True)
    for v in variants:
        ratio = v.mse / fp8_raw_mse if fp8_raw_mse > 0 else float("nan")
        print(f"{v.name:<25} {v.bits_per_value:>6.2f}  {v.mse:>14.6e}  {ratio:>15.3f}", flush=True)

    # Go/no-go signal
    turbo4_fp8 = next(v for v in variants if v.name == "turbo4_fp8")
    ratio = turbo4_fp8.mse / fp8_raw_mse if fp8_raw_mse > 0 else float("inf")
    print(f"\n[fit_centroids] GO/NO-GO: turbo4_fp8 MSE / fp8_raw MSE = {ratio:.3f}", flush=True)
    if ratio <= 1.5:
        print("[fit_centroids] => PROCEED to Phase 1 (kernel implementation)", flush=True)
        verdict = "PROCEED"
    elif ratio <= 3.0:
        print("[fit_centroids] => MARGINAL — revisit design (per-head centroids? K/V split? Hadamard?)", flush=True)
        verdict = "MARGINAL"
    else:
        print("[fit_centroids] => STOP — fundamental codebook fit is bad, redesign needed", flush=True)
        verdict = "STOP"

    report = {
        "input_dir": str(in_dir),
        "block_size": args.block_size,
        "subsample_tokens": args.subsample_tokens,
        "fp8_raw_mse": fp8_raw_mse,
        "verdict": verdict,
        "turbo4_fp8_ratio": ratio,
        "variants": [
            {
                "name": v.name,
                "bits_per_value": v.bits_per_value,
                "mse": v.mse,
                "mse_over_fp8_raw": v.mse / fp8_raw_mse if fp8_raw_mse > 0 else float("nan"),
                "extra": v.extra,
            }
            for v in variants
        ],
    }
    Path(args.output_report).write_text(json.dumps(report, indent=2))
    print(f"\n[fit_centroids] Report written: {args.output_report}", flush=True)


if __name__ == "__main__":
    main()
