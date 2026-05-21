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

def block_normalize(
    x: np.ndarray, block_size: int, scale_dtype: str = "fp32"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape x to (..., n_blocks, block_size), compute per-block scale =
    max(|values|) (optionally rounded to FP16 precision), and return
    (normalized, scale). Last dimension of x must be divisible by block_size.

    scale_dtype: 'fp32' (default, full precision) or 'fp16' (simulates the
    lower-precision scale storage option that halves per-block overhead).
    """
    last = x.shape[-1]
    assert last % block_size == 0, f"last dim {last} not divisible by block_size {block_size}"
    n_blocks = last // block_size
    blocked = x.reshape(*x.shape[:-1], n_blocks, block_size)
    scale = np.max(np.abs(blocked), axis=-1, keepdims=True)
    if scale_dtype == "fp16":
        # Round-trip through fp16 to simulate the precision loss
        scale = scale.astype(np.float16).astype(np.float32)
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
    normalized, scale = block_normalize(x, block_size, scale_dtype="fp16")
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
    x: np.ndarray, n_magnitudes: int, block_size: int, name: str, scale_dtype: str = "fp16"
) -> QuantResult:
    """
    Classical turbo3/turbo4: unsigned magnitudes via Lloyd-Max + sign bit +
    per-block FP16 scale. Centroids in FP32 (no E4M3 constraint).
    """
    normalized, scale = block_normalize(x, block_size, scale_dtype=scale_dtype)
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
    scale_bits = 16 if scale_dtype == "fp16" else 32
    bpv = float(np.log2(n_magnitudes) + 1 + scale_bits / block_size)
    return QuantResult(name, mse, bpv, {"centroids": mags_centroids.tolist(), "scale_dtype": scale_dtype})


def quant_turbo_fp8(
    x: np.ndarray, n_magnitudes: int, block_size: int, pos_lattice: np.ndarray, name: str,
    scale_dtype: str = "fp32",
) -> QuantResult:
    """
    turbo3/4/5-FP8: unsigned magnitude centroids constrained to positive E4M3
    lattice + sign bit + per-block scale (FP32 default per hipfire safety; FP16
    optional for tighter memory at risk of saturation on extreme blocks).

    Two-step optimization:
      1. Fit unconstrained Lloyd-Max magnitude centroids
      2. Snap each centroid to nearest positive E4M3 representable value
    Then quantize all values via the constrained centroids.
    """
    normalized, scale = block_normalize(x, block_size, scale_dtype=scale_dtype)
    flat = normalized.ravel()
    samp_size = min(200_000, len(flat))
    samp = np.random.default_rng(0).choice(flat, size=samp_size, replace=False)
    unconstrained = fit_lloyd_max_unsigned_magnitudes(samp, n_magnitudes)
    # Snap each unconstrained magnitude centroid to nearest positive E4M3 value
    pos_in_unit = pos_lattice[pos_lattice <= 1.0]
    constrained = np.array([snap_to_lattice(np.array([c]), pos_in_unit)[0] for c in unconstrained])
    constrained = np.unique(constrained)
    n_unique_after_snap = len(constrained)

    signs = np.sign(flat)
    mags = np.abs(flat)
    snapped_mags = snap_to_lattice(mags, constrained)
    deq_flat = signs * snapped_mags
    deq = (deq_flat.reshape(normalized.shape) * scale).reshape(x.shape)
    mse = float(np.mean((x - deq) ** 2))
    scale_bits = 16 if scale_dtype == "fp16" else 32
    bpv = float(np.log2(n_magnitudes) + 1 + scale_bits / block_size)
    return QuantResult(
        name,
        mse,
        bpv,
        {
            "n_magnitudes_requested": n_magnitudes,
            "n_magnitudes_after_e4m3_snap": int(n_unique_after_snap),
            "centroids_unconstrained": unconstrained.tolist(),
            "centroids_constrained": constrained.tolist(),
            "scale_dtype": scale_dtype,
        },
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_kv_samples(
    input_dir: Path,
    subsample_tokens: int | None,
    hadamard_mode: str = "none",
) -> dict:
    """
    Stream-load .npz files with per-prompt subsampling AND optional per-prompt
    Walsh-Hadamard rotation along head_dim. Returns dict keyed by
    ('k' or 'v', layer_idx) → fp32 np array (n_kv_heads, total_subsampled, head_dim).

    Memory safety: peak memory is roughly one prompt's worth of array data
    (~tens of MB). All transforms happen per-prompt before discard. The
    Hadamard matrix is kept in fp32 to avoid fp32×fp64 → fp64 broadcasts that
    would double peak memory during matmul.

    hadamard_mode: 'none' | 'k_only' | 'k_and_v'. The math is QH @ (KH)^T = QK^T
    so any rotation here only models what would happen at inference if the same
    H were applied to Q.
    """
    manifest = json.loads((input_dir / "manifest.json").read_text())
    rng = np.random.default_rng(0)
    n_prompts = len(manifest["prompts"])
    if subsample_tokens:
        per_prompt_budget = max(1, subsample_tokens // n_prompts)
    else:
        per_prompt_budget = None

    # Lazy-init H once we know head_dim (after first array load).
    H: np.ndarray | None = None

    pieces: dict[tuple[str, int], list[np.ndarray]] = {}
    for prompt_meta in manifest["prompts"]:
        prompt_dir = input_dir / prompt_meta["prompt_id"]
        for layer_meta in prompt_meta["layers"]:
            L = layer_meta["layer"]
            for kv_label, fname in (("k", f"layer_{L:03d}_k.npz"), ("v", f"layer_{L:03d}_v.npz")):
                arr = np.load(prompt_dir / fname)[kv_label].astype(np.float32, copy=False)
                if per_prompt_budget and arr.shape[1] > per_prompt_budget:
                    idx = rng.choice(arr.shape[1], size=per_prompt_budget, replace=False)
                    arr = arr[:, idx, :]
                if hadamard_mode != "none":
                    do_rotate = (hadamard_mode == "k_and_v") or (hadamard_mode == "k_only" and kv_label == "k")
                    if do_rotate:
                        if H is None:
                            H = hadamard_matrix(arr.shape[-1]).astype(np.float32)
                        # fp32 @ fp32 → fp32; in-place semantics not possible but no fp64 inflation
                        arr = arr @ H
                pieces.setdefault((kv_label, L), []).append(arr)
    out: dict[tuple[str, int], np.ndarray] = {}
    for key, arrs in pieces.items():
        out[key] = np.concatenate(arrs, axis=1)
    return out


# ---------------------------------------------------------------------------
# Walsh-Hadamard rotation (outlier suppression)
# ---------------------------------------------------------------------------

def hadamard_matrix(n: int) -> np.ndarray:
    """
    Standard Hadamard matrix of size n (n must be a power of 2), normalized so
    H @ H.T == I. Built via Sylvester recursion.
    """
    assert n & (n - 1) == 0 and n > 0, f"n must be power of 2, got {n}"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float64)


def apply_hadamard(arr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply Hadamard rotation along the last dimension. arr: (..., head_dim)."""
    return arr @ H


# ---------------------------------------------------------------------------
# Granularity dispatch
# ---------------------------------------------------------------------------

VARIANT_SPECS = [
    # (display_name, callable(x_blocked, block_size, e4m3, pos_e4m3) -> QuantResult)
    ("fp8_raw",                  lambda x, bs, e, p: quant_fp8_raw(x.astype(np.float32), e)),
    ("int4_q4_0",                lambda x, bs, e, p: quant_int4_blockwise(x, bs)),
    ("turbo3_fp16_scale",        lambda x, bs, e, p: quant_turbo_unconstrained(x, 8,  bs, "turbo3_fp16_scale")),
    ("turbo4_fp16_scale",        lambda x, bs, e, p: quant_turbo_unconstrained(x, 16, bs, "turbo4_fp16_scale")),
    # FP32-scale FP8 variants (the safe path per hipfire — no E4M3 saturation risk)
    ("turbo3_fp8",               lambda x, bs, e, p: quant_turbo_fp8(x, 8,  bs, p, "turbo3_fp8",        scale_dtype="fp32")),
    ("turbo4_fp8",               lambda x, bs, e, p: quant_turbo_fp8(x, 16, bs, p, "turbo4_fp8",        scale_dtype="fp32")),
    ("turbo5_fp8",               lambda x, bs, e, p: quant_turbo_fp8(x, 32, bs, p, "turbo5_fp8",        scale_dtype="fp32")),
    # FP16-scale FP8 variants (tighter bpv, risk of saturation on extreme blocks)
    ("turbo3_fp8_fp16scale",     lambda x, bs, e, p: quant_turbo_fp8(x, 8,  bs, p, "turbo3_fp8_fp16scale", scale_dtype="fp16")),
    ("turbo4_fp8_fp16scale",     lambda x, bs, e, p: quant_turbo_fp8(x, 16, bs, p, "turbo4_fp8_fp16scale", scale_dtype="fp16")),
    ("turbo5_fp8_fp16scale",     lambda x, bs, e, p: quant_turbo_fp8(x, 32, bs, p, "turbo5_fp8_fp16scale", scale_dtype="fp16")),
]


def reshape_for_blocks(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Pad to multiple of block_size and reshape to (n_blocks, block_size)."""
    flat = arr.ravel()
    pad = (-flat.size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    return flat.reshape(-1, block_size)


def run_variants_global(
    samples: dict, block_size: int, e4m3: np.ndarray, pos_e4m3: np.ndarray
) -> list[QuantResult]:
    """One centroid set fit globally across all (kv, layer) elements."""
    all_x = np.concatenate([arr.ravel() for arr in samples.values()])
    print(f"[fit_centroids][global] Total elements: {all_x.size:,}", flush=True)
    x_blocked = reshape_for_blocks(all_x, block_size)
    return [spec[1](x_blocked, block_size, e4m3, pos_e4m3) for spec in VARIANT_SPECS]


def run_variants_per_key(
    samples: dict,
    block_size: int,
    e4m3: np.ndarray,
    pos_e4m3: np.ndarray,
    per_head: bool = False,
) -> tuple[list[QuantResult], dict]:
    """
    Fit centroids per key. With per_head=False, the key is (kv, layer) — one
    centroid set across all heads in that layer. With per_head=True, each
    (kv, layer) array is split along axis 0 (n_kv_heads) so we fit per
    (kv, layer, head).

    Returns (aggregated_results, per_key_breakdown). Aggregation is an
    element-count-weighted MSE across all keys.
    """
    granularity_tag = "per_kv_layer_head" if per_head else "per_kv_layer"

    # Build the working set of keys (and the array each refers to)
    if per_head:
        work_items: list[tuple[str, np.ndarray]] = []
        for (kv, L), arr in sorted(samples.items()):
            n_kv_heads = arr.shape[0]
            for h in range(n_kv_heads):
                # Keep 3D shape so reshape_for_blocks treats correctly
                work_items.append((f"{kv}_L{L}_H{h}", arr[h:h+1]))
    else:
        work_items = [(f"{kv}_L{L}", samples[(kv, L)]) for (kv, L) in sorted(samples.keys())]

    print(
        f"[fit_centroids][{granularity_tag}] Fitting per key across {len(work_items)} keys...",
        flush=True,
    )

    per_key_results: dict[str, list[tuple]] = {name: [] for name, _ in VARIANT_SPECS}
    centroid_log: dict[str, dict] = {name: {} for name, _ in VARIANT_SPECS}

    for key_label, arr in work_items:
        x_blocked = reshape_for_blocks(arr, block_size)
        n_elements = x_blocked.size
        for vname, vfunc in VARIANT_SPECS:
            qr = vfunc(x_blocked, block_size, e4m3, pos_e4m3)
            per_key_results[vname].append((key_label, qr.mse, n_elements))
            if "centroids_constrained" in qr.extra:
                centroid_log[vname][key_label] = qr.extra["centroids_constrained"]
            elif "centroids" in qr.extra:
                centroid_log[vname][key_label] = qr.extra["centroids"]
        # Only log progress for per-(kv,layer) to keep per-head output manageable
        if not per_head or key_label.endswith("_H0"):
            print(
                f"[fit_centroids][{granularity_tag}]   {key_label} done ({n_elements:,} elements)",
                flush=True,
            )

    aggregated: list[QuantResult] = []
    for vname, vfunc in VARIANT_SPECS:
        total_se = sum(mse * n for _, mse, n in per_key_results[vname])
        total_n = sum(n for _, _, n in per_key_results[vname])
        agg_mse = total_se / max(1, total_n)
        first_arr = work_items[0][1]
        first_blocked = reshape_for_blocks(first_arr, block_size)
        qr_ref = vfunc(first_blocked, block_size, e4m3, pos_e4m3)
        aggregated.append(
            QuantResult(
                name=vname,
                mse=agg_mse,
                bits_per_value=qr_ref.bits_per_value,
                extra={"per_key_mse": {k: m for k, m, _ in per_key_results[vname]}},
            )
        )
    return aggregated, centroid_log


def run_variants_per_kv_layer(samples, block_size, e4m3, pos_e4m3):
    """Backwards-compatible wrapper — delegates to run_variants_per_key."""
    return run_variants_per_key(samples, block_size, e4m3, pos_e4m3, per_head=False)


def _print_table(title: str, variants: list[QuantResult], fp8_raw_mse: float) -> None:
    print(f"\n[fit_centroids] === {title} ===", flush=True)
    print(f"{'variant':<25} {'bpv':>6}  {'MSE':>14}  {'MSE / fp8_raw':>15}", flush=True)
    for v in variants:
        ratio = v.mse / fp8_raw_mse if fp8_raw_mse > 0 else float("nan")
        print(f"{v.name:<25} {v.bits_per_value:>6.2f}  {v.mse:>14.6e}  {ratio:>15.3f}", flush=True)


def _verdict(turbo4_ratio: float) -> str:
    if turbo4_ratio <= 1.5:
        return "PROCEED"
    elif turbo4_ratio <= 3.0:
        return "MARGINAL"
    else:
        return "STOP"


def _variant_summary(variants: list[QuantResult], fp8_mse: float, include_per_key: bool) -> list[dict]:
    out = []
    for v in variants:
        entry = {
            "name": v.name,
            "bits_per_value": v.bits_per_value,
            "mse": v.mse,
            "mse_over_fp8_raw": v.mse / fp8_mse if fp8_mse > 0 else float("nan"),
        }
        if include_per_key:
            entry["per_key_mse"] = v.extra.get("per_key_mse", {})
        else:
            entry["extra"] = v.extra
        out.append(entry)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--subsample-tokens", type=int, default=4096, help="Per (kv, layer) token cap")
    p.add_argument(
        "--granularity",
        choices=["global", "per_kv_layer", "per_kv_layer_head", "both"],
        default="both",
        help="Centroid fitting granularity. 'per_kv_layer_head' splits each layer's "
             "KV array by head (one centroid set per (kv, layer, head)). 'both' runs "
             "global + per_kv_layer side-by-side.",
    )
    p.add_argument(
        "--hadamard",
        choices=["none", "k_only", "k_and_v"],
        default="none",
        help="Apply Walsh-Hadamard rotation along head_dim before quantization "
             "(outlier suppression per KVQuant/RotateKV/QuaRot). 'k_only' matches "
             "common practice since K has worse channel asymmetry than V.",
    )
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    print(f"[fit_centroids] Loading KV samples from {in_dir} (hadamard={args.hadamard})", flush=True)
    samples = load_kv_samples(in_dir, args.subsample_tokens, hadamard_mode=args.hadamard)
    print(f"[fit_centroids] Loaded {len(samples)} (kv, layer) keys", flush=True)

    e4m3 = enumerate_e4m3_finite()
    pos_e4m3 = positive_e4m3_finite()
    print(
        f"[fit_centroids] E4M3 lattice: {len(e4m3)} signed values, {len(pos_e4m3)} positive",
        flush=True,
    )

    report: dict = {
        "input_dir": str(in_dir),
        "block_size": args.block_size,
        "subsample_tokens": args.subsample_tokens,
        "granularity_mode": args.granularity,
        "hadamard": args.hadamard,
        "results": {},
    }
    summary: dict[str, dict] = {}

    if args.granularity in ("global", "both"):
        variants_g = run_variants_global(samples, args.block_size, e4m3, pos_e4m3)
        fp8_g = next(v.mse for v in variants_g if v.name == "fp8_raw")
        t4_g = next(v.mse for v in variants_g if v.name == "turbo4_fp8")
        ratio_g = t4_g / fp8_g if fp8_g > 0 else float("inf")
        _print_table("RESULTS — global centroids", variants_g, fp8_g)
        verdict_g = _verdict(ratio_g)
        print(f"[fit_centroids] global        turbo4_fp8 MSE / fp8_raw = {ratio_g:.3f}  => {verdict_g}", flush=True)
        summary["global"] = {"ratio": ratio_g, "verdict": verdict_g}
        report["results"]["global"] = {
            "fp8_raw_mse": fp8_g,
            "turbo4_fp8_ratio": ratio_g,
            "verdict": verdict_g,
            "variants": _variant_summary(variants_g, fp8_g, include_per_key=False),
        }

    if args.granularity in ("per_kv_layer", "both"):
        variants_p, centroid_log = run_variants_per_kv_layer(samples, args.block_size, e4m3, pos_e4m3)
        fp8_p = next(v.mse for v in variants_p if v.name == "fp8_raw")
        t4_p = next(v.mse for v in variants_p if v.name == "turbo4_fp8")
        ratio_p = t4_p / fp8_p if fp8_p > 0 else float("inf")
        _print_table("RESULTS — per (kv, layer) centroids", variants_p, fp8_p)
        verdict_p = _verdict(ratio_p)
        print(f"[fit_centroids] per_kv_layer  turbo4_fp8 MSE / fp8_raw = {ratio_p:.3f}  => {verdict_p}", flush=True)
        summary["per_kv_layer"] = {"ratio": ratio_p, "verdict": verdict_p}
        report["results"]["per_kv_layer"] = {
            "fp8_raw_mse": fp8_p,
            "turbo4_fp8_ratio": ratio_p,
            "verdict": verdict_p,
            "variants": _variant_summary(variants_p, fp8_p, include_per_key=True),
            "centroid_log": centroid_log,
        }

    if args.granularity == "per_kv_layer_head":
        variants_h, centroid_log_h = run_variants_per_key(
            samples, args.block_size, e4m3, pos_e4m3, per_head=True
        )
        fp8_h = next(v.mse for v in variants_h if v.name == "fp8_raw")
        t4_h = next(v.mse for v in variants_h if v.name == "turbo4_fp8")
        ratio_h = t4_h / fp8_h if fp8_h > 0 else float("inf")
        _print_table("RESULTS — per (kv, layer, head) centroids", variants_h, fp8_h)
        verdict_h = _verdict(ratio_h)
        print(f"[fit_centroids] per_kv_layer_head  turbo4_fp8 MSE / fp8_raw = {ratio_h:.3f}  => {verdict_h}", flush=True)
        summary["per_kv_layer_head"] = {"ratio": ratio_h, "verdict": verdict_h}
        report["results"]["per_kv_layer_head"] = {
            "fp8_raw_mse": fp8_h,
            "turbo4_fp8_ratio": ratio_h,
            "verdict": verdict_h,
            "variants": _variant_summary(variants_h, fp8_h, include_per_key=False),
        }

    # Final go/no-go: best ratio across granularities tried
    best_mode, best_info = min(summary.items(), key=lambda kv: kv[1]["ratio"])
    final = _verdict(best_info["ratio"])
    print(
        f"\n[fit_centroids] FINAL GO/NO-GO (best granularity={best_mode}, ratio={best_info['ratio']:.3f}): {final}",
        flush=True,
    )
    if final == "PROCEED":
        print("[fit_centroids] => PROCEED to Phase 1 (kernel implementation)", flush=True)
    elif final == "MARGINAL":
        print("[fit_centroids] => MARGINAL — revisit design (per-head centroids? K/V split? Hadamard?)", flush=True)
    else:
        print("[fit_centroids] => STOP — fundamental codebook fit is bad, redesign needed", flush=True)

    report["final_verdict"] = final
    report["best_granularity"] = best_mode
    report["best_ratio"] = best_info["ratio"]
    Path(args.output_report).write_text(json.dumps(report, indent=2))
    print(f"\n[fit_centroids] Report written: {args.output_report}", flush=True)


if __name__ == "__main__":
    main()
