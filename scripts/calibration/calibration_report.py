#!/usr/bin/env python3
"""calibration_report.py — pretty-print summary of a calibrate_ml8.py run.

Reads manifest.json from a calibration output directory and shows:
  - Per-layer SNR distribution (Y_SNR, W_SNR) with min/median/max
  - Per-linear-type aggregation (gate_proj vs up_proj vs down_proj)
  - Total quantization size + ratio vs fp16 baseline
  - PPL deltas if --eval-ppl was used
  - Outlier layers (worst SNR) so the user knows where quality drops

Usage:
    python3 scripts/calibration/calibration_report.py /tmp/ml8-qwen3-4b-full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def fmt_size_gb(bits: int) -> str:
    return f"{bits / 8 / 1e9:.2f} GB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calibration_dir", type=Path,
                   help="Output dir from calibrate_ml8.py (contains manifest.json)")
    p.add_argument("--top-n-worst", type=int, default=5,
                   help="Show N layers with the worst Y_SNR (default: 5).")
    args = p.parse_args()

    manifest_path = args.calibration_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    m = json.loads(manifest_path.read_text())
    results = m["results"]
    if not results:
        print("error: manifest has no results", file=sys.stderr)
        sys.exit(1)

    print(f"=== {m['model']}  ({len(results)} linears) ===")
    print()

    # Aggregate Y_SNR and W_SNR distributions
    y_snrs = [r["y_snr_db"] for r in results]
    w_snrs = [r["w_snr_db"] for r in results]
    print("SNR distribution (across all linears):")
    print(f"  Y_SNR (output, GPTQ-optimized):  "
          f"min={min(y_snrs):.2f}  median={statistics.median(y_snrs):.2f}  "
          f"max={max(y_snrs):.2f}  mean={statistics.mean(y_snrs):.2f}")
    print(f"  W_SNR (element-wise):            "
          f"min={min(w_snrs):.2f}  median={statistics.median(w_snrs):.2f}  "
          f"max={max(w_snrs):.2f}  mean={statistics.mean(w_snrs):.2f}")
    print()

    # Per-linear-type aggregation (gate_proj vs up_proj vs down_proj vs etc)
    by_kind = defaultdict(list)
    for r in results:
        # Take the trailing component, e.g. "model.layers.0.mlp.gate_proj" → "gate_proj"
        kind = r["name"].rsplit(".", 1)[-1]
        by_kind[kind].append(r)

    print("By linear type:")
    print(f"  {'kind':20s} {'count':>6s} {'Y_SNR med':>10s} {'Y_SNR min':>10s} {'numel total':>14s}")
    for kind in sorted(by_kind.keys()):
        rows = by_kind[kind]
        yk = [r["y_snr_db"] for r in rows]
        numel = sum(r["shape"][0] * r["shape"][1] for r in rows)
        print(f"  {kind:20s} {len(rows):>6d} {statistics.median(yk):>10.2f} "
              f"{min(yk):>10.2f} {numel:>14,d}")
    print()

    # Total size
    total_numel = sum(r["shape"][0] * r["shape"][1] for r in results)
    # We don't store bpv per layer; recompute. Assume 4-bit indices + fp16 scales + fp32 centroids
    # bpv = 4 + (16/group_size) + n_centroids*32/(rows*group_size)
    # For typical Qwen layers, ≈ 4.125 bpv
    # For a precise number we'd reload the .pt files; this is just a summary, ~4.125 is enough.
    approx_bpv = 4.125
    total_quant_bits = total_numel * approx_bpv
    total_fp16_bits = total_numel * 16
    print(f"Quantized weights size:")
    print(f"  total parameters quantized: {total_numel:,}")
    print(f"  ~{fmt_size_gb(int(total_quant_bits))} at ~{approx_bpv:.3f} bpv  "
          f"(vs {fmt_size_gb(total_fp16_bits)} at fp16)")
    print(f"  ratio: {total_quant_bits / total_fp16_bits:.3f} of fp16 size")
    print()

    # PPL if present
    if "ppl_baseline" in m and "ppl_quantized" in m:
        b = m["ppl_baseline"]["ppl"]
        q = m["ppl_quantized"]["ppl"]
        delta = q - b
        print(f"Perplexity (wikitext-2 test):")
        print(f"  baseline (f16):       {b:.4f}")
        print(f"  quantized (ml8-4):    {q:.4f}")
        print(f"  Δ_PPL:                {delta:+.4f}  ({delta/b*100:+.2f}%)")
        if delta < 0.08:
            print(f"  ✓ MAD-223 gate PASSED (< 0.08)")
        else:
            print(f"  ⚠ MAD-223 gate triggers AWQ B.5 (≥ 0.08)")
        print()

    # Worst layers
    print(f"Worst {args.top_n_worst} layers by Y_SNR:")
    worst = sorted(results, key=lambda r: r["y_snr_db"])[:args.top_n_worst]
    for r in worst:
        print(f"  {r['name']:50s}  Y_SNR={r['y_snr_db']:6.2f}  "
              f"W_SNR={r['w_snr_db']:6.2f}  shape={r['shape']}")


if __name__ == "__main__":
    main()
