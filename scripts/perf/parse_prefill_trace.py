#!/usr/bin/env python3
"""Parse a rocprofv3 --hip-trace kernel_trace CSV into a kernel-time breakdown.

Determines the prefill window by the first→last dispatch of the tile kernel,
then aggregates GPU time by kernel name within that window. Outputs a JSON
summary suitable for storing in tests/perf-baseline/.

Usage:
    parse_prefill_trace.py <kernel_trace.csv> [--tile-kernel NAME]
                          [--out baseline.json] [--ctx-tokens N]
                          [--prefill-ms M] [--prefill-tps T]
                          [--commit SHA] [--label LABEL]

If --out is given, writes JSON baseline. Always prints a text summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# How to canonicalize various kernel name patterns to a stable short label.
NAME_RULES = [
    # (substring match, canonical name)
    ("mt_paged_attention_tile_mw_kernel",      "mt_paged_attention_tile_mw_kernel"),
    ("mt_paged_attention_tile_kernel",         "mt_paged_attention_tile_kernel"),
    ("mt_paged_attention_kernel",              "mt_paged_attention_kernel (scalar)"),
    ("scatter_kv",                             "launch_scatter_kv_*"),
    ("Cijk_",                                  "Cijk_* (rocBLAS Tensile GEMM)"),
]


def canonical(name: str) -> str:
    for needle, label in NAME_RULES:
        if needle in name:
            return label
    return name.split("<")[0][:55]


def parse(path: Path, tile_kernel: str) -> dict:
    tile_first = None
    tile_last = None
    with path.open() as f:
        for row in csv.DictReader(f):
            if tile_kernel in row["Kernel_Name"]:
                t0 = int(row["Start_Timestamp"])
                t1 = int(row["End_Timestamp"])
                if tile_first is None or t0 < tile_first:
                    tile_first = t0
                if tile_last is None or t1 > tile_last:
                    tile_last = t1
    if tile_first is None:
        raise SystemExit(
            f"no dispatches of '{tile_kernel}' found in {path} — "
            f"check kernel name or whether the tile path was actually exercised"
        )

    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    with path.open() as f:
        for row in csv.DictReader(f):
            end = int(row["End_Timestamp"])
            if end > tile_last:
                continue
            short = canonical(row["Kernel_Name"])
            dur = end - int(row["Start_Timestamp"])
            totals[short] += dur
            counts[short] += 1

    total_gpu_ns = sum(totals.values())
    span_ns = tile_last - tile_first

    items = []
    for name, t in sorted(totals.items(), key=lambda x: -x[1]):
        items.append({
            "kernel": name,
            "calls": counts[name],
            "gpu_ms": round(t / 1e6, 2),
            "gpu_pct": round(t / total_gpu_ns * 100, 2),
            "us_per_call": round(t / counts[name] / 1000, 2),
        })

    return {
        "prefill_window_s": round(span_ns / 1e9, 4),
        "gpu_active_ms": round(total_gpu_ns / 1e6, 1),
        "gpu_util_pct": round(total_gpu_ns / span_ns * 100, 2),
        "kernels": items,
    }


def fmt(summary: dict) -> str:
    out = []
    out.append(
        f"=== prefill window: {summary['prefill_window_s']}s wall, "
        f"{summary['gpu_active_ms']}ms GPU active, "
        f"util={summary['gpu_util_pct']}% ==="
    )
    out.append(f"{'Kernel':<48} {'calls':>7} {'ms':>10} {'%':>6} {'us/call':>10}")
    for k in summary["kernels"][:15]:
        out.append(
            f"  {k['kernel']:<46} {k['calls']:>7} "
            f"{k['gpu_ms']:>10.1f} {k['gpu_pct']:>6.1f} {k['us_per_call']:>10.1f}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_csv", type=Path)
    ap.add_argument("--tile-kernel", default="mt_paged_attention_tile_mw_kernel")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ctx-tokens", type=int)
    ap.add_argument("--prefill-ms", type=float)
    ap.add_argument("--prefill-tps", type=float)
    ap.add_argument("--commit")
    ap.add_argument("--label")
    args = ap.parse_args()

    summary = parse(args.trace_csv, args.tile_kernel)
    print(fmt(summary))

    if args.out:
        record = {k: v for k, v in vars(args).items()
                  if k not in {"trace_csv", "out", "tile_kernel"} and v is not None}
        record["prefill"] = summary
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
        print(f"\nwrote baseline: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
