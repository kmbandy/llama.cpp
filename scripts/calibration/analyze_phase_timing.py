# scripts/calibration/analyze_phase_timing.py
"""Summarize a phase_timing.json from a Phase-1 instrumented calibration.

Prints the phase breakdown, the forward share, and the per-target Hessian-forward
distribution that exposes N-times redundant re-forwarding. Optionally folds in a
dtype_probe.json for the fp32-vs-WMMA ratio.

Usage:
  python3 analyze_phase_timing.py <output_dir>
"""
import json
import sys
from pathlib import Path


def main(out_dir: str) -> None:
    d = Path(out_dir)
    s = json.loads((d / "phase_timing.json").read_text())
    phases = s["phases"]
    total = s["total_seconds"]
    print(f"# Phase breakdown ({d.name}) — total {total:.1f}s")
    for lbl, p in sorted(phases.items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"  {lbl:20s} {p['seconds']:9.1f}s  "
              f"{100*p['seconds']/max(total,1e-9):5.1f}%  calls={p['calls']}")

    tgt = [e for e in s["events"] if e.get("label") == "hessian_forward_target"]
    if tgt:
        n = len(tgt)
        secs = sorted(e["seconds"] for e in tgt)
        agg = phases.get("hessian_forward", {}).get("seconds", 0.0)
        per = agg / max(n, 1)
        print(f"\n# Hessian-forward re-forwarding")
        print(f"  targets (N)            {n}")
        print(f"  aggregate forward      {agg:.1f}s")
        print(f"  mean per-target        {per:.1f}s  (= one full corpus forward)")
        print(f"  implied 1-pass forward {per:.1f}s  vs  N-pass {agg:.1f}s "
              f"→ up to {n:.0f}x headroom if collapsed to a single pass")
        print(f"  per-target min/median/max  "
              f"{secs[0]:.1f} / {secs[n//2]:.1f} / {secs[-1]:.1f}s")

    probe = d / "dtype_probe.json"
    if probe.exists():
        pj = json.loads(probe.read_text())
        print(f"\n# fp32-vs-WMMA matmul tax")
        print(f"  allow_tf32=False  {pj['fp32_s']:.2f}s / {pj['k_samples']} samp")
        print(f"  allow_tf32=True   {pj['tf32_s']:.2f}s / {pj['k_samples']} samp")
        print(f"  ratio             {pj['ratio']:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
