#!/usr/bin/env python3
"""
hipBLASLt fp8/bf16 vendor bar, MEASURED TO MATCH THE DSWS HARNESS.

WHY THIS EXISTS (2026-07-21). The prior vendor harness (~/dsws_gpu_logs/bench_hipblaslt_ml8.py)
is arithmetically CORRECT -- it derives TF from a measured wall time, no string parsing, full
precision in its JSON. Nothing about it was fabricated. But the resulting COMPARISON against DSWS
was methodologically inconsistent in ways that all favoured the vendor:

  1. STATISTIC. It reports best = min(...) over 20-50 iterations -- the vendor's single luckiest
     run -- while the DSWS harness reports a MEAN across reps. Best-case vs average.
  2. CARD STATE. It was run without a board claim, so competing GPU work is unknown.
  3. SHAPE LIST. It carried its own copy of the shapes, free to drift from the DSWS list.

This version fixes 1 and 3 and is meant to be run under a held claim (2).

  - PRIMARY STATISTIC IS THE MEAN, matching DSWS. min/median/max/stdev are all reported so the
    old min-of-N numbers remain reconstructible and the two eras stay comparable.
  - SHAPES ARE IMPORTED FROM dsws_realshape_bench.SHAPES. Both sides measure the identical list
    BY CONSTRUCTION; there is no second copy to drift.

WHAT THIS STILL CANNOT EQUALIZE, and must be stated wherever these numbers are cited:
    DSWS runs under the compositor-safe cap (512 tiles/dispatch by default); hipBLASLt runs as ONE
    unchunked call. That cap is worth ~2.2x mean / 17.7x worst on DSWS (RESULTS_DSWS_BASELINE
    2026-07-21 §5). It is a constraint we accept and the vendor does not. It is NOT a measurement
    artifact and must never be silently absorbed into a ratio.

Also note DSWS pads M up to its super-tile and corrects TF back to real FLOP; the vendor runs the
REAL M. Both sides are therefore quoted on real FLOP. No padding credit either way.

GPU USE: this dispatches real GEMMs on the display GPU. Hold a board claim. Iteration counts are
bounded (per-call work is small; the whole sweep is well under a second of GPU time per shape).
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from dsws_realshape_bench import SHAPES  # single source of truth for the shape list

PEAK_FP8 = 307.0          # measured fp8 WMMA ceiling on this card
BW       = 644.6e9        # R9700 GDDR6


def timed_calls(fn, iters, warmup):
    """Return EVERY per-call time. The caller picks the statistic; nothing is discarded here."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def fp8_gemm(dev, M, K, N):
    import torch
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)  # .t() -> K-major
    sa = torch.tensor(1.0, device=dev)
    sb = torch.tensor(1.0, device=dev)
    f = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    f()   # shape/layout check before timing; _scaled_mm rejects N%16 != 0
    return f


def bf16_gemm(dev, M, K, N):
    import torch
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    return lambda: torch.matmul(a, b)


def stats(times, flop):
    """TF for each statistic. NOTE the inversion: MIN TIME is MAX TF, so tf_min_time is the
    optimistic figure the old harness reported as its headline."""
    tf = lambda t: flop / t / 1e12
    return dict(
        iters=len(times),
        tf_mean=tf(statistics.mean(times)),            # PRIMARY -- matches DSWS
        tf_median=tf(statistics.median(times)),
        tf_min_time=tf(min(times)),                    # == the OLD harness's reported number
        tf_max_time=tf(max(times)),
        spread_percent=(max(times) - min(times)) / max(times) * 100.0,
        seconds_mean=statistics.mean(times),
        seconds_min=min(times),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--skip-bf16", action="store_true", help="fp8 only (halves GPU time)")
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "no GPU"
    dev = torch.device("cuda:0")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx1201" in arch, f"expected gfx1201, got {arch}"

    prov = dict(device=torch.cuda.get_device_name(0), arch=arch,
                torch=torch.__version__, hip=torch.version.hip,
                iters=args.iters, warmup=args.warmup,
                primary_statistic="tf_mean",
                shape_source="dsws_realshape_bench.SHAPES",
                note="DSWS runs under the 512-tile compositor cap; this runs UNCHUNKED. "
                     "That asymmetry is real and is NOT corrected here.")
    print(f"# {prov['device']} ({arch})  torch {prov['torch']} hip {prov['hip']}")
    print(f"# iters={args.iters} warmup={args.warmup}  PRIMARY = tf_mean (matches DSWS)")
    print(f"# shapes from {prov['shape_source']}  n={len(SHAPES)}")
    print()

    rows = []
    hdr = f"{'shape':<24}{'M':>6}{'N':>7}{'K':>6} | {'fp8 mean':>9}{'fp8 min-t':>10}{'spread':>8} | {'bf16 mean':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, M, N, K in SHAPES:
        flop = 2.0 * M * N * K
        rec = dict(shape=label, M=M, N=N, K=K, gflop=flop / 1e9, fp8=None, bf16=None, fp8_error=None)
        try:
            rec["fp8"] = stats(timed_calls(fp8_gemm(dev, M, K, N), args.iters, args.warmup), flop)
        except Exception as exc:                       # _scaled_mm rejects N%16 != 0
            rec["fp8_error"] = f"{type(exc).__name__}: {exc}"[:200]
        if not args.skip_bf16:
            try:
                rec["bf16"] = stats(timed_calls(bf16_gemm(dev, M, K, N), args.iters, args.warmup), flop)
            except Exception as exc:
                rec["bf16_error"] = f"{type(exc).__name__}: {exc}"[:200]
        f8, bf = rec["fp8"], rec["bf16"]
        c_mean = format(f8["tf_mean"], "9.3f") if f8 else "   REJECT"
        c_mint = format(f8["tf_min_time"], "10.3f") if f8 else "         -"
        c_sprd = (format(f8["spread_percent"], "7.1f") + "%") if f8 else "        -"
        c_bf16 = format(bf["tf_mean"], "10.3f") if bf else "         -"
        print(f"{label:<24}{M:6d}{N:7d}{K:6d} | {c_mean}{c_mint}{c_sprd} | {c_bf16}")
        rows.append(rec)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(provenance=prov, rows=rows), open(args.json, "w"), indent=1)
    with open(args.table, "w") as fh:
        fh.write(f"# {json.dumps(prov)}\n{hdr}\n")
        for r in rows:
            f8, bf = r["fp8"], r["bf16"]
            fh.write(f"{r['shape']:<24}{r['M']:6d}{r['N']:7d}{r['K']:6d} | "
                     f"{(f8['tf_mean'] if f8 else float('nan')):9.3f}"
                     f"{(f8['tf_min_time'] if f8 else float('nan')):10.3f}"
                     f"{(f8['spread_percent'] if f8 else float('nan')):7.1f}% | "
                     f"{(bf['tf_mean'] if bf else float('nan')):10.3f}\n")
    ok = sum(1 for r in rows if r["fp8"])
    print(f"\nshapes={len(rows)} fp8_measured={ok} fp8_rejected={len(rows)-ok}")
    print(f"JSON : {args.json}\ntable: {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
