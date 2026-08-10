#!/usr/bin/env python3
"""Fit SEGK slopes from gpu_run logs without using rendered TF."""

import argparse
import glob
import re
import sys
from collections import defaultdict

CONFIG = re.compile(r"\bG=(\d+) SEGK=(\d+) FM=(\d+) FN=(\d+)")
WORK = re.compile(r"^\s*\[dsws2 WORK-EXACT\] computed == G\*TOTAL_super == (\d+)\b", re.M)
SUSTAINED = re.compile(r"^\s*\[dsws2 SUSTAINED\] reps=(\d+)\b", re.M)
TIMING = re.compile(
    r"^\s*\[dsws2 THROUGHPUT\].*?span=(\d+) ticks / \d+ chunk\(s\) @\s*([0-9.]+) MHz\s*$", re.M
)
ARM = re.compile(r"(?:^|[_./-])(baseline|off|stayfat|nodsadd|nobload|cfassign|bndprobe)(?:[_./-]|$)", re.I)


def reject(path, reason):
    raise ValueError(f"{path}: {reason}")


def parse(path):
    text = open(path, encoding="utf-8").read()
    if "WORK-EXACT: CANNOT-EVALUATE" in text or "WORK-INEXACT" in text:
        reject(path, "not WORK-EXACT")
    work = WORK.search(text)
    cfg = CONFIG.search(text)
    timing = TIMING.search(text)
    if not (work and cfg and timing):
        reject(path, "missing config, WORK-EXACT, or span timing")
    g, segk, fm, fn = map(int, cfg.groups())
    expected = int(work.group(1))
    sustained = SUSTAINED.search(text)
    if not sustained:
        reject(path, "missing explicit fixed-repetition count")
    reps = int(sustained.group(1))
    if reps < 1 or expected <= 0:
        reject(path, "invalid repetition/work count")
    ticks = int(timing.group(1))
    mhz = float(timing.group(2))
    # occ[3]-occ[2] is the summed span over all completed chunks. MHz is ticks/us.
    ms_per_rep = ticks / (mhz * 1000.0 * reps)
    label = ARM.search(path.lower())
    arm = label.group(1).lower() if label else "unknown"
    if arm == "off":
        arm = "baseline"
    return arm, segk, ms_per_rep, expected, reps, (fm, fn, g)


def fit(points):
    if len(points) < 2:
        raise ValueError("need at least two SEGK points")
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    den = sum((x - xm) ** 2 for x in xs)
    if den == 0:
        raise ValueError("SEGK points are not distinct")
    slope = sum((x - xm) * (y - ym) for x, y in points) / den
    intercept = ym - slope * xm
    return intercept, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="gpu_run logs or shell globs")
    args = ap.parse_args()
    paths = []
    for item in args.logs:
        matches = glob.glob(item)
        paths.extend(matches or [item])
    by_arm = defaultdict(list)
    for path in paths:
        try:
            arm, segk, ms, expected, reps, geom = parse(path)
        except (OSError, ValueError) as exc:
            print(f"REJECT: {exc}", file=sys.stderr)
            return 2
        by_arm[arm].append((256 // segk, ms))
        print(f"ACCEPT {path}: arm={arm} SEGK={segk} n_kseg={256 // segk} time_ms_per_rep={ms:.6f} computed={expected} reps={reps}")
    results = {}
    for arm, points in sorted(by_arm.items()):
        a, b = fit(points)
        results[arm] = (a, b)
    if "baseline" not in results:
        print("REJECT: baseline arm is required for decomposition", file=sys.stderr)
        return 2
    base_slope = results["baseline"][1]
    print("\narm        intercept_ms  slope_ms/n_kseg  slope_drop_ms/n_kseg")
    for arm, (a, b) in sorted(results.items()):
        print(f"{arm:10s} {a:13.6f} {b:16.6f} {base_slope - b:22.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
