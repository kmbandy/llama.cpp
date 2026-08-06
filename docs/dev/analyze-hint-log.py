#!/usr/bin/env python3
"""Separate MISPREDICT from LATE in a WP_HINT_LOG event stream.

Usage:  analyze-hint-log.py hint-warm_1-w-480.txt [more logs...]

WHY THIS EXISTS. The 2026-08-02 vocabulary note requires that any prefetch
evaluation report mispredict rate and late rate SEPARATELY from page-in rate,
"or it will be uninterpretable". Arm 2 (2026-08-06) reported neither: it had one
lumped bucket of speculative reads and a "utilisation %", which cannot tell
"we fetched an expert that was never selected" from "we fetched the right expert
and it was evicted before its layer arrived". Those have different fixes -- the
first says the prediction is wrong, the second says the LRU band is.

THE VOCABULARY (kmbandy, standing):
  RESIDENT    already in a VRAM slot. No I/O.
  PAGE-IN     read from NVMe because it was not resident. EXPECTED OPERATION.
  MISPREDICT  speculatively paged in, never selected. THE ONLY TRUE MISS.
  LATE        predicted correctly, but it had to be demand-read anyway.

THE STREAM. WP_HINT_LOG interleaves four event kinds in ONE file, in order,
because the order is the only thing that separates the two failures:
  H <layer> <id>...   hint received      -- the PREDICTION
  S <layer> <expert>  speculative page-in -- the COST
  R <layer> <id>...   dispatch reference  -- the GROUND TRUTH
  D <layer> <expert>  demand page-in      -- what speculation failed to prevent
  C <counters>        running counters    -- tail -1 is the durable summary

R is emitted BEFORE the dispatch it describes, so every D provoked by a
reference follows its R. That is what makes one forward pass sufficient.
"""
import sys
from collections import defaultdict


def analyze(path):
    # (layer, expert) -> True once speculatively read, until its fate is decided.
    spec_open = {}
    hinted = set()
    n_hint_ids = n_spec = n_demand = 0
    used = late = 0
    # The reference currently being resolved, and the demand reads it provoked.
    cur_ref, cur_demand = None, set()

    def resolve():
        """Decide the fate of every speculative page the open reference selected."""
        nonlocal used, late, cur_ref
        if cur_ref is None:
            return
        layer, ids = cur_ref
        for e in ids:
            key = (layer, e)
            if spec_open.pop(key, False):
                # Selected while we held a speculative read of it. If the same
                # request still had to page it in, the read was wasted: the page
                # was reclaimed before its layer arrived.
                if e in cur_demand:
                    late += 1
                else:
                    used += 1
        cur_ref = None

    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'C':
                continue
            if tag == 'H':
                resolve()
                layer = int(parts[1])
                for e in parts[2:]:
                    hinted.add((layer, int(e)))
                    n_hint_ids += 1
            elif tag == 'S':
                n_spec += 1
                spec_open[(int(parts[1]), int(parts[2]))] = True
            elif tag == 'R':
                resolve()
                cur_ref = (int(parts[1]), [int(e) for e in parts[2:]])
                cur_demand = set()
            elif tag == 'D':
                n_demand += 1
                cur_demand.add(int(parts[2]))
    resolve()

    # Anything still open was speculatively read and never selected afterwards.
    mispredict = len(spec_open)
    return dict(path=path, hint_ids=n_hint_ids, spec=n_spec, demand=n_demand,
                used=used, late=late, mispredict=mispredict,
                noop=len(hinted) and n_hint_ids - n_spec)


def main(paths):
    tot = defaultdict(int)
    rows = []
    for p in paths:
        r = analyze(p)
        rows.append(r)
        for k in ('hint_ids', 'spec', 'demand', 'used', 'late', 'mispredict'):
            tot[k] += r[k]

    w = max(len(r['path'].split('/')[-1]) for r in rows) if rows else 10
    print(f"{'log':{w}} {'hinted':>8} {'spec_pi':>8} {'demand_pi':>10} "
          f"{'USED':>7} {'LATE':>7} {'MISPRED':>8} {'used%':>7}")
    for r in rows:
        pct = r['used'] / r['spec'] if r['spec'] else 0.0
        print(f"{r['path'].split('/')[-1]:{w}} {r['hint_ids']:8d} {r['spec']:8d} "
              f"{r['demand']:10d} {r['used']:7d} {r['late']:7d} "
              f"{r['mispredict']:8d} {pct:6.1%}")
    if len(rows) > 1:
        pct = tot['used'] / tot['spec'] if tot['spec'] else 0.0
        print('-' * (w + 60))
        print(f"{'TOTAL':{w}} {tot['hint_ids']:8d} {tot['spec']:8d} "
              f"{tot['demand']:10d} {tot['used']:7d} {tot['late']:7d} "
              f"{tot['mispredict']:8d} {pct:6.1%}")

    if tot['spec']:
        print()
        print(f"speculative page-ins : {tot['spec']}")
        print(f"  used               : {tot['used']:6d}  ({tot['used']/tot['spec']:.1%})  "
              f"the win -- became RESIDENT instead of a demand page-in")
        print(f"  LATE               : {tot['late']:6d}  ({tot['late']/tot['spec']:.1%})  "
              f"right expert, evicted before its layer arrived")
        print(f"  MISPREDICT         : {tot['mispredict']:6d}  ({tot['mispredict']/tot['spec']:.1%})  "
              f"never selected -- pure waste")
        print()
        print("LATE says the eviction band is wrong. MISPREDICT says the prediction is.")
        print("They have different fixes, which is why they must never be one bucket.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
