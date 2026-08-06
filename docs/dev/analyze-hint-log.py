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
    # (layer, expert) -> COUNT of speculative reads whose fate is still undecided.
    # A count, not a flag: the same page is speculatively read many times over a
    # run (read, used, evicted, read again). A boolean collapses those into one
    # and the accounting silently stops closing -- the first version of this
    # script classified 1872 of 4326 reads and lost the rest without complaint.
    # The reconciliation check at the end of main() exists to catch exactly that.
    # value is the PHASE the read happened in, or None when nothing outstanding.
    # Phase is attributed at READ time, not at resolution time: the NVMe read is
    # the cost, and the amplification gate is a statement about bytes spent.
    spec_open = {}
    hinted = set()
    n_hint_ids = n_spec = n_demand = 0
    # Split by phase because the two are judged differently and always have been:
    # during PREFILL the shared SN750 is at its ceiling, so a wasted speculative
    # read steals bandwidth from demand. During DECODE the drive is ~78% idle, so
    # the same wasted read is spending capacity that would otherwise go unused.
    # A single blended rate hides which of those we are looking at.
    used = defaultdict(int)
    late = defaultdict(int)
    mis  = defaultdict(int)
    n_spec_ph = defaultdict(int)
    # A decode request carries the draft block's experts (1-7 on this model); a
    # prefill request carries the union over a 2048-token ubatch (19-52). The
    # histogram is strictly bimodal with nothing between 7 and 19, and the low
    # bucket reconciles EXACTLY with the req log's decode request count, so this
    # is a clean partition rather than a tuned threshold.
    PREFILL_MIN_IDS = 10
    phase = 'prefill'   # prefill runs first
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
            ph = spec_open.pop(key, None)
            if ph is not None:
                # Selected while we held an outstanding speculative read of it.
                # If the same request still had to page it in, that read was
                # wasted: the page was reclaimed before its layer arrived.
                if e in cur_demand:
                    late[ph] += 1
                else:
                    used[ph] += 1
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
                key = (int(parts[1]), int(parts[2]))
                # A SECOND speculative read of a page whose first read is still
                # unresolved means the first was reclaimed before any reference
                # touched it -- spec_pagein skips anything still in a slot, so a
                # re-read proves an eviction. That is LATE, not a mispredict: the
                # expert was predicted correctly and the page was lost before its
                # layer arrived. Charging it to mispredict would blame the oracle
                # for the eviction band's behaviour, and the oracle here is a pure
                # token-id lookup that cannot be wrong.
                n_spec_ph[phase] += 1
                prev = spec_open.get(key)
                if prev is not None:
                    late[prev] += 1
                spec_open[key] = phase
            elif tag == 'R':
                resolve()
                ids = [int(e) for e in parts[2:]]
                phase = 'prefill' if len(ids) >= PREFILL_MIN_IDS else 'decode'
                cur_ref = (int(parts[1]), ids)
                cur_demand = set()
            elif tag == 'D':
                n_demand += 1
                cur_demand.add(int(parts[2]))
    resolve()

    # Anything still outstanding was speculatively read and never selected after.
    for ph in spec_open.values():
        mis[ph] += 1
    # EVERY speculative read must land in exactly one bucket, in its own phase.
    # If this trips, the classifier is dropping events and no rate means anything.
    for ph in ('prefill', 'decode'):
        assert used[ph] + late[ph] + mis[ph] == n_spec_ph[ph], (
            f"{path}/{ph}: {n_spec_ph[ph]} speculative page-ins but "
            f"{used[ph]}+{late[ph]}+{mis[ph]}={used[ph]+late[ph]+mis[ph]} classified")
    return dict(path=path, hint_ids=n_hint_ids, spec=n_spec, demand=n_demand,
                used=dict(used), late=dict(late), mispredict=dict(mis),
                n_spec_ph=dict(n_spec_ph), hinted_pages=len(hinted))


def main(paths):
    rows = []
    for p in paths:
        try:
            rows.append(analyze(p))
        except FileNotFoundError:
            # Loud, never silent. A missing worker log means the run covered
            # fewer workers than the caller thinks, and a total that quietly
            # omits the R9700 -- which holds experts 85..255, the majority
            # shard -- is worse than no total at all.
            print(f"!! MISSING: {p} -- totals below EXCLUDE this worker", file=sys.stderr)
    if not rows:
        return 1

    def agg(key, ph):
        return sum(r[key].get(ph, 0) for r in rows)

    w = max(len(r['path'].split('/')[-1]) for r in rows)
    # DENOMINATOR IS THE HINT STREAM, NOT spec_pi.
    #
    # spec_pagein_submit skips any page already in a slot, so the number of reads
    # ISSUED depends on pool residency, which depends on the eviction policy --
    # the very thing an A/B varies. Measured: an identical hint stream (17377
    # experts both arms) produced spec_pi 4401 vs 5061 and mispredict 6 vs 41
    # purely from pool composition. A used% over that denominator is therefore
    # not comparable across policies, and several figures quoted on 2026-08-06
    # were wrong for exactly this reason.
    #
    # hinted ids ARE fixed by the spine and the token stream, so used/hinted is
    # stable. spec_pi stays on the line as the COST, which is what it measures.
    print(f"{'log':{w}} {'phase':>8} {'hinted':>7} {'spec_pi':>8} {'USED':>7} "
          f"{'LATE':>7} {'MISPRED':>8} {'used/hint':>10}")
    for r in rows:
        name = r['path'].split('/')[-1]
        for ph in ('prefill', 'decode'):
            n = r['n_spec_ph'].get(ph, 0)
            u = r['used'].get(ph, 0)
            h = r['hint_ids'] if ph == 'decode' else 0
            pct = u / h if h else 0.0
            print(f"{name:{w}} {ph:>8} {h:7d} {n:8d} {u:7d} "
                  f"{r['late'].get(ph, 0):7d} {r['mispredict'].get(ph, 0):8d} {pct:9.1%}")
    print('-' * (w + 50))
    hinted_all = sum(r['hint_ids'] for r in rows)
    for ph in ('prefill', 'decode'):
        n, u = agg('n_spec_ph', ph), agg('used', ph)
        h = hinted_all if ph == 'decode' else 0
        pct = u / h if h else 0.0
        print(f"{'TOTAL':{w}} {ph:>8} {h:7d} {n:8d} {u:7d} "
              f"{agg('late', ph):7d} {agg('mispredict', ph):8d} {pct:9.1%}")

    print()
    for ph in ('prefill', 'decode'):
        n = agg('n_spec_ph', ph)
        if not n:
            continue
        u, l, m = agg('used', ph), agg('late', ph), agg('mispredict', ph)
        print(f"{ph.upper()}: {n} speculative page-ins")
        print(f"  used       : {u:6d}  ({u/n:5.1%})  became RESIDENT instead of a demand page-in")
        print(f"  LATE       : {l:6d}  ({l/n:5.1%})  right expert, evicted before its layer arrived")
        print(f"  MISPREDICT : {m:6d}  ({m/n:5.1%})  never selected -- pure waste")
    print()
    print("LATE says the eviction band is wrong. MISPREDICT says the prediction is.")
    print("They have different fixes, which is why they must never be one bucket.")
    print("And the two phases have different economics: the drive is at its ceiling")
    print("during prefill and ~78% idle during decode, so a wasted read costs")
    print("bandwidth in one and spends spare capacity in the other.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
