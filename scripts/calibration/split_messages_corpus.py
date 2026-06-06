#!/usr/bin/env python3
"""Split a messages-jsonl corpus into DISJOINT calibration and held-out pools.

The calibration corpus defines the Hessian H that the method descends; evaluating
on any record that also fed H is Goodhart leakage. This util carves a held-out
eval pool that the calibration sampler will never see, by a SEEDED shuffle + slice
so the partition is deterministic and auditable (same seed → same split, forever).

Physically separate output files (not a hash filter at sample time) so the calib
sampler can only ever open the calib pool — the disjointness is structural, not a
runtime invariant that could regress.

    python split_messages_corpus.py \
        --in  ~/models/calib_sources/claude_traces.jsonl \
        --calib ~/models/calib_sources/claude_traces.calib.jsonl \
        --heldout ~/models/calib_sources/claude_traces.heldout.jsonl \
        --heldout-n 60 --seed 0
"""
import argparse
import json
import random
from pathlib import Path


def split_records(records, heldout_n, seed=0):
    """Return (calib_records, heldout_records) — a seeded, disjoint partition.

    The shuffle is over a copy keyed by a stable index so the split is reproducible
    and independent of input order. heldout_n records go to the held-out pool; the
    remainder to calibration. heldout_n is clamped to [0, len(records)].
    """
    n = len(records)
    heldout_n = max(0, min(heldout_n, n))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    held_idx = set(order[:heldout_n])
    heldout = [records[i] for i in order[:heldout_n]]
    calib = [records[i] for i in range(n) if i not in held_idx]
    return calib, heldout


def _read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path, records):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    records = _read_jsonl(Path(a.in_path).expanduser())
    calib, heldout = split_records(records, a.heldout_n, a.seed)
    _write_jsonl(a.calib, calib)
    _write_jsonl(a.heldout, heldout)
    print(f"[split] {len(records)} records (seed {a.seed}) → "
          f"{len(calib)} calib / {len(heldout)} heldout (DISJOINT)")


if __name__ == "__main__":
    main()
