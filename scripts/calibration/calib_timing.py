# scripts/calibration/calib_timing.py
"""Lightweight phase-timer for calibration instrumentation (MAD-256 Phase 1).

One responsibility: accumulate wall time per labelled phase + optional per-call
events, dump to JSON. Zero deps beyond stdlib so it never perturbs the run.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseTimer:
    def __init__(self) -> None:
        self._phases: dict[str, dict] = {}
        self._events: list[dict] = []

    @contextmanager
    def phase(self, label: str, **meta):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            p = self._phases.setdefault(label, {"seconds": 0.0, "calls": 0})
            p["seconds"] += dt
            p["calls"] += 1
            if meta:
                self._events.append({"label": label, "seconds": dt, **meta})

    def summary(self) -> dict:
        total = sum(p["seconds"] for p in self._phases.values())
        return {
            "phases": self._phases,
            "total_seconds": total,
            "events": self._events,
        }

    def dump_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2))
