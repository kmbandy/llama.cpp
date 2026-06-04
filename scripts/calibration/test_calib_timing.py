# scripts/calibration/test_calib_timing.py
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from calib_timing import PhaseTimer


def test_accumulates_seconds_and_calls():
    t = PhaseTimer()
    for _ in range(3):
        with t.phase("hessian_forward"):
            time.sleep(0.01)
    s = t.summary()
    assert s["phases"]["hessian_forward"]["calls"] == 3
    assert s["phases"]["hessian_forward"]["seconds"] >= 0.025
    assert s["total_seconds"] >= 0.025


def test_records_per_call_events_with_metadata():
    t = PhaseTimer()
    with t.phase("hessian_forward", target="blk.0.ffn_down", n_tok=2048):
        time.sleep(0.005)
    s = t.summary()
    ev = s["events"]
    assert len(ev) == 1
    assert ev[0]["label"] == "hessian_forward"
    assert ev[0]["target"] == "blk.0.ffn_down"
    assert ev[0]["n_tok"] == 2048
    assert ev[0]["seconds"] >= 0.004


def test_multiple_labels_kept_separate():
    t = PhaseTimer()
    with t.phase("corpus_load"):
        time.sleep(0.005)
    with t.phase("gptq_quantize"):
        time.sleep(0.005)
    s = t.summary()
    assert set(s["phases"]) == {"corpus_load", "gptq_quantize"}


def test_dump_json_roundtrips(tmp_path):
    t = PhaseTimer()
    with t.phase("corpus_load"):
        time.sleep(0.001)
    out = tmp_path / "phase_timing.json"
    t.dump_json(out)
    loaded = json.loads(Path(out).read_text())
    assert "corpus_load" in loaded["phases"]
    assert loaded["total_seconds"] >= 0.0


def test_exception_in_phase_still_records():
    t = PhaseTimer()
    try:
        with t.phase("gptq_quantize"):
            raise ValueError("boom")
    except ValueError:
        pass
    s = t.summary()
    assert s["phases"]["gptq_quantize"]["calls"] == 1
