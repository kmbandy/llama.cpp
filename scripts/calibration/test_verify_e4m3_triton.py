# scripts/calibration/test_verify_e4m3_triton.py
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_e4m3_triton import sweep_inputs, compare_codes
from ml8_e4m3_sim import e4m3_roundtrip


def test_oracle_against_itself_zero_mismatches():
    x = sweep_inputs()
    # A "cast" that is exactly the oracle must produce zero mismatches.
    mism = compare_codes(x, cast_fn=e4m3_roundtrip)
    assert mism["n_mismatch"] == 0


def test_sweep_covers_subnormal_and_saturation():
    x = sweep_inputs()
    assert (x.abs() < 2.0 ** -6).any()    # subnormal e4m3 region
    assert (x.abs() > 448.0).any()         # saturation region
