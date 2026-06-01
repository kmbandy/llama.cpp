# scripts/calibration/tests/test_ml8_e4m3_sim.py
import struct, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
from ml8_e4m3_sim import fp32_to_e4m3_bits   # noqa: E402

GOLDEN = Path("/tmp/ml8_e4m3_golden.bin")

def _ensure_golden():
    if GOLDEN.exists():
        return
    src = CALIB / "tools/ml8_e4m3_golden.c"
    exe = Path("/tmp/ml8_e4m3_golden")
    subprocess.run(["cc", "-O2", "-o", str(exe), str(src)], check=True)
    subprocess.run([str(exe)], check=True)

def _load_golden():
    _ensure_golden()
    with open(GOLDEN, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        xs = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
        cs = np.frombuffer(f.read(n), dtype=np.uint8).copy()
    return xs, cs

def test_gate_a_bit_match_kernel():
    xs, golden = _load_golden()
    got = np.array([fp32_to_e4m3_bits(float(x)) for x in xs], dtype=np.uint8)
    mism = np.nonzero(got != golden)[0]
    assert mism.size == 0, (
        f"{mism.size} mismatches; first: x={xs[mism[0]]!r} "
        f"got=0x{got[mism[0]]:02x} want=0x{golden[mism[0]]:02x}")
