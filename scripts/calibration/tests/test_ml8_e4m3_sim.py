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

# append to tests/test_ml8_e4m3_sim.py
import torch
from ml8_e4m3_sim import (fp32_to_e4m3_bits, e4m3_bits_to_fp32,
                          e4m3_roundtrip, quantize_act_per_row)

def test_vectorized_roundtrip_matches_scalar():
    g = torch.Generator().manual_seed(0)
    # mix of ranges: normal, the 256..448 band, subnormal, saturation
    x = torch.cat([
        torch.randn(4000, generator=g) * 50.0,
        torch.linspace(250.0, 460.0, 1000),
        torch.linspace(-0.02, 0.02, 1000),
    ])
    vec = e4m3_roundtrip(x)
    scal = torch.tensor([e4m3_bits_to_fp32(fp32_to_e4m3_bits(float(v))) for v in x])
    # NaN slot can appear only for |x|>448 already handled; compare finite
    assert torch.equal(vec, scal), (
        f"max abs diff {torch.nan_to_num(vec - scal).abs().max().item()}")

def test_quantize_act_per_row_scale_and_eps():
    x = torch.tensor([[448.0, 224.0, 0.0, -448.0],
                      [0.0, 0.0, 0.0, 0.0]])          # row 1 all-zero -> eps path
    q = quantize_act_per_row(x)
    # row 0: absmax=448 -> scale=1.0 -> values land on lattice unchanged
    assert torch.allclose(q[0], torch.tensor([448.0, 224.0, 0.0, -448.0]))
    # row 1: all-zero stays zero (no nan/inf from eps division)
    assert torch.equal(q[1], torch.zeros(4))
