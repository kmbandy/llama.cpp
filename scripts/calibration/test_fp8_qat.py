# test_fp8_qat.py
import pytest
import torch
from fp8_qat import fp8_quant, FP8_E4M3_MAX, FP8_E5M2_MAX
from fp8_qat import pad_to_multiple

def test_fp8_quant_e4m3_roundtrip_per_row():
    x = torch.tensor([[1.0, 2.0, 4.0], [100.0, 200.0, 400.0]])
    q, scale = fp8_quant(x, fmt="e4m3")
    assert q.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1)                       # per-row
    recon = q.float() * scale
    assert torch.allclose(recon, x, rtol=0.1)          # fp8 rounding only
    # row amax maps to <= FP8_E4M3_MAX after scaling
    assert q.float().abs().max() <= FP8_E4M3_MAX + 1e-3

def test_fp8_quant_zero_row_guard():
    x = torch.zeros(1, 4)
    q, scale = fp8_quant(x, fmt="e4m3")
    assert scale.item() == 1.0                          # no div-by-zero
    assert q.float().abs().max() == 0.0

def test_fp8_quant_e5m2_wider_range():
    x = torch.full((1, 2), 20000.0)
    q, scale = fp8_quant(x, fmt="e5m2")
    assert q.dtype == torch.float8_e5m2
    assert torch.allclose(q.float() * scale, x, rtol=0.1)

def test_pad_to_multiple_pads_and_unpads():
    x = torch.randn(20, 8)
    xp, n_pad = pad_to_multiple(x, 16, dim=0)
    assert xp.shape[0] == 32 and n_pad == 12
    assert torch.equal(xp[:20], x) and xp[20:].abs().sum() == 0
    assert torch.equal(xp[: xp.shape[0] - n_pad], x)

def test_pad_to_multiple_noop_when_aligned():
    x = torch.randn(16, 8)
    xp, n_pad = pad_to_multiple(x, 16, dim=0)
    assert n_pad == 0 and torch.equal(xp, x)

import pytest
from fp8_qat import ml8_ref_linear

@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 GEMM needs GPU")
def test_ref_linear_matches_dequant_matmul():
    dev = "cuda"
    x = torch.randn(16, 64, device=dev) * 0.3
    W = torch.randn(32, 64, device=dev) * 0.1          # [N, K]
    y = ml8_ref_linear(x, W)                            # fp8 fwd
    y_ref = x @ W.t()
    rel = (y.float() - y_ref).norm() / y_ref.norm()
    assert rel < 0.1                                    # fp8 rounding band
