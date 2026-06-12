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


from fp8_qat import Ml8Fp8Fn
from act_replay_student import AttachedTarget
from test_act_replay_cli import _mk_state   # reuse the tiny ml8 target builder

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fp8fn_forward_matches_ste_weight():
    dev = "cuda"
    at = AttachedTarget(_mk_state(N=32, K=128, G=2)).to(dev)
    x = torch.randn(16, 128, device=dev) * 0.3
    with torch.no_grad():
        y = Ml8Fp8Fn.apply(x, at.centroids, at.scales, at.indices, at.gidx)
        y_ref = x @ at.weight().t()                    # bf16 STE dequant path
    rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
    assert rel < 0.12                                  # fp8 vs bf16 rounding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fp8fn_backward_matches_ste_grads():
    dev = "cuda"
    s = _mk_state(N=32, K=128, G=2)
    at_a = AttachedTarget(s).to(dev); at_b = AttachedTarget(s).to(dev)
    x = torch.randn(16, 128, device=dev) * 0.3
    g = torch.randn(16, 32, device=dev)
    # fp8 path
    y_a = Ml8Fp8Fn.apply(x, at_a.centroids, at_a.scales, at_a.indices, at_a.gidx)
    y_a.backward(g)
    # bf16 STE reference path
    y_b = x @ at_b.weight().t(); y_b.backward(g)
    # cosine of centroid grads should be high (same descent direction)
    ca, cb = at_a.centroids.grad.flatten(), at_b.centroids.grad.flatten()
    cos = torch.nn.functional.cosine_similarity(ca, cb, dim=0)
    assert cos > 0.95, f"centroid grad cosine {cos:.3f}"
    assert at_a.scales.grad is not None and torch.isfinite(at_a.scales.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fp8fn_backward_stashes_dLdW_by_indices_id():
    dev = "cuda"
    at = AttachedTarget(_mk_state(N=32, K=128, G=2)).to(dev)
    x = torch.randn(16, 128, device=dev) * 0.3
    y = Ml8Fp8Fn.apply(x, at.centroids, at.scales, at.indices, at.gidx)
    y.sum().backward()
    key = id(at.indices)
    assert key in Ml8Fp8Fn.last_dLdW
    assert Ml8Fp8Fn.last_dLdW[key].shape == at.indices.shape   # [N,K]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fp8fn_backward_stashes_curvature_h_by_indices_id():
    # D.1: backward stashes per-input-column GPTQ Hessian diagonal h_k = E[x_k^2]
    # (mean over the M batch rows, matching the token-mean loss convention of g).
    dev = "cuda"
    at = AttachedTarget(_mk_state(N=32, K=128, G=2)).to(dev)
    K = at.indices.shape[1]                          # 128
    x = torch.randn(16, K, device=dev) * 0.1
    x[:, K // 2:] *= 10.0                             # second half has ~100x the energy
    y = Ml8Fp8Fn.apply(x, at.centroids, at.scales, at.indices, at.gidx)
    y.sum().backward()
    key = id(at.indices)
    assert key in Ml8Fp8Fn.last_h
    h = Ml8Fp8Fn.last_h[key]
    assert h.shape == (K,)                            # per input column
    assert torch.isfinite(h).all() and (h >= 0).all()
    # h tracks per-column second moment: high-energy half >> low-energy half
    assert h[K // 2:].mean() > h[:K // 2].mean() * 10
