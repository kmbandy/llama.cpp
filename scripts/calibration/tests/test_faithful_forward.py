# scripts/calibration/tests/test_faithful_forward.py
import sys
from pathlib import Path
import torch
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
from faithful_forward import build_rotations   # noqa: E402

def test_build_rotations_matches_inline_formula():
    # mirrors calibrate_ml8_paged.py:1390-1397 seed math
    dims = {("L0", "ffn_gate"): 2560}
    seeds = {("L0", "ffn_gate"): 5 + 0 * 7 + 0}
    rots = build_rotations(dims, seeds, max_b=1024)
    from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
    a, b = factor_for_dim(2560, max_b=1024)
    ref = KroneckerRotation(h_a=random_orthogonal(a, seed=5), b_dim=b)
    x = torch.randn(3, 2560)
    assert torch.allclose(rots[("L0", "ffn_gate")].forward(x), ref.forward(x), atol=1e-6)

# append to tests/test_faithful_forward.py
import torch.nn as nn
from faithful_forward import FaithfulActHook
from ml8_e4m3_sim import quantize_act_per_row

def _rot(K):
    from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
    a, b = factor_for_dim(K, max_b=1024)
    return KroneckerRotation(h_a=random_orthogonal(a, seed=1), b_dim=b)

def test_x_eff_is_faithful_output_via_unchanged_linear():
    K, N, T = 256, 8, 5
    lin = nn.Linear(K, N, bias=False)
    rot = _rot(K)
    hook = FaithfulActHook(rot, enabled=True)
    x = torch.randn(T, K)
    # reference faithful output: e4m3(x@Q) @ (Q^T W^T)
    aq = quantize_act_per_row(rot.forward(x))
    W = lin.weight.data.float()
    y_ref = aq @ rot.forward(W).t()          # rot.forward(W) = W@Q ; (W@Q)^T
    # hook replaces input with x_eff; unchanged linear then yields y_ref
    h = lin.register_forward_pre_hook(hook, with_kwargs=False)
    y_got = lin(x)
    h.remove()
    assert torch.allclose(y_got, y_ref, atol=1e-4)

def test_disabled_hook_is_identity():
    K = 256
    lin = nn.Linear(K, 4, bias=False)
    hook = FaithfulActHook(_rot(K), enabled=False)
    x = torch.randn(3, K)
    h = lin.register_forward_pre_hook(hook)
    y_got = lin(x)
    h.remove()
    assert torch.allclose(y_got, torch.nn.functional.linear(x, lin.weight))

def test_hessian_accumulates_in_rotated_quant_space():
    K, T = 256, 7
    rot = _rot(K)
    hook = FaithfulActHook(rot, enabled=True)
    hook.set_hessian_target(True)
    lin = nn.Linear(K, 4, bias=False)
    h = lin.register_forward_pre_hook(hook)
    x = torch.randn(T, K)
    lin(x)
    h.remove()
    aq = quantize_act_per_row(rot.forward(x))
    assert torch.allclose(hook.H, aq.t() @ aq, atol=1e-3)
    assert hook.n_tokens == T

# append to tests/test_faithful_forward.py
from faithful_forward import fp8_weight_override

def test_fp8_weight_override_roundtrips_through_scaled_fp8():
    import torch
    from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8
    w = torch.randn(16, 64)
    got = fp8_weight_override(w, group_size=32)
    want = dequantize_scaled_fp8(quantize_scaled_fp8(w, group_size=32))
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, w)        # it actually changed the weights
