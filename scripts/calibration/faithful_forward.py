# scripts/calibration/faithful_forward.py
"""Deployment-faithful (W4A8) calibration forward: rotation precompute + the
activation-e4m3 pre-hook. See docs/superpowers/specs/2026-05-31-w4a8-faithful-calibration-design.md.
"""
import torch
from kronecker_rotation import (KroneckerRotation, random_orthogonal,
                                factor_for_dim)

def build_rotations(dims: dict, seeds: dict, max_b: int = 1024) -> dict:
    """dims/seeds keyed by (layer_key, kind) -> rotation. Built from dims+seeds
    only (never from H values), so it can run before Hessian collection."""
    rots = {}
    for key, K in dims.items():
        a, b = factor_for_dim(K, max_b=max_b)
        rots[key] = KroneckerRotation(h_a=random_orthogonal(a, seed=int(seeds[key])), b_dim=b)
    return rots

# append to faithful_forward.py
from ml8_e4m3_sim import quantize_act_per_row

class FaithfulActHook:
    """forward_pre_hook on an ml8-4 linear. When enabled, replaces the input x
    with x_eff = e4m3(x@Q) @ Q^T so the unchanged linear yields the faithful
    W4A8 output, and (when this layer is the active Hessian target) accumulates
    H += a_q^T a_q in rotated+quantized space (so rotate_hessian must NOT run)."""
    def __init__(self, rotation, enabled: bool = True):
        self.rotation = rotation
        self.enabled = enabled
        self._is_target = False
        self.H = None
        self.n_tokens = 0

    def set_hessian_target(self, on: bool):
        self._is_target = on

    def reset_hessian(self):
        self.H = None
        self.n_tokens = 0

    def __call__(self, module, args):
        if not self.enabled:
            return None                      # no-op: original input flows
        x = args[0]
        orig_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1]).float()      # [T, K]
        a_rot = self.rotation.forward(flat)            # x@Q
        a_q = quantize_act_per_row(a_rot)              # e4m3 per-row
        if self._is_target:
            XtX = a_q.t() @ a_q
            self.H = XtX if self.H is None else self.H + XtX
            self.n_tokens += a_q.shape[0]
        x_eff = self.rotation.inverse(a_q)             # a_q @ Q^T  (= inverse since Q orthogonal)
        x_eff = x_eff.reshape(x.shape).to(orig_dtype)
        return (x_eff,) + tuple(args[1:])

def assert_not_double_rotated(faithful_acts: bool, rotate_hessian_called: bool):
    """Guard: with faithful-acts the rotation is already baked into H by the
    forward; calling rotate_hessian again double-rotates."""
    if faithful_acts and rotate_hessian_called:
        raise RuntimeError(
            "double-rotation: faithful-acts builds H in rotated space; "
            "rotate_hessian must be skipped.")

def fp8_weight_override(w, group_size: int = 32):
    """Quant->dequant a weight through the scaled-FP8 tier (Exec T2). Returns the
    dequantized fp32 weight to install as a forward-time override."""
    from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8
    return dequantize_scaled_fp8(quantize_scaled_fp8(w.float(), group_size=group_size)).to(w.dtype)

@torch.no_grad()
def collect_hessians_single_pass(hooks_by_index, calib, model, device):
    """Collect every target's faithful Hessian in ONE forward pass.

    `hooks_by_index`: {target_index: FaithfulActHook}. All hooks are already
    installed as forward pre-hooks and `enabled` (they transform activations on
    every forward regardless of target state). We reset + target ALL of them,
    run one forward over `calib`, then untarget and return each hook's H.

    This is STATIC-Hessian GPTQ: every target's H is built against the ORIGINAL
    (unquantized) model in a single pass. It is NOT bit-identical to the dense
    per-target loop, which is TRUE-SEQUENTIAL — that path writes each quantized
    weight back (weight_override / weight.data) so the next target's H sees the
    quantized upstream (GPTQ cross-layer error propagation). The two agree only
    when cross-layer propagation is negligible; validate by PPL, not byte-diff.
    Returns {index: (H, n_tokens)}.
    """
    for hk in hooks_by_index.values():
        hk.reset_hessian()
        hk.set_hessian_target(True)
    for ids in calib:
        model(ids.to(device))
    for hk in hooks_by_index.values():
        hk.set_hessian_target(False)
    return {i: (hk.H, hk.n_tokens) for i, hk in hooks_by_index.items()}
