# scripts/calibration/test_faithful_single_pass.py
"""Single-pass dense Hessian collection must be BIT-IDENTICAL to the per-target
sequential approach, because every FaithfulActHook sees the same all-transforms-
active deterministic forward in both. Toy-model proof (CPU, fast)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
from faithful_forward import FaithfulActHook, collect_hessians_single_pass


def _toy():
    torch.manual_seed(0)
    d = 16
    model = nn.Sequential(nn.Linear(d, d, bias=False),
                          nn.Linear(d, d, bias=False),
                          nn.Linear(d, d, bias=False))
    targets = [(f"l{i}", model[i]) for i in range(3)]
    hooks = {}
    for i, (_n, lyr) in enumerate(targets):
        a, b = factor_for_dim(d, max_b=1024)
        rot = KroneckerRotation(h_a=random_orthogonal(a, seed=100 + i), b_dim=b)
        hk = FaithfulActHook(rot, enabled=True)
        lyr.register_forward_pre_hook(hk)
        hooks[i] = hk
    calib = [torch.randn(5, d) for _ in range(4)]
    return model, hooks, calib


def _sequential(model, hooks, calib):
    """Replicate today's per-target path: target ONE hook at a time, one forward
    over calib per target, read that hook's H."""
    out = {}
    for i, hk in hooks.items():
        for h in hooks.values():
            h.reset_hessian(); h.set_hessian_target(False)
        hk.reset_hessian(); hk.set_hessian_target(True)
        with torch.no_grad():
            for ids in calib:
                model(ids)
        hk.set_hessian_target(False)
        out[i] = (hk.H, hk.n_tokens)
    return out


def test_single_pass_hessians_bit_identical_to_sequential():
    model, hooks, calib = _toy()
    seq = _sequential(model, hooks, calib)
    one = collect_hessians_single_pass(hooks, calib, model, device="cpu")
    assert set(one) == set(seq)
    for i in seq:
        H_seq, n_seq = seq[i]
        H_one, n_one = one[i]
        assert n_one == n_seq
        assert torch.equal(H_one, H_seq), f"target {i} Hessian not bit-identical"


def test_single_pass_resets_stale_state():
    model, hooks, calib = _toy()
    for hk in hooks.values():
        hk.set_hessian_target(True)
        with torch.no_grad():
            model(calib[0])
        hk.set_hessian_target(False)
    one = collect_hessians_single_pass(hooks, calib, model, device="cpu")
    model2, hooks2, calib2 = _toy()
    seq = _sequential(model2, hooks2, calib2)
    for i in seq:
        assert torch.equal(one[i][0], seq[i][0]), f"target {i} leaked stale H"
