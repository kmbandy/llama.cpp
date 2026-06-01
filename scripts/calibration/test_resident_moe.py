"""CPU toy-tests for ResidentMoeExperts (the no-pager MoE block).

Validates the module logic — routing forward, Hessian accumulation, and the
drop-in interface the calibration quant loop drives. The full GGUF-load + 35B
resident path validates on the MI300X (67 GB of experts don't fit locally).
"""
import torch
import torch.nn.functional as F

from resident_moe import ResidentMoeExperts


def _toy(seed=0):
    torch.manual_seed(seed)
    E, H, I = 4, 8, 16
    gate = torch.randn(E, I, H)
    up   = torch.randn(E, I, H)
    down = torch.randn(E, H, I)
    return E, H, I, gate, up, down


def _ref_moe(x, gate, up, down, idx, wts, act=F.silu):
    """Plain reference: per token, sum over its top-k experts."""
    T = x.shape[0]
    out = torch.zeros_like(x)
    for t in range(T):
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            g = x[t] @ gate[e].t()
            u = x[t] @ up[e].t()
            h = act(g) * u
            out[t] += wts[t, j] * (h @ down[e].t())
    return out


def test_forward_matches_reference():
    E, H, I, gate, up, down = _toy(1)
    mod = ResidentMoeExperts(gate, up, down, E, I, H, act_fn=F.silu)
    T, k = 6, 2
    torch.manual_seed(2)
    x = torch.randn(T, H)
    idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])      # [T, k]
    wts = torch.softmax(torch.randn(T, k), dim=-1)                    # [T, k]
    got = mod(x, idx, wts)
    ref = _ref_moe(x, gate, up, down, idx, wts)
    assert got.shape == (T, H)
    assert torch.allclose(got, ref, atol=1e-4), \
        f"forward != reference; max diff {(got-ref).abs().max().item():.3e}"
    print("PASS test_forward_matches_reference")


def test_hessian_accumulation():
    E, H, I, gate, up, down = _toy(3)
    mod = ResidentMoeExperts(gate, up, down, E, I, H, act_fn=F.silu)
    mod.reset_calibration_acc()
    mod.collect_pre_gate_up = True
    mod.collect_pre_down = True
    T, k = 10, 2
    torch.manual_seed(4)
    x = torch.randn(T, H)
    idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
    wts = torch.softmax(torch.randn(T, k), dim=-1)
    mod(x, idx, wts)
    # gate_up Hessian is XᵀX over the gate-input space (H), down Hessian over I.
    assert mod.pre_gate_up_acc is not None and mod.pre_gate_up_acc.shape == (H, H)
    assert mod.pre_down_acc is not None and mod.pre_down_acc.shape == (I, I)
    # symmetric PSD-ish (XᵀX)
    assert torch.allclose(mod.pre_gate_up_acc, mod.pre_gate_up_acc.t(), atol=1e-4)
    assert mod.pre_gate_up_n_tok == T * k          # each token routed to k experts
    assert mod.pre_down_n_tok == T * k
    print("PASS test_hessian_accumulation")


def test_interface_dropin():
    E, H, I, gate, up, down = _toy(5)
    mod = ResidentMoeExperts(gate, up, down, E, I, H)
    # properties return the resident stacks, right shapes
    assert mod.gate_proj.shape == (E, I, H)
    assert mod.up_proj.shape == (E, I, H)
    assert mod.down_proj.shape == (E, H, I)
    # release_cached is a no-op (no exception), reset clears accumulators
    mod.collect_pre_gate_up = True
    mod.pre_gate_up_acc = torch.ones(H, H)
    mod.pre_gate_up_n_tok = 5
    mod.release_cached()
    mod.reset_calibration_acc()
    assert mod.pre_gate_up_acc is None and mod.pre_gate_up_n_tok == 0
    assert mod.collect_pre_gate_up is False
    # default act_fn is silu
    assert mod.act_fn is F.silu
    print("PASS test_interface_dropin")


def test_survives_to_empty_semantics():
    """The expert stacks are plain attributes, so they are NOT in buffers()/
    parameters() — which is exactly why model.to_empty() leaves them intact."""
    E, H, I, gate, up, down = _toy(6)
    mod = ResidentMoeExperts(gate, up, down, E, I, H)
    assert len(list(mod.buffers())) == 0, "expert stacks must NOT be registered buffers"
    assert len(list(mod.parameters())) == 0, "expert stacks must NOT be parameters"
    # they're still reachable + usable
    assert mod._gate.shape == (E, I, H)
    print("PASS test_survives_to_empty_semantics")


if __name__ == "__main__":
    test_forward_matches_reference()
    test_hessian_accumulation()
    test_interface_dropin()
    test_survives_to_empty_semantics()
    print("\nALL RESIDENT-MOE TESTS PASSED")
