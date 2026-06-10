"""Stub-level tests for act_replay_student (Task 5, act-replay KL trainer).

No HF model required: we build tiny nn.Linear modules and ml8 state dicts by
hand. Run from scripts/calibration with PYTHONPATH=../../gguf-py.
"""
import torch
import torch.nn as nn

from act_replay_student import AttachedTarget, attach_to_linear, select_targets
from centroid_quantizer import snap_to_e4m3
from kronecker_rotation import KroneckerRotation, sylvester


def _mk_state(N=8, K=128, G=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = torch.randn(G, 16, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    return {"indices": torch.randint(0, 16, (N, K), generator=g),
            "scales": torch.rand(N, G, generator=g) + 0.1,
            "centroids": cent, "rotation": None}


def _ref_weight(t):
    """Reference dequant W[r,c] = cent[gidx[c], idx[r,c]] * scl[r, gidx[c]]."""
    N, K = t["indices"].shape
    G = t["centroids"].shape[0]
    gidx = torch.arange(K) // (K // G)
    return t["centroids"][gidx, t["indices"]] * t["scales"][:, gidx]


def test_step0_bit_equal():
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    attach_to_linear(lin, t)
    W = _ref_weight(t)
    x = torch.randn(3, 128)
    assert torch.equal(lin(x), x @ W.t())


def test_indices_buffer_is_uint8():
    """Regression: AttachedTarget stores indices as a uint8 buffer (8x smaller
    than long), and step-0 dequant is still bit-exact against the reference."""
    t = _mk_state()  # _mk_state produces int64 indices (torch.randint default)
    lin = nn.Linear(128, 8, bias=False)
    at = attach_to_linear(lin, t)
    assert at.indices.dtype == torch.uint8
    W = _ref_weight(t)
    x = torch.randn(3, 128)
    assert torch.equal(lin(x), x @ W.t())


def test_grads_only_cent_scl():
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    w0 = lin.weight.detach().clone()
    at = attach_to_linear(lin, t)
    lin(torch.randn(3, 128)).sum().backward()
    assert at.centroids.grad is not None and at.scales.grad is not None
    # The original Linear weight must remain an untouched, non-trained leaf.
    assert lin.weight.grad is None and torch.equal(lin.weight.detach(), w0)


def test_ste_off_lattice():
    """After an off-e4m3-lattice nudge to centroids, the forward must use the
    SNAPPED centroids (snap in forward), and backward must still flow a gradient
    into the raw centroids (identity-grad straight-through)."""
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    at = attach_to_linear(lin, t)
    with torch.no_grad():
        at.centroids += 0.003  # nudge off the e4m3 lattice

    N, K = t["indices"].shape
    G = t["centroids"].shape[0]
    gidx = torch.arange(K) // (K // G)
    W = snap_to_e4m3(at.centroids)[gidx, t["indices"]] * at.scales[:, gidx]

    x = torch.randn(2, 128)
    assert torch.equal(lin(x), x @ W.t())

    # STE: gradient reaches the (off-lattice) raw centroids despite the snap.
    at.centroids.grad = None
    lin(torch.randn(2, 128)).sum().backward()
    assert at.centroids.grad is not None


def test_bf16_host_dtype_cast():
    """BUG 1 regression: a bf16 host nn.Linear + bf16 activations must run the
    attached forward without an fp32/bf16 dtype crash in F.linear. The fp32
    centroids/scales are cast to the activation dtype AFTER the STE dequant, so
    gradients still flow back to the fp32 leaves and stay finite (CPU bf16 OK)."""
    t = _mk_state(N=8, K=128)
    lin = nn.Linear(128, 8, bias=True).to(torch.bfloat16)  # bf16 host weight + bias
    at = attach_to_linear(lin, t)

    x = torch.randn(3, 128, dtype=torch.bfloat16)
    y = lin(x)                                  # must not raise on dtype mismatch
    assert y.dtype == torch.bfloat16

    # backward flows into the fp32 codebook leaves; grads are finite.
    y.sum().backward()
    assert at.centroids.grad is not None and at.scales.grad is not None
    assert torch.isfinite(at.centroids.grad).all()
    assert torch.isfinite(at.scales.grad).all()
    # codebook leaves stay fp32 despite the bf16 forward cast.
    assert at.centroids.dtype == torch.float32 and at.scales.dtype == torch.float32


def test_faithful_acts():
    """With a rotation present, the forward must apply the W4A8 faithful path:
    x_eff = quantize_act_per_row(x @ Q) @ Q.T, then F.linear(x_eff, W)."""
    from ml8_e4m3_sim import quantize_act_per_row

    t = _mk_state()
    h_a = torch.eye(4)
    b_dim = 32
    t["rotation"] = {"h_a": h_a, "a_dim": 4, "b_dim": b_dim}

    lin = nn.Linear(128, 8, bias=False)
    attach_to_linear(lin, t)  # auto-faithful: rotation present

    rot = KroneckerRotation(h_a=h_a, b_dim=b_dim)
    W = _ref_weight(t)
    x = torch.randn(3, 128)

    x_eff = rot.inverse(quantize_act_per_row(rot.forward(x)))
    expected = x_eff @ W.t()
    assert torch.allclose(lin(x), expected, atol=1e-6)


def test_attach_rotation_dim_mismatch_raises():
    """attach_to_linear must raise ValueError when rotation a_dim*b_dim != in_features."""
    t = _mk_state(N=8, K=128)
    # 4 * 16 = 64, but lin.in_features = 128 — mismatch
    t["rotation"] = {"h_a": torch.eye(4), "a_dim": 4, "b_dim": 16}
    lin = nn.Linear(128, 8, bias=False)
    try:
        attach_to_linear(lin, t)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "rotation dim mismatch" in str(exc)


def test_select_targets_skip_down():
    names = [
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        "blk.1.ffn_gate.weight", "blk.1.ffn_down.weight",
    ]
    sel = select_targets(names, train="ml8", skip="*ffn_down*")
    assert set(sel) == {
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.1.ffn_gate.weight",
    }


def test_select_targets_train_glob():
    names = [
        "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",
        "blk.1.ffn_gate.weight", "blk.1.ffn_down.weight",
    ]
    sel = select_targets(names, train="blk.0*", skip=None)
    assert set(sel) == {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"}
