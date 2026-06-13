"""Stub-level tests for act_replay_student (Task 5, act-replay KL trainer).

No HF model required: we build tiny nn.Linear modules and ml8 state dicts by
hand. Run from scripts/calibration with PYTHONPATH=../../gguf-py.
"""
import torch
import torch.nn as nn

from act_replay_student import AttachedTarget, attach_to_linear, select_targets
from centroid_quantizer import snap_to_e4m3
from kronecker_rotation import KroneckerRotation, random_orthogonal
from ml8_e4m3_sim import quantize_act_per_row


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
    """With a rotation present, the forward must apply the W4A8 faithful path in
    the ROTATED basis (NO inverse): x_eff = quantize_act_per_row(x @ Q), then
    F.linear(x_eff, W_rot). The weight is rehydrated from GGUF already rotated,
    so the deployed kernel is y = e4m3(x@Q) @ W_rot.T with no derotation."""
    t = _mk_state()
    h_a = torch.eye(4)
    b_dim = 32
    t["rotation"] = {"h_a": h_a, "a_dim": 4, "b_dim": b_dim}

    lin = nn.Linear(128, 8, bias=False)
    attach_to_linear(lin, t)  # auto-faithful: rotation present

    rot = KroneckerRotation(h_a=h_a, b_dim=b_dim)
    W = _ref_weight(t)  # this IS W_rot (the GGUF-dequant'd rotated weight)
    x = torch.randn(3, 128)

    x_eff = quantize_act_per_row(rot.forward(x))  # NO inverse
    expected = x_eff @ W.t()
    assert torch.allclose(lin(x), expected, atol=1e-6)


def test_faithful_acts_rotated_basis_no_inverse():
    """Regression for the KL 12.15 fingerprint: AttachedTarget weights rehydrated
    from GGUF are in the ROTATED basis (W_rot), so the deployed kernel computes
    y = e4m3(x @ Q) @ W_rot.T with NO inverse rotation. Keeping the inverse (as
    calibration's FaithfulActHook did, because it had overridden the weight with
    the UNROTATED W_unrot) leaves W un-derotated and blows up holdout KL.

    Two assertions:
      1. EXACT (fp32) equality to the rotated-basis kernel math.
      2. allclose to the calibration-faithful identity
         e4m3(x@Q) @ Q.T @ W_unrot.T == e4m3(x@Q) @ W_rot.T, with
         W_unrot = W_rot @ Q.T."""
    h_a = random_orthogonal(4, seed=0)
    b_dim = 32
    t = _mk_state()
    t["rotation"] = {"h_a": h_a, "a_dim": 4, "b_dim": b_dim}

    lin = nn.Linear(128, 8, bias=False)
    attach_to_linear(lin, t)  # auto-faithful: rotation present

    rot = KroneckerRotation(h_a=h_a, b_dim=b_dim)
    W_rot = _ref_weight(t)  # GGUF dequant — already rotated basis
    x = torch.randn(3, 128)

    a_q = quantize_act_per_row(rot.forward(x))  # rotated-basis quantized acts

    # (1) EXACT: deployed kernel y = e4m3(x@Q) @ W_rot.T, no inverse.
    reference = a_q @ W_rot.t()
    assert torch.equal(lin(x), reference)

    # (2) Identity: calibration's hook (inverse + UNROTATED weight) is equivalent.
    # W_unrot = W_rot @ Q.T = rot.inverse(W_rot) (inverse right-multiplies by Q.T).
    W_unrot = rot.inverse(W_rot)
    calib_faithful = rot.inverse(a_q) @ W_unrot.t()
    assert torch.allclose(reference, calib_faithful, atol=1e-4)


def test_faithful_acts_with_input_column_reorder():
    """ml8 target needing a GGUF(tiled)->HF(grouped) INPUT-column reorder — the
    linear-attn out_proj V-head reorder on asymmetric-head models (4B: 32 value
    vs 16 key heads). The reorder is handled by permuting the INPUT ACTIVATION
    hf->tiled at the FRONT of apply_acts, leaving W/Q/scales/centroids/indices in
    pristine GGUF order: per-row act-quant commutes with a column permutation and
    the Kronecker rotation is never conjugated. The student fed HF-order input must
    reproduce the deployment kernel fed GGUF-order input EXACTLY.

    Uses a NON-involutive (3-cycle) column permutation so a wrong perm direction
    (forgetting the argsort inverse) is caught — an involutive swap would pass even
    with the inverse dropped.
    """
    K, N, G = 192, 8, 3
    a_dim, b_dim = 6, 32  # a_dim*b_dim = 192 = K
    h_a = random_orthogonal(a_dim, seed=2)
    t = _mk_state(N=N, K=K, G=G, seed=3)
    t["rotation"] = {"h_a": h_a, "a_dim": a_dim, "b_dim": b_dim}

    # tiled->hf column index (gguf_to_hf_perm convention: W_hf = W_gguf[:, index]).
    # 3-cycle on the three 64-wide groups: hf = [group2, group0, group1].
    g0 = torch.arange(0, 64); g1 = torch.arange(64, 128); g2 = torch.arange(128, 192)
    index = torch.cat([g2, g0, g1]).long()
    assert not torch.equal(torch.argsort(index), index)  # genuinely non-involutive

    # The target carries the INVERSE (hf->tiled) ACTIVATION permutation.
    t["col_perm"] = torch.argsort(index)

    lin = nn.Linear(K, N, bias=False)
    attach_to_linear(lin, t)  # auto-faithful: rotation present

    rot = KroneckerRotation(h_a=h_a, b_dim=b_dim)
    W_gguf = _ref_weight(t)                       # GGUF-order rotated weight
    x_gguf = torch.randn(3, K)
    reference = quantize_act_per_row(rot.forward(x_gguf)) @ W_gguf.t()

    x_hf = x_gguf.index_select(-1, index)         # what the HF module emits
    assert torch.equal(lin(x_hf), reference)


def test_attach_one_input_reorder_routes_to_activation_perm():
    """_attach_one must NOT reject an axis-1 (input-column) V-head reorder on an
    ml8 target. It converts gguf_to_hf_perm's tiled->hf index into the inverse
    (hf->tiled) ACTIVATION permutation carried on the AttachedTarget, leaving the
    ml8 weight/scales/centroids untouched in GGUF order (applied in apply_acts)."""
    from types import SimpleNamespace

    from act_replay import _attach_one, gguf_to_hf_perm, map_gguf_to_hf

    K, N, G = 96, 8, 3
    a_dim, b_dim = 3, 32  # a_dim*b_dim = 96 = K
    # 6 value vs 2 key heads (num_v_per_k=3) -> a NON-involutive V-head reorder.
    cfg = SimpleNamespace(linear_num_value_heads=6, linear_num_key_heads=2,
                          linear_value_head_dim=16, linear_key_head_dim=16)
    gname = "blk.0.ssm_out.weight"
    ax, index = gguf_to_hf_perm(gname, (N, K), cfg)
    assert ax == 1 and not torch.equal(torch.argsort(index), index)

    t = _mk_state(N=N, K=K, G=G, seed=5)
    t["rotation"] = {"h_a": random_orthogonal(a_dim, seed=5),
                     "a_dim": a_dim, "b_dim": b_dim}
    modules = {map_gguf_to_hf(gname): nn.Linear(K, N, bias=False)}

    at = _attach_one(modules, gname, t, cfg)      # must NOT raise on axis-1
    assert at.col_perm is not None
    assert torch.equal(at.col_perm.cpu(), torch.argsort(index))


def test_free_host_weight_releases_dead_bf16():
    """The host nn.Linear.weight is DEAD after attach — the patched forward uses the
    ml8 dequant (or fp8 engine), never lin.weight. Measured on the 4B: keeping it
    double-stores the model (bf16 + ml8) = ~7GB wasted. free_host_weight=True releases
    it (0-element Parameter) and the forward stays bit-exact via the ml8 dequant."""
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    W = _ref_weight(t)
    x = torch.randn(3, 128)
    expected = x @ W.t()
    at = attach_to_linear(lin, t, free_host_weight=True)
    assert lin.weight.numel() == 0              # dead bf16 weight released
    assert torch.equal(lin(x), expected)        # forward still exact via ml8 dequant


def test_free_host_weight_default_keeps_weight():
    """Default keeps the host weight intact (preserves callers/tests that inspect it)."""
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    w0 = lin.weight.detach().clone()
    attach_to_linear(lin, t)
    assert lin.weight.numel() == 128 * 8 and torch.equal(lin.weight.detach(), w0)


def test_keep_w_orig_false_skips_anchor():
    """W_orig (the [N,K] fp32 mse/pv reassign anchor) summed over all ml8 targets is
    ~the whole model in fp32 — a major VRAM cost that OOM'd the 4B smoke. Arms that
    never run the mse/pv E-step (frozen / gptq / gptq-interleave) must be able to
    skip it: keep_w_orig=False -> at.W_orig is None, and the dequant forward stays
    bit-exact (W_orig is only an mse anchor, never used in the forward path)."""
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    at = attach_to_linear(lin, t, keep_w_orig=False)
    assert at.W_orig is None
    W = _ref_weight(t)
    x = torch.randn(3, 128)
    assert torch.equal(lin(x), x @ W.t())


def test_keep_w_orig_default_true_preserves_anchor():
    """Default (keep_w_orig unset) still captures the W_orig anchor — the mse/pv
    reassign path depends on it, so the default must not regress."""
    t = _mk_state()
    lin = nn.Linear(128, 8, bias=False)
    at = attach_to_linear(lin, t)
    assert at.W_orig is not None and at.W_orig.shape == t["indices"].shape


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


import pytest
from test_act_replay_cli import _mk_state as _mk_state_cli


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_attach_fp8_forward_backprops_to_centroids():
    dev = "cuda"
    N, K, G = 32, 128, 2
    state = _mk_state_cli(N=N, K=K, G=G)
    lin = nn.Linear(K, N, bias=False).to(dev).to(torch.bfloat16)
    at = attach_to_linear(lin, state, fp8=True)
    x = torch.randn(16, K, device=dev, dtype=torch.bfloat16) * 0.3
    y = lin(x)                              # monkeypatched fp8 forward
    assert torch.isfinite(y).all()
    y.float().sum().backward()
    assert at.centroids.grad is not None and torch.isfinite(at.centroids.grad).all()
    assert at.scales.grad is not None and torch.isfinite(at.scales.grad).all()
    # W_orig anchor captured
    assert at.W_orig.shape == (N, K)
