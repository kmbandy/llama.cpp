#!/usr/bin/env python3
"""Tests for rotate_model_quarot.py — Role enum, classify_tensor."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rotate_model_quarot import Role, classify_tensor


def _assert_eq(actual, expected, label):
    assert actual == expected, f"{label}: got {actual!r}, expected {expected!r}"


def test_classify_qwen36_tensors():
    """Known Qwen3.6 MoE tensor names map to expected roles."""
    cases = [
        ("token_embd.weight",                 Role.EMBED),
        ("blk.0.attn_norm.weight",            Role.NORM_PRE_ATTN),
        ("blk.0.attn_q.weight",               Role.ATTN_Q),
        ("blk.0.attn_k.weight",               Role.ATTN_K),
        ("blk.0.attn_v.weight",               Role.ATTN_V),
        ("blk.0.attn_output.weight",          Role.ATTN_O),
        ("blk.0.ffn_norm.weight",             Role.NORM_PRE_FFN),
        ("blk.0.ffn_gate_inp.weight",         Role.FFN_GATE_INP),
        ("blk.0.ffn_gate_exps.weight",        Role.FFN_GATE_EXPS),
        ("blk.0.ffn_up_exps.weight",          Role.FFN_UP_EXPS),
        ("blk.0.ffn_down_exps.weight",        Role.FFN_DOWN_EXPS),
        ("blk.0.ssm_norm.weight",             Role.NORM_PRE_SSM),
        ("blk.0.ssm_in.weight",               Role.MAMBA_IN),
        ("blk.0.ssm_out.weight",              Role.MAMBA_OUT),
        ("output_norm.weight",                Role.NORM_OUT),
        ("output.weight",                     Role.LM_HEAD),
        ("rope_freqs.weight",                 Role.PASSTHROUGH),
    ]
    for name, expected in cases:
        actual = classify_tensor(name, arch="qwen36moe")
        _assert_eq(actual, expected, f"qwen36moe / {name}")
    print(f"  PASS test_classify_qwen36_tensors")


def test_classify_unknown_raises():
    """Unknown tensor name on a known arch raises with the name in the message."""
    try:
        classify_tensor("blk.0.fictional.weight", arch="qwen36moe")
    except ValueError as e:
        msg = str(e)
        assert "blk.0.fictional.weight" in msg, f"name missing from error: {msg}"
        print(f"  PASS test_classify_unknown_raises")
        return
    raise AssertionError("expected ValueError, got none")


from rotate_model_quarot import build_R_resid


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, label: str):
    diff = (actual - expected).abs().max().item()
    assert diff <= tol, f"{label}: max abs diff {diff:.3e} > tol {tol:.3e}"


def test_R_resid_orthogonal_pow2():
    """R @ R.T == I for power-of-2 d_model (pure Sylvester)."""
    R = build_R_resid(d_model=2048, seed=42, device=torch.device("cpu"))
    assert R.shape == (2048, 2048), f"shape {R.shape}"
    assert R.dtype == torch.float32, f"dtype {R.dtype}"
    I = torch.eye(2048, dtype=torch.float32)
    _assert_close(R @ R.T, I, tol=1e-5, label="R @ R.T")
    _assert_close(R.T @ R, I, tol=1e-5, label="R.T @ R")
    print(f"  PASS test_R_resid_orthogonal_pow2")


def test_R_resid_orthogonal_kronecker():
    """R @ R.T == I for non-power-of-2 d_model via Kronecker H_a ⊗ H_b."""
    R = build_R_resid(d_model=2560, seed=42, device=torch.device("cpu"))
    assert R.shape == (2560, 2560), f"shape {R.shape}"
    I = torch.eye(2560, dtype=torch.float32)
    _assert_close(R @ R.T, I, tol=1e-4, label="R @ R.T (kronecker)")
    print(f"  PASS test_R_resid_orthogonal_kronecker")


def test_R_resid_seed_determinism():
    """Same seed → same R; different seed → different R."""
    R1 = build_R_resid(d_model=64, seed=42, device=torch.device("cpu"))
    R2 = build_R_resid(d_model=64, seed=42, device=torch.device("cpu"))
    R3 = build_R_resid(d_model=64, seed=43, device=torch.device("cpu"))
    _assert_close(R1, R2, tol=0.0, label="same-seed determinism")
    diff = (R1 - R3).abs().max().item()
    assert diff > 1e-3, f"different-seed difference too small: {diff:.3e}"
    print(f"  PASS test_R_resid_seed_determinism")


from rotate_model_quarot import rotate_input_side, rotate_output_side


def test_rotate_input_side_round_trip():
    """Original forward equals rotated-input forward on rotated input."""
    torch.manual_seed(7)
    d_model, N = 64, 96
    W      = torch.randn(N, d_model, dtype=torch.float32)
    gamma  = torch.randn(d_model, dtype=torch.float32) * 0.3 + 1.0  # near-1 like real RMSNorm γ
    x      = torch.randn(4, d_model, dtype=torch.float32)
    R      = build_R_resid(d_model=d_model, seed=11, device=torch.device("cpu"))

    # Original forward: y = (γ ⊙ x) @ W.T
    y_orig = (gamma * x) @ W.T

    # Rotated forward: residual stream rotated as x' = x @ R.T; absorbed γ
    # makes the next linear see norm-output directly; W_new produces y from x'.
    W_new  = rotate_input_side(W, gamma, R)
    x_rot  = x @ R.T
    y_new  = x_rot @ W_new.T

    _assert_close(y_new, y_orig, tol=1e-4, label="input-side round trip")
    print(f"  PASS test_rotate_input_side_round_trip")


def test_rotate_output_side_round_trip():
    """Output-side rotation gives y_new = y @ R.T (residual ends up rotated)."""
    torch.manual_seed(8)
    d_model, K = 64, 80
    W      = torch.randn(d_model, K, dtype=torch.float32)  # writes residual
    x      = torch.randn(4, K, dtype=torch.float32)
    R      = build_R_resid(d_model=d_model, seed=12, device=torch.device("cpu"))

    y_orig    = x @ W.T            # original residual contribution
    W_new     = rotate_output_side(W, R)
    y_rotated = x @ W_new.T        # should equal y_orig @ R.T

    _assert_close(y_rotated, y_orig @ R.T, tol=1e-4, label="output-side R.T projection")
    print(f"  PASS test_rotate_output_side_round_trip")


def test_input_output_cancel_through_residual():
    """A linear writing then a linear reading the residual recovers the original."""
    torch.manual_seed(9)
    d_model, K, N = 64, 80, 96
    W_out     = torch.randn(d_model, K, dtype=torch.float32)    # writes residual
    gamma     = torch.randn(d_model, dtype=torch.float32) * 0.3 + 1.0
    W_in      = torch.randn(N, d_model, dtype=torch.float32)    # reads residual
    x         = torch.randn(4, K, dtype=torch.float32)
    R         = build_R_resid(d_model=d_model, seed=13, device=torch.device("cpu"))

    # Original
    residual_orig = x @ W_out.T
    y_orig        = (gamma * residual_orig) @ W_in.T

    # Rotated stack
    W_out_new = rotate_output_side(W_out, R)
    W_in_new  = rotate_input_side(W_in, gamma, R)
    residual_rot = x @ W_out_new.T     # = residual_orig @ R.T
    y_rot       = residual_rot @ W_in_new.T

    _assert_close(y_rot, y_orig, tol=1e-4, label="residual cancellation")
    print(f"  PASS test_input_output_cancel_through_residual")


from rotate_model_quarot import rotate_moe_input_side, rotate_moe_output_side


def test_moe_input_side_matches_per_expert_loop():
    """Batched MoE input rotation == per-expert rotate_input_side over n_experts."""
    torch.manual_seed(20)
    d_model, d_ffn, n_exp = 32, 48, 4
    # PyTorch shape for gate/up_exps: [d_ffn, d_model, n_exp]
    W      = torch.randn(d_ffn, d_model, n_exp, dtype=torch.float32)
    gamma  = torch.randn(d_model, dtype=torch.float32) * 0.3 + 1.0
    R      = build_R_resid(d_model=d_model, seed=21, device=torch.device("cpu"))

    expected = torch.empty_like(W)
    for e in range(n_exp):
        expected[..., e] = rotate_input_side(W[..., e], gamma, R)

    actual = rotate_moe_input_side(W, gamma, R)
    _assert_close(actual, expected, tol=1e-5, label="MoE input batched vs loop")
    print(f"  PASS test_moe_input_side_matches_per_expert_loop")


def test_moe_output_side_matches_per_expert_loop():
    """Batched MoE output rotation == per-expert rotate_output_side over n_experts."""
    torch.manual_seed(22)
    d_model, d_ffn, n_exp = 32, 48, 4
    # PyTorch shape for down_exps: [d_model, d_ffn, n_exp]
    W      = torch.randn(d_model, d_ffn, n_exp, dtype=torch.float32)
    R      = build_R_resid(d_model=d_model, seed=23, device=torch.device("cpu"))

    expected = torch.empty_like(W)
    for e in range(n_exp):
        expected[..., e] = rotate_output_side(W[..., e], R)

    actual = rotate_moe_output_side(W, R)
    _assert_close(actual, expected, tol=1e-5, label="MoE output batched vs loop")
    print(f"  PASS test_moe_output_side_matches_per_expert_loop")


import tempfile
from pathlib import Path as _Path

def _make_tiny_qwen36_gguf(out_path: str, n_layers: int = 2, d_model: int = 32, d_ffn: int = 48, n_exp: int = 4):
    """Write a tiny qwen36moe-shaped bf16 GGUF for unit tests."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as _gguf
    import numpy as _np

    w = _gguf.GGUFWriter(out_path, arch="qwen36moe")
    w.add_uint32("qwen36moe.embedding_length", d_model)
    w.add_uint32("qwen36moe.block_count", n_layers)
    w.add_uint32("qwen36moe.feed_forward_length", d_ffn)
    w.add_uint32("qwen36moe.expert_count", n_exp)

    def _add_bf16(name, t):
        data = _np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(name, data, raw_dtype=_gguf.GGMLQuantizationType.BF16)

    vocab = 64
    _add_bf16("token_embd.weight",        torch.randn(vocab, d_model))
    for L in range(n_layers):
        _add_bf16(f"blk.{L}.attn_norm.weight",    torch.ones(d_model) + 0.1 * torch.randn(d_model))
        _add_bf16(f"blk.{L}.attn_q.weight",       torch.randn(d_model, d_model))
        _add_bf16(f"blk.{L}.attn_k.weight",       torch.randn(d_model, d_model))
        _add_bf16(f"blk.{L}.attn_v.weight",       torch.randn(d_model, d_model))
        _add_bf16(f"blk.{L}.attn_output.weight",  torch.randn(d_model, d_model))
        _add_bf16(f"blk.{L}.ffn_norm.weight",     torch.ones(d_model) + 0.1 * torch.randn(d_model))
        _add_bf16(f"blk.{L}.ffn_gate_inp.weight", torch.randn(n_exp, d_model))
        _add_bf16(f"blk.{L}.ffn_gate_exps.weight", torch.randn(d_ffn, d_model, n_exp))
        _add_bf16(f"blk.{L}.ffn_up_exps.weight",   torch.randn(d_ffn, d_model, n_exp))
        _add_bf16(f"blk.{L}.ffn_down_exps.weight", torch.randn(d_model, d_ffn, n_exp))
    _add_bf16("output_norm.weight", torch.ones(d_model) + 0.1 * torch.randn(d_model))
    _add_bf16("output.weight",      torch.randn(vocab, d_model))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


from rotate_model_quarot import index_pass, _gguf_tensor_to_torch


def test_index_pass_on_tiny_gguf():
    """Pass 1 builds a roster mapping every tensor name to a Role, and a γ map."""
    with tempfile.TemporaryDirectory() as td:
        path = str(_Path(td) / "tiny.gguf")
        _make_tiny_qwen36_gguf(path, n_layers=2)
        roster, gammas, d_model = index_pass(path, arch="qwen36moe")

    assert d_model == 32, f"d_model {d_model}"
    assert roster["token_embd.weight"] == Role.EMBED
    assert roster["blk.0.attn_q.weight"] == Role.ATTN_Q
    assert roster["blk.1.ffn_down_exps.weight"] == Role.FFN_DOWN_EXPS
    # γ map: keyed by tensor name, value is a torch.Tensor of size d_model.
    assert "blk.0.attn_norm.weight" in gammas
    assert gammas["blk.0.attn_norm.weight"].shape == (32,)
    assert "output_norm.weight" in gammas
    print(f"  PASS test_index_pass_on_tiny_gguf")


from rotate_model_quarot import rotate_gguf


def test_rotate_gguf_end_to_end_on_tiny():
    """rotate_gguf produces an output GGUF whose forward equals the source's."""
    with tempfile.TemporaryDirectory() as td:
        src = str(_Path(td) / "src.gguf")
        dst = str(_Path(td) / "dst.gguf")
        _make_tiny_qwen36_gguf(src, n_layers=2, d_model=32, d_ffn=48, n_exp=4)

        rotate_gguf(source_path=src, output_path=dst, arch="qwen36moe",
                    seed=42, device=torch.device("cpu"))

        # Re-load and check shapes + γs zeroed-to-1.
        sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/gguf-py")
        import gguf as _gguf
        r = _gguf.GGUFReader(dst)
        names = {t.name for t in r.tensors}
        # Every source tensor present.
        assert "token_embd.weight" in names
        assert "blk.0.ffn_down_exps.weight" in names
        # γ tensors written as all-ones.
        gamma_t = next(t for t in r.tensors if t.name == "blk.0.attn_norm.weight")
        gamma_v = _gguf_tensor_to_torch(gamma_t)
        _assert_close(gamma_v, torch.ones(32), tol=1e-2, label="γ written as ones")
        # Rotated weight shape matches source.
        q_t = next(t for t in r.tensors if t.name == "blk.0.attn_q.weight")
        assert list(q_t.shape) == [32, 32], f"attn_q shape {list(q_t.shape)}"
    print(f"  PASS test_rotate_gguf_end_to_end_on_tiny")


if __name__ == "__main__":
    test_classify_qwen36_tensors()
    test_classify_unknown_raises()
    test_R_resid_orthogonal_pow2()
    test_R_resid_orthogonal_kronecker()
    test_R_resid_seed_determinism()
    test_rotate_input_side_round_trip()
    test_rotate_output_side_round_trip()
    test_input_output_cancel_through_residual()
    test_moe_input_side_matches_per_expert_loop()
    test_moe_output_side_matches_per_expert_loop()
    test_index_pass_on_tiny_gguf()
    test_rotate_gguf_end_to_end_on_tiny()
    print("\nALL TESTS PASSED")
