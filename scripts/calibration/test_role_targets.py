import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# gguf-py for the authoritative TensorNameMap the classifier now delegates to
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from role_targets import classify_role, Tier, assert_main_stack_covered, configure

# Names below are taken from the ACTUAL Qwen3.5-9B/0.8B checkpoints' named_modules()
# (verified 2026-05-31), not invented. classify_role now resolves them through
# llama.cpp's authoritative TensorNameMap (same map convert_hf_to_gguf uses), so this
# test also proves our tier table lines up with the real HF->GGUF name resolution.
configure("qwen35", 32)

def test_ml8_gemm_roles():
    cases = {
        # full-attention block (self_attn.*)
        "model.language_model.layers.3.self_attn.q_proj":   ("blk.3.attn_q.weight",      "attn_q",   Tier.ML8),
        "model.language_model.layers.3.self_attn.k_proj":   ("blk.3.attn_k.weight",      "attn_k",   Tier.ML8),
        "model.language_model.layers.3.self_attn.v_proj":   ("blk.3.attn_v.weight",      "attn_v",   Tier.ML8),
        "model.language_model.layers.3.self_attn.o_proj":   ("blk.3.attn_output.weight", "attn_out", Tier.ML8),
        # gated delta-net block (linear_attn.*)
        "model.language_model.layers.0.linear_attn.in_proj_qkv": ("blk.0.attn_qkv.weight",  "attn_qkv",  Tier.ML8),
        "model.language_model.layers.0.linear_attn.in_proj_z":   ("blk.0.attn_gate.weight", "attn_gate", Tier.ML8),
        "model.language_model.layers.0.linear_attn.out_proj":    ("blk.0.ssm_out.weight",   "ssm_out",   Tier.ML8),
        # dense MLP (mlp.*)
        "model.language_model.layers.0.mlp.gate_proj": ("blk.0.ffn_gate.weight", "ffn_gate", Tier.ML8),
        "model.language_model.layers.0.mlp.up_proj":   ("blk.0.ffn_up.weight",   "ffn_up",   Tier.ML8),
        "model.language_model.layers.0.mlp.down_proj": ("blk.0.ffn_down.weight", "ffn_down", Tier.ML8),
        # global (lm_head / tied output)
        "lm_head":                                     ("output.weight",         "lm_head",  Tier.ML8),
    }
    for hf, expected in cases.items():
        assert classify_role(hf) == expected, f"{hf} -> {classify_role(hf)} (expected {expected})"

def test_scaled_fp8_roles():
    # alpha = in_proj_a, beta = in_proj_b (Design A tier 2 = 8-bit FP8)
    assert classify_role("model.language_model.layers.2.linear_attn.in_proj_a") == ("blk.2.ssm_alpha.weight", "ssm_alpha", Tier.FP8)
    assert classify_role("model.language_model.layers.2.linear_attn.in_proj_b") == ("blk.2.ssm_beta.weight",  "ssm_beta",  Tier.FP8)
    assert classify_role("model.embed_tokens")[2] is Tier.FP8

def test_native_left_alone():
    # SSM core + norms are NOT quantized (and are not nn.Linear at runtime)
    for n in ("model.language_model.layers.0.linear_attn.A_log",
              "model.language_model.layers.0.linear_attn.conv1d",
              "model.language_model.layers.0.linear_attn.dt_bias",
              "model.language_model.layers.0.input_layernorm",
              "model.language_model.layers.0.self_attn.q_norm"):
        assert classify_role(n)[2] is Tier.NATIVE, f"{n} should be NATIVE"

def _mock_block(*, drift=False):
    import torch.nn as nn
    blk = nn.Module(); la = nn.Module()
    la.in_proj_qkv = nn.Linear(8, 8, bias=False)
    la.in_proj_z   = nn.Linear(8, 8, bias=False)
    la.out_proj    = nn.Linear(8, 8, bias=False)
    # alpha: real name in_proj_a, or the OLD invented name when simulating drift
    setattr(la, "alpha_proj" if drift else "in_proj_a", nn.Linear(8, 8, bias=False))
    la.in_proj_b   = nn.Linear(8, 8, bias=False)
    blk.linear_attn = la
    mlp = nn.Module()
    mlp.gate_proj = nn.Linear(8, 8, bias=False)
    mlp.up_proj   = nn.Linear(8, 8, bias=False)
    mlp.down_proj = nn.Linear(8, 8, bias=False)
    blk.mlp = mlp
    return blk

def _mock_model(*, drift=False):
    import torch.nn as nn
    root = nn.Module(); wrapper = nn.Module(); lm = nn.Module()
    lm.layers = nn.ModuleList([_mock_block(drift=drift)])
    wrapper.language_model = lm; root.model = wrapper   # -> model.language_model.layers.0.…
    return root

def test_guard_passes_on_real_names():
    n = assert_main_stack_covered(_mock_model(drift=False))
    assert n >= 6, f"expected >=6 covered linears, got {n}"

def test_guard_raises_on_drift():
    try:
        assert_main_stack_covered(_mock_model(drift=True))
    except ValueError as e:
        assert "NATIVE" in str(e) and "alpha_proj" in str(e), f"unexpected msg: {e}"
        return
    raise AssertionError("guard did NOT raise on a drifted (uncovered) main-stack Linear")

if __name__ == "__main__":
    test_ml8_gemm_roles(); test_scaled_fp8_roles(); test_native_left_alone()
    test_guard_passes_on_real_names(); test_guard_raises_on_drift()
    print("ALL ROLE TESTS PASSED")
