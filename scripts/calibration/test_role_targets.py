import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from role_targets import classify_role, Tier

def test_ml8_gemm_roles():
    cases = {
        "model.layers.0.self_attn.q_proj":   ("blk.0.attn_q.weight",     "attn_q",   Tier.ML8),
        "model.layers.3.self_attn.k_proj":   ("blk.3.attn_k.weight",     "attn_k",   Tier.ML8),
        "model.layers.3.self_attn.v_proj":   ("blk.3.attn_v.weight",     "attn_v",   Tier.ML8),
        "model.layers.3.self_attn.o_proj":   ("blk.3.attn_output.weight","attn_out", Tier.ML8),
        "model.layers.5.linear_attn.out_proj":("blk.5.ssm_out.weight",   "ssm_out",  Tier.ML8),
        "lm_head":                            ("output.weight",          "lm_head",  Tier.ML8),
    }
    for hf, (gguf, role, tier) in cases.items():
        assert classify_role(hf) == (gguf, role, tier), f"{hf} -> {classify_role(hf)}"

def test_scaled_fp8_roles():
    assert classify_role("model.embed_tokens")[2] is Tier.FP8
    assert classify_role("model.layers.2.linear_attn.alpha_proj")[2] is Tier.FP8
    assert classify_role("model.layers.2.linear_attn.beta_proj")[2] is Tier.FP8

def test_native_left_alone():
    assert classify_role("model.layers.0.linear_attn.conv1d")[2] is Tier.NATIVE
    assert classify_role("model.layers.0.input_layernorm")[2] is Tier.NATIVE

if __name__ == "__main__":
    test_ml8_gemm_roles(); test_scaled_fp8_roles(); test_native_left_alone()
    print("ALL ROLE TESTS PASSED")
