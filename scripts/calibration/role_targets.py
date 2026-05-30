import enum, re

class Tier(enum.Enum):
    ML8 = "ml8"        # 4-bit ml8 GEMM, full GPTQ pipeline
    FP8 = "fp8"        # 8-bit scaled-FP8, direct cast
    NATIVE = "native"  # leave as-is (A/dt/conv/norms handled by caller)

_ML8 = {
    "q_proj": ("attn_q", "attn_q"), "k_proj": ("attn_k", "attn_k"),
    "v_proj": ("attn_v", "attn_v"), "o_proj": ("attn_output", "attn_out"),
    "qkv_proj": ("attn_qkv", "attn_qkv"), "gate_proj_attn": ("attn_gate", "attn_gate"),
    "out_proj": ("ssm_out", "ssm_out"),
    "gate_proj": ("ffn_gate", "ffn_gate"), "up_proj": ("ffn_up", "ffn_up"),
    "down_proj": ("ffn_down", "ffn_down"),
}
_FP8 = {"alpha_proj": ("ssm_alpha", "ssm_alpha"), "beta_proj": ("ssm_beta", "ssm_beta")}

def _layer_idx(name):
    parts = name.split(".")
    try: return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError): return None

def classify_role(hf_name: str):
    """Return (gguf_name, role, Tier). NATIVE for anything we don't quantize."""
    if hf_name in ("lm_head", "model.lm_head"):
        return ("output.weight", "lm_head", Tier.ML8)
    if hf_name.endswith("eh_proj"):
        L = _layer_idx(hf_name)
        return (f"blk.{L}.nextn.eh_proj.weight" if L is not None else "nextn.eh_proj.weight",
                "eh_proj", Tier.ML8)
    if hf_name in ("model.embed_tokens", "model.embed_tokens.weight"):
        return ("token_embd.weight", "token_embd", Tier.FP8)
    L = _layer_idx(hf_name)
    suffix = hf_name.split(".")[-1]
    if suffix in _ML8 and L is not None:
        g, role = _ML8[suffix]; return (f"blk.{L}.{g}.weight", role, Tier.ML8)
    if suffix in _FP8 and L is not None:
        g, role = _FP8[suffix]; return (f"blk.{L}.{g}.weight", role, Tier.FP8)
    return (hf_name, "native", Tier.NATIVE)
