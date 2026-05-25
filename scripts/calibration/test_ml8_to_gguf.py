#!/usr/bin/env python3
"""Tests for ml8_to_gguf — patching ml8-4 reconstructed tensors into a base f16 GGUF."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ml8_to_gguf import hf_to_gguf_name  # noqa: E402


def test_hf_to_gguf_name_qwen_mlp():
    """Qwen MLP HF names map to GGUF blk.N.ffn_*.weight format."""
    cases = {
        "model.layers.0.mlp.gate_proj":  "blk.0.ffn_gate.weight",
        "model.layers.5.mlp.up_proj":    "blk.5.ffn_up.weight",
        "model.layers.31.mlp.down_proj": "blk.31.ffn_down.weight",
    }
    for hf, gguf in cases.items():
        got = hf_to_gguf_name(hf)
        assert got == gguf, f"hf_to_gguf_name({hf!r}) = {got!r}, expected {gguf!r}"
    print(f"  PASS test_hf_to_gguf_name_qwen_mlp")


def test_hf_to_gguf_name_rejects_unknown():
    """Names we don't have a mapping for should raise, not silently mangle."""
    for unknown in [
        "model.layers.0.self_attn.q_proj",  # attention proj — not yet supported
        "model.embed_tokens",                # embedding — different naming
        "lm_head",                            # output — different naming
    ]:
        try:
            got = hf_to_gguf_name(unknown)
        except (ValueError, KeyError) as e:
            continue
        assert False, f"hf_to_gguf_name({unknown!r}) silently returned {got!r}, should raise"
    print(f"  PASS test_hf_to_gguf_name_rejects_unknown")


if __name__ == "__main__":
    test_hf_to_gguf_name_qwen_mlp()
    test_hf_to_gguf_name_rejects_unknown()
    print("\nALL TESTS PASSED")
