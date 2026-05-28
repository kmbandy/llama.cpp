#!/usr/bin/env python3
"""rotate_model_quarot.py — QuaRot-R1 rotation pass over a bf16 GGUF.

See docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-design.md.
"""
from __future__ import annotations

import enum
import re
from typing import Optional


class Role(enum.Enum):
    PASSTHROUGH      = "passthrough"
    EMBED            = "embed"
    NORM_PRE_ATTN    = "norm_pre_attn"
    NORM_PRE_FFN     = "norm_pre_ffn"
    NORM_PRE_SSM     = "norm_pre_ssm"
    NORM_OUT         = "norm_out"
    ATTN_Q           = "attn_q"
    ATTN_K           = "attn_k"
    ATTN_V           = "attn_v"
    ATTN_O           = "attn_o"
    FFN_GATE_INP     = "ffn_gate_inp"
    FFN_GATE_EXPS    = "ffn_gate_exps"
    FFN_UP_EXPS      = "ffn_up_exps"
    FFN_DOWN_EXPS    = "ffn_down_exps"
    MAMBA_IN         = "mamba_in"
    MAMBA_OUT        = "mamba_out"
    LM_HEAD          = "lm_head"


# Per-arch regex pattern table. Patterns are ordered most-specific-first
# because re.fullmatch is tried in declaration order.
_ROLE_PATTERNS: dict[str, list[tuple[str, Role]]] = {
    "qwen36moe": [
        (r"token_embd\.weight",                   Role.EMBED),
        (r"output_norm\.weight",                  Role.NORM_OUT),
        (r"output\.weight",                       Role.LM_HEAD),
        (r"blk\.\d+\.attn_norm\.weight",          Role.NORM_PRE_ATTN),
        (r"blk\.\d+\.ffn_norm\.weight",           Role.NORM_PRE_FFN),
        (r"blk\.\d+\.ssm_norm\.weight",           Role.NORM_PRE_SSM),
        (r"blk\.\d+\.attn_q\.weight",             Role.ATTN_Q),
        (r"blk\.\d+\.attn_k\.weight",             Role.ATTN_K),
        (r"blk\.\d+\.attn_v\.weight",             Role.ATTN_V),
        (r"blk\.\d+\.attn_output\.weight",        Role.ATTN_O),
        (r"blk\.\d+\.ffn_gate_inp\.weight",       Role.FFN_GATE_INP),
        (r"blk\.\d+\.ffn_gate_exps\.weight",      Role.FFN_GATE_EXPS),
        (r"blk\.\d+\.ffn_up_exps\.weight",        Role.FFN_UP_EXPS),
        (r"blk\.\d+\.ffn_down_exps\.weight",      Role.FFN_DOWN_EXPS),
        (r"blk\.\d+\.ssm_in\.weight",             Role.MAMBA_IN),
        (r"blk\.\d+\.ssm_out\.weight",            Role.MAMBA_OUT),
        # Known passthroughs (rope_freqs, biases, ssm internals, etc.)
        (r"rope_freqs\.weight",                   Role.PASSTHROUGH),
    ],
}


def classify_tensor(name: str, arch: str) -> Role:
    """Map a GGUF tensor name to its rotation role.

    Raises ValueError if arch is unknown OR if no pattern matches (the
    PASSTHROUGH catch-all only fires on tensors we've decided not to touch;
    a tensor that fails to match ANY pattern indicates the arch table is
    incomplete and we should hard-fail rather than silently passthrough).
    """
    patterns = _ROLE_PATTERNS.get(arch)
    if patterns is None:
        raise ValueError(f"unknown arch {arch!r}; add a pattern table to _ROLE_PATTERNS")
    for pattern, role in patterns:
        if re.fullmatch(pattern, name):
            return role
    raise ValueError(
        f"tensor {name!r} matched no pattern under arch={arch!r}; "
        f"the catch-all should have fired — check the pattern table."
    )
