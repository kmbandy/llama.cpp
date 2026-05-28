#!/usr/bin/env python3
"""rotate_model_quarot.py — QuaRot-R1 rotation pass over a bf16 GGUF.

See docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-design.md.
"""
from __future__ import annotations

import enum
import re
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kronecker_rotation import sylvester, factor_for_dim


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


def build_R_resid(d_model: int, seed: int, device: torch.device) -> torch.Tensor:
    """Construct a random Hadamard rotation of size d_model.

    R_resid = D ⊙ H, where:
      - H is Sylvester(d_model) for pure-power-of-2 d_model,
        or H_a_random ⊗ H_b_sylvester via factor_for_dim() otherwise.
      - D = diag(±1) sampled from Bernoulli(0.5) seeded by `seed`.

    Returned tensor is fp32 on the requested device. Deterministic in `seed`.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # Sign-flip diagonal — independent of H form, applied as elementwise row scale.
    signs = (torch.randint(0, 2, (d_model,), generator=gen, dtype=torch.float32) * 2.0 - 1.0)

    # Hadamard core
    if (d_model & (d_model - 1)) == 0:
        H = sylvester(d_model).to(dtype=torch.float32)
    else:
        a, b = factor_for_dim(d_model, max_b=1024)
        # Random-orthogonal H_a; deterministic Sylvester H_b.
        # Use a separate sub-seed so changing d_model factoring doesn't perturb
        # the sign-flip stream.
        from kronecker_rotation import random_orthogonal
        H_a = random_orthogonal(a, seed=int(seed) + 1_000_003)
        H_b = sylvester(b).to(dtype=torch.float32)
        H = torch.kron(H_a.contiguous(), H_b.contiguous())

    R = (signs.unsqueeze(1) * H).to(device=device, dtype=torch.float32)
    return R


def rotate_input_side(W: torch.Tensor, gamma: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """Apply γ absorption + R_resid input rotation to a residual-reading linear.

    W shape: [N, d_model] (PyTorch [out, in]).
    gamma shape: [d_model] — RMSNorm γ that precedes this linear in the original graph.
    R_resid shape: [d_model, d_model] — orthogonal.

    Math: original forward is y = (γ ⊙ x) @ W.T. In the rotated stream the input
    is x' = x @ R.T (γ already absorbed at construction time). The rotated weight
    must satisfy y = x' @ W_new.T, giving W_new = (W ⊙ γ_row) @ R.T.
    """
    Wg = W * gamma.unsqueeze(0)            # [N, d_model], column-wise γ
    return Wg @ R_resid.T                  # [N, d_model]


def rotate_output_side(W: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """Apply R_resid output rotation to a residual-writing linear.

    W shape: [d_model, K] (PyTorch [out, in]). K is whatever feeds this linear.
    R_resid shape: [d_model, d_model] — orthogonal.

    Math: original forward y = x @ W.T contributes to the residual. To produce
    y_new = y @ R.T directly, we need W_new = R @ W.
    """
    return R_resid @ W                     # [d_model, K]


def rotate_moe_input_side(W: torch.Tensor, gamma: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """MoE input-side rotation, batched along the expert axis.

    W shape: [d_ffn, d_model, n_experts]. Applies rotate_input_side to each
    expert in one batched matmul instead of a Python loop.
    """
    d_ffn, d_model, n_exp = W.shape
    # Reshape so the d_model axis stays adjacent for the matmul.
    # [d_ffn, d_model, n_exp] → permute → [d_ffn, n_exp, d_model] → flatten outer → [d_ffn*n_exp, d_model]
    W_p     = W.permute(0, 2, 1).contiguous().view(d_ffn * n_exp, d_model)
    W_rot   = (W_p * gamma.unsqueeze(0)) @ R_resid.T
    # Reshape back to [d_ffn, n_exp, d_model] → permute → [d_ffn, d_model, n_exp]
    return W_rot.view(d_ffn, n_exp, d_model).permute(0, 2, 1).contiguous()


def rotate_moe_output_side(W: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """MoE output-side rotation, batched along the expert axis.

    W shape: [d_model, d_ffn, n_experts]. Applies rotate_output_side
    (R @ W on the d_model axis) to each expert in one batched matmul.
    """
    d_model, d_ffn, n_exp = W.shape
    # Flatten the trailing dims so the d_model axis is the matmul axis.
    W_flat  = W.reshape(d_model, d_ffn * n_exp)
    W_rot   = R_resid @ W_flat
    return W_rot.view(d_model, d_ffn, n_exp).contiguous()


import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


_NORM_ROLES = {Role.NORM_PRE_ATTN, Role.NORM_PRE_FFN, Role.NORM_PRE_SSM, Role.NORM_OUT}


def _gguf_tensor_to_torch(t) -> torch.Tensor:
    """Materialize a GGUFReader tensor as a torch float32 tensor.

    Assumes bf16 storage — the source GGUF is the bf16 conversion produced by
    gguf_f16_to_bf16.py and used by calibrate_ml8_paged.py.
    """
    if t.tensor_type.name != "BF16":
        raise ValueError(f"{t.name}: expected BF16, got {t.tensor_type.name}")
    arr = np.asarray(t.data, dtype=np.uint8).copy()  # detach from mmap
    return torch.from_numpy(arr).view(torch.bfloat16).view(*t.shape).to(torch.float32)


def index_pass(source_path: str, arch: str) -> tuple[dict[str, Role], dict[str, torch.Tensor], int]:
    """Pass 1 — walk source GGUF, classify each tensor, pull only γ vectors into RAM.

    Returns (roster, gammas, d_model). Roster covers every tensor in the file.
    Gammas only contains the RMSNorm tensors (kept resident — ~200 KB total even
    for 35B-A3B). d_model is read from the arch's embedding_length KV.
    """
    r = gguf.GGUFReader(source_path)
    roster: dict[str, Role] = {}
    gammas: dict[str, torch.Tensor] = {}
    for t in r.tensors:
        role = classify_tensor(t.name, arch=arch)
        roster[t.name] = role
        if role in _NORM_ROLES:
            gammas[t.name] = _gguf_tensor_to_torch(t)

    # d_model from arch KV (e.g. "qwen36moe.embedding_length"). Fall back to
    # token_embd.weight's last dim if the KV isn't present.
    d_model: Optional[int] = None
    for f in r.fields.values():
        if f.name == f"{arch}.embedding_length":
            d_model = int(f.parts[f.data[0]][0])
            break
    if d_model is None:
        embed_t = next(t for t in r.tensors if t.name == "token_embd.weight")
        d_model = int(embed_t.shape[-1])

    return roster, gammas, d_model
