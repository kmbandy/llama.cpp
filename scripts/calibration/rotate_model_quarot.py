#!/usr/bin/env python3
"""rotate_model_quarot.py — QuaRot-R1 rotation pass over a bf16 GGUF.

See docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-design.md.
"""
from __future__ import annotations

import argparse
import enum
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kronecker_rotation import sylvester, factor_for_dim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


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
    "qwen35": [
        (r"token_embd\.weight",                Role.EMBED),
        (r"output_norm\.weight",               Role.NORM_OUT),
        (r"output\.weight",                    Role.LM_HEAD),
        (r"blk\.\d+\.attn_norm\.weight",       Role.NORM_PRE_ATTN),
        (r"blk\.\d+\.ffn_norm\.weight",        Role.NORM_PRE_FFN),
        (r"blk\.\d+\.attn_q\.weight",          Role.ATTN_Q),
        (r"blk\.\d+\.attn_k\.weight",          Role.ATTN_K),
        (r"blk\.\d+\.attn_v\.weight",          Role.ATTN_V),
        (r"blk\.\d+\.attn_output\.weight",     Role.ATTN_O),
        # Dense FFN: gate/up are input-residual linears, down is output-residual.
        # Reuse FFN_GATE_EXPS/UP_EXPS/DOWN_EXPS roles; rank-based dispatch in
        # rotate_gguf selects input_2d/output_2d for 2D tensors vs MoE for 3D.
        (r"blk\.\d+\.ffn_gate\.weight",        Role.FFN_GATE_EXPS),
        (r"blk\.\d+\.ffn_up\.weight",          Role.FFN_UP_EXPS),
        (r"blk\.\d+\.ffn_down\.weight",        Role.FFN_DOWN_EXPS),
        (r"rope_freqs\.weight",                Role.PASSTHROUGH),
        # Intentionally no `.*` catch-all — unknown names raise (matches qwen36moe).
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


_NORM_ROLES = {Role.NORM_PRE_ATTN, Role.NORM_PRE_FFN, Role.NORM_PRE_SSM, Role.NORM_OUT}


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


def _rotation_plan(role: Role, name: str) -> tuple[str, Optional[str]]:
    """Decide (rotation_kind, gamma_tensor_name) for a given tensor."""
    if role == Role.PASSTHROUGH:
        return ("passthrough", None)
    if role in _NORM_ROLES:
        return ("norm_to_ones", None)
    if role == Role.EMBED:
        return ("embed", None)
    if role == Role.LM_HEAD:
        return ("input_2d", "output_norm.weight")
    # Per-block roles: pull the layer index from the name.
    m = re.match(r"blk\.(\d+)\.", name)
    if not m:
        raise ValueError(f"expected blk.N. prefix on {name!r} for role {role}")
    L = int(m.group(1))
    if role in (Role.ATTN_Q, Role.ATTN_K, Role.ATTN_V):
        return ("input_2d", f"blk.{L}.attn_norm.weight")
    if role == Role.ATTN_O:
        return ("output_2d", None)
    if role in (Role.FFN_GATE_INP,):
        return ("input_2d", f"blk.{L}.ffn_norm.weight")
    if role in (Role.FFN_GATE_EXPS, Role.FFN_UP_EXPS):
        return ("input_moe", f"blk.{L}.ffn_norm.weight")
    if role == Role.FFN_DOWN_EXPS:
        return ("output_moe", None)
    if role == Role.MAMBA_IN:
        return ("input_2d", f"blk.{L}.ssm_norm.weight")
    if role == Role.MAMBA_OUT:
        return ("output_2d", None)
    raise ValueError(f"no rotation plan for role {role}")


_GGUF_DTYPE_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16":  torch.float16,
    "F32":  torch.float32,
}


def _bytes_from_fp32(t: torch.Tensor, gguf_dtype) -> np.ndarray:
    """Cast a fp32 tensor to raw bytes matching the given GGUF dtype.

    Real GGUFs from gguf_f16_to_bf16.py keep RMSNorm γ as F32 even when the
    large weights are BF16 — round-tripping in source dtype preserves the
    file's dtype mix.
    """
    target = _GGUF_DTYPE_TO_TORCH.get(gguf_dtype.name)
    if target is None:
        raise ValueError(f"unsupported write dtype {gguf_dtype.name!r}")
    return np.ascontiguousarray(t.contiguous().to(target).view(torch.uint8).numpy())


def _gguf_tensor_to_torch(t) -> torch.Tensor:
    """Materialize a GGUFReader BF16/F16/F32 tensor as a fp32 torch tensor in
    PyTorch (row-major / C) axis order.

    GGUFReader stores shape in GGUF file order, which is the reverse of C order.
    Reversing t.shape recovers the [out, in, ...] PyTorch convention. For 1D γ
    vectors the reverse is a no-op, so this function handles both linears and
    norm weights.
    """
    target = _GGUF_DTYPE_TO_TORCH.get(t.tensor_type.name)
    if target is None:
        raise ValueError(f"{t.name}: unsupported dtype {t.tensor_type.name}")
    arr = np.asarray(t.data, dtype=np.uint8).copy()  # detach from mmap
    torch_shape = [int(s) for s in reversed(list(t.shape))]
    return torch.from_numpy(arr).view(target).reshape(*torch_shape).to(torch.float32)


# Internal GGUF metadata fields and auto-added fields to skip during KV copy.
_GGUF_INTERNAL_FIELDS = frozenset({"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                                    "general.architecture"})


def rotate_gguf(source_path: str, output_path: str, arch: str, seed: int,
                device: torch.device) -> dict:
    """Pass 2 — stream source GGUF, rotate every tensor according to its role,
    write to a new GGUF.

    Returns a manifest dict suitable for the sidecar JSON.
    """
    # Pass 1
    roster, gammas, d_model = index_pass(source_path, arch=arch)
    R_resid = build_R_resid(d_model=d_model, seed=seed, device=device)

    # Re-open source for streaming. Open writer.
    r = gguf.GGUFReader(source_path)
    w = gguf.GGUFWriter(output_path, arch=arch)

    # Copy KV fields verbatim, skipping internal/auto fields.
    for f in r.fields.values():
        if f.name in _GGUF_INTERNAL_FIELDS:
            continue
        vtype = f.types[0]
        val = f.parts[f.data[0]]
        if vtype == gguf.GGUFValueType.UINT32:
            w.add_key_value(f.name, int(val[0]), gguf.GGUFValueType.UINT32)
        elif vtype == gguf.GGUFValueType.UINT64:
            w.add_key_value(f.name, int(val[0]), gguf.GGUFValueType.UINT64)
        elif vtype == gguf.GGUFValueType.FLOAT32:
            w.add_key_value(f.name, float(val[0]), gguf.GGUFValueType.FLOAT32)
        elif vtype == gguf.GGUFValueType.FLOAT64:
            w.add_key_value(f.name, float(val[0]), gguf.GGUFValueType.FLOAT64)
        elif vtype == gguf.GGUFValueType.INT32:
            w.add_key_value(f.name, int(val[0]), gguf.GGUFValueType.INT32)
        elif vtype == gguf.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(val).decode("utf-8"), gguf.GGUFValueType.STRING)
        else:
            # Skip exotic types (ARRAY, BOOL, etc.) not needed for arch metadata.
            pass

    rotated: list[str] = []
    absorbed: list[str] = []

    for t in r.tensors:
        role = roster[t.name]
        kind, gamma_name = _rotation_plan(role, t.name)
        # Rank-based dispatch: FFN_GATE_EXPS/UP_EXPS/DOWN_EXPS roles map to MoE
        # primitives by default, but dense 2D tensors (e.g. Qwen3.5-4B ffn_gate/
        # ffn_up/ffn_down) use the 2D primitives instead.
        if kind == "input_moe" and len(t.shape) == 2:
            kind = "input_2d"
        if kind == "output_moe" and len(t.shape) == 2:
            kind = "output_2d"

        if kind == "passthrough":
            # Copy raw bytes unchanged; let gguf-py derive shape from the uint8 array.
            arr = np.asarray(t.data, dtype=np.uint8).copy()
            w.add_tensor(t.name, arr, raw_dtype=t.tensor_type)
            continue

        if kind == "norm_to_ones":
            # Write ones in bf16 with the same element shape.
            # t.shape is GGUF-order; reversed gives PyTorch shape.
            torch_shape = [int(s) for s in reversed(list(t.shape))]
            ones = torch.ones(*torch_shape, dtype=torch.float32)
            w.add_tensor(t.name, _bytes_from_fp32(ones, t.tensor_type), raw_dtype=t.tensor_type)
            absorbed.append(t.name)
            continue

        # Otherwise materialise the tensor in PyTorch-convention fp32 order.
        W_fp32 = _gguf_tensor_to_torch(t).to(device)

        if kind == "embed":
            # token_embd shape (PyTorch): [vocab, d_model].
            # Output-side rotation: embed @ R_resid makes emb vectors live in the
            # rotated space that downstream input-side linears expect.
            W_new = W_fp32 @ R_resid
        elif kind == "input_2d":
            gamma = gammas[gamma_name].to(device) if gamma_name else torch.ones(d_model, device=device)
            W_new = rotate_input_side(W_fp32, gamma, R_resid)
        elif kind == "output_2d":
            W_new = rotate_output_side(W_fp32, R_resid)
        elif kind == "input_moe":
            gamma = gammas[gamma_name].to(device) if gamma_name else torch.ones(d_model, device=device)
            W_new = rotate_moe_input_side(W_fp32, gamma, R_resid)
        elif kind == "output_moe":
            W_new = rotate_moe_output_side(W_fp32, R_resid)
        else:
            raise ValueError(f"unknown rotation kind {kind!r}")

        # Write back without raw_shape: gguf-py infers shape from the uint8 array,
        # which encodes the PyTorch-convention shape. The writer reverses it to
        # GGUF file order, so read-back shapes match the source.
        w.add_tensor(t.name, _bytes_from_fp32(W_new.cpu(), t.tensor_type), raw_dtype=t.tensor_type)
        rotated.append(t.name)

        # Free memory before moving to the next tensor.
        del W_fp32, W_new
        if device.type == "cuda":
            torch.cuda.empty_cache()

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    return {
        "seed": int(seed),
        "d_model": int(d_model),
        "arch": arch,
        "rotated_tensors": rotated,
        "absorbed_norms": absorbed,
    }


def _save_sidecar(output_path: str, manifest: dict) -> None:
    sidecar_path = output_path + ".quarot_r1.json"
    with open(sidecar_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Apply QuaRot-R1 rotation to a bf16 GGUF.")
    p.add_argument("--source", required=True, help="path to source bf16 GGUF")
    p.add_argument("--output", required=True, help="path to write rotated bf16 GGUF")
    p.add_argument("--arch", default="qwen36moe", choices=sorted(_ROLE_PATTERNS.keys()),
                   help="GGUF architecture for tensor-name classification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="torch device string for the rotation matmul (e.g. cuda:0)")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    manifest = rotate_gguf(source_path=args.source, output_path=args.output,
                           arch=args.arch, seed=args.seed, device=device)
    _save_sidecar(args.output, manifest)
    print(f"wrote {args.output} ({len(manifest['rotated_tensors'])} rotated, "
          f"{len(manifest['absorbed_norms'])} absorbed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
