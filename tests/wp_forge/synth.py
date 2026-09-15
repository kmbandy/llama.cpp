"""Synthetic DeepSeek-V4.1-shaped HF repo for wp-forge tests.

Tiny, deterministic, bf16 everywhere. Expert tensors are plain bf16 with NO
`.scale` partners, so the lossless MXFP4 repack path does not apply here (the
stage falls through to dequant->quantize). One safetensors shard per layer,
one per MTP stage, one for embed/head/norm -- the DS4.1 layout minus the
+3 shard offset the real repo has.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def _t(rng: np.random.Generator, *shape: int) -> torch.Tensor:
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)).to(torch.bfloat16)


def make_synthetic_hf_repo(
    dir: Path,
    *,
    n_layer: int = 2,
    n_expert: int = 4,
    n_ff: int = 64,
    n_embd: int = 32,
    nextn: int = 1,
    engram: bool = True,
) -> Path:
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_shards = n_layer + nextn + 1
    weight_map: dict[str, str] = {}
    total = 0

    def shard_name(i: int) -> str:
        return f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"

    def write(i: int, tensors: dict[str, torch.Tensor]) -> None:
        nonlocal total
        name = shard_name(i)
        save_file(tensors, str(dir / name))
        for k, v in tensors.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()

    def experts(prefix: str) -> dict[str, torch.Tensor]:
        out = {}
        for e in range(n_expert):
            out[f"{prefix}.{e}.w1.weight"] = _t(rng, n_ff, n_embd)
            out[f"{prefix}.{e}.w3.weight"] = _t(rng, n_ff, n_embd)
            out[f"{prefix}.{e}.w2.weight"] = _t(rng, n_embd, n_ff)
        return out

    for layer in range(n_layer):
        t = experts(f"layers.{layer}.ffn.experts")
        t[f"layers.{layer}.attn.wq.weight"] = _t(rng, n_embd, n_embd)
        t[f"layers.{layer}.ffn.shared_experts.w1.weight"] = _t(rng, n_ff, n_embd)
        t[f"layers.{layer}.attn_norm.weight"] = _t(rng, n_embd)
        if engram:
            t[f"layers.{layer}.engram.embed.weight"] = _t(rng, 16, n_embd)
        write(layer, t)
    for stage in range(nextn):
        t = experts(f"mtp.{stage}.ffn.experts")
        t[f"mtp.{stage}.attn.wq.weight"] = _t(rng, n_embd, n_embd)
        write(n_layer + stage, t)
    write(n_layer + nextn, {
        "embed.weight": _t(rng, 64, n_embd),
        "head.weight": _t(rng, 64, n_embd),
        "norm.weight": _t(rng, n_embd),
    })

    (dir / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV41ForCausalLM"],
        "num_hidden_layers": n_layer,
        "num_nextn_predict_layers": nextn,
        "n_routed_experts": n_expert,
        "dspark_n_routed_experts": n_expert,
        "moe_intermediate_size": n_ff,
        "hidden_size": n_embd,
        "num_experts_per_tok": 2,
    }, indent=1))
    (dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": weight_map,
    }, indent=1))
    return dir
