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


class SyntheticSpineBuilder:
    """Test double for wp-forge's SpineBuilder.

    Reads a peeled spine (dense + spec_head.dense safetensors + config.json)
    and writes a spine GGUF with the deepseek41 KV, one f32 tensor per peeled
    tensor. Tensor names: a leading "layers." is replaced by "blk.", the rest
    is kept unchanged; other names are unchanged. The peeled tensor names are
    recorded on ``self.peeled`` (in shard order) so tests can assert on them.
    """

    def __init__(self) -> None:
        self.peeled: list[str] = []

    def build(self, peel_dir, out_gguf, quant) -> Path:
        import gguf
        from safetensors import safe_open

        peel_dir = Path(peel_dir)
        out_gguf = Path(out_gguf)
        cfg = json.loads((peel_dir / "config.json").read_text())
        w = gguf.GGUFWriter(str(out_gguf), arch="deepseek41")
        w.add_name("synthetic")
        w.add_block_count(int(cfg["num_hidden_layers"]) + int(cfg.get("num_nextn_predict_layers", 0)))
        w.add_embedding_length(int(cfg["hidden_size"]))
        w.add_expert_feed_forward_length(int(cfg["moe_intermediate_size"]))
        w.add_expert_count(int(cfg["n_routed_experts"]))
        w.add_expert_used_count(int(cfg["num_experts_per_tok"]))
        for st in sorted(peel_dir.glob("*.safetensors")):
            with safe_open(str(st), framework="pt", device="cpu") as f:
                for name in f.keys():
                    self.peeled.append(name)
                    w.add_tensor(
                        self._gguf_name(name),
                        f.get_tensor(name).float().numpy(),
                    )
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return out_gguf

    @staticmethod
    def _gguf_name(name: str) -> str:
        if name.startswith("layers."):
            return "blk." + name[len("layers."):]
        return name
