#!/usr/bin/env python3
"""
MAD-214 Phase 0: Extract KV cache samples from a HuggingFace model.

Runs prefill on a calibration corpus (wikitext-2 + NIAH-style prompts) and dumps
the resulting per-layer past_key_values to .npz files. The downstream calibration
fit script (fit_centroids.py) consumes these to learn FP8-constrained Lloyd-Max
centroids.

Design notes (MAD-214):
- We need the K/V *distribution* not the actual cache layout of any specific
  inference engine. HF transformers + the un-quantized model weights gives us
  the cleanest possible source. GGUF/llama.cpp would produce a distribution
  downstream of weight quantization.
- past_key_values shape per layer: (batch, n_kv_heads, seq_len, head_dim) for
  each of K and V. We dump K and V separately, per layer, per prompt.
- Output format: numpy .npz files at <out_dir>/<prompt_id>/layer_<L>_{k,v}.npz
  Each file contains a single array of shape (n_kv_heads, seq_len, head_dim).
  fp32 storage so the calibration script doesn't need to know the source dtype.

Usage:
  python3 extract_kv.py \\
      --model Qwen/Qwen3.5-4B \\
      --corpus wikitext+niah \\
      --output-dir /tmp/mad214_kv \\
      --num-wikitext 32 \\
      --num-niah 32 \\
      --wikitext-chunk-tokens 512 \\
      --niah-ctx-tokens 8192
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def load_wikitext_chunks(path: Path, num_chunks: int, tokens_per_chunk: int, tokenizer) -> list[str]:
    """Slice wikitext-2 raw into roughly `tokens_per_chunk`-token chunks."""
    text = path.read_text(encoding="utf-8")
    # Token-counted slicing: walk through, accumulating until target reached
    all_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    chunks: list[str] = []
    pos = 0
    while pos + tokens_per_chunk <= len(all_ids) and len(chunks) < num_chunks:
        chunk_ids = all_ids[pos : pos + tokens_per_chunk]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        pos += tokens_per_chunk
    return chunks


def build_niah_prompts(num_prompts: int, ctx_tokens: int, tokenizer) -> list[str]:
    """
    Build N synthetic NIAH-style prompts at ~ctx_tokens length.
    Pattern: filler context + needle ("the magic word for {id} is {token}") + query.
    Provides longer-context KV distributions for calibration coverage.
    """
    filler_seed = (
        "In the realm of distributed systems, consensus algorithms ensure that "
        "multiple nodes agree on a single value despite network partitions and "
        "faulty actors. The Paxos protocol, introduced by Leslie Lamport in 1989, "
        "remains a foundational reference. Raft, designed for understandability, "
        "achieves equivalent guarantees with clearer leader-election semantics. "
    )
    # Multiply filler to roughly target length
    filler_tokens_per_block = len(tokenizer(filler_seed, add_special_tokens=False).input_ids)
    blocks_needed = max(1, ctx_tokens // filler_tokens_per_block)
    base_filler = filler_seed * blocks_needed

    prompts: list[str] = []
    for i in range(num_prompts):
        needle_id = 100 + i
        needle_token = f"quokka-{i:04d}-marmot"
        needle = f"\n\nIMPORTANT: The magic word for slot {needle_id} is {needle_token}.\n\n"
        # Insert needle around middle of filler
        mid = len(base_filler) // 2
        full = base_filler[:mid] + needle + base_filler[mid:]
        full += f"\n\nWhat is the magic word for slot {needle_id}? Answer:"
        prompts.append(full)
    return prompts


def dump_past_key_values(past_kv, out_dir: Path, prompt_id: str) -> dict:
    """
    past_kv is a tuple of (K, V) per layer. Each tensor: (batch, n_kv_heads, seq_len, head_dim).
    Dumps fp32 .npz per layer, returns metadata dict.
    """
    prompt_dir = out_dir / prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)

    meta = {"prompt_id": prompt_id, "n_layers": len(past_kv), "layers": []}
    for layer_idx, (k, v) in enumerate(past_kv):
        # Assume batch=1; squeeze it out
        k_np = k.detach().to(torch.float32).cpu().numpy()[0]  # (n_kv_heads, seq_len, head_dim)
        v_np = v.detach().to(torch.float32).cpu().numpy()[0]
        np.savez_compressed(prompt_dir / f"layer_{layer_idx:03d}_k.npz", k=k_np)
        np.savez_compressed(prompt_dir / f"layer_{layer_idx:03d}_v.npz", v=v_np)
        meta["layers"].append(
            {
                "layer": layer_idx,
                "n_kv_heads": int(k_np.shape[0]),
                "seq_len": int(k_np.shape[1]),
                "head_dim": int(k_np.shape[2]),
            }
        )
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local path (e.g. Qwen/Qwen3.5-4B)")
    p.add_argument("--wikitext-path", default="wikitext-2-raw/wiki.train.raw")
    p.add_argument("--output-dir", required=True, help="Directory to dump KV .npz files")
    p.add_argument("--num-wikitext", type=int, default=32)
    p.add_argument("--wikitext-chunk-tokens", type=int, default=512)
    p.add_argument("--num-niah", type=int, default=32)
    p.add_argument("--niah-ctx-tokens", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-prompts", type=int, default=None, help="Cap total prompts (debug)")
    args = p.parse_args()

    # Lazy import so --help doesn't pay the cost
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_kv] Loading tokenizer + model: {args.model} ({args.dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    cfg = model.config
    print(
        f"[extract_kv] Model config: n_layers={getattr(cfg, 'num_hidden_layers', '?')} "
        f"n_kv_heads={getattr(cfg, 'num_key_value_heads', '?')} "
        f"head_dim={getattr(cfg, 'hidden_size', 0) // max(1, getattr(cfg, 'num_attention_heads', 1))}",
        flush=True,
    )

    # Build prompts
    prompts: list[tuple[str, str]] = []  # (prompt_id, text)
    wiki_path = Path(args.wikitext_path)
    if args.num_wikitext > 0 and wiki_path.exists():
        chunks = load_wikitext_chunks(wiki_path, args.num_wikitext, args.wikitext_chunk_tokens, tokenizer)
        for i, c in enumerate(chunks):
            prompts.append((f"wiki_{i:03d}", c))
        print(f"[extract_kv] Loaded {len(chunks)} wikitext chunks", flush=True)
    elif args.num_wikitext > 0:
        print(f"[extract_kv] WARN: wikitext path missing: {wiki_path}", flush=True)

    if args.num_niah > 0:
        niah = build_niah_prompts(args.num_niah, args.niah_ctx_tokens, tokenizer)
        for i, c in enumerate(niah):
            prompts.append((f"niah_{i:03d}", c))
        print(f"[extract_kv] Built {len(niah)} NIAH prompts (~{args.niah_ctx_tokens} tokens each)", flush=True)

    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    print(f"[extract_kv] Total prompts: {len(prompts)}", flush=True)

    all_meta = []
    with torch.no_grad():
        for idx, (pid, text) in enumerate(prompts):
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            # Truncate to model max_position_embeddings if needed
            max_pos = getattr(cfg, "max_position_embeddings", 32768)
            if ids.shape[1] > max_pos:
                ids = ids[:, :max_pos]
            print(
                f"[extract_kv] [{idx + 1}/{len(prompts)}] {pid}  tokens={ids.shape[1]}",
                flush=True,
            )
            out = model(ids, use_cache=True)
            meta = dump_past_key_values(out.past_key_values, out_dir, pid)
            meta["tokens"] = int(ids.shape[1])
            all_meta.append(meta)

            # Free per-prompt; KV is the only big tensor we keep on GPU
            del out
            torch.cuda.empty_cache()
            gc.collect()

    # Manifest for the calibration script
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "model": args.model,
        "dtype": args.dtype,
        "n_prompts": len(all_meta),
        "prompts": all_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[extract_kv] Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
