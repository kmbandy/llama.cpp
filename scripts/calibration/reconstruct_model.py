#!/usr/bin/env python3
"""reconstruct_model.py — load a HF model and overlay ml8-4 quantized weights.

Given a HF model name + a directory of per-layer .pt blobs (output of
calibrate_ml8.py), this loads the original f16/bf16 model and replaces
each quantized layer's weight with the dequantized version reconstructed
from its blob. Result: a model in memory whose target linears now match
exactly what a real ml8-4 deployment would produce at inference time.

Use cases:
  1. Re-evaluate PPL on a calibration artifact (vs the in-memory PPL
     eval that calibrate_ml8.py does immediately after quantization —
     this verifies the disk format actually round-trips).
  2. Run sanity prompts via .generate() on the quantized model.
  3. Inspect per-layer quality after the fact.

Usage:
    python3 scripts/calibration/reconstruct_model.py \\
        --model Qwen/Qwen3.5-4B \\
        --calibration-dir /tmp/ml8-qwen3-4b-full \\
        --eval-ppl --ppl-max-tokens 100000 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from ml8_io import (  # noqa: E402
    load_ml8_layer, reconstruct_weight, reconstruct_inference_weight, bits_per_value,
)


def get_module(model, name: str):
    """model.layers.0.mlp.up_proj → traverse attribute chain."""
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


@torch.no_grad()
def overlay_ml8_weights(model, calibration_dir: Path, device: str,
                       verbose: bool = True) -> dict:
    """Load all per-layer .pt blobs and overwrite the corresponding linears.

    Returns a summary dict: counts, total bits, original-fp bits, etc.
    """
    blobs = sorted(calibration_dir.glob("*.pt"))
    if not blobs:
        raise RuntimeError(f"no .pt files found in {calibration_dir}")

    n_loaded = 0
    n_skipped = 0
    total_quant_bits = 0
    total_orig_bits_fp16 = 0
    per_layer = []

    for path in blobs:
        blob = load_ml8_layer(path)
        name = blob["name"]
        target = None
        try:
            target = get_module(model, name)
        except (AttributeError, IndexError):
            if verbose:
                print(f"  [skip] {name}: not found in model")
            n_skipped += 1
            continue
        if not isinstance(target, nn.Linear):
            if verbose:
                print(f"  [skip] {name}: not a Linear ({type(target).__name__})")
            n_skipped += 1
            continue

        # Reconstruct on the target device for fewest transfers. The "inference"
        # variant absorbs any per-layer Kronecker rotation into the dequantized
        # weight so the model's standard forward is mathematically equivalent to
        # the rotated calibration's input-rotated form. No-op for legacy blobs.
        W_recon = reconstruct_inference_weight(blob).to(device)
        if tuple(W_recon.shape) != tuple(target.weight.shape):
            print(f"  [skip] {name}: shape mismatch "
                  f"{tuple(W_recon.shape)} vs {tuple(target.weight.shape)}")
            n_skipped += 1
            continue

        # Replace weight in-place. Preserve original dtype.
        target.weight.data.copy_(W_recon.to(target.weight.dtype))

        bpv = bits_per_value(blob)
        numel = target.weight.numel()
        total_quant_bits += numel * bpv
        total_orig_bits_fp16 += numel * 16
        per_layer.append({"name": name, "numel": numel, "bpv": bpv})
        n_loaded += 1
        if verbose:
            print(f"  [load] {name}  shape={tuple(W_recon.shape)}  bpv={bpv:.3f}")

    summary = {
        "n_loaded": n_loaded,
        "n_skipped": n_skipped,
        "total_quant_bits": total_quant_bits,
        "total_orig_bits_fp16": total_orig_bits_fp16,
        "size_ratio_vs_fp16": total_quant_bits / max(total_orig_bits_fp16, 1),
        "per_layer": per_layer,
    }
    return summary


@torch.no_grad()
def eval_ppl_wikitext(model, tokenizer, dev: str, seq_len: int = 2048,
                     stride: int = 1024, max_tokens: int | None = None) -> dict:
    """Same eval as calibrate_ml8.py — duplicated to keep this module standalone."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids.to(dev)
    n_tokens = ids.shape[1]
    if max_tokens is not None:
        n_tokens = min(n_tokens, max_tokens)
        ids = ids[:, :n_tokens]
    print(f"  [ppl] {n_tokens} tokens, window={seq_len} stride={stride}")

    nll_sum = 0.0
    n_pred = 0
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + seq_len, n_tokens)
        target_len = end - prev_end
        input_ids = ids[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100
        outputs = model(input_ids, labels=target_ids)
        n_unmasked = (target_ids != -100).sum().item()
        nll_sum += outputs.loss.item() * n_unmasked
        n_pred += n_unmasked
        prev_end = end
        if end >= n_tokens:
            break

    return {"ppl": math.exp(nll_sum / max(n_pred, 1)),
            "avg_nll": nll_sum / max(n_pred, 1),
            "n_tokens_scored": n_pred}


def overlay_ml8_kernels(model, calibration_dir: Path, device: str,
                        verbose: bool = True) -> dict:
    """Phase D.1+D.2: kernel-path overlay — swap each Linear with an Ml8Linear
    that invokes the ml8 Triton kernel at forward time.

    Parallel to `overlay_ml8_weights` (the dequant-and-replace path).
    Use --use-ml8-kernel flag to select this path.

    Returns a summary dict in the same shape as overlay_ml8_weights.

    Handles all blob variants (MAD-245 / Phase D.2):
      - Plain blob → kernel-only Ml8Linear
      - Rotated blob → Ml8Linear applies Q to activations at forward time
      - AWQ'd blob → Ml8Linear applies per-channel scale to activations
      - Rotated + AWQ'd → both, in calibration-mirrored order
    """
    # Import here so the dequant-only path doesn't pay the cost
    import sys
    _THIS = Path(__file__).resolve().parent
    if str(_THIS) not in sys.path:
        sys.path.insert(0, str(_THIS))
    from ml8_runtime import ml8_linear_from_blob

    blobs = sorted(calibration_dir.glob("*.pt"))
    if not blobs:
        raise RuntimeError(f"no .pt files found in {calibration_dir}")

    n_loaded = 0
    n_skipped = 0
    total_quant_bits = 0
    total_orig_bits_fp16 = 0
    per_layer = []

    for path in blobs:
        blob = load_ml8_layer(path)
        name = blob["name"]
        target = None
        try:
            target = get_module(model, name)
        except (AttributeError, IndexError):
            if verbose:
                print(f"  [skip] {name}: not found in model")
            n_skipped += 1
            continue
        if not isinstance(target, nn.Linear):
            if verbose:
                print(f"  [skip] {name}: not a Linear ({type(target).__name__})")
            n_skipped += 1
            continue

        # Build a ready-to-use Ml8Linear (with rotation + AWQ wired in if
        # the blob carries them) via the unified factory (MAD-245).
        ml8_linear = ml8_linear_from_blob(
            blob,
            bias=target.bias.detach().clone() if target.bias is not None else None,
            out_dtype=target.weight.dtype,
            device=device,
        )
        # In-place swap on the parent
        parent_path, _, leaf = name.rpartition(".")
        if parent_path:
            parent = get_module(model, parent_path)
        else:
            parent = model
        setattr(parent, leaf, ml8_linear.to(device))

        bpv = bits_per_value(blob)
        numel = target.weight.numel()
        total_quant_bits += numel * bpv
        total_orig_bits_fp16 += numel * 16
        per_layer.append({"name": name, "numel": numel, "bpv": bpv})
        n_loaded += 1
        if verbose:
            extras = []
            if ml8_linear.rotation is not None:
                extras.append("rot")
            if ml8_linear.awq_scale is not None:
                extras.append("awq")
            tag = ("+".join(extras)) if extras else "plain"
            print(
                f"  [kern] {name}  N={ml8_linear.out_features} "
                f"K={ml8_linear.in_features} bpv={bpv:.3f} ({tag})"
            )

    summary = {
        "n_loaded": n_loaded,
        "n_skipped": n_skipped,
        "total_quant_bits": total_quant_bits,
        "total_orig_bits_fp16": total_orig_bits_fp16,
        "size_ratio_vs_fp16": total_quant_bits / max(total_orig_bits_fp16, 1),
        "per_layer": per_layer,
        "overlay_mode": "kernel",
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calibration-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--eval-ppl", action="store_true")
    p.add_argument("--ppl-max-tokens", type=int, default=None)
    p.add_argument("--also-eval-baseline", action="store_true",
                   help="Eval PPL on the original f16 model FIRST, before overlay, "
                        "for delta computation.")
    p.add_argument("--use-ml8-kernel", action="store_true",
                   help="Phase D.1: swap target Linears with Ml8Linear that calls "
                        "the ml8 Triton kernel at forward time, instead of dequant+copy. "
                        "Currently rejects blobs with rotation/awq (Phase D.2 work).")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] {args.model}  dtype={dtype}  device={args.device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(args.device).eval()

    baseline_ppl = None
    if args.eval_ppl and args.also_eval_baseline:
        print("\n[ppl-baseline] f16 baseline PPL...")
        baseline_ppl = eval_ppl_wikitext(model, tok, args.device,
                                          max_tokens=args.ppl_max_tokens)
        print(f"  baseline PPL = {baseline_ppl['ppl']:.4f}")

    if args.use_ml8_kernel:
        print(f"\n[overlay-kernel] loading {args.calibration_dir}/*.pt → Ml8Linear swap ...")
        summary = overlay_ml8_kernels(model, args.calibration_dir, args.device)
    else:
        print(f"\n[overlay] loading {args.calibration_dir}/*.pt ...")
        summary = overlay_ml8_weights(model, args.calibration_dir, args.device)
    print(f"\n[overlay summary] {summary['n_loaded']} layers loaded, "
          f"{summary['n_skipped']} skipped")
    print(f"  size: {summary['total_quant_bits']/8/1e9:.2f} GB quantized vs "
          f"{summary['total_orig_bits_fp16']/8/1e9:.2f} GB fp16  "
          f"(ratio {summary['size_ratio_vs_fp16']:.3f})")

    if args.eval_ppl:
        print("\n[ppl-quant] reconstructed model PPL...")
        quant_ppl = eval_ppl_wikitext(model, tok, args.device,
                                       max_tokens=args.ppl_max_tokens)
        print(f"  reconstructed PPL = {quant_ppl['ppl']:.4f}")
        if baseline_ppl is not None:
            delta = quant_ppl["ppl"] - baseline_ppl["ppl"]
            print(f"\n  Δ_PPL = {delta:+.4f}  "
                  f"({delta / baseline_ppl['ppl'] * 100:+.2f}%)")
            if delta < 0.08:
                print(f"  ✓ Δ_PPL < 0.08 — MAD-223 gate PASSED")
            else:
                print(f"  ⚠ Δ_PPL ≥ 0.08 — MAD-223 gate triggers AWQ B.5")


if __name__ == "__main__":
    main()
