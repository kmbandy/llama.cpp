#!/usr/bin/env python3
"""calibrate_ml8_paged.py — calibrate ml8 weights with paged weight access.

MAD-238 iteration 5. Same calibration math as calibrate_ml8.py — reuses its
helpers (compute_hessian, gptq_quantize_linear, eval_ppl_wikitext) directly.
The difference is how the MLP linears' weights are sourced:

    calibrate_ml8.py        : HF loads entire model into VRAM via .to(device)
    calibrate_ml8_paged.py  : HF loads non-MLP weights normally; MLP linears
                              are wp_native.PagedLinear instances that
                              page-fault their .weight from a GGUF via wp_native.

Iteration 5a scope (this script): paged MLP only, dense models (Qwen3.5-4B).
Validates the swap+pager pipeline end-to-end against the known Cell C result.

Iteration 5b (future): page ALL weights (attn, norms, embed) for 35B-A3B.
Requires PagedEmbedding + paged attention linears + per-block resident-vs-paged
classification. Out of scope for the parity gate.

Usage:
    python3 calibrate_ml8_paged.py \\
        --model Qwen/Qwen3.5-4B \\
        --gguf /home/kmbandy/models/Qwen3.5-4B-f16.gguf \\
        --output-dir /home/kmbandy/models/cell-paged \\
        --n-samples 32 --seq-len 1024 \\
        --rotation kronecker --snap-centroids e4m3 \\
        --fit-loss mse --group-size 64 --n-centroids 16 \\
        --eval-ppl --ppl-max-tokens 100000
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-use existing calibration helpers (compute_hessian / gptq_quantize_linear / eval_ppl_wikitext / etc.)
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_ml8 import (  # noqa: E402
    collect_wikitext_calibration,
    compute_hessian,
    gptq_quantize_linear,
    eval_ppl_wikitext,
    find_target_linears,
    filter_by_layer_limit,
)
from centroid_quantizer import CentroidQuantizer  # noqa: E402
from kronecker_rotation import (  # noqa: E402
    KroneckerRotation, random_orthogonal, factor_for_dim, rotate_hessian,
)
from awq import compute_awq_scale, apply_awq_to_weight, absorb_awq_in_reconstruction  # noqa: E402

# Pybind11 weight pager (in repo's python_bindings/wp/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_bindings" / "wp"))
import wp_native  # noqa: E402
from paged_linear import PagedLinear, swap_linears_with_paged  # noqa: E402


# ─── HF naming → GGUF naming map for Qwen MLP linears ──────────────────────

def _qwen_mlp_name_map(model: nn.Module) -> dict[str, str]:
    """Build {HF_module_path → GGUF_tensor_name} for every MLP linear in the model.

    Examples:
      "model.layers.0.mlp.gate_proj"  → "blk.0.ffn_gate.weight"
      "model.layers.5.mlp.up_proj"    → "blk.5.ffn_up.weight"
      "model.layers.31.mlp.down_proj" → "blk.31.ffn_down.weight"
    """
    hf_to_gguf_suffix = {
        "gate_proj": "ffn_gate",
        "up_proj":   "ffn_up",
        "down_proj": "ffn_down",
    }
    name_map = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # Match "model.layers.N.mlp.{gate_proj|up_proj|down_proj}"
        parts = name.split(".")
        if len(parts) < 5 or parts[-2] != "mlp" or parts[-1] not in hf_to_gguf_suffix:
            continue
        # Find the layer index
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        gguf_name = f"blk.{layer_idx}.{hf_to_gguf_suffix[parts[-1]]}.weight"
        name_map[name] = gguf_name
    return name_map


# ─── Pager bootstrap ───────────────────────────────────────────────────────

def build_pager_from_gguf(gguf_path: str, device_idx: int,
                          n_slots: int, prefetch_depth: int = 4,
                          name_filter=None) -> wp_native.WeightPager:
    """Read the GGUF metadata via gguf-py, populate the wp catalog, init the pager.

    n_slots = size of VRAM ring. Pool size = n_slots × max_page_size where
    max_page_size = the largest *cataloged* tensor's size. Caller is responsible
    for VRAM math (per mad-lab GPU utilization rule).

    name_filter: optional callable str→bool. If provided, only tensors whose
    names pass the filter get cataloged. This is critical when only a subset of
    tensors will ever be page-faulted (e.g., MLP-only calibration) — including
    huge unused tensors like token_embd would balloon max_page_size to ~1.27 GB
    and make even modest n_slots OOM (96 slots × 1.27 GB ≈ 122 GB).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib

    pager = wp_native.WeightPager()
    reader = gguf_lib.GGUFReader(gguf_path)
    n_added = n_skipped = 0
    max_sz = 0
    for t in reader.tensors:
        if name_filter is not None and not name_filter(t.name):
            n_skipped += 1
            continue
        pager.add_page(t.name, 0, int(t.data_offset), int(t.n_bytes))
        max_sz = max(max_sz, int(t.n_bytes))
        n_added += 1
    print(f"[pager] catalog: added={n_added} skipped={n_skipped} "
          f"max_page={max_sz/1e6:.1f} MB  pool={n_slots*max_sz/1e9:.2f} GB")
    cfg = wp_native.Config()
    cfg.n_slots = n_slots
    cfg.prefetch_depth = prefetch_depth
    cfg.prefer_async_io = False  # SyncPread path — simpler for first cut
    ok = pager.init_for_device(cfg, device_idx, [gguf_path])
    if not ok:
        raise RuntimeError("wp_native.WeightPager.init_for_device returned False")
    return pager


def _mlp_only_name_filter(name: str) -> bool:
    """Catalog only block MLP weights (ffn_gate / ffn_up / ffn_down)."""
    return name.startswith("blk.") and any(
        name.endswith(f".{suffix}.weight")
        for suffix in ("ffn_gate", "ffn_up", "ffn_down")
    )


# ─── Main driver ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B",
                   help="HF model name. Loaded with empty weights for non-MLP, "
                        "paged for MLP. (Iter 5a: only MLP is paged.)")
    p.add_argument("--gguf", required=True,
                   help="GGUF file that backs the paged MLP weights. Must contain "
                        "the same model as --model; mismatched offsets = silent garbage.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=25)
    p.add_argument("--fit-loss", choices=("mse", "mag_weighted"), default="mse")
    p.add_argument("--mag-weight-p", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--eval-ppl", action="store_true")
    p.add_argument("--ppl-max-tokens", type=int, default=None)
    p.add_argument("--rotation", choices=("none", "kronecker"), default="none")
    p.add_argument("--rotation-seed", type=int, default=42)
    p.add_argument("--rotation-max-b", type=int, default=1024)
    p.add_argument("--snap-centroids", choices=("none", "e4m3"), default="none")
    p.add_argument("--awq", choices=("none", "mean"), default="none")
    p.add_argument("--awq-alpha", type=float, default=0.5)
    p.add_argument("--pager-slots", type=int, default=8,
                   help="WeightPager VRAM ring size. Pool = slots × max_page_size. "
                        "For Qwen3.5-4B, max_page_size ≈ 1.27 GB (token_embd) → "
                        "8 slots = ~10 GB pool. Don't exceed available headroom.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

    print(f"[load-hf] {args.model}  dtype={dtype}  device={args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # NOTE: iter 5a — full HF model load. iter 5b will use init_empty_weights.
    # `torch_dtype` is deprecated and silently ignored in newer transformers;
    # use `dtype=` AND explicit `.to(dtype)` so the dtype is actually honored.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(args.device).to(dtype).eval()
    actual_dtypes = {p.dtype for p in model.parameters()}
    print(f"[load-hf] post-load dtypes={actual_dtypes}")
    if actual_dtypes != {dtype}:
        raise RuntimeError(
            f"Model load did not honor dtype={dtype}; got {actual_dtypes}. "
            f"Paged calibration requires uniform model dtype matching the GGUF.")

    print(f"[pager] building from {args.gguf}  slots={args.pager_slots}")
    pager = build_pager_from_gguf(args.gguf, device_idx, args.pager_slots,
                                   name_filter=_mlp_only_name_filter)
    print(f"[pager] catalog: {pager.n_pages()} tensors  max_page={pager.max_page_size()/1e6:.1f} MB")

    name_map = _qwen_mlp_name_map(model)
    print(f"[swap] HF↔GGUF MLP mapping: {len(name_map)} linears")
    n_swapped = swap_linears_with_paged(model, pager, name_map, dtype=dtype, device_idx=device_idx)
    print(f"[swap] replaced {n_swapped} nn.Linear → PagedLinear")
    if n_swapped == 0:
        raise RuntimeError("swap_linears_with_paged found nothing to swap — check name_map")

    # ─── Calibration corpus + baseline PPL (paged forward proves the swap works) ───
    print(f"[calib] loading {args.n_samples} samples seq_len={args.seq_len}")
    calib = collect_wikitext_calibration(tokenizer, n_samples=args.n_samples,
                                          seq_len=args.seq_len)
    print(f"[calib] got {len(calib)} samples (tokens ≈ {sum(c.numel() for c in calib)})")

    targets = list(find_target_linears(model))   # finds MLP linears (now PagedLinear)
    targets = filter_by_layer_limit(targets, args.max_layers)
    print(f"[targets] {len(targets)} linears to quantize")

    manifest = {"model": args.model, "gguf": args.gguf, "args": vars(args), "results": []}

    if args.eval_ppl:
        print(f"\n[ppl-baseline] computing f16 baseline PPL (paged forward path)...")
        baseline = eval_ppl_wikitext(model, tokenizer, args.device,
                                      max_tokens=args.ppl_max_tokens)
        print(f"  baseline PPL = {baseline['ppl']:.4f}")
        manifest["ppl_baseline"] = baseline

    # ─── Per-layer calibration loop (same math as calibrate_ml8.py) ───
    for i, (name, layer) in enumerate(targets):
        t0 = time.time()
        rows, in_feat = layer.weight.shape   # PagedLinear.weight page-faults here
        print(f"\n[{i+1}/{len(targets)}] {name}  shape=({rows}, {in_feat})")

        W_orig_snapshot = layer.weight.detach().clone()

        collect_awq = args.awq != "none"
        H, n_tok, sum_abs = compute_hessian(layer, calib, model, args.device,
                                            collect_awq=collect_awq)
        t_hess = time.time() - t0
        print(f"  hessian: {H.shape}  n_tok={n_tok}  t={t_hess:.1f}s")

        # AWQ rescale
        awq_s = None
        awq_blob = None
        if collect_awq and sum_abs is not None:
            mean_abs = (sum_abs / max(n_tok, 1)).clamp_min(1e-8)
            awq_s = mean_abs.pow(args.awq_alpha).to(H.device)
            print(f"  awq: kind={args.awq} alpha={args.awq_alpha} "
                  f"s_max={awq_s.max().item():.3f} s_min={awq_s.min().item():.3f}")
            awq_blob = {"kind": args.awq, "alpha": args.awq_alpha, "s": awq_s.detach().cpu()}
            H = H * awq_s.unsqueeze(0) * awq_s.unsqueeze(1)
            W_new = apply_awq_to_weight(layer.weight.float().to(awq_s.device), awq_s)
            layer.weight_override = W_new.to(dtype)
        else:
            # No AWQ: still need to seed weight_override with original so subsequent
            # math operates on a Tensor (not a property page-fault each access).
            layer.weight_override = layer.weight.detach().clone()

        # Rotation
        rotation = None
        rotation_blob = None
        if args.rotation == "kronecker":
            a, b = factor_for_dim(in_feat, max_b=args.rotation_max_b)
            h_a = random_orthogonal(a, seed=args.rotation_seed + i)
            rotation = KroneckerRotation(h_a=h_a, b_dim=b)
            rotation_blob = rotation.to_dict()
            rotation_blob["seed"] = args.rotation_seed + i
            print(f"  rotation: kronecker a={a} b={b}")
            H = rotate_hessian(H, rotation)
            layer.weight_override = rotation.forward(
                layer.weight_override.float().to(H.device)).to(dtype)

        q = CentroidQuantizer(n_centroids=args.n_centroids, n_iter=args.n_iter).to(args.device)
        q.configure(bits=4, sym=True, fit_loss=args.fit_loss,
                    mag_weight_p=args.mag_weight_p,
                    snap_centroids=args.snap_centroids)
        q.hessian_diag = torch.diag(H).clone()

        effective_percdamp = args.percdamp
        if awq_s is not None:
            effective_percdamp = max(args.percdamp, 0.05)

        try:
            # gptq_quantize_linear modifies layer.weight in place — but layer is a
            # PagedLinear and .weight is a @property. Use a temporary nn.Linear shim
            # wrapping weight_override so gptq's `layer.weight.data.copy_(Q...)` works.
            shim = nn.Linear(in_feat, rows, bias=False).to(args.device)
            shim.weight = nn.Parameter(layer.weight_override.to(dtype))
            export = gptq_quantize_linear(shim, H, q,
                                          group_size=args.group_size,
                                          percdamp=effective_percdamp)
            # After GPTQ: shim.weight holds the dequantized rotated/AWQ'd weight.
            layer.weight_override = shim.weight.data.detach().clone()
            del shim
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            layer.weight_override = W_orig_snapshot
            continue
        t_quant = time.time() - t0 - t_hess

        # Inverse rotation, then absorb AWQ — leave weight_override at inference-equivalent.
        if rotation is not None:
            layer.weight_override = rotation.inverse(
                layer.weight_override.float().to(rotation.h_a.device)).to(dtype)
        if awq_s is not None:
            layer.weight_override = absorb_awq_in_reconstruction(
                layer.weight_override.float().to(awq_s.device), awq_s).to(dtype)

        # CRITICAL: weight_override must live on the model's device so the next
        # layer's forward pass (during compute_hessian) finds it on cuda. Rotation
        # math may have moved it to CPU (rotation.h_a is on CPU by design — see
        # KroneckerRotation.to_dict) — pull it back to args.device.
        layer.weight_override = layer.weight_override.to(args.device)

        # Save the blob (identical schema to calibrate_ml8.py output)
        out_path = Path(args.output_dir) / f"{name.replace('.', '_').replace('/', '_')}.pt"
        blob = {
            "name": name,
            "shape": [rows, in_feat],
            "group_size": args.group_size,
            "n_centroids": args.n_centroids,
            "indices": export["indices"].cpu(),
            "centroids_per_group": export["centroids_per_group"].cpu(),
            "scale_per_group": export["scale_per_group"].cpu(),
            "mse": export["mse"],
            "w_snr_db": export["w_snr_db"],
            "y_snr_db": export["y_snr_db"],
            "rel_err": export["rel_err"],
        }
        if rotation_blob is not None:
            blob["rotation"] = rotation_blob
        if awq_blob is not None:
            blob["awq"] = awq_blob
        torch.save(blob, out_path)

        print(f"  saved: {out_path.name}  "
              f"Y_SNR={export['y_snr_db']:.1f}dB  W_SNR={export['w_snr_db']:.1f}dB  "
              f"t_quant={t_quant:.1f}s")

        manifest["results"].append({
            "name": name, "shape": [rows, in_feat],
            "mse": float(export["mse"]),
            "y_snr_db": float(export["y_snr_db"]),
            "w_snr_db": float(export["w_snr_db"]),
            "t_hess_s": float(t_hess),
            "t_quant_s": float(t_quant),
        })

        del H, q
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.eval_ppl:
        print(f"\n[ppl-quant] computing quantized PPL...")
        quantized = eval_ppl_wikitext(model, tokenizer, args.device,
                                       max_tokens=args.ppl_max_tokens)
        print(f"  quantized PPL = {quantized['ppl']:.4f}")
        manifest["ppl_quantized"] = quantized
        delta = quantized["ppl"] - manifest["ppl_baseline"]["ppl"]
        manifest["ppl_delta"] = delta
        print(f"\n  Δ_PPL = {delta:+.4f}")

    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] manifest: {manifest_path}")

    pager.shutdown()


if __name__ == "__main__":
    main()
