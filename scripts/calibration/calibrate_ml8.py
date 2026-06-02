#!/usr/bin/env python3
"""calibrate_ml8.py — drive CentroidQuantizer through a HF model via a minimal GPTQ loop.

MAD-223 Phase B.2.

Replaces auto-gptq (which doesn't install on Python 3.14) with a self-contained
GPTQ implementation. The hard part — Lloyd-Max centroid fitting — is in
CentroidQuantizer; this driver provides:

  1. Calibration data collection (wikitext-2 via HF datasets)
  2. Per-layer Hessian accumulation via forward hooks
  3. Per-column GPTQ snap + error propagation loop
  4. In-place weight replacement so subsequent layers calibrate against
     the QUANTIZED activations from prior layers (matches auto-gptq pattern)
  5. Per-linear save (indices + centroids per group) + manifest

Usage:
    python3 scripts/calibration/calibrate_ml8.py \\
        --model Qwen/Qwen3.5-4B \\
        --output-dir /tmp/ml8-qwen3-4b \\
        --n-samples 64 \\
        --seq-len 2048 \\
        --group-size 128 \\
        --max-layers 1   # MVP: validate pipeline on first layer

Algorithm sketch (per linear with weight W [rows, in_features] and Hessian H):
    H = (1/N) sum_i x_i x_i^T     # in_features x in_features
    H += damp * mean(diag(H)) * I
    L_inv = cholesky(H^-1, upper=True)
    for col in range(in_features):
        if col % group_size == 0:
            quantizer.find_params(W[:, col:col+group_size])
        q = quantizer.quantize(W[:, col:col+1])
        err = (W[:, col:col+1] - q) / L_inv[col, col]
        W[:, col+1:] -= err * L_inv[col, col+1:]
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

# Force unbuffered prints so live `tail -f` works on long runs piped to file.
print = functools.partial(print, flush=True)  # noqa: A001 (deliberate shadow)

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make centroid_quantizer importable
sys.path.insert(0, str(Path(__file__).parent))
from centroid_quantizer import CentroidQuantizer  # noqa: E402
from kronecker_rotation import (  # noqa: E402
    KroneckerRotation, random_orthogonal, factor_for_dim, rotate_hessian,
)
from awq import compute_awq_scale, apply_awq_to_weight  # noqa: E402
from batched_gptq import _cholesky_inv_upper  # noqa: E402


# ───────────────────────────── Calibration data ─────────────────────────────

def collect_wikitext_calibration(tokenizer, n_samples: int = 64, seq_len: int = 2048,
                                  dataset_name: str = "Salesforce/wikitext",
                                  config: str = "wikitext-2-raw-v1") -> list[torch.Tensor]:
    """Return list of input_ids tensors, each [1, seq_len]."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, config, split="train")
    samples = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=seq_len).input_ids
        # Skip very short samples — calibration quality hurts
        if ids.shape[1] < seq_len // 4:
            continue
        samples.append(ids)
        if len(samples) >= n_samples:
            break
    return samples


# ───────────────────────────── Target selection ─────────────────────────────

def find_target_linears(model):
    """Yield (name, module) for each Linear we want to quantize.

    Dense Qwen: mlp.gate_proj, mlp.up_proj, mlp.down_proj.
    MoE Qwen:   mlp.experts.{E}.{gate,up,down}_proj (every expert).

    Attention (q/k/v/o_proj) and the MoE router gate (mlp.gate) are skipped —
    different sensitivity, left un-quantized. Suffix-matching handles both
    shapes because attention uses distinct *_proj suffixes.
    """
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix in ("gate_proj", "up_proj", "down_proj"):
            yield name, mod


def filter_by_layer_limit(targets, max_layers: int | None):
    if max_layers is None:
        return targets
    keep = []
    for name, mod in targets:
        for i in range(max_layers):
            # Match ".layers.<i>." with bounded digits
            tag = f".layers.{i}."
            if tag in name:
                keep.append((name, mod))
                break
    return keep


# ───────────────────────────── Hessian collection ───────────────────────────

@torch.no_grad()
def compute_hessian(layer: nn.Linear, calibration_ids: list[torch.Tensor],
                    model, dev: str, collect_awq: bool = False
                    ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    """Collect H = (1/N) sum X X^T (in_features x in_features) for `layer`.

    If collect_awq=True, also accumulate per-channel sum(|x|) so AWQ scales
    can be computed from the same forward pass without doubling calibration time.
    Returns (H, n_tokens, sum_abs_per_channel_or_None).
    """
    H_acc = None
    sum_abs = None  # only filled when collect_awq=True
    n_total = 0

    def hook(module, inputs, output):
        nonlocal H_acc, sum_abs, n_total
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).float()  # [N, in_features]
        XtX = x.t() @ x
        if H_acc is None:
            H_acc = XtX
        else:
            H_acc += XtX
        if collect_awq:
            sa = x.abs().sum(dim=0)
            if sum_abs is None:
                sum_abs = sa
            else:
                sum_abs += sa
        n_total += x.shape[0]

    h = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():            # Hessian is forward-only statistics — never
            for ids in calibration_ids:  # build/retain a backward graph. (Missing this
                model(ids.to(dev))       # retained the full-model graph → ~11GB → OOM.)
    finally:
        h.remove()

    if H_acc is None:
        raise RuntimeError(f"No activations collected for {layer}")
    return H_acc / max(n_total, 1), n_total, sum_abs


# ───────────────────────────── GPTQ loop ────────────────────────────────────

@torch.no_grad()
def gptq_quantize_linear(layer: nn.Linear, H: torch.Tensor,
                         quantizer: CentroidQuantizer,
                         group_size: int = 128,
                         percdamp: float = 0.01) -> dict:
    """Quantize `layer.weight` in-place using the GPTQ algorithm.

    Returns the quantizer's export dict (indices + centroids_per_group).
    `layer.weight.data` is replaced with the dequantized values so subsequent
    layers see the post-quantization output.
    """
    dev = H.device
    W = layer.weight.data.float().to(dev).clone()  # [rows, in_features]
    out_rows, in_features = W.shape

    # Save undamped H for the post-quant Y_SNR metric (the damping is for
    # numerical stability; it's not part of the actual reconstruction loss).
    H_orig = H.clone()

    # Damping — needed for numerical stability in Cholesky.
    damp = percdamp * torch.mean(torch.diag(H))

    # Upper-Cholesky factor of (H+damp)^-1 for triangular GPTQ error propagation.
    # `_cholesky_inv_upper` reproduces the classic
    #   L = cholesky(H+damp); H_inv = cholesky_inverse(L); cholesky(H_inv, upper=True)
    # bit-for-bit on well-conditioned H (attempt 0), and on an ill-conditioned
    # faithful-acts Hessian it symmetrizes + escalates the damping until PD rather
    # than raising (which upstream would turn into a bf16 backfill). The batched
    # path uses the same helper, so scalar and batched stay consistent.
    eye = torch.eye(in_features, device=dev, dtype=H.dtype)
    try:
        Hinv_chol, _ = _cholesky_inv_upper(H, damp, eye)  # upper triangular
    except RuntimeError as e:
        raise RuntimeError(f"Cholesky failed (try higher percdamp): {e}") from e

    quantizer.reset_capture()
    Q = torch.zeros_like(W)

    for col in range(in_features):
        # Fit centroids at each group boundary
        if col % group_size == 0:
            g_end = min(col + group_size, in_features)
            quantizer.set_group_offset(col)
            quantizer.find_params(W[:, col:g_end])

        w = W[:, col:col+1]                   # [rows, 1]
        d = Hinv_chol[col, col]
        q = quantizer.quantize(w)             # [rows, 1] dequantized
        Q[:, col:col+1] = q

        # GPTQ error propagation
        err = (w - q) / d
        if col + 1 < in_features:
            # Vectorized: err [rows, 1] * Hinv_chol[col, col+1:] [in_features - col - 1]
            W[:, col+1:] -= err * Hinv_chol[col, col+1:].unsqueeze(0)

    # Reconstruction quality. TWO metrics:
    #   - W_SNR: element-wise weight reconstruction. What naive snap optimizes.
    #   - Y_SNR: output-space reconstruction weighted by H. What GPTQ optimizes.
    # Y_SNR is the one that matters for inference quality (it measures how
    # close the layer's OUTPUT activations are to the original layer's).
    # Per the diagnose_calibration.py findings, MSE+GPTQ should give Y_SNR
    # >25 dB on Qwen-class MLP linears.
    import math
    orig = layer.weight.data.float().to(dev)
    diff = orig - Q
    # Element-wise
    mse = diff.pow(2).mean().item()
    sig_w = orig.pow(2).mean().clamp_min(1e-30).item()
    w_snr_db = 10.0 * math.log10(sig_w / max(mse, 1e-30))
    # Output-space (use H_orig — undamped — so we measure actual reconstruction
    # loss, not the regularized one we used during Cholesky).
    err_y = (diff @ H_orig @ diff.t()).diagonal().sum().item()
    sig_y = (orig @ H_orig @ orig.t()).diagonal().sum().clamp_min(1e-30).item()
    y_snr_db = 10.0 * math.log10(sig_y / max(err_y, 1e-30))
    rel_err = (mse / sig_w) ** 0.5

    # Replace weights so the next layer's calibration sees quantized output
    layer.weight.data.copy_(Q.to(layer.weight.dtype))

    export = quantizer.export()
    export["mse"] = mse
    export["w_snr_db"] = w_snr_db
    export["y_snr_db"] = y_snr_db
    export["rel_err"] = rel_err
    return export


# ───────────────────────────── PPL eval ─────────────────────────────────────

@torch.no_grad()
def eval_ppl_wikitext(model, tokenizer, dev: str, seq_len: int = 2048,
                     stride: int = 1024, max_tokens: int | None = None) -> dict:
    """Compute wikitext-2 test-split PPL via sliding-window evaluation.

    Standard pattern (matches HF eval + llama.cpp --perplexity):
      1. Concatenate all test text, tokenize once
      2. Slide a window of seq_len with overlap (stride)
      3. Compute next-token CE loss on the non-overlapping tail
      4. PPL = exp(mean(loss))
    """
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
        # Mask out the overlap tokens (already scored)
        target_ids[:, :-target_len] = -100
        outputs = model(input_ids, labels=target_ids)
        # outputs.loss is mean over unmasked tokens
        # Re-compute: nll_sum gets total CE, normalize at end
        n_unmasked = (target_ids != -100).sum().item()
        nll_sum += outputs.loss.item() * n_unmasked
        n_pred += n_unmasked
        prev_end = end
        if end >= n_tokens:
            break

    avg_nll = nll_sum / max(n_pred, 1)
    ppl = math.exp(avg_nll)
    return {"ppl": ppl, "avg_nll": avg_nll, "n_tokens_scored": n_pred}


# ───────────────────────────── Driver ───────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=25)
    # DEFAULT FIT_LOSS: MSE for weights (not mag_weighted).
    # The MAD-214 winner `mag_weighted p=5` was fit on KV-CACHE ACTIVATIONS,
    # which have heavy outliers. Weights are roughly Gaussian centered at zero
    # — mag_weighted p=5 pushes centroids to empty tails and starves the
    # dense center. Diagnosed via scripts/calibration/diagnose_calibration.py:
    # for Qwen3.5-4B layer 0 mlp.gate_proj, MSE gave 27.67 dB output-SNR vs
    # mag_p5's 17.01 dB. 10 dB difference.
    p.add_argument("--fit-loss", choices=("mse", "mag_weighted"), default="mse")
    p.add_argument("--mag-weight-p", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Limit to first N transformer layers (fast iteration)")
    p.add_argument("--eval-ppl", action="store_true",
                   help="Compute wikitext-2 test-split PPL before and after "
                        "quantization. Reports Δ_PPL vs f16 baseline.")
    p.add_argument("--ppl-max-tokens", type=int, default=None,
                   help="Cap PPL eval tokens (default: full test split ~280k).")
    # MAD-223 Phase B.3 (calibration sweep): optional Kronecker rotation on weights.
    # Saturday's Hadamard finding ([[turbo-fp8-calibration-hadamard-vs-fit]]) was on
    # KV cache; this lever ports the same idea to weight quantization. Q = H_a ⊗ H_b
    # where H_b is Sylvester (power of 2 ≤ 1024) and H_a is small random orthogonal.
    # Stored factored in the per-layer blob; reconstruct_model.py absorbs it on overlay.
    p.add_argument("--rotation", choices=("none", "kronecker"), default="none",
                   help="Per-layer input rotation applied before GPTQ quantization. "
                        "'kronecker' picks (a, b) for each layer via factor_for_dim().")
    p.add_argument("--rotation-seed", type=int, default=42,
                   help="Seed for the H_a random orthogonal factor of each layer's "
                        "rotation. Per-layer seeds are derived as base_seed + layer_idx "
                        "so multi-layer runs are reproducible.")
    p.add_argument("--rotation-max-b", type=int, default=1024,
                   help="Largest H_b Sylvester dim allowed. Defaults to the existing "
                        "FWHT kernel's upper bound so the factorization stays runtime-compatible.")
    # MAD-223 Phase B.3 (true ml8): snap Lloyd-Max centroids to the E4M3 lattice
    # so the LUT lookup feeds FP8 WMMA directly at inference. 'none' is the
    # back-compat path (float centroids → FP16 WMMA); 'e4m3' is the production
    # ml8 path.
    p.add_argument("--snap-centroids", choices=("none", "e4m3"), default="none",
                   help="Snap fitted centroids to E4M3 lattice. 'e4m3' unlocks the "
                        "FP8 WMMA inference path the ml8 storage format is designed for.")
    # AWQ rescaling: protects salient input channels from group-shared quant scale waste.
    # Applied BEFORE rotation in the pipeline; both transforms compose into the
    # reconstructed weight at HF eval time via reconstruct_inference_weight.
    p.add_argument("--awq", choices=("none", "mean"), default="none",
                   help="Per-input-channel AWQ rescaling. 'mean' computes "
                        "s_i = mean(|x_i|)^alpha from calibration activations.")
    p.add_argument("--awq-alpha", type=float, default=0.5,
                   help="AWQ scaling exponent. 0.5 is the AWQ paper default; "
                        "0.0 disables (s=1 everywhere); 1.0 maximally aggressive.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[load] {args.model}  dtype={dtype}  device={args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(args.device).eval()

    print(f"[calib] loading {args.n_samples} samples seq_len={args.seq_len}")
    calib = collect_wikitext_calibration(tokenizer, n_samples=args.n_samples,
                                          seq_len=args.seq_len)
    print(f"[calib] got {len(calib)} samples "
          f"(tokens total ≈ {sum(c.numel() for c in calib)})")

    targets = list(find_target_linears(model))
    targets = filter_by_layer_limit(targets, args.max_layers)
    print(f"[targets] {len(targets)} linears to quantize")

    manifest = {"model": args.model, "args": vars(args), "results": []}

    # Baseline PPL on the f16 model BEFORE any quantization
    if args.eval_ppl:
        print(f"\n[ppl-baseline] computing f16 baseline PPL...")
        baseline = eval_ppl_wikitext(model, tokenizer, args.device,
                                      max_tokens=args.ppl_max_tokens)
        print(f"  baseline PPL = {baseline['ppl']:.4f}  "
              f"(n_tokens={baseline['n_tokens_scored']})")
        manifest["ppl_baseline"] = baseline

    for i, (name, layer) in enumerate(targets):
        t0 = time.time()
        rows, in_feat = layer.weight.shape
        print(f"\n[{i+1}/{len(targets)}] {name}  shape=({rows}, {in_feat})")

        # Snapshot the original layer weight BEFORE any AWQ/rotation/GPTQ mutation.
        # If anything in the pipeline below throws, we restore from this snapshot
        # so subsequent layers' Hessian collection sees correct activations.
        # (Cell D 2026-05-24 PPL explosion bug: a Cholesky failure on layer 25
        # down_proj left the weight in AWQ-rescaled + rotated state, corrupting
        # all downstream layer calibrations.)
        W_orig_snapshot = layer.weight.data.clone()

        collect_awq = args.awq != "none"
        H, n_tok, sum_abs = compute_hessian(layer, calib, model, args.device,
                                            collect_awq=collect_awq)
        t_hess = time.time() - t0
        print(f"  hessian: {H.shape}, "
              f"diag_mean={H.diag().mean().item():.4g}, "
              f"n_tok={n_tok}, t={t_hess:.1f}s")

        # Optional AWQ rescaling: applied BEFORE rotation. s_i = (sum_abs/N)^alpha.
        # Rescale W column-wise (divide salient columns), and rescale H accordingly:
        # H_awq = diag(s) @ H @ diag(s)  (because x_awq = x * s → H_awq = E[x_awq^T x_awq]).
        awq_s = None
        awq_blob = None
        if collect_awq and sum_abs is not None:
            mean_abs = (sum_abs / max(n_tok, 1)).clamp_min(1e-8)
            awq_s = mean_abs.pow(args.awq_alpha).to(H.device)
            print(f"  awq: kind={args.awq} alpha={args.awq_alpha} "
                  f"s_max={awq_s.max().item():.3f} s_min={awq_s.min().item():.3f}")
            awq_blob = {"kind": args.awq, "alpha": args.awq_alpha, "s": awq_s.detach().cpu()}
            # Rescale H: diag(s) @ H @ diag(s)
            H = H * awq_s.unsqueeze(0) * awq_s.unsqueeze(1)
            # Rescale W in place
            W_dtype = layer.weight.dtype
            W_new = apply_awq_to_weight(layer.weight.data.float().to(awq_s.device), awq_s)
            layer.weight.data.copy_(W_new.to(W_dtype))

        # Optional rotation: rotate W and H into the rotated basis. GPTQ then runs
        # exactly as before on the rotated tensors; we restore the layer's weight
        # to the inference-equivalent (un-rotated) form after quantization so the
        # next layer's Hessian collection sees the right activation distribution.
        rotation = None
        rotation_blob = None
        if args.rotation == "kronecker":
            a, b = factor_for_dim(in_feat, max_b=args.rotation_max_b)
            h_a = random_orthogonal(a, seed=args.rotation_seed + i)
            rotation = KroneckerRotation(h_a=h_a, b_dim=b)
            rotation_blob = rotation.to_dict()
            rotation_blob["seed"] = args.rotation_seed + i
            print(f"  rotation: kronecker a={a} b={b} (d={in_feat})")

            # Rotate H: H_rot = Q.T @ H @ Q
            H = rotate_hessian(H, rotation)
            # Rotate W in-place so GPTQ operates on the rotated weight
            W_dtype = layer.weight.dtype
            layer.weight.data.copy_(
                rotation.forward(layer.weight.data.float().to(H.device)).to(W_dtype)
            )

        q = CentroidQuantizer(n_centroids=args.n_centroids,
                              n_iter=args.n_iter).to(args.device)
        q.configure(bits=4, sym=True,
                   fit_loss=args.fit_loss,
                   mag_weight_p=args.mag_weight_p,
                   snap_centroids=args.snap_centroids)
        q.hessian_diag = torch.diag(H).clone()

        # AWQ-rescaled Hessian needs more damping for Cholesky stability —
        # diag(s) @ H @ diag(s) widens the eigenvalue spread proportionally to
        # min(s)². Default percdamp=0.01 of the rescaled mean diagonal is too
        # weak to lift the smallest eigenvalue above zero on down_proj layers
        # whose activations have wide channel-magnitude spread.
        # Measured 2026-05-24: layer 25 down_proj with s_min=0.236 → Cholesky failed.
        effective_percdamp = args.percdamp
        if awq_s is not None:
            effective_percdamp = max(args.percdamp, 0.05)
            if effective_percdamp != args.percdamp:
                print(f"  percdamp: bumped {args.percdamp:.3g} → {effective_percdamp:.3g} for AWQ stability")

        try:
            export = gptq_quantize_linear(layer, H, q,
                                          group_size=args.group_size,
                                          percdamp=effective_percdamp)
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            # Restore the original weight so subsequent layers see uncorrupted
            # forward activations during their Hessian collection.
            layer.weight.data.copy_(W_orig_snapshot)
            continue
        t_quant = time.time() - t0 - t_hess

        # Inverse-rotate the dequantized layer weight back to inference-equivalent
        # form (so subsequent layers' Hessian collection sees the correct activations).
        # The blob still stores the rotated quantized form; reconstruct_model.py
        # applies the inverse rotation again when overlaying.
        if rotation is not None:
            W_dtype = layer.weight.dtype
            layer.weight.data.copy_(
                rotation.inverse(layer.weight.data.float().to(rotation.h_a.device)).to(W_dtype)
            )

        # Absorb AWQ scale back into the layer's weight so subsequent layers see
        # the inference-equivalent activations (W_inference = W_rescaled * s per col).
        if awq_s is not None:
            W_dtype = layer.weight.dtype
            from awq import absorb_awq_in_reconstruction
            W_back = absorb_awq_in_reconstruction(
                layer.weight.data.float().to(awq_s.device), awq_s
            )
            layer.weight.data.copy_(W_back.to(W_dtype))

        out_path = Path(args.output_dir) / f"{name.replace('.', '_').replace('/', '_')}.pt"
        blob = {
            "name": name,
            "shape": [rows, in_feat],
            "group_size": args.group_size,
            "n_centroids": args.n_centroids,
            # Reconstruction:
            #   W_rot[r, c] = centroids_per_group[c // group_size][indices[r, c]]
            #              * scale_per_group[r, c // group_size]
            #   W_inference = W_rot @ Q.T   (rotation.inverse, applied in reconstruct_model.py)
            "indices": export["indices"].cpu(),                    # [rows, in_features] int8 (values 0..15)
            "centroids_per_group": export["centroids_per_group"].cpu(),  # [n_groups, n_centroids] fp32
            "scale_per_group": export["scale_per_group"].cpu(),    # [rows, n_groups] fp32
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
              f"groups={export['centroids_per_group'].shape[0]}  "
              f"Y_SNR={export['y_snr_db']:.1f}dB  "
              f"W_SNR={export['w_snr_db']:.1f}dB  "
              f"t_quant={t_quant:.1f}s")

        manifest["results"].append({
            "name": name,
            "shape": [rows, in_feat],
            "n_groups": int(export["centroids_per_group"].shape[0]),
            "mse": float(export["mse"]),
            "w_snr_db": float(export["w_snr_db"]),
            "y_snr_db": float(export["y_snr_db"]),
            "rel_err": float(export["rel_err"]),
            "t_hess_s": float(t_hess),
            "t_quant_s": float(t_quant),
        })

        del H, q
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Post-quantization PPL on the in-place modified model
    if args.eval_ppl:
        print(f"\n[ppl-quant] computing quantized PPL...")
        quantized = eval_ppl_wikitext(model, tokenizer, args.device,
                                       max_tokens=args.ppl_max_tokens)
        print(f"  quantized PPL = {quantized['ppl']:.4f}")
        manifest["ppl_quantized"] = quantized
        delta = quantized["ppl"] - manifest["ppl_baseline"]["ppl"]
        manifest["ppl_delta"] = delta
        print(f"\n  Δ_PPL = {delta:+.4f}  "
              f"({delta / manifest['ppl_baseline']['ppl'] * 100:+.2f}%)")
        # MAD-223 gate: <0.08 PPL → proceed without AWQ phase B.5
        if delta < 0.08:
            print(f"  ✓ Δ_PPL < 0.08 — MAD-223 gate PASSED (AWQ phase B.5 not needed)")
        else:
            print(f"  ⚠ Δ_PPL ≥ 0.08 — MAD-223 gate triggers AWQ phase B.5 consideration")

    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
