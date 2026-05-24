#!/usr/bin/env python3
"""diagnose_calibration.py — disambiguate where the SNR loss is coming from.

Runs THREE quantizations on the SAME weight matrix from Qwen3.5-4B layer 0:

  (1) UNIFORM INT4 — 16 evenly-spaced levels per row, no Lloyd-Max,
      no error propagation. Simplest possible 4-bit quant baseline.

  (2) CENTROID NAIVE — CentroidQuantizer LUT + per-row scale, but NO
      GPTQ error propagation. Just snap each column independently.

  (3) CENTROID GPTQ — full CentroidQuantizer + GPTQ error propagation
      (= what calibrate_ml8.py does today).

Each variant runs with multiple fit_loss settings to also test the
hypothesis that mag_weighted p=5 (the MAD-214 KV winner) is wrong for
weights (which are Gaussian-near-zero, not heavy-tailed like activations).

Decision matrix:
  - (1) better than (2) → CentroidQuantizer / Lloyd-Max is broken
  - (2) better than (3) → GPTQ error propagation is broken or wrong-sign
  - mse better than mag_p5 → confirms mag_weighted is wrong loss for weights
  - all of (1)(2)(3) at low SNR → fundamentally wrong scale or shape assumption

Usage:
    python3 scripts/calibration/diagnose_calibration.py \\
        --model Qwen/Qwen3.5-4B --n-samples 32 --seq-len 1024
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from centroid_quantizer import CentroidQuantizer  # noqa: E402


def snr_db(orig: torch.Tensor, recon: torch.Tensor) -> tuple[float, float, float]:
    """Element-wise SNR_dB, MSE, rel_err. (What naive snap optimizes.)"""
    orig_f = orig.float()
    recon_f = recon.float()
    mse = (orig_f - recon_f).pow(2).mean().item()
    sig_pow = orig_f.pow(2).mean().clamp_min(1e-30).item()
    snr = 10.0 * math.log10(sig_pow / max(mse, 1e-30))
    rel = (mse / sig_pow) ** 0.5
    return snr, mse, rel


def output_snr_db(orig: torch.Tensor, recon: torch.Tensor, H: torch.Tensor) -> tuple[float, float, float]:
    """Output-space (activation-space) SNR_dB, weighted by Hessian.

    This is what GPTQ actually optimizes:
        loss = trace((Q - W) H (Q - W)^T) / trace(W H W^T)

    Equivalent to comparing layer OUTPUTS rather than weight matrix elements.
    """
    diff = (orig - recon).float()
    W = orig.float()
    # numerator: sum over rows of diff_row^T H diff_row
    err_pow = (diff @ H @ diff.t()).diagonal().sum().item()
    sig_pow = (W @ H @ W.t()).diagonal().sum().clamp_min(1e-30).item()
    snr = 10.0 * math.log10(sig_pow / max(err_pow, 1e-30))
    rel = (err_pow / sig_pow) ** 0.5
    return snr, err_pow, rel


# ───────────────────────────── Calibration ──────────────────────────────────

def collect_wikitext(tokenizer, n_samples: int, seq_len: int) -> list[torch.Tensor]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    samples = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=seq_len).input_ids
        if ids.shape[1] < seq_len // 4:
            continue
        samples.append(ids)
        if len(samples) >= n_samples:
            break
    return samples


@torch.no_grad()
def compute_hessian(layer, calib, model, dev):
    H_acc = None
    n = 0
    def hook(_m, inputs, _o):
        nonlocal H_acc, n
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        XtX = x.t() @ x
        if H_acc is None:
            H_acc = XtX
        else:
            H_acc += XtX
        n += x.shape[0]
    h = layer.register_forward_hook(hook)
    try:
        for ids in calib:
            model(ids.to(dev))
    finally:
        h.remove()
    return H_acc / max(n, 1), n


# ───────────────────────────── Quantizers ───────────────────────────────────

def quant_uniform_int4(W: torch.Tensor) -> torch.Tensor:
    """16 evenly-spaced signed levels [-1, 1] per row. Symmetric per-row scale."""
    W = W.float()
    # Per-row max abs → scale
    scale = W.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
    # 16 levels span [-1, 1]: centers at (-7.5, -6.5, ..., 6.5, 7.5) / 7.5
    levels = (torch.arange(16, device=W.device, dtype=torch.float32) - 7.5) / 7.5
    x_norm = W / scale  # [rows, cols]
    # Distance to each level
    dist = (x_norm.unsqueeze(-1) - levels).abs()
    idx = dist.argmin(dim=-1)
    return levels[idx] * scale


def quant_centroid_naive(W: torch.Tensor, group_size: int, fit_loss: str,
                         mag_p: float = 5.0, dev=None) -> torch.Tensor:
    """CentroidQuantizer per-group, no GPTQ propagation. Just snap."""
    if dev is None:
        dev = W.device
    rows, cols = W.shape
    Q = torch.zeros_like(W, device=dev)
    q = CentroidQuantizer(n_centroids=16, n_iter=25).to(dev)
    q.configure(bits=4, sym=True, fit_loss=fit_loss, mag_weight_p=mag_p)
    for g_start in range(0, cols, group_size):
        g_end = min(g_start + group_size, cols)
        q.set_group_offset(g_start)
        q.find_params(W[:, g_start:g_end])
        for c in range(g_start, g_end):
            Q[:, c:c+1] = q.quantize(W[:, c:c+1])
    return Q


def quant_centroid_gptq(W: torch.Tensor, H: torch.Tensor, group_size: int,
                        fit_loss: str, mag_p: float = 5.0,
                        percdamp: float = 0.01) -> torch.Tensor:
    """Full GPTQ loop with CentroidQuantizer."""
    dev = H.device
    W = W.float().to(dev).clone()
    rows, cols = W.shape
    H = H.clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(cols, device=dev)
    H[diag, diag] += damp
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    Hinv_chol = torch.linalg.cholesky(H_inv, upper=True)
    q = CentroidQuantizer(n_centroids=16, n_iter=25).to(dev)
    q.configure(bits=4, sym=True, fit_loss=fit_loss, mag_weight_p=mag_p)
    Q = torch.zeros_like(W)
    for col in range(cols):
        if col % group_size == 0:
            g_end = min(col + group_size, cols)
            q.set_group_offset(col)
            q.find_params(W[:, col:g_end])
        w = W[:, col:col+1]
        d = Hinv_chol[col, col]
        qv = q.quantize(w)
        Q[:, col:col+1] = qv
        err = (w - qv) / d
        if col + 1 < cols:
            W[:, col+1:] -= err * Hinv_chol[col, col+1:].unsqueeze(0)
    return Q


# ───────────────────────────── Driver ───────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target", default="model.layers.0.mlp.gate_proj")
    args = p.parse_args()

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.to(args.device).eval()

    print(f"[calib] {args.n_samples} samples × {args.seq_len} tokens")
    calib = collect_wikitext(tok, args.n_samples, args.seq_len)

    # Find target
    layer = None
    for name, mod in model.named_modules():
        if name == args.target:
            layer = mod
            break
    if layer is None:
        print(f"target not found: {args.target}", file=sys.stderr)
        return 1
    W_orig = layer.weight.data.float().clone().to(args.device)
    print(f"[target] {args.target}  shape={tuple(W_orig.shape)}  "
          f"|W|_mean={W_orig.abs().mean().item():.4g}  "
          f"W²_mean={W_orig.pow(2).mean().item():.4g}")

    print(f"[hessian] collecting from {len(calib)} samples...")
    H, n_tok = compute_hessian(layer, calib, model, args.device)
    print(f"          H {tuple(H.shape)}  diag_mean={H.diag().mean().item():.4g}  n_tok={n_tok}")

    # ── Run all variants on the SAME starting W ──
    print()
    print(f"{'variant':40s} {'W_SNR':>8s} {'Y_SNR':>8s}  {'W_relerr':>9s} {'Y_relerr':>9s}")
    print("─" * 82)
    print("(W_SNR = element-wise weight SNR;  Y_SNR = output-space SNR weighted by H)")
    print("(GPTQ optimizes Y_SNR. Naive snap optimizes W_SNR.)")
    print()

    variants = [
        ("1. Uniform INT4 (no LUT, no GPTQ)",
            lambda: quant_uniform_int4(W_orig)),
        ("2a. Centroid naive, fit_loss=mse",
            lambda: quant_centroid_naive(W_orig, args.group_size, "mse")),
        ("2b. Centroid naive, fit_loss=mag_p5",
            lambda: quant_centroid_naive(W_orig, args.group_size, "mag_weighted", 5.0)),
        ("3a. Centroid + GPTQ, fit_loss=mse",
            lambda: quant_centroid_gptq(W_orig, H, args.group_size, "mse")),
        ("3b. Centroid + GPTQ, fit_loss=mag_p5",
            lambda: quant_centroid_gptq(W_orig, H, args.group_size, "mag_weighted", 5.0)),
    ]

    for name, fn in variants:
        Q = fn()
        w_snr, _, w_rel = snr_db(W_orig, Q)
        y_snr, _, y_rel = output_snr_db(W_orig, Q, H)
        print(f"{name:40s} {w_snr:>7.2f}  {y_snr:>7.2f}  {w_rel:>8.2%} {y_rel:>8.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
