#!/usr/bin/env python3
"""Surgical resolution test on the down_proj FLOOR (MAD-256). down_proj is the
worst-quantizing kind (20-22 dB vs gate/up 24-32) and eats the fat-tailed SwiGLU
intermediate. Uniform gs32 wastes bits on gate/up (already fine); spending
resolution ONLY on down is the max-PPL-per-bit play. down is 1/3 of expert
params, so a +1bpv bump on down = +0.33 bpv OVERALL → experts 4.25→4.58, still
UNDER UD's 4.876.

Reuses lever_sweep's reconstruction (true SwiGLU intermediate H_down).
Reports med Y_SNR on down for nc16/nc32 x gs64/gs32, vs the nc16/gs64 baseline.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from lever_sweep import load_experts, build_down_H, busiest, quantize

# (label, group_size, n_centroids, bpv_note)  — down_proj only.
# bpv(this kind) = log2(nc) idx + 32/gs fp32-scale  (centroid LUT amortized).
CONFIGS = [
    ("nc16 gs64  (BASELINE)", 64, 16, "4.50 bpv/kind"),
    ("nc16 gs128",           128, 16, "4.25 bpv/kind (coarser scale)"),
    ("nc32 gs64",             64, 32, "5.50 bpv/kind"),
    ("nc32 gs128",           128, 32, "5.25 bpv/kind (the ask)"),
    ("nc32 gs256",           256, 32, "5.125 bpv/kind"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--snap", default="e4m3")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"down_proj resolution sweep  snap={args.snap}  E={args.n_experts}  layers {sorted(blob.keys())}")
    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        Eall = blob[L]["num_experts"]
        Wg = load_experts(args.gguf, int(L), "ffn_gate_exps", dev)
        Wu = load_experts(args.gguf, int(L), "ffn_up_exps", dev)
        H, n_tot = build_down_H(x, logits, Wg, Wu, args.top_k, dev); del Wg, Wu
        Wd = load_experts(args.gguf, int(L), "ffn_down_exps", dev)
        pick = busiest(logits, min(args.n_experts, Eall), args.top_k, Eall, dev)
        W_stack = Wd[pick].contiguous(); del Wd
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"\n{'='*72}\nL{L} down_proj:  E={W_stack.shape[0]}  W_e={tuple(W_stack.shape[1:])}  I={H.shape[0]}")
        print(f"   {'config':<24}{'med Y_SNR':>10}   bpv note")
        base = None
        for label, gs, nc, note in CONFIGS:
            med = quantize(W_stack, H, gs, nc, args.snap, "mse", 5.0)
            if base is None:
                base = med
            print(f"   {label:<24}{med:>9.2f}   {note}   (Δ {med-base:+.2f})")


if __name__ == "__main__":
    main()
