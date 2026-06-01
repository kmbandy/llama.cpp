#!/usr/bin/env python3
"""Offline sweep of ml8 QUANTIZER levers (group_size / n_centroids / e4m3) using the
REAL ml8 quantizer (batched_gptq_quantize) on captured experts + the SHARED Hessian
(confirmed best by quantize_both_ways.py).

The Hessian/data dimension is settled (shared H is best; per-expert hurts). This rig
isolates the quantizer-side levers that actually carry the +0.046 gap vs Q4_K_XL —
all offline, no calibration run.

Baseline config (gs64/nc16/e4m3) should reproduce the overnight gate_proj Y_SNR
(~24.28 dB) — that's the rig-faithfulness cross-check.

Usage:
  python3 sweep_quantizer.py --capture route-capture-L0-L20.pt \
      --gguf /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf \
      --gguf-suffix ffn_gate_exps --n-experts 64 --device cuda:0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from batched_gptq import batched_gptq_quantize  # noqa: E402


def load_expert_weights(gguf_path, layer, suffix, device):
    r = gguf.GGUFReader(gguf_path)
    name = f"blk.{layer}.{suffix}.weight"
    t = next((t for t in r.tensors if t.name == name), None)
    if t is None:
        raise RuntimeError(f"{name} not found")
    arr = np.asarray(t.data, dtype=np.uint8).copy()
    shp = [int(s) for s in reversed(list(t.shape))]
    return torch.from_numpy(arr).view(torch.bfloat16).reshape(*shp).to(torch.float32).to(device)


# (label, group_size, n_centroids, snap, approx_bpv_note)
CONFIGS = [
    ("baseline gs64 nc16 e4m3", 64, 16, "e4m3", "~4.25 (shipped)"),
    ("gs32   nc16 e4m3",        32, 16, "e4m3", "higher (2x group meta)"),
    ("gs128  nc16 e4m3",       128, 16, "e4m3", "lower (~4.1)"),
    ("gs64   nc32 e4m3",        64, 32, "e4m3", "higher (5-bit idx)"),
    ("gs64   nc16 NONE",        64, 16, "none", "~4.25, fp16 centroids (e4m3-cost isolation)"),
    ("gs32   nc32 e4m3",        32, 32, "e4m3", "highest (rich)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--gguf-suffix", default="ffn_gate_exps")
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"loaded {args.capture}: layers {sorted(blob.keys())}  device={dev}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        N, K = x.shape
        H_shared = (x.t() @ x) / N                                  # [K, K] confirmed-best
        We = load_expert_weights(args.gguf, int(L), args.gguf_suffix, dev)  # [E, out, in]
        E = min(args.n_experts, We.shape[0])
        # busiest experts (by router top-k) for a representative sample
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        sel = logits.topk(args.top_k, dim=1).indices
        oh = torch.zeros(N, We.shape[0], device=dev, dtype=torch.bool); oh.scatter_(1, sel, True)
        pick = oh.sum(0).argsort(descending=True)[:E]
        W_stack = We[pick].contiguous()                            # [E, out, in]
        print(f"\n{'='*72}\nLAYER {L} {args.gguf_suffix}:  E_sample={E}  W_e={tuple(W_stack.shape[1:])}  K={K}")
        print(f"   {'config':<26}{'med Y_SNR':>10}   {'bpv note'}")
        base = None
        for label, gs, nc, snap, bpv in CONFIGS:
            H_stack = H_shared.unsqueeze(0).repeat(E, 1, 1)        # fresh (batched_gptq damps in place)
            out = batched_gptq_quantize(W_stack, H_stack, n_centroids=nc,
                                        group_size=gs, snap_centroids=snap, fit_loss="mse")
            y = out["y_snr_db"]
            y = y[~torch.isnan(y)]
            med = float(y.median()) if y.numel() else float("nan")
            if base is None:
                base = med
            delta = med - base
            print(f"   {label:<26}{med:>9.2f}   {bpv}   (Δvs base {delta:+.2f})")
            del H_stack, out
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
