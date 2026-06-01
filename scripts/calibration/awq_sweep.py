#!/usr/bin/env python3
"""AWQ (activation-aware weight scaling) lever sweep on the offline ml8 rig.

THE LEVER (bpv-neutral): for input channel k, scale the weight column up by
s[k] and the activation down by 1/s[k]:  W @ x = (W·diag(s)) @ (diag(1/s)·x).
Salient (high-activation) channels get finer *relative* quantization grain at
the SAME bit count — n_centroids/group_size unchanged. This is the first lever
from MAD-256 #1 and the one we lead with: it targets expert *quality* without
spending bits, which is exactly the axis we must win (our experts are already
4.782 bpv < UD's 4.876, but +0.046 PPL behind).

WHY IT'S MEASURABLE FOR FREE IN THIS RIG:
  GPTQ minimizes tr((W-Q) H (W-Q)^T). In AWQ-scaled coords, W' = W·diag(s) and
  the activation Hessian becomes H' = diag(1/s)·H·diag(1/s). The dequantized
  original-space weight is Q = Q'·diag(1/s), and:
      (W - Q'·diag(1/s)) = (W' - Q')·diag(1/s)
  so  tr((W-Q) H (W-Q)^T) = tr((W'-Q') H' (W'-Q')^T).
  => batched_gptq_quantize(W', H')'s reported y_snr_db IS the original-space
     output SNR. Directly comparable to the α=0 baseline. No recompute.

WHY IT'S FOLDABLE FOR MoE:
  All 256 experts of a layer share the same input x (post-norm MoE hidden
  state). A shared per-input-channel scale s (from the shared Hessian diag) can
  be absorbed ONCE into the input path (router/norm), not per-expert. So a win
  here is implementable, not just a paper number.

s[k] = (diag(H)[k])^(α/2)  — i.e. (RMS activation of channel k)^α. Normalized to
mean 1 for numerical cleanliness (a global scalar on s is a no-op: the per-row
group scale absorbs it and the y_snr ratio is scale-invariant — verified by the
α=0 == baseline sanity gate). α=0 → s=1 → exact baseline.

A second variant (--weight-aware) tries the canonical AWQ ratio
s[k] = act[k]^α / wmean[k]^β over a small β grid.

Usage:
  python3 awq_sweep.py --capture route-capture-L0-L20.pt \
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


def quantize_get_ysnr(W_stack, H, gs, nc, snap):
    """Run the real ml8 quantizer on E experts under a single shared H; return
    median y_snr_db over experts. H is [K,K]; broadcast to [E,K,K]."""
    E = W_stack.shape[0]
    H_stack = H.unsqueeze(0).repeat(E, 1, 1)  # fresh copy (damped in place)
    out = batched_gptq_quantize(W_stack, H_stack, n_centroids=nc,
                                group_size=gs, snap_centroids=snap, fit_loss="mse")
    y = out["y_snr_db"]
    y = y[~torch.isnan(y)]
    del H_stack, out
    if W_stack.is_cuda:
        torch.cuda.empty_cache()
    return float(y.median()) if y.numel() else float("nan")


def awq_transform(W_stack, H, s):
    """Apply AWQ column scaling. s: [K] per-input-channel scale (mean ~1).
    Returns (W', H') in scaled coordinates such that the quantizer's y_snr
    equals the original-space output SNR (see module docstring)."""
    inv = 1.0 / s
    Wp = W_stack * s.view(1, 1, -1)              # scale weight columns (input dim K)
    Hp = H * inv.view(-1, 1) * inv.view(1, -1)   # diag(1/s) H diag(1/s)
    return Wp.contiguous(), Hp.contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--gguf-suffix", default="ffn_gate_exps")
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--snap", default="e4m3")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--weight-aware", action="store_true",
                   help="also sweep s = act^alpha / wmean^beta (beta in --betas)")
    p.add_argument("--betas", default="0.25,0.5")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    alphas = [float(a) for a in args.alphas.split(",")]
    betas = [float(b) for b in args.betas.split(",")]
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"loaded {args.capture}: layers {sorted(blob.keys())}  device={dev}")
    print(f"config: gs{args.group_size} nc{args.n_centroids} snap={args.snap}  "
          f"suffix={args.gguf_suffix}  E_sample={args.n_experts}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        N, K = x.shape
        H = (x.t() @ x) / N                                          # [K,K] shared-best
        We = load_expert_weights(args.gguf, int(L), args.gguf_suffix, dev)
        E = min(args.n_experts, We.shape[0])
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        sel = logits.topk(args.top_k, dim=1).indices
        oh = torch.zeros(N, We.shape[0], device=dev, dtype=torch.bool); oh.scatter_(1, sel, True)
        pick = oh.sum(0).argsort(descending=True)[:E]
        W_stack = We[pick].contiguous()                              # [E, out, in]
        act_rms = torch.diagonal(H).clamp_min(1e-12).sqrt()          # [K] RMS act per channel
        wmean = W_stack.abs().mean(dim=(0, 1)).clamp_min(1e-12)      # [K] mean |w| per in-channel

        print(f"\n{'='*72}\nLAYER {L} {args.gguf_suffix}:  E={E}  W_e={tuple(W_stack.shape[1:])}  K={K}")
        print(f"   {'config':<26}{'med Y_SNR':>10}   note")
        base = None
        for a in alphas:
            s = act_rms.pow(a)
            s = s / s.mean()
            Wp, Hp = awq_transform(W_stack, H, s)
            med = quantize_get_ysnr(Wp, Hp, args.group_size, args.n_centroids, args.snap)
            if base is None:
                base = med  # alpha=0 is the no-scaling baseline
            tag = "BASELINE (s=1)" if a == 0.0 else f"AWQ act^{a}"
            print(f"   {tag:<26}{med:>9.2f}   (Δ {med-base:+.2f})")
        if args.weight_aware:
            for a in alphas:
                if a == 0.0:
                    continue
                for b in betas:
                    s = act_rms.pow(a) / wmean.pow(b)
                    s = s / s.mean()
                    Wp, Hp = awq_transform(W_stack, H, s)
                    med = quantize_get_ysnr(Wp, Hp, args.group_size, args.n_centroids, args.snap)
                    print(f"   {'AWQ act^%.2g/w^%.2g'%(a,b):<26}{med:>9.2f}   (Δ {med-base:+.2f})")


if __name__ == "__main__":
    main()
