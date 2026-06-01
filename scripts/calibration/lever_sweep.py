#!/usr/bin/env python3
"""Matched-lever sweep: test the bpv-neutral levers that fit PER-TOKEN outliers,
which AWQ (per-channel) could not (MAD-256, KG b7b79682).

LEVERS:
  fit_loss=mag_weighted (mag_weight_p) — up-weights high-|sample| values in the
    Lloyd-Max centroid fit. High-magnitude samples ARE the outlier-token
    contributions, so this is per-sample/per-token-aware and TRULY bpv-neutral
    (only the centroid objective changes; bit layout identical). This is the
    lead lever for Qwen's per-token outliers.
  group_size (gs32/gs16 vs gs64) — finer per-group scales adapt to local
    magnitude regardless of channel/token alignment. Small bpv cost (down-only
    gs32 ≈ +0.25 bpv on down_proj ≈ +0.08 bpv overall); measured as a fallback.

KINDS:
  gate_proj / up_proj consume x (router input, captured directly).
  down_proj consumes the reconstructed SwiGLU intermediate silu(gate*x)*(up*x)
    (see down_proj_rig.py; matches calibrate_ml8_paged.py:875).

All at fixed nc16/e4m3 (the shipped 4.25 bpv expert config). Baseline row =
mse/gs64 must reproduce the overnight Y_SNR (rig-faithfulness gate).

Usage:
  python3 lever_sweep.py --capture /home/kmbandy/models/route-capture-L0-L20.pt \
      --gguf /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf \
      --kinds gate,up,down --n-experts 64 --device cuda:0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from batched_gptq import batched_gptq_quantize  # noqa: E402


def load_experts(gguf_path, layer, suffix, device):
    r = gguf.GGUFReader(gguf_path)
    name = f"blk.{layer}.{suffix}.weight"
    t = next((t for t in r.tensors if t.name == name), None)
    if t is None:
        raise RuntimeError(f"{name} not found")
    arr = np.asarray(t.data, dtype=np.uint8).copy()
    shp = [int(s) for s in reversed(list(t.shape))]
    return torch.from_numpy(arr).view(torch.bfloat16).reshape(*shp).to(torch.float32).to(device)


def build_down_H(x, logits, Wg, Wu, top_k, device):
    """Pooled shared H_down = E[interm interm^T] over all routed (token,expert)."""
    N, K = x.shape
    Eall, I, _ = Wg.shape
    sel = logits.topk(top_k, dim=1).indices
    oh = torch.zeros(N, Eall, device=device, dtype=torch.bool); oh.scatter_(1, sel, True)
    H = torch.zeros(I, I, device=device, dtype=torch.float32); n_tot = 0
    for e in range(Eall):
        idx = oh[:, e].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        Xe = x[idx]
        interm = F.silu(Xe @ Wg[e].t()) * (Xe @ Wu[e].t())
        H += interm.t() @ interm; n_tot += idx.numel()
    return H / max(n_tot, 1), n_tot


def busiest(logits, n_experts, top_k, Eall, device):
    N = logits.shape[0]
    sel = logits.topk(top_k, dim=1).indices
    oh = torch.zeros(N, Eall, device=device, dtype=torch.bool); oh.scatter_(1, sel, True)
    return oh.sum(0).argsort(descending=True)[:n_experts]


def quantize(W_stack, H, gs, nc, snap, fit_loss, mag_p):
    E = W_stack.shape[0]
    H_stack = H.unsqueeze(0).repeat(E, 1, 1)
    out = batched_gptq_quantize(W_stack, H_stack, n_centroids=nc, group_size=gs,
                                snap_centroids=snap, fit_loss=fit_loss, mag_weight_p=mag_p)
    y = out["y_snr_db"]; y = y[~torch.isnan(y)]
    del H_stack, out
    if W_stack.is_cuda:
        torch.cuda.empty_cache()
    return float(y.median()) if y.numel() else float("nan")


# (label, group_size, fit_loss, mag_p, bpv_note)
CONFIGS = [
    ("mse  gs64  (BASELINE)",      64, "mse",          5.0, "4.25 shipped"),
    ("mag_weighted p3  gs64",      64, "mag_weighted", 3.0, "bpv-NEUTRAL"),
    ("mag_weighted p5  gs64",      64, "mag_weighted", 5.0, "bpv-NEUTRAL"),
    ("mag_weighted p8  gs64",      64, "mag_weighted", 8.0, "bpv-NEUTRAL"),
    ("mse  gs32",                  32, "mse",          5.0, "+~0.25 bpv (this kind)"),
    ("mag_weighted p5  gs32",      32, "mag_weighted", 5.0, "+~0.25 bpv (this kind)"),
]

SUFFIX = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--kinds", default="gate,up,down")
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--snap", default="e4m3")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    kinds = [k.strip() for k in args.kinds.split(",")]
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"loaded {args.capture}: layers {sorted(blob.keys())}  device={dev}")
    print(f"levers @ nc{args.n_centroids}/{args.snap}  kinds={kinds}  E_sample={args.n_experts}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        Eall = blob[L]["num_experts"]
        pick = busiest(logits, min(args.n_experts, Eall), args.top_k, Eall, dev)
        for kind in kinds:
            We = load_experts(args.gguf, int(L), SUFFIX[kind], dev)
            if kind == "down":
                Wg = load_experts(args.gguf, int(L), "ffn_gate_exps", dev)
                Wu = load_experts(args.gguf, int(L), "ffn_up_exps", dev)
                H, n_tot = build_down_H(x, logits, Wg, Wu, args.top_k, dev)
                del Wg, Wu
            else:
                H = (x.t() @ x) / x.shape[0]; n_tot = x.shape[0]
            W_stack = We[pick].contiguous(); del We
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"\n{'='*72}\nL{L} {kind}_proj:  E={W_stack.shape[0]}  W_e={tuple(W_stack.shape[1:])}  "
                  f"K={H.shape[0]}  n_tok={n_tot}")
            print(f"   {'config':<28}{'med Y_SNR':>10}   bpv note")
            base = None
            for label, gs, fl, mp, note in CONFIGS:
                med = quantize(W_stack, H, gs, args.n_centroids, args.snap, fl, mp)
                if base is None:
                    base = med
                print(f"   {label:<28}{med:>9.2f}   {note}   (Δ {med-base:+.2f})")
            del W_stack, H
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
