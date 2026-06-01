#!/usr/bin/env python3
"""LEAD 4-bit lever test (MAD-256, KG b7adae6e): does REWEIGHTING the GPTQ Hessian
beat the plain pooled H=XᵀX on down_proj? Three literature lines converge here:
  - MoEQuant AGQ (2505.03804): weight by router-gate affinity, H=(X⊙c)Xᵀ.
  - RSQ (2503.01820): weight by token importance, H=X·R²·Xᵀ.
  - per-token-outlier diagnosis (ours): outlier tokens (kurtosis 120-183) DOMINATE
    the shared H and pull the codebook toward fitting noise → DOWN-weight/clip them.

All bit-free, uniform, drop-in to Hessian accumulation. We test on down_proj (the
floor, SwiGLU-fed) using the reconstructed intermediate.

RIGOR (no cheating the metric): held-out token split. Candidate Hessians are built
on FIT tokens; Y_SNR is evaluated on EVAL tokens under the TRUE output error —
reported under BOTH an unweighted eval-H and a routing-affinity-weighted eval-H
(the latter ≈ how much each token's expert output is actually used → closest PPL
proxy). A reweighting that only helps its own objective won't show up on held-out
eval. Sanity: 'plain' candidate == current pipeline baseline.

Usage:
  python3 reweighted_hessian_rig.py --capture /home/kmbandy/models/route-capture-L0-L20.pt \
      --gguf /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf --n-experts 64 --device cuda:0
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
from lever_sweep import load_experts, busiest  # reuse loaders


def reconstruct_pairs(x, logits, Wg, Wu, top_k, device):
    """Return per-(token,expert) intermediates + gate weights + token ids.
    INTERM [P, I], GATE [P] (softmax over selected top-k logits), TOK [P]."""
    N, K = x.shape
    Eall, I, _ = Wg.shape
    topv, topi = logits.topk(top_k, dim=1)               # [N, top_k]
    gate = torch.softmax(topv, dim=1)                    # [N, top_k] normalized gate weights
    interm_list, gate_list, tok_list = [], [], []
    for e in range(Eall):
        # tokens routed to e, and their gate weight for e
        hit = (topi == e)                                # [N, top_k] bool
        tok_idx, slot = hit.nonzero(as_tuple=True)       # tokens routed to e + which slot
        if tok_idx.numel() == 0:
            continue
        Xe = x[tok_idx]
        interm = F.silu(Xe @ Wg[e].t()) * (Xe @ Wu[e].t())   # [n_e, I]
        interm_list.append(interm)
        gate_list.append(gate[tok_idx, slot])
        tok_list.append(tok_idx)
    INTERM = torch.cat(interm_list, 0)
    GATE = torch.cat(gate_list, 0)
    TOK = torch.cat(tok_list, 0)
    return INTERM, GATE, TOK


def weighted_H(interm, w):
    """Σ w_p · h_p h_pᵀ / Σ w_p  → [I, I]."""
    wsum = w.sum().clamp_min(1e-12)
    return (interm * w.unsqueeze(1)).t() @ interm / wsum


def y_snr_under(W, Q, H_eval):
    diff = (W - Q).float()
    err = torch.einsum("nij,jk,nik->n", diff.unsqueeze(0) if diff.dim()==2 else diff, H_eval,
                       diff.unsqueeze(0) if diff.dim()==2 else diff)
    # operate per-expert: W,Q are [E,N,K]
    return None  # unused; we use batched path below


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--snap", default="e4m3")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"reweighted-Hessian down_proj test  nc{args.n_centroids}/gs{args.group_size}/{args.snap}  "
          f"E={args.n_experts}  layers {sorted(blob.keys())}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        Eall = blob[L]["num_experts"]; N = x.shape[0]
        Wg = load_experts(args.gguf, int(L), "ffn_gate_exps", dev)
        Wu = load_experts(args.gguf, int(L), "ffn_up_exps", dev)
        INTERM, GATE, TOK = reconstruct_pairs(x, logits, Wg, Wu, args.top_k, dev); del Wg, Wu
        Wd = load_experts(args.gguf, int(L), "ffn_down_exps", dev)
        pick = busiest(logits, min(args.n_experts, Eall), args.top_k, Eall, dev)
        W_stack = Wd[pick].contiguous(); del Wd
        if dev.startswith("cuda"): torch.cuda.empty_cache()
        I = INTERM.shape[1]

        # held-out token split (deterministic)
        g = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(N, generator=g).to(dev)
        fit_tok = torch.zeros(N, dtype=torch.bool, device=dev); fit_tok[perm[:N//2]] = True
        is_fit = fit_tok[TOK]
        Pf, Pe = INTERM[is_fit], INTERM[~is_fit]
        Gf, Ge = GATE[is_fit], GATE[~is_fit]

        # outlier gauge on FIT
        norm = Pf.norm(dim=1)
        nmed = norm.median(); cap = torch.quantile(norm, 0.95)

        # candidate FIT Hessians
        ones_f = torch.ones_like(Gf)
        cand = {}
        cand["plain (BASELINE)"]    = weighted_H(Pf, ones_f)
        cand["routing-affinity c"]  = weighted_H(Pf, Gf)
        # winsorize: rescale vectors with norm>cap down to cap
        scale_w = torch.ones_like(norm); over = norm > cap; scale_w[over] = (cap / norm[over])
        cand["winsor p95 (clip)"]   = weighted_H(Pf * scale_w.unsqueeze(1), ones_f)
        # soft down-weight of high-norm tokens
        wdn = 1.0 / (1.0 + (norm / nmed.clamp_min(1e-6))**2)
        cand["downweight 1/(1+r^2)"]= weighted_H(Pf, wdn)
        cand["affinity x downweight"]= weighted_H(Pf, Gf * wdn)

        # eval Hessians on held-out EVAL tokens
        H_eval_plain = weighted_H(Pe, torch.ones_like(Ge))
        H_eval_aff   = weighted_H(Pe, Ge)

        print(f"\n{'='*78}\nL{L} down_proj  E={W_stack.shape[0]} W_e={tuple(W_stack.shape[1:])} I={I}  "
              f"pairs fit={Pf.shape[0]} eval={Pe.shape[0]}  norm med={float(nmed):.2f} p95={float(cap):.2f}")
        print(f"   {'candidate H (fit)':<26}{'YSNR|evalPlain':>15}{'YSNR|evalAffinity':>18}")
        base_p = base_a = None
        for name, Hc in cand.items():
            Hs = Hc.unsqueeze(0).repeat(W_stack.shape[0], 1, 1)
            out = batched_gptq_quantize(W_stack, Hs, n_centroids=args.n_centroids,
                                        group_size=args.group_size, snap_centroids=args.snap, fit_loss="mse")
            Q = out["Q"]
            dq = (W_stack - Q).float()
            yp = (torch.einsum("eij,jk,eik->e", dq, H_eval_plain, dq).clamp_min(1e-30))
            sp = (torch.einsum("eij,jk,eik->e", W_stack, H_eval_plain, W_stack).clamp_min(1e-30))
            ya = (torch.einsum("eij,jk,eik->e", dq, H_eval_aff, dq).clamp_min(1e-30))
            sa = (torch.einsum("eij,jk,eik->e", W_stack, H_eval_aff, W_stack).clamp_min(1e-30))
            ysnr_p = float((10*torch.log10(sp/yp)).median())
            ysnr_a = float((10*torch.log10(sa/ya)).median())
            if base_p is None: base_p, base_a = ysnr_p, ysnr_a
            print(f"   {name:<26}{ysnr_p:>9.2f}{'':>6}{ysnr_a:>9.2f}   (Δp {ysnr_p-base_p:+.2f} Δa {ysnr_a-base_a:+.2f})")
            del Hs, out, Q, dq
            if dev.startswith("cuda"): torch.cuda.empty_cache()
        del INTERM, GATE, TOK, Pf, Pe, W_stack
        if dev.startswith("cuda"): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
