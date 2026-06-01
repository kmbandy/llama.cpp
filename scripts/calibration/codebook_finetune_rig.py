#!/usr/bin/env python3
"""THE MISSING PREMISE test (MAD-256): one-shot PTQ vs PTQ + gradient codebook
fine-tuning. Every refuted lever (AWQ, mag_weighted, reweighted-H, rotation) was a
different *one-shot* closed-form solve. The near-lossless 4-bit frontier (QuIP#,
AQLM, OmniQuant) instead uses one-shot as INIT, then gradient-descends the
CONTINUOUS quantization params (centroid values + per-group scales) against the
bf16 teacher — indices FROZEN, bit-width unchanged (true 4-bit).

We learn ~17 floats per group (16 centroids + 1 scale); the 4-bit indices and the
argmin assignment stay frozen (argmin is non-differentiable → frozen). Dequant
w=centroid[idx]·scale is differentiable in centroids+scales, so a block-output loss
tr((W-Wq)·H·(W-Wq)ᵀ) [= output MSE over calib tokens] backprops into them.

RIGOR: held-out token split. Fine-tune on H_fit; report Y_SNR on H_eval (true,
non-circular output error). Report BOTH (a) tuned-fp32 centroids (ceiling) and
(b) tuned-then-e4m3-snapped (realistic ml8, FP8-WMMA-compatible). Sanity: step-0
== GPTQ one-shot baseline. NOTE (KG): per-matrix Y_SNR underpredicts model ΔPPL —
a modest lift here likely compounds across layers; flat is not fully damning.

down_proj only (the floor). Usage:
  python3 codebook_finetune_rig.py --capture /home/kmbandy/models/route-capture-L0-L20.pt \
      --gguf /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf --n-experts 64 --steps 300 --device cuda:0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from batched_gptq import batched_gptq_quantize  # noqa: E402
from centroid_quantizer import snap_to_e4m3  # noqa: E402
from lever_sweep import load_experts, busiest  # noqa: E402


def reconstruct_down(x, logits, Wg, Wu, top_k, device):
    """Per-(token,expert) SwiGLU intermediate + token id."""
    N, K = x.shape; Eall, I, _ = Wg.shape
    topi = logits.topk(top_k, dim=1).indices
    oh = torch.zeros(N, Eall, device=device, dtype=torch.bool); oh.scatter_(1, topi, True)
    interm_l, tok_l = [], []
    for e in range(Eall):
        idx = oh[:, e].nonzero(as_tuple=True)[0]
        if idx.numel() == 0: continue
        Xe = x[idx]
        interm_l.append(F.silu(Xe @ Wg[e].t()) * (Xe @ Wu[e].t()))
        tok_l.append(idx)
    return torch.cat(interm_l, 0), torch.cat(tok_l, 0)


def dequant(indices, centroids, scales, gidx, snap_ste=False):
    """Differentiable in centroids+scales. indices[E,N,K] long (frozen),
    centroids[E,G,nc], scales[E,N,G], gidx[K]→group. Returns Wq[E,N,K].
    snap_ste=True: forward uses e4m3-snapped centroids, backward uses identity
    gradient (straight-through) → optimizes the LATTICE-CONSTRAINED centroids
    directly, so no post-hoc snap loss."""
    if snap_ste:
        centroids = centroids + (snap_to_e4m3(centroids) - centroids).detach()
    cent_per_col = centroids[:, gidx, :]                       # [E,K,nc] (view-ish)
    E, N, K = indices.shape
    gathered = cent_per_col.unsqueeze(1).expand(E, N, K, -1).gather(
        3, indices.long().unsqueeze(-1)).squeeze(-1)           # [E,N,K]
    return gathered * scales[:, :, gidx]                       # × per-col scale


def y_snr(W, Wq, H_eval):
    diff = (W - Wq).float()
    err = torch.einsum("eij,jk,eik->e", diff, H_eval, diff).clamp_min(1e-30)
    sig = torch.einsum("eij,jk,eik->e", W, H_eval, W).clamp_min(1e-30)
    return 10*torch.log10(sig/err)


@torch.no_grad()
def gptq_assign_fixed(W, H, centroids, scales, gidx, group_size, act_order=False, percdamp=0.05):
    """Re-ASSIGN 4-bit indices via GPTQ error-propagation against a FIXED codebook
    (the heavy half: AQLM/PV-tuning-style code update). Unlike batched_gptq this does
    NOT fit Lloyd-Max — it consumes the given (gradient-tuned) centroids+scales and
    re-picks each weight's index Hessian-awarely. With act_order, columns are swept in
    descending diag(H) importance (permuted Cholesky); each column keeps its ORIGINAL
    group's centroids/scales. Returns (indices[E,N,K] long, Q[E,N,K]).
    """
    E, N, K = W.shape
    Hb = H.unsqueeze(0).expand(E, K, K) if H.dim() == 2 else H
    if act_order:
        diag = (H.diagonal() if H.dim() == 2 else H.diagonal(dim1=-2, dim2=-1).mean(0))
        perm = torch.argsort(diag, descending=True)
    else:
        perm = torch.arange(K, device=W.device)
    Wp = W[:, :, perm].clone().float()                 # [E,N,K] permuted cols
    gidx_p = gidx[perm]                                 # original group of each permuted col
    Hp = Hb[:, perm][:, :, perm].float()
    # damped Cholesky inverse (upper) — same path as batched_gptq
    dm = Hp.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).view(E, 1, 1)
    eye = torch.eye(K, device=W.device, dtype=Hp.dtype)
    Hinv = torch.empty_like(Hp)
    for s in range(0, E, 8):
        e = min(s + 8, E)
        Hd = Hp[s:e] + percdamp * dm[s:e] * eye
        L = torch.linalg.cholesky(Hd)
        Hi = torch.cholesky_inverse(L)
        Hinv[s:e] = torch.linalg.cholesky(Hi, upper=True)
        del L, Hi, Hd
    idx_p = torch.zeros((E, N, K), dtype=torch.long, device=W.device)
    Qp = torch.zeros_like(Wp)
    for c in range(K):
        g = int(gidx_p[c])
        sc = scales[:, :, g]                            # [E,N]
        cg = centroids[:, g, :]                         # [E,nc]
        xn = Wp[:, :, c] / sc
        d = (xn.unsqueeze(-1) - cg.unsqueeze(1)).abs()  # [E,N,nc]
        i = d.argmin(-1)                                # [E,N]
        q = cg.gather(1, i) * sc
        idx_p[:, :, c] = i
        Qp[:, :, c] = q
        dcol = Hinv[:, c, c].clamp_min(1e-30)
        err = (Wp[:, :, c] - q) / dcol.unsqueeze(1)
        if c + 1 < K:
            Wp[:, :, c + 1:].sub_(err.unsqueeze(2) * Hinv[:, c, c + 1:].unsqueeze(1))
    # un-permute back to original column order
    idx = torch.zeros_like(idx_p); Q = torch.zeros_like(Qp)
    idx.index_copy_(2, perm, idx_p); Q.index_copy_(2, perm, Qp)
    return idx, Q


SUFFIX = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def finetune_kind(W, Hf, He, args, dev, tag):
    """GPTQ one-shot init → freeze indices → gradient-tune centroids+scales.
    Reports baseline (one-shot e4m3), fp32 ceiling, and straight-through-e4m3
    (the realistic, deployable number). Returns the STE Δ."""
    E, Nr, K = W.shape
    gidx = (torch.arange(K, device=dev) // args.group_size)
    out = batched_gptq_quantize(W, Hf.unsqueeze(0).repeat(E, 1, 1), n_centroids=args.n_centroids,
                                group_size=args.group_size, snap_centroids="e4m3", fit_loss="mse")
    idxs = out["indices"].to(torch.long)
    base = float(y_snr(W, out["Q"], He).median())
    print(f"   {tag:<10} GPTQ one-shot e4m3   Y_SNR|eval = {base:.2f}  (baseline)")

    def run(snap_ste, label):
        cent = out["centroids_per_group"].clone().requires_grad_(True)
        scl = out["scale_per_group"].clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [cent], "lr": args.lr_cent},
                                {"params": [scl], "lr": args.lr_scale}])
        for step in range(args.steps):
            opt.zero_grad()
            diff = W - dequant(idxs, cent, scl, gidx, snap_ste=snap_ste)
            loss = torch.einsum("eij,jk,eik->e", diff, Hf, diff).sum()
            loss.backward(); opt.step()
        with torch.no_grad():
            # final reported number always uses real e4m3-snapped centroids
            ce = snap_to_e4m3(cent)
            y = float(y_snr(W, dequant(idxs, ce, scl, gidx), He).median())
        del cent, scl, opt
        if dev.startswith("cuda"): torch.cuda.empty_cache()
        print(f"   {tag:<10} {label:<20} Y_SNR|eval = {y:.2f}  (Δ {y-base:+.2f})")
        return y - base

    d_post = run(False, "tune→post-snap")     # tune fp32, snap after (the v1 result)
    d_ste = run(True, "tune w/ STE-e4m3")      # snap-in-forward (recover snap loss)
    del idxs, out, W, Hf, He
    if dev.startswith("cuda"): torch.cuda.empty_cache()
    return d_ste


def finetune_kind_heavy(W, Hf, He, args, dev, tag):
    """HEAVY = alternating tune ↔ re-assign (AQLM/PV-tuning). GPTQ init, then each
    round: (1) gradient-tune centroids+scales (fp32) on the current indices, (2) snap
    centroids to e4m3 and RE-ASSIGN indices via Hessian-aware GPTQ error-prop against
    that fixed codebook (act_order optional). All assignment/tuning on FIT Hessian;
    Y_SNR reported on held-out He (deployable e4m3 dequant). Sanity: round-0 ==
    light tune→post-snap."""
    E, Nr, K = W.shape
    gidx = (torch.arange(K, device=dev) // args.group_size)
    out = batched_gptq_quantize(W, Hf.unsqueeze(0).repeat(E, 1, 1), n_centroids=args.n_centroids,
                                group_size=args.group_size, snap_centroids="e4m3", fit_loss="mse")
    idxs = out["indices"].to(torch.long)
    cent = out["centroids_per_group"].clone()
    scl = out["scale_per_group"].clone()
    base = float(y_snr(W, out["Q"], He).median())
    print(f"   {tag:<10} GPTQ one-shot e4m3      Y_SNR|eval = {base:.2f}  (baseline)  act_order={args.act_order}")
    for r in range(args.rounds):
        # (1) gradient-tune cent+scl (fp32) on current indices
        cent = cent.detach().requires_grad_(True)
        scl = scl.detach().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [cent], "lr": args.lr_cent},
                                {"params": [scl], "lr": args.lr_scale}])
        for _ in range(args.steps):
            opt.zero_grad()
            diff = W - dequant(idxs, cent, scl, gidx)
            (torch.einsum("eij,jk,eik->e", diff, Hf, diff).sum()).backward()
            opt.step()
        cent = cent.detach(); scl = scl.detach()
        with torch.no_grad():
            y_tune = float(y_snr(W, dequant(idxs, snap_to_e4m3(cent), scl, gidx), He).median())
        # (2) re-assign indices against the snapped (deployable) codebook
        cent_snap = snap_to_e4m3(cent)
        idxs, Q = gptq_assign_fixed(W, Hf, cent_snap, scl, gidx, args.group_size,
                                    act_order=args.act_order, percdamp=args.percdamp)
        y_reassign = float(y_snr(W, Q, He).median())
        print(f"   {tag:<10} round {r}: tune={y_tune:.2f} (Δ{y_tune-base:+.2f})  "
              f"reassign={y_reassign:.2f} (Δ{y_reassign-base:+.2f})")
        del opt
        if dev.startswith("cuda"): torch.cuda.empty_cache()
    del idxs, out, W, Hf, He, cent, scl
    if dev.startswith("cuda"): torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--kinds", default="gate,up,down")
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr-cent", type=float, default=1e-2)
    p.add_argument("--lr-scale", type=float, default=1e-3)
    p.add_argument("--heavy", action="store_true", help="alternating tune<->reassign (AQLM-style)")
    p.add_argument("--rounds", type=int, default=4, help="heavy: alternating rounds")
    p.add_argument("--act-order", action="store_true", help="heavy: Hessian-desc column order in reassign")
    p.add_argument("--percdamp", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    kinds = [k.strip() for k in args.kinds.split(",")]
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"codebook fine-tune test  nc{args.n_centroids}/gs{args.group_size}/e4m3  "
          f"E={args.n_experts} steps={args.steps} kinds={kinds} layers {sorted(blob.keys())}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        Eall = blob[L]["num_experts"]; N = x.shape[0]
        pick = busiest(logits, min(args.n_experts, Eall), args.top_k, Eall, dev)
        g = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(N, generator=g).to(dev)
        fit_tok = torch.zeros(N, dtype=torch.bool, device=dev); fit_tok[perm[:N//2]] = True
        print(f"\n{'='*74}\nL{L}  E_sample={len(pick)}  fit_tok={int(fit_tok.sum())} eval_tok={int((~fit_tok).sum())}")

        # down needs the reconstructed SwiGLU intermediate (held-out split by token)
        Hf_down = He_down = None
        if "down" in kinds:
            Wg = load_experts(args.gguf, int(L), "ffn_gate_exps", dev)
            Wu = load_experts(args.gguf, int(L), "ffn_up_exps", dev)
            INTERM, TOK = reconstruct_down(x, logits, Wg, Wu, args.top_k, dev); del Wg, Wu
            isf = fit_tok[TOK]
            Hf_down = INTERM[isf].t() @ INTERM[isf] / isf.sum().clamp_min(1)
            He_down = INTERM[~isf].t() @ INTERM[~isf] / (~isf).sum().clamp_min(1)
            del INTERM, TOK
            if dev.startswith("cuda"): torch.cuda.empty_cache()
        # gate/up consume the router-input hidden state x (held-out split by token)
        xf, xe = x[fit_tok], x[~fit_tok]
        Hf_x = xf.t() @ xf / xf.shape[0]
        He_x = xe.t() @ xe / xe.shape[0]

        fn = finetune_kind_heavy if args.heavy else finetune_kind
        for kind in kinds:
            W = load_experts(args.gguf, int(L), SUFFIX[kind], dev)[pick].contiguous()
            if dev.startswith("cuda"): torch.cuda.empty_cache()
            if kind == "down":
                fn(W, Hf_down, He_down, args, dev, kind)
            else:
                fn(W, Hf_x, He_x, args, dev, kind)
        del Hf_x, He_x, Hf_down, He_down
        if dev.startswith("cuda"): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
