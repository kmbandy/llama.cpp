#!/usr/bin/env python3
"""Offline quantize-both-ways: does a SHRINKAGE (or per-expert) Hessian beat the
SHARED layer Hessian for GPTQ-quantizing MoE experts?

Inputs:
  - capture .pt from calibrate_ml8_paged.py --capture-router  (router x + logits)
  - the source bf16 GGUF (to read the expert weights)

For each sampled expert e of a (layer, gate_proj):
  - split its routed tokens 50/50 → FIT / EVAL  (held-out eval: per-expert H can't
    win by overfitting its own Hessian)
  - build candidate Hessians from FIT tokens:
        shared    = pooled XᵀX over ALL experts' FIT tokens  (the current pipeline)
        per-expert= X_eᵀX_e over expert e's FIT tokens        (rank-deficient)
        shrink(α) = α·H_e + (1-α)·H_shared
  - GPTQ-quantize W_e (4-bit) under each candidate Hessian
  - score by Y_SNR on EVAL tokens: H_eval = X_e_evalᵀX_e_eval (the TRUE output error)
  - report median Y_SNR per strategy.

Sanity gates (catch a buggy GPTQ before we trust any conclusion):
  1. GPTQ(H_shared) Y_SNR  >  round-to-nearest Y_SNR   (GPTQ must help)
  2. GPTQ(H=I) Y_SNR  ≈  round-to-nearest Y_SNR         (identity H = no error-prop)

Usage:
  python3 quantize_both_ways.py --capture route-capture-L0-L20.pt \
      --gguf /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf \
      --kind gate_proj --gguf-suffix ffn_gate_exps \
      --n-experts 48 --top-k 8 --device cuda:0
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


# ── 4-bit per-output-row symmetric quantizer (shared by all strategies, so the
# comparison isolates the Hessian choice, not the snap scheme). ──────────────
def quant_rows_4bit(W: torch.Tensor) -> torch.Tensor:
    """Symmetric 4-bit (15 levels, [-7,7]) per output row. W: [out, in]."""
    qmax = 7.0
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(W / scale).clamp_(-qmax, qmax) * scale


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, percdamp: float = 0.05) -> torch.Tensor:
    """Standard GPTQ (Frantar et al.) column sweep with per-row 4-bit snap.

    W: [out, in] (we quantize along `in` columns, propagating error rightward).
    H: [in, in] symmetric PSD. Damping escalates until the factorization is PD —
    rank-deficient (per-expert) Hessians genuinely need more damping, which is the
    honest cost shrinkage avoids.
    """
    W = W.clone().float()
    out_dim, in_dim = W.shape
    H = H.clone().float()
    dead = torch.diagonal(H) == 0
    H[dead, dead] = 1.0
    base_diag = torch.diagonal(H).mean()
    idx = torch.arange(in_dim, device=H.device)
    Hinv = None
    for mult in (1.0, 2.0, 5.0, 20.0, 100.0, 500.0):
        Hd = H.clone()
        Hd[idx, idx] += percdamp * mult * base_diag
        try:
            L = torch.linalg.cholesky(Hd)
            Hi = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hi, upper=True)
            break
        except Exception:
            continue
    if Hinv is None:
        raise RuntimeError("GPTQ: Hessian not conditionable even at 500x damp")

    Q = torch.zeros_like(W)
    qmax = 7.0
    scale = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax  # per-row, fixed
    for i in range(in_dim):
        w = W[:, i]
        d = Hinv[i, i]
        q = torch.round(w / scale.squeeze(1)).clamp(-qmax, qmax) * scale.squeeze(1)
        Q[:, i] = q
        err = (w - q) / d
        if i + 1 < in_dim:
            W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return Q


def y_snr_db(W: torch.Tensor, Q: torch.Tensor, H_eval: torch.Tensor) -> float:
    diff = (W - Q).float()
    err = (diff @ H_eval @ diff.t()).diagonal().sum()
    sig = (W @ H_eval @ W.t()).diagonal().sum()
    if err <= 0 or sig <= 0:
        return float("nan")
    return float(10.0 * torch.log10(sig / err))


def load_expert_weights(gguf_path: str, layer: int, suffix: str, device: str):
    """Return [n_exp, out, in] fp32 for blk.{layer}.{suffix}.weight."""
    r = gguf.GGUFReader(gguf_path)
    name = f"blk.{layer}.{suffix}.weight"
    t = next((t for t in r.tensors if t.name == name), None)
    if t is None:
        raise RuntimeError(f"{name} not found in {gguf_path}")
    arr = np.asarray(t.data, dtype=np.uint8).copy()
    torch_shape = [int(s) for s in reversed(list(t.shape))]  # GGUF ne → torch
    w = torch.from_numpy(arr).view(torch.bfloat16).reshape(*torch_shape).to(torch.float32)
    return w.to(device)  # [n_exp, out, in]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--kind", default="gate_proj")
    p.add_argument("--gguf-suffix", default="ffn_gate_exps")
    p.add_argument("--n-experts", type=int, default=48, help="experts sampled per layer")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--alphas", default="0.0,0.5,0.9,1.0",
                   help="shrinkage α grid (0=shared, 1=per-expert)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    alphas = [float(a) for a in args.alphas.split(",")]

    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"loaded {args.capture}: layers {sorted(blob.keys())}  device={dev}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)        # [N, K]
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        E = int(blob[L]["num_experts"]); N, K = x.shape
        We = load_expert_weights(args.gguf, int(L), args.gguf_suffix, dev)  # [E, out, in]
        assert We.shape[2] == K, f"weight in-dim {We.shape[2]} != x dim {K}"
        print(f"\n{'='*72}\nLAYER {L} {args.kind}:  N={N} K={K} E={E}  W_e={tuple(We.shape[1:])}")

        sel = logits.topk(args.top_k, dim=1).indices
        onehot = torch.zeros(N, E, device=dev, dtype=torch.bool)
        onehot.scatter_(1, sel, True)

        # FIT/EVAL split of token rows (global, deterministic).
        g = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(N, generator=g).to(dev)
        fit_mask = torch.zeros(N, dtype=torch.bool, device=dev); fit_mask[perm[: N // 2]] = True

        Xf = x[fit_mask]
        H_shared = (Xf.t() @ Xf) / Xf.shape[0]

        # round-to-nearest reference + identity-H GPTQ (sanity gates) on a few experts.
        results = {f"shrink{a}": [] for a in alphas}
        results["nearest"] = []; results["gptq_I"] = []
        cand = onehot.sum(0).argsort(descending=True)[: args.n_experts]  # busiest experts
        I = torch.eye(K, device=dev)
        for e in cand.tolist():
            tok = onehot[:, e]
            fit_idx = (tok & fit_mask).nonzero(as_tuple=True)[0]
            eval_idx = (tok & ~fit_mask).nonzero(as_tuple=True)[0]
            if fit_idx.numel() < 8 or eval_idx.numel() < 8:
                continue
            Xe_fit = x[fit_idx]; Xe_eval = x[eval_idx]
            H_eval = (Xe_eval.t() @ Xe_eval) / Xe_eval.shape[0]
            H_e = (Xe_fit.t() @ Xe_fit) / Xe_fit.shape[0]
            W = We[e]
            results["nearest"].append(y_snr_db(W, quant_rows_4bit(W), H_eval))
            results["gptq_I"].append(y_snr_db(W, gptq_quantize(W, I), H_eval))
            for a in alphas:
                Hq = a * H_e + (1.0 - a) * H_shared
                results[f"shrink{a}"].append(y_snr_db(W, gptq_quantize(W, Hq), H_eval))

        def med(k):
            v = torch.tensor([r for r in results[k] if not np.isnan(r)])
            return float(v.median()) if v.numel() else float("nan")

        n = len([r for r in results["nearest"] if not np.isnan(r)])
        print(f"  (median held-out Y_SNR over {n} experts)")
        print(f"   nearest (no GPTQ)      : {med('nearest'):.3f} dB")
        print(f"   GPTQ H=I (sanity)      : {med('gptq_I'):.3f} dB")
        for a in alphas:
            tag = "shared" if a == 0.0 else ("per-expert" if a == 1.0 else f"shrink α={a}")
            print(f"   GPTQ α={a:<4} ({tag:<11}): {med(f'shrink{a}'):.3f} dB")

        # Sanity verdict.
        shared, nearest, gI = med("shrink0.0"), med("nearest"), med("gptq_I")
        print(f"  [sanity] GPTQ(shared) > nearest? {shared > nearest}  "
              f"(Δ={shared-nearest:+.2f})   GPTQ(I)≈nearest? "
              f"{abs(gI-nearest)<0.5} (Δ={gI-nearest:+.2f})")
        best_a = max(alphas, key=lambda a: med(f"shrink{a}"))
        print(f"  [verdict] best α={best_a}  (shared={med('shrink0.0'):.2f} → "
              f"best={med(f'shrink{best_a}'):.2f}, Δ={med(f'shrink{best_a}')-med('shrink0.0'):+.2f} dB)")


if __name__ == "__main__":
    main()
