#!/usr/bin/env python3
"""Analyze captured router I/O to answer the MoE shared-vs-per-expert Hessian question.

Input: a .pt from calibrate_ml8_paged.py --capture-router, mapping
  {layer_idx: {"x": [N_tok, d_model] fp16, "logits": [N_tok, n_experts] fp16, "num_experts": E}}

`x` is the MoE hidden state = the input the gate/up experts consume. `logits` is the
router output; top-k of it gives each token's expert assignment.

We answer two questions the current pipeline's shared-per-layer Hessian raises:

  Q1 (mismatch): how different is each expert's OWN input covariance H_e = E[x xᵀ | routed to e]
       from the layer-shared H = E[x xᵀ] (over all tokens)? GPTQ currently quantizes every
       expert against the shared H — if per-expert covariances diverge, that's the wrong
       objective per expert. We report:
         - trace-normalized Frobenius divergence ||Ĥ_e - Ĥ_shared|| / ||Ĥ_shared||  (shape, scale-free)
         - cosine similarity of diag(H_e) vs diag(H_shared) — GPTQ's per-column importance weights

  Q2 (starvation): if we *fixed* the mismatch by going per-expert, is H_e even well-estimated
       at its token count? A [K,K] covariance needs n_tok >> K to be full-rank. We report
       per-expert n_tok vs K and the effective rank of H_e.

Usage: python3 analyze_router_capture.py /path/to/capture.pt [--top-k 8] [--device cuda:0]
"""
from __future__ import annotations
import argparse
import torch


def eff_rank(eigs: torch.Tensor) -> float:
    """Participation ratio (Σλ)²/Σλ² — a soft 'how many dimensions carry energy'."""
    eigs = eigs.clamp_min(0)
    s = eigs.sum()
    if s <= 0:
        return 0.0
    return float((s * s) / (eigs.pow(2).sum()))


def pctl(t: torch.Tensor, q: float) -> float:
    return float(t.float().quantile(q))


def analyze_layer(L: int, blob: dict, top_k: int, device: str):
    x = blob["x"].to(device=device, dtype=torch.float32)        # [N, K]
    logits = blob["logits"].to(device=device, dtype=torch.float32)  # [N, E]
    E = int(blob["num_experts"])
    N, K = x.shape
    print(f"\n{'='*72}\nLAYER {L}:  N_tok={N}  d_model(K)={K}  n_experts={E}  top_k={top_k}")

    # Router top-k → per-expert token index lists.
    sel = logits.topk(top_k, dim=1).indices            # [N, top_k]
    onehot = torch.zeros(N, E, device=device, dtype=torch.bool)
    onehot.scatter_(1, sel, True)                      # [N, E] membership
    n_tok = onehot.sum(0)                              # [E] tokens per expert

    print(f"\n[Q2 token counts] per-expert n_tok vs K={K}:")
    print(f"   min={int(n_tok.min())}  p10={int(pctl(n_tok,0.1))}  median={int(n_tok.median())}"
          f"  p90={int(pctl(n_tok,0.9))}  max={int(n_tok.max())}")
    for thr_name, thr in (("0 (cold)", 1), ("< K/4", K // 4), ("< K/2", K // 2), ("< K", K)):
        frac = float((n_tok < thr).float().mean()) * 100 if thr > 1 else float((n_tok == 0).float().mean()) * 100
        print(f"   experts with n_tok {thr_name:>9}: {frac:5.1f}%")

    # Shared (layer) Hessian.
    H_shared = (x.t() @ x) / N                         # [K, K]
    tr_s = torch.diagonal(H_shared).sum()
    Hs_n = H_shared / tr_s                             # trace-normalized
    Hs_n_fro = Hs_n.norm()
    diag_s = torch.diagonal(H_shared)

    # Per-expert: covariance, divergence, diag-cosine, effective rank.
    diverg = torch.zeros(E, device=device)
    diagcos = torch.zeros(E, device=device)
    erank = torch.zeros(E, device=device)
    for e in range(E):
        idx = onehot[:, e].nonzero(as_tuple=True)[0]
        ne = idx.numel()
        if ne < 2:
            diverg[e] = float("nan"); diagcos[e] = float("nan"); erank[e] = 0.0
            continue
        Xe = x[idx]                                    # [ne, K]
        He = (Xe.t() @ Xe) / ne
        tr_e = torch.diagonal(He).sum()
        He_n = He / tr_e
        diverg[e] = (He_n - Hs_n).norm() / Hs_n_fro    # scale-free shape divergence
        de = torch.diagonal(He)
        diagcos[e] = torch.dot(de, diag_s) / (de.norm() * diag_s.norm() + 1e-12)
        # effective rank from eigenvalues (eigvalsh is symmetric-PSD fast path)
        eigs = torch.linalg.eigvalsh(He)
        erank[e] = eff_rank(eigs)

    valid = ~torch.isnan(diverg)
    dv = diverg[valid]; dc = diagcos[valid]; er = erank[valid]; ntk = n_tok[valid].float()

    print(f"\n[Q1 mismatch] per-expert input-covariance vs the SHARED layer Hessian:")
    print(f"   trace-norm Frobenius divergence ||Ĥ_e - Ĥ_shared||/||Ĥ_shared||:")
    print(f"      median={dv.median():.3f}  p90={pctl(dv,0.9):.3f}  max={dv.max():.3f}")
    print(f"   diag(H_e) vs diag(H_shared) cosine  (GPTQ column-importance match; 1.0=identical):")
    print(f"      median={dc.median():.4f}  p10={pctl(dc,0.1):.4f}  min={dc.min():.4f}")

    print(f"\n[Q2 conditioning] per-expert H_e effective rank vs K={K}:")
    print(f"      median eff_rank={er.median():.0f}  p10={pctl(er,0.1):.0f}  "
          f"(full rank would be {K}; eff_rank≈n_tok ⇒ rank-starved)")
    print(f"      median n_tok={ntk.median():.0f}  → eff_rank/K = {float(er.median())/K:.2f}, "
          f"n_tok/K = {float(ntk.median())/K:.2f}")

    # Verdict heuristics.
    mismatch = float(dv.median())
    diagmis = float(dc.median())
    starved = float(ntk.median()) < K
    print(f"\n[READ L{L}]")
    if mismatch > 0.25 or diagmis < 0.95:
        print(f"   → Q1: per-expert covariances DIVERGE from shared (div={mismatch:.2f}, "
              f"diag-cos={diagmis:.3f}) — shared-H is a real mismatch; per-expert could help.")
    else:
        print(f"   → Q1: per-expert covariances are CLOSE to shared (div={mismatch:.2f}, "
              f"diag-cos={diagmis:.3f}) — shared-H is a fine approximation; gap is elsewhere.")
    if starved:
        print(f"   → Q2: per-expert H is RANK-STARVED (median n_tok {int(ntk.median())} < K {K}) "
              f"— a per-expert switch needs more calibration tokens to be well-posed.")
    else:
        print(f"   → Q2: per-expert H is adequately sampled (median n_tok ≥ K).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture", help="path to --capture-router .pt")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"loaded {args.capture}: layers {sorted(blob.keys())}  device={dev}")
    for L in sorted(blob.keys()):
        analyze_layer(int(L), blob[L], args.top_k, dev)


if __name__ == "__main__":
    main()
