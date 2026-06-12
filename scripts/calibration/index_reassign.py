# index_reassign.py
import torch

def mse_estep(W_orig, centroids, scales, gidx):
    """Per-element argmin_j ||W_orig - centroids[g,j]*scale||^2. Returns uint8 [N,K]."""
    N, K = W_orig.shape
    cand = centroids[gidx].unsqueeze(0) * scales[:, gidx].unsqueeze(-1)   # [N,K,NC]
    err = (cand - W_orig.unsqueeze(-1)) ** 2                              # [N,K,NC]
    return err.argmin(dim=-1).to(torch.uint8)

def pv_vstep(indices, dLdW, h, centroids, scales, gidx, frac=0.1, max_step=1):
    """Curvature-corrected, trust-region-bounded PV-tuning discrete step (OBQ-style).

    For each weight element the local loss model is the second-order Taylor
    expansion in the weight change Δw = (centroids[g,j]-centroids[g,cur])*scale:

        ΔL(j) ≈ g·Δw + ½·h·Δw²        g = dLdW[n,k],  h = h_k (diag curvature)

    Because h ≥ 0 this is a convex parabola in Δw, so the continuous optimum is
    the Newton point W − g/h. Two guards make the discrete flip safe:

      1. Quadratic criterion (vs the old linear g·Δw): argmin lands near the
         Newton point, not on a codebook endpoint for low-gradient weights.
      2. `max_step` TRUST REGION: a flip may move at most `max_step` codebook
         value-levels from the current centroid. High-gradient weights whose
         Newton point lies BEYOND the codebook (|g/h| ≫ codebook range) would
         otherwise be slammed to an extreme (catastrophic Δw); the trust region
         caps them to a safe one-level nudge and leaves the large move to Axis A
         (centroid tuning). This is what stops the divergence the count-based
         `frac` trust region could not.

    Flip the top-`frac` elements by predicted improvement. `h` is the per-input
    -column [K] second moment from Ml8Fp8Fn.last_h. Returns (new_idx, n_flips)."""
    N, K = indices.shape
    g = gidx                                                       # [K]
    scale_col = scales[:, g]                                       # [N,K]
    cent_cols = centroids[g]                                       # [K,NC]
    cur = cent_cols.unsqueeze(0).expand(N,-1,-1).gather(
              2, indices.long().unsqueeze(-1)).squeeze(-1)         # [N,K] current centroid val
    dW = (cent_cols.unsqueeze(0) - cur.unsqueeze(-1)) * scale_col.unsqueeze(-1)  # [N,K,NC]
    h_b = h.reshape(1, K, 1)                                       # broadcast curvature over rows/centroids
    dL = dLdW.unsqueeze(-1) * dW + 0.5 * h_b * dW * dW             # [N,K,NC] quadratic in Δw
    # Trust region: forbid candidates more than max_step value-levels from cur.
    rank = centroids.argsort(dim=1).argsort(dim=1)                 # [G,NC] value-rank of each entry
    rank_cols = rank[g]                                            # [K,NC]
    cur_rank = rank_cols.unsqueeze(0).expand(N,-1,-1).gather(
                  2, indices.long().unsqueeze(-1)).squeeze(-1)     # [N,K]
    allowed = (rank_cols.unsqueeze(0) - cur_rank.unsqueeze(-1)).abs() <= max_step  # [N,K,NC]
    dL = torch.where(allowed, dL, torch.full_like(dL, float("inf")))
    best_dL, best_j = dL.min(dim=-1)                              # [N,K]; cur gives ΔL=0 so best_dL ≤ 0
    improve = (-best_dL).clamp_min(0.0)                           # >0 where a flip helps
    n_candidates = int(improve.numel() * frac)
    if n_candidates == 0 or improve.max() == 0:
        return indices.clone(), 0
    thresh = torch.topk(improve.reshape(-1), n_candidates).values.min()
    do_flip = (improve >= thresh) & (improve > 0)
    new_idx = torch.where(do_flip, best_j.to(torch.uint8), indices)
    return new_idx, int(do_flip.sum())

def index_reassign(indices, mode, W_orig, dLdW, h, centroids, scales, gidx, frac=0.1, max_step=1):
    if mode == "none":
        return indices.clone(), 0
    if mode == "mse":
        return mse_estep(W_orig, centroids, scales, gidx), -1
    if mode == "pv":
        return pv_vstep(indices, dLdW, h, centroids, scales, gidx, frac=frac, max_step=max_step)
    raise ValueError(f"unknown reassign mode {mode}")
