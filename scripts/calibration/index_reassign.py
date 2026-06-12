# index_reassign.py
import torch

def mse_estep(W_orig, centroids, scales, gidx):
    """Per-element argmin_j ||W_orig - centroids[g,j]*scale||^2. Returns uint8 [N,K]."""
    N, K = W_orig.shape
    cand = centroids[gidx].unsqueeze(0) * scales[:, gidx].unsqueeze(-1)   # [N,K,NC]
    err = (cand - W_orig.unsqueeze(-1)) ** 2                              # [N,K,NC]
    return err.argmin(dim=-1).to(torch.uint8)

def pv_vstep(indices, dLdW, centroids, scales, gidx, frac=0.1):
    """PV-tuning-style discrete step. For each element, predicted ΔL(j) =
    dLdW * (centroids[g,j]-centroids[g,cur]) * scale. Flip the top-`frac` elements
    (by predicted improvement) to their argmin-ΔL centroid. Returns (new_idx, n_flips)."""
    N, K = indices.shape
    g = gidx                                                       # [K]
    scale_col = scales[:, g]                                       # [N,K]
    cent_cols = centroids[g]                                       # [K,NC]
    cur = cent_cols.unsqueeze(0).expand(N,-1,-1).gather(
              2, indices.long().unsqueeze(-1)).squeeze(-1)         # [N,K] current centroid val
    # ΔL(j) = dLdW * (cent_j - cur) * scale  → minimize over j
    dW = (cent_cols.unsqueeze(0) - cur.unsqueeze(-1)) * scale_col.unsqueeze(-1)  # [N,K,NC]
    dL = dLdW.unsqueeze(-1) * dW                                   # [N,K,NC]
    best_dL, best_j = dL.min(dim=-1)                              # [N,K]
    improve = (-best_dL).clamp_min(0.0)                           # >0 where a flip helps
    n_candidates = int(improve.numel() * frac)
    if n_candidates == 0 or improve.max() == 0:
        return indices.clone(), 0
    thresh = torch.topk(improve.reshape(-1), n_candidates).values.min()
    do_flip = (improve >= thresh) & (improve > 0)
    new_idx = torch.where(do_flip, best_j.to(torch.uint8), indices)
    return new_idx, int(do_flip.sum())
