# index_reassign.py
import torch

def mse_estep(W_orig, centroids, scales, gidx):
    """Per-element argmin_j ||W_orig - centroids[g,j]*scale||^2. Returns uint8 [N,K]."""
    N, K = W_orig.shape
    cand = centroids[gidx].unsqueeze(0) * scales[:, gidx].unsqueeze(-1)   # [N,K,NC]
    err = (cand - W_orig.unsqueeze(-1)) ** 2                              # [N,K,NC]
    return err.argmin(dim=-1).to(torch.uint8)
