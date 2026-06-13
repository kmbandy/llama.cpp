"""Fused ml8-QAT backward weight-gradient kernels.

Computes dcent[G,16] and dscales[N,G] from dW_raw[N,K] + indices[N,K], for the
fp8 QAT trainer's Ml8Fp8Fn.backward. Training-only (never ships in a GGUF).

ml8 groups are CONTIGUOUS K-blocks (gidx == arange(K)//gsz), which is what makes
dscales a plain reshape-sum instead of a scatter. See
docs/superpowers/specs/2026-06-13-ml8-qat-fused-wgrad-kernel-design.md.
"""
import torch


def ml8_wgrad_torch(dW_raw, indices, centroids, scales, gsz):
    """Pure-torch reference/fallback. Exact dscales via contiguous reshape;
    dcent via index_put_ (the best pure-torch option per the bench).

    Args:
        dW_raw    [N,K] fp32  -- (dy8*sdy).T @ x
        indices   [N,K] uint8 -- centroid index per (row, col)
        centroids [G,16] fp32 -- master fp32 centroids (cent_e4m3 in caller)
        scales    [N,G] fp32
        gsz       int          -- group size (K must be divisible; G = K//gsz)
    Returns:
        (dcent [G,16], dscales [N,G]) fp32
    """
    N, K = indices.shape
    G = scales.shape[1]
    assert K == G * gsz, f"K={K} != G*gsz={G*gsz}"
    idx = indices.long()
    scales_exp = scales.repeat_interleave(gsz, dim=1)                # [N,K], col->group
    # dscales[n,g] = sum_{k in g} dW_raw[n,k] * centroids[g, idx[n,k]]
    centval = torch.gather(
        centroids.repeat_interleave(gsz, dim=0).unsqueeze(0).expand(N, -1, -1),
        2, idx.unsqueeze(-1)).squeeze(-1)                            # [N,K] = cent[g,idx]
    dscales = (dW_raw * centval).view(N, G, gsz).sum(2)              # contiguous reshape
    # dcent[g,c] = sum_{n, k in g, idx=c} dW_raw[n,k] * scales[n,g]
    dW_scaled = dW_raw * scales_exp                                  # [N,K]
    gidx = (torch.arange(K, device=dW_raw.device) // gsz).long()
    flat_g = gidx.unsqueeze(0).expand(N, -1).reshape(-1)
    flat_i = idx.reshape(-1)
    dcent = torch.zeros_like(centroids)
    dcent.index_put_((flat_g, flat_i), dW_scaled.reshape(-1), accumulate=True)
    return dcent, dscales
