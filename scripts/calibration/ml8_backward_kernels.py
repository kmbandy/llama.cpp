"""Fused ml8-QAT backward weight-gradient kernels.

Computes dcent[G,16] and dscales[N,G] from dW_raw[N,K] + indices[N,K], for the
fp8 QAT trainer's Ml8Fp8Fn.backward. Training-only (never ships in a GGUF).

ml8 groups are CONTIGUOUS K-blocks (gidx == arange(K)//gsz), which is what makes
dscales a plain reshape-sum instead of a scatter. See
docs/superpowers/specs/2026-06-13-ml8-qat-fused-wgrad-kernel-design.md.
"""
import os
import warnings

import torch
import triton
import triton.language as tl


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


@triton.jit
def _ml8_wgrad_kernel(
    dW_ptr, idx_ptr, cent_ptr, scales_ptr,      # inputs
    dcent_ptr, dscales_ptr,                      # outputs
    N, K,
    stride_dw_n, stride_dw_k,
    stride_idx_n, stride_idx_k,
    stride_cent_g, stride_cent_c,
    stride_s_n, stride_s_g,
    stride_dc_g, stride_dc_c,
    stride_ds_n, stride_ds_g,
    GSZ: tl.constexpr,
    N_CENT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Grid (G, cdiv(N, BLOCK_N)). Program (g, nt) owns rows [nt*BLOCK_N:...]
    and the contiguous K-slab [g*GSZ:(g+1)*GSZ]. Emits dscales[rows,g] (no
    atomics) and atomic-adds the 16-bin dcent[g,:] histogram."""
    g = tl.program_id(0)
    nt = tl.program_id(1)

    offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = g * GSZ + tl.arange(0, GSZ)                          # cols of this group

    dw = tl.load(
        dW_ptr + offs_n[:, None] * stride_dw_n + offs_k[None, :] * stride_dw_k,
        mask=mask_n[:, None], other=0.0).to(tl.float32)           # [BLOCK_N, GSZ]
    idx = tl.load(
        idx_ptr + offs_n[:, None] * stride_idx_n + offs_k[None, :] * stride_idx_k,
        mask=mask_n[:, None], other=0).to(tl.int32)               # [BLOCK_N, GSZ]
    scal = tl.load(scales_ptr + offs_n * stride_s_n + g * stride_s_g,
                   mask=mask_n, other=0.0).to(tl.float32)         # [BLOCK_N]

    dw_scaled = dw * scal[:, None]                                # [BLOCK_N, GSZ]
    centval = tl.zeros((BLOCK_N, GSZ), dtype=tl.float32)
    for c in tl.static_range(N_CENT):
        cent_c = tl.load(cent_ptr + g * stride_cent_g + c * stride_cent_c)  # scalar
        is_c = idx == c
        centval = tl.where(is_c, cent_c, centval)
        bin_sum = tl.sum(tl.where(is_c, dw_scaled, 0.0))         # scalar over tile
        tl.atomic_add(dcent_ptr + g * stride_dc_g + c * stride_dc_c, bin_sum)

    dscales_row = tl.sum(dw * centval, axis=1)                    # [BLOCK_N]
    tl.store(dscales_ptr + offs_n * stride_ds_n + g * stride_ds_g,
             dscales_row, mask=mask_n)


def ml8_wgrad_triton(dW_raw, indices, centroids, scales, gsz, block_n=64):
    """Fused Triton wgrad: returns (dcent [G,16], dscales [N,G]) fp32."""
    N, K = indices.shape
    G = scales.shape[1]
    assert K == G * gsz, f"K={K} != G*gsz={G*gsz}"
    N_CENT = centroids.shape[1]
    dW_raw = dW_raw.contiguous()
    indices = indices.contiguous()
    centroids = centroids.contiguous()
    scales = scales.contiguous()
    dcent = torch.zeros_like(centroids)                          # atomic_add target
    dscales = torch.empty_like(scales)
    grid = (G, triton.cdiv(N, block_n))
    _ml8_wgrad_kernel[grid](
        dW_raw, indices, centroids, scales,
        dcent, dscales,
        N, K,
        dW_raw.stride(0), dW_raw.stride(1),
        indices.stride(0), indices.stride(1),
        centroids.stride(0), centroids.stride(1),
        scales.stride(0), scales.stride(1),
        dcent.stride(0), dcent.stride(1),
        dscales.stride(0), dscales.stride(1),
        GSZ=gsz, N_CENT=N_CENT, BLOCK_N=block_n,
        num_stages=1,   # gfx1201 RDNA4: num_stages>=2 triggers UAF (forward audit)
        num_warps=4,
    )
    return dcent, dscales


_BACKEND_CACHE = None   # memoized backend choice ("torch" | "triton")


def _probe_backend(dW_raw, indices, centroids, scales, gsz):
    """One-time backend choice (memoized in _BACKEND_CACHE).

    The fused Triton kernel beats the torch scatter on every real ml8 shape
    (39-49x; see bench_ml8_wgrad.py), so prefer it whenever it runs without
    error. We deliberately do NOT time-race the two backends per-shape: the
    trainer calls ml8_wgrad across many shapes per micro, and a timing race
    decided on the first (possibly tiny) layer could lock the whole loop to the
    slow path. Env ML8_WGRAD_BACKEND={torch,triton} overrides; a kernel failure
    falls back to torch but warns (never silently degrades)."""
    forced = os.environ.get("ML8_WGRAD_BACKEND")
    if forced in ("torch", "triton"):
        return forced
    try:
        ml8_wgrad_triton(dW_raw, indices, centroids, scales, gsz)
        return "triton"
    except Exception as e:
        warnings.warn(
            f"ml8_wgrad: Triton kernel failed ({type(e).__name__}: {e}); "
            f"falling back to the slower torch wgrad path.", RuntimeWarning)
        return "torch"


def ml8_wgrad(dW_raw, indices, centroids, scales, gsz):
    """Dispatch to the chosen backend (memoized after first call)."""
    global _BACKEND_CACHE
    if _BACKEND_CACHE is None:
        _BACKEND_CACHE = _probe_backend(dW_raw, indices, centroids, scales, gsz)
    if _BACKEND_CACHE == "triton":
        return ml8_wgrad_triton(dW_raw, indices, centroids, scales, gsz)
    return ml8_wgrad_torch(dW_raw, indices, centroids, scales, gsz)
