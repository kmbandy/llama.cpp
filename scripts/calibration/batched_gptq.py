"""batched_gptq.py — vectorized GPTQ over the expert axis.

Mirrors the scalar `gptq_quantize_linear` in calibrate_ml8.py, but operates
on stacked weights [E, N, K] and stacked Hessians [E, K, K] so all E experts
of one (layer, kind) tuple are quantized in one pass. The hot path — the
per-column error-propagation update — becomes a single batched outer-product
add instead of E independent ones, which is the win at large E.

Lloyd-Max (per-group centroid fitting) stays sequential over E. It runs
only `n_groups` times per linear (=K/group_size, typically 32), and each
call is a small 1-D k-means that wouldn't gain much from batching.

This module is import-clean: no side effects, no global state. Each call
returns the indices / centroids / scales the .pt blob writer expects.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

# Reuse the existing helpers — same Lloyd-Max + E4M3 snap as the scalar path.
from centroid_quantizer import _lloyd_max_signed, snap_to_e4m3


@torch.no_grad()
def batched_gptq_quantize(
    W_stack: torch.Tensor,        # [E, N, K] fp32 (or bf16 — cast to fp32 inside)
    H_stack: torch.Tensor,        # [E, K, K] fp32
    *,
    n_centroids: int = 16,
    group_size: int = 64,
    n_iter: int = 25,
    fit_loss: str = "mse",
    mag_weight_p: float = 5.0,
    snap_centroids: str = "none",
    percdamp: float = 0.05,
    eps_for_skipped: float = 1e-8,
    min_tokens_per_expert: int = 0,
    n_tokens_per_expert: torch.Tensor | None = None,
    chunk_E: int = 8,
) -> dict:
    """Quantize E experts simultaneously via GPTQ-with-centroids.

    Same algorithm as scalar `gptq_quantize_linear`, but the per-column
    update is a batched outer-product add over E. Lloyd-Max centroid
    fitting per (E, group) is done in a Python loop over E (small N_groups
    × per-call cost; not the bottleneck).

    Inputs:
        W_stack: [E, N, K] weight stack. Each W_stack[e] is one expert's
                 weight matrix.
        H_stack: [E, K, K] Hessian stack. Each H_stack[e] = E[x x^T] for
                 expert e's input activations (only routed tokens).

    Optional safety:
        min_tokens_per_expert + n_tokens_per_expert ([E] int):
            experts below the threshold get their weight passed through
            un-quantized (centroids stored as the row-max-abs anchors,
            indices set to nearest, scale = row_max / max(centroid)).
            Use to avoid Cholesky blowup on dead/cold experts.

    Returns dict:
        indices             [E, N, K]                int8 (values in [0, n_centroids-1])
        centroids_per_group [E, n_groups_k, n_centroids] fp32 (signed, sorted)
        scale_per_group     [E, N, n_groups_k]       fp32 (per-row scale per group)
        mse                 [E]                      fp32
        w_snr_db            [E]                      fp32
        y_snr_db            [E]                      fp32
        rel_err             [E]                      fp32
    """
    if snap_centroids not in ("none", "e4m3"):
        raise ValueError(f"snap_centroids must be 'none' or 'e4m3', got {snap_centroids!r}")
    if fit_loss not in ("mse", "mag_weighted"):
        raise ValueError(f"fit_loss must be 'mse' or 'mag_weighted', got {fit_loss!r}")

    dev = H_stack.device
    if W_stack.device != dev:
        W_stack = W_stack.to(dev)
    W = W_stack.float().clone()       # [E, N, K]
    E, N, K = W.shape
    assert H_stack.shape == (E, K, K), f"H_stack shape {tuple(H_stack.shape)} != ({E}, {K}, {K})"
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    n_groups = K // group_size

    if n_tokens_per_expert is None:
        n_tokens_per_expert = torch.full((E,), 1, device=dev, dtype=torch.long)
    cold_expert_mask = n_tokens_per_expert < max(min_tokens_per_expert, 1)
    n_cold = int(cold_expert_mask.sum().item())

    # ── Damp + Cholesky inverse, chunked over E.
    # Chunking is required: the batched hipBLAS triangular solve used by
    # torch.cholesky_inverse hits HIPBLAS_STATUS_ALLOC_FAILED at larger
    # batch sizes (workspace allocator limit). Chunking also keeps peak
    # memory bounded for E=128 / K=2048 on 35B-A3B-scale runs.
    #
    # Math: H = L L^T (Cholesky). H^-1 = L^-T L^-1. We need Hinv_chol such
    # that Hinv_chol^T Hinv_chol = H^-1; that's exactly L^-T (upper-tri).
    # Compute Hinv_chol = L^-T directly via one upper-triangular solve —
    # avoids forming H_inv explicitly + a second Cholesky.
    diag_means = H_stack.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # [E, 1]
    damp = (percdamp * diag_means).view(E, 1, 1)
    eye_K = torch.eye(K, device=dev, dtype=H_stack.dtype)

    H_orig = H_stack       # alias for SNR (undamped)

    # Use the SAME numerical path as scalar `gptq_quantize_linear`:
    #   L = cholesky(H), H_inv = cholesky_inverse(L), Hinv_chol = cholesky(H_inv, upper=True).
    # Although Hinv_chol == L^-T algebraically, the two compute paths produce
    # slightly different fp32 values via PyTorch's lapack/hipBLAS calls. To stay
    # bit-equivalent with the scalar reference (so the verification harness
    # passes), we follow the scalar path exactly. Chunked over E to bound the
    # workspace that batched hipBLAS triangular-solve uses.
    Hinv_chol = torch.empty((E, K, K), device=dev, dtype=torch.float32)
    for chunk_start in range(0, E, chunk_E):
        chunk_end = min(chunk_start + chunk_E, E)
        H_chunk = H_stack[chunk_start:chunk_end] + damp[chunk_start:chunk_end] * eye_K.unsqueeze(0)
        try:
            L_chunk = torch.linalg.cholesky(H_chunk)
            H_inv_chunk = torch.cholesky_inverse(L_chunk)
            Hinv_chol[chunk_start:chunk_end] = torch.linalg.cholesky(H_inv_chunk, upper=True)
        except RuntimeError:
            # Cholesky failure → bump damping for this chunk and retry once.
            H_chunk = H_stack[chunk_start:chunk_end] + (2.0 * damp[chunk_start:chunk_end]) * eye_K.unsqueeze(0)
            L_chunk = torch.linalg.cholesky(H_chunk)
            H_inv_chunk = torch.cholesky_inverse(L_chunk)
            Hinv_chol[chunk_start:chunk_end] = torch.linalg.cholesky(H_inv_chunk, upper=True)
        del L_chunk, H_inv_chunk, H_chunk

    Q = torch.zeros_like(W)                                            # [E, N, K]
    indices = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
    centroids_all = torch.zeros((E, n_groups, n_centroids), device=dev, dtype=torch.float32)
    scales_all = torch.zeros((E, N, n_groups), device=dev, dtype=torch.float32)

    for col in range(K):
        if col % group_size == 0:
            g_idx = col // group_size
            g_end = min(col + group_size, K)
            gs = g_end - col

            # Per-expert per-group: row-max-abs scale + Lloyd-Max centroids.
            group_slice = W[:, :, col:g_end]                            # [E, N, gs]
            scale = group_slice.abs().amax(dim=2, keepdim=True).clamp_min_(eps_for_skipped)  # [E, N, 1]
            x_norm = group_slice / scale                                # [E, N, gs]
            scales_all[:, :, g_idx:g_idx + 1] = scale

            # Lloyd-Max per expert (sequential — not the bottleneck).
            # Each expert gets its own Hessian-diagonal weighting over the
            # group's columns, identical to scalar `find_params` when
            # `quantizer.hessian_diag` is set. Drop this and the centroids
            # land at slightly different optima → measurable PPL drift.
            col_hw = H_orig.diagonal(dim1=-2, dim2=-1)[:, col:g_end].float()  # [E, gs]
            col_idx_template = torch.arange(gs, device=dev).repeat(N)         # [N*gs]
            for e in range(E):
                samples = x_norm[e].flatten()                           # [N*gs]
                centroids_e = _lloyd_max_signed(
                    samples,
                    sample_col_idx=col_idx_template,
                    col_weights=col_hw[e],
                    n_levels=n_centroids,
                    n_iter=n_iter,
                    fit_loss=fit_loss,
                    mag_weight_p=mag_weight_p,
                )                                                       # [n_centroids]
                centroids_e = centroids_e.to(dev)
                if snap_centroids == "e4m3":
                    centroids_e = snap_to_e4m3(centroids_e)
                centroids_all[e, g_idx] = centroids_e

        # ── Per-column quantize + GPTQ propagate (batched over E).
        g_idx = col // group_size
        w_col = W[:, :, col]                                            # [E, N]
        sc = scales_all[:, :, g_idx]                                    # [E, N]
        centroids_g = centroids_all[:, g_idx]                           # [E, n_centroids]

        x_norm_col = w_col / sc                                         # [E, N]
        # Distance to each centroid: [E, N, n_centroids]
        dist = (x_norm_col.unsqueeze(-1) - centroids_g.unsqueeze(1)).abs()
        idx = dist.argmin(dim=-1)                                       # [E, N]
        # Dequantize: gather centroid at idx, multiply by scale.
        q = centroids_g.gather(1, idx) * sc                             # [E, N]

        Q[:, :, col] = q
        indices[:, :, col] = idx.to(torch.int8)

        # GPTQ error propagation.
        diag_col = Hinv_chol[:, col, col].clamp_min(1e-30)              # [E]
        err = (w_col - q) / diag_col.unsqueeze(1)                       # [E, N]
        if col + 1 < K:
            tail = Hinv_chol[:, col:col + 1, col + 1:]                  # [E, 1, K-col-1]
            # Batched outer product subtract: [E, N, 1] @ [E, 1, K-col-1] → [E, N, K-col-1]
            update = err.unsqueeze(2) * tail
            W[:, :, col + 1:].sub_(update)

    # ── Reconstruction metrics (per expert).
    orig = W_stack.float()
    diff = orig - Q                                                     # [E, N, K]
    mse = diff.pow(2).mean(dim=(1, 2))                                  # [E]
    sig_w = orig.pow(2).mean(dim=(1, 2)).clamp_min(1e-30)               # [E]
    w_snr_db = 10.0 * torch.log10(sig_w / mse.clamp_min(1e-30))         # [E]
    # Output-space (use H_orig — undamped):
    # err_y_e = trace(diff_e @ H_orig_e @ diff_e^T)
    # sig_y_e = trace(orig_e @ H_orig_e @ orig_e^T)
    # Compute via einsum to avoid forming the [E, N, N] product.
    err_y = torch.einsum("eij,ejk,eik->e", diff, H_orig, diff).clamp_min(1e-30)
    sig_y = torch.einsum("eij,ejk,eik->e", orig, H_orig, orig).clamp_min(1e-30)
    y_snr_db = 10.0 * torch.log10(sig_y / err_y)
    rel_err = (mse / sig_w).sqrt()

    return {
        "indices": indices,                  # [E, N, K] int8
        "centroids_per_group": centroids_all, # [E, n_groups, n_centroids] fp32
        "scale_per_group": scales_all,        # [E, N, n_groups] fp32
        "Q": Q,                              # [E, N, K] fp32 (dequantized, for weight_override)
        "mse": mse,                          # [E]
        "w_snr_db": w_snr_db,                # [E]
        "y_snr_db": y_snr_db,                # [E]
        "rel_err": rel_err,                  # [E]
        "n_cold_experts": n_cold,
    }
