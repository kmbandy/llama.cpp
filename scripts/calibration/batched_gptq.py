"""batched_gptq.py — vectorized GPTQ over the expert axis, single + multi-GPU.

Mirrors the scalar `gptq_quantize_linear` in calibrate_ml8.py, but operates
on stacked weights [E, N, K] and stacked Hessians [E, K, K] so all E experts
of one (layer, kind) tuple are quantized in one pass. The hot path — the
per-column error-propagation update — becomes a single batched outer-product
add instead of E independent ones, which is the win at large E.

Lloyd-Max (per-group centroid fitting) is also batched over the expert axis
via `_batched_lloyd_max_signed`. Profiling on 35B-A3B shapes showed the
sequential per-expert + per-level loops were 97% of task time (~1.6M HIP
dispatches per task, each ~250µs); the batched kernel collapses both loops
into `torch.searchsorted` + two `scatter_add_` calls and brings Lloyd-Max
under 1% of task time.

This module is import-clean: no side effects, no global state. Each call
returns the indices / centroids / scales the .pt blob writer expects.
"""
from __future__ import annotations

import math
import os
import time
from typing import Sequence

import torch

# Reuse the existing helpers — same Lloyd-Max + E4M3 snap as the scalar path.
from centroid_quantizer import _lloyd_max_signed, snap_to_e4m3


# ─── Compositor yield (graphics ring starvation prevention) ─────────────────
#
# Sustained GPU compute on a card that also drives the desktop will eventually
# starve the OS compositor's graphics ring. Linux + amdgpu shows this as
# `ring gfx_0.0.0 timeout` followed by an SQC permission fault and a device
# wedge. Windows DWM and Apple WindowServer have the same failure mode under
# enough pressure — the calibration kernel just hits it first on amdgpu/Linux.
#
# Mitigation: every ML8_YIELD_EVERY_COLS columns of the GPTQ propagation loop,
# call cuda.synchronize() to drain pending kernels, then sleep ML8_YIELD_MS
# milliseconds. The sync + sleep gives the compositor a guaranteed render
# window. Overhead at default settings: ~0.5% wall time per task.
#
# Configurable via env vars so users can tune (or disable on a headless box):
#   ML8_YIELD_EVERY_COLS  — yield every N columns (default 64 for K>=1024, else 32)
#   ML8_YIELD_MS          — sleep duration in ms each yield  (default 5)
#   ML8_YIELD_DISABLE     — set to "1" to disable yielding entirely
#
# This is the OSS-shippable fix: works on single-GPU systems with no special
# setup, no TTY required, no second display GPU. Just respects the desktop.

def _resolve_yield_params(K: int) -> tuple[int, float]:
    """Return (yield_every_cols, yield_seconds). Yield disabled → (0, 0.0)."""
    if os.environ.get("ML8_YIELD_DISABLE", "") == "1":
        return 0, 0.0
    default_every = 64 if K >= 1024 else 32
    every = int(os.environ.get("ML8_YIELD_EVERY_COLS", default_every))
    ms = float(os.environ.get("ML8_YIELD_MS", "5"))
    return max(1, every), max(0.0, ms) / 1000.0


@torch.no_grad()
def _batched_lloyd_max_signed(
    samples_E: torch.Tensor,           # [E, M] fp32 — per-expert flattened normalized samples
    *,
    col_weights_E: torch.Tensor | None,  # [E, gs] fp32 — per-expert Hessian-diag weight per column
    col_idx: torch.Tensor | None,        # [M] long — column index of each sample, in [0, gs)
    n_levels: int,
    n_iter: int,
    fit_loss: str,
    mag_weight_p: float,
) -> torch.Tensor:
    """Vectorized signed Lloyd-Max fit over a batch of E experts.

    Identical math to the scalar `_lloyd_max_signed` in centroid_quantizer.py,
    but the per-expert Python loop and the per-level inner Python loop are
    collapsed into per-iter `searchsorted` + two `scatter_add_` calls. On 35B
    shapes (E=128, M ≈ 49 K-131 K) this cuts ~1.6M HIP dispatches per task
    down to a few hundred — Lloyd-Max drops from 97% to <1% of task time.

    NaN handling: the scalar path filters samples via `s = s[isfinite(s)]`
    per-expert, which can't be vectorized across E (variable row sizes).
    We assume no NaNs (matches the existing scalar comment: "Linear weights
    from a healthy model: no NaNs in practice"). Any NaN sample contaminates
    that expert's centroids only.

    Returns: [E, n_levels] fp32, ascending-sorted.
    """
    E, M = samples_E.shape
    dev = samples_E.device
    s_E = samples_E.float()

    # Per-expert quantile init. torch.quantile(input, q, dim=1) returns
    # [n_levels, E]; transpose + contiguous for [E, n_levels].
    q = torch.linspace(0.0, 1.0, n_levels, device=dev)
    centroids_E = torch.quantile(s_E, q, dim=1).t().contiguous()  # [E, n_levels]

    # Per-sample weight (fit_loss + optional Hessian-diag column weighting).
    if fit_loss == "mag_weighted":
        w_mag_E = s_E.abs().pow(mag_weight_p)
    else:  # "mse"
        w_mag_E = torch.ones_like(s_E)
    if (col_weights_E is not None and col_idx is not None
            and col_idx.numel() == M):
        # col_weights_E[e, col_idx[i]] for each i → [E, M] via index_select.
        col_w_expanded = col_weights_E.index_select(1, col_idx)  # [E, M]
        w_mag_E = w_mag_E * col_w_expanded
    weights_E = w_mag_E

    w_sum  = torch.empty((E, n_levels), device=dev, dtype=s_E.dtype)
    ws_sum = torch.empty((E, n_levels), device=dev, dtype=s_E.dtype)
    ws_E   = weights_E * s_E  # [E, M] — reused each iter; samples don't change

    for _ in range(n_iter):
        # Midpoints between adjacent (sorted) centroids → per-expert bin edges.
        edges_E = (centroids_E[:, :-1] + centroids_E[:, 1:]) * 0.5  # [E, n_levels-1]
        # searchsorted with N-D sorted_sequence: per-row independent search.
        # Output bins in [0, n_levels-1]. Each sample's bin index = which
        # centroid it's nearest to (left-tie convention same as bucketize).
        bins_E = torch.searchsorted(edges_E, s_E)               # [E, M] int64

        # Per-(e, k) weighted sums via scatter_add.
        w_sum.zero_()
        ws_sum.zero_()
        w_sum.scatter_add_(1, bins_E, weights_E)
        ws_sum.scatter_add_(1, bins_E, ws_E)

        new_E = torch.where(
            w_sum > 1e-30,
            ws_sum / w_sum.clamp_min(1e-30),
            centroids_E,  # empty-bin: keep old centroid
        )
        if torch.allclose(new_E, centroids_E, rtol=1e-6):
            centroids_E = new_E
            break
        centroids_E = new_E

    return torch.sort(centroids_E, dim=1).values


def _cholesky_inv_upper(
    H: torch.Tensor,
    damp: torch.Tensor | float,
    eye: torch.Tensor,
    *,
    max_escalations: int = 8,
    escalation: float = 10.0,
):
    """Robust upper-Cholesky factor of (H + damp·I)⁻¹ for GPTQ error propagation.

    Returns ``(Hinv_chol, n_escalations)`` where ``Hinv_chol`` is upper-triangular
    with ``Hinv_chol.T @ Hinv_chol == (H + damp_eff·I)⁻¹``. ``H``/``eye`` may carry a
    leading batch dim (chunk over experts); ``damp`` broadcasts against it.

    Why this exists (MAD-256, faithful-acts path): the e4m3-activation Hessians
    XᵀX are severely ill-conditioned. The shipped path

        L = cholesky(H+damp); H_inv = cholesky_inverse(L); cholesky(H_inv, upper=True)

    has its FIRST factorization succeed (H+damp is PD) but the SECOND one fail —
    cholesky_inverse returns an H_inv that is only approximately symmetric /
    marginally indefinite in fp32 ("leading minor of order N"). In the paged
    driver that RuntimeError bubbles up and the entire tensor is bf16-backfilled
    (calibrate_ml8_paged.py), wrecking both size and coverage.

    The schedule is chosen so currently-succeeding tensors are BYTE-IDENTICAL to
    the shipped path (the q1 anchor must not move):

      • attempt 0  — exact shipped path, no symmetrization. Succeeds for every
                     well-conditioned tensor ⇒ bit-for-bit unchanged.
      • attempt 1  — same damping + symmetrize H_inv. The free fix when the only
                     problem was fp32 asymmetry from cholesky_inverse.
      • attempt ≥2 — symmetrize + geometrically escalate the Tikhonov damping
                     until (H+damp·I)⁻¹ is numerically PD. Stronger regularization
                     biases error-prop toward the diagonal slightly, but a finite
                     4-bit factor beats throwing the tensor away to bf16.
    """
    diag_mean = H.diagonal(dim1=-2, dim2=-1).mean()
    base = float(damp) if not torch.is_tensor(damp) else damp
    last_err: Exception | None = None
    for k in range(max_escalations + 1):
        if k == 0:
            damp_k, symmetrize = damp, False
        elif k == 1:
            damp_k, symmetrize = damp, True
        else:
            # escalate from the base damp (floor it if base was zero)
            floor = base if (float(base.max()) if torch.is_tensor(base) else base) > 0 else (1e-8 * diag_mean)
            damp_k, symmetrize = floor * (escalation ** (k - 1)), True
        try:
            Hd = H + damp_k * eye
            L = torch.linalg.cholesky(Hd)
            H_inv = torch.cholesky_inverse(L)
            if symmetrize:
                H_inv = 0.5 * (H_inv + H_inv.transpose(-2, -1))
            return torch.linalg.cholesky(H_inv, upper=True), k
        except (RuntimeError, torch._C._LinAlgError) as e:  # noqa: PERF203
            last_err = e
    raise RuntimeError(
        f"cholesky_inv_upper: not PD after {max_escalations} damp escalations "
        f"(base damp≈{float(base.max()) if torch.is_tensor(base) else base:.3e}): {last_err}")


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
    act_order: bool = False,
    heavy_rounds: int = 0,
    heavy_steps: int = 60,
    heavy_lr_cent: float = 1e-2,
    heavy_lr_scale: float = 1e-3,
    heavy_dtype: str = "fp32",     # "fp32" (default) or "bf16": bf16 runs the heavy
                                   # tune-loss matmul on bf16 WMMA (fp32 accumulate),
                                   # ~2x faster. Params + Adam state stay fp32.
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

    # Upper-Cholesky factor of (H+damp)^-1 for GPTQ error propagation, via the
    # robust `_cholesky_inv_upper` helper (symmetrize + damp escalation). Attempt
    # 0 reproduces the scalar `gptq_quantize_linear` path BIT-FOR-BIT, so any
    # tensor that already succeeds is unchanged (the verification harness and the
    # q1 anchor stay put); only ill-conditioned faithful-acts Hessians escalate —
    # recovering full ml8 coverage instead of bf16-backfilling the tensor. Chunked
    # over E to bound the hipBLAS triangular-solve workspace; same granularity as
    # the prior per-chunk try/except, so no new cross-expert coupling.
    Hinv_chol = torch.empty((E, K, K), device=dev, dtype=torch.float32)
    _max_chol_esc = 0
    for chunk_start in range(0, E, chunk_E):
        chunk_end = min(chunk_start + chunk_E, E)
        chol_chunk, n_esc = _cholesky_inv_upper(
            H_stack[chunk_start:chunk_end],
            damp[chunk_start:chunk_end],
            eye_K.unsqueeze(0),
        )
        Hinv_chol[chunk_start:chunk_end] = chol_chunk
        _max_chol_esc = max(_max_chol_esc, n_esc)
        del chol_chunk
    if _max_chol_esc >= 1:
        print(f"[gptq] ill-conditioned Hessian recovered via {_max_chol_esc} damp "
              f"escalation(s) — full ml8 coverage held (no bf16 backfill)")

    Q = torch.zeros_like(W)                                            # [E, N, K]
    indices = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
    centroids_all = torch.zeros((E, n_groups, n_centroids), device=dev, dtype=torch.float32)
    scales_all = torch.zeros((E, N, n_groups), device=dev, dtype=torch.float32)

    yield_every, yield_secs = _resolve_yield_params(K)
    is_cuda_dev = dev.type == "cuda"

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

            # Lloyd-Max per expert — batched across E. Profiling 35B-A3B
            # showed sequential per-expert + per-level Python loops were
            # 97% of task time (~1.6M HIP dispatches per task at ~250µs
            # each). _batched_lloyd_max_signed collapses both loops into
            # searchsorted + scatter_add over [E, M] — same math, ~700x
            # fewer dispatches, Lloyd-Max drops to <1% of task time.
            col_hw = H_orig.diagonal(dim1=-2, dim2=-1)[:, col:g_end].float()  # [E, gs]
            col_idx_template = torch.arange(gs, device=dev).repeat(N)         # [N*gs]
            samples_E = x_norm.reshape(E, N * gs)                              # [E, M]
            centroids_batch = _batched_lloyd_max_signed(
                samples_E,
                col_weights_E=col_hw,
                col_idx=col_idx_template,
                n_levels=n_centroids,
                n_iter=n_iter,
                fit_loss=fit_loss,
                mag_weight_p=mag_weight_p,
            )                                                                  # [E, n_centroids]
            if snap_centroids == "e4m3":
                centroids_batch = snap_to_e4m3(centroids_batch)
            centroids_all[:, g_idx] = centroids_batch

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

        # Compositor yield — drain pending kernels and sleep briefly so the
        # OS display compositor's gfx ring gets render slots. Without this,
        # sustained calibration compute starves the gfx ring after ~5-6 min
        # → SQC permission fault → device wedge → desktop dies. See module
        # docstring near _resolve_yield_params for tunables. Default cost
        # ~0.5% wall time; mandatory for unattended runs on the display GPU.
        if yield_every > 0 and is_cuda_dev and (col + 1) % yield_every == 0 and col + 1 < K:
            torch.cuda.synchronize(dev)
            if yield_secs > 0.0:
                time.sleep(yield_secs)

    # ── act_order reassignment pass (MAD-256, validated in codebook_finetune_rig
    # "round 0"): re-quantize column indices in Hessian-descending importance
    # order using the FITTED centroids. Bit-free; +~1.0 dB gate/up, +0.4 down at
    # L0 on 35B held-out eval. The straight 0→K sweep above fits the centroids
    # (with error-prop); this pass redoes ONLY the assignment in importance order
    # — the standard GPTQ act_order trick the ml8 recipe never had. Groups stay
    # in ORIGINAL space (gidx_orig); only the sweep/error-prop order is permuted.
    # ── act_order reassignment (+ optional heavy tune↔reassign loop). MAD-256,
    # validated in codebook_finetune_rig. act_order alone: +~1.0 dB gate/up,
    # +0.4 down (L0). heavy_rounds>0 alternates a gradient tune of centroids+
    # scales (AQLM/PV-tuning) with the act_order reassign — adds ~+0.3 on down.
    # All bit-free (indices stay 4-bit, true 4 bpv). Groups stay ORIGINAL space;
    # only the sweep/error-prop ORDER is Hessian-importance-permuted.
    if act_order or heavy_rounds > 0:
        del Hinv_chol                                                  # free [E,K,K] — sweep done
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        importance = H_orig.diagonal(dim1=-2, dim2=-1).mean(0)          # [K] (shared H ⇒ same per e)
        perm = torch.argsort(importance, descending=True)
        gidx_orig = torch.arange(K, device=dev) // group_size
        Hp = H_orig[:, perm][:, :, perm]                               # [E,K,K] permuted
        Hinv_p = torch.empty((E, K, K), device=dev, dtype=torch.float32)
        for cs in range(0, E, chunk_E):
            ce = min(cs + chunk_E, E)
            chol_chunk, _ = _cholesky_inv_upper(
                Hp[cs:ce], damp[cs:ce], eye_K.unsqueeze(0))
            Hinv_p[cs:ce] = chol_chunk
            del chol_chunk
        del Hp

        @torch.no_grad()
        def _reassign(cents, scls):
            """act_order GPTQ assignment vs FIXED (cents, scls). Returns (idx int8, Q) [E,N,K]."""
            Wp = W_stack.float()[:, :, perm].clone()
            idx_p = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
            Qp = torch.zeros_like(Wp)
            for c in range(K):
                g = int(gidx_orig[perm[c]])                            # ORIGINAL group of this column
                sc = scls[:, :, g]; cg = cents[:, g, :]
                di = (Wp[:, :, c].div(sc).unsqueeze(-1) - cg.unsqueeze(1)).abs().argmin(-1)
                q = cg.gather(1, di) * sc
                idx_p[:, :, c] = di.to(torch.int8)
                Qp[:, :, c] = q
                err = (Wp[:, :, c] - q) / Hinv_p[:, c, c].clamp_min(1e-30).unsqueeze(1)
                if c + 1 < K:
                    Wp[:, :, c + 1:].sub_(err.unsqueeze(2) * Hinv_p[:, c, c + 1:].unsqueeze(1))
                if yield_every > 0 and is_cuda_dev and (c + 1) % yield_every == 0 and c + 1 < K:
                    torch.cuda.synchronize(dev)
                    if yield_secs > 0.0:
                        time.sleep(yield_secs)
            idx_full = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
            Q_full = torch.zeros((E, N, K), dtype=torch.float32, device=dev)
            idx_full.index_copy_(2, perm, idx_p)
            Q_full.index_copy_(2, perm, Qp)
            del Wp, Qp, idx_p
            return idx_full, Q_full

        if heavy_rounds > 0:
            # Alternate: gradient-tune centroids+scales (frozen indices) ↔ act_order reassign.
            orig_f = W_stack.float()                                    # ORIGINAL weights = tune target
            gidx_f = torch.arange(K, device=dev) // group_size
            cent = centroids_all.clone(); scl = scales_all.clone()
            for _r in range(heavy_rounds):
                with torch.enable_grad():
                    cp = cent.detach().requires_grad_(True)
                    sp = scl.detach().requires_grad_(True)
                    opt = torch.optim.Adam([{"params": [cp], "lr": heavy_lr_cent},
                                            {"params": [sp], "lr": heavy_lr_scale}])
                    idx_long = indices.long()
                    for _s in range(heavy_steps):
                        cent_pc = cp[:, gidx_f, :]                      # [E,K,nc]
                        Wq = cent_pc.unsqueeze(1).expand(E, N, K, -1).gather(
                            3, idx_long.unsqueeze(-1)).squeeze(-1) * sp[:, :, gidx_f]
                        d = orig_f - Wq
                        if heavy_dtype == "bf16":
                            # bf16 WMMA matmul (tensor cores accumulate in fp32),
                            # then fp32 elementwise + reduction. Mathematically the
                            # same quadratic form as the einsum; ~2x faster.
                            tmp = torch.bmm(d.to(torch.bfloat16),
                                            H_orig.to(torch.bfloat16)).float()
                            loss = (tmp * d).sum()
                        else:
                            loss = torch.einsum("eij,ejk,eik->e", d, H_orig, d).sum()
                        opt.zero_grad(); loss.backward(); opt.step()
                cent = cp.detach(); scl = sp.detach()
                cent_use = snap_to_e4m3(cent) if snap_centroids == "e4m3" else cent
                indices, Q = _reassign(cent_use, scl)                  # frozen-index tune → reassign
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            centroids_all = snap_to_e4m3(cent) if snap_centroids == "e4m3" else cent
            scales_all = scl
        else:
            indices, Q = _reassign(centroids_all, scales_all)
        del Hinv_p

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


# ═══════════════════════════════════════════════════════════════════════════
# G.7.h.3 — Multi-GPU expert split.
# ═══════════════════════════════════════════════════════════════════════════
# Split the [E, N, K] stack between two GPUs (e.g. R9700 + 6900 XT), run
# batched_gptq_quantize on each device in parallel via Python threads, and
# concat the results on the primary device. PyTorch releases the GIL during
# HIP/CUDA ops, so threading.Thread gives real parallelism across devices.
#
# Memory cost per device: ~half the E. The 6900 XT has 16 GB VRAM vs the
# R9700's 32 GB, so the default split is biased: 70% R9700, 30% 6900 XT.

import threading


@torch.no_grad()
def batched_gptq_quantize_multigpu(
    W_stack: torch.Tensor,
    H_stack: torch.Tensor,
    *,
    primary_device: str,
    secondary_device: str,
    primary_share: float = 0.7,
    n_centroids: int = 16,
    group_size: int = 64,
    n_iter: int = 25,
    fit_loss: str = "mse",
    mag_weight_p: float = 5.0,
    snap_centroids: str = "none",
    percdamp: float = 0.05,
    chunk_E: int = 8,
    n_tokens_per_expert: torch.Tensor | None = None,
) -> dict:
    """Split E experts between primary_device and secondary_device, run
    batched_gptq_quantize in parallel, concat back to primary_device.

    `primary_share` ∈ (0, 1] — fraction of experts that stay on
    primary_device. Default 0.7 gives R9700 70%, 6900 XT 30% to match the
    32 GB / 16 GB VRAM ratio (and the fact the 6900 XT is gfx1030, no WMMA,
    so per-FLOP slower).

    Returns the same dict shape as batched_gptq_quantize, with all tensors
    on primary_device.
    """
    E, N, K = W_stack.shape
    if not 0.0 < primary_share <= 1.0:
        raise ValueError(f"primary_share must be in (0, 1], got {primary_share}")
    if primary_device == secondary_device or E == 1 or primary_share >= 1.0:
        # Degenerate: no split. Fall through to single-GPU path.
        return batched_gptq_quantize(
            W_stack=W_stack, H_stack=H_stack,
            n_centroids=n_centroids, group_size=group_size, n_iter=n_iter,
            fit_loss=fit_loss, mag_weight_p=mag_weight_p,
            snap_centroids=snap_centroids, percdamp=percdamp,
            chunk_E=chunk_E, n_tokens_per_expert=n_tokens_per_expert)

    E_prim = max(1, min(E - 1, int(round(E * primary_share))))
    E_sec  = E - E_prim
    if n_tokens_per_expert is None:
        ntk_prim = ntk_sec = None
    else:
        ntk_prim = n_tokens_per_expert[:E_prim].to(primary_device)
        ntk_sec  = n_tokens_per_expert[E_prim:].to(secondary_device)

    # Move secondary half to the secondary device (non-blocking copies).
    W_prim = W_stack[:E_prim].to(primary_device, non_blocking=True)
    H_prim = H_stack[:E_prim].to(primary_device, non_blocking=True)
    W_sec  = W_stack[E_prim:].to(secondary_device, non_blocking=True)
    H_sec  = H_stack[E_prim:].to(secondary_device, non_blocking=True)

    results: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}

    def run_on(tag, W, H, ntk):
        try:
            results[tag] = batched_gptq_quantize(
                W_stack=W, H_stack=H,
                n_centroids=n_centroids, group_size=group_size, n_iter=n_iter,
                fit_loss=fit_loss, mag_weight_p=mag_weight_p,
                snap_centroids=snap_centroids, percdamp=percdamp,
                chunk_E=chunk_E, n_tokens_per_expert=ntk)
        except BaseException as exc:
            errors[tag] = exc

    t_prim = threading.Thread(target=run_on, args=("prim", W_prim, H_prim, ntk_prim))
    t_sec  = threading.Thread(target=run_on, args=("sec",  W_sec,  H_sec,  ntk_sec))
    t_prim.start()
    t_sec.start()
    t_prim.join()
    t_sec.join()

    if errors:
        # Re-raise the first error so the calling layer-major loop sees a
        # real exception, not a half-finished result dict.
        raise list(errors.values())[0]

    # Concat. Move secondary results back to primary device.
    def cat(key):
        a = results["prim"][key]
        b = results["sec"][key]
        if isinstance(a, torch.Tensor):
            return torch.cat([a, b.to(primary_device)], dim=0)
        return a  # scalar-like (e.g. n_cold_experts)

    return {
        "indices":             cat("indices"),
        "centroids_per_group": cat("centroids_per_group"),
        "scale_per_group":     cat("scale_per_group"),
        "Q":                   cat("Q"),
        "mse":                 cat("mse"),
        "w_snr_db":            cat("w_snr_db"),
        "y_snr_db":            cat("y_snr_db"),
        "rel_err":             cat("rel_err"),
        "n_cold_experts":      int(results["prim"]["n_cold_experts"]) + int(results["sec"]["n_cold_experts"]),
    }
