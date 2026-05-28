#!/usr/bin/env python3
"""test_batched_lloyd_max.py — equivalence vs scalar Lloyd-Max.

Fixed-seed comparison: run `_batched_lloyd_max_signed` over E experts and
the scalar `_lloyd_max_signed` per-expert; centroids should match to within
~1e-5 (small drift from different reduction order in scatter_add vs
mask-then-sum). Also a smoke test for the full batched_gptq_quantize call
to confirm SNRs are sane after the kernel swap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from centroid_quantizer import _lloyd_max_signed
from batched_gptq import _batched_lloyd_max_signed, batched_gptq_quantize


def test_batched_vs_scalar(device="cuda:0"):
    torch.manual_seed(7)
    E, N, gs, n_levels, n_iter = 8, 96, 64, 16, 25
    M = N * gs

    # Fake x_norm in [-1, 1]-ish
    x_norm = torch.randn(E, N, gs, device=device, dtype=torch.float32) * 0.7
    col_hw = torch.rand(E, gs, device=device, dtype=torch.float32) + 0.1

    col_idx_template = torch.arange(gs, device=device).repeat(N)  # [N*gs]

    # Scalar reference (per expert)
    scalar_centroids = torch.zeros(E, n_levels, device=device, dtype=torch.float32)
    for e in range(E):
        s = x_norm[e].flatten()
        ce = _lloyd_max_signed(
            s, sample_col_idx=col_idx_template,
            col_weights=col_hw[e],
            n_levels=n_levels, n_iter=n_iter,
            fit_loss="mse", mag_weight_p=5.0,
        )
        scalar_centroids[e] = ce.to(device)

    # Batched
    samples_E = x_norm.reshape(E, M)
    batched_centroids = _batched_lloyd_max_signed(
        samples_E,
        col_weights_E=col_hw,
        col_idx=col_idx_template,
        n_levels=n_levels, n_iter=n_iter,
        fit_loss="mse", mag_weight_p=5.0,
    )

    abs_diff = (scalar_centroids - batched_centroids).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    # Per-expert relative SNR vs scalar
    sig = scalar_centroids.pow(2).mean(dim=1).clamp_min(1e-30)
    err = (scalar_centroids - batched_centroids).pow(2).mean(dim=1).clamp_min(1e-30)
    snr_db = 10.0 * torch.log10(sig / err)
    print(f"[mse  ] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  "
          f"per-expert SNR(dB) min={snr_db.min().item():.1f}  "
          f"med={snr_db.median().item():.1f}")
    assert max_abs < 5e-2, f"max abs diff {max_abs:.3e} > 5e-2 — math drift"
    assert snr_db.min().item() > 30.0, f"per-expert SNR floor {snr_db.min().item():.1f} dB too low"

    # mag-weighted path
    scalar_mw = torch.zeros(E, n_levels, device=device, dtype=torch.float32)
    for e in range(E):
        s = x_norm[e].flatten()
        ce = _lloyd_max_signed(
            s, sample_col_idx=col_idx_template,
            col_weights=col_hw[e],
            n_levels=n_levels, n_iter=n_iter,
            fit_loss="mag_weighted", mag_weight_p=5.0,
        )
        scalar_mw[e] = ce.to(device)
    batched_mw = _batched_lloyd_max_signed(
        samples_E, col_weights_E=col_hw, col_idx=col_idx_template,
        n_levels=n_levels, n_iter=n_iter,
        fit_loss="mag_weighted", mag_weight_p=5.0,
    )
    abs_diff = (scalar_mw - batched_mw).abs()
    sig = scalar_mw.pow(2).mean(dim=1).clamp_min(1e-30)
    err = (scalar_mw - batched_mw).pow(2).mean(dim=1).clamp_min(1e-30)
    snr_db = 10.0 * torch.log10(sig / err)
    print(f"[mw   ] max_abs={abs_diff.max().item():.3e}  "
          f"mean_abs={abs_diff.mean().item():.3e}  "
          f"per-expert SNR(dB) min={snr_db.min().item():.1f}  "
          f"med={snr_db.median().item():.1f}")
    assert snr_db.min().item() > 30.0, f"mw path per-expert SNR floor too low"

    print("PASS — batched matches scalar to >30 dB per expert on both loss modes")


def test_batched_gptq_smoke(device="cuda:0"):
    """End-to-end: batched_gptq_quantize with the new kernel still produces
    sane SNRs on a small fixture."""
    torch.manual_seed(11)
    E, N, K, gs = 4, 64, 256, 64
    W = torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.05
    H_one = torch.randn(K, K, device=device, dtype=torch.float32) * 0.02
    H_one = H_one @ H_one.T + 0.1 * torch.eye(K, device=device)
    H_stack = H_one.unsqueeze(0).expand(E, K, K)

    out = batched_gptq_quantize(
        W_stack=W, H_stack=H_stack,
        n_centroids=16, group_size=gs, n_iter=25,
        fit_loss="mse", snap_centroids="e4m3", percdamp=0.05,
    )
    w_snr_med = float(out["w_snr_db"].median())
    y_snr_med = float(out["y_snr_db"].median())
    print(f"[smoke] W_SNR_med={w_snr_med:.1f}dB  Y_SNR_med={y_snr_med:.1f}dB  "
          f"E={E} N={N} K={K} gs={gs}")
    assert w_snr_med > 10.0, f"W_SNR_med {w_snr_med:.1f} dB too low — kernel broken?"
    assert y_snr_med > 10.0, f"Y_SNR_med {y_snr_med:.1f} dB too low — kernel broken?"
    print("PASS — batched_gptq end-to-end SNRs in expected range")


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"== device={dev}  torch={torch.__version__}")
    test_batched_vs_scalar(dev)
    test_batched_gptq_smoke(dev)
    print("\nALL OK")
