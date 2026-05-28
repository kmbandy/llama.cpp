#!/usr/bin/env python3
"""profile_lloyd_max.py — measure Lloyd-Max share of batched_gptq wall time.

Runs batched_gptq_quantize on representative Qwen3.6-35B-A3B MoE shapes
(gate_proj/up_proj: E=128, N=768, K=2048; down_proj: E=128, N=2048, K=768)
with timers wrapped around the Lloyd-Max inner loop.

Reports per-task wall time, Lloyd-Max time, and Lloyd-Max share. If share
is ≥20%, CPU multiprocess Lloyd-Max is worth porting.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import centroid_quantizer as cq
import batched_gptq as bg


# ─── Instrument Lloyd-Max — wrap the original with a timer accumulator. ────
_lm_total_s = 0.0
_lm_calls = 0
_orig_lloyd = cq._lloyd_max_signed


def _timed_lloyd(*args, **kwargs):
    global _lm_total_s, _lm_calls
    torch.cuda.synchronize()
    t0 = time.time()
    out = _orig_lloyd(*args, **kwargs)
    torch.cuda.synchronize()
    _lm_total_s += time.time() - t0
    _lm_calls += 1
    return out


# Patch the symbol that batched_gptq imported.
bg._lloyd_max_signed = _timed_lloyd


def run_task(*, E: int, N: int, K: int, group_size: int, n_centroids: int,
             n_iter: int, fit_loss: str, snap_centroids: str, device: str,
             label: str) -> None:
    global _lm_total_s, _lm_calls
    _lm_total_s = 0.0
    _lm_calls = 0

    print(f"\n══ {label}: E={E}, N={N}, K={K}, group_size={group_size}, "
          f"n_iter={n_iter}, fit_loss={fit_loss}, snap={snap_centroids} ══")

    torch.manual_seed(0)
    W = torch.randn(E, N, K, device=device, dtype=torch.float32) * 0.05
    H_one = torch.randn(K, K, device=device, dtype=torch.float32) * 0.02
    H_one = H_one @ H_one.T + 0.1 * torch.eye(K, device=device)
    H_stack = H_one.unsqueeze(0).expand(E, K, K)

    torch.cuda.synchronize()
    t0 = time.time()
    out = bg.batched_gptq_quantize(
        W_stack=W, H_stack=H_stack,
        n_centroids=n_centroids, group_size=group_size,
        n_iter=n_iter, fit_loss=fit_loss,
        snap_centroids=snap_centroids,
        percdamp=0.05,
    )
    torch.cuda.synchronize()
    total_s = time.time() - t0

    share = 100.0 * _lm_total_s / max(total_s, 1e-9)
    n_groups = K // group_size
    expected_calls = E * n_groups
    per_call_ms = 1000.0 * _lm_total_s / max(_lm_calls, 1)

    print(f"  total task: {total_s:.2f}s")
    print(f"  lloyd-max:  {_lm_total_s:.2f}s ({share:.1f}%)  "
          f"calls={_lm_calls} (expected {expected_calls})")
    print(f"  per-call:   {per_call_ms:.2f}ms")
    print(f"  remainder:  {total_s - _lm_total_s:.2f}s "
          f"({100.0*(total_s-_lm_total_s)/total_s:.1f}%)  "
          f"= Cholesky + GPTQ propagation + SNR metrics")

    del W, H_one, H_stack, out
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=25)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--fit-loss", default="mse",
                   choices=["mse", "mag_weighted"])
    p.add_argument("--snap-centroids", default="e4m3",
                   choices=["none", "e4m3"])
    args = p.parse_args()

    print(f"device={args.device}  group_size={args.group_size}  "
          f"n_iter={args.n_iter}  fit_loss={args.fit_loss}  "
          f"snap={args.snap_centroids}")
    print(f"torch={torch.__version__}  hip={torch.version.hip}")

    # Qwen3.6 35B-A3B shapes (hidden=2048, intermediate=768, n_experts=128).
    # gate/up: N=intermediate, K=hidden ; down: N=hidden, K=intermediate.
    run_task(E=128, N=768, K=2048, group_size=args.group_size,
             n_centroids=args.n_centroids, n_iter=args.n_iter,
             fit_loss=args.fit_loss, snap_centroids=args.snap_centroids,
             device=args.device, label="gate/up shape")
    run_task(E=128, N=2048, K=768, group_size=args.group_size,
             n_centroids=args.n_centroids, n_iter=args.n_iter,
             fit_loss=args.fit_loss, snap_centroids=args.snap_centroids,
             device=args.device, label="down shape")


if __name__ == "__main__":
    main()
