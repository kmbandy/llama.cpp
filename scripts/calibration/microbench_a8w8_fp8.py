"""4-cell fp8 GEMM microbench (gfx1201): aiter Triton a8w8-blockscale vs
torch._scaled_mm, at the real 4B linear shapes. Emits JSON. The toolchain is the
variable across runs (pre-bump / post-bump / +configs); this script is constant."""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch


def gemm_tflops(M: int, N: int, K: int, seconds: float) -> float:
    return (2.0 * M * N * K) / seconds / 1e12


def default_shapes():
    # (name, N=out, K=in) — Qwen3.5-4B (hidden=2560, intermediate=9216)
    return [("gate", 9216, 2560), ("up", 9216, 2560),
            ("down", 2560, 9216), ("o_proj", 2560, 2560)]


def _median_seconds(fn, *, warmup=5, iters=30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
