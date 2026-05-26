"""tune_gemm_ml8 — sweep mt_ml8_gemm BLOCK config space for Qwen3.5-4B shapes.

MAD-223 Phase G.6.a. The current `get_gemm_config` in
ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py returns the explicit
Phase-A "permissive defaults" stub (BLOCK_M=32, BLOCK_N=64, NUM_WARPS=4,
GROUP_SIZE_M=8). Measured impact: cell-E PPL pass time 14 s vs f16's 0.74 s
(19x slower, ~3% of FP8 peak).

This script benchmarks the underlying @triton.jit kernel directly across a
config grid for our production shapes:
    gate/up: K=2560, N=2560
    down:    K=9216, N=2560
× M tier: prefill (M=512) | decode (M=16)
× group_size: 64 (Cell E)

Outputs the best config per (shape, M tier) to JSON for `get_gemm_config`
to consume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ggml" / "src" / "ggml-cuda" / "aiter-integration" / "kernels"))

from ml8_runtime import Ml8Layer, ml8_layer_from_blob  # noqa: E402
from ml8_io import load_ml8_layer as load_ml8_blob  # noqa: E402
import gemm_ml8  # noqa: E402


SHAPES = [
    # (name, calibration_pt, K, N) — Qwen3.5-4B: hidden=2560, intermediate=9216
    ("gate", "model_layers_0_mlp_gate_proj.pt", 2560, 9216),
    ("up",   "model_layers_0_mlp_up_proj.pt",   2560, 9216),
    ("down", "model_layers_0_mlp_down_proj.pt", 9216, 2560),
]

# M tiers. Decode prefill is M=1..16 (we pad to 16); prefill is the chunk
# we're actually optimizing for PPL (M=512). 16 covers the worst-case
# decode dispatch.
M_TIERS = {"decode": 16, "prefill": 512}

# Sweep grid. Constraints baked in:
#   - BLOCK_SIZE_K = group_size = 64 (kernel constraint GROUP_K==BLOCK_K)
#   - NUM_STAGES = 1 (gfx1201 RDNA4 UAF for num_stages >= 2)
#   - WEIGHT_FORMAT = 1 (ml8 LUT path)
SWEEP_BLOCK_M = [16, 32, 64, 128]
SWEEP_BLOCK_N = [32, 64, 128, 256]
SWEEP_GROUP_SIZE_M = [1, 4, 8]
SWEEP_NUM_WARPS = [2, 4, 8]
SWEEP_KPACK = [1, 2]


def _benchmark_config(layer: Ml8Layer, M: int, cfg: dict, *, warmup: int = 3, iters: int = 20) -> float:
    """Compile + run the kernel with `cfg`, return median per-iter ms.
    Returns +inf if the config compiles but fails to run, or if it's illegal."""
    N = layer.n_rows
    K = layer.n_cols
    device = layer.indices_packed.device

    BLOCK_SIZE_M = cfg["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = cfg["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = layer.group_size  # forced

    if M % BLOCK_SIZE_M != 0 or N % BLOCK_SIZE_N != 0 or K % BLOCK_SIZE_K != 0:
        return float("inf")

    a_fp8 = torch.randn(M, K, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, device=device, dtype=torch.float32)
    c = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    stride_am, stride_ak = a_fp8.stride()
    stride_bk, stride_bn = layer.indices_packed.stride()
    stride_cm, stride_cn = c.stride()
    stride_bscale_k, stride_bscale_n = layer.scales_fp32.stride()
    stride_lut_k = layer.centroids_fp8.stride(0)

    grid_mn = (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)
    grid = (grid_mn,)
    even_k = (K % BLOCK_SIZE_K == 0)

    def _launch():
        gemm_ml8._gemm_a8w8_blockscale_kernel[grid](
            a_fp8, layer.indices_packed, c, a_scale, layer.scales_fp32,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            0, stride_cm, stride_cn,
            1, 0,
            stride_bscale_k, stride_bscale_n,
            GROUP_K=layer.group_size,
            GROUP_N=1,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            NUM_KSPLIT=1,
            SPLITK_BLOCK_SIZE=K,
            EVEN_K=even_k,
            GRID_MN=grid_mn,
            num_warps=cfg["NUM_WARPS"],
            num_stages=1,
            WEIGHT_FORMAT=1,
            N_CENTROIDS=layer.n_centroids,
            centroid_lut_ptr=layer.centroids_fp8,
            stride_lut_k=stride_lut_k,
            kpack=cfg["KPACK"],
        )

    # Warmup (also triggers JIT compile)
    try:
        for _ in range(warmup):
            _launch()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return float("inf")

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for _ in range(iters):
        start.record()
        _launch()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    median = times_ms[len(times_ms) // 2]
    return median


def _theoretical_us(M: int, K: int, N: int, peak_tflops: float = 600.0) -> float:
    """Theoretical lower bound assuming `peak_tflops` FP8 peak."""
    flops = 2.0 * M * K * N
    return flops / (peak_tflops * 1e12) * 1e6  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-dir", type=Path, default=Path("/home/kmbandy/models/cell-e"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("/home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8_tune.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    all_results = {}

    for shape_name, pt_filename, K_exp, N_exp in SHAPES:
        pt_path = args.calib_dir / pt_filename
        if not pt_path.exists():
            print(f"[skip] {shape_name}: {pt_path} missing")
            continue

        # cell-e blobs are torch.save'd dicts; ml8_io.load_ml8_layer reads them,
        # ml8_runtime.ml8_layer_from_blob promotes to a kernel-ready Ml8Layer.
        blob = load_ml8_blob(pt_path)
        layer = ml8_layer_from_blob(blob, device=device)
        assert layer.n_cols == K_exp, f"{shape_name}: K mismatch {layer.n_cols} vs {K_exp}"
        assert layer.n_rows == N_exp, f"{shape_name}: N mismatch {layer.n_rows} vs {N_exp}"

        for tier_name, M in M_TIERS.items():
            print(f"\n=== {shape_name} ({K_exp}×{N_exp}) M={M} ({tier_name}) ===")
            print(f"  theoretical FP8 peak: {_theoretical_us(M, K_exp, N_exp):.1f} µs")
            print(f"  (sweeping {len(SWEEP_BLOCK_M) * len(SWEEP_BLOCK_N) * len(SWEEP_GROUP_SIZE_M) * len(SWEEP_NUM_WARPS) * len(SWEEP_KPACK)} configs)")

            results = []
            t0 = time.time()
            for bm, bn, gsm, nw, kp in product(
                SWEEP_BLOCK_M, SWEEP_BLOCK_N, SWEEP_GROUP_SIZE_M, SWEEP_NUM_WARPS, SWEEP_KPACK
            ):
                if bm > M:  # block can't exceed M
                    continue
                cfg = {
                    "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                    "GROUP_SIZE_M": gsm, "NUM_WARPS": nw, "KPACK": kp,
                }
                t_ms = _benchmark_config(layer, M, cfg)
                if t_ms != float("inf"):
                    results.append((t_ms, cfg))
                    if len(results) % 10 == 0:
                        elapsed = time.time() - t0
                        print(f"    [{len(results)} configs ok, {elapsed:.0f}s elapsed] best so far: {min(results)[0]:.3f} ms")

            results.sort(key=lambda x: x[0])
            print(f"  [done in {time.time() - t0:.0f}s] {len(results)} legal configs")
            print(f"  TOP {args.top_k}:")
            for i, (t_ms, cfg) in enumerate(results[:args.top_k]):
                print(f"    #{i+1}: {t_ms:.3f} ms  {cfg}")

            if results:
                key = f"{shape_name}_M{M}"
                all_results[key] = {
                    "shape": {"M": M, "K": K_exp, "N": N_exp,
                              "group_size": layer.group_size, "n_centroids": layer.n_centroids},
                    "best": {"time_ms": results[0][0], **results[0][1]},
                    "top_k": [{"time_ms": t, **c} for t, c in results[:args.top_k]],
                }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
