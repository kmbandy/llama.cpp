#!/usr/bin/env python3
"""MAD-299 — correctness oracle + harness-math tests for the ml8 LUT GEMM
optimization. The oracle (kernel == dequant-in-torch) is the invariant that must
stay green through every kernel edit; the %-of-383 bench (bench_ml8_gemm.py) is
the perf ratchet."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))


def test_tflops_and_pct_math():
    import bench_ml8_gemm as B
    expected_tflops = 2 * 1024**3 / 1e-3 / 1e12
    actual_tflops = B.tflops(1024, 1024, 1024, 1e-3)
    assert abs(actual_tflops - expected_tflops) < 1e-9, (
        f"tflops(1024,1024,1024,1e-3): got {actual_tflops}, expected {expected_tflops}")
    actual_pct_383 = B.pct_of_dense(383.0)
    assert abs(actual_pct_383 - 100.0) < 1e-9, (
        f"pct_of_dense(383.0): got {actual_pct_383}, expected 100.0")
    expected_pct_11 = 11.0 / 383.0 * 100.0
    actual_pct_11 = B.pct_of_dense(11.0)
    assert abs(actual_pct_11 - expected_pct_11) < 1e-6, (
        f"pct_of_dense(11.0): got {actual_pct_11}, expected {expected_pct_11}")
    assert B.DENSE_FP8_TFLOPS == 383.0, (
        f"DENSE_FP8_TFLOPS: got {B.DENSE_FP8_TFLOPS}, expected 383.0")


def _pack_kn(indices_kn, N: int, K: int):
    """[K,N] int8 indices -> [K//2,N] uint8 lo-first packed (kernel layout)."""
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))
    from ml8_to_packed import pack_indices  # noqa: E402
    packed_bytes = pack_indices(indices_kn.T.cpu().contiguous(), nibble_lo_first=True)
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    return __import__("torch").from_numpy(packed_np.T.copy()).contiguous().to(indices_kn.device)


def _oracle_case(M, N, K, group_size, seed, tol):
    import torch
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_ml8_kernel_stage1_dequant import reference_dequant_gemm, run_ml8_kernel  # noqa: E402

    device = torch.device("cuda")
    torch.manual_seed(seed)
    n_centroids = 16
    n_groups_k = K // group_size

    a_fp8 = ((torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)).to(torch.float8_e4m3fn)
    centroids_fp8 = (torch.randn(n_groups_k, n_centroids, device=device) * 0.5).to(torch.float8_e4m3fn)
    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)
    b_scale = torch.randn(n_groups_k, N, device=device).abs() * 0.1 + 0.01
    a_scale = torch.randn(M, device=device).abs() * 0.1 + 0.01

    C_ref = reference_dequant_gemm(
        a_fp8.to(torch.float32), indices, centroids_fp8.to(torch.float32),
        b_scale, a_scale, group_size)
    b_packed = _pack_kn(indices, N, K)
    C_kernel = run_ml8_kernel(a_fp8, b_packed, centroids_fp8, b_scale, a_scale,
                              group_size=group_size, n_centroids=n_centroids)
    max_err = (C_kernel.to(torch.float32) - C_ref.to(torch.bfloat16).to(torch.float32)).abs().max().item()
    # Tolerance rationale (mirrors test_ml8_kernel_stage1_dequant.py):
    # bf16 output-cast rounding + fp32->bf16 + fp8 quant noise on A and centroids.
    # Single-tile is looser (5e-2) because fp8 noise is a larger fraction of a small sum;
    # multi-tile/large shapes are tighter (1e-2) as errors average down.
    assert max_err < tol, f"M={M} N={N} K={K}: max_err {max_err:.4g} exceeds {tol}"


def test_oracle_single_tile():
    _oracle_case(M=16, N=16, K=64, group_size=64, seed=42, tol=5e-2)


def test_oracle_multi_tile_cross_kgroup():
    _oracle_case(M=32, N=32, K=256, group_size=64, seed=123, tol=1e-2)


def test_oracle_real_4b_shape():
    _oracle_case(M=64, N=2560, K=9216, group_size=64, seed=7, tol=1e-2)


def test_oracle_asymmetric_shape():
    _oracle_case(M=16, N=48, K=64, group_size=64, seed=7, tol=2e-2)
