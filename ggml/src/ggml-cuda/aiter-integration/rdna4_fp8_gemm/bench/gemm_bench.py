#!/usr/bin/env python3
"""Perf ratchet for the RDNA4 fp8 GEMM: TFLOPS, % of measured 307 TF ceiling, x hipBLASLt."""
import statistics, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from oracle_harness import gemm_fp8, gemm_ml8, _pack_kn  # noqa: E402

CEILING_TF = 307.0          # measured raw fp8 WMMA peak (wmma_peak.hip)


def tflops(M, N, K, sec):
    return 2.0 * M * N * K / sec / 1e12


def _median(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def shapes():
    return [("gate", 9216, 2560), ("down", 2560, 9216), ("o_proj", 2560, 2560)]


def _build_ml8(N, K, dev, group_size=64, n_centroids=16, seed=0):
    """Synthetic ml8 layer at a (N=out, K=in) shape — mirrors build_synthetic_layer:
    random centroids->fp8, per-(group,N) fp32 scales, random indices [0,15] packed [K/2,N]."""
    torch.manual_seed(seed); ng = K // group_size
    centroids = (torch.randn(ng, n_centroids, device=dev) * 0.5).to(torch.float8_e4m3fn)
    b_scale = torch.randn(ng, N, device=dev).abs() * 0.1 + 0.01
    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=dev)
    b_idx = _pack_kn(indices, N, K)
    return b_idx, centroids, b_scale, group_size


def main():
    dev = torch.device("cuda")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ml8 = "--ml8" in sys.argv
    M = int(args[0]) if args else 8192
    print(f"M={M}  ceiling={CEILING_TF} TF")
    if ml8:
        # ml8 4-bit front-end on the same tiled core: TFLOPS vs the fp8 core + the dense ceiling.
        print(f"{'shape':7} {'fp8 TF':>8} {'%307':>6} {'ml8 TF':>8} {'%307':>6} {'ml8/fp8':>8}")
        for name, N, K in shapes():
            a = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
            b = (torch.randn(K, N, device=dev) * 0.3).to(torch.float8_e4m3fn)
            asc = torch.ones(M, device=dev); bsc = torch.ones(N, device=dev)
            fp8 = tflops(M, N, K, _median(lambda: gemm_fp8(a, b, asc, bsc)))
            b_idx, cent, b_scale, gs = _build_ml8(N, K, dev)
            m8 = tflops(M, N, K, _median(
                lambda: gemm_ml8(a, b_idx, asc, cent, b_scale, M, N, K, gs)))
            print(f"{name:7} {fp8:8.1f} {fp8/CEILING_TF*100:5.1f}% "
                  f"{m8:8.1f} {m8/CEILING_TF*100:5.1f}% {m8/fp8:8.2f}")
        return
    print(f"{'shape':7} {'ours TF':>8} {'%307':>6} {'hipBLASLt TF':>12} {'x hbl':>6}")
    for name, N, K in shapes():
        a = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
        b = (torch.randn(K, N, device=dev) * 0.3).to(torch.float8_e4m3fn)
        asc = torch.ones(M, device=dev); bsc = torch.ones(N, device=dev)
        ours = tflops(M, N, K, _median(lambda: gemm_fp8(a, b, asc, bsc)))
        bt = b.t().contiguous().t()
        sa = torch.ones((M, 1), device=dev); sb = torch.ones((1, N), device=dev)
        hbl = tflops(M, N, K, _median(lambda: torch._scaled_mm(a, bt, scale_a=sa, scale_b=sb,
                                                               out_dtype=torch.bfloat16)))
        print(f"{name:7} {ours:8.1f} {ours/CEILING_TF*100:5.1f}% {hbl:12.1f} {ours/hbl:5.2f}")


if __name__ == "__main__":
    main()
