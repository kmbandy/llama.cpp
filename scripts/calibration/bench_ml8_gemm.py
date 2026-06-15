"""MAD-299 ml8 LUT GEMM benchmark — %-of-383 (RDNA4 dense FP8) ratchet.

Builds synthetic ml8 layers at the real Qwen3.5-4B linear shapes, launches the
WEIGHT_FORMAT=1 kernel at a chosen tile/warp config, times the kernel only
(layer is prebuilt — no per-call packing in the timed region), and reports
TFLOPS and % of dense FP8. Emits JSON. The kernel is the variable across runs;
this harness is constant."""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
_KERNELS = _THIS.parent.parent / "ggml/src/ggml-cuda/aiter-integration/kernels"
if str(_KERNELS) not in sys.path:
    sys.path.insert(0, str(_KERNELS))

DENSE_FP8_TFLOPS = 383.0  # official R9700 dense FP8 (E4M3/E5M2) matrix peak


def tflops(M: int, N: int, K: int, seconds: float) -> float:
    return (2.0 * M * N * K) / seconds / 1e12


def pct_of_dense(tf: float) -> float:
    return tf / DENSE_FP8_TFLOPS * 100.0


def default_shapes():
    # (name, N=out, K=in) — Qwen3.5-4B (hidden=2560, intermediate=9216)
    return [("gate", 9216, 2560), ("up", 9216, 2560),
            ("down", 2560, 9216), ("o_proj", 2560, 2560)]


def build_synthetic_layer(N, K, group_size=64, n_centroids=16, device="cuda", seed=0):
    from ml8_runtime import layer_from_components
    g = torch.Generator().manual_seed(seed)
    G = K // group_size
    centroids = torch.randn(G, n_centroids, generator=g) * 0.5            # fp32 [G,16]
    scales = torch.randn(N, G, generator=g).abs() * 0.1 + 0.01            # fp32 [N,G]
    indices = torch.randint(0, n_centroids, (N, K), generator=g, dtype=torch.uint8)
    gidx = torch.arange(K) // group_size                                  # [K]
    return layer_from_components(centroids, scales, indices, gidx, device=device)


def launch(a_fp8, layer, a_scale, *, block_m=16, block_n=16, num_warps=4):
    """Direct WEIGHT_FORMAT=1 launch at a chosen config. Returns C [M,N] bf16."""
    import gemm_ml8
    M, K = a_fp8.shape
    N = layer.n_rows
    gs = layer.group_size
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    stride_am, stride_ak = a_fp8.stride()
    stride_bk, stride_bn = layer.indices_packed.stride()
    stride_cm, stride_cn = c.stride()
    stride_bscale_k, stride_bscale_n = layer.scales_fp32.stride()
    grid_mn = (M // block_m) * (N // block_n)
    gemm_ml8._gemm_a8w8_blockscale_kernel[(grid_mn,)](
        a_fp8, layer.indices_packed, c, a_scale, layer.scales_fp32,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        0, stride_cm, stride_cn,
        1, 0, stride_bscale_k, stride_bscale_n,
        GROUP_K=gs, GROUP_N=1,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=gs,
        GROUP_SIZE_M=1, NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=K,
        EVEN_K=(K % gs == 0), GRID_MN=grid_mn, num_stages=1,
        WEIGHT_FORMAT=1, N_CENTROIDS=layer.n_centroids,
        centroid_lut_ptr=layer.centroids_fp8, stride_lut_k=layer.centroids_fp8.stride(0),
        num_warps=num_warps,
    )
    return c


def _median_seconds(fn, *, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_shape(name, N, K, M, dev, *, block_m, block_n, num_warps):
    layer = build_synthetic_layer(N, K, device=dev)
    a_fp8 = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, dtype=torch.float32, device=dev)
    sec = _median_seconds(lambda: launch(
        a_fp8, layer, a_scale, block_m=block_m, block_n=block_n, num_warps=num_warps))
    tf = tflops(M, N, K, sec)
    return dict(shape=name, M=M, N=N, K=K, block_m=block_m, block_n=block_n,
                num_warps=num_warps, ms=sec * 1e3, tflops=tf, pct383=pct_of_dense(tf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--m-tiers", type=int, nargs="+", default=[2048])
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=16)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/tmp/mad299_bench.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    for name, N, K in default_shapes():
        for M in args.m_tiers:
            try:
                rows.append(bench_shape(name, N, K, M, dev,
                                        block_m=args.block_m, block_n=args.block_n,
                                        num_warps=args.num_warps))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(shape=name, M=M, N=N, K=K, error=str(e)[:300]))
    out = dict(label=args.label, triton_version=__import__("triton").__version__, rows=rows)
    args.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        if "tflops" in r:
            print(f"{r['shape']:7s} M={r['M']:5d} bm={r['block_m']:3d} bn={r['block_n']:3d} "
                  f"w={r['num_warps']} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TF  {r['pct383']:5.1f}% of 383")
        else:
            print(f"{r['shape']:7s} M={r['M']:5d} ERROR {r['error']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
