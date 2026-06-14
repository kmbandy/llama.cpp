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


def _run_aiter_a8w8(M, N, K, dev):
    import sys
    sys.path.insert(0, '/home/kmbandy/GitHub/aiter')
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    # x: (M, K) row-major fp8; w: (N, K) fp8 (kernel transposes internally)
    # x_scale: (M, ceil(K/128)); w_scale: (ceil(N/128), ceil(K/128))
    # block_size_n=128, block_size_k=128 per aiter reference test
    BLOCK = 128
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    xs = torch.ones((M, (K + BLOCK - 1) // BLOCK), device=dev, dtype=torch.float32)
    ws = torch.ones(((N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK), device=dev, dtype=torch.float32)
    return lambda: gemm_a8w8_blockscale(x, w, xs, ws, dtype=torch.bfloat16)


def _run_scaled_mm(M, N, K, dev):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    # torch._scaled_mm requires x row-major and w column-major.
    # .t() of (N, K) row-major produces (K, N) with column-major strides — do NOT .contiguous().
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    sa = torch.ones((M, 1), device=dev, dtype=torch.float32)
    sb = torch.ones((1, N), device=dev, dtype=torch.float32)
    return lambda: torch._scaled_mm(x, w, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. pre-bump / post-bump / post-bump+configs")
    ap.add_argument("--m-tiers", type=int, nargs="+", default=[16, 512, 2048])
    ap.add_argument("--out", type=Path, default=Path("/tmp/phase0_microbench.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    for name, N, K in default_shapes():
        for M in args.m_tiers:
            for cell, mk in (("aiter_a8w8", _run_aiter_a8w8), ("scaled_mm", _run_scaled_mm)):
                try:
                    sec = _median_seconds(mk(M, N, K, dev))
                    rows.append(dict(shape=name, M=M, N=N, K=K, cell=cell,
                                     ms=sec * 1e3, tflops=gemm_tflops(M, N, K, sec)))
                except Exception as e:  # noqa: BLE001 — record, don't abort the sweep
                    rows.append(dict(shape=name, M=M, N=N, K=K, cell=cell, error=str(e)[:200]))
    out = dict(label=args.label, triton_version=__import__("triton").__version__, rows=rows)
    args.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        if "tflops" in r:
            print(f"{r['shape']:7s} M={r['M']:5d} {r['cell']:11s} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TFLOPS")
        else:
            print(f"{r['shape']:7s} M={r['M']:5d} {r['cell']:11s} ERROR {r['error']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
