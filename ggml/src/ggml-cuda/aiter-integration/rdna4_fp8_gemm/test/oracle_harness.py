#!/usr/bin/env python3
"""Correctness oracle for the RDNA4 fp8 GEMM: kernel == torch._scaled_mm within fp8."""
import ctypes, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent.parent
LIB = HERE / "out" / "librdna4_gemm.so"


def _lib():
    lib = ctypes.CDLL(str(LIB))
    lib.rdna4_gemm_fp8_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.rdna4_gemm_fp8_forward.restype = None
    return lib


def gemm_fp8(a_fp8, b_fp8, a_scale, b_scale):
    M, K = a_fp8.shape; N = b_fp8.shape[1]
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    _lib().rdna4_gemm_fp8_forward(
        a_fp8.data_ptr(), b_fp8.data_ptr(), c.data_ptr(),
        a_scale.data_ptr(), b_scale.data_ptr(), M, N, K, None)
    torch.cuda.synchronize()
    return c


def _case(M, N, K, seed, tol):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    a = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    b = (torch.randn(K, N, device=dev) * 0.3).to(torch.float8_e4m3fn)
    a_scale = (torch.rand(M, device=dev) * 0.1 + 0.01)
    b_scale = (torch.rand(N, device=dev) * 0.1 + 0.01)
    # reference: torch._scaled_mm wants x row-major, w col-major; scales [M,1],[1,N]
    ref = torch._scaled_mm(a, b.t().contiguous().t(),
                           scale_a=a_scale[:, None].float(), scale_b=b_scale[None, :].float(),
                           out_dtype=torch.bfloat16).to(torch.float32)
    out = gemm_fp8(a, b, a_scale, b_scale).to(torch.float32)
    max_err = (out - ref).abs().max().item()
    assert max_err < tol, f"M={M} N={N} K={K}: max_err {max_err:.4g} >= {tol}"


def test_single_tile():   _case(16, 16, 16, 1, 5e-2)
def test_square():        _case(256, 256, 256, 2, 5e-2)
def test_real_shape():    _case(2048, 2560, 9216, 3, 5e-2)  # down-proj-ish

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no GPU"); sys.exit(1)
    for t in (test_single_tile, test_square, test_real_shape):
        t(); print(f"  ✓ {t.__name__}")
    print("PASS")
