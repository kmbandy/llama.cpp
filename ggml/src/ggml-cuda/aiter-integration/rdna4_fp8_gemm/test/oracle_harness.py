#!/usr/bin/env python3
"""Correctness oracle for the RDNA4 fp8 GEMM: kernel == torch._scaled_mm within fp8."""
import ctypes, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent.parent
LIB = HERE / "out" / "librdna4_gemm.so"

# Reach the repo's tests/ + scripts/calibration for the ml8 reference + packing helper.
# HERE = <repo>/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
_REPO_ROOT = HERE.parents[4]
for _p in (_REPO_ROOT / "tests", _REPO_ROOT / "scripts" / "calibration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


_LIB_CACHE = None


def _lib():
    # Cache the handle: the bench (gemm_bench.py) calls gemm_fp8 in a hot timed
    # loop, so rebuilding the ctypes wrapper per call would skew the timing.
    global _LIB_CACHE
    if _LIB_CACHE is None:
        lib = ctypes.CDLL(str(LIB))
        lib.rdna4_gemm_fp8_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.rdna4_gemm_fp8_forward.restype = None
        _LIB_CACHE = lib
    return _LIB_CACHE


def gemm_fp8(a_fp8, b_fp8, a_scale, b_scale):
    M, K = a_fp8.shape; N = b_fp8.shape[1]
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    _lib().rdna4_gemm_fp8_forward(
        a_fp8.data_ptr(), b_fp8.data_ptr(), c.data_ptr(),
        a_scale.data_ptr(), b_scale.data_ptr(), M, N, K, None)
    torch.cuda.synchronize()
    return c


def _lib_ml8():
    # The ml8 front-end shares the same .so as the fp8 path; declare its signature
    # lazily on the cached handle (only the first ml8 call pays the setup cost).
    lib = _lib()
    if not getattr(lib, "_ml8_ready", False):
        lib.rdna4_gemm_ml8_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # A, B_idx, C
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,   # a_scale, centroids_fp8, b_group_scale
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]  # M,N,K,group_size,stream
        lib.rdna4_gemm_ml8_forward.restype = None
        lib._ml8_ready = True
    return lib


def gemm_ml8(a_fp8, b_idx, a_scale, centroids_fp8, b_group_scale, M, N, K, group_size):
    """ml8 front-end: A plain fp8 [M,K]; B = packed 4-bit indices [K/2,N] (lo-first) +
    per-K-group fp8 centroid LUT [n_groups_k,16] + per-(group,N) fp32 scale [n_groups_k,N]."""
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    _lib_ml8().rdna4_gemm_ml8_forward(
        a_fp8.data_ptr(), b_idx.data_ptr(), c.data_ptr(),
        a_scale.data_ptr(), centroids_fp8.data_ptr(), b_group_scale.data_ptr(),
        M, N, K, group_size, None)
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


# --- ml8 4-bit LUT-dequant front-end (same tiled WMMA core, ml8 B operand) -------
# Ground truth is tests/test_ml8_kernel_stage1_dequant.reference_dequant_gemm and the
# scripts/calibration packing helper (mirror them exactly so kernel == reference).

def _pack_kn(indices_kn, N, K):
    """[K,N] int8 indices -> [K//2,N] uint8 lo-first packed (the kernel's B layout)."""
    import numpy as np
    from ml8_to_packed import pack_indices
    packed_bytes = pack_indices(indices_kn.T.cpu().contiguous(), nibble_lo_first=True)
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    return torch.from_numpy(packed_np.T.copy()).contiguous().to(indices_kn.device)


def _ml8_case(M, N, K, group_size, seed, tol):
    from test_ml8_kernel_stage1_dequant import reference_dequant_gemm
    dev = torch.device("cuda"); torch.manual_seed(seed)
    n_centroids = 16
    n_groups_k = K // group_size

    # Build a synthetic ml8 layer (mirrors bench_ml8_gemm.build_synthetic_layer /
    # the stage1 test): random A->fp8, random centroids->fp8, random indices [0,15],
    # per-(group,N) fp32 scales, per-row fp32 a_scale.
    a_fp8 = ((torch.randn(M, K, device=dev) * 0.3).clamp(-1.5, 1.5)).to(torch.float8_e4m3fn)
    centroids_fp8 = (torch.randn(n_groups_k, n_centroids, device=dev) * 0.5).to(torch.float8_e4m3fn)
    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=dev)
    b_scale = torch.randn(n_groups_k, N, device=dev).abs() * 0.1 + 0.01
    a_scale = torch.randn(M, device=dev).abs() * 0.1 + 0.01

    # Reference: fp32 dequant -> GEMM -> a_scale, then bf16 output cast (matches kernel).
    C_ref = reference_dequant_gemm(
        a_fp8.to(torch.float32), indices, centroids_fp8.to(torch.float32),
        b_scale, a_scale, group_size).to(torch.bfloat16).to(torch.float32)

    b_idx = _pack_kn(indices, N, K)
    C_kernel = gemm_ml8(a_fp8, b_idx, a_scale, centroids_fp8, b_scale,
                        M, N, K, group_size).to(torch.float32)
    max_err = (C_kernel - C_ref).abs().max().item()
    # Same class of tolerance as the fp8 cases (5e-2 rel-scale): fp8 quant noise on A +
    # centroids, bf16 output cast. NOT loosened — set by what fp8 rounding produces.
    assert max_err < tol, f"ml8 M={M} N={N} K={K} gs={group_size}: max_err {max_err:.4g} >= {tol}"


def test_ml8_single_tile():  _ml8_case(16, 16, 64, 64, 42, 5e-2)
def test_ml8_real_shape():   _ml8_case(64, 2560, 9216, 64, 7, 5e-2)  # ml8-4 down-proj-ish


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no GPU"); sys.exit(1)
    for t in (test_single_tile, test_square, test_real_shape,
              test_ml8_single_tile, test_ml8_real_shape):
        t(); print(f"  ✓ {t.__name__}")
    print("PASS")
