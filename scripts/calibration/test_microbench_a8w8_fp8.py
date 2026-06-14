# scripts/calibration/test_microbench_a8w8_fp8.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microbench_a8w8_fp8 import gemm_tflops, default_shapes


def test_gemm_tflops_matches_2mnk_over_seconds():
    # 2*M*N*K flops; at M=N=K=1024 and 1 ms -> 2*1024^3 / 1e-3 / 1e12 TFLOPS
    tflops = gemm_tflops(M=1024, N=1024, K=1024, seconds=1e-3)
    assert abs(tflops - (2 * 1024**3 / 1e-3 / 1e12)) < 1e-6


def test_default_shapes_cover_4b_mlp_and_oproj():
    shapes = {(n, k) for (_name, n, k) in default_shapes()}
    assert (9216, 2560) in shapes   # gate/up
    assert (2560, 9216) in shapes   # down
    assert (2560, 2560) in shapes   # o_proj
