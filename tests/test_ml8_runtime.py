#!/usr/bin/env python3
"""MAD-223 Phase C.1 — ml8_runtime full pipeline integration test.

Exercises the end-to-end Python API:
  synthesize Ml8 blob → pack to .ml8 binary on disk → load_ml8_layer →
  ml8_gemm vs PyTorch reference (dequantize + matmul).

This is the same correctness validation as the Stage 1/2/3 kernel tests
but going through the real file-load + Ml8Layer path that Phase D's
`reconstruct_model.py --use-ml8-kernel` will use.

Usage:
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \\
    /home/kmbandy/venvs/agents/bin/python3 tests/test_ml8_runtime.py
"""

import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from ml8_to_packed import pack_layer  # noqa: E402
from ml8_runtime import (  # noqa: E402
    Ml8Layer,
    load_ml8_layer,
    ml8_gemm,
    dequantize_ml8_layer,
)
from test_ml8_to_packed import synth_blob  # noqa: E402


def write_synth_layer_to_disk(blob: dict, path: Path) -> None:
    """Pack a synthetic blob and write to disk as .ml8."""
    packed = pack_layer(blob, nibble_lo_first=True)
    path.write_bytes(packed)


# --- Tests ----------------------------------------------------------------


def test_load_roundtrip_dequant_matches_blob():
    """`load_ml8_layer` → `dequantize_ml8_layer` should reconstruct the
    same fp32 weight as the on-disk blob's reconstruct formula."""
    blob = synth_blob(n_rows=32, n_cols=128, n_centroids=16, group_size=64, seed=11)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "synth.ml8"
        write_synth_layer_to_disk(blob, path)
        layer = load_ml8_layer(path, device="cpu")

    print(f"  loaded: {layer}")

    # Reconstruct via ml8_io (the calibration-side reference)
    sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))
    from ml8_io import reconstruct_weight  # noqa: E402
    W_ref = reconstruct_weight(blob)

    # Reconstruct via the runtime's dequantize helper
    W_runtime = dequantize_ml8_layer(layer).cpu()

    max_diff = (W_ref - W_runtime).abs().max().item()
    assert max_diff < 1e-6, f"dequantize_ml8_layer mismatch: max_diff={max_diff}"
    print(f"  ✓ dequantize matches ml8_io.reconstruct_weight (max_diff={max_diff:.2e})")


def test_ml8_gemm_matches_dequant_matmul():
    """`ml8_gemm(A, layer)` should equal `A @ dequant(layer).T` within fp8
    quantization noise (bf16 output cast)."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=16, n_cols=64, n_centroids=16, group_size=64, seed=22)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "synth.ml8"
        write_synth_layer_to_disk(blob, path)
        layer = load_ml8_layer(path, device=device)

    print(f"  loaded: {layer}")

    # Synthesize A (fp8) and a_scale (fp32)
    M = 16
    K = layer.n_cols
    N = layer.n_rows
    torch.manual_seed(33)
    a_fp32 = (torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    a_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    # Reference: dequant layer to fp32 W [N, K], compute (A @ W.T) * a_scale[:, None]
    W = dequantize_ml8_layer(layer)                    # [N, K]
    C_ref = (a_fp8.to(torch.float32) @ W.T) * a_scale[:, None]

    # Kernel path
    C_kernel = ml8_gemm(a_fp8, layer, a_scale=a_scale, out_dtype=torch.bfloat16)

    C_ref_bf16 = C_ref.to(torch.bfloat16).to(torch.float32)
    C_kernel_fp32 = C_kernel.to(torch.float32)
    diff = (C_kernel_fp32 - C_ref_bf16).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    print(f"  M={M}, N={N}, K={K}")
    print(f"  max_err = {max_err:.4g}")
    print(f"  rms_err = {rms_err:.4g}")

    assert max_err < 1e-2, f"max_err {max_err:.4g} exceeds 1e-2"
    print("  ✓ ml8_gemm matches dequantize+matmul reference within bf16 tolerance")


def test_ml8_gemm_multi_kgroup_via_runtime():
    """End-to-end multi-K-group through the runtime API (not direct kernel call)."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=32, n_cols=256, n_centroids=16, group_size=64, seed=44)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "multi.ml8"
        write_synth_layer_to_disk(blob, path)
        layer = load_ml8_layer(path, device=device)

    M = 32
    N = layer.n_rows
    K = layer.n_cols
    torch.manual_seed(55)
    a_fp8 = ((torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)).to(torch.float8_e4m3fn)
    a_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    W = dequantize_ml8_layer(layer)
    C_ref = (a_fp8.to(torch.float32) @ W.T) * a_scale[:, None]
    C_kernel = ml8_gemm(a_fp8, layer, a_scale=a_scale)

    diff = (C_kernel.to(torch.float32) - C_ref.to(torch.bfloat16).to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  M={M}, N={N}, K={K}, n_groups_k={layer.n_groups_k}")
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 1e-2
    print("  ✓ multi-K-group end-to-end via runtime API matches reference")


def test_ml8_gemm_no_a_scale():
    """`ml8_gemm` with a_scale=None (defaults to ones) — common path when
    activations are already pre-scaled upstream."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=16, n_cols=64, n_centroids=16, group_size=64, seed=77)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "synth.ml8"
        write_synth_layer_to_disk(blob, path)
        layer = load_ml8_layer(path, device=device)

    torch.manual_seed(88)
    a_fp8 = ((torch.randn(16, 64, device=device) * 0.3).clamp(-1.5, 1.5)).to(torch.float8_e4m3fn)

    W = dequantize_ml8_layer(layer)
    C_ref = a_fp8.to(torch.float32) @ W.T  # no per-row scale
    C_kernel = ml8_gemm(a_fp8, layer)       # a_scale defaults to ones

    diff = (C_kernel.to(torch.float32) - C_ref.to(torch.bfloat16).to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 1e-2
    print("  ✓ ml8_gemm with default a_scale=ones matches reference")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print()

    print("# test_load_roundtrip_dequant_matches_blob")
    test_load_roundtrip_dequant_matches_blob()
    print()

    print("# test_ml8_gemm_matches_dequant_matmul (single K-group, full pipeline)")
    test_ml8_gemm_matches_dequant_matmul()
    print()

    print("# test_ml8_gemm_multi_kgroup_via_runtime (4 K-iters, full pipeline)")
    test_ml8_gemm_multi_kgroup_via_runtime()
    print()

    print("# test_ml8_gemm_no_a_scale (default scale=ones path)")
    test_ml8_gemm_no_a_scale()

    print()
    print("=== PASS: ml8_runtime full pipeline verified end-to-end ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
