#!/usr/bin/env python3
"""MAD-223 Phase D.1 wiring test — Ml8Linear + ml8_layer_from_blob.

Validates the `--use-ml8-kernel` overlay path without requiring a full
HuggingFace model or real Cell C calibration artifact:

  1. Synthesize an ml8 blob (calibration-side format)
  2. Build an Ml8Linear via ml8_layer_from_blob + the Ml8Linear wrapper
  3. Run forward, compare to standard nn.Linear with the dequantized
     weight (same math, different code path)

This catches:
  - in-memory blob → Ml8Layer conversion (no file round-trip)
  - Per-row max-abs activation quantization at forward time
  - Ml8Linear handling of batch dim flattening + restore
  - Bias handling
  - Output dtype cast back to input dtype

Does NOT test (Phase D proper):
  - Real HF model overlay loop
  - End-to-end PPL on Qwen3.5-4B
  - Rotation / AWQ blobs (Phase D.2)

Usage:
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \\
    /home/kmbandy/venvs/agents/bin/python3 tests/test_ml8_linear_overlay.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from ml8_runtime import (  # noqa: E402
    Ml8Layer,
    Ml8Linear,
    ml8_layer_from_blob,
    ml8_linear_from_blob,
    dequantize_ml8_layer,
)
from kronecker_rotation import KroneckerRotation, factor_for_dim, random_orthogonal  # noqa: E402
from test_ml8_to_packed import synth_blob  # noqa: E402


def _make_rotation_for(in_features: int, seed: int = 0) -> KroneckerRotation:
    """Build a KroneckerRotation Q = H_a ⊗ H_b for the given in_features."""
    a_dim, b_dim = factor_for_dim(in_features, max_b=in_features)
    h_a = random_orthogonal(a_dim, seed=seed)
    return KroneckerRotation(h_a=h_a, b_dim=b_dim)


def _add_rotation_to_blob(blob: dict, rotation: KroneckerRotation) -> dict:
    """Attach a rotation field to a synth blob (does NOT re-quantize W)."""
    blob = dict(blob)  # shallow copy
    blob["rotation"] = rotation.to_dict()
    return blob


def _add_awq_to_blob(blob: dict, awq_scale: torch.Tensor) -> dict:
    """Attach an AWQ field to a synth blob."""
    blob = dict(blob)
    blob["awq"] = {"kind": "per_channel", "s": awq_scale.detach().cpu()}
    return blob


def test_ml8_layer_from_blob_roundtrip():
    """`ml8_layer_from_blob` produces a kernel-friendly Ml8Layer whose
    `dequantize_ml8_layer` matches `ml8_io.reconstruct_weight`."""
    blob = synth_blob(n_rows=32, n_cols=128, n_centroids=16, group_size=64, seed=101)

    layer = ml8_layer_from_blob(blob, device="cpu")
    print(f"  {layer}")

    from ml8_io import reconstruct_weight  # noqa: E402
    W_ref = reconstruct_weight(blob)
    W_runtime = dequantize_ml8_layer(layer)

    max_diff = (W_ref - W_runtime).abs().max().item()
    assert max_diff < 1e-6, f"max_diff {max_diff}"
    print(f"  ✓ in-memory conversion: dequant matches ml8_io.reconstruct_weight (max_diff={max_diff:.2e})")


def test_ml8_linear_forward_matches_reference_no_bias():
    """`Ml8Linear(x)` ≈ `x @ dequant(layer).T` within fp8 quant noise."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=16, n_cols=64, n_centroids=16, group_size=64, seed=202)

    layer = ml8_layer_from_blob(blob, device=device)
    ml8_lin = Ml8Linear(layer, bias=None, out_dtype=torch.bfloat16).to(device)

    # Build the reference nn.Linear from the dequantized weight
    W = dequantize_ml8_layer(layer)  # [N, K]
    ref_lin = nn.Linear(layer.n_cols, layer.n_rows, bias=False).to(device).to(torch.bfloat16)
    ref_lin.weight.data.copy_(W.to(torch.bfloat16))

    # Forward both with the same input
    M = 16
    torch.manual_seed(303)
    x = (torch.randn(M, layer.n_cols, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)

    y_ml8 = ml8_lin(x)
    y_ref = ref_lin(x)

    diff = (y_ml8.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    print(f"  M={M}, N={layer.n_rows}, K={layer.n_cols}, bias=None")
    print(f"  max_err = {max_err:.4g}")
    print(f"  rms_err = {rms_err:.4g}")
    # Tolerance: fp8 activation quant noise dominates.
    # For per-row max-abs scaling, fp8 noise on activations ~= 1/128 of max
    # → output noise scales as O(sqrt(K) * eps_a * |W|). For K=64, |W|~0.5,
    # eps_a~0.01: expect ~0.04 RMS.
    assert max_err < 0.2, f"max_err {max_err:.4g} too large — likely a wiring bug, not quant noise"
    print(f"  ✓ Ml8Linear matches dense reference within fp8 quant noise")


def test_ml8_linear_with_bias():
    """Bias is added correctly after the GEMM."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=32, n_cols=64, n_centroids=16, group_size=64, seed=404)

    layer = ml8_layer_from_blob(blob, device=device)
    bias = torch.randn(layer.n_rows, device=device, dtype=torch.bfloat16) * 0.1
    ml8_lin = Ml8Linear(layer, bias=bias, out_dtype=torch.bfloat16).to(device)

    W = dequantize_ml8_layer(layer)
    ref_lin = nn.Linear(layer.n_cols, layer.n_rows, bias=True).to(device).to(torch.bfloat16)
    ref_lin.weight.data.copy_(W.to(torch.bfloat16))
    ref_lin.bias.data.copy_(bias)

    x = (torch.randn(16, layer.n_cols, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)
    y_ml8 = ml8_lin(x)
    y_ref = ref_lin(x)

    diff = (y_ml8.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  N={layer.n_rows}, bias∈[{bias.min().item():.4f},{bias.max().item():.4f}]")
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 0.2
    print(f"  ✓ bias added correctly after GEMM")


def test_ml8_linear_batch_dim_flatten():
    """3D input [B, M, K] flattens to [B*M, K] for kernel, then reshapes back."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=16, n_cols=64, n_centroids=16, group_size=64, seed=505)
    layer = ml8_layer_from_blob(blob, device=device)
    ml8_lin = Ml8Linear(layer, bias=None, out_dtype=torch.bfloat16).to(device)

    B, M = 2, 16
    x = (torch.randn(B, M, layer.n_cols, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)
    y = ml8_lin(x)
    expected_shape = (B, M, layer.n_rows)
    assert y.shape == expected_shape, f"shape mismatch: {y.shape} vs {expected_shape}"
    print(f"  input {tuple(x.shape)} → output {tuple(y.shape)} ✓ shape preserved")

    # Compare to reference
    W = dequantize_ml8_layer(layer)
    ref_lin = nn.Linear(layer.n_cols, layer.n_rows, bias=False).to(device).to(torch.bfloat16)
    ref_lin.weight.data.copy_(W.to(torch.bfloat16))
    y_ref = ref_lin(x)
    diff = (y.to(torch.float32) - y_ref.to(torch.float32)).abs().max().item()
    print(f"  max_err vs reference = {diff:.4g}")
    assert diff < 0.2
    print(f"  ✓ batch-dim flatten + restore preserves values")


def test_ml8_linear_dtype_preservation():
    """Output dtype matches input dtype (model's expected dtype)."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=16, n_cols=64, n_centroids=16, group_size=64, seed=606)
    layer = ml8_layer_from_blob(blob, device=device)

    for in_dtype in (torch.float16, torch.bfloat16, torch.float32):
        ml8_lin = Ml8Linear(layer, bias=None, out_dtype=torch.bfloat16).to(device)
        x = torch.randn(16, layer.n_cols, device=device, dtype=in_dtype) * 0.3
        y = ml8_lin(x)
        assert y.dtype == in_dtype, f"input {in_dtype} → output {y.dtype} (expected {in_dtype})"
        print(f"  input dtype={in_dtype}, output dtype={y.dtype} ✓")
    print(f"  ✓ output dtype matches input dtype for fp16/bf16/fp32")


def test_ml8_linear_from_blob_legacy_no_rotation_no_awq():
    """`ml8_linear_from_blob` on a plain blob is bit-equivalent to the
    Ml8Linear(ml8_layer_from_blob(...)) shorthand. Backward-compat check."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=16, n_cols=64, n_centroids=16, group_size=64, seed=707)

    ml8_lin = ml8_linear_from_blob(blob, bias=None, out_dtype=torch.bfloat16, device=device)
    assert ml8_lin.rotation is None
    assert ml8_lin.awq_scale is None

    layer = ml8_layer_from_blob(blob, device=device)
    ml8_lin_manual = Ml8Linear(layer, bias=None, out_dtype=torch.bfloat16).to(device)

    x = (torch.randn(16, 64, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)
    y_factory = ml8_lin(x)
    y_manual  = ml8_lin_manual(x)

    diff = (y_factory.to(torch.float32) - y_manual.to(torch.float32)).abs().max().item()
    assert diff == 0.0, f"factory vs manual diff {diff} should be exact"
    print(f"  ✓ ml8_linear_from_blob on legacy blob matches manual construction exactly")


def test_ml8_linear_with_rotation():
    """Rotation-only: Ml8Linear with rotation produces forward-time rotated activations."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=32, n_cols=64, n_centroids=16, group_size=64, seed=808)
    rotation = _make_rotation_for(in_features=64, seed=42)
    blob_rot = _add_rotation_to_blob(blob, rotation)

    ml8_lin = ml8_linear_from_blob(blob_rot, bias=None, out_dtype=torch.bfloat16, device=device)
    assert ml8_lin.rotation is not None, "rotation should have been wired in"
    assert ml8_lin.awq_scale is None

    # Reference path: dequant the layer, then compute the forward-time rotated
    # activation matmul in fp32 (no fp8 noise on activations).
    layer = ml8_layer_from_blob(blob_rot, device=device)
    W_stored = dequantize_ml8_layer(layer)  # [N, K] — in rotated basis

    torch.manual_seed(909)
    x = (torch.randn(16, 64, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)

    # Reference: x_rot = rotation.forward(x); y_ref = x_rot @ W_stored.T
    x_rot_ref = rotation.forward(x.to(torch.float32))
    y_ref = (x_rot_ref @ W_stored.T).to(torch.bfloat16)

    y_kernel = ml8_lin(x)

    diff = (y_kernel.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 0.2, f"max_err {max_err} too large — likely a wiring bug"
    print(f"  ✓ rotation-only Ml8Linear matches rotated-activation reference")


def test_ml8_linear_with_awq():
    """AWQ-only: Ml8Linear scales activations by per-channel awq_scale."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=32, n_cols=64, n_centroids=16, group_size=64, seed=1010)

    # Construct an AWQ scale with realistic spread
    torch.manual_seed(1111)
    awq_scale = (torch.rand(64) * 1.5 + 0.5)  # ∈ [0.5, 2.0]

    blob_awq = _add_awq_to_blob(blob, awq_scale)

    ml8_lin = ml8_linear_from_blob(blob_awq, bias=None, out_dtype=torch.bfloat16, device=device)
    assert ml8_lin.rotation is None
    assert ml8_lin.awq_scale is not None
    assert torch.allclose(ml8_lin.awq_scale.cpu(), awq_scale, atol=0)

    layer = ml8_layer_from_blob(blob_awq, device=device)
    W_stored = dequantize_ml8_layer(layer)

    x = (torch.randn(16, 64, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)
    x_awq = x.to(torch.float32) * awq_scale.to(device)
    y_ref = (x_awq @ W_stored.T).to(torch.bfloat16)

    y_kernel = ml8_lin(x)

    diff = (y_kernel.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  max_err = {max_err:.4g}  (awq_scale range [{awq_scale.min().item():.3f}, {awq_scale.max().item():.3f}])")
    assert max_err < 0.3, f"max_err {max_err} too large — likely a wiring bug"
    print(f"  ✓ AWQ-only Ml8Linear matches scaled-activation reference")


def test_ml8_linear_with_rotation_and_awq():
    """Both rotation + AWQ. Forward-time order must be AWQ first, then rotation
    (matches calibration's W transform order in reverse)."""
    device = torch.device("cuda")
    blob = synth_blob(n_rows=32, n_cols=64, n_centroids=16, group_size=64, seed=1212)
    rotation = _make_rotation_for(in_features=64, seed=42)
    torch.manual_seed(1313)
    awq_scale = (torch.rand(64) * 1.5 + 0.5)
    blob_full = _add_rotation_to_blob(_add_awq_to_blob(blob, awq_scale), rotation)

    ml8_lin = ml8_linear_from_blob(blob_full, bias=None, out_dtype=torch.bfloat16, device=device)
    assert ml8_lin.rotation is not None
    assert ml8_lin.awq_scale is not None

    layer = ml8_layer_from_blob(blob_full, device=device)
    W_stored = dequantize_ml8_layer(layer)

    x = (torch.randn(16, 64, device=device) * 0.3).clamp(-1.5, 1.5).to(torch.bfloat16)
    # Forward-time order: AWQ rescale, THEN rotation (mirrors calibration's
    # W transform: rotate W, then divide by AWQ scale, with the rotation
    # applied to AWQ'd weight in calibrate_ml8.py)
    x_awq = x.to(torch.float32) * awq_scale.to(device)
    x_rot = rotation.forward(x_awq)
    y_ref = (x_rot @ W_stored.T).to(torch.bfloat16)

    y_kernel = ml8_lin(x)

    diff = (y_kernel.to(torch.float32) - y_ref.to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 0.3, f"max_err {max_err} too large — likely a wiring bug"
    print(f"  ✓ rotation + AWQ Ml8Linear matches full forward-time reference")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print()

    print("# test_ml8_layer_from_blob_roundtrip")
    test_ml8_layer_from_blob_roundtrip()
    print()

    print("# test_ml8_linear_forward_matches_reference_no_bias")
    test_ml8_linear_forward_matches_reference_no_bias()
    print()

    print("# test_ml8_linear_with_bias")
    test_ml8_linear_with_bias()
    print()

    print("# test_ml8_linear_batch_dim_flatten")
    test_ml8_linear_batch_dim_flatten()
    print()

    print("# test_ml8_linear_dtype_preservation")
    test_ml8_linear_dtype_preservation()
    print()

    print("# test_ml8_linear_from_blob_legacy_no_rotation_no_awq")
    test_ml8_linear_from_blob_legacy_no_rotation_no_awq()
    print()

    print("# test_ml8_linear_with_rotation")
    test_ml8_linear_with_rotation()
    print()

    print("# test_ml8_linear_with_awq")
    test_ml8_linear_with_awq()
    print()

    print("# test_ml8_linear_with_rotation_and_awq")
    test_ml8_linear_with_rotation_and_awq()

    print()
    print("=== PASS: Ml8Linear overlay wiring verified (incl. MAD-245 rotation + AWQ) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
