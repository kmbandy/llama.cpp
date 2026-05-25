#!/usr/bin/env python3
"""Tests for wp_torch (MAD-238 iteration 3): torch tensor wrapping over VRAM pointers.

Most tests need a real GGUF + GPU. Skip behavior with WP_TEST_SKIP_GPU=1.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_native
from wp_torch import ensure_as_torch, gguf_type_to_torch


def test_gguf_type_to_torch_known_types():
    """Maps standard ggml types to torch dtypes."""
    # Build minimal stub objects mimicking gguf-py's tensor_type
    class _StubType:
        def __init__(self, name): self.name = name
    assert gguf_type_to_torch(_StubType("F16")) == torch.float16
    assert gguf_type_to_torch(_StubType("F32")) == torch.float32
    assert gguf_type_to_torch(_StubType("BF16")) == torch.bfloat16
    assert gguf_type_to_torch(_StubType("I8")) == torch.int8
    print("  PASS test_gguf_type_to_torch_known_types")


def test_gguf_type_to_torch_rejects_quantized():
    """Quantized ggml types (Q4_K etc) need C++-side dequant; reject explicitly."""
    class _StubType:
        def __init__(self, name): self.name = name
    for unsupported in ("Q4_K", "Q6_K", "Q8_0", "MXFP4"):
        try:
            gguf_type_to_torch(_StubType(unsupported))
        except NotImplementedError:
            continue
        assert False, f"expected NotImplementedError for {unsupported!r}"
    print("  PASS test_gguf_type_to_torch_rejects_quantized")


def test_ensure_as_torch_round_trip_on_real_gguf():
    """End-to-end: init pager on Qwen3.5-4B f16, ensure_as_torch one tensor,
    compare to the same tensor read directly via gguf-py.

    GATED: needs GGUF + free GPU.
    """
    gguf_path = os.path.expanduser("~/models/Qwen3.5-4B-f16.gguf")
    if not os.path.exists(gguf_path):
        print("  SKIP test_ensure_as_torch_round_trip_on_real_gguf (no Qwen3.5-4B-f16.gguf)")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib
    import numpy as np

    reader = gguf_lib.GGUFReader(gguf_path)
    target_name = "blk.0.ffn_gate.weight"
    target_tensor = next((t for t in reader.tensors if t.name == target_name), None)
    assert target_tensor is not None

    # Reference data straight from gguf-py mmap (numpy)
    ref = np.array(target_tensor.data).copy()

    # Populate pager catalog
    p = wp_native.WeightPager()
    target_idx = -1
    for t in reader.tensors:
        idx = p.add_page(t.name, 0, int(t.data_offset), int(t.n_bytes))
        if t.name == target_name:
            target_idx = idx
    assert target_idx >= 0

    cfg = wp_native.Config()
    cfg.n_slots = 4   # ~5 GB pool, safe alongside Cell E
    cfg.prefetch_depth = 2
    cfg.prefer_async_io = False
    assert p.init_for_device(cfg, 0, [gguf_path])

    # Get torch tensor via the wrapper
    torch_dtype = gguf_type_to_torch(target_tensor.tensor_type)
    # gguf-py reports shape in ne-natural order; torch wants numpy-natural (reverse)
    torch_shape = tuple(reversed([int(d) for d in target_tensor.shape]))
    wrapped = ensure_as_torch(p, target_idx, torch_shape, torch_dtype, device_idx=0)

    assert wrapped.device.type == "cuda"
    assert wrapped.dtype == torch_dtype
    assert tuple(wrapped.shape) == torch_shape

    # Compare to reference data: bring wrapped to CPU and compare element-wise.
    wrapped_cpu = wrapped.cpu().numpy()
    # ref's shape comes from gguf-py with numpy-natural shape (different from t.shape).
    # Use shape match as the alignment, not transpose.
    assert wrapped_cpu.shape == ref.shape, f"shape {wrapped_cpu.shape} vs ref {ref.shape}"
    max_diff = float(np.abs(wrapped_cpu.astype(np.float32) - ref.astype(np.float32)).max())
    assert max_diff == 0.0, f"wrapped tensor doesn't match GGUF source: max diff {max_diff:.3e}"
    print(f"  PASS test_ensure_as_torch_round_trip_on_real_gguf (shape={torch_shape}, max diff 0.0)")

    p.shutdown()


if __name__ == "__main__":
    test_gguf_type_to_torch_known_types()
    test_gguf_type_to_torch_rejects_quantized()
    if os.environ.get("WP_TEST_SKIP_GPU"):
        print("  SKIP test_ensure_as_torch_round_trip_on_real_gguf (WP_TEST_SKIP_GPU)")
    else:
        test_ensure_as_torch_round_trip_on_real_gguf()
    print("\nALL TESTS PASSED")
