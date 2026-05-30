#!/usr/bin/env python3
"""Tests for ml8_to_gguf — patching ml8-4 reconstructed tensors into a base f16 GGUF."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ml8_to_gguf import (  # noqa: E402
    hf_to_gguf_name,
    evaluate_coverage,
    pack_scaled_fp8_blocks,
)


def test_hf_to_gguf_name_qwen_mlp():
    """Qwen MLP HF names map to GGUF blk.N.ffn_*.weight format."""
    cases = {
        "model.layers.0.mlp.gate_proj":  "blk.0.ffn_gate.weight",
        "model.layers.5.mlp.up_proj":    "blk.5.ffn_up.weight",
        "model.layers.31.mlp.down_proj": "blk.31.ffn_down.weight",
    }
    for hf, gguf in cases.items():
        got = hf_to_gguf_name(hf)
        assert got == gguf, f"hf_to_gguf_name({hf!r}) = {got!r}, expected {gguf!r}"
    print(f"  PASS test_hf_to_gguf_name_qwen_mlp")


def test_hf_to_gguf_name_rejects_unknown():
    """Names we don't have a mapping for should raise, not silently mangle."""
    for unknown in [
        "model.layers.0.self_attn.q_proj",  # attention proj — not yet supported
        "model.embed_tokens",                # embedding — different naming
        "lm_head",                            # output — different naming
    ]:
        try:
            got = hf_to_gguf_name(unknown)
        except (ValueError, KeyError) as e:
            continue
        assert False, f"hf_to_gguf_name({unknown!r}) silently returned {got!r}, should raise"
    print(f"  PASS test_hf_to_gguf_name_rejects_unknown")


def test_coverage_ffn_only_dense_refuses():
    """The 9B regression: a dense hybrid model where only the FFN is calibrated.
    FFN is a minority of weight params, so coverage is far below threshold and
    the guardrail must flag it (below_threshold=True). Numbers approximate the
    real 9B: ~24% quantized."""
    params_ml8 = 24          # FFN params (stand-in units)
    params_fp8 = 0
    params_bf16 = 76         # attn + SSM + lm_head + embed left bf16
    coverage, below, breakdown = evaluate_coverage(
        params_ml8, params_fp8, params_bf16, min_coverage=0.85)
    assert abs(coverage - 0.24) < 1e-9, f"coverage {coverage} != 0.24"
    assert below is True, "24%-quantized FFN-only dense must be flagged below threshold"
    print("  PASS test_coverage_ffn_only_dense_refuses")


def test_coverage_moe_experts_dominant_passes():
    """A MoE model where the experts (the FFN) ARE the bulk of the weight.
    Even with attn/embed left bf16, coverage clears the threshold."""
    params_ml8 = 94          # experts dominate
    params_fp8 = 0
    params_bf16 = 6          # attn + embed
    coverage, below, breakdown = evaluate_coverage(
        params_ml8, params_fp8, params_bf16, min_coverage=0.85)
    assert abs(coverage - 0.94) < 1e-9, f"coverage {coverage} != 0.94"
    assert below is False, "94%-quantized MoE must clear the threshold"
    print("  PASS test_coverage_moe_experts_dominant_passes")


def test_coverage_boundary_and_empty():
    """Exactly-at-threshold passes (strict <), and a degenerate all-zero model
    reports 0.0 coverage (and is flagged), never divides by zero."""
    cov, below, _ = evaluate_coverage(85, 0, 15, min_coverage=0.85)
    assert abs(cov - 0.85) < 1e-9 and below is False, "exactly at threshold must pass"
    cov0, below0, _ = evaluate_coverage(0, 0, 0, min_coverage=0.85)
    assert cov0 == 0.0 and below0 is True, "empty model: 0.0 coverage, flagged, no div0"
    print("  PASS test_coverage_boundary_and_empty")


# ── New tests for Tasks 6 + 7 ─────────────────────────────────────────────


def test_coverage_credits_fp8_tier():
    """FP8-tier params are credited as quantized; combined coverage clears threshold.

    Scenario: 88% ml8-4-bit, 11% fp8-8-bit, 1% residual bf16 → total 99% quant.
    """
    coverage, below, breakdown = evaluate_coverage(
        params_ml8=88, params_fp8=11, params_passthrough_weight=1, min_coverage=0.85)
    assert abs(coverage - 0.99) < 1e-9, f"coverage {coverage} != 0.99"
    assert below is False, "99% quantized must not be flagged below threshold"
    assert abs(breakdown["ml8"] - 0.88) < 1e-9, f"ml8 fraction {breakdown['ml8']} != 0.88"
    assert abs(breakdown["fp8"] - 0.11) < 1e-9, f"fp8 fraction {breakdown['fp8']} != 0.11"
    assert abs(breakdown["bf16"] - 0.01) < 1e-9, f"bf16 fraction {breakdown['bf16']} != 0.01"
    print("  PASS test_coverage_credits_fp8_tier")


def test_coverage_ffn_only_still_refuses():
    """FFN-only calibration (no FP8) still refuses when params_fp8=0."""
    coverage, below, breakdown = evaluate_coverage(
        params_ml8=24, params_fp8=0, params_passthrough_weight=76, min_coverage=0.85)
    assert abs(coverage - 0.24) < 1e-9, f"coverage {coverage} != 0.24"
    assert below is True, "24% coverage must be flagged below threshold"
    print("  PASS test_coverage_ffn_only_still_refuses")


def test_pack_scaled_fp8_blocks_layout():
    """pack_scaled_fp8_blocks: verify byte count and that scale round-trips correctly."""
    N, K = 4, 64
    n_blocks = K // 32  # = 2

    # Build a simple e4m3 weight (values safe on the fp8 lattice).
    e4m3_f32 = torch.ones(N, K, dtype=torch.float32) * 2.0
    # Build scales: block 0 = 1.0, block 1 = 2.0 for all rows.
    scale_vals = torch.tensor([[1.0, 2.0]] * N, dtype=torch.float16)

    packed = pack_scaled_fp8_blocks(e4m3_f32, scale_vals)

    # 1. Total byte count.
    expected_bytes = N * n_blocks * 34
    assert packed.shape == (N, n_blocks * 34), (
        f"packed shape {packed.shape} != ({N}, {n_blocks * 34})")
    assert packed.nbytes == expected_bytes, (
        f"byte count {packed.nbytes} != {expected_bytes}")

    # 2. Scale of block 0, row 0: first 2 bytes must decode to 1.0 fp16.
    scale_bytes_blk0 = packed[0, 0:2].tobytes()
    recovered_scale = np.frombuffer(scale_bytes_blk0, dtype=np.float16)[0]
    assert float(recovered_scale) == 1.0, (
        f"block-0 scale decoded to {recovered_scale}, expected 1.0")

    # 3. Scale of block 1, row 0: bytes at offset 34 (= 1 * 34) must decode to 2.0.
    scale_bytes_blk1 = packed[0, 34:36].tobytes()
    recovered_scale1 = np.frombuffer(scale_bytes_blk1, dtype=np.float16)[0]
    assert float(recovered_scale1) == 2.0, (
        f"block-1 scale decoded to {recovered_scale1}, expected 2.0")

    # 4. e4m3 bytes follow the scale (bytes 2..33 of block 0).
    qs_block0 = packed[0, 2:34]
    assert len(qs_block0) == 32, f"qs byte count {len(qs_block0)} != 32"

    print("  PASS test_pack_scaled_fp8_blocks_layout")


if __name__ == "__main__":
    test_hf_to_gguf_name_qwen_mlp()
    test_hf_to_gguf_name_rejects_unknown()
    test_coverage_ffn_only_dense_refuses()
    test_coverage_moe_experts_dominant_passes()
    test_coverage_boundary_and_empty()
    test_coverage_credits_fp8_tier()
    test_coverage_ffn_only_still_refuses()
    test_pack_scaled_fp8_blocks_layout()
    print("\nALL TESTS PASSED")
