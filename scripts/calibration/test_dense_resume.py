#!/usr/bin/env python3
"""Tests for dense-path checkpoint/resume feature.

Tasks covered:
  Task 1 — reconstruct_weight_from_blob fidelity (no rotation/AWQ, then with both)
  Task 2 — dense_completed_prefix contiguous-prefix logic
  Task 3 — Dense resume reload integration (toy nn.Sequential, CPU-only)
  Task 4 — Consistency: MoE skip path unchanged, --no-resume help text
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:           # tests are also runnable via the __main__ block
    pytest = None
import torch
import torch.nn as nn

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Imports resolved lazily in each test so import errors surface cleanly.
# ---------------------------------------------------------------------------

def _import_ml8_io():
    from ml8_io import reconstruct_weight_from_blob
    return reconstruct_weight_from_blob


def _import_prefix_fn():
    from calibrate_ml8_paged import dense_completed_prefix
    return dense_completed_prefix


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 — reconstruct_weight_from_blob fidelity
# ═══════════════════════════════════════════════════════════════════════════════

def _make_spd_hessian(k: int, seed: int = 42) -> torch.Tensor:
    """Random symmetric positive-definite Hessian of shape [k, k]."""
    torch.manual_seed(seed)
    A = torch.randn(k, k, dtype=torch.float32)
    H = A @ A.T + torch.eye(k) * k * 0.1   # positive definite
    return H


def test_reconstruct_weight_from_blob_plain():
    """Task 1 Step 1+3: no rotation, no AWQ — reconstruct must match out['Q'][0].

    We run batched_gptq_quantize on a tiny weight, save a blob exactly as the
    dense writer does, then assert reconstruct_weight_from_blob(blob) ≈ out['Q'][0].
    """
    from batched_gptq import batched_gptq_quantize

    reconstruct_weight_from_blob = _import_ml8_io()

    N, K, n_centroids, group_size = 32, 64, 16, 32
    torch.manual_seed(7)
    W = torch.randn(N, K, dtype=torch.float32)
    H = _make_spd_hessian(K, seed=7)

    out = batched_gptq_quantize(
        W_stack=W.unsqueeze(0),
        H_stack=H.unsqueeze(0),
        n_centroids=n_centroids,
        group_size=group_size,
        snap_centroids="e4m3",
        act_order=True,
        heavy_rounds=0,
    )

    # Build blob exactly as the dense write path does (calibrate_ml8_paged.py ~1463)
    blob = {
        "name": "test.linear",
        "shape": [N, K],
        "group_size": group_size,
        "n_centroids": n_centroids,
        "indices":             out["indices"][0].clone().contiguous(),
        "centroids_per_group": out["centroids_per_group"][0].clone().contiguous(),
        "scale_per_group":     out["scale_per_group"][0].clone().contiguous(),
        "mse":      float(out["mse"][0].item()),
        "w_snr_db": float(out["w_snr_db"][0].item()),
        "y_snr_db": float(out["y_snr_db"][0].item()),
        "rel_err":  float(out["rel_err"][0].item()),
    }

    reconstructed = reconstruct_weight_from_blob(blob)
    Q0 = out["Q"][0]

    # No rotation/AWQ, so reconstruct_weight_from_blob == dequant == Q directly.
    assert reconstructed.shape == Q0.shape, \
        f"shape mismatch: {reconstructed.shape} vs {Q0.shape}"
    assert torch.allclose(reconstructed.float(), Q0.float(), atol=1e-4), \
        f"plain reconstruction mismatch; max abs diff = " \
        f"{(reconstructed.float() - Q0.float()).abs().max().item():.5e}"
    print("PASS test_reconstruct_weight_from_blob_plain")


def test_reconstruct_weight_from_blob_with_rotation_and_awq():
    """Task 1 Step 4: rotation + AWQ — reconstruct must match what live loop writes to weight.data.

    Replicates calibrate_ml8_paged.py lines ~1392–1459:
      1. apply AWQ to weight
      2. rotate Hessian and weight
      3. batched_gptq_quantize → out['Q'][0]  (= weight_override after quant)
      4. rotation.inverse(weight_override)
      5. absorb_awq_in_reconstruction(weight_override, awq_s)
      → this is what ends up in layer.weight.data / weight_override

    Then we build the blob (with rotation_blob and awq_blob) and assert
    reconstruct_weight_from_blob(blob) matches that final value.
    """
    from batched_gptq import batched_gptq_quantize
    from kronecker_rotation import KroneckerRotation, random_orthogonal, rotate_hessian
    from awq import apply_awq_to_weight, absorb_awq_in_reconstruction

    reconstruct_weight_from_blob = _import_ml8_io()

    # Tiny dimensions that satisfy KroneckerRotation: in_features = a * b, b=power-of-2
    N, a, b = 16, 4, 8   # K = 32
    K = a * b
    group_size = 8
    n_centroids = 16
    dtype = torch.float32

    torch.manual_seed(13)
    W_orig = torch.randn(N, K, dtype=dtype)
    H = _make_spd_hessian(K, seed=13)

    # ── Simulate live AWQ ──────────────────────────────────────────────────
    awq_alpha = 0.5
    # In the live code: sum_abs / n_tok → mean_abs; we fake a small calibration.
    mean_abs = (torch.rand(K) + 0.5).clamp_min(1e-8)
    awq_s = mean_abs.pow(awq_alpha)
    awq_blob = {"kind": "mean", "alpha": awq_alpha, "s": awq_s.cpu()}

    W_awq = apply_awq_to_weight(W_orig.float(), awq_s)   # W / s per col
    H_awq = H * awq_s.unsqueeze(0) * awq_s.unsqueeze(1)

    # ── Simulate live rotation ─────────────────────────────────────────────
    rotation = KroneckerRotation(h_a=random_orthogonal(a, seed=99), b_dim=b)
    rotation_blob = rotation.to_dict()
    rotation_blob["seed"] = 99

    H_rot = rotate_hessian(H_awq, rotation)
    W_rot = rotation.forward(W_awq.float())

    # ── Quantize ─────────────────────────────────────────────────────────
    Wr = W_rot.unsqueeze(0)     # [1, N, K]
    Hr = H_rot.unsqueeze(0)    # [1, K, K]
    out = batched_gptq_quantize(
        W_stack=Wr, H_stack=Hr,
        n_centroids=n_centroids, group_size=group_size,
        snap_centroids="e4m3", act_order=False, heavy_rounds=0,
    )

    weight_override = out["Q"][0].to(dtype)   # = layer.weight_override after quant

    # ── Inverse rotation + absorb AWQ (mirrors live lines ~1434-1439) ─────
    weight_override = rotation.inverse(weight_override.float()).to(dtype)
    weight_override = absorb_awq_in_reconstruction(
        weight_override.float(), awq_s.to(weight_override.device)
    ).to(dtype)
    # This is exactly what ends up in layer.weight.data (resident) or weight_override (paged).
    live_weight = weight_override

    # ── Build blob (mirrors live lines ~1463-1479) ─────────────────────
    blob = {
        "name": "test.rot_awq_linear",
        "shape": [N, K],
        "group_size": group_size,
        "n_centroids": n_centroids,
        "indices":             out["indices"][0].clone().contiguous(),
        "centroids_per_group": out["centroids_per_group"][0].clone().contiguous(),
        "scale_per_group":     out["scale_per_group"][0].clone().contiguous(),
        "mse":      float(out["mse"][0].item()),
        "w_snr_db": float(out["w_snr_db"][0].item()),
        "y_snr_db": float(out["y_snr_db"][0].item()),
        "rel_err":  float(out["rel_err"][0].item()),
        "rotation": rotation_blob,
        "awq":      awq_blob,
    }

    reconstructed = reconstruct_weight_from_blob(blob)

    assert reconstructed.shape == live_weight.shape, \
        f"shape mismatch: {reconstructed.shape} vs {live_weight.shape}"
    # atol=1e-4: fp32 chain, should be bit-exact; give small tolerance for dtype casts.
    assert torch.allclose(reconstructed.float(), live_weight.float(), atol=1e-4), \
        f"rotation+AWQ reconstruction mismatch; max abs diff = " \
        f"{(reconstructed.float() - live_weight.float()).abs().max().item():.5e}"
    print("PASS test_reconstruct_weight_from_blob_with_rotation_and_awq")


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 — dense_completed_prefix
# ═══════════════════════════════════════════════════════════════════════════════

def _blob_path(output_dir: Path, name: str) -> Path:
    """Same naming as the dense writer in calibrate_ml8_paged.py."""
    return output_dir / f"{name.replace('.', '_').replace('/', '_')}.pt"


def test_dense_completed_prefix_all_present():
    """All blobs exist → prefix == len(names)."""
    dense_completed_prefix = _import_prefix_fn()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        names = ["model.layers.0.mlp.gate_proj",
                 "model.layers.0.mlp.up_proj",
                 "model.layers.0.mlp.down_proj"]
        for n in names:
            _blob_path(d, n).touch()
        result = dense_completed_prefix(names, d)
    assert result == 3, f"expected 3, got {result}"
    print("PASS test_dense_completed_prefix_all_present")


def test_dense_completed_prefix_gap_in_middle():
    """a, b, (gap), d present → prefix == 2 (stop at first missing)."""
    dense_completed_prefix = _import_prefix_fn()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        names = ["a.gate_proj", "b.up_proj", "c.down_proj", "d.gate_proj"]
        # Present: a, b, d (gap at c)
        _blob_path(d, names[0]).touch()
        _blob_path(d, names[1]).touch()
        _blob_path(d, names[3]).touch()
        result = dense_completed_prefix(names, d)
    assert result == 2, f"expected 2 (gap at index 2), got {result}"
    print("PASS test_dense_completed_prefix_gap_in_middle")


def test_dense_completed_prefix_none_present():
    """No blobs → prefix == 0."""
    dense_completed_prefix = _import_prefix_fn()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        names = ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]
        result = dense_completed_prefix(names, d)
    assert result == 0, f"expected 0, got {result}"
    print("PASS test_dense_completed_prefix_none_present")


def test_dense_completed_prefix_only_first():
    """Only the first blob exists → prefix == 1."""
    dense_completed_prefix = _import_prefix_fn()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        names = ["a.gate_proj", "b.up_proj", "c.down_proj"]
        _blob_path(d, names[0]).touch()
        result = dense_completed_prefix(names, d)
    assert result == 1, f"expected 1, got {result}"
    print("PASS test_dense_completed_prefix_only_first")


def test_dense_completed_prefix_stale_blobs_after_gap_discarded():
    """Blobs after the first gap must be ignored (stale). a, (gap), c, d → prefix=1."""
    dense_completed_prefix = _import_prefix_fn()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        names = ["a.gate_proj", "b.up_proj", "c.down_proj", "d.gate_proj"]
        # Present: a, c, d — b is missing
        _blob_path(d, names[0]).touch()
        _blob_path(d, names[2]).touch()
        _blob_path(d, names[3]).touch()
        result = dense_completed_prefix(names, d)
    assert result == 1, f"expected 1 (gap at index 1), got {result}"
    print("PASS test_dense_completed_prefix_stale_blobs_after_gap_discarded")


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3 — Dense resume reload integration (toy model, CPU-only)
#
# Strategy: build a 2-linear Sequential, run a fake dense calibration that
# saves blobs for both linears, then simulate a crash after linear 0 by deleting
# linear 1's blob.  "Resume" must:
#   (a) detect prefix=1
#   (b) reload linear 0's blob into model weights
#   (c) skip linear 0 (no Hessian recompute)
#   (d) recompute linear 1
# Final state: linear 0's weight == what was loaded from blob (matches full run).
# ═══════════════════════════════════════════════════════════════════════════════

def _save_toy_blob(output_dir: Path, name: str, W: torch.Tensor,
                   group_size: int = 8, n_centroids: int = 8) -> dict:
    """Quantize W trivially and save a blob. Returns the blob dict."""
    from batched_gptq import batched_gptq_quantize
    N, K = W.shape
    H = _make_spd_hessian(K, seed=hash(name) % 1000)
    out = batched_gptq_quantize(
        W_stack=W.float().unsqueeze(0),
        H_stack=H.unsqueeze(0),
        n_centroids=n_centroids, group_size=group_size,
        snap_centroids="none", act_order=False, heavy_rounds=0,
    )
    blob = {
        "name": name,
        "shape": [N, K],
        "group_size": group_size,
        "n_centroids": n_centroids,
        "indices":             out["indices"][0].clone().contiguous(),
        "centroids_per_group": out["centroids_per_group"][0].clone().contiguous(),
        "scale_per_group":     out["scale_per_group"][0].clone().contiguous(),
        "mse":      float(out["mse"][0].item()),
        "w_snr_db": float(out["w_snr_db"][0].item()),
        "y_snr_db": float(out["y_snr_db"][0].item()),
        "rel_err":  float(out["rel_err"][0].item()),
    }
    out_path = _blob_path(output_dir, name)
    torch.save(blob, out_path)
    return blob


def test_dense_resume_reload_restores_weight():
    """Task 3 Step 4: resume reloads linear-0 weight identically to a full run.

    Toy setup:
      - 2 nn.Linear modules (no bias), CPU fp32
      - Quantize both, save blobs → "full run" state
      - Reset model weights to original
      - Delete linear 1's blob (simulate crash after linear 0)
      - Call load_dense_prefix_into_model(prefix=1, names, model, output_dir)
      - Assert model.linear0.weight matches the reconstructed blob weight
    """
    from ml8_io import reconstruct_weight_from_blob, load_ml8_layer
    from calibrate_ml8_paged import dense_completed_prefix, load_dense_prefix_into_model

    N, K = 16, 32
    torch.manual_seed(42)
    W0_orig = torch.randn(N, K)
    W1_orig = torch.randn(N, K)

    # Build tiny model
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = nn.Linear(K, N, bias=False)
            self.linear1 = nn.Linear(K, N, bias=False)

        def named_target_linears(self):
            return [("linear0", self.linear0), ("linear1", self.linear1)]

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        model = ToyModel()
        model.linear0.weight.data.copy_(W0_orig)
        model.linear1.weight.data.copy_(W1_orig)

        names = ["linear0", "linear1"]

        # Save blobs for both (simulate full run)
        blob0 = _save_toy_blob(d, names[0], W0_orig)
        blob1 = _save_toy_blob(d, names[1], W1_orig)

        # Expected weight for linear0 after reconstruction
        expected_W0 = reconstruct_weight_from_blob(blob0).float()

        # Simulate crash after linear 0: delete linear1's blob
        _blob_path(d, names[1]).unlink()

        # Reset model weights (pretend we just loaded fresh)
        model.linear0.weight.data.copy_(W0_orig)
        model.linear1.weight.data.copy_(W1_orig)

        # Dense prefix scan
        prefix = dense_completed_prefix(names, d)
        assert prefix == 1, f"expected prefix=1, got {prefix}"

        # Load prefix back into model
        result_metrics = load_dense_prefix_into_model(
            prefix_count=prefix,
            target_names=names,
            model=model,
            output_dir=d,
            resident=True,   # copies into weight.data
        )

        # linear0 weight must match reconstruction of saved blob
        actual_W0 = model.linear0.weight.data.float()
        diff = (actual_W0 - expected_W0).abs().max().item()
        assert diff < 1e-4, \
            f"resume: linear0 weight mismatch after reload; max abs diff = {diff:.5e}"

        # linear1 was NOT in the prefix — its weight must remain the original
        diff1 = (model.linear1.weight.data.float() - W1_orig.float()).abs().max().item()
        assert diff1 < 1e-6, \
            f"linear1 weight should be untouched, but diff = {diff1:.5e}"

        # Metrics list should have 1 entry
        assert len(result_metrics) == 1, \
            f"expected 1 metric entry for prefix=1, got {len(result_metrics)}"
        assert result_metrics[0]["name"] == "linear0"

    print("PASS test_dense_resume_reload_restores_weight")


def test_dense_resume_zero_prefix_noop():
    """load_dense_prefix_into_model with prefix=0 must be a no-op."""
    from calibrate_ml8_paged import load_dense_prefix_into_model

    N, K = 8, 16
    torch.manual_seed(5)
    W0 = torch.randn(N, K)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = nn.Linear(K, N, bias=False)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        model = ToyModel()
        model.linear0.weight.data.copy_(W0)

        names = ["linear0"]
        metrics = load_dense_prefix_into_model(
            prefix_count=0,
            target_names=names,
            model=model,
            output_dir=d,
            resident=True,
        )
        assert metrics == [], f"expected empty metrics for prefix=0, got {metrics}"
        diff = (model.linear0.weight.data.float() - W0.float()).abs().max().item()
        assert diff < 1e-6, f"weight should be untouched for prefix=0, diff={diff:.5e}"

    print("PASS test_dense_resume_zero_prefix_noop")


def test_dense_resume_paged_override_on_device():
    """Paged branch (resident=False) must place weight_override on the requested
    device. Regression for the bug where W (reconstructed from a blob loaded with
    map_location='cpu') was left on CPU — so on resume the NEXT layer's Hessian
    forward pushed GPU activations through a CPU weight_override → 'mat2 is on cpu'.
    The resident branch always moved to device; the paged branch forgot.

    GPU-free check: device='meta' is the sentinel. cpu (the bug) != meta (fixed),
    so this distinguishes the regression without needing a GPU."""
    from calibrate_ml8_paged import dense_completed_prefix, load_dense_prefix_into_model

    N, K = 16, 32
    torch.manual_seed(7)
    W0 = torch.randn(N, K)

    class ToyPaged(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear0 = nn.Linear(K, N, bias=False)
            self.linear0.weight_override = None   # mimic PagedLinear's override slot

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        model = ToyPaged()
        names = ["linear0"]
        _save_toy_blob(d, names[0], W0)

        prefix = dense_completed_prefix(names, d)
        assert prefix == 1, f"expected prefix=1, got {prefix}"

        load_dense_prefix_into_model(
            prefix_count=prefix, target_names=names, model=model,
            output_dir=d, resident=False, dtype=torch.float32, device="meta")

        wo = model.linear0.weight_override
        assert wo is not None, "paged branch did not set weight_override"
        assert wo.device.type == "meta", (
            f"weight_override landed on {wo.device}, expected requested device "
            f"'meta' — the bug leaves it on cpu (no device move in the paged branch)")

    print("PASS test_dense_resume_paged_override_on_device")


# ═══════════════════════════════════════════════════════════════════════════════
# Task 4 — Consistency checks
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_resume_help_text_mentions_dense_and_moe():
    """Task 4 Step 2: --no-resume help text must mention both dense and MoE."""
    import argparse
    # We can't run main() but we can import the module and inspect its parser.
    # The parser is built inside main(); parse_known_args with --help would sys.exit.
    # Instead: read the source text.
    src = Path(_HERE / "calibrate_ml8_paged.py").read_text()
    # Look for the --no-resume help string and check it mentions both strategies.
    import re
    m = re.search(r'add_argument\(.*?--no-resume.*?help=.*?"(.*?)"', src, re.DOTALL)
    if m is None:
        m = re.search(r"add_argument\(.*?'--no-resume'.*?help=.*?'(.*?)'", src, re.DOTALL)
    if m is None:
        # Try multi-line: find the help= value near --no-resume
        idx = src.find("--no-resume")
        assert idx != -1, "--no-resume argument not found in source"
        snippet = src[idx:idx+400]
        assert "dense" in snippet.lower(), \
            f"--no-resume help does not mention 'dense'. Snippet:\n{snippet}"
        assert "moe" in snippet.lower() or "MoE" in snippet, \
            f"--no-resume help does not mention 'MoE'. Snippet:\n{snippet}"
    else:
        help_text = m.group(1).lower()
        assert "dense" in help_text, f"--no-resume help missing 'dense': {help_text}"
        assert "moe" in help_text, f"--no-resume help missing 'moe': {help_text}"
    print("PASS test_no_resume_help_text_mentions_dense_and_moe")


def test_ml8_readme_documents_resume():
    """Task 4 Step 3: ML8_README.md must document resume behavior and contiguous prefix rule."""
    readme = Path(_HERE / "ML8_README.md")
    assert readme.exists(), f"ML8_README.md not found at {readme}"
    text = readme.read_text().lower()
    assert "resume" in text, "ML8_README.md does not mention 'resume'"
    assert "contiguous" in text or "prefix" in text, \
        "ML8_README.md does not mention 'contiguous' or 'prefix' for dense resume rule"
    print("PASS test_ml8_readme_documents_resume")


if __name__ == "__main__":
    test_reconstruct_weight_from_blob_plain()
    test_reconstruct_weight_from_blob_with_rotation_and_awq()
    test_dense_completed_prefix_all_present()
    test_dense_completed_prefix_gap_in_middle()
    test_dense_completed_prefix_none_present()
    test_dense_completed_prefix_only_first()
    test_dense_completed_prefix_stale_blobs_after_gap_discarded()
    test_dense_resume_reload_restores_weight()
    test_dense_resume_zero_prefix_noop()
    test_no_resume_help_text_mentions_dense_and_moe()
    test_ml8_readme_documents_resume()
    print("\nALL TESTS PASSED")
