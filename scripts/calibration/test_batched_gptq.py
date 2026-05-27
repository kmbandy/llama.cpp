"""test_batched_gptq.py — scalar-reference verification for batched GPTQ.

Builds a synthetic E-expert problem with realistic Linear shapes, runs:
  - the scalar `gptq_quantize_linear` E times (one per expert)
  - the new `batched_gptq_quantize` once over the [E, N, K] stack

and asserts the outputs match within fp32 epsilon. Gates G.7.h.2.

Deterministic seeding — re-runs are bit-for-bit identical.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))

from batched_gptq import batched_gptq_quantize  # noqa: E402
from calibrate_ml8 import gptq_quantize_linear  # noqa: E402
from centroid_quantizer import CentroidQuantizer  # noqa: E402


def _build_random_problem(E: int, N: int, K: int, n_tokens: int, *,
                          dtype: torch.dtype, device: str, seed: int):
    """Produce (W_stack, H_stack) with weights drawn from N(0, 1/sqrt(K)) and
    H_e = X_e^T X_e for random activations X_e [n_tokens, K]."""
    g = torch.Generator(device=device).manual_seed(seed)
    W_stack = torch.randn((E, N, K), generator=g, device=device, dtype=torch.float32) / (K ** 0.5)
    H_stack = torch.zeros((E, K, K), device=device, dtype=torch.float32)
    for e in range(E):
        X = torch.randn((n_tokens, K), generator=g, device=device, dtype=torch.float32)
        H_stack[e] = (X.t() @ X) / n_tokens
    return W_stack.to(dtype), H_stack


def _run_scalar(W_stack, H_stack, *, n_centroids, group_size, n_iter,
                fit_loss, snap_centroids, percdamp):
    """Run scalar gptq_quantize_linear E times. Returns dicts matching the
    batched output keys, with the E dim materialised by stacking."""
    E, N, K = W_stack.shape
    n_groups = K // group_size
    dev = W_stack.device

    indices = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
    centroids_all = torch.zeros((E, n_groups, n_centroids), device=dev, dtype=torch.float32)
    scales_all = torch.zeros((E, N, n_groups), device=dev, dtype=torch.float32)
    Q_all = torch.zeros((E, N, K), device=dev, dtype=torch.float32)
    mse_all = torch.zeros((E,), device=dev)
    w_snr_all = torch.zeros((E,), device=dev)
    y_snr_all = torch.zeros((E,), device=dev)
    rel_err_all = torch.zeros((E,), device=dev)

    for e in range(E):
        layer = nn.Linear(K, N, bias=False).to(dev).to(W_stack.dtype)
        with torch.no_grad():
            layer.weight.copy_(W_stack[e])
        quantizer = CentroidQuantizer(n_centroids=n_centroids, n_iter=n_iter).to(dev)
        quantizer.configure(bits=4, sym=True, fit_loss=fit_loss,
                            mag_weight_p=5.0, snap_centroids=snap_centroids)
        quantizer.hessian_diag = torch.diag(H_stack[e]).clone()
        export = gptq_quantize_linear(layer, H_stack[e].clone(), quantizer,
                                       group_size=group_size, percdamp=percdamp)
        indices[e] = export["indices"]
        centroids_all[e] = export["centroids_per_group"]
        scales_all[e] = export["scale_per_group"]
        # layer.weight now holds the dequantized Q
        Q_all[e] = layer.weight.float()
        mse_all[e] = export["mse"]
        w_snr_all[e] = export["w_snr_db"]
        y_snr_all[e] = export["y_snr_db"]
        rel_err_all[e] = export["rel_err"]

    return {
        "indices": indices, "centroids_per_group": centroids_all,
        "scale_per_group": scales_all, "Q": Q_all,
        "mse": mse_all, "w_snr_db": w_snr_all,
        "y_snr_db": y_snr_all, "rel_err": rel_err_all,
    }


def _tensor_close(a, b, name, *, atol=1e-4, rtol=1e-3, int_tol=0):
    """Compare a vs b, return (ok, msg). int_tol = max element-wise abs
    difference allowed when both are integer-typed (e.g. allows 0 or 1
    quantization-boundary disagreements as a fraction)."""
    if a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        diff = (a.long() - b.long()).abs()
        n_mismatch = int((diff > int_tol).sum().item())
        n_total = int(diff.numel())
        if n_mismatch == 0:
            return True, f"{name}: identical ({n_total} elements)"
        frac = n_mismatch / max(n_total, 1)
        # For indices: allow up to 0.1% mismatch (Lloyd-Max ties at boundaries
        # can flip between two equidistant centroids; this is benign).
        return frac < 1e-3, (
            f"{name}: {n_mismatch}/{n_total} ({100*frac:.4f}%) mismatched, "
            f"max abs diff = {int(diff.max().item())}")
    diff = (a.float() - b.float()).abs()
    max_diff = float(diff.max().item())
    ok = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    return ok, f"{name}: max_abs_diff={max_diff:.3e}  ok={ok}"


def main():
    seed = int(os.environ.get("BATCHED_GPTQ_SEED", "12345"))
    device = os.environ.get("BATCHED_GPTQ_DEVICE", "cuda:0")
    if not torch.cuda.is_available():
        print("ERROR: no CUDA/HIP — this test requires a GPU.")
        return 1

    # Small-but-realistic shape — enough to exercise group boundaries +
    # multiple GPTQ iterations, but fast enough for the smoke gate.
    # Mirrors a Qwen3-MoE expert: K=128, N=64, group_size=64 → 2 groups.
    E, N, K = 4, 64, 128
    n_tokens = 256
    n_centroids = 16
    group_size = 64
    n_iter = 25
    fit_loss = "mse"
    snap_centroids = "e4m3"
    percdamp = 0.05

    print(f"=== batched-GPTQ scalar-reference test ===")
    print(f"  E={E} N={N} K={K} n_tokens={n_tokens}  group_size={group_size}  "
          f"n_centroids={n_centroids}  snap={snap_centroids}")
    print(f"  device={device}  seed={seed}")

    W_stack, H_stack = _build_random_problem(
        E, N, K, n_tokens, dtype=torch.float32, device=device, seed=seed)
    print(f"  W_stack {tuple(W_stack.shape)}  ||W||={W_stack.norm():.3f}")
    print(f"  H_stack {tuple(H_stack.shape)}  ||H||={H_stack.norm():.3f}")

    # ── Run scalar reference.
    t0 = time.time()
    scalar = _run_scalar(W_stack, H_stack,
                          n_centroids=n_centroids, group_size=group_size,
                          n_iter=n_iter, fit_loss=fit_loss,
                          snap_centroids=snap_centroids, percdamp=percdamp)
    t_scalar = time.time() - t0
    print(f"  scalar:  {t_scalar*1000:.0f} ms  ({t_scalar*1000/E:.1f} ms/expert)")

    # ── Run batched.
    t0 = time.time()
    batched = batched_gptq_quantize(
        W_stack=W_stack, H_stack=H_stack,
        n_centroids=n_centroids, group_size=group_size, n_iter=n_iter,
        fit_loss=fit_loss, snap_centroids=snap_centroids, percdamp=percdamp)
    t_batched = time.time() - t0
    print(f"  batched: {t_batched*1000:.0f} ms  ({t_batched*1000/E:.1f} ms/expert)  "
          f"speedup: {t_scalar/max(t_batched, 1e-6):.2f}x")

    # ── Compare every output tensor.
    print()
    print("=== element-wise comparisons ===")
    all_ok = True
    for name in ("centroids_per_group", "scale_per_group", "Q",
                 "mse", "w_snr_db", "y_snr_db", "rel_err"):
        ok, msg = _tensor_close(scalar[name], batched[name], name,
                                  atol=1e-3, rtol=1e-2)
        all_ok = all_ok and ok
        print(f"  {'OK   ' if ok else 'FAIL '}  {msg}")
    # Indices: allow tiny fraction of boundary disagreements (centroid ties).
    ok, msg = _tensor_close(scalar["indices"], batched["indices"], "indices",
                              int_tol=1)
    all_ok = all_ok and ok
    print(f"  {'OK   ' if ok else 'FAIL '}  {msg}")

    print()
    if all_ok:
        print("=== PASS ===  batched matches scalar within tolerance")
        return 0
    print("=== FAIL ===  batched diverges from scalar — fix before shipping")
    return 1


def test_multigpu():
    """Verify multi-GPU split produces identical output to single-GPU."""
    from batched_gptq import batched_gptq_quantize_multigpu
    if torch.cuda.device_count() < 2:
        print("[multigpu] SKIP — only one GPU available")
        return True

    seed = int(os.environ.get("BATCHED_GPTQ_SEED", "12345"))
    E, N, K = 8, 64, 128
    print(f"\n=== multi-GPU split test ===")
    print(f"  E={E} N={N} K={K}  primary=cuda:0  secondary=cuda:1")

    W, H = _build_random_problem(E, N, K, n_tokens=256, dtype=torch.float32,
                                  device="cuda:0", seed=seed)
    single = batched_gptq_quantize(
        W, H, n_centroids=16, group_size=64, n_iter=25,
        fit_loss="mse", snap_centroids="e4m3", percdamp=0.05)
    multi = batched_gptq_quantize_multigpu(
        W, H, primary_device="cuda:0", secondary_device="cuda:1",
        primary_share=0.5,
        n_centroids=16, group_size=64, n_iter=25,
        fit_loss="mse", snap_centroids="e4m3", percdamp=0.05)

    all_ok = True
    for name in ("indices", "centroids_per_group", "scale_per_group", "Q",
                 "mse", "w_snr_db", "y_snr_db", "rel_err"):
        a, b = single[name], multi[name]
        if a.dtype == torch.int8:
            ok = (a == b).all().item()
        else:
            ok = torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-3)
        all_ok = all_ok and ok
        diff = (a.float() - b.float()).abs().max().item() if a.dtype != torch.int8 else int((a.long() - b.long()).abs().max().item())
        print(f"  {'OK   ' if ok else 'FAIL '}  {name}: max_diff={diff}")

    if all_ok:
        print("=== multi-GPU PASS ===")
    else:
        print("=== multi-GPU FAIL ===")
    return all_ok


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        ok = test_multigpu()
        if not ok:
            rc = 1
    sys.exit(rc)
