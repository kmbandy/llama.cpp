"""Sanity check: row-wise FWHT (as implemented in turbo_fp8_hadamard.cuh)
vs explicit Sylvester-Hadamard right-multiply.

If these don't match to fp32 epsilon, the FWHT swap in
ggml_cuda_op_ml8_apply_rotation (G.6.f) introduces a real error vs the
dense H_b matmul calibration assumed.

Run: python3 scripts/calibration/test_fwht_vs_sylvester.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kronecker_rotation import sylvester  # type: ignore


def python_fwht(x: np.ndarray) -> np.ndarray:
    """Mirrors mt_turbo_fp8_fwht_kernel exactly.

    Per-row in-place butterflies. For each stage s in 0..log2(D)-1:
        partner = tid XOR (1 << s)
        if (tid & (1 << s)) == 0:  smem[tid] = a + b
        else:                       smem[tid] = b - a  (= partner_value - our_value)
    Then divide entire row by sqrt(D).

    x: (n_rows, D) float32
    returns: (n_rows, D) float32
    """
    n_rows, D = x.shape
    assert (D & (D - 1)) == 0 and D >= 2 and D <= 1024, f"D must be pow2 in [2,1024], got {D}"

    out = x.astype(np.float32).copy()  # work in fp32 like the kernel
    stage = 0
    while (1 << stage) < D:
        stride = 1 << stage
        new_out = np.empty_like(out)
        for tid in range(D):
            partner = tid ^ stride
            a = out[:, tid]
            b = out[:, partner]
            if (tid & stride) == 0:
                new_out[:, tid] = a + b
            else:
                new_out[:, tid] = b - a
        out = new_out
        stage += 1

    return out / math.sqrt(D)


def test_one_size(D: int, n_rows: int = 4):
    H = sylvester(D).to(torch.float32).numpy()  # (D, D), orthogonal Sylvester
    # Random inputs
    rng = np.random.default_rng(0xC0FFEE + D)
    x = rng.standard_normal((n_rows, D)).astype(np.float32)

    y_dense = x @ H                # (n_rows, D)
    y_fwht  = python_fwht(x)       # (n_rows, D)

    abs_err = np.abs(y_dense - y_fwht).max()
    rel_err = abs_err / max(np.abs(y_dense).max(), 1e-30)

    # Same vs sign-flipped — Hadamard has 2^D sign-equivalent forms;
    # if FWHT produced -H @ X instead of H @ X, that's also "correct"
    # algorithmically but would not match calibration. Check both.
    abs_err_sign = np.abs(y_dense + y_fwht).max()

    print(f"  D={D:5d} n_rows={n_rows}: "
          f"max|dense - fwht| = {abs_err:.3e}  "
          f"max|dense + fwht| = {abs_err_sign:.3e}  "
          f"rel = {rel_err:.3e}")

    # Print first row of each for tiny D so we can see what's happening
    if D <= 8 and n_rows >= 1:
        print(f"    x[0]      = {x[0]}")
        print(f"    dense[0]  = {y_dense[0]}")
        print(f"    fwht [0]  = {y_fwht[0]}")

    return abs_err, abs_err_sign


def main():
    print("=== FWHT vs Sylvester-Hadamard dense matmul ===")
    for D in [2, 4, 8, 16, 64, 256, 512, 1024]:
        test_one_size(D, n_rows=4)


if __name__ == "__main__":
    main()
