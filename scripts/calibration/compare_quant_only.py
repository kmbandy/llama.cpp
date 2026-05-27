"""Isolate fp8 activation quant: HIP ml8_quantize_activations_kernel vs torch.float8_e4m3fn.

Inputs:
  /tmp/ml8_hip_x_prequant.bin — fp32 activations going INTO the quant kernel (post-rotation, padded)
  /tmp/ml8_hip_a_fp8.bin      — fp8 e4m3 bytes emitted by HIP kernel
  /tmp/ml8_hip_a_scale.bin    — per-row fp32 scale emitted by HIP kernel

Python recreates the same Ml8Linear.forward quant logic on the same x_prequant
and compares bytes-of-fp8 + scale element-wise.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch


def read_dump(path: Path, dtype=np.float32):
    with open(path, "rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = list(struct.unpack(f"<{ndim}q", f.read(8 * ndim)))
        n = 1
        for s in shape: n *= s
        item_size = np.dtype(dtype).itemsize
        data = np.frombuffer(f.read(item_size * n), dtype=dtype).copy()
    np_shape = tuple(reversed(shape))
    return data.reshape(np_shape)


def main():
    x      = read_dump(Path("/tmp/ml8_hip_x_prequant.bin"), np.float32)
    fp8_hip = read_dump(Path("/tmp/ml8_hip_a_fp8.bin"), np.uint8)
    sc_hip  = read_dump(Path("/tmp/ml8_hip_a_scale.bin"), np.float32)
    M, K = x.shape
    print(f"x_prequant  shape: {x.shape}")
    print(f"a_fp8       shape: {fp8_hip.shape}")
    print(f"a_scale     shape: {sc_hip.shape}")

    # Python quant: same as Ml8Linear.forward()
    FP8_MAX = 448.0
    x_t = torch.from_numpy(x).to(torch.float32).cuda()
    row_max = x_t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    sc_py_t = (row_max / FP8_MAX).squeeze(1)             # scale per row
    sc_py = sc_py_t.cpu().numpy()
    fp8_py = (x_t / sc_py_t.unsqueeze(1)).to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()

    print(f"\n=== Per-row scale ===")
    diff_sc = np.abs(sc_hip - sc_py)
    for r in range(M):
        flag = "✓" if diff_sc[r] < 1e-6 else "✗"
        print(f"  row {r}: HIP={sc_hip[r]:.6e}  PY={sc_py[r]:.6e}  diff={diff_sc[r]:.2e}  {flag}")

    print(f"\n=== fp8 bytes ===")
    eq = (fp8_hip == fp8_py)
    n_eq = int(eq.sum())
    n_total = fp8_hip.size
    print(f"  match: {n_eq}/{n_total} = {n_eq/n_total*100:.3f}%")
    print(f"  mismatch count: {n_total - n_eq}")

    if n_eq != n_total:
        # Find first few mismatches
        ne_idx = np.where(~eq)
        print(f"  first 10 mismatches:")
        for i in range(min(10, len(ne_idx[0]))):
            r, c = ne_idx[0][i], ne_idx[1][i]
            print(f"    [{r},{c}]: x={x[r,c]:+.6f}  HIP_byte=0x{fp8_hip[r,c]:02x}  PY_byte=0x{fp8_py[r,c]:02x}")

    # Dequantize both to fp32 and diff
    fp8_hip_t = torch.from_numpy(fp8_hip).view(torch.float8_e4m3fn).cuda()
    fp8_py_t  = torch.from_numpy(fp8_py).view(torch.float8_e4m3fn).cuda()
    dq_hip = fp8_hip_t.float().cpu().numpy()
    dq_py  = fp8_py_t.float().cpu().numpy()
    print(f"\n=== Dequantized fp8 → fp32 diff ===")
    diff_dq = np.abs(dq_hip - dq_py)
    print(f"  max|diff|  = {diff_dq.max():.6e}")
    print(f"  mean|diff| = {diff_dq.mean():.6e}")
    print(f"  n_diff > 0 = {(diff_dq > 0).sum()} / {diff_dq.size}")

    # Reconstruct effective activation (fp8 * scale per row) and compare to x
    eff_hip = dq_hip * sc_hip[:, None]
    eff_py  = dq_py  * sc_py[:, None]
    print(f"\n=== Effective activation (fp8 * scale): HIP vs Python vs original ===")
    print(f"  HIP vs Python:   max|diff|={np.abs(eff_hip-eff_py).max():.6e}  mean={np.abs(eff_hip-eff_py).mean():.6e}")
    print(f"  HIP vs original: max|diff|={np.abs(eff_hip-x   ).max():.6e}  mean={np.abs(eff_hip-x   ).mean():.6e}")
    print(f"  PY  vs original: max|diff|={np.abs(eff_py -x   ).max():.6e}  mean={np.abs(eff_py -x   ).mean():.6e}")


if __name__ == "__main__":
    main()
