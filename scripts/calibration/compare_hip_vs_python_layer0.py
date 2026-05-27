"""Per-stage bit-equivalence: HIP vs Python ml8 pipeline (G.6.g.C).

Dumps loaded:
  /tmp/ml8_hip_x_in.bin       — pre-rotation activations (input to ml8_apply_rotation)
  /tmp/ml8_hip_x_rotated.bin  — post-rotation activations (input to ml8_mul_mat)
  /tmp/ml8_hip_y_out.bin      — final ml8_mul_mat output (after bf16→fp32)

Tests:
  Stage A: Python rotation on x_in  ↔  HIP x_rotated     (isolates rotation kernel)
  Stage B: Python quant+gemm on HIP x_rotated  ↔  HIP y_out  (isolates quant + gemm)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml8_io import load_ml8_layer as load_blob  # noqa: E402
from ml8_runtime import ml8_layer_from_blob, ml8_gemm  # noqa: E402
from kronecker_rotation import KroneckerRotation  # noqa: E402


PT_PATH = Path("/home/kmbandy/models/cell-e/model_layers_0_mlp_gate_proj.pt")


def read_dump(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = list(struct.unpack(f"<{ndim}q", f.read(8 * ndim)))
        n = 1
        for s in shape: n *= s
        data = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
    np_shape = tuple(reversed(shape))  # GGML ne[0]=innermost → reverse for numpy
    return data.reshape(np_shape)


def stage_diff(a: np.ndarray, b: np.ndarray, label: str):
    if a.shape != b.shape:
        print(f"  {label}: ✗ SHAPE MISMATCH a={a.shape} b={b.shape}")
        return
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    rel = diff / denom
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    cos = (a_flat * b_flat).sum(axis=1) / (
        np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-12)
    print(f"  {label}:")
    print(f"    max|diff|     = {diff.max():.6e}")
    print(f"    mean|diff|    = {diff.mean():.6e}")
    print(f"    max rel diff  = {rel.max():.6e}")
    print(f"    cos(row) mean = {cos.mean():.6f}   min = {cos.min():.6f}")
    flat_idx = diff.argmax()
    r, c = np.unravel_index(flat_idx, diff.shape)
    print(f"    worst [{r},{c}]: HIP={a[r,c]:.6f}  PY={b[r,c]:.6f}")


def main():
    print("=== HIP dump load ===")
    x_in_hip      = read_dump(Path("/tmp/ml8_hip_x_in.bin"))
    x_rotated_hip = read_dump(Path("/tmp/ml8_hip_x_rotated.bin"))
    y_out_hip     = read_dump(Path("/tmp/ml8_hip_y_out.bin"))
    M, K = x_in_hip.shape
    N = y_out_hip.shape[1]
    print(f"  M={M}  K={K}  N={N}")

    blob = load_blob(PT_PATH)
    print(f"\n  blob rotation kind: {blob.get('rotation', {}).get('kind')}")
    device = torch.device("cuda:0")

    # Build the rotation from the blob (same recipe Ml8Linear uses internally)
    rotation_dict = blob["rotation"]
    rotation = KroneckerRotation.from_dict(rotation_dict)
    # rotation.h_a is on CPU by class contract; move to device explicitly.
    rotation.h_a = rotation.h_a.to(device=device, dtype=torch.float32)

    # ──────────────── Stage A: rotation only ────────────────
    print("\n=== Stage A: Python rotation ↔ HIP rotation ===")
    x_in_torch = torch.from_numpy(x_in_hip).contiguous().to(device)
    x_rotated_py = rotation.forward(x_in_torch).float().cpu().numpy()
    stage_diff(x_rotated_hip, x_rotated_py, "rotation output (HIP vs Python dense)")

    # ──────────────── Stage B: quant + gemm starting from HIP's rotated ─────────
    print("\n=== Stage B: Python quant+gemm (input = HIP rotated_x) ↔ HIP final ===")
    # Pad to multiple of 16
    pad_m = ((M + 15) // 16) * 16
    if pad_m != M:
        pad_rows = pad_m - M
        x_rot_padded = np.concatenate([x_rotated_hip, np.zeros((pad_rows, K), dtype=np.float32)], axis=0)
    else:
        x_rot_padded = x_rotated_hip

    layer = ml8_layer_from_blob(blob, device=device)
    x_rot_torch = torch.from_numpy(x_rot_padded).contiguous().to(device)

    # Quant: same math as Ml8Linear.forward
    FP8_MAX = 448.0
    row_max = x_rot_torch.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    a_scale = (row_max / FP8_MAX).squeeze(1).contiguous()
    x_fp8 = (x_rot_torch / a_scale.unsqueeze(1)).to(torch.float8_e4m3fn).contiguous()

    with torch.inference_mode():
        y_bf16 = ml8_gemm(x_fp8, layer, a_scale=a_scale, out_dtype=torch.bfloat16)
    y_py = y_bf16.float().cpu().numpy()[:M]
    stage_diff(y_out_hip, y_py, "final output (HIP vs Python quant+gemm)")


if __name__ == "__main__":
    main()
