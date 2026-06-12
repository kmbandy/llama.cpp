"""ml8_runtime — Python runtime API for ml8-4 inference.

Provides the user-facing wrappers around the vendored Triton kernels
(`ggml/src/ggml-cuda/aiter-integration/kernels/{gemm_ml8,moe_op_gemm_ml8}.py`):

    layer  = load_ml8_layer("/path/to/blk.0.ffn_gate.weight.ml8", device)
    output = ml8_gemm(activations_fp8, layer, out_dtype=torch.bfloat16)

The kernel itself was validated end-to-end in:
    tests/test_ml8_kernel_stage1_dequant.py — Stage 1 + multi-tile
    tests/test_ml8_kernel_moe.py            — 1-expert MoE

This module is the Python-side glue that Phase D's `reconstruct_model.py
--use-ml8-kernel` flag will consume. The C++ counterpart (mt_ml8_gemm.h,
mt_ml8_moe_gemm.h, Phase C.2) is the eventual llama.cpp inference entry
point; it uses `aiter::Registry::get_or_compile` rather than calling
Python.

Loading and kernel-launch convention (matches kernel expectations):
  - indices_packed: uint8 [K // 2, N]  (K-major; transposed from on-disk [N, K//2])
  - centroids_fp8:  fp8   [n_groups_k, 16]  (as-stored)
  - scales_fp32:    fp32  [n_groups_k, N]  (transposed from on-disk [N, n_groups_k]
                                            for K-major access pattern)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Local helpers
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from ml8_to_packed import (  # noqa: E402
    ML8_HEADER_SIZE,
    ML8_MAGIC,
    ML8_VERSION,
    FLAG_NIBBLE_LO_FIRST,
)

# Kernel modules — sys.path managed by caller (or via the conventional location).
_KERNELS_DIR = (
    _THIS_DIR.parent.parent
    / "ggml/src/ggml-cuda/aiter-integration/kernels"
)
if str(_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELS_DIR))

import gemm_ml8  # noqa: E402


# ─── Data structure ────────────────────────────────────────────────────────


@dataclass
class Ml8Layer:
    """One ml8-quantized Linear layer ready for kernel invocation.

    Tensors are in KERNEL-FRIENDLY layout (transposes already applied vs
    the on-disk .ml8 layout). The kernel expects K-major access for both
    indices and scales.
    """
    n_rows: int                       # N (output dim, "rows" in calibration nomenclature)
    n_cols: int                       # K (input dim, "in_features")
    group_size: int
    n_centroids: int
    indices_packed: torch.Tensor      # uint8 [K // 2, N]  — kernel layout
    centroids_fp8: torch.Tensor       # fp8   [n_groups_k, 16]
    scales_fp32: torch.Tensor         # fp32  [n_groups_k, N]
    nibble_lo_first: bool             # kept for round-trip / debug

    @property
    def n_groups_k(self) -> int:
        return self.n_cols // self.group_size

    def __repr__(self) -> str:
        return (
            f"Ml8Layer(N={self.n_rows}, K={self.n_cols}, "
            f"group_size={self.group_size}, n_centroids={self.n_centroids}, "
            f"n_groups_k={self.n_groups_k})"
        )


# ─── Loading ──────────────────────────────────────────────────────────────


import struct  # noqa: E402

_HEADER_STRUCT = "<IIIIIIII"  # matches ml8_to_packed


def load_ml8_layer(path: str | Path, device: torch.device | str = "cpu") -> Ml8Layer:
    """Load a single-layer .ml8 packed binary and produce a kernel-ready Ml8Layer.

    Reads the on-disk format produced by `scripts/calibration/ml8_to_packed.py`
    (per ML8_WMMA_KERNEL_DESIGN.md Appendix A.2) and applies the transposes
    needed to land in kernel-friendly layout.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ml8 layer file not found: {path}")
    device = torch.device(device)

    raw = path.read_bytes()
    if len(raw) < ML8_HEADER_SIZE:
        raise ValueError(f"{path}: file too short ({len(raw)}) for header")

    magic, version, n_rows, n_cols, group_size, n_centroids, flags, _reserved = (
        struct.unpack(_HEADER_STRUCT, raw[:ML8_HEADER_SIZE])
    )
    if magic != ML8_MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:08x}")
    if version != ML8_VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    nibble_lo_first = bool(flags & FLAG_NIBBLE_LO_FIRST)
    n_groups_k = (n_cols + group_size - 1) // group_size

    # Section sizes (16-byte aligned per pack_layer)
    indices_size = n_rows * (n_cols // 2)
    indices_size_padded = ((indices_size + 15) // 16) * 16
    centroids_size = n_groups_k * n_centroids
    centroids_size_padded = ((centroids_size + 15) // 16) * 16
    scales_size = n_rows * n_groups_k * 4

    off = ML8_HEADER_SIZE
    indices_bytes = raw[off:off + indices_size]
    off += indices_size_padded
    centroids_bytes = raw[off:off + centroids_size]
    off += centroids_size_padded
    scales_bytes = raw[off:off + scales_size]

    # Indices: on-disk [N, K//2] uint8 → kernel layout [K//2, N]
    indices_np = np.frombuffer(indices_bytes, dtype=np.uint8).reshape(n_rows, n_cols // 2)
    indices_packed = torch.from_numpy(indices_np.T.copy()).contiguous().to(device)

    # Centroids: on-disk [n_groups_k, n_centroids] fp8 → kernel layout same
    cent_np = np.frombuffer(centroids_bytes, dtype=np.uint8).reshape(n_groups_k, n_centroids)
    centroids_fp8 = (
        torch.from_numpy(cent_np.copy()).view(torch.float8_e4m3fn).contiguous().to(device)
    )

    # Scales: on-disk [N, n_groups_k] fp32 → kernel layout [n_groups_k, N]
    scales_np = np.frombuffer(scales_bytes, dtype=np.float32).reshape(n_rows, n_groups_k)
    scales_fp32 = torch.from_numpy(scales_np.T.copy()).contiguous().to(device)

    return Ml8Layer(
        n_rows=n_rows,
        n_cols=n_cols,
        group_size=group_size,
        n_centroids=n_centroids,
        indices_packed=indices_packed,
        centroids_fp8=centroids_fp8,
        scales_fp32=scales_fp32,
        nibble_lo_first=nibble_lo_first,
    )


# ─── Kernel invocation ────────────────────────────────────────────────────


def ml8_gemm(
    a_fp8: torch.Tensor,
    layer: Ml8Layer,
    a_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_size_m: int = 16,
    block_size_n: int = 16,
) -> torch.Tensor:
    """Compute C = A @ W via the ml8 LUT kernel (WEIGHT_FORMAT=1).

    Args:
        a_fp8: fp8 e4m3 activations [M, K]. K must equal layer.n_cols.
        layer: Ml8Layer (call `load_ml8_layer` to obtain).
        a_scale: per-row fp32 scale [M]. Defaults to all-ones if None
                 (matches "no upstream per-row activation scale" case).
        out_dtype: output tensor dtype (typically bf16 or fp16).
        block_size_m, block_size_n: tile sizes (16 × 16 is the safe default
                                    for gfx1201 WMMA; Phase F will tune).

    Returns:
        C: [M, N] tensor of dtype `out_dtype`.

    Notes:
        - K (= layer.n_cols) MUST be a multiple of `layer.group_size`
        - M MUST be a multiple of `block_size_m` (no padding handled here)
        - N (= layer.n_rows) MUST be a multiple of `block_size_n`
        Phase D / production wrapper will add padding/masking helpers.
    """
    assert a_fp8.dim() == 2, f"a_fp8 must be 2D, got shape {a_fp8.shape}"
    M, K = a_fp8.shape
    N = layer.n_rows
    group_size = layer.group_size
    n_centroids = layer.n_centroids
    device = a_fp8.device

    if K != layer.n_cols:
        raise ValueError(f"K mismatch: a_fp8 K={K}, layer K={layer.n_cols}")
    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be a multiple of group_size ({group_size})")
    if M % block_size_m != 0:
        raise ValueError(f"M ({M}) must be a multiple of block_size_m ({block_size_m})")
    if N % block_size_n != 0:
        raise ValueError(f"N ({N}) must be a multiple of block_size_n ({block_size_n})")

    if a_scale is None:
        a_scale = torch.ones(M, dtype=torch.float32, device=device)
    else:
        assert a_scale.shape == (M,), f"a_scale must be shape ({M},), got {a_scale.shape}"
        a_scale = a_scale.to(torch.float32).contiguous()

    c = torch.empty(M, N, dtype=out_dtype, device=device)

    # Kernel-side meta (matches successful invocation from Stage 1 tests)
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n
    BLOCK_SIZE_K = group_size      # GROUP_K == BLOCK_K constraint
    GROUP_K = group_size
    GROUP_N = 1                    # per-N b_scale matches ml8 calibration
    GROUP_SIZE_M = 1
    NUM_KSPLIT = 1
    SPLITK_BLOCK_SIZE = K
    NUM_STAGES = 1                 # gfx1201 num_stages>=2 UAF per RDNA4 audit

    # Strides
    stride_am, stride_ak = a_fp8.stride()
    stride_bk, stride_bn = layer.indices_packed.stride()
    stride_cm, stride_cn = c.stride()
    stride_ck = 0
    stride_ascale_m = 1
    stride_ascale_k = 0
    stride_bscale_k, stride_bscale_n = layer.scales_fp32.stride()
    stride_lut_k = layer.centroids_fp8.stride(0)

    grid_mn = (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)
    grid = (grid_mn * NUM_KSPLIT,)
    even_k = (K % BLOCK_SIZE_K == 0)

    gemm_ml8._gemm_a8w8_blockscale_kernel[grid](
        a_fp8, layer.indices_packed, c, a_scale, layer.scales_fp32,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_ck, stride_cm, stride_cn,
        stride_ascale_m, stride_ascale_k,
        stride_bscale_k, stride_bscale_n,
        GROUP_K=GROUP_K,
        GROUP_N=GROUP_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
        # LOCAL PATCH #3: EVEN_K/GRID_MN are now explicit kwargs (was heuristic)
        EVEN_K=even_k,
        GRID_MN=grid_mn,
        num_stages=NUM_STAGES,
        WEIGHT_FORMAT=1,
        N_CENTROIDS=n_centroids,
        centroid_lut_ptr=layer.centroids_fp8,
        stride_lut_k=stride_lut_k,
    )
    torch.cuda.synchronize()
    return c


# ─── Convenience: dequantize back to fp32 (reference path) ────────────────


def ml8_layer_from_blob(
    blob: dict[str, Any], device: torch.device | str = "cpu"
) -> Ml8Layer:
    """Build an Ml8Layer directly from an in-memory calibration blob
    (output of `ml8_io.load_ml8_layer`), skipping the on-disk .ml8
    file round-trip.

    Use this when reconstruct_model.py or other Python pipelines already
    have the .pt blob loaded and want to invoke the kernel directly.

    NOTE (MAD-245): this helper returns ONLY the kernel-friendly Ml8Layer.
    Rotation + AWQ metadata travel separately through `ml8_linear_from_blob`,
    which constructs an Ml8Linear that applies rotation/AWQ to activations
    at forward time. If you want a single call that returns a ready-to-use
    nn.Module honoring all blob metadata, use `ml8_linear_from_blob`.
    """
    device = torch.device(device)
    indices = blob["indices"]
    centroids = blob["centroids_per_group"]
    scales = blob["scale_per_group"]
    n_rows, n_cols = blob["shape"]
    group_size = int(blob["group_size"])
    n_centroids = int(blob["n_centroids"])
    n_groups_k = (n_cols + group_size - 1) // group_size

    if n_cols % 2 != 0:
        raise ValueError(f"ml8_layer_from_blob: n_cols ({n_cols}) must be even")
    if indices.dtype != torch.int8:
        raise TypeError(f"ml8_layer_from_blob: indices must be int8, got {indices.dtype}")

    # Pack indices [N, K] int8 → [K // 2, N] uint8 (lo-first convention)
    idx_np = indices.to(torch.uint8).cpu().contiguous().numpy()
    if (idx_np > 15).any():
        raise ValueError(
            f"ml8_layer_from_blob: indices out of [0,15] range; max={idx_np.max()}"
        )
    lo = idx_np[:, 0::2]
    hi = idx_np[:, 1::2]
    packed_n_kp = (lo & 0x0F) | ((hi & 0x0F) << 4)        # [N, K // 2] uint8
    packed_kp_n = packed_n_kp.T.copy()                     # [K // 2, N] kernel layout
    indices_packed = torch.from_numpy(packed_kp_n).contiguous().to(device)

    # Cast centroids fp32 → fp8 e4m3 (kernel layout same as on-disk)
    centroids_fp8 = (
        centroids.to(torch.float32).to(torch.float8_e4m3fn).contiguous().to(device)
    )

    # Scales: [N, n_groups_k] → [n_groups_k, N] kernel layout (K-major)
    scales_fp32 = (
        scales.to(torch.float32).T.contiguous().to(device)
    )

    return Ml8Layer(
        n_rows=n_rows,
        n_cols=n_cols,
        group_size=group_size,
        n_centroids=n_centroids,
        indices_packed=indices_packed,
        centroids_fp8=centroids_fp8,
        scales_fp32=scales_fp32,
        nibble_lo_first=True,
    )


def layer_from_components(
    centroids: torch.Tensor,
    scales: torch.Tensor,
    indices: torch.Tensor,
    gidx: torch.Tensor,
    device: torch.device | str = "cpu",
) -> Ml8Layer:
    """Build a kernel-ready Ml8Layer from live trainer tensors.

    Args:
        centroids: fp32 or fp8-e4m3 [G, 16] codebook centroids.
        scales:    fp32 [N, G] per-(row, group) scale.
        indices:   uint8 [N, K] centroid index per weight element (values 0..15).
        gidx:      long [K] group index per input column — MUST be uniform
                   contiguous grouping (gidx[c] == c // group_size for all c);
                   the ml8 kernel only supports this layout.
        device:    target device for the returned Ml8Layer tensors.

    Returns:
        Ml8Layer in kernel-friendly layout (indices packed lo-first nibbles,
        scales transposed to [G, N], centroids cast to fp8-e4m3).

    Raises:
        ValueError: if gidx does not match uniform contiguous grouping.
    """
    device = torch.device(device)
    indices = indices.to(torch.uint8)
    N, K = indices.shape
    G = centroids.shape[0]
    n_centroids = centroids.shape[1]
    group_size = K // G

    # Validate uniform contiguous grouping — kernel only supports this layout.
    expected_gidx = torch.arange(K, dtype=torch.long) // group_size
    if not torch.equal(gidx.cpu().long(), expected_gidx):
        raise ValueError(
            f"layer_from_components: gidx does not match uniform contiguous grouping "
            f"(group_size={group_size}, K={K}, G={G}). "
            "The ml8 kernel only supports uniform contiguous K-groups."
        )

    # Pack indices [N, K] uint8 → [K//2, N] uint8 (lo-first nibble convention).
    idx_np = indices.cpu().contiguous().numpy()   # [N, K]
    lo = idx_np[:, 0::2]
    hi = idx_np[:, 1::2]
    packed_n_kp = (lo & 0x0F) | ((hi & 0x0F) << 4)   # [N, K//2] uint8
    indices_packed = (
        torch.from_numpy(packed_n_kp.T.copy()).contiguous().to(device)
    )  # [K//2, N]

    # Cast centroids to fp8 e4m3 (accept fp32 or already-fp8).
    if centroids.dtype == torch.float8_e4m3fn:
        centroids_fp8 = centroids.contiguous().to(device)
    else:
        centroids_fp8 = (
            centroids.to(torch.float32).to(torch.float8_e4m3fn).contiguous().to(device)
        )

    # Transpose scales [N, G] → [G, N] for K-major kernel access.
    scales_fp32 = scales.to(torch.float32).T.contiguous().to(device)  # [G, N]

    return Ml8Layer(
        n_rows=N,
        n_cols=K,
        group_size=group_size,
        n_centroids=n_centroids,
        indices_packed=indices_packed,
        centroids_fp8=centroids_fp8,
        scales_fp32=scales_fp32,
        nibble_lo_first=True,
    )


class Ml8Linear(torch.nn.Module):
    """Drop-in nn.Linear replacement that invokes the ml8 kernel.

    Quantizes activations to fp8 e4m3 with per-row max-abs scaling at forward
    time, calls `ml8_gemm`, and returns output in the requested dtype
    (typically bf16 → cast to the model's original Linear output dtype).

    MAD-245 (Phase D.2) extension — when the source blob was calibrated with
    Hadamard rotation and/or AWQ rescaling, the stored W is in the
    `(W / awq_scale) @ Q` basis. We undo both at forward time on activations
    instead of absorbing into W:

        y = ml8_gemm(rotate(x * awq_scale), W_calibrated) + bias

    This is exact in floating point (AWQ commutativity + rotation orthogonality):
        (x * s) @ Q  ·  ((W / s) @ Q).T
      = (x * s) @ Q · Q.T · (W / s).T
      = (x * s) · (W / s).T
      = x · W.T

    so the result equals the un-rotated, un-AWQ'd linear up to fp8 quant noise
    on the rotated/scaled activations (which is what the calibration optimized
    for anyway).

    Shape contract:
      - input: [..., in_features] any float dtype
      - output: [..., out_features] same dtype as input

    Constraints (Phase D.1 v1; padding/masking left for v2):
      - input.shape[-1] must equal layer.n_cols (K)
      - K must be a multiple of layer.group_size
      - flattened batch dim (M = product of leading dims) must be a
        multiple of block_size_m (16 by default)
      - layer.n_rows (N) must be a multiple of block_size_n (16 by default)

    Activation quantization math (computed AFTER rotation + AWQ):
      a_scale[m] = max(|x[m]|) / fp8_max     where fp8_max ≈ 448 for e4m3
      x_fp8[m]  = x[m] / a_scale[m]          (in ~ [-448, 448])
      y[m]      = (x_fp8 @ W.T)[m] * a_scale[m] * b_scale[m, :]
      —the kernel does the scale multiplies internally.
    """

    _FP8_E4M3_MAX = 448.0

    def __init__(
        self,
        layer: "Ml8Layer",
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
        block_size_m: int = 16,
        block_size_n: int = 16,
        rotation=None,            # KroneckerRotation | None (MAD-245)
        awq_scale: torch.Tensor | None = None,   # [in_features] | None (MAD-245)
    ):
        super().__init__()
        self.layer = layer
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.register_buffer("bias", None)
        self.out_dtype = out_dtype
        self.block_size_m = block_size_m
        self.block_size_n = block_size_n
        self.in_features = layer.n_cols
        self.out_features = layer.n_rows
        # MAD-245: forward-time rotation + AWQ on activations.
        # rotation is kept as a regular attribute (not a buffer) since the
        # KroneckerRotation object owns its own dtype/device handling via
        # `_factors_on`. awq_scale IS a buffer because it's a plain tensor
        # we need to keep on the right device.
        self.rotation = rotation
        if awq_scale is not None:
            if awq_scale.dim() != 1 or awq_scale.shape[0] != self.in_features:
                raise ValueError(
                    f"Ml8Linear: awq_scale must be 1D with shape [{self.in_features}], "
                    f"got shape {tuple(awq_scale.shape)}"
                )
            self.register_buffer("awq_scale", awq_scale.contiguous())
        else:
            self.register_buffer("awq_scale", None)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"groups={self.layer.n_groups_k}, centroids={self.layer.n_centroids}, "
            f"bias={self.bias is not None}, "
            f"rotation={'yes' if self.rotation is not None else 'no'}, "
            f"awq={'yes' if self.awq_scale is not None else 'no'}, "
            f"out_dtype={self.out_dtype}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape

        # Flatten any leading batch dims to a single M.
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, K = x_2d.shape
        if K != self.in_features:
            raise ValueError(
                f"Ml8Linear: input K={K} does not match in_features={self.in_features}"
            )

        # MAD-245: AWQ + rotation on activations (calibration applied them
        # to W in reverse order, so the forward-time order is mirrored:
        # AWQ rescale → rotation forward).
        x_fp32 = x_2d.to(torch.float32)
        if self.awq_scale is not None:
            x_fp32 = x_fp32 * self.awq_scale.to(x_fp32.dtype)
        if self.rotation is not None:
            x_fp32 = self.rotation.forward(x_fp32)

        # Per-row max-abs activation quantization to fp8 e4m3.
        row_max = x_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [M, 1]
        a_scale = (row_max / self._FP8_E4M3_MAX).squeeze(1).contiguous()   # [M]
        x_fp8 = (x_fp32 / a_scale.unsqueeze(1)).to(torch.float8_e4m3fn).contiguous()

        # Kernel call.
        y = ml8_gemm(
            x_fp8, self.layer,
            a_scale=a_scale,
            out_dtype=self.out_dtype,
            block_size_m=self.block_size_m,
            block_size_n=self.block_size_n,
        )  # [M, N] in out_dtype

        # Bias.
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)

        # Restore shape + dtype.
        return y.reshape(*orig_shape[:-1], self.out_features).to(orig_dtype)


def ml8_linear_from_blob(
    blob: dict[str, Any],
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    block_size_m: int = 16,
    block_size_n: int = 16,
) -> "Ml8Linear":
    """Build a ready-to-use Ml8Linear from an in-memory calibration blob,
    automatically wiring up rotation + AWQ metadata if present (MAD-245).

    This is the right caller-facing API for full inference — it handles all
    blob variants (plain, rotated, AWQ'd, rotated+AWQ'd) with a single call.

    For legacy blobs with no rotation/AWQ metadata, behaves identically to
    `Ml8Linear(ml8_layer_from_blob(blob, device=device), bias=bias, ...)`.
    """
    # Local imports — keep top-level surface clean
    import sys as _sys
    _THIS = Path(__file__).resolve().parent
    if str(_THIS) not in _sys.path:
        _sys.path.insert(0, str(_THIS))
    from ml8_io import get_rotation, get_awq  # noqa: E402

    layer = ml8_layer_from_blob(blob, device=device)
    rotation = get_rotation(blob)
    awq_scale = get_awq(blob)
    if awq_scale is not None:
        awq_scale = awq_scale.to(device=torch.device(device))

    return Ml8Linear(
        layer,
        bias=bias,
        out_dtype=out_dtype,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
        rotation=rotation,
        awq_scale=awq_scale,
    )


def dequantize_ml8_layer(layer: Ml8Layer) -> torch.Tensor:
    """Reconstruct the full fp32 weight W [N, K] from an Ml8Layer.

    Useful for differential testing (compare ml8_gemm output to A @ W via
    standard matmul) and for layers that should run unquantized at some
    debug/validation step. Inverse of the calibration write path.
    """
    n_rows = layer.n_rows
    n_cols = layer.n_cols
    group_size = layer.group_size
    device = layer.indices_packed.device

    # Reconstruct un-packed indices [N, K] from packed [K//2, N]
    packed = layer.indices_packed.T.contiguous().cpu().numpy()  # [N, K//2]
    out = np.empty((n_rows, n_cols), dtype=np.int64)
    if layer.nibble_lo_first:
        out[:, 0::2] = packed & 0x0F
        out[:, 1::2] = (packed >> 4) & 0x0F
    else:
        out[:, 0::2] = (packed >> 4) & 0x0F
        out[:, 1::2] = packed & 0x0F
    indices = torch.from_numpy(out)

    centroids = layer.centroids_fp8.to(torch.float32).cpu()           # [n_groups_k, 16]
    scales = layer.scales_fp32.T.contiguous().cpu()                   # [N, n_groups_k]

    col_to_group = torch.arange(n_cols) // group_size                 # [K]
    group_idx = col_to_group.unsqueeze(0).expand(n_rows, n_cols)      # [N, K]
    cent_lookup = centroids[group_idx, indices]                       # [N, K]
    scale_lookup = scales.gather(1, group_idx)                        # [N, K]
    W = cent_lookup * scale_lookup
    return W.to(device)
