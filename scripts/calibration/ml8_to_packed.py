"""ml8_to_packed — convert ml8 calibration `.pt` blob → packed `.ml8` binary.

Pipeline:
    calibrate_ml8.py output (.pt, int8 indices, fp32 centroids, fp32 scales)
        → ml8_to_packed.py (pack indices to 4-bit, cast centroids to fp8)
        → .ml8 binary (kernel-consumable per ML8_WMMA_KERNEL_DESIGN.md Appendix A.2)

Per-layer file layout (32-byte header + 3 sections, each 16-byte-aligned):

    offset 0   : HEADER (32 bytes, little-endian)
        u32 magic        = 0x4D4C3849 ("ML8I")
        u32 version      = 1
        u32 n_rows       (output dim)
        u32 n_cols       (input dim)
        u32 group_size   (e.g. 64 for Cell C ml8-4)
        u32 n_centroids  (e.g. 16 for ml8-4)
        u32 flags        (bit 0 = nibble_order_lo_first)
        u32 reserved     = 0

    offset 32                  : indices_packed   uint8[n_rows, n_cols / 2]
    offset (32 + idx_size_pad) : centroids_fp8    uint8[n_groups, n_centroids]  (fp8 e4m3 raw bytes)
    offset (... + cent_pad)    : scales_fp32      float32[n_rows, n_groups]

Notes:
- Nibble packing: lo-nibble of byte j = index for column 2j; hi-nibble = column 2j+1.
  Default is lo-first; pass --nibble-hi-first to flip.
- Centroid cast (fp32 → fp8 e4m3) is bit-preserving when input is pre-snapped to
  the E4M3 lattice (which Cell C calibration does via --snap-centroids e4m3).
- Scale stays fp32 — it's per-(row, group_k), shape matches AITER blockscale's
  b_scale layout exactly.

Usage:
    # Single layer
    python3 scripts/calibration/ml8_to_packed.py \\
        --input /tmp/ml8-cellC/blk.0.ffn_gate.weight.pt \\
        --output /tmp/ml8-cellC-packed/blk.0.ffn_gate.weight.ml8

    # Batch mode (whole calibration directory)
    python3 scripts/calibration/ml8_to_packed.py \\
        --input-dir /tmp/ml8-cellC/ \\
        --output-dir /tmp/ml8-cellC-packed/
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --- Format constants -----------------------------------------------------

ML8_MAGIC: int = 0x4D4C3849      # "ML8I" little-endian
ML8_VERSION: int = 1
ML8_HEADER_SIZE: int = 32
HEADER_STRUCT: str = "<IIIIIIII"  # 8 × u32

FLAG_NIBBLE_LO_FIRST: int = 0x1   # bit 0 of flags


# --- Core conversion helpers ---------------------------------------------

def pack_indices(indices_int8: torch.Tensor, nibble_lo_first: bool = True) -> bytes:
    """Pack int8 indices in [0, 15] → uint8 nibble pairs.

    For lo-first convention (default):
        byte[i, j] = ((indices[i, 2j+1] & 0x0F) << 4) | (indices[i, 2j] & 0x0F)

    Args:
        indices_int8: shape [n_rows, n_cols], dtype int8, values in [0, 15].
        nibble_lo_first: if True, column 2j lands in low nibble; else high nibble.

    Returns:
        Packed bytes of length n_rows * (n_cols // 2).
    """
    if indices_int8.dtype != torch.int8:
        raise TypeError(f"indices must be int8, got {indices_int8.dtype}")
    if indices_int8.dim() != 2:
        raise ValueError(f"indices must be 2D, got shape {indices_int8.shape}")

    n_rows, n_cols = indices_int8.shape
    if n_cols % 2 != 0:
        raise ValueError(f"n_cols ({n_cols}) must be even for nibble packing")

    idx_np = indices_int8.to(torch.uint8).contiguous().numpy()
    if (idx_np > 15).any() or (idx_np < 0).any():
        raise ValueError(
            f"indices out of [0, 15] range; max={idx_np.max()}, min={idx_np.min()}"
        )

    lo = idx_np[:, 0::2]   # column 2j
    hi = idx_np[:, 1::2]   # column 2j+1

    if nibble_lo_first:
        packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    else:
        packed = (hi & 0x0F) | ((lo & 0x0F) << 4)

    return packed.astype(np.uint8).tobytes()


def unpack_indices(
    packed: bytes, n_rows: int, n_cols: int, nibble_lo_first: bool = True
) -> torch.Tensor:
    """Inverse of pack_indices (round-trip testing + reference unpack)."""
    n_packed = n_cols // 2
    expected = n_rows * n_packed
    if len(packed) != expected:
        raise ValueError(
            f"packed length {len(packed)} != expected {expected} "
            f"(n_rows={n_rows}, n_cols={n_cols})"
        )

    arr = np.frombuffer(packed, dtype=np.uint8).reshape(n_rows, n_packed)
    out = np.empty((n_rows, n_cols), dtype=np.int8)
    if nibble_lo_first:
        out[:, 0::2] = (arr & 0x0F).astype(np.int8)
        out[:, 1::2] = ((arr >> 4) & 0x0F).astype(np.int8)
    else:
        out[:, 0::2] = ((arr >> 4) & 0x0F).astype(np.int8)
        out[:, 1::2] = (arr & 0x0F).astype(np.int8)
    return torch.from_numpy(out)


def cast_centroids_to_fp8_bytes(centroids_fp32: torch.Tensor) -> bytes:
    """Cast fp32 centroids → fp8 e4m3 → raw bytes.

    Cell C calibration pre-snaps centroids to E4M3-representable values via
    `--snap-centroids e4m3`, so this cast is bit-preserving for properly
    calibrated input. For un-snapped input the cast still works but introduces
    rounding error — flagged in PPL eval, not by this script.
    """
    if centroids_fp32.dtype != torch.float32:
        raise TypeError(f"centroids must be float32, got {centroids_fp32.dtype}")
    cent_fp8 = centroids_fp32.to(torch.float8_e4m3fn).contiguous()
    return cent_fp8.view(torch.uint8).numpy().tobytes()


def _pad_to_16(b: bytes) -> bytes:
    """Pad a byte buffer up to the next 16-byte boundary."""
    rem = len(b) % 16
    return b + b"\x00" * ((16 - rem) % 16)


# --- Layer-level pack/unpack ---------------------------------------------

def pack_layer(blob: dict[str, Any], nibble_lo_first: bool = True) -> bytes:
    """Convert one ml8 calibration blob → packed binary.

    Input blob shape per scripts/calibration/ml8_io.py V1:
        {
            "shape": [n_rows, n_cols],
            "group_size": int,
            "n_centroids": int,
            "indices": int8 [n_rows, n_cols],
            "centroids_per_group": fp32 [n_groups, n_centroids],
            "scale_per_group": fp32 [n_rows, n_groups],
            # optional fields ignored (name, mse, rotation, awq, etc.)
        }

    Returns:
        Packed bytes ready to write to disk per Appendix A.2 layout.
    """
    indices = blob["indices"]
    centroids = blob["centroids_per_group"]
    scales = blob["scale_per_group"]
    n_rows, n_cols = blob["shape"]
    group_size = int(blob["group_size"])
    n_centroids = int(blob["n_centroids"])

    # Shape validation
    n_groups = (n_cols + group_size - 1) // group_size
    if tuple(indices.shape) != (n_rows, n_cols):
        raise ValueError(
            f"indices shape {tuple(indices.shape)} != ({n_rows}, {n_cols})"
        )
    if tuple(centroids.shape) != (n_groups, n_centroids):
        raise ValueError(
            f"centroids shape {tuple(centroids.shape)} != ({n_groups}, {n_centroids})"
        )
    if tuple(scales.shape) != (n_rows, n_groups):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} != ({n_rows}, {n_groups})"
        )

    flags = FLAG_NIBBLE_LO_FIRST if nibble_lo_first else 0
    header = struct.pack(
        HEADER_STRUCT,
        ML8_MAGIC, ML8_VERSION,
        int(n_rows), int(n_cols),
        group_size, n_centroids,
        flags, 0,
    )

    indices_bytes = pack_indices(indices, nibble_lo_first=nibble_lo_first)
    centroids_bytes = cast_centroids_to_fp8_bytes(centroids)
    scales_bytes = scales.contiguous().to(torch.float32).numpy().tobytes()

    return (
        header
        + _pad_to_16(indices_bytes)
        + _pad_to_16(centroids_bytes)
        + _pad_to_16(scales_bytes)
    )


def unpack_layer(packed: bytes) -> dict[str, Any]:
    """Round-trip inverse of pack_layer.

    Returns a dict with kernel-facing tensors:
        {
            "n_rows", "n_cols", "group_size", "n_centroids", "nibble_lo_first",
            "indices":   int8     [n_rows, n_cols],
            "centroids": float32  [n_groups, n_centroids],  (cast from fp8)
            "scales":    float32  [n_rows, n_groups],
        }
    """
    if len(packed) < ML8_HEADER_SIZE:
        raise ValueError(f"packed bytes too short ({len(packed)}) for header")

    magic, version, n_rows, n_cols, group_size, n_centroids, flags, _reserved = (
        struct.unpack(HEADER_STRUCT, packed[:ML8_HEADER_SIZE])
    )
    if magic != ML8_MAGIC:
        raise ValueError(f"bad magic: 0x{magic:08x} != 0x{ML8_MAGIC:08x}")
    if version != ML8_VERSION:
        raise ValueError(f"unsupported version: {version}")

    nibble_lo_first = bool(flags & FLAG_NIBBLE_LO_FIRST)
    n_groups = (n_cols + group_size - 1) // group_size

    indices_size = n_rows * (n_cols // 2)
    indices_size_padded = ((indices_size + 15) // 16) * 16
    centroids_size = n_groups * n_centroids
    centroids_size_padded = ((centroids_size + 15) // 16) * 16
    scales_size = n_rows * n_groups * 4

    expected_total = ML8_HEADER_SIZE + indices_size_padded + centroids_size_padded + scales_size
    if len(packed) < expected_total:
        raise ValueError(
            f"packed bytes truncated: have {len(packed)}, need at least {expected_total}"
        )

    off = ML8_HEADER_SIZE
    indices_bytes = packed[off:off + indices_size]
    off += indices_size_padded
    centroids_bytes = packed[off:off + centroids_size]
    off += centroids_size_padded
    scales_bytes = packed[off:off + scales_size]

    indices = unpack_indices(indices_bytes, n_rows, n_cols, nibble_lo_first)
    cent_fp8 = torch.frombuffer(
        bytearray(centroids_bytes), dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    centroids = cent_fp8.to(torch.float32).reshape(n_groups, n_centroids)
    scales = torch.frombuffer(
        bytearray(scales_bytes), dtype=torch.float32
    ).reshape(n_rows, n_groups).clone()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "group_size": group_size,
        "n_centroids": n_centroids,
        "nibble_lo_first": nibble_lo_first,
        "indices": indices,
        "centroids": centroids,
        "scales": scales,
    }


# --- CLI -----------------------------------------------------------------

def _convert_one(pt_path: Path, ml8_path: Path, nibble_lo_first: bool) -> int:
    blob = torch.load(pt_path, weights_only=False)
    packed = pack_layer(blob, nibble_lo_first=nibble_lo_first)
    ml8_path.parent.mkdir(parents=True, exist_ok=True)
    ml8_path.write_bytes(packed)
    return len(packed)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert ml8 calibration .pt blobs to packed .ml8 binary"
    )
    ap.add_argument("--input", type=Path, help="single-layer .pt input")
    ap.add_argument("--output", type=Path, help="single-layer .ml8 output")
    ap.add_argument("--input-dir", type=Path, help="batch: directory of .pt files")
    ap.add_argument("--output-dir", type=Path, help="batch: output directory")
    ap.add_argument(
        "--nibble-hi-first", action="store_true",
        help="use hi-nibble-first convention (default: lo-first)",
    )
    args = ap.parse_args()

    nibble_lo_first = not args.nibble_hi_first

    if args.input_dir:
        if not args.output_dir:
            ap.error("--input-dir requires --output-dir")
        if not args.input_dir.is_dir():
            ap.error(f"--input-dir does not exist: {args.input_dir}")
        pt_files = sorted(args.input_dir.glob("*.pt"))
        if not pt_files:
            print(f"No .pt files found in {args.input_dir}")
            return 1
        print(f"Converting {len(pt_files)} layer(s) (nibble_lo_first={nibble_lo_first})...")
        total_bytes = 0
        for pt_path in pt_files:
            ml8_path = args.output_dir / (pt_path.stem + ".ml8")
            n = _convert_one(pt_path, ml8_path, nibble_lo_first)
            total_bytes += n
            print(f"  {pt_path.name} → {ml8_path.name}  ({n:,} bytes)")
        print(f"Total: {len(pt_files)} layers, {total_bytes:,} bytes written")
    else:
        if not (args.input and args.output):
            ap.error("single mode requires --input and --output")
        if not args.input.exists():
            ap.error(f"--input does not exist: {args.input}")
        n = _convert_one(args.input, args.output, nibble_lo_first)
        print(f"Wrote {args.output}  ({n:,} bytes, nibble_lo_first={nibble_lo_first})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
