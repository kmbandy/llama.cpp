#!/usr/bin/env python3
"""MAD-223 Phase B.0b round-trip + convention test for ml8_to_packed.

Verifies the calibration `.pt` → packed `.ml8` binary converter:

  1. pack → unpack returns equivalent fields (indices exact, centroids
     bit-exact when input is pre-snapped to E4M3, scales fp32-exact).
  2. Reconstruct from packed bytes matches scripts/calibration/ml8_io.py's
     reference reconstruct_weight() output.
  3. Lo-first nibble convention produces the documented byte layout (so
     the kernel's unpack matches the converter's pack).

This is a regression test for the format itself — re-run after any change
to ml8_to_packed.py, ml8_io.py's blob schema, or the kernel's unpack logic.

Usage:
  /home/kmbandy/venvs/agents/bin/python3 tests/test_ml8_to_packed.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "calibration"))

from ml8_to_packed import (  # noqa: E402
    pack_layer, unpack_layer,
    ML8_MAGIC, ML8_VERSION, ML8_HEADER_SIZE,
)
from ml8_io import reconstruct_weight  # noqa: E402


def synth_blob(
    n_rows: int = 16,
    n_cols: int = 128,
    n_centroids: int = 16,
    group_size: int = 64,
    seed: int = 42,
) -> dict:
    """Synthesize a Cell C-shape blob with E4M3-snapped centroids.

    Matches ml8_io.py V1 schema. Centroid values are round-tripped through
    fp8 e4m3 so they're exactly representable (mimicking calibration's
    --snap-centroids e4m3 behavior).
    """
    torch.manual_seed(seed)
    n_groups = n_cols // group_size

    # Centroids: pre-snap to fp8 e4m3 so subsequent cast is bit-preserving
    raw = torch.randn(n_groups, n_centroids) * 0.5
    cent_fp8 = raw.to(torch.float8_e4m3fn)
    centroids = cent_fp8.to(torch.float32)

    indices = torch.randint(0, n_centroids, (n_rows, n_cols), dtype=torch.int8)
    scales = torch.randn(n_rows, n_groups, dtype=torch.float32).abs() * 0.1 + 0.01

    return {
        "name": "synthetic_test_layer",
        "shape": [n_rows, n_cols],
        "group_size": group_size,
        "n_centroids": n_centroids,
        "indices": indices,
        "centroids_per_group": centroids,
        "scale_per_group": scales,
        # Optional fields (ml8_io.py tolerates absence)
        "mse": 0.0, "w_snr_db": 0.0, "y_snr_db": 0.0, "rel_err": 0.0,
    }


# --- Tests ----------------------------------------------------------------

def test_header_fields() -> None:
    """Header magic + version + dims serialize correctly."""
    blob = synth_blob(n_rows=8, n_cols=16, group_size=8)
    packed = pack_layer(blob)

    import struct
    h = struct.unpack("<IIIIIIII", packed[:ML8_HEADER_SIZE])
    magic, version, n_rows, n_cols, group_size, n_centroids, flags, reserved = h

    assert magic == ML8_MAGIC, f"magic 0x{magic:08x} != 0x{ML8_MAGIC:08x}"
    assert version == ML8_VERSION, f"version {version} != {ML8_VERSION}"
    assert n_rows == 8
    assert n_cols == 16
    assert group_size == 8
    assert n_centroids == 16
    assert flags & 0x1, "expected nibble_lo_first flag set"
    assert reserved == 0
    print("  ✓ header fields: magic, version, dims, flags all correct")


def test_roundtrip_pack_unpack() -> None:
    """pack → unpack returns equivalent fields."""
    blob = synth_blob()
    packed = pack_layer(blob, nibble_lo_first=True)
    unpacked = unpack_layer(packed)

    assert unpacked["n_rows"] == blob["shape"][0]
    assert unpacked["n_cols"] == blob["shape"][1]
    assert unpacked["group_size"] == blob["group_size"]
    assert unpacked["n_centroids"] == blob["n_centroids"]
    assert unpacked["nibble_lo_first"] is True

    assert torch.equal(unpacked["indices"], blob["indices"]), \
        "indices not preserved through round-trip"
    assert torch.allclose(unpacked["centroids"], blob["centroids_per_group"], atol=0), \
        "centroids not bit-exact (input should have been pre-snapped to fp8 e4m3)"
    assert torch.equal(unpacked["scales"], blob["scale_per_group"]), \
        "scales not preserved through round-trip"
    print("  ✓ round-trip: indices exact, centroids bit-exact, scales fp32-exact")


def test_reconstruct_matches_ml8_io() -> None:
    """Reconstruct from packed bytes matches scripts/calibration/ml8_io.reconstruct_weight."""
    blob = synth_blob()

    W_baseline = reconstruct_weight(blob)

    packed = pack_layer(blob, nibble_lo_first=True)
    u = unpack_layer(packed)
    n_rows, n_cols, group_size = u["n_rows"], u["n_cols"], u["group_size"]

    # Vectorized reconstruction (same formula as ml8_io.reconstruct_weight)
    # W[r, c] = centroids[c // group_size][indices[r, c]] * scales[r, c // group_size]
    col_to_group = torch.arange(n_cols) // group_size            # [n_cols]
    indices_long = u["indices"].long()                            # [n_rows, n_cols]
    group_idx = col_to_group.unsqueeze(0).expand(n_rows, n_cols)  # [n_rows, n_cols]
    cent_lookup = u["centroids"][group_idx, indices_long]         # [n_rows, n_cols]
    scale_lookup = u["scales"].gather(1, group_idx)               # [n_rows, n_cols]
    W_packed = cent_lookup * scale_lookup

    max_diff = (W_baseline - W_packed).abs().max().item()
    assert max_diff < 1e-6, f"reconstruct mismatch: max_diff={max_diff}"
    print(f"  ✓ reconstruct matches ml8_io baseline (max_diff={max_diff:.2e})")


def test_nibble_lo_first_convention() -> None:
    """Verify the lo-first nibble layout produces the documented byte pattern.

    With nibble_lo_first=True:
      indices[i, 2j  ] lands in low nibble of byte (i, j)
      indices[i, 2j+1] lands in high nibble of byte (i, j)
    """
    # Known indices: row 0 = [0xA, 0x5, 0x3, 0xC]
    indices = torch.tensor([[0xA, 0x5, 0x3, 0xC]], dtype=torch.int8)
    blob = {
        "name": "nibble_test",
        "shape": [1, 4],
        "group_size": 4,
        "n_centroids": 16,
        "indices": indices,
        "centroids_per_group": torch.zeros(1, 16, dtype=torch.float32),
        "scale_per_group": torch.zeros(1, 1, dtype=torch.float32),
    }

    packed = pack_layer(blob, nibble_lo_first=True)
    indices_bytes = packed[ML8_HEADER_SIZE:ML8_HEADER_SIZE + 2]

    # Expected with lo-first:
    #   byte 0 = (0x5 << 4) | 0xA = 0x5A
    #   byte 1 = (0xC << 4) | 0x3 = 0xC3
    assert indices_bytes[0] == 0x5A, f"byte 0: 0x{indices_bytes[0]:02x} != 0x5A"
    assert indices_bytes[1] == 0xC3, f"byte 1: 0x{indices_bytes[1]:02x} != 0xC3"
    print(f"  ✓ lo-first: byte 0 = 0x{indices_bytes[0]:02x}, byte 1 = 0x{indices_bytes[1]:02x}")

    # And verify the hi-first variant produces the opposite layout:
    packed_hi = pack_layer(blob, nibble_lo_first=False)
    hi_bytes = packed_hi[ML8_HEADER_SIZE:ML8_HEADER_SIZE + 2]
    #   byte 0 = (0xA << 4) | 0x5 = 0xA5
    #   byte 1 = (0x3 << 4) | 0xC = 0x3C
    assert hi_bytes[0] == 0xA5, f"hi-first byte 0: 0x{hi_bytes[0]:02x} != 0xA5"
    assert hi_bytes[1] == 0x3C, f"hi-first byte 1: 0x{hi_bytes[1]:02x} != 0x3C"
    print(f"  ✓ hi-first: byte 0 = 0x{hi_bytes[0]:02x}, byte 1 = 0x{hi_bytes[1]:02x}")


def test_multiple_layer_shapes() -> None:
    """Pack/unpack works for varied shapes (especially non-square)."""
    shapes = [
        (2560, 2560, 64),          # Qwen3.5-4B gate_proj K=2560
        (11008, 2560, 64),         # gate_proj output
        (2560, 11008, 64),         # down_proj input
        (16, 128, 64),             # tiny stress shape
        (32, 64, 64),              # exactly 1 group
    ]
    for n_rows, n_cols, group_size in shapes:
        blob = synth_blob(n_rows=n_rows, n_cols=n_cols, group_size=group_size)
        packed = pack_layer(blob)
        unpacked = unpack_layer(packed)
        assert torch.equal(unpacked["indices"], blob["indices"])
        assert torch.equal(unpacked["scales"], blob["scale_per_group"])
        print(f"  ✓ shape ({n_rows}, {n_cols}) gs={group_size}: round-trip OK")


def main() -> int:
    print("# test_header_fields")
    test_header_fields()
    print("\n# test_roundtrip_pack_unpack")
    test_roundtrip_pack_unpack()
    print("\n# test_reconstruct_matches_ml8_io")
    test_reconstruct_matches_ml8_io()
    print("\n# test_nibble_lo_first_convention")
    test_nibble_lo_first_convention()
    print("\n# test_multiple_layer_shapes")
    test_multiple_layer_shapes()
    print()
    print("=== PASS: ml8_to_packed round-trip + convention + multi-shape verified ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
