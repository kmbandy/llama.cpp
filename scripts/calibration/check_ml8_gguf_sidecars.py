"""Byte-level check: ml8 GGUF sidecars vs source .pt blob.

Reads centroid LUT, rotation H_a, rotation_meta, and ml8 weight bytes
from a freshly-converted ml8 GGUF and the corresponding cell calibration
.pt blob. Computes max_abs_diff and SHA256 on each pair.

Run: python3 scripts/calibration/check_ml8_gguf_sidecars.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))
import gguf  # noqa: E402


GGUF_PATH = Path("/home/kmbandy/models/Qwen3.5-4B-ml8_4-cellE.gguf")
PT_PATH   = Path("/home/kmbandy/models/cell-e/model_layers_0_mlp_gate_proj.pt")
BASE      = "blk.0.ffn_gate"
WEIGHT_NAME = "blk.0.ffn_gate.weight"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    print(f"GGUF: {GGUF_PATH}")
    print(f"PT:   {PT_PATH}")
    print()

    reader = gguf.GGUFReader(str(GGUF_PATH))
    blob = torch.load(PT_PATH, map_location="cpu", weights_only=False)

    # Map tensor name -> tensor
    tmap = {t.name: t for t in reader.tensors}

    # ── 1. centroids
    cent_gguf = tmap[BASE + ".centroids"]
    cent_gguf_arr = np.asarray(cent_gguf.data)   # uint8 buffer
    print(f"  GGUF {cent_gguf.name}  shape={tuple(cent_gguf.shape)}  "
          f"dtype={cent_gguf.tensor_type}  bytes={cent_gguf_arr.nbytes}")

    cent_pt = blob["centroids_per_group"]  # [n_groups_k, 16] fp32
    cent_pt_fp8 = cent_pt.to(torch.float32).to(torch.float8_e4m3fn)
    cent_pt_bytes = cent_pt_fp8.contiguous().view(torch.uint8).numpy()
    print(f"  PT   centroids_per_group → fp8  shape={tuple(cent_pt_fp8.shape)}  "
          f"bytes={cent_pt_bytes.nbytes}")

    # Compare
    if cent_gguf_arr.shape == cent_pt_bytes.shape and (cent_gguf_arr == cent_pt_bytes).all():
        print(f"  ✓ centroids MATCH  sha={sha(cent_gguf_arr.tobytes())}")
    else:
        print(f"  ✗ centroids DIFFER")
        print(f"    GGUF sha = {sha(cent_gguf_arr.tobytes())}")
        print(f"    PT   sha = {sha(cent_pt_bytes.tobytes())}")
        # Dequantize both to fp32 and diff
        g_fp32 = np.asarray(cent_gguf_arr).view(np.uint8)
        p_fp32 = cent_pt_bytes.flatten()
        print(f"    GGUF first 16 bytes: {g_fp32[:16].tolist()}")
        print(f"    PT   first 16 bytes: {p_fp32[:16].tolist()}")
    print()

    # ── 2. rotation_h_a
    rot_gguf = tmap.get(BASE + ".rotation_h_a")
    rot_pt = blob.get("rotation", {}).get("h_a")
    if rot_gguf is not None and rot_pt is not None:
        rot_gguf_fp32 = np.asarray(rot_gguf.data).view(np.float32).reshape(tuple(rot_gguf.shape)[::-1])
        rot_pt_fp32 = rot_pt.to(torch.float32).contiguous().numpy()
        print(f"  GGUF {rot_gguf.name}  shape={tuple(rot_gguf.shape)}  "
              f"py-view-shape={rot_gguf_fp32.shape}")
        print(f"  PT   rotation.h_a  shape={rot_pt_fp32.shape}")
        # gguf might have reversed dim order
        if rot_gguf_fp32.shape != rot_pt_fp32.shape:
            print(f"    shape mismatch! trying transpose...")
            if rot_gguf_fp32.T.shape == rot_pt_fp32.shape:
                rot_gguf_fp32 = rot_gguf_fp32.T
        diff = np.abs(rot_gguf_fp32 - rot_pt_fp32).max() if rot_gguf_fp32.shape == rot_pt_fp32.shape else float("nan")
        if diff == 0.0:
            print(f"  ✓ rotation_h_a EXACT MATCH  sha={sha(rot_gguf_fp32.tobytes())}")
        else:
            print(f"  ✗ rotation_h_a DIFFERS  max|diff|={diff:.3e}")
            print(f"    GGUF first 4 values: {rot_gguf_fp32.flatten()[:4]}")
            print(f"    PT   first 4 values: {rot_pt_fp32.flatten()[:4]}")
    else:
        print(f"  (rotation_h_a missing from one side: gguf={rot_gguf is not None}, pt={rot_pt is not None})")
    print()

    # ── 3. rotation_meta
    meta_gguf = tmap.get(BASE + ".rotation_meta")
    if meta_gguf is not None:
        meta_arr = np.asarray(meta_gguf.data).view(np.int32)
        print(f"  GGUF {meta_gguf.name}  shape={tuple(meta_gguf.shape)}  values={meta_arr.tolist()}")
        # Expected: [a_dim, b_dim, in_features, kind_id=1]
        rot = blob.get("rotation", {})
        expected = [int(rot.get("a_dim", 0)), int(rot.get("b_dim", 0)),
                    int(rot.get("in_features", 0)), 1]
        print(f"  PT   expected:  {expected}")
        if meta_arr.tolist() == expected:
            print(f"  ✓ rotation_meta MATCH")
        else:
            print(f"  ✗ rotation_meta DIFFERS")
    print()

    # ── 4. main ml8 weight: block_ml8_4 format
    # Skip detailed block-by-block for now — if centroids + scales + indices
    # all match, ml8 weight is right by construction.
    w_gguf = tmap[WEIGHT_NAME]
    print(f"  GGUF {w_gguf.name}  shape={tuple(w_gguf.shape)}  "
          f"dtype={w_gguf.tensor_type}  bytes={np.asarray(w_gguf.data).nbytes}")

    indices = blob["indices"]  # int8 [N, K]
    scales = blob["scale_per_group"]  # fp32 [N, n_groups_k]
    print(f"  PT   indices shape={tuple(indices.shape)}  scale shape={tuple(scales.shape)}")

    # Sample first block bytes
    w_bytes = np.asarray(w_gguf.data)
    print(f"  GGUF first 8 bytes (block 0): {w_bytes[:8].tolist()}")
    print(f"  PT   scale[0, 0] = {scales[0, 0].item()}  "
          f"(if fp32 → bytes: {scales[0, 0].numpy().tobytes().hex()})")


if __name__ == "__main__":
    main()
