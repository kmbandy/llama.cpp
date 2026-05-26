"""ml8_to_gguf — produce a GGUF with native ml8-4 quantized MLP tensors + sidecars.

MAD-223 Phase G.5 rewrite. Replaces the prior "dequant-and-embed-as-fp16"
patcher that was a stopgap before the GGML ml8 type existed (G.1).

Input:
    --base-gguf BASE.gguf       # f16/bf16 GGUF holding the unmodified model
    --calib-dir DIR/            # directory of .pt ml8 calibration blobs
                                # (one per Linear, matching naming convention
                                #  model.layers.{L}.mlp.{gate,up,down}_proj)
Output:
    --out-gguf OUT.gguf         # GGUF with the matching MLP weights replaced
                                # by GGML_TYPE_ML8_4 tensors + sidecars:
                                #   blk.{L}.ffn_{gate,up,down}.weight              ML8_4
                                #   blk.{L}.ffn_{gate,up,down}.weight.centroids    F8_E4M3
                                #   blk.{L}.ffn_{gate,up,down}.weight.rotation_h_a F32  (opt)
                                #   blk.{L}.ffn_{gate,up,down}.weight.rotation_meta I32 (opt)
                                #   blk.{L}.ffn_{gate,up,down}.weight.awq_scale    F32  (opt)

Non-MLP tensors pass through unchanged from the base GGUF. RAM peak ~one
tensor at a time (~100 MB on Qwen3.5-4B; the base GGUF is mmapped).

See aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md for the on-disk format
and the rationale for the sidecar approach.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from ml8_io import load_ml8_layer, get_rotation, get_awq  # noqa: E402


# Block constants — must match ggml/src/ggml-common.h::block_ml8_4 + QK_ML8
QK_ML8 = 64
ML8_BLOCK_BYTES = 4 + QK_ML8 // 2  # 4-byte fp32 scale + 32 packed bytes = 36
N_CENTROIDS = 16

# Structural GGUF fields managed by the writer itself — don't try to re-add.
_SKIP_FIELDS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}

# Map HF Linear → GGUF MLP tensor name. Same as the prior patcher.
_HF_MLP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$")
_MLP_SUFFIX_MAP = {
    "gate_proj": "ffn_gate",
    "up_proj":   "ffn_up",
    "down_proj": "ffn_down",
}


def hf_to_gguf_name(hf_name: str) -> str:
    m = _HF_MLP_PATTERN.match(hf_name)
    if not m:
        raise ValueError(f"not an MLP HF name: {hf_name!r}")
    layer = m.group(1)
    suffix = _MLP_SUFFIX_MAP[m.group(2)]
    return f"blk.{layer}.{suffix}.weight"


# ─── Block packing ─────────────────────────────────────────────────────────


def pack_ml8_blocks(indices: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    """Pack one Linear's worth of ml8 indices + per-group scales into the
    `block_ml8_4` layout expected by ggml.

    Args:
        indices: [N, K] int8 (values 0..15). N rows, K columns.
        scales : [N, n_groups_k] fp32. K must be divisible by QK_ML8 = 64.

    Returns:
        uint8 ndarray of shape (N, n_groups_k * ML8_BLOCK_BYTES).
        Layout per block: 4-byte float32 scale + 32 packed bytes (lo-nibble first).
        Total bytes per row = n_groups_k * 36.
    """
    if indices.dim() != 2:
        raise ValueError(f"indices must be 2D, got {tuple(indices.shape)}")
    N, K = indices.shape
    if K % QK_ML8 != 0:
        raise ValueError(f"K={K} not divisible by QK_ML8={QK_ML8}")
    n_groups_k = K // QK_ML8
    if scales.shape != (N, n_groups_k):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} != expected ({N}, {n_groups_k})"
        )

    idx_np = indices.detach().cpu().contiguous().numpy().astype(np.uint8)
    if (idx_np > 15).any() or (idx_np < 0).any():
        raise ValueError(f"indices out of [0,15] range; min={idx_np.min()}, max={idx_np.max()}")
    scales_np = scales.detach().cpu().contiguous().numpy().astype(np.float32)

    # For each (row, group), pack:
    #   bytes[0..3]  = scale (little-endian fp32)
    #   bytes[4..35] = 32 bytes, each byte = (lo_idx & 0x0F) | ((hi_idx & 0x0F) << 4)
    #                 where lo_idx = idx_np[n, g*64 + 2i], hi_idx = idx_np[n, g*64 + 2i+1]
    out = np.empty((N, n_groups_k * ML8_BLOCK_BYTES), dtype=np.uint8)
    for g in range(n_groups_k):
        block_offset = g * ML8_BLOCK_BYTES
        # Scale: 4-byte fp32 at block start
        scale_bytes = scales_np[:, g].astype('<f4').view(np.uint8).reshape(N, 4)
        out[:, block_offset : block_offset + 4] = scale_bytes
        # Indices: K-positions [g*64 .. g*64+63], pack to 32 bytes (lo-first)
        col_start = g * QK_ML8
        lo = idx_np[:, col_start    : col_start + QK_ML8 : 2]   # [N, 32]
        hi = idx_np[:, col_start + 1: col_start + QK_ML8 : 2]   # [N, 32]
        packed = (lo & 0x0F) | ((hi & 0x0F) << 4)               # [N, 32]
        out[:, block_offset + 4 : block_offset + ML8_BLOCK_BYTES] = packed
    return out


def cast_centroids_to_fp8(centroids: torch.Tensor) -> np.ndarray:
    """Cast fp32 centroids [n_groups_k, N_CENTROIDS] → fp8 e4m3 byte array
    of the same shape. Uses PyTorch's `float8_e4m3fn` conversion (round-to-
    nearest-even, saturate to ±448, no inf).
    """
    if centroids.dim() != 2 or centroids.shape[1] != N_CENTROIDS:
        raise ValueError(
            f"centroids shape {tuple(centroids.shape)}; expected (n_groups_k, {N_CENTROIDS})"
        )
    fp8 = centroids.detach().cpu().to(torch.float32).to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8).contiguous().numpy()


# ─── Sidecar tensor helpers ────────────────────────────────────────────────


def _rotation_meta_bytes(blob_rotation: dict, in_features: int) -> np.ndarray | None:
    """Build the rotation_meta tensor [a_dim, b_dim, in_features, kind_id]
    (int32 × 4). Returns None when rotation is absent/identity.
    """
    if blob_rotation is None:
        return None
    kind = blob_rotation.get("kind")
    if kind in (None, "identity"):
        return None
    if kind != "kronecker_orth_sylvester":
        raise NotImplementedError(f"unsupported rotation kind: {kind!r}")
    a_dim = int(blob_rotation["a_dim"])
    b_dim = int(blob_rotation["b_dim"])
    blob_in = int(blob_rotation.get("in_features", in_features))
    if blob_in != in_features:
        raise ValueError(
            f"rotation in_features mismatch: blob says {blob_in}, weight has {in_features}"
        )
    # kind_id: 1 = kronecker_orth_sylvester. Reserve 0 for identity (never serialized here).
    return np.array([a_dim, b_dim, in_features, 1], dtype=np.int32)


# ─── Main conversion ──────────────────────────────────────────────────────


def _build_blob_map(calib_dir: Path) -> dict[str, Path]:
    """Map GGUF tensor name → blob path for every .pt file in calib_dir."""
    out: dict[str, Path] = {}
    for p in calib_dir.glob("*.pt"):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        hf_name = blob.get("name", p.stem.replace("_", "."))
        try:
            gguf_name = hf_to_gguf_name(hf_name)
        except ValueError:
            print(f"[skip] {p.name}: HF name {hf_name!r} doesn't match MLP pattern")
            continue
        if gguf_name in out:
            raise RuntimeError(f"duplicate blob mapping for {gguf_name}: {out[gguf_name]} vs {p}")
        out[gguf_name] = p
    return out


def _copy_field(writer: gguf.GGUFWriter, name: str, field) -> None:
    """Replicate a base-GGUF metadata field into the new writer."""
    types = field.types
    value = field.contents()
    primary = types[0]
    if primary == gguf.GGUFValueType.ARRAY:
        if len(types) < 2:
            raise ValueError(f"field {name!r}: ARRAY type without sub-type")
        sub_type = types[1]
        writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub_type=sub_type)
    else:
        writer.add_key_value(name, value, primary)


def convert_to_ml8_gguf(base_gguf: Path, calib_dir: Path, out_gguf: Path) -> dict:
    reader = gguf.GGUFReader(base_gguf)
    arch = reader.fields["general.architecture"].contents()
    print(f"[base] {base_gguf}  arch={arch!r}  "
          f"fields={len(reader.fields)}  tensors={len(reader.tensors)}")

    blob_map = _build_blob_map(calib_dir)
    print(f"[blobs] {len(blob_map)} calibrated layers → ml8 tensors + sidecars")

    writer = gguf.GGUFWriter(str(out_gguf), arch=arch)

    n_fields_copied = 0
    for name, field in reader.fields.items():
        if name in _SKIP_FIELDS:
            continue
        _copy_field(writer, name, field)
        n_fields_copied += 1
    # Informational metadata so downstream consumers can identify ml8 artifacts.
    writer.add_key_value("ml8.format_version", 1, gguf.GGUFValueType.UINT32)
    print(f"[fields] copied {n_fields_copied} + 1 ml8 marker")

    n_ml8 = 0
    n_centroids = 0
    n_rot = 0
    n_awq = 0
    n_copied = 0
    for tensor in reader.tensors:
        if tensor.name in blob_map:
            blob = load_ml8_layer(blob_map[tensor.name])

            indices  = blob["indices"]               # [N, K] int8
            cent     = blob["centroids_per_group"]   # [n_groups_k, 16] fp32
            scales   = blob["scale_per_group"]       # [N, n_groups_k] fp32
            N, K = indices.shape
            n_groups_k = K // QK_ML8
            if cent.shape != (n_groups_k, N_CENTROIDS):
                raise ValueError(
                    f"{tensor.name}: centroids shape {tuple(cent.shape)} "
                    f"!= expected ({n_groups_k}, {N_CENTROIDS})"
                )
            if scales.shape != (N, n_groups_k):
                raise ValueError(
                    f"{tensor.name}: scales shape {tuple(scales.shape)} "
                    f"!= expected ({N}, {n_groups_k})"
                )

            # ── Main ML8_4 tensor ──────────────────────────────────────────
            packed = pack_ml8_blocks(indices, scales)
            # ggml ne ordering is inner-first. For weight matrix viewed as [N, K]
            # numpy (out_features, in_features), ne = (K, N). The packed bytes
            # have shape (N, n_groups_k*36); ne_byte_shape = (n_groups_k*36, N).
            # Writer's quant_shape_from_byte_shape then converts inner byte-dim
            # back to element count: ne = (K, N). Match the patcher's transpose-
            # avoiding approach: pass packed in (N, n_groups_k*36) — the writer
            # treats shape[-1] as the inner byte-dim and converts it back.
            writer.add_tensor(tensor.name, packed, raw_dtype=GGMLQuantizationType.ML8_4)
            n_ml8 += 1

            # Sidecar names follow the llama.cpp convention used for `_s` scale
            # tensors: same root as the main weight, different suffix. Drop the
            # `.weight` middle name so the per-arch tn() builder can address
            # them as `tn(LLM_TENSOR_FFN_*, "centroids", il)` etc.
            base = tensor.name[:-len(".weight")] if tensor.name.endswith(".weight") else tensor.name

            # ── Sidecar: centroids (F8_E4M3) ───────────────────────────────
            cent_fp8 = cast_centroids_to_fp8(cent)  # [n_groups_k, 16] uint8
            writer.add_tensor(
                base + ".centroids",
                cent_fp8,
                raw_dtype=GGMLQuantizationType.F8_E4M3,
            )
            n_centroids += 1

            # ── Sidecar: rotation_h_a + rotation_meta (optional) ──────────
            rotation_dict = blob.get("rotation")
            meta = _rotation_meta_bytes(rotation_dict, in_features=K)
            if meta is not None:
                h_a = rotation_dict["h_a"].detach().cpu().to(torch.float32).contiguous().numpy()
                writer.add_tensor(base + ".rotation_h_a", h_a)
                writer.add_tensor(base + ".rotation_meta", meta)
                n_rot += 1

            # ── Sidecar: AWQ scale (optional) ──────────────────────────────
            awq_s = get_awq(blob)
            if awq_s is not None:
                awq_np = awq_s.detach().cpu().to(torch.float32).contiguous().numpy()
                if awq_np.shape != (K,):
                    raise ValueError(
                        f"{tensor.name}: awq_scale shape {awq_np.shape} != ({K},)"
                    )
                writer.add_tensor(base + ".awq_scale", awq_np)
                n_awq += 1
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            n_copied += 1

    print(
        f"[tensors] main ml8={n_ml8}, centroids={n_centroids}, rotation={n_rot}, "
        f"awq={n_awq}, copied unchanged={n_copied}"
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "out_path":         str(out_gguf),
        "n_ml8":            n_ml8,
        "n_centroids":      n_centroids,
        "n_rotation":       n_rot,
        "n_awq":            n_awq,
        "n_copied":         n_copied,
        "n_fields_copied":  n_fields_copied,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-gguf", type=Path, required=True,
                   help="Existing f16/bf16 GGUF to extend with ml8 MLP tensors")
    p.add_argument("--calib-dir", type=Path, required=True,
                   help="Directory of ml8-4 per-layer .pt blobs")
    p.add_argument("--out-gguf", type=Path, required=True,
                   help="Output GGUF path")
    args = p.parse_args()
    summary = convert_to_ml8_gguf(args.base_gguf, args.calib_dir, args.out_gguf)
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
