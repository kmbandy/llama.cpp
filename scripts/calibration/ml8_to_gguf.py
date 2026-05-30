"""ml8_to_gguf — produce a GGUF with native ml8-4 quantized MLP tensors + sidecars.

MAD-223 Phase G.5 rewrite. Replaces the prior "dequant-and-embed-as-fp16"
patcher that was a stopgap before the GGML ml8 type existed (G.1).

Input:
    --base-gguf BASE.gguf       # f16/bf16 GGUF holding the unmodified model
    --calib-dir DIR/            # directory of .pt ml8 calibration blobs
                                # (one per Linear, matching naming convention
                                #  model.layers.{L}.mlp.{gate,up,down}_proj)
                                # Also accepts full-model blobs from
                                # calibrate_ml8_paged --dense-coverage full:
                                #   {name}.pt      — new ML8-tier roles
                                #   {name}.fp8.pt  — FP8-tier (token_embd, ssm_*)
Output:
    --out-gguf OUT.gguf         # GGUF with the matching MLP weights replaced
                                # by GGML_TYPE_ML8_4 tensors + sidecars:
                                #   blk.{L}.ffn_{gate,up,down}.weight              ML8_4
                                #   blk.{L}.ffn_{gate,up,down}.weight.centroids    F8_E4M3
                                #   blk.{L}.ffn_{gate,up,down}.weight.rotation_h_a F32  (opt)
                                #   blk.{L}.ffn_{gate,up,down}.weight.rotation_meta I32 (opt)
                                #   blk.{L}.ffn_{gate,up,down}.weight.awq_scale    F32  (opt)
                                # New ML8-tier roles (attn/ssm/lm_head/eh_proj):
                                #   Same sidecar structure as FFN.
                                # FP8-tier (token_embd, ssm_alpha, ssm_beta):
                                #   <name>.weight                                  ML8_FP8

Non-MLP tensors pass through unchanged from the base GGUF. RAM peak ~one
tensor at a time (~100 MB on Qwen3.5-4B; the base GGUF is mmapped).

See aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md for the on-disk format
and the rationale for the sidecar approach.
"""
from __future__ import annotations

import argparse
import mmap as _mmap_mod
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def _advise_dontneed(fd: int, offset: int, length: int) -> None:
    """Tell the kernel we won't re-read this byte range — drop its page cache.

    Critical for converting large base GGUFs (35B+ class) on RAM-constrained
    boxes. Without this, sequentially reading every tensor in a 71 GB GGUF
    keeps all those pages hot in the page cache, pushing other processes
    (e.g. a running inference server) into swap and risking an OOM kill.
    """
    if not hasattr(os, "posix_fadvise") or fd < 0:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from ml8_io import load_ml8_layer, get_rotation, get_awq  # noqa: E402
from role_targets import classify_role, Tier  # noqa: E402


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

# Map HF Linear → GGUF MLP tensor name. Two patterns:
#   Dense:  model.layers.{L}.mlp.{gate,up,down}_proj         → blk.{L}.ffn_{gate,up,down}.weight
#   MoE:    model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj
#                                                            → blk.{L}.ffn_{gate,up,down}_exps.weight
# (MAD-223 G.7: MoE pattern, n_experts blobs stacked into one ggml tensor.)
_HF_MLP_PATTERN     = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$")
_HF_MOE_PATTERN     = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
_MLP_SUFFIX_MAP = {
    "gate_proj": "ffn_gate",
    "up_proj":   "ffn_up",
    "down_proj": "ffn_down",
}
_MOE_SUFFIX_MAP = {
    "gate_proj": "ffn_gate_exps",
    "up_proj":   "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}


def parse_hf_name(hf_name: str) -> tuple[str, int | None]:
    """Return (gguf_tensor_name, expert_id_or_None). Raises ValueError if
    the HF name isn't a recognized MLP / MoE-expert weight."""
    m = _HF_MOE_PATTERN.match(hf_name)
    if m:
        layer  = m.group(1)
        expert = int(m.group(2))
        suffix = _MOE_SUFFIX_MAP[m.group(3)]
        return f"blk.{layer}.{suffix}.weight", expert
    m = _HF_MLP_PATTERN.match(hf_name)
    if m:
        layer  = m.group(1)
        suffix = _MLP_SUFFIX_MAP[m.group(2)]
        return f"blk.{layer}.{suffix}.weight", None
    raise ValueError(f"not an MLP/MoE HF name: {hf_name!r}")


def hf_to_gguf_name(hf_name: str) -> str:
    """Back-compat shim — returns just the GGUF name. New code should use
    parse_hf_name() which also tells you the expert id (or None for dense)."""
    return parse_hf_name(hf_name)[0]


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


def pack_ml8_blocks_soa(indices: torch.Tensor, scales: torch.Tensor) -> bytes:
    """Pack one expert's ml8 indices + per-group scales into the stored-as-repacked
    SOA byte layout consumed directly by the AITER MoE GEMM kernel — no runtime
    repack needed.

    Layout (per expert):
        bytes [0                .. K/2 × N - 1]                = b_packed
        bytes [K/2 × N          .. K/2 × N + n_groups_k × N × 4 - 1] = b_scale

    b_packed is uint8 in (K/2, N) row-major. b_scale is fp32 in
    (n_groups_k, N) row-major. These are the exact arrays that
    `ggml_cuda_ml8_repack_blocks_moe` would produce at runtime from the AOS
    block layout — pre-computing them in the GGUF eliminates the runtime
    transform + cache.

    Args:
        indices: [N, K] int (values 0..15)
        scales : [N, n_groups_k] fp32. K must be divisible by QK_ML8 = 64.

    Returns:
        Raw bytes of length N × K × 9 / 16 = same total as pack_ml8_blocks for
        the corresponding AOS form.
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

    # b_packed [K/2, N] uint8 — nibbles only.
    # Pack each adjacent K-pair into one byte: lo nibble = idx at 2i,
    # hi nibble = idx at 2i+1. Result has shape [N, K/2] in source order.
    lo = idx_np[:, 0::2]                                       # [N, K/2]
    hi = idx_np[:, 1::2]                                       # [N, K/2]
    packed_NK2 = ((lo & 0x0F) | ((hi & 0x0F) << 4)).astype(np.uint8)  # [N, K/2]
    b_packed = np.ascontiguousarray(packed_NK2.T)              # [K/2, N] row-major

    # b_scale [n_groups_k, N] fp32 — scales only.
    b_scale = np.ascontiguousarray(scales_np.T)                # [n_groups_k, N] row-major

    return b_packed.tobytes() + b_scale.tobytes()


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


# FP8 block constants — must match ML8_FP8 ggml block layout.
_FP8_GROUP_SIZE = 32   # FIXED — matches GGML_QUANT_SIZES[ML8_FP8] block_size
_FP8_BLOCK_BYTES = 34  # 2 bytes fp16 scale + 32 bytes e4m3 = 34


def pack_scaled_fp8_blocks(e4m3_f32: torch.Tensor,
                           scale_fp16: torch.Tensor) -> np.ndarray:
    """Pack a scaled-FP8 weight into the ML8_FP8 on-disk block layout.

    On-disk layout: for weight [N rows, K cols], K divisible by 32, stored as
    N × (K/32) blocks row-major.  Each block is 34 bytes:
        [scale : 1×float16 little-endian (2 bytes)][qs : 32×uint8 e4m3]

    Args:
        e4m3_f32  : [N, K] float32 tensor whose values lie on the e4m3 lattice
                    (i.e. already cast to fp8 then back; or just in-range fp32).
        scale_fp16: [N, K//32] float16 (or float32) per-block scales.

    Returns:
        uint8 ndarray of shape (N, (K//32) * 34) — ready to hand to
        writer.add_tensor(..., raw_dtype=GGMLQuantizationType.ML8_FP8).
    """
    # Sanity-check GGML_QUANT_SIZES so the assert is caught early if the
    # constant file is inconsistent with this packer.
    _bs, _ts = GGML_QUANT_SIZES[GGMLQuantizationType.ML8_FP8]
    assert _bs == _FP8_GROUP_SIZE and _ts == _FP8_BLOCK_BYTES, (
        f"ML8_FP8 GGML_QUANT_SIZES mismatch: expected ({_FP8_GROUP_SIZE}, "
        f"{_FP8_BLOCK_BYTES}), got ({_bs}, {_ts})"
    )

    if e4m3_f32.dim() != 2:
        raise ValueError(f"e4m3_f32 must be 2D, got {tuple(e4m3_f32.shape)}")
    N, K = e4m3_f32.shape
    if K % _FP8_GROUP_SIZE != 0:
        raise ValueError(f"K={K} not divisible by FP8 group size {_FP8_GROUP_SIZE}")
    n_blocks = K // _FP8_GROUP_SIZE
    if scale_fp16.shape != (N, n_blocks):
        raise ValueError(
            f"scale shape {tuple(scale_fp16.shape)} != expected ({N}, {n_blocks})"
        )

    # Encode e4m3 bytes: canonical encoding identical to cast_centroids_to_fp8.
    qs_bytes = (e4m3_f32.detach().cpu()
                .to(torch.float32)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .contiguous()
                .numpy())  # [N, K] uint8

    out = np.empty((N, n_blocks * _FP8_BLOCK_BYTES), dtype=np.uint8)
    for b in range(n_blocks):
        block_off = b * _FP8_BLOCK_BYTES
        # Scale: 2-byte fp16 little-endian at block start.
        scale_row = scale_fp16[:, b].detach().cpu().to(torch.float16).contiguous()
        scale_u8 = scale_row.view(torch.uint8).numpy()  # [N, 2]
        out[:, block_off : block_off + 2] = scale_u8.reshape(N, 2)
        # e4m3 bytes: 32 bytes per block.
        col_start = b * _FP8_GROUP_SIZE
        out[:, block_off + 2 : block_off + _FP8_BLOCK_BYTES] = (
            qs_bytes[:, col_start : col_start + _FP8_GROUP_SIZE]
        )
    return out


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


def _build_blob_map(calib_dir: Path) -> dict[str, list[tuple[int | None, Path]]]:
    """Map GGUF tensor name → list of (expert_id, blob path).

    Dense entries have a single-element list with expert_id=None. MoE
    entries have one list element per expert (already sorted by expert id
    at the end so the stack ordering is deterministic).

    Discovers both legacy MLP/MoE blobs (via parse_hf_name) and new full-model
    ML8-tier blobs for attn/ssm/lm_head/eh_proj roles (via classify_role).
    FP8-tier blobs (*.fp8.pt) are excluded here — they are handled by
    _build_fp8_blob_map.
    """
    out: dict[str, list[tuple[int | None, Path]]] = {}
    for p in calib_dir.glob("*.pt"):
        # FP8-tier blobs share the *.pt suffix pattern — exclude them here.
        if p.name.endswith(".fp8.pt"):
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        hf_name = blob.get("name", p.stem.replace("_", "."))
        try:
            gguf_name, expert_id = parse_hf_name(hf_name)
        except ValueError:
            # Not an MLP/MoE pattern — try the full-model role classifier.
            gguf_name_r, _role, tier = classify_role(hf_name)
            if tier is Tier.ML8:
                # Dense new-role blob (attn/ssm/lm_head/eh_proj/etc.).
                out.setdefault(gguf_name_r, []).append((None, p))
            else:
                print(
                    f"[skip] {p.name}: HF name {hf_name!r} doesn't match "
                    f"MLP/MoE pattern and role is {tier.value} (not ML8)"
                )
            continue
        out.setdefault(gguf_name, []).append((expert_id, p))
    # Validate + sort.
    for gguf_name, entries in out.items():
        kinds = {e[0] is None for e in entries}
        if len(kinds) != 1:
            raise RuntimeError(
                f"{gguf_name}: mixed dense/MoE blobs for the same tensor")
        if entries[0][0] is None:
            if len(entries) != 1:
                raise RuntimeError(
                    f"{gguf_name}: dense pattern but {len(entries)} blobs")
        else:
            entries.sort(key=lambda x: x[0])
            seen = [e[0] for e in entries]
            if seen != list(range(len(entries))):
                raise RuntimeError(
                    f"{gguf_name}: expert ids not 0..N-1 contiguous; got {seen}")
    return out


def _build_fp8_blob_map(calib_dir: Path) -> dict[str, Path]:
    """Map GGUF tensor name → FP8 blob path for scaled-FP8 tier weights.

    Globs *.fp8.pt blobs and maps each blob's 'name' field via classify_role.
    Only Tier.FP8 blobs are included (others would be unexpected and are warned).
    """
    out: dict[str, Path] = {}
    for p in calib_dir.glob("*.fp8.pt"):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        hf_name = blob.get("name", "")
        if not hf_name:
            print(f"[skip-fp8] {p.name}: blob has no 'name' field")
            continue
        gguf_name, _role, tier = classify_role(hf_name)
        if tier is not Tier.FP8:
            print(
                f"[skip-fp8] {p.name}: HF name {hf_name!r} has tier "
                f"{tier.value} (expected FP8)"
            )
            continue
        if gguf_name in out:
            raise RuntimeError(
                f"Duplicate FP8 blob for {gguf_name!r}: {out[gguf_name]} and {p}"
            )
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


def evaluate_coverage(params_ml8: int, params_fp8: int,
                      params_passthrough_weight: int,
                      min_coverage: float) -> tuple[float, bool, dict]:
    """Matmul-weight quantization coverage and whether it's below the refuse
    threshold. Pure arithmetic — factored out of convert_to_ml8_gguf so the
    guardrail decision is unit-testable without building a GGUF.

    coverage = (ml8_params + fp8_params) / total_weight_params.
    The FP8 tier (8-bit scaled-FP8 for embed/ssm) is credited as quantized
    since it represents a genuine compression from bf16.

    Returns (coverage_fraction, below_threshold, breakdown_dict) where
    breakdown_dict has keys "ml8", "fp8", "bf16" (each as a fraction of total).
    """
    total = params_ml8 + params_fp8 + params_passthrough_weight
    coverage = (params_ml8 + params_fp8) / total if total else 0.0
    breakdown = {
        "ml8": params_ml8 / total if total else 0.0,
        "fp8": params_fp8 / total if total else 0.0,
        "bf16": params_passthrough_weight / total if total else 0.0,
    }
    return coverage, coverage < min_coverage, breakdown


def convert_to_ml8_gguf(base_gguf: Path, calib_dir: Path, out_gguf: Path,
                        min_coverage: float = 0.85,
                        allow_partial: bool = False) -> dict:
    reader = gguf.GGUFReader(base_gguf)
    arch = reader.fields["general.architecture"].contents()
    print(f"[base] {base_gguf}  arch={arch!r}  "
          f"fields={len(reader.fields)}  tensors={len(reader.tensors)}")

    # ── Memory-pressure mitigation for large bases (e.g. 35B-A3B = 71 GB GGUF):
    # 1. Tell kernel sequential access pattern on the reader's mmap so it
    #    doesn't aggressively keep pages hot once we've moved past them.
    # 2. Pull the underlying file descriptor so we can issue per-tensor
    #    POSIX_FADV_DONTNEED after each pass-through, which immediately
    #    releases that tensor's page cache.
    base_fd = -1
    try:
        underlying = reader.data._mmap  # numpy memmap → mmap.mmap
        if hasattr(underlying, "madvise"):
            try:
                underlying.madvise(_mmap_mod.MADV_SEQUENTIAL)
            except (OSError, ValueError):
                pass
        # Try to get fd via the file the memmap was opened from. numpy
        # doesn't expose it cleanly; fall back to reopening the path RO.
        try:
            base_fd = os.open(str(base_gguf), os.O_RDONLY)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(base_fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except OSError:
            base_fd = -1
    except AttributeError:
        pass

    blob_map = _build_blob_map(calib_dir)
    fp8_blob_map = _build_fp8_blob_map(calib_dir)
    print(f"[blobs] {len(blob_map)} ml8 calibrated layers → ml8 tensors + sidecars; "
          f"{len(fp8_blob_map)} fp8 blob(s) → ML8_FP8 tensors")

    # Spool tensor bytes to a temp file on the same disk as the output
    # (NVMe) — keeps process heap bounded regardless of total output size.
    # CRITICAL: do NOT let SpooledTemporaryFile default to /tmp if /tmp is
    # tmpfs (CachyOS / many Linux distros) — that just moves the RAM cost.
    out_dir = out_gguf.parent if out_gguf.parent != Path("") else Path(".")
    tempfile.tempdir = str(out_dir)
    print(f"[tempdir] tensor spool → {tempfile.tempdir} "
          f"(use_temp_file=True keeps heap bounded)")

    writer = gguf.GGUFWriter(str(out_gguf), arch=arch, use_temp_file=True)

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
    n_moe = 0
    n_fp8 = 0
    n_centroids = 0
    n_rot = 0
    n_awq = 0
    n_copied = 0
    # ── Quantization-coverage accounting (the guardrail) ───────────────────
    # The failure this prevents: silently emitting a GGUF where most of the
    # model's matmul weight stayed bf16 (e.g. FFN-only calibration on a dense
    # hybrid model → 76% un-quantized) and calling it a "4-bit" artifact.
    # We count *original parameters* (not packed bytes) so a quantized tensor
    # is credited its full original weight share — apples-to-apples with the
    # bf16 tensors it's compared against. "Weight" = any 2-D+ `.weight` tensor
    # (norms/biases are 1-D and legitimately stay high-precision in every
    # scheme, so they're excluded from the coverage denominator).
    # params_ml8 = 4-bit ml8 quantized params; params_fp8 = 8-bit FP8 params.
    params_ml8 = 0
    params_fp8 = 0
    params_passthrough_weight = 0
    passthrough_weights: list[tuple[str, int, str]] = []   # (name, params, dtype)
    for tensor in reader.tensors:
        if tensor.name in blob_map:
            entries = blob_map[tensor.name]
            is_moe = entries[0][0] is not None

            # Load all blobs for this tensor (1 for dense, n_experts for MoE).
            blobs = [load_ml8_layer(path) for _, path in entries]
            n_experts = len(blobs)

            # Shape derivation from blob 0; assert the rest match.
            b0 = blobs[0]
            indices0 = b0["indices"]
            cent0    = b0["centroids_per_group"]
            scales0  = b0["scale_per_group"]
            N, K = indices0.shape
            n_groups_k = K // QK_ML8
            if cent0.shape != (n_groups_k, N_CENTROIDS):
                raise ValueError(
                    f"{tensor.name}: centroids shape {tuple(cent0.shape)} "
                    f"!= expected ({n_groups_k}, {N_CENTROIDS})")
            if scales0.shape != (N, n_groups_k):
                raise ValueError(
                    f"{tensor.name}: scales shape {tuple(scales0.shape)} "
                    f"!= expected ({N}, {n_groups_k})")
            for ei, b in enumerate(blobs[1:], 1):
                if b["indices"].shape != indices0.shape:
                    raise ValueError(
                        f"{tensor.name}: expert {ei} indices shape mismatch")
                if b["centroids_per_group"].shape != cent0.shape:
                    raise ValueError(
                        f"{tensor.name}: expert {ei} centroids shape mismatch")
                if b["scale_per_group"].shape != scales0.shape:
                    raise ValueError(
                        f"{tensor.name}: expert {ei} scales shape mismatch")

            # ── Main ML8 tensor ────────────────────────────────────────────
            # Dense (ML8_4 AOS): packed bytes shape (N, n_groups_k*36). The
            # runtime layout transform is small (one tensor) so we keep this
            # layout for the dense path.
            # MoE (ML8_4_SOA): bytes per expert = K/2 × N nibbles + n_groups_k
            # × N × 4 scales, in the AITER kernel's expected SOA layout. Stored
            # this way to eliminate the runtime repack cache that doesn't fit
            # in VRAM at 35B+ scale (MAD-244).
            if is_moe:
                # K = indices.shape[1], N = indices.shape[0] (per expert).
                K_logical = int(blobs[0]["indices"].shape[1])
                N_per     = int(blobs[0]["indices"].shape[0])
                K_bytes   = (K_logical * 9) // 16  # K * 0.5625 exactly
                expert_payloads = [
                    pack_ml8_blocks_soa(b["indices"], b["scale_per_group"])
                    for b in blobs]
                all_bytes = b"".join(expert_payloads)
                # Sanity: total bytes match the (K, N, n_experts) tensor size.
                expected = n_experts * N_per * K_bytes
                if len(all_bytes) != expected:
                    raise RuntimeError(
                        f"{tensor.name}: SOA byte count {len(all_bytes)} != "
                        f"expected {expected}")
                # Shape order for gguf-py: (n_experts, N, K_bytes) gives GGUF
                # shape (K, N, n_experts) after reversal. The ML8_4_SOA dtype
                # tells the loader to treat the bytes as kernel-native SOA
                # (b_packed then b_scale per expert), not as AOS blocks.
                arr = np.frombuffer(all_bytes, dtype=np.uint8).reshape(
                    n_experts, N_per, K_bytes)
                writer.add_tensor(tensor.name, arr,
                                  raw_dtype=GGMLQuantizationType.ML8_4_SOA)
                n_moe += 1
                params_ml8 += n_experts * N_per * K_logical
            else:
                packed = pack_ml8_blocks(indices0, scales0)
                writer.add_tensor(tensor.name, packed,
                                  raw_dtype=GGMLQuantizationType.ML8_4)
                n_ml8 += 1
                params_ml8 += N * K

            # Sidecar names follow the llama.cpp convention used for `_s` scale
            # tensors: same root as the main weight, different suffix.
            base = tensor.name[:-len(".weight")] if tensor.name.endswith(".weight") else tensor.name

            # ── Sidecar: centroids (F8_E4M3) ───────────────────────────────
            # Dense: [n_groups_k, 16].  MoE: [n_experts, n_groups_k, 16].
            if is_moe:
                cent_stack = np.stack(
                    [cast_centroids_to_fp8(b["centroids_per_group"]) for b in blobs],
                    axis=0)
                writer.add_tensor(base + ".centroids", cent_stack,
                                  raw_dtype=GGMLQuantizationType.F8_E4M3)
            else:
                cent_fp8 = cast_centroids_to_fp8(cent0)
                writer.add_tensor(base + ".centroids", cent_fp8,
                                  raw_dtype=GGMLQuantizationType.F8_E4M3)
            n_centroids += 1

            # ── Sidecar: rotation_h_a + rotation_meta (optional) ───────────
            # Rotation is applied OUTSIDE the matmul kernel (on x), so MoE
            # layers must use the same rotation across experts. Validate
            # this contract and write a single rotation_h_a per layer/kind.
            rotation_dict = b0.get("rotation")
            meta = _rotation_meta_bytes(rotation_dict, in_features=K)
            if meta is not None:
                h_a_ref = rotation_dict["h_a"].detach().cpu().to(torch.float32).contiguous()
                if is_moe:
                    for ei, b in enumerate(blobs[1:], 1):
                        r_e = b.get("rotation")
                        if r_e is None:
                            raise ValueError(
                                f"{tensor.name}: expert 0 has rotation but expert {ei} does not")
                        h_a_e = r_e["h_a"].detach().cpu().to(torch.float32).contiguous()
                        if not torch.allclose(h_a_e, h_a_ref, atol=0.0, rtol=0.0):
                            raise ValueError(
                                f"{tensor.name}: expert {ei} rotation_h_a differs from expert 0 "
                                f"— MoE inference requires identical rotation across experts.")
                writer.add_tensor(base + ".rotation_h_a", h_a_ref.numpy())
                writer.add_tensor(base + ".rotation_meta", meta)
                n_rot += 1

            # ── Sidecar: AWQ scale (optional) ──────────────────────────────
            # Same rationale as rotation — applied on x upstream, must be
            # identical across experts in an MoE layer.
            awq_ref = get_awq(b0)
            if awq_ref is not None:
                awq_np_ref = awq_ref.detach().cpu().to(torch.float32).contiguous()
                if awq_np_ref.shape != (K,):
                    raise ValueError(
                        f"{tensor.name}: awq_scale shape {awq_np_ref.shape} != ({K},)")
                if is_moe:
                    for ei, b in enumerate(blobs[1:], 1):
                        awq_e = get_awq(b)
                        if awq_e is None:
                            raise ValueError(
                                f"{tensor.name}: expert 0 has awq but expert {ei} does not")
                        awq_e_t = awq_e.detach().cpu().to(torch.float32).contiguous()
                        if not torch.allclose(awq_e_t, awq_np_ref, atol=0.0, rtol=0.0):
                            raise ValueError(
                                f"{tensor.name}: expert {ei} awq_scale differs from expert 0 "
                                f"— MoE inference requires identical AWQ scale across experts.")
                writer.add_tensor(base + ".awq_scale", awq_np_ref.numpy())
                n_awq += 1
        elif tensor.name in fp8_blob_map:
            # FP8-tier blob: write as ML8_FP8 instead of copying bf16.
            fp8_path = fp8_blob_map[tensor.name]
            fp8_blob = torch.load(fp8_path, map_location="cpu", weights_only=False)
            e4m3_f32 = fp8_blob["e4m3"]          # float32 on e4m3 lattice [N, K]
            scale_fp16 = fp8_blob["scale"]        # fp16 [N, K//32]
            N_fp8, K_fp8 = e4m3_f32.shape
            packed_fp8 = pack_scaled_fp8_blocks(e4m3_f32, scale_fp16)
            writer.add_tensor(tensor.name, packed_fp8,
                              raw_dtype=GGMLQuantizationType.ML8_FP8)
            if base_fd >= 0:
                _advise_dontneed(base_fd, tensor.data_offset, tensor.n_bytes)
            n_fp8 += 1
            params_fp8 += N_fp8 * K_fp8
        else:
            # Pass-through non-MLP tensor. Copy bytes from the mmap into a
            # fresh allocation before handing to the writer (use_temp_file
            # mode spools to disk on add_tensor, so this clone is consumed
            # immediately, not held), then advise the kernel to drop the
            # source pages from cache so the next tensor doesn't pile on.
            cloned = np.ascontiguousarray(tensor.data)
            writer.add_tensor(tensor.name, cloned, raw_dtype=tensor.tensor_type)
            del cloned
            if base_fd >= 0:
                _advise_dontneed(base_fd, tensor.data_offset, tensor.n_bytes)
            n_copied += 1
            # Count un-quantized matmul weights against coverage. A 2-D+ tensor
            # named `.weight` is a linear/embedding weight that a real 4-bit
            # artifact would quantize; leaving it bf16 is exactly the gap we
            # refuse to ship silently. 1-D norms/biases are excluded.
            if tensor.name.endswith(".weight") and len(tensor.shape) >= 2:
                p = int(np.prod(tensor.shape))
                params_passthrough_weight += p
                passthrough_weights.append(
                    (tensor.name, p, tensor.tensor_type.name))

    print(
        f"[tensors] dense_ml8={n_ml8}, moe_ml8={n_moe}, fp8={n_fp8}, "
        f"centroids={n_centroids}, rotation={n_rot}, awq={n_awq}, "
        f"copied unchanged={n_copied}"
    )

    # ── Coverage report + guardrail ────────────────────────────────────────
    total_weight_params = params_ml8 + params_fp8 + params_passthrough_weight
    coverage, below_threshold, breakdown = evaluate_coverage(
        params_ml8, params_fp8, params_passthrough_weight, min_coverage)
    print(
        f"[coverage] quantized matmul weight: {coverage*100:.1f}% total "
        f"({total_weight_params:,} params) — "
        f"4-bit ml8: {breakdown['ml8']*100:.1f}% ({params_ml8:,}), "
        f"8-bit fp8: {breakdown['fp8']*100:.1f}% ({params_fp8:,}), "
        f"bf16: {breakdown['bf16']*100:.1f}% ({params_passthrough_weight:,})"
    )
    if passthrough_weights:
        passthrough_weights.sort(key=lambda t: t[1], reverse=True)
        print("[coverage] largest UN-quantized weight tensors:")
        for nm, p, dt in passthrough_weights[:10]:
            print(f"             {p/total_weight_params*100:5.1f}%  {nm}  "
                  f"({p:,} params, {dt})")
    # Record coverage in the artifact so downstream consumers (and the gauntlet)
    # can read it without re-deriving — and so "is this actually 4-bit?" is a
    # metadata lookup, not a surprise.
    writer.add_key_value("ml8.weight_coverage", float(coverage),
                         gguf.GGUFValueType.FLOAT32)
    writer.add_key_value("ml8.ml8_fraction", float(breakdown["ml8"]),
                         gguf.GGUFValueType.FLOAT32)
    writer.add_key_value("ml8.fp8_fraction", float(breakdown["fp8"]),
                         gguf.GGUFValueType.FLOAT32)

    if below_threshold:
        msg = (f"REFUSING to emit: only {coverage*100:.1f}% of matmul weight is "
               f"ml8-quantized (threshold {min_coverage*100:.0f}%). "
               f"{params_passthrough_weight:,} params would ship as bf16 — this "
               f"is NOT a 4-bit artifact. The top un-quantized tensors are listed "
               f"above (typically attn/SSM/lm_head/embed on a dense model whose "
               f"calibration only covered the FFN). Either extend calibration to "
               f"those tensors, or pass --allow-partial to ship this intentionally "
               f"(it will be clearly labelled partial).")
        if not allow_partial:
            writer.close()
            try:
                out_gguf.unlink()   # don't leave a half-written, mis-labellable file
            except OSError:
                pass
            raise SystemExit(f"[coverage] {msg}")
        print(f"[coverage] WARNING — proceeding under --allow-partial: {msg}")
        writer.add_key_value("ml8.partial", True, gguf.GGUFValueType.BOOL)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "out_path":         str(out_gguf),
        "n_ml8":            n_ml8,
        "n_moe":            n_moe,
        "n_fp8":            n_fp8,
        "n_centroids":      n_centroids,
        "n_rotation":       n_rot,
        "n_awq":            n_awq,
        "n_copied":         n_copied,
        "n_fields_copied":  n_fields_copied,
        "weight_coverage":  coverage,
        "params_ml8":       params_ml8,
        "params_fp8":       params_fp8,
        "params_bf16":      params_passthrough_weight,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-gguf", type=Path, required=True,
                   help="Existing f16/bf16 GGUF to extend with ml8 MLP tensors")
    p.add_argument("--calib-dir", type=Path, required=True,
                   help="Directory of ml8-4 per-layer .pt blobs")
    p.add_argument("--out-gguf", type=Path, required=True,
                   help="Output GGUF path")
    p.add_argument("--min-coverage", type=float, default=0.85,
                   help="Minimum fraction of matmul-weight params that must be "
                        "ml8-quantized, else refuse to emit (default 0.85). "
                        "Guards against silently shipping a mostly-bf16 '4-bit' "
                        "artifact (e.g. FFN-only calibration on a dense model).")
    p.add_argument("--allow-partial", action="store_true",
                   help="Emit even if coverage is below --min-coverage. The "
                        "artifact is tagged ml8.partial=true and a warning is "
                        "printed. Use only when a partial quant is intentional.")
    args = p.parse_args()
    summary = convert_to_ml8_gguf(args.base_gguf, args.calib_dir, args.out_gguf,
                                  min_coverage=args.min_coverage,
                                  allow_partial=args.allow_partial)
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
