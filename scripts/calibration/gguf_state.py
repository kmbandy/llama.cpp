# gguf_state.py
"""Rehydrate act-replay trainer state from an ml8 GGUF (exact inverse of ml8_to_gguf packing)."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))
from ml8_to_gguf import QK_ML8, ML8_BLOCK_BYTES, _FP8_GROUP_SIZE, _FP8_BLOCK_BYTES

def unpack_ml8_blocks(packed, N, K):
    """packed [N, n_g*36] uint8 -> (indices long [N,K], scales fp32 [N,K//64])."""
    if K % QK_ML8 != 0:
        raise ValueError(f"K={K} not divisible by QK_ML8={QK_ML8}")
    n_g = K // QK_ML8
    blocks = np.ascontiguousarray(packed).reshape(N, n_g, ML8_BLOCK_BYTES)
    scales = blocks[:, :, :4].copy().view('<f4').reshape(N, n_g)
    qs = blocks[:, :, 4:]
    idx = np.empty((N, n_g, QK_ML8), dtype=np.uint8)
    idx[:, :, 0::2] = qs & 0x0F
    idx[:, :, 1::2] = qs >> 4
    return (torch.from_numpy(idx.reshape(N, K).astype(np.int64)),
            torch.from_numpy(scales.astype(np.float32)))

def unpack_scaled_fp8_blocks(packed, N, K):
    """packed [N, n_b*34] uint8 -> (e4m3 fp32 [N,K], scale fp16 [N,K//32])."""
    if K % _FP8_GROUP_SIZE != 0:
        raise ValueError(f"K={K} not divisible by FP8 group size {_FP8_GROUP_SIZE}")
    n_b = K // _FP8_GROUP_SIZE
    blocks = np.ascontiguousarray(packed).reshape(N, n_b, _FP8_BLOCK_BYTES)
    scale = torch.from_numpy(blocks[:, :, :2].copy()).view(torch.float16).reshape(N, n_b)
    qs = torch.from_numpy(blocks[:, :, 2:].copy()).view(torch.float8_e4m3fn)
    return qs.to(torch.float32).reshape(N, K), scale

def decode_centroids_fp8(cent_u8):
    return torch.from_numpy(np.ascontiguousarray(cent_u8)).view(torch.float8_e4m3fn).to(torch.float32)


# ─── Rehydration ───────────────────────────────────────────────────────────


# Frozen-tensor footprint depends on load_ml8_gguf's frozen_mode: "all" keeps
# fp8+passthrough as fp32 (~2x bf16 model size); "fp8" keeps only ML8_FP8 as bf16
# (the act-replay default on a 15GB host); "none" keeps nothing.
@dataclass
class Ml8State:
    """Trainer state rehydrated from an ml8 GGUF.

    ml8:    {tensor_name: {"indices": long [N,K], "scales": fp32 [N,n_g],
                           "centroids": fp32 [n_g,16],
                           "rotation": None | {"h_a": fp32 [a,a], "a_dim", "b_dim"}}}
    frozen: {tensor_name: [N,K]} — ML8_FP8 dequantized (e4m3*scale, expanded over
            groups; fp32 in "all" mode, bf16 in "fp8" mode) plus, in "all" mode,
            any BF16/F16/F32 pass-through tensor cast to fp32. See frozen_mode.
    meta:   {"arch", "block_count", ...} — cheap scalar metadata.
    """
    ml8: dict = field(default_factory=dict)
    frozen: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _logical_N_bytes(tensor):
    """Return (N, bytes_per_row) for a quantized tensor.

    gguf-py reshapes tensor.data to [N, bytes_per_row] (row-major, N rows) and
    reports tensor.shape in reversed GGUF order so shape[-1] == N. We derive
    everything from tensor.data.shape (already [N, bytes]) and cross-check N
    against shape[-1] to catch any reader behavior change.
    """
    data = tensor.data
    if data.ndim == 2:
        N, nbytes = int(data.shape[0]), int(data.shape[1])
    elif data.ndim == 1:
        # Flat data fallback: derive N from reversed shape, infer bytes.
        N = int(tensor.shape[-1])
        if N == 0 or data.shape[0] % N != 0:
            raise ValueError(
                f"{tensor.name}: cannot reshape flat data {data.shape} with N={N}")
        nbytes = int(data.shape[0]) // N
    else:
        raise ValueError(f"{tensor.name}: unexpected data.ndim={data.ndim}")
    shape_N = int(tensor.shape[-1])
    if shape_N != N:
        raise ValueError(
            f"{tensor.name}: N from data ({N}) != shape[-1] ({shape_N}); "
            f"shape={tuple(int(s) for s in tensor.shape)} data.shape={data.shape}")
    return N, nbytes


def _row_major_bytes(tensor, N, nbytes):
    """tensor.data as a contiguous [N, nbytes] uint8 array."""
    # always copy — tensor.data is a view into the reader's mmap; consumers may outlive the reader.
    arr = np.ascontiguousarray(tensor.data).copy().view(np.uint8).reshape(N, nbytes)
    return arr


def load_ml8_gguf(path, frozen_mode="all") -> Ml8State:
    """Rehydrate an Ml8State from an ml8 GGUF in one pass over reader.tensors.

    First collect ML8_4 main tensors, then attach .centroids / .rotation_h_a /
    .rotation_meta sidecars to their base. `.awq_scale` sidecars are skipped with
    a warning (the trainer does not consume them yet).

    frozen_mode controls what lands in `frozen`:
      * "all"  (default) — ML8_FP8 dequantized to fp32 AND every BF16/F16/F32
                pass-through tensor cast to fp32 (legacy behavior).
      * "fp8"  — ONLY ML8_FP8 tensors, dequantized and stored as BF16 (halves
                their RAM vs fp32; the host has only 15GB). Pass-through tensors
                are skipped entirely — the HF student already carries those bf16
                weights, so materializing them here is pure RAM tax.
      * "none" — nothing frozen (empty dict).
    """
    import gguf
    from gguf import GGMLQuantizationType

    if frozen_mode not in ("all", "fp8", "none"):
        raise ValueError(f"frozen_mode must be 'all'|'fp8'|'none', got {frozen_mode!r}")

    reader = gguf.GGUFReader(str(path))

    arch = reader.fields["general.architecture"].contents() \
        if "general.architecture" in reader.fields else None
    meta = {"arch": arch}
    if arch is not None:
        bc = reader.fields.get(f"{arch}.block_count")
        if bc is not None:
            meta["block_count"] = int(bc.contents())

    st = Ml8State(meta=meta)

    # Sidecar payloads keyed by base name (tensor.name minus ".weight").
    centroids_by_base: dict[str, np.ndarray] = {}
    rot_h_a_by_base: dict[str, np.ndarray] = {}
    rot_meta_by_base: dict[str, np.ndarray] = {}

    def _base_of(name: str, suffix: str) -> str:
        return name[: -len(suffix)]

    for tensor in reader.tensors:
        name = tensor.name
        ttype = tensor.tensor_type

        if name.endswith(".centroids"):
            N, nbytes = _logical_N_bytes(tensor)
            centroids_by_base[_base_of(name, ".centroids")] = \
                _row_major_bytes(tensor, N, nbytes)
            continue
        if name.endswith(".rotation_h_a"):
            rot_h_a_by_base[_base_of(name, ".rotation_h_a")] = \
                np.ascontiguousarray(tensor.data)
            continue
        if name.endswith(".rotation_meta"):
            rot_meta_by_base[_base_of(name, ".rotation_meta")] = \
                np.ascontiguousarray(tensor.data).astype(np.int64).reshape(-1)
            continue
        if name.endswith(".awq_scale"):
            print(f"[skip] {name}: awq_scale sidecar not consumed by trainer state")
            continue

        if ttype == GGMLQuantizationType.ML8_4:
            N, nbytes = _logical_N_bytes(tensor)
            K = nbytes // ML8_BLOCK_BYTES * QK_ML8
            packed = _row_major_bytes(tensor, N, nbytes)
            idx, scl = unpack_ml8_blocks(packed, N, K)
            st.ml8[name] = {"indices": idx, "scales": scl,
                            "centroids": None, "rotation": None}
            continue

        if ttype == GGMLQuantizationType.ML8_FP8:
            if frozen_mode == "none":
                continue
            N, nbytes = _logical_N_bytes(tensor)
            K = nbytes // _FP8_BLOCK_BYTES * _FP8_GROUP_SIZE
            packed = _row_major_bytes(tensor, N, nbytes)
            e4m3, scale = unpack_scaled_fp8_blocks(packed, N, K)
            # Expand per-group fp16 scale over the 32-wide groups and dequantize.
            n_b = K // _FP8_GROUP_SIZE
            scale_cols = scale.to(torch.float32).repeat_interleave(_FP8_GROUP_SIZE, dim=1)
            dequant = (e4m3 * scale_cols).contiguous()
            # "fp8" mode stores BF16 (halves RAM vs fp32; host has only 15GB).
            if frozen_mode == "fp8":
                dequant = dequant.to(torch.bfloat16)
            st.frozen[name] = dequant
            continue

        # Pass-through (BF16/F16/F32). In "fp8"/"none" we skip these entirely —
        # the HF student already carries those bf16 weights, so re-materializing
        # them here is pure RAM tax (~10GB on a 4B model). Only "all" keeps them.
        if frozen_mode != "all":
            continue
        # cast to fp32. tensor.data is already the dequantized ndarray for
        # F32/F16/BF16 (gguf-py widens BF16/F16 → float32).
        arr = np.array(tensor.data, dtype=np.float32, copy=True)
        st.frozen[name] = torch.from_numpy(arr).contiguous()

    # Attach sidecars to their ML8_4 base tensors.
    for name, entry in st.ml8.items():
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        cent_u8 = centroids_by_base.get(base)
        if cent_u8 is None:
            raise ValueError(f"{name}: missing centroids sidecar for base {base!r}")
        entry["centroids"] = decode_centroids_fp8(cent_u8)

        h_a = rot_h_a_by_base.get(base)
        if h_a is not None:
            rmeta = rot_meta_by_base.get(base)
            a_dim = int(rmeta[0]) if rmeta is not None and rmeta.size >= 1 else int(h_a.shape[0])
            b_dim = int(rmeta[1]) if rmeta is not None and rmeta.size >= 2 else None
            entry["rotation"] = {
                "h_a": torch.from_numpy(np.ascontiguousarray(h_a)).to(torch.float32),
                "a_dim": a_dim,
                "b_dim": b_dim,
            }

    return st


def dequant_ml8_state(t) -> torch.Tensor:
    """Dequantize one ml8 entry to fp32 [N,K] per the ml8_io formula:
    W[r,c] = centroids[c//QK_ML8, indices[r,c]] * scales[r, c//QK_ML8].
    """
    idx = t["indices"]                       # long [N,K]
    scales = t["scales"]                     # fp32 [N,n_g]
    cent = t["centroids"]                    # fp32 [n_g,16]
    N, K = idx.shape
    gidx = torch.arange(K) // QK_ML8         # [K]
    W = cent[gidx, idx] * scales[:, gidx]    # [N,K] * [N,K]
    return W


# ─── Bit-equality gate ──────────────────────────────────────────────────────


def _bitcheck(path) -> None:
    import gguf
    from gguf import GGMLQuantizationType
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8

    reader = gguf.GGUFReader(str(path))
    n_ml8 = 0
    n_fp8 = 0
    n_pass = 0
    n_cent = 0

    # Index ML8_4 tensors and centroid sidecars for cross-checking.
    for tensor in reader.tensors:
        name = tensor.name
        ttype = tensor.tensor_type

        if ttype == GGMLQuantizationType.ML8_4:
            N, nbytes = _logical_N_bytes(tensor)
            K = nbytes // ML8_BLOCK_BYTES * QK_ML8
            raw = _row_major_bytes(tensor, N, nbytes)
            idx, scl = unpack_ml8_blocks(raw, N, K)
            repacked = pack_ml8_blocks(idx.to(torch.int8), scl)
            if repacked.shape != raw.shape or not np.array_equal(repacked, raw):
                raise AssertionError(
                    f"{name}: re-pack mismatch ({repacked.shape} vs {raw.shape})")
            n_ml8 += 1
        elif name.endswith(".centroids"):
            N, nbytes = _logical_N_bytes(tensor)
            raw = _row_major_bytes(tensor, N, nbytes)
            cent = decode_centroids_fp8(raw)
            again = cast_centroids_to_fp8(cent)
            if not np.array_equal(np.ascontiguousarray(again).reshape(raw.shape), raw):
                raise AssertionError(f"{name}: centroid e4m3 roundtrip not idempotent")
            n_cent += 1
        elif name.endswith(".rotation_h_a") or name.endswith(".rotation_meta") \
                or name.endswith(".awq_scale"):
            continue
        elif ttype == GGMLQuantizationType.ML8_FP8:
            n_fp8 += 1
        else:
            n_pass += 1

    print(f"bitcheck OK on {n_ml8} ml8 tensors, {n_fp8} fp8 frozen, {n_pass} passthrough")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gguf", type=Path, required=True, help="ml8 GGUF to inspect")
    p.add_argument("--bitcheck", action="store_true",
                   help="Re-pack every ML8_4 tensor and byte-compare against the "
                        "raw GGUF bytes; assert e4m3-roundtrip(centroids) idempotent.")
    args = p.parse_args()
    if args.bitcheck:
        _bitcheck(args.gguf)
    else:
        st = load_ml8_gguf(args.gguf)
        print(f"loaded: {len(st.ml8)} ml8, {len(st.frozen)} frozen, meta={st.meta}")


if __name__ == "__main__":
    main()
