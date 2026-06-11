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
    """packed [N, n_g*36] uint8 -> (indices uint8 [N,K], scales fp32 [N,K//64]).

    Indices are codebook entries in [0,15]; uint8 suffices (8x smaller than the
    int64 that .long() would allocate over all 136 ml8 tensors ≈ 21GB → 2.6GB).
    Consumers that need a long tensor convert per-forward with .long()."""
    if K % QK_ML8 != 0:
        raise ValueError(f"K={K} not divisible by QK_ML8={QK_ML8}")
    n_g = K // QK_ML8
    blocks = np.ascontiguousarray(packed).reshape(N, n_g, ML8_BLOCK_BYTES)
    scales = blocks[:, :, :4].copy().view('<f4').reshape(N, n_g)
    qs = blocks[:, :, 4:]
    idx = np.empty((N, n_g, QK_ML8), dtype=np.uint8)
    idx[:, :, 0::2] = qs & 0x0F
    idx[:, :, 1::2] = qs >> 4
    return (torch.from_numpy(idx.reshape(N, K)),
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

    ml8:    {tensor_name: {"indices": uint8 [N,K], "scales": fp32 [N,n_g],
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


def open_ml8_gguf(path, frozen_mode="all"):
    """STREAMING rehydrate: return (meta, iterator) over an ml8 GGUF.

    The iterator yields (kind, name, payload) one MAIN tensor at a time:
      kind "ml8"    -> payload {"indices" uint8 [N,K], "scales" fp32 [N,n_g],
                                "centroids" fp32 [n_g,16],
                                "rotation" None | {"h_a","a_dim","b_dim"}}
      kind "frozen" -> payload dequantized weight (fp32 for frozen_mode="all",
                                bf16 for "fp8")

    Sidecars (.centroids/.rotation_*; KBs each) are collected in a metadata
    pre-pass and attached to their base entry as it is yielded. The heavyweight
    main tensors are unpacked ONE AT A TIME so the host never materializes the
    whole trainer state at once (~5-6GB on the 4B — the 15GB-host rehydrate
    spike; consumers attach/install each payload to the GPU and drop it).

    frozen_mode controls which non-ml8 tensors are yielded as "frozen":
      * "all"  — ML8_FP8 dequantized to fp32 AND every BF16/F16/F32
                pass-through tensor cast to fp32 (legacy behavior).
      * "fp8"  — ONLY ML8_FP8 tensors, dequantized to BF16. Pass-through
                tensors are skipped — the HF student already carries them.
      * "none" — no frozen tensors yielded.
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

    def _base_of(name: str, suffix: str) -> str:
        return name[: -len(suffix)]

    # Sidecar pre-pass (small payloads only; main tensor data is NOT touched —
    # reader.tensors are mmap views, so nothing heavyweight is faulted in).
    centroids_by_base: dict[str, np.ndarray] = {}
    rot_h_a_by_base: dict[str, np.ndarray] = {}
    rot_meta_by_base: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        name = tensor.name
        if name.endswith(".centroids"):
            N, nbytes = _logical_N_bytes(tensor)
            centroids_by_base[_base_of(name, ".centroids")] = \
                _row_major_bytes(tensor, N, nbytes)
        elif name.endswith(".rotation_h_a"):
            rot_h_a_by_base[_base_of(name, ".rotation_h_a")] = \
                np.ascontiguousarray(tensor.data)
        elif name.endswith(".rotation_meta"):
            rot_meta_by_base[_base_of(name, ".rotation_meta")] = \
                np.ascontiguousarray(tensor.data).astype(np.int64).reshape(-1)

    def _stream():
        for tensor in reader.tensors:
            name = tensor.name
            ttype = tensor.tensor_type

            if (name.endswith(".centroids") or name.endswith(".rotation_h_a")
                    or name.endswith(".rotation_meta")):
                continue
            if name.endswith(".awq_scale"):
                print(f"[skip] {name}: awq_scale sidecar not consumed by trainer state")
                continue

            if ttype == GGMLQuantizationType.ML8_4:
                N, nbytes = _logical_N_bytes(tensor)
                K = nbytes // ML8_BLOCK_BYTES * QK_ML8
                packed = _row_major_bytes(tensor, N, nbytes)
                idx, scl = unpack_ml8_blocks(packed, N, K)
                entry = {"indices": idx, "scales": scl,
                         "centroids": None, "rotation": None}
                base = name[: -len(".weight")] if name.endswith(".weight") else name
                cent_u8 = centroids_by_base.get(base)
                if cent_u8 is None:
                    raise ValueError(
                        f"{name}: missing centroids sidecar for base {base!r}")
                entry["centroids"] = decode_centroids_fp8(cent_u8)
                h_a = rot_h_a_by_base.get(base)
                if h_a is not None:
                    rmeta = rot_meta_by_base.get(base)
                    a_dim = int(rmeta[0]) if rmeta is not None and rmeta.size >= 1 \
                        else int(h_a.shape[0])
                    b_dim = int(rmeta[1]) if rmeta is not None and rmeta.size >= 2 \
                        else None
                    entry["rotation"] = {
                        "h_a": torch.from_numpy(np.ascontiguousarray(h_a)).to(torch.float32),
                        "a_dim": a_dim,
                        "b_dim": b_dim,
                    }
                yield "ml8", name, entry
                continue

            if ttype == GGMLQuantizationType.ML8_FP8:
                if frozen_mode == "none":
                    continue
                N, nbytes = _logical_N_bytes(tensor)
                K = nbytes // _FP8_BLOCK_BYTES * _FP8_GROUP_SIZE
                packed = _row_major_bytes(tensor, N, nbytes)
                e4m3, scale = unpack_scaled_fp8_blocks(packed, N, K)
                # Expand per-group fp16 scale over the 32-wide groups, dequantize.
                scale_cols = scale.to(torch.float32).repeat_interleave(
                    _FP8_GROUP_SIZE, dim=1)
                dequant = (e4m3 * scale_cols).contiguous()
                if frozen_mode == "fp8":
                    dequant = dequant.to(torch.bfloat16)
                yield "frozen", name, dequant
                continue

            # Pass-through (BF16/F16/F32) — "all" mode only; see docstring.
            if frozen_mode != "all":
                continue
            arr = np.array(tensor.data, dtype=np.float32, copy=True)
            yield "frozen", name, torch.from_numpy(arr).contiguous()

    return meta, _stream()


def list_ml8_names(path):
    """Names of the ML8_4 main tensors in a GGUF (metadata scan, no unpack)."""
    import gguf
    from gguf import GGMLQuantizationType

    reader = gguf.GGUFReader(str(path))
    return [t.name for t in reader.tensors
            if t.tensor_type == GGMLQuantizationType.ML8_4]


def load_ml8_gguf(path, frozen_mode="all") -> Ml8State:
    """Rehydrate a FULL Ml8State (everything resident on host).

    Implemented on open_ml8_gguf's stream — kept for tests and small models;
    the act-replay trainer consumes the stream directly so the host holds one
    tensor at a time (see open_ml8_gguf).
    """
    meta, stream = open_ml8_gguf(path, frozen_mode=frozen_mode)
    st = Ml8State(meta=meta)
    for kind, name, payload in stream:
        if kind == "ml8":
            st.ml8[name] = payload
        else:
            st.frozen[name] = payload
    return st


def dequant_ml8_state(t) -> torch.Tensor:
    """Dequantize one ml8 entry to fp32 [N,K] per the ml8_io formula:
    W[r,c] = centroids[c//QK_ML8, indices[r,c]] * scales[r, c//QK_ML8].
    """
    idx = t["indices"].long()                # uint8 [N,K] -> long for indexing
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
