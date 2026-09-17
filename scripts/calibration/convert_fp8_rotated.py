#!/usr/bin/env python3
"""convert_fp8_rotated — data-free BF16 GGUF -> GGUF with ML8_FP8 weights in a
rotated basis, for tensor-parallel inference (design "ML8_FP8 + rotation under
tensor parallel", 2026-09-17).

No calibration data, no GPTQ. Every 2D GEMM weight in the role allowlist is
cast to GGML_TYPE_ML8_FP8 (per-32-element block: fp16 scale + 32 OCP e4m3fn
bytes, scale = absmax/448 — see scaled_fp8.py) in the ROTATED basis, plus
rotation sidecars. Everything else (norms, biases, ssm_conv1d, ...) is copied
verbatim.

Rotation convention (matches ml8_runtime.py's Ml8Linear and calibrate_ml8_paged.py):
    W_rot[n, :] = rotation.forward(W[n, :])          (applied to the GEMM weight, at convert time)
    x_rot       = rotation.forward(x)                (applied to the activation, at inference time)
    y = W_rot . x_rot = W . x                        (exact, by orthogonality of `rotation`)
The C++ ml8 op already implements this forward/inverse convention for
calibrated ml8 blobs, so no C++ change is needed to consume these tensors.

Role allowlist (by GGUF tensor name, e.g. blk.3.attn_output.weight):
  K-split (TP)  -> block_hadamard(b=local_b): attn_output, ffn_down, ssm_out
  N-split (TP)  -> kronecker_orth_sylvester:  attn_q, attn_k, attn_v, attn_qkv,
                                               attn_gate, ffn_gate, ffn_up, output
  no rotation, Q8_0: token_embd, ssm_alpha, ssm_beta
  everything else (norms, biases, ssm_conv1d, ssm_a, ssm_dt.bias, nextn.*):
      copied verbatim, type unchanged.

The GGUF tensor is [N rows, K cols] with K = ne[0] (GGUFReader.shape is ne
order, i.e. shape[0] == K). Rotation acts along K.

Streams one tensor at a time from an mmapped GGUFReader; RAM stays bounded
regardless of checkpoint size (a 27B-class bf16 GGUF is ~55 GB, host RAM here
is 15 GB). Rotation matmuls run on --device (default cuda:0) in fp32.
"""
from __future__ import annotations

import argparse
import mmap as _mmap_mod
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

from kronecker_rotation import (  # noqa: E402
    KroneckerRotation, BlockHadamardRotation, factor_for_dim, random_orthogonal,
    KRONECKER_ORTH_SYLVESTER_KIND_ID, BLOCK_HADAMARD_KIND_ID,
)
from scaled_fp8 import quantize_scaled_fp8  # noqa: E402
from ml8_to_gguf import pack_scaled_fp8_blocks, _FP8_GROUP_SIZE, _FP8_BLOCK_BYTES  # noqa: E402


# ─── Role allowlist ─────────────────────────────────────────────────────────
# Roles are the tensor name with the leading "blk.{L}." (if any) stripped and
# the trailing ".weight" stripped, e.g. "blk.3.attn_output.weight" -> "attn_output",
# "output.weight" -> "output", "token_embd.weight" -> "token_embd".
K_SPLIT_HADAMARD_ROLES = {"attn_output", "ffn_down", "ssm_out"}
N_SPLIT_KRONECKER_ROLES = {
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate",
    "ffn_gate", "ffn_up", "output",
}
Q8_0_ROLES = {"token_embd", "ssm_alpha", "ssm_beta"}

_SKIP_FIELDS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}



def _sidecar_base(weight_name: str) -> str:
    """Sidecar tensors are named `<weight name without .weight>.<suffix>`, e.g.
    `blk.3.attn_output.rotation_meta` — the C++ loader builds the name with
    tn(tensor_id, "rotation_meta", il) (llama-arch.cpp LLM_TN_IMPL::str), which
    replaces the ".weight" suffix rather than appending to it; ml8_to_gguf.py
    follows the same convention."""
    return weight_name[:-len(".weight")] if weight_name.endswith(".weight") else weight_name

def tensor_role(name: str) -> tuple[str | None, bool]:
    """Return (role, is_weight). role is None for non-'.weight' tensors
    (norms/biases/etc — always copied verbatim regardless of role table)."""
    rest = name.split(".", 2)[2] if name.startswith("blk.") else name
    if not rest.endswith(".weight"):
        return None, False
    return rest[: -len(".weight")], True


def classify_tensor(name: str, shape: tuple, first_nextn_layer: int | None = None) -> tuple[str, str | None]:
    """Return (action, role). action in {"rotate_hadamard", "rotate_kronecker",
    "q8_0", "copy"}. Falls back to "copy" for anything not 2D even if the role
    would otherwise match (defensive — every allowlisted role in this model is
    a 2D GEMM weight, but this keeps the converter honest if that ever changes).

    Layers at or past `first_nextn_layer` (the MTP / nextn draft block, e.g.
    blk.64 when block_count=65 and nextn_predict_layers=1) are loaded by the
    C++ side without the ml8 sidecar registration, so a rotated weight there
    would leave its sidecars unconsumed ("wrong number of tensors"). Their 2D
    GEMM weights become plain Q8_0 instead."""
    role, is_weight = tensor_role(name)
    if not is_weight or len(shape) != 2:
        return "copy", role
    if first_nextn_layer is not None and name.startswith("blk."):
        if int(name.split(".", 2)[1]) >= first_nextn_layer:
            return "q8_0", role
    if role in K_SPLIT_HADAMARD_ROLES:
        return "rotate_hadamard", role
    if role in N_SPLIT_KRONECKER_ROLES:
        return "rotate_kronecker", role
    if role in Q8_0_ROLES:
        return "q8_0", role
    return "copy", role


def _advise_dontneed(fd: int, offset: int, length: int) -> None:
    if not hasattr(os, "posix_fadvise") or fd < 0:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


# Default row-chunk size for GPU processing. A GEMM weight's rotation and
# quantization are both purely row-local (rotation.forward only mixes within
# a row's K elements; per-block FP8/Q8_0 scaling is per-row-per-group), so
# chunking along N (rows) is exact — see test_convert_chunked_matches_unchunked.
# Bounds peak GPU (and transient host) memory to ~chunk_rows * K * 4 bytes
# regardless of total tensor size: at 8192 rows the biggest K in this model
# (17408) is ~571 MB fp32, comfortably inside the R9700's 32 GB, vs. the ~4.9 GB
# single-shot fp32 materialization of token_embd/output.weight (1.27B elements)
# that OOM'd the first attempt at this (the GPU was also nearly full from
# unreleased prior-tensor allocations — torch.cuda.empty_cache() per chunk
# below addresses that side of it too).
_CHUNK_ROWS = 8192


def _bf16_rows_to_fp32_gpu(tensor, device: torch.device, row_start: int, row_end: int) -> torch.Tensor:
    """Widen rows [row_start:row_end) of a GGUFReader BF16 tensor to fp32 on `device`.

    `tensor.data` is a uint8 memmap view of the on-disk bytes; `.view(np.uint16)`
    is a zero-copy reinterpretation (no host allocation), and slicing rows out
    of that view is also zero-copy — only `np.ascontiguousarray` on the row
    slice actually materializes bytes, and only for this chunk.
    """
    if tensor.tensor_type != GGMLQuantizationType.BF16:
        raise ValueError(f"{tensor.name}: expected BF16 source, got {tensor.tensor_type.name}")
    u16_all = tensor.data.view(np.uint16)               # [N, K] zero-copy view of the mmap
    # np.array(..., copy=True) (not ascontiguousarray, which no-ops — and
    # leaves a read-only view — when the slice is already contiguous, as a
    # plain row-slice of a C-contiguous 2D array always is) to guarantee a
    # writable owned copy of just this chunk; torch.from_numpy requires that.
    chunk = np.array(u16_all[row_start:row_end], copy=True)
    t16 = torch.from_numpy(chunk).to(device=device)
    t32 = t16.to(torch.int32) << 16
    return t32.view(torch.float32).contiguous()


def _row_chunk_bounds(N: int, chunk_rows: int) -> list[tuple[int, int]]:
    """(start, end) row ranges covering [0, N), each of size chunk_rows except
    possibly the last — merged into the previous chunk if it would otherwise
    be a lone single row.

    That single-row case isn't just an aesthetic nit: ml8_to_gguf.pack_scaled_
    fp8_blocks (reused by _rotate_and_fp8) does
    `scale_fp16[:, b].detach().cpu().to(torch.float16).contiguous().view(torch.uint8)`,
    and PyTorch's `.contiguous()` is a documented no-op for a size-1 dim (it's
    trivially "contiguous" regardless of its stride) — so the stale non-unit
    stride from the `[:, b]` column-slice survives, and `.view(torch.uint8)`
    then rejects it (it checks the literal stride, not logical contiguity).
    Never handing that function a 1-row chunk sidesteps the bug entirely.
    """
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    bounds: list[list[int]] = []
    start = 0
    while start < N:
        end = min(start + chunk_rows, N)
        bounds.append([start, end])
        start = end
    if len(bounds) >= 2 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2][1] = bounds[-1][1]
        bounds.pop()
    return [(s, e) for s, e in bounds]


def _process_rotate_chunked(tensor, e: dict, device: torch.device, rotation,
                            chunk_rows: int = _CHUNK_ROWS) -> np.ndarray:
    """Rotate + scaled-FP8-quantize one GEMM weight, one row-chunk of the GPU
    at a time. Returns the fully assembled packed ML8_FP8 bytes (CPU numpy)."""
    K, N = e["shape"]
    n_blocks = K // _FP8_GROUP_SIZE
    out = np.empty((N, n_blocks * _FP8_BLOCK_BYTES), dtype=np.uint8)
    for start, end in _row_chunk_bounds(N, chunk_rows):
        w = _bf16_rows_to_fp32_gpu(tensor, device, start, end)
        out[start:end] = _rotate_and_fp8(w, rotation)
        del w
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _process_q8_0_chunked(tensor, e: dict, device: torch.device,
                          chunk_rows: int = _CHUNK_ROWS) -> np.ndarray:
    """Q8_0-quantize one weight, one row-chunk of the GPU at a time."""
    K, N = e["shape"]
    n_blocks = K // _FP8_GROUP_SIZE
    out = np.empty((N, n_blocks * _FP8_BLOCK_BYTES), dtype=np.uint8)
    for start, end in _row_chunk_bounds(N, chunk_rows):
        w = _bf16_rows_to_fp32_gpu(tensor, device, start, end)
        out[start:end] = _quantize_q8_0_gpu(w)
        del w
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _copy_field(writer: "gguf.GGUFWriter", name: str, field) -> None:
    types = field.types
    value = field.contents()
    primary = types[0]
    if primary == gguf.GGUFValueType.ARRAY:
        if len(types) < 2:
            raise ValueError(f"field {name!r}: ARRAY type without sub-type")
        writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub_type=types[1])
    else:
        writer.add_key_value(name, value, primary)


def _rotate_and_fp8(w: torch.Tensor, rotation, group_size: int = _FP8_GROUP_SIZE) -> np.ndarray:
    """Apply `rotation.forward` along K (last dim), quantize to scaled-FP8, pack."""
    w_rot = rotation.forward(w)
    q = quantize_scaled_fp8(w_rot, group_size=group_size)
    return pack_scaled_fp8_blocks(q["e4m3"], q["scale"])


def _round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Round-half-away-from-zero (matches gguf.quants.np_roundf), on whatever
    device `x` lives on. torch.round is round-half-to-even, which would only
    ever disagree at an exact .5 tie — never observed for real fp32 weight
    data, but we match the reference convention exactly rather than rely on
    that being true."""
    a = x.abs()
    floored = torch.floor(a)
    b = floored + torch.floor(2 * (a - floored))
    return torch.sign(x) * b


def _quantize_q8_0_gpu(w: torch.Tensor) -> np.ndarray:
    """Bit-exact-with-gguf.quants Q8_0 quantizer that keeps the (potentially
    huge, e.g. token_embd/output.weight = 1.27B elements) fp32 activation on
    `w`'s device the whole time — only the packed uint8 bytes (8.5 bits/elem)
    ever reach the host. See test_convert_fp8_rotated.py for the bit-exact
    comparison against gguf.quants.quantize(..., Q8_0)."""
    N, K = w.shape
    assert K % _FP8_GROUP_SIZE == 0, f"K={K} not divisible by {_FP8_GROUP_SIZE}"
    n_blocks = K // _FP8_GROUP_SIZE
    blocks = w.reshape(N, n_blocks, _FP8_GROUP_SIZE)
    d = blocks.abs().amax(dim=-1, keepdim=True) / 127.0
    inv_d = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    qs = _round_away_from_zero(blocks * inv_d).to(torch.int8)
    d16 = d.squeeze(-1).to(torch.float16)  # [N, n_blocks]

    d_bytes = d16.contiguous().cpu().numpy().view(np.uint8).reshape(N, n_blocks, 2)
    qs_bytes = qs.contiguous().cpu().numpy().view(np.uint8).reshape(N, n_blocks, _FP8_GROUP_SIZE)
    return np.concatenate([d_bytes, qs_bytes], axis=-1).reshape(N, n_blocks * _FP8_BLOCK_BYTES)


def _emission_specs(tensor, e: dict) -> list[dict]:
    """Return the list of {name, byte_shape, dtype, nbytes, raw_dtype} blobs
    this GGUF tensor expands to (main weight + optional sidecars), computed
    purely from shapes/metadata — no tensor data is read or computed. Used to
    register tensor_info (pass 1) and must be produced identically (same
    order, same nbytes) as the actual data written in pass 2."""
    action = e["action"]
    if action in ("rotate_kronecker", "rotate_hadamard"):
        K, N = e["shape"]
        n_blocks = K // _FP8_GROUP_SIZE
        specs = [{
            "name": tensor.name,
            "byte_shape": (N, n_blocks * _FP8_BLOCK_BYTES),
            "dtype": np.uint8,
            "nbytes": N * n_blocks * _FP8_BLOCK_BYTES,
            "raw_dtype": GGMLQuantizationType.ML8_FP8,
        }]
        if action == "rotate_kronecker":
            a = e["a"]
            specs.append({
                "name": _sidecar_base(tensor.name) + ".rotation_h_a",
                "byte_shape": (a, a), "dtype": np.float32,
                "nbytes": a * a * 4, "raw_dtype": None,
            })
        specs.append({
            "name": _sidecar_base(tensor.name) + ".rotation_meta",
            "byte_shape": (4,), "dtype": np.int32,
            "nbytes": 16, "raw_dtype": None,
        })
        return specs
    if action == "q8_0":
        K, N = e["shape"]
        n_blocks = K // _FP8_GROUP_SIZE
        return [{
            "name": tensor.name,
            "byte_shape": (N, n_blocks * _FP8_BLOCK_BYTES),
            "dtype": np.uint8,
            "nbytes": N * n_blocks * _FP8_BLOCK_BYTES,
            "raw_dtype": GGMLQuantizationType.Q8_0,
        }]
    # copy: byte-identical to the source tensor's own on-disk layout.
    return [{
        "name": tensor.name,
        "byte_shape": tensor.data.shape,
        "dtype": tensor.data.dtype,
        "nbytes": tensor.n_bytes,
        "raw_dtype": tensor.tensor_type,
    }]


def _first_nextn_layer(reader: "gguf.GGUFReader") -> int | None:
    """block_count - nextn_predict_layers from the GGUF KV (any arch prefix), or
    None when the model has no nextn block."""
    block_count = None
    nextn = 0
    for key, field in reader.fields.items():
        if key.endswith(".block_count"):
            block_count = int(field.parts[field.data[0]][0])
        elif key.endswith(".nextn_predict_layers"):
            nextn = int(field.parts[field.data[0]][0])
    if block_count is None or nextn <= 0:
        return None
    return block_count - nextn


def build_plan(reader: "gguf.GGUFReader", rotation_seed: int, local_b: int, max_b: int) -> list[dict]:
    """Compute the per-tensor conversion plan (name, shape, action, rotation
    kind/a/b) without doing any tensor math. Kronecker seeds are assigned by a
    stable index over the sorted names of kronecker-rotated tensors, so the
    plan (and the resulting rotation) doesn't depend on GGUF tensor order."""
    first_nextn = _first_nextn_layer(reader)
    kron_names = sorted(
        t.name for t in reader.tensors
        if classify_tensor(t.name, tuple(int(s) for s in t.shape), first_nextn)[0] == "rotate_kronecker"
    )
    kron_index = {name: i for i, name in enumerate(kron_names)}

    plan = []
    for t in reader.tensors:
        shape = tuple(int(s) for s in t.shape)  # ne order: shape[0] == K for 2D
        action, role = classify_tensor(t.name, shape, first_nextn)
        entry = {"name": t.name, "shape": shape, "action": action, "role": role}
        if action == "rotate_kronecker":
            K = shape[0]
            a, b = factor_for_dim(K, max_b=max_b)
            entry.update(kind="kronecker_orth_sylvester", a=a, b=b,
                         seed=rotation_seed + kron_index[t.name])
        elif action == "rotate_hadamard":
            K = shape[0]
            if K % local_b != 0:
                raise ValueError(
                    f"{t.name}: K={K} not divisible by --local-b={local_b} "
                    f"(block_hadamard requires this)")
            entry.update(kind="block_hadamard", a=K // local_b, b=local_b)
        plan.append(entry)
    return plan


def print_plan(plan: list[dict]) -> None:
    counts: dict[str, int] = {}
    for e in plan:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    print(f"[dry-run] {len(plan)} tensors total")
    for action, n in sorted(counts.items()):
        print(f"  {action}: {n}")
    print("[dry-run] per-tensor plan:")
    for e in plan:
        if e["action"] in ("rotate_kronecker", "rotate_hadamard"):
            print(f"  {e['name']:45s} shape={e['shape']} action={e['action']:16s} "
                  f"kind={e['kind']} a={e['a']} b={e['b']}"
                  + (f" seed={e['seed']}" if "seed" in e else ""))
        else:
            print(f"  {e['name']:45s} shape={e['shape']} action={e['action']}")


def convert(src: Path, out: Path, rotation_seed: int, device_str: str,
           local_b: int, max_b: int, chunk_rows: int = _CHUNK_ROWS) -> dict:
    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    print(f"[base] {src}  arch={arch!r}  fields={len(reader.fields)}  tensors={len(reader.tensors)}")

    plan = build_plan(reader, rotation_seed, local_b, max_b)
    plan_by_name = {e["name"]: e for e in plan}

    device = torch.device(device_str)

    # Memory-pressure mitigation: sequential madvise on the reader's mmap +
    # per-tensor POSIX_FADV_DONTNEED, same trick as ml8_to_gguf.py — keeps
    # the source GGUF's pages from piling up in cache on a 15 GB host.
    base_fd = -1
    try:
        underlying = reader.data._mmap
        if hasattr(underlying, "madvise"):
            try:
                underlying.madvise(_mmap_mod.MADV_SEQUENTIAL)
            except (OSError, ValueError):
                pass
        try:
            base_fd = os.open(str(src), os.O_RDONLY)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(base_fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except OSError:
            base_fd = -1
    except AttributeError:
        pass

    # use_temp_file=False (default): with the two-pass registration below we
    # write each tensor's bytes straight to the output file via
    # write_tensor_data as it is computed, instead of spooling every
    # converted tensor (main weight + sidecars, ~29 GB total) through a
    # temp-file/page-cache buffer ahead of a single final copy — that spool
    # is what blew the 10 GB cgroup cap on a 15 GB host on the first attempt.
    writer = gguf.GGUFWriter(str(out), arch=arch)

    for name, field in reader.fields.items():
        if name in _SKIP_FIELDS:
            continue
        _copy_field(writer, name, field)
    writer.add_key_value("fp8rot.format_version", 1, gguf.GGUFValueType.UINT32)
    writer.add_key_value("fp8rot.rotation_seed", int(rotation_seed), gguf.GGUFValueType.INT32)
    writer.add_key_value("fp8rot.local_b", int(local_b), gguf.GGUFValueType.INT32)

    # ── Pass 1: register every tensor's (name, shape, dtype, nbytes) — no
    # tensor data is read or computed here, so this is a cheap metadata-only
    # walk of the plan. add_tensor_info requires this to happen before the
    # output file is opened (write_header_to_file), and write_tensor_data
    # (pass 2) consumes these registrations strictly in insertion order.
    n_fields = sum(1 for name in reader.fields if name not in _SKIP_FIELDS) + 3
    emission_order: list[dict] = []
    for tensor in reader.tensors:
        e = plan_by_name[tensor.name]
        for spec in _emission_specs(tensor, e):
            writer.add_tensor_info(
                spec["name"], spec["byte_shape"], np.dtype(spec["dtype"]),
                spec["nbytes"], raw_dtype=spec["raw_dtype"])
            emission_order.append(spec)
    print(f"[fields] copied {n_fields - 3} + 3 fp8rot markers; "
          f"[pass1] registered {len(emission_order)} tensor blobs "
          f"({len(reader.tensors)} source tensors)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ── Pass 2: stream actual bytes, one GGUF tensor's worth of GPU work at a
    # time, straight into the file via write_tensor_data (no spooling).
    counts = {"rotate_hadamard": 0, "rotate_kronecker": 0, "q8_0": 0, "copy": 0}
    t_start = time.time()
    for i, tensor in enumerate(reader.tensors):
        e = plan_by_name[tensor.name]
        action = e["action"]
        if action == "rotate_kronecker":
            K, N = e["shape"]
            h_a = random_orthogonal(e["a"], seed=e["seed"]).to(device=device, dtype=torch.float32)
            rot = KroneckerRotation(h_a=h_a, b_dim=e["b"])
            packed = _process_rotate_chunked(tensor, e, device, rot, chunk_rows=chunk_rows)
            writer.write_tensor_data(packed)
            meta = np.array([e["a"], e["b"], K, KRONECKER_ORTH_SYLVESTER_KIND_ID], dtype=np.int32)
            writer.write_tensor_data(h_a.detach().cpu().contiguous().numpy())
            writer.write_tensor_data(meta)
            del h_a
            if device.type == "cuda":
                torch.cuda.empty_cache()
            counts["rotate_kronecker"] += 1
        elif action == "rotate_hadamard":
            K, N = e["shape"]
            rot = BlockHadamardRotation(in_features=K, b_dim=e["b"])
            packed = _process_rotate_chunked(tensor, e, device, rot, chunk_rows=chunk_rows)
            writer.write_tensor_data(packed)
            meta = np.array([e["a"], e["b"], K, BLOCK_HADAMARD_KIND_ID], dtype=np.int32)
            writer.write_tensor_data(meta)
            counts["rotate_hadamard"] += 1
        elif action == "q8_0":
            packed = _process_q8_0_chunked(tensor, e, device, chunk_rows=chunk_rows)
            writer.write_tensor_data(packed)
            counts["q8_0"] += 1
        else:
            cloned = np.ascontiguousarray(tensor.data)
            writer.write_tensor_data(cloned)
            del cloned
            counts["copy"] += 1

        if base_fd >= 0:
            _advise_dontneed(base_fd, tensor.data_offset, tensor.n_bytes)

        if (i + 1) % 25 == 0 or (i + 1) == len(reader.tensors):
            elapsed = time.time() - t_start
            print(f"[progress] {i+1}/{len(reader.tensors)} tensors "
                  f"({elapsed:.1f}s elapsed) — {tensor.name} -> {action}")

    writer.close()

    out_size = out.stat().st_size
    print(
        f"[done] wrote {out} ({out_size/1e9:.2f} GB) — "
        f"rotate_hadamard={counts['rotate_hadamard']} "
        f"rotate_kronecker={counts['rotate_kronecker']} "
        f"q8_0={counts['q8_0']} copy={counts['copy']} "
        f"({time.time()-t_start:.1f}s)"
    )
    return {"out_path": str(out), "out_size": out_size, **counts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True, help="Source BF16 GGUF")
    p.add_argument("--out", type=Path, required=True, help="Output GGUF path")
    p.add_argument("--rotation-seed", type=int, default=0,
                   help="Base seed for kronecker h_a matrices (default 0)")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device for rotation matmuls (default cuda:0)")
    p.add_argument("--local-b", type=int, default=128,
                   help="block_hadamard block size along K (default 128)")
    p.add_argument("--max-b", type=int, default=1024,
                   help="kronecker factor_for_dim max_b (default 1024)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-tensor plan and exit without converting")
    p.add_argument("--chunk-rows", type=int, default=_CHUNK_ROWS,
                   help="Row-chunk size for GPU rotation/quantization work "
                        f"(default {_CHUNK_ROWS}) — bounds peak GPU memory "
                        "regardless of tensor size (e.g. the 1.27B-element "
                        "token_embd/output.weight); chunking along rows is "
                        "exact since rotation only mixes within a row's K")
    args = p.parse_args()

    reader = gguf.GGUFReader(args.src)
    plan = build_plan(reader, args.rotation_seed, args.local_b, args.max_b)
    print_plan(plan)
    if args.dry_run:
        return

    convert(args.src, args.out, args.rotation_seed, args.device, args.local_b, args.max_b,
           chunk_rows=args.chunk_rows)


if __name__ == "__main__":
    main()
