#!/usr/bin/env python3
"""Re-encode the bf16 NON-EXPERT path of an ml8 GGUF to Q8_0 — the MAD-256 #2
size win. Leaves the ML8_4_SOA routed experts (and their sidecars) BYTE-FOR-BYTE
untouched; only the bf16 attn/ssm/router/embed/shared-expert weights become Q8_0.

WHY: bpv accounting (KG cd8fa620) showed our ml8_4_soa is 24.76 GB vs UD-Q4_K_XL
22.84 GB, and the ENTIRE 1.92 GB deficit is the bf16 non-expert path (16 bpv vs
UD's Q8_0 8.5 bpv). Matching UD's Q8_0 on non-experts saves ~2.3 GB → ~22.46 GB,
UNDER UD, experts untouched. Our non-experts are bf16 = HIGHER precision than
UD's Q8_0, so Q8_0 is ~free on PPL (Q8_0 vs bf16 is near-lossless). This is the
guaranteed size axis win, decoupled from the expert-quality (#1) work.

RULE per tensor:
  - name contains "_exps"  → ml8 routed experts + sidecars: copy raw, untouched.
  - type==BF16, ndim>=2, last logical dim % 32 == 0 → quantize to Q8_0.
    (Catches attn/ssm/router/embed/output AND shared experts "_shexp" — UD Q8s
     those too. ne0 % 32 == 0 is the Q8_0 block constraint.)
  - else (norms F32/1-D, odd dims) → copy raw, untouched.

OOM-safe (host 15 GB RAM, file 24 GB): GGUFWriter use_temp_file=True spools to
the OUTPUT dir (NVMe), never /tmp; per-tensor POSIX_FADV_DONTNEED drops source
page cache. Mirrors ml8_to_gguf.py's large-base handling.

Usage:
  python3 requantize_nonexpert.py \
      --in  /home/kmbandy/models/Qwen3.6-35B-A3B-ml8_4_soa.gguf \
      --out /home/kmbandy/models/Qwen3.6-35B-A3B-ml8_4_soa-q8ne.gguf
  python3 requantize_nonexpert.py --self-test     # tiny synthetic round-trip gate
"""
from __future__ import annotations
import argparse
import mmap as _mmap_mod
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf import quants  # noqa: E402

_SKIP_FIELDS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
_QK8 = 32  # Q8_0 block size along ne0


def _advise_dontneed(fd: int, offset: int, length: int) -> None:
    if not hasattr(os, "posix_fadvise") or fd < 0:
        return
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def _copy_field(writer: gguf.GGUFWriter, name: str, field) -> None:
    types = field.types
    value = field.contents()
    primary = types[0]
    if primary == gguf.GGUFValueType.ARRAY:
        writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub_type=types[1])
    else:
        writer.add_key_value(name, value, primary)


def _should_q8(tensor) -> bool:
    """True iff this bf16 tensor is a non-expert weight we can Q8_0-encode."""
    if "_exps" in tensor.name:           # ml8 routed experts + sidecars — never touch
        return False
    if T(tensor.tensor_type) != T.BF16:  # only bf16 weights (norms are F32/1-D)
        return False
    shape = [int(s) for s in tensor.shape]   # GGUF ne-order; ne0 = fastest = row length
    if len(shape) < 2:
        return False
    return shape[0] % _QK8 == 0          # Q8_0 quantizes along ne0


def _bf16_to_f32_rows(tensor) -> np.ndarray:
    """Reader exposes bf16 as uint8 [ne1.., ne0*2]; return f32 [ne1.., ne0]."""
    raw = np.ascontiguousarray(tensor.data)
    return torch.from_numpy(raw).view(torch.bfloat16).float().numpy()


def convert(in_path: Path, out_path: Path) -> dict:
    reader = gguf.GGUFReader(in_path)
    arch = reader.fields["general.architecture"].contents()
    print(f"[in] {in_path}  arch={arch!r}  fields={len(reader.fields)}  tensors={len(reader.tensors)}")

    # Drop source pages as we go (RAM-constrained host).
    base_fd = -1
    try:
        underlying = reader.data._mmap
        if hasattr(underlying, "madvise"):
            try:
                underlying.madvise(_mmap_mod.MADV_SEQUENTIAL)
            except (OSError, ValueError):
                pass
        base_fd = os.open(str(in_path), os.O_RDONLY)
    except (AttributeError, OSError):
        base_fd = -1

    out_dir = out_path.parent if str(out_path.parent) else Path(".")
    tempfile.tempdir = str(out_dir)   # spool on NVMe, never /tmp
    writer = gguf.GGUFWriter(str(out_path), arch=arch, use_temp_file=True)

    for name, field in reader.fields.items():
        if name not in _SKIP_FIELDS:
            _copy_field(writer, name, field)
    writer.add_key_value("ml8.nonexpert_quant", "Q8_0", gguf.GGUFValueType.STRING)

    n_q8 = n_copy = 0
    bytes_before = bytes_after = 0
    for tensor in reader.tensors:
        if _should_q8(tensor):
            f32 = _bf16_to_f32_rows(tensor)               # [.., ne0]
            q = quants.quantize(f32, T.Q8_0)              # uint8 packed [.., ne0/32*34]
            # No raw_shape: the writer derives the logical ne from the byte
            # shape via quant_shape_from_byte_shape, same as the copy path.
            writer.add_tensor(tensor.name, q, raw_dtype=T.Q8_0)
            bytes_before += tensor.n_bytes
            bytes_after += q.nbytes
            n_q8 += 1
            del f32, q
        else:
            cloned = np.ascontiguousarray(tensor.data)
            writer.add_tensor(tensor.name, cloned, raw_dtype=T(tensor.tensor_type))
            bytes_before += tensor.n_bytes
            bytes_after += tensor.n_bytes
            del cloned
            n_copy += 1
        if base_fd >= 0:
            _advise_dontneed(base_fd, tensor.data_offset, tensor.n_bytes)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if base_fd >= 0:
        os.close(base_fd)

    print(f"[tensors] Q8_0-encoded={n_q8}  copied-unchanged={n_copy}")
    print(f"[size] touched tensors {bytes_before/1e9:.2f} GB → {bytes_after/1e9:.2f} GB "
          f"(saved {(bytes_before-bytes_after)/1e9:.2f} GB on those)")
    return {"n_q8": n_q8, "n_copy": n_copy, "saved_gb": (bytes_before - bytes_after) / 1e9}


def verify(in_path: Path, out_path: Path, n_check: int = 6) -> None:
    """Gate: experts byte-identical, non-experts now Q8_0 with sane dequant."""
    a = gguf.GGUFReader(in_path)
    b = gguf.GGUFReader(out_path)
    bt = {t.name: t for t in b.tensors}
    assert len(a.tensors) == len(b.tensors), f"tensor count drift {len(a.tensors)}≠{len(b.tensors)}"
    checked_q8 = checked_exp = 0
    cos_min = 1.0
    for t in a.tensors:
        assert t.name in bt, f"missing {t.name} in output"
        ot = bt[t.name]
        assert tuple(t.shape) == tuple(ot.shape), f"{t.name} shape drift {tuple(t.shape)}≠{tuple(ot.shape)}"
        if "_exps" in t.name:
            # expert + sidecars must be byte-identical (type + bytes)
            assert t.tensor_type == ot.tensor_type, f"{t.name} expert type changed!"
            if checked_exp < n_check:
                assert np.array_equal(np.asarray(t.data), np.asarray(ot.data)), \
                    f"{t.name} expert bytes changed!"
                checked_exp += 1
        elif _should_q8(t):
            assert T(ot.tensor_type) == T.Q8_0, f"{t.name} expected Q8_0, got {T(ot.tensor_type).name}"
            if checked_q8 < n_check:
                orig = _bf16_to_f32_rows(t).reshape(-1)
                deq = quants.dequantize(np.asarray(ot.data), T.Q8_0).reshape(-1).astype(np.float32)
                cos = float(np.dot(orig, deq) / (np.linalg.norm(orig) * np.linalg.norm(deq) + 1e-12))
                cos_min = min(cos_min, cos)
                # 0.997 floor: embeddings (token_embd) have wide per-block dynamic
                # range so Q8_0 lands ~0.998; attn/ssm/output are ~1.0. UD-Q4_K_XL
                # ships Q8_0 embeddings too, so this matches the reference exactly.
                assert cos > 0.997, f"{t.name} Q8_0 dequant cosine {cos:.5f} too low"
                checked_q8 += 1
    print(f"[verify] OK — experts byte-identical ({checked_exp} spot-checked), "
          f"non-experts Q8_0 (min dequant cosine {cos_min:.5f} over {checked_q8})")


def self_test() -> None:
    """Tiny synthetic GGUF round-trip — proves layout/orientation before the 24 GB run."""
    import tempfile as _tf
    d = Path(_tf.mkdtemp())
    src = d / "tiny.gguf"
    w = gguf.GGUFWriter(str(src), arch="llama")
    w.add_uint32("llama.block_count", 1)
    rng = np.random.default_rng(0)
    # non-expert bf16 weight (ne0=64 %32==0) → should Q8
    attn = torch.from_numpy(rng.standard_normal((128, 64)).astype(np.float32)).bfloat16()
    w.add_tensor("blk.0.attn_qkv.weight", attn.view(torch.uint8).numpy(),
                 raw_shape=(64, 128), raw_dtype=T.BF16)
    # a 1-D norm (F32) → must be copied untouched
    w.add_tensor("blk.0.attn_norm.weight", rng.standard_normal(64).astype(np.float32))
    # a fake "expert" bf16 (name has _exps) → must be copied untouched even though bf16
    exp = torch.from_numpy(rng.standard_normal((32, 64)).astype(np.float32)).bfloat16()
    w.add_tensor("blk.0.ffn_gate_exps.weight", exp.view(torch.uint8).numpy(),
                 raw_shape=(64, 32), raw_dtype=T.BF16)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    out = d / "tiny-q8.gguf"
    res = convert(src, out)
    assert res["n_q8"] == 1, f"expected 1 Q8 tensor, got {res['n_q8']}"
    assert res["n_copy"] == 2, f"expected 2 copied, got {res['n_copy']}"
    verify(src, out)
    # explicit: expert stayed bf16, attn became Q8_0
    bt = {t.name: T(t.tensor_type).name for t in gguf.GGUFReader(out).tensors}
    assert bt["blk.0.ffn_gate_exps.weight"] == "BF16", bt
    assert bt["blk.0.attn_qkv.weight"] == "Q8_0", bt
    assert bt["blk.0.attn_norm.weight"] == "F32", bt
    print("[self-test] PASS — expert untouched, attn→Q8_0, norm untouched, shapes preserved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp")
    p.add_argument("--out", dest="out")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.inp or not args.out:
        p.error("--in and --out required (or --self-test)")
    inp, out = Path(args.inp), Path(args.out)
    if not args.verify_only:
        convert(inp, out)
    verify(inp, out)


if __name__ == "__main__":
    main()
