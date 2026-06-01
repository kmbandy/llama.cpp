#!/usr/bin/env python3
"""Merge a multi-part (split) GGUF into a single file, for tools that read via
gguf-py GGUFReader (single-file, no split-following) — e.g. the ml8 calibration
pager. Copies KV metadata from part 1 (dropping split.* keys), then streams every
tensor from all parts IN ORDER, byte-for-byte (raw_dtype preserved).

OOM-safe (host RAM << file size): GGUFWriter use_temp_file=True spools to the
output dir (must be NVMe, never /tmp tmpfs); per-tensor POSIX_FADV_DONTNEED drops
source page cache.

Usage:
  python3 merge_gguf_parts.py --out /home/kmbandy/models/Qwen3.6-27B-bf16.gguf \
      /home/kmbandy/models/qwen36-27b/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf \
      /home/kmbandy/models/qwen36-27b/BF16/Qwen3.6-27B-BF16-00002-of-00002.gguf
"""
from __future__ import annotations
import argparse, os, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

_STRUCT = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}


def _advise(fd, off, ln):
    if hasattr(os, "posix_fadvise") and fd >= 0:
        try: os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED)
        except OSError: pass


def _copy_field(w, name, field):
    t = field.types; v = field.contents()
    if t[0] == gguf.GGUFValueType.ARRAY:
        w.add_key_value(name, v, gguf.GGUFValueType.ARRAY, sub_type=t[1])
    else:
        w.add_key_value(name, v, t[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("parts", nargs="+")
    args = p.parse_args()
    out = Path(args.out)
    parts = [Path(x) for x in args.parts]

    r0 = gguf.GGUFReader(parts[0])
    arch = r0.fields["general.architecture"].contents()
    tempfile.tempdir = str(out.parent)   # spool on NVMe
    w = gguf.GGUFWriter(str(out), arch=arch, use_temp_file=True)

    n_fields = 0
    for name, field in r0.fields.items():
        if name in _STRUCT or name.startswith("split."):
            continue
        _copy_field(w, name, field); n_fields += 1
    print(f"[merge] arch={arch!r}  copied {n_fields} fields (dropped split.*)")

    total = 0
    for pi, part in enumerate(parts):
        r = gguf.GGUFReader(part)
        fd = -1
        try: fd = os.open(str(part), os.O_RDONLY)
        except OSError: fd = -1
        for t in r.tensors:
            cloned = np.ascontiguousarray(t.data)
            w.add_tensor(t.name, cloned, raw_dtype=gguf.GGMLQuantizationType(t.tensor_type))
            _advise(fd, t.data_offset, t.n_bytes)
            del cloned
            total += 1
        if fd >= 0: os.close(fd)
        print(f"[merge] part {pi+1}/{len(parts)} ({part.name}): +{len(r.tensors)} tensors")

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"[merge] wrote {total} tensors → {out}")


if __name__ == "__main__":
    main()
