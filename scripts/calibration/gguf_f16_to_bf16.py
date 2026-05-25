#!/usr/bin/env python3
"""Convert an F16-only GGUF to BF16 by recasting every F16 tensor.

Used by MAD-238 parity gate: Qwen3.5-4B HF native dtype is bf16 (torch_dtype=
float16 is silently ignored by modern transformers), so the paged calibration
needs a bf16 GGUF to match the bf16 model's weights exactly.

Usage:
    python3 gguf_f16_to_bf16.py <input.gguf> <output.gguf>

Non-F16 tensors (Q4_K, F32 norm scales, etc.) are passed through unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402


def _f16_bytes_to_bf16_bytes(raw: np.ndarray) -> np.ndarray:
    """Reinterpret f16 bytes → bf16 bytes via torch (numpy lacks bf16)."""
    f16 = torch.from_numpy(raw.view(np.float16).copy())
    bf16 = f16.to(torch.bfloat16)
    return bf16.view(torch.uint8).numpy().copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()

    reader = gguf.GGUFReader(args.input)
    arch = reader.fields["general.architecture"]
    arch_str = bytes(arch.parts[arch.data[0]]).decode("utf-8")
    print(f"[read] {args.input}  arch={arch_str}  {len(reader.tensors)} tensors")

    writer = gguf.GGUFWriter(args.output, arch_str)

    # Copy non-architecture metadata fields.
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        if key.startswith("GGUF."):
            continue
        # field.parts[i] is a numpy array; field.data is indices into parts.
        # field.types[0] is the GGUFValueType.
        vtype = field.types[0]
        # Reassemble the value via the parts/data indices.
        try:
            if vtype == gguf.GGUFValueType.STRING:
                value = bytes(field.parts[field.data[0]]).decode("utf-8")
                writer.add_string(key, value)
            elif vtype == gguf.GGUFValueType.ARRAY:
                # Array: parts[1] = inner type, data is index list
                inner_type = field.types[1]
                items = []
                for idx in field.data:
                    part = field.parts[idx]
                    if inner_type == gguf.GGUFValueType.STRING:
                        items.append(bytes(part).decode("utf-8"))
                    else:
                        items.append(part[0].item() if hasattr(part[0], "item") else part[0])
                writer.add_array(key, items)
            else:
                # Scalar numeric
                part = field.parts[field.data[0]]
                val = part[0].item() if hasattr(part[0], "item") else part[0]
                # Use type-specific add_* methods
                _add = {
                    gguf.GGUFValueType.UINT8:  writer.add_uint8,
                    gguf.GGUFValueType.INT8:   writer.add_int8,
                    gguf.GGUFValueType.UINT16: writer.add_uint16,
                    gguf.GGUFValueType.INT16:  writer.add_int16,
                    gguf.GGUFValueType.UINT32: writer.add_uint32,
                    gguf.GGUFValueType.INT32:  writer.add_int32,
                    gguf.GGUFValueType.FLOAT32:writer.add_float32,
                    gguf.GGUFValueType.UINT64: writer.add_uint64,
                    gguf.GGUFValueType.INT64:  writer.add_int64,
                    gguf.GGUFValueType.FLOAT64:writer.add_float64,
                    gguf.GGUFValueType.BOOL:   writer.add_bool,
                }.get(vtype)
                if _add is None:
                    print(f"  [skip-meta] {key}: unsupported type {vtype}")
                    continue
                _add(key, val)
        except Exception as e:
            print(f"  [warn] couldn't copy field {key}: {e}")

    n_converted = 0
    n_passthrough = 0
    for t in reader.tensors:
        # GGUF shape is in ne-natural order (reversed vs numpy).
        # gguf_writer wants numpy-natural shape.
        shape = tuple(reversed([int(d) for d in t.shape]))
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            raw_bytes = bytes(t.data)
            f16_array = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
            bf16_bytes = _f16_bytes_to_bf16_bytes(f16_array.flatten())
            # Reshape bf16 raw bytes back to shape (each element is 2 bytes).
            bf16_view = bf16_bytes.view(np.uint16).reshape(shape)
            writer.add_tensor(t.name, bf16_view, raw_dtype=gguf.GGMLQuantizationType.BF16)
            n_converted += 1
        else:
            # Passthrough — keep original dtype + bytes.
            writer.add_tensor(t.name, np.array(t.data).reshape(shape),
                              raw_dtype=t.tensor_type)
            n_passthrough += 1

    print(f"[write] {args.output}  converted={n_converted}  passthrough={n_passthrough}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("[done]")


if __name__ == "__main__":
    main()
