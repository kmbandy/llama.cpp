"""ml8_to_gguf — patch ml8-4 reconstructed tensors into a base f16 GGUF.

Reads a base f16 GGUF + a directory of ml8-4 .pt blobs, writes a new GGUF with
the matching MLP tensors replaced by their inference-equivalent reconstructions
(rotation absorbed via reconstruct_inference_weight). Non-MLP tensors are copied
unchanged.

Why this exists rather than the HF round-trip:
  Loading the full HF model into CPU RAM to call .save_pretrained() OOM-killed
  twice on a 15 GB RAM box (the f16 model itself is 8 GB). This patcher mmaps
  the base GGUF, never holds more than one tensor in heap memory, and peaks at
  ~1 GB RAM total.

Naming map (HF → GGUF) for Qwen-class MLP linears:
  model.layers.N.mlp.gate_proj  →  blk.N.ffn_gate.weight
  model.layers.N.mlp.up_proj    →  blk.N.ffn_up.weight
  model.layers.N.mlp.down_proj  →  blk.N.ffn_down.weight

Shape: HF stores Linear weight as (out_features, in_features); GGUF stores it
as (in_features, out_features). The reconstructed tensor must be transposed
before writing.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))
import gguf  # noqa: E402
from ml8_io import load_ml8_layer, reconstruct_inference_weight  # noqa: E402


# Structural GGUF fields managed by the writer itself — don't try to re-add.
_SKIP_FIELDS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}


_HF_MLP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$")
_MLP_SUFFIX_MAP = {
    "gate_proj": "ffn_gate",
    "up_proj":   "ffn_up",
    "down_proj": "ffn_down",
}


def hf_to_gguf_name(hf_name: str) -> str:
    """Map a HF Linear tensor name to its GGUF counterpart.

    Only MLP linears are supported today (Saturday's MAD-223 calibration scope).
    Raises ValueError for unrecognized names so a typo in a calling script
    surfaces immediately instead of silently producing a no-op patch.
    """
    m = _HF_MLP_PATTERN.match(hf_name)
    if not m:
        raise ValueError(
            f"hf_to_gguf_name: no mapping for {hf_name!r}. "
            f"Supported: model.layers.N.mlp.{{gate_proj,up_proj,down_proj}}"
        )
    layer_idx, hf_suffix = m.group(1), m.group(2)
    return f"blk.{layer_idx}.{_MLP_SUFFIX_MAP[hf_suffix]}.weight"


def _build_blob_map(calib_dir: Path) -> dict[str, Path]:
    """Index .pt blobs by their GGUF tensor name. Skips blobs whose HF name
    isn't an MLP linear (e.g., future attention blobs)."""
    blob_map: dict[str, Path] = {}
    for path in sorted(calib_dir.glob("*.pt")):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        try:
            gguf_name = hf_to_gguf_name(blob["name"])
        except ValueError:
            print(f"  [skip-blob] {path.name}: {blob['name']!r} not in MLP namespace")
            continue
        blob_map[gguf_name] = path
    return blob_map


def _copy_field(writer: "gguf.GGUFWriter", name: str, field: "gguf.gguf_reader.ReaderField") -> None:
    """Copy a single field from the reader to the writer.

    Handles scalars (STRING, BOOL, *INT*, FLOAT32) and ARRAY-of-scalar/STRING.
    """
    value = field.contents()
    types = field.types
    primary = types[0]
    if primary == gguf.GGUFValueType.ARRAY:
        if len(types) < 2:
            raise ValueError(f"field {name!r}: ARRAY type without sub-type")
        sub_type = types[1]
        writer.add_key_value(name, value, gguf.GGUFValueType.ARRAY, sub_type=sub_type)
    else:
        writer.add_key_value(name, value, primary)


def patch_gguf(base_gguf: Path, calib_dir: Path, out_gguf: Path) -> dict:
    """Write a new GGUF that is the base with MLP tensors replaced by ml8-4 reconstructions.

    Returns a summary dict: n_tensors_total, n_patched, n_copied, out_path.
    Mmaps the base GGUF — peak RAM ~one tensor (~100 MB max for Qwen3.5-4B).
    """
    reader = gguf.GGUFReader(base_gguf)
    arch = reader.fields["general.architecture"].contents()
    print(f"[base] {base_gguf}  arch={arch!r}  "
          f"fields={len(reader.fields)}  tensors={len(reader.tensors)}")

    blob_map = _build_blob_map(calib_dir)
    print(f"[blobs] {len(blob_map)} GGUF tensors will be replaced")

    writer = gguf.GGUFWriter(str(out_gguf), arch=arch)

    n_fields_copied = 0
    for name, field in reader.fields.items():
        if name in _SKIP_FIELDS:
            continue
        _copy_field(writer, name, field)
        n_fields_copied += 1
    print(f"[fields] copied {n_fields_copied}")

    n_patched = 0
    n_copied = 0
    for tensor in reader.tensors:
        if tensor.name in blob_map:
            blob = load_ml8_layer(blob_map[tensor.name])
            # Inference-equivalent reconstruction (rotation absorbed if present).
            # HF stores numpy shape (out_features, in_features). gguf-py reverses
            # numpy shape into ggml ne ordering (innermost-first), so we pass the
            # HF-natural shape and let gguf-py do the reversal. Earlier transpose
            # attempt produced an off-by-one swap that llama-cpp loader rejected
            # (verified 2026-05-24 with tiny probe in gguf-py).
            W = reconstruct_inference_weight(blob).numpy().astype(np.float16, copy=False)
            # tensor.shape from reader is in ggml ne order (reversed from numpy).
            # We compare against the reversed expected: reader.shape should equal
            # tuple(reversed(W.shape)).
            expected_reader_shape = tuple(reversed(W.shape))
            if tuple(tensor.shape) != expected_reader_shape:
                raise RuntimeError(
                    f"shape mismatch on {tensor.name}: blob numpy shape {W.shape} "
                    f"→ expected reader-shape {expected_reader_shape}, GGUF reports {tuple(tensor.shape)}"
                )
            writer.add_tensor(tensor.name, W, raw_dtype=tensor.tensor_type)
            n_patched += 1
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            n_copied += 1

    print(f"[tensors] patched {n_patched}, copied unchanged {n_copied}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "out_path": str(out_gguf),
        "n_tensors_total": n_patched + n_copied,
        "n_patched": n_patched,
        "n_copied": n_copied,
        "n_fields_copied": n_fields_copied,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-gguf", type=Path, required=True,
                   help="Existing f16 GGUF to patch from")
    p.add_argument("--calib-dir", type=Path, required=True,
                   help="Directory of ml8-4 per-layer .pt blobs")
    p.add_argument("--out-gguf", type=Path, required=True,
                   help="Output GGUF path")
    args = p.parse_args()
    summary = patch_gguf(args.base_gguf, args.calib_dir, args.out_gguf)
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
