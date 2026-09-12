#!/usr/bin/env python3
"""Per-layer DeepSeek-V4.1 routed-expert convert + v1 wp-repack.

HF layout is one layer per safetensor (00003 = L0 ... 00042 = L39). Experts
are already mxfp4-pack-quantized; we lossless-repack into ggml MXFP4, write a
one-layer GGUF, then llama-wp-repack --shard-by-layer --allow-partial.

Shared-expert + attn tensors in the same shard are peeled to spine staging.
The HF shard is deleted after a successful convert.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gguf  # noqa: E402
from conversion.base import ModelBase  # noqa: E402

logger = logging.getLogger("dsv41-experts")

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
# shard N = layer (N-3); layer 0 lives in model-00003-of-00048.safetensors
SHARD_OF_LAYER = 3
N_SHARDS = 48
PROJ = {
    "w1": gguf.MODEL_TENSOR.FFN_GATE_EXP,
    "w2": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
    "w3": gguf.MODEL_TENSOR.FFN_UP_EXP,
}
WP_REPACK = Path("/home/kmbandy/GitHub/llama.cpp-wt/dsv41/build-cpu/bin/llama-wp-repack")


def shard_name(layer: int) -> str:
    n = layer + SHARD_OF_LAYER
    return f"model-{n:05d}-of-{N_SHARDS:05d}.safetensors"


def tensor_name(key: gguf.MODEL_TENSOR, bid: int) -> str:
    return gguf.TENSOR_NAMES[key].format(bid=bid) + ".weight"


def flatten_hparams(cfg: dict) -> dict:
    out = dict(cfg)
    for k, v in (cfg.get("text_config") or {}).items():
        out.setdefault(k, v)
    return out


def download_shard(repo: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    if path.is_file() and path.stat().st_size > 1_000_000:
        logger.info("already have %s (%.2f GiB)", path.name, path.stat().st_size / 1024 ** 3)
        return path
    logger.info("hf download %s", filename)
    cmd = [
        "hf", "download", repo, filename,
        "--local-dir", str(dest_dir),
    ]
    subprocess.run(cmd, check=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def peel_spine(st_path: Path, layer: int, spine_dir: Path) -> Path:
    """Copy every non-routed-expert tensor from this shard into a small safetensors."""
    spine_dir.mkdir(parents=True, exist_ok=True)
    out = spine_dir / f"spine-L{layer:02d}.safetensors"
    if out.is_file():
        return out
    prefix = f"layers.{layer}.ffn.experts."
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.startswith(prefix):
                continue
            tensors[name] = f.get_tensor(name)
    save_safetensors(tensors, str(out))
    logger.info("peeled %d spine tensors -> %s (%.1f MiB)",
                len(tensors), out.name, out.stat().st_size / 1024 ** 2)
    return out


def convert_layer_experts(
    st_path: Path,
    layer: int,
    n_experts: int,
    gguf_path: Path,
    hf_prefix: str | None = None,
) -> None:
    t0 = time.time()
    os.environ.setdefault("TMPDIR", str(gguf_path.parent))
    writer = gguf.GGUFWriter(str(gguf_path), arch="deepseek41", use_temp_file=True)
    writer.add_block_count(40)
    writer.add_expert_count(n_experts)
    writer.add_expert_feed_forward_length(2304)

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for proj, key in PROJ.items():
            stacked = None
            for eid in range(n_experts):
                base = hf_prefix or f"layers.{layer}.ffn.experts"
                wname = f"{base}.{eid}.{proj}.weight"
                sname = f"{base}.{eid}.{proj}.scale"
                if wname not in keys or sname not in keys:
                    raise KeyError(f"missing {wname} or {sname} in {st_path.name}")
                packed = ModelBase.repack_mxfp4_blocks(f.get_tensor(wname), f.get_tensor(sname))
                if stacked is None:
                    stacked = np.empty((n_experts, *packed.shape), dtype=packed.dtype)
                stacked[eid] = packed
                del packed
            assert stacked is not None
            logger.info("  L%d %s packed %s %.2f GiB", layer, proj, stacked.shape,
                        stacked.nbytes / 1024 ** 3)
            writer.add_tensor(tensor_name(key, layer), stacked,
                              raw_dtype=gguf.GGMLQuantizationType.MXFP4)
            del stacked

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("wrote %s (%.2f GiB) in %.1fs", gguf_path.name,
                gguf_path.stat().st_size / 1024 ** 3, time.time() - t0)


def repack_layer(gguf_path: Path, layer: int, raw_dir: Path, wp_repack: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = raw_dir / f"L{layer:02d}"
    manifest = Path(str(base) + "-experts-manifest.json")
    if manifest.is_file():
        logger.info("repack already present for L%d", layer)
        return manifest
    cmd = [
        str(wp_repack),
        "--allow-partial",
        "--layer-ranges", f"{layer}-{layer}",
        str(gguf_path),
        str(base),
    ]
    logger.info("wp-repack %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return manifest


MODEL_FILES = [str(Path.home() / "models/dsv41-spine.gguf")]


def aggregate_identity(shards: list[dict]) -> dict:
    """Manifest identity for a stitched set: sha256 over the ordered per-shard
    blob hashes. Each layer was repacked from its own throwaway GGUF, so there
    is no single input-model hash to inherit."""
    import hashlib
    h = hashlib.sha256()
    h.update(b"llama.cpp.wp-repack.layer-ranges.v1")
    for s in shards:
        ch = s["content_hash"]
        h.update(f'{ch["algorithm"]}:{ch["value"]}\n'.encode())
    return {"algorithm": "sha256-of-shards", "value": h.hexdigest()}


def stitch(sources: list[tuple[int, Path, Path]], out_dir: Path, output_base: str,
           n_expert: int, model_files: list[str] | None = None,
           input_model: str = REPO, expert_type: str = "mxfp4") -> Path:
    """Rename per-layer blobs into one NNNNN-of-MMMMM set + manifest.

    sources: (layer, index_json, blob) per layer, ascending. The copied index
    gets blob_file/model_files rewritten so worker/descriptor cross-checks hold
    without the per-layer GGUFs (deleted after repack).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_files = model_files or MODEL_FILES
    n = len(sources)
    shards = []
    total = 0
    layers = [layer for layer, _, _ in sources]
    for i, (layer, src_idx, src_blob) in enumerate(sources):
        dst_blob = f"{output_base}-experts-{i + 1:05d}-of-{n:05d}.wpb"
        dst_idx = f"{output_base}-experts-{i + 1:05d}-of-{n:05d}.wpi.json"
        dst_blob_path = out_dir / dst_blob
        dst_idx_path = out_dir / dst_idx
        idx = json.loads(src_idx.read_text())
        if src_blob.resolve() != dst_blob_path.resolve():
            if dst_blob_path.exists():
                dst_blob_path.unlink()
            try:
                os.link(src_blob, dst_blob_path)
            except OSError:
                shutil.copy2(src_blob, dst_blob_path)
        if idx["layer_first"] != layer or idx["layer_last"] != layer:
            raise ValueError(f"{src_idx}: index is not layer {layer}")
        idx["blob_file"] = dst_blob
        idx["shard_index"] = i
        idx["shard_count"] = n
        idx["model_files"] = model_files
        if src_idx.resolve() != dst_idx_path.resolve() and dst_idx_path.exists():
            dst_idx_path.unlink()
        dst_idx_path.write_text(json.dumps(idx, indent=2) + "\n")
        entry = {
            "blob_file": dst_blob,
            "index_file": dst_idx,
            "shard_index": i,
            "layer_first": layer,
            "layer_last": layer,
            "group_count": idx["group_count"],
            "blob_bytes": idx["blob_bytes"],
            "content_hash": idx["content_hash"],
        }
        shards.append(entry)
        total += int(entry["blob_bytes"])

    ranges = []
    lo = prev = layers[0]
    for layer in layers[1:]:
        if layer != prev + 1:
            ranges.append(f"{lo}-{prev}")
            lo = layer
        prev = layer
    ranges.append(f"{lo}-{prev}")

    combined = {
        "format": "llama.cpp.weight-pager.expert-shard-manifest",
        "version": 1,
        "sharding_mode": "layer-ranges",
        "layer_ranges": ranges,
        "allow_partial": True,
        "input_model": input_model,
        "model_files": model_files,
        "retained_expert_range": {"first": 0, "last": n_expert - 1},
        "expert_ggml_type": expert_type,
        "content_hash": aggregate_identity(shards),
        "total_group_count": sum(s["group_count"] for s in shards),
        "total_blob_bytes": total,
        "shard_count": n,
        "shards": shards,
        "note": f"V4.1-Flash experts layers {','.join(ranges)}, {expert_type} v1 up|gate|down packed",
    }
    out_man = out_dir / f"{output_base}-experts-manifest.json"
    out_man.write_text(json.dumps(combined, indent=2) + "\n")
    logger.info("stitched %d shards -> %s (%.1f GiB)", n, out_man, total / 1024 ** 3)
    return out_man


def raw_sources(raw_dir: Path, layers: list[int]) -> list[tuple[int, Path, Path]]:
    """(layer, index, blob) for llama-wp-repack's per-layer PREFIX-experts-00001-of-00001 output."""
    out = []
    for layer in layers:
        base = raw_dir / f"L{layer:02d}"
        man = json.loads(Path(str(base) + "-experts-manifest.json").read_text())
        shard = man["shards"][0]
        out.append((layer, man_dir(base) / shard["index_file"], man_dir(base) / shard["blob_file"]))
    return out


def man_dir(base: Path) -> Path:
    return base.parent


def rsync_to(out_dir: Path, dest: str) -> None:
    cmd = [
        "rsync", "-a", "--info=progress2",
        "--exclude", "raw/",
        str(out_dir).rstrip("/") + "/", dest.rstrip("/") + "/",
    ]
    logger.info("%s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--layers", default="26-39", help="inclusive layer range")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / "models/dsv41-hf-experts")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "models/dsv41-eslice-encoder")
    ap.add_argument("--name", default="dsv41-encoder", help="blob/manifest prefix")
    ap.add_argument("--spine-dir", type=Path, default=Path.home() / "models/dsv41-spine-staging")
    ap.add_argument("--wp-repack", type=Path, default=WP_REPACK)
    ap.add_argument("--rsync-to", default="", help="host:path; omit to keep blobs local")
    ap.add_argument("--keep-gguf", action="store_true")
    ap.add_argument("--keep-hf", action="store_true")
    ap.add_argument("--skip-rsync", action="store_true")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-", 1))
    layers = list(range(lo, hi + 1))
    if not args.wp_repack.is_file():
        raise SystemExit(f"llama-wp-repack not found: {args.wp_repack}")

    os.environ.setdefault("TMPDIR", str(args.cache_dir))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.cache_dir / "shards"
    gguf_dir = args.cache_dir / "layer-gguf"
    raw_dir = args.out_dir / "raw"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = args.cache_dir / "config.json"
    if not cfg_path.is_file():
        subprocess.run(["hf", "download", args.repo, "config.json", "--local-dir", str(args.cache_dir)], check=True)
    hparams = flatten_hparams(json.loads(cfg_path.read_text()))
    n_experts = int(hparams["n_routed_experts"])
    logger.info("n_routed_experts=%d layers=%s", n_experts, layers)

    progress_path = args.cache_dir / f"progress-{lo}-{hi}.json"
    done = set()
    if progress_path.is_file():
        done = set(json.loads(progress_path.read_text()).get("done_layers", []))

    for layer in layers:
        if layer in done:
            logger.info("skip L%d (progress)", layer)
            continue
        gguf_path = gguf_dir / f"L{layer:02d}.gguf"
        man = Path(str(raw_dir / f"L{layer:02d}") + "-experts-manifest.json")
        if not man.is_file():
            if not gguf_path.is_file():
                st = download_shard(args.repo, shard_name(layer), shards_dir)
                peel_spine(st, layer, args.spine_dir)
                convert_layer_experts(st, layer, n_experts, gguf_path)
                if not args.keep_hf:
                    st.unlink(missing_ok=True)
                    logger.info("deleted HF shard L%d", layer)
            repack_layer(gguf_path, layer, raw_dir, args.wp_repack)
            if not args.keep_gguf:
                gguf_path.unlink(missing_ok=True)
        done.add(layer)
        progress_path.write_text(json.dumps({"done_layers": sorted(done)}) + "\n")

    stitched = stitch(raw_sources(raw_dir, layers), args.out_dir, args.name, n_experts)
    if args.rsync_to and not args.skip_rsync:
        rsync_to(args.out_dir, args.rsync_to)
        logger.info("rsync complete; local out-dir still at %s", args.out_dir)
    logger.info("done %s", stitched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
