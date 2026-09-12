#!/usr/bin/env python3
"""Assemble V4.1 spine GGUF from layer peels + embed/head/MTP shards.

Routed experts are omitted (already .wpb). MTP routed experts are packed
separately as layers 40-42 .wpb for the 2026 decoder box.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

logger = logging.getLogger("dsv41-spine")

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
PEEL_DIR = Path.home() / "models/dsv41-spine-staging"
SPINE_DIR = Path.home() / "models/dsv41-hf-spine"
SHARDS = SPINE_DIR / "shards"
OUT_GGUF = Path.home() / "models/dsv41-spine.gguf"
MTP_WPB = Path.home() / "models/dsv41-eslice-mtp"
CONVERT = Path("/home/kmbandy/GitHub/llama.cpp-wt/dsv41/convert_hf_to_gguf.py")
EXPERTS = Path("/home/kmbandy/GitHub/llama.cpp-wt/dsv41/conversion/dsv41_experts_from_hf.py")


def extract_mtp(src: Path, stage: int, spine_out: Path, expert_out: Path) -> tuple[int, int]:
    prefix = f"mtp.{stage}."
    exp_prefix = f"{prefix}ffn.experts."
    spine, experts = {}, {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name.startswith(exp_prefix):
                experts[name] = t
            elif name.startswith(prefix) or name.startswith("mtp."):
                spine[name] = t
    spine_out.parent.mkdir(parents=True, exist_ok=True)
    expert_out.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(spine, str(spine_out))
    save_safetensors(experts, str(expert_out))
    logger.info("mtp.%d spine %d tensors %.1f MiB, experts %d tensors %.1f MiB",
                stage, len(spine), spine_out.stat().st_size / 1024 ** 2,
                len(experts), expert_out.stat().st_size / 1024 ** 2)
    return len(spine), len(experts)


def build_index(model_dir: Path) -> None:
    weight_map = {}
    for st in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(str(st), framework="pt") as f:
            for name in f.keys():
                weight_map[name] = st.name
    idx = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(idx) + "\n")
    logger.info("index %d tensors in %d files", len(weight_map), len(set(weight_map.values())))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TMPDIR", str(SPINE_DIR))
    SPINE_DIR.mkdir(parents=True, exist_ok=True)

    # Symlink peels as model-*.safetensors so convert's prefix matcher sees them.
    for layer in range(40):
        src = PEEL_DIR / f"spine-L{layer:02d}.safetensors"
        dst = SPINE_DIR / f"model-spine-L{layer:02d}.safetensors"
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    for src_name, dst_name in (
        ("model-00002-of-00048.safetensors", "model-embed.safetensors"),
        ("model-00043-of-00048.safetensors", "model-head.safetensors"),
    ):
        src = SHARDS / src_name
        dst = SPINE_DIR / dst_name
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    mtp_exp_dir = SPINE_DIR / "mtp-experts"
    mtp_exp_dir.mkdir(exist_ok=True)
    for stage in range(3):
        src = SHARDS / f"model-{44 + stage:05d}-of-00048.safetensors"
        extract_mtp(
            src, stage,
            SPINE_DIR / f"model-mtp-{stage}-spine.safetensors",
            mtp_exp_dir / f"mtp-{stage}-experts.safetensors",
        )

    build_index(SPINE_DIR)

    cmd = [
        sys.executable, "-u", str(CONVERT),
        str(SPINE_DIR),
        "--outfile", str(OUT_GGUF),
        "--outtype", "auto",
        "--use-temp-file",
    ]
    logger.info("%s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("spine GGUF %s (%.2f GiB)", OUT_GGUF, OUT_GGUF.stat().st_size / 1024 ** 3)

    # MTP routed experts: 128 experts x 3 stages -> layers 40-42 .wpb on 2026.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conversion.dsv41_experts_from_hf import convert_layer_experts, repack_layer, stitch, raw_sources

    wp = Path("/home/kmbandy/GitHub/llama.cpp-wt/dsv41/build-cpu/bin/llama-wp-repack")
    gguf_dir = MTP_WPB / "layer-gguf"
    raw_dir = MTP_WPB / "raw"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    layers = []
    for stage in range(3):
        layer = 40 + stage
        layers.append(layer)
        src = mtp_exp_dir / f"mtp-{stage}-experts.safetensors"
        gguf_path = gguf_dir / f"L{layer:02d}.gguf"
        if not gguf_path.is_file():
            convert_layer_experts(
                src, layer, 128, gguf_path,
                hf_prefix=f"mtp.{stage}.ffn.experts",
            )
        repack_layer(gguf_path, layer, raw_dir, wp)
        gguf_path.unlink(missing_ok=True)
    stitch(raw_sources(raw_dir, layers), MTP_WPB, "dsv41-mtp", 128)
    dest = "mad-lab-2026:/mnt/nvme/dsv41-eslice-mtp/"
    subprocess.run(
        ["rsync", "-a", "--info=progress2", "--exclude", "raw/",
         str(MTP_WPB).rstrip("/") + "/", dest],
        check=True,
    )
    logger.info("mtp experts rsynced to %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
