#!/usr/bin/env python3
"""Re-stitch already-stitched V4.1 expert sets into one self-describing
layer-ranges manifest (manifest v1 + rewritten indexes; blobs are hardlinked,
never rewritten). Used once to repair the 2026-09-10 sets and to merge the
2026 decoder + MTP sets into a single 26-42 band.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conversion.dsv41_experts_from_hf import stitch  # noqa: E402

logger = logging.getLogger("dsv41-restitch")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True, type=Path,
                    help="existing stitched manifest (repeatable, any layer order)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="blob/manifest prefix, e.g. dsv41-decoder")
    ap.add_argument("--n-expert", type=int, default=384, help="model expert_count (main stack)")
    ap.add_argument("--model-file", default=str(Path.home() / "models/dsv41-spine.gguf"))
    ap.add_argument("--prune", action="store_true",
                    help="delete <name>-experts-* files in out-dir not named by the new manifest")
    args = ap.parse_args()

    sources = []
    for man_path in args.manifest:
        man = json.loads(man_path.read_text())
        for shard in man["shards"]:
            idx = man_path.parent / shard["index_file"]
            blob = man_path.parent / shard["blob_file"]
            if not idx.is_file() or not blob.is_file():
                raise FileNotFoundError(f"{idx} / {blob}")
            if blob.stat().st_size != shard["blob_bytes"]:
                raise ValueError(f"{blob}: size {blob.stat().st_size} != {shard['blob_bytes']}")
            sources.append((shard["layer_first"], idx, blob))
    sources.sort(key=lambda t: t[0])
    layers = [s[0] for s in sources]
    if len(set(layers)) != len(layers):
        raise ValueError(f"repeated layers: {layers}")

    out_man = stitch(sources, args.out_dir, args.name, args.n_expert, model_files=[args.model_file])
    keep = {out_man.name}
    man = json.loads(out_man.read_text())
    for shard in man["shards"]:
        keep.add(shard["blob_file"])
        keep.add(shard["index_file"])
    stale = [p for p in args.out_dir.glob(f"{args.name}-experts-*") if p.name not in keep]
    for p in stale:
        if args.prune:
            p.unlink()
            logger.info("pruned %s", p.name)
        else:
            logger.info("stale (use --prune): %s", p.name)
    logger.info("layers %s -> %s", layers, out_man)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
