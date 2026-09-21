#!/usr/bin/env python3
"""Stream the Qwen3.8-Flash-Next PLE n-gram table from Hugging Face and pack it MXFP4.

The table is 128 BF16 shards `layers.1.ple.ple_embedding.ngram_embedding.shard_N`
of [2,500,012 x 160] (~102 GB) spread over model-00005..00037. Upstream's
converter concatenates them in index order into ONE tensor
`per_layer_token_embd.weight` [320,001,536 x 160]; this script produces that
tensor as MXFP4 (~27 GB) in a sidecar GGUF, the way dsv41_engram_from_hf.py
does for the DeepSeek-V4.1 Engram tables. The main model GGUF is converted
without the table and loads this file with --model-sidecar.

Row width 160 = 5 MXFP4 blocks of 32, so the packing is exact per row.
Streaming is by HTTP Range in CHUNK_ROWS-row pieces (1M rows = 320 MB BF16);
MXFP4 packing uses the fork pool from the DS4.1 script. Resume: the staging
file + progress JSON continue from the last completed row.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf  # noqa: E402
from gguf.utility import SafetensorRemote  # noqa: E402

from dsv41_engram_from_hf import (  # noqa: E402
    ELEM_BYTES,
    _mxfp4_quantize_with_pool,
    _mxfp4_worker_init,
    bf16_to_f32,
    flatten_hparams,
    hf_headers,
    http_range,
    load_json_url,
    make_session,
    require_free,
    resolve_url,
    write_progress,
)

logger = logging.getLogger("qwen38-ple")

REPO = "Qwen/Qwen3.8-Flash-Next"
SHARD_SUFFIX = ".ple.ple_embedding.ngram_embedding.shard_"
CHUNK_ROWS = 1_000_000
DEFAULT_WORKERS = 12


def probe(session, repo: str):
    base = f"{SafetensorRemote.BASE_DOMAIN}/{repo}/resolve/main"
    logger.info("fetching config + weight map from %s", repo)
    hparams = flatten_hparams(load_json_url(session, f"{base}/config.json"))
    weight_map = load_json_url(session, f"{base}/model.safetensors.index.json")["weight_map"]

    n_parts = int(hparams["split_ngram_parts"])
    shards: dict[int, tuple[str, str]] = {}   # idx -> (tensor name, file)
    consts: dict[str, str] = {}               # const suffix -> file
    for name, fname in weight_map.items():
        if SHARD_SUFFIX in name:
            idx = int(name.rpartition(".shard_")[2].partition(".")[0])
            shards[idx] = (name, fname)
        for suffix in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
            if name.endswith("ple_embedding." + suffix):
                consts[suffix] = fname
    if sorted(shards) != list(range(n_parts)):
        raise RuntimeError(f"expected shards 0..{n_parts - 1}, got {sorted(shards)}")
    if len(consts) != 3:
        raise RuntimeError(f"PLE hash constants missing: have {sorted(consts)}")

    files = sorted({f for _, f in shards.values()} | set(consts.values()))
    logger.info("%d shards over %d files (%s .. %s)", n_parts, len(files), files[0], files[-1])
    remote = {}
    for fname in files:
        found = SafetensorRemote.get_list_tensors(f"{base}/{fname}")
        remote[fname] = found
    return hparams, shards, consts, remote, base


def read_consts(session, base: str, consts: dict[str, str], remote) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for suffix, fname in consts.items():
        found = remote[fname]
        name = next(k for k in found if k.endswith("ple_embedding." + suffix))
        meta = found[name]
        if meta.dtype != "I64":
            raise RuntimeError(f"{name}: expected I64, got {meta.dtype}")
        cdn = resolve_url(session, f"{base}/{fname}")
        raw = http_range(session, cdn, meta.offset_start, meta.size)
        out[suffix] = [int(x) for x in np.frombuffer(raw, dtype=np.int64)]
        logger.info("%s = %s", suffix, out[suffix] if len(out[suffix]) <= 16 else f"[{len(out[suffix])} values]")
    return out


def convert_table(session, base: str, shards, remote, staging_dir: Path, chunk_rows: int, workers: int) -> tuple[Path, int, int]:
    # geometry from the shards themselves
    n_rows = 0
    n_cols = None
    for idx in sorted(shards):
        name, fname = shards[idx]
        meta = remote[fname][name]
        r, c = (int(x) for x in meta.shape)
        if meta.dtype != "BF16":
            raise RuntimeError(f"{name}: expected BF16, got {meta.dtype}")
        if n_cols is None:
            n_cols = c
        elif c != n_cols:
            raise RuntimeError(f"{name}: row dim {c} != {n_cols}")
        n_rows += r
    assert n_cols is not None
    if n_cols % 32:
        raise ValueError(f"row width {n_cols} not multiple of MXFP4 block 32")

    qtype = gguf.GGMLQuantizationType.MXFP4
    row_bytes = int(gguf.quantize(np.zeros((1, n_cols), dtype=np.float32), qtype).nbytes)
    row_in = n_cols * ELEM_BYTES["BF16"]
    staging = staging_dir / "ple_MXFP4.bin"
    progress_path = staging_dir / "ple_MXFP4.progress.json"
    logger.info("PLE table: %d shards -> MXFP4 %d x %d (%.1f GB) at %s",
                len(shards), n_rows, n_cols, n_rows * row_bytes / 1e9, staging)

    start_row = 0
    if staging.is_file() and progress_path.is_file():
        prog = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(prog.get("n_rows", -1)) == n_rows and int(prog.get("row_bytes", -1)) == row_bytes:
            start_row = min(int(prog.get("rows_done", 0)), n_rows)
            if start_row:
                logger.info("resuming at row %d / %d", start_row, n_rows)
    if start_row == 0:
        require_free(staging_dir, n_rows * row_bytes + 8 * 1024 ** 3)
        out = np.memmap(staging, dtype=np.uint8, mode="w+", shape=(n_rows, row_bytes))
        out.flush()
    else:
        if staging.stat().st_size != n_rows * row_bytes:
            raise RuntimeError(f"staging size {staging.stat().st_size} != {n_rows * row_bytes}")
        out = np.memmap(staging, dtype=np.uint8, mode="r+", shape=(n_rows, row_bytes))

    # global row -> (shard idx, row within shard); shards are contiguous in index order
    starts = []
    acc = 0
    for idx in sorted(shards):
        name, fname = shards[idx]
        starts.append((acc, idx))
        acc += int(remote[fname][name].shape[0])

    cdn_cache: dict[str, str] = {}

    def ranged_one(fname: str, start: int, size: int) -> bytes:
        url = cdn_cache.get(fname) or resolve_url(session, f"{base}/{fname}")
        cdn_cache[fname] = url
        try:
            return http_range(session, url, start, size)
        except PermissionError:
            logger.warning("CDN auth expired, re-resolving %s", fname)
            cdn_cache[fname] = resolve_url(session, f"{base}/{fname}")
            return http_range(session, cdn_cache[fname], start, size)

    # HF caps per-connection throughput (~20 MB/s measured), not the link:
    # split every range over N parallel connections
    n_conn = max(1, int(os.environ.get("PLE_CONNS", "4")))
    dl_pool = ThreadPoolExecutor(max_workers=n_conn)

    def ranged(fname: str, start: int, size: int) -> bytes:
        if n_conn == 1 or size < 8 << 20:
            return ranged_one(fname, start, size)
        step = (size + n_conn - 1) // n_conn
        parts = [(start + i * step, min(step, size - i * step)) for i in range(n_conn) if i * step < size]
        futs = [dl_pool.submit(ranged_one, fname, s0, n0) for s0, n0 in parts]
        return b"".join(f.result() for f in futs)

    def fetch_rows(g0: int, g1: int) -> np.ndarray:
        """Rows [g0, g1) of the concatenated table as f32, spanning shard boundaries."""
        pieces = []
        g = g0
        while g < g1:
            # find shard containing g
            s_start, idx = max((s, i) for s, i in starts if s <= g)
            name, fname = shards[idx]
            meta = remote[fname][name]
            s_rows = int(meta.shape[0])
            lo = g - s_start
            hi = min(s_rows, lo + (g1 - g))
            raw = ranged(fname, meta.offset_start + lo * row_in, (hi - lo) * row_in)
            pieces.append(bf16_to_f32(raw, (hi - lo, n_cols)))
            g += hi - lo
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    n_workers = max(1, int(workers))
    pool: ProcessPoolExecutor | None = None
    raw_in = raw_out = None
    if n_workers > 1:
        ctx = get_context("fork")
        raw_in = ctx.RawArray("f", chunk_rows * n_cols)
        raw_out = ctx.RawArray("B", chunk_rows * row_bytes)
        pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                   initializer=_mxfp4_worker_init,
                                   initargs=(raw_in, raw_out, chunk_rows, n_cols, row_bytes))
        logger.info("MXFP4 process pool workers=%d", n_workers)

    t_all = time.time()
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    try:
        for start in range(start_row, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            nrows = stop - start
            t0 = time.time()
            f32 = fetch_rows(start, stop)
            t_dl = time.time()
            if pool is not None and nrows >= 8192:
                packed = _mxfp4_quantize_with_pool(f32, raw_in, raw_out, pool, n_workers, row_bytes)
            else:
                packed = gguf.quantize(f32, qtype).reshape(nrows, row_bytes)
            out[start:stop] = packed.reshape(nrows, row_bytes)
            del f32, packed
            ci = start // chunk_rows
            if start == start_row or stop == n_rows or ci % 10 == 0:
                out.flush()
                write_progress(progress_path, {"rows_done": stop, "n_rows": n_rows,
                                               "row_bytes": row_bytes, "elapsed_s": time.time() - t_all})
            dt = time.time() - t0
            if start == start_row or stop == n_rows or ci % 5 == 0:
                remain = n_chunks - (ci + 1)
                logger.info("  %d / %d rows (%.1f%%)  dl %.1fs q %.1fs tot %.1fs  ETA %.1fh",
                            stop, n_rows, 100.0 * stop / n_rows, t_dl - t0, time.time() - t_dl, dt,
                            remain * dt / 3600.0)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    out.flush()
    write_progress(progress_path, {"rows_done": n_rows, "n_rows": n_rows, "row_bytes": row_bytes,
                                   "elapsed_s": time.time() - t_all, "done": True})
    logger.info("staging complete in %.1f min", (time.time() - t_all) / 60.0)
    return staging, n_rows, n_cols


def write_gguf(outfile: Path, hparams: dict, consts: dict[str, list[int]], staging: Path, n_rows: int, n_cols: int) -> None:
    writer = gguf.GGUFWriter(str(outfile), arch="qwen4exp", use_temp_file=False)
    # informational: the main GGUF carries the authoritative PLE keys; these let the
    # sidecar describe itself and be checked against the main file
    writer.add_ple_layers([int(i) - 1 for i in hparams["ple_layer_ids"]])
    writer.add_ple_ngram_size(int(hparams["ngram_size"]))
    writer.add_ple_heads_per_ngram(int(hparams["heads_per_ngram"]))
    writer.add_embedding_length_per_layer_input(n_cols)
    writer.add_ple_layer_multipliers(consts["layer_multipliers"])
    writer.add_ple_head_offsets(consts["ngram_heads_offsets"])
    writer.add_ple_head_vocab_sizes(consts["ngram_heads_vocab_sizes"])

    qtype = gguf.GGMLQuantizationType.MXFP4
    row_bytes = int(gguf.quantize(np.zeros((1, n_cols), dtype=np.float32), qtype).nbytes)
    mm = np.memmap(staging, dtype=np.uint8, mode="r")
    if mm.size != n_rows * row_bytes:
        raise RuntimeError(f"staging size {mm.size} != {n_rows * row_bytes}")
    mm = mm.reshape(n_rows, row_bytes)
    name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight"
    writer.add_tensor(name, mm, raw_dtype=qtype)

    logger.info("writing GGUF %s", outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("wrote %s (%.1f GB)", outfile, outfile.stat().st_size / 1e9)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--cache-dir", default=str(Path.home() / "models" / "qwen38-hf-ple"))
    ap.add_argument("--outfile", default=str(Path.home() / "models" / "qwen38-ple-mxfp4.gguf"))
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--probe", action="store_true", help="print shard map and exit")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    session = make_session()
    hparams, shards, consts_files, remote, base = probe(session, args.repo)
    if args.probe:
        for idx in sorted(shards):
            name, fname = shards[idx]
            meta = remote[fname][name]
            logger.info("shard %3d %s %s %s", idx, fname, meta.dtype, tuple(meta.shape))
        return 0

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "config.flat.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    consts = read_consts(session, base, consts_files, remote)
    (cache_dir / "ple_consts.json").write_text(json.dumps(consts), encoding="utf-8")

    staging, n_rows, n_cols = convert_table(session, base, shards, remote, cache_dir, args.chunk_rows, args.workers)
    outfile = Path(args.outfile)
    require_free(outfile.parent, staging.stat().st_size + 1024 ** 3)
    write_gguf(outfile, hparams, consts, staging, n_rows, n_cols)
    if not args.keep_staging:
        staging.unlink()
        logger.info("removed staging %s", staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
