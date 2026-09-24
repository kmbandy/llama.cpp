#!/usr/bin/env python3
"""Stream DeepSeek-V4.1 Engram tables from Hugging Face via HTTP Range.

The 510 GB checkpoint does not fit. Engram lives alone in two shards
(~94.6 GiB each) and is the only content in those files:

  layers.1.engram.*  -> model-00047-of-00048.safetensors
  layers.14.engram.* -> model-00048-of-00048.safetensors

Prefers a complete local shard under cache-dir/shards/ (or cache-dir/).
Otherwise Range-gets 1M-row chunks of F8_E4M3 weight + F8_E8M0 scale.
Dequant is cheap; MXFP4 is the cost — gguf.quantize walks 16-row groups
in Python (~14s/chunk on one core). A 12-process fork pool shares the
chunk via RawArray so the 3900X's cores all run those cache-sized
groups (~1.7s/chunk). Peak extra disk is the two MXFP4 staging files
(~49 GiB each) plus the output GGUF.

Resume: if a staging file and sidecar progress JSON exist, conversion
continues from the last completed row.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import MXFP4  # noqa: E402
from gguf.utility import SafetensorRemote  # noqa: E402

logger = logging.getLogger("dsv41-engram")

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
CHUNK_ROWS = 1_000_000
DEFAULT_WORKERS = 12
ELEM_BYTES = {
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}
MIN_FREE_BYTES = 200 * 1024 ** 3

# Filled in the MXFP4 worker processes by _mxfp4_worker_init.
_MXFP4_RAW: tuple | None = None


def _blas_limits(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass


def _mxfp4_worker_init(raw_in, raw_out, max_rows: int, ncols: int, row_bytes: int) -> None:
    _blas_limits(1)
    global _MXFP4_RAW
    _MXFP4_RAW = (raw_in, raw_out, max_rows, ncols, row_bytes)


def _mxfp4_worker(job: tuple[int, int, int]) -> None:
    lo, hi, nrows = job
    if hi <= lo:
        return
    raw_in, raw_out, max_rows, ncols, row_bytes = _MXFP4_RAW  # type: ignore[misc]
    src = np.frombuffer(raw_in, dtype=np.float32, count=nrows * ncols).reshape(nrows, ncols)
    dst = np.frombuffer(raw_out, dtype=np.uint8, count=nrows * row_bytes).reshape(nrows, row_bytes)
    packed = gguf.quantize(np.ascontiguousarray(src[lo:hi]), gguf.GGMLQuantizationType.MXFP4)
    dst[lo:hi] = packed.reshape(hi - lo, row_bytes)


def mxfp4_quantize_parallel(f32: np.ndarray, workers: int) -> np.ndarray:
    """Bit-exact MXFP4 via gguf.quantize. Process-parallel for chunks large enough to bother."""
    if f32.ndim != 2:
        raise ValueError(f"expected 2D (rows, cols), got {f32.shape}")
    nrows, ncols = f32.shape
    if ncols % MXFP4.block_size:
        raise ValueError(f"ncols {ncols} not multiple of {MXFP4.block_size}")
    row_bytes = (ncols // MXFP4.block_size) * MXFP4.type_size
    n_workers = max(1, int(workers))
    if n_workers == 1 or nrows < 8192:
        return gguf.quantize(f32, gguf.GGMLQuantizationType.MXFP4).reshape(nrows, row_bytes)
    ctx = get_context("fork")
    raw_in = ctx.RawArray("f", nrows * ncols)
    raw_out = ctx.RawArray("B", nrows * row_bytes)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_mxfp4_worker_init,
        initargs=(raw_in, raw_out, nrows, ncols, row_bytes),
    ) as pool:
        return _mxfp4_quantize_with_pool(f32, raw_in, raw_out, pool, n_workers, row_bytes)


def _mxfp4_quantize_with_pool(
    f32: np.ndarray,
    raw_in,
    raw_out,
    pool: ProcessPoolExecutor,
    workers: int,
    row_bytes: int,
) -> np.ndarray:
    nrows, ncols = f32.shape
    src = np.frombuffer(raw_in, dtype=np.float32, count=nrows * ncols).reshape(nrows, ncols)
    src[:] = f32
    n_workers = min(workers, nrows)
    bounds = np.linspace(0, nrows, n_workers + 1, dtype=np.int64)
    jobs = [(int(bounds[i]), int(bounds[i + 1]), nrows) for i in range(n_workers)]
    list(pool.map(_mxfp4_worker, jobs))
    dst = np.frombuffer(raw_out, dtype=np.uint8, count=nrows * row_bytes).reshape(nrows, row_bytes)
    return np.array(dst, copy=True)


def find_local_shard(cache_dir: Path, filename: str) -> Path | None:
    name = Path(filename).name
    for candidate in (cache_dir / "shards" / name, cache_dir / name):
        marker = Path(str(candidate) + ".complete")
        expected_path = Path(str(candidate) + ".expected")
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        if marker.is_file():
            return candidate
        if expected_path.is_file():
            try:
                expected = int(expected_path.read_text(encoding="utf-8").strip())
            except ValueError:
                expected = -1
            if expected > 0 and size == expected:
                return candidate
    return None


def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "dsv41-engram-convert"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_url(session: requests.Session, url: str) -> str:
    r = session.get(url, headers={**hf_headers(), "Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.url


def http_range(session: requests.Session, url: str, start: int, size: int, timeout: int = 600) -> bytes:
    if size <= 0:
        return b""
    headers = {**hf_headers(), "Range": f"bytes={start}-{start + size - 1}"}
    last: Exception | None = None
    for attempt in range(8):
        try:
            r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if r.status_code in (401, 403) and "huggingface.co" not in url:
                # CDN auth expired; caller should re-resolve. Surface as error.
                raise PermissionError(f"HTTP {r.status_code} for range {start}+{size} at {url[:80]}")
            if r.status_code in (200, 206) and len(r.content) >= size:
                return r.content[:size]
            last = RuntimeError(f"HTTP {r.status_code} len={len(r.content)} want={size}")
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 — retry net/HTTP flakes
            last = exc
        time.sleep(min(60.0, 1.5 ** attempt))
    raise RuntimeError(f"range {start}+{size} failed after retries") from last


def flatten_hparams(raw: dict) -> dict:
    h = dict(raw)
    for key, value in (raw.get("text_config") or {}).items():
        h.setdefault(key, value)
    return h


def load_json_url(session: requests.Session, url: str) -> dict:
    r = session.get(url, headers=hf_headers(), timeout=120, allow_redirects=True)
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8"))


def f8e4m3_to_f32(buf: bytes, rows: int, cols: int) -> torch.Tensor:
    arr = np.frombuffer(buf, dtype=np.uint8, count=rows * cols).reshape(rows, cols).copy()
    return torch.from_numpy(arr).view(torch.float8_e4m3fn).float()


def e8m0_to_f32(buf: bytes, rows: int, cols: int) -> torch.Tensor:
    bits = np.frombuffer(buf, dtype=np.uint8, count=rows * cols).reshape(rows, cols)
    return torch.from_numpy(np.exp2(bits.astype(np.float32) - 127.0))


def bf16_to_f32(buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    arr = np.frombuffer(buf, dtype=np.uint8, count=n * 2).copy()
    t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape).float()
    return t.numpy().astype(np.float32)


def dequant_fp8_block32(weight: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] = (32, 32)) -> np.ndarray:
    br, bc = block
    out_f, in_f = (int(x) for x in weight.shape)
    scale_f = scale.repeat_interleave(br, 0)[:out_f]
    scale_f = scale_f.repeat_interleave(bc, 1)[:, :in_f]
    return (weight * scale_f).numpy().astype(np.float32)


def require_free(path: Path, needed: int) -> None:
    free = shutil.disk_usage(path).free
    if free < needed:
        raise RuntimeError(f"need {needed / 1e9:.1f} GB free on {path}, have {free / 1e9:.1f} GB")


def write_progress(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def convert_table(
    session: requests.Session,
    hf_url: str,
    tensors: dict[str, gguf.utility.RemoteTensor],
    bid: int,
    staging_dir: Path,
    chunk_rows: int,
    local_path: Path | None = None,
    workers: int = DEFAULT_WORKERS,
) -> tuple[Path, dict[str, np.ndarray]]:
    weight_name = f"layers.{bid}.engram.embed.weight"
    scale_name = f"layers.{bid}.engram.embed.scale"
    wt = tensors[weight_name]
    st = tensors[scale_name]
    n_rows, n_cols = (int(x) for x in wt.shape)
    scale_groups = int(st.shape[1])
    w_elem = ELEM_BYTES[wt.dtype]
    s_elem = ELEM_BYTES[st.dtype]
    if n_cols % 32:
        raise ValueError(f"engram row width {n_cols} not multiple of MXFP4 block 32")
    if n_cols % scale_groups:
        raise ValueError(f"engram row width {n_cols} not divisible by {scale_groups} scale groups")
    per_group = n_cols // scale_groups
    row_w = n_cols * w_elem
    row_s = scale_groups * s_elem

    qtype = gguf.GGMLQuantizationType.MXFP4
    row_bytes = int(gguf.quantize(np.zeros((1, n_cols), dtype=np.float32), qtype).nbytes)
    staging = staging_dir / f"engram_{bid}_MXFP4.bin"
    progress_path = staging_dir / f"engram_{bid}_MXFP4.progress.json"
    extras_path = staging_dir / f"engram_{bid}_extras.npz"

    logger.info(
        "engram layer %d: %s %s + %s %s -> MXFP4 %d x %d (%.1f GB) at %s",
        bid, wt.dtype, tuple(wt.shape), st.dtype, tuple(st.shape),
        n_rows, n_cols, n_rows * row_bytes / 1e9, staging,
    )

    extras: dict[str, np.ndarray] = {}
    use_local = local_path is not None and local_path.is_file()
    cdn = hf_url if use_local else resolve_url(session, hf_url)
    if use_local:
        logger.info("layer %d: reading local shard %s (%.1f GiB)", bid, local_path, local_path.stat().st_size / 1024 ** 3)

    def ranged(start: int, size: int) -> bytes:
        nonlocal cdn
        if use_local:
            assert local_path is not None
            with open(local_path, "rb") as fh:
                fh.seek(start)
                buf = fh.read(size)
            if len(buf) != size:
                raise RuntimeError(f"{local_path} short read at {start}+{size}: got {len(buf)}")
            return buf
        try:
            return http_range(session, cdn, start, size)
        except PermissionError:
            logger.warning("CDN auth expired, re-resolving %s", hf_url)
            cdn = resolve_url(session, hf_url)
            return http_range(session, cdn, start, size)

    # Small extras first (k/q BF16, wkv FP8+[32,32] e8m0).
    if extras_path.is_file():
        loaded = np.load(extras_path)
        extras = {k: loaded[k] for k in loaded.files}
        logger.info("reusing extras %s", extras_path)
    else:
        def pull(name: str):
            meta = tensors[name]
            return meta, ranged(meta.offset_start, meta.size)

        k_name = f"layers.{bid}.engram.k_weight"
        q_name = f"layers.{bid}.engram.q_weight"
        w_name = f"layers.{bid}.engram.wkv.weight"
        s_name = f"layers.{bid}.engram.wkv.scale"
        if k_name in tensors:
            meta, raw = pull(k_name)
            extras["k"] = bf16_to_f32(raw, tuple(int(x) for x in meta.shape))
        if q_name in tensors:
            meta, raw = pull(q_name)
            extras["q"] = bf16_to_f32(raw, tuple(int(x) for x in meta.shape))
        if w_name in tensors and s_name in tensors:
            wmeta, wraw = pull(w_name)
            smeta, sraw = pull(s_name)
            wr, wc = (int(x) for x in wmeta.shape)
            sr, sc = (int(x) for x in smeta.shape)
            extras["wkv"] = dequant_fp8_block32(
                f8e4m3_to_f32(wraw, wr, wc),
                e8m0_to_f32(sraw, sr, sc),
            )
            logger.info("wkv dequant layer %d: %s -> %s", bid, tuple(wmeta.shape), extras["wkv"].shape)
        np.savez(extras_path, **extras)
        logger.info("wrote extras %s", extras_path)

    start_row = 0
    if staging.is_file() and progress_path.is_file():
        prog = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(prog.get("n_rows", -1)) == n_rows and int(prog.get("row_bytes", -1)) == row_bytes:
            start_row = int(prog.get("rows_done", 0))
            if start_row > n_rows:
                start_row = 0
            if start_row:
                logger.info("resuming layer %d at row %d / %d", bid, start_row, n_rows)
    if start_row == 0:
        require_free(staging_dir, n_rows * row_bytes + 8 * 1024 ** 3)
        out = np.memmap(staging, dtype=np.uint8, mode="w+", shape=(n_rows, row_bytes))
        out.flush()
    else:
        if staging.stat().st_size != n_rows * row_bytes:
            raise RuntimeError(
                f"staging {staging} size {staging.stat().st_size} != expected {n_rows * row_bytes}"
            )
        out = np.memmap(staging, dtype=np.uint8, mode="r+", shape=(n_rows, row_bytes))

    t_all = time.time()
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    n_workers = max(1, int(workers))
    pool: ProcessPoolExecutor | None = None
    raw_in = raw_out = None
    if n_workers > 1:
        ctx = get_context("fork")
        raw_in = ctx.RawArray("f", chunk_rows * n_cols)
        raw_out = ctx.RawArray("B", chunk_rows * row_bytes)
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_mxfp4_worker_init,
            initargs=(raw_in, raw_out, chunk_rows, n_cols, row_bytes),
        )
        logger.info("layer %d: MXFP4 process pool workers=%d", bid, n_workers)

    try:
        for start in range(start_row, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            nrows = stop - start
            t0 = time.time()
            wbuf = ranged(wt.offset_start + start * row_w, nrows * row_w)
            sbuf = ranged(st.offset_start + start * row_s, nrows * row_s)
            t_dl = time.time()
            chunk = f8e4m3_to_f32(wbuf, nrows, n_cols)
            scale = e8m0_to_f32(sbuf, nrows, scale_groups)
            chunk = chunk * scale.repeat_interleave(per_group, 1)[:, :n_cols]
            f32 = chunk.cpu().numpy().astype(np.float32, copy=False)
            del chunk, scale, wbuf, sbuf
            t_dq = time.time()
            if pool is not None and raw_in is not None and raw_out is not None and nrows >= 8192:
                packed = _mxfp4_quantize_with_pool(f32, raw_in, raw_out, pool, n_workers, row_bytes)
            else:
                packed = gguf.quantize(f32, qtype).reshape(nrows, row_bytes)
            out[start:stop] = packed.reshape(nrows, row_bytes)
            del f32, packed
            if start == start_row or stop == n_rows or ((start // chunk_rows) % 10 == 0):
                out.flush()
                write_progress(progress_path, {
                    "bid": bid,
                    "rows_done": stop,
                    "n_rows": n_rows,
                    "row_bytes": row_bytes,
                    "elapsed_s": time.time() - t_all,
                })
            dt = time.time() - t0
            done_chunks = (stop + chunk_rows - 1) // chunk_rows
            remain = n_chunks - done_chunks
            eta = remain * dt
            if start == start_row or stop == n_rows or (start // chunk_rows) % 5 == 0:
                logger.info(
                    "  layer %d: %d / %d rows (%.1f%%)  dl %.1fs dq %.1fs q %.1fs tot %.1fs  ETA %.1fh",
                    bid, stop, n_rows, 100.0 * stop / n_rows,
                    t_dl - t0, t_dq - t_dl, time.time() - t_dq, dt, eta / 3600.0,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    out.flush()
    write_progress(progress_path, {
        "bid": bid,
        "rows_done": n_rows,
        "n_rows": n_rows,
        "row_bytes": row_bytes,
        "elapsed_s": time.time() - t_all,
        "done": True,
    })
    logger.info("engram layer %d: staging complete in %.1f min", bid, (time.time() - t_all) / 60.0)
    return staging, extras


def write_gguf(
    outfile: Path,
    hparams: dict,
    tables: dict[int, Path],
    extras: dict[int, dict[str, np.ndarray]],
) -> None:
    writer = gguf.GGUFWriter(str(outfile), arch="deepseek41", use_temp_file=False)
    writer.add_engram_head_count(int(hparams["engram_n_heads"]))
    writer.add_engram_key_length(int(hparams["engram_head_dim"]))
    writer.add_engram_max_ngram_size(int(hparams["engram_max_ngram_size"]))
    writer.add_engram_layer_ids(list(hparams["engram_layer_ids"]))
    if "engram_vocab_size" in hparams:
        writer.add_engram_vocab_size(int(hparams["engram_vocab_size"]))
    if "engram_pad_token_id" in hparams:
        writer.add_engram_pad_token_id(int(hparams["engram_pad_token_id"]))
    if "engram_compressed_vocab_size" in hparams:
        writer.add_engram_compressed_vocab_size(int(hparams["engram_compressed_vocab_size"]))

    qtype = gguf.GGMLQuantizationType.MXFP4
    n_cols = int(hparams["engram_head_dim"])
    row_bytes = int(gguf.quantize(np.zeros((1, n_cols), dtype=np.float32), qtype).nbytes)

    for bid, path in tables.items():
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        n_rows = mm.size // row_bytes
        mm = mm.reshape(n_rows, row_bytes)
        name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_EMBD].format(bid=bid) + ".weight"
        writer.add_tensor(name, mm, raw_dtype=qtype)
        extra = extras.get(bid, {})
        if "k" in extra:
            tname = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_K].format(bid=bid) + ".weight"
            writer.add_tensor(tname, np.ascontiguousarray(extra["k"]))
        if "q" in extra:
            tname = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_Q].format(bid=bid) + ".weight"
            writer.add_tensor(tname, np.ascontiguousarray(extra["q"]))
        if "wkv" in extra:
            tname = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_WKV].format(bid=bid) + ".weight"
            writer.add_tensor(tname, np.ascontiguousarray(extra["wkv"]))

    logger.info("writing GGUF %s", outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("wrote %s (%.1f GB)", outfile, outfile.stat().st_size / 1e9)


def probe(session: requests.Session, repo: str) -> tuple[dict, dict[int, str], dict[str, gguf.utility.RemoteTensor]]:
    base = f"{SafetensorRemote.BASE_DOMAIN}/{repo}/resolve/main"
    logger.info("fetching config + weight map from %s", repo)
    hparams = flatten_hparams(load_json_url(session, f"{base}/config.json"))
    index = load_json_url(session, f"{base}/model.safetensors.index.json")
    weight_map = index["weight_map"]
    layer_ids = [int(x) for x in hparams["engram_layer_ids"]]
    shards: dict[int, str] = {}
    for bid in layer_ids:
        key = f"layers.{bid}.engram.embed.weight"
        if key not in weight_map:
            raise KeyError(f"{key} missing from weight_map")
        shards[bid] = weight_map[key]

    logger.info("Engram layer ids %s", layer_ids)
    logger.info("Engram shards: %s", shards)
    remote: dict[str, gguf.utility.RemoteTensor] = {}
    for bid, filename in shards.items():
        url = f"{base}/{filename}"
        found = SafetensorRemote.get_list_tensors(url)
        n_engram = sum(1 for k in found if ".engram." in k)
        n_other = len(found) - n_engram
        logger.info("%s: %d tensors (%d engram, %d other)", filename, len(found), n_engram, n_other)
        for name, meta in found.items():
            logger.info(
                "  %s  dtype=%s shape=%s size=%.2f GiB offset=%d",
                name, meta.dtype, tuple(meta.shape), meta.size / 1024 ** 3, meta.offset_start,
            )
            if n_other:
                logger.warning("shard %s contains non-engram tensors", filename)
        remote.update(found)
    return hparams, shards, remote


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--cache-dir", default=str(Path.home() / "models" / "dsv41-hf-engram"))
    ap.add_argument("--outfile", default=str(Path.home() / "models" / "dsv41-engram-mxfp4.gguf"))
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="MXFP4 process-pool size (default 12 = 3900X physical cores; leave SMT for llama-server)",
    )
    ap.add_argument("--probe", action="store_true", help="print shard map and exit")
    ap.add_argument("--keep-staging", action="store_true")
    ap.add_argument(
        "--only-layers", type=int, nargs="+", default=None,
        help="convert only these Engram layer ids (skip GGUF write). Resume later without this flag.",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_dir / "staging"
    staging_dir.mkdir(exist_ok=True)

    require_free(cache_dir, MIN_FREE_BYTES)

    session = make_session()
    hparams, shards, remote = probe(session, args.repo)
    (cache_dir / "config.flat.json").write_text(json.dumps({
        k: hparams[k] for k in hparams if k.startswith("engram_") or k in (
            "hidden_size", "num_hidden_layers", "num_nextn_predict_layers",
        )
    }, indent=2, default=str), encoding="utf-8")

    if args.probe:
        return 0

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    tables: dict[int, Path] = {}
    extras: dict[int, dict[str, np.ndarray]] = {}
    base = f"{SafetensorRemote.BASE_DOMAIN}/{args.repo}/resolve/main"
    workers = max(1, int(args.workers))
    only = set(args.only_layers) if args.only_layers else None
    logger.info("MXFP4 workers=%d only_layers=%s", workers, sorted(only) if only else "all")
    for bid, filename in shards.items():
        if only is not None and bid not in only:
            logger.info("skipping layer %d (--only-layers)", bid)
            continue
        local = find_local_shard(cache_dir, filename)
        if local is None:
            logger.info("layer %d: no complete local shard for %s, streaming via HTTP Range", bid, filename)
        staging, extra = convert_table(
            session, f"{base}/{filename}", remote, bid, staging_dir, args.chunk_rows,
            local_path=local, workers=workers,
        )
        tables[bid] = staging
        extras[bid] = extra

    if only is not None:
        logger.info("only-layers done; not writing GGUF")
        return 0

    write_gguf(outfile, hparams, tables, extras)
    if not args.keep_staging:
        for path in tables.values():
            logger.info("deleting staging %s", path)
            path.unlink(missing_ok=True)
            path.with_name(path.name.replace(".bin", ".progress.json")).unlink(missing_ok=True)
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
