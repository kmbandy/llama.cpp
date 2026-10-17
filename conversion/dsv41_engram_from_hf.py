#!/usr/bin/env python3
"""Stream DeepSeek-V4.1 Engram tables from Hugging Face via HTTP Range.

The 510 GB checkpoint does not fit. Engram lives alone in two shards
(~94.6 GiB each) and is the only content in those files:

  layers.{1,14}.engram.* -> resolved from model.safetensors.index.json (V4.1: shards 47, 48)

Prefers a complete local shard under cache-dir/shards/ (or cache-dir/).
Otherwise Range-gets 1M-row chunks of F8_E4M3 weight + F8_E8M0 scale.
Dequant is cheap; MXFP4 is the cost — gguf.quantize walks 16-row groups
in Python (~14s/chunk on one core). A 12-process fork pool shares the
chunk via RawArray so the 3900X's cores all run those cache-sized
groups (~1.7s/chunk). The MXFP4 rows are written straight into the final
GGUF through a memmap (no staging copy): peak extra disk is the output itself.

Resume: the output and outfile.progress.json hold the layer's rows_done, the
extras flag and a done marker; a rerun continues from the last flushed chunk.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import MXFP4  # noqa: E402
from gguf.utility import RemoteTensor, SafetensorRemote  # noqa: E402

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


def make_ranged(session: requests.Session | None, hf_url: str, local_path: Path | None) -> Callable[[int, int], bytes]:
    """Byte-range reader over one shard: the local file when present, else HF Range (CDN URL resolved once)."""
    use_local = local_path is not None and local_path.is_file()
    if use_local:
        logger.info("reading local shard %s (%.1f GiB)", local_path, local_path.stat().st_size / 1024 ** 3)
    cdn = hf_url if use_local else resolve_url(session, hf_url)

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

    return ranged


def local_shard_tensors(path: Path) -> dict[str, RemoteTensor]:
    """Tensor metadata read from a local safetensors header (same fields as SafetensorRemote.get_list_tensors)."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n).decode("utf-8"))
    data_start = 8 + n
    out: dict[str, RemoteTensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = RemoteTensor(
            dtype=meta["dtype"], shape=tuple(int(x) for x in meta["shape"]),
            offset_start=data_start + start, size=end - start, url=str(path),
        )
    return out


def progress_path_for(outfile: Path) -> Path:
    return outfile.with_name(outfile.name + ".progress.json")


def progress_done(outfile: Path) -> bool:
    p = progress_path_for(outfile)
    return p.is_file() and bool(json.loads(p.read_text(encoding="utf-8")).get("done"))


def engram_names(bid: int) -> dict[str, str]:
    """GGUF tensor names for one layer: embd (MXFP4), k/q/wkv (f32)."""
    return {
        "embd": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_EMBD].format(bid=bid) + ".weight",
        "k": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_K].format(bid=bid) + ".weight",
        "q": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_Q].format(bid=bid) + ".weight",
        "wkv": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_WKV].format(bid=bid) + ".weight",
    }


def _extra_keys(bid: int, tensors: dict[str, RemoteTensor]) -> list[tuple[str, str]]:
    """(key, source tensor) for each f32 extra present in this layer, in write order."""
    keys = [(k, f"layers.{bid}.engram.{k}_weight") for k in ("k", "q")]
    keys = [(k, s) for k, s in keys if s in tensors]
    w = f"layers.{bid}.engram.wkv.weight"
    if w in tensors and f"layers.{bid}.engram.wkv.scale" in tensors:
        keys.append(("wkv", w))
    return keys


def engram_extras(ranged: Callable[[int, int], bytes], tensors: dict[str, RemoteTensor], bid: int) -> dict[str, np.ndarray]:
    """k/q (BF16) and wkv (FP8 + 32x32 E8M0 block scales), all dequantized to f32."""
    extras: dict[str, np.ndarray] = {}
    for key, src in _extra_keys(bid, tensors):
        meta = tensors[src]
        raw = ranged(meta.offset_start, meta.size)
        shape = tuple(int(x) for x in meta.shape)
        if key == "wkv":
            smeta = tensors[f"layers.{bid}.engram.wkv.scale"]
            sraw = ranged(smeta.offset_start, smeta.size)
            wr, wc = shape
            sr, sc = (int(x) for x in smeta.shape)
            extras[key] = dequant_fp8_block32(f8e4m3_to_f32(raw, wr, wc), e8m0_to_f32(sraw, sr, sc))
            logger.info("wkv dequant layer %d: %s -> %s", bid, shape, extras[key].shape)
        else:
            extras[key] = bf16_to_f32(raw, shape)
    return extras


LOOKAHEAD = 2  # chunks in flight ahead of the one being quantized


def convert_table(
    ranged: Callable[[int, int], bytes],
    tensors: dict[str, RemoteTensor],
    bid: int,
    target: Path,
    offset: int,
    *,
    start_row: int = 0,
    chunk_rows: int = CHUNK_ROWS,
    workers: int = DEFAULT_WORKERS,
    checkpoint: Callable[[int], None] | None = None,
) -> None:
    """Stream the layer's FP8 rows (from row start_row) into target at byte offset as MXFP4 rows.

    checkpoint(rows_done) runs after each chunk is flushed, so a restart resumes there.
    """
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

    logger.info(
        "engram layer %d: %s %s + %s %s -> MXFP4 %d x %d (%.1f GB) at %s+%d",
        bid, wt.dtype, tuple(wt.shape), st.dtype, tuple(st.shape),
        n_rows, n_cols, n_rows * row_bytes / 1e9, target, offset,
    )
    out = np.memmap(target, dtype=np.uint8, mode="r+", offset=offset, shape=(n_rows, row_bytes))

    t_all = time.time()
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    n_workers = max(1, int(workers))
    pool: ProcessPoolExecutor | None = None
    raw_in = raw_out = None
    if n_workers > 1 and n_rows >= 8192:  # the pool is only used for chunks of >= 8192 rows
        ctx = get_context("fork")
        cap = min(chunk_rows, n_rows)
        raw_in = ctx.RawArray("f", cap * n_cols)
        raw_out = ctx.RawArray("B", cap * row_bytes)
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_mxfp4_worker_init,
            initargs=(raw_in, raw_out, cap, n_cols, row_bytes),
        )
        logger.info("layer %d: MXFP4 process pool workers=%d", bid, n_workers)

    # Download overlaps quantization: chunks are fetched LOOKAHEAD ahead, each
    # weight range split over `conns` parallel Range requests (one serial
    # stream left the link idle during every quantize step).
    conns = max(1, int(os.environ.get("WP_FORGE_DOWNLOADS", "8")))
    conn_pool = ThreadPoolExecutor(max_workers=conns, thread_name_prefix="engram-range")
    ahead_pool = ThreadPoolExecutor(max_workers=LOOKAHEAD, thread_name_prefix="engram-chunk")

    def fetch(start: int) -> tuple[bytes, bytes]:
        nrows = min(start + chunk_rows, n_rows) - start
        w0, wn = wt.offset_start + start * row_w, nrows * row_w
        step = -(-wn // conns)
        parts = [conn_pool.submit(ranged, w0 + o, min(step, wn - o)) for o in range(0, wn, step)]
        sbuf = ranged(st.offset_start + start * row_s, nrows * row_s)
        return b"".join(f.result() for f in parts), sbuf

    starts = list(range(start_row, n_rows, chunk_rows))
    pending = deque(ahead_pool.submit(fetch, s) for s in starts[:LOOKAHEAD])
    try:
        for i, start in enumerate(starts):
            stop = min(start + chunk_rows, n_rows)
            nrows = stop - start
            t0 = time.time()
            wbuf, sbuf = pending.popleft().result()
            if i + LOOKAHEAD < len(starts):
                pending.append(ahead_pool.submit(fetch, starts[i + LOOKAHEAD]))
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
            out.flush()
            if checkpoint is not None:
                checkpoint(stop)
            dt = time.time() - t0
            done_chunks = (stop + chunk_rows - 1) // chunk_rows
            eta = (n_chunks - done_chunks) * dt
            if start == start_row or stop == n_rows or (start // chunk_rows) % 5 == 0:
                logger.info(
                    "  layer %d: %d / %d rows (%.1f%%)  dl %.1fs dq %.1fs q %.1fs tot %.1fs  ETA %.1fh",
                    bid, stop, n_rows, 100.0 * stop / n_rows,
                    t_dl - t0, t_dq - t_dl, time.time() - t_dq, dt, eta / 3600.0,
                )
    finally:
        ahead_pool.shutdown(wait=True, cancel_futures=True)
        conn_pool.shutdown(wait=True, cancel_futures=True)
        if pool is not None:
            pool.shutdown(wait=True)

    out.flush()
    logger.info("engram layer %d: MXFP4 rows complete in %.1f min", bid, (time.time() - t_all) / 60.0)


def _engram_kv(writer: gguf.GGUFWriter, hparams: dict) -> None:
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


def plan_engram_gguf(
    hparams: dict, remote_tensors: dict[str, RemoteTensor],
) -> list[tuple[str, tuple[int, ...], gguf.GGMLQuantizationType | None, int]]:
    """Tensor infos write_gguf emits, in its order: per layer embd, then k/q/wkv when present.

    Each entry is (gguf name, shape, raw_dtype, nbytes). embd is (rows, row_bytes) MXFP4 bytes;
    raw_dtype None means F32 with the numpy shape given.
    """
    qtype = gguf.GGMLQuantizationType.MXFP4
    plan: list[tuple[str, tuple[int, ...], gguf.GGMLQuantizationType | None, int]] = []
    for bid in (int(x) for x in hparams["engram_layer_ids"]):
        names = engram_names(bid)
        n_rows, n_cols = (int(x) for x in remote_tensors[f"layers.{bid}.engram.embed.weight"].shape)
        row_bytes = int(gguf.quantize(np.zeros((1, n_cols), dtype=np.float32), qtype).nbytes)
        plan.append((names["embd"], (n_rows, row_bytes), qtype, n_rows * row_bytes))
        for key, src in _extra_keys(bid, remote_tensors):
            shape = tuple(int(x) for x in remote_tensors[src].shape)
            plan.append((names[key], shape, None, int(np.prod(shape)) * 4))
    return plan


def write_skeleton(
    path: Path, hparams: dict, layout: list, preallocate: bool,
) -> tuple[dict[str, int], int]:
    """GGUF header, KV and tensor infos through GGUFWriter. Returns each tensor's absolute data
    offset and the final file size. preallocate sizes the file to that size (sparse); otherwise
    only the header is written, which is enough to get the offsets."""
    writer = gguf.GGUFWriter(str(path), arch="deepseek41", use_temp_file=False)
    _engram_kv(writer, hparams)
    for name, shape, raw, nbytes in layout:
        if raw is None:
            writer.add_tensor_info(name, shape, np.dtype(np.float32), nbytes)
        else:
            writer.add_tensor_info(name, shape, np.dtype(np.uint8), nbytes, raw_dtype=raw)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    fout = writer.fout[0]
    writer.write_padding(fout, fout.tell())
    offsets: dict[str, int] = {}
    pos = fout.tell()
    for name, _, _, nbytes in layout:
        offsets[name] = pos
        pos += gguf.GGUFWriter.ggml_pad(nbytes, writer.data_alignment)
    fout.flush()
    if preallocate:
        fout.truncate(pos)
    writer.close()
    return offsets, pos


def _write_bytes(path: Path, offset: int, buf: np.ndarray) -> None:
    mm = np.memmap(path, dtype=np.uint8, mode="r+", offset=offset, shape=(buf.size,))
    mm[:] = buf
    mm.flush()


def build_engram_gguf(
    outfile: Path,
    hparams: dict,
    remote: dict[str, RemoteTensor],
    ranges: dict[int, Callable[[int, int], bytes]],
    *,
    chunk_rows: int = CHUNK_ROWS,
    workers: int = DEFAULT_WORKERS,
    only: set[int] | None = None,
    on_layer: Callable[[int, float], None] | None = None,
) -> bool:
    """Write the final engram GGUF in place: skeleton first, then MXFP4 rows and f32 extras straight
    into it through memmap. Resumable through outfile + '.progress.json'. Returns True when complete."""
    layout = plan_engram_gguf(hparams, remote)
    names = [name for name, *_ in layout]
    scratch = outfile.with_name(outfile.name + ".layout")
    offsets, size = write_skeleton(scratch, hparams, layout, preallocate=False)
    scratch.unlink()

    prog_path = progress_path_for(outfile)
    if prog_path.is_file():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        if prog.get("size") != size or prog.get("names") != names or outfile.stat().st_size != size:
            raise RuntimeError(f"{prog_path} does not match the plan; delete {outfile} and {prog_path} to restart")
        if prog.get("done"):
            logger.info("engram GGUF already complete: %s", outfile)
            return True
        logger.info("resuming engram GGUF %s", outfile)
    else:
        if outfile.exists():
            raise FileExistsError(f"{outfile} exists without {prog_path}; refusing to overwrite")
        require_free(outfile.parent, size + 8 * 1024 ** 3)
        write_skeleton(outfile, hparams, layout, preallocate=True)
        prog = {"size": size, "names": names, "layers": {}, "done": False}
        write_progress(prog_path, prog)

    layer_ids = [int(x) for x in hparams["engram_layer_ids"]]
    for bid in layer_ids:
        if only is not None and bid not in only:
            continue
        lp = prog["layers"].setdefault(str(bid), {"rows_done": 0, "extras": False, "done": False})
        if lp["done"]:
            continue
        t0 = time.time()
        ranged = ranges[bid]
        keys = engram_names(bid)
        if not lp["extras"]:
            for key, arr in engram_extras(ranged, remote, bid).items():
                buf = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
                _write_bytes(outfile, offsets[keys[key]], buf)
            lp["extras"] = True
            write_progress(prog_path, prog)

        def checkpoint(rows_done: int, lp: dict = lp) -> None:
            lp["rows_done"] = rows_done
            write_progress(prog_path, prog)

        convert_table(
            ranged, remote, bid, outfile, offsets[keys["embd"]],
            start_row=lp["rows_done"], chunk_rows=chunk_rows, workers=workers, checkpoint=checkpoint,
        )
        lp["done"] = True
        write_progress(prog_path, prog)
        if on_layer is not None:
            on_layer(bid, time.time() - t0)

    prog["done"] = all(prog["layers"].get(str(b), {}).get("done") for b in layer_ids)
    write_progress(prog_path, prog)
    return bool(prog["done"])


def write_gguf(
    outfile: Path,
    hparams: dict,
    tables: dict[int, Path],
    extras: dict[int, dict[str, np.ndarray]],
) -> None:
    """Reference writer: MXFP4 tables already staged as bins, then copied into a new GGUF."""
    writer = gguf.GGUFWriter(str(outfile), arch="deepseek41", use_temp_file=False)
    _engram_kv(writer, hparams)

    qtype = gguf.GGMLQuantizationType.MXFP4
    n_cols = int(hparams["engram_head_dim"])
    row_bytes = int(gguf.quantize(np.zeros((1, n_cols), dtype=np.float32), qtype).nbytes)

    for bid, path in tables.items():
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        n_rows = mm.size // row_bytes
        mm = mm.reshape(n_rows, row_bytes)
        names = engram_names(bid)
        writer.add_tensor(names["embd"], mm, raw_dtype=qtype)
        extra = extras.get(bid, {})
        for key in ("k", "q", "wkv"):
            if key in extra:
                writer.add_tensor(names[key], np.ascontiguousarray(extra[key]))

    logger.info("writing GGUF %s", outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("wrote %s (%.1f GB)", outfile, outfile.stat().st_size / 1e9)


def probe(session: requests.Session, repo: str, revision: str = "main") -> tuple[dict, dict[int, str], dict[str, gguf.utility.RemoteTensor]]:
    base = f"{SafetensorRemote.BASE_DOMAIN}/{repo}/resolve/{revision}"
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
    ap.add_argument("--revision", default="main", help="HF commit sha/tag/branch to pin every fetch to")
    ap.add_argument("--cache-dir", default=str(Path.home() / "models" / "dsv41-hf-engram"))
    ap.add_argument("--outfile", default=str(Path.home() / "models" / "dsv41-engram-mxfp4.gguf"))
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="MXFP4 process-pool size (default 12 = 3900X physical cores; leave SMT for llama-server)",
    )
    ap.add_argument("--probe", action="store_true", help="print shard map and exit")
    ap.add_argument(
        "--keep-staging", action="store_true",
        help="accepted for compatibility; the in-place path writes no staging files",
    )
    ap.add_argument(
        "--only-layers", type=int, nargs="+", default=None,
        help="convert only these Engram layer ids into the output. Rerun without this flag to finish.",
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    hparams, shards, remote = probe(session, args.repo, args.revision)
    (cache_dir / "config.flat.json").write_text(json.dumps({
        k: hparams[k] for k in hparams if k.startswith("engram_") or k in (
            "hidden_size", "num_hidden_layers", "num_nextn_predict_layers",
        )
    }, indent=2, default=str), encoding="utf-8")

    if args.probe:
        return 0

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    base = f"{SafetensorRemote.BASE_DOMAIN}/{args.repo}/resolve/{args.revision}"
    workers = max(1, int(args.workers))
    only = set(args.only_layers) if args.only_layers else None
    logger.info("MXFP4 workers=%d only_layers=%s", workers, sorted(only) if only else "all")
    ranges: dict[int, Callable[[int, int], bytes]] = {}
    for bid, filename in shards.items():
        if only is not None and bid not in only:
            logger.info("skipping layer %d (--only-layers)", bid)
            continue
        local = find_local_shard(cache_dir, filename)
        if local is None:
            logger.info("layer %d: no complete local shard for %s, streaming via HTTP Range", bid, filename)
        ranges[bid] = make_ranged(session, f"{base}/{filename}", local)

    done = build_engram_gguf(
        outfile, hparams, remote, ranges, chunk_rows=args.chunk_rows, workers=workers, only=only,
        on_layer=lambda bid, secs: logger.info("layer %d written in %.1f min", bid, secs / 60.0),
    )
    if not done:
        logger.info("only-layers done; output is incomplete, rerun without --only-layers to finish")
        return 0
    logger.info("wrote %s (%.1f GB)", outfile, outfile.stat().st_size / 1e9)
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
