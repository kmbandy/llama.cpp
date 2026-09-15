"""Per-expert quantization for wp-forge.

Generalizes the MXFP4 process-pool pattern from dsv41_engram_from_hf to any
gguf-py qtype.  The fork pool shares the f32 source and packed destination
via multiprocessing.RawArray so no per-row copy crosses the process boundary.

Imports:
  - _blas_limits, dequant_fp8_block32  from conversion.dsv41_engram_from_hf
  - ModelBase (repack_mxfp4_blocks)    from conversion.base
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "gguf-py"))
sys.path.insert(0, str(_REPO))

import gguf  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

from conversion.dsv41_engram_from_hf import _blas_limits, dequant_fp8_block32  # noqa: E402
from conversion.base import ModelBase  # noqa: E402


class QuantError(Exception):
    """Raised when a quantized output's row byte-size does not match the gguf-py expectation."""


# ---------------------------------------------------------------------------
# Process-pool plumbing (same pattern as dsv41_engram_from_hf._mxfp4_worker*)
# ---------------------------------------------------------------------------

# Set in worker processes by _quant_worker_init; None in the parent.
_WORKER_CTX: tuple | None = None


def _quant_worker_init(
    raw_in, raw_out, max_rows: int, ncols: int, row_bytes: int,
    qtype: gguf.GGMLQuantizationType,
) -> None:
    _blas_limits(1)
    global _WORKER_CTX
    _WORKER_CTX = (raw_in, raw_out, max_rows, ncols, row_bytes, qtype)


def _quant_worker(job: tuple[int, int, int]) -> None:
    lo, hi, nrows = job
    if hi <= lo:
        return
    raw_in, raw_out, _max_rows, ncols, row_bytes, qtype = _WORKER_CTX  # type: ignore[misc]
    src = np.frombuffer(raw_in, dtype=np.float32, count=nrows * ncols).reshape(nrows, ncols)
    dst = np.frombuffer(raw_out, dtype=np.uint8, count=nrows * row_bytes).reshape(nrows, row_bytes)
    packed = gguf.quantize(np.ascontiguousarray(src[lo:hi]), qtype)
    dst[lo:hi] = packed.reshape(hi - lo, row_bytes)


def _quant_with_pool(
    f32: np.ndarray, raw_in, raw_out,
    pool: ProcessPoolExecutor, workers: int, row_bytes: int,
) -> np.ndarray:
    nrows, ncols = f32.shape
    src = np.frombuffer(raw_in, dtype=np.float32, count=nrows * ncols).reshape(nrows, ncols)
    src[:] = f32
    n_workers = min(workers, nrows)
    bounds = np.linspace(0, nrows, n_workers + 1, dtype=np.int64)
    jobs = [(int(bounds[i]), int(bounds[i + 1]), nrows) for i in range(n_workers)]
    list(pool.map(_quant_worker, jobs))
    dst = np.frombuffer(raw_out, dtype=np.uint8, count=nrows * row_bytes).reshape(nrows, row_bytes)
    return np.array(dst, copy=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_expert(
    role_arrays: dict[str, np.ndarray],
    qtype: str,
    *,
    workers: int,
) -> dict[str, np.ndarray]:
    """Quantize f32 gate/up/down arrays for ONE expert into packed rows.

    For ``workers > 1`` the work is chunked over rows across a fork process
    pool (same pattern as dsv41_engram_from_hf.mxfp4_quantize_parallel).
    For ``workers <= 1`` quantization runs inline with no pool.

    Every output row's byte width is checked against the size gguf-py derives
    for the given qtype and shape (GGML_QUANT_SIZES); a mismatch raises
    QuantError naming the role, shape, and qtype.
    """
    qt = gguf.GGMLQuantizationType[qtype.upper()]
    block_size, type_size = GGML_QUANT_SIZES[qt]
    result: dict[str, np.ndarray] = {}

    for role, f32 in role_arrays.items():
        if f32.ndim != 2:
            raise QuantError(f"{role}: expected 2D (rows, cols), got {f32.shape}")
        nrows, ncols = f32.shape
        if ncols % block_size:
            raise QuantError(
                f"{role}: cols {ncols} not a multiple of {qt.name} block_size {block_size}"
            )
        expected_row_bytes = ncols // block_size * type_size

        n_workers = max(1, int(workers))
        if n_workers == 1:
            packed = gguf.quantize(f32, qt)
        else:
            ctx = get_context("fork")
            raw_in = ctx.RawArray("f", nrows * ncols)
            raw_out = ctx.RawArray("B", nrows * expected_row_bytes)
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_quant_worker_init,
                initargs=(raw_in, raw_out, nrows, ncols, expected_row_bytes, qt),
            ) as pool:
                packed = _quant_with_pool(f32, raw_in, raw_out, pool, n_workers, expected_row_bytes)

        # Row-size check: the GLM layer-8 lesson applied at the source.
        actual_row_bytes = packed.shape[1] if packed.ndim == 2 else packed.size
        if actual_row_bytes != expected_row_bytes:
            raise QuantError(
                f"{role}: row byte size {actual_row_bytes} != expected {expected_row_bytes} "
                f"(shape={f32.shape}, qtype={qtype})"
            )
        result[role] = packed

    return result


def lossless_repack(
    weight, scale, source_format: str, qtype: str,
) -> np.ndarray | None:
    """Return lossless-repacked bytes when the source is already in the target format.

    DS4.1 mxfp4 packs → ggml MXFP4: delegates to ModelBase.repack_mxfp4_blocks.
    Returns None for any other (source_format, qtype) combination so the caller
    falls back to dequant→quant.
    """
    if source_format == "mxfp4" and qtype.upper() == "MXFP4":
        return ModelBase.repack_mxfp4_blocks(weight, scale)
    return None


def dequant(weight, scale, source_format: str) -> np.ndarray:
    """Dequantize a packed/scaled weight to f32.

    Single entry point for the fallback path.

    Supported source_format values:
      - "fp8_block32" – DS4.1 engram F8_E4M3 weight + F8_E8M0 scale, 32x32 block
    """
    if source_format == "fp8_block32":
        return dequant_fp8_block32(weight, scale)
    raise ValueError(f"dequant: unsupported source_format {source_format!r}")
