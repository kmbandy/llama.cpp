"""In-process quantization to any ggml type, through libggml-base's own
``ggml_quantize_chunk`` (ctypes).

gguf-py only produces a handful of types (F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/
TQ*/MXFP4); the K-quants and the IQ family exist only in C. Calling the C
quantizer directly gives the forge every type llama-quantize can make,
byte-identical to it, with an optional importance matrix, and without a
BF16 GGUF round trip through llama-quantize.

The library is found, in order: $WP_FORGE_GGML_LIB, then
<repo>/build-forge/bin/libggml-base.so (a CPU-only build made for the forge:
``cmake -B build-forge -DGGML_HIP=OFF -DGGML_CUDA=OFF && cmake --build
build-forge --target ggml-base``). Nothing else is searched: GPU builds of
the same repo belong to other work.

ctypes releases the GIL for the duration of each foreign call, so row chunks
run in parallel on a thread pool.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "gguf-py"))
import gguf  # noqa: E402

# The cases of ggml_quantize_chunk's switch (ggml/src/ggml.c) that gguf-py's
# GGMLQuantizationType also names -- a type outside this set would hit
# GGML_ABORT inside the library and take the whole process down, so it is
# refused here instead. ML8_FP8 / FP8_B128 are left out on purpose: the C
# quantizers are the unrotated forms, the forge's ml8 types are rotated
# (see ml8.py).
QUANTIZABLE = frozenset({
    "F16", "BF16", "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "MXFP4", "NVFP4", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "TQ1_0", "TQ2_0",
    "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ1_S", "IQ1_M",
    "IQ4_NL", "IQ4_XS",
}) & frozenset(t.name for t in gguf.GGMLQuantizationType)

_lib = None
_lib_lock = threading.Lock()
_inited: set[int] = set()


class GgmlQuantError(ValueError):
    """A type, shape or library problem, named."""


def lib_path() -> Path:
    env = os.environ.get("WP_FORGE_GGML_LIB")
    if env:
        return Path(env)
    return _REPO / "build-forge" / "bin" / "libggml-base.so"


def _load():
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        p = lib_path()
        if not p.is_file():
            raise GgmlQuantError(
                f"libggml-base not found at {p}: build it with `cmake -B build-forge "
                "-DGGML_HIP=OFF -DGGML_CUDA=OFF && cmake --build build-forge --target "
                "ggml-base -j`, or point WP_FORGE_GGML_LIB at one")
        lib = ctypes.CDLL(str(p))
        lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        lib.ggml_quantize_init.restype = None
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ggml_row_size.restype = ctypes.c_size_t
        _lib = lib
        return lib


def _qtype(name: str) -> gguf.GGMLQuantizationType:
    up = name.upper()
    if up not in QUANTIZABLE:
        raise GgmlQuantError(f"ggml type {name!r} is not quantizable in-process "
                             f"(have: {', '.join(sorted(t.lower() for t in QUANTIZABLE))})")
    return gguf.GGMLQuantizationType[up]


def is_quantizable(name: str) -> bool:
    return name.upper() in QUANTIZABLE


def block_size(name: str) -> int:
    return gguf.GGML_QUANT_SIZES[_qtype(name)][0]


def row_bytes(name: str, n_per_row: int) -> int:
    bs, ts = gguf.GGML_QUANT_SIZES[_qtype(name)]
    if n_per_row % bs:
        raise GgmlQuantError(f"{name}: row of {n_per_row} is not a multiple of block {bs}")
    return n_per_row // bs * ts


def requires_imatrix(name: str) -> bool:
    return bool(_load().ggml_quantize_requires_imatrix(int(_qtype(name))))


def quantize(f32: np.ndarray, name: str, imatrix: np.ndarray | None = None,
             workers: int | None = None, chunk_rows: int = 256) -> np.ndarray:
    """f32 [..., n_per_row] -> uint8 [..., row_bytes]. imatrix: f32 [n_per_row]
    (per-column importance), required for the types the library says need it."""
    qt = _qtype(name)
    lib = _load()
    src = np.ascontiguousarray(f32, dtype=np.float32)
    if src.ndim < 1:
        raise GgmlQuantError(f"{name}: need at least 1 dim, got a scalar")
    n_per_row = src.shape[-1]
    nrows = src.size // n_per_row if n_per_row else 0
    rb = row_bytes(name, n_per_row)
    if lib.ggml_quantize_requires_imatrix(int(qt)) and imatrix is None:
        raise GgmlQuantError(f"{name} needs an importance matrix (quant.imatrix)")
    im = None
    if imatrix is not None:
        im = np.ascontiguousarray(imatrix, dtype=np.float32)
        if im.shape != (n_per_row,):
            raise GgmlQuantError(f"imatrix shape {im.shape} != ({n_per_row},)")
    dst = np.empty((*src.shape[:-1], rb), dtype=np.uint8)
    with _lib_lock:
        if int(qt) not in _inited:  # init is not safe to race; do it once up front
            lib.ggml_quantize_init(int(qt))
            _inited.add(int(qt))
    sp = src.ctypes.data
    dp = dst.ctypes.data
    ip = im.ctypes.data if im is not None else None

    def run(r0: int) -> None:
        n = min(chunk_rows, nrows - r0)
        got = lib.ggml_quantize_chunk(int(qt), sp, dp, r0 * n_per_row, n, n_per_row, ip)
        if got != n * rb:
            raise GgmlQuantError(f"{name}: rows {r0}..{r0 + n} wrote {got} bytes, expected {n * rb}")

    starts = range(0, nrows, chunk_rows)
    workers = workers or os.cpu_count() or 1
    if workers == 1 or nrows <= chunk_rows:
        for r0 in starts:
            run(r0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, starts))
    return dst
