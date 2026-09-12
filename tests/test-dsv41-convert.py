#!/usr/bin/env python3
"""CPU-only checks for the DeepSeek-V4.1 convert/arch slice. No GPU, no HF weights."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
import numpy as np  # noqa: E402


def test_arch_registration() -> None:
    assert gguf.MODEL_ARCH.DEEPSEEK41 is not None
    assert gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DEEPSEEK41] == "deepseek41"
    tensors = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DEEPSEEK41]
    for t in (
        gguf.MODEL_TENSOR.ENGRAM_EMBD,
        gguf.MODEL_TENSOR.ENGRAM_K,
        gguf.MODEL_TENSOR.ENGRAM_Q,
        gguf.MODEL_TENSOR.ENGRAM_WKV,
        gguf.MODEL_TENSOR.INDEXER_K_NORM,
        gguf.MODEL_TENSOR.INDEXER_ATTN_K,
    ):
        assert t in tensors, t
    # V4.1 has no output_hc_* (Single-Pass mHC)
    for t in (
        gguf.MODEL_TENSOR.HC_HEAD_FN,
        gguf.MODEL_TENSOR.HC_HEAD_BASE,
        gguf.MODEL_TENSOR.HC_HEAD_SCALE,
        gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE,
    ):
        assert t not in tensors, t


def test_mxfp4_parallel_matches_gguf() -> None:
    sys.path.insert(0, str(ROOT))
    from conversion.dsv41_engram_from_hf import mxfp4_quantize_parallel

    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 256), dtype=np.float32)
    qtype = gguf.GGMLQuantizationType.MXFP4
    ref = gguf.quantize(x, qtype).reshape(64, 136)
    serial = mxfp4_quantize_parallel(x, workers=1)
    assert serial.shape == (64, 136)
    assert np.array_equal(serial, ref)
    # 8192-row cutoff uses the process pool
    y = rng.standard_normal((9000, 256), dtype=np.float32)
    ref_y = gguf.quantize(y, qtype).reshape(9000, 136)
    pooled = mxfp4_quantize_parallel(y, workers=4)
    assert np.array_equal(pooled, ref_y)


def test_engram_mxfp4_row_bytes() -> None:
    qtype = gguf.GGMLQuantizationType.MXFP4
    block_elems, type_size = gguf.GGML_QUANT_SIZES[qtype]
    assert block_elems == 32
    assert type_size == 17
    n_cols = 256
    row = np.linspace(-1.0, 1.0, n_cols, dtype=np.float32).reshape(1, n_cols)
    packed = gguf.quantize(row, qtype)
    assert packed.shape[0] == 1
    assert packed.nbytes == (n_cols // block_elems) * type_size
    assert packed.nbytes == 136


def test_keys() -> None:
    arch = "deepseek41"
    assert gguf.Keys.LLM.ENGRAM_LAYER_IDS.format(arch=arch) == "deepseek41.engram.layer_ids"
    assert gguf.Keys.Attention.KV_SOURCE_LAYER_IDS.format(arch=arch) == "deepseek41.attention.kv_source_layer_ids"


def test_v41_class_keeps_own_arch() -> None:
    sys.path.insert(0, str(ROOT))
    from conversion.deepseek import DeepseekV4Model, DeepseekV41Model

    assert DeepseekV41Model.model_arch == gguf.MODEL_ARCH.DEEPSEEK41
    assert DeepseekV4Model.model_arch == gguf.MODEL_ARCH.DEEPSEEK4
    # Parent MTP integration must not be overwritten: the method exists on V4
    # and V4.1 must not assign block_count in __init__ after super().
    src = Path(__file__).resolve().parents[1] / "conversion" / "deepseek.py"
    text = src.read_text(encoding="utf-8")
    v41 = text.split("class DeepseekV41Model")[1].split("class DeepseekV4DSparkModel")[0]
    assert "self.block_count =" not in v41


if __name__ == "__main__":
    test_arch_registration()
    test_engram_mxfp4_row_bytes()
    test_mxfp4_parallel_matches_gguf()
    test_keys()
    test_v41_class_keeps_own_arch()
    print("test-dsv41-convert: ok")
