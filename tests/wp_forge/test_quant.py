"""CPU-only checks for conversion/wp_forge/quant.py.

Conftest already puts the in-tree gguf-py on sys.path (tests/wp_forge/conftest.py).
Matrices are tiny (8 x 512) so the pool round-trips stay fast; 512 is a multiple
of 256 so K-quants would work too.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import gguf
from gguf.constants import GGML_QUANT_SIZES

from conversion.wp_forge.quant import QuantError, dequant, lossless_repack, quantize_expert

ROWS, COLS = 8, 512
SEED = 20260915


def _f32() -> np.ndarray:
    return np.random.RandomState(SEED).randn(ROWS, COLS).astype(np.float32)


@pytest.mark.parametrize("qtype", ["q8_0", "mxfp4"])
@pytest.mark.parametrize("workers", [1, 2])
def test_quantize_expert_pool_matches_single_call(qtype: str, workers: int) -> None:
    f32 = _f32()
    expected = gguf.quantize(f32, gguf.GGMLQuantizationType[qtype.upper()])
    got = quantize_expert({"gate": f32, "up": f32.copy(), "down": f32.copy()}, qtype, workers=workers)
    assert set(got) == {"gate", "up", "down"}
    for role, out in got.items():
        assert out.dtype == np.uint8
        assert out.tobytes() == expected.tobytes()


def test_quantize_expert_row_size_mismatch_raises(monkeypatch) -> None:
    import conversion.wp_forge.quant as quant_mod

    f32 = _f32()

    def wrong_width(arr: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
        # 255 bytes per row instead of 16 * 34 = 544.
        return np.zeros((arr.shape[0], 255), dtype=np.uint8)

    monkeypatch.setattr(quant_mod.gguf, "quantize", wrong_width)
    with pytest.raises(QuantError, match="gate.*544.*q8_0"):
        quantize_expert({"gate": f32}, "q8_0", workers=1)


def _hf_pack_from_ggml_mxfp4(ggml_packed: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert ModelBase.repack_mxfp4_blocks.

    ggml block (17 bytes per 32 elems): 1 scale byte + 16 code bytes, code byte j
    holds elem j in the low nibble and elem j+16 in the high one.
    HF layout: uint8 [rows, cols/2], elem 2i in the low nibble, 2i+1 in the high.
    """
    rows, width = ggml_packed.shape
    n_blocks = width // 17
    src = ggml_packed.reshape(rows, n_blocks, 17)
    s = torch.from_numpy(src[:, :, 0].copy())
    qs = torch.from_numpy(src[:, :, 1:17].copy())
    # ggml code byte j: e_j low nibble, e_{j+16} high nibble.
    e = torch.cat((qs & 0x0F, qs >> 4), dim=-1).reshape(rows, n_blocks, 32)
    # HF byte k: e_{2k} low nibble, e_{2k+1} high nibble.
    packed = (e[:, :, 0::2] | (e[:, :, 1::2] << 4)).reshape(rows, n_blocks * 16)
    return packed.to(torch.uint8), s


def test_lossless_repack_mxfp4_roundtrip() -> None:
    # Build a synthetic HF mxfp4 pack by quantizing f32 to MXFP4 with gguf-py and
    # inverting the ggml block layout, then check the lossless path matches the
    # gguf-py bytes exactly.
    f32 = _f32()
    ggml_packed = gguf.quantize(f32, gguf.GGMLQuantizationType.MXFP4)
    hf_packed, hf_scale = _hf_pack_from_ggml_mxfp4(ggml_packed)
    assert hf_packed.shape == (ROWS, COLS // 2)
    assert hf_scale.shape == (ROWS, COLS // 32)

    repacked = lossless_repack(hf_packed, hf_scale, "mxfp4", "mxfp4")
    assert repacked is not None
    assert repacked.shape == ggml_packed.shape
    assert repacked.tobytes() == ggml_packed.tobytes()
    # Row byte size equals the gguf-py derivation for MXFP4 at this shape.
    block_size, type_size = GGML_QUANT_SIZES[gguf.GGMLQuantizationType.MXFP4]
    assert repacked.shape[1] == COLS // block_size * type_size


def test_lossless_repack_non_matching_format_returns_none() -> None:
    f32 = _f32()
    ggml_packed = gguf.quantize(f32, gguf.GGMLQuantizationType.MXFP4)
    hf_packed, hf_scale = _hf_pack_from_ggml_mxfp4(ggml_packed)
    assert lossless_repack(hf_packed, hf_scale, "fp8_block32", "mxfp4") is None
    assert lossless_repack(hf_packed, hf_scale, "mxfp4", "q8_0") is None


def test_dequant_fp8_block32() -> None:
    # Mirrors the DS4.1 converter call path: e4m3/e8m0 tensors are converted to
    # f32 first (f8e4m3_to_f32 / e8m0_to_f32) before dequant_fp8_block32.
    from conversion.dsv41_engram_from_hf import e8m0_to_f32, f8e4m3_to_f32

    wr, wc, sr, sc = 32, 32, 1, 1
    # Small e4m3 exponents: bits 0..15 are all finite (1.0..16.0).
    wraw = bytes(torch.randint(0, 16, (wr * wc,), dtype=torch.uint8).numpy())
    sraw = bytes(torch.randint(120, 134, (sr * sc,), dtype=torch.uint8).numpy())
    out = dequant(f8e4m3_to_f32(wraw, wr, wc), e8m0_to_f32(sraw, sr, sc), "fp8_block32")
    assert out.shape == (wr, wc)
    assert out.dtype == np.float32
    assert (np.abs(out) > 0).any()
    with pytest.raises(ValueError, match="source_format"):
        dequant(f8e4m3_to_f32(wraw, wr, wc), e8m0_to_f32(sraw, sr, sc), "bogus")
