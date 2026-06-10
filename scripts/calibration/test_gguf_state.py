import numpy as np, torch
import pytest
from ml8_to_gguf import pack_ml8_blocks, pack_scaled_fp8_blocks, cast_centroids_to_fp8
from gguf_state import unpack_ml8_blocks, unpack_scaled_fp8_blocks, decode_centroids_fp8

def test_ml8_roundtrip():
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 16, (8, 128), generator=g, dtype=torch.int8)
    scl = torch.rand(8, 2, generator=g) + 0.1
    packed = pack_ml8_blocks(idx, scl)
    idx2, scl2 = unpack_ml8_blocks(packed, N=8, K=128)
    assert idx2.dtype == torch.uint8
    assert torch.equal(idx2.long(), idx.to(torch.long))
    assert torch.equal(scl2, scl)

def test_fp8_roundtrip():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(4, 64, generator=g)
    e4m3 = w.to(torch.float8_e4m3fn).to(torch.float32)
    scale = (torch.rand(4, 2, generator=g) + 0.5).to(torch.float16)
    packed = pack_scaled_fp8_blocks(e4m3, scale)
    e2, s2 = unpack_scaled_fp8_blocks(packed, N=4, K=64)
    assert torch.equal(e2, e4m3) and torch.equal(s2, scale)

def test_centroid_roundtrip():
    g = torch.Generator().manual_seed(2)
    c = torch.randn(2, 16, generator=g)
    on_lattice = c.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(decode_centroids_fp8(cast_centroids_to_fp8(c)), on_lattice)

def test_unpack_rejects_bad_K():
    with pytest.raises(ValueError):
        unpack_ml8_blocks(np.zeros((2, 36), np.uint8), N=2, K=63)
    with pytest.raises(ValueError):
        unpack_scaled_fp8_blocks(np.zeros((2, 34), np.uint8), N=2, K=33)

def test_rehydrate_synthetic(tmp_path):
    import gguf
    from gguf import GGMLQuantizationType
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 16, (8, 128), generator=g, dtype=torch.int8)
    scl = torch.rand(8, 2, generator=g) + 0.1
    cent = torch.randn(2, 16, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    p = tmp_path / "mini.gguf"
    w = gguf.GGUFWriter(str(p), arch="qwen35")
    w.add_key_value("qwen35.block_count", 1, gguf.GGUFValueType.UINT32)
    w.add_tensor("blk.0.ffn_up.weight", pack_ml8_blocks(idx, scl), raw_dtype=GGMLQuantizationType.ML8_4)
    w.add_tensor("blk.0.ffn_up.centroids", cast_centroids_to_fp8(cent), raw_dtype=GGMLQuantizationType.F8_E4M3)
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(128, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    from gguf_state import load_ml8_gguf, dequant_ml8_state
    st = load_ml8_gguf(p)
    t = st.ml8["blk.0.ffn_up.weight"]
    assert t["indices"].dtype == torch.uint8
    assert torch.equal(t["indices"].long(), idx.long()) and torch.equal(t["scales"], scl) and torch.equal(t["centroids"], cent)
    gidx = torch.arange(128) // 64
    W = cent[gidx, t["indices"].long()] * scl[:, gidx]
    assert torch.equal(dequant_ml8_state(t), W)
    assert "blk.0.ffn_norm.weight" in st.frozen


def _write_mini_gguf_with_fp8(tmp_path):
    """A mini GGUF with one ML8_4 + sidecar, one ML8_FP8 tensor, one passthrough."""
    import gguf
    from gguf import GGMLQuantizationType
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8, pack_scaled_fp8_blocks
    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, 16, (8, 128), generator=g, dtype=torch.int8)
    scl = torch.rand(8, 2, generator=g) + 0.1
    cent = torch.randn(2, 16, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    # one ML8_FP8 tensor: N=4, K=64 -> n_b=2 groups of 32
    e4m3 = torch.randn(4, 64, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    fp8_scale = (torch.rand(4, 2, generator=g) + 0.5).to(torch.float16)
    p = tmp_path / "mini_fp8.gguf"
    w = gguf.GGUFWriter(str(p), arch="qwen35")
    w.add_key_value("qwen35.block_count", 1, gguf.GGUFValueType.UINT32)
    w.add_tensor("blk.0.ffn_up.weight", pack_ml8_blocks(idx, scl), raw_dtype=GGMLQuantizationType.ML8_4)
    w.add_tensor("blk.0.ffn_up.centroids", cast_centroids_to_fp8(cent), raw_dtype=GGMLQuantizationType.F8_E4M3)
    w.add_tensor("blk.0.ssm_alpha.weight", pack_scaled_fp8_blocks(e4m3, fp8_scale), raw_dtype=GGMLQuantizationType.ML8_FP8)
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(128, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return p, e4m3, fp8_scale


def test_frozen_mode_fp8_only_bf16(tmp_path):
    """frozen_mode='fp8' keeps ONLY ML8_FP8 tensors in frozen, stored as bf16,
    skipping all passthrough tensors (halves RAM; host has 15GB)."""
    from gguf_state import load_ml8_gguf
    p, e4m3, fp8_scale = _write_mini_gguf_with_fp8(tmp_path)
    st = load_ml8_gguf(p, frozen_mode="fp8")
    # only the fp8 tensor is frozen; the passthrough ffn_norm is skipped.
    assert set(st.frozen.keys()) == {"blk.0.ssm_alpha.weight"}
    fz = st.frozen["blk.0.ssm_alpha.weight"]
    assert fz.dtype == torch.bfloat16
    # value matches the dequantized e4m3 * expanded scale (within bf16 precision).
    scale_cols = fp8_scale.to(torch.float32).repeat_interleave(32, dim=1)
    expected = (e4m3 * scale_cols).to(torch.bfloat16)
    assert torch.equal(fz, expected)
    # the ml8 tensor is still rehydrated.
    assert "blk.0.ffn_up.weight" in st.ml8


def test_frozen_mode_none_empty(tmp_path):
    from gguf_state import load_ml8_gguf
    p, _e4m3, _scale = _write_mini_gguf_with_fp8(tmp_path)
    st = load_ml8_gguf(p, frozen_mode="none")
    assert st.frozen == {}
    assert "blk.0.ffn_up.weight" in st.ml8


def test_frozen_mode_all_default(tmp_path):
    """frozen_mode='all' (default) keeps fp8 + passthrough, as before."""
    from gguf_state import load_ml8_gguf
    p, _e4m3, _scale = _write_mini_gguf_with_fp8(tmp_path)
    st = load_ml8_gguf(p)  # default 'all'
    assert "blk.0.ssm_alpha.weight" in st.frozen
    assert "blk.0.ffn_norm.weight" in st.frozen


def test_rehydrated_indices_are_uint8(tmp_path):
    """Regression: rehydrated ml8 indices stay uint8 (not int64). Stored uint8
    is 8x smaller than long over all ml8 tensors (~21GB -> 2.6GB rehydrate RAM)."""
    from gguf_state import load_ml8_gguf
    p, _e4m3, _scale = _write_mini_gguf_with_fp8(tmp_path)
    st = load_ml8_gguf(p, frozen_mode="none")
    for name, entry in st.ml8.items():
        assert entry["indices"].dtype == torch.uint8, name
