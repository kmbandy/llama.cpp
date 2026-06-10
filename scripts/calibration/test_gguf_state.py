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
    assert torch.equal(idx2, idx.to(torch.long))
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
