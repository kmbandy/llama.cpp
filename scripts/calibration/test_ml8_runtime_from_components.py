import torch
import ml8_runtime

def _unpack(indices_packed, N, K, lo_first=True):
    # inverse of the lo-first nibble packing: [K//2, N] uint8 -> [N, K] uint8
    p = indices_packed.t().cpu().numpy()              # [N, K//2]
    out = torch.zeros(N, K, dtype=torch.uint8)
    out[:, 0::2] = torch.from_numpy((p & 0x0F).copy())
    out[:, 1::2] = torch.from_numpy(((p >> 4) & 0x0F).copy())
    return out

def test_layer_from_components_roundtrips():
    N, K, G, NC = 4, 8, 2, 16
    gsize = K // G
    centroids = torch.randn(G, NC).to(torch.float8_e4m3fn).float()   # e4m3 lattice vals
    scales = torch.rand(N, G) + 0.5
    indices = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    gidx = torch.arange(K) // gsize                                  # uniform grouping
    layer = ml8_runtime.layer_from_components(
        centroids=centroids, scales=scales, indices=indices, gidx=gidx)
    assert layer.n_rows == N and layer.n_cols == K
    assert layer.group_size == gsize and layer.n_centroids == NC
    assert layer.indices_packed.shape == (K // 2, N)
    assert layer.centroids_fp8.shape == (G, NC) and layer.centroids_fp8.dtype == torch.float8_e4m3fn
    assert layer.scales_fp32.shape == (G, N)
    # packed indices unpack back to originals
    assert torch.equal(_unpack(layer.indices_packed, N, K), indices)
    # scales transposed correctly
    assert torch.allclose(layer.scales_fp32, scales.t())
