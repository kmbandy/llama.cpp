# test_ml8_backward_kernels.py
import pytest
import torch


def _reference_grads(dW_raw, indices, gidx, centroids, scales):
    """Oracle: byte-for-byte the two scatter blocks from Ml8Fp8Fn.backward
    (fp8_qat.py:116-124), the path this work replaces."""
    N, K = indices.shape
    G = scales.shape[1]
    # W[n,k] = centroids[gidx[k], indices[n,k]] * scales[n, gidx[k]]
    cent_per_col = centroids[gidx]                                    # [K,16]
    W = cent_per_col.unsqueeze(0).expand(N, -1, -1).gather(
        2, indices.long().unsqueeze(-1)).squeeze(-1) * scales[:, gidx]
    dW_scaled = dW_raw * scales[:, gidx]                              # [N,K]
    dcent = torch.zeros_like(centroids)                              # [G,16]
    flat_g = gidx.unsqueeze(0).expand(N, -1).reshape(-1)
    flat_i = indices.long().reshape(-1)
    dcent.index_put_((flat_g, flat_i), dW_scaled.reshape(-1), accumulate=True)
    dscales = torch.zeros_like(scales)                              # [N,G]
    contrib = (dW_raw * W / scales[:, gidx].clamp_min(1e-12))
    dscales.index_add_(1, gidx, contrib)
    return dcent, dscales


def _mk_case(N, K, G, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    gsz = K // G
    dW_raw = torch.randn(N, K, generator=g, device=device)
    indices = torch.randint(0, 16, (N, K), generator=g, dtype=torch.uint8, device=device)
    centroids = torch.randn(G, 16, generator=g, device=device) * 0.1
    scales = torch.rand(N, G, generator=g, device=device) * 0.05 + 0.01
    gidx = (torch.arange(K, device=device) // gsz).long()
    return dW_raw, indices, gidx, centroids, scales, gsz


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_wgrad_torch_matches_reference():
    from ml8_backward_kernels import ml8_wgrad_torch
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(64, 256, 4, dev)
    dcent_ref, dscales_ref = _reference_grads(dW_raw, indices, gidx, cent, scales)
    dcent, dscales = ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)
    assert torch.allclose(dcent, dcent_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(dscales, dscales_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_wgrad_torch_dscales_reshape_is_exact():
    """The contiguous-group reshape must equal index_add_ tightly (design claim)."""
    from ml8_backward_kernels import ml8_wgrad_torch
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(128, 1024, 8, dev, seed=3)
    _, dscales_ref = _reference_grads(dW_raw, indices, gidx, cent, scales)
    _, dscales = ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)
    assert (dscales - dscales_ref).abs().max().item() < 1e-3
