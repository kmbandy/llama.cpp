# test_ml8_backward_kernels.py
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 _scaled_mm needs GPU")
@pytest.mark.parametrize("M,N,K", [(64, 2560, 9216), (48, 4096, 2560),
                                   (16, 2560, 2560), (33, 1024, 512)])
def test_backward_gemms_run_fp8_and_match_fp32(M, N, K):
    """The two backward GEMMs (dx, dW_raw) must run on fp8 tensor cores
    (_scaled_mm), not the fp32 `@` placeholder in Ml8Fp8Fn.backward
    (fp8_qat.py:105-106) — the MAD-290 ~3.4s bottleneck. Result must track the
    fp32 reference within fp8 precision (relative Frobenius error)."""
    from ml8_backward_kernels import ml8_backward_gemms
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    dy = torch.randn(M, N, generator=g, device=dev)
    W = torch.randn(N, K, generator=g, device=dev) * 0.05
    x = torch.randn(M, K, generator=g, device=dev) * 0.1

    dx_ref = dy @ W              # [M,K]
    dW_ref = dy.t() @ x          # [N,K]

    dx, dW_raw = ml8_backward_gemms(dy, W, x)
    assert dx.shape == (M, K) and dW_raw.shape == (N, K)

    def relerr(a, b):
        return (a.float() - b).norm() / b.norm().clamp_min(1e-12)

    assert relerr(dx, dx_ref) < 0.10, f"dx relerr {relerr(dx, dx_ref):.3f}"
    assert relerr(dW_raw, dW_ref) < 0.10, f"dW_raw relerr {relerr(dW_raw, dW_ref):.3f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 _scaled_mm needs GPU")
def test_backward_gemms_handle_noncontiguous_inputs():
    """In the live autograd graph dy is a reshaped gradient and W is a gathered
    (strided) reconstruction — both non-contiguous. _scaled_mm requires strict
    row-major x col-major operands, so the helper must normalize layout itself.
    Regression for the cuBLASLt 'row-major and column-major' crash."""
    from ml8_backward_kernels import ml8_backward_gemms
    dev = "cuda"
    M, N, K = 48, 2560, 9216
    g = torch.Generator(device=dev).manual_seed(1)
    # non-contiguous dy (transpose view) and W (transpose view), as in the real path
    dy = (torch.randn(N, M, generator=g, device=dev)).t()        # [M,N] non-contig
    W = (torch.randn(K, N, generator=g, device=dev) * 0.05).t()  # [N,K] non-contig
    x = (torch.randn(K, M, generator=g, device=dev) * 0.1).t()   # [M,K] non-contig
    assert not dy.is_contiguous() and not W.is_contiguous() and not x.is_contiguous()

    dx, dW_raw = ml8_backward_gemms(dy, W, x)        # must not raise
    assert dx.shape == (M, K) and dW_raw.shape == (N, K)
    relerr = lambda a, b: (a.float() - b).norm() / b.norm().clamp_min(1e-12)
    assert relerr(dx, dy @ W) < 0.10
    assert relerr(dW_raw, dy.t() @ x) < 0.10


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
@pytest.mark.parametrize("N,K,G", [(64, 256, 4), (128, 1024, 8), (256, 1216, 76)])
def test_wgrad_triton_matches_torch(N, K, G):
    from ml8_backward_kernels import ml8_wgrad_torch, ml8_wgrad_triton
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(N, K, G, dev, seed=N)
    dc_t, ds_t = ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)
    dc_k, ds_k = ml8_wgrad_triton(dW_raw, indices, cent, scales, gsz)
    assert torch.allclose(dc_k, dc_t, atol=2e-2, rtol=2e-2), (dc_k - dc_t).abs().max()
    assert torch.allclose(ds_k, ds_t, atol=1e-2, rtol=1e-2), (ds_k - ds_t).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_wgrad_triton_odd_N_masking():
    """N not a multiple of BLOCK_N must still be exact (masked tail rows)."""
    from ml8_backward_kernels import ml8_wgrad_torch, ml8_wgrad_triton
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(70, 512, 4, dev, seed=7)
    dc_t, ds_t = ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)
    dc_k, ds_k = ml8_wgrad_triton(dW_raw, indices, cent, scales, gsz)
    assert torch.allclose(dc_k, dc_t, atol=2e-2, rtol=2e-2)
    assert torch.allclose(ds_k, ds_t, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_dispatch_env_forces_backend(monkeypatch):
    import ml8_backward_kernels as M
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(64, 256, 4, dev)
    dc_ref, ds_ref = M.ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)
    for backend in ("torch", "triton"):
        monkeypatch.setenv("ML8_WGRAD_BACKEND", backend)
        M._BACKEND_CACHE = None  # reset memoized choice
        dc, ds = M.ml8_wgrad(dW_raw, indices, cent, scales, gsz)
        assert torch.allclose(dc, dc_ref, atol=2e-2, rtol=2e-2)
        assert torch.allclose(ds, ds_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_dispatch_auto_picks_a_valid_backend(monkeypatch):
    import ml8_backward_kernels as M
    monkeypatch.delenv("ML8_WGRAD_BACKEND", raising=False)
    M._BACKEND_CACHE = None
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(64, 256, 4, dev)
    dc_ref, ds_ref = M.ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)
    dc, ds = M.ml8_wgrad(dW_raw, indices, cent, scales, gsz)
    assert M._BACKEND_CACHE in ("torch", "triton")
    assert torch.allclose(dc, dc_ref, atol=2e-2, rtol=2e-2)
    assert torch.allclose(ds, ds_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_dispatch_prefers_triton_on_real_shape(monkeypatch):
    """The probe must not let a per-shape timing race lock the multi-shape
    training loop to the slow path: triton wins on every real ml8 shape, so the
    auto choice should be triton whenever the kernel runs without error."""
    import ml8_backward_kernels as M
    monkeypatch.delenv("ML8_WGRAD_BACKEND", raising=False)
    M._BACKEND_CACHE = None
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(64, 256, 4, dev)
    M.ml8_wgrad(dW_raw, indices, cent, scales, gsz)
    assert M._BACKEND_CACHE == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_dispatch_falls_back_and_warns_when_triton_raises(monkeypatch):
    """A kernel failure must fall back to torch AND surface a warning, never
    silently degrade to the slow path with no signal."""
    import ml8_backward_kernels as M
    monkeypatch.delenv("ML8_WGRAD_BACKEND", raising=False)
    M._BACKEND_CACHE = None
    dev = "cuda"
    dW_raw, indices, gidx, cent, scales, gsz = _mk_case(64, 256, 4, dev)
    dc_ref, ds_ref = M.ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz)

    def _boom(*a, **k):
        raise RuntimeError("simulated kernel failure")
    monkeypatch.setattr(M, "ml8_wgrad_triton", _boom)
    with pytest.warns(RuntimeWarning):
        dc, ds = M.ml8_wgrad(dW_raw, indices, cent, scales, gsz)
    assert M._BACKEND_CACHE == "torch"
    assert torch.allclose(dc, dc_ref, atol=2e-2, rtol=2e-2)
    assert torch.allclose(ds, ds_ref, atol=1e-2, rtol=1e-2)
