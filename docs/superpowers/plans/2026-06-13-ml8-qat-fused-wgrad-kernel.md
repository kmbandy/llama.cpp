# Fused ml8-QAT Backward Wgrad Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two atomic-scatter blocks in `Ml8Fp8Fn.backward` with one fused Triton kernel that computes `dcent` and `dscales` directly from `dW_raw` + `indices`, eliminating the scatter contention that made the backward ~41 ms/layer.

**Architecture:** A single Triton kernel, grid `(G, n_tiles)`, where program `(g, nt)` owns one contiguous K-group's `[BLOCK_N, gsz]` slab. It writes `dscales[n,g]` directly (each output owned by one program, no atomics) and atomic-adds a 16-bin centroid histogram into `dcent[g,:]` (contention only across N-tiles). A pure-torch path (`dscales` reshape + `index_put_`) serves as both the test oracle's sibling and the runtime fallback, selected by a one-time perf probe.

**Tech Stack:** Python, PyTorch (custom-built ROCm editable env at `/home/kmbandy/venvs/agents/bin/python`), Triton (gfx1201 / RDNA4), pytest.

**Conventions (read before starting):**
- All commands run from `/home/kmbandy/GitHub/llama.cpp/scripts/calibration/`.
- The Python interpreter is `/home/kmbandy/venvs/agents/bin/python` (has ROCm torch + Triton). **NEVER `pip install` into it** — it is an editable PyTorch build; clobbering costs a multi-hour rebuild.
- Run tests as: `/home/kmbandy/venvs/agents/bin/python -m pytest <file>::<test> -v`
- GPU tests are gated with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")`. The R9700 is `cuda:0`; the 6900xt (`cuda:1`) runs the fleet server — **use `cuda:0` / `"cuda"` only** and keep VRAM small.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Targeted `git add <files>` only — never `-A`/`-am`.
- Current branch: `sync/upstream-2026-06-09`.

**Group-structure invariant (the whole basis of this work):** ml8 groups are contiguous K-blocks: `gidx == torch.arange(K)//gsz`, where `gsz = K // G` and `G = scales.shape[1]`. Every task relies on this.

---

### Task 1: Pure-torch reference path `ml8_wgrad_torch` (the fallback + oracle sibling)

This is the exact, contiguous-reshape `dscales` + `index_put_` `dcent` path. It is the runtime fallback and proves the reshape equivalence the whole design rests on.

**Files:**
- Create: `scripts/calibration/ml8_backward_kernels.py`
- Test: `scripts/calibration/test_ml8_backward_kernels.py`

- [ ] **Step 1: Write the failing test (oracle = the current backward's two scatter blocks)**

Create `scripts/calibration/test_ml8_backward_kernels.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py -v`
Expected: FAIL — `ImportError: cannot import name 'ml8_wgrad_torch' from 'ml8_backward_kernels'` (module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/calibration/ml8_backward_kernels.py`:

```python
"""Fused ml8-QAT backward weight-gradient kernels.

Computes dcent[G,16] and dscales[N,G] from dW_raw[N,K] + indices[N,K], for the
fp8 QAT trainer's Ml8Fp8Fn.backward. Training-only (never ships in a GGUF).

ml8 groups are CONTIGUOUS K-blocks (gidx == arange(K)//gsz), which is what makes
dscales a plain reshape-sum instead of a scatter. See
docs/superpowers/specs/2026-06-13-ml8-qat-fused-wgrad-kernel-design.md.
"""
import torch


def ml8_wgrad_torch(dW_raw, indices, centroids, scales, gsz):
    """Pure-torch reference/fallback. Exact dscales via contiguous reshape;
    dcent via index_put_ (the best pure-torch option per the bench).

    Args:
        dW_raw    [N,K] fp32  -- (dy8*sdy).T @ x
        indices   [N,K] uint8 -- centroid index per (row, col)
        centroids [G,16] fp32 -- master fp32 centroids (cent_e4m3 in caller)
        scales    [N,G] fp32
        gsz       int          -- group size (K must be divisible; G = K//gsz)
    Returns:
        (dcent [G,16], dscales [N,G]) fp32
    """
    N, K = indices.shape
    G = scales.shape[1]
    assert K == G * gsz, f"K={K} != G*gsz={G*gsz}"
    idx = indices.long()
    scales_exp = scales.repeat_interleave(gsz, dim=1)                # [N,K], col->group
    # dscales[n,g] = sum_{k in g} dW_raw[n,k] * centroids[g, idx[n,k]]
    centval = torch.gather(
        centroids.repeat_interleave(gsz, dim=0).unsqueeze(0).expand(N, -1, -1),
        2, idx.unsqueeze(-1)).squeeze(-1)                            # [N,K] = cent[g,idx]
    dscales = (dW_raw * centval).view(N, G, gsz).sum(2)              # contiguous reshape
    # dcent[g,c] = sum_{n, k in g, idx=c} dW_raw[n,k] * scales[n,g]
    dW_scaled = dW_raw * scales_exp                                  # [N,K]
    gidx = (torch.arange(K, device=dW_raw.device) // gsz).long()
    flat_g = gidx.unsqueeze(0).expand(N, -1).reshape(-1)
    flat_i = idx.reshape(-1)
    dcent = torch.zeros_like(centroids)
    dcent.index_put_((flat_g, flat_i), dW_scaled.reshape(-1), accumulate=True)
    return dcent, dscales
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_backward_kernels.py scripts/calibration/test_ml8_backward_kernels.py
git commit -m "feat(mad-281): ml8_wgrad_torch reference/fallback wgrad path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Fused Triton kernel `_ml8_wgrad_kernel` + `ml8_wgrad_triton` wrapper

**Files:**
- Modify: `scripts/calibration/ml8_backward_kernels.py`
- Test: `scripts/calibration/test_ml8_backward_kernels.py`

- [ ] **Step 1: Write the failing test (Triton vs the torch path, incl. odd-N masking)**

Append to `scripts/calibration/test_ml8_backward_kernels.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py -k triton -v`
Expected: FAIL — `ImportError: cannot import name 'ml8_wgrad_triton'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `scripts/calibration/ml8_backward_kernels.py`:

```python
import triton
import triton.language as tl
```

Append to `scripts/calibration/ml8_backward_kernels.py`:

```python
@triton.jit
def _ml8_wgrad_kernel(
    dW_ptr, idx_ptr, cent_ptr, scales_ptr,      # inputs
    dcent_ptr, dscales_ptr,                      # outputs
    N, K,
    stride_dw_n, stride_dw_k,
    stride_idx_n, stride_idx_k,
    stride_cent_g, stride_cent_c,
    stride_s_n, stride_s_g,
    stride_dc_g, stride_dc_c,
    stride_ds_n, stride_ds_g,
    GSZ: tl.constexpr,
    N_CENT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Grid (G, cdiv(N, BLOCK_N)). Program (g, nt) owns rows [nt*BLOCK_N:...]
    and the contiguous K-slab [g*GSZ:(g+1)*GSZ]. Emits dscales[rows,g] (no
    atomics) and atomic-adds the 16-bin dcent[g,:] histogram."""
    g = tl.program_id(0)
    nt = tl.program_id(1)

    offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_k = g * GSZ + tl.arange(0, GSZ)                          # cols of this group

    dw = tl.load(
        dW_ptr + offs_n[:, None] * stride_dw_n + offs_k[None, :] * stride_dw_k,
        mask=mask_n[:, None], other=0.0).to(tl.float32)           # [BLOCK_N, GSZ]
    idx = tl.load(
        idx_ptr + offs_n[:, None] * stride_idx_n + offs_k[None, :] * stride_idx_k,
        mask=mask_n[:, None], other=0).to(tl.int32)               # [BLOCK_N, GSZ]
    scal = tl.load(scales_ptr + offs_n * stride_s_n + g * stride_s_g,
                   mask=mask_n, other=0.0).to(tl.float32)         # [BLOCK_N]

    dw_scaled = dw * scal[:, None]                                # [BLOCK_N, GSZ]
    centval = tl.zeros((BLOCK_N, GSZ), dtype=tl.float32)
    for c in tl.static_range(N_CENT):
        cent_c = tl.load(cent_ptr + g * stride_cent_g + c * stride_cent_c)  # scalar
        is_c = idx == c
        centval = tl.where(is_c, cent_c, centval)
        bin_sum = tl.sum(tl.where(is_c, dw_scaled, 0.0))         # scalar over tile
        tl.atomic_add(dcent_ptr + g * stride_dc_g + c * stride_dc_c, bin_sum)

    dscales_row = tl.sum(dw * centval, axis=1)                    # [BLOCK_N]
    tl.store(dscales_ptr + offs_n * stride_ds_n + g * stride_ds_g,
             dscales_row, mask=mask_n)


def ml8_wgrad_triton(dW_raw, indices, centroids, scales, gsz, block_n=64):
    """Fused Triton wgrad: returns (dcent [G,16], dscales [N,G]) fp32."""
    N, K = indices.shape
    G = scales.shape[1]
    assert K == G * gsz, f"K={K} != G*gsz={G*gsz}"
    N_CENT = centroids.shape[1]
    dW_raw = dW_raw.contiguous()
    indices = indices.contiguous()
    centroids = centroids.contiguous()
    scales = scales.contiguous()
    dcent = torch.zeros_like(centroids)                          # atomic_add target
    dscales = torch.empty_like(scales)
    grid = (G, triton.cdiv(N, block_n))
    _ml8_wgrad_kernel[grid](
        dW_raw, indices, centroids, scales,
        dcent, dscales,
        N, K,
        dW_raw.stride(0), dW_raw.stride(1),
        indices.stride(0), indices.stride(1),
        centroids.stride(0), centroids.stride(1),
        scales.stride(0), scales.stride(1),
        dcent.stride(0), dcent.stride(1),
        dscales.stride(0), dscales.stride(1),
        GSZ=gsz, N_CENT=N_CENT, BLOCK_N=block_n,
        num_stages=1,   # gfx1201 RDNA4: num_stages>=2 triggers UAF (forward audit)
        num_warps=4,
    )
    return dcent, dscales
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py -k triton -v`
Expected: PASS (4 passed: 3 parametrized + odd-N).

If the kernel errors on RDNA4 with a compile/UAF fault, confirm `num_stages=1` is set (it is above) — do not raise it.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_backward_kernels.py scripts/calibration/test_ml8_backward_kernels.py
git commit -m "feat(mad-281): fused Triton ml8 wgrad kernel (dcent+dscales, no scatter)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `ml8_wgrad` dispatcher — auto-probe fallback + `ML8_WGRAD_BACKEND` env

**Files:**
- Modify: `scripts/calibration/ml8_backward_kernels.py`
- Test: `scripts/calibration/test_ml8_backward_kernels.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/calibration/test_ml8_backward_kernels.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py -k dispatch -v`
Expected: FAIL — `AttributeError: module 'ml8_backward_kernels' has no attribute 'ml8_wgrad'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/calibration/ml8_backward_kernels.py`:

```python
import os

_BACKEND_CACHE = None   # memoized backend choice ("torch" | "triton")


def _probe_backend(dW_raw, indices, centroids, scales, gsz):
    """One-time choice. Env override wins; else time both on the live shape and
    pick the faster, falling back to torch if the kernel errors."""
    forced = os.environ.get("ML8_WGRAD_BACKEND")
    if forced in ("torch", "triton"):
        return forced
    import time
    def _t(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize(); return time.perf_counter() - s
    try:
        t_tri = _t(lambda: ml8_wgrad_triton(dW_raw, indices, centroids, scales, gsz))
    except Exception:
        return "torch"
    t_tor = _t(lambda: ml8_wgrad_torch(dW_raw, indices, centroids, scales, gsz))
    return "triton" if t_tri < t_tor else "torch"


def ml8_wgrad(dW_raw, indices, centroids, scales, gsz):
    """Dispatch to the chosen backend (memoized after first call)."""
    global _BACKEND_CACHE
    if _BACKEND_CACHE is None:
        _BACKEND_CACHE = _probe_backend(dW_raw, indices, centroids, scales, gsz)
    if _BACKEND_CACHE == "triton":
        return ml8_wgrad_triton(dW_raw, indices, centroids, scales, gsz)
    return ml8_wgrad_torch(dW_raw, indices, centroids, scales, gsz)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py -k dispatch -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_backward_kernels.py scripts/calibration/test_ml8_backward_kernels.py
git commit -m "feat(mad-281): ml8_wgrad dispatcher with probe fallback + env override

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire `ml8_wgrad` into `Ml8Fp8Fn.backward`

Replace the two scatter blocks; keep `dy_quant`, `W`, `dx`, `dW_raw`, and the `capture_dLdW` stash byte-identical.

**Files:**
- Modify: `scripts/calibration/fp8_qat.py:116-124`
- Test: `scripts/calibration/test_fp8_qat.py`

- [ ] **Step 1: Write the failing test (full backward equality vs a frozen reference + capture path intact)**

Append to `scripts/calibration/test_fp8_qat.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_backward_grads_match_scatter_reference():
    """Ml8Fp8Fn.backward (now kernel-backed) must match the old scatter math."""
    import torch
    from fp8_qat import Ml8Fp8Fn, fp8_quant
    dev = "cuda"
    N, K, G, M = 32, 256, 4, 48
    gsz = K // G
    x8 = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    sx = torch.rand(M, 1, device=dev) * 0.01 + 0.01
    cent = torch.randn(G, 16, device=dev) * 0.1
    scales = torch.rand(N, G, device=dev) * 0.05 + 0.01
    indices = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=dev)
    gidx = (torch.arange(K, device=dev) // gsz).long()
    dy = torch.randn(M, N, device=dev)

    class Ctx:  # minimal ctx stand-in for backward
        pass
    ctx = Ctx()
    ctx.saved_tensors = (x8, sx, cent, scales, indices, gidx)
    ctx.indices_id = id(indices)
    Ml8Fp8Fn.capture_dLdW = False
    dx, dcent, dscales, _, _ = Ml8Fp8Fn.backward(ctx, dy)

    # Reference: the pre-kernel scatter math.
    from test_ml8_backward_kernels import _reference_grads
    dyf = dy
    dy8, sdy = fp8_quant(dyf, "e5m2")
    xq = x8.float() * sx
    dW_raw = (dy8.float() * sdy).t() @ xq
    dcent_ref, dscales_ref = _reference_grads(dW_raw, indices, gidx, cent, scales)
    assert torch.allclose(dcent, dcent_ref, atol=2e-2, rtol=2e-2)
    assert torch.allclose(dscales, dscales_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_backward_capture_dLdW_still_populated():
    """The pv/Axis-B side channel must still receive dW_raw + h when enabled."""
    import torch
    from fp8_qat import Ml8Fp8Fn
    dev = "cuda"
    N, K, G, M = 32, 256, 4, 48
    gsz = K // G
    x8 = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    sx = torch.rand(M, 1, device=dev) * 0.01 + 0.01
    cent = torch.randn(G, 16, device=dev) * 0.1
    scales = torch.rand(N, G, device=dev) * 0.05 + 0.01
    indices = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=dev)
    gidx = (torch.arange(K, device=dev) // gsz).long()
    dy = torch.randn(M, N, device=dev)

    class Ctx:
        pass
    ctx = Ctx()
    ctx.saved_tensors = (x8, sx, cent, scales, indices, gidx)
    ctx.indices_id = id(indices)
    Ml8Fp8Fn.last_dLdW.clear(); Ml8Fp8Fn.last_h.clear()
    Ml8Fp8Fn.capture_dLdW = True
    try:
        Ml8Fp8Fn.backward(ctx, dy)
        assert Ml8Fp8Fn.last_dLdW[id(indices)].shape == (N, K)
        assert Ml8Fp8Fn.last_h[id(indices)].shape == (K,)
    finally:
        Ml8Fp8Fn.capture_dLdW = False
        Ml8Fp8Fn.last_dLdW.clear(); Ml8Fp8Fn.last_h.clear()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_fp8_qat.py -k "backward_grads_match or capture_dLdW_still" -v`
Expected: the equality test currently PASSES (old scatter path already matches the reference, since the reference *is* the old math) — so to make it a real RED, first confirm the wiring is not yet present, then this test guards the refactor. If it passes pre-change that is acceptable: it is a refactor-safety test. The capture test should also pass pre-change. Proceed to Step 3 (the refactor) and re-run; both must stay green.

> Note: This task is a behavior-preserving refactor, so the test is a *characterization* test (green before and after). That is the correct TDD shape for a refactor — the test locks the behavior, then you swap the implementation under it.

- [ ] **Step 3: Apply the refactor**

In `scripts/calibration/fp8_qat.py`, add the import near the top (after `from centroid_quantizer import snap_to_e4m3`):

```python
from ml8_backward_kernels import ml8_wgrad
```

Then replace lines 116-124 (the `dW_scaled`/`dcent`/`dscales` block):

```python
        # chain dW_raw -> dcentroids (scatter-add over (group, index)) and -> dscales
        dW_scaled = dW_raw * scales[:, gidx]                          # dW/dcent path
        dcent = torch.zeros_like(cent_e4m3)                          # [G,16]
        flat_g = gidx.unsqueeze(0).expand(N, -1).reshape(-1)         # [N*K]
        flat_i = indices.long().reshape(-1)
        dcent.index_put_((flat_g, flat_i), dW_scaled.reshape(-1), accumulate=True)
        dscales = torch.zeros_like(scales)                          # [N,G]
        contrib = (dW_raw * W / scales[:, gidx].clamp_min(1e-12))   # d(W)/dscale = cent
        dscales.index_add_(1, gidx, contrib)                        # sum cols per group
```

with:

```python
        # Fused wgrad: dcent[G,16] + dscales[N,G] from dW_raw + indices, no dense
        # W-scatter. gsz = K // G; groups are contiguous (gidx == arange(K)//gsz).
        gsz = K // scales.shape[1]
        dcent, dscales = ml8_wgrad(dW_raw, indices, cent_e4m3, scales, gsz)
```

- [ ] **Step 4: Run to verify both tests pass (and the wider suite is green)**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_fp8_qat.py -v`
Expected: PASS — all existing fp8_qat tests plus the two new ones. The two new tests confirm the kernel-backed backward matches the old scatter math and the `capture_dLdW` path still fills.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/fp8_qat.py scripts/calibration/test_fp8_qat.py
git commit -m "feat(mad-281): route Ml8Fp8Fn.backward through fused ml8_wgrad

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Perf re-time gate + full-suite green

Prove the kernel actually wins (the design's measure-gate) and nothing regressed.

**Files:**
- Create: `scripts/calibration/bench_ml8_wgrad.py`

- [ ] **Step 1: Write the benchmark**

Create `scripts/calibration/bench_ml8_wgrad.py`:

```python
"""Re-time the wgrad path: fused kernel vs the old index_put_/index_add_ scatter,
across representative 4B ml8 shapes. Prints per-shape ms and the speedup; exits
nonzero if the kernel is not faster than the old scatter on any shape."""
import sys, time
import torch
from ml8_backward_kernels import ml8_wgrad_triton, ml8_wgrad_torch
from test_ml8_backward_kernels import _reference_grads, _mk_case

SHAPES = [("attn", 1024, 2560, 20), ("mlp_up", 1024, 9728, 20),
          ("mlp_down", 1024, 2560, 76)]  # (name, N, K, G) — N=out, K=in for wgrad


def t(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3


def main():
    assert torch.cuda.is_available()
    ok = True
    for name, N, K, G in SHAPES:
        dW_raw, indices, gidx, cent, scales, gsz = _mk_case(N, K, G, "cuda", seed=N)
        old = t(lambda: _reference_grads(dW_raw, indices, gidx, cent, scales))
        new = t(lambda: ml8_wgrad_triton(dW_raw, indices, cent, scales, gsz))
        tor = t(lambda: ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz))
        print(f"[{name:9s} N={N} K={K} G={G}] old_scatter={old:7.3f}ms "
              f"torch_fallback={tor:7.3f}ms triton={new:7.3f}ms  "
              f"speedup_vs_old={old/new:5.2f}x")
        ok = ok and (new < old)
    print("PASS" if ok else "FAIL: kernel not faster than old scatter on some shape")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `/home/kmbandy/venvs/agents/bin/python bench_ml8_wgrad.py`
Expected: three lines with `speedup_vs_old` > 1.0 and a final `PASS`. Record the numbers in the commit message.

If the kernel is *slower* than `old_scatter` on a shape (FAIL): the auto-probe in Task 3 will already pick `torch` at runtime, so correctness is safe. Report the numbers — if `torch_fallback` beats `old_scatter` (it should, via the dscales reshape) the half-win still lands; investigate the kernel's `BLOCK_N` (try 128) before concluding. Do not raise `num_stages`.

- [ ] **Step 3: Run the full calibration test suites that touch this code**

Run: `/home/kmbandy/venvs/agents/bin/python -m pytest test_ml8_backward_kernels.py test_fp8_qat.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibration/bench_ml8_wgrad.py
git commit -m "feat(mad-281): wgrad perf bench gate (kernel vs old scatter)

<paste the three speedup lines here>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Non-goals (do not build):** a dgrad LUT kernel, `dx` fusion, HIP-graph capture, or the streaming-memory model. The micro-bench showed the GEMMs are at the FLOP ceiling and the backward is GPU-compute-bound on scatter, not launch-bound. Streaming memory is a separate follow-up spec.
- **If Triton on gfx1201 misbehaves:** keep `num_stages=1`; try `BLOCK_N` 128 vs 64; never enable `expandable_segments` (faults this hardware). The torch fallback guarantees correctness regardless.
- After all tasks: the 4B Axis-B verdict (frozen vs gptq vs gptq-interleave) can be re-run on the faster trainer — that is the next milestone, out of scope for this plan.
