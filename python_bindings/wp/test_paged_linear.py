#!/usr/bin/env python3
"""Tests for PagedLinear (MAD-238 iteration 4)."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paged_linear import PagedLinear, swap_linears_with_paged  # noqa: E402


def test_paged_linear_constructs_with_no_initial_weight():
    """PagedLinear does NOT allocate a backing weight at construction —
    the pager owns it. .in_features / .out_features / .bias are inherited."""
    pl = PagedLinear(
        pager=None,           # not yet wired; weight_override is the only path
        page_idx=0,
        weight_shape=(8, 16), # (out, in)
        weight_dtype=torch.float16,
        bias=False,
    )
    assert pl.in_features == 16
    assert pl.out_features == 8
    # No `weight` parameter in state dict (it's NOT an nn.Parameter).
    assert "weight" not in dict(pl.named_parameters())
    print("  PASS test_paged_linear_constructs_with_no_initial_weight")


def test_paged_linear_override_path_forward_matches_nn_linear():
    """When weight_override is set, forward should match a plain nn.Linear
    with the same weight values bit-exactly (in fp32)."""
    torch.manual_seed(0)
    in_f, out_f, batch = 16, 8, 4
    W = torch.randn(out_f, in_f, dtype=torch.float32)
    x = torch.randn(batch, in_f, dtype=torch.float32)

    # Reference: plain nn.Linear
    ref = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        ref.weight.copy_(W)
    y_ref = ref(x)

    # PagedLinear with override
    pl = PagedLinear(pager=None, page_idx=0, weight_shape=(out_f, in_f),
                     weight_dtype=torch.float32, bias=False)
    pl.weight_override = W
    y_pl = pl(x)

    diff = (y_ref - y_pl).abs().max().item()
    assert diff == 0.0, f"PagedLinear override forward differs: {diff:.3e}"
    print("  PASS test_paged_linear_override_path_forward_matches_nn_linear")


def test_paged_linear_raises_without_pager_or_override():
    """Without weight_override and without a real pager, forward should
    raise clearly rather than silently producing zeros / garbage."""
    pl = PagedLinear(pager=None, page_idx=0, weight_shape=(4, 4),
                     weight_dtype=torch.float32, bias=False)
    try:
        pl(torch.randn(2, 4))
    except (RuntimeError, AttributeError) as e:
        msg = str(e).lower()
        assert "weight" in msg or "pager" in msg or "override" in msg, \
            f"error message should reference weight/pager/override, got: {e!r}"
        print(f"  PASS test_paged_linear_raises_without_pager_or_override ({type(e).__name__})")
        return
    assert False, "expected RuntimeError or AttributeError"


class _MockPager:
    """Minimal pager mock that returns a controllable src_ptr per page_idx.

    Tests can flip `ptr_for_idx[idx]` to simulate slot eviction-and-refill
    (where src_ptr changes for the same page_idx).
    """
    def __init__(self, ptr_for_idx):
        self.ptr_for_idx = dict(ptr_for_idx)
        self.ensure_calls = []

    def ensure(self, page_idx):
        self.ensure_calls.append(page_idx)
        return self.ptr_for_idx.get(page_idx, 0)


class _CachedTestLinear(PagedLinear):
    """PagedLinear test subclass: counts _materialize_weight calls and returns
    a CPU tensor (no GPU needed)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.materialize_calls = []

    def _materialize_weight(self, src_ptr):
        self.materialize_calls.append(src_ptr)
        # Return a fresh tensor each call so identity check is meaningful.
        return torch.full(self.weight_shape, float(src_ptr), dtype=torch.float32)


def test_paged_linear_caches_weight_when_src_ptr_unchanged():
    """Repeated .weight calls hit the cache while src_ptr is stable.

    This is the all-resident path (--pager-slots high enough to hold every page).
    Without caching, each forward pass would memcpy ~47 MB per linear, which
    dominated the 4B paged calibration runtime at 96 linears x 32 calib samples.
    """
    pager = _MockPager({0: 0xCAFE0000})
    pl = _CachedTestLinear(pager=pager, page_idx=0, weight_shape=(4, 4),
                            weight_dtype=torch.float32, bias=False)
    # First access: cache miss, materializes
    w1 = pl.weight
    assert len(pl.materialize_calls) == 1, "first access must materialize"
    assert pl.materialize_calls[0] == 0xCAFE0000
    # Second access: cache hit, same tensor object, no new materialization
    w2 = pl.weight
    assert len(pl.materialize_calls) == 1, "second access must NOT materialize"
    assert w1 is w2, "cached path must return same tensor object"
    # Repeated for good measure
    for _ in range(10):
        assert pl.weight is w1
    assert len(pl.materialize_calls) == 1
    print("  PASS test_paged_linear_caches_weight_when_src_ptr_unchanged")


def test_paged_linear_invalidates_cache_when_src_ptr_changes():
    """Slot eviction → re-fetch into different slot → src_ptr changes → cache miss.

    This is the LRU-eviction path (--pager-slots too small to hold every page).
    When the pager evicts our page and later re-faults it into a different slot,
    the src_ptr differs, the cache must invalidate, and we re-materialize.
    """
    pager = _MockPager({0: 0xCAFE0000})
    pl = _CachedTestLinear(pager=pager, page_idx=0, weight_shape=(4, 4),
                            weight_dtype=torch.float32, bias=False)
    w1 = pl.weight
    assert len(pl.materialize_calls) == 1
    # Simulate slot eviction-and-refill: same page_idx now lives at a different ptr
    pager.ptr_for_idx[0] = 0xBEEF0000
    w2 = pl.weight
    assert len(pl.materialize_calls) == 2, "ptr change must trigger re-materialize"
    assert w1 is not w2, "cache must return fresh tensor after invalidation"
    assert pl.materialize_calls == [0xCAFE0000, 0xBEEF0000]
    print("  PASS test_paged_linear_invalidates_cache_when_src_ptr_changes")


def test_paged_linear_weight_override_clears_cache():
    """Setting weight_override frees the paged-path cached tensor.

    After GPTQ writes back to weight_override, the original (pre-quant) cached
    weight is no longer needed — release its VRAM so the next layer's
    materialization doesn't double the high-water mark."""
    pager = _MockPager({0: 0xCAFE0000})
    pl = _CachedTestLinear(pager=pager, page_idx=0, weight_shape=(4, 4),
                            weight_dtype=torch.float32, bias=False)
    _ = pl.weight
    assert pl._cached_weight is not None
    assert pl._cached_src_ptr == 0xCAFE0000
    # Set override
    override = torch.ones(4, 4, dtype=torch.float32)
    pl.weight = override
    assert pl._cached_weight is None, "setter must clear cached tensor"
    assert pl._cached_src_ptr == 0, "setter must clear cached ptr"
    # Override takes precedence in getter
    assert pl.weight is override
    print("  PASS test_paged_linear_weight_override_clears_cache")


def test_paged_linear_raises_when_pager_ensure_returns_null():
    """If pager.ensure returns 0, raise a clear error (don't return zero tensor)."""
    pager = _MockPager({0: 0})   # page returns null
    pl = _CachedTestLinear(pager=pager, page_idx=0, weight_shape=(4, 4),
                            weight_dtype=torch.float32, bias=False)
    try:
        _ = pl.weight
    except RuntimeError as e:
        assert "page_idx=0" in str(e) and "null" in str(e).lower()
        print("  PASS test_paged_linear_raises_when_pager_ensure_returns_null")
        return
    assert False, "expected RuntimeError on null src_ptr"


def test_paged_linear_forward_is_dtype_transparent():
    """PagedLinear.forward computes in weight dtype but returns in input dtype.

    Contract: from the caller's perspective the layer is dtype-transparent —
    `out.dtype == in.dtype`. The page-loaded weight dtype is an internal detail.

    The matmul itself happens in weight dtype (the page-loaded bytes are
    authoritative — that's what an FP8 WMMA inference kernel will see), but
    the cast back to input dtype keeps the downstream model layers seeing
    what they expect.

    Bug surfaced in MAD-238 parity gate: HF dtype inference sees the
    parameter-less PagedLinear and emits bf16 hidden states even when the
    model was nominally loaded with torch_dtype=float16. v1 fix (input → weight
    cast only) made downstream non-paged linears mismatch. v2 fix (also output
    → input cast) restores transparency.
    """
    torch.manual_seed(0)
    in_f, out_f, batch = 8, 4, 2
    W_f16 = torch.randn(out_f, in_f, dtype=torch.float16)
    x_bf16 = torch.randn(batch, in_f, dtype=torch.bfloat16)

    pl = PagedLinear(pager=None, page_idx=0, weight_shape=(out_f, in_f),
                     weight_dtype=torch.float16, bias=False)
    pl.weight_override = W_f16

    # Must not raise even though input is bf16 and weight is f16
    y = pl(x_bf16)
    # Transparency: output dtype matches INPUT dtype, not weight dtype
    assert y.dtype == torch.bfloat16, f"expected output bf16 (matching input), got {y.dtype}"

    # The matmul should match nn.Linear-in-f16-then-cast-to-bf16
    ref = nn.Linear(in_f, out_f, bias=False).half()
    with torch.no_grad():
        ref.weight.copy_(W_f16)
    y_ref = ref(x_bf16.to(torch.float16)).to(torch.bfloat16)
    diff = (y.float() - y_ref.float()).abs().max().item()
    assert diff == 0.0, f"output diverges from explicit-cast reference: {diff:.3e}"
    print("  PASS test_paged_linear_forward_is_dtype_transparent")


def test_swap_linears_with_paged_replaces_matching_modules():
    """swap_linears_with_paged walks an nn.Module tree, replaces nn.Linear
    instances whose names match the predicate with PagedLinear stubs."""
    # Build a tiny model with 3 linears
    model = nn.Sequential(
        nn.Linear(8, 16, bias=False),   # 0
        nn.ReLU(),                       # 1 (skip)
        nn.Linear(16, 4, bias=False),    # 2
        nn.Linear(4, 2, bias=False),     # 3 (skip via predicate)
    )

    # Mock a pager that just returns a known page_idx per name
    class FakePager:
        def find_page(self, name):
            return {"l_0": 100, "l_2": 200}.get(name, -1)
        def page_meta(self, idx):
            class M: pass
            m = M()
            m.size = 0
            return m

    # Map module-path → catalog name. Skip the 3rd Linear (name "l_3") on purpose.
    name_map = {"0": "l_0", "2": "l_2"}

    n_swapped = swap_linears_with_paged(
        model, FakePager(),
        name_map=name_map,
        dtype=torch.float16,
    )
    assert n_swapped == 2, f"expected 2 swapped, got {n_swapped}"
    assert isinstance(model[0], PagedLinear)
    assert isinstance(model[2], PagedLinear)
    assert isinstance(model[3], nn.Linear) and not isinstance(model[3], PagedLinear)
    print("  PASS test_swap_linears_with_paged_replaces_matching_modules")


if __name__ == "__main__":
    test_paged_linear_constructs_with_no_initial_weight()
    test_paged_linear_override_path_forward_matches_nn_linear()
    test_paged_linear_raises_without_pager_or_override()
    test_paged_linear_caches_weight_when_src_ptr_unchanged()
    test_paged_linear_invalidates_cache_when_src_ptr_changes()
    test_paged_linear_weight_override_clears_cache()
    test_paged_linear_raises_when_pager_ensure_returns_null()
    test_paged_linear_forward_is_dtype_transparent()
    test_swap_linears_with_paged_replaces_matching_modules()
    print("\nALL TESTS PASSED")
