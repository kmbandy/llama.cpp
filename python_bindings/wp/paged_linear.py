"""PagedLinear — nn.Linear whose .weight comes from a wp_native.WeightPager.

Design: at model-load time, swap each target nn.Linear in an HF transformers
model with a PagedLinear. HF's forward code uses `self.q_proj(x)` and accesses
`.weight` transparently — the paged page-fault is invisible to the layer's
arithmetic.

Two weight paths:
  1. weight_override (post-quant): a torch tensor set by the calibration loop
     after quantizing this layer. While set, .weight returns the override and
     the pager is bypassed. Used so subsequent layers' forward pass sees the
     quantized output of this layer.
  2. paged (pre-quant): ensure_as_torch() reads from the pager slot. Used
     during forward passes that need the original weight (Hessian collection
     pass for THIS layer).

Lifetime: once weight_override is set, the pager slot is free to LRU-evict.

Caching: the paged path caches the materialized tensor keyed by the slot's
src_ptr. Because GGUF pages are read-only, identical src_ptr → identical bytes,
so cache reuse is safe. Each .weight access calls pager.ensure() (O(1) hash
lookup when resident); if the returned ptr matches the cached one, the cached
tensor is returned without re-memcpy. When eviction reassigns the slot to a
different page, src_ptr changes and the cache invalidates automatically.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn


class PagedLinear(nn.Linear):
    """nn.Linear with weight backed by a WeightPager (or a post-quant override)."""

    def __init__(self, pager, page_idx: int,
                 weight_shape, weight_dtype: torch.dtype,
                 bias: bool = False):
        # weight_shape is (out_features, in_features) per HF convention.
        out_features, in_features = int(weight_shape[0]), int(weight_shape[1])
        super().__init__(in_features, out_features, bias=bias)

        # Drop the auto-allocated weight Parameter — pager (or override) owns it.
        if "weight" in self._parameters:
            del self._parameters["weight"]

        self.pager = pager
        self.page_idx = page_idx
        self.weight_shape = (out_features, in_features)
        self.weight_dtype = weight_dtype
        self.weight_override: Optional[torch.Tensor] = None
        # Device index for the paged path (only used when override is unset).
        self.device_idx = 0
        # Cache for the page-faulted weight. _cached_src_ptr=0 is the
        # "no cache" sentinel (real GPU pointers are always nonzero).
        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_src_ptr: int = 0

    def _materialize_weight(self, src_ptr: int) -> torch.Tensor:
        """Allocate a fresh CUDA tensor and memcpy the slot bytes into it.

        Extracted as a method so tests can override it without touching GPU.
        Subclasses MUST return a torch.Tensor with shape=self.weight_shape and
        dtype=self.weight_dtype.
        """
        import wp_native
        out = torch.empty(self.weight_shape, dtype=self.weight_dtype,
                          device=f"cuda:{self.device_idx}")
        nbytes = out.element_size() * out.numel()
        wp_native.device_memcpy(out.data_ptr(), src_ptr, nbytes)
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Resolved weight tensor: override first, then cache, then materialize.

        Caching: pager.ensure() returns the slot's src_ptr (O(1) when resident).
        If src_ptr matches what we last materialized from, the cached tensor is
        still valid (GGUF bytes are immutable) and we return it without copying.
        When eviction reassigns the slot to a different page, src_ptr changes
        and we re-materialize.
        """
        if self.weight_override is not None:
            return self.weight_override
        if self.pager is None:
            raise RuntimeError(
                "PagedLinear.weight: no pager attached AND no weight_override set. "
                "Either wire a wp_native.WeightPager + page_idx, or assign weight_override."
            )
        src_ptr = self.pager.ensure(self.page_idx)
        if src_ptr == 0:
            raise RuntimeError(
                f"PagedLinear.weight: pager.ensure(page_idx={self.page_idx}) "
                f"returned null — page out of range or pager error."
            )
        if self._cached_weight is not None and self._cached_src_ptr == src_ptr:
            return self._cached_weight
        out = self._materialize_weight(src_ptr)
        self._cached_weight = out
        self._cached_src_ptr = src_ptr
        return out

    @weight.setter
    def weight(self, value):
        """Allow `layer.weight = some_tensor` to set the override.

        Clears the paged-path cache. The override takes precedence in the getter,
        but releasing the cached tensor frees its VRAM eagerly — useful in the
        calibration loop where post-quant overrides are set per-layer."""
        self.weight_override = value
        self._cached_weight = None
        self._cached_src_ptr = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute in weight dtype, return in input dtype — fully transparent.

        Page-loaded weights have fixed dtype (from the GGUF). The upstream
        layer may produce a different dtype (notably: HF's dtype inference
        sees the parameter-less PagedLinear and falls back to its model
        default). Cast input to the weight's dtype, do the matmul, then
        cast output back to the input's dtype so the next layer in the
        model gets what it expects.
        """
        import torch.nn.functional as F
        w = self.weight
        in_dtype = input.dtype
        if in_dtype != w.dtype:
            input = input.to(w.dtype)
        out = F.linear(input, w, self.bias)
        if out.dtype != in_dtype:
            out = out.to(in_dtype)
        return out


def swap_linears_with_paged(
    root: nn.Module,
    pager,
    name_map: Mapping[str, str],
    dtype: torch.dtype,
    device_idx: int = 0,
) -> int:
    """Walk root, replace each nn.Linear whose module path is in name_map with PagedLinear.

    name_map: {module_path_in_root → catalog_name_in_pager}.
    e.g. {"model.layers.0.mlp.gate_proj": "blk.0.ffn_gate.weight"}.

    Returns the number of modules replaced.
    """
    n_swapped = 0
    # First pass: collect (parent_module, child_name, old_linear, catalog_name)
    # tuples so we don't mutate while iterating.
    to_swap = []
    for module_path, catalog_name in name_map.items():
        try:
            parent_path, _, child_name = module_path.rpartition(".")
            parent = root
            if parent_path:
                for part in parent_path.split("."):
                    parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
            child = getattr(parent, child_name) if not child_name.isdigit() else parent[int(child_name)]
        except (AttributeError, IndexError):
            continue
        if not isinstance(child, nn.Linear):
            continue
        page_idx = pager.find_page(catalog_name)
        if page_idx < 0:
            continue
        to_swap.append((parent, child_name, child, page_idx))

    # Second pass: replace
    for parent, child_name, old, page_idx in to_swap:
        new_layer = PagedLinear(
            pager=pager,
            page_idx=page_idx,
            weight_shape=old.weight.shape,  # (out, in) for nn.Linear
            weight_dtype=dtype,
            bias=old.bias is not None,
        )
        new_layer.device_idx = device_idx
        if old.bias is not None:
            new_layer.bias = old.bias  # keep existing bias parameter
        if child_name.isdigit():
            parent[int(child_name)] = new_layer
        else:
            setattr(parent, child_name, new_layer)
        n_swapped += 1
    return n_swapped


__all__ = ["PagedLinear", "swap_linears_with_paged"]
