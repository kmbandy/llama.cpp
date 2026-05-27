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


class PagedMoeExperts(nn.Module):
    """Paged replacement for Qwen3_5MoeExperts (and similar consolidated MoE blocks).

    Stock HF stores all experts in a single 3D Parameter:
      gate_up_proj [n_experts, 2*intermediate, hidden]   (gate stacked over up)
      down_proj    [n_experts, hidden, intermediate]

    For 35B-A3B that's ~1.5 GB per layer × 41 layers = 60 GB — too large to keep
    resident. Iter 5b pages at LAYER granularity: three pages per layer (gate,
    up, down), each one consolidated `[n_experts, ...]` slab. The standard MoE
    forward is identical to Qwen3_5MoeExperts.forward except gate/up are read
    from two separate paged tensors (matches GGUF tensor layout where
    `ffn_gate_exps` and `ffn_up_exps` are distinct tensors).

    Hessian-collection hook: when `experts_hessian_callback` is set, the
    forward calls it with the pre-MoE hidden_states subset routed to each
    expert. Calibration can use this OR (simpler) register a forward_hook on
    PagedMoeExperts to capture inputs[0] (all hidden_states entering the
    block) for a shared per-(layer, kind) Hessian.
    """

    def __init__(self, pager,
                 gate_page_idx: int, up_page_idx: int, down_page_idx: int,
                 n_experts: int, intermediate_dim: int, hidden_dim: int,
                 weight_dtype: torch.dtype, device_idx: int = 0,
                 act_fn=None):
        super().__init__()
        self.pager = pager
        self.gate_page_idx = gate_page_idx
        self.up_page_idx   = up_page_idx
        self.down_page_idx = down_page_idx
        self.num_experts = n_experts
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.weight_dtype = weight_dtype
        self.device_idx = device_idx
        if act_fn is None:
            import torch.nn.functional as F
            self.act_fn = F.silu
        else:
            self.act_fn = act_fn
        # Override slots (post-quant) — same lifecycle pattern as PagedLinear.
        self.gate_override: Optional[torch.Tensor] = None
        self.up_override:   Optional[torch.Tensor] = None
        self.down_override: Optional[torch.Tensor] = None
        # Calibration hooks. Set `collect_pre_down` / `collect_pre_gate_up` to True
        # to accumulate the activations seen by down_proj / gate+up_proj respectively
        # into `pre_down_acc` / `pre_gate_up_acc` (Hessian = sum X^T X, n_tokens
        # counters). Calibration clears these between layers; reset_calibration_acc().
        self.collect_pre_gate_up: bool = False
        self.collect_pre_down: bool = False
        self.pre_gate_up_acc: Optional[torch.Tensor] = None
        self.pre_gate_up_n_tok: int = 0
        self.pre_down_acc: Optional[torch.Tensor] = None
        self.pre_down_n_tok: int = 0
        # Materialization cache, keyed by pager src_ptr (same trick as PagedLinear).
        self._cached_gate = None;  self._cached_gate_ptr = 0
        self._cached_up   = None;  self._cached_up_ptr   = 0
        self._cached_down = None;  self._cached_down_ptr = 0

    def _materialize(self, page_idx: int, shape, cached_attr_t: str, cached_attr_p: str):
        """Copy the pager slot's bytes into a fresh torch tensor.

        Does NOT cache the torch tensor — calibration forward touches all 40
        layers in sequence and caching every layer's 1.5 GB of experts would
        OOM a 32 GB GPU. The pager slot stays resident (managed by wp_native
        LRU), so re-materializing is just a fast device→device memcpy. The
        returned tensor lives only as long as the caller's local scope.
        """
        import wp_native
        src_ptr = self.pager.ensure(page_idx)
        if src_ptr == 0:
            raise RuntimeError(f"PagedMoeExperts: pager.ensure(page_idx={page_idx}) returned null")
        out = torch.empty(shape, dtype=self.weight_dtype,
                          device=f"cuda:{self.device_idx}")
        nbytes = out.element_size() * out.numel()
        wp_native.device_memcpy(out.data_ptr(), src_ptr, nbytes)
        return out

    @property
    def gate_proj(self) -> torch.Tensor:
        if self.gate_override is not None: return self.gate_override
        return self._materialize(self.gate_page_idx,
                                  (self.num_experts, self.intermediate_dim, self.hidden_dim),
                                  "_cached_gate", "_cached_gate_ptr")

    @property
    def up_proj(self) -> torch.Tensor:
        if self.up_override is not None: return self.up_override
        return self._materialize(self.up_page_idx,
                                  (self.num_experts, self.intermediate_dim, self.hidden_dim),
                                  "_cached_up", "_cached_up_ptr")

    @property
    def down_proj(self) -> torch.Tensor:
        if self.down_override is not None: return self.down_override
        return self._materialize(self.down_page_idx,
                                  (self.num_experts, self.hidden_dim, self.intermediate_dim),
                                  "_cached_down", "_cached_down_ptr")

    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """Mirrors Qwen3_5MoeExperts.forward, but reads weights from paged
        tensors and treats gate/up as two distinct projections (matching the
        GGUF storage convention).
        """
        import torch.nn.functional as F
        gate_w = self.gate_proj
        up_w   = self.up_proj
        down_w = self.down_proj

        in_dtype = hidden_states.dtype
        if in_dtype != gate_w.dtype:
            hidden_states = hidden_states.to(gate_w.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            e = expert_idx[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            if self.collect_pre_gate_up and current_state.numel() > 0:
                xf = current_state.detach().float()
                xtx = xf.t() @ xf
                if self.pre_gate_up_acc is None:
                    self.pre_gate_up_acc = xtx
                else:
                    self.pre_gate_up_acc = self.pre_gate_up_acc + xtx
                self.pre_gate_up_n_tok += xf.shape[0]
            gate = F.linear(current_state, gate_w[e])
            up   = F.linear(current_state, up_w[e])
            current_hidden_states = self.act_fn(gate) * up
            if self.collect_pre_down and current_hidden_states.numel() > 0:
                yf = current_hidden_states.detach().float()
                yty = yf.t() @ yf
                if self.pre_down_acc is None:
                    self.pre_down_acc = yty
                else:
                    self.pre_down_acc = self.pre_down_acc + yty
                self.pre_down_n_tok += yf.shape[0]
            current_hidden_states = F.linear(current_hidden_states, down_w[e])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        if final_hidden_states.dtype != in_dtype:
            final_hidden_states = final_hidden_states.to(in_dtype)
        return final_hidden_states

    def release_cached(self):
        """Evict cached materialized tensors. Useful between layers in the
        calibration loop to keep the working set bounded."""
        self._cached_gate = None; self._cached_gate_ptr = 0
        self._cached_up   = None; self._cached_up_ptr   = 0
        self._cached_down = None; self._cached_down_ptr = 0

    def reset_calibration_acc(self):
        """Clear Hessian accumulators (call between layers in calibration)."""
        self.collect_pre_gate_up = False
        self.collect_pre_down = False
        self.pre_gate_up_acc = None
        self.pre_gate_up_n_tok = 0
        self.pre_down_acc = None
        self.pre_down_n_tok = 0


__all__ = ["PagedLinear", "swap_linears_with_paged", "PagedMoeExperts"]
