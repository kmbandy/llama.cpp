"""Resident (no-pager) MoE expert block for ml8 calibration.

`PagedMoeExperts` (python_bindings/wp/paged_linear.py) page-faults expert weights
from a GGUF via the wp_native HIP pager — needed on a 32 GB box where a 35B-A3B's
67 GB of experts don't fit. On a box that DOES fit the whole model (MI300X, 192 GB),
that machinery is unnecessary and, worse, the pager `.so` is arch-specific (won't
load on gfx942). `ResidentMoeExperts` holds the consolidated expert stacks resident
in VRAM and exposes the *exact same interface* the calibration quant loop drives
(`gate_proj`/`up_proj`/`down_proj` properties, `release_cached()`,
`reset_calibration_acc()`, and the routing forward with inline Hessian
accumulation). It is pure torch + gguf — NO wp_native — so it imports anywhere.

The forward is intentionally a verbatim copy of `PagedMoeExperts.forward`; the only
difference is the weight source (resident buffers vs page-fault materialize). Keep
them in sync if the routing/calibration math ever changes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts$")


def _decode_gguf_tensor(t) -> torch.Tensor:
    """Decode a GGUFReader tensor's raw bytes into a flat torch tensor of the
    stored dtype (bf16/f16/f32). Mirrors load_resident_to_model's decode."""
    from calibrate_ml8_paged import _gguf_dtype_to_torch  # reuse the dtype map
    td, npd = _gguf_dtype_to_torch(t.tensor_type)
    expected_numel = int(np.prod(t.shape))
    raw = np.asarray(t.data)
    if npd is not None:
        if raw.dtype == npd:
            arr = raw
        elif raw.dtype == np.uint8 and raw.size == expected_numel * np.dtype(npd).itemsize:
            arr = raw.view(npd)
        else:
            arr = raw.astype(npd)
        return torch.from_numpy(arr.copy())
    # BF16 (numpy has no native dtype): reinterpret as uint16, view as bfloat16.
    if raw.dtype == np.uint16 and raw.size == expected_numel:
        arr16 = raw
    elif raw.dtype == np.uint8 and raw.size == expected_numel * 2:
        arr16 = raw.view(np.uint16)
    else:
        raise ValueError(f"{t.name}: unexpected BF16 raw layout "
                         f"(dtype={raw.dtype}, size={raw.size}, numel={expected_numel})")
    return torch.from_numpy(arr16.copy()).view(torch.bfloat16)


def _load_expert_stack(t, shape, dtype, device) -> torch.Tensor:
    """Decode + reshape a GGUF expert stack to torch order. GGUF ne is reversed
    vs torch, so a torch [E, A, B] stack is stored with ne = [B, A, E]; decode
    flat then reshape to `shape` (which is the torch order we want)."""
    flat = _decode_gguf_tensor(t)
    if flat.numel() != int(np.prod(shape)):
        raise ValueError(f"{t.name}: numel {flat.numel()} != expected {int(np.prod(shape))} "
                         f"for shape {shape}")
    return flat.reshape(shape).to(dtype=dtype, device=device).contiguous()


class ResidentMoeExperts(nn.Module):
    """No-pager consolidated MoE experts. Drop-in for PagedMoeExperts."""

    def __init__(self, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor,
                 num_experts: int, intermediate_dim: int, hidden_dim: int, act_fn=None):
        super().__init__()
        self.num_experts = int(num_experts)
        self.intermediate_dim = int(intermediate_dim)
        self.hidden_dim = int(hidden_dim)
        self.act_fn = act_fn if act_fn is not None else F.silu
        # Resident weight stacks as PLAIN attributes (NOT register_buffer): the swap
        # runs before model.to_empty(), and to_empty() discards registered-buffer
        # storage — which would wipe these 67 GB of just-loaded expert weights. Plain
        # attributes are invisible to to_empty()/.to(), so they survive untouched.
        # They're already on the right device+dtype (loaded in swap_moe_experts_resident).
        # gate/up: [E, I, H]; down: [E, H, I].
        self._gate = gate_w
        self._up   = up_w
        self._down = down_w
        # Calibration Hessian accumulators (identical semantics to PagedMoeExperts).
        self.collect_pre_gate_up = False
        self.collect_pre_down = False
        self.pre_gate_up_acc = None
        self.pre_gate_up_n_tok = 0
        self.pre_down_acc = None
        self.pre_down_n_tok = 0

    # ── weight access (resident; no materialize/page-fault) ──
    @property
    def gate_proj(self) -> torch.Tensor:
        return self._gate

    @property
    def up_proj(self) -> torch.Tensor:
        return self._up

    @property
    def down_proj(self) -> torch.Tensor:
        return self._down

    def release_cached(self):
        """No-op: nothing is cached/paged in the resident path."""
        return

    def reset_calibration_acc(self):
        self.collect_pre_gate_up = False
        self.collect_pre_down = False
        self.pre_gate_up_acc = None
        self.pre_gate_up_n_tok = 0
        self.pre_down_acc = None
        self.pre_down_n_tok = 0

    # ── forward: VERBATIM from PagedMoeExperts.forward (weight source aside) ──
    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
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
                self.pre_gate_up_acc = xtx if self.pre_gate_up_acc is None else self.pre_gate_up_acc + xtx
                self.pre_gate_up_n_tok += xf.shape[0]
            gate = F.linear(current_state, gate_w[e])
            up   = F.linear(current_state, up_w[e])
            current_hidden_states = self.act_fn(gate) * up
            if self.collect_pre_down and current_hidden_states.numel() > 0:
                yf = current_hidden_states.detach().float()
                yty = yf.t() @ yf
                self.pre_down_acc = yty if self.pre_down_acc is None else self.pre_down_acc + yty
                self.pre_down_n_tok += yf.shape[0]
            current_hidden_states = F.linear(current_hidden_states, down_w[e])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        if final_hidden_states.dtype != in_dtype:
            final_hidden_states = final_hidden_states.to(in_dtype)
        return final_hidden_states


def swap_moe_experts_resident(model: nn.Module, gguf_path: str,
                              dtype: torch.dtype, device: str) -> int:
    """Replace every consolidated HF MoE block with a ResidentMoeExperts, loading
    its gate/up/down expert stacks resident from the GGUF. Returns the count.

    Must run BEFORE model.to_empty() the same way the paged swap does: it removes
    the HF expert Parameters from named_parameters (so to_empty won't materialize
    67 GB of meta experts), and holds the real weights resident instead.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib
    reader = gguf_lib.GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}

    to_swap = []
    for name, mod in model.named_modules():
        m = _HF_LAYER_RE.match(name)
        if m is None or not (hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj")):
            continue
        L = int(m.group(1))
        n_exp = int(getattr(mod, "num_experts", 0))
        inter = int(getattr(mod, "intermediate_dim", 0))
        hid   = int(getattr(mod, "hidden_dim", 0))
        if n_exp == 0 or inter == 0 or hid == 0:
            continue
        to_swap.append((name, mod, L, n_exp, inter, hid, getattr(mod, "act_fn", None)))

    n_swapped = 0
    for name, old, L, n_exp, inter, hid, act_fn in to_swap:
        gate_t = by_name.get(f"blk.{L}.ffn_gate_exps.weight")
        up_t   = by_name.get(f"blk.{L}.ffn_up_exps.weight")
        down_t = by_name.get(f"blk.{L}.ffn_down_exps.weight")
        if gate_t is None or up_t is None or down_t is None:
            print(f"  WARNING: layer {L} missing expert tensors in GGUF; skipping")
            continue
        gate_w = _load_expert_stack(gate_t, (n_exp, inter, hid), dtype, device)   # [E, I, H]
        up_w   = _load_expert_stack(up_t,   (n_exp, inter, hid), dtype, device)   # [E, I, H]
        down_w = _load_expert_stack(down_t, (n_exp, hid, inter), dtype, device)   # [E, H, I]
        new_mod = ResidentMoeExperts(gate_w, up_w, down_w, n_exp, inter, hid, act_fn)

        parent_path, _, child = name.rpartition(".")
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        setattr(parent, child, new_mod)
        n_swapped += 1
    return n_swapped
