#!/usr/bin/env python3
"""calibrate_ml8_paged.py — calibrate ml8 weights with paged weight access.

MAD-238 iteration 5. Same calibration math as calibrate_ml8.py — reuses its
helpers (compute_hessian, gptq_quantize_linear, eval_ppl_wikitext) directly.
The difference is how the MLP linears' weights are sourced:

    calibrate_ml8.py        : HF loads entire model into VRAM via .to(device)
    calibrate_ml8_paged.py  : HF loads non-MLP weights normally; MLP linears
                              are wp_native.PagedLinear instances that
                              page-fault their .weight from a GGUF via wp_native.

Strategy modes (--strategy):

  dense (default, iter 5a) — full HF from_pretrained → all weights on GPU,
    then swap dense MLP linears (gate/up/down_proj) with PagedLinear.
    Works for dense models that fit host RAM (e.g. Qwen3.5-4B). Validated
    against the Cell C / Cell E results.

  moe (iter 5b) — bypass from_pretrained: instantiate the model with
    `torch.device("meta")` (zero host RAM), load resident (non-expert)
    tensors from the bf16 GGUF directly into GPU, and register consolidated
    MoE expert tensors in the pager with `n_experts=N` so each expert is its
    own sub-page. Swap every expert linear with PagedLinear targeting its
    per-expert sub-page (`...#expert.E`). For Qwen3.6 35B-A3B and friends.

Usage:
    python3 calibrate_ml8_paged.py \\
        --model Qwen/Qwen3.5-4B \\
        --gguf /home/kmbandy/models/Qwen3.5-4B-f16.gguf \\
        --output-dir /home/kmbandy/models/cell-paged \\
        --n-samples 32 --seq-len 1024 \\
        --rotation kronecker --snap-centroids e4m3 \\
        --fit-loss mse --group-size 64 --n-centroids 16 \\
        --eval-ppl --ppl-max-tokens 100000
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Re-use existing calibration helpers (compute_hessian / gptq_quantize_linear / eval_ppl_wikitext / etc.)
sys.path.insert(0, str(Path(__file__).parent))
from calibrate_ml8 import (  # noqa: E402
    collect_wikitext_calibration,
    compute_hessian,
    gptq_quantize_linear,
    eval_ppl_wikitext,
    find_target_linears,
    filter_by_layer_limit,
)
from centroid_quantizer import CentroidQuantizer  # noqa: E402
from kronecker_rotation import (  # noqa: E402
    KroneckerRotation, random_orthogonal, factor_for_dim, rotate_hessian,
)
from awq import compute_awq_scale, apply_awq_to_weight, absorb_awq_in_reconstruction  # noqa: E402
from batched_gptq import batched_gptq_quantize, batched_gptq_quantize_multigpu  # noqa: E402  (G.7.h.2/3)

# Pybind11 weight pager (in repo's python_bindings/wp/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_bindings" / "wp"))
import wp_native  # noqa: E402
from paged_linear import PagedLinear, swap_linears_with_paged, PagedMoeExperts  # noqa: E402


# ─── HF naming → GGUF naming map for Qwen MLP linears ──────────────────────

def _qwen_mlp_name_map(model: nn.Module) -> dict[str, str]:
    """Build {HF_module_path → GGUF_tensor_name} for every MLP linear in the model.

    Examples:
      "model.layers.0.mlp.gate_proj"  → "blk.0.ffn_gate.weight"
      "model.layers.5.mlp.up_proj"    → "blk.5.ffn_up.weight"
      "model.layers.31.mlp.down_proj" → "blk.31.ffn_down.weight"
    """
    hf_to_gguf_suffix = {
        "gate_proj": "ffn_gate",
        "up_proj":   "ffn_up",
        "down_proj": "ffn_down",
    }
    name_map = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # Match "model.layers.N.mlp.{gate_proj|up_proj|down_proj}"
        parts = name.split(".")
        if len(parts) < 5 or parts[-2] != "mlp" or parts[-1] not in hf_to_gguf_suffix:
            continue
        # Find the layer index
        try:
            layer_idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        gguf_name = f"blk.{layer_idx}.{hf_to_gguf_suffix[parts[-1]]}.weight"
        name_map[name] = gguf_name
    return name_map


# ─── Pager bootstrap ───────────────────────────────────────────────────────

def build_pager_from_gguf(gguf_path: str, device_idx: int,
                          n_slots: int, prefetch_depth: int = 4,
                          name_filter=None) -> wp_native.WeightPager:
    """Read the GGUF metadata via gguf-py, populate the wp catalog, init the pager.

    n_slots = size of VRAM ring. Pool size = n_slots × max_page_size where
    max_page_size = the largest *cataloged* tensor's size. Caller is responsible
    for VRAM math (per mad-lab GPU utilization rule).

    name_filter: optional callable str→bool. If provided, only tensors whose
    names pass the filter get cataloged. This is critical when only a subset of
    tensors will ever be page-faulted (e.g., MLP-only calibration) — including
    huge unused tensors like token_embd would balloon max_page_size to ~1.27 GB
    and make even modest n_slots OOM (96 slots × 1.27 GB ≈ 122 GB).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib

    pager = wp_native.WeightPager()
    reader = gguf_lib.GGUFReader(gguf_path)
    n_added = n_skipped = 0
    max_sz = 0
    for t in reader.tensors:
        if name_filter is not None and not name_filter(t.name):
            n_skipped += 1
            continue
        pager.add_page(t.name, 0, int(t.data_offset), int(t.n_bytes))
        max_sz = max(max_sz, int(t.n_bytes))
        n_added += 1
    print(f"[pager] catalog: added={n_added} skipped={n_skipped} "
          f"max_page={max_sz/1e6:.1f} MB  pool={n_slots*max_sz/1e9:.2f} GB")
    cfg = wp_native.Config()
    cfg.n_slots = n_slots
    cfg.prefetch_depth = prefetch_depth
    cfg.prefer_async_io = False  # SyncPread path — simpler for first cut
    ok = pager.init_for_device(cfg, device_idx, [gguf_path])
    if not ok:
        raise RuntimeError("wp_native.WeightPager.init_for_device returned False")
    return pager


def _mlp_only_name_filter(name: str) -> bool:
    """Catalog only block MLP weights (ffn_gate / ffn_up / ffn_down)."""
    return name.startswith("blk.") and any(
        name.endswith(f".{suffix}.weight")
        for suffix in ("ffn_gate", "ffn_up", "ffn_down")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Iter 5b — MoE-aware paths.
# ═══════════════════════════════════════════════════════════════════════════

# Matches `blk.<L>.ffn_(gate|up|down)_exps.weight` — the consolidated MoE
# expert stack tensor that the wp_native catalog splits into N sub-pages
# via `add_page(..., n_experts=N)`.
_MOE_EXPS_PATTERN = re.compile(r"^blk\.\d+\.(ffn_(?:gate|up|down)_exps)\.weight$")


def _moe_aware_name_map(model: nn.Module) -> dict[str, str]:
    """HF module path → pager catalog name, covering both dense MLP and MoE.

    Dense:  model.layers.{L}.mlp.{gate,up,down}_proj
            → blk.{L}.ffn_{gate,up,down}.weight
    MoE:    model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj
            → blk.{L}.ffn_{gate,up,down}_exps.weight#expert.{E}
    """
    dense_suffix = {"gate_proj": "ffn_gate",      "up_proj": "ffn_up",      "down_proj": "ffn_down"}
    moe_suffix   = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}
    out: dict[str, str] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        parts = name.split(".")
        # MoE: ...layers.{L}.mlp.experts.{E}.{kind}_proj  → 7+ parts
        if (len(parts) >= 7 and parts[-1] in moe_suffix
                and parts[-3] == "experts" and parts[-4] == "mlp"):
            try:
                L = int(parts[parts.index("layers") + 1])
                E = int(parts[-2])
            except (ValueError, IndexError):
                continue
            out[name] = f"blk.{L}.{moe_suffix[parts[-1]]}.weight#expert.{E}"
            continue
        # Dense: ...layers.{L}.mlp.{kind}_proj
        if parts[-1] in dense_suffix and len(parts) >= 2 and parts[-2] == "mlp":
            try:
                L = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            out[name] = f"blk.{L}.{dense_suffix[parts[-1]]}.weight"
    return out


def build_pager_iter5b(gguf_path: str, device_idx: int,
                       n_slots: int, n_experts: int,
                       prefetch_depth: int = 4,
                       consolidated_mode: bool = True) -> wp_native.WeightPager:
    """Build a pager catalog for the bf16 GGUF backing iter 5b.

    Strategy (consolidated_mode=True — the right mode for HF MoE classes
    that store experts as one Parameter):
      - `blk.L.ffn_*_exps.weight` → add_page (one page per kind per layer,
        slot size = the full consolidated tensor: n_experts × N × K × 2 bytes).
        Matches what PagedMoeExperts.materialize() expects to copy into VRAM.
      - `blk.L.ffn_*_shexp.weight` → add_page (shared expert MLP — one page)
      - `blk.L.ffn_{gate,up,down}.weight` (dense fallback) → add_page (one page)
      - all other tensors → skipped, loaded resident.

    consolidated_mode=False keeps the older sub-expert split (still usable
    for per-expert PagedLinear architectures).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib

    pager = wp_native.WeightPager()
    reader = gguf_lib.GGUFReader(gguf_path)
    n_moe = n_shexp = n_dense = n_skipped = 0
    for t in reader.tensors:
        if _MOE_EXPS_PATTERN.match(t.name):
            if consolidated_mode:
                pager.add_page(t.name, 0, int(t.data_offset), int(t.n_bytes))
            else:
                pager.add_page(t.name, 0, int(t.data_offset), int(t.n_bytes),
                               n_experts=n_experts)
            n_moe += 1
        elif t.name.startswith("blk.") and any(
                t.name.endswith(f".{suffix}.weight")
                for suffix in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")):
            # Shared experts are SMALL (~6 MB per layer, ~250 MB total for
            # 35B-A3B). Keep them resident — paging would inflate the slot
            # size if mixed with routed-expert pages and waste VRAM.
            n_shexp += 1
            n_skipped += 1
        elif t.name.startswith("blk.") and any(
                t.name.endswith(f".{suffix}.weight")
                for suffix in ("ffn_gate", "ffn_up", "ffn_down")):
            pager.add_page(t.name, 0, int(t.data_offset), int(t.n_bytes))
            n_dense += 1
        else:
            n_skipped += 1

    max_sz = pager.max_page_size()
    print(f"[pager-5b] moe_consolidated={n_moe} shared_expert={n_shexp} "
          f"dense_mlp={n_dense} skipped(resident-bound)={n_skipped}  "
          f"max_slot={max_sz/1e6:.1f} MB  pool={n_slots * max_sz / 1e9:.2f} GB")

    cfg = wp_native.Config()
    cfg.n_slots = n_slots
    cfg.prefetch_depth = prefetch_depth
    cfg.prefer_async_io = False
    ok = pager.init_for_device(cfg, device_idx, [gguf_path])
    if not ok:
        raise RuntimeError("wp_native.WeightPager.init_for_device returned False")
    return pager


_MOE_GATE_EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")


def swap_moe_experts_with_paged(model: nn.Module, pager: wp_native.WeightPager,
                                 dtype: torch.dtype, device_idx: int) -> int:
    """Walk `model`, replace every Qwen3_5MoeExperts (consolidated MoE block)
    with PagedMoeExperts using the pager. Returns the number of replacements.

    Looks up three page indices per layer:
        blk.{L}.ffn_gate_exps.weight
        blk.{L}.ffn_up_exps.weight
        blk.{L}.ffn_down_exps.weight
    """
    import re as _re
    HF_LAYER_RE = _re.compile(r"^model\.layers\.(\d+)\.mlp\.experts$")
    n_swapped = 0
    to_swap = []
    for name, mod in model.named_modules():
        m = HF_LAYER_RE.match(name)
        if m is None:
            continue
        # Must be the HF consolidated MoE block (has gate_up_proj + down_proj
        # Parameters or stub'd-as-Parameter on meta).
        if not (hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj")):
            continue
        L = int(m.group(1))
        n_experts = int(getattr(mod, "num_experts", 0))
        intermediate = int(getattr(mod, "intermediate_dim", 0))
        hidden = int(getattr(mod, "hidden_dim", 0))
        if n_experts == 0 or intermediate == 0 or hidden == 0:
            continue
        gate_idx = pager.find_page(f"blk.{L}.ffn_gate_exps.weight")
        up_idx   = pager.find_page(f"blk.{L}.ffn_up_exps.weight")
        down_idx = pager.find_page(f"blk.{L}.ffn_down_exps.weight")
        if min(gate_idx, up_idx, down_idx) < 0:
            print(f"  WARNING: layer {L} missing pages "
                  f"(gate={gate_idx} up={up_idx} down={down_idx}); skipping")
            continue
        to_swap.append((name, mod, L, gate_idx, up_idx, down_idx,
                        n_experts, intermediate, hidden,
                        getattr(mod, "act_fn", None)))

    for (name, old, L, gate_idx, up_idx, down_idx, n_experts, intermediate, hidden, act_fn) in to_swap:
        parent_path, _, child_name = name.rpartition(".")
        parent = model
        for part in parent_path.split("."):
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        new_mod = PagedMoeExperts(pager, gate_idx, up_idx, down_idx,
                                    n_experts=n_experts,
                                    intermediate_dim=intermediate,
                                    hidden_dim=hidden,
                                    weight_dtype=dtype,
                                    device_idx=device_idx,
                                    act_fn=act_fn)
        setattr(parent, child_name, new_mod)
        n_swapped += 1
    return n_swapped


# ─── Resident weight loader (iter 5b) ───────────────────────────────────────

# Maps GGUF tensor type code to numpy/torch dtype for *unquantized* tensors
# (bf16/f16/f32 — everything we need for resident loading). Quantized types
# would need dequant; the backing GGUF for iter 5b is expected to be bf16
# (via convert_hf_to_gguf.py with --outtype bf16) so this map covers us.
def _gguf_dtype_to_torch(t):
    import gguf as gguf_lib
    GGMLQuantizationType = gguf_lib.GGMLQuantizationType
    if t == GGMLQuantizationType.F32:
        return torch.float32, np.float32
    if t == GGMLQuantizationType.F16:
        return torch.float16, np.float16
    if t == GGMLQuantizationType.BF16:
        return torch.bfloat16, None   # numpy has no native bf16 — we'll wrap uint16
    raise ValueError(f"resident loader: unsupported GGUF dtype {t!r} "
                     f"(only F32/F16/BF16 are unquantized). Re-convert the "
                     f"backing GGUF with --outtype bf16 if needed.")


def _hf_param_to_gguf_name(arch_name: str, n_blocks: int) -> "callable":
    """Returns a function param_path → gguf_name (or None if no mapping).

    Wraps gguf-py's TensorNameMap so we get the same forward mapping that
    convert_hf_to_gguf.py uses. The lookup handles `.weight`/`.bias` suffixes.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    from gguf import MODEL_ARCH, TENSOR_NAMES
    from gguf.tensor_mapping import get_tensor_name_map

    # Try to resolve the MODEL_ARCH enum value from a config arch string.
    # Qwen3 MoE: arch="qwen3moe", enum=MODEL_ARCH.QWEN3MOE.
    arch_enum = None
    for ma in MODEL_ARCH:
        if ma.name.lower() == arch_name.lower():
            arch_enum = ma
            break
    if arch_enum is None:
        raise ValueError(f"resident loader: no MODEL_ARCH match for arch_name={arch_name!r}")
    name_map = get_tensor_name_map(arch_enum, n_blocks)

    def lookup(param_path: str) -> str | None:
        # TensorNameMap.get_name accepts the bare param path with optional
        # suffix stripping; mirror convert_hf_to_gguf.py's call style.
        for suffix in (".weight", ".bias", ""):
            if param_path.endswith(suffix) and suffix != "":
                stem = param_path[: -len(suffix)]
                gguf_stem = name_map.get_name(stem)
                if gguf_stem is not None:
                    return gguf_stem + suffix
        gguf_stem = name_map.get_name(param_path)
        return gguf_stem
    return lookup


def load_resident_to_model(model: nn.Module, gguf_path: str,
                            arch_name: str, n_blocks: int,
                            dtype: torch.dtype, device: str) -> int:
    """Load every non-expert tensor from the bf16 GGUF into model parameters.

    Walks `model.named_parameters()`. For each param:
      1. Compute its GGUF tensor name via TensorNameMap.
      2. Skip if name matches the MoE expert stack pattern — those are paged.
      3. Decode the GGUF tensor bytes (mmap'd by GGUFReader) into a torch
         tensor on `device` with `dtype`, then assign via `.data.copy_`.

    Tensors that have no GGUF counterpart (e.g. rotary_emb buffers, biases
    HF inserts but GGUF doesn't emit) are left at whatever value the meta-
    materialize put there. For Qwen3MoE that's only the rotary inv_freq
    buffer, which isn't a Parameter anyway.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib

    reader = gguf_lib.GGUFReader(gguf_path)
    gguf_by_name = {t.name: t for t in reader.tensors}
    lookup = _hf_param_to_gguf_name(arch_name, n_blocks)

    n_loaded = n_skipped_expert = n_no_mapping = n_missing_in_gguf = 0
    missing_examples: list[str] = []
    for param_path, param in model.named_parameters():
        gguf_name = lookup(param_path)
        if gguf_name is None:
            n_no_mapping += 1
            if len(missing_examples) < 6:
                missing_examples.append(f"NO-MAP {param_path}")
            continue
        # Skip MoE expert stacks — those are paged.
        if _MOE_EXPS_PATTERN.match(gguf_name):
            n_skipped_expert += 1
            continue
        t = gguf_by_name.get(gguf_name)
        if t is None:
            n_missing_in_gguf += 1
            if len(missing_examples) < 6:
                missing_examples.append(f"MISSING {param_path} -> {gguf_name}")
            continue

        # Decode GGUF bytes to torch tensor. GGUFReader returns t.data as a
        # numpy view that may be uint8 (raw bytes) regardless of stored dtype.
        # Use t.tensor_type + the expected element count to reinterpret safely.
        td, npd = _gguf_dtype_to_torch(t.tensor_type)
        expected_numel = int(np.prod(t.shape))
        raw = np.asarray(t.data)
        if npd is not None:
            # F32 / F16 — view (no copy) when dtypes already match, otherwise
            # reinterpret bytes.
            if raw.dtype == npd:
                arr = raw
            elif raw.dtype == np.uint8 and raw.size == expected_numel * np.dtype(npd).itemsize:
                arr = raw.view(npd)
            else:
                arr = raw.astype(npd)
            tensor = torch.from_numpy(arr.copy())
        else:
            # BF16: numpy has no native dtype. Reinterpret raw bytes as uint16
            # (each pair of bytes = one bf16 element), then torch.view as bfloat16.
            if raw.dtype == np.uint16 and raw.size == expected_numel:
                arr16 = raw
            elif raw.dtype == np.uint8 and raw.size == expected_numel * 2:
                arr16 = raw.view(np.uint16)
            else:
                raise ValueError(
                    f"{t.name}: unexpected BF16 raw layout "
                    f"(dtype={raw.dtype}, size={raw.size}, expected_numel={expected_numel})")
            tensor = torch.from_numpy(arr16.copy()).view(torch.bfloat16)
        # GGUF stores ne[0] as the contiguous dim (column-major-ish to HF
        # eyes for 2D weights). The shape from t.shape is reversed from
        # HF's (out_features, in_features) for Linear. Normalize:
        gguf_shape = tuple(int(d) for d in t.shape)
        # Reverse to match HF parameter shape if dimensions match in reversed form.
        if tensor.numel() != int(np.prod(param.shape)):
            raise ValueError(
                f"{param_path}: numel mismatch (param={param.numel()} "
                f"gguf={tensor.numel()} via gguf_name={gguf_name})")
        # Try param.shape directly first; fall back to reversed.
        try:
            tensor = tensor.reshape(param.shape)
        except RuntimeError:
            tensor = tensor.reshape(tuple(reversed(gguf_shape))).reshape(param.shape)

        tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
        with torch.no_grad():
            param.data = tensor
        n_loaded += 1

    print(f"[resident-load] loaded={n_loaded}  skipped_expert={n_skipped_expert}  "
          f"no_mapping={n_no_mapping}  missing_in_gguf={n_missing_in_gguf}")
    if missing_examples:
        print(f"[resident-load] examples (first {len(missing_examples)}):")
        for ex in missing_examples:
            print(f"    {ex}")
    return n_loaded


def _detect_moe_n_experts(config) -> int | None:
    """Return n_experts if `config` describes an MoE model, else None.

    Qwen3MoE uses `num_experts`. Some MoE classes use `num_local_experts`
    or `n_routed_experts`. Probe in that order.
    """
    for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 1:
            return v
    return None


def _group_moe_targets_by_layer_and_kind(targets):
    """Bucket `targets` (yielded by find_target_linears) by (layer_idx, kind).

    Returns dict[(layer_idx, kind)] → list[(name, module)] sorted by expert id.
    `kind` is one of "gate_proj", "up_proj", "down_proj".
    Layers/linears that don't match the MoE expert pattern are dropped.
    """
    out: dict[tuple[int, str], list[tuple[str, nn.Module]]] = {}
    for name, mod in targets:
        parts = name.split(".")
        if (len(parts) < 7 or parts[-3] != "experts" or parts[-4] != "mlp"):
            continue
        if parts[-1] not in ("gate_proj", "up_proj", "down_proj"):
            continue
        try:
            L = int(parts[parts.index("layers") + 1])
            E = int(parts[-2])
        except (ValueError, IndexError):
            continue
        out.setdefault((L, parts[-1]), []).append((E, name, mod))
    # Sort each bucket by expert id; strip the expert id from the result tuple.
    out2: dict[tuple[int, str], list[tuple[str, nn.Module]]] = {}
    for k, v in out.items():
        v.sort(key=lambda x: x[0])
        seen_eids = [t[0] for t in v]
        if seen_eids != list(range(len(v))):
            raise RuntimeError(
                f"layer {k[0]} kind {k[1]}: expert ids not contiguous 0..N-1; got {seen_eids}")
        out2[k] = [(n, m) for _, n, m in v]
    return out2


def _collect_hessians_layer_moe(experts, calib_ids, model, dev, collect_awq=False):
    """Single forward pass over `calib_ids`, hooks on every expert linear in
    `experts`, accumulates H_e and (optionally) sum_abs_e per expert.

    Returns:
        H_list:       [len(experts)] of torch.Tensor [K, K]  (per-expert H = E[x x^T])
        n_tok_list:   [len(experts)] of int                  (tokens routed to that expert)
        sum_abs_list: [len(experts)] of torch.Tensor [K] OR None
    """
    H_acc = [None] * len(experts)
    sum_abs = [None] * len(experts) if collect_awq else None
    n_tot = [0] * len(experts)

    def make_hook(e_idx):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()  # [N_tok_e, K]
            XtX = x.t() @ x
            if H_acc[e_idx] is None:
                H_acc[e_idx] = XtX
            else:
                H_acc[e_idx] += XtX
            if collect_awq:
                sa = x.abs().sum(dim=0)
                if sum_abs[e_idx] is None:
                    sum_abs[e_idx] = sa
                else:
                    sum_abs[e_idx] += sa
            n_tot[e_idx] += x.shape[0]
        return hook

    handles = []
    for e_idx, (_, mod) in enumerate(experts):
        handles.append(mod.register_forward_hook(make_hook(e_idx)))
    try:
        with torch.no_grad():
            for ids in calib_ids:
                model(ids.to(dev))
    finally:
        for h in handles:
            h.remove()

    # Normalize per-expert H to E[x x^T] by dividing by per-expert n_tok (matches
    # the scalar `compute_hessian` semantics).
    H_list = []
    for e_idx in range(len(experts)):
        if H_acc[e_idx] is None:
            # Cold expert — no tokens routed. Use identity-scaled H as a
            # placeholder; downstream code can detect via n_tok==0 and skip.
            K_dim = experts[e_idx][1].in_features
            H_list.append(torch.eye(K_dim, device=dev) * 1e-6)
        else:
            H_list.append(H_acc[e_idx] / max(n_tot[e_idx], 1))
    return H_list, n_tot, sum_abs


def _build_model_meta(model_name: str, dtype: torch.dtype):
    """Instantiate `model_name` on the meta device — no host RAM allocated
    for parameters. Returns (model, text_cfg, n_blocks).

    For multimodal vision-language MoE configs (e.g. Qwen3_5MoeForConditionalGeneration
    on Qwen3.6 35B-A3B), the top-level config doesn't carry vocab_size /
    num_hidden_layers — those live under `text_config`. AutoModelForCausalLM
    on the full config errors out. Detect that and route to the text-only
    CausalLM class directly using the text_config.

    PyTorch's `with torch.device("meta"):` context routes all newly-created
    tensors to the meta device. The model is fully usable for
    `.named_parameters()` (resident loading) and `.named_modules()`
    (PagedLinear swap).
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Multimodal carrier with a text sub-config? Extract the text decoder.
    text_cfg = config
    if hasattr(config, "text_config") and config.text_config is not None:
        text_cfg = config.text_config

    with torch.device("meta"):
        # Try the generic AutoModel path first. Falls back to importing the
        # text-only class by name when AutoModel can't infer it.
        try:
            model = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=True)
        except (AttributeError, ValueError, KeyError):
            text_arch = type(text_cfg).__name__.replace("Config", "ForCausalLM")
            import transformers as _tf
            model_cls = getattr(_tf, text_arch, None)
            if model_cls is None:
                raise RuntimeError(
                    f"could not resolve text-only CausalLM class for text_config "
                    f"type {type(text_cfg).__name__!r}; tried '{text_arch}'.")
            model = model_cls(text_cfg)

    n_blocks = int(getattr(text_cfg, "num_hidden_layers",
                           getattr(text_cfg, "n_layer", 0)))
    if n_blocks <= 0:
        raise RuntimeError(f"could not determine n_blocks from text_config "
                           f"{type(text_cfg).__name__}")
    return model, text_cfg, n_blocks


# ─── Main driver ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B",
                   help="HF model name. Loaded with empty weights for non-MLP, "
                        "paged for MLP. (Iter 5a: only MLP is paged.)")
    p.add_argument("--gguf", required=True,
                   help="GGUF file that backs the paged MLP weights. Must contain "
                        "the same model as --model; mismatched offsets = silent garbage.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=25)
    p.add_argument("--fit-loss", choices=("mse", "mag_weighted"), default="mse")
    p.add_argument("--mag-weight-p", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--eval-ppl", action="store_true")
    p.add_argument("--ppl-max-tokens", type=int, default=None)
    p.add_argument("--rotation", choices=("none", "kronecker"), default="none")
    p.add_argument("--rotation-seed", type=int, default=42)
    p.add_argument("--rotation-max-b", type=int, default=1024)
    p.add_argument("--snap-centroids", choices=("none", "e4m3"), default="none")
    p.add_argument("--awq", choices=("none", "mean"), default="none")
    p.add_argument("--awq-alpha", type=float, default=0.5)
    p.add_argument("--pager-slots", type=int, default=8,
                   help="WeightPager VRAM ring size. Pool = slots × max_page_size. "
                        "For Qwen3.5-4B, max_page_size ≈ 1.27 GB (token_embd) → "
                        "8 slots = ~10 GB pool. Don't exceed available headroom.")
    p.add_argument("--strategy", choices=("dense", "moe"), default="dense",
                   help="dense (iter 5a) = from_pretrained, all weights on GPU, "
                        "paged dense MLPs. moe (iter 5b) = torch.device('meta') "
                        "instantiate, load resident from GGUF directly, page "
                        "consolidated MoE experts. Use 'moe' for any model that "
                        "doesn't fit in host RAM (e.g. Qwen3.6 35B-A3B).")
    p.add_argument("--arch", default=None,
                   help="(--strategy moe only) MODEL_ARCH name for TensorNameMap "
                        "(e.g. 'qwen3moe'). If omitted, derived from the config "
                        "class name lowercased.")
    p.add_argument("--secondary-device", default=None,
                   help="(--strategy moe only) Second GPU to split MoE batched-GPTQ "
                        "across (e.g. 'cuda:1'). When set, the layer-major loop "
                        "dispatches half the experts to the primary device and the "
                        "rest to this one in parallel via Python threads. "
                        "Output is bit-identical to single-GPU.")
    p.add_argument("--primary-share", type=float, default=0.7,
                   help="(--secondary-device only) Fraction of experts to keep on "
                        "the primary device. Default 0.7 matches R9700(32GB)/6900XT(16GB) "
                        "VRAM ratio + the gfx1030 secondary being slower per FLOP.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.strategy == "dense":
        # ── Iter 5a: from_pretrained (all weights resident on GPU), paged dense MLPs.
        print(f"[load-hf] strategy=dense  {args.model}  dtype={dtype}  device={args.device}")
        # NOTE: `torch_dtype` is deprecated and silently ignored in newer
        # transformers; use `dtype=` AND explicit `.to(dtype)` so the dtype
        # is actually honored.
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        model = model.to(args.device).to(dtype).eval()
        actual_dtypes = {p.dtype for p in model.parameters()}
        print(f"[load-hf] post-load dtypes={actual_dtypes}")
        if actual_dtypes != {dtype}:
            raise RuntimeError(
                f"Model load did not honor dtype={dtype}; got {actual_dtypes}. "
                f"Paged calibration requires uniform model dtype matching the GGUF.")

        print(f"[pager] building from {args.gguf}  slots={args.pager_slots}")
        pager = build_pager_from_gguf(args.gguf, device_idx, args.pager_slots,
                                       name_filter=_mlp_only_name_filter)
        print(f"[pager] catalog: {pager.n_pages()} tensors  max_page={pager.max_page_size()/1e6:.1f} MB")

        name_map = _qwen_mlp_name_map(model)
        print(f"[swap] HF↔GGUF MLP mapping: {len(name_map)} linears")
        n_swapped = swap_linears_with_paged(model, pager, name_map, dtype=dtype, device_idx=device_idx)
        print(f"[swap] replaced {n_swapped} nn.Linear → PagedLinear")
        if n_swapped == 0:
            raise RuntimeError("swap_linears_with_paged found nothing to swap — check name_map")
    else:
        # ── Iter 5b: meta-device instantiate, resident load from GGUF, paged MoE experts.
        print(f"[load-hf] strategy=moe  {args.model}  dtype={dtype}  device={args.device}")
        model, config, n_blocks = _build_model_meta(args.model, dtype)
        n_experts = _detect_moe_n_experts(config)
        if n_experts is None:
            raise RuntimeError(
                f"--strategy moe expects an MoE config (num_experts / num_local_experts / "
                f"n_routed_experts). Got {type(config).__name__} with none set. "
                f"Use --strategy dense for dense models.")
        arch_name = args.arch or type(config).__name__.replace("Config", "").lower()
        print(f"[meta-init] arch={arch_name}  n_blocks={n_blocks}  n_experts={n_experts}")

        # ORDER MATTERS — for a 35B-A3B MoE the experts alone would OOM the
        # GPU if to_empty() materializes them. Sequence:
        #   1. Build pager catalog (catalog only — VRAM pool allocated by init).
        #   2. Swap PagedLinear for every expert UNDER torch.device("meta")
        #      so PagedLinear's super().__init__() Linear weight stays on
        #      meta (then gets del'd from _parameters). After this the
        #      expert weights are no longer reachable via named_parameters.
        #   3. to_empty(device) — only materializes the REMAINING resident
        #      parameters (token_embd, attention, ssm_*, norms, MTP, etc.).
        #   4. Resident loader fills those params from GGUF bytes.
        print(f"[pager] iter 5b — building catalog from {args.gguf}  "
              f"slots={args.pager_slots}  n_experts={n_experts}")
        pager = build_pager_iter5b(args.gguf, device_idx, args.pager_slots,
                                    n_experts=n_experts)
        print(f"[pager] catalog: {pager.n_pages()} entries  "
              f"max_page={pager.max_page_size()/1e6:.1f} MB")

        # ── Swap consolidated MoE expert blocks with PagedMoeExperts.
        #    HF stores routed experts as one Parameter per kind per layer
        #    (gate_up_proj [E, 2*I, H], down_proj [E, H, I]). PagedMoeExperts
        #    replaces the whole block with paged read-on-demand tensors.
        n_swapped_moe = swap_moe_experts_with_paged(model, pager, dtype, device_idx)
        print(f"[swap] replaced {n_swapped_moe} consolidated MoE expert blocks → PagedMoeExperts")
        if n_swapped_moe == 0:
            raise RuntimeError("swap_moe_experts_with_paged found 0 expert blocks")

        # Shared experts (mlp.shared_expert.*) are kept RESIDENT — they're small
        # (~250 MB total for 35B-A3B) and live in the model as ordinary
        # nn.Linear modules. to_empty + resident loader handle them.

        # Now materialize the remaining resident params.
        n_resident_params = sum(p.numel() for p in model.parameters())
        print(f"[meta-empty] materializing {n_resident_params/1e9:.2f}B resident params on {args.device}")
        model = model.to_empty(device=args.device).to(dtype).eval()

        # Load every non-expert tensor from the bf16 GGUF into the model.
        n_loaded = load_resident_to_model(
            model, args.gguf, arch_name=arch_name, n_blocks=n_blocks,
            dtype=dtype, device=args.device)
        if n_loaded == 0:
            raise RuntimeError(
                "[resident-load] loaded 0 tensors from GGUF — arch_name mismatch? "
                f"Tried arch={arch_name!r}. Use --arch to override.")

        # G.7.i smoke: short-circuit BEFORE calibration so we verify the model
        # loads (swap + to_empty + resident load) without OOM. Run a single
        # tiny forward pass to prove the PagedMoeExperts path works end-to-end.
        # Remove this block once the consolidated layer-major loop is in.
        if os.environ.get("ML8_MOE_LOAD_SMOKE") == "1":
            print("[smoke] running single dummy forward to verify model + pager...")
            with torch.no_grad():
                dummy = torch.zeros((1, 16), dtype=torch.long, device=args.device)
                out = model(dummy)
            print(f"[smoke] forward OK — output logits shape={tuple(out.logits.shape)}")
            pager.shutdown()
            return

    # ─── Calibration corpus + baseline PPL (paged forward proves the swap works) ───
    print(f"[calib] loading {args.n_samples} samples seq_len={args.seq_len}")
    calib = collect_wikitext_calibration(tokenizer, n_samples=args.n_samples,
                                          seq_len=args.seq_len)
    print(f"[calib] got {len(calib)} samples (tokens ≈ {sum(c.numel() for c in calib)})")

    targets = list(find_target_linears(model))   # finds MLP linears (now PagedLinear)
    targets = filter_by_layer_limit(targets, args.max_layers)
    print(f"[targets] {len(targets)} linears to quantize")

    manifest = {"model": args.model, "gguf": args.gguf, "args": vars(args), "results": []}

    if args.eval_ppl:
        print(f"\n[ppl-baseline] computing f16 baseline PPL (paged forward path)...")
        baseline = eval_ppl_wikitext(model, tokenizer, args.device,
                                      max_tokens=args.ppl_max_tokens)
        print(f"  baseline PPL = {baseline['ppl']:.4f}")
        manifest["ppl_baseline"] = baseline

    # ═══════════════════════════════════════════════════════════════════
    # MoE layer-major calibration path (G.7.h.1). One forward pass per
    # layer per "stage" (gate+up share a stage, down is its own), captures
    # Hessians for all 128 experts × 3 kinds via per-linear hooks. Then
    # rotates, AWQs, stacks, runs batched_gptq_quantize over the [E, N, K]
    # tensor. Save per-expert .pt blobs identical to scalar output so
    # ml8_to_gguf.py just works.
    # ═══════════════════════════════════════════════════════════════════
    if args.strategy == "moe":
        groups = _group_moe_targets_by_layer_and_kind(targets)
        layer_ids = sorted({L for (L, _) in groups})
        if args.max_layers is not None:
            layer_ids = layer_ids[: args.max_layers]
        print(f"[moe-loop] {len(layer_ids)} layers x 3 kinds; groups={len(groups)}")

        for layer_idx in layer_ids:
            t_layer = time.time()
            print(f"\n=== Layer {layer_idx} ===")

            # Two stages per layer:
            #   stage 1: gate_proj + up_proj experts (same input X)
            #   stage 2: down_proj experts (uses quantized gate/up outputs)
            for stage, kinds in (("gate_up", ("gate_proj", "up_proj")),
                                  ("down",    ("down_proj",))):
                experts_flat: list[tuple[str, nn.Module, int, str]] = []
                for kind in kinds:
                    grp = groups.get((layer_idx, kind))
                    if not grp:
                        continue
                    for e_id, (nm, mod) in enumerate(grp):
                        experts_flat.append((nm, mod, e_id, kind))
                if not experts_flat:
                    continue

                t_h = time.time()
                H_list, n_tok_list, sum_abs_list = _collect_hessians_layer_moe(
                    [(n, m) for n, m, _, _ in experts_flat], calib, model, args.device,
                    collect_awq=(args.awq != "none"))
                t_h_done = time.time() - t_h
                tot_tok = sum(n_tok_list)
                print(f"  [{stage}] {len(experts_flat)} linears  n_tok_total={tot_tok}  "
                      f"t_hessian={t_h_done:.1f}s")

                # Process per-kind so each kind's experts form a clean [E, N, K] stack.
                for kind in kinds:
                    kind_entries = [(idx, e) for idx, e in enumerate(experts_flat) if e[3] == kind]
                    if not kind_entries:
                        continue
                    flat_indices = [idx for idx, _ in kind_entries]
                    E_cnt = len(flat_indices)
                    # Reference dimensions from expert 0.
                    ref_layer = experts_flat[flat_indices[0]][1]
                    rows, K = ref_layer.weight.shape   # PagedLinear.weight page-faults here

                    # ── Per-(layer, kind) rotation + per-expert AWQ.
                    rotation = None
                    rotation_blob = None
                    h_a_to_save = None
                    if args.rotation == "kronecker":
                        a, b = factor_for_dim(K, max_b=args.rotation_max_b)
                        # Deterministic seed per (layer, kind) — shared across experts.
                        seed = args.rotation_seed + layer_idx * 7 + {"gate_proj": 0, "up_proj": 1, "down_proj": 2}[kind]
                        h_a = random_orthogonal(a, seed=seed)
                        rotation = KroneckerRotation(h_a=h_a, b_dim=b)
                        rotation_blob = rotation.to_dict()
                        rotation_blob["seed"] = seed
                        h_a_to_save = rotation_blob["h_a"]
                        print(f"  [{stage}/{kind}] rotation kronecker a={a} b={b}")

                    # Build stacked W and H. Also build per-expert weight snapshot
                    # for fallback in case GPTQ fails for an expert.
                    W_stack = torch.empty((E_cnt, rows, K), device=args.device, dtype=dtype)
                    H_stack = torch.empty((E_cnt, K, K), device=args.device, dtype=torch.float32)
                    awq_s_stack: list[torch.Tensor | None] = [None] * E_cnt
                    n_tok_kind = torch.zeros((E_cnt,), dtype=torch.long, device=args.device)
                    cold_eids: list[int] = []
                    for j, (flat_idx, (nm, mod, e_id, _kd)) in enumerate(zip(flat_indices, [experts_flat[fi] for fi in flat_indices])):
                        W_orig = mod.weight.detach().float()
                        H_e = H_list[flat_idx].to(args.device).float()
                        if n_tok_list[flat_idx] == 0:
                            cold_eids.append(e_id)
                        # Per-expert AWQ
                        if args.awq != "none" and sum_abs_list is not None and sum_abs_list[flat_idx] is not None:
                            mean_abs = (sum_abs_list[flat_idx] / max(n_tok_list[flat_idx], 1)).clamp_min(1e-8)
                            awq_e = mean_abs.pow(args.awq_alpha).to(args.device)
                            awq_s_stack[j] = awq_e
                            H_e = H_e * awq_e.unsqueeze(0) * awq_e.unsqueeze(1)
                            W_orig = apply_awq_to_weight(W_orig, awq_e)
                        if rotation is not None:
                            H_e = rotate_hessian(H_e, rotation)
                            W_orig = rotation.forward(W_orig.to(rotation.h_a.device)).to(args.device)
                        W_stack[j] = W_orig.to(dtype)
                        H_stack[j] = H_e
                        n_tok_kind[j] = n_tok_list[flat_idx]
                    if cold_eids:
                        print(f"  [{stage}/{kind}] WARNING {len(cold_eids)} cold experts (n_tok=0): {cold_eids[:8]}...")

                    # ── Batched GPTQ over all experts of this (layer, kind).
                    t_q = time.time()
                    effective_percdamp = args.percdamp
                    if any(s is not None for s in awq_s_stack):
                        effective_percdamp = max(args.percdamp, 0.05)
                    if args.secondary_device:
                        out = batched_gptq_quantize_multigpu(
                            W_stack=W_stack.float(),
                            H_stack=H_stack,
                            primary_device=args.device,
                            secondary_device=args.secondary_device,
                            primary_share=args.primary_share,
                            n_centroids=args.n_centroids,
                            group_size=args.group_size,
                            n_iter=args.n_iter,
                            fit_loss=args.fit_loss,
                            mag_weight_p=args.mag_weight_p,
                            snap_centroids=args.snap_centroids,
                            percdamp=effective_percdamp,
                            n_tokens_per_expert=n_tok_kind,
                        )
                    else:
                        out = batched_gptq_quantize(
                            W_stack=W_stack.float(),
                            H_stack=H_stack,
                            n_centroids=args.n_centroids,
                            group_size=args.group_size,
                            n_iter=args.n_iter,
                            fit_loss=args.fit_loss,
                            mag_weight_p=args.mag_weight_p,
                            snap_centroids=args.snap_centroids,
                            percdamp=effective_percdamp,
                            n_tokens_per_expert=n_tok_kind,
                        )
                    t_q_done = time.time() - t_q
                    indices = out["indices"]                  # [E, N, K] int8
                    centroids = out["centroids_per_group"]    # [E, n_groups, n_centroids]
                    scales = out["scale_per_group"]           # [E, N, n_groups]
                    Q_all = out["Q"]                          # [E, N, K] fp32 dequantized
                    mse_all = out["mse"]; w_snr = out["w_snr_db"]; y_snr = out["y_snr_db"]; rel = out["rel_err"]

                    # Inverse rotation + AWQ absorb (per expert) so subsequent
                    # layers' forward pass sees inference-equivalent weights.
                    for j, (flat_idx, exp) in enumerate(zip(flat_indices, [experts_flat[fi] for fi in flat_indices])):
                        nm, mod, e_id, _kd = exp
                        W_q = Q_all[j].float().to(args.device)
                        if rotation is not None:
                            W_q = rotation.inverse(W_q.to(rotation.h_a.device)).to(args.device)
                        if awq_s_stack[j] is not None:
                            W_q = absorb_awq_in_reconstruction(W_q.float(), awq_s_stack[j].to(W_q.device))
                        mod.weight_override = W_q.to(dtype).to(args.device)

                        # Per-expert .pt blob (same schema as scalar path).
                        out_path = Path(args.output_dir) / f"{nm.replace('.', '_').replace('/', '_')}.pt"
                        blob = {
                            "name": nm,
                            "shape": [rows, K],
                            "group_size": args.group_size,
                            "n_centroids": args.n_centroids,
                            "indices": indices[j].cpu(),
                            "centroids_per_group": centroids[j].cpu(),
                            "scale_per_group": scales[j].cpu(),
                            "mse": float(mse_all[j].item()),
                            "w_snr_db": float(w_snr[j].item()),
                            "y_snr_db": float(y_snr[j].item()),
                            "rel_err": float(rel[j].item()),
                        }
                        if rotation_blob is not None:
                            blob["rotation"] = dict(rotation_blob)
                        if awq_s_stack[j] is not None:
                            blob["awq"] = {"kind": args.awq, "alpha": args.awq_alpha,
                                            "s": awq_s_stack[j].detach().cpu()}
                        torch.save(blob, out_path)
                        manifest["results"].append({
                            "name": nm, "shape": [rows, K],
                            "mse": float(mse_all[j].item()),
                            "y_snr_db": float(y_snr[j].item()),
                            "w_snr_db": float(w_snr[j].item()),
                        })

                    print(f"  [{stage}/{kind}] batched_gptq E={E_cnt} K={K} N={rows}  "
                          f"t_quant={t_q_done:.1f}s  median Y_SNR={float(y_snr.median().item()):.2f}dB")
                    del W_stack, H_stack, indices, centroids, scales, Q_all
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()

            print(f"  layer {layer_idx} total t={time.time()-t_layer:.1f}s")

        # Done with the MoE path — skip the dense per-linear loop.
        if args.eval_ppl:
            print(f"\n[ppl-quant] computing quantized PPL...")
            quantized = eval_ppl_wikitext(model, tokenizer, args.device,
                                           max_tokens=args.ppl_max_tokens)
            print(f"  quantized PPL = {quantized['ppl']:.4f}")
            manifest["ppl_quantized"] = quantized
            delta = quantized["ppl"] - manifest["ppl_baseline"]["ppl"]
            manifest["ppl_delta"] = delta
            print(f"\n  Δ_PPL = {delta:+.4f}")

        manifest_path = Path(args.output_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[done] manifest: {manifest_path}")
        pager.shutdown()
        return

    # ─── Per-layer calibration loop (same math as calibrate_ml8.py) ───
    for i, (name, layer) in enumerate(targets):
        t0 = time.time()
        rows, in_feat = layer.weight.shape   # PagedLinear.weight page-faults here
        print(f"\n[{i+1}/{len(targets)}] {name}  shape=({rows}, {in_feat})")

        W_orig_snapshot = layer.weight.detach().clone()

        collect_awq = args.awq != "none"
        H, n_tok, sum_abs = compute_hessian(layer, calib, model, args.device,
                                            collect_awq=collect_awq)
        t_hess = time.time() - t0
        print(f"  hessian: {H.shape}  n_tok={n_tok}  t={t_hess:.1f}s")

        # AWQ rescale
        awq_s = None
        awq_blob = None
        if collect_awq and sum_abs is not None:
            mean_abs = (sum_abs / max(n_tok, 1)).clamp_min(1e-8)
            awq_s = mean_abs.pow(args.awq_alpha).to(H.device)
            print(f"  awq: kind={args.awq} alpha={args.awq_alpha} "
                  f"s_max={awq_s.max().item():.3f} s_min={awq_s.min().item():.3f}")
            awq_blob = {"kind": args.awq, "alpha": args.awq_alpha, "s": awq_s.detach().cpu()}
            H = H * awq_s.unsqueeze(0) * awq_s.unsqueeze(1)
            W_new = apply_awq_to_weight(layer.weight.float().to(awq_s.device), awq_s)
            layer.weight_override = W_new.to(dtype)
        else:
            # No AWQ: still need to seed weight_override with original so subsequent
            # math operates on a Tensor (not a property page-fault each access).
            layer.weight_override = layer.weight.detach().clone()

        # Rotation
        rotation = None
        rotation_blob = None
        if args.rotation == "kronecker":
            a, b = factor_for_dim(in_feat, max_b=args.rotation_max_b)
            h_a = random_orthogonal(a, seed=args.rotation_seed + i)
            rotation = KroneckerRotation(h_a=h_a, b_dim=b)
            rotation_blob = rotation.to_dict()
            rotation_blob["seed"] = args.rotation_seed + i
            print(f"  rotation: kronecker a={a} b={b}")
            H = rotate_hessian(H, rotation)
            layer.weight_override = rotation.forward(
                layer.weight_override.float().to(H.device)).to(dtype)

        q = CentroidQuantizer(n_centroids=args.n_centroids, n_iter=args.n_iter).to(args.device)
        q.configure(bits=4, sym=True, fit_loss=args.fit_loss,
                    mag_weight_p=args.mag_weight_p,
                    snap_centroids=args.snap_centroids)
        q.hessian_diag = torch.diag(H).clone()

        effective_percdamp = args.percdamp
        if awq_s is not None:
            effective_percdamp = max(args.percdamp, 0.05)

        try:
            # gptq_quantize_linear modifies layer.weight in place — but layer is a
            # PagedLinear and .weight is a @property. Use a temporary nn.Linear shim
            # wrapping weight_override so gptq's `layer.weight.data.copy_(Q...)` works.
            shim = nn.Linear(in_feat, rows, bias=False).to(args.device)
            shim.weight = nn.Parameter(layer.weight_override.to(dtype))
            export = gptq_quantize_linear(shim, H, q,
                                          group_size=args.group_size,
                                          percdamp=effective_percdamp)
            # After GPTQ: shim.weight holds the dequantized rotated/AWQ'd weight.
            layer.weight_override = shim.weight.data.detach().clone()
            del shim
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            layer.weight_override = W_orig_snapshot
            continue
        t_quant = time.time() - t0 - t_hess

        # Inverse rotation, then absorb AWQ — leave weight_override at inference-equivalent.
        if rotation is not None:
            layer.weight_override = rotation.inverse(
                layer.weight_override.float().to(rotation.h_a.device)).to(dtype)
        if awq_s is not None:
            layer.weight_override = absorb_awq_in_reconstruction(
                layer.weight_override.float().to(awq_s.device), awq_s).to(dtype)

        # CRITICAL: weight_override must live on the model's device so the next
        # layer's forward pass (during compute_hessian) finds it on cuda. Rotation
        # math may have moved it to CPU (rotation.h_a is on CPU by design — see
        # KroneckerRotation.to_dict) — pull it back to args.device.
        layer.weight_override = layer.weight_override.to(args.device)

        # Save the blob (identical schema to calibrate_ml8.py output)
        out_path = Path(args.output_dir) / f"{name.replace('.', '_').replace('/', '_')}.pt"
        blob = {
            "name": name,
            "shape": [rows, in_feat],
            "group_size": args.group_size,
            "n_centroids": args.n_centroids,
            "indices": export["indices"].cpu(),
            "centroids_per_group": export["centroids_per_group"].cpu(),
            "scale_per_group": export["scale_per_group"].cpu(),
            "mse": export["mse"],
            "w_snr_db": export["w_snr_db"],
            "y_snr_db": export["y_snr_db"],
            "rel_err": export["rel_err"],
        }
        if rotation_blob is not None:
            blob["rotation"] = rotation_blob
        if awq_blob is not None:
            blob["awq"] = awq_blob
        torch.save(blob, out_path)

        print(f"  saved: {out_path.name}  "
              f"Y_SNR={export['y_snr_db']:.1f}dB  W_SNR={export['w_snr_db']:.1f}dB  "
              f"t_quant={t_quant:.1f}s")

        manifest["results"].append({
            "name": name, "shape": [rows, in_feat],
            "mse": float(export["mse"]),
            "y_snr_db": float(export["y_snr_db"]),
            "w_snr_db": float(export["w_snr_db"]),
            "t_hess_s": float(t_hess),
            "t_quant_s": float(t_quant),
        })

        del H, q
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.eval_ppl:
        print(f"\n[ppl-quant] computing quantized PPL...")
        quantized = eval_ppl_wikitext(model, tokenizer, args.device,
                                       max_tokens=args.ppl_max_tokens)
        print(f"  quantized PPL = {quantized['ppl']:.4f}")
        manifest["ppl_quantized"] = quantized
        delta = quantized["ppl"] - manifest["ppl_baseline"]["ppl"]
        manifest["ppl_delta"] = delta
        print(f"\n  Δ_PPL = {delta:+.4f}")

    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[done] manifest: {manifest_path}")

    pager.shutdown()


if __name__ == "__main__":
    main()
