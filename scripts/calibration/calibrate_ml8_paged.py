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
import queue
import re
import sys
import threading
import time
from pathlib import Path

print = functools.partial(print, flush=True)

# ── Deterministic calibration — env half (OPT-IN via ML8_DETERMINISTIC=1; MUST precede `import torch`) ───
# ml8 GPTQ calibration was nondeterministic: the GPU Hessian forward used unforced reduction order
# and GPTQ's sequential error-feedback amplified the low-bit noise into DIFFERENT weight assignments
# run-to-run (~0.6 PPL spread — swamps the ±0.05 levers). These flags pin it bit-identically.
# VERIFIED 2026-06-02: full-model deterministic calibration is bit-reproducible (187/188 tensors
# byte-identical across two independent runs; the rest are FP8). The earlier "breaks at the first
# self-attn layer" failure was NOT a determinism bug — use_deterministic_algorithms(True) enables
# fill_uninitialized_memory, which surfaced a latent bug: the rotary inv_freq buffer (no GGUF
# counterpart) was left uninitialized by meta+to_empty → NaN-filled → garbage RoPE. Fixed by
# reinit_rotary_buffers() (runs every calibration; also repairs the silent non-deterministic case).
_DETERMINISTIC = os.environ.get("ML8_DETERMINISTIC") == "1"
if _DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS/hipBLAS GEMM
    os.environ.setdefault("HIPBLASLT_DETERMINISTIC", "1")

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── Deterministic calibration — torch half (see env half above) ───
if _DETERMINISTIC:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for _attr in ("allow_tf32",):
        try: setattr(torch.backends.cudnn, _attr, False)
        except AttributeError: pass
    for _attr, _val in (("allow_tf32", False),
                        ("allow_fp16_reduced_precision_reduction", False),
                        ("allow_bf16_reduced_precision_reduction", False)):
        try: setattr(torch.backends.cuda.matmul, _attr, _val)
        except AttributeError: pass
    print("[determinism] OPT-IN ENABLED: use_deterministic_algorithms(True) + seeded + pinned GEMM "
          "— full-model calibration is bit-reproducible (rotary inv_freq reinit applied post-load)")

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
from calib_corpus import collect_calibration  # noqa: E402  (content sweep: named compositions)
from fla_compat import apply_fla_arch_shim, apply_fla_cpu_fallback  # noqa: E402  (RDNA fla bf16 fdot2 fp32 workaround; CPU torch-ref fallback)
from ml8_io import load_ml8_layer, reconstruct_weight_from_blob  # noqa: E402  (dense resume)
from role_targets import classify_role, Tier, assert_main_stack_covered, configure as configure_roles  # noqa: E402  (--dense-coverage full tier routing)
from faithful_forward import FaithfulActHook, assert_not_double_rotated, fp8_weight_override, collect_hessians_single_pass  # noqa: E402  (W4A8 faithful tiers)
from scaled_fp8 import quantize_scaled_fp8  # noqa: E402  (FP8 tier, fixed group_size=32)
from calib_timing import PhaseTimer  # noqa: E402  (MAD-256 Phase-1 instrumentation)

# Pybind11 weight pager (in repo's python_bindings/wp/). The .so is compiled for
# a specific GPU arch (gfx1201/gfx1030 locally) and links libllama/ggml-hip, so
# it will NOT load on a different arch (e.g. the MI300X's gfx942). The pager is
# only needed for the PAGED path; --resident runs pure torch and must import
# cleanly even where wp_native is absent. Guard the import and fail loudly only
# if someone actually asks for the paged path on a host that can't provide it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_bindings" / "wp"))
try:
    import wp_native  # noqa: E402
    from paged_linear import PagedLinear, swap_linears_with_paged, PagedMoeExperts  # noqa: E402
    _WP_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # ImportError, or HIP/arch load failure
    wp_native = None  # type: ignore[assignment]
    PagedLinear = swap_linears_with_paged = PagedMoeExperts = None  # type: ignore[assignment]
    _WP_IMPORT_ERROR = _e


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


# ─── Dense coverage target enumeration (--dense-coverage ffn|full) ──────────

# Deterministic append order for the NEW ML8 roles added by `full` coverage,
# AFTER the FFN linears. Documented + fixed so resume blobs stay valid and the
# converter can rely on a stable ordering. Per-layer roles are emitted layer by
# layer (ascending layer index); within a layer the suffixes follow this list.
# Per-layer ML8 suffixes — VERIFIED against Qwen3.5-9B/0.8B named_modules()
# (2026-05-31). Full-attention blocks: self_attn.{q,k,v,o}_proj. Gated delta-net
# blocks: linear_attn.{in_proj_qkv (fused qkv), in_proj_z (output gate), out_proj}.
# (Earlier values "qkv_proj"/"gate_proj_attn" never matched a real checkpoint and
# silently dropped the SSM input projections to bf16 — see assert_main_stack_covered.)
# Per-layer linear suffixes enumerated in `full` coverage, layer-major. This list
# fixes ENUMERATION ORDER only — the TIER of each role comes from
# role_targets.classify_role (overridable via ML8_TIER_OVERRIDE), NOT from which
# list a suffix sits in. So moving a role between tiers is a pure classify_role
# change and never silently drops a tensor. Self-attn q/k/v/o; gated delta-net
# in_proj_qkv (fused qkv) + in_proj_z (output gate) + out_proj (SSM out); ssm gates
# in_proj_a/in_proj_b. VERIFIED against Qwen3.5-9B/0.8B named_modules() (2026-05-31).
_FULL_PER_LAYER_ORDER = (
    "q_proj", "k_proj", "v_proj", "o_proj",   # self-attention
    "in_proj_qkv", "in_proj_z",               # linear_attn fused qkv + output gate
    "out_proj",                               # linear_attn (SSM) output
    "in_proj_a", "in_proj_b",                 # linear_attn (SSM) gates alpha/beta
)
# Model-global roles (no layer index), enumerated after all per-layer linears.
_FULL_GLOBAL_ORDER = ("lm_head", "eh_proj")


def _dense_layer_idx(name: str):
    parts = name.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None


def find_dense_full_targets(model, coverage: str = "ffn"):
    """Enumerate quantization targets for the dense path, tier-tagged.

    Yields (name, module, tier) where tier is a role_targets.Tier.

    coverage="ffn"  : EXACTLY the FFN linears yielded by find_target_linears,
                      in the same order, all tagged Tier.ML8. Provably identical
                      to today's behavior (assert in test_dense_coverage.py).
    coverage="full" : the FFN linears FIRST (same order), then per-layer
                      attention/SSM linears (layer-major, _FULL_PER_LAYER_ORDER),
                      then global roles (lm_head/eh_proj), then the embedding —
                      EACH routed to ML8/FP8/NATIVE by role_targets.classify_role
                      (overridable via ML8_TIER_OVERRIDE). All ML8 targets are
                      emitted before all FP8 targets, embedding always last.

    Tier comes from classify_role, never from which enumeration list a suffix sits
    in — so moving a role between tiers (ML8_TIER_OVERRIDE) re-buckets it cleanly
    and can never silently drop a tensor. With the default tier map this yields the
    exact historical order, so existing resume-blob prefixes remain valid.
    """
    # FFN linears first — identical to find_target_linears (order preserved).
    ffn = list(find_target_linears(model))

    if coverage == "ffn":
        for name, mod in ffn:
            yield name, mod, Tier.ML8
        return
    if coverage != "full":
        raise ValueError(f"unknown dense coverage {coverage!r} (expected 'ffn' or 'full')")

    # Index every module once for deterministic lookups.
    by_name = dict(model.named_modules())
    n_layers = max((_dense_layer_idx(n) for n in by_name if _dense_layer_idx(n) is not None),
                   default=-1) + 1

    # Skip lm_head when the model TIES embeddings: a tied lm_head IS token_embd
    # (served at inference by the tied token_embd tier), has no independent GGUF
    # tensor (the converter emits no output.weight), and meta+to_empty leaves its
    # weight UNINITIALIZED — quantizing it wastes time on silent garbage.
    _cfg = getattr(model, "config", None)
    _tied = bool(getattr(getattr(_cfg, "text_config", _cfg), "tie_word_embeddings", False)
                 or getattr(_cfg, "tie_word_embeddings", False))

    # Build the ordered candidate list (name, mod): FFN, then per-layer linears
    # (layer-major), then global roles. The embedding is collected separately so it
    # can always be emitted last.
    candidates = list(ffn)
    for L in range(n_layers):
        for suffix in _FULL_PER_LAYER_ORDER:
            for name, mod in by_name.items():
                if (_dense_layer_idx(name) == L
                        and name.rsplit(".", 1)[-1] == suffix
                        and isinstance(mod, nn.Linear)):
                    candidates.append((name, mod))
    for role in _FULL_GLOBAL_ORDER:
        if role == "lm_head" and _tied:
            print("[targets] skipping lm_head target: tie_word_embeddings=True "
                  "(served by the tied token_embd at inference)")
            continue
        for name, mod in by_name.items():
            if name.rsplit(".", 1)[-1] == role and isinstance(mod, nn.Linear):
                candidates.append((name, mod))

    # Route each candidate by its classify_role tier, ML8 bucket then FP8 bucket.
    # De-dup by name (FFN linears are also reachable by suffix) preserving order.
    seen, ml8, fp8 = set(), [], []
    for name, mod in candidates:
        if name in seen:
            continue
        seen.add(name)
        _, _, tier = classify_role(name)
        if tier is Tier.ML8:
            ml8.append((name, mod))
        elif tier is Tier.FP8:
            fp8.append((name, mod))
        # NATIVE: not a quant target (shouldn't occur for these Linears).
    for name, mod in ml8:
        yield name, mod, Tier.ML8
    for name, mod in fp8:
        yield name, mod, Tier.FP8

    # Embedding (nn.Embedding) — emitted LAST, routed by token_embd's tier.
    for name, mod in by_name.items():
        if name == "model.embed_tokens" and isinstance(mod, nn.Embedding):
            _, _, tier = classify_role(name)
            if tier is not Tier.NATIVE:
                yield name, mod, tier


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

    # Hand-rolled fallbacks for param paths the TensorNameMap doesn't carry
    # in this transformers/gguf-py version. Qwen3.5 linear_attn has a 1-D
    # `dt_bias` Parameter (Mamba-3 SSM time-step bias) whose HF stem doesn't
    # match the SSM_DT key out of the box; map it manually.
    _LINEAR_ATTN_DT_BIAS = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.dt_bias$")

    def lookup(param_path: str) -> str | None:
        m = _LINEAR_ATTN_DT_BIAS.match(param_path)
        if m:
            return f"blk.{m.group(1)}.ssm_dt.bias"
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


def reinit_rotary_buffers(model: nn.Module, device: str) -> int:
    """Recompute every text rotary-embedding ``inv_freq`` buffer from config.

    inv_freq is a NON-PERSISTENT buffer derived from rope_theta — it has no GGUF
    counterpart, so the resident loader (which walks ``named_parameters``) never
    fills it, and ``model.to_empty()`` leaves it UNINITIALIZED. The damage is
    silent without determinism: uninitialized memory is finite garbage, and since
    cos()/sin() clamp to [-1, 1] the forward never NaNs — it just applies WRONG
    positional encoding to every full-attention layer (linear-attn layers don't
    use RoPE). With ``use_deterministic_algorithms(True)`` that same uninitialized
    memory is NaN-filled, so the first RoPE turns q/k into NaN and the whole
    forward (and every downstream Hessian) blows up. Both are the SAME bug; the
    fix is to rebuild inv_freq the way the module's __init__ does and keep it in
    fp32 (HF never downcasts it; our blanket .to(dtype) did). Idempotent — safe to
    run on every calibration. Returns the number of rotary modules repaired.
    """
    n = 0
    for mod in model.modules():
        cls = mod.__class__.__name__
        if "RotaryEmbedding" not in cls or "Vision" in cls:
            continue   # text rotary only; vision rotary has a different __init__
        if not hasattr(mod, "inv_freq") or getattr(mod, "config", None) is None:
            continue
        try:
            fresh = type(mod)(mod.config, device=torch.device(device))
        except Exception as e:   # noqa: BLE001 — never let a probe-fix abort calibration
            print(f"[rotary-fix] WARN: could not rebuild {cls} inv_freq: {e}")
            continue
        with torch.no_grad():
            # assign (not copy_) so the buffer keeps fresh's fp32 dtype, not the
            # bf16 the model-wide .to(dtype) left it as.
            mod.inv_freq = fresh.inv_freq.to(device=torch.device(device))
            if hasattr(mod, "original_inv_freq") and hasattr(fresh, "original_inv_freq"):
                mod.original_inv_freq = fresh.original_inv_freq.to(device=torch.device(device))
        if hasattr(fresh, "attention_scaling"):
            mod.attention_scaling = fresh.attention_scaling
        n += 1
    return n


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

def dense_completed_prefix(names: "list[str]", output_dir) -> int:
    """Count the leading dense linears (in quantization order) whose blob .pt
    already exists, STOPPING AT THE FIRST GAP.

    Dense calibration propagates quantization error across layers, so a blob that
    exists *after* a gap was computed against a different (un-resumed) upstream
    state and is stale. Only a contiguous completed prefix is safe to resume from.
    Returns the count of leading names with a blob present (0 if the first is
    missing); blobs after the first gap are ignored.
    """
    out = Path(output_dir)
    n = 0
    for name in names:
        blob_path = out / f"{name.replace('.', '_').replace('/', '_')}.pt"
        if not blob_path.exists():
            break
        n += 1
    return n


def _get_module_by_name(model, name: str):
    """Resolve a dotted module path (e.g. 'model.layers.0.mlp.gate_proj') to the
    module object. Also handles flat names ('linear0')."""
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


def load_dense_prefix_into_model(prefix_count: int, target_names: "list[str]",
                                 model, output_dir, resident: bool,
                                 dtype=None, device=None) -> "list[dict]":
    """Reload the first `prefix_count` completed dense linears' quantized weights
    back into the model (the dense-resume reload). For each, dequant the blob to its
    inference-equivalent weight and place it where the forward path reads it:
    resident → layer.weight.data; paged → layer.weight_override. This restores the
    exact upstream state later layers' Hessians depend on. Returns the per-linear
    metric dicts (for the manifest), in order. prefix_count=0 is a no-op.

    `device` (paged path only): blobs load via torch.load(map_location="cpu"), so the
    reconstructed weight is on CPU. The paged forward runs on the calibration device,
    so the override MUST be moved there — otherwise the NEXT layer's Hessian forward
    pushes device activations through a CPU weight_override and F.linear raises
    "mat2 is on cpu, different from cuda:0". Pass the calib device (args.device).
    The resident path already moves to layer.weight.device, so it ignores `device`.
    """
    results: "list[dict]" = []
    for name in target_names[:prefix_count]:
        layer = _get_module_by_name(model, name)
        blob_path = Path(output_dir) / f"{name.replace('.', '_').replace('/', '_')}.pt"
        blob = load_ml8_layer(blob_path)
        W = reconstruct_weight_from_blob(blob)          # dequant → inv-rotation → absorb-AWQ
        if resident:
            with torch.no_grad():
                layer.weight.data.copy_(
                    W.to(dtype=layer.weight.dtype, device=layer.weight.device))
        else:
            # Move to the calib device: W is on CPU (blob loaded map_location="cpu"),
            # but the paged forward runs on `device`. .to(dtype=None, ...) keeps dtype;
            # .to(device=None) keeps device — so this is a no-op when both are None,
            # preserving the prior signature's behavior for callers that omit them.
            layer.weight_override = W.to(dtype=dtype, device=device)
        results.append({
            "name": name, "shape": list(blob["shape"]),
            "mse": float(blob.get("mse", 0.0)),
            "y_snr_db": float(blob.get("y_snr_db", 0.0)),
            "w_snr_db": float(blob.get("w_snr_db", 0.0)),
            "t_hess_s": 0.0, "t_quant_s": 0.0,
        })
        del blob, W
    return results


def quantize_one_target(name, layer, target_index, H, n_tok, sum_abs,
                        rotation_hook, args, dtype, manifest, out_dir, timer, recipe=None):
    """Quantize ONE dense ml8 target end-to-end: AWQ -> rotation -> batched_gptq ->
    inverse/absorb -> writeback (resident propagation) -> save blob + manifest.
    H/n_tok/sum_abs are INPUTS. recipe overrides group_size/n_centroids per-role;
    None = global args. Bit-identical to the inline per-target loop body it replaces."""
    import time
    t0 = time.time()
    W_orig_snapshot = layer.weight.detach().clone()
    rows, in_feat = layer.weight.shape
    collect_awq = args.awq != "none"
    gs = (recipe or {}).get("group_size", None)
    nc = (recipe or {}).get("n_centroids", None) or args.n_centroids

    # Per-kind group size (down_proj may use a finer grid via --group-size-down).
    kind = name.rsplit(".", 1)[-1]   # gate_proj / up_proj / down_proj
    gs_for_kind = gs if gs is not None else args.group_size
    if kind == "down_proj" and args.group_size_down is not None and gs is None:
        gs_for_kind = args.group_size_down

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
        if args.faithful_acts:
            # SAME Q the activation hook used (rotation already baked into H).
            rotation = rotation_hook
        else:
            a, b = factor_for_dim(in_feat, max_b=args.rotation_max_b)
            h_a = random_orthogonal(a, seed=args.rotation_seed + target_index)
            rotation = KroneckerRotation(h_a=h_a, b_dim=b)
        rotation_blob = rotation.to_dict()
        rotation_blob["seed"] = args.rotation_seed + target_index
        print(f"  rotation: kronecker (faithful={args.faithful_acts})")
        if not args.faithful_acts:
            H = rotate_hessian(H, rotation)
        assert_not_double_rotated(args.faithful_acts, rotate_hessian_called=False)
        layer.weight_override = rotation.forward(
            layer.weight_override.float().to(H.device)).to(dtype)

    effective_percdamp = args.percdamp
    if awq_s is not None:
        effective_percdamp = max(args.percdamp, 0.05)

    try:
        # Unified quantizer: route the single dense linear through
        # batched_gptq_quantize as a [1, N, K] stack so it gets the SAME
        # levers as the MoE path — including act_order + the heavy tune loop
        # (bit-free). weight_override is already rotated/AWQ'd; H matches it.
        Wr = layer.weight_override.float().to(args.device).unsqueeze(0)   # [1, N, K]
        Hr = H.to(args.device).unsqueeze(0)                              # [1, K, K]
        with timer.phase("gptq_quantize", target=name):
            out = batched_gptq_quantize(
                W_stack=Wr, H_stack=Hr,
                n_centroids=nc, group_size=gs_for_kind,
                n_iter=args.n_iter, fit_loss=args.fit_loss,
                mag_weight_p=args.mag_weight_p,
                snap_centroids=args.snap_centroids,
                percdamp=effective_percdamp,
                act_order=args.act_order or args.heavy_rounds > 0,
                heavy_rounds=args.heavy_rounds, heavy_steps=args.heavy_steps,
                heavy_dtype=args.heavy_dtype,
                heavy_lr_cent=args.heavy_lr_cent,
                heavy_lr_scale=args.heavy_lr_scale)
        layer.weight_override = out["Q"][0].to(dtype)   # dequantized rotated/AWQ'd weight
        export = {
            "indices":             out["indices"][0].clone().contiguous(),
            "centroids_per_group": out["centroids_per_group"][0].clone().contiguous(),
            "scale_per_group":     out["scale_per_group"][0].clone().contiguous(),
            "mse":      float(out["mse"][0].item()),
            "w_snr_db": float(out["w_snr_db"][0].item()),
            "y_snr_db": float(out["y_snr_db"][0].item()),
            "rel_err":  float(out["rel_err"][0].item()),
        }
        del Wr, Hr, out
    except RuntimeError as e:
        print(f"  FAILED: {e}")
        layer.weight_override = W_orig_snapshot
        if args.resident:
            layer.weight.data.copy_(W_orig_snapshot.to(layer.weight.dtype))
            layer.weight_override = None   # same resident-leak fix as success path
        return
    t_quant = time.time() - t0

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

    # Resident path: the model forward reads layer.weight (a real Parameter),
    # NOT weight_override — so copy the calibrated weight back so the NEXT
    # layer's Hessian sees the quantized upstream (GPTQ cross-layer error
    # propagation). PagedLinear's forward reads weight_override directly, so
    # this branch is a no-op there.
    if args.resident:
        with torch.no_grad():
            layer.weight.data.copy_(layer.weight_override.to(layer.weight.dtype))
        # Resident forward reads layer.weight, NOT weight_override — so the
        # override is a full-size GPU duplicate of this layer's weights once
        # it's been copied back. Drop it or it accumulates ~one MLP-worth of
        # VRAM per layer (the resident-mode leak fixed 2026-05-30).
        layer.weight_override = None

    # Save the blob (identical schema to calibrate_ml8.py output)
    out_path = out_dir / f"{name.replace('.', '_').replace('/', '_')}.pt"
    blob = {
        "name": name,
        "shape": [rows, in_feat],
        "group_size": gs_for_kind,
        "n_centroids": nc,
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
        "t_hess_s": 0.0,
        "t_quant_s": float(t_quant),
    })
    return  # side effects only: weight writeback + saved .pt + manifest append


class _BlockHessianHook:
    """Adapter that lets run_walk drive the ALREADY-INSTALLED, always-on
    FaithfulActHook on an ml8 leaf. install() locates that hook (does NOT register
    a new one — that would double-transform); remove() is a no-op (the shared hook
    stays active so the faithful e4m3 transform applies during propagation too)."""
    def __init__(self):
        self._fa = None
        self.rotation = None
    def install(self, block, leaf):
        mod = dict(block.named_modules())[leaf]
        fa = next((h for h in mod._forward_pre_hooks.values()
                   if isinstance(h, FaithfulActHook)), None)
        if fa is None:
            raise RuntimeError(f"block-sequential: no FaithfulActHook installed on '{leaf}' "
                               f"(faithful-acts hooks must be installed before the walk)")
        self._fa = fa
        self.rotation = fa.rotation
    def set_hessian_target(self, on): self._fa.set_hessian_target(on)
    def reset_hessian(self): self._fa.reset_hessian()
    @property
    def H(self): return self._fa.H
    @property
    def n_tokens(self): return self._fa.n_tokens
    def remove(self):
        pass   # shared always-on hook; do not deregister


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
    p.add_argument("--corpus", default="wiki",
                   help="calibration corpus composition: 'wiki' (wikitext-2 control) or a "
                        "named mix from calib_corpus.COMPOSITIONS (mix|code|math|chat). "
                        "The content lever — defines the Hessian everything descends.")
    p.add_argument("--corpus-seed", type=int, default=0,
                   help="RNG seed for byte-offset sampling + shuffle of mixed corpora")
    p.add_argument("--token-budget", type=int, default=None,
                   help="if set, draw every corpus to this many total tokens (wiki trimmed, "
                        "mixes drawn by token-share) instead of using --n-samples. The "
                        "token-matched control for the content sweep — equalizes the Hessian "
                        "sample size across corpora regardless of per-doc length.")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--act-order", action="store_true",
                   help="GPTQ act_order: extra Hessian-importance-ordered reassignment "
                        "pass after the straight sweep (MAD-256: +~1dB gate/up, bit-free)")
    p.add_argument("--heavy-rounds", type=int, default=0,
                   help="MAD-256 heavy codebook: alternating gradient-tune↔act_order-reassign "
                        "rounds (AQLM/PV-tuning). 0=off. Implies act_order. Bit-free.")
    p.add_argument("--heavy-steps", type=int, default=60,
                   help="Adam steps per heavy round")
    p.add_argument("--heavy-dtype", choices=("fp32", "bf16"), default="fp32",
                   help="heavy tune-loop matmul precision. bf16 ~2x faster on WMMA "
                        "(fp32 accumulate; params/Adam stay fp32). Validate Y_SNR "
                        "matches fp32 within noise before trusting.")
    p.add_argument("--heavy-lr-cent", type=float, default=1e-2,
                   help="Adam LR for the continuous centroids in the heavy tune loop "
                        "(was hardcoded 1e-2; exposed so it can be swept — LR is the "
                        "dominant un-tuned knob of the heavy method).")
    p.add_argument("--heavy-lr-scale", type=float, default=1e-3,
                   help="Adam LR for the per-group scales in the heavy tune loop "
                        "(was hardcoded 1e-3; exposed for sweeping).")
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--n-iter", type=int, default=25)
    p.add_argument("--fit-loss", choices=("mse", "mag_weighted"), default="mse")
    p.add_argument("--mag-weight-p", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--capture-router", default=None,
                   help="DIAGNOSTIC: comma-separated layer indices. Hook each layer's "
                        "router (.mlp.gate) Linear, capture its input x (MoE hidden) + "
                        "output logits over the calib corpus, save to --capture-out, then "
                        "exit before any Hessian/GPTQ work. For shared-vs-per-expert "
                        "Hessian analysis (MoE quant-gap investigation).")
    p.add_argument("--capture-out", default=None,
                   help="output .pt path for --capture-router")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--eval-ppl", action="store_true")
    p.add_argument("--ppl-max-tokens", type=int, default=None)
    p.add_argument("--rotation", choices=("none", "kronecker"), default="none")
    p.add_argument("--faithful-acts", action="store_true",
                   help="W4A8: collect Hessians on rotated, per-row e4m3-quantized "
                        "activations and propagate them (drops algebraic rotate_hessian). "
                        "Requires --rotation kronecker.")
    p.add_argument("--faithful-weights", action="store_true",
                   help="W4A8: simulate the fp8 weight tiers (token_embd, ssm alpha/beta) "
                        "via scaled-FP8 quant->dequant overrides during the calib forward. "
                        "No-op unless --dense-coverage full populates the FP8 tier.")
    p.add_argument("--rotation-seed", type=int, default=42)
    p.add_argument("--rotation-max-b", type=int, default=1024)
    p.add_argument("--snap-centroids", choices=("none", "e4m3"), default="none")
    p.add_argument("--awq", choices=("none", "mean"), default="none")
    p.add_argument("--awq-alpha", type=float, default=0.5)
    p.add_argument("--pager-slots", type=int, default=8,
                   help="WeightPager VRAM ring size. Pool = slots × max_page_size. "
                        "For Qwen3.5-4B, max_page_size ≈ 1.27 GB (token_embd) → "
                        "8 slots = ~10 GB pool. Don't exceed available headroom.")
    p.add_argument("--resident", action="store_true",
                   help="dense strategy only: load ALL weights resident (no pager, "
                        "no PagedLinear) and quantize in-place. Use when the model "
                        "fits in VRAM (9B/4B on 32GB; anything on the 192GB MI300X). "
                        "Eliminates the per-linear paged Hessian thrash AND the "
                        "pager-pool VRAM. REQUIRED on non-RDNA GPUs (the HIP pager "
                        "won't load on gfx942).")
    p.add_argument("--dense-coverage", choices=("ffn", "full"), default="ffn",
                   help="ffn (default) = quantize only FFN linears (today's behavior, "
                        "bit-identical). full = also quantize attention/SSM ML8 linears "
                        "and emit FP8-tier blobs (alpha/beta proj + embed_tokens). The "
                        "FP8 tensors await converter support — see manifest dense_coverage marker.")
    p.add_argument("--strategy", choices=("dense", "moe"), default="dense",
                   help="dense (iter 5a) = from_pretrained, all weights on GPU, "
                        "paged dense MLPs. moe (iter 5b) = torch.device('meta') "
                        "instantiate, load resident from GGUF directly, page "
                        "consolidated MoE experts. Use 'moe' for any model that "
                        "doesn't fit in host RAM (e.g. Qwen3.6 35B-A3B).")
    p.add_argument("--phase-timing", action="store_true",
                   help="Phase-1 instrumentation: accumulate wall time per phase "
                        "(corpus/hessian_forward/gptq_quantize) and write "
                        "phase_timing.json into --output-dir. No effect on results.")
    p.add_argument("--forward-dtype-probe", type=int, default=0, metavar="K",
                   help="Phase-1: before the main loop, time K calib samples through "
                        "model() with allow_tf32 False (current/deterministic) vs True, "
                        "to isolate the fp32-vs-WMMA matmul tax. 0 = off.")
    p.add_argument("--hessian-mode", choices=("single", "per-target", "block-sequential"), default="single",
                   help="Dense Hessian collection. 'single' (default): ONE forward "
                        "populates all target Hessians from the ORIGINAL model "
                        "(static-Hessian GPTQ; requires --faithful-acts and --awq "
                        "none); ~Nx faster. NOT bit-identical to 'per-target', which "
                        "is true-sequential (each target's H sees quantized upstream "
                        "via weight_override write-back = GPTQ cross-layer error "
                        "propagation); validate 'single' by PPL, not byte-diff. "
                        "'per-target': the exact true-sequential reference path.")
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
    p.add_argument("--task-e", type=int, default=128,
                   help="Expert-axis chunk size for task-queue MoE calibration. "
                        "Smaller = more tasks, more parallelism, less VRAM per task.")
    p.add_argument("--workers-primary", type=int, default=2,
                   help="Number of concurrent workers on --device.")
    p.add_argument("--workers-secondary", type=int, default=1,
                   help="Number of concurrent workers on --secondary-device.")
    p.add_argument("--group-size-down", type=int, default=None,
                   help="Override --group-size for down_proj kind only (post-SwiGLU "
                        "activations have heavier dynamic range, may need finer "
                        "grouping). If unset, uses --group-size for all kinds.")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable checkpoint resume (covers BOTH dense and MoE "
                        "strategies). By default the script will: (a) load cached "
                        "Hessians from {output_dir}/hessians.pt if params match, "
                        "(b) for MoE, skip per-expert blobs that already exist on "
                        "disk, (c) for dense, reload the contiguous completed-prefix "
                        "blobs back into the model (so cross-layer error propagation "
                        "stays correct) and resume at the first gap, (d) reuse prior "
                        "manifest results. --no-resume forces a clean run from scratch.")
    args = p.parse_args()
    TIMER = PhaseTimer()  # accumulates only when --phase-timing; cheap regardless

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0

    # Hard VRAM guard: cap torch to a fraction of the device so an over-budget
    # allocation raises a *catchable* RuntimeError (the per-layer loop falls the
    # layer back and continues) instead of a driver-level OOM. Critical when the
    # device also drives the display — a real OOM on the R9700 takes down the
    # Wayland compositor (Hyprland) and the whole session, which resume can't undo.
    _mem_frac = os.environ.get("ML8_MEM_FRACTION")
    if _mem_frac and args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(_mem_frac), device_idx)
        print(f"[mem-guard] torch capped to {float(_mem_frac):.0%} of cuda:{device_idx} "
              f"VRAM — OOM becomes a caught fallback, protecting the display compositor")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.strategy == "dense":
        # ── Iter 5a (PAGED): meta-init + paged dense MLPs, mirroring the MoE path.
        # The old from_pretrained→.to(device) path materialized the FULL model on
        # the GPU before swapping — fine for 4B (8 GB) but impossible for a >VRAM
        # dense model (27B = 54 GB on a 32 GB card). Now: build on meta, page the
        # ffn_{gate,up,down} linears, to_empty only the resident params
        # (attn/ssm/embed/norms), resident-load those from the GGUF.
        print(f"[load-hf] strategy=dense ({'resident' if args.resident else 'paged'})  {args.model}  dtype={dtype}  device={args.device}")
        model, config, n_blocks = _build_model_meta(args.model, dtype)
        arch_name = args.arch or type(config).__name__.replace("Config", "").lower()
        print(f"[meta-init] arch={arch_name}  n_blocks={n_blocks}")

        if args.resident:
            # ── RESIDENT (no pager): leave MLP linears as nn.Linear so the GGUF
            # loader fills them too (load_resident_to_model skips only MoE `_exps`
            # stacks). to_empty below materializes ALL params; the GGUF load fills
            # MLP + the rest. For models that FIT in VRAM (9B/4B @ 32GB; anything
            # on the 192GB MI300X). Pager path remains for >VRAM dense (27B/35B).
            pager = None
        else:
            if wp_native is None:
                raise RuntimeError(
                    "paged dense path requires the wp_native pager, which failed to "
                    f"import ({_WP_IMPORT_ERROR}). On a host whose GPU arch the .so "
                    "wasn't built for (e.g. MI300X/gfx942), use --resident instead.")
            print(f"[pager] building dense MLP pager from {args.gguf}  slots={args.pager_slots}")
            pager = build_pager_from_gguf(args.gguf, device_idx, args.pager_slots,
                                           name_filter=_mlp_only_name_filter)
            print(f"[pager] catalog: {pager.n_pages()} tensors  max_page={pager.max_page_size()/1e6:.1f} MB")

            name_map = _qwen_mlp_name_map(model)
            print(f"[swap] HF↔GGUF MLP mapping: {len(name_map)} linears")
            # Swap UNDER meta so PagedLinear's weight allocates on meta (zero real
            # memory) and is immediately del'd — same trick the MoE expert swap relies on.
            with torch.device("meta"):
                n_swapped = swap_linears_with_paged(model, pager, name_map, dtype=dtype,
                                                    device_idx=device_idx)
            print(f"[swap] replaced {n_swapped} nn.Linear → PagedLinear")
            if n_swapped == 0:
                raise RuntimeError("swap_linears_with_paged found nothing to swap — check name_map")

        n_resident = sum(p.numel() for p in model.parameters())
        print(f"[meta-empty] materializing {n_resident/1e9:.2f}B resident params on {args.device}")
        model = model.to_empty(device=args.device).to(dtype).eval()
        n_loaded = load_resident_to_model(
            model, args.gguf, arch_name=arch_name, n_blocks=n_blocks,
            dtype=dtype, device=args.device)
        if n_loaded == 0:
            raise RuntimeError(
                f"[resident-load] loaded 0 tensors from GGUF — arch_name mismatch? "
                f"Tried arch={arch_name!r}. Use --arch to override.")
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
        if args.resident:
            # ── RESIDENT MoE (no pager): load the consolidated expert stacks
            # straight into VRAM. The 35B-A3B's ~67 GB of experts fit the MI300X's
            # 192 GB. Pure torch + gguf (no wp_native) → runs on gfx942. Loaded
            # BEFORE to_empty; ResidentMoeExperts holds them as plain attributes
            # that to_empty leaves untouched.
            from resident_moe import swap_moe_experts_resident
            pager = None
            print(f"[resident-moe] loading expert stacks from {args.gguf}  n_experts={n_experts}")
            n_swapped_moe = swap_moe_experts_resident(model, args.gguf, dtype, args.device)
            print(f"[swap] replaced {n_swapped_moe} consolidated MoE blocks → ResidentMoeExperts")
        else:
            if wp_native is None:
                raise RuntimeError(
                    "paged MoE path requires the wp_native pager, which failed to import "
                    f"({_WP_IMPORT_ERROR}). On a host whose GPU arch the .so wasn't built "
                    "for (e.g. MI300X/gfx942), use --resident instead.")
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
            raise RuntimeError("found 0 MoE expert blocks to swap")

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
            if pager is not None: pager.shutdown()
            return

    # ─── fla arch shim: on RDNA the gated-delta-rule Triton kernel can only run
    # in fp32 (bf16 fdot2 doesn't lower → core dump); fp32 also matches the
    # deployed f32 recurrence core. No-op on CDNA3/NVIDIA/CPU or if fla isn't
    # installed. MUST run before the first forward. ───
    apply_fla_arch_shim(model, args.device)
    # ─── fla CPU fallback: when fla IS installed and device=cpu, the Triton
    # kernel is still bound (chunk_gated_delta_rule or torch_… → fla wins when
    # fla is present). Triton cannot dispatch CPU tensors → ValueError. Swap the
    # Triton binding back to the HF torch reference so CPU calibration works.
    # No-op on GPU paths. MUST run before the first forward. ───
    apply_fla_cpu_fallback(model, args.device)

    # ─── Repair rotary inv_freq: meta-build + to_empty leaves this non-persistent,
    # non-GGUF buffer uninitialized → silently-wrong RoPE (and a hard NaN under
    # determinism). MUST run before the first forward, on every calibration. ───
    _n_rotary = reinit_rotary_buffers(model, args.device)
    print(f"[rotary-fix] reinitialized inv_freq on {_n_rotary} text rotary module(s)")

    # ─── NaN probe (opt-in ML8_NAN_PROBE=1): hook every decoder layer + the submodules
    # of the first full_attention layer; on the FIRST forward, print per-module
    # in-nan/out-nan/out-inf/finite-absmax in completion order, then self-remove. Used to
    # pinpoint the determinism-induced NaN in the first full-attn layer (deferred bug). ───
    if os.environ.get("ML8_NAN_PROBE") == "1":
        _dec = [m for _n, m in model.named_modules()
                if m.__class__.__name__.endswith("DecoderLayer")]
        _lt = list(getattr(getattr(model.config, "text_config", model.config),
                           "layer_types", ["?"] * len(_dec)))
        _first_full = next((i for i, t in enumerate(_lt) if "full" in str(t)), 3)
        _pb = {"ev": [], "done": False, "h": []}

        def _pstat(t):
            if isinstance(t, (tuple, list)) and t and torch.is_tensor(t[0]):
                t = t[0]
            if not torch.is_tensor(t):
                return (None, None, float("nan"))
            tf = t.float()
            fin = tf[torch.isfinite(tf)]
            return (int(torch.isnan(tf).sum()), int(torch.isinf(tf).sum()),
                    float(fin.abs().max()) if fin.numel() else float("nan"))

        def _mk(label):
            def _h(mod, inp, out):
                if _pb["done"]:
                    return
                it = inp[0] if isinstance(inp, (tuple, list)) and inp else inp
                inn = int(torch.isnan(it.float()).sum()) if torch.is_tensor(it) else -1
                nan, inf, amax = _pstat(out)
                _pb["ev"].append((label, inn, nan, inf, amax))
            return _h

        for i, m in enumerate(_dec):
            _pb["h"].append(m.register_forward_hook(
                _mk(f"LAYER[{i:02d}] ({_lt[i] if i < len(_lt) else '?'})")))
            if i == _first_full:
                for sn, sm in m.named_modules():
                    if sn:
                        _pb["h"].append(sm.register_forward_hook(
                            _mk(f"  L{i}.{sn} <{sm.__class__.__name__}>")))

        def _dump(mod, inp, out):
            if _pb["done"]:
                return
            _pb["done"] = True
            print("\n=== [NAN-PROBE] first forward, completion order ===")
            fb = None
            for label, inn, nan, inf, amax in _pb["ev"]:
                bad = (nan and nan > 0) or (inf and inf > 0)
                mark = "  <<< FIRST NaN/Inf" if bad and fb is None else ""
                if bad and fb is None:
                    fb = label
                print(f"{label:<50} in_nan={inn} out_nan={nan} out_inf={inf} "
                      f"absmax={amax:.4g}{mark}")
            print(f"[NAN-PROBE] first NaN/Inf at: {fb}\n", flush=True)
            for h in _pb["h"]:
                h.remove()
        _dec[-1].register_forward_hook(_dump)
        print(f"[NAN-PROBE] installed: {len(_dec)} decoder layers + submodules of "
              f"layer {_first_full} (first full_attention)")

    # ─── Calibration corpus + baseline PPL (paged forward proves the swap works) ───
    print(f"[calib] loading {'budget ' + str(args.token_budget) + ' tok' if args.token_budget else str(args.n_samples) + ' samples'} "
          f"seq_len={args.seq_len} corpus={args.corpus}")
    with TIMER.phase("corpus_load"):
        calib = collect_calibration(tokenizer, n_samples=args.n_samples,
                                     seq_len=args.seq_len, composition=args.corpus,
                                     seed=args.corpus_seed, token_budget=args.token_budget)
    print(f"[calib] got {len(calib)} samples (tokens ≈ {sum(c.numel() for c in calib)})")

    if args.forward_dtype_probe > 0:
        import torch.backends.cuda as _bcuda
        import torch.backends.cudnn as _bcudnn
        K = min(args.forward_dtype_probe, len(calib))
        probe = calib[:K]

        def _time_forward(tag):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for ids in probe:
                    model(ids.to(args.device))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"[dtype-probe] {tag:18s} {dt:7.2f}s for {K} samples "
                  f"({dt/K*1000:7.1f} ms/sample)")
            return dt

        # Untimed warmup: absorb lazy CUDA init / cuBLAS handles / kernel
        # autotune so the FIRST timed leg (fp32) isn't biased slow → clean ratio.
        with torch.no_grad():
            model(probe[0].to(args.device))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        _saved = (_bcuda.matmul.allow_tf32, _bcudnn.allow_tf32)
        _bcuda.matmul.allow_tf32 = False
        _bcudnn.allow_tf32 = False
        dt_fp32 = _time_forward("allow_tf32=False")
        _bcuda.matmul.allow_tf32 = True
        _bcudnn.allow_tf32 = True
        dt_tf32 = _time_forward("allow_tf32=True")
        _bcuda.matmul.allow_tf32, _bcudnn.allow_tf32 = _saved
        print(f"[dtype-probe] fp32/tf32 forward ratio = {dt_fp32/max(dt_tf32,1e-9):.2f}x "
              f"(matmul-precision tax on this card)")
        _pp = Path(args.output_dir); _pp.mkdir(parents=True, exist_ok=True)
        (_pp / "dtype_probe.json").write_text(json.dumps(
            {"k_samples": K, "fp32_s": dt_fp32, "tf32_s": dt_tf32,
             "ratio": dt_fp32 / max(dt_tf32, 1e-9)}, indent=2))

    # Tier-tagged enumeration. For the DENSE strategy with --dense-coverage full
    # this appends attention/SSM ML8 linears + FP8-tier tensors after the FFN
    # linears (which always come first, in find_target_linears order). For
    # coverage=ffn (default) OR the MoE strategy, this is exactly today's FFN set.
    _coverage = args.dense_coverage if args.strategy == "dense" else "ffn"
    # Build the authoritative HF->GGUF name map for tier classification (delegates
    # name resolution to llama.cpp's TensorNameMap instead of a hand-written table).
    configure_roles(arch_name, n_blocks)
    if _coverage == "full":
        # Source-of-truth guard: refuse to calibrate if any main-stack nn.Linear
        # (attention / SSM / MLP) would be silently left NATIVE/bf16 because the
        # checkpoint's module names aren't resolved by the arch's TensorNameMap.
        n_cov = assert_main_stack_covered(model)
        print(f"[role-guard] {n_cov} main-stack linears covered (ML8|FP8); none left NATIVE")
    full_targets = list(find_dense_full_targets(model, coverage=_coverage))
    ml8_full = [(n, m) for (n, m, t) in full_targets if t is Tier.ML8]
    fp8_full = [(n, m) for (n, m, t) in full_targets if t is Tier.FP8]
    # nn.Embedding ML8 targets (e.g. token_embd at ml8-4 via ML8_TIER_OVERRIDE) have
    # NO activation Hessian — a gather, not a GEMM forward. They cannot go through
    # the GPTQ main loop; route them to the data-free ml8-4 embed pass below
    # (identity Hessian = per-group Lloyd-Max + scale, NO rotation/AWQ — an embed
    # gather has no downstream matmul to cancel a rotation against; see Design A).
    ml8_embed_full = [(n, m) for (n, m) in ml8_full if isinstance(m, nn.Embedding)]
    ml8_full = [(n, m) for (n, m) in ml8_full if not isinstance(m, nn.Embedding)]
    targets = filter_by_layer_limit(ml8_full, args.max_layers)
    fp8_targets = filter_by_layer_limit(fp8_full, args.max_layers)
    ml8_embed_targets = ml8_embed_full   # global tensors: no per-layer limit
    print(f"[targets] {len(targets)} ML8 linears to quantize "
          f"(coverage={_coverage})"
          + (f" + {len(fp8_targets)} FP8 tensors" if fp8_targets else "")
          + (f" + {len(ml8_embed_targets)} ML8 embedding(s)" if ml8_embed_targets else ""))

    # ── W4A8 faithful weight tiers (--faithful-weights): quant->dequant the
    # FP8-tier weights (token_embd, ssm alpha/beta) in place so the calibration
    # forward propagates fp8 embed/α/β, matching deployment. No-op when the FP8
    # tier is empty (e.g. --dense-coverage ffn). The later FP8 pass re-quantizes
    # these for the blobs (e4m3->e4m3 is ~idempotent, sub-noise).
    if args.faithful_weights:
        n_fw = 0
        for _fwn, _fwm in fp8_targets:
            with torch.no_grad():
                _fwm.weight.data.copy_(fp8_weight_override(_fwm.weight.data))
            n_fw += 1
        print(f"[faithful-weights] overrode {n_fw} fp8-tier tensors (embed, ssm a/b)")

    manifest = {"model": args.model, "gguf": args.gguf, "args": vars(args), "results": []}
    manifest_path = Path(args.output_dir) / "manifest.json"
    hessian_cache_path = Path(args.output_dir) / "hessians.pt"

    # ─── Resume: load prior manifest results + ppl_baseline if available ───
    if not args.no_resume and manifest_path.exists():
        try:
            with open(manifest_path) as f:
                prior = json.load(f)
            if prior.get("model") == args.model and prior.get("gguf") == args.gguf:
                manifest["results"] = prior.get("results", [])
                if "ppl_baseline" in prior:
                    manifest["ppl_baseline"] = prior["ppl_baseline"]
                print(f"[resume] loaded {len(manifest['results'])} prior results "
                      f"from {manifest_path}")
            else:
                print(f"[resume] prior manifest model/gguf mismatch — ignoring")
        except Exception as e:
            print(f"[resume] failed to load prior manifest ({e}) — starting fresh")

    if args.eval_ppl and "ppl_baseline" not in manifest:
        print(f"\n[ppl-baseline] computing f16 baseline PPL (paged forward path)...")
        baseline = eval_ppl_wikitext(model, tokenizer, args.device,
                                      max_tokens=args.ppl_max_tokens)
        print(f"  baseline PPL = {baseline['ppl']:.4f}")
        manifest["ppl_baseline"] = baseline
    elif args.eval_ppl:
        print(f"[ppl-baseline] reusing cached baseline PPL = "
              f"{manifest['ppl_baseline']['ppl']:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # MoE layer-major calibration path (G.7.h.1). One forward pass per
    # layer per "stage" (gate+up share a stage, down is its own), captures
    # Hessians for all 128 experts × 3 kinds via per-linear hooks. Then
    # rotates, AWQs, stacks, runs batched_gptq_quantize over the [E, N, K]
    # tensor. Save per-expert .pt blobs identical to scalar output so
    # ml8_to_gguf.py just works.
    # ═══════════════════════════════════════════════════════════════════
    if args.strategy == "moe":
        # Consolidated-MoE layer-major loop. Finds every PagedMoeExperts in the
        # model, processes one layer at a time:
        #   1. Hook the block to accumulate H_gate_up (from each expert's input)
        #      and H_down (from the silu(gate)*up output entering down_proj).
        #      Both are SHARED across experts in this (layer, kind).
        #   2. One forward pass over the calibration corpus populates both H.
        #   3. Read consolidated gate_proj / up_proj / down_proj from the
        #      PagedMoeExperts (page-faults via wp_native), get [E, N, K]
        #      stacks, rotate/AWQ, run batched_gptq_quantize.
        #   4. Save per-expert .pt blobs so ml8_to_gguf.py can stack them
        #      into ffn_*_exps tensors with sidecars.
        from resident_moe import ResidentMoeExperts
        # Both block types expose the same interface (gate_proj/up_proj/down_proj,
        # release_cached, the collect_* Hessian accumulators). PagedMoeExperts is
        # None when wp_native didn't import (e.g. gfx942), so filter it out.
        _moe_cls = tuple(c for c in (PagedMoeExperts, ResidentMoeExperts) if c is not None)
        moe_blocks_by_layer = {}
        for nm, mod in model.named_modules():
            if isinstance(mod, _moe_cls):
                parts = nm.split(".")
                try:
                    L = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                moe_blocks_by_layer[L] = (nm, mod)
        layer_ids = sorted(moe_blocks_by_layer)
        if args.max_layers is not None:
            layer_ids = layer_ids[: args.max_layers]
        print(f"[moe-loop] {len(layer_ids)} PagedMoeExperts blocks queued")

        # ── DIAGNOSTIC: capture router I/O for shared-vs-per-expert Hessian
        # analysis, then exit before Hessian/GPTQ. The router's INPUT is the MoE
        # hidden state = the gate/up experts' input, so bucketing it by the
        # router's top-k selection yields per-expert input sets offline.
        if getattr(args, "capture_router", None):
            import re as _re_cap
            cap_layers = [int(s) for s in args.capture_router.split(",") if s.strip() != ""]
            router_by_layer = {}
            for nm, m in model.named_modules():
                mt = _re_cap.search(r"layers\.(\d+)\.mlp\.gate$", nm)
                if mt:  # hook by name — gate may not be a plain nn.Linear in this arch
                    Lr = int(mt.group(1))
                    if Lr in cap_layers:
                        router_by_layer[Lr] = m
            missing = [L for L in cap_layers if L not in router_by_layer]
            if missing:
                gates = [nm for nm, _ in model.named_modules() if nm.endswith(".mlp.gate")]
                raise RuntimeError(f"[capture] router (.mlp.gate Linear) not found for "
                                   f"layers {missing}; sample gate modules: {gates[:5]}")
            cap = {L: {"x": [], "logits": []} for L in cap_layers}
            handles = []
            def _mk_cap(L):
                def _hook(mod, inp, out):
                    xt = inp[0]
                    ot = out[0] if isinstance(out, (tuple, list)) else out
                    xin = xt.detach().reshape(-1, xt.shape[-1]).to(torch.float16).cpu()
                    lo = ot.detach().reshape(-1, ot.shape[-1]).to(torch.float16).cpu()
                    cap[L]["x"].append(xin)
                    cap[L]["logits"].append(lo)
                return _hook
            for L in cap_layers:
                handles.append(router_by_layer[L].register_forward_hook(_mk_cap(L)))
            print(f"[capture] hooked routers on layers {cap_layers}; "
                  f"forward over {len(calib)} samples...")
            t_cap = time.time()
            with torch.no_grad():
                for s_i, ids in enumerate(calib):
                    model(ids.to(args.device))
                    if (s_i + 1) % 4 == 0 or s_i == len(calib) - 1:
                        print(f"  [capture] {s_i+1}/{len(calib)}  elapsed={time.time()-t_cap:.1f}s")
            for h in handles:
                h.remove()
            blob = {}
            for L in cap_layers:
                blob[L] = {
                    "x": torch.cat(cap[L]["x"], dim=0),
                    "logits": torch.cat(cap[L]["logits"], dim=0),
                    "num_experts": int(moe_blocks_by_layer[L][1].num_experts),
                }
                print(f"  [capture] L{L}: x={tuple(blob[L]['x'].shape)} "
                      f"logits={tuple(blob[L]['logits'].shape)}")
            torch.save(blob, args.capture_out)
            print(f"[capture] saved → {args.capture_out}")
            return

        kinds_specs = (
            ("gate_proj", "gate_proj", "ffn_gate_exps", "hidden"),
            ("up_proj",   "up_proj",   "ffn_up_exps",   "hidden"),
            ("down_proj", "down_proj", "ffn_down_exps", "intermediate"),
        )

        # ── Try loading cached Hessians first; on a 35B-A3B restart this
        # saves ~30 min by skipping the full calibration forward pass. The
        # cache is invalidated if any input that affects the Hessian content
        # has changed: model, gguf, n_samples, seq_len, max_layers, strategy.
        H_gate_up_per_layer: dict[int, torch.Tensor] = {}
        H_down_per_layer: dict[int, torch.Tensor] = {}
        cached_hessians_loaded = False
        if not args.no_resume and hessian_cache_path.exists():
            try:
                cached = torch.load(hessian_cache_path, map_location="cpu",
                                    weights_only=False)
                cache_keys_match = (
                    cached.get("model") == args.model
                    and cached.get("gguf") == args.gguf
                    and cached.get("n_samples") == args.n_samples
                    and cached.get("seq_len") == args.seq_len
                    and cached.get("corpus", "wiki") == args.corpus
                    and cached.get("corpus_seed", 0) == args.corpus_seed
                    and cached.get("max_layers") == args.max_layers
                    and cached.get("strategy") == args.strategy
                    and set(cached.get("H_gate_up_per_layer", {}).keys()) == set(layer_ids)
                    and set(cached.get("H_down_per_layer", {}).keys()) == set(layer_ids))
                if cache_keys_match:
                    H_gate_up_per_layer = {L: H.to(args.device)
                                            for L, H in cached["H_gate_up_per_layer"].items()}
                    H_down_per_layer = {L: H.to(args.device)
                                         for L, H in cached["H_down_per_layer"].items()}
                    cached_hessians_loaded = True
                    print(f"[hessian-cache] loaded Hessians for {len(layer_ids)} "
                          f"layers from {hessian_cache_path} — skipping forward pass")
                else:
                    print(f"[hessian-cache] cache present but params mismatch — recomputing")
            except Exception as e:
                print(f"[hessian-cache] load failed ({e}) — recomputing")

        if not cached_hessians_loaded:
            # ── Collect Hessians for ALL layers in ONE forward pass over the
            # calibration corpus. Each forward through the 35B-A3B model takes
            # ~58s due to expert paging from NVMe; doing one-forward-per-layer
            # would take 40 hours. Setting collect flags on every PagedMoeExperts
            # at once lets a single full forward populate all per-layer accs.
            # VRAM cost: per layer H = 2048² + 512² fp32 = 17 MB; × 40 = 680 MB.
            print(f"\n[hessian-pass] collecting H_gate_up + H_down on all "
                  f"{len(layer_ids)} layers in one forward pass...")
            for L in layer_ids:
                _, mb = moe_blocks_by_layer[L]
                mb.reset_calibration_acc()
                mb.collect_pre_gate_up = True
                mb.collect_pre_down    = True
            t_h = time.time()
            with torch.no_grad():
                for s_i, ids in enumerate(calib):
                    model(ids.to(args.device))
                    if (s_i + 1) % 4 == 0 or s_i == len(calib) - 1:
                        print(f"  [hessian-pass] {s_i+1}/{len(calib)} samples  "
                              f"elapsed={time.time()-t_h:.1f}s")
            for L in layer_ids:
                _, mb = moe_blocks_by_layer[L]
                mb.collect_pre_gate_up = False
                mb.collect_pre_down    = False
            t_h_done = time.time() - t_h
            print(f"[hessian-pass] done in {t_h_done:.1f}s ({t_h_done/len(calib):.1f}s/sample)")

            # ── Snapshot Hessians per layer; free accumulators after capture.
            for layer_idx in layer_ids:
                _, mb = moe_blocks_by_layer[layer_idx]
                hidden = mb.hidden_dim; interm = mb.intermediate_dim
                n_tok_gu = mb.pre_gate_up_n_tok; n_tok_dn = mb.pre_down_n_tok
                if mb.pre_gate_up_acc is None or n_tok_gu == 0:
                    print(f"  WARN L{layer_idx} zero pre_gate_up tokens; identity Hessian fallback")
                    H_gate_up_per_layer[layer_idx] = torch.eye(hidden, device=args.device, dtype=torch.float32)
                else:
                    H_gate_up_per_layer[layer_idx] = (mb.pre_gate_up_acc / max(n_tok_gu, 1)).to(args.device)
                if mb.pre_down_acc is None or n_tok_dn == 0:
                    print(f"  WARN L{layer_idx} zero pre_down tokens; identity Hessian fallback")
                    H_down_per_layer[layer_idx] = torch.eye(interm, device=args.device, dtype=torch.float32)
                else:
                    H_down_per_layer[layer_idx] = (mb.pre_down_acc / max(n_tok_dn, 1)).to(args.device)
                mb.reset_calibration_acc()
                print(f"  L{layer_idx} hessians: gate_up n_tok={n_tok_gu}  down n_tok={n_tok_dn}")

            # ── Persist Hessians to disk for resume. Atomic write via .tmp +
            # replace so a crash mid-save doesn't leave a corrupted cache.
            tmp_cache = hessian_cache_path.with_suffix(".pt.tmp")
            cache_blob = {
                "model": args.model, "gguf": args.gguf,
                "n_samples": args.n_samples, "seq_len": args.seq_len,
                "corpus": args.corpus, "corpus_seed": args.corpus_seed,
                "max_layers": args.max_layers, "strategy": args.strategy,
                "H_gate_up_per_layer": {L: H.cpu() for L, H in H_gate_up_per_layer.items()},
                "H_down_per_layer": {L: H.cpu() for L, H in H_down_per_layer.items()},
            }
            torch.save(cache_blob, tmp_cache)
            os.replace(tmp_cache, hessian_cache_path)
            cache_mb = hessian_cache_path.stat().st_size / 1e6
            print(f"[hessian-cache] saved {cache_mb:.1f} MB → {hessian_cache_path}")

        # ── Pre-compute per-(layer, kind) rotation + rotated Hessian. All workers
        # share these read-only tensors. Rotation must be deterministic per
        # (layer, kind) since all expert chunks of the same key need identical
        # rotation for the inference-time kernel.
        state_per_key = {}  # (layer_idx, kind_name) -> dict
        kind_seed_offset = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
        for layer_idx in layer_ids:
            mb = moe_blocks_by_layer[layer_idx][1]
            for kind, prop_name, gguf_suffix, dim_kind in kinds_specs:
                H = H_gate_up_per_layer[layer_idx] if dim_kind == "hidden" else H_down_per_layer[layer_idx]
                K = H.shape[0]
                rotation = None
                rotation_blob = None
                if args.rotation == "kronecker":
                    a, b = factor_for_dim(K, max_b=args.rotation_max_b)
                    seed = args.rotation_seed + layer_idx * 7 + kind_seed_offset[kind]
                    h_a = random_orthogonal(a, seed=seed)
                    rotation = KroneckerRotation(h_a=h_a, b_dim=b)
                    rotation_blob = rotation.to_dict()
                    rotation_blob["seed"] = seed
                    H = rotate_hessian(H, rotation)
                state_per_key[(layer_idx, kind)] = {
                    "H": H, "rotation": rotation, "rotation_blob": rotation_blob,
                    "K": K, "prop_name": prop_name, "moe_block": mb,
                }

        # ── Build task queue: each task = (layer, kind, expert_start, expert_count).
        TASK_E = args.task_e
        work = queue.Queue()
        for layer_idx in layer_ids:
            E_total = moe_blocks_by_layer[layer_idx][1].num_experts
            for kind, _, _, _ in kinds_specs:
                for e_start in range(0, E_total, TASK_E):
                    e_count = min(TASK_E, E_total - e_start)
                    work.put((layer_idx, kind, e_start, e_count))
        total_tasks = work.qsize()

        worker_config = [(args.device, args.workers_primary)]
        if args.secondary_device:
            worker_config.append((args.secondary_device, args.workers_secondary))
        print(f"\n[task-queue] {total_tasks} tasks (E={TASK_E}-chunks); workers: {worker_config}")
        if args.group_size_down is not None and args.group_size_down != args.group_size:
            print(f"[task-queue] down_proj kind uses group_size={args.group_size_down} "
                  f"(other kinds use {args.group_size})")

        manifest_lock = threading.Lock()
        moe_block_locks = {L: threading.Lock() for L in layer_ids}
        done_state = {"n": 0}
        done_lock = threading.Lock()
        t_dispatch_start = time.time()
        worker_errors = []

        def worker(device):
            try:
                while True:
                    try:
                        task = work.get_nowait()
                    except queue.Empty:
                        return
                    layer_idx, kind, e_start, e_count = task
                    st = state_per_key[(layer_idx, kind)]
                    K = st["K"]; H = st["H"]
                    rotation = st["rotation"]; rotation_blob = st["rotation_blob"]
                    moe_block = st["moe_block"]
                    prop_name = st["prop_name"]
                    t_task = time.time()

                    # ── Skip-if-exists: if every per-expert output blob in this
                    # chunk already exists on disk (from a prior partial run),
                    # skip all compute + pager activity for this task. Workers
                    # check independently, no lock needed — Path.exists is racy
                    # but the worst case is a single redundant recompute.
                    expected_paths = []
                    for j in range(e_count):
                        e_global = e_start + j
                        nm_pre = f"model.layers.{layer_idx}.mlp.experts.{e_global}.{kind}"
                        expected_paths.append(
                            Path(args.output_dir) /
                            f"{nm_pre.replace('.', '_').replace('/', '_')}.pt")
                    if not args.no_resume and all(p.exists() for p in expected_paths):
                        with done_lock:
                            done_state["n"] += 1
                            n_done = done_state["n"]
                        elapsed = time.time() - t_dispatch_start
                        print(f"  [{device}] {n_done}/{total_tasks}  L{layer_idx} "
                              f"{kind} E[{e_start}:{e_start+e_count}]  "
                              f"(skipped — all {e_count} blobs already present)  "
                              f"elapsed={elapsed/60:.1f}m")
                        continue

                    # Materialize consolidated weights on primary device, slice, move to worker device.
                    # Lock per-layer so concurrent workers on the same layer don't race the pager.
                    with moe_block_locks[layer_idx]:
                        consolidated = getattr(moe_block, prop_name)
                        W_slice_src = consolidated[e_start:e_start + e_count].clone()
                        moe_block.release_cached()
                    W_slice = W_slice_src.to(device).float()
                    del W_slice_src
                    H_dev = H.to(device)

                    if rotation is not None:
                        E_c, N_c, _ = W_slice.shape
                        W_flat = W_slice.reshape(E_c * N_c, K).to(rotation.h_a.device)
                        W_rot = rotation.forward(W_flat).to(device)
                        W_slice = W_rot.reshape(E_c, N_c, K)
                        del W_flat, W_rot

                    # Shared H across experts in this (layer, kind): pass as a
                    # zero-stride expand view, NOT materialized contiguous. The
                    # chunked Cholesky inside batched_gptq materializes only
                    # chunk_E × K × K at a time via the `+ damp * eye` add, so
                    # the upfront E-copy is wasted ~1 GB per gate/up worker.
                    H_stack = H_dev.unsqueeze(0).expand(e_count, K, K)
                    gs_for_kind = args.group_size
                    if kind == "down_proj" and args.group_size_down is not None:
                        gs_for_kind = args.group_size_down
                    out = batched_gptq_quantize(
                        W_stack=W_slice, H_stack=H_stack,
                        n_centroids=args.n_centroids, group_size=gs_for_kind,
                        n_iter=args.n_iter, fit_loss=args.fit_loss,
                        mag_weight_p=args.mag_weight_p,
                        snap_centroids=args.snap_centroids,
                        percdamp=args.percdamp,
                        act_order=args.act_order or args.heavy_rounds > 0,
                        heavy_rounds=args.heavy_rounds,
                        heavy_steps=args.heavy_steps, heavy_dtype=args.heavy_dtype,
                        heavy_lr_cent=args.heavy_lr_cent,
                        heavy_lr_scale=args.heavy_lr_scale)

                    # Move full stack to CPU once, but DO NOT save slices directly —
                    # `indices[j]` is a strided view into the [E, N, K] storage and
                    # torch.save serializes underlying storage (not the view), which
                    # would balloon every per-expert blob by E× (~140 MB instead of
                    # ~1 MB at E=128). Each slice MUST be .clone()'d when packed into
                    # its blob dict so it owns standalone storage.
                    indices = out["indices"].cpu()
                    centroids = out["centroids_per_group"].cpu()
                    scales = out["scale_per_group"].cpu()
                    mse_all = out["mse"]; w_snr = out["w_snr_db"]
                    y_snr = out["y_snr_db"]; rel = out["rel_err"]
                    N_c = W_slice.shape[1]

                    new_results = []
                    for j in range(e_count):
                        e_global = e_start + j
                        nm = f"model.layers.{layer_idx}.mlp.experts.{e_global}.{kind}"
                        out_path = Path(args.output_dir) / f"{nm.replace('.', '_').replace('/', '_')}.pt"
                        blob = {
                            "name": nm, "shape": [N_c, K],
                            "group_size": gs_for_kind, "n_centroids": args.n_centroids,
                            "indices":             indices[j].clone().contiguous(),
                            "centroids_per_group": centroids[j].clone().contiguous(),
                            "scale_per_group":     scales[j].clone().contiguous(),
                            "mse": float(mse_all[j].item()),
                            "w_snr_db": float(w_snr[j].item()),
                            "y_snr_db": float(y_snr[j].item()),
                            "rel_err": float(rel[j].item()),
                        }
                        if rotation_blob is not None:
                            blob["rotation"] = dict(rotation_blob)
                        torch.save(blob, out_path)
                        new_results.append({
                            "name": nm, "shape": [N_c, K],
                            "mse": float(mse_all[j].item()),
                            "y_snr_db": float(y_snr[j].item()),
                            "w_snr_db": float(w_snr[j].item()),
                        })
                    # Single lock acquisition: extend manifest + write to disk
                    # atomically (.tmp → replace). Cheap — manifest is JSON,
                    # 35B-A3B run = ~15 K entries × ~100 B = ~1.5 MB final size.
                    with manifest_lock:
                        manifest["results"].extend(new_results)
                        tmp_manifest = manifest_path.with_suffix(".json.tmp")
                        with open(tmp_manifest, "w") as f:
                            json.dump(manifest, f, indent=2)
                        os.replace(tmp_manifest, manifest_path)

                    t_task_done = time.time() - t_task
                    with done_lock:
                        done_state["n"] += 1
                        n_done = done_state["n"]
                    elapsed = time.time() - t_dispatch_start
                    eta_min = (elapsed / n_done) * (total_tasks - n_done) / 60 if n_done else 0
                    print(f"  [{device}] {n_done}/{total_tasks}  L{layer_idx} {kind} "
                          f"E[{e_start}:{e_start+e_count}]  t={t_task_done:.1f}s  "
                          f"Y_SNR_med={float(y_snr.median().item()):.2f}dB  "
                          f"elapsed={elapsed/60:.1f}m  ETA={eta_min:.1f}m")
                    del W_slice, H_stack, H_dev, consolidated, out, indices, centroids, scales
            except Exception as e:
                import traceback
                worker_errors.append(f"[{device}] {type(e).__name__}: {e}\n{traceback.format_exc()}")
                raise

        threads = []
        for device, n_workers in worker_config:
            for _ in range(n_workers):
                t = threading.Thread(target=worker, args=(device,), daemon=False)
                threads.append(t)
                t.start()
        for t in threads:
            t.join()

        if worker_errors:
            print(f"\n[task-queue] {len(worker_errors)} worker error(s):")
            for err in worker_errors[:3]:
                print(err)
            raise RuntimeError("task-queue workers failed; see above")

        t_dispatch_done = time.time() - t_dispatch_start
        print(f"\n[task-queue] {total_tasks} tasks completed in {t_dispatch_done/60:.1f}min "
              f"({t_dispatch_done/total_tasks:.1f}s/task average)")

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

        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest, manifest_path)
        print(f"\n[done] manifest: {manifest_path}")
        if pager is not None: pager.shutdown()
        return

    # ─── Dense resume: reload the completed contiguous prefix into the model ───
    # The dense path computes each layer's Hessian against the running, partially
    # QUANTIZED model (resident mode copies quantized weights back precisely so the
    # next layer's Hessian sees quantized upstream — GPTQ cross-layer error
    # propagation). So resuming correctly requires reloading every completed layer's
    # quantized weight BEFORE continuing, and trusting ONLY a contiguous prefix: the
    # blob for unit k+1 was computed against unit k being done, so the first gap
    # invalidates everything after it (those blobs are stale — ignore them).
    resume_start = 0
    if not args.no_resume:
        resume_start = dense_completed_prefix([n for n, _ in targets], args.output_dir)
        if resume_start > 0:
            # Blobs are authoritative for the prefix — rebuild results from them so a
            # stale/short manifest.json can't desync from what's actually on disk.
            manifest["results"] = load_dense_prefix_into_model(
                resume_start, [n for n, _ in targets], model, args.output_dir,
                resident=args.resident, dtype=dtype, device=args.device)
            nxt = targets[resume_start][0] if resume_start < len(targets) else "(all done)"
            print(f"[resume] dense: restored {resume_start}/{len(targets)} completed "
                  f"linears from blobs, resuming at {nxt}")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ── W4A8 deployment-faithful activation path (--faithful-acts) ──
    # Install a persistent pre-hook on every target so activations propagate as
    # x_eff = e4m3(x@Q)@Qᵀ; the per-layer Hessian is read from the hook (already
    # in rotated+quant space) instead of rotate_hessian(compute_hessian(...)).
    faithful_hooks = {}
    if args.faithful_acts:
        if args.rotation != "kronecker":
            raise ValueError("--faithful-acts requires --rotation kronecker")
        for j, (fname, flayer) in enumerate(targets):
            _frows, fin = flayer.weight.shape
            fa, fb = factor_for_dim(fin, max_b=args.rotation_max_b)
            frot = KroneckerRotation(
                h_a=random_orthogonal(fa, seed=args.rotation_seed + j), b_dim=fb)
            hk = FaithfulActHook(frot, enabled=True)
            flayer.register_forward_pre_hook(hk)
            faithful_hooks[j] = (hk, frot)
        print(f"[faithful-acts] installed {len(faithful_hooks)} activation-e4m3 pre-hooks")

    _precollected_H = None
    if args.hessian_mode == "single":
        if not args.faithful_acts or args.awq != "none":
            raise SystemExit("[hessian-mode] 'single' requires --faithful-acts and "
                             "--awq none; use --hessian-mode per-target otherwise.")
        if resume_start > 0:
            # Single-pass precollects ALL Hessians against the original model in one
            # forward; a resumed prefix would be loaded as quantized, so the single
            # forward would see a quantized-prefix/original-suffix mix that matches
            # neither a fresh run nor the per-target path. Disallow rather than
            # silently produce incoherent Hessians.
            raise SystemExit(f"[hessian-mode] 'single' is incompatible with resume "
                             f"(found {resume_start} completed targets); rerun with "
                             f"--no-resume, or use --hessian-mode per-target.")

    if args.hessian_mode == "block-sequential":
        if not args.faithful_acts or args.awq != "none":
            raise SystemExit("[hessian-mode] 'block-sequential' requires --faithful-acts and --awq none.")
        if resume_start > 0:
            raise SystemExit("[hessian-mode] 'block-sequential' is incompatible with resume; rerun with --no-resume.")

    if args.hessian_mode == "single":
        print(f"\n[hessian-single] collecting H for all {len(targets)} targets in ONE "
              f"forward pass over {len(calib)} samples...")
        _t_sp = time.time()
        with TIMER.phase("hessian_forward"):
            _precollected_H = collect_hessians_single_pass(
                {i: faithful_hooks[i][0] for i in range(len(targets))},
                calib, model, args.device)
        print(f"[hessian-single] done in {time.time()-_t_sp:.1f}s "
              f"({len(_precollected_H)} Hessians, 1 forward)")

    # ─── Per-layer calibration loop (same math as calibrate_ml8.py) ───
    if args.hessian_mode == "block-sequential":
        from block_arch_adapter import get_adapter
        from block_sequential import run_walk
        adapter = get_adapter(args.arch)
        ml8_full_names = {n for n, _ in targets}
        # is_ml8 is called from adapter.ml8_targets(block, b_idx, is_ml8) with a
        # leaf name (e.g. "linear_attn.in_proj_qkv"). We need the full name check
        # ("model.layers.{b_idx}.{leaf}") so that blocks beyond --max-layers are
        # excluded (they have no FaithfulActHook installed). Wrap the adapter so
        # that ml8_targets injects a b_idx-aware is_ml8 before dispatching.
        _base_adapter = adapter
        class _ScopedAdapter:
            def iter_blocks(self, model):
                return _base_adapter.iter_blocks(model)
            def run_block(self, block, args, kwargs):
                return _base_adapter.run_block(block, args, kwargs)
            def ml8_targets(self, block, b_idx, is_ml8_ignored):
                prefix = f"model.layers.{b_idx}."
                def _is_ml8_full(leaf):
                    return (prefix + leaf) in ml8_full_names
                return _base_adapter.ml8_targets(block, b_idx, _is_ml8_full)
        adapter = _ScopedAdapter()
        _walked_names = set()
        def quantize_fn(full_name, layer, idx, H, n_tok, sum_abs, rotation_hook):
            _walked_names.add(full_name)
            quantize_one_target(full_name, layer, idx, H, n_tok, sum_abs, rotation_hook,
                                args, dtype, manifest, Path(args.output_dir), TIMER)
        with TIMER.phase("hessian_forward"):
            n_done = run_walk(model, adapter, calib, args.device,
                              is_ml8=lambda n: n in ml8_full_names,  # fallback; _ScopedAdapter overrides
                              quantize_fn=quantize_fn,
                              hook_factory=_BlockHessianHook)
        print(f"[block-sequential] quantized {n_done} ml8 targets via causal walk")
        # Fail loud on any enumerated ml8 target inside a walked block that the
        # adapter's sub-groups never visited — an omitted leaf is silently written
        # bf16 by the converter (the B3 ssm_out bug, 2026-06-09). Targets outside
        # model.model.layers (lm_head/eh_proj/embedding) are handled elsewhere.
        _n_walk_blocks = len(_base_adapter.iter_blocks(model))
        _expected = set()
        for _n in ml8_full_names:
            _m = re.match(r"model\.layers\.(\d+)\.", _n)
            if _m and int(_m.group(1)) < _n_walk_blocks:
                _expected.add(_n)
        _missed = _expected - _walked_names
        if _missed:
            raise RuntimeError(
                f"[block-sequential] walk MISSED {len(_missed)} enumerated ml8 target(s) — "
                f"the {args.arch!r} adapter's sub-groups don't cover them; they would be "
                f"silently left bf16. First 8: {sorted(_missed)[:8]}")
    else:
        for i, (name, layer) in enumerate(targets):
            if i < resume_start:
                continue
            rows, in_feat = layer.weight.shape
            print(f"\n[{i+1}/{len(targets)}] {name}  shape=({rows}, {in_feat})")
            collect_awq = args.awq != "none"
            _t_hess0 = time.time()
            if args.hessian_mode == "single":
                H, n_tok = _precollected_H[i]; sum_abs = None
            elif args.faithful_acts:
                hk_i, _frot_i = faithful_hooks[i]
                hk_i.reset_hessian(); hk_i.set_hessian_target(True)
                with TIMER.phase("hessian_forward"):
                    _H_discard, n_tok, sum_abs = compute_hessian(
                        layer, calib, model, args.device, collect_awq=collect_awq)
                hk_i.set_hessian_target(False); H = hk_i.H
            else:
                with TIMER.phase("hessian_forward"):
                    H, n_tok, sum_abs = compute_hessian(
                        layer, calib, model, args.device, collect_awq=collect_awq)
            if args.phase_timing:
                TIMER._events.append({
                    "label": "hessian_forward_target", "target": name,
                    "n_tok": int(n_tok), "seconds": time.time() - _t_hess0,
                    "shape": [int(rows), int(in_feat)]})
            rotation_hook = faithful_hooks[i][1] if args.faithful_acts else None
            quantize_one_target(name, layer, i, H, n_tok, sum_abs, rotation_hook,
                                args, dtype, manifest, Path(args.output_dir), TIMER)
            del H
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if args.phase_timing:
        _pt_path = Path(args.output_dir) / "phase_timing.json"
        TIMER.dump_json(_pt_path)
        _s = TIMER.summary()
        print("\n=== [phase-timing] aggregate ===")
        for _lbl, _d in sorted(_s["phases"].items(),
                               key=lambda kv: -kv[1]["seconds"]):
            print(f"  {_lbl:18s} {_d['seconds']:9.1f}s  "
                  f"({100*_d['seconds']/max(_s['total_seconds'],1e-9):5.1f}%)  "
                  f"calls={_d['calls']}")
        print(f"  {'TOTAL':18s} {_s['total_seconds']:9.1f}s")
        print(f"[phase-timing] wrote {_pt_path}")

    # ─── ML8-4 embedding pass (data-free) ──────────────────────────────────
    # token_embd at ml8-4 (via ML8_TIER_OVERRIDE=token_embd=ml8): an embedding is a
    # gather with no activation Hessian and (per Design A) no rotation/AWQ — so
    # quantize its [vocab, hidden] weight DATA-FREE with an identity Hessian. With
    # H=I, batched_gptq reduces to per-K-group Lloyd-Max + per-row scale and the
    # GPTQ error-propagation tail is zero, i.e. plain ml8-4 codebook quant. The
    # blob carries the same schema as the GPTQ loop MINUS rotation/awq sidecars, so
    # the converter writes a rotation-free ml8_4 tensor and inference applies no
    # rotation for the embed (a gather has no downstream matmul to undo it against).
    for ei, (name, module) in enumerate(ml8_embed_targets):
        w = module.weight.detach().float()                       # [vocab, hidden] = [N, K]
        N, K = w.shape
        if K % args.group_size != 0:
            print(f"[ml8-embed] SKIP {name}: K={K} not divisible by group_size={args.group_size}")
            continue
        # Large embedding → quantize on CPU to respect the VRAM guard (per-group
        # Lloyd-Max + nearest-centroid is CPU-friendly and runs once).
        emb_dev = "cpu" if (N * K > 64_000_000) else (args.device if args.device.startswith("cuda") else "cpu")
        Wr = w.to(emb_dev).unsqueeze(0)                                       # [1, N, K]
        Hr = torch.eye(K, device=emb_dev, dtype=torch.float32).unsqueeze(0)   # identity → data-free
        out = batched_gptq_quantize(
            W_stack=Wr, H_stack=Hr,
            n_centroids=args.n_centroids, group_size=args.group_size,
            n_iter=args.n_iter, fit_loss=args.fit_loss,
            mag_weight_p=args.mag_weight_p,
            snap_centroids=args.snap_centroids,
            percdamp=args.percdamp,
            act_order=False, heavy_rounds=0)
        out_path = Path(args.output_dir) / f"{name.replace('.', '_').replace('/', '_')}.pt"
        blob = {
            "name": name,
            "shape": [N, K],
            "group_size": args.group_size,
            "n_centroids": args.n_centroids,
            "indices": out["indices"][0].cpu(),
            "centroids_per_group": out["centroids_per_group"][0].cpu(),
            "scale_per_group": out["scale_per_group"][0].cpu(),
            "mse": float(out["mse"][0].item()),
            "w_snr_db": float(out["w_snr_db"][0].item()),
            "y_snr_db": float(out["y_snr_db"][0].item()),
            "rel_err": float(out["rel_err"][0].item()),
        }
        torch.save(blob, out_path)
        print(f"[ml8-embed {ei+1}/{len(ml8_embed_targets)}] {name}  shape=({N},{K})  "
              f"dev={emb_dev}  W_SNR={blob['w_snr_db']:.1f}dB  saved: {out_path.name}")
        del Wr, Hr, out
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ─── FP8 pass (--dense-coverage full only) ─────────────────────────────
    # Separate, simpler pass: NO Hessian / rotation / AWQ / GPTQ. Read the
    # weight directly and per-group (group_size FIXED at 32, matching the
    # on-disk ML8_FP8 ggml block of 32 e4m3 + 1 fp16 scale) cast to e4m3.
    # alpha/beta_proj are nn.Linear ([N,K]); embed_tokens is nn.Embedding
    # ([vocab, hidden]) — quantize_scaled_fp8 groups along K=hidden either way.
    fp8_blob_names: list[str] = []
    FP8_GROUP_SIZE = 32   # FIXED — must match ML8_FP8 ggml block; NOT args.group_size
    for fi, (name, module) in enumerate(fp8_targets):
        # Read the weight directly. Resident nn.Linear / nn.Embedding expose a
        # real Parameter; this does NOT touch weight_override (so the resident
        # weight_override-assignment concern does not apply to this pass).
        w = module.weight.detach()
        N, K = w.shape
        if K % FP8_GROUP_SIZE != 0:
            print(f"[fp8] SKIP {name}: K={K} not divisible by {FP8_GROUP_SIZE}")
            continue
        # Honor memory guards: a ~1B-param embedding ([vocab, hidden]) may not
        # fit alongside everything else. Do FP8 on CPU when the tensor is large
        # or the device is CUDA-tight; the math is a cheap per-group amax+cast.
        fp8_dev = "cpu" if (N * K > 64_000_000) else (args.device if args.device.startswith("cuda") else "cpu")
        packed = quantize_scaled_fp8(w.float().to(fp8_dev), group_size=FP8_GROUP_SIZE)
        out_path = Path(args.output_dir) / f"{name.replace('.', '_').replace('/', '_')}.fp8.pt"
        blob = {
            "name": name,
            "tier": "fp8",
            "shape": [N, K],
            "group_size": FP8_GROUP_SIZE,
            "e4m3": packed["e4m3"].to(torch.float32).cpu(),   # [N, K]
            "scale": packed["scale"].to(torch.float16).cpu(), # [N, K/32]
        }
        torch.save(blob, out_path)
        fp8_blob_names.append(name)
        print(f"[{fi+1}/{len(fp8_targets)}] [fp8] {name}  shape=({N}, {K})  "
              f"dev={fp8_dev}  saved: {out_path.name}")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ─── Record dense-coverage scope in the manifest (D — explicit, not silent).
    # These FP8-tier blobs exist on disk and await converter (ml8_to_gguf.py)
    # support; the marker records that fact for downstream tooling.
    manifest["dense_coverage"] = _coverage
    if _coverage == "full":
        manifest["fp8_tensors"] = fp8_blob_names
        # NOTE: FP8-tier blobs (*.fp8.pt) await converter ingestion — separate task.

    if args.eval_ppl:
        print(f"\n[ppl-quant] computing quantized PPL...")
        quantized = eval_ppl_wikitext(model, tokenizer, args.device,
                                       max_tokens=args.ppl_max_tokens)
        print(f"  quantized PPL = {quantized['ppl']:.4f}")
        manifest["ppl_quantized"] = quantized
        delta = quantized["ppl"] - manifest["ppl_baseline"]["ppl"]
        manifest["ppl_delta"] = delta
        print(f"\n  Δ_PPL = {delta:+.4f}")

    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)
    print(f"\n[done] manifest: {manifest_path}")

    if pager is not None: pager.shutdown()


if __name__ == "__main__":
    main()
