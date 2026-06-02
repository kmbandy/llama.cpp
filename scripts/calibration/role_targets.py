import enum

class Tier(enum.Enum):
    ML8 = "ml8"        # 4-bit ml8 GEMM, full GPTQ pipeline
    FP8 = "fp8"        # 8-bit scaled-FP8, direct cast
    NATIVE = "native"  # leave as-is (A/dt/conv/norms handled by caller)

# Tier is keyed on the STABLE GGUF role leaf — our own ~14-name convention — NOT on
# fragile per-arch HF submodule names. The HF-name -> GGUF-name resolution is
# delegated to llama.cpp's authoritative TensorNameMap (the SAME map convert_hf_to_gguf
# uses and that every GGUF conversion exercises), so any architecture / model variant /
# transformers rename the converter already supports is handled here for free. This is
# the 2026-05-31 fix for "role table written from assumption, never probed against a real
# checkpoint" — we no longer maintain a parallel (and previously wrong) HF-name table.
# BASE (default) role->tier map. The live map `_GGUF_ROLE_TIER` is rebuilt from
# this each configure() call, optionally patched by ML8_TIER_OVERRIDE so a whole
# ROLE can be moved between tiers uniformly ("tread shaping" — every tensor of a
# role shares one tier; never per-tensor). Edit the BASE only to change defaults.
_GGUF_ROLE_TIER_BASE = {
    # ML8 (4-bit) GEMMs
    "attn_q": ("attn_q", Tier.ML8), "attn_k": ("attn_k", Tier.ML8),
    "attn_v": ("attn_v", Tier.ML8), "attn_output": ("attn_out", Tier.ML8),
    "attn_qkv": ("attn_qkv", Tier.ML8), "attn_gate": ("attn_gate", Tier.ML8),
    "ssm_out": ("ssm_out", Tier.ML8),
    "ffn_gate": ("ffn_gate", Tier.ML8), "ffn_up": ("ffn_up", Tier.ML8),
    "ffn_down": ("ffn_down", Tier.ML8),
    "output": ("lm_head", Tier.ML8),                 # lm_head / tied output projection
    # FP8 (8-bit) tier (Design A tier 2)
    "ssm_alpha": ("ssm_alpha", Tier.FP8), "ssm_beta": ("ssm_beta", Tier.FP8),
    "token_embd": ("token_embd", Tier.FP8),
}
# Live map, set by configure(); starts as a copy of the base.
_GGUF_ROLE_TIER = dict(_GGUF_ROLE_TIER_BASE)

_TIER_BY_NAME = {"ml8": Tier.ML8, "fp8": Tier.FP8, "native": Tier.NATIVE}


def _apply_tier_overrides(spec):
    """Rebuild the live _GGUF_ROLE_TIER from the base, then apply `spec`.

    spec: 'leaf=tier,leaf=tier' (e.g. 'token_embd=ml8,ssm_out=fp8'). Keys are
    GGUF role leaves from _GGUF_ROLE_TIER_BASE; values are ml8|fp8|native. Only the
    Tier changes — the role name is preserved. Fails loud on an unknown leaf or
    tier (a typo silently mis-tiering the whole model would be a quiet quality bug).
    Returns the list of applied 'leaf->tier' strings (for logging).
    """
    global _GGUF_ROLE_TIER
    _GGUF_ROLE_TIER = dict(_GGUF_ROLE_TIER_BASE)
    if not spec:
        return []
    applied = []
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        leaf, sep, tname = pair.partition("=")
        leaf, tname = leaf.strip(), tname.strip().lower()
        if not sep:
            raise ValueError(f"ML8_TIER_OVERRIDE: malformed entry {pair!r} (want 'leaf=tier')")
        if tname not in _TIER_BY_NAME:
            raise ValueError(f"ML8_TIER_OVERRIDE: unknown tier {tname!r} in {pair!r} "
                             f"(use ml8|fp8|native)")
        if leaf not in _GGUF_ROLE_TIER_BASE:
            raise ValueError(f"ML8_TIER_OVERRIDE: unknown role leaf {leaf!r} "
                             f"(known: {sorted(_GGUF_ROLE_TIER_BASE)})")
        role_name = _GGUF_ROLE_TIER_BASE[leaf][0]
        _GGUF_ROLE_TIER[leaf] = (role_name, _TIER_BY_NAME[tname])
        applied.append(f"{leaf}->{tname}")
    return applied

# Parents whose nn.Linear children MUST be quantized (ML8 or FP8) in the main
# language-model stack. Any nn.Linear here that resolves to NATIVE means the
# checkpoint's names aren't covered — a silent coverage-loss bug.
_MAIN_STACK_LINEAR_PARENTS = (".linear_attn.", ".self_attn.", ".mlp.", ".attention.")

# Module-global authoritative name map, set by configure(). Kept module-global (not
# threaded through every call site) because every classify_role() call happens in the
# main process of calibrate_ml8_paged.py / ml8_to_gguf.py — never in a quant worker.
_TNM = None
_ARCH = None


def configure(arch_name: str, n_blocks: int, tier_override=None) -> None:
    """Build the authoritative HF->GGUF TensorNameMap for `arch_name`.

    MUST be called once before classify_role(). Fails loudly on an unknown arch
    rather than silently mis-tiering.

    Role->tier overrides come from `tier_override` (a 'leaf=tier,...' string) or, if
    None, the ML8_TIER_OVERRIDE env var. Both calibrate_ml8_paged and ml8_to_gguf
    call configure() with the same env, so calibration and conversion stay tier-
    consistent across the two processes.
    """
    global _TNM, _ARCH
    import os
    from gguf import MODEL_ARCH                       # local: keep import-light for callers
    from gguf.tensor_mapping import get_tensor_name_map
    arch_enum = next((ma for ma in MODEL_ARCH if ma.name.lower() == arch_name.lower()), None)
    if arch_enum is None:
        raise ValueError(f"role_targets.configure: no MODEL_ARCH match for arch_name={arch_name!r}")
    _TNM = get_tensor_name_map(arch_enum, n_blocks)
    _ARCH = arch_name

    spec = tier_override if tier_override is not None else os.environ.get("ML8_TIER_OVERRIDE")
    applied = _apply_tier_overrides(spec)
    if applied:
        print(f"[role-tier] ML8_TIER_OVERRIDE applied (role-uniform): {', '.join(applied)}")


def _layer_idx(name):
    parts = name.split(".")
    try: return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError): return None


def _resolve_gguf(hf_name: str):
    """HF module/param name -> GGUF tensor name (no .weight), or None if unmapped."""
    if _TNM is None:
        raise RuntimeError("role_targets.configure(arch_name, n_blocks) must be called "
                           "before classify_role()")
    # Normalize the multimodal wrapper: the LM tensors live under
    # 'model.language_model.…' but the TensorNameMap keys are 'model.layers.{bid}.…',
    # so collapse anything up to and including 'language_model.' down to 'model.'.
    # (Pure-text arches without that wrapper pass through unchanged.)
    h = hf_name
    if "language_model." in h:
        h = "model." + h.split("language_model.", 1)[1]
    h = h.rsplit(".weight", 1)[0].rsplit(".bias", 1)[0]
    return _TNM.get_name(h)


def classify_role(hf_name: str):
    """Return (gguf_name, role, Tier). NATIVE for anything we don't quantize.

    Resolves the GGUF name via the authoritative TensorNameMap, then keys the tier
    off the stable GGUF role leaf. configure() must have been called first.
    """
    # MTP / NextN head: not in the generic per-block map; classify explicitly.
    if hf_name.endswith("eh_proj"):
        L = _layer_idx(hf_name)
        g = f"blk.{L}.nextn.eh_proj" if L is not None else "nextn.eh_proj"
        return (g + ".weight", "eh_proj", Tier.ML8)

    gguf = _resolve_gguf(hf_name)
    if gguf is None:
        return (hf_name, "native", Tier.NATIVE)
    leaf = gguf.split(".")[-1]
    role, tier = _GGUF_ROLE_TIER.get(leaf, (leaf, Tier.NATIVE))
    return (gguf + ".weight", role, tier)


def assert_main_stack_covered(model) -> int:
    """Fail loudly if any main-stack nn.Linear is classified NATIVE.

    Walks the actual loaded model — the source of truth — and refuses to calibrate
    a model whose SSM/attention/MLP linears we'd silently drop to bf16. Returns the
    number of covered (ML8|FP8) main-stack linears; raises ValueError listing the
    offenders otherwise. (This is the guard the 2026-05-31 silent-coverage bug should
    have tripped.)
    """
    import torch.nn as nn  # local import: role_targets stays import-light for tests
    missed, covered = [], 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if _layer_idx(name) is None:                       # skip vision/global (no layer idx)
            continue
        if not any(p in name for p in _MAIN_STACK_LINEAR_PARENTS):
            continue
        _, _, tier = classify_role(name)
        if tier is Tier.NATIVE:
            missed.append(name)
        else:
            covered += 1
    if missed:
        raise ValueError(
            f"[role-guard] {len(missed)} main-stack nn.Linear(s) classified NATIVE — "
            f"the checkpoint's module names aren't resolved by the {_ARCH!r} TensorNameMap "
            f"(would be silently left bf16/uncovered). Offenders (first 12): {missed[:12]}")
    return covered
