"""Arch-aware flash-linear-attention (fla) dtype shim for ml8 calibration.

WHY THIS EXISTS
---------------
fla's gated-delta-rule Triton kernel emits ``llvm.amdgcn.fdot2.bf16.bf16`` — a
CDNA/GFX9-class bf16 dot-product intrinsic. RDNA's LLVM backend (gfx10/11/12)
cannot lower it, so the kernel JIT aborts the whole process:

    LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16

The same kernel runs fine in fp16 and fp32 on RDNA (verified on gfx1201/R9700).
We run the recurrence scan in **fp32** on RDNA because that ALSO matches the
deployed f32 recurrence core (ssm_a / dt / conv1d / norms are all f32), so it's
faithful, not just a workaround. On CDNA3 (gfx94x / MI300X) the bf16 kernel
lowers natively, so we leave it alone there.

WITHOUT fla installed, HF qwen3.5 binds these attributes to the slow
``torch_chunk_gated_delta_rule`` reference (CPU-dispatch-bound). This shim skips
that by name, so it is a safe no-op in every configuration except the one that
needs it: RDNA + fla installed.

CPU FALLBACK (apply_fla_cpu_fallback)
--------------------------------------
When fla IS installed and the calibration device is CPU, the ``__init__`` of
``Qwen3_5GatedDeltaNet`` binds ``self.chunk_gated_delta_rule`` to the fla Triton
kernel (``chunk_gated_delta_rule or torch_chunk_gated_delta_rule`` — the "or"
only fires when fla is absent). Triton cannot run on CPU tensors:

    ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)

``apply_fla_cpu_fallback`` swaps those attributes back to the torch reference
implementations so CPU calibration (used in unit tests) works even with fla
installed. GPU paths are unaffected (they short-circuit immediately).

See docs/superpowers/2026-05-31-calibration-fidelity-fla-rdna.md.
"""
from __future__ import annotations
import torch

# AMD ISAs where fla's bf16 fdot2 lowers natively (CDNA family). Everything else
# AMD (RDNA: gfx10/gfx11/gfx12) needs fp32. NVIDIA handles bf16 fine.
_CDNA_PREFIXES = ("gfx90a", "gfx94", "gfx95")  # MI200 / MI300 / MI350

# fla attributes the qwen3.5 linear-attn layer binds (prefill chunk + decode
# recurrent). Calibration uses the chunk path; we wrap both for safety.
_FLA_ATTRS = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")


def _needs_fp32_scan(device) -> bool:
    """True iff the linear-attn scan must run in fp32 to compile on this device
    (AMD RDNA). False for CDNA, NVIDIA, CPU, or anything we can't identify."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return False
    try:
        arch = (getattr(torch.cuda.get_device_properties(dev.index or 0),
                         "gcnArchName", "") or "")
    except Exception:
        return False
    if not arch or not arch.startswith("gfx"):
        return False                      # NVIDIA / unknown → bf16 is fine
    if arch.startswith(_CDNA_PREFIXES):
        return False                      # CDNA → native bf16 fdot2
    return True                           # RDNA → needs fp32


def _wrap(fn, scan_dtype):
    """Cast q/k/v (positional) and beta (kwarg) to scan_dtype, run the fla kernel,
    cast the core output back to the original input dtype. ``g`` stays fp32 (HF
    already passes it fp32); all other kwargs (initial_state, output_final_state,
    use_qk_l2norm_in_kernel, ...) pass through untouched."""
    def wrapped(q, k, v, *args, **kw):
        odt = q.dtype
        q = q.to(scan_dtype); k = k.to(scan_dtype); v = v.to(scan_dtype)
        if kw.get("beta") is not None:
            kw["beta"] = kw["beta"].to(scan_dtype)
        out = fn(q, k, v, *args, **kw)
        if isinstance(out, tuple):
            core = out[0]
            return ((core.to(odt) if core is not None else core),) + tuple(out[1:])
        return out.to(odt)
    wrapped._fla_dtype_wrapped = True
    return wrapped


def apply_fla_arch_shim(model, device, scan_dtype=torch.float32, verbose=True) -> int:
    """Wrap each linear-attn layer's fla kernel(s) to run in ``scan_dtype`` on RDNA.

    Returns the number of (module, attr) pairs wrapped. 0 means no-op: CDNA3 /
    NVIDIA / CPU, or fla not installed (torch reference in use)."""
    if not _needs_fp32_scan(device):
        return 0
    n = 0
    for mod in model.modules():
        for attr in _FLA_ATTRS:
            fn = getattr(mod, attr, None)
            if fn is None or getattr(fn, "_fla_dtype_wrapped", False):
                continue
            # Skip the pure-pytorch reference — it runs bf16 fine and wrapping it
            # to fp32 would be pointless (and it isn't the thing that crashes).
            if getattr(fn, "__name__", "").startswith("torch_"):
                continue
            setattr(mod, attr, _wrap(fn, scan_dtype))
            n += 1
    if verbose:
        print(f"[fla-shim] RDNA detected: wrapped {n} fla kernel binding(s) to "
              f"{scan_dtype} scan (bf16 fdot2 workaround; matches deployed f32 "
              f"recurrence).")
    return n


# Map from fla Triton attr name → the torch reference attr name that HF qwen3.5
# would have used if fla were absent. Both live as module-level callables in
# transformers.models.qwen3_5.modeling_qwen3_5 and are CPU-compatible.
_FLA_TO_TORCH_REF = {
    "chunk_gated_delta_rule":      "torch_chunk_gated_delta_rule",
    "recurrent_gated_delta_rule":  "torch_recurrent_gated_delta_rule",
}


def apply_fla_cpu_fallback(model, device, verbose=True) -> int:
    """Replace fla Triton kernels with their torch reference equivalents on CPU.

    When fla is installed, ``Qwen3_5GatedDeltaNet.__init__`` binds several
    attributes to fla Triton kernels regardless of the runtime device:

      * ``self.chunk_gated_delta_rule``     → fla Triton chunk scan
      * ``self.recurrent_gated_delta_rule`` → fla Triton recurrent scan
      * ``self.norm``                       → ``FusedRMSNormGated`` (Triton layernorm)

    Triton cannot dispatch to CPU tensors and raises:

        ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)

    This function swaps each of these back to CPU-compatible equivalents so that
    CPU calibration (unit tests, CI) works even when fla is installed:

      * Callable attrs → HF torch reference implementations (``torch_*`` names)
      * ``FusedRMSNormGated`` norm modules → ``Qwen3_5RMSNormGated`` with weights
        copied from the fla instance (both have ``self.weight`` of shape
        ``[hidden_size]``; the computation is numerically equivalent).

    GPU paths are untouched — this function is a no-op for non-CPU devices.

    Returns the total number of replacements made (callables + norm modules);
    0 means no-op."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cpu":
        return 0

    # Lazily import HF qwen3.5 references.  Guarded so this is safe for any
    # architecture that doesn't have these classes (non-qwen3.5 models).
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as _q35
        torch_refs = {
            attr: getattr(_q35, torch_name, None)
            for attr, torch_name in _FLA_TO_TORCH_REF.items()
        }
        FusedRMSNormGated = getattr(_q35, "FusedRMSNormGated", None)
        Qwen3_5RMSNormGated = getattr(_q35, "Qwen3_5RMSNormGated", None)
    except (ImportError, AttributeError):
        return 0  # not a qwen3.5 model or transformers too old — safe no-op

    n = 0

    # 1. Replace callable fla Triton kernel attributes with torch references.
    for mod in model.modules():
        for attr, ref_fn in torch_refs.items():
            if ref_fn is None:
                continue
            fn = getattr(mod, attr, None)
            if fn is None:
                continue
            # Skip if already the torch reference (name starts with "torch_").
            if getattr(fn, "__name__", "").startswith("torch_"):
                continue
            # Skip if already wrapped by apply_fla_arch_shim (shouldn't happen
            # on CPU, but be defensive).
            if getattr(fn, "_fla_dtype_wrapped", False):
                continue
            setattr(mod, attr, ref_fn)
            n += 1

    # 2. Replace FusedRMSNormGated norm sub-modules with Qwen3_5RMSNormGated.
    #    Both have a single `weight` Parameter of shape [hidden_size]; we copy
    #    the weight from the fla instance into the new pure-torch module.
    if FusedRMSNormGated is not None and Qwen3_5RMSNormGated is not None:
        for parent in list(model.modules()):
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, FusedRMSNormGated):
                    continue
                hidden_size = child.hidden_size
                eps = child.eps
                replacement = Qwen3_5RMSNormGated(hidden_size, eps=eps)
                # Copy trained weight (both have self.weight of shape [hidden_size]).
                if child.weight is not None and replacement.weight is not None:
                    with torch.no_grad():
                        replacement.weight.copy_(child.weight.to(replacement.weight.dtype))
                replacement = replacement.to(next(child.parameters(), torch.empty(0)).device
                                             if any(True for _ in child.parameters()) else "cpu")
                setattr(parent, child_name, replacement)
                n += 1

    if verbose and n:
        print(f"[fla-cpu] CPU device: replaced {n} fla Triton component(s) with "
              f"torch reference (Triton cannot dispatch to CPU tensors).")
    return n
