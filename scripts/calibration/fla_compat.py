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
