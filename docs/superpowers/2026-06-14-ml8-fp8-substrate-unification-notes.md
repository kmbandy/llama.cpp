# ml8 FP8 Kernel-Substrate Unification — Working Notes

**Date:** 2026-06-14
**Status:** capture/notes — NOT the spec. Spec to follow.
**Purpose:** bank the aggregated recon, the build items, and next steps before we lose the thread. Decision-grade summary; the formal spec is a separate doc.

---

## 0. How we got here (one paragraph)

MAD-290 closed: the real fp8 backward landed (`ml8_backward_gemms` via `torch._scaled_mm`), taking the 4B trainer micro-step **10.6s → ~4.3s**. Profiling the residual showed the step is now **host-bound**: ~498 ms GPU sync (11% of wall) vs **1744 ms CPU dispatch over ~80K kernel launches** (GPU idle ~89%). `TRACE_ELEM` attributed the launch storm to the **ml8 quant prologue**, dominated by `ml8_e4m3_sim.e4m3_roundtrip` — a ~40-op pure-torch **software emulation of e4m3 rounding**, called ~400×/step — plus the per-call `layer_from_components` rebuild. That raised the bigger issue: we maintain **three parallel kernel implementations** of the same fp8/e4m3/Hadamard math (training torch+Triton, inference HIP + AOT-Triton, KV HIP+Triton), which is a faithfulness-debt generator. **Decision: unify the kernel substrate on Triton** (already the cross-runtime through-line — the inference GEMM `gemm_ml8.py` is AOT-compiled Triton vendored from aiter, same a8w8-blockscale source the trainer's `ml8_gemm` uses).

---

## 1. Aggregated recon findings (2026-06-14)

Source docs: `docs/superpowers/recon/2026-06-14-{triton,aiter,rocm-therock}-rdna4-fp8.md`.
Hardware: **R9700 = RDNA4 = gfx1201** (dev/train), **MI300X = CDNA3 = gfx942** (datacenter). **gfx1250 is a newer arch we do NOT have** — gfx1250-only upstream work is not for us.

### Substrate decision: Triton/aiter, confirmed by elimination

The ROCm libraries are **not** a viable RDNA4 fp8 path today, so there is no library-level escape hatch to reconsider:

- **hipBLASLt #6365** ("Add gfx1200/gfx1201 to FP8 architecture list") — **OPEN, +1/−1, HW test unchecked, stale-bot threatening auto-close.** Split from #5462 (open); #5455 (CK+MIOpen RDNA4 fp8 tile filtering) closed-unmerged.
- **Tensile #7192** — gfx1201 resolves to a nonexistent `gfx1200.dat` with no tuned configs → **this is exactly the skinny-tile/wrong-config failure that pushed us off rocBLAS.** OPEN, no fix, reported on ROCm 7.2.1/R9700.
- **Composable Kernel** — the tree now *has* WMMA fp8 **GEMM** instances (`device_gemm_wmma_universal_f8_*`), but **no gfx1201 tuning/validation**. Credible *future* escape hatch; unproven now.
- **TheRock** — gfx1200/gfx1201 are first-class *build* targets; the blocker is upstream fp8 enablement+tuning, not the build.
- **No ROCm ≤ 7.2.x makes gfx1201 fp8 solid.** Ecosystem corroborates bypassing the libs on RDNA4 (TransformerEngine, vllm, and a hand-written RDNA4 WMMA fp8 kernel hitting **40.8 TFLOPS ≈ 3.8× over dequant+hipBLAS**).
- (NOTE: ROCm consolidated hipBLASLt/Tensile/rocBLAS/MIOpen/CK into the **`ROCm/rocm-libraries` monorepo**; the PR numbers live there.)

### Triton (clone `~/GitHub/triton` @ `4768da5`, 2026-05-16; 150 behind `origin/main` @ `007ef1530`)

- main is **3.8.0-dev, no 3.8.0 tag** → every relevant commit is **unreleased**; a bump means pinning a main SHA. Latest real tag = v3.7.0.
- **#10458 `bb5acbe59` (2026-06-04) "Fix fp8 conversions on archs not natively supporting fnuz" — the standout. Landed AFTER our HEAD.** Without it, Triton can emit **wrong fp8** on RDNA4 (OCP e4m3/e5m2 vs fnuz). For a quant trainer this is a latent correctness bug → **must verify** (see §4).
- **Perf gap (the ~20%) is NOT addressed by any in-range commit.** fp8 WMMA instruction-selection for gfx12 already exists in our HEAD; the RDNA4 InThreadTranspose/wide-LDS lever was enabled pre-HEAD (#10185). A bump likely will **not** recover the 20% — needs profiling.
- **AOT `--target` bug (#170) is NOT fixed upstream.** `tools/compile.py` on main is byte-identical to ours and still forces GPU init during C-stub emission; upstream tracking issue **#4219 open, no fix**. **We carry our own `compile.py` patch regardless of which SHA we pin.**
- Provenance: #10290 (kWidth kPack>1) is gfx942/CDNA3-only (helps MI300X, not RDNA4); the gfx1250 cluster and #10202 (NVIDIA TMEM) are not for us.

### aiter (clone `~/GitHub/aiter`; vendored @ `9c79a5b59` 2026-05-25; clone @ `69cbe3ff8`; latest release v0.1.15.post1, 2026-06-08)

- **Shape coverage is poor for us.** Of our 16 (N,K) combos from {2560, 4096, 8192, 9216}, **exactly ONE — (8192,8192)** — has a tuned per-shape `gfx1201-GEMM-A8W8_BLOCKSCALE-N=..-K=..json`. The other 15 fall back to the generic `gfx1201-GEMM-A8W8_BLOCKSCALE.json`, whose large-M path collapses onto a single `"any"` config. There are **zero** files for N=2560, N=9216, K=2560, K=9216. **Fix is data-only** (generate/vendor our own tuned JSONs) — no kernel/Triton change.
- **No backward GEMM exists in aiter — anywhere** (main, branches, PRs). No dgrad/wgrad/autograd in the gemm tree. (The web's "cast_transpose_bgrad fp8 training" is **ROCm/TransformerEngine**, not aiter.) → **Confirms our hand-written `_scaled_mm` backward (MAD-290) was the only path**, and we author the QAT backward ourselves (or crib from TE).
- **Reusable upstream:** fused fp8-e4m3 **quant** prologues exist and are arch-generic (`quant/fused_fp8_quant.py`: RMS/gated/silu_mul + e4m3 group/per-tensor). **Fused rotation+quant does NOT exist** (all "rotat*" hits are RoPE). **LUT-weight fp8 does NOT exist.** → the rot+quant prologue and LUT path are **net-new, ours to build**.
- `is_fp8_avail()` now includes gfx1200/gfx1201 (the old silent-fp32-fallback bug is fixed in our line).
- **Coupling is fine:** aiter requires Triton **≥ 3.6.0**; we're on 3.7.0. Re-vendoring newer aiter does NOT force a Triton downgrade, and a Triton *bump* (for #10458) stays compatible.
- In-flight: merged #3484 (gfx1201 Qwen3-8B tuning), #3611 (padded-K bpreshuffle); open/relevant #3343 (gfx1201 blockscale fallback+tuning), #2350, #1829 — all **forward-inference tuning, none add backward.**

---

## 2. What we build (substrate components)

| # | Component | Lang | New or reuse | Notes |
|---|---|---|---|---|
| **A** | **"Stack" kernel: fused rotation (FWHT) + e4m3 quant prologue** (fwd **and** bwd) | Triton | **NET-NEW** (no upstream) | The unification phase-1 kernel. Replaces `e4m3_roundtrip` emulation + the dense/FWHT rotation in one kernel. Mirrors deployment `ml8.cu::ml8_fused_rot_quant_kernel` **by numerics**. Bit-exact TDD vs the existing `e4m3_roundtrip` torch oracle. Consumed by training now (JIT + autograd); AOT'd to inference later (→ replaces the HIP kernel). |
| **B** | **e4m3 quant primitive** (per-row + per-tensor, RNE) | Triton | NET-NEW (crib aiter `fused_fp8_quant.py` pattern) | One primitive, reused by A's forward, the backward `_quant_tensorwise`, and activation quant. May be folded into A. |
| **C** | **CUDA/HIP graph capture of the training micro-step** | torch/HIP | NET-NEW | **Orthogonal** host-bound killer — collapses dispatch for ALL ~80K launches, not just the quant region. Additive on top of A/B. Requires hoisting CPU-side control flow (`layer_from_components` rebuild, `_validate_gidx_once`, weakref pack cache, any `.item()`) out of the captured region + a static memory pool. Static shapes already hold (fixed micro-step). |
| **D** | **gfx1201 tuned a8w8 configs for our shapes** | JSON (data) | NET-NEW data | Generate tuning JSONs for our (N,K) so we stop hitting the generic fallback. Addresses part of the ~20% perf gap. No kernel change. |
| **E** | **Triton bump → main SHA ≥ #10458** + carry AOT `--target` patch | toolchain | enabling | fp8 OCP correctness fix. AOT patch is ours to maintain (upstream #4219 unfixed). |
| **F** | `layer_from_components` per-call rebuild caching | Python | cheap | Centroids constant within a step's fwd+recompute → cache the built layer per step. Small; fold into A/C work. |

**Graphs vs the stack kernel — they stack, they don't compete:**
- **A/B** reduce launch *count* in the quant region (~20-25K of 80K launches) AND give a deployment-faithful, AOT-able kernel.
- **C** removes dispatch *overhead* for the whole step (toward the GPU-bound floor) but produces no kernel and no faithfulness asset.
- Estimated wins (from the profile): A/B alone ~4.3s → ~3.8-4.0s; C (whole-step) ~4.3s → ~2.8-3.2s. Do A/B first (faithful asset + de-risks C by removing python-side ops), then C.

---

## 3. Unification / faithfulness model

- **One Triton kernel source → JIT (training) + AOT (inference)**, bit-exact by construction instead of by testing. This kills the faithfulness-drift class that burned the act-replay work (training torch-emulation vs deployment HIP).
- The **GEMM is already unified** (AOT-Triton `gemm_ml8.py` ⇄ trainer `ml8_gemm`, same a8w8-blockscale source). The duplicated parts are the **prologue/quant** kernels (HIP `ml8_fused_rot_quant_kernel` / `ml8_quantize_activations_kernel` for inference vs torch `e4m3_roundtrip`/`quantize_act_per_row` for training). Component A collapses those onto one source.
- **Keep hand-HIP only where Triton can't win** — notably `gemv`/decode (batch=1), where Triton is weak and inference latency is non-negotiable. Pragmatic, not dogmatic.
- **aiter-CK as a future MI300X escape hatch** only if a microbench proves it (CK is currently untuned for gfx1201; for MI300X its CK/ASM kernels are the absolute-peak fp8 path).
- **KV kernel is a separate, third system** (different bit-layout `ml8-4-kv` 5bpv sign+mag) — out of scope for this work.

---

## 4. Open questions / to verify (measure, don't assume)

1. **Does our current Triton (`4768da5`) actually miscompute fp8 on RDNA4?** #10458 says it *can* (fnuz vs OCP). But turbo4_fp8 PPL/NIAH validation passed (f16-equivalent), so we're probably not hitting the broken path — **verify before/after #10458** rather than assume either way.
2. **Where do the TFLOPS come from — microbench at our real shapes on R9700:** Triton-a8w8 (old vendored) vs (Triton ≥#10458) vs (+our tuned gfx1201 configs) vs `torch._scaled_mm`. This is the substrate-confirming measurement and tells us if D closes the ~20% gap.
3. **CUDA-graph feasibility:** enumerate the host syncs / dynamic allocations in the micro-step that would block capture (the prereq list for component C).

---

## 5. Next steps (ordered)

1. **(done — this doc)** bank the aggregated findings + build items.
2. **Write the formal spec** for the unification epic. Proposed phase structure:
   - **Phase 0 — currency + microbench (gated, measured):** bump Triton ≥ #10458 + carry AOT patch (E); generate gfx1201 tuned configs (D); microbench (§4.2) + fp8 correctness verify (§4.1). Output: substrate confirmed on data, free perf banked.
   - **Phase 1 — the stack kernel (A/B):** build the fused rot+quant prologue Triton kernel (fwd+bwd), bit-exact TDD vs `e4m3_roundtrip`, wire into the trainer, re-profile. + rebuild caching (F).
   - **Phase 2 — graph capture (C):** hoist CPU control flow, static mem pool, CUDA-graph the micro-step; re-profile.
   - **Phase 3 — AOT to inference:** AOT-compile the Phase-1 prologue kernel into llama.cpp, replace HIP `ml8_fused_rot_quant_kernel`, equivalence-gate (PPL/NIAH). This is where unification pays off.
3. Execute Phase 0, then re-decide scope of 1-3 on the measured numbers.

**Holding the spec until the user says go.** This doc is the capture; the spec is next.
