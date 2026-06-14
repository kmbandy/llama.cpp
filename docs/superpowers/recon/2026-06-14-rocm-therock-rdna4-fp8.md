# Recon: ROCm / TheRock RDNA4 (gfx1201) FP8 GEMM — is the library stack a viable escape hatch from Triton/aiter?

Date: 2026-06-14
Scope: read-only investigation. Question: has the ROCm stack (TheRock, hipBLASLt, Tensile, Composable Kernel) improved RDNA4 (gfx1200/gfx1201) FP8 GEMM enough to be a **library-level** substitute for our Triton/aiter FP8 path?

Hardware in scope: **gfx1201** (R9700, our RDNA4 training card) and **gfx1200** (also RDNA4). Not for us: **gfx1250** (newer arch we do not own — flagged as not-for-us throughout). Datacenter reference: gfx942 (MI300X, CDNA3) — already well-served.

---

## TL;DR / Bottom line

**Triton remains our best FP8 path on RDNA4 today.** Recent ROCm progress is real but **not yet sufficient** to flip the decision:

- The single most direct fix — **hipBLASLt PR #6365 "Add gfx1200/gfx1201 to FP8 architecture list" — is still OPEN, unmerged, and being repeatedly threatened with stale-bot auto-closure** (last touched 2026-06-09). It is a one-line change (+1/-1) with an **unchecked test plan** (FP8 GEMM dispatch never verified on hardware). Until it lands, hipBLASLt does not even *advertise* gfx1201 as FP8-capable.
- **Tensile lookup bug for gfx1201 is OPEN** (rocm-libraries #7192): a gfx1201 GPU resolves to a nonexistent `gfx1200.dat`, and on top of that there are no *tuned* FP8 logic files for either RDNA4 target. This is exactly the "skinny-tile / wrong-config" failure mode that motivated our Triton choice — and it is unfixed as of ROCm 7.2.x.
- **Composable Kernel has WMMA FP8 *GEMM* instance directories** (`device_gemm_wmma_universal_f8_*`) — a genuine change from the historically CDNA/MFMA-only situation — **but there is no changelog/release evidence they are tuned or validated for gfx1201**. The documented gfx12 WMMA work is for **FMHA (attention), not GEMM**. So CK is a *possible* future escape hatch, not a proven one.
- No ROCm version yet makes gfx1201 FP8 GEMM "solid" at the library level. The optimization plumbing exists (gfx12+ 8-bit `s_delay_alu` / Expert-Scheduling tweaks landed around ROCm 7.0–7.2), but the **enablement + tuning + tested-on-hardware** trifecta is incomplete.

**Recommendation:** stay on Triton/aiter for RDNA4 FP8. Track PR #6365 and issue #7192 as the gating signals; re-evaluate CK's `device_gemm_wmma_universal_f8_*` once a release explicitly tunes/validates them for gfx1201.

> Note on repo topology: ROCm has **consolidated hipBLASLt, Tensile, rocBLAS, MIOpen and Composable Kernel into the `ROCm/rocm-libraries` monorepo**. PR/issue numbers cited (#6365, #7192, #5462, #5455, #512) are in `ROCm/rocm-libraries`. The standalone `ROCm/composable_kernel` and `ROCm/hipBLASLt` repos are now read-only mirrors (their `develop` CHANGELOGs are still authoritative for release history).

---

## Per-repo findings

### 1. ROCm/TheRock (build/packaging)

| Item | Detail |
|---|---|
| `SUPPORTED_GPUS.md` (main) | **gfx1201 and gfx1200 are first-class build targets**: Linux shows Build Passing / Sanity Tested / Release Ready all green. Windows: Build Passing green, Sanity/Release-Ready not yet. |
| FP8 caveat | The support matrix is about **build/sanity**, not FP8 GEMM quality. No footnote asserts FP8 GEMM is tuned or fast on these targets. |
| gfx1250 | Newer arch — **not for us**, ignore. |

**Read:** gfx1201 is a fully supported *build* target (you get binaries that load and run), and TheRock release-readiness is not the blocker. The blocker is upstream library FP8 enablement + tuning, below.

### 2. ROCm/hipBLASLt (in rocm-libraries) — the direct FP8 enablement question

| Item | State | Detail |
|---|---|---|
| **PR #6365** "[hipBLASLt] Add gfx1200/gfx1201 to FP8 architecture list" | **OPEN, NOT merged** | Branch `gfx120x-hipblaslt-fp8`, base `develop`. Created 2026-04-11, last update 2026-06-09. 1 commit, **+1/-1** (literally adds the two arches to the FP8-capable list, enabling OCP FP8 E4M3FN/E5M2 via WMMA). Test plan ("verify FP8 GEMM dispatch on target HW", "no regression on gfx942/gfx950") is **unchecked**. Stale-bot has issued repeated auto-close warnings; no human review. Split out of #5462. |
| **PR #5462** "Add gfx1201 (RDNA4) support to MIOpen and hipBLASLt" | **OPEN** | Larger multi-project PR. Adds gfx12 to whitelists, FP32 intrinsic detection fix, FP8 enablement. Reported +52% FP16 / +12% BF16 ResNet50 training throughput on RX 9070 XT (note: FP16/BF16 numbers, **not** FP8 GEMM). |
| **PR #5455** "Add gfx1201 (RDNA4) support across CK Dispatcher, MIOpen, and hipBLASLt" | **CLOSED, not merged** (mergeable_state dirty) | Fixed CK Tile dispatcher "RDNA4 warp tile filtering for BF16/FP8/INT8". Notes FMHA JIT limits from gfx11↔gfx12 template distribution mismatch. Components were re-split into #5462. |
| CHANGELOG (hipBLASLt develop) | — | gfx12+ 8-bit (FP8/BF8/I8) **NN/NT** perf via `s_delay_alu`; 8-/16-bit **TN** via Expert Scheduling Mode — these gfx12+ micro-opts are present (associated with the 1.0.0 / ROCm 7.0 era). Mixed fp8×bf8 universal GEMM + weight-preshuffle GEMM added. **But no gfx1200/gfx1201 entry**, and the explicit "FP8/BF8 input → … output" support line is still scoped to **gfx94x** (CDNA). |

**Read:** the kernels/scheduling for gfx12 8-bit GEMM exist, but the **architecture-enablement gate (#6365) is unmerged** and the FP8 I/O support statement still names only CDNA. hipBLASLt does not yet present gfx1201 as a supported FP8 GEMM target in any release.

### 3. ROCm/Tensile (in rocm-libraries) — the skinny-tile / lookup question

| Item | State | Detail |
|---|---|---|
| **Issue #7192** "[rocBLASLt] gfx1201 GPU causes lookup of wrong Tensile file (gfx1200.dat)" | **OPEN** | Reported 2026-05-08 on ROCm **7.2.1**, R9700/gfx1201. gfx1201 → resolves to `TensileLibrary_lazy_gfx1200.dat`, which doesn't exist → SIGKILL on model load. Root cause: Tensile target-resolution maps gfx1201→gfx1200 fallback and **neither tuned `.dat` exists**. Assigned (slojosic-amd); **no fix PR, no fixed-in version**. Only comment points at an unrelated Ollama PR. |
| **Issue #512** "gfx1200 Windows optimized rocBLAS tensile logics" | **CLOSED** | gfx1200 on Windows (ROCm 7.0.0) using only fallback Tensile logics → very slow. Confirms RDNA4 Tensile tuning has been thin. Not FP8-specific. |

**Read:** This is the crux of our original rationale. On RDNA4, Tensile/rocBLAS either can't find a config (#7192) or falls back to untuned generic tiles (#512) — i.e. the "bad skinny-tile" behavior. **Unresolved through ROCm 7.2.1.** There is no evidence of tuned FP8 small-M/skinny-tile configs for gfx1200/gfx1201.

### 4. ROCm/composable_kernel (in rocm-libraries) — the CK escape-hatch question

| Item | State | Detail |
|---|---|---|
| WMMA FP8 **GEMM** instances | **Present in tree** | `library/.../gemm_universal` contains `device_gemm_wmma_universal_f8_f16_f16` and `device_gemm_wmma_universal_f8_f8_bf16` (WMMA path), alongside the CDNA `device_gemm_xdl_universal_f8_*` (MFMA). This is **new vs. the historical CDNA/MFMA-only state** — WMMA FP8 GEMM building blocks now exist. |
| CK CHANGELOG (develop / 1.2.0 for ROCm 7.2.0) | — | "Added WMMA (gfx12) support for **FMHA**." FP8 work is heavily **FMHA/attention** + MX FP8/FP4 (much of MX scoped to gfx950, i.e. CDNA). **No gfx1200/gfx1201/RDNA4 GEMM entry.** |
| `CK_USE_FP8_ON_UNSUPPORTED_ARCH` | — | Build flag to force-build FP8 instances on arches lacking native FP8 — signals FP8-instance gating is arch-aware, but doesn't prove gfx1201 GEMM tuning. |

**Read:** CK has crossed the key threshold of *having WMMA FP8 GEMM instances at all*. **However**, the only **documented/changelogged** gfx12 WMMA work is FMHA, not GEMM, and there is no release note tuning or validating these `wmma_universal_f8` GEMM instances on gfx1201. So CK is the **most credible future escape hatch**, but currently **unproven** for our case.

---

## Answers to the three definitive questions

**(a) Is rocBLAS/hipBLASLt FP8 on gfx1201 materially better now — could it beat Triton?**
No, not at the library level today. The enablement PR (#6365) is **unmerged and untested-on-HW**; the Tensile lookup/tuning for gfx1201 is **broken/absent** (#7192 open on 7.2.1, #512). The gfx12+ 8-bit scheduling micro-opts exist but ride on top of configs that aren't tuned for RDNA4 skinny/small-M tiles. There is **no evidence** hipBLASLt FP8 beats Triton on gfx1201; the more likely current outcome is dispatch failure or untuned fallback.

**(b) Does CK have RDNA4 FP8 GEMM instances (escape hatch)?**
Partially / unproven. CK **does** now ship `device_gemm_wmma_universal_f8_*` (WMMA FP8 GEMM) instances — a real step beyond CDNA-only. But the **documented** gfx12 WMMA enablement is for **FMHA**, and there is **no release evidence** the WMMA FP8 *GEMM* instances are tuned/validated for gfx1201. Treat as a candidate to pilot, not a ready escape hatch.

**(c) Which ROCm version first makes gfx1201 FP8 solid?**
**None yet (as of ROCm 7.2.x).** gfx12+ 8-bit GEMM scheduling opts appear around 7.0; CK WMMA-for-FMHA + fp8×bf8 universal GEMM around 7.2.0. But the gating enablement (#6365) and tuned-config/lookup fix (#7192) are **still open**, so no shipped version makes gfx1201 FP8 GEMM "solid." Earliest plausible candidate is whichever release first merges #6365 **and** ships tuned gfx1201 Tensile FP8 logic — not yet scheduled in anything we can cite.

---

## Does this change "Triton is our best FP8 path on RDNA4"?

**No.** Stay on Triton/aiter for gfx1201 FP8 GEMM. The specific weakness that drove that decision — immature rocBLAS/hipBLASLt FP8 GEMM hitting bad skinny tiles on RDNA4 — is **directly confirmed still-open** (Tensile #7192, plus untuned-fallback #512) and the enablement fix (#6365) **isn't merged**.

Watch items that would justify re-evaluating:
- **hipBLASLt #6365** merges AND its hardware FP8-dispatch test is checked off.
- **Tensile #7192** fixed AND tuned `gfx1201` FP8 `.dat` logic ships (kills the skinny-tile fallback).
- A CK release that **tunes/validates `device_gemm_wmma_universal_f8_*` for gfx1201** (not just FMHA) — this is the cleanest library escape hatch if it materializes.

### Stated unknowns / not verified
- Exact ROCm release tag any gfx12+ 8-bit opt first shipped in (CHANGELOG attribution is approximate; couldn't pull the consolidated ROCm changelog — docs domain fetch was blocked in this session).
- Whether CK's `wmma_universal_f8` GEMM instances actually compile+run correctly on gfx1201 vs. being gfx11-targeted or guarded off (no source-level arch-guard confirmation; GitHub code-search needed auth).
- Real FP8 GEMM perf numbers for hipBLASLt/CK on gfx1201 (none found; #5462's +52%/+12% are FP16/BF16 training, not FP8 GEMM).
- PR #6365's live state is ambiguous: API reports `state=open, merged=false, closed_at=null` (2026-06-09), yet stale-bot posted an auto-close notice 2026-06-08 — likely reopened or a bot/label race. Either way it is **not merged**.
- gfx1250 deliberately excluded (newer arch, not our hardware).

### Key references
- hipBLASLt FP8 enablement: `ROCm/rocm-libraries` **PR #6365** (open), **#5462** (open), **#5455** (closed).
- Tensile lookup/skinny-tile: `ROCm/rocm-libraries` **issue #7192** (open, ROCm 7.2.1), **#512** (closed, gfx1200 Windows untuned).
- CK instances: `ROCm/composable_kernel` `library/src/tensor_operation_instance/gpu/gemm_universal/` — `device_gemm_wmma_universal_f8_f16_f16`, `device_gemm_wmma_universal_f8_f8_bf16`; CK CHANGELOG 1.2.0 (ROCm 7.2.0) "WMMA (gfx12) support for FMHA."
- TheRock: `SUPPORTED_GPUS.md` (gfx1200/gfx1201 Linux Build/Sanity/Release-Ready green).
- Corroborating ecosystem signals (FP8 silently falling back / not enabled on gfx1201): `ROCm/TransformerEngine` #359, #520; `vllm-project/vllm` #28649; community hand-written WMMA FP8/MXFP4 kernel (`JohnTDI-cpu/rdna4-wmma-guide`, 40.8 TFLOPS, ~3.8x over dequant+hipBLAS) — i.e. people still bypass the libraries on RDNA4.
