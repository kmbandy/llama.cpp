# RDNA4 fp8 GEMM — P0/P1 + dynamic-VGPR spike results

**Date:** 2026-06-15. **Epic:** MAD-293. **Story:** MAD-300.
**Hardware:** AMD R9700 / gfx1201 (RDNA4), 64 CU = 32 WGP, wave32.
**Spec:** `docs/superpowers/specs/2026-06-15-rdna4-fp8-gemm-occupancy-unlock-design.md`.
**Plan:** `docs/superpowers/plans/2026-06-15-rdna4-fp8-gemm-p0-p1-foundation.md` (T1–T8, fully executed).
**Code:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`.

---

## TL;DR

We hand-built a correct, unified, oracle-green fp8 **and** ml8-4-bit forward GEMM on gfx1201
that reaches **~90 TF** (fp8, 30% of the measured 307 TF WMMA ceiling, 0.6× hipBLASLt). We then
proved — by rigorous elimination, not assumption — that the remaining 90→307 gap is **occupancy
gated by VGPR pressure**, and that the single lever to break it is **RDNA4 dynamic VGPR
allocation (`s_alloc_vgpr`)**. The instruction is real RDNA4 silicon (it assembles on gfx1201,
is rejected on RDNA3), **but the software path to launch a wave in dynamic-VGPR mode does not
exist on the HIP/ROCr compute stack.** So 300+ is not reachable by a kernel-only edit today —
it requires a precisely-scoped upstream contribution. That gap *is* the AMD first-impression.

---

## The ladder (M=8192 unless noted, % of measured 307 TF ceiling)

| Stage | fp8 TF | %307 | ×hbl | Note |
|---|---|---|---|---|
| Raw WMMA ceiling microbench | **306.5** | 100% | — | back-to-back `mma_sync`, fragments loaded once (no mem in hot loop) |
| Naive (1 wave / 16×16 tile) | 13 | 4.2% | 0.08 | correctness-first baseline |
| **T4 LDS-tiled core** | **~90** | **~30%** | **~0.6** | int4-vectorized global→LDS copy was the lever (tiling alone: ~19 TF) |
| hipBLASLt (`torch._scaled_mm`) | 144–163 | ~50% | 1.0 | vendor reference (also occupancy-capped) |
| ml8 4-bit (T7), gs=64 → gs=9216 | 27 → 71 | 9–23% | — | same WMMA core; dequant is cheap, per-group fold is the cost |

Oracle (kernel ≡ `torch._scaled_mm` for fp8 / `reference_dequant_gemm` for ml8, within fp8)
stayed green at every step: 6/6 cases at the final commit.

---

## What we built (committed, oracle-green)

- `bench/wmma_peak.hip` — the 307 TF ceiling oracle (reproduced live at **306.5 TF**, 2.06× f16).
- `gemm_capi.h` / `gemm_wmma.hip` — a single templated `gemm_tiled_body<bool ML8>` WMMA compute
  core fed by **two front-ends**: plain fp8, and ml8 4-bit (nibble→fp8 LUT dequant in the
  global→LDS fill, identical WMMA). One stable C API (`rdna4_gemm_fp8_forward`,
  `rdna4_gemm_ml8_forward`). This is the unified inference + QAT-forward substrate.
- `test/oracle_harness.py` — correctness arbiter (3 fp8 + 3 ml8 cases, 5e-2).
- `bench/gemm_bench.py` — perf ratchet (% of 307, × hipBLASLt), fp8 + `--ml8`.
- `spike/FINDINGS.md` + `spike/gemm_wmma_raw_intrinsic_verified.hip` — the dynamic-VGPR
  reachability report + a verified raw-WMMA-intrinsic kernel (the P2 kernel basis).

---

## The convergence: the wall is occupancy, not feeding or read-width

Three independent, oracle-green measurements on the 90 TF core all point to the same wall:

1. **Feeding is already hidden (T5).** Double-buffering the K-loop *regressed* in all 4 variants
   (79/73 vs 95). A compute-isolation test — remove the per-tile global→LDS fill entirely —
   gained only **+3–11%**. Feed latency is ~90% hidden by the 8-wave occupancy. The plan's
   "occupancy/feeding overlap is the lever" premise was **disproven**.

2. **Read-width is not the wall (T6).** Disassembly showed the inner loop feeds 32 `v_wmma` from
   **64× `ds_load_u8`** (byte-wise LDS reads — rocWMMA's fp8 fragment-load lowering; entirely the
   B operand, A is already wide-read). But widening it *regressed*: a transposed-LDS layout drove
   the byte reads 64→0 yet dropped 87→50 TF (the transposed fill needs a per-byte scatter). A
   hand-built raw-intrinsic kernel **matched baseline** → rocWMMA's lowering is already optimal in
   the read path. The 307 TF microbench only hits the ceiling because it loads fragments *once*,
   outside the loop; our GEMM pays the (irreducible-at-this-occupancy) load every K-step.

3. **The ml8 fold confirms it (T7).** The per-K-group scale fold needs a second accumulator,
   pushing the kernel to **256 VGPR / occupancy 5 / spill** at the real gs=64 → 27 TF; throughput
   climbs monotonically to 71 TF as the group grows and the fold thins.

Static config sweeps confirm: **big tiles (needed for reuse) and high occupancy are mutually
exclusive under static VGPR allocation** — 256-tiles hit the 256-VGPR cap and spill to 6–33 TF.
This is exactly the tradeoff the spec named. We are at ~90 TF because the matrix unit idles at
8/16 waves; hipBLASLt's 143 is the same wall a notch higher.

---

## The lever and the verdict: dynamic VGPR (`s_alloc_vgpr`, ISA §3.3.3)

Start a wave **lean** (high occupancy), `s_alloc_vgpr` **up** for the big-tile compute phase —
breaking the tile-vs-occupancy tradeoff. Safe-probe findings (no GPU launched in dyn-VGPR mode;
all independently re-verified):

- **The instruction is real RDNA4 silicon.** `s_alloc_vgpr 64` assembles on gfx1201
  (`0xBE8053C0`, opcode 83); **rejected on gfx1100/RDNA3** ("instruction not supported").
- **Emit path = inline-asm only.** clang builtin `__builtin_amdgcn_s_alloc_vgpr` is undeclared;
  the IR intrinsic `llvm.amdgcn.s.alloc.vgpr` lowers to a *function call*, not the instruction;
  the assembler and HIP `asm volatile` both emit it correctly.
- **Launch in dynamic-VGPR mode is NOT reachable on HIP/ROCr.** Without it, `S_ALLOC_VGPR` is a
  hardware no-op (§3.3.3). LLVM 22's dyn-VGPR machinery (`.dynamic_vgpr_en`,
  `SI_CS_CHAIN_TC_*_DVGPR`, `HW_REG_DVGPR_ALLOC`) is bound to the `amdgpu_cs_chain` calling
  convention (refused under amdhsa); for a normal `amdgpu_kernel` the
  `amdgpu-dynamic-vgpr-block-size` attribute is silently inert; the `.amdhsa_*dynamic_vgpr*`
  `.kd` directives are rejected; HSA/ROCr/HIP expose no dyn-VGPR launch bit (grep-confirmed).

**Verdict: NEEDS-HSA-PATH.** Emit = YES; launch = NO. 300+ is not achievable by a kernel-only
edit. The occupancy-rise GPU spike is currently *un-runnable* (no launch path) — gated on the
upstream work below, or a deep raw-HSA/AQL dispatch experiment.

### Upstream contribution scope (the AMD first-impression)

1. **LLVM** — add `llvm.amdgcn.s.alloc.vgpr` intrinsic + `__builtin_amdgcn_s_alloc_vgpr` clang
   builtin (smallest, isolated first PR).
2. **LLVM/AMDGPU** — emit `.dynamic_vgpr_en` metadata for an `amdgpu_kernel` (not just `cs_chain`).
3. **HSA/amdhsa + ROCr** — define a dynamic-VGPR enable bit in the kernel descriptor; ROCr sets
   `DYN_VGPR_EN` at dispatch.
4. **HIP runtime** — expose it as a kernel attribute / `hipModuleLaunchKernel` flag.

The pitch writes itself: AMD's own RDNA4 silicon has a powerful occupancy lever its compute
software stack cannot reach today; we mapped exactly what's missing across all four layers and
built the fp8/ml8 GEMM that would consume it.

---

## Status / next (deferred to direction call)

- **Shippable now:** the unified ~90 TF fp8 + ml8 forward core (oracle-green). Useful for
  inference prefill and the QAT forward as-is; it inherits any later dynamic-VGPR speedup for free.
- **Not in this work (later plans):** backward pass (dgrad/wgrad — spec §4.1); consumer
  integration (llama.cpp `ggml-cuda` op + trainer ctypes/pybind); the dynamic-VGPR P2 (gated on
  the upstream path above or a raw-HSA experiment); final P0/P1 code review + branch finishing.

All work committed on `sync/upstream-2026-06-09`; baseline kernel at `eb8dce81`, T5/T6 reverted
per the no-regression ratchet, spike at `14995432b`.

---

## ADDENDUM (2026-06-15, later session): dynamic-VGPR reachability — the real map

The earlier "NEEDS-HSA-PATH / AMD-side change" verdict was built on a **wrong-bit** experiment.
A from-scratch re-investigation this session mapped the feature at every layer on the R9700
(gfx1201) silicon. Spike: `spike/dvgpr_probe/` (corrected probe) + `spike/dvgpr_pm4/` (the unlock).

**1. We had patched the wrong bit.** There are two `ENABLE_DYNAMIC_VGPR` enables:
gfx1250 = `COMPUTE_PGM_RSRC3` bit 17 (what #258 patched — *reserved* on gfx1201, correctly ignored);
**RDNA4 gfx1200/1201 = `COMPUTE_PGM_RSRC2` bit 6** (the bit that is `ENABLE_TRAP_HANDLER` on
GFX6–11). Both are in our installed ROCm 7.2.3 `AMDHSAKernelDescriptor.h` (lines 158, 230) and
documented for GFX120* in the 7.13.0-preview LLVM `AMDGPUUsage` doc.

**2. The chip-wide gate is already ON.** Live `umr` read (`spike/dvgpr_probe/patch_kd_rsrc2.py`
proved KD-patching rsrc2.6 is harmless; then `umr -r '*.*.SQ_DYN_VGPR'`):
`regSQ_DYN_VGPR = 0xff` → `WAVE_LIMIT=15`, `FWD_PROGRESS=1`, `MAX_BLOCK_ALLOC=7`. Dynamic VGPR is
**enabled at the SQ level**; the open kernel never writes this register, so the MES firmware (or a
silicon reset default) programs it.

**3. Layer map (all proven this session):**

| Layer | dyn-VGPR? | Evidence |
|---|---|---|
| Silicon (SQ) | ✅ enabled | `regSQ_DYN_VGPR=0xff`, WAVE_LIMIT=15 |
| `rsrc2.6` enable | ✅ open, settable | documented GFX120; flipped on silicon, no hang |
| LLVM (compute path) | ❌ won't emit | `amdgpu_kernel` + `amdgpu-dynamic-vgpr-block-size` → plain KD; bit 6 still "TRAP_HANDLER". dyn-VGPR codegen is `amdgpu_cs_chain`-only |
| ROCr | ❌ no concept | 0 `dynamic_vgpr` refs in ROCR-Runtime; passes the KD through verbatim |
| MES (compute/AQL) | ❌ drops the bit | rsrc2.6 set on an AQL dispatch → `STATUS.DYN_VGPR_EN=0`, no hang |
| PAL / Mesa (graphics) | ✅ works | PAL sets the same bit (`gfx12PipelineChunkCs.cpp:383`); MES arms it for work-graph queues |

**4. The unlock — raw PM4 on a KFD compute queue (lift Vulkan/PAL's method).** The card uses
whatever is in `mmCOMPUTE_PGM_RSRC2`; the only difference between paths is *who writes it*. HIP/AQL
lets the MES program it from the KD (drops bit 6); Vulkan/PAL — and ROCm's own
`libhsakmt/tests/kfdtest/src/Dispatch.cpp` — write it via `SET_SH_REG` directly in the PM4 IB, and
the CP consumes it verbatim (MES bypassed). `Dispatch.cpp:149-152` gates bit 6 behind
`if (m_FamilyId < FAMILY_GFX12)` (because bit 6 *became* `DYNAMIC_VGPR` on GFX12) — so the lift is
**one line**: `if (family >= GFX12 && dynvgpr) pgmRsrc2 |= (1u << 6);`. No LLVM, ROCr, or firmware
change. No prior art exists for this on RDNA4 compute (GitHub/forums/web checked).

**Build (in progress, `spike/dvgpr_pm4/`):** `PLAN.md` (approach + protocol); `probe.s` **done &
verified** (32-byte gfx1201 raw ISA: `s_getreg STATUS[30]` → `global_store`). Remaining:
`pm4_defs.h` (vendored PM4 packets + gfx12 reg offsets), `pm4_dispatch.cpp` (links the installed
`libhsakmt.a`), `build.sh`. **Run protocol:** baseline (bit clear → must read 0) then lift
(bit set → `DYN_VGPR_EN`). Risk: raw PM4 on the **headless** R9700 (GPU[1], `0000:42:00.0`) can
hang the compute queue / MODE1 reset (recoverable). Tooling: self-built `umr` 1.0.11 (pinned,
reviewed) at `~/GitHub/umr/build/src/app/umr`.
