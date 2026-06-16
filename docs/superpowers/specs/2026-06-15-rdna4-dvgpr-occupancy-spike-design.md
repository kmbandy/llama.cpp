# RDNA4 dynamic-VGPR occupancy de-risk spike — design

**Date:** 2026-06-15
**Epic:** MAD-293 (RDNA4 fp8 GEMM occupancy unlock) · follows **MAD-304** (dyn-VGPR arming proven)
**Phase:** 2 (de-risk spike). The real fp8 GEMM dyn-VGPR implementation is a **separate follow-on plan**, not this one.

## Goal

Prove the **dynamic-VGPR occupancy lever** on gfx1201 (RDNA4 / R9700) compute: a kernel
that launches at a *small* VGPR block, `s_alloc_vgpr`s up to a GEMM-scale register count,
runs a real fp8 WMMA accumulate on the grown registers (verified bit-faithful vs a CPU
oracle), shrinks back down, and demonstrates **more resident waves per SIMD** than a
static-VGPR twin. Exit = the lever demonstrably delivers occupancy *and* WMMA works under
the allocated region.

## Why this phase exists (what is and isn't proven)

MAD-304 proved **arming**: setting `COMPUTE_PGM_RSRC2` bit 6 (`DYNAMIC_VGPR` on GFX12) via
raw PM4 launches a gfx1201 wave with `STATUS[30] (DYN_VGPR_EN) = 1`. That proves the wave is
*in* dynamic-VGPR mode. It does **not** prove:

1. that a wave can actually `s_alloc_vgpr` more registers at runtime and use them correctly;
2. that launching at a small block lets **more waves** go resident (the actual occupancy win);
3. that a real fp8 WMMA produces correct results when its fragments live in the grown block.

This spike closes all three before any investment in a full GEMM rewrite. It is a *mechanism*
proof, not a throughput measurement — so it is independent of GEMM shape / roofline regime.

## Verified feasibility (pre-design checks, on this exact toolchain)

- `s_alloc_vgpr <imm>` and `s_alloc_vgpr <sreg>` **assemble** for gfx1201
  (`/opt/rocm/llvm/bin/clang -mcpu=gfx1201`). There is **no `s_dealloc_vgpr`**; the wave
  **shrinks by `s_alloc_vgpr`-ing back down** to a smaller count.
- The fp8 WMMA op is `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
  (the one the verified kernel `spike/gemm_wmma_raw_intrinsic_verified.hip` uses): operands
  2×int32 per lane (8 fp8 bytes each), accumulator/result **v8f32 per lane**, wave32.
- The raw-PM4 compute-dispatch vehicle is proven and recoverable (MAD-304, `spike/dvgpr_pm4/`).

## Architecture

Reuse the proven `spike/dvgpr_pm4/` PM4 dispatch substrate (vendored kfdtest PM4 encoder +
the hand-rolled ring/alloc layer). Two raw-ISA kernels share one WMMA core, dispatched by an
extended harness that runs both back-to-back and reports the A/B.

- **dyn kernel:** `PGM_RSRC1` small block (32 VGPRs) + `RSRC2` bit 6 = 1 (the proven arm).
  Flow: occupancy probe (at the small block) → long busy-wait → `s_alloc_vgpr N` → load fp8
  fragments → WMMA accumulate → store v8f32 result → `s_alloc_vgpr 32` (shrink) → `s_endpgm`.
- **static twin:** `PGM_RSRC1 = N` VGPRs, bit 6 = 0. Same probe + WMMA, but reserves N for its
  whole lifetime → occupancy capped by N.

The only intended difference between the two is the launch VGPR footprint (small+dynamic vs
static-N), so the occupancy delta is attributable to the lever.

### Parameters (defaults, tunable in the plan)

- **Small launch block** = 32 VGPRs (`PGM_RSRC1.VGPRS = 4`). High occupancy.
- **Alloc-up target N** = 128 VGPRs default — a representative fp8-GEMM accumulator footprint;
  must be ≥ the lifted WMMA sequence's max VGPR. Tunable.
- **Static twin** reserves the same N (128) for apples-to-apples. Optionally re-runnable against
  the *current* `gemm_wmma.hip`'s measured VGPR count to anchor the delta to the real kernel.

### Compiler-as-WMMA-oracle (emission approach)

LLVM will not emit dyn-VGPR codegen for `amdgpu_kernel` (`spike/FINDINGS.md`), so the WMMA is
produced by the compiler and the dyn-VGPR envelope is hand-stitched:

1. `wmma_seed.hip` — a tiny HIP kernel doing **one** `…wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
   Compile to gfx1201 assembly.
2. `extract.sh` — disassemble and lift the exact `v_wmma_*` instruction plus the fragment VGPR
   layout (which VGPRs hold A = 2×i32, B = 2×i32, C/D = v8f32 per lane).
3. Hand-assemble both kernels so the lifted WMMA sequence's VGPR numbers fall **within the
   grown block** (after `s_alloc_vgpr N`), while the probe/busy-wait use only the small block.

## Components (new dir `spike/dvgpr_occ/`, reusing `spike/dvgpr_pm4/vendor/`)

| File | Responsibility |
|---|---|
| `wmma_seed.hip` | Compiler WMMA oracle: one fp8 WMMA, source of the real instruction + fragment layout. |
| `extract.sh` | Disassemble `wmma_seed`, emit the WMMA instruction sequence + VGPR map for hand-stitching. |
| `occ_kernel.s` | The hand-stitched raw-ISA kernel. Dyn vs static selected by assembler `-defsym` (block size + bit6 emitted by the harness via RSRC, body shared). |
| `fp8_oracle.{h,cpp}` | CPU fp8 e4m3 16×16×16 matmul reference for WMMA verification. |
| `occ_dispatch.cpp` | Extends the dvgpr_pm4 harness: alloc occupancy buffer `[live,maxlive]` + WMMA A/B/out buffers, launch a large grid, run static then dyn, read `maxlive` + WMMA result, compare to oracle, print the A/B table. |
| `build.sh` | RAM-capped build (assemble seed + kernel, compile harness, link `libhsakmt.a`). |
| `RESULT.md` | Findings + the A/B occupancy table + verdict. |

## Occupancy measurement

Atomic **max-resident-waves** probe — self-contained, no external counters/tooling:

- Global buffer `{ uint32 live; uint32 maxlive; }` (host-visible GTT, uncached).
- Each wave: `global_atomic_add(live, +1)` (returns old) → `global_atomic_max(maxlive, live_new)`
  → **busy-wait** a fixed, long interval (so many waves overlap) → `global_atomic_add(live, -1)`.
- Launch ~2048 single-wave workgroups (≫ achievable occupancy) so the scheduler packs as many
  as the VGPR footprint allows; `maxlive` ≈ peak concurrent waves.
- The busy-wait dominates and the WMMA excursion is brief → the measurement reflects real
  *small-block* occupancy and gives an honest read on staggered `s_alloc` behavior (it does not
  rig a best case where only one wave ever allocs up).

**Expected:** dyn (32-VGPR launch block) packs materially more waves/SIMD than static
(N-VGPR), e.g. dyn ≥ 2× static, trending toward the small-block theoretical max.

## WMMA-under-dyn verification

Known A/B fp8 fragments are supplied from a buffer; the kernel runs the WMMA on the
`s_alloc`'d registers and stores the v8f32 result. `fp8_oracle` computes the same
16×16×16 e4m3 matmul on the CPU. Match (within fp accumulation tolerance) proves the
dynamically-allocated registers carry correct WMMA state — i.e. `s_alloc_vgpr` is not a no-op.

## Success gates

1. **Functional:** dyn-kernel WMMA result matches the CPU oracle.
2. **Occupancy:** `dyn maxlive` clearly exceeds `static maxlive` (target ≥ 2×, or near the
   small-block theoretical max for a 32-VGPR block).
3. **Stability:** no queue hang / GPU reset; both variants complete cleanly (supervised run).

A clean miss is still informative: if occupancy does **not** rise (e.g. concurrent `s_alloc`
serializes via `FWD_PROGRESS`), that is a real, documented finding that reshapes the Phase-3
GEMM strategy rather than a failure to hide.

## Phase-3 constraint (pinned here, implemented later)

The follow-on GEMM TFLOPS gate **must** benchmark **compute-bound, training-shaped** matmuls
(large M/N/K, including backward **dgrad/wgrad**), never memory-bound decode shapes. Occupancy
only converts to TFLOPS in the compute-bound, latency-stalled regime (MAD-300: the wall was
occupancy-gated byte-wise LDS fragment reads). This spike is regime-independent; Phase 3 is not.

## Risks

- **`s_alloc_vgpr` stall under concurrent alloc** — the SQ `FWD_PROGRESS` bit guarantees ≥1
  wave proceeds (no deadlock) but may cap concurrency at peak footprint. This is precisely what
  the probe measures; it is not assumed away.
- **VGPR-number alignment** — the lifted compiler WMMA sequence must reference VGPRs inside the
  grown block; mitigated by choosing N ≥ the WMMA's max VGPR and verifying via disassembly.
- **Raw PM4 on R9700** — proven safe/recoverable in MAD-304; supervised, the gfx12 node is the
  headless compute GPU (graphics clients also attached → a hang could blip the desktop).
- **Atomic correctness on host-visible GTT** — use device global atomics; the EOP fence +
  uncached mapping ensure CPU sees final `maxlive`.

## Out of scope (explicitly not this plan)

- The full fp8 GEMM dyn-VGPR rewrite and its TFLOPS-toward-307 measurement (separate plan).
- `amdgpu_cs_chain` compiler codegen of dyn-VGPR (a Phase-3 option, not needed for the spike).
- Backward (dgrad/wgrad) and consumer integration.
