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
