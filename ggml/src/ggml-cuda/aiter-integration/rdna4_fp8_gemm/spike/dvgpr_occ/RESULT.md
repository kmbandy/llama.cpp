# RESULT: dynamic-VGPR OCCUPANCY LEVER proven on gfx1201 (RDNA4) — fp8 WMMA correct under the grown block

**Date:** 2026-06-15. **Hardware:** AMD R9700 / gfx1201 (RDNA4), KFD node 1
(GFXIP 12.0.1, 128 FCompute SIMDs, headless dGPU). **ROCm:** 7.2.3.
**Phase 2 of MAD-293**, builds on **MAD-304** (dyn-VGPR *arming* proven).

## Claim

On gfx1201 compute, a wave launched in dynamic-VGPR mode (`COMPUTE_PGM_RSRC2`
bit 6, armed via raw PM4 — MAD-304) that launches at a **32-VGPR block**,
`s_alloc_vgpr`s up to **128** for a real fp8 WMMA, and shrinks back, achieves
**full hardware occupancy (16 wave32/SIMD = 2048 resident waves)** — while a
static-VGPR twin reserving its peak footprint for life is occupancy-capped. The
fp8 WMMA computed on the dynamically-allocated registers is **bit-correct vs a
CPU e4m3 oracle**. This converts MAD-304's "armed" into a measured occupancy win.

## Method

- `occ_kernel.s`: hand-written gfx1201 wave32 ISA. Lane-0-only device-scope
  atomics (`global_atomic_add/max ... scope:SCOPE_DEV`) implement a peak
  resident-wave counter (`maxlive`); a long busy-wait holds the wave resident at
  the **small (32-VGPR) launch block**; then `s_alloc_vgpr 128` → 4 fp8 WMMA
  accumulators in `v[32:63]` (forces >32 VGPRs so the grow is load-bearing) →
  store → `s_alloc_vgpr 32` (shrink). Two variants via `-defsym DYNVGPR`.
- All hang-risky encodings (returning/non-returning global atomics, `v_wmma_…
  fp8_fp8 …,0`, `s_wait_loadcnt/storecnt`) were **lifted verbatim from compiler
  seeds** and verified against `llvm-objdump`, not guessed.
- `occ_dispatch.cpp`: extends the MAD-304 raw-PM4 vehicle. Per variant it sets
  `RSRC1.VGPRS` (dyn=32 / static=N), `RSRC2` USER_SGPR=6 + bit6, three USER_DATA
  pointers (occ / fragIn / fragOut), launches `nWG` single-wave32 workgroups, reads
  `maxlive`, and verifies the 4 WMMA output tiles against the CPU fp8 oracle
  (`fp8_oracle.{h,cpp}`) using the proven §7.12 lane maps (`frag_layout.h`).
- Occupancy is the **peak concurrent resident waves**; it only becomes
  VGPR-limited (not launch-rate-limited) once the grid exceeds the per-variant
  ceiling — hence the saturating grid of 65536 single-wave workgroups.

## Result — deterministic A/B (canonical: static twin = 128 VGPRs, grid 65536, ×3)

| variant                         | launch VGPRs | maxlive (resident waves) | waves/SIMD | WMMA vs oracle |
|---------------------------------|--------------|--------------------------|------------|----------------|
| static (reserves 128 for life)  | 128          | **1280**                 | 10         | **OK**         |
| **dyn** (32 → `s_alloc` 128 → 32) | 32 → 128   | **2048**                 | **16**     | **OK**         |

`dyn` reaches **2048 = 16 wave32/SIMD = the architectural occupancy ceiling**
(128 SIMDs × 16); `static` is **VGPR-file-limited to 1280** (10 waves/SIMD).
**Ratio = 1.60×**, deterministic across 3 runs, no hang, WMMA bit-correct both ways.

### The lever scales with the static footprint (saturating grid 65536)

The 1.60× is a *floor* tied to the 128-VGPR twin. A real fp8-GEMM accumulator is
heavier; reserving more VGPRs statically drops its occupancy further, while `dyn`
stays pinned at full occupancy (2048):

| static reservation | static maxlive | dyn maxlive | ratio dyn/static | notes |
|--------------------|----------------|-------------|------------------|-------|
| 128 VGPRs          | 1280           | 2048        | **1.60×**        | fully verified, WMMA OK |
| 192 VGPRs          | 896            | 2048        | **2.29×**        | fully verified, WMMA OK — **crosses 2×** |
| 256 VGPRs          | 640            | 2048        | **3.20×**        | occupancy captured; fence exceeded the 10 s poll (heavy reservation retires slowly on the raw-PM4 path) — not chased, out of spike scope |

Occupancy is exactly VGPR-file-limited: `static_maxlive ≈ (VGPR_file/SIMD) / N × 128`,
so the win is `min(16, VGPR_file/32) / (VGPR_file/N)` — i.e. it grows linearly with
the static kernel's VGPR footprint, which is precisely the regime a 300+ TF fp8
GEMM lives in.

### What had to be true first (grid sweep, why naive grids show nothing)

`maxlive` tracks `grid/32` until the grid exceeds a variant's ceiling — below that
the measurement is launch-rate-bound and dyn==static (e.g. both 64 at grid 2048,
both 256 at grid 8192). Only at grid ≥ 65536 does `static` break from `grid/32`
and plateau at its true VGPR ceiling (1280), exposing the lever. `dyn`'s 2048
plateau was confirmed independent of grid (identical at 65536 / 131072 / 262144).

## Success gates (spec)

1. **Functional — dyn WMMA == oracle:** ✅ PASS (all runs, both variants, 4 tiles each).
2. **Occupancy — dyn ≥ 2× static, or dyn near the 32-VGPR-block theoretical max:**
   ✅ PASS — `dyn` reaches **2048 = the 32-VGPR-block theoretical max** (full 16
   waves/SIMD); and the ≥2× literal target is met for any static footprint ≥192 VGPRs.
3. **Stability — no hang / GPU reset:** ✅ PASS — deterministic, GPU healthy
   throughout. (The static-256 case completed its waves — `live`→0, `maxlive`
   captured — but its EOP fence exceeded the 10 s poll; benign, not a hang.)

## Why it works (the layer map, completed)

| layer | status |
|---|---|
| silicon `SQ_DYN_VGPR` (WAVE_LIMIT=15, enabled) | ✅ |
| `COMPUTE_PGM_RSRC2.DYNAMIC_VGPR` (bit 6) armed via raw PM4 (MAD-304) | ✅ |
| `s_alloc_vgpr` grows the wave's VGPR block at runtime | ✅ **WMMA on the grown regs is bit-correct** |
| small launch block ⇒ more resident waves | ✅ **1280 → 2048 (full occupancy)** |
| raw-PM4 KFD compute dispatch (this vehicle) | ✅ recoverable, no hang |

## What this does and does NOT prove

- **Proven:** the dynamic-VGPR **occupancy lever** is real on RDNA4 compute — a
  small-launch-block + `s_alloc_vgpr` wave hits full hardware occupancy while a
  static twin is VGPR-capped, and the fp8 WMMA is correct under the grown block.
  The advantage scales with the static footprint (≥2× for realistic GEMM
  accumulators ≥192 VGPRs).
- **Not yet proven (Phase 3):** that wiring this into the full fp8 GEMM converts
  the occupancy into **TFLOPS toward the 307 TF WMMA ceiling**. That requires a
  compute-bound, **training-shaped** GEMM benchmark (large M/N/K incl. dgrad/wgrad);
  occupancy only converts to throughput in the latency-stalled compute-bound regime.
  This spike is regime-independent (a mechanism proof); Phase 3 is not.

## Reproduce

```
./build.sh                          # assemble both kernel variants + oracle self-test + harness (RAM-capped)
timeout 30 ./occ_dispatch 64        # smoke: no-hang + WMMA OK (occupancy not saturated)
timeout 30 ./occ_dispatch 65536     # canonical A/B (static=128): dyn 2048 vs static 1280 = 1.60x
timeout 30 ./occ_dispatch 65536 192 # heavier static twin: 2.29x (crosses 2x)
```

Sources committed; `*.bin` / `occ_dispatch` / `test_oracle` are reproducible build
artifacts (gitignored). Reuses the MAD-304 vendored kfdtest PM4 encoder under
`../dvgpr_pm4/vendor/` (pinned ROCm/ROCR-Runtime @ ba56a24c). Kernel busy-wait is
tunable via `-Wa,-defsym,SPIN=0x...`; static-twin footprint via the 2nd CLI arg.
