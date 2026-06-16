# RDNA4 dyn-VGPR GEMM occupancy de-risk — design

**Date:** 2026-06-16. **Phase 3 of MAD-293 (story MAD-305).** Builds on the
Phase-2 spike (`spike/dvgpr_occ/`, occupancy lever proven on gfx1201) and the
MAD-300 fp8 GEMM campaign findings.

**Hardware:** AMD R9700 / gfx1201 (RDNA4), KFD node 1, wave32. **ROCm 7.2.3.**

---

## 1. Purpose

Phase 2 proved the dynamic-VGPR *occupancy lever* on gfx1201: a wave that
launches lean (32 VGPR) and `s_alloc_vgpr`s up for a WMMA reaches full
hardware occupancy while a static twin is VGPR-capped. MAD-305 asks the next
question: **does that occupancy convert to TFLOPS in a real, training-shaped fp8
GEMM, toward the measured 307 TF WMMA ceiling?**

Before committing to a full hand-written ASM GEMM (the Phase-4 build), this spike
buys the answer cheaply. The greenlight gate is **directional line-of-sight**: we
do not need a final TFLOPS number from the spike, only a defensible "the lever
converts past hipBLASLt's 143 TF when stacked with the known feed-width fix."

## 2. The honest ceiling (why the target is reframed)

Phase 2's 3.2× does **not** transfer to a GEMM, and the spec is explicit about
this so no one over-claims downstream.

- Phase 2's 3.2× exists because the wave is **lean for ~all its life** (32-VGPR
  busy-wait) and the fat `s_alloc 128` burst is brief. 16 waves × 128 VGPR = 2048
  exceeds the ~1280–1536-VGPR/SIMD file; "maxlive 2048" is only physical because
  the waves are almost never simultaneously fat.
- A GEMM holds its accumulators (16 frags = **128 VGPR**) live across the *entire*
  K-loop. Heavy-accumulator compute therefore **cannot run at occ 16** (16×128 >
  file). The hard ceiling for 128-VGPR-live waves is ~occ 10 (empirical: Phase-2
  static-128 → 1280 resident = 10/SIMD).
- So dyn-VGPR's GEMM win is **peak-shaving, not 8→16.** The real kernel sits at
  occ ~7–8 because its *peak* footprint (accumulators **+** A/B fragments **+**
  addressing **+** epilogue convert/store staging, reserved for life) is larger
  than its steady-state K-loop footprint. dyn-VGPR runs the K-loop at the
  steady-state footprint and balloons to peak only for the short epilogue. **The
  win equals the peak-vs-steady VGPR gap.**

**Implication:** 307 is reached by **occupancy (peak-shaving) × feed-width
(raw-intrinsic wide reads)** stacked — dyn-VGPR is one of two multipliers. The
spike measures the occupancy multiplier's real slope so we can project honestly.

## 3. What we measure (two prongs + one cross-check)

### Prong 1 — does the matrix unit convert occupancy to throughput? (static only)

The prize hinges on the slope of throughput-vs-occupancy. This is the same curve
the whole "143 at occ ~8 → 307 at occ 16" thesis rests on; we measure it directly
on silicon, with **no dyn-VGPR needed**.

- A timed WMMA-throughput kernel: each wave loads A/B fragments **once** (compute
  isolation — no feed in the hot loop, like the 307 microbench), then runs
  `KDEPTH` iterations of `NACC` **independent** WMMA accumulations (ILP = NACC).
- Fix `NACC = 8` (the microbench's saturating ILP; footprint ~64–80 VGPR → reaches
  occ 16 statically). Sweep occupancy by setting `RSRC1.VGPRS` to a ladder of
  reservations {≈80, 96, 128, 160, 192, 256} → occ {16, …, 5}. Compute is
  identical across the sweep; only the reserved-VGPR knob (already wired as the
  `staticVgprs` arg) changes, so occupancy is isolated.
- Output: TFLOPS-vs-occupancy curve. **GREEN-1** if throughput rises materially
  from occ 8→16 (headroom exists). **NO-GO (full stop)** if it is flat by occ 8
  (matrix unit already saturated; no occupancy mechanism can help).

### Prong 2 — can dyn-VGPR realize occupancy over a long fat phase? (dyn vs static)

The serialization worry: Phase 2's fat phase was a brief burst; a GEMM holds the
fat footprint across many K-steps. Does the held `s_alloc` introduce a stall/
serialization penalty that Phase 2's short burst never exercised?

- Heavy chain: `NACC = 16` (128 VGPR accumulators, GEMM-representative; static
  occ ~8). A/B:
  - **static** — launch directly at footprint `F`, run the chain.
  - **dyn** — launch lean (32), `s_alloc_vgpr` to `F`, run the same chain, shrink.
  - Both measured by **wall-clock WMMA throughput**, sweeping `KDEPTH` (fat-phase
    duration).
- Output: throughput ratio dyn/static vs KDEPTH. **GREEN-2** if dyn ≥ static (no
  penalty) across realistic KDEPTH — i.e. holding the alloc over a long compute
  phase does not stall. Combined with Prong 1's curve and the real kernel's
  peak-vs-steady gap (cross-check below), this projects the GEMM win.

### Cross-check (Approach 2) — peak-vs-steady gap on the *real* kernel (HIP, no PM4)

Quantifies the actual win magnitude and sanity-checks Prong 1 on the real kernel.

- (a) **Occupancy→TFLOPS on the real kernel:** take `gemm_fp8_tiled`
  (raw-intrinsic), walk occupancy *down* from its natural ~8 via
  `amdgpu_waves_per_eu` / a `volatile`-anchored dead-VGPR pad, bench real GEMM
  TFLOPS vs occupancy through the normal HIP launch (no hang risk). Confirms
  Prong 1's curve shape with real memory traffic. (Can only show ≤8; the >8
  regime is what dyn-VGPR unlocks and is covered by Prongs 1–2.)
- (b) **Measure the peak-vs-steady VGPR gap:** disassemble the kernel
  (`llvm-objdump --mcpu=gfx1201`), count VGPRs live in the K-loop steady state vs
  the epilogue peak. The gap × Prong-1 slope = the projected dyn-VGPR GEMM win.

## 4. Measurement method

- **Timing:** host wall-clock of the PM4 submit→EOP-fence interval (the harness
  already polls this fence). Work is sized (grid × KDEPTH) so GPU-busy time
  dominates launch overhead (target ≳50 ms); a tiny-work baseline run is
  subtracted to remove fixed overhead. The directional gate does not need
  cycle-accurate timing. *Fallback if host noise is too high:* lane-0 reads the
  GPU shader clock (`s_getreg` HW_REG_SHADER_CYCLES, encoding lifted from a
  compiler seed) and writes the delta — same compiler-as-oracle method as Phase 2.
- **FLOPS:** `grid × KDEPTH × NACC × (16·16·16·2)` flops ÷ time.
- **Correctness gate:** one verification pass per variant stores the WMMA result
  and checks it against the CPU fp8 e4m3 oracle (reuse `fp8_oracle`,
  `frag_layout`). Timed passes need not store. This guarantees we are timing real
  fp8 WMMAs, not instructions the hardware elided.

## 5. Components / files (all under `spike/dvgpr_occ/`, reusing Phase 2)

| file | change |
|---|---|
| `occ_kernel.s` | Extend to a parameterized timed throughput chain: load A/B once; `NACC` independent accumulators; `KDEPTH` inner iterations; keep dyn/static `s_alloc` envelope; keep the one-pass oracle store. New `-defsym` knobs: `NACC`, `KDEPTH`, plus existing `DYNVGPR`. |
| `occ_dispatch.cpp` | Add a throughput driver: host-timed runs, FLOPS calc, the `RSRC1.VGPRS` occupancy sweep (Prong 1) and the dyn-vs-static heavy A/B over a KDEPTH sweep (Prong 2). Keep the existing correctness path. |
| `build.sh` | Assemble the kernel variants across the `NACC`/`KDEPTH`/`DYNVGPR` matrix; keep the RAM-capped harness build. |
| `gemm_occ_pad.hip` *(new)* | Approach-2 cross-check: real-kernel occupancy-down sweep + a `--dump-asm` path for the peak-vs-steady VGPR count. |
| `RESULT_P3.md` *(new)* | Curves, ratios, the projected GEMM win, GREEN-1/GREEN-2 verdict, reproduce. |

Reuses verbatim: the PM4 vehicle (`../dvgpr_pm4/vendor/`, pinned ROCR @ ba56a24c),
`fp8_oracle.{h,cpp}`, `frag_layout.h`, the §7.12 lane maps, the proven WMMA /
atomic / wait encodings.

## 6. Success gates

1. **GREEN-1 (prize exists):** TFLOPS rises materially across occ 8→16 (Prong 1).
   If flat by occ 8 → **NO-GO**, redirect campaign to feed/tiling.
2. **GREEN-2 (mechanism delivers):** dyn ≥ static throughput at GEMM-representative
   KDEPTH (Prong 2) — no serialization penalty over a long fat phase.
3. **Projection:** (real-kernel peak-vs-steady gap) × (Prong-1 slope) projects a
   credible path past 143 TF when stacked with feed-width. Directional, not exact.
4. **Stability:** no hang / GPU reset across all supervised runs.

GREEN-1 ∧ GREEN-2 ∧ projection-past-143 ⇒ greenlight the Phase-4 ASM GEMM build.

## 7. Risks & supervision

- **Supervised GPU runs:** the Prong-1/Prong-2 dispatches use raw PM4 on the
  headless gfx12 node — recoverable (MODE1 reset) but can blip the attached
  desktop. **STOP before each supervised run for explicit go**, exactly as Phase 2.
  The Approach-2 HIP cross-check uses the normal launch path (no hang risk) and
  may run unsupervised.
- **Encoding risk:** any new hang-risky encoding (shader-clock read, if used) is
  lifted from a compiler seed and verified vs `llvm-objdump` — never guessed.
- **Build RAM safety:** all builds RAM-capped via `systemd-run --user --scope -p
  MemoryMax` (host ~15 GB); never an uncapped build.

## 8. What this does / does not prove

- **Proves (if green):** the dyn-VGPR occupancy lever *converts to fp8 WMMA
  throughput* and the mechanism survives a GEMM-shaped long fat phase, with a
  measured win-magnitude projection. Decides go/no-go on the Phase-4 ASM GEMM.
- **Does not prove:** a final sustained GEMM TFLOPS number — that is Phase 4
  (the full tiled kernel with real feed, stacking occupancy × feed-width on a
  training-shaped matmul incl. dgrad/wgrad). This spike is regime-isolated.

## 9. Reproduce

```
./build.sh                                  # assemble the NACC/KDEPTH/DYNVGPR matrix + harness (RAM-capped)
timeout 30 ./occ_dispatch --prong1          # SUPERVISED: occupancy→throughput curve (static)
timeout 30 ./occ_dispatch --prong2          # SUPERVISED: dyn vs static heavy, KDEPTH sweep
./gemm_occ_pad --sweep                       # cross-check on the real kernel (normal launch)
```
