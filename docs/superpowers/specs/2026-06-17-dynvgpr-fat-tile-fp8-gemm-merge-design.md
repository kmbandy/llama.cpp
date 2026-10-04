# Configurable dyn-VGPR fat-tile fp8 WMMA GEMM (the "merge") — Design

**Date:** 2026-06-17  **Ticket:** MAD-305  **HW:** AMD R9700 / gfx1201 (RDNA4), wave32, ROCm 7.2.3
**Branch:** `sync/upstream-2026-06-09`

## Goal

One hand-written, **fully `-defsym`-configurable** gfx1201 fp8 WMMA GEMM that fuses the two
half-solutions we proved complementary:

- the **HIP kernel's** hipcc-quality WMMA inner loop (good codegen, but capped at 256 VGPR), and
- the **PM4 micro-batch vehicle's** dynamic-VGPR fat accumulator tile (breaks the 256-VGPR cap, but
  had poor hand-asm codegen),

so a single kernel can hold a large register-blocked accumulator tile **and** issue WMMAs at
hipcc-grade density — the only path measured to climb past the 155-TF feed wall toward the 250-TF
pitch bar, on **training (wgrad) shapes**.

## Why this, why now (the settled measurements behind it)

Decisive `ISSUE_PROBE` result (KG `2601d691`, 2026-06-17), measured on a bit-faithful clone of the
155-TF winner `gemm_fp8_lever<128,128,2,2,32,AFEED=0,DBUF=1>`:

1. **The 54% / 155-TF wall is ISSUE-PORT-bound on feed instructions, not latency.** Occupancy sweep
   (`__launch_bounds__` minWaves, fixed inner loop) is flat at 8.3 WMMA/cyc across minW 4–12 at
   ~20 waves/SIMD; a deeper A+B software pipeline made it *worse* (64–75 TF). Latency is already
   hidden; adding instructions to hide more backfires.
2. **The ceiling is NOT 155.** `NOFEED` (same kernel, same occupancy, feed instructions removed)
   = **284–289 TF = 94%**. The silicon will do ~290; the gap is purely feed-instruction density on
   the one VALU issue port.
3. **The lever is operand reuse.** A larger register-blocked accumulator tile reuses each loaded
   fragment more times → fewer feed instructions per WMMA. 4×4 reuses each fragment 4×; **8×8
   reuses 8×**, halving loads/WMMA. >256 VGPR ⇒ **dynamic-VGPR**, which is **armable on gfx1201 via
   raw PM4** (RSRC2 bit 6, MAD-304) — the basis of the existing micro-batch vehicle. Not locked;
   only the clean HIP/compiler path is.

ISA confirmation (hipcc `-save-temps` device `.s`, winner inner loop): hipcc issues **FM*FN=16
independent accumulator chains back-to-back**, B-loads released by fine-grained `s_wait_loadcnt
0x6→0x0`, **all address math hoisted** to two pointer increments per 32 WMMAs. This is a fixed,
transcribable pattern — not compiler black-box. The hand-asm vehicle's prior 64-TF (21%) feed-free
result is explained as **too few interleaved chains** to cover WMMA latency; the fat tile supplies
*more* independent chains (64 at 8×8) for free, fixing codegen and feed density at once.

## Architecture

Evolve, don't restart. The vehicle is the existing PM4 spike, which already provides dyn-VGPR
arming, the persistent atomic work-queue, the e4m3 oracle, and the `-Wa,-defsym` build matrix:

- **`spike/dvgpr_occ/occ_kernel_mbgemm.s`** — the kernel. Replace its inner loop with an
  `.rept`/`.macro` generator that emits the hipcc-transcribed pattern, parameterized to an FM×FN
  accumulator tile; keep its dyn-VGPR grow/ship/shrink + queue scaffold.
- **`spike/dvgpr_occ/occ_dispatch.cpp`** — the KFD/PM4 host harness: arms RSRC2 bit 6, pre-shuffles
  B (tile-major for `global_load_tr`), runs the per-tile chained-`wmma_ref` oracle, sweeps configs.
- **`spike/dvgpr_occ/build.sh`** — RAM-capped `-defsym` build of the config matrix.

### Inner-loop macro (the heart)

Per K-step, per wave, the macro emits:
1. **Feed**: load FM A-fragments (`AFEED`: `ds_read` from LDS, or direct) + FN B-fragments
   (`BFEED`: `global_load_tr` direct, or LDS-staged + reused), at `LDW` width (b64 or b128 = 2
   K-steps/instruction), with `PIPE` stages of prefetch ahead.
2. **Compute**: FM*FN `v_wmma_f32_16x16x16_fp8_fp8` over FM*FN independent v8f32 accumulators,
   issued back-to-back, each gated only by the precise `s_wait_loadcnt` for its operand.
3. **Advance**: pointer-increment address update (hoisted; no per-step multiply).

dyn-VGPR (`DYNVGPR=1`): `s_alloc_vgpr` grows the register window to hold the FM×FN fat accumulator
for the compute phase, then shrinks to `LEANV` between tiles. `DYNVGPR=0` statically reserves
(valid only while FM*FN*8 + overhead ≤ 256 VGPR).

### The full lever set (`-Wa,-defsym`)

| group | levers | purpose |
|---|---|---|
| tile / reuse | `FM`, `FN`, `TWM`, `TWN`, `TBK` | accumulator fat-tile (feed-density lever), cooperating waves, K-steps/tile |
| feed | `AFEED`, `BFEED`, `LDW`, `PIPE` | A via LDS/direct, B via tr/LDS-reuse, load width b64/b128, prefetch depth 0/1/2 |
| dyn-VGPR / queue | `DYNVGPR`, `LEANV`, `BATCH` | grow-shrink vs static, between-tile lean VGPRs, tiles per atomic grab |
| diagnostics | `NOFEED`, `PROFILE` | feed-free ceiling probe; in-kernel REALTIME phase timers |

## Data flow

Host pre-shuffles B to tile-major (trperm) once → device buffers (A row-major fp8, B shuffled,
per-row/col scales, C bf16). Persistent waves atomic-grab `BATCH` output tiles from the global
queue → (dyn-VGPR grow) → full-K reduction streaming A+B per the feed levers → scaled bf16 store →
(dyn-VGPR shrink) → grab next, until drained.

## Correctness & performance gates

- **G1 — bit-exact:** every built config matches the chained-`wmma_ref` oracle (err < oracle gate)
  at the correctness shape. Non-negotiable; a config that fails the oracle is not measured for perf.
- **G2 — macro fidelity (go/no-go for the whole approach):** config `FM=4,FN=4,AFEED=0,BFEED=0,
  TBK=32,DYNVGPR=0` (static, fits 256 VGPR) must reproduce hipcc's winner within ~5% (≥ ~147 TF /
  ~50%). This proves the hand-asm inner loop matches hipcc *before* the fat tile is involved. If it
  fails, the gap is pure scheduling (macro tuning), isolated from the tile.
- **G3 — the climb:** with dyn-VGPR + a fat tile (e.g. FM=8,FN=8 = 64 chains) + feed levers, beat
  155 TF on wgrad; stretch toward 250. Read %-of-peak and WMMA/cyc per config from the sweep.

Shapes: wgrad-shaped — `4096²×K{16384,32768}` and a realistic `4096×14336×K16384`.

## Configurability ↔ codegen-quality tension (managed, not ignored)

A fully general macro can emit a *correct* interleave that schedules worse than hipcc's hand-tuned
`s_wait_loadcnt` placement. Mitigation: the macro emits the waitcnt/interleave pattern **modeled on
the ISA already extracted**, and G2 gives an immediate, known-target fidelity check at 4×4. A config
that schedules poorly is a macro-tuning fix (waitcnt counts, issue order), not a redesign.

## Testing

- CPU: existing e4m3 oracle self-test stays green (`test_oracle`).
- Device (SUPERVISED, PM4): per-config oracle gate (G1) then perf; G2 as the first checkpoint; then
  the FM×FN + feed lever sweep (G3). PM4 runs are user-supervised; no auto-teardown of a hung queue.
- `NOFEED`/`PROFILE` configs available throughout to attribute any shortfall (feed vs compute vs
  per-tile overhead) without leaving the kernel.

## Risks / open questions

- **G2 may need macro iteration** to match hipcc's scheduling; expected, bounded (it's waitcnt/order
  tuning against a known-good reference).
- **dyn-VGPR grow/shrink cost** at large FM×FN could eat the reuse win; `PROFILE` + the `DYNVGPR=0`
  static path (up to 256) bracket it. If grow/shrink dominates, BATCH amortizes it.
- **Pool sizing under dyn-VGPR** must keep occ×fat-VGPR within the SIMD file to avoid the lockstep
  grow deadlock that wedged the box previously (KG `40cd2823`): cap occ×FATREGS ≤ file budget.

## Out of scope (YAGNI)

- MoE / ragged stream-K load-balancing (user retargeted to training/wgrad; the queue is kept for
  persistence, not for MoE shapes).
- A clean HIP/compiler dyn-VGPR path (settled: none on gfx1201; KG `0dbcb65f`). Hand-asm only.
- Production graph integration (this is the kernel + harness; wiring into llama.cpp is later work).

## References

KG: `2601d691` (ISSUE_PROBE result), `0dbcb65f` (dyn-VGPR check-first / settled state),
`40cd2823` (fat-file deadlock cap), `324cefb7` (305 ceiling + 155=LDS), `f50237e2` (resume).
Code: `bench/gemm_fp8_levers.hip` (ISSUE_PROBE + hipcc reference loop), `spike/dvgpr_occ/`.
