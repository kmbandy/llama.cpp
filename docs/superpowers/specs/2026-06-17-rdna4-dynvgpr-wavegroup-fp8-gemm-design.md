# RDNA4 dyn-VGPR wave-group fp8 WMMA GEMM — Design

**Date:** 2026-06-17  **Ticket:** MAD-305  **HW:** AMD R9700 / gfx1201 (RDNA4), wave32, ROCm 7.2.3

## Goal

Reach toward **250-300 TFLOPS** on gfx1201 fp8 e4m3 GEMM by using the one capability that can break
the static HIP tile limit: **dynamic VGPR**, armed through raw **PM4** and consumed with hand-written
assembly (`s_alloc_vgpr`).

The key change from the prior single-wave micro-batch direction:

- Do **not** make one isolated wave extremely fat.
- Instead, make a **multi-wave workgroup** compute a larger logical tile while each wave grows to a
  moderate dynamic-VGPR footprint.
- Use LDS to share operand feed across the wave group so dynamic accumulators buy reuse without
  collapsing residency.

The success metric is not occupancy by itself and not latency hiding by itself. The success metric is
**higher WMMA/cycle** from fewer issued feed/address instructions per useful WMMA, while keeping enough
resident waves to feed the machine.

## Non-Negotiable Facts From Current Measurements

1. The fp8 WMMA silicon is capable of ~300 TFLOPS. `NOFEED` and raw WMMA probes reached ~284-305 TF.
2. The best real HIP fed kernel is ~161 TF on `4096^2 x K16384` and ~145 TF on
   `4096x14336 x K16384`. It is issue-port-bound, not latency-bound.
3. On RDNA4, fp8 WMMA and feed instructions share the SIMD VALU issue path. Operand feed and address
   math steal issue slots from WMMAs.
4. A larger accumulator tile is the clean lever because it increases WMMAs per operand load.
5. Larger-than-static tiles require dynamic VGPR. On gfx1201, the working path is raw PM4:
   `COMPUTE_PGM_RSRC2` dynamic-VGPR enable plus hand asm `s_alloc_vgpr`.
6. The existing single-wave PM4 micro-batch proves dynamic VGPR works, but it is occupancy-dead for
   uniform GEMM. It reaches only ~3.5 waves/SIMD at the tested footprint and cannot hide real feed.

Therefore, PM4/dyn-VGPR remains central, but the throughput vehicle must change from
**single-wave fat tile** to **wave-group logical fat tile**.

## Status Update: 2026-06-17 PM

The wave-group direction has advanced from design to a correct PM4 compute vehicle, but the
HIP-parity gate is not yet passed.

Completed and measured:

- **Phase 0 anchor:** HIP 4-wave 4x4 winner re-measured at **161.1 TF** on `4096^2 x K16384`
  (`8.34 WMMA/cyc`) and **145.4 TF** on `4096x14336 x K16384` (`7.53 WMMA/cyc`). These replace
  the older generic "155 TF" as the shape-specific G2 parity bars.
- **TGID pivot:** raw PM4 `DISPATCH_DIRECT` does not deliver `TGID_X` to any SGPR in this harness,
  despite `RSRC2.TGID_X_EN`. Grid-stride scheduling is abandoned.
- **Canonical tile distribution:** persistent workgroups now use a global atomic tile claim plus
  LDS broadcast to the 4 waves. This path is proven by `occ_kernel_wglds.s`.
- **LDS/barrier foundation:** raw-PM4 LDS allocation is encoded via
  `COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE = ceil(bytes/512) << 15`; LDS broadcast and workgroup
  barriers are verified.
- **Correct compute vehicle:** `occ_kernel_wggemm2.s` implements a 4-wave cooperative fp8 GEMM:
  atomic claim, A tile in LDS, B via `global_load_tr_b64`, per-wave static 4x4 WMMA accumulation.
  It is bit-exact through `512^2 x K2048`.
- **Phase 3 attribution:** the untuned fed kernel is only **~1.1-1.4 TF**, while `NOFEED` reaches
  **104 TF** on `4096^2 x K16384`. A double-buffer, B prefetch, and NOBAR are all flat. The current
  blocker is not correctness or basic compute viability; it is the coarse all-fragments wait on
  per-K-tile feed. The next lever is the HIP-style fine descending `s_wait_loadcnt` ladder.

Current decision point:

- Implement the fine `s_wait_loadcnt` release ladder on the B-prefetch kernel.
- If it moves substantially, continue PM4 scheduling toward G2 parity.
- If it remains near 1-3 TF, test a smaller per-wave tile such as 2x2 to determine whether the
  wave-group vehicle is still occupancy-limited at 4x4/192 VGPR.

## Thesis

The path to 250-300 TF is:

```
logical tile reuse > per-wave private tile reuse
```

A single wave growing to an 8x8 accumulator tile increases reuse but spends too much of the register
file per resident wave. A wave group can expose a larger logical tile, share A and possibly B through
LDS, and keep each wave's dynamic VGPR allocation moderate.

The design must reduce the feed issue density roughly like a fat tile, without admitting only a few
waves per SIMD.

## Architecture

### Vehicle

Use the existing PM4/KFD vehicle as the base because it is the only proven dyn-VGPR route on gfx1201:

- `spike/dvgpr_occ/occ_dispatch.cpp`: PM4 dispatch, RSRC2 dynamic-VGPR enable, config sweep, oracle,
  crash-survivable logging.
- `spike/dvgpr_occ/occ_kernel_wggemm2.s`: current correct 4-wave PM4 compute kernel.
- `spike/dvgpr_occ/occ_kernel_wglds.s`: atomic-claim + LDS-broadcast smoke foundation.
- `spike/dvgpr_occ/occ_kernel_mbgemm.s`: older single-wave micro-batch reference and source of the
  existing B-prefetch / wait-ladder macro patterns.
- `spike/dvgpr_occ/build.sh`: `-Wa,-defsym` matrix build.

The wave-group path no longer relies on `TGID_X`. Tile IDs come from the same global atomic queue
pattern already proven in the PM4 harness, then are broadcast to all waves in the workgroup through
LDS.

### Workgroup Shape

Initial target:

| parameter | first target | purpose |
|---|---:|---|
| waves/workgroup | 4 | match the HIP winner's cooperative structure |
| wave layout | `TWM=2`, `TWN=2` | two wave rows, two wave cols |
| logical C tile | `128x128` baseline, then `128x256` / `192x128` | increase logical reuse |
| per-wave accumulator | `4x4`, then `4x6` / `6x4`; **`2x2` for the occupancy/ceiling test** | tile-vs-occupancy |
| K tile | 32 (fp8 max; `K=32` is INT4-only, not an fp8 lever) | match proven HIP winner |
| A feed | LDS shared across all waves | proven HIP win |
| B feed | `global_load_tr_b64` (fp8's only transpose-load; b128 TR is 16-bit-only), then B sharing | reduce B feed issue count |

The first milestone is not the final 250-TF kernel. It is proving that a PM4 wave-group kernel can
match the 155-TF HIP winner while dynamic VGPR is armed and available.

### Dynamic VGPR Policy

Each workgroup launches lean and grows only around the compute phase.

1. Start with a lean register footprint for queueing, tile decode, and LDS fill.
2. After the workgroup claims a tile and reaches the compute phase, run `s_alloc_vgpr FATREGS`.
3. Compute the K stream with the chosen per-wave accumulator tile.
4. Store C.
5. Shrink back to `LEANV` before claiming the next tile.

The point is not to maximize `FATREGS`. The point is to choose the largest accumulator tile that still
keeps useful residency.

Safety rule: persistent pool sizing must leave slack in the SIMD register file. Prior hangs occurred
when `pool * per-wave VGPR` approached the full file. The host must compute and print an admission
estimate before dispatch and refuse known-dangerous cells.

### Operand Sharing

#### A Sharing

A is staged once per K tile into LDS by the workgroup. All waves read their A fragments from LDS.

This is mandatory for the first wave-group prototype because the HIP 155-TF kernel already proved it
is the best safe A path.

Current state: A sharing is implemented and bit-exact. A double-buffering alone did not improve the
fed PM4 wave-group kernel because the dominant exposed latency is B feed plus coarse waits.

#### B Direct Feed

Start with the proven `global_load_tr_b64` B feed from the preshuffled tile-major buffer.

This isolates the wave-group/dyn-VGPR change from B-layout risk.

Current state: direct B feed is implemented and bit-exact. Coarse `s_wait_loadcnt 0x0` before each
WMMA block leaves the fed kernel at ~1.4 TF. B register prefetch without the fine ladder is also flat.
The next required change is the descending release ladder so WMMAs begin as individual B fragments
arrive instead of waiting for all outstanding B loads.

#### B Sharing

Then test B reuse across wave rows. In a `TWM=2,TWN=2` block, waves with the same `wave_n` and
different `wave_m` consume the same B fragment at a given `kk`. If staging that fragment once in LDS
costs less than the duplicated `global_load_tr` instructions, it can reduce B feed issue density.

This must be treated as an experiment, not assumed: LDS reads and barriers also issue instructions.

#### ~~Wide B Feed (b128)~~ — STRUCK

`GLOBAL_LOAD_TR_B128` is **16-bit-data only** per RDNA4 ISA; fp8/wave32 has no wider transpose-load
than `global_load_tr_b64` (256 B/lane = one 16×16 fp8 tile). There is no instruction-count win to
chase here. Remaining B-feed levers: B sharing (above), the fine `s_wait_loadcnt` ladder, occupancy.

## Kernel Variants

The plan should build variants in this order:

1. **WG4 static 4x4 parity:** 4-wave workgroup, A in LDS, B trfeed, no dyn-VGPR growth required.
   This must reproduce the HIP 155-TF structure closely enough to be credible.
2. **WG4 dyn 4x4 parity:** same as above, but with PM4 dyn-VGPR armed and grow/shrink enabled.
   This measures grow/shrink overhead while holding the tile constant.
3. **WG4 dyn medium tile:** per-wave `4x6`, `6x4`, maybe `6x6` if admission remains reasonable.
4. **WG4 dyn logical-wide tile:** logical `128x256` or `192x128`, using wave-group sharing to reduce
   feed density without making each wave an 8x8 monster.
5. **WG4 + B sharing:** only after the base wave-group dyn path is correct and measurable. (b128 TR
   struck — fp8 has no wide transpose-load.)

## Go / No-Go Gates

### G1: Correctness

Every fed config must pass the existing chained fp8 WMMA CPU oracle before perf is trusted.
`NOFEED` configs are perf-only diagnostics.

### G2: HIP-Parity

The PM4 wave-group kernel at logical `128x128`, per-wave `4x4`, A-LDS, B-trfeed must reach within
~5-10% of the HIP winner. For the current target shape this means:

- hard pass: `>=153 TF` on `4096^2 x K16384`
- acceptable first pass: `>=145 TF` on `4096^2 x K16384`

If this fails, do not move to fatter dynamic tiles. The issue is scheduler/codegen parity, not tile
strategy.

Current state: G2 is not passed. The fine `s_wait_loadcnt` ladder is **done and flat (1.3 TF)** — B
feed was already coverable, so wait-polish is exhausted. **Critically, `NOFEED=104 TF` is itself below
the G2 bar (161):** this kernel's hand-WMMA-only ceiling is the binding wall (HIP's is ~272). G2 now
hinges on the **2x2 + NOFEED@2x2** test — if the `104` ceiling rises with the ~6→~16 waves/SIMD static
occupancy jump, it was occupancy-limited; if it stays ~104, the hand WMMA schedule must be tightened
(deeper unroll / interleave / loop-overhead removal) before G2 is reachable.

### G3: Dyn-VGPR Overhead

The same 4x4 kernel with dynamic grow/shrink enabled must not regress materially versus static.

If grow/shrink overhead is large, increase `BATCH` or keep waves grown across multiple claimed tiles,
but only under the deadlock guard.

### G4: WMMA/Cycle Climb

Medium/fat wave-group tiles must increase WMMA/cycle over the 155-TF baseline. A result that raises
occupancy but not WMMA/cycle is a failure for this goal.

### G5: 250-TF Candidate

A candidate becomes worth deep tuning only if it clears ~190-200 TF early. Below that, the remaining
gap to 250 is probably too large for waitcnt polish alone.

## Expected Failure Modes

- **Register-file admission collapse:** dynamic tiles grow too large and reproduce the single-wave
  PM4 failure mode. Fix by reducing per-wave tile, not by requesting a larger pool.
- **Barrier/LDS issue pressure:** B sharing may replace global feed with LDS/barrier instructions and
  fail to improve WMMA/cycle.
- **Hand-asm scheduling gap:** a correct PM4 kernel may still miss hipcc's wait ladder. This must be
  fixed at 4x4 before testing larger dynamic tiles.
- **TGID delivery gap:** raw PM4 `TGID_X` is unavailable in this harness. Do not build future
  wave-group scheduling on `s15`; use atomic claim + LDS broadcast.
- **Coarse wait feed collapse:** waiting for all B fragments (`s_wait_loadcnt 0x0`) before WMMA
  exposes the full feed latency. The fine release ladder (`BLADDER`) fixed this for B — but it landed
  **flat (1.3 TF)** because B was already coverable; the binding stall is the per-K-tile A round-trip.
- **WMMA-issue ceiling below the bar:** `NOFEED=104 TF` < G2 (161). Hand-asm WMMA scheduling, not feed,
  may be the true wall. The `2x2 + NOFEED@2x2` test isolates occupancy-limited vs schedule-limited.
- **dyn-VGPR scope:** raw PM4 proved `DYN_VGPR_EN=1` + `s_alloc_vgpr` on gfx1201; AQL/HIP drops it.
  The current WG kernel is **static**, so `SQ_DYN_VGPR` is not its limiter — do not re-probe it now.
  dyn-VGPR is a later campaign lever, not part of the static feed/schedule/occupancy question.

## Deliverable

A PM4 dyn-VGPR wave-group GEMM sweep that answers:

1. Can PM4 hand asm match the HIP 4-wave 4x4 winner (first as static; dyn-VGPR is a later lever)?
2. Is the `NOFEED=104` ceiling occupancy-limited or hand-WMMA-schedule-limited (2x2 + NOFEED@2x2)?
3. Can moderate tiles + B sharing increase WMMA/cycle without collapsing residency, toward a 250-TF
   candidate? (fp8 B-load fixed at `global_load_tr_b64`; b128 TR struck.)

If yes, the winner becomes the production/research kernel. If no, the result is still a precise AMD
silicon/tooling dossier: fp8 WMMA is capable of 300 TF, but RDNA4's shared issue port plus lack of an
occupancy-preserving compiler dyn-VGPR path blocks uniform GEMM from reaching it.
