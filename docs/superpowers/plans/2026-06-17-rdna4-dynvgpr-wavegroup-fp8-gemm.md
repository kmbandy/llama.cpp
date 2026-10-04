# RDNA4 dyn-VGPR wave-group fp8 WMMA GEMM — Implementation Plan

**Design:** `docs/superpowers/specs/2026-06-17-rdna4-dynvgpr-wavegroup-fp8-gemm-design.md`

**Goal:** Replace the failed single-wave dyn-VGPR throughput vehicle with a PM4-enabled
multi-wave cooperative GEMM. Use dynamic VGPR for moderate per-wave accumulator growth while LDS
sharing creates a larger logical tile and reduces feed issue density.

## Ground Rules

- PM4 is required because it is the proven way to arm dynamic VGPR on gfx1201.
- All GPU runs are supervised. Raw PM4 can wedge the GPU.
- Do not run pool sizes that approach full VGPR-file occupancy. The host must print and enforce a
  deadlock guard before dispatch.
- Do not tune fat tiles until the 4-wave 4x4 PM4 kernel reaches HIP-parity.
- Every fed config must pass the oracle before perf is trusted.
- Track WMMA/cycle and TFLOPS. Occupancy is supporting data, not the success metric.

## Files

| file | role |
|---|---|
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_mbgemm.s` | existing PM4 asm kernel; can be extended or used as reference |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_wggemm.s` | obsolete TGID grid-stride smoke; keep only as TGID diagnostic history |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_wgdiag.s` | SGPR probe proving raw-PM4 TGID is unavailable |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_wglds.s` | atomic-claim + LDS-broadcast wave-group smoke foundation |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_wggemm2.s` | current correct 4-wave fp8 GEMM compute kernel |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_dispatch.cpp` | PM4 dispatch, oracle, sweep, safety guard |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/build.sh` | RAM-capped build matrix |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/RESULT_WGGEMM.md` | result dossier |

## Phase 0: Anchor The Reference (DONE)

- [x] Extract the HIP winner ISA again from `bench/gemm_fp8_levers.hip` with `hipcc -save-temps=obj`.
- [x] Save the 4x4 A-LDS/B-trfeed hot loop path under `/tmp/cg/` or a local notes file.
- [x] Count the hot-loop instruction mix: WMMAs, B `global_load_tr`, A `ds_read`, address math,
      waitcnts.
- [x] Record the current HIP winner numbers for `4096^2 x K16384` and `4096x14336 x K16384`.

Exit criteria:

- A concrete 4x4 HIP ISA snippet is available for transcription.
- Baseline TFLOPS and WMMA/cycle are written into the result dossier.

Result:

- `4096^2 x K16384`: HIP winner **161.1 TF**, `8.34 WMMA/cyc`.
- `4096x14336 x K16384`: HIP winner **145.4 TF**, `7.53 WMMA/cyc`.
- G2 hard-pass bar on the square target is **>=153 TF**; acceptable first-pass bar is **>=145 TF**.

## Phase 1: Build The 4-Wave PM4 Workgroup Skeleton (DONE, PIVOTED)

- [x] Add `occ_kernel_wggemm.s` or isolate a new mode in `occ_kernel_mbgemm.s`.
- [x] Define `TWM=2`, `TWN=2`, `WAVES=4`, `BLOCK_THREADS=128`.
- [x] Keep the existing PM4 user-SGPR ABI where possible: queue pointer, A, Bshuf, C, KT, K stride,
      B tile stride, total tiles, N tile decode fields.
- [x] Implement workgroup tile decode: one claimed tile maps to a logical `128x128` C tile.
- [x] Implement lane/wave mapping identical to the HIP winner:
      `wave_m = wid / TWN`, `wave_n = wid % TWN`, `lane = tid % 32`.
- [x] Add a no-compute smoke mode that claims tiles and writes a recognizable pattern to C.
- [x] Wire a `--wggemm-smoke` dispatch path in `occ_dispatch.cpp`.

Exit criteria:

- The PM4 workgroup launches with 4 waves/block.
- Tile decode covers the expected C tiles exactly once.
- Smoke output proves all four waves are active.

Pivot and final Phase 1 result:

- The initial `TGID_X` grid-stride smoke failed: raw PM4 `DISPATCH_DIRECT` did not deliver TGID to
  any SGPR. `occ_kernel_wgdiag.s` proved no `s8..s23` value contained a workgroup-id permutation.
- Grid-stride scheduling is abandoned for this path.
- Canonical tile distribution is now **leader atomic claim + LDS broadcast + workgroup barrier**.
- `occ_kernel_wglds.s` / `--wglds-smoke` passed:
  `512^2`, `1024^2`, and `4096^2`, with every tile claimed exactly once and all 4 wave marks correct.
- Raw-PM4 LDS allocation is programmed with `COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE =
  ceil(bytes/512) << 15`.

## Phase 2: Add A-LDS Fill And 4x4 Static Compute (DONE)

- [x] Add LDS storage for `A[128][32]`.
- [x] Implement cooperative global-to-LDS A fill using all 128 lanes.
- [x] Add barriers equivalent to the HIP A-LDS path.
- [x] Use preshuffled B and `global_load_tr_b64` exactly as the HIP winner.
- [x] Implement per-wave `4x4` accumulators without dynamic growth first.
- [ ] Transcribe the HIP waitcnt ladder as closely as possible.
- [x] Store diagnostic C with the known gfx12 accumulator lane layout.
- [x] Add this as `--wggemm-compute`.

Exit criteria:

- Oracle passes for a small shape.
- `llvm-objdump` shows the expected structure: A `ds_read`, B `global_load_tr`, 16 WMMAs per K step
  per wave, no branch inside the WMMA issue run.

Result:

- `occ_kernel_wggemm2.s` is bit-exact with chained `wmma_ref` through `512^2 x K2048`.
- Static resource use: ~192 VGPR, LDS 4100 B for single-buffer A + tile broadcast.
- Fine wait ladder intentionally remains pending for Phase 3 because Phase 2 prioritized correctness.

## Phase 3: G2 HIP-Parity Gate (IN PROGRESS)

- [x] Build the static 4-wave 4x4 kernel.
- [x] Run supervised correctness on a small and medium shape.
- [x] Run supervised perf on `4096^2 x K16384`.
- [x] Compare against the HIP winner.
- [x] Add `STORE=0` minimal-store perf path so diagnostic flat C stores do not dominate timing.
- [x] Add A double-buffer probe.
- [x] Add B register prefetch probe.
- [x] Add NOBAR attribution probe.
- [x] Add NOFEED compute-ceiling probe.
- [x] Add fine descending `s_wait_loadcnt` ladder (`BLADDER`, build `DBUF=0`).
- [x] Re-measure fed perf after the ladder — **flat at 1.3 TF** (oracle bit-exact). B feed was already
  coverable; it is not the wave-group bottleneck.
- [ ] **Formal contingency (ladder flat → now active): 2x2 + NOFEED@2x2.** Drop the per-wave tile
  `4x4 -> 2x2` (FM=FN=2, ~32-VGPR acc, ~6 -> ~16 waves/SIMD static) AND measure `NOFEED@2x2`. The pair
  answers the decisive question: is the `NOFEED=104` ceiling occupancy-limited (rises with waves) or
  hand-WMMA-schedule-limited (stays ~104)?

Decision:

- Pass: within ~5-10% of HIP. Continue.
- Fail: stop fat-tile work. Tune waitcnt placement, WMMA ordering, address hoisting, and LDS fill
  until parity is reached.

Exit criteria:

- `RESULT_WGGEMM.md` records the static PM4 4x4 TFLOPS, WMMA/cycle, and parity verdict.

Current Phase 3 measurements:

| variant @ `4096^2 x K16384` | TF | reading |
|---|---:|---|
| baseline, coarse waits | 1.1 | correct but structurally stalled |
| A double-buffer | 1.4 | flat; A latency is not the blocker |
| B register prefetch, coarse wait | 1.3 | flat; prefetch without release ladder is ineffective |
| NOBAR | 1.3 | flat; not barrier-dominated alone |
| NOFEED | 104 | compute path is viable; per-K-tile feed is the poison |

Current interpretation:

- The PM4 wave-group vehicle is correct and can issue WMMAs (`NOFEED=104 TF`), but the fed path is
  waiting on all fragments with coarse `s_wait_loadcnt/dscnt 0x0`.
- A/B prefetch and barrier removal do not help while the kernel still waits for the full feed group.
- The HIP-style fine descending `s_wait_loadcnt` ladder (`0x7 -> 0x0`, `BLADDER`) is **implemented and
  measured: flat at 1.3 TF** (oracle bit-exact). It hides B correctly, but B was already coverable —
  because the 8 B loads are issued before the A-frag `ds_load`+`dscnt` wait, frag-0 lands by the first
  `0x7` and the rest hide behind WMMA work. The exposed cost is the per-K-tile **A round-trip**
  (global load -> `ds_store` -> barrier -> `ds_load` -> barrier), a serial latency chain the B ladder
  cannot touch.
- **Two walls, not one.** `NOFEED=104 TF` is itself **below the G2 bar (161)** — it is this kernel's
  hand-WMMA-only ceiling (HIP's is ~272). So feed-hiding + occupancy alone cannot reach parity unless
  the WMMA-issue ceiling also rises. The decisive next test is **2x2 + NOFEED@2x2**: the big static
  occupancy jump (~6 -> ~16 waves/SIMD) tells us whether `104` is occupancy-limited (rises) or
  schedule-limited (stays ~104, demanding WMMA-stream tightening).
- **dyn-VGPR note (campaign scope, not this kernel):** raw PM4 has proven `DYN_VGPR_EN=1` +
  `s_alloc_vgpr` work on gfx1201; AQL/HIP drops the bit. The current wave-group kernel is **static**
  VGPR, so `SQ_DYN_VGPR` is not its limiter. dyn-VGPR stays a later lever once the static
  feed/schedule/occupancy picture is settled — do not re-probe `SQ_DYN_VGPR` now.

## Phase 4: Enable Dynamic VGPR Without Changing Tile Shape

- [ ] Add PM4 dynamic-VGPR-enabled binary for the same 4x4 workgroup.
- [ ] Insert `s_alloc_vgpr FATREGS` before compute and shrink to `LEANV` after store.
- [ ] Add host-side reporting for `LEANV`, `FATREGS`, requested pool, admitted/max-live waves if
      measurable.
- [ ] Enforce the deadlock guard before dispatch.
- [ ] Add `--wggemm-g2-dyn`.

Decision:

- Pass: dyn 4x4 is close to static 4x4. Continue.
- Fail: measure grow/shrink overhead. Try larger `BATCH` or keep-grown-across-batch only if the
  deadlock guard remains conservative.

Exit criteria:

- Dynamic VGPR overhead is quantified independently from tile reuse.

## Phase 5: Medium Dynamic Tiles

Test moderate per-wave growth before attempting extreme 8x8.

- [ ] Add `FM`, `FN` defsyms for per-wave accumulator tile.
- [ ] Build `4x4`, `4x6`, `6x4`, and optionally `6x6`.
- [ ] Keep logical tile simple first: `128x128` or the smallest logical expansion needed for the
      tile.
- [ ] For each config, compute and print `FATREGS`.
- [ ] Oracle-gate every config.
- [ ] Run supervised perf on both wgrad shapes.

Decision:

- Continue only if WMMA/cycle rises above the 4x4 baseline.
- Drop any config whose TFLOPS improves only by reducing overhead on small shapes but does not improve
  WMMA/cycle on big-K.

Exit criteria:

- A ranked table of medium dynamic tiles exists.
- The best config either clears 155 TF or explains why it does not.

## Phase 6: Logical-Wide Wave-Group Tiles

Use wave-group structure to increase logical reuse without making each wave enormous.

- [ ] Add logical tile variants:
      `128x256` with `TWM=2,TWN=4` if occupancy/admission permits, or
      `192x128` with `TWM=3,TWN=2` if the scheduler supports it cleanly.
- [ ] Keep per-wave accumulator moderate: `4x4`, `4x6`, or `6x4`.
- [ ] Update tile decode, LDS A layout, B tile indexing, and C store mapping.
- [ ] Oracle-gate each variant.
- [ ] Run supervised perf.

Decision:

- A logical-wide variant is a real candidate only if it improves WMMA/cycle, not just tile overhead.
- If it collapses occupancy or introduces too much LDS/barrier overhead, revert to the best Phase 5
  tile.

Exit criteria:

- The best logical-wide tile is compared directly with the best medium tile.

## Phase 7: B Feed Reduction Experiments

Run these only after the wave-group dyn path is correct.

### B Sharing

- [ ] Add `BFEED=1` mode that stages B fragments in LDS for reuse across waves with the same
      `wave_n`.
- [ ] Keep `BFEED=0` as the default direct `global_load_tr` path.
- [ ] Structurally verify the K loop removes duplicated `global_load_tr` instructions.
- [ ] Oracle-gate.
- [ ] Measure TFLOPS and WMMA/cycle.

### ~~b128 Probe~~ — STRUCK (ISA: no fp8 path)

**Removed.** Per RDNA4 ISA, `GLOBAL_LOAD_TR_B128` is **16-bit-data only**; for fp8/wave32 the only
transpose-load path is `GLOBAL_LOAD_TR_B64` (64 bits/lane = one 16×16 fp8 tile). There is no wider fp8
load. Do not spend time here. The B-feed levers that remain are B sharing (above), the fine
`s_wait_loadcnt` ladder (Phase 3, done), and occupancy.

Exit criteria:

- A clear yes/no answer for B sharing.
- If it improves WMMA/cycle, combine it with the best Phase 5/6 tile.

## Phase 8: Full Candidate Sweep

- [ ] Sweep the bounded final grid:
      best two tile shapes x `BFEED={0,1}` x `BATCH={1,2,4}` (fp8 B-load is fixed `global_load_tr_b64`).
- [ ] Run on:
      `4096^2 x K16384`,
      `4096^2 x K32768`,
      `4096x14336 x K16384`.
- [ ] Record correctness, TFLOPS, percent of 307-TF ceiling, WMMA/cycle, `FATREGS`, and pool.
- [ ] Run `NOFEED` on the winning shape to confirm compute headroom remains.
- [ ] Run `PROFILE=1` if the candidate stalls below 200 TF.

Exit criteria:

- A single winning config is identified.
- The result is classified:
      `250+ candidate`, `155-200 partial unlock`, or `no wave-group unlock`.

## Phase 9: Result Dossier

Write `RESULT_WGGEMM.md` with:

- Baseline HIP winner numbers.
- PM4 static 4x4 parity result.
- PM4 dyn 4x4 overhead result.
- Medium tile sweep.
- Logical-wide tile sweep.
- B sharing / b128 results.
- Final winner and verdict against 155, 250, and 300 TF.
- Operational safety notes, including any rejected pool sizes.

## Success Criteria

Minimum useful result:

- PM4 wave-group 4x4 reaches near-HIP parity while dynamic VGPR is armed.

Strong result:

- A moderate dyn-VGPR tile clears 155 TF and improves WMMA/cycle.

Pitch result:

- Final wave-group dyn-VGPR candidate reaches 250+ TF on at least one big-K wgrad shape.

Stop condition:

- If 4x4 PM4 parity cannot be reached, do not continue to fat tiles. The kernel is codegen-bound.
- If parity is reached but all dynamic tiles fail to improve WMMA/cycle, the remaining uniform-GEMM
  path is likely blocked by RDNA4's shared WMMA/feed issue port and register-file residency tradeoff.
