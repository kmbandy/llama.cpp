# DSWS Substrate v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-found the DSWS GEMM substrate so matrix work is *claimed* by whichever wave holds a role (not *owned* by compile-time wave identity), with split-K folded in — enabling correct, brick-free runtime role conversion.

**Architecture:** New gfx1201 kernel `occ_kernel_dsws.s`. Work = a pool of `(mblk, tcol, ksi)` super-tiles; a pinned claimer (`wid 0`) broadcasts the current super-tile; live waves of each role drain shared LDS atomic counters (compute claims rowblks, feeds claim operand frags) against resident-in-LDS A/B for that super-tile; per-segment partials combine into C via `global_atomic_add`. Build static-substrate-first (oracle-green), then layer conversion.

**Tech Stack:** Hand-written gfx1201 (RDNA4, wave32) assembly; raw-PM4 dispatch (`occ_dispatch.cpp`); CPU fp8 e4m3 oracle (`fp8_oracle.cpp`); control law (`dsws_ctrl_model.cpp`, unchanged); dyn-VGPR via `s_alloc_vgpr` (PM4 RSRC2 bit 6).

**Source of truth:** `SPEC_DSWS_SUBSTRATE_V2.md` (this plan implements it). Read it first.

## Global Constraints

- A GPU brick is a **BUG**, never an accepted tax. A hang = full STOP + report; never auto-fire the next variant.
- **The user greenlights EVERY GPU dispatch individually.** Display GPU → only sub-second compositor-safe-chunked dispatches: `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, timeout 30.
- **NEVER pass `--gl2c`** (MES-crash landmine). SAFEPROBE + bounds gate + padding stay ON.
- Commit only when the user asks.
- Barrier-free / lock-free: LDS atomics + busy-wait only. **No `s_barrier`.**
- `occ_kernel_coop.s` (1716B `DSWS=0` coop binary) is **never modified** — it is the known-good reference.
- Single-variable isolation; fix bugs don't dodge them; never declare a wall from an unmeasured assumption.
- Every offline task gate = (1) assembles clean, (2) RGA **0-spill**. GPU oracle gates are batched and **[SUPERVISED]**.

## TDD adaptation for hand-asm

This is hand-written assembly, so the red/green cycle is **build-gate-driven**:
- **"Failing test" →** the gate that is red before the step (RGA reports spill / wrong bin size / oracle mismatch / missing symbol).
- **"Green" →** assembles clean + RGA 0-spill (offline) and, at milestones, GPU oracle `bad=0` (supervised).
- CPU-only logic (oracle tolerance math, control law) keeps true pytest-style TDD.

## File Structure

- **Create `occ_kernel_dsws.s`** — the v2 kernel. Owns: prologue/arming scaffold (lifted from `occ_kernel_coop.s`), v2 LDS layout, the pinned claimer, resident-A/B feed loops (claim-based), the compute loop (claim rowblk → WMMA over `SEGK` → partial-flush), role-agnostic terminal, and (Phase B) conversion actuation.
- **Modify `fp8_oracle.cpp`** — add a relative-tolerance compare mode (Tier 2); exact bit-match stays the default (Tier 1).
- **Modify `occ_dispatch.cpp`** — add the v2 launch path: super-tile pool sizing, `G`/`SEGK`/`n_kseg` params, v2 LDS size, RSRC2 arm, tiered oracle, reuse compositor-safe chunking + streaming + snapshot readback.
- **Modify `build_dsws.sh`** — add v2 build target(s) with `G/SEGK/FM/FN/NCOMP/NAFEED/NBFEED` defsyms + RGA gate (mirrors the existing `mk()` pattern).
- **Modify `RESULT_DSWS.md`** — append v2 records.
- **Unchanged:** `dsws_ctrl_model.cpp`, `test_dsws_ctrl_model.cpp` (carry over verbatim).

## Naming / symbols (used across tasks — define once, reuse exactly)

- `G` = compile-time cooperative M-extent (rowblks per super-tile) = `NCOMP_MAX`.
  **First fitting config (LDS-budget-derived, A1):** `G=6, SEGK=64, FM=2, FN=4` → resident
  A(`192·SEGK`) + B(`64·SEGK`) = `256·SEGK = 16 KB` at `SEGK=64`, fits the 32 KB group segment.
  `SEGK=256` busts it (64 KB); `SEGK` is the LDS/combine-traffic knob swept in Phase C.
- `SEGK` = split-K segment size (K-elements/segment); `n_kseg = KT/SEGK`; `ksi ∈ [0,n_kseg)` = segment index.
- Super-tile id `sti ∈ [0, TOTAL_super)`, `TOTAL_super = (M/(G·16·FM)) · NTL · n_kseg`. Decode: `ksi = sti % n_kseg`; `t = sti / n_kseg`; `tcol = t % NTL`; `mblk = t / NTL`.
- LDS claim/handshake words (v2): `STI_OFF` (broadcast super-tile id), `ROWBLK_NEXT_OFF` (per-super-tile rowblk claim counter), `ROWBLK_DONE_OFF` (per-super-tile completion counter), plus the carried-over `EPOCH_OFF`, role slots `NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT`, `GATE_OFF[4]`, `VRESV_OFF`, `SEGCNT_OFF` (from T2.1). Resident regions: `BRES_OFF` (resident B), `ARES_OFF` (resident A). Feed claim counters: `BFRAG_NEXT_OFF`, `AROW_NEXT_OFF` (reset per super-tile by the claimer).
- Kernel build flag: `DSWS2=1` selects the v2 path/file (analogous to existing `DSWS`).

---

## Execution status (2026-06-30, EOD)

- **A1–A7 + A8 host-launch wiring: DONE, offline-gated** (ASSEMBLE_OK, RGA 0-spill, `mk2` bin 4840B,
  harness compiles, dry-print clean). Reviewed via two round-tables (Opus + Codex) which caught **5
  brick/correctness bugs**, all fixed + re-gated (see `RESULT_DSWS.md` "SUBSTRATE v2" section).
- Config landed differs from the early task text: **`SEGK=64` (not 256)** for LDS fit; **15-kernarg
  contract** (`n_kseg` derived in-kernel, chunk terminal via `occ[24]`, sentinel retire) — NOT the s15/s16/s17
  scheme; **tiered oracle = tight-vs-loose tolerance** (not bit-exact). These supersede the inline task text
  below; `RESULT_DSWS.md` + `occ_kernel_dsws.s` header are authoritative.
- **NEXT = A8 [SUPERVISED GPU], morning 2026-07-01.** First cell:
  `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1 DSWS2_NKSEG=1 timeout 30 ./occ_dispatch --dsws2 4c2a2b`
  → expect `ok=32 bad=0` (TIGHT), `occ[0]=0`, fence FIRED, no brick. kmbandy greenlights. Then LOOSE + mixes.
- Phase-B caveat: quiesce sentinels use compile-time role counts (static-only); conversion must use live
  counts / epoch-snapshot drained counters.

---

## Phase A — Static split-K + claim substrate (fixed roles, no conversion)

### Task A1: v2 scaffold + LDS layout + harness pool sizing

**Files:**
- Create: `occ_kernel_dsws.s` (copy the proven prologue/arming/WMMA-macro/addressing/fp8-LUT/common-retire scaffold from `occ_kernel_coop.s`; gut the work-decomposition body, leave a `s_endpgm` stub in each role).
- Modify: `occ_dispatch.cpp` (add `run_dsws2(...)` entry computing `G/SEGK/n_kseg/TOTAL_super`, allocating C, sizing LDS; wire a `--dsws2` arg; no oracle yet).
- Modify: `build_dsws.sh` (add `mk2()` building `occ_kernel_dsws.s` with `DSWS2=1 FM FN G SEGK NCOMP NAFEED NBFEED`).

**Interfaces:**
- Produces: the v2 LDS symbol block (all `*_OFF` above) with a `.if LDS_TOTAL_DSWS2 > 65536 .error` assert; `run_dsws2` launch; `mk2` build target.
- Consumes: nothing (scaffold).

- [ ] **Step 1 (red):** `./build_dsws.sh` has no v2 target → adding `mk2 ...` fails to assemble (file absent). Confirm the failure message.
- [ ] **Step 2:** Create `occ_kernel_dsws.s` = scaffold + the v2 LDS symbol block. Lay out offsets sequentially; assert `LDS_TOTAL_DSWS2 ≤ 32768` (group segment). Each role label (`.Lclaimer`, `.Lbfeed`, `.Lafeed`, `.Lcompute`) is a bare `s_endpgm` stub.
- [ ] **Step 3 (green):** assemble + RGA. Expected: assembles clean, RGA `livereg` within budget, **0 spill**.

```bash
# assemble (mirror build_dsws.sh mk()):
L=/opt/rocm/llvm/bin; cd <spikedir>
$L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
  -Wa,-defsym,NCOMP=4 -Wa,-defsym,NAFEED=2 -Wa,-defsym,NBFEED=2 -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=1 \
  -c occ_kernel_dsws.s -o occ_dsws2.o && echo ASSEMBLE_OK
# RGA 0-spill gate:
KSRC=occ_kernel_dsws.s ./rga_check.sh dsws2_a1 DSWS2=1 FM=2 FN=4 G=6 SEGK=64 NCOMP=4 NAFEED=2 NBFEED=2 SAFEPROBE=1 | grep -E "livereg|spill"
# Expected: livereg printed, "spill" count = 0
```

- [ ] **Step 4:** Build the harness; confirm `--dsws2` parses and prints the computed `G/SEGK/n_kseg/TOTAL_super/LDS bytes` without launching.

```bash
ROCM=/opt/rocm; PM4=../dvgpr_pm4
systemd-run --user --scope -p MemoryMax=4G clang++ -std=c++17 -O2 -Wall -Wno-unused \
  -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" \
  occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" \
  "$ROCM/lib/libhsakmt.a" -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch && echo BUILD_OK
# dry print (no GPU): a --dsws2 path that prints params and returns before dispatch when DSWS2_DRYRUN=1
DSWS2_DRYRUN=1 ./occ_dispatch --dsws2   # Expected: prints G=4 SEGK=256 n_kseg=.. TOTAL_super=.. LDS=..B
```

- [ ] **Step 5:** (no commit — await user.)

### Task A2: Tiered oracle (CPU TDD)

**Files:**
- Modify: `fp8_oracle.h` / `fp8_oracle.cpp` (add `OracleCmp oracle_compare(const float* got, const float* ref, long n, float rel, float abs_)` returning `{bool ok; long bad; double max_rel;}` — generalizes the existing inline gate `fabs(got-ref) > rel*fabs(ref)+abs_`).
- Modify: `occ_dispatch.cpp` (define tier thresholds `TIGHT={5e-3,1e-2}`, `LOOSE={3e-2,2e-2}`; the v2 oracle path picks TIGHT when `n_kseg==1`, LOOSE when `n_kseg>1` — actual v2 compare call is wired in A8).
- Test: a `main()`-guarded self-test in `fp8_oracle.cpp` (`#ifdef ORACLE_SELFTEST`).

**Interfaces:**
- Produces: `OracleCmp oracle_compare(got, ref, n, rel, abs_)`; tier threshold constants.
- Consumes: nothing.

- [ ] **Step 1 (red):** write the self-test. TIGHT(5e-3) PASSES identical / 0.1%-perturbed, FAILS 1%-perturbed; LOOSE(3e-2) PASSES 1%, FAILS 5%.

```cpp
// ORACLE_SELFTEST main (rel,abs):
assert( oracle_compare(ident,      ident, 256, 5e-3f,1e-2f).ok);   // identical
assert( oracle_compare(perturb_0p1pct, ident, 256, 5e-3f,1e-2f).ok);
assert(!oracle_compare(perturb_1pct,   ident, 256, 5e-3f,1e-2f).ok);   // tight rejects 1%
assert( oracle_compare(perturb_1pct,   ident, 256, 3e-2f,2e-2f).ok);   // loose accepts 1%
assert(!oracle_compare(perturb_5pct,   ident, 256, 3e-2f,2e-2f).ok);   // loose rejects 5%
```
(Use ref values ~O(1) so the abs term doesn't dominate the rel term in the test.)

- [ ] **Step 2 (red):** compile self-test → fails (function undefined).

```bash
clang++ -std=c++17 -DORACLE_SELFTEST fp8_oracle.cpp -o /tmp/oracle_selftest 2>&1 | head
```

- [ ] **Step 3 (green):** implement `oracle_compare`; rebuild self-test → all asserts pass.

```bash
clang++ -std=c++17 -DORACLE_SELFTEST fp8_oracle.cpp -o /tmp/oracle_selftest && /tmp/oracle_selftest && echo ORACLE_OK
```

- [ ] **Step 4:** add the `TIGHT`/`LOOSE` tier constants + the `n_kseg`-based selector in `occ_dispatch.cpp` (call site wired in A8). Harness rebuilds (command from A1 Step 4).
- [ ] **Step 5:** (no commit — await user.)

### Task A3: Pinned claimer + super-tile broadcast

**Files:**
- Modify: `occ_kernel_dsws.s` (`.Lclaimer` = `wid 0` only: loop { `global_atomic_add` next `sti` at the claim offset; decode `(mblk,tcol,ksi)`; reset `ROWBLK_NEXT_OFF=0`, `ROWBLK_DONE_OFF=0`, `BFRAG_NEXT_OFF=0`, `AROW_NEXT_OFF=0`; publish `STI_OFF`; bump `EPOCH_OFF`; also bump `SEGCNT_OFF` (the clock); gate next claim on `ROWBLK_DONE==G` (stub `G` constant until A7) }; terminal publish `sti≥TOTAL_super`).

**Interfaces:**
- Produces: `STI_OFF` broadcast + epoch bump in the proven `STI`-before-`EPOCH` order; per-super-tile counter resets.
- Consumes: the LDS layout (A1), the claim atomic pattern (lifted from coop `global_atomic_add offset:20`).

- [ ] **Step 1:** implement `.Lclaimer`. Decode uses the formulas in "Naming/symbols". Publish order: write `STI_OFF` → reset the 4 per-super-tile counters → `s_add EPOCH_OFF` LAST (followers read `STI` before the epoch bump). Non-`wid0` waves skip claimer and fall to their role loop.
- [ ] **Step 2 (green):** assemble + RGA 0-spill (commands from A1 Step 3, retag `dsws2_a3`).
- [ ] **Step 3:** (no commit — await user.)

### Task A4: Resident-B feed loop (claim frags → stage B)

**Files:**
- Modify: `occ_kernel_dsws.s` (`.Lbfeed`: follow epoch/`STI`; decode `(tcol,ksi)`; loop { `ds`-atomic-add `BFRAG_NEXT_OFF` to claim frag `f ∈ [0,FN)`; if `f≥FN` break; `global_load_tr_b64` B for `(tcol,ksi,f)`; `ds_store` into resident `BRES_OFF + f*<fragstride>` }; then wait next super-tile. `wid 0` (claimer) ALSO runs this after claiming — it is a B-feed-class wave).

**Interfaces:**
- Produces: resident B fully staged for the current super-tile at `BRES_OFF`; identity-free frag claiming via `BFRAG_NEXT_OFF`.
- Consumes: `STI_OFF`/`EPOCH_OFF` broadcast (A3); B global addressing (lift from coop B-feed `s_mul..tcol` decode).

- [ ] **Step 1:** implement the claim-frag → load_tr → ds_store resident staging. B segment size = `FN·16` cols × `SEGK` K → resident bytes at `BRES_OFF` (verify within `LDS_TOTAL_DSWS2`).
- [ ] **Step 2 (green):** assemble + RGA 0-spill (retag `dsws2_a4`).
- [ ] **Step 3:** (no commit — await user.)

### Task A5: Resident-A feed loop (claim rowblk-A → stage A)

**Files:**
- Modify: `occ_kernel_dsws.s` (`.Lafeed`: follow epoch/`STI`; decode `(mblk,ksi)`; loop { `ds`-atomic-add `AROW_NEXT_OFF` to claim rowblk-A index `r ∈ [0,G)`; if `r≥G` break; `global_load` A for absolute rowblk `mblk*G+r`, segment `ksi` (rows `(mblk*G+r)*16*FM`, K-range `ksi*SEGK`); `ds_store` into resident `ARES_OFF + r*<rowstride>` }; wait next super-tile).

**Interfaces:**
- Produces: resident A fully staged for the current super-tile at `ARES_OFF`; identity-free rowblk-A claiming via `AROW_NEXT_OFF`.
- Consumes: A3 broadcast; A global addressing (lift from coop compute A-saddr `rowblk*(16*FM)*K + k*16`, with `rowblk = mblk*G + r`, K offset `ksi*SEGK`).

- [ ] **Step 1:** implement claim-rowblk-A → global_load → ds_store resident staging. A segment size = `G·16·FM` rows × `SEGK` K → resident bytes at `ARES_OFF` (verify fit).
- [ ] **Step 2 (green):** assemble + RGA 0-spill (retag `dsws2_a5`).
- [ ] **Step 3:** (no commit — await user.)

### Task A6: Compute loop — claim rowblk → WMMA → partial-flush

**Files:**
- Modify: `occ_kernel_dsws.s` (`.Lcompute`: follow epoch/`STI`; wait until resident A/B staged (busy-wait on the feed claim counters reaching `FN`/`G`, or a `staged` flag the feeds set); loop { `ds`-atomic-add `ROWBLK_NEXT_OFF` to claim `rowblk r ∈ [0,G)`; if `r≥G` break; `s_alloc_vgpr NFV`; WMMA `FM×FN` over `SEGK/16` k-steps reading resident A[r]/B from LDS, accumulate fp32; `global_atomic_add_f32` the `FM×FN` partials into `C[(mblk*G+r) , tcol]`; `s_alloc_vgpr 32` (shrink); `ds`-atomic-add `ROWBLK_DONE_OFF`++ } ).

**Interfaces:**
- Produces: partial-summed C contributions; `ROWBLK_DONE` increments (consumed by A7 handshake).
- Consumes: resident A/B (A4/A5); WMMA macros + C addressing (lift from coop compute body); the fp8-LUT front-end if `ML8` (carry the coop variant).

- [ ] **Step 1:** **verify `global_atomic_add_f32` encodes on gfx1201** (assemble a one-liner). If rejected → fall back to a CAS loop (`ds`/`global_atomic_cmpswap` on the C word) and note it in `RESULT_DSWS.md`.

```bash
echo 'global_atomic_add_f32 v0, v1, s[0:1]' | $L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c - -o /dev/null 2>&1 && echo ADDF32_OK || echo ADDF32_REJECTED_use_CAS
```

- [ ] **Step 2:** implement the compute loop. Staging-ready wait = busy-wait on `BFRAG_NEXT≥FN && AROW_NEXT≥G` (feeds advance these as they stage). Grow/shrink each guarded by SCC-retry (brick-class rule).
- [ ] **Step 3 (green):** assemble + RGA 0-spill (retag `dsws2_a6`). Resident-read + WMMA path must stay ≤ budget under the grown `NFV`.
- [ ] **Step 4:** (no commit — await user.)

### Task A7: Completion handshake + role-agnostic terminal

**Files:**
- Modify: `occ_kernel_dsws.s` (claimer's next-claim gate now real: spin until `ROWBLK_DONE_OFF == G` before claiming/broadcasting the next super-tile — frees resident A/B safely. Terminal: every role, at its decision boundary, checks broadcast `sti ≥ TOTAL_super` → retire via the shared `.Lretire` path).

**Interfaces:**
- Produces: the safe resident-B/A lifetime (claimer advances only when all `G` rowblks done+flushed) + uniform retire.
- Consumes: `ROWBLK_DONE_OFF` (A6), `STI_OFF`/`TOTAL_super`.

- [ ] **Step 1:** wire the claimer advance-gate (`ROWBLK_DONE==G`) and the role-agnostic `sti≥TOTAL_super` terminal in all four roles.
- [ ] **Step 2 (green):** assemble + RGA 0-spill (retag `dsws2_a7`). Add the 6-bin v2 build set to `build_dsws.sh` (`mk2` for the swept role mixes at `n_kseg=1` and a `n_kseg>1` config); `./build_dsws.sh` → all OK, `fail=0`.
- [ ] **Step 3:** (no commit — await user.)

### Task A8 [SUPERVISED GPU]: Step-1 BIG GATE — static substrate oracle-green

**Files:** none (run only). **STOP and request greenlight before EACH dispatch.**

- [ ] **Step 1:** Prepare ONE gated dispatch — static mix (e.g. `4c2a2b`), `n_kseg=1`, compositor-safe chunked, streamed. **STOP for go.**

```bash
ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1 \
DSWS2_NKSEG=1 timeout 30 ./occ_dispatch --dsws2 4c2a2b   # Expected: ok=N bad=0 (EXACT), no brick, clean teardown
```

- [ ] **Step 2:** On green, repeat at `n_kseg>1` (TOL gate) and at the other mixes (`6c1a1b`, `2c3a3b`) — **one greenlit dispatch at a time**. Expected each: `bad=0`, fence FIRED, clean teardown.
- [ ] **Step 3:** Record Step-1 results (mixes, `n_kseg`, ok/bad, LDS bytes, RGA livereg) in `RESULT_DSWS.md`. **Gate: do not start Phase B until the static substrate is oracle-green at both tiers.**

---

## Phase B — Conversion (layered on the green static substrate)

### Task B1: Port gate-CAS ticket + reservation + sensors into v2 (no actuation)

**Files:**
- Modify: `occ_kernel_dsws.s` (lift the proven `try_gate dir, swin` macro, the `mark`/`mark_set`/`mark_inc` snapshot macros, the sensors, and the reservation init from `occ_kernel_coop.s` DSWS path; call `try_gate` at the per-kseg compute boundary as a unit, incrementing a would-win snapshot slot — **no role change**).

**Interfaces:**
- Produces: `try_gate`, sensors, `vgpr_reserved` init in v2; a would-win counter in the snapshot.
- Consumes: role slots + `GATE_OFF` + `SEGCNT_OFF` + `VRESV_OFF` (A1 layout).

- [ ] **Step 1:** port the macros + reservation init (`VRESV_OFF = NCOMP*NFV + (NAFEED+NBFEED)*VLEAN`); add the unit `try_gate` call at the compute per-kseg boundary.
- [ ] **Step 2 (green):** assemble + RGA 0-spill (retag `dsws2_b1`).
- [ ] **Step 3:** (no commit — await user.)

### Task B2: compute→feed shrink actuation

**Files:**
- Modify: `occ_kernel_dsws.s` (at the compute per-kseg boundary, on `occ_X<LOW & try_gate(compute→feedX) win & nComp>1`: CAS-dec `nComp`, inc `nFeedX`, `atomic_sub vgpr_reserved` by `NFV−VLEAN`, `s_alloc_vgpr 32`, flip private role register, `s_branch` into the feed loop. Shrink always succeeds).

**Interfaces:**
- Produces: a wave that leaves compute and starts claiming feed work (no work-item handoff — the payoff of the claim model).
- Consumes: `try_gate` (B1), role slots, reservation, sensors.

- [ ] **Step 1:** implement the shrink conversion on the `try_gate` win flag (dirs 0/1). Floor-guard `nComp>1` via CAS-dec-if-greater. A converted wave falls through to the terminal check before doing feed work.
- [ ] **Step 2 (green):** assemble + RGA 0-spill (retag `dsws2_b2`).
- [ ] **Step 3:** (no commit — await user.)

### Task B3: feed→compute grow actuation + envelope abort

**Files:**
- Modify: `occ_kernel_dsws.s` (at a feed inter-claim point, on `occ_X>HIGH & try_gate(feedX→compute) win & nFeedX>1`: `reserve_grow` = `atomic_add vgpr_reserved` by `NFV−VLEAN`, validate `≤ BUDGET`; on success CAS-dec `nFeedX`/inc `nComp`, flip role register, `s_alloc_vgpr NFV`, `s_branch` into compute; on over-budget `atomic_sub` + abort, stay feed).

**Interfaces:**
- Produces: a wave that leaves feed and starts claiming rowblks; the sum-envelope abort path.
- Consumes: `try_gate` (B1), reservation model, `BUDGET` (per-SIMD VGPR budget defsym from RGA/ISA).

- [ ] **Step 1:** implement the grow conversion with reserve→validate→commit-or-undo (transcribe `reserve_grow` from `dsws_ctrl_model.cpp` exactly). Order: reserve first, validate, then slot move + grow; on fail, undo + abort (NO slot change).
- [ ] **Step 2 (green):** assemble + RGA 0-spill (retag `dsws2_b3`); `./build_dsws.sh` v2 set all OK; DSWS2 static-mix still assembles for a degenerate `LOW=0,HIGH=RINGD` (no-conversion) build to keep an offline-comparable control.
- [ ] **Step 3:** (no commit — await user.)

### Task B4 [SUPERVISED GPU]: Step-2 GATE — dynamic oracle-green + conversion storm

**Files:** none (run only). **STOP for greenlight before EACH dispatch.**

- [ ] **Step 1:** ONE gated dispatch — dynamic mix enabled, normal watermarks, `n_kseg=1` EXACT, streamed. **STOP for go.** Expected: `ok=N bad=0`, conversions visible in the snapshot (role slots move), clean teardown, no brick.
- [ ] **Step 2:** On green — conversion-storm: tight `DSWS_LOW`/`DSWS_HIGH` + `EPOCH_SHIFT=0` (max conversion rate) + ×10 repeats, `n_kseg=1` EXACT then `n_kseg>1` TOL. Expected: STILL `bad=0`. Any `bad>0` → STOP, a conversion race exists; bisect offline.
- [ ] **Step 3:** Record Step-2 results in `RESULT_DSWS.md`. **Gate: do not proceed to tuning until dynamic + storm are both clean.**

---

## Phase C — Adaptivity proof + tuning

### Task C1: Mix-over-time + conversion-counter instrumentation

**Files:**
- Modify: `occ_kernel_dsws.s` (4 per-direction conversion counters into the snapshot; emit role slots each snapshot).
- Modify: `occ_dispatch.cpp` (decode + print `t, nComp, nAfeed, nBfeed, convs[4]`).

- [ ] **Step 1:** add the counters + harness table. **Step 2 (green):** assemble + RGA 0-spill; harness builds. **Step 3:** (no commit — await user.)

### Task C2 [SUPERVISED GPU]: Converge-from-wrong-start proof

**Files:** none. **STOP for go.**

- [ ] **Step 1:** feed-bound shape (`down`, N=2560), deliberately wrong launch mix (`6c1a1b`). **STOP for go.**
- [ ] **Step 2:** run; watch mix table + TF. Expected (the money shot): controller shifts toward feed (nAfeed/nBfeed climb, nComp falls), TF climbs, settles near static-optimal. Compare adaptive-from-wrong vs static-wrong vs static-optimal.
- [ ] **Step 3:** record convergence trace + TF curve in `RESULT_DSWS.md`.

### Task C3 [SUPERVISED GPU]: Tuning sweep + issue-mix

**Files:** none (record in `RESULT_DSWS.md`). **STOP for go each cell.**

- [ ] **Step 1:** sweep `{LOW, HIGH, RINGD, EPOCH_SHIFT, G, SEGK}` on `down`/`down_pf`, **one gated dispatch at a time** (oracle STORE=1 before perf STORE=0 for any new geometry).
- [ ] **Step 2:** `--att` issue-mix on the winner. Expected: compute waves issue near-pure WMMA — measurably fewer non-WMMA issues per WMMA than the 31:32 static baseline.
- [ ] **Step 3: Success-metric gate:** adaptive kernel (a) oracle-correct, (b) beats the static 3-role baseline AND the 165.7 TF winner on `down`/`down_pf`, (c) demonstrably adapts across shapes (different settled mixes). Record TF, settled mixes, `--att` deltas.

### Task C4: Bank the outcome

**Files:**
- Modify: `RESULT_DSWS.md` (final), `MAD305_DSWS_MASTER.md` (numbers + next steps).

- [ ] **Step 1:** finalize `RESULT_DSWS.md` (numbers, settled mixes, issue-mix deltas, honest verdict incl. any null result). **Step 2:** `mneme_write` a session_summary banking the result + v2 substrate design. **Step 3:** commit (when greenlit) + Jira MAD-305 update.

---

## Self-Review

**Spec coverage** (every `SPEC_DSWS_SUBSTRATE_V2.md` section → a task):
- §1 work pool & claim model → A1 (layout), A3 (super-tile claim), A6 (rowblk claim), A7 (coverage via handshake). ✓
- §2 resident A/B + completion handshake → A4 (B), A5 (A), A7 (handshake). ✓
- §3 role tags + pinned claimer → A1 (slots), A3 (`wid 0` claimer/clock). ✓
- §4 partial-C + tiered oracle → A2 (oracle), A6 (`global_atomic_add_f32` flush + encodability check). ✓
- §5 conversion actuation → B1 (gate/sensors/reservation), B2 (shrink), B3 (grow+abort). ✓
- §6 file structure + build sequence + gates → A1 (new file + build), A8/B4/C2/C3 (supervised gates). ✓
- Testing (tiered oracle, RGA, storm, control-law) → A2, every offline step's RGA, B4 storm, carried-over `test_dsws_ctrl_model.cpp`. ✓
- Risks (LDS budget, atomic-add-f32, combine traffic, G overshoot) → A1 assert, A6 Step 1, C3 measure, B-floor guards. ✓

**Placeholder scan:** build/RGA/oracle commands are concrete; `<spikedir>` is the only literal path placeholder (the dvgpr_occ dir). Resident-region byte formulas are explicit; exact instruction sequences for the WMMA/load bodies are lifted from the named coop labels (discovered against RGA per the hand-asm TDD note).

**Symbol consistency:** `G`, `SEGK`, `ksi`, `n_kseg`, `sti`, `TOTAL_super`, `*_OFF`, `NFV`, `VLEAN`, `try_gate` used identically across tasks; decode formulas defined once in "Naming/symbols" and referenced.
