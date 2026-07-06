# DSWS Phase B — Runtime Role Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime {compute↔A-feed↔B-feed} role conversion to the GPU-proven v2 claim-based split-K substrate (`occ_kernel_dsws.s`) without orphaning output, jamming a feed, or bricking.

**Architecture:** Per-epoch snapshot quiesce (Decision 1) + bail-time commit (Decision 2), so the claimer's quiesce counter *is* the snapshot handshake. Conversion control law is ported near-verbatim from the proven `occ_kernel_coop.s` DSWS path (`try_gate`, reservation envelope, sensors), transcribing `dsws_ctrl_model.cpp` exactly. Offline-first: extend + TDD the CPU control model, then assemble/RGA/dry-print gate every kernel change, then three supervised GPU gates.

**Tech Stack:** Hand-written gfx1201 (RDNA4, wave32) assembly; raw-PM4 dispatch (`occ_dispatch.cpp`); CPU fp8 e4m3 oracle (`fp8_oracle.cpp`); C++17 control model (`dsws_ctrl_model.cpp`); `s_alloc_vgpr` dyn-VGPR armed by PM4 RSRC2 bit 6.

**Design spec:** `SPEC_DSWS_PHASEB_CONVERSION.md` (this directory). Read it before starting.

## Global Constraints

- A GPU brick is a **BUG**, never a tax. A hang = full STOP + report; never auto-fire the next variant.
- **The user greenlights EVERY GPU dispatch individually.** Display GPU → only compositor-safe chunked sub-second dispatches: `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, `timeout 30`.
- **NEVER pass `--gl2c`.** SAFEPROBE + bounds gate + padding stay ON. `ML8_COOP_STREAM=1` always.
- **No `s_barrier`** (mixed dyn-VGPR + `s_barrier` hard-deadlocks — proven). LDS-atomic busy-wait only.
- `occ_kernel_coop.s` is **NEVER modified** — known-good reference. All work is additive in `occ_kernel_dsws.s`.
- **Commit to git only when the user explicitly asks.** (Overrides the skill's per-task commit step: do the `git add`/`commit` step only on user request; otherwise leave the tree dirty and report.)
- Config held at Phase-A values **`G=6, SEGK=64, FM=2, FN=4`** for all Phase-B stages.
- Pre-existing dirty files (`occ_kernel_coop.s`, `occ_dispatch.cpp`, `fp8_oracle.*`, `mt_pagedattn*`) are **not ours** — never stage them; flag before any `git diff`.
- Shell is **fish** — do not use bash `set -- $var` word-splitting; use explicit args or `bash script.sh`.
- Round-table discipline (kmbandy's): implement → adversarial review (Fable + Codex) → kmbandy greenlights each GPU dispatch.

## Existing LDS layout in `occ_kernel_dsws.s` (do not renumber — append only)

```
STI_OFF=0  EPOCH_OFF=4  ROWBLK_NEXT_OFF=8  BFRAG_NEXT_OFF=16  AROW_NEXT_OFF=20
NCOMP_SLOT=24  NAFEED_SLOT=28  NBFEED_SLOT=32  GATE_OFF=36 (u32[4]→36,40,44,48)
VRESV_OFF=52  SEGCNT_OFF=56   ... then RES/ARES tile regions (ARES_OFF..ARES_BYTES)
```
The claimer already publishes `NCOMP/NAFEED/NBFEED_SLOT`, `GATE_OFF[4]=0`, `VRESV_OFF=NCOMP*NFV+(NAFEED+NBFEED)*VLEAN`, `SEGCNT_OFF=0`. Existing LDS macros: `lds_put OFF, val`, `lds_get dst, OFF`, `lds_fetch_add dst, OFF, imm`.

## File Structure

- **`dsws_ctrl_model.cpp`** (modify) — add `snapshot`/`quiesce_ready`/invariant reference logic + the `N−1` cross-check. CPU source of truth the asm transcribes.
- **`test_dsws_ctrl_model.cpp`** (modify) — add thread-race tests for the new logic.
- **`occ_kernel_dsws.s`** (modify, additive) — new LDS slots; sensing; ported `try_gate`+`reserve` macros; bail-time commit; claimer snapshot/quiesce. Guarded under a `DSWS2_CONV` defsym so the pre-conversion substrate stays assemblable/testable in isolation.
- **`build_dsws.sh`** (modify) — thread `DSWS2_CONV` into `mk2()`; keep the no-conversion bins buildable.
- **`run_mix_gates.sh`** (reference, in scratchpad) — the supervised-gate driver pattern from Phase A (env `DSWS_NCOMP/NAFEED/NBFEED` + `DSWS2_NKSEG` + `timeout 30` + dmesg check, abort-on-anomaly).

---

### Task 1: Extend the CPU control model — snapshot/quiesce interaction (offline TDD)

**Files:**
- Modify: `dsws_ctrl_model.cpp`
- Test: `test_dsws_ctrl_model.cpp`

**Interfaces:**
- Consumes: existing `watermark_decision(occ,low,high)`, `epoch_of(seg,shift)`, `gate_try_win(gate,E)`, `reserve_grow(resv,delta,budget)`.
- Produces: `struct WgSnap { uint32_t nC, nA, nB; }`; `WgSnap snapshot_counts(uint32_t nC,uint32_t nA,uint32_t nB)`; `bool quiesce_ready(uint32_t rowblk_next,uint32_t bfrag_next,uint32_t arow_next, const WgSnap& s, uint32_t G,uint32_t FN)`; `bool quiesce_ready_nm1(uint32_t quiesce_cnt, uint32_t N)`. Used by Task 5 as the transcription reference.

- [ ] **Step 1: Write the failing test** — append to `test_dsws_ctrl_model.cpp` `main()` before the final `printf`:

```cpp
  // ---- snapshot/quiesce (Phase B Decision 1) ----
  {
    // snapshot freezes the counts used to size the quiesce sentinels
    WgSnap s = snapshot_counts(4, 2, 2);           // G=6, FN=4
    // not ready: rowblk short of G + nC terminal bails
    assert(!quiesce_ready(6 + 3, 4 + 2, 6 + 2, s, 6, 4));  // rowblk 9 < 6+4
    // ready: every counter reached threshold + snapshot bails
    assert( quiesce_ready(6 + 4, 4 + 2, 6 + 2, s, 6, 4));
    // a moved partition (3c3a2b) needs different sentinels; old snapshot is wrong high
    WgSnap s2 = snapshot_counts(3, 3, 2);                  // sentinels: rowblk>=9, bfrag>=6, arow>=9
    assert( quiesce_ready(6 + 3, 4 + 2, 6 + 3, s2, 6, 4)); // 9,6,9 all meet -> ready
    assert(!quiesce_ready(6 + 3, 4 + 2, 6 + 2, s2, 6, 4)); // arow 8 < 9 -> NOT ready
    // N-1 cross-check agrees at the ready point (N=8 -> 7 bails)
    assert( quiesce_ready_nm1(7, 8));
    assert(!quiesce_ready_nm1(6, 8));
  }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <dvgpr_occ> && g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/test_dsws_ctrl 2>&1 | head`
Expected: compile error — `snapshot_counts` / `quiesce_ready` / `WgSnap` not declared.

- [ ] **Step 3: Write minimal implementation** — append to `dsws_ctrl_model.cpp`:

```cpp
struct WgSnap { uint32_t nC, nA, nB; };

static inline WgSnap snapshot_counts(uint32_t nC, uint32_t nA, uint32_t nB) {
    return WgSnap{nC, nA, nB};
}

// Sentinels = work-threshold + snapshot role-count terminal bails (Phase A arithmetic,
// with compile-time constants replaced by the per-epoch snapshot).
static inline bool quiesce_ready(uint32_t rowblk_next, uint32_t bfrag_next,
                                 uint32_t arow_next, const WgSnap& s,
                                 uint32_t G, uint32_t FN) {
    return rowblk_next >= (G  + s.nC)
        && bfrag_next  >= (FN + s.nB)
        && arow_next   >= (G  + s.nA);
}

// Role-agnostic safety net: fixed N waves, wid0 claimer never bails -> exactly N-1 bails.
static inline bool quiesce_ready_nm1(uint32_t quiesce_cnt, uint32_t N) {
    return quiesce_cnt >= (N - 1);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <dvgpr_occ> && g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/test_dsws_ctrl && /tmp/test_dsws_ctrl`
Expected: `dsws_ctrl_model: ALL PASS`

- [ ] **Step 5: Add the interleaving race test** — append inside the same test block:

```cpp
  {
    // Under any interleaving of N-1 bails, quiesce_ready_nm1 must not fire before the last bail.
    for (uint32_t trial = 0; trial < 64; ++trial) {
      std::atomic<uint32_t> cnt{0};
      std::atomic<bool> early{false};
      std::vector<std::thread> ts;
      const uint32_t N = 8;
      for (uint32_t w = 0; w < N - 1; ++w)
        ts.emplace_back([&]{
          if (quiesce_ready_nm1(cnt.load(), N)) early.store(true); // read BEFORE our bump
          cnt.fetch_add(1, std::memory_order_acq_rel);
        });
      for (auto& t : ts) t.join();
      assert(!early.load());               // never ready with a bail still outstanding
      assert(quiesce_ready_nm1(cnt.load(), N)); // ready once all N-1 landed
    }
  }
```

- [ ] **Step 6: Run to verify pass** — same command as Step 4. Expected: `ALL PASS`.

- [ ] **Step 7: Commit** — *only if the user asks* (see Global Constraints). Otherwise report tree dirty and continue.

```bash
git add dsws_ctrl_model.cpp test_dsws_ctrl_model.cpp
git commit -m "feat(dsws): CPU model for Phase-B snapshot/quiesce + N-1 cross-check"
```

---

### Task 2: Add Phase-B LDS state to `occ_kernel_dsws.s` (offline assemble/RGA)

**Files:**
- Modify: `occ_kernel_dsws.s` (append LDS offsets after `SEGCNT_OFF`; add `DSWS2_CONV` defsym default 0)

**Interfaces:**
- Produces: LDS offsets `SNAP_BASE` (u32[6] = nC/nA/nB × 2 epoch-parity buffers), `QUIESCE_CNT_OFF` (u32). Consumed by Tasks 3–5.

- [ ] **Step 1: Add the defsym gate + offsets** — after the `SEGCNT_OFF` line, add:

```asm
.ifndef DSWS2_CONV
  .set DSWS2_CONV, 0        // 0 = pre-conversion static substrate (Phase A green); 1 = Phase B
.endif
.set SNAP_BASE,      (SEGCNT_OFF + 4)          // u32[6]: [parity*3 + {0:nC,1:nA,2:nB}]
.set QUIESCE_CNT_OFF,(SNAP_BASE + 6*4)          // u32 role-agnostic bail counter
.set DSWS2_STATE_END,(QUIESCE_CNT_OFF + 4)
```

Then move the `RES/ARES` tile region base to start at `DSWS2_STATE_END` (find the current `.set` that begins the resident region right after the controller state and repoint its base to `DSWS2_STATE_END`, so the new slots don't overlap resident A/B).

- [ ] **Step 2: Initialize the new slots in the claimer** — in `.Lclaimer` init block (near the existing `lds_put SEGCNT_OFF, 0`), add:

```asm
    lds_put QUIESCE_CNT_OFF, 0
    lds_put (SNAP_BASE + 0), NCOMP     // parity-0 snapshot = launch mix
    lds_put (SNAP_BASE + 4), NAFEED
    lds_put (SNAP_BASE + 8), NBFEED
    lds_put (SNAP_BASE + 12), NCOMP    // parity-1 = launch mix too (init)
    lds_put (SNAP_BASE + 16), NAFEED
    lds_put (SNAP_BASE + 20), NBFEED
```

- [ ] **Step 3: Assemble both no-conversion and conversion variants (offline, no GPU)**

Run:
```bash
cd <dvgpr_occ> && L=/opt/rocm/llvm/bin
for CONV in 0 1; do
  $L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    -Wa,-defsym,DSWS2=1 -Wa,-defsym,DSWS2_CONV=$CONV -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 \
    -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=1 \
    -Wa,-defsym,NCOMP=4 -Wa,-defsym,NAFEED=2 -Wa,-defsym,NBFEED=2 \
    -c occ_kernel_dsws.s -o /tmp/t2_$CONV.o 2>/tmp/t2_$CONV.err \
    && echo "CONV=$CONV ASSEMBLE_OK" || { echo "CONV=$CONV FAIL"; sed -n '1,15p' /tmp/t2_$CONV.err; }
done
```
Expected: both `ASSEMBLE_OK`.

- [ ] **Step 4: Confirm `DSWS2_CONV=0` bin is byte-identical to the Phase-A gated bin** (proves the new state is inert when conversion is off)

Run:
```bash
cd <dvgpr_occ> && /opt/rocm/llvm/bin/llvm-objcopy -O binary --only-section=.text /tmp/t2_0.o /tmp/t2_0.bin
cmp /tmp/t2_0.bin occ_dsws2_4c2a2b_gd.bin && echo "INERT-OK: CONV=0 byte-identical to Phase-A green bin" \
  || echo "REGRESSION: CONV=0 diverged from Phase-A bin — investigate before proceeding"
```
Expected: `INERT-OK`. (If the new `lds_put` init lines shift the binary, gate them under `.if DSWS2_CONV` so CONV=0 stays byte-identical — the init is only needed when conversion is on.)

- [ ] **Step 5: RGA 0-spill gate** — Run:
```bash
cd <dvgpr_occ> && KSRC=occ_kernel_dsws.s ./rga_check.sh t2_conv \
  DSWS2=1 DSWS2_CONV=1 FM=2 FN=4 G=6 SEGK=64 NCOMP=4 NAFEED=2 NBFEED=2 SAFEPROBE=1 DIAG=1 2>&1 \
  | grep -iE "SGPR_SPILLS|VGPR_SPILLS|livereg"
```
Expected: `SGPR_SPILLS=0`, `VGPR_SPILLS=0`.

- [ ] **Step 6: Assert LDS fit** — the assemble already fails if `LDS_TOTAL_DSWS2 > 32768` (line 141 guard). Confirm no assembler error mentioning that guard fired in `/tmp/t2_1.err`. Expected: no LDS-overflow error.

- [ ] **Step 7: Commit** — *only if user asks* (message: `feat(dsws): Phase-B LDS snapshot + quiesce slots (inert at DSWS2_CONV=0)`).

---

### Task 3: Port consume-point sensing into `occ_kernel_dsws.s` (offline)

**Files:**
- Modify: `occ_kernel_dsws.s` (add an `occ_sample` macro; wire a DIAG readout)

**Interfaces:**
- Produces: `occ_sample \dst_a, \dst_b` macro computing `occ_A = prod_a − min(cons_a)`, `occ_B = prod_b − min(cons_b)` at the consume point. Consumed by Task 5.

- [ ] **Step 1: Add the sensing macro** (gate under `.if DSWS2_CONV`). Mirror the coop `occ_a/occ_b` computation. Use only lean-safe temp registers (v14/v15 and scalars ≤ s65) — this code is reachable pre-grow, so a `>v15` temp is OOR-poison (SPEC §4). Read the existing claim counters (`ROWBLK_NEXT_OFF` drain vs `AROW_NEXT_OFF`, `BFRAG_NEXT_OFF`) at the point compute/feed consume, not at the segment boundary.

```asm
.if DSWS2_CONV
.macro occ_sample dst_a, dst_b
    // occ = producer - consumer, sampled where the value is consumed (SPEC §2).
    // Reuse the A-ring / B-frag claim counters already maintained by the substrate.
    lds_get \dst_a, AROW_NEXT_OFF          // A staged so far (producer side)
    lds_get s60, ROWBLK_NEXT_OFF           // A consumed so far (compute claims)
    s_sub_u32 \dst_a, \dst_a, s60          // occ_A
    lds_get \dst_b, BFRAG_NEXT_OFF         // B frags staged
    // B consume proxy: frags retired this super-tile (compute reads all FN) -> depth vs FN
    s_sub_u32 \dst_b, \dst_b, s60          // occ_B proxy (bounded [0, RINGD])
.endm
.endif
```

*Note to implementer:* the exact producer/consumer counter identities must match how `occ_kernel_dsws.s` maintains the A-ring/B-frag rings — confirm against the `.Lbfeed`/`.Lafeed`/`.Lcompute` claim sites (grep `lds_fetch_add`) before finalizing the subtraction operands. The invariant to preserve: at the consume point `occ ∈ [0, RINGD]`.

- [ ] **Step 2: Add a DIAG readout** — under `.if DIAG`, have `wid 0` write the last-sampled `occ_A`/`occ_B` to two spare `occ[]` output slots (e.g. `occ[26]`, `occ[27]`) so the harness can confirm the sensor oscillates (nonzero, varies) rather than reading a stuck 0.

- [ ] **Step 3: Assemble + RGA (offline)** — repeat Task 2 Steps 3 & 5 (CONV=1). Expected: `ASSEMBLE_OK`, `SGPR_SPILLS=0 VGPR_SPILLS=0`.

- [ ] **Step 4: Commit** — *only if user asks* (`feat(dsws): Phase-B consume-point sensing (read-only)`).

---

### Task 4: Port the `try_gate` ticket + reservation envelope into `occ_kernel_dsws.s` (offline)

**Files:**
- Modify: `occ_kernel_dsws.s` (add `try_gate` + `reserve_try` macros, transcribing coop verbatim)

**Interfaces:**
- Produces: `try_gate \dir, \swin` (sets `\swin=1` iff this wave won the `(dir,epoch)` ticket) and `reserve_try \delta, \won` (shrink always succeeds; grow validates ≤ BUDGET). Consumed by Task 5.

- [ ] **Step 1: Transcribe `try_gate` from `occ_kernel_coop.s`** — copy the proven macro (coop lines ~297–320) verbatim into `occ_kernel_dsws.s` under `.if DSWS2_CONV`. It computes `E = segcnt >> EPOCH_SHIFT`, reads `gate[dir]`, and does the single-winner `ds_cmpstore_rtn_b32` with operand order `vsrc0=new=E`, `vsrc1=cmp=g` (KG `9ed04f3c` — getting this backwards makes every racer "win"). Keep its scratch registers (coop uses `s62..s65`, `v5/v6/v7`) — verify these are free in every `occ_kernel_dsws.s` role body before finalizing.

- [ ] **Step 2: Transcribe the reservation envelope** — add `reserve_try \delta, \won`: `compute→feed` shrink = `lds_fetch_add VRESV_OFF, -delta` (always `\won=1`); `feed→compute` grow = `lds_fetch_add VRESV_OFF, +delta` then compare the returned prior+delta vs `BUDGET`; if over, `lds_fetch_add VRESV_OFF, -delta` and `\won=0`. Transcribes `reserve_grow` in `dsws_ctrl_model.cpp` exactly.

- [ ] **Step 3: Add a DIAG self-test path** — under `.if DIAG && DSWS2_CONV`, add a compile-time-selectable stub (`DSWS2_TICKET_SELFTEST` defsym) where every non-claimer wave calls `try_gate 0, s50` once and writes the win-count to an `occ[]` slot. Assemble-only; used to sanity-check the ticket wins exactly once per epoch on GPU in Task 6 if desired.

- [ ] **Step 4: Assemble + RGA (offline)** — Task 2 Steps 3 & 5 (CONV=1). Expected `ASSEMBLE_OK`, 0 spills. Also assemble with `DSWS2_TICKET_SELFTEST=1` → `ASSEMBLE_OK`.

- [ ] **Step 5: Commit** — *only if user asks* (`feat(dsws): port try_gate ticket + reservation envelope`).

---

### Task 5: Wire bail-time commit + claimer snapshot/quiesce (offline integration — the crux)

**Files:**
- Modify: `occ_kernel_dsws.s` (`.Lcompute`/`.Lafeed`/`.Lbfeed` terminal-bail paths; `.Lclaimer_wait_done` + broadcast)

**Interfaces:**
- Consumes: `occ_sample`, `try_gate`, `reserve_try`, `WgSnap` sentinel arithmetic (Task 1), the LDS slots (Task 2).
- Produces: the complete Phase-B conversion path (`DSWS2_CONV=1`).

- [ ] **Step 1: Add the decision at each role's kseg boundary** — under `.if DSWS2_CONV`, before a non-claimer wave loops to claim the next super-tile: `occ_sample s_a, s_b` → `watermark_decision` (inline: `occ<LOW`→dir starve, `occ>HIGH`→dir over-serve) → `try_gate dir, s_win`. Store `s_win` + intended `dir` in private scalars (NOT LDS). All temps v14/v15 / scalar-only (pre-grow OOR guard).

- [ ] **Step 2: Add the commit at the terminal bail — ordered BEFORE the QUIESCE_CNT bump** — in each role's terminal-bail path (where it currently exits the super-tile), insert, guarded by `s_win`:

```asm
    // --- Phase B bail-time commit (SPEC §3.4); runs only if this wave won a ticket ---
    // (a) floor guard: CAS-dec source slot only if > 1
    // (b) reserve_try delta, s_ok   (shrink always ok; grow may abort)
    // (c) on ok: CAS role slots (dec source / inc dest); flip private role reg;
    //     s_alloc_vgpr GROW(NFV)/SHRINK(32) with SCC-retry
    // (d) on floor-fail or reserve-abort: cancel conversion, remain current role
    // ORDERING: all of the above completes BEFORE the QUIESCE_CNT increment below.
```

Then the existing terminal path increments the (new) `QUIESCE_CNT`:

```asm
    lds_fetch_add s61, QUIESCE_CNT_OFF, 1     // exactly one bump per non-claimer wave/super-tile
```

Implementer: the CAS-dec floor guard and the role-slot dec/inc are `ds_cmpstore_rtn_b32` loops on `NCOMP_SLOT`/`NAFEED_SLOT`/`NBFEED_SLOT`; the ticket already bounds concurrency to ≤2 writers/boundary. Keep `s_alloc_vgpr` GROW/SHRINK exactly as coop does (SCC-retry loop), and keep every pre-grow temp in v14/v15.

- [ ] **Step 2b: Guard the `s_alloc_vgpr` OOR-poison window (SPEC §4 — #1 brick risk)** — audit every register live across the GROW in Step 2c. Any LDS/atomic temp read *before* GROW completes must be v14/v15. Add an in-code comment block marking the pre-grow window, exactly as `occ_kernel_coop.s` does. **This is the review focus before Task 7.**

- [ ] **Step 3: Switch the quiesce sentinels to the snapshot** — in `.Lclaimer_wait_done`, replace the compile-time constants:

```asm
    // BEFORE (Phase A):  ROWBLK_NEXT >= G+NCOMP ; BFRAG_NEXT >= FN+NBFEED ; AROW_NEXT >= G+NAFEED
    // AFTER  (Phase B):  read this-epoch parity snapshot, size sentinels from it.
    lds_get s45, EPOCH_OFF
    s_and_b32 s45, s45, 1                       // parity
    s_lshl_b32 s45, s45, ...                    // -> byte offset into SNAP_BASE (parity*12)
    // load snap.nC/nA/nB, compute G+nC, FN+nB, G+nA, compare against the three claim counters
```

Keep the existing three-counter structure — only the RHS changes from constants to `G + snap.nC` etc. Additionally gate the advance on `QUIESCE_CNT_OFF >= (WAVES-1)` and, under `.if DIAG`, assert it agrees with the three snapshot sentinels (write a mismatch flag to an `occ[]` slot).

- [ ] **Step 4: Snapshot at broadcast + reset QUIESCE_CNT** — in the claimer's next-super-tile broadcast (near `lds_put STI_OFF` / epoch bump), after quiesce passes: write live `NCOMP/NAFEED/NBFEED_SLOT` into the `[E+1 parity]` `SNAP_BASE` slots, then `lds_put QUIESCE_CNT_OFF, 0`, then bump epoch LAST (preserve the `STI_OFF`-before-`EPOCH_OFF` ordering).

- [ ] **Step 5: Assemble all variants + RGA (offline)** — Run:
```bash
cd <dvgpr_occ> && L=/opt/rocm/llvm/bin
for MIX in "4 2 2" "6 1 1" "2 3 3"; do read NC NA NB <<< "$MIX"
  $L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DSWS2=1 \
    -Wa,-defsym,DSWS2_CONV=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
    -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=1 -Wa,-defsym,NCOMP=$NC -Wa,-defsym,NAFEED=$NA -Wa,-defsym,NBFEED=$NB \
    -c occ_kernel_dsws.s -o /tmp/t5.o 2>/tmp/t5.err && echo "$MIX ASSEMBLE_OK" || { echo "$MIX FAIL"; sed -n '1,15p' /tmp/t5.err; }
done
```
*(Run this via `bash script.sh` — fish won't `read <<<` the same way.)* Expected: all three `ASSEMBLE_OK`.
Then RGA (Task 2 Step 5) → `SGPR_SPILLS=0 VGPR_SPILLS=0`.

- [ ] **Step 6: Dry-print sanity (offline, no GPU)** — Run:
```bash
cd <dvgpr_occ> && DSWS2_DRYRUN=1 DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2 DSWS2_NKSEG=1 \
  ./occ_dispatch --dsws2 4c2a2b 2>&1 | grep -iE "NCOMP|n_kseg|tier|REFUSE"
```
Expected: prints `NCOMP=4 NAFEED=2 NBFEED=2`, no `REFUSE`. (Rebuild `occ_dispatch` first only if `occ_dispatch.cpp` changed — it should not for Phase B.)

- [ ] **Step 7: Re-run the CPU control model** — `g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/test_dsws_ctrl && /tmp/test_dsws_ctrl` → `ALL PASS` (guards the transcription reference didn't drift).

- [ ] **Step 8: Commit** — *only if user asks* (`feat(dsws): Phase-B bail-time commit + claimer snapshot/quiesce`).

---

### Task 6: [SUPERVISED GPU] Static-mix-through-conversion re-baseline gate

**Goal:** Prove the conversion code path is *inert-safe* — with watermarks set so **no conversion fires**, reproduce the Phase-A green across all 3 mixes × 2 tiers. This isolates "did adding the conversion machinery regress the proven substrate" from "does conversion itself work" (Task 7).

- [ ] **Step 1: Build the conversion bins** — assemble `occ_dsws2_{4c2a2b,6c1a1b,2c3a3b}_gd.bin` with `DSWS2_CONV=1` (Task 5 Step 5 loop + `llvm-objcopy` to `.bin`). Confirm each 0-spill.

- [ ] **Step 2: STOP — request greenlight.** Present the offline gates (assemble, RGA, dry-print, CPU model all green) and ask kmbandy to greenlight the first Phase-B GPU dispatch. Do not proceed without explicit go.

- [ ] **Step 3: Run the re-baseline sweep** — set watermarks unreachable so no conversion fires (e.g. `DSWS_LOW=0` and `DSWS_HIGH` ≥ RINGD, or a `DSWS2_CONV_FREEZE=1` env if wired). Use the Phase-A driver pattern (`scratchpad/run_mix_gates.sh`, `DSWS_NCOMP/NAFEED/NBFEED` matching each mix, `DSWS2_NKSEG` 1 then 8, `timeout 30`, dmesg check, abort-on-anomaly).
Expected per gate: `dsws2 oracle CLEAN`, `ok=1536 bad=0`, `occ[0]=0`, exit 0, dmesg silent. **Under DIAG: the `QUIESCE_CNT==N−1` vs snapshot-sentinel agreement flag must read agree (0 mismatches).**

- [ ] **Step 4: On any anomaly (bad>0, hang/124, dmesg fault, DIAG mismatch): full STOP + bisect.** Report; do not run Task 7.

- [ ] **Step 5: Record** — on all-green, write a KG note (mirror the Phase-A milestone entry) and update the plan checkboxes.

---

### Task 7: [SUPERVISED GPU] Dynamic-mix gate (conversions firing)

**Goal:** Watermarks that *do* fire conversions; oracle stays green as the partition moves.

- [ ] **Step 1: Round-table the `s_alloc_vgpr` OOR window (Task 5 Step 2b) before any dispatch.** Fable + Codex adversarial review of the pre-grow register discipline in the commit path — this is the #1 brick risk. Fix findings offline; re-gate (assemble/RGA).

- [ ] **Step 2: STOP — request greenlight** for the first conversion-firing dispatch.

- [ ] **Step 3: Run TIGHT first, single mix** — `DSWS2_NKSEG=1` (exact tier), `4c2a2b`, watermarks that fire (`DSWS_LOW=1 DSWS_HIGH=RINGD-1 EPOCH_SHIFT=3`), `timeout 30`, dmesg check.
Expected: `oracle CLEAN ok=… bad=0`, `occ[0]=0`, exit 0, dmesg silent, DIAG agreement. The conversion actually fired: confirm via a DIAG conversion-count `occ[]` slot > 0.

- [ ] **Step 4: On green, widen** — LOOSE tier (`DSWS2_NKSEG=8`), then the other two mixes, one supervised dispatch at a time. Any anomaly → full STOP + bisect.

- [ ] **Step 5: Record** — KG note + checkboxes on all-green.

---

### Task 8: [SUPERVISED GPU] Storm gate (race-hunt)

**Goal:** The lock-free race stress that historically caught 136/552 — tight watermarks + `EPOCH_SHIFT=0` (every segment its own epoch = max conversion pressure) + ×10 repeats.

- [ ] **Step 1: STOP — request greenlight.**

- [ ] **Step 2: Run the storm** — `EPOCH_SHIFT=0`, `DSWS_LOW`/`DSWS_HIGH` tight, loop each (mix × tier) ×10 via a `bash` driver, `timeout 30` each, dmesg check between. Expected every iteration: `oracle CLEAN bad=0`, exit 0, dmesg silent, DIAG agreement, conversion-count > 0.

- [ ] **Step 3: On any single-iteration anomaly: full STOP + bisect** (a storm failure is a real race — do not average it away).

- [ ] **Step 4: Record the Phase-B completion milestone** — KG session_summary: dynamic role conversion GPU-proven through storm, both tiers, all mixes, zero bricks. Note Phase 4 (adaptivity + tuning + `--att` issue-mix on ml8 `down`/`down_pf`) as the next campaign, separate spec.

---

## Self-Review

**Spec coverage:** Decision 1 (snapshot quiesce) → Tasks 1,2,5; Decision 2 (bail-time commit) → Task 5; sensing §2 → Task 3; ticket+reservation §3 → Task 4; safety/OOR §4 → Task 5 Step 2b + Task 7 Step 1; control-model §5 → Task 1; build sequence §6 (static→dynamic→storm) → Tasks 6,7,8; testing → offline gates in every task + GPU gates 6–8; success metric → Task 8. `N−1` DIAG safety net → Task 1 + Task 5 Step 3 + Task 6 Step 3. All spec sections covered.

**Placeholder scan:** The asm tasks intentionally show the load-bearing blocks (offsets, snapshot arithmetic, commit ordering, sentinel edit) as concrete code and mark the two spots (occ counter identities in Task 3 Step 1; try_gate scratch-register freeness in Task 4 Step 1) where the implementer must confirm against the live kernel before finalizing — these are verification instructions, not deferred design. No "TBD/add error handling/similar to Task N".

**Type/name consistency:** `WgSnap`/`snapshot_counts`/`quiesce_ready`/`quiesce_ready_nm1` consistent Task 1↔5. LDS names (`SNAP_BASE`, `QUIESCE_CNT_OFF`) consistent Tasks 2↔5. `try_gate \dir,\swin`, `reserve_try \delta,\won`, `occ_sample` consistent Tasks 3,4↔5. Gate commands use the same defsym set throughout.
