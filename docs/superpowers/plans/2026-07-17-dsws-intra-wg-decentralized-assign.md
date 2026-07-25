# DSWS Intra-WG Decentralized Assign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`.
> This is a hand-written amdgcn (gfx1201) assembly kernel. The TDD cycle is adapted: each task's "test" =
> **OFFLINE GATE** (assemble 0-spill + `DECENTASN=0` byte-identity + hot-loop grep) then a **SILICON GATE**
> (one greenlit `./gpu_run.sh`, full stride=1 oracle). Spec: `docs/superpowers/specs/2026-07-17-dsws-intra-wg-decentralized-assign-design.md`.

**Goal:** Decentralize the *assign* tier (the last one pinned to wid0) so the next-available wave grabs and emits
work, while each workgroup claims WHOLE tiles (keeping the banked on-chip reduce valid).

**Architecture:** Reuse the existing lock-free slot-reserve (`.Lflow_da_resv`/`.Lflow_da_won`, ~3328) but replace
its **global super-tile claim** (`gi = occ[20]++`) with a **WG-local whole-tile claim + decentralized (group,ksi)
cursor**. Port the coordinator's GROUPS>1 cursor + `zero_banks` + new-tile barrier onto the decentralized path.
Then compose stagger/baton/deep-J. All new code `.if DECENTASN`-gated.

**Tech Stack:** amdgcn asm (`occ_kernel_dsws_flow.s`), `build_flow.sh` (assemble), `./gpu_run.sh` (dispatch),
host `occ_dispatch.cpp`, CPU oracle.

## Global Constraints (verbatim from spec §6 — every task)
- **G1 same-WG-combine:** a WG owns a whole tile. Never re-introduce a global super-tile claim.
- **River:** NO blocking read / cap / wait / hard partition in the hot loop. Assign is bounded-CAS-then-bail.
- **`MSDRAIN=1`** for all POOL_N>1 runs. **`WOFLUSH=0 BANKZERO=1`** (banked). **`RBU=1`**.
- **Off-path inertness:** `DECENTASN=0` build stays byte-identical `22bc8d0d` (baton geom). All edits `.if DECENTASN`.
- **Verdicts:** full stride=1 oracle for any correctness claim; `computed == G*MTLsuper*NTL*n_kseg` every run; fed
  (deep-K, ≥1s) before any TF verdict.
- **Safety:** dispatch ONLY via `./gpu_run.sh`; ONE greenlit dispatch at a time; changed kernel = ONE bring-up then
  STOP+report; hang/DMFAT/oracle-BAD/INCOMPLETE = full stop; DEADMAN 0.5s never raised. Commit ONLY when kmbandy asks.
- Host geometry MUST match the bin (`DSWS2_G/ACC_N/FLOW_POOL_N/DSWS2_SEGK/FLOW_WAVES=30`) or silent no-launch.

---

## File Structure
- **Modify:** `occ_kernel_dsws_flow.s` — the only kernel file. Regions: the DECENTASN assign block
  (`.Lflow_da_*`, ~3328–3420); the LDS layout (`COORD_T_OFF`/`COORD_KSI_OFF`/`ASSIGN_LOCK_OFF`, ~506–512); the
  guards (`DECENTASN && JDEPTH>1` line ~778); `zero_banks` reachability (~905–932). No new files.
- **Possibly modify:** `occ_dispatch.cpp` only if a new occ diagnostic slot is needed (avoid if possible).
- **Log:** append one entry per silicon gate to `DSWS_TESTING_LOG.md`; update `DSWS_STATE.md` at session end.

---

## Task 1: WG-local tile claim + decentralized (t,ksi) cursor — J=1, GROUPS=1

**Files:** Modify `occ_kernel_dsws_flow.s`: the `.Lflow_da_won` claim (~3358–3401); add a WG-local
`(current_tile, ksi_cursor)` in LDS; make `zero_banks` + new-tile barrier reachable from the decentralized path.

**Interfaces:**
- Consumes: existing `SL_STI`/`SL_GEN`/`SL_RBNEXT`/slot layout; `zero_banks` macro; `occ[20]` global tile pool;
  `s68`=ceil-log2 shift, `s66`=COUNT(=n_kseg-1), `s69`=chunkHi.
- Produces: a decentralized assign that stamps `SL_STI = (t<<shift)|ksi` where `t` is a WG-owned tile and `ksi`
  walks `0..n_kseg-1`; a WG-local cursor word contract used by Task 2.

- [ ] **Step 1 — Add the WG-local cursor LDS words.** In the LDS layout (~506–512), define (DECENTASN-only, no
  alias with `ROLE`/`COORD` — reuse the now-freed coordinator `COORD_T_OFF`=144 / `COORD_KSI_OFF`=140 since wid0
  is unreachable under DECENTASN, per doc §2): `DA_TILE_OFF` = current WG tile `t` (init `0xFFFFFFFF` sentinel =
  "no tile, claim one"); `DA_KSI_OFF` = next ksi to emit (init 0). Add a `.if DECENTASN` init in the barrier-free
  LDS-init (~2449, next to `ASSIGN_HEAD=0`).

- [ ] **Step 2 — Replace the global super-tile claim.** In `.Lflow_da_won` (3359–3372), REMOVE the per-assign
  `gi = occ[20]++` global super-tile. Instead, after winning the slot-reserve `r` (s44):
  - Lane-0 reads `DA_KSI_OFF` (ksi) and `DA_TILE_OFF` (t).
  - If `ksi <= s66` (COUNT) AND `t != 0xFFFFFFFF`: **same tile** — `gi = (t<<shift)|ksi`; CAS-advance `DA_KSI_OFF`
    `ksi→ksi+1` (if the CAS loses, another wave advanced it → re-read and retry within the retry budget).
  - Else **tile exhausted / none**: claim a NEW WG tile — lane-0 `occ[20]++` → `t_new`; if `t_new >= s69`
    (chunkHi) → terminal (existing `.Lflow_da_termslot` rollback path); else drain-gate (`DRAIN>=ASSIGN`, the
    new-tile bank-reuse barrier) — if not drained, roll back the slot-reserve and bail (retry next iter);
    when drained, `zero_banks`, write `DA_TILE_OFF=t_new`, `DA_KSI_OFF=1`, `gi=(t_new<<shift)|0`.
  - Exactly-one-claims: the `occ[20]++` is atomic (unique t per WG); the "who claims" race is resolved by a CAS
    on `DA_TILE_OFF` (`0xFFFFFFFF→t_new`) — loser rolls back its `occ[20]` tile? No: to avoid leaking a claimed
    tile, gate the `occ[20]++` behind winning a `DA_TILE_OFF` transition CAS FIRST (reserve the right to claim),
    then `occ[20]++`. Enumerate this in Step 6.
- [ ] **Step 3 — Stamp unchanged.** The existing SL_* reset + `SL_STI=gi` + `SL_GEN=r` release-fence (3373–3400)
  stays byte-for-byte; only the source of `gi` changed. At J=1, `SL_RBNEXT=RB_PENDING` (3384) unchanged.

- [ ] **Step 4 — OFFLINE GATE.**
```
# byte-identity of the off path:
FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=32 STAGGER=1 BATONGATE=1 MSDRAIN=1 \
  RBU=1 STAGINSTR=1 TFPROBE=1 DECENTASN=0 ./build_flow.sh && md5sum occ_dsws2_w30_flow_gd.bin
# EXPECT: 22bc8d0d7d45e99f198f144782c66767
# the decentralized build (J=1 GROUPS=1) assembles 0-spill:
FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=1 MSDRAIN=1 DECENTASN=1 RBU=1 \
  STAGINSTR=1 TFPROBE=1 ./build_flow.sh
grep -iE 'spill|error' /tmp/flow_build.err   # EXPECT empty
```
- [ ] **Step 5 — Hot-loop grep (river check).** Confirm NO new blocking read / spin / cap in the assign:
```
sed -n '/\.Lflow_da_resv/,/\.Lflow_loop/p' occ_kernel_dsws_flow.s | grep -nE 's_sleep|spin|\.L.*wait'
# EXPECT: only the existing bounded retry-budget bail (s48>=4 -> feedmt_sleep), NO unbounded loop.
```
- [ ] **Step 6 — Write the claim-enumeration** (in a comment block above `.Lflow_da_won`): every `DA_KSI` /
  `DA_TILE` transition, the CAS ownership, and the pairing that exactly one wave claims each tile and runs
  `zero_banks` once, drain-gated. (The O1/S2 discipline.)

- [ ] **Step 7 — SILICON GATE A (one greenlit dispatch).** Real ml8 shape, **full oracle**:
```
./gpu_run.sh da_gateA_j1g1 -- FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=3 \
  DSWS2_SEGK=32 DSWS2_K=32768 DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=32 ML8_COOP_CHUNK=384 \
  ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=1 ./occ_dispatch --dsws2
```
Expected: `computed == 6*6*32*1024 = 1179648` (work-exact), **oracle CLEAN bad=0 [full stride=1]**, `occ[0]=0`,
no DMFAT, no reset. This proves the WG-local decentralized claim is correct — the exact thing the global version
failed. Then STOP + report + append to `DSWS_TESTING_LOG.md`.

- [ ] **Step 8 — Commit** (ONLY if kmbandy asks): `git add occ_kernel_dsws_flow.s && git commit -m "dsws: intra-WG decentralized assign (WG-local tile claim), J=1 GROUPS=1"`.

---

## Task 2: Port GROUPS>1 cursor onto the decentralized assign

**Files:** Modify `occ_kernel_dsws_flow.s`: the Task-1 cursor advance (add group dimension); mirror the
coordinator's group-boundary `zero_banks` + group-aware STAMP already in `.Lflow_same_tile`.

**Interfaces:**
- Consumes: Task 1's `DA_TILE_OFF`/`DA_KSI_OFF` contract; the coordinator's proven GROUPS>1 logic (COUNT-based
  `(group,ksi)` advance, group-boundary drain+`zero_banks`, `STAMP=(group<<28)|sti`, first-crosser TILEDONE).
- Produces: a decentralized cursor that walks `(group, ksi)` for `group=0..GROUPS-1, ksi=0..n_kseg-1`.

- [ ] **Step 1 — Extend the cursor to (group, ksi).** In the Task-1 exhaust check, replace `ksi<=COUNT` with the
  COUNT-based `(group,ksi)` walk from the coordinator: advance ksi 0..COUNT, at `ksi==COUNT` roll `group+1,
  ksi=0`; at `group==GROUPS` the tile is exhausted → claim new tile. Stamp `SL_STI = (group<<28)|((t<<shift)|ksi)`
  (the `STAMP_GSHIFT` packing). At each **group boundary** (ksi wraps, group>0), drain-gate + `zero_banks` (mirror
  the coordinator's `.Lflow_same_tile` group-boundary logic added 2026-07-17).
- [ ] **Step 2 — OFFLINE GATE.** `DECENTASN=0` still `22bc8d0d`. Build `DECENTASN=1 JDEPTH=1 G=18 ACC_N=6`
  (GROUPS=3) 0-spill, LDS fits. Grep: no new blocking read.
- [ ] **Step 3 — SILICON GATE B (one greenlit).** `G=18 ACC_N=6 GROUPS=3 J=1`, real shape, **full oracle**:
```
./gpu_run.sh da_gateB_g18grp3 -- FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=18 DSWS2_ACC_N=6 FLOW_POOL_N=2 \
  DSWS2_SEGK=32 DSWS2_K=32768 DSWS2_ORACLE_MTL=2 DSWS2_ORACLE_NTL=64 ML8_COOP_CHUNK=384 \
  ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=1 ./occ_dispatch --dsws2
```
Expected: `computed == 18*2*64*1024 = 2359296` work-exact, **oracle CLEAN bad=0 [full]**, occ0=0, no DMFAT. STOP+report+log.
- [ ] **Step 4 — Commit** (only if asked).

---

## Task 3: Solve S1 (deep-J poison encoding) + enable JDEPTH>1

**Files:** Modify `occ_kernel_dsws_flow.s`: the `SL_RBNEXT` encoding + the poison sites (`.Lflow_da_won` stamp
3378–3384; the compute claim CAS; `side_final`); the guard at ~778.

**Interfaces:**
- Consumes: Task 2's decentralized (group,ksi) assign.
- Produces: `SL_RBNEXT` that carries BOTH the poison-until-staged state AND the deep-J lead/non-lead marker.

- [ ] **Step 1 — Choose + document the encoding** (spec S1). Widen `SL_RBNEXT`: keep `RB_PENDING=0xC0000000`
  (unstaged), add a distinct `J_NONLEAD` sentinel in a free high bit (e.g. `0x20000000`) for non-lead poison, so
  `next` low bits (`0..ACC_N`) coexist with both flags. Write the truth table (staged+lead-claimable / staged+
  non-lead-poison / unstaged) as a comment; prove exactly-once arm + never-false-claimable.
- [ ] **Step 2 — Wire it.** At `.Lflow_da_won` (3378–3384): lead (`ksi%J==0`) → `RB_PENDING` (poison until staged,
  then armed to `0`=claimable); non-lead → `RB_PENDING | J_NONLEAD` (stays un-claimable). The claim CAS + stage-
  arm (`side_final`) updated to the new truth table. **Delete/relax the guard at ~778** (replace `.error` with the
  new encoding's `.if` sanity checks).
- [ ] **Step 3 — OFFLINE GATE.** `DECENTASN=0` `22bc8d0d`. Build `DECENTASN=1 JDEPTH=2 POOL_N=2` (J≤POOL_N),
  0-spill. Enumerate every `next++`↔`RBDONE++` for the decentralized J-carrier (O1 for the jloop).
- [ ] **Step 4 — SILICON GATE C (one greenlit).** `G=18 GROUPS=3 J=2`, real shape, **full oracle** (same shape as
  Gate B but `DSWS2_SEGK=32 JDEPTH=2` build). Expected: work-exact, **oracle CLEAN [full]**, occ0=0, no DMFAT. STOP+report+log.
- [ ] **Step 5 — Commit** (only if asked).

---

## Task 4: Compose stagger + baton + tier order; measure the payoff

**Files:** Modify `occ_kernel_dsws_flow.s` only if the tier-priority order (spec §4) needs the dispatch reordered;
otherwise measurement-only.

**Interfaces:** Consumes Tasks 1–3 (decentralized assign + GROUPS + deep-J). Produces the fed A/B answering
whether decentralized assign lifts the assign-bound and lets the budget bind + the baton engage.

- [ ] **Step 1 — Tier order (spec §4 / Q1).** If deliver-first (O-A) requires reordering the `.Lflow_dispatch`
  ladder, make that change `.if DECENTASN`-gated; else note the current order already realizes it.
- [ ] **Step 2 — OFFLINE GATE.** `DECENTASN=0` `22bc8d0d`. Build the full compose: `DECENTASN=1 STAGGER=1
  BATONGATE=1 JDEPTH=2 G=18 ACC_N=6 MSDRAIN=1`, 0-spill, no `.error`.
- [ ] **Step 3 — SILICON GATE D (one greenlit), FED.** Real ml8 shape scaled for ≥1s steady state (deep-K, more
  tiles/WG), sampled oracle for correctness sanity + the perf readout:
```
./gpu_run.sh da_gateD_compose_FED -- FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=18 DSWS2_ACC_N=6 \
  FLOW_POOL_N=2 DSWS2_SEGK=32 DSWS2_K=262144 DSWS2_ORACLE_MTL=16 DSWS2_ORACLE_NTL=64 ML8_COOP_CHUNK=256 \
  ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=128 ./occ_dispatch --dsws2
```
Read: `occ[86]` STARVATION (was ~92% on wid0 — does it drop?), `door4 grow-fail` (does budget bind now?),
`occ[98]` baton (does it engage?), TF, `computed` work-exact. This is the payoff measurement the whole
architecture exists for. STOP+report+log.
- [ ] **Step 4 — If O-A vs O-C undecided:** one greenlit A/B run swapping the tier order, fed identically, compare.
- [ ] **Step 5 — Commit + update `DSWS_STATE.md`** (only if asked).

---

## Self-Review
- **Spec coverage:** §3 change → Task 1; §3.3 GROUPS port → Task 2; §5 S1 → Task 3; §5 S2 → Task 1 Step 2/6;
  §4 tier order → Task 4 Step 1; §7 gates A/B/C/D → Task 1/2/3/4 silicon steps. All covered.
- **Placeholder scan:** every step has exact build/dispatch commands + expected `computed`/oracle values. No TBD.
- **Consistency:** `DA_TILE_OFF`/`DA_KSI_OFF` named identically across Tasks 1–2; `SL_RBNEXT` encoding defined in
  Task 3 Step 1 before use in Step 2; expected `computed = G*MTLsuper*NTL*n_kseg` applied per shape each gate.
- **Known limitation surfaced:** the exact register allocation / final asm is written during execution by reading
  the surrounding code — the plan specifies the LDS/label/control-flow contract + the offline & silicon gates,
  which is the correct granularity for this hand-asm kernel (writing final register-allocated asm blind would be
  a guess; the offline gate catches spills/aliases).
