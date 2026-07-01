# DSWS Substrate v2 — Phase B: Runtime Role Conversion (design)

**Status:** approved design (2026-07-01). Realizes *step 2* of the build sequence in
`SPEC_DSWS_SUBSTRATE_V2.md` §6 ("Add conversion"). Phase A (static claim-based split-K
substrate) is GPU-proven green: all 3 role mixes × 2 split-K tiers oracle-CLEAN, bit-exact,
zero bricks (KG `0c5537e6`, 2026-07-01). This spec covers only the conversion actuation added
on top of that green substrate.

**Goal (one sentence):** Let a workgroup move waves between {compute / A-feed / B-feed} at
runtime — the DSWS payload — on the claim-based v2 substrate, without orphaning output, jamming
a feed, or bricking, so the partition `(nComp, nAfeed, nBfeed)` self-tunes to the bottleneck.

## Relationship to prior work

- The **conversion control law** already exists and is unit-tested in `dsws_ctrl_model.cpp`
  (`watermark_decision`, `epoch_of`, `gate_try_win`, `reserve_grow`) and is transcribed
  exactly by the `try_gate` macro + reservation logic in `occ_kernel_coop.s`. Phase B **ports**
  that proven actuation into `occ_kernel_dsws.s` and binds it to the claim-based role branch —
  it does not re-derive it.
- The blocker that killed the *coop* substrate's conversion (work bound to wave identity;
  KG `86e33108`) is structurally absent in v2: a converted wave inherits **no work item**, it
  just changes its role tag and starts claiming from the dest role's counters. This spec relies
  on that property and does not re-litigate it.

## The two locked design decisions

### Decision 1 — Quiesce model: per-epoch snapshot (+ `N−1` DIAG assert)

The claimer's `.Lclaimer_wait_done` quiesce currently waits on **compile-time**
`NCOMP/NAFEED/NBFEED` (`ROWBLK_NEXT ≥ G+NCOMP`, `BFRAG_NEXT ≥ FN+NBFEED`,
`AROW_NEXT ≥ G+NAFEED`). Correct for static roles only. Once conversions move the partition,
those constants are wrong → the claimer advances early (straggler races the counter reset →
brick / stale-resident wrong-oracle) or waits forever (deadlock).

**Fix:** the claimer snapshots the *live* role counts `(nComp, nAfeed, nBfeed)` at super-tile
broadcast into per-epoch LDS slots; the quiesce sentinels read the snapshot, not the constants:
`ROWBLK_NEXT ≥ G + snap.nComp`, `BFRAG_NEXT ≥ FN + snap.nBfeed`, `AROW_NEXT ≥ G + snap.nAfeed`.

**Why this model** (vs. the two rejected alternatives): smallest diff from the GPU-proven
Phase-A quiesce (swap three constants for three LDS reads — tiny bisect surface); preserves the
per-operand tripwire (a jammed single role's counter is still caught *at* the quiesce, not only
downstream at the oracle); and maps 1:1 onto `dsws_ctrl_model.cpp` so the snapshot/quiesce
interaction is CPU-unit-testable offline before any GPU dispatch.
- Rejected: **role-agnostic `N−1` counter alone** — more robustly partition-independent, but
  changes more proven code and blinds the per-operand tripwire.
- Rejected: **live per-role drained counters w/ mid-super-tile conversion** — reopens the exact
  cross-wave ordering hazards v2 was built to avoid; reactivity payoff is moot given split-K's
  already-short super-tiles. Deferred to Phase 4 only if measured need appears.

**Safety net (the rejected alternative, kept as an assert):** the `N−1` counter is compiled in
as `QUIESCE_CNT` and, under `DIAG`, cross-checked against `Σ snapshot sentinels`. Because wave
count is fixed and `wid 0` (claimer) never converts, exactly `N−1` non-claimer waves are alive
each super-tile and each does exactly one terminal bail — so `QUIESCE_CNT == N−1` must coincide
with the three snapshot sentinels being satisfied. A disagreement is an ordering bug, caught
immediately (offline model + supervised runs).

### Decision 2 — Actuation mechanism: bail-time commit (Approach A)

A converting wave **decides** during the super-tile but **commits** at its terminal bail,
sequenced immediately *before* it increments `QUIESCE_CNT`.

**Why this composes with Decision 1 for free:** the claimer's quiesce already waits for all
`N−1` bails. If the role-slot CAS (the commit) is ordered *before* the bail-count bump, then
"quiesce satisfied" *implies* "all conversions for this boundary have landed" — the quiesce
counter **is** the publish/snapshot handshake. The claimer therefore snapshots the E+1
partition only after every commit is visible. No extra barrier, no second ordering, no
intent-buffer. (Rejected: claimer-mediated commit — splits the commit across two waves and adds
a claimer→wave-resize ordering, since `s_alloc_vgpr` must run on the converting wave anyway.
Rejected: immediate mid-tile commit + pending-counter — reintroduces mid-super-tile population
mutation = Option-3 hazard.)

## Global constraints (inherited verbatim from `SPEC_DSWS_SUBSTRATE_V2.md`)

- A GPU brick is a **BUG**, never a tax. A hang = full STOP + report; never auto-fire the next
  variant. **The user greenlights EVERY GPU dispatch individually.**
- Display GPU → only compositor-safe chunked sub-second dispatches
  (`ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, `timeout 30`).
- **NEVER `--gl2c`.** SAFEPROBE + bounds gate + padding stay ON. `ML8_COOP_STREAM=1` always.
- **No `s_barrier`** (mixed dyn-VGPR + `s_barrier` hard-deadlocks — proven). Pure LDS-atomic
  busy-wait coordination only.
- `occ_kernel_coop.s` is **never modified** — known-good reference. All work is additive in
  `occ_kernel_dsws.s`.
- Commit to git only when the user asks. Single-variable isolation; fix bugs, don't dodge them.

## Section 1 — New LDS state

**Symbols (inherited from `SPEC_DSWS_SUBSTRATE_V2.md` / the coop layer):** `N` = total waves/WG
(fixed at launch); `G=6` cooperative M-extent (= `nComp` ceiling); `FN=4`/`FM=2` N/M frag
counts; `NFV` = fat-compute VGPR target (grow size); `VLEAN=32` = lean feed VGPR; `BUDGET` =
per-SIMD VGPR budget the sum-envelope must not exceed; `RINGD` = ring depth; `SEGK=64`,
`n_kseg = KT/(SEGK/16)`.

Ported from the coop actuation layer (same relative layout), plus two Phase-B additions:
- `GATE_OFF[4]` — `gate[dir]` = last epoch direction `dir` fired (the `try_gate` CAS target).
- `VRESV_OFF` — `vgpr_reserved` sum-envelope counter. Init = `nComp·NFV + (nAfeed+nBfeed)·VLEAN`.
- `SEGCNT_OFF` — per-WG decision clock source (`epoch = segcnt >> EPOCH_SHIFT`).
- Thresholds `CTRL_LOW`, `CTRL_HIGH = RINGD−1`, `EPOCH_SHIFT` (mirror `occ_dispatch` env).
- **NEW — `SNAP_NC/NA/NB`**, double-buffered by epoch parity (`[E&1]`) so the snapshot being
  read for `quiesce(E)` is never clobbered by the claimer staging `E+1`.
- **NEW — `QUIESCE_CNT`** — role-agnostic `N−1` bail counter (Decision 1 safety net + advance
  gate).

The role-count slots `NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT` and `VRESV_OFF` already exist in
`occ_kernel_dsws.s` (published by the claimer today); Phase B makes them **mutable** (CAS'd on
conversion) instead of write-once. LDS budget delta is a handful of u32 slots — re-assert the
32 KB group-segment fit (Phase A used 16640 B of 16896 B alloc; ample margin).

## Section 2 — Sensing (faithful port)

`occ_X` = ring producer minus consumer at the **consume point**:
`occ_A = prod_a − min(cons_a)`, `occ_B` analogously from the B-frag claim counters. Sampled
where the value is consumed, **not** at the segment boundary — the coop lesson (KG `0a3135b2`):
at the boundary the ring is drained so `occ ≈ 0` and the controller would read "always
starved." Fed to `watermark_decision(occ, CTRL_LOW, CTRL_HIGH)`:
`occ_X < LOW` → compute starved for X → shrink compute → feed-X; `occ_X > HIGH` → feed-X
over-serving → grow feed-X → compute. Read-only; no actuation here.

## Section 3 — Conversion lifecycle (Approach A, per non-claimer wave)

At each **kseg boundary** (= super-tile boundary; roles are frozen *within* a super-tile):

1. **Sense** (§2) → `watermark_decision` → candidate direction `dir` (or none).
2. **Win ticket** — `try_gate(dir)`: `E = segcnt >> EPOCH_SHIFT`; win iff `gate[dir] < E` via
   the single-winner LDS CAS (`ds_cmpstore_rtn_b32`, operand order per KG `9ed04f3c`:
   `vsrc0=new=E`, `vsrc1=cmp=g`). ≤1 winner per `(dir, epoch)`. Non-winners continue unchanged.
3. **Work to terminal bail** in the current role (unchanged claim loop).
4. **Commit — ordered strictly before the `QUIESCE_CNT` bump:**
   a. **Floor guard** — CAS-dec the source role slot only if `> 1` (compute floor ≥ 1, feed
      floor ≥ 1). Fail → abort conversion, bail as current role.
   b. **Reservation envelope** — `compute→feed` shrink: `atomic_sub(vgpr_reserved, NFV−VLEAN)`
      (always succeeds). `feed→compute` grow: `atomic_add(vgpr_reserved, NFV−VLEAN)` then
      validate `≤ BUDGET`; if over, `atomic_sub` + abort (stay in role this epoch).
   c. **CAS role slots** — dec source, inc dest (bounded to ≤2 concurrent writers/boundary by
      the ticket; plain atomic-LDS).
   d. **Flip private role register** + `s_alloc_vgpr` GROW(`NFV`) / SHRINK(32), each SCC-retry
      guarded.
5. **Bump `QUIESCE_CNT`** (every non-claimer wave does this exactly once/super-tile, converted
   or not), then enter the dest role's claim loop (or re-enter own).

**Claimer (pinned `wid 0`, never converts):** on `QUIESCE_CNT == N−1` → all commits landed →
reset per-super-tile counters, **snapshot** live `(nComp,nAfeed,nBfeed)` into the `[E+1 & 1]`
slots, clear `QUIESCE_CNT`, bump epoch (proven `TI_OFF`-before-`EPOCH_OFF` ordering). Terminal:
`sti ≥ TOTAL_super` remains the role-agnostic retire signal every role checks at its boundary.

## Section 4 — Safety / anti-brick invariants

- **Ordering contract (the crux):** commit (4c CAS) precedes the `QUIESCE_CNT` bump (step 5),
  which precedes the claimer's E+1 snapshot. Therefore the claimer never snapshots a stale
  partition. Self-enforced by the bail sequence; no barrier.
- **Floors** `nComp, nAfeed, nBfeed ≥ 1`; **ceiling** `nComp ≤ G` (asserted in the control
  model — surplus compute waves would find `rowblk_next` exhausted and idle, but floors+ceiling
  keep `nComp ≤ G` by construction).
- **`s_alloc_vgpr` OOR-poison guard (highest brick-risk item):** on RDNA4, any LDS/atomic temp
  register reachable *before* a grow must be v14/v15 under dyn — a `>v15` source pre-grow is
  poison (coop learned this; `occ_kernel_coop.s` gates every pre-grow-reachable temp to
  v14/v15). Every new pre-grow-reachable temp in the conversion path (sense, ticket, floor,
  envelope) inherits this constraint. Verified by RGA + the sense/ticket temps living in the
  lean-safe register window. **This is the single most likely place to brick — reviewed
  explicitly by the round table before the first dynamic-mix GPU dispatch.**
- **No `s_barrier`.** All coordination is LDS-atomic busy-wait.
- **Clock never converts:** `SEGCNT` is bumped by the pinned claimer (`wid 0`), which is
  non-convertible — kills the clock-stall failure mode.

## Section 5 — Control model changes (offline, TDD, no GPU)

Extend `dsws_ctrl_model.cpp` + `test_dsws_ctrl_model.cpp`:
- Model the **snapshot/quiesce interaction**: a `snapshot(E)` reads role counts; conversions
  mutate counts at the boundary; assert `quiesce(E)` uses `snap(E)` and that
  `Σ snap(E) sentinels ⟺ QUIESCE_CNT == N−1` under arbitrary interleavings (thread-race test,
  as the existing `gate_try_win` test does).
- Keep `watermark_decision / epoch_of / gate_try_win / reserve_grow` tests green (unchanged
  semantics). All `ALL PASS` before any assemble.

## Section 6 — Build sequence & gates (isolation-preserving)

Each stage a supervised GPU gate; you greenlight each; brick = full STOP + bisect. Offline
before each: `dsws_ctrl_model` tests green, RGA 0-spill, dry-print sane.

Config held at Phase-A values **`G=6, SEGK=64`** for stage 1 so the conversion code is the only
variable vs the green static gate.

1. **Static-mix through the conversion path** — conversion code wired but watermarks set so
   **none fire** (`CTRL_LOW=0` / unreachable). Must reproduce the Phase-A green (all 3 mixes ×
   both tiers, `ok=… bad=0`). Proves the ported actuation + snapshot machinery is inert-safe /
   non-regressing. **[SUPERVISED GPU — the re-baseline gate.]**
2. **Dynamic-mix** — watermarks that *do* fire conversions; oracle stays green as roles move.
   Start `n_kseg=1` TIGHT (exact) at one mix, then LOOSE, then the other mixes. **[SUPERVISED.]**
3. **Storm** — tight watermarks + `EPOCH_SHIFT = 0` + ×10 repeats: the lock-free race-hunt
   (the strong-oracle-plus-repeats discipline that caught 136/552). **[SUPERVISED.]**

Phase 4 (separate spec): adaptivity proof (converge-from-wrong-start),
`{LOW,HIGH,RINGD,EPOCH_SHIFT,G,SEGK}` sweep, `--att` issue-mix on ml8 `down`/`down_pf`.

## Testing

- **CPU oracle gate** (`fp8_oracle.cpp`): Tier-1 tight (`5e-3` rel / `1e-2` abs) at `n_kseg=1`;
  Tier-2 loose (`3e-2` / `2e-2`) at `n_kseg>1`. Before every perf run and every kernel change.
- **RGA static gate:** 0-spill, live-VGPR within budget, every assemble.
- **Control-law unit tests:** extended per §5, `ALL PASS`.
- **Storm stress:** §6 stage 3.
- **DIAG cross-check:** `QUIESCE_CNT == N−1` ⟺ `Σ snapshot sentinels` (Decision-1 safety net).

## Success metric

Oracle-green through the **storm** at dynamic mix on both tiers, all mixes, **zero bricks** —
the partition provably moves at runtime with correct output. The DSWS *thesis* payoff (beats
static baseline + `--att` shows cut non-WMMA issues on compute waves, on ml8 `down`/`down_pf`)
is Phase 4, not a Phase-B gate.

## Risks & open items

- **`s_alloc_vgpr` OOR-poison in the conversion path** — highest brick risk (§4). Round-table
  the register assignment before the first dynamic-mix dispatch.
- **Concurrent role-slot CAS** — bounded to ≤2 writers/boundary by the ticket; verify the CAS
  is genuinely atomic-LDS and the floor-guard dec/inc pair can't transiently violate a floor.
- **Snapshot double-buffer parity** — confirm `[E&1]` indexing can't alias when the pool is
  ≤2 super-tiles deep (degenerate tiny-shape oracle case).
- **`vgpr_reserved` under grow-abort** — the atomic_add-then-sub-on-over-budget must not leave a
  transient over-count visible to a *second* concurrent grower; both are ticket-serialized per
  epoch, but verify across directions.
- **Round-table structure** (kmbandy's): implement (Sonnet) → adversarial review (Fable + Codex)
  → kmbandy greenlights each GPU dispatch. It caught 5 offline bricks in Phase A; keep it.
