# DSWS — Intra-WG Decentralized Assign (Fork A) — Design Spec

**Date:** 2026-07-17 · **Author:** claude__main + kmbandy · **Task:** #45
**Authorization:** kmbandy explicit (2026-07-17) — see `DSWS_STATE.md` DECISION JOURNAL.
**Governing model:** `DSWS_STATE.md` § DEFINITIVE ARCHITECTURE (the river / tier system).

---

## 1. Goal (one sentence)

Decentralize the **assign** tier so the *next available wave* grabs and emits work — removing the single-wid0
producer bottleneck — **while keeping each workgroup's claim WG-local (whole tiles)** so the on-chip (banked)
reduce stays valid. This is the one tier of the river still centralized; every other tier already flows.

## 2. Why the *global* decentralized assign failed (do not repeat)

The existing `DECENTASN=1` path (`occ_kernel_dsws_flow.s` `.Lflow_feed_empty`, ~3328) is a correct **lock-free
slot-reserve** (any wave CAS-reserves a slot on `ASSIGN_HEAD`), BUT line 3359 claims work as
`gi = occ[20]++` — a **global SUPER-TILE** index (`t = gi>>shift`, `ksi = gi & mask`). So each wave grabs a
super-tile from a *different tile*, and a WG's `POOL_N` slots fill with slices of unrelated tiles. The banked
reduce (`TILEDONE[group]`, ACC banks) is **per-WG LDS**, so a tile whose slices scattered across WGs never
accumulates in any one WG → C never stored (**silicon-refuted 2026-07-16, `391c7530`: `bad=9216`**).
**Decentralized *global* claim ⊥ per-WG banked reduce.**

## 3. The fix (what changes, what stays)

**STAYS (reuse, do not rebuild):**
- The **lock-free slot-reserve** (CAS on `ASSIGN_HEAD`, `.Lflow_da_resv`/`.Lflow_da_won`) — the decentralized
  core. Many waves reserve *different* slots in parallel; refill rate scales with free waves.
- **Poison-until-staged** claim + **`SL_GEN`** release-fence generation gates (a reserved-but-unstamped slot
  can't be consumed) — the reuse-during-compute safety.
- **`TILEDONE` completer** (= lazy carry-off / "deep-J delivers to DRAM"), **banking**, **baton**, **stagger**,
  **dyn-VGPR**, **GROUPS>1** (fixed 2026-07-17), **lazy role accounting**.

**CHANGES (the delta):**
1. **Claim WG-local whole tiles, not global super-tiles.** Replace the per-assign `gi = occ[20]++`
   (global super-tile) with:
   - a **per-WG current-tile** register/LDS word (`COORD_T`-style) + a **per-WG ksi cursor**;
   - each decentralized assign emits `gi = (t << shift) | ksi_cursor` and advances the WG-local ksi cursor;
   - when the WG-local cursor **exhausts** the tile (`ksi == n_kseg`, GROUPS: after all groups), the wave that
     exhausts it does **one** `occ[20]++` to claim the WG's **next whole TILE** (t), resets the cursor, and (bank
     reuse) runs `zero_banks`. This is the coordinator's tile-claim + ksi-walk, done **decentralized** (any wave).
2. **Make the WG-local cursor + tile-claim + `zero_banks` reachable by any wave**, not coordinator-only. Today
   `zero_banks` and the ksi cursor live on the wid0 path (`.Lflow_assign_top`, ~2538). They must move to /be
   callable from the decentralized assign, guarded so exactly one wave claims each new tile and zeroes its banks
   (CAS on the tile-claim; drain-gated `zero_banks` exactly like the coordinator's new-tile barrier).
3. **Port the GROUPS>1 cursor logic** I built on the coordinator path (COUNT-based `(group,ksi)` advance,
   group-boundary drain+`zero_banks`, group-aware lead/non-lead poison) onto the decentralized cursor. These
   currently live in `.Lflow_same_tile` / the wid0 advance — the decentralized path needs the same behavior.

## 4. The tier ladder (assign becomes a role)

Each wave, each loop iteration, does the **first available** productive action (river: never blocks). Proposed
order — **the one design choice I want kmbandy's eye on:**

| # | tier | action | current |
|---|---|---|---|
| 1 | **deliver** | a tile fully reduced → carry it to DRAM (TILEDONE completer) | exists, decentralized |
| 2 | **grow/compute** | a K-slice staged AND I can grow (budget) → grow + WMMA; baton-poked → grow now | exists (baton) |
| 3 | **assign** | frontier hungry (pool has room) → reserve a slot, emit the next super-tile | **wid0-only → decentralize** |
| 4 | **feed/stage** | operands not staged for a pending slot → stage A/B | exists |
| 5 | **coast** | nothing available → brief `s_sleep`, retry | exists |

**Open scheduling choice (options, not a silent pick):**
- **(O-A) deliver-first** (above): free banks/slots ASAP so producers/computers don't stall on a full pool.
- **(O-B) compute-first**: prioritize keeping the WMMA units hot; deliver only when no compute is possible.
- **(O-C) assign-ahead**: prioritize assign whenever the frontier has room, so computers never wait on work.
Recommendation: **O-A** (deliver-first) — a full pool is the thing that most directly stalls the river (a
delivered tile frees `POOL_N` room + banks). But this interacts with the baton's "≥1 wave at peak" goal;
resolve empirically after the bring-up (fed A/B of O-A vs O-C).

## 5. The two real sub-problems to resolve in the plan (flagged, not hidden)

- **S1 — JDEPTH>1 poison collision (guard line 778).** `SL_RBNEXT` currently carries EITHER the poison-until-
  staged bits (`RB_PENDING=0xC0000000`) OR the low rowblk counter `0..ACC_N`. The deep-J non-lead poison writes
  `SL_RBNEXT=ACC_N`, which collides with the staged-poison encoding → the guard forbids `DECENTASN && JDEPTH>1`.
  **Options:** (a) widen the encoding so `SL_RBNEXT` carries both the J-lead/non-lead bit AND the pending bit
  (there are free high bits between `ACC_N<0x100` and `RB_PENDING=0xC…`); (b) move the J-poison to a separate
  per-slot word. **The plan must pick one and prove exactly-once arm + no false-claimable.** This is the gate
  that lets deep-J (register-hold) compose with decentralized assign; note **lazy-carry (delivery) already
  composes** (TILEDONE, unaffected).
- **S2 — tile-claim / bank-zero races under decentralization.** Exactly one wave must claim each new WG tile and
  zero its banks, drain-gated (banks can't zero while a prior tile's slice is draining — the same-WG-combine
  barrier). Reuse the coordinator's `DRAIN>=ASSIGN` new-tile barrier + a CAS on the tile-claim so a second wave
  that races the exhaust loses cleanly and just retries the tier ladder. Enumerate every `next++`↔`RBDONE++`
  pairing (the O1 discipline from `c7d9407e`) for the decentralized J-carrier.

## 6. Constraints (non-negotiable — from `DSWS_STATE.md`)

- **G1 same-WG-combine:** a WG must own a whole tile. (This design's whole point.)
- **No blocking reads / caps / waits / hard partitions** in the hot loop (the river). A wave reads only its own
  mailbox; assign is bounded-CAS-then-bail (`.Lflow_da_resv` retry budget = 4, already correct).
- **MSDRAIN=1** mandatory (POOL_N>1 out-of-order drain walk).
- **Fed-only verdicts**, **full stride=1 oracle** for any correctness claim (sampled oracle gave false-CLEAN
  twice on 2026-07-17), **`computed == G*MTLsuper*NTL*n_kseg`** every run.
- **Off-path inertness:** `DECENTASN=0` must stay byte-identical (`22bc8d0d` at the baton geom); all new code
  `.if DECENTASN`-gated.

## 7. Test gates

- **Offline:** assembles 0-spill; `DECENTASN=0` byte-identical; hot-loop grep shows NO new blocking-read/cap/
  wait; the S1/S2 enumerations written out; LDS algebra re-checked (banked banks + POOL_N ≤ 64KB).
- **Silicon (each a single greenlit `./gpu_run.sh`):**
  1. **Gate A — decentralized assign, J=1, GROUPS=1**, real ml8 shape, **full stride=1 oracle**: `bad=0`,
     work-exact, `occ[0]=0`, no DMFAT/reset. Proves the WG-local decentralized claim is correct (the thing the
     global version failed).
  2. **Gate B — + GROUPS>1** (full oracle): the ported group cursor is correct decentralized.
  3. **Gate C — + deep-J (JDEPTH>1)** once S1 is solved (full oracle).
  4. **Gate D — compose stagger + baton**, fed to steady state: does the assign-bound lift (`occ[86]` down from
     ~92%)? does the budget bind (`grow-fail>0`)? does the baton engage (`occ[98]>0`)? — the payoff measurement.

## 8. Success criterion

The full river runs on a real ml8 shape, fed: all tiers decentralized (assign now included), oracle-clean +
work-exact, and the assign-bound is measurably relieved vs the wid0 baseline — establishing whether budget then
binds and the baton engages (the question the whole architecture exists to answer).

## 9. Open questions (resolve in plan or during the gate)

- **Q1:** tier priority order (§4 O-A/O-B/O-C) — decide by A/B after Gate A, or commit to O-A up front?
- **Q2 (S1):** widen `SL_RBNEXT` encoding vs separate J-poison word — which is cleaner + provably exactly-once?
- **Q3:** does the WG-local ksi cursor live in LDS (shared, CAS-advanced by any wave) or is there a cheaper
  per-WG scheme? (LDS CAS is the obvious first cut; measure contention.)
- **Q4:** non-pow2 `n_kseg` — the current DECENTASN assign fails SAFE on non-pow2 (3337). GROUPS>1 needs the
  COUNT-walk (already built) for arbitrary-K; confirm the decentralized cursor uses COUNT, not mask.
