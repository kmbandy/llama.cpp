# DSWS Substrate v2 — Claim-Based Work Decomposition + Split-K (design)

**Status:** approved design (2026-06-30). Supersedes the Phase-3 *actuation* plan in
`PLAN_DSWS_CONTROLLER.md` (Tasks 3.2–3.4), which assumed role conversion could be bolted
onto the static-partition coop substrate. It cannot — see "The blocker" below. Phases 1–2
(static 3-role substrate, sensing, role-count slots, gate-CAS ticket, reservation model)
remain valid and are reused.

**Goal (one sentence):** Re-found the DSWS substrate so matrix work is *claimed by whatever
wave currently holds a role* rather than *owned by a wave's compile-time identity*, and fold
in split-K — so the controller can move waves between {compute / A-feed / B-feed} at runtime
without orphaning output, jamming a feed, or bricking.

**Architecture (2-3 sentences):** A workgroup launches a fixed N waves. Work is a pool of
`(mblk, tcol, kseg)` super-tiles; a pinned claimer broadcasts the current super-tile, and the
live waves of each role drain shared atomic counters (compute claims rowblks, feeds claim
which operand fragment to stage) against resident-in-LDS A/B for that super-tile. Because work
is claimed, not owned, `nComp/nAfeed/nBfeed` can change at any per-kseg boundary with no
work-item handoff.

**Tech stack:** Hand-written gfx1201 (RDNA4, wave32) assembly (`occ_kernel_dsws.s`, NEW file),
raw-PM4 dispatch (`occ_dispatch.cpp`), CPU fp8 e4m3 oracle (`fp8_oracle.cpp`), control law
(`dsws_ctrl_model.cpp`, unchanged). dyn-VGPR via `s_alloc_vgpr` (armed by PM4 RSRC2 bit 6).

## Global constraints (verbatim, every task inherits these)

- A GPU brick is a **BUG**, never an accepted tax. A hang = full STOP + report, never
  auto-fire the next variant.
- **The user greenlights EVERY GPU dispatch individually.** Display GPU (R9700 drives the
  monitors) → only sub-second, compositor-safe-chunked dispatches
  (`ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1`, timeout 30).
- **NEVER pass `--gl2c`** (MES-crash landmine). SAFEPROBE + bounds gate + padding stay ON.
- Every run streams to disk (`ML8_COOP_STREAM=1`).
- Commit only when the user asks.
- Single-variable isolation; fix bugs, don't dodge them; never declare a wall from an
  unmeasured assumption; don't blame the GPU/model prematurely.
- Barrier-free / lock-free: pure LDS atomics + busy-wait flags. **No `s_barrier`** (mixed
  dyn-VGPR allocations + `s_barrier` hard-deadlock the GPU — proven).
- The proven `occ_kernel_coop.s` (1716B `DSWS=0` coop binary) is **never modified** — it is
  the known-good reference. v2 is an additive new file.

## The blocker (why v2 exists)

3/3 consensus (kmbandy + Claude + Codex, session 019f19da, 2026-06-30; KG `86e33108`): the
proven coop substrate binds the matrix **work decomposition** to compile-time role counts and
physical wave identity (`wid`). Naive role conversion therefore **bricks** (the hang fires
before any wrong-output is even observable):

1. Compute M-rows: `rowblk = trow*P + cid`, `P = NCOMP` compile-time, `cid` fixed from `wid`.
   A departed `cid`'s rows have **no writer** → orphaned output.
2. A-feed: `NCOMP` bands paired 1:1 to compute `cid` via `prod_a[cid]/cons_a[cid]`. A departed
   consumer → `prod_a` blocks at `RINGD_A` → **WG hangs**. B-feed's `min_cons` also scans the
   `P` compute counters, so a departed `cid` wedges B production too.
3. B-feed frags: `owner = ni % NBFEED`; a converted-in wave with `b_id ≥ NBFEED` computes a
   `prod_b` address that **overflows into A-ring/LDS storage** → memory corruption.

Root cause: work bound to wave **identity** instead of being **claimable by whoever holds the
role**. v2 removes that binding.

## Architecture — the role economy (unchanged from SPEC_DSWS_CONTROLLER.md)

Fixed N waves/WG; each wave is in exactly one role at any instant. The controller governs only
the partition `(nComp, nAfeed, nBfeed)` with `nComp + nAfeed + nBfeed = N`. Wave count never
changes; only the partition moves. State = three atomic LDS role-count slots (already built,
T2.1). Floors: each role `≥ 1`. Sum-envelope: `Σ instantaneous VGPR alloc < per-SIMD budget`,
enforced at grow-time via the `vgpr_reserved` counter (already built, T2.2).

## Section 1 — Work pool & claim model

**Super-tile** = `(mblk, tcol, ksi)`. K-loop terms used throughout: `SEGK` = segment size
(K-elements per split-K segment, a compile-time knob); `n_kseg = KT / SEGK` = number of
segments; `ksi ∈ [0, n_kseg)` = segment index.
- `mblk` indexes a group of `G` consecutive rowblks — `G` = compile-time cooperative M-extent
  (set to the launch's max compute count, `NCOMP_MAX`). Bounds resident A.
- `tcol` = the shared N-column tile (`FN` frags). `ksi` = the split-K segment index.
- Pool size = `(M / (G·16·FM)) × NTL × n_kseg` super-tiles.

**Two claim levels** (reuse the proven "claimer claims + broadcasts, followers wait on epoch"
machinery):
- **Super-tile claim:** a pinned claimer wave (Section 3) does the existing `global_atomic_add`
  to grab the next super-tile index `sti`, decodes `(mblk, tcol, kseg)`, publishes it, then
  bumps epoch (the proven `TI_OFF`-before-`EPOCH_OFF` ordering). All WG waves work the same
  current super-tile.
- **Rowblk claim (within a super-tile):** a per-super-tile LDS counter `rowblk_next`. Live
  compute waves `ds`-atomic-add to grab the next `rowblk ∈ [0, G)`. Exhausted (`≥ G`) → this
  super-tile's compute is done.

**Coverage proof:** the super-tile pool covers every `(mblk, tcol, kseg)` once; `rowblk_next`
covers every rowblk in the group once. So every `(rowblk, tcol, kseg)` is computed exactly
once and `C[rowblk,tcol] = Σ_kseg partial`. Holds for **any** live `nComp ≤ G` — fewer compute
waves just drain `rowblk_next` slower. (If the controller ever set `nComp > G`, the extra waves
find the counter exhausted and idle/convert; floors+ceilings keep `nComp ≤ G`.)

## Section 2 — Resident A/B lifetime & completion handshake

Split-K keeps each segment's operands small enough to stage **resident** in LDS:
- **Resident B** for `(tcol, ksi)` = `FN·16` cols × `SEGK` K. Loaded by B-feed waves (each
  claims which frag to stage from a frag counter — identity-free).
- **Resident A** for `(mblk, ksi)` = `G` rowblks × `16·FM` rows × `SEGK` K. Loaded by A-feed
  waves (each claims which rowblk's A to stage from an A-load counter — identity-free). Compute
  reads its claimed rowblk's A from this resident region (replayable — any compute wave
  re-reads freely; this is what decouples rowblk-count from `nComp`).

**Completion handshake** (barrier-free; the safety core):
- **`rowblk_done`** (per-super-tile LDS counter). A compute wave, after flushing its rowblk's
  partial, `ds`-atomic-increments `rowblk_done`.
- **Claimer gate:** the claimer may claim/broadcast the *next* super-tile only once
  `rowblk_done == G` for the current one (all rowblks computed *and* flushed). This frees the
  resident A/B safely — a counter compare, not `s_barrier`.
- A compute wave that finishes its claimed rowblks before the super-tile is globally complete
  spins on the completion gate (or attempts a conversion) rather than racing ahead — the
  busy-wait discipline the kernel already uses.

**Terminal:** super-tile claim returning `sti ≥ TOTAL_super` is the role-agnostic retire
signal; every role checks it at its decision boundary (replaces the per-tile POOLTERM, now
sub-tile aware). A just-converted wave re-checks immediately.

## Section 3 — Role tags & the pinned claimer (simplification over a published role map)

The claim-counter model makes a *published per-wave role/rank map* unnecessary — every role
claims work from shared counters, so no role needs a stable logical rank; count affects only
speed, never coverage. What remains:
- **Role-count slots** (already built, T2.1) — CAS'd on conversion; drive sensing + floor
  guards.
- **Per-wave private current-role register** — the wave branches to its role loop; on
  conversion it rewrites this register and jumps. Cross-wave visibility is carried by the
  atomic work-counters themselves, not a broadcast map.
- **Pinned claimer + clock = physical `wid 0`** (never converts). Permanently owns super-tile
  claiming, the `ti`/epoch broadcast, and the `SEGCNT` controller clock. This kills the
  clock-stall failure mode — the clock owner can never convert away.

This is a deliberate, documented deviation from the consensus "generationed role map" step: the
claim model dissolved the need, removing a class of cross-wave ordering hazards.

## Section 4 — Partial-C reduction & tiered oracle

Split-K's headroom requires a wave to do **one kseg then release** (brief VGPR peak), so
different waves compute different ksegs of the same `(rowblk,tcol)` → their partials combine
**across waves** → the low bits of `C` are no longer bit-deterministic (fp combine order). The
combine uses `global_atomic_add` of fp32 partials into `C`.

**Tiered oracle** — tight tolerance where the risk is, looser only where split-K fp
reassociation forces it. (NB: the CPU `wmma_ref` chain is not bit-identical to the GPU fp8→fp32
WMMA even today, so the established gate is already a *tight tolerance*, `5e-3` rel + `1e-2` abs,
not bit-exact — that is the discipline that caught 136/552.)
- **Tier 1 — tight.** Run the correctness gate at `n_kseg = 1`: one partial per `(rowblk,tcol)`,
  no cross-wave combine. Reuse the **existing tight tolerance** (`5e-3` rel / `1e-2` abs). This
  exercises all the dangerous new logic (claim, coverage, resident A/B, handshake, conversion —
  identical code regardless of kseg count).
- **Tier 2 — loose.** Run `n_kseg > 1` against the reference with a looser tolerance
  (`~3e-2` rel / `2e-2` abs). A structural combine bug (missed/double kseg) is a *large* error →
  caught; only the benign split-K reassociation slips under the looser bound.

Build-time check: confirm `global_atomic_add_f32` is encodable on gfx1201; if not, the combine
uses a CAS loop or a scratch-slot + final-reduction path (the scratch path also recovers exact
determinism at `n_kseg > 1` if ever needed).

## Section 5 — Conversion actuation

Reuses the already-built gate-CAS ticket (T3.1), reservation model (T2.2), and sensors (T2.3).
At a per-kseg boundary (frequent, sub-tile) a wave eligible to convert:
1. **Sense** ring/counter occupancy → watermark decision (`watermark_decision`).
2. **Win the epoch ticket** `try_gate(dir)` — single winner per `(dir, epoch)`.
3. **Floor guard** — CAS-dec the source role-count only if `> 1`.
4. **Reservation envelope** — `compute→feed` shrink: `atomic_sub vgpr_reserved` (always
   succeeds). `feed→compute` grow: `atomic_add` then validate `≤ BUDGET`, else `atomic_sub` +
   abort (stay in current role this epoch).
5. **Actuate** — CAS role slots (dec source, inc dest), flip own private role register,
   `s_alloc_vgpr` GROW(`NFV`)/SHRINK(32) (each guarded by SCC-retry), jump to the dest role's
   loop.

**Payoff:** a converted wave inherits **no work item** — it changes its tag and starts claiming
from the dest role's counters like any other wave of that role. No orphaned rows, no rank
handoff, no jammed feed. Every blocker failure mode is structurally absent. Barrier-free
throughout.

## Section 6 — File structure, build sequencing & gates

**New file `occ_kernel_dsws.s`.** Diverges enough (resident A/B, split-K partial-combine,
claim-counters) that it is a clean new kernel; `occ_kernel_coop.s` stays pristine as the
known-good reference. Trade-off: no literal "byte-identical to 1716B" guard inside the new
file, bought with the proven kernel never being touched (smallest blast radius).

**Build sequence (isolation within the folded scope — a brick/oracle break stays bisectable):**
1. **Static split-K + claim-counter substrate, fixed roles, no conversion.** Oracle-green at
   `n_kseg = 1` (exact) *and* `n_kseg > 1` (tolerance). Proves claim-coverage + resident A/B +
   partial-combine + completion handshake with static roles. RGA 0-spill. **[SUPERVISED GPU —
   the big gate.]**
2. **Add conversion** (Section 5). Oracle-green static-mix, then dynamic-mix (conversions
   firing), then the storm (tight watermarks, `EPOCH_SHIFT = 0`, ×10 repeats). **[SUPERVISED
   GPU.]**
3. **Adaptivity proof + tuning** (Phase 4 carries over: converge-from-wrong-start, `{LOW, HIGH,
   RINGD, EPOCH_SHIFT, G, SEGK}` sweep, `--att` issue-mix). **[SUPERVISED.]**

Every GPU dispatch: compositor-safe chunked, one at a time, user greenlights each, brick = full
STOP + bisect.

## Testing

- **CPU oracle gate** (`fp8_oracle.cpp`): Tier-1 exact bit-match at `n_kseg = 1`; Tier-2 tight
  tolerance at `n_kseg > 1`. Gate before every perf run and on every kernel change.
- **RGA static gate:** 0-spill, live-VGPR within budget, every assemble.
- **Control-law unit tests** (`test_dsws_ctrl_model.cpp`): unchanged, still `ALL PASS`.
- **Storm stress:** tight watermarks + `EPOCH_SHIFT = 0` + ×10 repeats — proves the lock-free
  protocol has no conversion race (the strong-oracle-plus-repeats discipline that caught
  136/552 before).

## Risks & open items

- **LDS budget for resident A+B.** Resident A (`G·16·FM` rows × `SEGK`) + resident B (`FN·16`
  cols × `SEGK`) + role/claim state + ring-free counters must fit the 32 KB group segment.
  `G`, `SEGK`, `FM`, `FN` are the knobs; pick a first config that fits with margin and verify
  in step 1.
- **`global_atomic_add_f32` on gfx1201** — verify encodable; fallback CAS loop / scratch path.
- **Partial-combine traffic** — split-K adds `n_kseg` atomic-adds per `(rowblk,tcol)`. The bet
  (per the DSWS thesis) is that the issue-port offload + fungibility gain exceeds this overhead;
  measured, not assumed, in Phase 4.
- **`G` vs `nComp` overshoot** — floors/ceilings must keep `nComp ≤ G`; assert in the
  controller.
- **Claimer single point** — `wid 0` pinned as claimer means it is always a B-feed-class wave;
  confirm the role economy still balances with one permanently-non-compute wave (it is one wave
  of N; negligible, but noted).
