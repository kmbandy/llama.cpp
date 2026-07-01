# DSWS Phase B — Universal Dynamic Dispatch + Grow-into-Budget Pool Economy

**Status:** design (brainstormed 2026-07-01, kmbandy). Extends and builds on the
committed Phase-B substrate (Tasks 1–5, branch `feat/dsws-phaseb-conversion`,
HEAD `6f3e36f45`).

**Goal:** Make a wave that commits a role conversion actually *execute the new
role's code*, and generalize the fixed 8-wave partition into an adaptive
wave-role economy on a lean-start pool — realizing the DSWS vision where the
kernel re-balances its {compute, A-feed, B-feed} mix into the VGPR budget at
runtime, per shape, per moment.

**Non-goal:** parking/dormant waves (explicitly dropped as YAGNI — see §7).

---

## 0. Context: what already exists (Tasks 1–5)

The committed substrate provides, gated under `.if DSWS2_CONV` (byte-identical
to Phase A when off):

- **Sensing** (`occ_sample`) — per-role occupancy proxy in `s55/s56`.
- **Ticket** (`try_gate \dir,\swin`) — single-winner `(dir,epoch)` LDS-CAS.
- **Reservation** (`reserve_try \delta,\won`) — signed-delta VGPR envelope vs
  `BUDGET`; shrink (−80) always wins, grow (+80) aborts over budget.
- **Bail-time commit** (`conv_apply`) — floor-guarded role-slot swap + VGPR
  resize (`s_alloc_vgpr`) + `s59` role-register flip, ordered strictly *before*
  the `QUIESCE_CNT` bump.
- **Snapshot/quiesce** — claimer snapshots the live role mix into the next
  epoch's parity half of `SNAP_BASE` at broadcast; per-role claim-count
  sentinels are sized from that snapshot; a `QUIESCE_CNT ≥ WAVES−1` cross-check
  is the safety net; a DIAG `occ[29]` flag asserts the two agree.

**The gap this spec closes:** `conv_apply` flips `s59` but nothing reads it — a
converted wave changes its role-slot count and VGPR footprint yet keeps running
its *original* role's code path. Conversion is currently accounting-only.

**Key constants (from `occ_kernel_dsws.s`):** `NFV=112` (fat compute VGPR),
`VLEAN=32` (lean feed VGPR), conversion delta `NFV−VLEAN=80`, role slot ids
`NCOMP_SLOT=24 / NAFEED_SLOT=28 / NBFEED_SLOT=32`, `SNAP_BASE=72` (u32[6],
parity-doubled {nC,nA,nB}), `QUIESCE_CNT_OFF=96`.

---

## 1. Architecture

Under `DSWS2_CONV=1` the workgroup becomes a **lean-start pool of `N_POOL`
waves** (empirically-tuned defsym, 12–16 range), all launched at `VLEAN=32`.
wid-0 is the claimer. A seed subset grows to fat compute at init; the remainder
are lean A-feed / B-feed. Role lives entirely in the scalar `s59`, and a single
dispatcher routes every wave — at entry and after every terminal bail — to the
code for the role `s59` names. Waves rebalance by growing into / shedding VGPR
budget (bounded by `reserve_try`/`BUDGET`), **never** by forking, merging, or
parking.

**Wave-count invariant (the hardware truth):** the launched wave count is fixed
for the kernel's life; `s_alloc_vgpr` resizes *one existing wave*, it does not
spawn or retire waves. The {compute, A, B} multiplier from the vision
(shed 1 fat compute → several lean feed light up) is realized in the **VGPR
budget**, not the wave count: shrinking a compute wave frees `80` VGPR of budget
that other waves can grow into or that lets a feed-heavy mix stay within budget.
Because feed waves are cheap (`32` VGPR), the pool is launched large enough that
a feed-heavy mix has many active feed waves.

**Byte-identity invariant:** all Model-B launch/dispatch lives under
`.if DSWS2_CONV`. With `DSWS2_CONV=0` the kernel assembles byte-identical to the
proven Phase-A fixed-partition kernel. Task 6's re-baseline runs `DSWS2_CONV=0`
as the untouched golden reference.

---

## 2. The dispatcher (universal dynamic dispatch)

**Today:** role is a wid-partition branch taken once at entry (`occ_kernel_dsws.s`
~L604–629); each role's `_follow` loop ends with an unconditional
`s_branch .L<samerole>_follow` (L907/L951/compute equivalent).

**New:** replace those three tail branches with one shared, scalar-only,
wave-uniform trampoline:

```asm
.Ldispatch:                         // s59 = current role slot id (24/28/32)
    s_cmp_eq_u32 s59, NCOMP_SLOT
    s_cbranch_scc1 .Lcompute_follow
    s_cmp_eq_u32 s59, NAFEED_SLOT
    s_cbranch_scc1 .Lafeed_follow
    s_branch .Lbfeed_follow
```

Three properties make this safe and cheap:

1. **Lands on `_follow`, skipping `_alloc`/`_init`.** `conv_apply` already set
   the wave's VGPR footprint to the target role's size, and `_init` (INITFLAG
   wait) ran once at launch. Re-running `_alloc` would wrongly resize the wave;
   re-running `_init` would deadlock (INITFLAG already consumed).
2. **Preserves the last-seen-epoch register** (`s35`). A re-dispatched wave
   therefore waits for the *next* epoch in its new role rather than
   re-processing the epoch it just converted in.
3. **Entry and re-dispatch are the same mechanism.** The entry wid-branch is
   repurposed to *seed* `s59` (and grow the seed-compute waves), then fall into
   `.Ldispatch`. A non-converting wave reads back its own role each epoch and
   loops exactly as today.

The claimer (wid-0) is **excluded**: it branches to `.Lclaimer` at entry (before
the seed/`.Ldispatch` path), runs its own `.Lclaim_loop`, never converts, and is
never counted among the `N_POOL−1` bailers. The dispatcher governs non-claimer
waves only.

Cost: initialize `s59` at each seed role (part of §3), the trampoline (~6
lines), replace 3 tail branches. Entirely additive; the proven `_follow` bodies
are untouched. `s59` read + `s_branch` is scalar-only → **zero** new OOR
exposure (§6).

Rejected alternatives: per-bail inline role check (less DRY, no benefit);
unified per-epoch dispatch loop that collapses the three `_follow` bodies
(rewrites GPU-proven control flow for no functional gain).

---

## 3. Launch & init (lean-start pool)

All `N_POOL` waves `s_alloc_vgpr 32` at entry (already the case for feeds and
claimer; new for compute). wid-0 → claimer (lean, stages B as today).

**Seed assignment by wid** (under `DSWS2_CONV`), replacing the fat-launch
partition:

- wid-0 → claimer.
- first `SEED_NCOMP` non-claimer wids → `s59 = NCOMP_SLOT`, grow to `NFV`.
- next `SEED_NAFEED` wids → `s59 = NAFEED_SLOT` (stay lean).
- remaining wids → `s59 = NBFEED_SLOT` (stay lean).

Then every wave falls into `.Ldispatch`.

**Defsyms (all empirically tuned, principled defaults):**

- `N_POOL` — total launched waves (harness dims must match). Default: the
  feed-heaviest useful count for the target shapes; start 12, sweep. Hard
  constraint: `N_POOL × VLEAN ≤ BUDGET` (every wave always fits lean → no
  parking ever needed).
- `SEED_NCOMP / SEED_NAFEED / SEED_NBFEED` — launch seed mix. Default: the
  shape's static-best guess (e.g. the current `4c2a2b` scaled to the pool).
  `SEED_NCOMP ≥ 1` (compute floor at launch).
- `BUDGET` — the real per-SIMD VGPR ceiling (headroom for growth), **not** the
  zero-headroom launch-footprint default Task 5 inherited. This is the knob that
  makes feed→compute grows able to succeed. Validated by the RGA live-VGPR gate
  and a real dispatch — never a guessed constant.
- `VRESV` seed = `SEED_NCOMP*NFV + (SEED_NAFEED+SEED_NBFEED)*VLEAN` (the live
  reservation the economy adjusts from).

---

## 4. Conversion policy (sense → decide → commit → re-dispatch)

At each terminal bail (`conv_apply` site), a wave runs Task 5's decision, plus
two additions:

- **Cooldown `K`** — a per-wave scalar counts down one per epoch; while `> 0`
  the wave skips the watermark check entirely (no ticket race, no conversion).
  On a committed conversion, reset to `K`. Default `K = 0` (spec-faithful; the
  storm gate sets 0 for maximal thrash). `K > 0` is a churn damper available if
  `s_alloc_vgpr` grow/shrink frequency starts bricking runs — its value is
  reducing the *number* of OOR windows per run. The cooldown counter joins the
  persistent scalar set (outside `s60–s65`, alongside `s57/s58/s59`).
- **Force-convert bring-up hook** — a `DSWS2_FORCE` defsym path where a
  designated wid converts a chosen dir at a chosen epoch, watermarks bypassed;
  everything else static. The deterministic first-proof lever (§7). Gated, off
  by default, emits zero bytes when off.

Direction semantics unchanged from Task 5: `occ_X < CTRL_LOW` → compute starved
for X → shrink one compute → feed-X (delta −80, always wins); `occ_X >
CTRL_HIGH_X` → feed-X over-serving → grow one feed-X → compute (delta +80, may
abort over `BUDGET` → stay). Floor guard keeps every role ≥ 1 wave.

After commit (or no-op), the wave bumps `QUIESCE_CNT` (Task 5 ordering) and
falls into `.Ldispatch`.

---

## 5. Quiesce reconciliation (the correctness crux)

**Invariant that must hold every epoch:** *the claimer's per-epoch snapshot of
the role mix equals the role mix the waves actually execute that epoch.*

Re-dispatch closes Task 5's loop:

1. A converting wave commits its slot swap in epoch *E* at bail time, **before**
   bumping `QUIESCE_CNT` (Task 5 ordering).
2. The claimer waits for quiesce (all `N_POOL−1` non-claimer waves bailed **and**
   the snapshot sentinels), then snapshots the **live** (now-updated) slot counts
   into the *E+1* parity half of `SNAP_BASE`, resets `QUIESCE_CNT`, and bumps the
   epoch last.
3. The converting wave, having flipped `s59`, enters its **new** role's `_follow`
   via `.Ldispatch` and wakes for *E+1*.
4. In *E+1* the wave contributes to the claim counters **as its new role** —
   exactly the mix the claimer snapshotted for *E+1*. Sentinels match observed
   contributions.

The `QUIESCE_CNT ≥ N_POOL−1` cross-check holds unchanged: **every wave bails
exactly once per epoch regardless of role** (no parking), so the count is
role-agnostic and invariant under conversion. The DIAG `occ[29]` flag asserts
snapshot-sentinels ⟺ quiesce-count agree; with re-dispatch wired it reads
*agree* even when conversions fire (whereas Task 5 alone would diverge — the
divergence the flag was built to catch). Task 7's dynamic gate verifies this on
silicon.

---

## 6. OOR-poison window (SPEC §4, #1 brick risk)

- **The dispatcher adds zero OOR exposure:** `.Ldispatch` is `s59` read +
  `s_branch`, scalar-only. A re-dispatched wave enters its new `_follow` with its
  footprint already correctly sized (`conv_apply` closed its grow window before
  the bump) and no pending grow.
- **`conv_apply`'s `s_alloc_vgpr` grow** is unchanged from Task 5's audit: waves
  are lean-32 at every bail, every pre-grow LDS/atomic temp ≤ v15.
- **One new grow site — seed-compute growing at launch init.** Cold path before
  any LDS racing, structurally identical to Phase-A compute's per-rowblk grow and
  equally lean-on-entry. Must carry the same pre-grow comment block and the
  ≤ v15 discipline.

---

## 7. Failure modes, floors, and no-parking

- **Role floors:** `conv_dec_floor` keeps compute ≥ 1 and each feed ≥ 1 — a role
  can never empty.
- **Budget safety:** `N_POOL × VLEAN ≤ BUDGET` guarantees the lean floor always
  fits; `reserve_try` guarantees grows only commit within the real ceiling; the
  atomic-add-then-undo-on-abort path leaves `vgpr_reserved` exactly restored (no
  leaked reservation, Task 5-verified).
- **No parking (YAGNI):** because the pool is sized so all waves stay
  active-lean, a wave never needs to go dormant. Surplus capacity does useful
  lean-feed work instead of sleeping. This is what preserves the simple
  `QUIESCE_CNT ≥ N_POOL−1` accounting and avoids a race-prone wake/sleep
  handshake on the brick-risk path.

---

## 8. Validation ladder

Offline (no GPU) → then supervised GPU gates, each needing an individual
greenlight; any brick/hang/DMESG-fault/DIAG-mismatch = full STOP + bisect, never
auto-advance.

1. **Offline** — assemble all mixes `DSWS2_CONV=1 DIAG=1` → `ASSEMBLE_OK`; RGA
   `SGPR_SPILLS=0 VGPR_SPILLS=0` and live-VGPR within `BUDGET`; `DSWS2_CONV=0`
   sha256 byte-identical to pre-change; CPU control model `ALL PASS`; dry-print
   sanity.
2. **Force-convert gate (SUPERVISED)** — `DSWS2_FORCE` one designated wave, one
   dir, one known epoch, watermarks bypassed. Expect oracle-CLEAN, `bad=0`,
   `occ[0]=0`, DIAG `occ[29]` agree, dmesg silent. The deterministic debut of a
   wave executing a converted footprint's role code.
3. **Dynamic gate (SUPERVISED)** — watermark-driven conversions fire; oracle
   stays CLEAN as roles move; `occ[29]` agree across all mixes × tiers.
4. **Storm gate (SUPERVISED)** — `K=0`, `EPOCH_SHIFT=0`, ×10 repeats — lock-free
   race-hunt under maximal conversion frequency.

---

## 9. Open parameters (resolved during implementation, not guessed)

- `N_POOL`, `SEED_*`, `BUDGET`, `K` — all empirically tuned defsyms with the
  principled defaults in §3–§4; final values set by RGA + real-dispatch
  measurement on the target ml8 shapes, not chosen a priori.
- Persistent-scalar assignment for the cooldown counter (must sit outside
  `s60–s65` and every macro's clobber set; candidates alongside `s57/s58/s59`) —
  fixed at implementation time against the live register map.
