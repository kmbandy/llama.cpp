# DSWS v2 — Fix #1a: D=2 double-buffered ring-of-slots pipeline

Design spec. Kills the two measured waits (FOLLOW_WAIT 15–22%, STAGE_WAIT 26–49%) by pipelining
super-tiles instead of lockstepping one at a time. This is increment **1a** of Fix #1 (see
`PHASE_PROFILE.md` → "the 3 core fixes"). The paddle (1c) and #2/#3 come after.

## Goal & success criteria
- **Goal:** compute waves never block on a central "next super-tile" publish, and rarely block on
  feeds — because the next super-tile's operands are already staged in the other slot.
- **Success:** on `576×512×2048` (the brick-safe regime), oracle CLEAN, no brick, and the phase
  profiler shows **FOLLOW_WAIT + STAGE_WAIT materially down** vs the current lockstep baseline
  (4c2a2b: 16.6% + 35.5% = 52% today). WMMA/FLUSH share rises correspondingly (same absolute work).

## Non-goals (explicitly deferred — NOT silently skipped)
- **Grow-stagger gate.** Multiple compute waves growing to 112 VGPR at once is the ISA §3.3.3.2
  deadlock (the M=1920 brick). The ring does **not** fix that — it's independent. 1a keeps the
  current per-rowblk grow (proven safe at M=576) and stays at M=576. Grow-stagger is required before
  training-M and is its own increment.
- **The paddle / role rebalance (1c), wave-count-from-budget (#2), feed:compute ratio (#3).** Later.
- **Split-K C-write amplification.** The long pole we address *after* the 3 fixes.

## Approach: a NEW file, safe bin untouched
Build `occ_kernel_dsws_ring.s` (fork of `occ_kernel_dsws.s`). The current kernel and its proven
`c62568f6` bin are never edited. The ring bin is built to `occ_dsws2_<mix>_gd.bin` for a run, then the
safe bin is restored (same discipline as every probe). WMMA / DECODE_STI / flush / oracle logic is
copied verbatim; only the LDS map + the claimer/feed/compute control loops change.

## LDS map (64 KB, D=2) — decision A (raise cap 32→64 KB, keep SEGK=64)
Two slots, each with its own control words **and** its own operand buffers. Operands dominate:
```
SLOT_STRIDE = 16 KB operands + control words, 256B-aligned
  per slot s:
    STI[s]      super-tile id in this slot (0xFFFFFFFF = sentinel/retire)
    GEN[s]      monotonic refill generation (consumers detect new occupant by GEN change)
    RB_NEXT[s]  rowblk claim counter     RB_DONE[s]  rowblks computed+flushed
    BF_NEXT[s]  B-frag claim counter     BF_DONE[s]  B-frags staged
    AR_NEXT[s]  A-rowblk claim counter   AR_DONE[s]  A-rowblks staged
    BRES[s]     resident B operands (FN*16*SEGK = 4 KB)
    ARES[s]     resident A operands (G*16*FM*SEGK = 12 KB)
shared (single copy): FILL_IDX, DRAIN_IDX, INITFLAG, NCOMP/NAFEED/NBFEED slot census
total ≈ 2 × 16.4 KB + shared ≈ 33 KB  (< 64 KB hardware limit; kernel cap raised to 65536)
```
Host: `run_dsws2` allocates/declares 64 KB group segment for the ring bin (currently 16896B). One
constant to bump; guard stays `LDS ≤ 65536`.

## Frontiers & steady state
Two moving indices into the D=2 ring:
- `FILL_IDX` — the slot the dispatcher just published and feeds are staging.
- `DRAIN_IDX` — the slot compute is consuming.

Steady state overlaps **fill(other) with drain(current)**:
1. Compute drains `DRAIN_IDX` (claims rowblks, WMMA, flush).
2. Concurrently, dispatcher has published the next `gsti` into the other slot (`FILL_IDX`), and feeds
   stage its A/B operands there.
3. When compute finishes `DRAIN_IDX` (`RB_DONE==G`) **and** `FILL_IDX` is fully staged
   (`BF_DONE==FN && AR_DONE==G`): swap — compute moves to `FILL_IDX`, the drained slot is recycled.

STAGE_WAIT is hidden whenever fill-time ≤ drain-time (feeds had all of drain(current) to prepare).
FOLLOW_WAIT collapses to just the swap check — no central publish to wait on.

### Init & fill-sequencing (keep exactly one slot ahead)
- **Init:** `DRAIN_IDX=0`. The dispatcher fills slot 0, then slot 1, before compute drains — so once
  compute starts on slot 0, slot 1 is already published (and staging). One look-ahead slot always.
- **One slot fills at a time.** All feed waves target the single `FILL_IDX`. The dispatcher advances
  `FILL_IDX` to the next free slot **only after** the current fill slot is fully staged
  (`BF_DONE==FN && AR_DONE==G` → READY). So feeds never split across slots; the "which slot do I
  stage" question is just "read `FILL_IDX`."
- **Dispatcher invariant:** keep the non-`DRAIN_IDX` slot filled-or-filling. Concretely: whenever a
  slot becomes FREE (its prior occupant's `RB_DONE==G`), the dispatcher claims the next `gsti` into it
  and sets `FILL_IDX` to it. With D=2 this strictly alternates. The dispatcher's own waits (for a slot
  to free, for staging to finish) are **off the compute critical path** — compute is draining the
  other slot throughout.

## Protocol refinements (derived while reading the substrate — 2026-07-04)
Two decisions that refine the pseudocode below, both discovered by mapping the spec onto the real
single-slot loops. They *simplify* correctness; record them so the barrier-free argument is reviewable.

1. **`GEN` is a GLOBAL-monotonic publish counter, not per-slot.** The dispatcher keeps one running
   counter (the old `EPOCH` role), `++` per publish, and writes it to the published slot's `SL_GEN`
   **last** (release fence). Every follower tracks a single last-seen gen (`s35`, exactly as the old
   `_follow` tracked epoch) and gates "new occupant in this slot" on `SL_GEN[idx] > s35`. A slow feed
   that skips a whole generation (reads a *newer* FILL_IDX/gen than the one it last staged) is
   **safe**: the slot it skipped was already fully staged by the other feed(s) before the dispatcher
   advanced FILL_IDX off it (the dispatcher's `wait READY` gate guarantees this), and compute gates on
   the DONE *counters*, not on which feed contributed. Old `SL_GEN` of any slot is always `< s35` for a
   caught-up wave, so a stale read just spins — never double-stages.
2. **The per-super-tile QUIESCE over-claim handshake is DROPPED.** In the single-slot kernel the
   claimer waited `*_NEXT >= depth + #role-waves` before resetting, so a descheduled straggler's next
   `fetch_add` couldn't land on index 0 of the *next* super-tile against stale state. The ring makes
   that race **structurally impossible**: a slot's counters reset only at FREE→FILLED, only by the
   dispatcher, only after `SL_RBDONE >= G`; and compute cannot bump `SL_RBDONE` until
   `SL_BFDONE==FN && SL_ARDONE==G`. So every in-flight feed claim on a slot is accounted (its DONE
   increment) *before* that slot can be recycled — no straggler is ever mid-claim at reset. The DONE
   counters alone gate recycling. (Feeds still over-claim `*_NEXT >= depth` to know when to STOP
   claiming a slot; that is within-occupant and reset-safe.) Net: no `QUIESCE_CNT`, no
   `*_NEXT >= depth+waves` waits, and the `NCOMP/NAFEED/NBFEED` LDS census is unused at CONV=0.

Consequence for wave roles: `wid0` is the pure **dispatcher** (no BSTAGE — deviates from the old
claimer, which was B-feed-class). Effective B-stager count is `NBFEED-1`; pick the launch mix
accordingly (the exact feed:compute ratio is fix #3, not 1a). GEN starts at 1 for the first real
publish (`SL_GEN==0` marks a never-published slot → the dispatcher's initial fill).

## Counter protocol (the barrier-free correctness — this is what bricks if wrong)
Each **slot owns its counters**; concurrent super-tiles never share one → the reset-race that forced
the original WG-wide quiesce cannot occur. Invariants:

**Dispatcher (wid0), per free slot:**
```
when slot s is FREE (RB_DONE[s]==G for its prior occupant, or s never used):
  gsti = atomic_inc(occ[20]); if gsti >= chunkHi: publish STI[s]=SENTINEL, bump GEN[s], mark terminal
  else:
    reset RB_NEXT[s]=BF_NEXT[s]=AR_NEXT[s]=0 ; RB_DONE[s]=BF_DONE[s]=AR_DONE[s]=0
    STI[s] = gsti
    GEN[s] += 1                 // publish LAST: the GEN bump is the release fence for feeds+compute
    FILL_IDX = s
```
Reset happens **only** here, **only** by the dispatcher, **after** the prior occupant's `RB_DONE==G`
is observed. `GEN[s]` bumped last = single release point; feeds/compute gate on GEN, so they never see
half-reset state.

**Feed wave (A or B), staging FILL_IDX:**
```
loop:
  wait GEN[FILL_IDX] advances (new occupant)         // no central epoch; per-slot gen
  sti = STI[FILL_IDX]; if sentinel: retire
  decode; claim frags/rowblks from BF_NEXT/AR_NEXT[FILL_IDX]; stage into BRES/ARES[FILL_IDX]
  on stage of each unit: BF_DONE/AR_DONE[FILL_IDX]++   // compute gates on DONE, not NEXT
  terminal over-claim bounds the claim (unchanged threshold logic, now per-slot)
```

**Compute wave, draining DRAIN_IDX:**
```
loop:
  wait GEN[DRAIN_IDX] advances                        // slot pre-filled -> minimal FOLLOW_WAIT
  sti = STI[DRAIN_IDX]; if sentinel: retire
  decode; wait BF_DONE[DRAIN_IDX]==FN && AR_DONE[DRAIN_IDX]==G   // minimal STAGE_WAIT if pre-staged
  claim rowblks RB_NEXT[DRAIN_IDX]; grow; WMMA(BRES/ARES[DRAIN_IDX]); flush C; shrink; RB_DONE[DRAIN_IDX]++
  the compute wave whose RB_DONE++ makes it hit G advances the frontier:
     if other slot READY (its BF_DONE==FN && AR_DONE==G): DRAIN_IDX = other; else spin until READY
```
Frontier advance is done by exactly one wave (the one that closes the slot), lock-free, detected by
`RB_DONE==G`. The just-drained slot's `RB_DONE==G` is the dispatcher's signal to recycle it.

## Correctness argument
- **No reset-race:** slot counters reset only at FREE→FILLED, only by the dispatcher, only after the
  prior occupant's `RB_DONE==G`. A straggler still finishing occupant N of slot s cannot touch slot
  s's next occupant's counters (they're the same slot but the reset waits for the straggler's DONE) —
  and cannot touch the *other* slot's counters (separate memory). The original cross-super-tile
  collision is structurally impossible.
- **No new deadlock:** the only wait cycles are (a) compute waits GEN (dispatcher bumps it — always
  makes progress while work remains), (b) compute waits FILL ready (feeds always progress), (c)
  dispatcher waits prior `RB_DONE==G` (compute always progresses). No cycle where each waits on the
  other indefinitely while work remains. Termination: sentinel STI propagates through GEN to all
  waves → retire. (The pre-existing multi-grow deadlock is unchanged and out of scope for 1a; M=576.)
- **dyn-VGPR:** unchanged per-rowblk grow/shrink (the profile says it's ~1%, safe at M=576).

## What stays identical (copied verbatim)
DECODE_STI, BSTAGE/ASTAGE inner staging math, the WMMA loop, the C flush (split-K
`global_atomic_add_f32`), SAFEPROBE clamps, TFPROBE + PHASEPROBE instrumentation (ported so the
scoreboard works), the oracle.

## Host changes (`run_dsws2`)
- Group-segment size 16896 → 65536 for the ring bin (env or bin-detected). Keep `LDS ≤ 65536` guard.
- Nothing else: same kernarg contract, same occ[20]/occ[24] claim bounds, same chunk/rep loop, same
  phase-accumulator readout.

## Test plan (scoreboard-driven, brick-safe)
1. Assemble; verify LDS ≤ 64 KB; 0 spill.
2. Run `576×512×2048`, POOL=16, single pass, streamed, `chunkMaxS` short. Gate: occ0==0, fence fired,
   **oracle CLEAN**, dmesg delta 0. Restore safe bin.
3. PHASEPROBE run, same shape → compare FOLLOW_WAIT+STAGE_WAIT vs the current-lockstep baseline row in
   `PHASE_PROFILE.md`. Success = the two waits materially down, oracle clean.
4. Only after clean + faster: consider D=3 (LDS permitting) and the next increments.

## Build status (2026-07-04) — Stages 1–5 assembled clean, awaiting greenlit GPU test
Implemented in `occ_kernel_dsws_ring.s` (fork; safe bins untouched) + host `occ_dispatch.cpp`:
- **Stage 1 (foundation):** RING_D=2 layout (FILL_IDX/DRAIN_IDX/RINGINIT at 0/4/8; per-slot control
  block SLOTC_BASE=32 stride 32 with SL_STI/GEN/RBNEXT/RBDONE/BFNEXT/BFDONE/ARNEXT/ARDONE; operands
  OP_BASE=256 OPSTRIDE=16384, BRES_ROFF=0 ARES_ROFF=BRES_BYTES; LDS_TOTAL_RING=33024<65536). Added
  ungated `lds_fetch_add_r`/`lds_inc_r`/`lds_put_r`. Descriptor group-seg 32768→65536. Foundation
  verified byte-identical to pristine `.text`.
- **Stage 2 (dispatcher, was claimer):** regs s36=gen (global-monotonic), s37=fill_slot (alternates),
  s34=fill_slot ctrl base, s17=sti, s69=chunkHi. Inits both slots + frontier, RINGINIT last; fill-free
  loop; publish order STI→FILL_IDX→SL_GEN(last); wait-READY before alternating; terminal sentinels
  fill_slot only + points FILL_IDX at it.
- **Stage 3 (feeds):** ring `BSTAGE_R`/`ASTAGE_R scb,sob`; loops gate on SL_GEN[FILL_IDX]>s35, read STI
  (sentinel→retire), stage into slot ops. Feed regs: s35=gen, s38=FILL_IDX, s48=scb, s52=sob.
- **Stage 4 (compute):** gate on SL_GEN[DRAIN_IDX]>s35, wait SL_BFDONE==FN && SL_ARDONE==G, claim
  SL_RBNEXT, grow(NFV), WMMA on v9+sob(+ARES_ROFF), flush C, shrink, `lds_fetch_add_r SL_RBDONE`
  (old==G-1 ⇒ closer ⇒ `lds_put DRAIN_IDX = ^1`). Compute regs: s35=gen, s46=DRAIN_IDX, s48=scb,
  s52=sob; PHASEPROBE ported. Assembles PHASEPROBE 0 & 1, **0 spill**, no dangling labels.
- **Stage 5 (host):** `DSWS2_RING=1` env ⇒ ldsBytes `256+2*16384=33024` + bin `occ_dsws2_<mix>_ring_gd.bin`
  (built by `build_ring.sh`). RINGD default 1 leaves the single-slot path byte-identical.

**Next (Stage 5 finish): the greenlit GPU test — NOT yet run.** One dispatch, streamed, at the
brick-safe shape; oracle CLEAN + dmesg-delta-0 gate; then a PHASEPROBE=1 delta vs the 4c2a2b baseline.

## Rollout / safety
- New file; current kernel + `c62568f6` bin untouched. Every ring dispatch is kmbandy-greenlit, one at
  a time, streamed, safe-bin restored after. Never `--gl2c`. Stay at M=576 (no grow-stagger yet).
