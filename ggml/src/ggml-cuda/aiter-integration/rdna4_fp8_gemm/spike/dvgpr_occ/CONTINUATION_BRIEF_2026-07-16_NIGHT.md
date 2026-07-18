# DSWS TRAVELING-PEAK BATON — CONTINUATION BRIEF (2026-07-16 NIGHT)

> ⚠️ **READ THIS IN FULL BEFORE TOUCHING ANYTHING.** On-disk backup of KG brief `78facbaf`
> (project `dsws-traveling-peak-baton`). Read first, in order: `FLOW_ECONOMY_DESIGN.md`,
> `DSWS_TRAVELING_PEAK_BATON_2026-07-16.md`, this brief, `DSWS_TESTING_LOG.md`.

## THE DEFINITIVE BATON DESIGN (kmbandy, verbatim tonight — supersedes ALL earlier baton framing)
> "The ONLY purpose of the baton is to **notify a wave that it's time to grow and grab a rowblk when a preceding
> wave has begun to shrink.** That's it. It doesn't cap, it doesn't limit, it doesn't gate. It just notifies a wave
> to go. Everything else is already handled." — and interpretation **(A)**: "if a wave is still on the compute path
> and isn't notified and hits the budget constraint, it grow-fails and coasts. Not a big deal. With baton, we
> notify a wave **as soon as it's possible to start growing**, with the idea being there should be **>=1 wave at
> peak at all times** to continuously keep compute going instead of in bursts. That's what the combo of stagger and
> baton is meant to do."

So the baton is a **PURE NOTIFICATION (a poke)**, never a gate/cap/seed. The accounting (dyn-VGPR grow/shrink,
grow-fail→coast, feed, claim) is ALREADY the river and handles everything. The baton only fixes **TIMING**: keep a
wave rising into freed budget so compute is continuous, not bursty.

## WHAT IS BUILT NOW — the (A) baton, bin `22bc8d0d`
`occ_kernel_dsws_flow.s`, all `STAGGER`-gated (STAGGER=0 byte-identical to inert md5
`386dc28643ffb58568623ad6d89cfe62` — **SAFETY INVARIANT, HOLDS**):
- **GROW-GATE = PURE RIVER.** No permit gate. A compute wave with a lead segment just grows; physical
  `s_alloc_vgpr` grow-fail (`.Lflow_growfail`) is the ONLY throttle. Concurrent-fat emerges to fill the budget.
  (The `.if STAGGER/.if BATONGATE` block at the grow-decision ~2723 is now just a comment — falls through to grow.)
- **POKE at shrink-START** (~3045, `.if STAGGER` before `.Lflow_bshrink`): round-robin next-available `wid≥3` via
  the proven 1-subtract magic-mod (`NCOMPUTE=WAVES-FIRST_COMPUTE_WID`, `BATON_MAGIC=floor(2^32/NCOMPUTE)`), then
  `lds_put_r NOTIFY[target]=1`. Lane-0 LDS write BEFORE the shrink `s_alloc_vgpr` (ACC dead post-flush, safe).
- **WAKE at the sleep site** `.Lflow_feedmt_sleep` (~3411, the CORRECT injection): before `s_sleep SLEEPN`, read own
  `NOTIFY[wid]`; if poked → clear it + `s_branch .Lflow_loop` (skip the nap, loop back and grow NOW); else yield
  normally. **This is what fills the valley** — a would-be-grower napping on freed VGPR is woken to rise into it.
- **NOTIFY mailbox** = `GROWPERMIT_BASE` @ LDS `0x150` (336), per-wave u32; `BATON_NEXT_OFF` @464 round-robin cursor.
- **KNOBS:** `STAGGER` (0=off/byte-identical, 1=baton on) ; `BATONGATE` (1=baton [default], 0=old FATTOK token layer
  for A/B) ; `GRELAX` (1=relax `WAVES>=2*ACC_N` → `WAVES>=ACC_N+STAGERS`, lets G exceed 15). `build_flow.sh` passes
  all three as optional `${VAR:+...}`. **NO `BATON_SEED`** (deleted — it was a defined-number cap; the peak count is
  emergent).
- **Dam-detector CLEAN:** the wake is forward-branch only (no spin), reads OWN mailbox only (no poll of others).

## THE PAINFUL COURSE-CORRECTIONS TONIGHT (do NOT repeat — each cost real time/trust)
1. **The baton is NOT a gate/cap/seed.** I built a permit-GATE (permit==0 → forced coast) and a `BATON_SEED`
   (tunable count of concurrent peaks). BOTH are dams. kmbandy: *"are you actively trying to sabotage… stop capping
   every tiny thing in the kernel."* RIPPED OUT. The baton only POKES; it never decides whether a wave may grow.
2. **WOFLUSH IS OFF THE TABLE.** It's the killed flush-wait (global atomics straight to C = 97% of runtime, reverted
   2026-07-13). *"No woflush. That's literally something we killed earlier because it introduced a wait."*
3. **Deep-J (banked) is the flush-paydown tool** (*"we built j deep for this. use it"*). Use BANKED (WOFLUSH=0) + deep-J.
4. **SYNTHETIC SQUARE shapes + J=n_kseg PLATEAUS are BANNED.** I wasted a greenlit GPU run on a synthetic 32768³
   J=1024 WOFLUSH=1 STAGGER=0 config (chasing the stale 37-TF number, which memory explicitly flags NOT-a-target).
   Every knob was the opposite of the design. Only REAL ml8 shapes, moderate/deep J, banked, STAGGER=1.
5. **STAGGER=0 only ever as an A/B at the IDENTICAL real config** — never off in its own plateau regime.

## MEASURED RESULTS (all real shape `576×4096×262144` deep-K fed, DECENTASN=0, banked)
| run | config | TF | computed | oracle | door4 grow-fail | jwait |
|---|---|---|---|---|---|---|
| **baton_A_g6_j32 (FINAL, the (A) baton)** | STAGGER=1 J=32 G=6 banked | **21.4** | 9437184 EXACT | CLEAN | 0 | 136,705,818 |
| river (permit-cap removed) J=32 G=6 | STAGGER=1 | 20.7 | EXACT | CLEAN | 0 | 128M |
| STAGGER=0 baseline J=32 G=6 | — | 22 | EXACT | CLEAN | 0 | ~ |
| (earlier permit-gate baton, J=2 G=6, 3×) | STAGGER=1 single-seed | 5.8–6.0 | EXACT | CLEAN | 0 | — |

All clean, work-exact, oracle-clean, **NO brick all session**. Latch clear.

## THE HEADLINE MEASURED FINDING (data-backed, 3 runs; aligns with kmbandy's own flush/MIMO memory)
At G=6 — and **every banked geometry reachable at the real shape** — the VGPR budget does **NOT** bind:
**`door4 grow-fail=0`**. The wall is **STAGE-WAIT**: `occ[88] jwait=136M` (carriers fat, holding ACC, waiting for
the FEED to stage their next segment). NOT budget, NOT compute. So the (A) baton is correct+clean but **has nothing
to bite on here**: it pokes a sleeper "grow now", the sleeper wakes, finds nothing staged, can't compute → TF
parity (21.4 vs river 20.7 / STAGGER=0 22). The baton fills BUDGET-valleys; at reachable G there are none.

## THE REAL OPEN PROBLEM (the crux for the morning)
The baton's value REQUIRES a **BINDING G** (grow-fail>0 → waves nap on freed VGPR → valleys the baton fills). But:
- Binding needs high G (>12 at FM=1: `30*32 + ACC_N*48 > 1536` ⇒ ACC_N>12).
- Banked (WOFLUSH=0) LDS **cannot fit G>12** at the real shape (banks = `ACC_N*FM*FN*1024` scale with ACC_N; even
  G=12 POOL_N=1 SEGK=32 = 57856B is the edge, and G=12 all-fat = exactly 1536 = VBUDGET = marginal binding).
- WOFLUSH frees G but is the killed flush-wait.

⇒ **Reaching a BINDING G at a real shape WITHOUT WOFLUSH is UNSOLVED.** It needs a NEW idea to pay down the flush at
high G (not a knob). This is the gating question for whether the baton can ever prove itself.

## TWO DIRECTIONS FOR THE MORNING (kmbandy's call)
1. **CHASE BINDING-G without WOFLUSH:** find/invent a geometry or flush-reduction that lets G exceed ~12 banked at a
   real shape so grow-fail>0 and budget-valleys form. Only then does the baton (or the pure river) show the 2×
   occupancy the whole architecture exists for.
2. **ATTACK THE MEASURED WALL — STAGING** (jwait=136M): feed width (`ds_read`/`global_load_tr_b64`, the CDNA lever),
   POOL depth, barrier/stage cadence. This is where throughput actually is RIGHT NOW, per this run + the MIMO
   root-cause (WMMA a sliver, rest STAGE_WAIT). The baton doesn't help staging; a feed fix would.

## DEEP RECONCILIATION (kmbandy's own 2026-07-16 KG decision, now data-confirmed)
The plain dyn-VGPR river (grow-fail→coast→retry + shrink-frees-room) ALREADY IS the emergent traveling peak. The
baton adds directed valley-filling but **only matters at a binding G**. Tonight's runs confirm: at non-binding G the
baton == river (parity). The measured wall is FEED. This is NOT "the baton is dead" — it's built, clean, gated,
ready for a binding G — it's "the reachable bottleneck is staging, not budget."

## SAFETY / DISPATCH (non-negotiable — CLAUDE.md + gpu_run.sh)
Dispatch ONLY via `./gpu_run.sh`, ONE per greenlight, changed kernel = ONE bring-up then STOP, hang = full stop,
DEADMAN 0.5s NEVER raised. The **FULL host geometry env MUST match the bin** or the WG silently never launches
(occ0=0) — KG `3c62677a`. For the (A) baton bin:
```
FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=3 DSWS2_SEGK=64 \
DSWS2_K=262144 DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64 ML8_COOP_CHUNK=384 ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=384
```
Rebuild:
```
FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=32 STAGGER=1 BATONGATE=1 \
MSDRAIN=1 RBU=1 STAGINSTR=1 TFPROBE=1 DECENTASN=0 ./build_flow.sh
```
Feed via deep-K.

## CURRENT CODE STATE (all uncommitted)
`occ_kernel_dsws_flow.s`: the (A) baton as above. Grow-gate pure river. `STAGGER`/`BATONGATE`/`GRELAX` knobs.
STAGGER=0 byte-identical `386dc28`. Dead-but-harmless: FATTOK/MAXFAT token layer no-op'd under BATONGATE
(`fat_release` macro is a no-op); `GROWPERMIT`/`BATON_NEXT`/`NCOMPUTE`/`BATON_MAGIC`/`FIRST_COMPUTE_WID` defs live
(used by poke/wake). `build_flow.sh` passes `STAGGER`/`RELSTART`/`BATONGATE`/`GRELAX`. `occ_dispatch.cpp`
`[dsws2 BATON]` occ[98] print is vestigial (reads 0 — the old batonwait spin gauge, dead since there's no spin).
Task #43.
