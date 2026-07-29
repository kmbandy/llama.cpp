# DSWS2_OVERLAP Phase 1 (operands L2-only) — 2026-07-24 (Sonnet builder)

**Status: STOP AND REPORT, before any edit.** Read-only investigation only. No `.s`/`.cpp`/`.sh` file
was modified; `CODEX_OVERLAP_P1_PROGRESS.md` (this file) is the only thing written. No GPU dispatch was
run at any point (no `./gpu_run.sh`, `./occ_dispatch`). No `git add`/commit/stash was performed.

## What I was asked to build

`DESIGN_OVERLAP_2WGCU_2026-07-24.md` §8 step 1 + §4: commit A and B operands to L2-only everywhere and
delete the `OP_BASE + POOL_N*OPSTRIDE` LDS operand pool (~40,960B at SEGK=256), gated by a new
`DSWS2_OVERLAP` defsym (default 0, byte-identical), keeping exactly one accumulator generation and
POOL_N=1/SEGK=256 unchanged. The task explicitly required resolving design §10 Q2 first: *"does
operand-L2-only leave the RCONV/feeder economy coherent... confirm it degrades cleanly to 'everyone
self-serves'... If removing the staged-operand path breaks something structural (a path that MUST stage
to LDS for correctness), STOP and report — do not improvise."*

**I found that it does not degrade cleanly. There is a structural dependency.** Details below.

## The wall (source-verified, file:line)

The design's premise (§4): *"On our 30c0a0b mix (0 feed waves) the whole 40,960B operand pool is
vestigial"* conflates two different things — **0 dedicated feed-role waves** (a role-assignment fact)
with **the ring/staged-operand path never executing** (a runtime fact). They are not the same claim, and
tracing the second one at source shows it is false in general, only true *empirically, today, at this one
numeric config*:

1. **Under `SELFSERVE=1` (required by A1), every reservation first attempts self-serve carry-through**
   (`occ_kernel_dsws_flow.s:4587-4599`): grow via `s_alloc_vgpr NFV` (:4593), and on success branch straight
   to `.Lflow_da_ss_decode`/`.Lflow_da_ss_rowblk` (:4647/:4716) — the L2-only body the design wants
   generalized. **The only way a reservation is instead published into the ring is `s_alloc_vgpr` physically
   failing** (`.Lflow_da_ss_growfail`, :4600-4604, reached by `s_cbranch_scc0` on the grow at :4594).
2. On grow-fail, execution falls through (no branch) into the shared "STAMP slot NORMALLY" block
   (:4606-4639), which writes `SL_RBNEXT=RB_PENDING` (poison-until-staged), `SL_STI`, then `SL_GEN=r` LAST
   — i.e. it **publishes the reservation as a genuine ring/pending item**, then branches to `.Lflow_loop`
   (:4644). The grow-failing wave does not retry that item itself.
3. That published item is later **fed** — writing real A/B bytes into the LDS operand pool via
   `ASTAGE_R`/`BSTAGE_R` (:1732/:1822) — from **two** call sites, and critically **neither requires a
   dedicated `ROLE_AFEED`/`ROLE_BFEED` wave**:
   - `.Lflow_feed` (:4045-4108), reached only by non-`ROLE_COMPUTE` waves — genuinely 0 at the 30c0a0b mix.
   - **`.Lflow_coast`'s opportunistic feed** (:5038-5107): a plain `ROLE_COMPUTE` wave that finds
     `DRAIN_HEAD >= STAGE_HEAD` (nothing to compute) falls through from `.Lflow_compute` (:3437-3440)
     straight into this code, checks `STAGE_HEAD < ASSIGN_HEAD`, and if so calls `ASTAGE_R`/`BSTAGE_R`
     itself (:5097/:5106) — **exactly like a feed-role wave, and unconditional: not gated on
     `DSWS2_RCONV`.** This is the mechanism that services grow-fail stragglers even with 30 compute-role
     waves and 0 feed-role waves.
4. The staged item is then **computed** by a *different* wave's `.Lflow_compute` → `.Lflow_havestage` →
   DECENTASN claim (:3552-3592) → `ds_load_b64` from `OP_BASE`-relative `sob` (:3643, :3651-3653 — this is
   the JDEPTH==1 path, i.e. exactly our A1 config; `s48`/`s52` for this path are set at `.Lflow_havestage`'s
   setup before line 3550, not only inside the `.if JDEPTH>1` block at :3614-3642). **This is a real LDS
   read from the operand pool, for our exact JDEPTH=1/DECENTASN=1/SELFSERVE=1 profile — not a JDEPTH>1-only
   artifact.**
5. **Consequence for Phase 1 as specified:** shrinking the LDS layout (`ACC_BASE = OP_BASE`, dropping the
   operand-pool bytes) while leaving steps 3-4's addressing untouched does not merely make an unused region
   optional — it relocates the accumulator-bank region (`ACC_BASE`) **on top of** the address range
   `ASTAGE_R`/`BSTAGE_R` still `ds_store`s into. The first grow-fail that ever fires would have a feeder
   silently **stomp live accumulator banks with staged operand bytes** — corruption of a *different*,
   currently-accumulating tile's partial sums, not merely a wrong-value read. This is worse than "dead code
   safely deleted"; it is a live footgun armed by the very act of shrinking the layout.

## Why "grow-fail is 0 in practice" does not resolve this

Every logged real-shape run through `DSWS_S1_STATUS_2026-07-23.md` (dated the day before this design) and
`DSWS_TESTING_LOG.md` reports **`door4 GROW-FAIL = 0%` in every cell measured** — *"the dyn-VGPR moat NEVER
engages... nowhere near VGPR-bound"* at the current `VBUDGET=1536`/`G=6`/`ACC_N=3` config. So today, on the
shapes tested, the path in steps 2-4 above is empirically never exercised. But:

- `grow-fail=0` is a property of the **current numeric tuning**, not a structural guarantee — `s_alloc_vgpr`
  reflects live SIMD occupancy at that instant, which this design's own later phases *intend to change*:
  §7 of the design doc says outright, about Phase 3's 2 WG/CU, *"32 waves/CU share VBUDGET=1536; grow-fail
  may finally BIND (moat engages — good)... Do NOT pre-judge; measure."* The design anticipates grow-fail
  going nonzero as a **desired outcome of its own later phase**, on the same kernel this phase would leave
  with a silently-corrupting fallback.
- `DSWS_SELFSERVE_DESIGN.md` §7 (the doc that introduced self-serve) predicted grow-fail should go
  *"0 → large (~10^5-scale)"* once self-serve gave the budget something to bind against — i.e. the fallback
  was designed assuming it would fire routinely, and the measured 0% to date is itself a separate,
  not-yet-resolved open question in this codebase (`DSWS_S1_STATUS_2026-07-23.md:54,66`), not a settled
  invariant Phase 1 can safely lean on.
- `DSWS_TESTING_LOG.md:1143`: *"grow-fail == 0 is NOT headroom to spend"* — the project's own standing rule.

Relying on an empirically-zero-but-not-structurally-impossible event to justify deleting its correctness
dependency is exactly the class of risk the task told me to STOP on, not improvise past.

## What would actually be required (why this is a second design pass, not a small reclaim)

Making operand-L2-only safe in general (not just "safe until grow-fail first fires") requires the ring
fallback itself to stop depending on staged LDS operands — i.e. converting **three** call sites (DECENTASN
claim + compute read at :3552-3653, `.Lflow_coast`'s opportunistic feed at :5038-5107, and the dedicated
`.Lflow_feed` at :4045-4108) to self-load A/B from global memory the same way `.Lflow_da_ss_rowblk`
(:4716-4839) already does, and additionally deciding what `STAGE_HEAD`/`RB_PENDING`/`SL_BFDONE`/`SL_ARDONE`
(the "is this slot's operands ready" bookkeeping) should mean once there is no operand data left to stage
readiness *for* — plausibly they collapse to "nothing to check, `SL_GEN==cursor` is sufficient," but that is
a real state-machine simplification across the claim protocol, not a data-layout change. That is a second,
independently-reviewable design pass (in the same spirit as the frontier work the *current* design doc
correctly scoped out of Phase 1 into §6/Phase 2) — not the "self-serve already does this, just delete the
buffer" enabler that §4/§8-step-1 describe.

## What I did NOT do

- No edit to `occ_kernel_dsws_flow.s`, `occ_dispatch.cpp`, or `build_flow.sh`. `DSWS2_OVERLAP`/`OVERLAP`
  scaffolding already exists from a prior (superseded-design) pass (`occ_kernel_dsws_flow.s:445-450`,
  `build_flow.sh` mkflow defsym list) and remains fully inert (no `.if DSWS2_OVERLAP` code exists anywhere),
  so `DSWS2_OVERLAP=0` is byte-identical to baseline by construction — I did not need to re-verify this by
  building, since I made zero changes that could affect it. (The prior pass's own gate already recorded
  `cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553` for `DSWS2_OVERLAP=0` at the A1
  profile in `CODEX_OVERLAP_PROGRESS.md`.)
- No LDS layout change, no host `ldsBytesRaw`/`kOpBase` change. `occ_dispatch.cpp:1907-1908`
  (`kOpBase`/static_assert) and `:1995-2024` (occupancy guard) read and traced for context only.
- No build/RGA run for a Phase-1 "ON" variant, since there is no correct mechanism yet to build.

## Recommendation

Design §4/§8-step-1 needs a second pass that either (a) explicitly designs the ring-fallback's conversion
to self-load-from-L2 (extending the "everyone self-serves" idiom to the grow-fail path too, with its own
claim-protocol-simplification writeup, analogous to how §6 was scoped out for the frontier), or (b) accepts
that Phase 1 must keep the LDS operand pool sized for **1 slot's worth of ring-fallback capacity** (not the
full POOL_N-generality it has today, but not zero either) until grow-fail is either measured nonzero on a
real shape or the ring path is properly converted — in which case the LDS reclaim is smaller than the
~40,960B figure the design assumed, and the "funds the second accumulator generation" argument in §4 needs
re-costing. Either is a kmbandy/claude__main design decision; I should not improvise a fallback-neutering
fix into a concurrency-critical kernel I cannot dispatch to verify.
