# DSWS Review — 2026-07-19 (Fable)

Requested by kmbandy: "review and thoughts, what could we improve." Read-only —
no GPU touched, nothing dispatched, nothing built. Basis: the kernel source,
the host harness, the tooling, tonight's six SELFSERVE run logs in
`~/dsws_gpu_logs`, and the doc corpus. Three deep sub-reviews (kernel
correctness, host/tooling, project history) back the findings; this is the
synthesis, most-actionable first.

---

## TL;DR

The carry-through (SELFSERVE) mechanism is, as-built, **internally sound on
every failure class this project has been bitten by** — I'd trust the code more
than the doc that describes it. But tonight's **15.1 TF is not yet a verified
win**, three of the strongest safety nets in this project live in the human's
head rather than the tooling, and the reviewed design on paper is **not the
design that assembles**. None of this is a fire; all of it is cheap to close.

---

## 1. The 15.1 TF number — promising, not yet a win (verify before trusting)

Tonight's `s1_peak_cheap` (20:16) posted **TF=15.1, 4.9% of the 307 TF fp8
peak** — nearly 2× yesterday's 8.2 TF frontier. Real and worth being excited
about. But by the project's *own* rules it doesn't count yet:

- **The config you pasted (`s1_peak_fixed`) never ran.** No log by that name
  exists. The two runs with numbers are `s1_peak_cheap` (TF=15.1) and
  `s1_peak_funnel` (TF=0.4).
- **Same work, 38× different span.** Both runs did `computed=50331648`
  identically; `funnel` spanned 66.25 s at TF=0.4, `cheap` spanned **1.75 s** at
  TF=15.1. The delta is entirely feed quality (`feed-stages` 230K vs 13.6M). TF
  here is measuring how well-fed the run was, not steady-state compute.
- **1.75 s is under your own 2 s clock-commit floor**, and the oracle was
  **sampled (32/16384 tiles, stride=512) + LOOSE tier** — the exact
  configuration `DSWS_STATE.md` says gave false-CLEAN twice.
- **SELFSERVE's own success fingerprint did NOT fire.** The design's §7 gate is
  "all four move together": door1 100%→<100%, grow-fail 0→large, baton occ98
  0→>0, TF>8.2. In `s1_peak_cheap`: door1 **still 100.0%**, grow-fail **still
  0**, baton **still 0**. Only TF moved. And `entered=16.7M` but
  `settled=2553` — the doc flags `entered==settled` as the clean expectation, so
  carry-through engaged but almost nothing settled through it.

**Read:** something made this run fast, but it wasn't the mechanism SELFSERVE
was built to unlock. The engine is still cold (door1=100%, grow-fail=0); the TF
jump most likely came from the shape/feed change (deep-K 131072, big MTL=256),
not from the budget binding. **Re-take at ≥2 s span with a full stride=1 oracle
before recording 15.1 as the number**, and report the four-tuple, not just TF.
If door1 is still pinned at 100%, SELFSERVE has not yet opened the wall — report
the next coupling rather than the TF.

## 2. The shipped kernel is not the design that was reviewed

`DSWS_SELFSERVE_DESIGN.md` §4–§12 describes an `SS_NEXT` per-group claim word
hooked on the coast→door1 fallthrough. **That code is not in the kernel** — no
`SS_NEXT`, no `.Lflow_selfserve`, and the door1 branch (`:2895`→`:2901`) is
byte-identical to SELFSERVE=0. What actually assembles is a third mechanism the
*code comments* call **"carry-through"** (`:2057, :3713–3984`): hooked on the
DECENTASN assign path, a starved wave computes its own reserved item in-register
instead of publishing it to the ring.

Consequence: the five Codex S2 passes documented in §11.7–§11.9 reviewed the
`SS_NEXT` model, which isn't what runs. The doc's own §11.9 caveat ("one change
not through independent review") badly understates this — the entire
claim/enumeration model differs. **The sound-looking carry-through code has not
had the adversarial review the doc claims it had.**

Good news, from a full manual trace of the as-built path: it is
**internally sound**. Work-exact `TILEDONE` accounting (exactly `n_kseg*ACC_N`
bumps/group, single elected C-store via first-crosser); order-independent
`ds_add_f32` banked reduce under BANKZERO; `GSTORED`-gated bank reuse so no
`zero_banks` races a C-store's read; Rule-5 `s_wait_storecnt 0` before *every*
reachable `s_alloc_vgpr`; grow-fail safely diverts the item to the proven ring
path (exactly-once preserved). The §12 hang cause (redundant `deadman_check`
doubling `s_sendmsg_rtn` in the coast spin) is genuinely gone from the burst.

**Actions:** (a) re-base the design doc onto the as-built carry-through path;
(b) run one Codex pass against carry-through specifically — the untested seam is
the carry-through↔ring interplay under a *saturated* budget (S3/S4's regime),
plus the `.Lflow_ss_noclaim` reroute that §11.9 itself flags as post-review.
Known-and-deferred: the 32-bit A/C offset truncation (>4 GiB, parity with the
ring's `ASTAGE_R`, not a new regression).

## 3. Tooling gaps that will cost a brick or a lost day (highest ROI to fix)

Ranked by "prevents a silent corruption / brick / wasted day":

1. **No check that dispatched geometry matches the built bin.** The `.bin`
   filename encodes **only WAVES** (`build_flow.sh:11`); builds differing in
   G/SEGK/FM/ACC_N/POOL_N but sharing WAVES overwrite the same file. The host's
   "geometry mismatch" gate (`occ_dispatch.cpp:6340`) only *range-checks*
   values — its own comment admits "the bin must still be BUILT with matching
   -defsym." This is exactly the documented "WG silently never launches /
   silent-C corruption" failure, unguarded. **Fix (cheap):** decorate the bin
   filename with the full geometry, or `.set CFG_HASH` into the kernel, have
   wave 0 store it to an `occ[]` slot, host compares before trusting output.
2. **The DEADMAN guard looks in the wrong place.** `gpu_run.sh:40` polices
   `DEADMAN_TICKS` on the *dispatch* env, but it's a *build-time* defsym the
   host never reads. So `DEADMAN_TICKS=1000000000 ./build_flow.sh` — the exact
   thing CLAUDE.md rule 4 calls "= 3 bricks" — sails straight through the
   wrapper. **Fix:** enforce the ceiling in `build_flow.sh`, and/or stamp the
   built value into the geom sidecar so `gpu_run.sh` reads the real one.
3. **No measurement gates.** TF prints for any duration (`occ_dispatch.cpp:2202`)
   — no sub-2s floor. `computed` is never compared to the expected
   `TOTAL_super*ACC_N` even though both are in scope (`:2091`, `:2268`,
   `:6274`); a dropped-work run reads *higher* TF and passes the sampled oracle.
   **Fix:** print `TF=n/a (sub-2s)` below 2 s; assert `computed==expected` and
   flag `TF INVALID` on mismatch. This is precisely what would have caught the
   07-14 deadman-drop and the 07-18 J=64 "31.5 TF while losing 34% of work."
4. **Stale-bin check vs bin-actually-opened can diverge.** `gpu_run.sh:69` stats
   the *newest* `flow_gd.bin`; the host opens the `w${FLOW_WAVES}` one
   (`occ_dispatch.cpp:6392`). Seven `w*` bins exist now — the freshness guard
   can bless a just-built `w16` while the host loads an older `w30`.
5. **`SPIN[sh/st/ta]` (occ[81/82/83]) are dead at FORENSICS=0** — gated `.if
   FORENSICS` (`:2273`), which is pinned off, yet printed unlabeled in the live
   stream (`occ_dispatch.cpp:1957`). A reader sees `SPIN[0 0 0]` and concludes
   no wave is parked, when the gauge simply isn't compiled in. (Good news:
   CLAUDE.md's other four named dead counters are now stale — `STINSTR_FEED` is
   wired at `:3549`; `NOCFLUSH/CSTORE/DIAG` are build-flag probes, not counters.
   Worth correcting that list so it doesn't cry wolf.)

## 4. Two prior-review ideas that hit the *exact* current wall, silently dropped

The three 2026-07-10 reviews mostly targeted a Run-6 exit-barrier hang that the
07-13 flush-kill rewrite made moot. But two recommendations went to the feed
wall the project is now attacking with a far more invasive rewrite — and were
never tried:

- **Feed-heavier static seed sweep** (raised independently by *two* reviewers:
  Fable F2 and Grok 2.5 — a `SEED_NAFEED/SEED_NBFEED` defsym to probe staging
  under-provision with no coordinator code). This is the cheapest possible test
  of "feeder-bound vs slot-bound." door1=100% held across all 18 POOL/SEGK cells
  — a near-zero-cost feeder sweep would have cleanly *separated* those two
  hypotheses before committing to SELFSERVE. Even now it's worth one dispatch as
  a control: if more feeders don't move door1, that's direct evidence the limit
  is slot residency (SELFSERVE's premise), not feeder count.
- **Compact group-local A / ARES group-shrink** (GPT5.6 #6 + Grok 3.2, "biggest
  LDS win," 33024→29952 B). This is the *incremental, lower-risk* version of
  SELFSERVE's core insight — hold less in LDS so more waves fit. `ARES_BYTES` is
  still G-sized. Keep it in the back pocket as the fallback if SELFSERVE's VGPR
  self-load proves too costly on L2 bandwidth (the §9 predicted next wall).

## 5. Doc hygiene — `DSWS_STATE.md` is stale and self-contradictory

It's designated the single source of truth and read first each session, so its
staleness is high-blast-radius:

- Still frames **deep-J J=2 GROUPS=3 as "the TARGET"** with "NEXT = SILICON,"
  but the 07-18/19 sweeps **retired deep-J (JDEPTH=1 LOCKED)** and 07-19
  SELFSERVE supersedes that whole bring-up. A reviewer reading only STATE chases
  an abandoned path.
- **Internal contradiction:** the canonical build line (L82) says
  `DECENTASN=1`; the LOAD-BEARING KNOBS table (L224) says `DECENTASN=0`. Same
  doc.
- **Best-number disagreement:** STATE says J=1/SEGK=256/ACC_N=3 = 9.5 TF; the
  frontier sweep says the *same geometry* = 8.2 TF (method-sensitive: 2.16 s
  big-M vs ≥3.2 s deep-fed). Pick one basis and annotate the other.

Refresh STATE to the post-frontier reality, and reconcile these three before the
next session reads it as gospel.

## 6. Process — the pattern worth naming

The project's discipline (safety rules, decision journal, byte-identity anchors,
work-exactness) is genuinely better than most production code. But the failure
mode is remarkably consistent, and it's not sloppy bisection — it's
**trusting a number without checking how it was produced**: a dead counter, a
sub-1 s DVFS-idle run, a stale bin, an uninitialised `FATTOK`, a 44×-perturbing
`PHASEPROBE`, a config copied from a handoff doc (`FLOW_POOL_N=1` unquestioned
for ~2 weeks). Static reasoning — *including two frontier-model adversarial
reviews that both "proved" the DECENTASN pin couldn't over-release* — lost to
silicon repeatedly. The 07-19 hang survived *four* Codex passes because it was
framed as a safety addition, not a hot-path message-traffic change: "reviewers
challenge what you point them at."

The meta-fix is the same as the tooling fixes above: **push the safety nets out
of the human's head and into the harness.** Every one of §3's gaps is a place
where a machine check would have caught what review didn't. That's the single
highest-leverage improvement available here — not any kernel change.

---

## What I'd do next (in order)

1. Re-take the peak run: full stride=1 oracle, ≥2 s committed span, report the
   §7 four-tuple. Confirm whether SELFSERVE actually moved door1/grow-fail or
   just the shape did. *(supervised GPU — your call, your rules)*
2. Add the two cheap harness gates (§3.3): work-exactness assert + sub-2s TF
   suppression. Pure host-side, no GPU, no risk.
3. Decorate the bin filename with geometry (§3.1) and move the DEADMAN ceiling
   into `build_flow.sh` (§3.2).
4. Re-base `DSWS_SELFSERVE_DESIGN.md` onto carry-through; one Codex pass on the
   carry-through↔ring saturated-budget seam.
5. Refresh + de-contradict `DSWS_STATE.md`.
6. Optional control: the feed-heavier seed sweep (§4) to separate slot-bound
   from feeder-bound directly.

Nothing here is blocking and nothing is a fire. The architecture is coherent,
the correctness discipline is real, and carry-through looks sound. The gap is
that the *evidence chain* still leans on human vigilance at exactly the points
where this project has repeatedly been fooled — and those are the cheapest
things on the board to fix.
