# DSWS2 Phase 2a — bidirectional role economy + grow-fail-flows-to-feed, 2026-07-24 (Sonnet builder)

**Status: GATES PASS.** OFFLINE ONLY throughout: no `./gpu_run.sh` / `./occ_dispatch` invocation, no GPU
dispatch, no `test_oracle`. No `git add`/commit/stash. Files touched this session: `occ_kernel_dsws_flow.s`
(the mechanism) and `build_flow.sh` (wired the new defsyms through to the assembler — without this the
`DSWS2_ROLEFLOW` env var would have been silently inert, since `mkflow()` only forwards defsyms it
explicitly lists). `occ_dispatch.cpp` was read-only this session (pre-existing P1 dirt, unmodified — see
Gate 3). `occ_kernel_coop.s` was never opened.

## New defsym: `DSWS2_ROLEFLOW` (default 0), `DSWS2_ROLEFLOW_BACK_N` (default 16)

Declared at `occ_kernel_dsws_flow.s:876-899` (right after the existing `DSWS2_OVERLAP` guards). Two
`.error` guards enforce the scope this design was reasoned through: `DSWS2_ROLEFLOW` requires
`DSWS2_RCONV=1` (there is no feed role to revert *from* without RCONV's own compute→feed conversion) and
`DSWS2_OVERLAP=1` (this whole mechanism exists to fix the stall that only occurs once feed is neutered by
the L2-only conversion — at `DSWS2_OVERLAP=0` a real feed wave already clears the grow-fail poison itself
via `side_final`, so P1's original stamp is already self-healing there and `DSWS2_ROLEFLOW` would have
nothing to fix).

## Mechanism 1 — feed→compute reversion (task item 1+2)

**Site:** `.Lflow_dispatch` (`occ_kernel_dsws_flow.s`, was the 3-line `s34==ROLE_COMPUTE ? compute : feed`
fork). Under `DSWS2_ROLEFLOW`, a non-compute-role wave (RETIRE already branched away earlier in
`.Lflow_body`, so only AFEED/BFEED reach here) reads the SAME live-state field `.Lflow_compute`'s own entry
already reads every iteration — `DRAIN_HEAD < STAGE_HEAD` ("a staged/ring-claimable item exists right
now"). This is the direct mirror of RCONV's own compute→feed signal (`.Lflow_compute`'s coast-fallthrough
fires when `DRAIN_HEAD >= STAGE_HEAD`, i.e. *nothing* staged); the reverse condition is exactly "the
condition that makes coasting-as-feed wrong" the task asked for.

**Hysteresis:** a new persistent SGPR `s75` counts consecutive positive observations, symmetric to RCONV's
own `s50` (increments in `.Lflow_coast` on the *negative*-for-compute signal, resets in `.Lflow_havestage`
on the *positive* one) but in reverse and using an *independent* register — reusing `s50` itself would
conflate the two directions' counters. Reaching `DSWS2_ROLEFLOW_BACK_N` (default 16) consecutive
observations writes `ROLE_COMPUTE` to the wave's own mailbox (`lds_put_r ROLE_BASE+wid*4`, byte-for-byte the
same idiom RCONV's own conversion uses) and resets the counter; falling short, or observing the negative
signal even once, resets the counter to 0 (no thrash from a single transient blip). The role change takes
effect on the *next* `.Lflow_body` visit (this pass still runs `.Lflow_feed`'s body) — deliberately mirroring
RCONV's own conversion, which likewise does not act on the new role until the next loop iteration, rather
than inventing a different immediate-effect pattern that would need its own justification.

**s50 re-arm (a bug I found while designing this, not asked for but load-bearing):** `.Lflow_coast`, where
`s50` increments, is reached *only* from `.Lflow_compute`'s own coast-fallthrough. The instant a wave leaves
ROLE_COMPUTE, `s50` is *frozen* at whatever value it held. RCONV's compare is exact-equality
(`s_cmp_eq_u32 s50, DSWS2_RCONV_COAST_N`), so if `s50` were left sitting above `DSWS2_RCONV_COAST_N` after a
feed→compute reversion (entirely possible — nothing bounds how far past the threshold it drifted before
whatever converted it to feed in the first place), it would never equal `DSWS2_RCONV_COAST_N` again short of
a 2^32 wrap: this wave's *next* compute→feed conversion would be silently disabled — an inverted one-way
ratchet, the same species of bug P2a exists to remove, just pointed the other way. Both places that convert
a wave TO `ROLE_COMPUTE` under this defsym (`.Lflow_dispatch`'s reversion, and the grow-fail retry success
below) reset `s50, 0` for exactly this reason.

## Mechanism 2 — grow-fail flows to feed (task item 3)

**Site:** `.Lflow_da_ss_growfail:` (was: bump two diagnostic counters, then fall through unconditionally
into the STAMP block that permanently poisons `SL_RBNEXT=RB_PENDING` under `DSWS2_OVERLAP`).

**Why retain-and-retry, not rollback (the "STOP if not expressible" check):** under `CFASSIGN=1` (required
by this profile), `.Lflow_da_rollback` is compiled out entirely (`.if !CFASSIGN` at its definition) — a
CFASSIGN reservation is a deterministic cohort-math result (`r = cohort_start + wid`), never a CAS that can
be lost, so there is no protocol anywhere in this file for "giving r back." By the time
`.Lflow_da_ss_growfail` is reached, `r` (register `s44`) and its STAMP payload `gi` (register `s52`) are
therefore **already permanently, uniquely, un-rollback-ably committed to this one wave** — this was true
before P2a and is unchanged by it. That leaves exactly two options for resolving `r`: publish something
another wave can claim (forbidden — BLOCKER 1's class), or retain it and finish it myself. Retaining requires
no rollback path at all, which is why it is expressible here: I am not undoing a commitment, I am fulfilling
the one that already exists.

**The mechanism:** on grow-fail, write `ROLE_AFEED` to the wave's own mailbox (telemetry/bookkeeping only —
`.Lflow_da_ss_growfail`'s own control flow never re-reads this mailbox; see the liveness argument below for
why this doesn't need to be undone here), then loop `deadman_check → s_sleep 1 → s_alloc_vgpr NFV → retry on
failure`. `r`/`gi` (s44/s52) are untouched by anything in the retry (`deadman_check`, `cnt_inc`, `flow_gauge`,
`lds_put_r`, `s_sleep`, `s_alloc_vgpr` — audited below, none of them touch s44/s45/s52). Once the grow
succeeds, the code falls through to `duty_grow`/`fat_inc`/`phase_stamp s80`/`s_branch .Lflow_da_ss_decode` —
**the exact same instructions the original first-try-success path already runs** — so the reservation is
decoded and carried through precisely as if grow had never failed. No new claim, no new publish, no
cross-wave signal at any point.

## Argument (a): no wave can be permanently stuck in a role

Before P2a: `ROLE_COMPUTE` is written only at cold-start init; RCONV converts compute→feed but nothing
converts back — a wave that coasts long enough is feed *forever*. After P2a: any AFEED/BFEED wave is
re-examined every `.Lflow_dispatch` visit against live state (`DRAIN_HEAD < STAGE_HEAD`) with a bounded
hysteresis (`DSWS2_ROLEFLOW_BACK_N`, finite, defsym-controlled) — it reverts to compute the first time that
signal holds for `N` consecutive loop iterations, and `s50`'s re-arm (above) guarantees this same wave's
*future* compute→feed conversion is not silently disabled either. The economy flows both ways on every
trip, not just the first. The one wave-state this mechanism does not itself resolve is the grow-fail-retry
park (a wave privately spinning on its own `r`) — but that is bounded by mechanism 2's own retry loop, not
by role at all: it exits either by growing (falls into the compute-labeled path, or gets picked up by
mechanism 1 next time it visits `.Lflow_body`) or, in the truly pathological case, by the kernel's existing
anti-brick deadman (see argument (b)'s caveat).

## Argument (b): grow-fail can no longer permanently stall STAGE_HEAD/DRAIN_HEAD

P1's failure mode: `r`'s slot gets stamped `RB_PENDING` (permanent poison, since `side_final` — its only
clearer — is unreachable under `DSWS2_OVERLAP`), so `STAGE_HEAD`/`DRAIN_HEAD` can never pass `r`, and since
`drain_advance` caps `DRAIN_HEAD` at `STAGE_HEAD`, the whole group/tile boundary (`GSTORED`-gated) stalls
behind it forever, unconditionally, the very first time grow-fail fires.

P2a's failure mode is narrower and bounded, not eliminated by fiat but genuinely smaller: the retry
succeeds as soon as *any* VGPR budget frees up, which is guaranteed to happen in finite time under this
kernel's own existing STAGGER/RIVER economy — every OTHER fat wave's compute burst is itself bounded (grow →
one WMMA burst → shrink, with `RELSTART` returning the budget at shrink-*start* specifically so the next
waiting wave's grow lands in the freed space immediately) — this is the same liveness property the
already-existing per-burst ring-compute grow-fail-and-coast (`.Lflow_growfail`, line ~3548, a *different*,
pre-commitment grow-fail that safely abandons its attempt because no claim was made yet) already silently
depends on for *its* next-iteration retry to eventually succeed. I am not introducing a new liveness
assumption; I am relying on the one this file already runs on. Once the retry succeeds, `r` resolves via the
exact success path — `STAGE_HEAD`/`DRAIN_HEAD` unblock immediately.

**The honest caveat (I will not round this up to airtight):** the retry loop keeps `deadman_check` armed
(the anti-brick floor is non-negotiable and I did not weaken it). If VGPR budget genuinely never frees —
i.e. a real, sustained, system-wide non-progress condition lasting `DEADMAN_TICKS`, not merely "this
particular grow-fail" — the wave force-retires having never stamped `r`, and `r` is left exactly as
unresolved as it would have been under P1 (a total stall). This is not a regression: it requires BOTH
grow-fail firing AND the retry itself outliving the SAME watchdog threshold every other wait-loop in this
file already relies on (`.Lflow_da_drain`'s own drain-wait spin included), rather than firing on the very
first grow-fail unconditionally. I considered adding a bespoke "fall back to the P1 stamp" path inside a
custom deadman variant for this one case, and rejected it: I worked out that the P1 stamp and simply doing
nothing are functionally *equivalent* terminal states here (both leave `SL_GEN` unable to satisfy the
`SL_GEN==head` gate `STAGE_HEAD`/`drain_advance` require — POOL_N=1 means there is only one physical slot,
so an un-stamped slot reads as "not yet produced," which stalls identically to an explicitly poisoned one),
so the extra mechanism would have added new code and a new correctness surface for zero actual liveness
gain. I am naming this rather than hiding it: this is the wall the task said to report rather than paper
over, and I believe it is now as small as it can be made without inventing new cross-wave synchronization —
which is exactly what "STOP rather than invent a claimable publish" asked for.

## Argument (c): nothing cross-wave-claimable is published without a full fence

Mechanism 1 writes only to `ROLE_BASE + wid*4` — the wave's own mailbox, read only by that same wave's own
future `.Lflow_body` visits (I verified: every read of `ROLE[wid]` in the file is a *self*-read; nothing
reads another wave's slot for control flow). Mechanism 2 writes the same self-mailbox at grow-fail entry,
and otherwise touches only local SGPRs (`s44/s45/s52/s75/s50/s34`, all audited below) and the VGPR
allocator. Neither mechanism writes `SL_RBNEXT`, `SL_STI`, `SL_GEN`, or any other per-slot field that a
*different* wave's claim path reads — the P1 STAMP block (which does write those, and which this task's own
prior review already audited for the release-fence ordering) is now reached only when `DSWS2_ROLEFLOW=0`,
completely unchanged. I grepped every `SL_RBNEXT` write site again post-edit: the same six sites P1's
progress doc enumerated, none of them touched by this pass.

## SGPR liveness audit

New persistent register: **s75** (feed→compute reversion hysteresis counter). Audited free by exhaustive
grep of the whole file before use: `s75` has exactly one other occurrence anywhere in `occ_kernel_dsws_flow.s`
— a *read* inside `trace_row` (`.if TRACE`) labeled "wg_id," with no writer anywhere in the file (TRACE's own
wg_id field appears to have never been wired, consistent with this project's documented pattern of
half-wired diagnostics — CLAUDE.md's own list of counters that "read 0 because they were never wired"). Every
build this task targets has `TRACE=0` (not part of the A1 profile string), so `s75` is provably unused
elsewhere in every gated configuration. RGA's independent livereg count corroborates this: SGPR peak went
from P1's 54 to 55 — **exactly +1**, matching the single new persistent register added, with 0 SGPR spills
and 0 VGPR spills. Had I collided with something live, I would expect either no visible change (masked) or a
much larger disruption (spills, or a wrong instruction count skew); a clean, exact +1 is the signature of an
isolated, correctly-audited addition, not proof by itself but a strong corroborating signal.

Scratch registers reused (all confirmed dead at the relevant program points, matching or directly mirroring
existing call sites):
- **s44, s45, s52** at `.Lflow_dispatch`'s reversion check: freshly clobbered here exactly as `.Lflow_compute`/
  `.Lflow_feed` themselves immediately clobber s44/s45/s46 as their own first action — nothing depends on
  their incoming values at this point in the loop.
- **s45** at the grow-fail retry's mailbox writes: dead at both sites (growfail entry, post-retry-success) —
  same role-mailbox-address idiom already used identically at `occ_kernel_dsws_flow.s` RCONV's own conversion
  site and at `.Lflow_body`'s role read, both of which treat s45 as pure scratch.
- **s44 (r), s52 (gi)** across the ENTIRE retry loop: this is the one register-liveness claim that actually
  matters for correctness, not just hygiene. I traced every instruction between `.Lflow_da_ss_growfail:` and
  `.Lflow_da_ss_decode:` (the retry loop body: `cnt_inc`, `flow_gauge`, `lds_put_r` ×2, `deadman_check`,
  `s_sleep`, `s_alloc_vgpr`, `duty_grow`, `fat_inc`, `phase_stamp`) and confirmed none of their macro bodies
  write s44, s45(as *data*, only as address scratch which is fine since s44/s52 aren't addresses here),
  s46, s47, or s52. `deadman_check` clobbers s62/s63/s70/s71/DM_PROG(s101) only. `cnt_inc`/`flow_gauge` are
  either pure-SALU-on-a-private-counter or entirely `FORENSICS`-gated no-ops in this profile. `duty_grow`/
  `fat_inc`/`phase_stamp` are entirely `DUTYPROBE`/`FATGAUGE`/`PHASEPROBE`-gated no-ops in this profile
  (all three default 0, none set by the A1 profile string) — so in the actual gate-tested ON build these
  three calls emit zero instructions; they are kept only for byte-for-byte parity with the original
  grow-success call sequence should someone build with those diagnostics enabled.
- **s50**: intentionally reset (not scratch) at both ROLE_COMPUTE-writing sites — see the "s50 re-arm" note
  above. This is a deliberate, audited write, not an accidental clobber.

No SGPR used by this pass overlaps `s24` (wid), `s34` (cur_role, deliberately written only at the two
documented reversion points), `s49`/`s57` (LDS-macro/generic exec-save convention — untouched, still free for
any macro this new code calls), `s67`/`s68` (mask/shift, live kernel-wide, never touched), `s69` (chunkHi),
`s70`/`s71` (deadman — touched only via the macro's own protocol, exactly as intended), or `s41` (group,
live within a compute burst — GROUPS=2 in this profile, so this register genuinely matters and I confirmed
neither mechanism writes it).

## Offline gates

**Gate 1 — `DSWS2_ROLEFLOW=0` byte-identical to `cac3ff7c...`.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_ROLEFLOW=0 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  LDS=54784B
sha256sum occ_dsws2_w30_flow_gd.bin
  cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match. (Required `build_flow.sh` fix: the script did not forward `DSWS2_ROLEFLOW`/
`DSWS2_ROLEFLOW_BACK_N` to the assembler at all before this pass — an env var set on the shell command line
would have been silently ignored and the `.ifndef` default used regardless. Wired both through next to the
existing `DSWS2_OVERLAP`/`OVERLAP` line so `DSWS2_ROLEFLOW=1` actually reaches `-Wa,-defsym`.)

**Gate 2 — ON build (`DSWS2_ROLEFLOW=1 DSWS2_OVERLAP=1 DSWS2_RCONV=1`, full A1 profile) assembles + links
0-spill.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 DSWS2_ROLEFLOW=1 \
  ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (34064B .text)  LDS=13824B
```
LDS still 13824B (P1's one-generation reclaim, unaffected by this pass — expected, nothing here touches
LDS layout). `.text` 34064B vs P1's 33808B (+256B, consistent with the added instructions).

RGA (linked `.co`, exact same defsym profile as above plus `RGADESC=1`, via
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin`, purely static, no GPU
dispatch):
```
DEVICE,...,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,...,ISA_SIZE
gfx1201,...,106,72,0,256,256,0,...,31236
Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  55, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** VGPR peak 48 — identical to P1 (nothing here touches a VGPR; the whole
pass is SALU + LDS bookkeeping). SGPR peak 55 vs P1's 54 — **exactly +1**, matching the single new
persistent register (`s75`); see the register audit above for why this is corroborating evidence, not just
a number. ISA_SIZE 31236 vs P1's 31176 (+60B, consistent with `.text` delta).

**Gate 3 — host `occ_dispatch.cpp` compiles; guards hold.**
`./build.sh` completed (`OK -> ./occ_dispatch`), same pre-existing 23 `-Wformat` warnings, 0 errors.
`occ_dispatch.cpp` was **not touched this session** (confirmed via `git diff --stat`: only `build_flow.sh`
and `occ_kernel_dsws_flow.s` carry this session's edits). Hand-verified still present: `kOpBase`/
`static_assert` (`occ_dispatch.cpp:1906-1907`), `dsws2Overlap`/`ldsBytesRaw` (`:1918-1919`) — unaffected
by this pass since it changes no LDS layout and no kernarg contract.

**Gate 4 — `.if`/`.endif` nesting.** Full-file balance check (accounting for trailing `//` comments on
`.endif` lines, which a naive exact-string match undercounts): depth 0, no unmatched directives.

## Scope discipline

Only `occ_kernel_dsws_flow.s` (the mechanism) and `build_flow.sh` (defsym wiring, required for the mechanism
to be reachable at all) were edited this pass. `occ_dispatch.cpp` was read-only (Gate 3 verification only).
`occ_kernel_coop.s` was never opened. Nothing staged (`git add`/commit/stash never run) or dispatched
(`./gpu_run.sh`/`./occ_dispatch` never invoked, not even `test_oracle`). The baseline bin
(`DSWS2_ROLEFLOW=0` A1 profile) was rebuilt as the LAST build of this session — `occ_dsws2_w30_flow_gd.bin`
on disk is `cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553`, verified by a final
`sha256sum` after the Gate 2/RGA builds. The `rga_out/p2a_on` scratch directory used for Gate 2's static
analysis was removed afterward.

## STOP items

None reached the level of "cannot proceed" — but one judgment call is flagged for adversarial review rather
than asserted as settled: the choice to let a deadman-triggered force-retire during the grow-fail retry
degrade to "unresolved, same as P1" rather than building a bespoke fallback-stamp deadman variant (see
argument (b)'s caveat). I believe the two are functionally equivalent terminal states given POOL_N=1's
single-physical-slot `SL_GEN==head` gating, and that adding a second mechanism for a case I could not verify
adds anything would have been exactly the kind of improvised new synchronization surface this task asked me
to lean away from — but this reasoning deserves a second set of eyes before any GPU run, same as every other
handoff-touching change to this file.
