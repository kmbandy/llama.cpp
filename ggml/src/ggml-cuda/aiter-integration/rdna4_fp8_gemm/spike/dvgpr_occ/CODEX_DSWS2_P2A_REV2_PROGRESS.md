# DSWS2 Phase 2a REV2 — grow-fail rebuilt as GROW-FIRST, COMMIT-AFTER, 2026-07-24 (Sonnet builder)

**Status: GATES PASS.** OFFLINE ONLY throughout: no `./gpu_run.sh` / `./occ_dispatch` invocation, no GPU
dispatch, no `test_oracle`. No `git add`/commit/stash. File touched this session: `occ_kernel_dsws_flow.s`
only. `build_flow.sh` was read (already forwards `DSWS2_ROLEFLOW`/`DSWS2_ROLEFLOW_BACK_N` from the prior
session — no change needed) and `occ_dispatch.cpp` was read-only (confirmed via `git diff --stat`: only
`occ_kernel_dsws_flow.s` carries this session's edits; `build_flow.sh`/`occ_dispatch.cpp` show only the
prior session's pre-existing, untouched-by-me diff). `occ_kernel_coop.s` was never opened.

## Why mechanism 2 (grow-fail spin-park) was rejected

The prior pass (`CODEX_DSWS2_P2A_PROGRESS.md`) implemented grow-fail as `.Lflow_da_ss_growfail_retry`: write
`ROLE_AFEED` (telemetry only), then `deadman_check` / `s_sleep 1` / `s_alloc_vgpr NFV` / branch back on
failure, never returning to `.Lflow_loop`/`.Lflow_body`. The wave **parked** — did no work, fed nothing,
computed nothing — while still holding its reservation `r`/`gi` (registers s44/s52) and, more importantly,
having **already marked its cohort served** (`s15`) and **already counted the emission**
(`cnt_inc CLAIM_NOPERSIST`, occ[96]) before the grow was even attempted. Since `drain_advance` only walks
consecutive completed slots and the boundary advance requires `DRAIN >= ASSIGN`, a parked wave head-of-line
blocks the frontier the instant grow-fail actually fires — invisible today only because grow-fail measures
0% in this profile, but P3/P4 exist specifically to make it bind.

## Root cause: the wave committed BEFORE it grew

Under `CFASSIGN=1` (required by this profile), the CFASSIGN cohort peek (`.Lflow_da_cf_decode` entry,
`occ_kernel_dsws_flow.s:4409-4429`) marked `s15` (the "already served this cohort" register) served
**before** computing `r` and **long before** the grow attempt at `s_alloc_vgpr NFV`
(`occ_kernel_dsws_flow.s:4848` on the original ordering). The `.Lflow_da_stamp` block additionally ran
`phase_stamp`/`phist_bump`/`cnt_inc CLAIM_NOPERSIST` (the counter that must equal `TOTAL_super` exactly)
**before** that same grow attempt. A grow-fail therefore always held something (the served-mark, the
emission count) that it could not complete and — since `CFASSIGN=1` compiles out `.Lflow_da_rollback`
(`.if !CFASSIGN` at its definition, `occ_kernel_dsws_flow.s:4712` area) — could not give back either.

## The commit-point map (exact file:line, pre-rev2 source)

Everything that made the reservation "taken" on the straight-line `CFASSIGN=1 SELFSERVE=1` path from
`.Lflow_da_cf_decode` to the grow, in program order (line numbers are the **pre-rev2** source, i.e. before
this session's edits — cited from the read taken at the start of this session):

1. **`s_mov_b32 s15, s45`** (cf_decode cohort peek, was line 4404) — marks this wid's cohort served. Purely
   a private, per-wave register (not cross-wave-visible), but it is the ONE thing that determines whether
   this wave will ever revisit/retry this cohort. Written **before** `r` was even computed.
2. **`s_add_u32 s44, s44, s24`** (r = cohort_start + wid, was line 4405) — register math only, no
   side effect, but from this point on `r`/`gi` are the reservation this wave is nominally responsible for.
3. **`phase_stamp s78`/`s79`** (was line 4788/4786, PHSPLIT-gated) — closes a timing interval. Inert in the
   gate-tested profile (`PHASEPROBE=0`, not in the A1 profile string) but a real diagnostic write when
   `PHASEPROBE=1`.
4. **`phist_bump PH_RESV_WIN`** (was line 4791) — inert in the gate-tested profile (`PHIST` defaults 0, not
   in the A1 profile string; the file's own comment marks it "PHIST-gated, default OFF").
5. **`cnt_inc CLAIM_NOPERSIST`** (was line 4792) — **the one commit point that is live and correctness-
   relevant in the gate-tested profile.** `STAGINSTR=1` is in the A1 profile, so `cnt_inc` is a real
   `s_add_u32 s[104], s[104], 1` (occ[96] at retire flush), and its own comment says it must equal
   `TOTAL_super` exactly. This fired **before** `s_alloc_vgpr NFV` on the original ordering, so any grow-
   fail-and-retry-on-a-later-pass structure that did not move it would double-count every retried unit.
6. **`s_alloc_vgpr NFV`** (was line 4799 in the pre-rev2 source) — the grow attempt itself, reached only
   after all five commit points above had already fired.

Nothing else publishes anything cross-wave-visible before the grow on this path (the STAMP block that
writes `SL_RBNEXT`/`SL_STI`/`SL_GEN` is reached only on a genuine grow-fail fallthrough or the non-SELFSERVE
path — never on the SELFSERVE success path, which branches straight to `.Lflow_da_ss_decode` instead).

## The rebuild: GROW FIRST, COMMIT AFTER

**`occ_kernel_dsws_flow.s`**, three sites, all gated by a new-vs-old fork so `DSWS2_ROLEFLOW=0` reproduces
the original instruction sequence verbatim:

### 1. Defer the served-mark (`occ_kernel_dsws_flow.s:4417-4427`)

```
.if DSWS2_ROLEFLOW && DYNVGPR
    // do NOT mark served yet -- s45 (cohort end) survives untouched to the grow decision (see below)
.else
    s_mov_b32 s15, s45                             // original: mark served immediately
.endif
    s_add_u32 s44, s44, s24                       // r = cohort_start + wid
    s_branch .Lflow_da_cf_decode
```

**Key fact that makes this free (no new register needed):** `s45` at this point holds "cohort end" — the
exact value the original code wrote into `s15`. I traced every instruction from here through the grow
decision on the non-phantom SELFSERVE path (the common cf_decode fallthrough at `:4502-4524`, then
`.Lflow_da_stamp` itself) and confirmed **nothing writes `s45`** anywhere on that path: `lds_get s52`
(DA_TILE), `lds_get s51` (DA_BASE), and the `within`/`ksi`/`gi` math all use `s44/s46/s47/s51/s52` as
scratch, never `s45`. `phase_stamp`/`phist_bump`/`cnt_inc` (audited macro bodies) touch `s57/s62-64/s77/
s104`/exec/v3/v4, never `s45`. `s_wait_storecnt`/`s_alloc_vgpr` touch no SGPRs besides SCC. So `s45` is
still "this wid's cohort end" at the grow decision, and the deferred write there uses the *same* value the
original, earlier write would have used — just later, once it is true.

The short-cohort/no-unit path (`.Lflow_da_cf_no_unit`, unaffected, still writes `s15` immediately — there is
no unit and therefore no grow to defer against) and the phantom path both still commit `s15` immediately
(see below) since neither one grows.

### 2. Commit the phantom case immediately (`occ_kernel_dsws_flow.s:4514-4523`)

A phantom index (`ksi >= n_kseg`) never self-serves or grows — it always resolves via the pre-completed
sentinel. It must still mark `s15` served (exactly as before, using the same still-live `s45`), just moved
from the old unconditional site to right at the phantom branch:

```
    s_cmp_gt_u32 s47, s66                            // ksi > n_kseg-1 -> phantom
.if DSWS2_ROLEFLOW && DYNVGPR
    s_cbranch_scc0 .Lflow_da_cf_notphantom          // not phantom -> served-mark stays deferred
    s_mov_b32 s15, s45                              // phantom: no grow risk, commit immediately
    s_branch .Lflow_da_sentinel
.Lflow_da_cf_notphantom:
.else
    s_cbranch_scc1 .Lflow_da_sentinel
.endif
```

### 3. Grow before the stamp/commit block (`occ_kernel_dsws_flow.s:4766-4866`)

`.Lflow_da_stamp` is forked wholesale:

```
.Lflow_da_stamp:
.if DSWS2_ROLEFLOW && SELFSERVE && DYNVGPR
    s_wait_storecnt 0x0
    s_alloc_vgpr NFV
    s_cbranch_scc0 .Lflow_da_ss_growfail_flowon
    s_mov_b32 s15, s45                               // commit: mark served (deferred from cf_decode)
    phase_stamp s78/s79 ; phist_bump PH_RESV_WIN ; cnt_inc CLAIM_NOPERSIST   // commit: same instructions,
    duty_grow ; fat_inc ; phase_stamp s80                                    //   now AFTER grow success
    s_branch .Lflow_da_ss_decode
.Lflow_da_ss_growfail_flowon:
    cnt_inc CNT_GROWFAIL ; flow_gauge FDIAG_SS_GROWFAIL_OFF, 1              // diagnostic only, no commit
    s_branch .Lflow_feedmt_sleep                     // FLOW ON -- same bail target every coasting/blocked
                                                      //   wave in this function already uses
.else
    <ORIGINAL pre-rev2 instruction sequence, byte-for-byte, including the original .Lflow_da_ss_growfail
     fallthrough to the STAMP/RB_PENDING block. The rejected spin-park retry has been DELETED, not merely
     disabled: it lived entirely inside `.if DSWS2_ROLEFLOW` nested inside this .else, where DSWS2_ROLEFLOW
     is provably always 0 (the outer fork already took this branch because NOT(ROLEFLOW && SELFSERVE &&
     DYNVGPR) held), so keeping it would have been dead text, not dead-but-needed-for-Gate-1 text.>
.endif
```

On grow-fail: `r=s44`/`gi=s52` are simply abandoned (never referenced again — the wave discards them by
branching away), `s15` was never written (still holds whatever it held from this wave's last successful
commit), and control lands at `.Lflow_feedmt_sleep` — the exact same "bail this iteration, retry next loop"
target used by `.Lflow_da_cf_no_unit`-adjacent bails, `.Lflow_da_boundary`'s contention bail, the pool-full
bail, the ZLOCK-lost bail, and the boundary drain-gate bail elsewhere in this same function. No new bail
path was invented.

## Why the retry is idempotent (not re-derived from scratch — corroborating the task's own framing)

The cohort peek's own comment (`occ_kernel_dsws_flow.s:4380-4381`, unchanged) states the invariant this
relies on: "DRAIN cannot cross that cohort until every eligible wid has published its unique generation."
Since this wave's `SL_GEN` publish for `r` is exactly the missing piece the cohort is waiting on, and this
wave abandoned `r` without publishing it, the cohort **cannot** have advanced by the time this wave next
visits `.Lflow_da_cf_decode` — so the DRAIN-derived cohort_start/cohort_end recompute to the *identical*
values, and `r = cohort_start + wid` recomputes identically. This is not a new liveness argument; it is the
file's own pre-existing cohort-completeness invariant, which the deferred-commit design leans on rather than
works around.

## Register audit: zero new persistent registers

The rebuild adds **no new SGPR**. It reuses `s45` (already live, already holding the exact value needed)
across a stretch already proven clobber-free above, and reuses `s44`/`s52` (r/gi, simply abandoned on
failure, never re-read). No new stash register, no new mailbox, no new counter register was introduced.
RGA's SGPR peak below (55) is identical to the prior (rejected) P2a build's 55 — confirming no net register
change from swapping one mechanism for the other, only from Mechanism 1 (kept as-is, +1 over P1's 54).

## Scope guards added (`occ_kernel_dsws_flow.s:905-916`)

Two new `.error` guards, matching the file's own "bail loud rather than silently mis-cost" idiom:

- `DSWS2_ROLEFLOW && DYNVGPR && !CFASSIGN` — the deferred served-mark only exists on the CFASSIGN cf_decode
  path; this profile already requires CFASSIGN=1, so this never fires here, but it stops a future silent
  misuse rather than leaving the non-CFASSIGN path (which uses a different rollback idiom entirely, no
  `s15`) to interact with the new fork in an unreasoned way.
- `DSWS2_ROLEFLOW && DYNVGPR && (BATCH > 1)` — the `.Lflow_da_ss_batch_next` continuation entry into
  `.Lflow_da_stamp` (reached only when `BATCH > 1`) does not set `s45` to the cf_decode cohort-end value at
  all before branching in (and under `CFASSIGN=1` never even initializes the `s72`/`s73` batch cursor, an
  already-pre-existing, already-unsupported combination) — deferring the served-mark there would read
  garbage. This profile requires `BATCH=1`, so the guard never fires here either; it exists purely so a
  future `BATCH>1` build fails at assemble time instead of silently corrupting `s15`.

Both guards pass trivially at this profile (`CFASSIGN=1`, `BATCH=1`).

## Semantics change, explicitly flagged (not a correctness bug)

`CNT_GROWFAIL` (occ[73]-ish diagnostic, `STAGINSTR`-gated private SGPR, one atomic flush at retire) may now
fire more than once for the same logical reservation across repeated flow-on retries on later passes,
whereas before it fired at most once per grow-fail *episode* (old P1: once ever per `r`; rejected P2a: once
per episode, since the spin-retry loop only counted it at episode entry). This is a semantics clarification
— it counts grow-fail *events*, and always did; it never claimed to count *units* — not a correctness
regression. The one counter that MUST equal `TOTAL_super` exactly, `CLAIM_NOPERSIST` (occ[96]), is
unaffected: it fires exactly once per real reservation, only on the grow-success path, never on the flow-on
path, by construction (see the commit-point map above — it was moved specifically to prevent this).

Reusing `.Lflow_feedmt_sleep` as the flow-on target also means `cnt_inc CNT_FEEDMT` (semantically "feed wave
ran, found nothing to stage") now additionally counts "grow-fail, flowed on" events. This is a deliberate
reuse of the canonical flow-on machinery per the task's own instruction ("returns to the normal dispatch
loop / feed path... does the next productive thing") rather than inventing a parallel bail path with its own
new counter and new correctness surface.

## Argument: on grow-fail the wave holds nothing, blocks nothing, and flows on

- **Holds nothing:** `r=s44`/`gi=s52` are abandoned registers, never referenced again on the fail path. No
  LDS write of any kind happens between "grow attempted" and "branch to `.Lflow_feedmt_sleep`" on that path
  — not `SL_RBNEXT`, not `SL_STI`, not `SL_GEN`, not the ROLE mailbox (the rejected mechanism's own mailbox
  write is deleted, not merely made unreachable). `s15` is untouched, so this wave's own bookkeeping still
  correctly reads "not yet served."
- **Blocks nothing:** since nothing is published to `SL_GEN` for `r`, no *other* wave's view of the world
  changes at all — `ASSIGN_HEAD`/`DRAIN_HEAD`/`STAGE_HEAD`/`DA_ZDONE` are all untouched by this attempt.
  There is no reservation left outstanding-but-unpublishable the way P1's `RB_PENDING` poison or the
  rejected mechanism's held-`r` park were — there is simply nothing here at all for `drain_advance` or the
  boundary gate to trip over.
- **Flows on:** branches straight to `.Lflow_feedmt_sleep`, the same bail-and-retry-next-loop idiom used
  throughout `.Lflow_da_*`, which itself falls through to the STAGGER baton-wake check and `s_branch
  .Lflow_loop` — this wave immediately re-enters the normal dispatch loop and is free to do any other
  productive thing (including, per Mechanism 1, converting role if it observes the reversion signal) before
  it happens to revisit this same cohort and retry.
- **Success path unchanged:** the `.if DSWS2_ROLEFLOW && SELFSERVE && DYNVGPR` success arm executes the
  identical instruction list (`s15` write / `phase_stamp` / `phist_bump` / `cnt_inc CLAIM_NOPERSIST` /
  `duty_grow` / `fat_inc` / `phase_stamp s80` / branch decode) the pre-rev2 code ran on success, just
  reordered to start after `s_alloc_vgpr NFV` returns success instead of before it starts.

## STOP items

**None.** No commitment on this path turned out to be un-deferrable past the grow — every prior "commit"
point (`s15`, the three diagnostics, the emission counter) is either a private per-wave register (safely
delayed with no cross-wave visibility ever) or provably inert in the gate-tested profile
(`phase_stamp`/`phist_bump`, both gated off by `PHASEPROBE=0`/`PHIST=0`), except `cnt_inc CLAIM_NOPERSIST`,
which is exactly the one the reorder was designed to fix. No new claimable LDS publish was introduced, and
no park/spin/poison survives in the rebuilt path.

## Offline gates

**Gate 1 — `DSWS2_ROLEFLOW=0` byte-identical to `cac3ff7c...`.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_ROLEFLOW=0 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  LDS=54784B
sha256sum occ_dsws2_w30_flow_gd.bin
  cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match, no `build_flow.sh` change needed (the prior session already wired the defsym
through).

**Gate 2 — ON build (`DSWS2_ROLEFLOW=1 DSWS2_OVERLAP=1 DSWS2_RCONV=1`, full A1 profile) assembles + links
0-spill.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 DSWS2_ROLEFLOW=1 \
  ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (33948B .text)  LDS=13824B
```
LDS still 13824B (unaffected — nothing in this pass touches LDS layout). `.text` 33948B vs the *rejected*
P2a build's 34064B (**-116B**, consistent with deleting the spin-retry loop: mailbox write + deadman_check +
s_sleep + s_alloc_vgpr + branch-back + duplicate duty_grow/fat_inc/phase_stamp/branch, none of which survive
in the rebuild).

RGA (`rga_check.sh`, linked `.co`, same defsym profile as above plus `RGADESC=1`, via
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin`, purely static, no GPU
dispatch):
```
DEVICE,...,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,...,ISA_SIZE
gfx1201,...,106,72,0,256,256,0,...,31132
Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  55, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** VGPR peak 48 — identical to every prior pass (nothing here touches a
VGPR). SGPR peak 55 — identical to the rejected P2a build (Mechanism 1's `s75` is the only persistent
addition over P1's 54; this rebuild adds zero more, as argued in the register audit above). ISA_SIZE 31132
vs the rejected build's 31236 (**-104B**, consistent with the `.text` delta).

**Gate 3 — host `occ_dispatch.cpp` compiles; guards hold.**
`./build.sh` completed (`OK -> ./occ_dispatch`), same pre-existing 23 `-Wformat` warnings, 0 errors.
`occ_dispatch.cpp` was **not touched this session** (`git diff --stat` shows no change beyond the prior
session's pre-existing, already-committed-to-working-tree diff).

**Gate 4 — `.if`/`.endif` nesting.** Full-file balance check (a corrected version of the one used last
session — the original script mis-flagged every `.ifndef`/`.ifdef` as unbalanced; fixed to count them as
openers too): depth reaches 0 at EOF, no negative-depth point anywhere in the file.

## Scope discipline

Only `occ_kernel_dsws_flow.s` was edited this session. `build_flow.sh` and `occ_dispatch.cpp` were read for
verification only (Gate 1/3), confirmed unchanged by this session via `git diff --stat`. `occ_kernel_coop.s`
was never opened. Nothing staged (`git add`/commit/stash never run) or dispatched (`./gpu_run.sh`/
`./occ_dispatch` never invoked, not even `test_oracle`). The baseline bin (`DSWS2_ROLEFLOW=0` A1 profile)
was rebuilt as the LAST build of this session — `occ_dsws2_w30_flow_gd.bin` on disk is
`cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553`, verified by a final `sha256sum` after
the Gate 2/RGA builds. The `rga_out/p2a_rev2_on` scratch directory used for Gate 2's static analysis was
removed afterward (confirmed via `ls rga_out` afterward — only pre-existing, much older scratch directories
from prior sessions remain, and `rga_out/` is gitignored).
