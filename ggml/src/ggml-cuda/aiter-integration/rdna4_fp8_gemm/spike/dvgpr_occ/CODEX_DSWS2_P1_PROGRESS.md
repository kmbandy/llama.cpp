# DSWS2 Phase 1 (redone) — grow-fail ring handoff removed, 2026-07-24 (Sonnet builder)

**Status: GATES PASS.** OFFLINE ONLY throughout: no `./gpu_run.sh` / `./occ_dispatch` invocation, no GPU
dispatch, no `test_oracle` even. No `git add`/commit/stash. Only `occ_kernel_dsws_flow.s` was edited this
pass (`occ_dispatch.cpp` and `build_flow.sh` were already dirty from the prior Phase-1B pass; confirmed
unchanged by this session — see Scope discipline).

## What this replaces

`CODEX_OVERLAP_P1B_PROGRESS.md`'s grow-fail handoff (STAMP `SL_RBNEXT=0` + the new
`.Lflow_da_gf_stage_walk`) is exactly the code `CODEX_OVERLAP_P1B_REVIEW.md` found 3 BLOCKERs in. Per
`DESIGN_DSWS2_ROLE_ECONOMY_2026-07-24.md` §6 step P1, this pass keeps Phase-1B's L2-reclaim (unchanged) and
replaces ONLY the grow-fail publish with **mechanism (b): revert to the original, pre-Phase-1B, non-claimable
`RB_PENDING` stamp semantics** — i.e., under `DSWS2_OVERLAP=1`, grow-fail now stamps and behaves BYTE-FOR-BYTE
identically to `DSWS2_OVERLAP=0`'s grow-fail path. No new claimable state, no new walk, no new mechanism at
all in this one spot.

## Why (b) over (a), and why it's provably minimal

The design doc offered two candidates: (a) hold-and-retry (the grow-failed wave keeps its reservation and
retries the grow next loop) or (b) non-claimable publish (revert the poison). I chose (b) because:

- **Under `CFASSIGN=1`** (required by the A1 profile), `.Lflow_da_rollback` — the ONLY mechanism that gives
  back a reservation index `r` for someone else to grab — is compiled out (`.if !CFASSIGN`,
  occ_kernel_dsws_flow.s:4604/4612 pre-edit). Once a wave reaches `.Lflow_da_stamp` for reservation `r` under
  our profile, it unconditionally owns publishing *something* for `r` (a fresh CAS-reservation cannot be
  "returned"). Option (a)'s "keeps its own reservation and retries the grow next loop" therefore cannot mean
  re-entering the top-level `.Lflow_loop` dispatch (which might attempt a NEW CAS reservation, or re-derive
  role, on a totally different index) — it would need a NEW, bespoke retry point that preserves `s44=r` and
  `s52=gi` (the payload) across a spin/backoff without touching `.Lflow_loop`'s existing state machine. That
  is new synchronization surface in exactly the region the task said to lean hardest against improvising in.
- Option (b) requires **zero** new synchronization: it makes the `DSWS2_OVERLAP` branch of the STAMP block
  literally identical to the `DSWS2_OVERLAP=0` branch that already exists, is already exercised by every
  non-overlap build, and already matches the kernel's own cold-start init poison
  (`occ_kernel_dsws_flow.s:3199`, `SL_RBNEXT=RB_PENDING`, DECENTASN cold-start comment above it). There is
  nothing new to get wrong.
- The design doc's own §6 P1 text explicitly pre-blesses exactly this tradeoff ("leaves the item
  un-completable under L2-only... a LATENT stall IF grow-fail fires... acceptable ONLY because grow-fail=0 in
  this regime... must NOT stall while grow-fail=0") — I did not need to invent a new argument, only verify it
  holds (below).

## The edit (file:line, on the pre-edit Phase-1B source)

1. **`occ_kernel_dsws_flow.s:4717-4730`** (pre-edit): the `.if DSWS2_OVERLAP / lds_put_r s45, 0 / .else /
   lds_put_r s45, RB_PENDING / .endif` around the `SL_RBNEXT` stamp is now a single unconditional
   `lds_put_r s45, RB_PENDING` (occ_kernel_dsws_flow.s:4732 post-edit) — no more `.if` at all at this site;
   `DSWS2_OVERLAP=0` and `=1` now emit the identical instruction here. This alone removes BLOCKER 1 at the
   root: the claimable state (`RB_PENDING` clear) this stamp used to create no longer exists on grow-fail, so
   there is nothing for the stale-STI race to race against.
2. **`occ_kernel_dsws_flow.s:4753-4786`** (pre-edit): the whole `.if DSWS2_OVERLAP … .Lflow_da_gf_stage_walk
   … .Lflow_da_gf_stage_done: drain_advance … .endif` block is deleted outright, replaced by a seven-line
   comment (occ_kernel_dsws_flow.s:4755-4762 post-edit) explaining why no walk belongs here anymore: that walk
   existed only to make the (now-removed) claimable stamp visible to a consumer; with nothing claimable
   published there is no consumer to signal.
3. Both edits are **pure deletions/reversions** — no new labels, no new branches, no new LDS addresses, no
   new registers. `s45`/`s46` scratch usage at this site is unchanged from the pre-Phase-1B original.

Everything else Phase-1B built is **kept, verbatim, untouched by this pass**:
- Ring-compute self-load (`occ_kernel_dsws_flow.s:3669-3900`, `.if DSWS2_OVERLAP` inside `.Lflow_havestage`'s
  claim body) — the L2 self-load conversion.
- Feed neuter (`:4136-4141`, `.Lflow_feed` unconditional redirect to `.Lflow_feed_empty`).
- Coast neuter (`:5161-5168`, `.Lflow_coast` unconditional redirect to `.Lflow_feed_empty`).
- `ACC_BASE` collapse to `OP_BASE` (`:882-888`) and its two `.error` guards (`SELFSERVE=1`, `POOL_N=1`)
  immediately above (`:873-880`).
- Host `occ_dispatch.cpp` `DSWS2_OVERLAP` dispatch-time env var / `ldsBytesRaw` co-change (`:1918-1946`),
  `kOpBase`/`static_assert` (`:1906-1907`) — verified present and unmodified, see Gate 3.

## The "nothing claimable" argument

**Claim:** on grow-fail, under `DSWS2_OVERLAP=1`, the wave publishes nothing another wave can claim.

The ONLY code path in the whole file that can transition a slot's `SL_RBNEXT` from poisoned/exhausted to a
claimable low value (`0 <= x < ACC_N`, no `RB_PENDING` bit) is the POISON-UNTIL-STAGED claim's own
`lds_cas_rtn` (`occ_kernel_dsws_flow.s:3600`), and that CAS only succeeds when the pre-check at `:3586-3588`
(`s_and_b32 s47, s33, RB_PENDING; s_cmp_lg_u32 s47, 0; s_cbranch_scc1 .Lflow_cmp_tryadv`) sees the pending bit
clear — i.e., it never *creates* claimability itself, it only *consumes* claimability that some producer
already published. I grepped every `SL_RBNEXT` write site in the file (6 total, unchanged from Phase-1B's own
enumeration plus my reversion):
- `:3199` cold-start init → `RB_PENDING` (poisoned).
- `:4619` `.Lflow_da_sentinel` (rollback/phantom) → `ACC_N` (exhausted, not claimable).
- **`:4732` grow-fail STAMP (this pass) → `RB_PENDING` (poisoned, matches `DSWS2_OVERLAP=0`).**
- `:4816` `.Lflow_da_ss_decode` (self-serve success sentinel) → `ACC_N` (exhausted, not claimable).
- The only *clearer* of `RB_PENDING` anywhere in the file is `side_final` (`:1370`ish, inside
  `ASTAGE_R`/`BSTAGE_R`), and its only two call sites are `.Lflow_feed` and `.Lflow_coast` — both
  unconditionally redirected to `.Lflow_feed_empty` before reaching those macros under `DSWS2_OVERLAP=1`
  (kept from Phase-1B, items 2-3 above).

So under `DSWS2_OVERLAP=1`: every producer of a new reservation publishes either an exhausted sentinel (no
claim possible) or a permanently-poisoned stamp (no claim possible, and nothing left alive to un-poison it).
**There is no code path, on grow-fail or otherwise, that publishes a claimable `SL_RBNEXT` state under this
defsym.** This dissolves all three BLOCKERs structurally, not by patching the race:
- **BLOCKER 1** (stale-STI claim): required a claim to succeed on a slot whose `SL_STI` hadn't been updated
  yet. With `SL_RBNEXT` never becoming claimable, no claim ever succeeds on THIS slot in the first place —
  there is no window to race, because there is no claim.
- **BLOCKER 2** (premature drain on reuse): required `drain_advance` to observe a stale `SL_RBDONE==ACC_N`
  for a slot holding real, uncomputed work. `drain_advance` (`:1303`) gates on `SL_GEN==DRAIN_HEAD` first
  (`:1319-1323`) then `SL_RBDONE>=ACC_N` (`:1331-1333`); for our permanently-poisoned growfail slot,
  `SL_RBDONE` is written `0` in this very stamp (`:4733-4734`, unchanged, still ordered before `SL_GEN`'s
  publish at `:4752-4753`) and nothing ever touches it again (nobody claims the slot to compute it), so
  `drain_advance` correctly sees `RBDONE=0 < ACC_N` and refuses to advance past it — a safe stall, not a
  silent skip.
- **BLOCKER 3** (no ring consumer after RCONV): this blocker assumed a claimable ring item exists and asked
  "who's left to claim it". P1 removes the premise: there is no claimable ring item to begin with, on
  grow-fail or otherwise, so the liveness question about consumers does not arise for this path. (RCONV's
  actual behavior — converting compute waves to `ROLE_AFEED`, which then no-ops via the same
  `.Lflow_feed_empty` redirect — is unchanged and was never itself broken; the blocker was specifically about
  what happens to a growfail-published item once the ring has no consumers, and that item no longer exists.)

## Correctness while grow-fail=0 (P1's actual required regime)

The reverted STAMP block (`:4706-4753` post-edit) is reached ONLY via fall-through from
`.Lflow_da_ss_growfail:` (`:4701`), which is reached ONLY via `s_cbranch_scc0 .Lflow_da_ss_growfail` right
after `s_alloc_vgpr NFV` (`:4693-4694`). Every measured shape reads grow-fail=0% (`DSWS_S1_STATUS_2026-07-23
.md`) because the kernel is assign-bound in this whole regime — starved waves never contend the VGPR budget.
Under grow-fail=0 this entire code path — both the STAMP write and the (now-absent) stage-walk — **never
executes**, on any build, dispatch, or shape this profile can produce. Its content is therefore provably
irrelevant to correctness in the regime P1 is required to be correct in; the "nothing claimable" argument
above is what makes it *also* safe on the rare/hypothetical occasion it does fire (full stall rather than
silent wrong-C), which is the P1 bar the design doc set ("correct-if-it-fires is enough").

## Known P1 limitation (by design, not an oversight) — the P2 seam

If grow-fail ever DOES fire under `DSWS2_OVERLAP=1` in a build/regime where it can (i.e., anything beyond
P1's own tested scope), the affected slot's `RB_PENDING` poison is now permanent (`side_final`, its only
clearer, is unreachable — feed/coast are neutered). `STAGE_HEAD` cannot pass this slot (its
`SL_RBNEXT & RB_PENDING != 0` forever), and since `drain_advance` caps `DRAIN_HEAD` at `STAGE_HEAD`
(`:1314-1315`), `DRAIN_HEAD` stalls at the same point — which stalls the `ASSIGN==DRAIN` tile/group boundary
close (`GSTORED`) for every reservation after it, not just this one. This is a **known, accepted, documented**
P1 limitation, explicitly named in `DESIGN_DSWS2_ROLE_ECONOMY_2026-07-24.md` §6 step P1 and §7's grow-fail
seam language. **P2 replaces this whole path with grow-fail-flows-to-feed** (a real, live consumer under the
bidirectional role economy), which is exactly the seam this comment block marks in the source
(`occ_kernel_dsws_flow.s:4717-4731` and `:4755-4762`). Nobody should silicon-test grow-fail against this P1
build; it is explicitly out of scope until P2/P3/P4 make more work concurrently exposed (§4 of the design
doc).

## Offline gates

**Gate 1 — `DSWS2_OVERLAP=0` byte-identical to baseline.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=0 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  LDS=54784B
sha256sum occ_dsws2_w30_flow_gd.bin
  cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match to the canonical A1 baseline sha in `HARNESS.md`. (Trivially expected here: my two
edits removed the `.if DSWS2_OVERLAP` branching at both sites rather than adding new `.else` content, so
there is no longer even a divergent code path to keep in sync for byte-identity at this location.)

**Gate 2 — `DSWS2_OVERLAP=1 DSWS2_RCONV=1` (full A1 profile) assembles + links + RGA 0-spill.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (33808B .text)  LDS=13824B
sha256sum occ_dsws2_w30_flow_gd.bin
  a421a4d714bb4a7fbec91439ac5fe3af2b7910fefa95b5851f4d6670e5db5206  occ_dsws2_w30_flow_gd.bin
```
LDS still 13824B (matches Phase-1B — same one-generation reclaim, unaffected by the grow-fail edit).
`.text` is 388B smaller than Phase-1B's 34196B (removing the stage-walk block removes instructions; no
functional path lost since it never executed for grow-fail=0 anyway).

RGA (linked `.co`, same full defsym set as `build_flow.sh`'s `mkflow()` plus `RGADESC=1`, run via
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin`, purely static
disassembly-based analysis, no GPU dispatch):
```
DEVICE,...,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,...,ISA_SIZE
gfx1201,...,106,72,0,256,256,0,...,31176
Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  54, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** Peak live VGPR 48 — identical to Phase-1B's own Gate 2 figure (expected:
the grow-fail STAMP/stage-walk is SALU-only bookkeeping; nothing in this pass touches a VGPR). ISA_SIZE
31176 vs Phase-1B's 31544 (368B smaller, consistent with the `.text` delta above). `AVAILABLE_LDS_BYTES`/
`USED_LDS_BYTES` read 65536/65536 here — same `RGADESC` fixed-descriptor artifact Phase-1B's Gate 2 noted
(`.amdhsa_group_segment_fixed_size 65536`, unconditional under `RGADESC`, unrelated to the real 13824B
allocation reported by `build_flow.sh` above).

**Gate 3 — host `occ_dispatch.cpp` compiles; `ldsBytesRaw`/guards hold.**
`./build.sh` completed (`OK -> ./occ_dispatch`), same pre-existing `-Wformat` warnings (`%u` vs `uint64_t`,
unrelated, 23 total), 0 errors. `occ_dispatch.cpp` was **not touched this session** (verified: only
`occ_kernel_dsws_flow.s` shows edits from this pass in `git diff --stat`; `occ_dispatch.cpp` and
`build_flow.sh` are dirty only from the already-present prior Phase-1B pass). Hand-verified still present and
unmodified: `kOpBase`/`static_assert` (`occ_dispatch.cpp:1906-1907`), the `dsws2Overlap` dispatch-time env
var and `ldsBytesRaw` formula (`:1918-1924`), the authoritative `.lds`-override mismatch-warning path
(`:1924-1946`), and the `>1 WG/CU` occupancy guard (`~2010-2024`, Phase-3/4 territory, untouched and out of
scope here).

**Gate 4 — `.if`/`.endif` nesting.** A full-file balance check (every `.if`/`.ifdef`/`.ifndef` vs `.endif`)
returns depth 0, no unmatched directives — the two block deletions did not leave a dangling conditional.

## Scope discipline

Only `occ_kernel_dsws_flow.s` was edited this pass, at exactly the two sites named in the task (the STAMP's
`SL_RBNEXT` write, and the block containing `.Lflow_da_gf_stage_walk`). Nothing else in the file was touched:
grepped and re-read the ring-compute self-load, feed neuter, coast neuter, and `ACC_BASE` reclaim sites above
to confirm they are byte-identical to the Phase-1B state this session inherited. `occ_dispatch.cpp` and
`build_flow.sh` were opened only to read (Gate 3 verification, and to reconstruct the exact RGA defsym line
for Gate 2) — no writes. `occ_kernel_coop.s` was never opened. Nothing was staged (`git add`/commit/stash
never run this session) or dispatched (`./gpu_run.sh` / `./occ_dispatch` never invoked; not even
`test_oracle`, since that requires running the built host binary — this pass stopped at `build.sh`'s own
compile-only check, which is sufficient for Gate 3 and keeps this pass strictly offline).

## STOP items

None. Both candidate mechanisms from the task were evaluated; (b) was chosen over (a) because it requires
literally zero new synchronization (a reversion, not new code) versus (a)'s need for a new retry point that
would have to coexist with `.Lflow_loop`'s existing dispatch without double-reserving work under `CFASSIGN=1`
— exactly the kind of new race-prone surface the task said to avoid inventing. The "nothing claimable"
argument above is structural (enumerates every `SL_RBNEXT` write site in the file), not probabilistic. The
one thing this pass explicitly does NOT claim: that grow-fail firing produces good *throughput* — only that
it cannot produce a *silent wrong-C*, and that it cannot fire at all in the regime this build is gated for.
That grow-fail-firing-and-recovering-usefully is P2's job (bidirectional flow + grow-fail-to-feed), per the
design doc's own phasing.
