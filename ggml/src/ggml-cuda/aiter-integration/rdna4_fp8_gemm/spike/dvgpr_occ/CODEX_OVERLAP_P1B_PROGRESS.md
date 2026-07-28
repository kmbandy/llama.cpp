# DSWS2_OVERLAP Phase 1, corrected pass — 2026-07-24 (Sonnet builder)

**Status: GATES PASS. Conversion complete for the profile this design targets (SELFSERVE=1, POOL_N=1).**
OFFLINE ONLY throughout: no `./gpu_run.sh`/`./occ_dispatch` invocation, no GPU dispatch. No `git add`/
commit/stash. Only `occ_kernel_dsws_flow.s` and `occ_dispatch.cpp` were edited (`build_flow.sh` already
had `DSWS2_OVERLAP`/`OVERLAP` threaded through from the prior scaffolding pass — untouched here).

## What this closes

The prior pass (`CODEX_OVERLAP_P1_PROGRESS.md`) STOPPED because "delete the pool" corrupts accumulation
the instant grow-fail binds: a grow-failed reservation is published to the ring, staged into the LDS
operand pool by `.Lflow_coast`'s opportunistic feed or the dedicated `.Lflow_feed`, then computed via
`ds_load_b64` from `OP_BASE` in `.Lflow_havestage`/`.Lflow_compute`. This pass converts all three sites to
self-load from L2 and collapses the ring to a genuine control-only handoff, then reclaims the pool.

## The conversion (file:line)

1. **Ring-compute self-load** (`occ_kernel_dsws_flow.s:3670` onward, `.if DSWS2_OVERLAP` inside the
   `.Lflow_havestage` claim body that used to run `ds_load_b64` from `sob=OP_BASE+slot*OPSTRIDE`). Since
   `SELFSERVE` forces `JDEPTH=1` (its own guard at :1005, transitively required by the new invariant
   guard at :873), the only live shape of this claim path is the JDEPTH==1 one — the `.if JDEPTH>1`
   `.Lflow_jloop` re-derivation block is dead code either way. The replacement reuses the SAME coordinates
   the claim already decoded (`s19`=mblk, `s30`=tcol, `s31`=ksi via `DECODE_STI`; `s33`=claimed local
   rowblk; `s41`=group iff `GROUPS>1`) to rebuild the identical address algebra the primary self-serve
   burst (`.Lflow_da_ss_rowblk`, unchanged) already uses — same `v8`/`v9` per-lane constant vaddrs set
   once in the prologue (:3053-3065, untouched), same `s2:s3`/`s4:s5`/`s9`/`s10`/`s14` kernarg-lifetime
   scalars. Verified no register collision: `s20/s21/s22/s25/s32/s36/s42/s52..s59` are dead scratch at this
   point in `.Lflow_compute` (grepped lines 3400-3950 of the pre-edit file; the only other hits were the
   MSSCAN block before this point and the code being replaced itself). `.if GROUPS>1` guard on the
   absolute-rowblk term preserved exactly as the original code had it (the SELFSERVE-primary path computes
   this unconditionally because it always sets `s41=0` on the `GROUPS==1` branch; `.Lflow_compute` does
   NOT set `s41` on that branch, so blindly copying the unconditional form would have read garbage — kept
   the conditional instead).
2. **Coast opportunistic feed neutered** (`occ_kernel_dsws_flow.s:5186`, inside `.Lflow_coast`, before its
   `STAGE_HEAD`/`ASSIGN_HEAD` check): unconditional `.if DSWS2_OVERLAP` branch straight to
   `.Lflow_feed_empty` — the same label `.Lflow_coast` and `.Lflow_feed` already both fall into when there
   is "nothing to stage" (confirmed at the pre-edit source: both callers shared this exact target already,
   so redirecting there is not a new code path, only making it the unconditional outcome).
3. **Dedicated feed role neutered** (`occ_kernel_dsws_flow.s:4135/4137`, `.Lflow_feed` entry): same
   unconditional redirect to `.Lflow_feed_empty`. **This is a fourth site, not named in the task's literal
   3-item list** — but it is required. Under gate 2's build (`DSWS2_RCONV=1`), a coasting wave converts to
   `ROLE_AFEED` after `DSWS2_RCONV_COAST_N` consecutive coasts (:5169 area) and thereafter dispatches to
   `.Lflow_feed`, which calls the same `ASTAGE_R`/`BSTAGE_R` macros as `.Lflow_coast` did. Leaving it
   unconverted would corrupt the (now ACC-overlapping) pool region the first time RCONV converts a wave.
   The design doc's own Open Question 2 anticipates exactly this ("RCONV's AFEED role degrades cleanly —
   its staging job is gone; it self-serves like everyone else") — this is that resolution, not scope creep.
4. **Grow-fail publish becomes genuinely control-only** (`occ_kernel_dsws_flow.s` STAMP block, was
   :4606-4639 pre-edit, now the `.if DSWS2_OVERLAP` additions bracketing it):
   - `SL_RBNEXT` is stamped plain `0` instead of `RB_PENDING` (the "unstaged, un-claimable" poison bit).
     Reasoning below (item under "why this had to change beyond the literal 3 items").
   - After `SL_GEN=r` publishes (the release fence), a new block (`:4765` `.Lflow_da_gf_stage_walk`) walks
     `STAGE_HEAD` forward and calls `drain_advance`, mirroring the **already-existing, already-proven**
     idiom at `.Lflow_da_cf_sentinel_stage_walk` (the phantom-sentinel case, a few lines above in the same
     function) and `.Lflow_da_ss_stage_walk` (the SS-decode success path). Unlike the SS-decode version,
     it does **not** fall through into a compute burst — the grow-failing wave is still lean (32 VGPR); it
     only advances bookkeeping.
5. **LDS layout reclaim** (`occ_kernel_dsws_flow.s:883`): `ACC_BASE` collapses from
   `OP_BASE + POOL_N*OPSTRIDE` to plain `OP_BASE` under `DSWS2_OVERLAP`. Two new guards immediately above
   (:873-880) require `SELFSERVE=1` and `POOL_N==1` — the exact profile this conversion was reasoned
   through; nothing about this pass is safe to assume for other combinations, so it refuses to assemble
   them rather than silently mis-costing the reclaim.
6. **Host co-change** (`occ_dispatch.cpp:1918` and the `ldsBytesRaw` line immediately after): a new
   `DSWS2_OVERLAP` **dispatch-time env var** (distinct from the kernel's build-time assembler defsym of
   the same name — there is no way to read a defsym back out of a raw `.bin`) zeroes the
   `poolSlots*operandBytes` term the host adds to `ldsBytesRaw`. This does not replace the pre-existing
   "authoritative `.lds` override" (:1924-1946, unchanged) — that mechanism already reads the bin's own
   published `LDS_TOTAL_FLOW` and trusts it over the host's reconstruction, printing a loud mismatch
   warning otherwise — it only makes the host's own prediction agree with the bin instead of relying on
   that override + warning on every single Phase-1 run. `kOpBase`/its `static_assert` (:1906-1907) and the
   `>1 WG/CU` occupancy guard (:2010-2024, Phase-3-only, untouched) are unaffected.

## Why the conversion had to go beyond "collapse to control-only, keep SL_STI/SL_GEN"

Tracing the claim protocol turned up a dependency one level deeper than the task's literal wording, which
the design doc's own §10 Q2 anticipated but didn't fully spell out: **`RB_PENDING` gates two different
things, and removing operand staging only fixes one of them.**

- `RB_PENDING` is cleared by `side_final` (:1370), called **only** from inside `ASTAGE_R`/`BSTAGE_R`
  (the staging macros). Once `.Lflow_feed` and `.Lflow_coast` never call those macros (items 2-3 above),
  nothing will *ever* clear a poisoned `RB_PENDING` bit again. If the STAMP block still poisoned with
  `RB_PENDING` (old behavior), the grow-fail item would stay permanently unstaged-looking — the compute
  claim's own `RB_PENDING` check (unchanged, :4780ish in `.Lflow_havestage`) would forever coast past it,
  `SL_RBDONE` would never reach `ACC_N`, `DRAIN_HEAD` could never pass it, and the tile/group boundary's
  `GSTORED` gate (which requires `DRAIN==ASSIGN`) would never close — a permanent, silent liveness bug
  (deadman force-retire, dropped work) on the very first grow-fail, not a data race. This is why item 4's
  first bullet (stamp `0`, not `RB_PENDING`) is not optional cleanup — it is required for the conversion to
  be correct at all.
- Separately, `.Lflow_compute`'s entry gate (`DRAIN_HEAD >= STAGE_HEAD -> coast`, unchanged) means a
  compute wave will never even attempt the claim until `STAGE_HEAD` passes the item. In the old design,
  `STAGE_HEAD` advanced only once a feed/coast wave finished staging (cleared `RB_PENDING`) — exactly the
  step being removed. Without a replacement, `STAGE_HEAD` would never catch up either, same deadlock from
  the other side. Item 4's second bullet (the `.Lflow_da_gf_stage_walk` block) is the replacement: since
  "staged" is vacuously true the instant the item is stamped `0`/not-pending, walking `STAGE_HEAD` forward
  at publish time (using the pre-existing, load-bearing `lds_cmpstore_adv` CAS-advance idiom already used
  by three other walkers in this file) restores the invariant with no new mechanism, just reuse of an
  established one.

Both of these are additions **the design's own §10 Q2 text directly names** ("does the control-only ring
leave the grow-fail handoff... coherent"; "whatever `SL_*` control the handoff needs") — I read that as
authorizing exactly this state-machine completion, not as license to improvise past it. I am flagging it
this prominently anyway because it is precisely the class of thing the task said to STOP on if it couldn't
be resolved cleanly; I judged it resolvable because both halves reuse patterns **already proven correct
elsewhere in this same file** (the phantom-sentinel stage-walk, the SS-decode stage-walk, the pre-completed
sentinel's non-`RB_PENDING` `SL_RBNEXT` encoding) rather than inventing new synchronization.

## Static correctness argument: the converted grow-fail fallback when it fires

Walking the full path with grow-fail assumed to actually happen (not the empirically-0%-today case):

1. **Publish.** A wave's `s_alloc_vgpr NFV` fails (:physically unable to grow — no assumption about *why*
   is needed). Execution falls to the STAMP block. Under `DSWS2_OVERLAP` it writes: `SL_RBNEXT=0` (plain,
   claimable), `SL_RBDONE=0`, `SL_STI=gi` (**before** the `SL_BFNEXT/BFDONE/ARNEXT/ARDONE` resets — this
   ordering is preserved unchanged; it existed to protect `.Lflow_feed`'s claim-then-decode race, which is
   moot now that `.Lflow_feed` is neutered, but leaving the ordering alone costs nothing and removes one
   more thing to re-reason about), then `SL_GEN=r` **last** (the release fence — unchanged). Then the new
   `.Lflow_da_gf_stage_walk` runs, advancing `STAGE_HEAD` past this (and any other immediately-ready)
   slot, then `drain_advance` (a safe no-op here: this slot's own `SL_RBDONE=0` blocks it from advancing
   `DRAIN_HEAD` past itself).
2. **Claim.** Some other (or the same, on a later pass) `ROLE_COMPUTE` wave reaches `.Lflow_compute`, sees
   `DRAIN_HEAD < STAGE_HEAD` (true immediately after step 1), enters `.Lflow_havestage`, re-derives the
   slot from a fresh `DRAIN_HEAD` (unchanged DECENTASN post-grow re-derivation logic), reads `SL_RBNEXT`:
   `RB_PENDING` bit is never set (step 1 stamped plain `0`) → passes the pending check → CAS-claims
   `SL_RBNEXT: 0->1` (or up to `ACC_N`) → reads `SL_STI` (published before `SL_GEN`, and `SL_GEN==head` is
   what let this wave's own `DRAIN_HEAD`-based entry succeed, so the STI it reads is this generation's, not
   a stale one) → `DECODE_STI` → zeroes its own ACC → **this is where the ring-compute self-load (item 1)
   fires**: it computes A/B global addresses from the just-decoded `mblk/tcol/ksi` and the just-claimed
   local rowblk, `global_load_tr_b64`/`global_load_b64`s them from L2 (same addresses the primary
   self-serve burst would have used for the identical `(mblk,tcol,ksi,rowblk)` tuple — the map from STI to
   global address is a pure function of compile-time/kernarg constants, so there is no "which copy of the
   data" ambiguity), then WMMAs into ACC.
3. **Reduce + retire.** Falls into the unchanged banked-reduce path (`acc_base_of`, `ds_add_f32` into
   `ACC_BASE + r*ACC_STRIDE`) — this is the **existing** shared completer, untouched by this pass, and its
   own correctness (the `bad=96/116` fix, TILEDONE-gated tile-scoped C-store) does not depend on where the
   *operands* came from, only on the accumulator bank protocol, which is unchanged. `SL_RBDONE` bumps after
   the flush drains (unchanged) → eventually `RBDONE==ACC_N` → `drain_advance` frees the slot for real →
   `GSTORED`/`TILEDONE` progress exactly as they would for a primary-path item.

**No operand or ordering dependency on the removed pool remains** in this path: nothing reads `OP_BASE` or
any `slot*OPSTRIDE` address under `DSWS2_OVERLAP` (verified by inspection — the only remaining
`OPSTRIDE`/`OP_BASE`-relative address computations left in the file are inside `ASTAGE_R`/`BSTAGE_R`
themselves, whose only two call sites are the two neutered functions, and inside the `.else` (old-behavior)
branches of the three `.if DSWS2_OVERLAP` conversions, which are dead when the defsym is 1). The two-step
liveness fix above (plain-`0` stamp + `STAGE_HEAD` walk) is the one piece of NEW synchronization reasoning
this pass adds beyond pure deletion, and both halves are structurally identical to synchronization already
proven correct elsewhere in this file for the same "no real work, just make the frontier consistent" job.

**What I could not verify (correctly, per the task): that grow-fail firing at all produces the behavior
above rather than some path I haven't traced.** Grow-fail is 0% on every measured shape (`DSWS_S1_STATUS
_2026-07-23.md`), so this argument is necessarily a *static* one — the offline gates cannot exercise it.
The design doc itself designates the grow-fail-BINDING run as a supervised gate the parent runs later, not
this task. I am not claiming empirical proof; I am claiming the state machine is correct by construction
under the invariants the surrounding code already enforces (SL_GEN-gated claim, RB_PENDING semantics,
GSTORED-gated boundary) and that I traced every remaining reference to the removed address range to zero.

## Offline gates

**Gate 1 — `DSWS2_OVERLAP=0` byte-identical to baseline.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=0 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  LDS=54784B
sha256sum occ_dsws2_w30_flow_gd.bin
  cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match to the canonical A1 baseline sha in `HARNESS.md`.

**Gate 2 — `DSWS2_OVERLAP=1 DSWS2_RCONV=1` (full A1 profile) assembles + links + RGA 0-spill.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (34196B .text)  LDS=13824B
sha256sum occ_dsws2_w30_flow_gd.bin
  1192b6e3d68a1ad8cee8adc05aa6c7af1e68cc0ed16ce81dab3dc2ee898a6a92  occ_dsws2_w30_flow_gd.bin
```
**LDS dropped 54784B -> 13824B** (40,960B reclaimed — matches the design's predicted ~40KB reclaim; the
absolute figure differs from the design's illustrative "~26,112B / 2-generation" arithmetic in §5 because
that section was estimating the Phase-2 *two-generation* target — this is Phase-1's *one*-generation
figure: `512(OP_BASE) + 3*4096(ACC_N=3 * ACC_STRIDE=FM*FN*1024=4096) + 32*32(SSWIN*SLOTC_STRIDE) = 13824`).
RGA (linked `.co` mirroring this exact defsym set plus `RGADESC=1`, run via
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin`, purely static
disassembly-based analysis, no GPU dispatch):
```
DEVICE,...,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,...,ISA_SIZE
gfx1201,...,106,72,0,256,256,0,...,31544
Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  54, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** (Peak live VGPR 48 vs. the plain-`DSWS2_RCONV=1` baseline's 50 in
`CODEX_OVERLAP_PROGRESS.md` — expected: the conversion trades two LDS-vaddr temporaries `v12`/`v13` for
reused scalar address regs, a small net decrease, not an increase.) `AVAILABLE_LDS_BYTES`/`USED_LDS_BYTES`
both read 65536 here — that is a **fixed artifact of the `RGADESC` analysis-only descriptor**
(`.amdhsa_group_segment_fixed_size 65536`, occ_kernel_dsws_flow.s, unconditional under `RGADESC`, not tied
to `LDS_TOTAL_FLOW`) and reads 65536 for every build gated through this harness, per the same figure in
the prior pass's Gate 2 at the *old* 54784B config — not a signal about the real (13824B) allocation, which
is the `build_flow.sh`-reported `.lds` figure above.

**Gate 3 — host `occ_dispatch.cpp` compiles; `ldsBytesRaw` matches; guards hold.**
`./build.sh` completed (`OK -> ./occ_dispatch`), same 23 pre-existing `-Wformat` warnings as before (`%u`
vs `uint64_t`, unrelated), 0 errors — the new `static_assert`-adjacent code compiles clean.
By hand-tracing the new host formula against a matching real dispatch (`DSWS2_OVERLAP=1 DSWS2_FLOW=1
FLOW_POOL_N=1 DSWS2_ACC_N=3 SSWIN=32`, i.e. the runtime env mirroring the Gate-2 kernel build):
`ldsBytesRaw = kOpBase(512) + 0 (pool term zeroed by dsws2Overlap) + accBytes(3*4096=12288) = 12800`, then
`ssWin(32) > poolSlots(1)` adds `32*32=1024` -> **13824** — an exact match to the kernel's published
`LDS_TOTAL_FLOW` above. If the operator forgets to pass the new host env var, the pre-existing
"authoritative `.lds` override" (unchanged, occ_dispatch.cpp ~1924-1946) still catches and corrects it from
the bin's own published figure with a loud mismatch print — never silently under-allocates.
`kOpBase`'s `static_assert` (occ_dispatch.cpp:1906-1907) is untouched and still holds (kOpBase=512
unchanged). The `>1 WG/CU` occupancy guard (occ_dispatch.cpp ~2010-2024, Phase 3 territory) is untouched
and out of scope for Phase 1 (default dispatch never exceeds `pool<=64` / 1 WG/CU, so it is not exercised
by anything this pass changes).

## Scope discipline

Only `occ_kernel_dsws_flow.s` and `occ_dispatch.cpp` were edited this pass. `build_flow.sh` already had
`DSWS2_OVERLAP`/`OVERLAP` threaded through `mkflow()` from the prior (scaffold-only) pass — confirmed
unchanged by this session (`git diff` shows it dirty only from that earlier, already-present edit, not
from anything touched here). `occ_kernel_coop.s` was never opened. Nothing was staged
(`git diff --cached --stat` empty) or committed. No `./gpu_run.sh` / `./occ_dispatch` was ever invoked;
`test_oracle` (CPU-only fp8 arithmetic self-test, part of `build.sh`) is the only executable that ran.
POOL_N=1, SEGK=256, ACC_N unchanged in all builds; JDEPTH forced to 1 (transitively, via SELFSERVE's own
guard); one accumulator generation only — no overlap frontier (design §6/Phase 2) or 2 WG/CU launch
(§8 step 3) attempted.

## STOP items

None that block this pass. The one deviation from the task's literal 3-item list (converting `.Lflow_feed`
as a fourth site, and adding the `RB_PENDING`->`0` + `STAGE_HEAD`-walk companion to the grow-fail STAMP) is
documented above as a required completion of the design's own stated intent (§10 Q2), not an improvisation
past a dependency I couldn't resolve — flagged prominently per the task's instruction rather than buried.
The only genuinely unverified claim is empirical, not structural: grow-fail is 0% on every measured shape,
so the converted path above has never executed on silicon. That run is explicitly the parent's supervised
gate, not mine.
