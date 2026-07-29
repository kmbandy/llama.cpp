# SPEC — DSWS runtime role conversion via the ROLE mailbox (the CFASSIGN adaptive half)

**Status:** design, ready to build. **Builder:** Codex gpt-5.6-terra. **Author:** claude__main, 2026-07-22.
**Kernel:** `occ_kernel_dsws_flow.s`. **Config:** the established A1 (`G=6 ACC_N=3 SEGK=256 POOL_N=1
WAVES=30 SSWIN=32 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1`).

> **No code in this spec by design.** It gives the mechanism, the exact insertion points at
> file:line, the invariants, and the gates. You write the assembly. If source contradicts anything
> here, **STOP and report** — do not improvise around it.

---

## 1. What this is, in one paragraph

A DSWS wave *is* whatever role its `ROLE[wid]` LDS mailbox says, re-read every dispatch pass
(`occ_kernel_dsws_flow.s:3295`, comment at `:541`: *"wave reads its ROLE[wid] mailbox each cycle and
simply IS that role"*). Today that mailbox is written in exactly two places — init (`:3032–3038`) and
terminal retire (`:3286`). **There is no runtime writer that converts a wave.** This spec adds it: a
persistently-starved compute wave drops a note into its own `ROLE[wid]` slot to become a feed
(stager) wave, and the CFASSIGN cohort accounting is extended to honor the changed membership. That
extension is the adaptive half CFASSIGN shipped without.

This is **not** Phase B. Do not use `conv_apply`, `occ_sample`, `try_gate`, the watermark thresholds,
or the snapshot/quiesce handshake — that machinery was transcribed from the barrier-synchronized
`occ_kernel_coop.s` and does not fit this decentralized kernel (it flips a *private* register, not the
LDS mailbox; `:2100+`, 0 invocations). Ignore it entirely; a later cleanup can delete it.

## 2. Why (measured, 2026-07-22)

PHASEPROBE on A1, fed, WORK-EXACT, oracle-clean: **RING_WAIT 56.0%**, SS_WAIT 21.2%, WMMA 19.3%, GROW
0.2%, SHRINK 1.4%, FLUSH 1.8%. The kernel is **starved, not slow** — ~27 compute waves
(`FIRST_COMPUTE_WID=3`, so NCOMPUTE = WAVES−3 = 27) contend for work that ~7 effective stagers can't
produce fast enough (`MSFEED` note at `:3934`). The manual proof this rebalancing works already
exists: dropping the launch to 5 waves gave **4.3×** on `ffn_gate_up M2048`. Conversion makes that
rebalancing *dynamic and per-wave* instead of a static launch count.

## 3. What is ALREADY built — do NOT rebuild any of this

- **The mailbox.** `ROLE[wid]` at `ROLE_BASE`, read at dispatch (`:3295`), acted on at `:3312`
  (`ROLE_COMPUTE` → `.Lflow_compute`, else `.Lflow_feed`).
- **The per-pass check.** Every loop, the wave re-reads its role and *is* that role. A changed
  mailbox slot takes effect the very next pass with zero extra plumbing.
- **The starvation signal.** A compute wave that finds `DRAIN >= STAGE` (nothing staged) falls to
  `.Lflow_coast` (`~:4868`) and bumps `CNT_COAST`. That is the wave knowing, locally, that it is
  starved — every pass it can't get work.
- **The role→fat model (dyn-VGPR).** All waves start lean-32 (`:2984`); a compute wave grows to fat
  *only for its WMMA burst* and shrinks straight back. So a coasting/starved compute wave is **already
  lean**. Conversion is therefore a **pure role flip with NO VGPR operation** — the existing dyn-VGPR
  machinery does all grow/shrink in the normal role paths, and STAGGER's traveling-peak baton
  serializes those grows. Conversion changes the *headcount*; the moat handles the *peaks*. Do not add
  any `s_alloc_vgpr` to the conversion path.
- **The telemetry.** The FORENSICS role-census stream (`:2844–2858`) already reads the live
  `NCOMP/NAFEED/NBFEED` slots AND a cumulative `convCount` at `CONVCNT_OFF`. The instrumentation to
  *watch conversions happen over time* is already wired.

## 4. What this spec builds — three pieces

### Piece 1 — the note-drop (the physical conversion) — small

At `.Lflow_coast` (`~:4868`), where the wave already knows it is starved: gate on a **persistence
threshold** (a private per-wave consecutive-coast counter crossing N — increment on coast, reset the
moment the wave computes), and on success write the new role into the wave's own mailbox slot
(`ROLE_BASE + wid*4`) and bump `convCount` at `CONVCNT_OFF`. Next dispatch, the wave is the new role.
That LDS store *is* the physical conversion. Nothing else actuates it.

- **Direction (primary):** `compute → feed`. The starved consumer becomes a producer/stager. This is
  the fix for the measured 56%.
- **A vs B:** convert to whichever feed side is currently shorter — readable from the `NAFEED_SLOT` /
  `NBFEED_SLOT` counts (`:2850–2854`). Keep the choice a one-comparison heuristic; do not build a
  scheduler.
- **Reverse direction (`feed → compute`), secondary:** include it for balance — a feed wave that
  finds staging over-serving (its stage attempts repeatedly find `STAGE >= ASSIGN`, nothing to stage)
  converts back to compute. This is where the resulting *grows* are serialized by STAGGER +
  `reserve_try` — i.e. the reverse direction rides the existing moat exactly as designed; the note-drop
  itself is still just a mailbox write.

### Piece 2 — the census bump — trivial

On any conversion, atomically adjust the role-count slots (`NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT`,
`:381–383`): decrement the source role, increment the destination. **These are telemetry-only** — the
only runtime reader is the FORENSICS census stream (`:2850–2854`); no compute path consumes them, so a
stale count cannot corrupt C. Bump them so the census stays honest and `convCount` reconciles.
Floor-guard the decrement so the source role never drops below 1 (the `conv_dec_floor` *pattern* at
`:2076` is the right shape — a CAS-with-floor — but you may inline a simpler version; do not pull in
the rest of `conv_apply`).

### Piece 3 — the CFASSIGN cohort eligible-wid re-key — THE design work

This is the only genuinely unbuilt piece and the reason CFASSIGN and conversion have been
"incompatible." The CFASSIGN cohort at `:4082+` is a WAVES-wide partition whose advance gate is
*"DRAIN cannot cross this cohort until every **eligible wid** has published its unique generation."*
That gate **assumes a static set of eligible wids.** When a wave converts, the eligible set changes,
and the gate either waits forever for a wid that converted away, or fails to account for one that
converted in.

The build: make the cohort's eligible-wid set **membership-aware** so a converted wave is correctly
included/excluded in the "every eligible wid published" condition. Trace the cohort math at `:4082`
and the served-cohort token `s15` at `:4104` before touching anything, and state in
`CODEX_PHASEB_PROGRESS.md` exactly how eligibility is currently derived (from `wid` vs from role vs
from the count slots) — because *that* determines the minimal correct re-key. **This is the piece most
likely to hide a correctness trap; if the honest minimal re-key is not obvious from source, STOP and
report the cohort's exact eligibility contract rather than guessing.**

## 5. Damping / stability (why this converges instead of oscillating)

Decentralized, no controller, three overlapping brakes:
1. **Self-extinguishing feedback (primary).** compute→feed adds stagers → the ring fills → the
   remaining compute waves stop hitting `DRAIN>=STAGE` → they stop converting. The trigger cures
   itself.
2. **Persistence threshold** (Piece 1): convert after N consecutive coasts, not the first — filters
   transient empties.
3. **Cooldown**: a converted wave cannot re-convert for a short window. The `CONV_COOLDOWN` defsym is
   already scaffolded (`:406`); wire it. Floor guard (Piece 2) is the hard stop that keeps ≥1 wave in
   each role.

## 6. Flag, and the byte-identity contract

Add a defsym — suggested `DSWS2_RCONV` (runtime role conversion), default 0. **`DSWS2_RCONV=0` MUST
assemble byte-identical to the current baseline** (`cac3ff7c2338e73f` at CFASSIGN=1) — re-check the
sha after every edit; if it diverges, an edit leaked outside `.if DSWS2_RCONV`. This flag **requires
CFASSIGN=1** (it *is* CFASSIGN's second half) — add a `.error` guard for `DSWS2_RCONV && !CFASSIGN`,
mirroring the existing guards at `:978`. Note this is the opposite polarity of the dead `DSWS2_CONV`
(which is guarded *against* CFASSIGN); the two must not both be set — guard that too.

## 7. Safety (non-negotiable — this kernel has bricked the box before)

- **The coast path is a Rule-5 hot spin.** The 2026-07-19 brick was added message-bus traffic in the
  self-serve coast. Piece 1 adds **one LDS store on the conversion event only** (gated by the
  threshold, so rare), never per-coast-iteration and never a `s_sendmsg_rtn` / global store. Confirm
  the store is on the taken-conversion branch, not the coast loop body.
- **No `s_alloc_vgpr` in the conversion path** (§3, dyn-VGPR reason). A role flip must not grow/shrink;
  the role paths already do.
- **No new barrier.** This kernel deadlocks on `s_barrier` under dyn-VGPR. The conversion is a
  decentralized mailbox write; there is no rendezvous and must not be one.
- Never modify `occ_kernel_coop.s`. Only touch files in this spike dir — the git tree is shared with a
  live weight-pager session; stage nothing, flag before any `git diff`.

## 8. Verification gates

**Offline (all green before any silicon):**
- `DSWS2_RCONV=0` sha == `cac3ff7c2338e73f` (regression guard, after every edit).
- `DSWS2_RCONV=1 CFASSIGN=1` assembles 0-spill (RGA / spill check).
- The `DSWS2_RCONV && !CFASSIGN` and `DSWS2_RCONV && DSWS2_CONV` guards `.error` correctly.
- CPU control model (`test_dsws_ctrl_model.cpp`) still passes if any shared logic is touched.

**GPU (human-run, NOT in your scope — leave these for kmbandy):** single-wave bring-up first (a forced
one-wave conversion via the threshold set to fire once), FORENSICS=1 so `convCount` and the census
stream prove exactly one conversion happened, dense stride=1 oracle, WORK-EXACT — then the dynamic
run. **You do not dispatch to the GPU.**

## 9. Open questions you resolve against source (state answers in the progress file)

1. The cohort's exact **eligibility contract** at `:4082` — is "eligible wid" derived from `wid`
   range, from `ROLE[wid]`, or from the count slots? (Determines the Piece-3 re-key.)
2. Whether a converted wave's **served-cohort token `s15`** (`:4104`) needs resetting so it can serve
   in its new role's cohort.
3. The reverse-direction (`feed→compute`) trigger: the exact "staging over-serves" condition on the
   feed path (`.Lflow_feed`, `:3922`).

## 10. STOP-AND-REPORT

Halt and report — do not improvise — if: the cohort eligibility contract (§9.1) has no obvious
membership-aware minimal re-key; the byte-identity check fails and the cause isn't an obvious
`.if DSWS2_RCONV` leak; or any insertion point in this spec does not match the live source at the
cited line. This kernel punishes guessing with bricks; a stop-and-report is a success, not a failure.
