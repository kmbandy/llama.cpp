# DSWS Phase B — Role-Conversion WIRING (flow kernel) — Continuation Plan

> **For the implementer (Codex gpt-5.6-terra):** This is a CONTINUATION of
> `PLAN_DSWS_PHASEB_CONVERSION.md` (Tasks 1–4 already landed) re-targeted to the
> live kernel `occ_kernel_dsws_flow.s`. Read that plan and
> `SPEC_DSWS_PHASEB_CONVERSION.md` first — they are the design authority for the
> arithmetic, the control law, and the safety analysis. This document does NOT
> restate the design; it records the current build state, the four things that
> changed since the original plan, and the remaining tasks. **You write the
> assembly.** This plan is deliberately code-free (kmbandy's standing rule:
> plans describe what/where/why + verification gates; Codex writes the code).

**Goal:** Make runtime {compute↔A-feed↔B-feed} role conversion actually fire on
the flow kernel, so starving compute waves convert to carriers — closing the
measured 56% RING_WAIT / 21% SS_WAIT idle gap — without orphaning output,
jamming a feed, or bricking.

**Why now (measured, 2026-07-22):** PHASEPROBE on the established config, fed,
WORK-EXACT, oracle-clean: RING_WAIT 56.0%, SS_WAIT 21.2%, WMMA 19.3%, GROW 0.2%,
SHRINK 1.4%, FLUSH 1.8%. The kernel is starved, not slow — ~77% of compute-wave
time is spent waiting to get work. Root cause (source-verified): roles split at
dispatch (`occ_kernel_dsws_flow.s:3308`), self-serve carriers compute privately
and mark the ring slot pre-completed (`:4525` "ring feed/compute never owns the
item"), and role conversions are 0 in every run. So the ~27 compute-role waves
(`FIRST_COMPUTE_WID=3`, NCOMPUTE = WAVES−3 = 27 at WAVES=30) are statically
pinned to a ring that self-serve empties by design. Phase B is the fix and it is
half-built.

---

## State of the build — VERIFIED by call-site census (2026-07-22)

Run this to reconfirm before starting; it is the definition of "what's left":

```
for m in occ_sample try_gate conv_apply conv_dec_floor reserve_try epoch_mark; do
  printf "%-14s invocations=%s\n" "$m" \
    "$(grep -nE "^\s+$m\b" occ_kernel_dsws_flow.s | grep -v '^\s*//' | grep -vE "\.macro $m" | wc -l)"
done
```

Current result and meaning:

| macro | invocations | status |
|---|---|---|
| `occ_sample` (sensor, def `:1843`) | 0 | **built, unwired** |
| `try_gate` (epoch ticket, def `:1992`) | 0 | **built, unwired** |
| `conv_apply` (the committer, def ~`:2100`) | 0 | **built, unwired** |
| `conv_dec_floor` (def ~`:2076`) | 1 | invoked *inside* `conv_apply` only |
| `reserve_try` (def `:2026`) | 1 | invoked *inside* `conv_apply` only |
| `epoch_mark` | 0 | DIAG helper, optional |

**Consequence proven:** `DSWS2_CONV=1` produces a bin byte-identical to
`DSWS2_CONV=0` on both the established config (`cac3ff7c…`) and at CFASSIGN=0
(`128500f7…`). The flag is inert because the actuators are defined and never
called. Turning on Phase B is NOT a flag flip — it is Task 5's wiring, ported to
the flow kernel. All the hard parts (the ticket CAS order, the reservation
envelope, the floor-guarded slot dec, the pre-grow OOR discipline) are already
written inside the macros; the missing work is the ~1 decision block + 1 commit
call + claimer handshake that invoke them at the right points.

---

## Four deltas from the original plan — the implementer MUST reconcile these

The original plan predates all four. Do not assume its label names or config.

1. **Kernel is `occ_kernel_dsws_flow.s`, not `occ_kernel_dsws.s`.** The original
   Task 5 references `.Lcompute/.Lafeed/.Lbfeed` terminal-bail paths and
   `.Lclaimer_wait_done`. The flow kernel's equivalents are different: the
   dispatch/role split is at `.Lflow_dispatch` (`:3308`), the compute ring check
   at `.Lflow_compute` (`:3314`), the feed/reservation path at `.Lflow_feed`
   (`:3922`) → `.Lflow_da_peek` (`:4071`) → `.Lflow_da_stamp` (`:4416`), and the
   universal bail/park site is `.Lflow_feedmt_sleep` (many `s_cbranch` into it,
   e.g. `:4067, :4111, :4159, :4213`). The role is read at runtime from
   `ROLE[wid]` (`:3292` `lds_get_r s35, s45 // role = ROLE[wid]`), which means a
   converted role already has a path to take effect — write the new role into
   `ROLE[wid]` and the next dispatch honors it. **Task B1 is to map the original
   plan's insertion points onto these flow-kernel labels, verified at source.**

2. **Config of record changed.** Original plan: `G=6 SEGK=64 FM=2 FN=4`. Current
   established config A1: `G=6 SEGK=256 FM=1 FN=4 ACC_N=3 POOL_N=1 WAVES=30
   SSWIN=32 CFASSIGN=1`. Bring-up config is decided in Task A, not assumed.

3. **CFASSIGN exists now and is the #1 open question.** CFASSIGN (counter-free
   assign) derives a wave's ASSIGN unit from a fixed wid→cohort map; it already
   forbids `BATCH>1` for the same reason (`:977` `.error "CFASSIGN requires
   BATCH=1"`). Role conversion changes a wave's role — and a compute wave needs a
   unit. There is **no guard** for CONV×CFASSIGN, so a naive `DSWS2_CONV=1
   CFASSIGN=1` build would assemble and could silently produce wrong C. **Task A
   resolves this against source and picks the bring-up config.** The safe,
   recommended default: **bring Phase B up at CFASSIGN=0**, matching the original
   plan's design intent (it had no CFASSIGN), and compare against the CFASSIGN=0
   baseline A0 (`128500f7314cafce`, mean 0.906), NOT A1. CFASSIGN+CONV
   coexistence is a deliberate follow-on, out of scope here.

4. **SELFSERVE carry-through exists now.** It is *why* conversion is valuable:
   carriers compute privately and bypass the ring (`:4525`), so the ring can no
   longer distribute to idle compute waves — conversion is how those waves stop
   being spectators. The original plan's snapshot/quiesce handshake (its Task 5
   Steps 3–4) must be reconciled with the flow kernel's existing quiesce
   machinery (`QUIESCE_GOFF` `:457`, the `gq_reset/gq_bump/gq_read` macros
   `:1863+`) rather than the old `.Lclaimer_wait_done` counters.

---

## Global constraints (inherit from the original plan; these additions supersede on conflict)

- A GPU brick is a **BUG**, never a tax. Hang / timeout / INCOMPLETE / WORK-INEXACT
  = FULL STOP, cleared only by kmbandy. Never auto-fire the next variant.
- **kmbandy greenlights every GPU dispatch individually.** All dispatch goes
  through `./gpu_run.sh` (single-run latch, real-disk log, journal capture).
- **Claim the R9700 via the board before any GPU work** (`board_check` then
  `board_claim`); one session per GPU; never claim on top of a holder.
- **No new `s_sendmsg_rtn` / store traffic in any coast or bail spin.** The
  2026-07-19 brick was a redundant `deadman_check` doubling RTC traffic in the
  self-serve coast. The conversion decision runs at the bail point, executed once
  per super-tile per wave — NOT in the inner coast spin. Confirm the insertion
  site's frequency against source before adding any bus/store op.
- **Pre-grow OOR window is the #1 brick risk** (SPEC §4). Every LDS/atomic temp
  read while a wave is lean-32, *before* its `s_alloc_vgpr` GROW completes, must
  be ≤ v15 / scalar. The macros already respect this; the WIRING must not read a
  >v15 source before the grow. This is the mandatory review focus before Task F.
- **`DSWS2_CONV=0` must stay byte-identical to baseline** at every step
  (`cac3ff7c…` at CFASSIGN=1, `128500f7…` at CFASSIGN=0). This is the regression
  guard — check it after every kernel edit.
- **Never modify `occ_kernel_coop.s`** (known-good reference). Never stage
  `docs/examples/router-fleet-main.ini` or any file outside this spike dir; the
  tree is shared with a live weight-pager session — flag before any `git diff`,
  stage only your own spike files by explicit path.
- **Commit only when kmbandy asks.** Otherwise leave the tree dirty and report.
- Shell is fish — use `bash script.sh` for `read`/word-splitting; `${=var}` for
  explicit zsh word-split; never rely on bash `$var` splitting.

---

## Tasks

### Task A — Resolve CFASSIGN×CONV and pick the bring-up config (offline; source audit + decision)

**Deliverable:** a written determination (append to this file) of (i) whether
role conversion can coexist with CFASSIGN's unit assignment, traced at source
through how a converted compute wave acquires its ASSIGN unit, and (ii) the
bring-up config. Recommended default if the interaction is unsafe or unclear:
CFASSIGN=0, baseline A0 `128500f7314cafce`. Add a `.error` guard for any
CONV×knob combination the audit proves unsafe (mirror the existing CFASSIGN×BATCH
guard at `:977`), so the incompatible build cannot silently run.

- Verify at source how `ROLE[wid]` (`:3292`) and the ASSIGN/unit path interact
  when a wave's role slot changes; state whether a just-converted compute wave
  lands in a valid unit cohort.
- Gate: the chosen bring-up config assembles at `DSWS2_CONV=0` to the expected
  baseline sha, and at `DSWS2_CONV=1` (still byte-identical until Task B wires
  anything).

### Task B — Wire the decision block at the flow bail point (offline; the crux, part 1)

**Deliverable:** under `.if DSWS2_CONV`, at the bail/park path a non-claimer wave
reaches when it fails to get work (map to `.Lflow_feedmt_sleep` and/or the coast
return — Task B1 fixes the exact site), a decision sequence that calls the
already-built macros: `occ_sample` → watermark test (`occ_X < CTRL_LOW` ⇒ starve
dir; `occ_X > CTRL_HIGH_{A,B}` ⇒ over-serve dir) → `try_gate dir, s_win`, storing
`s_win` and `dir` in private scalars (never LDS). Honor the `DSWS2_FORCE` hook so
a forced wid/dir/epoch bypasses the watermark for bring-up.

- **B1:** map the insertion point onto the flow-kernel labels and prove (source +
  the frequency argument) it is per-super-tile, not per-coast-iteration.
- All temps ≤ v15 / scalar (pre-grow OOR). No new bus/store op in a coast spin.
- Gate: assembles 0-spill at `DSWS2_CONV=1`; `DSWS2_CONV=0` still byte-identical.

### Task C — Wire the commit at the terminal bail, ordered before the quiesce bump (offline; the crux, part 2)

**Deliverable:** guarded by `s_win`, invoke `conv_apply src_slot, dst_slot,
delta, alloc_sz` with the correct slot pair and sign per `dir`
(compute→A/B-feed = shrink, `delta=-(NFV−VLEAN)`, `alloc_sz=32`; A/B-feed→compute
= grow, `delta=+(NFV−VLEAN)`, `alloc_sz=NFV`), then have the wave update
`ROLE[wid]` to the new role so the next `.Lflow_dispatch` honors it, then the
existing quiesce bump — in that strict order (`conv_apply` completes before the
quiesce counter increment, per SPEC §3.4).

- The macro already does the floor-guarded dec, reservation, dest inc, role-reg
  flip, and SCC-retry grow. Task C only supplies the correct arguments and the
  `ROLE[wid]` writeback + ordering.
- Gate: assembles 0-spill; `DSWS2_CONV=0` byte-identical; CPU control model
  (`test_dsws_ctrl_model.cpp` if present for the flow kernel) still passes.

### Task D — Claimer snapshot / quiesce reconciliation + role-slot init (offline)

**Deliverable:** ensure `NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT` (`:381–383`),
`GATE_OFF[4]` (`:384`), `VRESV_OFF` (`:385`), `SEGCNT_OFF` (`:386`) are
initialized by the claimer for the flow kernel, and that the epoch/snapshot
handshake the decision relies on is driven by the flow kernel's existing quiesce
machinery (`QUIESCE_GOFF`, `gq_*`) rather than the old plan's `.Lclaimer_wait_done`
counters. Under `.if DIAG`, keep the `N−1` cross-check that the snapshot sentinels
agree with the quiesce count (write a mismatch flag to a free `occ[]` slot).

- Gate: assembles 0-spill; `DSWS2_CONV=0` byte-identical.

### Task E — Offline gate battery (no GPU)

**Deliverable:** all green before any silicon.

- `DSWS2_CONV=0` sha == baseline (regression guard).
- `DSWS2_CONV=1` assembles 0-spill (RGA / `llvm-mc` spill check) across the
  bring-up mix(es).
- `DSWS2_CONV=1 DSWS2_FORCE=0` with watermarks set impossibly (so none can fire):
  prove the *inert* path is byte-identical to `DSWS2_CONV=0`, isolating "adding
  the wiring regressed nothing" from "conversion itself does something."
- The CONV×(incompatible-knob) `.error` from Task A fires.

### Task F — [SUPERVISED GPU] `DSWS2_FORCE` single-wave / single-epoch bring-up (the proof)

**Deliverable:** with FORENSICS=1, `DSWS2_FORCE=1 DSWS2_FORCE_WID=<one>
DSWS2_FORCE_DIR=<one> DSWS2_FORCE_EPOCH=1`, exactly one wave converts once,
watermarks bypassed. This is the reproducible proof that a role flip + regrow
does not brick and does not corrupt. ONE dispatch, known-good short shape, dense
stride=1 oracle, then STOP and report. Expected: WORK-EXACT, oracle bad=0, the
conversion-commit DIAG counter reads exactly 1, no dmesg amdgpu fault. Round-table
the pre-grow OOR window (Task C) before this dispatch — it is the #1 brick risk.

### Task G — [SUPERVISED GPU] static-inert re-baseline

**Deliverable:** `DSWS2_CONV=1` with watermarks set so no conversion fires,
fed run on the bring-up config. Prove the conversion machinery compiled-in but
dormant reproduces the Phase-A baseline (A0 if CFASSIGN=0) within noise
(≤ ~4% span spread, the measured instrument noise floor). Isolates "machinery
regressed the substrate" from "conversion helps." One greenlit dispatch.

### Task H — [SUPERVISED GPU] dynamic — watermarks live, measure the win

**Deliverable:** watermarks at the specced `CTRL_LOW/CTRL_HIGH_{A,B}` defaults,
fed run, PHASEPROBE on. Success = RING_WAIT falls materially from 56% and the
wave-count curve's slope moves toward positive (the agreed falsifier: a real fix
converts idle waves to work rather than removing them). WORK-EXACT + oracle clean
are preconditions, not results. Report the full phase breakdown; quote no TF from
a PHASEPROBE build. One greenlit dispatch; on any anomaly, full STOP + bisect.

---

## Handoff notes for Codex

- Design authority: `PLAN_DSWS_PHASEB_CONVERSION.md` + `SPEC_DSWS_PHASEB_CONVERSION.md`.
  This file only re-targets the crux (Task 5 of the original) to the flow kernel
  and adds Tasks A (CFASSIGN) + the flow-label mapping.
- The macros are done and safety-analyzed — do not rewrite them; invoke them.
- Verify every insertion point against source at file:line before editing; the
  label names in the original plan are for the OLD kernel and will not match.
- After each kernel edit, re-run the `DSWS2_CONV=0` byte-identity check. If it
  ever diverges, an edit leaked outside `.if DSWS2_CONV` — stop and fix before
  proceeding.
- Do not touch files outside this spike dir; the tree is shared with a live
  weight-pager session. Stage nothing without kmbandy; flag before any `git diff`.
