# DSWS S1 (MAD-305) — END-OF-DAY BRIEF, 2026-07-22

**READ THIS AND §0 FIRST. Do not re-derive anything below; it was verified against source this
session, at the cited file:line.** Prior authoritative docs still stand: baseline results =
`RESULTS_DSWS_BASELINE_2026-07-21.md`; yesterday's process retractions = `DSWS_BRIEF_2026-07-21_EOD.md`.

---

## §0 — ONE-LINE STATE

The day was a **measurement-integrity + diagnosis** day, not a throughput day. The PHASEPROBE
instrument was made trustworthy (and committed), the real profile was measured, and the bottleneck was
diagnosed and traced to a specific fix. **No throughput was gained and none was attempted.** Tree
clean except the two uncommitted guards/docs listed in §9. Card untouched. Latch clear.

**The single open item for the morning is one architecture decision (yours), stated in §6.**

---

## §1 — THE CONFIG (verify against this; do not assume)

Established baseline **A1**, unchanged and current:
`WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 VBUDGET=1536 JDEPTH=1 STAGGER=1 DECENTASN=1 SELFSERVE=1
BANKZERO=1 RBU=1 INITBAR=1 TERMFIX=1 BATCH=1 G=6 ACC_N=3 CFASSIGN=1`.
Kernel bin `occ_dsws2_w30_flow_gd.bin` sha256 prefix **`cac3ff7c2338e73f`** — **reproduced from source
this session, byte-identical**. A0 (CFASSIGN=0) = `128500f7314cafce`. Identify builds by COMMIT +
DEFSYMS, never a remembered sha.

---

## §2 — THE HEADLINE MEASUREMENT (trustworthy, committed)

PHASEPROBE on A1, **fed** (`DSWS2_TARGET_SECS=6`, 4 reps, spread 0.9–1.4%), WORK-EXACT
(`computed=760320`), oracle `bad=0`, `ffn_gate_up M2048`, 512-tile cap:

```
WORK_WAIT  77.5%   (split: RING_WAIT 56.0% + SS_WAIT 21.2%)
WMMA       18.9–19.3%
FLUSH       1.8%
SHRINK      1.4%
GROW        0.2%
C_STORE     0.1%
```

**THE KERNEL IS STARVED, NOT SLOW.** ~77% of compute-wave time is spent *getting work*; only ~19% is
compute. Every compute-path phase combined is ~22%. dyn-VGPR is **free** (grow 0.2 + shrink 1.4 =
1.6%), which **vindicates the 2026-07-04 reading** of 1.3–2.8% — the moat's runtime cost is not the
problem (it is also currently *inert*: door3/door4 = 0, budget never binds — consistent with kmbandy's
own 2026-07-16 finding that at G=6 the wall is FEED/STAGE, not budget).

**RING_WAIT (56%)** = compute waves stuck at `DRAIN>=STAGE` (`.Lflow_compute`, `:3320` → coast) —
nothing staged. **SS_WAIT (21%)** = self-serve reservation wait. The staging/consume path is the wall.

---

## §3 — THE INSTRUMENT FIXES (all committed in `0607b4e71`)

The PHASEPROBE profiler was **unaudited and I quoted it 3 times before auditing; 2 of the 3 quotes were
wrong.** All fixed and gated so `PHASEPROBE=0` stays byte-identical to `cac3ff7c`:

1. **Two inverted phase labels** (`occ_dispatch.cpp`): occ[64]/`FOLLOW_WAIT` is actually a stage-WAIT
   (stamped at `.Lflow_havestage` `:3314`); occ[65]/`STAGE_WAIT` is actually the C-STORE (`:3859`, the
   kernel comment already said so). Corrected; each label now cites its stamp line.
2. **The missing self-serve wait stamp** (`occ_kernel_dsws_flow.s:4416`, `.Lflow_da_stamp`):
   `phase_stamp` bills all time since the *previous* stamp, so the unstamped self-serve reservation
   wait was being **charged to GROW** (read 21.3%, actually 0.2%). Added the stamp; GROW dropped to
   0.2% and the 21 points moved to WORK_WAIT — the fix proving itself, TF unchanged.
3. **`PHSHIFT`** (default 8; used 10): the u32 occ slots accumulate across all waves and **wrap after
   22.4 ms at 1920 waves** — no fed run could be measured. Right-shift at emit (ratio-preserving)
   raises the ceiling to 22.9 s.
4. **`PHSPLIT`** (default 0): splits WORK_WAIT into RING_WAIT (s78) vs SS_WAIT (s79) by re-targeting one
   stamp. Zero new slots.

Other committed knobs from earlier in the day: **`CNTLEAN`** (trim `cnt_flush` to the 2 WORK-EXACT
counters), **`SPANFLIP`** (residency-window probe), the **WORK-EXACT CANNOT-EVALUATE** host verdict,
the **`pool=min(poolD,64,TOTAL)`** idle-workgroup fix + raised ceiling + co-residency guard, and the
harness `WORK_EXACT_CANNOT_EVALUATE` refusal. Full list in the commit message.

---

## §4 — THE DIAGNOSIS AND WHY SELF-SERVE ISN'T WORKING (verified at source)

- Roles split at dispatch (`:3308`): `ROLE_COMPUTE` → `.Lflow_compute` (ring consumer);
  everything else → `.Lflow_feed` (stager) → self-serve carry.
- A self-serve carrier **computes privately and marks the ring slot pre-completed** (`:4525`: *"ring
  feed/compute never owns the item"*). So self-serve **bypasses** the ring, it does not fill it.
- **Role conversions = 0** in every run → roles are **static**. A compute wave never becomes a carrier.
- **`FIRST_COMPUTE_WID=3`** → NCOMPUTE = WAVES−3 = **27 compute** waves, 3 feed. So ~27 waves are
  statically pinned to a ring that self-serve empties by design; only ~7 are effective stagers
  (`MSFEED` note `:3934`). That is the 56% RING_WAIT.

**Phase B was investigated and rejected as the vehicle** (this is important — do not restart it): the
Phase B machinery (`occ_sample`/`try_gate`/`conv_apply`/watermarks/snapshot-quiesce) is **transcribed
from the barrier-synchronized `occ_kernel_coop.s`** (`try_gate` comment says so). It is an orchestrated
epoch controller needing a per-super-tile follower rendezvous the flow kernel **deliberately does not
have** (SELFSERVE replaced it). `DSWS2_CONV=1` is byte-identical to `=0` at both CFASSIGN settings —
its actuators (`occ_sample`, `try_gate`, `conv_apply`) have **0 invocations**; `conv_apply` flips a
*private register*, not the LDS mailbox, so it wouldn't even work here. **Ignore/delete Phase B.**

---

## §5 — THE RIGHT FIX (kmbandy's architecture, confirmed against source)

**Decentralized role conversion via the ROLE mailbox = the adaptive second half of CFASSIGN.** No
controller, no epoch, no snapshot. A wave *is* whatever `ROLE[wid]` says, re-read every pass (`:3295`,
comment `:541`). A persistently-starved compute wave drops a note into its own `ROLE[wid]` slot to
become a feed/stager (compute→feed); next dispatch it stages. Direction that fixes the 56% is
**compute→feed**, and it **self-extinguishes** (more stagers → ring fills → remaining compute waves
un-starve → conversions stop). It is a **pure role flip, NO VGPR op** — waves are lean when coasting
(dyn-VGPR: fat only during the WMMA burst, `:2984`); the existing dyn-VGPR + STAGGER handle all fat
mechanics. The baton is a **pure nudge** (kmbandy's settled definition), not the serializer; STAGGER's
concurrent-fat discipline serializes the *grow-back* (feed→compute) direction.

**What is ALREADY BUILT (do not rebuild):** the mailbox (`ROLE[wid]`, `:3295`); the per-pass check
(`:3312`); the starvation signal (`.Lflow_coast` `:4870`, `CNT_COAST`); `convCount` counter
(`CONVCNT_OFF`/occ[48], `:480`, labeled *"proves waves switch role"*); the `CONV_COOLDOWN` defsym
(`:406`).

**Spec written:** `SPEC_ROLE_CONVERSION_MAILBOX_2026-07-22.md` (prose-only). Three pieces: (1)
note-drop at the coast spot, (2) census bump, (3) the CFASSIGN cohort question. **BUT the spec's Piece
3 premise is WRONG — see §6.**

---

## §6 — ⭐ THE ONE OPEN ITEM: an architecture decision, yours ⭐

Codex terra took the spec, correctly **stopped at Piece 3 before writing anything** (byte-identity
green, nothing landed), and its source trace **disproved my Piece 3 premise**:

- I specified Piece 3 as *"make the CFASSIGN cohort membership-aware so a converted wave is
  included/excluded."* **The cohort is NOT role-based.** It assigns units by `r = cohort_start + wid`
  — **wid-positional, role-blind** (`:4085–4114`; `s15` served-token is an absolute cohort-end token,
  not membership-keyed). Conversion doesn't change a wave's wid, so **there is no "membership" to
  re-key.** I mis-framed the crux.

- **The real question** (source does not answer it — it is a design decision): the cohort's advance
  gate holds *"DRAIN cannot cross until every positionally-eligible wid has published its unique
  generation"* (`:4082`). **When a compute wave converts to feed, does it still publish that
  generation (gate advances, conversion is transparent) or stop (DRAIN stalls forever waiting on a wid
  that is now feeding)?** The source does not define what a converted wave does with respect to the
  cohort publish. Answering it — what should determine an eligible vs ineligible wid once roles are
  dynamic — is the architecture call. Codex correctly refused to invent it. This is the single thing
  between here and a buildable conversion.

**Also found (corrects §5's "telemetry is ready"):** the census slots `NCOMP/NAFEED/NBFEED_SLOT`
(`:381–383`) have **ZERO runtime writers** — read by telemetry (`:2850`), never written. The role
census is dead. Piece 2 can't be honest until those slots' init + ownership are established.
(`convCount` at occ[48] is a separate counter and may be live — not verified.)

---

## §7 — CORRECTIONS I MADE TODAY (things I told you that were wrong, now fixed)

Bank these so they don't re-inject as fact:
1. PHASEPROBE labels inverted (§3.1) — "STAGE_WAIT is 0.1%, feed path is free" was **backwards**; it's
   56%.
2. GROW "21.3%" was reservation-wait mis-billed (§3.2); real GROW is 0.2%. dyn-VGPR is free.
3. "3.0 µs/wave is a measurement artifact" (decision `4cfb6b62`) — **RETRACTED** by `180fb2db`: it's
   real and general (Arm A/B on a 3168-tile shape, 3.59 µs/added-wave).
4. "Phase B just needs wiring" → it doesn't fit the flow kernel at all (§4).
5. "CFASSIGN and conversion are architecturally opposed" → wrong; they're complementary (initial +
   adapt). CFASSIGN is the static front-half; conversion is the missing adaptive half.
6. "compute→feed is a shrink" → wrong; waves are lean by default, conversion is a pure role flip, no
   VGPR op (you corrected me; dyn-VGPR).
7. "CFASSIGN cohort needs a membership re-key" → wrong; cohort is role-blind (§6).

**Pattern to hold:** the *measurement* (77% wait, starved-not-slow) is rock-solid. My *mechanism
framing* needed source correction repeatedly. Hold the framing loosely; verify at file:line before
asserting "the real work is X."

---

## §8 — DO NOT RE-DERIVE / RE-DIAGNOSE

- The 77.5% profile and its split — measured, committed. Fed, WORK-EXACT, 0.9% spread.
- Phase B does not fit the flow kernel (§4). Do not restart it.
- CFASSIGN intent = remove the shared ASSIGN cursor CAS via a fixed wid→cohort map (testing-log 1679).
  It is the *static* half of an initial-plus-adapt design; there is **no CFASSIGN design doc**.
- The baton is a pure nudge, not a cap (kmbandy, settled). Do not try to make it a serializer.
- STAGGER is inert at G=6 (budget doesn't bind); the wall is staging.
- dyn-VGPR is free (1.6%), and currently inert (door3/door4=0).

---

## §9 — STATE OF THE TREE

- **HEAD `0607b4e71`** on `feat/laguna-arch` (a SHARED branch — a live weight-pager session commits
  here too; only touch the spike dir, never stage their files or `router-fleet-main.ini`).
  **UNPUSHED.**
- Committed today (`0607b4e71`, 21 files, spike-only): the instrument audit — `CNTLEAN`, `SPANFLIP`,
  `PHSHIFT`, `PHSPLIT`, cannot-evaluate gate, pool clamp + co-residency guard, harness refusal, the
  corrected labels + missing stamp, the Codex per-wave artifact, the baseline results + retraction
  docs.
- **UNCOMMITTED (2 items, your call whether to commit):**
  - `occ_kernel_dsws_flow.s` — the `CFASSIGN && DSWS2_CONV` `.error` guard at `:980` (from the first
    Codex/Phase-B run). Byte-identity of `cac3ff7c` is intact (guard only fires when both set).
  - New docs: `SPEC_ROLE_CONVERSION_MAILBOX_2026-07-22.md`, `PLAN_PHASEB_WIRING_FLOW_2026-07-22.md`
    (the Phase-B continuation plan — now superseded by the mailbox spec; keep for the reasoning trail),
    `CODEX_PHASEB_PROGRESS.md`, `CODEX_RCONV_PROGRESS.md` (Codex's eligibility-contract trace — read it
    in the morning), this brief.
- KG: **eleven `mneme_code_decide` decisions** filed to `repo__kmbandy__llama.cpp` today (config
  `0f594c8c`; instrument fixes `e8ceb6b7`/`58d574a4`/`0ebfcb89`/`967170f4`/`dc5ce3d5`/`0d6f2df4`;
  idle-WG `4cfb6b62` + retraction `180fb2db`; 2-WG/CU `e8edadc2`; ASSIGN-bound `9ccbf559`). Test log
  entries closed for each GPU run.
- **Card:** free/held-by-weight-pager (not us). **Latch clear. No board claim held by us.**

---

## §10 — NEXT ACTIONS (morning)

1. **The one decision (§6):** what should determine cohort eligibility once roles are dynamic — does a
   converted wave still publish its generation, or is it removed from the advance-gate's expected set?
   Read `CODEX_RCONV_PROGRESS.md` first (its exact contract trace). Everything else is downstream.
2. Once decided: the build is small — note-drop (one LDS store at `.Lflow_coast`, threshold-gated) +
   census-slot init & bump + the eligibility rule from (1). New defsym `DSWS2_RCONV` (default 0,
   byte-identical, requires CFASSIGN=1, mutually exclusive with DSWS2_CONV). Codex terra thread
   `a5b77e336fd22588b` can resume with the eligibility rule supplied.
3. GPU bring-up (ours, gated): single-wave first (threshold fires once) with FORENSICS=1 so `convCount`
   proves exactly one conversion; dense stride=1 oracle; WORK-EXACT; then the dynamic run measuring
   whether RING_WAIT falls from 56%.
4. Offline first, always: `DSWS2_RCONV=0` byte-identical to `cac3ff7c` after every edit.

---

## §11 — PROCESS NOTES

- Board discipline held all day: `board_check` before every claim, released promptly, queued behind
  the weight-pager session (never claimed on top). One false "changed hands" was my own monitor-script
  bug (matched claim-id against the holder field), not a real event.
- Two Codex terra handoffs, both offline-scoped (no GPU), both correctly hit stop-and-report: the
  Phase-B run stopped at the missing follower-rendezvous; the mailbox run stopped at the cohort
  eligibility contract. **The stop-and-report discipline caught flaws in MY specs twice, before any
  code.** That is the mechanism working.
- `DSWS2_RCONV`/conversion is the CFASSIGN adaptive half; frame it that way, not as "Phase B."
