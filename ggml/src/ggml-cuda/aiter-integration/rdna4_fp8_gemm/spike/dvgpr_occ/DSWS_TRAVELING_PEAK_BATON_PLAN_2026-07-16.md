# Stagger + Baton — Implementation Plan (DEFINITIVE, 2026-07-16 evening)

> Implements `DSWS_TRAVELING_PEAK_BATON_2026-07-16.md` (definitive spec). Build the **baton first** (§4 of the
> spec), then stagger. Each build task's test cycle is the OFFLINE GATE. Silicon tasks are ONE greenlit
> `./gpu_run.sh` each — STOP and report after each; kmbandy greenlights every dispatch. Supersedes the earlier
> dam-based plans.

**Goal:** replace the non-deterministic "race to grow when budget frees" with a **deterministic directed
grow-turn hand-off** (the baton): when a compute wave starts shrinking it PUSHES "grow now" into the next ready
wave's mailbox; that wave reads its OWN mailbox (non-blocking, like ROLE) and grows into the freed registers.
Exactly one grows per shrink → the peak travels, no wave waits, no cap, no cross-wave poll.

## Global Constraints (verbatim, every task — these are the checks that would have caught today)
- **RIVER-SAFETY (the thing I violated all day):** NO blocking read (a wave reads only its OWN mailbox), NO
  artificial cap (physical `s_alloc_vgpr` grow-fail is the only throttle), NO wait/spin, NO hard partition.
  Grep the compute hot loop after every edit: any new poll of a shared word, any `s_cmp` against a software
  count that gates progress, any spin = STOP, it's a dam.
- `STAGGER=0` MUST stay byte-identical `386dc28643ffb58568623ad6d89cfe62`. All new code `.if STAGGER`-gated.
- A refused/permit-less wave COASTS to FEED and retries next pass — never stalls.
- `DEADMAN_TICKS` 0.5s (never raised). Work-exactness `computed == TOTAL_super*ACC_N` every silicon run.
- ONE greenlit dispatch at a time; changed kernel = ONE bring-up then STOP; hang/DMFAT/BAD = full stop. Fed
  ONLY via deep-K, guard ON. No TF verdict from <~1s.
- **REPEAT every bring-up 2–3× (each greenlit).** The bootstrap race hid behind single "clean" runs all day;
  a config is only "clean" if it's clean on repeats.
- Line numbers approximate — grep the named label/macro and confirm against CURRENT (post-cleanup) code before
  editing. Representative asm below follows the PROVEN ROLE-mailbox pattern; confirm exact regs/offsets on read.

---

### Task 0 (offline, no code): confirm the substrate the baton sits on
- [ ] Grep the CURRENT compute path (`.Lflow_compute` → grow → claim → `.Lflow_bshrink`) and confirm what
  remains of the token layer after the cleanup: `fat_acquire`, `FATTOK_OFF`, `FATCAP_EFF`, `RELSTART` sites.
  The baton REPLACES the `fat_acquire` token gate; `RELSTART` (release-at-shrink-START) is kept but becomes
  the *physical* half only. Write down the exact grow site (`s_alloc_vgpr NFV`), the shrink site
  (`.Lflow_bshrink` / `fat_dec`), the claim site (`lds_fetch_add SL_RBNEXT`), and the ROLE mailbox
  (`ROLE_BASE`, `lds_put (ROLE_BASE + w*4)`) — the baton mirrors ROLE exactly.
- [ ] Confirm a free per-WG LDS region for the permit mailbox (needs `WAVES*4` bytes, like ROLE) and a free
  SGPR for the round-robin target index. (ROLE_BASE region math is at ~471; SLOTC/OP/ACC bases follow it.)

---

### Task 1: The grow-permit mailbox (LDS) + init + BOOTSTRAP SEED
**Files:** `occ_kernel_dsws_flow.s` — new `GROWPERMIT_BASE` (LDS, `WAVES*4`, mirrors `ROLE_BASE`); init loop
(~2454, next to the ROLE init); the coordinator init.

**Interfaces:** Produces `GROWPERMIT_BASE` — per-wave u32 mailbox: `1` = "you hold the grow-turn, grow now";
`0` = "no turn, keep feeding". Consumed by Task 3 (read own) and Task 2 (push to next).

- [ ] **Step 1:** define `GROWPERMIT_BASE` after the ROLE mailbox (so it doesn't alias — there's a guard at
  ~577 that errors if a control word overruns ROLE; place after SLOTC or extend carefully, CONFIRM no alias).
- [ ] **Step 2:** init all permits to 0 in the coordinator's barrier-free init loop (next to `lds_put
  (ROLE_BASE + w*4), ...`), STAGGER-gated.
- [ ] **Step 3 (THE SEED — without it nothing ever grows):** the baton chain is self-perpetuating (a shrink
  hands the turn onward) but needs a STARTING turn. Seed the first compute wid's permit = 1 at init (or seed
  `K` of them to fill the initial peak; start with `K=1`, the minimal chain — one wave grows → computes →
  shrinks → passes the turn → the chain runs). CONFIRM which wid is the first compute wave in the emergent
  role init and seed that one. *(If a run deadlocks with `door` all-coast and no grows, a missing/lost seed is
  the first suspect.)*
- [ ] **Step 4 OFFLINE GATE:** inert `386dc28`; STAGGER=1 assembles 0-spill; disasm shows the permit region
  init writes + the one seed.

---

### Task 2: Push the baton at shrink-START (directed, next-available, O(1))
**Files:** `occ_kernel_dsws_flow.s` — the `.Lflow_bshrink` block (where `RELSTART` `fat_release` fires today).

**Interfaces:** Consumes `GROWPERMIT_BASE`, a shared round-robin index (new SGPR/LDS word `BATON_NEXT`).
Produces: exactly one `GROWPERMIT[next]=1` write per shrink-start → the next wave's grow-turn.

- [ ] **Step 1:** at shrink-START (before/with `RELSTART`'s release, since the physical registers free at the
  shrink), compute the next-available target = round-robin over the COMPUTE wids. Simplest O(1): a per-WG LDS
  counter `BATON_NEXT`, `lds_fetch_add(BATON_NEXT, 1)`, map to a compute wid (`first_compute + (idx mod
  n_compute)`), write `GROWPERMIT[wid] = 1`. NO inspection of the target's state (river-safety) — if the
  target is already fat or busy, it consumes/ignores the permit on its next read (Task 3 handles idempotently)
  and the chain self-heals via the next shrink. (This is the "next-available, not predictive" decision from
  spec §1.2 — do NOT read candidates' phases.)
- [ ] **Step 2:** this write is a plain `lds_put` (lane-0, no ACC live at shrink — SAFE, mirrors ROLE
  writes). NOT a store during `s_alloc_vgpr` (that corrupts the register file — CLAUDE.md r5); do it BEFORE the
  `.Lflow_bshrink` `s_alloc_vgpr 32` spin, same as `RELSTART`'s release.
- [ ] **Step 3 OFFLINE GATE:** inert `386dc28`; assembles 0-spill; disasm: the permit write precedes the
  shrink `s_alloc_vgpr`; grep confirms NO poll/spin added.

---

### Task 3: Compute path reads its OWN permit to take the grow-turn (replaces the token race)
**Files:** `occ_kernel_dsws_flow.s` — `.Lflow_compute` dispatch / the grow gate (where `fat_acquire` is today).

**Interfaces:** Consumes `GROWPERMIT[wid]`. Produces: a compute wave grows+claims ONLY when it holds the
permit; otherwise coasts to FEED. Replaces the `fat_acquire`/`FATTOK` token gate entirely.

- [ ] **Step 1:** at the compute grow-decision, read `GROWPERMIT[my wid]` (own mailbox, non-blocking):
  - permit == 1 → **clear it** (`lds_put GROWPERMIT[wid], 0`, so one permit = one grow-turn, idempotent),
    then `s_alloc_vgpr NFV` grow, claim a rowblk, compute the burst. (If the physical grow-fails — budget
    genuinely full despite the hand-off — coast; the permit is spent, the next shrink re-seeds a turn.)
  - permit == 0 → coast to FEED (productive) and retry next pass. NO wait.
- [ ] **Step 2:** REMOVE the `fat_acquire`/`FATTOK`/`FATCAP_EFF` token machinery (it is the redundant dam the
  permit replaces). Keep `RELSTART` (physical release-at-shrink-start). Confirm the `.Lfa_*` labels and the
  `FATTOK` pool word are fully removed; the cap symbols (`MAXFAT`/`PEAK_CONC_EFF`/`FATCAP_EFF`) can retire too.
- [ ] **Step 3 OFFLINE GATE:** inert `386dc28`; assembles 0-spill; grep the compute hot loop — the ONLY read
  that gates the grow is `GROWPERMIT[own wid]` (no `FATTOK` poll, no cap compare, no spin). Enumerate the chain
  in writing: init→seed (T1) → own-read+clear+grow (T3) → shrink+push-next (T2) → next wave's own-read.

---

### Task 4: [SUPERVISED GPU] correctness bring-up — REPEATED (deterministic, non-binding G)
**Prereqs:** T1–T3 offline-green; latch clear (human).
- [ ] **Step 1:** build the baton bin (J=2 moderate, G=6 non-binding — this is correctness, NOT a TF verdict):
  `DECENTASN=0 FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=2 MSDRAIN=1 STAGGER=1 RBU=1 STAGINSTR=1 TFPROBE=1 ./build_flow.sh`
- [ ] **Step 2:** present for greenlight, run fed deep-K guard ON (the standard baton_* invocation). Expected
  `computed==9437184`, oracle CLEAN, `claim` advances (not stuck at 64), `occ[0]=0`, no DMFAT/reset.
- [ ] **Step 3:** **REPEAT the SAME bin 2 more times (each greenlit).** PASS only if all 3 are clean +
  work-exact. Any deadlock on any repeat = the bootstrap chain is still racy (suspect the seed / a lost
  permit) → STOP, diagnose offline. This repeat is the check that today's single-run "clean"s skipped.

---

### Task 5: [SUPERVISED GPU] binding-G measurement (the first real verdict)
**Prereqs:** T4 clean on repeats. Only now does the design's value become measurable (spec §3).
- [ ] **Step 1 (offline):** find a geometry where VGPR BINDS (grow-fail > 0) and LDS fits — WOFLUSH=1 (no bank
  LDS) opens ACC_N/POOL_N room; target the 2026-07-13 regime that hit grow-fail=1588. Assemble 0-spill.
- [ ] **Step 2:** greenlit fed run(s), one at a time. Confirm grow-fail > 0 (budget is the wall). Measure WMMA
  duty + TF vs the best STAGGER=0 baseline at matched fed conditions. VERDICT: does the traveling peak keep
  compute continuous (higher effective occupancy → higher TF) where the budget binds?

---

### Task 6 (separate, after the baton is proven): STAGGER — lean-launch metering
Scope against what already exists (the resident-wave model may already cover most of it). Stagger's only job:
launch lean waves one-at-a-time while ≥32 VGPR free and below max waves. Its own spec §1.1 + plan, built and
gated the same way. Do NOT entangle it with the baton or the budget.

---

## Self-review notes
- Spec coverage: §1.2 baton (push/read-own/next-available) → T1–T3; §0 river-safety → global constraints +
  every gate's hot-loop grep; §3 binding-G → T5; §1.1 stagger → T6.
- The two failure-catchers today lacked: (a) the hot-loop "no new blocking-read/cap/wait" grep is now a gate on
  every task; (b) the REPEATED bring-up (T4 S3) is now mandatory — single clean runs hid the race.
- KEY RISK called out: the BOOTSTRAP SEED (T1 S3). The chain is self-perpetuating but needs a starting turn; a
  missing/lost seed presents exactly as today's all-coast deadlock. First thing to check on any T4 deadlock.
- Open before build (T0): confirm the free LDS region for `GROWPERMIT_BASE` (no ROLE-mailbox alias, guard
  ~577) and the first-compute-wid to seed; confirm removing `FATTOK`/`fat_acquire` leaves the grow site intact.
