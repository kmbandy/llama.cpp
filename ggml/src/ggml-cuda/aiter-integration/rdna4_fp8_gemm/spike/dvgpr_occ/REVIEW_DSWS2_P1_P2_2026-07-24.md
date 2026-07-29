# ADVERSARIAL REVIEW — DSWS2 P1/P2a/P2b (2026-07-24)

Two INDEPENDENT reviewers (fresh Claude agents, deliberately denied the design docs, the builder progress
docs, and all of claude__main's conclusions — only the source, the defsym names, the build profile, and
domain facts). Different lenses: **R1 = liveness/progress/work-conservation**, **R2 = cross-wave ordering/races**.
Convergent findings are therefore corroboration, not coherence.

**VERDICT: DO NOT DISPATCH. The P1+P2a+P2b stack is not fit to run.** The defsyms all default 0 and are
byte-identical to `cac3ff7c` when off, so the tree is safe — but no ON build should reach the GPU.

---

## FALSE POSITIVE (record it so it doesn't propagate)

**R2-F2 "the profile does not assemble at SEGK=256" is WRONG** — caused by claude__main omitting `FM=1`
from the review brief, so R2 tested the `FM=2` default. At `FM=2`, `ARES_BYTES = G*16*FM*SEGK = 49152`
and the total overflows 65536. At our actual `FM=1`, `ARES_BYTES = 24576`, total 40960 — fine. Verified:
the full ON profile builds (`.text` 34260B, LDS 13824B) and the baseline rebuilds to `cac3ff7c`.
LESSON: a review brief must carry the COMPLETE build profile; a missing defsym produces phantom blockers.

## CONVERGENT FINDINGS (both reviewers, independently)

1. **The ring-compute path is UNREACHABLE under `DSWS2_OVERLAP`.** (R1-F5, R2-F3)
   Every slot that can exist carries `RB_PENDING` (init/stamp) or `ACC_N` (sentinels), and `RB_PENDING`'s
   only clearer (`side_final`, reached solely via `ASTAGE_R`/`BSTAGE_R`) is unreachable because both
   `.Lflow_feed` and `.Lflow_coast` branch away under OVERLAP. So the claim gates reject every slot and the
   whole `.if DSWS2_OVERLAP` self-load body (~:3788-3846) **has never executed and cannot be exercised by
   any build.** P1 converted dead code. Worse (R1): each futile attempt still does a full grow+shrink,
   occupying VGPR budget — i.e. OVERLAP *manufactures* the grow-fail events ROLEFLOW exists to survive.

2. **The ROLEFLOW reversion trigger measures the wrong thing.** (R1-F4, R2-F3)
   `DRAIN < STAGE` under OVERLAP is not "claimable work exists" — nothing is ever claimable (finding 1).
   It is the transient between a publisher's `lds_cmpstore_adv STAGE_HEAD` and its own `drain_advance`.
   So reversion fires on a race artifact, and requiring 16 consecutive observations makes it near-inert →
   **the one-way ratchet P2a was built to fix is not fixed**; waves still end permanently in `ROLE_AFEED`.

3. **The own-mailbox `ROLE` write races the terminal `ROLE_RETIRE` broadcast.** (R1-F6, R2-F1)
   `.Lflow_body` does read-then-act; the reversion does a blind `lds_put_r`. A wave can overwrite a
   `ROLE_RETIRE` that landed in the window, never see it again, and loop until `deadman_check` kills it —
   `QUIESCE_CNT` never reaches `WAVES`, so the collective exit degrades to the `RETBAR_MAX`/deadman
   backstop. The file's own note calls that ~18s of resident spin = the compositor-starve/safemode
   condition. **Two new instances** (ROLEFLOW reversion + the RCONV write, both new in this diff).
   FIX SHAPE: CAS `ROLE[wid]` from the expected old role, or re-read and abort if `ROLE_RETIRE`.

4. **grow-fail retry has no `deadman_progress` → silent loss of a whole field.** (R1-F2, R2-F4)
   Under CFASSIGN, `r = cohort_start + wid` is owned solely by that wave and `drain_advance` requires
   `SL_GEN == DRAIN`, so DRAIN can never pass `r` until that wave publishes it. The rev2 retry loop sets no
   progress, so after `DEADMAN_TICKS` the watchdog force-retires the wave; `r` is never stamped; STAGE/DRAIN
   park forever; the boundary drain-gate never opens; the whole WG deadmans out and C is missing `r` and
   everything after it in the field. Not a regression vs P1 (same end state) — but rev2's claim to have
   removed the wedge is FALSE: it converts "always wedges on grow-fail" into "wedges if the grow doesn't
   succeed within 0.5s". This is exactly the "missing `deadman_progress` SITE" case `CLAUDE.md` rule 4 names.

5. **Two extra `lds_get`s added to the hot dispatch path.** (R1-F7, R2-F6)
   The file records that ONE extra LDS read on this path cost **16× (97.3 → 5.9 TF)** because "at 98% coast
   the peek IS the hot path and 30 waves hammering an extra LDS read serializes". RCONV drives all waves
   into the feed role, so this hits ~100% of waves on ~100% of iterations.

6. **Phase accounting inverted.** (R1-F8, R2-F5) rev2 stamps after the grow, so the grow interval bills to
   `PH_WORK_WAIT` and `PH_GROW` reads ~0 — the exact mis-billing the 2026-07-22 note fixed, run backwards.
   Correctness-neutral; invalidates future PHASEPROBE runs of this build.

## R1-UNIQUE: the prefetch is a no-op that costs bandwidth (kills P2b as built)

Under `CFASSIGN`, `ASSIGN_HEAD` is **not** "next to be reserved" — it is the field-completion target `z`
(the source says so directly at the peek). `DA_BASE` is 2^shift-aligned and `ASSIGN_HEAD - DA_BASE` is a
whole multiple of the field width, so `(ASSIGN_HEAD + pf_i) - DA_BASE` masks to **exactly `pf_i` on every
visit**. Every feed wave, every pass, forever, prefetches `ksi ∈ {0,1,2,3}` of the CURRENT tile — lines
already resident. **Zero prefetch value.** Cost: `v9 = lane*8` makes each `global_load_tr_b64` a 256B
wave-wide fetch → ~1KB per wave-visit × 30 waves × every coast iteration × 64 WGs, **in the coast spin** —
the rule-7 shape that killed the desktop on 2026-07-14. Plus `s_wait_loadcnt 0x0` lands on the cohort
critical path. (R2 reached "prefetch is largely disabled" independently, via the feed-role-entry route.)

## CONFIRMED CLEAN (could not falsify — with the invariant that saves each)

- **No work duplication / retry is idempotent.** `r` is a pure function of (cohort_start, wid);
  `cohort_start` a pure function of DRAIN; DRAIN cannot pass an unstamped index (`SL_GEN == DRAIN`);
  `s15` is written only on paths that also publish. R2 traced `.Lflow_da_cf_decode` → `.Lflow_da_stamp`
  instruction-by-instruction: **`s45` is untouched**, so the deferred `s_mov_b32 s15, s45` is correct.
- **Prefetch cannot go out of bounds** (no page-fault/brick risk): `ksi` clamped to `[0, n_kseg-1]` in-path;
  `DA_TILE` validated `< chunkHi` before publish so `tcol < NTL`; the address is a strict subset of the real
  self-serve access set. NOTE (R2, worth doing): it is the only `tcol` derivation that skips SAFEPROBE's
  clamp — one `s_min_u32` would make it structurally safe rather than argument-safe.
- **No new cross-wave publish races** beyond the ROLE mailbox (finding 3). Sentinels publish exhausted
  values with `SL_GEN` last; both consumers re-validate `SL_GEN == head`.
- **Work-exactness intact.** `CLAIM_NOPERSIST`/occ[96] increments on the success path only. The gate the
  host actually enforces is `computed == G*TOTAL_super` (occ[71]); findings 4's losses surface as an
  UNDER-count, i.e. WORK-INEXACT, i.e. detected.
- **No register collisions** (`s75`, `s16-s20`, `s25-s29`, `v[16:17]`), no fat-token leak.

## ROOT CAUSE — one error, not six

claude__main designed against a model of the frontier that **`CFASSIGN` does not implement**:
- `ASSIGN_HEAD` is NOT a "next work" pointer — it is the field completion target `z`.
- The ring is NOT a live path — nothing is ever claimable under OVERLAP.
- `DRAIN < STAGE` is NOT "work is available" — it is a publish transient.

P2b's prefetch target, P2a's reversion trigger, and P1's ring conversion were all aimed at machinery that
behaves differently than assumed. **This is a re-derive, not a patch-three-things.**

## NEXT

1. Re-derive, from source, what "the next super-tile" and "work is available" actually mean under
   `CFASSIGN` — written down, verified at file:line, BEFORE any new build.
2. Only then redesign P2a's trigger and P2b's prefetch target on real signals.
3. Fix independently of the above (they are true bugs regardless): the ROLE/`ROLE_RETIRE` race (CAS the
   mailbox), and the missing `deadman_progress` on the grow-fail retry.
4. Decide what to do about the unreachable ring-compute path — delete it or make it reachable; leaving
   never-executed code in the file hides that grow-fail has no consumer.

## PROCESS NOTE

Codex usage was exhausted, so independence was reproduced with fresh Claude reviewers denied all prior
conclusions. It worked: claude__main's own review of this same code found NO blockers. That is the FOURTH
time static self-review has missed real defects in this handoff region. Independence — not model choice —
is the active ingredient. Also: give the reviewer the COMPLETE build profile (see the false positive above).
