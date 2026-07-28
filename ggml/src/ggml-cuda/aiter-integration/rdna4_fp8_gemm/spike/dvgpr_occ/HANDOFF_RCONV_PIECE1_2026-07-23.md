# HANDOFF — DSWS2_RCONV Piece 1 (the note-drop) — Codex terra

**Supersedes** `SPEC_ROLE_CONVERSION_MAILBOX_2026-07-22.md`. **Author:** claude__main, 2026-07-23.
**Builder:** Codex gpt-5.6-terra (thread `a5b77e336fd22588b`). **Kernel:** `occ_kernel_dsws_flow.s`.
**Config:** A1 (`G=6 ACC_N=3 SEGK=256 POOL_N=1 WAVES=30 SSWIN=32 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1`).

> **No code in this handoff by design.** Mechanism + exact file:line insertion points + invariants +
> gates. You write the assembly. If live source contradicts anything here, **STOP and report** — do
> not improvise. This kernel bricks the box on guesses; a stop-and-report is a success.

---

## 0. What changed since the last spec — READ THIS FIRST

The 2026-07-22 spec had **three** pieces and flagged Piece 3 (a "CFASSIGN cohort re-key") as an
open architecture question. **That question is now resolved against source, and Piece 3 does not
exist.** Your own trace in `CODEX_RCONV_PROGRESS.md` was right; this handoff acts on it.

**The resolution (verified at file:line, 2026-07-23):** the CFASSIGN cohort publish is keyed on
`wid`, not role, and **both roles already funnel through the identical publish path**, so runtime
role conversion is *transparent* to the cohort gate:

- The served-cohort token `s15` is initialized **once at wave birth** (`:3005`, `s15=0xFFFFFFFF`)
  and thereafter touched **only** inside `.Lflow_da_peek` (`:4109`, `:4113`). It is a private
  per-wave register, **role-invariant**.
- `.Lflow_da_peek` (`:4074`) is reached by **feed waves** (`.Lflow_feed :3925 → .Lflow_feed_empty
  :4037 → :4074`) **and** by **coasting compute waves** (`.Lflow_coast :4870 → .Lflow_feed_empty
  :4876 → :4074`). Same code, same `s15`, same `wid` (`s24`).
- The cohort math is `r = cohort_start + wid` (`:4110`) with **no ROLE read, no count-slot read, no
  role-dependent branch** in the path.

**Consequence:** a converting wave keeps its `wid`, keeps its `s15` (which it has been maintaining
every time it coasted), and lands in the same publish path afterward. **No re-key. No `s15` reset.
No DRAIN stall.** A permanent feed wave reaches `da_peek` on every idle pass, where an oscillating
compute wave only reached it when starved — so its wid publishes *more* readily after conversion,
never less. This is strictly gate-favorable.

**Therefore this build is Piece 1 only.** Piece 2 (census) is deferred (telemetry-only, and the
slots are never initialized anyway — see §5). Piece 3 is deleted.

## 1. Why (measured, 2026-07-22)

PHASEPROBE on A1, fed, WORK-EXACT, oracle-clean: **RING_WAIT 56.0%**, SS_WAIT 21.2%, WMMA 19.3%,
rest <2%. The kernel is **starved, not slow** — ~27 compute waves (`FIRST_COMPUTE_WID=3`,
NCOMPUTE=27) contend for work ~7 effective stagers can't produce fast enough. Manual proof the
rebalancing works: dropping the launch to 5 waves gave **4.3×** on `ffn_gate_up M2048`. Conversion
makes that rebalancing *dynamic and per-wave* instead of a static launch count.

## 2. The build — Piece 1, the note-drop (unidirectional compute→feed)

At `.Lflow_coast` (`:4870`) — where a compute wave *already knows it is starved* (it reached coast
because `DRAIN >= STAGE`, nothing staged) — add a threshold-gated single LDS store that rewrites the
wave's own role mailbox to a feed role. That store **is** the physical conversion; the existing
dispatch machinery does everything else.

**Direction: compute→feed only, this build.** The measured problem is one-directional (too many
consumers). Reverse (feed→compute) is explicitly **deferred** — do not build it now.

**Mechanism, in order:**

1. **Persistence gate.** Maintain a private per-wave consecutive-coast counter: increment each time
   this wave coasts, **reset to 0 the moment it computes** (i.e. on the `.Lflow_havestage` path,
   `:3324`). Convert only when the counter crosses a threshold `N` (a defsym, default e.g. 64 — you
   choose a safe first value and state it). This filters transient empties; it is brake #2 (§4).
   - **SGPR budget is tight and hostile here.** `s66` holds `n_kseg-1` LIVE for the whole kernel
     (this is why `CONV_COOLDOWN>0` hard-`.error`s at `:2144`). `s15`=cohort token, `s24`=wid,
     `s34/s35`=roles are all live. You **must** find a genuinely dead SGPR across the coast↔compute
     round trip for the counter, or state that none exists and STOP. Do not reuse a live one.
2. **The role target.** Write the feed role the born-feed waves use. Role constants (`:626–629`):
   `ROLE_COMPUTE=0 ROLE_AFEED=1 ROLE_BFEED=2 ROLE_RETIRE=3`, mailbox at `ROLE_BASE=20` slot
   `ROLE_BASE + wid*4`. **Pick one deterministic feed role for this build** — suggest `ROLE_AFEED`,
   since init seeds only one A-feeder (wid1) vs two B-feeders (wid0,wid2), so A is the scarcer side.
   Do **not** build the A-vs-B "shorter side" heuristic: it reads `NAFEED_SLOT/NBFEED_SLOT`, which
   have **zero runtime writers** (§5) — the heuristic would branch on garbage.
3. **The store.** On threshold crossing: `lds_put (ROLE_BASE + wid*4) = ROLE_AFEED`. Next dispatch
   pass, `:3295` reads it, `:3298` sees role≠cur_role, `:3305` defensively shrinks to lean (a **no-op
   here** — a coasting compute wave is already lean; **confirm no `s_alloc_vgpr` grow fires**), sets
   cur_role, and dispatches down `.Lflow_feed`. **The seam is this write ↔ the `:3295` read; confirm
   the full loop closes and the new role is actually consumed** (do not leave a write with no
   effective reader).
4. **Commit counter.** Bump `convCount` at `CONVCNT_OFF=192` (occ[48]) so FORENSICS proves
   conversions happened. NOTE: an existing `global_atomic_add` to this offset lives at `:2138`
   **inside the dead DSWS2_CONV path** — you need your **own** bump on the note-drop event; do not
   route through `conv_apply`.

## 3. What is ALREADY built — do NOT rebuild

- The mailbox + per-pass read/act (`:3295`, `:3312`) and the role-change resize (`:3300–3310`).
- The starvation signal: reaching `.Lflow_coast` with `CNT_COAST` bump (`:4871`) already **is** the
  wave knowing locally it is starved.
- dyn-VGPR: all waves start lean (`:2984`); a compute wave grows fat only for its WMMA burst and
  shrinks straight back. A coasting compute wave is **already lean** → conversion is a **pure role
  flip with NO VGPR op**. Add no `s_alloc_vgpr` to the conversion path.

## 4. Damping — why this converges (no controller)

1. **Self-extinguishing (primary):** compute→feed adds stagers → ring fills → remaining compute
   waves stop hitting `DRAIN>=STAGE` → they stop coasting → they stop converting. The trigger cures
   itself.
2. **Persistence threshold** (§2.1): convert after N consecutive coasts, not the first.
3. **Cooldown:** `CONV_COOLDOWN` is scaffolded (`:405`) but **KEEP IT 0** this build — enabling it
   `.error`s at `:2144` (clobbers live `s66`). Do not give it an SGPR now; the two brakes above
   suffice for bring-up.
4. **No floor guard needed this build:** unidirectional compute→feed with WAVES=30 and only ~3 born
   feeders cannot drain compute below 1 in a single bring-up conversion; but **state the worst-case
   headcount** in your progress note so kmbandy can sanity-check before silicon.

## 5. Deferred — Piece 2 (census), do NOT build now

`NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT` (`:381–383`) are read only by the FORENSICS trace row
(`:2850–2855`) and are **never written by the coordinator** (`:3010–3085` writes none of them —
your finding). They are telemetry-only; no compute path consumes them, so a stale count cannot
corrupt C. `convCount` (§2.4) already proves conversions for bring-up. Leave the census slots alone.

## 6. Flag + byte-identity contract

- Defsym `DSWS2_RCONV`, **default 0**. All new code inside `.if DSWS2_RCONV`.
- **`DSWS2_RCONV=0` MUST assemble byte-identical to `cac3ff7c2338e73f`** (A1, CFASSIGN=1). Re-check
  the sha after **every** edit; any divergence = an edit leaked outside the guard.
- **Requires `CFASSIGN=1`** (it *is* CFASSIGN's adaptive half): add a `.error` guard for
  `DSWS2_RCONV && !CFASSIGN`, mirroring `:971–984`.
- Mutually exclusive with the dead `DSWS2_CONV` (opposite polarity): add a `.error` for
  `DSWS2_RCONV && DSWS2_CONV`.

## 7. Safety — NON-NEGOTIABLE (this kernel has bricked the box)

- **Rule 5 (hot path / message bus):** the coast path is a hot spin. Your store must be on the
  **taken-conversion branch only** (gated by the threshold, so rare) — **never** per-coast-iteration,
  **never** an `s_sendmsg_rtn`, **never** a global store in the loop body. One LDS store on the
  conversion event; one global atomic for `convCount` on the same rare event.
- **No `s_alloc_vgpr` in the conversion path** (§3). A role flip must not grow/shrink.
- **No new barrier.** This kernel deadlocks on `s_barrier` under dyn-VGPR. Conversion is a
  decentralized mailbox write; there is no rendezvous and must not be one.
- **You do NOT dispatch to the GPU.** Offline gates only. Never modify `occ_kernel_coop.s`. Only
  touch files in this spike dir; the git tree is shared with a live weight-pager session — stage
  nothing, and flag before any `git diff`. Never stage `docs/examples/router-fleet-main.ini`.

## 8. Offline verification gates (all green before you report done)

- `DSWS2_RCONV=0` sha **==** `cac3ff7c2338e73f` (after every edit — the regression guard).
- `DSWS2_RCONV=1 CFASSIGN=1` assembles **0-spill** (RGA / spill check). If the coast-counter SGPR
  forces a spill, STOP and report — do not spill.
- `DSWS2_RCONV && !CFASSIGN` and `DSWS2_RCONV && DSWS2_CONV` each `.error` correctly.
- Update `CODEX_RCONV_PROGRESS.md`: the SGPR you chose for the coast counter and why it's dead; the
  threshold `N` and worst-case surviving-compute headcount; confirmation the `:3295` read consumes
  the written role and `:3305` does not grow.

## 9. STOP-AND-REPORT if

- No genuinely dead SGPR exists for the coast counter across the coast↔compute round trip.
- The byte-identity check fails and the cause isn't an obvious `.if DSWS2_RCONV` leak.
- Any insertion point here does not match live source at the cited line.
- Assembling `DSWS2_RCONV=1` spills.

## 10. CANONICAL CONFIG + bring-up build commands (verified 2026-07-23)

**The `cac3ff7c` baseline is NOT `build_flow.sh`'s defaults.** The definitive A1 profile (anchored across
every 2026-07-22 decision/test-log entry, and the recorded recipe at `CODEX_PHASEB_PROGRESS.md:6`)
requires **`STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1`** — `build_flow.sh` defaults those to `0`, so a
plain invocation builds a *different* kernel (`55a6983d`, not `cac3ff7c`). Always pass the full profile.

`build_flow.sh` was extended (2026-07-23, claude__main) to pass `DSWS2_RCONV` + `DSWS2_RCONV_COAST_N`
(env-gated, default `0/64`, byte-inert when unset — verified: RCONV=0 canonical build still == `cac3ff7c`).

- **Canonical baseline (RCONV off) → `cac3ff7c2338e73f`:**
  `STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 ./build_flow.sh`
- **Bring-up conversion kernel (RCONV on) → `53a309f76a9bbea7`:** same line prefixed with
  `DSWS2_RCONV=1` (and `DSWS2_RCONV_COAST_N=<N>` to override the coast threshold; default 64).
- Byte-identity regression gate after any edit: the canonical baseline line must still print `cac3ff7c`.

Note: "fire exactly once" is not a single defsym — the design converts *any* wave crossing N consecutive
coasts, so a low `COAST_N` can fire multiple conversions. Use `FORENSICS=1` and watch `convCount`
(`CONVCNT_OFF=192`, occ[48]) to see how many fired; that is the bring-up observable, not a hard 1-shot.

## 11. GPU bring-up is kmbandy's, not yours

**GPU bring-up is kmbandy's, not yours:** single forced one-shot conversion (threshold set to fire
once), FORENSICS=1 so `convCount` proves exactly one conversion, dense stride=1 oracle, WORK-EXACT —
then the dynamic run to see whether RING_WAIT falls from 56%. Leave all of that to kmbandy.
