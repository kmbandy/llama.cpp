# DESIGN — the boundary-advance FUNNEL (a readiness gate, adaptive by construction)

**Author:** claude__main, 2026-07-23 (rev 2, after kmbandy round-table). **Status:** design for build.
**Kernel:** `occ_kernel_dsws_flow.s`, `.Lflow_da_boundary`. **Config:** A1 canonical + `DSWS2_RCONV=1`.

> Prose + mechanism + insertion points + invariants only. No assembly — the builder writes it.
> Touches the boundary interlock (where the INITBAR/TERMFIX races lived) → paper-design-first,
> oracle-CLEAN + WORK-EXACT gated, no throughput claim without a probe-off ≥2s steady-state run.

## 0. What changed in rev 2 (read this)

Rev 1 proposed a token-bucket rate limiter with a refill rate and an adaptive controller. kmbandy
rejected it as off-ethos, and correctly: a discrete token that gets consumed can run dry = a **dam**,
and a fixed refill rate is a **static cap** = also a dam. The right frame is a **readiness gate**, not
a rate limiter — and it is **adaptive by construction**, needs **no token, no fixed K, no controller,
and no new LDS state.** Rev 1's token/refill/controller/setpoint machinery is deleted.

## 1. The problem (measured 2026-07-23, BNDSPLIT)

With RCONV clearing RING_WAIT, the wall is SS_WAIT (50.7%). BNDSPLIT localized it: of every boundary
entry, **93.1% LOSE the ZLOCK election CAS** (herd), DRAINGATE_BAIL = 0.0%, CSTOREGATE_BAIL = 6.8%,
ADVANCE = 0.1%. ~930 waves storm the single election for every advance that happens.

## 2. The principle (kmbandy — the settled frame)

- **A bail means the funnel is broken.** The funnel admits a wave to advance ONLY when there is
  genuinely an advance to make. If an admitted wave hits a closed gate and bails, admission fired on a
  false signal. So: **design admission = the true readiness condition, and a bail becomes impossible.**
- **Limiter, not designator.** No privileged/positional wave. Whoever arrives while the gate is open goes.
- **Gate, not dam.** A closed gate makes a wave FLOW ON (do the next productive thing), never wait/spin.
- **Adaptive, not static.** The gate opens on real completion events, so its rate IS the live pipeline
  throughput — no fixed K, no controller, no setpoint to tune.

## 3. What "true readiness" is (from source)

An advance can only proceed when the finishing group is truly done and its banks are free to zero. The
kernel already encodes this as **two conditions, currently checked AFTER the election** in
`.Lflow_da_boundary`:
- **Drain-gate:** `DRAIN >= ASSIGN` (`occ_kernel_dsws_flow.s:4249`). Measured 0.0% — always true here.
- **C-store gate:** `GSTORED >= z>>shift` (`:4258`). Measured 6.8% — the ONLY real bail. Per the
  Codex-C1 comment (`:4251`): the banked completer bumps `RBDONE` (marking drained) *before* it reads
  the banks to write C, so a group can read `DRAIN==ASSIGN` while its C-store is still in flight;
  zeroing then would stomp the completer's read. `GSTORED` counts group C-stores whose `s_wait_storecnt`
  has drained — i.e. **`GSTORED >= z>>shift` is the signal that the advance is genuinely ready.**

The bug in rev 1's framing: readiness was tested *downstream* of the contention it should govern.

## 4. The mechanism — move readiness UPSTREAM of the election (rev 2)

Add a **read-only readiness pre-gate** at the top of `.Lflow_da_boundary` (terra-validated insertion at
`:4268`, before the ZLOCK CAS at `:4272`). Before a wave commits to the election it reads the existing
signals and tests the same readiness condition:

```
advance_ready  ==  (DRAIN >= ASSIGN)  AND  (GSTORED >= z>>shift)
```

- **Ready** → proceed to the unchanged ZLOCK election → advance. By construction it will NOT bail (the
  admission condition *is* the former bail condition).
- **Not ready** → **flow on**: branch to `.Lflow_feedmt_sleep` (`:4895`, the existing river path the
  election-loser and boundary-bail already use). The wave does the next productive thing and re-checks
  next pass. It does **not** storm the election against a boundary that cannot advance yet.

That is the whole funnel. Properties, all by construction:

- **No bail, ever** (§2): admission = readiness, so an admitted wave always advances.
- **Adaptive, no controller** (§2): the gate opens exactly when a C-store completes (`GSTORED` bumps),
  so the advance rate *is* the true pipeline-completion rate — it rises/falls with consumer count
  (incl. live RCONV conversions), shape, `n_kseg`, everything, because it is driven by work finishing,
  not a number we chose. No fixed K, no setpoint, no proportional gain, no timescale-vs-RCONV question.
- **No new state, no store, no brick surface:** the pre-gate is 3–4 LDS *reads* (`GSTORED`, `DA_ZDONE`,
  `DRAIN`, `ASSIGN`) + a conditional branch. No admission token, no counter, **no store at all** — so
  it is not a Rule-5 hot-store risk (unlike the BNDSPLIT counters, which had to be throttled). No free
  LDS slot is needed for the base gate. No clock — irrelevant that `s_memtime` is unsupported on gfx1201.

## 5. What this kills, and what it does NOT (be honest — measure it)

- **Kills the not-ready storm + all bails.** Today waves storm the exhausted boundary continuously
  during the window where the C-store is still in flight (the 6.8% that bail, plus the election losses
  incurred in that same window). The pre-gate closes that window to them entirely → they flow on.
- **Does NOT automatically zero the *ready-window* contention.** When a C-store lands and the gate opens,
  several waves sitting at the boundary can all pass the readiness test on the same pass and race the one
  real advance; the ZLOCK still elects one, the rest lose and flow on. BNDSPLIT cannot split "stormed
  while not-ready" from "raced a ready advance" a priori, so I will not claim the herd goes to zero —
  the residual is a burst per completion (~waves-at-boundary per advance), which is a large drop from
  93% but not necessarily to 0. **This is a measurement, post-build.**
- **If** the ready-window burst is still material after measuring, the second step is a light *limiter*
  on how many waves race per opening — and it, too, must be adaptive/ethos-aligned (not a static cap),
  designed only once the measurement shows it is needed. Do NOT pre-build it.

## 6. Insertion point + flow (terra-validated live source)

- Pre-gate: top of `.Lflow_da_boundary` (`:4268`), before the ZLOCK CAS (`:4272`). The existing
  `PH_BOUNDARY`/BNDSPLIT entry probes at `:4269–4270` are `s71`-throttled and stay.
- Not-ready target: `.Lflow_feedmt_sleep` (`:4895`) — the same river path `:4275` (election loser) and
  `:4404` (boundary bail) already use.
- The reads: `GSTORED_OFF`, `DA_ZDONE_OFF` (= z; mask off the ZLOCK bit before `>>shift`),
  `DRAIN_HEAD_OFF`, `ASSIGN_HEAD_OFF`. Reproduce the `z>>shift` math from `:4256`.
- **Keep the post-election drain-gate and C-store gate exactly as they are** (`:4249`, `:4258`) as the
  correctness backstop: between the pre-gate read and winning the CAS the state can move, and the
  post-checks catch that (rare) case and bail safely. The pre-gate makes them almost never fire; it does
  not replace them.

## 7. Invariants / constraints (non-negotiable)

- **Gate, never dam:** not-ready → flow on to `.Lflow_feedmt_sleep`, never wait/spin/block.
- **Limiter, never designator:** admission is "any wave that finds the gate open," no privileged wave.
- **Admission == readiness:** a bail is a design failure (§2). If the built gate can still bail, the
  gate condition is wrong — STOP and report, do not patch it with a lease/rollback.
- **Correctness untouched:** ZLOCK election + drain-gate + `GSTORED` gate + zero_banks + rebase +
  `DA_ZDONE` release all unchanged.
- **No store on the pre-gate** (it is reads + a branch) → no throttle needed, no Rule-5 surface. If the
  build finds it needs a store, STOP and report (it means the frame drifted).
- **Byte-identical off:** new defsym `DSWS2_FUNNEL`, default 0, all new code `.if DSWS2_FUNNEL`;
  `DSWS2_FUNNEL=0` byte-identical to `cac3ff7c2338e73f` at the A1 canonical profile, after every edit.
- **No `s_alloc_vgpr`, no barrier, no clock.** ACC is dead on this lean path (terra-confirmed).

## 8. Open questions (much smaller now — for the build)

1. Confirm at live source that the pre-gate can read `GSTORED`/`DA_ZDONE`/`DRAIN`/`ASSIGN` with only
   free scratch SGPRs at `:4268` (no clobber of the boundary handler's live registers). If not, STOP.
2. The `z>>shift` math and the ZLOCK-bit mask on `DA_ZDONE` must match `:4256`/`:4241` exactly.
3. Nothing else — the token/refill/controller/setpoint/RCONV-composition questions from rev 1 are
   dissolved by the readiness-gate frame; do not reintroduce them.

## 9. Verification gates

- `DSWS2_FUNNEL=0` byte-identical to `cac3ff7c2338e73f` (A1 canonical), after every edit.
- `DSWS2_FUNNEL=1 DSWS2_RCONV=1` assembles + links 0-spill (RGA).
- Single bring-up: dense oracle CLEAN + WORK-EXACT, `convCount` sane, no brick/hang. **And: zero bails**
  — with `BNDSPLIT=1` also on, `CSTOREGATE_BAIL` and the not-ready election losses must collapse; if any
  admitted wave still bails, the gate condition is wrong (§7).
- Then re-measure SS_WAIT + the BNDSPLIT split to see how far the herd fell and whether a ready-window
  limiter (§5) is warranted.
