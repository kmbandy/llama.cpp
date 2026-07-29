# DSWS S1 — Morning Brief, 2026-07-25

**Supersedes** `DSWS_BRIEF_2026-07-24_AM.md`. Full detail in `DSWS_TESTING_LOG.md` (2026-07-24 entries),
`REVIEW_DSWS2_P1_P2_2026-07-24.md`, `REVIEW_DSWS2_CF0_2026-07-24.md`.

---

# ⛔ RULE FOR THE MORNING: MEASURE FIRST. NO BUILDING.

**kmbandy, 2026-07-24 EOD: "before anything else, we need to take measurements. whether it's more timings,
the sleepn stuff, we need to measure measure measure."**

Do **not** open the kernel to add a mechanism. Do **not** design a fix. The single most expensive pattern
this project has is *building against an unmeasured assumption*, and it burned most of 2026-07-24: three
mechanisms were designed against a model of the frontier that the hardware does not implement, all three
came back correct-but-inert, and two separate rounds of adversarial review were needed to establish it.

**The next artifact is a NUMBER, not a mechanism.**

---

## 1. What is now MEASURED and SETTLED (do not re-litigate)

| question | verdict | evidence |
|---|---|---|
| Is the boundary election the wall? | **NO — settled from both directions** | winning pass 264t (ADVPROBE), losing pass 36.8t (BNDTIME), combined **<1% of wave-time** |
| Does 2 WG/CU help *today*? | **NO — 3.7% slower**, but see §3 | 27,485,692 vs 26,498,256 ticks, waves ~matched (2048 vs 1920) |
| Does the dyn-VGPR budget ever bind? | **NO** — `grow-fail = 0` even at **2048 resident waves across 2 WG/CU** | every run |
| Are carriers stalled? | **NO** — `occ[88] = 0` | every run |
| Is drain blocking? | **NO** — `drainwait ≈ 0` | PHIST census |
| Is the CF0 stack correct on silicon? | **YES** — dense oracle stride=1 (ALL 3168 tiles), work-exact | bring-up + 2WG + BNDTIME runs |
| Is the 40KB LDS reclaim real? | **YES** — 54,784B → 13,824B on hardware | every CF0 run |

**By elimination, the time is IDLE WAITING.** 95–99% of passes bail to `.Lflow_feedmt_sleep`. That is
ADVPROBE's unexplained ~90% gap, and it is still unexplained. **That gap is the target.**

## 2. RETRACTIONS — do not carry these forward

- ❌ **"The boundary is the dominant activity (≈73% of passes)."** I multiplied PHIST's 78.9% boundary-entry
  census by BNDSPLIT's 93.1% election-loss split and presented the product as a *time* claim. BNDTIME then
  measured those passes at **<1% of wave-time**. **A census counter can never establish a time share; only
  a timer can.** This is the exact error the morning rule above exists to prevent.
- ❌ **`door1 NOTHING-STAGED = 100%` and `occ[86] STARVATION = 100%` are NOT evidence of supply starvation.**
  Both are structurally ~100% under `SELFSERVE` regardless of what is happening (flagged in the log on
  2026-07-19 AND 2026-07-21; I cited them anyway on 07-23 to conclude "ASSIGN-bound"). **Do not quote them.**
- ❌ **`gatefull` / `zlock` / `terminal` / `bnd-lost` / `growfail` reading 0 in PHIST are NOT measurements** —
  those doors have **no bump sites**. Sixth instance of this project's "zeros that were never measurements"
  trap.

## 3. WHY 2 WG/CU IS STILL A LONG-TERM BET (kmbandy's read, and it is sound)

It measured 3.7% slower **today**, but that number was taken in a regime where nothing it unlocks can pay off:

- The kernel is **idle-waiting ~90% of the time**, so adding co-resident work cannot help until the idle is
  fixed — there is nothing to overlap *with*.
- `grow-fail = 0` at 2048 waves means the **dyn-VGPR moat still never engages**. 2 WG/CU is the only lever
  that plausibly makes the per-SIMD budget contend, which is the entire architectural premise of DSWS.
- The **prior "2 WG/CU is garbage" verdict was a clamp artifact** (`ML8_POOL=128` silently clamped to 64,
  `occ_dispatch.cpp:1995`). Yesterday was the **first genuine 2 WG/CU run in the project's history**
  (confirmed: `occ[20]=3296` ⇒ 128 WGs vs 3232 ⇒ 64).
- It is **now physically possible at all** only because of the 40KB LDS reclaim (13,824B/WG ⇒ 2×27,648 <
  65,536, and WAVES≤16 for the 32-wave-slot ceiling). That capability is banked and correct.

**So: 2 WG/CU is not refuted — it is untimely.** Re-test it *after* the idle-wait is understood, not before.
Do not spend effort tuning it now.

## 4. THE MEASUREMENT QUEUE (pick from here; each is one dispatch)

**A. `SLEEPN` — the top candidate, because it has a built-in ablation.**
Waves bail to `.Lflow_feedmt_sleep` → `s_sleep SLEEPN` → loop. If they sleep *through* work becoming
available, that latency is **chosen, not inherent**. `SLEEPN` is a defsym: measurable AND directly ablatable.
⚠ A historical `SLEEPN` sweep (2/8/32/128) returned **flat span** and was read as "cadence not sleep-bound."
That was a much older kernel, and this project has since learned **flatness is usually the fingerprint of a
fixed cost elsewhere, not a null result.** Re-derive it; do not inherit the conclusion.

**B. Time the idle directly.** BNDTIME's pattern (throttled `s71` 1-in-64 RTC, register-accumulate, single
emit) applied to the park→wake interval would attribute the ~90% instead of inferring it by elimination.
This is the honest "where does the time go" instrument we still lack. Reuse `bndtime_*` as the template.

**C. Wire the missing PHIST bump sites** (`zlock`, `gatefull`, `bnd-lost`) so the census stops lying by
omission. Cheaper than B but yields counts, not time — and §2 is the warning about over-reading counts.

**Instrument cost, measured, so you can budget:** ADVPROBE ~free · BNDTIME **4.3×** · PHIST **10.4×** ·
PHASEPROBE **~44×** (unthrottled `s_sendmsg_rtn` — the rule-5 brick pattern; effectively unusable).
**Never quote TF from a probe build.** Always keep a probe-off control: **25,483,124** ticks (2026-07-23
baseline, WAVES=30 1 WG/CU, `ML8_COOP_CHUNK=96`).

> **CORRECTION 2026-07-25:** an earlier draft of this line quoted **26,498,256** as the probe-off control on
> canonical `cac3ff7c`. That is wrong — 26,498,256 is the **CF0 bring-up** span (bin `85954d3c`), which is
> 4.0% SLOWER than the real baseline. Using it as the control would have hidden a 4% regression as "flat".

## 5. Tree state (verified at EOD)

- On disk: canonical **`cac3ff7c`** (`occ_dsws2_w30_flow_gd.bin`). Nothing staged. Latch clear. Card released.
- Kernel defsyms all default 0 and byte-identical off: `DSWS2_OVERLAP`, `DSWS2_ROLEFLOW`, `DSWS2_PREFETCH`,
  `DSWS2_BNDTIME`, `DSWS2_ADVPROBE`, `BNDSPLIT`, `PHIST`, `PHASEPROBE`.
- Bin shas: canonical `cac3ff7c` · CF0 baseline `128500f7` · CF0 stack w30 `85954d3c` · CF0 stack w16
  `98c97456` · CF0+ADVPROBE w16 `065da39a` · CF0+BNDTIME w16 `48519446` · PHIST-on-canonical `15b91d20`.
- **The CF0 stack is correct but its mechanisms are inert** (prefetch warms ~1.5% of its footprint; roles
  gate no work; grow-before-CAS costs drains ∝ contention). Fixing those is *design*, not tuning — and it is
  **behind** the measurement work, not ahead of it.

## 6. Pre-flight gotchas that cost time yesterday

- **Check the HOST binary freshness, not just the kernel bin sha.** `gpu_run.sh` guards a stale *kernel* bin
  but NOT a stale `occ_dispatch` — a run was burned producing no output because the host predated the print.
  `ls -la occ_dispatch occ_dispatch.cpp` before dispatching.
- ~~**`board_claim` does NOT queue.**~~ **SUPERSEDED 2026-07-25: it DOES queue now.** The enhancement was
  built — "a busy resource puts the caller in line and reports their queue position." So claiming a held
  resource is now the CORRECT action, not a mistake. Still `board_check` immediately before every claim.
  `board_claim(test=...)` also opens a repo test-log entry — but note it resolved to the WRONG repo
  (`mad-lab-mcp` instead of llama.cpp) when claimed from this spike dir; check `test_slot` in the response.
- **Give any reviewer the COMPLETE build profile.** A missing `FM=1` produced a phantom "does not assemble"
  blocker in review round 1.
- Dispatch env names ≠ build defsym names (`HARNESS.md` §34). Full ml8 shape needs
  `DSWS2_ORACLE_MTL=22 DSWS2_ORACLE_NTL=144`.

## 7. Process that worked — keep it

Two **independent** adversarial reviewers (fresh agents denied the design docs, the builder's progress docs,
and all of my conclusions; different lenses; complete build profile) found **6 real defects** across two
rounds — including a silent-wrong-C race and a safety regression I introduced — that **my own review missed
both times.** Independence is the active ingredient, not model choice. Codex usage was exhausted; fresh
Claude subagents reproduced it fine. **Any handoff-region change gets this before it runs.**
