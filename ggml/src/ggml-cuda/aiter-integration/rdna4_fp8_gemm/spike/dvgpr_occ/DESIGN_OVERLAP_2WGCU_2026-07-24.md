# DESIGN — Two-Generation Overlap × 2 WG/CU, 2026-07-24

**Author:** claude__main, from a kmbandy round-table (2026-07-24). **Status:** design for build.
**Kernel:** `occ_kernel_dsws_flow.s` + host `occ_dispatch.cpp`. **Config:** A1 canonical + `DSWS2_RCONV=1`.
**Builder:** a Sonnet subagent writes the assembly/host. **PROSE + mechanism + invariants ONLY — no code.**
**Supersedes** `DESIGN_OVERLAP_PREFETCH_2026-07-24.md` (its `ASSIGN−DRAIN` trigger was proven wrong — see §3).

> Concurrency-sensitive (dual live generations of frontier coordination state + a launch/occupancy change).
> The POOL_N>1 era hit `bad=96/116` from exactly this race class. Paper-design first, oracle-CLEAN +
> WORK-EXACT gated, no throughput claim without a supervised ≥2s steady-state run. Everything new is
> defsym-gated; OFF is byte-identical to `cac3ff7c`.

---

## 1. The measured problem

Assign-bound, precisely: the frontier exposes **one field = n_kseg = K/SEGK = 10 split-K slices** of one
super-tile at a time; the advance is gated on the finishing tile's *real* completion. With **30 waves but
10 slices exposed**, the funnel admits ~10 and **~20 waves slam it and sleep** (cohort clipped at field
end, `:4197-4202`). ADVPROBE: the advance mechanism is ~264 ticks (~10%); the other **~90% is the dead
gap** where nothing new is exposed. "Assign-bound," "100% empty-frontier," "93% lose the election," and
"20 waves idle" are all the same fact: **too few work-units exposed, too many waves competing for them.**

## 2. Two composed levers (they are synergistic, not alternatives)

**(A) Two-generation overlap, per WG.** Don't wait for the current field to fully complete before exposing
the next. When the current tile is *nearly* complete, open the **next** super-tile into a **second
accumulator generation** so waves start it behind the current tail. Exposes ~2 fields (~20 slices) → no
dead gap.

**(B) 2 WG/CU split.** Launch **16 waves/WG** and pack **two WGs per CU** (2×16 = 32 = the CU's wave-slot
ceiling). Each WG runs its own frontier + funnel, so each funnel arbitrates **10 slices among 16 waves
(10/16)** instead of 30 slamming one funnel for 10 slots. Halves the herd; doubles the number of
independent frontiers making progress.

Composed: two WGs × {16 waves over a 2-field overlap window} → **30+ waves productive across 4 concurrent
fields, two small well-behaved funnels, no dead gap.**

| | waves | slices exposed | working | idle/herd |
|---|---|---|---|---|
| today | 30 (1 WG) | 10 | 10 | 20 |
| **overlap × 2 WG/CU** | 16 ×2 | 20 per WG | 16 ×2 = 32 | ~0 |

## 3. Corrections baked in (do not repeat)

- **The trigger must read REAL completion, not `DRAIN`.** Under `SELFSERVE`, the self-serve path publishes
  a **pre-completed sentinel** (`SL_RBDONE=ACC_N`, `SL_GEN=r` last) at `.Lflow_da_ss_decode` (:4664-4684)
  **before** the compute burst (`.Lflow_da_ss_rowblk`, :4690). So `DRAIN` races to `ASSIGN` on reservation
  bookkeeping — `ASSIGN−DRAIN ≈ 0` for the whole dead gap. The real completion signal is **`GSTORED` /
  `TILEDONE`** (bumped after the actual `ds_add`/C-store, :4828 / :3971). The overlap trigger reads those.
- **The prior "2 WG/CU is garbage" verdict is INVALID** — a clamp artifact (`occ_dispatch.cpp:1995`:
  ML8_POOL=128 was silently clamped to 64, so it was never actually 2/CU). 2 WG/CU at the real config is
  **untested**, not disproven.
- **POOL_N stays 1. SEGK stays 256 (duty cycle / moat).** This is NOT a deeper operand pool.

## 4. The enabling reclaim — operands to L2-only (all paths)

The primary self-serve compute burst already loads **both A and B from L2** (`global_load_tr_b64` :4755 /
:4764) and never reads the LDS operand pool. It is tempting to call the 40,960B operand pool
(`OP_BASE + POOL_N*OPSTRIDE`) vestigial because our 30c0a0b steady state never touches it — **but that is
regime-dependent, not structural.** The pool is the **operand path for the `s_alloc_vgpr` grow-fail
fallback**: on grow-fail (`:4593-4594`) a reservation is published into the ring (`:4606-4639`,
`SL_BFNEXT/ARNEXT=0` = needs staging), then a coast-feeding wave stages its operands into the pool
(`.Lflow_coast` opportunistic feed, ~:5038-5107, **not** gated on RCONV) and a different wave computes
them via `ds_load_b64` from `OP_BASE` (`.Lflow_jloop`, ~:3552-3653). This path is dead **only because
grow-fail=0 at today's tuning** — and 2 WG/CU (32 waves sharing VBUDGET) is *designed* to make grow-fail
bind (the moat engaging). So deleting the pool as-is would corrupt accumulation the instant the design
starts working. (This is the second-Sonnet STOP, source-verified.)

**Therefore the reclaim = convert ALL compute paths to self-load, then remove the pool:**
- Primary self-serve path: already self-loads A+B from L2. No change.
- **Grow-fail fallback + coast-feed + ring-compute:** convert to **self-load A+B from L2** (mirror
  `.Lflow_da_ss_rowblk`). The ring **collapses to a control-only work-handoff** — it still does its real
  job (hand a grow-failed item to a wave that *can* grow fat), but carries only the small `SL_*` control
  state, no operands. B-reuse stays exactly as it already is for the dominant path: **L2 warm-cache reuse**
  (per DSWS_GRESIDENT §5, an L2 hit not HBM), never LDS sharing. kmbandy: full-L2, all paths (2026-07-24).

With no path reading the pool, the 40,960B `OP_BASE + POOL_N*OPSTRIDE` region is removed from the LDS
layout. This funds BOTH the second accumulator generation AND the 2 WG/CU headroom. Safe now (assign-bound
→ operand traffic isn't the critical path); the operand-bandwidth question is **deferred, not tested** — a
NOBLOAD run in this regime reads falsely flat. It becomes real only after the wall moves (and would then
apply to all paths equally, since all now self-load).

## 5. LDS + occupancy budget (confirmed at source)

Host occupancy guard (`occ_dispatch.cpp:2010-2024`): 2 WG/CU requires per CU `2×ldsBytesRaw ≤ 65536`
(per-WG LDS ≤ **32,768B**) AND `2×WAVES ≤ 32` (**WAVES ≤ 16**; 32 wave-slots/CU, verified via rocminfo).

- Today: `ldsBytesRaw = kOpBase(512) + POOL_N*OPSTRIDE(40,960) + ACC(12,288) + SSWIN*32(1,024) = 54,784B`,
  WAVES=30 → guard correctly **refuses** 2/CU.
- Target: operands L2-only (OPSTRIDE→0) + **two** ACC generations + control:
  `512 + 2×12,288 + 1,024 ≈ 26,112B per WG`. Then `2×26,112 = 52,224B < 65,536` ✓ and WAVES=16 → `2×16=32`
  ✓ — guard **passes**. Confirm the exact bytes at source (and the `kOpBase`/host static-assert co-change).
- Launch mechanics: workgroup = `WAVES_LAUNCH` waves (`dims NUM_THREAD_X = WAVES*32`, :1884); grid = `pool`
  WGs (:2118); per-WG LDS → RSRC2 granule via `ldsRsrc2Bits` (:1960). Hardware auto-packs WGs/CU by
  LDS+wave-slot+VGPR — no explicit "2/CU" flag; make each WG small enough and set `ML8_POOL=128`.

## 6. The two-generation frontier (the hard part; prose — builder writes it)

Per WG, split today's single atomic boundary action (check-complete → `zero_banks` → rebase → advance,
done by one ZLOCK-elected wave) into **two independently-synchronized actions**, with a **generation tag**
on the frontier (2 entries, ping-pong even/odd super-tile):

1. **Open-next-field (early).** When the current tile's *real* completion is within `OVERLAP` of done
   (measured on `GSTORED`/`TILEDONE`, per §3 — NOT `DRAIN`), expose the next super-tile's field into the
   **other** generation: zero *that* generation's ACC banks (never the one still draining), publish its
   field so idle waves reserve it. Bounded to exactly **1-deep** (two generations, never three).
2. **Retire-this-field.** Still gated on the current generation's *real* `GSTORED` completion; frees that
   generation for reuse two fields hence. It must NOT itself re-trigger an early-open (open is action 1's
   job only) — otherwise the depth is unbounded.

`TILEDONE`, `GSTORED`, and `zero_banks` become **per-generation**: `TILEDONE_BASE` is already a
`GROUPS`-array (:740) → make it 2-generation-indexed; `zero_banks` zeroes only the newly-opened
generation (:1238-1243 today zeroes all); `GSTORED` either 2-entry or its boundary math re-derived per
generation. The **cohort math** (`field_start = z − field_width`, :4181-4197) must read the *generation's*
z, not a single global scalar — this is the corruption the builder flagged: with two open fields there are
two `field_start`s.

## 7. Invariants / constraints (STOP and report if any cannot hold)

- **Per-generation completion protection.** The `bad=96/116` race (a completer storing half-summed banks;
  `zero_banks` stomping a live read) MUST hold per generation: gen A's C never zeroed/stored while gen A
  still accumulates, independent of gen B. This is the correctness crux.
- **Exactly 1-deep.** Two generations, never three. Open is gated so it cannot run twice before a retire.
- **Trigger on real completion** (`GSTORED`/`TILEDONE`), never `DRAIN` (§3).
- **River ethos.** Early-open is a limiter/gate that keeps flow going; not-ready → flow on, never block.
  No designated advancer; whoever is free takes the newly-opened slices.
- **POOL_N=1, SEGK=256, JDEPTH, VBUDGET unchanged.** Operands L2-only; only the accumulator doubles.
- **Byte-identical off.** New defsyms (`DSWS2_OVERLAP` default 0, `OVERLAP` default 2); all new code gated;
  OFF byte-identical to `cac3ff7c` at A1. The 2 WG/CU launch (WAVES=16, ML8_POOL=128) is a runtime/host
  choice, not a kernel-OFF change — the WAVES=16 bin is a different build, not a regression of the WAVES=30 bin.
- **Moat is a measurement, not an assumption.** 32 waves/CU share VBUDGET=1536; grow-fail may finally BIND
  (moat engages — good) or throttle. Do NOT pre-judge; measure. Do not raise VBUDGET to dodge it.

## 8. Suggested build order (each phase oracle-CLEAN + WORK-EXACT before the next)

1. **Operands L2-only, all paths** (`DSWS2_OVERLAP` phase-1 gate): convert the grow-fail fallback +
   coast-feed + ring-compute paths to self-load A+B from L2 (primary path already does); collapse the ring
   to a control-only handoff (keep `SL_*` control, drop operand staging); then remove the
   `OP_BASE + POOL_N*OPSTRIDE` pool from the LDS layout. Per-WG LDS falls to ~14KB (one generation).
   Oracle-prove correctness — **including a grow-fail-BINDING run** (force the budget to bind so the
   converted fallback path actually executes; grow-fail=0 does NOT exercise it — that's the whole point).
   This alone changes little steady-state perf (not bandwidth-bound) but is the enabler and independently
   verifiable.
2. **Two-generation overlap frontier** (still 1 WG, WAVES=30 or 16): the §6 mechanism. Oracle-prove.
3. **Flip to 2 WG/CU**: build WAVES=16, dispatch `ML8_POOL=128`; the guard now passes. Supervised measure.

Phases 1-2 are pure kernel + host-LDS; phase 3 is a launch change. Each is independently gatable, which is
how a concurrency-critical change earns trust. Do not fuse them into one un-bisectable build.

## 9. Gates (builder = OFFLINE ONLY; never dispatches)

- `DSWS2_OVERLAP=0` byte-identical to `cac3ff7c` at A1.
- Each phase's ON build assembles + links **0-spill** (RGA; report the `.co` if the sandbox blocks `~/.rga`).
- Host `occ_dispatch.cpp` compiles; the occupancy guard math + `kOpBase` static-assert hold for the new
  LDS and for WAVES=16 / ML8_POOL=128 (it should PASS, not refuse — verify).
- Written self-audit vs §6/§7: per-generation completion reasoning, 1-deep bound, cohort-math generation
  tagging, operand-L2-only coherence. **If per-generation correctness needs a second OPERAND slot, or the
  cohort/frontier can't carry a generation tag cleanly, STOP and report.**
- Oracle-CLEAN + WORK-EXACT + steady-state TF is a **supervised dispatch claude__main runs** with a
  greenlight — never the builder.

## 10. Open questions (resolve at source; STOP-and-report if they contradict this)

1. Can `TILEDONE`/`GSTORED`/`zero_banks`/cohort-`field_start` all be made per-generation within the
   existing `DA_ZDONE`/`DA_BASE`/`DA_TILE` frontier without a second operand slot? If not, STOP.
2. Does the control-only ring leave the grow-fail handoff + RCONV/feeder coherent? The ring must still
   hand a grow-failed item to a wave that can grow fat — it just carries `SL_*` control, no operands, and
   the receiving wave self-loads from L2. Confirm coast-feed (~:5038-5107) and ring-compute (~:3552-3653)
   have no *other* dependency on the LDS operand region, and that RCONV's AFEED role degrades cleanly (its
   staging job is gone; it self-serves like everyone else). **Verify with a grow-fail-binding build, not
   grow-fail=0.** If the handoff needs LDS operands for correctness, STOP.
3. Exact per-WG LDS at {operands-L2-only + 2 generations} and confirm the host guard passes at WAVES=16 /
   ML8_POOL=128 (perCU=2, ldsNeed<65536, waveNeed=32).

## 11. What NOT to do

No second operand slot / POOL_N change. No SEGK/JDEPTH/duty-cycle change. No `DRAIN`-based trigger. No
raising VBUDGET to dodge grow-fail. No fusing phases 1-3 into one build. No assembly in this doc. Builder
never dispatches. Do not run/gate a B-bandwidth (NOBLOAD) test now — deferred until the wall moves.
