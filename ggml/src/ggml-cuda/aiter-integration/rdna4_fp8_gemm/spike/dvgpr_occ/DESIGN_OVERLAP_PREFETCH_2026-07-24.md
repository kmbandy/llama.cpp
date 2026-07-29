# DESIGN — Super-tile Overlap Prefetch (frontier double-accumulator), 2026-07-24

**Author:** claude__main, from a kmbandy round-table (2026-07-24). **Status:** design for build.
**Kernel:** `occ_kernel_dsws_flow.s`. **Config:** A1 canonical + `DSWS2_RCONV=1` (see `HARNESS.md`).
**Builder:** a Sonnet subagent writes the assembly. **This doc is PROSE + mechanism + insertion points +
invariants ONLY — no assembly.** Codex is reserved for a later secondary review.

> Touches the boundary interlock and adds a second live accumulation generation. It is
> **concurrency-sensitive** (the POOL_N>1 era hit `bad=96/116` from exactly this class of race — a
> completer storing half-summed banks). Paper-design first; **oracle-CLEAN + WORK-EXACT gated**; no
> throughput claim without a supervised ≥2s steady-state run. `DSWS2_*=0` byte-identical to `cac3ff7c`.

---

## 1. The measured problem (this session, 2026-07-24)

We are **assign-bound**, and the mechanism is now nailed down at source and confirmed by ADVPROBE:

- The frontier exposes **one field at a time** = `n_kseg = K/SEGK = 10` split-K slices of a single
  super-tile group. Under `CFASSIGN`, `ASSIGN == z == DA_ZDONE` and is **frozen** there until the
  boundary advances (`:4175`). Nothing past the current field is reservable.
- The boundary advance is **drain-gated**: `DA_ZDONE` (hence `ASSIGN`) does not move to the next field
  until `DRAIN >= ASSIGN` (all 10 slices drained, `:4340-4343`) **and** `GSTORED >= z>>shift` (the
  finishing tile's C-store landed, `:4350-4353`). Only then `zero_banks` + re-base + advance
  (`:4402-4421`).
- Consequence: with **30 waves but 10 slices exposed**, the cohort is clipped at the field end
  (`:4197-4202`) so ~10 waves get a unit and ~20 **sleep**. The 10 drain, everyone waits for the gated
  advance, then the next 10 appear. Measured: **100% empty-ASSIGN feed iters, 95% coast**, and ADVPROBE
  showed the advance *mechanism* is only ~264 ticks (~10%) of the ~2600-tick interval — **~90% is the
  dead gap between fields.**

The 10-wide exposure is the **natural split-K width of one tile** (its 10 K-slices are independent).
Widening it means either more tiles in flight (**POOL_N — DEAD, do not touch**) or more slices per tile
(**SEGK down — rejected: raises duty cycle, kills the dyn-VGPR moat**). This design takes the third path.

## 2. The idea (kmbandy) — overlap prefetch, not a deeper pool

Do **not** wait for all 10 slices to drain before exposing the next super-tile. When the current field
has only a **few slices left to drain**, advance the frontier **early** so the ~20 idle waves reserve
and begin the **next** super-tile's slices — getting right behind the last waves finishing the current
tile. The tail of tile N overlaps the head of tile N+1; **the frontier never goes empty.** That dead
gap (the ~90%) is what we are removing.

This is a **software pipeline at the frontier**, triggered **adaptively on drain-progress** (fires when
the field is nearly drained, so its rate tracks the real drain rate — no fixed cadence, no designated
advancer, flow-on not block). It is a **limiter/early-release**, ethos-aligned.

## 3. Why it needs a second accumulator (the load-bearing fact)

The accumulator is **per-super-tile, not per-slice.** All `n_kseg` slices `ds_add` into the **same**
`ACC_N=3` banks (`ACC_BASE + bank*ACC_STRIDE`, `ACC_STRIDE = FM*FN*1024 = 4096B` → **12,288B** for one
super-tile), and `zero_banks` resets them per tile (`:1223`). The drain-gate exists precisely to stop
`zero_banks`/reuse from stomping a still-draining or still-C-storing accumulator.

So overlapping N (tail draining into its accumulator) with N+1 (head accumulating) requires **two live
accumulators** — the current one and a prefetched one. There is no per-slice sub-accumulator to grow;
the second accumulator is a full `ACC_N=3` region (12,288B). This is a **1-deep overlap** (two
accumulators = the current tile + one prefetched), which is all that's needed to close the dead gap —
the goal is "no empty frontier between fields," not "all 30 waves always busy."

## 4. Funding it WITHOUT touching the pool — reclaim B's LDS

The per-slot operand region (`OP_BASE + slot*OPSTRIDE`) splits into a **B-staging** part
(`BRES_BYTES = FN*16*SEGK = 16,384B` at our SEGK=256) and an **A-staging** part
(`ARES_BYTES = G*16*FM*SEGK = 24,576B`). The self-serve compute path **fetches B straight from L2**
(`global_load_tr_b64`, :4755) and never reads the LDS B copy on that path. The **16,384B B region is
larger than the 12,288B a full second accumulator needs.**

Reclaim it: **commit B to L2-only** (drop the staged-B LDS read at `:3632` and any feeder prestage of
B), free 16KB, spend ~12KB on the second accumulator. Net LDS **drops ~4KB** (remove 16,384, add
12,288) — comfortably within the current 54,784B / 65,536B envelope. **No POOL_N change, no SEGK change,
no duty-cycle change, no dyn-VGPR change.**

**Why this is safe to do NOW and why we do NOT test B-bandwidth first:** we are assign-bound, so B
traffic is *not* on the critical path — any measurement now (e.g. a `NOBLOAD` ablation) runs in a regime
where B cost cannot manifest and would come back falsely flat ("B is free forever"). That is the
"measuring where the thing can't happen" trap. B-bandwidth becomes a **real, measurable question only
AFTER** this fix lets us rip super-tiles faster and the wall moves. It is **deferred, not tested away**:
if B traffic becomes the new wall post-fix, address it then (re-share B across waves, a lighter scheme).
Do **not** gate this build on a B-bandwidth measurement.

## 5. Mechanism (prose; the builder writes the assembly)

Add a defsym (suggested `DSWS2_OVERLAP`, default 0; B-L2-only lives on its ON path so OFF stays
byte-identical). On the ON path:

1. **Two accumulator generations.** Allocate a second `ACC_N*ACC_STRIDE` region in the LDS freed by
   removing B-staging. The current tile uses accumulator gen A; the prefetched next tile uses gen B;
   they **ping-pong** as the frontier advances (even/odd super-tile → gen 0/1).
2. **Early frontier advance (the prefetch trigger).** Where the boundary currently requires `DRAIN >=
   ASSIGN` before advancing, instead expose the next field when the current field is **within `OVERLAP`
   slices of fully drained** (`ASSIGN - DRAIN <= OVERLAP`), publishing the next super-tile's field into
   the *other* accumulator generation while the current tail keeps draining into its own. `OVERLAP` is a
   small defsym (start 2–3); the trigger is drain-progress-relative, i.e. adaptive.
3. **Funnel the idle waves in.** When the early advance fires, the next field's indices become
   reservable, so the waves currently sleeping with "no unit" (`:4206`, `wid >= field width`) find units
   in the newly-exposed field and start computing behind the current tail. No designated wave; whoever
   is free takes the newly-open slices.
4. **B from L2 everywhere.** Remove the staged-B LDS read (`:3632`) and feeder B-prestage; both compute
   paths self-load B from global (the self-serve path already does).

## 6. Invariants / constraints (non-negotiable — STOP and report if any cannot hold)

- **Per-accumulator drain/C-store protection.** The exact race the single-accumulator drain-gate +
  `GSTORED` + `TILEDONE` protect (a completer storing half-summed banks; `zero_banks` stomping a live
  read — the POOL_N>1 `bad=96/116` failure) MUST hold **per generation**: gen A's C must never be
  zeroed or stored while gen A's tail still drains/stores, independent of gen B's progress. `TILEDONE`,
  `GSTORED`, and the zero must be tracked/gated per generation. **This is the correctness crux.**
- **River ethos.** Early-advance is a limiter/gate that keeps flow going — never a dam (not-ready →
  flow on, never block/spin). Adaptive (drain-progress trigger), never a fixed cadence. No designated
  advancer — the election stays "whoever is free."
- **POOL_N untouched (=1).** This is NOT a deeper operand pool. Operands stay L2-self-loaded; only the
  accumulator doubles. If the design starts needing a second *operand* slot, STOP — it has drifted into
  POOL_N and is wrong.
- **Duty cycle / moat untouched.** SEGK, JDEPTH, VBUDGET, the dyn-VGPR stagger all unchanged.
- **Byte-identical off.** `DSWS2_OVERLAP=0` byte-identical to `cac3ff7c` at the A1 canonical profile,
  after every edit. All new code `.if DSWS2_OVERLAP` (including B-L2-only).
- **1-deep only.** Exactly two accumulator generations. Do not generalize to N — that reintroduces the
  pool-depth complexity this design avoids.

## 7. LDS budget (confirm at source before building)

Current ~54,784B of 65,536B. Remove B-staging `FN*16*SEGK = 16,384B`; add a second `ACC_N*ACC_STRIDE =
12,288B` accumulator. Net **−4,096B** → ~50,688B. Confirm the exact figures and that `ACC_BASE`/
`OP_BASE`/host `kOpBase` coupling stays consistent (host `occ_dispatch.cpp` has a matching LDS/`kOpBase`
static-assert — any base move is a **co-change** with the host, per `:718`).

## 8. Gates (builder does OFFLINE ONLY — never dispatches to the GPU)

- `DSWS2_OVERLAP=0` byte-identical to `cac3ff7c2338e73f` at the A1 canonical build profile.
- `DSWS2_OVERLAP=1 DSWS2_RCONV=1` assembles + links **0-spill** (RGA; the builder's sandbox may block
  `~/.rga` — if so, report the linked `.co` and I run RGA).
- Host `occ_dispatch.cpp` still compiles; any `kOpBase`/LDS constant co-changed and its static-assert
  passes.
- A written **self-audit** against §6: per-generation drain/C-store/zero reasoning, ping-pong index
  correctness, B-L2-only leaves the feeder economy coherent (feeders prestage A only, or the staged
  path cleanly drops B). **If removing staged-B breaks the staged/feeder path structurally, STOP and
  report — do not improvise a workaround.**
- The GPU oracle-CLEAN + WORK-EXACT + steady-state measurement is a **supervised dispatch I run** with
  kmbandy's greenlight — NOT the builder. The builder's deliverable ends at the offline gates + audit.

## 9. Open questions for the builder (resolve at live source; STOP-and-report if they contradict this)

1. Is B-staging (`BRES`) genuinely the only consumer of that 16KB, and does B-L2-only leave the staged
   compute path (`:3632`) and the emergent-feeder economy coherent? (Feeders prestage A only, or the
   staged path drops the B-from-LDS read.) If not cleanly separable, STOP.
2. Can `TILEDONE`/`GSTORED`/`zero_banks` be made per-generation within the existing frontier bookkeeping
   (`DA_ZDONE`/`DA_BASE`/`DA_TILE`) without a second operand slot? If it forces pool-like slot indexing
   on operands, STOP.
3. The `OVERLAP` trigger: confirm `ASSIGN - DRAIN <= OVERLAP` is readable with free scratch at the peek
   (`:4170`) and boundary (`:4340`) sites without clobbering live handler registers.

## 10. What NOT to do

- Do not raise POOL_N or add an operand slot. Do not change SEGK/JDEPTH/duty cycle. Do not put assembly
  in this doc. Do not dispatch to the GPU (builder). Do not run or gate on a B-bandwidth/`NOBLOAD` test
  now — that question is deferred to after the wall moves. Do not generalize past 1-deep.
