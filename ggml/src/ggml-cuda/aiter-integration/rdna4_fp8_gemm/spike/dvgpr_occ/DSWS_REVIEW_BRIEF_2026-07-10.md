# DSWS — External Review Brief (2026-07-10)

**You are one of four independent reviewers.** This document is your only briefing. It describes
what the DSWS kernel is, how it works, where we are, and where we're stuck. Read it, then form your
**own** conclusions from the source and from experiments you run yourself.

**Repo pin:** branch `feat/wp-dflash-ds4`, commit `f61ec4f85`. All DSWS work lives in
`ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`.

---

## 0. Ground rules for your review (please read first)

1. **Do NOT read any prior AI-review output**, and do not read the other reviewers' responses. We
   want four *independent* looks, not a consensus that inherited one opinion. (There are no
   standalone "review" docs in the tree; just don't seek out or assume prior model conclusions.
   Note that a few of our own design docs contain earlier model-authored rewrites — treat those as
   *our* design artifacts, but be aware they are not gospel.)
2. **Do NOT trust this document's claims.** Every load-bearing assertion below (e.g. "conversions
   are provably dormant," "CONV=0 is byte-identical," "the pool is ~7× under the bind threshold")
   is something you should re-derive from source at `file:line` or re-measure yourself. Some of our
   past conclusions were wrong (see §7). If you find one of these claims is false, that is the most
   valuable thing you can report.
3. **RUN the code, don't just read it.** Reading finds what is written down; executing finds what is
   true. The GPU itself is off-limits to you (see §3 — a bad dispatch bricks a live desktop), but
   **everything else is fair game and cheap**: assemble the kernels, run the CPU race models,
   run RGA, diff `.text` sha256 across build flags, and — most important for hand-written asm —
   **disassemble the `.bin` and check the emitted ISA against what the source claims it emits**
   (much of our correctness rests on `.if`-gated code paths). §9 has copy-paste commands; §10 has
   raw outputs from our own runs so you can compare.
4. **No claim without backing.** For every hypothesis or suggested fix you offer, state the
   **offline experiment** that would confirm or refute it (assemble/disasm/RGA/CPU-model), mirroring
   how we bisect. A hypothesis with no offline discriminator isn't actionable for us — GPU dispatches
   are rationed.
5. **Two things we do NOT need from you:** (a) a summary of what this doc already says — this is our
   own work, we know it; (b) generic "add tests / add docs" advice. We want architectural judgment
   and specific, source-grounded findings.

What we *do* want is in §8.

### Your deliverable (required format)

Write your review as a **new markdown file in this directory** (`dvgpr_occ/`), named exactly:

```
2026-07-10-<your-agent-slug>-dsws-review.md
```

Use the slug assigned to you — one of: **`fable`**, **`musacoder`**, **`gpt5.6-sol`**, **`grok4.5`**
(e.g. `2026-07-10-fable-dsws-review.md`). Do not overwrite or edit this brief, and do not create any
other files except your review (plus, if useful, transient scratch under `/tmp`).

Structure your review around §8's asks. For every finding: cite `file:line`, state the **offline
experiment** that confirms/refutes it, and mark your confidence. Put your highest-severity, most
source-grounded findings first.

**Independence rule (hard):** there are (or will be) up to four files matching
`2026-07-10-*-dsws-review.md` in this directory — one per reviewer. **Do NOT open, read, or reference
any of them.** They are the other reviewers' output; reading them defeats the entire purpose of four
independent looks. Read only the source, our design docs (§11), and evidence you generate yourself.

---

## 1. What DSWS is and why it exists

**DSWS = "Dynamic-Split Wave-Specialization."** One persistent-workgroup fp8 GEMM kernel for
**gfx1201 (AMD R9700 / Navi48, RDNA4, wave32)** that **re-balances its own mix of wave roles at
runtime** to the current bottleneck, using AMD's dynamic-VGPR primitive to make the VGPR budget
*fungible across roles*.

**North star:** 250–300 TF fp8 GEMM on gfx1201. Context (verify against `MAD305_DSWS_MASTER.md` §2–3):

- fp8 WMMA hardware ceiling ≈ **307 TF** (one WMMA = 16×16×16 = 8192 FLOP; R9700 = 64 CU × 2 SIMD32 =
  128 SIMDs).
- Best *static* kernel ≈ **165 TF (~52% of ceiling)** on square shapes; ~113 TF on the real ml8
  tall-skinny shapes (`down`: M=2048 K=9216 N=2560).
- **The wall is the VALU issue port, not memory and not occupancy.** Measured ~31 non-WMMA
  instructions issued per 32 WMMAs → caps at 8.34/15.9 WMMA/cyc = 52%. Occupancy is *flat* (minWaves
  4→8 barely moves TF). So the lever is **cutting non-WMMA issues per WMMA**, by specializing waves:
  the math wave issues ~only WMMAs while other waves absorb the address/feed VALU. That specialization,
  tuned at runtime to whatever the workload needs, is DSWS.

**The moat — dynamic VGPR (`s_alloc_vgpr`).** It's armable on a compute queue **only via raw PM4**
(writing `COMPUTE_PGM_RSRC2` bit 6 on a KFD queue via libhsakmt, bypassing ROCr). HIP/the toolchain
cannot emit it for compute. This is why the entire harness (`occ_dispatch.cpp` + `../dvgpr_pm4/`)
dispatches via raw PM4, not HIP. A compute wave can `s_alloc_vgpr`-grow (lean 32 → fat ~112 to hold
fp32 WMMA accumulators) or shrink, letting one fat compute wave's registers fund several lean feeders
or vice-versa.

**Novelty (our claim — challenge it):** a prior-art scan concluded that *runtime-adaptive
producer:consumer wave-role rebalancing with dynamic per-wave register reallocation* is not present in
known GPU GEMM kernels. CUTLASS warp-specialization uses **static** `setmaxnreg` at launch;
`s_alloc_vgpr` is per-wave static with no in-kernel sensing; Stream-K / persistent kernels balance at
the CTA level, not intra-CTA wave roles with mid-flight register realloc. If you know
counter-examples, that is high-value. (Prior-art detail: `MAD305_DSWS_MASTER.md` §6.)

---

## 2. Split-K is the enabler (not an occupancy play)

Full-K GEMM holds the fp32 accumulators (~64 of 112 VGPR) at peak for the *entire* K-loop (~95% duty —
a square wave, not a trapezoid), so there's no window to reallocate VGPR. **Split-K** creates brief
accumulator-peak windows separated by lean phases = the room to shift VGPR between roles at a lean
boundary. This is why DSWS is built on a split-K substrate.

---

## 3. Hardware & environment constraints (a fix that violates these is a non-starter)

These shape every design decision. Please internalize them before proposing changes.

- **No `s_barrier`, ever.** The dyn-VGPR + barrier combination has a documented deadlock history on
  this target. All cross-wave coordination is **barrier-free** (LDS atomics + published flags).
- **gfx1201 has NO release/acquire ordering for barrier-free cross-wave LDS.** LDS words can tear;
  LDS-vs-global are not cross-ordered between waves. Our LDS macros each end with `s_wait_dscnt 0x0`
  (drains a wave's *own* LDS ops) but that does **not** impose cross-wave ordering. Reliable handshakes
  are built from *monotone counters* and *single-writer* fields, not from "write A then write B and
  hope another wave sees them in order." (This fact was learned the hard way on the sibling
  `occ_kernel_coop.s`; it is the single most important constraint for reasoning about correctness.)
- **Dyn-VGPR / OOR-poison rule.** `s_alloc_vgpr` grows/shrinks the wave's VGPR file; it uses SCC-retry
  (loop until it succeeds). Any VGPR temp with index above the *lean* range read **before** a GROW
  completes is out-of-range poison. Pre-grow temps must stay in the lean register window. A grow-stagger
  multi-wave deadlock class exists and is deliberately **out of scope** for now (we stay at a fixed
  VGPR budget `M=576` / `VBUDGET=1536` where per-rowblk grow is proven safe).
- **The GPU drives the user's display.** The R9700 (gfx1201) is the compositor's GPU (there is a second
  card, a 6900XT/gfx1030, but it can't init pre-login and is *not* the target-user configuration — the
  kernel must coexist with a live compositor on a single display GPU). A hung/faulting dispatch starves
  the gfx ring → MODE1 reset → desktop freeze/reboot. Consequences:
  - Only **sub-second, bounded** dispatches are safe; the host chunks work and yields between chunks.
  - **A brick is a bug we will fix — but we accept bricks to collect the diagnostic data that tells us
    how to fix them.** (So "just avoid bricking" is not a design goal; "be diagnosable and recoverable"
    is. See the DEADMAN watchdog, §4.4.)
  - Every GPU dispatch is individually human-greenlit. **You cannot run the GPU; propose offline
    experiments instead.**
- **Never pass `--gl2c`** (a separate MES-crash landmine). Keep SAFEPROBE address clamps on.
- **Read-only / not-ours files** — do not propose edits inside these (they're shared substrate or
  reference): `occ_kernel_coop.s` (the proven cooperative kernel DSWS descends from), `fp8_oracle.*`
  (the correctness oracle). `occ_dispatch.cpp` is the raw-PM4 host harness — you may *read* it to
  understand the launch/observability, and suggest host-side changes, but treat it as delicate.

---

## 4. Architecture

DSWS has gone through a lineage; the **current** north-star kernel is the **flow economy**
(`occ_kernel_dsws_flow.s`). Earlier kernels remain in the tree as working references and as solved
lessons. Read the current design first, then the lineage.

### 4.1 The current architecture — the non-blocking FLOW economy (`occ_kernel_dsws_flow.s`)

Authoritative design doc: **`FLOW_ECONOMY_DESIGN.md`** (read it in full). Summary:

**The flood model.** Water flowing past a floodgate doesn't stop and read the gate — it flows; the gate
biases the flow. A wave never asks "is the next super-tile ready?" It reads its assigned role, grabs the
next unit of work for that role, and goes. When the accounting lags, the wave **keeps doing what it did
last cycle** until accounting catches up. **There is no blocking read anywhere in a wave's hot loop** —
that is the whole point. (This replaced an earlier design where every super-tile boundary every wave
polled a published generation flag and spun — that poll *was* the dominant measured wait.)

Three layers:

1. **Work-flow layer — a 3-frontier pipeline.** A deep operand pool of `N` LDS slots (`POOL_N`, default
   3; each slot = BRES 4KB + ARES 12KB = 16KB). Three **monotone** frontiers with invariant
   `DRAIN ≤ STAGE ≤ ASSIGN ≤ DRAIN + N`:
   - `ASSIGN_HEAD` — **single writer = coordinator (wid0)**. When a slot is free (`ASSIGN−DRAIN<N`) and
     work remains: claim the next global super-tile from `occ[20]`, reset that slot's counters, set its
     STAMP, then `ASSIGN_HEAD++` (release last). Reset is a clean single-writer-after-drain — no CAS,
     no reset-race.
   - `STAGE_HEAD` — feeds stage slot `STAGE mod N` cooperatively; any feed that sees a slot fully staged
     does a lock-free monotone `ds_cmpstore(STAGE_HEAD, sh, sh+1)` (losers retry).
   - `DRAIN_HEAD` — compute pulls rowblks from slot `DRAIN mod N`; any compute wave that sees a slot
     fully drained does `ds_cmpstore(DRAIN_HEAD, dh, dh+1)`, freeing it for the coordinator.

   `try_grab(role)` is **one atomic returning work-or-empty, never a spin.** The only lock-free
   arbitration is the two monotone frontier CAS bumps (idempotent; losers retry). Everything else is
   single-writer.

2. **Role-mailbox layer (the economy).** `ROLE[wid] ∈ {COMPUTE, AFEED, BFEED}` in LDS, **written only
   by the coordinator** (single writer → no races, no single-winner election). Read by every wave each
   cycle. A stale mailbox literally contains the previous role, so "coast on lag" is free — there is no
   flag to poll and no branch-back-to-sleep.

3. **Coordinator layer (wid0).** A lean feed worker that *also*, on a slow cadence (`COORD_PERIOD`),
   senses pool fill (`STAGE − DRAIN`) and **nudges one** `ROLE[wid]` toward the starved side, with
   hysteresis + cooldown held in its own registers (trivial — exactly one decider, no distributed
   epoch arbitration). Its lag never stalls anyone.

**Fungibility / the "coast" is a code-path flip, not a resize** (the key to it being free):
- **compute→feed coast is FREE**: a fat compute wave (112 VGPR) runs the lean feed code path as-is
  (feed code only touches the lean register window; the extra VGPRs sit unused). A compute wave starved
  for staged work instantly helps *stage* — no `s_alloc_vgpr`, no churn — refilling the pool. This is the
  common high-value case and is what structurally deletes the wait.
- **feed→compute coast is NOT free** (compute needs the fat allocation), so a feed that finds the pool
  full yields briefly; the coordinator, sensing pool-full, migrates feed→compute via the mailbox — a
  *deliberate* grow at adopt, damped by hysteresis.
- `s_alloc_vgpr` resize happens **only** on a coordinator-directed mailbox role change, at a lean
  boundary.

### 4.2 The emergent economy (what "no baked mix" means)

The current flow kernel deletes the compile-time `NCOMP/NAFEED/NBFEED` mix entirely. Waves launch with a
**compute-biased seed**; the host derives `W_launch` (total waves) and passes a physical `VBUDGET`; the
coordinator discovers the right mix at runtime via the mailbox. See `DSWS_EMERGENT_ECONOMY_DESIGN.md` /
`DSWS_EMERGENT_ECONOMY_PLAN.md`.

### 4.3 Group-split (the current occupancy lever — see §5)

`occ_kernel_dsws_flow.s` also carries a `GROUPS = G/ACC_N` decomposition that reduces a tile's rowblks
through `ACC_N` LDS banks in sequential passes, shrinking resident LDS so more workgroups fit per CU.
`ACC_N = G` (GROUPS=1) is the byte-identical baseline; `ACC_N < G` is the occupancy variant. All new
logic is `.if GROUPS>1`-gated. Details + edit map: `MORNING_2026-07-07.md`.

### 4.4 DEADMAN watchdog (brick-safety, always relevant)

Every wave stamps its start RTC; a `deadman_check` at every loop head force-retires the wave via the
normal path if it's been alive longer than `DEADMAN_TICKS` (default 50M ≈ 0.5s @ 100MHz, under the host's
0.75s chunk guard). A coordination hang → all waves hit the deadline → all retire → `occ[0]→0`, queue
idle, EOP fence fires → host sees a clean (incomplete) completion, **no wedge, no brick**; the oracle
flags the incomplete result. `DEADMAN=0` emits zero bytes. **Caveat:** it covers loop-head spins; it does
*not* cover an `s_alloc_vgpr` grow-spin (that never reaches a loop head) — hence M=576 only.

### 4.5 Lineage (for context; critique the *current* kernel, §4.1/§4.3)

- `occ_kernel_coop.s` — proven cooperative shared-B split-K GEMM. The barrier-free prod/cons ring
  counters DSWS senses. **Not ours to edit.**
- `occ_kernel_dsws.s` — "Phase-B conversion" lineage: a pinned claimer broadcasts super-tiles; role
  waves convert between compute/feed via a snapshot/quiesce handshake. **This lineage's CONV=1 failure is
  now solved** (see §7). Retained as a reference and as a solved-lessons artifact.
- `occ_kernel_dsws_ring.s` — an N-slot ring that halved the coordination wait but kept the
  poll-a-published-flag paradigm; superseded by flow. Retained as a working reference.
- `occ_kernel_dsws_flow.s` — **current.** The non-blocking flood model above.

---

## 5. Current direction and the reasoning behind it

**Where we are:** the flow economy assembles and runs; the immediate objective is **making the dynamic-VGPR
pool actually bind** — i.e., getting enough concurrent fat compute waves per SIMD that runtime role
rebalancing has headroom to work with.

**The number that drives everything right now** (from Run 5, silicon-measured, `MORNING_2026-07-07.md`):
the concurrent-fat gauge read `fatPeak = 263` across the device = **~2 fat waves/SIMD** (263 ÷ 128), while
the estimated bind threshold is **~13–14/SIMD**. So the pool is **~7× under** what's needed. That is why
occupancy suddenly matters even though the static wall (§1) says "occupancy is flat" — at static occupancy
the dyn-VGPR economy has almost no waves to rebalance.

**The lever:** the **group-split** (§4.3). Shrinking resident LDS (`ACC_N=3 → 33KB`, `ACC_N=2 → 25KB`) is
what lets **2 WG/CU** co-reside so the fat-wave pool can grow toward the bind threshold. B-reuse is
retained because each group re-scans `B[:,tcol]` from L2 (the whole B column stays warm — measured L2 =
8MB), not HBM.

**The reasoning chain, explicitly:** issue-port wall → specialize waves (DSWS) → specialization needs a
runtime-rebalanceable VGPR pool → the pool only has ~2/SIMD fat waves today → need higher occupancy → need
smaller resident LDS → group-split. Each link is a place to challenge us.

---

## 6. Recent struggles (the live problem)

**The group-split deadlocks.** `ACC_N=6` (whole-tile, GROUPS=1) runs clean; `ACC_N=3` (GROUPS=2) hangs.
Since GROUPS=1 is byte-identical to the known-good baseline, the hang is **group-split-specific**. Run
history (`MORNING_2026-07-07.md`):

- **Run 6** (ACC_N=3, diagnostics off): hung — `occ0=0 occ20=26 fence=--` from +0.2s; clean recovery at
  the 40s host timeout. `occ0=0` *fast* (waves retired quickly) but the WG never terminated → a
  **completion deadlock**, not a compute stall.
- **Run 7** (ACC_N=3, un-throttled instrumentation): **bricked ~14s** — MODE1, `MES failed to
  REMOVE_QUEUE`, **no VM/page fault**. Root cause of the *brick* (not the deadlock): the diagnostics wrote
  global memory *every spin iteration*, keeping the memory engine hot so MES couldn't evict the queue →
  REMOVE_QUEUE wedge. (Lesson: on a completion deadlock, instrumentation that hammers global memory
  converts a recoverable quiet hang into a brick. Throttle it, and stream survivable state every 200ms —
  a brick skips any end-of-run readout.)

**Leading hypotheses for the deadlock itself** (undecided — Run 8, §7-next, is designed to discriminate):
- **Staging deadlock:** the emergent all-compute seed relies on coast-feeding; if a group super-tile's
  staging never completes (`BFDONE<FN` or `ARDONE<G`), all compute coasts, `RBDONE` stalls < ACC_N, DRAIN
  never advances.
- **Exit-barrier:** a wave never reaches `.Lflow_dead`, so the count-to-WAVES collective exit never closes
  and the retired waves spin.
- **Completer/DRAIN:** some emitted `(tile,group,ksi)` super-tile never advances DRAIN.

Two distinct **brick taxonomies** we've now catalogued (useful for you to keep separate): (a) **OOB page
fault → gfxhub fault → MODE1** (an addressing bug; see §7); (b) **MES REMOVE_QUEUE wedge with no page
fault** (a live-spinning queue the memory engine can't evict; a *coordination/ instrumentation* problem,
not an addressing one). They look similar from the outside (desktop dies) but have opposite root causes.

---

## 7. What we recently got wrong, and the current next step (data — form your own view)

**We include hardware *data* here (not opinions) because withholding it would just burn your review on
dead ends we already closed. Challenge the conclusions.**

**(a) A "hang" that was actually an OOB fault — our biggest recent misdiagnosis.** For an extended period
we chased what looked like a *liveness deadlock* in the Phase-B (`occ_kernel_dsws.s`) claimer's
snapshot/quiesce handshake under CONV=1. We built CPU thread-models of the quiesce protocol, bisected the
quiesce gate on hardware, and reasoned extensively about memory-ordering races. **All of it was chasing a
ghost.** The real cause (commit `f0131142f`): a racy/torn `sti` read during the claimer's per-super-tile
republish decoded a garbage tile index → out-of-bounds A/B/C scalar base → gfxhub page fault → MODE1. The
"hang" was **MES stuck *after* the page fault**, not a liveness bug. Fix: a `t = min(sti>>shift, TOTAL-1)`
clamp in `DECODE_STI` (`occ_kernel_dsws.s:390-396`) that the SAFEPROBE comment had promised and never
implemented. Result: the exact config that bricked 3× **now completes clean and correct across 7/7
dispatches, dmesg silent.** *Lesson for you: on this hardware a "hang" and an "OOB page fault" are hard to
tell apart from the host side; do not assume a stalled dispatch is a liveness bug.*

**(b) Data points from that investigation that still stand** (offline-reproducible — see §10):
- The Phase-B CPU protocol models find **no stall**: `test_dsws_quiesce_race` 0/400 under both seq_cst and
  relaxed ordering; `test_dsws_envelope_race` 0/200. This corroborates "the protocol logic is race-free;
  the bug was elsewhere (addressing)."
- `test_dsws_straggler` shows the protocol only stalls if a follower is **(near-)permanently** blocked
  *before* its bail (bounded delays self-resolve). Under both orderings the stall gate is `s50` (the
  over-claim sentinel), *not* the quiesce counter `s51`.
- A hardware bisection that **disabled the quiesce gate** (forced its "done" condition true) made the
  Phase-B kernel hang **earlier**, not later — i.e., that gate is **load-bearing**; removing it lets the
  claimer race ahead of un-quiesced followers. (One earlier static-analysis pass had suggested making that
  gate diagnostic-only; the hardware refuted it. This is exactly why we now insist reviewers run/measure.)

**(c) Current next step — GPU Run 8 (built, awaiting individual greenlight).** A throttled
frontier-localization dispatch on the *flow/group-split* kernel: `ACC_N=3 STAGINSTR=1`, small shape,
`ML8_COOP_STREAM=1`, streaming the frontier freeze-frame to real disk every 200ms. Expected: a *quiet*
hang (throttle removes Run 7's brick escalator) whose last streamed `FRONTIER ASSIGN/STAGE/DRAIN
slot[RB/BF/AR] barrier` line names which pipeline stage froze — which then points at the specific
group-split edit to fix. **We would value your critique of this diagnostic plan before we fire it** (a bad
dispatch costs a brick + reboot): is the frontier instrumentation sufficient to disambiguate the three
hypotheses in §6? Is there a cheaper offline way to decide first? (Run 8 detail + the frontier-line
decision table + the group-split edit map: `MORNING_2026-07-07.md`.)

---

## 8. What we want from you

For each, cite `file:line` and (for any proposed change or hypothesis) the **offline experiment** that
confirms/refutes it.

1. **What you like** about the direction and architecture — briefly, and only where it informs a
   recommendation. (No need to restate our design back to us.)
2. **What you would change** in anything already implemented — the flow economy (`occ_kernel_dsws_flow.s`),
   the 3-frontier protocol, the coordinator/mailbox, the group-split, the DEADMAN watchdog, the host
   launch/observability. Concrete, source-grounded.
3. **Insight on the current group-split deadlock (§6)** — which of the three hypotheses the source most
   supports, any *fourth* mechanism we haven't listed, and how to move forward. Alternatives to the
   group-split for getting the dyn-VGPR pool to bind are welcome (the goal is 2 WG/CU with B-reuse
   retained; is there a better lever?).
4. **Scrutinize the load-bearing claims** (§0.2): re-derive or re-measure at least the ones your findings
   depend on. Tell us which are wrong.
5. **Review code we have NOT independently vetted:** newer work in `occ_kernel_dsws.s` postdating the
   Phase-B lineage — a "SENSOR FIX", `OCCA_PUB_OFF` (a claimer-published mid-drain occupancy peak), and the
   per-chunk diag — has not had a careful second read. Also the emergent-economy seed / `RETBARRIER`
   count-to-WAVES exit in the flow kernel.
6. **Pointed questions** (answer any that you can ground in source):
   - Is "every non-claimer wave bails exactly once per epoch/super-tile" actually true under the real ISA
     control flow, for both the Phase-B and the flow kernels? Where could a wave skip or double-count?
   - Given no cross-wave LDS ordering (§3), are the flow kernel's three monotone frontier CAS-advances and
     the single-writer `ASSIGN_HEAD`/`ROLE[]`/STAMP-reset discipline actually sufficient, or is there a
     torn-read / lost-update path?
   - Is the **coast** (fat compute wave running the lean feed code path with no resize) actually safe —
     no VGPR aliasing, no OOR temp, exec-mask-correct — as emitted? (Disassemble and check.)
   - Is a single resettable frontier the right primitive, or should these be monotone never-reset counters
     compared against `epoch*(WAVES-1)`-style expected totals (the pattern that made the coop kernel
     reliable)?
   - Is the group-split's per-group C-store base offset (`+= group*(ACC_N*FM*FN*1024)`) and its
     `STAMP=(group<<28)|sti` packing correct and collision-free for all shapes?
7. **Falsifiable next steps** — for anything you'd have us try on the GPU, give the exact build flags +
   run params + the observable that would confirm/refute, matching §9's conventions.

---

## 9. Reproduction & observability (what you can run offline)

All commands from the `dvgpr_occ/` directory. `L=/opt/rocm/llvm/bin`.

**Assemble the flow kernel (current) across the group-split configs:**
```bash
for acc in 6 3 2; do WAVES=16 G=6 SEGK=32 ACC_N=$acc POOL_N=1 STAGINSTR=0 DIAG=0 ./build_flow.sh; done
# spill check: llvm-objdump -d occ_dsws2_w16_flow_gd.o | grep -ci scratch_   (expect 0)
```

**Assemble the Phase-B kernel (CONV=0 baseline vs CONV=1):** see `build_dsws.sh` (`mk2`) and §10 for the
exact defsym line; diff `.text` sha256 to check the byte-identity claim.

**RGA static occupancy/spill:** `KSRC=<kernel.s> ./rga_check.sh <label> <DEFSYM=val ...>` (needs the RDTS
RGA install referenced in `rga_check.sh`).

**CPU protocol/race models (no GPU):**
```bash
for t in test_dsws_ctrl_model test_dsws_quiesce_race test_dsws_straggler test_dsws_envelope_race; do
  g++ -std=c++17 -O2 -pthread $t.cpp -o /tmp/$t && /tmp/$t; done
```

**GPU dispatch (for your reference only — you cannot run it).** The flow small-shape diag run:
```
DSWS2_FLOW=1 DSWS2_SEGK=32 DSWS2_ACC_N=3 DSWS2_G=6 FLOW_POOL_N=1 FLOW_WAVES=16 \
DSWS2_ORACLE_MTL=3 DSWS2_ORACLE_NTL=8 DSWS2_NKSEG=64 ML8_POOL=24 ML8_COOP_CHUNK=2 \
ML8_CHUNK_DIAG=1 ML8_COOP_STREAM=1 ML8_COOP_CHUNK_MAXS=0.75 timeout 12 ./occ_dispatch --dsws2
```
**Pass criteria:** `occ[0]==0` (all waves retired), **oracle CLEAN** (`ok=<n> bad=0`), `dmesg` delta 0.

**Observability map (occ words the kernel writes; host reads a subset):** frontiers occ[74]=ASSIGN
[75]=STAGE [76]=DRAIN [77]=RBDONE [78]=BFDONE [79]=ARDONE [80]=barrier; occ[70]=coast [71]=computed
[72]=feed [73]=grow-fail; occ[57]=FATLIVE [58]=FATMAX(peak concurrent fat). **Known observability gap:**
the host's dsws2 path prints `occ[0]`/`occ[20]` and (on the flow path) the streamed frontier line, but not
every DIAG word — proposals to close specific gaps cheaply are welcome.

---

## 10. Raw evidence from our runs (compare against your own)

Captured at `f61ec4f85`, this machine (gfx1201 toolchain; CPU models on host g++).

**Flow kernel (`occ_kernel_dsws_flow.s`) assemble matrix — all 0-spill:**
```
ACC_N=6  GROUPS=1  .text 16196B  scratch insns=0
ACC_N=3  GROUPS=2  .text 11624B  scratch insns=0
ACC_N=2  GROUPS=3  .text 10072B  scratch insns=0
RGA (ACC_N=3): gfx1201  SGPR_SPILLS=0  VGPR_SPILLS=0  USED_VGPRs=256(alloc)  livereg max VGPR used=85
```

**Phase-B kernel (`occ_kernel_dsws.s`) — CONV=0 vs CONV=1, both 0-spill:**
```
CONV=0  .text 4872B  sha c62568f6...  scratch=0
CONV=1  .text 8956B  sha ec2e3594...  scratch=0
DECODE_STI ti-clamp fix present at occ_kernel_dsws.s:390-396 (the real CONV=1 brick fix)
```

**CPU protocol/race models — all pass; the two race models find NO stall:**
```
test_dsws_ctrl_model : dispatch/cooldown/pool OK ; entry-contract OK ; envelope invariant OK ; ALL PASS
test_dsws_quiesce_race: [STRONG seq_cst] 0/400 stalled ; [WEAK relaxed] 0/400 stalled ; NO STALL
test_dsws_envelope_race: peak_conc 1/2/3 -> 0/200 stalled each ; NO STALL (envelope guarantees progress)
test_dsws_straggler:
  control (no straggler)                : 0/100 stalled   (both seq_cst and relaxed)
  PERMANENT block of 1 {compute|afeed|bfeed} @tile8 : 100/100 stalled  gate=s50  (identical both orderings)
  bounded delay {10,1000} @tile8        : 0/100 stalled  (self-resolves under the watchdog)
  bounded delay {100000, 5000000}       : 100/100 stalled gate=s50
  => stall is LIVENESS/scheduling, not memory ordering; only a HARD-stuck follower hangs; gate is s50.
```

**Interpretation we drew (challenge it):** the Phase-B protocol is logically race-free; its historical
CONV=1 failure was the OOB addressing bug (§7a), now fixed. The *current* problem is the flow kernel's
group-split completion deadlock (§6), which is a different kernel and a different mechanism.

---

## 11. Doc map (our own artifacts — read as needed; these are not "reviews")

- `FLOW_ECONOMY_DESIGN.md` — **the current architecture** (read first).
- `DSWS_EMERGENT_ECONOMY_DESIGN.md` / `..._PLAN.md` — the no-baked-mix emergent economy.
- `MORNING_2026-07-07.md` — current state, group-split mechanism + edit map, Run history 5–7, Run 8 plan.
- `MAD305_DSWS_MASTER.md` — north star, the wall, the moat, prior-art verdict, standing safety.
- `RESULT_DSWS.md` — phased results log (RGA constants, control-law model).
- `RING_SLOTS_DESIGN.md` — the superseded ring (why flow replaced it).
- `SPEC_DSWS_*` / `PLAN_DSWS_*` — the Phase-B/substrate lineage specs and plans.
- Kernels: `occ_kernel_dsws_flow.s` (current), `occ_kernel_dsws_ring.s`, `occ_kernel_dsws.s`,
  `occ_kernel_coop.s` (not-ours reference).
- CPU models: `dsws_ctrl_model.cpp` + `test_dsws_*.cpp`.

Thank you — independent, source-grounded, experiment-backed findings are exactly what we need.
