# DSWS STAGGER+BATON — CONTINUATION BRIEF (2026-07-16 evening)

> ⚠️ **READ THIS THOROUGHLY BEFORE TOUCHING ANYTHING.** This is the on-disk backup of KG brief `13f9b38e`
> (project `dsws-traveling-peak-baton`). If you skim it you WILL rebuild the dams that ate an entire day.
> The whole day was building HARD LIMITS onto a kernel whose architecture forbids them.

## Read in full FIRST (in this order)
1. `FLOW_ECONOMY_DESIGN.md` — THE LAW: *"There is no blocking read anywhere in a wave's hot loop. That is the whole design."*
2. `DSWS_TRAVELING_PEAK_BATON_2026-07-16.md` — the definitive, kmbandy-approved spec.
3. `DSWS_TRAVELING_PEAK_BATON_PLAN_2026-07-16.md` — the plan to execute (baton first).
4. `DSWS_TESTING_LOG.md` — the full arc, every failed run and its root cause.
- KG (auto-injects): `78483218` (definitive design), `eaa49d93` (river principle).

---

## THE GOVERNING PRINCIPLE (never violate)
**THE WAVES ARE A RIVER THAT NEVER STOPS FLOWING. THE ACCOUNTING IS GATES THAT BIAS/REDIRECT THE FLOW — NEVER A
DAM THAT STOPS IT.** Every wave every pass does the most productive thing it can; if it can't, it does the NEXT
productive thing (feed) and retries next pass. NO blocking reads (a wave reads only its OWN mailbox, always
valid — like ROLE), NO artificial caps (physical `s_alloc_vgpr` grow-fail is the ONLY throttle), NO
waits/spins, NO hard partitions.

**When you catch yourself adding a cap / wait / token count / carrier split — STOP. It's a dam.**

---

## THE DESIGN — three DECOUPLED single-responsibility mechanisms (I kept conflating them; don't)
1. **STAGGER = launch rate.** All waves launch LEAN (32 VGPR). Only job: while ≥32 VGPR free AND below
   max-waves, launch waves ONE AT A TIME as fast as possible. Keeps a pool of ready lean waves resident.
   Touches nothing else. Analogy: on-ramp with NO metering light — merge rate dictated by the highway (free
   VGPR), one car at a time.
2. **BATON = grow-turn hand-off (the traveling peak).** When an in-flight compute wave STARTS SHRINKING it
   PUSHES "grow now" into ONE ready (not-yet-grown) wave's mailbox. That wave (was feeding) reads its OWN
   mailbox each pass (non-blocking, like ROLE), sees the baton, grabs a rowblk, grows, computes. Exactly one
   grows per shrink → peak travels, always ≥1 at peak, never 2 into the same registers. **PUSH not poll.**
   **TARGET = next-available** (round-robin, O(1)). Predictive target REJECTED (needs O(N) cross-wave phase
   inspection = slow + a blocking read + against principle; keeping the pool full is STAGGER's job, not the
   baton's).
3. **ACCOUNTING = dyn-VGPR + fungible budget.** Per-wave grow/shrink; physical budget the only throttle
   (grow-fail→coast, already handled `.Lflow_growfail`/door4). Budget fungible across roles (shrink 1 fat ≈
   room for 3–7 lean feed = the adaptive economy). **dyn-VGPR is PER-WAVE; stagger+baton are CROSS-WAVE.**

---

## DO-NOT-REPEAT — today's dams (ALL deadlocked identically: `computed=0, ASSIGN=0, claim=64 stuck, occ[98] runaway ~2–3e9`)
- **MAXFAT<ACC_N software cap** → head-coverage race (head ksi needs all ACC_N rowblks, but only MAXFAT<ACC_N
  can be fat → DRAIN freezes). And `MAXFAT=4` was UNJUSTIFIED at G=6 anyway (budget allows all 6+).
- **`.Lflow_batonwait` spin-poll** (unbounded, bounded, AND carrier-split variants — all wedged the
  coordinator; a lean wave hammering the FATTOK word IS the forbidden blocking read).
- **NCARR carrier/feeder hard split** → fought the emergent coordinator-tuned role mix.
- The whole **STAGGER=1 software token layer** (FATTOK/MAXFAT/fat_acquire/fat_release) is a REDUNDANT
  re-implementation of what `s_alloc_vgpr` does physically; the redundancy introduced non-determinism (SAME
  config gave clean AND deadlock = a **bootstrap race**). Plain `STAGGER=0` already IS the river and ran
  clean+deterministic.

---

## KEY NUMBERS / FACTS
- **STAGGER=0 clean+deterministic**, deep-K shape `576x4096x262144` (fed via deep-K `DSWS2_K=262144`): J=1=5.2,
  J=2=8.8, J=8=16.9, J=16=20.2, J=32=22.0 TF. Work-exact `computed=9437184 = TOTAL_super(1572864)·ACC_N(6)`.
- **BUDGET MATH:** `VBUDGET=1536, VLEAN=32, NFV=80` (FM=1; disasm `s_alloc_vgpr 0x50`). All-ACC_N-fat =
  `30·32 + ACC_N·48 ≤ 1536` → **binds at ACC_N>12**. At G=6: `1248<1536` = **NON-binding** → `door4
  GROW-FAIL=0` every run → stagger/baton can buy NOTHING (the "measure where it can't happen" trap).
  **Measurement ONLY matters at a BINDING G (grow-fail>0).**
- grow-fail HAS bound before (2026-07-13, WOFLUSH POOL_N=3 → **grow-fail=1588**, GROW 46% of compute) — a
  binding regime is reachable; **WOFLUSH removes bank LDS → opens ACC_N/POOL_N room.**
- **JDEPTH = burst length = the master knob** (duty cycle). Moderate J = short peaks; J→n_kseg = square wave =
  defeats the stagger.

---

## PLAN TO EXECUTE (baton first) — see the PLAN doc for full asm-level steps
`T0` confirm substrate → `T1` grow-permit LDS mailbox **+ BOOTSTRAP SEED** → `T2` push-at-shrink-START,
next-available round-robin (O(1), `lds_put` BEFORE the `s_alloc_vgpr`) → `T3` compute reads its OWN permit
(permit=1 → clear+grow+claim; permit=0 → feed; **REMOVE the FATTOK token layer**) → `T4` **REPEATED** greenlit
bring-up → `T5` binding-G measure → `T6` stagger (scope vs existing).

- ⚠️ **#1 RISK = the BOOTSTRAP SEED (T1):** the baton chain is self-perpetuating but needs a STARTING
  grow-turn. Seed the first compute wid's permit=1 at init. A missing/lost seed presents EXACTLY as the
  all-coast deadlock — **first thing to check on any deadlock.**
- **Two failure-catchers (baked into the plan):** (a) after EVERY edit, grep the compute hot loop for any new
  blocking-read / cap / wait (the dam-detector); (b) the bring-up must pass on **2–3 REPEATS**, not one run
  (single "clean"s hid the bootstrap race all day).

---

## CURRENT KERNEL STATE (all uncommitted)
- `occ_kernel_dsws_flow.s`: dam machinery (MAXFAT/FATTOK/fat_acquire/`.Lflow_batonwait`/NCARR/CNT_BATON)
  **RIPPED OUT.** Only `RELSTART` survives (default 1 = `fat_release` at shrink-START = the register-freeing
  HALF of the baton, but with **NO directed hand-off yet** = the non-determinism the baton's push-mailbox
  fixes).
- **STAGGER=0 INERT byte-identical `386dc28643ffb58568623ad6d89cfe62`** (`DECENTASN=0 FM=2 G=3 ACC_N=3 POOL_N=3
  WAVES=30 SEGK=64 WOFLUSH=1`) — the safety invariant, MUST stay.
- Last STAGGER=1 no-wait build = `c5f91a28` (deadlocked ONCE = the bootstrap race; NOT trustworthy).
- **Latch CLEAR, no brick all session.** On-disk bin is an inert build.
- ⚠️ Earlier I corrupted the SOURCE tail with a `head -n -13` probe restore (stripped `.end_amdgpu_metadata` /
  `.endif`) — FIXED. If a build errors on metadata, check the file tail.

---

## SAFETY (CLAUDE.md — non-negotiable)
Dispatch ONLY via `./gpu_run.sh`; ONE dispatch per greenlight (kmbandy greenlights each); changed kernel = ONE
bring-up then STOP; hang/DMFAT/BAD = full stop; DEADMAN 0.5s NEVER raised; feed via DEEP-K (`DSWS2_K`, guard
ON: `ML8_COOP_CHUNK` bounded + `ML8_COOP_CHUNK_MAXS=3.0`, NEVER `CHUNK=0`); no TF verdict from <~1s;
work-exactness `computed==TOTAL_super·ACC_N` every run.

## PROCESS LESSONS (banked)
READ the architecture's core design docs BEFORE building — `FLOW_ECONOMY_DESIGN.md` would have prevented the
whole day. Never add a hard limit to this kernel; it's a dam, the design wants a gate. Verify baselines (same
config gave clean AND deadlock = a race, not a fix). REPEAT runs to catch races. dyn-VGPR (per-wave) ≠
stagger/baton (cross-wave). Task #43.
