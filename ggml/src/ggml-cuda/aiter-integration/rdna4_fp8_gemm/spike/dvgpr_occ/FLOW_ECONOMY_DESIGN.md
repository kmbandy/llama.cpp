# DSWS v2 — Fix #1 (the real one): the non-blocking FLOW economy

Design spec. This **supersedes the ring's consumption model** (`RING_SLOTS_DESIGN.md` / Fix #1a). The
ring cut the coordination wait in half but kept the *paradigm*: every super-tile boundary, every wave
still **polls a published flag** (`SL_GEN[DRAIN_IDX]`) and spins until it changes. That poll *is*
FOLLOW_WAIT. Measured: balancing rowblks made it worse (6c2a2b → 29%), adding feeds made STAGE worse
(4c3a3b → 20%) — because the poll/slot-sync is the disease, not the tuning.

**The flood model (kmbandy):** water flowing past a floodgate doesn't stop and read the gate — it just
flows; the gate biases the flow. A wave should never ask "is the next super-tile ready?" It reads its
assigned role, grabs the next unit of work for that role, and goes. When the accounting lags, the wave
**keeps doing what it did last time** until the accounting catches up — "a bit of a perf hit, barely
noticeable." **There is no blocking read anywhere in a wave's hot loop.** That is the whole design.

This is also fixes **#2** (wave count) and **#3** (feed:compute ratio) — they're the same knob: the
role mix, tuned in the background by the coordinator. Split-K's C-write amplification (FLUSH, the
measured 60–71%) is the memory-bandwidth floor we attack *after* this, as planned.

---

## The one idea that makes it non-blocking (kmbandy's twist)

Each wave has a mailbox slot `ROLE[wid]` in LDS that **always holds a valid role** — the last one the
coordinator wrote, or the launch role. So the wave never needs a "is my assignment fresh?" check:

- Coordinator kept up → slot holds a new role → the wave switches (dyn-VGPR resize) and flows.
- Coordinator lagged → slot **still holds the old role** → the wave keeps being that and flows.

The lag-fallback is **free** — a not-yet-updated mailbox literally contains the previous value. There is
no `s_cbranch`-back-to-sleep to write. FOLLOW_WAIT cannot exist because there is no publish to poll.

---

## The wave hot loop (every wave, every cycle — the whole thing)

```
loop:
  role = ROLE[wid]                       // read my assignment; stale == last role == coast (free)
  if role != cur_role:
      dyn-VGPR resize to role footprint  // compute=NFV(112) / feed=VLEAN(32); at a lean boundary
      cur_role = role
  got = try_grab(role)                   // ONE atomic fetch-add on this role's work counter; no spin
  if got:
      do_work(role, got)                 // compute a rowblk / stage an A-rowblk / stage a B-frag
  else:
      // my channel is momentarily dry -> FUNGIBILITY: do the complementary work for one cycle,
      //   which almost always has some (compute dry => there's staging to do; feeds dry/pool-full
      //   => there's compute to do). Never spin. The coordinator's job is to make this rare.
      alt = complement(role)
      got2 = try_grab(alt)
      if got2: do_work(alt, got2)  else: s_sleep SLEEPN   // true floor: both dry (rare transient)
```

No follow-loop. No `GEN` poll. No slot-boundary barrier between waves. The only `s_sleep` is the genuine
"there is literally no work anywhere this instant" floor, which the coordinator drives toward zero.

---

## Three layers

### 1. Work-flow layer — a 3-frontier pipeline, each advanced by a lock-free closer
A **deep operand pool** of `N` slots (reuse the ring's per-slot layout: BRES 4KB + ARES 12KB = 16KB
each). Super-tiles are consumed in claim order by local index `k`; slot = `k mod N`. **Three monotone
frontiers** (LDS u32), invariant `DRAIN ≤ STAGE ≤ ASSIGN ≤ DRAIN + N`:

- **`ASSIGN_HEAD`** — next local index to assign a global super-tile to. **Single writer = coordinator
  (wid0).** When `ASSIGN − DRAIN < N` (a slot is free) and work remains: claim `occ[20]`→`gsti`, **reset
  that slot's counters, set `STAMP[slot]=gsti`, then `ASSIGN_HEAD++` (release LAST).** Because the free
  gate means slot `ASSIGN mod N`'s prior occupant (`ASSIGN−N`) is already drained, the reset is a clean
  single-writer-after-drain — no reset-race, no CAS. This is the ring's "reset only at FREE, only by one
  writer, after the occupant drained" discipline, kept.
- **`STAGE_HEAD`** — oldest assigned-but-not-fully-staged index. **Feeds** stage slot `STAGE mod N`
  cooperatively (`SL_BFNEXT`/`SL_ARNEXT` fetch-add → `BSTAGE_R`/`ASTAGE_R` → bump `SL_BFDONE`/`SL_ARDONE`,
  verbatim from the ring). Any feed that observes slot `STAGE mod N` fully staged (`BFDONE==FN &&
  ARDONE==G`) does `ds_cmpstore(STAGE_HEAD, sh, sh+1)` — a lock-free monotone bump; losers just retry.
- **`DRAIN_HEAD`** — oldest not-fully-drained index. **Compute** pulls rowblks from slot `DRAIN mod N`
  (`SL_RBNEXT` fetch-add, same barrier-free protocol as the ring). Any compute wave that observes slot
  `DRAIN mod N` fully drained (`SL_RBDONE==G`) does `ds_cmpstore(DRAIN_HEAD, dh, dh+1)` — frees the slot
  for the coordinator to reassign.

`try_grab` per role is one atomic returning **work-or-empty**, never a spin:
- compute empty ⇔ `DRAIN >= STAGE` (nothing fully staged) → coast to feed.
- feed empty ⇔ `STAGE >= ASSIGN` (nothing assigned; coordinator behind) → coast; or pool-full for the
  assigner. Over-claim (`fetch_add ≥ max`) → attempt the CAS-advance, then retry/empty.

The CAS advances (`STAGE_HEAD`, `DRAIN_HEAD`) are the only lock-free arbitration, and they're monotone
counter bumps (idempotent, losers retry) — **not** the old `try_gate` role-election that caused the
reacting-controller bugs. `ASSIGN_HEAD` and every `STAMP`/reset is single-writer (wid0).

### 2. Role-mailbox layer (the economy) — which counter a wave pulls from
`ROLE[wid]` ∈ {COMPUTE, AFEED, BFEED}. Written **only** by the coordinator (single writer → no races,
no CAS, no single-winner gate — the exact machinery that caused the `try_gate`/`conv` reacting-controller
bugs is *absent by construction*). Read by every wave each cycle. Determines the steady-state mix.

### 3. Coordinator layer (wid0, a flowing worker with a side-duty)
`wid0` is a **lean feed worker** (it flows and stages like any feed) that **also**, on a slow cadence
(every `COORD_PERIOD` cycles), runs sense→nudge:

- **Sense** (cheap, LDS-local): pool fill = `stagedFrontier − drainFrontier`. Low fill → compute is
  out-running staging → operand-starved → **shift toward feeds**. High/full fill → feeds are ahead,
  compute-bound → **shift toward compute**. (This is the prod/cons ring-occupancy sensor that already
  exists.)
- **Nudge**: move **one** wave's `ROLE[wid]` toward the starved side (bias-on-next-adopt, not a live
  yank — the wave switches at its next lean cycle boundary). One step per period.
- **Anti-thrash**: hysteresis band + a cooldown (min cycles between nudges), held in wid0's own
  registers — trivial because there is exactly one decider. No shared thrash state, no distributed
  epoch arbitration.

Because waves **coast on lag** and a single GEMM's optimum is fixed, the coordinator can be as lazy as
we like: it settles the mix in a few nudges and then mostly senses. Its lag never stalls anyone.

---

## LDS layout (64 KB; N = 3 pool)
```
shared frontier + mailbox (bytes 0..~256):
  STAGE_HEAD     next super-tile index to stage (feeds claim; = occ[20] mirror or its own counter)
  DRAIN_HEAD     oldest not-fully-drained slot's super-tile index (compute advances)
  RINGINIT       barrier-free init publish flag (0xACED last)
  ROLE[0..WAVES-1]   per-wave mailbox (u32 each), coordinator-written
per-slot control block  s in [0,N):  (STAMP=super-tile id, GEN unused, SL_RBNEXT/RBDONE/BFNEXT/BFDONE/
  ARNEXT/ARDONE) — same fields as the ring
per-slot operands  OP_BASE + s*OPSTRIDE (16 KB each):  N=3 -> 48 KB + ~1 KB control < 64 KB  (N=4 = 64KB, too tight)
```
`N=3` gives compute up to 18 staged rowblks of runway — plenty to smooth super-tile boundaries so the
grab always finds ready work. (`N` is a defsym; 3 is the LDS-safe default, occupancy already 1 WG/WGP
above 32 KB either way.)

---

## Fungibility (the coast) — a code-path flip, NOT a resize (this is the key to it being free)
Critical distinction that avoids resize churn:

- **The coast flip is a CODE-PATH branch, not a `s_alloc_vgpr`.** A wave that gets "empty" from its own
  role just branches to the complementary role's code for one cycle. No VGPR resize.
- **compute→feed coast is FREE.** A fat compute wave (112 VGPR) can run the lean feed code path as-is
  (feed code only touches ≤v31; the extra VGPRs sit unused). So a compute wave starved for staged work
  instantly helps *stage* — no resize, no churn — which refills the pool and lets it compute again next
  cycle. **This is the common, high-value case** (compute starving is exactly the FOLLOW/STAGE wait),
  and it costs nothing. It is what structurally deletes the wait.
- **feed→compute coast is NOT free** — compute needs the fat allocation (accumulators at v32+), so a
  lean feed can't compute without growing. So a feed that finds the pool full **yields briefly**
  (`s_sleep`) rather than panic-growing; the coordinator, sensing pool-full, moves feed waves to the
  COMPUTE role via mailbox (a *deliberate* grow at adopt). Pool-full is the rare case (too many feeds),
  so this floor is small.

- **`s_alloc_vgpr` resize happens ONLY on a mailbox role change** (coordinator-directed, damped by
  hysteresis/cooldown), at a lean cycle boundary. Grow (feed→compute, 32→112) is the brick-class op;
  the multi-wave grow-stagger deadlock (ISA §3.3.3.2) is **out of scope** exactly as for the ring —
  **stay at M=576**, per-rowblk grow proven safe there. Grow-stagger is its own later increment.

So: coasts are free code-path branches (no churn); resizes are rare, deliberate, coordinator-driven.

## Non-blocking correctness argument
- **No spin on a producer anywhere.** `try_grab` is one atomic returning work-or-empty; "empty" flips
  to complementary work, never waits. The only `s_sleep` is the both-channels-dry transient.
- **No reset-race** — same barrier-free per-slot counter discipline as the ring: a slot's counters
  reset only at FREE→FILLING, only after its occupant's `SL_RBDONE==G`, and compute can't bump
  `SL_RBDONE` until `SL_BFDONE==FN && SL_ARDONE==G`. Carried over verbatim.
- **No mailbox race** — single writer (coordinator). Readers see a torn-free u32 (aligned word).
- **Termination** — a SENTINEL super-tile (past `chunkHi`) propagates: feeds staging it mark the slot
  terminal; compute draining a terminal slot retires; the coordinator, seeing all work claimed +
  drained, writes a RETIRE role to all mailboxes. `occ[0]` live-counter gate unchanged.
- **The true floor** = both channels dry for a wave = pool full *and* all staged rowblks claimed = a
  transient other waves resolve within a rowblk. Measured as the residual; the coordinator minimizes it.

## What carries over from the ring (not wasted)
Per-slot operand layout + `BSTAGE_R`/`ASTAGE_R`, slot-indexed counter macros (`lds_*_r`), the
barrier-free counter protocol, WMMA loop, split-K C-flush, PHASEPROBE/TFPROBE, host `DSWS2_RING`-style
LDS sizing (bump to N slots). **New file** `occ_kernel_dsws_flow.s` (fork of `occ_kernel_dsws_ring.s`);
the ring bin stays as a working reference; safe single-slot bins untouched.

## Non-goals (deferred, explicit)
- **Split-K C-write amplification** (FLUSH 60–71%) — the bandwidth floor, addressed AFTER flow.
- **Grow-stagger** — stay at M=576; training-M needs it and it's a separate increment.
- **Cross-WG global mix** — the economy is per-WG (LDS rings are per-WG); coordinator is per-WG (wid0).

## Test plan (scoreboard-driven, brick-safe)
1. Assemble clean, 0 spill, LDS ≤ 64 KB.
2. Greenlit `576×512×2048`, POOL=16, streamed: `occ[0]==0`, **oracle CLEAN**, dmesg delta 0. Restore.
3. PHASEPROBE run → **FOLLOW_WAIT should be ~0** (no poll left). Compare STAGE + the new both-dry floor
   vs the ring's 25%. Success = FOLLOW gone, total wait materially below the ring, oracle clean.
4. TFPROBE/sustained-reps **wall-clock** — the number that actually matters; compare to ring + baseline.

## Build status (2026-07-04)
- **Stage 1 (foundation) DONE:** `occ_kernel_dsws_flow.s` forked from the ring. LDS reworked to the
  3-frontier + mailbox + N=3 pool layout (`ASSIGN/STAGE/DRAIN_HEAD`, `ROLE_BASE` mailbox,
  `SLOTC_BASE=148`, `OP_BASE=256`, `LDS_TOTAL_FLOW=49408<65536`; `POOL_N`/`COORD_PERIOD` defsyms). Host
  `occ_dispatch.cpp`: `DSWS2_FLOW`→ldsBytes `256+POOL_N*16384` + bin `occ_dsws2_<mix>_flow_gd.bin`.
  `build_flow.sh` added. Descriptor group-seg 65536 (inherited).
- **Stages 2–4 (the unified loop) NOT YET CUT — file does not assemble yet** (old ring role loops still
  reference removed symbols `FILL_IDX_OFF`/`DRAIN_IDX_OFF`/`RING_D`/`.Ldispatcher`). This is expected:
  the flow control flow is a ground-up rewrite of the whole role section, all-or-nothing for assembly.
  The ring bin (`occ_kernel_dsws_ring.s`, GPU-verified) and safe single-slot bins are untouched.

**Resume plan — replace the ring's role section (branch + dispatcher + .Lbfeed/.Lafeed/.Lcompute) with
one unified flow section:**
1. **Coordinator (wid0):** LDS init (frontiers=0, mailboxes=launch mix, RINGINIT last); load chunkHi;
   loop { ASSIGN duty: if `ASSIGN−DRAIN<POOL_N` && more work → claim occ[20], reset slot, STAMP,
   `ASSIGN_HEAD++`; SENSE/NUDGE every `COORD_PERIOD`: pool-fill `STAGE−DRAIN` → nudge one `ROLE[wid]`,
   hysteresis in regs; then do a lean B-feed grab like any feed }.
2. **Unified wave loop (all wid>0, and wid0's feed part):** read `ROLE[wid]`; if `RETIRE`→retire; if
   `role!=cur` resize (grow/shrink at lean boundary); dispatch: COMPUTE→`try_compute_grab`
   (`DRAIN>=STAGE`?empty: fetch_add `SL_RBNEXT[DRAIN%N]`; `<G`→grow+WMMA+flush+`SL_RBDONE`++, on
   `RBDONE==G` `ds_cmpstore(DRAIN_HEAD)`); AFEED/BFEED→`try_feed_grab` (`STAGE>=ASSIGN`?empty: fetch_add
   `SL_ARNEXT/SL_BFNEXT[STAGE%N]`→`ASTAGE_R`/`BSTAGE_R`, on both-done `ds_cmpstore(STAGE_HEAD)`);
   **coast:** compute empty→run BFEED code (fat wave runs lean feed, FREE); feed empty→`s_sleep`.
3. Add a `lds_cmpstore_r` macro (ds_cmpstore_rtn_b32 wrapper) for the monotone frontier CAS advances.
4. Terminal: coordinator sets `FLOWTERM`+writes `ROLE_RETIRE` to all mailboxes once occ[20] past
   chunkHi AND all drained; waves read `RETIRE`→`occ[0]` live--→endpgm.
5. Assemble clean (0 spill, LDS≤64KB, PP 0&1) → greenlit `576×512×2048` (occ0==0, oracle CLEAN, dmesg
   0) → PHASEPROBE (FOLLOW→~0) → TFPROBE wall vs ring+baseline.

## DEADMAN watchdog (2026-07-04) — makes hangs recoverable, enables scale-stress
Every wave stamps its start RTC (`s[70:71]`, `s_sendmsg_rtn GET_REALTIME`); `deadman_check` at every
loop head (`.Lflow_loop`, `.Lflow_wait_init`) force-retires via the normal path if alive >
`DEADMAN_TICKS` (default 50M = 0.5s @ 100MHz, < host `chunkMaxS` 0.75s). A coordination hang (a frontier
that never advances) → all waves hit the deadline → all retire → `occ[0]→0`, queue idle, EOP fence fires
→ host sees a clean completion, **no wedge, no desktop brick**; the result is incomplete and the oracle
flags it. `DEADMAN=0` → zero bytes. **Validated** (1ms-deadline bin, greenlit): `occ[0]==0`,
`occ[20]=464`≠784 (fired early), oracle `bad=528` (incomplete, flagged), no hang/brick.
**Why it was needed:** the host's hang path did *not* destroy a hung queue ("brick-avoidance; process-exit
reclaims"), but process-exit reclaim of a live-spinning-wave queue faults the shared gfx1201 ring →
brick. The deadman drains the waves *before* that, in-kernel.
**Caveat:** covers the coordination-hang class (loop-head spins). Does NOT cover the `s_alloc_vgpr`
grow-spin (grow-stagger, ISA §3.3.3.2) — that never reaches a loop head; separate gate, M=576 only.

## Rollout / safety
New file; ring + `c62568f6` bins untouched. Every dispatch kmbandy-greenlit, one at a time, streamed,
safe-bin restored after; halt on any INCOMPLETE; never `--gl2c`; stay at M=576.
