# DSWS stagger + baton — design spec (DEFINITIVE, 2026-07-16 evening)

**Status:** re-derived from scratch with kmbandy after a full day of wrong builds. THIS supersedes every
earlier version of this file. The earlier versions all made the same mistake — they built **dams** (hard
caps, waits, splits) onto a kernel whose entire architecture is a **river** that must never stop flowing.
Read `FLOW_ECONOMY_DESIGN.md` first; this spec is its stagger/baton extension.

---

## 0. The governing principle (the river)

**The waves are a river that never stops flowing. The accounting is gates that bias and redirect the flow —
NEVER a dam that stops it.** Every wave, every loop pass, does the most productive thing it can; if it can't
(can't grow, nothing to compute), it does the NEXT productive thing (feed) and comes back next pass. No wave
ever blocks, waits, spins on another wave, or hits an artificial cap.

Concrete rules that follow (violating any = a dam = wrong):
- **No blocking reads.** A wave reads only its OWN mailbox (which always holds a valid value); it never polls
  another wave's state or spins on a shared word.
- **No artificial caps.** The only throttle is PHYSICAL: `s_alloc_vgpr` grow-fail = budget full → coast+retry
  (already handled: `.Lflow_growfail`/door4). No `MAXFAT`, no software token count.
- **No hard partitions.** Roles are biased by the coordinator in the background, never compile-time split.
- **Everything self-meters from the flow**, not from a rule we write.

**Anti-patterns proven fatal 2026-07-16 (never reintroduce):** a `MAXFAT<ACC_N` software cap (→ head-coverage
race/deadlock); an explicit `.Lflow_batonwait` spin-poll (→ coordinator wedge — a blocking read); a hard
carrier/feeder `NCARR` split (→ fights the emergent role mix). Same config gave clean AND deadlock because the
software token layer is a redundant dam on top of the physical allocator.

---

## 1. Three decoupled mechanisms

The design is exactly three separate, single-responsibility mechanisms. They do NOT reach into each other.

### 1.1 STAGGER — the launch rate
Every wave launches at the same LEAN footprint (32 VGPR). Stagger's ONLY job: **whenever ≥32 VGPR is free AND
we are below the max wave count, launch waves — one at a time, in succession, as fast as possible.** It just
keeps a pool of ready lean waves resident. It does NOT touch grow, shrink, budget, or roles — those are the
accounting. Think: an on-ramp with **no metering light** — cars merge as fast as the highway (free VGPR)
allows; the flow itself dictates the rate, one car at a time.

### 1.2 BATON — the grow-turn hand-off (the traveling peak)
Among the compute waves already in flight, some are growing / at peak. **When one of them starts SHRINKING, it
passes the baton** — it stamps "grow now" into ONE ready (not-yet-grown, in-flight) wave's mailbox. That wave,
which was doing feed work, reads its own mailbox on its next pass, sees the baton, grabs a rowblk chunk, grows,
and computes. So **exactly one wave grows as one shrinks → the peak travels**, always ≥1 wave at peak, and two
waves never grow into the same registers.

- **Push, not poll.** The shrinking wave WRITES the signal into the target's mailbox (like the coordinator
  writes ROLE). The target reads only its OWN mailbox, non-blocking — it feeds until the baton lands, never
  spins, never inspects other waves.
- **Target = next-available** not-yet-grown in-flight wave (round-robin / next index). O(1), no cross-wave
  inspection. Decision (kmbandy + engineering, 2026-07-16): the "predictive" target — pick the wave that best
  maintains continuous coverage by reading every candidate's phase — is REJECTED for the first build: it
  requires O(N) cross-wave phase inspection in the hot shrink path (slow + a blocking-read/brick-adjacent
  pattern + against the principle), and it is unnecessary because keeping the ready pool non-empty is
  STAGGER's job, not the baton's. If WMMA-coverage gaps are ever MEASURED, the first lever is "launch more
  lean waves" (stagger), not a smarter baton. Fallback is already the recommendation: next-available.

### 1.3 ACCOUNTING — dyn-VGPR + the fungible budget
The per-wave `s_alloc_vgpr` grow/shrink and the physical per-SIMD VGPR budget. The budget is FUNGIBLE across
roles (shrink 1 fat compute wave ≈ frees room for 3–7 lean feed waves — the adaptive wave-role economy). This
layer is the physics; stagger and baton only choose WHEN a wave launches and WHEN it takes its grow-turn.

---

## 2. Why this produces the traveling peak (continuous compute, no valleys)

Stagger keeps ready lean waves resident. Baton hands the grow-turn from each shrinking wave to a ready one at
shrink-START, so as one peak falls another rises into the freed registers. The sum of instantaneous
allocations stays pinned near the budget `B` (governing rule, 2026-06-25: trapezoid time-avg ≈ peak/2 →
~2× the all-at-peak occupancy). The WMMA pipe never sees a valley because there is always ≥1 wave at peak, and
the budget is never over-subscribed because exactly one grows per shrink. Nobody waits; a wave without the
baton feeds (productive) until it gets it.

**Master knob (2026-06-25):** burst length (K-windows held at peak before shrink; = JDEPTH). Short burst → low
duty → more phase-slots → more occupancy, but needs more concurrent waves to keep WMMA full and more
grow/shrink overhead. Optimum = shortest burst that still keeps the WMMA pipe full. Pairs with MODERATE J
(J→n_kseg is a square wave = 100% duty = defeats the stagger).

---

## 3. Measurement precondition — this only shows value where the BUDGET BINDS

The 2× occupancy is only reachable when the physical VGPR budget is the binding constraint — i.e. `s_alloc_vgpr`
grow-fails actually happen (fat peaks contend for the register file). At G=6 nothing binds
(`WAVES·32 + ACC_N·48 = 30·32 + 6·48 = 1248 < 1536 = VBUDGET`, and `door4 GROW-FAIL = 0` every run) — the peak
never fills, so it looks identical to plain deep-J and the stagger/baton can buy nothing. **Grow-fail has bound
before** (2026-07-13: WOFLUSH POOL_N=3 → grow-fail=1588, GROW 46% of compute time — "fat peaks colliding," the
exact thing this design exists to phase-offset). So a binding regime is reachable; the first job is to get back
into one and MEASURE.

**Prereq gate (do NOT skip):** confirm VGPR is the binding occupancy constraint in the test geometry (grow-fail
> 0, fed ≥1s single-chunk). If something else binds (LDS, WG-slots, the ~600× frontier round-trip cost), fix
that first — the stagger/baton headroom only converts to throughput when VGPR is the wall.

---

## 4. Build order (each piece independently, river-safe)

1. **Baton, next-available, push-mailbox.** At shrink-START, the shrinking wave stamps "grow now" into the next
   ready wave's mailbox; a lean in-flight wave reads its own mailbox each pass (like ROLE) and, on the baton,
   grabs a rowblk + grows. NO wait, NO cap, NO poll of other waves. (The physical shrink already frees the
   registers; the baton just names who grows into them, one per shrink.)
2. **Stagger, lean-launch metering.** Launch lean waves one-at-a-time while ≥32 VGPR free and below max waves.
   (May already be substantially present as the resident-wave model; scope this against what exists before
   adding anything.)
3. **Offline gate each:** assemble, 0-spill, `STAGGER=0` byte-identical `386dc28`, and — the check that would
   have caught today — enumerate that NO new blocking read / cap / wait was introduced (grep the hot loop).
4. **One greenlit fed bring-up** at a NON-binding G first (correctness/liveness: oracle-clean, work-exact,
   deterministic across repeats — a repeat is what would have caught the bootstrap race).
5. **Then a binding-G run** (§3): grow-fail > 0, and measure whether the traveling peak keeps compute
   continuous (WMMA duty, TF vs the best STAGGER=0 baseline).

## 5. Open questions
- (O1) Does next-available leave measurable WMMA-coverage gaps at the binding regime? (If yes → more lean
  waves, not a predictive baton.)
- (O2) The geometry that makes VGPR bind while fitting LDS (WOFLUSH removes bank LDS → ACC_N/POOL_N room).
- (O3) The adaptive role-mix (coordinator biasing feed:compute from the ring sensors) — the Phase-4 economy,
  layered on the working traveling peak.

## 6. Current code state (2026-07-16 evening)
`occ_kernel_dsws_flow.s`: the dam-based `MAXFAT`/`FATTOK`/wait/`NCARR` machinery has been RIPPED OUT; the only
survivor is `RELSTART` (fat_release at shrink-START, default 1) — which is the *register-freeing half* of the
baton but currently has NO directed hand-off (the freed budget is grabbed by whichever wave races to it, which
is the non-determinism source). `STAGGER=0` inert `386dc28` intact; bin `c5f91a28` is the current no-wait
build (deadlocked once = the bootstrap race). **This spec's baton (directed push-mailbox hand-off) is what
replaces the race with a deterministic one-grow-per-shrink.** Nothing here is validated yet — build order §4.
