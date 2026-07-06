# DSWS Stagger — Design (2026-07-05)

The "S" in DSWS (Dynamic **Staggered** Wave-Spec). This builds the staggered traveling-peak
occupancy layer on top of the write-once-C foundation, inside the existing flow economy
(`occ_kernel_dsws_flow.s`). It supersedes the split-K atomic-flush compute path and fills the
occupancy/latency headroom the write-once calibration exposed.

Sibling docs: `FLOW_ECONOMY_DESIGN.md` (the mailbox/coordinator economy this extends),
`RING_SLOTS_DESIGN.md`, `SPEC_WAVESPEC.md` (MAD-305 traveling-peak origin).

---

## 1. Where we are (the measured starting line)

- **Write-once-C is calibrated.** The grind kernel (`occ_kernel_grind.s`) — which *is* the
  rowblk-owner shape (1 wave/WG, full-K accumulate in VGPR, one `global_store`) — peaks at
  **1.5 TF at pool=64** on 576×512×2048, 2×4 tile, oracle-clean. That is ~2× over split-K's
  shape-invariant ~0.7 TF. Write-once-C is a real but **bounded ~2×** lever at our size (the
  earlier "10×" was a large-shape artifact: grind 6.9 TF at 1920×2048×2048).
- **The wall at our size is NOT memory bandwidth.** Back-of-envelope: 0.813 ms for ~1.2 GFLOP,
  moving ~10–30 MB → ~30 GB/s against a card that does hundreds. We are **occupancy/latency-bound**:
  at the 1.5 TF peak, occupancy is **0.5 waves/SIMD (3.1%)**. A lone wave streaming a full-K
  accumulator has almost nothing to overlap its load latency against.
- **Stagger is aimed at exactly this wall:** raise *effective* occupancy so load-latency stalls are
  hidden, without paying more than average VGPR footprint per wave.

## 2. Thesis: temporal register multiplexing (the traveling peak)

Governing rule (kmbandy / MAD-305): **the sum of all resident waves' *instantaneous* VGPR
allocations must stay ≤ the per-SIMD budget B at every instant.** If a compute wave's fat period is
*brief* (a trapezoid, not a square wave), its time-averaged footprint ≈ peak/2, so you can pack ~2×
as many resident waves into B as their peak would allow. Fatness becomes **temporal, not
simultaneous** — this is CDNA "ping-pong" expressed through `s_alloc_vgpr`.

**Precondition = short bursts.** A full-K accumulator held at peak for the entire K-loop is a
~95%-duty **square wave**: avg ≈ peak, staggering buys nothing, and high pool oversubscribes the
per-SIMD file → the classic grow-spin deadlock. This is why the earlier "stagger" attempt (full-K,
phase-offset independent waves, no feedback) **died** — lockstep re-synchronized it. Split-K short
bursts create the **trapezoid duty cycle** that makes both the packing *and* the stagger-maintaining
feedback (§5) physically possible. Split-K is not a side optimization; it is the precondition.

## 3. Substrate: split-K short bursts → per-rowblk LDS accumulator → write-once C

The compute path changes from "one K-segment, atomic-flush to C" to a **WG-local reduction that
writes each C cell once** — and the pin unit is a **rowblk-group that shares each B fetch** (NOT
serialized rowblks, which would cost `G×` B traffic and is rejected):

- **Enumeration change:** the global work counter `occ[20]` enumerates **`(mblk,tcol)` tiles**, not
  `(mblk,tcol,ksi)` segments. A WG pins a tile and walks `ksi = 0..n_kseg-1` internally (`SL_KSI`
  cursor). The host terminal count drops the `n_kseg` factor.
- **Rowblk-group sharing B (the reuse-preserving core):** a **burst = one K-segment computed across a
  group of `g` rowblks that share the single `B[ksi]` fetch**, each rowblk accumulating into its own
  **LDS bank** (`ACC_N = g` banks, 8 KB each). This keeps the *segment-outer* ordering that gives
  today's B-reuse: `B[ksi]` is fetched once and consumed by all `g` rowblks. `g = G` (=6) = **identical
  reuse to the current kernel, zero regression**. Within a burst the wave iterates the `g` rowblks
  (WMMA into VGPR → `ds_add_f32` into that rowblk's bank → next), so VGPR peak is **one** rowblk's
  accumulator while the `g` banks live in LDS. Then the wave shrinks — the burst is still just one
  segment of K (brief → trapezoid → staggerable).
- **Write-once:** after the `n_kseg`-segment walk, each of the `g` banks holds a rowblk's full-K sum;
  each is `global_store`d **once**. C-write traffic drops `n_kseg`× with **no loss of B-reuse**.
- **Reduction primitive: `ds_add_f32`** (confirmed present on gfx1201: encodings `D8540000` /
  `ds_add_rtn_f32 D9E40000`; final confirmation is the oracle on a GPU run). A compute wave's burst:
  `try-grow → WMMA one K-segment into VGPR acc → ds_add_f32 into the 8 KB LDS rowblk accumulator →
  shrink`. The **last** segment to complete the rowblk (its done-counter hits `n_kseg`) emits **one
  `global_store`** of the accumulator to C — no atomics on the C-write, no per-cell amplification.
- **Fallback A (deterministic):** if `ds_add_f32` misbehaves on hardware, fall back to a
  frontier-ordered `ds_store`+`v_add` reduction (segment `s` adds after `s−1`, gated by a counter —
  reuses DRAIN_HEAD machinery). A is bit-exact; B is not (FP add non-associative) — but B is **not a
  determinism regression**, since today's `global_atomic_add_f32` is already unordered.
- **Bursts, not epochs.** In the current flow kernel a compute wave grows *once* on adopting the
  compute role and stays fat across many rowblks. For the trapezoid we need grow/shrink **per burst**
  (per K-segment, or per short window of K-segments). Burst length is the master knob (§5).
- **Accumulator lifecycle:** on claiming a rowblk, zero its 8 KB LDS accumulator (`ds_store` zeros,
  or first-segment-writes-instead-of-adds); each segment `ds_add_f32`s its partial; the completer
  (`RBDONE == n_kseg`) `global_store`s once, then the accumulator region is recycled for the next
  rowblk. Codex's forward-progress rule holds: the completer must observe completion only *after* all
  prior segments have drained their DS ops (`s_wait_dscnt 0` before the done-increment).

**LDS budget (a real constraint, not a footnote).** `ACC_N = g` = the rowblk-group size = number of
banks that share each `B[ksi]` fetch. **Target `g = G = 6` for full B-reuse parity.** The banks have
**rowblk-group lifetime** (live across the whole `n_kseg` walk), longer than the **segment lifetime**
of operands. Budget: `OP_BASE + operands + g*8192 ≤ 65536`. At `g=6` the banks alone are 49152 B, so
the operand region must be **lean**: stage `B[ksi]` (4096) once per segment, and **stream A
per-rowblk** (`FM` frags = 2048) rather than staging all `g` rowblks' A at once (12288). That gives
`256 + (4096 + 2048) + 49152 = 55552 ≤ 65536` — `g=6` fits with room, full reuse, no double-buffer on
A. Alternatives if the A-stream stalls compute: `g=5` (40960 banks + a 16 KB double-buffered operand
group = 57600, B re-fetched 1.2×) — a clean fallback that still keeps ~full reuse. The exact operand
packing (A-stream vs small double-buffer) is **build-stage-1 sizing**, driven by whether A-streaming
starves the burst; `g` itself stays at `G` unless the packing forces `g=5`.

## 4. The envelope: coast-on-grow-fail (no per-wave poll)

The per-SIMD budget is enforced by the **hardware allocator itself**, not by any software counter a
wave reads. `s_alloc_vgpr` grow sets **SCC** (SCC0 = allocation failed). Today the kernel *spins* on
failure (`s_cbranch_scc0 .Lca_alloc` — the unbounded grow-spin the deadman does **not** cover; the
reason we're pinned to M=576). We replace the spin with **coast**:

- **Try-grow; on SCC0 (budget full) → COAST** (run lean feed code, do useful work) and return to the
  loop head to try again next lap. No shared budget counter, no reservation, no wait-for-permission
  poll. The hardware allocator *is* the floodgate; coast *is* the water flowing around it.
- **This is explicitly NOT a reserve-or-wait gate.** A software "reserve your delta, wait if the
  counter is full" scheme re-introduces the published-value poll (FOLLOW_WAIT) the flow economy
  removed. Rejected for that reason.
- **Invariant (load-bearing): commit nothing until the grow succeeds.** Claim the rowblk *after* the
  grow; flip any role/population state *after* the grow. On SCC0 the wave has touched **zero** shared
  state — a failed grow is a complete no-op to the accounting. Ordering is rigid:
  `try-grow → (SCC0) coast, nothing committed` / `(SCC1) now claim + WMMA + count`.
- **Closes the brick gap.** Because coast returns to the loop head — where the deadman watches — the
  former uncovered grow-spin becomes a deadman-covered graceful coast. **Stagger and scale-safety are
  the same change.**

## 5. Maintaining the stagger: a self-organizing equilibrium

We do **not** launch staggered and hope it sticks (that is precisely what the dead attempt did;
lockstep is an attractor and it re-synchronized). Instead the coast-on-fail gate is a **restoring
force** that creates and holds the stagger:

- When two waves drift toward the same phase (both want peak at once), the SIMD budget can hold only
  ~k fat waves; the (k+1)th's grow **fails → it coasts → it is time-shifted away** from the collision.
  The gate **repels coincident peaks.** The system self-organizes into "≈k fat at staggered phases"
  and *stays* there, because every attempt to clump triggers a re-spacing coast.
- **The stagger is maintained dynamically by the gate, not preserved from launch.** A launch-time
  phase offset (stagger each wave's first grow) is a **warm-up** that speeds convergence and avoids an
  initial thundering herd — but it is not the mechanism.
- **Burst length is the master knob.** Long burst = better matrix efficiency + amortized grow cost,
  but higher duty → weaker stagger (toward the dead square-wave). Short burst = max stagger but more
  grow/shrink overhead and needs more concurrent waves to fill the WMMA pipe. **Optimum = shortest
  burst that keeps WMMA saturated.** We sweep it.
- **Convergence quality is empirical.** The gate guarantees no over-budget and provides the restoring
  force, but whether it settles into a good equilibrium (pipe-saturating, low thrash) vs. a mediocre
  one (collide-and-coast oscillation, or too few fat to fill the pipe) is measured, not assumed (§8).

## 6. Accounting consistency under coast

The flow economy's hard state tracks **work done, not intended roles.** Frontiers and done-counters
(`RBDONE`, `BFDONE`/`ARDONE`, ASSIGN/STAGE/DRAIN heads) are incremented by *whoever actually does the
work*. The ROLE mailbox is a soft **bias**, never a contract. Therefore a compute-biased wave that
coasts (grow failed) is fully consistent:

- It didn't bump the compute counter — truthful (it didn't compute). If it fed while coasting, it
  bumped feed counters — truthful. **No lie enters the ledger.**
- The rowblk it would have computed stays unclaimed (claim happens *after* grow, §4); the DRAIN
  frontier simply doesn't advance for it until *someone* computes it. Nothing lost or double-counted.
- **Self-heals on survey:** the coordinator reads actual progress off the frontiers and re-biases the
  mailbox. Coast is transparent to accounting except as "no compute progress from that wave this lap,"
  which is the truth.
- **Survey tuning (control, not correctness):** the coordinator must not misread budget-*saturated*
  compute lag as "assign more compute waves" (that would starve feed — a milder cousin of the old
  runaway). The survey should look at **budget saturation** (grow-fail rate, concurrent-fat count),
  not raw compute lag. Coast makes a mis-bias *safe* (the extra compute-biased wave just coasts), but
  we design the survey to see saturation.

## 7. The handoff gap (coast-on-fail's one real cost)

`s_alloc_vgpr` allocates registers free *at that instant*; it cannot reserve against a neighbor's
*future* shrink. So a wave that calls grow a hair before the fat wave shrinks fails and coasts. This
is handled correctly and does **not** waste the slot — **provided coast is a short useful detour with
a prompt retry:** the wave feeds one small unit, retries grow, and catches the shrink a beat later;
the freed registers get reused. The cost is a small **utilization gap** at each handoff (the window
between a shrink and the next wave's next grow attempt).

- We **do not** pre-engineer this away with a proactive "shrinking-now" handshake — that re-introduces
  the published-signal poll and is fragile (mis-timed → over budget anyway). YAGNI: coast-on-fail
  first, **measure the gap** (§8), build a tighter handoff only if data demands it.
- **Coast granularity is the gap dial:** one small feed unit per retry — enough to be useful and not
  spin (spinning on grow burns issue cycles and contends with the very waves it waits on), short
  enough to catch the shrink window. Swept alongside burst length.

## 8. Instrumentation (required from the start, not bolted on)

We must *see* the equilibrium form. `PHASEPROBE`-style, occ-buffer atomics, gated (zero bytes when
off):

- **Concurrent-fat-wave count** — each grow `atomic_inc`s, each shrink `atomic_dec`s; `atomic_max`
  captures the peak; a time-integral gives average concurrent-fat. The direct readout of "is the
  stagger working" (≈k steady vs. clump-to-peak-then-zero lockstep) and of "sum vs. budget."
- **Grow-fail / coast rate** — count SCC0 grows and cycles-coasted-before-success. This *is* the
  handoff-gap measurement (§7).
- **Fat duty cycle** — fraction of a compute wave's life spent fat (the trapezoid shape; warns if we
  drifted back toward the square-wave).
- **Confirm per-SIMD budget B** — never directly measured on gfx1201 (MAD-305 R0). Pin it down
  empirically (grow until fail from a known-idle SIMD); every packing calc depends on it.

## 9. Correctness, oracle, safety

- **Oracle-gated every run.** Expect bit-exact-to-tolerance (grind's tight tier rel 5e-3 / abs 1e-2).
  B's reduction is unordered (non-associative FP) — same acceptance basis as today's atomic flush.
- **Deadman now covers the former grow-spin** (coast returns to loop head). Scale-stress becomes safe
  in a way it wasn't: this is a safety *improvement*.
- **GPU runs are individually greenlit** by kmbandy; logs to real disk `/home/kmbandy/dsws_gpu_logs`,
  never tmpfs. Single-shot first, then sustained (`DSWS2_TARGET_SECS`) with greenlight. Keep
  SAFEPROBE + bounds gate; never `--gl2c`.

## 10. Build stages (incremental, measure between)

1. **Write path first:** convert the flow compute body to split-K short burst → `ds_add_f32` into an
   8 KB per-rowblk LDS accumulator → single `global_store` on rowblk completion. Grow/shrink still
   per-role-epoch (no stagger yet). Oracle-verify write-once at 576×512×2048; confirm it matches the
   grind's write-once number (~1.5 TF) inside the economy. **Gate: bit-exact + not slower than split-K.**
2. **Per-burst grow/shrink + coast-on-fail:** move grow/shrink to per-burst; replace the grow-spin
   with coast-on-SCC0; add the launch-offset warm-up. Add the §8 instrumentation. **Gate: no wedge,
   deadman-clean at scale; instrumentation shows concurrent-fat > 1 and a real duty cycle.**
3. **Tune the equilibrium:** sweep burst length and coast granularity; read grow-fail rate, concurrent
   -fat vs. budget, duty cycle; find the shortest burst that keeps WMMA saturated. **Gate: TF beats the
   1.5 TF write-once baseline; occupancy (concurrent-fat) meaningfully > 0.5 waves/SIMD.**

## 11. Success criteria & open risks

**Success:** at 576×512×2048, 2×4 tile, staggered write-once beats the 1.5 TF write-once baseline
with concurrent-fat-waves > the grind's 0.5/SIMD and a healthy trapezoid duty cycle, oracle-clean, no
wedge. (Tile size is the *next* foundational step and is held fixed here.)

**Risks / open questions (measure, don't assume):**
- **Feed may not keep N fat waves fed.** If the feed path (operand staging) can't supply more
  concurrent compute waves, stagger raises occupancy but not throughput. The instrumentation will show
  compute waves starving on operands even when fat — if so, the lever is the feed, not occupancy.
- **The gate may settle into a mediocre equilibrium** (collide-and-coast thrash, or under-fill).
  Burst length + coast granularity are the dials; §8 tells us which way to turn.
- **Per-SIMD vs. per-WG budget placement.** Waves in a WG spread across SIMDs; the gate is per-SIMD
  (hardware). Confirm the economy's compute-wave count doesn't systematically overload one SIMD.
- **`ds_add_f32` hardware behavior** — assembler-accepted; oracle on real hardware is the proof. A is
  the fallback.
