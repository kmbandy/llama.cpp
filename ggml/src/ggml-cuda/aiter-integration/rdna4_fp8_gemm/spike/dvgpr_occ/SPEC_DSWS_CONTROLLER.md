# Spec: DSWS Adaptive Wave-Role Controller (MAD-305)

Date: 2026-06-27. Substrate: `occ_kernel_coop.s` (dyn-VGPR grow/shrink + split-K +
POOLTERM cooperative kernel, raw-PM4 dispatch on gfx1201/RDNA4 wave32).
Companions: [[MAD305_DSWS_MASTER.md]] (campaign master), [[SPEC_WAVESPEC.md]]
(static wave-spec lineage). Prior-art verdict (Murmur `5ec8a958`, KG `2dc2332d`):
the runtime-adaptive rebalancing is **novel**; the lean/fat mechanism is borrowed.

## Goal

One fp8 GEMM kernel that **senses its in-kernel bottleneck and rebalances its mix
of wave roles at runtime** to attack the measured wall (the VALU **issue port**:
~31 non-WMMA issues per 32 WMMA = 52% of the 307 TF ceiling). The lever is
wave-specialization — put feed/address instructions on *separate* waves so compute
waves issue near-pure WMMA — with the feed:compute split **found by the kernel per
shape, per moment** instead of hand-tuned and frozen at launch.

This spec covers the **controller** (the novel "brain"). The static 3-role
substrate it sits on is a prerequisite build phase (Phase 1 below).

## Scope decisions (settled in brainstorming, 2026-06-27)

- **3 roles** from the start: `compute` (fat VGPR, holds fp32 accumulators),
  `A-feed` (lean), `B-feed` (lean). Rationale: tests the actual novel thesis (the
  multi-role economy), and on the `coop` substrate **both** A-feed and B-feed
  relieve the issue-port wall — coop currently has compute load its *own* A direct,
  so an A-feed role offloads those A-load issues off the compute stream. The
  control-law code is role-count-parametric, so 3-role-first avoids throwaway.
- **Data path fixed for v1** (no runtime A-in-LDS-vs-direct switching). That is a
  legitimate *future* second control axis (slow/structural cadence, richer sensing)
  but it makes compute's hot loop branchy, which fights the issue-port wall.
- **Feed floor = 1; compute floor = 1.** Feeds can be driven down to their last
  wave (throw everyone else at the bottleneck) but not to 0 — floor-0 would force a
  dual-path (fed-vs-direct) branch into compute's K-loop, adding non-WMMA issues on
  the exact waves we're trying to keep pure. Floor-0 is bundled with the future
  data-path-switch extension.
- **Build sequencing:** static 3-role substrate oracle-green **first**, then layer
  the controller. Do not couple "new A-feed role" and "new control law" in one step.

## Architecture — the role economy

A workgroup launches a **fixed N waves**. Each wave is in exactly one role at any
instant. The controller governs only the **partition**
`(n_compute, n_Afeed, n_Bfeed)` with `n_compute + n_Afeed + n_Bfeed = N`. The wave
count never changes; only the partition moves.

State = three atomic LDS **role-count slots** (`n_compute`, `n_Afeed`, `n_Bfeed`) —
the single source of truth for the current mix. A conversion is one wave atomically
moving itself between slots, then physically re-roling (grow/shrink + switch job
loop), **only at a split-K segment boundary** (the one point a wave is safe to
change: partial flushed, accumulators released).

Key property: **the controller moves the *target partition*; waves migrate
themselves toward it at their next safe boundary.** This is what keeps it lock-free
and barrier-free — no wave waits on another, no rendezvous.

Invariants (always hold):
- Floors: `n_compute ≥ 1`, `n_Afeed ≥ 1`, `n_Bfeed ≥ 1`.
- Sum-envelope: `Σ instantaneous VGPR alloc < per-SIMD budget`, enforced at
  grow-time (below), never by the controller's arithmetic.

## Sensing — ring pressure

Each ring (A-ring, B-ring; depth `RINGD`) is a supply/demand **pressure gauge**.
At its decision boundary a wave reads occupancy from LDS:

    occ_X = prod_count_X − min_cons_X        (clamped to [0, RINGD])

Watermark bands give hysteresis (the **spatial** damping):
- `occ_X < LOW`  → ring draining empty → **starved** for X.
- `occ_X > HIGH` → ring backing up full → **over-served** for X.
- `LOW ≤ occ_X ≤ HIGH` → dead-zone, no action.

Both conversion directions read the *same* gauge from opposite sides:
- a **compute** wave (consumes from ring X) acts on `occ_X < LOW`;
- a **feed-X** wave (fills ring X) acts on its own `occ_X > HIGH`.

## Control law — conversion, both directions

**compute → feed-X** (compute is starved for X):
- Boundary: split-K **segment** boundary (compute holds accumulator state).
- Guard: `occ_X < LOW`, win the epoch ticket for this direction (below),
  `n_compute > 1` (floor).
- Actuation: CAS-decrement `n_compute`, increment `n_feedX`, `s_alloc_vgpr` **SHRINK**
  to lean (a shrink *always* succeeds), enter the feed loop.

**feed-X → compute** (feed-X is over-serving):
- Boundary: any inter-frag point (a feed wave holds no accumulator state → far more
  frequent safe boundaries), gated to the epoch.
- Guard: `occ_X > HIGH`, win the epoch ticket, `n_feedX > 1` (floor).
- Actuation: CAS-decrement `n_feedX`, increment `n_compute`, `s_alloc_vgpr` **GROW**
  to fat — *gated on the sum-envelope* (below). On grow-fail: **abort cleanly**
  (undo the slot move + reservation, stay feed this epoch). On success: enter the
  compute loop (claim a K-segment, accumulate).

Designed asymmetry (the good kind): the cheap-to-move role (lean feeds) reacts
**fast**; the expensive-to-move role (fat compute) reacts at **segment** granularity.
Shrink-always-succeeds / grow-can-abort means budget is *freed* promptly but
*consumed* only when proven-available.

## Epoch + ticket — lock-free single-winner per window

**Epoch** = a clock derived from work progress (per-workgroup):

    E = (segments_processed >> EPOCH_SHIFT)

ticks every `2^EPOCH_SHIFT` segments. `EPOCH_SHIFT` is the **decision-cadence knob**
(small = reactive, large = damped). No one "advances" it; it's a function of
throughput.

**Ticket** = one LDS `gate[d]` per conversion direction `d` (4 directions:
compute→Afeed, compute→Bfeed, Afeed→compute, Bfeed→compute), each holding the last
epoch in which `d` fired. To convert in direction `d` at epoch `E`:

    g = gate[d]
    if g >= E: back off                              # d already fired this epoch
    else:      won = (atomic_cmpswap(gate[d], g, E) == g)
               if won: proceed   else: back off

Exactly one wave's CAS succeeds when many race the same `g < E`. **≤1 conversion per
direction per `2^EPOCH_SHIFT` segments**, per workgroup — the **temporal** damping.

This one CAS does triple duty: (1) anti-thrash, (2) the **phase-stagger** the
rolling-dyn-VGPR thesis needs — grows are automatically spread across epochs, never
simultaneous — and (3) keeps the sum-envelope satisfiable by construction (you can
never get N waves spiking VGPR at once).

Combined damping: a feeder can't flip back to compute until **both** its ring exceeds
HIGH (spatial) **and** a fresh epoch ticket is available (temporal). Tuning surface =
`{LOW, HIGH, RINGD, EPOCH_SHIFT}` — small and interpretable.

Gates + role slots + epoch clock are all **per-workgroup** (in each WG's LDS); every
WG self-balances independently with zero cross-WG coordination.

## Safety & failure modes

- **Sum-envelope, concretely:** one LDS counter `vgpr_reserved`. A feed→compute grow:
  `r = atomic_add(vgpr_reserved, V_fat − V_lean)`; if `r + Δ > BUDGET` →
  `atomic_sub` back and **abort**. compute→feed shrink: `atomic_sub` (always
  succeeds). This reservation correctly arbitrates the ≤2 concurrent grows an epoch
  permits (the atomic serializes them; the second to validate sees the first's
  reservation and backs off).
- **Role floors:** CAS-guarded decrements (compute→feed needs `n_compute > 1`;
  feed→compute needs `n_feedX > 1`).
- **No barrier, by construction:** pure LDS atomics + busy-wait flags (inherits
  coop's `BUSYWAIT`); conversions add zero rendezvous → the dyn-VGPR / §3.3.3.2
  barrier deadlock is structurally absent.
- **POOLTERM terminal must be role-agnostic** (the one genuinely new requirement):
  every role's loop checks the `ti ≥ TOTAL` terminal broadcast at the *same* boundary
  it checks for conversion, so no wave is stranded in a role past drain, and a wave
  that converts then sees the terminal exits immediately.
- **No livelock:** a full envelope just makes feed→compute grows keep aborting — the
  wave stays feeding (productive); the mix holds at a safe, correct partition. No
  state spins doing nothing.

All failure modes degrade to "stay in current role, keep doing useful work" — never
to a brick, never to a stall.

## Testing & verification

- **CPU oracle = correctness ground truth, under ANY schedule.** DSWS is a pure
  *performance* transform — the adaptive mix must never change the math. Gate: for
  any (shape, mix, conversion schedule), stored C matches the fp8 e4m3 oracle.
  Split-K flushes fp32 partials via atomic-add (nondeterministic ordering) → gate
  with an fp32-reorder **tolerance** or a deterministic-accumulation gate mode
  (carry coop's existing choice — resolve in planning).
- **Two-gate sequencing:**
  - *Gate 1 — static 3-role green:* fixed `(n_compute, n_Afeed, n_Bfeed)`, no
    controller, oracle-clean across shapes + several hand-set mixes.
  - *Gate 2 — dynamic green:* same oracle, conversions firing.
- **Conversion-storm race stress:** oracle + many repeats under adversarial tuning
  (tight watermarks + tiny `EPOCH_SHIFT` → maximal conversion rate) to expose
  cross-wave races the strong oracle catches and `acc00` would miss.
- **Prove it adapts (not just runs):** instrument role-count slots + conversion
  counters into the occ-snapshot stream. (a) asymmetric shape → mix moves the right
  way and settles; (b) **start from a deliberately wrong mix** (e.g. all-compute on a
  feed-bound shape) → controller converges to a better partition **and TF climbs**.
- **Success metric (v1)** on ml8 `down` (M=2048 K=9216 N=2560) / `down_pf` (M=512):
  (a) oracle-correct, (b) converges to a mix that beats the static 3-role baseline
  AND the 165 TF static winner, (c) demonstrably adapts when shape changes —
  confirmed with `--att` showing **cut non-WMMA issues on the compute waves**.
- **Supervised GPU discipline:** display GPU = brick risk. Compositor-safe chunking +
  per-chunk hang-abort; oracle (STORE=1) before perf (STORE=0); **one gated dispatch
  at a time, no sweeps until proven safe**; stream to disk for brick forensics; user
  greenlights each dispatch. Never `--gl2c`; SAFEPROBE + bounds gate stay on.

## Build phases (high level; detailed plan is the next step)

1. **Static 3-role substrate** on coop: add the A-feed role (A-ring in LDS, A-feed
   waves) — *port* the proven A-LDS-share pattern from `occ_kernel_wggemm2.s` rather
   than invent it. Fixed mix. **Gate 1** oracle-green.
2. **Sensing + role slots:** ring-occupancy reads, watermark bands, the LDS
   role-count slots + `vgpr_reserved` reservation counter (no conversions yet —
   read-only sensing + a STATIC mix still). Verify the sensors report sane occupancy.
3. **Conversion + epoch/ticket:** the CAS gate, both-direction conversion with floor
   + envelope guards, role-agnostic POOLTERM terminal. **Gate 2** oracle-green +
   conversion-storm stress.
4. **Adaptivity + tuning:** instrument role counts; converge-from-wrong-start proof;
   sweep `{LOW, HIGH, RINGD, EPOCH_SHIFT}`; measure on ml8 `down`/`down_pf` with
   `--att`. RESULT doc + KG bank.

## Open details to resolve in planning

- Oracle accumulation determinism vs tolerance (carry coop's choice).
- Exact source/owner of the per-WG `segments_processed` counter feeding `E`.
- A-feed ring sizing / LDS budget at the chosen tile (must fit A-ring + B-ring +
  counters within 64 KB; confirm not occupancy-binding since we're not occupancy-maxxing).
- Fat/lean VGPR footprints `V_fat`, `V_lean` and the per-SIMD `BUDGET` constant
  (from RGA on the static substrate).
- Feed wave's exact inter-frag check cadence (every frag vs every K-window).

## Pointers

- KG: `2dc2332d` (prior-art verdict), `dac0bb8c`/`0a2cea44` (POOLTERM), `5fedf098`
  (rolling-peak origin), `dc9faf0d`/`63583120` (DSWS v1/v2), `8a9ce97f`/`17f209af`
  (dyn-VGPR OOR-temp deadlock fix — the barrier-free precedent).
- Master: `MAD305_DSWS_MASTER.md` §5–§6. Jira epic MAD-305. Tasks #323/#324.
