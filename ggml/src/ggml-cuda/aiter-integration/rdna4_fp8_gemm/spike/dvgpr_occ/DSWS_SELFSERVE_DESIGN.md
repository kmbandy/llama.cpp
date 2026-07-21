# DSWS Self-Serve — Design (2026-07-19)

**The last coupling.** Break compute's dependence on a resident ring slot so the *already-built*
dyn-VGPR traveling-peak economy can finally bind. This is the fuel line, not a new engine.

Sibling docs: `DSWS_STAGGER_DESIGN.md` (the traveling peak this feeds), `FLOW_ECONOMY_DESIGN.md`
(the mailbox/role economy), `RING_SLOTS_DESIGN.md` (the pool this demotes to a fast-path),
`DSWS_GRESIDENT_DESIGN.md §6` (the ksi-run burst body this reuses).

---

## 1. The starting line (measured — the LDS-split frontier, 2026-07-18/19)

Full `SEGK × POOL_N × ACC_N` frontier at real ml8 attn_q (19200×4096, big-M fed ≥3.2s, clock-committed,
every cell oracle-gated `bad=0`). Peak = **SEGK=256 / ACC_N=3 / POOL_N=1 / GROUPS=2 = 8.2 TF** (~2.7%
of the 307 TF fp8 peak). Two load-bearing facts hold across **all 18 cells**:

- **`door1` (nothing-staged) = 100.0% of coast in every cell.** No LDS knob moves it. Compute is always
  starved of staged operands.
- **`grow-fail = 0` everywhere** (door4). The per-SIMD VGPR budget never binds → the dyn-VGPR moat and
  the stagger/traveling-peak **never engage**. `occ98` baton-wait = 0 (the peak never travels).

The three knobs only *redistribute* the 64KB LDS: SEGK sets flush-amortization + wave peak-duty (the
lever); POOL_N sets pipeline depth (weak; LDS-capped to 1 at SEGK=256); ACC_N/GROUPS trades bank
footprint for the SEGK it unlocks. **None of them feed faster, and none of them make the budget bind.**

## 2. Diagnosis — why the engine is cold

The whole DSWS machine is present and idle:
- **split-K short bursts** (SEGK=256 ≈ 13% peak-duty = trapezoid, the stagger precondition) ✓ built
- **banked `ds_add_f32` reduce** + **write-once C** + **completer C-store** ✓ built
- **per-burst dyn-VGPR grow/shrink** (the moat) ✓ built
- **stagger admission token + traveling-peak baton** (`occ_kernel_dsws_flow.s:596`) ✓ built
- **B parked hot in L2/L3** — every B re-access is an L3 hit, never HBM (`DSWS_GRESIDENT_DESIGN.md §5`) ✓
- **DECENTASN decentralized assign** ✓ built

It does not run because of **one coupling**: *a compute wave may only consume a resident ring slot.*
The compute tier reads operands from `OP_BASE + slot*OPSTRIDE` (`.Lflow_havestage`, ~2802); at SEGK=256
the LDS fits **POOL_N=1**, so only ~ACC_N waves are ever fat, `Σ VGPR ≪ budget`, the budget can't bind,
grow-fail stays 0, and the traveling peak has nothing to travel. When the single slot isn't staged,
compute **coasts** (`.Lflow_compute` → `door1` → `.Lflow_coast`, lines 2783–2786). 98% of wave-life is
that coast. The code says it verbatim (line 598): *"grow-fail was 0 because ACC_N==G capped the number of
waves that could EVER be fat → the budget was STRUCTURALLY UNABLE to bind."*

## 3. Thesis

Make **"ring empty" mean "self-serve compute," not "coast."** A wave that finds nothing staged claims a
work-item directly, self-loads its own A/B **from L2/L3** (cheap — B is resident by construction), grows,
WMMA-accumulates in VGPR, `ds_add`s into the shared bank, shrinks. The ring stays as an *opportunistic
fast-path* (staged slot = zero-latency operands) refilled during slack; it is no longer a gate.

Because self-serve holds operands in **VGPR, not a shared LDS slot**, in-flight parallelism becomes
**wave count (30), not slot count (1)** — with no change to the LDS budget. That is what finally puts
enough waves fat at once for the budget to bind and the stagger admission token to do its job.

## 4. Design — the coast→self-serve tier

Insert a tier between `compute-from-slot` and `coast`, on the `door1` fallthrough:

```
.Lflow_compute:
    dh = DRAIN_HEAD ; sh = STAGE_HEAD
    if dh < sh:  → .Lflow_havestage   (FAST PATH: operands pre-staged in slot, unchanged)
    else:        → .Lflow_selfserve   (NEW; was: cnt_inc door1 ; s_branch .Lflow_coast)

.Lflow_selfserve:                      (NEW)
    claim (tile,rowblk,ksi) single-winner via the SAME SL_RBNEXT fetch-add path as the ring
    if no claimable work (frontier exhausted / boundary in progress): → .Lflow_coast   (real idle)
    acquire fat token (existing stagger admission); token full → .Lflow_coast (go feed) [unchanged rule]
    grow (existing per-burst s_alloc_vgpr)
    self-load A(rowblk) + B(ksi) from global → VGPR   (global_load_tr idiom from BSTAGE, NO ds_store)
    WMMA the K-segment into VGPR ACC
    ds_add_f32 ACC → rowblk's shared bank   (existing banked reduce)
    bump SL_RBDONE ; if == n_kseg the completer stores C once  (existing write-once path)
    shrink ; re-loop
```

Key point: **the work-item claim is decoupled from the operand source.** Ring-served and self-served
waves both advance the *same* `SL_RBNEXT`/`SL_RBDONE` counters, so a tile is computed exactly once
regardless of who staged its operands.

**Granularity (a real choice):** self-serve at *rowblk* granularity means g waves reload the same
segment's B (g× L2 reads — cheap, cache-hit, but not free). Self-serve at *whole-ksi-run* granularity
(one wave grinds all g rowblks of a segment, reusing B in-register) = `GRESIDENT §6` burst body, zero
redundancy, fatter wave. **Start rowblk-granular** (smallest delta from today's compute body); measure
L2 pressure; promote to ksi-run only if L2 read bandwidth shows up as the new wall.

## 5. Correctness

- **Concurrent self-servers `ds_add` a shared bank — safe by the H1 model** (`DSWS_GRESIDENT_DESIGN.md
  §7`): banks zero-init at rowblk-claim + *every* merge is `ds_add_f32` ⇒ order-independent ⇒ correct
  under any number of concurrent same-rowblk adds. `s_wait_dscnt 0` before the done-increment
  (completer observes completion only after all prior segment DS ops drained) — the existing rule holds.
- **Single-compute guarantee:** claim via `SL_RBNEXT` fetch-add (single-winner); ring vs self-serve is
  only *where operands came from*, never *whether the item is claimed*. No double-compute, no dropped work.
- **Gate:** full stride=1 oracle `bad=0` + `computed == G*MTLsuper*NTL*n_kseg` (work-exact) at a bounded
  shape, at POOL_N=1 (self-serve is the only path) AND POOL_N=2 (both paths live) — same discipline that
  caught the POOL_N=4 GROUPS=1 race this session.

## 6. Stagger / dyn-VGPR interaction (why this is the fuel line)

Self-serve is what the stagger was *built for* and starved of. With 30 waves able to go fat on demand:
the admission token (`FATTOK`, line 617) starts refusing (waves coast→feed when the peak is full), the
baton hands the peak wave-to-wave (`GROWPERMIT`), average footprint (not peak) fits the budget, and the
physical `s_alloc_vgpr` grow-fail stops being 0. **No new stagger code** — it engages by getting a
population. (Precondition already satisfied: SEGK=256 short bursts = trapezoid, not plateau.)

## 7. The measurable gate — the "floodgates" fingerprint

This is a hypothesis with a signature, not a declared win. On the SAME shape (peak config), self-serve
is validated **iff** all four move together vs the 8.2 TF baseline:

1. `grow-fail` (door4): **0 → large** (budget binds — the code predicts ~10⁵-scale).
2. `door1` nothing-staged: **100% → materially < 100%** (waves compute instead of coasting).
3. baton `occ98` / fat-population: **0 → > 0** (the peak travels).
4. **TF > 8.2**, ideally by a lot.

If grow-fail stays 0 or door1 stays pinned → self-serve did NOT open it; report the actual next coupling
(measure it, do not eulogize a wall). No premature verdict either direction.

## 8. Build order (gate-defined; each step states its exact gate)

- **S0 — offline scaffolding.** New defsym `SELFSERVE` (default 0 → byte-identical to today). Gate:
  `SELFSERVE=0` bin sha == current canonical `8a7c50f9`.
- **S1 — the self-load compute body.** Add `.Lflow_selfserve` (claim + self-load + WMMA + ds_add), gated
  `.if SELFSERVE`. Reuse BSTAGE's `global_load_tr` idiom minus the `ds_store`. Gate: assembles 0-spill,
  RGA livereg ≤ budget, `SELFSERVE=0` still byte-identical.
- **S2 — adversarial review** (Codex, no prior-review contamination): the claim/ds_add race under
  concurrent self-servers + the ring↔self-serve claim-coupling. Same gate the DECENTASN v3 got.
- **S3 — [SUPERVISED GPU] correctness bring-up.** Bounded K, full stride=1 oracle, POOL_N=1 and =2.
  Gate: `bad=0`, work-exact, no DMFAT, no latch.
- **S4 — [SUPERVISED GPU] the fingerprint run.** Peak config, fed ≥2s. Gate: report the §7 four-tuple.

## 9. Risks / open questions

- **L2 read bandwidth** becomes the candidate new wall (rowblk-granular self-serve re-reads B). That is a
  *higher, more real* wall than depth-1 serialization — and the ksi-run granularity (§4) is the lever if
  it bites. Watch HBM traffic on the first run (Rule 7: small chunk first; L3-resident B should keep HBM
  flat, but A loads + sheer parallelism could raise it — do not run full-scale on the first dispatch).
- **Grow-stagger deadlock** (ISA §3.3.3.2) is the *reason* stagger exists; self-serve creates the fat
  pressure that makes it load-bearing. If the admission token is mis-sized the peak can oversubscribe —
  the token cap (`MAXFAT`) is the knob; start conservative.
- **Refill policy** for the demoted ring (who stages the fast-path, how often) — start with the existing
  feed tier unchanged (feeds keep staging into the pool); self-serve is purely additive on the coast edge.
## 10. S1 implementation recipe (code-level — nailed 2026-07-19 from reading the compute body)

**Discovery that reshaped S1:** the existing compute half is NOT reusable as-is — it is welded to the
staged slot. The claim (`poison-until-staged`, occ_kernel_dsws_flow.s:2892) reads `SL_RBNEXT`/`SL_STI`
FROM the slot, and the WMMA reads operands FROM resident LDS (`BRES`/`ARES`). So self-serve needs a
slotless claim + a self-loading WMMA. The banked reduce + completer, however, ARE reusable.

**What STAYS (do NOT touch):**
- **Tile pinning.** A WG still pins ONE tile (`occ[20]++`/`DA_TILE`), so the `ACC_N` per-rowblk banks
  (`acc_base_of s39,r` = `ACC_BASE + r*ACC_STRIDE`, :3104 — slot-INDEPENDENT) are the pinned tile's G
  rowblk accumulators, shared by all self-serve waves. This is why self-serve waves work the SAME tile
  concurrently and ACC_N=G=6 banks suffice (not one-per-wave).
- **The ds_add reduce** (:3113-3121): `ds_add_f32 v12, v[ACC+..]` into bank r. Order-independent under
  BANKZERO (H1). Concurrent self-servers adding different ksi of the same rowblk r = safe.
- **The completer** store-once-per-rowblk-when-done (the `.Lflow_cmp_tryadv` C-store path). Reuse it.
- **grow/shrink + fat token** (:2833-2861): a self-serve wave acquires the fat token then grows, identical.

**What is NET-NEW (the two pieces to write):**
1. **Slotless `(rowblk, ksi)` claim + enumeration.** The pinned tile has `G * n_kseg` work-items. Add a
   per-WG monotone counter `SS_NEXT` (new LDS u32 in the control gap below OP_BASE, `.if SELFSERVE` only).
   Claim = `lds_fetch_add SS_NEXT,1` → `w`; if `w >= G*n_kseg` → tile's self-serve work exhausted →
   `.Lflow_coast` (or roll the next tile via the existing DA tile-claim). Decode `ksi = w mod n_kseg`
   (use `s66`=n_kseg-1 / the ceil-log2 radix already in the file), `rowblk r = w / n_kseg`. Single-winner
   by the atomic fetch-add. Needs per-rowblk done counters `RBDONE_SS[r]` (G u32) so the wave that brings
   rowblk r to `n_kseg` fires the ONE C-store (SL_RBDONE is per-slot; add a per-rowblk array or repurpose
   bank-adjacent control words). Self-serve does NOT reserve ASSIGN or stage a slot — it BYPASSES the
   POOL_N cap (that cap is exactly what we're breaking); the ring feed path is untouched, runs in parallel.
2. **WMMA self-load global→VGPR** (replaces the slot `ds_read`). Reuse the global address math from
   `BSTAGE_R`/`ASTAGE_R` (:1411/:1500) — B addr `Bshuf + tcol*(FN*256) + ksi*KSEG_STEPS*(NT*256)` with the
   64-bit `s_mul_hi_u32` fix, A addr for `(mblk,rowblk,ksi)` — but `global_load_tr_b64` straight into the
   WMMA operand frag VGPRs and DROP the `ds_store`. Then run the existing WMMA macro + ds_add.
   **⚠ THE ONE READ THE IMPLEMENTER MUST DO FIRST:** the exact VGPR frag registers the JDEPTH=1 WMMA
   consumes — read the WMMA operand `ds_read`/macro just above the ds_add at :3100 (not traced here). The
   self-load must write those exact regs. This is the single unverified register-mapping detail.

**Hook (:2791):** `.if SELFSERVE`, at the door1 branch (`DRAIN>=STAGE`) replace `s_branch .Lflow_coast`
with `s_branch .Lflow_selfserve`. Everything lives in `.if SELFSERVE` blocks so SELFSERVE=0 stays
byte-identical to `66d76aa0`.

**Register hazards:** drain stores before `s_alloc_vgpr` (register-file corruption); keep
`deadman_progress` on each computed segment; no new message-bus traffic; the DECENTASN head-pin invariants
(occ[95] straddle, post-grow re-derivation) live on the RING path — the self-serve path is separate and
must NOT reuse `s46` as a DRAIN cursor.

**S2 review targets (hand Codex these specifically):** (a) SS_NEXT claim single-winner + the RBDONE_SS
completer election (exactly one C-store per rowblk); (b) concurrent ds_add vs the completer's bank read
(`s_wait_dscnt 0` ordering, mirror Codex C1 from the DA boundary); (c) the WMMA operand register map;
(d) does bypassing the POOL_N cap create any straddle with the parallel ring feed path.

---

## 11. AS-BUILT (2026-07-19) — supersedes §10 where they disagree

§10 was the pre-implementation sketch. Reading the code changed three of its load-bearing choices, and the
S2 adversarial review (Codex, job `task-mrruaejr`) then found five defects in the first draft. This section
is the **as-built** design. Where §10 and §11 conflict, §11 is correct.

### 11.1 Three divergences from §10

1. **The ring is NOT a parallel tier — SELFSERVE is the AUTHORITATIVE claim.** §10 assumed self-serve could
   run alongside the ring, "both advancing the same `SL_RBNEXT`/`SL_RBDONE`". Those counters are **per-slot**,
   and a self-serve wave has no slot — so a flat self-serve claim plus the ring's poison-encoded per-slot
   claim would enumerate the same `G*n_kseg` work items **twice** (double-compute). There is no pre-existing
   flat claim counter to share. So under `SELFSERVE`: `.Lflow_compute` and `.Lflow_feed` both branch to
   `.Lflow_selfserve`, and `.Lflow_coast` yields instead of entering the DA reservation. The ring/DECENTASN
   frontier is unreachable (verified by review). Ring-as-operand-cache is deferred to **S1b**.
2. **No `RBDONE_SS` array.** The existing **`TILEDONE`** completer is already slot-independent (the C-store is
   elected by the first wave to cross `ACC_N*n_kseg`), so self-serve reuses it as-is. §10's per-rowblk done
   array is unnecessary. The elected closer — unique by construction — also drives the group/tile boundary,
   so the boundary needs **no ZLOCK**.
3. **`SS_NEXT` is per-group with a folded generation, not a flat tile-wide counter.** A flat counter needs a
   divide by `ACC_N` (non-pow2) to recover the group. Counting *within* a group makes the decode divide-free.

### 11.2 The claim word (the S2 repair)

```
SS_NEXT = (curg << SS_GSHIFT=16) | item        item in [0, ACC_N*n_kseg),  curg in [0, GROUPS)
sentinels (all >= SS_RESV=0xFFFFFFFD, never claimable):
  0xFFFFFFFF = needs first tile claim   0xFFFFFFFE = first-claim election in flight   0xFFFFFFFD = terminal
```

One CAS claims **generation + item atomically**, and `curg` is decoded **from the claimed value**. This is
load-bearing for two independent reasons:

- **Generation safety.** A claimant stalled across a boundary simply loses the CAS; it can never compute an
  item under a stale group/tile identity.
- **Register safety.** `curg` never has to survive an LDS helper macro. **`s49` is this kernel's reserved
  `exec_lo` save slot for every `lds_put` / `lds_*_add` / `lds_inc` / `lds_cmpstore_adv` / `lds_cas_rtn`
  macro.** The first draft held `curg` in `s49`; every claim CAS destroyed it (deterministic corruption on
  every dispatch). The working copy now lives in **`s41`**, the register the ring already uses for "group".

`DA_TILE` is published **before** `SS_NEXT`; `SS_NEXT` is the sole release fence. No ZLOCK: `item >=
ACC_N*n_kseg` naturally blocks all claims for the entire boundary window.

### 11.3 Decode (divide-free; requires pow2 `n_kseg`)

```
curg    = w >> 16                 ksi     = item & mask        (mask = n_kseg-1)
item    = w & 0xFFFF              localrb = item >> shift      (bank index, 0..ACC_N-1)
abs_rowblk = curg*ACC_N + localrb            gi = (t << shift) | ksi   -> DECODE_STI -> mblk, tcol, ksi
```

Valid **only for power-of-two `n_kseg`**: otherwise `shift`/`mask` describe a *padded* field while `SS_NEXT`
counts *densely*, so items alias onto invalid K-segments (K=6144 -> n_kseg=24 is a live example). The old
DECENTASN path fail-safes on this at `.Lflow_da_peek`, which SELFSERVE bypasses — so the guard (pow2 **and**
`n_kseg > 1`) now sits at the first claim and fails **safe** to terminal.

### 11.4 Ordering rules that must not be broken

- **Grow BEFORE claim.** A grow that fails after a claim would drop the item (short count -> wrong C).
- **`ds_add` drains (`s_wait_dscnt 0`) before the `TILEDONE` bump**, so `TILEDONE == target` proves every
  segment of the group is globally visible — that is what makes the closer's bank read safe.
- **C-store drains (`s_wait_storecnt 0`) before `zero_banks`** reuses those banks for the next group.
- **`DA_TILE` before `SS_NEXT`** (release fence).
- Terminal parks `SS_NEXT = SS_RESV` **before** broadcasting `FLOWTERM`, so no wave can claim against a tile
  that was never published.

### 11.5 Known deferred

- **32-bit A/C offset truncation** (>4 GiB offsets wrap). Real, but it reproduces `ASTAGE_R`'s pre-existing
  ring limitation (the math was lifted verbatim) — parity, not a self-serve regression.
- **S1b:** ring-as-operand-cache fast path; the baton poke in the self-serve shrink (perf only).

### 11.6 Offline gates (both green as of 2026-07-19)

`SELFSERVE=0` byte-identical to canonical `43beb082`. `SELFSERVE=1` (WAVES=30 G=6 **FM=1** SEGK=256 POOL_N=1
ACC_N=3 JDEPTH=1 DECENTASN=1 BANKZERO=1 STAGGER=1) = `a563a9f1`, 0 scratch instructions,
`private_segment_fixed_size: 0`; `POOL_N=2/SEGK=128` also assembles. **Note FM=1** — the winning frontier
config is FM=1, not FM=2 (`ARES_BYTES = G*16*FM*SEGK` overflows LDS at FM=2/SEGK=256).

**S3/S4 remain gated: no GPU run yet.**

### 11.7 S2 pass-2 (verification of the repairs) — 2026-07-19

Repairs **2 (generation-coupled CAS), 3 (terminal), 4 (pow2 guard placement)** confirmed **CLOSED**, with `s41`
verified clobber-free through the whole live range in the emitted ISA. Repair **1 was NOT CLOSED** and two new
issues surfaced; all three are now fixed:

1. **Item-field capacity guard was unsound.** The build-time check assumed a maximum `n_kseg`, but `n_kseg` is
   **runtime** (`KT/SEGK`). With `ACC_N*n_kseg > 0xFFFF` the claim's `item++` **carries into the `curg` field**:
   the group never reaches its completion target (no closer elected, C never stored) while waves start
   computing under a bogus group with unzeroed banks. Replaced with a **runtime fail-safe at first-claim**
   (`ACC_N*n_kseg > SS_ITEMMASK` -> terminal), alongside the pow2 check. *Lesson: any bound involving `n_kseg`
   is a RUNTIME bound.*
2. **`KMAJOR=1` incompatibility.** Self-serve builds a tile-major `gi = (t<<shift)|ksi`, but `DECODE_STI` under
   `KMAJOR` decodes `ksi*TOTAL + tile` — silently wrong tile+segment, no guard existed. Now a build-time
   `.error`: **SELFSERVE v1 requires `KMAJOR=0`** (verified: `KMAJOR=1` now refuses to assemble).
3. **Rule-5 store drain before the self-serve shrink.** `fat_dec`/`flow_gauge` can emit a non-returning global
   atomic with no drain of its own (FATGAUGE/STAGINSTR builds), and `s_alloc_vgpr` does *not* drain VMEM
   stores. Added an explicit `s_wait_storecnt 0x0` before the shrink rather than depending on build config.

Post-fix gates: `SELFSERVE=0` byte-identical (`43beb082`); `SELFSERVE=1` = `e2606a24`, 0 scratch instructions;
`KMAJOR=1` correctly refuses; `POOL_N=2/SEGK=128` assembles.

### 11.8 S2 pass-3 — 2026-07-19

Repair **2 (`KMAJOR=0` guard) CLOSED** (verified `KMAJOR=1` actually fails assembly). Repairs 1 and 3 were
**NOT CLOSED**; both are now fixed:

1. **The overflow guard itself overflowed.** `s_mul_i32 ACC_N * n_kseg` truncates to 32 bits *before* the
   comparison: at `ACC_N=8, n_kseg=2^29` the product is exactly `2^32` -> reads as **0** -> `0 > 0xFFFF` is
   false -> guard passes, `items` reads 0, every gate coasts forever, and the already-claimed global tile is
   never computed (incomplete C). **Fix: never compute the product.** `ACC_N` is a build constant, so compare
   `n_kseg` against the compile-time `SS_ITEMMASK / ACC_N`. Exact in both directions, cannot overflow.
   (The `items == SS_ITEMMASK` boundary was independently confirmed correct: last claim is `0xFFFE`, stores
   `0xFFFF`, no carry.)
2. **The Rule-5 drain covered only one of the reachable reallocation sites.** Now drained on the self-serve
   side of *every* one: before the **grow**, before the **post-claim shrink**, and via a new
   `.Lflow_ss_noclaim` trampoline before branching into the shared `.Lflow_cmp_tryadv` shrink. The trampoline
   exists because that shared path is pre-existing ring code with no drain of its own, and editing it would
   break `SELFSERVE=0` byte-identity. Undrained emitters this defends against: the entry `live++` atomic,
   `fat_inc`/`fat_dec` under `FATGAUGE`, `TRACE`'s atomic max, and wid0's `FORENSICS` `flow_snapshot` — wid0's
   feed role now redirects into self-serve, which is what made that last one newly reachable.

**PRE-EXISTING, NOT FIXED HERE (kmbandy's call, affects the ring identically):** the entry `live++` atomic,
`TRACE`'s atomic, and `FORENSICS` `flow_snapshot` reach the *initial* `s_alloc_vgpr` (role adoption, ~2520 /
~2826) with no drain, independent of `SELFSERVE`. Fixing that means editing shared code and giving up the
byte-identical `SELFSERVE=0` baseline, so it is filed rather than silently changed.

Gates: `SELFSERVE=0` byte-identical `43beb082`; `SELFSERVE=1` = `cb0ef618`, 0 scratch instructions;
`KMAJOR=1` refuses; `POOL_N=2/SEGK=128` and the `ACC_N=8/SEGK=16` overflow geometry both assemble.

### 11.9 S2 pass-4 — 2026-07-19 (partial: two job failures) + a follow-up fix

**Repair 1 (overflow-free capacity bound): CLOSED — corroborated independently twice.** Both my own audit and
the (partial) pass-4 audit reached the same conclusion: the floor bound `n_kseg > (SS_ITEMMASK / ACC_N)` is
exact for every legal `ACC_N` — it neither admits a geometry whose real product exceeds the field nor rejects
one that would fit (verified for ACC_N 1..8, including the non-dividing 3/5/6/7), and every remaining 32-bit
product is downstream of that bound. The degenerate `n_kseg == 0` case (KT < SEGK, `s66` wraps to
`0xFFFFFFFF`) is rejected by the **pow2 test**, which runs first. That ordering is now called out in the
source as load-bearing: the capacity test alone would ADMIT it (`s66+1` wraps to 0 -> items 0 -> every gate
coasts forever on an already-claimed tile). **Do not reorder those two checks.**

**Repair 2 (Rule-5 drains): the pass-3 fix was insufficient; now closed properly.** Draining *before* branching
into the shared `.Lflow_cmp_tryadv` does not help, because that path runs `fat_dec` -> `flow_gauge` ->
`s_alloc_vgpr`: under **`FATGAUGE=1 FORENSICS=0`**, `fat_dec` emits a `global_atomic_add_u32` with no drain of
its own, and `flow_gauge` — which carries the only `s_wait_storecnt` — is **compiled out** by `FORENSICS=0`.
So a store is in flight across the realloc regardless of what is drained beforehand. Fix: self-serve no longer
enters the shared shrink at all; `.Lflow_ss_noclaim` now branches to `.Lflow_ss_shrink`, which orders its drain
**after** `fat_dec`. Verified: no self-serve path reaches `.Lflow_cmp_tryadv`, and every `s_alloc_vgpr`
reachable from self-serve is preceded by a drain that follows the last emitter. (Diag note: grew-but-no-claim
now counts the burst-shrink gauge rather than TASHRINK.)

Gates: `SELFSERVE=0` byte-identical `43beb082`; `SELFSERVE=1` = `1e9f70ee`; `FATGAUGE=1 FORENSICS=0` assembles.

**Caveat for the next session:** the `.Lflow_ss_noclaim` reroute above was found and fixed after the last
completed review pass, so it is the one change in this feature that has NOT been through an independent
adversarial review. It is a single branch retarget to an already-reviewed shrink, but it should be the first
thing a pass-5 looks at.

---

## 12. First silicon run: HANG — root cause, fix, and the instrumentation gap (2026-07-19)

The first-ever `SELFSERVE=1` dispatch (bounded oracle, 576x4096x4096, n_kseg=16, GROUPS=2, chunk=96) **hung**.
Card was undamaged: no MODE1 reset, no page fault, VRAM back to idle, `gpu_run.sh` latched as designed.

### 12.1 Root cause — a "safety improvement" I added

I had added a `deadman_check` to the SELFSERVE coast spin. It was **100% redundant**: `.Lflow_loop` (:2639)
already runs `deadman_check` at every loop head, and the coast path is literally `coast -> s_sleep ->
.Lflow_loop`. So it **doubled the `s_sendmsg_rtn` REALTIME message traffic from idle coasting waves**.
`deadman_check` sends a REALTIME message every 64th call then `s_wait_kmcnt 0`; with thousands of synchronized
coast waves the duplicated traffic saturates the SQ message path, KMCNT never returns, the wave blocks in
`s_wait_kmcnt` forever, never retires, and the dispatch wedges. This is the exact class CLAUDE.md Rule 5 names
and that this source documents at :1111-1124.

**The meta-lesson.** The comment I wrote *in the same edit* said "self-throttles its RTC read 1-in-64, so it is
safe in a coast spin (Rule 5)". I reasoned correctly about the throttle and never asked the prior question:
**is there already a watchdog on this path?** There was, one branch away. Citing a rule is not checking the
invariant the rule protects. **Before adding any watchdog/probe/counter to a spin path in this kernel, walk the
loop to its head and enumerate the message-bus and store traffic already there.** Note also that this survived
FOUR adversarial passes — because it was framed (by me, to myself and to reviewers) as an obviously-good safety
addition rather than as a change to hot-path message traffic. Reviewers challenge what you point them at.

### 12.2 Fixes

1. Removed the redundant `deadman_check`; the loop-head watchdog owns it.
2. The FAT post-grow `SS_NEXT` CAS retry was the one self-serve spin with no watchdog above it. Closed it
   **without** adding a deadman (that repeats the mistake): a **bounded retry budget of 8**, mirroring
   `.Lflow_da_peek`'s peek budget — on exhaustion bail to shrink+coast, which reaches the loop-head watchdog.
   No claim is held there, so nothing is lost, and it adds **zero** message traffic.

### 12.3 The instrumentation gap (and its fix)

The run could not explain itself. Every `CNT_*` is a per-wave SGPR flushed by `cnt_emit` **at retire** — so when
the failure *is* "waves never retire" they all read 0 and carry no information. `fatPeak`/`residentPeak` are
FATGAUGE/TRACE-gated and were structurally 0. The self-serve lifecycle had exactly **one** eager observable
(`occ[20]`).

Added an **eager** lifecycle snapshot, `.if SELFSERVE && FORENSICS`, folded into the existing `flow_snapshot`
so it inherits that macro's wid0-only scope and 1-in-64 throttle (7 stores/coord-cycle already carry a
documented MES-quiesce risk, so this deliberately does **not** get its own cadence, and stores only 3-4 words):

| slot | field | reads |
|---|---|---|
| `occ[40]` | raw `SS_NEXT` | `0xFFFFFFFF` no tile yet, `0xFFFFFFFE` first-claim in flight, `0xFFFFFFFD` terminal; else `(curg<<16)\|item` |
| `occ[41]` | `DA_TILE` | the WG's pinned tile |
| `occ[42]` / `occ[43]` | `TILEDONE[0]` / `[1]` | segments completed per group (target `ACC_N*n_kseg`) |

These live in the LOW control region the host re-zeroes **per chunk**, so on a timeout they show the *failing*
chunk. The host prints them under `[timeout forensics] SELFSERVE` with a read-key: `item` short with
`TILEDONE[curg]` shorter still == claimed-but-never-completed; `item` maxed with `TILEDONE` short == a claimant
died holding an item.

Gates: `FORENSICS=0` byte-identical both ways (`SELFSERVE=0` = `43beb082`, `SELFSERVE=1` = `e952e6ef`);
`FORENSICS=1 SELFSERVE=1` assembles (22948B); host compiles clean.

**Next silicon run must be the `FORENSICS=1` build** so the death is self-explaining. Requires: human clears
`.gpu_last_hang`, fresh greenlight, new board claim.
