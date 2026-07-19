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
