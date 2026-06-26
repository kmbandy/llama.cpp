# Hybrid dyn-VGPR Cooperative GEMM — Sequential Design (MAD-305)

Living design doc. Built step-by-step, each step locked with kmbandy before the next.
Target silicon: gfx1201 (R9700), wave32. Vehicle: raw-PM4 KFD compute dispatch (occ_dispatch.cpp).

NORTH STAR (the actual goal): **max COMPUTE on training/prefill shapes → drive toward 250–300 TF**
(feed-free matrix ceiling = ~305 = 99.5% of the 307 fp8 WMMA peak). The cooperative reuse lever exists to
close the feed/issue-density gap toward this. Decode shape = SEPARATE regime: **max MEM BANDWIDTH**
(M≈1 GEMV, memory-bound, NOT reuse-bound — likely a different code path; sibling target, not this kernel).
Landmark (NOT a wall): best real-fed gfx1201 GEMM shown so far ~155–165 (square; hipBLASLt lands here too);
push past it, measure, kmbandy calls any ceiling.

FLOOR (must NOT regress below — the deployed static alternative, batched 2x8, §6.6; this is the floor, NOT
the target): down 21.0, gate/up 20.4, attn_q 11.0, attn_o 11.3, gtup_pf 6.7, attn_kv ~3.0, down_pf 6.6,
q_pf 3.0, kv_pf 0.8, o_pf 3.0.

---
## STEP 0 — What we compute, pinned

GEMM per ml8 linear: **C[M,N] = A[M,K] · B[K,N]**, fp8 e4m3 inputs, fp32 accumulate.
WMMA primitive: `v_wmma_f32_16x16x16_fp8_fp8` — one op = 16(M)×16(N) output block over 16 K,
consuming 1 A-frag (16M×16K) + 1 B-frag (16K×16N). Per-wave tile FM×FN = FM·FN WMMAs/K-step,
needs FM A-frags + FN B-frags/K-step, covers (16·FM)×(16·FN) output.

Memory layout (inherited from occ_kernel_mbgemm.s): A = row-major [M,K], loaded direct via
`global_load_b64` (8 fp8/lane). B = **preshuffled Bshuf**, loaded via `global_load_tr_b64`
(hardware-transpose into the K16×N16 frag) — this is the BINDING feed (B-tr/MAC). C = [M,N] out.

The 10 ml8 shapes, in 16-blocks (M/16 row-blocks, K/16 K-steps, N/16 col-blocks):

| shape | M | K | N | M/16 | K/16 | N/16 | A bytes | B bytes | heavier |
|-------|---|---|---|------|------|------|---------|---------|---------|
| down    | 2048 | 9216 | 2560 | 128 | 576 | 160 | 18.9M | 23.6M | **B** |
| gate/up | 2048 | 2560 | 9216 | 128 | 160 | 576 |  5.2M | 23.6M | **B** |
| attn_q  | 2048 | 2560 | 4096 | 128 | 160 | 256 |  5.2M | 10.5M | **B** |
| attn_kv | 2048 | 2560 | 1024 | 128 | 160 |  64 |  5.2M |  2.6M | A (only) |
| attn_o  | 2048 | 4096 | 2560 | 128 | 256 | 160 |  8.4M | 10.5M | **B** |
| down_pf | 512  | 9216 | 2560 |  32 | 576 | 160 |  4.7M | 23.6M | **B** |
| gtup_pf | 512  | 2560 | 9216 |  32 | 160 | 576 |  1.3M | 23.6M | **B** |
| q_pf    | 512  | 2560 | 4096 |  32 | 160 | 256 |  1.3M | 10.5M | **B** |
| kv_pf   | 512  | 2560 | 1024 |  32 | 160 |  64 |  1.3M |  2.6M | **B** |
| o_pf    | 512  | 4096 | 2560 |  32 | 256 | 160 |  2.1M | 10.5M | **B** |

Observations that drive the design:
- **B is the heavier operand on 9/10** (attn_kv is the lone A-heavy exception, and it's a tiny GEMM).
- **K varies 160–576**, M is 128 or 32 row-blocks, N is 64–576 col-blocks.
- Two regimes: **dense (M=2048)** and **prefill (M=512)**. Prefill has 4× fewer M-row-blocks → fewer
  total tiles → most sensitive to per-tile/launch overhead (matches the gauntlet's widening gap).

---
## STEP 1 — The decomposition (the foundational fork): shared-B col-stationary

**Decision: the P-wave cluster is COL-STATIONARY / shared-B.** P compute waves work the SAME N-columns
(share the FN B-frags) and DIFFERENT M-rows (each owns its FM A-frags). B is loaded once into LDS by the
feed wave; all P compute waves `ds_load` it. Rationale: B is the heavier feed on 9/10 shapes AND it's the
binding `global_load_tr` path — sharing it cuts the dominant traffic and attacks the binding feed.
(attn_kv is A-heavy but tiny — accept suboptimal there in v1; revisit as a switchable later.)

Per-compute-wave tile = **2x4** (FM=2 own M-rows, FN=4 shared N-cols) — the largest dyn-able grower
(RGA: 94 live / 120 HW-alloc ≤ 128 cap). P waves stack in M → WG output tile = (16·2·P)×(16·4) =
(32P)×64.

**Effective reuse (the whole point) — climbs with P:**
per WG-tile per K-step: WMMAs = P·FM·FN ; feed frags = (A: P·FM own) + (B: FN shared).

| P | WMMAs/Kstep | A-frags | B-frags(shared) | total feed | **reuse = WMMA/feed** | Σ VGPR (P·120 + 64 feed) |
|---|-------------|---------|-----------------|------------|-----------------------|--------------------------|
| 1 (lone 2x4) | 8 | 2 | 4 | 6 | 1.33 | — |
| **2** | 16 | 4 | 4 | 8 | **2.00** | 304 |
| **3** | 24 | 6 | 4 | 10 | **2.40** | 424 |
| **4** | 32 | 8 | 4 | 12 | **2.67** | 544 |

P=2 already matches a 4x4 static tile's reuse (2.0) using two dyn-able 120-VGPR waves instead of one
un-launchable 192-VGPR tile. P=3 (2.4) and P=4 (2.67) exceed it. All Σ VGPR ≤ 1152/SIMD (oversubscribe
invariant green, §6.6 + RGA). **Sweet-spot candidate: P=3 compute + 1 feed = 4 waves = one per WGP SIMD**
(clean placement; growers don't share a SIMD).

Persistent-claim retained (not grid): the feed/claim wave runs the atomic work-queue ahead into an LDS
tile-ring → hides the ATOMIC wall (the measured 36–40% framework bubble) AND produces the shared B.

### STEP 1 — LOCKED (kmbandy sign-off 2026-06-23)
1. **Shared-B**, single scheme for v1. attn_kv (A-heavy) accepted suboptimal; switchable revisit later.
2. **P parameterized** (build defsym like FM/FN/BATCH). **Open at P=3** = 1 feed + 3 compute = 4 waves =
   one-per-SIMD on the 4-SIMD WGP (cleanest safety + issue separation). Sweep P∈{2,4} once correct.
   Accumulator type: **fp32, locked** (ISA-fixed for fp8 WMMA + numerics + faithfulness contract; the
   stored OUTPUT may down-convert in the epilogue — separate Step-8 decision).
3. **Persistent atomic-claim + claim-wave**, not grid dispatch.

---
## STEP 2 — Workgroup geometry & role assignment

**Waves/WG = 1 + P** (P parameterized; open P=3 → 4 waves = 128 threads, wave32×4).
- **wave 0 = FEED/CLAIM** (lean, never grows): owns the atomic work-queue claim (run-ahead into an LDS
  tile-ring) + loads the shared B-frags (`global_load_tr`) into LDS + signals via LDS counter.
- **waves 1..P = COMPUTE** (dyn, launch lean@32 → `s_alloc` grow to 120 = 2x4 fp32 acc): each owns a
  distinct 32-row M-band of the WG tile; loads its own A (direct `global_load`); `ds_load`s the shared B
  from LDS; runs FM·FN=8 WMMA/K-step; deferred-stores its sub-tile.
- **Role identity:** wave id = (flat_tid / 32). wave 0 → feed; else compute index c = wave_id-1 ∈ [0,P).
- **SIMD placement:** RDNA4 round-robins the 4 WG waves to the WGP's 4 SIMDs → `[1,1,1,1]` at P=3 (feed
  + 3 growers, one each). VERIFY empirically (placement probe). At P=4 (5 waves) one SIMD doubles up.

**Output partition (the WG tile = (32·P)×64 for FM=2,FN=4):**
- compute wave c owns output M-rows `[c·32, (c+1)·32)` (its FM=2 A-frag rows), **all share the same 64
  N-cols** (the FN=4 shared B-frags). N-partitioning is NONE — shared-B means every compute wave covers
  the full 64 N of the WG tile, differing only in M-band. The WG claims tiles in (M-band-group × N-strip)
  units; the claim wave hands each compute wave its M-band offset via the LDS ring.

**Feed split (v1):** B shared through LDS (feed wave produces); **A stays per-compute-wave direct load**
(A is M-band-distinct → nothing to share; keep its issue on the compute wave, off the feed wave). The
binding/heavier feed (B) is the one offloaded — matches the shared-B traffic rationale. (Possible later:
also stage A through LDS to further unload compute issue — Step-3 evaluate, not v1.)

**Register budget per role:** feed ~32–64 (B-load staging + claim/ring pointers, never grows);
compute launch 32 → grow 120 (RGA-confirmed 2x4). Σ/WG @P=3 = 64 + 3·120 = 424 ≤ 1152/SIMD ✓.

### STEP 2 — LOCKED (kmbandy sign-off 2026-06-23)
1. A-direct / B-shared for v1 (A un-shareable across M-bands → keep its issue local; offload only B).
2. WG-level claim (BATCH tiles/grab) by the claim wave; fan the P M-bands to compute via the ring.
   ONE atomic claimer per WG.

---
## STEP 3 — LDS layout + the lock-free producer/consumer protocol (NO s_barrier)

Structure is TWO nested pipelines, both driven by the feed wave, both coordinated by LDS busy-wait only:
- **Tile level (coarse):** feed wave claims BATCH (M-band-group × N-strip) tiles/atomic, publishes the
  WG's "current tile" (M-base, N-base) to all waves. All P compute + feed work the SAME tile together.
- **K-step level (fine):** within a tile, feed wave PRODUCES shared B[k] into a depth-D ring; the P
  compute waves CONSUME each B[k] (broadcast — every consumer reads every B[k]).

### LDS byte layout (P=3, FN=4, depth D; B-frag = 16K×16N fp8 = 256 B)
```
B_ring      : D * (FN * 256)      // D * 1024 B  — the shared B-frags, D K-steps deep
prod_count  : u32                 // feed-written: # B-steps published (monotonic)
cons_count  : u32 * P             // compute c-written: # B-steps wave c has drained (monotonic)
tile_slot   : u32 * 3             // feed-written: (M_base, N_base, tile_epoch) of current WG tile
```
D=2,P=3 → ~2080 B of 65536 → LDS is NOT the occupancy limiter (≥30 WG by LDS); VGPR is. So D is free to
tune (knob; open D=2).

### Sharing mechanism (B loaded from DRAM ONCE)
Feed wave: `global_load_tr B[k]` (global→VGPR, the hw-transpose) → `ds_store` (VGPR→B_ring slot). Compute
waves: `ds_load` (B_ring→VGPR). DRAM B-traffic = 1× (was P×). The long-latency DRAM load lives on the feed
wave (hidden); compute waves only ever hit short-latency LDS for B. A stays per-compute-wave direct load,
double-buffered in regs (the existing pf path), overlapping its own WMMA.

### The protocol (per tile, k = 0..KT-1)  — counters only, ZERO s_barrier
```
FEED (producer):                         COMPUTE wave c (consumer):
 for k in 0..KT-1:                         for k in 0..KT-1:
   s = k mod D                               s = k mod D
   # slot-free gate: all P drained k-D       # ready gate: B[k] published
   spin until min_c cons_count[c] > k-D      spin until prod_count > k
   global_load_tr B[k] -> VGPR               ds_load B_ring[s] -> VGPR      (+ prefetch own A[k])
   s_waitcnt vmcnt(0)        # B in reg       s_waitcnt lgkmcnt(0)          # B in reg
   ds_store VGPR -> B_ring[s]                 WMMA x(FM*FN)  accumulate
   s_waitcnt lgkmcnt(0)     # B in LDS *first*  cons_count[c] = k+1          # release slot s
   prod_count = k+1         # publish *after*
```

### Memory-ordering = the whole correctness/safety core (NO barrier, so fences must be exact)
- **Producer order:** write B to LDS, `s_waitcnt lgkmcnt(0)` (B-store COMPLETE), THEN bump prod_count.
  Guarantees any consumer that sees the bump reads valid B. (LDS is one coherent WGP SRAM — once the
  producer's ds_store drains, other waves' ds_loads see it; no cache coherence needed.)
- **Consumer order:** read prod_count; when > k, ds_load B, `s_waitcnt lgkmcnt(0)`, then use.
- **Slot reuse:** producer's "min cons_count > k-D" gate prevents overwriting B[k-D] before all P drained.
- **RACE CLASS (must gate):** a misplaced fence → consumer reads stale B → **wrong frags, silent
  corruption — NOT a hang/brick.** => the 512³ oracle catches it deterministically. This is the #307
  race-gate situation: build it, oracle-gate HARD (acc[0][0] + full-frag forensics), the race surfaces as
  BAD tiles, never as a wedge. (Brick-safety is orthogonal: still zero s_barrier, bounded grow, Σ≤pool.)

### Ring depth D (knob, open D=2)
All P consumers read the same B[k] and do identical work (FM·FN WMMA) → they drain in near-lockstep, so the
producer needs only ~1 step lead; D=2 (double-buffer) is the floor, D=3 a safety margin. Tune as a defsym.

### STEP 3 — LOCKED (kmbandy sign-off 2026-06-23)
1. Counter-based busy-wait (monotonic prod_count / cons_count[P]).
2. v1 = **tile-synchronous**. Cross-tile B-ring overlap → FUTURE ENHANCEMENT (see below).
3. D=2 open, tunable defsym knob.

---
## FUTURE ENHANCEMENTS (deferred, post-correctness tuning levers)
- **FE-1 — Cross-tile B-ring overlap.** Feed wave begins producing tile T+1's B[0..D] while compute
  finishes T's tail → hides the per-tile B cold-start (~D/KT). On ml8 worth only ~0.3–1.2% (deep K,
  KT=160–576). Cost: ring slots must carry an N-strip tag + claim-ahead + more race surface. Add as an
  isolated lever after correctness; keep only if the ~1% shows up.
- **FE-2 — Descending s_wait_loadcnt release ladder.** Replace the simple full-drain (`s_waitcnt lgkmcnt(0)`
  before the WMMAs) with a staged ladder (`lgkmcnt(3)`→WMMA→`lgkmcnt(2)`→... ) so the first WMMA starts on
  the first-ready operand and overlaps the rest of the load latency with WMMA execution (#291/#292 lever).
  Same correctness. EXPECTED SMALL here (B is short-latency LDS, A is prefetched) — add ONLY if the phase
  timer/ATT shows feed-LATENCY is the residual wall after the kernel is oracle-green. Measure, don't guess.
- **FE-3 — e4m3 (fp8) output down-convert — DEPLOYMENT FEATURE (not just polish).** v1 stores fp32 (oracle-
  clean, 0 TF cost). For DEPLOYMENT the epilogue should down-convert to **e4m3** (`OUTDT` knob; fp16/bf16
  also): (1) FAITHFULNESS — the deployed ml8 path runs e4m3 activations; the next GEMM re-quantizes to e4m3
  anyway, so e4m3-out matches deployed numerics. (2) BANDWIDTH — 4× cheaper inter-layer activation reads vs
  fp32. (3) FUSION — fuses the activation-quant into the GEMM epilogue (skip a separate quant op + the fp32
  activation round-trip; aligns with the existing ml8 prologue/epilogue fusion). NOTE: PIPELINE win, NOT a
  this-kernel-TF win (store is small + overlapped + compute-bound) — do it for the product, not the benchmark.

---
## STEP 4 — Grow/shrink placement + the honest scope of what dyn carries

### 4a. The grow-ordering rule (the DEFERGROW brick lesson — non-negotiable)
Compute wave lifecycle: launch lean (32) → `s_alloc_vgpr 120` (SCC-retry) → zero acc → K-loop → store →
`s_alloc_vgpr 32` shrink. **GROW BEFORE loading any operand into the grown VGPR range.** Load-then-grow
WEDGES: `s_alloc_vgpr` races the VGPR write-back of the just-loaded operands (the exact dg hazard). So:
grow → zero accumulators → THEN the first A load / B ds_load. Each compute wave grows independently; at P=3
(one grower per SIMD) the ISA "one wave/SIMD reaches max" guarantee makes every grow SCC=1 immediately.
Grow placement: per-tile (grow after claim, shrink after store) — GROW/SHRINK measured ~0% so the per-tile
tax is negligible and it keeps occupancy clean. (grow-once-stay-fat = simpler but holds fat across idle.)

### 4b. HONEST SCOPE — what dyn actually buys here (so it's load-bearing, not decorative)
The REUSE lever (cooperative shared-B via LDS) does NOT itself require dyn — a *static* multi-wave
LDS-cooperative WG already achieves it (this is exactly the proven 155-TF kernel: 4 waves share A in LDS,
no barrier, no dyn). So in THIS design dyn's contribution is specifically:
- **Heterogeneous per-wave sizing:** feed wave stays at 32 while compute are 120, vs a static dispatch's
  UNIFORM 120-for-all-4-waves (which wastes ~88 VGPR on the lean feed wave). Saves VGPR/WG → occupancy.
- **The moat:** HIP cannot express per-wave heterogeneous sizing at all. dyn (raw-PM4 RSRC2 bit6) can.
- (NOT occupancy-via-lean-launch-then-grow: the accumulators are live across the whole K-loop, so compute
  waves can't be lean during feed. That thesis doesn't apply to a live-accumulator tile.)

### 4c. Sequencing implication (de-risks "right on the first go")
Because the reuse core is static-provable (155-kernel pattern) and dyn is the enhancement on top, the
SAFEST build order is:
  **(i) STATIC cooperative core first** — 1 feed + 3 compute, shared-B LDS ring, no barrier, NO dyn,
      uniform 120 VGPR. ZERO brick risk (it's the proven LDS-cooperative pattern). Validates the REUSE win
      on the real ml8 shapes vs the 21.0 baseline. Oracle-gated.
  **(ii) THEN layer dyn** — drop the feed wave to lean 32 (heterogeneous sizing) for the occupancy bonus +
      the moat, with the grow-ordering rule above. The brick-safety (Σ≤pool, no barrier, SCC-retry) was
      already proven green; dyn only changes the feed wave's footprint.
dyn stays central — it's the enhancement + the moat — but we prove the reuse foundation first, where there
is no brick surface at all, then add dyn onto a known-good kernel.

### STEP 4 — LOCKED (kmbandy sign-off 2026-06-23)
1. **Build dyn-cooperative FIRST** (dyn = the cornerstone + the moat artifact). static-cooperative is the
   FALLBACK if dyn-cooperative misses the 21.0 baseline. Sequence is self-disambiguating on failure
   (static-fallback isolates reuse-idea vs dyn-layer). Win attribution (dyn-over-static) deferred = optional
   dossier question, not a blocker.
2. **Grow per-tile** (~0% tax, clean occupancy).
3. dyn is the CORNERSTONE — the lever is OPEN, not closed. (The 2026-06-22 "closed for deployment" line
   was Claude's premature self-verdict, never user-endorsed, and only covered the SINGLE-WAVE case; the
   cooperative cluster is a new regime it never tested.) Honest technical scoping, NOT a doubt: the reuse
   is delivered by cooperation (static-provable) AND dyn adds heterogeneous lean-feed sizing + the moat —
   the cooperative-cluster measurement is what tells us how much dyn converts. Build dyn-cooperative.

---
## STEP 5 — Dispatch / launch + harness wiring

### 5a. WG geometry & launch (the PM4 packet)
- **WG = (1+P) waves = 128 threads** at P=3 (was 32 = single wave). RSRC workgroup-size = 128.
- **dyn-arm:** RSRC2 bit6 = 1 (MAD-304). **Lean launch VGPR field = 32** — ALL waves launch at 32; only
  compute waves grow.
- **Grid = persistent pool of WGs**, sized so per-SIMD resident fat ≤ ~1152/SIMD. One-wave-per-SIMD
  ([feed,c,c,c] over 4 SIMDs) → per-SIMD demand = max(120,32) = 120 → ≤ ~9 WG/SIMD. Harness caps the grid
  (the brick knob; today's `pool` param, made multi-wave-aware).

### 5b. New defsyms (compose with FM/FN/BATCH/GENDIV/DYNVGPR)
`COOP=1` (selects the cooperative path), `P` (compute waves/WG, open 3, sweep {2,4}), `RINGD` (B-ring
depth, open 2). FM=2/FN=4 fixed for the cluster tile; GENDIV=1 for non-pow2 N.

### 5c. Harness mode `--mbml8coop`
Per shape: launch the 128-thread-WG cooperative kernel, dyn-armed, pool-capped, GENDIV, sustained TF +
oracle. Tile decode = (M-band-group × N-strip); claim wave fans P M-bands. Reuses run_mbgemm scaffolding
(freq / TF formula / compositor yield) with a 4-wave WG variant. Report per-shape TF vs the 21.0 floor and
toward the 250–300 north star.

### 5d. SIMD-placement probe — BUILT + RUN, CONFIRMED ✓ (2026-06-23, simdprobe.hip)
RESULT: 16/16 WGs placed **one-wave-per-SIMD within a single WGP** on the R9700. Deterministic map
(w0→SIMD0, w1→SIMD2, w2→SIMD1, w3→SIMD3, same WGP); WGs round-robin across WGPs. The [1,1,1,1] geometry
is REAL on silicon. → within-WG: feed + 3 compute each alone on a SIMD (issue-port separation ✓, ≤1
grower/SIMD per WG ✓). CAVEAT (honest): probe is tiny-VGPR/high-occupancy; the within-WG round-robin is a
fixed scheduler behavior (holds at the real kernel's occupancy), but CROSS-WG multiple resident WGs DO
stack growers on a SIMD — that leg is bounded by the admission pool (Σ≤1152/SIMD → ≤9 WG/SIMD), not by
placement. Re-confirm under the real COOP kernel (RGA + dispatch) when it exists. Both brick-invariant legs
(within-WG placement + cross-WG pool cap) now accounted for.

### 5d-orig. SIMD-placement probe (de-risk the [1,1,1,1] assumption BEFORE the kernel body)
Tiny safe kernel (no dyn, no heavy compute): each of the 4 WG waves reads its hardware SIMD id via
`s_getreg HW_ID` and writes it to a per-wave slot. Harness reads back → confirm 4 DISTINCT SIMDs. This
underpins BOTH the brick-invariant tightest guarantee (≤1 grower/SIMD) AND the issue-port separation. If
they don't separate → fall back to the strict Σ≤pool cap (still brick-safe), note issue-port benefit
reduced. **Run this FIRST — ~free, safe, validates an assumption the whole geometry rests on.**

### 5e. Pre-first-dispatch safety gate (in order)
1. **RGA livereg** on the ACTUAL COOP=1 kernel → feed peak-live ≤ ~64, compute ≤ 128, Σ/SIMD ≤ pool (offline).
2. **SIMD-placement probe** (5d) → confirm [1,1,1,1].
3. **512³ oracle gate** → correctness + no-hang on a sub-second dispatch (wedge = free reboot, not 100 min in).
4. **Supervised escalation** — one shape, compositor yield ON, tee to persistent file, kmbandy present.

### Open questions for STEP 5 (sign-off before STEP 6 = compute-wave inner loop / WMMA scheduling)
1. Confirm `--mbml8coop` + COOP/P/RINGD defsyms as the wiring shape?
2. Build the SIMD-placement probe as the FIRST concrete artifact (before any kernel body)? ~free, safe,
   validates the geometry everything rests on.

---
## STEP 6 — Compute-wave inner loop + WMMA scheduling (the issue-density attack on the north star)

### 6a. ISA-grounded load widths (fp8) — the cooperative b128 recovery
- B (fp8) transpose from DRAM = **GLOBAL_LOAD_TR_B64 only** (no TR_B128 for 8-bit; TR_B128 is 16-bit data).
  → the FEED wave does TR_B64 (its SIMD), staging 2 K16 frags contiguously per N-col in the LDS ring.
- Compute wave reads B from LDS with **DS_LOAD_B128** = 2 K16 B-frags/load (B already in frag layout, no
  LDS transpose needed). A from DRAM via **GLOBAL_LOAD_B128** = 2 K16 A-halves/load, double-buffered.
- So the compute wave processes K in **K32 chunks**, b128 reads for BOTH operands. This is the double-K /
  issue-density lever — available to the cooperative compute wave precisely because the narrow TR_B64 is
  offloaded to the feed wave.

### 6b. The load-issue win (the concrete attack on the 54%/155 issue-port wall)
Per K32 chunk, compute wave: A = 2×`global_load_b128` (FM=2, each 2 K-halves) + B = 4×`ds_load_b128`
(FN=4, each 2 K-halves) = **6 loads for 16 WMMA = 0.375 loads/WMMA**.
vs single-wave 2x4 b64: per K16, 2 A-b64 + 4 B-tr-b64 = 6 loads / 8 WMMA = **0.75 loads/WMMA**.
→ **HALVED feed-issue per WMMA on the compute wave** (0.75 → 0.375). Plus the narrow TR_B64 B-loads are on
the FEED wave's issue port (separate SIMD, confirmed §5d), not competing with WMMA. This is the lever
aimed at lifting real-fed past the ~155 issue wall toward the 250–300 north star.

### 6c. The per-K32-chunk inner loop (compute wave c)
```
for chunk in 0..KT/2-1:
  spin until prod_count > 2*chunk+1            # both K16 of this chunk published (LDS busy-wait gate)
  ds_load_b128 B[chunk] (FN=4) from B_ring[s]  # 2 K16 B-frags/col, wide
  s_waitcnt lgkmcnt(0)                          # B in regs
  cons_count[c] = chunk+1                       # RELEASE LDS slot NOW (B is in our regs, before WMMA)
  global_load_b128 A[chunk+PF] (FM=2)           # prefetch next A (double-buffer), latency hidden
  # --- WMMA: 16 ops = 8 acc frags x 2 K-halves; 8 INDEPENDENT accumulator chains (ILP) ---
  WMMA x16  (acc[0..7] += A_half0*B_half0 ; acc[0..7] += A_half1*B_half1)
```
Notes: (1) RELEASE after the ds_load drains (not after WMMA) → producer can refill the slot sooner.
(2) the 8 accumulator frags are independent → 8-wide ILP feeds the matrix unit back-to-back.
(3) A's DRAM latency hidden by double-buffer prefetch + cooperative occupancy; B latency = short LDS.

### 6d. s_wait release ladder (mixed counters)
B is lgkmcnt (LDS), A is vmcnt (global). Issue all loads, then a DESCENDING ladder releases WMMA as each
operand lands (don't drain ALL before the first WMMA). Open as simple full-drain first (correctness), then
tune the ladder (the proven #291/#292 lever) once oracle-green.

### Open questions for STEP 6 (sign-off before STEP 7 = epilogue / store + the oracle)
1. Confirm K32 double-K chunking + b128 reads (A global_load_b128, B ds_load_b128 from LDS) as the inner
   loop, with the narrow TR_B64 B-loads on the feed wave?
2. RELEASE-after-ds_load (free the slot before WMMA) — agree?
3. s_wait ladder: open simple full-drain, tune descending later — agree?

### STEP 6 — LOCKED (kmbandy sign-off 2026-06-23)
1. K32 double-K chunking; b128 reads (A `global_load_b128`, B `ds_load_b128` from LDS); narrow `TR_B64`
   B-loads on the feed wave. → 0.375 loads/WMMA on compute (half of single-wave 0.75).
2. RELEASE-after-ds_load (free the LDS slot before the WMMAs).
3. s_wait = **simple full-drain for v1**; descending ladder → FUTURE ENHANCEMENT (FE-2, measure-then-add).

---
## STEP 7 — Epilogue (store) + the oracle (correctness gate)

### 7a. Output dtype — v1 = fp32 store (oracle-clean); down-convert = deployment knob
Accumulators are fp32 (locked). v1 STORES fp32 → matches the fp32 reference bit-for-bit, zero epilogue
error, cleanest oracle. The output bytes (M·N·4) are small vs the K-loop A/B traffic (M·K + K·N), so fp32
store costs little throughput. Down-convert (fp16/bf16/fp8) is a DEPLOYMENT-graph decision (what the next
ml8 op wants) → wire as an epilogue `OUTDT` knob later, NOT a v1 concern. (FUTURE: FE-3 output down-convert.)

### 7b. Store addressing (the cooperative partition)
Each compute wave c stores its 8 fp32 frags (its 32M×64N sub-tile) to C at row base = tile_M_base + c·32,
col base = tile_N_base. WMMA output (D_HW) lane→(row,col) layout: row=(lane/16)·8+slot, col=lane%16
(the documented RDNA4 WMMA D transpose). `global_store` per frag at the computed C addresses. Reuses the
existing mbgemm deferred-store logic, offset by the wave's M-band.

### 7c. Deferred store + grow/shrink interaction
Store is DEFERRED (issued, overlaps the next tile's decode/setup — `s_wait_storecnt` at next-tile top).
With grow-per-tile (Step 4): per tile = grow → zero → K-loop → deferred-store → shrink. GROW GRANULARITY = per-BATCH
(grow when a batch is claimed -> run the batch's K-loops fat -> shrink after the batch's stores). This is
the ELASTIC design: the register footprint TRACKS the work = dyn being dyn. grow-ONCE-stay-fat is REJECTED
(kmbandy): after one grow the compute wave is static-120 for its whole life = static-in-disguise, abandons
dyn's elasticity, and would mean the kernel doesn't even TEST the dyn lever. Whether elastic occupancy
converts is a MEASUREMENT on the real kernel, not a pre-judged reason to go static. Build rule: grow-BEFORE-
load every batch (DEFERGROW ordering).

### 7d. THE ORACLE — the gate that must be green before ANY perf number
512³ chained `wmma_ref` vs C (existing harness). For the cooperative kernel it validates BOTH:
(1) the GEMM math (the cooperative partition + WMMA scheduling compute the right C), AND
(2) the lock-free busy-wait protocol (a fence/ordering bug surfaces as a STALE-B read → BAD frags,
    deterministically — §3 race class). So the oracle is the single gate for correctness AND the no-barrier
    protocol. acc[0][0] + full-frag forensics. MUST be 256/256 (or whatever the gate count) before any TF
    is reported or trusted. Brick-safety is orthogonal (zero s_barrier + Σ≤pool + SCC-retry, already green).

### STEP 7 — LOCKED (kmbandy sign-off 2026-06-23)
1. v1 output = **fp32 store** (oracle-clean, 0 TF cost). e4m3 down-convert = FE-3 (deployment feature).
2. Oracle = **512³ chained wmma_ref**, gates BOTH GEMM math AND the busy-wait protocol; green before ANY TF.
3. Grow = **per-BATCH ELASTIC**; grow-once-stay-fat REJECTED (static-in-disguise).

---
## STEP 8 — Build order / task breakdown (each gated; P=1 protocol bring-up first)

**B0 — SIMD-placement probe — ✓ DONE** ([1,1,1,1] confirmed on R9700, §5d).

**B1 — Harness `--mbml8coop` wiring — ✓ DONE** (occ_dispatch.cpp, 2026-06-23). Added `run_mbcoop()` (the
(1+P)-wave dispatch primitive) + `MBML8COOP` mode. Geometry: WG = (1+P)·32 threads, grid = pool WGs,
dyn-arm (RSRC2 bit6 via BuildPgmRsrc1/2(dyn)), lean launch field 4 (32 VGPR), LDS sized for the Step-3
B-ring layout (RINGD·FN·256 + counters), GENDIV (non-pow2 ml8 N). Fuses run_wavespec_compute's cooperative
oracle partition (compute wave c owns M-band cid·FM·16, shares FN·16 N-cols) with run_mbgemm's
sustained-reps + compositor-yield + VRAM-guard. Operand prep mirrors mbgemm (A plain / B preshuffled for
TR); userdata ABI identical to mbgemm. Mode: per-shape oracle-gate (512-ish, frag-exact) → sustained
real-shape TF vs FLOOR + north star; env knobs ML8_P/RINGD/FM/FN/BATCH/POOL/DYN/ONLY/TGT. **BRICK-GUARD:**
refuses to dispatch if `occ_coop_<FM>x<FN>_p<P>_r<RINGD>_b<BATCH>_d<dyn>_gd.bin` (built in B2) is absent —
the expected B1 state, makes B1 inherently GPU-safe. Compiles clean (0 new errors/warnings). **Offline,
no GPU contact.** NEXT = B2a (P=1 protocol bring-up: occ_kernel_coop.s — first GPU contact, gated).

**B2 — Cooperative kernel body** (occ_kernel_coop.s + build_coop.sh). INCREMENTAL:
- **B2a — P=1 protocol bring-up — ✓ AUTHORED + ASSEMBLED (offline), 2026-06-23.** occ_kernel_coop.s:
  feed wave (claim @off20 + GENDIV tcol decode + global_load_tr_b64 FN B-frags + ds_store ring + publish
  prod_count) + compute wave (epoch-wait + GENDIV trow decode + SCC-retry grow + busy-wait prod_count gate +
  ds_load_b64 shared B + global_load_b64 direct A + WMMA×FM*FN + cons_count release + full-frag fp32 store +
  shrink). Lock-free counter protocol (Step 3), ZERO K-loop s_barrier; ONE symmetric pre-grow s_barrier
  publishes LDS-control init (safe: all waves lean-32, NOT the brick condition — wavespec's proven idiom).
  Userdata ABI identical to mbgemm. P-general gates (min_c cons_count[c]) so B2b is a defsym flip.
  TWO faithful simplifications (oracle-equivalent, layered in once green): (1) K16/b64 inner loop, not yet
  Step-6 K32/ds_load_b128 — isolates the protocol from the load-width optimization; (2) full FM*FN fp32
  store (Step 7a v1). Builds clean (dyn + static cells). Phase-2 self-review (addressing / exec-mask role
  branch / counter release-acquire ordering / slot vaddr / grow-before-load / terminal-drain) all pass.
- **B2b — scale to P=2 then P=3.** Defsym flip (gates already P-general). Re-oracle at each P.

**B3 — RGA livereg — ✓ DONE (offline), 2026-06-23.** occ_coop_2x4_p1 (dyn): **peak-live 81 VGPR, no spills,
SGPR 19/50**. Disasm-confirmed `s_alloc_vgpr 0x70`=112 grow (clean 16-block, SAFELY < 128 cap, OFF the 128
exact-fill edge) / `s_alloc_vgpr 32` shrink. Feed wave never grows (lean 32). Re-run on P=2/3 after B2b.

**B4 — 512³ oracle gate** (COOP), at EACH P (1→2→3). Correctness + the lock-free protocol + no-hang, on a
sub-second dispatch. Catch race/fence bugs at the SIMPLEST config (P=1) before scaling. Wedge = free reboot.

**B5 — Supervised first escalation.** One shape (down or o_pf), compositor yield ON, tee to persistent file,
kmbandy present. Watch Hyprland. Then the per-shape sweep across all 10.

**B6 — Measure + sweep.** Per-shape TF vs the 21.0 FLOOR and toward the 250–300 NORTH STAR. Sweep P∈{2,4},
RINGD, BATCH. rocprof/phase-timer on the residual wall → decides FE-1 (cross-tile) / FE-2 (ladder) / next.

Gate between every step. B1 is offline (safe to start now). B2a (P=1) is the riskiest first GPU contact →
RGA (B3) + tiny oracle (B4) before any escalation (B5). kmbandy decides every GPU dispatch.

=== DESIGN COMPLETE (Steps 0–8). Build starts at B1 (harness wiring, offline). ===
