# DSWS — burst-owns-a-ksi-run: engaging the dyn-VGPR stagger without a flush

**Date:** 2026-07-06  **Kernel:** `occ_kernel_dsws_flow.s` (gfx1201 / RDNA4 / R9700, wave32, raw PM4)
**Status:** design **v2** — folds council review (Fable). Supersedes the shelved `POOL_N=2` framing, the
rejected `WOFLUSH` (atomic-flush) lever, and v1's *count-only* concurrent-fat model.

> **What changed v1 → v2 (why read this):** v1 modeled concurrent-fat as a pure *count* problem
> (`POOL_N·G_resident`) and proposed shrinking banks (`G_resident < G`) to afford a high `POOL_N`. Fable's
> review found three defects in that: **H1** a silent correctness race in the reduction, **H3** an
> off-by-256 that made every "feasible 64KB" row actually overflow, **H5** a group/tile-boundary bank
> collision the partition scheme creates. v2 adopts Fable's **burst-owns-a-ksi-run**: it makes the
> reduction commutative (kills H1), keeps **whole-tile groups** (`G_resident=G`, no boundary → kills H5),
> and reframes concurrent-fat as **count × duty-cycle** with **burst length** as a free knob on the
> duty axis that costs *no* LDS. Bank-shrink (`G_resident<G`) is demoted from "the plan" to an *optional*
> throughput lever behind a proven bind.

---

## 1. The problem (measured, not inferred)

The dyn-VGPR "traveling-peak" stagger only does work if enough compute waves are **simultaneously fat**
that the per-SIMD VGPR pool (~1536 VGPR; a fat wave = NFV≈112 → ~13 fat/SIMD) actually **binds** →
`s_alloc_vgpr` starts failing → `grow-fail > 0`.

**Run 4 (2026-07-05):** full grid (128 WGs, saturated 256-tile 3072×1024×2048 shape), `grow-fail = 0`,
98.8% coast. The pool never binds. Root cause is the LDS accumulator:

- Write-once-C keeps each rowblk's fp32 C sub-tile in an **8KB LDS bank**, persisting across all
  `n_kseg` split-K segments; `G=6` rowblks → **48KB** of banks.
- LDS/WG = 57,600B → only **~1 WG/CU** resident (gfx1201 CU ≈ 64KB LDS) → few fat compute waves/SIMD.
  Nothing forces waves to compete for VGPR.

## 2. The trilemma (why this is hard) — and the escape

The K-reduction running-sum must live *somewhere* on-chip to be cheap, and to stagger (short fat bursts)
the fat accumulator must be **evicted between bursts** — a dyn-VGPR *shrink* is only legal if its contents
are safe first. v1 framed the three eviction targets as a hard trilemma:

| Running-sum home | Cost |
|---|---|
| **VGPR, whole K** (grind) | square wave — no trapezoid, no stagger; wave held fat entire K-loop |
| **LDS bank** (write-once) | 48KB persistent → caps occupancy at 1 WG/CU → **pool never binds** |
| **Global atomic** (WOFLUSH) | RMW round-trip + same-cell serialization (the 26–51% FLUSH we deleted) |

**The escape (Fable):** the trilemma is false because "VGPR whole-K" and "LDS bank every ksi" are the two
*endpoints* of a knob, not the only choices. Let a fat burst own a **run of J consecutive ksi** for one
rowblk: accumulate those J ksi in the **VGPR ACC** (no LDS touch between them), then do **one** `ds_add`
merge of that partial into the rowblk's LDS bank. `J = n_kseg` is the whole-K endpoint (square wave);
`J = 1` is the write-once endpoint (bank every ksi). **J is a continuous duty-cycle knob between them**,
and it costs no LDS. The bank still persists across bursts (it holds the running sum of merged runs), but
it is *touched* only once per burst, not once per ksi.

WOFLUSH stays rejected: it re-introduces the exact atomic-flush wait write-once removed.

## 3. The insight — concurrent-fat is *count × duty-cycle*, not count

v1's error: it treated instantaneous-fat as `min(POOL_N·G_resident, WAVES)` — a pure **count**. But a
wave is only fat *while it grinds*; between bursts it goes thin (fetch operands, merge, re-claim). So:

> **instantaneous-fat ≈ (waves resident) × (fraction of time each is fat) = count × duty-cycle.**

To bind the pool you need instantaneous-fat/SIMD ≳ 13–14. Two independent levers get you there:

1. **Count** — more concurrent `(rowblk, ksi-run)` claims live at once (`POOL_N`, `G_resident`, `WAVES`).
   Costs LDS (operand slots + banks). This was v1's only lever.
2. **Duty-cycle** — longer fat bursts (**larger J**) → each resident wave spends more of its life fat →
   more are simultaneously fat. Costs **no LDS**. This is the lever v1 was blind to.

Burst length is the free knob on the duty axis. That is what lets us keep **whole-tile groups**
(`G_resident = G`, 48KB banks, 1 WG/CU — the config Run 4 *already* had) and still reach the bind, by
cranking J instead of shrinking banks.

**The H1 fix that makes any of this legal:** the reduction must be **order-independent**. Today it is not —
`ksi==0` does `ds_store` (init the bank), `ksi>0` does `ds_add`; with concurrent runs of the same rowblk
the init can land *after* an add → silent wrong C. v2: **zero-initialize the banks up front, then *every*
merge (all ksi, all runs) is `ds_add`.** `ds_add` is atomic per-op in LDS, so concurrent merges into one
bank serialize correctly and the result is order-independent. (fp32 add is non-associative, so ordering
still perturbs rounding — but write-once *already* has non-deterministic `ds_add` order across ksi and
passes oracle `bad=0` within tolerance; v2 adds no new nondeterminism class.)

## 4. The binding model (corrected arithmetic — H3 fixed)

Per-SIMD fat waves ≈ `instantaneous-fat / 2` (a WG spans 2 SIMDs). Bind ⇒ need **≳ 13–14 fat/SIMD**.

**LDS constraint (the H3 fix):** `256 + POOL_N·OPSTRIDE + G_resident·8192 ≤ 65536`.
Measured OPSTRIDE: **8192 @ SEGK=32, 4096 @ SEGK=16** (from Run-4 `LDS=57600 = 256+8192+6·8192`).
*(v1 omitted the +256 header; every "64KB" row there actually summed to 65792 = 256 **over** the limit and
the kernel's `.error` would have fired.)*

### Path A — whole-tile groups (`G_resident = G = 6`), the correctness+bind proof

Dodges the boundary problem (§7) entirely. Banks = 48KB fixed → **1 WG/CU**. Binding rides on **duty-cycle
(J)**, since the count ceiling is low:

| SEGK | OPSTRIDE | POOL_N max | banks | LDS (+256) | count = POOL_N·G |
|---|---|---|---|---|---|
| 32 | 8192 | 1 | 48KB | 57,600 | 6 ← **Run 4** |
| 16 | 4096 | 3 | 48KB | 61,696 | 18 |

Both are **below** the count-threshold of ~26 — so Path A **cannot bind by count alone**; it must bind by
**duty-cycle**. That is the whole point: crank **J** (burst length) up until enough of the ≤30 resident
waves are simultaneously fat. This is the cheapest possible first step — it changes *scheduling*
(burst granularity), not the LDS layout, and reuses Run-4's exact bank geometry.

### Path B — partition (`G_resident < G`), an *optional* throughput lever behind a proven bind

Only if Path A binds but leaves throughput on the table do we shrink banks to raise the **count** lever
too. Needs the §7 boundary resolution. Corrected feasible points (SEGK=16, OPSTRIDE=4096, 1 WG/CU),
requiring `POOL_N·4096 + G_resident·8192 ≤ 65280`:

| POOL_N | G_resident | LDS (+256) | count = POOL_N·G_r | binds by count? |
|---|---|---|---|---|
| 7 | 4 | 61,696 | 28 | **yes** (≥26) |
| 9 | 3 | 61,696 | 27 | **yes** |
| 5 | 5 | 61,696 | 25 | marginal |

**Key finding (revised):** one WG at `WAVES=30` can bind *either* by count (Path B, `POOL_N·G_resident≳26`)
*or* by duty-cycle (Path A, large J) *or* by both. Path A is the milestone-1 default because it is
correct-by-construction (no boundary hazard) and free (no layout change). `SEGK=16` (halved operand slots)
is what buys headroom on *both* paths. A further lever — dropping banks small enough for **2 WG/CU** — is
noted but not pursued until 1-WG/CU binding is characterized.

## 5. B-stationary residency + the cache math (why re-fetch is free)

*(Relevant only to Path B, where splitting a tile into `G/G_resident` groups re-consumes B once per group.
Path A re-fetches nothing new.)* Measured gfx1201 caches: **L2 = 8MB, L3/Infinity-Cache = 64MB**
(`rocminfo`, chip 0x7551).

- One B panel `B[:,tcol]` = `K·FN·16` fp8 = **128KB** (oracle) … **576KB** (real ml8 down, K=9216).
- Under **B-stationary** scan order (co-resident WGs share `tcol` — the existing `KMAJOR` hook), the L2
  working set = a *handful* of shared panels = **0.5–4.6MB** → fits the 8MB L2 with margin.
- Safety net: the **entire B** fits L3 for our shapes — real down B = 9216×2560 fp8 = **23.6MB < 64MB**;
  oracle B = 2MB < 8MB L2. After first touch, every B (re)access is **at worst an L3 hit, never HBM.**

Re-fetching B per group is a cache hit **by construction**. (DE frame: a partitioned hash-agg — fewer
group-by states in `work_mem`/LDS → more concurrent workers; the re-scan of the small dimension table B
stays in the buffer pool.)

## 6. Design

- **Reduction (H1 fix, both paths):** at rowblk-claim time, **zero-initialize** the `G_resident` banks
  (`ds_store 0`). Drop the `ksi==0` special case. *Every* merge is `ds_add_f32`.
- **Burst = a ksi-run:** a fat compute wave claims `(tile, rowblk, ksi_lo)` and grinds **J** consecutive
  ksi `[ksi_lo, ksi_lo+J)` in the **VGPR ACC** (WMMA-accumulate, no LDS between them), then does **one**
  `ds_add` merge of the ACC into the rowblk's bank, shrinks, and re-claims. **J is a build/dispatch knob.**
- **Enumeration:** coordinator emits `(tile, rowblk, ksi_run)` where `ksi_run` indexes `⌈n_kseg/J⌉`
  runs. Completer stores each rowblk's bank to C **once** (plain `global_store`, no atomic) after its last
  run merges; banks recycle.
- **Path A (default):** `G_resident = G` (whole tile, all rowblks' banks co-resident). No group axis, so
  no group-boundary drain — only the existing tile-boundary drain. `SEGK=16` → `POOL_N ≤ 3`.
- **Path B (optional, post-bind):** `ACC_N = G_resident < G`; add a `group ∈ [0, G/G_resident)` axis;
  B re-staged per group from L2/L3; requires §7 drain-before-admit. *(Impl caveat: OPSTRIDE/operand-staging
  currently bake `G=6`; Path B must re-derive staging for `G_resident`.)*
- **Waves:** `WAVES = 30` (kernel cap; barrier/mailbox already supports ≤30).
- **Unchanged:** on-chip `ds_add_f32` reduction (fp32), count-to-WAVES exit barrier, deadman throttle,
  chunking + compositor yield.

## 7. Correctness

- fp32 on-chip reduction preserved → oracle `bad=0` expected (same math class as write-once).
- **H1 (was a silent race — now resolved):** zero-init banks + all-`ds_add` ⇒ order-independent ⇒ correct
  under any number of concurrent same-rowblk runs. This is a *precondition* for J>1 and for `POOL_N>1`.
- **Path A boundary hazard: none.** Whole-tile groups have no intra-tile group boundary; the pool can only
  span a tile boundary, handled by the existing tile-drain.
- **Path B boundary hazard (H5):** with `POOL_N>1` the pool can span two groups whose rowblks map to the
  *same* (group-relative) banks; group g+1's runs would `ds_add` into group g's not-yet-stored sums.
  **Resolution = drain-before-admit:** the coordinator does not admit group g+1 until group g's banks are
  stored + `s_wait` (serializes at *group* granularity only). Slot-indexed banks (`POOL_N·G_resident`
  banks) are rejected — too much LDS. This is why Path A goes first.

## 8. Gates (sequenced, per kmbandy)

1. **Correctness** — builds, W-anything, oracle `bad=0`, brick-free. Flag-gated so `J=1, G_resident=G`
   is byte-identical to today's write-once (the safe fallback).
2. **Bind** — at a work-heavy shape, sweep **J** (Path A) until `grow-fail > 0` — the proof the stagger
   engages. *This is the milestone Run 4 failed, and J is the new lever to clear it.*
3. **Is the bind the *real* constraint? (the honesty gate — Fable's caveat):** duty-cycle headroom only
   converts to throughput **if VGPR is the binding occupancy constraint.** A prior TF finding hinted one
   cap was **structural (tile geometry), not VGPR.** So at Gate 2 also record whether TF *moves* with J —
   if `grow-fail>0` but TF is flat, the wall is elsewhere and Path B/POOL_N gymnastics won't help. Measure
   before building more.
4. **Throughput** — TF vs the write-once bin; sweep `J × POOL_N × (G_resident) × SEGK × WAVES`.

## 9. Open questions for the council

1. Does `ds_add` **contention on a shared bank** (many concurrent same-rowblk runs merging into one bank[r])
   become the new wall as J shrinks / POOL_N grows? (J>1 *reduces* merge frequency — is it enough?)
2. Is `instantaneous-fat = count × duty-cycle` the right silicon model, or does something else cap first
   (claim-rate, coordinator throughput, feed bandwidth, the structural/tile-geometry cap from Gate 3)?
3. Optimal `J` for a target shape — and does the best J leave *enough* occupancy headroom to still stagger,
   or does binding Path A force J so high it's effectively a square wave (H2's 91%-duty fragility)?
4. If Path A binds but Gate-3 says the wall is structural, is Path B worth building at all — or does the
   answer become "fix tile geometry," not "shrink banks"?
5. Any reason B-stationary + Path-B re-staging thrashes L2 in a way the §5 math misses?

## 10. Instrumentation TODO (before Gate 2)

Wire the **dead `occ[58]` FATMAX** counter (defined, never written; non-TRACE) to record the running max of
instantaneous-fat, so Gate 2 can *measure* duty-cycle × count directly instead of inferring it from
`grow-fail`. `occ[57]` FATLIVE (current fat count) pairs with it. Without this we can only see *whether* the
pool binds, not *how close* a non-binding config came — which is exactly what J-sweeping needs.
