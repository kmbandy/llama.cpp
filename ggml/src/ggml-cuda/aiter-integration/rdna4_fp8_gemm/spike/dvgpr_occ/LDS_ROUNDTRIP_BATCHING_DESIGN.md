# DSWS — Cutting the 35% Fixed Per-Event Cost (LDS round-trip batching)

**Date:** 2026-08-08 (after §87/§88) · **Status:** code APPLIED in-tree, OFFLINE only (no build run) · **Target:** `occ_kernel_dsws_flow.s` + `build_flow.sh`

> ## ⚠ TREE STATE (2026-08-08 night) — code is ALREADY APPLIED, NOT YET BUILT
> The kernel edits in §1/§2 and the `build_flow.sh` wiring are **committed to the working tree** (uncommitted,
> like everything else here). `BATCHLDS=0` / `WIDESLOT=0` are the defaults, so a bare `./build_flow.sh` must
> reproduce the canonical `58e965a46f3e162d` **byte-identical** — verify that FIRST before any `BATCHLDS=1` build.
> **No GPU dispatch until the full-stride oracle is clean.**

> **This is a measurement-arm design, not a commitment to ship.** Every item below is a
> correctness-preserving ablation that changes the *claim / coordination* path. A wrong address there is a
> **silent wrong C** — the oracle + WORK-EXACT gate is the only thing that has ever caught this class.
> **No GPU dispatch until the full-stride oracle is clean.**
>
> Read first (the evidence this attacks): `DSWS_TESTING_LOG.md` §85, §87, §88 · `GUARD_ABLATION_REPORT_2026-08-08.md`.

---

## 0. TL;DR — what the 20.8 ns/event is, and the one sentence that proves it

**LEANGUARD (§87) removed 682 instructions (−9.5% of `.text`, −38% of all guard bookkeeping on the hottest
coordination paths) and moved per-event time 0.0%.** The exec-guard ALU is on the **SALU pipe**, which issues
*in parallel with* the LDS/MEM pipe — so it was already fully hidden under the `s_wait_dscnt` of every LDS
access. The cost is **not instructions**. It is the **serialized LDS round-trips**: every `lds_*` accessor is

```
v_mov_b32  v[RG_A], addr
ds_load_*  v[RG_D], v[RG_A]
s_wait_dscnt 0x0            // FULL drain wait — serializes THIS access
v_readfirstlane_b32 dst, v[RG_D]
```

A wave doing one real event **plus** one failed peek serializes **12–18 such round-trips** (claim, flush,
`drain_advance`, `da_peek`), each ~30–40 LDS-latency cycles with a hard wait. That is the fixed 20.8 ns. The
math (`ds_add`, WMMA) is ~free — your own NOWMMA/NODSADD/NOCFLUSH/NOBLOAD ablations proved compute is not the
bound.

**The fix family: (1) fuse independent loads to share one wait, (2) wide-load the slot control block so one
wait returns many fields, (3) amortize the fixed cost over more work-per-event, (4) cut the poll/park loop.**

> ⚠ **A reframe that matters:** `grow-fail = 0` in every run ever. The baton/traveling-peak machinery is
> runtime-inert — the physical VGPR file never binds. Waves coast because **there is no staged work to claim**,
> not because the peak is full. The 20.8 ns is the **coordination fabric**, not the wave economy. Fix the
> fabric first; then the fat-tile / wave-count levers reopen.

---

## 1. The primary arm: `BATCHLDS` — fuse independent LDS reads into one wait

**What it does:** replaces N serialized `lds_get` round-trips with N `ds_load_b32` issued back-to-back + **one**
`s_wait_dscnt 0x0` + N `v_readfirstlane`. Identical results, ~N× fewer waits. This is the direct, causal attack
on the fixed term — the sibling of LEANGUARD that actually touches the waits.

**Correctness:** pure reordering/deferral of waits. No new atomics, no new writes, no CAS semantics change.
It is a **correctness-preserving** knob (like LEANGUARD), **not** ORACLE-INVALID (unlike NODSADD/NOBLOAD).

### 1.1 Add the defsym — `occ_kernel_dsws_flow.s`, levers block (~line 253, after `NODSADD`)

```asm
    .set BATCHLDS, 0     // FUSE INDEPENDENT LDS READS INTO ONE WAIT (2026-08-08): replaces N serialized
                         //   lds_get round-trips (each with its own s_wait_dscnt 0x0) with N ds_load_b32
                         //   back-to-back + ONE wait. Pure wait-deferral -> correctness-preserving.
                         //   Targets the §87 LEANGUARD conclusion: the 20.8 ns/event fixed term is LDS
                         //   round-trip latency, not instruction issue. Oracle-valid; gate on full oracle.
```

### 1.2 Fuse the claim-path `SL_RBNEXT` + `SL_STI` reads — `occ_kernel_dsws_flow.s:5315`

The STI read (line 5340) is **fully independent** of the claim CAS — it depends only on `s48` (slot base),
which is stable. So pre-load it with the RBNEXT read and drop the post-CAS re-read.

**Replace (line 5315–5316):**
```asm
    s_add_u32 s45, s48, SL_RBNEXT
    lds_get_r s33, s45                            // x = current SL_RBNEXT state
```
**with:**
```asm
    s_add_u32 s45, s48, SL_RBNEXT                 // keep s45 = &SL_RBNEXT for the CAS target below
.if BATCHLDS
    // BATCHLDS: read SL_RBNEXT + SL_STI in ONE round-trip. STI is independent of the claim CAS, so
    //   pre-load it now and drop the post-CAS SL_STI re-read. v14=addr, v11/v12=data (all interior
    //   <=v15, pre-grow safe). v11/v12 are dead here (CAS clobbers v11/v13 later, after extraction).
    v_mov_b32 v[RG_D], s48
    ds_load_b32 v11, v[RG_D] offset:SL_STI        // STI  -> v11
    ds_load_b32 v12, v[RG_D] offset:SL_RBNEXT     // x    -> v12
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s17, v11                   // gsti (stable: slot pinned by the pending/claim protocol)
    v_readfirstlane_b32 s33, v12                   // x = SL_RBNEXT
.else
    lds_get_r s33, s45                            // x = current SL_RBNEXT state
.endif
```

**Replace (line 5339–5340):**
```asm
    s_and_b32 s33, s33, NEXT_MASK                  // s33 = k = rowblk index (0..ACC_N-1)
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45                             // gsti of the CLAIMED occupant (stable: pinned by the claim)
```
**with:**
```asm
    s_and_b32 s33, s33, NEXT_MASK                  // s33 = k = rowblk index (0..ACC_N-1)
.if BATCHLDS
    // (STI already read into s17 at the top of the claim -- no SL_STI re-read round-trip.)
.else
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45                             // gsti of the CLAIMED occupant (stable: pinned by the claim)
.endif
```

### 1.3 Fuse the post-grow `DRAIN_HEAD` + `STAGE_HEAD` reads — `occ_kernel_dsws_flow.s:5284–5285`

**Replace:**
```asm
    lds_get s46, DRAIN_HEAD_OFF                   // fresh dh (post-grow) -- overrides the pre-grow read
    lds_get s44, STAGE_HEAD_OFF                   // fresh sh
```
**with:**
```asm
.if BATCHLDS
    v_mov_b32 v[RG_D], 0
    ds_load_b32 v11, v[RG_D] offset:DRAIN_HEAD_OFF
    ds_load_b32 v12, v[RG_D] offset:STAGE_HEAD_OFF
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s46, v11                  // fresh dh (post-grow)
    v_readfirstlane_b32 s44, v12                  // fresh sh
.else
    lds_get s46, DRAIN_HEAD_OFF
    lds_get s44, STAGE_HEAD_OFF
.endif
```
> `DRAIN_HEAD_OFF` / `STAGE_HEAD_OFF` are in the shared frontier region (bytes 0–256), well within
> `ds_load_b32`'s signed-16-bit offset range, so a zero base + offset works.

---

## 2. The structural arm (v2, higher value): wide-load the 32-byte slot control block

Your per-slot control block is **eight consecutive u32 in one 32-byte aligned slot** (`SLOTC_BASE + slot*32`):

```
SL_STI(0) SL_GEN(4) SL_RBNEXT(8) SL_RBDONE(12) SL_BFNEXT(16) SL_BFDONE(20) SL_ARNEXT(24) SL_ARDONE(28)
```

The claim path reads `RBNEXT`, `STI`, `GEN` as **three separate serialized round-trips** today. A single
`ds_load_b128` returns `STI/GEN/RBNEXT/RBDONE` in **one wait**. This is the single most impactful change
available because the slot layout was *designed* for it and you're not using it that way.

**Caveat (address-reg conflict):** `ds_load_b128 v[11:14], v[11]` is a hazard (addr reg is also a dest). The
address must live in a register *not* in `v[11:14]` — use `v15` (`RM_A`, free in the claim region; it's only
used by the LEANMARSH `DRAIN_HEAD` store, which the claim path does not reach).

**Replacement for §1.2's fused read (BATCHLDS=1, WIDESLOT=1 — use instead of the b32 pair):**
```asm
    s_add_u32 s45, s48, SL_RBNEXT                 // keep &SL_RBNEXT for the CAS target
.if BATCHLDS && WIDESLOT
    // WIDESLOT: one ds_load_b128 returns STI/GEN/RBNEXT/RBDONE in ONE wait. v15 (RM_A) = addr
    //   (free here; not in v[11:14]). MUST v_readfirstlane RBNEXT out of v13 BEFORE the CAS below
    //   clobbers v13. RBDONE(v14) + GEN(v12) are free bonus reads for the drain path.
    v_mov_b32 v[RM_A], s48                        // addr = &slot[0]  (32B aligned -> b128-safe)
    ds_load_b128 v[11:14], v[RM_A]                // STI=v11 GEN=v12 RBNEXT=v13 RBDONE=v14
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s17, v11                   // gsti
    v_readfirstlane_b32 s33, v13                   // x = SL_RBNEXT
.elseif BATCHLDS
    ... §1.2 fused b32 pair ...
.else
    lds_get_r s33, s45
.endif
```

**Why it's worth the extra care:** the same wide slot read can feed the **flush → drain** path
(`drain_advance` currently does ~5 separate round-trips: `DRAIN_HEAD`, `STAGE_HEAD`, `SL_GEN`, `SL_RBDONE`,
then the CAS). With the slot wide-loaded once at claim, `RBDONE`/`GEN` are already in registers when the
burst flushes — collapsing a large fraction of the post-burst serialization too. This is the natural v2 once
§1's b32 batching is proven causal on silicon.

---

## 3. Amortize the fixed cost: bigger work-per-event (no new code — run the built arms)

Your own model: `base` vs `fm4fn2` = 0.3% apart at matched work-per-event → **the frag grid is irrelevant; the
*event* is the unit of taxation.** The win is more rowblks/K per claim, not more frags. Two already-built levers:

### 3a. `SEGK_STAYFAT` (C1 arm) — run the pre-registered SEGK_ATTRIBUTION matrix
Already implemented + gated (`SEGK_ATTRIBUTION_DESIGN_2026-08-08.md`). It skips the grow/shrink between
consecutive bursts while the wave stays fat. Run the 12-cell matrix (3 baseline + 9 ablation) exactly as
specified there. **Do not** time it as a correctness claim (it is labeled ORACLE-INVALID until separately
gated) — its job is to price the grow/shrink share of the per-segment slope.

### 3b. `BATCH` cursor-batch — `occ_kernel_dsws_flow.s:238`, default 1
Claims N consecutive ASSIGN indices per reservation, amortizing the claim CAS + drain across N rowblks.
It's a defsym (default 1); sweep it up. This is the direct "reduce event count" lever.

### 3c. Larger SEGK headroom
SEGK is the strongest axis (17.5/10.2/5.6 TF at 256/128/64) and its per-segment cost is *not* in the
per-event model. If §3a shows grow/shrink is material, `SEGK_STAYFAT` + a relaxed `DUTY_KMAX` opens the axis
above 256. Only after §3a prices it.

---

## 4. Cut the poll/park loop (stage 5+6 = 54% of wave-time)

Since `grow-fail = 0`, waves park because **there's no work, not because VGPR is full**. Two cheap knobs:

- **Right-size WAVES.** §85: `waves16` is a real −14% with `coast/computed = 13.0` (10× the poll passes).
  At 1 WG/CU the optimum is "enough to feed, not enough to poll." Sweep 3–8 with the §81 clock-normalized
  instrument (raw TF is meaningless across cells).
- **Give the coasters real work.** `door1 = 100% of coast` means the pool is always empty. Items 1–2 raise
  the staging/stage rate, which is what gives coasting waves something to do — the tier ladder then works as
  designed.

---

## 5. Build wiring — `build_flow.sh`

Add `BATCHLDS` (and optionally `WIDESLOT`) to the three places the other arms live:

**Defaults block (~line 49–50):**
```bash
: ${LEANGUARD:=0}; : ${GUARDHOIST:=0}; : ${LEANMARSH:=0}
: ${SEGK_STAYFAT:=0}
: ${BATCHLDS:=0}; : ${WIDESLOT:=0}
```

**Print/audit list (line 85):** add `BATCHLDS WIDESLOT` next to `LEANGUARD GUARDHOIST LEANMARSH SEGK_STAYFAT`.

**`mkflow` -Wa,-defsym line (line 142):** add
```bash
-Wa,-defsym,BATCHLDS=${BATCHLDS:-0} -Wa,-defsym,WIDESLOT=${WIDESLOT:-0} \
```
(append to the same line that carries `LEANGUARD ... SEGK_STAYFAT`).

> **Refusal policy:** BATCHLDS/WIDESLOT are **correctness-preserving** (wait deferral only), so unlike
> `SEGK_STAYFAT` they do **not** need `DSWS_ALLOW_NONSTD`. But they touch the claim path → **every build must
> pass the full-stride oracle + WORK-EXACT before any TF number is quoted.** A wrong address is a silent wrong C.

---

## 6. Verification & measurement protocol (mirror §87 — same-session, fixed-rep)

1. **Assemble + static gate:** `RGA` 0-spill, `SGPR=72` unchanged, `.text` delta only from the fused loads.
   Record SHA-256 of each bin. `BATCHLDS=0` must reproduce the canonical `58e965a46f3e162d` exactly (the
   `.else` arms are byte-identical).
2. **Bring-up (one dispatch, then STOP):** chunk=64, `DSWS2_REPS=1`, **full stride=1 oracle (320/320 tiles)
   bad=0**, WORK-EXACT, `occ[0]=0`, `occ[95]=0`, canary clean, no reset. The §1.2 STI pre-load is the
   correctness-critical bit — a wrong `s17` is a silent wrong C.
3. **Measure (same-session A/B, fixed `DSWS2_REPS`, `SSWIN=32`):** run OFF vs BATCHLDS at the config of
   record (`WAVES=6 FM=2 FN=4 G=8 ACC_N=4 SEGK=256`, `ML8_POOL=64`), plus the `fm1` control cell for the
   two-point fit. Re-fit `ns_per_event = b0 + 37.36×MFLOP`.
4. **Decision rule (pre-registered):** if `b0` (currently ~19.9–20.8 ns) drops **≥ ~4 ns** with BATCHLDS,
   the round-trip-batching hypothesis is causal → pursue WIDESLOT (§2). If `b0` does not move, the fixed term
   is the **polling cadence / contention** (waves hammering the same words), and the pivot is §4
   (right-size WAVES, reduce pollers) — a *result*, not a failure.
5. **Then:** WIDESLOT arm; then §3a (SEGK_STAYFAT matrix); then §4 (WAVES sweep, clock-normalized).

---

## 7. Expected result & why it's worth it

The 20.8 ns fixed term is **35% of the 60 ns event** at the config of record. §1 collapses the claim path
from 2 round-trips to 1 and the post-grow path from 2 to 1 — a ~2–4× reduction in the *claim* serialization.
§2 collapses the whole slot read to one wait. If even half of the fixed term is claim/coordination round-trip
latency, this is a **double-digit-% throughput** move — and unlike every instruction-economy cut (measured
flat twice), it attacks the *waits* that the LEANGUARD result proved are the real cost.

---

## 8. Safety & standing gates (do not skip)

- **Offline-first.** All of the above assembles + RGA's on the CPU. No GPU dispatch until 6.2 passes.
- **One greenlit dispatch at a time**; new/changed kernel = one bring-up then STOP. `DEADMAN_TICKS` stays 0.5s.
- **Oracle + WORK-EXACT are the gate for every TF claim**, not throughput. A wrong claim-path address is silent.
- **Stay at M=576 / 1 WG/CU** for any new arm until the occupancy axis is re-justified.
- **Do not run PHASEPROBE** (unthrottled RTC read = the 07-14 brick vector). Use STAGINSTR counters.
