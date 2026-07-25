# Decentralized assign v3 — LOCK-FREE, per-slot generation tag (occ_kernel_dsws_flow.s, DECENTASN)

## Context / failure history (why v3 exists)
Goal: many waves produce super-tiles into a POOL_N-slot ring IN PARALLEL, so refill-rate scales with
#free waves and the ASSIGN-bound wall (every real ml8 shape 92-100% ASSIGN-BOUND) breaks.

- Single-writer coordinator (baseline): 1 producer, ~1 super-tile/loop-iter. ASSIGN-bound.
- BATCHASN (wave0 fills pool per visit): no help -- deepens per-visit, not per-time. Still bound.
- DECENTASN v1 (lock-free, reserve + in-order ASSIGN_HEAD publish): oracle BAD, barely produced
  (occ[20]=2572 vs 12288). FLAW: a wave PARKED in a room-wait holding a reservation got
  deadman-retired -> left a permanent hole in the in-order ASSIGN_HEAD publish chain -> every
  higher-index wave waited forever -> cascade. TWO root causes: (a) PARKING (wave holds a
  reservation while spin-waiting for room), (b) the in-order monotonic ASSIGN_HEAD is a single
  ordering point a dead wave can wedge.
- DECENTASN v2 (mobile-lock: cmpswap a mutex, one wave batch-fills, releases): oracle 90% ok
  (produced ALL work: occ[20]=12397 ~= 12288), but STILL 99.8% ASSIGN-BOUND / TF=0.0, because a
  LOCK serializes production == mobile batch-assign == the wall it was meant to remove. Also a
  residual ~10% race (444/4608 bad, max_rel=0.46) never pinned.

## v3 invariants (the two flaws, each explicitly killed)
1. NO PARKING. A producer never holds a reservation while waiting. It reserves ONLY via a CAS
   that succeeds, and the reserve->stamp->ready sequence has NO wait in it (a few LDS writes,
   microseconds << deadman). If there's no room or the CAS loses, it holds NOTHING and returns
   to being a worker.
2. NO SINGLE ORDERING POINT. No monotonic ASSIGN_HEAD on the producer side. Each slot carries
   SL_GEN = its reservation index; consumers verify a slot by SL_GEN, so a dead/slow producer
   stalls at most one slot briefly, never a whole chain.

## LDS state
- RESV_HEAD (new u32): monotonic reservation counter. slot = RESV index mod POOL_N.
- DRAIN_HEAD (existing): oldest not-fully-drained reservation index.
- STAGE_HEAD (existing): oldest not-fully-staged reservation index.
- Per slot: SL_GEN (existing field, currently unused in flow) = the reservation index this slot
  currently holds. SL_STI = sti (=gi for pow2). Counters RBNEXT/RBDONE/BFNEXT/BFDONE/ARNEXT/ARDONE.
- ASSIGN_HEAD: RETIRED under DECENTASN (or repurposed = RESV_HEAD). Consumers stop reading it.

## Producer (a starved wave at .Lflow_feed_empty), lock-free
    // pow2 guard (sti==gi) else fail-safe; FLOWTERM check.
  .retry (bounded, e.g. <= 4 tries then bail to help):
    r = RESV_HEAD ; d = DRAIN_HEAD
    if r - d >= POOL_N: bail -> go help (pool full; NO reservation held)
    if CAS(RESV_HEAD, r -> r+1) lost: (someone else took r) -> re-read, retry (bounded)
    // WON slot index r (r - d < POOL_N held at CAS -> slot r's POOL_N-ago occupant is drained)
    gi = atomic_inc(occ[20])            // GLOBAL work claim
    t = gi >> shift ; if t >= chunkHi: terminal (see below)   // NOTE: RESV already incremented -> see Q1
    slot = r mod POOL_N
    stamp slot: RBNEXT(=0 or J-poison)/RBDONE/BFNEXT/BFDONE/ARNEXT/ARDONE = 0 ; SL_STI = gi
    SL_GEN[slot] = r                    // RELEASE, written LAST (fence: slot fully stamped)
    go help / loop                      // NO wait anywhere after the CAS

## Feeder (replaces the STAGE>=ASSIGN gate)
    s = STAGE_HEAD
    slot = s mod POOL_N
    if SL_GEN[slot] != s: nothing produced at the head yet -> yield (.Lflow_feed_empty -> producer path)
    ... stage A or B (existing ASTAGE_R / BSTAGE_R) ...
    // STAGE walk: while slot(STAGE) fully staged (BFDONE==FN && ARDONE==G) AND SL_GEN(slot)==STAGE: CAS STAGE++
Compute/DRAIN: UNCHANGED (drains [DRAIN, STAGE); drain_advance walk/bump as today).

## Terminal (out of global work)
A producer that draws gi >= chunkHi: it has ALREADY incremented RESV (Q1). Options:
  (chosen) it stamps slot r with a TERMINAL marker SL_GEN = r but SL_STI = SENTINEL, and sets a
  per-slot "dead" so the STAGE walk skips it and DRAIN treats it as instantly-drained... TOO complex.
  (simpler, RECOMMENDED) claim gi FIRST, check terminal, and ONLY reserve (CAS RESV) if gi < bound.
  Reorder: gi = occ[20]++ ; if gi>=bound -> set FLOWTERM, drain-watch(RESV), retire (NO RESV taken).
           else CAS-reserve r ; stamp ; SL_GEN=r.
  Then RESV only ever counts PLACED super-tiles -> drain-watch waits DRAIN>=RESV (exact, no hole).
  Cost: a gi is claimed before room is known; if pool full after claiming, we hold gi but no slot.
  -> gi would be LOST (claimed, never placed) -> DROPPED work. BAD.
  => Need: claim gi and reserve slot as one commit. Since they are two atomics (global + LDS),
     do: reserve slot FIRST (CAS RESV, gated by room). WON -> claim gi. gi>=bound -> this slot is
     a "no-op": publish SL_GEN=r with SL_STI=SENTINEL; consumers must treat SENTINEL slot as
     zero-work (stage: skip; drain: advance immediately). Exactly ONE such no-op slot per WG at the
     tail. This keeps RESV dense and the walks gap-free.  <-- Q2: verify consumers handle SENTINEL.

## Open questions FOR THE ADVERSARIAL REVIEW
Q1. RESV-before-gi vs gi-before-RESV ordering: which avoids both (a) a lost gi and (b) a RESV hole?
    The doc leans reserve-first + SENTINEL no-op tail slot. Is there a cleaner commit order?
Q2. SENTINEL tail slot: do the feeder (stage) and compute (drain) paths correctly no-op a slot
    whose SL_STI == 0xFFFFFFFF? If not, that slot wedges STAGE/DRAIN.
Q3. SL_GEN release ordering: producer writes counters then SL_GEN=r LAST. A feeder reads
    SL_GEN==STAGE then reads the slot's operands/counters. Is LDS visibility guaranteed s.t. the
    feeder can't see SL_GEN=r but stale counters? (lds_put does ds_store + s_wait_dscnt 0x0 each;
    is that a sufficient fence within a WG for another wave's subsequent read?)
Q4. Slot reuse: r and r+POOL_N share a physical slot. r+POOL_N reserves only when
    (r+POOL_N)-DRAIN < POOL_N -> DRAIN > r -> slot r drained. When r+POOL_N writes SL_GEN=r+POOL_N,
    could a LAGGING consumer still be reading SL_GEN==r for the old occupant? (DRAIN>r means the
    old occupant fully drained -> STAGE and DRAIN both passed r -> no consumer on it. Confirm.)
Q5. The residual v2 10% race (444 bad, max_rel=0.46 partial over-accumulation) -- does ANY of it
    come from a mechanism v3 shares (slot reuse, stamp/consume ordering), or was it lock-specific?
Q6. Bounded CAS-retry: if 30 waves hammer RESV_HEAD CAS, is starvation/livelock possible? (bounded
    retry -> bail to help is the backstop; confirm no correctness issue if a wave bails mid-contention.)
Q7. Deadman coverage: the producer has NO wait after CAS, so no deadman needed there. The feeder
    "SL_GEN != s -> yield" path loops via .Lflow_feed_empty (has s_sleep + loop-head deadman). OK?

## Build/verify plan
DECENTASN=0 byte-identical (386dc28). DECENTASN=1 assembles. Oracle-gate (CLEAN + computed exact)
BEFORE any real-shape run. moe_attn_kv (n_kseg=32 pow2) is the indicator: does STARVATION collapse?
