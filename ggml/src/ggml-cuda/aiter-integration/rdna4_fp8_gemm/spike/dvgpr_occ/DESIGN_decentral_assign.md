# Decentralized "assign is a role" — design (flat global super-tile index)

Gated behind **DECENTASN** (default 0 → byte-identical). WOFLUSH=1 only (super-tiles independent).

## The problem it fixes
Every real ml8/mlambaformer shape is 92–100% ASSIGN-BOUND: wave 0 is the single producer,
publishing ~1 super-tile per loop iteration into a pool, for 30 consumers. Batch-assign
(fill the pool per visit) did NOT help (moe_attn_kv 99.9%→99.2%, TF 0.0→0.2) because it
changes how MUCH per visit, not how OFTEN. Fix: make assign a ROLE any starved wave does →
refill rate scales with #free waves.

## Two indices (the crux)
- **gi** — GLOBAL super-tile id, claimed from occ[20] (cross-WG atomic). Identifies the WORK
  (decodes to sti → mblk/tcol/ksi → A/B/C addresses). Range [0, chunkHi_super).
- **r (li)** — LOCAL pool slot index, per-WG. Drives slot = r mod POOL_N and the STAGE/DRAIN
  frontier. ASSIGN_HEAD is the published r.
These are INDEPENDENT counters. Which gi lands in which local slot does not matter — each
slot's STAMP carries its gi's sti, and under WOFLUSH order is irrelevant (commutative atomic-add).

## Decode gi → sti (needs a divide; n_kseg is NOT pow2 in general)
sti is sparse: sti = (t << shift) | ksi, shift = ceil-log2(n_kseg), ksi ∈ [0,n_kseg).
gi is dense: gi = t*n_kseg + ksi.  So:
    t   = gi / n_kseg          ← divide by non-pow2 runtime n_kseg
    ksi = gi - t*n_kseg
    sti = (t << shift) | ksi   ← then the EXISTING DECODE_STI works unchanged
HOST passes a magic reciprocal (magic_kseg M, extra shift S) so t = mulhi(gi,M)>>S — ~5 SALU,
race-free. (The v1 contract already carried magic_kseg; we re-add it.) Store M/S in spare SGPRs
(s66 is free after the prologue derives mask/shift).

## The assign operation (a starved wave), gap-free, in-order publish
Inserted at .Lflow_feed_empty (where a starved wave currently just s_sleep + loops):

  1. gi = atomic_inc(occ[20])          [GLOBAL]   // claim work FIRST (cheap terminal check)
     if gi >= chunkHi_super:                        // out of global work
         set FLOWTERM; broadcast ROLE_RETIRE to all; retire self.   // single clean terminal, NO slot held
  2. r = atomic_inc(ASSIGN_RESV)       [LDS]      // reserve a local publish index (NEW counter)
  3. POOL-ROOM WAIT: while (r - DRAIN_HEAD >= POOL_N) { deadman_check; s_sleep }   // bounded: compute drains
  4. slot = r mod POOL_N; decode sti from gi; STAMP slot (reset RBNEXT/RBDONE/BFNEXT/BFDONE/
     ARNEXT/ARDONE/GEN, write SL_STI = sti).       // same stamp body as today's coordinator
  5. IN-ORDER PUBLISH: while (ASSIGN_HEAD != r) { deadman_check }   // wait my turn
     lds_cmpstore_adv(ASSIGN_HEAD, r)  → r+1        // idempotent CAS, same primitive DRAIN uses
  6. s_branch .Lflow_loop                            // then go be a normal wave (compute/feed)

### Why gap-free (no ASSIGN_HEAD stall)
Every claimed gi (< bound) gets a reserved r (dense, monotonic) and is ALWAYS placed — step 3
WAITS for a slot rather than dropping the work. So ASSIGN_HEAD advances 0,1,2,… with no holes.
The only non-placement is gi >= bound, which never reserves an r (step 1 returns before step 2).

### Why no torn read
RESV (reserve) and HEAD (publish) are SEPARATE. A consumer reads ASSIGN_HEAD, which only
advances AFTER the slot at r is fully stamped (step 5 gated on step 4). Same discipline as the
old single-writer, now multi-writer-safe via the in-order CAS.

### Why the pool-room WAIT is bounded (no deadlock/brick)
r - DRAIN >= POOL_N means the pool is full → some slot is staged & being computed → DRAIN WILL
advance. deadman_check in the wait converts a genuine wedge into a clean retire (no MODE1).
Same for the in-order-publish wait: lower r's are held by waves doing a few LDS writes, no
external dependency → they publish, ASSIGN_HEAD reaches r.

## Concurrency safety summary (the MSSCAN lesson)
- occ[20] atomic_inc: dense unique gi per wave. No dup, no skip. (Cost: ~n_kseg× more global
  atomics vs per-tile. Acceptable; if IT becomes the wall, batch-claim K gi per atomic.)
- ASSIGN_RESV atomic_inc: dense unique r. Slot = r mod POOL_N distinct within any POOL_N window.
- Two waits, both bounded + deadman-guarded.
- STAMP before publish (release ordering via in-order CAS).
- Correctness backstop: oracle CLEAN + work-exactness (computed == chunkHi_super × G × chunks).

## What wave 0 keeps vs loses (under DECENTASN)
KEEP: one-time init — RINGINIT (0xACED publish), initial slot-counter zeroing, ASSIGN_RESV=0.
LOSE: the per-iteration single-assign block (the .if wid==0 coordinator duty). Under DECENTASN
wave 0 becomes a peer that also assigns-on-starve. flow_snapshot (STAGINSTR instrumentation)
→ gate to any-wave or drop when !STAGINSTR. Terminal RETIRE broadcast → done by whoever hits
gi >= bound (step 1).

## Host co-changes (occ_dispatch.cpp)
- occ[24] chunkHi: write in SUPER-TILES (chunk tile-range × n_kseg), not tiles.
- Pass magic_kseg (M, S) for the gi→t divide (a spare occ[] slot or kernarg).
- occ[20] semantics note: now counts super-tiles (the completion/liveness print already treats
  it as a liveness signal, not an exact count — banner text update only).

## New LDS word
ASSIGN_RESV_OFF — one u32 in the control block (below OP_BASE=512). Init 0 by wave 0.
Guard: it must not collide with existing frontier offsets (ASSIGN_HEAD=0, STAGE=4, DRAIN=8,
RINGINIT=12, INITFLAG=68, COORD_KSI/T tail). Place at a free offset, add a .error alias guard.

## Build/verify plan (offline first, rule 6)
1. DECENTASN=0 build → md5 == current baseline (byte-identical inert).
2. DECENTASN=1 build → assembles (all .error guards, new LDS offset).
3. Oracle bring-up (rule 2): DECENTASN=1 on the oracle shape → oracle CLEAN + work-exact.
4. THEN moe_attn_kv: read STARVATION% — expect empty-assign to COLLAPSE (the indicator).

## Open questions for review
Q1. occ[20] as super-tile counter is cross-WG; per-super-tile global atomics = n_kseg× traffic.
    Start simple (per-gi), or batch-claim from the start? → RECOMMEND start simple, measure.
Q2. Fully remove wave-0 coordinator, or keep it AS a producer alongside decentralized (belt +
    suspenders)? → RECOMMEND fully decentralized (one code path; wave 0 = peer after init).
Q3. Keep BATCHASN in the tree (inert) or rip it out now that it's a proven dead end? → keep
    inert for now; remove in a cleanup pass.
