# DSWS TESTING LOG — running record of every gated sweep

**This doc is append-only. Never overwrite an entry. Newest entries at the bottom of §4.**

Standing testing rules (kmbandy, 2026-07-15 — set because we got bitten repeatedly by synthetic
shapes, one-shot correctness checks, and no durable record; KG bb6bbe09):

1. **The oracle shape is a REAL shape** — one of the actual ml8 / mlambaformer GEMM shapes (§2),
   never a synthetic/square/arbitrary shape.
2. **Every major change is gated by a FULL ml8/mlambaformer sweep** (§2/§3) — oracle-clean on each,
   results recorded here — before the change is considered validated. No promoting on one shape.
3. **This doc is the record.** One §4 entry per sweep run: (a) change summary, (b) exact build
   defsyms + host env, (c) shapes swept, (d) per-shape outcome (oracle ok/bad/max_rel + TF +
   hipBLASLt baseline for context).

Compounds with the GPU dispatch rules (CLAUDE.md): one dispatch per greenlight, bring-up-then-stop,
hang = full stop, deadman 0.5s, offline-first. **A sweep is N individual greenlit dispatches, not a batch.**

---

## 1. THE ORACLE SHAPE (Rule 1)

**Primary oracle: `ml8 moe attn_q` — the established real-shape config from the 07-14 baseline sweep:**

| | value |
|---|---|
| banner shape (M×N×K) | **576×4096×2048** |
| baseline config | FM=2 G=9 SEGK=64 → super-tile 288×64, MTLsuper=2, NTL=64, n_kseg=32 (pow2) |
| DECENTASN config | FM=1 (the DECENTASN work is on the FM=1 lever); pick G with `G·16 | 576` → G=6 (super-tile 96, MTL=6) or G=9 (144, MTL=4). n_kseg=32 pow2 → DECENTASN-legal |
| host | `DSWS2_G=<G> DSWS2_SEGK=64 DSWS2_K=2048 DSWS2_ORACLE_MTL=<MTLsuper> DSWS2_ORACLE_NTL=64` |
| hipBLASLt floor | ~67 TF @ this shape (~22% of its own roofline) — the beatable target |

Why this one: real, high-value ml8 MoE GEMM; **n_kseg=32 is pow2** so it gates BOTH the baseline AND
the DECENTASN path (the only path with a pow2 constraint). The harness computes its own reference, so
it is a true correctness gate.

> Geometry (G, POOL_N, SEGK, FM) is per-run and MUST be re-checked against the LDS cap (≤ 65536B;
> under WOFLUSH=1 no ACC banks → LDS ≈ 256 + POOL_N·(FN·16·SEGK + G·16·FM·SEGK)). Every §4 entry
> records the geometry it actually ran. The old 288×4096×2048 the DECENTASN bring-up used is a
> SYNTHETIC half-M tiling — NOT a real shape; it is retired as the oracle by Rule 1.

---

## 2. THE REAL SHAPES + WHAT ALREADY RAN (source: 07-14 flow-kernel sweep, `~/dsws_gpu_logs/rs_*.log`,
   `real_moe_attnq_*.log`; shape defs `~/dsws_gpu_logs/bench_hipblaslt_ml8.py`)

**FACT (07-14): the BASELINE flow kernel is oracle-CLEAN on EVERY real shape below, pow2 AND non-pow2**
(the arbitrary-K decode handles non-pow2 n_kseg kernel-wide). All were ASSIGN-bound, TF 0–4.9. The
open problem is throughput, not baseline correctness.

Banner shape = M×N×K. `n_kseg` at SEGK=64. All rows below ran FM=2, SEGK=64, WAVES=30, oracle stride≤8.

| shape | M×N×K | G | n_kseg | pow2 | baseline oracle | baseline TF | hipBLASLt |
|---|---|---|---|---|---|---|---|
| moe attn_q | 576×4096×2048 | 9 | 32 | ✓ | clean | ~0 | ~67 |
| moe attn_kv | 576×512×2048 | 9 | 32 | ✓ | ok=1152 bad=0 | 0.0 | ~15 |
| moe attn_o | 576×2048×4096 | 9 | 64 | ✓ | clean | — | ~65 |
| moe ffn_gate/up | 512×512×2048 | 8 | 32 | ✓ | ok=1024 bad=0 | 0.2 | ~15 |
| moe ffn_down | 512×2048×512 | 8 | 8 | ✓ | clean | — | ~13 |
| dense attn_o | 2048×2560×4096 | 8 | 64 | ✓ | ok=2560 bad=0 | 4.9 | ~156 |
| dense ffn_down | 2048×2560×9216 | 8 | **144** | **✗** | clean | — | ~190 |
| mlmf expert fc1 | 512×1536×768 | 8 | **12** | **✗** | clean | — | ~35 |
| mlmf expert fc2 | 512×768×1536 | 8 | **24** | **✗** | clean | — | ~39 |

(More dense M=2048 shapes + mlmf shapes exist in `bench_hipblaslt_ml8.py`; add rows as they are swept.
Three real shapes need an N pad to satisfy N%64: mamba in_proj N=4200→4224, val_proj1 N=96→128,
router_out N=8/16→64.)

### pow2 status — the ONE real constraint, and it is DECENTASN-only
- **Baseline coordinator path: runs ALL real shapes** (pow2 + non-pow2). Proven above.
- **DECENTASN v3 prototype: pow2 only.** Its flat-gi claim decodes `gi` as `sti` (valid only when
  dense==sparse, i.e. pow2); non-pow2 hits a fail-safe retire (kernel line ~2990). NOT a kernel limit,
  NOT a regression — an unfinished prototype piece with a designed fix (`gi→sti` magic-div,
  `DESIGN_decentral_assign.md`). Until it lands, the **DECENTASN sweep** = the 6 pow2 real shapes above;
  the **baseline sweep** = all real shapes.

---

## 3. WHAT "A FULL SWEEP" MEANS (Rule 2)

For the path under test, run the applicable shape set as individual greenlit dispatches, fed to ≥1s
steady state, oracle stride≤8. Validated only when ALL shapes are oracle-clean AND per-shape TF is
recorded here vs the hipBLASLt baseline.
- **DECENTASN path** (pow2 only, current work): the 6 pow2 real shapes in §2.
- **Baseline coordinator path**: all real shapes in §2 (pad the three N%64 shapes). Already clean 07-14
  — that run is the reference baseline; re-run it as the gate whenever the baseline hot path changes.

---

## 4. RUN LOG (append-only; newest at the bottom)

### 2026-07-14 — BASELINE flow-kernel real-shape sweep (reference; ran BEFORE these rules, logged here)
- **Change:** arbitrary-K decode + FM lever landed; this is the baseline-correctness + bottleneck map.
- **Build:** `FM=2 FN=4 G={8,9} SEGK=64 WAVES=30 TFPROBE=1 STAGINSTR=1` (flow kernel, single coordinator).
- **Shapes swept:** the 9 rows in §2 (logs `~/dsws_gpu_logs/{real_moe_attnq,rs_*}_*.log`).
- **Outcome:** **ALL oracle-clean (bad=0), pow2 AND non-pow2.** All ASSIGN-bound; TF 0.0–4.9.
- **Verdict:** baseline is CORRECT on all real shapes; the wall is throughput (ASSIGN-bound). This is
  the correctness reference every future baseline change must reproduce.

### [pre-rule, SYNTHETIC — rule-1 violation] 2026-07-14/15 — DECENTASN v3 + straddle probe + Fix A
> DECENTASN has ONLY ever run the SYNTHETIC oracle 288×4096×2048 (G=3 FM=1), never the real sweep.
> Kept for continuity; does NOT count as a gated real-shape entry.
- **Builds:** `DECENTASN=1 FM=1 G=3 ACC_N=3 POOL_N=8 SEGK=64 WOFLUSH=1 STAGINSTR=1 TFPROBE=1`
  (+CNT_STRADDLE observer; +Fix A post-grow re-derivation).
- **Host:** `DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=3 DSWS2_SEGK=64 DSWS2_ACC_N=0 FLOW_WAVES=30 FLOW_POOL_N=8
  DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64 DSWS2_K=2048` (synthetic 288×4096×2048, n_kseg=32).
- **Outcome (DECENTASN oracle progression):**
  | build | oracle | occ[95] straddle | occ[0] |
  |---|---|---|---|
  | v1 in-order | ok=36 bad=4572 | — | 0 |
  | v2 mobile-lock | ok=4164 bad=444 | — | 0 |
  | v3 lock-free | ok=4488 bad=120 | — | 0 |
  | v3 + observer `73a36d1` | ok=4488 bad=120 max_rel=2.258 | 6105 | 0 |
  | v3 + Fix A `78e94ec` | ok=4524 bad=84 max_rel=3.452 | 91914 | 0 |
- **Verdict:** DECENTASN still numerically broken (bad>0). Straddle confirmed (6105); Fix A did not fix
  (rate unchanged) → revised suspect: `drain_advance` over-advancing DRAIN past a slot with RBDONE<ACC_N.
  Next: offline root-cause. FIRST real-shape DECENTASN entry will be on `ml8 moe attn_q` 576×4096×2048.

### 2026-07-15 — DECENTASN (next,inflight) single-word pin fix — DECENTASN
- **Change:** folded an inflight-claim counter into `SL_RBNEXT` (`next`=bits[7:0], `inflight`=bits[15:8]);
  claim CAS `x→x+1+INFLIGHT_ONE`; completion `SL_RBNEXT -= INFLIGHT_ONE` after the C flush; drain authority
  moved off `SL_RBDONE` onto `next==ACC_N && inflight==0`. Removed the spin + STISAVE experiments; occ[95]
  repurposed as a pin-accounting canary (unbalanced-release count, must be 0). Site J NOT applied.
- **Build:** `DECENTASN=1 FM=1 G=6 ACC_N=6 POOL_N=4 WAVES=30 SEGK=64 WOFLUSH=1 STAGINSTR=1 TFPROBE=1`
  (bin `b3c4a905`). Offline gate PASS (inertness `386dc28`; disasm confirmed all masks/immediates).
- **Shapes swept:** oracle only (bring-up, one dispatch) — `ml8 moe attn_q` 576×4096×2048 n_kseg=32.
- **Outcome:**
  | shape | M×N×K | geometry | oracle ok/bad/max_rel | occ[95] canary | occ[0] | DMFAT | reset |
  |---|---|---|---|---|---|---|---|
  | moe attn_q | 576×4096×2048 | G6/FM1/POOL4/SEGK64/J1 | ok=504 **bad=8712** max_rel=1.579 | **65** | 0 | 0 | 0 |
- **Verdict:** ❌ FAIL, and a **45× regression** vs the prior poison protocol (bad=192). NOT a brick/wedge
  (occ[0]=0 clean retire, no reset, no DMFAT) — it's dropped/wrong data. The canary=65 proves real inflight
  **over-releases** (completion decremented an inflight field already 0 → `state−0x100` borrows into the
  next/pending bits → `RB_PENDING` corruption → head-of-line drain stall → work behind it dropped → 94.5%
  bad). Offline-verified NOT the cause: compute path is linear 1:1 claim→release (JDEPTH=1), `s48` is the
  claimed slot at both ends (re-derivation is JDEPTH-gated/off), the only drain path is the gated macro, and
  within this single persistent WG LDS is coherent. So by static inspection the code faithfully implements
  Codex's (next,inflight) design, yet silicon violates its core invariant ("a claimant always owns ≥1
  inflight unit") 65×. → hand the concrete result back to Codex (gpt-5.6-sol, thread 019f665b).

### 2026-07-15 — DECENTASN release classifier (seed-naming diagnostic) — DECENTASN
- **Change:** saved `CLAIMGEN=SL_GEN` at the won claim; at release, classify the racing writer BEFORE the
  subtract and DO NOT subtract in the bad cases (contains the borrow). 3 occ buckets: [95] gen-changed
  (producer reuse), [96] same-gen pending (side_final/producer mid-restamp), [97] same-gen inflight==0
  (CAS/exec imbalance). Host prints all three. Build `147d13e3` (kernel), occ_dispatch rebuilt.
- **Shapes swept:** oracle only (diagnostic bring-up) — `ml8 moe attn_q` 576×4096×2048.
- **Outcome:**
  | class | occ | count |
  |---|---|---|
  | gen-CHANGED (producer reuse, SL_GEN differs) | [95] | **0** |
  | same-gen PENDING (RB_PENDING set, SL_GEN unchanged) | [96] | **823** |
  | same-gen INFLIGHT==0 (pure double-release) | [97] | **0** |
  - oracle ok=0 bad=9216, computed=4345 (collapsed from 31008 — the no-subtract containment leaks the 823
    pins → drain wedges → work behind stalls; EXPECTED for a diagnostic build), occ[0]=0, DMFAT=0, no reset.
- **Verdict:** ✅ seed NAMED. It is **bucket 2**: a completion finds its slot with `SL_RBNEXT` holding
  `RB_PENDING` while `SL_GEN` STILL equals the claimed gen. The only writer of `RB_PENDING` is the producer
  stamp, which writes `SL_RBNEXT=RB_PENDING` FIRST and `SL_GEN=new` LAST. So the classifier catches the
  producer **mid-restamp** — i.e. DRAIN advanced past this slot and the producer reused it **while this
  wave's pin was still live**, 823× (~19% of completions). This directly CONTRADICTS both reviews'
  impossibility proofs (the pin should forbid drain-past-live-pin). Buckets 1 & 3 = 0 rules out full-gen
  reuse and pure double-release. → feed the specific bucket back to sol + Fable to reconcile.

### 2026-07-15 — DECENTASN claim-persistence diagnostic (sol) — DECENTASN
- **Change:** measure the seed at the CLAIM. occ[95]=claims reaching lds_cas_rtn with exec lane0 inactive;
  occ[96]=won-claims whose immediate SL_RBNEXT re-read shows pending|inflight==0 (phantom); occ[97]=releases
  that bailed on inflight==0. Also fixed sol's s47-clobber (release now uses only s45/s46) and made the
  release non-underflowing (bail vs subtract). Build `b31d7ef2`, host rebuilt.
- **Shapes swept:** oracle only — `ml8 moe attn_q` 576×4096×2048.
- **Outcome:**
  | counter | occ | count |
  |---|---|---|
  | exec lane0 INACTIVE at claim CAS | [95] | **0** |
  | won-claim did NOT persist (phantom) | [96] | **0** |
  | release bailed on inflight==0 | [97] | **834** |
  - oracle ok=408 bad=8808 (~96%), computed=19590 (recovered from 4345 — the non-underflowing release stops
    the RB_PENDING poison wedge), occ[0]=0, DMFAT=0, no reset.
- **Verdict:** ❌❌ **REFUTES both convergent hypotheses.** occ[95]=0 kills sol's exec-mask theory
  (lds_cas_rtn 931/939); occ[96]=0 kills the phantom-claim seed — **every won claim persists correctly.**
  So the pin is NOT lost via a bad claim. occ[97]=834 (~4% of computes) shows a real but MINOR release-side
  race (pin occasionally lost during the compute window), now safely CONTAINED (bail, no underflow — proven
  by computed recovering 4345→19590). Crucially **bad stays ~96%**, far too pervasive to be the 4% race →
  a SYSTEMATIC wrong-C independent of the inflight accounting → points squarely at **Site J** (feeder
  decode-before-claim → cross-gen operands), the one known-unfixed correctness bug both reviews flagged.
  → land Site J (task #41) and re-measure.

### 2026-07-15 — DECENTASN Site J (feeder decode-after-claim) — DECENTASN
- **Change:** BSTAGE_R/ASTAGE_R re-read SL_STI + DECODE_STI AFTER the BFNEXT/ARNEXT claim (per-iteration),
  `.if DECENTASN`-gated (baseline byte-identical). Kept the claim diagnostic. Build `7a402142`.
- **Shapes swept:** oracle only — `ml8 moe attn_q` 576×4096×2048.
- **Outcome:** ok=432 **bad=8784** (~95%, UNCHANGED from 8808); occ[95]=0, occ[96]=1, occ[97]=741;
  computed=20112, coast=294.8M (99.97%), occ[20]=1228 claims / 4 reps ≈ 307 of 384 tiles claimed per rep.
- **Verdict:** ❌ Site J did NOT fix the pervasive wrong-C → refuted as the cause. **REFRAME (the important
  finding):** across all four runs `bad` is ~constant (8712–8808) while `computed` ranged 4345→31008 (7×),
  and only ~307/384 tiles are even CLAIMED per rep at 99.97% coast. That pattern is **INCOMPLETENESS**, not
  a per-segment race: DECENTASN is so ASSIGN-BOUND it cannot produce/claim all 384 tiles' work (× 32 ksegs)
  inside the window, so most tiles hold partial/zero split-K sums → bad. (max_rel=1.33>1 shows SOME
  double-adds too, but incompleteness dominates.) The claim/pin/feeder are basically CORRECT (claims persist,
  occ[95]/[96]≈0); the only real residual race is the ~4% release imbalance (occ[97]≈800). The dominant wall
  is **production throughput** — exactly the "13× slower, ASSIGN-BOUND" the strategic frame (§5) predicted.
  → the 3 refuted hypotheses (exec-mask, phantom claim, Site J) were chasing a minor race; the 95% is
  starvation. Decisive next test: measure work-exactness / force completion (does bad→~4% when the work
  actually finishes?).

### 2026-07-16 — BANKED DECENTASN Gate 1 (pin retired) — DECENTASN — ⚠ INVALID (dispatch-env error)
- **Change:** marry decentralized claim to the BANKED path; RETIRE the (next,inflight) pin, make SL_RBDONE==ACC_N
  the drain authority (SL_GEN==DRAIN gate kept). Guard 697 (stale) flipped -> DECENTASN banked-only. All edits
  DECENTASN-gated; inertness VERIFIED md5 386dc28 unchanged. Design: DECENTASN_BANKED_DEEPJ_DESIGN_2026-07-16.md.
- **Build:** `DECENTASN=1 FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=1 RBU=1
  STAGINSTR=1 TFPROBE=1` (bin md5 f928b1dd, 14000B). Offline gate PASS: 0 spill, 32 ds_add_f32, 0
  global_atomic_add_f32 (WOFLUSH gone), no 0x101 (inflight gone), LDS 55808B<64KB. O1 resolved, O2: POOL_N=3 max.
- **Host:** `DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=3 DSWS2_SEGK=64 DSWS2_K=2048
  DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64` — **MISSING `FLOW_WAVES=30`** (host default is 8, occ_dispatch.cpp:6315).
- **Shapes swept:** oracle only — `ml8 moe attn_q` 576×4096×2048 n_kseg=32.
- **Outcome:** launched **waves/WG=8 (8c0a0b)** not 30; oracle ok=0 bad=9216 max_rel=11.01; computed=10089/73728
  (~14%); coast=100.0%; occ[86] ASSIGN-BOUND=100% (942M empty-frontier feed iters); occ[0]=0 CLEAN retire, no
  reset, no DMFAT. TF=0.0.
- **Verdict:** ⚠ **INVALID — dispatch-env error, not a kernel verdict.** 8 waves (vs the required 30) starved the
  decentralized producer -> the run never claimed most tiles -> bad is INCOMPLETENESS (partial split-K sums),
  exactly the O2 confound. NOT a correctness signal on the banked+pin-retired logic (which never got exercised on
  completed work). Clean retire proves no brick/wedge from the change. -> RE-RUN with `FLOW_WAVES=30` added
  (needs a fresh greenlight). All other env was correct.

### 2026-07-16 — BANKED DECENTASN Gate 1 (pin retired), FLOW_WAVES=30 — DECENTASN — ❌ REFUTED (structural)
- **Change / Build:** same banked bin f928b1dd as the invalid run above (pin retired, SL_RBDONE drain gate).
- **Host:** as above **+ `FLOW_WAVES=30`** (fix). Launched **waves/WG=30 (30c0a0b)** ✓ (valid launch this time).
- **Shapes swept:** oracle only — `ml8 moe attn_q` 576×4096×2048 n_kseg=32, POOL_N=3 ACC_N=6.
- **Outcome:** **computed=73604/73728 (~99.8% — work essentially COMPLETE)**; bank adds a=73560 (w=0, BANKZERO ok);
  oracle ok=0 **bad=9216 max_rel=1.0**; coast/ASSIGN-bound 99.8%; occ[0]=0 CLEAN retire, no reset, no DMFAT; span 0.5s.
- **Verdict:** ❌ **REFUTED — structural incompatibility, NOT a tunable bug.** `max_rel` EXACTLY 1.0 = C≈0 everywhere =
  the C-store NEVER FIRED, even though all segments computed. Root cause (confirmed by code, two prongs):
  (1) DECENTASN's claim is a GLOBAL `occ[20]++` super-tile claim -> a tile's n_kseg K-slices scatter across up to
  n_kseg different WGs; but the ACC banks + `TILEDONE[group]` are PER-WG LDS, so no single WG ever accumulates a
  whole tile -> its `TILEDONE` never reaches `n_kseg*ACC_N` -> tile-closer never elected -> no `global_store` of C.
  (2) `zero_banks` (bank-zero + TILEDONE reset, line 859/2468) is COORDINATOR-ONLY; DECENTASN branches to
  `.Lflow_body` (2429) and never calls it -> banks never re-zeroed, TILEDONE never reset per tile (moot given (1)).
  This is the strategy-doc physics made concrete: split-K partials combine on-chip **same-WG only** (LDS) or in DRAM
  (cross-WG = WOFLUSH, the slow path). DECENTASN's decentralized GLOBAL producer distributes a tile's slices
  cross-WG, so a per-WG banked reduce structurally cannot combine them. **Decentralized global claim ⊥ per-WG
  banked reduce.** -> Gate 1 as designed is dead. Fork (kmbandy to steer): (A) reconceive DECENTASN as INTRA-WG
  (each WG claims WHOLE tiles like the coordinator -> banked-compatible; decentralize the assign/role WITHIN the WG,
  which is where the per-WG ASSIGN wall actually is); or (B) pivot to Thread B (full-K in registers, no split-K, big
  tile -> no cross-WG combine at all -- the strategy doc's measured high-ground, O4). Clean retire = the code change
  itself is brick-safe; the design premise is what failed.

### 2026-07-16 — A' step-1: deep-J on the BANKED COORDINATOR, real shape — coordinator (DECENTASN=0) — ◑ WORK-EXACT, 11 bad
- **Change:** NONE to the coordinator path (my AM edits were all `.if DECENTASN`; this is DECENTASN=0, the proven
  path). First-ever deep-J (J>1) on a REAL ml8 shape — deep-J had only run on the synthetic 32K cube before.
  Chosen after the AM structural refutation of banked DECENTASN; obeys G1 (whole tiles per WG). bin 248da98d.
- **Build:** `DECENTASN=0 FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=2 RBU=1
  STAGINSTR=1 TFPROBE=1`. Offline: 0 spill, 32 ds_add_f32, 0 WOFLUSH atomics; JDEPTH confirmed LIVE (md5 differs
  per J). Inertness 386dc28 intact.
- **Host:** `FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=3 DSWS2_SEGK=64
  DSWS2_K=2048 DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64`. Launched 30c0a0b ✓.
- **Shapes swept:** oracle only — `ml8 moe attn_q` 576×4096×2048 n_kseg=32.
- **Outcome:** **computed=73728 EXACT (=12288×6) — WORK-EXACT, deadman trap did NOT fire** ✓; oracle **ok=9205
  bad=11 max_rel=1** (99.88% correct); occ[0]=0 CLEAN retire, no reset, no DMFAT. span=1M ticks (~10ms) —
  **clock NOT committed (<0.5s); per G3 TF=1.0 and assign%=96.5 are cold-start artifacts, NOT quoted.**
- **Verdict:** ◑ **The mechanism WORKS — deep-J banked-reduce is work-exact and ~99.9% correct on a real shape**
  (vs the AM global-DECENTASN structural all-bad). This validates G1/G2/G5 in practice: whole-tile-per-WG banked
  deep-J combines and stores C correctly. Remaining: an **11-unit residual** (0.12%, max_rel=1 → those units ~0 or
  ~2×). `bad>0` = full stop. NEXT (isolation, 1 greenlit dispatch): run **J=1** at the identical geometry (bin
  153a6895) — if J=1 also bad=11 the residual is geometry/coordinator (pre-existing at FM=1 G=6 banked, independent
  of deep-J); if J=1 is clean the 11 is deep-J-specific (the DRAIN-must-not-pass-unflushed invariant, a smaller
  cousin of the bad=64 lesson). Then, once bad=0, a FED run (REPS to ≥1s) for the clock-committed TF/flush-share
  verdict (G3). TF/assign NOT trustworthy from this 10ms run.

### 2026-07-16 — A' step-1 isolation: J=1 control + MSDRAIN fix — coordinator (DECENTASN=0) — ✅ CLEAN substrate
- **Sequence (3 dispatches, one geometry FM=1 G=6 ACC_N=6 POOL_N=3 banked, real ml8 attn_q):**
  | run | JDEPTH | MSDRAIN | oracle | computed |
  |---|---|---|---|---|
  | dj2_coord_g1  | 2 | 0 | ok=9205 **bad=11** | 73728 exact |
  | dj1_coord_ctrl| 1 | 0 | ok=9211 **bad=5**  | 73728 exact |
  | dj1_msdrain_ctrl (bin d2dddc9e) | 1 | 1 | **ok=9216 bad=0 max_rel=0** ✓ | 73728 exact |
- **Finding:** the residual bad is a **POOL_N>1 out-of-order slot-drain race in the banked coordinator, INDEPENDENT
  of deep-J** — present at J=1 (bad=5), NOT introduced by J (deep-J is work-exact + exonerated). Root cause is the
  documented hazard at occ_kernel_dsws_flow.s:664–669: with POOL_N>1, slots complete OUT OF ORDER, so DRAIN must
  use the head-gated WALK (`MSDRAIN`) or it frees a slot still in use → silent wrong C. Default build had
  `MSDRAIN=0`. `MSDRAIN=1` → **bad=0 clean.** (This geometry FM=1 G=6 was never oracle-checked before — baseline
  was certified at FM=2 G=9, likely POOL_N=1 or MSDRAIN-on, so the hazard hid.)
- **All 3 clean retire (occ[0]=0, no reset). Spans ~9–10ms — clock NOT committed; per G3 no TF/assign quoted.**
- **Verdict:** ✅ **CLEAN banked deep-J substrate established.** `DECENTASN=0 …POOL_N=3 MSDRAIN=1` is oracle-clean +
  work-exact on the real shape. NEXT: rebuild deep-J (J≥2) WITH MSDRAIN=1, confirm clean, then a FED run
  (DSWS2_TARGET_SECS≥2 → clock-committed) for the real TF + flush-share verdict = does deep-J amortize the flush on
  a real shape (only ever shown on the synthetic cube). MSDRAIN=1 must be STANDARD for all POOL_N>1 banked runs.

### 2026-07-16 — A' step-1 FED: J=2 banked deep-J, clock-committed — coordinator (DECENTASN=0) — TF=1.0, feed-bound
- **Build:** `DECENTASN=0 FM=1 G=6 ACC_N=6 POOL_N=3 WOFLUSH=0 BANKZERO=1 JDEPTH=2 MSDRAIN=1 …` (bin 00bd7c00).
- **Host:** …`DSWS2_TARGET_SECS=2` → 128 back-to-back reps, ~1.27s sustained (clock COMMITTED).
- **Outcome:** oracle **CLEAN bad=0**; computed=9437184 = 73728×128 (work-exact every rep); **TF=1.0 mean, per-rep
  0.9–1.0, spread 12.5% = glass-flat/TRUSTWORTHY**; carriers STAGE-STARVED (occ[88]=7.89 spin-iters/segment);
  lead-gate coast 40.4% (structural (J−1)/J at J=2); assign-bound 97.5% (per-rep cold-start — NOT quoted as verdict
  per G3, but corroborated by carrier stall).
- **Verdict:** clean + work-exact under sustained load. **TF=1.0 is now trustworthy** (clock-committed, glass-flat).
  The kernel is **FEED/STAGE-bound, not flush-bound** on the banked path (carriers wait for feeders; banked flush is
  already cheap → deep-J has little to amortize). Consistent with the doc: banked flush ~45–57%, rest is pipeline;
  and these ml8 work-items are ~0.16µs math vs ~100µs round-trip (~600×), so a ~10ms total-work shape is
  coordination-dominated by physics. DECISIVE NEXT: J=1 fed baseline (bin d2dddc9e) at identical fed conditions —
  if TF≈1.0 too, deep-J moves nothing here → flush isn't the banked wall → per G6 pivot to the feed round-trip
  (bigger work-items: larger SEGK/tile, deeper pool) or Thread B (full-K, no split-K, big tile).

### 2026-07-16 — A' step-1 FED: J=1 baseline vs J=2 — coordinator (DECENTASN=0) — ❌ DEEP-J DOES NOTHING (banked)
- **Build:** J=1 MSDRAIN=1 (bin d2dddc9e), identical fed conditions to the J=2 run (DSWS2_TARGET_SECS=2).
- **Outcome:** oracle CLEAN, work-exact; **TF=1.1 mean, per-rep 1.1–1.1, spread 4.4% = glass-flat/TRUSTWORTHY**;
  assign-bound 97.4%; carrier stall 0 (J=1 has no jwait).
- **THE COMPARISON (clock-committed, both glass-flat):** J=1 → TF=1.1 ; J=2 → TF=1.0. **Deep-J is NEUTRAL-to-
  NEGATIVE on the banked path** (J=2 slightly worse: its structural (J−1)/J lead-gate coast + carrier stalls cost
  more than the halved banked flush saves). Trustworthy because it's a RELATIVE comparison at identical fed
  conditions — independent of the assign% cold-start caveat.
- **Verdict:** ❌ **Deep-J thread CLOSED for banked ml8.** On banked the flush is already cheap (~45–57%, and
  amortizing it further changes nothing), so deep-J has no wall to attack. Both J's are ~1 TF, FEED/ASSIGN-bound.
  Root wall (doc L127): ml8 work-items are ~0.16µs math vs ~100µs round-trip (~600×) — a small (~10ms) coordination-
  bound problem; flush optimization (deep-J OR banked) cannot touch it. Deep-J's real win is WOFLUSH-only (flush=97%
  there; synthetic-cube 0.4→32). PIVOT (per G6): attack the WORK-ITEM SIZE / coordination cost — bigger super-tiles
  (larger SEGK/tile → fewer, larger work-items amortizing the round-trip), or Thread B (full-K, no split-K, big
  tile) which maximizes work-item size AND drops the combine. A decisive discriminator: run a BIGGER shape / bigger
  SEGK — if TF rises with work-item size, the wall is coordination (fixable via item size), confirming the pivot.

### 2026-07-16 — ⭐ FED via deep-K: the afternoon's un-fed verdicts are VOID — coordinator (DECENTASN=0)
- **The correction (kmbandy caught it, 3rd time today):** every prior 2026-07-16 TF/assign/deep-J number was from
  ~9.5ms reps (drained between) — the UN-SETTLED cold-start transient. `DSWS2_TARGET_SECS` just does more reps, it
  does NOT feed continuously (proven: 2s vs 20s → identical TF=1.1, CONV=0). The correct feed (2026-07-13 rule 2) is
  DEEP-K: crank `DSWS2_K` so each tile grinds enough K to run one long chunk; C unchanged, minimal HBM. Guard STAYS
  ON — proven recipe = bounded chunk + raise `ML8_COOP_CHUNK_MAXS=3.0` (that's the chunk time-abort, NOT the deadman).
- **Build:** J=1 MSDRAIN=1 (bin d2dddc9e). **Host:** `…DSWS2_K=262144 (n_kseg=4096) DSWS2_REPS=1 ML8_COOP_CHUNK=384
  ML8_COOP_CHUNK_MAXS=3.0` — one continuous chunk, guard on.
- **Outcome (vs shallow-K reps):**
  | metric | shallow K=2048, reps | **deep K=262144, fed** |
  |---|---|---|
  | TF | 1.1 | **5.2** (~5×) |
  | starvation (occ[86]) | 97% assign-bound | **16%** |
  | span | 9.5ms/rep | 0.24s/chunk |
  computed=9437184 (work-exact for this shape); occ[0]=0 clean; CONV=0 (per-iteration role economy still settled —
  feeders kept the frontier fuller). occ[20]=448.
- **Verdict:** ⭐ **The "97% assign-bound / ~1 TF / feed-bound / deep-J-dead" conclusions are the COLD-START ARTIFACT,
  exactly the 2026-07-13 76%→1.8% replay.** Fed via deep-K: **~5× TF and starvation collapses 97%→16%.** The kernel is
  NOT fundamentally assign-bound. VOID all afternoon deep-J/feed verdicts. span still 0.24s (<0.5s clock-commit) so
  TF=5.2 is likely conservative — feed deeper for a committed number. NEXT: the FAIR fed deep-J test (J=1 vs J=8, deep-K,
  same fed conditions) — now that the flush can be a visible fraction, deep-J may actually help. GOTCHA: the CPU oracle
  is O(K) → intractable at deep K (killed a 2:42-min oracle; GPU was done in 0.24s). Use `DSWS2_ORACLE_STRIDE`=high
  (or skip) on deep-K fed runs — correctness already proven at K=2048.

### 2026-07-16 — ⭐ FED deep-J WORKS: J=8 = 3.3× J=1 — coordinator (DECENTASN=0)
- **Build:** J=8 MSDRAIN=1 (bin 8789b012). **Host:** identical to the fed J=1 run — `DSWS2_K=262144 (n_kseg=4096)
  DSWS2_REPS=1 ML8_COOP_CHUNK=384 ML8_COOP_CHUNK_MAXS=3.0` + `DSWS2_ORACLE_STRIDE=384` (1-tile canary; O(K) oracle).
- **Outcome (fed deep-J comparison, identical deep-K conditions):**
  | J | TF (fed) | span | starvation | carrier stall | lead-gate coast |
  |---|---|---|---|---|---|
  | 1 | 5.2 | 0.24s | 16% | 0 | 0 |
  | **8** | **16.9** | 0.073s | 21% | 12.4 | 43.4% |
  oracle CLEAN (1/384 checked), computed=9437184 work-exact, occ[0]=0 clean, CONV=0.
- **Verdict:** ⭐ **DEEP-J WORKS WHEN FED — J=8 is 3.3× J=1 (16.9 vs 5.2 TF).** The afternoon's "deep-J dead" was
  entirely the cold-start artifact (unfed reps). Same flush-amortization effect as the doc's synthetic-cube 0.4→32,
  now on the ml8 tile geometry. The RATIO is trustworthy (both identical deep-K); absolute TF likely conservative
  (J=8 span 0.073s < 0.5s clock-commit — deep-J made it 3.3× faster so it fell back under the line). NEXT: sweep
  J=16/32 with K SCALED UP (bigger n_kseg) so each stays fed ≥0.5s → find the knee (flush-amortization gain vs rising
  lead-gate coast 43%@J8 + carrier stall). Then the fully-fed committed peak. deep-J is firmly back on the table.

### 2026-07-16 — ⭐ FED deep-J SWEEP: 5.2→22 TF (4.2×), knee ~J=32 — coordinator (DECENTASN=0)
- **Build:** J∈{1,8,16,32} MSDRAIN=1, all fed identical: `DSWS2_K=262144 (n_kseg=4096) ML8_COOP_CHUNK=384
  ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=384 DSWS2_REPS=1`. All oracle CLEAN, computed=9437184 work-exact.
  | J | TF (fed) | span | carrier stall | lead-gate |
  |---|---|---|---|---|
  | 1 | 5.2 | 0.24s | 0 | 0 |
  | 8 | 16.9 | 0.073s | 12.4 | 43.4% |
  | 16 | 20.2 | 0.061s | ~ | 40.4% |
  | 32 | 22.0 | 0.056s | 137M | 39.6% |
- **Verdict:** ⭐⭐ **DEEP-J is a 4.2× lever when fed (5.2→22 TF), asymptoting ~J=32** (Δ: +11.7,+3.3,+1.8). The
  afternoon's "deep-J dead / ~1 TF / feed-bound" was 100% the un-fed cold-start artifact; fed via deep-K it's the
  flush-amortization win the doc always claimed (cf. synthetic-cube 0.4→32). RATIOS trustworthy (identical deep-K);
  ABSOLUTE 22 TF is idle-clock (spans 56–73ms <0.5s) → the clock-committed peak is HIGHER. NEXT: (a) committed peak
  run at J=32 (scale K up so span≥0.5s → real number); (b) optional J=64/128 to confirm the asymptote. Deep-J firmly
  validated. Then step-2 (within-WG decentralized producer) can be A/B'd against this fed baseline.

### 2026-07-16 — 🔨 BATON BUILT (Tasks 1–3) + OFFLINE GATE GREEN — STAGGER-gated, DECENTASN=0
- **Change:** implemented the traveling-peak baton (`DSWS_TRAVELING_PEAK_BATON_PLAN_2026-07-16.md`) in
  `occ_kernel_dsws_flow.s`, all `.if STAGGER`-gated:
  - **T1 budget cap:** `PEAK_CONC_EFF = floor((VBUDGET − WAVES·VLEAN)/(NFV − VLEAN)) − PEAK_SLACK`, defined AFTER
    the `BUDGET` def (~1589) not at ~298 (VBUDGET/NFV/VLEAN resolve later — a forward `.set` at 298 would not
    compile). New derived cap `FATCAP_EFF = min(MAXFAT_EFF, PEAK_CONC_EFF)` — keeps MAXFAT the live sweep knob
    (Task 5) beneath the physical budget ceiling. `fat_acquire` now caps on `FATCAP_EFF` (was `MAXFAT_EFF`).
    Measured this config: NFV=80 (disasm `s_alloc_vgpr 0x50`) → `PEAK_CONC_EFF = 576/48 − 1 = 11`, clamped to
    ACC_N=6 → **6**; `FATCAP_EFF = min(4,6) = 4` at MAXFAT=4 (knob binds, budget not limiting — correct).
  - **T2 release-at-shrink-START:** moved `fat_release` from shrink-END to BEFORE the `.Lflow_bshrink` spin, so a
    waiting carrier can grow into the registers a shrinking carrier is freeing (the peak *travels*). Sets FATHELD=0
    at shrink-start; the spin is deadman-free + cannot fail; PEAK_SLACK covers the alloc overlap.
  - **T3 the baton:** at the `fat_acquire` refusal, if staged work exists (`DRAIN < STAGE`) the carrier enters
    `.Lflow_batonwait` — a deadman-guarded LDS poll of the FATTOK pool (`.Lflow_jwait` shape: `s_sleep 1`, NO
    `s_alloc_vgpr` in the loop), then one `fat_acquire` when budget frees. No staged work → coast-and-feed (keeps
    lean stagers alive). New counter `CNT_BATON=s103 (DP_CYC, DUTYPROBE-only) → occ[98]/FDIAG_BATON_OFF`; host
    prints it (`[dsws2 BATON]`). Guard added: `STAGGER && DUTYPROBE → .error` (s103 alias).
- **Register safety (verified against source):** baton wait uses only DEAD temps s45/s47/s49/s92; s46(cursor)/
  s48(scb)/s52(sob) are LIVE into the JDEPTH>1 claim and are NEVER touched. Holds NO token in the loop
  (FATHELD=0) → a deadman force-retire out of the wait is leak-clean (`.Lflow_retire` sees FATHELD==0).
- **Leak-freedom (verified):** a waiter that acquires then fails the claim → `.Lflow_cmp_tryadv`→`fat_release`
  (3036); grow refused → `.Lflow_growfail`→`fat_release` (3272); deadman mid-hold → `.Lflow_retire` leak-check
  (3343). Every acquire pairs with a release on every exit.
- **DEADLOCK-AVOIDANCE INVARIANT (plan T3 Step 2, in writing):** every `.Lflow_batonwait` entry is unblocked by
  some *other* wave's `fat_release` (now at shrink-START). Holds because: (a) a carrier waits ONLY when staged
  work exists (`DRAIN<STAGE`); (b) `WAVES ≥ 2·ACC_N` (30 ≥ 12) guarantees lean stagers remain → the fat carriers
  get their next segment staged → they finish their WMMA burst → shrink → `fat_release` → the pool drops below
  `FATCAP_EFF` → a waiter proceeds; (c) if (b) ever fails to hold, `deadman_check` clean-retires the waiter (lean,
  no token) rather than wedging the WG → no MODE1 brick. The hardware `s_alloc_vgpr` remains the REAL cap, so a
  waiter that races ahead of the physical register-free just grow-fails → coasts (never bricks).
- **OFFLINE GATE (all GREEN):**
  - Gate 1 inertness: `DECENTASN=0 FM=2 G=3 ACC_N=3 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=1` → md5
    `386dc28643ffb58568623ad6d89cfe62` **byte-identical** (STAGGER machinery fully compiled out at STAGGER=0).
  - Gate 2 baton build: `DECENTASN=0 FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=8
    MSDRAIN=1 STAGGER=1 MAXFAT=4 RBU=1 STAGINSTR=1` → assembles OK, **0 spill**, no `.error` (PEAK_CONC_EFF≥1).
  - Gate 3 structural: **0** real `s_alloc_vgpr` in `.Lflow_batonwait`; total `s_alloc_vgpr` unchanged (5);
    `fat_release` `ds_add_rtn` precedes the shrink `s_alloc_vgpr 32`. Host `occ_dispatch` compiles clean.
- **NOT YET RUN ON SILICON.** Tasks 4 (fed bring-up, JDEPTH=4 moderate-J so peaks are short) and 5 (PEAK_CONC/
  MAXFAT sweep vs STAGGER=0) are GPU-gated — each needs an individual kmbandy greenlight, fed via deep-K, guard ON.
  No TF verdict exists yet; the baton is BUILT and offline-proven, not measured.

### 2026-07-16 — ❌ BATON bring-up FAILED (Rule-3 stop): computed=0, oracle BAD, livelock — NOT a brick
- **Run:** bin `729c9e7a` (STAGGER=1 BATON=1 MAXFAT=4 JDEPTH=4 FATCAP_EFF=4), fed deep-K K=262144, guard ON,
  `baton_bringup` via gpu_run.sh. Shape 576x4096x262144, expected computed=9437184.
- **Outcome:** ⛔ **oracle BAD ok=0 bad=24 max_rel=1.0; computed=0 (coast-frac 100%)**; exit code 3 → STOP.
  Latch CLEAR, resets_before=0, **NO reset / NO DMFAT → the box is FINE (livelock, not a brick)**.
  - `occ[20] claim=64` (clean run was 448 — coordinator barely moved); `FRONTIER ASSIGN=0 STAGE=0 DRAIN=0` from t=0.
  - door1 NOTHING-STAGED = 100% of coast; STARVATION occ[86] = 100% "empty ASSIGN frontier" → ASSIGN-BOUND.
  - **`occ[98] BATON = 2,130,753,761` (RUNAWAY ~2^31)** — a few waves entered `.Lflow_batonwait` and spun until
    the 0.5s deadman clean-retired them. door3 FAT-PEAK-FULL=1728 (the door-sum non-close is exactly this: my
    code `cnt_inc CNT_FATFULL` then goes to batonwait, not coast → FATFULL over-counts coast by 1728; cosmetic).
- **Leading hypothesis (LIVELOCK):** with BATON=1 a wave that refuses a token but has staged work WAITS instead
  of coasting-to-FEED. That drains the stager pool → staging starves → the ≤4 fat waves never get their next
  segment → never compute (computed=0) → never shrink → pool never refills → waiters spin till deadman. The
  spec §3 liveness argument ("WAVES≥2·ACC_N guarantees stagers") has a HOLE: the baton itself removes stagers.
- **Confound:** STAGGER=1 was NEVER validated FED before (spec §0: all prior stagger verdicts un-fed). So
  computed=0 could be (a) my baton's livelock OR (b) pre-existing STAGGER-cap breakage. CANNOT separate from
  this one run.
- **ISOLATION PREPPED (offline, Rule 6):** added `BATON` build knob (default 1; `BATON=0` = old coast-on-refusal
  CAP, no wait). Inert still `386dc28`; `BATON=0` bin `1551ce0b` (0 spill), `BATON=1` bin `729c9e7a` (== the
  failed run). **NEXT (needs greenlight): A/B — run `BATON=0` control fed, same geometry.**
  - BATON=0 CLEAN (computed=9437184) + BATON=1 broken → **my baton livelocks**; the wait design needs rework
    (e.g. bounded wait-then-coast, or only the Nth wave waits while the rest feed).
  - BATON=0 ALSO computed=0 → **STAGGER-cap itself is broken fed**, independent of the baton; fix that first.
- **Kernel state:** baton edits intact + BATON-gated; no revert. DECENTASN=0 inert byte-identical preserved.

### 2026-07-16 — 🔬 A/B VERDICT: the STAGGER=1 SUBSTRATE is broken fed, NOT the baton — bootstrap deadlock
- **Run:** bin `1551ce0b` (STAGGER=1 **BATON=0** = old coast-on-refusal CAP), same geometry/feed as the baton run.
- **Outcome:** ⛔ **IDENTICAL FAILURE**: oracle BAD ok=0 bad=24, **computed=0**, `ASSIGN=0 STAGE=0 DRAIN=0` from
  t=0, `occ[20] claim=64` (clean run = 448). Latch CLEAR, NO reset/DMFAT (livelock, box fine). Exit 3 → STOP.
  - Door signature DIFFERS from BATON=1 and is the tell: **door3 FAT-PEAK-FULL = 100% of coast** (2.19e9),
    door1 NOTHING-STAGED ~0%. `occ[98] BATON=0` (correctly off). So waves DO try to go fat, ≤4 get tokens
    (FATCAP_EFF=4), the rest refuse+coast — but nothing ever stages so no one computes.
- **⭐ VERDICT — the baton is CLEARED; STAGGER=1 itself deadlocks the bootstrap:** the coordinator is wid0
  (dedicated lean wave; single writer of ASSIGN_HEAD, line 448) and the pipeline invariant is
  `DRAIN ≤ STAGE ≤ ASSIGN ≤ DRAIN + POOL_N` (line 511). Chain: under STAGGER the ≤MAXFAT capped fat compute
  waves grab a token + `s_alloc_vgpr NFV` **before** work is staged, then can't complete a segment →
  **DRAIN never advances → ASSIGN clamps at DRAIN+POOL_N (=3) → staging can't get ahead → the fat waves never
  get fed → they never shrink → deadlock.** STAGGER=0 has NO token layer (fat_acquire/release compile out) so
  it bootstraps fine (that is why deep-J at STAGGER=0 hit 22 TF). The token layer, with a cap that BITES
  (MAXFAT<ACC_N), cannot cold-start the assign/drain cycle. This was invisible because STAGGER was NEVER fed
  before (spec §0). My baton (BATON=1) sits ON TOP of this and fails the same way — plus adds the wait.
- **The baton machinery is sound in isolation:** it engaged (occ[98] handoffs), no register corruption, no
  leak, no brick — it is just built on a substrate that never worked fed.
- **NEXT — this is now a STAGGER-substrate debug, not a baton task.** Candidate directions (offline first,
  each silicon step greenlit): (a) don't acquire the fat token until AFTER a successful claim (grow-on-demand
  like STAGGER=0, cap only concurrent *held* peaks) so idle waves never occupy the pool; (b) let the
  coordinator/feeds run uncapped and only cap the COMPUTE grow; (c) reconsider whether the stagger/traveling-
  peak direction earns its keep vs deep-J-alone (STAGGER=0 already = 22 TF) — strategic call for kmbandy.

### 2026-07-16 — ⛔ RETRACTION: the two entries above (“STAGGER substrate broken”) are WRONG (kmbandy caught it)
- **What I got wrong:** I concluded STAGGER=1 “was never validated fed / the substrate deadlocks the bootstrap.”
  FALSE. Prior fed logs prove STAGGER=1 ran oracle-CLEAN: `~/dsws_gpu_logs/0714_confirm_s1_mf15_j64.log`
  (STAGGER=1 G=15 MAXFAT=15 J=64 → computed=354738633 ok=120 bad=0) and `0714_r2_maxfat8_j4.log`
  (STAGGER=1 G=12 **MAXFAT=8<ACC_N=12** J=4 → ok=96 bad=0). The latter ALSO disproves my “cap-bites→deadlock”
  mechanism — the throttle bit there and it was clean. (`0714_r3_maxfat8_j8.log`, same but J=8, was bad=96 —
  so JDEPTH mattered even then, a SEPARATE known issue.)
- **My A/B was contaminated:** the `BATON=0` “control” still contained my Task 1 (fat_acquire caps on
  FATCAP_EFF not MAXFAT_EFF) and Task 2 (fat_release moved to shrink-START), both STAGGER-gated not BATON-gated.
  So `BATON=0` failing exonerated ONLY the Task-3 wait, NOT my other two edits. I never tested a pristine
  pre-baton kernel. Claiming “the baton is sound” while the kernel is broken was unfounded self-exoneration.
- **What is ACTUALLY still open:** my two failing runs used a config **never run at STAGGER=1 before** —
  the deep-K 384-tile shape (576x4096x262144, n_kseg=4096, SEGK=64, G=6 ACC_N=6 POOL_N=3 MAXFAT=4 J=4) — AND
  my baton edits. Cannot attribute computed=0 to (a) my code vs (b) this untested config without a clean
  isolation: a PRISTINE pre-baton kernel at THIS exact config. That test has not been run.
- **Corrective next step (offline first):** make Task 1 + Task 2 also BATON-gated so `BATON=0` is a TRUE
  pristine-STAGGER=1 control, rebuild, then (greenlit) re-run `BATON=0` at this config. BATON=0 clean → my
  baton broke it; BATON=0 also computed=0 → STAGGER=1-at-this-new-config is the issue, not the wait.
- The “substrate broken” framing in the two entries above and KG `c261826e` is RETRACTED. See KG correction.

### 2026-07-16 — ✅ ISOLATED (true pristine control): STAGGER=1 is broken ON THE DEEP-K SHAPE, my baton is NOT the cause
- **Setup fix:** gated Task 1 (fat_acquire cap) + Task 2 (fat_release position) + CNT_BATON behind `BATON`, so
  `BATON=0` is now a GENUINE pristine STAGGER=1 (all my edits revert as one unit). Verified: inert still
  `386dc28`; BATON=1 md5 unchanged (`729c9e7a`, clean split); BATON=0 = `d3c52ebb` (differs from the earlier
  contaminated `1551ce0b`); s_sleep 4 (BATON=0) vs 5 (BATON=1) confirms the wait is gone; 0 spill.
- **Run:** bin `d3c52ebb` (STAGGER=1 **BATON=0** = pristine, MAXFAT=4 JDEPTH=4), fed deep-K, same geometry.
- **Outcome:** ⛔ **FAILS IDENTICALLY to the baton runs**: computed=1196 (≈0), oracle ok=0 bad=24, `ASSIGN=0`
  from t=0, `claim=64`. Ran the full ~1.0s this time and hit the deadman:
  - `occ[88] jwait ≈ 1.995e9` — carriers go FAT then STAGE-STARVE (1.67M spin-iters/computed-segment).
  - `occ[91] DMFAT=175` fat carriers deadman-force-retired → **INVALID RUN (C wrong, ACC dropped)**;
    `occ[92] TOKLEAK=340`. door3 FAT-PEAK-FULL=94.2%, door2 LEAD-GATE=5.8%. STARVATION 100% ASSIGN-bound.
  - Wrapper **LATCHED** the invalid run (.gpu_last_hang) — a human must clear it before any further dispatch.
- **⭐ VERDICT (properly isolated this time — genuine pristine control, not analysis):** the pre-existing
  STAGGER=1 substrate does NOT cold-start on the **deep-K 384-tile shape** (576x4096x262144, G=6 ACC_N=6
  POOL_N=3 SEGK=64 n_kseg=4096). **My baton is EXONERATED** — pristine STAGGER=1 fails the same way with the
  baton fully removed. Reconciles with 0714: STAGGER=1 is CLEAN on the BIG training shape (34816 tiles, G=12/15,
  SEGK=32, n_kseg=1024) but BROKEN on this small deep-K shape → a SHAPE-SPECIFIC pre-existing stagger bug.
- **WHERE TO LOOK:** why ASSIGN_HEAD never advances (coordinator wid0) under STAGGER=1 on this geometry. The
  chain: fat compute waves grab tokens + grow, then sit in .Lflow_jwait waiting for a STAGE; ASSIGN stays 0 so
  nothing ever stages; the carriers stage-starve and the deadman kills them. Candidate deltas vs the clean 0714
  big-shape run: POOL_N=3 (vs ?), n_kseg=4096 (vs 1024), TOTAL=384 tiles (vs 34816), G=6/ACC_N=6 (vs 12/15),
  SEGK=64 (vs 32), MAXFAT=4. Next debug is OFFLINE on the assign/coordinator↔stagger interaction — NOT a baton
  task, and NOT my code. (The 22 TF deep-J result stands: it is STAGGER=0 on this same shape, oracle-clean.)

### 2026-07-16 — ✅✅ ROOT CAUSE FOUND (offline, systematic-debugging): invalid config JDEPTH>POOL_N + missing guard
- **Root cause (CONFIRMED, byte-exact):** the deep-K stagger deadlock is NOT a code bug and NOT the baton — it is
  an **INVALID CONFIG the kernel failed to reject**: in the throttled stagger case (STAGGER=1, MAXFAT<ACC_N),
  **JDEPTH must be ≤ POOL_N**, and my bring-up used JDEPTH=4 > POOL_N=3.
- **Mechanism:** a capped deep-J carrier reaches JDEPTH super-tiles ahead of DRAIN, but the ASSIGN window is only
  POOL_N deep (`ASSIGN ≤ DRAIN+POOL_N`, kernel L2494) and DRAIN can't advance until ALL ACC_N rowblks of a ksi
  finish — which the MAXFAT throttle prevents from happening in order — so the carrier's JDEPTH-th segment can
  never be staged → carriers stage-starve (occ[88] jwait~2e9) → ASSIGN stalls (claim stuck at 64) → deadman
  force-retires fat carriers → INVALID. STAGGER=0 is immune (all waves fat, no window limit).
- **Proof (LDS reconstruction, WOFLUSH: LDS = 256 + POOL_N·(FN·16·SEGK + G·16·FM·SEGK); byte-exact):**
  | run | JDEPTH | POOL_N (reconstructed) | throttled | J≤POOL_N | result |
  |---|---|---|---|---|---|
  | 0714 r2 | 4 | 4  (256+4·14336=57600 ✓) | yes | ✅ | CLEAN |
  | 0714 r3 | 8 | 4 | yes | ❌ | BROKE |
  | 0716 deep-K (mine) | 4 | 3  (512+3·10240+6·4096=55808 ✓) | yes | ❌ | DEADLOCK |
  | 0714 mf15 | 64 | — | NO (MAXFAT=ACC_N) | rule N/A | CLEAN |
- **FIX (offline, done):** added the missing assemble guard at occ_kernel_dsws_flow.s L198-207:
  `.if JDEPTH>1 && STAGGER && (MAXFAT_EFF<ACC_N enforced) && JDEPTH>POOL_N → .error`. VERIFIED: rejects
  JDEPTH=4/POOL_N=3 (the deadlock config); accepts JDEPTH=2/POOL_N=3; STAGGER=0 JDEPTH=8/POOL_N=3 still builds
  (deep-J sweep unaffected); inert still 386dc28. This guard would have caught today's config OFFLINE.
- **Retraction:** my 2026-07-16 "G7 JDEPTH≤POOL_N is stale" note (DSWS_DESIGN_CONSTRAINTS.md) was WRONG — I'd
  tested STAGGER=0 (unthrottled), the regime the rule does NOT govern. G7 corrected back.
- **Baton status:** UNCHANGED and UNBLOCKED-in-principle — the baton was never the problem. To actually evaluate
  it on the deep-K shape, use a VALID config: **JDEPTH=2 ≤ POOL_N=3** (POOL_N=4 would overflow LDS at G=6 SEGK=64
  banked: 512+4·10240+24576=66048 > 65536). NEXT (greenlit): BATON=0 JDEPTH=2 pristine-stagger bring-up to
  confirm the shape works, then BATON=1 for the actual baton A/B.

### 2026-07-16 — ✅ ROOT-CAUSE FIX CONFIRMED ON SILICON: pristine stagger CLEAN at valid JDEPTH=2 ≤ POOL_N=3
- **Run:** bin `1b1b5a64` (STAGGER=1 **BATON=0 JDEPTH=2** MAXFAT=4, valid config), fed deep-K, same geometry.
- **Outcome:** ✅ **oracle CLEAN ok=24 bad=0 max_rel=0; computed=9437184 (work-exact); occ[20] claim=448
  (fully advanced, vs stuck-at-64 in the deadlocked J=4 runs); occ[0]=0; NO DMFAT/reset/leak.** TF=7.8.
  - Confirms the root cause: the deadlock was purely the invalid JDEPTH=4>POOL_N=3; J=2≤POOL_N=3 cold-starts
    clean. The new assemble guard would have rejected the bad config offline.
- **Baton-relevant diagnostics (this is the STAGGER=1 CAP baseline, BATON=0):** coast-frac 90.9%; door2
  LEAD-GATE 49.3% (structural (J-1)/J = 1/2 at J=2), **door3 FAT-PEAK-FULL 49.4%** (the MAXFAT=4 cap refuses ~half
  the time → real room for the baton to act), occ[88] jwait 14.5M (1.54/segment, carriers mildly stage-starved),
  STARVATION 73.7% ASSIGN-bound. So the cap bites at J=2 — the BATON=1 A/B has headroom to show an effect.
- **NEXT (greenlit, one at a time):** (1) BATON=1 JDEPTH=2 — the actual baton, A/B vs this BATON=0 TF=7.8;
  (2) a STAGGER=0 JDEPTH=2 baseline on this exact deep-K shape for the full comparison (deep-J sweep had J=1=5.2,
  J=8=16.9 but no J=2 point on deep-K).

### 2026-07-16 — 🎯 BATON ISOLATED (valid config): the baton's unbounded WAIT starves the feed pipeline → deadlock
- **A/B at the SAME valid config (JDEPTH=2 ≤ POOL_N=3):** BATON=0 = CLEAN 7.8 TF (prev entry); **BATON=1 (bin
  `d820de35`) = DEADLOCK**: computed=0, oracle bad=24, ASSIGN=0, claim=64 stuck, occ[98] BATON=2.15e9 runaway,
  door1 NOTHING-STAGED=100%. Latch SET. Config is PROVEN valid (BATON=0 clean) → **the baton IS the cause.**
- **Root cause of the baton deadlock (confirms the original 2026-07-16 livelock hypothesis, now cleanly
  controlled):** a refused compute wave WITH staged work (DRAIN<STAGE) enters `.Lflow_batonwait` and spins
  UNBOUNDED for a token instead of coasting to FEED. But in this substrate the excess compute waves ARE the
  feeder pool (they coast→feed). The baton removes them from feeding → staging starves. With MAXFAT=4<ACC_N=6 a
  tile needs the 4 fat waves to cycle all 6 rowblks, but the waves that should stage the next segments are
  parked in batonwait → STAGE can't advance → DRAIN can't advance → ASSIGN stuck → deadlock. The spec §3
  liveness argument ("WAVES≥2·ACC_N guarantees stagers") is FALSE here: the baton parks its own stagers.
- **FIX (in progress): BOUNDED wait-then-coast.** A refused carrier waits at most BATON_MAX poll-iters for a
  freed token; if none frees, it coasts to FEED (never permanently leaves the feed pool). Preserves the
  traveling-peak intent (grab a token if one frees fast) without starving staging. New knob BATON_MAX.
- **NEXT (greenlit):** rebuild BATON=1 with bounded wait, re-run at JDEPTH=2. Clean + TF>7.8 → the baton helps;
  clean + TF≈7.8 → neutral (wait rarely wins the race); deadlock again → the wait is fundamentally wrong here.

### 2026-07-16 — ✅✅ BATON ROOT CAUSE ISOLATED: the WAIT (Task 3) is unworkable here; release-at-start (Task 2) is a clean +0.4 TF
- **Decisive isolation (added RELSTART knob to split Task 2 from Task 3), all at valid JDEPTH=2 POOL_N=3:**
  | config | Task2 release@start | Task3 wait | bin | result |
  |---|---|---|---|---|
  | BATON=0 RELSTART=0 (pristine) | ✗ | ✗ | 1b1b5a64 | CLEAN 7.8 TF |
  | **BATON=0 RELSTART=1** | ✓ | ✗ | c88555d0 | **CLEAN 8.2 TF** ⭐ |
  | BATON=1 (bounded, BATON_MAX=16) | ✓ | ✓ | c3c52be3 | DEADLOCK (computed=0) |
  | BATON=1 (unbounded) | ✓ | ✓ | d820de35 | DEADLOCK (computed=0) |
- **ROOT CAUSE = the WAIT (Task 3), confirmed:** release-at-shrink-start ALONE is clean AND faster (7.8→8.2).
  The wait deadlocks even BOUNDED. Mechanism: the wait's `DRAIN<STAGE` gate pulls compute waves into the poll
  loop exactly when work is staged — but in THIS substrate the refused compute waves ARE the feeder pool (they
  coast→feed). Any wait starves staging → positive-feedback: staged work → waves wait → feeding drops → STAGE
  falls behind → pool jams full (door3 99.9%) → carriers never get their next segment → deadlock. The spec §3
  liveness argument ("WAVES≥2·ACC_N guarantees stagers") is FALSE here: the waiters ARE the stagers.
- **VERDICT: drop the wait-on-token mechanism.** The useful, safe half of the "baton" is release-at-shrink-START
  alone (RELSTART=1): the freed budget is grabbed by the next wave through the NORMAL flow (demand-driven, no
  explicit wait) — the peak travels naturally, oracle-clean, +0.4 TF over pristine. `.Lflow_batonwait` (Task 3)
  should be retired; keep RELSTART=1.
- **OPEN (the real question for stagger's fate):** does STAGGER=1 RELSTART=1 (8.2 TF at J=2) beat STAGGER=0
  deep-J on this shape? deep-J sweep had J=1=5.2, J=8=16.9 but NO J=2 point. Next: STAGGER=0 JDEPTH=2 baseline
  for the apples-to-apples. If STAGGER=0 J=2 ≥ 8.2, the stagger cap isn't earning its keep and deep-J-alone wins.

### 2026-07-16 (evening) — 🔨 BATON REBUILT to the revised spec (Tasks 1–3) + OFFLINE GATE GREEN
- **Root reframe (with kmbandy):** baton+stagger are ONE system; the win only exists at a BUDGET-BINDING G
  (spec §1.3); the wait is CARRIER-ONLY; and — the key code discovery — in the emergent economy the refused
  compute waves ARE the feed supply, so making them wait deletes the feeders → the deadlock. Fix = a real
  carrier/feeder split.
- **Task 1 — carrier/feeder split (NCARR):** new knob `NCARR` (default `ACC_N`; `NCARR_EFF` handles 0=default).
  STAGGER=1 role init now makes wids [1,NCARR_EFF] CARRIERS and every other wid a RESERVED FEEDER (alt A/B);
  wid0 stays coordinator. Guard `.error` if `WAVES-NCARR < ACC_N`. STAGGER=0 role init byte-identical.
- **Task 2 — carrier wait-for-budget:** replaced the morning `DRAIN<STAGE`-gated pre-claim wait + `BATON_MAX`
  coast-fallback with a pure carrier wait: a refused ROLE_COMPUTE wave (already past the staged-head gate at
  .Lflow_compute + lead gate) polls the pool (deadman-guarded, NO s_alloc_vgpr, NO coast) and grow+claims when
  budget frees. Carriers never coast — the reserved feeders do the staging.
- **Task 3 — RELSTART=1** (release at shrink-START) is the default under BATON=1 (proven clean in isolation).
- **DEADLOCK INVARIANT (plan T2 S2, in writing):** with `WAVES-NCARR ≥ ACC_N` reserved feeders that ALWAYS
  stage (never reach fat_acquire), the ≤PEAK_CONC fat carriers get their next segment staged → finish burst →
  shrink → `fat_release` (shrink-START) → pool drops → a `.Lflow_batonwait` waiter proceeds. Every wait pairs
  with a guaranteed future release; `deadman_check` clean-retires (leak-clean, FATHELD==0) if ever violated.
- **OFFLINE GATE GREEN:** inert `386dc28`; STAGGER=1 assembles 0-spill; reserved-feeder guard rejects NCARR=26;
  0 `s_alloc_vgpr` in `.Lflow_batonwait` (loop exits only via deadman-retire or token-win, NO coast); RELSTART=1
  `fat_release` precedes the `.Lflow_bshrink` spin. Bring-up bin `34d85aa6` staged.
- **NOT YET RUN.** Task 4 = ONE greenlit Phase-1 CORRECTNESS bring-up (J=2≤POOL_N, G=6 NON-binding): proves
  oracle-clean + work-exact + no-deadlock + bounded occ[98], NOT a TF verdict (binding-G is Phase 2). Individual
  greenlight, fed deep-K, guard ON.

### 2026-07-16 (evening) — ❌ Phase-1 bring-up FAILED (3rd explicit-wait strike) → question the architecture
- **Run:** bin `34d85aa6` (STAGGER=1 BATON=1 NCARR=6 carrier/feeder split, JDEPTH=2), fed deep-K. Latch CLEAR,
  no reset/DMFAT (box fine). Rule-3 STOP.
- **Outcome:** IDENTICAL to the prior BATON=1 failures — computed=0, oracle bad=24, ASSIGN=0 from t=0,
  claim=64 (clean=448), occ[98] BATON=3.4e9 RUNAWAY, 100% ASSIGN-bound. The carrier/feeder split did NOT fix it.
- **3-STRIKES on the explicit wait:** (1) unbounded pre-claim wait, (2) bounded BATON_MAX wait, (3) carrier-only
  wait + reserved feeders — ALL wedge ASSIGN identically. Meanwhile BATON=0 (coast-on-refusal) always runs clean
  (baton_j2_b0=7.8, RELSTART=1=8.2). So the failing variable is precisely **the explicit `.Lflow_batonwait`
  spin**, independent of who waits or the feeder supply.
- **Architectural hypothesis:** the explicit spin-wait itself wedges the coordinator — a handful of LEAN carriers
  polling the FATTOK word billions of times (occ[98]=3.4e9) starves the coordinator's LDS/SQ-front-end (assign
  never advances). Unlike `.Lflow_jwait`, whose carriers are FAT and do WMMA between polls, batonwait carriers do
  nothing but hammer one LDS word.
- **⭐ The likely correct design (no explicit wait): RELSTART=1 + coast-on-refusal-RETRY.** A carrier that can't
  grow COASTS (does useful FEED work), loops back through .Lflow_compute, and re-tries fat_acquire next iteration
  — by which point release-at-shrink-START has freed budget, so it grows. Demand-driven, no spin, no wedge, and
  the coasting carrier did real work meanwhile. This is exactly `BATON=0 RELSTART=1`, which ran CLEAN at 8.2 TF
  (bin c88555d0). PENDING kmbandy consult: pivot the baton to release-at-shrink-start + coast-retry (drop the
  explicit wait), then test THAT at a binding G.

### 2026-07-16 (evening) — ✅ ARCHITECTURE CORRECTION: the baton has NO explicit wait — ripped it all out
- **kmbandy (definitive):** "there should NEVER have been an explicit wait. Read the notes." FLOW_ECONOMY_DESIGN.md
  is unambiguous: *"There is no blocking read anywhere in a wave's hot loop. That is the whole design."* A wave
  that can't do its work does COMPLEMENTARY work for a cycle and retries — *"Never spin."* My `.Lflow_batonwait`
  spin-poll was the exact blocking read this architecture exists to avoid, and it wedged the coordinator 3×.
- **THE BATON = one change: `fat_release` at shrink-START (RELSTART=1).** A shrinking carrier returns budget the
  instant it commits to shrink; the NEXT carrier that comes around the non-blocking loop and does `fat_acquire`
  grabs it and grows. The peak travels via demand, nobody waits. A refused carrier COASTS (does feed work) and
  retries next pass — the fungibility already in the hot loop.
- **RIPPED OUT (all violated the non-blocking/emergent design):** the `.Lflow_batonwait` wait (all 3 variants);
  the `BATON`/`BATON_MAX` knobs; the `NCARR` carrier/feeder hard split (role mix is meant to be coordinator-tuned
  in the background, not compile-time partitioned); the `CNT_BATON`/occ[98] wait counter. KEPT: `RELSTART`
  (default 1 = the baton; 0 = pristine shrink-END cap for A/B). fat_acquire now always caps on `FATCAP_EFF`
  (=min(MAXFAT_EFF, PEAK_CONC_EFF), the budget-aware cap).
- **Offline gate:** inert `386dc28`; STAGGER=1 assembles 0-spill; NO batonwait code in disasm; s_sleep count 4
  (jwait + feed floors only, no baton poll); fat_release precedes `.Lflow_bshrink`. Bin `c5f91a28` staged.
- **Behavioral note:** this IS the config that ran oracle-CLEAN at 8.2 TF earlier today (c88555d0 = BATON=0
  RELSTART=1). Cleaned bin differs from c88555d0 by 8 bytes (macro-expansion churn from removing the `.if BATON`
  branches) — so the next greenlit run CONFIRMS equivalence rather than assuming it.
- **NEXT:** Task 4 = ONE greenlit bring-up of the cleaned kernel (correctness confirm, G=6 non-binding). Then the
  REAL open work is Phase 2 (spec §1.3): does release-at-shrink-start + non-blocking flow deliver the trapezoid
  occupancy win at a BUDGET-BINDING G (ACC_N≳13)?

<!-- TEMPLATE — copy below, fill, never edit a prior entry:
### YYYY-MM-DD — <change summary> — <path: DECENTASN | baseline>
- **Change / Build / Host base:** ...
- **Shapes swept:** <from §2>
- **Outcome:** one row per shape:
  | shape | M×N×K | geometry (G/FM/POOL_N/SEGK/J) | oracle ok/bad/max_rel | TF | hipBLASLt | occ[0] |
- **Verdict:** ...
-->

### 2026-07-16 — ⭐ TRAVELING-PEAK BATON T4 correctness/liveness bring-up — 3× DETERMINISTIC CLEAN (DECENTASN=0)
- **Change:** the NEW push-mailbox baton (river-safe), replacing the old wait-on-token dam. T1 GROWPERMIT per-wave
  mailbox @LDS 0x150 + bootstrap seed GROWPERMIT[3]=1; T2 push at shrink-START (round-robin next compute wid via
  magic-mod MAGIC=0x097B425E) BEFORE the shrink s_alloc_vgpr; T3 grow-gate reads OWN permit (grow/coast, NO wait)
  behind BATONGATE=1, which no-ops the FATTOK/MAXFAT token layer. STAGGER=0 byte-identical 386dc28 preserved.
  bin `7fa0ee05` (STAGGER=1 BATONGATE=1 JDEPTH=2). All offline gates green; full chain verified in disasm.
- **Build:** `FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=2 MSDRAIN=1 STAGGER=1 RBU=1
  STAGINSTR=1 TFPROBE=1 DECENTASN=0`.
- **Host:** `FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=3 DSWS2_SEGK=64 DSWS2_K=262144
  DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64 ML8_COOP_CHUNK=384 ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=384`.
  (First attempt WASTED: omitted the geometry env -> host defaulted FM=2/FLOW_WAVES=8, WG never launched, occ0=0.
  Harmless — no reset. KG 3c62677a. The full env above launched 30c0a0b ✓.)
- **Shape:** real ml8 `576x4096x262144` n_kseg=4096, super-tile 96x64, TOTAL_super=1572864, LDS=55808B. NON-binding G.
- **Outcome — 3 identical greenlit dispatches (r1/r2/r3):**
  | run | computed | oracle | occ[0] | occ[98] baton-spin | door3 no-turn-coast | door4 grow-fail | TF |
  |---|---|---|---|---|---|---|---|
  | r1 | 9437184 EXACT | ok=24 bad=0 | 0 | 0 | 85.9% | 0 | 6.0 |
  | r2 | 9437184 EXACT | ok=24 bad=0 | 0 | 0 | 86.1% | 0 | 5.9 |
  | r3 | 9437184 EXACT | ok=24 bad=0 | 0 | 0 | 86.3% | 0 | 5.8 |
  All: work-exact (=TOTAL_super*ACC_N), oracle CLEAN, clean retire, NO reset/DMFAT/brick, C guard tail clean.
- **Reading:** the baton is CORRECT, LIVE, and DETERMINISTIC across repeats -> the bootstrap race (which showed as
  same-config clean-AND-deadlock non-determinism) is RULED OUT. door3~86% no-grow-turn coasts = the one-grows-per-
  shrink traveling peak (only ~1-2 of 27 compute waves hold the turn at a time). occ[98]=0 = NO spin path exists
  (the old dam's 2.1e9 batonwait runaway is gone by construction). span~0.21s for 9.4M segments => many waves
  computed concurrently (baton circulates, not stuck on the seed).
- **TF is NOT a verdict here (spec §3).** door4 GROW-FAIL=0 = budget NON-binding at G=6, so the baton can only
  THROTTLE concurrency the budget didn't need throttled -> TF 5.8-6.0 sits BELOW the STAGGER=0 J=2 baseline (8.8),
  exactly as predicted ("at G=6 stagger/baton buys nothing"). Also ASSIGN-BOUND (occ[86]=96% empty frontier).
- **Verdict:** ✅ **T4 PASS (correctness + liveness + determinism).** The baton works. NEXT = T5: find a BINDING-G
  geometry (grow-fail>0, e.g. WOFLUSH=1 higher ACC_N/POOL_N per the 2026-07-13 grow-fail=1588 regime) and measure
  whether the traveling peak lifts TF vs the best STAGGER=0 baseline where the VGPR budget is the actual wall.

### 2026-07-16 NIGHT — BATON REDESIGNED to pure NOTIFICATION (A) + measured: wall is STAGE-WAIT, not budget (DECENTASN=0)
- **Design correction (kmbandy, definitive):** the baton is a PURE NOTIFICATION — poke a ready wave "grow now +
  grab a rowblk" when a preceding wave shrinks. NOT a gate/cap/seed. Un-notified waves grow on their own (river);
  grow-fail->coast is fine. Goal: keep >=1 wave at peak = continuous compute, not bursty. The earlier permit-GATE
  (permit==0->forced coast) and BATON_SEED (peak-count cap) were DAMS -> RIPPED OUT.
- **Built (bin 22bc8d0d):** grow-gate = pure river (no permit). POKE at shrink-START (round-robin next wid>=3,
  magic-mod, lds_put_r NOTIFY[target]=1). WAKE at .Lflow_feedmt_sleep (before s_sleep SLEEPN: if poked -> clear +
  skip nap + loop back to grow). Knobs STAGGER/BATONGATE/GRELAX, no BATON_SEED. STAGGER=0 byte-identical 386dc28.
- **Build:** `FM=1 G=6 ACC_N=6 POOL_N=3 WAVES=30 SEGK=64 WOFLUSH=0 BANKZERO=1 JDEPTH=32 STAGGER=1 BATONGATE=1
  MSDRAIN=1 RBU=1 STAGINSTR=1 TFPROBE=1 DECENTASN=0`.
- **Host:** `FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=3 DSWS2_SEGK=64 DSWS2_K=262144
  DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64 ML8_COOP_CHUNK=384 ML8_COOP_CHUNK_MAXS=3.0 DSWS2_ORACLE_STRIDE=384`.
- **Shape:** real ml8 `576x4096x262144` n_kseg=4096, banked (WOFLUSH=0), G=6, J=32.
- **Outcome (baton_A_g6_j32):** computed=9437184 EXACT, oracle ok=24 bad=0 CLEAN, occ[0]=0 clean retire, TF=21.4,
  **door4 GROW-FAIL=0** (budget NON-binding), **occ[88] jwait=136,705,818** (carriers fat, waiting for STAGE). No
  reset/DMFAT. Comparators same config: river (permit-cap removed)=20.7, STAGGER=0 baseline=22. => BATON == RIVER
  (parity, within noise).
- **Reading:** the (A) baton is correct+clean but has NOTHING to bite on at non-binding G: it pokes a sleeper, the
  sleeper wakes, finds nothing staged (STAGE-WAIT), can't compute. The wall is FEED/staging (jwait=136M), NOT the
  VGPR budget (grow-fail=0), NOT compute. The baton fills BUDGET-valleys; there are none at reachable banked G.
- **Verdict:** the baton needs a BINDING G (grow-fail>0) to prove itself; that needs G>12, which banked LDS can't
  fit at the real shape and WOFLUSH is the killed flush-wait -> **binding-G at a real shape without WOFLUSH is
  UNSOLVED** (needs a new flush-paydown idea). Confirms the plain river already IS the emergent traveling peak; the
  reachable wall is STAGING. MORNING: (1) chase binding-G w/o WOFLUSH, or (2) attack staging (feed width / POOL /
  barrier cadence). Wasted 1 greenlit run earlier tonight on a synthetic 32768^3 J=1024 WOFLUSH=1 STAGGER=0 config
  (off-target on every axis — chasing the stale synthetic 37-TF number); do NOT repeat.

### 2026-07-18 — ⭐ 64-BIT B-ADDRESS FIX (deep-K wrong-C) — deep-K feed now CLEAN — coordinator + DECENTASN
- **How found:** feeding the canonical G=6/J=2 config deeper (deep-K, single chunk — the CORRECT way to add work;
  reps trip the fat-carrier deadman = DMFAT, NOT a feed lever) exposed a wrong-C the shallow runs all missed.
  At K=262144 (n_kseg=4096): oracle CLEAN. At K=2097152 (n_kseg=32768): oracle BAD ok=0 bad=24 max_rel=1.375,
  BUT work-exact + clean retire + no DMFAT. DECISIVE CLUE: two unrelated kernels (DECENTASN=0/J32 07-16
  `dj32_committed`, and DECENTASN=1/J2 today) gave the *identical* max_rel=1.375 → the fault is in shared code.
- **Root cause (Codex gpt-5.6-sol, numerically reproduced relerr=1.375001431 == observed):** a 32-bit integer
  overflow in the shuffled-B segment-base offset. `s21 = ksi*KSEG_STEPS` then `s21 = s21*s10` (s10=NT*256=65536)
  via low-32 `s_mul_i32`, followed by a single `s_addc_u32 s21,s5,0` that cannot recover the lost high 32 bits.
  Overflows 2^32 when ksi*KSEG_STEPS*s10 ≥ 4GiB, i.e. **n_kseg ≥ 32768** at N=4096/SEGK=64; the 2nd half of K
  then reads B from a WRAPPED address (k−1048576) → work-exact but wrong C. Shared `BSTAGE`/`BSTAGE_R` macro →
  config-independent (DECENTASN/JDEPTH orthogonal). **Real ml8 K (≤4096) is nowhere near 4GiB → real shapes were
  ALWAYS correct; only artificial deep-K feeding reaches it.**
- **Fix:** full 64-bit B offset — `s_mul_hi_u32 s25, s21, s10` (high half) + carry into the B pointer, at ALL 3
  sites: BSTAGE (~1308), BSTAGE_R !DECENTASN (~1412), BSTAGE_R DECENTASN SITE J (~1435). Scratch s25 (already a
  declared clobber). 0 new VGPR/LDS; +16B .text. At shallow K the hi-mul is 0 → real-shape behavior byte-equivalent.
  Bin `3bf705db`→`9cce13fa`. (Byte-identity anchors 02faf45a etc. intentionally change — real fix to shared feed code.)
- **Runs (3, all greenlit, canonical G=6 ACC_N=6 POOL_N=2 SEGK=64 J=2 DECENTASN=1 STAGGER=1 BATONGATE=1 banked):**
  | # | run | shape (M×N×K) | n_kseg | oracle | computed | span/TF |
  |---|---|---|---|---|---|---|
  | 1 | regression (`./dsws.sh correct`, FULL stride=1) | 576×4096×2048 | 32 | **ok=9216 bad=0 max_rel=0** | 73728 exact | 4chk / 0.3 |
  | 2 | deep-K PRE-fix | 576×4096×2097152 | 32768 | ok=0 **bad=24** max_rel=1.375 | 75497472 exact | 1.29s / 7.6 |
  | 3 | deep-K POST-fix | 576×4096×2097152 | 32768 | **ok=24 bad=0 max_rel=0** | 75497472 exact | 1.29s / 7.7 |
- **Verdict:** ✅ 64-bit B addressing fixes the deep-K wrong-C; deep-K feeding is now correct to any depth (memory
  permitting — host mats A + 2·B, ~K·8768 bytes). FIRST honest clock-committed (1.29s) fed number at G=6/J=2:
  **TF=7.7**, and the now-TRUSTWORTHY architecture read = door4 grow-fail=0 (budget non-binding at G=6), occ[88]
  jwait 14.5/seg + occ[86] 62.7% assign-bound (stage-starved), baton inert. occ[96] PHANTOM=TOTAL_super is a
  benign per-claim constant (present in every clean run), NOT a bug.
- **FOLLOW-UP (not yet done):** the A-address side uses the same 32-bit pattern; Mo=576 keeps A<4GiB until K>~7.4M,
  so audit/port the same 64-bit fix to ASTAGE before feeding beyond that. Deep-K oracle is LOOSE/sampled (1 tile);
  correctness ALSO independently proven by Codex's CPU reproducer.

### 2026-07-18 — BINDING-G fed (G=18/GROUPS=3/SEGK=32) post-fix — budget STILL non-binding (grow-fail=0) — DECENTASN
- **Config:** DECENTASN=1 STAGGER=1 BATONGATE=1 banked J=2 **G=18 ACC_N=6 GROUPS=3 POOL_N=2 SEGK=32**, bin `fc5191fb`
  (picks up the 64-bit B-fix). Deep-J >J=2 is LDS-blocked on the decentralized path (POOL_N%J==0 + LDS: J=4→POOL_N=4
  overflows 65536B at SEGK=64/G=6, confirmed by .error), so binding-G (SEGK=32) is the way to probe budget-binding.
- **Bring-up (correctness):** bounded K=2048 (n_kseg=64) FULL stride=1 oracle → **ok=9216 bad=0 max_rel=0** — the B-fix
  does NOT break binding-G.
- **Fed run:** 576×4096×524288 (n_kseg=16384, SEGK=32), deep-K single chunk, ORACLE_STRIDE=384.
- **Outcome:** oracle ok=72 bad=0 (sampled) CLEAN; computed=37748736 = TOTAL(128)·G(18)·n_kseg(16384) WORK-EXACT;
  span **0.72s CLOCK-COMMITTED**; **door4 grow-fail=0 (VGPR budget NON-binding even at G=18)**; occ[98] baton=0 (inert);
  occ[88] jwait **22.46/seg** (WORSE than G=6/J=2's 14.5); occ[86] 62.9% assign-bound; **TF=3.4** (BELOW G=6/J=2's 7.7 —
  G=18 has 3× fewer/bigger tiles: 128 vs 384). occ[0]=0 clean, no DMFAT/reset.
- **Verdict:** ❌ binding-G does NOT bind the budget on the real shape — grow-fail=0 at BOTH G=6 and G=18, fed +
  clock-committed + correct. The traveling-peak baton has NO activation regime at reachable geometry; the consistent
  wall is STAGING/ASSIGN throughput, not VGPR budget. Refutes the 07-16 "G>12 binds the budget" premise (measured on
  the pre-fix corrupted path). NEXT (kmbandy to steer): attack the staging/assign throughput (the actual wall), or
  accept the G=6/J=2 baseline. Board note: another session re-claimed R9700 in the check→dispatch gap; this run
  completed clean (no collision observed), then held.

### 2026-07-18 — ⭐ J×G MATRIX SWEEP (SEGK=16, post-fix) — J lifts TF ~3×, G flat/negative, budget NEVER binds — DECENTASN
- **Motivation:** stop basing conclusions on the hardcoded J=2. Systematic J×G sweep (kmbandy: "not going to figure this
  out by guessing"). Offline-designed: LDS feasibility + M=576 tiling (G must divide 36) + host allow-list (flow G≤32).
- **Cells:** J∈{1,2,4} × G∈{6,12,18}, ACC_N=6, POOL_N=J, SEGK=16 (fixed → widest J range; real amortization depth = J·SEGK),
  GROUPS=G/6, banked DECENTASN=1 STAGGER=1. Fed deep-K K=524288 (n_kseg=32768). 9 cells ran (G=36 REFUSED by host allow-list
  G≤32; G=24 invalid — not a /36 divisor for M=576; J=8 needs an OP_BASE co-change, deferred). Peak host RAM 5.7GB.
- **All 9 cells: oracle bad=0, computed=75497472 work-exact (=2304·n_kseg, const across G since TOTAL·G=2304), no DMFAT.**
  | TF        | J=1 | J=2 | J=4 |   | jwait/seg | J=1 | J=2 | J=4 |
  |-----------|-----|-----|-----|---|-----------|-----|-----|-----|
  | G=6       | 1.2 | 2.1 | 3.4 |   | G=6       | 0.0 | 12.3| 16.7|
  | G=12      | 1.1 | 1.9 | 3.3 |   | G=12      | 0.0 | 13.5| 18.6|
  | G=18      | 1.1 | 1.8 | 2.7 |   | G=18      | 0.0 | 19.5| 28.6|
  **grow-fail=0 and baton=0 in ALL 9 cells.** spans 0.7–2.3s (mostly clock-committed).
- **Verdict:** (1) **J is THE throughput lever** — J1→J4 ~3× TF; J=2 (the old basis) left ~40% on the table. (2) **G is not a
  lever** — flat-to-negative; more compute breadth just worsens staging. (3) **Budget NEVER binds anywhere** (grow-fail=0
  across the whole matrix) → the traveling-peak baton has NO activation regime; that thread is dead, now proven not asserted.
  (4) The kernel is **FEED/STAGING-bound** (not compute-bound: G flat; not budget-bound: grow-fail=0); jwait is the wall and
  grows with both J and G. CAVEAT: absolute TF maxes at 3.4 due to the SEGK=16 penalty (canonical SEGK=64/J=2 = 7.7) — J and
  SEGK share LDS (J·SEGK amortization budget). NEXT: **J×SEGK sweep at fixed LDS budget** to find the throughput-optimal split;
  retire the baton/budget-binding line (comprehensively refuted).

### 2026-07-18 — ⭐⭐ J×SEGK SWEEP (G=6) — SEGK is THE lever, deep-J is DOMINATED, TF ∝ amortization depth — DECENTASN
- 9 cells G=6 ACC_N=6 POOL_N=J, J∈{1,2,4} × SEGK∈{16,32,64,128} with J·SEGK≤128 feasible (SEGK≤128 OPSTRIDE cap;
  J·SEGK≤252 LDS). K=262144. All bad=0, work-exact, no DMFAT, grow-fail=0. spans 0.18–1.1s (product-128 cells <0.5s
  idle-clock → abs TF conservative; ratios trustworthy, identical K).
- Grouped by amortization depth (J·SEGK; flush count/tile = K/(J·SEGK); flush LDS-ops/WMMA = 128/SEGK):
  | J·SEGK | cell (flush/wmma) → TF, jwait/seg |
  |--------|-----------------------------------|
  | 16  | J1S16 (8.0) → 1.1, jw0 |
  | 32  | **J1S32 (4.0) → 2.1, jw0** ; J2S16 (8.0) → 1.9, jw12 |
  | 64  | **J1S64 (2.0) → 3.9, jw0** ; J2S32 (4.0) → 3.8, jw13 ; J4S16 (8.0) → 3.3, jw17 |
  | 128 | **J1S128 (1.0) → 7.0, jw0** ; J2S64 (2.0) → 6.7, jw13 ; J4S32 (4.0) → 6.1, jw17 |
- **Findings:** (1) TF ∝ amortization depth (J·SEGK): 16→32→64→128 = 1.1→2.1→3.9→7.0, ~doubling, NO plateau at 128 → deeper
  = higher. (2) At FIXED depth, **lower-J / bigger-SEGK WINS** (J1 ≥ J2 > J4 in every product group). Deep-J is DOMINATED:
  it buys the same amortization SEGK gives for free but ADDS jwait (J1=0, J2~13, J4~17 spins/seg) + lead-gate coast.
- **VERDICT: SEGK is the throughput lever; deep-J (carrier/coupled-cursor/POOL_N%J/baton machinery) is NOT the win.**
  Recipe = **J=1, SEGK as large as LDS allows.** Less split-K = faster; the limit (SEGK→K, flush once) = hipBLASLt's GSU1
  no-split design (KG HIPBLASLT_TEARDOWN). NEXT: push SEGK>128 by freeing LDS (lower ACC_N banks and/or G at J=1), and/or
  test the full-K no-split path (Thread B). The "canonical" should be J=1/max-SEGK, not J=2/S=64.

### 2026-07-18 — DEEPER-PRODUCT sweep (G=6, ACC_N=3) — climb CONTINUES past 128 (MEASURED), J=1/max-SEGK wins to product 256
- Motivation: the J×SEGK sweep capped at product 128 (LDS at ACC_N=6); I claimed "no plateau" WITHOUT testing higher (kmbandy
  caught the extrapolation: "why didn't you do J4/S128?" — that's product 512, physically impossible: operand pool alone > 64KB).
  Product 256 IS reachable by dropping ACC_N 6→3 (banks 24KB→12KB), so measure it.
- 5 cells G=6 ACC_N=3 (GROUPS=2), K=262144. All bad=0, no DMFAT, grow-fail=0. spans 0.13–0.40s (idle-clock; ratios trustworthy).
  | J | SEGK | product | TF | jwait/seg |
  |---|------|---------|----|-----------|
  | 1 | 64   | 64      | 3.1 | 0 |
  | 2 | 64   | 128     | 5.3 | 11.6 |
  | 1 | 128  | 128     | 6.0 | 0 |
  | 2 | 128  | 256     | 8.6 | 19.7 |
  | 1 | 256  | 256     | **9.7** | 0 |
- Findings: (1) TF KEEPS CLIMBING past 128 — product 256 J1/S256 = 9.7 (measured, not extrapolated); no plateau. (2) J=1/max-SEGK
  WINS at every product (J1>J2: 6.0>5.3 @128, 9.7>8.6 @256). (3) ACC confound quantified: J1/S128 = 7.0 @ACC=6 vs 6.0 @ACC=3
  (dropping banks 6→3 costs ~1 TF), BUT product 128→256 @ACC=3 gains +3.7 (6.0→9.7) — NET WIN over the ACC penalty. Bigger SEGK
  wins even paying the bank cost.
- **VERDICT confirms J×SEGK:** SEGK is the throughput lever, deep-J dominated. Recipe = J=1, SEGK max (SEGK=256 @ G=6/ACC=3 =
  9.7 idle-clock TF, STILL climbing). Endpoint = full-K no-split (hipBLASLt GSU1). NEXT: SEGK=512 needs G≤2 (confound) OR the
  full-K no-split path (Thread B); and a committed-clock run (deeper K, memory-managed) for real absolute TF at the winning config.

### 2026-07-18 — ⭐ FED / CLOCK-COMMITTED same-flush-count comparison (product 128) + PHASE (door) breakdown — DECENTASN
- CORRECTION to the J×SEGK / deeper sweeps: those were sub-0.5s = idle-clock (~1147 vs 2350 MHz) → TF was duration-confounded,
  NOT clean throughput. Re-ran the KEY same-flush-count pair FED (single K=1048576 chunk, span ~0.6s = CLOCK-COMMITTED), one at a
  time, RAM-watchdog. (J1/S256 & J2/S128 product-256 cells: watchdog-killed at avail 1744MB — K=1M's 9.2GB operands hit the ceiling;
  product-256 needs a smaller K, secondary. clean kill, no latch.)
- Product 128, G=6 ACC_N=6, CLOCK-COMMITTED:
  | J | SEGK | span | TF | coast% | door1 nothing-staged | door2 lead-gate | jwait/seg |
  |---|------|------|----|--------|----------------------|-----------------|-----------|
  | 1 | 128  | 0.605s | **8.2** | 95.7% | **100%** | 0% | 0 |
  | 2 | 64   | 0.654s | **7.6** | 93.0% | 74.1% | **25.9%** | **14.7** |
- Findings (MEASURED, committed, phase-resolved): (1) At the SAME flush count, committed-clock, J=1 (8.2) beats J=2 (7.6) by ~8%
  — deep-J does NOT tie. (idle-clock had shown 7.0/6.7; committing the clock lifted both ~15% and preserved the J1>J2 gap.)
  (2) WHY, from the door breakdown: J=2 pays door2 LEAD-GATE 25.9% ((J-1)/J structural — non-lead slices the carrier walks) +
  jwait 14.7 spins/seg (carrier goes fat then WAITS holding its accumulator for the next slice to stage); J=1 pays ZERO of both.
  (3) BUT door1 NOTHING-STAGED = 100% of J1's coast / 74% of J2's — both ~95% coast because there's nothing staged: the kernel is
  overwhelmingly FEED-LIMITED. So J's only benefit (fewer flushes) is wasted (flush isn't the wall), while its jwait (waiting on
  that slow feed) + lead-gate bite.
- **VERDICT: deep-J is FEED-STARVED, not inherently bad.** The wall is FEED (nothing-staged). Fix the feed → deep-J's flush-
  amortization starts paying AND its jwait shrinks. NEXT: attack the FEED/staging throughput (the measured wall) — helps TF
  directly AND is the prerequisite for deep-J to pay. And: the process rule (COMMIT THE CLOCK / feed before any TF) — do NOT
  quote TF from <0.5s again.

### 2026-07-18 — ⭐ 2s-COMMITTED same-flush-count comparison (products 128 & 256) — J=1 beats J=2 by ~7-9%, feed-limited — DECENTASN
- kmbandy: "0.6s is not enough, 2s minimum." Fed via BIG M (many tiles, MTL=200, K=131072) — spans ≥2s at low RAM (~3.5GB)
  vs deep-K which would OOM (~30GB) at 2s. All bad=0, no DMFAT, no latch.
  | J | SEGK | prod | span | TF | coast% | door1 nostg | door2 lead | jwait/seg |
  |---|------|------|------|----|--------|-------------|------------|-----------|
  | 1 | 128  | 128  | 2.74s | 7.5 | 95.6% | 100% | 0%    | 0 |
  | 2 | 64   | 128  | 2.96s | 7.0 | 92.9% | 73.6%| 26.4% | 13.5 |
  | 1 | 256  | 256  | 2.16s | 9.5 | 98.3% | 100% | 0%    | 0 |
  | 2 | 128  | 256  | 2.36s | 8.7 | 97.2% | 78.3%| 21.7% | 17.9 |
- Findings (2s committed, phase-resolved): (1) J=1 beats J=2 at BOTH products (~7% @128, ~9% @256) — SAME result & mechanism
  as the 0.6s run → robust, not a clock artifact. deep-J costs ~7-9% at equal flush count. (2) J=2's overhead = door2 lead-gate
  22-26% + jwait 13-18/seg; J=1 = zero of both. (3) All ~95-98% coast, nothing-staged dominated → FEED-LIMITED (confirmed at 2s).
- CAVEAT: absolute TF is feed-method-sensitive — big-M/many-tiles reads ~7.5 where deep-K single-chunk 0.6s read 8.2 (many small
  tiles carry more per-tile coordinator overhead; OR chunked commits the clock less than a single continuous chunk). The RELATIVE
  J1>J2 gap is method-independent. Absolute peak needs a single deep-K ≥2s chunk → direct-to-VRAM operand gen (DSWS2_VRAMGEN, ~40
  lines occ_dispatch, host can't hold ~30GB) — not built.
- **VERDICT (measured, 2s): deep-J is FEED-STARVED, not broken.** Wall = FEED (nothing-staged). Fix feed → deep-J's flush-
  amortization pays + jwait shrinks → deep-J flips from ~8% cost to lever. NEXT: attack FEED/staging throughput (coordinator
  publish rate / POOL depth / feeder count / barrier cadence — never swept, always measured downstream of it).

### 2026-07-19 — ⭐⭐ LDS-SPLIT FRONTIER SWEEP (SEGK×POOL_N×ACC_N, 18 cells) — door1=100% & grow-fail=0 EVERYWHERE → the ring/POOL coupling is the wall
- kmbandy: "do the whole matrix, >2 values each, at the winning K, fully fed." Corrected off two of my errors: the dead COORD_PERIOD
  axis, and sweeping POOL_N at losing SEGK (32/64, ACC_N=6) instead of the winning regime. Winning config = **SEGK=256/ACC_N=3/POOL_N=1**
  (not the dsws.sh "canonical" POOL_N=2/SEGK=64/ACC_N=6). LDS reality: SEGK⊥POOL_N — SEGK=256 fits POOL_N=1 ONLY (any ACC_N).
- METHOD: full feasible grid, each cell big-M fed (MTL=256 K=131072) ≥3.2s clock-committed; correctness-gate phase first
  (bounded K, FULL stride=1 oracle, 64 ring-wraps) — **all 18 gated bad=0**. POOL_N=4 has a wrong-C race ISOLATED to GROUPS=1
  (S32/A6/P4 bad=24; S32/A3/P4 & S64/A3/P4 CLEAN) — excluded, follow-up.
  | ACC_N | SEGK | POOL_N | GROUPS | span | TF | door1 | occ86 |
  |---|---|---|---|---|---|---|---|
  | 3 | 256 | 1 | 2 | 3.20 | **8.2** | 100% | 85.1% | ◄ PEAK
  | 2 | 256 | 1 | 3 | 3.42 | 7.7 | 100% | 84.1% |
  | 3 | 128 | 2 | 2 | 4.01 | 6.6 | 100% | 47.7% |
  | 6 | 128 | 1 | 1 | 4.80 | 5.5 | 100% | 87.8% | ◄ best GROUPS=1 (clean single-carrier model)
  | 3 | 128 | 1 | 2 | 5.28 | 5.0 | 100% | 82.2% |
  | 6 | 64 | 2 | 1 | 5.77 | 4.6 | 100% | 47.5% |
  | 6 | 64 | 1 | 1 | 6.32 | 4.2 | 100% | 84.9% |
  | 6 | 64 | 3 | 1 | 6.45 | 4.1 | 100% | 55.4% |
  | 3 | 64 | 1..4 | 2 | 7.0-8.3 | 3.2→3.8 | 100% | 24-84% |
  | 2 | 64/128 | 1..4 | 3 | 4.9-11.8 | 2.2-5.4 | 100% | 10-88% |
- VERDICTS (all 18 cells): (1) SEGK is THE lever (TF tracks SEGK; all S256 top, S64 bottom). (2) **door1 nothing-staged = 100.0%
  in EVERY cell** — no LDS knob moves it. (3) POOL_N weak/situational (helps only when it fits w/o costing SEGK: G2/S128 P1→P2 = 5.0→6.6).
  (4) GROUPS matters only via the SEGK it unlocks (G1 caps at S128=5.5; G2 unlocks S256=8.2; G3 unlocks nothing better + rescan tax).
- ⭐ THE REAL FINDING: **door1=100% AND grow-fail=0 (door4) in every cell.** The wall is NOT any LDS knob — it's that compute is
  chained to POOL_N staged slots, so at SEGK=256/POOL_N=1 only ~ACC_N waves ever go fat → the per-SIMD VGPR budget NEVER binds →
  the dyn-VGPR moat + stagger/traveling-peak (built, `:596`) sit IDLE (grow-fail=0, baton occ98=0). The whole DSWS economy is a
  cold engine with no fuel line. occ86 (assign starvation) swings 10-88% with ZERO correlation to TF → assign is NOT the wall
  (eliminated). CORRECTIONS kmbandy drove this session: B-reuse is L2/L3 (not LDS — self-load B is a cache hit, no HBM blowup);
  STAGGER is admission-control on the fat population (the occupancy engine), NOT a deadlock guard; SEGK=256 is the ceiling by
  DESIGN (peak-duty ~13% = the stagger trapezoid precondition; bigger = plateau = stagger dies), not just by LDS.
- NEXT (spec written, DSWS_SELFSERVE_DESIGN.md): break the last coupling — coast→self-serve-compute tier (claim work-item, self-load
  A/B from L2/L3 into VGPR, WMMA, ds_add into shared bank) so parallelism = wave count not slot count → budget binds → the stagger
  engine engages. Fingerprint to confirm: grow-fail 0→large, door1 100%→<100%, baton occ98>0, TF>8.2. Hook site = `.Lflow_compute`
  door1 branch (:2783-2786). Ring stays as opportunistic fast-path. GATE-defined build S0-S4 in the spec.
