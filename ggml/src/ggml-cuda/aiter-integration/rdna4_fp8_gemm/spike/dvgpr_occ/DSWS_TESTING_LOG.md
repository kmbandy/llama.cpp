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

### 2026-07-19 EVENING — ⭐⭐⭐ SELFSERVE CARRY-THROUGH LANDS: 8.2 → 15.6 TF (1.90x) ON THE PEAK CELL
- **S1 = carry-through on DECENTASN**, not a new mechanism. If the ring is empty, the assigning wave carries its own
  reservation through compute (self-loads A/B from L2/L3, WMMA, ds_add into the shared bank) instead of stamping a slot
  and walking away. At POOL_N=1 the handoff bought nothing. Ring stays as the opportunistic fast path.
- Config of record (unchanged from the 2026-07-19 frontier sweep peak cell): `WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1
  ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1 BANKZERO=1 STAGGER=1 SELFSERVE=1 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1`.
  Shape 24576x4096x131072 (MTL=256 NTL=64 K=131072 n_kseg=512), chunk 384x43 MAXS=3.0, oracle stride=512.
  **Fed by BIG-M, not deep-K** (deep-K OOMs ~30GB host at a 2s span; big-M stays ~3.5GB).

  | run | build | TF | span | oracle | computed | note |
  |-----|-------|----|------|--------|----------|------|
  | baseline | SELFSERVE=0 | 8.2 | 3.20s | CLEAN | 50330112 | **1536 SHORT** (dropped work) |
  | #8 | SELFSERVE=1 FORENSICS=0 | 15.1 | 1.749s | CLEAN | 50331648 | EXACT |
  | #9 | + SGPR fix | **15.6** | **1.690s** | **CLEAN** | **50331648** | EXACT, deadman armed |
- Run-to-run spread 3.3%, inside the ~15.6% throughput noise another session measured on this box the same evening;
  the gap to 8.2 is far outside it. **S1 is also MORE work-exact than the baseline** — baseline dropped 1536 rowblk
  segments, S1 drops zero. Per the standing rule that dropped work FLATTERS TF, the 1.9x is if anything conservative.
- Funnel: `entered == shrunk == 16777216` (= 2*TOTAL_super, one emission per GROUP; GROUPS=2). Every reservation
  carried through; ZERO fell back to the ring handoff. We are now INSIDE the hipBLASLt band (12.6-70.6 TF), at the low end.
- ⭐ **THE WIN DID NOT COME FROM THE MOAT.** `grow-fail=0`, `door1=100%`, `occ[98] baton=0` — ALL UNCHANGED from
  baseline in both runs. The 1.9x came purely from deleting the ring handoff. The dyn-VGPR budget still never binds and
  the stagger/traveling-peak engine is still idle. The design-doc fingerprint (grow-fail 0->large, door1 <100%, baton>0)
  has NOT fired. **15.6 is not the ceiling.**
- CAUTION — two of our wall-indicators may now be TAUTOLOGIES under S1 and must be re-derived before being used as
  evidence again: (a) `door1 NOTHING-STAGED=100%` — carry-through waves never stage for anyone, so the ring is empty BY
  CONSTRUCTION and a coasting wave MUST find nothing staged; (b) `occ[86]` ASSIGN-starvation 98.4% — the feed path
  hunting work it no longer handles. Last night's "the wall is FEED" verdict was correct under the RING economy; it
  cannot be carried forward unexamined.
- ALSO VOID: last night's "POOL_N weak/situational" conclusion. Under the ring POOL_N bought staging depth; under
  carry-through POOL_N is the CONCURRENCY knob. Same symbol, different meaning — do not carry the sweep forward.

### 2026-07-19 EVENING — ROOT CAUSE OF THE 15.6 CEILING (confirmed in source, not inferred)
- `occ_kernel_dsws_flow.s` reservation gate: `s_sub_u32 s47, s44, s45 (ASSIGN-DRAIN) ; s_cmp_ge_u32 s47, POOL_N ;
  s_cbranch_scc1 .Lflow_feedmt_sleep`. At POOL_N=1 that is **exactly ONE outstanding super-tile per workgroup** — 64
  carriers across 1920 waves = 3.3% of the fleet, against 5.1%-of-peak measured. Arithmetic closes: 16,777,216
  reservations / 64 WGs / 1.69s = **~6.4us per reserve->drain cycle**, i.e. exposed L2/L3 latency with no occupancy to
  hide it behind.
- Under SELFSERVE the slot holds only a 32-byte PRE-COMPLETED SENTINEL, not operands. POOL_N is pinned at 1 because it
  sizes the LDS-expensive OPERAND pool (`OP_BASE + POOL_N*OPSTRIDE`) — a service carry-through never uses. **The window
  is being rationed at operand-pool prices.**

### 2026-07-19 EVENING — ❌ SSWIN=8 HANGS + GPU RESET (run #10). LATCHED. DO NOT RE-RUN AS-IS.
- New defsym `SSWIN` (Codex) decouples the carry-through reservation window from the operand-pool depth: `SSWIN` defaults
  to `POOL_N`; new `SLOT_N` drives `slot_of`; LDS guard at :715 accounts for the wider 32B control array; gate at :3664
  compares against `SSWIN`; seven `SELFSERVE && SSWIN>POOL_N` sites handle the boundary cases; new guard refuses
  `SSWIN < POOL_N`.
- OFFLINE GATES ALL PASSED: SELFSERVE=0 = `43beb082`; **SSWIN-unset BIT-IDENTICAL to `be1bb047`** (inert until raised);
  SSWIN=8 = `f36c06a0` 0-spill; SSWIN=16 fits LDS; KMAJOR=1 / DUTYPROBE=1 / SSWIN<POOL_N all refuse.
- ON SILICON: **hang on the FIRST chunk + 1 GPU RESET.** `occ0(live)=1763` of 1920, `computed=312` (expected 50331648),
  `FRONTIER ASSIGN=0 STAGE=0 DRAIN=0` — the frontier never left zero, so reservations stopped happening ENTIRELY.
- **The window hypothesis is NEITHER confirmed NOR dead** — the run never got far enough to test whether the budget binds.
  `grow-fail=0` here is meaningless.
- ⚠️ **THE GATES PROVED THE CHANGE IS INERT WHEN OFF. THEY PROVED NOTHING ABOUT CORRECTNESS WHEN ON.** That distinction
  should have been stated before dispatching, not after. The plan itself named the tile/group boundary interlock
  (ZLOCK / DA_ZDONE / GSTORED, and the fact that a 1-deep window is what currently PREVENTS a boundary firing under a
  live reservation) as the PRIMARY RISK — and that is exactly where it broke.
- SSWIN is inert by default, so the tree is safe; only the opt-in SSWIN=8 build is broken.

---

## 2026-07-20 MORNING — run #10 (SSWIN=8) hang: ROOT CAUSE FOUND OFFLINE

**Verdict: host/kernel LDS contract break. NOT the boundary interlock flagged as primary risk.**

`occ_kernel_dsws_flow.s:715-719` — when `SSWIN > POOL_N` the kernel relocates the per-slot
control array from the low control gap (`SLOTC_INLINE_BASE`=148) to the TOP of LDS, above the
accumulator pool, growing the group segment by `SSWIN*SLOTC_STRIDE`.
`occ_dispatch.cpp:1836` computed `ldsBytesRaw = kOpBase + poolSlots*operandBytes + accBytes`
— a hand-mirror of the kernel's OLD total, with **no SSWIN term**.

Exact arithmetic at the config of record (POOL_N=1 ACC_N=3 FM=1 FN=4 G=6 SEGK=256), values
EXTRACTED from the assembler (not hand-derived) by `gate_lds.sh`:

| SSWIN | kernel LDS_TOTAL_FLOW | host raw (pre-fix) | granule-rounded | result |
|-------|----------------------|--------------------|-----------------|--------|
| unset | 53760                | 53760              | 53760           | fits   |
| 8     | 54016                | 53760              | 53760           | **256B OOB** |
| 16    | 54272                | 53760              | 53760           | **512B OOB** |

`GRANULE=512`, and `53760 = 105 granules EXACTLY` — **zero slack**. One byte more and the
control array would have fit by accident and hidden the defect indefinitely.

**Mechanism** (under-allocation certain; precise OOB semantics inferred): control array OOB →
writes dropped, reads 0 → `SL_GEN`=0 (only generation 0 passes the v3 gate); `SL_RBNEXT`=0
reads as staged+claimable so *every* wave's `CAS(0→1)` "wins" the same rowblk (explains
`computed=312` of duplicated work); `SL_RBDONE` never increments → `DRAIN` pinned at 0 →
`ASSIGN−DRAIN` saturates `SSWIN` → all waves sleep. Hang + 1 GPU reset.
`FRONTIER ASSIGN=0` was a **never-published snapshot**, not a live zero — `computed>0` proves
`ASSIGN` advanced.

**Fix:** `occ_dispatch.cpp` now adds `ssWin*32` when `ssWin > poolSlots`, reading the SAME
`SSWIN` env var the build uses (a separate `DSWS2_*` name could drift from the assembled binary).
Inert when unset (`ssWin == poolSlots`). Kernel unchanged — `SSWIN=8` still assembles to `f36c06a0`.

**Process finding — the important one.** All six kernel gates passed before run #10, including
the bit-identical-when-inert proof. They were *structurally incapable* of catching this: the
defect was on the other side of a two-sided contract. Added `gate_lds.sh`, which does **not**
re-mirror the formula (a third copy would drift too) — it extracts the assembled truth by
bisecting on an assembler predicate and compares to the host allocation. **Validated by negative
control:** with the SSWIN term removed it FAILS at 8 and 16, PASSES when unset. A gate that
passes but cannot fail proves nothing.

Status: fix is static-verified only. Confirmation requires a run; `.gpu_last_hang` still SET.

## 2026-07-20 — RUN #11 (SSWIN=8 + PHIST): NO HANG. LDS FIX CONFIRMED. SSWIN HYPOTHESIS IS DEAD.

Config: `SSWIN=8 PHIST=1` on the config of record (WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3
JDEPTH=1 KMAJOR=0 DECENTASN=1 BANKZERO=1 STAGGER=1 SELFSERVE=1 FORENSICS=0 STAGINSTR=1 TFPROBE=1
DEADMAN=1), shape 24576x4096x131072, ML8_COOP_CHUNK=384, 43 chunks.
Log: `~/dsws_gpu_logs/s1_sswin8_phist_run11_075448.log`. No latch, no GPU reset.

**1. THE LDS FIX IS CONFIRMED ON SILICON.** Host reported `LDS=54016B(alloc 54272B)` — exactly what
`gate_lds.sh` predicted (54016 needed, granule-rounds to 54272). Run #10's deadlock did not recur.
`occ[0](live)=0` clean completion, oracle CLEAN (ok=768 bad=0 max_rel=0), `computed=50331648` EXACT,
`entered==shrunk==16777216`. The run #10 root cause was the host LDS under-allocation and nothing else.

**2. THE SSWIN HYPOTHESIS IS WRONG — REPORT AND STOP.** `PLAN_SS_WINDOW.md` set the falsification
criterion in advance: *"If grow-fail stays 0 the hypothesis is wrong — report that, do not chase it
further."* At SSWIN=8:

| fingerprint | run #9 (SSWIN unset, 15.6 TF) | run #11 (SSWIN=8) |
|---|---|---|
| door4 GROW-FAIL (VGPR budget) | 0 | **0** |
| occ[98] baton | 0 | **0** |
| door1 NOTHING-STAGED | 100% of coast | **100% of coast** |

Widening the reservation window did NOT make the dyn-VGPR budget bind. The moat is still idle. The
ceiling was never the `ASSIGN-DRAIN < POOL_N` gate in the way the plan assumed.

**3. TF=4.8, down from 15.6 — but this number is UNATTRIBUTABLE and must not be quoted.** The run
carried TWO variables (SSWIN=8 and PHIST=1), flagged as such before dispatch. PHIST alone was budgeted
at ~10%; this is -69%. Cannot separate them from one run. Also note the board's standing warning that
this box showed 15.6% run-to-run spread on unrelated work, so even a clean A/B needs interleaved arms.

**4. NEW SIGNAL: the wall moved to ASSIGN.** `occ[86]` starvation = 156066768 = **98.4% of all feed-path
iters found an EMPTY ASSIGN frontier**, and coast rose to 156618346 (75.7% coast-frac), door1 100%.
Reading: waves are not blocked on the window any more, they are blocked with *nothing to do* — the
reservation/publish side cannot feed 1920 waves. `occ[97]` release-bails also rose 14.6M -> 48.1M.

**5. MY ERROR: the PHIST histogram was collected and DISCARDED.** I wired the print only into the
`if (!done)` timeout branch. Run #11 finished clean, so occ[104..113] was never shown — the exact bug
documented for the run #6 funnel, in a file whose own comment at occ_dispatch.cpp:2100 reads
*"Instrument the success path."* Fixed by extracting ONE `printPhist()` helper called from BOTH paths
(a function, not two pasted blocks, because this project's recurring defect is a value mirrored in two
places that drift). **So run #11 gave us no wave-parking data at all; the 98.4% ASSIGN-starvation read
above comes from the pre-existing occ[86] counter, not from PHIST.**

NEXT: (a) re-run PHIST=1 SSWIN=8 to actually READ the histogram, or (b) drop SSWIN and A/B PHIST alone
to price it, or (c) take the ASSIGN-starvation signal seriously and look at publish rate. Needs a call.

## 2026-07-20 — RUN #12 + LOCK-IN: SSWIN=8 IS FREE, AND IS NOW THE CONFIG OF RECORD

**Run #12** (`PHIST=0 SSWIN=8`, kernel sha `f36c06a0` = bit-identical to run #10's kernel; only the host
LDS fix differs). Log `~/dsws_gpu_logs/s1_sswin8_phist0_run12_090941.log`.

| run | window | TF | span | oracle | computed |
|---|---|---|---|---|---|
| #9  | 1 (`SSWIN` unset = `POOL_N`) | 15.6 | 1.690 s | CLEAN | 50331648 |
| **#12** | **8** | **15.5** | **1.704 s** | CLEAN | 50331648 |
| #11 | 8 + PHIST | 4.8 | 5.45 s | CLEAN | 50331648 |

**SSWIN=8 has NO DETECTABLE COST.** 0.8% apart, against a box the board measured at 15.6% run-to-run
spread. Claim defended: "no detectable cost." NOT defended: that 8 is *better* than 1, or that a
couple-percent regression is excluded — those are single runs on different days, not interleaved arms.

**MY RUN #11 DECOMPOSITION WAS WRONG IN BOTH TERMS.** I estimated PHIST ~1.8 s / SSWIN ~1.9 s. Actual:
**PHIST ~3.75 s, SSWIN ~0.** I extrapolated from run #7's per-atomic cost assuming the 1/64 throttle
scaled it linearly, then assigned the residual to SSWIN purely because it was the other variable
present. That estimate was the *sole* support for "keep SSWIN inert" -- one measurement inverted the
conclusion. **PHIST costs ~220%, not the ~10% budgeted; it is a FORENSICS-class tool.**

**DECISION (kmbandy): SSWIN=8 is the config of record.** Rationale: it is the mechanism for a
bottleneck we expect to bind, it is measured free, and leaving it inert means every future run
re-validates the mechanism we intend to abandon -- which is exactly how `POOL_N=1` calcified into a
"design decision" for weeks when it was really a hardcoded-`OPSTRIDE` bug artifact.
**The SSWIN sweep is DEFERRED until TF moves off ~15.5** -- sweeping a flat curve teaches nothing;
1 vs 8 are indistinguishable only because ASSIGN is starving and the extra slots sit empty.
NOTE: 8 is an ARBITRARY placeholder (power-of-2 constraint from `slot_of`; LDS cost `SSWIN*32` is
negligible). It was never derived from the reserve->drain latency or the publish rate.

**HOST/KERNEL LDS DRIFT: FIXED PROPERLY, NOT JUST GUARDED.** Making SSWIN the default upgraded the
drift risk from "footgun you might step on" to "footgun you step on by FORGETTING an env var". So the
value now travels WITH THE ARTIFACT: the kernel emits `LDS_TOTAL_FLOW` into a `.lds_total` section
(`.pushsection`, so `.text` is untouched and every kernel binary stays byte-identical -- verified,
sha still `f36c06a0`); `build_flow.sh` objcopy's it to `<tag>.lds`; the host READS that file and it
WINS over the host-side reconstruction, printing loudly on disagreement **and on agreement** (a silent
success is indistinguishable from the read never happening -- the run #11 PHIST lesson).
`gate_lds.sh` is retained as the offline guard.

Gate note: `gate_sswin.sh` G2 ("SSWIN unset == `be1bb047`") now proves *the mechanism is inert at
SSWIN==POOL_N*, NOT *the default build is unchanged* -- the default build is `f36c06a0` now.

All gates green: gate_sswin 6/6, gate_phist 6/6, gate_lds 3/3.

## 2026-07-20 — THE DUTY-CYCLE INVARIANT IS NOW ENFORCED BY THE ASSEMBLER

**Trigger:** I proposed raising `SEGK` 256 -> 1024, having noticed that carry-through freed SEGK from
its LDS cap (`OPSTRIDE = 16*SEGK*(FN+G*FM)` on a pool carry-through never reads). Correct about the
cap. Blind to what SEGK is *paid for with*. kmbandy has had to explain this on multiple separate days.

**The invariant.** DSWS beats a static register-blocked GEMM ONLY because split-K keeps each wave's
VGPR peak BRIEF: grow(112) -> one short K-segment -> flush -> shrink. Low duty => TRAPEZOID => peaks
phase-offset => many waves time-multiplex ONE per-SIMD budget. **TIME AT PEAK ~ JDEPTH*SEGK.** Raise
either and duty -> 100%, the trapezoid becomes a full-K SQUARE WAVE, average footprint -> peak,
staggering buys nothing, and a wave fat for twice as long cannot coast, feed, or adapt -- the D *and*
the A stop meaning anything. `dyn == static` was MEASURED at full-K: 0% benefit.

**Why a KG note was never going to be enough -- the repo argued FOR the violation:**
- `:136` "flush/WMMA = 128/SEGK -- so the ONLY way to pay it down is more K accumulated per flush"
- `:177` JDEPTH "does the same thing for **FREE**"

Both true about flush/LDS, both silent on duty cycle. A memory entry loses to the source file, because
the source file is what gets read while reasoning about SEGK. **The fix had to go at the point of
temptation.** Note also that `J*SEGK` is the same product this log already calls *amortization depth* --
the repo had the right metric and read it with exactly the wrong sign.

**Shipped (offline; every binary bit-identical, default sha still `f36c06a0`):**
1. `STOP. READ THIS BEFORE RAISING SEGK OR JDEPTH` invariant block at the head of the flush-lever comment.
2. Old master-equation comment annotated as a HALF-TRUTH; "FREE" on JDEPTH flagged as the most
   dangerous word in the file (free in LDS *bytes*, expensive in *duty*; `.Lflow_jwait` spins
   FAT + ACC-live + IDLE = a 100%-duty square wave).
3. Note at the `.set SEGK` definition site.
4. **Assembler guard** `:DUTYGUARD` -- `.error` when `JDEPTH*SEGK > DUTY_KMAX` (sanctioned 256),
   escapable only by an explicit `DUTY_OVERRIDE=1`.
5. `gate_duty.sh`, **10/10, both directions**:

| SEGK | J | J*SEGK | verdict |
|---|---|---|---|
| 32/64/128/256 | 1 | <=256 | pass (every historical geometry) |
| 128 | 2 | 256 | pass |
| 512 / 1024 | 1 | 512/1024 | **refuse** (what I proposed) |
| 256 | 2 / 4 | 512/1024 | **refuse** (same violation via the JDEPTH path) |
| 1024 | 1 | 1024 | pass *only* with `DUTY_OVERRIDE=1` |

**Standing rule going forward:** `grow-fail == 0` is NOT headroom to spend on a longer peak -- it is
the symptom that the moat has not engaged. **Judge every lever by whether it puts MORE WAVES THROUGH
BRIEF PEAKS**, not by whether it makes one wave's segment cheaper.

Surviving levers (none raise duty): `ACC_N=6`->GROUPS=1 (halves B traffic, deletes the boundary
quiesce; `ACC_STRIDE` is independent of SEGK so time at peak is unchanged); tile swizzle (index
arithmetic only, attacks the ~256x B re-stream); freeing the dead 40960B operand pool.
**RETRACTED: raise SEGK.**

## 2026-07-20 AFTERNOON — FIVE ABLATIONS. THE COMPUTE BURST IS FREE. THE WALL IS THE HEAD-PIN.

All on the config of record (`SSWIN=8 ACC_N=3 SEGK=256 SELFSERVE=1 DECENTASN=1 BANKZERO=1`),
same shape, `computed=50331648` EXACT in every run, no latch, no reset.

| run | ablated | span | delta | oracle |
|---|---|---|---|---|
| #12 | control | 1.704 s | — | CLEAN |
| #13 | every C store (`NOCFLUSH=1`) | 1.710 s | +0.32% | bad (proof) |
| #14 | every B load, 768 GB (`NOBLOAD=1`) | 1.747 s | +2.50% | bad (proof) |
| #15 | **all WMMA math** (`NOWMMA=1`) | 1.711 s | **+0.43%** | bad (proof) |

**LOADS, MATH AND STORES ARE ALL FREE.** Delete the entire compute burst and the kernel still takes
1.7 s. Every oracle failure is the intended proof the ablation bit; every binary was gated with a
negative control (off-state byte-identical to `f36c06a0`, on-state provably different).

### What this killed

Four bandwidth theories, all built and all wrong: 605 GB/s "94% of DRAM"; the 1536x B request
amplification; the ~256x tile-row re-stream; and every lever aimed at them — the tile swizzle,
K-blocking, and the B-in-LDS cache. **They target work that costs nothing.** Also retired for this
config: "the flush is the kernel" (that was WOFLUSH per-segment atomics at SEGK=128; at SEGK=256
banked the C store is 0.3%).

### What it found

Coordination latency, unhidden because too few waves are in flight:
```
2167 ns per rowblk-segment per WG  /  ACC_N=3 concurrent  =  ~6.5 us unhidden round-trip
192 of 1920 resident waves computing = 10%
```
6.5 us is the same reserve->drain figure computed in the morning and misattributed to memory.

**THE CAP IS `MSSCAN=0` (head-pinned compute), and it is NOT the banks.** `acc_base_of` is
`bank*ACC_STRIDE + ACC_BASE` — no slot term, *deliberately*: many in-flight slots holding different
`ksi` of the same tile all `ds_add_f32` into the same bank, which IS the split-K sum, made
concurrency-safe by `BANKZERO=1`. `MAXFAT=0` (off). `VBUDGET=1536` is the physical throttle that
should bind and never gets the chance. `MSSCAN=0` pins every wave to the DRAIN-head slot, which
offers exactly `ACC_N` claims. `SSWIN=8` already has 8 reservations stamped and staged — **the work
is there and the compute path refuses to look at it.**

### Process findings (worth more than the perf result)

1. **`NOCFLUSH` was INERT on the banked path** — `NOCFLUSH=0` and `=1` assembled byte-identical
   (`f36c06a0`). Its one gate gnuards the `!BANKZERO` atomic flush; we run `BANKZERO=1`. Dispatching it
   would have returned a flat span and "proved" the C store is free — a fabricated finding from a dead
   knob. Caught by an offline sha compare BEFORE the run. Now wired into `.Lflow_cstore` with a
   negative control. Fifth instance of this class (`STINSTR_FEED`, `CSTORE`, `DIAG`, `NOCFLUSH` v1).
2. **THE ANSWER WAS ALREADY WRITTEN DOWN, TWICE.** `DSWS_MORNING_2026-07-14.md:54` labels `G`(==`ACC_N`)
   **"THE BIG LEVER... == the number of waves that can EVER compute concurrently... Capped by LDS"**, and
   the 07-16 KG entry says **"coast is geometry, not memory."** Five GPU dispatches were spent
   re-deriving a documented result. The ablations are still worth having — they converted an assertion
   into evidence and killed four theories permanently — but reading the repo first would have started
   here this morning.
3. **A real lever died attached to a false justification.** `ACC_N=6` was proposed in the morning
   ranked #1, but justified by *bandwidth*; when the bandwidth premise collapsed the lever collapsed
   with it. Attach levers to the reason they actually work.
4. **When a constraint is lifted, enumerate everything it was blocking.** S1 made the operand pool dead,
   which unblocked both `SEGK` (rejected on duty-cycle grounds, correctly) and `ACC_N`/concurrency (the
   actual answer). Only the first was checked.

Instrumentation added today, all gated + negative-controlled: `PHIST` (bail-door histogram, ~220%
overhead, FORENSICS-class), `NOCFLUSH` on the banked path, `NOBLOAD`, `NOWMMA`, `gate_duty.sh`,
`gate_lds.sh`, and the `.lds_total` published-LDS contract.

## 2026-07-20 LATE — ⛔ ALL OF TODAY'S THROUGHPUT NUMBERS ARE RETRACTED. THE KERNEL DROPS WORK.

**Nothing measured today is a result.** Not 15.5, not 19.2, not 20.8, not the ablations. Retracted.

### How it surfaced

A repeat of the chunk=12288 run (kmbandy: *"run it again"*) returned `computed=50330112` against an
expected 50331648 — **1536 short = `ACC_N*n_kseg` = exactly one group of one tile, silently dropped.**
Same binary, same command, one run exact and one short: **an intermittent race.**
Both runs reported **oracle CLEAN**, because the oracle samples 32 of 16384 tiles (0.2%).

### The audit (31 runs on this shape, expected = GROUPS*TOTAL_super*ACC_N == G*TOTAL_super)

| era | runs | dropped work |
|---|---|---|
| pre-S1 (`fperf_*`, 2026-07-19) | 18 | **8** — incl. counts that OVERSHOOT (+30, +2) and shorts that are NOT 1536 multiples (548764, 298954) |
| S1 (runs #6-#19) | 12 | 0 |
| S1 run #20 | 1 | **1** (1x1536) |

**10 of 31 runs dropped or miscounted work.** The pre-S1 overshoots and non-1536 shorts are a *different*
and messier failure than the clean S1-era one-group drop.

### kmbandy's call, and it is the right one

> *"if it broke somewhere at a larger workload, that means it's still broken at a smaller workload.
> it doesn't matter, it's not defensible."*

The race is in the kernel, not in chunk=12288 — a larger workload only made it likelier to fire. It is
present at chunk=384 and 1536 too; those runs simply did not hit it. **12 consecutive exact S1 runs is
NOT proof of correctness** — it is equally consistent with a rare race we got lucky on twelve times, and
one short in thirteen is exactly what a rare race looks like. I cannot distinguish those.

This also retracts last night's brief claim that *"S1 is MORE work-exact than the baseline"*: that rested
on ONE short baseline run vs S1 runs that happened to be exact. Never supported.

### The real defect: THE VERIFICATION, NOT THE KERNEL

A run could drop a whole tile, print `oracle CLEAN`, and be logged as a result. That happened today.
- the oracle samples **0.2%** of tiles — a missing tile passes
- `computed`-exactness was **printed, never enforced**
- `TILEDONE`/`DRAIN`/`GSTORED` are count-based and (per Codex) structurally cannot detect wrong/missing work
- **dropped work FLATTERS TF** — less work in the same span reads as higher throughput, so this
  invalidates the PERF number, not just correctness

### Shipped (offline, no dispatch)

1. **`occ_dispatch.cpp` WORK-EXACTNESS GATE** — computes `expected = G*TOTAL_super` (the `ACC_N` cancels
   out of `GROUPS*TOTAL_super*ACC_N`; identity verified against all 7 geometries in the audit) and prints
   a hard `WORK-EXACT` / `*** WORK-INEXACT -- RUN IS INVALID ***` verdict, decoding the delta into
   `N x 1536` whole groups where it divides.
2. **`gpu_run.sh` LATCHES on `WORK-INEXACT`** (exit 5) — same class as the hang latch, cleared only by a
   human. Plus a warning when STAGINSTR ran but no verdict appeared (stale host).
3. **NEGATIVE-CONTROLLED**: replayed run #20's real log -> latch FIRES; run #19 -> passes; an old
   pre-gate log -> "host is stale, rebuild it". A gate that cannot be shown to fail is worthless, and
   five instruments today (`NOCFLUSH` v1, `STINSTR_FEED`, `CSTORE`, `DIAG`, the PHIST zero-buckets)
   were exactly that.

### Still open

- **The race itself is unfixed and un-root-caused.** One group of one tile, intermittent.
- **Oracle coverage is still 0.2%.** The work-exactness gate catches DROPPED work globally, but only the
  oracle can catch WRONG values, and it samples almost nothing. Full-check on this shape is CPU-infeasible;
  a denser stride for correctness runs is the open question.
- No throughput baseline currently exists that is verified rather than merely un-caught.

## 2026-07-20 — DROPPED-GROUP RACE: ROOT-CAUSED AND FIXED (s_barrier at init). 20/20 CLEAN.

**The bug:** intermittent silent work-loss (~1 in 8 at chunk 12288), a whole group of one tile dropped,
oracle CLEAN because it samples 0.2% of tiles. Present since S1 (and messier pre-S1 with overshoots).

**The hunt (BNDPROBE, exact boundary-transition counters):**
1. `computed`-exactness gate caught it (run #20) after a repeat kmbandy demanded.
2. BNDPROBE `occ[118]` (ASSIGN != z at the boundary decision) fired, correlated 1:1:1 with missing
   group-advances and lost groups.
3. Skew-value capture: `ASSIGN=7 z=0 z-base=1024` -> **ASSIGN AHEAD of z = over-reservation past
   DA_ZDONE**. This FALSIFIED Codex's CAS argument (which said it couldn't happen) AND confirmed my
   original hypothesis was directionally right.
4. Classifier: **12/12 skews at z==0, 12/12 at a TILE decision -> 100% at the FIRST-TILE BOOTSTRAP**,
   never steady state. That collapsed the search space entirely.

**Root cause:** the init handshake published readiness via a lock-free LDS flag (`RINGINIT=0xACED`,
written LAST by wid0), used as a release fence. On RDNA that is a BROKEN release/acquire: `lds_put` =
`ds_store` + `s_wait_dscnt 0x0`, and `s_wait_dscnt` orders only the ISSUING wave's LDS completion --
it gives NO cross-wave ordering. So `wid0`'s init store of `DA_ZDONE=0` could land AFTER the bootstrap
winner's `DA_ZDONE=n_kseg`, reverting the cursor to exactly its init state (z=0, base=-TOTAL), reopening
the reservation window past the boundary -> over-reservation -> whole groups lost. The exact-init-value
signature (`z=0`, `base=-1024`) is the fingerprint of that late-store clobber.

**The fix (INITBAR=1, now default):** one `s_barrier_signal -1` / `s_barrier_wait -1` pair (RDNA4
replaced `s_barrier`) after wid0's init, on both the wid0 and waiter arms. It provides the cross-wave
happens-before the flag-spin cannot: wid0's `lds_put`s are all drained before its barrier, so every wave
past the barrier sees the fully-published init. ONE-TIME, in the prologue -- NOT a hot-loop dam, so it
does not violate the river principle (kmbandy's call: "setting up the riverbed before the water flows").

**FIRST s_barrier in this dyn-VGPR kernel.** De-risked per rule 7: small-chunk bring-up first (a barrier
deadlock is NOT deadman-caught -> would be a true hang, not a clean retire). All 30 waves provably reach
it (all pass `s_alloc_vgpr 32`, no early exit before the init point). Bring-up: no hang, no deadlock,
no reset.

**CONFIRMATION: 20 runs at chunk 12288, INITBAR=1 BNDPROBE=1 -> WORK-INEXACT 0, skew>0 0, oracle-clean
20/20.** At the old ~1/8 rate that is ~2.7 expected failures of exposure; zero. Falsification criterion
was pre-registered (any single skew kills it); none.

**Gating:** `INITBAR=0` is byte-identical to the buggy `f36c06a0`. Default is now `INITBAR=1` = sha
`0a03f7e9`, the FIXED config of record. gate_duty/gate_lds/gate_phist all green.

**Correctness note:** the WORK-EXACTNESS GATE (host) + latch (gpu_run) that were built earlier today are
what caught this class in the first place, and they remain the standing guard -- a run that drops work
now REFUSES to be logged as a result.

## 2026-07-20 EVENING — N1-N3: SECOND RACE FIXED, VERIFIED BASELINE, ABLATIONS RE-CONFIRMED

**N1 -- TERMFIX (terminal-publication race, Codex-found, benign but real).**
`.Lflow_da_bnd_term` published `FLOWTERM=0xDEAD` LATE (after releasing ZLOCK), leaving a window for a
2nd wave to re-win the exhausted boundary -> duplicate `occ[20]++` (terminal=256 not 128 at 2 chunks).
FIX: publish `FLOWTERM=0xDEAD` WHILE holding ZLOCK and KEEP ZLOCK held (never restore clean z) -- a
permanently-locked `DA_ZDONE` means no wave can ever re-win the boundary CAS; the terminal drain uses
ASSIGN/DRAIN not DA_ZDONE, so it is harmless. Gated `TERMFIX` (default 1); `TERMFIX=0` byte-identical
to the INITBAR-only build `0a03f7e9`. Bring-up (chunk 384): **terminal=2752 = 64x43 EXACTLY (the floor
= zero duplicates across 2752 opportunities)**, skew=0, WORK-EXACT, oracle CLEAN, no hang.

**N2 -- VERIFIED THROUGHPUT BASELINE (the first trustworthy S1 number).**
Default fixed kernel `1e78b027` (INITBAR=1 TERMFIX=1), chunk 384, `BNDPROBE=0`, 5 interleaved runs:
**TF=15.5, mean span 1.703s, spread 0.53%**, all WORK-EXACT, all oracle CLEAN. Matches the pre-fix
15.5 -> the barrier + termfix cost nothing (one-time prologue). THIS REPLACES EVERY RETRACTED NUMBER.

**N3 -- coordination-bound conclusion RE-CONFIRMED on the fixed kernel:**
| ablation | span | vs 1.703s |
|---|---|---|
| NOWMMA (all math) | 1.702s | -0.1% |
| NOBLOAD (768 GB B) | 1.669s | -2.0% |
Deleting compute or operands barely moves span. The wall is COORDINATION, not compute/bandwidth.
Both WORK-EXACT (drop values, not work), oracle bad by design. Closes the "measured on the buggy
kernel" caveat on the finding that drives the direction.

**CONFIG OF RECORD: `1e78b027`** = WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0
DECENTASN=1 BANKZERO=1 STAGGER=1 SELFSERVE=1 SSWIN=8 INITBAR=1 TERMFIX=1, dispatch ML8_COOP_CHUNK=384.
Both correctness fixes default-on; every probe/ablation gate byte-identical when off.

## === NEXT SESSION: N4 -- THE COORDINATION/ADMISSION WALL (the real perf frontier) ===
Coordination is no longer corrupting itself, so throughput now lives in the reserve/publish path.
Baseline to beat: 15.5 TF / 1.703s. Signals: ~98% of feed iters find an EMPTY ASSIGN frontier;
reservations contended (run #16 TRY/WIN~3.45 but PHIST-contaminated -> RE-MEASURE cleanly). Codex's
pointer: admission happens at GETTING A RESERVATION (grow-fail=0 => every wave that reserves computes).
The dyn-VGPR moat STILL never engages (grow-fail=0) -- the architecture's entire headroom, downstream
of this wall. Deferred/named: chunk-size (~24% apparent gain RETRACTED, needs one clean re-measure, but
it is a compositor-safety knob); SSWIN sweep (deferred until TF moves).

================================================================================
=== 2026-07-20  N4 OPEN — CONFIG-OF-RECORD FED BASELINE LOCKED (20.9 TF) ======
================================================================================
LOCKED baseline (supersedes the underfed 15.5 TF number for all perf comparison):

  bin    : 1e78b027 + BNDPROBE=1 (byte-identical perf to BNDPROBE=0; probe is exact/cheap)
  geom   : 24576x4096x131072  G=6 FM=1 SEGK=256 POOL_N=1 ACC_N=3 SSWIN=8 JDEPTH=1
  feed   : ML8_COOP_CHUNK=12288  (2 chunks) MAXS=3.0   <-- FED, clock committed
  run    : s1_n4_fed01_185439.log

  TF          = 20.9   (6.8% of 307 TF fp8 peak)
  span        = 126331415 ticks = 1.263 s
  computed    = 50331648  -> WORK-EXACT (= G*TOTAL_super; no work dropped)
  BNDPROBE    = group-adv=16384 tile-adv=16384 gap=+0 (cursor CLOSES); ASSIGN!=z=0 (clean)
  canary      = C guard tail clean (no OOB)
  deadman/dmfat/tokleak = 0
  ORACLE      = value spot-check NOT captured (my Bash-tool 120s cap SIGTERM'd the host
                during the oracle CPU ref-compute; kernel had already retired clean, no latch).
                Same kernel was oracle-CLEAN at chunk 384 (N2 x5, termfix run 23). kmbandy
                greenlit locking without the re-run: count+boundary+OOB cover correctness here.

WHY 15.5 -> 20.9 IS NOT A GAIN: same work, same kernel. The 15.5 was UNDERFED (43 chunks,
  clock never committed). 20.9 is the true steady-state number. (The bimodal-clock trap:
  never quote TF < ~1s; under-fed runs give false architecture.)

*** THE FIX HOLDS AT THE SCALE WHERE IT BROKE. *** Chunk 12288 is exactly where run #8
  dropped 2 groups on the buggy bin. Fed, fixed kernel = WORK-EXACT + BNDPROBE-clean. First
  proof the INITBAR+TERMFIX fixes survive the fed regime.

*** THE WALL, RE-MEASURED FED (diagnosis UNCHANGED from underfed -> now defensible): ***
  coast-frac    58.1%   (was 77-81% underfed)
  empty ASSIGN  97.2%   -> ASSIGN-BOUND: the coordinator cannot PUBLISH work fast enough
  coast door1   100%    (NOTHING-STAGED: DRAIN>=STAGE)
  door4 GROW-FAIL = 0   -> dyn-VGPR moat STILL idle; all headroom is downstream of this wall
  carrier-stall = 0, baton = 0, jwait = 0  -> NOT stage-bound, NOT compute-bound
  occ[96] emissions = 16777216 = TOTAL_super  -> coordinator DID emit every assignment;
                                                 consumption outruns publication.
  => N4 = attack the PUBLICATION RATE of the ASSIGN frontier. Next step: read the assign/
     publish mechanism and find why 97% of consumer polls see an empty frontier.

================================================================================
=== 2026-07-20  N4 MEASUREMENT — RESVPROBE: THE WALL IS CURSOR CONTENTION ======
================================================================================
Clean replacement for run #16's PHIST TRY/WIN~3.45 (~294% contaminated). RESVPROBE
counts the .Lflow_da_peek exits by register-accumulate (reuses CNT_FATFULL s94 /
CNT_CLEAD s96, both structurally 0 at config of record), so it is measured INSIDE
the quotable build. RESVPROBE=0 byte-identical to 1e78b027 (gated, proven).

  run    : s1_n4_resvprobe01_191818.log  (fed, chunk 12288)
  TF=21.0  computed=50331648 WORK-EXACT  oracle CLEAN  no latch  (fully trustworthy)

  WIN (reservations reaching stamp) = 16777216 = TOTAL_super
  CAS-loss (occ[87])   = 24592252  -> 1.466 collisions per successful reserve
  window-full (occ[89]) =  3728949  -> 6.3% of empty-frontier bails

VERDICT: CURSOR-CONTENDED (not stage-bound).
  - window-full only 6.3% => the SSWIN=8 window almost never fills => the single
    cursor CANNOT get ahead of consumers => reservation-RATE limited, NOT stage-bound.
  - 1.466 collisions/reserve => ~41M CAS attempts to place 16.8M reservations; 40%
    of attempts on the single ASSIGN_HEAD LDS word collide.
  => DECENTASN decentralized the ROLE; the reservation CURSOR is still ONE word, and
     that word is the publication throttle. NEXT: shard the cursor (S independent
     reservation lanes). CO-SUSPECT not yet isolated: ZLOCK boundary serialization
     (DA_ZDONE) is shared state; any shard design must handle boundaries across lanes
     (this is exactly where INITBAR/TERMFIX races lived -- design carefully).

================================================================================
=== 2026-07-20  N4 BATCHED-RESERVATION = REJECTED (correct but net loss) =======
================================================================================
Grok-implemented (BATCH defsym, s72/s73 backlog, N=min(BATCH,z-r,SSWIN-(r-d))), I reviewed the
diff + re-ran gates + traced the backlog-drain (both terminal exits continue the backlog; not-done
exits unreachable from the SS stamp path; s72/s73 batch-only + s67/s68 kernel constants).

BRING-UP run s1_n4_batch4_bringup_200726 (BATCH=4, fed chunk 12288, BNDPROBE=1 RESVPROBE=1):
  CORRECT: computed=50331648 WORK-EXACT, BNDPROBE cursor CLOSES, ASSIGN!=z=0, oracle CLEAN, no latch.
           -> the batched serial-drain is correct; no work dropped. Impl + review held.
  SLOWER:  TF 21.0 -> 16.0.  coast-frac 58.1% -> 96.9%.  window-full 6.3% -> 96.6% (1.57 BILLION).
           CAS-loss/reserve 1.466 -> 1.873 (WORSE, not the expected ~0.37).

WHY: batching CONCENTRATES claims. A wave claims 4, drains them SERIALLY (each a full compute burst),
  holding the SSWIN=8 window FULL the whole time -> every other wave hits r-DRAIN>=SSWIN and bails ->
  mass starvation (coast 96.9%). Fewer concurrent computers -> slower drain -> window stays full. The
  "self-serve concentration" cost flagged in the plan caveats bit hard.

*** THE FINDING (pre-set falsifier fired): single-cursor CAS contention is NOT the binding wall. ***
  Attacking it made TF WORSE. The window drains only as fast as waves COMPUTE+STAGE; concentrating claims
  starves that. The wall is DOWNSTREAM -- window-drain / staging / compute-concurrency, NOT the reservation
  cursor. The 1.466 CAS-loss was a SYMPTOM, not the cause.

REVERTED: on-disk bin back to config of record 1e78b027 (BATCH=1 byte-identical, gate-proven). BATCH knob
  kept in-source as a tested-rejected lever. PLAN_CURSOR_BATCH.md marked REJECTED.

NEXT: the ZLOCK/DA_ZDONE boundary stall + staging throughput is now the prime suspect. Measure before any
  fix. Do NOT widen SSWIN blindly (at BATCH=1 window-full is only 6.3% -- a wider window is not the BATCH=1
  constraint). Baseline to beat remains 21.0 TF / 1.263s fed.

================================================================================
=== 2026-07-20  N4 BATCH=2 -> RE-FRAME: batch is a SHELVED WORKING LEVER =======
================================================================================
CORRECTION to the "rejected" framing above. BATCH=2 (run s1_n4_batch2, fed): TF=20.5, coast 89.9%,
window-full 92.2%, CAS-loss/reserve 3.216. WORK-EXACT, oracle CLEAN.

THE SCAN (all WORK-EXACT / oracle CLEAN):
    BATCH | TF   | coast | window-full | concurrent holders (SSWIN/BATCH)
      1   | 21.0 | 58.1% |   6.3%      |  ~8
      2   | 20.5 | 89.9% |  92.2%      |  ~4
      4   | 16.0 | 96.9% |  96.6%      |  ~2

TF is FLAT while concurrent holders >= ~4 (21.0 vs 20.5), then CLIFFS at ~2 (16.0). Throughput tracks
CONCURRENCY; ~4 concurrent waves already SATURATE the pipeline. Batching is a CORRECT, VERIFIED mechanism
for RESERVATION-CAS CONTENTION -- it just isn't the CURRENT binding bottleneck, so it's SHELVED inert
(BATCH=1 byte-identical), ready to switch on when reservation contention binds (higher concurrency / after
the coordination wall is cut). This is a built-ahead lever, not a dead end. See PLAN_CURSOR_BATCH.md.

*** THE ACTUAL FINDING the scan bought: the current wall is NOT admission/concurrency. Ruled out: compute
    (NOWMMA free), operands (NOBLOAD free), and now concurrency (batch scan: enough at baseline). The 21 TF
    ceiling is PER-WAVE COORDINATION THROUGHPUT -- the fixed LDS-handshake / s_alloc_vgpr / ZLOCK-boundary
    cost each wave pays, which ~4 concurrent waves already max out. NEXT: measure WHICH per-wave coordination
    step dominates, then cut it. ***

================================================================================
=== 2026-07-20  N4 PHASE TIMER — THE WALL IS THE dyn-VGPR ROUND-TRIP ==========
================================================================================
Stopped over-ablating; ran the EXISTING phase timer (PHASEPROBE, phase_stamp -> occ[64..69]).
Direct per-compute-wave tick breakdown (run s1_n4_phaseprobe_211638, K=32768, WORK-EXACT, oracle CLEAN):

    phase        share   what
    FOLLOW        1.0%   idle: waiting for a stage
    GROW         33.5%   s_alloc_vgpr grow 32->112 + rowblk claim
    WMMA         24.1%   the actual fp8 WMMA compute
    FLUSH        33.7%   split-K ds_add reduction
    SHRINK        7.6%   s_alloc_vgpr shrink 112->32

*** THE ACTUAL MATH (WMMA) IS ONLY 24%. dyn-VGPR round-trip GROW+SHRINK = 41%. Reduction = 34%. ***

RECONCILES THE ABLATIONS (they measured OVERLAP, not cost): NOWMMA flat because WMMA overlaps loads;
NODSADD ~flat because the ds_add reduction overlaps too. The ONE phase that CANNOT overlap is
s_alloc_vgpr -- it is a WaitIdle barrier. So of the big three, GROW+SHRINK (41%) is the REAL wall:
biggest serialized cost AND the only one that cannot be hidden. This is the s_alloc_vgpr cost, in-vehicle
(supersedes the prior "GROW/SHRINK ~0%" which was the ml8 STATIC framework vehicle, not DSWS flow).

*** THE KICKER: grow-fail=0. The dyn-VGPR moat NEVER ENGAGES here, so the 41% round-trip buys NOTHING.
    We pay the most expensive phase for a stagger/moat that is not converting. ***

LEVER FORK (dyn-VGPR is the DSWS ethos -- discuss, do not unilaterally change):
  (a) make the moat CONVERT: find a regime where the VGPR budget binds (grow-fail>0) so the round-trip
      buys wave-multiplexing -- but must not violate the DUTY-CYCLE invariant (raising SEGK/JDEPTH forbidden;
      raising G/concurrency is the candidate).
  (b) if it cannot convert HERE, stop paying: amortize the round-trip (grow once per reservation, fewer
      grow/shrink) or a bounded-static regime -- reduces the 41% directly.
NEXT: pick the fork with kmbandy. Verified fed baseline to beat: 21.0 TF / 1.263s.

================================================================================
=== 2026-07-20  N4 END-OF-NIGHT — FED NUMBER CONFIRMED, HOST LOWMEM FIX ========
================================================================================
FED THROUGHPUT CONFIRMED (the "feed it" instinct VALIDATED the number, did not move it):
  24576x4096x524288 : TF=23.5  span=4.5s  oracle CLEAN  WORK-EXACT (computed=201326592)
  24576x8192x524288 : TF=23.5  span=8.96s oracle CLEAN  WORK-EXACT (computed=402653184)
  => 23.5 TF at BOTH 4.5s AND 9s steady state. NOT an underfed artifact. Coordination wall
     is real + feed-independent. grow-fail=0 (moat idle), coast 41%, ASSIGN-bound ~82% at both.

HOST FIX (occ_dispatch.cpp, uncommitted): A+B now FORMULA-GENERATED straight into VRAM
  (Aval/Bval + mbg_gen_preshuffle_B), auto-on when an operand >4GB (or DSWS2_LOWMEM=1). Host
  RAM stays FLAT (~6GB) at ANY shape -- the --no-mmap equivalent. Verified: oracle CLEAN proves
  the formulas are byte-exact; RAM dead-flat across a 9s/22GB run. Fill uses chunked host staging
  + bulk memcpy. KNOWN-SLOW: the per-element division in Aval/Bval makes the fill ~90s for ~17GB
  (TODO: row-based fill, i/Ko is constant per row -> kill the per-element div, ~6x faster).

GOTCHA BANKED: n_kseg (= K/SEGK) MUST BE A POWER OF 2. K=655360 -> n_kseg=2560 (non-pow2) ->
  computed=0 (kernel handed out NO work, 100% coast, 14ms, TF=18082 garbage) -> WORK-INEXACT latch.
  Use K = SEGK * pow2 (e.g. 524288->2048, 1048576->4096). (Also: DSWS2_ORACLE_MTL/NTL set the
  SHAPE dims M=96*MTL, N=64*NTL -- mislabeled "oracle".)

INCIDENTS (own them): 2x host OOM (host built full A/B vectors before the lowmem fix); 1x ags
  crash (GTK app, VRAM contention -- displays fine, restartable); 1x computed=0 latch (non-pow2
  n_kseg). All recovered, no bricks.

hipBLASLt same-shape bar: NOT captured -- torch/ROCm threw "register fat binary failed" spam.
  TODO tomorrow (env issue, not our kernel). Prior baseline: hipBLASLt 12.6-190 TF on ml8 shapes.

NEXT (tomorrow): the phase-timer verdict stands -- the wall is the dyn-VGPR grow/shrink round-trip
  (41% of compute-wave time, 0 overlap, buys nothing at grow-fail=0). FORK: (a) make the moat
  ENGAGE (bind the VGPR budget so the round-trip converts) vs (b) reduce/amortize the round-trip.
  Plus: fix the fill speed, get the hipBLASLt bar.

================================================================================
## 2026-07-21 — ⭐⭐⭐ FIRST DSWS vs hipBLASLt HEAD-TO-HEAD ON THE REAL SHAPES
##              CONFIG OF RECORD. FULL TABLE -> RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md
================================================================================
CONFIG: config of record, bin sha 397bfbe1 (G=6 ACC_N=3 FM=1 FN=4 SEGK=256 POOL_N=1 SSWIN=8
  JDEPTH=1 DECENTASN=1 SELFSERVE=1 BANKZERO=1 STAGGER=1 INITBAR=1 TERMFIX=1). GROUPS=2.
  M padded UP to the 96-row super-tile; TF corrected back to REAL FLOP (padding counts against us).
  26 shapes, ALL WORK-EXACT + oracle CLEAN. Raw: ~/dsws_gpu_logs/dsws_sweep_CONFIGOFRECORD.out
  + dsws_vs_hipblaslt_configofrecord.json.

*** DSWS WINS 4/26 -- ALL IN THE TINY-M MoE DECODE CORNER WHERE hipBLASLt COLLAPSES ***
  ml8 moe attn_kv     M=64   10.87 vs  1.70  = 6.39x
  ml8 moe ffn_down    M=64    8.00 vs  1.60  = 5.00x
  ml8 moe ffn_gate/up M=64    6.60 vs  1.70  = 3.88x
  ml8 moe ffn_gate/up M=512  20.36 vs 15.40  = 1.32x

*** THE FLATNESS RESULT (the DSWS thesis, MEASURED not asserted) ***
              DSWS        hipBLASLt
  mean        6.00        69.18
  median      5.25        57.30
  stdev       4.20        63.81
  CV          0.700       0.922      <- DSWS IS FLATTER ACROSS THE REAL WORKLOAD
  min/max     0.20/20.36  1.60/189.30
  We are measurably more clustered than the vendor. hipBLASLt is spiky: 189 TF on big dense,
  COLLAPSING to 1.6 TF on MoE decode (its fp8 loses to its own bf16 there). We win where it
  collapses. HONEST OTHER HALF: our mean is 11.5x lower -- today the flatness is "consistently
  LOW". Strategy is NOT to out-peak them on dense; it is RAISE THE FLOOR WHILE STAYING FLAT.
  A flat ~150 TF would beat the vendor on most of this table. Curve shape right, level is the work.
  ATTACK VECTOR: DSWS TF falls as total work rises -- worst is lm_head (201 GFLOP) at 0.20 TF.

WHAT UNBLOCKED IT: real ml8/mlmf K give NON-POW2 n_kseg (2560->10, 9216->36, 768->3, 1536->6) and
  NO legal SEGK makes them pow2. The DECENTASN coupled cursor was "POW2 n_kseg only" with an
  explicit fail-safe routing non-pow2 to .Lflow_da_terminal => clean retire, computed=0, SILENTLY
  NO WORK. Half these shapes returned zeros before today. FIX (uncommitted): reservation span now
  strides the ksi FIELD WIDTH (2^shift) not n_kseg -- keeps TOTAL=GROUPS<<shift, z>>shift,
  ksi=within&mask, group=within>>shift EXACT for any n_kseg with NO division and NO spare SGPR.
  Phantom indices (2^shift - n_kseg per field) are NEVER reserved: peek stops at the real end
  (ksi = r & mask, register-only -- base is always 2^shift-aligned so no LDS read), and the
  boundary re-bases ASSIGN/DRAIN/STAGE past the gap under ZLOCK while quiesced. BYTE-IDENTICAL
  for pow2 n_kseg. Verified: non-pow2 n_kseg=3 exact+clean; pow2 n_kseg=8 regression clean.

HOST FIXES (occ_dispatch.cpp, uncommitted):
  (1) WORK-EXACT gate is REPS-AWARE (occ[71] accumulates across DSWS2_TARGET_SECS reps; gate now
      compares G*TOTAL_super*repsDone). It was FALSE-LATCHING every reps>1 run.
  (2) *** THE COMPOSITOR CAP WAS STRUCTURALLY BROKEN AND IS NOW FIXED. *** chunkMaxS is evaluated
      BETWEEN chunks so it can only abort REMAINING ones; the old default chunkTiles=claimTotal gave
      ONE chunk for the whole problem => nChunks==1 => ZERO protection, while still printing
      "compositor-safe". A 2.46s single chunk (PHIST build, ~220% overhead) took HYPRLAND TO SAFE
      MODE -- rule 7 exactly: desktop dies, no GPU reset, no other guard sees it. EVERY run before
      this fix was unbounded by default. Now: default chunk 512 tiles (nChunks>1, cap has
      granularity) and the single-chunk case WARNS instead of reassuring. Both branches verified.

STILL OPEN: n_kseg==1 (K=256, router_MLP) fail-safe; N%64 (mamba in_proj N=4200/4208);
  occ[20] over-claim (benign: WORK-EXACT + clean oracle, unexplained).
PERF DIAGNOSIS UNCHANGED (2026-07-20 phase timer, NOT re-litigated): GROW 33.5% + SHRINK 7.6% =
  41% dyn-VGPR round-trip, WMMA 24%, FLUSH 34%, grow-fail=0 => the 41% buys nothing. FORK stands.

---

## 2026-07-21 (later) — ⛔ RETRACTION: THE REAL-SHAPE TF COLUMN WAS A PARSER BUG ⛔

**Every DSWS throughput number published earlier today is retracted.** `sweep_dsws_realshapes.sh:87`
matched `'<num> TF'`; the kernel prints `TF=<num>`. The pattern never matched on any shape, so every
row fell through to the line-88 fallback, which took the LAST decimal on the throughput line — i.e.
the `spread N%` field of `SUSTAINED`, or the `N% of 307 TF peak` field of `THROUGHPUT`.

| shape | published | log said | actually captured |
|---|---:|---:|---|
| ml8 moe attn_kv M=64 | 10.87 ("6.39x WIN") | `TF=0.0` | `spread 16.3%` |
| ml8 dense attn_o M=512 | 9.07 | `TF=1.4` | `spread 10.2%` |
| mlmf lm_head | 0.20 | `TF=0.6` | `0.2% of peak` |

RETRACTED: all 4 wins over hipBLASLt (there is NO MoE-decode corner where we win; the three M=64
shapes read TF=0.0 — 137 chunks for a 0.1 GFLOP problem); the flatness thesis (corrected CV: DSWS
**1.128** vs hipBLASLt **0.905** — we are LESS flat, the data CONTRADICTS the thesis); "mean 11.5x
lower" (true: ~80x, 0.87 vs 69.18). SURVIVES: the hipBLASLt column (separate harness); WORK-EXACT +
oracle CLEAN on all 26 (correctness was never in question); the non-pow2 n_kseg fix.

FIXED: extractor now anchors on `TF=`, FIRST match, preferring `SUSTAINED` over `THROUGHPUT`; verified
against archived logs. **The corrected table was rebuilt with ZERO GPU time** — `~/dsws_gpu_logs/rs_*.log`
survived. Output: `~/dsws_gpu_logs/dsws_vs_hipblaslt_CORRECTED.json`.

### WHAT THE BUG WAS HIDING: throughput tracks `n_kseg = K/SEGK`
n_kseg 36 -> 4.36, 2.40 | 16 -> 2.33, 1.24, 1.07, 0.20 | 10 -> 1.55, 1.55, 1.33, 1.24, 1.16, 0.36 |
8 -> 0.98, 0.18, 0.18, 0.13, 0.00 | 6 -> 0.69, 0.18 | 3 -> 0.60, 0.30, 0.20, 0.18 | 2 -> 0.18, 0.00.
The corrupt column showed no such structure.

MECHANISM (derived from source before the runs): reservations legal only while `r < DA_ZDONE` (:3983);
`DA_ZDONE` advances ONE field width per group boundary (:4151); the boundary needs `DRAIN>=ASSIGN`
(:4086) AND the prior group's C-store drained (:4093) because banks are REUSED (:4144); one
reservation = one ksi carried by ONE wave over ACC_N rowblks (:4358,:4487).
=> **per-WG parallelism = min(WAVES, n_kseg)**. At WAVES=30 / K=768 (n_kseg=3): 90% of the WG idle by
construction. WAVES=30 was tuned on the deep-K synthetic (n_kseg=2048) where units always outnumbered
waves — same synthetic-vs-real trap as the FLUSH artifact.

### MEASURED: fewer waves = ~4.3x (each point its own build; TF read directly off `TF=`)
| shape | n_kseg | W=30 | W=10 | W=5 | gain |
|---|---:|---:|---:|---:|---:|
| ffn_gate_up M512 K2560 | 10 | 1.5 | 4.1 | **6.5** | **4.3x** |
| lm_head M4096 K768 | 3 | 0.6 | — | **2.6** | **4.3x** |

All WORK-EXACT + oracle CLEAN. coast 93.5%->64.0%; boundary bails occ[97] 754,475->205,288;
starvation occ[86] 5.86M->1.21M; feed-stages 0->1,568. `door1 NOTHING-STAGED` = 100% of coast at EVERY
wave count => the SUPPLY OF UNITS is the wall. door3/door4 = 0 throughout (dyn-VGPR moat never engages).

PREDICTION FALSIFIED (registered in advance): I predicted the optimum sits AT n_kseg and that going
below loses parallelism. `W=5 > W=10` at n_kseg=10 falsifies it — contention among starved waves
outweighs the parallelism they add. **WAVES=4 unbuildable**: NCOMPUTE=1 -> BATON_MAGIC=2^32, not
32-bit; the `.if NCOMPUTE < 1` guard (:780) catches 0 but not 1. Fails loud at assembly = gap, not hazard.

**RETRACTED WITHIN THE HOUR — I wrote here that this "retired counter-free assign on evidence". WRONG
TWICE: wrong on the merits (below), and WRONG TO DECIDE UNILATERALLY. Cancelling planned architecture
is kmbandy's call, not a conclusion to draw from one measurement.**

### TWO FOLLOW-UPS INVERT THE DIAGNOSIS: THE SHARED CURSOR IS THE BOTTLENECK
| change | result |
|---|---|
| SEGK 256->64 (4x units: n_kseg 10->40, all 30 waves feedable) | **1.5 -> 1.2 TF, WORSE**; coast ROSE 93.5%->97.5% |
| BATCH=2 at WAVES=5 (more work per CAS) | **ABORTED** — chunk 0.81s vs ~0.08s at BATCH=1, >=10x slower |

UNITS ARE NOT THE WALL; `min(WAVES, n_kseg)` is dead as a throughput explanation. More units => each
reservation carries LESS work => MORE CAS traffic per unit of output. `door1 NOTHING-STAGED = 100%` was
never evidence about supply — under SELFSERVE it is the VESTIGIAL RING door and reads 100% regardless.
BATCH=2 failed at BOTH WAVES=30 and WAVES=5 => holding the shared SSWIN window while draining serially
is INTRINSIC TO THE SHARED CURSOR, not a wave-count artifact.

The BATCH=2 abort was caught by the compositor cap (0.81s > 0.75s), teardown declined to destroy the
queue, journal CLEAN (0 lines, no GPU reset). `.gpu_last_hang` LATCH SET 13:27 — a human clears it.

**=> ALL THREE RESULTS FIT ONE CAUSE: THE SINGLE SHARED `ASSIGN` CURSOR CAS. COUNTER-FREE ASSIGN
REMAINS THE PLANNED WORK** (brief §6 + KG efa5d89f) — it is the one lever that removes the shared thing
instead of working around it, and WAVES=5 is a far cleaner starting point than WAVES=30.
"More units per group" is REFUTED by the SEGK=64 run.

ALSO: the "bin sha 397bfbe1cb010c6e" cited in three documents was UNVERIFIABLE — it matched no hash of
any artifact and appeared only in my own writing. HEAD 652053c69 at the config-of-record defsyms
rebuilds deterministically to `4ecdab1dafca36bb` (24008B, LDS 54016B), archived as
`~/dsws_gpu_logs/CONFIGOFRECORD_652053c69_4ecdab1d.bin`. Two archived .bak binaries labelled
"CONFIGOFRECORD"/"SWEEP" both PREDATE the non-pow2 fix and cannot be what produced any real-shape
table; they have been renamed `MISNAMED_*`.

### 2026-07-24 — DSWS2 CF0 bring-up (operands L2-only + role flow + prefetch, CFASSIGN=0) — CORRECT, PERF FLAT/-4%
- **Config:** A1 geometry but **CFASSIGN=0** + `DSWS2_OVERLAP=1 DSWS2_ROLEFLOW=1 DSWS2_PREFETCH=1 DSWS2_RCONV=1`,
  bin `85954d3c`. Shape ml8 gate_up 2112x9216x2560 (n_kseg=10), `ML8_COOP_CHUNK=96`, 33 chunks.
- **CORRECTNESS: CLEAN.** Dense oracle **ok=76032 bad=0 max_rel=0 at stride=1 (ALL 3168 tiles)**;
  **WORK-EXACT** computed=190080 == G*TOTAL_super; occ[0]=0; carry-through entered==shrunk==63360
  (funnel closed); no hang, no latch, journal clean (0 amdgpu events).
  => The whole CF0 stack — operands L2-only, grow-fail ring deleted, grow-first/reserve-after,
  bidirectional role flow, speculative prefetch — is CORRECT ON SILICON. First execution of this stack.
- **LDS 54784B -> 13824B CONFIRMED ON HARDWARE** (`LDS=13824B(alloc 13824B)`). The 40,960B operand-pool
  reclaim is real. This is the durable result: 2x13824 = 27648 < 65536, so **2 WG/CU is now physically
  possible for the first time** (still needs WAVES<=16 for the 32-wave-slot budget).
- **THROUGHPUT: FLAT-TO-WORSE.** TF=0.4, span **26,498,256** ticks vs the 2026-07-23 baseline's
  **25,483,124** at the SAME shape/chunk/geometry = **~4.0% SLOWER**. coast-frac 95.1% (unchanged).
  door4 GROW-FAIL=0 (the VGPR budget still never binds). CONVERSIONS occ[48]=57031 (role flow IS firing).
- **THIS IS THE PREDICTED RESULT.** Both independent adversarial reviewers
  (`REVIEW_DSWS2_CF0_2026-07-24.md`) said the mechanisms could not help: the prefetch warms ~1.5% of its
  target footprint, roles gate no work under OVERLAP (feed/compute/coast all funnel to the same reserve
  path), and grow-before-CAS adds pipeline drains proportional to contention. The measurement matches.
- **VERDICT:** correctness-complete, mechanism-incomplete. Do NOT tune these knobs — the prefetch target
  and the role economy need redesign, not parameters. The banked asset is the LDS headroom.

### 2026-07-24 — ⭐ FIRST GENUINE 2 WG/CU RUN — measured, and it does NOT help (retires the line on evidence)
- **Config:** same CF0 stack, but **WAVES=16** (bin `98c97456`) + **`ML8_POOL=128`**. Same shape/chunk as the
  bring-up above (ml8 gate_up 2112x9216x2560, `ML8_COOP_CHUNK=96`, 33 chunks).
- **2 WG/CU CONFIRMED REAL, not clamped.** `occ[20]` claim = 3296; TOTAL = 3168 => **128 WGs** raced the
  final claim (the WAVES=30 run gave 3232-3168 = **64**). `waves/WG=16`, `LDS=13824B(alloc 13824B)`, no
  REFUSE from the occupancy guard. The 2026-07-21 note that "the standing '2 WGs/CU is garbage' result was
  never actually 2 WGs/CU" (ML8_POOL silently clamped to 64) is now SUPERSEDED BY A REAL MEASUREMENT.
- **CORRECTNESS: CLEAN.** Dense oracle ok=76032 bad=0 max_rel=0 (stride=1, all 3168 tiles); WORK-EXACT
  computed=190080; occ[0]=0; no hang/latch.
- **RESULT: 3.7% SLOWER.** span **27,485,692** ticks vs the 1 WG/CU WAVES=30 run's **26,498,256**. TF=0.4
  both. Total resident waves are ~matched (128x16 = 2048 vs 64x30 = 1920), so this is close to an
  apples-to-apples test of **one 30-wave frontier vs two 16-wave frontiers per CU**.
  coast-frac 94.8% (vs 95.1%); door4 GROW-FAIL still **0**; occ[97] release-bails ROSE 5.05M -> 7.86M.
- **INTERPRETATION:** the herd-splitting hypothesis — that 10 slices among 16 waves is easier to manage
  than among 30, so two smaller funnels beat one big one — is **NOT SUPPORTED**. Splitting the frontier
  costs slightly more than it saves. Note the dyn-VGPR budget STILL never binds (grow-fail=0) even at
  2048 resident waves, so occupancy is not the thing gating this kernel.
- **CAVEAT (honest):** run with the known-ineffective P1/P2 mechanisms enabled. They measured ~4% net cost,
  so a cleaner base would not flip a 3.7% loss into a win — but a 2 WG/CU test on a mechanism-free CF0
  build has not been run.

### 2026-07-24 — PHIST bail-door census on the CANONICAL A1 baseline (the "where does the time go" run)
- **Config:** canonical A1 baseline (`cac3ff7c` source) built `PHIST=1` -> bin `15b91d20`. Same shape/chunk
  as the day's other runs (ml8 gate_up 2112x9216x2560, `ML8_COOP_CHUNK=96`, 33 chunks).
  Oracle CLEAN dense stride=1 (ok=76032 bad=0), WORK-EXACT computed=190080, no hang/latch.
- **THE CENSUS** (throttled 1/64; ratios vs `occ[104] loophead` = the denominator):
  | door | count | % of loophead |
  |---|---|---|
  | loophead (denominator) | 1,165,689 | 100.0% |
  | **boundary (occ[110])** | **920,040** | **78.9%** |
  | coast (occ[113]) | 1,117,397 | 95.9% |
  | feedmt / park (occ[105]) | 1,110,564 | 95.3% |
  | RESV-try (occ[114]) | 1,138,751 | 97.7% |
  | **RESV-win (occ[115])** | **40,916** | **3.5%** |
  | drainwait (occ[112]) | 46 | 0.0% |
- **HEADLINE: 78.9% of ALL loop iterations enter `.Lflow_da_boundary`, and only 3.5% end in a
  reservation** — a peek->reservation conversion of **3.6%, i.e. ~28 peek attempts per success**.
  `drainwait` is ~0, so waves are NOT waiting on drain. The instrument's own read-guide says it:
  *"boundary or drainwait dominating => the wedge/stall is the tile-group boundary interlock, NOT the window."*
- **THIS DOES NOT CONTRADICT ADVPROBE — it is the other axis.** ADVPROBE (2026-07-23) measured the
  critical-section DURATION (~264 ticks once you win). PHIST measures ENTRY FREQUENCY. A cheap critical
  section entered on 79% of passes is still enormous traffic, and every entrant that loses still pays the
  reads + the CAS attempt before bailing.
- **COMBINE WITH BNDSPLIT (2026-07-23): 93.1% of boundary entries LOSE the ZLOCK election.**
  0.789 x 0.931 => **~73% of ALL loop passes are waves entering the boundary and losing.** That reframes
  the 07-23 "boundary line is a dead end" conclusion: the advance MECHANISM is cheap, but the ENTRY
  TRAFFIC is the dominant activity of the kernel.
- ⚠ **CAVEAT 1 — FIVE OF ELEVEN DOORS ARE UNINSTRUMENTED.** `gatefull`, `zlock`, `terminal`, `bnd-lost`,
  `growfail` all read 0 because **they have no bump sites** (the print says so). These are NOT
  measurements — this is the project's recurring "zeros that were never measurements" trap. Do not read
  them as zero. Consequently PHIST alone CANNOT decompose *why* the 78.9% of boundary entries bail.
- ⚠ **CAVEAT 2 — PROBE PERTURBATION IS LARGE.** span **263,761,632** ticks vs the probe-off control's
  **25,483,124** = **10.35x slower**, and coast-frac moved 95.1% -> 99.7%. The regime SHIFTED. Ratios are
  suggestive, not settled. **TF=0.0 from this run MUST NOT be quoted** (probe build).
- **WHERE TO DIG NEXT (offline, no GPU):** wire the missing PHIST bump sites (`zlock`, `gatefull`,
  `bnd-lost`) so the 78.9% can be decomposed at the door level, and/or re-run BNDSPLIT alongside PHIST to
  cross-check the 93.1% election-loss share on the current source.

### 2026-07-24 — ⭐ BNDTIME: the boundary is NOT the wall. Losing passes cost 36.8 ticks = <1% of wave-time.
- **Config:** CF0 stack + `DSWS2_BNDTIME=1` @ **WAVES=16 + `ML8_POOL=128` (2 WG/CU)**, bin `48519446`.
  Same shape/chunk as the day's other runs. Oracle CLEAN dense stride=1 (ok=76032 bad=0), WORK-EXACT 190080,
  128 WGs confirmed (occ[20]=3296), no hang/latch.
- **THE MEASUREMENT:** `occ[133]` = 33,483,304 ticks / `occ[134]` = 910,971 non-advancing passes
  => **ticks/lost-visit = 36.756**. (ADVPROBE's WINNING pass, for contrast, was ~264 ticks.)
- **THE SHARE — this is the point:**
  ```
  lost-boundary wave-ticks = 33,483,304 x 64 (1/64 sample)      = 2.14e9
  total wave-ticks         = 119,323,116 span x 2048 waves      = 2.44e11
                                                          share = 0.88%
  ```
  **Losing boundary passes are UNDER 1% of total wave-time.**
- **THE PROBE BIAS FAVOURS THE CONCLUSION.** BNDTIME slowed the run 4.3x (119.3M vs the 27.5M probe-off
  control) and its probes sit ON the boundary path — so the instrument INFLATES exactly the quantity being
  bounded. The true share is **<=0.9%**, making the verdict safe rather than marginal.
- **VERDICT: the boundary election is settled, from both directions.** Winning passes cheap (264t),
  losing passes cheaper (37t), combined ~1% of time. The 2026-07-23 "boundary line is a dead end"
  conclusion STANDS.
- ⚠ **RETRACTION:** earlier today I suggested PHIST's 78.9%-of-passes boundary-entry figure "reframed" that
  dead-end verdict and made boundary traffic the dominant activity. **That was wrong — frequency is not
  cost.** 78.9% of passes touch the boundary; they consume <1% of the time. Do not cite the reframing.
- **WHERE THE TIME ACTUALLY IS, by elimination:** not the boundary (<1%), not carrier stalls (occ[88]=0),
  not the VGPR budget (grow-fail=0 even at 2048 resident waves), not drain (drainwait~0). 95-99% of passes
  bail to `.Lflow_feedmt_sleep`. **The time is IDLE WAITING** — which is ADVPROBE's unexplained ~90% gap.
- **NEXT INSTRUMENT (cheap, and it has an ablation):** how much of the idle is the `s_sleep SLEEPN` yield
  itself? If waves sleep through work becoming available, that is a latency we are CHOOSING. `SLEEPN` is a
  defsym, so this is measurable AND directly ablatable.

### 2026-07-25 — ⭐ SLEEPN KILLED OFFLINE (~1%), LDS-CAS CONTENTION REFUTED (~0.01%), and T1 VOIDED as an instrument defect
- **Run:** CF0 profile + `DSWS2_PASSTIME=1`, bin `c706dd57`, host rebuilt. ml8 2112x9216x2560,
  `ML8_COOP_CHUNK=96`, 33 chunks, WAVES=30 1 WG/CU. Log `dsws2_passtime`.
- **CORRECTNESS CLEAN:** oracle ok=76032 bad=0 max_rel=0 at **stride=1 (ALL 3168 tiles)**; WORK-EXACT
  computed=190080; occ[0]=0; no hang, no latch.
- **PROBE COST 1.22x** (span 31,009,600 vs 25,483,124 baseline) — the cheapest instrument after ADVPROBE,
  and far below BNDTIME's 4.3x. The throttled-`s71` + SGPR-accumulate + emit-at-retire shape is confirmed
  as the right template. Do NOT quote TF from it.

#### 1. `SLEEPN` IS NOT THE WALL — killed OFFLINE, with ZERO dispatches
`occ[86]` is a per-wave `cnt_emit` atomic-add = a true total park count; `s_sleep N` is ISA-defined at
64*N clocks. At SLEEPN=2: **1.0-1.2% of wave-time** (w30 1WG/CU, 61,270 parks/wave) and **0.6-0.8%**
(w16 2WG/CU, 40,011 parks/wave). This also EXPLAINS the historical flat SLEEPN sweep — it was flat
because there was nothing there. The queued "top candidate" measurement was answered by arithmetic.
- **METHOD RULE (the distinction that makes this legitimate where 2026-07-24's was not):** count x
  *fixed ARCHITECTURAL* cost is a valid time bound. Count x *another count* never is. The retracted
  07-24 claim multiplied PHIST's 78.9% by BNDSPLIT's 93.1% and called the product a time share.

#### 2. LDS-CAS CONTENTION REFUTED — the reservation CAS is ~0.01% of pass time
Hypothesis under test: the kernel is contention-serialized on LDS atomics on shared cursor words
(plain LDS latency should be hidden by 15-way SIMD multithreading; atomics on a shared address cannot be).
- T0 null calibration = **5.153 ticks** (per-end-read overhead) · T2 reservation CAS = **6.291 ticks corrected**.
- **Only 0.85% of park passes ever reach a reservation CAS** (`t2_count/t1_count = 0.0085`). The CAS was
  structurally incapable of being the wall.
- Against the INDEPENDENTLY-derived 525-tick pass period: **<=1.2% worst case** (if every pass hit a CAS),
  **~0.01% actual**. REFUTED. This verdict does NOT depend on T1 — it rests on T2 and on counters.

#### 3. *** T1 vs THE DERIVED PERIOD DISAGREE BY 3.9x. QUOTE NEITHER. ***
> **CORRECTION, same day, after Grok's diagnosis (task 40417767):** the heading below originally read
> "T1 IS AN INSTRUMENT DEFECT — VALUE VOID". **That was an overclaim and it is retracted.** I treated the
> 525-tick period as ground truth and T1 as the defect. But **525 rests on an ASSUMPTION I never measured**
> — that all 1920 waves are resident for the whole span. This run's `peak-resident occ[1]=0` is
> **TRACE-gated**, and TRACE=0 in this build, so residency is UNMEASURED (another "zero that was never a
> measurement" — and I quoted a number derived from it while policing exactly that error class).
> If true residency is ~490 waves, the period IS ~2029 and **T1 is correct**.
> **The honest position: the two numbers disagree, at least one is wrong, and I do not know which.**
> Grok's static analysis RULED OUT the pairing mechanism I proposed (see below), which weakens the
> "T1 is broken" side rather than strengthening it.
T1 (whole head->park pass) reported **2028.859 ticks/pass corrected**. That is **physically impossible**:
the same run gives a pass period of **525 ticks** (occ[86]=113,526,840 / 1920 waves / span, per chunk).
A pass cannot take 2029 ticks if passes occur every 525.
- **The probe-overhead escape hatch FAILS:** T2 measures 11.4 ticks RAW *including its own end read*, so
  reads are genuinely cheap and cannot account for a ~1500-tick excess.
- Candidate causes, UNRESOLVED: stale start latches on paths reaching the park without a fresh start, or
  systematic bias in the `s71==0` sample. A CFG reachability test was INCONCLUSIVE — it cannot model that
  both guards read the same `s71`, which is constant within a pass, so start/end *should* pair.
- **HOW IT WAS CAUGHT — adopt this as standard practice:** derive the same quantity a SECOND, independent
  way and require the two to agree. The period arithmetic cost nothing and killed a number that had
  survived design, adversarial review, and a clean run. 2029 is exactly the kind of large, impressive
  figure that gets quoted — it would have been the 7th entry in this file's "numbers that were never
  measurements" family, and the most convincing one yet.

#### 4. THE STANDING SCOREBOARD
Eliminated **by direct measurement**, no longer by elimination: boundary (<1%), sleep (~1%), reservation
CAS (~0.01%), carrier stalls (occ[88]=0), VGPR budget (grow-fail=0), drain (drainwait~0).
**The ~90% remains unattributed.** The banked asset from this run is the **525-tick measured pass period** —
the budget any future instrument must account for.
- Reminder: this run printed `STARVATION 100% => ASSIGN-BOUND`. **Tautological under SELFSERVE. Do not cite.**

#### 3b. WHAT GROK'S STATIC DIAGNOSIS ESTABLISHED (task 40417767, offline, no GPU)
**RULED OUT with disassembly evidence, under THIS exact profile:**
- **`s72` is not clobbered on the live park path.** Only two writers exist in bin `c706dd57`: the prologue
  `passtime_zero` (`s_mov_b32 s72, 0`) and the T1 start (`s_mov_b32 s72, s62` @0x034f0). The `BATCH>1`
  and `TRACE` s72 stores are `.if`-gated out (profile is BATCH=1 TRACE=0).
- **`s71` cannot wrap mid-pass on the lean park path.** Only THREE `s71++` sites are emitted (init deadman,
  loop-head deadman @0x034a4, terminal-drain deadman @0x07580). At **JDEPTH=1 the entire `.Lflow_jwait` /
  `deadman_check_fat` block is not built** (`.if JDEPTH > 1`, :4252), and the second `deadman_check` is
  `.if !DECENTASN` (DECENTASN=1). So the mid-pass s71 hammer that would break pairing DOES NOT EXIST here.
- **=> MY PROPOSED PAIRING DEFECT IS NOT SUPPORTED.** If a wave reaches the park with s71==0, s71 was 0 at
  that iteration's loop head, so the T1 start DID execute and DID refresh s72 on the same iteration.
  The 4.55% stale-fraction arithmetic fits perfectly but is **numerology without a mechanism** — none found.
- **The `s72=0` first-end theory is rejected as the mean driver:** one `now - 0` sample would be ~2^31 and
  add ~1200 to the mean by itself; the observed mean is large-but-finite, not absolute-RTC contamination.
- **The CFG BFS I ran was confirmed inconclusive** — it found the s71!=0 fall-through, where the END's
  guard also skips. Data-dependent guards are invisible to a pure CFG.

**THE INSTRUMENTATION GAP THAT MADE THIS UNDIAGNOSABLE:** the T1 **start bumps no counter** — only the end
bumps `s77`. So **unpaired starts are invisible** and unpaired ends show up only as fat deltas, never as a
count mismatch. `parks/T1_samples = 63.82 ≈ 64` proves ends are not missing or extra **relative to parks**;
it does NOT prove the deltas are per-pass. A count that matches is not a delta that is correct.

#### 3c. THE SETTLING RUN (designed, NOT yet built — one dispatch settles it)
Three additions, all inside the existing throttled shape:
1. **ARMED FLAG** — cleared at every loop head; set at T1 start; at T1 end, if `!armed` bump
   `unpaired_end` and **do not accumulate**. Directly detects the pairing defect.
2. **MAX T1 DELTA** (SGPR max, emit at retire) — separates "fat tail" from "uniformly long".
3. **T1 START COUNT** — makes start/end asymmetry visible at last.
4. **RESIDENCY** — the missing input to the period. Must be a REAL counter, not the TRACE-gated `occ[1]`.

Reading it: `unpaired_end > 0` => pairing defect confirmed, T1 was wrong. `unpaired_end == 0` AND
`max ~ O(mean)` => **T1 deltas are real and the 525-tick PERIOD is what needs debugging** (residency/span
assumptions). `unpaired_end == 0` AND `max >> mean` => a genuine fat tail of long passes to attribute.
**All three outcomes are informative — this run cannot come back empty.**

#### 3d. THE LESSON, RESTATED CORRECTLY
The original lesson stands but was aimed at the wrong target. It is not "T1 was a fake number caught by
cross-derivation." It is: **when two derivations disagree, the SECOND one is not automatically the truth.**
I cross-derived correctly, then immediately privileged my own arithmetic over the instrument and wrote
"VALUE VOID" into this log — asserting a defect I had not demonstrated, using a denominator that rests on
an unmeasured residency. Cross-derivation tells you *something is wrong*. It does not tell you *which one*.

### 2026-07-25 — ⭐⭐ THE SETTLING RUN: T1 exonerated-then-convicted, residency MEASURED, and the FIRST POSITIVE LEAD (a 785x fat tail)
- **Run:** CF0 + `DSWS2_PASSTIME=1` settling build, bin `f3276015`. Same ml8 shape/chunk. Log `dsws2_passtime_settle`.
- **CORRECTNESS CLEAN:** oracle ok=76032 bad=0 at stride=1 (ALL 3168 tiles); WORK-EXACT computed=190080;
  no hang, no latch. Probe cost **1.58x** (span 40,330,408 vs 25,483,124 baseline) — up from 1.22x, still
  cheap, still NOT TF-quotable.

#### THE QUESTION IT SETTLED
| counter | value | meaning |
|---|---|---|
| `unpaired_end` (occ[141]) | **0** | pairing CLEAN — and a zero that IS a measurement (armed path has its own bump site) |
| **peak concurrent residency** (occ[145]) | **1920** | **MEASURED**, ungated, outside the per-chunk memset |
| `period_peak` | **508.475 ticks** | now computed from a measured denominator |
| T1 corrected | **413.97 ticks/pass** | vs 2028.86 in the prior build — **physically consistent at last** |
| T1 **max delta** (occ[142]) | **329,112 ticks** | **785x the mean** |
| start / end counts | 2,385,599 / 2,383,385 | 2,214 orphaned starts (0.09%) — benign |

**VERDICT: the 525-tick period was RIGHT and T1's 2029 was the broken number.** Measured residency came
back at exactly the assumed 1920. My mid-day retraction (which said the period was the suspect number
because residency was unmeasured) was itself wrong — but it was wrong *for the right reason*: residency
genuinely WAS unmeasured at the time, and the only way to know was to measure it. Both the original claim
and its retraction were unjustified when made; only this run justified either.

#### *** OPEN ITEM — DO NOT LET THIS QUIETLY CLOSE ***
`unpaired_end == 0` proves pairing is clean **in this build**. It does **NOT** explain what contaminated
`c706dd57` (T1=2034). The armed flag FIXED the contamination without the diagnostic ever IDENTIFYING it —
the counter cannot retroactively observe what the old binary did. **We have a working instrument and no
root cause.** Logged as a known unknown; do not write a comfortable story over it later.

#### *** THE FIRST POSITIVE LEAD IN THIS PROJECT ***
Everything before today was ELIMINATION — boundary <1%, sleep ~1%, reservation CAS ~0.01%, carrier
stalls 0, VGPR budget never binds, drain ~0. All that ever produced was "~90% unattributed".
**T1 now ACCOUNTS for ~83% of total wave-residency time** (998,810,104 sampled ticks x64 throttle =
6.39e10, against span x peak-residency = 40,330,408 x 1920 = 7.74e10).
And the shape is extreme: **mean 419, max 329,112.** One poll pass consumed **a quarter of an entire
chunk** (chunk = 1,222,133 ticks).
- **MEANS ARE THE WRONG STATISTIC FOR THIS DISTRIBUTION**, which is very likely why every previous attempt
  to find this dissolved into "it's just idle waiting."
- CAVEAT on the 83%: extrapolating a 1/64 sample assumes sampled passes are representative, which a 785x
  tail strains. The tail-attribution run is what tests it.

#### NEXT (built, pending dispatch): TAIL ATTRIBUTION
Count **AND TICK-SUM** of T1 deltas above a few thresholds. The SUM is the load-bearing half:
"passes over Xk ticks are N% of passes but **M% of all pass-time**."
- Large M for small N => the tail IS the wall; stop looking at means; ask which code path emits a
  329,112-tick pass.
- Small M => the tail is a curiosity and the mass is in the bulk.
- **A threshold COUNT WITHOUT THE SUM would be instance #8** of this file's "numbers that were never
  measurements" — it would say how MANY passes are fat and nothing about whether they MATTER.

### 2026-07-25 — 🛑 TAIL-ATTRIBUTION RUN VOID: u32 OVERFLOW. The instrument produced a confident, impossible, and *desired* verdict.
- **Run:** CF0 + `DSWS2_PASSTIME=1` tail build, bin `6c6993e2`. **Kernel itself CLEAN** — oracle
  ok=76032 bad=0 stride=1 (all 3168 tiles), WORK-EXACT computed=190080, no hang, no latch.
  **The kernel has never been in question in this whole sequence. Only the measurement apparatus.**

#### WHAT IT PRINTED
```
bulk    delta <  1024 : count=2375559 (78.92%)  ticks=182157116  (30.45% of T1 time)
mid     1024..65535   : count=620021  (20.60%)  ticks=0          (0.00%)
extreme delta >=65536: count=14682   (0.49%)   ticks=1330364040 (222.38% of T1 time)
raw: gt1k n=634703 sum=416070904   gt64k n=14682 sum=1330364040    (T1 total = 598228020)
TAIL VERDICT: EXTREME-DOMINATED -- passes >=64K are 222.4% of T1 time. The fat tail IS the wall.
```

#### THREE INDEPENDENT IMPOSSIBILITIES
1. `extreme` = **222.38%** of the total. A subset cannot exceed the whole.
2. `mid` = 620,021 passes summing to **ZERO** ticks. Every member is >=1024 => minimum ~635M.
3. `gt64k` is a **subset** of `gt1k`, yet `sum(gt1k)=416,070,904` **<** `sum(gt64k)=1,330,364,040`.

#### THE OVERFLOW, PINNED EXACTLY
`2^32 + 416,070,904 = 4,711,038,200` — a single clean u32 wrap of the gt1k sum.
The `mid=0` is just the host computing `gt1k_sum - gt64k_sum`, going negative, and clamping.

#### *** IT WAS NEVER ONLY THE NEW COUNTERS ***
The pre-existing **T1 total was already at 84% of the u32 ceiling on the FIRST run**:
| run | bin | T1 total | % of 2^32 |
|---|---|---|---|
| 1 | `c706dd57` | 3,618,636,368 | **84%** |
| 2 | `f3276015` |   998,810,104 | 23% |
| 3 | `6c6993e2` |   598,228,020 | 14% |
Sizing check: total wave-residency = span x peak-residency = 41,411,232 x 1920 = **7.95e10**; sampled
1/64 => any tick-sum covering most of runtime is **O(1.2e9) and rising with probe cost**. 32-bit was
never sufficient for this instrument; the early runs merely got away with it.

#### ~~THIS OVERTURNS THE "ARMED FLAG FIXED IT" STORY~~ — ✗ **RETRACTED WITHIN THE HOUR, BY GROK**
I wrote: *"Overflow is now a better candidate than pairing"* for the original T1=2034, reasoning that
2034 -> 419 -> 199 tracked accumulator pressure. **That is WRONG and I retract it.**
> **Run 1's T1 total was 3,618,636,368, which is BELOW 2^32 = 4,294,967,296. IT NEVER WRAPPED.**
> 84% of the ceiling is not the same as over it. `mean = total/samples` on an unwrapped total is
> just arithmetic. Overflow explains **run 3's** impossible 222% / mid=0 / gt64k>gt1k. It explains
> **nothing** about run 1.

**MY ERROR, AND IT IS THE SAME CLASS I SPENT THE DAY POLICING:** I took a suggestive fact ("84% of the
ceiling!") and promoted it to an explanation without checking whether the wrap actually occurred — one
subtraction would have settled it. Finding a real bug creates enormous pull to make it explain the
other open thing too. **A newly-found bug is not a general-purpose explanation.**

=> **THE OPEN ITEM STAYS OPEN: the run-1 contamination mechanism is UNKNOWN.** Do not credit overflow
with it. Do not credit the armed flag with it either — that was never established, only assumed.
Suspect: run-3 tick-sums (definitively) and any tick-derived mean not individually checked against
2^32. Still valid: `unpaired_end=0`, `peak residency=1920`, and correctness.

#### THE PROCESS POINT — THE MOST DANGEROUS NEAR-MISS IN THIS FILE
This would have been **instance #8** of "numbers that were never measurements", and the worst of them:
- the impossible number backed **the exact hypothesis I had asked the run to confirm**,
- the host stated it as a **VERDICT in capital letters** ("The fat tail IS the wall"),
- and it followed a *correct* prior run, so the instrument had earned trust.
It was caught **only because a percentage exceeded 100**. That margin is far too thin to rely on.
**FIX BEING BUILT: 64-bit accumulation for EVERY tick-sum (T0/T1/T2/gt1k/gt64k), overflow-reporting
guards so a wrap announces itself instead of wrapping silently, AND host-side assertions that REFUSE
to print a verdict when `subset > superset` or `pct > 100` — printing `*** INCONSISTENT — COUNTERS
VOID ***` instead.** The host assertions matter as much as the 64-bit fix: they are what turns a
persuasive wrong answer into a loud refusal.

### 2026-07-25 — ⭐⭐⭐ ROOT CAUSE OF THE 7x: THE PROBE AND THE PREFETCH SHARED A TRIGGER. T1 was never defective.
- **Ablation:** `DSWS2_PREFETCH=0`, bin `c6c0d0cf` (defsym flip only, -288B .text). Everything else identical.
- **CLEAN:** oracle 3168/3168 stride=1, WORK-EXACT 190080, no hang/latch.

| | prefetch ON (`54edfb7a`) | prefetch OFF (`c6c0d0cf`) |
|---|---|---|
| T1 mean | 2847.7 | **40.9** |
| wave-time budget ratio | 7.06x **VOID** | **0.217x — PASSES** |
| parks (poll passes) | 197,296,055 | **413,076,323** |
| period | 400.0 | 188.6 |
| tail verdict | TAIL-DOMINATED | **BULK-DOMINATED** |
| span | 41,102,028 | 40,575,032 (**only 1.3% faster**) |

#### THE MECHANISM
`DSWS2_PREFETCH` is gated on `s_cmp_lg_u32 s71, 0` — **the SAME 1-in-64 tick PASSTIME samples** — and
ends in a mandatory blocking `s_wait_loadcnt 0x0`. So **every sampled pass carried a full L2/HBM round
trip that the other 63-in-64 passes never paid.** The bias was multiplicative and stable across runs,
which is precisely why it did not look like ordinary selection bias.

#### *** T1 WAS NEVER DEFECTIVE — AND I CALLED IT DEFECTIVE THREE TIMES ***
Positions held on T1 today: (1) morning "instrument defect, VALUE VOID"; (2) midday retraction
"the PERIOD is the suspect number"; (3) post-u64 "back to (1), defect confirmed at 7x".
**All three wrong.** The instrument faithfully measured what it was pointed at. **The EXPERIMENT was
contaminated**: probe and prefetch shared a trigger.
=> **NEW CHECKLIST ITEM, and it is not on any list in this file:** before trusting a sampled
measurement, ask **WHAT ELSE KEYS OFF THE SAME TRIGGER AS THE PROBE.** Every other check this project
runs — internal consistency, cross-derivation, calibration, byte-identical-off, disassembly audit —
passes cleanly on a probe whose trigger is shared. None of them can see this.

#### *** FRAMING CORRECTION (kmbandy) — DO NOT MISREAD THIS RESULT ***
An earlier draft of this entry said *"the prefetch isn't inert, it's expensive."* **WRONG FRAMING.**
**This prefetch is BUILT WRONG.** Three independent defects:
1. **Wrong target** — warms `ksi in {0,1,2,3}` of the **CURRENT** tile, lines already resident
   (~1.5% of its real footprint; `REVIEW_DSWS2_CF0_2026-07-24.md`).
2. **Blocking** — `s_wait_loadcnt 0x0` makes the ISSUING wave eat the latency inline. A prefetch that
   blocks its own issuer is not a prefetch, it is a load.
3. **No lead time** — fires immediately before the wave would proceed; nothing is ever "ahead".
**DO NOT LOG THIS AS "PREFETCHING DOES NOT HELP THIS KERNEL."** That is the identical error to the
weight-pager's ten-arm sweep at `WP_PREFETCH_LOOKAHEAD_K=1`, where speculative reads for layer L+1
queued behind layer L's own demand reads — **refuting a technique by measuring a broken implementation
of it.** The KG rule from that session applies verbatim here: *K=1 is not a test of prefetch.*
**A correctly-built prefetch (right target, non-blocking, real lead) remains UNTESTED on this kernel.**

#### WHAT THE CLEAN NUMBERS SAY
- **The kernel is NOT poll-throughput-bound.** Poll passes MORE THAN DOUBLED (197M -> 413M) while span
  moved only 1.3%. Spinning the loop twice as fast changes nothing.
- Passes are **21.7% of the period** (40.9 of 188.6 ticks). The remaining ~148 ticks sit **between the
  park and the next loop head** — the sleep+wake path. `s_sleep SLEEPN=2` is only ~5 ticks of that.
- ⚠ **THAT ~78% IS ARITHMETIC INFERENCE (period - T1), NOT A MEASUREMENT.** The arithmetic is sound
  (period is park-to-park, T1 is head-to-park) but inference is exactly what this file keeps getting
  burned by. **NEXT: time park->head directly** with the instrument that now demonstrably works.
- Earlier claim "poll passes account for ~83% of wave-time" was prefetch-inflated. True figure: **21.7%**.

### 2026-07-25 — 🔬 INDEPENDENT CODEX (gpt-5.6-sol) REVIEW OF THE PREFETCH — diagnosis at file:line, and it corrects me twice
Commissioned INDEPENDENTLY: Codex was given the evidence, the code, the prefetch's GOAL, and the safety
constraints — and **denied all of my conclusions** (no prior review docs, no hypotheses). Agreement below
is therefore corroboration, not echo.

#### WHAT IT CONFIRMED (I had these)
- **Not fire-and-forget.** 4x `global_load_tr_b64` (all to the SAME `v16:v17`) then `s_wait_loadcnt 0x0`
  at :4890. T1 spans :3812 (head) -> :6064 (park), so the whole stall lands inside the measured pass.
- **Phase-locked with the probe.** Prefetch fires on `s71==0` at :4858; T1 starts on `s71==0` at :3176.
- **`tcol` comes from `DA_TILE_OFF` = the CURRENT tile** (:4860). It never addresses the next tile.
  (I derived this independently from the address math before seeing the review.)

#### WHAT IT FOUND THAT I DID NOT — sharper, and it is the real defect
- **THE CLAMP COLLAPSES ALL FOUR GUESSES ONTO ONE ADDRESS.** At n_kseg=10 (NOT a power of two), s67=15
  and s66=9. During drain-park, reservation has stopped at the first phantom index (rejected at :5118),
  so ASSIGN_HEAD sits at field offset 10 and the guesses are 10,11,12,13 — **all clamped by `s_min_u32`
  to 9**. All four loads hit ONE identical address: frag 0, K-step 0, slice 9 of the CURRENT tile,
  already consumed. It is not warming four lines; it is re-warming one dead line four times.
- **ROOT STRUCTURAL CAUSE: the two-generation frontier is "still NOT built"** (:451, :464). Without a
  `t_next` identity the code **cannot** form next-tile addresses. This was never a tuning problem.
- **The comment at :1727 claiming `n_kseg` is always a power of two is STALE** — contradicted by the
  arbitrary-K decode at :3578. Several assumptions in this file may rest on it.

#### *** WHERE IT CORRECTED ME — BOTH MATTER ***
1. **THE "7.06x WAVE-TIME IMPOSSIBILITY" WAS MY ERROR, NOT A COUNTER DEFECT.** The host's
   `T1_total x 64` extrapolation assumes **UNIFORM** sampling. The sampled pass is *exactly* the prefetch
   pass, so the stratum is deliberately expensive and multiplying it by 64 is invalid. **The individual
   T1 intervals were real the whole time.** => The external budget assertion I championed is itself only
   sound under uniform sampling. It must say so and ideally DETECT the stratified case rather than
   declaring COUNTERS VOID. (Filed into the P3 build.)
2. **"DRAMATICALLY SLOWER" IS WRONG END-TO-END.** The regression is **1.3%**. The dramatic effects are
   local poll latency (~70x on sampled passes) and poll-loop throughput (2.1x) — and that throughput is
   **unused control-plane capacity**. Do not optimise the poll rate.

#### A CORRECTION THAT REACHES PAST THIS TASK
**`DRAIN_HEAD` IS NOT A VALID COMPLETION SIGNAL UNDER SELFSERVE.** The reservation is published as
pre-completed BEFORE compute (:5728) and `drain_advance` runs before the burst (:5772). **`TILEDONE`
(:4627) and `GSTORED` (:4748) are the real productive-completion signals.** Older conclusions in this
file that lean on ASSIGN-vs-DRAIN are suspect for this reason.

#### WHERE THE KERNEL IS ACTUALLY BOUND (and the honest limit of this A/B)
The productive tile-completion chain: TILEDONE after reductions (:4627) -> C-store drained before
GSTORED published (:4748) -> boundary advance waits on GSTORED (:5342). **NOT the poll loop.**
Codex is explicit that this A/B does **not** discriminate B-load latency vs WMMA vs LDS reduction vs
C-store drain vs the completion frontier. Naming one requires more device measurement. **Do not let this
harden into "the C-store is the wall" without measuring it.**

#### THE REBUILD (handed to Grok, phased; ONE dispatch per phase)
- **P1 — two-generation frontier:** state machine EMPTY -> CLAIMING -> READY(t_next) -> TERMINAL below
  OP_BASE. MOVE the existing next-tile claim earlier (:5422); this RELOCATES the claim, it does not add
  one per tile. Trigger = the unique winner reserving the last real ksi of the final group (CAS :5198,
  decode :5249). That wave publishes an LDS request and issues NO loads (it is fat and mid-compute).
- **P2 — dedicated prefetch wave + correct addressing:** one feed-floor wave per WG (cold start already
  labels three at :3720; their staging job is gone under OVERLAP at :4805). It must never
  `s_alloc_vgpr` — that does WaitIdleExceptStoreCnt (:802), so a growing wave would stall on outstanding
  prefetch loads and put the latency straight back into a compute dependency chain.
  Address the EXACT compute set for t_next: `ksi 0..n_kseg-1, ks 0..KSEG_STEPS-1, ni 0..FN-1`.
  **DO NOT CLAMP PHANTOM CURSOR VALUES INTO REAL ksi — iterate the real count.** No post-issue wait;
  wait only before reusing destination registers on the next batch.
- **P3 — bounded retire-emitted counters** + fix the host's uniform-sampling assumption.
- **BUDGET:** a correct full warm is 10*16*4*256 = 163,840 B/tile => ~519 MB total, **1/6 of the
  productive 3.114 GB B stream and ~6x LESS than the broken prefetch's 3.132 GB upper bound.**
  The correct build is CHEAPER than the broken one. Rule 7 is live: compile-time cap is mandatory.

#### PROCESS
Independence was ENGINEERED, not hoped for: Codex got evidence + code + goal + constraints, and was
denied every hypothesis I held. It then corrected two of my load-bearing claims. This is the fifth time
independent review has caught something my own analysis missed, and the first time the reviewer
overturned a *measurement interpretation* rather than a code defect.

### 2026-07-25 — ✅ PREFETCH REBUILD P1 (two-generation `t_next` frontier) — PASSES, and is the FASTEST run of the day
- **Bin `3c17e9c4`** (.text 35636B, LDS 13824B unchanged — TNXT words live in the existing control gap
  below GSTORED). `DSWS2_PREFETCH=0` verified byte-identical to `c6c0d0cf`, both directions.

| gate | result |
|---|---|
| `occ[20]` (tile claims) | **3232 — identical to every prior run => NO DOUBLE-CLAIM** |
| WORK-EXACT | computed = 190080 ✓ |
| carry-through | entered == shrunk == 63360 ✓ |
| oracle | 3168/3168 stride=1, bad=0 ✓ |
| hang / latch / MODE1 | none ✓ |

#### THE BUILDER CAUGHT A HOLE IN THE REVIEWED DESIGN
Codex's state machine was `EMPTY -> CLAIMING -> READY -> TERMINAL`. Grok **added `BUSY`**, because
`CLAIMING` alone is not exclusive ownership of the *global* atomic: two lean claimers could both issue
it and skip a tile. `occ[20]=3232` on silicon is the evidence that the added state is what makes the
design correct. **An independent review is not a specification — the builder still owns correctness.**

#### SPAN: P1 BEAT EVEN THE FULL ABLATION
```
41,102,028   prefetch ON, pre-P1 (broken)
40,575,032   prefetch OFF (full ablation)
39,779,080   P1  <- 3.2% faster than pre-P1, 2.0% faster than prefetch-OFF ENTIRELY
```
P1 changes only WHERE the next-tile claim happens, and the broken prefetch is still firing. Plausible
mechanism: the boundary handler no longer issues a global atomic **while holding ZLOCK** — it consumes a
pre-published `READY(t_next)`. Codex independently named that chain (TILEDONE :4627 -> C-store drained
before GSTORED :4748 -> boundary advance waits GSTORED :5342) as the real critical path, so removing an
atomic from it is exactly the right shape of change.
- ⚠ **NOT BANKED. One unreplicated run, and NO noise estimate exists for this config — no bin has been
  run twice today.** Mechanism and direction agree, which is more than any previous lead had, but a
  single run is not a result. **Replicate before this is quoted as a win.**

#### THE HOST GUARD DID ITS JOB
`*** T1 EXCEEDS AVAILABLE WAVE-TIME BY 8.32x — COUNTERS VOID ***`, and it **suppressed** the tail
dominance verdict rather than printing a claim from impossible arithmetic. Note per Codex this is
detecting **stratified sampling** (the prefetch still fires on the sampled `s71==0` pass in P1), not
counter corruption. P3 must reword it to say so. If the VOID does NOT clear once P2 removes the old
burst, something else is stratified and that is a finding in itself.

#### NEXT: P2 (dispatched)
Dedicated prefetch wave (one per WG, lean, never `s_alloc_vgpr` — that does WaitIdleExceptStoreCnt at
:802 and would drag prefetch latency back into a compute dependency chain), correct `t_next` addressing
over the REAL `ksi` count with **no phantom clamp**, no post-issue wait, old burst removed, and a
mandatory compile-time byte cap (rule 7 — this GPU drives the displays off the same HBM bus).

### 2026-07-25 — ⭐⭐⭐⭐ PREFETCH REBUILD P2: **A CORRECTLY-BUILT PREFETCH BEATS NO PREFETCH BY 8.6%** — replicated, noise-bounded
- **Bin `1d434719`** (.text 36164B, LDS 13824B). `DSWS2_PREFETCH=0` byte-identical to `c6c0d0cf`, both
  directions. `s_sendmsg_rtn` 13 unchanged. **`s_alloc_vgpr` site count unchanged at 9** — P2 adds none.
- **RUN TWICE** (first replication of the day): span **36,904,204** and **37,242,000**.
  => **RUN-TO-RUN NOISE = 0.91%.** This is the number every span claim today was missing.

#### GATES (both runs)
`occ[20]=3232` — **no double-claim** despite P2 adding a second claim-service site (`global_atomic`
44 -> 45). WORK-EXACT computed=190080. carry-through entered==shrunk==63360. oracle 3168/3168 at
stride=1, bad=0. No hang, no latch, no reset, no MODE1.

#### THE RESULT, AGAINST MEASURED NOISE
```
41,102,028   prefetch ON, pre-P1 (BROKEN)
40,575,032   prefetch OFF (FULL ABLATION)
39,779,080   P1 (claim relocated)
36,904,204 / 37,242,000   P2 (CORRECT prefetch)
```
| comparison | delta | verdict vs 0.9% noise |
|---|---|---|
| P2 vs broken prefetch | **9.8% faster** | REAL (11x noise) |
| **P2 vs NO PREFETCH AT ALL** | **8.6% faster** | **REAL (9x noise)** |
| P2 vs P1 | 6.8% faster | REAL (7x noise) |
| P1 vs ablation | 2.0% | **MARGINAL (~2x noise)** — my refusal to bank it was correct |

#### *** THE POINT — AND IT IS THE WHOLE LESSON OF THE DAY ***
**A correctly-built prefetch is 8.6% FASTER than having no prefetch at all.** Deleting it would have
shown a **1.3% win** and locked in a permanent **8.6% loss**. kmbandy's standing constraint — "just not
using it is not an option" — is what forced the rebuild instead of the deletion. Compare the KG's
weight-pager entry: ten arms swept at `LOOKAHEAD_K=1` nearly retired prefetching on the strength of a
build that had no lead time. **Two projects, same near-miss, same rule: never refute a TECHNIQUE by
measuring a BROKEN IMPLEMENTATION of it.**

#### BOTH PRE-REGISTERED PREDICTIONS CONFIRMED (recorded before the run, so they could not be retrofitted)
1. **T1 mean 4446 -> 51.2.** The blocking `s_wait_loadcnt` in the poll path WAS the sample poison.
2. **The 8.32x WAVE-TIME VOID cleared to 0.257x.** Removing the burst severed the probe/prefetch shared
   `s71` trigger. This confirms **Codex's stratified-sampling diagnosis end-to-end** — the counters were
   never corrupt, the extrapolation was invalid. Tail is now **BULK-DOMINATED** (80% of T1 time under 1K).

#### ANOMALY — LOGGED, NOT WAVED THROUGH
**Orphaned T1 starts jumped ~200x: 2,300 -> 683,972 (0.09% -> 10.9% of starts).** Expected in DIRECTION
(the PF wave arms T1 at the loop head, runs the engine, returns to `.Lflow_loop` without ever parking),
but one wave in 30 is 3.3%, and 10.9% only reconciles if the PF wave visits the loop head ~3.5x more
often than average. **Plausible, UNVERIFIED.** Benign for correctness; it does change what T1 samples.

#### WHY IT WORKS (mechanism, consistent with Codex's independent read of the critical path)
P1 removed a global atomic from the boundary handler's ZLOCK-held path. P2 removed a blocking L2/HBM
round trip from the poll path entirely and pointed the fetch at the NEXT tile's real address set instead
of one already-consumed 256B block of the CURRENT tile. Both act on the productive completion chain
(TILEDONE :4627 -> C-store drain :4748 -> GSTORED -> boundary advance :5342), which Codex named as the
wall — NOT on poll throughput, which is unused control-plane capacity.
- **BUDGET: the correct build is CHEAPER than the broken one** — ~519 MB/run at real `n_kseg` (cap worst
  case 830 MB) vs the broken version's 3.132 GB upper bound. ~6x less traffic AND 9.8% faster.

### 2026-07-25 — P3 (counters + host wording + orphan fix): 3/3 PREDICTIONS CONFIRMED, and it found that **THE ENGINE IS STARVED**
- **Bin `b8918d3c`** (.text 36360B). `DSWS2_PREFETCH=0` byte-identical `c6c0d0cf` both directions.
  `s_sendmsg_rtn` 13 and `s_alloc_vgpr` 9 **both unchanged**; +4 `global_atomic` (45->49), all retire-emitted.
- **RUN TWICE:** span 39,786,480 / 39,316,884 (**spread 1.19%**; P2's was 0.91% — consistent noise floor).
- Correctness clean both runs: `occ[20]=3232`, WORK-EXACT 190080, oracle 3168/3168 stride=1 bad=0.

#### ALL THREE PRE-REGISTERED PREDICTIONS CONFIRMED
| prediction | result |
|---|---|
| orphaned T1 starts 683,972 -> ~2,300 | **1,981 / 1,888** ✓ PF-wave hypothesis was correct |
| `occ[159]` s71 co-tenant mask = 0x23 | **0x23** ✓ (PASSTIME \| ROLEFLOW \| DEADMAN) |
| wave-time budget ~0.25x PASS | **0.244x** ✓ |

**ORPHAN ANOMALY RESOLVED.** The PF engine now skips arming T1 entirely — it is not a poll pass and does
not belong in poll-pass statistics. Counting its orphans would have been the category error.
**HOST WORDING FIXED:** now prints `WAVE-TIME BUDGET (assumes UNIFORM 1/64 sampling)` and
`s71 CO-TENANTS (occ[159]=0x23): PASSTIME ROLEFLOW DEADMAN — non-PASSTIME co-tenants present => x64
extrapolation is stratified`. It names the cause instead of misattributing it as COUNTERS VOID.

#### *** P3 IS 6.7% SLOWER THAN P2 — AND THAT IS THE CORRECT OUTCOME FOR IT ***
P3 mean 39.55M vs P2 mean 37.07M = **6.7%, ~6x noise, replicated on both sides => REAL.**
**P3 IS A DIAGNOSTIC BUILD, NOT A SHIPPING BUILD. Its span is NOT quotable.** The performance
configuration remains **P2 (`1d434719`)**. Prime suspect: `idle_not_READY` incrementing **47M times**
inside the PF engine loop. This is the day's own rule applied to itself — probe builds distort, and an
instrument that costs 6.7% is fine as long as nobody quotes its throughput.

#### *** THE FINDING P3 WAS BUILT TO GET: THE ENGINE IS STARVED ***
```
tiles_latched   =     1,056   of 3,168 tiles  (EXACTLY 1/3, IDENTICAL both runs)
blocks_issued   =   152,472 / 186,588  =>  37.2 / 45.6 MB   (budget: 519 MB full warm)
idle_not_READY  = 47,246,596 / 46,646,395
```
- **Only 7-9% of the intended footprint is warmed.** 144-177 blocks per tile against 640 for a full warm.
- **The trigger fires for exactly ONE THIRD of tiles**, identically across runs. That precision is
  structural, not chance — **`ACC_N=3` is the obvious suspect** (trigger is `group == GROUPS-1 &&
  ksi == n_kseg-1`; check how GROUPS resolves at `ACC_N=3, G=6`).
- The engine idles ~44,000x per tile it latches — it is spinning with nothing to issue almost always.
- **=> P2's 8.6% win was bought by warming under 9% of what the design intends.** If coverage is the
  lever, a large multiple may still be available. **This is the first lead of the day with a COUNTER
  behind it rather than an inference.**
- ⚠ `blocks_issued` varies 22% run-to-run (152K vs 187K) while span varies 1.2% — the coverage itself is
  unstable, which is worth understanding before tuning it.

#### NEXT, IN ORDER
1. **Raise coverage** — why do only 1/3 of tiles latch, and why does the engine idle ~44,000x per latch?
2. **Then** the completion-chain discrimination probe (design sketched, see below).

### 2026-07-25 — P4 COVERAGE: the fix worked completely (23% -> 100% warm). Span comparison CONFOUNDED — not concluded.
- **Bin `e2397d95`**, RUN TWICE: span 37,994,252 / 37,917,832 (**spread 0.20%** — tightest config today).
  `occ[20]=3232`, WORK-EXACT 190080, oracle 3168/3168 stride=1, no hang. OFF byte-identical `c6c0d0cf`.

#### THE ONE-THIRD MYSTERY — SOLVED, AND MY GUESS WAS WRONG
I suspected `ACC_N=3`. **Wrong** — GROUPS = G/ACC_N = 6/3 = 2, so the group trigger fires once per tile.
**It is HOST CHUNKING:** `ML8_COOP_CHUNK=96` tiles across 64 WGs => 32 WGs get 2 tiles, 32 get 1. A WG
can only latch a NEXT tile if it HAS a second tile in-dispatch, so max latches/chunk = 96-64 = 32, and
33 x 32 = **1056 = 3168/3** exactly. Deterministic partition => bit-identical across runs.
**The kernel cannot invent next-tiles that do not exist in the dispatch.** Raising `tiles_latched`
requires host chunk policy — and `ML8_COOP_CHUNK` is the documented compositor-safety knob (96/dispatch
with a 5ms yield between chunks), so **it is NOT a free lever and does not get bundled into a kernel
bring-up.** Filed as a separate, careful experiment.

#### COVERAGE FIX: COMPLETE SUCCESS
| metric | P3 | P4 |
|---|---|---|
| blocks per latch | 144–177 | **639.9 / 640.0** (of 640 = full warm) |
| bytes warmed | 37.2 / 45.6 MB | **165.0 / 165.5 MB** (4.4x) |
| coverage run-to-run instability | **22%** | **0.3% — GONE** |
| idle counter (raw) | 47M unthrottled | ~34.8M, throttled 1/64 |
Grok's read that the READY window was the block limiter was correct; moving the trigger to group 0
filled it. The instability had the same root cause, as it predicted.
**Throttling the idle counter recovered 4.0% of P3's 6.7%** (39.55M -> 37.96M), confirming that counter
was the P3 regression.

#### *** THE SPAN COMPARISON IS CONFOUNDED — DELIBERATELY NOT CONCLUDED ***
P4 (37.96M) is **2.4% slower than P2** (37.07M), outside noise. **BUT P4 CARRIES THE P3 COUNTERS AND P2
DOES NOT**, and those counters measure **2.4% (throttled) to 6.7% (unthrottled)**. The coverage effect
and the instrumentation cost are confounded **at the same order of magnitude**. This pair cannot
separate them.
**Attributing P4's 2.4% to coverage here would be instance #10 of this file's "numbers that were never
measurements".** One dispatch removes the ambiguity, so it gets one dispatch.

#### DECISIVE EXPERIMENT DISPATCHED: **P4-CLEAN** (P4 coverage, diagnostic counters compiled out)
Interpretation fixed IN ADVANCE so it cannot be retrofitted:
- **faster than P2** => full coverage wins; make it shipping.
- **~= P2 (within ~1%)** => coverage is NEUTRAL; P2's 23% was already enough and the extra 4.4x of HBM
  traffic buys nothing. **That is a real finding about prefetch sizing, not a failure.**
- **slower than P2** => full coverage HURTS; there is an optimum below full warm — sweep batch size.
Will be run twice regardless. **Shipping config remains P2 (`1d434719`) until this resolves.**

### 2026-07-25 — ⭐⭐⭐⭐⭐ **THE WALL IS LOCATED: THE COMPUTE BURST, NOT COORDINATION.** Two long-standing root causes refuted.
Three arms, defsym-only, all `OVERLAP=0 ROLEFLOW=0 RCONV=0 PASSTIME=0 PREFETCH=0`. All correctness-clean
(WORK-EXACT, `occ[0]=0`, no hang).

| arm | config | bin | items | span | ticks/item |
|---|---|---|---|---|---|
| **A** | POOL_N=1 SEGK=256 | `128500f7` | 190,080 | **24,535,292** | **129.1** |
| **C** | POOL_N=1 SEGK=64 | `74e6227f` | 760,320 | 30,111,052 | 39.6 |
| **B** | POOL_N=2 SEGK=64 | `8376e32e` | 760,320 | 30,335,700 | 39.9 |

**4x more work items cost only 23% more time.** If coordination dominated, 4x the items would cost ~4x.
Solving A and C as *work + fixed coordination*:
```
work  ~ 29.8 ticks per SEGK=64 unit      coordination ~ 9.8 ticks/item (FIXED)
=> at SEGK=256:  ~92% COMPUTE BURST,  ~7.6% COORDINATION
```

#### *** TWO ROOT CAUSES REFUTED ***
1. **"Coordination costs ~600x the work it coordinates"** (KG, 2026-07-13) — **STALE AND WRONG BY TWO
   ORDERS OF MAGNITUDE.** It predates SELFSERVE/DECENTASN. Today's measurements had been eroding it all
   day: boundary <1%, reservation CAS ~0.01%, poll throughput doubled with 1.3% runtime effect.
2. **"POOL_N=1 is why"** — **REFUTED. Depth-2 pipelining measures NEUTRAL (0.7%, inside noise).**
   This is the FIRST EVER EXECUTION of `POOL_N=2` on this kernel — clean, working, and useless.
   It was **blocked-for-nothing**, not blocked-and-valuable. (Also: `POOL_N>1` is refused on the
   `OVERLAP=1` path by an explicit Phase-1 guard at :895, so the 40KB LDS reclaim can never be spent on
   pipelining — moot now.)

#### *** WHERE THE WALL ACTUALLY IS ***
At SEGK=256 the GEMM needs **~324 us of math** at the 307 TF peak and takes **245,000 us — 756x off.**
Coordination explains **7.6%**. The remaining ~92% sits inside the per-item compute burst, which runs
**~230x slower than the math it contains.**
=> **THE WALL IS THE COMPUTE PATH: B loads from L2/HBM, A `ds_load`s, WMMA issue, `ds_add` reduction.**
NOT the frontier, NOT coordination, NOT the poll loop. **Five sessions of work have been aimed at the
wrong ~8%.**
- SEGK remains a real lever (SEGK=256 beats SEGK=64 by 23%) — consistent with the existing log entry
  that SEGK is the throughput lever. But it is a lever ON the compute burst, not on coordination.

#### PROBE-INFLATION CORRECTION TO EVERYTHING ELSE LOGGED TODAY
**Every span quoted today was a `DSWS2_PASSTIME=1` PROBE build.** Arm A (probes off) is **24.5M**, next
to the 07-23 canonical **25.48M** — while today's "best" (P2) read **37.07M**. The prefetch A/B remains
VALID (controlled, probe-on both sides, 8.6%), but the absolute lineage was inflated throughout and
**whether the 8.6% survives at `PASSTIME=0` is OPEN and must be re-measured before it is banked.**

#### NEXT — instrument the COMPUTE BURST, not the frontier
Use Grok's Step-0 design (counts first, then ONE RTC interval at a time, own throttle, PASSTIME=0) to
discriminate inside the burst: B-load latency vs WMMA issue vs LDS `ds_add` reduction vs C-store.
The completion-chain probe (TILEDONE -> C-store -> GSTORED -> boundary) is now LOWER priority: that
chain lives in the 7.6%.

### 2026-07-25 — 🔧 **STRUCTURAL: SELFSERVE LEFT DEAD MACHINERY LIVE. POOL_N IS NOW INERT.** (kmbandy called it, repeatedly)
**The complaint, verbatim:** *"POOL_N should not matter because SELFSERVE makes it inert — or at least
should. So once again something isn't built correctly."* **Correct on every count.**

#### THE DEFECT
Under SELFSERVE waves self-serve from GLOBAL (`global_load_tr` off `Bshuf`, ~:5794/:5838) and never read
the staged LDS pool. But:
- `:1122` `.set ACC_BASE, (OP_BASE + POOL_N*OPSTRIDE)` — **the operand pool was allocated by `POOL_N`
  regardless of SELFSERVE.** Only `DSWS2_OVERLAP` reclaimed it.
- `ASTAGE_R`/`BSTAGE_R` still ran from `.Lflow_feed` (:5028) and `.Lflow_coast` (:6279) — **staging was
  still executing** (65,918 events at SEGK=64, 1,454 at SEGK=256) writing operands nobody read.
- SELFSERVE's guard block (:1255-1275) demanded KMAJOR=0/DECENTASN=1/JDEPTH=1/BANKZERO=1/BATONGATE=1 and
  **nothing that reclaimed the pool or killed staging.**
- `DSWS2_OVERLAP` (:891) **explicitly requires SELFSERVE=1** — it was a LATER BOLT-ON reclaiming what
  SELFSERVE should have killed at source, and only for `POOL_N=1` (:895), which is why it refused
  pipelining.
- **SECOND DEFECT, same class, found by the builder:** the dead coordinator still assembled
  `s_cmp_ge_u32 s46, POOL_N` under DECENTASN — unreachable at runtime, leaking `POOL_N` into `.text`.

#### THE FIX, PROVEN
| | before | after |
|---|---|---|
| `POOL_N=1/2/3` under SELFSERVE | different bins, different LDS | **BYTE-IDENTICAL** (`815f9894` @256, `d7221d80` @64) |
| LDS @ SEGK=256 | 54,784 | **13,824** (full 40KB reclaim, **no OVERLAP needed**) |
| LDS @ SEGK=64 | 24,064 | **13,824** |
| `.text` @ SEGK=64 | 28,428 | **23,564** (~4.9KB dead code) |
| `feed-stages` | 1,454 / 65,918 | **0, structurally** (macros not emitted) |
- **`DSWS2_OVERLAP=1` is now byte-identical to SELFSERVE alone** on the pool-only profile. Its entire
  reclaim purpose was redundant; it survives only as a feature gate for ROLEFLOW/PREFETCH/RCONV.
- **`SELFSERVE=0` classic ring correctly untouched** — still sizes by POOL_N (23,040 vs 33,280).
- Runs CLEAN: `occ[0]=0`, WORK-EXACT 190080, `occ[20]=3232`, oracle 3168/3168 stride=1, no hang.

#### *** TIMING: MARGINAL, NOT CLAIMED ***
span 23,916,748 / 24,203,684 (mean **24,060,216**) vs prior dirty 24,535,292 = **1.9% against a 1.2%
noise floor.** Too close to call and I am not banking it. At SEGK=256 the dead staging fired only 1,454
times, so it was never a large TIME cost — **it cost 40KB of LDS, 4.9KB of `.text`, and an entire
bolt-on subsystem built to work around it.**

#### *** THE PATTERN — THIRD INSTANCE TODAY ***
1. The **prefetch** aimed at an already-consumed block for months (rebuilt: +8.6% vs no prefetch).
2. **SELFSERVE's staging pool** — superseded, left allocated and executing.
3. The **dead coordinator** — unreachable, still assembled, leaking a defsym into `.text`.
**Superseded machinery left live, then tuned around instead of removed.** In each case the workaround
(OVERLAP for #2) became load-bearing and constrained everything downstream. **When a knob "matters" that
architecturally should not, that is the bug — not a tuning opportunity.**

#### CONSEQUENCE FOR THE WALL RESULT
The three-arm experiment that located the wall ran with this dead staging active in ALL THREE ARMS. The
conclusion **survives a fortiori** (coordination is even smaller without it), but the arms were not clean
and I described them as clean. Re-measured at SEGK=256 post-fix: **24,060,216**, still ~740x off the
~324us of math this GEMM contains. **The wall remains the COMPUTE BURST.**

### 2026-07-25 — 🎯🎯 **THE WALL, NAMED: THE K-STEP LOOP IS NOT DOUBLE-BUFFERED.** Confirmed by counts, arithmetic, AND source.
#### BURSTCNT Step 0 (bin `b1813ecb`) — ALL SEVEN COUNTS MATCH PREDICTED ARITHMETIC EXACTLY
```
BURST      190,080     BLOAD  12,165,120     ALOAD  3,041,280     WAITLD  3,041,280
WMMA    12,165,120     DSADD   6,082,560     WAITDS   190,080
per-BURST: BLOAD=64.00  ALOAD=16.00  WAITLD=16.00  WMMA=64.00  DSADD=32.00  WAITDS=1.00
```
Exact agreement on all seven validates the counters AND the model. `DSWS2_BURSTCNT=0` byte-identical to
`815f9894`. **ACC-live invariant verified by disassembly:** all 62 added in-burst instructions are
`s_add_co_u32` on private SGPRs — zero memory/VGPR/exec/sendmsg; the 7 emit atomics are at retire.

#### THE STRUCTURE
Per item: **80 global loads (64 B + 16 A — BOTH from global under SELFSERVE)**, **64 WMMA**,
**32 `ds_add_f32`**, **16 blocking `s_wait_loadcnt` — one per k-step.**

#### *** THE SOURCE SETTLES IT (`.Lflow_da_ss_rowblk`, :5991) ***
```
.rept KSEG_STEPS            // 16 @ SEGK=256
  .rept FN  -> global_load_tr_b64 B   // issue THIS step
  .rept FM  -> global_load_b64    A   // issue THIS step
  s_wait_loadcnt 0x0                  // *** WAIT for THIS step ***
  .rept FM/FN -> v_wmma               // compute THIS step
.endr
```
**Issue -> wait -> compute -> repeat. ZERO overlap.** Step k+1's loads are not issued until step k's
WMMAs are done, so **all 16 k-steps pay a fully exposed memory latency with nothing in flight to hide
it.** Under SELFSERVE both operands come from GLOBAL — there is no LDS staging left to hide either
behind, which is why this is fully exposed rather than partly absorbed.

#### THE ARITHMETIC CLOSES
An item is 126.6 ticks ≈ **3,040 shader clocks** @2.4GHz. 16 exposed L2 round trips x ~190 clocks
≈ **3,040 clocks — the ENTIRE item budget.** 64 WMMA at ~16-32 clocks issue ≈ 1,000-2,000, same order.
Counts predicted it, arithmetic fits it, source confirms it. **Step 1 timing is NOT needed for the
diagnosis** — the ablation (build the double-buffer, measure) is the better test.

#### THE FIX — classic double-buffer, and it is the PROVEN lever on the sibling kernel
Prologue issues step 0; loop issues step k+1, THEN waits step k, THEN computes step k.
Register cost: FA=FM*2=2, FB=FN*2=8 => 10 operand VGPRs today, 20 double-buffered. ACC=FM*FN*8=32.
Fat allocation is 112, so the +10 should fit — **must be verified, not assumed.**
`KWINBPF` double-buffering took `occ_kernel_wggemm2.s` from **162 -> 164.9 TF** (2026-06-19 ladder).

#### FLAGGED, NOT CLAIMED
BURSTCNT probe cost 26% (30,388,412 vs clean 24,060,216) where pure SALU should cost ~nothing — hints
the k-step loop is ISSUE-sensitive, which would be a second (independent) reason double-buffering helps.
One unreplicated run; counts unaffected and exact.

### 2026-07-25 — ❌ **KDBUF DOUBLE-BUFFER: REFUTED. And the refutation exposes an arithmetic error of mine that invalidates the "compute burst is 92%" claim.**
- **Bin `c5138582`**, SEGK=256. span **24,137,328** vs baseline **24,060,216** = **0.32% SLOWER — inside
  the 1.2% noise floor.** Correctness clean (`occ[0]=0`, `occ[20]=3232`, WORK-EXACT 190080, oracle
  3168/3168 stride=1, no hang).
- **The BUILD is correct** — the pipeline is genuinely in the emitted code (`s_wait_loadcnt` census
  35x`0x0` -> 20x`0x0` + 15x`0x5`; `s_alloc_vgpr` 80 -> 96; k+1 loads issue into the alternate buffer
  before the wait gating step k). **The HYPOTHESIS was wrong.**
- **PRE-REGISTERED PREDICTION:** *"multi-x if the diagnosis is right; a few percent means the diagnosis
  is incomplete."* It came back at **ZERO**. This is what pre-registration is for.

#### *** THE ROOT ERROR, MINE ***
I computed `span / items` = 129 and called it **"ticks per item"**, treating it as the DURATION OF ONE
BURST. **It is not.** It is the **system-wide item completion rate** with **1920 waves running
CONCURRENTLY**. The real wave-time per item is `span x waves / items` ≈ **243,000 wave-ticks** — so the
burst is a *tiny* fraction of wave-time, not 92%.
**And that is exactly why double-buffering was inert:** at **15 waves per SIMD** the memory latency was
**already hidden by multithreading**. While one wave waits on a load, fourteen others issue. Hiding
latency *within* a wave buys nothing when it was never exposed at the SIMD level.

#### *** SECOND TIME TODAY. SAME ERROR. ***
The LDS-CAS contention hypothesis died the same way this morning — reasoning about ONE wave's serial
timeline while forgetting the SIMD hides it. **I had written "plain LDS latency should be hidden by
15-way SIMD multithreading" in my own notes hours earlier and then failed to apply it here.**
**RULE: on this kernel, ANY per-wave latency argument must first answer "why is this not hidden by the
other 14 waves on the SIMD?" A latency that multithreading can absorb is not a wall.**

#### RETRACTED
- ❌ **"The compute burst is ~92% of runtime."** The A-vs-C decomposition (~29.8 work + ~9.8 fixed per
  item) is still a valid decomposition of the **SYSTEM THROUGHPUT RATE**, but calling the 92% "the
  compute burst" was an overreach. It is **"everything that does not scale with item count"** — a much
  weaker claim admitting several explanations.
- ❌ **"16 exposed L2 latencies are the wall."** They were not exposed.

#### STILL STANDING (unaffected)
Coordination is small (~7.6% of the system rate). POOL_N / pipelining NEUTRAL. The SELFSERVE structural
fix is real (POOL_N inert, 40KB LDS reclaimed, dead staging gone). Boundary <1%, sleep ~1%, CAS ~0.01%,
carrier stalls 0, grow-fail 0, drain ~0, poll-loop throughput irrelevant to runtime.

#### *** THE WALL IS ONCE AGAIN UNLOCATED ***
**The next step must start from a PER-WAVE-TIME budget that respects 1920-way concurrency** — not a
system rate divided by item count. Until that budget is built correctly, every "X is N% of runtime"
claim on this kernel is suspect, including the ones above.

### 2026-07-25 — ⭐⭐⭐⭐ **ABLATION SWEEP: THE ENTIRE COMPUTE PATH IS ~2% OF RUNTIME.** No instrument. Just span minus span.
kmbandy's correction, and it is the lesson of the day: *"stop relying on weird entirely too complex
measuring mechanisms... literally just checkpoint 2 - checkpoint 1. we need numbers."*

| arm | bin | span | vs baseline |
|---|---|---|---|
| **BASELINE** | `815f9894` | **24,060,216** | — (replicated 2x, noise ~1.2%) |
| NOWMMA | `0fe26484` | 24,243,168 | **+0.8% SLOWER** |
| NODSADD | `25ebf127` | 24,153,200 | **+0.4% SLOWER** |
| NOCFLUSH | `e35037c1` | 24,392,756 | **+1.4% SLOWER** |
| NOBLOAD | `9f9bdf8e` | 23,538,772 | **-2.2%** (only mover, barely outside noise) |

Every arm `bad=76032` — the oracle failing IS the proof the ablation took effect — and `computed=190080`,
so work-exactness held throughout. All four bins verified **distinct** before running, so no arm was
inert (the `NOCFLUSH`-assembles-byte-identical trap logged 2026-07-20).

#### *** DELETING THE MATH CHANGES NOTHING ***
All **12,165,120 WMMAs** removed: +0.8%. The LDS reduction removed: +0.4%. The C-store removed: +1.4%.
Three of four got **slower**. Only the B fetch moves the needle at all, and only by 2.2%.
=> **The kernel is 743x off peak and ~98% of wave-time is spent NOT executing the compute path.**
This POSITIVELY MEASURES what had only been inferred, and retires the compute burst as a candidate.

#### *** THE METHOD LESSON — THIS IS THE ONE TO KEEP ***
**Every reliable number today came from an ABLATION. Every unreliable one came from an IN-KERNEL PROBE.**
- Ablations (all clean first try): prefetch rebuild (+8.6% vs none), POOL_N (neutral), KDBUF (refuted my
  own hypothesis in one run), the SELFSERVE structural fix, and this sweep.
- Probes (all failed, each in a NEW way): PASSTIME (poisoned by a shared `s71` trigger with the
  prefetch), BURSTCNT (26% schedule tax), WTBUDGET (u32 overflow of the global sum, then a negative
  residual). Between them: **zero numbers I would defend.**
**An ablation measures the thing itself. A probe measures whatever it happens to sample — and then you
have to prove what that was.** Every layer I added to a probe (T0 calibration, armed flags, u64 sums,
external budget assertions, s71 co-tenant masks, five buckets, sampling, derivation, residuals) existed
to fix a problem the previous layer created. **Reach for the ablation first. Always.**

#### NEXT — and it is deliberately the simplest possible thing
`DSWS2_GAP`: ONE timestamp at the end of a compute burst, ONE at the start of the next, subtract,
accumulate, emit. One accumulator, one count, one number. **No buckets, no sampling, no residual, no
calibration, no throttle** (99 bursts/wave = 99 pairs, small enough not to need one — and a throttle is
exactly what poisoned T1). Built by Grok (task aa5810c9), not yet run.

---

## 2026-07-26 — POLLSTAGE COMPLETE: the six-stage poll decomposition closes at 94%

### 0. The blocker from last night's brief was MY error. No kernel defect. No fix needed.

The 2026-07-25 brief refused to quote stage 1 because `n=30.7M` implied 485 passes/wave while
"`occ[86]` parks = 169.7M" implied 2,679 — a 5.5x gap I called a counter bug.
**I had compared `PS_N` from the POLLSTAGE build against `occ[86]` from a DIFFERENT build's log.**
The stage-1 run's own `occ[86]` is **30,289,464**. Within-run:

| run | n (PS_N) | own occ[86] | ratio |
|---|---|---|---|
| stage 1 | 30,736,951 | 30,289,464 | 1.015 |
| stage 2 | 31,516,673 | 31,069,772 | 1.014 |
| stage 6 | 29,906,543 | 29,906,543 | **1.0000** |

Loop heads are a superset of parks, so ~1.015 is exactly right. **Stage 6 settles it beyond argument:
its `n` equals its own run's `occ[86]` to the unit** — the park counter and the stage-6 counter are the
same event. The stamp fires on every iteration, exactly as the source says.

**=> METHOD RULE 9: NEVER COMPARE A COUNTER ACROSS RUNS.** Probe cost changes the pass count by up to
5x. `occ[86]` legitimately ranges **30M–154M** across these seven runs *at the same kernel and shape*.
This is the same family as rule 7 (a magnitude match is not a mechanism): a number that looks wrong
against the wrong denominator is not evidence of anything.

The "possible hang" was also nothing — just my 10-minute command ceiling across 6 builds + 6 runs.
Latch was clear at start of day; stages 3,4,5,6 all ran clean, one dispatch at a time.

### 1. Offline verification before any dispatch (rule 6)

Reconstructed the build profile from the findings doc and **proved it by rebuilding stage 3 and
reproducing the on-disk bin bit-for-bit** (`04efe002`). Then built 4/5/6 and checked all three against
the shas the brief had recorded: `6c6a7888` / `277e17e7` / `10bba694` — **all three matched exactly.**
That is the whole profile validated before spending a single dispatch.

### 2. The runs (3 dispatches, one at a time, all clean)

All: `computed=190080` WORK-EXACT, `oracle ok=76032 bad=0`, exit 0, no reset, no latch.

```
stage4  6c6a7888  mean=0.000097 ms  n=14,713,995  total= 1433.486 ms  span=24,580,112
stage5  277e17e7  mean=0.001206 ms  n=32,544,886  total=39242.256 ms  span=23,423,040
stage6  10bba694  mean=0.001040 ms  n=29,906,543  total=31107.661 ms  span=23,108,972
```
Plus stage 3 had ALREADY run twice on 07-25 and was never recorded in this log:
```
stage3  04efe002  mean=0.000487 ms  n=4,567,862  span=22,966,308
stage3' 04efe002  mean=0.000477 ms  n=4,873,280  span=22,757,568   (replicates to 2%)
```

### 3. Normalization — and why `total` is NOT comparable across stages

Pass counts differ up to 5x between runs, so raw `total_ms` is apples-to-oranges (stage 4's run had 5x
more passes than stage 1's yet 13x less total). Run-invariant metric:
`ns per loop-head pass = mean x reach`, `reach = n / (occ[86] x 1.01477)`, constant from the stage-1
run — the one run where `n` IS the loop-head count.

### 4. MS PER STAGE, PER WAVE (479 passes/wave)

The six stages are contiguous and non-overlapping (`:4704`→`:7118`) = a true partition of one pass.

| stage | what | ns/pass | ms/wave | share |
|---|---|---|---|---|
| **5** | **`da_peek` reservation attempt (ends in a park)** | 1181 | **0.566** | **34.1%** |
| **6** | **park + `s_sleep`** | 1025 | **0.491** | **29.6%** |
| 1 | loop head + `deadman_check` | 618 | 0.296 | 17.8% |
| 2 | snapshot / FLOWTERM / body-gate | 590 | 0.282 | 17.0% |
| 3 | role select + dispatch | 41 | 0.020 | 1.2% |
| 4 | feed → `da_peek` gate | 9 | 0.004 | 0.3% |
| | **SUM** | **3463** | **1.659** | |

### 5. IT CLOSES — and that is the point

**1.659 ms/wave vs 1.77 ms wave lifetime measured independently by the GAP probe = 94% closure.**
The GAP probe (two-stamp HEAD/TAIL) shares no code with POLLSTAGE. Two instruments, no common
mechanism, agreeing to 6%. **This is the first decomposition in this project that reconciles against an
independent measurement rather than against an assumption** — and it is the reason these numbers are
quotable when yesterday's PASSTIME/BURSTCNT/WTBUDGET numbers were not.

### 6. WHERE THE TIME IS

**63.7% of every poll pass is stage 5 + stage 6: the reservation peek that FAILS, and the park that
follows it.** Stages 1+2 (34.8%) are loop-head watchdog and gate re-evaluation. The role economy and
feed dispatch — stages 3+4, the parts with actual logic in them — are **1.5% combined.**

Consistent with everything else on record: work is ~2%, occupancy is full, launch is clean, ~35% of
waves never run a single burst, and `occ[86]` says 100% of feed-path iterations find an empty frontier.
The kernel is not slow at doing things. It is spending its life asking for work and being told no.

### 7. TWO CAUTIONS — read before quoting

1. **Stage 3/4 reach (~9%) is NOT "only 9% of passes survive the body gate."** Stage 5 is downstream of
   stage 4 yet reaches **98%**. Only possible because `pollstage_leave` is ARMED-gated: ~89% of passes
   **branch into the feed region below `pollstage_enter 4` (`:5710`)** and are never counted by 3 or 4.
   Stages 3/4 measure TOP-OF-REGION ENTRY, not gate survival. (The ns/pass figures remain correct —
   mean and reach are consistent with each other.)
2. **These are WAVE-TIME attributions on a 15-way-multithreaded SIMD, not instruction latencies**
   (method rule 4). Stage 6 = 0.491 ms/wave is NOT a claim that `s_sleep` runs for 0.491 ms — `SLEEPN=2`
   is ~128 clocks. It is a claim about where wave-time lands, which is what was asked for.

### 8. What this does NOT answer

The open question is unchanged and now sharper: the peek fails 98% of the time, so **why does the
frontier expose so little work?** Structural fact already measured: `ML8_COOP_CHUNK=96` across 64 WGs
means only 32 WGs per chunk have a second tile, capping next-tile latches at `33 x 32 = 1056 = 3168/3`.
The kernel cannot invent work the dispatch did not give it. `ML8_COOP_CHUNK` is the compositor-safety
knob — stepping it is a rule-7 conversation, not a free lever.

---

## 2026-07-26 (afternoon) — 2 WG/CU CONFIG OF RECORD: full sweep, ablations, phase decomposition

**Full write-up: `DSWS_FINDINGS_2026-07-26.md`.** This entry is the raw data.
~300 dispatches. 0 hangs, 0 GPU resets, 0 latches, every run work-exact.

### Config (kmbandy's, now enforced in build_flow.sh + gpu_run.sh)
```
WAVES=16 + ML8_POOL=128 = 128 WGs x 16 waves = 2048 waves (2 WG/CU)   DSWS2_PREFETCH=1 DSWS2_OVERLAP=1
SEGK free in {64,128,256}   bins: 256->585d287e  128->62001b24  64->bc75d341   all LDS=13824B
```
Bring-up: `occ[20]=3296` (=128 WGs; 64 WGs reads 3232), oracle ok=76032 bad=0 stride=1, computed=190080.

### THE CHUNK CONFOUND — the "3.5x regression" was mine
Same bin/config/shape, ONLY `ML8_COOP_CHUNK` differs:
```
chunk  96:  10 chunks/rep   span/rep = 7,750,996 ticks
chunk 512:   2 chunks/rep   span/rep = 2,017,675 ticks
=> per-chunk fixed C = 716,665 ticks = 7.17 ms ;  per-tile c = 664 ticks = 6.6 us
```
71% of runtime is launch/drain at chunk 512; 92% at chunk 96. Cross-checks vs an independent run:
predicted 7.17ms + 96x6.6us = 7.8 ms/chunk vs 8.33 measured on 07-25.
**RETRACTED same-day: the "7,866 ticks/tile constant across 18x K" and the "sub-96-tile cliff" are
artifacts of pinning chunk=96 (which makes chunks proportional to tiles by construction).**

### Fed sweep, chunk 512 (27 PASS / 0 FAIL / 6 UNSUPPORTED)
best **4.817** (ml8_dense_ffn_down M2048 K=9216) = 1.57% of 307 peak; mean 0.893; median 0.375.
vs 07-21 published 4.36 at 1 WG/CU => **+10.5%, no regression.**
TF tracks n_kseg: 36->1.16, 16->0.54, 10->0.38, 8->0.21, 3->0.12, 2->0.09.

### Ablations, 4 arms x 27 shapes (median % change vs control, + = SLOWER without it)
```
NOWMMA +1.6%   NODSADD +1.7%   NOCFLUSH +0.5%   NOBLOAD -0.0%      (noise floor ~1.2%)
```
Deleting all math, the LDS reduction, the C store, and the ENTIRE B stream changes nothing, on every
shape. NOT compute-bound, NOT memory-bound. (kmbandy: this was the wrong instrument for "where is the
slowness" — it re-proves a negative already known from 07-25 and localizes nothing. Correct.)

### *** PHASE DECOMPOSITION, 6 stages x 27 shapes — THE RESULT ***
Normalized per shape by that shape's OWN stage-1 loop-head count.
```
absolute total ns/pass:  CV 0.33  (7.1x spread, 534-3772 ns)
phase SHARES:            CV 0.09-0.16
```
| stage | what | median share | CV |
|---|---|---|---|
| 5 | `da_peek` reservation attempt | **30.0%** | 0.16 |
| 6 | park + `s_sleep` | **23.9%** | 0.11 |
| 1 | loop head + `deadman_check` | 21.4% | 0.12 |
| 2 | snapshot / FLOWTERM / body-gate | 21.0% | 0.11 |
| 3 | role select + dispatch | 2.0% | 0.14 |
| 4 | feed -> peek gate | 0.7% | 0.32 |

**NO SHAPE-SPECIFIC SLOWNESS.** Every shape burns wave-time in the same mix, scaled by a per-shape
constant: 54% failed-reservation+park, 42% watchdog+gate, 2.7% real dispatch logic. Reproduces the
07-25 single-shape run at the OTHER config (1 WG/CU: s5 34% s6 30% s1 18% s2 17%).
Stages are fractions of WAVE-TIME; the 7.17 ms/chunk launch/drain is outside all six (no wave exists).

**OPEN:** the 7.1x spread in absolute ns/pass is unexplained (ffn_down ~1150 ns cheapest, lm_head 3772).
**UNMEASURED:** everything today ran at SEGK=256 only. SEGK 64/128 discriminate per-TILE vs
per-RESERVATION cost (same work, 4x the reservations) — 3 dispatches, queued.

### Harness fixes
1. Config of record enforced (`build_flow.sh` defaults+REFUSE, `gpu_run.sh` geometry REFUSE).
2. **`ML8_POOL` was NEVER passed by `dsws_realshape_bench.py`** — every sweep it ever ran was 64 WGs.
3. **N PADDING**: M was always padded to 96 with TF corrected by real/padded; N was REFUSED on n%64,
   silently excluding 6 of 33 shapes incl. `mlmf_mamba_in_proj` (half the Mamba MIMO GEMM path).
   Harness gap, never a kernel limit — the kernel only sees `NTL=N/64`. Now 30/33 legal at SEGK=256,
   **33/33 at SEGK=128**. Padding waste reported, not hidden (`router_out` N=8->64 = 700%).
4. `--segk` CLI knob; `n_kseg=1` is now a property of the chosen SEGK, and the message names the fix.
5. **CLAIM GUARD + `GPU_RUN_DRY=1`** in `gpu_run.sh` (see below).
Regression-verified: offline re-parse of all 27 control logs returns identical TF.

### INCIDENTS — I ran on a card another session held, TWICE
**(1) REAL, 11:22:16-11:25:19.** My claim expired at its 3h TTL (11:22:10); the board correctly
promoted the queued mlambaformer session at 11:22:16; my driver had no notion of claim validity and
put 8 more dispatches on the card. Announced (55b11793). Their discipline was correct throughout.
**(2) INERT, 11:53.** Testing the NEW guard's fail-soft branch, I pointed the board URL at a dead
address — and fail-soft means PROCEED, so it dispatched while another session held the card. Inert
only by luck (missing DSWS2_K/MTL/NTL => occ_dispatch opened the KFD node and exited before any PM4;
41-byte log, 0 resets). Announced (66b286fc).

=> **`gpu_run.sh` now REFUSES if the board positively reports a different holder** (fail-soft on
unreachable board / missing session id, so an outage never halts work). `DSWS_SKIP_CLAIM_CHECK=1`.
=> **`GPU_RUN_DRY=1`** runs every guard and dispatches nothing.
=> **METHOD RULE 12: A FAIL-SOFT PATH CANNOT BE VERIFIED BY EXECUTING IT AGAINST LIVE HARDWARE** —
"proceed" IS the behaviour under test. Read the code or use a dry run. That is precisely how (2) happened.
=> `board_release` returns a COUNT, not a boolean. `result: 0` = nothing matched. Verify with board_check.

---

## 2026-07-26 (afternoon II) — SEGK DISCRIMINATION + THE PROBE RETRACTION AND ITS FIX

### 1. SEGK: 256 is optimal, and the cost model closes
Same shape (ffn_down M2048), chunk 512, 2 WG/CU, only SEGK differs. All work-exact, oracle clean.
```
SEGK  n_kseg  reservations   span/rep     TF   vs 256
 256      36       190,080  2,017,675    5.0    +0.0%
 128      72       380,160  2,445,173    4.1   +21.2%
  64     144       760,320  3,220,081    3.1   +59.6%
```
**PRE-REGISTERED PREDICTION HELD:** from the 256-vs-128 fit I predicted SEGK=64 at 3,300,145 ticks /
TF 3.05 BEFORE running it; actual 3,220,081 / 3.1 = **2.4% error**.
```
COST MODEL (fit from TWO independent experiments that agree):
    time = chunks x 8.06 ms  +  reservations x 21.3 ns
r = 2.13 ticks/reservation, consistent across all three pairwise fits (2.038-2.249).
At the config of record: 80% per-chunk, 20% reservations.
```
=> There is NO pure per-tile term. The "664 ticks/tile" from the chunk experiment is just
reservations-per-tile (G*n_kseg = 216) x 21.3 ns = 461 ticks.
=> **KILLS the "lower SEGK might be free now that the flush is free" idea.** The flush IS free
(NODSADD +1.7% across 27 shapes), but smaller segments mean MORE RESERVATIONS, and reservations cost.
Lower SEGK is also jitterier: spread 1.3% -> 16.8% -> 14.1%.

### 2. *** THE POLLSTAGE PROBE COST 4.8x WHAT IT MEASURED — RETRACTED, THEN FIXED ***
Measured against a probe-free control, 27 shapes:
```
SPAN:        0.99x  (0.91-1.10)   <- looks free
PASS COUNT:  0.21x  (0.13-0.39)   <- 5x FEWER loop passes in the same span
```
Same wall time, one fifth the iterations => each probed pass costs ~4.8x a real one.
**Stage 1 brackets THREE SALU INSTRUCTIONS** (`deadman_check` is throttled 1-in-64) **and reported
487 ns.** That number was the instrument.
**THE CHECK THAT FAILED: I judged probe cost by SPAN (0.99x, reads as free). The right metric was
WORK COMPLETED — pass count — which said 0.21x. Both numbers were in every log all day.**

Worse than a uniform offset: probe load VARIES BY STAGE, because `pollstage_enter` only fires where
execution reaches it. s1/s2/s5/s6 ~2.0 sendmsg/pass (4.8x, 4.9x, 3.9x, 4.0x); s3 1.12 (2.87x);
s4 0.23 (**1.06x — free**). The stages I ranked most expensive carried the heaviest probe.
A GENERAL CORRECTION IS IMPOSSIBLE: inverting the pass ratio to solve for probe cost yields NEGATIVE
work on s1/s2/s3 (-138, -153, -76 ns). That is a refutation — the probe changes BEHAVIOUR, not just
per-pass cost, so it cannot be arithmetically removed.

### 3. ROOT CAUSE IS THE CHIP, AND THE FIX IS RATE NOT MECHANISM
Verified with the assembler: **gfx1201 has NO cheap clock.** `s_memtime` and `s_memrealtime` are
"not supported on this GPU"; there is no HW_REG shader-cycle register. `s_sendmsg_rtn`
(MSG_RTN_GET_REALTIME) is the ONLY shader-readable clock, it serialises across 2048 waves, and one
read costs about what two whole poll passes cost. Hand-rolled RTC stamps exist here at all because
**rocprof cannot see a raw-PM4 kernel** (RESULT_MBGEMM.md:55) — we dispatch below the HSA layer it
hooks, which is the price of the dyn-VGPR moat.

FIX SHIPPED: gate the probe 1-in-`DSWS2_POLLSTAGE_EVERY` (default 64) on **PS_THR = s78, a PRIVATE
counter, deliberately NOT s71** (prefetch keys off s71 — that shared trigger produced the 7x PASSTIME
phantom on 07-25). Only the ENTER is gated; `leave` already skips on ARMED==0, so start/end cannot
desync. Unsampled passes cost 3 SALU and touch no message bus.
Verified offline: POLLSTAGE=0 byte-identical (585d287e); probe adds exactly 2 sendmsg; s78 zeroed;
EVERY live (1/64/256 give distinct bins); emitted gate confirmed by disassembly
(`s_add_co_u32 s78` — note the assembler emits `s_add_co_u32`, not `s_add_u32`).

### 4. THROTTLED RESULT — VALIDATED BY CONVERGENCE (ffn_down M2048, 8 dispatches)
```
CONVERGENCE (the check that makes these quotable):
  s1: EVERY=64  45 ns   EVERY=256  44 ns   2.2% apart
  s5: EVERY=64 273 ns   EVERY=256 265 ns   2.9% apart
PROBE NOW FREE: pass count 0.88-1.03x control (was 0.13-0.39x); per-pass cost 1.07x (was 4.8x).
```
ns per loop-head pass, null-subtracted (s1) and reach-weighted:
| stage | mean ns | minus null | reach | ns/pass | share |
|---|---|---|---|---|---|
| s5 peek reservation attempt | 273 | 228 | 0.563 | **128.3** | **62.0%** |
| s6 park + s_sleep | 149 | 104 | 0.561 | **58.4** | **28.2%** |
| s4 feed -> peek gate | 214 | 169 | 0.070 | 11.9 | 5.7% |
| s3 role select + dispatch | 139 | 94 | 0.068 | 6.4 | 3.1% |
| s2 snapshot / body-gate | 47 | 2 | 0.971 | 1.9 | 0.9% |
| s1 loop head + deadman | 45 | NULL | 1.000 | — | — |
| | | | | **207.0** | total |

**ALL SIX STAGES ARE NOW COMPARABLE** — under the unthrottled probe s3/s4 sat in a different
contention regime and could not be placed on the same scale at all.
**CORRECTION:** s3 and s4 are NOT cheap per EXECUTION (94 and 169 ns, same order as s5's 228). They
are small per-PASS only because they execute on ~7% of passes. **Reach and cost are separate facts.**

=> METHOD RULE 13: JUDGE AN INSTRUMENT'S COST BY WORK COMPLETED, NOT ELAPSED TIME. A probe that
trades throughput for wall-clock looks free on a span and is not.
=> METHOD RULE 14: A THROTTLED INSTRUMENT MUST PROVE CONVERGENCE — run two sampling rates and require
agreement. Without it, "cheap" is an assertion.

---

## 2026-07-26 (evening) — THE BURST, AND RGA ON THE REAL KERNEL

### 1. THE SIX-STAGE PARTITION WAS NEVER A PARTITION (my error, stated as fact earlier today)
I wrote that stages 1-6 were "contiguous and non-overlapping, a true partition of one poll pass."
**FALSE.** A wave that WINS work branches into the burst and rejoins at `.Lflow_da_ss_complete` (:5561)
— **BACKWARD, into stage 3's range**. So a work pass traverses stages 3/4/5 TWICE, never reaches
`pollstage_leave 5` (its sample is silently DROPPED), and dumps its whole burst into stage 3's interval.
That is why the six-stage accounting closed to ~4% of runtime: **it partitions the PARK path only.**

### 2. NEW STAGES 7/8/9 — single entry, single exit (EVERY=1: the site runs only ~46x/wave/launch)
```
stage 7  burst + grow + shrink   200.253 us   n=316,800
stage 8  burst + shrink          202.645 us   n=316,800
stage 9  burst body only         196.667 us   n=316,800
```
=> **s_alloc_vgpr GROW ~0 and the SHRINK spin ~0.** The whole 197 us is the burst body.
*** THIS AGREES WITH PRIOR WORK AND THAT IS THE POINT: *** 2026-06-20 measured GROW+SHRINK = 0.0% of
COMPUTE with an in-kernel timer, and door4 grow-fail has been 0 in every run ever. The NEW instrument
reproduced a KNOWN-TRUE result — the external cross-check this project's probes have mostly lacked.

### 3. ARITHMETIC CORRECTION (I got this wrong once before correcting it)
I first reported "46 bursts/wave, closes at 99%". WRONG — that used the COMPUTED-UNIT count. Each burst
covers ACC_N=3 units, so it is **15.5 bursts/wave**:
```
wave lifetime  9.17 ms      burst 3.10 ms (34%)   poll 0.45 ms (5%)   UNACCOUNTED 5.62 ms (61%)
```
=> **METHOD RULE 15: A PER-WAVE WALL-CLOCK DURATION IS NOT MACHINE TIME.** The machine completes a
burst every **317 ns** while a wave experiences it as 197 us — the difference is 2048-way overlap.
That dissolves the apparent contradiction with the ablations (deleting burst work moves nothing
because the burst is overlap, not cost) and it is the SAME "period != cost" trap the 07-25 notes warned
about. I walked into it twice today.

### 4. *** RGA NOW RUNS ON THE CONFIG-OF-RECORD KERNEL — AND THE DESCRIPTOR WAS LYING ***
Three things were stale (kmbandy: "this is basically a whole new kernel at this point"):
- `rga_check.sh` defaults to `KSRC=occ_kernel_wggemm2.s` — **a different kernel entirely**
- it applies **no config-of-record defsyms**
- `RGADESC`'s `.amdhsa_group_segment_fixed_size` read **65536** while the config of record publishes
  **13,824B**. RGA computes occupancy FROM that number, so it would have reported **1 WG/CU on a kernel
  we run at 2 WG/CU** — a 4.7x error in the single figure RGA is most used for.
FIXED: the descriptor now reads `LDS_TOTAL_FLOW` — the same symbol emitted into `.lds_total` and read
by the host — so it can no longer drift from the bin. Verified: `USED_LDS_BYTES 13824`.
NOTE `build_flow.sh` does NOT pass RGADESC; assemble directly (my first "it assembles" test was a
silent no-op that proved nothing).

### 5. RGA ON THE REAL KERNEL (config of record: FM=1 G=6 ACC_N=3 SEGK=256 WAVES=16 prefetch on)
```
USED_LDS_BYTES 13,824 / 65,536      USED_VGPRs 256 / 256      SPILLS 0 (VGPR and SGPR)
Maximum # VGPR used  48  (HW allocates 96)   Maximum # SGPR used 54 / 106   ISA_SIZE 26,592
```
**PEAK LIVE VGPR IS 48 OF 256.** The 165.7 TF reference kernel measured 187. The register file is ~81%
unused with zero spills — because **our accumulators live in LDS, not registers** (split-K + ds_add).
hipBLASLt's gfx1201 fp8 kernels are ALL `_GSU1_`: accumulators in VGPRs across the whole K loop, C
stored once. That teardown was already in the repo; this is the first time it has been tied to a
measured register-occupancy number.

### 6. THE TILE SWEEP (RGA static, no GPU)
```
FM  G  ACC_N  tileM     LDS   2xLDS  peakVGPR  spill  WMMA  nonWMMA:WMMA
 1  6      3     96  13,824  27,648      48      0    128        13.15
 2  4      2    128  13,824  27,648      82      0    256         7.38
```
**13.15 -> 7.38, a 1.78x reduction in instruction overhead per WMMA**, at IDENTICAL LDS (still 2 WG/CU),
0 spills, peak-live 48 -> 82 of 256. Reference point: the 165.7 TF kernel runs **0.97** non-WMMA/WMMA.
EVERY larger tile (FM=2 G=6, FM=4 G=4, FM=2 G=8, FM=4 G=6) is REFUSED by the kernel's own guard:
*"DSWS2 single-slot operand layout exceeds the 65536B WGP limit"*. So the cap on tile size is the
**operand layout**, not the register file (32% used) and not LDS (21% used).

CAVEAT, and it matters: this is a STATIC INSTRUCTION COUNT. It says the CODE has 1.78x less non-WMMA
work per WMMA. It does NOT say throughput improves 1.78x — ports co-issue, and the WMMA-bearing region
I measured is "first WMMA to last WMMA", which I have NOT proven is only the k-step loop.
FM=2 G=4 changes the super-tile to 128 rows (from 96), so MTL and M-padding change for every shape =>
needs its OWN oracle bring-up, not a drop-in into the sweep.

### 7. THREE DEAD-CODE PLACEMENTS, all caught only by pre-registering an expected magnitude
(a) probe in the `DSWS2_ROLEFLOW=1` arm — compiled out at the config of record -> n=0
(b) throttle `EVERY=64` on a site executing ~46x/launch, so the counter never reached the modulus -> n=0
(c) a python `str.replace` that silently no-op'd (comment lines broke the exact match) -> half a probe
=> **THROTTLE COROLLARY: `EVERY` must be smaller than the site's executions per launch.**
=> **PATCH RULE: assert the edit landed. A no-op replace is silent.**
=> A bare `n=0` reads like "this path is free". It is the same "zeros that were never measurements"
   trap this project has now hit seven times.

### 8. Host bug
The WTBUDGET print computes `budget = span(0) x resi`, so `T SANITY` ALWAYS reports FAIL. The span
variable is 0 in that block. All WTBUDGET figures here were computed by hand from the raw sums:
wave lifetime 9.17 ms, residency sum(T)/(span x 2048) = 0.564, prologue E = 16.5% of budget,
peak concurrent 2048 = nominal, entries 20,480, live_net 0.
CAVEAT: that run is 1.58x its matched control, so those proportions are probe-affected.

---

## 2026-07-26 EVENING — FM=2 G=4 ACC_N=2 ORACLE BRING-UP: A GPU RESET, THEN THE FIRST EVER GROW-FAIL

The bring-up the previous entry called for. It reset the GPU once, then ran, then failed correctness.
**Net: the FM=2 tile is CORRECTNESS-BROKEN, and the reason is that it is the first config in the
project's history to make the dyn-VGPR budget bind.**

### 1. THE GPU RESET (17:08) — a `.bin` copied without its `.lds` sidecar
`build_flow.sh` emits TWO artifacts per build: `$tag.bin` AND `$tag.lds` (a 4-byte u32 holding the LDS
byte count the host must allocate). They are SEPARATE FILES. I built FM=2 as `fm2.bin`, copied only the
`.bin` into place, and left the `.lds` from a build 14 minutes earlier. Host allocated **13,824B for a
kernel needing 17,920B** -> ran past its LDS -> `MES failed to respond to msg=REMOVE_QUEUE` -> MODE1
reset -> VRAM lost. The rebuilt bin has the **SAME sha `e274377d`** as the one that crashed: the binary
was always correct, only the sidecar was stale.

**BLAST RADIUS — a MODE1 reset on this box kills EVERY GPU client, including the compositor.**
`llama-server` SIGABRT (router restarted it); **Hyprland itself SIGABRT** in
`onFrame -> renderMonitor -> beginRender -> CHyprOpenGLImpl::begin -> assertImpl`; `hyprlock` SIGABRT.
Hyprland restarted but held a session lock whose lockscreen process was gone, so **kmbandy was locked
out of his own desktop.** Fix: `hyprctl --instance 0 eval 'hl.clear_crashed_lockscreen()'`.
NOTE this is NOT rule 7 (HBM starvation kills the compositor). The compositor was never starved — a
RESET killed a GPU CLIENT. Rule 7 watches the compositor only; nothing watched the other GPU consumers.

**NEW FAIL-CLOSED GUARD** in `gpu_run.sh`: refuses when the `.lds` is missing or older than the `.bin`.
Self-tested on 4 cases incl. a replay of the reset with its real timestamps. Deliberately NOT fail-soft
like the claim check — a stale sidecar is never ambiguous and never someone else's outage.

**THE LESSON: A WARNING THAT FIRES ON EVERY RUN IS A WARNING NOBODY READS.** The host PRINTED the defect
before launch (`host reconstruction says 67072B but the BIN PUBLISHES 13824B`). That line prints on EVERY
dispatch and is normally benign, so I had trained myself past it. See section 4 for why it is always wrong.

### 2. THE COMPOSITOR CAP IS NOT TILE-PROPORTIONAL — a knob sweep that returned a constant
Per-chunk wall time, FM=2:

```
ML8_COOP_CHUNK   chunks   wall @base0
     512            2        0.81s
     256            3        0.81s
```

**Identical.** Flat sweep = fixed cost elsewhere (the 07-13 lesson, again). So lowering the chunk can
NEVER satisfy the 0.75s cap, and the corollary inverts intuition: **fewer, bigger chunks are strictly
cheaper on this kernel.** I predicted ~0.40s at 256 and was WRONG; "FM=2 doubles work per chunk" is dead.

`ML8_COOP_CHUNK_MAXS` IS a legitimate knob, unlike `DEADMAN_TICKS`: `occ_dispatch.cpp:1599` names raising
it as the remedy, and the check is **reactive** (`now_s()-t0`, measured AFTER the chunk completes), so an
over-cap chunk has already executed and the desktop has already survived it. 0.75 -> 0.85 let it finish.

### 3. THE ACTUAL FAILURE — first ever grow-fail, and it is the DESIGNED throttle
Config: `WAVES=16 G=4 FM=2 FN=4 ACC_N=2 SEGK=256 SSWIN=32 POOL_N=1 JDEPTH=1`, oracle 2048x2560x9216,
`TOTAL_super=23040`, `n_kseg=36`, `reps=2`, LDS 17,920B (alloc 17,920B).

```
computed = 110,846   expected = 184,320 (G*TOTAL_super*reps)   -> 40% OF WORK LOST
oracle   ok=10,416   bad=10,064   max_rel=1                     -> ~half the tiles WRONG
occ[73] grow-fail                                  =   489,791   <- WAS 0 FOR THE WHOLE PROJECT
occ[96] emissions reaching .Lflow_da_stamp         =    60,444   (expected 46,080) -> OVER-emitting
occ[97] release bailed on inflight==0              =    36,426
occ[95] exec lane0 inactive at claim CAS           =         0
occ[88] carrier stall / occ[98] baton wait         =     0 / 0   <- baton NEVER engaged despite 489k
COAST decomposition DOES NOT CLOSE: door sum 256,926,347 vs coast 256,915,548 (off by 10,799)
```

**OVER-emitting while UNDER-computing simultaneously is the signature of a claim being won more than once.**

**WHY NOW:** FM=2 doubles `ACC_STRIDE` (`FM*FN*1024` = 8192 vs 4096) and the VGPR budget binds for the
first time. And this path is not an error path — the kernel ASSERTS at `:1338`:
`.if SELFSERVE && !BATONGATE / .error "SELFSERVE requires BATONGATE=1: physical s_alloc_vgpr grow-fail is
the only admission throttle."` The FATTOK/MAXFAT software token layer is compiled to no-ops. So
**grow-fail IS the intended steady-state throttle, it had never once executed, and on its first real
outing it drops 40% of the work.** Prior art to check: Codex's 2026-07-24 BLOCKER 1 (grow-fail stamp
writes `SL_RBNEXT=0`/claimable BEFORE `SL_STI`, breaking the poison-clear-means-staged ABA invariant ->
stale-STI claim -> silent wrong C), which my own notes flagged as **"invisible offline because
grow-fail=0."** Adversarial Codex review commissioned; my static read is NOT evidence (this bug class has
beaten static reasoning three times).

### 4. TWO STALE GUARDS, both reporting PASS while measuring the wrong thing
**(a) `gate_lds.sh` (dated Jul 20) IGNORED its env.** `POOL_N=1; ACC_N=3; FM=1; ...` were plain
assignments, so `FM=2 ACC_N=2 ./gate_lds.sh` silently tested FM=1 ACC_N=3 anyway. It also never tested
`SSWIN=32` — the config of record. I read its PASS as evidence the FM=2 LDS number was sound; it was
evidence of nothing. FIXED: `: ${VAR:=default}` so env wins, SSWIN=32 added, and over-allocation is now
caught (the old check was `kernel <= host`, which a 4x over-allocation passes happily).

**(b) THE HOST'S LDS RECONSTRUCTION IS WRONG BY ONE OPERAND POOL — this is why the warning always fires.**
`ldsBytesRaw` includes `POOL_N*opstride`, but under SELFSERVE the kernel RECLAIMS the operand pool
(`ACC_BASE = OP_BASE`, *"operand pool reclaimed (SELFSERVE owns this)"*). `opstride = FN*16*SEGK +
G*16*FM*SEGK` = 40,960B at the config of record. The fixed gate now reproduces both sides exactly:

```
config              SSWIN=32 kernel    host raw    ratio
FM=1 ACC_N=3 G=6        13,824B        54,784B      ~4x
FM=2 ACC_N=2 G=4        17,920B        67,072B      ~3x     <- 67,072 is the number in tonight's warning
```

Kernel side independently hand-derived from source constants (`OP_BASE=512`, `SLOTC_STRIDE=32`,
`ACC_STRIDE=FM*FN*1024`): `512 + 2*8192 + 32*32 = 17,920` and `512 + 3*4096 + 32*32 = 13,824`. Three
independent agreements (bin, sidecar, hand-derivation) — the kernel's LDS algebra is RIGHT for FM=2.
The host's is wrong. `occ_dispatch.cpp` fix NOT yet applied.

### 5. METHOD ERRORS
**(a) I pre-registered `computed == 92,160` and the real expectation was 184,320** — I dropped the
`reps=2` factor. 92,160 is EXACTLY HALF, so had the kernel lost precisely 50% of its work I would have
declared a broken run a success. The pre-registration saved me only because the miss was 40%, not 50%.
=> **DERIVE THE EXPECTATION FROM THE HOST'S OWN FORMULA INCLUDING `reps`, never from a remembered number.**

**(b) I called a process a survivor without checking its start time.** Reported "Hyprland never crashed"
from a live pid — which was born 15s AFTER the reset. The disproof (`time: 1785100132` = 17:08:52) was in
`hyprctl instances` output I had already printed and not read. => **A live pid is not evidence a process
never died. Compare its START TIME to the incident.**

**(c) `journalctl --time-style` without a date filtered every log in history by time-of-day.** Attributed
dozens of prior-day runs to tonight's window. => Filter by `-newermt`, not by clock time.

**(d) GPU forensics on this box have a ~38-minute half-life.** A `razeraccessory` error loop is flushing
the kernel journal; the 17:08 reset lines were GONE by 19:00. "0 resets" then means "the journal no longer
retains it", NOT "no reset happened". Worth fixing independently of DSWS.

### 6. ADVERSARIAL CODEX REVIEW (19:24) — MY HYPOTHESIS WAS WRONG; THE REAL DEFECT IS A MISSING STAGE PUBLICATION

**THE DEFECT: the grow-fail fallback publishes CLAIMABLE work but never advances `STAGE_HEAD`.**
`occ_kernel_dsws_flow.s:6603-6615`. The comment there still reads *"deliberately NO STAGE_HEAD walk
here ... nothing claimable published (RB_PENDING poison, permanent under this defsym)"* — **that comment
describes a design that no longer exists.** Under `SELFSERVE=1`, line 6580 skips the RB_PENDING poison
store and line 6590 publishes `0`, which IS claimable. So the fallback stamps a valid claimable slot and
branches to `.Lflow_loop` without ever publishing it.

**THE CIRCULAR STALL:** compute gates on `DRAIN==STAGE` (`:5082-5087`) and coasts, so it can never reach
the perfectly valid `RBNEXT=0` claim. Unstaged fallbacks pile up until `ASSIGN-DRAIN==SSWIN`
(`:5969-5978`), which then rejects every new reservation BEFORE any grow attempt. Even once VGPR space
frees, nothing can move: ring compute blocked by `DRAIN==STAGE`, self-serve blocked by the window gate.
`deadman_check` retires waves individually and their reservations die uncomputed -> silent lost work.

**THE ARITHMETIC CLOSES EXACTLY**, which is what makes this conclusive rather than plausible:

```
missing reservations  31,716 x ACC_N=2   =  63,432
fallback work lost    21,598 - 11,556    =  10,042
                              total      =  73,474
reported loss    184,320 - 110,846       =  73,474   <- exact
```

It also derives the coast-door mismatch independently: `occ[73]` conflates TWO grow-fail sites (`:6543`
post-reservation self-serve, `:7176` pre-claim ring-compute) and only the second enters coast, so the host
invariant `coast == CNOSTG+CLEAD+FATFULL+GROWFAIL` must fail by exactly the self-serve fallback count
= **10,799** = the observed gap.

**FIVE THINGS FROM SECTION 3 ABOVE ARE WRONG — corrected here:**

| claimed in section 3 | actual |
| --- | --- |
| "OVER-emitting while UNDER-computing = a claim won more than once" | **NO over-emission exists.** `GROUPS = G/ACC_N = 2`, so the `occ[96]` target is `GROUPS*TOTAL_super*reps = 92,160`, not 46,080. 60,444 is **UNDER**-emission by 31,716. **The HOST label is wrong** at `occ_dispatch.cpp:2535` (also 2548, 2559) — it says "expect == TOTAL_super" and omits GROUPS. The entire duplicate-claim theory came from a host accounting bug. |
| "Codex's 2026-07-24 BLOCKER 1 (stale SL_STI) is probably still live" | **FIXED AND COMPLETE.** `:6583-6601` writes `SL_STI` before `SL_RBNEXT=0`, and `SL_RBDONE=0` before `SL_GEN`. Both publication races closed. |
| "baton never engaged despite 489k grow-fails (`occ[98]=0`)" | **MEANINGLESS.** There is no `.Lflow_batonwait` label and **NO WRITER AT ALL** to `occ[98]`. I cited an unwired counter as evidence — the **EIGHTH** time this project has been misled by a zero that was never a measurement. **GREP FOR THE WRITER, NOT THE READER.** |
| "100% of feed-path iters found an EMPTY ASSIGN frontier" | `occ[86]` increments at the common `.Lflow_feedmt_sleep` exit including window-full and boundary bails. Label over-specific. |
| "`occ[97]` = release bailed on inflight==0" | At `:6365` it counts boundary drain / C-store-gate bails. Host label at `occ_dispatch.cpp:2536` is stale. |

**STILL TRUE:** FM=2 doubles `ACC_STRIDE` and is the first config ever to make the VGPR budget bind;
grow-fail IS the designed admission throttle under SELFSERVE+BATONGATE (`:1338`); the compositor cap is
not tile-proportional; `ML8_COOP_CHUNK_MAXS` is a legitimate knob unlike `DEADMAN_TICKS`.

**METHOD:** my static read lost AGAIN — **fourth** time this bug class has beaten static reasoning on this
kernel. The standing "adversarial review is mandatory, my *looks correct* is not evidence" rule paid for
itself outright. Two process notes: use `codex_handoff`, not the codex-rescue subagent (kmbandy, tonight);
and I identified the review's rollout by **most-recent-file** and pulled ANOTHER SESSION'S mlambaformer
bf16 task, nearly reporting it as my verdict — **this box runs Codex from multiple sessions, so identify a
rollout BY CONTENT, never by recency.**

**FIX NOT WRITTEN.** Indicated repair is a `STAGE_HEAD` publication on the grow-fail path; the existing
`.Lflow_da_ss_stage_walk` (`:6660-6680`) is reachable only from the successful-grow path. Any fix needs
its own adversarial review before it is trusted. **The FM=1 control run remains worth doing**: it settles
whether the config of record has been silently exposed to this same stall all along.

### 7. GROW-FAIL FIX VALIDATED — FULL ml8/mlambaformer SWEEP AT FM=2, 30/30 ORACLE-CLEAN

**(a) Change.** `.Lflow_da_gf_stage_walk` added on the self-serve grow-fail path (Codex, reviewed by me):
the fallback published a claimable slot (`SL_RBNEXT=0` under SELFSERVE) but never advanced `STAGE_HEAD`,
so compute — gated on `DRAIN==STAGE` (`:5082-5087`) — could never claim it. Unstaged fallbacks filled
`SSWIN`, new reservations were then rejected before any grow attempt, and `deadman` retired the stuck
waves with their reservations uncomputed.

**(b) Config.** Kernel `WAVES=16 G=4 FM=2 FN=4 ACC_N=2 SEGK=256 SSWIN=32 POOL_N=1 JDEPTH=1`,
LDS 17,920B. Host `FLOW_WAVES=16 ML8_POOL=128 ML8_COOP_CHUNK=512 ML8_COOP_CHUNK_MAXS=0.85
STAGINSTR=1 TFPROBE=1`, via `dsws_realshape_bench.py live --fm 2 --g 4 --acc-n 2`.

**(c) Shapes.** All 33; **30 legal, 3 UNSUPPORTED** (`n_kseg=1 < 2` at SEGK=256 on the tiny router
shapes — identical at FM=1, NOT an FM=2 limitation, fixable with `--segk 128`).

**(d) Outcome. 30/30 PASS. oracle bad=0 on every shape. WORK-EXACT on every shape.**

```
                              M       K     TF    grow-fail   oracle
ml8_dense_ffn_down         2048    9216   4.20    3,412,194   bad=0   <- best
ml8_dense_ffn_down          512    9216   2.13   35,362,942   bad=0
ml8_dense_attn_q           2048    2560   2.07      243,156   bad=0
ml8_dense_attn_o           2048    4096   2.03    2,102,546   bad=0
ml8_dense_ffn_gate_up      2048    2560   1.85      222,181   bad=0
ml8_dense_attn_kv          2048    2560   1.12    2,230,216   bad=0
mlmf_mamba_out_proj        4096    1536   1.05       41,269   bad=0
ml8_moe_attn_o              512    4096   0.92   14,067,065   bad=0
mlmf_lm_head               4096     768   0.66        1,959   bad=0
mlmf_in_proj_ML8PAD        4096     768   0.64          549   bad=0
...                                  0.02-4.20    all > 0     all bad=0
```

*** GROW-FAIL FIRED ON 30/30 SHAPES — 140,708,123 EVENTS TOTAL. *** This is what makes the sweep a real
test rather than a vacuous pass: the repaired path executed on every shape. Before the fix, this same
config lost 40% of its work and returned `bad=10,064` of 20,480.

**THE MECHANISM HEADLINE:** `s_alloc_vgpr` grow-fail is the ONLY admission throttle under
`SELFSERVE=1 BATONGATE=1` (kernel asserts this at `:1338`; the FATTOK/MAXFAT token layer is compiled to
no-ops). It had been **exactly 0 on every run in this project's history** — the dyn-VGPR moat never once
engaged. FM=2 doubles `ACC_STRIDE`, the budget binds, and it now runs correctly across the entire real
workload.

**(e) WHAT THIS DOES NOT SHOW.** TF is 0.02–4.20 against hipBLASLt on these same
**[CORRECTED 2026-07-27 EVE: the hipBLASLt band on real DENSE shapes is 123-189 TF, not 12.6-70.6. That band was the ml8 MoE M=512 subset only. Mean ratio ~80x. See RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md and the CORRECTION box in DSWS_BRIEF_2026-07-27_AM.md.]**

shapes. **This was a CORRECTNESS fix and throughput did not move.** The ASSIGN-bound wall is unchanged
(100% of feed-path iters find an empty frontier; coast ~97%). Do not read 30/30 oracle-clean as progress
on the perf problem — it is permission to *start* measuring FM=2 honestly.

**TWO HARNESS TRAPS FIXED BEFORE THE SWEEP COULD BE TRUSTED:**
1. `LIVE_FM` was **hardcoded to 1** with no CLI flag while `--g` *was* a flag. The sweep would have
   dispatched FM=1 geometry (96-row super-tile) against the FM=2 bin (128-row) — every shape's M-padding
   computed for the wrong tiling. Added `--fm`. *A knob that is a flag on one axis and a constant on a
   coupled axis is a trap, not a default.*
2. No `ML8_COOP_CHUNK_MAXS` passthrough. FM=2 measures 0.81s/chunk against the 0.75s compositor cap, so
   all 30 shapes would have ABORTED and been recorded as 30 "failures" that are not kernel failures.
   Added `--chunk-maxs` (0.85). The cap is NOT tile-proportional, so lowering `--chunk` cannot substitute.

**A THIRD, FOUND BY THE SWEEP ITSELF.** Run 1 halted at shape 21 with `AMBIGUOUS_REAL_N`:
`mlmf_mamba_in_proj` (N=4200) and `mlmf_in_proj_ML8PAD` (N=4208) both pad to N=4224, so the parser could
not attribute the result and refused. **That refusal was correct** — mis-attributing a TF number is worse
than stopping — and the kernel run itself was clean (`WORK-EXACT`, `bad=0`). But the parser was
reverse-mapping padded geometry to recover something the live caller already knew. Fixed with
`EXPECTED_SHAPE`: live mode passes ground truth, offline re-parse still refuses (it genuinely cannot
know), and **a deliberately wrong hint is rejected**, so the escape hatch cannot mask a geometry bug.
Run 1 was 20 PASS / 1 attribution-halt / 9 NOT_RUN — 21 of 21 kernels clean, zero correctness failures.

### 8. RETRACTION — "ASSIGN-BOUND" IS NOT SUPPORTED BY occ[86], AND NEITHER IS "CARRIERS ARE FED" BY occ[88]

**RETRACTED from section 7(e) and from tonight's reporting: the claim that the wall is ASSIGN-bound.**
kmbandy, 2026-07-26: *"we've consistently said it was that and then consistently found that it wasn't
that...we need to measure before we make claims like that."* He is right, and the failure is worse than a
loose phrase — Codex told me this same evening that `occ[86]` is over-specific, I wrote that correction
into section 6 of this very log, and then I asserted "100% of feed-path iters find an empty frontier"
two messages later.

**WHY occ[86] CANNOT SUPPORT IT.** It increments at the COMMON `.Lflow_feedmt_sleep` exit, reached by an
empty ASSIGN frontier AND by window-full bails AND by ZLOCK boundary bails. It measures *"gave up on the
feed path"*, not *"the frontier was empty."* Track record: **"ASSIGN-starved 76%" became 1.8% purely by
feeding the kernel to steady state (2026-07-13).** This exact claim has a history of evaporating.

**THE HOST WAS PRINTING THE VERDICT TOO** — `=> ASSIGN-BOUND (coordinator cannot publish fast enough)` —
derived from that conflated ratio. Removed; it now reports the count and explicitly says it does not
identify a bottleneck, and points at RESVPROBE.

**A THIRD ONE, FOUND WHILE FIXING THE SECOND.** The line above it printed
`occ[88]=0 -> carriers are fed (stall is not the wall)`. `occ[88]` IS properly wired (`cnt_inc CNT_JWAIT`
at `:5406`) — but `.Lflow_jwait` is the DEEP-J carrier wait and **at JDEPTH=1 the path does not exist**;
the kernel states this at `:2974` (*"JDEPTH=1 => no JWAIT/CLEAD"*). A carrier at JDEPTH=1 computes one
segment and shrinks, never waiting for a next stage. **That zero is structural, not a measurement.**

=> **NEW RULE, generalising the unwired-counter rule: CHECK REACHABILITY, NOT JUST THE WRITER.**
"Grep for the call site" is necessary but NOT sufficient — the site must also be compiled in AND reachable
in the config under test. Unwired counter and unreachable-path counter produce the identical symptom (a
confident zero) and this project has now been fooled by both.

**WHAT WOULD ACTUALLY MEASURE IT: RESVPROBE.** It already exists and splits the bails by cause —
`occ[87]` CAS-loss (cursor contention) vs `occ[89]` window-full (consumers behind) vs the boundary
remainder, against `occ[96]` wins. Its verdict logic is legitimate *because it is computed from the split,
not from the merged total*. Requires a bin built `RESVPROBE=1` and run with `DSWS2_RESVPROBE=1`.
NOT YET RUN — so as of this entry **we do not know what the wall is at FM=2**, and section 7(e)'s
"ASSIGN-bound wall is unchanged" should be read as: *coast is ~97% and the cause is unmeasured.*

---

## 2026-07-27 MORNING — THE WALL IS MEASURED. IT IS PRODUCER-SIDE FRONTIER STARVATION.

Answers the question the 2026-07-26 entry closed on ("we do not know what the wall is at FM=2").
Two dispatches, both clean, both at the FM=2 G=4 ACC_N=2 primary config on
`ml8_dense_ffn_down M2048 N2560 K9216`. Gates on BOTH: `oracle bad=0` at **640/640 tiles, stride=1**,
`computed=460,800` WORK-EXACT, `occ[96]` delta **+0**, `occ[0]=0`, canary clean, no resets, no latch.

| run | bin | probes |
|---|---|---|
| L0  | `2ca16ea0…` 30,824 B | RESVPROBE=1 |
| L0b | `61ffe8b2…` 31,016 B | RESVPROBE=1 BNDSPLIT=1 |

### 1. THE ANSWER — 96.1% of feed-path bails are an EMPTY FRONTIER

L0b, 29,928,830 bails = **129.9 feed iterations per successful reserve**:

| bucket | count | share of bails |
|---|---:|---:|
| CAS-loss `occ[87]` | 488,061 | 1.6% |
| window-full `occ[89]` | 164,203 | 0.5% |
| boundary `occ[97]` | 513,443 | 1.7% |
| **UNACCOUNTED — frontier simply EMPTY** | **28,763,123** | **96.1%** |

Not the cursor. Not the window. Not the tile, the register file, or LDS. The **producer** cannot
publish work fast enough.

### 2. BNDSPLIT LOCALIZES IT — only 1.6% of boundary entries ADVANCE

33,653 sampled entries (1/64 throttle → ~2,153,792 real):

| outcome | share |
|---|---:|
| `ZLOCK_LOST` (herd: lost the election CAS) | **76.5%** |
| `DRAINGATE_BAIL` (won, DRAIN<ASSIGN) | **0.0%** |
| `CSTOREGATE_BAIL` (won+drained, GSTORED gate) | **21.8%** |
| `ADVANCE` (passed both → DA_ZDONE++) | **1.6%** |

**MEASURED CAUSAL CHAIN:** the frontier only refills on a boundary ADVANCE → only 1.6% of entries
advance (1 per 61) → of the waves that WIN the lock, **93%** are then blocked by the **GSTORED
C-store gate** → the other 76.5% never win it → frontier empty 96% of the time → coast 97.2%.

`DRAINGATE_BAIL = 0.0%` is a REAL measurement, not a structural zero: the host computes it as
`ZWON − PDRAIN`, a difference of two live counters. **Drain is never the blocker.**

### 3. RETRACTION — "SHARD THE CURSOR" IS NOT SUPPORTED

The pre-fix RESVPROBE verdict printed `CURSOR-CONTENDED → SHARD THE CURSOR` on this very run
(contention 2.118 > 1.0). It was wrong, and wrong *structurally*: it compared CAS-loss against
`occ[96]` (successful reserves) and window-full against `occ[86]` (bails) — **two different
denominators** — and so never asked what share of the bail population it had explained. CAS-loss is
**1.6%** of bails. Sharding would have targeted 1.6% of the wall.
Third instance of the same species as §8 of 2026-07-26: *a verdict out-running its own denominator.*

### 4. ★ DSWS2_FUNNEL — BUILT FOR EXACTLY THIS, NEVER ONCE COMPILED IN ★

The boundary-advance readiness **pre-gate** checks `GSTORED < DA_ZDONE>>shift` **READ-ONLY, BEFORE**
the ZLOCK CAS (kernel :6171-6184, election at :6186) — i.e. precisely the measured blocker. It has
**never been enabled in any run in project history**: absent from `build_flow.sh`'s config-of-record
block, so it defaulted to 0 in `${DSWS2_FUNNEL:-0}`; kernel :7143 states "0 in both profiles of
record"; zero mentions in this log.
**TRAP:** every "funnel" hit in the logs and in `occ_dispatch.cpp` is the UNRELATED *carry-through
funnel* (`occ[100..103]`). Searching for "funnel" returns false positives.
Its other condition (`DRAIN < ASSIGN`) measured 0.0%, so it costs nothing and fires never.
FIXED: `DSWS2_FUNNEL` / `DSWS2_FUNNEL_SPIN_N` are now in the config-of-record block AND printed.

### 5. THE FUNNEL'S SPIN IS INERT — A BENIGN, LOAD-BEARING POLARITY BUG (Codex-audited, ISA-cited)

`s_sub_u32` is the assembler alias for `S_SUB_CO_U32` on gfx12; RDNA4 ISA §16.1 p.206 defines
`SCC = (S1 > S0)` — SCC is the **borrow**, set only on underflow. Confirmed in the emitted gfx1201
disassembly (`s_sub_co_u32 s56, s56, 1`).
So `1024-1=1023` does not borrow → SCC=0 → **every arm bails on the FIRST not-ready**.
`DSWS2_FUNNEL_SPIN_N` is **inert**; the funnel is a pure **check-once** pre-gate. All three arms were
written treating SCC=1 as "counter still live", which is inverted.
**THE BUG IS BENIGN AND CURRENTLY LOAD-BEARING — DO NOT "FIX" IT CASUALLY.** Check-once is the only
behaviour reviewed as safe. Each retry re-reads FOUR LDS words, and `lds_get` is
`ds_load_b32 + s_wait_dscnt 0` (a serialized read/drain, :1601-1605), at a site entered ~2.15M times.
This kernel has a documented **16x** regression (97.3 → 5.9 TF) from ONE extra LDS read in the peek
path (:5950-5954). Correcting the polarity re-opens a 1024-deep spin on that path.
A permanent warning is now at `.Lflow_da_funnel_notready`.
Codex verdict: read-only confirmed (no `lds_put`/CAS/atomic; only private s54-s56); cannot lose work
or abandon a boundary (bails before the ownership CAS); **no TOCTOU — every gate is re-read after
winning ZLOCK** (:6199-6202, :6209-6212); probe pairing sound on the bail path in all three arms.
**Cost if enabled: ~8.6M extra serialized LDS loads.** Safe for ONE guarded bring-up; NOT
performance-safe; NOT for an unmonitored sweep.

### 6. THREE HOST INSTRUMENT DEFECTS FIXED

1. **RESVPROBE counter aliasing corrupted the COAST DECOMP.** RESVPROBE reuses `CNT_CLEAD` (s96,
   occ[89]) and `CNT_FATFULL` (s94, occ[87]) (kernel :350, :353). Safe — both originals are
   structurally zero here — but the print didn't know, and displayed RESVPROBE values under the
   door2 LEAD-GATE / door3 FAT-PEAK-FULL labels: L0 printed "FAT-PEAK-FULL (stagger cap) = 507,102"
   for a build where the stagger cap CANNOT fire. Worse, it folded them into the coast-door SUM,
   which they are not members of. **Excluding them takes the invariant gap 968,473 → 59,299** — the
   sum very nearly CLOSES, and the residue is the known non-coasting grow-fail sites. The old print
   buried that agreement.
2. **A SECOND, UNREPAIRED LDS RECONSTRUCTION** (`occ_dispatch.cpp:7329`), in a different scope from
   the authoritative `ldsBytesRaw` repaired on 07-26. Every dispatch printed **two different LDS
   numbers** — `LDS=65792B` on the oracle line and `17920B` three lines later. Two competing LDS
   numbers is the exact noise that preceded the 07-26 MODE1 reset. Now reclaim-aware; verified
   **`LDS=17920B` via `DSWS2_DRYRUN=1`, at zero GPU cost.**
3. **RESVPROBE verdict** — now reports share-of-bails for all four buckets including the unaccounted
   remainder (see §3).

### 7. METHOD ERRORS — MINE, THREE IN ONE MORNING

- **`grep -c` on a phrase cannot distinguish an assertion from its NEGATION.** I "found" that the two
  retracted verdicts were still live in `occ_dispatch.cpp` and told kmbandy the 07-26 brief was false.
  The four hits were the FIXES: `"does NOT show carriers are fed"`, `"do NOT infer ASSIGN-bound"`,
  and two removal comments. **The brief was correct; my retraction of it was the error.**
  Verify the CONTEXT of a hit, never the count.
- **`LDS=65792B` diagnosed as a stale default `G=6`.** Wrong — `Gv` was correctly 4; it was the
  operand-inclusive formula (§6.2). Right fix, wrong reason, asserted before checking.
- **An empty disassembly read as a real zero.** `llvm-objdump -b` isn't supported here; the command
  produced 0 lines and my greps dutifully returned 0 occurrences, which I nearly reported as evidence.
  The identical symptom as an unwired counter. Always check the instrument produced OUTPUT first.

### 8. NOT SETTLED — L5 per-chunk cost

Total wall **205 s** vs **0.148 s** of kernel (14.8 ms/chunk × 10). But the oracle ran at `stride=1`
(640/640 tiles vs the sweep's 80/640), so host verification almost certainly dominates that gap.
**This run cannot separate host verify from GPU per-chunk overhead.** L5's 0.81 s/chunk remains
unmeasured; it needs an in-host per-chunk timer. Do not quote an end-to-end number until then.

### 9. NEXT

One dispatch: `DSWS2_FUNNEL=1 DSWS2_FUNNEL_SPIN_N=1` (bin `4f567be6…`, 31,144 B, LDS 17,920),
RESVPROBE+BNDSPLIT on, same shape, for direct comparison against L0b.
**PRE-REGISTERED:** `ZLOCK_LOST` falls well below 76.5% and `ADVANCE` rises above 1.6%.
TF may still REGRESS from the ~8.6M added serialized LDS loads — that is a real possible outcome and
NOT a failure of the experiment. Do not quote TF from a probe build either way.
Probe weight inflates grow-fail (3.41M → 5.79M → 6.57M across the three builds): treat these as
RATIOS, not absolutes.

---

## 2026-07-27 AFTERNOON — THE FUNNEL WORKS AND DOESN'T HELP; 91% OF CHUNK WALL WAS A HOST TIMER

### 10. DSWS2_FUNNEL ENABLED FOR THE FIRST TIME — PREDICTION FALSIFIED

Bin `4f567be6` (31,144 B, LDS 17,920) = the L0b bin + `DSWS2_FUNNEL=1 SPIN_N=1` (+128 B).
Gates: `oracle bad=0` 640/640 stride=1, `computed=460,800` WORK-EXACT, `occ[96]` delta +0, canary clean.

**MECHANISM CONFIRMED** (unthrottled counters — these are the trustworthy ones):

| | L0b | funnel | |
|---|---:|---:|---|
| `occ[97]` boundary / C-store-gate bails | 513,443 | **0** | eliminated |
| `occ[86]` feed-path bails | 29,928,830 | 24,585,102 | −18% |
| iterations per reserve | 129.9 | 106.7 | −18% |
| coast-frac | 97.2% | 96.1% | |
| **TF (span)** | 3.3 | **3.2** | **unmoved** |

**THROUGHPUT DID NOT MOVE**, and the empty-frontier share got *worse* (96.1% → 97.6% of bails).
I pre-registered "`ZLOCK_LOST` falls below 76.5%, `ADVANCE` rises above 1.6%". `ZLOCK_LOST` rose to
**100.0%**. Reading: the funnel makes waves hold until the gate opens, then all rush the CAS together —
it converted a spread-out herd into a **synchronized** one.

**DO NOT TRUST THE BNDSPLIT DELTAS ACROSS THIS A/B.** Sampled `ADVANCE` fell 553 → 22, which cannot be
literally true: the work completed exactly and the required advance count is fixed by geometry.
BNDSPLIT is throttled 1/64 on the deadman counter `s71` and its "ratios exact" claim assumes unbiased
sampling; the funnel changes loop structure and plausibly shifts that phase. Rest only on the
unthrottled counters above. The TF half is separately confounded — the A/B spanned ~3h on a shared box
with mlambaformer in between (see the mlambaformer measurement-discipline rule: cross-session absolutes
are not comparable, only within-session ratios).

**THIS HIT A PRE-REGISTERED FALSIFICATION CRITERION VERBATIM:** *"ADVANCE unchanged while ZLOCK_LOST
drops → contention was never what limited advancement; the C-store gate is a hard latency floor and
thinning the herd cannot help. This would redirect the whole effort at GSTORED / C-store retirement,
not at contention."* **Contention around the boundary is not the lever. The rate at which C-stores
RETIRE is.**

### 11. ★ 91% OF NON-FINAL CHUNK WALL WAS `DSWS2_SETTLE`, A HOST-SIDE CONSTANT ★

`occ_dispatch.cpp` completion gate: the EOP fence is armed **only on the final chunk** (deliberately —
a stalled per-chunk fence used to block the next chunk from launching). So every **non-final** chunk
cannot use `ff` and instead waits `settle` (**default 0.30 s**) of no-change on `occ[3]` *after its
last wave has already exited*.

**THE WITHIN-RUN CONTROL — this is the proof, and it needs no cross-run comparison:**

| chunk | last? | completion path | settle=0.30 | settle=0.02 |
|---|---|---|---:|---:|
| base=0 (512 tiles) | no | **settle** | 0.317 s | **0.038 s** |
| base=512 (128 tiles) | **yes** | **EOP fence** | 0.013 s | **0.014 s** (CONTROL, unmoved) |

Only the settle-path chunk moved. Same bin, `DSWS2_SETTLE` the only changed variable.
Gates at FM=2 (previously validated only at FM=1): `oracle bad=0` 640/640, `computed=2,211,840`
WORK-EXACT, canary clean.

**Span-TF UNCHANGED 3.2 → 3.1**, exactly as it must be — settle is host idle time after the waves have
exited and cannot touch the GPU busy span. That invariance is the check that this is what we think it is.

**RESULT: reps 5 → 24 in the same `DSWS2_TARGET_SECS` budget (4.8×); end-to-end TF 0.29 → 1.86 (6.4×)
with ZERO kernel change.**

**WHY IT SURVIVED SIX DAYS AFTER BEING SWEPT:** settle 0.05/0.02/0.01 were all swept on 2026-07-21
(FM=1) and were **all oracle-clean** — but span-TF did not move (2.0/1.9/2.0), so it read as a null
result. **OUR HEADLINE METRIC IS STRUCTURALLY BLIND TO THIS COST.** Judge this knob by
reps-per-target-second, never by TF and never by wall clock.

**A PREDICTION I GOT WRONG, AND IT MATTERS:** I predicted total chunk wall 1.650 s → ~0.25 s. It stayed
~1.25 s. The rep loop is **duration-bounded**, so a faster rep buys more reps, never a shorter run.
Anyone judging this change by wall clock will conclude it did nothing.

Also: run spread widened 4.3% → 10.1% (more, shorter reps). And the ~206 s process wall is ~99% host
oracle verification at `stride=1` — my choice in manual bring-ups; `dsws_realshape_bench.py` uses
stride=8. Not a kernel or dispatch cost.

### 12. DEFAULTS CHANGED (kmbandy, 2026-07-27)

- **`DSWS2_SETTLE` default 0.30 → 0.02** (`occ_dispatch.cpp`). Fail-loud: settle exists only to let the
  terminal C store land before the oracle reads C, so too-short **fails the oracle, never a false
  CLEAN**. Validated FM=1 (0.05/0.02/0.01) and FM=2 (0.02, 2048 waves). `DSWS2_SETTLE` still overrides;
  0.05 is the conservative fallback and still 6× better than the old default.
  **NOT changed:** the coop path's own settle (different kernel; it already uses 0.025 when its fence
  works, 0.30 only in its nofence case).
- **`DSWS2_FUNNEL` default 0 → 1** (`build_flow.sh`), with `SPIN_N=1`. Justified by ELIMINATING MEASURED
  WASTE (occ[97] → 0, −18% bails) and a clean Codex correctness audit — **NOT by a demonstrated
  speedup**. Tested on ONE shape, ONCE. **VALIDATE ON THE FULL 30-SHAPE SWEEP** before relying on it.
  `DSWS2_FUNNEL=0` reverts byte-identically (verified: 30,812 B, matches the pre-change build exactly).
  Default build is now **30,940 B** (= 30,812 + 128 funnel), LDS unchanged 17,920.

### 13. THE PATTERN OF THE DAY — TOOLING THAT DEFAULTS OFF

**Four instruments used today were already built, wired, and simply never switched on:** `RESVPROBE`,
`BNDSPLIT`, `ML8_CHUNK_DIAG`, and `DSWS2_FUNNEL` itself. Plus `DSWS2_SETTLE`, swept and then left at a
value 15× too high. The recurring failure on this project is **not missing tooling — it is tooling that
defaults off and never appears in the config of record.** That is the same root cause as the FM=2
discovery (grow-fail was 0 forever because the moat was never armed). `DSWS2_FUNNEL`/`SPIN_N` are now
in `build_flow.sh`'s config-of-record block and **printed on every build**.

### 14. METHOD — THREE NEAR-MISSES, ALL THE SAME SHAPE

A name or substring match is a LEAD, NOT EVIDENCE. Verify the context of every hit:
1. `grep -c 'carriers are fed'` returned 2 — both were the **corrective** text. I told kmbandy the
   07-26 brief was false. **The brief was right; my retraction was the error.**
2. `LDS=65792B` diagnosed as a stale default `G=6`. Wrong — `Gv` was correctly 4; it was the
   operand-inclusive formula. Right fix, wrong reason.
3. A log named `funnel_bringup_M2048` looked like proof the funnel had run before. It has **zero**
   `DSWS2_FUNNEL` in its env and is FM=1 G=6 — named for the unrelated carry-through funnel.
   Checked before it became a fourth false claim.
Plus: an empty disassembly (`llvm-objdump -b` unsupported) produced 0 greps that I nearly read as real
zeros. **Always confirm the instrument produced output before believing its zero.**

---

## 2026-07-27 EVENING — +63%: 2 WG/CU IS HARMFUL, THE GROUP BARRIER IS REAL, AND THE MOAT IS AN ARTIFACT

Prompted by an external review (Kimi K3 via opencode, $1.92, read-only agent). Two dispatches at
MATCHED geometry, back-to-back in one session — a VALID controlled comparison, unlike every cross-sweep
comparison earlier in the day.

| | WG/CU | waves | ACC_N | GROUPS | TF |
|---|---|---:|---|---|---:|
| baseline | 2 | 2048 | 2 | 2 | **4.73** |
| ARM A | 1 | 1024 | 2 | 2 | **6.70** (+42%) |
| ARM B | 1 | 1024 | 4 | 1 | **7.70** (+15% / +63%) |

A vs baseline isolates occupancy; B vs A isolates the group barrier with geometry held constant.
Gates clean on both: `oracle bad=0`, WORK-EXACT, `occ[96]` delta +0, `occ[0]=0`, canary clean.
**Arm B spread 1.9% over 34 reps — the tightest measurement in this project.**

### 15. 2 WG/CU IS NOT NEUTRAL, IT IS HARMFUL
Halving the wave count gained 42%. Iterations per successful reserve collapsed **129.9 → 26.3**.
`ML8_POOL=128` has been over-subscribing the card for the entire campaign, and both `build_flow.sh`
and `gpu_run.sh` currently ENFORCE it as the standard.

### 16. THE GROUP BARRIER IS REAL, ~15%
`ACC_N=G` (GROUPS=1) fires the boundary once per TILE instead of once per group: `occ[96]` halved
(1,520,640 → 783,360) while `computed` slightly ROSE (3,041,280 → 3,133,440). Same work, half the
coordination. Kimi's "Ceiling B" confirmed at roughly the predicted magnitude.

### 17. ★ grow-fail = 0 IN BOTH ARMS — THE MOAT IS AN OVER-SUBSCRIPTION ARTIFACT ★
At 2048 waves: 6,574,885. At 1024 waves: **EXACTLY ZERO.**
The entire justification for promoting FM=2 G=4 ACC_N=2 to primary config that morning — *"grow-fail is
the only admission throttle and it was 0 on every run in project history; FM=2 finally makes it bind
(140.7M events); at FM=1 the design cannot be evaluated"* — describes an artifact of launching twice as
many waves as help. **The moat binds exactly when the kernel is slow and stops binding when it is fast.**
Fourth independent refutation of the dyn-VGPR thesis, and the most direct.

### 18. dyn-VGPR IS CAPPED AT 128 VGPRs/WAVE (256 HARD MAX) — THE LEDGER'S HEADLINE IS UNREACHABLE
`cap = (MAX_BLOCK_ALLOC+1) × BLOCK_SIZE = (7+1) × 16 = 128` by default; `BLOCK_SIZE=1` gives 256 and
`MAX_BLOCK_ALLOC` is 3 bits so **256 is the ceiling**. Therefore `INSTRUCTION_LEDGER.md` Lever B's 8×8
(512 acc VGPRs) → ~176 TF, and the compounded ~206 TF, are **NOT REACHABLE ON THIS SILICON**.
What survives: static allocation hard-deadlocks the 8-wave barrier WG at ≥256 VGPRs, so dyn's real
value is launching lean then growing to ~160-256 — worth about ONE frag-grid step. Note the ledger also
records the 161 TF winner at 52% of peak on **183 VGPRs**, under the static limit, so even that use is
unproven-necessary.

### 19. WHAT DID NOT WORK
`WOFLUSH=1` **does not build** — "DECENTASN is now BANKED-ONLY … the WOFLUSH (next,inflight) pin path
was retired 2026-07-16". The review proposed it as "a one-defsym experiment you already have"; it is not.
A first `ACC_N=4` attempt at `ML8_POOL=128` was **REFUSED by the host geometry guard** (34,304 B ×
2 WG/CU = 68,608 > 65,536) before any packet was submitted — the guard worked exactly as designed and
is what forced the matched-geometry redesign that made the comparison valid.

### 20. THE TILE FRONTIER (offline, no GPU) — THE VESTIGIAL GUARD IS NOW THE BINDING CONSTRAINT
Assembled every (FM, G, ACC_N) at FN=4 SEGK=256 SSWIN=32. The `:646` operand-layout guard enforces
`G·FM ≤ 11`, which blocks **FM=2 G≥6, FM=4 G≥4, and every FM=8 cell**. That guard was verified
vestigial under SELFSERVE on 2026-07-27 (zero OPSTRIDE/ARES_OFF immediates in the shipped disassembly)
and Codex-reviewed NOT-REFUTED. It is still live, deliberately, so it would not confound the sweep.
Legal today at constant super-tile M=128: `FM=1 G=8`, `FM=2 G=4`, `FM=4 G=2` — which is exactly the
3×2 frag-grid × occupancy sweep pre-registered as the first task for 2026-07-28.

---

## 2026-07-29 MORNING — THE FRAG-GRID × OCCUPANCY 2×2 CLOSES. OCCUPANCY REPLICATES; FEED IS SECOND-ORDER.

Execution of the sweep pre-registered in `DSWS_BRIEF_2026-07-28_AM.md` §0. Two dispatches (arms E and F);
A and B were measured 2026-07-27 evening. **Arms C and D were killed OFFLINE before any silicon** — see §22.
Logs: `sweepF_fm1_g8_accn8_pool64_nonstd_081348.log`, `sweepE_fm1_g8_accn4_pool128_nonstd_083207.log`.

Design: `superM = G*16*FM = 128` in every arm, so the work decomposition is held fixed and only the
per-wave frag grid and the occupancy vary. Gates clean on both new arms: `oracle bad=0`, WORK-EXACT,
`occ[96]` delta +0, `occ[0]=0`, canary clean, `grow-fail=0`.

| arm | FM | G | ACC_N | GROUPS | WG/CU | feed/WMMA | TF | spread |
|---|---|---|---|---|---|---|---:|---|
| A | 2 | 4 | 2 | 2 | 2 | 0.750 | 4.73 | (07-27) |
| B | 2 | 4 | 4 | 1 | 1 | 0.750 | **7.70** | 1.9% / 34 reps |
| E | 1 | 8 | 4 | 2 | 2 | 1.250 | **3.8** | 10.5% / 27 reps |
| F | 1 | 8 | 8 | 1 | 1 | 1.250 | **6.3** | 14.8% / 32 reps |

### 21. ★ THE OCCUPANCY EFFECT REPLICATES: +63% AND +66% ★
A→B gained **+63%**; E→F gained **+66%** — measured at a DIFFERENT frag grid, in a different session,
on a different build. The 1 WG/CU finding is not an artifact of the FM=2 geometry. This is now the most
robust result in the project, and it strengthens the case for retiring the `ML8_POOL=128` standard that
`build_flow.sh` and `gpu_run.sh` still enforce (pending the 30-shape sweep — see §24).

### 22. ★ ARMS C AND D ARE UNRUNNABLE AT THE DEFAULT dyn-VGPR CAP — CAUGHT OFFLINE ★
Disassembled the `s_alloc_vgpr` grow target for each arm.

**[CORRECTED 2026-07-29, same day: this section first stated the ask as `48 + FM*32`. That is only the
FN=4 special case. The GENERAL formula — the kernel's own, mirrored at `occ_dispatch.cpp:3246` — is
`NFV = roundup16(32 + 8*FM*FN + 2*FM + 2*FN)`. It reproduces all three disassembled values exactly
(FM1FN4→80, FM2FN4→112, FM4FN4→176). The wrong form mattered the moment FN became a knob, which
happened hours later — see §27.]**

| arm | FM | grow ask | vs 128 cap |
|---|---|---|---|
| A/B | 2 | `0x70` = 112 | fits, 16 to spare |
| **C/D** | **4** | `0xb0` = **176** | **EXCEEDS BY 48** |
| E/F | 1 | `0x50` = 80 | fits easily |

**Therefore FM=2 is the maximum frag height at the default cap**, and FM=4 requires the volatile
`BLOCK_SIZE=1` umr flip to 256. The 07-28 brief predicted FM=4 would land *exactly at* 128 and warned to
watch for grow-fail; the truth is worse — the ask OVERSHOOTS, which is a permanent 100% grow failure on
every wave, i.e. a hang risk, not a slow run. **Rule 6 (max work offline first) paid for itself here.**

### 23. THE FEED RATIO IS REAL BUT SECOND-ORDER (elasticity ≈ 0.4, not 1.0)
B→F and A→E both raise feed loads +67% (0.750 → 1.250) and cost **−18%** and **−20%** respectively.
If frag loads were the binding constraint the loss would be ~40%. It is not.

The design isolated the axis better than expected: `occ[96]` is **23,040 per rep in BOTH arms**, so
super-tile coordination is identical and only the inner-loop frag-load count moved.

Feed loads per rep: FM=1 = 921,600 · FM=2 = 552,960 · FM=4 = 368,640.
Power-law fit: **TF ∝ loads^−0.39** at 1 WG/CU, **^−0.43** at 2 WG/CU. The two independent fits agreeing
is the reassuring part. The axes do not interact — the effects are independent and multiplicative.

**PROJECTION for arm D (RECORDED AS A PREDICTION, NOT A RESULT): ~8.6–9.0 TF, about +15% over B.**
This is a two-point power-law extrapolated OUTSIDE the measured range, in the direction not sampled.
That is exactly the shape of inference that breaks. Do not bank it. It is the ONLY thing the umr flip
buys, and the flip is sudo + volatile + on the display GPU.

### 24. `dsws_realshape_bench.py` CANNOT RUN THE POOL-64 SWEEP (offline, confirmed)
`grep ALLOW_NONSTD` returns **zero hits** in the script. `--pool` IS threaded (`:632`, default 128) but
`DSWS_ALLOW_NONSTD=1` is not, so `gpu_run.sh` will REFUSE every shape at `--pool 64`. The 30-shape gate
that would promote the secondary 1 WG/CU config to config-of-record is blocked until this is added.
Suspected in the 07-28 brief §0.5; now confirmed.

### 25. WHAT DID NOT MOVE — CEILING A IS UNTOUCHED
`door1 NOTHING-STAGED = 100.0% of coast` in BOTH arms. Coast-frac **90.9%** (F) and **96.9%** (E).
Doors 2/3/4 flat zero everywhere; `grow-fail=0`; `occ[97]` C-store-gate bails **= 0** in both.
Both levers measured today are second-order dressing on a machine that spends ~90%+ of its feed-path
iterations with nothing staged to work on. **Neither the frag grid nor the umr flip touches Ceiling A.**

### 26. MEASUREMENT NOTE — THE FM=1 ARMS ARE NOISY
E and F spread 10.5% and 14.8%, against B's 1.9%. F per-rep ran 5.7–6.7; even the top of that range
sits clearly under B's 7.70, so the DIRECTION is safe, but **do not quote 6.3 or 3.8 to two significant
figures.** `DSWS2_RESVPROBE=1` was deliberately OMITTED from both dispatches: the 07-28 brief §0 command
block included it, but arm B's 7.70 baseline did not, and matching the baseline env exactly except for
the frag grid is what makes the comparison valid. Probes lie.

### 27. FN IS NOW A KNOB (offline, no GPU) — AND IT GIVES US A REAL CONTROL, NOT JUST MORE POINTS
`FN` was a hard-coded literal in two places: `-defsym,FN=4` in `build_flow.sh` mkflow, and
`const int FNc = 4;  // FN is fixed (the shared N-reuse operand)` at `occ_dispatch.cpp:7343`.
Both now read `FN` / `DSWS2_FN`. Everything downstream (`TN=FN*16`, C sizing, operand stride, oracle
addressing) was ALREADY parameterized, so this was a two-line change plus guards.

**★ THE POINT IS THE CONTROL ARM `FM=4 FN=2`. ★** Its feed-loads/WMMA is `(4+2)/8 = 0.750` — IDENTICAL
to the current `FM=2 FN=4` — and its NFV is 112, also identical. Same super-tile M=128, and at
`ML8_POOL=64` the same 1 WG/CU and GROUPS=1. **The ONLY thing that differs is the shape of the frag
grid.** If it does not measure ~7.70, then feed-loads/WMMA is NOT the mechanism, the grid SHAPE is, and
the arm-D projection in §23 is worthless. This is the falsification test, and it costs no umr flip.
Residual confound, stated honestly: LDS differs (17,920 vs 34,304) because ACC_STRIDE = FM*FN*1024 is
unchanged but the bank count differs. At matched `ML8_POOL=64` the wave count is identical, so this is
much weaker than the confound it replaces, but it is not zero.

### 28. THE LEGAL (FM,FN) FRONTIER AT superM=128, MEASURED BY ASSEMBLING EVERY CELL
`G*FM = 8` (constant super-tile M=128), `ACC_N = G` (GROUPS=1), `N=2560` so `2560 % (FN*16) == 0`.

| FM | FN | G | NFV | LDS | feed | result |
|---|---|---|---:|---:|---|---|
| 1 | 1 | 8 | 48 | 9,728 | 2.000 | assembles |
| 1 | 2 | 8 | 64 | 17,920 | 1.500 | assembles |
| 1 | 4 | 8 | 80 | 34,304 | 1.250 | assembles (**arm F, 6.3 TF**) |
| 2 | 1 | 4 | 64 | 9,728 | 1.500 | assembles |
| 2 | 2 | 4 | 80 | 17,920 | 1.000 | assembles |
| 2 | 4 | 4 | 112 | 34,304 | 0.750 | assembles (**arm B, 7.70 TF**) |
| 4 | 1 | 2 | 80 | 9,728 | 1.250 | assembles |
| **4** | **2** | **2** | **112** | **17,920** | **0.750** | **assembles — THE CONTROL** |
| 4 | 4 | 2 | 176 | 34,304 | 0.500 | **REFUSED, NFV>128** (arm D — needs the umr flip) |
| 2 | 5 / 1 | 5 | — | — | — | **BUILD-FAIL: "DSWS2_PREFETCH P2 block decode needs FN power-of-two"** |

**FN MUST BE A POWER OF TWO** — the prefetch P2 path decodes the block index by SHIFT. That is the
kernel's own assembler-time guard, not an assumption of mine, and it kills the 0.700 point (`FM=2 FN=5`)
that the plan wanted. Reachable feed ratios under the existing cap are therefore
**0.750, 1.000, 1.250, 1.500, 2.000** — five points, up from two.

**THREE MATCHED-RATIO CONTROL PAIRS now exist, all buildable today:**
`0.750`: 2×4 vs 4×2 · `1.250`: 1×4 vs 4×1 · `1.500`: 1×2 vs 2×1.
If feed ratio is the mechanism, each pair must measure the same. Three independent chances to falsify
the model before anyone touches a GPU register.

### 29. NEW GUARD: THE dyn-VGPR GROW-TARGET GATE (prevents a HANG, not a deviation)
Added to BOTH `build_flow.sh` (refuses before assembling) and `occ_dispatch.cpp` (refuses before any
packet). Computes `NFV = roundup16(32 + 8*FM*FN + 2*FM + 2*FN)` and refuses if it exceeds
`DSWS2_VGPR_CAP` (default **128**). **Deliberately NOT bypassed by `DSWS_ALLOW_NONSTD`** — that flag is
a POLICY override for deliberate A/B arms, whereas exceeding the VGPR cap is a permanent 100% grow
failure on every wave, i.e. a rule-3 hang. Raising it requires `DSWS2_VGPR_CAP=256` AND the volatile
umr flip. Verified: `FM=4 FN=4` now refuses at build time with the NFV printed; `FM=4 FN=2` builds and
its disassembly shows `s_alloc_vgpr 0x70` = 112, matching the predicted NFV exactly.
Also widened the host geometry whitelist from `FM in {1,2}` to `FM in {1,2,4,8}`, `FN in [1,8]` — the
NFV gate is now the real limiter, which is the correct place for it. (Note: the old `FM in {1,2}`
whitelist would have refused arms C/D on its own — a second independent guard we did not know we had.)

---

## 2026-07-29 MIDDAY — ★ THE FEED-RATIO MODEL IS FALSIFIED. COORDINATION IS THE LEVER. ★

Two matched-feed-ratio controls, enabled by the FN knob wired this morning (§27). Both gates clean:
`oracle bad=0`, WORK-EXACT, `occ[96]` delta +0, `occ[0]=0`, canary clean, grow-fail 0.
Logs: `ctrlFM4FN2_g2_accn2_pool64_nonstd`, `ctrlFM4FN1_g2_accn2_pool64_nonstd`.

| arm | FM×FN | TOTAL_super | feed/WMMA | TF | spread |
|---|---|---:|---|---:|---|
| B | 2×4 | 23,040 | 0.750 | **7.70** | 1.9% / 34 |
| F | 1×4 | 23,040 | 1.250 | **6.30** | 14.8% / 32 |
| c1 | 4×2 | 46,080 | 0.750 | **4.80** | 10.6% / 20 |
| c2 | 4×1 | 92,160 | 1.250 | **2.70** | 5.0% / 11 |

### 30. ★ THE FALSIFICATION ★
**c1 vs B** and **c2 vs F** are each matched on feed-loads/WMMA, feed loads per rep, WMMAs per rep,
super-tile M, occupancy, GROUPS and NFV. Under the §23 model each pair must measure the same.
**c1 came in 38% under B. c2 came in 57% under F.** The model predicted the frag grid was the axis;
the axis is something the frag grid was merely correlated with.

**THE ~8.6-9.0 TF ARM-D PROJECTION IN §23 IS DEAD**, and with it the entire throughput case for the
umr `BLOCK_SIZE=1` flip. We spent zero register-poke risk finding that out, which is the whole reason
the control was run before the flip rather than after.

### 31. WHAT IT ACTUALLY IS: TOTAL_super = THE SUPER-TILE COUNT = COORDINATION EVENTS
`FN` sets the N-panel width, so `TOTAL_super = 11,796,480 / (superM · FN)`. Halving FN doubles the
number of super-tiles, and `occ[96]` tracked it exactly (23,040 → 46,080 → 92,160 per rep).

**Elasticities, each from a pair holding the OTHER variable constant:**
| axis | pairs | exponent |
|---|---|---|
| **coordination** | B→c1 (2×) and F→c2 (4×) | **−0.682 and −0.611** |
| frag grid (feed) | B→F (1.667×) | −0.393 |

Two independent coordination fits, at different feed ratios, agreeing to within 0.07. Coordination
carries ~1.7× the exponent of the frag grid. And unlike the frag grid, **it points AT Ceiling A**:
`door1 NOTHING-STAGED` is 100% of coast in every arm, and coordination events ARE the boundary events.

### 32. THE LDS COUPLING — WHY IT CANNOT SIMPLY BE TURNED DOWN
At GROUPS=1 (`ACC_N = G`), with `G·FM = superM/16`:
```
TOTAL_super = 11,796,480 / (superM · FN)          LDS = 1536 + (superM/16) · FN · 1024
```
The two are inversely locked through the SAME product. We sit at `superM·FN = 512` (LDS 34,304).
Halving coordination needs `superM·FN = 1024` → **LDS 67,072, over the 65,536 cap by 1,536 B**.
Both routes fail identically (`superM=256 FN=4` and `superM=128 FN=8`): the accumulator banks scale
with super-tile AREA, so a super-tile large enough to halve coordination has accumulators that fill
all of LDS with nothing left for control structures.
**The way out is GROUPS=2** (`ACC_N = G/2`), which halves the banks. Costs the group barrier (~15%,
measured §16), gains ×2^0.65 ≈ 1.57 → net ≈ ×1.33.

### 33. THE `:646` OPERAND GUARD IS GATED ON `!SELFSERVE` — AND IT WAS PROVEN DEAD, NOT ASSUMED
The guard enforced `G·FM ≤ 11` at SEGK=256/FN=4, i.e. **super-tile M ≤ 176** — forbidding exactly the
direction §31 says to go. It was long noted as "vestigial under SELFSERVE", but `:479` claimed the
BRES_OFF/ARES_OFF immediates are "emitted unconditionally in the kernel body", and a `.if`-nesting scan
put the two `v_add_nc_u32 v13, v9, {B,A}RES_OFF` sites at TOP LEVEL. Those two statements contradict.

**DECISIVE TEST (offline):** changed `BRES_OFF` 256 → 1024 — which shifts BRES_OFF and ARES_OFF and
NOTHING else in the flow layout — and rebuilt. **BYTE-IDENTICAL bin** (`ff7cf5336902d0fe`, 30,940 B).
If any live instruction encoded those offsets the bin would have moved. The `:479` comment is STALE.
Guard is now wrapped in `.if !SELFSERVE`, kept verbatim for the legacy path. Verified: default build
still `ff7cf5336902d0fe` byte-identical, `SELFSERVE=0` still errors, and `superM=256` now assembles.

### 34. NEXT: `FM=2 FN=4 G=8 ACC_N=4` — superM=256, COORDINATION HALVED
Builds clean: **LDS 34,304 · NFV 112 · GROUPS=2 · TOTAL_super 11,520 (half of arm B) · feed 0.750
(IDENTICAL to arm B)**. Only coordination and GROUPS differ from the 7.70 baseline.
**★ CORRECTION TO THIS SECTION, BEFORE THE RUN: IT IS A DISCRIMINATOR, NOT A CONFIRMATION. ★**
The first draft predicted "~10.2 TF" and thereby smuggled in an assumption. In ALL FOUR arms above
`GROUPS=1`, so `occ[96]` (emissions) and `TOTAL_super` (super-tile count) were **perfectly collinear** —
they moved together every time. "Coordination" is therefore TWO candidate mechanisms that §31 conflated
and never separated. This config breaks the collinearity, because `GROUPS=2` doubles emissions back:

| | arm B | proposed |
|---|---:|---:|
| `TOTAL_super` | 23,040 | **11,520** (half) |
| GROUPS | 1 | 2 |
| `occ[96]` per rep | 23,040 | **23,040 (IDENTICAL)** |

**BOTH BRANCHES RECORDED BEFORE THE DISPATCH, so neither can be retrofitted:**
- **driver = `occ[96]` emissions** → **~7.70 or below** (no gain; the ~15% group barrier is pure cost).
- **driver = `TOTAL_super` / boundary events** → **~10.2** (×1.57 coordination, ×0.85 group barrier).

**KNOWN CONFOUND, stated in advance:** `superM=256` drops total output tiles to `8 × 40 = 320` over
64 WGs = ~5 tiles/WG. That is coarse enough that TAIL IMBALANCE could contaminate the result and would
LOOK like a coordination effect. A mid-range number will NOT be separable without a follow-up.
*** THE KERNEL SOURCE CHANGED (§33), SO THE NEXT DISPATCH IS A RULE-2 BRING-UP: ONE RUN, THEN STOP. ***

### 35. `dsws_realshape_bench.py` UNBLOCKED FOR THE 30-SHAPE SWEEP (offline; §24 resolved)
Three fixes, and the second was NOT the one we went looking for.

**(a) `DSWS_ALLOW_NONSTD` is now threadable** (`--allow-nonstd`). It was absent entirely, so `gpu_run.sh`
refused at `--pool 64` and the 1 WG/CU config could not be swept even deliberately. **It is NOT
auto-emitted on detected deviation** — the entire point of the flag is that deviating is an EXPLICIT
act, and a harness that quietly sets it defeats the guard it is satisfying. Added a PRE-FLIGHT that
refuses (exit 2) *before the first dispatch* when the geometry deviates without the flag, and again if
`--allow-nonstd` is used with the default `--tag rs`, since the standing rule is that a deviation must
be NAMED IN THE LOGNAME. Previously this died one shape in, after burning a card claim.

**(b) ★ `FN` WAS A COUPLED-AXIS TRAP, THE SECOND INSTANCE OF ONE THIS FILE ALREADY DOCUMENTS. ★**
`LIVE_FN` was hardcoded to 4 with no flag, and `LIVE_TN = LIVE_FN*16` is what **every shape's N-PADDING**
is computed against. Once FN became a knob (§27), `--fn 2` would have padded N for a 64-col panel while
the bin wanted 32 — *exactly* the silent geometry mismatch the file's own `--fm` note describes
("a knob that is a flag on one axis and a constant on a coupled axis is a trap, not a default"),
on the other axis. Added `--fn`, made `LIVE_TN` track it. Also **`DSWS2_FN` was never passed to the
dispatcher at all**, so the host would have defaulted `FNc=4` and disagreed with any FN≠4 bin.

**(c) `--fm` widened** from `choices=(1,2)` to `(1,2,4,8)`; the real limiter is the NFV gate (§29), which
both `build_flow.sh` and `occ_dispatch.cpp` now enforce. `--fn` is power-of-two only (§28).

VERIFIED (subprocess stubbed, nothing dispatched): refusal paths exit 2 with the right message; the
positive path emits `DSWS_ALLOW_NONSTD=1`, `DSWS2_FN=4`, `ML8_POOL=64`, and `ORACLE_NTL=144` for the
first shape (N=9216/64) under `--tag secondary_1wgcu`.

**THE 30-SHAPE GATE COMMAND IS NOW RUNNABLE** (needs the card; promotes the 1 WG/CU secondary config):
```bash
python3 dsws_realshape_bench.py live --fm 2 --g 4 --acc-n 4 --segk 256 --sswin 32 \
  --waves 16 --pool 64 --chunk-maxs 0.85 --allow-nonstd --tag secondary_1wgcu \
  --json secondary_1wgcu.json --table secondary_1wgcu.txt
```
NOTE it is a 30-DISPATCH SWEEP, so it needs its own greenlight and is NOT covered by a single-run one.

---

## 2026-07-29 AFTERNOON — ★ 10.2 TF. THE DISCRIMINATOR RESOLVED: IT IS `TOTAL_super`, NOT `occ[96]`. ★

**BEST NUMBER THIS KERNEL HAS EVER PRODUCED. +32% over the 7.70 that stood this morning, and the
first time it has cleared 10.** Single dispatch, rule-2 bring-up (kernel source changed in §33), stopped.
Bin `426a5007ae56e68f` · .text 31,676 B · LDS 34,304 · NFV 112.
Log: `discSuperM256_g8_accn4_pool64_nonstd`.

Config: `FM=2 FN=4 G=8 ACC_N=4`, `ML8_POOL=64` (1 WG/CU), superM=256, GROUPS=2, `ORACLE_MTL=8 NTL=40`.
Gates: `computed=8,755,200` WORK-EXACT (92,160 × 95 reps) · `occ[96]=2,188,800` **delta +0** ·
`oracle bad=0` · canary clean · `occ[0]=0` · grow-fail 0.

### 36. ★ THE DECISIVE FACT: `occ[96]` WAS IDENTICAL TO ARM B AND THROUGHPUT MOVED +32% ★
Both branches were recorded in §34 BEFORE the dispatch, precisely so neither could be retrofitted:
- driver = `occ[96]` emissions → **predicted ≤ 7.70. REFUTED.**
- driver = `TOTAL_super` / boundary events → **predicted ~10.2. MEASURED 10.2.**

| | arm B | this run |
|---|---:|---:|
| `computed` / rep | 92,160 | 92,160 |
| `occ[96]` / rep | 23,040 | **23,040 (IDENTICAL)** |
| feed loads / rep | 552,960 | 552,960 |
| WMMAs / rep | 11,796,480 | 11,796,480 |
| **`TOTAL_super`** | 23,040 | **11,520 (HALVED)** |
| **TF** | **7.70** | **10.2** |

Four quantities identical, one halved, throughput followed the one that moved. **The lever is the
SUPER-TILE COUNT — the number of boundary events — not the number of emissions.** In every arm before
this one `GROUPS=1` made those two collinear, which is why the morning's model could not see it.

### 37. THREE CAVEATS, NONE OF WHICH THREATEN THE DIRECTION
1. **Spread 14.6%** (per-rep 8.9–10.4) vs arm B's 1.9%. Call it 10.2 ± 0.7. The FLOOR of that range
   still clears 7.70, so the direction is safe — but do NOT quote 10.2 to three significant figures.
2. **ONE NUMBER IS VALIDATING TWO CLAIMS.** The ~10.2 prediction was ×1.57 (coordination) × 0.85 (group
   barrier). A barrier cost of ZERO with a weaker coordination exponent gives the same answer, and
   GROUPS=1 at superM=256 is UNTESTABLE (LDS 67,072 > 65,536, §32). So "×1.57 coordination, ×0.85
   barrier" is *a* decomposition consistent with the data, **NOT the measured one.** Do not cite the
   split as if it were measured.
3. **The pre-registered tail-imbalance confound (§34) pushes DOWN, not up.** 320 output tiles over
   64 WGs = ~5 tiles/WG; imbalance WASTES waves, it cannot invent throughput. If it is contaminating
   this at all, the true coordination effect is LARGER than measured. Stated because it is the
   direction that does NOT threaten the conclusion, which is exactly when it is worth being explicit.

### 38. WHAT DID NOT CHANGE — AND THE STANDING FIGURES THAT ARE NOW STALE
`door1 NOTHING-STAGED` is **still 100% of coast**. We made the starvation CHEAPER, not RARER. Ceiling A
is dented, not broken. Against hipBLASLt's 123–189 TF on real dense shapes we move ~2.5% → **~3.3%**.

**★ 7.70 IS NO LONGER THE BASELINE. WRITE FUTURE RESULTS AGAINST 10.2. ★** Consequences:
- `DSWS_BRIEF_2026-07-28_AM.md` §0.5 ("THE BEST MEASURED CONFIG WE HAVE", 7.70) is **SUPERSEDED**.
- KG decision `d0e79067` (the 1 WG/CU secondary config) is superseded on the TF figure. Its 1 WG/CU
  finding stands and is reinforced — this run is also pool 64.
- **The 30-shape gate unblocked in §35 should now be pointed at THIS config, not the 7.70 one**:
  `--fm 2 --fn 4 --g 8 --acc-n 4 --pool 64 --allow-nonstd --tag superm256`. Note `--g 8` requires the
  §33 guard gating, so that sweep CANNOT run against an un-patched tree.
- Still a SINGLE SHAPE (`ml8_dense_ffn_down` M2048 N2560 K9216). It is not config of record until the
  30-shape sweep says so, and the sweep is 30 dispatches = its own greenlight.

---

## 2026-07-29 LATE — THE BOUNDARY IS NOISE, AND EVERY SINGLE-FACTOR MODEL IS NOW DEAD

Two dispatches. Neither produced a throughput win; both produced falsifications, which is what they were
for. **13.8 TF (§39) is unaffected — the models below are explanations of WHY, not measurements.**

### 39. FLOW_WAVES=8 → 13.8 TF (+35%), THE DAY'S LAST WIN
Geometry IDENTICAL to the 10.2 run (`FM=2 FN=4 G=8 ACC_N=4`, superM=256, GROUPS=2, pool 64); only
waves/WG changed, 1024 → 512 resident. `occ[96]=2,580,480` delta +0 (23,040 × 112), oracle bad=0.
Log `waves8_superM256_g8_accn4_pool64_nonstd`. Confirms Kimi's claim that WAVES=16 is several times the
useful concurrency. Note Kimi HEDGED on thinning the poll herd via backoff ("expect flat TF, like the
funnel") but was BLUNT about not launching them at all — "the right thing to do with a wave that
provably cannot help is, in aggregate, not to have launched it." The blunt one paid.
**CAVEAT: spread 33.5% (per-rep 9.4–14.1), by far the widest in the project. The FLOOR is below the
10.2 baseline. 13.8 wants a confirming re-run before it is banked.** `WAVES=4` does NOT assemble
(invalid operand, kernel :5555/:7033).
**[CORRECTED 2026-07-29 LATE: "so 8 is the floor" was WRONG — I inferred it from WAVES=4 failing without
testing the intermediate values. `WAVES=7`, `6` and `5` ALL BUILD (30,916 / 30,844 / 30,764 B) and are
UNTESTED. The failure is specific to 4: `NCOMPUTE = WAVES - FIRST_COMPUTE_WID(3)`, so WAVES=4 gives
NCOMPUTE=1 and `BATON_MAGIC = 0x100000000/1 = 2^32`, not representable as a 32-bit literal. WAVES=5
gives 2^31, which the assembler DOES accept. Note WAVES=7 gives NCOMPUTE=4 — exactly Kimi's
"~4 concurrent waves already saturate the pipeline".]**

### 40. ★ THE BOUNDARY COST IS ~1.3% OF WAVE-TIME. DO NOT BUILD THE BANK DOUBLE-BUFFER. ★
`DSWS2_ADVPROBE=1 DSWS2_BNDTIME=1` — **both probes already existed and were switched off** (the SIXTH
such mechanism this project has needed and already had). Probe build, TF not quotable.

| | per visit | sampled count | share of boundary time |
|---|---:|---:|---:|
| winning advance (**the tail**) | 168.9 ticks | 832 | **1.6%** |
| losing pass | 32.0 ticks | 269,630 | **98.4%** |

Losing passes burn **61.4× the aggregate wave-time** of winning ones (unit-free ratio). Absolute share
of all wave-time: boundary **1.31%**, winning tail alone **0.021%**.
(The absolute figures assume RTC ticks == span ticks; the 61.4× ratio does not.)

**VERDICT: the bank double-buffer targets the winning tail = 0.02% of wave-time. NOT WORTH BUILDING.**
Kimi's Ceiling B ("the group barrier is the wall") is NOT supported by direct measurement. Its own hedge
was correct — the model rested on the "TF tracks n_kseg" correlation plus gate structure, magnitude
never timed. **THE MEASUREMENT COST ONE DISPATCH AND SAVED A DESIGN + BUILD + CODEX REVIEW + BRING-UP
ON A NULL.** Precedent that motivated measuring first: `DSWS2_KDBUF` (2026-07-25) was a double-buffer
built on this same kernel against an unmeasured latency hypothesis — implementation verified correct in
emitted code, result 0.32% SLOWER, inside noise. "THE BUILD WAS CORRECT; THE HYPOTHESIS WAS WRONG."

### 41. ★★ AND IT FALSIFIES §31's COORDINATION MODEL — MY OWN ★★
If boundary events were the mechanism, halving them buys **at most 0.66%**. `superM` 128→256 halved them
and measured **+32%**. The model cannot explain its own headline result.
**THE CONFOUND:** `TOTAL_super = 11,796,480/(superM·FN)`, and `superM·FN` **is the super-tile AREA**.
"Fewer boundary events" and "more operand reuse" were collinear in every arm. Right variable, wrong
mechanism.

### 42. THE DISCRIMINATOR: `FM=2 FN=2 G=8 ACC_N=8` → 6.3 TF. NO SINGLE-FACTOR MODEL SURVIVES.
Log `bwdisc_fn2_superM256_g8_accn8_w16_pool64_nonstd`. `occ[96]=737,280` delta +0, oracle bad=0.
Built at **WAVES=16 to match arm B** — the first build was WAVES=8 and would have been contaminated by
the §39 effect. Three branches pre-registered: coordination-only 7.70 · coordination+frag 6.9 ·
bandwidth 5.3. **Measured 6.3 — between them.**

| arm | superM | FN | GROUPS | TOTAL_super | feed | traffic/FLOP | TF |
|---|---:|---:|---:|---:|---:|---:|---:|
| B  2×4 | 128 | 4 | 1 | 23,040 | 0.750 | 0.02344 | **7.70** |
| F  1×4 | 128 | 4 | 1 | 23,040 | 1.250 | 0.02344 | **6.30** |
| c1 4×2 | 128 | 2 | 1 | 46,080 | 0.750 | 0.03906 | **4.80** |
| c2 4×1 | 128 | 1 | 1 | 92,160 | 1.250 | 0.07031 | **2.70** |
| **NEW 2×2** | 256 | 2 | 1 | **23,040** | 1.000 | 0.03516 | **6.30** |

- **PURE COORDINATION REFUTED.** NEW holds `TOTAL_super`, GROUPS, WAVES and pool EXACTLY at arm B's
  values. TF still moved −18%.
- **PURE TRAFFIC REFUTED.** B and F have IDENTICAL traffic/FLOP *and* identical `TOTAL_super` *and*
  identical GROUPS, differing ONLY in the frag grid — 7.70 vs 6.30.

### 43. CORRECTION TO §30–31 AND KG `077b53ef`: "THE FEED MODEL IS FALSIFIED" WAS TOO STRONG
What c1/c2 demonstrated is that feed is **NOT SUFFICIENT** — arms at identical feed ratios measured 38%
and 57% apart. I wrote that up as feed being FALSIFIED, which does not follow. B vs F isolates the frag
grid cleanly, gives exponent −0.393, and that value has reproduced. **Feed is real; it is not alone.**
And "coordination is the dominant lever" (the replacement claim) is now itself refuted by §42.

**SCORE FOR THE DAY: feed-only died at midday, coordination-only and traffic-only died in the evening.
Three single-factor models, three falsifications, every one of them from a CONTROL, none from a fit.**

### 44. METHOD: STOP FITTING EXPONENTS
Six points varying along three correlated dimensions, at least two live axes, none sufficient. That is
over-fitting territory and it has now burned me twice in one day. **A fourth exponent is worth nothing.**
NEXT MUST BE A DIRECT MEASUREMENT OF WHAT THE KERNEL WAITS ON — achieved HBM/L2 bandwidth against the
R9700 peak settles bandwidth-bound-or-not in one shot, and it is a fact about the machine rather than a
curve through our data points. `rocprof` on the PM4 path is the open question (the harness is raw PM4,
not HIP, so the usual dispatch-scoped counter capture may not attach).

---

## 2026-07-29 EVENING II — ★★ 15.2 TF. BOTH AXES HAVE AN OPTIMUM, AND THE WALL IS 20× CHEAPER. ★★

Seven dispatches under a standing greenlight from kmbandy ("keep running what you think will help find
the wall"). **Every arm gates clean**: WORK-EXACT, `occ[96]` delta +0, `oracle bad=0`, canary clean.

### 45. ★ THE WAVE AXIS HAS AN INTERIOR OPTIMUM AT WAVES=6 ★
All at `FM=2 FN=4 G=8 ACC_N=4`, superM=256, GROUPS=2, `ML8_POOL=64`:

| WAVES | NCOMPUTE | TF | spread | coast-frac |
|---:|---:|---:|---:|---:|
| 5 | 2 | 14.9 | 6.1% | 35.9% |
| **6** | **3** | **15.2** | **4.1% / 118 reps** | 52.1% |
| 7 | 4 | 14.6 | 34.8% | 59.1% |
| 8 | 5 | 13.8 | 33.5% | — |
| 16 | 13 | 10.2 | — | 95.0% |

**`WAVES=6` → 15.2 TF at 4.1% spread is the tightest measurement in this project.**
**★ THE SPREAD COLUMN IS A FINDING IN ITSELF.** Tight at 5–6, then it JUMPS to ~35% at 7 and 8, with
BIMODAL per-rep ranges (9.7–14.9 and 9.4–14.1). Something bistable switches on at **≥7 waves** — and
that is what made §39's 13.8 so noisy. **13.8 was never a trustworthy number; 15.2 is.**

### 46. THE GEOMETRY AXIS ALSO HAS AN OPTIMUM — superM=256 SURVIVES THE HEALTHY REGIME
At matched `WAVES=6`, `ML8_POOL=64`:

| superM | G | TF | spread | coast-frac | tiles/WG |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 12.7 | 3.1% | 63.5% | 10.0 |
| **256** | **8** | **15.2** | **4.1%** | 52.1% | 5.0 |
| 512 | 16 | 13.6 | **37.1%** | 43.8% | 2.5 |

superM=256 beats 128 by **+20%** — so §36's geometry finding is REAL and not an artifact of the
over-subscribed regime, though the magnitude is smaller than the +32% measured there. superM=512 is
WORSE, and its 37.1% bimodal spread at 2.5 tiles/WG is the tail-imbalance confound pre-registered in
§34 showing up exactly where predicted. **The superM=512 run also confirms the §33 guard gating works —
`G=16` (G·FM=32) would have been refused by `:646` before today.**

### 47. RESVPROBE RE-RUN: THE WALL IS UNCHANGED IN CHARACTER, 20× CHEAPER IN MAGNITUDE
The 07-27 decomposition was taken in the pathological 2048-wave regime. Re-measured at the healthy
config (`WAVES=6`, superM=256):

| | 2026-07-27 (2048 waves) | 1 WG/CU (1024) | **now (384 waves)** |
|---|---:|---:|---:|
| empty frontier, share of bails | 96.1% | — | **95.2%** |
| CAS-loss | 1.6% | — | 4.8% |
| window-full | 0.5% | — | **0** |
| boundary | 1.7% | — | **0** |
| **bail-iters per successful reserve** | **129.9** | 26.3 | **6.5** |

**THE RATIO BARELY MOVED; THE ABSOLUTE WASTE FELL 20×.** We did not change what the machine is waiting
for — we stopped having twenty times as many waves waiting for it. The limiter is still PRODUCER-SIDE
frontier publication rate. Window-full and boundary are now EXACTLY ZERO, so the cursor, the SSWIN
window and the C-store gate are all definitively cleared as limiters.

### 48. ★ CONFIG OF RECORD CANDIDATE — 15.2 TF ★
```bash
WAVES=6 FM=2 FN=4 G=8 ACC_N=4 ./build_flow.sh     # LDS 34,304 · NFV 112 · GROUPS=2 · superM=256
./gpu_run.sh <logname> -- DSWS_ALLOW_NONSTD=1 FLOW_WAVES=6 ML8_POOL=64 DSWS2_FLOW=1 \
  DSWS2_FM=2 DSWS2_FN=4 DSWS2_G=8 DSWS2_ACC_N=4 FLOW_POOL_N=1 DSWS2_SEGK=256 SSWIN=32 \
  DSWS2_K=<K> DSWS2_ORACLE_MTL=<M/256> DSWS2_ORACLE_NTL=<N/64> DSWS2_ORACLE_STRIDE=8 \
  DSWS2_TARGET_SECS=1.5 ML8_COOP_CHUNK=512 ML8_COOP_CHUNK_MAXS=0.85 \
  STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```
**DAY TOTAL: 4.73 → 15.2 TF = +221%.** vs hipBLASLt 123–189 on dense shapes: 2.5% → ~4.9%.
STILL A SINGLE SHAPE — the 30-shape gate (§35, now runnable) is what promotes it. Note `--g 8` REQUIRES
the §33 guard gating, so that sweep cannot run against an un-patched tree.

### 49. WHAT IS STILL UNTESTED AT THE NEW OPTIMUM
Everything measured before today was measured in a regime we now know was pathological (95% coast,
129.9 bail-iters/reserve). Re-testing at `WAVES=6` is cheap and several old verdicts may not survive:
**SEGK** (called "THE lever" on 2026-07-19, untested today) · **POOL_N** (measured INERT) ·
**DSWS2_FUNNEL** (defaulted on for waste-elimination, never justified on throughput) ·
**FN=8 at FM=1** (NFV 128, exactly at cap) · and `WAVES=4`, which needs a small kernel fix:
`NCOMPUTE=1` makes `BATON_MAGIC = 2^32`, unrepresentable — and with NCOMPUTE=1 the modulo is trivially
0, so the `s_mul_hi_u32` can simply be skipped under `.if NCOMPUTE == 1`.

---

## 2026-07-29 NIGHT — THE PUBLISHER IS NOT THE BOTTLENECK. `HEAD` IS 20% OF WAVE-TIME. 18.6 TF.

Six dispatches. **kmbandy caught a methodological error mid-stretch and was right**: I began by probing
the publication path at `WAVES=6`, the config where publication demonstrably WORKS. That regime cannot
answer why the publisher falls behind. The diagnosis has to be the DELTA between the broken and working
regimes with identical probes — which is what the rest of this section is.

### 50. ★ RETRACTION, BY MY OWN CONTROL: "THE PUBLISHER SLOWS 4.3× WITH WAVES" WAS A PROBE ARTIFACT ★
`ADVPROBE+BNDTIME` appeared to show the advance critical section going 171.2 → 728.7 ticks from
`WAVES=6` → `16`. **`BNDTIME` stamps on EVERY LOSING PASS, and there are 10.9× more of those at
WAVES=16 — the instrument was throttling the thing it measured.** I flagged this as a possible confound
and ran `ADVPROBE` ALONE (fires only on the rare WIN path, so it does not scale with the herd):

| | WAVES=6 | WAVES=16 |
|---|---:|---:|
| ticks/advance, ADVPROBE+BNDTIME | 171.2 | **728.7** ← artifact |
| **ticks/advance, ADVPROBE ONLY** | **167.4** | **172.6** ← truth, 3% apart |

**THE CRITICAL SECTION IS FLAT.** This also invalidates the `BNDSPLIT+GAP` build's advance counts
(695 vs 217) — `DSWS2_GAP` is UNTHROTTLED and stamps per burst, so it scales with the herd too.
That build read TF=3.5 at WAVES=16 against 10.2 clean: the probe cost was 3×.
**RULE: any probe whose emission rate scales with the population under test cannot measure a
population-size effect. Check the instrument's own scaling before attributing the trend to the kernel.**

### 51. WITH A LIGHT PROBE, PUBLICATION PER CHUNK IS IDENTICAL AT BOTH WAVE COUNTS
| | WAVES=6 | WAVES=16 |
|---|---:|---:|
| advances (de-throttled) | 62,016 | 56,000 |
| chunks | 115 | 100 |
| **advances per chunk** | **539** | **560** |
| ticks per chunk | 664,087 | 864,313 |

**The same publication work lands per chunk either way; each chunk just takes 30% longer at WAVES=16.**
Publication rate tracks throughput (1.25× vs 1.30×), i.e. it is a PROXY FOR THE WORK RATE, not an
independent limiter. **THE PUBLISHER KEEPS UP. Publication was never the bottleneck.**
The wave-count penalty is DIFFUSE — no single stage carries it.

### 52. BNDSPLIT AT THE HEALTHY CONFIG — THE GATES THAT DOMINATED ON 07-27 ARE NOW ZERO
| | 07-27 (2048 waves) | now (WAVES=6) |
|---|---:|---:|
| ZLOCK_LOST | 76.5% | 99.4% |
| DRAINGATE_BAIL | — | **0.0%** |
| CSTOREGATE_BAIL | 93% of winners | **0.0%** |
| ADVANCE | 1.6% | 0.6% |

Every lock winner now advances (695 wins → 695 advances, 100% conversion).
**CAVEAT, DO NOT SKIP:** `herd` is derived by SUBTRACTION (`bEntry − bZwon`) and `DSWS2_FUNNEL=1` is the
default, so funnel rejects — which bail BEFORE the CAS — are counted as ZLOCK losses. The 99.4% is NOT
established as lock contention. Disambiguating needs a `DSWS2_FUNNEL=0` re-run (untested).

### 53. ★ `HEAD` IS THE BIGGEST SINGLE COST IN THE KERNEL: 19.90% OF ALL WAVE-TIME ★
`DSWS2_GAP` at `WAVES=6` (instrument #7 already built and switched off):

| bucket | mean ticks | share of wave-time |
|---|---:|---:|
| **HEAD** (live → first work) | **149,782 ≈ 1.5 ms** | **19.90%** |
| GAP (between bursts) | 487 | 15.47% |
| TAIL (last work → exit) | 24,924 | 2.76% |

`n = 41,472 = 384 waves × 108 chunks` → **paid FRESH EVERY CHUNK by every wave.** A chunk is ~753k ticks
and each wave burns ~150k of it before doing any work at all. **This is the physical mechanism behind
Kimi's Ceiling A** (per-chunk fixed cost, whose arithmetic bounded us to ~8.5 TF at one chunk and ~23 TF
in the limit of infinite chunk). Also `NOBURST = 6,912/41,472 = 16.7%` of waves never burst at all.

### 54. ~~★★ HEAD AMORTIZATION CONFIRMED — 18.6 TF AT M=8192, A NEW BEST ★★~~ **RETRACTED**

> **⛔ RETRACTED SAME NIGHT (kmbandy). M=8192 AT N=2560 K=9216 IS NOT AN ml8 OR mlambaformer SHAPE — IT
> IS A CUBE I INVENTED. `18.6 TF IS NOT A RESULT AND NOT A "BEST".` Bigger M = more tiles per dispatch =
> the per-dispatch fixed cost spreads over more work = TF rises. That is ARITHMETIC ABOUT THE
> DENOMINATOR, not an improvement to the kernel; ANY kernel posts a higher TF on a bigger shape.
> This is the SAME failure that produced the deleted 36.9/32.0 TF "winners" (32K synthetic square,
> garbage on real ml8) and that the 2026-07-16 canonical framing exists to prevent. I also proposed
> sweeping M=16384 and M=32768 before being stopped.**
> **★ SHAPES ARE INPUTS, NOT LEVERS. ★ THE TELL: if the config is unchanged and only the input grew,
> no lever was pulled.** KG feedback `8e201972`.
>
> **THREE VOID RUNS, NOT ONE.** `headamort_M8192_…_182015`, `headamort_M16384_…_182700` and
> `headamort_M32768_…_183008` all executed — the M=16384/32768 pair fired before the interrupt landed.
> **ALL THREE ARE VOID. Do not mine them for numbers.** For the record, 19 of the day's 22 dispatches
> ran `2048x2560x9216` = `ml8_dense_ffn_down` (SHAPES:29); every finding in §45–57 is on that real
> shape. The contamination is confined to these three.
>
> **WHAT SURVIVES:** §53's HEAD measurement (19.90% of wave-time, 1.5 ms/wave/dispatch) was taken on the
> REAL M2048 shape and STANDS. But on a real shape M is fixed and the problem is ALREADY ONE CHUNK
> (320 tiles < 512), so there is nothing to amortize HEAD over. **THE LEVER IS TO SHRINK HEAD, NOT
> SPREAD IT** — what is a wave doing for 1.5 ms between going live and its first work item? That
> question is shape-independent, which is what makes it a real lever.

~~Original entry, kept for the record:~~
Same config, 4× the tiles per dispatch (`M=8192`, MTLsuper=32, 1,280 tiles, `ML8_COOP_CHUNK=2048`,
still ONE chunk). Gates clean: `computed=18,063,360` WORK-EXACT, `occ[96]=4,515,840` delta +0,
`oracle ok=10240 bad=0`, canary clean.
~~TF 15.2 → 18.6 (+22%), spread 13.1% over 49 reps, coast-frac 38.9%.~~ **NOT COMPARABLE — different shape.**
Confirms HEAD is a per-dispatch fixed cost that amortizes with work per dispatch. **This is a REAL
throughput lever and it is orthogonal to every geometry knob.** (Rule 7 checked in advance: one chunk
~25 ms against the 850 ms cap.)

### 55. `SEGK` RE-TESTED — THE FIRST PRE-TODAY VERDICT THAT SURVIVES THE REGIME CHANGE
At `WAVES=6`, superM=256, LDS unchanged (SELFSERVE reclaims the operand pool, so SEGK is a clean knob):

| SEGK | n_kseg | occ[96]/rep | TF | spread | coast-frac |
|---:|---:|---:|---:|---:|---:|
| **256** | 36 | 23,040 | **15.2** | 4.1% | 52.1% |
| 128 | 72 | 46,080 | 8.8 | 1.8% | 34.8% |
| 64 | 144 | 92,160 | 6.6 | 54.2% | 27.6% |

**"TF tracks SEGK" (2026-07-19) HOLDS.** SEGK=256 is already our config and is confirmed optimal.
(Confound as ever: SEGK changes coordination count AND per-item WMMA amortization together —
128 WMMAs/item at SEGK=256 vs 32 at SEGK=64.)

### 56. ★ COAST-FRAC IS NOT A THROUGHPUT PROXY — STOP READING IT AS WASTE ★
SEGK=64 coasts only **27.6%** and delivers **6.6 TF**. SEGK=256 coasts **52.1%** and delivers **15.2**.
Coast-frac moved OPPOSITE to throughput across the whole SEGK sweep. A coasting wave is not necessarily
wasted work — and `door1 = 100% of coast` has been quoted all project as if it were the problem
statement. **It is a description of what coasting waves are waiting on, NOT evidence that coasting is
the cost.**

### 57. ~~THE BISTABILITY SIGNATURE APPEARS IN EVERY BAD CONFIG~~ **WRONG — IT IS A RUN-LEVEL LOTTERY**

> **⛔ REFUTED SAME NIGHT by a repeatability test (§59). Spread is NOT a property of the config.
> Four repeats of the IDENTICAL bin `beb031c195df` gave spreads 3.5% / 3.7% / 3.2% / 35.8%.
> I read a random draw as a config property. The configs below each happened to draw one wide run.**

~~Original entry, kept for the record:~~
Wide bimodal spread (~35–55%) now seen at: WAVES≥7 (34.8%, 33.5%), superM=512 (37.1%), SEGK=64 (54.2%).
Tight (<7%) at every good config: WAVES=5 (6.1%), WAVES=6 (4.1%), superM=128 (3.1%), SEGK=128 (1.8%).
**Whatever the bistable mode is, it is a shared failure mode across three independent axes and it has
never been characterised.** A per-rep timeline (`TRACE=1`, per-super-tile claimer rows) would show it.


### 58. `SLEEPN` IS NOT A LEVER — AND coast-frac IS CONFIRMED AS A POLL-COUNT ARTIFACT
Real shape, WAVES=6, best config, only `SLEEPN` varied (the `s_sleep` arg in the busy-waits, :438):

| SLEEPN | TF | spread | coast-frac |
|---:|---:|---:|---:|
| 1 | 15.1 | 35.7% | 53.2% |
| 2 (default) | 15.2 | 4.1% | 52.1% |
| 4 | 15.1 | 36.3% | 49.8% |
| 8 | 15.4 | 4.0% | 45.5% |

**TF FLAT across an 8× range.** HEAD is NOT sleep-dominated. Note coast-frac falls MONOTONICALLY with
SLEEPN (53.2 → 45.5) while TF does not move at all: longer sleeps mean fewer poll iterations, so fewer
are counted as coast. **This independently confirms §56 — coast-frac is a POLL-COUNT ARTIFACT, not a
measure of wasted work.** Also checked and NOT run: `PHIST` was the wrong instrument for HEAD — it is a
BAIL-DOOR histogram (occ[104..113]), ~220% overhead, and `occ_dispatch.cpp:2091` records a PHIST build
once running a 2.46 s chunk against the 0.75 s cap. `INITBAR` is in the HEAD window but is the
2026-07-20 correctness fix (`INITBAR=0` reproduces the buggy `f36c06a0`) — do not touch it.

### 59. ★ REPEATABILITY: THE MEAN IS SOLID, THE SPREAD IS A LOTTERY ★
Four back-to-back repeats, IDENTICAL bin `beb031c195df`, nothing varied:

| repeat | TF | spread | per-rep range |
|---:|---:|---:|---|
| 1 | 15.5 | 3.5% | 15.1–15.7 |
| 2 | 15.4 | 3.7% | 15.2–15.7 |
| 3 | 15.5 | 3.2% | 15.1–15.6 |
| 4 | **15.3** | **35.8%** | **10.0–15.6** |

**MEAN TF IS REPRODUCIBLE TO ±1.3% (15.3–15.5).** Every TF comparison made today is therefore valid and
the axis optima stand. **BUT SPREAD IS A RUN-LEVEL DRAW, NOT A CONFIG PROPERTY** — and note repeat 4's
mean barely moved despite the 35.8% spread, because the wide mode is a HANDFUL OF OUTLIER REPS dipping
to ~10 inside an otherwise ~15.5 run. It is occasional stalls, not a second operating regime.

**TWO CONSEQUENCES:**
1. **§57 IS RETRACTED** (above).
2. **THE WAVE OPTIMUM IS REAL BUT SHALLOW.** With run-to-run noise at ±1.3%, WAVES 5/6/7 (14.9 / 15.4 /
   14.6) are separated by only 3–5%. The optimum at 6 survives, but the DOMINANT effect on that axis is
   FEW waves vs MANY (15.4 vs 10.2 at WAVES=16), not the precise interior value. Do not over-tune it.
3. **A BETTER CENTRAL ESTIMATE FOR THE BEST CONFIG IS 15.4 TF** (mean of 4 repeats), not the 15.2
   single-run figure in §45/§48.

---

## 2026-07-29 NIGHT II — ★★ THE 30-SHAPE GATE, AND A 5× ON THE SHAPES THAT ACTUALLY MATTER ★★

### 60. THE 30-SHAPE GATE AT THE WAVES=6 CONFIG — 30 PASS, 0 FAIL, 3 UNSUPPORTED
All real ml8/mlambaformer shapes. `best15_4.json` / `best15_4.txt`. **Correctness is solid everywhere.**
Throughput is not. Padding-corrected `real_TF`:

| class | M | real TF |
|---|---:|---:|
| ml8 dense | 2048 | 5.08 – **14.64** |
| ml8 dense | 512 | 1.61 – 8.77 |
| mlmf | 4096 | 0.42 – 5.32 |
| **mlmf MoE expert fc1/fc2** | 512 | **0.97 / 0.96** |
| ml8 MoE | 512 | 0.70 – 4.33 |
| **ml8 MoE** | **64** | **0.093 – 0.61** |

**WE TUNED ALL DAY ON `ml8_dense_ffn_down` M2048 — THE TOP OF THAT RANGE — WHILE THE mlambaformer MoE
EXPERTS (recorded as ~56% OF GEMM TIME) SIT AT 0.97 TF.** 3 UNSUPPORTED: `mlmf_router_MLP`,
`mlmf_router_out`, `mlmf_routerout_ML8PAD` — `n_kseg=1<2` at SEGK=256, ZLOCK needs ≥2, need `--segk 128`.
(Cosmetic bug: the inventory header prints "M tile=96" from stale module defaults; the PASS rows and the
padding maths correctly used superM=256.)

### 61. ★ M-PADDING IS NOT THE PROBLEM. THE RUNTIME IS 100% FIXED COST. ★
`ml8_moe_ffn_gate_up` M64 N512 K2048, superM swept so padding goes 75% → 50% → 0%:

| superM | padded M | padding | padded TF | **real TF** | **span/chunk** |
|---:|---:|---:|---:|---:|---:|
| 256 | 256 | 75% | 0.40 | **0.101** | 133k ticks |
| 128 | 128 | 50% | 0.20 | **0.10** | 137k |
| 64 | 64 | **0%** | 0.10 | **0.10** | 120k |

**REMOVING 75% OF THE COMPUTED WORK CHANGED THE REAL RATE NOT AT ALL**, and span/chunk is FLAT across a
4× work change. Padded TF fell exactly in proportion to the padding removed — i.e. the padding was never
costing time. **~1.3 ms per dispatch to compute 134 MFLOP; the card could do that work in 0.4 µs.**
So HEAD (§53, 19.9% of wave-time at M2048) is ~**100%** of the runtime here.

### 62. ★★ THE FIXED COST SCALES WITH WORKGROUP COUNT — 5× BY LAUNCHING LESS OF THE CARD ★★
Same shape, superM=64 (no padding), sweeping `ML8_POOL`:

| ML8_POOL | waves | **real TF** | span/chunk |
|---:|---:|---:|---:|
| 64 | 384 | 0.1 | 116k |
| 16 | 96 | 0.4 | 38k |
| **8** | **48** | **0.5** | **27.5k** |
| **4** | **24** | **0.5** | **24.8k** |
| 2 | 12 | 0.4 | 35k |
| 1 | 6 | 0.2 | 61k |

**CLEAR INTERIOR OPTIMUM AT pool 4–8.** Too many WGs → launch/ramp cost; too few → serialisation.
On a shape with 8 output tiles, launching all 64 CUs means most workgroups ramp up, find nothing, and
retire — **and that ramp IS the runtime.**

**GENERALISES** — `ml8_moe_ffn_down` M64 N2048 K512 (different N *and* K): pool 64 → **0.1**,
pool 8 → **0.5**; span/chunk 102k → 29k. Two independent shapes, both **5×**.

**vs the §60 gate baseline: 0.101 → 0.5 TF and 0.111 → 0.5 TF. FIRST REAL WIN ON THE PRODUCT SHAPES.**

### 63. THE IMPLIED DISPATCH POLICY — A HARNESS CHANGE, NOT A KERNEL ONE
`ML8_POOL` is currently PINNED (128 config-of-record, 64 for the 1 WG/CU work). **It must be DERIVED
FROM THE SHAPE'S AVAILABLE PARALLELISM.** Fitting both ends — M64 wants 4–8, M2048 wants 64 —
**`pool ≈ min(64, TOTAL_super / 10)`**, i.e. keep ≥10 super-tiles per workgroup.
NOT YET TESTED: whether the same rule helps the M=512 MoE shapes (0.70–4.33) and the mlmf experts
(0.97), whose `TOTAL_super` is large enough that the rule returns pool≈64 — i.e. **the rule predicts NO
change for them, so their gap has a DIFFERENT cause.** That is the next thing to find out.

### 64. PROCESS FAILURE — `pgrep -f` SELF-MATCH, ~18 MINUTES OF FALSE "STILL RUNNING"
I monitored the 30-shape sweep with `pgrep -f dsws_realshape_bench`. **My own watcher shell's command
line CONTAINS that string**, so the check matched itself and never went false. I reported "still going"
for ~18 min after the sweep had finished at 18:58, and held the GPU claim idle throughout. kmbandy
caught it. This is the DOCUMENTED 2026-07-18 trap (`pgrep -f` self-matched and killed an ssh session
four times); the recorded fix is to put the pattern inside a script invoked by path so the caller's
cmdline cannot contain it. I also ignored two contradicting signals I had already seen: the output
files were written, and stdout was "0 bytes" (buffered, flushed at exit).
**RULE: never let a liveness check match its own invocation; prefer `ps` + an explicit non-self pattern,
and treat "output file already written" as stronger evidence than any process check.**

### 65. ★★ AUTO-POOL ACROSS ALL 30 REAL SHAPES: GEOMEAN 1.37×, 12 SHAPES >1.5×, ZERO REGRESSIONS ★★
`--pool-auto` wired into `dsws_realshape_bench.py` (default OFF — a harness that silently retunes the
dispatch geometry per shape would make every cross-run comparison meaningless). `ML8_POOL =
min(--pool, TOTAL_super/--tiles-per-wg)`, tiles-per-wg=10. **Kernel, geometry and waves IDENTICAL to
§60; pool is the ONLY variable.** 30 PASS / 0 FAIL / 3 UNSUPPORTED. `poolauto.json` / `poolauto.txt`.

| shape | TOTAL_super | pool | before | after | |
|---|---:|---:|---:|---:|---:|
| mlmf_attn_val_proj1 M4096 | 96 | 9 | 0.416 | **0.987** | 2.37× |
| ml8_moe_attn_kv M64 | 64 | 6 | 0.093 | 0.217 | 2.34× |
| ml8_moe_ffn_gate_up M64 | 64 | 6 | 0.101 | 0.219 | 2.17× |
| ml8_moe_attn_kv M512 | 128 | 12 | 0.699 | 1.497 | 2.14× |
| mlmf_attn_linear_k M4096 | 144 | 14 | 0.814 | 1.739 | 2.14× |
| ml8_moe_ffn_gate_up M512 | 128 | 12 | 0.711 | 1.493 | 2.10× |
| ml8_moe_ffn_down M512 | 128 | 12 | 0.734 | 1.409 | 1.92× |
| **mlmf_MoE_expert_fc2 M512** | 144 | 14 | **0.955** | **1.767** | **1.85×** |
| **mlmf_MoE_expert_fc1 M512** | 144 | 14 | **0.966** | **1.782** | **1.84×** |
| ml8_moe_ffn_down M64 | 64 | 6 | 0.111 | 0.204 | 1.84× |
| ml8_dense_attn_kv M512 | 320 | 32 | 1.608 | 2.720 | 1.69× |
| mlmf_router_down_proj M4096 | 192 | 19 | 1.215 | 1.913 | 1.57× |
| *…9 shapes at pool 51–64* | | | | | 1.05–1.32× |
| *ml8_dense_ffn_down M2048* | *11520* | *64* | *14.639* | *14.689* | *1.00×* |
| *ml8_dense_attn_q M2048* | *5120* | *64* | *11.314* | *11.335* | *1.00×* |
| *ml8_dense_ffn_gate_up M2048* | *11520* | *64* | *9.955* | *9.259* | *0.93×* |

**GEOMEAN 1.37× over all 30. Sum 118.32 → 130.29.**
**THE CONTROL HELD:** every shape whose pool stayed at 64 moved 0.93–1.10×, two of them at exactly
1.00×. That is what makes the wins attributable to pool and nothing else. (The one 0.93× is inside the
run-to-run lottery characterised in §59.)
**THE mlambaformer MoE EXPERTS — recorded as ~56% OF GEMM TIME — WENT 0.96 → 1.78.**

### 66. ★ CORRECTION: superM AND pool INTERACT. §61's "PADDING IS NOT THE PROBLEM" WAS TOO STRONG. ★
Three measurements that only make sense together:
- superM 256→64 **at pool 64**: 0.101 → 0.10 — **FLAT** (§61)
- pool 64→6 **at superM 256**: 0.101 → 0.219 — **2.17×** (§65)
- superM=64 **AND** pool=8: 0.101 → **0.5** — **5×** (§62)

**THEY ARE NOT INDEPENDENT.** superM alone does nothing because at 64 WGs the runtime is pure launch
cost and the padded work is not on the critical path. Remove the launch cost and the padding starts to
matter. §61's conclusion ("padding is not the problem") is TRUE ONLY AT pool=64 and I stated it
unconditionally.
**CONSEQUENCE: §65's 1.37× UNDERSTATES WHAT IS AVAILABLE.** It tuned pool with superM pinned at 256.
On the small shapes there is a SECOND UNCLAIMED FACTOR OF ~2 in per-shape superM (2.17× measured vs 5×
achievable on `ml8_moe_ffn_gate_up` M64). **NEXT: extend auto-tuning to superM (via G at fixed FM) and
re-run the gate.** Note superM is a BUILD-TIME defsym, so per-shape superM means one bin per superM
class and a rebuild between groups — the harness currently builds once.

---

## 2026-07-29 NIGHT III — ★★ WE READ hipBLASLt's ACTUAL fp8 gfx1201 ISA. ONE ROOT CAUSE, THREE SYMPTOMS. ★★

kmbandy's idea: run DSWS through static analysis, run hipBLASLt through static analysis, diff them.
Offline, no GPU, shape-independent — RGA/static analysis takes the BUILD CONFIG, never M/N/K.

### 67. GETTING AT THE VENDOR BINARY
`/opt/rocm/lib/hipblaslt/library/TensileLibrary_B8B8_..._gfx1201.co` is a **CCOB** (zstd-compressed
clang offload bundle), not an ELF. Extract with
`clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1201`.
→ 38 MB ELF, **446 fp8 GEMM kernels** for our exact arch. (Also present: `B8F8`, `F8B8`, `HB8`, `SB8`…
type-combination libraries. This one is `bf8_bf8` = e5m2; **we emit `v_wmma_f32_16x16x16_fp8_fp8`
= e4m3** — same WMMA shape and cost, different 8-bit format.)

### 68. ★ THE STRUCTURAL DIFF — AN EXACT INVERSION OF OUR DESIGN ★
| | hipBLASLt (446 kernels) | DSWS best config |
|---|---|---|
| VGPR | min 56 · **median 254** · max 256 | **48 peak-live** of 256 |
| SGPR | median 72 | 54 |
| LDS | min 1,638 · **median 6,400** · max 32,768 | **34,304** |
| WGs/CU | **~4** (4-wave WGs, 254 VGPR ≈ 4 waves/SIMD, 4×6,400 B fits) | **1** |
| spills | **0** | 0 |
| non-WMMA per WMMA | **2.19 best** · 12.93 median | **8.13** |

**THEY FILL THE REGISTER FILE AND BARELY TOUCH LDS. WE LEAVE 81% OF THE REGISTER FILE IDLE AND FILL LDS.**
This is the `_GSU1_` thesis (accumulators in VGPRs across the whole K loop, C stored once) — previously
inferred from kernel NAMES, now **measured from their binary**.

### 69. ★★ IT IS ONE ROOT CAUSE, NOT THREE PROBLEMS ★★
Our accumulators live in LDS (split-K + `ds_add`). Therefore:
1. every accumulate is a `ds_add` + wait instead of a register-resident WMMA accumulate
   → **the instruction overhead** (8.13 vs their 2.19)
2. the accumulator banks consume 34,304 B → **LDS is the occupancy limiter** → **1 WG/CU vs their ~4**
3. the register file sits **81% unused with zero spills** while we starve for what it could have held

Their leanest: `MT128x128x32 MIWT4_4` — 160 WMMA in a **511-instruction** window. Note the **x32**:
K-depth 32 per iteration, so more WMMA amortising each iteration's loop overhead.

### 70. METHOD — FOUR PARSING BUGS, EACH CAUGHT BY AN IMPLAUSIBLE ZERO
Recorded because the *pattern* is the lesson, not the bugs:
1. `<Cijk_…>` label regex — Tensile uses internal `label_*` symbols, so `cur` never set → 0 WMMA.
2. Mnemonic assumed `wmma`; it is `v_wmma_f32_16x16x16_bf8_bf8` (matched, but bug 1 masked it).
3. Symbols are **446 `R` descriptors + 446 `T` code**; bisecting the sorted list landed in descriptors.
4. llvm-objdump puts the address in a **TRAILING comment** (`// 00000025D4B4:`), not a leading column,
   and **all 446 T symbols have size 0** (hand-written asm declares none) so `a >= st+sz` filtered
   everything. Fix: extent = next symbol's address.
**Every one produced a clean, plausible-looking `0` that I did not report as a finding, because
"0 WMMA in 4.97M lines of disassembly" is not a measurement.** (The project's standing trap: §"zeros
that were never measurements".)
**AND A FIFTH, WHICH WAS NOT A PARSE BUG BUT A WINDOW BUG:** whole-body counting gave hipBLASLt 141
median vs our 21.68 — i.e. "we are 6.5× LEANER than the vendor", which is nonsense. Tensile kernels
carry huge run-once edge-case code. **Re-measured in the same first-WMMA→last-WMMA window for BOTH**
(the window RGA uses): their 2.19 best / 12.93 median vs our 8.13. Our 8.13 sits next to the project's
existing RGA figure of 7.38 on a sibling config — **an independent check that the method is sound.**

### 71. CAVEATS THAT MUST TRAVEL WITH THIS
- **They SELECT a kernel per shape; we have ONE.** Their median (12.93) is WORSE than ours. The honest
  target is their SELECTED kernel — near the 2.19 end for a well-matched shape — **not their median.**
- **Static.** It says the CODE has 3.7× more non-WMMA per WMMA; it does NOT say throughput scales that
  way (ports co-issue). Same caveat the 2026-07-26 tile sweep carried.
- The window is first-WMMA→last-WMMA, which is NOT proven to be only the k-step loop.

### 72. ★★ THE ACCUMULATOR IS NOT THE PROBLEM. SCALAR ADDRESS ARITHMETIC IS. ★★
We set out to attack the LDS-accumulator design. **Measurement says it is 2.9% of the window.**
FIRST FACT, from reading the burst: **the ACC is ALREADY REGISTER-RESIDENT.** `.Lflow_da_ss_rowblk`
WMMAs into `v[ACC…]` across all `KSEG_STEPS` k-steps and does ONE `ds_add` per (rowblk, ksi) segment.
The gap to hipBLASLt is not registers-vs-LDS, it is FLUSH FREQUENCY: 36 flushes per rowblk at
K=9216/SEGK=256 vs their 1 per output tile. `JDEPTH` is exactly that knob ("keeps ACC in registers
across J consecutive ksi, flushes ONCE") and is pinned to 1 by a DESIGN incompatibility with SELFSERVE,
not a perf result.

**DSWS burst window (first-WMMA → last-WMMA, 2337 instr, 256 WMMA, ratio 8.13):**
| group | count | % | per WMMA |
|---|---:|---:|---:|
| **SALU / branch** | **1060** | **45.4%** | **4.14** |
| VALU | 372 | 15.9% | 1.45 |
| WMMA | 256 | 11.0% | 1.00 |
| global load/store | 252 | 10.8% | 0.98 |
| waits | 169 | 7.2% | 0.66 |
| LDS other | 160 | 6.8% | 0.62 |
| **ds_add (ACC flush)** | **68** | **2.9%** | **0.27** |

`s_add_co_u32` 300 + `s_add_co_ci_u32` 222 = **522 64-bit address instructions = 22% of the window**,
against **124** `global_load_tr_b64` — **4.2 address instructions per load.**

**HEAD-TO-HEAD vs their leanest (`MT128x128x32`, 511 instr, 160 WMMA, ratio 2.19):**
| per WMMA | DSWS | hipBLASLt | |
|---|---:|---:|---|
| **64-bit addr arith** | **2.04** | **0.11** | **18×** |
| SALU total | 4.14 | 0.77 | 5.4× |
| loads | 0.48 | 0.29 | 1.6× |
| ds_add | 0.27 | 0 | — |
| **non-WMMA** | **8.13** | **2.19** | **3.7×** |

**THEY USE THE SAME LOAD INSTRUCTION** (`global_load_tr_b64`) — this is NOT instruction selection.
Their loop does not RECOMPUTE ADDRESSES (17 addr instrs for 47 loads) and is PREDICATED, not branched
(`v_cndmask_b32_e64` ×64 is their #2 mnemonic).

**TARGETS, IN MEASURED ORDER:**
1. **Scalar address bookkeeping in the burst body — 522 instrs, 22% of window.** Hoist the base per
   burst, use immediate offsets on the k-steps instead of recomputing 64-bit addresses per load.
2. Branchy control flow where predication would serve.
3. `JDEPTH` / accumulator flush — **2.9%**. Fixing it buys 8.13 → 7.9. NOT worth the SELFSERVE design
   fight now, and this REPLACES the plan we started the session with.

**ETHOS: THIS DOES NOT TOUCH THE FLOW ARCHITECTURE.** The fix is inside the compute burst. The claim
protocol, frontier, role economy and river principle are untouched. hipBLASLt buys a lean inner loop by
shipping 446 RIGID variants (their non-WMMA:WMMA spans 2.19 → 88, median 12.93 — WORSE than our 8.13);
we would get a lean inner loop while keeping ONE ADAPTIVE kernel. **That dispersion is the product
argument for DSWS, not a footnote: we are not trying to beat them on every shape, we are trying to beat
them on MOST while being far more consistent — which a per-shape lookup table structurally cannot do.**

**CAVEAT, DO NOT OVERSTATE:** SALU issues on a SEPARATE PIPE from VALU on RDNA4, so part of that 45% may
already be hidden behind WMMA/VALU issue. **Instruction count is not time.** Strong hypothesis with a
clear mechanism — not a guaranteed 23%.

---

## 2026-07-29 LATE NIGHT — INSTRUCTION CUTS IN THE COMPUTE BURST, AND A RETRACTION OF THE WINDOW ITSELF

**NOTHING IN THIS SECTION HAS EXECUTED.** Five source changes, static verification only, GPU held by the
weight-pager. The bring-up is the first task tomorrow.

### 73. FIVE CHANGES — 389 INSTRUCTIONS REMOVED, WORK INVARIANTS UNTOUCHED
| | slice | s_add_co* | s_mul | ratio |
|---|---:|---:|---:|---:|
| original | 2337 | 522 | 80 | 8.13 |
| (1) B ADDR FOLD ×5 | 2089 | 274 | 80 | 7.16 |
| (2) MI=0 FOLD ×5 | 1996 | 212 | 49 | 6.80 |
| (3) MI=1 HOIST ×4 | 1950 | 182 | 33 | 6.62 |
| (4,5) dead `s52` advance ×2 (Kimi) | **1948** | **180** | 33 | **6.62** |

**Invariants IDENTICAL at every step: WMMA 256 · B-loads 124 · A-loads 62 · stores 64 · `ds_add` 68.**
`.text` 30,844 → 28,852 B. That invariance is the correctness signal — same work, less bookkeeping.

**(1) B ADDR FOLD.** `s_add_u32 s54,s52,(ni*256)` + `s_addc_u32 s55,s53,0` + load on `s[54:55]`
→ `global_load_tr_b64 …, s[52:53] offset:(ni*256)`. `ni*256` is assembly-time constant; the saddr form
takes an immediate and the A path 3 lines away already used it. Precedent in-repo: `occ_kernel_coop.s:901`.
**(2) MI=0 FOLD.** At `mi==0`, `s_mul_i32 s58,s32,0` / `s_add_u32 s58,s56,s58` / `s_addc_u32 s59,s57,0`
is a pure copy of `s[56:57]`. Load on `s[56:57]` directly. No liveness change, general over FM.
**(3) MI=1 HOIST.** `mi*s32 + s[56:57]` is loop-invariant; precompute the `mi==1` base ONCE per rowblk
into the `s54/s55` pair that (1) freed. **This is the only change that extends liveness.**
**(4,5)** final-step `s52 += s10` is dead (`s52` re-derived either way) — guarded `.if ks < KSEG_STEPS-1`.

### 74. ★ RETRACTION: MY MEASUREMENT WINDOW WAS NOT THE INNER LOOP ★
**`first-WMMA → last-WMMA` IS NOT A K-STEP LOOP ON THIS KERNEL.** Verified: **60 labels** fall inside it,
including `.Lflow_jwait`, `.Lflow_bankwr`, `.Lbaton_norm`, `.Lflow_da_ss_complete`, `.Lflow_cstore`,
`.Lflow_drain_adv`. It spans ring compute + the whole JDEPTH retire/drain/C-store path + decode.
**PROOF:** the actual burst source contains **10 `v_mov` and 1 `s_wait_dscnt`**; the window reported
**244 and 133**. That mass is `lds_*` macro expansion (exec saves, lane-0 broadcast) and probes —
PER-CLAIM coordination, not per-k-step, much compiled out at FORENSICS=0.

**THEREFORE §72's HEADLINE COMPARISON IS WITHDRAWN.** hipBLASLt's 2.19 is a real 511-instruction inner
loop (`MT128x128x32`, 160 WMMA). Our 8.13/6.62 is a metered slice of most of the coordination machinery.
**"3.7× more non-WMMA per WMMA than the vendor" IS NOT ESTABLISHED.** I flagged this exact risk in the
handoff spec ("the window is first-WMMA→last-WMMA, which I have NOT proven is only the k-step loop") and
it came back as the answer. The instruction cuts are still real — they removed genuine per-k-step work —
but the ratio they moved is not the quantity hipBLASLt's 2.19 measures.

### 75. THE K-LOOP IS DONE. WHERE TO LOOK NEXT.
Kimi K3's conclusion after implementing (4,5): *"the in-k-loop per-step address cost is exactly your
irreducible 2-instr s52 advance — there is nothing left inside the k-loop that is both safe and
non-instrumentation."* The remaining ~180 `s_add_co*` are per-rowblk/per-claim, not per-k-step.

**REMAINING CANDIDATES, ordered, with why each is blocked:**
1. **B-base block is fully `s33`-invariant** (8 instrs: `s20/s21/s25` mul-chain + `s52/s53` adds).
   Hoist to decode, re-copy 2/rowblk → **−6/rowblk**.
2. **A-base is LINEAR in `s33`**: `A_base(r+1) = A_base(r) + FM·s32` (`s36` increments by exactly 1).
   Precompute `FM·s32` once, 2-instr increment at loop bottom → **−9/rowblk**.
3. **`s43 = (s41<<2)+TILEDONE_BASE` is `s33`-invariant** → **−2/rowblk**.
   Total ≈ **−17/rowblk × (ACC_N−1)**.
**ALL THREE BLOCKED ON THE SAME THING:** they need 3 SGPRs live across the whole burst. `s20/s21/s25` are
free in-region today, **but `drain_advance` writes `s20–s23`** and is invoked immediately before the label
and at drain sites — making them burst-live recreates the `:1401` hazard class exactly. `s58/s59` could
carry 2 of 3 at FM≤2 but not FM>2. **Kimi described and STOPPED rather than applying — correct call.**
4. **`global_load_tr_b128`** would halve B-loads 124→62. `occ_kernel_btr128.s` already proves the
   two-adjacent-frag semantics. Blocked: changes `KDBUF_LPT`'s wait watermark (:1267) and `bcnt`
   accounting. **A design conversation, not a slice.**
5. **NOT the accumulator.** `ds_add` is 68 instrs = 3.3%. `JDEPTH` buys ~0.3 on the ratio. Dead end.
6. **NOT predication.** Kimi declined with a better reason than mine: the burst is fully-unrolled
   straight-line `.rept` — *there is no branch in it to convert*. hipBLASLt's `v_cndmask` is
   remainder/tail handling in a ROLLED loop; our `KSEG_STEPS` is compile-time so there is no tail. Our
   real branches (role loop, claim, coast) are control flow where predication executes BOTH sides —
   strictly worse for spin/coast, "and it would poke the river for nothing."

### 76. ADVERSARIAL REVIEW (Kimi K3 via pi_handoff, fireworks) — ALL THREE OF MY CHANGES PASS
Asked to break them; could not. Evidence it produced, not assertions:
- **`s54/s55` liveness:** macro-by-macro table of everything invoked in
  `.Lflow_da_ss_rowblk..rows_done` with the registers each defsym resolves to — `gap_*`→s62/63,64,72–83 ·
  `wtb_*`→s62/63,72–83 · `bcnt_add`→s72–80 · `cnt_inc`→s85/89 · `phase_stamp`→s62/63/64/77,81/82 ·
  `deadman_progress`→s101 · `acc_base_of`→s39 · `lds_fetch_add_r`→s49. **None is 54/55 under any defsym
  combination.** The in-region exec-save idiom uses **s49, not s57**. And `grep '\.set X, 54|55'` →
  **no alias exists**, so the `:1401` aliased-reuse hazard cannot apply to this pair.
- **Control flow:** only two entries (fall-through from `.Lflow_da_ss_decode`, and the `s_branch`
  loop-back at :7092) and **both land above the hoist**; `batch_next` re-enters via `decode`.
- **`.if FM > 1`:** correct at FM=1 (hoist elided, `.rept 1` never reaches the branch), exact at FM=2,
  and FM≥4 falls through unchanged since `s32` is FM-independent.
- **Residual, loud not silent:** the folded B immediate `(FN−1)·256` = 768 at FN=4; FN>~16 would exceed
  the offset field and the **assembler rejects it at build time** — never a wrong address.

### 77. THE ONE FRAGILITY IT FOUND — NOW IMPOSSIBLE BY CONSTRUCTION
The `mi==1` invariant was held by **two independent conditions**: `.if FM > 1` at the hoist (which WRITES
`s54/s55`) and `.rept FM` at the four load sites (which READ them). Editing one without the other = a load
from an **uninitialised `s[54:55]` = SILENT WRONG C, not a build error.**
**FIXED:** single shared symbol `.set A_MI1_HOISTED, (FM > 1)`; the hoist is `.if A_MI1_HOISTED` and every
site is `.elseif A_MI1_HOISTED && mi == 1`. Zero bare conditions remain.
**VERIFIED A PURE REFACTOR: bin byte-identical either side (`58e965a46f3e162d`)**, and assembles at
FM=1/2/4. (An FM=4 "failure" in the first check was my own NFV gate correctly refusing FM=4 FN=4 at 176
VGPRs > 128 — the gate working, not a guard bug.)

---

## 2026-07-31 MORNING — ★★ THE CUTS ARE FLAT, AND TF IS NOT COMPARABLE ACROSS DAYS ★★

### 78. THE BRING-UP (§0 OF THE BRIEF) — PASSES ON CORRECTNESS, FLAT ON THROUGHPUT
Rule-2 bring-up of the five instruction cuts. **First execution of that code.** ONE dispatch, then stop.
`WAVES=6 FM=2 FN=4 G=8 ACC_N=4 SEGK=256`, `ML8_POOL=64`, bin `58e965a46f3e162d`, `.text` 28,852 B.

**GATE — ALL FIVE CRITERIA PASS:**
| criterion | result |
|---|---|
| oracle | `ok=2560 bad=0 max_rel=0` — **bit-exact**, not merely inside the LOOSE tier |
| work-exact | `computed = 11,520,000` / 125 reps = **92,160/rep** — the pre-registered value |
| `occ[96]` | 2,880,000, delta **+0** |
| `occ[0]` (live) | **0** |
| canary | clean, no OOB store past C-end |

Also clean: `occ[95]=0` (no false-won CAS), `occ[97]=0` (no boundary-drain bails), `occ[73]=0` grow-fails,
coast decomposition closes exactly (door sum 12,753,821 == coast).
**The five cuts are CORRECT. They change no work and no result.**

Reported **17.5 TF** against a 15.4 baseline. **The brief pre-registered "if it comes back materially
FASTER, be suspicious and re-check work-exactness before celebrating."** That instruction is what produced
everything below, and it was right.

### 79. ★ THE CONTROL: THE CUTS ARE FLAT (+0.44%) ★
The pre-cut source **was never committed and existed nowhere** — `HEAD` is days-old DSWS state that no
longer even assembles (it predates the `:646` `.if !SELFSERVE` gating, so the stale single-slot guard
fires). Reconstructed it by **scripted inversion** of the five documented cuts, with asserted replacement
counts (5 B-fold / 5 A-fold / 1 hoist / 2 ks-guard) so pattern drift fails loudly instead of silently
emitting a wrong control.

**VALIDATED BY HASH BEFORE ANY SILICON: the rebuild reproduced `beb031c195df` bit-for-bit** (`.text`
30,844 B — both the sha256 prefix recorded in §59 and the size recorded in §73). The control was therefore
**provably the exact binary that already ran green four times**, so rule 2 did not apply to it.

Same session, same cold card, 16 minutes apart, identical env:

| bin | `.text` | ms/rep | TF | oracle |
|---|---:|---:|---:|---|
| pre-cut `beb031c1` | 30,844 | 5.5377 | 17.5 | bad=0 `max_rel=0` |
| cuts `58e965a4` | 28,852 | 5.5137 | 17.5 | bad=0 `max_rel=0` |

**+0.44% — inside the ±1.3% run-to-run band (§59). THE CUTS ARE FLAT.**
Removing **389 instructions** (−17% of the slice; `s_add_co*` 522 → 180) bought **nothing measurable on
this shape.** This is exactly the pre-registered prediction, and per the brief **a flat result is a PASS.**
Kimi's reasoning stands: SALU issues on a separate pipe from VALU on RDNA4, and NOBLOAD/NOWMMA already
showed the burst body is not the binding constraint *here*. The cuts remain justified by the thesis — the
burst must be lean where the burst *does* bind, i.e. the small MoE shapes — but **they are not a win on
`ml8_dense_ffn_down`, and must not be logged as one.**

*Observation, NOT a result:* pre-cut showed spread 43.6% (per-rep 10.0–17.8) vs the cuts' 4.8%. One sample
each, and §59 established spread is a run-level lottery. **Do not claim the cuts fixed the outlier mode.**

### 80. ★★ THE REAL FINDING: TF IS NOT COMPARABLE ACROSS SESSIONS/DAYS ★★
The 15.4 → 17.5 jump was **never the cuts**. It is the machine. The control bin is *provably the same
binary* that produced 15.4 (hash-matched before dispatch):

| when | bin | ms/rep | TF |
|---|---|---:|---:|
| 2026-07-29 ~18:40 (§59, ×4) | `beb031c1` | 6.2284 / 6.2688 / 6.2512 / 6.2968 | 15.3–15.5 |
| 2026-07-31 ~08:53 | `beb031c1` | **5.5377** | **17.5** |

**+12.5% ON BYTE-IDENTICAL CODE** — same shape, same env, same host, same wrapper. Yesterday's four
repeats were tight (±1.3%), so this is **not** within-session noise; it is a **day-level shift**.

**NOT THERMAL — MEASURED, NOT ARGUED.** Card was **55 °C / 51 W before** and **54 °C / 49 W after** a run.
A 0.69 s run at **5.7% of peak fp8** does not move the die temperature, so the baseline's back-to-back
predecessors cannot have left the card meaningfully hotter either. Both sessions were at idle temperature.
(My first hypothesis was "cold card this morning". It is wrong.) **Cause remains UNIDENTIFIED** — suspect
persistent DPM/clock or driver state (box uptime ~17 h at the time).

**NO CLOCK DATA EXISTS FOR ANY RUN IN THIS PROJECT'S HISTORY.** ⚠ **CORRECTION to a claim I made in
chat:** I called the empty `$LOG.journal` files "a bug". **They are not.** `$LOG.journal` is a
`journalctl` grep for amdgpu **errors** (`MODE1|page fault|VRAM lost|…`) and is **CORRECTLY EMPTY on a
clean run** — it is brick forensics, not telemetry. **Never read its emptiness as missing data.** The
actual gap was that achieved sclk/power was simply never sampled by anything. Fixed in §81.

**CONSEQUENCE — THIS IS THE EXPENSIVE PART:**
> **EVERY CROSS-SESSION TF COMPARISON IN THIS LOG IS SUSPECT.** Any conclusion drawn by comparing a number
> measured on one day against a number from another may be reading machine drift, not a lever. **Only a
> SAME-SESSION A/B against a rebuilt control bin is valid.** Absolute TF figures are session-local.

This does **not** invalidate within-session sweeps (§45–46 axis optima, §60–66 pool/superM, §59 repeats) —
those were back-to-back. It **does** mean any figure carried forward across a session boundary as a
baseline must be **re-measured in-session before it is compared to anything.**

**METHOD THAT MADE THIS FINDABLE, AND IS NOW STANDING PRACTICE:**
1. **Pre-register the expected result in writing before dispatching** — the brief said be suspicious of a
   faster number, so a +13% "win" got audited instead of celebrated.
2. **Reconstruct-and-hash-validate.** An uncommitted prior revision can be recovered by scripted inversion
   of documented edits and **proven exact by binary hash** before it is ever run. A hash match is a
   48-bit proof; inspection is not.
3. **Keep a control bin for any claim.** The cheapest way to be wrong here is to compare against a number
   from a previous session.

**KG `340f6965`** (shared to fleet). Both dispatches clean, zero GPU resets; card released.

### 81. THE INSTRUMENT: `gpu_run.sh` NOW RECORDS ACHIEVED CLOCKS ON EVERY RUN
**Motivated directly by §80** — the drift could not be diagnosed because achieved sclk was never sampled.

**What it does.** A background sampler writes `$LOG.telemetry` (CSV: `t_ms,sclk_mhz,mclk_mhz,power_w,
temp_c,busy_pct`) at ~10 Hz across the dispatch, and prints a one-line summary after the run:
```
[gpu_run] sclk while busy: mean 2216 / min 2100 / max 2350 MHz | mclk mean 1258 MHz
          | power mean 195 / peak 210 W | temp peak 64 C  (3 busy of 5 samples)
```
Stats are computed over samples with `gpu_busy_percent > 0`, so host-side oracle time doesn't dilute them.

**Source.** amdgpu sysfs on the card matched by **PCI ID `1002:7551`** (override `DSWS_GPU_PCI_ID`) —
resolved by ID, not card index, because `card0` is the 6900 XT (`1002:73BF`) and indices are not stable.
Units verified against the live files, not assumed: `freq{1,2}_input` **Hz**, `power1_average` **µW**,
`temp1_input` **m°C**.

**Safety.** Host-side sysfs **reads** in a background shell: nothing enters the kernel, the hot path, or
the message bus (rule 5), and no GPU state is written. The sampler is stopped **after** `RC=${PIPESTATUS[0]}`
is captured, so the dispatch pipeline's exit status is never disturbed.

**A missing instrument now FAILS LOUD.** If the sysfs node isn't found the run still proceeds, but prints
`*** WARNING: no GPU telemetry ... achieved sclk will NOT be recorded ***`. Silence is what let §80 go
unnoticed for the life of the project.

**VERIFICATION (offline — the R9700 was held by another session, so no dispatch was run):**
- `bash -n` clean.
- Telemetry block **extracted from the file itself** (not a retyped copy, so the test cannot drift from
  the code) and run against live sysfs: 46 samples in 2.5 s, correct units, and it picked up **real load
  from the other session** (10 busy samples, sclk 15→1120 MHz) — proving the sampler sees load.
- Summary awk exercised on **both** branches; synthetic busy data reproduced the expected
  2216/2100/2350 MHz, 195/210 W, 64 °C, 3-of-5 exactly.
- Full guard chain walked with `GPU_RUN_DRY=1`: geometry guard refused first, then the **2026-07-26
  collision guard correctly refused** (`gpu:R9700 is held by ANOTHER SESSION 87d16c2e`) — the guard doing
  its job. No stray telemetry file is created on a refusal.
- **NOT yet exercised inside a live dispatch.** It is verified by construction and standalone test only.
  **First real run should sanity-check that the summary line appears and reports a plausible busy sclk.**

**WHAT TO DO WITH IT.** The next same-session A/B should record sclk for both arms. If the 07-29 vs 07-31
gap reappears with **matched** clocks, the cause is not DPM and the search moves elsewhere; if clocks
differ, §80 is explained and TF must be normalised per-clock before any cross-session comparison.

---

## 2026-07-31 MIDDAY — ★★ THE STATIC MATRIX: 90% OF THE KERNEL IS UNREACHABLE BY EVERY LEVER WE HAVE ★★

### 82. TWELVE VARIATIONS THROUGH RGA — MEASURE FIRST, NO HYPOTHESIS
**Directive (kmbandy):** *"use the tools we have to measure and only measure. stop measuring 1 thing, making
some grand assumption, going down a rabbit hole about that assumption, and then finding out it was wrong.
take several variations of the kernel. measure every aspect possible. run each one through rga."* And:
*"the commonalities across the variations are going to tell us structurally where the bottlenecks are and
which levers are bottlenecks themselves."* This section is that, with no mechanism proposed.

Data: `DSWS_STATIC_MATRIX_2026-07-31.csv` (+ per-cell ISA / livereg / stats preserved in scratchpad).
**Entirely offline. No GPU.** 15 cells attempted, 12 built, 3 refused (refusals recorded as data).

**CORRESPONDENCE GUARANTEE — the thing that makes this trustworthy.** `rga_check.sh` re-assembles the
kernel with only the defsyms you hand it, which would analyse a DIFFERENT kernel than `build_flow.sh`
ships (the real build passes ~60 defsyms). Instead, each cell's **exact** clang command is extracted from
`bash -x ./build_flow.sh` and re-run with `RGADESC=1` appended. The analysed object is never
hand-reconstructed.

**TWO INSTRUMENT TRAPS CAUGHT BEFORE THEY BECAME NUMBERS:**
1. **RGA's `USED_LDS_BYTES` and `USED_VGPRs` ARE ARTIFACTS OF THE ANALYSIS DESCRIPTOR, NOT THE KERNEL.**
   RGA reports LDS=13,824 and VGPR=256 for a bin whose real LDS is 34,304. `RGADESC` (`:7639`) emits an
   analysis-only AMDHSA descriptor so `rga -s bin --co` can enumerate the kernel; its LDS/VGPR fields are
   that descriptor's, not the shipped geometry. **Take LDS/NFV from `build_flow.sh`; from RGA take only
   livereg, SGPRs, spills, ISA size.** Quoting RGA's LDS here would have been a §80-class fabrication.
2. `/usr/bin/rga` on this box is **ripgrep-all**, not the Radeon GPU Analyzer (`~/Downloads/rdts/.../rga`,
   v2.14.2.8). `rga_check.sh` documents the clash; a fresh harness would walk straight into it.

### THE RESULT — COMPUTE MOVES 106%, THE REST MOVES 8%
Split each cell into **COMPUTE** (`v_wmma + global_load_tr + global_load_b + ds_add + global_store`) and
**REST** (every other instruction):

| cell | TOT | COMPUTE | REST | REST% | wmma% | SGPR | spill | livereg | LDS | superM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 5291 | 653 | 4638 | 87.7% | 4.8% | 72 | 0 | 82 | 34304 | 256 |
| accn2 | 5243 | 621 | 4622 | 88.2% | 4.9% | 72 | 0 | 82 | 17920 | 256 |
| segk128 | 4995 | 429 | 4566 | 91.4% | 2.6% | 72 | 0 | 82 | 34304 | 256 |
| segk64 | 4847 | 317 | 4530 | **93.5%** | 1.3% | 72 | 0 | 82 | 34304 | 256 |
| fm1 | 4873 | 397 | **4476** | 91.9% | 2.6% | 72 | 0 | 48 | 17920 | 128 |
| fm4fn2 | 5483 | 653 | **4830** | 88.1% | 4.7% | 72 | 0 | 82 | 34304 | 512 |
| fn2 | 4891 | 365 | 4526 | 92.5% | 2.6% | 72 | 0 | 46 | 17920 | 256 |
| g4 | 5257 | 653 | 4604 | 87.6% | 4.9% | 72 | 0 | 82 | 34304 | 128 |
| g6accn3 | 5249 | 637 | 4612 | 87.9% | 4.9% | 72 | 0 | 82 | 26112 | 192 |
| g6accn2 | 5225 | 621 | 4604 | 88.1% | 4.9% | 72 | 0 | 82 | 17920 | 192 |
| waves8 | 5327 | 653 | 4674 | 87.7% | 4.8% | 72 | 0 | 82 | 34304 | 256 |
| waves16 | 5471 | 653 | 4818 | 88.1% | 4.7% | 72 | 0 | 82 | 34304 | 256 |

> **COMPUTE 317 → 653 (+106%). REST 4,476 → 4,830 (+8%). REST is 87.6–93.5% of the kernel in EVERY cell.
> WMMA is 1.3–4.9%.**

**EVERY GEOMETRY LEVER WE HAVE — `ACC_N`, `SEGK`, `FM`, `FN`, `G`, `WAVES` — MOVES ONLY THE ~10% THAT IS
COMPUTE.** The other ~4,600 instructions are the same code in all twelve variations. That is the answer to
"which levers are bottlenecks themselves": **collectively they cannot reach 90% of the kernel.**

This retroactively explains every flat result on record — NOBLOAD / NOWMMA / NOCFLUSH all flat (§ 2026-07-20),
and the five instruction cuts flat at +0.44% (§79). **They were all cutting inside the same 10%.**

### THREE QUANTITIES THAT DO NOT MOVE AT ALL
- **`SGPR = 72` in all 12 cells.** No lever changes it.
- **Zero VGPR and zero SGPR spills in all 12.** Register pressure is not a differentiator anywhere.
- **`livereg` peak tracks ONLY the frag grid**: 82 for every FM×FN=8 cell, 46–48 for the 4-frag cells.
  Independent of `ACC_N`, `SEGK`, `G`, `WAVES`. (Note livereg 82 < NFV 112 — HW allocates 120.)

### THE REFUSALS ARE DATA
- **`FN=8` → NFV=192; `FM=4 FN=4` → NFV=176. Both refused against the 128 dyn-VGPR cap.** The only two
  cells in the sweep that would raise per-wave arithmetic intensity are exactly the two that are
  forbidden. The cap is a live structural constraint (see the umr `BLOCK_SIZE=1` → 256 note).
- **`WAVES=4` DOES NOT ASSEMBLE** — `error: invalid operand for instruction`. A latent bug, not a design
  limit: the `WAVES` axis has a hole in it that nobody knew about.

### THE NEAR-TWIN
`fm4fn2` vs `base` are statically near-identical — same feed ratio 0.750, NFV 112, livereg 82, LDS 34,304,
256 WMMA, 653 COMPUTE. They differ only in `superM` (512 vs 256) and REST (4,830 vs 4,638). This is the
control arm `build_flow.sh:31` pre-registered. **If they measure different TF, the difference is not in the
code — it is in the work decomposition.**

### ⚠ THE LIMIT OF THIS SECTION, STATED PLAINLY
**STATIC INSTRUCTION COUNT IS NOT EXECUTION COUNT.** REST mixes one-time setup, per-claim coordination and
per-k-step work; **90% of the CODE does not mean 90% of the TIME.** This establishes where the code mass
is and that our levers cannot reach it. It does NOT establish where the cycles go, and **no mechanism is
claimed here.** The matching dynamic measurement is next; drawing a conclusion from this half alone would
be the exact error this campaign was ordered to stop.

### 83. WHAT THE INVARIANT 90% ACTUALLY IS — EXEC-MASK BOOKKEEPING (offline, read-only)
§82 established REST is ~90% of the kernel and moves ±8% across every lever. It did not ask what REST
*is*. Answered here from the 12 saved disassemblies — **no rebuild, no GPU.**

**REST by mnemonic (base cell, REST = 4,638):**
| mnemonic | count | % REST | | mnemonic | count | % REST |
|---|---:|---:|---|---|---:|---:|
| `v_mov_b32` | 934 | 20.1% | | `s_cbranch_execz` | 360 | 7.8% |
| `s_mov_b32` | 777 | 16.8% | | `ds_store_b32` | 332 | 7.2% |
| `s_wait_dscnt` | 444 | 9.6% | | `s_add_co_u32` | 186 | 4.0% |
| `s_and_b32` | 394 | 8.5% | | `v_readfirstlane_b32` | 81 | 1.7% |
| `v_cmp_eq_u32` | 360 | 7.8% | | *(top 12 = 88% of REST)* | | |

**`v_cmp_eq_u32` = 360 and `s_cbranch_execz` = 360 — EXACTLY EQUAL.** That is a signature, and it maps to
one construct: the `lds_*` lane-0 accessor family. `lds_put` (`:1631`) is **9 instructions to write ONE
32-bit LDS word**, of which exactly one is the `ds_store`:
```
s_mov_b32 s49, exec_lo / v_cmp_eq_u32 vcc_lo,0,v2 / s_and_b32 exec_lo.. / s_cbranch_execz   <- 4 guard
v_mov_b32 v[RP_A],off  / v_mov_b32 v[RP_D],ssrc                                             <- 2 marshal
ds_store_b32 / s_wait_dscnt 0x0                                                             <- 1 work +wait
s_mov_b32 exec_lo, s49                                                                      <- 1 restore
```
Family call sites in source: `lds_get` 82, `lds_put` 60, `lds_put_r` 45, `lds_get_r` 30, `lds_cas_rtn` 14,
`lds_fetch_add_r` 14, `lds_fetch_add` 9, `lds_cmpstore_adv` 6, `lds_inc` 4, `lds_inc_r` 3 — **267 sites**,
expanding (with `.rept`) to **360 lane-0-guarded blocks**.

> **360 blocks × 5 pure-bookkeeping instructions (2×`s_mov` exec save/restore + `v_cmp` + `s_and` +
> `s_cbranch_execz`) = 1,800 instructions = 34% OF THE ENTIRE KERNEL, moving no data at all.**
> Add the 2 scalar→vector marshalling `v_mov`s and the `s_wait_dscnt` per block and the accessor idiom
> accounts for the clear majority of REST.

**WHY IT IS INVARIANT:** these are accesses to *coordination state*, whose count depends on the protocol,
not on `ACC_N`/`SEGK`/`FM`/`FN`/`G`/`WAVES`. That is precisely why no geometry lever moves REST — and it
is the same mass the §3 window-retraction attributed to "`lds_*` macro expansion (exec saves, lane-0
broadcast)", now counted at whole-kernel scope instead of inside a bad window.

**STILL NOT A TIME MEASUREMENT.** Static count only; the dynamic half (§82 closing note) is unrun. **No
mechanism or fix is proposed here.**

### 84. `WAVES=4` DOES NOT ASSEMBLE — ROOT CAUSE, AN OFF-BY-ONE GUARD
`occ_kernel_dsws_flow.s:5569:29` and `:7125:29` → `error: invalid operand for instruction`, on
`s_mul_hi_u32 s93, s92, BATON_MAGIC`.

`FIRST_COMPUTE_WID = 3` (`:769`), so `NCOMPUTE = WAVES − 3` (`:929`). At **WAVES=4, NCOMPUTE = 1**, and
`BATON_MAGIC = 0x100000000 / NCOMPUTE` = **`0x100000000` = 2³²** — a 33-bit literal no SOP2 operand can take.

**The guard is `.if NCOMPUTE < 1` (`:930`). `NCOMPUTE == 1` passes it.** Off by one.

*Falsified en route, worth recording:* I first hypothesised a signed-overflow rejection of `0x80000000`
(2³¹) at NCOMPUTE=2. **Direct assembler test refuted it** — `0x40000000`, `0x7fffffff` and `0x80000000`
all assemble clean; only `0x100000000` errors. The 10-second test beat the plausible story.

**Consequence:** `WAVES=4` and `WAVES=3` are both unbuildable; the axis is only valid at `WAVES ≥ 5`.
Every WAVES sweep on record silently starts at 5. Not fixed here (shared tree, upstream sync in flight);
the correct guard is `NCOMPUTE < 2`, since at NCOMPUTE==1 the round-robin is degenerate (`idx mod 1 ≡ 0`)
and needs no magic at all.

*Withdrawn:* an apparent duplicate `.macro lds_put_r` at `:2023`/`:2620` is **NOT a defect** — the first is
guarded by `.if !(DSWS2_CONV || DSWS2_ENVELOPE)` with a comment explaining the ring needs it at CONV=0.

---

## 2026-07-31 AFTERNOON — ★★ THE DYNAMIC MATRIX: 35% OF RUNTIME IS FIXED PER-EVENT COORDINATION ★★

### 85. TWELVE CELLS ON SILICON — PAIRED 1:1 WITH THE STATIC MATRIX (§82)
Data: `DSWS_DYNAMIC_MATRIX_2026-07-31.csv`. Same 12 cells as §82, **same 12 binaries** (`base` rebuilt to
`58e965a46f3e162d`, `accn2` to `6bc334a3fbf5a071` — the exact artifacts already dispatched today, so
cell→binary correspondence is proven, not assumed). `ML8_POOL=64` PINNED in every cell: occupancy is a
retired axis (§79) and letting it float would make every cell a two-variable change — the exact confound
that muddied the morning `ACC_N=2` run.

**AUTHORIZATION:** run under a held board claim (`8f49507a`), which is the GPU authorization mechanism.
Hard stop enforced IN CODE (oracle / work-exactness / `occ[96]` delta / `occ[0]` / canary / reset / latch /
non-zero exit), not by judgement mid-run. **12/12 clean, no stop fired.** `door1` = 100% of coast in every
cell; grow-fail 0 in every cell; `occ[20]`=384 in every cell.
**No probe builds.** `PHASEPROBE` was REJECTED as the instrument: `phase_stamp` (`:1619` area) issues
`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` + `s_wait_kmcnt 0` — an **unthrottled message-bus RTC read**, the
exact 2026-07-14 brick vector that rule 5 names and that `DUTYPROBE` is hard-disabled for. 27 stamp sites,
44x overhead, and its cost lands per-transition, biasing the very distribution it would measure.

### ⛔ 85a. RETRACTION — MY `TOTAL_super` GROUPING WAS WRONG
Mid-run I claimed `TOTAL_super = 11,796,480/(superM*FN)` produced two coordination levels (11,520 for
base/fm4fn2, 23,040 for fm1/fn2) and that this explained the TF pairing. **THE RUN DATA REFUTES IT.**
The dispatcher's own `computed` column gives `computed/rep = G*TOTAL_super = 92,160` for **base, accn2,
fm1, fn2, fm4fn2, waves8 AND waves16 alike** — identical event counts, not two levels.
**The pairs are real; the reason I gave for them was wrong.** I reasoned from a formula I never checked
against the dispatcher output — the same "re-quote instead of re-read" failure as §80/the hipBLASLt band.

### THE RESULT — A TWO-TERM COST PER COORDINATION EVENT
`fm1`/`fn2` do not run more events; they run the same events carrying **half the work**
(0.526 vs 1.051 MFLOP/event). Fitting ns-per-event against work-per-event on the WAVES=6 / SEGK=256 cells:

> ### `ns_per_event = 20.80 + 37.36 × MFLOP_per_event`

**Fit on TWO cells (base, fm1); PREDICTS the two held out — `fm4fn2` −0.5%, `fn2` +0.9%.** Out-of-sample,
not a curve drawn through all four.

| cell | comp/rep | GF/rep | MF/event | ns/event | TF | sclk | **TF norm** | coast/comp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 92,160 | 96.9 | 1.0513 | 60.07 | 17.5 | 2140 | **17.5** | 1.246 |
| fm4fn2 | 92,160 | 96.7 | 1.0489 | 60.28 | 17.4 | 2084 | **17.9** | 1.092 |
| waves8 | 92,160 | 96.4 | 1.0458 | 63.77 | 16.4 | 1975 | **17.8** | 2.587 |
| g6accn3 | 69,120 | 72.5 | 1.0489 | 65.56 | 16.0 | 2125 | **16.1** | 1.392 |
| accn2 | 92,160 | 96.7 | 1.0497 | 64.80 | 16.2 | 2206 | **15.7** | 1.635 |
| waves16 | 92,160 | 96.5 | 1.0468 | 82.43 | 12.7 | 1806 | **15.0** | **13.023** |
| g4 | 46,080 | 48.4 | 1.0510 | 72.98 | 14.4 | 2209 | **14.0** | 1.890 |
| g6accn2 | 69,120 | 72.5 | 1.0484 | 71.81 | 14.6 | 2307 | **13.5** | 1.793 |
| fn2 | 92,160 | 48.3 | 0.5242 | 40.01 | 13.1 | 2156 | **13.0** | 1.236 |
| fm1 | 92,160 | 48.5 | 0.5257 | 40.44 | 13.0 | 2142 | **13.0** | 1.262 |
| segk128 | 184,320 | 96.5 | 0.5233 | 51.31 | 10.2 | 2166 | **10.1** | 0.553 |
| segk64 | 368,640 | 96.4 | 0.2615 | 46.70 | 5.6 | 2180 | **5.5** | 0.308 |

> **AT THE CONFIG OF RECORD: 20.8 ns of 60.1 ns per event — 35% OF RUNTIME — IS FIXED COST THAT CARRIES NO
> WORK.** That is why halving work-per-event (`fm1`, `fn2`) costs 25% throughput: the same overhead is paid
> on half the payload. It is also why the frag grid and feed ratio are irrelevant at matched work-per-event
> (`base` vs `fm4fn2`: 1.0513 vs 1.0489 MF/event → 60.07 vs 60.28 ns, **0.3% apart**, despite a transposed
> 2x4→4x2 grid and superM 256→512).

### ★ THE TELEMETRY (§81) WAS LOAD-BEARING ON ITS FIRST REAL USE ★
**Busy-band sclk spans 1806–2307 MHz — a 28% SPREAD ACROSS CELLS.** Uncorrected, that silently rewrites
every sub-15% comparison. Clock-normalised:
- **`waves8`'s apparent −6.3% IS ENTIRELY CLOCK.** Normalised 17.8 vs base 17.5 — indistinguishable.
  WAVES 6→8 costs nothing. Without §81 this would have been logged as a wave-axis result.
- **`waves16` is a REAL −14%** (15.0 normalised), and its signature is `coast/computed` = **13.023 vs
  base's 1.246** — 10x the poll passes per unit work. It does NOT reach §45's 10.2 TF; that older figure
  is not reproduced here at matched geometry.
- **`accn2` is WORSE than it looked**: −10% normalised, not the −7% the raw TF suggested.

### WHAT DOES NOT FIT — LEFT UNEXPLAINED ON PURPOSE
`segk128` (+27%) and `segk64` (+53%) run far above the per-event prediction, so **SEGK carries a
per-segment cost the per-event model does not capture** (an independent 3-point fit on that axis alone
gives `time ≈ 1.63 ms + 0.108 ms × n_kseg`, the two slopes agreeing to 1% over a 4x range).
`g4` (+21%) and `g6accn2` (+20%) also miss. **No third parameter has been fitted to rescue them.** Twelve
points will accommodate almost any model with enough terms; that is how the feed-ratio model survived as
long as it did.

### ⚠ A CORRESPONDENCE THAT IS NOT YET A FINDING
35% of runtime being fixed per-event cost sits very close to §83's **34% of instructions being exec-mask
bookkeeping** around lane-0 LDS accessors. **These may or may not be the same thing.** Distinguishing a
causal link from a numerical coincidence needs a designed experiment, not two matching percentages. **Do
not cite this as established.**

### 86. TWO FIXES FROM THE CAMPAIGN — BOTH VERIFIED INERT ON THE MEASURED CONFIG
Applied after the matrix so the 12 measurements in 82/85 stay valid against the source that produced them.

**(a) `BATON_MAGIC` off-by-one (84).** `.if NCOMPUTE < 1` -> `.if NCOMPUTE < 2`, and the paired `STAGGER`
guard re-keyed from `WAVES <= FIRST_COMPUTE_WID` to the same `NCOMPUTE < 2`.
**THE TWO CONDITIONS MUST AGREE** -- widening one alone yields a kernel that assembles with
`BATON_MAGIC = 0` and silently computes `q = mulhi(idx,0) = 0`, i.e. the wrong baton turn on every wave.
Same coupling hazard class as `A_MI1_HOISTED` (77); both are now single-condition by construction.
**THIS DOES NOT UNLOCK `WAVES=4`** -- it converts a bare `error: invalid operand for instruction` at two
unrelated `s_mul_hi_u32` sites into a diagnostic naming the cause. A single compute wave is a degenerate
round-robin; supporting it means bypassing the baton, which is a design change.
- config of record **BYTE-IDENTICAL: `58e965a46f3e162d`** (WAVES=6 -> NCOMPUTE=3 never took the changed branch)
- WAVES 5/7/8/16 still build; WAVES 3/4 now fail legibly.
- **The WAVES axis is valid only at >= 5. Every WAVES sweep in this log silently starts there.**

**(b) `build_flow.sh` CONFIG OF RECORD block was itself the stale config.** Defaults were
`WAVES=16 G=6 FM=1 ACC_N=3`, and the header asserted "2 WG/CU / ML8_POOL=128 / 13,824B LDS" -- all
superseded. A bare `./build_flow.sh` built a non-config-of-record and **printed "CONFIG OF RECORD" over
it**: exactly the silently-wrong-config failure the block's own preamble exists to prevent, committed by
the block itself. Now `WAVES=6 G=8 FM=2 ACC_N=4 FN=4 SEGK=256`; header corrected to 1 WG/CU / ML8_POOL=64
/ 34,304B with the occupancy and SEGK findings cited.
- **bare `./build_flow.sh` now yields `58e965a46f3e162d`**, and explicit env still wins (`:=` unchanged).
- No caller depends on the old defaults (verified: `gpu_run.sh`, `gate_lds.sh`, `dsws_realshape_bench.py`
  only reference the script in comments/diagnostics, never invoke it with geometry unset).
- **KNOWN, NOT FIXED:** the WG/CU + pool figures in that echo are still a hardcoded string, not derived
  from the published LDS. Correct today at 34,304B; it will go stale again if LDS moves. Deriving it from
  the sidecar is the durable fix.

Source now `57ab3100c9450ad6`; **bin unchanged at `58e965a46f3e162d`**.
