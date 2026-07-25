# DSWS S1 — N4 CONTINUATION BRIEF (2026-07-20 night). READ FULLY BEFORE TOUCHING ANYTHING.

Self-contained resume doc. Cross-check the config knobs here against DSWS_TESTING_LOG.md (2026-07-20
entries) and the code before running — do NOT execute a stale command from memory. KG session_summary:
2d7ba806. (Recurring lesson: reconcile DECENTASN/BATCHASN/WOFLUSH/BANKZERO/JDEPTH/GROUPS/POOL_N/SSWIN
against the logs, not assumption.)

================================================================================
## 1. HEADLINE — where we are
================================================================================
- **Verified fed throughput: 23.5 TF** (7.7% of 307 TF fp8 peak) at BOTH 4.5s and 9s of steady state,
  oracle CLEAN, WORK-EXACT. Feeding it VALIDATED the number (did not move it). This is the real
  coordination-bound ceiling of the config of record, NOT an underfed artifact.
- **The wall is pinned (phase timer, direct):** the dyn-VGPR grow/shrink ROUND-TRIP is 41% of a compute
  wave's time and buys NOTHING (grow-fail=0, the moat never engages). The matrix math (WMMA) is only 24%.
- Two silent-work-loss races (INITBAR, TERMFIX) were fixed earlier today; both default-on in config of record.
- Tree: kernel bin = config of record 1e78b027ec349396. All host+kernel changes UNCOMMITTED, inert by default.
  Latch clear, card idle.

================================================================================
## 2. CONFIG OF RECORD — exact build + dispatch (verify before use)
================================================================================
BUILD (offline, CPU; produces occ_dsws2_w30_flow_gd.bin sha = 1e78b027ec349396):
```
WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1 BANKZERO=1 STAGGER=1 \
SELFSERVE=1 SSWIN=8 PHIST=0 NOCFLUSH=0 NOBLOAD=0 NOWMMA=0 NODSADD=0 BNDPROBE=0 RESVPROBE=0 BATCH=1 \
INITBAR=1 TERMFIX=1 STAGINSTR=1 TFPROBE=1 DEADMAN=1 ./build_flow.sh
```
DISPATCH — the fed ~9s run (24576x8192x524288), the current best measurement geometry:
```
./gpu_run.sh <name> -- SSWIN=8 FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 \
FLOW_POOL_N=1 DSWS2_SEGK=256 DSWS2_K=524288 DSWS2_ORACLE_MTL=256 DSWS2_ORACLE_NTL=128 \
DSWS2_ORACLE_STRIDE=32768 ML8_COOP_CHUNK=12288 ML8_COOP_CHUNK_MAXS=20.0 STAGINSTR=1 FORENSICS=0 \
TFPROBE=1 ./occ_dispatch --dsws2
```
- Expected computed = G*TOTAL_super. At this shape = 402653184. A short/zero count = WORK-INEXACT = full stop.
- 4.5s variant: DSWS2_ORACLE_NTL=64 DSWS2_K=524288 (= 24576x4096x524288, computed=201326592).
- ALWAYS dispatch via ./gpu_run.sh (enforces single-run, latch-on-hang, deadman<=0.5s, real-disk logs).

================================================================================
## 3. THE WALL — phase-timer breakdown (PHASEPROBE=1, run s1_n4_phaseprobe)
================================================================================
Per compute-wave time: FOLLOW(idle) 1.0% | GROW 33.5% (s_alloc_vgpr grow + rowblk claim) |
WMMA 24.1% | FLUSH 33.7% (split-K ds_add reduction) | SHRINK 7.6% (s_alloc_vgpr shrink).
- dyn-VGPR round-trip GROW+SHRINK = **41%**, reduction = 34%, actual math = only 24%.
- RECONCILES the ablations (NOWMMA/NODSADD were flat because WMMA + the ds_add reduction OVERLAP).
  The ONE phase that CANNOT overlap is s_alloc_vgpr (a WaitIdle barrier) -> GROW+SHRINK is the REAL wall.
- **grow-fail=0**: the dyn-VGPR moat never engages, so the 41% round-trip is pure overhead that never
  converts to wave-multiplexing. coast 41%, ASSIGN-bound ~82% (occ[86]) -- but batch scan (below) proved
  admission/concurrency is NOT the current wall.

================================================================================
## 4. NEXT — THE FORK (dyn-VGPR is the DSWS ethos; discuss, don't decide unilaterally)
================================================================================
(a) MAKE THE MOAT ENGAGE: push the config until the VGPR budget BINDS (grow-fail>0) so the round-trip
    buys wave-multiplexing instead of idling. Must NOT violate the DUTY-CYCLE INVARIANT (TIME AT PEAK ~
    JDEPTH*SEGK; raising SEGK/JDEPTH forbidden). Raising G/concurrency is the candidate. This is the same
    "find the regime where dyn CONVERTS" question as the 2026-06-23 recalibration.
(b) STOP PAYING FOR IT HERE: amortize the round-trip (grow once per reservation, fewer grow/shrink) or a
    bounded-static regime -> cut the 41% directly.

================================================================================
## 5. BUILT TONIGHT (all uncommitted, gated, byte-identical when off)
================================================================================
- HOST LOWMEM + FAST FILL (occ_dispatch.cpp, run_dsws2 + mbg_gen_preshuffle_B): A AND B formula-generated
  straight into VRAM (Aval/Bval), auto-on when an operand >4GB or DSWS2_LOWMEM=1. Host RAM stays FLAT
  (~6GB) at ANY shape -- the --no-mmap equivalent; lets us feed arbitrarily large. Oracle-CLEAN proves the
  formulas are byte-exact. Fill = chunked host staging + bulk memcpy.
  *** TODO: fill is ~90s for ~17GB because Aval/Bval do a per-element integer division (i/Ko). i/Ko is
      CONSTANT per row -> fill row-by-row to kill the div (~6x faster). ***
- RESVPROBE (kernel, occ[87]=CAS-loss occ[89]=window-full; reuses CNT_FATFULL s94 / CNT_CLEAD s96, both
  structurally 0 at config of record; RESVPROBE=0 byte-identical). Measured 1.466 CAS collisions/reserve
  -- real but NOT the binding wall.
- BATCH reservation (kernel; BATCH=1 byte-identical, N-claim per CAS): SHELVED WORKING LEVER. Correct on
  silicon (WORK-EXACT+oracle CLEAN at BATCH 2 and 4) but a NET LOSS now because concurrency isn't the wall
  (scan: TF flat 21.0->20.5 while holders>=4, cliffs to 16.0 at holders=2). Un-shelve when reservation
  contention binds; eager-stamp variant (stamp all N up front) avoids the window-starvation seen here.
  See PLAN_CURSOR_BATCH.md (marked SHELVED).
- NODSADD (kernel, gated): ds_add reduction ablation -- layout-confounded, SUPERSEDED by the phase timer.

================================================================================
## 6. GOTCHAS BANKED TONIGHT
================================================================================
- n_kseg (= K/SEGK) MUST BE A POWER OF 2. K=655360 -> n_kseg=2560 (non-pow2) -> computed=0 (kernel handed
  out NO work: 100% coast, 14ms, TF=18082 garbage) -> WORK-INEXACT latch. USE K = SEGK * pow2
  (524288->n_kseg=2048; 1048576->4096).
- DSWS2_ORACLE_MTL / DSWS2_ORACLE_NTL set the SHAPE: M = 96*MTL (96 = super-tile M), N = 64*NTL. Mislabeled
  "oracle". DSWS2_ORACLE_STRIDE is the separate oracle SAMPLING knob (high = fast oracle; correctness spot-check).
- Fed-run VRAM: A = M*K DOMINATES. To feed LONGER at fixed A, grow N (grows B=K*N and C, both smaller than A),
  not just K. 24576x4096 can't reach 10s (A>32GB at K-for-10s); 24576x8192xK524288 = ~9s at ~22GB.
- The CPU oracle is O(sampled_tiles * KT) -> slow at deep K. Use DSWS2_ORACLE_STRIDE high for feed runs
  (correctness already proven). WORK-EXACT + canary cover count+OOB; oracle covers value on a few tiles.

================================================================================
## 7. OPEN TODO (tomorrow)
================================================================================
1. THE FORK (section 4) -- the main decision.
2. hipBLASLt same-shape bar: tonight `python3 ~/dsws_gpu_logs/bench_hipblaslt_thisshape.py` (BM/BN/BK env)
   threw a wall of "register fat binary failed" (HIP code-object/ROCm-load env issue, NOT our kernel). Prior
   working baseline (2026-07-13, bench_hipblaslt_ml8.py) got hipBLASLt 12.6-190 TF on ml8 shapes. Fix the env
   (torch/ROCm), then run at 24576x8192x524288 for the DSWS-vs-vendor bar.
3. Fill speed (section 5 TODO).

================================================================================
## 8. GPU SAFETY (unchanged, non-negotiable)
================================================================================
board_check IMMEDIATELY before every board_claim; release by claim_id. ONE dispatch per greenlight. A
changed kernel = ONE bring-up then stop. A hang/timeout/INCOMPLETE/WORK-INEXACT = FULL STOP, latch cleared
only by a human. NEVER raise DEADMAN_TICKS. Nothing new in the hot path touching the message bus or emitting
stores. Max work offline first. kmbandy does NOT hold a display/duration rule (a display/ags hiccup recovers;
don't lecture about it) -- but calculate HOST RAM (16GB box) and VRAM out loud before launch.

INCIDENTS TONIGHT (own them, all recovered, no bricks): 2x host OOM (host built full A/B vectors pre-lowmem-fix);
1x ags crash (GTK app, VRAM contention, displays fine); 1x computed=0 latch (non-pow2 n_kseg). The lowmem host
fix removes the OOM class entirely.
