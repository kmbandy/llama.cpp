// occ_kernel_dsws.s  (gfx1201, wave32) -- MAD-305 DSWS v2 SUBSTRATE SCAFFOLD (PLAN_DSWS_SUBSTRATE_V2.md, Task A1).
//
// v2 re-founds the DSWS GEMM substrate so matrix work is *claimed* (not owned by compile-time wave id), with
// split-K folded in. Work = a pool of (mblk, tcol, ksi) super-tiles; a pinned claimer (wid 0) broadcasts the
// current super-tile; live waves of each role drain shared LDS atomic counters against resident-in-LDS A/B.
//
// Task A1 is SCAFFOLD-ONLY: lift the proven prologue/arming setup from occ_kernel_coop.s (NEVER modified),
// assert the v2 LDS layout fits the 32 KB group segment, and leave each role as a distinct s_endpgm stub.
// The real claimer / feed / compute loops land in A3..A7. The only gates here: (1) assembles clean,
// (2) RGA 0-spill, (3) harness --dsws2 dry-prints the computed params.
//
// ============================================================================================
//  v2 KERNARG CONTRACT (USER_SGPR=15, s0..s14; hardware-preloaded user SGPRs). The host A8 launch
//  MUST set COMPUTE_PGM_RSRC2.USER_SGPR=15 and load COMPUTE_USER_DATA_0..14 to match EXACTLY:
//    s0:s1 = occ buffer base  (>=0x1000B; host zero-inits; see CLAIM-COUNTER / completion offsets below)
//    s2:s3 = A   base (fp8 e4m3, row-major, 1 byte/elem)
//    s4:s5 = Bshuf base (the shuffled-B layout the coop B-feed consumes; same global_load_tr_b64 idiom)
//    s6:s7 = C   base (fp32; HOST MUST MEMSET C=0 before dispatch -- compute uses global_atomic_add_f32)
//    s8    = KT          (total K16-steps for the whole matrix = K/16)
//    s9    = K(bytes/A-row)   (= K, fp8 1 byte/elem)
//    s10   = NT*256      (B-saddr advance per K16-step)
//    s11   = TOTAL       (coop-style total *tiles* = MTL*NTL; carried for addressing compat, NOT the pool size)
//    s12   = magic(ceil(2^32/NTL))     (unsigned-div magic for /NTL ; tcol/mblk decode)
//    s13   = NTL         (number of N tile-columns)
//    s14   = FN*256      (B-saddr stride per N-frag)
//    (TGID_X now lands in s15 -- UNUSED; this kernel is pool-claim, not workgroup-id based.)
//  NOTE: G and SEGK are COMPILE-TIME defsyms (baked into instruction immediates); they are NOT kernargs.
//        FIX 1 (round-table Opus+Codex pass): v1 of this contract passed n_kseg/TOTAL_super/magic_kseg as
//        s15/s16/s17, but the PM4 host only preloads COMPUTE_USER_DATA_0..15 (USER_SGPR<=16; every proven
//        launch path in this tree uses 15) -- s16/s17 could NEVER actually arrive in hardware SGPRs, AND
//        s16 was independently being reused per-chunk on the host as the compositor-safe chunk terminal
//        (a second, unrelated collision on the same slot). This file now drops s15/s16/s17 entirely:
//          n_kseg  is DERIVED in-kernel from KT (s8) and the compile-time KSEG_STEPS=SEGK/16:
//                  n_kseg = KT >> NKSEG_SHIFT, where NKSEG_SHIFT=log2(KSEG_STEPS) is a compile-time `.set`
//                  (small .if ladder over KSEG_STEPS in {1,2,4,8,16}; SEGK is always a power-of-two
//                  multiple of 16, so KSEG_STEPS is always a power of two in that set).
//          shift/mask (the sti -> (t,ksi) split) are derived ONCE in the prologue from n_kseg:
//                  shift = s_ff1_i32_b32(n_kseg)   (bit index of n_kseg's single set bit; n_kseg=1 -> 0)
//                  mask  = n_kseg - 1
//                  DECODE_STI then does  ksi = sti & mask ; t = sti >> shift  -- this handles n_kseg=1 for
//                  free (shift=0, mask=0 -> ksi=0, t=sti), so the old magic-div n_kseg==1 special-case is
//                  GONE (it's no longer needed, not just hidden).
//          the chunk terminal (old TOTAL_super/"chunkHi") is now MEMORY-CARRIED instead of a kernarg: the
//                  host writes the current chunk's terminal sti bound to occ[24] (occW[6]) once per chunk;
//                  the claimer reads occ[24] ONCE per dispatch (stable for the whole chunk) instead of
//                  receiving it as a broadcast kernarg. On sti >= occ[24] the claimer publishes a SENTINEL
//                  (0xFFFFFFFF) into STI_OFF instead of the raw over-claimed sti; followers (b-feed/a-feed/
//                  compute) retire when STI_OFF == 0xFFFFFFFF instead of comparing against the (now
//                  nonexistent) TOTAL_super kernarg.
//        See "CLAIM-COUNTER & completion occ offsets" by .Lclaimer for the full occ-buffer layout
//        (occ[24]/occW[6] = chunk terminal bound, added by FIX 1).
//
//  SCALAR REGS (derived in the prologue, before any clobber; none collide with DECODE_STI's own clobber
//    list s18/s36, lds_*'s s49, the claimer's s16/s17/s35/s44, or any role body's transients, all <= s65):
//      s66 = n_kseg   (derived; dead after shift/mask below are computed -- kept only for that derivation)
//      s67 = mask     (n_kseg - 1) -- LIVE for the whole kernel; read by every DECODE_STI call, every role.
//      s68 = shift    (log2 n_kseg) -- LIVE for the whole kernel; read by every DECODE_STI call, every role.
//      s69 = chunkHi  (claimer-only; loaded once per dispatch from occ[24] right before .Lclaim_loop).
//
// Everything new is gated behind the fresh `DSWS2` build symbol (analogous to coop's `DSWS`).

.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"

// ---- tile defsyms (lifted from occ_kernel_coop.s) ----
.ifndef FM
    .set FM, 2                              // per-compute-wave M-frags (M-band = FM*16 rows)
.endif
.ifndef FN
    .set FN, 4                              // shared N-frags (the reuse operand)
.endif
.ifndef RGADESC
    .set RGADESC, 0                         // 1 = emit analysis-only AMDHSA descriptor for RGA livereg
.endif
.ifndef DIAG
    .set DIAG, 0                            // 1 = phase-marker instrumentation (unused in the A1 scaffold)
.endif
.ifndef SAFEPROBE
    .set SAFEPROBE, 0                       // 1 = clamp per-lane vector address regs into a provable in-buffer bound
.endif
.ifndef TFPROBE
    .set TFPROBE, 0                         // 1 = emit s_sendmsg_rtn GET_REALTIME wall-span capture (each wave stamps
.endif                                      //     occ[2]=min entry tick, occ[3]=max exit tick -> host TF readout).
                                            //     Default 0 => ZERO bytes emitted, .text byte-identical to the
                                            //     production bin (uses only already-allocated regs s30/s31/s49/v5).
.ifndef PHASEPROBE
    .set PHASEPROBE, 0                      // 1 = in-kernel PHASE TIMER: each compute wave stamps GET_REALTIME at every
.endif                                      //   phase boundary and atomic-adds the delta into per-phase occ accumulators
                                            //   (occ[64..69], bytes 256..276, ABOVE the per-chunk memset -> accumulate over
                                            //   the whole run). Host prints ticks + % per phase. Default 0 => byte-identical.
.ifndef BANKZERO
    .set BANKZERO, 1                        // 1 = ZERO the LDS accumulator banks ONCE PER TILE and make EVERY ksi an
.endif                                      //     ds_add_f32. 0 = the old ksi==0 fresh-write (ds_store_b32).
                                            // WHY (measured 2026-07-13): the fresh-write is what forced POOL_N=1.
                                            //   With a deep pool, ksi=0 and ksi=1 are in flight together, and if
                                            //   ksi=1's ds_add_f32 lands FIRST, ksi=0's ds_store_b32 WIPES IT OUT.
                                            //   The "POOL_N=1 required for correctness" note was a WORKAROUND that
                                            //   disabled the N-deep pool -- and the pool is the whole flow economy.
                                            //   POOL_N 1->3 drops FOLLOW_WAIT from 75.9% to 4.3%.
.ifndef FORENSICS
    .set FORENSICS, 0                       // !!! DEFAULT 0 -- see the RDNA4 dyn-VGPR VGPR-WRITE HAZARD below. !!!
                                            // The SPIN GAUGES (flow_gauge at SHRINK/TASHRINK) sit adjacent to
                                            // s_alloc_vgpr, and ANY VALU VGPR WRITE there CORRUPTS THE REGISTER FILE
                                            // (measured 2026-07-13; see the s_alloc_vgpr guard banner). flow_snapshot
                                            // itself is INNOCENT (proven CLEAN in isolation) -- it is only the two
                                            // gauges bracketing s_alloc_vgpr that are unsafe. Until they are relocated
                                            // to a point with no realloc nearby, FORENSICS stays OFF.
                                            //   FORENSICS=0 STAGINSTR=1 -> oracle CLEAN 6/6 WITH working counters.
                                            // Original intent (still right, re-enable once the gauges are moved):
                                            // ALWAYS-ON hang/fault telemetry: flow_snapshot (coordinator cursors) + the
.endif                                      //   completer-SPIN gauges. Deliberately SPLIT from STAGINSTR: a build under
                                            //   test must never be able to compile out its own black box. (2026-07-12: a
                                            //   STAGINSTR=0 A/B bricked the box and could not explain its own death --
                                            //   every forensic field logged 0, because the flag under test also gated the
                                            //   forensics.) Every FORENSICS site fires where the WMMA ACC regs are DEAD.
.ifndef FATGAUGE
    .set FATGAUGE, 0                        // PEAK-concurrency gauge (fat_inc/fat_dec -> occ[57] FATLIVE, occ[58] FATMAX).
.endif                                      //   A live cross-wave gauge CANNOT be made wave-local, so it inherently needs
                                            //   global atomics in the burst path -> it PERTURBS. Opt-in, default OFF, and
                                            //   never trust a correctness run with it on.
.ifndef WOFLUSH
    .set WOFLUSH, 0                         // BURST-SCOPED FLUSH (LDS-halving lever, council 2026-07-05): 1 = drop the
.endif                                      //   per-rowblk LDS accumulator banks entirely; each compute burst atomic-adds
                                            //   its fp32 ACC frags DIRECTLY to C (global_atomic_add_f32, fp32-exact, same
                                            //   addresses as the write-once completer store). Build with ACC_N=0 (host
                                            //   DSWS2_ACC_N=0) -> LDS/WG ~8KB -> ~7 WGs/CU -> per-SIMD VGPR pool BINDS ->
                                            //   the dyn-VGPR traveling-peak finally engages (grow-fail>0). Re-incurs the
                                            //   n_kseg-x C-write atomic traffic write-once removed; tunable later by burst
                                            //   K-depth J + KMAJOR. Default 0 => byte-identical to the write-once bin.
.ifndef NOCFLUSH
    .set NOCFLUSH, 0                        // PERF PROBE ONLY: 1 = skip the global_atomic_add_f32 C-flush loop (keep ALL
.endif                                      //   other bookkeeping/handshake). Isolates the device-atomic C-reduction cost
                                            //   from the coordination handshake. Result is WRONG (C never written -> oracle
                                            //   fails) -- span/TF only. Default 0 => byte-identical.
// ============================================================================================
// K-DEPTH J (2026-07-13). THE flush lever. Measured: the C flush is 57-97% of runtime, and
//   flush/WMMA = 128/SEGK -- so the ONLY way to pay it down is more K accumulated per flush.
//
//   SEGK does that by making each super-tile deeper -> but OPSTRIDE = SEGK*16*(FN+G*FM), so it
//   costs LDS *linearly*, and we hit the 32KB operand-pool cap at SEGK=128 (SEGK=256 is
//   unreachable). SEGK is a lever we have already spent.
//
//   J does the same thing for FREE: the wave keeps its ACC in *registers* across J consecutive
//   ksi of the SAME rowblk and flushes ONCE. J-fold fewer ds_add_f32, ZERO extra LDS -- it walks
//   the same POOL_N slot buffers sequentially as they re-stage, instead of demanding J resident.
//
//   OWNERSHIP (the subtle part): the carrier holds rowblk r across slots WITHOUT re-claiming it,
//   so a *fresh* wave arriving at a mid-group slot must not claim r again. The coordinator
//   therefore POISONS SL_RBNEXT = ACC_N on every non-lead slot (ksi % J != 0), and the existing
//   `r >= ACC_N -> tryadv` check turns fresh waves away for free. Only ksi%J==0 admits claims.
//
//   DEADLOCK: the ACC_N carriers sit FAT waiting for their next segment to stage. Someone must
//   still be LEAN to stage it -> WAVES must exceed ACC_N (guarded below). deadman_check runs in
//   the wait loop so a genuine stall retires cleanly instead of wedging the queue.
//
//   REQUIRES: J a power of 2 and J | n_kseg (n_kseg is already a power of 2).
.ifndef JDEPTH
    .set JDEPTH, 1                          // 1 = OFF, byte-identical to the known-good 8.8 TF bin.
.endif
.if (JDEPTH & (JDEPTH - 1)) != 0
  .error "JDEPTH must be a power of 2 (it must divide n_kseg, and ksi%J is done with an AND mask)"
.endif
.ifndef STAGGER
    .set STAGGER, 0                              // 0 = OFF, byte-identical to the known-good bin.
.endif
.ifndef RELSTART
    .set RELSTART, 1                             // *** THE BATON *** (non-blocking, FLOW_ECONOMY_DESIGN.md): 1 = fat_release
.endif                                          //   at shrink-START -> a shrinking carrier returns budget the instant it
                                                //   commits to shrink; the next carrier's fat_acquire grabs it on its next
                                                //   loop pass (no wave waits on any other -- the peak travels via demand).
                                                //   0 = release at shrink-END (the pristine cap, for A/B). There is NO
                                                //   explicit wait: a refused carrier COASTS+retries (the flood model).
.ifndef BATONGATE
    .set BATONGATE, 1                            // 1 = the traveling-peak BATON (GROWPERMIT push-mailbox grow-turn
.endif                                           //   handoff) IS the grow-gate; the FATTOK/MAXFAT software token layer
                                                 //   (fat_acquire/fat_release) is compiled to no-ops and the MAXFAT cap
                                                 //   guard is bypassed -- the physical s_alloc_vgpr grow-fail is the only
                                                 //   throttle (the river). 0 = old software token cap, for A/B. Only
                                                 //   meaningful under STAGGER=1 (STAGGER=0 is byte-identical either way).
.ifndef MAXFAT
    .set MAXFAT, 0
.endif
.if MAXFAT == 0                                  // 0 == "no cap" -> the cap IS ACC_N (today's behaviour).
    .set MAXFAT_EFF, ACC_N                       //   (-defsym symbols cannot be re-.set, hence the derived symbol)
.else
    .set MAXFAT_EFF, MAXFAT
.endif
// ---- STAGER-SUPPLY GUARD (rewritten 2026-07-14). The old form was `WAVES >= 2*ACC_N`, which assumed
//   ALL ACC_N rowblk-carriers can be fat SIMULTANEOUSLY. That is only true with STAGGER=0. It is what
//   welded G (=ACC_N = rowblk supply = our single biggest throughput lever) to the wave budget and
//   pinned G at 15. *** THE THROTTLE IS EXACTLY WHAT BREAKS THAT WELD: with STAGGER=1 at most
//   MAXFAT_EFF waves are EVER fat, so WAVES-MAXFAT_EFF waves are GUARANTEED lean and free to stage. ***
//   So the real requirement is a stager SUPPLY, not a 2x wave tax:  WAVES >= MAXFAT_EFF + STAGERS.
.ifndef STAGERS
    .set STAGERS, 4                              // lean waves guaranteed available to feed the carriers
.endif
.ifndef GRELAX
    .set GRELAX, 0                               // T5 (2026-07-16): 1 = relax the JDEPTH>1 lean-stager guard from
.endif                                           //   WAVES>=2*ACC_N to WAVES>=ACC_N+STAGERS, letting G exceed the
                                                 //   30-wave/2*ACC_N cap (G>15). Tests whether more compute BREADTH lifts
                                                 //   TF past the 37-TF G=15 ceiling. Liveness (no stage-starve) is
                                                 //   EMPIRICAL -- a GRELAX build MUST pass a supervised bring-up before trust.
.if JDEPTH > 1
  .if STAGGER && !BATONGATE
    .if WAVES < (MAXFAT_EFF + STAGERS)
      .error "JDEPTH>1 + STAGGER needs WAVES >= MAXFAT_EFF + STAGERS: only MAXFAT_EFF waves can be fat at once, but at least STAGERS lean waves must remain to stage their next segment. Lower MAXFAT, lower STAGERS, or raise WAVES."
    .endif
    .if MAXFAT_EFF >= ACC_N
      .error "JDEPTH>1 + STAGGER with MAXFAT >= ACC_N is the UNTHROTTLED case -- it needs WAVES >= 2*ACC_N. Set MAXFAT < ACC_N (that is the whole point of the stagger) or build STAGGER=0."
    .endif
    .if JDEPTH > POOL_N
      // *** MISSING GUARD, added 2026-07-16 after this exact config (JDEPTH=4 POOL_N=3) deadlocked fed on the
      //   deep-K shape: computed=0, ASSIGN stuck, carriers stage-starved, deadman force-retired 175 fat carriers.
      //   In the THROTTLED stagger case (MAXFAT<ACC_N, enforced just above) a capped deep-J carrier reaches
      //   JDEPTH super-tiles ahead of DRAIN, but the ASSIGN window is only POOL_N deep and DRAIN cannot advance
      //   until ALL ACC_N rowblks of a ksi complete -- which the throttle prevents from happening in order -- so
      //   the carrier's JDEPTH-th segment can never be staged. CONFIRMED across 0714 runs: J=4/POOL_N=4 clean,
      //   J=8/POOL_N=4 broke, J=4/POOL_N=3 broke. (STAGGER=0 is UNAFFECTED -- all waves fat, no window limit --
      //   which is why the deep-J sweep ran J=8/16/32 clean at POOL_N=3.) ***
      .error "JDEPTH>1 + STAGGER (throttled) needs JDEPTH <= POOL_N: a capped deep-J carrier reaches JDEPTH super-tiles ahead but the ASSIGN window is only POOL_N deep -> its JDEPTH-th segment never stages -> stage-starve deadlock. Lower JDEPTH to <= POOL_N, or raise POOL_N (watch the LDS cap), or build STAGGER=0."
    .endif
  .else
    .if GRELAX
      // T5: relaxed lean-stager floor. The 2*ACC_N rule assumed ALL ACC_N carriers fat simultaneously; with
      //   budget-throttling (only ~VBUDGET-worth fat at once) the rest are lean and CAN stage, so ACC_N+STAGERS
      //   MAY suffice -- letting G exceed 15. Whether it actually stays live is EMPIRICAL (supervised bring-up).
      .if WAVES < (ACC_N + STAGERS)
        .error "GRELAX: JDEPTH>1 needs WAVES >= ACC_N + STAGERS (relaxed floor). Lower ACC_N/STAGERS or raise WAVES."
      .endif
    .else
      .if WAVES < (2 * ACC_N)
        .error "JDEPTH>1 needs WAVES >= 2*ACC_N: ACC_N waves sit FAT carrying ACC while waiting for their next segment, so there must be at least as many other waves left LEAN to stage it -- otherwise the carriers deadlock the feeder. (Or turn the STAGGER on: MAXFAT < ACC_N caps concurrent-fat and frees the rest to stage.)  Or GRELAX=1 to try WAVES>=ACC_N+STAGERS."
      .endif
    .endif
  .endif
.endif
.ifndef KMAJOR
    .set KMAJOR, 0                          // PERF PROBE: 1 = K-MAJOR super-tile traversal. Default decode packs ksi in
.endif                                      //   the LOW bits (all n_kseg segments of a C cell claimed consecutively ->
                                            //   up to n_kseg WGs hammer one C cell at once = max atomic contention).
                                            //   KMAJOR decodes ksi = sti / TOTAL (high), t = sti % TOTAL (low) via a
                                            //   magic-div (magic_TOTAL from occ[62], loaded to s76 in prologue) -> the
                                            //   32 segments of a cell are spread TOTAL apart in claim order (near-zero
                                            //   concurrent contention) + adjacent tiles (shared operand bands) claimed
                                            //   close in time (L2 reuse). Correctness-preserving (C add is commutative).
.ifndef CSTORE
    .set CSTORE, 0                          // PERF PROBE ONLY: 1 = replace the flush's global_atomic_add_f32 with an
.endif                                      //   equal-count NON-atomic global_store_b32 (same #mem-ops, same addresses, NO
                                            //   RMW/contention). Isolates atomic-contention from raw write-bandwidth: if
                                            //   TF jumps vs atomics -> contention; if flat -> bandwidth-bound. Result WRONG
                                            //   (last-writer, no accumulation) -- span/TF only. Default 0 => byte-identical.
.ifndef TRACE
    .set TRACE, 0                           // 1 = per-super-tile CLAIMER trace: append one row/super-tile {tick, segcnt,
.endif                                      //     epoch, nComp/nAfeed/nBfeed live role slots, ring occA/occB peak,
                                            //     convCount, vresv, sti, quiesce} to a host-provided buffer (VA in
                                            //     occ[52:53], cap in occ[54]). Time-series of the adaptive wave-role
                                            //     economy. Requires DSWS2_CONV=1 (rows written in the quiesce path).
                                            //     Default 0 => ZERO bytes; uses free high SGPRs s70..s74.
.ifndef DYNVGPR
    .set DYNVGPR, 1                         // 1 = compute waves s_alloc_vgpr-grow per rowblk; feeds/claimer stay lean 32
.endif
.ifndef SLEEPN
    .set SLEEPN, 2                          // s_sleep arg in the busy-waits (yield issue cycles to partner waves)
.endif

// ============================================================================================
// DSWS v2 LDS layout (bytes from group-segment base; words u32 unless noted). Mirrors the placement
// of the coop file's LDS `.set` block. Defined unconditionally (uses only G/SEGK/FM/FN, always set).
// ============================================================================================
.ifndef DSWS2
  .set DSWS2, 0
.endif
.ifndef G
  .set G, 6            // cooperative M-extent (rowblks per super-tile) = LDS accumulator-bank count (ACC_N)
.endif
.ifndef SEGK
  .set SEGK, 64        // split-K segment size in K-elements (multiple of 16)
.endif
// ---- v2 control/claim words ----
.set STI_OFF,        0      // broadcast super-tile id
.set EPOCH_OFF,      4
.set ROWBLK_NEXT_OFF, 8     // per-super-tile rowblk claim counter
.set ROWBLK_DONE_OFF, 12    // per-super-tile completion counter
.set BFRAG_NEXT_OFF, 16     // B-frag claim counter
.set AROW_NEXT_OFF,  20     // A-rowblk claim counter
.set NCOMP_SLOT,     24
.set NAFEED_SLOT,    28
.set NBFEED_SLOT,    32
.set GATE_OFF,       36     // u32[4] -> 36,40,44,48 (conversion gates)
.set VRESV_OFF,      52     // vgpr_reserved
.set SEGCNT_OFF,     56     // controller clock
// ---- A3..A7 additions (still inside the 0..256 control region; A1 offsets 0..56 unchanged) ----
.set BFRAG_DONE_OFF, 60     // B-frag STORE-completion counter (compute gates on this, NOT the claim ctr)
.set AROW_DONE_OFF,  64     // A-rowblk STORE-completion counter (compute gates on this)
.set INITFLAG_OFF,   68     // barrier-free LDS-init publish flag (claimer writes 0xACED LAST)
// ---- Phase-B (DSWS2_CONV) control state: role-mix snapshot slots + quiesce counter ----
//   Based at INITFLAG_OFF+4 (NOT the brief's SEGCNT_OFF+4): the brief predates the A3..A7 control
//   words (BFRAG_DONE/AROW_DONE/INITFLAG at 60/64/68), so SEGCNT_OFF+4=60 would collide with them.
//   Basing after the LAST control word keeps the new state inside the 0..255 control gap BELOW the
//   fixed resident region (BRES_OFF=256), so NO resident-region repoint is needed -- the resident
//   BRES_OFF/ARES_OFF immediates (emitted unconditionally in the kernel body) stay untouched, which
//   is what keeps the DSWS2_CONV=0 binary byte-identical to the Phase-A green bin. All `.set`s here
//   are inert (emit no bytes); the only new code (claimer init) is gated under `.if DSWS2_CONV`.
.ifndef DSWS2_CONV
  .set DSWS2_CONV, 0        // 0 = pre-conversion static substrate (Phase A green); 1 = Phase B
.endif
.ifndef DSWS2_TICKET_SELFTEST
  .set DSWS2_TICKET_SELFTEST, 0   // DIAG-only try_gate single-winner smoke (Task 4 Step 3); default 0 = no bytes
.endif
.ifndef CONV_COOLDOWN
  .set CONV_COOLDOWN, 0     // Task 4: per-wave post-conversion cooldown epochs. 0 = spec-faithful (no
                             //   cooldown, byte-identical to pre-Task-4); >0 damps thrash (skip N epochs
                             //   of watermark decision after a wave converts role).
.endif
// Task 5: deterministic bring-up hook. DSWS2_FORCE=1 makes exactly wave DSWS2_FORCE_WID convert
//   direction DSWS2_FORCE_DIR at epoch DSWS2_FORCE_EPOCH, watermarks bypassed -- a reproducible,
//   single-wave/single-epoch GPU proof of role conversion. Default DSWS2_FORCE=0 emits ZERO bytes
//   (byte-identical to pre-Task-5).
.ifndef DSWS2_FORCE
  .set DSWS2_FORCE, 0
.endif
.ifndef DSWS2_FORCE_WID
  .set DSWS2_FORCE_WID, 0
.endif
.ifndef DSWS2_FORCE_DIR
  .set DSWS2_FORCE_DIR, 0            // 0/1 = compute->A/B ; 2/3 = A/B->compute
.endif
.ifndef DSWS2_FORCE_EPOCH
  .set DSWS2_FORCE_EPOCH, 1
.endif
// Rolling dyn-VGPR sum-envelope (2026-07-02 spec). ENVELOPE routes the per-rowblk compute burst grow
//   through the shared vgpr_reserved counter so at most PEAK_CONC waves hold peak at once (the
//   multi-grower collision, ISA 3.3.3.2, becomes unreachable). All default to the byte-identical value:
//   ENVELOPE=0/STAGGER=0 emit ZERO new bytes and PEAK_CONC/STAGGER_PERIOD are inert unless their gate is on.
.ifndef DSWS2_ENVELOPE
  .set DSWS2_ENVELOPE, 0        // 1 = route the per-rowblk compute burst grow through the vgpr_reserved
.endif                          //     sum-envelope. 0 = HEAD (bare .Lcompute_grow) -> .text byte-identical.
.ifndef PEAK_CONC
  .set PEAK_CONC, 2             // concurrent compute peaks the budget admits (R3 sweep). Used iff ENVELOPE=1.
.endif
.ifndef DSWS2_STAGGER
  .set DSWS2_STAGGER, 0        // 1 = lock-free phase-token stagger (Task 9). 0 -> emergent envelope stagger.
.endif
.ifndef STAGGER_PERIOD
  .set STAGGER_PERIOD, 4       // phase slots in the stagger ring (R3 sweep). Used iff STAGGER=1 (inert here).
.endif
// SNAP_BASE / QUIESCE_CNT_OFF -- *** RELOCATED below FATTOK_OFF (they ALIASED the ROLE mailbox) ***
// SENSOR FIX: the claimer publishes its MID-DRAIN ring-occupancy PEAK here each super-tile; the conversion
//   decisions read THESE instead of sampling occ_sample at their own quiesce (where occ_X reads ~0 post-drain
//   -> always "starved" -> the 4/2/2->1/6/1 compute->feed runaway). Mid-drain peak = the true demand signal.
// OCCA_PUB_OFF / OCCB_PUB_OFF -- *** RELOCATED below FATTOK_OFF (same alias) ***
.set DSWS2_STATE_END,(OCCB_PUB_OFF + 4)
// DSWS2_GQUIESCE (2026-07-02 SUSPECT #2 candidate fix): route the QUIESCE handshake through a DEVICE-SCOPED
//   GLOBAL atomic in the uncached occ buffer (byte QUIESCE_GOFF), mirroring the GREEN occ[20] claim/occ[0]
//   live handshake, instead of the barrier-free LDS counter (whose cross-wave visibility is unguaranteed and
//   is the leading SUSPECT #2 hang mechanism). occ buffer = AllocGpu 0x1000 (1024 u32, uncached); host uses
//   occ[0..6] + DIAG scratch (<= byte 116); byte 200 (occ[50]) is provably free. Default 0 => LDS path,
//   .text byte-identical. Requires DSWS2_CONV (QUIESCE only exists there).
.ifndef DSWS2_GQUIESCE
  .set DSWS2_GQUIESCE, 0
.endif
.set QUIESCE_GOFF,   200                         // occ[] byte offset for the global QUIESCE counter (occ[50])
// ---- TRACE (per-super-tile time-series) occ handshake words + row layout ----
.set TRACE_PTR_OFF,  208                         // occ[52:53] = trace buffer VA (host writes lo/hi per chunk)
.set TRACE_CAP_OFF,  216                         // occ[54]    = MAXROWS (host-provided row capacity)
.set TRACE_IDX_OFF,  220                         // occ[55]    = GLOBAL row-claim counter (all WGs' claimers share it)
.set TRACE_WGID_OFF, 224                         // occ[56]    = GLOBAL wg-id dispenser (claim-order 0..pool-1)
.set FATLIVE_OFF,    228                         // occ[57]    = live count of GROWN (fat NFV-VGPR) compute waves
.set FATMAX_OFF,     232                         // occ[58]    = PEAK concurrent fat waves -> x NFV = VGPR in flight (== B probe)
.set ALLLIVE_OFF,    240                         // occ[60]    = live count of ALL resident waves (++entry/--exit)
                                                 // occ[1] (byte 4) = PEAK concurrent resident waves (vs 2048 HW ceiling)
.set TRACE_ROW_BYTES, 64                         // 16 u32/row
// DSWS2_BAILMARK (SUSPECT #2 localization, 2026-07-03): each follower publishes its OWN epoch (s35) to a
//   PER-WAVE occ slot (BAIL_BASE + wid*4) at its _quiesce bail. One-shot per super-tile per wave -> minimal
//   timing perturbation (NOT the claimer's per-spin DIAG poll stores, which are the heisenbug source and stay
//   DIAG-only). After a watchdog abort the host reads occ[BAIL_BASE/4 + wid]: every follower's slot == the
//   hung epoch => all reached their bail (=> a QUIESCE visibility/lost-update, gq relevant); ONE slot stale
//   at the prior epoch => that exact wave is the STRAGGLER (stuck in _alloc/_init/_follow; gq irrelevant).
//   Per-WAVE (not per-role): 4 compute share one role, so a role mark's last-writer-wins would hide a single
//   straggler. Default 0 => no bytes, .text byte-identical. Requires DSWS2_CONV.
.ifndef DSWS2_BAILMARK
  .set DSWS2_BAILMARK, 0
.endif
.set BAIL_BASE,      160                         // occ[] byte offset base for per-wave bail marks: occ[40..47]
.set CONVCNT_OFF,    192                         // occ[48]: DIAG conversion-commit counter (proves waves switch role)
                                                 //   (host prints occ[40..47] as BAIL[w0..w7]; clear of the
                                                 //    occ[32..36]/occ[39] DSWS sensor+roles slots and occ[50] gq)
.set KSEG_STEPS,     (SEGK/16)             // K16-steps per split-K segment = SEGK K-elements / 16
// FIX 1(b): NKSEG_SHIFT = log2(KSEG_STEPS), so the prologue can derive n_kseg = KT >> NKSEG_SHIFT instead
//   of receiving it as a (now-dropped) kernarg. SEGK is always a power-of-two multiple of 16 in every
//   config this file is built with, so KSEG_STEPS is always a power of two in {1,2,4,8,16}; a static
//   ladder over that small set is simpler/safer than a general-purpose compile-time log2.
.if KSEG_STEPS == 1
  .set NKSEG_SHIFT, 0
.elseif KSEG_STEPS == 2
  .set NKSEG_SHIFT, 1
.elseif KSEG_STEPS == 4
  .set NKSEG_SHIFT, 2
.elseif KSEG_STEPS == 8
  .set NKSEG_SHIFT, 3
.elseif KSEG_STEPS == 16
  .set NKSEG_SHIFT, 4
.else
  .error "KSEG_STEPS (SEGK/16) must be a power of two in {1,2,4,8,16}"
.endif
// resident regions aligned to 256B
.set BRES_OFF,       256                       // resident B for current super-tile
.set BRES_BYTES,     (FN*16*SEGK)              // = 4*16*64 = 4096 at the default config
.set ARES_OFF,       (BRES_OFF + BRES_BYTES)   // resident A for current super-tile
.set ARES_BYTES,     (G*16*FM*SEGK)            // = 6*16*2*64 = 12288 at the default config
.set LDS_TOTAL_DSWS2, (ARES_OFF + ARES_BYTES)
// (old single-slot cap check retained; 16640 < 32768 -> always passes. The RING layout below is what
//  the ring role loops actually use; its own cap check follows.)
.if LDS_TOTAL_DSWS2 > 65536
  .error "DSWS2 single-slot operand layout exceeds the 65536B WGP limit"
.endif
// *** WAS `> 32768` -- a STALE guard from the PRE-FLOW single-slot layout, whose own comment said
//   "16640 < 32768 -> always passes". It never fired because SEGK was always small. It fires now, and it
//   was BLOCKING SEGK>=128 at legal LDS (SEGK=128/G=8/POOL_N=1 = 41,472B, well under the real 64KB cap
//   enforced by LDS_TOTAL_FLOW below). SEGK is the direct lever on SEGMENT COUNT -- the thing costing us
//   ~4,350 cycles of coordination per ~72 cycles of WMMA on real ml8 shapes. ***
// ============================================================================================
// FIX 1 (flow) -- FLOW ECONOMY LDS layout (N-deep pool + per-wave ROLE mailbox). NEW symbols; the
//   single-slot control words above stay DEFINED (no bytes; only used under .if DSWS2_CONV/DIAG/TRACE,
//   all 0). The flow role loops reference ONLY the symbols below. There is NO publish flag to poll:
//   compute streams rowblks from the READY pool (DRAIN_HEAD), feeds stage the next super-tile into the
//   next FREE slot (STAGE_HEAD); both frontiers are super-tile indices, slot = (index mod POOL_N). Each
//   wave reads its ROLE[wid] mailbox each cycle and simply IS that role (stale mailbox = last role =
//   coast). Single writer (coordinator wid0) -> no CAS. See FLOW_ECONOMY_DESIGN.md.
.ifndef POOL_N
  .set POOL_N,      3         // pool depth: 3 slots -> 48KB operands + ctrl < 64KB (4 = 64KB, too tight)
.endif
.ifndef ACC_N
  .set ACC_N,       1         // per-rowblk fp32 reduction accumulator banks (rowblk-lifetime). CO-BUDGET with
.endif                        //   POOL_N: OP_BASE + POOL_N*OPSTRIDE + ACC_N*ACC_STRIDE <= 65536. Defaults keep
                              //   the POOL_N=3 bare build legal (57600); stagger build uses POOL_N=2 ACC_N=2 (49408).
// ---- GROUP-SPLIT (B-reuse-in-L2 occupancy lever) ----------------------------------------------------
//   The tile's G rowblks are reduced through ACC_N LDS banks in GROUPS = G/ACC_N sequential passes.
//   ACC_N < G shrinks the resident bank footprint (ACC_N*8192 vs G*8192) -> more WGs/CU -> the pool can
//   bind. Each group RE-SCANS B[:,tcol] (the feed re-stages B per emitted super-tile); since the whole B
//   column stays warm in L2 (measured L2=8MB), the re-scan is an L2 hit, NOT an HBM refetch -> B-reuse is
//   retained, no fetch-lag reintroduced. group is packed into the STAMP high bits (STAMP=(group<<28)|sti);
//   sti < 2^28 for every real shape. ACC_N=G => GROUPS=1 => every `.if GROUPS>1` path drops => the whole-
//   tile write-once codegen is BYTE-IDENTICAL. POOL_N=1 keeps groups strictly sequential (group g's banks
//   store + recycle before g+1's ksi=0 re-inits them), so Fable's H1(init-race)/H5(boundary) never fire.
.if G % ACC_N
  .error "G must be divisible by ACC_N (group-split needs even rowblk groups)"
.endif
// GROUPS>1 RE-ENABLED 2026-07-17. The two historical bugs are fixed together at the coordinator cursor:
//   (a) numerical (was ok=24 bad=24 max_rel=1): NOT address overflow -- it was STALE BANK RESIDUE. The
//       GROUPS>1 code assumed the old non-BANKZERO `ksi==0 fresh-write` re-inits banks per group; under
//       BANKZERO=1 (forced by DECENTASN) every ksi is a pure ds_add_f32 and banks are zeroed only at
//       TILE claim, so group g>0 accumulated onto group g-1's un-cleared banks. FIX: re-zero the banks at
//       every GROUP boundary (drain-gated, mirrors the new-tile path) -- see .Lflow_same_tile.
//   (b) arbitrary-K: the (group,ksi) cursor was walked DENSELY over [0,GROUPS*2^shift), emitting ksi>=n_kseg
//       at non-pow2 n_kseg. FIX: COUNT-based advance (roll group at ksi==COUNT=s66) -- see the cursor advance.
//   Both fixes are `.if GROUPS>1`-gated; GROUPS=1 (ACC_N==G) codegen is byte-identical.
.set GROUPS,        (G / ACC_N)                 // rowblk-reduction passes per tile (1 = whole-tile, today)
.set STAMP_GSHIFT,  28                          // group in STAMP[31:28]; sti in STAMP[27:0]
.set STI_MASK,      ((1 << STAMP_GSHIFT) - 1)   // = 0x0FFFFFFF
.ifndef COORD_PERIOD
  .set COORD_PERIOD, 64       // coordinator sense/nudge cadence (loop cycles); lazy is fine (waves coast)
.endif
// DEADMAN: per-wave wall-clock watchdog. Every wave stamps its start RTC and, at each loop head, force-
//   retires if it has been alive > DEADMAN_TICKS. Converts a COORDINATION hang (a frontier that never
//   advances -> waves spin the loop forever) into a CLEAN drain: all waves retire, occ[0]->0, the queue
//   goes idle and the EOP fence fires -> NO wedge, NO desktop brick (the result is just incomplete, which
//   the oracle flags). This is what makes scale-stress SAFE to run. Fires at 0.5s < host chunkMaxS(0.75s)
//   so the host observes the clean drain before its own bail. It does NOT cover the s_alloc_vgpr grow-spin
//   (grow-stagger, ISA 3.3.3.2) -- that spin never reaches the loop head; it's a separate gate (M=576 only).
//   DEADMAN=0 -> zero bytes (for clean perf/byte-identity bins).
.ifndef DEADMAN
  .set DEADMAN, 1
.endif
.ifndef RETBARRIER
  .set RETBARRIER, 1             // count-to-WAVES collective exit: each wave checks in at .Lflow_dead and all
.endif                           //   s_endpgm TOGETHER once the WG's count hits WAVES -> the EOP registers a
.ifndef RETBAR_MAX               //   clean coordinated dispatch completion -> the fence FIRES (the staggered
  .set RETBAR_MAX, 1000000       //   coordinator-broadcast retire fires at 8 waves but not 16). Bounded-wait
.endif                           //   (no message bus / no s_alloc) so it can NEVER hang the wave.
.ifndef DEADMAN_TICKS
  .set DEADMAN_TICKS, 50000000   // 0.5s @ 100MHz RTC. *** THIS IS THE ANTI-BRICK GUARD, NOT A TUNING KNOB. It turns a wedged
                                 //   WG into a clean retire BEFORE MES gives up on REMOVE_QUEUE and the driver falls back to
                                 //   MODE1 reset. On 2026-07-14 I raised it to 10s to stop the deadman FALSE-KILLING healthy
                                 //   long-lived waves -- that removed the brick guard and cost 3 MODE1 RESETS. The false kills
                                 //   were never a threshold problem: the watchdog measured WAVE LIFETIME, not STALL TIME. It is
                                 //   now STALL-SCOPED (re-stamped on forward progress), so 0.5s is both safe AND correct.
                                 //   *** NEVER "fix" a false kill by raising this -- find the missing deadman_progress site. ***
.endif
// ---- shared frontiers + mailbox (single copy, at the front). 3-frontier pipeline:
//        DRAIN_HEAD <= STAGE_HEAD <= ASSIGN_HEAD <= DRAIN_HEAD + POOL_N   (all monotone u32) ----
.set ASSIGN_HEAD_OFF, 0      // next local index to assign a global super-tile (SINGLE writer = coordinator wid0)
.set STAGE_HEAD_OFF,  4      // oldest assigned-but-not-fully-staged index (feeds; ds_cmpstore advance)
.set DRAIN_HEAD_OFF,  8      // oldest not-fully-drained index (compute; ds_cmpstore advance)
.set RINGINIT_OFF,    12     // barrier-free LDS-init publish flag (coordinator writes 0xACED LAST)
.set FLOWTERM_OFF,    16     // terminal flag: coordinator sets 0xDEAD once all super-tiles claimed
.set ROLE_BASE,       20     // per-wave mailbox: ROLE[wid] at ROLE_BASE + wid*4  (coordinator-written)
// role codes stored in ROLE[wid]:
.set ROLE_COMPUTE,   0
.set ROLE_AFEED,     1
.set ROLE_BFEED,     2
.set ROLE_RETIRE,    3
// BATON bootstrap-seed target: the role init below (~2412) makes wid0/1/2 feeds and wid>=3 COMPUTE,
//   so wid 3 is the FIRST compute wave. It is seeded with the opening grow-turn (Task 1, GROWPERMIT=1).
.set FIRST_COMPUTE_WID, 3
// ---- per-slot control block:  SLOTC_BASE + slot*SLOTC_STRIDE + field  (same fields as the ring) ----
// coordinator-local TILE-CLAIM state (single writer = wid0), tucked in the reserved 32-wave mailbox
//   tail (safe while WAVES<=30). occ[20] claims whole TILES; the coordinator emits a tile's n_kseg
//   super-tiles (sti=(t<<shift)|ksi) into its OWN WG's slots so per-WG LDS banks accumulate a full tile.
.set COORD_KSI_OFF, (ROLE_BASE + 32*4 - 8)      // = 140: next ksi to emit for the current tile (init sentinel)
.set COORD_T_OFF,   (ROLE_BASE + 32*4 - 4)      // = 144: current tile index t
.set ASSIGN_LOCK_OFF, COORD_KSI_OFF             // DECENTASN: assign mutex (0=free,1=held). REUSES the ksi-cursor slot
                                                //   (140) -- under DECENTASN the old coordinator/ksi-cursor is gated
                                                //   off, so the two are mutually exclusive (no new LDS word). The lock
                                                //   holder batch-fills the pool as the SINGLE writer, then releases.
// ---- INTRA-WG DECENTRALIZED ASSIGN cursor (Fork A / COUPLED-CURSOR rewrite 2026-07-18). Repurposes the
//   coordinator tile-claim slots (140/144), UNREACHABLE under DECENTASN (wid0 does no coordinator duty).
//   ksi is DERIVED from the reservation index -- within = r - DA_BASE ; ksi = within & mask ; group = within
//   >> shift -- so pool POSITION == ksi ORDER (REQUIRED for deep-J: the carrier walks consecutive positions
//   trusting each carries the next ksi; a decoupled ksi grab would permute them and scatter the J-window ->
//   silent wrong-C). The WG owns WHOLE tiles (occ[20]++ claims a TILE) so a tile's n_kseg slices land in THIS
//   WG's LDS banks (banked-valid). DA_ZDONE gates reservations so a group's banks are drain-gated + zeroed
//   before its ksi are handed out -> GROUPS>1 for free. NO over-reservation (reserve gated r < DA_ZDONE <=
//   DA_BASE+TOTAL) -> boundaries hit exactly at ASSIGN==DA_ZDONE. TOTAL = GROUPS*n_kseg = GROUPS<<shift. ----
.set DA_BASE_OFF,  COORD_KSI_OFF    // =140: reservation level at which the current tile's within=0 begins
.set DA_TILE_OFF,  COORD_T_OFF      // =144: the WG's current tile index t (from occ[20]++)
.set ZLOCK,        1                // DA_ZDONE bit 0: a wave is handling a group/tile boundary -> others bail.
//   Bit 0 (not the top bit) because DA_ZDONE is ALWAYS a multiple of n_kseg (= base + q*n_kseg, base = k*TOTAL,
//   both multiples of n_kseg = 2^shift >= 2), so its low bit is structurally 0 and can NEVER alias a real level
//   -- unlike a top-bit lock, which would alias at level 2^31 (Codex A1). Requires n_kseg >= 2 (deep-J needs
//   J|n_kseg, J>=2; the peek fail-safes n_kseg==1 to terminal alongside the non-pow2 check).
//   (DA_ZDONE_OFF -- level up to which banks are zeroed -- is defined below, after OP_BASE, in the control gap.)
//   (safe while WAVES<=30: the real mailbox uses only ROLE_BASE..ROLE_BASE+WAVES*4 -- these live in the tail.)
.set SLOTC_BASE,    (ROLE_BASE + 32*4)          // after a 32-wave-max mailbox (WAVES<=32); = 148
.set SLOTC_STRIDE,  32
.set SL_STI,        0        // super-tile id resident in this slot (STAMP; 0xFFFFFFFF = sentinel)
.set SL_GEN,        4        // (unused in flow; kept so BSTAGE_R/ASTAGE_R field offsets are unchanged)
.set SL_RBNEXT,     8        // rowblk claim counter (compute pulls; barrier-free)
.set SL_RBDONE,     12       // rowblks computed+flushed; slot FREE when >= G
.set SL_BFNEXT,     16       // B-frag claim counter (B-feeds)
.set SL_BFDONE,     20       // B-frags stored; slot B-ready when == FN
.set SL_ARNEXT,     24       // A-rowblk claim counter (A-feeds)
.set SL_ARDONE,     28       // A-rowblks staged; slot A-ready when == G  (slot READY = B-ready && A-ready)
// ---- DECENTASN POISON-UNTIL-STAGED encoding of SL_RBNEXT (Codex gpt-5.6-sol design, 2026-07-15) ----
//   SL_RBNEXT holds EITHER pending bits (unstaged) OR a low rowblk counter (staged+claimable). Producer
//   stamps RB_PENDING; each staging side clears its bit via side_final; the side that finishes SECOND
//   CAS-arms it to 0. Compute CLAIMS via CAS(x -> x+1) only when x has no pending bit and x < ACC_N; a
//   pending/exhausted/lost slot -> COAST (help), never wait. Requires ACC_N < B_PENDING (trivially true).
.set A_PENDING,   0x80000000  // A side not yet fully staged
.set B_PENDING,   0x40000000  // B side not yet fully staged
.set RB_PENDING,  0xC0000000  // A_PENDING | B_PENDING  (freshly-stamped, unstaged, un-claimable)
// ---- (next,inflight) SINGLE-WORD PIN -- RETIRED 2026-07-16 (banked DECENTASN, DECENTASN_BANKED_DEEPJ_DESIGN) ----
//   The pin packed inflight_claims into SL_RBNEXT[15:8] to make drain-authority atomic with the claim under
//   WOFLUSH. On silicon it over-released (a stray -INFLIGHT_ONE underflowed the field -> 0x..06 borrow ->
//   manufactured RB_PENDING -> head-of-line drain stall; measured occ[97]~800). On the BANKED path the race it
//   guarded does not arise: SL_RBDONE is bumped only AFTER a claim's ds_add_f32 drains, so RBDONE==ACC_N is a
//   sufficient drain gate (O1: every won CAS -> exactly one RBDONE++). The claim CAS is now x -> x+1 (next only);
//   DRAIN gates on SL_GEN==DRAIN && SL_RBDONE==ACC_N. INFLIGHT_MASK/INFLIGHT_ONE are now UNUSED (kept for the
//   sentinel's bit-pattern documentation); NEXT_MASK is still used to read next out of SL_RBNEXT.
.set NEXT_MASK,     0x000000ff  // bits[7:0]  = next_rowblk (still used)
.set INFLIGHT_MASK, 0x0000ff00  // bits[15:8] = inflight_claims -- UNUSED since 2026-07-16 (pin retired)
.set INFLIGHT_ONE,  0x00000100  // one inflight unit -- UNUSED since 2026-07-16 (pin retired)
.if DECENTASN && (ACC_N >= 0x100)
  .error "DECENTASN (next,inflight) pin needs ACC_N < 256 (next_rowblk is 8 bits)."
.endif
.if DECENTASN && (WAVES >= 0x100)
  .error "DECENTASN (next,inflight) pin needs WAVES < 256 (inflight_claims is 8 bits)."
.endif
// ---- per-slot operand buffers:  OP_BASE + slot*OPSTRIDE ;  BRES at +BRES_ROFF, ARES at +ARES_ROFF ----
.set OP_BASE,       512                          // *** CO-CHANGE: MUST MATCH kOpBase in occ_dispatch.cpp. ***
                                                 //   RAISED 256->512 (2026-07-13). At 256 the per-slot control
                                                 //   blocks capped POOL_N at 3: SLOTC_BASE(148) + N*32 <= OP_BASE
                                                 //   => N <= 3. That 256-byte base address was throttling the
                                                 //   ENTIRE flow economy -- measured: ASSIGN can never lead DRAIN
                                                 //   by more than POOL_N (see the frontier invariant below), so a
                                                 //   3-deep pool left 16 waves fighting over 3 in-flight super-tiles
                                                 //   and 76.4% of all idle iterations died on an empty ASSIGN
                                                 //   frontier (occ[86] feedMT = 5.70M of 7.46M coasts).
                                                 //   The host under-allocates LDS if it disagrees -> the workgroup
                                                 //   SILENTLY NEVER LAUNCHES (every counter reads 0; looks like a
                                                 //   hang, is actually a dispatch that could not fit). The .error
                                                 //   below makes the kernel side self-checking; the host side is
                                                 //   guarded by kOpBase + its own static_assert.
.if (SLOTC_BASE + POOL_N*SLOTC_STRIDE) > OP_BASE   // unconditional (was only checked on the banked path, which
  .error "slot control blocks overrun OP_BASE -- raise OP_BASE (and kOpBase in occ_dispatch.cpp)"
.endif                                             //   meant WOFLUSH builds could silently overlap them)
// ---- per-TILE (per-group) completion counter: the C-store must be gated on the TILE being fully
//   reduced, NOT on one SLOT finishing. With a deep pool the slot holding ksi==mask can finish BEFORE
//   the slots holding earlier ksi have accumulated -> the completer would store half-summed banks.
//   (measured 2026-07-13: POOL_N=2/3 -> bad=96/116.) Every rowblk-segment bumps TILEDONE[group];
//   whoever brings it to ACC_N*n_kseg owns the C-store. Decouples the completer from POOL_N entirely.
.set TILEDONE_BASE, ((SLOTC_BASE + POOL_N*SLOTC_STRIDE + 15) & ~15)   // GROUPS u32, 16B-aligned, above the slots
.if (BANKZERO && !WOFLUSH) && ((TILEDONE_BASE + GROUPS*4) > OP_BASE)
  .error "TILEDONE overruns OP_BASE -- raise OP_BASE"
.endif
// ============================================================================================
// THE STAGGER / TRAVELING PEAK (2026-07-13 night). Admission control on the FAT population.
//
//   WHY IT FINALLY HAS A JOB: `grow-fail` was pinned to 0 for the whole project, and I twice
//   concluded from that that the stagger was dead. WRONG BOTH TIMES. grow-fail was 0 because
//   ACC_N == G == 4 capped the number of waves that could EVER be fat at 4 -> 4*112 = 448 VGPRs
//   of a ~1536/SIMD budget -> the budget was STRUCTURALLY UNABLE to bind. Unlocking G (the LDS
//   banks were what capped it; K-DEPTH J made them unnecessary) put 15 carriers/WG in flight and
//   the budget BINDS: measured grow-fail = 162,783 at J=64.
//
//   WHY A FAILED GROW IS EXPENSIVE: `s_alloc_vgpr` does WaitIdleExceptStoreCnt() -- it DRAINS THE
//   WAVE'S ENTIRE PIPELINE before it can even refuse. We are paying a full drain 162k times just
//   to be told "no". Admission control replaces that with ONE LDS atomic.
//
//   THE MECHANISM: a per-WG token counter caps the number of SIMULTANEOUSLY FAT waves at MAXFAT.
//   A wave acquires a token BEFORE attempting the grow; if the peak is full it coasts (= goes and
//   FEEDS) instead. So the instantaneous VGPR peak is bounded by construction, the waves take
//   TURNS being fat, and the AVERAGE footprint -- not the peak -- is what has to fit the budget.
//   That is the traveling peak: fatness becomes TEMPORAL rather than SIMULTANEOUS.
//
//   Pairs with a MODERATE J. At J = n_kseg the wave is fat for the whole K -- a PLATEAU, not a
//   peak -- and there is nothing to interleave. Short bursts are what make the stagger possible.
.set FATTOK_OFF,    ((TILEDONE_BASE + GROUPS*4 + 15) & ~15)   // per-WG count of currently-FAT waves

// *** LDS ALIAS FIX (2026-07-14). These NINE control words sat at bytes 72..107 -- INSIDE the 32-entry
//   per-wave ROLE mailbox [ROLE_BASE=20, 148). At WAVES=30 they aliased ROLE[13]..ROLE[21], which are
//   REAL wave mailboxes. Live consequences, all traced:
//     - the coordinator's terminal broadcast writes ROLE_RETIRE=3 into EVERY mailbox, so QUIESCE_CNT
//       (== ROLE[19]) became 3 -> the count-to-WAVES retire barrier released ~3 waves EARLY;
//     - every deadman force-retire does lds_inc on QUIESCE_CNT -> wave 19 reads role 1, then 2, then
//       3 (RETIRE) and retires SPURIOUSLY. If wave 19 was a FAT J-carrier it exits with ACC UNFLUSHED
//       -> SILENT WRONG C, and NO counter fires;
//     - the forensics stream's `barrier=` field was reading wave 19's role word, not the barrier.
//   The existing .error only guarded WAVES>30 (COORD_KSI/T at ROLE[30]/[31]) and missed this entirely.
//   Found by the Fable audit pass. ***
.set CTRL_BASE,      ((FATTOK_OFF + 4 + 15) & ~15)   // above FATTOK, below OP_BASE -- OUT of the mailbox
.set SNAP_BASE,      CTRL_BASE                        // u32[6] role-mix snapshots
.set QUIESCE_CNT_OFF,(SNAP_BASE + 6*4)                // u32 role-agnostic bail counter
.set OCCA_PUB_OFF,   (QUIESCE_CNT_OFF + 4)            // claimer-published occ_A peak
.set OCCB_PUB_OFF,   (OCCA_PUB_OFF + 4)               // claimer-published occ_B peak
.if (OCCB_PUB_OFF + 4) > OP_BASE
  .error "control words overrun OP_BASE -- raise OP_BASE (and kOpBase in occ_dispatch.cpp)"
.endif
// GUARD: no control word may EVER land inside the per-wave ROLE mailbox again.
.if (SNAP_BASE < (ROLE_BASE + 32*4)) && ((OCCB_PUB_OFF + 4) > ROLE_BASE)
  .error "an LDS control word ALIASES the ROLE mailbox [ROLE_BASE, ROLE_BASE+128) -- at WAVES=30 this silently corrupts wave 13..21's role and can retire a FAT carrier with unflushed ACC"
.endif
.if STAGGER && ((FATTOK_OFF + 4) > OP_BASE)
  .error "FATTOK collides with OP_BASE -- raise OP_BASE (and kOpBase in occ_dispatch.cpp)"
.endif
// ---- TRAVELING-PEAK BATON grow-permit mailbox (2026-07-16). Mirrors the ROLE mailbox EXACTLY. ----
//   GROWPERMIT[wid] (u32): 1 = "you hold the grow-turn, grow now" ; 0 = "no turn, keep feeding".
//   A shrinking compute wave PUSHES 1 into the next-available compute wave's slot (Task 2); each compute
//   wave reads only its OWN slot each pass (non-blocking, like ROLE) and grows on a 1 (Task 3). No wave
//   ever polls another wave's slot -- that is the river principle (FLOW_ECONOMY_DESIGN.md). 32-wave
//   reservation (WAVES-independent offsets, exactly like ROLE) placed in the control gap below OP_BASE.
.set GROWPERMIT_BASE, ((OCCB_PUB_OFF + 4 + 15) & ~15)   // = 336 @ bring-up geometry; [BASE, BASE+128)
.set BATON_NEXT_OFF,  (GROWPERMIT_BASE + 32*4)          // per-WG round-robin cursor for the next grow-turn (Task 2)
.set NCOMPUTE,        (WAVES - FIRST_COMPUTE_WID)       // # compute waves = the baton round-robin span
.if NCOMPUTE < 1
  .set BATON_MAGIC, 0                                   // invalid geometry; the STAGGER guard below errors out
.else
  .set BATON_MAGIC, (0x100000000 / NCOMPUTE)           // floor(2^32/NCOMPUTE): unsigned-div magic for idx mod NCOMPUTE.
.endif                                                  //   PROVEN (full-u32 sweep): q=mulhi(idx,MAGIC) never overshoots
                                                        //   and rem < 2*NCOMPUTE, so ONE conditional subtract normalizes.
.if STAGGER && ((BATON_NEXT_OFF + 4) > OP_BASE)
  .error "GROWPERMIT/BATON_NEXT overrun OP_BASE -- raise OP_BASE (and kOpBase in occ_dispatch.cpp), or lower POOL_N"
.endif
.if STAGGER && (WAVES <= FIRST_COMPUTE_WID)
  .error "BATON needs WAVES > FIRST_COMPUTE_WID (3): there must be at least one compute wave to seed the opening grow-turn"
.endif
// ---- COUPLED-CURSOR (DECENTASN) group-zero gate. DA_ZDONE = reservation level up to which the WG's banks are
//   zeroed (top bit ZLOCK = a wave is mid boundary-handle). Lives in the control gap just below OP_BASE; a plain
//   .set costs zero bytes and is never referenced under DECENTASN=0 (byte-identical). ----
.set DA_ZDONE_OFF,  (OP_BASE - 4)                // = 508 @ OP_BASE=512, in the [BATON_NEXT..OP_BASE) control gap
.set GSTORED_OFF,   (OP_BASE - 8)                // = 504: per-WG count of GROUP C-stores whose s_wait_storecnt completed.
                                                 //   The boundary handler waits GSTORED >= (z>>shift) before zero_banks so a
                                                 //   finishing group's C-store (which READS the banks) can't race the reuse-
                                                 //   zeroing (Codex C1: DRAIN==ASSIGN alone does NOT exclude the post-RBDONE,
                                                 //   pre-C-store interval).
.if DECENTASN && ((BATON_NEXT_OFF + 4) > GSTORED_OFF)
  .error "DA_ZDONE/GSTORED collide with the BATON grow-turn mailbox -- raise OP_BASE (and kOpBase) or lower POOL_N"
.endif
.if DECENTASN && (DSWS2_STATE_END > GSTORED_OFF)
  .error "DA_ZDONE/GSTORED collide with DSWS2 Phase-B state (SNAP/QUIESCE/OCC_PUB) -- raise OP_BASE (and kOpBase)"
.endif
.if DECENTASN && (JDEPTH > 1) && ((POOL_N / JDEPTH) * JDEPTH != POOL_N)
  .error "DECENTASN deep-J requires POOL_N % JDEPTH == 0: else a physical slot maps to generations of DIFFERENT ksi%J (lead-ness), so a lead-gate that passed pre-CAS can be fooled by an ABA recycle to a non-lead generation (Codex D1 variant). Set POOL_N to a multiple of JDEPTH (JDEPTH<=POOL_N already required)."
.endif
.set OPSTRIDE,      (BRES_BYTES + ARES_BYTES)    // 4096 + 12288 = 16384 per slot
.set BRES_ROFF,     0                            // resident B within a slot
.set ARES_ROFF,     BRES_BYTES                   // resident A within a slot (after B)
// ---- per-rowblk reduction accumulator pool: ACC_BASE + bank*ACC_STRIDE (bank in [0,ACC_N)) ----
//   fp32, rowblk-lifetime (persists across all n_kseg K-segments of a rowblk); DISTINCT from the
//   segment-lifetime operand pool above. One bank = one C-rowblk = FM*FN frags x 1024B.
.set ACC_BASE,      (OP_BASE + POOL_N*OPSTRIDE)  // after the operand pool
.set ACC_STRIDE,    (FM*FN*1024)                 // = 8192 @ FM=2 FN=4 (one C-rowblk)
.if WOFLUSH
.set LDS_TOTAL_FLOW,(ACC_BASE)                     // WOFLUSH: NO LDS accumulator banks -- each burst atomic-adds
                                                   //   ACC straight to C. Reserving ACC_N*8KB here ANYWAY was
                                                   //   silently capping POOL_N -- the whole flow economy -- on
                                                   //   memory the kernel never touches. (found 2026-07-13)
.else
.set LDS_TOTAL_FLOW,(ACC_BASE + ACC_N*ACC_STRIDE)  // POOL3/ACC1: 57600 ; POOL2/ACC2: 49408
.endif
.if LDS_TOTAL_FLOW > 65536
  .error "FLOW LDS layout exceeds 65536B group segment (hardware WGP limit) -- lower POOL_N or ACC_N"
.endif
.if (SLOTC_BASE + POOL_N*SLOTC_STRIDE) > OP_BASE
  .error "FLOW per-slot control blocks overlap the operand region (raise OP_BASE)"
.endif
// Phase-B state must fit in the control gap below the OPERAND POOL.
// *** WAS `> BRES_OFF` (=256) -- the PRE-FLOW single-slot layout. The FLOW path puts operands at
//   OP_BASE(512) + slot*OPSTRIDE and never uses BRES_OFF (BSTAGE/ASTAGE non-ring are dead here; flow
//   calls BSTAGE_R/ASTAGE_R). That stale bound was blocking the ROLE-mailbox alias fix from placing the
//   control words anywhere sane. Same species as the stale `> 32768` LDS guard. ***
.if DSWS2_STATE_END > OP_BASE
  .error "DSWS2 Phase-B state (SNAP_BASE/QUIESCE_CNT) overruns OP_BASE -- raise OP_BASE (and kOpBase in occ_dispatch.cpp)"
.endif
// and it must not collide with the per-slot control blocks / TILEDONE / FATTOK below it
.if SNAP_BASE < (SLOTC_BASE + POOL_N*SLOTC_STRIDE)
  .error "control words collide with the per-slot control blocks"
.endif

.if DSWS2
  // ---- launch wave count (EMERGENT economy: NO baked compute/feed mix; roles emerge at runtime) ----
  .ifndef WAVES
    .set WAVES, 16                           // waves/WG launched; host launches the SAME count.
  .endif
  .if WAVES > 30
    .error "WAVES>30 collides with COORD_KSI/T at ROLE[30]/ROLE[31] -- relocate coord state first"
  .endif
.endif

// ============================================================================================
// VGPR layout (lifted from occ_kernel_coop.s) -- compute frags live ABOVE the lean-32 block and are
//   only touched AFTER s_alloc_vgpr NFV. Feeds/claimer stay in the lean block (v0..v31).
// ============================================================================================
.set ACC, 32                                 // accumulators: FM*FN frags x 8 f32 (v32..)
.set FA,  (ACC + 8*FM*FN)                     // compute A frags (from resident LDS): FM x 2
.set FB,  (FA + 2*FM)                         // compute B frags (from resident LDS): FN x 2
.set NFV, ((FB + 2*FN + 15) & ~15)            // grown footprint, rounded to a 16-VGPR dyn block (=112 @ 2x4)
.set VLEAN, 32                                // lean footprint (feeds, claimer, compute pre/post rowblk)
.set BSTG, 16                                 // staging regs (lean block, < 32): B-feed FN-frag / A-feed FM-frag
;; ===========================================================================================
;;  NOFEED -- PERF PROBE ONLY (oracle WILL fail; that is what proves it took).
;;  Skip ONLY the global_load in the staging path. KEEP the ds_store, the waits, the rowblk
;;  claim/publish and the whole frontier. So it isolates exactly ONE thing: the cost of reading
;;  A and B FROM MEMORY. Everything else in the economy still runs.
;;    NOFEED ~= FED  -> we are NOT feed-bound; the wall is coordination/LDS, and the 290x re-read
;;                      of A/B is essentially free (it is hitting cache).
;;    NOFEED >> FED  -> feed IS the game, and the lever is CACHE BLOCKING / tile scheduling
;;                      (make the re-read hit L2), not more loads in flight.
;;  (RBU 1->2 doubled loads-in-flight for +2.1%, so load batching is saturated. This is the
;;   measurement that says what to do next instead of guessing.)
;; ===========================================================================================
;; ===========================================================================================
;;  MULTISLOT -- THE CURSOR FIX (2026-07-14, kmbandy: "why are we EVER limiting the number of
;;  waves that can compute? that's literally going against the entire point of ADAPTIVE dsws")
;;
;;  .Lflow_compute read DRAIN_HEAD and worked ONLY that slot (`slot_of(dh)`). A slot has ACC_N
;;  (== G) rowblks, handed out by SL_RBNEXT. So:
;;        MAX CONCURRENT COMPUTE WAVES = G.  Regardless of WAVES. Regardless of POOL_N.
;;  On the synthetic cube G=15/W=30 -> invisible. On REAL shapes M % (G*FM*16) forces G<=9
;;  (M=576 -> G in {2,3,6,9}), so 21-27 of 30 waves per WG were STRUCTURALLY FORBIDDEN from ever
;;  computing -- ~1,700 waves across the GPU doing nothing but spin-polling the LDS frontier.
;;  That is ALSO why POOL_N measured DEAD FLAT: a deeper pool stages more slots, but compute could
;;  only ever touch ONE of them. The pool was structurally unable to help.
;;
;;  MULTISLOT=1: a compute wave scans the staged window [DRAIN, STAGE) for a claimable slot,
;;  starting at (wid mod window) so waves SPREAD instead of piling on the head.
;;        concurrent compute = G * POOL_N.
;;
;;  *** CO-REQUIREMENT (this would be a SILENT WRONG-C bug without it): the old DRAIN advance was
;;  an UNCONDITIONAL `DRAIN++` on any slot completion -- correct ONLY because compute was pinned to
;;  the head, so slots necessarily completed IN ORDER. Once slots can complete OUT OF ORDER, a
;;  completer of slot dh+1 would free slot dh while it is still in use -> its operands get
;;  overwritten by the next stage. drain_advance below now WALKS: it advances only while the head
;;  slot's SL_RBDONE == ACC_N, and keeps walking (several may now be done). ***
;;
;;  NOTE it only pays at JDEPTH=1: at J>1 the coordinator POISONS every non-lead slot
;;  (SL_RBNEXT = ACC_N) because those rowblks are owned by J-carriers. That is fine -- on real
;;  shapes NOCFLUSH measured FLAT, so the flush costs nothing and J exists only to amortize it.
;; ===========================================================================================
.ifndef MULTISLOT
    .set MULTISLOT, 0                            // umbrella: sets both halves
.endif
.ifndef MSCOMP
    .set MSCOMP, MULTISLOT                       // umbrella for the compute side
.endif
.ifndef MSSCAN
    .set MSSCAN, MSCOMP                          // compute-side: scan the window instead of pinning to head
.endif
.ifndef MSDRAIN
    .set MSDRAIN, MSCOMP                         // DRAIN advances via the head-gated WALK (not unconditional bump)
.endif
.ifndef MSFEED
    .set MSFEED, MULTISLOT                       // feed/coast window scan + STAGE walk
.endif
.ifndef BATCHASN
    .set BATCHASN, 0                             // coordinator batch-assign: fill the pool per visit instead of 1/visit
.endif
.ifndef DECENTASN
    .set DECENTASN, 0                            // decentralized assign: assign is a ROLE any starved wave does (kills the single-producer wall)
.endif
.if DECENTASN && WOFLUSH
  .error "DECENTASN is now BANKED-ONLY (build WOFLUSH=0 BANKZERO=1). The WOFLUSH (next,inflight) pin path was retired 2026-07-16 -- guard 697's 'SL_GEN aliases the store-claim' collision was STALE (the banked completer elects the store owner via TILEDONE, not SL_GEN; every SL_GEN reader is DECENTASN-gated). See DECENTASN_BANKED_DEEPJ_DESIGN_2026-07-16.md."
.endif
.if DECENTASN && !BANKZERO
  .error "DECENTASN banked path needs BANKZERO=1 (pre-zeroed LDS accumulator banks -> every ksi is a pure ds_add_f32)."
.endif
.if DECENTASN && MSSCAN
  .error "DECENTASN compute is head-pinned: the straddle observer (occ[95]) and the post-grow slot re-derivation both compare against DRAIN (s46). MSSCAN reassigns s46 to a spread cursor, which would break both. Build MSSCAN=0."
.endif
.if DECENTASN && RBU > 1
  .error "DECENTASN poison-until-staged elects the A-side finalizer by old_SL_ARDONE==G-1; RBU>1 increments SL_ARDONE by RBU and can SKIP G-1 -> the A finalizer is never elected -> the slot never arms -> wedge. Build RBU=1 (or generalize the A-side election to 'the increment that crosses G')."
.endif
// DECENTASN && JDEPTH>1 (deep-J + decentralized assign): RESOLVED 2026-07-18. The collision (non-lead poison
//   SL_RBNEXT=ACC_N vs poison-until-staged) is gone: the COUPLED CURSOR (ksi = r - DA_BASE) makes pool position ==
//   ksi order (so the carrier's J-window is aligned -- the correctness precondition deep-J needs), and non-lead
//   slots are turned away from the CLAIM by the pre-grow lead-gate + a post-grow lead RE-CHECK instead of an
//   ACC_N poison, so EVERY slot stamps RB_PENDING and arms to 0 normally. side_final is unchanged.
.ifndef NOFEED
    .set NOFEED, 0
.endif
.ifndef RBU
    .set RBU, 1                              // rowblks staged per claim. RBU=2 => 2x the loads in flight.
.endif
.if RBU > 2
  .error "RBU must be 1 or 2 (the odd-G tail path only handles a shortfall of 1)"
.endif
.set ASTG, BSTG                              // A-feed staging regs: A now needs FM*KSEG_STEPS*2 (one pair per
                                             //   (mi,ks)) so ALL its loads can be in flight at once. B and A
                                             //   stage in SEPARATE calls, so they may share the same window.
// (no VGPR guard needed any more: astage_frags CHUNKS the loads into groups of MAXIF=(VLEAN-ASTG)/2,
//  so ANY SEGK works with a constant 8 loads in flight. The guard that used to live here capped SEGK<=64
//  -- and SEGK is the lever on segment count, which is the dominant cost on real ml8 shapes.)
.if (VLEAN - ASTG) < 4
  .error "the lean staging window v[ASTG..VLEAN-1] must hold at least 2 loads"
.endif

// ---- dyn-VGPR PRE-GROW temp-reg ceiling (coop death-cert: a >v15 src pre-grow is poison under dyn).
//   Gate every PRE-grow-reachable LDS/atomic temp to v11/v14 (INTERIOR to the launched 16-VGPR block). ----
.if DYNVGPR
  .set RG_A, 11                              // lds_get / fetch_add address
  .set RG_D, 14                              // lds_get / fetch_add data+return
  .set RP_A, 11                              // lds_put address
  .set RP_D, 14                              // lds_put data
.else
  .set RG_A, 27
  .set RG_D, 28
  .set RP_A, 28
  .set RP_D, 29
.endif

// ============================================================================================
// LDS helper macros (s49 = exec save; v2 = lane = tid&31, set in prologue).
// ============================================================================================
.macro lds_get sdst, off                     // wave-uniform read LDS[off] -> scalar sdst
    v_mov_b32 v[RG_A], \off
    ds_load_b32 v[RG_D], v[RG_A]
    s_wait_dscnt 0x0
    v_readfirstlane_b32 \sdst, v[RG_D]
.endm
.macro lds_get_r sdst, saddr                  // wave-uniform read LDS[saddr] (RUNTIME addr in a sreg) -> sdst
    v_mov_b32 v[RG_A], \saddr
    ds_load_b32 v[RG_D], v[RG_A]
    s_wait_dscnt 0x0
    v_readfirstlane_b32 \sdst, v[RG_D]
.endm
.macro lds_put off, ssrc                      // lane-0-of-wave writes scalar ssrc -> LDS[off]
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lput_skip\@
    v_mov_b32 v[RP_A], \off
    v_mov_b32 v[RP_D], \ssrc
    ds_store_b32 v[RP_A], v[RP_D]
    s_wait_dscnt 0x0
.Lput_skip\@:
    s_mov_b32 exec_lo, s49
.endm
.macro lds_fetch_add sdst, off, val          // sdst <- old LDS[off]; LDS[off]+=val (lane-0 atomic, broadcast)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfa_skip\@
    v_mov_b32 v[RP_A], \off
    v_mov_b32 v[RP_D], \val
    ds_add_rtn_u32 v[RP_D], v[RP_A], v[RP_D]   // v[RP_D] <- old; LDS[off] += val
    s_wait_dscnt 0x0
.Lfa_skip\@:
    s_mov_b32 exec_lo, s49
    v_readfirstlane_b32 \sdst, v[RP_D]         // broadcast lane-0's old value
.endm
.macro lds_inc off                            // lane-0-of-wave LDS[off] += 1 (no return)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Linc_skip\@
    v_mov_b32 v[RP_A], \off
    v_mov_b32 v[RP_D], 1
    ds_add_u32 v[RP_A], v[RP_D]
    s_wait_dscnt 0x0
.Linc_skip\@:
    s_mov_b32 exec_lo, s49
.endm
// ---- FIX 1a ring: RUNTIME-address variants (slot-indexed counters live at SLOTC_BASE+slot*32+field,
//   a runtime scalar). Mirror lds_fetch_add / lds_inc but take the address in a sreg. ----
.macro lds_fetch_add_r sdst, saddr, val      // sdst <- old LDS[saddr]; LDS[saddr]+=val (lane-0 atomic, bcast)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfar_skip\@
    v_mov_b32 v[RP_A], \saddr
    v_mov_b32 v[RP_D], \val
    ds_add_rtn_u32 v[RP_D], v[RP_A], v[RP_D]
    s_wait_dscnt 0x0
.Lfar_skip\@:
    s_mov_b32 exec_lo, s49
    v_readfirstlane_b32 \sdst, v[RP_D]
.endm
.macro lds_inc_r saddr                        // lane-0-of-wave LDS[saddr] += 1 (RUNTIME addr, no return)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lincr_skip\@
    v_mov_b32 v[RP_A], \saddr
    v_mov_b32 v[RP_D], 1
    ds_add_u32 v[RP_A], v[RP_D]
    s_wait_dscnt 0x0
.Lincr_skip\@:
    s_mov_b32 exec_lo, s49
.endm
// ---- FIX 1 flow: slot-of-head (index mod POOL_N) + monotone CAS frontier advance ----
.macro slot_of dst, head, scr                 // \dst = \head mod POOL_N  (\scr = scratch, only used for N=3)
.if POOL_N == 1
    s_mov_b32 \dst, 0                          // single tile in flight (stagger model: one tile's g banks fill LDS)
.elseif POOL_N == 3
    s_mul_hi_u32 \dst, \head, 0xAAAAAAAB       // q ~ head/3  (magic-div; q = mulhi>>1)
    s_lshr_b32 \dst, \dst, 1
    s_mul_i32 \scr, \dst, 3
    s_sub_u32 \dst, \head, \scr                 // slot = head - 3*q
.elseif (POOL_N & (POOL_N - 1)) == 0           // any power of 2 (2,4,8,16,...): slot = head & (POOL_N-1).
    s_and_b32 \dst, \head, (POOL_N - 1)         //   (byte-identical to the old explicit N==2 -> &1, N==4 -> &3)
.else
    .error "slot_of: POOL_N must be 1, 3, or a power of 2 (deep pools need pow2 for the mask)"
.endif
.endm

// acc_base_of: \dst = LDS byte address of accumulator bank \bank (bank in [0,ACC_N)) = ACC_BASE + bank*ACC_STRIDE.
//   ACC_STRIDE is a compile-time constant, so s_mul_i32 is exact for any FM*FN (no pow2 assumption).
.macro zero_banks                            // BANKZERO: wipe the ACC_N banks so every ksi can be a pure ds_add_f32.
.if BANKZERO                                 //   Runs on the COORDINATOR only, at TILE CLAIM, with the pool DRAINED.
    v_mov_b32 v16, 0
    v_mov_b32 v17, 0
    v_mov_b32 v18, 0
    v_mov_b32 v19, 0
    v_lshlrev_b32 v12, 4, v2                 // lane*16  (32 lanes x 16B = 512B per iteration)
    v_add_nc_u32 v12, ACC_BASE, v12          // v12 = ACC_BASE + lane*16
    s_mov_b32 s45, (ACC_N*ACC_STRIDE/512)    // iterations
.Lzb\@:
    ds_store_b128 v12, v[16:19]
    v_add_nc_u32 v12, 512, v12
    s_sub_u32 s45, s45, 1
    s_cmp_gt_u32 s45, 0
    s_cbranch_scc1 .Lzb\@
    s_mov_b32 s45, 0                         // and reset the per-group TILE completion counters
    .set gi, 0
    .rept GROUPS
      lds_put (TILEDONE_BASE + gi*4), s45
      .set gi, gi+1
    .endr
    s_wait_dscnt 0x0                         // banks + TILEDONE visible before ASSIGN_HEAD++ publishes the tile
.endif
.endm
.macro acc_base_of dst, bank
    s_mul_i32 \dst, \bank, ACC_STRIDE
    s_add_u32 \dst, \dst, ACC_BASE
.endm
.macro drain_advance                          // advance DRAIN as far as the head slot allows
.if MSDRAIN || DECENTASN
    // Slots can complete OUT OF ORDER now, so DRAIN may only advance while the CURRENT HEAD slot is
    //   genuinely complete (SL_RBDONE == ACC_N) -- and then it may be able to advance SEVERAL steps.
    //   The old unconditional bump would free a slot that is still in use. SILENT WRONG C.
    //   DECENTASN also requires SL_GEN==DRAIN (a reserved-but-unstamped slot holds the prior occupant's
    //   maxed RBDONE) AND this walk is what drains the pre-completed terminal sentinel (RBDONE==ACC_N,
    //   no computer) -- so it is also invoked from the terminal drain-watch.
.Ldadv\@:
    lds_get s20, DRAIN_HEAD_OFF
    lds_get s21, STAGE_HEAD_OFF
    s_cmp_ge_u32 s20, s21
    s_cbranch_scc1 .Ldadv_end\@                 // nothing staged -> nothing to drain
    slot_of s22, s20, s23
    s_lshl_b32 s22, s22, 5
    s_add_u32 s22, s22, SLOTC_BASE              // s22 = scb
.if DECENTASN
    s_add_u32 s23, s22, SL_GEN
    lds_get_r s23, s23
    s_cmp_lg_u32 s23, s20
    s_cbranch_scc1 .Ldadv_end\@                 // head NOT stamped for DRAIN -> stop (sentinel passes: its SL_GEN==DRAIN)
    // BANKED DECENTASN (pin retired 2026-07-16): DRAIN authority is SL_RBDONE==ACC_N, identical to the baseline
    //   banked path below. RBDONE is bumped AFTER each segment's ds_add_f32 drains (s_wait_dscnt in the banked
    //   flush), so RBDONE==ACC_N proves every claim's bank write is visible and no wave is still using the slot
    //   -- exactly the job the inflight pin used to do (O1: every won CAS -> exactly one RBDONE++). The
    //   SL_GEN==DRAIN gate above still blocks advancing past a reserved-but-unstamped slot (whose RBDONE holds
    //   the PRIOR occupant's maxed value). Together == the sentinel's expected gate (see .Lflow_da_termslot:
    //   RBDONE=ACC_N + SL_GEN=r).
    s_add_u32 s23, s22, SL_RBDONE
    lds_get_r s23, s23
    s_cmp_lt_u32 s23, ACC_N
    s_cbranch_scc1 .Ldadv_end\@                 // head slot NOT complete (RBDONE < ACC_N) -> must not advance
.else
    s_add_u32 s23, s22, SL_RBDONE
    lds_get_r s23, s23
    s_cmp_lt_u32 s23, ACC_N
    s_cbranch_scc1 .Ldadv_end\@                 // head slot NOT complete -> must not advance
.endif
    lds_cmpstore_adv DRAIN_HEAD_OFF, s20        // CAS dh -> dh+1 (idempotent; losers no-op)
    s_branch .Ldadv\@                           // the next one may be complete too
.Ldadv_end\@:
.else
    lds_get s44, DRAIN_HEAD_OFF
    lds_cmpstore_adv DRAIN_HEAD_OFF, s44        // in-order by construction (compute pinned to the head)
.endif
.endm
.macro lds_cmpstore_adv off, sexp             // lane0 monotone bump: if LDS[off]==\sexp -> LDS[off]=\sexp+1
    s_mov_b32 s49, exec_lo
    s_add_u32 s60, \sexp, 1                     // scalar ALU ignores exec -> safe under the mask
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcasadv_skip\@
    v_mov_b32 v11, \off                         // vaddr
    v_mov_b32 v14, s60                          // vNEW = exp+1
    v_mov_b32 v13, \sexp                        // vCMP -> vdst old (return unused)
    ds_cmpstore_rtn_b32 v13, v11, v14, v13      // MEM=(MEM==exp)?exp+1:MEM  (idempotent; losers no-op)
    s_wait_dscnt 0x0
.Lcasadv_skip\@:
    s_mov_b32 exec_lo, s49
.endm
.macro lds_cas_rtn dst, off, scmp, snew       // DECENTASN: \dst <- old LDS[off]; if old==\scmp -> LDS[off]=\snew (lane0)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcasr_skip\@
    v_mov_b32 v11, \off                         // vaddr
    v_mov_b32 v14, \snew                        // vNEW
    v_mov_b32 v13, \scmp                        // vCMP -> vdst returns old
    ds_cmpstore_rtn_b32 v13, v11, v14, v13      // MEM=(MEM==\scmp)?\snew:MEM ; v13 <- old MEM
    s_wait_dscnt 0x0
.Lcasr_skip\@:
    s_mov_b32 exec_lo, s49
    v_readfirstlane_b32 \dst, v13               // old (uniform = lane0's result)
.endm
.macro side_final scb, my_pending, saddr, sold   // DECENTASN: this side just finished staging (unique final done-inc).
    // Exactly-once arming (Codex gpt-5.6-sol): clear MY pending bit; if the OTHER side had already cleared theirs
    //   (so only mine remained), I finished SECOND -> I perform the sole CAS to 0. Either side may finish first;
    //   whichever finishes SECOND arms. No generic late store can clobber a live row counter -- STAGE cannot pass
    //   (and thus compute cannot claim) while EITHER pending bit is still set.
    s_add_u32 \saddr, \scb, SL_RBNEXT
    lds_cas_rtn \sold, \saddr, RB_PENDING, (RB_PENDING ^ \my_pending)   // clear my bit (only if BOTH still pending)
    s_cmp_eq_u32 \sold, RB_PENDING
    s_cbranch_scc1 .Lsf_done\@                    // old==RB_PENDING -> I finished FIRST; the other side will arm
    lds_cas_rtn \sold, \saddr, \my_pending, 0     // only my bit remained -> I finished SECOND -> UNIQUE arm to 0
.Lsf_done\@:
.endm
// ---- DEADMAN watchdog: s70 = this wave's start RTC (low 32b); s71 = throttle counter (repurposed high-RTC
//   reg, which is unused at TRACE=0 -- deadman_check only reads s70). The message-bus RTC read (s_sendmsg_rtn)
//   is an SQ-front-end op; hundreds of idle COAST waves hitting it EVERY loop iteration spam the front-end,
//   starving the compositor's SQC(inst) fetch (2026-07-05 MODE1 brick) AND destabilizing the coast wall
//   (identical STAGINSTR work measured 0.32s vs 2.0s). THROTTLE: only read the RTC every DEADMAN_EVERY iters. ----
.ifndef DEADMAN_EVERY
  .set DEADMAN_EVERY, 64          // message-bus RTC-read cadence (in loop iters); force-retire slack = DEADMAN_EVERY iters
.endif
;; ===========================================================================================
;;  DUTYPROBE v2 -- NO MESSAGE BUS. (v1 bricked the box on 2026-07-14 and cost ~1M tokens.)
;;
;;  v1 used s_sendmsg_rtn (SQ message-bus RTC read) at grow/shrink. Its traffic scales INVERSELY
;;  with J -- J=1024 barely touched the bus and ran CLEAN; J=2 spammed it and HUNG; J=1 BRICKED.
;;  deadman_check throttles that same read 1-in-64 with the comment "idle coast waves would else
;;  spam it -> brick". The warning was three lines from where I typed.
;;
;;  v2 reads HW_REG_SHADER_CYCLES_LO instead: a 32-bit free-running shader-cycle counter.
;;    - ONE SALU instruction. No message bus. No store. No VGPR. No wait.
;;    - => SAFE adjacent to s_alloc_vgpr (that hazard is an in-flight STORE during realloc).
;;    - => cheap enough that NO SAMPLING is needed, so the peak/cycle ratio is EXACT.
;;      (v1 gated on CNT_COMP & 63, but CNT_COMP counts SEGMENTS not BURSTS, so the effective
;;       sample rate silently depended on J. That is why v1 printed duty = 3602%.)
;;  32-bit wrap: s_sub_u32 is modular, so any interval < 2^32 cycles (~1.8s) differences correctly.
;;
;;  duty = DP_FAT / DP_CYC = (time held at PEAK) / (grow-to-grow cycle).
;;    low duty  -> trapezoid -> the traveling peak has headroom (budget = AVERAGE footprint)
;;    duty ~1   -> square wave -> staggering cannot help (peak == average)
;; ===========================================================================================
.ifndef DUTYPROBE
    .set DUTYPROBE, 0
.endif
;; ===========================================================================================
;;  NTLOAD -- keep the C accumulator RESIDENT in L2 by refusing to evict it with our own operands.
;;  MEASURED (2026-07-14): a tile's C is 120 KB; 64 WGs in flight => C working set = 7.9 MB
;;  against an 8 MB L2. And we push 1.14 GB of A/B staging traffic through that same 8 MB.
;;  So C -- the ONLY thing here with temporal reuse (accumulated n_kseg/J times per tile) -- is
;;  evicted between every accumulate, and each "L2 atomic" is really an HBM round trip.
;;  A and B are staged straight to LDS: from L2's point of view they are a READ-ONCE FIREHOSE.
;;  th:TH_LOAD_NT tells the cache not to allocate for them.
;;  CAVEAT (this is why it is a knob, not a default): A/B DO have CROSS-WG reuse in L2 (tiles
;;  sharing an M-band reuse A; tiles sharing an N-panel reuse B). NT throws that away. MEASURE.
;; ===========================================================================================
.ifndef NTLOAD
    .set NTLOAD, 0                               // 0 = today. 1 = A+B staging loads are non-temporal.
.endif
.set DP_FAT,  102                                // sum of (shrink - grow)  = cycles held at PEAK
.set DP_CYC,  103                                // sum of (grow  - grow)   = full cycle
.set DP_TG,   104                                // cycle-stamp at peak START (0 = no peak open)
.set DP_TP,   105                                // cycle-stamp at the PREVIOUS grow
.macro duty_init
.if DUTYPROBE
    s_mov_b32 s[DP_FAT], 0
    s_mov_b32 s[DP_CYC], 0
    s_mov_b32 s[DP_TG], 0
    s_mov_b32 s[DP_TP], 0
.endif
.endm
.macro duty_grow                                 // PEAK START -- right after s_alloc_vgpr NFV succeeds
.if DUTYPROBE
    s_getreg_b32 s62, hwreg(HW_REG_SHADER_CYCLES_LO)   // pure SALU. NO message bus. NO store.
    s_cmp_eq_u32 s[DP_TP], 0
    s_cbranch_scc1 .Ldg_first\@                 // first grow: no previous cycle to difference
    s_sub_u32 s63, s62, s[DP_TP]                 // modular u32 -> wrap-safe
    s_add_u32 s[DP_CYC], s[DP_CYC], s63
.Ldg_first\@:
    s_mov_b32 s[DP_TP], s62
    s_mov_b32 s[DP_TG], s62                      // != 0 -> a peak is now OPEN
.endif
.endm
.macro duty_shrink                               // PEAK END -- once the wave is LEAN again
.if DUTYPROBE
    s_cmp_eq_u32 s[DP_TG], 0
    s_cbranch_scc1 .Lds_skip\@                  // no peak open (defensive shrink w/o a grow)
    s_getreg_b32 s62, hwreg(HW_REG_SHADER_CYCLES_LO)
    s_sub_u32 s62, s62, s[DP_TG]
    s_add_u32 s[DP_FAT], s[DP_FAT], s62
    s_mov_b32 s[DP_TG], 0                        // peak closed
.Lds_skip\@:
.endif
.endm
.set DM_PROG, 101                             // per-wave flag: "I made forward progress since the last watchdog tick".
                                               //   ONE SALU write -> safe next to LIVE ACC and next to s_alloc_vgpr.
.macro deadman_progress                        // *** call wherever the wave ACTUALLY GETS WORK DONE ***
.if DEADMAN
    s_mov_b32 s[DM_PROG], 1
.endif
.endm
.macro deadman_stamp                          // stamp start RTC (low 32b in s70) once at entry
.if DEADMAN
    s_sendmsg_rtn_b64 s[70:71], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s71, 0                           // repurpose the (TRACE=0-unused) high-RTC reg as the throttle counter
    s_mov_b32 s[DM_PROG], 0
.endif
.endm
.macro deadman_check_fat                       // deadman for the FAT J-carrier spin. Retires (anti-brick, non-negotiable)
                                                //   but COUNTS the kill: a fat retire DROPS AN UNFLUSHED ACC, so any nonzero
                                                //   CNT_DMFAT means the run's C is WRONG. Loud beats silent.
.if DEADMAN
    s_add_u32 s71, s71, 1
    s_cmp_ge_u32 s71, DEADMAN_EVERY
    s_cbranch_scc0 .Ldmf_skip\@
    s_mov_b32 s71, 0
    s_sendmsg_rtn_b64 s[62:63], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_cmp_lg_u32 s[DM_PROG], 0
    s_cbranch_scc0 .Ldmf_stalled\@
    s_mov_b32 s70, s62                           // PROGRESS since the last tick -> RE-STAMP. A wave that is doing
    s_mov_b32 s[DM_PROG], 0                      //   work can never be killed, however long it lives. THIS is the fix.
    s_branch .Ldmf_skip\@
.Ldmf_stalled\@:
    s_sub_u32 s62, s62, s70                      // elapsed SINCE THE LAST PROGRESS (not since wave entry)
    s_cmp_ge_u32 s62, DEADMAN_TICKS
    s_cbranch_scc0 .Ldmf_skip\@
    cnt_inc CNT_DMFAT                            // *** genuine stall: FAT wave, ACC unflushed -> host marks run INVALID ***
    s_branch .Lflow_retire
.Ldmf_skip\@:
.endif
.endm
.macro deadman_check                           // if alive > DEADMAN_TICKS -> clean force-retire (no wedge)
.if DEADMAN
    s_add_u32 s71, s71, 1                        // THROTTLE: touch the SQ-front-end message bus only every
    s_cmp_ge_u32 s71, DEADMAN_EVERY              //   DEADMAN_EVERY iters (idle coast waves would else spam it -> brick)
    s_cbranch_scc0 .Ldm_skip\@
    s_mov_b32 s71, 0
    s_sendmsg_rtn_b64 s[62:63], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_cmp_lg_u32 s[DM_PROG], 0
    s_cbranch_scc0 .Ldm_stalled\@
    s_mov_b32 s70, s62                           // PROGRESS since the last tick -> RE-STAMP. A wave that is doing
    s_mov_b32 s[DM_PROG], 0                      //   work can never be killed, however long it lives. THIS is the fix.
    s_branch .Ldm_skip\@
.Ldm_stalled\@:
    s_sub_u32 s62, s62, s70                      // elapsed SINCE THE LAST PROGRESS (not since wave entry)
    s_cmp_ge_u32 s62, DEADMAN_TICKS
    s_cbranch_scc1 .Lflow_retire
.Ldm_skip\@:
.endif
.endm
// lds_put_r (RUNTIME-addr write) is also defined inside the .if DSWS2_CONV||DSWS2_ENVELOPE block below;
//   the ring needs it at CONV=0/ENV=0, so define an identical copy here, guarded to avoid a double-def
//   when either gate is on (the ring is always built CONV=0 ENV=0).
.if !(DSWS2_CONV || DSWS2_ENVELOPE)
.macro lds_put_r saddr, ssrc                  // lane-0 write ssrc -> LDS[saddr] (RUNTIME addr)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lputr_skip\@
    v_mov_b32 v[RP_A], \saddr
    v_mov_b32 v[RP_D], \ssrc
    ds_store_b32 v[RP_A], v[RP_D]
    s_wait_dscnt 0x0
.Lputr_skip\@:
    s_mov_b32 exec_lo, s49
.endm
.endif

// ============================================================================================
// Super-tile decode + resident A/B staging macros (A3..A6). Decode (Naming/symbols):
//   ksi = sti & mask ; t = sti >> shift ; mblk = t / NTL ; tcol = t % NTL.
//   FIX 1(d): n_kseg is ALWAYS a power of two (it's KT >> NKSEG_SHIFT, both compile-time-shift-derived),
//   so the sti->(t,ksi) split is an exact shift/mask -- no magic-div, no n_kseg==1 special-case (shift=0,
//   mask=0 falls out of the general path for free: ksi=0, t=sti). /NTL still goes via magic (s12),
//   unsigned-division mul_hi (coop GENDIV idiom), since NTL is not generally a power of two.
// ============================================================================================
.macro DECODE_STI                  // in: s17=sti, s67=mask, s68=shift ; out: s19=mblk s30=tcol s31=ksi ; clob: s18,s36
.if KMAJOR
    s_mul_hi_u32 s31, s17, s76                // ksi = sti / TOTAL   (magic-div, magic_TOTAL in s76; K-major high bits)
    s_mul_i32    s36, s31, s11                // ksi * TOTAL
    s_sub_u32    s18, s17, s36                // t   = sti - ksi*TOTAL  (low)
.else
    s_and_b32    s31, s17, s67                // ksi = sti & mask  (mask-bounded -> ksi in [0,n_kseg-1])
    s_lshr_b32   s18, s17, s68                // t   = sti >> shift
.endif
.if SAFEPROBE
    // brick-PROOF ti clamp (the "future ti clamp" line 752 promised; COOP_STATUS.md:145 racy-garbage-ti->OOB).
    //   A racy/torn sti read (during the claimer's per-super-tile republish) can decode a garbage t -> garbage
    //   mblk/tcol -> the A/B/C SCALAR base goes out of buffer -> gfxhub page fault -> MODE1 brick. SAFEPROBE
    //   already pins the per-lane vaddr (v8/v9/v10); this pins the tile index too, so EVERY global address is
    //   provably in-buffer. s11=TOTAL is userdata, never clobbered. s36 is DECODE_STI scratch (rewritten below).
    s_sub_u32    s36, s11, 1                  // TOTAL-1
    s_min_u32    s18, s18, s36                // t clamped to [0,TOTAL-1] -> mblk<MTL, tcol<NTL (garbage -> in-bounds)
.endif
    s_mul_hi_u32 s19, s18, s12                // mblk = t / NTL
    s_mul_i32    s36, s19, s13                // mblk * NTL
    s_sub_u32    s30, s18, s36                // tcol = t - mblk*NTL
.endm

// RESIDENT B FRAG LAYOUT:  B frag (kstep ks, frag f) at  BRES_OFF + (ks*FN + f)*256
//   (each frag = the SAME 256B block coop stores per B-ring slot; lane*8 vaddr base = v9).
//   Built here as: dst vbase = v9 + BRES_OFF + f*256 , ds_store offset:(ks*FN*256).
// B global addr (lift coop B-feed): Bshuf + tcol*(FN*256=s14) + (seg k0)*  [ksi*KSEG_STEPS*(NT*256=s10)]
//   + f*256 (frag, folded into saddr) + ks*(NT*256=s10) (k-step, folded into saddr).
.macro astage_frags nf                       // stage \nf contiguous M-frags, ALWAYS with MAXIF loads in flight
    // *** SEGK UN-CAPPED (2026-07-14). The first version of this demanded ALL \nf*KSEG_STEPS loads be in
    //   flight at once, which needs \nf*KSEG_STEPS*2 VGPRs. The lean staging window is only 16 regs, so it
    //   .error'd above SEGK=64 -- i.e. MY OWN STAGING FIX CAPPED SEGK, and SEGK is the direct lever on the
    //   number of frontier round-trips (K=2048: SEGK=64 -> 32 segments, SEGK=256 -> 8). On real ml8 shapes
    //   each segment costs ~4,350 cycles of coordination to deliver ~72 cycles of WMMA, so segment COUNT is
    //   exactly what we need to cut. Now the loads are CHUNKED: issue MAXIF, drain, store, repeat. We keep
    //   the full memory-level parallelism (8 loads in flight) at ANY SEGK. ***
    .set NLD,   (\nf * KSEG_STEPS)               // total 8-byte loads for these frags
    .set MAXIF, ((VLEAN - ASTG) / 2)             // loads we can hold in flight = staging regs / 2  (=8)
    .set LD, 0
    .rept ((NLD + MAXIF - 1) / MAXIF)            // chunk count
      .set NN, MAXIF
      .if (LD + NN) > NLD
        .set NN, (NLD - LD)                      // short final chunk
      .endif
      .set q, 0
      .rept NN                                   // ---- issue this chunk's loads ----
        .set jj, ((LD+q) / KSEG_STEPS)           // M-frag index
        .set kk, ((LD+q) % KSEG_STEPS)           // k-step within the frag
        .if kk == 0                              // s44 advances ONCE per new frag; emission is sequential,
          .if jj == 0                            //   so it stays correct ACROSS chunk boundaries.
            s_mov_b32 s44, s40
            s_mov_b32 s45, s41
          .else
            s_add_u32  s44, s44, s32             // += 16*K  (next M-frag; also crosses the rowblk boundary)
            s_addc_u32 s45, s45, 0
          .endif
        .endif
.if NOFEED
        // NOFEED: no load. ds_store below writes whatever is in the reg -> garbage C (by design).
.elseif NTLOAD
        global_load_b64 v[ASTG+q*2:ASTG+q*2+1], v8, s[44:45] offset:(kk*16) th:TH_LOAD_NT
.else
        global_load_b64 v[ASTG+q*2:ASTG+q*2+1], v8, s[44:45] offset:(kk*16)
.endif
        .set q, q+1
      .endr
.if !NOFEED
      s_wait_loadcnt 0x0                         // ONE drain for the whole chunk
.endif
      .set q, 0
      .rept NN                                   // ---- store this chunk (v13 + immediate offsets only) ----
        .set jj, ((LD+q) / KSEG_STEPS)
        .set kk, ((LD+q) % KSEG_STEPS)
        ds_store_b64 v13, v[ASTG+q*2:ASTG+q*2+1] offset:((kk*G*FM + jj)*256)
        .set q, q+1
      .endr
      s_wait_dscnt 0x0
      .set LD, (LD + NN)
    .endr
.endm
.macro BSTAGE                                 // in: s30=tcol s31=ksi ; clob: s20,s21,s23,s25,s26,s27,v13,v[BSTG..]
    s_mul_i32  s20, s30, s14                  // tcol * FN*256
    s_mul_i32  s21, s31, KSEG_STEPS           // ksi * KSEG_STEPS
    s_mul_hi_u32 s25, s21, s10                // *** 64-bit B-offset FIX: hi32(ksi*KSEG_STEPS * NT*256)
    s_mul_i32  s21, s21, s10                  // * NT*256  -> segment k-start byte offset (lo32)
    s_add_u32  s20, s20, s21
    s_addc_u32 s25, s25, 0                     // fold lo-add carry into the high offset word
    s_add_u32  s20, s4, s20
    s_addc_u32 s21, s5, s25                    // s[20:21] = B base (tcol,ksi, seg k-step 0) -- full 64-bit
.Lbcl\@:
    lds_fetch_add s23, BFRAG_NEXT_OFF, 1       // claim frag f
    s_cmp_ge_u32 s23, FN
    s_cbranch_scc1 .Lbsd\@                      // f>=FN -> all frags claimed
    s_lshl_b32 s25, s23, 8                      // f*256
    s_add_u32  s26, s20, s25
    s_addc_u32 s27, s21, 0                      // s[26:27] = frag f base (seg k0)
    v_add_nc_u32 v13, v9, BRES_OFF
    v_add_nc_u32 v13, v13, s25                  // resident B dst vbase for frag f
    // *** CHUNKED (2026-07-14). This demanded KSEG_STEPS*2 VGPRs in flight. At SEGK=256 (KSEG_STEPS=16)
    //   that is v[16..47] -- but THE LEAN BLOCK IS ONLY v0..v31. It wrote 16 registers PAST the lean
    //   allocation -> garbage B -> wrong C. The oracle went ***BAD*** at SEGK=256 the instant I lifted the
    //   .error guard that had been silently protecting this. I chunked ASTAGE and forgot BSTAGE. ***
    .set BMAXIF, ((VLEAN - BSTG) / 2)             // loads in flight = staging regs / 2  (=8)
    .set BLD, 0
    .rept ((KSEG_STEPS + BMAXIF - 1) / BMAXIF)
      .set BNN, BMAXIF
      .if (BLD + BNN) > KSEG_STEPS
        .set BNN, (KSEG_STEPS - BLD)
      .endif
      .set q, 0
      .rept BNN
.if !NOFEED
  .if NTLOAD
        global_load_tr_b64 v[BSTG+q*2:BSTG+q*2+1], v9, s[26:27] th:TH_LOAD_NT
  .else
        global_load_tr_b64 v[BSTG+q*2:BSTG+q*2+1], v9, s[26:27]
  .endif
.endif
        s_add_u32  s26, s26, s10                  // next k-step (advances sequentially ACROSS chunks)
        s_addc_u32 s27, s27, 0
        .set q, q+1
      .endr
.if !NOFEED
      s_wait_loadcnt 0x0
.endif
      .set q, 0
      .rept BNN
        ds_store_b64 v13, v[BSTG+q*2:BSTG+q*2+1] offset:((BLD+q)*FN*256)   // GLOBAL k-step = BLD+q
        .set q, q+1
      .endr
      s_wait_dscnt 0x0
      .set BLD, (BLD + BNN)
    .endr
    lds_inc BFRAG_DONE_OFF                      // frag f STORED -> publish completion (compute gates on this)
    s_branch .Lbcl\@
.Lbsd\@:
.endm

// RESIDENT A FRAG LAYOUT:  A frag (kstep ks, rowblk r, mi) at  ARES_OFF + ((ks*G + r)*FM + mi)*256
//   Built as: dst vbase = v9 + ARES_OFF + r*(FM*256) , ds_store offset:((ks*G*FM + mi)*256).
// A global addr (lift coop compute/A-feed): A + rowblk_abs*(16*FM)*K + mi*16*K + koff, rowblk_abs=mblk*G+r,
//   koff = ksi*SEGK (segment K byte offset, fp8 1B/elem), k-step within segment via global offset:ks*16.
.macro ASTAGE                                 // in: s19=mblk s31=ksi ; clob: s22,s23,s25,s32,s36,s40,s41,s44,s45,v13,v[BSTG..]
    s_lshl_b32 s32, s9, 4                       // rowstride16 = 16*K
.Lacl\@:
    lds_fetch_add s23, AROW_NEXT_OFF, RBU       // claim RBU rowblks: r .. r+RBU-1
    s_cmp_ge_u32 s23, G
    s_cbranch_scc1 .Lasd\@
    s_mul_i32  s36, s19, G
    s_add_u32  s36, s36, s23                     // rowblk_abs = mblk*G + r
    s_mul_i32  s22, s36, (16*FM)
    s_mul_i32  s22, s22, s9                       // rowblk_abs*(16*FM)*K
    s_mul_i32  s25, s31, SEGK                      // ksi*SEGK (segment K byte offset)
    s_add_u32  s22, s22, s25
    s_add_u32  s40, s2, s22
    s_addc_u32 s41, s3, 0                          // s[40:41] = A base (rowblk_abs, mi0, seg k0)
    s_mul_i32  s25, s23, (FM*256)                  // r*FM*256
    v_add_nc_u32 v13, v9, ARES_OFF
    v_add_nc_u32 v13, v13, s25                      // resident A dst vbase for rowblk r
.if RBU > 1
    s_add_u32 s36, s23, (RBU-1)                     // s36 (rowblk_abs) is DEAD here -- reuse as scratch
    s_cmp_lt_u32 s36, G
    s_cbranch_scc1 .Lafull\@                        // all RBU rowblks in range -> full batch
    astage_frags FM                                 // odd-G tail: only ONE rowblk left
    lds_inc AROW_DONE_OFF
    s_branch .Lacl\@
.Lafull\@:
.endif
    astage_frags (RBU*FM)                           // *** RBU*FM*KSEG_STEPS loads ALL IN FLIGHT ***
.if RBU > 1
    lds_fetch_add s36, AROW_DONE_OFF, RBU           // publish RBU rowblks staged
.else
    lds_inc AROW_DONE_OFF
.endif
    s_branch .Lacl\@
.Lasd\@:
.endm

// ============================================================================================
//  FIX 1a -- RING staging macros: slot-indexed BSTAGE_R / ASTAGE_R. Identical math to BSTAGE/ASTAGE
//    but claim/done counters live in the per-slot control block (\scb = SLOTC_BASE + slot*32, runtime)
//    and operands land in the per-slot buffer (\sob = OP_BASE + slot*OPSTRIDE, runtime; B at
//    +BRES_ROFF=0, A at +ARES_ROFF). \scb and \sob are READ-only (never clobbered). Internal address
//    scratch: s46/s47 (free in the feed context). ds offset immediates are vbase-relative -> unchanged.
// ============================================================================================
.macro BSTAGE_R scb, sob             // in: s30=tcol s31=ksi ; clob: s20,s21,s23,s25,s26,s27,s46,s47,v13,v[BSTG..]
.if !DECENTASN
    s_mul_i32  s20, s30, s14                  // tcol * FN*256
    s_mul_i32  s21, s31, KSEG_STEPS           // ksi * KSEG_STEPS
    s_mul_hi_u32 s25, s21, s10                // *** 64-bit B-offset FIX: hi32(ksi*KSEG_STEPS * NT*256)
    s_mul_i32  s21, s21, s10                  // * NT*256  -> segment k-start byte offset (lo32)
    s_add_u32  s20, s20, s21
    s_addc_u32 s25, s25, 0                     // fold lo-add carry into the high offset word
    s_add_u32  s20, s4, s20
    s_addc_u32 s21, s5, s25                    // s[20:21] = B base (tcol,ksi, seg k-step 0) -- full 64-bit
.endif
    s_add_u32  s46, \scb, SL_BFNEXT            // &SL_BFNEXT[slot]
.Lbclr\@:
    lds_fetch_add_r s23, s46, 1                // claim frag f
    s_cmp_ge_u32 s23, FN
    s_cbranch_scc1 .Lbsdr\@                     // f>=FN -> all frags claimed
.if DECENTASN
    // *** SITE J FIX (2026-07-15, sol+Fable): decode the CURRENTLY-resident STI AFTER the frag claim, so the B
    //   operand addresses match the generation whose frag we just claimed. A feeder delayed across a
    //   g -> g+POOL_N reuse would otherwise store OLD-gen operands under a NEW-gen frag claim -> systematic
    //   wrong C. Post-claim (per-iteration) decode shrinks the stale window to the claim->STI-read gap. ***
    s_add_u32  s27, \scb, SL_STI
    lds_get_r  s17, s27
.if GROUPS > 1
    s_and_b32  s17, s17, STI_MASK
.endif
    DECODE_STI                                 // -> s30=tcol s31=ksi (post-claim, gen-consistent)
    s_mul_i32  s20, s30, s14                   // tcol * FN*256  (SITE J: B base recomputed from the post-claim decode)
    s_mul_i32  s21, s31, KSEG_STEPS
    s_mul_hi_u32 s25, s21, s10                  // *** 64-bit B-offset FIX: hi32(ksi*KSEG_STEPS * NT*256)
    s_mul_i32  s21, s21, s10
    s_add_u32  s20, s20, s21
    s_addc_u32 s25, s25, 0                      // fold lo-add carry into the high offset word
    s_add_u32  s20, s4, s20
    s_addc_u32 s21, s5, s25                     // s[20:21] = B base -- full 64-bit
.endif
    s_lshl_b32 s25, s23, 8                      // f*256
    s_add_u32  s26, s20, s25
    s_addc_u32 s27, s21, 0                      // s[26:27] = frag f base (seg k0)
    v_add_nc_u32 v13, v9, \sob                  // + slot operand base
    v_add_nc_u32 v13, v13, s25                  // + f*256   (BRES_ROFF = 0)
    // *** CHUNKED (2026-07-14). This demanded KSEG_STEPS*2 VGPRs in flight. At SEGK=256 (KSEG_STEPS=16)
    //   that is v[16..47] -- but THE LEAN BLOCK IS ONLY v0..v31. It wrote 16 registers PAST the lean
    //   allocation -> garbage B -> wrong C. The oracle went ***BAD*** at SEGK=256 the instant I lifted the
    //   .error guard that had been silently protecting this. I chunked ASTAGE and forgot BSTAGE. ***
    .set BMAXIF, ((VLEAN - BSTG) / 2)             // loads in flight = staging regs / 2  (=8)
    .set BLD, 0
    .rept ((KSEG_STEPS + BMAXIF - 1) / BMAXIF)
      .set BNN, BMAXIF
      .if (BLD + BNN) > KSEG_STEPS
        .set BNN, (KSEG_STEPS - BLD)
      .endif
      .set q, 0
      .rept BNN
.if !NOFEED
  .if NTLOAD
        global_load_tr_b64 v[BSTG+q*2:BSTG+q*2+1], v9, s[26:27] th:TH_LOAD_NT
  .else
        global_load_tr_b64 v[BSTG+q*2:BSTG+q*2+1], v9, s[26:27]
  .endif
.endif
        s_add_u32  s26, s26, s10                  // next k-step (advances sequentially ACROSS chunks)
        s_addc_u32 s27, s27, 0
        .set q, q+1
      .endr
.if !NOFEED
      s_wait_loadcnt 0x0
.endif
      .set q, 0
      .rept BNN
        ds_store_b64 v13, v[BSTG+q*2:BSTG+q*2+1] offset:((BLD+q)*FN*256)   // GLOBAL k-step = BLD+q
        .set q, q+1
      .endr
      s_wait_dscnt 0x0
      .set BLD, (BLD + BNN)
    .endr
    s_add_u32  s47, \scb, SL_BFDONE
.if DECENTASN
    lds_fetch_add_r s25, s47, 1                 // s25 = old SL_BFDONE ; this frag STORED (operands drained above)
    s_cmp_eq_u32 s25, (FN - 1)                  // did THIS increment complete the B side? (unique: +1 each, exactly FN)
    s_cbranch_scc0 .Lbnf\@
    side_final \scb, B_PENDING, s26, s27        // unique B finalizer -> clear B bit / arm if I finished second
.Lbnf\@:
.else
    lds_inc_r s47                               // frag f STORED -> compute gates on SL_BFDONE==FN
.endif
    s_branch .Lbclr\@
.Lbsdr\@:
.endm

.macro ASTAGE_R scb, sob             // in: s19=mblk s31=ksi ; clob: s22,s23,s25,s32,s36,s40,s41,s44,s45,s46,s47,v13,v[BSTG..]
    s_lshl_b32 s32, s9, 4                       // rowstride16 = 16*K
    s_add_u32  s46, \scb, SL_ARNEXT             // &SL_ARNEXT[slot]
.Laclr\@:
    lds_fetch_add_r s23, s46, RBU               // claim RBU rowblks: r .. r+RBU-1
    s_cmp_ge_u32 s23, G
    s_cbranch_scc1 .Lasdr\@
.if DECENTASN
    // *** SITE J FIX (2026-07-15, sol+Fable): decode the CURRENTLY-resident STI AFTER the rowblk claim, so the
    //   A operand addresses match the generation whose rowblk we just claimed (see BSTAGE_R). ***
    s_add_u32  s25, \scb, SL_STI
    lds_get_r  s17, s25
.if GROUPS > 1
    s_and_b32  s17, s17, STI_MASK
.endif
    DECODE_STI                                  // -> s19=mblk s31=ksi (post-claim, gen-consistent)
.endif
    s_mul_i32  s36, s19, G
    s_add_u32  s36, s36, s23                     // rowblk_abs = mblk*G + r
    s_mul_i32  s22, s36, (16*FM)
    s_mul_i32  s22, s22, s9                       // rowblk_abs*(16*FM)*K
    s_mul_i32  s25, s31, SEGK                      // ksi*SEGK
    s_add_u32  s22, s22, s25
    s_add_u32  s40, s2, s22
    s_addc_u32 s41, s3, 0                          // s[40:41] = A base
    s_mul_i32  s25, s23, (FM*256)                  // r*FM*256
    v_add_nc_u32 v13, v9, \sob                     // + slot operand base
    v_add_nc_u32 v13, v13, ARES_ROFF               // + A-within-slot offset (BRES_BYTES)
    v_add_nc_u32 v13, v13, s25                      // + r*FM*256
    s_add_u32  s47, \scb, SL_ARDONE
.if RBU > 1
    s_add_u32 s36, s23, (RBU-1)                     // s36 (rowblk_abs) is DEAD here -- reuse as scratch
    s_cmp_lt_u32 s36, G
    s_cbranch_scc1 .Lafullr\@
    astage_frags FM                                 // odd-G tail: only ONE rowblk left
    lds_inc_r s47
    s_branch .Laclr\@
.Lafullr\@:
.endif
    astage_frags (RBU*FM)                           // *** RBU*FM*KSEG_STEPS loads ALL IN FLIGHT ***
.if RBU > 1
    lds_fetch_add_r s36, s47, RBU                   // publish RBU rowblks staged  (DECENTASN guards RBU=1, so unreached)
.else
.if DECENTASN
    lds_fetch_add_r s45, s47, 1                     // s45 = old SL_ARDONE ; this rowblk STAGED (operands drained above)
    s_cmp_eq_u32 s45, (G - 1)                        // did THIS increment complete the A side? (unique at RBU=1)
    s_cbranch_scc0 .Lanf\@
    side_final \scb, A_PENDING, s44, s45             // unique A finalizer -> clear A bit / arm if I finished second
.Lanf\@:
.else
    lds_inc_r s47
.endif
.endif
    s_branch .Laclr\@
.Lasdr\@:
.endm

// ============================================================================================
//  Phase-B (DSWS2_CONV) consume-point ring-occupancy sensor -- Task 3, READ-ONLY (actuation is Task 5).
//   Mirrors the coop occ_a/occ_b sensor (occ = producer - consumer, sampled where the value is
//   CONSUMED, not at the segment boundary). The claimer's A7 wait-done spin runs CONCURRENTLY with the
//   compute drain, so it observes the ring mid-flight; at the segment boundary the resident region has
//   fully drained and occ would read a stuck ~0 (permanent false-starvation) -- exactly what SPEC warns.
//
//   COUNTER IDENTITIES (confirmed against the live claim/consume sites -- see report):
//     producer = the STORE-completion counters the compute wave actually gates on:
//                  A-ring: AROW_DONE_OFF   (A rowblks resident, monotonic in [0,G]; lds_inc @ ASTAGE)
//                  B-ring: BFRAG_DONE_OFF  (B frags   resident, monotonic in [0,FN]; lds_inc @ BSTAGE)
//                NOT the *_NEXT claim counters: AROW_NEXT/BFRAG_NEXT overshoot the ring depth by the role
//                terminal-bails (G+NAFEED / FN+NBFEED), which would break the occ <= depth bound.
//     consumer = ROWBLK_NEXT_OFF, the compute rowblk-claim clock (consume progress through the super-tile:
//                each claimed rowblk r consumes A(r) and re-reads all FN shared B frags).
//     min-clamp: cons is clamped to prod before the subtract so the u32 result cannot underflow when the
//                consume clock outruns a shallower ring (G=6 > FN=4 -> ROWBLK_NEXT can exceed BFRAG_DONE).
//   INVARIANT preserved: occ_A in [0,G], occ_B in [0,FN]  (nonnegative, bounded by ring depth).
//
//   REGISTER DISCIPLINE (brick-critical; this path is reachable pre-grow -- a >v15 vector temp is
//   OOR-poison under dyn-VGPR, SPEC S4): scalars <= s65 only (s60/s61 scratch; callers pass dst in
//   [s62,s65]); the only vector temps are inside lds_get, which uses v11/v14 (INTERIOR to the launch
//   16-VGPR block) -- NO >v15 temp is introduced here.
// GATE: DSWS2_CONV || DSWS2_ENVELOPE. reserve_try + the BUDGET default are the pool-economy primitives the
//   rolling envelope needs INDEPENDENTLY of role conversion (they touch only VRESV_OFF/lds_fetch_add), so the
//   envelope must be able to run at CONV=0 (the isolation config). Everything in this block is macro/.set
//   definition (emits ZERO bytes), so widening the gate is byte-identical at CONV=0/ENV=0 and CONV=1.
.if DSWS2_CONV || DSWS2_ENVELOPE
.macro occ_sample dst_a, dst_b               // out: \dst_a=occ_A in [0,G], \dst_b=occ_B in [0,FN]; clob s60,s61
    lds_get \dst_a, AROW_DONE_OFF            // prod_a: A rowblks resident (store-completion)
    lds_get \dst_b, BFRAG_DONE_OFF           // prod_b: B frags   resident (store-completion)
    lds_get s60,    ROWBLK_NEXT_OFF          // cons  : compute rowblk-claim consume clock
    s_min_u32  s61, s60, \dst_a              // cons_a = min(clock, prod_a)  (clamp -> no u32 underflow)
    s_sub_u32  \dst_a, \dst_a, s61           // occ_A  = prod_a - cons_a   in [0,G]
    s_min_u32  s61, s60, \dst_b              // cons_b = min(clock, prod_b)
    s_sub_u32  \dst_b, \dst_b, s61           // occ_B  = prod_b - cons_b   in [0,FN]
.endm

// ---- DSWS2_GQUIESCE: device-scoped GLOBAL QUIESCE handshake (mirrors the green occ[20]/occ[0] pattern).
//   All three ops are lane-0-masked (v2==0), exec saved/restored via s49 (the LDS-macro convention -- s49 is
//   never live across a macro boundary, so it is provably free at every site these replace an lds_* op).
//   vaddr = v4 (the stable occ-base per-lane offset, =0, prologue-set), data/dst = v3/v5 (occ scratch vregs,
//   same as the claim/live ops). scope:SCOPE_DEV + uncached occ buffer => device-coherent visibility (the
//   fix). s_wait_storecnt/loadcnt drain before proceeding so the poll observes committed bumps.
.macro gq_reset                              // claimer: occ[QUIESCE_GOFF] = 0 (committed before EPOCH publish)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lgqr_skip\@
    v_mov_b32 v3, 0
    global_store_b32 v4, v3, s[0:1] offset:QUIESCE_GOFF scope:SCOPE_DEV
    s_wait_storecnt 0x0
.Lgqr_skip\@:
    s_mov_b32 exec_lo, s49
.endm
.macro gq_bump                               // follower: occ[QUIESCE_GOFF] += 1 (one bump/wave/super-tile)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lgqb_skip\@
    v_mov_b32 v3, 1
    global_atomic_add_u32 v4, v3, s[0:1] offset:QUIESCE_GOFF scope:SCOPE_DEV
    s_wait_storecnt 0x0
.Lgqb_skip\@:
    s_mov_b32 exec_lo, s49
.endm
.macro gq_read dst                           // claimer: \dst = occ[QUIESCE_GOFF] (lane0 load + broadcast)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lgqrd_skip\@
    global_load_b32 v5, v4, s[0:1] offset:QUIESCE_GOFF scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lgqrd_skip\@:
    s_mov_b32 exec_lo, s49
    v_readfirstlane_b32 \dst, v5
.endm

// ---- Pool-T7 chunk-2 wedge localization (DIAG-only; DSWS2_CONV=0 emits nothing -> .text byte-identical).
//   epoch_mark: lane-0 publishes this role's live epoch (s35) to a host-streamed occ slot so a hung dispatch
//   shows how far each role advanced (stream field roles[C/A/B]). v14<=v15 (feeds/compute are lean-32 at the
//   _quiesce call sites), v4=0 (occ base lane offset, prologue), s49 exec-save (LDS-macro convention). ----
.macro epoch_mark off
.if DSWS2_CONV && DIAG
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lem_skip\@
    v_mov_b32 v14, s35
    global_store_b32 v4, v14, s[0:1] offset:\off scope:SCOPE_DEV
.Lem_skip\@:
    s_mov_b32 exec_lo, s49
.endif
.endm

// bail_mark: PER-WAVE localization mark. Lane-0 writes this wave's epoch (s35) to occ[BAIL_BASE + wid*4]
//   (runtime vaddr since the offset depends on wid=s24). s48 scratch, s49 exec-save (macro-local; free at the
//   _quiesce bail sites), v13 vaddr, v14 data (both <=v15; the wave is lean-32 at every bail site). One-shot
//   per super-tile -> negligible perturbation vs the DIAG per-spin claimer stores. Enabled by DIAG OR BAILMARK.
.macro bail_mark
.if DSWS2_CONV && (DIAG || DSWS2_BAILMARK)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lbmk_skip\@
    s_lshl_b32 s48, s24, 2                        // wid*4
    s_add_u32  s48, s48, BAIL_BASE                // occ byte offset for THIS wave
    v_mov_b32  v13, s48                           // vaddr = per-wave byte offset (lane0)
    v_mov_b32  v14, s35                           // data  = this wave's current epoch
    global_store_b32 v13, v14, s[0:1] scope:SCOPE_DEV
.Lbmk_skip\@:
    s_mov_b32 exec_lo, s49
.endif
.endm

// --------------------------------------------------------------------------------------------
//  Phase-B controller thresholds + sum-envelope budget (Task 4). EPOCH_SHIFT mirrors coop /
//  occ_dispatch (epoch = segcnt >> EPOCH_SHIFT). BUDGET is the per-WG VGPR sum-envelope ceiling
//  the reservation counter must never exceed; default = the launch reservation, which makes the
//  envelope a strict conservation law (a feed->compute grow can only fit if a compute->feed shrink
//  already freed the delta). Task 5 may re-tune via `-defsym BUDGET=` if per-SIMD headroom exists.
// --------------------------------------------------------------------------------------------
.ifndef EPOCH_SHIFT
  .set EPOCH_SHIFT, 3        // decision clock: epoch = segcnt >> EPOCH_SHIFT (small = reactive)
.endif
.ifndef VBUDGET
  .set VBUDGET, 1536        // physical VGPR-file credit ceiling (R9700 wave32, per SIMD). Calibrate.
.endif                       //   Sanity ceiling only: the hardware s_alloc_vgpr is the real concurrent-fat cap.
.ifndef BUDGET
.if DSWS2_ENVELOPE
  .set BUDGET, (WAVES*VLEAN + PEAK_CONC*(NFV-VLEAN))   // rolling: lean floor + concurrent-peak headroom
.else
  .set BUDGET, VBUDGET      // EMERGENT: budget is PHYSICAL, not mix-derived (ledger is dormant; conv-only).
.endif
.endif

// emergent-economy PHYSICAL sanity (always on): all waves fit lean, and >=1 can grow.
.if (WAVES * VLEAN) > BUDGET
  .error "WAVES*VLEAN exceeds VBUDGET -- pool cannot stay all-lean"
.endif
.if (WAVES*VLEAN + (NFV-VLEAN)) > BUDGET
  .error "VBUDGET admits < 1 concurrent grow -- compute can never make progress"
.endif

// ---- TRAVELING-PEAK BATON: concurrent-fat budget cap (2026-07-16). Defined HERE, not at the PEAK_CONC
//   def (~298): it references VBUDGET/NFV/VLEAN, all defined ABOVE this line -- a forward .set at 298 would
//   not resolve. Trapezoid rule (spec DSWS_TRAVELING_PEAK_BATON_2026-07-16.md §1): a fat wave's footprint
//   is lean->peak->lean, so the per-SIMD budget admits ~B/avg-footprint concurrent peaks, NOT B/peak.
.ifndef PEAK_SLACK
  .set PEAK_SLACK, 1        // grow-steps of headroom kept in the pool (brick-avoidance: covers the shrink/grow
.endif                      //   overlap when fat_release fires at shrink-START -- see .Lflow_bshrink). >= 1.
.set PEAK_CONC_EFF, ((VBUDGET - WAVES*VLEAN) / (NFV - VLEAN)) - PEAK_SLACK
.if STAGGER && (PEAK_CONC_EFF < 1)
  .error "PEAK_CONC_EFF < 1: VBUDGET too small for one fat peak above the WAVES*VLEAN lean floor -- raise VBUDGET, or lower WAVES/NFV/PEAK_SLACK."
.endif
.if STAGGER && (PEAK_CONC_EFF > ACC_N)
  .set PEAK_CONC_EFF, ACC_N   // more budget than rowblks -> cap (can't have more concurrent carriers than rowblks handed out)
.endif
// The fat cap the pool enforces = the TIGHTER of the manual throttle (MAXFAT_EFF, the sweep knob) and the
//   physical budget ceiling (PEAK_CONC_EFF). This keeps MAXFAT as a live sweep knob (Task 5) while the
//   budget stays the hard brick-safety ceiling. MAXFAT_EFF defaults to ACC_N when MAXFAT=0 (see ~169).
.set FATCAP_EFF, MAXFAT_EFF
.if STAGGER && (PEAK_CONC_EFF < MAXFAT_EFF)
  .set FATCAP_EFF, PEAK_CONC_EFF
.endif

// try_gate: the lock-free single-winner conversion ticket (transcribed VERBATIM from occ_kernel_coop.s,
//   which transcribes dsws_ctrl_model.cpp gate_try_win + epoch_of EXACTLY). E = segcnt>>EPOCH_SHIFT.
//   gate[dir] holds the last epoch dir fired. Among many waves racing the same (g<E), exactly ONE wins
//   per epoch via the LDS compare-swap; the rest see g advanced. \swin <- 1 iff THIS wave won the
//   (dir,epoch) ticket, else 0. Read-only on state besides gate[dir]; NO role actuation here (Task 5
//   acts on \swin). Scratch: s62..s65 (free at the Task-5 occ_sample->try_gate->reserve_try point --
//   occ_sample's s62/s63 result is consumed into `dir` BEFORE this runs), v5/v6/v7 (<=v15: pre-grow /
//   lean-safe). CAS operand order (gfx1201, GCN order -- NOT flipped, KG 9ed04f3c):
//   ds_cmpstore_rtn_b32 vdst,vaddr,vNEW,vCMP -> MEM=(MEM==vCMP)?vNEW:MEM, vdst<-old. So vsrc0=E (new),
//   vsrc1=g (compare). WIN iff returned-old == g. (Swapping them leaves gate stuck so old==g for ALL
//   racers -> every racer "wins" -> would-win ~= NCOMP*epochs instead of ~= epochs.)
.macro try_gate dir, swin
    lds_get s62, SEGCNT_OFF                    // E = epoch_of(segcnt, EPOCH_SHIFT)
    s_lshr_b32 s62, s62, EPOCH_SHIFT
    lds_get s63, (GATE_OFF + (\dir)*4)         // g = gate[dir]
    s_mov_b32 \swin, 0
    s_cmp_ge_u32 s63, s62                       // g >= E -> dir already fired this/later epoch -> lose
    s_cbranch_scc1 .Ltg_done\@
    s_mov_b32 s65, exec_lo                      // lane0-only CAS (one ticket attempt per WAVE)
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ltg_restore\@
    v_mov_b32 v5, (GATE_OFF + (\dir)*4)        // vaddr = &gate[dir]
    v_mov_b32 v6, s63                           // v6 = g  (vsrc1 = CMP/expected)
    v_mov_b32 v7, s62                           // v7 = E  (vsrc0 = NEW value to store)
    ds_cmpstore_rtn_b32 v6, v5, v7, v6          // gate[dir] = (gate[dir]==g) ? E : gate[dir]; v6 <- old
    s_wait_dscnt 0x0
.Ltg_restore\@:
    s_mov_b32 exec_lo, s65
    v_readfirstlane_b32 s64, v6                 // s64 = old (lane0's CAS result, broadcast)
    s_cmp_eq_u32 s64, s63                        // WIN iff old == g (we were the swapper)
    s_cbranch_scc0 .Ltg_done\@
    s_mov_b32 \swin, 1
.Ltg_done\@:
.endm

// reserve_try: the VGPR sum-envelope reservation (transcribes reserve_grow, dsws_ctrl_model.cpp:47).
//   Reserve first (atomic add of SIGNED \delta on vgpr_reserved), then validate prev+delta <= BUDGET;
//   on over-budget cleanly UNDO (atomic add of -\delta) and reject. The LDS atomic serializes the <=2
//   concurrent grows an epoch permits: the second to validate sees the first's reservation and backs off.
//     GROW  (feed->compute): pass \delta = +(NFV-VLEAN). Over-budget -> undo, \won=0 (stay in role).
//     SHRINK(compute->feed): pass \delta = -(NFV-VLEAN). new = prev+delta < prev <= BUDGET, so the
//                            validate branch is a proven no-op -> \won=1 ALWAYS (shrink never fails).
//   One macro, one call site (Task 5 `reserve_try delta, s_ok`); direction is the sign of \delta.
//   Scratch: s62/s63 (free at the bail-commit point -- try_gate's s62..s65 are long dead by then).
.macro reserve_try delta, won
    lds_fetch_add s62, VRESV_OFF, (\delta)     // s62 = prev reserved; vgpr_reserved += delta
    s_add_u32 s63, s62, (\delta)               // s63 = new reservation = prev + delta
    s_mov_b32 \won, 1
    s_cmp_le_u32 s63, BUDGET                    // new <= BUDGET -> commit (win); shrink always passes
    s_cbranch_scc1 .Lrt_done\@
    lds_fetch_add s62, VRESV_OFF, -(\delta)    // over-budget: undo the reservation, reject
    s_mov_b32 \won, 0
.Lrt_done\@:
.endm

// --------------------------------------------------------------------------------------------
//  Phase-B (Task 5) watermark thresholds + LDS put-runtime helper + bail-time commit macros.
//  Watermark decision (SPEC; mirrors coop CTRL_LOW/CTRL_HIGH, occ_dispatch DSWS_LOW/HIGH):
//    occ_X < CTRL_LOW  -> compute STARVED for X  -> shrink a compute wave into feed-X.
//    occ_X > CTRL_HIGH_X -> feed-X OVER-SERVING  -> grow a feed-X wave into compute.
//  occ_A in [0,G], occ_B in [0,FN] (occ_sample bounds), so the HIGH marks are per-ring-depth.
// --------------------------------------------------------------------------------------------
.ifndef CTRL_LOW
  .set CTRL_LOW, 1                             // occ_X < 1 (== 0, ring empty at consume) -> starved
.endif
.ifndef CTRL_HIGH_A
  .set CTRL_HIGH_A, (G-1)                      // occ_A > G-1 -> A-ring saturated -> A-feed over-serving
.endif
.ifndef CTRL_HIGH_B
  .set CTRL_HIGH_B, (FN-1)                     // occ_B > FN-1 -> B-ring saturated -> B-feed over-serving
.endif

// lds_put_r: lane-0-of-wave write scalar \ssrc -> LDS[\saddr] (RUNTIME byte offset in a sreg). Mirrors
//   the coop lds_put_v idiom but takes a SCALAR address (symmetry with lds_get_r). Used by the claimer's
//   Step-4 snapshot write into the runtime parity half of SNAP_BASE. Temps RP_A/RP_D are v11/v14 (<=v15,
//   pre-grow safe); s49 is the exec save (matches lds_put).
.macro lds_put_r saddr, ssrc
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lputr_skip\@
    v_mov_b32 v[RP_A], \saddr
    v_mov_b32 v[RP_D], \ssrc
    ds_store_b32 v[RP_A], v[RP_D]
    s_wait_dscnt 0x0
.Lputr_skip\@:
    s_mov_b32 exec_lo, s49
.endm

// conv_dec_floor: floor-guarded ATOMIC decrement of a role slot -- \ok <- 1 iff it decremented \slot_off
//   (only when the current value was > 1), else 0 (floor hit; source role must keep >= 1 wave). A
//   ds_cmpstore_rtn_b32 CAS loop (re-reads on a lost race), so two same-source converters in one epoch
//   (e.g. compute->Afeed and compute->Bfeed both dec NCOMP_SLOT) can never drive the slot below 1.
//   Clob: s52 (read value), s53 (new/CAS-return), s65 (exec save); v5/v6/v7 (<=v15, pre-grow safe).
.macro conv_dec_floor slot_off, ok
    s_mov_b32 \ok, 0
.Lcdf_retry\@:
    lds_get s52, \slot_off                     // s52 = current source-slot count
    s_cmp_le_u32 s52, 1
    s_cbranch_scc1 .Lcdf_done\@                 // <=1 -> at floor, cannot convert away (ok stays 0)
    s_sub_u32 s53, s52, 1                       // new = old - 1
    s_mov_b32 s65, exec_lo                      // lane0-only CAS (one attempt per WAVE)
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcdf_restore\@
    v_mov_b32 v5, \slot_off                     // vaddr = &slot
    v_mov_b32 v6, s52                           // v6 = expected old (CMP)
    v_mov_b32 v7, s53                           // v7 = new value (NEW)
    ds_cmpstore_rtn_b32 v6, v5, v7, v6          // slot = (slot==old)? new : slot ; v6 <- prior
    s_wait_dscnt 0x0
.Lcdf_restore\@:
    s_mov_b32 exec_lo, s65
    v_readfirstlane_b32 s53, v6                 // s53 = prior (lane0 CAS result, broadcast)
    s_cmp_eq_u32 s53, s52                        // success iff prior == expected (we were the swapper)
    s_cbranch_scc0 .Lcdf_retry\@                // lost the race -> re-read and retry
    s_mov_b32 \ok, 1
.Lcdf_done\@:
.endm

// conv_apply: the bail-time role-conversion COMMIT (SPEC 3.4 Approach A). Precondition: s58 = s_win
//   (1 iff this wave won the (dir,epoch) ticket). Ordered strictly BEFORE the QUIESCE_CNT bump the
//   CALLER emits after this macro (the quiesce counter is the snapshot handshake).
//     ORDER: (a) floor-guarded dec of \src_slot -> (b) reserve the VGPR sum-envelope \delta (shrink
//     always ok; grow may abort over BUDGET) -> (c) on ok: inc \dst_slot, flip private role reg (s59),
//     s_alloc_vgpr \alloc_sz (GROW=NFV feed->compute / SHRINK=32 compute->feed) with SCC-retry ->
//     (d) on floor-fail or reserve-abort: cancel, remain current role (undo the source dec if a
//     reservation abort happened after the dec).
//   PRE-GROW OOR WINDOW (SPEC 4, #1 brick risk): the wave is lean-32 on entry; every LDS/atomic temp
//   read before the s_alloc_vgpr GROW is <=v15 (occ_sample/try_gate v5/v6/v7 + v11/v14; conv_dec_floor
//   v5/v6/v7; lds_fetch_add v11/v14) and every carried scalar is <=s65. NO >v15 source before GROW.
//   Clob: s52,s53,s54 (+ conv_dec_floor / reserve_try scratch); s59 = new role slot id (record).
.macro conv_apply src_slot, dst_slot, delta, alloc_sz
    s_cmp_eq_u32 s58, 0
    s_cbranch_scc1 .Lca_skip\@                  // lost the ticket -> no conversion this bail
    conv_dec_floor \src_slot, s54               // (a) floor-guarded atomic dec of source slot
    s_cmp_eq_u32 s54, 0
    s_cbranch_scc1 .Lca_skip\@                  // floor-fail (source at 1) -> cancel, remain current role
    reserve_try (\delta), s53                   // (b) reserve VGPR envelope (grow may abort; shrink ok)
    s_cmp_eq_u32 s53, 0
    s_cbranch_scc0 .Lca_commit\@
    lds_fetch_add s52, \src_slot, 1             // (d) reserve aborted: UNDO the source dec, cancel
    s_branch .Lca_skip\@
.Lca_commit\@:
    lds_fetch_add s52, \dst_slot, 1             // (c) inc dest slot (unbounded -> plain atomic add)
    s_mov_b32 s59, \dst_slot                    //     flip private current-role reg (records new role slot id)
.if DIAG || TRACE
    // conversion-commit counter (proves a wave ACTUALLY switched role). Lean-32 pre-grow here -> v3/v4<=v15
    //   OOR-safe; s49 exec-save (macro-local). (DIAG||TRACE)-gated -> DSWS2_CONV/DIAG=0/TRACE=0 byte-identical.
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lca_cm_skip\@
    v_mov_b32 v3, 1
    global_atomic_add_u32 v4, v3, s[0:1] offset:CONVCNT_OFF scope:SCOPE_DEV   // occ[48] += 1
    s_wait_storecnt 0x0
.Lca_cm_skip\@:
    s_mov_b32 exec_lo, s49
.endif
.if CONV_COOLDOWN > 0
  .error "CONV_COOLDOWN clobbers s66, which now holds n_kseg-1 LIVE for the whole kernel (arbitrary-K decode). Give the cooldown its own SGPR before enabling CONV."
.endif
    // ---- s_alloc_vgpr resize: THE pre-grow OOR window closes here; all reads above were <=v15 ----
.Lca_alloc\@:
// ---- RDNA4 dyn-VGPR HAZARD (root-caused 2026-07-13; ISA line 14366) --------------------------
//   S_ALLOC_VGPR's own pseudocode is:   WaitIdleExceptStoreCnt();  n = ReallocVgprs(...)
//   It drains everything EXCEPT STORECNT. An in-flight STORE still sources its data/address from
//   the VGPR file, so reallocating VGPRs underneath it is UNDEFINED -- the store reads registers
//   that realloc has moved/freed. Symptom: PARTIAL, low-magnitude corruption of C with PERFECT
//   counters (the stores DO land; they corrupt the register file on the way out). This cost us
//   two days: flow_gauge fired a global_atomic_add_u32 and did not wait, and the very next
//   instruction was s_alloc_vgpr. The two gauges bracketing s_alloc_vgpr corrupted; the one that
//   did not (the C-store gauge, which already had an s_wait_storecnt) was clean.
//   RULE: NEVER execute s_alloc_vgpr with an outstanding store. Drain first. Enforced here.
// ---------------------------------------------------------------------------------------------
    s_alloc_vgpr \alloc_sz                       // GROW(NFV) / SHRINK(32); SCC-retry (brick-class rule)
    s_cbranch_scc0 .Lca_alloc\@
.Lca_skip\@:
.endm
.endif

// ============================================================================================
//  TFPROBE wall-span capture (TF throughput probe). Realtime-tick min/max into occ[2]/occ[3],
//    mirroring occ_kernel_coop.s's proven timer idiom. Each wave stamps lane-0 only (exec-masked
//    via s49, the DSWS exec-save convention); base addr v4==0 holds kernel-wide (prologue, line
//    ~765; invariant per the "v4=0 occ base lane offset" note). s[30:31] free at entry (DECODE_STI
//    outputs, computed only inside role bodies) and dead at every terminal. Emits ZERO bytes at
//    TFPROBE=0 -> production .text byte-identical.
// ============================================================================================
.macro tfspan op:req, off:req              // op = min (entry, occ[2]/off 8) | max (exit, occ[3]/off 12)
.if TFPROBE
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v2              // lane 0 of each wave only (v2 = tid & 31)
    s_mov_b32 s49, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ltfspan_skip\@
    v_mov_b32 v5, s30                       // low 32 bits of the realtime tick
    global_atomic_\op\()_u32 v4, v5, s[0:1] offset:\off scope:SCOPE_DEV
.Ltfspan_skip\@:
    s_mov_b32 exec_lo, s49
.endif
.endm

// ============================================================================================
//  PHASEPROBE: in-kernel per-phase tick timer for the COMPUTE wave (the critical path). s77 holds
//    this wave's last-stamp RTC (low 32b). phase_reset seeds it; phase_stamp accumulates (now-s77)
//    into occ[\off] and re-seeds. Lane-0-only atomic add; occ slots are ABOVE the per-chunk memset
//    so they sum over the whole run. Scratch s62/s63 (RTC), s64 (delta) -- all free in CONV=0 compute.
//    Six phases -> six occ accumulators (bytes 256..276):
.set PH_FOLLOW_OFF, 256                     // occ[64]: waiting on claimer to publish next super-tile
.set PH_STAGE_OFF,  260                     // occ[65]: waiting on A/B feeds to stage this super-tile
.set PH_GROW_OFF,   264                     // occ[66]: claim rowblk + s_alloc_vgpr GROW 32->112
.set PH_WMMA_OFF,   268                     // occ[67]: LDS frag loads + v_wmma compute
.set PH_FLUSH_OFF,  272                     // occ[68]: global_atomic_add_f32 C flush (split-K reduction)
.set PH_SHRINK_OFF, 276                     // occ[69]: s_alloc_vgpr SHRINK 112->32
// ---- STAGINSTR: lightweight write-once diagnostic counters (is the wall feed or compute?). Lane0
//   atomic at branch points ONLY (never inside the WMMA timing region -> no s_wait_storecnt pollution).
//   STAGINSTR=0 => emits nothing => byte-identical. Ratio COAST/(COAST+COMP) = compute-wave feed-starve.
.ifndef STAGINSTR
    .set STAGINSTR, 0
.endif
.set STINSTR_COAST, 280                     // occ[70]: compute-wave loop iters spent coasting (feed-starved)
.set STINSTR_COMP,  284                     // occ[71]: rowblk-segments actually computed + reduced
.set STINSTR_FEED,  288                     // occ[72]: feed stage completions (staging throughput)
.set STINSTR_GROWFAIL, 292                  // occ[73]: per-burst grow SCC0 (budget full -> coast) = stagger repulsion events
// ---- FLOW-FRONTIER FREEZE-FRAME (occ[74..80]): wid0 snapshots the 3 pipeline heads + the drain-slot's
//   staging/reduction counters + the exit-barrier count EVERY coordinator cycle. On a HANG the coordinator
//   keeps cycling (drainwait->body->loop->here), so the last value on disk (via ML8_COOP_STREAM, or the
//   host timeout-forensics readout) pinpoints WHICH stage stalled: heads frozen w/ RBDONE<ACC_N => a
//   super-tile never computed; BFDONE<FN / ARDONE<G => staging never finished; DRAIN<ASSIGN w/ all
//   counters full => completer/DRAIN-advance bug; QUIESCE<WAVES => barrier never closed. STAGINSTR=0 => none.
.set FDIAG_ASSIGN_OFF,  296                 // occ[74] ASSIGN_HEAD (super-tiles emitted)
.set FDIAG_STAGE_OFF,   300                 // occ[75] STAGE_HEAD  (super-tiles fully staged)
.set FDIAG_DRAIN_OFF,   304                 // occ[76] DRAIN_HEAD  (super-tiles completed+stored)
.set FDIAG_RBDONE_OFF,  308                 // occ[77] drain-slot SL_RBDONE (rowblks reduced; target ACC_N)
.set FDIAG_BFDONE_OFF,  312                 // occ[78] drain-slot SL_BFDONE (B-frags staged; target FN)
.set FDIAG_ARDONE_OFF,  316                 // occ[79] drain-slot SL_ARDONE (A-rowblks staged; target G)
.set FDIAG_QUIESCE_OFF, 320                 // occ[80] count-to-WAVES exit-barrier check-ins (target WAVES)
// ---- COMPLETER-SPIN gauges (2026-07-10 diagnostic): "# waves currently parked in each unbounded, deadman-
//   FREE inner spin". At a wedge the host reads the last-landed value -> pinpoints which spin holds a resident
//   wave (the safemode cause). inc on enter / dec on exit; nets to 0 on a clean run, >0 == a stuck wave. ----
.set FDIAG_SHRINK_OFF,  324                 // occ[81] waves currently in .Lflow_bshrink  (compute burst-shrink)
.set FDIAG_STORE_OFF,   328                 // occ[82] waves currently in the C-store s_wait_storecnt (completer)
.set FDIAG_TASHRINK_OFF,332                 // occ[83] waves currently in .Lflow_tashrink (grew-but-exhausted shrink)
// ---- split-K bank accumulation counters (2026-07-10, group-split bug hunt): monotone counts of bank
//   fresh-writes (ksi==0) vs accumulate-adds (ksi>0). Expected (24 tiles x GROUPS x ACC_N rowblks): writes
//   == that product; adds == writes*(n_kseg-1). A count anomaly at n_kseg=64 => a deterministic re-init/
//   re-process at the group boundary; counts exactly right but C wrong => wrong operands (address overflow). ----
.set FDIAG_BWRITE_OFF,  336                 // occ[84] bank fresh-write (ds_store, ksi==0) events
.set FDIAG_BADD_OFF,    340                 // occ[85] bank accumulate-add (ds_add_f32, ksi>0) events
.set FDIAG_FEEDMT_OFF,  344                 // occ[86] feed-path iterations that found NOTHING to stage
.set FDIAG_FATFULL_OFF, 348                 // occ[87] STAGGER: fat-peak was full -> coasted with NO s_alloc_vgpr drain
.set FDIAG_JWAIT_OFF,   352                 // occ[88] carrier spin-iters in .Lflow_jwait (FAT, holding ACC, starved of stages)
.set FDIAG_CLEAD_OFF,   356                 // occ[89] coast door 2: lead-gate reject (ksi%J != 0)
.set FDIAG_CNOSTG_OFF,  360                 // occ[90] coast door 1: DRAIN >= STAGE (nothing staged)
.set FDIAG_DUTYFAT_OFF, 372                 // occ[93] DUTYPROBE: summed PEAK ticks (>>8)
.set FDIAG_DUTYCYC_OFF, 376                 // occ[94] DUTYPROBE: summed CYCLE ticks (>>8, x DUTY_EVERY)
.set FDIAG_TOKLEAK_OFF, 368                 // occ[92] *** waves that RETIRED still holding a fat token (leaked it) ***
.set FDIAG_DMFAT_OFF,   364                 // occ[91] *** deadman force-retired a FAT carrier -> UNFLUSHED ACC -> RUN INVALID ***
.set FDIAG_STRADDLE_OFF, 380                // occ[95] CLAIM DIAG: claims reaching lds_cas_rtn with exec lane0 INACTIVE. If >0,
                                            //   the skip-and-stale-return false-'won' path (occ_kernel_dsws_flow.s:931/939) is armed.
.set FDIAG_DA_RESET_OFF, 384                // occ[96] CLAIM DIAG: won-claims that did NOT persist (immediate re-read pending|inflight==0)
                                            //   = phantom claims = the SEED both reviews converged on.
.set FDIAG_DA_IMBAL_OFF, 388                // occ[97] release found inflight==0 -> bailed (containment; no underflow poison).
.set FDIAG_BATON_OFF,   392                 // occ[98] BATON: carrier baton-wait spin-iters (>0 => a carrier waited on the
                                            //   VGPR-budget pool and the traveling peak actually engaged). Buffer is 1024
                                            //   words (0x1000B); occ[98] was free (FDIAG block ended at occ[97]=388).
// ============================================================================================
// PERF COUNTERS (STAGINSTR): REGISTER-ACCUMULATE + SINGLE-EMIT-AT-RETIRE.
//   WHY (root-caused 2026-07-13): the previous per-event counters emitted a global_atomic AND flipped
//   exec INSIDE the compute burst. Two of them (BADD/BWRITE) fired between the WMMA and the
//   `ds_add_f32 v12, v[ACC+..]` that consumes its accumulators -- the ONLY instrumentation in the
//   kernel that ran with ACC live. Result: counts came out EXACTLY theoretical while C was garbage
//   (ok=0 bad=1152, max_rel 2.66). Same kernel with STAGINSTR=0 -> oracle CLEAN. The counters were
//   corrupting the very data path they were counting.
//   INVARIANT ENFORCED HERE: a counter must touch NO memory, NO VGPR, and NOT exec, while ACC is live.
//   cnt_inc is therefore pure SALU on a private per-wave SGPR (s84+, above the s71 high-water; SGPRs
//   are per-wave, so no cross-wave race). One lane0 atomic per counter at .Lflow_retire restores the
//   aggregate the host prints -- occ offsets below are UNCHANGED, so the host needs no edit.
//   (This is the same shape PHASEPROBE already used: SGPR accumulators + a single flush at retire.)
// ============================================================================================
.set CNT_COAST,    84                       // -> STINSTR_COAST
.set CNT_COMP,     85                       // -> STINSTR_COMP
.set CNT_FEED,     86                       // -> STINSTR_FEED
.set CNT_GROWFAIL, 87                       // -> STINSTR_GROWFAIL
.set CNT_BWRITE,   88                       // -> FDIAG_BWRITE_OFF   (fires with ACC LIVE -- must stay SALU)
.set CNT_BADD,     89                       // -> FDIAG_BADD_OFF     (fires with ACC LIVE -- must stay SALU)
.set CNT_FEEDMT,   90                       // -> FDIAG_FEEDMT_OFF   (feed wave ran, found NOTHING to stage)
.set CNT_FATFULL,  94                       // -> FDIAG_FATFULL_OFF  (STAGGER: peak was full -> coasted WITHOUT a failed grow)
.set CNT_JWAIT,    95                       // -> FDIAG_JWAIT_OFF    (*** THE INVISIBLE ONE ***: carrier spins in .Lflow_jwait
                                            //    holding ACC in registers, waiting for its next segment to be STAGED. This is
                                            //    the ONLY place a FAT wave burns time and it was never counted.)
.set CNT_CLEAD,    96                       // -> FDIAG_CLEAD_OFF    (coast door 2: ksi%J != 0 -> not my segment to lead)
.set CNT_CNOSTG,   97                       // -> FDIAG_CNOSTG_OFF   (coast door 1: DRAIN >= STAGE -> nothing staged yet)
.set FATHELD,      99                       // 1 = this wave currently holds a fat token (NOT a counter -- a flag)
.set CNT_TOKLEAK,  100                      // -> FDIAG_TOKLEAK_OFF  (*** retired while holding a token = LEAKED it ***)
.set CLAIM_EXECBAD,   102                   // -> occ[95]. CLAIM DIAG: reached lds_cas_rtn with exec lane0 INACTIVE
                                            //   (the exact precondition for the skip-and-stale-return false 'claim won').
.set CLAIM_NOPERSIST, 104                   // -> occ[96]. CLAIM DIAG: a won-claim whose IMMEDIATE SL_RBNEXT re-read shows
                                            //   RB_PENDING set OR inflight==0 -> the claim CAS did NOT persist (phantom claim = the SEED).
.set REL_IMBAL,       105                   // -> occ[97]. release found inflight already 0 -> bailed (no underflow).
                                            //   *** s102=DP_FAT (free @ DUTYPROBE=0, guarded); s104/s105 always free on GFX12
                                            //   (RSRC1 SGPRS term is 0 = all 106 SGPRs). s103 now unused. ***
.if DECENTASN && DUTYPROBE
  .error "DECENTASN claim-diagnostic counter aliases DP_FAT=102; do not combine DECENTASN with DUTYPROBE while this diag exists."
.endif
.set CNT_DMFAT,    98                       // -> FDIAG_DMFAT_OFF    (*** DEADMAN FIRED ON A FAT CARRIER = SILENT DATA LOSS ***
                                            //    .Lflow_retire assumes "ACC dead, wave lean" and does NOT flush ACC. A carrier
                                            //    force-retired out of .Lflow_jwait drops its unflushed partial sum AND never
                                            //    advances the slot's RBDONE. ANY NONZERO VALUE HERE INVALIDATES THE RUN.)
.macro cnt_zero                             // prologue: zero the wave's private accumulators
.if STAGINSTR
    s_mov_b32 s[CNT_COAST], 0
    s_mov_b32 s[CNT_COMP], 0
    s_mov_b32 s[CNT_FEED], 0
    s_mov_b32 s[CNT_GROWFAIL], 0
    s_mov_b32 s[CNT_BWRITE], 0
    s_mov_b32 s[CNT_BADD], 0
    s_mov_b32 s[CNT_FEEDMT], 0
    s_mov_b32 s[CNT_FATFULL], 0                // *** WAS MISSING: occ[87] accumulated an UNINITIALISED SGPR. The
                                               //     "CNT_FATFULL wraps" note was a MISDIAGNOSIS -- it was never zeroed.
    s_mov_b32 s[CNT_JWAIT], 0
    s_mov_b32 s[CNT_CLEAD], 0
    s_mov_b32 s[CNT_CNOSTG], 0
    s_mov_b32 s[CNT_DMFAT], 0
    s_mov_b32 s[CNT_TOKLEAK], 0
    s_mov_b32 s[FATHELD], 0
.if DECENTASN
    s_mov_b32 s[CLAIM_EXECBAD], 0
    s_mov_b32 s[CLAIM_NOPERSIST], 0
    s_mov_b32 s[REL_IMBAL], 0
.endif
.endif
.endm
.macro cnt_inc reg                          // ONE SALU op. No memory, no VGPR, no exec, no vcc. ACC-safe.
.if STAGINSTR
    s_add_u32 s[\reg], s[\reg], 1
.endif
.endm
.macro cnt_emit reg, off                    // lane0 atomic-add this wave's total into occ[\off]
    s_cmp_eq_u32 s[\reg], 0
    s_cbranch_scc1 .Lce_skip\@              // nothing to add -> no traffic
    v_mov_b32 v3, s[\reg]
    global_atomic_add_u32 v4, v3, s[0:1] offset:\off scope:SCOPE_DEV
.Lce_skip\@:
.endm
.macro cnt_flush                            // called ONCE per wave at .Lflow_retire (ACC dead, wave lean)
.if STAGINSTR
    s_mov_b32 s57, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcf_skip\@
    cnt_emit CNT_COAST,    STINSTR_COAST
    cnt_emit CNT_COMP,     STINSTR_COMP
    cnt_emit CNT_FEED,     STINSTR_FEED
    cnt_emit CNT_GROWFAIL, STINSTR_GROWFAIL
    cnt_emit CNT_BWRITE,   FDIAG_BWRITE_OFF
    cnt_emit CNT_BADD,     FDIAG_BADD_OFF
    cnt_emit CNT_FEEDMT,   FDIAG_FEEDMT_OFF
    cnt_emit CNT_JWAIT,    FDIAG_JWAIT_OFF
    cnt_emit CNT_CLEAD,    FDIAG_CLEAD_OFF
    cnt_emit CNT_CNOSTG,   FDIAG_CNOSTG_OFF
    cnt_emit CNT_DMFAT,    FDIAG_DMFAT_OFF
    cnt_emit CNT_TOKLEAK,  FDIAG_TOKLEAK_OFF
.if DECENTASN
    cnt_emit CLAIM_EXECBAD,   FDIAG_STRADDLE_OFF
    cnt_emit CLAIM_NOPERSIST, FDIAG_DA_RESET_OFF
    cnt_emit REL_IMBAL,       FDIAG_DA_IMBAL_OFF
.endif
.if DUTYPROBE
    s_lshr_b32 s[DP_FAT], s[DP_FAT], 12           // >>12: a wave burns ~1e9 shader cycles; x1920 waves would
    s_lshr_b32 s[DP_CYC], s[DP_CYC], 12           //       overflow the u32 occ slot. Shifting BOTH keeps the ratio.
    cnt_emit DP_FAT, FDIAG_DUTYFAT_OFF
    cnt_emit DP_CYC, FDIAG_DUTYCYC_OFF
.endif
    cnt_emit CNT_FATFULL,  FDIAG_FATFULL_OFF
    s_wait_storecnt 0x0
.Lcf_skip\@:
    s_mov_b32 exec_lo, s57
.endif
.endm
.macro instr_inc off
.if STAGINSTR
    s_mov_b32 s57, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2                // lane0 of the wave only
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lii_skip\@
    v_mov_b32 v3, 1
    global_atomic_add_u32 v4, v3, s[0:1] offset:\off scope:SCOPE_DEV   // v4=occ base vaddr(0), v3=1
.Lii_skip\@:
    s_mov_b32 exec_lo, s57
.endif
.endm
// FAT gauge (STAGINSTR): a live count of GROWN (fat NFV-VGPR) compute waves + its running peak. fat_inc
//   fires on grow-SUCCESS (once per burst), fat_dec on the paired burst-shrink -> occ[57] FATLIVE is a
//   live gauge that nets to ~0 at retire (nonzero => inc/dec imbalance bug), occ[58] FATMAX = peak
//   concurrent fat waves (x NFV = VGPR in flight). GLOBAL (aggregate over co-resident WGs): at a FIXED
//   grid the peak's TREND vs the burst-length knob J is the Gate-2 signal; grow-fail stays the direct
//   per-SIMD bind flag. STAGINSTR=0 => emits nothing => byte-identical. lane0-only, off the WMMA path.
.macro fat_inc
.if FATGAUGE
    s_mov_b32 s57, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2                // lane0 of the wave only
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfi_skip\@
    v_mov_b32 v3, 1
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:FATLIVE_OFF th:TH_ATOMIC_RETURN scope:SCOPE_DEV  // v5=old live, v4=addr(0)
    s_wait_loadcnt 0x0
    v_add_nc_u32 v5, v5, 1                    // new live = old+1
    global_atomic_max_u32 v4, v5, s[0:1] offset:FATMAX_OFF scope:SCOPE_DEV                            // peak = max(peak, new)
.Lfi_skip\@:
    s_mov_b32 exec_lo, s57
.endif
.endm
.macro fat_dec
.if FATGAUGE
    s_mov_b32 s57, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfd_skip\@
    v_mov_b32 v3, -1
    global_atomic_add_u32 v4, v3, s[0:1] offset:FATLIVE_OFF scope:SCOPE_DEV                           // v4=addr(0), v3=-1
.Lfd_skip\@:
    s_mov_b32 exec_lo, s57
.endif
.endm
// COMPLETER-SPIN gauge: lane0 +/-1 into occ[81/82/83] to count waves currently parked in an unbounded,
//   deadman-free inner spin (bshrink / C-store wait / tashrink). Mirrors fat_dec (lane0-only, STAGINSTR-gated,
//   v3=val v4=addr(0), s57=exec save). The inc's write LANDS in device memory even if a following store-wait
//   hangs (separate op), so the host's frozen read at a wedge shows exactly which spin holds a resident wave.
.macro flow_gauge off, val                  // SPIN gauges ONLY (FORENSICS). Every call site fires with ACC DEAD:
                                            //   after the bank flush (bshrink), after s_alloc_vgpr 32 (C-store,
                                            //   tashrink). NEVER add a call site while ACC is live -- use cnt_inc.
.if FORENSICS
    s_mov_b32 s57, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lgg_skip\@
    v_mov_b32 v3, \val
    global_atomic_add_u32 v4, v3, s[0:1] offset:\off scope:SCOPE_DEV
.Lgg_skip\@:
    s_mov_b32 exec_lo, s57
    // *** THE ONE MISSING INSTRUCTION (2026-07-14). ***
    //   S_ALLOC_VGPR = WaitIdleExceptStoreCnt() + ReallocVgprs(): it drains EVERYTHING EXCEPT STORECNT.
    //   These gauges bracket the shrink spins, so the atomic above was STILL IN FLIGHT when the very next
    //   s_alloc_vgpr reallocated the register file underneath it -> the store sources data/addr from VGPRs
    //   that realloc had already moved -> register-file corruption. That is why FORENSICS has been pinned
    //   OFF, which is why a hung run could never say WHERE its wave was parked.
    //   The third gauge (the C-store one) ALREADY had an s_wait_storecnt and was MEASURED CLEAN -- draining
    //   is the proven fix, so the gauges do NOT need relocating. Drain here and FORENSICS becomes safe.
    s_wait_storecnt 0x0
.endif
.endm
// flow_snapshot: wid0 writes the pipeline freeze-frame to occ[74..80] each coordinator cycle (POOL_N=1 ->
//   drain slot is always slot 0 = SLOTC_BASE). Reads are wave-uniform (full exec); only lane0 stores.
//   STAGINSTR=0 => nothing emitted. Scratch s54..s61 are free at coordinator-duty entry (the emit re-reads).
.macro flow_snapshot                        // FORENSICS: coordinator-only (s24==0 path) -> never runs with ACC live.
.if FORENSICS
    s_cmp_eq_u32 s71, 0                       // THROTTLE to the deadman's 64-cycle boundary (requires DEADMAN=1):
    s_cbranch_scc0 .Lfsnap_done\@              //   7 global stores EVERY coord cycle keep a hung WG's memory engine
                                                //   hot -> MES can't quiesce -> REMOVE_QUEUE wedge -> MODE1 (Run 7 brick).
    lds_get s54, ASSIGN_HEAD_OFF
    lds_get s55, STAGE_HEAD_OFF
    lds_get s56, DRAIN_HEAD_OFF
    lds_get s58, (SLOTC_BASE + SL_RBDONE)
    lds_get s59, (SLOTC_BASE + SL_BFDONE)
    lds_get s60, (SLOTC_BASE + SL_ARDONE)
    lds_get s61, QUIESCE_CNT_OFF
    s_mov_b32 s57, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfsnap_skip\@
    v_mov_b32 v3, s54
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_ASSIGN_OFF scope:SCOPE_DEV
    v_mov_b32 v3, s55
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_STAGE_OFF scope:SCOPE_DEV
    v_mov_b32 v3, s56
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_DRAIN_OFF scope:SCOPE_DEV
    v_mov_b32 v3, s58
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_RBDONE_OFF scope:SCOPE_DEV
    v_mov_b32 v3, s59
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_BFDONE_OFF scope:SCOPE_DEV
    v_mov_b32 v3, s60
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_ARDONE_OFF scope:SCOPE_DEV
    v_mov_b32 v3, s61
    global_store_b32 v4, v3, s[0:1] offset:FDIAG_QUIESCE_OFF scope:SCOPE_DEV
.Lfsnap_skip\@:
    s_mov_b32 exec_lo, s57
.Lfsnap_done\@:
.endif
.endm
// Per-wave phase accumulators live in SGPRs s78..s83 (NO per-stamp store -> zero memory perturbation, no
//   s_wait_storecnt pollution). s77 = last-stamp RTC. phase_flush emits them ONCE at compute retire.
// STAGGER tokens. SAFE to use VGPRs here (unlike cnt_inc): fat_acquire runs BEFORE the grow and
//   fat_release AFTER the shrink, so ACC is DEAD at every call site. s92/s93 are private scratch.
.macro fat_acquire dst:req                     // \dst = 1 -> I HOLD a token ; 0 -> refused (NET-ZERO writes)
.if STAGGER
    // *** READ-FIRST (2026-07-14). The old form did an UNCONDITIONAL fetch_add(+1) and rolled back on refusal.
    //   The compute loop reaches here on EVERY iteration, and coast-frac is 83% -- so billions of speculative
    //   +1/-1 pairs hammered one LDS word and kept FATTOK CHRONICALLY INFLATED ABOVE MAXFAT. Legitimate carriers
    //   were then refused by "holders" that were already rolling back. That is a livelock, and at the endgame it
    //   strands the last rowblk of the last slot -> RBDONE never reaches ACC_N -> DRAIN never advances -> one wave
    //   spins forever (observed: occ0=1, comp frozen at 99.95% of the chunk, 2 wedges in 6 runs).
    //   THE REFUSAL PATH MUST WRITE NOTHING. Only a wave that plausibly has room even attempts the claim. ***
    s_mov_b32 s92, FATTOK_OFF
    lds_get_r \dst, s92                        // read-only probe
    s_cmp_lt_u32 \dst, FATCAP_EFF              // budget-aware cap = min(MAXFAT_EFF, PEAK_CONC_EFF)
    s_cbranch_scc0 .Lfa_no\@                   // full -> refuse, WITHOUT touching the counter
    lds_fetch_add_r \dst, s92, 1               // room looked available -> claim it
    s_cmp_lt_u32 \dst, FATCAP_EFF              // post-claim validate against the same budget-aware cap
    s_cbranch_scc1 .Lfa_yes\@
    lds_fetch_add_r s93, s92, -1               // lost a GENUINE race for the last slot -> roll back (now rare)
.Lfa_no\@:
    s_mov_b32 \dst, 0
    s_branch .Lfa_end\@
.Lfa_yes\@:
    s_mov_b32 \dst, 1
.Lfa_end\@:
.endif
.endm
.macro fat_release                             // give the token back (shrunk, or grow refused)
.if STAGGER && !BATONGATE                        // BATONGATE=1: NO-OP -- there is no software token (the baton's
    s_mov_b32 s92, FATTOK_OFF                     //   permit-mailbox replaces FATTOK). Neutralizing the macro here
    lds_fetch_add_r s93, s92, -1                  //   makes ALL 5 fat_release call sites no-ops with no underflow risk.
    s_mov_b32 s[FATHELD], 0
.endif
.endm
.macro phase_reset
.if PHASEPROBE
    s_mov_b32 s78, 0
    s_mov_b32 s79, 0
    s_mov_b32 s80, 0
    s_mov_b32 s81, 0
    s_mov_b32 s82, 0
    s_mov_b32 s83, 0
    s_sendmsg_rtn_b64 s[62:63], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s77, s62
.endif
.endm
.macro phase_stamp acc:req                  // \acc += (now - s77); s77 = now  (pure scalar, no store)
.if PHASEPROBE
    s_sendmsg_rtn_b64 s[62:63], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_sub_u32 s64, s62, s77                 // delta (mod 2^32; phase << 2^32 so wrap-safe)
    s_mov_b32 s77, s62
    s_add_u32 \acc, \acc, s64
.endif
.endm
.macro phase_flush                          // lane0 atomic-adds s78..s83 -> occ[64..69]; drained here (not the hot loop)
.if PHASEPROBE
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s49, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lphf_skip\@
    v_mov_b32 v5, s78
    global_atomic_add_u32 v4, v5, s[0:1] offset:PH_FOLLOW_OFF scope:SCOPE_DEV
    v_mov_b32 v5, s79
    global_atomic_add_u32 v4, v5, s[0:1] offset:PH_STAGE_OFF scope:SCOPE_DEV
    v_mov_b32 v5, s80
    global_atomic_add_u32 v4, v5, s[0:1] offset:PH_GROW_OFF scope:SCOPE_DEV
    v_mov_b32 v5, s81
    global_atomic_add_u32 v4, v5, s[0:1] offset:PH_WMMA_OFF scope:SCOPE_DEV
    v_mov_b32 v5, s82
    global_atomic_add_u32 v4, v5, s[0:1] offset:PH_FLUSH_OFF scope:SCOPE_DEV
    v_mov_b32 v5, s83
    global_atomic_add_u32 v4, v5, s[0:1] offset:PH_SHRINK_OFF scope:SCOPE_DEV
    s_wait_storecnt 0x0
.Lphf_skip\@:
    s_mov_b32 exec_lo, s49
.endif
.endm

// ============================================================================================
//  TRACE: per-super-tile time-series row (claimer, lane 0). Written once per super-tile at the
//    quiesce-satisfied drain-exit (.Lqc_q_ok). Captures the adaptive wave-role economy over time:
//    the LIVE role slots (do waves convert?), the per-super-tile ring-occupancy PEAK (s73/s74,
//    tracked across the wait_done spins), the cumulative conversion count, and the envelope vresv.
//    16 u32/row -> buffer[segcnt*64]; bounded by MAXROWS (s72). Emits ZERO bytes at TRACE=0.
//    Persistent trace regs: s70:s71 = buffer VA, s72 = MAXROWS, s73/s74 = ring occA/occB peak.
// ============================================================================================
.macro alllive_dec                             // TRACE: --live on wave exit (pairs with the entry ++ for peak-concurrent)
.if TRACE
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s49, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lalld_skip\@
    v_mov_b32 v3, -1
    global_atomic_add_u32 v4, v3, s[0:1] offset:ALLLIVE_OFF scope:SCOPE_DEV
.Lalld_skip\@:
    s_mov_b32 exec_lo, s49
.endif
.endm

.macro trace_row
.if TRACE
    // claim a GLOBALLY-unique row index (all WGs' claimers share occ[55]) -> no per-WG SEGCNT collision.
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s49, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    v_mov_b32 v14, 1
    global_atomic_add_u32 v15, v4, v14, s[0:1] offset:TRACE_IDX_OFF th:TH_ATOMIC_RETURN scope:SCOPE_DEV   // v15=old idx, v4=addr(0), v14=data(1)
    s_wait_loadcnt 0x0
    s_mov_b32 exec_lo, s49
    v_readfirstlane_b32 s52, v15               // s52 = unique row index (old value returned by the atomic)
    s_cmp_ge_u32 s52, s72                       // row >= MAXROWS -> skip (buffer bound)
    s_cbranch_scc1 .Ltrow_skip\@
    s_lshl_b32 s53, s52, 6                      // row * TRACE_ROW_BYTES(64)
    s_add_u32 s60, s70, s53
    s_addc_u32 s61, s71, 0                      // s[60:61] = row base VA
    s_sendmsg_rtn_b64 s[58:59], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    lds_get s55, SEGCNT_OFF                     // this WG's per-WG super-tile counter (data field, not index)
    v_mov_b32 v14, s58                          //  0 tick_lo
    v_mov_b32 v15, s55                          //  1 segcnt (per-WG)
    v_mov_b32 v16, s35                          //  2 epoch
    lds_get s54, NCOMP_SLOT
    v_mov_b32 v17, s54                          //  3 nComp   (live role slot)
    lds_get s54, NAFEED_SLOT
    v_mov_b32 v18, s54                          //  4 nAfeed
    lds_get s54, NBFEED_SLOT
    v_mov_b32 v19, s54                          //  5 nBfeed
    v_mov_b32 v20, s73                          //  6 occA peak (across wait_done spins)
    v_mov_b32 v21, s74                          //  7 occB peak
    global_load_b32 v22, v4, s[0:1] offset:CONVCNT_OFF scope:SCOPE_DEV   //  8 convCount (cumulative)
    lds_get s54, VRESV_OFF
    v_mov_b32 v23, s54                          //  9 vresv (envelope budget)
    v_mov_b32 v24, s17                          // 10 sti (claimed super-tile id)
    lds_get s54, QUIESCE_CNT_OFF
    v_mov_b32 v25, s54                          // 11 quiesce (final)
    v_mov_b32 v26, s59                          // 12 tick_hi
    v_mov_b32 v27, s69                          // 13 chunkHi (context)
    v_mov_b32 v28, s75                          // 14 wg_id (which workgroup's economy this row belongs to)
    v_mov_b32 v29, 0                            // 15 reserved
    s_wait_loadcnt 0x0                          // convCount load drained before the row store
    v_cmp_eq_u32 vcc_lo, 0, v2                  // lane 0 of the claimer writes the row
    s_mov_b32 s49, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ltrow_wskip\@
    global_store_b128 v4, v[14:17], s[60:61] offset:0  scope:SCOPE_DEV
    global_store_b128 v4, v[18:21], s[60:61] offset:16 scope:SCOPE_DEV
    global_store_b128 v4, v[22:25], s[60:61] offset:32 scope:SCOPE_DEV
    global_store_b128 v4, v[26:29], s[60:61] offset:48 scope:SCOPE_DEV
.Ltrow_wskip\@:
    s_mov_b32 exec_lo, s49
.Ltrow_skip\@:
.endif
.endm

// ============================================================================================
//  KERNEL
// ============================================================================================
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- FIX 1(b,c): derive n_kseg from KT (s8) + the compile-time NKSEG_SHIFT, then the shift/mask
    //   decode pair, into the reserved high SGPRs s66/s67/s68 BEFORE any clobber (SAFEPROBE below reuses
    //   s16 purely as scratch; the role bodies keep all transients <= s65). No v2 kernargs are read here
    //   anymore -- s15/s16/s17 are NOT hardware-preloaded under USER_SGPR=15 (see KERNARG CONTRACT above). ----
    // *** ARBITRARY-K DECODE (2026-07-14). The old form did shift = EXACT log2(n_kseg) via s_ff1, which is
    //   only correct when n_kseg is a POWER OF TWO. That single line made 10 of our 18 real shapes ILLEGAL:
    //   every mlambaformer GEMM but the router (K=768 -> n_kseg=24, K=1536 -> 48) and most of ml8 dense
    //   (K=9216 -> 288, K=2560 -> 80). No SEGK in {16..256} can make those n_kseg a power of two.
    //   FIX: pack ksi into a power-of-2-SIZED FIELD BIG ENOUGH TO HOLD IT, instead of one that must EQUAL it.
    //     shift = CEIL(log2(n_kseg))      mask = (1<<shift)-1      sti = (t<<shift) | ksi,  ksi < n_kseg <= 2^shift
    //   Decode stays a pure AND/SHIFT -- ZERO extra instructions in DECODE_STI, no magic-div, no kernarg
    //   (all 15 user SGPRs are full; a magic constant could not have been passed anyway).
    //   sti just becomes SPARSE (values with ksi >= n_kseg are never emitted), and nothing enumerates sti
    //   densely -- the coordinator claims a TILE and walks ksi = 0..n_kseg-1 itself.
    //   *** mask and (n_kseg-1) COINCIDE only when n_kseg is a power of 2 -- which is why one register used
    //   to serve both. They now DIVERGE (n_kseg=24 -> mask=31, count=23), so they get separate regs:
    //     s66 = n_kseg-1  (COUNT bound: "is this tile's last ksi?")   <- was s67's second job
    //     s67 = mask      (BIT-EXTRACT: ksi = sti & mask)
    //   For any power-of-2 n_kseg, s66 == s67 and this is BIT-IDENTICAL to the old behaviour. ***
    s_lshr_b32    s66, s8, NKSEG_SHIFT        // n_kseg = KT >> NKSEG_SHIFT   (KT=s8)
    s_sub_u32     s66, s66, 1                 // s66 = n_kseg - 1  (COUNT -- LIVE for the whole kernel)
    s_cmp_eq_u32  s66, 0
    s_cbranch_scc1 .Lnk_one                   // n_kseg == 1 -> shift=0, mask=0 (t == sti)
    s_flbit_i32_b32 s68, s66                  // clz(n_kseg-1)   (src != 0 here, so never returns -1)
    s_sub_u32     s68, 32, s68                // shift = 32 - clz(n_kseg-1) = CEIL(log2 n_kseg)
    s_lshl_b32    s67, 1, s68
    s_sub_u32     s67, s67, 1                 // mask = (1<<shift) - 1
    s_branch      .Lnk_done
.Lnk_one:
    s_mov_b32     s68, 0
    s_mov_b32     s67, 0
.Lnk_done:
    // ---- identity (lifted from coop prologue; v0=tid hardware-preloaded) ----
    v_lshrrev_b32 v1, 5, v0                  // wid  = tid >> 5
    v_and_b32     v2, 31, v0                 // lane = tid & 31
    v_and_b32     v6, 15, v0                 // lane & 15 (A vaddr)
    v_mov_b32     v4, 0
    cnt_zero                                 // STAGINSTR: zero this wave's private SGPR counters (s84..s89)
    phase_reset                              // PHASEPROBE: zero s78..s83, stamp s77 = now
.if KMAJOR
    global_load_b32 v3, v4, s[0:1] offset:248 scope:SCOPE_DEV   // occ[62] = magic(TOTAL), host-written
    s_wait_loadcnt 0x0
    v_readfirstlane_b32 s76, v3                // s76 = magic_TOTAL, persistent for every DECODE_STI (K-major)
.endif
    // ---- per-lane address constants (mbgemm-identical; dyn-VGPR arming compatible) ----
    v_mul_lo_u32  v8, v6, s9                 // (lane&15)*K
    v_bfe_u32     v7, v0, 4, 1
    v_lshlrev_b32 v7, 3, v7
    v_add_nc_u32  v8, v8, v7                 // v8 = A vaddr = (lane&15)*K + colhi*8
    v_lshlrev_b32 v9, 3, v2                  // v9 = B/ds vaddr = lane*8
    v_lshlrev_b32 v10, 5, v2                 // v10 = C store vaddr = lane*32
.if SAFEPROBE
    // brick-PROOF: clamp the per-lane VECTOR address regs to a loose upper bound (>= true max) so even a
    //   grow-corrupted vaddr cannot push a global access past the data+guard (pairs with the future ti clamp).
    s_lshl_b32 s16, s9, 4                     // 16*K  (>= v8 max = (lane&15)*K + colhi*8 = 15*K+8)
    v_min_u32 v8, s16, v8                     // clamp A vaddr
    v_min_u32 v9, 0x100, v9                   // clamp B/ds vaddr (256 >= lane*8 max 248)
    v_min_u32 v10, 0x400, v10                 // clamp C vaddr    (1024 >= lane*32 max 992)
.endif

    tfspan min, 8                             // TFPROBE: every wave stamps occ[2] = min entry tick (wall-span start)
.if TRACE
    // total-occupancy: every wave ++live at entry, atomic-max the peak concurrent resident count (occ[1]).
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s49, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lall_enter_skip
    v_mov_b32 v3, 1
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:ALLLIVE_OFF th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v5, v5, 1
    global_atomic_max_u32 v4, v5, s[0:1] offset:4 scope:SCOPE_DEV       // occ[1] = peak concurrent resident waves
.Lall_enter_skip:
    s_mov_b32 exec_lo, s49
.endif

.if DSWS2
// ============================================================================================
//  FIX 1 -- FLOW ECONOMY unified role section (replaces the ring's dispatcher/feed/compute).
//    Every wave runs ONE loop: read ROLE[wid] mailbox -> be that role (resize on change) ->
//    try_grab one atomic (work-or-empty) -> do work, or COAST (code-path flip, no resize). NO
//    publish poll anywhere. wid0 = coordinator (single writer): assigns super-tiles to free slots,
//    seeds/nudges mailboxes, and also does lean B-feed work. See FLOW_ECONOMY_DESIGN.md.
//  Persistent regs:  s24=wid  s34=cur_role  s50=coord period ctr  s69=chunkHi
//  3-frontier pipeline (LDS):  DRAIN_HEAD <= STAGE_HEAD <= ASSIGN_HEAD <= DRAIN_HEAD+POOL_N
// ============================================================================================
    v_readfirstlane_b32 s24, v1                // wid (uniform)
.if DYNVGPR
.Lflow_alloc:
    s_alloc_vgpr 32                            // all start lean; compute grows on adopting ROLE_COMPUTE
    s_cbranch_scc0 .Lflow_alloc               // (NOTE: a bounded-exit + s_endpgm here BRICKS -- s_endpgm from a
.endif                                         //   wave that failed s_alloc_vgpr corrupts the SIMD dyn-VGPR pool
                                               //   -> OOB page fault. Do NOT exit a starved wave; cap W_launch.)
    // live++ : lane0 occ[0] += 1
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lflow_live
    v_mov_b32 v3, 1
    global_atomic_add_u32 v4, v3, s[0:1] scope:SCOPE_DEV
.Lflow_live:
    s_mov_b32 exec_lo, s16
    s_mov_b32 s34, 0xFFFFFFFF                   // cur_role = none -> first .Lflow_body forces a resize
    s_mov_b32 s50, 0                            // coordinator period counter
    deadman_stamp                               // s[70:71] = start RTC (watchdog baseline)
    duty_init
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lflow_wait_init            // non-coordinator waits for LDS init
    // ---- coordinator (wid0) barrier-free LDS init ----
    lds_put ASSIGN_HEAD_OFF, 0
    lds_put STAGE_HEAD_OFF, 0
    lds_put DRAIN_HEAD_OFF, 0
    lds_put FLOWTERM_OFF, 0
.if STAGGER && !BATONGATE
    lds_put FATTOK_OFF, 0                      // *** THE STAGGER'S FAT-WAVE COUNTER. I SHIPPED IT UNINITIALIZED. ***
                                                //   Without this it comes up as whatever garbage is in LDS, so the
                                                //   `fat < MAXFAT` test is decided by that garbage -- IDENTICALLY at
                                                //   every MAXFAT. Signature: TF pinned to the same value (27.9) for
                                                //   MAXFAT=4..12 and a BAD oracle. An uninitialised counter does not
                                                //   look like a broken cap; it looks like a cap that ISN'T THERE.
.endif
.if RETBARRIER
    lds_put QUIESCE_CNT_OFF, 0                  // count-to-WAVES collective-exit counter (reset per dispatch)
.endif
    // EMERGENT economy seed: minimal liveness FLOOR + everything else COMPUTE. wid0=coordinator (runs
    //   lean B-feed between ASSIGN duties), wid1=dedicated A-feed, wid2=dedicated B-feed; wid>=3=COMPUTE.
    //   Excess compute waves self-distribute to feed via .Lflow_coast; concurrent-fat emerges from the
    //   hardware s_alloc_vgpr grow-fail. NO baked NCOMP/NAFEED/NBFEED.
    .set w, 0
    .rept WAVES
      .if w == 0
        lds_put (ROLE_BASE + w*4), ROLE_BFEED
      .elseif w == 1
        lds_put (ROLE_BASE + w*4), ROLE_AFEED
      .elseif w == 2
        lds_put (ROLE_BASE + w*4), ROLE_BFEED
      .else
        lds_put (ROLE_BASE + w*4), ROLE_COMPUTE
      .endif
      .set w, w+1
    .endr
    // init POOL_N slot control blocks: STAMP = sentinel, all counters 0
    .set sl, 0
    .rept POOL_N
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_STI),    0xFFFFFFFF
.if DECENTASN
      // *** Codex gpt-5.6-sol (2026-07-18) COLD-START fix: SL_GEN must init to a NON-generation sentinel, NOT 0.
      //   The feed's gen gate (SL_GEN==cursor, ~3347) and drain both compare SL_GEN to a reservation index; init
      //   SL_GEN=0 ALIASES the first real generation 0, so a feeder passes the gate on the UNSTAMPED slot 0,
      //   claims its zero-init counter, reads the sentinel SL_STI, and stages garbage -> a normal feeder then
      //   re-stages the same frag -> double-staging -> compute consumes whichever image won -> over-large C
      //   (work-exact, timing-varying: the residual after the SL_STI reorder). 0xFFFFFFFF is never a real gen, so
      //   no feed/drain gate passes until the stamp publishes real gen 0 LAST. (The RB_PENDING poison below only
      //   guards the STAGE-walk/compute, NOT the feed's SL_GEN gate.)
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_GEN),    0xFFFFFFFF
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_RBNEXT), RB_PENDING   // POISON: an UNSTAMPED slot must read as
                                                                       //   pending so the pending-gated STAGE-walk
                                                                       //   never advances over it (init 0 would look
                                                                       //   ARMED -> stage over unstaged slot r=0 where
                                                                       //   SL_GEN=0==STAGE=0 -> garbage; ~1 bad/WG).
.else
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_GEN),    0             // coordinator: SL_GEN unused in flow; keep 0 (byte-identity)
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_RBNEXT), 0
.endif
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_RBDONE), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_BFNEXT), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_BFDONE), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_ARNEXT), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_ARDONE), 0
      .set sl, sl+1
    .endr
.if DECENTASN
    // COUPLED-CURSOR init: DA_BASE = -(GROUPS<<shift) = -TOTAL, DA_ZDONE = 0. The FIRST reservation (ASSIGN=0)
    //   sees ASSIGN==DA_ZDONE -> boundary, and (DA_ZDONE - DA_BASE)>>shift == GROUPS -> a TILE boundary, which
    //   claims the WG's first tile (occ[20]++), re-bases DA_BASE=0, zeroes group 0, sets DA_ZDONE=n_kseg.
    s_lshl_b32 s45, GROUPS, s68                 // TOTAL = GROUPS << shift
    s_sub_u32 s45, 0, s45                        // -TOTAL (unsigned wrap; first boundary reads as a TILE claim)
    lds_put DA_BASE_OFF, s45
    lds_put DA_ZDONE_OFF, 0                      // nothing zeroed yet -> ASSIGN(0)==DA_ZDONE(0) -> boundary
    lds_put DA_TILE_OFF, 0                       //   t is don't-care until the first tile is claimed
    lds_put GSTORED_OFF, 0                       // C1: no group C-store has completed yet
.else
    lds_put COORD_KSI_OFF, 0xFFFFFFFF          // tile-claim sentinel: first ASSIGN claims a fresh tile
.endif
    lds_put RINGINIT_OFF, 0xACED               // LAST: publishes "LDS ready"
    global_load_b32 v6, v4, s[0:1] offset:24 scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_readfirstlane_b32 s69, v6                // chunkHi
    s_branch .Lflow_loop
.Lflow_wait_init:
    s_sleep 1
    deadman_check                               // watchdog: wid0 never published init -> clean retire, no wedge
    lds_get s44, RINGINIT_OFF
    s_cmp_eq_u32 s44, 0xACED
    s_cbranch_scc0 .Lflow_wait_init
    global_load_b32 v6, v4, s[0:1] offset:24 scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_readfirstlane_b32 s69, v6                // chunkHi

// ======================= the unified flow loop =======================
.Lflow_loop:
    deadman_check                               // watchdog at every loop head: a stalled frontier -> clean drain
.if DECENTASN
    // DECENTASN: no privileged coordinator -- every wave adopts a role; starved waves ASSIGN at .Lflow_feed_empty.
    //   wid0 takes a FRONTIER SNAPSHOT (occ[74..80]) each loop for hang/stall diagnosis, then does role work like
    //   any wave. flow_snapshot is FORENSICS-gated + self-throttled (s71==0, 1/64) -> byte-identical at FORENSICS=0.
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lflow_body                  //   non-wid0 -> role work
    flow_snapshot                               //   wid0 -> frontier telemetry, then role work
    s_branch .Lflow_body
.else
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lflow_body                 // non-coordinator -> straight to role work
.endif
    // ---- coordinator duty (wid0): ASSIGN + (later) sense/nudge ----
    flow_snapshot                              // pipeline freeze-frame -> occ[74..80] (STAGINSTR; hang forensics)
    lds_get s44, FLOWTERM_OFF
    s_cmp_eq_u32 s44, 0xDEAD
    s_cbranch_scc1 .Lflow_drainwait            // already terminal -> wait for drain
.Lflow_assign_top:                             // BATCHASN batch loop re-enters HERE (snapshot/FLOWTERM done once/visit)
    lds_get s44, ASSIGN_HEAD_OFF               // ah
    lds_get s45, DRAIN_HEAD_OFF               // dh
    s_sub_u32 s46, s44, s45
    s_cmp_ge_u32 s46, POOL_N
    s_cbranch_scc1 .Lflow_coord_period         // pool full -> stop assigning (BATCHASN: this is the batch EXIT)
    // TILE-CLAIM: write-once needs a WG to own a whole tile's n_kseg segments so its LDS banks sum a
    //   full tile. occ[20] now counts TILES; emit n_kseg super-tiles sti=(t<<shift)|ksi per claimed tile.
    lds_get s55, COORD_KSI_OFF                   // combined cursor = group*n_kseg + ksi (GROUPS=1: just ksi)
.if GROUPS > 1
    s_lshl_b32 s46, GROUPS, s68                   // GKSI_MAX = GROUPS * n_kseg  (n_kseg = 1<<shift)
    s_cmp_lt_u32 s55, s46                          // cursor < GROUPS*n_kseg -> more (group,ksi) for this tile
    s_cbranch_scc1 .Lflow_same_tile
.else
    s_cmp_le_u32 s55, s66                        // ksi <= n_kseg-1 -> continue current tile  (s66=COUNT, not the mask)
    s_cbranch_scc1 .Lflow_same_tile
.endif
.if BANKZERO && !WOFLUSH
    // NEW TILE: its banks are about to be REUSED, and the PREVIOUS tile's completer may still be reading
    //   them for its C-store. Require the pool FULLY DRAINED first. NON-BLOCKING: if it is not drained we
    //   just go do our own work and retry next loop -- no spin, no new hang surface.
    lds_get s44, ASSIGN_HEAD_OFF
    lds_get s45, DRAIN_HEAD_OFF
    s_cmp_lt_u32 s45, s44
    s_cbranch_scc1 .Lflow_coord_period          // still draining -> retry later
    zero_banks                                  // -> every ksi is now a pure ds_add_f32
.endif
    // ksi exhausted (or sentinel) -> claim a NEW tile: lane0 occ[20]++ (counts tiles)
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lflow_claim_done
    v_mov_b32 v3, 1
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lflow_claim_done:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s56, v5                  // t = claimed tile index
    s_cmp_ge_u32 s56, s69                        // t >= chunkHi (=TOTAL tiles) -> terminal
    s_cbranch_scc1 .Lflow_set_term
    lds_put COORD_T_OFF, s56                     // remember current tile
    s_mov_b32 s55, 0                             // ksi cursor = 0
    s_branch .Lflow_form_sti
.Lflow_same_tile:
.if GROUPS > 1
.if BANKZERO && !WOFLUSH
    // GROUP BOUNDARY re-zero (bug-a fix): reaching same_tile with ksi==0 (cursor>0) means we are starting
    //   group g>0 of the SAME tile. Its ACC_N banks still hold group g-1's summed values (stored to C but
    //   never cleared -- BANKZERO zeros banks only at TILE claim). Re-zero them, drain-gated exactly like the
    //   new-tile path, so group g accumulates from clean zeros. ksi!=0 -> plain same-group continuation, skip.
    s_and_b32 s46, s55, s67                      // ksi = cursor & mask
    s_cmp_eq_u32 s46, 0
    s_cbranch_scc0 .Lflow_same_tile_go           // ksi != 0 -> not a group boundary
    lds_get s44, ASSIGN_HEAD_OFF                 // prev group's completer store must be drained before we wipe
    lds_get s45, DRAIN_HEAD_OFF
    s_cmp_lt_u32 s45, s44
    s_cbranch_scc1 .Lflow_coord_period           // still draining -> retry later (NON-BLOCKING, cursor unchanged)
    zero_banks                                   // banks clean -> group g's every ksi is a pure ds_add_f32
.Lflow_same_tile_go:
.endif
.endif
    lds_get s56, COORD_T_OFF                     // reuse current tile
.Lflow_form_sti:
.if GROUPS > 1
    s_and_b32 s47, s55, s67                       // ksi   = cursor & mask
    s_lshl_b32 s17, s56, s68                       // t << shift
    s_or_b32 s17, s17, s47                          // sti = (t<<shift) | ksi
    s_lshr_b32 s47, s55, s68                        // group = cursor >> shift  (0..GROUPS-1)
    s_lshl_b32 s47, s47, STAMP_GSHIFT
    s_or_b32 s17, s17, s47                          // STAMP = (group<<28) | sti
.else
    s_lshl_b32 s17, s56, s68                     // t << shift
    s_or_b32 s17, s17, s55                       // sti = (t<<shift) | ksi
.endif
.if GROUPS > 1
    // COUNT-based advance (bug-b fix): walk ksi in [0,n_kseg), roll to next group at ksi==COUNT. Keeps the
    //   (group<<shift)|ksi packing (extraction/STAMP above unchanged) but never emits ksi>=n_kseg at non-pow2.
    //   At pow2 n_kseg (COUNT==mask) this is byte-equivalent to `+1` with carry into the group field.
    s_and_b32 s46, s55, s67                      // ksi = cursor & mask
    s_cmp_eq_u32 s46, s66                         // ksi == COUNT (n_kseg-1) -> last ksi of this group?
    s_cbranch_scc0 .Lflow_ksi_inc
    s_sub_u32 s55, s55, s46                       // cursor -= ksi  (clear the ksi field to 0)
    s_add_u32 s55, s55, s67                       // += mask ...
    s_add_u32 s55, s55, 1                         // ... +1  == +2^shift  -> group++, ksi=0
    s_branch .Lflow_ksi_done
.Lflow_ksi_inc:
    s_add_u32 s55, s55, 1                         // ksi++
.Lflow_ksi_done:
.else
    s_add_u32 s55, s55, 1
.endif
    lds_put COORD_KSI_OFF, s55                   // advance (group,ksi) cursor for next assign
    // assign sti (s17) to slot(ah): reset counters, STAMP=sti, then ASSIGN_HEAD++ (release LAST)
    slot_of s46, s44, s47                        // slot = ah mod POOL_N
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE
    s_add_u32 s45, s48, SL_RBNEXT
.if JDEPTH > 1
    // K-DEPTH J OWNERSHIP: only a GROUP-LEAD segment (ksi % J == 0) admits rowblk claims. On a
    //   non-lead segment the rowblks are already OWNED by the J-carriers still holding ACC in
    //   registers, so POISON RBNEXT to ACC_N -- the compute path's existing `r >= ACC_N -> tryadv`
    //   then turns fresh waves away with no new branch. ksi lives in the LOW bits of sti (and of
    //   STAMP, since the group only occupies the high bits), so s17 & (J-1) IS ksi % J.
    s_and_b32 s47, s17, (JDEPTH - 1)
    s_cmp_eq_u32 s47, 0
    s_cselect_b32 s47, 0, ACC_N                  // lead -> 0 (claimable) ; non-lead -> ACC_N (poisoned)
    lds_put_r s45, s47
.else
    lds_put_r s45, 0
.endif
    s_add_u32 s45, s48, SL_RBDONE
    lds_put_r s45, 0
    s_add_u32 s45, s48, SL_BFNEXT
    lds_put_r s45, 0
    s_add_u32 s45, s48, SL_BFDONE
    lds_put_r s45, 0
    s_add_u32 s45, s48, SL_ARNEXT
    lds_put_r s45, 0
    s_add_u32 s45, s48, SL_ARDONE
    lds_put_r s45, 0
    s_add_u32 s45, s48, SL_GEN
    lds_put_r s45, 0                              // store-claim reset (single-winner bank store; SL_GEN reused)
    s_add_u32 s45, s48, SL_STI
    lds_put_r s45, s17                            // STAMP = gsti
    s_add_u32 s44, s44, 1
    lds_put ASSIGN_HEAD_OFF, s44                  // ASSIGN_HEAD++ (single writer; release)
.if BATCHASN
    // BATCH-ASSIGN (2026-07-14): the coordinator was the WALL -- it published ONE super-tile then went to
    //   do its own feed work, so 30 waves ran at wave-0's publish rate (every real shape 92-100% ASSIGN-bound,
    //   occ[86]). Instead: keep publishing until the pool is FULL, THEN help. BOUNDED (<= POOL_N iters/visit:
    //   ASSIGN can lead DRAIN by at most POOL_N, then the pool-full check at .Lflow_assign_top exits), so no
    //   deadman needed inside. Single-writer to ASSIGN_HEAD is PRESERVED (still only wid0) -> no new race;
    //   STAGE <= ASSIGN still holds. Terminal (out of tiles) exits via .Lflow_set_term inside the loop body.
    s_branch .Lflow_assign_top
.endif
.Lflow_coord_period:
    // (sense/nudge deferred to a later increment -- static launch mix for the first flow build)
    s_branch .Lflow_body                          // coordinator then does its own (lean B-feed) work
.Lflow_set_term:
    lds_put FLOWTERM_OFF, 0xDEAD                  // no more super-tiles; stop claiming occ[20]
.Lflow_drainwait:
    lds_get s44, ASSIGN_HEAD_OFF
    lds_get s45, DRAIN_HEAD_OFF
    s_cmp_lt_u32 s45, s44
    s_cbranch_scc1 .Lflow_body                    // still draining -> keep helping (coordinator feeds/coasts)
    // all assigned super-tiles drained -> tell everyone to retire, then retire self
    .set w, 0
    .rept WAVES
      lds_put (ROLE_BASE + w*4), ROLE_RETIRE
      .set w, w+1
    .endr
    s_branch .Lflow_retire

// ---- role adopt + dispatch (every wave) ----
.Lflow_body:
    s_lshl_b32 s45, s24, 2
    s_add_u32 s45, s45, ROLE_BASE
    lds_get_r s35, s45                            // role = ROLE[wid]  (stale == last role == coast, free)
    s_cmp_eq_u32 s35, ROLE_RETIRE
    s_cbranch_scc1 .Lflow_retire
    s_cmp_eq_u32 s35, s34                         // role unchanged?
    s_cbranch_scc1 .Lflow_dispatch
    // role changed -> resize at this lean boundary.
    // STAGGER: compute waves grow PER-BURST inside .Lflow_compute (the trapezoid), NOT once at role-adopt.
    //   So a role change never GROWS here; every wave sits LEAN at the loop head (a compute wave shrank
    //   after its last burst). A defensive shrink-to-lean preserves that invariant if ever fat crossing here.
.if DYNVGPR
.Lflow_shrink:
    s_alloc_vgpr 32
    s_cbranch_scc0 .Lflow_shrink
.Lflow_resized:
.endif
    s_mov_b32 s34, s35                            // cur_role = role
.Lflow_dispatch:
    s_cmp_eq_u32 s34, ROLE_COMPUTE
    s_cbranch_scc1 .Lflow_compute
    s_branch .Lflow_feed

// ---- COMPUTE work (wave is fat): pull one rowblk from the DRAIN_HEAD slot, WMMA, flush ----
.Lflow_compute:
    lds_get s46, DRAIN_HEAD_OFF                 // dh
    lds_get s44, STAGE_HEAD_OFF                 // sh
    s_cmp_ge_u32 s46, s44                        // DRAIN >= STAGE -> nothing fully staged -> coast to feed
    s_cbranch_scc0 .Lflow_havestage
    cnt_inc CNT_CNOSTG                           // coast door 1
    s_branch .Lflow_coast
.Lflow_havestage:
    phase_stamp s78                              // PH_FOLLOW: all time since last stamp was WAITING for a stage
.if MSSCAN
    // *** spread the waves across the STAGED WINDOW [dh, sh) instead of all piling on the head slot.
    //   s46=dh, s44=sh here. s20/s21/s22 are feed-path scratch -- dead on the compute path. ***
    s_sub_u32 s20, s44, s46                      // window = sh - dh   (1 .. POOL_N)
    s_mov_b32 s21, s24                           // wid
.Lflow_msw:
    s_cmp_lt_u32 s21, s20
    s_cbranch_scc1 .Lflow_mswd
    s_sub_u32 s21, s21, s20                      // wid mod window (window <= 4: a tiny ladder beats a div)
    s_branch .Lflow_msw
.Lflow_mswd:
    s_add_u32 s46, s46, s21                      // my slot = dh + (wid mod window)   -> still < sh
.endif
    slot_of s45, s46, s47                        // slot = (my cursor) mod POOL_N
    s_lshl_b32 s48, s45, 5
    s_add_u32 s48, s48, SLOTC_BASE              // scb
    s_mul_i32 s52, s45, OPSTRIDE                 // slot * OPSTRIDE. WAS `s_lshl_b32 s52, s45, 14` = slot*16384, which is the
                                                  //   SEGK=64 stride HARDCODED. At SEGK=32 OPSTRIDE is 8192, so slot 1
                                                  //   read its operands from the WRONG address -> POOL_N>1 had NEVER
                                                  //   worked at SEGK=32, and SEGK=32 is the only size that fits LDS.
                                                  //   THIS is why POOL_N got nailed to 1. (found 2026-07-13)
    s_add_u32 s52, s52, OP_BASE                // sob
.if JDEPTH > 1
    // LEAD-SEGMENT GATE (before the grow, on purpose): with J>1 only ksi%J==0 admits a claim; the
    //   other J-1 segments are owned by carriers still holding ACC. Without this check a fresh wave
    //   would GROW to NFV, hit the poisoned RBNEXT, and immediately shrink again -- burning a
    //   grow/shrink pair on (J-1)/J of all slots. Test it LEAN and coast (= go feed) instead.
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
    s_and_b32 s47, s17, (JDEPTH - 1)             // ksi % J  (ksi is the LOW bits of sti/STAMP)
    s_cmp_lg_u32 s47, 0
    s_cbranch_scc0 .Lflow_leadok
    cnt_inc CNT_CLEAD                            // coast door 2: (J-1)/J of slots hit this BY CONSTRUCTION
    s_branch .Lflow_coast                        // mid-group segment -> not ours to lead -> help stage
.Lflow_leadok:
.endif
.if STAGGER
.if BATONGATE
    // ---- RIVER (no permit cap): a compute wave with a lead segment just GROWS. The physical s_alloc_vgpr
    //   grow-fail (.Lflow_growfail) is the ONLY throttle. Concurrent-fat EMERGES to fill the VGPR budget --
    //   no seed count, no permit gate, no hardcoded number. The peak TRAVELS because RELSTART frees budget
    //   at shrink-START and the next wave's grow flows straight into the freed space. Fall through to grow.
.else
    // ---- TRAVELING PEAK (old software token layer, BATONGATE=0 A/B path): take a fat-token BEFORE the
    //   grow. If the peak is full, coast without issuing the grow (a refused s_alloc_vgpr is a full drain).
    fat_acquire s49                              // s49 = 1 -> I hold a token ; 0 -> refused (wrote nothing)
    s_cmp_eq_u32 s49, 1
    s_cbranch_scc1 .Lflow_fatok                  // got a token -> I may go fat
    cnt_inc CNT_FATFULL                          // refused: pool full. NO fat_release here (net-zero on refusal).
    s_branch .Lflow_coast                         // NON-BLOCKING: coast + retry next pass (the flood model)
.endif
.Lflow_fatok:
.if !BATONGATE
    s_mov_b32 s[FATHELD], 1                      // BATONGATE=0: token held, must return -- see .Lflow_retire.
.endif                                            //   BATONGATE=1: no software token -> nothing to track.
.endif
    // PER-BURST GROW: trapezoid peak starts here (fat through WMMA+ds_add, lean otherwise). COAST-ON-FAIL
    //   is the floodgate: if the SIMD VGPR budget is full, grow SCC0 -> coast lean, committing NO claim.
.if DYNVGPR
    s_alloc_vgpr NFV
    s_cbranch_scc0 .Lflow_growfail
    duty_grow                                    // *** PEAK START (SALU + RTC only -- no VGPR, no store) ***
    fat_inc                                      // grow-success: ++peak-concurrent fat gauge (STAGINSTR)
    phase_stamp s80                              // PH_GROW (SALU-only -> safe next to s_alloc_vgpr)
.endif
.if DECENTASN
    // POST-GROW SLOT RE-DERIVATION (perf only; correctness comes from the POISON-UNTIL-STAGED CLAIM below).
    //   The grow above is a long WaitIdleExceptStoreCnt drain; DRAIN often advances during it. Re-deriving
    //   the slot from a FRESH post-grow DRAIN (a) points the CAS-claim at the CURRENT head so fewer waves
    //   contend on an already-drained slot, and (b) if DRAIN caught up to STAGE during the grow, backs out
    //   via the no-claim shrink+release path instead of loading a slot it will only coast on. It does NOT
    //   affect correctness -- the poison CAS makes any stale/reused slot un-claimable regardless.
    //   ASSUMES head-pinned compute (MSSCAN=0); guarded at the top of the file.
    lds_get s46, DRAIN_HEAD_OFF                   // fresh dh (post-grow) -- overrides the pre-grow read
    lds_get s44, STAGE_HEAD_OFF                   // fresh sh
    s_cmp_ge_u32 s46, s44                          // caught up during the grow? (DRAIN >= STAGE = nothing staged)
    s_cbranch_scc1 .Lflow_cmp_tryadv              // -> fat_dec + shrink-to-lean + fat_release + loop (NO claim made)
    slot_of s45, s46, s47                          // slot = dh mod POOL_N   (fresh)
    s_lshl_b32 s48, s45, 5
    s_add_u32 s48, s48, SLOTC_BASE                // scb (fresh)
    s_mul_i32 s52, s45, OPSTRIDE
    s_add_u32 s52, s52, OP_BASE                   // sob (fresh)
.endif
.if DECENTASN && JDEPTH > 1
    // POST-GROW LEAD RE-CHECK: the pre-grow lead-gate (.Lflow_leadok) tested the OLD pre-grow slot; the grow's
    //   long drain may have moved DRAIN to a NON-LEAD slot (ksi%J!=0) whose rowblks are owned by a J-carrier.
    //   Under the coupled cursor non-lead slots stamp RB_PENDING->0 (claimable-looking, NOT ACC_N-poisoned), so
    //   re-test the re-derived slot's STI and COAST if non-lead -- else a fresh wave could claim a carrier-owned
    //   rowblk. (The carrier reaches non-lead slots via its cursor walk, not this claim path.)
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
    s_and_b32 s47, s17, (JDEPTH - 1)               // ksi % J  (ksi = low bits of STAMP)
    s_cmp_lg_u32 s47, 0
    s_cbranch_scc1 .Lflow_cmp_tryadv               // non-lead -> coast (not ours to lead)
.endif
.if DECENTASN
    // POISON-UNTIL-STAGED CLAIM (Codex gpt-5.6-sol design, 2026-07-15): one load + at most one CAS, and it
    //   NEVER waits. SL_RBNEXT holds pending bits (unstaged) OR a low rowblk counter (staged+claimable).
    //   Pending / exhausted / lost-CAS => COAST (go help), with NO obligation. The ONLY thing that creates
    //   an obligation is a WON CAS(x -> x+1) with x<ACC_N -- and pending==clear proves the slot is fully
    //   staged and armed, so the operands are valid and there is no straddle (ABA-safe: reuse re-writes
    //   RB_PENDING first, so a stale low-x CAS can't succeed on an unstaged occupant). s33 = k = rowblk.
    s_add_u32 s45, s48, SL_RBNEXT
    lds_get_r s33, s45                            // x = current SL_RBNEXT state
    s_and_b32 s47, s33, RB_PENDING
    s_cmp_lg_u32 s47, 0
    s_cbranch_scc1 .Lflow_cmp_tryadv              // UNSTAGED (pending bit set) -> coast, NO claim, NO wait
    s_and_b32 s47, s33, NEXT_MASK                  // next = x & NEXT_MASK (ignore the inflight field in bits[15:8])
    s_cmp_ge_u32 s47, ACC_N
    s_cbranch_scc1 .Lflow_cmp_tryadv              // next >= ACC_N: all rowblks claimed -> coast
    s_add_u32 s47, s33, 1                          // BANKED (pin retired): CAS just bumps next by 1; no inflight field
    // CLAIM DIAG (sol): is lane 0 active for the upcoming lds_cas_rtn? If not, the helper skips the CAS and
    //   v_readfirstlane returns a STALE v13 that can falsely equal the compare -> false 'claim won' with NO LDS write.
    s_and_b32 s46, exec_lo, 1
    s_cmp_eq_u32 s46, 0
    s_cbranch_scc0 .Lflow_claim_execok
    cnt_inc CLAIM_EXECBAD                           // occ[95]: lane0 INACTIVE at the claim CAS (false-success precondition)
.Lflow_claim_execok:
    lds_cas_rtn s46, s45, s33, s47                 // CAS(SL_RBNEXT, x, x+1); s46 = old
    s_cmp_eq_u32 s46, s33
    s_cbranch_scc0 .Lflow_cmp_tryadv              // LOST (contention / slot reused) -> coast, NO claim
    // WON: the CAS bumped next by 1. This wave now owns rowblk k and WILL bump SL_RBDONE after its banked
    //   flush -- the O1 invariant (every won claim -> exactly one RBDONE++) is what makes RBDONE the drain gate
    //   now that the inflight pin is retired. (Persistence diag removed: with no inflight field it would always
    //   read inflight==0 and false-fire; occ[96]/CLAIM_NOPERSIST is kept defined and prints 0.)
    s_and_b32 s33, s33, NEXT_MASK                  // s33 = k = rowblk index (0..ACC_N-1)
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45                             // gsti of the CLAIMED occupant (stable: pinned by the claim)
.else
    s_add_u32 s45, s48, SL_RBNEXT
    lds_fetch_add_r s33, s45, 1                  // claim LOCAL rowblk (0..ACC_N-1) within this group
    s_cmp_ge_u32 s33, ACC_N                       // (== G when GROUPS=1) group's rowblks exhausted?
    s_cbranch_scc1 .Lflow_cmp_tryadv             // rowblks exhausted (we are fat) -> shrink + try advance
    // read STAMP (gsti) for C addressing
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
.endif
.if GROUPS > 1
    s_lshr_b32 s41, s17, STAMP_GSHIFT             // s41 = group (persists to A-base + completer C-base)
.if SAFEPROBE
    s_min_u32 s41, s41, (GROUPS - 1)               // brick-proof: a torn STAMP read can't push C-base OOB (mirrors the ti clamp)
.endif
    s_and_b32 s17, s17, STI_MASK                   // s17 = sti (strip group high bits)
.endif
    DECODE_STI                                   // s19=mblk s30=tcol s31=ksi
    // zero FM*FN accumulators
    .set idx, 0
    .rept FM*FN
      v_mov_b32 v[ACC+idx*8+0], 0
      v_mov_b32 v[ACC+idx*8+1], 0
      v_mov_b32 v[ACC+idx*8+2], 0
      v_mov_b32 v[ACC+idx*8+3], 0
      v_mov_b32 v[ACC+idx*8+4], 0
      v_mov_b32 v[ACC+idx*8+5], 0
      v_mov_b32 v[ACC+idx*8+6], 0
      v_mov_b32 v[ACC+idx*8+7], 0
      .set idx, idx+1
    .endr
.if JDEPTH > 1
    // ===== K-DEPTH J: accumulate JDEPTH consecutive ksi of THIS rowblk into ACC, flush ONCE. =====
    //   s46 = our own cursor (starts at dh), s91 = j. s33=r / s19=mblk / s30=tcol persist across
    //   the whole group -- every segment of a tile shares them, so DECODE_STI is NOT redone.
    //   ACC is NOT re-zeroed between segments: that is the entire point.
    s_mov_b32 s91, 0                             // j = 0
.if DECENTASN
    // *** Codex D1: re-establish the jloop cursor. The DECENTASN claim CAS (line ~2879, `lds_cas_rtn s46,...`)
    //   CLOBBERED s46 with the rowblk counter; nothing else reloads it, so without this the carrier would walk
    //   from the wrong index (deterministic wrong-C, esp. POOL_N=2/J=2). The cursor MUST be the CLAIMED slot's
    //   LOGICAL index = its SL_GEN (== the reservation index r stamped at .Lflow_da_stamp). Head-pinned compute
    //   claimed the drain-head slot, but DRAIN may have moved during the grow/deschedule -- SL_GEN is the
    //   authoritative logical position, and the coupled cursor makes r,r+1,... consecutive ksi. s48 still holds
    //   the claimed slot's scb. Stable: the lead slot can't drain/recycle until this carrier bumps its RBDONE.
    s_add_u32 s45, s48, SL_GEN
    lds_get_r s46, s45                           // s46 = cursor = SL_GEN[claimed slot] (logical reservation index)
.endif
.Lflow_jloop:
    slot_of s45, s46, s47                        // slot = cursor mod POOL_N  (re-derived each segment)
    s_lshl_b32 s48, s45, 5
    s_add_u32 s48, s48, SLOTC_BASE               // scb for THIS segment's slot
    s_mul_i32 s52, s45, OPSTRIDE
    s_add_u32 s52, s52, OP_BASE                  // sob for THIS segment's slot
.endif
    v_add_nc_u32 v12, v9, s52                    // B resident base (BRES_ROFF=0)
.if GROUPS > 1
    s_mul_i32 s42, s41, ACC_N                      // actual rowblk = group*ACC_N + local ...
    s_add_u32 s42, s42, s33
    s_mul_i32 s37, s42, (FM*256)                   // ... indexes the all-G resident-A staging
.else
    s_mul_i32 s37, s33, (FM*256)
.endif
    v_add_nc_u32 v13, v9, s52
    v_add_nc_u32 v13, v13, ARES_ROFF
    v_add_nc_u32 v13, v13, s37                    // A resident base for rowblk r
    .set ks, 0
    .rept KSEG_STEPS
      .set ni, 0
      .rept FN
        ds_load_b64 v[FB+ni*2:FB+ni*2+1], v12 offset:((ks*FN+ni)*256)
        .set ni, ni+1
      .endr
      .set mi, 0
      .rept FM
        ds_load_b64 v[FA+mi*2:FA+mi*2+1], v13 offset:((ks*G*FM+mi)*256)
        .set mi, mi+1
      .endr
      s_wait_dscnt 0x0
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+mi*2:FA+mi*2+1], v[FB+ni*2:FB+ni*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set ks, ks+1
    .endr
    phase_stamp s81                              // PH_WMMA: ACC zero + frag loads + v_wmma burst just ended.
                                                  //   HOISTED out of the .else (2026-07-13): it used to sit at the top of
                                                  //   the banked reduce, so under WOFLUSH=1 it was COMPILED OUT -- WMMA read
                                                  //   0 and its time fell through into PH_SHRINK. Pure SALU, and ACC is live
                                                  //   here, which is fine (no VGPR write -- see the cnt_inc banner).
.if JDEPTH > 1
    s_add_u32 s91, s91, 1
    s_cmp_ge_u32 s91, JDEPTH
    s_cbranch_scc1 .Lflow_jdone                  // *** LAST segment of the group: fall through to the flush and let
                                                  //   the SHARED post-flush path settle its RBDONE/DRAIN. ***
                                                  //   THE BUG THIS FIXES (measured bad=64 at J=2/4/8, every fragment):
                                                  //   retiring the LAST slot here would advance DRAIN while our ACC is
                                                  //   still IN REGISTERS. At a tile's end that makes DRAIN==ASSIGN fire
                                                  //   with sums unflushed -> the coordinator calls zero_banks and opens
                                                  //   the NEXT tile -> the carriers then flush into freshly-zeroed banks
                                                  //   of the WRONG tile. Every tile loses its last group AND poisons its
                                                  //   successor. DRAIN MUST NEVER PASS AN UNFLUSHED SEGMENT.
    // ---- mid-group segment: safe to retire the slot NOW (its operands are consumed, and it is not the
    //   tile's last segment, so no bank-zero gate can trip). This is what keeps J's LDS cost at ZERO:
    //   we never hold J slots resident -- we hold J segments' worth of SUM in the register file.
    cnt_inc CNT_COMP
    deadman_progress                             // a J-segment was computed -> this carrier is ALIVE AND WORKING
    s_add_u32 s45, s48, SL_RBDONE
    lds_fetch_add_r s47, s45, 1                  // s47 = old RBDONE for this segment's slot
    s_add_u32 s47, s47, 1
    s_cmp_lt_u32 s47, ACC_N
    s_cbranch_scc1 .Lflow_jnext                  // not the last rowblk in this slot -> nothing to drain
    drain_advance                                // (was an unconditional DRAIN++ -- under MULTISLOT slots can
                                                 //  complete OUT OF ORDER, so this must WALK. See the banner.)
.Lflow_jnext:
    s_add_u32 s46, s46, 1                        // cursor++ -> the next ksi of this same tile
.Lflow_jwait:
    cnt_inc CNT_JWAIT                            // *** THE INVISIBLE STALL: FAT, ACC LIVE, starved. SALU-only. ***
    deadman_check_fat                            // *** was a bare deadman_check -> branched to .Lflow_retire ("ACC dead, wave
                                                  //   lean") WITH ACC LIVE AND FAT. Silent data loss: at J=64 it ate 34% of the
                                                  //   computed segments and the 1-tile oracle sample never saw it. Now counted.
    // We are FAT here, holding ACC, waiting for our next segment to be staged. The WAVES>=2*ACC_N guard
    //   guarantees lean waves remain to stage it. deadman_check_fat (above) retires clean on a genuine stall
    //   (no MODE1 brick) WHILE counting CNT_DMFAT so a dropped-ACC is LOUD.
.if !DECENTASN
    //   *** Codex B1: this SECOND, bare `deadman_check` shares s71 with the fat check, so on alternating
    //   iterations IT (not the fat check) crosses DEADMAN_EVERY and retires the FAT carrier via .Lflow_retire
    //   WITHOUT CNT_DMFAT -> a SILENT dropped-ACC. Dropped for DECENTASN (one watchdog only). Kept for the
    //   coordinator path (DECENTASN=0) to preserve its verified byte-identical behavior; the fat check alone
    //   still provides the anti-brick retire there too, but do not perturb the proven bin without a re-baseline. ***
    deadman_check
.endif
    lds_get s44, STAGE_HEAD_OFF
    s_cmp_lt_u32 s46, s44
    s_cbranch_scc1 .Lflow_jloop                  // staged -> compute the next segment into the SAME ACC
    s_sleep 1
    s_branch .Lflow_jwait
.Lflow_jdone:
.endif
.if WOFLUSH
    // BURST-SCOPED FLUSH (no LDS bank): atomic-add this segment's fp32 ACC frags STRAIGHT to C[rowblk r].
    //   C is memset 0 by the host; every segment of every rowblk atomic-adds -> C = full split-K sum.
    //   Same addressing as the write-once completer store (v10=lane*32, offset frag*1024+e*4) so it lands
    //   in the identical C locations -> correct by construction. s19=mblk s30=tcol s33=rowblk r (all live).
    s_mul_i32 s38, s19, s13                        // mblk*NTL
    s_add_u32 s38, s38, s30                        // + tcol
    s_mul_i32 s38, s38, (G*FM*FN*1024)            // * per-tile C bytes
    s_mul_i32 s40, s33, (FM*FN*1024)             // + rowblk r * per-rowblk C bytes
    s_add_u32 s38, s38, s40
    s_add_u32 s28, s6, s38
    s_addc_u32 s29, s7, 0                          // s[28:29] = C rowblk base
.if !NOCFLUSH                                      // NOCFLUSH was a DEAD KNOB (defined, never referenced -- same trap as
                                                   //   STINSTR_FEED). WIRED 2026-07-13 so the split-K atomic cost can
                                                   //   actually be ablated. =1 keeps ALL bookkeeping/handshake/WMMA and
                                                   //   drops ONLY the C atomics -> C is never written -> oracle MUST fail.
                                                   //   span/TF only. Never ship with this set.
    .set frag, 0
    .rept FM*FN
      .set e, 0
      .rept 8
        global_atomic_add_f32 v10, v[ACC+frag*8+e], s[28:29] offset:(frag*1024 + e*4) scope:SCOPE_DEV
        .set e, e+1
      .endr
      .set frag, frag+1
    .endr
    s_wait_storecnt 0x0                            // J=1 correctness baseline: drain this wave's atomics
.endif
    phase_stamp s82                              // PH_FLUSH: the global_atomic_add_f32 C burst + its drain. The WOFLUSH
                                                  //   twin of the banked reduce's stamp below -- without it this whole
                                                  //   flush was invisible and its time was charged to PH_SHRINK.
.else
    // WRITE-ONCE REDUCE: accumulate this segment's partial into LDS bank[r] (mirrors C frag layout;
    //   vaddr = v10=lane*32, base = ACC_BASE + r*ACC_STRIDE). ksi==0 (tile's first segment, POOL_N=1
    //   guarantees it drains before any later ksi) WRITES; ksi>0 ADDS. C is stored ONCE at ksi==mask
    //   (last segment) in .Lflow_cmp_tryadv. s31=ksi (survives WMMA), s33=rowblk r.
    acc_base_of s39, s33                          // s39 = ACC_BASE + r*ACC_STRIDE  (PH_WMMA now stamped above the .if,
                                                  //   shared by both paths -- was here, where WOFLUSH could not see it)
    v_add_nc_u32 v12, v10, s39                     // v12 = bank r ds vaddr (lane*32 + bankbase)
.if !BANKZERO                                     // BANKZERO=1 -> ALWAYS ds_add_f32 (banks pre-zeroed)
    s_cmp_eq_u32 s31, 0                            // first segment of this tile's rowblk?
    s_cbranch_scc1 .Lflow_bankwr
.endif
    cnt_inc CNT_BADD                                // DIAG count: a ksi>0 accumulate-add event. SALU-ONLY: ACC is LIVE
                                                    //   here (consumed by the ds_add_f32 below) -- see cnt_inc banner.
    .set frag, 0
    .rept FM*FN
      .set e, 0
      .rept 8
        ds_add_f32 v12, v[ACC+frag*8+e] offset:(frag*1024 + e*4)
        .set e, e+1
      .endr
      .set frag, frag+1
    .endr
    s_branch .Lflow_bankdn
.if !BANKZERO                                     // BANKZERO=1: banks pre-zeroed, no fresh-write path
.Lflow_bankwr:
    cnt_inc CNT_BWRITE                              // DIAG count: a ksi==0 fresh-write event. SALU-ONLY: ACC is LIVE
                                                    //   here (consumed by the ds_store_b32 below) -- see cnt_inc banner.
    .set frag, 0
    .rept FM*FN
      .set e, 0
      .rept 8
        ds_store_b32 v12, v[ACC+frag*8+e] offset:(frag*1024 + e*4)
        .set e, e+1
      .endr
      .set frag, frag+1
    .endr
.endif
.Lflow_bankdn:
    s_wait_dscnt 0x0
    phase_stamp s82                              // PH_FLUSH: the split-K LDS bank reduce just ended
.endif
    cnt_inc CNT_COMP                              // diag: a rowblk-segment was actually computed+reduced
    deadman_progress                              // rowblk-segment done -> ALIVE AND WORKING
    s_add_u32 s45, s48, SL_RBDONE                 // s48 = the group's LAST slot (JDEPTH>1) or its only one (J=1)
    lds_fetch_add_r s47, s45, 1                   // s47 = old RBDONE. Under DECENTASN this is now DIAGNOSTIC ONLY
                                                  //   (drain authority moved to the SL_RBNEXT inflight field); under the
                                                  //   baseline it is still the drain gate. Bumped AFTER the flush: DRAIN
                                                  //   must never pass a segment whose sum is still in registers.
.if DECENTASN
    // BANKED DECENTASN (pin retired 2026-07-16): NO inflight release. The SL_RBDONE++ just above IS the
    //   completion signal -- it was bumped AFTER this segment's ds_add_f32 drained (s_wait_dscnt in the banked
    //   flush at .Lflow_bankdn), so RBDONE==ACC_N proves all bank writes are globally visible and DRAIN may free
    //   the slot. This is the pin's old job, done by RBDONE. (REL_IMBAL/occ[97] retired; host still prints 0.)
.endif
.if DYNVGPR
    fat_dec                                       // burst end: --live fat gauge (STAGINSTR) before the shrink
.if RELSTART
    fat_release                                   // *** THE BATON *** (RELSTART=1): return peak-budget to the pool AT
                                                  //   shrink-START, not shrink-end. The instant we commit to shrinking the
                                                  //   budget is free, so the NEXT carrier that comes around the loop and
                                                  //   does fat_acquire grabs it and grows -- the traveling peak, with NO
                                                  //   wave blocking on any other (flood model, FLOW_ECONOMY_DESIGN.md).
                                                  //   PEAK_SLACK reserves headroom for the shrink-latency overlap; the
                                                  //   hardware s_alloc_vgpr is the REAL cap, so a carrier that races ahead
                                                  //   of the physical free just grow-fails -> .Lflow_growfail -> coasts
                                                  //   (never bricks). Sets FATHELD=0 here; the shrink spin below is
                                                  //   deadman-free and cannot fail, and .Lflow_retire sees no token held.
.endif
.if STAGGER
    // ---- BATON POKE (A): I'm about to shrink and FREE my peak budget. Poke ONE other wave "grow now" so it
    //   rises into the space I'm vacating -> keeps >=1 wave at peak (continuous compute, no valley). Next-
    //   available round-robin, O(1), plain lane-0 LDS write BEFORE the shrink s_alloc_vgpr (ACC dead post-
    //   flush -- same safe scratch fat_release used here). NOT a gate/cap: the poke only NUDGES; the poked
    //   wave still grows only if the physics allow (grow-fail -> coast). No poll of anyone's state.
    lds_fetch_add s92, BATON_NEXT_OFF, 1          // atomic free-running cursor (wave-uniform)
    s_mul_hi_u32  s93, s92, BATON_MAGIC           // q ~= idx / NCOMPUTE
    s_mul_i32     s93, s93, NCOMPUTE
    s_sub_u32     s92, s92, s93                   // rem in [0, 2*NCOMPUTE)
    s_cmp_ge_u32  s92, NCOMPUTE
    s_cbranch_scc0 .Lbaton_norm
    s_sub_u32     s92, s92, NCOMPUTE              // normalize into [0, NCOMPUTE)
.Lbaton_norm:
    s_add_u32     s92, s92, FIRST_COMPUTE_WID     // target wid in [FIRST_COMPUTE_WID, WAVES)
    s_lshl_b32    s92, s92, 2
    s_add_u32     s92, s92, GROWPERMIT_BASE       // &NOTIFY[target]
    lds_put_r     s92, 1                          // NOTIFY[target] = 1  ("grow now")
.endif
    flow_gauge FDIAG_SHRINK_OFF, 1                // DIAG: entering the unbounded/deadman-free burst-shrink
.Lflow_bshrink:
    s_alloc_vgpr 32                               // SHRINK -> lean (close the trapezoid burst) BEFORE any store
    s_cbranch_scc0 .Lflow_bshrink
    duty_shrink                                   // *** PEAK END ***
    phase_stamp s83                              // PH_SHRINK (SALU-only -> safe next to s_alloc_vgpr)
.if !RELSTART
    fat_release                                   // RELSTART=0: pristine -- release at shrink-END (the original position)
.endif
    flow_gauge FDIAG_SHRINK_OFF, -1               // DIAG: shrink succeeded (occ[81] nets to 0)
.endif
.if BANKZERO && !WOFLUSH
    // ---- TILE-SCOPED COMPLETER (2026-07-13). The C-store must wait for the whole TILE to be reduced,
    //   not for one SLOT to finish. With a deep pool the slot holding ksi==mask can finish BEFORE the
    //   slots holding earlier ksi have accumulated -> it would store half-summed banks (measured:
    //   POOL_N=2/3 -> bad=96/116). Every rowblk-segment bumps TILEDONE[group]; whoever brings it to
    //   ACC_N*n_kseg owns the store. NOTE all ksi of a tile share the same t, so mblk/tcol (s19/s30)
    //   are identical across them -- ANY wave of the tile can do the store, not just ksi==mask.
.if GROUPS > 1
    s_lshl_b32 s43, s41, 2                        // group*4
.else
    s_mov_b32 s43, 0
.endif
    s_add_u32 s43, s43, TILEDONE_BASE
    lds_fetch_add_r s51, s43, JDEPTH              // s51 = old TILEDONE[group].  J segments landed in ONE flush,
                                                  //   so this wave closes JDEPTH of the tile's segments at once.
                                                  //   Bumped AFTER the flush (not per-segment) so TILEDONE can never
                                                  //   reach its target while a carrier's ACC is still unflushed --
                                                  //   that would fire the C-store on half-summed banks.
.if GROUPS > 1
    // ROBUST first-crosser election (old < target <= old+JDEPTH). WAS exact-equality (new==target), which
    //   FIRED NEVER on any TILEDONE overshoot -> the whole group's C-store was DROPPED (measured GROUPS=3:
    //   ~1 group-store lost per pipeline, growing with K). Crossing fires EXACTLY once (atomic monotonic
    //   bumps -> the unique carrier takes old from <target to >=target) and cannot be skipped. GROUPS=1 keeps
    //   the byte-identical exact-equality in the .else (single TILEDONE, no per-group reset -> hits target exactly).
    s_add_u32 s43, s66, 1                         // n_kseg = (n_kseg-1)+1
    s_mul_i32 s43, s43, ACC_N                     // target = n_kseg * ACC_N
    s_cmp_lt_u32 s51, s43                          // old < target ? (no -> a prior flush already owned it)
    s_cbranch_scc0 .Lflow_notcloser
    s_add_u32 s51, s51, JDEPTH                    // new = old + JDEPTH
    s_cmp_ge_u32 s51, s43                          // new >= target ? -> I crossed it, I own the C-store
    s_cbranch_scc1 .Lflow_cstore
.Lflow_notcloser:
.else
    s_add_u32 s51, s51, JDEPTH                    // GROUPS=1: byte-identical original order (s_add before target calc)
    s_add_u32 s43, s66, 1                         // n_kseg = (n_kseg-1)+1   (s66 = COUNT; mask would be WRONG at non-pow2 K)
    s_mul_i32 s43, s43, ACC_N                     // target = n_kseg * ACC_N
    s_cmp_eq_u32 s51, s43
    s_cbranch_scc1 .Lflow_cstore                  // I closed the TILE -> I own the C-store
.endif
.endif
    s_add_u32 s47, s47, 1
    s_cmp_ge_u32 s47, ACC_N                         // (old+1) >= ACC_N (== G at GROUPS=1) -> I completed this group
    s_cbranch_scc0 .Lflow_loop                     // not the completer -> done, loop
.Lflow_slotdone:                                 // slot's ACC_N rowblks are in -> the slot can be recycled
.if BANKZERO && !WOFLUSH
    s_branch .Lflow_drain_adv                    // BANKZERO: ONLY the tile-closer stores C (it branched to
                                                 //   .Lflow_cstore above). A mere slot-completer just drains.
.endif
    // COMPLETER (single wave -> NO race, NO redundant store, NO spinning losers): the super-tile is fully
    //   reduced. If it's the tile's LAST ksi (ksi==mask), store the G banks to C ONCE, s_wait_storecnt,
    //   THEN advance DRAIN -> the next tile's ksi=0 cannot overwrite the banks until this store is drained.
    //   s19/s30/s31 still hold mblk/tcol/ksi from this wave's own DECODE_STI (untouched by the reduce).
.if !BANKZERO                                   // BANKZERO: the C-store is TILE-scoped (below), not ksi==mask
    s_cmp_eq_u32 s31, s66                          // ksi == n_kseg-1 -> tile complete?  (s66 = COUNT, not the mask)
    s_cbranch_scc0 .Lflow_drain_adv               // not last ksi -> just advance DRAIN (no store)
.endif
.Lflow_cstore:                                   // BANKZERO entry: the tile-closer stores C
.if !WOFLUSH
    flow_gauge FDIAG_STORE_OFF, 1                 // DIAG: entering the completer C-store phase (store loop + wait)
    s_mul_i32 s38, s19, s13                        // mblk*NTL
    s_add_u32 s38, s38, s30                        // + tcol
    s_mul_i32 s38, s38, (G*FM*FN*1024)            // * per-tile C bytes
    s_add_u32 s28, s6, s38
    s_addc_u32 s29, s7, 0                          // s[28:29] = C tile base (rowblk 0)
.if GROUPS > 1
    s_mul_i32 s39, s41, (ACC_N*FM*FN*1024)          // + group * per-group C bytes -> group's first rowblk
    s_add_u32 s28, s28, s39
    s_addc_u32 s29, s29, 0
.endif
    // ---- WIDE C-STORE (2026-07-13). Was: per-DWORD ds_load_b32 + s_wait_dscnt + global_store_b32,
    //   x8 elems x FM*FN frags x ACC_N banks = 384 loads + 384 FULL LDS DRAINS + 384 stores per tile.
    //   A full s_wait_dscnt PER 4 BYTES. The coop kernel has always used global_store_b128 for this job.
    //   The 8 f32 a lane owns in a frag are CONTIGUOUS 32B on BOTH sides (LDS vaddr v12 = lane*32 + bank
    //   base, offsets +e*4; global vaddr v10 = lane*32, same +e*4) -> two b128 quads per frag, both 16B
    //   aligned (ACC_BASE/ACC_STRIDE are 1024B multiples; lane*32 is 32B aligned).
    //   => 4 loads + 1 wait + 4 stores per frag: 24 instrs -> 9  (~4.8x fewer on the widest feed path).
    //   v16..v23: free in the lean block (only the TRACE row-builder touches them, and that is a one-shot
    //   whose values die at its store -- no liveness overlap with the C-store).
    .set r, 0
    .rept ACC_N                                     // ACC_N local banks this group (== G at GROUPS=1)
      s_mov_b32 s39, (ACC_BASE + r*(FM*FN*1024))   // bank r LDS base (compile-time, local 0..ACC_N-1)
      v_add_nc_u32 v12, v10, s39                    // v12 = bank r ds vaddr (lane*32)
      .set frag, 0
      .rept FM*FN
        ds_load_b128 v[16:19], v12 offset:(frag*1024 +  0)    // lane's f32[0..3] of this frag
        ds_load_b128 v[20:23], v12 offset:(frag*1024 + 16)    // lane's f32[4..7]
        s_wait_dscnt 0x0                                       // ONE drain for the whole frag (was 8)
        global_store_b128 v10, v[16:19], s[28:29] offset:(r*(FM*FN*1024) + frag*1024 +  0) scope:SCOPE_DEV
        global_store_b128 v10, v[20:23], s[28:29] offset:(r*(FM*FN*1024) + frag*1024 + 16) scope:SCOPE_DEV
.if 0                                              // --- superseded per-dword path, kept for reference ---
        .set e, 0
        .rept 8
          ds_load_b32 v13, v12 offset:(frag*1024 + e*4)
          s_wait_dscnt 0x0
          global_store_b32 v10, v13, s[28:29] offset:(r*(FM*FN*1024) + frag*1024 + e*4) scope:SCOPE_DEV
          .set e, e+1
        .endr
.endif
        .set frag, frag+1
      .endr
      .set r, r+1
    .endr
    s_wait_storecnt 0x0                            // store COMPLETE before DRAIN++ -> banks safe to reuse
    phase_stamp s79                              // PH_CSTORE (occ[65]; host label says STAGE_WAIT)
    flow_gauge FDIAG_STORE_OFF, -1                // DIAG: C-store drained (occ[82] nets to 0)
.if DECENTASN
    // *** Codex C1: this group's banks have now been READ + the stores globally drained (s_wait_storecnt above).
    //   Publish completion so a boundary handler cannot zero_banks (reuse the banks) until this store is done.
    //   The banked completer bumped RBDONE BEFORE this store, so DRAIN==ASSIGN alone does NOT exclude this
    //   read-banks interval; GSTORED does. LAST (all the ds_load/global_store are already drained). ***
    s_mov_b32 s45, GSTORED_OFF
    lds_fetch_add_r s43, s45, 1                   // ++GSTORED (per-WG count of group C-stores fully drained)
.endif
.endif                                             // WOFLUSH: atomics already wrote C incrementally -> no store, just DRAIN++
.if BANKZERO && !WOFLUSH
    s_add_u32 s47, s47, 1                        // the tile-closer still has to settle its OWN slot
    s_cmp_ge_u32 s47, ACC_N
    s_cbranch_scc0 .Lflow_loop                   // slot not full -> just loop
.endif
.Lflow_drain_adv:
    drain_advance
    s_branch .Lflow_loop
.Lflow_cmp_tryadv:
.if DYNVGPR
    fat_dec                                       // grew-but-exhausted: --live fat gauge (STAGINSTR) before the shrink
    flow_gauge FDIAG_TASHRINK_OFF, 1              // DIAG: entering the grew-but-exhausted shrink
.Lflow_tashrink:
    s_alloc_vgpr 32                               // grew but rowblks exhausted (no claim) -> shrink back lean
    s_cbranch_scc0 .Lflow_tashrink
    duty_shrink                                   // *** PEAK END (grew-but-exhausted still held the registers) ***
    phase_stamp s83                              // PH_SHRINK: this shrink had NO stamp (2026-07-13) -- the whole
                                                  //   grew-but-exhausted round-trip leaked into PH_FOLLOW on the next
                                                  //   iteration, inflating the exact number we are trying to drive down.
    fat_release                                   // STAGGER: grew-but-exhausted -> peak over -> release the token
    flow_gauge FDIAG_TASHRINK_OFF, -1             // DIAG: shrink succeeded (occ[83] nets to 0)
.endif
    s_branch .Lflow_loop                          // the bank store + DRAIN advance are done by the COMPLETER
                                                  //   (the unique wave whose SL_RBDONE inc hit G, in .Lflow_bankdn)

// ---- FEED work: stage the STAGE_HEAD slot (A if ROLE_AFEED, B if ROLE_BFEED), then try-advance STAGE ----
.Lflow_feed:
    lds_get s44, STAGE_HEAD_OFF                 // sh
    lds_get s45, ASSIGN_HEAD_OFF               // ah
    s_cmp_ge_u32 s44, s45                        // STAGE >= ASSIGN -> nothing assigned to stage -> yield
    s_cbranch_scc1 .Lflow_feed_empty
.if MSFEED
    // *** FEED-SIDE CURSOR FIX (2026-07-14, from the Fable audit). Feeders pinned to slot_of(STAGE_HEAD),
    //   so useful stagers = FN (B-frags) + G (A-rowblks) = 7 of 30 at the real-shape config. The other
    //   ~23 coasting waves burned an lds_fetch_add_r on SL_BFNEXT/SL_ARNEXT every iteration, got an index
    //   >= FN/G, and bailed -- a pure wasted LDS atomic on a per-WG hot word. Same bug as the compute
    //   side; the pool was decorative on BOTH ends. Now: scan the assigned window [STAGE, ASSIGN),
    //   starting at (wid mod window) so feeders SPREAD across slots. ***
    s_sub_u32 s20, s45, s44                      // window = ah - sh   (1 .. POOL_N)
    s_mov_b32 s21, s24                           // wid
.Lflow_fmswa:
    s_cmp_lt_u32 s21, s20
    s_cbranch_scc1 .Lflow_fmswb
    s_sub_u32 s21, s21, s20                      // wid mod window
    s_branch .Lflow_fmswa
.Lflow_fmswb:
    s_add_u32 s44, s44, s21                      // my slot = sh + (wid mod window)   -> still < ah
.endif
    slot_of s46, s44, s47                        // slot = (my cursor) mod POOL_N
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE              // scb
    s_mul_i32 s52, s46, OPSTRIDE                 // slot * OPSTRIDE. WAS `s_lshl_b32 s52, s46, 14` = slot*16384, which is the
                                                  //   SEGK=64 stride HARDCODED. At SEGK=32 OPSTRIDE is 8192, so slot 1
                                                  //   read its operands from the WRONG address -> POOL_N>1 had NEVER
                                                  //   worked at SEGK=32, and SEGK=32 is the only size that fits LDS.
                                                  //   THIS is why POOL_N got nailed to 1. (found 2026-07-13)
    s_add_u32 s52, s52, OP_BASE                // sob
.if DECENTASN
    // v3 GATE: the slot is PRODUCED for logical index s44 only if SL_GEN==s44. Between reserve and stamp it still
    //   holds the PRIOR occupant (stale STI -> OOB DECODE). Verify BEFORE reading SL_STI. (s44 = cursor index.)
    s_add_u32 s45, s48, SL_GEN
    lds_get_r s47, s45
    s_cmp_lg_u32 s47, s44
    s_cbranch_scc1 .Lflow_feed_empty            // not produced yet -> yield (do NOT decode a stale/prior STI)
.endif
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45                           // gsti = STAMP (assigned -> set)
.if GROUPS > 1
    s_and_b32 s17, s17, STI_MASK                   // strip group bits (feed stages by tcol/ksi, group-agnostic)
.endif
    DECODE_STI
    s_cmp_eq_u32 s34, ROLE_BFEED
    s_cbranch_scc1 .Lflow_stageB
    ASTAGE_R s48, s52
    s_branch .Lflow_stage_adv
.Lflow_stageB:
    BSTAGE_R s48, s52
.Lflow_stage_adv:
    // try-advance STAGE_HEAD: if the CURRENT STAGE_HEAD slot is fully staged, CAS-bump it.
    // (Already CONDITIONAL on BFDONE==FN && ARDONE==G, so out-of-order staging under MULTISLOT is SAFE
    //  here -- unlike DRAIN, which was an unconditional bump. Under MULTISLOT several slots may become
    //  complete at once, so WALK instead of advancing a single step.)
.Lflow_stage_walk:
    lds_get s44, STAGE_HEAD_OFF
    lds_get s45, ASSIGN_HEAD_OFF
    s_cmp_ge_u32 s44, s45
    s_cbranch_scc1 .Lflow_loop
    slot_of s46, s44, s47
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE
.if DECENTASN
    // v3 GATE (THE corruption fix): STAGE < RESV no longer implies "stamped" -- a reserved-but-unstamped slot holds
    //   the PRIOR occupant's MAXED BFDONE/ARDONE, which would falsely satisfy the walk and advance STAGE over an
    //   unstamped slot. Advance only if this slot is stamped FOR STAGE (SL_GEN==STAGE).
    s_add_u32 s45, s48, SL_GEN
    lds_get_r s47, s45
    s_cmp_lg_u32 s47, s44
    s_cbranch_scc1 .Lflow_loop
    // POISON-UNTIL-STAGED gate (replaces BFDONE/ARDONE): the slot is fully staged AND ARMED iff SL_RBNEXT has NO
    //   pending bit. side_final clears the last pending bit (-> 0) only AFTER both feed sides drained their operand
    //   stores, so 'pending clear' is the release fence: STAGE must NOT advance (and compute must NOT be able to
    //   claim) while either bit is set. This is the exactly-once arm; no BFDONE/ARDONE read is needed here.
    s_add_u32 s45, s48, SL_RBNEXT
    lds_get_r s47, s45
    s_and_b32 s47, s47, RB_PENDING
    s_cmp_lg_u32 s47, 0
    s_cbranch_scc1 .Lflow_loop                    // still pending -> not armed -> do not advance STAGE
.else
    s_add_u32 s45, s48, SL_BFDONE
    lds_get_r s47, s45
    s_cmp_lt_u32 s47, FN
    s_cbranch_scc1 .Lflow_loop
    s_add_u32 s45, s48, SL_ARDONE
    lds_get_r s47, s45
    s_cmp_lt_u32 s47, G
    s_cbranch_scc1 .Lflow_loop
.endif
    cnt_inc CNT_FEED                              // <-- WAS NEVER WIRED. A completed feed stage.
    deadman_progress                              // a stage completed -> this feeder is ALIVE AND WORKING
    lds_cmpstore_adv STAGE_HEAD_OFF, s44
.if MSFEED
    s_branch .Lflow_stage_walk                    // several slots may be complete at once -> keep walking
.else
    s_branch .Lflow_loop
.endif
.Lflow_feed_empty:
.if DECENTASN
    // ===== COUPLED-CURSOR decentralized assign (2026-07-18; replaces the decoupled DA_KSI grab). Many waves
    //   reserve DIFFERENT slots in PARALLEL (CAS on ASSIGN) -> refill scales with #free waves. ksi is DERIVED
    //   from the reservation index (within = r - DA_BASE), so pool POSITION == ksi ORDER -- REQUIRED for deep-J
    //   (the carrier walks consecutive positions trusting consecutive ksi; a decoupled grab permutes them and
    //   scatters the J-window -> silent wrong-C). DA_ZDONE gates reservations so a group's banks are drain-gated
    //   + zeroed before its ksi are handed out (GROUPS>1 for free). NO over-reservation (reserve gated r <
    //   DA_ZDONE <= DA_BASE+TOTAL) -> group/tile boundaries hit exactly at ASSIGN==DA_ZDONE, handled by ONE wave
    //   (ZLOCK). A won reservation is stamped STRAIGHT-LINE, SL_GEN=r LAST as the RELEASE fence; every consumer
    //   verifies its slot by SL_GEN==head (.Lflow_feed pick, .Lflow_stage_walk, .Lflow_coast pick, drain_advance).
    //   POW2 n_kseg only. TOTAL = GROUPS*n_kseg = GROUPS<<shift. =======
    s_cmp_eq_u32 s67, s66                          // pow2 n_kseg? (mask == n_kseg-1)
    s_cbranch_scc0 .Lflow_da_terminal             //   non-pow2 -> fail SAFE (clean retire, no corruption)
    s_cmp_eq_u32 s66, 0                             // n_kseg == 1 (COUNT==0)? -> the bit-0 ZLOCK needs n_kseg>=2
    s_cbranch_scc1 .Lflow_da_terminal             //   n_kseg==1 -> fail SAFE (degenerate: no split-K to decentralize)
.if JDEPTH > 1
    // *** Codex D1 residual: a deep-J carrier's J-window (J consecutive ksi) must fit WITHIN a group of n_kseg
    //   segments. If JDEPTH does not divide n_kseg (incl. JDEPTH > n_kseg), a window straddles a group boundary
    //   and the carrier waits for the NEXT group's segments -- which the DA_ZDONE/GSTORED gate won't open until
    //   this group's C-store, which can't happen until the carrier flushes: a CIRCULAR wait -> deadman retire +
    //   incomplete C. n_kseg is RUNTIME (K/SEGK), so this is a runtime fail-safe (host must pick n_kseg % J == 0).
    s_add_u32 s46, s66, 1                           // n_kseg = COUNT + 1
    s_and_b32 s46, s46, (JDEPTH - 1)                // n_kseg mod JDEPTH  (JDEPTH pow2)
    s_cmp_lg_u32 s46, 0
    s_cbranch_scc1 .Lflow_da_terminal             //   JDEPTH does NOT divide n_kseg -> fail SAFE (clean retire, no wedge)
.endif
    lds_get s46, FLOWTERM_OFF
    s_cmp_eq_u32 s46, 0xDEAD
    s_cbranch_scc1 .Lflow_feedmt_sleep            // already terminal -> yield
    s_mov_b32 s48, 0                               // peek retry budget
.Lflow_da_peek:
    s_cmp_ge_u32 s48, 8
    s_cbranch_scc1 .Lflow_feedmt_sleep            // too contended -> bail to help (hold NOTHING; retry next loop)
    lds_get s51, DA_ZDONE_OFF                       // z (top bit ZLOCK = a wave is handling a boundary)
    s_and_b32 s47, s51, ZLOCK
    s_cmp_lg_u32 s47, 0
    s_cbranch_scc1 .Lflow_feedmt_sleep            // a wave is handling a group/tile boundary -> bail (retry next loop)
    lds_get s44, ASSIGN_HEAD_OFF                   // r = ASSIGN
    lds_get s45, DRAIN_HEAD_OFF                    // d = DRAIN
    s_cmp_ge_u32 s44, s51                           // r >= DA_ZDONE -> at a group/tile boundary (banks not zeroed past here)
    s_cbranch_scc1 .Lflow_da_boundary
    s_sub_u32 s47, s44, s45                          // r - d
    s_cmp_ge_u32 s47, POOL_N
    s_cbranch_scc1 .Lflow_feedmt_sleep            // pool full -> bail (hold nothing; cursor untouched)
    s_add_u32 s45, s44, 1                          // r+1
    lds_cas_rtn s47, ASSIGN_HEAD_OFF, s44, s45     // reserve r (r->r+1); s47 = old
    s_cmp_eq_u32 s47, s44
    s_cbranch_scc0 .Lflow_da_peek_retry            // lost the reservation -> retry peek (nothing consumed)
    // r=s44 reserved (unstamped). DA_TILE/DA_BASE are FROZEN: my slot keeps DRAIN<ASSIGN, so no tile boundary can
    //   fire (and my reserve won only because ASSIGN was unchanged since the peek -> no boundary advanced base).
    //   within = r - base in [0,TOTAL). ksi = within & mask ; group = within >> shift.
    lds_get s52, DA_TILE_OFF                         // t   (frozen)
    lds_get s51, DA_BASE_OFF                         // base (frozen)
    s_sub_u32 s51, s44, s51                          // within = r - base
    s_and_b32 s47, s51, s67                          // ksi = within & mask
    s_lshl_b32 s52, s52, s68                          // t << shift
    s_or_b32 s52, s52, s47                            // gi = (t<<shift) | ksi
.if GROUPS > 1
    s_lshr_b32 s47, s51, s68                          // group = within >> shift
    s_lshl_b32 s47, s47, STAMP_GSHIFT
    s_or_b32 s52, s52, s47                            // STAMP = (group<<GSHIFT) | gi
.endif
    s_branch .Lflow_da_stamp                          // stamp slot(r): SL_STI=s52 (STAMP), SL_GEN=r LAST (release)
.Lflow_da_peek_retry:
    s_add_u32 s48, s48, 1
    s_branch .Lflow_da_peek
.Lflow_da_boundary:
    // ASSIGN == DA_ZDONE: a group or tile boundary. Elect ONE handler: CAS(DA_ZDONE: z -> z|ZLOCK). Losers bail.
    s_or_b32 s45, s51, ZLOCK
    lds_cas_rtn s47, DA_ZDONE_OFF, s51, s45
    s_cmp_eq_u32 s47, s51
    s_cbranch_scc0 .Lflow_feedmt_sleep            // lost the boundary claim -> another wave handles it -> bail
    // I own the boundary (ZLOCK held -> reservers bail, ASSIGN frozen at z). DRAIN-GATE (I hold NO slot -> NOT
    //   self-blocking; == the coordinator's DRAIN>=ASSIGN barrier): the finished group/tile occupies [.,z) and
    //   its banks can't be reused until it fully drains. DRAIN < ASSIGN -> release ZLOCK, bail (retry later).
    lds_get s44, ASSIGN_HEAD_OFF                    // ASSIGN (== z)
    lds_get s46, DRAIN_HEAD_OFF                     // DRAIN
    s_cmp_lt_u32 s46, s44
    s_cbranch_scc1 .Lflow_da_bnd_bail              // still draining -> release ZLOCK, bail
    // *** Codex C1: also wait for the finishing group's C-store to DRAIN before reusing (zeroing) its banks.
    //   DRAIN==ASSIGN proves all RBDONE bumped, but the banked completer bumps RBDONE BEFORE it READS the banks
    //   for the C-store, so zeroing here could race that read (-> stores zeros). GSTORED counts group C-stores
    //   whose s_wait_storecnt drained; groups fully drained so far == z>>shift (each drains one C-store), so
    //   require GSTORED >= z>>shift. Non-blocking (release ZLOCK + retry; the C-store owner needs no lock). ***
    s_lshr_b32 s46, s51, s68                        // expected = z >> shift = # groups drained (one C-store each)
    lds_get s47, GSTORED_OFF
    s_cmp_lt_u32 s47, s46
    s_cbranch_scc1 .Lflow_da_bnd_bail              // a drained group's C-store not yet done -> release ZLOCK, retry
    // TILE vs GROUP: (z - base) == TOTAL -> tile exhausted ; < TOTAL -> next group of the SAME tile.
    lds_get s53, DA_BASE_OFF
    s_sub_u32 s53, s51, s53                          // z - base  (= groups_zeroed * n_kseg)
    s_lshl_b32 s46, GROUPS, s68                       // TOTAL = GROUPS << shift
    s_cmp_ge_u32 s53, s46                            // (z - base) >= TOTAL -> tile boundary
    s_cbranch_scc1 .Lflow_da_bnd_tile
    // ---- GROUP boundary: prior group drained -> zero next group's banks, advance DA_ZDONE by n_kseg (clears ZLOCK).
    zero_banks
    s_add_u32 s46, s66, 1                            // n_kseg
    s_add_u32 s45, s51, s46                          // z + n_kseg   (s51 is clean z -> top bit clears)
    lds_put DA_ZDONE_OFF, s45                        // advance (release) -> this group's ksi now reservable
    s_branch .Lflow_da_peek
.Lflow_da_bnd_tile:
    // ---- TILE boundary: claim the WG's next global tile occ[20]++, re-base, zero group 0, advance DA_ZDONE ----
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lflow_da_bnd_giok
    v_mov_b32 v3, 1
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lflow_da_bnd_giok:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s53, v5                      // t_new
    s_cmp_ge_u32 s53, s69                            // t_new >= chunkHi -> WG out of global work
    s_cbranch_scc1 .Lflow_da_bnd_term              // terminal: release ZLOCK, broadcast retire
    zero_banks                                       // group 0 banks fresh for the new tile (pool drained above)
    lds_put DA_TILE_OFF, s53                          // publish t_new
    lds_put DA_BASE_OFF, s51                          // base = z   (release-ordered BEFORE the DA_ZDONE advance below)
    s_add_u32 s46, s66, 1                            // n_kseg
    s_add_u32 s45, s51, s46                          // z + n_kseg
    lds_put DA_ZDONE_OFF, s45                        // advance (clears ZLOCK) -> group 0 of t_new now reservable
    s_branch .Lflow_da_peek
.Lflow_da_bnd_bail:
    cnt_inc REL_IMBAL                                 // *** occ[97]: boundary drain-gate bails (finished group/tile still draining) ***
    lds_put DA_ZDONE_OFF, s51                        // release ZLOCK (restore clean z); retry next loop
    s_branch .Lflow_feedmt_sleep
.Lflow_da_bnd_term:
    lds_put DA_ZDONE_OFF, s51                        // release ZLOCK (restore clean z); go terminal
    s_branch .Lflow_da_terminal
.Lflow_da_rollback:
    // transient bail: un-reserve slot r (CAS ASSIGN_HEAD: r+1 -> r). If lost, publish a pre-completed sentinel
    //   at slot r so no consumer wedges. NON-terminal (retry next loop). Cursor is left untouched -> no work lost.
    s_add_u32 s46, s44, 1
    lds_cas_rtn s47, ASSIGN_HEAD_OFF, s46, s44
    s_cmp_eq_u32 s47, s46
    s_cbranch_scc1 .Lflow_feedmt_sleep               // rolled back cleanly -> bail, retry next loop iter
    slot_of s46, s44, s47                            // rollback lost -> pre-completed sentinel at slot(r)
    s_lshl_b32 s46, s46, 5
    s_add_u32 s46, s46, SLOTC_BASE
    s_add_u32 s45, s46, SL_RBNEXT
    lds_put_r s45, ACC_N
    s_add_u32 s45, s46, SL_RBDONE
    lds_put_r s45, ACC_N
    s_add_u32 s45, s46, SL_BFNEXT
    lds_put_r s45, FN
    s_add_u32 s45, s46, SL_BFDONE
    lds_put_r s45, FN
    s_add_u32 s45, s46, SL_ARNEXT
    lds_put_r s45, G
    s_add_u32 s45, s46, SL_ARDONE
    lds_put_r s45, G
    s_add_u32 s45, s46, SL_STI
    lds_put_r s45, 0
    s_add_u32 s45, s46, SL_GEN
    lds_put_r s45, s44                               // SL_GEN=r LAST -> release. Walk passes, pickers bail.
    s_branch .Lflow_feedmt_sleep                      // sentinel published -> bail (retry), NOT terminal
.Lflow_da_stamp:
    cnt_inc CLAIM_NOPERSIST                          // *** INSTRUMENT (occ[96], repurposed): count REAL super-tile EMISSIONS (expect==TOTAL_super) ***
    // ---- STAMP slot(r mod POOL_N) NORMALLY: reset counters, SL_STI=gi, then SL_GEN=r LAST (release fence). ----
    slot_of s46, s44, s47                          // slot = r mod POOL_N ; s47 scratch
    s_lshl_b32 s46, s46, 5
    s_add_u32 s46, s46, SLOTC_BASE                 // scb
    s_add_u32 s45, s46, SL_RBNEXT
    // POISON-UNTIL-STAGED for EVERY slot (lead AND non-lead), regardless of JDEPTH: side_final arms it to 0 once
    //   BOTH feed sides stage. The OLD J>1 path stamped non-lead = ACC_N (bare, no pending) -> it read as
    //   "staged, exhausted" BEFORE the operands were fed, colliding with poison-until-staged (guard 786). Under
    //   the coupled cursor a non-lead slot is turned away from CLAIMING by the pre-grow lead-gate AND the post-grow
    //   lead RE-CHECK (see .Lflow_leadok / the DECENTASN&&J>1 recheck) -- NOT by an ACC_N poison -- so it can arm
    //   to 0 like a lead; the carrier consumes it via the cursor walk (bumping SL_RBDONE, never SL_RBNEXT).
    lds_put_r s45, RB_PENDING
    s_add_u32 s45, s46, SL_RBDONE
    lds_put_r s45, 0
    // *** Codex gpt-5.6-sol (2026-07-18): PUBLISH SL_STI BEFORE the feed claim-counter resets. The feed claims a
    //   frag via fetch_add(SL_BFNEXT/SL_ARNEXT) then decodes SL_STI (SITE-J, ~1428/1503) to address operands, but
    //   does NOT re-check SL_GEN. If the counters are reset (claimable) BEFORE STI is published, a feeder delayed
    //   across a g->g+POOL_N slot reuse claims the NEW gen's reset counter yet reads the OLD gen's STI -> stages
    //   stale-K operands into the new gen's buffer, and compute adds the wrong K-segment (work-EXACT, wrong value,
    //   ~2.3x double-count on silicon; coordinator immune because it publishes ASSIGN_HEAD++ last). Making STI
    //   land before BFNEXT/ARNEXT are reset means "counter is claimable" implies "STI is the new gen's". ***
    s_add_u32 s45, s46, SL_STI
    lds_put_r s45, s52                             // SL_STI = gi (payload) -- MUST precede the feed-counter resets below
    s_add_u32 s45, s46, SL_BFNEXT
    lds_put_r s45, 0
    s_add_u32 s45, s46, SL_BFDONE
    lds_put_r s45, 0
    s_add_u32 s45, s46, SL_ARNEXT
    lds_put_r s45, 0
    s_add_u32 s45, s46, SL_ARDONE
    lds_put_r s45, 0
    s_add_u32 s45, s46, SL_GEN
    lds_put_r s45, s44                             // SL_GEN=r LAST -> release fence. (Do NOT batch these into one
                                                   //   trailing wait -- the per-store ordering IS the fence.)
    s_branch .Lflow_loop
.Lflow_da_termslot:
    // gi>=bound: try to ROLL BACK the reservation (CAS RESV: r+1 -> r). Wins unless a later wave already reserved r+1.
    s_add_u32 s46, s44, 1                          // r+1
    lds_cas_rtn s47, ASSIGN_HEAD_OFF, s46, s44     // if RESV==r+1 -> RESV=r ; s47=old
    s_cmp_eq_u32 s47, s46
    s_cbranch_scc1 .Lflow_da_terminal             // rolled back -> no slot consumed, no sentinel needed
    // rollback lost -> publish a PRE-COMPLETED sentinel at slot(r): all NEXT/DONE pre-maxed + safe STI=0, so the
    //   STAGE-walk passes it, feed/coast/compute pickers all BAIL, drain_advance passes -> no wedge, no OOB.
    slot_of s46, s44, s47
    s_lshl_b32 s46, s46, 5
    s_add_u32 s46, s46, SLOTC_BASE                 // scb
    s_add_u32 s45, s46, SL_RBNEXT
    lds_put_r s45, ACC_N                           // RBNEXT=ACC_N -> compute pick fetch_add >=ACC_N -> tryadv (bail)
    s_add_u32 s45, s46, SL_RBDONE
    lds_put_r s45, ACC_N                           // RBDONE=ACC_N -> drain_advance passes this slot
    s_add_u32 s45, s46, SL_BFNEXT
    lds_put_r s45, FN
    s_add_u32 s45, s46, SL_BFDONE
    lds_put_r s45, FN                              // BFDONE=FN -> STAGE-walk B-side satisfied
    s_add_u32 s45, s46, SL_ARNEXT
    lds_put_r s45, G
    s_add_u32 s45, s46, SL_ARDONE
    lds_put_r s45, G                               // ARDONE=G -> STAGE-walk A-side satisfied
    s_add_u32 s45, s46, SL_STI
    lds_put_r s45, 0                               // SL_STI=0 -> a stray pick decodes t=0 (in-bounds, never OOB)
    s_add_u32 s45, s46, SL_GEN
    lds_put_r s45, s44                             // SL_GEN=r LAST -> release. Walk passes, pickers bail.
    // fall through to terminal
.Lflow_da_terminal:
    lds_put FLOWTERM_OFF, 0xDEAD                    // stop NEW claims
.Lflow_da_drain:
    drain_advance                                  // WALK DRAIN past completed/sentinel slots: the pre-completed
                                                   //   sentinel has RBDONE==ACC_N but NO computer, so only this walk
                                                   //   (SL_GEN==DRAIN, RBDONE==ACC_N) advances DRAIN past it.
    lds_get s44, ASSIGN_HEAD_OFF                   // RESV: EXACT count of reserved slots (each drains -> DRAIN reaches it,
    lds_get s45, DRAIN_HEAD_OFF                     //   incl. the pre-completed sentinel which drains instantly)
    s_cmp_lt_u32 s45, s44
    s_cbranch_scc0 .Lflow_da_alldrained            // DRAIN >= RESV -> all reserved slots drained
    deadman_check
    s_sleep 1
    s_branch .Lflow_da_drain
.Lflow_da_alldrained:
    .set daw, 0
    .rept WAVES
      lds_put (ROLE_BASE + daw*4), ROLE_RETIRE     // broadcast retire (idempotent if several waves hit terminal)
      .set daw, daw+1
    .endr
    s_branch .Lflow_retire
.Lflow_feedmt_sleep:
.endif
    cnt_inc CNT_FEEDMT                            // feed wave ran, nothing assigned to stage
.if STAGGER
    // ---- BATON WAKE (A): a shrinking neighbor poked me "grow now" -- skip the yield and loop straight back
    //   to retry the compute/grow so I rise into the just-freed budget instead of napping. THIS is what keeps
    //   >=1 wave at peak (fills the valley the s_sleep would open). Read own mailbox only. NOT a gate/cap: an
    //   un-poked wave yields as normal, and a woken wave still only grows if the physics allow (grow-fail->coast).
    s_lshl_b32 s92, s24, 2
    s_add_u32  s92, s92, GROWPERMIT_BASE          // &NOTIFY[wid]
    lds_get_r  s93, s92
    s_cmp_eq_u32 s93, 0
    s_cbranch_scc1 .Lflow_feedmt_yield            // not poked -> yield as normal
    s_mov_b32  s93, 0
    lds_put_r  s92, s93                            // consume the poke
    s_branch   .Lflow_loop                         // poked -> skip the yield, loop back and grow NOW
.Lflow_feedmt_yield:
.endif
    s_sleep SLEEPN                                // lean feed can't compute without a grow -> yield; coordinator rebalances
    s_branch .Lflow_loop

// ---- COAST: a fat COMPUTE wave with no staged work runs feed code (FREE, no resize) to help staging ----
.Lflow_growfail:
    fat_release                                     // STAGGER: acquired a token but the grow was refused -> return it
    cnt_inc CNT_GROWFAIL                            // diag: per-burst grow failed (budget full) -> coast = stagger repulsion
    // fall through: the failed grow allocated nothing, so we are still lean and safe to coast
.Lflow_coast:
    cnt_inc CNT_COAST                              // diag: a wave coasted (no staged work, or a grow-fail)
    lds_get s44, STAGE_HEAD_OFF
    lds_get s45, ASSIGN_HEAD_OFF
    s_cmp_ge_u32 s44, s45
    s_cbranch_scc1 .Lflow_feed_empty              // nothing assigned to stage -> yield
.if MSFEED
    // *** FEED-SIDE CURSOR FIX (2026-07-14, from the Fable audit). Feeders pinned to slot_of(STAGE_HEAD),
    //   so useful stagers = FN (B-frags) + G (A-rowblks) = 7 of 30 at the real-shape config. The other
    //   ~23 coasting waves burned an lds_fetch_add_r on SL_BFNEXT/SL_ARNEXT every iteration, got an index
    //   >= FN/G, and bailed -- a pure wasted LDS atomic on a per-WG hot word. Same bug as the compute
    //   side; the pool was decorative on BOTH ends. Now: scan the assigned window [STAGE, ASSIGN),
    //   starting at (wid mod window) so feeders SPREAD across slots. ***
    s_sub_u32 s20, s45, s44                      // window = ah - sh   (1 .. POOL_N)
    s_mov_b32 s21, s24                           // wid
.Lflow_cmswa:
    s_cmp_lt_u32 s21, s20
    s_cbranch_scc1 .Lflow_cmswb
    s_sub_u32 s21, s21, s20                      // wid mod window
    s_branch .Lflow_cmswa
.Lflow_cmswb:
    s_add_u32 s44, s44, s21                      // my slot = sh + (wid mod window)   -> still < ah
.endif
    slot_of s46, s44, s47
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE
    s_mul_i32 s52, s46, OPSTRIDE                 // slot * OPSTRIDE. WAS `s_lshl_b32 s52, s46, 14` = slot*16384, which is the
                                                  //   SEGK=64 stride HARDCODED. At SEGK=32 OPSTRIDE is 8192, so slot 1
                                                  //   read its operands from the WRONG address -> POOL_N>1 had NEVER
                                                  //   worked at SEGK=32, and SEGK=32 is the only size that fits LDS.
                                                  //   THIS is why POOL_N got nailed to 1. (found 2026-07-13)
    s_add_u32 s52, s52, OP_BASE
.if DECENTASN
    // v3 GATE: same as the feed pick -- verify SL_GEN==cursor before touching this slot's DONE/STI (an
    //   unstamped reserved slot holds the prior occupant -> stale STI -> OOB DECODE).
    s_add_u32 s45, s48, SL_GEN
    lds_get_r s47, s45
    s_cmp_lg_u32 s47, s44
    s_cbranch_scc1 .Lflow_feed_empty
.endif
    s_add_u32 s45, s48, SL_BFDONE
    lds_get_r s47, s45
    s_cmp_lt_u32 s47, FN
    s_cbranch_scc1 .Lflow_coastB                  // B behind -> help B
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
.if GROUPS > 1
    s_and_b32 s17, s17, STI_MASK
.endif
    DECODE_STI
    ASTAGE_R s48, s52
    s_branch .Lflow_stage_adv
.Lflow_coastB:
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
.if GROUPS > 1
    s_and_b32 s17, s17, STI_MASK
.endif
    DECODE_STI
    BSTAGE_R s48, s52
    s_branch .Lflow_stage_adv

.Lflow_retire:
.if STAGGER
    // *** LEAK FIX (2026-07-14): a wave force-retired out of .Lflow_jwait (deadman) or exiting any other
    //   way while FAT still HELD ITS TOKEN. FATTOK lives in per-WG LDS and is never reset, so each leak
    //   permanently burned one of the MAXFAT fat slots -> FATTOK saturates -> NOBODY can go fat -> the
    //   workgroup wedges. fat_release was wired at all FOUR normal exits and NOT at the abnormal one. ***
    s_cmp_eq_u32 s[FATHELD], 1
    s_cbranch_scc0 .Lflow_notok
    cnt_inc CNT_TOKLEAK                          // count it: a nonzero occ[92] means we leaked (and now recovered)
    fat_release
.Lflow_notok:
.endif
    cnt_flush                                  // STAGINSTR: SINGLE emit of this wave's counter totals (ACC dead, wave lean)
    phase_flush                                // PHASEPROBE: single emit of s78..s83 -> occ[64..69]
    // live-- : lane0 occ[0] -= 1 (harness completion gate)
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lflow_dead
    v_mov_b32 v3, -1
    global_atomic_add_u32 v4, v3, s[0:1] scope:SCOPE_DEV
.Lflow_dead:
    s_mov_b32 exec_lo, s16
    tfspan max, 12                             // TFPROBE: wall-span end
    alllive_dec
.if RETBARRIER
    // COUNT-TO-WAVES collective exit: check in, then all s_endpgm TOGETHER once the WG's count hits WAVES.
    //   This coordinated exit is what the EOP needs to register a clean dispatch completion -> fence FIRES
    //   (the coordinator's staggered RETIRE broadcast fires the fence at 8 waves but not 16). Bounded wait
    //   (plain counter + s_sleep; NO RTC/message bus, NO s_alloc) -> can never hang the wave.
    lds_inc QUIESCE_CNT_OFF                     // this wave checked in at the exit
    s_mov_b32 s52, 0
.Lflow_retbar:
    lds_get s53, QUIESCE_CNT_OFF
    s_cmp_ge_u32 s53, WAVES                     // all WAVES waves in -> exit together
    s_cbranch_scc1 .Lflow_endpgm
    s_add_u32 s52, s52, 1
    s_cmp_ge_u32 s52, RETBAR_MAX                // hard iteration backstop -> exit anyway (never hang)
    s_cbranch_scc1 .Lflow_endpgm
.if DEADMAN
    // WALL-TIME BOUND (fix 2026-07-10): the RETBAR_MAX *iteration* bound stretches to ~18s of resident spin
    //   under compositor contention (waves get scheduled slowly), which starves the gfx ring -> safemode. Bound
    //   total ALIVE time to the deadman deadline instead -- but THROTTLED to every DEADMAN_EVERY iters so the
    //   s_sendmsg_rtn msg-bus read cannot itself spam the SQ front end (the 2026-07-05 brick class; see the
    //   deadman_check note at :544). s70 = this wave's start RTC (deadman_stamp). Assumes DEADMAN_EVERY is pow2.
    s_and_b32 s53, s52, (DEADMAN_EVERY-1)
    s_cbranch_scc1 .Lflow_rb_sleep              // not a DEADMAN_EVERY-th iter -> skip the RTC read
    s_sendmsg_rtn_b64 s[62:63], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_sub_u32 s62, s62, s70                      // elapsed alive (u32 wrap-safe; deadline << 42s)
    s_cmp_ge_u32 s62, DEADMAN_TICKS              // resident past the deadman deadline -> drain out NOW (never starve)
    s_cbranch_scc1 .Lflow_endpgm
.Lflow_rb_sleep:
.endif
    s_sleep SLEEPN
    s_branch .Lflow_retbar
.Lflow_endpgm:
.endif
    s_endpgm
.else
    s_endpgm                                   // DSWS2=0 has no v2 body (this file is always built DSWS2=1)
.endif
    .size occ_kernel, .-occ_kernel

// ---- RGADESC: analysis-only descriptor so `rga -s bin --co` can enumerate + livereg this kernel.
//   vgpr 256 ceiling so livereg reports the true s_alloc-grown peak-live. NOT emitted for the PM4 .bin. ----
.if RGADESC
.amdhsa_kernel occ_kernel
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 106                  // body uses up to s71; STAGINSTR counters live in s84..s89 (cnt_* macros).
                                               //   (RGA-analysis descriptor only -- the PM4 host builds COMPUTE_PGM_RSRC1
                                               //   itself, occ_dispatch.cpp:216, and GFX10+ does not wave-allocate SGPRs.
                                               //   Kept accurate anyway so RGA/static tools report the real high-water.)
    .amdhsa_group_segment_fixed_size 65536     // FIX 1a ring: D=2 needs 33024B (RGA-analysis descriptor only)
    .amdhsa_user_sgpr_count 15                 // FIX 1(g): v2 contract now s0..s14 only (n_kseg/TOTAL_super/
                                                //   magic_kseg dropped -- derived in-kernel / memory-carried)
    .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            occ_kernel
    .symbol:          occ_kernel.kd
    .kernarg_segment_size: 60
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .wavefront_size:  32
    .sgpr_count:      72
    .vgpr_count:      256
    .max_flat_workgroup_size: 256
    .args:            []
.end_amdgpu_metadata
.endif
