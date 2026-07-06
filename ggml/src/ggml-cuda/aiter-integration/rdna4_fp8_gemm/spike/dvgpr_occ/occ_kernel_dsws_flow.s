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
.set SNAP_BASE,      (INITFLAG_OFF + 4)         // u32[6]: [parity*3 + {0:nC,1:nA,2:nB}] role-mix snapshots
.set QUIESCE_CNT_OFF,(SNAP_BASE + 6*4)          // u32 role-agnostic bail counter (LDS; DSWS2_GQUIESCE=0)
// SENSOR FIX: the claimer publishes its MID-DRAIN ring-occupancy PEAK here each super-tile; the conversion
//   decisions read THESE instead of sampling occ_sample at their own quiesce (where occ_X reads ~0 post-drain
//   -> always "starved" -> the 4/2/2->1/6/1 compute->feed runaway). Mid-drain peak = the true demand signal.
.set OCCA_PUB_OFF,   (QUIESCE_CNT_OFF + 4)      // claimer-published occ_A peak
.set OCCB_PUB_OFF,   (OCCA_PUB_OFF + 4)         // claimer-published occ_B peak
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
.if LDS_TOTAL_DSWS2 > 32768
  .error "DSWS2 LDS layout exceeds 32768B group segment"
.endif
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
  .set DEADMAN_TICKS, 50000000   // 0.5s @ 100MHz RTC; a normal chunk is ~ms, so this fires ONLY on a wedge
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
// ---- per-slot control block:  SLOTC_BASE + slot*SLOTC_STRIDE + field  (same fields as the ring) ----
// coordinator-local TILE-CLAIM state (single writer = wid0), tucked in the reserved 32-wave mailbox
//   tail (safe while WAVES<=30). occ[20] claims whole TILES; the coordinator emits a tile's n_kseg
//   super-tiles (sti=(t<<shift)|ksi) into its OWN WG's slots so per-WG LDS banks accumulate a full tile.
.set COORD_KSI_OFF, (ROLE_BASE + 32*4 - 8)      // = 140: next ksi to emit for the current tile (init sentinel)
.set COORD_T_OFF,   (ROLE_BASE + 32*4 - 4)      // = 144: current tile index t
//   (safe while WAVES<=30: the real mailbox uses only ROLE_BASE..ROLE_BASE+WAVES*4; WAVES=8 here. WAVES is
//    .set later so it can't be asserted at this point -- these live in the 32-wave reservation's tail.)
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
// ---- per-slot operand buffers:  OP_BASE + slot*OPSTRIDE ;  BRES at +BRES_ROFF, ARES at +ARES_ROFF ----
.set OP_BASE,       256                          // 256B-aligned; below it: frontier + mailbox + POOL_N ctrl blocks
.set OPSTRIDE,      (BRES_BYTES + ARES_BYTES)    // 4096 + 12288 = 16384 per slot
.set BRES_ROFF,     0                            // resident B within a slot
.set ARES_ROFF,     BRES_BYTES                   // resident A within a slot (after B)
// ---- per-rowblk reduction accumulator pool: ACC_BASE + bank*ACC_STRIDE (bank in [0,ACC_N)) ----
//   fp32, rowblk-lifetime (persists across all n_kseg K-segments of a rowblk); DISTINCT from the
//   segment-lifetime operand pool above. One bank = one C-rowblk = FM*FN frags x 1024B.
.set ACC_BASE,      (OP_BASE + POOL_N*OPSTRIDE)  // after the operand pool
.set ACC_STRIDE,    (FM*FN*1024)                 // = 8192 @ FM=2 FN=4 (one C-rowblk)
.set LDS_TOTAL_FLOW,(ACC_BASE + ACC_N*ACC_STRIDE) // POOL3/ACC1: 57600 ; POOL2/ACC2: 49408
.if LDS_TOTAL_FLOW > 65536
  .error "FLOW LDS layout exceeds 65536B group segment (hardware WGP limit) -- lower POOL_N or ACC_N"
.endif
.if (SLOTC_BASE + POOL_N*SLOTC_STRIDE) > OP_BASE
  .error "FLOW per-slot control blocks overlap the operand region (raise OP_BASE)"
.endif
// Phase-B state must fit in the control gap below the resident region (inert compile check, no bytes).
.if DSWS2_STATE_END > BRES_OFF
  .error "DSWS2 Phase-B state (SNAP_BASE/QUIESCE_CNT) overlaps resident B region (BRES_OFF)"
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
.elseif POOL_N == 2
    s_and_b32 \dst, \head, 1
.elseif POOL_N == 4
    s_and_b32 \dst, \head, 3
.elseif POOL_N == 3
    s_mul_hi_u32 \dst, \head, 0xAAAAAAAB       // q ~ head/3  (magic-div; q = mulhi>>1)
    s_lshr_b32 \dst, \dst, 1
    s_mul_i32 \scr, \dst, 3
    s_sub_u32 \dst, \head, \scr                 // slot = head - 3*q
.else
    .error "slot_of: POOL_N must be in {2,3,4}"
.endif
.endm

// acc_base_of: \dst = LDS byte address of accumulator bank \bank (bank in [0,ACC_N)) = ACC_BASE + bank*ACC_STRIDE.
//   ACC_STRIDE is a compile-time constant, so s_mul_i32 is exact for any FM*FN (no pow2 assumption).
.macro acc_base_of dst, bank
    s_mul_i32 \dst, \bank, ACC_STRIDE
    s_add_u32 \dst, \dst, ACC_BASE
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
// ---- DEADMAN watchdog: s70 = this wave's start RTC (low 32b); s71 = throttle counter (repurposed high-RTC
//   reg, which is unused at TRACE=0 -- deadman_check only reads s70). The message-bus RTC read (s_sendmsg_rtn)
//   is an SQ-front-end op; hundreds of idle COAST waves hitting it EVERY loop iteration spam the front-end,
//   starving the compositor's SQC(inst) fetch (2026-07-05 MODE1 brick) AND destabilizing the coast wall
//   (identical STAGINSTR work measured 0.32s vs 2.0s). THROTTLE: only read the RTC every DEADMAN_EVERY iters. ----
.ifndef DEADMAN_EVERY
  .set DEADMAN_EVERY, 64          // message-bus RTC-read cadence (in loop iters); force-retire slack = DEADMAN_EVERY iters
.endif
.macro deadman_stamp                          // stamp start RTC (low 32b in s70) once at entry
.if DEADMAN
    s_sendmsg_rtn_b64 s[70:71], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s71, 0                           // repurpose the (TRACE=0-unused) high-RTC reg as the throttle counter
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
    s_sub_u32 s62, s62, s70                     // elapsed = now_lo - start_lo  (u32 wrap-safe; deadline << 42s)
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
.macro BSTAGE                                 // in: s30=tcol s31=ksi ; clob: s20,s21,s23,s25,s26,s27,v13,v[BSTG..]
    s_mul_i32  s20, s30, s14                  // tcol * FN*256
    s_mul_i32  s21, s31, KSEG_STEPS           // ksi * KSEG_STEPS
    s_mul_i32  s21, s21, s10                  // * NT*256  -> segment k-start byte offset
    s_add_u32  s20, s20, s21
    s_add_u32  s20, s4, s20
    s_addc_u32 s21, s5, 0                      // s[20:21] = B base (tcol,ksi, seg k-step 0)
.Lbcl\@:
    lds_fetch_add s23, BFRAG_NEXT_OFF, 1       // claim frag f
    s_cmp_ge_u32 s23, FN
    s_cbranch_scc1 .Lbsd\@                      // f>=FN -> all frags claimed
    s_lshl_b32 s25, s23, 8                      // f*256
    s_add_u32  s26, s20, s25
    s_addc_u32 s27, s21, 0                      // s[26:27] = frag f base (seg k0)
    v_add_nc_u32 v13, v9, BRES_OFF
    v_add_nc_u32 v13, v13, s25                  // resident B dst vbase for frag f
    .set ks, 0
    .rept KSEG_STEPS
      global_load_tr_b64 v[BSTG+ks*2:BSTG+ks*2+1], v9, s[26:27]
      s_add_u32  s26, s26, s10                  // next k-step (last iter over-advances; unused)
      s_addc_u32 s27, s27, 0
      .set ks, ks+1
    .endr
    s_wait_loadcnt 0x0
    .set ks, 0
    .rept KSEG_STEPS
      ds_store_b64 v13, v[BSTG+ks*2:BSTG+ks*2+1] offset:(ks*FN*256)
      .set ks, ks+1
    .endr
    s_wait_dscnt 0x0
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
    lds_fetch_add s23, AROW_NEXT_OFF, 1         // claim rowblk r
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
    .set mi, 0
    .rept FM
      .if mi == 0
        s_mov_b32 s44, s40
        s_mov_b32 s45, s41
      .else
        s_add_u32  s44, s44, s32                    // += 16*K (next M-frag)
        s_addc_u32 s45, s45, 0
      .endif
      .set ks, 0
      .rept KSEG_STEPS
        global_load_b64 v[BSTG:BSTG+1], v8, s[44:45] offset:(ks*16)
        s_wait_loadcnt 0x0
        ds_store_b64 v13, v[BSTG:BSTG+1] offset:((ks*G*FM + mi)*256)
        s_wait_dscnt 0x0
        .set ks, ks+1
      .endr
      .set mi, mi+1
    .endr
    lds_inc AROW_DONE_OFF                           // rowblk r fully STAGED -> publish completion
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
    s_mul_i32  s20, s30, s14                  // tcol * FN*256
    s_mul_i32  s21, s31, KSEG_STEPS           // ksi * KSEG_STEPS
    s_mul_i32  s21, s21, s10                  // * NT*256  -> segment k-start byte offset
    s_add_u32  s20, s20, s21
    s_add_u32  s20, s4, s20
    s_addc_u32 s21, s5, 0                      // s[20:21] = B base (tcol,ksi, seg k-step 0)
    s_add_u32  s46, \scb, SL_BFNEXT            // &SL_BFNEXT[slot]
.Lbclr\@:
    lds_fetch_add_r s23, s46, 1                // claim frag f
    s_cmp_ge_u32 s23, FN
    s_cbranch_scc1 .Lbsdr\@                     // f>=FN -> all frags claimed
    s_lshl_b32 s25, s23, 8                      // f*256
    s_add_u32  s26, s20, s25
    s_addc_u32 s27, s21, 0                      // s[26:27] = frag f base (seg k0)
    v_add_nc_u32 v13, v9, \sob                  // + slot operand base
    v_add_nc_u32 v13, v13, s25                  // + f*256   (BRES_ROFF = 0)
    .set ks, 0
    .rept KSEG_STEPS
      global_load_tr_b64 v[BSTG+ks*2:BSTG+ks*2+1], v9, s[26:27]
      s_add_u32  s26, s26, s10
      s_addc_u32 s27, s27, 0
      .set ks, ks+1
    .endr
    s_wait_loadcnt 0x0
    .set ks, 0
    .rept KSEG_STEPS
      ds_store_b64 v13, v[BSTG+ks*2:BSTG+ks*2+1] offset:(ks*FN*256)
      .set ks, ks+1
    .endr
    s_wait_dscnt 0x0
    s_add_u32  s47, \scb, SL_BFDONE
    lds_inc_r s47                               // frag f STORED -> compute gates on SL_BFDONE==FN
    s_branch .Lbclr\@
.Lbsdr\@:
.endm

.macro ASTAGE_R scb, sob             // in: s19=mblk s31=ksi ; clob: s22,s23,s25,s32,s36,s40,s41,s44,s45,s46,s47,v13,v[BSTG..]
    s_lshl_b32 s32, s9, 4                       // rowstride16 = 16*K
    s_add_u32  s46, \scb, SL_ARNEXT             // &SL_ARNEXT[slot]
.Laclr\@:
    lds_fetch_add_r s23, s46, 1                 // claim rowblk r
    s_cmp_ge_u32 s23, G
    s_cbranch_scc1 .Lasdr\@
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
    .set mi, 0
    .rept FM
      .if mi == 0
        s_mov_b32 s44, s40
        s_mov_b32 s45, s41
      .else
        s_add_u32  s44, s44, s32                    // += 16*K (next M-frag)
        s_addc_u32 s45, s45, 0
      .endif
      .set ks, 0
      .rept KSEG_STEPS
        global_load_b64 v[BSTG:BSTG+1], v8, s[44:45] offset:(ks*16)
        s_wait_loadcnt 0x0
        ds_store_b64 v13, v[BSTG:BSTG+1] offset:((ks*G*FM + mi)*256)
        s_wait_dscnt 0x0
        .set ks, ks+1
      .endr
      .set mi, mi+1
    .endr
    s_add_u32  s47, \scb, SL_ARDONE
    lds_inc_r s47                                   // rowblk r STAGED -> compute gates on SL_ARDONE==G
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
    s_mov_b32 s66, CONV_COOLDOWN                //     Task 4: committed conversion -> arm cooldown
.endif
    // ---- s_alloc_vgpr resize: THE pre-grow OOR window closes here; all reads above were <=v15 ----
.Lca_alloc\@:
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
// Per-wave phase accumulators live in SGPRs s78..s83 (NO per-stamp store -> zero memory perturbation, no
//   s_wait_storecnt pollution). s77 = last-stamp RTC. phase_flush emits them ONCE at compute retire.
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
    s_lshr_b32    s66, s8, NKSEG_SHIFT        // n_kseg = KT >> NKSEG_SHIFT   (KT=s8)
    s_ff1_i32_b32 s68, s66                    // shift  = log2(n_kseg) (bit index of the single set bit; n_kseg=1 -> 0)
    s_sub_u32     s67, s66, 1                 // mask   = n_kseg - 1
    // ---- identity (lifted from coop prologue; v0=tid hardware-preloaded) ----
    v_lshrrev_b32 v1, 5, v0                  // wid  = tid >> 5
    v_and_b32     v2, 31, v0                 // lane = tid & 31
    v_and_b32     v6, 15, v0                 // lane & 15 (A vaddr)
    v_mov_b32     v4, 0
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
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lflow_wait_init            // non-coordinator waits for LDS init
    // ---- coordinator (wid0) barrier-free LDS init ----
    lds_put ASSIGN_HEAD_OFF, 0
    lds_put STAGE_HEAD_OFF, 0
    lds_put DRAIN_HEAD_OFF, 0
    lds_put FLOWTERM_OFF, 0
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
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_GEN),    0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_RBNEXT), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_RBDONE), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_BFNEXT), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_BFDONE), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_ARNEXT), 0
      lds_put (SLOTC_BASE + sl*SLOTC_STRIDE + SL_ARDONE), 0
      .set sl, sl+1
    .endr
    lds_put COORD_KSI_OFF, 0xFFFFFFFF          // tile-claim sentinel: first ASSIGN claims a fresh tile
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
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lflow_body                 // non-coordinator -> straight to role work
    // ---- coordinator duty (wid0): ASSIGN + (later) sense/nudge ----
    lds_get s44, FLOWTERM_OFF
    s_cmp_eq_u32 s44, 0xDEAD
    s_cbranch_scc1 .Lflow_drainwait            // already terminal -> wait for drain
    lds_get s44, ASSIGN_HEAD_OFF               // ah
    lds_get s45, DRAIN_HEAD_OFF               // dh
    s_sub_u32 s46, s44, s45
    s_cmp_ge_u32 s46, POOL_N
    s_cbranch_scc1 .Lflow_coord_period         // pool full -> no assign this cycle
    // TILE-CLAIM: write-once needs a WG to own a whole tile's n_kseg segments so its LDS banks sum a
    //   full tile. occ[20] now counts TILES; emit n_kseg super-tiles sti=(t<<shift)|ksi per claimed tile.
    lds_get s55, COORD_KSI_OFF                   // next ksi to emit for the current tile
    s_cmp_le_u32 s55, s67                        // ksi <= mask (=n_kseg-1) -> continue current tile
    s_cbranch_scc1 .Lflow_same_tile
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
    lds_get s56, COORD_T_OFF                     // reuse current tile
.Lflow_form_sti:
    s_lshl_b32 s17, s56, s68                     // t << shift
    s_or_b32 s17, s17, s55                       // sti = (t<<shift) | ksi
    s_add_u32 s55, s55, 1
    lds_put COORD_KSI_OFF, s55                   // advance ksi cursor for next assign
    // assign sti (s17) to slot(ah): reset counters, STAMP=sti, then ASSIGN_HEAD++ (release LAST)
    slot_of s46, s44, s47                        // slot = ah mod POOL_N
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE
    s_add_u32 s45, s48, SL_RBNEXT
    lds_put_r s45, 0
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
    s_cbranch_scc1 .Lflow_coast
    slot_of s45, s46, s47                        // slot = dh mod N
    s_lshl_b32 s48, s45, 5
    s_add_u32 s48, s48, SLOTC_BASE              // scb
    s_lshl_b32 s52, s45, 14
    s_add_u32 s52, s52, OP_BASE                // sob
    // PER-BURST GROW: trapezoid peak starts here (fat through WMMA+ds_add, lean otherwise). COAST-ON-FAIL
    //   is the floodgate: if the SIMD VGPR budget is full, grow SCC0 -> coast lean, committing NO claim.
.if DYNVGPR
    s_alloc_vgpr NFV
    s_cbranch_scc0 .Lflow_growfail
.endif
    s_add_u32 s45, s48, SL_RBNEXT
    lds_fetch_add_r s33, s45, 1                  // claim rowblk r (committed only AFTER grow succeeds)
    s_cmp_ge_u32 s33, G
    s_cbranch_scc1 .Lflow_cmp_tryadv             // rowblks exhausted (we are fat) -> shrink + try advance
    // read STAMP (gsti) for C addressing
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
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
    v_add_nc_u32 v12, v9, s52                    // B resident base (BRES_ROFF=0)
    s_mul_i32 s37, s33, (FM*256)
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
.else
    // WRITE-ONCE REDUCE: accumulate this segment's partial into LDS bank[r] (mirrors C frag layout;
    //   vaddr = v10=lane*32, base = ACC_BASE + r*ACC_STRIDE). ksi==0 (tile's first segment, POOL_N=1
    //   guarantees it drains before any later ksi) WRITES; ksi>0 ADDS. C is stored ONCE at ksi==mask
    //   (last segment) in .Lflow_cmp_tryadv. s31=ksi (survives WMMA), s33=rowblk r.
    acc_base_of s39, s33                          // s39 = ACC_BASE + r*ACC_STRIDE
    v_add_nc_u32 v12, v10, s39                     // v12 = bank r ds vaddr (lane*32 + bankbase)
    s_cmp_eq_u32 s31, 0                            // first segment of this tile's rowblk?
    s_cbranch_scc1 .Lflow_bankwr
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
.Lflow_bankwr:
    .set frag, 0
    .rept FM*FN
      .set e, 0
      .rept 8
        ds_store_b32 v12, v[ACC+frag*8+e] offset:(frag*1024 + e*4)
        .set e, e+1
      .endr
      .set frag, frag+1
    .endr
.Lflow_bankdn:
    s_wait_dscnt 0x0
.endif
    instr_inc STINSTR_COMP                        // diag: a rowblk-segment was actually computed+reduced
    s_add_u32 s45, s48, SL_RBDONE
    lds_fetch_add_r s47, s45, 1                   // s47 = old RBDONE; old==G-1 -> I am the UNIQUE completer
.if DYNVGPR
.Lflow_bshrink:
    s_alloc_vgpr 32                               // SHRINK -> lean (close the trapezoid burst) BEFORE any store
    s_cbranch_scc0 .Lflow_bshrink
.endif
    s_add_u32 s47, s47, 1
    s_cmp_ge_u32 s47, G                            // (old+1) >= G -> I completed this super-tile
    s_cbranch_scc0 .Lflow_loop                     // not the completer -> done, loop
    // COMPLETER (single wave -> NO race, NO redundant store, NO spinning losers): the super-tile is fully
    //   reduced. If it's the tile's LAST ksi (ksi==mask), store the G banks to C ONCE, s_wait_storecnt,
    //   THEN advance DRAIN -> the next tile's ksi=0 cannot overwrite the banks until this store is drained.
    //   s19/s30/s31 still hold mblk/tcol/ksi from this wave's own DECODE_STI (untouched by the reduce).
    s_cmp_eq_u32 s31, s67                          // ksi == mask (n_kseg-1) -> tile complete?
    s_cbranch_scc0 .Lflow_drain_adv               // not last ksi -> just advance DRAIN (no store)
.if !WOFLUSH
    s_mul_i32 s38, s19, s13                        // mblk*NTL
    s_add_u32 s38, s38, s30                        // + tcol
    s_mul_i32 s38, s38, (G*FM*FN*1024)            // * per-tile C bytes
    s_add_u32 s28, s6, s38
    s_addc_u32 s29, s7, 0                          // s[28:29] = C tile base (rowblk 0)
    .set r, 0
    .rept G
      s_mov_b32 s39, (ACC_BASE + r*(FM*FN*1024))   // bank r LDS base (compile-time)
      v_add_nc_u32 v12, v10, s39                    // v12 = bank r ds vaddr (lane*32)
      .set frag, 0
      .rept FM*FN
        .set e, 0
        .rept 8
          ds_load_b32 v13, v12 offset:(frag*1024 + e*4)
          s_wait_dscnt 0x0
          global_store_b32 v10, v13, s[28:29] offset:(r*(FM*FN*1024) + frag*1024 + e*4) scope:SCOPE_DEV
          .set e, e+1
        .endr
        .set frag, frag+1
      .endr
      .set r, r+1
    .endr
    s_wait_storecnt 0x0                            // store COMPLETE before DRAIN++ -> banks safe to reuse
.endif                                             // WOFLUSH: atomics already wrote C incrementally -> no store, just DRAIN++
.Lflow_drain_adv:
    lds_get s44, DRAIN_HEAD_OFF
    lds_cmpstore_adv DRAIN_HEAD_OFF, s44          // completer advances DRAIN (unique wave; store already done)
    s_branch .Lflow_loop
.Lflow_cmp_tryadv:
.if DYNVGPR
.Lflow_tashrink:
    s_alloc_vgpr 32                               // grew but rowblks exhausted (no claim) -> shrink back lean
    s_cbranch_scc0 .Lflow_tashrink
.endif
    s_branch .Lflow_loop                          // the bank store + DRAIN advance are done by the COMPLETER
                                                  //   (the unique wave whose SL_RBDONE inc hit G, in .Lflow_bankdn)

// ---- FEED work: stage the STAGE_HEAD slot (A if ROLE_AFEED, B if ROLE_BFEED), then try-advance STAGE ----
.Lflow_feed:
    lds_get s44, STAGE_HEAD_OFF                 // sh
    lds_get s45, ASSIGN_HEAD_OFF               // ah
    s_cmp_ge_u32 s44, s45                        // STAGE >= ASSIGN -> nothing assigned to stage -> yield
    s_cbranch_scc1 .Lflow_feed_empty
    slot_of s46, s44, s47                        // slot = sh mod N
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE              // scb
    s_lshl_b32 s52, s46, 14
    s_add_u32 s52, s52, OP_BASE                // sob
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45                           // gsti = STAMP (assigned -> set)
    DECODE_STI
    s_cmp_eq_u32 s34, ROLE_BFEED
    s_cbranch_scc1 .Lflow_stageB
    ASTAGE_R s48, s52
    s_branch .Lflow_stage_adv
.Lflow_stageB:
    BSTAGE_R s48, s52
.Lflow_stage_adv:
    // try-advance STAGE_HEAD: if the CURRENT STAGE_HEAD slot is fully staged, CAS-bump it
    lds_get s44, STAGE_HEAD_OFF
    lds_get s45, ASSIGN_HEAD_OFF
    s_cmp_ge_u32 s44, s45
    s_cbranch_scc1 .Lflow_loop
    slot_of s46, s44, s47
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE
    s_add_u32 s45, s48, SL_BFDONE
    lds_get_r s47, s45
    s_cmp_lt_u32 s47, FN
    s_cbranch_scc1 .Lflow_loop
    s_add_u32 s45, s48, SL_ARDONE
    lds_get_r s47, s45
    s_cmp_lt_u32 s47, G
    s_cbranch_scc1 .Lflow_loop
    lds_cmpstore_adv STAGE_HEAD_OFF, s44
    s_branch .Lflow_loop
.Lflow_feed_empty:
    s_sleep SLEEPN                                // lean feed can't compute without a grow -> yield; coordinator rebalances
    s_branch .Lflow_loop

// ---- COAST: a fat COMPUTE wave with no staged work runs feed code (FREE, no resize) to help staging ----
.Lflow_growfail:
    instr_inc STINSTR_GROWFAIL                      // diag: per-burst grow failed (budget full) -> coast = stagger repulsion
    // fall through: the failed grow allocated nothing, so we are still lean and safe to coast
.Lflow_coast:
    instr_inc STINSTR_COAST                        // diag: a wave coasted (no staged work, or a grow-fail)
    lds_get s44, STAGE_HEAD_OFF
    lds_get s45, ASSIGN_HEAD_OFF
    s_cmp_ge_u32 s44, s45
    s_cbranch_scc1 .Lflow_feed_empty              // nothing assigned to stage -> yield
    slot_of s46, s44, s47
    s_lshl_b32 s48, s46, 5
    s_add_u32 s48, s48, SLOTC_BASE
    s_lshl_b32 s52, s46, 14
    s_add_u32 s52, s52, OP_BASE
    s_add_u32 s45, s48, SL_BFDONE
    lds_get_r s47, s45
    s_cmp_lt_u32 s47, FN
    s_cbranch_scc1 .Lflow_coastB                  // B behind -> help B
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
    DECODE_STI
    ASTAGE_R s48, s52
    s_branch .Lflow_stage_adv
.Lflow_coastB:
    s_add_u32 s45, s48, SL_STI
    lds_get_r s17, s45
    DECODE_STI
    BSTAGE_R s48, s52
    s_branch .Lflow_stage_adv

.Lflow_retire:
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
    s_cmp_ge_u32 s52, RETBAR_MAX                // safety bound -> exit anyway (never hang)
    s_cbranch_scc1 .Lflow_endpgm
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
    .amdhsa_next_free_sgpr 72                  // body uses up to s69 (s66=n_kseg s67=mask s68=shift s69=chunkHi, FIX 1)
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
