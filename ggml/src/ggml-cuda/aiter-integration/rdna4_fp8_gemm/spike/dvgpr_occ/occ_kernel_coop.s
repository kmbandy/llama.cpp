// occ_kernel_coop.s  (gfx1201, wave32) -- MAD-305 HYBRID COOPERATIVE fp8 WMMA GEMM (HYBRID_DESIGN.md).
//
// B2a P=1 PROTOCOL BRING-UP (correctness-first; the riskiest piece = the lock-free busy-wait protocol).
// One (1+P)-wave workgroup, SHARED-B through LDS, NO s_barrier in the K-loop:
//   wave 0      = FEED/CLAIM: atomic-claims a tile, then per K16-step global_load_tr_b64's the FN shared
//                 B-frags, ds_stores them into a depth-RINGD ring, and publishes prod_count (monotonic).
//   waves 1..P  = COMPUTE: each grows (dyn) to its 2x4 footprint, owns a 32-row M-band, busy-waits the
//                 ring (prod_count gate), ds_load_b64's the shared B + global_load_b64's its OWN A direct,
//                 runs FM*FN accumulating WMMA, bumps cons_count[cid] (release), and full-frag fp32-stores.
// The feed wave's slot-free gate spins until min_c cons_count[c] > k-RINGD (no overwrite of an undrained
// slot). This is Step 3's exact counter protocol. The ONLY brick-relevant rendezvous is ONE symmetric
// pre-grow s_barrier that publishes the LDS-control init -- safe because every wave is still lean-32 there
// (the brick condition is a rendezvous at DIFFERENT allocations, which this is not; wavespec uses the same
// idiom on this silicon). dyn is the cornerstone: feed launches+STAYS lean 32, compute grow-per-tile.
//
// SIMPLIFICATIONS vs the locked design (both oracle-equivalent; layered in once green):
//   * K16/b64 inner loop (one K16 frag/step), not yet Step-6 K32/ds_load_b128. Isolates the protocol from
//     the load-width issue-density optimization.
//   * v1 stores ALL FM*FN fp32 frags per (tile, compute-wave) -> the run_mbcoop oracle layout. (Step 7a.)
//
// User data (USER_SGPR=15, s0..s14) -- IDENTICAL to occ_kernel_mbgemm.s:
//   s0:s1=occ s2:s3=A s4:s5=Bshuf s6:s7=C s8=KT s9=K(bytes/A-row) s10=NT*256 s11=TOTAL
//   s12=magic(ceil(2^32/NTL)) s13=NTL s14=FN*256 ; s15=TGID_X
//   (GENDIV decode only -- the ml8 N are non-pow2; matches run_mbcoop's useGenDiv=true.)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"

.ifndef FM
    .set FM, 2                              // per-compute-wave M-frags (M-band = FM*16 rows)
.endif
.ifndef FN
    .set FN, 4                              // shared N-frags (the reuse operand)
.endif
.ifndef P
    .set P, 1                               // compute waves / WG (B2a=1; B2b scales 2->3). WAVES = 1+P.
.endif
.ifndef RINGD
    .set RINGD, 2                           // B-ring depth (double-buffer floor). MUST be a power of two.
.endif
.ifndef BATCH
    .set BATCH, 1                           // tiles per atomic claim (grow granularity). B2a=1.
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 1                         // 1 = compute waves s_alloc_vgpr-grow (the moat). feed stays lean.
.endif
.ifndef RGADESC
    .set RGADESC, 0                         // 1 = emit analysis-only AMDHSA descriptor for RGA livereg
.endif
.ifndef DIAG
    .set DIAG, 0                            // 1 = phase-marker instrumentation: lane0 overwrites its current
.endif                                       //   location into occ[6]=feed-phase occ[7]=compute-phase occ[8]=feed-ti
                                             //   occ[9]=compute-ti. Localizes a protocol/exit hang. (occ buf is 0x1000.)
.ifndef SAFEPROBE
    .set SAFEPROBE, 0                       // 1 = BRICK-SAFE diagnostic: record RAW compute ti to occ[22] (DIAG),
.endif                                       //   then CLAMP ti to [0,TOTAL-1] so EVERY derived global address is
                                             //   provably in-buffer (no OOB possible, even on a garbage LDS read).
                                             //   Converts the dyn page-fault brick into an observable oracle-BAD.
.ifndef BUSYWAIT
    .set BUSYWAIT, 0                        // 1 = replace the ONE pre-grow s_barrier (LDS-init publish) with an LDS
.endif                                       //   busy-wait flag spin. The dyn brick is a DEADLOCK (ring timeout, no
                                             //   VM fault) = s_barrier-under-dyn-VGPR hangs (wavespec BRICK#4 + this
                                             //   coop death cert). s_barrier-FREE coordination is proven under dyn
                                             //   (GRING probe). BUSYWAIT=1 removes the last hw barrier -> THE dyn fix.
.ifndef PARKFEED
    .set PARKFEED, 0                        // 1 = feed SPINS (never s_endpgm) at terminal -> isolate whether the
.endif                                       //   feed wave's retirement wedges the compute's last-tile store-wait.
.ifndef STOREWAIT
    .set STOREWAIT, 0                        // 1 = drain global stores (s_wait_storecnt 0) before s_endpgm so the
.endif                                       //   terminal C store reaches L2 -> the EOP RELEASE_MEM fence FIRES ->
                                             //   the KFD queue becomes IDLE -> clean teardown (no process-exit brick).
                                             //   Targets the 2026-06-25 pool>=2 teardown wedge (EOP never fired in 5s
                                             //   -> non-idle queue -> KFD reclaim wedges the GPU). storecnt drains on
                                             //   L2-accept (not memory-completion) so this should NOT hang. UNTESTED
                                             //   on silicon -> gated; static d0 stays byte-identical at STOREWAIT=0.
.ifndef POOLTERM
    .set POOLTERM, 0                        // 1 = pool>1 compute terminal. The pool=1 terminal counts tiles to
.endif                                       //   TOTAL (assumes ONE WG owns all tiles). At pool>=2 the WGs SPLIT
                                             //   the TOTAL tiles via the shared global claim, so a per-WG compute
                                             //   only ever receives its share (<TOTAL) and the count-to-TOTAL
                                             //   terminal NEVER fires -> compute spins at .Lwait_epoch forever ->
                                             //   the WG never retires -> EOP never fires -> teardown wedges the GPU
                                             //   (THE 2026-06-26 pool>=2 brick; the count terminal was a pool=1-only
                                             //   diagnostic stub). FIX: each compute exits when ITS feed broadcasts a
                                             //   terminal ti (>=TOTAL) into the per-WG LDS TI_OFF (the feed already
                                             //   publishes it before .Lfeed_exit) -- a per-WG feed->compute handshake,
                                             //   correct for ANY WG count. POOLTERM=0 keeps the count terminal so the
                                             //   proven static d0 stays byte-identical; the dyn pool>=2 build sets 1.
.ifndef SLEEPN
    .set SLEEPN, 2                          // s_sleep arg in the cooperative busy-waits (DYNVGPR only). The grown
.endif                                       //   compute's TIGHT spin starves the lean feed under dyn -> deadlock
                                             //   (DIAG: computePhase=4, occ[10]=1 then wedge). s_sleep yields issue
                                             //   cycles to the partner. Steady-state cost ~0 (loops spin ~0x when fed).

// ---- DSWS (MAD-305 adaptive wave-role controller, SPEC_DSWS_CONTROLLER.md). Phase 1 = STATIC 3-role
//   substrate {compute / A-feed / B-feed}, all NEW behavior gated behind `.if DSWS` so DSWS=0 stays
//   BYTE-IDENTICAL to the proven 2-role coop d0 (1716B). Design (T1.2 decision, KG 4ce31886):
//     * band-partitioned A: A-feed wave a owns strided bands {a, a+NAFEED, ...}; band c has ONE producer
//       (prod_a[c]) and ONE consumer (compute c -> cons_a[c]). No min over consumers needed for A.
//     * frag-partitioned B: B-feed wave b owns strided frags {b, b+NBFEED, ...} of the FN shared frags;
//       each bumps prod_b[b]; compute gates on min(prod_b[*]). The lead B-feed (wid 0) is the SOLE tile
//       claimer + ti/epoch broadcaster; A-feed + non-lead B-feed FOLLOW the broadcast like compute.
//     * ZERO inter-feed rendezvous (the consumer-side min is the only cross-wave wait) -> the barrier-free
//       invariant that keeps dyn-VGPR safe is preserved. ----
.ifndef DSWS
    .set DSWS, 0
.endif
.if DSWS
  .ifndef NCOMP
    .set NCOMP, 4                            // compute waves (fat, dyn-grow). Compute floor >= 1.
  .endif
  .ifndef NAFEED
    .set NAFEED, 2                           // A-feed waves (lean). Feed floor >= 1.
  .endif
  .ifndef NBFEED
    .set NBFEED, 2                           // B-feed waves (lean). Feed floor >= 1.
  .endif
  .ifndef RINGD_A
    .set RINGD_A, RINGD                      // A-ring depth (defaults to the B-ring depth). MUST be power of two.
  .endif
  // ---- Controller thresholds (Phase 3 actuation; mirror occ_dispatch DSWS_LOW/HIGH/EPOCHSHIFT). ----
  .ifndef EPOCH_SHIFT
    .set EPOCH_SHIFT, 3                       // decision clock: epoch = segcnt >> EPOCH_SHIFT (small=reactive)
  .endif
  .ifndef CTRL_LOW
    .set CTRL_LOW, 1                          // occ_X < CTRL_LOW  -> compute starved for X (shrink compute->feedX)
  .endif
  .ifndef CTRL_HIGH
    .set CTRL_HIGH, (RINGD-1)                 // occ_X > CTRL_HIGH -> feed-X over-serving (grow feedX->compute)
  .endif
  // The existing compute body is parameterized by P = "#compute waves" (rowblk=trow*P+cid, store strides).
  //   Under DSWS the compute count is NCOMP -> bind P=NCOMP so all that addressing auto-targets NCOMP bands.
  .set P, NCOMP
  .set WAVES, (NCOMP + NAFEED + NBFEED)       // total waves launched per WG (harness dims must match)
.else
  .set WAVES, (1 + P)                          // total waves launched per WG (harness dims must match)
.endif

// ---- LDS layout (bytes) -- matches run_mbcoop sizing: RINGD*FN*256 + 4*(1 + P + 3) ----
.set BRING_OFF, 0                            // B ring: RINGD slots, each FN frags x 256B
.set PROD_OFF,  (RINGD*FN*256)               // prod_count (u32)   -- feed-written
.set CONS_OFF,  (PROD_OFF + 4)               // cons_count[P] (u32 each) -- compute-c-written
.set TI_OFF,    (CONS_OFF + 4*P)             // current tile index (u32) -- feed broadcast
.set EPOCH_OFF, (TI_OFF + 4)                 // tile epoch (u32) -- feed bump publishes a new tile
.set INITFLAG_OFF, (EPOCH_OFF + 4)           // BUSYWAIT init-publish flag (the former spare u32)
.set LDS_TOTAL, (EPOCH_OFF + 4 + 4)          // + 1 spare u32 (now INITFLAG; sizing unchanged)
.if DSWS
  // ---- DSWS extra LDS, all APPENDED after the DSWS=0 region so the proven offsets stay byte-identical ----
  //   prod_b[0] reuses PROD_OFF (the frag-partition lead); prod_b[1..NBFEED-1] live just past INITFLAG.
  .set PRODB_HI_OFF, LDS_TOTAL                              // prod_b[1..NBFEED-1] : NBFEED-1 u32 (0 bytes when NBFEED=1)
  .set ARING_OFF,    (PRODB_HI_OFF + 4*(NBFEED-1))          // A-ring: RINGD_A slots x NCOMP bands x FM frags x 256B
  .set PROD_A_OFF,   (ARING_OFF + RINGD_A*NCOMP*FM*256)     // prod_a[NCOMP] (band-partitioned producers)
  .set CONS_A_OFF,   (PROD_A_OFF + 4*NCOMP)                 // cons_a[NCOMP] (compute A release; one consumer per band)
  // ---- Controller state (Phase 2 sensing/slots/reservation; Phase 3 actuates). Mirrors dsws_ctrl_model.cpp. ----
  //   Role slots: live count of waves currently in each role (leader inits to the launch mix; conversions adjust).
  .set NCOMP_SLOT,  (CONS_A_OFF + 4*NCOMP)                  // u32 nComp   (live fat-compute wave count)
  .set NAFEED_SLOT, (NCOMP_SLOT + 4)                        // u32 nAfeed  (live A-feed wave count)
  .set NBFEED_SLOT, (NAFEED_SLOT + 4)                       // u32 nBfeed  (live B-feed wave count)
  //   Conversion gates: gate[dir] holds the last epoch in which direction `dir` fired (gate_try_win CAS target).
  //   4 directions: 0=compute->Afeed, 1=compute->Bfeed, 2=Afeed->compute, 3=Bfeed->compute.
  .set GATE_OFF,    (NBFEED_SLOT + 4)                       // u32[4] gate[dir]
  .set VRESV_OFF,   (GATE_OFF + 4*4)                        // u32 vgpr_reserved (sum-envelope: reserve_grow target)
  .set SEGCNT_OFF,  (VRESV_OFF + 4)                         // u32 segments_processed (per-WG decision clock source)
  .set LDS_TOTAL_DSWS, (SEGCNT_OFF + 4)
  .set ABAND_STRIDE, (RINGD_A*FM*256)                       // bytes per compute band's A sub-ring
  .if LDS_TOTAL_DSWS > 65536
    .error "DSWS LDS exceeds 64KB/WG (gfx1201): reduce RINGD_A / NCOMP / tile"
  .endif
.endif

// ---- VGPR layout ----
.set ACC, 32                                 // accumulators: FM*FN frags x 8 f32 (compute; above the lean block)
.set FA,  (ACC + 8*FM*FN)                    // compute A frags: FM x 2
.set FB,  (FA + 2*FM)                        // compute B frags (from LDS): FN x 2
// next_free_vgpr ROUNDED to a 16-VGPR dyn-alloc block. 2x4 -> 112 (highest index 107 needs >=108): a clean
// 7-block alloc, SAFELY below the 128 cap and OFF the 128 exact-fill edge (the strategic dyn danger zone).
.set NFV, ((FB + 2*FN + 15) & ~15)
.if DSWS
  // Controller reservation-envelope units (hardware ALLOC footprint, not live-peak): a fat-compute wave
  //   reserves NFV (its s_alloc_vgpr target); a lean feed wave reserves VLEAN (its `s_alloc_vgpr 32`).
  //   Phase-3 grow delta = NFV - VLEAN; vgpr_reserved init (Phase 2) = NCOMP*NFV + (NAFEED+NBFEED)*VLEAN.
  .set VLEAN, 32
.endif
.set BSTG, 16                                // feed B staging: FN x 2 at v16.. (lean block, < 32)
.if DSWS
  .set ASTG, 16                              // A-feed staging: FM x 2 at v16.. (A-feed is a distinct lean wave,
.endif                                        //   so it reuses the same lean block as the B-feed's BSTG)

// ---- dyn-VGPR PRE-GROW VGPR CEILING (THE deadlock + dead-marks root cause; Codex + RDNA4 ISA confirmed) ----
// A dyn-armed (RSRC2 bit6) wave LAUNCHES with exactly ONE 16-VGPR block backed (v0..v15); RSRC1.VGPRS is
// IGNORED in dyn mode (it only sets the static-mode footprint). The shared init/rendezvous AND the compute
// epoch-wait run BEFORE any s_alloc_vgpr -> at lean-16. Per RDNA4 OOR rules a >v15 source there is poison:
//   - LDS address reg OOR -> aliases v0 -> ds_load/ds_store hit the WRONG LDS addr -> the rendezvous spin
//     never sees INITFLAG=0xACED -> infinite spin = THE "deadlock" (not a barrier/dyn incompatibility).
//   - memory-instruction source reg OOR -> undefined -> the global_atomic marks write NOTHING = dead markers.
// FIX: gate every PRE-grow-reachable LDS/atomic temp to v14/v15 under dyn. Static keeps v27..v31 verbatim
// (full alloc, no OOR) so the proven-green d0 bin stays BYTE-IDENTICAL.
.if DYNVGPR
  // v11 + v14: both unused elsewhere AND comfortably INTERIOR to the v0..v15 backed block (not on the v15
  // edge), so they stay valid under either a 16- or 32-VGPR dyn block size (defense-in-depth at the exact
  // boundary where this OOR bug class lives).
  .set RG_A, 11                              // lds_get   address
  .set RG_D, 14                              // lds_get   data
  .set RP_A, 11                              // lds_put   address
  .set RP_D, 14                              // lds_put   data
  .set RPV_D, 14                             // lds_put_v data
  .set RM_A, 11                              // mark      address (self-zeroed vaddr)
  .set RM_D, 14                              // mark      data
  .set RI_A, 11                              // init-direct LDS address
  .set RI_D, 14                              // init-direct LDS data
.else
  .set RG_A, 27
  .set RG_D, 28
  .set RP_A, 28
  .set RP_D, 29
  .set RPV_D, 29
  .set RM_A, 31
  .set RM_D, 30
  .set RI_A, 28
  .set RI_D, 29
.endif

// ===========================================================================================
// LDS helper macros. Temp regs gated above (dyn: v14/v15 backed pre-grow; static: v27..v31 full alloc).
// s49 = exec save for the lane-0 masked writes. v2 = lane = tid & 31 (set in prologue).
// ===========================================================================================
.macro lds_get sdst, off                     // wave-uniform read LDS[off] -> scalar sdst (all lanes load same)
    v_mov_b32 v[RG_A], \off
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
.macro lds_put_v vaddr, ssrc                  // lane-0-of-wave writes scalar ssrc -> LDS[vaddr] (runtime addr)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lputv_skip\@
    v_mov_b32 v[RPV_D], \ssrc
    ds_store_b32 \vaddr, v[RPV_D]
    s_wait_dscnt 0x0
.Lputv_skip\@:
    s_mov_b32 exec_lo, s49
.endm
.macro mark off, val                          // DIAG: lane0-of-wave atomic-MAX \val into occ[off] (byte offset)
.if DIAG                                        //   -> records the FURTHEST point each wave reached (proven timer pattern)
    s_mov_b32 s58, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lmk\@
    v_mov_b32 v[RM_A], 0                        // self-zeroed vaddr (robust vs any v4 clobber)
    v_mov_b32 v[RM_D], \val
    global_atomic_max_u32 v[RM_A], v[RM_D], s[0:1] offset:\off scope:SCOPE_DEV
.Lmk\@:
    s_mov_b32 exec_lo, s58
.endif
.endm

.macro mark_set off, val                      // DIAG: lane0-of-wave PLAIN-store \val -> occ[off] (INSTANTANEOUS, overwrites).
.if DIAG                                        //   Unlike `mark` (atomic-MAX = furthest-reached), this streams the CURRENT
    s_mov_b32 s58, exec_lo                      //   value so the harness's 200ms poll sees a sensor oscillate (occ_a/occ_b).
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lms\@
    v_mov_b32 v[RM_A], 0                        // self-zeroed vaddr (same robustness as mark)
    v_mov_b32 v[RM_D], \val
    global_store_b32 v[RM_A], v[RM_D], s[0:1] offset:\off scope:SCOPE_DEV
.Lms\@:
    s_mov_b32 exec_lo, s58
.endif
.endm

.macro mark_inc off                           // DIAG: lane0-of-wave atomic-ADD 1 -> occ[off] (a free-running counter).
.if DIAG
    s_mov_b32 s58, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lmi\@
    v_mov_b32 v[RM_A], 0
    v_mov_b32 v[RM_D], 1
    global_atomic_add_u32 v[RM_A], v[RM_D], s[0:1] offset:\off scope:SCOPE_DEV
.Lmi\@:
    s_mov_b32 exec_lo, s58
.endif
.endm

// try_gate: the lock-free single-winner conversion ticket (transcribes dsws_ctrl_model.cpp gate_try_win +
//   epoch_of EXACTLY). E = segcnt>>EPOCH_SHIFT. gate[dir] holds the last epoch dir fired. Among many waves
//   racing the same (g<E), exactly ONE wins per epoch via the LDS compare-swap; the rest see g advanced.
//   \swin <- 1 if THIS wave won the (dir,epoch) ticket, else 0. Read-only on state besides gate[dir]; NO role
//   actuation here (3.2/3.3 act on \swin). Scratch: s62..s65 (free in all roles), v5/v6/v7 (lean-safe, 0-ref).
//   CAS operand order (gfx1201, verified via LLVM cmpxchg lowering -- it is the GCN order, NOT flipped):
//   ds_cmpstore_rtn_b32 vdst,vaddr,vNEW,vCMP -> MEM=(MEM==vCMP)?vNEW:MEM, vdst<-old. So vsrc0=E (new),
//   vsrc1=g (compare). WIN iff returned-old == g. (T3.1 micro-check CAUGHT the swapped form: it left gate
//   stuck at 0 so old==g held for all racers -> would-win ~= NCOMP*epochs instead of ~= epochs.)
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

.macro min_cons sdst                          // sdst = min over c in 0..P-1 of cons_count[c]
    lds_get \sdst, CONS_OFF
    .set cc, 1
    .rept P-1
      lds_get s46, (CONS_OFF + cc*4)
      s_min_u32 \sdst, \sdst, s46
      .set cc, cc+1
    .endr
.endm
.if DSWS
.macro lds_get_r sdst, saddr                  // wave-uniform read LDS[saddr] -> scalar sdst (RUNTIME addr in a sreg)
    v_mov_b32 v[RG_A], \saddr
    ds_load_b32 v[RG_D], v[RG_A]
    s_wait_dscnt 0x0
    v_readfirstlane_b32 \sdst, v[RG_D]
.endm
.macro min_prod sdst                          // sdst = min over b in 0..NBFEED-1 of prod_b[b] (frag-partitioned B).
    lds_get \sdst, PROD_OFF                     //   prod_b[0] is PROD_OFF; prod_b[1..] live in the DSWS hi region.
    .set pw, 1
    .rept NBFEED-1
      lds_get s46, (PRODB_HI_OFF + (pw-1)*4)
      s_min_u32 \sdst, \sdst, s46
      .set pw, pw+1
    .endr
.endm
.endif

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- identity ----
    v_lshrrev_b32 v1, 5, v0                  // wid = tid >> 5  (0 = feed, 1..P = compute)
    v_and_b32     v2, 31, v0                 // lane = tid & 31
    v_and_b32     v6, 15, v0                 // lane & 15 (A vaddr)
    v_mov_b32     v4, 0
    // ---- per-lane address constants (mbgemm-identical) ----
    v_mul_lo_u32  v8, v6, s9                 // (lane&15)*K
    v_bfe_u32     v7, v0, 4, 1
    v_lshlrev_b32 v7, 3, v7
    v_add_nc_u32  v8, v8, v7                 // v8 = A vaddr = (lane&15)*K + colhi*8
    v_lshlrev_b32 v9, 3, v2                  // v9 = B/ds vaddr = lane*8  (v2=lane, NOT v0=flat-tid: compute wave
                                             //   has v0=32..63 -> v0*8 shifted the LDS B-read +256B = wrong frag)
    v_lshlrev_b32 v10, 5, v2                 // v10 = C store vaddr = lane*32  (same: must be per-wave lane)
.if SAFEPROBE
    // brick-PROOF: cap the per-lane VECTOR address regs to a loose upper bound (>= true max, so VALID lanes are
    //   unaffected) -> even a grow-corrupted vaddr cannot push a global access past the data+guard. Pairs with the
    //   ti clamp (bounds the saddr) to make EVERY global address provably in-buffer (Codex safety review).
    s_lshl_b32 s16, s9, 4                     // 16*K  (>= v8 max = (lane&15)*K + colhi*8 = 15*K+8)
    v_min_u32 v8, s16, v8                      // clamp A vaddr   (s16 reused for exec save just below -> dead after)
    v_min_u32 v9, 0x100, v9                    // clamp B/ds vaddr (256 >= lane*8 max 248)
    v_min_u32 v10, 0x400, v10                  // clamp C vaddr    (1024 >= lane*32 max 992)
.endif

    // ---- admission bookkeeping (leader tid==0 == feed-wave lane0): live++, maxlive, total++ ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v3, 1
    global_atomic_add_u32 v5, v4, v3, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v5, v5, 1
    global_atomic_max_u32 v4, v5, s[0:1] offset:4 scope:SCOPE_DEV
    global_atomic_add_u32 v4, v3, s[0:1] offset:16 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s16
    mark 92, 1                                 // DIAGFINE occ[23] = reached post-admission
    // ---- timer t0 (leader writes min start tick to occ[2]) ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v5, s30
    global_atomic_min_u32 v4, v5, s[0:1] offset:8 scope:SCOPE_DEV
.Lafter_t0:
    s_mov_b32 exec_lo, s16
    mark 96, 1                                 // DIAGFINE occ[24] = reached post-timer (s_sendmsg_rtn returned)

    // ---- LDS-control init: leader (tid==0) zeroes prod_count, cons_count[P], epoch. The ONE symmetric
    //   pre-grow s_barrier below publishes it to all waves (every wave lean-32 here -> NOT the brick
    //   condition). After this, compute waves see epoch=0 and wait for the feed's first bump to >0. ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Linit_done
    v_mov_b32 v[RI_D], 0
    v_mov_b32 v[RI_A], PROD_OFF
    ds_store_b32 v[RI_A], v[RI_D]
    .set cc, 0
    .rept P
      v_mov_b32 v[RI_A], (CONS_OFF + cc*4)
      ds_store_b32 v[RI_A], v[RI_D]
      .set cc, cc+1
    .endr
    v_mov_b32 v[RI_A], EPOCH_OFF
    ds_store_b32 v[RI_A], v[RI_D]
.if DSWS
    // zero the DSWS monotonic counters: prod_b[1..NBFEED-1], prod_a[NCOMP], cons_a[NCOMP] (start at 0).
    //   (the A-ring storage needs no zeroing -- it is overwritten before read, gated by prod_a.)
    .set zw, 0
    .rept (NBFEED-1)
      v_mov_b32 v[RI_A], (PRODB_HI_OFF + zw*4)
      ds_store_b32 v[RI_A], v[RI_D]
      .set zw, zw+1
    .endr
    .set zc, 0
    .rept NCOMP
      v_mov_b32 v[RI_A], (PROD_A_OFF + zc*4)
      ds_store_b32 v[RI_A], v[RI_D]
      v_mov_b32 v[RI_A], (CONS_A_OFF + zc*4)
      ds_store_b32 v[RI_A], v[RI_D]
      .set zc, zc+1
    .endr
    // ---- Controller state init (Phase 2): role slots <- launch mix, gates <- 0, vgpr_reserved <- launch
    //   envelope footprint, segcnt <- 0. Leader-lane writes constants to WG-shared LDS (idempotent across
    //   waves: every lane-0 writes identical compile-time values). v[RI_D] still holds 0 from above. ----
    v_mov_b32 v[RI_A], NCOMP_SLOT
    v_mov_b32 v[RI_D], NCOMP
    ds_store_b32 v[RI_A], v[RI_D]
    v_mov_b32 v[RI_A], NAFEED_SLOT
    v_mov_b32 v[RI_D], NAFEED
    ds_store_b32 v[RI_A], v[RI_D]
    v_mov_b32 v[RI_A], NBFEED_SLOT
    v_mov_b32 v[RI_D], NBFEED
    ds_store_b32 v[RI_A], v[RI_D]
    v_mov_b32 v[RI_D], 0
    .set zg, 0
    .rept 4
      v_mov_b32 v[RI_A], (GATE_OFF + zg*4)         // gate[dir] = 0 (no conversion has fired)
      ds_store_b32 v[RI_A], v[RI_D]
      .set zg, zg+1
    .endr
    v_mov_b32 v[RI_A], VRESV_OFF                    // vgpr_reserved = NCOMP*NFV + (NAFEED+NBFEED)*VLEAN
    v_mov_b32 v[RI_D], (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)
    ds_store_b32 v[RI_A], v[RI_D]
    v_mov_b32 v[RI_A], SEGCNT_OFF                   // segments_processed = 0
    v_mov_b32 v[RI_D], 0
    ds_store_b32 v[RI_A], v[RI_D]
.endif
    s_wait_dscnt 0x0
.Linit_done:
    s_mov_b32 exec_lo, s16
    mark 100, 1                                // DIAGFINE occ[25] = reached post-LDS-init (prod/cons/epoch zeroed)
.if BUSYWAIT
    // ---- s_barrier-FREE init publish (THE dyn fix): leader (tid==0) writes INITFLAG LAST (after the prod/cons/
    //   epoch zeros above); ALL waves spin until they see it. No hardware workgroup barrier -> no dyn-VGPR
    //   barrier deadlock. Cross-wave LDS visibility under dyn is proven (GRING probe). ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lflag_pub
    v_mov_b32 v[RI_D], 0xACED
    v_mov_b32 v[RI_A], INITFLAG_OFF
    ds_store_b32 v[RI_A], v[RI_D]
    s_wait_dscnt 0x0
.Lflag_pub:
    s_mov_b32 exec_lo, s16
    mark 104, 1                                // DIAGFINE occ[26] = reached post-INITFLAG-write (leader wrote / others skipped)
.Lwait_init:
    s_sleep 1
    lds_get s44, INITFLAG_OFF
    s_cmp_eq_u32 s44, 0xACED
    s_cbranch_scc0 .Lwait_init                 // spin until the leader's INITFLAG is visible (all waves rendezvous)
    mark 108, 1                                // DIAGFINE occ[27] = reached post-rendezvous (escaped the init spin)
.else
    s_barrier_signal -1                       // SAFE: symmetric, all waves lean-32 (init publish only)
    s_barrier_wait -1
    mark 108, 1                                // DIAGFINE occ[27] = reached post-rendezvous (passed s_barrier)
.endif

.if DSWS
    // ===== DSWS 3-role branch (wid uniform per wave; scalar-only -> exec stays full for every role).
    //   wid [0,NBFEED)              -> B-feed  (wid 0 = lead: sole tile-claimer + ti/epoch broadcaster)
    //   wid [NBFEED,NBFEED+NAFEED)  -> A-feed
    //   wid [NBFEED+NAFEED, WAVES)  -> compute, cid = wid - (NBFEED+NAFEED) =====
    v_readfirstlane_b32 s24, v1               // wid (uniform per wave)
    s_cmp_lt_u32 s24, NBFEED
    s_cbranch_scc1 .Lbfeed_role
    s_cmp_lt_u32 s24, (NBFEED+NAFEED)
    s_cbranch_scc1 .Lafeed_role
    s_sub_u32 s47, s24, (NBFEED+NAFEED)       // cid
    s_branch .Ldsws_compute_init
.else
    // ============ ROLE BRANCH: wave 0 -> feed ; waves 1..P -> compute ============
    v_cmp_eq_u32 vcc_lo, 0, v1
    s_mov_b32 s25, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcompute_role
    s_mov_b32 exec_lo, s25                     // only the feed wave reaches here -> restore full (feed) exec
    s_branch .Lfeed_role
.Lcompute_role:
    s_mov_b32 exec_lo, s25                     // only compute waves reach here
.endif

    // ========================================================================================
    // COMPUTE BODY (waves 1..P). cid = wid-1 (DSWS=0) or wid-(NBFEED+NAFEED) (DSWS). Owns M-band
    //   [cid*FM*16 ..). Shares the FN N-cols. DSWS: also consumes its A-band from the A-ring.
    // ========================================================================================
.if DSWS == 0
    v_sub_nc_u32 v3, v1, 1
    v_readfirstlane_b32 s47, v3               // cid (uniform per wave)
.endif
.Ldsws_compute_init:                          // DSWS path joins here with s47=cid already set
    // cons_count[cid] LDS address (runtime): v12 = CONS_OFF + cid*4
    s_lshl_b32 s36, s47, 2
    s_add_u32  s36, s36, CONS_OFF
    v_mov_b32  v12, s36
.if DSWS
    // per-band A-consume state: &prod_a[cid], &cons_a[cid], A-band ring base, a_step (cumulative A consume).
    s_lshl_b32 s52, s47, 2
    s_add_u32  s52, s52, PROD_A_OFF           // &prod_a[cid]
    s_lshl_b32 s53, s47, 2
    s_add_u32  s53, s53, CONS_A_OFF           // &cons_a[cid]
    s_mul_i32  s51, s47, ABAND_STRIDE
    s_add_u32  s51, s51, ARING_OFF            // A-band ring base (this cid's sub-ring)
    s_mov_b32  s54, 0                          // a_step = 0 (cumulative A consume; matches producer prod_a[cid])
.endif
    s_mov_b32  s35, 0                          // local_epoch = 0 (matches the init epoch)
    s_mov_b32  s57, 0                          // tiles processed (POOL=1 count-to-TOTAL terminal, see below)
    s_mov_b32  s56, 0                          // GLOBAL cons step (cumulative consumed; MONOTONIC, matches feed s55)
.if DSWS
    // ---- Phase-2 sensing: stream the live role slots (read-only; static in Phase 2) once per compute wave.
    //   Proves the leader's controller-state init round-trips through WG-shared LDS into the occ snapshot.
    lds_get s59, NCOMP_SLOT
    mark_set 136, s59                           // occ[34] = nComp  (live fat-compute wave count)
    lds_get s59, NAFEED_SLOT
    mark_set 140, s59                           // occ[35] = nAfeed (live A-feed wave count)
    lds_get s59, NBFEED_SLOT
    mark_set 144, s59                           // occ[36] = nBfeed (live B-feed wave count)
.endif

.Lcompute_loop:
    mark 28, 1
    // ---- POOL=1 DIAGNOSTIC TERMINAL: a single WG processes EXACTLY TOTAL tiles, so the compute counts them
    //   and exits at TOTAL -- NO cross-wave terminal signal needed. This sidesteps the proven cross-wave
    //   LDS<->global ordering problem (occ[10] evidence: the compute can observe the LDS epoch update BEFORE
    //   the global claim update, so a global-claim terminal can never be trusted here). Lets the ORACLE run to
    //   test whether the per-tile cooperative B-handoff is numerically correct. (General pool>1 terminal needs
    //   an lds_barrier-class rendezvous; TBD once the per-tile math is proven.)
.Lwait_epoch:
.if DYNVGPR
    s_sleep SLEEPN                          // yield to the lean feed (grown-wave tight-spin starves it under dyn)
.endif
    lds_get s44, EPOCH_OFF
    s_cmp_eq_u32 s44, s35
    s_cbranch_scc1 .Lwait_epoch                // wait for the next tile (epoch change; drain-enforced)
    s_mov_b32 s35, s44                         // adopt new epoch
    lds_get s17, TI_OFF                        // ti (drain-enforced reliable for real tiles)
.if POOLTERM
    // pool>1 TERMINAL (the real cross-WG retire fix): the feed publishes its claimed ti into TI_OFF + bumps
    //   epoch BEFORE it exits, INCLUDING the terminal claim (ti>=TOTAL) -- see .Lfeed_exit path. So when this
    //   compute observes a broadcast ti>=TOTAL, ITS feed is done and there is no further tile for THIS WG.
    //   Exit now. Per-WG handshake -> correct for any WG count (each compute retires on its own feed's terminal),
    //   unlike the count-to-TOTAL below which only fires when one WG owns all TOTAL tiles. MUST test the RAW ti
    //   here, before the SAFEPROBE clamp pins it to TOTAL-1 and hides the signal.
    s_cmp_ge_u32 s17, s11
    s_cbranch_scc1 .Lcompute_exit
.endif
.if SAFEPROBE
    mark 88, s17                               // SAFEPROBE occ[22] = MAX raw ti the compute ever read (garbage shows here)
    s_sub_u32 s36, s11, 1                       // TOTAL-1
    s_min_u32 s17, s17, s36                     // CLAMP ti -> all A/C addresses provably in-buffer (no OOB brick)
.endif
    mark 28, 2
    mark 36, s17

    // ---- decode trow/tcol (GENDIV); A row-block base = trow*P + cid ----
    s_mul_hi_u32 s19, s17, s12                 // trow = mul_hi(ti, magic)
    s_mul_i32  s18, s19, s13                   // trow*NTL
    s_sub_u32  s18, s17, s18                   // tcol = ti - trow*NTL   (unused by compute; B is shared)
    s_mul_i32  s36, s19, P                     // trow*P
    s_add_u32  s36, s36, s47                   // rowblk = trow*P + cid
    // A frag saddrs: A_saddr(0) = A + rowblk*(16*FM)*K ; A_saddr(mi) = prev + 16*K
    s_lshl_b32 s32, s9, 4                       // rowstride16 = 16*K
    s_mul_i32  s22, s36, (16*FM)
    s_mul_i32  s22, s22, s9
    s_add_u32  s40, s2, s22
    s_addc_u32 s41, s3, 0
    .set mi, 1
    .rept FM-1
      s_add_u32  s[40+2*mi], s[40+2*(mi-1)], s32
      s_addc_u32 s[41+2*mi], s[41+2*(mi-1)], 0
      .set mi, mi+1
    .endr

    // ---- GROW first (before any operand load into the grown range), then zero accumulators ----
.if DYNVGPR
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
.Lgrow_retry:
    s_alloc_vgpr NFV                            // SCC=1 success / 0 reject -> spin-retry (the guard)
    s_cbranch_scc0 .Lgrow_retry
.endif
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

    mark 28, 3
    // ---- K-loop: consume the shared B ring + direct-load own A + accumulate ----
    s_mov_b32 s26, 0                           // k = 0
    mark 28, 4
.Lcons_k:
    // ready gate (GLOBAL step): spin until prod > s56
.Lwait_prod:
.if DYNVGPR
    s_sleep SLEEPN                          // yield to the lean feed so it can publish the next B-step
.endif
.if DSWS
    min_prod s44                               // B ready = min over frag-partitioned producers prod_b[*] > s56
.else
    lds_get s44, PROD_OFF
.endif
    s_cmp_le_u32 s44, s56
    s_cbranch_scc1 .Lwait_prod
    mark 56, s56                                // DIAGFINE occ[14] = max step that PASSED the prod-wait (about to ds_load B)
.if DSWS
    // Phase-2 sensor: occ_b = prod_b_min - cons, sampled HERE (consume point, pre-release) where it is the
    //   live ring backlog and -- by the gate above (prod>cons) -- provably in [1, RINGD]. Read-only, NO action.
    s_sub_u32 s59, s44, s56
    mark_set 128, s59                            // occ[32] = occ_b (instantaneous B-ring backlog, [1,RINGD])
.endif
    // ring slot = s56 & (RINGD-1)  (GLOBAL step -> matches the feed's continuous ring)
    s_and_b32 s45, s56, (RINGD-1)
    s_mul_i32 s45, s45, (FN*256)
    v_add_nc_u32 v13, v9, s45
    // ds_load FN shared B frags from the ring slot
    .set ni, 0
    .rept FN
      ds_load_b64 v[FB+ni*2:FB+ni*2+1], v13 offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_dscnt 0x0
    mark 60, s56                                // DIAGFINE occ[15] = max step with B ds_load'd from the ring
    // RELEASE the slot now (B is in our regs, before the WMMAs): cons = s56+1 (GLOBAL, monotonic)
    s_add_u32 s56, s56, 1
    lds_put_v v12, s56
    mark 40, s56                               // DIAG: occ[10] = max GLOBAL cons step reached (KT*32=1024 = all consumed)
.if DSWS
    // ---- consume FM own A frags from the A-ring band cid (gated on prod_a[cid] > a_step); release cons_a[cid] ----
.Lwait_proda:
  .if DYNVGPR
    s_sleep SLEEPN                             // yield to the lean A-feed so it can publish the next A-step
  .endif
    lds_get_r s44, s52                          // prod_a[cid]
    s_cmp_le_u32 s44, s54
    s_cbranch_scc1 .Lwait_proda
    // Phase-2 sensor: occ_a = prod_a[cid] - a_step, sampled HERE (A-consume point) where the gate guarantees
    //   prod_a > a_step -> occ_a in [1, RINGD_A]. Read-only, NO action.
    s_sub_u32 s59, s44, s54
    mark_set 132, s59                            // occ[33] = occ_a (instantaneous A-band backlog, [1,RINGD_A])
    // A-ring slot addr = A-band base (s51) + (a_step & (RINGD_A-1))*FM*256, per-lane lane*8 (v9)
    s_and_b32 s45, s54, (RINGD_A-1)
    s_mul_i32 s45, s45, (FM*256)
    s_add_u32 s45, s45, s51
    v_add_nc_u32 v13, v9, s45
    .set mi, 0
    .rept FM
      ds_load_b64 v[FA+mi*2:FA+mi*2+1], v13 offset:mi*256
      .set mi, mi+1
    .endr
    s_wait_dscnt 0x0
    // RELEASE the A slot now (A is in our regs, before WMMA): cons_a[cid] = a_step+1
    s_add_u32 s54, s54, 1
    v_mov_b32 v13, s53
    lds_put_v v13, s54
.else
    // direct-load FM own A frags; advance A saddrs +16 (next K16)
    .set mi, 0
    .rept FM
      global_load_b64 v[FA+mi*2:FA+mi*2+1], v8, s[40+2*mi:41+2*mi]
      .set mi, mi+1
    .endr
    .set mi, 0
    .rept FM
      s_add_u32  s[40+2*mi], s[40+2*mi], 16
      s_addc_u32 s[41+2*mi], s[41+2*mi], 0
      .set mi, mi+1
    .endr
    s_wait_loadcnt 0x0
.endif
    mark 64, s56                                // DIAGFINE occ[16] = max step with A global_load'd (about to WMMA)
    // FM*FN accumulating WMMA
    .set mi, 0
    .rept FM
      .set ni, 0
      .rept FN
        v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+mi*2:FA+mi*2+1], v[FB+ni*2:FB+ni*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
        .set ni, ni+1
      .endr
      .set mi, mi+1
    .endr
    mark 68, s56                                // DIAGFINE occ[17] = max step with WMMA accumulation done
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s8                        // k < KT ?
    s_cbranch_scc1 .Lcons_k
    mark 28, 5
    mark 48, s56                               // DIAG: occ[12] = global cons step at K-loop EXIT (1024=exited last tile -> store hang ; 992=stuck in K-loop tail)

    // ---- STORE all FM*FN fp32 frags: C + ti*(P*FM*FN*1024) + cid*(FM*FN*1024) + frag*1024 ----
    s_mul_i32 s27, s17, (P*FM*FN*1024)
    s_mul_i32 s28, s47, (FM*FN*1024)
    s_add_u32 s27, s27, s28
    s_add_u32 s28, s6, s27
    s_addc_u32 s29, s7, 0
    .set frag, 0
    .rept FM*FN
      global_store_b128 v10, v[ACC+frag*8:ACC+frag*8+3], s[28:29] offset:(frag*1024)
      global_store_b128 v10, v[ACC+frag*8+4:ACC+frag*8+7], s[28:29] offset:(frag*1024+16)
      .set frag, frag+1
    .endr
    // ---- terminal? decide BEFORE the store-wait. The per-tile s_wait_storecnt exists ONLY to stop the NEXT
    //   tile's ACC re-zero from clobbering in-flight store data (a WAR hazard). The FINAL tile has no next
    //   tile, so it skips the wait and retires immediately -- the dispatch end-of-pipe fence drains the
    //   in-flight stores for host readback. (FIX: a lone terminal wave's s_wait_storecnt never drains without
    //   concurrent WG memory traffic -> it hung forever. Signal done + move on, per the WAR being moot here.) ----
    s_add_u32 s57, s57, 1                       // counted one more tile (issued its stores)
    mark 44, s57                                // DIAG: occ[11] = tiles the compute COMPLETED (issued+retired)
.if DSWS
    // ---- Phase-2 per-WG segments_processed bump (the controller's decision clock): lead compute (cid==0)
    //   owns it (POOL=1: one tile = one WG segment, all cids lock-step on the broadcast ti). Single writer.
    //   occ_a/occ_b are sampled at the per-K CONSUME points (where the ring is mid-flight), NOT here -- at
    //   the segment boundary the ring has fully DRAINED so occ reads ~0, which is both uninformative for
    //   validation and the wrong signal for the eventual controller (would always read "starved"). ----
    s_cmp_eq_u32 s47, 0
    s_cbranch_scc0 .Lseg_done
    lds_get s44, SEGCNT_OFF
    s_add_u32 s44, s44, 1
    lds_put SEGCNT_OFF, s44
.Lseg_done:
    // ---- T3.1 UNIT (no actuation): every compute wave races the (compute->Afeed, dir=0) ticket each segment.
    //   The LDS-CAS in try_gate must yield <=1 winner per (dir,epoch) across all NCOMP racers -> the would-win
    //   counter occ[39] should land near #epochs (= segcnt>>EPOCH_SHIFT range), NOT NCOMP*#epochs. This is the
    //   harness-side validation of the gate-CAS before 3.2/3.3 wire real role conversions onto \swin. ----
    try_gate 0, s59
    s_cmp_eq_u32 s59, 1
    s_cbranch_scc0 .Ltg_unit_done
    mark_inc 156                                 // occ[39] = would-win count (compute->Afeed ticket)
.Ltg_unit_done:
.endif
    s_cmp_lt_u32 s57, s11                        // processed < TOTAL ?
    s_cbranch_scc0 .Lcompute_exit                // terminal -> skip store-wait; dispatch fence drains the stores
    s_wait_storecnt 0x0                          // non-terminal: stores MUST drain before the next tile re-zeros ACC
    mark 52, s56                               // DIAG: occ[13] = cons step past a NON-terminal store-wait
    mark 28, 6
    // ---- SHRINK back to lean (per tile) ----
.if DYNVGPR
    s_wait_loadcnt 0x0                          // drain BEFORE realloc: no op may reference a VGPR being freed
    s_wait_storecnt 0x0                         //   (defense-in-depth; the next-tile grow also drains)
.Lshrink_retry:
    s_alloc_vgpr 32                             // SCC-retry guard on EVERY s_alloc_vgpr (brick-class rule), even
    s_cbranch_scc0 .Lshrink_retry              //   though a shrink/dealloc realistically never rejects
.endif
    mark 28, 7
    s_branch .Lcompute_loop
.Lcompute_exit:
    mark 28, 8                                 // terminal: compute's lone C store historically didn't drain -> EOP
.if STOREWAIT
    s_wait_storecnt 0x0                         // STOREWAIT: drain the terminal C store to L2 BEFORE retire so the EOP
                                               //   RELEASE_MEM fires -> queue idle -> clean teardown (pool>=2 brick fix)
.endif
    s_endpgm                                   //   (NOFENCE path still completes on occ[0]==0 regardless of the fence)

.if DSWS
    // ========================================================================================
    // DSWS B-FEED (frag-partitioned producers). wid 0 = LEAD: sole tile-claimer + ti/epoch broadcaster.
    //   Every B-feed wave (b_id=wid) owns the strided frag subset {ni : ni%NBFEED==b_id} of the FN shared
    //   frags, fills them into the shared B-ring slot, and bumps ITS OWN prod_b[b_id]. Compute gates on
    //   min(prod_b[*]) so no B-feed ever waits on another B-feed (consumer-side min = barrier-free).
    // ========================================================================================
.Lbfeed_role:
    mark 24, 1                                  // DIAG occ[6] feedPhase=1: B-feed role entered
    s_mov_b32 s55, 0                            // GLOBAL prod step (cumulative; monotonic, never reset)
  .if DYNVGPR
.Lbfeed_lean_alloc:
    s_alloc_vgpr 32                             // commit this lean B-feed at 32 (dyn WG-allocator consistency)
    s_cbranch_scc0 .Lbfeed_lean_alloc
  .endif
    // &prod_b[b_id]: lead (b_id==0) -> PROD_OFF ; else -> PRODB_HI_OFF + (b_id-1)*4
    s_sub_u32 s50, s24, 1
    s_lshl_b32 s50, s50, 2
    s_add_u32 s50, s50, PRODB_HI_OFF
    s_mov_b32 s48, PROD_OFF
    s_cmp_eq_u32 s24, 0
    s_cselect_b32 s50, s48, s50
    s_mov_b32 s35, 0                            // local epoch (non-lead follow clock)
.Lbfeed_loop:
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lbfeed_follow               // non-lead: follow the lead's broadcast
    // ---- LEAD: claim BATCH=1 tile (the ONE device-scope atomic), publish ti, bump epoch ----
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lbfeed_after_grab
    v_mov_b32 v3, BATCH
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lbfeed_after_grab:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v5               // ti
    lds_put TI_OFF, s17                          // publish ti FIRST...
    lds_get s44, EPOCH_OFF
    s_add_u32 s44, s44, 1
    s_mov_b32 s35, s44
    lds_put EPOCH_OFF, s44                       // ...then bump epoch (followers see ti before the bump)
    s_branch .Lbfeed_have_ti
.Lbfeed_follow:
  .if DYNVGPR
    s_sleep SLEEPN
  .endif
    lds_get s44, EPOCH_OFF
    s_cmp_eq_u32 s44, s35
    s_cbranch_scc1 .Lbfeed_follow               // spin until the lead bumps epoch
    s_mov_b32 s35, s44
    lds_get s17, TI_OFF
.Lbfeed_have_ti:
    s_cmp_ge_u32 s17, s11                        // terminal tile? (broadcast already done -> all roles retire)
    s_cbranch_scc1 .Lfeed_exit
.if SAFEPROBE
    // SAFEPROBE: clamp the broadcast ti -> [0,TOTAL-1] before B-saddr decode (mirrors compute lines 471-472).
    //   Pairs with the common-prologue v9 vaddr clamp so EVERY B global address is provably in-buffer even if
    //   the broadcast read returns a garbage ti (defense-in-depth; terminal test above already retires ti>=TOTAL).
    s_sub_u32 s36, s11, 1                         // TOTAL-1  (s36 dead here; B-feed recomputes it nowhere)
    s_min_u32 s17, s17, s36
.endif
    mark 24, 2                                    // DIAG occ[6] feedPhase=2: B-feed has a (clamped) ti
    mark 32, s17                                  // DIAG occ[8] feedTi: the broadcast/claimed ti B is producing
    // ---- B col-tile saddr (k=0): Bshuf + tcol*(FN*256) ----
    s_mul_hi_u32 s19, s17, s12
    s_mul_i32  s18, s19, s13
    s_sub_u32  s18, s17, s18                     // tcol
    s_mul_i32 s20, s18, s14
    s_add_u32 s20, s4, s20
    s_addc_u32 s21, s5, 0
    s_mov_b32 s26, 0                            // per-tile k
.Lbprod_k:
    // slot-free gate (GLOBAL step) on min compute cons: don't overwrite an undrained slot
    s_cmp_lt_u32 s55, RINGD
    s_cbranch_scc1 .Lbslot_ok
    s_sub_u32 s45, s55, RINGD
.Lbwait_slot:
  .if DYNVGPR
    s_sleep SLEEPN
  .endif
    min_cons s44
    s_cmp_le_u32 s44, s45
    s_cbranch_scc1 .Lbwait_slot
.Lbslot_ok:
    mark 72, s55                                 // DIAGFINE occ[18] slotok: B step that passed the slot-free gate
    s_and_b32 s45, s55, (RINGD-1)
    s_mul_i32 s45, s45, (FN*256)
    v_add_nc_u32 v13, v9, s45                    // ring vaddr base (lane*8 + slot)
    // pass 1: global_load_tr owned frags into staging (compile-time owner; runtime guard on b_id)
    .set ni, 0
    .rept FN
      .set owner, ni % NBFEED
      s_cmp_eq_u32 s24, owner
      s_cbranch_scc0 1f
      global_load_tr_b64 v[BSTG+ni*2:BSTG+ni*2+1], v9, s[20:21] offset:ni*256
1:
      .set ni, ni+1
    .endr
    s_wait_loadcnt 0x0
    mark 76, s55                                 // DIAGFINE occ[19] loadtr: B step global_load_tr'd into staging
    // pass 2: ds_store owned frags into the ring slot
    .set ni, 0
    .rept FN
      .set owner, ni % NBFEED
      s_cmp_eq_u32 s24, owner
      s_cbranch_scc0 2f
      ds_store_b64 v13, v[BSTG+ni*2:BSTG+ni*2+1] offset:ni*256
2:
      .set ni, ni+1
    .endr
    s_wait_dscnt 0x0
    mark 80, s55                                 // DIAGFINE occ[20] dsstore: B step ds_store'd into ring (pre-publish)
    // publish prod_b[b_id] = s55+1 (this wave's owned frags for this step are now visible)
    s_add_u32 s55, s55, 1
    v_mov_b32 v13, s50
    lds_put_v v13, s55
    mark 84, s55                                 // DIAGFINE occ[21] publish: max prod_b count PUBLISHED by B-feed
    s_add_u32 s20, s20, s10                      // advance B saddr += NT*256
    s_addc_u32 s21, s21, 0
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s8
    s_cbranch_scc1 .Lbprod_k
    // lead drains (waits compute consumed) before next claim; non-lead just loops to wait next epoch
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lbfeed_loop
.Lbdrain:
  .if DYNVGPR
    s_sleep SLEEPN
  .endif
    min_cons s44
    s_cmp_lt_u32 s44, s55
    s_cbranch_scc1 .Lbdrain
    s_branch .Lbfeed_loop

    // ========================================================================================
    // DSWS A-FEED (band-partitioned producers). FOLLOWS the lead's ti/epoch broadcast (like compute).
    //   A-feed wave a_id owns the strided band subset {bnd : bnd%NAFEED==a_id}; band bnd has ONE producer
    //   (this wave) and ONE consumer (compute bnd). prod_a[bnd] doubles as the band's cumulative step.
    // ========================================================================================
.Lafeed_role:
    s_sub_u32 s24, s24, NBFEED                   // a_id = wid - NBFEED
  .if DYNVGPR
.Lafeed_lean_alloc:
    s_alloc_vgpr 32
    s_cbranch_scc0 .Lafeed_lean_alloc
  .endif
    s_mov_b32 s35, 0                            // local epoch (follow)
    s_mov_b32 s60, 0                            // GLOBAL A-step (gk): cumulative across tiles, mirrors compute s54
                                                 //   and the B-ring's continuous step. The per-tile k (s26) is for
                                                 //   A-MATRIX addressing only; the ring protocol (publish/slot/gate)
                                                 //   MUST be global or tile>=1 re-publishes prod_a<=a_step -> A-starve.
.Lafeed_loop:
  .if DYNVGPR
    s_sleep SLEEPN
  .endif
    lds_get s44, EPOCH_OFF
    s_cmp_eq_u32 s44, s35
    s_cbranch_scc1 .Lafeed_loop
    s_mov_b32 s35, s44
    lds_get s17, TI_OFF
    s_cmp_ge_u32 s17, s11                        // terminal? -> retire (role-agnostic exit)
    s_cbranch_scc1 .Lfeed_exit
.if SAFEPROBE
    // SAFEPROBE: clamp the broadcast ti -> [0,TOTAL-1] before A-saddr decode (mirrors compute lines 471-472).
    //   s36 is dead here (the per-band loop recomputes it from scratch); pairs with the common-prologue v8
    //   vaddr clamp so EVERY A global address is provably in-buffer even on a garbage broadcast ti.
    s_sub_u32 s36, s11, 1                         // TOTAL-1
    s_min_u32 s17, s17, s36
.endif
    mark 112, 1                                  // DIAGFINE occ[28] Afeed: received broadcast ti, decoding bands
    s_mul_hi_u32 s19, s17, s12                   // trow = mul_hi(ti, magic)
    // ===== K-OUTER / band-INNER (the feed-starvation fix, 2026-06-29): the OLD band-outer/K-inner loop drained
    //   band b's ENTIRE K-loop before touching band b+NAFEED, so when an A-feed wave owns >1 band the not-yet-fed
    //   compute bands starved -> min_cons stalled -> the RINGD-deep B-ring jammed -> WG wedged at ~step RINGD
    //   (confirmed: 2c3a3b NAFEED>=NCOMP greened; 4c2a2b/6c1a1b hung). NOW: every K-step produces ALL owned bands
    //   one step, so all compute bands advance lock-step with the ring (mirrors B-feed's K-outer/frag-inner).
    //   With lock-step production astep == k for every owned band -> use s26 (k) directly; no per-band astep state.
    s_lshl_b32 s32, s9, 4                         // 16*K  (constant across the tile; per-frag stride)
    s_mov_b32 s26, 0                              // K-OUTER counter k (== astep for every owned band)
.LafeedK:
    .set bnd, 0
    .rept NCOMP
      .set owner, bnd % NAFEED
      s_cmp_eq_u32 s24, owner
      s_cbranch_scc0 3f                          // not owned by this A-feed wave -> skip for this k
      // slot-free gate (band bnd at GLOBAL step s60): if gk>=RINGD_A wait cons_a[bnd] > gk-RINGD_A
      s_cmp_lt_u32 s60, RINGD_A
      s_cbranch_scc1 6f
      s_sub_u32 s45, s60, RINGD_A
5:
  .if DYNVGPR
      s_sleep SLEEPN
  .endif
      lds_get s46, (CONS_A_OFF + bnd*4)
      s_cmp_le_u32 s46, s45
      s_cbranch_scc1 5b
6:
      // recompute A saddrs for (band bnd, step k): rowblk=trow*NCOMP+bnd ;
      //   saddr(mi) = A + rowblk*(16*FM)*K + mi*16*K + k*16
      s_mul_i32 s36, s19, NCOMP
      s_add_u32 s36, s36, bnd
      s_mul_i32 s22, s36, (16*FM)
      s_mul_i32 s22, s22, s9
      s_lshl_b32 s44, s26, 4                      // k*16
      s_add_u32 s22, s22, s44
      s_add_u32 s40, s2, s22
      s_addc_u32 s41, s3, 0
      .set mi2, 1
      .rept FM-1
        s_add_u32  s[40+2*mi2], s[40+2*(mi2-1)], s32
        s_addc_u32 s[41+2*mi2], s[41+2*(mi2-1)], 0
        .set mi2, mi2+1
      .endr
      // global_load FM A-frags
      .set mi3, 0
      .rept FM
        global_load_b64 v[ASTG+mi3*2:ASTG+mi3*2+1], v8, s[40+2*mi3:41+2*mi3]
        .set mi3, mi3+1
      .endr
      s_wait_loadcnt 0x0
      mark 116, s26                              // DIAGFINE occ[29] Afeed: step global_load'd from A
      // ds_store into A-ring band bnd, slot (GLOBAL gk & (RINGD_A-1)); base folds ARING_OFF+bnd*ABAND_STRIDE
      s_and_b32 s45, s60, (RINGD_A-1)
      s_mul_i32 s45, s45, (FM*256)
      s_add_u32 s45, s45, (ARING_OFF + bnd*ABAND_STRIDE)
      v_add_nc_u32 v13, v9, s45
      .set mi3, 0
      .rept FM
        ds_store_b64 v13, v[ASTG+mi3*2:ASTG+mi3*2+1] offset:mi3*256
        .set mi3, mi3+1
      .endr
      s_wait_dscnt 0x0
      mark 120, s26                              // DIAGFINE occ[30] Afeed: step ds_store'd into A-ring (pre-publish)
      // publish prod_a[bnd] = gk+1 (GLOBAL cumulative count -> matches compute's global a_step consumer)
      s_add_u32 s54, s60, 1
      lds_put (PROD_A_OFF + bnd*4), s54
      mark 124, s54                              // DIAGFINE occ[31] Afeed: max prod_a count PUBLISHED
3:
      .set bnd, bnd+1
    .endr
    s_add_u32 s26, s26, 1                         // per-tile k (A-matrix addressing)
    s_add_u32 s60, s60, 1                         // GLOBAL gk (ring protocol; never resets across tiles)
    s_cmp_lt_u32 s26, s8                          // k < KT ?
    s_cbranch_scc1 .LafeedK
    s_branch .Lafeed_loop
.endif   // DSWS feed bodies

    // ========================================================================================
    // FEED BODY (wave 0). Claims a tile, publishes (ti, reset counters, epoch++), produces the FN shared
    // B-frags per K16-step into the ring, then drains (waits all P compute) before claiming the next tile.
    // (v1 = tile-synchronous; cross-tile B-ring overlap = FUTURE ENHANCEMENT FE-1.) DSWS=1: DEAD (the DSWS
    //  role branch never reaches it) but kept assembled so .Lfeed_exit stays the shared retire path.
    // ========================================================================================
.Lfeed_role:
    s_mov_b32 s55, 0                            // GLOBAL prod step (cumulative B-steps published; MONOTONIC,
                                                 //   never reset -> the drain reads the true count, no stale-reset race)
.if DYNVGPR
    // ---- FEED PARTICIPATES in the dyn-VGPR WG allocation protocol (mirrors wavespec's loaders:
    //   occ_kernel_wavespec.s:206 `s_alloc_vgpr LEANREG`). The coop feed previously called NO s_alloc_vgpr
    //   at all -> a non-allocating wave in a dyn-armed (RSRC2 bit6) WG leaves the per-WG VGPR allocator
    //   inconsistent when its partner compute wave grows 32->112 -> page fault / MES wedge. THE 2-WAVE DYN FIX. ----
.Lfeed_lean_alloc:
    s_alloc_vgpr 32                             // commit feed at its lean size (SCC-retry guard; a lean alloc
    s_cbranch_scc0 .Lfeed_lean_alloc            //   should not reject, but every s_alloc_vgpr is guarded)
.endif
.Lfeed_loop:
    mark 24, 1
    // ---- claim BATCH=1 tile (the ONE device-scope atomic; mbgemm offset 20) ----
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_grab
    v_mov_b32 v3, BATCH
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lafter_grab:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v5               // ti (claim base; BATCH=1)
    mark 24, 2
    mark 32, s17

    // ---- publish ti, then bump epoch (MONOTONIC prod/cons: NO per-tile reset -- that was the stale-read
    //   race; the slot-free gate + cumulative drain handle flow control across the continuous ring) ----
    lds_put TI_OFF, s17
    // epoch++ : read, increment, write (feed is the only writer of epoch)
    lds_get s44, EPOCH_OFF
    s_add_u32 s44, s44, 1
    lds_put EPOCH_OFF, s44
    mark 24, 3
    // terminal tile? publish already done so compute exits too -> now feed exits.
    s_cmp_ge_u32 s17, s11
    s_cbranch_scc1 .Lfeed_exit

    // ---- B col-tile saddr (k=0): Bshuf + tcol*(FN*256) ----
    s_mul_hi_u32 s19, s17, s12                 // trow (unused) -- decode tcol
    s_mul_i32  s18, s19, s13
    s_sub_u32  s18, s17, s18                   // tcol
    s_mul_i32 s20, s18, s14                     // tcol * FN*256
    s_add_u32 s20, s4, s20
    s_addc_u32 s21, s5, 0                       // s[20:21] = B base (this tile, k=0)

    // ---- produce K-loop: s26 = per-tile k (B address); s55 = GLOBAL step (ring slot / gate / publish) ----
    s_mov_b32 s26, 0                           // per-tile k = 0
.Lprod_k:
    // slot-free gate (GLOBAL step): if s55 >= RINGD, spin until min_c cons > s55-RINGD
    s_cmp_lt_u32 s55, RINGD
    s_cbranch_scc1 .Lslot_ok
    s_sub_u32 s45, s55, RINGD                   // thresh = global_step - RINGD
.Lwait_slot:
.if DYNVGPR
    s_sleep SLEEPN                          // yield to the grown compute so it can release a ring slot
.endif
    min_cons s44
    s_cmp_le_u32 s44, s45
    s_cbranch_scc1 .Lwait_slot
.Lslot_ok:
    mark 72, s55                                // DIAGFINE occ[18] = max step that PASSED the slot-free gate (about to load_tr)
    // global_load_tr_b64 the FN shared B-frags into staging (lean block)
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[BSTG+ni*2:BSTG+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_loadcnt 0x0
    mark 76, s55                                // DIAGFINE occ[19] = max step with B global_load_tr'd into staging
    // ds_store into ring slot s = s55 & (RINGD-1)  (GLOBAL step -> continuous ring, no per-tile reset)
    s_and_b32 s45, s55, (RINGD-1)
    s_mul_i32 s45, s45, (FN*256)
    v_add_nc_u32 v13, v9, s45
    .set ni, 0
    .rept FN
      ds_store_b64 v13, v[BSTG+ni*2:BSTG+ni*2+1] offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_dscnt 0x0                            // B in LDS *first*...
    mark 80, s55                                // DIAGFINE occ[20] = max step with B ds_store'd into the ring (pre-publish)
    // publish: prod = s55+1 (GLOBAL, monotonic)
    s_add_u32 s55, s55, 1
    lds_put PROD_OFF, s55                       // ...then publish (any consumer seeing the bump reads valid B)
    mark 84, s55                                // DIAGFINE occ[21] = max prod count PUBLISHED by the feed
    // advance B saddr += NT*256 ; per-tile k++
    s_add_u32 s20, s20, s10
    s_addc_u32 s21, s21, 0
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s8
    s_cbranch_scc1 .Lprod_k
    mark 24, 5

    // ---- tile drain (CUMULATIVE): wait until min_c cons has reached the global step s55. cons is monotonic
    //   (never reset) so this reads the TRUE consumed count -- no stale-KT-from-the-previous-tile race. ----
.Ldrain:
.if DYNVGPR
    s_sleep SLEEPN                          // yield to the grown compute so it can consume + release (drain)
.endif
    min_cons s44
    s_cmp_lt_u32 s44, s55
    s_cbranch_scc1 .Ldrain
    mark 24, 6
    s_branch .Lfeed_loop

.Lfeed_exit:
    mark 24, 7
    // ---- end timer (leader) + live-- so the harness completion gate (occ[0]==0) can fire ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v5, s30
    global_atomic_max_u32 v4, v5, s[0:1] offset:12 scope:SCOPE_DEV
.Lafter_t1:
    s_mov_b32 exec_lo, s16
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfeed_done
    v_mov_b32 v3, -1
    global_atomic_add_u32 v4, v3, s[0:1] scope:SCOPE_DEV
.Lfeed_done:
    s_mov_b32 exec_lo, s16
.if PARKFEED
.Lfeed_park:                                   // ISOLATION: keep the feed wave RESIDENT (never retires). If the
    s_branch .Lfeed_park                        //   compute's last-tile store-wait then clears, feed s_endpgm is the cause.
.endif
.if STOREWAIT
    s_wait_storecnt 0x0                         // STOREWAIT: drain any pending global stores before the feed retires too
.endif
    s_endpgm
    .size occ_kernel, .-occ_kernel

// ---- RGADESC: analysis-only descriptor so `rga -s bin --co` can enumerate + livereg this kernel.
//   vgpr 256 ceiling so livereg reports the true s_alloc-grown peak-live. NOT emitted for the PM4 .bin. ----
.if RGADESC
.amdhsa_kernel occ_kernel
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 50
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_user_sgpr_count 15
    .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            occ_kernel
    .symbol:          occ_kernel.kd
    .kernarg_segment_size: 64
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .wavefront_size:  32
    .sgpr_count:      50
    .vgpr_count:      256
    .max_flat_workgroup_size: 128
    .args:            []
.end_amdgpu_metadata
.endif
