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
  .set G, 6            // cooperative M-extent (rowblks per super-tile) = NCOMP_MAX
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
.set SNAP_BASE,      (INITFLAG_OFF + 4)         // u32[6]: [parity*3 + {0:nC,1:nA,2:nB}] role-mix snapshots
.set QUIESCE_CNT_OFF,(SNAP_BASE + 6*4)          // u32 role-agnostic bail counter
.set DSWS2_STATE_END,(QUIESCE_CNT_OFF + 4)
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
.if LDS_TOTAL_DSWS2 > 32768
  .error "DSWS2 LDS layout exceeds 32768B group segment"
.endif
// Phase-B state must fit in the control gap below the resident region (inert compile check, no bytes).
.if DSWS2_STATE_END > BRES_OFF
  .error "DSWS2 Phase-B state (SNAP_BASE/QUIESCE_CNT) overlaps resident B region (BRES_OFF)"
.endif

.if DSWS2
  // ---- role counts (lifted from coop's `.ifndef NCOMP` etc., gated under DSWS2) ----
  .ifndef NCOMP
    .set NCOMP, 4                            // compute waves (fat, dyn-grow). Compute floor >= 1.
  .endif
  .ifndef NAFEED
    .set NAFEED, 2                           // A-feed waves (lean). Feed floor >= 1.
  .endif
  .ifndef NBFEED
    .set NBFEED, 2                           // B-feed waves (lean). Feed floor >= 1.
  .endif
  .set WAVES, (NCOMP + NAFEED + NBFEED)      // total waves launched per WG (harness dims must match)
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

// ============================================================================================
// Super-tile decode + resident A/B staging macros (A3..A6). Decode (Naming/symbols):
//   ksi = sti & mask ; t = sti >> shift ; mblk = t / NTL ; tcol = t % NTL.
//   FIX 1(d): n_kseg is ALWAYS a power of two (it's KT >> NKSEG_SHIFT, both compile-time-shift-derived),
//   so the sti->(t,ksi) split is an exact shift/mask -- no magic-div, no n_kseg==1 special-case (shift=0,
//   mask=0 falls out of the general path for free: ksi=0, t=sti). /NTL still goes via magic (s12),
//   unsigned-division mul_hi (coop GENDIV idiom), since NTL is not generally a power of two.
// ============================================================================================
.macro DECODE_STI                  // in: s17=sti, s67=mask, s68=shift ; out: s19=mblk s30=tcol s31=ksi ; clob: s18,s36
    s_and_b32    s31, s17, s67                // ksi = sti & mask
    s_lshr_b32   s18, s17, s68                // t   = sti >> shift
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
.if DSWS2_CONV
.macro occ_sample dst_a, dst_b               // out: \dst_a=occ_A in [0,G], \dst_b=occ_B in [0,FN]; clob s60,s61
    lds_get \dst_a, AROW_DONE_OFF            // prod_a: A rowblks resident (store-completion)
    lds_get \dst_b, BFRAG_DONE_OFF           // prod_b: B frags   resident (store-completion)
    lds_get s60,    ROWBLK_NEXT_OFF          // cons  : compute rowblk-claim consume clock
    s_min_u32  s61, s60, \dst_a              // cons_a = min(clock, prod_a)  (clamp -> no u32 underflow)
    s_sub_u32  \dst_a, \dst_a, s61           // occ_A  = prod_a - cons_a   in [0,G]
    s_min_u32  s61, s60, \dst_b              // cons_b = min(clock, prod_b)
    s_sub_u32  \dst_b, \dst_b, s61           // occ_B  = prod_b - cons_b   in [0,FN]
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
.ifndef BUDGET
  .set BUDGET, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)   // = VRESV_OFF init (conservation ceiling)
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
.endif

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

.if DSWS2
    // ===== DSWS v2 role branch (wid uniform per wave; scalar-only -> exec stays full for every role).
    //   wid == 0                    -> claimer (pinned super-tile broadcaster; A3)
    //   wid [0,NBFEED)              -> B-feed  (A4)
    //   wid [NBFEED,NBFEED+NAFEED)  -> A-feed  (A5)
    //   wid [NBFEED+NAFEED, WAVES)  -> compute (A6) =====
    // A1: every role label is just a distinct s_endpgm stub (unique s50 tag keeps them at distinct addresses).
    v_readfirstlane_b32 s24, v1               // wid (uniform per wave)
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc1 .Lclaimer
.if DSWS2_CONV && DIAG && DSWS2_TICKET_SELFTEST
    // Task 4 Step 3 -- try_gate single-winner SMOKE (assemble-only stub; default off). Every non-claimer
    //   wave races the (dir=0) ticket ONCE and atomic-adds its win (0/1) into occ[28] (byte offset 112,
    //   clear of the 0/20/24/104/108 control+probe words). On GPU (Task 6, if enabled) the sum should land
    //   near #epochs, NOT NCOMP*#epochs -- the harness-side proof the LDS-CAS yields <=1 winner/(dir,epoch).
    //   v4=0 (set in prologue), v2=lane; try_gate temps v5/v6/v7 are <=v15 (pre-grow safe). wid (s24)
    //   survives -- try_gate touches only s62..s65 / s16. NOTE: pre-init-rendezvous placement -> a real run
    //   reads gate/segcnt before the claimer publishes them; fine for an assemble/smoke stub.
    try_gate 0, s50
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ltg_selftest_skip
    v_mov_b32 v5, s50                          // win flag (0/1) for THIS wave
    global_atomic_add_u32 v4, v5, s[0:1] offset:112 scope:SCOPE_DEV   // occ[28] += win
.Ltg_selftest_skip:
    s_mov_b32 exec_lo, s16
.endif
    s_cmp_lt_u32 s24, NBFEED
    s_cbranch_scc1 .Lbfeed
    s_cmp_lt_u32 s24, (NBFEED+NAFEED)
    s_cbranch_scc1 .Lafeed
    s_branch .Lcompute

// ============================================================================================
//  A3 -- .Lclaimer : pinned wid-0. Owns the super-tile claim+broadcast, the SEGCNT clock, the
//    barrier-free LDS init, the completion live++/live-- (harness occ[0]==0 gate), AND -- being a
//    B-feed-class wave -- stages B for the current super-tile each iteration (A4 body via BSTAGE).
//
//  CLAIM-COUNTER & completion occ-buffer offsets (occ base = s0:s1; host zero-inits the whole buffer):
//    occ[0]  (offset 0)  = live counter (claimer +1 at entry, -1 at terminal; harness polls ==0)
//    occ[20] (offset 20) = GLOBAL super-tile claim counter (mirrors coop's tile-claim at offset:20)
//    (offsets 4/8/12/16 stay reserved for the coop-style maxlive/timers/total bookkeeping; unused here.)
// ============================================================================================
.Lclaimer:
.if DYNVGPR
.Lclaimer_alloc:
    s_alloc_vgpr 32                            // commit lean (dyn WG-allocator consistency); SCC-retry guard
    s_cbranch_scc0 .Lclaimer_alloc
.endif
    // live++ : lane0 occ[0] += 1
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lclaimer_live
    v_mov_b32 v3, 1
    global_atomic_add_u32 v4, v3, s[0:1] scope:SCOPE_DEV
.Lclaimer_live:
    s_mov_b32 exec_lo, s16
    // --- barrier-free LDS-control init: zero all control words; INITFLAG = 0xACED LAST ---
    lds_put STI_OFF, 0
    lds_put EPOCH_OFF, 0
    lds_put ROWBLK_NEXT_OFF, 0
    lds_put ROWBLK_DONE_OFF, 0
    lds_put BFRAG_NEXT_OFF, 0
    lds_put AROW_NEXT_OFF, 0
    lds_put BFRAG_DONE_OFF, 0
    lds_put AROW_DONE_OFF, 0
    lds_put NCOMP_SLOT, NCOMP
    lds_put NAFEED_SLOT, NAFEED
    lds_put NBFEED_SLOT, NBFEED
    lds_put GATE_OFF, 0
    lds_put (GATE_OFF+4), 0
    lds_put (GATE_OFF+8), 0
    lds_put (GATE_OFF+12), 0
    lds_put VRESV_OFF, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)
    lds_put SEGCNT_OFF, 0
.if DSWS2_CONV
    // Phase-B: seed BOTH epoch-parity role-mix snapshots with the launch mix, zero the quiesce counter.
    // Gated so DSWS2_CONV=0 emits ZERO new bytes -> byte-identical to the Phase-A green bin.
    lds_put QUIESCE_CNT_OFF, 0
    lds_put (SNAP_BASE + 0), NCOMP     // parity-0 snapshot = launch mix
    lds_put (SNAP_BASE + 4), NAFEED
    lds_put (SNAP_BASE + 8), NBFEED
    lds_put (SNAP_BASE + 12), NCOMP    // parity-1 = launch mix too (init)
    lds_put (SNAP_BASE + 16), NAFEED
    lds_put (SNAP_BASE + 20), NBFEED
.endif
    lds_put INITFLAG_OFF, 0xACED               // LAST: publishes "LDS ready" to all follower waves
    // FIX 1(e): load this dispatch's chunk terminal bound from occ[24] (host writes occW[6] per chunk;
    //   FIX 1j on the host side). All lanes read the same address -> no exec masking needed, just a
    //   plain broadcast load; stable for the WHOLE chunk, so load it ONCE here, not per-claim.
    global_load_b32 v6, v4, s[0:1] offset:24 scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_readfirstlane_b32 s69, v6                // s69 = chunkHi (this dispatch's terminal sti bound)
    s_mov_b32 s35, 0                           // claimer local epoch
.Lclaim_loop:
    // claim next sti: lane0 global_atomic_add occ[20] += 1, return old
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lclaim_grabbed
    v_mov_b32 v3, 1
    global_atomic_add_u32 v5, v4, v3, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lclaim_grabbed:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v5                // sti
    s_cmp_ge_u32 s17, s69                       // sti >= chunkHi (occ[24]) -> terminal
    s_cbranch_scc1 .Lclaimer_terminal
    DECODE_STI                                 // -> s19=mblk s30=tcol s31=ksi
    // reset per-super-tile claim/completion counters BEFORE the epoch bump (followers see them reset)
    lds_put ROWBLK_NEXT_OFF, 0
    lds_put ROWBLK_DONE_OFF, 0
    lds_put BFRAG_NEXT_OFF, 0
    lds_put AROW_NEXT_OFF, 0
    lds_put BFRAG_DONE_OFF, 0
    lds_put AROW_DONE_OFF, 0
    lds_put STI_OFF, s17                        // publish STI FIRST...
    lds_get s44, SEGCNT_OFF                     // bump SEGCNT (controller clock; +1/super-tile)
    s_add_u32 s44, s44, 1
    lds_put SEGCNT_OFF, s44
    lds_get s44, EPOCH_OFF                      // ...then bump EPOCH LAST
    s_add_u32 s44, s44, 1
    s_mov_b32 s35, s44
    lds_put EPOCH_OFF, s44
    BSTAGE                                      // claimer helps stage B for this super-tile (s30,s31)
    // A7 advance gate: free resident A/B only when ALL G rowblks are computed+flushed
.Lclaimer_wait_done:
    s_sleep SLEEPN
.if DSWS2_CONV
.if DIAG
    // Phase-B DIAG probe (Task 3): wid 0 samples the LIVE ring occupancy (compute is mid-drain here)
    //   and publishes the last-sampled occ_A/occ_B so a GPU run can confirm the sensor OSCILLATES
    //   rather than reading a stuck 0. Written every wait-done spin -> a live poller observes it vary.
    //   Spare word-indexed slots occ[26]/occ[27] -> byte offsets 104/108 (well clear of the
    //   byte-indexed control words occ[0]/occ[20]/occ[24]). READ-ONLY sensing, NO actuation.
    occ_sample s62, s63
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Locc_diag_skip
    v_mov_b32 v14, s62                          // occ_A (v14 <= v15: pre-grow safe)
    v_mov_b32 v15, s63                          // occ_B
    global_store_b32 v4, v14, s[0:1] offset:104 scope:SCOPE_DEV   // occ[26] = last occ_A
    global_store_b32 v4, v15, s[0:1] offset:108 scope:SCOPE_DEV   // occ[27] = last occ_B
.Locc_diag_skip:
    s_mov_b32 exec_lo, s16
.endif
.endif
    lds_get s44, ROWBLK_DONE_OFF                // (a) all G rowblks computed + flushed
    s_cmp_lt_u32 s44, G
    s_cbranch_scc1 .Lclaimer_wait_done
    lds_get s44, BFRAG_DONE_OFF                 //     all B frags stored
    s_cmp_lt_u32 s44, FN
    s_cbranch_scc1 .Lclaimer_wait_done
    lds_get s44, AROW_DONE_OFF                  //     all A rowblks staged
    s_cmp_lt_u32 s44, G
    s_cbranch_scc1 .Lclaimer_wait_done
    // (b) QUIESCE the CLAIM counters before reset: each role wave must have executed its terminal
    //   over-claim (fetch_add returns >=threshold, then bails) BEFORE we reset, else a descheduled
    //   straggler's next fetch_add returns 0 and claims index 0 of the NEXT super-tile against stale
    //   decode/resident state (round-table finding #1). Sentinels = threshold + #role-waves (each does
    //   exactly one terminal bail). NOTE: compile-time NCOMP/NAFEED/NBFEED is correct for STATIC roles;
    //   Phase-B conversion must switch these to live role counts / epoch-snapshot drained counters.
    lds_get s44, ROWBLK_NEXT_OFF                // G claims + NCOMP terminal bails
    s_cmp_lt_u32 s44, (G + NCOMP)
    s_cbranch_scc1 .Lclaimer_wait_done
    lds_get s44, BFRAG_NEXT_OFF                 // FN claims + NBFEED terminal bails
    s_cmp_lt_u32 s44, (FN + NBFEED)
    s_cbranch_scc1 .Lclaimer_wait_done
    lds_get s44, AROW_NEXT_OFF                  // G claims + NAFEED terminal bails
    s_cmp_lt_u32 s44, (G + NAFEED)
    s_cbranch_scc1 .Lclaimer_wait_done
    s_branch .Lclaim_loop
.Lclaimer_terminal:
    lds_put STI_OFF, 0xFFFFFFFF                 // FIX 1(e): publish SENTINEL (not the raw over-claimed sti)...
    lds_get s44, EPOCH_OFF                      // ...bump epoch so followers wake + retire (A7 terminal)
    s_add_u32 s44, s44, 1
    lds_put EPOCH_OFF, s44
    // live-- : lane0 occ[0] -= 1 (harness completion gate fires)
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lclaimer_dead
    v_mov_b32 v3, -1
    global_atomic_add_u32 v4, v3, s[0:1] scope:SCOPE_DEV
.Lclaimer_dead:
    s_mov_b32 exec_lo, s16
    s_endpgm

// ============================================================================================
//  A4 -- .Lbfeed : B-feed wave. Follows EPOCH/STI, decodes (tcol,ksi), stages its claimed B frags.
// ============================================================================================
.Lbfeed:
.if DYNVGPR
.Lbfeed_alloc:
    s_alloc_vgpr 32
    s_cbranch_scc0 .Lbfeed_alloc
.endif
.Lbfeed_init:
    s_sleep 1
    lds_get s44, INITFLAG_OFF
    s_cmp_eq_u32 s44, 0xACED
    s_cbranch_scc0 .Lbfeed_init                 // wait for the claimer's barrier-free LDS init
    s_mov_b32 s35, 0                            // local epoch
.Lbfeed_follow:
    s_sleep SLEEPN
    lds_get s44, EPOCH_OFF
    s_cmp_eq_u32 s44, s35
    s_cbranch_scc1 .Lbfeed_follow                // wait next super-tile (epoch change)
    s_mov_b32 s35, s44
    lds_get s17, STI_OFF
    s_cmp_eq_u32 s17, 0xFFFFFFFF                  // FIX 1(f): sentinel (A7) -> retire (was: STI>=TOTAL_super)
    s_cbranch_scc1 .Lretire
    DECODE_STI                                   // s30=tcol s31=ksi (mblk unused)
    BSTAGE
    s_branch .Lbfeed_follow

// ============================================================================================
//  A5 -- .Lafeed : A-feed wave. Follows EPOCH/STI, decodes (mblk,ksi), stages its claimed A rowblks.
// ============================================================================================
.Lafeed:
.if DYNVGPR
.Lafeed_alloc:
    s_alloc_vgpr 32
    s_cbranch_scc0 .Lafeed_alloc
.endif
.Lafeed_init:
    s_sleep 1
    lds_get s44, INITFLAG_OFF
    s_cmp_eq_u32 s44, 0xACED
    s_cbranch_scc0 .Lafeed_init
    s_mov_b32 s35, 0
.Lafeed_follow:
    s_sleep SLEEPN
    lds_get s44, EPOCH_OFF
    s_cmp_eq_u32 s44, s35
    s_cbranch_scc1 .Lafeed_follow
    s_mov_b32 s35, s44
    lds_get s17, STI_OFF
    s_cmp_eq_u32 s17, 0xFFFFFFFF                  // FIX 1(f): sentinel (A7) -> retire (was: STI>=TOTAL_super)
    s_cbranch_scc1 .Lretire
    DECODE_STI                                   // s19=mblk s31=ksi (tcol unused)
    ASTAGE
    s_branch .Lafeed_follow

// ============================================================================================
//  A6 -- .Lcompute : compute wave. Follows EPOCH/STI, decodes (mblk,tcol,ksi), waits resident A/B
//    fully staged (DONE counters), then claims rowblks, runs WMMA over the SEGK segment, and flushes
//    fp32 partials into C via global_atomic_add_f32 (split-K segments accumulate into the same C cell).
// ============================================================================================
.Lcompute:
.if DYNVGPR
.Lcompute_alloc:
    s_alloc_vgpr 32                             // lean baseline; grow per rowblk
    s_cbranch_scc0 .Lcompute_alloc
.endif
.Lcompute_init:
    s_sleep 1
    lds_get s44, INITFLAG_OFF
    s_cmp_eq_u32 s44, 0xACED
    s_cbranch_scc0 .Lcompute_init
    s_mov_b32 s35, 0
.Lcompute_follow:
    s_sleep SLEEPN
    lds_get s44, EPOCH_OFF
    s_cmp_eq_u32 s44, s35
    s_cbranch_scc1 .Lcompute_follow
    s_mov_b32 s35, s44
    lds_get s17, STI_OFF
    s_cmp_eq_u32 s17, 0xFFFFFFFF                  // FIX 1(f): sentinel (A7) -> retire (was: STI>=TOTAL_super)
    s_cbranch_scc1 .Lretire
    DECODE_STI                                  // s19=mblk s30=tcol s31=ksi
    // wait until resident A AND B fully STAGED (B: FN frags stored, A: G rowblks stored)
.Lcompute_staged:
    s_sleep SLEEPN
    lds_get s44, BFRAG_DONE_OFF
    s_cmp_lt_u32 s44, FN
    s_cbranch_scc1 .Lcompute_staged
    lds_get s44, AROW_DONE_OFF
    s_cmp_lt_u32 s44, G
    s_cbranch_scc1 .Lcompute_staged
    // C tile-term: ti = mblk*NTL + tcol ; ti*(G*FM*FN*1024)  (ksi-INDEPENDENT -> split-K accumulates)
    s_mul_i32 s38, s19, s13
    s_add_u32 s38, s38, s30
    s_mul_i32 s38, s38, (G*FM*FN*1024)
.Lcompute_claim:
    lds_fetch_add s33, ROWBLK_NEXT_OFF, 1       // claim rowblk r in [0,G)
    s_cmp_ge_u32 s33, G
    s_cbranch_scc1 .Lcompute_drained
.if DYNVGPR
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
.Lcompute_grow:
    s_alloc_vgpr NFV                            // grow (SCC-retry guarded, brick-class rule)
    s_cbranch_scc0 .Lcompute_grow
.endif
    // zero FM*FN fp32 accumulators
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
    // resident operand bases (lane*8 + region [+ r*FM*256 for A])
    v_add_nc_u32 v12, v9, BRES_OFF
    s_mul_i32 s37, s33, (FM*256)
    v_add_nc_u32 v13, v9, ARES_OFF
    v_add_nc_u32 v13, v13, s37
    // WMMA over the SEGK segment (KSEG_STEPS k-steps); read resident B(ks) + A(ks,r) from LDS
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
    // flush: C base = C + ti-term + r*(FM*FN*1024) ; per (frag,elem) atomic-add one fp32 (vaddr v10=lane*32)
    s_mul_i32  s39, s33, (FM*FN*1024)
    s_add_u32  s39, s38, s39
    s_add_u32  s28, s6, s39
    s_addc_u32 s29, s7, 0
    .set frag, 0
    .rept FM*FN
      .set e, 0
      .rept 8
        global_atomic_add_f32 v10, v[ACC+frag*8+e], s[28:29] offset:(frag*1024 + e*4) scope:SCOPE_DEV
        .set e, e+1
      .endr
      .set frag, frag+1
    .endr
    s_wait_storecnt 0x0                          // atomic-adds READ ACC -> must drain before shrink frees ACC
.if DYNVGPR
.Lcompute_shrink:
    s_alloc_vgpr 32                             // shrink (SCC-retry guarded)
    s_cbranch_scc0 .Lcompute_shrink
.endif
    lds_inc ROWBLK_DONE_OFF                      // rowblk r computed + flushed (frees the A7 advance gate)
    s_branch .Lcompute_claim
.Lcompute_drained:
    s_branch .Lcompute_follow                    // this super-tile's compute drained -> re-check epoch/terminal

// ---- A7 role-agnostic terminal (followers): retire. (Claimer retires via .Lclaimer_terminal.) ----
.Lretire:
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
    .amdhsa_group_segment_fixed_size 32768
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
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .wavefront_size:  32
    .sgpr_count:      72
    .vgpr_count:      256
    .max_flat_workgroup_size: 256
    .args:            []
.end_amdgpu_metadata
.endif
