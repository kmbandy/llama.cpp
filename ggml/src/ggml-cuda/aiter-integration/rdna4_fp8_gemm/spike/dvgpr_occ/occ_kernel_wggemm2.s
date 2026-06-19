// occ_kernel_wggemm2.s  (gfx1201, wave32) -- MAD-305 Phase 2: 4-WAVE COOPERATIVE fp8 WMMA GEMM,
// A-in-LDS + B-global_load_tr, per-wave static 4x4 accumulator. Built on the proven atomic-claim +
// LDS-broadcast foundation (occ_kernel_wglds.s). NO dyn-VGPR, NO B-LDS, NO b128-B (static G2 first).
//
// One claimed tile = logical 128x128 C tile. 4 waves (TWM=2 x TWN=2), wave wid owns the
// (wave_m, wave_n) 64x64 quadrant = FM x FN = 4x4 frags. TBK=32 (2 kk-steps/K-tile).
//
// Per claimed tile:
//   leader atomic_add-claims ti -> ds_store ti to LDS[TI_OFF] -> barrier -> all waves load ti.
//   zero acc[16][8] (v[32:159]).
//   for t in 0..NTILES-1 (NTILES=K/32):
//     cooperative A[128][32] fill into LDS[0..4095] (each lane 2x global_load_b128 -> ds_store_b128)
//     barrier
//     for kk in 0,1:
//       4 A-frags from LDS (ds_load_b64) + 4 B-frags (global_load_tr_b64); 16 WMMA
//       advance B saddr by NT*256 (next kt)
//     barrier   (before next A fill overwrites LDS)
//   store each wave's 16 frags FLAT (diagnostic layout): C[ti*65536 + wid*16384 + frag*1024].
//
// User data (USER_SGPR=15):
//   s0:s1=occ  s2:s3=A  s4:s5=Bshuf  s6:s7=C  s8=K(bytes/A-row)  s9=NT*256  s10=NTL_MASK  s11=NTL_LOG2
//   s12=NTILES(K/32)  s13=TOTAL
// LDS: [0..4095]=A tile (128x32 fp8), [4096..4099]=ti broadcast.  -> request 4100 B (units=9, 4608).
//
// VGPRs: v0 tid, v1 wid, v2 lane, v3 wave_m, v4 wave_n, v5/v6 scratch, v7=0, v8 ti,
//   v9 lane*8 (B vaddr), v10 lane*32 (C vaddr), v11 ldsbase (A frag read), v12 ldsoff0=tid*16,
//   v13 ldsoff1=tid*16+2048, v14 rc=(tid>>1)*K+(tid&1)*16, v24 TI_OFF=4096,
//   v16:19/v20:23 A-fill int4 bufs, acc v[32:159], fa v[160:167], fb v[168:175].

.ifndef STORE
    .set STORE, 1                          // 1 = full 16-frag diagnostic store (oracle). 0 = minimal:
.endif                                      //     only acc[0][0]/wave -> C[ti*4096+wid*1024] (perf: 16x less store traffic)
.ifndef DBUF
    .set DBUF, 1                           // 0 = single-buffer A (baseline). 1 = A ping-pong LDS, prefetch A(t+1) during compute(t)
.endif
.ifndef BDBUF
    .set BDBUF, 0                           // 1 = B register double-buffer (ping-pong bregs, prefetch B(t+1) one K-tile ahead,
.endif                                      //     unroll-by-2 for compile-time slots). A kept single-buffer here to isolate B-prefetch.
.ifndef BLDS
    .set BLDS, 0                            // 1 = B-IN-LDS with M-dedup (DBUF==0 path only): only wave_m==0 loads B from global
.endif                                      //   (global_load_tr -> ds_store), both wave_m ds_load the byte-identical frag back. Halves
                                            //   global B traffic for TWM=2 (M-waves share B). Frag is ROUND-TRIPPED (not re-derived from a
                                            //   plain layout) so no ds_load_tr / transpose-mapping risk. LDS += TWN*(2*FN*256) for the B region.
.ifndef NOBAR
    .set NOBAR, 0                           // 1 = skip the per-K-tile workgroup barriers (DBUF=0 path) -- INCORRECT (LDS race),
.endif                                      //     perf-only probe to attribute barrier cost.
.ifndef BLADDER
    .set BLADDER, 0                         // 1 = fine descending s_wait_loadcnt ladder on JIT-B (single-buffer A). HIP
.endif                                      //     WMMABUF_WAIT: load all 8 B frags up front, release 0x7->0x0 so the first
                                            //     WMMA fires when frag-0 lands. Implies DBUF=0 (build with DBUF=0).
.ifndef KWIN
    .set KWIN, 0                            // 0 = off. >=2 = A-LDS-ring K-WINDOW: publish KWIN A-slices into KWIN LDS
.endif                                      //     slots, ONE barrier, consume KWIN slices (32*KWIN WMMA), barrier before reuse.
                                            //     Amortizes the 4-wave A-publish barrier over KWIN K-tiles (correctness-preserving).
                                            //     LDS needs KWIN*ATILE + 4 B. NTILES must be divisible by KWIN. Overrides DBUF.
.ifndef KWINBPF
    .set KWINBPF, 0                         // KWIN consume: 1 = B-prefetch one slice ahead (2 B slots Bcur=176/Bnext=192,
.endif                                      //     loadcnt B-only since A is from LDS). Hides the per-slice B wait. 0 = simple.
.ifndef KWINPUB2
    .set KWINPUB2, 0                        // KWIN publish: 1 = overlap 2 A-slices' global loads per wait (v16-23 + v176-183)
.endif                                      //     -> KWIN/2 waits instead of KWIN. Halves publish A-load exposure. KWIN even.
.ifndef KWINPW
    .set KWINPW, 1                          // PUBLISH WIDTH: A-slices whose global loads are issued before each s_wait (MLP
.endif                                      //     depth). 1=plain, 2=old 2-wide. Slots w0->v16 w1->v176 w2->v192 w3->v200 (v192-207
.if KWINPUB2                                //     = field-26 headroom above v191; no occ cost). Must divide KWIN; max 4.
    .set KWINPW, 2                          // back-compat: KWINPUB2=1 == publish width 2
.endif
.if KWINPW > 4
    .error "KWINPW max 4 (only 4 register slots: v16/v176/v192/v200)"
.endif
.ifndef KWINNOBAR
    .set KWINNOBAR, 0                       // KWIN PERF PROBE: 1 = strip BOTH KWIN barriers (publish + tail). INCORRECT
.endif                                      //   (LDS race) -- wall-only, to attribute the barrier share of the 119->201 feed gap.
.ifndef KWINNOTAIL
    .set KWINNOTAIL, 0                      // KWIN: 1 = drop the tail (reuse) barrier; rely on the NEXT window's publish
.endif                                      //     barrier as the reuse fence. SINGLE-BUFFER -> POTENTIAL LDS RACE; gate hard with the
                                            //     full-fragment oracle (STORE=1) before trusting. 0 = safe 2-barrier.
.ifndef KUNROLL
    .set KUNROLL, 1                         // NOFEED inner-loop unroll: emit KUNROLL copies of the 2*FM*FN WMMA block back-to-back
.endif                                      //     before the loop backedge. Isolates backedge cost vs intrinsic WMMA-run density.
                                            //     Requires NTILES % KUNROLL == 0 (test shapes K{2048,4096,16384}/32 = {64,128,512} ok).
.ifndef ANOLDS
    .set ANOLDS, 0                          // LDS-FREE A (issue-density test): load A per-wave straight from
.endif                                      //     global (no LDS publish, no barriers, no A-tile LDS, no dscnt).
                                            //     A is K-contiguous so global_load_b64 IS the WMMA A-frag (no
                                            //     transpose). Simple per-slice loop (KWIN windowing moot w/o
                                            //     barriers). 2x A BW (2 wave_n dup; BW-abundant). Reuses s46-s53
                                            //     (PROFILE phase sums) -> not PROFILE-compatible. Gate w/ STORE=1.
.ifndef ANOLDSTR
    .set ANOLDSTR, 0                        // LDS-FREE A via global_load_tr (COALESCED): the FIX for ANOLDS's strided
.endif                                      //     catastrophe. A fed exactly like B -- from an A-shuf (mbg_preshuffle on
                                            //     A, M<->N) via global_load_tr: coalesced DRAM + HW transpose -> A-frag,
                                            //     no LDS, no barriers. Harness rebinds s2:3=Ashuf base, s14=MT*256.
.ifndef BAND
    .set BAND, 1                            // BAND-CLAIM (Step 2.5): ONE atomic_add(counter, BAND) per BAND tiles, then stride the band
.endif                                      //     with ZERO atomics. Cuts the global-atomic-claim contention that capped NOFEED. BAND=1 =
                                            //     legacy per-tile claim. The leader broadcasts base_ti once/band via LDS+barrier.
.ifndef FEEDONLY
    .set FEEDONLY, 0                        // FEED-ONLY probe (BLADDER path): keep the entire per-K-tile feed (A-fill, B-loads, barriers,
.endif                                      //     waits, fine ladder) but emit ZERO WMMA. If wall(FEEDONLY) ~= wall(FED), the feed alone is
                                            //     the whole wall (compute is already hidden) -> the loads are serial/latency-bound.
.ifndef PROFILE
    .set PROFILE, 0                         // 1 = sampled in-kernel per-phase realtime timers around the BLADDER FEEDONLY K-loop.
.endif                                      //     ONE global profiler wave (first to grab occ[6] token) accumulates 7 phase tick-sums
                                            //     (s47..s53) + K-tile count (s54) -> stored to occ[8..15]. Non-profilers branch over.
.ifndef STAGGER
    .set STAGGER, 0                        // rung 8: 1 = inert per-WG phase stagger at DBUF==1 K-loop entry to decorrelate
.endif                                      //     the inter-WG barrier lockstep. 0 = byte-identical to perf bin. NO timers/atomics.
.ifndef STAGGER_MASK
    .set STAGGER_MASK, 15                  // delay = ((ti*13 + wid*3) & STAGGER_MASK) << STAGGER_SHIFT busy-loop iters (per claimed tile)
.endif
.ifndef STAGGER_SHIFT
    .set STAGGER_SHIFT, 5                  // unit multiplier (5 = x32 iters). Sweep MASK {0,3,7,15,31}, SHIFT {4,5,6}. MASK=0 -> delay==0 control.
.endif
.ifndef PB
    .set PB, 0                             // rung 9: bisect the PROFILE 70x. 1=per-K sendmsg+kmcnt (all waves); 2=per-tile token atomic
.endif                                      //   (leader); 3=per-K cmp/branch skeleton (all waves); 4=per-K inert busy-loop (all waves). 0=off (byte-id).
.ifndef PB4_ITERS
    .set PB4_ITERS, 64                     // PB==4 per-K busy-loop iters (~3 cyc each)
.endif
// PT: at a phase boundary, profiler reads REALTIME, adds (now - prev) to \sumreg, sets prev=now.
.macro PT sumreg
.if PROFILE
    s_cmp_eq_u32 s55, 1
    s_cbranch_scc0 .Lpt_\@
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_sub_u32 s56, s30, s57
    s_add_u32 \sumreg, \sumreg, s56
    s_mov_b32 s57, s30
.Lpt_\@:
.endif
.endm
// ---- wave-grid: TWM x TWN waves cooperate on ONE claimed tile (TWM*FM*16 x TWN*FN*16). TWM/TWN
//      overridable via -defsym (powers of 2). Growing TWN amortizes the shared LDS A-strip over more
//      N-waves (A-reuse); growing TWM adds A rows reused over the same N. Larger tile = higher
//      arithmetic intensity = less feed/MAC -> the lever toward the 297 NOFEED ceiling. ----
.ifndef TWM
    .set TWM, 2
.endif
.ifndef TWN
    .set TWN, 2
.endif
.if TWM == 1
    .set TWM_LOG2, 0
.elseif TWM == 2
    .set TWM_LOG2, 1
.elseif TWM == 4
    .set TWM_LOG2, 2
.else
    .error "TWM must be 1, 2 or 4"
.endif
.if TWN == 1
    .set TWN_LOG2, 0
.elseif TWN == 2
    .set TWN_LOG2, 1
.elseif TWN == 4
    .set TWN_LOG2, 2
.else
    .error "TWN must be 1, 2 or 4"
.endif
.set WAVES,      TWM*TWN                    // waves per workgroup (4 @ 2x2, 8 @ 2x4, 16 @ 4x4)
.set WG_LOG2,    TWM_LOG2+TWN_LOG2+5        // log2(WAVES*32) = log2(workgroup threads)
.ifndef FM
    .set FM, 4                              // per-wave accumulator rows in frags (4 = 64-row quadrant; 2 = 32-row, 2x2 occ test)
.endif
.ifndef FN
    .set FN, 4                              // per-wave accumulator cols in frags
.endif
.set ACC,  32
.set AF,   160                             // (DBUF==0 4x4 path only) per-kk fa base
.set BF,   168                             // (DBUF==0 4x4 path only) per-kk fb base
// ---- derived tile geometry (so the SAME source builds 4x4 and 2x2 via -defsym FM/FN) ----
//   tile = (TWM*FM*16) x (TWN*FN*16) M x N; per-wave quadrant = (FM*16) x (FN*16); A-tile in LDS = TM*32 B.
.set TM,    TWM*FM*16                       // claimed-tile M rows (128 @ 4x4, 64 @ 2x2)
.set ATILE, TM*32                           // A-tile bytes in LDS (4096 @ 4x4, 2048 @ 2x2)
.set BTILE, TWN*(2*FN*256)                  // B-in-LDS per-slice bytes (TWN wave_n * 2 kk * FN frags * 256)
.if ANOLDS || ANOLDSTR
    .set TI_OFF, 0                          // LDS-FREE: no A tile; ti-broadcast scratch sits at byte 0 (~512B LDS total)
.elseif KWIN
  .if BLDS
    .set TI_OFF, KWIN*ATILE + KWIN*BTILE    // KWIN A-ring + KWIN B-ring, ti past both (KWIN=2: 8192+8192=16384)
  .else
    .set TI_OFF, KWIN*ATILE                 // KWIN A-slots; ti broadcast just past the ring
  .endif
.elseif DBUF
    .set TI_OFF, 2*ATILE                    // As[2] ping-pong; ti broadcast just past it
.else
    .set TI_OFF, ATILE                      // single-buffer As; ti broadcast at ATILE
.endif
// shift amounts (FM,FN are powers of 2): A-read wave_m*(FM*512); tile_row*(TM); B tile_col*(FN*512); wave_n*(FN*256)
.if FM == 8
    .set AROW_SH, 12                        // wave_m * 4096   (128-row quadrant half; FM=8 reuse tile, TWM=2 -> TM=256)
    .set TROW_SH, 8                         // tile_row * 256
.elseif FM == 4
    .set AROW_SH, 11                        // wave_m * 2048   (64-row quadrant half)
    .set TROW_SH, 7                         // tile_row * 128
.elseif FM == 2
    .set AROW_SH, 10                        // wave_m * 1024   (32-row quadrant half)
    .set TROW_SH, 6                         // tile_row * 64
.else
    .error "unsupported FM (only 2, 4, or 8)"
.endif
.if FN == 4
    .set WN_SH,   10                        // wave_n * (FN*256)=1024  (wave's own 64-col N-block)
.elseif FN == 2
    .set WN_SH,   9                         // wave_n * (FN*256)=512
.else
    .error "unsupported FN (only 2 or 4)"
.endif
// tile_col stride in Bshuf = full tile N-width = TWN * (FN*256) bytes -> shift = WN_SH + TWN_LOG2.
// (Bshuf is [kt][n_frag][16x16] tile-major = tile-size-agnostic; the kernel indexes absolute N-frags,
//  so this is the ONLY B-side change needed when TWN grows. @ 2x2: WN_SH+1 = old 11/10, byte-identical.)
.set TCOL_SH, WN_SH + TWN_LOG2
// A-shuf (ANOLDSTR) mirrors Bshuf with M<->N: wave_m*(FM*256) shift = WM_SH; tile_row stride = WM_SH + TWM_LOG2.
.if FM == 8
    .set WM_SH, 11                          // wave_m * (FM*256) = 2048
.elseif FM == 4
    .set WM_SH, 10                          // wave_m * (FM*256) = 1024
.else
    .set WM_SH, 9                           // FM==2: wave_m * (FM*256) = 512
.endif
.set ATCOL_SH, WM_SH + TWM_LOG2
// A-fill cooperative bands: WG has WAVES*32 threads (2/row), fills the TM-row A-strip in NBANDS passes.
//   NBANDS = (TM*32)/(WAVES*32*16) = FM/TWN. band row-stride = WAVES*16 rows; LDS band-stride = WAVES*512 B.
//   @ 4x4 TWN=2: NBANDS=2 (the old FM==4 two-band fill). @ TWN=4: NBANDS=1 (256 threads fill 128 rows, 1 pass).
.set NBANDS, FM/TWN
.if NBANDS < 1
    .error "TWN too large for FM (NBANDS<1): need TWN <= FM"
.endif
// B-IN-LDS region base: per wave_n slot = wave_n*(2*FN*256)+lane*8; per-slice stride = BTILE (KWIN ring).
//   KWIN path: B-ring just past the KWIN A-ring. DBUF==0 path: single B region past A tile + ti.
.if KWIN
    .set B_BLDS_OFF, KWIN*ATILE             // B-ring base (KWIN=2: 8192)
.else
    .set B_BLDS_OFF, ATILE + 512            // single-buffer B region past A + ti
.endif
.set BWN_SH,     WN_SH + 1                  // wave_n stride = 2*FN*256 (2 kk)  -> shift
.if BLDS && KWIN && KWINBPF
    .error "BLDS requires KWINBPF=0 (B-in-LDS uses the simple KWIN consume, not the B-prefetch path)"
.endif
// compacted frag bases for the NOFEED/BLADDER paths: fa right after acc, fb right after fa.
// (4x4 -> FA=160, FB=176 = unchanged; 2x2 -> FA=64, FB=72 so max VGPR ~80 -> ~16 waves/SIMD.)
.set FA, ACC + FM*FN*8                      // fa holds 2*FM frags (2 kk)
.set FB, FA + 2*FM*2                        // fb holds 2*FN frags (all B for the K-tile)

// ---- BSUB curB, prefB: one K-tile (index s26) for the BDBUF pipeline. A single-buffer (As[0]);
// B for THIS tile already resident in bregs[curB]; prefetch tile s26+1's B into bregs[prefB] (its
// latency hides behind the reuse barrier + next tile's A-fill, retired by the next A-fill's loadcnt
// wait). 32 WMMA from As[0] x bregs[curB]. curB/prefB are VGPR bases (176 or 192). ----
.macro BSUB curB, prefB
    s_lshl_b32 s27, s26, 5                   // t*32
    s_add_u32  s27, s25, s27                 // row_base_K + k0
    v_add_nc_u32 v15, v14, s27
    global_load_b128 v[16:19], v15, s[2:3]
    v_add_nc_u32 v15, v15, s29
    global_load_b128 v[20:23], v15, s[2:3]
    s_wait_loadcnt 0x0                        // A-fill done + prior B-prefetch into bregs[\curB] retired
    ds_store_b128 v12, v[16:19]
    ds_store_b128 v13, v[20:23]
    s_wait_dscnt 0x0
    s_barrier_signal -1
    s_barrier_wait -1
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        ds_load_b64 v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v11 offset:(mi*512 + kk*16)
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    s_wait_dscnt 0x0
    // prefetch tile s26+1's B -> bregs[\prefB] (s_Bbase already points at it); overlaps the WMMAs below
    s_add_u32  s42, s26, 1
    s_cmp_ge_u32 s42, s12
    s_cbranch_scc1 .Lno_bpf\@
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[\prefB+ni*2:\prefB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[\prefB+(FN+ni)*2:\prefB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_lshl_b32 s43, s9, 1
    s_add_u32  s20, s20, s43                  // s_Bbase -> next tile's B
    s_addc_u32 s21, s21, 0
.Lno_bpf\@:
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[\curB+(kk*FN+ni)*2:\curB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    s_barrier_signal -1
    s_barrier_wait -1                         // all waves done reading As[0] before next tile's A-fill
    s_add_u32  s26, s26, 1
.endm

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- identity ----
    v_lshrrev_b32 v1, 5, v0
    v_and_b32     v2, 31, v0
    v_lshrrev_b32 v3, TWN_LOG2, v1          // wave_m = wid >> log2(TWN)   (wid = wave_m*TWN + wave_n)
    v_and_b32     v4, (TWN-1), v1           // wave_n = wid & (TWN-1)
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9, 3, v2                 // v9 = lane*8 (B trfeed vaddr)
    v_lshlrev_b32 v10, 5, v2                // v10 = lane*32 (C store)
    v_mov_b32     v24, TI_OFF
.if BLDS
    // B-LDS staging base for this wave = B_BLDS_OFF + wave_n*(2*FN*256) + lane*8  (both wave_m of a wave_n share it)
    v_lshlrev_b32 v25, BWN_SH, v4
    v_add_nc_u32  v25, v25, v9
    v_add_nc_u32  v25, B_BLDS_OFF, v25
.endif
    // ---- A-fill per-thread invariants: ldsoff0=tid*16, ldsoff1=+2048, rc=(tid>>1)*K+(tid&1)*16 ----
    v_lshlrev_b32 v12, 4, v0                // ldsoff0 = tid*16
    v_add_nc_u32  v13, (TWM*TWN*512), v12   // ldsoff1 = tid*16 + WAVES*512 (band-2 LDS base; 2048 @ 2x2)
    v_lshrrev_b32 v5, 1, v0                 // r0 = tid>>1
    v_mul_lo_u32  v14, v5, s8               // r0*K
    v_and_b32     v6, 1, v0                 // c = tid&1
    v_lshlrev_b32 v6, 4, v6                 // c*16
    v_add_nc_u32  v14, v14, v6              // rc = r0*K + c*16
    // ---- A frag read base: ldsbase = wave_m*(FM*512) + (lane&15)*32 + ((lane>>4)&1)*8 ----
    v_lshlrev_b32 v11, AROW_SH, v3          // wave_m * (FM*512)  (2048 @ 4x4, 1024 @ 2x2)
    v_and_b32     v5, 15, v2                // lane&15
    v_lshlrev_b32 v5, 5, v5                 // (lane&15)*32
    v_add_nc_u32  v11, v11, v5
    v_bfe_u32     v5, v2, 4, 1              // (lane>>4)&1
    v_lshlrev_b32 v5, 3, v5                 // colhi*8
    v_add_nc_u32  v11, v11, v5

    // ---- admission: leader (tid==0) live++, maxlive, total++ ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v5, 1
    global_atomic_add_u32 v6, v7, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v6, v6, 1
    global_atomic_max_u32 v7, v6, s[0:1] offset:4 scope:SCOPE_DEV
    global_atomic_add_u32 v7, v5, s[0:1] offset:16 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s16
    // ---- timer t0 ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v5, s30
    global_atomic_min_u32 v7, v5, s[0:1] offset:8 scope:SCOPE_DEV
.Lafter_t0:
    s_mov_b32 exec_lo, s16
    // precompute band global row-stride = (WAVES*16)*K  (the A-fill band-2 row jump; 64*K @ 2x2)
    s_lshl_b32 s28, s8, (TWM_LOG2+TWN_LOG2+4)
    s_mov_b32 s29, s28                       // keep band-stride in s29 for the tile loop
.if PROFILE
    // ---- designate ONE global profiler wave (first lane0 to grab occ[6] token); init phase sums ----
    s_mov_b32 s55, 0
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lprof_np
    v_mov_b32 v5, 1
    global_atomic_add_u32 v6, v7, v5, s[0:1] offset:24 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lprof_np:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s56, v6              // old token (lane0); == 0 -> this wave is the profiler
    s_cmp_eq_u32 s56, 0
    s_cbranch_scc0 .Lprof_done
    s_mov_b32 s55, 1
.Lprof_done:
    s_mov_b32 s47, 0
    s_mov_b32 s48, 0
    s_mov_b32 s49, 0
    s_mov_b32 s50, 0
    s_mov_b32 s51, 0
    s_mov_b32 s52, 0
    s_mov_b32 s53, 0
    s_mov_b32 s54, 0
.endif

    // ============ BAND-CLAIM PERSISTENT LOOP (one atomic per BAND tiles; broadcast base once/band) ====
.Lclaim_loop:
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_claim
    v_mov_b32 v5, BAND                       // grab BAND contiguous tiles with ONE atomic
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    ds_store_b32 v24, v8                     // LDS[TI_OFF] = base_ti
    s_wait_dscnt 0x0
.Lafter_claim:
    s_mov_b32 exec_lo, s16
    s_barrier_signal -1
    s_barrier_wait -1
    ds_load_b32 v8, v24
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s38, v8             // base_ti (broadcast to all 4 waves)
    s_mov_b32 s39, 0                         // local band index
.Lband:
    s_add_u32  s17, s38, s39                 // ti = base_ti + local  (NO atomic inside the band)
    s_cmp_ge_u32 s17, s13
    s_cbranch_scc1 .Lexit

    // ---- decode tile_row/col, B col base, row_base_K ----
    s_lshr_b32 s18, s17, s11                 // tile_row
    s_and_b32  s19, s17, s10                 // tile_col
    // s_Bbase = Bshuf + tile_col*(FN*512) + wave_n*(FN*256)   (nt_base*256, nt_base=tile_col*(FN*2)+wave_n*FN)
    v_readfirstlane_b32 s22, v4             // wave_n (scalar)
    s_lshl_b32 s23, s19, TCOL_SH             // tile_col*(FN*512)  (2048 @ 4x4, 1024 @ 2x2)
    s_lshl_b32 s24, s22, WN_SH               // wave_n *(FN*256)   (1024 @ 4x4,  512 @ 2x2)
    s_add_u32  s23, s23, s24
    s_add_u32  s20, s4, s23
    s_addc_u32 s21, s5, 0
    // s_rowbaseK = tile_row*TM*K = (tile_row<<TROW_SH)*K
    s_lshl_b32 s25, s18, TROW_SH             // tile_row*TM  (128 @ 4x4, 64 @ 2x2)
    s_mul_i32  s25, s25, s8                  // *K -> row_base_K
    // ---- zero accumulators v[32:159] ----
    .set i, 0
    .rept FM*FN*8
      v_mov_b32 v[ACC+i], 0
      .set i, i+1
    .endr
.if ANOLDS && PROFILE
    .error "ANOLDS reuses PROFILE SGPRs s46-s53; build without PROFILE"
.endif
.if ANOLDS
    // ANOLDS per-tile A addressing. Per-lane vaddr v11 = (lane&15)*K + colhi*8 + slice*32 (slice term advanced
    //   +32/slice IN the K-loop -> 1 VALU/slice, vs 8 SALU to bump 4 SGPR bases). Reset here each tile.
.ifndef ACOAL
    .set ACOAL, 0                          // COALESCING DIAGNOSTIC: 1 = v11 = lane*8 (contiguous/coalesced A
.endif                                     //   access) instead of the strided (lane&15)*K frag layout. WRONG
                                           //   DATA (oracle will fail) -- perf-only, to isolate whether the
                                           //   strided/uncoalesced access is the ANOLDS regression cause.
.if ACOAL
    v_lshlrev_b32 v11, 3, v2               // lane*8 -> 256B contiguous/lane-group = COALESCED (wrong data)
.else
    v_and_b32     v5, 15, v2               // lane&15 (row within frag)
    v_mul_lo_u32  v11, v5, s8              // (lane&15)*K  (global A row stride in bytes) -> STRIDED/uncoalesced
    v_bfe_u32     v5, v2, 4, 1             // colhi=(lane>>4)&1
    v_lshlrev_b32 v5, 3, v5                // colhi*8
    v_add_nc_u32  v11, v11, v5             // v11 = (lane&15)*K + colhi*8  (slice=0)
.endif
    // FIXED per-tile A frag bases (NOT advanced; slice K carried by v11): s[46:47]=mi0 .. s[52:53]=mi3,
    //   each = A_base(s2:3) + tile_row*TM*K (s25) + wave_m*64*K + mi*16*K. s28=16*K stride, s29=wave_m*64*K temp.
    v_readfirstlane_b32 s29, v3            // wave_m (uniform per wave)
    s_lshl_b32 s29, s29, 6                 // wave_m*64
    s_mul_i32  s29, s29, s8                // *K
    s_add_u32  s29, s29, s25               // + tile_row*TM*K (row_base_K)
    s_add_u32  s46, s2, s29                // mi0 base lo
    s_addc_u32 s47, s3, 0                  // mi0 base hi
    s_lshl_b32 s28, s8, 4                  // 16*K (mi row stride)
    s_add_u32  s48, s46, s28
    s_addc_u32 s49, s47, 0                 // mi1
    s_add_u32  s50, s48, s28
    s_addc_u32 s51, s49, 0                 // mi2
    s_add_u32  s52, s50, s28
    s_addc_u32 s53, s51, 0                 // mi3
.endif
.if ANOLDSTR && PROFILE
    .error "ANOLDSTR reuses SGPRs s46-s49; build without PROFILE"
.endif
.if ANOLDS && ANOLDSTR
    .error "ANOLDS and ANOLDSTR are mutually exclusive"
.endif
.if ANOLDSTR
    // ANOLDSTR per-tile A-tr base: s[46:47] = Ashuf(s2:3) + (tile_row*(TWM*FM) + wave_m*FM)*256. Mirrors B's
    //   s20 base computation (M<->N). Advances +2*s14 (=2*MT*256) per slice. s14 = MT*256 (A kt stride in Ashuf).
    v_readfirstlane_b32 s29, v3            // wave_m (uniform)
    s_lshl_b32 s28, s18, ATCOL_SH          // tile_row*(TWM*FM*256)
    s_lshl_b32 s29, s29, WM_SH             // wave_m*(FM*256)
    s_add_u32  s28, s28, s29
    s_add_u32  s46, s2, s28                // Ashuf + mt_base*256
    s_addc_u32 s47, s3, 0
.endif
.ifndef NOFEED
    .set NOFEED, 0
.endif
.if ANOLDS
    // ===== LDS-FREE A: each wave loads its own A frags straight from global (no LDS, no barriers). The A
    //   staged-in-LDS optimizes BANDWIDTH (cross-wave reuse); we are ISSUE-bound, so the publish + barriers
    //   are overhead for the wrong bottleneck. A is K-contiguous -> global_load_b64 IS the WMMA A-frag.
    //   Simple per-slice loop; B path unchanged (global_load_tr). loadcnt-only (no dscnt). =====
    s_mov_b32 s26, 0                          // t = slice index (0..NTILES-1)
.Lkt_loop:
    .set mi, 0                                // A: 4 mi x 2 kk = 8 global_load_b64 (v11=lane vaddr, s[46+mi*2]=base)
    .rept FM
      global_load_b64 v[160+(0*FM+mi)*2:160+(0*FM+mi)*2+1], v11, s[46+mi*2:46+mi*2+1] offset:0    // kk=0
      global_load_b64 v[160+(1*FM+mi)*2:160+(1*FM+mi)*2+1], v11, s[46+mi*2:46+mi*2+1] offset:16   // kk=1
      .set mi, mi+1
    .endr
    .set ni, 0                                // B: 2 kk-groups of FN frags (kk1 @ s20+s9), unchanged
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_loadcnt 0x0                        // all A + B global loads landed (loadcnt only; no LDS -> no dscnt)
    .set kk, 0                                // 32 WMMA
    .rept 2
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    s_lshl_b32 s43, s9, 1                     // advance B base by 2*NT*256 (1 K-tile)
    s_add_u32  s20, s20, s43
    s_addc_u32 s21, s21, 0
    v_add_nc_u32 v11, v11, 32                  // advance A slice-K (next slice); FIXED s[46..53] bases unchanged
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lkt_loop
.elseif ANOLDSTR
    // ===== LDS-FREE A via global_load_tr (COALESCED -- the fix for ANOLDS's strided catastrophe). A fed EXACTLY
    //   like B: from an A-shuf via global_load_tr (coalesced DRAM read + HW transpose -> the A-frag). No LDS, no
    //   barriers. A base s46:47 (Ashuf + mt_base*256), kk1 @ +s14 (MT*256); B base s20:21 (Bshuf), kk1 @ +s9. =====
    s_mov_b32 s26, 0                          // t = slice index (0..NTILES-1)
.Lkt_loop:
    .set mi, 0                                // A frags: FM via global_load_tr (kk0 @ s46, kk1 @ s46+s14), offset mi*256
    .rept FM
      global_load_tr_b64 v[160+mi*2:160+mi*2+1], v9, s[46:47] offset:mi*256
      .set mi, mi+1
    .endr
    s_add_u32  s48, s46, s14
    s_addc_u32 s49, s47, 0
    .set mi, 0
    .rept FM
      global_load_tr_b64 v[160+(FM+mi)*2:160+(FM+mi)*2+1], v9, s[48:49] offset:mi*256
      .set mi, mi+1
    .endr
    .set ni, 0                                // B frags: FN via global_load_tr (kk0 @ s20, kk1 @ s20+s9), unchanged
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_loadcnt 0x0                        // all A + B coalesced transpose loads landed (loadcnt only)
.if FEEDONLY == 0
    .set kk, 0                                // 32 WMMA (FEEDONLY=1 strips these -> isolate feed throughput vs WMMA serialization)
    .rept 2
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
.endif
    s_lshl_b32 s43, s14, 1                     // advance A base by 2*MT*256 (1 K-tile)
    s_add_u32  s46, s46, s43
    s_addc_u32 s47, s47, 0
    s_lshl_b32 s43, s9, 1                      // advance B base by 2*NT*256 (1 K-tile)
    s_add_u32  s20, s20, s43
    s_addc_u32 s21, s21, 0
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lkt_loop
.elseif NOFEED
    // ===== NOFEED compute-ceiling probe: fill A once, read 8 A + 8 B frags ONCE, K-loop = 32 WMMA only
    //       (no per-K feed, no barriers). Result is garbage; measures the wave-group WMMA-only ceiling. =====
    v_add_nc_u32 v15, v14, s25
    global_load_b128 v[16:19], v15, s[2:3]
.if NBANDS > 1
    v_add_nc_u32 v15, v15, s29              // +band-stride (second A-strip band; NBANDS=2 only @ 2x2/4x4)
    global_load_b128 v[20:23], v15, s[2:3]
.endif
    s_wait_loadcnt 0x0                      // ALL A-fill loads landed before any ds_store
    ds_store_b128 v12, v[16:19]
.if NBANDS > 1
    ds_store_b128 v13, v[20:23]
.endif
    s_wait_dscnt 0x0
    s_barrier_signal -1
    s_barrier_wait -1
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        ds_load_b64 v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v11 offset:(mi*512 + kk*16)
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32 s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_loadcnt 0x0
    s_wait_dscnt 0x0
    s_mov_b32 s26, 0
.Lkt_loop:
    // KUNROLL copies of the 2*FM*FN WMMA block back-to-back (no feed) -> (KUNROLL*2*FM*FN) WMMA per backedge
    .rept KUNROLL
      .set kk, 0
      .rept 2
        .set mi, 0
        .rept FM
          .set ni, 0
          .rept FN
            v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
            .set ni, ni+1
          .endr
          .set mi, mi+1
        .endr
        .set kk, kk+1
      .endr
    .endr
    s_add_u32 s26, s26, KUNROLL
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lkt_loop
.elseif BDBUF
    // ===== B register double-buffer (unroll-by-2) + A single-buffer. bregs slot0=v176-191, slot1=v192-207 =====
    // prologue: load tile0 B -> bregs[0]; advance s_Bbase to tile 1
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_lshl_b32 s43, s9, 1
    s_add_u32  s20, s20, s43                  // s_Bbase -> tile 1's B
    s_addc_u32 s21, s21, 0
    s_mov_b32 s26, 0                          // t = 0
.Lkt_loop:
    BSUB 176, 192                             // tile t  (B-slot0), prefetch t+1 -> slot1; s26 -> t+1
    s_cmp_ge_u32 s26, s12
    s_cbranch_scc1 .Lkt_done
    BSUB 192, 176                             // tile t+1 (B-slot1), prefetch t+2 -> slot0; s26 -> t+2
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lkt_loop
.Lkt_done:
.elseif BLADDER
    // ===== Step 2.2/2.3: FINE descending s_wait_loadcnt ladder on JIT-B (single-buffer A). HIP WMMABUF_WAIT.
    //   Load all 2*FN B frags up front (loadcnt -> 2*FN), then release frag-by-frag (2*FN-1)->0x0 so the
    //   first FM WMMA fire when frag-0 lands instead of waiting all (coarse 0x0). A stays resident in LDS,
    //   gated by dscnt ONLY -- never mixed with B's loadcnt. Compacted frags: fa@FA, fb@FB (2x2 -> ~80 VGPR). =====
    s_mov_b32 s26, 0                          // t = 0
.Lkt_loop:
.if PROFILE
    s_cmp_eq_u32 s55, 1
    s_cbranch_scc0 .Lpt_top
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s46, s30                          // prev = K-tile top timestamp
    s_add_u32 s54, s54, 1                        // K-tile count++
.Lpt_top:
.endif
    // ---- single-buffer A fill: As[tid*16] (rows 0..TM-1); 4x4 adds a 2nd 64-row half ----
    s_lshl_b32 s27, s26, 5                     // t*32
    s_add_u32  s27, s25, s27                   // row_base_K + k0
    v_add_nc_u32 v15, v14, s27                 // voff0 = rc + s_rowk0
    global_load_b128 v[16:19], v15, s[2:3]
.if NBANDS > 1
    v_add_nc_u32 v15, v15, s29                 // voff1 = +band-stride (second A-strip band; NBANDS=2 only)
    global_load_b128 v[20:23], v15, s[2:3]
.endif
    s_wait_loadcnt 0x0                          // ALL A-fill loads landed (no B in flight yet -> loadcnt pure-A)
    PT s47                                       // phase 1: A global-load span
    ds_store_b128 v12, v[16:19]
.if NBANDS > 1
    ds_store_b128 v13, v[20:23]
.endif
    s_wait_dscnt 0x0
    PT s48                                       // phase 2: A ds_store + dscnt span
    s_barrier_signal -1
    s_barrier_wait -1
    PT s49                                       // phase 3: A publish barrier span
    // ---- issue ALL 2*FN B frags up front (loadcnt -> 2*FN): kk0 @ s[20:21], kk1 @ s[20:21]+NT*256 ----
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    // ---- A frag reads from LDS (2*FM frags), waited on dscnt ONLY -- B keeps loading in background ----
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        ds_load_b64 v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v11 offset:(mi*512 + kk*16)
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    s_wait_dscnt 0x0                            // A frags resident; B's loadcnt still counting 2*FN in flight
    PT s50                                       // phase 4: B global_load_tr issue + A ds_load (dscnt) span
    // ---- descending B ladder (2*FN-1)->0x0 + 2*FM*FN WMMA (two back-to-back FM*FN-WMMA kk groups, ni-outer/mi-inner) ----
    .set kk, 0
    .rept 2
      .set ni, 0
      .rept FN
        s_wait_loadcnt ((2*FN - 1) - (kk*FN + ni))   // frag (kk,ni) is the (kk*FN+ni)-th load; release when landed
.if FEEDONLY == 0
        .set mi, 0
        .rept FM
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set mi, mi+1
        .endr
.endif
        .set ni, ni+1
      .endr
      .set kk, kk+1
    .endr
    PT s51                                       // phase 5: descending B s_wait_loadcnt ladder span
    // ---- advance B base by 2*NT*256 (consumed 2 kt), barrier before next A fill overwrites LDS, t++ ----
    s_lshl_b32 s43, s9, 1
    s_add_u32  s20, s20, s43
    s_addc_u32 s21, s21, 0
    s_barrier_signal -1
    s_barrier_wait -1
    PT s52                                       // phase 6: tail barrier span
    s_add_u32  s26, s26, 1
    s_cmp_lt_u32 s26, s12
    PT s53                                       // phase 7: loop bookkeeping span
    s_cbranch_scc1 .Lkt_loop
.elseif KWIN
    // ===== A-LDS-RING K-WINDOW (GPT structural lever): publish KWIN A-slices into KWIN LDS slots, ONE barrier,
    //   consume KWIN slices (32*KWIN WMMA + per-slice B feed), barrier before reuse. Amortizes the 4-wave
    //   A-publish barrier over KWIN K-tiles (2 barriers / KWIN K-tiles). Correctness-preserving: same GEMM. =====
    s_mov_b32 s26, 0                          // t = 0 (K-tile index, advances by KWIN)
.Lkt_loop:
    // ---- PUBLISH: cooperatively load A-slices [t, t+KWIN) into LDS slots 0..KWIN-1 ----
    // ---- PUBLISH width KWINPW: issue KWINPW slices' A global-loads, ONE s_wait, KWINPW ds_stores. ----
    //   Each slice = 2x b128 (128-row A-strip = two 64-row halves). Reg slots by w: 0->v16, 1->v176 (B-frag
    //   regs, free in publish), 2->v192, 3->v200 (v192-207 = field-26 headroom above the v191 top, no occ cost).
    //   v24 (ti-broadcast) untouched. KWINPW=1 == plain publish; KWINPW=2 == old 2-wide. Must divide KWIN.
    .if KWIN != (KWIN/KWINPW)*KWINPW
        .error "KWINPW must divide KWIN"
    .endif
.if NBANDS > 2
    // ---- GENERAL NBANDS publish (NBANDS>2, e.g. 8x2 @ TWN=2 -> 4 bands): the 2-band fast path below can't hold
    //   NBANDS x KWINPW bands in registers, so go band-sequential: for each (p-group, band b) issue KWINPW slices'
    //   b128 A-loads into the 4 per-slice slots (16/176/192/200), ONE wait, KWINPW ds_stores with the band folded
    //   into the LDS immediate (b*WAVES*512). NBANDS waits/group (amortized over KWIN K-tiles); reg pressure stays
    //   KWINPW*4 (low 4 regs/slot, bands reuse them). Produces the IDENTICAL As LDS layout (row r -> offset r*32). ----
    .set p, 0
    .rept (KWIN / KWINPW)
      .set b, 0
      .rept NBANDS                               // sequential band pass b in [0,NBANDS)
        .set w, 0
        .rept KWINPW                             // issue KWINPW slices' band-b A-loads (KWINPW-way MLP per band)
          s_add_u32  s27, s26, (p*KWINPW + w)
          s_lshl_b32 s27, s27, 5
          s_add_u32  s27, s25, s27
          v_add_nc_u32 v15, v14, s27             // band-0 global row for this slice
          .set bb, 0
          .rept b                                // advance to band b: + b*band-stride (s29 = WAVES*16*K)
            v_add_nc_u32 v15, v15, s29
            .set bb, bb+1
          .endr
          .if w == 0
            global_load_b128 v[16:19],   v15, s[2:3]
          .elseif w == 1
            global_load_b128 v[176:179], v15, s[2:3]
          .elseif w == 2
            global_load_b128 v[192:195], v15, s[2:3]
          .else
            global_load_b128 v[200:203], v15, s[2:3]
          .endif
          .set w, w+1
        .endr
        s_wait_loadcnt 0x0
        .set w, 0
        .rept KWINPW                             // store band b: base v12, band folded into the LDS immediate
          .if w == 0
            ds_store_b128 v12, v[16:19]   offset:((p*KWINPW+0)*ATILE + b*(TWM*TWN*512))
          .elseif w == 1
            ds_store_b128 v12, v[176:179] offset:((p*KWINPW+1)*ATILE + b*(TWM*TWN*512))
          .elseif w == 2
            ds_store_b128 v12, v[192:195] offset:((p*KWINPW+2)*ATILE + b*(TWM*TWN*512))
          .else
            ds_store_b128 v12, v[200:203] offset:((p*KWINPW+3)*ATILE + b*(TWM*TWN*512))
          .endif
          .set w, w+1
        .endr
        s_wait_dscnt 0x0
        .set b, b+1
      .endr
      .set p, p+1
    .endr
.else
    .set p, 0
    .rept (KWIN / KWINPW)
      .set w, 0
      .rept KWINPW                              // issue KWINPW slices' A loads (slot per w: 16/176/192/200).
        s_add_u32  s27, s26, (p*KWINPW + w)     //   each slice = TM/64 bands of b128 (FM=4 -> 2 bands, FM=2 -> 1).
        s_lshl_b32 s27, s27, 5
        s_add_u32  s27, s25, s27
        v_add_nc_u32 v15, v14, s27
        .if w == 0
          global_load_b128 v[16:19],   v15, s[2:3]
          .if NBANDS > 1
            v_add_nc_u32 v15, v15, s29
            global_load_b128 v[20:23], v15, s[2:3]
          .endif
        .elseif w == 1
          global_load_b128 v[176:179], v15, s[2:3]
          .if NBANDS > 1
            v_add_nc_u32 v15, v15, s29
            global_load_b128 v[180:183], v15, s[2:3]
          .endif
        .elseif w == 2
          global_load_b128 v[192:195], v15, s[2:3]
          .if NBANDS > 1
            v_add_nc_u32 v15, v15, s29
            global_load_b128 v[196:199], v15, s[2:3]
          .endif
        .else
          global_load_b128 v[200:203], v15, s[2:3]
          .if NBANDS > 1
            v_add_nc_u32 v15, v15, s29
            global_load_b128 v[204:207], v15, s[2:3]
          .endif
        .endif
        .set w, w+1
      .endr
      s_wait_loadcnt 0x0                        // all KWINPW slices' loads landed together
      .set w, 0
      .rept KWINPW
        .if w == 0
          ds_store_b128 v12, v[16:19]   offset:((p*KWINPW+0)*ATILE)
          .if NBANDS > 1
            ds_store_b128 v13, v[20:23] offset:((p*KWINPW+0)*ATILE)
          .endif
        .elseif w == 1
          ds_store_b128 v12, v[176:179] offset:((p*KWINPW+1)*ATILE)
          .if NBANDS > 1
            ds_store_b128 v13, v[180:183] offset:((p*KWINPW+1)*ATILE)
          .endif
        .elseif w == 2
          ds_store_b128 v12, v[192:195] offset:((p*KWINPW+2)*ATILE)
          .if NBANDS > 1
            ds_store_b128 v13, v[196:199] offset:((p*KWINPW+2)*ATILE)
          .endif
        .else
          ds_store_b128 v12, v[200:203] offset:((p*KWINPW+3)*ATILE)
          .if NBANDS > 1
            ds_store_b128 v13, v[204:207] offset:((p*KWINPW+3)*ATILE)
          .endif
        .endif
        .set w, w+1
      .endr
      s_wait_dscnt 0x0
      .set p, p+1
    .endr
.endif
.if BLDS
    // ---- B-IN-LDS dedup fill: only wave_m==0 loads KWIN slices' B (global_load_tr) -> ds_store to B-ring[slot p].
    //   Both M-waves ds_load from the ring in the consume. Halves global B traffic. Frag round-tripped (no re-derive). ----
    v_readfirstlane_b32 s40, v3               // wave_m (uniform)
    s_cmp_eq_u32 s40, 0
    s_cbranch_scc0 .Lkbfill_skip
    .set p, 0
    .rept KWIN
      .set ni, 0                              // slice p: kk0 @ s[20:21], kk1 @ s[20:21]+s9 -> staging v176-191
      .rept FN
        global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
        .set ni, ni+1
      .endr
      s_add_u32  s44, s20, s9
      s_addc_u32 s45, s21, 0
      .set ni, 0
      .rept FN
        global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
        .set ni, ni+1
      .endr
      s_wait_loadcnt 0x0
      .set kk, 0                              // store 8 frags to B-ring slot p (v25 base + p*BTILE)
      .rept 2
        .set ni, 0
        .rept FN
          ds_store_b64 v25, v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1] offset:(p*BTILE + (kk*FN+ni)*256)
          .set ni, ni+1
        .endr
        .set kk, kk+1
      .endr
      s_lshl_b32 s43, s9, 1                    // advance B base to next slice (loader only)
      s_add_u32  s20, s20, s43
      s_addc_u32 s21, s21, 0
      .set p, p+1
    .endr
    s_wait_dscnt 0x0
.Lkbfill_skip:
.endif
.if KWINNOBAR == 0
    s_barrier_signal -1
    s_barrier_wait -1                         // ONE publish barrier per KWIN K-tiles (publishes A-ring AND B-ring)
.endif
    // ---- CONSUME: KWIN slices ----
.if KWINBPF
    // B-PREFETCH-ONE-AHEAD: 2 B slots (Bcur=176/Bnext=192) ping-pong. A is from LDS (dscnt) so loadcnt is B-ONLY.
    //   prologue issues B[t+0]; each iter issues B[t+u+1] then s_wait_loadcnt 8 retires B[u] (newest 8 = B[u+1] stay
    //   in flight, hidden behind slice u's WMMA). gfx12 loadcnt is in-order. Slice u -> B slot (u&1).
    .set ni, 0                                // prologue: B[t+0] -> slot0, advance s20
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_lshl_b32 s43, s9, 1
    s_add_u32  s20, s20, s43
    s_addc_u32 s21, s21, 0
    .set u, 0
    .rept KWIN
      .if (u + 1) < KWIN
        .set BN, (FB + (((u + 1) & 1) * (4*FN)))   // prefetch B[t+u+1] into the other slot, advance s20 (slots FB / FB+4FN; 4x4 FB=176 -> 176/192, 8x2 FB=192 -> 192/200)
        .set ni, 0
        .rept FN
          global_load_tr_b64 v[BN+ni*2:BN+ni*2+1], v9, s[20:21] offset:ni*256
          .set ni, ni+1
        .endr
        s_add_u32  s44, s20, s9
        s_addc_u32 s45, s21, 0
        .set ni, 0
        .rept FN
          global_load_tr_b64 v[BN+(FN+ni)*2:BN+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
          .set ni, ni+1
        .endr
        s_lshl_b32 s43, s9, 1
        s_add_u32  s20, s20, s43
        s_addc_u32 s21, s21, 0
      .endif
      .set kk, 0                                // A-frags from LDS slot u
      .rept 2
        .set mi, 0
        .rept FM
          ds_load_b64 v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v11 offset:(u*ATILE + mi*512 + kk*16)
          .set mi, mi+1
        .endr
        .set kk, kk+1
      .endr
      .if (u + 1) < KWIN
        s_wait_loadcnt (2*FN)                  // B[t+u] landed; keep next slice's 2*FN B-loads in flight (4x4=8, 8x2=4)
      .else
        s_wait_loadcnt 0x0
      .endif
      s_wait_dscnt 0x0
      .set BC, (FB + ((u & 1) * (4*FN)))           // 32 WMMA with B slot (u&1)  (FB / FB+4FN; matches the BN prefetch slots)
      .set kk, 0
      .rept 2
        .set mi, 0
        .rept FM
          .set ni, 0
          .rept FN
            v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[BC+(kk*FN+ni)*2:BC+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
            .set ni, ni+1
          .endr
          .set mi, mi+1
        .endr
        .set kk, kk+1
      .endr
      .set u, u+1
    .endr
.else
    .set u, 0
    .rept KWIN
      .set kk, 0
      .rept 2
        .set mi, 0
        .rept FM
          ds_load_b64 v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v11 offset:(u*ATILE + mi*512 + kk*16)
          .set mi, mi+1
        .endr
        .set kk, kk+1
      .endr
.if BLDS
      .set kk, 0                               // B frags from B-ring slot u (filled by wave_m==0; both M-waves read)
      .rept 2
        .set ni, 0
        .rept FN
          ds_load_b64 v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v25 offset:(u*BTILE + (kk*FN+ni)*256)
          .set ni, ni+1
        .endr
        .set kk, kk+1
      .endr
      s_wait_dscnt 0x0                         // A + B both from LDS -> dscnt only
.else
      .set ni, 0
      .rept FN
        global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
        .set ni, ni+1
      .endr
      s_add_u32  s44, s20, s9
      s_addc_u32 s45, s21, 0
      .set ni, 0
      .rept FN
        global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
        .set ni, ni+1
      .endr
      s_wait_loadcnt 0x0                       // all 8 B landed
      s_wait_dscnt 0x0                         // A frags resident
.endif
.if FEEDONLY == 0
      .set kk, 0
      .rept 2
        .set mi, 0
        .rept FM
          .set ni, 0
          .rept FN
            v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
            .set ni, ni+1
          .endr
          .set mi, mi+1
        .endr
        .set kk, kk+1
      .endr
.endif
.if BLDS == 0
      s_lshl_b32 s43, s9, 1                    // advance B by 2*NT*256 (1 K-tile); BLDS loader already advanced
      s_add_u32  s20, s20, s43
      s_addc_u32 s21, s21, 0
.endif
      .set u, u+1
    .endr
.endif
.if (KWINNOTAIL == 0) && (KWINNOBAR == 0)
    s_barrier_signal -1
    s_barrier_wait -1                          // tail/reuse barrier: all waves done reading the KWIN slots before overwrite
.endif
    s_add_u32 s26, s26, KWIN
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lkt_loop
.elseif DBUF == 0
    // ================= SINGLE-BUFFER K-loop (baseline) =================
    s_mov_b32 s26, 0                         // t = 0
.Lkt_loop:
.if PROFILE
    s_cmp_eq_u32 s55, 1
    s_cbranch_scc0 .Lpt_dtop
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s46, s30
    s_add_u32 s54, s54, 1
.Lpt_dtop:
.endif
    s_lshl_b32 s27, s26, 5                   // t*32
    s_add_u32  s27, s25, s27                 // row_base_K + k0
    v_add_nc_u32 v15, v14, s27               // voff0 = rc + s_rowk0
    global_load_b128 v[16:19], v15, s[2:3]
    v_add_nc_u32 v15, v15, s29               // voff1 = +64*K
    global_load_b128 v[20:23], v15, s[2:3]
    s_wait_loadcnt 0x0
    PT s47                                    // phase 1: A global-load span
    ds_store_b128 v12, v[16:19]
    ds_store_b128 v13, v[20:23]
    s_wait_dscnt 0x0
    PT s48                                    // phase 2: A ds_store + dscnt span
.if BLDS
    // ---- B-IN-LDS dedup fill: only wave_m==0 loads B (both kk) from global -> ds_store to B-region[wave_n].
    //   wave_m==1 skips the global load entirely and reads wave_m==0's frags from LDS in the consume. Halves global B. ----
    v_readfirstlane_b32 s40, v3               // wave_m (uniform per wave)
    s_cmp_eq_u32 s40, 0
    s_cbranch_scc0 .Lbfill_skip
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_lshl_b32 s43, s9, 1
    s_add_u32  s20, s20, s43                   // loader advances B base one K-tile (wave_m==1 s20 unused)
    s_addc_u32 s21, s21, 0
    s_wait_loadcnt 0x0
    .set kk, 0
    .rept 2
      .set ni, 0
      .rept FN
        ds_store_b64 v25, v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1] offset:((kk*FN+ni)*256)
        .set ni, ni+1
      .endr
      .set kk, kk+1
    .endr
    s_wait_dscnt 0x0
.Lbfill_skip:
.endif
.if NOBAR == 0
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    PT s49                                    // phase 3: A publish barrier span
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        ds_load_b64 v[AF+mi*2:AF+mi*2+1], v11 offset:(mi*512 + kk*16)
        .set mi, mi+1
      .endr
.if BLDS
      .set ni, 0                              // B frags from LDS B-region[wave_n] (filled by wave_m==0; both M-waves read)
      .rept FN
        ds_load_b64 v[BF+ni*2:BF+ni*2+1], v25 offset:((kk*FN+ni)*256)
        .set ni, ni+1
      .endr
      s_wait_dscnt 0x0                        // A + B both from LDS -> dscnt only, no loadcnt
.else
      .set ni, 0
      .rept FN
        global_load_tr_b64 v[BF+ni*2:BF+ni*2+1], v9, s[20:21] offset:ni*256
        .set ni, ni+1
      .endr
      s_wait_loadcnt 0x0
      s_wait_dscnt 0x0
.endif
      PT s50                                 // phase 4: per-kk B feed + A ds_load wait (x2/K-tile)
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[AF+mi*2:AF+mi*2+1], v[BF+ni*2:BF+ni*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      PT s51                                 // phase 5: per-kk WMMA span (x2/K-tile; hidden -> should be small)
.if BLDS == 0
      s_add_u32 s20, s20, s9                  // non-BLDS: each consumer advances its own B base per kk
      s_addc_u32 s21, s21, 0
.endif
      .set kk, kk+1
    .endr
.if NOBAR == 0
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    PT s52                                   // phase 6: tail barrier span
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s12
    PT s53                                   // phase 7: loop bookkeeping span
    s_cbranch_scc1 .Lkt_loop
.else
    // ================= A DOUBLE-BUFFER K-loop (ping-pong As[2], prefetch A(t+1) during compute) =====
    // fa: 8 frags v[160:175] (kk0 mi0-3, kk1 mi0-3); fb: 8 frags v[176:191]; areg v[16:23].
    // prologue: fill As[0] for t=0
    v_add_nc_u32 v15, v14, s25               // voff0 = rc + row_base_K (k0=0)
    global_load_b128 v[16:19], v15, s[2:3]
    v_add_nc_u32 v15, v15, s29               // voff1 = +64*K
    global_load_b128 v[20:23], v15, s[2:3]
    s_wait_loadcnt 0x0
    ds_store_b128 v12, v[16:19]              // As[0][tid*16]
    ds_store_b128 v13, v[20:23]              // As[0][tid*16+2048]
    s_wait_dscnt 0x0
    s_barrier_signal -1
    s_barrier_wait -1
    s_mov_b32 s26, 0                         // t = 0
.if STAGGER
    // ---- rung 8: inert per-WG phase stagger (NO timers, NO atomics). Offset this claimed tile's K-loop
    //      start by delay = ((ti*13 + wid*3) & STAGGER_MASK) << STAGGER_SHIFT busy-loop iters. ti (s17) is
    //      per-WG (same for all 4 waves, differs across WGs) -> a PERSISTENT inter-WG phase offset (no
    //      cross-WG barrier re-syncs it); the wid term adds first-iter within-WG jitter (eaten by the
    //      per-K barrier). Scratch s58-s60 (free in every build; PROFILE tops out at s57). One-time per tile. ----
    s_mul_i32  s58, s17, 13                  // ti*13
    v_readfirstlane_b32 s59, v1              // wid (uniform across the wave's lanes)
    s_lshl_b32 s60, s59, 1
    s_add_u32  s59, s60, s59                 // wid*3
    s_add_u32  s58, s58, s59                 // ti*13 + wid*3
    s_and_b32  s58, s58, STAGGER_MASK
    s_lshl_b32 s58, s58, STAGGER_SHIFT       // delay iters
    s_cmp_eq_u32 s58, 0
    s_cbranch_scc1 .Lstag_done
.Lstag_loop:
    s_sub_u32  s58, s58, 1
    s_cmp_lg_u32 s58, 0
    s_cbranch_scc1 .Lstag_loop
.Lstag_done:
.endif
.if PB == 2
    // ---- rung 9 B2: one extra global_atomic per CLAIMED TILE (leader lane -> occ[6], offset:24), no sendmsg,
    //      no per-K. Tests whether the PROFILE token/admission path (an extra global atomic) is the 70x lever.
    //      Mirrors the claim atomic exactly (s[0:1]=occ base, v7=addr-off=0, v5=data). exec saved in s58. ----
    s_mov_b32 s58, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v0               // leader = tid==0
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lpb2_skip
    v_mov_b32 v5, 1
    global_atomic_add_u32 v6, v7, v5, s[0:1] offset:24 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lpb2_skip:
    s_mov_b32 exec_lo, s58
.endif
.Lkt_loop:
.if PROFILE
    s_cmp_eq_u32 s55, 1
    s_cbranch_scc0 .Lpt_btop
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s57, s30                        // prev = K-tile top
    s_add_u32 s54, s54, 1                      // K-tile count++
.Lpt_btop:
.endif
.if PB == 1
    // ---- rung 9 B1: per-K-tile s_sendmsg_rtn + s_wait_kmcnt, ALL waves, no timing math/branch/atomic.
    //      Matches the profiler's per-K cadence; tests whether sendmsg/kmcnt-wait perturbs scheduler/waitcnt
    //      state. (Profiler did this on ONE wave; this is the stronger all-wave version.) Dest s[58:59]. ----
    s_sendmsg_rtn_b64 s[58:59], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
.endif
.if PB == 3
    // ---- rung 9 B3: per-K-tile control-flow skeleton ONLY (the cmp+branch every wave runs in PROFILE), no
    //      sendmsg/atomic/wait. s26 (=t) is never -1 -> branch always taken, empty body (= "branch over"). ----
    s_cmp_eq_u32 s26, 0xFFFFFFFF
    s_cbranch_scc0 .Lpb3_skip
    s_nop 0
.Lpb3_skip:
.endif
.if PB == 4
    // ---- rung 9 B4: per-K-tile inert delay (scalar busy loop, ALL waves), no sendmsg. Control vs B1 to
    //      separate a "special s_sendmsg side effect" from a generic "per-K perturbation". s58 scratch. ----
    s_mov_b32 s58, PB4_ITERS
.Lpb4_loop:
    s_sub_u32 s58, s58, 1
    s_cmp_lg_u32 s58, 0
    s_cbranch_scc1 .Lpb4_loop
.endif
    s_and_b32  s40, s26, 1                   // buf = t&1
    s_lshl_b32 s41, s40, 12                  // buf*4096
    v_add_nc_u32 v25, v11, s41               // A-read base = ldsbase + buf*4096
    // 8 A-frag reads from As[buf]
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        ds_load_b64 v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v25 offset:(mi*512 + kk*16)
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    // 8 B frags: kk0 @ s[20:21], kk1 @ s[20:21]+NT*256
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+ni*2:FB+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    s_add_u32  s44, s20, s9
    s_addc_u32 s45, s21, 0
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[FB+(FN+ni)*2:FB+(FN+ni)*2+1], v9, s[44:45] offset:ni*256
      .set ni, ni+1
    .endr
    s_wait_loadcnt 0x0                        // all 8 B landed (no A-prefetch issued yet -> exact, no stale frags)
    PT s47                                    // phase 1: 8 B global_load_tr wait span
    s_wait_dscnt 0x0                          // A-frag reads done
    PT s48                                    // phase 2: A-frag ds_load wait span
    // A-prefetch for t+1 issued AFTER the B wait -> overlaps the 32 WMMAs below, waited at ds_store
    s_add_u32  s42, s26, 1                    // t+1
    s_cmp_ge_u32 s42, s12
    s_cbranch_scc1 .Lno_pf
    s_lshl_b32 s43, s42, 5                    // (t+1)*32
    s_add_u32  s43, s25, s43                  // row_base_K + (t+1)*32
    v_add_nc_u32 v15, v14, s43
    global_load_b128 v[16:19], v15, s[2:3]
    v_add_nc_u32 v15, v15, s29
    global_load_b128 v[20:23], v15, s[2:3]
.Lno_pf:
    // 32 WMMA (kk0: fa[0:3] x fb[0:3]; kk1: fa[4:7] x fb[4:7]). FEEDONLY=1 skips them -> matched feed-only
    // probe on the REAL DBUF==1 path (all loads/prefetch/barriers kept; output garbage; wall = feed cost).
.if FEEDONLY == 0
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+(kk*FM+mi)*2:FA+(kk*FM+mi)*2+1], v[FB+(kk*FN+ni)*2:FB+(kk*FN+ni)*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
.endif
    PT s49                                    // phase 3: 32 WMMA span (A-prefetch overlaps -> should be small if hidden)
    s_lshl_b32 s43, s9, 1                     // advance B by 2*NT*256 (2 kt)
    s_add_u32  s20, s20, s43
    s_addc_u32 s21, s21, 0
    // land A-prefetch -> As[other]
    s_add_u32  s42, s26, 1
    s_cmp_ge_u32 s42, s12
    s_cbranch_scc1 .Lno_store
    s_wait_loadcnt 0x0                        // A-prefetch landed
    s_and_b32  s46, s42, 1                    // other = (t+1)&1
    s_lshl_b32 s46, s46, 12                   // other*4096
    v_add_nc_u32 v26, v12, s46                // store dst0 = tid*16 + other*4096
    v_add_nc_u32 v27, v13, s46
    ds_store_b128 v26, v[16:19]
    ds_store_b128 v27, v[20:23]
    s_wait_dscnt 0x0
.Lno_store:
    PT s50                                   // phase 4: A-prefetch land + ds_store-to-other-slot span
    s_barrier_signal -1
    s_barrier_wait -1
    PT s51                                   // phase 5: tail barrier span
    s_add_u32  s26, s26, 1
    s_cmp_lt_u32 s26, s12
    PT s52                                   // phase 6: loop bookkeeping span
    s_cbranch_scc1 .Lkt_loop
.endif

    v_readfirstlane_b32 s22, v1             // wid (scalar)
.if STORE
    // ---- STORE=1: 16 frags FLAT: C + ti*(WAVES*16384) + wid*16384 + frag*1024 ----
    s_lshl_b32 s23, s17, (TWM_LOG2+TWN_LOG2+14)   // ti*(WAVES*16384)  (65536 @ 4-wave 2x2)
    s_lshl_b32 s24, s22, 14                  // wid*16384
    s_add_u32  s23, s23, s24
    s_add_u32  s28, s6, s23
    s_addc_u32 s29, s7, 0
    .set frag, 0
    .rept FM*FN
      global_store_b128 v10, v[ACC+frag*8:ACC+frag*8+3], s[28:29] offset:(frag*1024)
      global_store_b128 v10, v[ACC+frag*8+4:ACC+frag*8+7], s[28:29] offset:(frag*1024+16)
      .set frag, frag+1
    .endr
.else
    // ---- STORE=0: minimal -- only acc[0][0]/wave -> C + ti*(WAVES*1024) + wid*1024 (perf, 16x less traffic) ----
    s_lshl_b32 s23, s17, (TWM_LOG2+TWN_LOG2+10)   // ti*(WAVES*1024)  (4096 @ 4-wave 2x2)
    s_lshl_b32 s24, s22, 10                  // wid*1024
    s_add_u32  s23, s23, s24
    s_add_u32  s28, s6, s23
    s_addc_u32 s29, s7, 0
    global_store_b128 v10, v[ACC:ACC+3], s[28:29]
    global_store_b128 v10, v[ACC+4:ACC+7], s[28:29] offset:16
.endif
    s_wait_storecnt 0x0
    // restore s29 = band global row-stride (WAVES*16)*K for the next claimed tile's K-loop
    s_lshl_b32 s29, s8, (TWM_LOG2+TWN_LOG2+4)
    // ---- band step: next tile in this band needs NO atomic; only re-claim when the band is exhausted ----
    s_add_u32  s39, s39, 1
    s_cmp_lt_u32 s39, BAND
    s_cbranch_scc1 .Lband
    s_branch .Lclaim_loop

.Lexit:
    s_wait_storecnt 0x0
.if PROFILE
    // ---- profiler wave stores 7 phase tick-sums (occ[8..14]) + K-tile count (occ[15]) ----
    s_cmp_eq_u32 s55, 1
    s_cbranch_scc0 .Lprof_store_done
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lprof_store_skip
    v_mov_b32 v5, s47
    global_store_b32 v7, v5, s[0:1] offset:32
    v_mov_b32 v5, s48
    global_store_b32 v7, v5, s[0:1] offset:36
    v_mov_b32 v5, s49
    global_store_b32 v7, v5, s[0:1] offset:40
    v_mov_b32 v5, s50
    global_store_b32 v7, v5, s[0:1] offset:44
    v_mov_b32 v5, s51
    global_store_b32 v7, v5, s[0:1] offset:48
    v_mov_b32 v5, s52
    global_store_b32 v7, v5, s[0:1] offset:52
    v_mov_b32 v5, s53
    global_store_b32 v7, v5, s[0:1] offset:56
    v_mov_b32 v5, s54
    global_store_b32 v7, v5, s[0:1] offset:60
    s_wait_storecnt 0x0
.Lprof_store_skip:
    s_mov_b32 exec_lo, s16
.Lprof_store_done:
.endif
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v5, s30
    global_atomic_max_u32 v7, v5, s[0:1] offset:12 scope:SCOPE_DEV
.Lafter_t1:
    s_mov_b32 exec_lo, s16
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v5, -1
    global_atomic_add_u32 v7, v5, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s16
    s_endpgm
    .size occ_kernel, .-occ_kernel
