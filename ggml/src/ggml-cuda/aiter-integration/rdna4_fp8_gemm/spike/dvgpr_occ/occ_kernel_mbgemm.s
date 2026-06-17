// occ_kernel_mbgemm.s  (gfx1201, wave32) -- FED FAT-TILE MICRO-BATCH GEMM, double-buffered (MAD-305).
//
// The user's micro-batch dynamic-queue architecture (occ_kernel_mb.s = 276 TF PURE-WMMA) on a REAL
// fp8 GEMM with real per-K feed AND intra-tile SOFTWARE PREFETCH. Persistent waves pull output-tiles
// from a global atomic work-queue; per tile dyn-VGPR GROWS to an FM x FN accumulator tile, streams
// the K-reduction with a DOUBLE-BUFFERED feed (issue K-step k+1's A/B loads before the k WMMAs run,
// so each wave hides its OWN ~400-cycle fetch latency behind compute -- the fix for the lockstep
// feed stall the single-buffer version hit), SHIPS acc[0][0], and SHRINKS back to lean.
//
// FAT TILE (reuse) x PREFETCH (latency hiding) x dyn-VGPR (elastic occupancy) -- the three stacked
// levers. reuse = FM*FN/(FM+FN) WMMAs/load. Double-buffering 2x's the A/B frag VGPRs:
//   1x1=48, 2x2=80, 2x4=120 fit the default 128 dyn cap (no umr); 4x4=192 needs BLOCK_SIZE=1.
//
// FEED: A direct-from-L2 global_load_b64 (8 K-bytes/lane -> v2i32); B global_load_tr_b64 from
// pre-shuffled tile-major Bshuf. Both verified bit-exact. Tile grid column count is a power of 2 ->
// ti -> (tile_row = ti>>LOG2, tile_col = ti & MASK), no ISA divide. K must be a multiple of 32
// (KT even) for the unroll-by-2; the tail pair is peeled so the last prefetch never reads past K.
//
// User data (USER_SGPR=15, s0..s14):
//   s0:s1=occ  s2:s3=A  s4:s5=Bshuf  s6:s7=C  s8=KT  s9=K(bytes/A-row)  s10=NT*256  s11=TOTAL_TILES
//   s12=NTILES_N_MASK  s13=NTILES_N_LOG2  s14=FN*256
// Scratch: s16 exec, s17 ti, s18 col, s19 row, s20:s21 B-saddr, s22 tmp, s26 K-pair counter,
//   s27..s29 store-addr, s30:s31 timer, s32 rowstride16, s40.. A-saddr pairs (FM of them).
.ifndef FM
    .set FM, 1
.endif
.ifndef FN
    .set FN, 1
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 1
.endif
.ifndef BATCH
    .set BATCH, 1                       // tiles claimed per atomic grab (amortizes the contended queue counter + grow/shrink)
.endif
.ifndef NOFEED
    .set NOFEED, 0                      // 1 = load operands ONCE, reuse across all K-steps (isolation probe: framework ceiling w/o feed)
.endif
.ifndef PROFILE
    .set PROFILE, 0                     // 1 = in-kernel REALTIME-timer phase breakdown (workgroup 0 writes occ[24..44])
.endif
.ifndef STAGGER
    .set STAGGER, 0                     // 0 = lockstep (persistent waves march K synchronized -> all s_wait at once).
.endif                                  // >0 = one-time startup spin = TGID_X*STAGGER, phase-offsets each wave so feed
                                        // stalls INTERLEAVE and the occupancy we already have hides the feed (KG 50147c07).
.set ABASE,   (32 + FM*FN*8)            // A frags: 2 buffers of FM*2 regs (buf b, frag mi @ ABASE+b*FM*2+mi*2)
.set BBASE,   (ABASE + FM*4)            // B frags: 2 buffers of FN*2 regs (buf b, frag ni @ BBASE+b*FN*2+ni*2)
.set FAT_RAW, (BBASE + FN*4)
.set FATREGS, ((FAT_RAW + 15) & ~15)    // grow target, rounded to a 16-VGPR block
.set WAITN,   (FM + FN)                 // outstanding loads for one buffer (wait target with prefetch in flight)

// ---- LOADBUF b: issue buffer b's FM A-frags + FN B-frags at the current saddrs, then advance ----
.macro LOADBUF b
    .set mi, 0
    .rept FM
      global_load_b64 v[ABASE+\b*FM*2+mi*2:ABASE+\b*FM*2+mi*2+1], v8, s[40+2*mi:41+2*mi]
      .set mi, mi+1
    .endr
    .set ni, 0
    .rept FN
      global_load_tr_b64 v[BBASE+\b*FN*2+ni*2:BBASE+\b*FN*2+ni*2+1], v9, s[20:21] offset:ni*256
      .set ni, ni+1
    .endr
    .set mi, 0
    .rept FM
      s_add_u32  s[40+2*mi], s[40+2*mi], 16
      s_addc_u32 s[41+2*mi], s[41+2*mi], 0
      .set mi, mi+1
    .endr
    s_add_u32 s20, s20, s10
    s_addc_u32 s21, s21, 0
.endm

// ---- WMMABUF b: FM*FN accumulating WMMAs from buffer b (accumulators pre-zeroed) ----
.macro WMMABUF b
    .set mi, 0
    .rept FM
      .set ni, 0
      .rept FN
        v_wmma_f32_16x16x16_fp8_fp8 v[32+(mi*FN+ni)*8:32+(mi*FN+ni)*8+7], v[ABASE+\b*FM*2+mi*2:ABASE+\b*FM*2+mi*2+1], v[BBASE+\b*FN*2+ni*2:BBASE+\b*FN*2+ni*2+1], v[32+(mi*FN+ni)*8:32+(mi*FN+ni)*8+7]
        .set ni, ni+1
      .endr
      .set mi, mi+1
    .endr
.endm

// ---- WMMABUF_WAIT b, w0: same FM*FN WMMAs, but the operand loads are RELEASED individually with a
// descending s_wait_loadcnt ladder (the hipcc transcription) instead of one coarse barrier, so the
// first WMMAs start while buffer b's later fragments are still arriving. The ladder is emitted on the
// first A-row (mi==0): the (mi=0,ni) WMMA needs B-frag ni (issued after the FM A-frags), so it gates
// on loadcnt = w0 + FN-1-ni, descending to w0 at the last B-frag (== the old coarse WAITN floor,
// guaranteeing buffer b fully landed by then). mi>=1 reuse already-awaited frags -> no wait. The
// exact w0 base is perf/oracle-tuned in T2; structurally this is the descending ladder + back-to-back
// WMMA run that matches hipcc's inner loop.
.macro WMMABUF_WAIT b, w0
    .set mi, 0
    .rept FM
      .set ni, 0
      .rept FN
        .if mi == 0
          s_wait_loadcnt (\w0 + FN - 1 - ni)
        .endif
        v_wmma_f32_16x16x16_fp8_fp8 v[32+(mi*FN+ni)*8:32+(mi*FN+ni)*8+7], v[ABASE+\b*FM*2+mi*2:ABASE+\b*FM*2+mi*2+1], v[BBASE+\b*FN*2+ni*2:BBASE+\b*FN*2+ni*2+1], v[32+(mi*FN+ni)*8:32+(mi*FN+ni)*8+7]
        .set ni, ni+1
      .endr
      .set mi, mi+1
    .endr
.endm

// ---- PROFILE: read the 100 MHz REALTIME counter, accumulate (now - prev) into phase reg \acc ----
// Wave-uniform scalar timing; phases live in s51..s56, prev=s50, scratch s[60:61]/s62. (2x4 profile
// config: A-saddrs use s40..s43, so s50+ is free.)
.macro TICK acc
.if PROFILE
    s_sendmsg_rtn_b64 s[60:61], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_sub_u32 s62, s60, s50
    s_add_u32 \acc, \acc, s62
    s_mov_b32 s50, s60
.endif
.endm

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- BREAK LOCKSTEP (KG 50147c07): one-time startup spin = TGID_X(s15) * STAGGER. Phase-shifts each
    // persistent wave so their ~400-cycle feed stalls INTERLEAVE instead of firing in a synchronized burst,
    // letting the occupancy we already have hide the feed (what the grid scheduler gives conventional
    // kernels for free). STAGGER=0 -> emits nothing (old lockstep). s22 = the per-kernel scratch tmp. ----
.if STAGGER
    s_mul_i32 s22, s15, STAGGER
    s_cmp_eq_u32 s22, 0
    s_cbranch_scc1 .Lstagger_done
.Lstagger:
    s_sub_u32 s22, s22, 1
    s_cmp_lg_u32 s22, 0
    s_cbranch_scc1 .Lstagger
.Lstagger_done:
.endif
    v_mov_b32 v4, 0
    // ---- lane-0-only admission occupancy counter ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV
    global_atomic_add_u32 v4, v2, s[0:1] offset:16 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s16
    // ---- per-lane address constants (frag- & K-step-invariant; lean block) ----
    v_and_b32 v6, 15, v0
    v_mul_lo_u32 v8, v6, s9              // (lane&15) * K
    v_bfe_u32 v7, v0, 4, 1
    v_lshlrev_b32 v7, 3, v7
    v_add_nc_u32 v8, v8, v7              // v8 = A vaddr = (lane&15)*K + colhi*8
    v_lshlrev_b32 v9, 3, v0             // v9 = B vaddr = lane*8
    v_lshlrev_b32 v10, 5, v0            // v10 = C store vaddr = lane*32
    // ---- timer t0 ----
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
.if PROFILE
    s_sendmsg_rtn_b64 s[60:61], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s50, s60                    // prev = entry time
    s_mov_b32 s51, 0                      // ATOMIC (grab)
    s_mov_b32 s52, 0                      // GROW  (s_alloc up)
    s_mov_b32 s53, 0                      // SETUP (decode+saddr+zero)
    s_mov_b32 s54, 0                      // COMPUTE (K-loop incl feed waits)
    s_mov_b32 s55, 0                      // STORE
    s_mov_b32 s56, 0                      // SHRINK (s_alloc down)
.endif

    // ============ PERSISTENT BATCHED WORK-QUEUE LOOP ============
.Lbatch_loop:
    // lane-0 claims BATCH tiles per atomic -> amortizes the ONE contended device-scope counter
    // (16384 grabs for a 2048^3 serialize the machine) AND the per-batch dyn-VGPR grow/shrink.
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_grab
    v_mov_b32 v2, BATCH
    global_atomic_add_u32 v3, v4, v2, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lafter_grab:
    s_mov_b32 exec_lo, s16
    v_readlane_b32 s33, v3, 0             // s33 = batch base tile index
    s_cmp_ge_u32 s33, s11                 // base >= TOTAL -> queue drained
    s_cbranch_scc1 .Ltiles_done
    TICK s51                              // ATOMIC: the grab (per batch)
.if DYNVGPR
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
    s_alloc_vgpr FATREGS                  // ---- GROW once per batch ----
.endif
    TICK s52                              // GROW (per batch)
    s_mov_b32 s34, 0                      // j = 0 (intra-batch tile index)
.Ltile_in_batch:
    s_add_u32 s17, s33, s34              // ti = base + j
    s_cmp_ge_u32 s17, s11                 // ti >= TOTAL -> partial last batch, stop
    s_cbranch_scc1 .Lbatch_end
    s_and_b32 s18, s17, s12              // tile_col = ti & MASK
    s_lshr_b32 s19, s17, s13             // tile_row = ti >> LOG2
    // ---- B col-tile saddr (k=0) ----
    s_mul_i32 s20, s18, s14
    s_add_u32 s20, s4, s20
    s_addc_u32 s21, s5, 0
    // ---- A frag saddrs (k=0): A_saddr(0)=A+tile_row*(16*FM)*K; A_saddr(mi)=A_saddr(mi-1)+16*K ----
    s_lshl_b32 s32, s9, 4
    s_mul_i32 s22, s19, (16*FM)
    s_mul_i32 s22, s22, s9
    s_add_u32 s40, s2, s22
    s_addc_u32 s41, s3, 0
    .set mi, 1
    .rept FM-1
      s_add_u32  s[40+2*mi], s[40+2*(mi-1)], s32
      s_addc_u32 s[41+2*mi], s[41+2*(mi-1)], 0
      .set mi, mi+1
    .endr
    s_wait_storecnt 0x0                   // prev tile's deferred store finishes here (overlapped the decode/setup above)
    // ---- zero accumulators (all WMMAs accumulate; no srcC=0 peel) ----
    .set idx, 0
    .rept FM*FN
      v_mov_b32 v[32+idx*8+0], 0
      v_mov_b32 v[32+idx*8+1], 0
      v_mov_b32 v[32+idx*8+2], 0
      v_mov_b32 v[32+idx*8+3], 0
      v_mov_b32 v[32+idx*8+4], 0
      v_mov_b32 v[32+idx*8+5], 0
      v_mov_b32 v[32+idx*8+6], 0
      v_mov_b32 v[32+idx*8+7], 0
      .set idx, idx+1
    .endr
    TICK s53                              // SETUP: decode + saddr + store-wait + zero (per tile)
    // ---- prologue: load step 0 into buffer 0 (advances saddrs to step 1) ----
    LOADBUF 0
.if NOFEED
    // ISOLATION: operands loaded once, reused for all KT WMMAs -> NO per-K feed. Same tile, same
    // KT, same atomic queue + grow/shrink. If TF >> the fed kernel, the FEED is the bottleneck;
    // if ~equal, the FRAMEWORK is. (Result is garbage -- perf probe only, oracle expected BAD.)
    s_wait_loadcnt 0x0
    s_mov_b32 s26, s8
.Lkloop:
    WMMABUF 0
    s_sub_u32 s26, s26, 1
    s_cmp_lg_u32 s26, 0
    s_cbranch_scc1 .Lkloop
.else
    s_lshr_b32 s26, s8, 1                // KT/2
    s_sub_u32 s26, s26, 1                // KT/2 - 1 full double-steps (tail pair peeled)
    s_cmp_eq_u32 s26, 0
    s_cbranch_scc1 .Ltail
.Lkloop:
    LOADBUF 1                            // prefetch the odd step (advance)
    WMMABUF_WAIT 0, WAITN                // even step: interleaved release ladder (odd still in flight)
    LOADBUF 0                            // prefetch the next even step (advance)
    WMMABUF_WAIT 1, WAITN                // odd step: interleaved release ladder
    s_sub_u32 s26, s26, 1
    s_cmp_lg_u32 s26, 0
    s_cbranch_scc1 .Lkloop
.Ltail:
    // last pair: buf0 holds the penultimate (even) step; load the final (odd) step, no prefetch past K
    LOADBUF 1
    WMMABUF_WAIT 0, WAITN
    s_wait_loadcnt 0x0
    WMMABUF 1
.endif
    TICK s54                              // COMPUTE: prologue load + K-loop (incl feed waits) (per tile)
    // ---- SHIP: store acc[0][0] (v[32:39]) to C[ti] (256 f32 = 1024 B) ----
    s_lshl_b32 s27, s17, 10
    s_add_u32 s28, s6, s27
    s_addc_u32 s29, s7, 0
    global_store_b128 v10, v[32:35], s[28:29]
    global_store_b128 v10, v[36:39], s[28:29] offset:16
    TICK s55                              // STORE (issue; wait deferred) (per tile)
    // store wait DEFERRED -> overlaps the next tile's decode/setup (waited before the accs are reused/freed)
    // ---- next tile in this batch (registers stay fat across the batch) ----
    s_add_u32 s34, s34, 1
    s_cmp_eq_u32 s34, BATCH
    s_cbranch_scc0 .Ltile_in_batch        // j != BATCH -> next tile
.Lbatch_end:
    s_wait_storecnt 0x0                   // last tile's store must complete before the shrink frees its regs
.if DYNVGPR
    s_alloc_vgpr 32                       // ---- SHRINK once per batch ----
.endif
    TICK s56                              // SHRINK + last-store-wait (per batch)
    s_branch .Lbatch_loop

.Ltiles_done:
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
.if PROFILE
    // ---- workgroup 0 writes its per-phase tick totals -> occ[24..44] (representative; no 32-bit overflow) ----
    s_cmp_eq_u32 s15, 0                   // s15 = TGID_X (workgroup id)
    s_cbranch_scc0 .Lprofdone
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lprofrestore
    v_mov_b32 v5, s51
    global_store_b32 v4, v5, s[0:1] offset:24
    v_mov_b32 v5, s52
    global_store_b32 v4, v5, s[0:1] offset:28
    v_mov_b32 v5, s53
    global_store_b32 v4, v5, s[0:1] offset:32
    v_mov_b32 v5, s54
    global_store_b32 v4, v5, s[0:1] offset:36
    v_mov_b32 v5, s55
    global_store_b32 v4, v5, s[0:1] offset:40
    v_mov_b32 v5, s56
    global_store_b32 v4, v5, s[0:1] offset:44
    s_wait_storecnt 0x0
.Lprofrestore:
    s_mov_b32 exec_lo, s16
.Lprofdone:
.endif
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v2, -1
    global_atomic_add_u32 v4, v2, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s16
    s_endpgm
    .size occ_kernel, .-occ_kernel
