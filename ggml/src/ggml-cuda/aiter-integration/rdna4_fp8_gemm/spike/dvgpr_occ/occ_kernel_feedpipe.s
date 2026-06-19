// occ_kernel_feedpipe.s  (gfx1201, wave32) -- MAD-305 Step A: FEED-ONLY depth-P pipeline BANDWIDTH probe.
//
// GPT "multi-stage feed pipeline" angle. The proven root cause (RESULT_WGGEMM.md 2.6): the fed GEMM's
// wall is 100% feed, compute is free, and the feed runs ~2.7 GB/s (<0.5% of card BW) because each K-tile
// drains `s_wait_loadcnt 0x0` + barrier before the next can start -> loads are serialized one round-trip
// at a time. This probe answers the gating question BEFORE any GEMM rework:
//
//   "If we keep P independent K-slices in flight (never drain the queue), does effective feed
//    bandwidth scale above ~2.7 GB/s?"
//
// NO WMMA, NO LDS, NO barriers. Pure load stream. Per WAVE (launch 32-thread WGs): a depth-PDEPTH
// register ring of FRAGS b64 loads/slice. Prologue issues PDEPTH slices (loadcnt -> PDEPTH*FRAGS). Steady
// state, per slice: s_wait_loadcnt((PDEPTH-1)*FRAGS) retires the OLDEST slice (gfx in-order loadcnt
// accounting), xor-consume it (forces the loads to actually retire -> no DCE, true back-pressure), then
// reissue that ring slot at the next address (back to PDEPTH in flight). Steady loop is UNROLLED by
// PDEPTH so every ring slot is a compile-time register index (asm can't index regs by a runtime slot).
//
//   P=1  -> wait_loadcnt 0  -> full drain every slice = the ~2.7 GB/s latency-bound baseline.
//   P>=2 -> (P-1)*FRAGS slices stay in flight -> overlap. BW should climb if the stream parallelizes.
//
// CONSTRAINT: (PDEPTH-1)*FRAGS must fit the gfx12 s_wait_loadcnt field (<=63). So FRAGS=8 for P<=8
// (max 56) and FRAGS=4 for P=16 (60). CLAIMCHUNK must be a multiple of PDEPTH.
//
// USER_SGPR=15: s0:1=occ  s2:3=A(64 MiB stream buf)  s4:5=B(unused)  s6:7=C(sink)  s8=K(unused)
//   s12=unused  s13=CLAIM_CEIL(total slices across all waves).  occ: [0]live [1]maxlive [2]tmin
//   [3]tmax [5]claim ctr(byte20) [16]admitted.  bytes moved = CLAIM_CEIL * FRAGS * 256 (harness computes).
//
// VGPR: v0 tid, v1 wid, v2 lane, v7=0, v8 claim base, v9 lane*8, v10 lane*32, v12:13 sink.
//       ring v[32 : 32+PDEPTH*FRAGS*2-1].
.ifndef PDEPTH
    .set PDEPTH, 2
.endif
.ifndef FRAGS
    .set FRAGS, 8                              // b64 loads per slice (1 slice/wave = FRAGS*256 bytes)
.endif
.ifndef CLAIMCHUNK
    .set CLAIMCHUNK, 256                       // slices grabbed per atomic claim (must be multiple of PDEPTH)
.endif
.ifndef SLICE_STRIDE
    .set SLICE_STRIDE, 8192                    // bytes between consecutive slices (> slice span -> fresh lines)
.endif
.ifndef BUF_MASK
    .set BUF_MASK, 0x3FFFFFF                   // wrap into a 64 MiB buffer (working set >> L2)
.endif
.set RING, 32
.set WAITN, ((PDEPTH-1)*FRAGS)                 // outstanding-load watermark = (PDEPTH-1) slices in flight
.set STEADY_ITERS, ((CLAIMCHUNK/PDEPTH) - 1)   // prologue issues 1 round of PDEPTH; steady does the rest
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_lshrrev_b32 v1, 5, v0
    v_and_b32     v2, 31, v0
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9, 3, v2                     // lane*8  (load vaddr)
    v_lshlrev_b32 v10, 5, v2                    // lane*32 (sink store vaddr)
    v_mov_b32     v12, 0                        // sink lo
    v_mov_b32     v13, 0                        // sink hi
    // ---- admission: leader (tid==0) live++, maxlive, admitted++ ----
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
    // ---- timer t0 (leader min occ[2]) ----
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

    // ================ per-wave depth-P pipeline claim loop ================
.Lclaim:
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lnoclaim
    v_mov_b32 v5, CLAIMCHUNK
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lnoclaim:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v8                 // base slice index for this chunk
    s_cmp_ge_u32 s17, s13
    s_cbranch_scc1 .Lexit
    // byte offset of first slice = (s17 * SLICE_STRIDE) & BUF_MASK
    s_mul_i32 s20, s17, SLICE_STRIDE
    s_and_b32 s20, s20, BUF_MASK

    // ---------- PROLOGUE: issue PDEPTH slices into ring slots 0..PDEPTH-1 (loadcnt -> PDEPTH*FRAGS) -------
    .set p, 0
    .rept PDEPTH
      s_add_u32  s22, s2, s20
      s_addc_u32 s23, s3, 0
      .set f, 0
      .rept FRAGS
        global_load_b64 v[RING+(p*FRAGS+f)*2:RING+(p*FRAGS+f)*2+1], v9, s[22:23] offset:(f*256)
        .set f, f+1
      .endr
      s_add_u32 s20, s20, SLICE_STRIDE
      s_and_b32 s20, s20, BUF_MASK
      .set p, p+1
    .endr

    // ---------- STEADY: STEADY_ITERS rounds, each consumes+reissues all PDEPTH slots (compile-time) ------
    s_mov_b32 s25, 0
.Lsteady:
    s_cmp_ge_u32 s25, STEADY_ITERS
    s_cbranch_scc1 .Ldrain
    .set p, 0
    .rept PDEPTH
      s_wait_loadcnt WAITN                      // oldest slice (slot p) retired; <= PDEPTH-1 slices in flight
      .set f, 0
      .rept FRAGS                               // consume slot p (RAW dep forces its loads to retire)
        v_xor_b32 v12, v12, v[RING+(p*FRAGS+f)*2]
        .set f, f+1
      .endr
      s_add_u32  s22, s2, s20                    // reissue slot p at the next address -> back to PDEPTH in flight
      s_addc_u32 s23, s3, 0
      .set f, 0
      .rept FRAGS
        global_load_b64 v[RING+(p*FRAGS+f)*2:RING+(p*FRAGS+f)*2+1], v9, s[22:23] offset:(f*256)
        .set f, f+1
      .endr
      s_add_u32 s20, s20, SLICE_STRIDE
      s_and_b32 s20, s20, BUF_MASK
      .set p, p+1
    .endr
    s_add_u32 s25, s25, 1
    s_branch .Lsteady

.Ldrain:
    s_wait_loadcnt 0x0                          // drain the final PDEPTH slices in flight, consume them
    .set p, 0
    .rept PDEPTH
      .set f, 0
      .rept FRAGS
        v_xor_b32 v13, v13, v[RING+(p*FRAGS+f)*2]
        .set f, f+1
      .endr
      .set p, p+1
    .endr
    s_branch .Lclaim

.Lexit:
    // ---- sink (prevents DCE of the whole stream) ----
    v_mov_b32 v14, v12
    v_mov_b32 v15, v13
    global_store_b64 v10, v[14:15], s[6:7]
    s_wait_storecnt 0x0
    // ---- timer t1 (leader max occ[3]) ----
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
    // ---- leader live-- ----
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
