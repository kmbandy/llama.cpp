// occ_kernel_mb.s  (gfx1201, wave32) -- MICRO-BATCHING DYNAMIC-QUEUE kernel (MAD-305).
//
// The user's design, on the proven dyn-VGPR vehicle: a fixed pool of PERSISTENT waves pulls
// output-tiles from a GLOBAL ATOMIC WORK-QUEUE and processes them non-stop until the queue
// drains. No static partition -> fast waves pull more, slow ones fewer, it self-levels (a big
// or complex tile just lets the others pull ahead and the queue catches up when tiles get
// cheaper). Per tile it MICRO-BATCHES the registers: s_alloc_vgpr GROWS to the accumulator
// footprint, computes, SHIPS (stores C), then s_alloc_vgpr SHRINKS back to lean -- the shrink
// is what lets the hardware admit MORE resident waves (the proven 1.60x occupancy lever), so
// the matrix unit stays fed.
//
// Built directly on occ_kernel.s (proven: s_alloc grow/shrink + accumulating fp8 WMMA,
// bit-correct on the dynamically-grown registers). STAGE 1: the per-tile compute reuses ONE
// A/B fragment held in the lean block (compute-isolated, like the 307 TF microbench) so this
// run isolates the QUEUE + dyn-VGPR micro-batch behavior; per-tile distinct A/B feed = Stage 2.
//
// User data (USER_SGPR=8):
//   s[0:1] = occ buffer : [live@0, maxlive@4, min(start)@8, max(end)@12, total@16, nextTile@20]
//   s[2:3] = fragIn     : A frag @0, B frag @256
//   s[4:5] = C base     : tile ti -> C + ti*1024  (256 f32 per tile)
//   s6     = KDEPTH      : accumulate iterations per tile (work per micro-batch)
//   s7     = TOTAL_TILES : queue length
//   s8     = TGID_X (auto)   s9..= scratch
//   v0     = lane (TIDIG_COMP_CNT, set by harness RSRC2)
//
// Encodings lifted verbatim from occ_kernel.s (already verified vs llvm-objdump).
.ifndef NACC
    .set NACC, 8
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 1
.endif
.if NACC >= 16
    .set FATREGS, 160            // v32..v159  (needs SQ_DYN_VGPR.BLOCK_SIZE=1 to exceed 128)
.elseif NACC >= 12
    .set FATREGS, 128            // v32..v127
.else
    .set FATREGS, 96             // v32..v95   (fits the default 128 dyn cap -- no umr needed)
.endif

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_mov_b32 v4, 0
    // ---- lane-0-only admission occupancy counter (labels the run) ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s10, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV
    global_atomic_add_u32 v4, v2, s[0:1] offset:16 scope:SCOPE_DEV   // total waves launched
.Lafter_inc:
    s_mov_b32 exec_lo, s10
    // (A/B are reloaded INSIDE each tile after the grow -- self-contained micro-batch, and the
    //  load+wait gives s_alloc_vgpr time to settle before the first accumulator write.)
    // ---- timer t0 (whole persistent span): occ[2] = min(start) ----
    s_sendmsg_rtn_b64 s[12:13], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s10, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v5, s12
    global_atomic_min_u32 v4, v5, s[0:1] offset:8 scope:SCOPE_DEV
.Lafter_t0:
    s_mov_b32 exec_lo, s10

    // ================= PERSISTENT WORK-QUEUE LOOP =================
.Ltile_loop:
    // lane-0 grabs the next tile index: returning atomic add on occ[nextTile @ offset 20]
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s10, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_grab
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lafter_grab:
    s_mov_b32 exec_lo, s10
    v_readlane_b32 s14, v3, 0            // s14 = tile index grabbed by lane 0 (broadcast to scalar)
    s_cmp_ge_u32 s14, s7                 // ti >= TOTAL_TILES -> queue drained
    s_cbranch_scc1 .Ltiles_done
.if DYNVGPR
    s_wait_loadcnt 0x0                   // drain in-flight VMEM before growing the block
    s_wait_storecnt 0x0
    s_alloc_vgpr FATREGS                 // ---- GROW (micro-batch open) ----
.endif
    // ---- reload A/B INSIDE the tile: self-contained micro-batch (no reliance on the lean block
    //      surviving the grow/shrink churn) AND the load+wait gives s_alloc time to settle ----
    v_lshlrev_b32 v6, 3, v0              // lane*8
    global_load_b64 v[16:17], v6, s[2:3]
    global_load_b64 v[18:19], v6, s[2:3] offset:256
    s_wait_loadcnt 0x0
    // ---- peel: init NACC accumulators (srcC = 0) ----
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[16:17], v[18:19], 0
    // ---- KDEPTH-1 accumulate iterations (srcC = acc) ----
    s_mov_b32 s11, s6
    s_sub_u32 s11, s11, 1
    s_cmp_eq_u32 s11, 0
    s_cbranch_scc1 .Lkdone
.Lkloop:
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[16:17], v[18:19], v[32:39]
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[16:17], v[18:19], v[40:47]
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[16:17], v[18:19], v[48:55]
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[16:17], v[18:19], v[56:63]
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[16:17], v[18:19], v[64:71]
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[16:17], v[18:19], v[72:79]
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[16:17], v[18:19], v[80:87]
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95], v[16:17], v[18:19], v[88:95]
    s_sub_u32 s11, s11, 1
    s_cmp_lg_u32 s11, 0
    s_cbranch_scc1 .Lkloop
.Lkdone:
    // ---- SHIP: store acc0 tile (256 f32 = 1024 B) to C[ti] ----
    v_lshlrev_b32 v7, 5, v0             // lane*32 (recompute post-compute; store offset within tile)
    s_lshl_b32 s15, s14, 10              // ti * 1024
    s_add_u32 s16, s4, s15
    s_addc_u32 s17, s5, 0
    global_store_b128 v7, v[32:35], s[16:17]
    global_store_b128 v7, v[36:39], s[16:17] offset:16
    s_wait_storecnt 0x0
.if DYNVGPR
    s_alloc_vgpr 32                      // ---- SHRINK back to lean (micro-batch close) ----
.endif
    s_branch .Ltile_loop

.Ltiles_done:
    // ---- timer t1: occ[3] = max(end) ----
    s_sendmsg_rtn_b64 s[12:13], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s10, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v5, s12
    global_atomic_max_u32 v4, v5, s[0:1] offset:12 scope:SCOPE_DEV
.Lafter_t1:
    s_mov_b32 exec_lo, s10
    // ---- live-- (so the harness sees the persistent pool drain to 0) ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s10, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v2, -1
    global_atomic_add_u32 v4, v2, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s10
    s_endpgm
    .size occ_kernel, .-occ_kernel
