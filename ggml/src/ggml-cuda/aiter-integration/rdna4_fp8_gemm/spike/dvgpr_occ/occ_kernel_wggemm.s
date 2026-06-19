// ============================================================================================
// DEPRECATED / DIAGNOSTIC ONLY (2026-06-17). This grid-stride-over-TGID_X variant DOES NOT WORK:
// raw-PM4 DISPATCH_DIRECT does not deliver TGID_X to any SGPR on gfx1201 (proven by occ_kernel_wgdiag.s
// / --sgpr-probe; mneme 5516935d). Kept only as the diagnostic that established that. The CANONICAL
// wave-group path is occ_kernel_wglds.s (atomic-claim + LDS-broadcast, --wglds-smoke). Do not build on
// this file.
// ============================================================================================
// occ_kernel_wggemm.s  (gfx1201, wave32) -- 4-WAVE COOPERATIVE WORKGROUP fp8 WMMA GEMM (MAD-305 wave-group).
//
// The wave-group vehicle: a 128-thread workgroup = TWM x TWN waves cooperates on one logical 128x128
// C tile; wave wid owns the (wave_m, wave_n) 64x64 quadrant (4x4 frags). The thesis (design doc
// 2026-06-17-rdna4-dynvgpr-wavegroup-fp8-gemm) is logical-tile reuse > per-wave-private reuse: a
// wave GROUP exposes a big logical tile + shares A through LDS so dyn-VGPR accumulators buy reuse
// WITHOUT collapsing residency (the failure mode of the single-wave fat micro-batch).
//
// TILE CLAIM -- DEVIATION from the design's atomic-queue + LDS-broadcast claim, deliberate:
//   tiles are mapped by GRID-STRIDE over TGID_X (s15). TGID_X is already workgroup-uniform, so all
//   TWM*TWN waves see the same ti with NO atomic and NO LDS broadcast. Rationale: (1) uniform GEMM
//   work needs no dynamic load-balancing queue; (2) it confines the LDS+barrier mechanism to Phase 2
//   (A-sharing), where LDS is actually required -- Phase 1 introduces only the 128-thread workgroup;
//   (3) grid-stride amortizes the Phase-4 dyn-VGPR grow/shrink over a whole workgroup lifetime (one
//   grow/shrink, not per atomic batch). Flagged for review.
//
// ============================ PHASE 1: SKELETON + SMOKE (SMOKE=1) ============================
// No compute. Each wave's lane 0 writes a decode mark to C[ti*4 + wid]:
//      mark = (tile_row<<20) | (tile_col<<8) | (wave_m<<4) | wave_n
// proving (a) the workgroup formed with TWM*TWN waves, (b) each wave computed the right
// wave_m/wave_n, (c) grid-stride tile decode covers each ti in [0,TOTAL) exactly once.
//
// User data (USER_SGPR=15, s0..s14); s15 = TGID_X (workgroup index):
//   s0:s1 = occ (atomic live/maxlive/timer buffer, same layout as occ_kernel_mbgemm.s)
//   s2:s3 = C base    s4 = TOTAL tiles    s5 = nWG (grid stride, in workgroups)
//   s6 = NTL_MASK (N tiles - 1, power of two)    s7 = NTL_LOG2
// Scratch: s16 exec-save, s17 ti, s18 tile_row, s19 tile_col, s20:s21 store tmp, s28:s29 store base,
//   s30:s31 timer.  VGPRs: v0 tid, v1 wid, v2 lane, v3 wave_m, v4 wave_n, v5 mark, v6 store-off, v7 tmp.

.ifndef TWM
    .set TWM, 2
.endif
.ifndef TWN
    .set TWN, 2
.endif
.ifndef SMOKE
    .set SMOKE, 1
.endif
.set WAVES, (TWM*TWN)

// LOG2TWN / (TWN-1): wave_m = wid / TWN, wave_n = wid % TWN (TWN a power of two).
.set TWN_MASK, (TWN - 1)
.if TWN == 1
    .set LOG2TWN, 0
.elseif TWN == 2
    .set LOG2TWN, 1
.elseif TWN == 4
    .set LOG2TWN, 2
.else
    .error "TWN must be 1, 2, or 4"
.endif

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- per-thread identity: wid = tid>>5, lane = tid&31, wave_m = wid>>LOG2TWN, wave_n = wid&TWN_MASK ----
    v_lshrrev_b32 v1, 5, v0
    v_and_b32     v2, 31, v0
    v_lshrrev_b32 v3, LOG2TWN, v1
    v_and_b32     v4, TWN_MASK, v1

    // ---- workgroup-leader (tid==0) admission counter: live++, maxlive=max, total++ ----
    v_mov_b32 v7, 0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v5, 1
    global_atomic_add_u32 v6, v7, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV   // v6 = old live
    s_wait_loadcnt 0x0
    v_add_nc_u32 v6, v6, 1
    global_atomic_max_u32 v7, v6, s[0:1] offset:4 scope:SCOPE_DEV                  // maxlive
    global_atomic_add_u32 v7, v5, s[0:1] offset:16 scope:SCOPE_DEV                 // total admitted
.Lafter_inc:
    s_mov_b32 exec_lo, s16

    // ---- timer t0 (workgroup-leader min start) ----
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

    // ============ GRID-STRIDE TILE LOOP: ti = TGID_X, += nWG ============
    s_mov_b32 s17, s15                    // ti = TGID_X
.Ltile_loop:
    s_cmp_ge_u32 s17, s4                  // ti >= TOTAL -> done
    s_cbranch_scc1 .Ltiles_done
    // decode: tile_row = ti >> NTL_LOG2 ; tile_col = ti & NTL_MASK
    s_lshr_b32 s18, s17, s7
    s_and_b32  s19, s17, s6
.if SMOKE
    // ---- SMOKE mark = (tile_row<<20)|(tile_col<<8)|(wave_m<<4)|wave_n ; per-wave lane-0 store ----
    v_lshl_or_b32 v5, v3, 4, v4           // (wave_m<<4) | wave_n
    v_mov_b32 v7, s19
    v_lshl_or_b32 v5, v7, 8, v5           // | (tile_col<<8)
    v_mov_b32 v7, s18
    v_lshl_or_b32 v5, v7, 20, v5          // | (tile_row<<20)
    v_lshlrev_b32 v6, 2, v1               // store offset = wid*4
    // store base = C + ti*16  (4 u32 marks per tile)
    s_lshl_b32 s20, s17, 4
    s_add_u32  s28, s2, s20
    s_addc_u32 s29, s3, 0
    // gate to lane 0 of EACH wave (lane==0) so exactly one store/wave
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_store
    s_wait_alu 0xfffe                      // VALU(v5,v6) -> VMEM hazard guard (no compiler here)
    global_store_b32 v6, v5, s[28:29]
.Lafter_store:
    s_mov_b32 exec_lo, s16
.endif
    s_add_u32 s17, s17, s5                 // ti += nWG
    s_branch .Ltile_loop

.Ltiles_done:
    s_wait_storecnt 0x0                    // smoke stores must land before exit
    // ---- timer t1 (workgroup-leader max end) ----
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
    // ---- workgroup-leader live-- (queue drains to 0 when all workgroups retire) ----
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
