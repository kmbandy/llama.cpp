// occ_kernel_wglds.s  (gfx1201, wave32) -- WAVE-GROUP ATOMIC-CLAIM + LDS-BROADCAST smoke (MAD-305).
//
// The Phase-1 pivot. Raw-PM4 TGID_X delivery is unavailable on this DISPATCH_DIRECT path (proven by
// occ_kernel_wgdiag.s: no SGPR carries the workgroup id), so the wave-group kernel distributes tiles
// the way the rest of this harness already does -- a GLOBAL ATOMIC work-queue -- and shares the
// claimed tile across the 4 waves via LDS + barrier (the design's original mechanism, and exactly the
// LDS+barrier machinery Phase 2's A-sharing needs). No TGID, no compute, no dyn-VGPR.
//
// Per persistent 128-thread (TWM*TWN-wave) workgroup, repeatedly:
//   1. leader (tid==0) ti = atomic_add(occ+20, 1); ds_store ti -> LDS[0]
//   2. barrier  (all waves observe LDS[0])
//   3. every wave ds_load ti <- LDS[0]; v_readfirstlane -> scalar ti
//   4. if ti >= TOTAL: all 4 waves exit together (uniform -> no partial barrier)
//   5. lane-0 of each wave writes mark = (tile_row<<20)|(tile_col<<8)|(wave_m<<4)|wave_n to C[ti*4+wid]
//   6. barrier  (all waves done reading LDS[0] before the leader overwrites it next claim)
//
// Proves: workgroup LDS allocates (RSRC2.GRANULATED_LDS_SIZE), leader->waves broadcast + barrier work,
// atomic tile distribution covers each tile exactly once with 4 correct wave marks. No reliance on s15.
//
// User data (USER_SGPR=15, s0..s14):
//   s0:s1 = occ (live/maxlive/timer + claim counter @occ[5]=offset:20)   s2:s3 = C base
//   s4 = TOTAL tiles   s6 = NTL_MASK   s7 = NTL_LOG2
// Scratch: s16 exec, s17 ti, s18 row, s19 col, s20 tmp, s28:s29 store base, s30:s31 timer.
// VGPRs: v0 tid, v1 wid, v2 lane, v3 wave_m, v4 wave_n, v5 mark, v6 storeoff, v7 ZERO(lds/atomic addr), v8 ti.
// LDS: 4 bytes at offset 0 (the ti broadcast slot).

.ifndef TWM
    .set TWM, 2
.endif
.ifndef TWN
    .set TWN, 2
.endif
.set TWN_MASK, (TWN - 1)
.if TWN == 2
    .set LOG2TWN, 1
.elseif TWN == 4
    .set LOG2TWN, 2
.else
    .set LOG2TWN, 0
.endif

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- identity ----
    v_lshrrev_b32 v1, 5, v0
    v_and_b32     v2, 31, v0
    v_lshrrev_b32 v3, LOG2TWN, v1
    v_and_b32     v4, TWN_MASK, v1
    v_mov_b32     v7, 0                 // ZERO: LDS addr 0 + atomic vaddr offset

    // ---- admission: leader (tid==0) live++, maxlive, total++ ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v5, 1
    global_atomic_add_u32 v6, v7, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v6, v6, 1
    global_atomic_max_u32 v7, v6, s[0:1] offset:4 scope:SCOPE_DEV    // v7=0 vaddr (NOT v8 -- garbage)
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

    // ============ PERSISTENT ATOMIC-CLAIM + LDS-BROADCAST LOOP ============
.Lclaim_loop:
    // (1) leader claims ti = atomic_add(occ+20, 1) -> LDS[0]
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_claim
    v_mov_b32 v5, 1
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    ds_store_b32 v7, v8                 // LDS[0] = ti
    s_wait_dscnt 0x0
.Lafter_claim:
    s_mov_b32 exec_lo, s16
    // (2) barrier: publish LDS[0] to all waves
    s_barrier_signal -1
    s_barrier_wait -1
    // (3) every wave loads ti, broadcasts to scalar
    ds_load_b32 v8, v7
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s17, v8
    // (4) drained? all 4 waves exit together (uniform ti)
    s_cmp_ge_u32 s17, s4
    s_cbranch_scc1 .Lexit
    // (5) decode + mark + lane-0 store
    s_lshr_b32 s18, s17, s7
    s_and_b32  s19, s17, s6
    v_lshl_or_b32 v5, v3, 4, v4
    v_mov_b32 v6, s19
    v_lshl_or_b32 v5, v6, 8, v5
    v_mov_b32 v6, s18
    v_lshl_or_b32 v5, v6, 20, v5
    v_lshlrev_b32 v6, 2, v1            // store off = wid*4
    s_lshl_b32 s20, s17, 4
    s_add_u32  s28, s2, s20
    s_addc_u32 s29, s3, 0
    v_cmp_eq_u32 vcc_lo, 0, v2         // lane==0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_store
    s_wait_alu 0xfffe
    global_store_b32 v6, v5, s[28:29]
.Lafter_store:
    s_mov_b32 exec_lo, s16
    // (6) barrier: all done reading LDS[0] before leader overwrites it next claim
    s_barrier_signal -1
    s_barrier_wait -1
    s_branch .Lclaim_loop

.Lexit:
    s_wait_storecnt 0x0
    // ---- timer t1 (leader) ----
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
