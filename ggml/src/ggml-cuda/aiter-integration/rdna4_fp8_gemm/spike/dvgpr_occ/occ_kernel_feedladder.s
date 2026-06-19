// occ_kernel_feedladder.s  (gfx1201, wave32) -- MAD-305 Step A localization ladder (GPT plan, rungs 1-5).
//
// Step A proved: a barrier-free / LDS-free load stream runs ~123 GB/s (45x the real GEMM's 2.7 GB/s feed),
// so the loads are NOT the wall. The wall is the machinery the real 4-wave vehicle wraps around the loads:
// the per-K-tile workgroup barrier, the LDS A round-trip, and the 4-wave lock-step coupling. This kernel
// adds those couplings back onto the 123 GB/s baseline ONE AT A TIME to find the collapse point:
//
//   rung 1: WAVES=1 BARRIER=0 LDSMODE=0   1-wave, no LDS, no barrier   (re-baseline of feedpipe P=1)
//   rung 2: WAVES=4 BARRIER=0 LDSMODE=0   4-wave WG, no LDS, no barrier (4-wave shape alone)
//   rung 3: WAVES=4 BARRIER=1 LDSMODE=0   4-wave + s_barrier per slice  (barrier cadence / lockstep)
//   rung 4: WAVES=4         LDSMODE=1     4-wave + LDS round-trip, NO global A (publication cost, no mem)
//   rung 5: WAVES=4         LDSMODE=2     4-wave + global A -> LDS -> barrier -> ds_load (real A-share path)
//
// No atomic-claim work distribution: every wave does a FIXED NSLICES (s13) slices, so all waves of a WG
// hit the same number of barriers (no deadlock) while touching wave-distinct addresses (seeded by a single
// startup atomic = unique wave serial). bytes moved = launched_waves * NSLICES * FRAGS * 256 (harness).
//
// USER_SGPR=15: s0:1=occ s2:3=A(64 MiB) s4:5=B(unused) s6:7=C(sink) s8=unused s12=unused s13=NSLICES.
//   occ: [0]live [1]maxlive(waves) [2]tmin [3]tmax [5]wave-serial ctr(byte20).
// VGPR: v0 tid,v1 wid,v2 lane,v7=0,v8 wser,v9 lane*8,v10 lane*32,v11 LDS base,v12:13 sink,
//   v[32:32+FRAGS*2-1] load/store-src, v[64:64+FRAGS*2-1] ds_load dst (LDSMODE>0).
.ifndef WAVES
    .set WAVES, 4
.endif
.ifndef FRAGS
    .set FRAGS, 8
.endif
.ifndef BARRIER
    .set BARRIER, 0
.endif
.ifndef LDSMODE
    .set LDSMODE, 0                            // 0=none 1=LDS round-trip from regs 2=global A->LDS->ds_load
.endif
.ifndef SLICE_STRIDE
    .set SLICE_STRIDE, 8192
.endif
.ifndef BUF_MASK
    .set BUF_MASK, 0x3FFFFFF                   // 64 MiB stream buffer
.endif
.set LDS_WSTRIDE, (FRAGS*256)                  // per-wave LDS region bytes (FRAGS frags * 256 B/frag)
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_lshrrev_b32 v1, 5, v0                     // wid
    v_and_b32     v2, 31, v0                    // lane
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9, 3, v2                     // lane*8
    v_lshlrev_b32 v10, 5, v2                    // lane*32 (sink store)
    v_mov_b32     v12, 0
    v_mov_b32     v13, 0
    // LDS base = wid*LDS_WSTRIDE + lane*8  (only used when LDSMODE>0)
    v_mul_lo_u32  v11, v1, LDS_WSTRIDE
    v_add_nc_u32  v11, v11, v9
    // ---- per-wave startup: unique wave serial (occ[5]) + live++/maxlive (occ[0]/[1]) ----
    v_cmp_eq_u32 vcc_lo, 0, v2                  // lane==0 (per WAVE)
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_start
    v_mov_b32 v5, 1
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV   // wser
    global_atomic_add_u32 v6, v7, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV             // live++
    s_wait_loadcnt 0x0
    v_add_nc_u32 v6, v6, 1
    global_atomic_max_u32 v7, v6, s[0:1] offset:4 scope:SCOPE_DEV                            // maxlive
.Lafter_start:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v8                 // wave serial -> address seed
    // ---- timer t0 (per-wave lane0 min occ[2]) ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v5, s30
    global_atomic_min_u32 v7, v5, s[0:1] offset:8 scope:SCOPE_DEV
.Lafter_t0:
    s_mov_b32 exec_lo, s16
    // address seed s20 = (wser * SLICE_STRIDE) & BUF_MASK
    s_mul_i32 s20, s17, SLICE_STRIDE
    s_and_b32 s20, s20, BUF_MASK
.if LDSMODE == 1
    // zero the store-source regs once (no global A in this rung)
    .set z, 0
    .rept FRAGS
      v_mov_b32 v[32+z*2], 0
      v_mov_b32 v[32+z*2+1], 0
      .set z, z+1
    .endr
.endif

    // ================ fixed-length slice loop (NSLICES = s13) ================
    s_mov_b32 s18, 0
.Lslice:
    s_cmp_ge_u32 s18, s13
    s_cbranch_scc1 .Lexit
.if LDSMODE != 1
    // ---- global A load ----
    s_add_u32  s22, s2, s20
    s_addc_u32 s23, s3, 0
    .set f, 0
    .rept FRAGS
      global_load_b64 v[32+f*2:32+f*2+1], v9, s[22:23] offset:(f*256)
      .set f, f+1
    .endr
    s_wait_loadcnt 0x0
.endif
.if LDSMODE > 0
    // ---- LDS publication: store -> barrier -> load (the A-share round trip) ----
    .set f, 0
    .rept FRAGS
      ds_store_b64 v11, v[32+f*2:32+f*2+1] offset:(f*256)
      .set f, f+1
    .endr
    s_wait_dscnt 0x0
    s_barrier_signal -1
    s_barrier_wait -1
    .set f, 0
    .rept FRAGS
      ds_load_b64 v[64+f*2:64+f*2+1], v11 offset:(f*256)
      .set f, f+1
    .endr
    s_wait_dscnt 0x0
.endif
    // ---- consume (force the loads/ds_loads to retire) ----
    .set f, 0
    .rept FRAGS
.if LDSMODE == 0
      v_xor_b32 v12, v12, v[32+f*2]
.else
      v_xor_b32 v12, v12, v[64+f*2]
.endif
      .set f, f+1
    .endr
.if (BARRIER == 1) && (LDSMODE == 0)
    // ---- standalone barrier cadence (rung 3) ----
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    // ---- advance ----
    s_add_u32 s20, s20, SLICE_STRIDE
    s_and_b32 s20, s20, BUF_MASK
    s_add_u32 s18, s18, 1
    s_branch .Lslice

.Lexit:
    // ---- sink ----
    v_mov_b32 v14, v12
    v_mov_b32 v15, v13
    global_store_b64 v10, v[14:15], s[6:7]
    s_wait_storecnt 0x0
    // ---- timer t1 (per-wave lane0 max occ[3]) ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v5, s30
    global_atomic_max_u32 v7, v5, s[0:1] offset:12 scope:SCOPE_DEV
.Lafter_t1:
    s_mov_b32 exec_lo, s16
    // ---- live-- ----
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v5, -1
    global_atomic_add_u32 v7, v5, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s16
    s_endpgm
    .size occ_kernel, .-occ_kernel
