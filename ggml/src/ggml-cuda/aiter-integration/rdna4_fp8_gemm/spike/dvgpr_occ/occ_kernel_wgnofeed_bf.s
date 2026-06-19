// occ_kernel_wgnofeed_bf.s  (gfx1201, wave32) -- MAD-305 Step 2.4 residency probe: BARRIER-FREE NOFEED.
// Same 4x4 WMMA stream as occ_kernel_wggemm2.s's NOFEED path, but with NO LDS broadcast and NO
// s_barrier_* at all. Each WAVE independently atomic-claims work units; operands are loaded ONCE into
// registers (garbage layout -- NOFEED, correctness N/A) and never touched again. Pure compute-residency
// probe: "does removing barriers/LDS let the vehicle admit more than the 192-WG wall, and does the
// identical WMMA stream run faster?" Launch with 128-thread WGs (4-wave probe) OR 32-thread WGs
// (one-wave clone) -- the kernel treats every wave independently, so WG size just changes the shape.
//
// USER_SGPR=15: s0:s1=occ  s2:s3=A  s4:s5=B  s6:s7=C  s8=K  s12=NTILES(K/32)  s13=CLAIM_CEIL(=M*N/4096)
//   (claim_ceil chosen so total WMMA = claims*K = M*N*K/4096 -> harness TF=2*M*N*K/wall stays calibrated.)
// occ: [0]=live [1]=maxlive [2]=tmin [3]=tmax [4]=- [5]=- [16]=admitted [20]=claim counter.
// VGPR: v0 tid, v1 wid, v2 lane, v7=0, v8 ti, v9 lane*8, v10 lane*32; acc v[32:159], fa v[160:175],
//       fb v[176:191].  ~192 VGPR (field 24).
.ifndef CLAIMCHUNK
    .set CLAIMCHUNK, 1                      // work-units grabbed per atomic claim. >1 = fewer atomics on the
.endif                                      //   shared claim counter -> tests whether atomic contention is the cap.
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_lshrrev_b32 v1, 5, v0
    v_and_b32     v2, 31, v0
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9, 3, v2                 // lane*8 (operand load vaddr)
    v_lshlrev_b32 v10, 5, v2                // lane*32 (C store vaddr)
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
    // ---- load operands ONCE into registers (garbage, resident): 8 A frags + 8 B frags ----
    .set f, 0
    .rept 8
      global_load_b64 v[160+f*2:160+f*2+1], v9, s[2:3] offset:(f*256)
      .set f, f+1
    .endr
    .set f, 0
    .rept 8
      global_load_b64 v[176+f*2:176+f*2+1], v9, s[4:5] offset:(f*256)
      .set f, f+1
    .endr
    s_wait_loadcnt 0x0
    // ---- zero accumulators v[32:159] ----
    .set i, 0
    .rept 128
      v_mov_b32 v[32+i], 0
      .set i, i+1
    .endr

    // ================ BARRIER-FREE per-wave claim loop ================
.Lclaim:
    // each wave's lane0 (lane==0) atomic-claims one work unit; readfirstlane broadcasts within the wave
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lnoclaim
    v_mov_b32 v5, CLAIMCHUNK
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lnoclaim:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v8            // base work-unit index (from this wave's lane0)
    s_cmp_ge_u32 s17, s13
    s_cbranch_scc1 .Lexit
    // ---- CLAIMCHUNK work-units, each = NTILES * 32 WMMA back-to-back (one atomic per chunk) ----
    s_mov_b32 s28, 0                        // chunk counter
.Lchunk:
    s_mov_b32 s26, 0
.Lkt:
    .set kk, 0
    .rept 2
      .set mi, 0
      .rept 4
        .set ni, 0
        .rept 4
          v_wmma_f32_16x16x16_fp8_fp8 v[32+(mi*4+ni)*8:32+(mi*4+ni)*8+7], v[160+(kk*4+mi)*2:160+(kk*4+mi)*2+1], v[176+(kk*4+ni)*2:176+(kk*4+ni)*2+1], v[32+(mi*4+ni)*8:32+(mi*4+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set kk, kk+1
    .endr
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lkt
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, CLAIMCHUNK
    s_cbranch_scc1 .Lchunk
    s_branch .Lclaim

.Lexit:
    // ---- store acc[0][0] (sink, prevents DCE) to C + ti*... minimal ----
    global_store_b128 v10, v[32:35], s[6:7]
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
