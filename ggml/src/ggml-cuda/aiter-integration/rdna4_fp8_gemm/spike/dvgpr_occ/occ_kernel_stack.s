// occ_kernel_stack.s  (gfx1201, wave32) -- MAD-305 STACK LADDER: rebuild the fast feed structure from a
// known-good minimal core, adding exactly ONE real-GEMM obligation per rung, with TF / GB/s / proof at
// every rung. The first rung that collapses from "fast" toward the real kernel's 2.7 GB/s / 1.4 TF NAMES
// the wall. (Replaces the single-variable neuter approach after the rung-7 PROFILE 70x was proven a
// measurement artifact -- see RESULT_WGGEMM.md rung 9.)
//
// Planned ladder (RUNG defsym):
//   1  load-only truthful base : 1-wave WG, no LDS, no barrier, real A+B buffers streamed, VERIFIABLE
//                                global checksum (every loaded dword summed -> occ[7], CPU-checked).  <-- THIS FILE
//   2  + real K-loop address progression (tile decode, s20 += 2*NT*256, A row*K+k).
//   3  + 4-wave (128-thread) WG shape, independent waves (no cooperation yet).
//   4  + LDS A publication (global A -> ds_store -> barrier -> ds_load), traffic-proofed.
//   5  + B global_load_tr_b64 with the real wait pattern (A+B feed, still no WMMA).
//   6  + accumulator / WMMA (1x1 -> 2x2 -> 4x4), acc00 oracle.
//   7  + output store.
//
// RULE: never credit TF/GB/s unless the third figure proves the work happened. Rung 1 proof = occ[7]
// checksum == CPU sum over EXACTLY the streamed slices (so no DCE / no early-exit can fake the bandwidth).
//
// USER_SGPR=15: s0:1=occ  s2:3=A(stream buf, >=64MiB+slack)  s4:5=B(stream buf)  s6:7=C(unused rung 1)
//   s8=K(unused) s9=SLICE_STRIDE s10=BUF_MASK s11..s12 unused  s13=CLAIM_CEIL (total slices; mult of CLAIMCHUNK).
//   occ: [0]live [1]maxlive [2]tmin [3]tmax [5]claim ctr(byte20) [7]checksum(byte28) [16]admitted(byte64).
//   bytes moved = CLAIM_CEIL * FRAGS * 256 * 2 (A+B); harness computes GB/s.
//
// VGPR: v0 tid, v2 lane(=tid, 1 wave), v5/v6 scratch, v7=0, v9 lane*8, v12 running u32 sum,
//       A frags v[16:16+2*FRAGS-1], B frags v[48:48+2*FRAGS-1].
.ifndef RUNG
    .set RUNG, 1
.endif
.ifndef FRAGS
    .set FRAGS, 8                                 // b64 loads per buffer per slice (FRAGS*256 = 2048 B/buf/slice)
.endif
.ifndef CLAIMCHUNK
    .set CLAIMCHUNK, 256                          // slices grabbed per atomic claim
.endif
.ifndef SLICE_STRIDE
    .set SLICE_STRIDE, 8192                       // bytes between consecutive slices (fresh lines)
.endif
.ifndef BUF_MASK
    .set BUF_MASK, 0x3FFFFFF                      // wrap into a 64 MiB window (working set >> L2)
.endif
.if RUNG != 1
    .error "occ_kernel_stack.s: only RUNG==1 implemented so far"
.endif
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_and_b32     v2, 31, v0                       // lane (1-wave WG -> lane == tid)
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9, 3, v2                        // lane*8 (load vaddr)
    v_mov_b32     v12, 0                           // data checksum: 4 PARALLEL accumulators v12..v15 (break the
    v_mov_b32     v13, 0                           //   serial dependency so the consume isn't the bottleneck)
    v_mov_b32     v14, 0
    v_mov_b32     v15, 0
    v_mov_b32     v11, 0                           // per-wave slice counter (count proof)
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
    global_atomic_add_u32 v7, v5, s[0:1] offset:64 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s16
    v_mov_b32 v7, 0
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

    // ================ per-wave claim loop: each claim = CLAIMCHUNK slices [s17, s17+CLAIMCHUNK) ===========
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
    v_readfirstlane_b32 s17, v8                    // base slice index for this chunk
    s_cmp_ge_u32 s17, s13
    s_cbranch_scc1 .Lexit
    s_mov_b32 s18, 0                               // local slice counter 0..CLAIMCHUNK-1
.Lslice:
    // ---- absolute slice index = s17 + s18 ; byte base = (idx * stride[s9]) & mask[s10] (userdata-driven) ----
    s_add_u32 s21, s17, s18
    s_mul_i32 s20, s21, s9                          // s9 = SLICE_STRIDE (userdata[9])
    s_and_b32 s20, s20, s10                         // s10 = BUF_MASK    (userdata[10])
    // ---- issue FRAGS A loads (s[2:3]) ----
    s_add_u32  s22, s2, s20
    s_addc_u32 s23, s3, 0
    .set f, 0
    .rept FRAGS
      global_load_b64 v[16+f*2:16+f*2+1], v9, s[22:23] offset:(f*256)
      .set f, f+1
    .endr
    // ---- issue FRAGS B loads (s[4:5]) ----
    s_add_u32  s24, s4, s20
    s_addc_u32 s25, s5, 0
    .set f, 0
    .rept FRAGS
      global_load_b64 v[48+f*2:48+f*2+1], v9, s[24:25] offset:(f*256)
      .set f, f+1
    .endr
    s_wait_loadcnt 0x0                             // ALL A+B loads retired (RAW dep below forces it -> no DCE)
    // ---- consume: 4 PARALLEL chains (A lo/hi -> v12/v13, B lo/hi -> v14/v15); sum == every loaded dword ----
    .set f, 0
    .rept FRAGS
      v_add_nc_u32 v12, v12, v[16+f*2]
      v_add_nc_u32 v13, v13, v[16+f*2+1]
      v_add_nc_u32 v14, v14, v[48+f*2]
      v_add_nc_u32 v15, v15, v[48+f*2+1]
      .set f, f+1
    .endr
    v_add_nc_u32 v11, v11, 1                       // count proof: this wave processed one more slice
    s_add_u32 s18, s18, 1
    s_cmp_lt_u32 s18, CLAIMCHUNK
    s_cbranch_scc1 .Lslice
    s_branch .Lclaim

.Lexit:
    // ---- count proof: leader lane atomic-adds this wave's slice count -> occ[6] (CPU verifies == CEIL) ----
    s_mov_b32 s26, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lno_cnt
    global_atomic_add_u32 v7, v11, s[0:1] offset:24 scope:SCOPE_DEV
.Lno_cnt:
    s_mov_b32 exec_lo, s26
    // ---- data proof: combine the 4 parallel accumulators, then all 32 lanes atomic-add -> occ[7] ----
    v_add_nc_u32 v12, v12, v13
    v_add_nc_u32 v14, v14, v15
    v_add_nc_u32 v12, v12, v14
    global_atomic_add_u32 v7, v12, s[0:1] offset:28 scope:SCOPE_DEV
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
