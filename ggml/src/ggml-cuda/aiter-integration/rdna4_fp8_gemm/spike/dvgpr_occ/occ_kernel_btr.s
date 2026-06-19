// occ_kernel_btr.s  (gfx1201, wave32) -- MAD-305 Step A rung 6: is B's global_load_tr_b64 the FED wall?
//
// Step A + ladder proved the loads, 4-wave shape, barrier cadence, and LDS A-share are ALL cheap
// (400-880x the real GEMM's 2.7 GB/s feed). The one big untested structural difference is B's fp8
// TRANSPOSE load `global_load_tr_b64` under the REAL preshuffled-B address pattern. This probe isolates
// it (GPT rungs 6a-6d). No WMMA / no LDS / no barrier; fixed NSLICES/wave; per-wave unique serial seeds
// the B address. bytes = launched_waves * NSLICES * 8 frags * 256 (harness).
//
//   TR=1 BADDR=0  (6a) global_load_tr_b64, synthetic cache-friendly stride  -> tr INSTRUCTION cost
//   TR=1 BADDR=1  (6b) global_load_tr_b64, REAL Bshuf addressing            -> real address pattern
//   TR=1 BADDR=1 + low residency (6c, harness VGPR/LDS)                     -> residency interaction
//   TR=0 BADDR=1  (6d) ordinary global_load_b64, same Bshuf addressing      -> negative control
//
// Real B recipe (from occ_kernel_wggemm2.s): s9=NT*256 (NT=N/16); per "slice" (= one K-tile of B) =
//   2 kk-groups of FN=4 frags: kk0 @ s[Bdesc] offset ni*256, kk1 @ s[Bdesc]+s9 offset ni*256; vaddr=lane*8;
//   per-slice advance s[Bdesc] += 2*s9. Wave seed offset = (wser*2048 + wave_n*1024) & BUF_MASK.
//
// USER_SGPR=15: s0:1=occ s2:3=A(unused) s4:5=Bshuf(64 MiB) s6:7=C(sink) s8=unused s9=NT*256
//   s12=unused s13=NSLICES.  occ:[0]live [1]maxlive(waves) [2]tmin [3]tmax [5]wave-serial ctr(byte20).
.ifndef TR
    .set TR, 1
.endif
.ifndef BADDR
    .set BADDR, 1                              // 0 = synthetic cache-friendly stride, 1 = real Bshuf
.endif
.ifndef FN
    .set FN, 4
.endif
.ifndef SYN_STRIDE
    .set SYN_STRIDE, 8192                      // synthetic per-slice stride (BADDR=0)
.endif
.ifndef BUF_MASK
    .set BUF_MASK, 0x3FFFFFF                   // 64 MiB
.endif
.macro BLOAD dst, sbase, off
.if TR
    global_load_tr_b64 \dst, v9, \sbase offset:\off
.else
    global_load_b64 \dst, v9, \sbase offset:\off
.endif
.endm
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_lshrrev_b32 v1, 5, v0                     // wid
    v_and_b32     v2, 31, v0                    // lane
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9, 3, v2                     // lane*8 (B trfeed vaddr)
    v_lshlrev_b32 v10, 5, v2                    // lane*32 (sink store)
    v_mov_b32     v12, 0
    v_mov_b32     v13, 0
    v_readfirstlane_b32 s14, v1                 // wid (uniform per wave)
    s_and_b32 s15, s14, 1                       // wave_n = wid & 1
    // ---- per-wave startup: unique serial (occ[5]) + live++/maxlive ----
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_start
    v_mov_b32 v5, 1
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    global_atomic_add_u32 v6, v7, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v6, v6, 1
    global_atomic_max_u32 v7, v6, s[0:1] offset:4 scope:SCOPE_DEV
.Lafter_start:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v8                 // wave serial
    // ---- timer t0 ----
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
    // ---- address seed s20 ----
.if BADDR == 1
    s_mul_i32 s20, s17, 2048                    // wser * 2048 (tile_col-like spread)
    s_lshl_b32 s27, s15, 10                     // wave_n * 1024
    s_add_u32  s20, s20, s27
.else
    s_mul_i32 s20, s17, SYN_STRIDE
.endif
    s_and_b32 s20, s20, BUF_MASK
    s_lshl_b32 s43, s9, 1                       // 2*NT*256 = per-slice B advance (BADDR=1)

    // ================ fixed-length slice loop (NSLICES = s13) ================
    s_mov_b32 s18, 0
.Lslice:
    s_cmp_ge_u32 s18, s13
    s_cbranch_scc1 .Lexit
    s_add_u32  s22, s4, s20
    s_addc_u32 s23, s5, 0
.if BADDR == 1
    // ---- real Bshuf: 2 kk-groups of FN frags (kk1 @ +s9) ----
    .set ni, 0
    .rept FN
      BLOAD v[32+ni*2:32+ni*2+1], s[22:23], (ni*256)
      .set ni, ni+1
    .endr
    s_add_u32  s24, s22, s9
    s_addc_u32 s25, s23, 0
    .set ni, 0
    .rept FN
      BLOAD v[32+(FN+ni)*2:32+(FN+ni)*2+1], s[24:25], (ni*256)
      .set ni, ni+1
    .endr
.else
    // ---- synthetic: 2*FN contiguous frags at offset f*256 ----
    .set f, 0
    .rept (2*FN)
      BLOAD v[32+f*2:32+f*2+1], s[22:23], (f*256)
      .set f, f+1
    .endr
.endif
    s_wait_loadcnt 0x0
    // ---- consume ----
    .set f, 0
    .rept (2*FN)
      v_xor_b32 v12, v12, v[32+f*2]
      .set f, f+1
    .endr
    // ---- advance ----
.if BADDR == 1
    s_add_u32 s20, s20, s43                     // += 2*NT*256
.else
    s_add_u32 s20, s20, SYN_STRIDE
.endif
    s_and_b32 s20, s20, BUF_MASK
    s_add_u32 s18, s18, 1
    s_branch .Lslice

.Lexit:
    v_mov_b32 v14, v12
    v_mov_b32 v15, v13
    global_store_b64 v10, v[14:15], s[6:7]
    s_wait_storecnt 0x0
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
