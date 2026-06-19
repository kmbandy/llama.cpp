// occ_kernel_bw.s  (gfx1201, wave32) -- MAD-305 CLEAN raw-PM4 STREAMING BANDWIDTH PROBE.
//
// Goal: prove the raw PM4 vehicle can move data NEAR SPEC (~640 GB/s on R9700) BEFORE any GEMM stacking.
// The stack rung-1 probe maxed at 24 GB/s (4% of peak) -- proven NOT memory-bound (cache==HBM) but capped by
// the per-slice atomic-claim / full-drain / dual-stream structure. This kernel strips ALL of that:
//
//   * ONE global atomic per WAVE -> dense worker index (TGID is dead under raw PM4; this is the only coord).
//   * Pure streaming hot loop: WIDE coalesced loads (b32/b64/b128), UNROLL-deep MLP, ONE drain per batch.
//   * NO atomics / NO LDS / NO barriers / NO per-element claim in the hot loop.
//   * Static per-worker contiguous span: worker w streams [w*SPAN, w*SPAN+SPAN) wrapped into the buffer.
//
// MODE (defsym): 0 = read-only (XOR checksum), 1 = copy (load->store), 2 = write-only (store fill).
// LDW (defsym): bytes/lane/load -- 4=b32, 8=b64, 16=b128.  UNROLL: loads in flight before a drain (MLP).
//   Constraint: UNROLL*STEP <= 4096 (load offset field). STEP = 32*LDW. So b128->UNROLL<=8, b64<=16, b32<=32.
//
// USER_SGPR=15: s0:1=occ  s2:3=buf(read src)  s4:5=buf2(write dst, copy/write)  s6:7=sink
//   s8..s10 unused  s11=BUF_MASK  s12=NWORKERS(cap; 0=no cap)  s13=STEPS(loads/worker; mult of UNROLL).
//   occ:[0]live [1]maxlive [2]tmin [3]tmax [5]worker ctr(byte20) [6]steps-done(byte24) [7]chk(byte28).
//   bytes moved = (worker ctr) * STEPS * 32 * LDW  (read or write; copy = 2x). Harness computes GB/s.
//
// VGPR: v0 tid, v2 lane, v7=0, v9 lane*LDW (load vaddr), v10 lane*LDW (store vaddr), v11 step ctr,
//       v12..v15 XOR accumulators, ring v[16 : 16+UNROLL*(LDW/4)-1].
.ifndef MODE
    .set MODE, 0                                  // 0=read 1=copy 2=write
.endif
.ifndef LDW
    .set LDW, 16                                  // 4=b32 8=b64 16=b128 (bytes per lane per load)
.endif
.ifndef UNROLL
    .set UNROLL, 8                                // loads in flight before one s_wait_loadcnt 0x0 (MLP depth)
.endif
.set LR, (LDW/4)                                  // VGPRs per lane per load
.set STEP, (32*LDW)                               // bytes advanced per coalesced wavefront load
.if LDW == 16
    .set LDW_SH, 4
.elseif LDW == 8
    .set LDW_SH, 3
.else
    .set LDW_SH, 2
.endif
.if LDW == 16
    .set STEP_SH, 9
.elseif LDW == 8
    .set STEP_SH, 8
.else
    .set STEP_SH, 7
.endif
.set RING, 16

.macro VLOAD slot, sbase
  .if LDW == 16
    global_load_b128 v[RING+\slot*4:RING+\slot*4+3], v9, \sbase offset:(\slot*STEP)
  .elseif LDW == 8
    global_load_b64  v[RING+\slot*2:RING+\slot*2+1], v9, \sbase offset:(\slot*STEP)
  .else
    global_load_b32  v[RING+\slot], v9, \sbase offset:(\slot*STEP)
  .endif
.endm
.macro VXOR slot
  .if LDW == 16
    v_xor_b32 v12, v12, v[RING+\slot*4]
    v_xor_b32 v13, v13, v[RING+\slot*4+1]
    v_xor_b32 v14, v14, v[RING+\slot*4+2]
    v_xor_b32 v15, v15, v[RING+\slot*4+3]
  .elseif LDW == 8
    v_xor_b32 v12, v12, v[RING+\slot*2]
    v_xor_b32 v13, v13, v[RING+\slot*2+1]
  .else
    v_xor_b32 v12, v12, v[RING+\slot]
  .endif
.endm
.macro VSTORE slot, sbase
  .if LDW == 16
    global_store_b128 v10, v[RING+\slot*4:RING+\slot*4+3], \sbase offset:(\slot*STEP)
  .elseif LDW == 8
    global_store_b64  v10, v[RING+\slot*2:RING+\slot*2+1], \sbase offset:(\slot*STEP)
  .else
    global_store_b32  v10, v[RING+\slot], \sbase offset:(\slot*STEP)
  .endif
.endm

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_and_b32     v2, 31, v0
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9,  LDW_SH, v2                  // lane*LDW (load vaddr)
    v_lshlrev_b32 v10, LDW_SH, v2                  // lane*LDW (store vaddr)
    v_mov_b32     v11, 0
    v_mov_b32     v12, 0
    v_mov_b32     v13, 0
    v_mov_b32     v14, 0
    v_mov_b32     v15, 0
    // for write/copy fill, seed the ring with a non-zero pattern derived from lane (MODE 2 has no loads)
    v_or_b32      v16, v2, 0x100
    // ---- live++ / maxlive (leader tid==0) ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lal
    v_mov_b32 v5, 1
    global_atomic_add_u32 v6, v7, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v6, v6, 1
    global_atomic_max_u32 v7, v6, s[0:1] offset:4 scope:SCOPE_DEV
.Lal:
    s_mov_b32 exec_lo, s16
    v_mov_b32 v7, 0
    // ---- timer t0 (leader min occ[2]) ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lt0
    v_mov_b32 v5, s30
    global_atomic_min_u32 v7, v5, s[0:1] offset:8 scope:SCOPE_DEV
.Lt0:
    s_mov_b32 exec_lo, s16

    // ================ ONE atomic per WAVE -> dense worker index (no TGID) ================
    v_cmp_eq_u32 vcc_lo, 0, v2                     // wave-leader lane (lane 0)
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lnoid
    v_mov_b32 v5, 1
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
.Lnoid:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s17, v8                    // worker_idx (broadcast from lane 0)
    // optional cap: if NWORKERS>0 and worker_idx>=NWORKERS, skip the stream
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lspan
    s_cmp_lt_u32 s17, s12
    s_cbranch_scc0 .Ldone
.Lspan:
    // base0 = (worker_idx * STEPS * STEP) & BUF_MASK ; SPAN = STEPS<<STEP_SH
    s_lshl_b32 s18, s13, STEP_SH                   // SPAN = STEPS*STEP
    s_mul_i32  s19, s17, s18                        // worker_idx*SPAN
    s_and_b32  s19, s19, s11                        // & BUF_MASK
    s_add_u32  s20, s2, s19                         // read base
    s_addc_u32 s21, s3, 0
    s_add_u32  s22, s4, s19                         // write base (copy/write)
    s_addc_u32 s23, s5, 0
    s_mov_b32  s24, 0                               // step counter

.Lstream:
.if MODE != 2
    // ---- issue UNROLL wide loads ----
    .set u, 0
    .rept UNROLL
      VLOAD u, s[20:21]
      .set u, u+1
    .endr
    s_wait_loadcnt 0x0
.endif
.if MODE == 0
    // read: XOR-consume (forces retirement + checksum)
    .set u, 0
    .rept UNROLL
      VXOR u
      .set u, u+1
    .endr
.endif
.if MODE != 0
    // copy/write: store UNROLL wide stores (copy stores loaded regs; write stores the seeded pattern)
    .set u, 0
    .rept UNROLL
      VSTORE u, s[22:23]
      .set u, u+1
    .endr
    s_wait_storecnt 0x0
    s_add_u32  s22, s22, (UNROLL*STEP)
    s_addc_u32 s23, s23, 0
.endif
.if MODE != 2
    s_add_u32  s20, s20, (UNROLL*STEP)
    s_addc_u32 s21, s21, 0
.endif
    v_add_nc_u32 v11, v11, UNROLL
    s_add_u32  s24, s24, UNROLL
    s_cmp_lt_u32 s24, s13
    s_cbranch_scc1 .Lstream

.Lexit:
    // ---- proofs: leader-lane atomic-add this wave's step count -> occ[6]; all lanes XOR-sum -> occ[7] ----
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lnocnt
    global_atomic_add_u32 v7, v11, s[0:1] offset:24 scope:SCOPE_DEV
.Lnocnt:
    s_mov_b32 exec_lo, s16
    v_xor_b32 v12, v12, v13
    v_xor_b32 v14, v14, v15
    v_xor_b32 v12, v12, v14
    global_atomic_add_u32 v7, v12, s[0:1] offset:28 scope:SCOPE_DEV
    s_wait_storecnt 0x0
.Ldone:
    // ---- timer t1 (leader max occ[3]) ----
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lt1
    v_mov_b32 v5, s30
    global_atomic_max_u32 v7, v5, s[0:1] offset:12 scope:SCOPE_DEV
.Lt1:
    s_mov_b32 exec_lo, s16
    // ---- live-- (leader) ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lend
    v_mov_b32 v5, -1
    global_atomic_add_u32 v7, v5, s[0:1] scope:SCOPE_DEV
.Lend:
    s_mov_b32 exec_lo, s16
    s_endpgm
    .size occ_kernel, .-occ_kernel
