// occ_kernel_combined.s  (gfx1201, wave32) -- ALL LEVERS STACKED: unroll x ILP x feed x dyn-VGPR.
//
// The "combine everything" kernel for MAD-305 #287. Builds on occ_kernel.s (the proven Phase-2/3
// dyn-VGPR vehicle) and adds the one lever the data said was missing: UNROLL (many WMMA rounds per
// branch, the way the 307 TF microbench reaches peak at modest NACC). Stacks:
//   - UNROLL : U accumulate-rounds per loop branch (kills the per-iter branch/scalar overhead)
//   - NACC   : independent accumulator chains (ILP to hide WMMA latency)
//   - FEED   : re-fetch B each round from global (the wide-feed lever's operand-gap; same addr ->
//              L2 hit, B unchanged, acc still = K*(A.B) so the oracle/loop check still pass)
//   - DYNVGPR: lean 32-VGPR launch -> s_alloc to the fat block -> full occupancy
//
// Loop semantics (underflow-safe): s6 = TRIP COUNT (loop iterations). Each iteration does UNROLL
// accumulate-rounds. Peel does 1 (init). So total accumulations = 1 + TRIP*UNROLL = effective K;
// the host computes that for the work/oracle. Decrement-by-1 of an exact trip count -> no wrap.
//
// Assemble (build.sh): -Wa,-defsym,NACC={8,16} DYNVGPR={0,1} FEED={0,1} UNROLL={1,4,8,16}
//
// Register map identical to occ_kernel.s: lean block v0..v31 (A=v16:17, B=v18:19, v6=lane*8,
// v12=timer); accumulators v[32+8k : 39+8k] ABOVE the lean block (dyn grows above 32).
.ifndef NACC
    .set NACC, 16
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 0
.endif
.ifndef FEED
    .set FEED, 0
.endif
.ifndef UNROLL
    .set UNROLL, 8
.endif
.if NACC >= 16
    .set FATREGS, 160
.elseif NACC >= 12
    .set FATREGS, 128
.else
    .set FATREGS, 96
.endif

// ---- one accumulate-round: NACC independent accumulating WMMAs (srcC = acc) ----
.macro acc_round
.if FEED
    global_load_b64 v[18:19], v6, s[2:3] offset:256   // FEED: re-fetch B (operand gap; same addr)
    s_wait_loadcnt 0x0
.endif
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39],   v[16:17], v[18:19], v[32:39]
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47],   v[16:17], v[18:19], v[40:47]
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55],   v[16:17], v[18:19], v[48:55]
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63],   v[16:17], v[18:19], v[56:63]
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71],   v[16:17], v[18:19], v[64:71]
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79],   v[16:17], v[18:19], v[72:79]
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87],   v[16:17], v[18:19], v[80:87]
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95],   v[16:17], v[18:19], v[88:95]
.if NACC >= 12
    v_wmma_f32_16x16x16_fp8_fp8 v[96:103],  v[16:17], v[18:19], v[96:103]
    v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[16:17], v[18:19], v[104:111]
    v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[16:17], v[18:19], v[112:119]
    v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[16:17], v[18:19], v[120:127]
.endif
.if NACC >= 16
    v_wmma_f32_16x16x16_fp8_fp8 v[128:135], v[16:17], v[18:19], v[128:135]
    v_wmma_f32_16x16x16_fp8_fp8 v[136:143], v[16:17], v[18:19], v[136:143]
    v_wmma_f32_16x16x16_fp8_fp8 v[144:151], v[16:17], v[18:19], v[144:151]
    v_wmma_f32_16x16x16_fp8_fp8 v[152:159], v[16:17], v[18:19], v[152:159]
.endif
.endm

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_mov_b32 v4, 0
    // ---- lane-0-only admission occupancy counter ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV
    global_atomic_add_u32 v4, v2, s[0:1] offset:16 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s8
    s_mov_b32 s9, s6                     // s9 = TRIP COUNT (loop iterations), from user SGPR s6
.if DYNVGPR
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
    s_alloc_vgpr FATREGS
.endif
    // ---- load A/B once ----
    v_lshlrev_b32 v6, 3, v0
    global_load_b64 v[16:17], v6, s[2:3]
    global_load_b64 v[18:19], v6, s[2:3] offset:256
    s_wait_loadcnt 0x0
    // ---- peel: init each accumulator (srcC = 0) ----
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95],   v[16:17], v[18:19], 0
.if NACC >= 12
    v_wmma_f32_16x16x16_fp8_fp8 v[96:103],  v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[16:17], v[18:19], 0
.endif
.if NACC >= 16
    v_wmma_f32_16x16x16_fp8_fp8 v[128:135], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[136:143], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[144:151], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[152:159], v[16:17], v[18:19], 0
.endif
    // ---- timer t0 (GPU clock) before the loop ----
    s_sendmsg_rtn_b64 s[10:11], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v12, s10
    global_atomic_min_u32 v4, v12, s[0:1] offset:8 scope:SCOPE_DEV
.Lafter_t0:
    s_mov_b32 exec_lo, s8
    // ---- main loop: TRIP iterations, each UNROLL accumulate-rounds ----
    s_cmp_eq_u32 s9, 0
    s_cbranch_scc1 .Lkdone                // trip==0 (correctness pass) -> peel only
.Lkloop:
    .rept UNROLL
        acc_round
    .endr
    s_sub_u32 s9, s9, 1                   // exact trip count -> no unsigned wrap
    s_cmp_lg_u32 s9, 0
    s_cbranch_scc1 .Lkloop
.Lkdone:
    // ---- timer t1 ----
    s_sendmsg_rtn_b64 s[10:11], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v12, s10
    global_atomic_max_u32 v4, v12, s[0:1] offset:12 scope:SCOPE_DEV
.Lafter_t1:
    s_mov_b32 exec_lo, s8
    // ---- store acc0 tile for the oracle ----
    v_lshlrev_b32 v7, 5, v0
    global_store_b128 v7, v[32:35], s[4:5]
    global_store_b128 v7, v[36:39], s[4:5] offset:16
    s_wait_storecnt 0x0
.if DYNVGPR
    s_alloc_vgpr 32
.endif
    // ---- lane-0-only: dec live ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v2, -1
    global_atomic_add_u32 v4, v2, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s8
    s_endpgm
    .size occ_kernel, .-occ_kernel
