// occ_kernel.s  (gfx1201, wave32) -- Phase 3 timed WMMA-THROUGHPUT chain.
//
// Phase 2 measured *occupancy* (resident-wave count). Phase 3 measures *throughput*:
// each wave loads A/B once (compute-isolated, like the 307 TF microbench), then runs
// KDEPTH iterations of NACC independent accumulating WMMAs. Host wall-clock over the
// dispatch -> TFLOPS. The lane-0 atomic counter is kept to LABEL each run with its
// achieved (admission-time) occupancy.
//
// Assemble across the matrix (build.sh):
//   -Wa,-defsym,NACC={8,16}     light (occ-reachable to 16) / heavy (GEMM-representative)
//   -Wa,-defsym,DYNVGPR={0,1}   static reserve / dyn lean-launch + s_alloc
//
// Runtime KDEPTH: read from occ[8] (the harness writes it before each dispatch), so we
// sweep KDEPTH without recompiling.
//
// Encodings (Phase 2 + the two lifted in P3-T1 vs llvm-objdump, NOT guessed):
//   accumulating WMMA : v_wmma_f32_16x16x16_fp8_fp8 v[D:D+7], vA[0:1], vB[0:1], v[D:D+7]  (srcC = acc reg)
//   peel/init WMMA    : v_wmma_f32_16x16x16_fp8_fp8 v[D:D+7], vA[0:1], vB[0:1], 0          (srcC = 0 literal, Phase-2-proven)
//   scalar load       : s_load_b32 s9, s[0:1], 0x8   then   s_wait_kmcnt 0x0
//   returning atom    : global_atomic_add_u32 vDst,vAddr,vData,s[b] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
//   non-ret atom      : global_atomic_<op>_u32 vAddr,vData,s[b] scope:SCOPE_DEV
//   wait model        : s_wait_loadcnt 0x0 (loads/returning-atomic), s_wait_storecnt 0x0 (stores)
//
// User data (USER_SGPR=6): s[0:1]=occ[live@0,maxlive@4,KDEPTH@8]  s[2:3]=fragIn(A@0,B@256)  s[4:5]=fragOut
// v0 = thread id x (lane 0..31) via TIDIG_COMP_CNT (set by the harness in RSRC2).
//
// Register map (NO register exceeds the reservation -- under-reserving = OOB = hang):
//   lean phase : v0=lane, v2/v3=atomic, v4=0                       (<= 32-VGPR launch block)
//   fat phase  : v6=lane*8, v7=lane*32, A=v[8:9], B=v[10:11],
//                acc_k = v[16+8k : 23+8k]   (NACC=8 -> v16..v79 ; NACC=16 -> v16..v143)
.ifndef NACC
    .set NACC, 16
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 0
.endif
.if NACC > 8
    .set FATREGS, 144          // v16..v143 -> 144 VGPRs
.else
    .set FATREGS, 80           // v16..v79  ->  80 VGPRs
.endif

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_mov_b32 v4, 0                      // v4 = 0 : address offset for the occ atomics
    // ---- lane-0-only: admission occupancy counter (labels the run) ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s8
    // ---- KDEPTH (loop count) <- occ[8], runtime ----
    s_load_b32 s9, s[0:1], 0x8
    s_wait_kmcnt 0x0
.if DYNVGPR
    s_alloc_vgpr FATREGS                 // grow lean launch block to the compute footprint
.endif
    // ---- load A/B fragments ONCE (compute isolation) ----
    v_lshlrev_b32 v6, 3, v0              // lane*8 bytes (2 i32)
    global_load_b64 v[8:9],   v6, s[2:3]            // A frag
    global_load_b64 v[10:11], v6, s[2:3] offset:256 // B frag (A block = 32*8 = 256 bytes)
    s_wait_loadcnt 0x0
    // ---- peel iteration 0: srcC = 0 (initializes each accumulator) ----
    v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[8:9], v[10:11], 0
.if NACC > 8
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87],   v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95],   v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[96:103],  v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[128:135], v[8:9], v[10:11], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[136:143], v[8:9], v[10:11], 0
.endif
    // ---- loop the remaining KDEPTH-1 iterations: srcC = acc (accumulate) ----
    s_sub_u32 s9, s9, 1                   // remaining = KDEPTH - 1
    s_cmp_eq_u32 s9, 0
    s_cbranch_scc1 .Lkdone                // KDEPTH==1 (correctness pass) -> skip the loop
.Lkloop:
    v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[8:9], v[10:11], v[16:23]
    v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[8:9], v[10:11], v[24:31]
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[8:9], v[10:11], v[32:39]
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[8:9], v[10:11], v[40:47]
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[8:9], v[10:11], v[48:55]
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[8:9], v[10:11], v[56:63]
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71], v[8:9], v[10:11], v[64:71]
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79], v[8:9], v[10:11], v[72:79]
.if NACC > 8
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87],   v[8:9], v[10:11], v[80:87]
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95],   v[8:9], v[10:11], v[88:95]
    v_wmma_f32_16x16x16_fp8_fp8 v[96:103],  v[8:9], v[10:11], v[96:103]
    v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[8:9], v[10:11], v[104:111]
    v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[8:9], v[10:11], v[112:119]
    v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[8:9], v[10:11], v[120:127]
    v_wmma_f32_16x16x16_fp8_fp8 v[128:135], v[8:9], v[10:11], v[128:135]
    v_wmma_f32_16x16x16_fp8_fp8 v[136:143], v[8:9], v[10:11], v[136:143]
.endif
    s_sub_u32 s9, s9, 1
    s_cmp_lg_u32 s9, 0
    s_cbranch_scc1 .Lkloop
.Lkdone:
    // ---- store acc0 tile (256 f32) for the CPU fp8 oracle ----
    v_lshlrev_b32 v7, 5, v0              // lane*32 bytes
    global_store_b128 v7, v[16:19], s[4:5]
    global_store_b128 v7, v[20:23], s[4:5] offset:16
    s_wait_storecnt 0x0
.if DYNVGPR
    s_alloc_vgpr 32                      // shrink back to the lean block
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
