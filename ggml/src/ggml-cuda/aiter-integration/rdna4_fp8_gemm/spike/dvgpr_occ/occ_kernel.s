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
//   returning atom    : global_atomic_add_u32 vDst,vAddr,vData,s[b] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
//   non-ret atom      : global_atomic_<op>_u32 vAddr,vData,s[b] scope:SCOPE_DEV
//   wait model        : s_wait_loadcnt 0x0 (loads/returning-atomic), s_wait_storecnt 0x0 (stores)
//
// User data (USER_SGPR=7): s[0:1]=occ[live@0,maxlive@4]  s[2:3]=fragIn(A@0,B@256)  s[4:5]=fragOut
//                          s6 = KDEPTH (runtime loop count, passed as a user SGPR -- NOT a
//                          memory load: the scalar K-cache is not invalidated by the dispatch's
//                          AcquireMem, so a memory KDEPTH reads stale across dispatches).
// v0 = thread id x (lane 0..31) via TIDIG_COMP_CNT (set by the harness in RSRC2).
//
// Register map (NO register exceeds the reservation -- under-reserving = OOB = hang):
//   lean phase : v0=lane, v2/v3=atomic, v4=0                       (<= 32-VGPR launch block)
//   fat phase  : v6=lane*8, v7=lane*32, v12=timer, A=v[8:9], B=v[10:11],
//                acc_k = v[32+8k : 39+8k]   (NACC=8 -> v32..v95 ; NACC=16 -> v32..v159)
//   accumulators are ABOVE the 32-VGPR lean block: dyn-VGPR grows above the launch size, so an
//   accumulator straddling v0..v31 corrupts the dyn variant (Phase-2 layout = v32+).
.ifndef NACC
    .set NACC, 16
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 0
.endif
// s_alloc_vgpr block granularity: Phase 2 proved 128 and 32 work; keep FATREGS a multiple of 32
// and >= the highest VGPR used (NACC=16 -> v159 -> 160 ; NACC=8 -> v95 -> 96).
.if NACC > 8
    .set FATREGS, 160          // v32..v159 (160 used) -> 160 VGPRs
.else
    .set FATREGS, 96           // v32..v95  ( 96 used) ->  96 VGPRs
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
    global_atomic_add_u32 v4, v2, s[0:1] offset:16 scope:SCOPE_DEV   // occ[4] += 1 : total waves launched
.Lafter_inc:
    s_mov_b32 exec_lo, s8
    // ---- KDEPTH (loop count) from user SGPR s6 (coherent; not a memory load) ----
    s_mov_b32 s9, s6
.if DYNVGPR
    // Drain the in-flight admission atomics (live/maxlive/total) BEFORE growing the VGPR block.
    // Phase 2's busy-wait incidentally covered this; without it, s_alloc_vgpr races the in-flight
    // VMEM and the grown registers come up wrong -> the A/B load reads 0 -> all-zero WMMA result.
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
    s_alloc_vgpr FATREGS                 // grow lean launch block to the compute footprint
.endif
    // ---- load A/B fragments ONCE (compute isolation) ----
    v_lshlrev_b32 v6, 3, v0              // lane*8 bytes (2 i32)
    global_load_b64 v[16:17], v6, s[2:3]            // A frag (v16:v19 = Phase-2 proven location)
    global_load_b64 v[18:19], v6, s[2:3] offset:256 // B frag (A block = 32*8 = 256 bytes)
    s_wait_loadcnt 0x0
    // ---- peel iteration 0: srcC = 0 (initializes each accumulator) ----
    // Accumulators live at v32+ (ABOVE the 32-VGPR lean launch block). Dyn-VGPR grows the block
    // above the launch size, so accumulators must NOT straddle the lean block (v0..v31), or the
    // dyn variant corrupts -- this matches Phase-2's proven v32+ layout.
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87],   v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95],   v[16:17], v[18:19], 0
.if NACC > 8
    v_wmma_f32_16x16x16_fp8_fp8 v[96:103],  v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[128:135], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[136:143], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[144:151], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[152:159], v[16:17], v[18:19], 0
.endif
    // ---- in-kernel timer (whole-dispatch span) t0: GPU clock (wall_clock64) BEFORE the loop ----
    // Host submit->fence timing does not bracket the kernel on this raw-PM4 path; the GPU clock
    // does. We capture the GLOBAL span: occ[2]=min(start) over all waves, occ[3]=max(end).
    // Throughput = nWG*(KDEPTH-1)*NACC / (span/freq) -- bounded by the matrix-unit ceiling by
    // construction (total grid work / total wall). (s_sendmsg_rtn lifted vs llvm-objdump.)
    s_sendmsg_rtn_b64 s[10:11], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0            // lane 0 only records the span (one atomic per wave)
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v12, s10                    // t0_lo
    global_atomic_min_u32 v4, v12, s[0:1] offset:8 scope:SCOPE_DEV    // occ[2] = min(start)
.Lafter_t0:
    s_mov_b32 exec_lo, s8
    // ---- loop the remaining KDEPTH-1 iterations: srcC = acc (accumulate) ----
    s_sub_u32 s9, s9, 1                   // remaining = KDEPTH - 1
    s_cmp_eq_u32 s9, 0
    s_cbranch_scc1 .Lkdone                // KDEPTH==1 (correctness pass) -> skip the loop
.Lkloop:
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39],   v[16:17], v[18:19], v[32:39]
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47],   v[16:17], v[18:19], v[40:47]
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55],   v[16:17], v[18:19], v[48:55]
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63],   v[16:17], v[18:19], v[56:63]
    v_wmma_f32_16x16x16_fp8_fp8 v[64:71],   v[16:17], v[18:19], v[64:71]
    v_wmma_f32_16x16x16_fp8_fp8 v[72:79],   v[16:17], v[18:19], v[72:79]
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87],   v[16:17], v[18:19], v[80:87]
    v_wmma_f32_16x16x16_fp8_fp8 v[88:95],   v[16:17], v[18:19], v[88:95]
.if NACC > 8
    v_wmma_f32_16x16x16_fp8_fp8 v[96:103],  v[16:17], v[18:19], v[96:103]
    v_wmma_f32_16x16x16_fp8_fp8 v[104:111], v[16:17], v[18:19], v[104:111]
    v_wmma_f32_16x16x16_fp8_fp8 v[112:119], v[16:17], v[18:19], v[112:119]
    v_wmma_f32_16x16x16_fp8_fp8 v[120:127], v[16:17], v[18:19], v[120:127]
    v_wmma_f32_16x16x16_fp8_fp8 v[128:135], v[16:17], v[18:19], v[128:135]
    v_wmma_f32_16x16x16_fp8_fp8 v[136:143], v[16:17], v[18:19], v[136:143]
    v_wmma_f32_16x16x16_fp8_fp8 v[144:151], v[16:17], v[18:19], v[144:151]
    v_wmma_f32_16x16x16_fp8_fp8 v[152:159], v[16:17], v[18:19], v[152:159]
.endif
    s_sub_u32 s9, s9, 1
    s_cmp_lg_u32 s9, 0
    s_cbranch_scc1 .Lkloop
.Lkdone:
    // ---- in-kernel timer (whole-dispatch span) t1: GPU clock AFTER the loop -> occ[3]=max(end) ----
    s_sendmsg_rtn_b64 s[10:11], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v12, s10                   // t1_lo
    global_atomic_max_u32 v4, v12, s[0:1] offset:12 scope:SCOPE_DEV   // occ[3] = max(end)
.Lafter_t1:
    s_mov_b32 exec_lo, s8
    // ---- store acc0 tile (256 f32) for the CPU fp8 oracle ----
    v_lshlrev_b32 v7, 5, v0              // lane*32 bytes
    global_store_b128 v7, v[32:35], s[4:5]
    global_store_b128 v7, v[36:39], s[4:5] offset:16
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
