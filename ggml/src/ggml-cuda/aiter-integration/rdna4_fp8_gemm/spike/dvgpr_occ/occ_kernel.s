// occ_kernel.s  (gfx1201, wave32). Assemble twice:
//   dyn:    clang ... -defsym DYNVGPR=1   (harness launches 32-VGPR block + RSRC2 bit6)
//   static: clang ... -defsym DYNVGPR=0   (harness launches 128-VGPR static block)
//
// All hang-risky encodings lifted verbatim from compiler seeds (Task 2), not guessed:
//   wmma          : v_wmma_f32_16x16x16_fp8_fp8 vDst[0:7], vA[0:1], vB[0:1], 0   (srcC=0 literal)
//   returning atom: global_atomic_add_u32 vDst,vAddr,vData,s[b] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
//   non-ret atom  : global_atomic_<op>_u32 vAddr,vData,s[b] scope:SCOPE_DEV
//   wait model    : s_wait_loadcnt 0x0 (loads / returning-atomic result), s_wait_storecnt 0x0 (stores)
//   WMMA->store    : no explicit hazard needed (HW stalls VMEM on the WMMA accumulator)
//
// User data (USER_SGPR=6): s[0:1]=occ[live@0,maxlive@4]  s[2:3]=fragIn(A@0,B@256)  s[4:5]=fragOut
// v0 = thread id x (lane 0..31) via TIDIG_COMP_CNT (set by the harness in RSRC2).
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_mov_b32 v4, 0                      // v4 = 0 : address offset for the occ atomics (all lanes)
    // ---- lane-0-only: bump live, sample maxlive ----
    v_cmp_eq_u32 vcc_lo, 0, v0           // lane 0?
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo   // exec = {lane0}
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV   // v3 = old live
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1                                                         // new live
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV                  // maxlive = max(.,new)
.Lafter_inc:
    s_mov_b32 exec_lo, s8                // restore full wave
    // ---- long busy-wait at the SMALL block (where occupancy is measured) ----
    s_movk_i32 s9, 0x4000
.Lspin:
    s_sub_u32 s9, s9, 1
    s_cmp_lg_u32 s9, 0
    s_cbranch_scc1 .Lspin
.if DYNVGPR
    s_alloc_vgpr 128                     // grow to 128 VGPRs for the WMMA
.endif
    // ---- per-lane fragment loads: A=fragIn[lane*8], B=fragIn[256 + lane*8] ----
    v_lshlrev_b32 v6, 3, v0             // lane*8 bytes (2 i32)
    global_load_b64 v[16:17], v6, s[2:3]            // A frag (2 i32)
    global_load_b64 v[18:19], v6, s[2:3] offset:256 // B frag (A block is 32*8 = 256 bytes)
    s_wait_loadcnt 0x0
    // ---- 4 WMMA accumulators in v[32:63] (srcC=0), all reuse A=v[16:17] B=v[18:19] ----
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[16:17], v[18:19], 0
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[16:17], v[18:19], 0
    // ---- store the 4 tiles: fragOut + tile*1024 + lane*32 (256 f32 per tile) ----
    v_lshlrev_b32 v7, 5, v0           // lane*32 bytes
    global_store_b128 v7, v[32:35], s[4:5]                 // tile0 lo
    global_store_b128 v7, v[36:39], s[4:5] offset:16       // tile0 hi
    global_store_b128 v7, v[40:43], s[4:5] offset:1024     // tile1
    global_store_b128 v7, v[44:47], s[4:5] offset:1040
    global_store_b128 v7, v[48:51], s[4:5] offset:2048     // tile2
    global_store_b128 v7, v[52:55], s[4:5] offset:2064
    global_store_b128 v7, v[56:59], s[4:5] offset:3072     // tile3
    global_store_b128 v7, v[60:63], s[4:5] offset:3088
    s_wait_storecnt 0x0               // stores committed before freeing v[32:63]
.if DYNVGPR
    s_alloc_vgpr 32                   // shrink back to the small block
.endif
    // ---- lane-0-only: dec live ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v2, -1
    global_atomic_add_u32 v4, v2, s[0:1] scope:SCOPE_DEV   // non-returning dec
.Ldone:
    s_mov_b32 exec_lo, s8
    s_endpgm
    .size occ_kernel, .-occ_kernel
