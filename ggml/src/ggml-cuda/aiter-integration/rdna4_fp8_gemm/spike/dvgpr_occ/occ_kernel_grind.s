// occ_kernel_grind.s  (gfx1201, wave32) -- MAD-305 "GRIND" CONTROL fp8 WMMA GEMM.
//
// PURPOSE (control experiment, 2026-07-03): the smallest NON-split-K, ONE-OUTPUT-TILE-PER-WORKGROUP
//   fp8 GEMM. It deliberately does NOT use the dyn-VGPR / split-K "moat" of occ_kernel_dsws.s. One wave
//   (one workgroup) OWNS one output tile (mblk,tcol), accumulates the FULL K dimension with the fp32
//   accumulators live in VGPR across the whole K loop, and writes C EXACTLY ONCE with a plain
//   global_store (NO C atomics, NO cross-WG reduction, NO ksi split). This is the baseline against which
//   the split-K kernel's 2.1 TF is measured: if grind beats it, split-K may not be worth defending.
//
// STRUCTURE (reused idioms from occ_kernel_dsws.s, machinery stripped):
//   - WMMA microkernel: v_wmma_f32_16x16x16_fp8_fp8  (identical fragment shapes, FM x FN frags).
//   - A/B staging: global_load(_tr)_b64 -> ds_store -> ds_load -> WMMA, streaming K through LDS in
//     KCHUNK-k16-step chunks (single-buffer; one wave so NO barrier -- s_wait_dscnt suffices).
//   - Tile hand-out: a global atomic claim counter (occ[20]) -- a persistent grid-stride over output
//     tiles (raw-PM4 DISPATCH_DIRECT does NOT deliver TGID_X on gfx1201, so a claim counter is the
//     proven work-distribution mechanism; see occ_kernel_wggemm.s deprecation note).
//   STRIPPED vs dsws: the (mblk,tcol,ksi) super-tile pool, the atomic-add C flush, the split-K ksi
//     dimension, the claimer/feed/compute role machinery, the s_alloc_vgpr grow/shrink churn.
//
// ============================================================================================
//  KERNARG CONTRACT (USER_SGPR=15, s0..s14; hardware-preloaded user SGPRs). BYTE-IDENTICAL to the
//  occ_kernel_dsws.s v2 contract EXCEPT s11=TOTAL is the *grind* tile count (finer M tiling: MTL =
//  M/(FM*16), not M/(G*FM*16)) and there is no ksi. The host may reuse the dsws2 userdata packing:
//    s0:s1 = occ buffer base  (>=0x1000B; host zero-inits; occ[20]=claim counter, host sets to `base`)
//    s2:s3 = A     base (fp8 e4m3, row-major MxK, 1 byte/elem)
//    s4:s5 = Bshuf base (shuffled-B layout consumed by global_load_tr_b64; same as dsws)
//    s6:s7 = C     base (fp32, FRAGMENT-TILED: tile ti -> C + ti*(FM*FN*1024); plain store, NO memset
//                         required for correctness, but zeroing is harmless)
//    s8    = KT          (total K16-steps for the whole matrix = K/16; MUST be a multiple of KCHUNK)
//    s9    = K           (bytes per A-row = K, fp8 1 byte/elem)
//    s10   = NT*256      (B-saddr advance per K16-step)
//    s11   = TOTAL       (grind tile count = MTL*NTL, MTL=M/(FM*16); ALSO the claim terminal / chunkHi)
//    s12   = magic(ceil(2^32/NTL))   (unsigned-div magic for /NTL -> mblk/tcol decode)
//    s13   = NTL         (number of N tile-columns = N/(FN*16))
//    s14   = FN*256      (B-saddr stride per N-frag)
//    (TGID_X lands in s15 -- UNUSED.)
//  occ layout (occ base = s0:s1; host zero-inits, sets occ[20]=base):
//    occ[0]  (byte 0)  = live counter (lane0 +1 at entry, -1 at exit; host polls ==0)
//    occ[1]  (byte 4)  = maxlive (bookkeeping)
//    occ[2]  (byte 8)  = min start realtime clock ; occ[3] (byte 12) = max end realtime clock
//    occ[20] (byte 20) = GLOBAL tile claim counter (host sets to `base`; grind atomic-adds 1/tile)
//
//  G and SEGK/split-K are GONE. KCHUNK (k16-steps staged per LDS chunk) is a COMPILE-TIME defsym.
// ============================================================================================

.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"

// ---- tile defsyms ----
.ifndef FM
    .set FM, 2                              // per-wave M-frags (M tile = FM*16 rows)
.endif
.ifndef FN
    .set FN, 4                              // per-wave N-frags (N tile = FN*16 cols)
.endif
.ifndef KCHUNK
    .set KCHUNK, 4                          // k16-steps staged through LDS per chunk (power of two)
.endif
.ifndef RGADESC
    .set RGADESC, 0                         // 1 = emit analysis-only AMDHSA descriptor for RGA livereg
.endif

// log2(KCHUNK) so n_chunks = KT >> KCH_SHIFT (KCHUNK is a power of two by construction).
.if KCHUNK == 1
  .set KCH_SHIFT, 0
.elseif KCHUNK == 2
  .set KCH_SHIFT, 1
.elseif KCHUNK == 4
  .set KCH_SHIFT, 2
.elseif KCHUNK == 8
  .set KCH_SHIFT, 3
.else
  .error "KCHUNK must be a power of two in {1,2,4,8}"
.endif

// ---- LDS layout (single-buffer; A then B; frag = 256B, lane*8 base) ----
.set ALDS_OFF,   0                                  // A frag (ks,mi) at ALDS_OFF + (ks*FM + mi)*256
.set ALDS_BYTES, (KCHUNK*FM*256)
.set BLDS_OFF,   ALDS_BYTES                         // B frag (ks,ni) at BLDS_OFF + (ks*FN + ni)*256
.set BLDS_BYTES, (KCHUNK*FN*256)
.set LDS_TOTAL,  (BLDS_OFF + BLDS_BYTES)            // = KCHUNK*(FM+FN)*256 (=6144 @ KCHUNK=4,FM=2,FN=4)
.if LDS_TOTAL > 32768
  .error "grind LDS layout exceeds 32768B group segment"
.endif

// ---- VGPR layout (STATIC alloc; full footprint live for the whole K loop -- this is the control's
//   high-VGPR-duty structure by design). Accumulators live from pre-K-loop zero through the single C store.
.set STG, 16                                 // staging pair (v16:v17) for one in-flight A/B frag
.set ACC, 32                                 // FM*FN fp32 accumulators x 8 (v32..)
.set FA,  (ACC + 8*FM*FN)                     // compute A frags (from LDS): FM x 2   (=v96 @ 2x4)
.set FB,  (FA + 2*FM)                         // compute B frags (from LDS): FN x 2   (=v100 @ 2x4)
.set NFV, (FB + 2*FN)                         // next-free vgpr (=v108 @ 2x4 -> ~112 alloc, under 128 cap)

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- per-thread identity (v0=tid hardware-preloaded); one wave -> wid=0, lane=tid ----
    v_and_b32     v2, 31, v0                  // lane = tid & 31
    v_and_b32     v6, 15, v0                  // lane & 15 (A row within frag)
    v_mov_b32     v4, 0                        // vaddr-0 for atomic claim (addr = occ_base + offset)
    // ---- per-lane address constants (dsws-identical fragment maps) ----
    v_mul_lo_u32  v8, v6, s9                   // (lane&15)*K
    v_bfe_u32     v7, v0, 4, 1                 // colhi = (tid>>4)&1
    v_lshlrev_b32 v7, 3, v7                    // colhi*8
    v_add_nc_u32  v8, v8, v7                   // v8 = A vaddr = (lane&15)*K + colhi*8
    v_lshlrev_b32 v9, 3, v2                    // v9 = B/LDS vaddr = lane*8
    v_lshlrev_b32 v10, 5, v2                   // v10 = C store vaddr = lane*32

    // ---- live++ : lane0 occ[0] += 1 ; maxlive book ----
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_live
    v_mov_b32 v5, 1
    global_atomic_add_u32 v11, v4, v5, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV   // v11 = old live
    s_wait_loadcnt 0x0
    v_add_nc_u32 v11, v11, 1
    global_atomic_max_u32 v4, v11, s[0:1] offset:4 scope:SCOPE_DEV                  // maxlive
.Lafter_live:
    s_mov_b32 exec_lo, s16

    // ---- timer t0 (min start) ----
    s_sendmsg_rtn_b64 s[36:37], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t0
    v_mov_b32 v5, s36
    global_atomic_min_u32 v4, v5, s[0:1] offset:8 scope:SCOPE_DEV
.Lafter_t0:
    s_mov_b32 exec_lo, s16

    // n_chunks = KT >> KCH_SHIFT  (KT = s8)
    s_lshr_b32 s33, s8, KCH_SHIFT
    // A mi-stride = 16*K bytes
    s_lshl_b32 s30, s9, 4

// ============================================================================================
//  CLAIM LOOP: ti = atomicAdd(occ[20], 1) ; if ti >= TOTAL(s11) done ; compute tile ti full-K ; store C.
// ============================================================================================
.Lclaim:
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lclaim_bcast
    v_mov_b32 v5, 1
    global_atomic_add_u32 v11, v4, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV  // v11=old ti
    s_wait_loadcnt 0x0
.Lclaim_bcast:
    s_mov_b32 exec_lo, s16
    v_readfirstlane_b32 s20, v11               // ti (uniform)
    s_cmp_ge_u32 s20, s11                       // ti >= TOTAL -> drained
    s_cbranch_scc1 .Ldone

    // ---- decode: mblk = ti/NTL (magic) ; tcol = ti - mblk*NTL ----
    s_mul_hi_u32 s21, s20, s12                  // mblk
    s_mul_i32    s22, s21, s13                  // mblk*NTL
    s_sub_u32    s22, s20, s22                  // tcol = ti - mblk*NTL

    // ---- tile scalar bases (k=0) ----
    // A: A + (mblk*FM*16)*K
    s_mul_i32    s38, s21, (FM*16)              // row_start = mblk*FM*16
    s_mul_i32    s34, s38, s9                    // low(row_start*K)
    s_mul_hi_u32 s35, s38, s9                    // high
    s_add_u32    s34, s2, s34
    s_addc_u32   s35, s3, s35                    // s[34:35] = A tile base (k=0), advances per chunk
    // B: Bshuf + tcol*(FN*256)
    s_mul_i32    s36, s22, s14                   // low(tcol*FN*256)
    s_mul_hi_u32 s37, s22, s14                   // high
    s_add_u32    s36, s4, s36
    s_addc_u32   s37, s5, s37                    // s[36:37] = B tile base (k=0), advances per chunk
    // C: C + ti*(FM*FN*1024)
    s_mul_i32    s28, s20, (FM*FN*1024)          // low
    s_mul_hi_u32 s29, s20, (FM*FN*1024)          // high
    s_add_u32    s28, s6, s28
    s_addc_u32   s29, s7, s29                    // s[28:29] = C tile base

    // ---- zero the FM*FN fp32 accumulators (live across the whole K loop) ----
    .set idx, 0
    .rept FM*FN
      v_mov_b32 v[ACC+idx*8+0], 0
      v_mov_b32 v[ACC+idx*8+1], 0
      v_mov_b32 v[ACC+idx*8+2], 0
      v_mov_b32 v[ACC+idx*8+3], 0
      v_mov_b32 v[ACC+idx*8+4], 0
      v_mov_b32 v[ACC+idx*8+5], 0
      v_mov_b32 v[ACC+idx*8+6], 0
      v_mov_b32 v[ACC+idx*8+7], 0
      .set idx, idx+1
    .endr

    s_mov_b32 s31, 0                             // chunk = 0
// ---- K-CHUNK LOOP: stage KCHUNK k16-steps of A/B into LDS, then WMMA-accumulate ----
.Lkloop:
    s_cmp_ge_u32 s31, s33
    s_cbranch_scc1 .Lkdone

    // ===== STAGE A into LDS: A frag (ks,mi) ; global_load_b64(v8, base + ks*16) -> ds_store =====
    .set mi, 0
    .rept FM
      // sAmi = A_chunk_base + mi*(16*K)
      .if mi == 0
        s_mov_b32 s44, s34
        s_mov_b32 s45, s35
      .else
        s_add_u32  s44, s44, s30
        s_addc_u32 s45, s45, 0
      .endif
      .set ks, 0
      .rept KCHUNK
        global_load_b64 v[STG:STG+1], v8, s[44:45] offset:(ks*16)
        s_wait_loadcnt 0x0
        ds_store_b64 v9, v[STG:STG+1] offset:(ALDS_OFF + (ks*FM + mi)*256)
        s_wait_dscnt 0x0
        .set ks, ks+1
      .endr
      .set mi, mi+1
    .endr

    // ===== STAGE B into LDS: B frag (ks,ni) ; global_load_tr_b64(v9, base + ks*s10 + ni*256) -> ds_store
    s_mov_b32 s42, s36                            // sBc = B_chunk_base (ks=0)
    s_mov_b32 s43, s37
    .set ks, 0
    .rept KCHUNK
      s_mov_b32 s40, s42                          // sf = sBc
      s_mov_b32 s41, s43
      .set ni, 0
      .rept FN
        global_load_tr_b64 v[STG:STG+1], v9, s[40:41]
        s_wait_loadcnt 0x0
        ds_store_b64 v9, v[STG:STG+1] offset:(BLDS_OFF + (ks*FN + ni)*256)
        s_wait_dscnt 0x0
        s_add_u32  s40, s40, 256                  // += frag stride (FN dir)
        s_addc_u32 s41, s41, 0
        .set ni, ni+1
      .endr
      s_add_u32  s42, s42, s10                    // += k16-step stride (NT*256)
      s_addc_u32 s43, s43, 0
      .set ks, ks+1
    .endr

    // ===== COMPUTE: read resident A/B from LDS, WMMA-accumulate into live ACC =====
    .set ks, 0
    .rept KCHUNK
      .set ni, 0
      .rept FN
        ds_load_b64 v[FB+ni*2:FB+ni*2+1], v9 offset:(BLDS_OFF + (ks*FN + ni)*256)
        .set ni, ni+1
      .endr
      .set mi, 0
      .rept FM
        ds_load_b64 v[FA+mi*2:FA+mi*2+1], v9 offset:(ALDS_OFF + (ks*FM + mi)*256)
        .set mi, mi+1
      .endr
      s_wait_dscnt 0x0
      .set mi, 0
      .rept FM
        .set ni, 0
        .rept FN
          v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+mi*2:FA+mi*2+1], v[FB+ni*2:FB+ni*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
          .set ni, ni+1
        .endr
        .set mi, mi+1
      .endr
      .set ks, ks+1
    .endr

    // advance chunk bases: A += KCHUNK*16 bytes ; B += KCHUNK*(NT*256)
    s_add_u32  s34, s34, (KCHUNK*16)
    s_addc_u32 s35, s35, 0
    s_mul_i32  s46, s10, KCHUNK
    s_add_u32  s36, s36, s46
    s_addc_u32 s37, s37, 0
    s_add_u32  s31, s31, 1                        // chunk++
    s_branch .Lkloop
.Lkdone:

    // ===== C STORE (once, plain non-atomic): C_tile + frag*1024 + {0,16} ; vaddr v10=lane*32 =====
    .set frag, 0
    .rept FM*FN
      global_store_b128 v10, v[ACC+frag*8+0:ACC+frag*8+3], s[28:29] offset:(frag*1024 + 0)
      global_store_b128 v10, v[ACC+frag*8+4:ACC+frag*8+7], s[28:29] offset:(frag*1024 + 16)
      .set frag, frag+1
    .endr
    s_wait_storecnt 0x0                            // stores must be issued before we reuse ACC next claim
    s_branch .Lclaim

.Ldone:
    s_wait_storecnt 0x0
    // ---- timer t1 (max end) ----
    s_sendmsg_rtn_b64 s[36:37], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_t1
    v_mov_b32 v5, s36
    global_atomic_max_u32 v4, v5, s[0:1] offset:12 scope:SCOPE_DEV
.Lafter_t1:
    s_mov_b32 exec_lo, s16
    // ---- live-- : lane0 occ[0] -= 1 ----
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_dec
    v_mov_b32 v5, -1
    global_atomic_add_u32 v4, v5, s[0:1] scope:SCOPE_DEV
.Lafter_dec:
    s_mov_b32 exec_lo, s16
    s_endpgm
    .size occ_kernel, .-occ_kernel

// ---- RGADESC: analysis-only AMDHSA descriptor so `rga -s bin --co` can enumerate + livereg. NOT
//   emitted for the PM4 .bin (the host provides RSRC1/RSRC2 directly). ----
.if RGADESC
.amdhsa_kernel occ_kernel
    .amdhsa_next_free_vgpr NFV
    .amdhsa_next_free_sgpr 50
    .amdhsa_group_segment_fixed_size LDS_TOTAL
    .amdhsa_user_sgpr_count 15
    .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            occ_kernel
    .symbol:          occ_kernel.kd
    .kernarg_segment_size: 60
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 6144
    .private_segment_fixed_size: 0
    .wavefront_size:  32
    .sgpr_count:      50
    .vgpr_count:      112
    .max_flat_workgroup_size: 32
    .args:            []
.end_amdgpu_metadata
.endif
