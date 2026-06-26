// occ_kernel_wavespec.s  (gfx1201, wave32) -- MAD-305 #323 LEAN WAVE-SPECIALIZED fp8 WMMA GEMM.
// Persistent atomic-claim. Per WG: NLOAD lean loader waves feed BOTH the A-frags and the shared
// B-panel into an LDS slot; NCOMP compute waves ds_load + WMMA ONLY (zero global in the compute
// stream). Clean-room from-scratch (adapts proven blocks: wggemm2 claim/broadcast, ANOLDSTR lean
// register packing + global_load_tr feed, BLDS round-tripped LDS frags). STATIC alloc here
// (DYNVGPR added in T4). Correctness-first: single LDS slot, 2 barriers/slice (double-buffer = later).
//
// Geometry: one claimed tile = (NCOMP*FM) M-frags x FN N-frags, contracted over K in 16-wide slices.
//   compute wave cid (= wid-NLOAD) owns M-band [cid*FM : cid*FM+FM] x all FN -> FM*FN frags.
//   B-panel (FN frags) is SHARED across all NCOMP compute waves (the reuse operand).
//
// Feed layout (matches the harness preshuffles, reused by the oracle):
//   A: Ashuf (mbg_preshuffle_A) [kt][mt]256, mt in 0..MT-1. loaders global_load_tr_b64 -> A-frag.
//   B: Bshuf (mbg_preshuffle_B) [kt][nt]256, nt in 0..NT-1. loaders global_load_tr_b64 -> B-frag.
//   loaders ds_store frags into the LDS slot; compute ds_load_b64 the byte-identical frag (no transpose).
//
// User data (USER_SGPR=15):
//   s0:s1=occ  s2:s3=Ashuf  s4:s5=Bshuf  s6:s7=C  s8=K  s9=NT*256 (B kt stride)  s10=NTL_MASK
//   s11=NTL_LOG2  s12=NTILES(K/16)  s13=TOTAL  s14=MT*256 (A kt stride)
// LDS slot: [0 .. ASLICE)=A frags (NCOMP*FM*256), [ASLICE .. SLOT)=B frags (FN*256), [SLOT]=ti bcast.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"

.ifndef FM
    .set FM, 2
.endif
.ifndef FN
    .set FN, 2
.endif
.ifndef NLOAD
    .set NLOAD, 1                          // loader-only waves (feed the LDS slot)
.endif
.ifndef NCOMP
    .set NCOMP, 4                          // compute waves (ds_load + WMMA)
.endif
.ifndef STORE
    .set STORE, 1                          // 1 = full FM*FN-frag diagnostic store (oracle). 0 = perf (no store).
.endif
.ifndef DYNVGPR
    .set DYNVGPR, 0                        // T4: 1 = loaders s_alloc_vgpr LEANREG, compute grow. 0 = static.
.endif
.ifndef LEANREG
    .set LEANREG, 32                       // T4 loader lean target (inert at DYNVGPR=0)
.endif
.ifndef RGADESC
    .set RGADESC, 0                        // 1 = emit analysis-only AMDHSA descriptor + metadata for RGA
.endif
.ifndef BUSYWAIT
    .set BUSYWAIT, 0                       // T6 fix: 1 = replace the 4 asymmetric K-slice s_barrier with an
.endif                                      //   LDS sense-reversing busy-wait (s_barrier under dyn-VGPR deadlocks
                                            //   when waves rendezvous at different allocations -- BRICK #4 root cause).
                                            //   Claim barrier (pre-grow, symmetric) stays hardware for clean init.

.set WAVES,  (NLOAD + NCOMP)              // total waves launched per WG (harness must match)

// ---- LDS slot layout (bytes) ----
.set ASLICE, (NCOMP*FM*256)              // A frags region: NCOMP*FM frags x 256B
.set BSLICE, (FN*256)                    // B frags region: FN frags x 256B
.set SLOT,   (ASLICE + BSLICE)
.set TI_OFF, SLOT                         // ti broadcast just past the slot
.set BAR_CNT_OFF,   (SLOT + 4)            // busy-wait barrier arrival counter (4B)
.set BAR_SENSE_OFF, (SLOT + 8)            // busy-wait barrier sense flag (4B)
.set LDS_TOTAL, (SLOT + 12)              // ti + bar_cnt + bar_sense (harness allocates the same)

// ---- VGPR layout ----
.set ACC, 32                             // accumulators: FM*FN frags x 8 f32
.set FA,  (ACC + 8*FM*FN)                // compute A frags: FM x 2 regs
.set FB,  (FA + 2*FM)                    // compute B frags: FN x 2 regs
.set LD,  32                             // loader data regs (loaders never touch ACC): NCOMP*FM*2 + FN*2
.set NFV, (FB + 2*FN + 16)               // next_free_vgpr (compute dominates); headroom 16

// ---- BUSYWAIT: sense-reversing LDS workgroup barrier (faithful all-waves rendezvous, self-resetting).
//   Replaces s_barrier -1, which deadlocks under dyn-VGPR when waves arrive at different allocations.
//   ALL temps are < v32 (v28..v30) so the macro is valid in the LEAN loaders too. s40 = persistent
//   per-wave sense (init 0 at entry). s41/s42 = transient. Lane 0 owns the atomic; all lanes spin.
.macro lds_barrier
    s_xor_b32 s40, s40, 1                       // flip my sense for this rendezvous
    s_mov_b32 s41, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2                  // lane 0 of the wave only
    s_and_b32 exec_lo, exec_lo, vcc_lo
    v_mov_b32 v28, BAR_CNT_OFF
    v_mov_b32 v29, 1
    ds_add_rtn_u32 v28, v28, v29                // v28 = old arrival count (atomic, WG-wide)
    s_wait_dscnt 0x0
    v_add_nc_u32 v28, v28, 1                    // arrived = old + 1
    v_cmp_eq_u32 vcc_lo, WAVES, v28             // am I the last to arrive?
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lbar_rel\@
    v_mov_b32 v29, 0
    v_mov_b32 v30, BAR_CNT_OFF
    ds_store_b32 v30, v29                       // reset count = 0 (for next use)
    v_mov_b32 v30, BAR_SENSE_OFF
    v_mov_b32 v29, s40
    ds_store_b32 v30, v29                       // publish sense -> releases the spinning waves
    s_wait_dscnt 0x0
.Lbar_rel\@:
    s_mov_b32 exec_lo, s41                      // restore full exec
.Lbar_wait\@:
    v_mov_b32 v28, BAR_SENSE_OFF
    ds_load_b32 v29, v28
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s42, v29
    s_cmp_eq_u32 s42, s40
    s_cbranch_scc0 .Lbar_wait\@                 // spin until BAR_SENSE == my sense
.endm

.text
.globl occ_kernel
.p2align 8
.type occ_kernel,@function
occ_kernel:
    // ---- identity ----
    v_lshrrev_b32 v1, 5, v0                 // wid = tid >> 5
    v_and_b32     v2, 31, v0                // lane = tid & 31
    v_mov_b32     v7, 0
    v_lshlrev_b32 v9,  3, v2                // v9  = lane*8  (global_load_tr vaddr AND ds vaddr)
    v_lshlrev_b32 v10, 5, v2                // v10 = lane*32 (C store vaddr)
    v_mov_b32     v24, TI_OFF

    // ---- admission bookkeeping: leader (tid==0) live++, maxlive, total++ ----
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

    // ---- timer t0 (leader writes min start tick to occ[2]) ----
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

.if BUSYWAIT
    // ---- busy-wait barrier init: every wave sets local sense=0; leader zeroes the LDS counter+sense.
    //   The hardware claim barrier (first .Lclaim_loop iteration) publishes this init to all waves. ----
    s_mov_b32 s40, 0
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lbar_init_done
    v_mov_b32 v29, 0
    v_mov_b32 v28, BAR_CNT_OFF
    ds_store_b32 v28, v29
    v_mov_b32 v28, BAR_SENSE_OFF
    ds_store_b32 v28, v29
    s_wait_dscnt 0x0
.Lbar_init_done:
    s_mov_b32 exec_lo, s16
.endif

    // ============ PERSISTENT CLAIM LOOP (one atomic/tile; leader broadcasts ti via LDS+barrier) ====
.Lclaim_loop:
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_claim
    v_mov_b32 v5, 1
    global_atomic_add_u32 v8, v7, v5, s[0:1] offset:20 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    ds_store_b32 v24, v8                     // LDS[TI_OFF] = ti  (cross-wave broadcast)
    s_wait_dscnt 0x0
.Lafter_claim:
    s_mov_b32 exec_lo, s16
    s_barrier_signal -1
    s_barrier_wait -1
    ds_load_b32 v8, v24                      // all waves read ti
    s_wait_dscnt 0x0
    v_readfirstlane_b32 s17, v8              // ti (scalar)
    s_cmp_ge_u32 s17, s13
    s_cbranch_scc1 .Lexit

    // ---- decode tile_row/col (A-stationary, N fastest): tile_row=ti>>log2NTL, tile_col=ti&NTL_MASK ----
    s_lshr_b32 s18, s17, s11                 // tile_row
    s_and_b32  s19, s17, s10                 // tile_col
    // A base = Ashuf + (tile_row*(NCOMP*FM))*256 ; B base = Bshuf + (tile_col*FN)*256
    s_mul_i32  s20, s18, (NCOMP*FM*256)
    s_add_u32  s20, s2, s20
    s_addc_u32 s21, s3, 0                     // s[20:21] = A base (this tile)
    s_mul_i32  s22, s19, (FN*256)
    s_add_u32  s22, s4, s22
    s_addc_u32 s23, s5, 0                     // s[22:23] = B base (this tile)

    // ============ ROLE BRANCH ============
    v_cmp_lt_u32 vcc_lo, v1, NLOAD           // wid < NLOAD -> loader
    s_mov_b32 s25, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcompute_role           // no loader lanes here -> compute
    // (loaders fall through; compute waves are masked off below by re-deriving exec)
    s_mov_b32 exec_lo, s25                    // restore full exec for the loader body (only loader waves reach here)

    // -------- LOADER BODY: fill A frags + B frags into the single LDS slot, per K-slice --------
.if DYNVGPR
    s_alloc_vgpr LEANREG                       // T4: loaders go lean
.endif
    s_mov_b32 s26, 0                          // kt = 0
    s_mov_b32 s36, s20
    s_mov_b32 s37, s21                        // running A base
    s_mov_b32 s38, s22
    s_mov_b32 s39, s23                        // running B base
.Lload_loop:
    // ---- fill the LDS slot: A frags (region [0..ASLICE)), then B frags (region [ASLICE..SLOT)) ----
.if DYNVGPR
    // LEAN (dyn): stream EACH frag through a single 2-reg window (v26:27) so the loader's live VGPR
    //   stays <=32 -> consistent with s_alloc_vgpr LEANREG. (The batched path below writes v[LD=32..],
    //   which would alias the forfeited registers after the shrink -> the WS_DYN=1 hang. This is the fix.)
    //   Serial load->store; correctness/lean first (windowed prefetch is a later perf knob).
    .set f, 0
    .rept (NCOMP*FM)
      global_load_tr_b64 v[26:27], v9, s[36:37] offset:f*256
      s_wait_loadcnt 0x0
      ds_store_b64 v9, v[26:27] offset:f*256
      s_wait_dscnt 0x0
      .set f, f+1
    .endr
    .set g, 0
    .rept FN
      global_load_tr_b64 v[26:27], v9, s[38:39] offset:g*256
      s_wait_loadcnt 0x0
      ds_store_b64 v9, v[26:27] offset:ASLICE+g*256
      s_wait_dscnt 0x0
      .set g, g+1
    .endr
.else
    // STATIC (oracle-proven 2x2): batch all loads into v[LD..] then all stores -> overlapped feed.
    .set f, 0
    .rept (NCOMP*FM)
      global_load_tr_b64 v[LD+f*2:LD+f*2+1], v9, s[36:37] offset:f*256
      .set f, f+1
    .endr
    .set g, 0
    .rept FN
      global_load_tr_b64 v[LD+(NCOMP*FM+g)*2:LD+(NCOMP*FM+g)*2+1], v9, s[38:39] offset:g*256
      .set g, g+1
    .endr
    s_wait_loadcnt 0x0
    .set f, 0
    .rept (NCOMP*FM)
      ds_store_b64 v9, v[LD+f*2:LD+f*2+1] offset:f*256
      .set f, f+1
    .endr
    .set g, 0
    .rept FN
      ds_store_b64 v9, v[LD+(NCOMP*FM+g)*2:LD+(NCOMP*FM+g)*2+1] offset:ASLICE+g*256
      .set g, g+1
    .endr
    s_wait_dscnt 0x0
.endif
.if BUSYWAIT
    lds_barrier                               // publish: compute may now ds_load this slot
.else
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    // advance bases: A += s14 (MT*256), B += s9 (NT*256)
    s_add_u32  s36, s36, s14
    s_addc_u32 s37, s37, 0
    s_add_u32  s38, s38, s9
    s_addc_u32 s39, s39, 0
.if BUSYWAIT
    lds_barrier                               // reuse fence: compute done reading before next overwrite
.else
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    s_add_u32  s26, s26, 1
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lload_loop
    s_branch .Lclaim_loop                     // loaders: claim next tile

    // -------- COMPUTE BODY: ds_load own A-band frags + shared B frags + WMMA --------
.Lcompute_role:
    s_mov_b32 exec_lo, s25                     // restore full exec (compute waves)
.if DYNVGPR
    s_alloc_vgpr NFV                           // T4: compute grow to tile footprint. No SCC guard: at lean
.endif                                          //   tiles NFV (45 @2x2) << cap, so grow is satisfiable by construction.
                                                //   For tiles whose peak-live > cap, the dispatch must umr-lift
                                                //   SQ_DYN_VGPR.BLOCK_SIZE=1 first (deferred until a >128-peak sweep cell).
    // cid = wid - NLOAD ; A-band LDS frag base = cid*FM*256
    v_sub_nc_u32 v3, v1, NLOAD
    v_readfirstlane_b32 s27, v3                // cid (uniform per wave)
    s_mul_i32 s28, s27, (FM*256)              // A-band byte base in the slot
    // zero accumulators
    .set i, 0
    .rept FM*FN*8
      v_mov_b32 v[ACC+i], 0
      .set i, i+1
    .endr
    s_mov_b32 s26, 0                          // kt = 0
.Lcomp_loop:
.if BUSYWAIT
    lds_barrier                               // wait for loader publish of this slot
.else
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    // ds_load FM A frags (this band) : addr = lane*8 ; offset = s28 + mi*256  -> needs per-mi offset
    //   (s28 is uniform; fold into a vaddr once)
    v_add_nc_u32 v11, v9, s28                  // v11 = lane*8 + cid*FM*256 (A-band base)
    .set mi, 0
    .rept FM
      ds_load_b64 v[FA+mi*2:FA+mi*2+1], v11 offset:mi*256
      .set mi, mi+1
    .endr
    // ds_load FN B frags (shared) : offset = ASLICE + ni*256
    .set ni, 0
    .rept FN
      ds_load_b64 v[FB+ni*2:FB+ni*2+1], v9 offset:ASLICE+ni*256
      .set ni, ni+1
    .endr
    s_wait_dscnt 0x0
    // FM*FN WMMA
    .set mi, 0
    .rept FM
      .set ni, 0
      .rept FN
        v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7], v[FA+mi*2:FA+mi*2+1], v[FB+ni*2:FB+ni*2+1], v[ACC+(mi*FN+ni)*8:ACC+(mi*FN+ni)*8+7]
        .set ni, ni+1
      .endr
      .set mi, mi+1
    .endr
.if BUSYWAIT
    lds_barrier                               // reuse fence (pair with loader's)
.else
    s_barrier_signal -1
    s_barrier_wait -1
.endif
    s_add_u32 s26, s26, 1
    s_cmp_lt_u32 s26, s12
    s_cbranch_scc1 .Lcomp_loop

.if STORE
    // store base = C + ti*(NCOMP*FM*FN*1024) + cid*(FM*FN*1024)
    s_mul_i32 s30, s17, (NCOMP*FM*FN*1024)
    s_mul_i32 s31, s27, (FM*FN*1024)
    s_add_u32 s30, s30, s31
    s_add_u32 s28, s6, s30
    s_addc_u32 s29, s7, 0
    .set frag, 0
    .rept FM*FN
      global_store_b128 v10, v[ACC+frag*8:ACC+frag*8+3], s[28:29] offset:(frag*1024)
      global_store_b128 v10, v[ACC+frag*8+4:ACC+frag*8+7], s[28:29] offset:(frag*1024+16)
      .set frag, frag+1
    .endr
    s_wait_storecnt 0x0
.endif
    s_branch .Lclaim_loop                      // compute: claim next tile

.Lexit:
    s_wait_storecnt 0x0
    // ---- end timer: leader (tid==0) writes occ[3] = max realtime tick ----
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
    // ---- live-- (leader): occ[0] -= 1 so the harness completion gate (occ[0]==0) can fire ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v5, -1
    global_atomic_add_u32 v7, v5, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s16
    s_endpgm

// ---- RGADESC: analysis-only AMDHSA descriptor + metadata so RGA (-s bin --co) can find/analyze the
//   kernel. NOT emitted for the PM4 .bin (harness builds its own descriptor from .text). Fixed box
//   (vgpr 256 / lds 32772) so livereg reports the TRUE peak-live across any swept config. ----
.if RGADESC
.amdhsa_kernel occ_kernel
    .amdhsa_next_free_vgpr 256
    .amdhsa_next_free_sgpr 46
    .amdhsa_group_segment_fixed_size 32772
    .amdhsa_user_sgpr_count 15
    .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name:            occ_kernel
    .symbol:          occ_kernel.kd
    .kernarg_segment_size: 64
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 32772
    .private_segment_fixed_size: 0
    .wavefront_size:  32
    .sgpr_count:      46
    .vgpr_count:      256
    .max_flat_workgroup_size: 256
    .args:            []
.end_amdgpu_metadata
.endif
