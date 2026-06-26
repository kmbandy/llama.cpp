// occ_kernel_dynsmoke.s — MAD-305: 2-wave dyn-VGPR COORDINATION isolation probe (NO GEMM, NO external buffers).
//
// Purpose: the cooperative GEMM bricks ONLY in dyn (DYNVGPR=1) with 2 waves/WG. Static analysis refuted
// addressing (page fault gone after feed s_alloc), under-allocation, arming, and Σ>pool (144 << 1152/SIMD).
// What's left: the gfx12 s_alloc_vgpr COORDINATION semantics across a 2-wave WG. This probe strips EVERYTHING
// except that: wave0 = feed (lean, optionally s_alloc 32, then park), wave1 = compute (elastic grow GROWSZ ->
// write top-of-grown VGPR -> store -> shrink -> signal). Only writes to its own occ buffer (in-bounds, brick
// can ONLY come from the s_alloc coordination itself, never an OOB address). Elastic grow PRESERVED (this is
// dyn being dyn — NOT grow-once).
//
// Variants (defsym): FEEDALLOC 1=feed issues s_alloc 32 (current coop fix) / 0=feed never allocs (orig coop).
//                    GROWSZ    compute grow target (default 112, the 2x4 footprint).
//                    LDSWAIT   0 = original write-top/shrink/signal smoke (occ-only, fa0/fa1 bins).
//                              1 = LDS PING-PONG: the GROWN compute wave spin-waits on the LEAN feed via an
//                                  LDS prod/cons handshake — i.e. a wave WAITS in a HIGH-VGPR section. This is
//                                  the EXACT hazard the real coop K-loop hits (RDNA4 ISA §3.3.3 guidance:
//                                  "waves should only wait in LOW-VGPR sections"). The passing fa1 smoke had
//                                  ONLY the lean feed wait; this reverses it. Needs LDS (run_dynsmoke env
//                                  ML8_SMOKE_LDS=512). occ-only + LDS-only => still in-bounds / brick-safe.
//                    ROUNDS    LDSWAIT ping-pong iterations (default 4). Coop wedged at ~step 1-2.
// occ status map (lane0-of-wave writes a tag; harness dumps occ[0..7]):
//   occ[0]=0xFEE0 feed alive   occ[1]=0xC0E0 compute grown   occ[2]=v[GROWSZ-1] readback OR last VAL (LDSWAIT)
//   occ[3]=0x5417 compute shrunk   occ[4]=0xDEAD compute DONE   occ[5]=0xF00D feed exiting (saw DONE)
//   occ[6]=max prod round published by feed   occ[7]=max cons round published by GROWN compute (LDSWAIT only)
// LDSWAIT=1: ALL set + EXIT 0 => a grown wave CAN wait on a lean wave under dyn (hypothesis REFUTED).
//            HANG with occ[6]=2 occ[7]=1 => reproduced the coop standoff (grown-wait hazard CONFIRMED), brick-safe.

.ifndef FEEDALLOC
    .set FEEDALLOC, 1
.endif
.ifndef GROWSZ
    .set GROWSZ, 112
.endif
.ifndef LDSWAIT
    .set LDSWAIT, 0
.endif
.ifndef ROUNDS
    .set ROUNDS, 4
.endif
.ifndef GVMEM
    .set GVMEM, 0                               // LDSWAIT add-on: issue an in-flight global_load (VMEM) ACROSS the
.endif                                          //   cross-wave wait in BOTH waves (in-bounds occ read). Tests the
                                                //   #1 coop-brick suspect: outstanding vmcnt to a VGPR while a wave
                                                //   is parked in a dyn cross-wave busy-wait (feed=global_load_tr B,
                                                //   compute=global_load A). Compute loads to a HIGH vgpr (v64).
.ifndef GWMMA
    .set GWMMA, 0                               // LDSWAIT add-on: grown compute runs an accumulating fp8 WMMA each
.endif                                          //   round into a LIVE high-VGPR acc (v80..v87) that survives across
                                                //   every cross-wave wait, read back to occ[2] before shrink. Tests
                                                //   WMMA + live-high-acc under a dyn cross-wave wait (the real coop step).
.ifndef GTR
    .set GTR, 0                                 // LDSWAIT add-on: LEAN feed issues a TRANSPOSE VMEM load
.endif                                          //   (global_load_tr_b64, the feed's one untested op) across its lean gate,
                                                //   in-bounds (occ). DIAG showed the coop feed wedges in its step-1 work
                                                //   (load_tr->ds_store->publish) BEFORE any flow-control gate. #1 suspect now.
.ifndef GRING
    .set GRING, 0                               // LDSWAIT add-on: the LARGEST untested structural delta. Replaces the
.endif                                          //   fixed-slot b32 handshake DATA path with the coop's REAL B-ring:
.ifndef RINGD                                   //   depth-RINGD wrapping ds_store_b64/ds_load_b64 at a 256B slot stride,
    .set RINGD, 2                               //   per-lane lane*8 addressing, coop-faithful slot-free gate (feed may run
.endif                                          //   RINGD ahead -> exercises the wraparound WAR concurrency). PROD/CONS
                                                //   gating kept. needs ML8_SMOKE_LDS=1024. occ+LDS-only -> brick-safe.
.set HIVGPR, (GROWSZ-1)                         // top of grown range to exercise (proves grown VGPRs usable)

// LDS layout. LDSWAIT (no ring): 3 u32 in the first granule. GRING: B-ring[RINGD*256] then the counters.
.if GRING
.set BRING_OFF, 0                               // B ring: RINGD slots x 256B (FN=1 frag/slot, 32 lanes x 8B = 256B)
.set PROD_OFF, (RINGD*256)
.set VAL_OFF,  (PROD_OFF + 4)                   // (unused under GRING; ring carries the payload)
.set CONS_OFF, (PROD_OFF + 8)
.else
.set PROD_OFF, 0                                // feed-published monotonic produce counter
.set VAL_OFF,  4                                // feed payload (0xBEEF0000 | round)
.set CONS_OFF, 8                                // compute-published monotonic consume counter (release)
.endif

.macro wstore off, vval                         // lane0-of-wave: occ[off/4] = vval (device scope)
    s_mov_b32 s16, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lws\@
    global_store_b32 v4, \vval, s[0:1] offset:\off scope:SCOPE_DEV
    s_wait_storecnt 0x0
.Lws\@:
    s_mov_b32 exec_lo, s16
.endm

.macro lds_get sdst, off                        // wave-uniform read LDS[off] -> scalar sdst (mirrors coop)
    v_mov_b32 v27, \off
    ds_load_b32 v28, v27
    s_wait_dscnt 0x0
    v_readfirstlane_b32 \sdst, v28
.endm

.macro lds_put off, ssrc                         // lane-0-of-wave writes scalar ssrc -> LDS[off] (mirrors coop)
    s_mov_b32 s49, exec_lo
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lput\@
    v_mov_b32 v28, \off
    v_mov_b32 v29, \ssrc
    ds_store_b32 v28, v29
    s_wait_dscnt 0x0
.Lput\@:
    s_mov_b32 exec_lo, s49
.endm

    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_lshrrev_b32 v1, 5, v0                      // wid = tid >> 5   (0 = feed, 1 = compute)
    v_and_b32     v2, 31, v0                     // lane = tid & 31
    v_mov_b32     v4, 0                          // per-lane store/load vaddr base = 0 (lean reg, survives grow)
.if GRING
    v_lshlrev_b32 v9, 3, v2                      // v9 = lane*8 = per-lane LDS byte base for the b64 ring (lean, survives grow)
.endif

    // ---- role split (mirrors coop): wid==0 -> feed, else -> compute ----
    v_cmp_eq_u32 vcc_lo, 0, v1
    s_mov_b32 s17, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lcompute
    s_mov_b32 exec_lo, s17                       // feed waves only

// ============================== FEED (wave 0, lean) ==============================
.if FEEDALLOC
.Lfeed_alloc:
    s_alloc_vgpr 32                              // PARTICIPATE in the dyn-VGPR WG protocol (the coop fix under test)
    s_cbranch_scc0 .Lfeed_alloc
.endif
    v_mov_b32 v5, 0xFEE0
    wstore 0, v5                                 // occ[0] = feed alive

.if LDSWAIT
// ---- LDS PING-PONG producer (lean): publish VAL+PROD each round, gate on CONS (slot-free) ----
//   feed waits ONLY here, while LEAN — the ALLOWED pattern. The hazard under test is the GROWN consumer.
    s_mov_b32 s40, 0                             // n = round counter (0-based)
.Lfeed_round:
.if GTR
    global_load_tr_b64 v[16:17], v4, s[0:1] offset:0 scope:SCOPE_DEV  // TRANSPOSE VMEM load (feed's untested op), occ in-bounds
.endif
.if GVMEM
    global_load_b64 v[6:7], v4, s[0:1] offset:0 scope:SCOPE_DEV  // in-flight VMEM (occ[0..1], in-bounds) across the lean gate
.endif
.if GRING
    // coop-faithful slot-free gate: step s40 may run up to RINGD ahead (exercises wraparound concurrency)
    s_cmp_lt_u32 s40, RINGD
    s_cbranch_scc1 .Lfeed_slotok                 // step < RINGD: slot free, no gate
    s_sub_u32 s46, s40, RINGD
.Lfeed_gw:
    s_sleep 2
    lds_get s42, CONS_OFF
    s_cmp_le_u32 s42, s46                         // cons <= step-RINGD -> the slot we want is still undrained -> spin
    s_cbranch_scc1 .Lfeed_gw
.Lfeed_slotok:
.else
    // slot-free gate: wait until cons >= n  (depth-1 handshake; n is the round we are about to produce)
.Lfeed_gate:
    s_sleep 2
    lds_get s42, CONS_OFF
    s_cmp_lt_u32 s42, s40
    s_cbranch_scc1 .Lfeed_gate                   // cons < n -> spin (lean wait, allowed)
.endif
.if (GVMEM || GTR)
    s_wait_loadcnt 0x0                           // consume the in-flight load(s) AFTER the cross-wave wait
.endif
    // produce: publish PROD = n+1 (after staging the payload)
    s_add_u32 s41, s40, 1
.if GRING
    // ds_store_b64 the payload into the wrapping ring slot (slot = step & (RINGD-1), 256B stride, per-lane lane*8)
    s_and_b32 s45, s40, (RINGD-1)
    s_mul_i32 s45, s45, 256
    v_add_nc_u32 v13, v9, s45
    v_mov_b32 v6, 0xB10C0000
    v_mov_b32 v7, s41                            // hi = round id (n+1) so the consumer can verify the wrapping slot
    ds_store_b64 v13, v[6:7]
    s_wait_dscnt 0x0
.else
    s_or_b32  s43, s41, 0xBEEF0000
    lds_put VAL_OFF, s43                          // payload first...
.endif
    lds_put PROD_OFF, s41                         // ...then publish (release): any consumer seeing PROD reads valid payload
    v_mov_b32 v5, s41
    wstore 24, v5                                 // occ[6] = max prod round published
    s_mov_b32 s40, s41
    s_cmp_lt_u32 s40, ROUNDS
    s_cbranch_scc1 .Lfeed_round
    v_mov_b32 v5, 0xF00D
    wstore 20, v5                                // occ[5] = feed done
    s_endpgm
.else
.Lfeed_park:
    s_sleep 4
    global_load_b32 v6, v4, s[0:1] offset:16 scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_readfirstlane_b32 s18, v6
    s_cmp_eq_u32 s18, 0xDEAD
    s_cbranch_scc0 .Lfeed_park                   // spin until compute signals DONE
    v_mov_b32 v5, 0xF00D
    wstore 20, v5                                // occ[5] = feed exiting
    s_endpgm
.endif

// ============================== COMPUTE (wave 1, elastic) ==============================
.Lcompute:
    s_mov_b32 exec_lo, s17                       // restore full exec for compute waves
.Lgrow:
    s_wait_loadcnt 0x0
    s_wait_storecnt 0x0
    s_alloc_vgpr GROWSZ                          // ELASTIC grow (SCC-retry guard)
    s_cbranch_scc0 .Lgrow
    v_mov_b32 v[HIVGPR], 0x5A5A5A5A              // exercise the TOP of the grown range
    v_mov_b32 v5, 0xC0E0
    wstore 4, v5                                 // occ[1] = compute grown
    wstore 8, v[HIVGPR]                          // occ[2] = readback (must be 0x5A5A5A5A)

.if LDSWAIT
// ---- LDS PING-PONG consumer (GROWN): spin-wait on PROD WHILE HOLDING GROWSZ VGPR — the hazard ----
.if GWMMA
    v_mov_b32 v68, 0x3C3C3C3C                    // A frag (fp8 bytes, nonzero) — value irrelevant, must just execute
    v_mov_b32 v69, 0x3C3C3C3C
    v_mov_b32 v70, 0x3C3C3C3C                    // B frag
    v_mov_b32 v71, 0x3C3C3C3C
    v_mov_b32 v80, 0                             // acc[0..7] = 0 (LIVE high-VGPR acc across all waits)
    v_mov_b32 v81, 0
    v_mov_b32 v82, 0
    v_mov_b32 v83, 0
    v_mov_b32 v84, 0
    v_mov_b32 v85, 0
    v_mov_b32 v86, 0
    v_mov_b32 v87, 0
.endif
    s_mov_b32 s40, 0                             // n = round counter (0-based)
.Lcomp_round:
.if GVMEM
    global_load_b64 v[64:65], v4, s[0:1] offset:0 scope:SCOPE_DEV  // in-flight VMEM to a HIGH vgpr across the GROWN gate
.endif
    // ready gate: wait until prod > n  (i.e. prod >= n+1) -- THE GROWN WAIT under test
.Lcomp_gate:
    s_sleep 2
    lds_get s41, PROD_OFF
    s_cmp_le_u32 s41, s40
    s_cbranch_scc1 .Lcomp_gate                   // prod <= n -> spin (GROWN wait — suspected deadlock)
.if GVMEM
    s_wait_loadcnt 0x0                           // consume the in-flight high-vgpr load AFTER the grown wait
.endif
    // consume: read payload, stash readback, then release cons = n+1
.if GRING
    // ds_load_b64 from the wrapping ring slot (slot = step & (RINGD-1), per-lane lane*8) — the real coop B read
    s_and_b32 s45, s40, (RINGD-1)
    s_mul_i32 s45, s45, 256
    v_add_nc_u32 v13, v9, s45
    ds_load_b64 v[66:67], v13
    s_wait_dscnt 0x0
    v_mov_b32 v5, v67                             // hi = round id from the slot (verifies wrapping ring visibility)
.else
    lds_get s43, VAL_OFF
    v_mov_b32 v5, s43
.endif
    wstore 8, v5                                  // occ[2] = last payload readback (cross-wave visible)
    s_add_u32 s44, s40, 1
    lds_put CONS_OFF, s44                          // release the slot
    v_mov_b32 v5, s44
    wstore 28, v5                                 // occ[7] = max cons round published by GROWN compute
.if GWMMA
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[68:69], v[70:71], v[80:87]  // accumulate into LIVE high acc (the coop step)
.endif
    s_mov_b32 s40, s44
    s_cmp_lt_u32 s40, ROUNDS
    s_cbranch_scc1 .Lcomp_round
.if GWMMA
    s_nop 0                                       // let the final WMMA retire
    v_mov_b32 v5, v80
    wstore 8, v5                                  // occ[2] = acc[0] after ROUNDS WMMAs (live high acc survived the waits)
.endif
.endif

.Lshrink:
    s_wait_storecnt 0x0
.if LDSWAIT
    s_wait_dscnt 0x0                             // drain the ping-pong ds traffic before shrink (LDSWAIT only)
.endif
    s_alloc_vgpr 32                              // ELASTIC shrink (SCC-retry guard)
    s_cbranch_scc0 .Lshrink
    v_mov_b32 v5, 0x5417
    wstore 12, v5                                // occ[3] = compute shrunk
    v_mov_b32 v5, 0xDEAD
    wstore 16, v5                                // occ[4] = DONE -> signals feed (non-LDSWAIT park path)
    s_endpgm
    .size occ_kernel, .-occ_kernel
