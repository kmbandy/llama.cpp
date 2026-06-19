// occ_kernel_ldsbound.s  (gfx1201) -- LDS BOUNDARY smoke: validate the raw-PM4 RSRC2.GRANULATED_LDS_SIZE
// encoding at a real Phase-2 size. One 128-thread (4-wave) workgroup. Leader writes a sentinel to
// LDS[0] AND LDS[lastoff] (s4 = ldsBytes-4), barrier, every wave reads both words back and (lane 0)
// writes the two readbacks to occ[8 + wid*2 .. +1]. Leader then sets occ[0]=0xD0 (done). If the
// allocation truly covers ldsBytes, all 4 waves read 0xAAAA1111 / 0xBBBB2222 at both ends.
//
// USER_SGPR=15: s0:s1 = occ base, s4 = lastByteOffset (= ldsBytes-4). nWG=1.
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_lshrrev_b32 v1, 5, v0           // wid
    v_and_b32     v2, 31, v0          // lane
    v_mov_b32     v7, 0               // LDS addr 0 / store vaddr base

    // ---- leader writes both sentinels ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_w
    v_mov_b32 v5, 0xAAAA1111
    ds_store_b32 v7, v5               // LDS[0]
    v_mov_b32 v6, s4
    v_mov_b32 v5, 0xBBBB2222
    ds_store_b32 v6, v5               // LDS[lastoff]
    s_wait_dscnt 0x0
.Lafter_w:
    s_mov_b32 exec_lo, s16
    s_barrier_signal -1
    s_barrier_wait -1

    // ---- every wave reads both back ----
    ds_load_b32 v8, v7               // LDS[0]
    v_mov_b32 v6, s4
    ds_load_b32 v9, v6               // LDS[lastoff]
    s_wait_dscnt 0x0

    // ---- lane 0 of each wave stores readbacks to occ[8 + wid*2 .. +1] ----
    v_cmp_eq_u32 vcc_lo, 0, v2
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_st
    v_lshlrev_b32 v6, 3, v1          // wid*8 bytes
    s_wait_alu 0xfffe
    global_store_b32 v6, v8, s[0:1] offset:32
    global_store_b32 v6, v9, s[0:1] offset:36
    s_wait_storecnt 0x0
.Lafter_st:
    s_mov_b32 exec_lo, s16
    s_barrier_signal -1
    s_barrier_wait -1

    // ---- leader sets done flag occ[0]=0xD0 (all per-wave stores ordered by the barrier above) ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s16, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lfin
    v_mov_b32 v5, 0xD0
    global_store_b32 v7, v5, s[0:1]
    s_wait_storecnt 0x0
.Lfin:
    s_mov_b32 exec_lo, s16
    s_endpgm
    .size occ_kernel, .-occ_kernel
