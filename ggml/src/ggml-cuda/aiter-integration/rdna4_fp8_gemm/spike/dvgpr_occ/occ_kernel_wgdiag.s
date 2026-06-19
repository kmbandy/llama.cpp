// occ_kernel_wgdiag.s  (gfx1201) -- SGPR PROBE: find which SGPR carries the per-workgroup id under
// raw PM4 with USER_SGPR=15. Each workgroup-leader (tid==0) grabs a unique ordinal via an atomic on
// occ[offset:16], then dumps the ENTRY values of s8..s23 to occ[ (256 + ord*64) + k*4 ]. The host
// reads back; the SGPR column whose values form a permutation of 0..nWG-1 across ordinals IS the
// workgroup-id register. s0:s1 = occ base (only USER_SGPR slot the probe needs).
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // capture s8..s23 into s40..s55 BEFORE any clobber (these hold the entry SGPR state)
    s_mov_b32 s40, s8
    s_mov_b32 s41, s9
    s_mov_b32 s42, s10
    s_mov_b32 s43, s11
    s_mov_b32 s44, s12
    s_mov_b32 s45, s13
    s_mov_b32 s46, s14
    s_mov_b32 s47, s15
    s_mov_b32 s48, s16
    s_mov_b32 s49, s17
    s_mov_b32 s50, s18
    s_mov_b32 s51, s19
    s_mov_b32 s52, s20
    s_mov_b32 s53, s21
    s_mov_b32 s54, s22
    s_mov_b32 s55, s23
    // leader-only (tid==0 over the whole workgroup)
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s60, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    // ord = atomic_add(occ + 16, 1) -> old value (unique 0..nWG-1)
    v_mov_b32 v1, 0
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v1, v2, s[0:1] offset:16 th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_readlane_b32 s61, v3, 0
    // dump base = occ + 256 + ord*64
    s_lshl_b32 s62, s61, 6
    s_add_u32 s62, s62, 256
    s_add_u32 s64, s0, s62
    s_addc_u32 s65, s1, 0
    v_mov_b32 v4, 0
    .set k, 0
    .rept 16
      v_mov_b32 v5, s[40+k]
      global_store_b32 v4, v5, s[64:65] offset:k*4
      .set k, k+1
    .endr
    s_wait_storecnt 0x0
.Ldone:
    s_mov_b32 exec_lo, s60
    s_endpgm
    .size occ_kernel, .-occ_kernel
