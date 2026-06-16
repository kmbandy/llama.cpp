// occ_timercheck.s  (gfx1201, wave32) -- measure the s_sendmsg REALTIME counter tick rate.
//
// The combined/throughput kernels convert REALTIME-span ticks to seconds via freq_hz (assumed
// 100 MHz from hsaKmtGetClockCounters). If that assumption is wrong, every PM4-timed TFLOPS number
// is off by the same factor. This kernel resolves it directly: busy-wait until REALTIME advances by
// s6 ticks, then exit. The host times the dispatch; running two targets (T and 2T) cancels the
// fixed submit/fence overhead -> actual_freq = (2T - T) / (secs(2T) - secs(T)).
//
// run_variant-compatible: occ[0]=live (++admit / --exit), occ[1]=maxlive, occ[4]=total,
// occ[2]=min(start REALTIME), occ[3]=max(end REALTIME). s6 = target ticks. grid = 1 wave.
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_mov_b32 v4, 0
    // ---- lane-0 admission: live++, maxlive, total++ ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .La
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV
    global_atomic_add_u32 v4, v2, s[0:1] offset:16 scope:SCOPE_DEV
.La:
    s_mov_b32 exec_lo, s8
    // ---- r0 = REALTIME (start) ----
    s_sendmsg_rtn_b64 s[10:11], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_mov_b32 s12, s10                    // start_lo
    // record start -> occ[2] = min(start)
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lb
    v_mov_b32 v12, s12
    global_atomic_min_u32 v4, v12, s[0:1] offset:8 scope:SCOPE_DEV
.Lb:
    s_mov_b32 exec_lo, s8
    // ---- busy-wait until (REALTIME - start) >= s6 ----
.Lwait:
    s_sendmsg_rtn_b64 s[10:11], sendmsg(MSG_RTN_GET_REALTIME)
    s_wait_kmcnt 0x0
    s_sub_u32 s13, s10, s12               // elapsed = now - start
    s_cmp_lt_u32 s13, s6                  // while elapsed < target
    s_cbranch_scc1 .Lwait
    // record end -> occ[3] = max(end)
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lc
    v_mov_b32 v12, s10
    global_atomic_max_u32 v4, v12, s[0:1] offset:12 scope:SCOPE_DEV
    v_mov_b32 v2, s13                     // elapsed ticks -> fragOut[0] (sanity)
    global_store_b32 v4, v2, s[4:5]
    s_wait_storecnt 0x0
.Lc:
    s_mov_b32 exec_lo, s8
    // ---- live-- ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ld
    v_mov_b32 v2, -1
    global_atomic_add_u32 v4, v2, s[0:1] scope:SCOPE_DEV
.Ld:
    s_mov_b32 exec_lo, s8
    s_endpgm
    .size occ_kernel, .-occ_kernel
