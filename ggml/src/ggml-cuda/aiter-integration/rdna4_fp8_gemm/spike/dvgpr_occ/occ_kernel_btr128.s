// occ_kernel_btr128.s  (gfx1201, wave32) -- MAD-305 Lever A micro-oracle (GPT step 1):
//   PROVE the fp8 fragment SEMANTICS of global_load_tr_b128 against the known-good global_load_tr_b64.
//
// Lever A (instruction ledger): replacing two global_load_tr_b64 (8B/lane each = two adjacent fp8 B
// K-frags) with one global_load_tr_b128 (16B/lane) HALVES the B feed-instruction count -- IF b128
// produces the SAME fp8 transpose layout (2 adjacent frags) and is NOT a 16-bit-oriented transpose
// (which would scramble fp8 bytes -> wrong WMMA operand). This isolates that question. No WMMA, no LDS.
//
// One wave (32 lanes). Inputs:
//   s[4:5] = B buffer, encoded so each byte VALUE = its global K-row index:
//            byte at offset o (0..511): tile=o/256, k_local=(o%256)/16, col=o%16 ; value = tile*16 + k_local.
//            tile0 = bytes [0,256) (global K 0..15), tile1 = bytes [256,512) (global K 16..31).
//   s[6:7] = sink, 32 bytes/lane: [ tr_b64(tile0) 8B | tr_b64(tile1) 8B | tr_b128(tile0+tile1) 16B ].
//
// global_load_tr vaddr is per-lane byte base (proven in occ_kernel_btr.s: tr_b64 uses lane*8). b128 is
// the natural analog at lane*16. The harness decodes the per-lane byte pattern: under the desired
// "2 adjacent fp8 frags" semantics, b128[0..7]==tr_b64(tile0) and b128[8..15]==tr_b64(tile1).
//
// USER_SGPR=15 (matches run_btr): s0:1=occ(unused) s2:3=A(unused) s4:5=B s6:7=sink.
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    v_and_b32     v2, 31, v0                 // lane = tid & 31
    v_lshlrev_b32 v9,  3, v2                  // lane*8   (tr_b64 per-lane vaddr)
    v_lshlrev_b32 v11, 4, v2                  // lane*16  (tr_b128 per-lane vaddr)
    v_lshlrev_b32 v10, 5, v2                  // lane*32  (sink per-lane store base)
    // ---- known-good: two fp8 B 16x16 frags via global_load_tr_b64 ----
    global_load_tr_b64  v[40:41], v9,  s[4:5] offset:0      // tile0 frag (global K 0..15)
    global_load_tr_b64  v[42:43], v9,  s[4:5] offset:256    // tile1 frag (global K 16..31)
    // ---- candidate: both frags in one wide load via global_load_tr_b128 ----
    global_load_tr_b128 v[44:47], v11, s[4:5] offset:0      // 16B/lane from [0,512)
    s_wait_loadcnt 0x0
    // ---- dump per-lane fragment bytes: [f0(8) | f1(8) | b128(16)] -> sink[lane*32 + {0,8,16}] ----
    global_store_b64  v10, v[40:41], s[6:7] offset:0
    global_store_b64  v10, v[42:43], s[6:7] offset:8
    global_store_b128 v10, v[44:47], s[6:7] offset:16
    s_wait_storecnt 0x0
    s_endpgm
    .size occ_kernel, .-occ_kernel
