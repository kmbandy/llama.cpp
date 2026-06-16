# gfx1201 `global_load_tr_b64` fp8 B-fragment contract (MAD-305 Phase 0, verified 2026-06-16)

Verified by `bench/global_load_tr_probe.hip`: the instruction lowers to `global_load_tr_b64`
(asm check, 2 sites) and a single 16×16×16 fp8 WMMA fed through it matches the CPU e4m3 oracle
**bit-exact (max_rel_err = 0.0000)**.

## The instruction

Per-lane (wave32): `__builtin_amdgcn_global_load_tr_b64_v2i32(const v2i32 addrspace(1)* p)`
returns a `v2i32` (8 fp8 bytes) — the lane's WMMA B-operand fragment. Lane L passes
`p = Bshuf + L*8` (8-byte aligned). The hardware performs a fixed 8×8 byte transpose across a
`{0,1,2,3,8,9,10,11}`-style lane grouping; it does **not** transpose a plain row-major `[K][N]`
tile into the fragment. The source must be **pre-shuffled** into the layout below.

## The transpose permutation (closed form, hand-verified against the probe table)

With lane L passing `Bshuf + L*8`, output `(lane L, slot s)` receives byte `Bshuf[trperm(L,s)]`:

```cpp
__host__ __device__ inline int trperm(int L, int s) {   // L in 0..31, s in 0..7
    int base = (L & 7) + ((L >> 3) & 1) * 32 + ((L >> 4) & 1) * 128;
    return base + (s & 3) * 8 + ((s >> 2) & 1) * 64;
}
```

## The pre-shuffle (B is static weights → repack once, free at runtime)

To make `tr_load8(Bshuf + L*8)` deliver the §7.12 WMMA B fragment — output lane L holds N-column
`n = L & 15`, K-bytes `k = ((L>>4)&1)*8 + s` for slot `s` — place each logical `B[k][n]` at:

```cpp
// for one 16(K) x 16(N) tile of logical B (row-major), build the 256-byte Bshuf tile:
for (int L = 0; L < 32; ++L)
  for (int s = 0; s < 8; ++s) {
    int k = ((L >> 4) & 1) * 8 + s, n = L & 15;
    Bshuf[trperm(L, s)] = Blogical[k * 16 + n];
  }
```

`trperm` is a bijection over `0..255`, so `Bshuf` is a fixed permutation of the tile's bytes.
This mirrors AITER's `ck_gemm_a8w8_bpreshuffle` pattern: the weight matrix is pre-shuffled once at
load time, and the GEMM hot loop then feeds the matrix core with a single `global_load_tr_b64`
per fragment — no byte-gather, no LDS staging for B.

## Phase-1 consumption

- Add a one-time host/prepass repack of the fp8 weight `B[K][N]` into 16×16 `Bshuf` tiles via the
  rule above (generalize the per-tile permutation across the `[K][N]` grid + tile origins).
- Replace the B byte-gather in `gemm_wmma.hip` with `tr_load8(Bshuf + <tile-addr> + lane*8)`,
  feeding the result straight into `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
- Open Phase-1 question (the crux): direct-from-global feed loses explicit LDS reuse for B —
  measure whether L2 reuse holds at 4096³, fallback = larger M-tile to amortize B re-fetch.
