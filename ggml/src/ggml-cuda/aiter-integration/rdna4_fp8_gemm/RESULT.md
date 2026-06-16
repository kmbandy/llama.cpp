# RESULT — RDNA4 fp8 WMMA GEMM, CDNA4-recipe ladder (MAD-305)

**Hardware:** AMD R9700 / gfx1201 (RDNA4), wave32, ROCm 7.2.3.
**Spec:** `docs/superpowers/specs/2026-06-16-rdna4-cdna4-transpose-fed-fp8-wmma-pipeline-design.md`
**Reference ceiling:** 307 TF raw-WMMA fp8 (`bench/wmma_peak.hip`); hipBLASLt = 143 TF.

Per-lever ladder, mirroring the CDNA4 fp8 GEMM recipe (ck_tile `comp_v6` + AITER blockscale),
ported onto the gfx12 WMMA path. Each phase gated on the fp8 e4m3 oracle.

| Phase | Lever | TFLOPS | % of 307 | × hipBLASLt | oracle |
|---|---|---|---|---|---|
| baseline | byte-gather B feed (`gemm_wmma.hip`) | ~69–90 | 22–29% | 0.5–0.6× | PASS |
| 0 | `global_load_tr_b64` layout probe (de-risk) | — | — | — | PASS (bit-exact) |
| 1 | wide feed (preshuffle + transpose-load) | _pending_ | | | |
| 2 | async double-buffer | _pending_ | | | |
| 3 | big tiles + scheduler + wave32 occupancy | _pending_ | | | |
| 4 | ml8 4-bit LUT front-end | _pending_ | | | |
| 5 | production integration + PPL-neutral | _pending_ | | | |

## Phase 0 — global_load_tr_b64 layout probe (GATE PASSED)

- `global_load_tr_b64` confirmed available + lowering on gfx1201 (asm: 2 sites). The gfx12 *global*
  transpose-load is the only hardware wide-feed (gfx950's `ds_read_tr8` LDS transpose is absent).
- The instruction does a fixed 8×8 byte transpose; a plain row-major `[K][N]` tile does **not**
  transpose into the WMMA fragment. Source must be **pre-shuffled**.
- Derived + hand-verified the closed-form byte permutation `trperm(L,s)` and the per-tile
  preshuffle rule (`bench/global_load_tr_contract.md`).
- Proved end-to-end: a single 16×16×16 fp8 WMMA fed via `global_load_tr` + preshuffle matches the
  CPU e4m3 oracle **bit-exact (max_rel_err = 0.0000)**.
- Production mechanism established: B is static weights, so the preshuffle is a one-time load-time
  repack (free at runtime) — same idea as AITER `ck_gemm_a8w8_bpreshuffle`. Hot loop = one
  `global_load_tr_b64` per fragment, no byte-gather, no LDS staging for B.

**Reproduce:**
```
./build.sh
./out/global_load_tr_probe            # discovery: per-lane transpose table
./out/global_load_tr_probe --validate # gate: single-tile WMMA vs oracle -> PASS (0.0000)
```
