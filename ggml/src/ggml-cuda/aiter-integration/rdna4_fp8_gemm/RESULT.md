# RESULT — RDNA4 fp8 WMMA GEMM, CDNA4-recipe ladder (MAD-305)

**Hardware:** AMD R9700 / gfx1201 (RDNA4), wave32, ROCm 7.2.3.
**Spec:** `docs/superpowers/specs/2026-06-16-rdna4-cdna4-transpose-fed-fp8-wmma-pipeline-design.md`
**Reference ceiling:** 307 TF raw-WMMA fp8 (`bench/wmma_peak.hip`); hipBLASLt = 143 TF.

Per-lever ladder, mirroring the CDNA4 fp8 GEMM recipe (ck_tile `comp_v6` + AITER blockscale),
ported onto the gfx12 WMMA path. Each phase gated on the fp8 e4m3 oracle.

| Phase | Lever | TFLOPS | % of 307 | × hipBLASLt | oracle |
|---|---|---|---|---|---|
| baseline | byte-gather B feed (`gemm_wmma.hip`) | 68.7 | 22% | 0.48× | PASS |
| 0 | `global_load_tr_b64` layout probe (de-risk) | — | — | — | PASS (bit-exact) |
| 1 | wide feed (preshuffle + transpose-load) | **140.3** | **46%** | **0.98×** | PASS |
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

## Phase 1 — wide feed: preshuffle + global_load_tr (GATE PASSED)

The B byte-gather (8 strided `ds_load_u8` per fragment, the ~69 TF wall) is replaced by ONE
`global_load_tr_b64` per B fragment, fed from a pre-shuffled global weight buffer (Phase-0
contract). A still stages in LDS and reads wide; B no longer touches LDS.

| variant | 4096³ TFLOPS | % of 307 | × hipBLASLt | oracle / agreement |
|---|---|---|---|---|
| baseline (byte-gather) | 68.7 | 22% | 0.48× | PASS |
| **trfeed bm128 (winner)** | **140.3** | **46%** | **0.98×** | PASS, bit-exact vs baseline |
| trfeed bm256 (larger-M) | 126.5 | 41% | 0.88× | PASS — **NO-GO, regresses** |

- **One lever, 2.04× throughput** (68.7 → 140.3 TF). The wide feed alone lands the kernel
  level with hipBLASLt (0.98×), bit-identical output to the byte-gather path (max|Δ|=0).
- **L2-reuse verdict:** the direct-from-global B feed holds — throughput more than doubled
  rather than walling at baseline, so L2 absorbs the B re-fetch at the 128×128 tile.
- **Larger-M fallback (bm256) is a measured NO-GO:** 126.5 TF < 140.3. `FRAGS_M` is 4 in both
  configs (per-wave B reuse unchanged); bm256 only doubles waves/block + `As` LDS, dropping
  occupancy under the 128-VGPR `acc[4][4]` pressure. This confirms the remaining 140→307 gap is
  **not** B re-fetch — it belongs to the later double-buffer (Phase 2) and big-tile/occupancy
  (Phase 3) levers of the CDNA4 ladder.

**Reproduce:**
```
./build.sh
./out/gemm_trfeed_bench   # 256³ oracle gate (baseline/trfeed128/trfeed256 PASS) + 4096³ perf
```
