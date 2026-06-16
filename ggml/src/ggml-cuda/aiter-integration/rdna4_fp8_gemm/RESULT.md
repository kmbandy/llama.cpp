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
| 2 | async double-buffer + 7 adjacent levers | 140 (no gain) | 46% | 0.98× | PASS — **wall = WMMA-issue-rate** |
| 3 | dyn-VGPR / WMMA-per-cycle (in progress) | _open_ | | | — |
| 4 | ml8 4-bit LUT front-end | _pending_ | | | |
| 5 | production integration + PPL-neutral | _pending_ | | | |

**rocprof ground truth (GRBM cycles, timer-independent):** `wmma_peak` ceiling = **15.9 WMMA/cycle**
(307 TF); the wide-feed GEMM = **7.35 WMMA/cycle** (142 TF) = **46% of the matrix unit's issue rate**.
On RDNA the WMMA runs on the SIMD VALU, so operand-feed/address/loop instructions steal ~54% of issue
slots. The lever to 300 is **raising WMMA/cycle = more reuse per operand load = bigger accumulator tile**
— and the big tiles (`acc[8][8]`≈512 VGPR) exceed the 256-VGPR static max, so they **require dyn-VGPR**
(armable on gfx1201 via raw PM4, MAD-304; cap 128→256 via `SQ_DYN_VGPR.BLOCK_SIZE=1`).

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

## Phase 2 — async double-buffer + 7 adjacent levers (the wall is WMMA-issue-rate)

Eight oracle-correct variants vs the 140 TF Phase-1 winner, all flat-to-negative:
`bk64` 141 (flat) · `wg24` 136 · `db` (A double-buffer) 134 · `rb` (register-block) 133 ·
`bm256` 127 · `wg44` (16 waves, 60 VGPR) 119 · `bslds` (B-in-LDS) 104 · `bk128` 71.
gfx1201 has **no** async / direct-to-LDS load (`global_load_lds` needs `vmem-to-lds-load-insts`,
absent; the `global_load_async_to_lds` family is `gfx1250`-only), so the CDNA double-buffer can't be
ported verbatim — but the software-pipelined form regressed anyway. **Conclusion (rocprof-confirmed):
the kernel is WMMA-issue-rate bound, not feed-latency or occupancy bound.** SQ busy ≈ 99.5%; the
matrix unit issues at 7.35/15.9 = 46% of peak.

## Phase 3 — dyn-VGPR / WMMA-per-cycle (open; the unified lever)

- **dyn-VGPR occupancy is a WEAK lever (+3–13%)**, measured on a raw-PM4 hand-asm vehicle
  (`spike/dvgpr_occ/`). The kernel is not occupancy-starved.
- **Timer validated:** the in-kernel `s_sendmsg REALTIME` counter = **exactly 100 MHz**
  (`./occ_dispatch --timercheck`, overhead-cancelled), so PM4 TFLOPS are real — the hand-asm
  vehicle genuinely runs ~5× slower than hipcc (`64 TF` @NACC=8 vs `wmma_peak` 307) and is the
  **wrong vehicle** to chase 300 (it's even slower than the real GEMM's 142).
- **The real lever (rocprof):** raise **WMMA/cycle** (7.35 → toward 15.9) = more reuse per operand
  load = **bigger accumulator tile**. `acc[6][4]`≈222 VGPR fits the 256-VGPR static-wave max (free
  to test in HIP); `acc[8][8]`≈512 exceeds it → **requires dyn-VGPR** (armable on gfx1201 via raw
  PM4 — MAD-304; cap 128→256 via `SQ_DYN_VGPR.BLOCK_SIZE=1`, a `sudo umr` write that reverts on
  idle). **So dyn-VGPR == the WMMA/cycle lever: it's the enabler for high-reuse tiles, not an
  occupancy play.**

**Next:** (1) free HIP preview — `acc[6][4]`/`acc[5][4]` `gemm_fp8_trfeed`, read WMMA/cycle vs the
7.35 baseline; rises → extrapolate `acc[8][8]`, flat → proves dyn-VGPR is required. (2) Big build —
`s_alloc_vgpr` surgery on the **compiled** `gemm_fp8_trfeed` `.s` + PM4 dispatch (arm
`COMPUTE_PGM_RSRC2` bit 6) + `BLOCK_SIZE=1`, measure WMMA/cycle on the big tile.

**Reproduce (Phase 3, raw PM4 — SUPERVISED):**
```
cd spike/dvgpr_occ && ./build.sh
timeout 60  ./occ_dispatch --timercheck   # REALTIME = 100 MHz (PM4 numbers are real)
timeout 260 ./occ_dispatch --combined     # unroll×ILP×feed×dyn; NACC=8→64TF, NACC=16→127TF (codegen-limited)
```
