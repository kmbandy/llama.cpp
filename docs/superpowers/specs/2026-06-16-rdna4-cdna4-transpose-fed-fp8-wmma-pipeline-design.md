# Design: CDNA4→RDNA4 transpose-fed fp8 WMMA GEMM pipeline (MAD-305 / Phase 4)

**Date:** 2026-06-16 · **Hardware:** AMD R9700 / gfx1201 (RDNA4), wave32, ROCm 7.2.3
**Branch:** `sync/upstream-2026-06-09` · **Tree:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`
**Status:** approved design, ready for implementation plan.

## Goal

Take AMD's *proven* CDNA4 fp8 GEMM recipe — the optimization ladder that sustains ~80% of
matrix-core peak on MI300X/MI355 — and port it, lever by lever, onto the RDNA4 (gfx1201) WMMA
path, landing the result in the production `gemm_wmma.hip` kernel and the llama.cpp ml8
inference path. Target: climb the measured 90 TF real kernel toward **~245 TF (80% of the
proven 307 TF fp8 WMMA ceiling)**, beating hipBLASLt's 143 TF on the way.

## Why this, why now (the reframe)

Four phases of the prior campaign (MAD-300/304/305 de-risk) chased the **occupancy / dynamic-VGPR**
lever — which the CDNA4 ladder shows is a *late* step worth ~+96%. The *first and largest* lever
(CDNA step 4, "vectorized ingress / kill the byte-feed", **+1021%**) was never built. Our own
measurements independently confirm it is the wall:

- **307 TF** raw-WMMA microbench (`bench/wmma_peak.hip`) = 80% of the 383 marketing peak — the
  real, sustainable ceiling. It feeds operands from registers with zero hot-loop memory.
- **~69–90 TF** real LDS-tiled kernel (`gemm_wmma.hip`) — walled by a **per-byte B-fragment
  gather**: 8 strided `ds_load_u8` ahead of every WMMA (A reads wide; B gathers).
- **143 TF** hipBLASLt (47% of 307) — better than ours, still far from the ceiling.
- **MI300X (CDNA3) fp8 hipBLASLt = 1228/1515 TF = 81%** of peak on aligned shapes — proof the
  silicon class sustains ~80% with the right pipeline.

A feed-width prototype this session (`bench/feedwidth_proto.hip`, oracle-gated) proved the layout
math correct but showed **naive software transpose-store *loses*** (51.8 vs 68.8 TF, 0.75×) — the
transpose cost merely relocates. The lever requires a **hardware** transpose-load.

**Key instruction-availability findings (verified against the clang builtin table + ck_tile):**

- gfx1201 **has** `__builtin_amdgcn_global_load_tr_b64_v2i32` (the *global* transpose-load, fp8,
  wave32; `ck/utility/amd_transpose_load.hpp::amd_global_load_transpose_to_vgpr`).
- gfx1201 **lacks** `ds_read_tr8_b64` (the *LDS* transpose-read) — it is `gfx950`-only.
- ck_tile's high-perf pipelines (`gemm_pipeline_ag_bg_cr_comp_v3..v6`/`comp_async`, carrying the
  CDNA4 async-double-buffer + scheduler levers) have **no WMMA/gfx12 path**. The gfx12 fp8 WMMA
  *leaf* is first-class (`warp_gemm_dispatcher.hpp:155`) but rides only the simple
  `wp_pipeline_agmem_bgmem_creg` (no transpose feed, no double-buffer).

**Consequence:** there is no turnkey high-perf gfx12 fp8 GEMM to instantiate; AMD itself leaves
RDNA4 fp8 under-pipelined. Building the transpose-fed, double-buffered fp8 pipeline for gfx1201 is
genuinely unbuilt — that is both the engineering and the AMD-partnership pitch.

## Approach (vehicle = C: evolve the production kernel)

Evolve the existing `gemm_wmma.hip` **in place**, structurally **mirroring** ck_tile's `comp_v6`
pipeline (the algorithm) and AITER's CDNA `ck_gemm_a8w8_blockscale` tile config (the parameters),
while hand-coding the gfx12-specific `global_load_tr` feed that ck_tile never wires. Rationale:

- `gemm_wmma.hip` already unifies the **fp8** path and the **ml8 4-bit LUT** path behind the
  production C-API (`rdna4_gemm_fp8_forward` / `rdna4_gemm_ml8_forward`) — it *is* the llama.cpp
  target, so "fully production-integrated" is satisfied by construction.
- Full control over `global_load_tr` (ck_tile does not expose it for gfx12).
- ml8 rides along: its unpacked B feeds the same WMMA core.

We lift the *design* from CDNA4/ck_tile/AITER without taking a heavy template dependency.

Rejected alternatives: (A) hand-write from scratch — discards the production kernel + ml8 we
already have; (B) extend ck_tile's `comp_v6` for gfx12 — most upstreamable but heavy template
surgery, and routing ml8 + llama.cpp through ck_tile is over-scoped now.

## The phased ladder

The whole ladder is specced here at a high level; we implement **one phase at a time** and
**evaluate at each gate** before continuing. Our start already clears CDNA steps 2–3 (LDS-tiled +
matrix-core), so we pick up at the byte-feed kill.

### Phase 0 — `global_load_tr_b64` layout probe (de-risk; no TFLOPS)
- **Do:** standalone `bench/global_load_tr_probe.hip`. Push a known fp8 pattern through
  `amd_global_load_transpose_to_vgpr`; dump device asm (`--save-temps`/`llvm-objdump`) to confirm
  it lowers to `global_load_tr_b64`; empirically derive the per-lane mapping — what global address
  each lane passes, and how the returned `v2i32` maps to the §7.12 WMMA B-fragment (lane→col, K-byte
  layout). Compiler-as-oracle; **no guessing**.
- **Gate:** a verified lane→register mapping table, hand-checked against a reference, sufficient to
  address the instruction correctly in Phase 1.

### Phase 1 — Wide feed (CDNA step 4, the +1021% lever)
- **Do:** replace the B byte-gather in `gemm_wmma.hip`'s fp8 path with `global_load_tr_b64`.
- **Architectural divergence from CDNA (the crux):** gfx1201 has only the *global* transpose-load,
  not the LDS one — so we **cannot** replicate CDNA's "stage B in LDS, then `ds_read_tr8`." The
  gfx12-idiomatic path is `global_load_tr_b64` **direct from global B**, relying on L2 for B-reuse.
- **Gate:** fp8 oracle PASS **and** measure vs 307. The make-or-break question: *does L2 reuse hold,
  or does B re-fetch wall it?* Expect to clear 90 and challenge 143.
- **Risk fallback:** if B re-fetch dominates, enlarge the M-tile (amortize each B fragment over more
  A rows) before abandoning the direct path.

### Phase 2 — Async double-buffer (CDNA step 7, +135%)
- **Do:** software-pipeline the K-loop — prefetch next A (`global_load_lds`, direct global→LDS,
  async) and next B (transpose-load) while computing current; ping-pong buffers à la `comp_v6`.
- **Gate:** fp8 oracle PASS + measure (expect the largest single jump).
- **Risk:** verify `global_load_lds` async lowering on gfx12 early; fallback = non-async (compute
  overlaps a plain prefetch into a second buffer).

### Phase 3 — Big tiles + wave scheduler + wave32 occupancy retune (CDNA steps 8–9, +96%/+17%)
- **Do:** scale the tile toward CDNA's 256-equivalent (wave32-adjusted), more waves/block,
  `s_setprio`/`sched_barrier` wave ping-pong; retune occupancy against the VGPR/LDS footprint.
  Mirror AITER `ck_gemm_a8w8_blockscale` / `hsa/gfx942/fp8gemm_blockscale` tile parameters.
- **Gate:** measure vs the ~245 TF target (80% of 307).
- **Risk:** RDNA4 LDS capacity under wave32 big tiles + double-buffer; tile/occupancy retune is part
  of this phase.

### Phase 4 — ml8 4-bit LUT front-end on the optimized core
- **Do:** re-attach the existing ml8 unpack + LUT-dequant + per-K-group scale fold onto the evolved
  feed/pipeline.
- **Wrinkle:** ml8's B is packed nibbles + fp8 LUT, **not** fp8-in-global, so it **cannot** use
  `global_load_tr` directly. ml8 unpacks-to-fp8 first (LDS/registers) then feeds the *same*
  optimized WMMA core. Dense-fp8 and ml8 diverge at the B-load, converge at the matrix core.
- **Gate:** ml8 LUT oracle PASS + measure ml8 TF.

### Phase 5 — Production integration + PPL-neutral
- **Do:** wire the evolved kernel through the existing llama.cpp ml8 path (already routes via
  `rdna4_gemm_*_forward`); run graph-level correctness and a PPL gate on a real model.
- **Gate:** graph correctness + PPL within tolerance vs baseline + end-to-end prefill TF improvement.

## Components / files

| File | Action | Responsibility |
|---|---|---|
| `bench/global_load_tr_probe.hip` | new | Phase 0 layout probe (asm dump + lane→reg table) |
| `gemm_wmma.hip` | evolve | production kernel; Phases 1–4 modify B-feed + add pipelining; fp8 + ml8 behind existing C-API |
| `bench/gemm_ladder_bench.hip` | new | drives fp8 + ml8 kernels; fp8 e4m3 oracle + ml8 LUT oracle; per-phase TF table vs 307 / ×143 |
| `build.sh` | extend | RAM-capped targets for probe + ladder bench |
| `RESULT.md` | new (append/phase) | pitch artifact: per-lever TF table + oracle status |
| llama.cpp ml8 graph wiring | Phase 5 | route through existing helper; PPL harness invocation |

## Correctness gates (hard — pass before any TF is trusted)

- **fp8 e4m3 CPU oracle** (reuse): 256³ max-rel-err < 3% **and** 4096³ variant-vs-baseline
  bit-agreement → every phase 1–3.
- **ml8 LUT oracle** (reuse, MAD-299) → Phase 4.
- **PPL-neutral** vs baseline on a real model → Phase 5.

A failing oracle invalidates the phase; the TF number is not reported until the oracle passes.

## Measurement methodology

- Shape: 4096³ compute-bound square (matches `wmma_peak.hip` + the cross-check).
- Timing: `hipEvent` with warmup + ≥30 iters.
- Report per phase: ms, TFLOPS, **% of 307**, **× hipBLASLt(143)**.
- Targets: credibility floor = **> 143** (beat hipBLASLt); headline = **~245 (80% of 307)**.

## Proven-recipe sources we mirror

- ck_tile `gemm_pipeline_ag_bg_cr_comp_v6` / `comp_async` — pipeline structure (double-buffer,
  scheduler) for Phases 2–3.
- AITER `csrc/ck_gemm_a8w8_blockscale` + `hsa/gfx942/fp8gemm_blockscale` — tile sizes / policy for
  Phase 3.
- `ck/utility/amd_transpose_load.hpp::amd_global_load_transpose_to_vgpr` — the Phase 1 feed
  primitive.

## Risks & mitigations

| Risk | Phase | Mitigation |
|---|---|---|
| Direct `global_load_tr` loses B-reuse (no LDS for B); L2 re-fetch walls it | 1 | the gate; fallback = larger M-tile to amortize B re-fetch |
| `global_load_lds` async unsupported / slow on gfx12 | 2 | verify lowering early; fallback = non-async double-buffer |
| ml8 B can't use `global_load_tr` (packed nibbles) | 4 | unpack-to-fp8-then-feed; shared WMMA core |
| wave32 big-tile LDS/occupancy pressure | 3 | tile + occupancy retune is part of the phase |

## Supervision & safety

All work is **normal HIP** (no raw PM4, no KFD queue submission) → **unsupervised, no GPU-hang
risk**, unlike the prior campaign's PM4 phases. Builds are RAM-capped (`systemd-run --user --scope
-p MemoryMax`) per standing constraint.

## Success criteria

1. Each phase passes its correctness gate and produces a measured TF datapoint.
2. The ladder beats hipBLASLt (143 TF) by Phase 1–2.
3. Headline: approach ~245 TF (80% of 307) by Phase 3.
4. ml8 path preserved and correct (Phase 4); production-integrated + PPL-neutral (Phase 5).
5. `RESULT.md` per-lever table is a clean AMD-pitch artifact.

## Out of scope

- Dynamic-VGPR occupancy (separate lever; revisit only if Phase 3 occupancy retune stalls).
- Upstreaming the pipeline into ck_tile itself (possible future contribution).
- Non-fp8/ml8 dtypes; non-square / heavily-ragged shapes beyond what llama.cpp needs.
