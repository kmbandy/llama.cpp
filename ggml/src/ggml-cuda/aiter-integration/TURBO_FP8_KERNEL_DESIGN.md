# MAD-214 Phase 1D: turbo-FP8 Triton/AITER kernel design

**Status:** locked. This is the spec the implementation in
`kernels/unified_attention.py` is built against. Mirrors the locked design in
the MAD-214 ticket; this file is the kernel-implementation-facing version.

**Calibration source for centroids:** `scripts/calibration/fit_centroids.py`
JSON → `scripts/calibration/export_centroids.py` → `turbo_fp8_centroids.h`.

---

## Kernel surface

Three variants, one shared kernel template, dispatched at AOT spec time via a
single `CACHE_TYPE` constexpr — mirrors the existing MAD-199 v2 pattern for
F16/TURBO3/TURBO4 (see `unified_attention.py` lines 117-155):

| `CACHE_TYPE` const | bpv | centroids | use case |
|---|---|---|---|
| `CACHE_TYPE_TURBO3_FP8` = 3 | 4.5 | 8 | q4_0-killer (same memory, better quality) |
| `CACHE_TYPE_TURBO4_FP8` = 4 | 5.5 | 16 | sweet spot (4.3× better than q4_0) |
| `CACHE_TYPE_TURBO5_FP8` = 5 | 6.5 | 32 | near-fp8_raw quality at 18% memory saving |

All three:
- `BLOCK_SIZE = 32` (calibrated; differs from QK_TURBO3/4 = 128 which are
  rotation-group sized — we carry Hadamard in the wrapper, not the kernel)
- `SCALE_DTYPE = fp16` (Phase 0 showed ~0% MSE delta vs FP32; hipfire FP32
  warning was about folding scale INTO FP8, which we don't do)
- Per-(kv, layer) centroid LUT (per-head granularity rejected in Phase 0:
  no quality improvement, simpler kernel)

## Cache type byte budgets (matches `ggml-common.h block_turbo{3,4,5}_fp8`)

```
BYTES_PER_TURBO3_FP8_BLOCK = 18  # 2 scale + 12 idx (3-bit × 32) + 4 sign
BYTES_PER_TURBO4_FP8_BLOCK = 22  # 2 scale + 16 idx (4-bit × 32) + 4 sign
BYTES_PER_TURBO5_FP8_BLOCK = 26  # 2 scale + 20 idx (5-bit × 32) + 4 sign
```

## Wrapper-layer responsibilities (NOT in Triton kernel)

1. **Hadamard rotation on K (only)** — applied at quant time (when KV cache is
   written). V is NOT rotated (Phase 0: K-only gives 12% MSE improvement vs
   no rotation; +V gives 15% — extra 3% not worth complexity of rotating
   output back to original space).
2. **Q rotation at inference** — `Q' = QH`. Required because attention math
   is `softmax(Q'·K'^T / sqrt(d)) · V = softmax(Q·K^T / sqrt(d)) · V`
   (H cancels in the inner product). Without rotating Q, attention scores
   would be wrong.
3. **Centroid LUT marshaling** — one `uint8_t[N_CENTROIDS]` per (kv, layer)
   passed to the kernel as a layer-indexed pointer table.

These are implemented in Phase 1E (shared `ggml-cuda/turbo_fp8_hadamard.cuh`
wrapper, used by both AITER and paged-tile paths).

---

## Kernel inner-loop design

### Hot path (each K-tile iteration of the outer attention loop)

```
1. Load BLOCK_M rows of packed K bytes for the current tile.
2. For each row's BLOCK_SIZE-element chunk:
   a. tl.load FP16 scale (2 bytes)
   b. tl.load packed indices (12/16/20 bytes per chunk)
   c. tl.load sign bits (4 bytes per chunk)
   d. Decode: for each element in chunk,
        - extract N-bit index (constexpr N)
        - look up E4M3 byte = layer_lut[index]
        - XOR with sign bit (high bit of byte)
   e. Output: (BLOCK_M × HEAD_DIM) of FP8 bytes (E4M3)
3. Q is already FP8 (quantized once per-token before the attention call —
   in the wrapper layer). Loaded as FP8 bytes.
4. tl.dot(Q_fp8, K_fp8, acc, ...) → FP32 accumulator
   → MUST codegen to v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 on gfx1201
5. Post-multiply by K-tile per-block scale (FP32 mul on the accumulator output)
6. Continue with softmax in FP32 (unchanged from upstream)
```

### Critical unknown: Triton `tl.dot` FP8 codegen on gfx1201

Whether Triton's autoscheduler emits `v_wmma_f32_16x16x16_fp8_fp8` for
`tl.dot(fp8_e4m3, fp8_e4m3, fp32)` on RDNA4 is not yet verified. **First
real kernel step: write a minimal `tl.dot` FP8 GEMM and inspect the emitted
assembly** (Triton's `triton_compile_options` exposes a way to dump amdgcn).

If `tl.dot` doesn't hit the FP8 WMMA instruction:
- Fall back to inline asm `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
  via Triton's `tl.inline_asm_elementwise` or an external @triton.jit helper.
- This is the canonical AMD intrinsic, verified to work on gfx1201 by the
  Phase 1D step 0 probe (`tests/wmma_rdna4_fp8_probe.cu`).

### P · V leg (after softmax)

Same recipe, but:
- P (softmax output) is FP32 in the existing kernel
- Need to requantize P to FP8 before the FP8 WMMA (one extra step that the
  existing FP16 path doesn't have)
- Decode V same way as K
- `tl.dot(P_fp8, V_fp8, out_acc, ...)` → FP32

The P→FP8 requantization is a per-tile rescale: find max(|P|) within the
softmax tile, divide, cast to FP8 E4M3, multiply scale back into the
accumulator output. This is the same pattern the lighttransport reference
uses (see audit doc Round 2 prior art).

## Two-level FP32 accumulation (F.A.-3 long-context drift fix)

The hot inner loop accumulates many FP8-WMMA partial sums into a single FP32
register per tile. At ~10^7 accumulations, FP32 lower bits get lost. F.A.-3
on Hopper observed 91% → 13% NIAH accuracy at 128K context without this fix.

**Implementation:** the per-tile accumulator periodically drains into a
running outer accumulator (every ~8 tiles or when the inner accumulator's
exponent approaches saturation). The drain operation is integrated with the
online-softmax max-rescale step that's already in the kernel.

**When to wire in:** Phase 1D wires the accumulator interface so this is a
2-3 line addition later. Initial implementation may have single-level acc;
multi-block correctness tests (Phase 2) will surface drift if present.

## File layout in the AITER integration tree

```
ggml/src/ggml-cuda/aiter-integration/
├── TURBO_FP8_KERNEL_DESIGN.md       # this file
├── turbo_fp8_centroids.h            # auto-generated, per-model
└── kernels/
    └── unified_attention.py         # MAD-199 v2 modified
                                     # adds: CACHE_TYPE_TURBO{3,4,5}_FP8,
                                     # decode helpers, FP8 path in main
                                     # attention kernels
```

The kernel work is additive on top of the existing MAD-199 v2 structure.
Per the mad-lab-aiter-upstream-compat principle: prefer constexpr branching
inside the existing kernel functions over forking new ones; mark all changes
in the "LOCAL ADDITIONS" header block; keep dequant helpers as
sibling `@triton.jit` functions.

## Phase order

1. **Decode helpers + LUT loading** — three `@triton.jit` functions that take
   packed bytes + LUT pointer, return per-element FP8 bytes. Constexpr-
   parameterized so the same template body handles 3-bit, 4-bit, 5-bit
   indices. Unit-testable in isolation.
2. **Q quantization helper** — wrapper-side. Takes FP32 Q tensor (already
   rotated by Hadamard at the wrapper), per-token rescales to E4M3 range,
   returns FP8 bytes + per-token scale tensor.
3. **`tl.dot` FP8 probe inside Triton** — does Triton emit FP8 WMMA on
   gfx1201? Verify via amdgcn assembly inspection BEFORE building full
   attention kernel on top.
4. **Q@K FP8 WMMA integration** — add the FP8 path to
   `kernel_unified_attention_2d` / `_3d` under `CACHE_TYPE ∈ {3,4,5}` branches.
5. **P FP8 requantization** — between softmax and P@V.
6. **P@V FP8 WMMA integration** — mirror of step 4.
7. **AOT spec blocks in CMakeLists.txt** — 3 spec blocks.
8. **Smoke test** — `mt_aiter_unified_attn` wrapper with `cache_type =
   MT_AITER_CACHE_TURBO4_FP8` produces coherent output on Qwen3.5-4B.

Steps 1-3 are "verify the foundation." Steps 4-6 are the bulk of the kernel
work. Steps 7-8 are wrap-up.

Phase 2 (PPL/NIAH validation) comes AFTER Phase 1H smoke passes, in a
separate task.
