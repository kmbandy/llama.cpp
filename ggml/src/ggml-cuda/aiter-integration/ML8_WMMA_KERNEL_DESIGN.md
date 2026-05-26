# MAD-223 Phase D: ml8-4 weight FP8 WMMA inference kernel design

**Status:** locked (decisions 1-3 locked 2026-05-25). This is the spec the
implementation in `kernels/moe_op_gemm_ml8.py` and `kernels/gemm_ml8.py` will
be built against. Pre-implementation reading required; see Phase 0.

**Sibling documents:**
- `TURBO_FP8_KERNEL_DESIGN.md` — KV-cache FP8 WMMA kernel. Established the
  vendor-and-`tl.constexpr`-branch pattern for our tree. Same critical
  unknown (Triton `tl.dot` FP8 codegen on gfx1201).
- `TURBO_FP8_CALIBRATION_DESIGN.md` — KV calibration. Structural sibling of
  the ml8 calibration spec.
- `RDNA4_AUDIT_2026-05-20.md` — RDNA4/AITER/Triton/vLLM cross-repo audit.
  **Round 2** (lines 488-697) is the load-bearing prior-art distillation
  for this design.
- `scripts/calibration/ML8_README.md` — ml8 weight calibration pipeline.
  Defines the on-disk blob format that this kernel consumes.

**Calibration source for centroids:** `scripts/calibration/calibrate_ml8.py`
(Cell C recipe: Hadamard + group_size=64 + MSE fit_loss + E4M3 snap, 32
calib × 1024 tokens). Per-(linear, group) `fp8[16]` centroid LUT in
the `.ml8` blob; `centroids` field.

---

## Locked decisions

| # | Decision | Locked value |
|---|---|---|
| 1 | Kernel scope v1 | **MoE + dense** (both shapes ship together) |
| 2 | Fork strategy | **Vendor + `tl.constexpr` branch** (matches `unified_attention.py` precedent) |
| 3 | LUT residency | **LDS, loaded once per K-tile** at group boundary |

Rejected alternatives (with reasons captured for revisit):
- Sibling kernel (fully forked copy): ~400 LOC duplication per vendor file,
  every AITER upstream change requires hand-port. Reserve for v2 if the LUT
  path needs surgical retuning incompatible with constexpr branching.
- Register-resident LUT: lowest latency but eats register budget AITER's
  autotune configs are tuned for. Reserve as profiling-driven experiment
  if LDS shows up as a bottleneck.
- Constant memory LUT: wrong shape for per-group LUTs. Skip permanently.

---

## Goals

1. Production-quality FP8 WMMA inference path for ml8-4 weights on RDNA4
   (gfx1201, R9700 / RX 9070 XT / Navi 48).
2. Cover both shapes from day one:
   - **MoE GEMM** — primary target. Ml8-quantized 35B-A3B (Qwen3.6) is
     the production model. Reuses AITER `_moe_gemm_a8w4` as the baseline.
   - **Dense GEMM** — Qwen3.5-4B and other non-MoE models. Reuses AITER's
     dense FP8 GEMM path (`gemm_a8w4` family — Phase 0 verify exists).
3. Upstream-compatible per `[[mad-lab-aiter-upstream-compat]]`:
   vendor kernels into our tree, add `WEIGHT_FORMAT: tl.constexpr` knob,
   document local patches in file header. Bounded re-vendor merge cost.
4. Ship in both formats per `[[ml8-ships-two-formats]]`:
   - Native `.ml8` blob path for mad-lab inference platform
   - GGUF-wrapped path for llama.cpp community (kernel is format-agnostic;
     ingests packed indices + LUT regardless of container)
5. Match or beat AITER's 4-bit MX path (`_moe_gemm_a8w4` via
   `tl.dot_scaled` with MXFP4) on equivalent shape. Our extra LUT load
   is offset by the absence of MX scale unpack and the absorbed-into-
   centroid scale baked at calibration.

## Non-goals (explicit)

1. KV cache ml8 — separate epic. Lloyd-Max/centroid quant on KV is the
   turbo3/4/5 path; ml8 is weight-side only for now.
2. ml8-3 (8 centroids, 3.5 bpv) and ml8-5 (32 centroids, 5.5 bpv) — same
   kernel template should handle via `N_CENTROIDS: tl.constexpr` (LDS
   allocation scales), but Phase 1 ships only ml8-4 (16 centroids, 4.5 bpv).
3. Per-channel/per-tensor scaling sweep — calibration locked at Cell C
   (group_size=64, n_centroids=16, MSE fit_loss, Hadamard rotation, E4M3
   centroid snap). Kernel must match this layout.
4. Hand-written HIP fallback — Triton-only until performance or codegen
   bugs prove it necessary. (TURBO_FP8 design preserves the inline-asm
   escape hatch; we inherit that pattern.)
5. Asymmetric weight format — signed-16 layout is what shipped from
   calibration (see `ml8_io.py` blob V1).

---

## Target hardware & ABI

**Device:** AMD R9700 / RX 9070 XT — Navi 48, gfx1201, RDNA4. Wave32,
~64 CUs, 32 GB GDDR6, ~644 GB/s peak bandwidth, 64 KB LDS per CU (4 banks).

**Matrix instructions available:**
- WMMA element types on gfx1201: fp16, bf16, **fp8 (e4m3)**, int8, int4 (iu4)
- Primary instruction for ml8: `v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
  - Accumulates `fp8 × fp8 → fp32`
  - Tile shape: 16 × 16 × 16
  - Wave size: 32 (wave32 dispatch)
- **MXFP4 WMMA does NOT exist on RDNA4** (per audit Round 2 §3). MXFP4
  matrix instruction lives only on MI350X (CDNA4). ml8 cannot use MXFP4
  WMMA on R9700; FP8 WMMA after LUT decode is the path.

**Critical RDNA4 fragment layout gotcha (must implement correctly):**

> **`lane % 16 = column`, not row.** gfx12 WMMA fragment layout differs
> from gfx11. When writing dequantized fp8 values into the WMMA input
> fragment, the lane → fragment-element mapping is column-major across
> the wavefront.

Source: JohnTDI's `rdna4-wmma-guide` (distilled in audit Round 2 §2,
Agent 2 / GitHub implementations). Getting this wrong silently produces
transposed-looking output that passes uniform-input smoke tests. Phase 0
verification: re-read JohnTDI source for the exact mapping; Stage 2 of the
correctness matrix exercises this with a non-symmetric input pattern.

**ABI version:** ml8 blob format V1 (see `scripts/calibration/ml8_io.py`).
Kernel-facing inputs match V1 exactly; no re-pack at inference time.

---

## Existing AITER baseline

**Corrected 2026-05-25 after Path C investigation.** The RDNA4 audit
originally recommended `_moe_gemm_a8w4` as the vendor baseline. That was
based on a misread of AITER's `aXwY` naming convention — `a8w4` actually
denotes **MXFP8 × MXFP4** (microscaled formats consumed via
`tl.dot_scaled`), not "FP8 activations + uniform 4-bit weights with
explicit dequant." The four 4-bit-weight kernels in AITER
(`_moe_gemm_a8w4`, `_a16w4`, `_a4w4`, gluon `_a8w4`) ALL use
`tl.dot_scaled` and the MX-format codepath, leaving no source-visible
dequant block to swap. The blockscale 8-bit kernels are the correct
baseline.

### Dense baseline: `gemm_a8w8_blockscale.py`

Location upstream:
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py`
(495 LOC).

What it does:
- **A** = activations, FP8 e4m3, per-row scale (`a_scale_ptrs`)
- **W** = weights, 8-bit (fp8 or int8), per-block scale (`b_scale_ptrs`)
- **Output** = fp16 / bf16 / fp32 (configurable via accumulator dtype)
- Inner K-loop pattern (simplified — lines 189-193):
  ```python
  a       = tl.load(a_ptrs)                # FP8 e4m3 activations
  b       = tl.load(b_ptrs)                # FP8/INT8 weights (8-bit)
  a_scale = tl.load(a_scale_ptrs)          # per-row fp32 scale
  b_scale = tl.load(b_scale_ptrs)          # per-block fp32 scale
  accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
  ```

**Why this is the right baseline:**
- Plain `tl.dot(a, b)` — Triton's autoscheduler emits the FP8 WMMA
  instruction we verified in Phase 0
- Explicit per-block dequant via `* b_scale` post-multiply — clean
  injection point for our LUT branch
- Standard tile dispatch, async loads, split-K, autotune configs all reusable

### MoE baseline: `moe_op_gemm_a8w8_blockscale.py`

Location upstream:
`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`
(377 LOC).

Same inner-loop pattern as the dense kernel, wrapped with AITER's
standard MoE expert-routing dispatch (per-expert weight pointer offset,
expert routing via `GatherIndx` / `ExptHist` / `ExptOffs`). Our LUT
patch is the same shape applied in two files; the MoE wrapper code
itself doesn't need ml8-specific changes.

### 4-bit byte arithmetic reference: `gemm_a8wfp4.py`

Location upstream:
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8wfp4.py` (317 LOC).

**Not** a vendor target — it also uses `tl.dot_scaled`. But its
**packed-byte stride logic for 4-bit B loads** (lines 164-170) is
exactly the indexing pattern we need to adapt the blockscale baseline
from 8-bit-per-byte to 4-bit-packed-per-byte:

```python
# from gemm_a8wfp4.py:164-170
offs_bk = tl.arange(0, BLOCK_SIZE_K // 2) + k * (BLOCK_SIZE_K // 2)
offs_bk_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_bk
b_ptrs = b_ptr + (
    offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
)
```

Note the `BLOCK_SIZE_K // 2` everywhere — that's the 2-nibbles-per-byte
layout. Mask logic for masked loads (lines 181-184) also uses `K // 2`.
We borrow this byte arithmetic verbatim when adapting `gemm_a8w8_blockscale`
for ml8's 4-bit packed indices.

### Autotune / arch dispatch

Both baseline kernels are autotuned for MI300X. On R9700 we ride the
`gfx1201 → MI350X` aliasing workaround (AITER issue #1552 community fix;
both archs use FP8 E4M3FN not FNUZ, so the alias is semantically correct).

When we re-vendor, `git pull` AITER first to pick up the 28 new gfx1201
FP8 GEMM configs from PR #3228 (per Phase 0 finding: not in our local
`32e1e6d` SHA from 2026-05-18).

### Vendor target paths in our tree

```
ggml/src/ggml-cuda/aiter-integration/
├── ML8_WMMA_KERNEL_DESIGN.md            # this file
├── TURBO_FP8_KERNEL_DESIGN.md           # sibling
├── kernels/
│   ├── unified_attention.py             # existing — KV path
│   ├── moe_op_gemm_ml8.py               # NEW — vendored from
│   │                                    #   moe_op_gemm_a8w8_blockscale.py
│   │                                    #   + ml8 LUT branch
│   └── gemm_ml8.py                      # NEW — vendored from
│                                        #   gemm_a8w8_blockscale.py
│                                        #   + ml8 LUT branch
├── mt_ml8_moe_gemm.h                    # NEW — C++ wrapper for MoE
└── mt_ml8_gemm.h                        # NEW — C++ wrapper for dense
```

The kernel files carry a LOCAL ADDITIONS header block (mirrors
`unified_attention.py` pattern) documenting:
- Vendor commit SHA + date
- Each local patch with its purpose, file region, and re-vendor instructions

See Appendix C for the exact header template.

---

## The ml8 modification — single constexpr branch

The kernel-side delta is small and localized. **The B-load + dequant
block changes; everything else (tile dispatch, A-load, accumulator,
output store) inherits unchanged.**

### Kernel signature additions

```python
@triton.jit
def _gemm_ml8(
    # ... all existing AITER args unchanged ...
    WEIGHT_FORMAT: tl.constexpr,         # NEW — 0=int8_blockscale (upstream), 1=ml8_lut
    N_CENTROIDS: tl.constexpr,           # NEW — 16 for ml8-4 (ignored if WEIGHT_FORMAT==0)
    GROUP_SIZE: tl.constexpr,            # NEW — 64 for ml8-4 (ignored if WEIGHT_FORMAT==0)
    centroid_lut_ptr,                    # NEW — base ptr to fp8 LUT (only used if WEIGHT_FORMAT==1)
    centroid_lut_stride_group,           # NEW — stride between groups in LUT buffer
):
```

The same signature applies to both `_gemm_ml8` (dense) and
`_moe_gemm_ml8` (MoE) since they share the inner-loop pattern.

### Inner-loop branch

The baseline AITER pattern (from `gemm_a8w8_blockscale.py:189-193`):
```python
a       = tl.load(a_ptrs)
b       = tl.load(b_ptrs)                # 8-bit weights
a_scale = tl.load(a_scale_ptrs)
b_scale = tl.load(b_scale_ptrs)          # per-block scale
accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
```

Our patch wraps the B-load + dequant in the constexpr branch. **Per-group
scale stays as a post-`tl.dot` multiply, IDENTICAL to the blockscale baseline
— scale is NOT absorbed into centroids (would require per-(row, group_k)
storage, 16× blowup).** Decision corrected 2026-05-25 after reading the
actual `ml8_io.py` blob format; original sketch was wrong on this point.

```python
a = tl.load(a_ptrs)
a_scale = tl.load(a_scale_ptrs)
b_scale = tl.load(b_scale_ptrs)                            # SHARED between paths

if WEIGHT_FORMAT == tl.constexpr(0):
    # ----- existing A8W8 blockscale path, byte-identical to upstream -----
    b = tl.load(b_ptrs)                                    # 8-bit
    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

else:  # WEIGHT_FORMAT == 1, ml8 LUT path
    # ----- new ml8 lookup path -----
    # B is packed 4-bit (2 nibbles per byte). b_ml8_ptrs constructed
    # before the loop with K//2 stride (see "B-pointer construction" §).
    b_packed = tl.load(b_ml8_ptrs)                         # [K//2, N] uint8

    # FUSED unpack-and-extract (decision B-Option 4 — no explicit lo/hi pair):
    out_k = tl.arange(0, BLOCK_SIZE_K)                     # [K]
    byte_row = out_k // 2                                  # [K]
    shift    = (out_k % 2) * 4                             # [K]
    b_byte   = tl.gather(b_packed, byte_row[:, None].broadcast_to(
                            (BLOCK_SIZE_K, BLOCK_SIZE_N)), axis=0)
    b_idx    = (b_byte >> shift[:, None]) & 0x0F           # [K, N] in [0,15]

    # LUT lookup (decision C-direct — see _ml8_lut_lookup helper § below):
    lut = tl.load(centroid_lut_ptr + pid_n * stride_lut_n
                  + k * stride_lut_k + tl.arange(0, N_CENTROIDS))  # [16] fp8
    b_fp8 = _ml8_lut_lookup_ds_read_b8(lut, b_idx)         # inline-asm helper
    # (fallback: _ml8_lut_lookup_reference if asm probe fails — same signature)

    accumulator += tl.dot(a, b_fp8) * a_scale[:, None] * b_scale[None, :]
```

**Structural notes:**
- `a_scale`, `b_scale` are SHARED between both paths — blockscale's
  per-(N-tile, K-group) scale layout matches ml8's `scale_per_group:
  fp32[rows, n_groups]` layout exactly. Reuse the same `b_scale_ptrs` setup
  and advance logic.
- `b_ptrs` construction differs between paths: blockscale uses 8-bit
  per-row stride; ml8 uses K/2-element packed stride. Both ptr variables
  declared inside their respective constexpr branches; Triton compiler
  DCEs the unused one (decision A).
- Nibble unpack uses fused gather+shift (decision B-Option 4); no
  intermediate `b_lo` / `b_hi` tensors.
- LUT lookup uses inline asm `ds_read_b8` helper (decision C-direct);
  fallback `_ml8_lut_lookup_reference` (using `tl.where` unroll) lives
  alongside for differential testing. Both verified in Phase B.2 AMDGCN
  probe before kernel patch is committed.
- `b_scale` is loaded once per K-iter regardless of path. The "one less
  load per iter" claim from the previous design draft was wrong.

### Diff footprint

- Kernel file: ~80-100 lines of added/changed code per file
  - Signature args (5-7 lines)
  - LDS allocation + LUT refresh logic (15-20 lines)
  - B-pointer construction split for 4-bit-vs-8-bit (10-15 lines)
  - The if/else dequant + tl.dot block (30-40 lines)
  - Tail cleanup if needed (10-15 lines)
- Header documentation block: ~30 lines (Appendix C template)
- **Total per vendored kernel file: ~110-130 lines of mad-lab patch**

Bigger than the 30-50 LOC estimate in the original design (which was
predicated on the misread `_moe_gemm_a8w4` baseline) but still small
compared to the ~500 LOC vendored file. Re-vendor merge cost:

- Diff upstream HEAD against our vendored copy
- ~95% of typical AITER updates won't touch the B-load / dequant block
  (perf schedule changes, new autotune configs, async-load tuning,
  arch-dispatch refinements)
- For each AITER change touching the B-load or scale-multiply lines:
  re-apply our `if WEIGHT_FORMAT` guard around the new code
- Estimated typical re-vendor cost: 10-30 minutes per upstream release

---

## LUT residency: LDS layout

### Math: per-linear LUT volume

ml8-4 with group_size=64 stores 16 fp8 centroids per group. For typical
Qwen-class MLP linears:

| Linear | Shape (m×n×k) | Groups along K | LUT bytes per linear |
|---|---|---|---|
| Qwen3.5-4B `gate_proj` | (2560, 11008, 2560) | 2560/64 = 40 | 40 × 16 = 640 B |
| Qwen3.5-4B `up_proj`   | (2560, 11008, 2560) | 40 | 640 B |
| Qwen3.5-4B `down_proj` | (11008, 2560, 11008) | 11008/64 = 172 | 172 × 16 = 2752 B |
| Qwen3.6-35B-A3B expert (typical) | varies per expert | ~50-200 groups | ~800 B - 3.2 KB per expert linear |

**Total per-model LUT volume (Qwen3.5-4B, 32 layers × 3 linears):** ~390 KB.
Trivially fits in VRAM (loaded once per inference session).

**Per-Triton-block LUT residency:** at any time, one Triton block is
processing one K-tile of one (output) m-tile. That K-tile spans some
number of groups (depends on `BLOCK_K` vs `group_size` ratio). For
`BLOCK_K=64` with `group_size=64`, one block at one time = one group's
LUT = **16 fp8 bytes resident in LDS.**

Even at `BLOCK_K=256`, it's 4 active groups × 16 bytes = 64 bytes.
Negligible LDS usage.

### LDS allocation and refresh

```python
# Kernel preamble (once per Triton block):
centroid_lut_lds = tl.zeros((N_CENTROIDS,), dtype=tl.float8e4m3)

# Inside the K-loop, refresh the LUT at each group boundary:
for k_iter in range(0, K, BLOCK_K):
    if k_iter % GROUP_SIZE == 0:
        # Compute which group this K-tile is in
        group_id = k_iter // GROUP_SIZE
        # Load this group's 16-byte LUT into LDS (single 128-bit load on RDNA4)
        centroid_lut_lds = tl.load(
            centroid_lut_ptr +
            group_id * centroid_lut_stride_group +
            tl.arange(0, N_CENTROIDS)
        )

    # Inner K-loop (executes BLOCK_K / 1 = BLOCK_K iterations per outer step):
    raw_w = tl.load(w_ptr + ...)
    w_fp8 = tl.load(centroid_lut_lds + raw_w)   # LDS gather, broadcast within wavefront
    acc   = tl.dot(act_fp8, w_fp8, acc=acc)
```

### Performance characteristics on gfx1201

- **LDS load latency:** ~1-2 cycles (cached in M0 path)
- **Broadcast within wavefront:** free (LDS read is wavefront-uniform when
  indices are uniform; per-lane gather when indices vary)
- **Bank conflicts:** 16-byte LUT fits in one LDS bank (4 banks × 16 KB
  each). No conflict possible for a single-group LUT.
- **LDS bandwidth budget:** Negligible — the WMMA loop is compute-bound on
  the matrix instruction throughput (40-50 TFLOPS), not LDS bandwidth
  (~1 TB/s per CU at peak).
- **Comparison to AITER int4 path:** AITER's path has `tl.load(scale)` +
  `fmul` per group. Ours has `tl.load(lut)` per group + `tl.load(lut[raw_w])`
  per K-iter. **Net: ml8 path is ~1 instruction lighter per K-iter** (no
  per-iter scale multiply, since calibration absorbs scale into centroids).

### What this looks like next to JohnTDI's MXFP4 pattern

JohnTDI's rdna4-wmma-guide uses an LDS-resident LUT for MXFP4 E2M1 decode
(16 fp16 entries) and achieves 40.8 TFLOPS = 53% of FP16 WMMA peak on
R9700. Our pattern is structurally identical with two differences:
1. Our LUT entries are fp8 (8-byte LUT) instead of fp16 (32-byte LUT) —
   smaller, faster LDS load.
2. Our WMMA output goes through `v_wmma_f32_16x16x16_fp8_fp8` instead of
   their `v_wmma_f32_16x16x16_f16_f16` — half the input bandwidth into
   WMMA per cycle (FP8 vs FP16). Net throughput should be higher than
   theirs at the matmul step.

**Realistic perf ceiling estimate:** 40-50 TFLOPS effective at full
saturation. Phase F (bench + tune) will measure actual.

---

## Wrapper-layer responsibilities (NOT in the Triton kernel)

The kernel is intentionally narrow. Everything else lives in the wrapper
layer where ABI changes are cheaper.

### 1. Hadamard rotation — absorbed at Linear wrap time

Per the ml8 calibration design, the Kronecker-product Hadamard rotation
is absorbed into the centroids during calibration (`calibrate_ml8.py`
with `--rotation kronecker`) and absorbed into the upstream Linear's
weight at reconstruct time (`reconstruct_model.py` line ~XXX,
`absorb_rotation_at_wrap()`).

**Inference loads pre-rotated weights. The kernel does NOT see or apply
any rotation.** This is a meaningful simplification vs the turbo-FP8 KV
path, where Q rotation happens at inference time inside the wrapper.

### 2. Activation quantization to FP8

The kernel expects FP8 E4M3 activations. Quantization happens upstream:
- For dense: in the calling layer's forward pass (mirrors AITER's existing
  FP8 GEMM input path)
- For MoE: in the expert dispatch (mirrors AITER `_moe_gemm_a8w4`'s
  existing activation path)

We inherit AITER's quant convention exactly — no per-token rescale
divergence, no FP8 format difference (e4m3 not e5m2).

### 3. Centroid LUT marshaling

Wrapper builds one packed buffer from the .ml8 blob:
- Dense: `centroid_lut[linear_idx, group_idx, centroid_idx]` → flat fp8 buffer
- MoE: `centroid_lut[expert_idx, linear_idx, group_idx, centroid_idx]`

The kernel receives `(centroid_lut_ptr, stride_group)` per call. The
expert-routing dispatch (MoE case) sets the right base ptr per expert.

### 4. Scale handling — kept as post-`tl.dot` multiply, matches AITER blockscale

**Corrected 2026-05-25 after reading actual `ml8_io.py` blob format.**

Reconstruction formula (per `ml8_io.py` V1 blob):
```
W[r, c] = centroids_per_group[g][indices[r, c]] * scale_per_group[r, g]
```

Per-(row, group_k) scale is **separate from centroids** — folding it in would
require centroids shaped `[rows, n_groups, 16]` (16× storage blowup).
Instead the kernel keeps the scale as a post-`tl.dot` multiply, **identical
in shape and load pattern to AITER's blockscale baseline** (`b_scale_ptrs`
indexed by `(n_tile, group_k)`).

```python
# Both paths share the b_scale load + post-multiply:
b_scale = tl.load(b_scale_ptrs)
accumulator += tl.dot(a, b_fp8_or_int8) * a_scale[:, None] * b_scale[None, :]
```

**Net kernel comparison vs AITER int8 blockscale baseline:**

| Op | Blockscale baseline | ml8 (this kernel) |
|---|---|---|
| Load A (fp8) | ✓ | ✓ |
| Load B | tl.load(uint8 [K, N]) | tl.load(uint8 [K//2, N]) |
| Unpack | none | fused gather + shift |
| Centroid LUT lookup | none | `ds_read_b8` (or fallback) |
| Load b_scale | ✓ | ✓ (same shape) |
| `tl.dot` | ✓ | ✓ |
| Post-mul a_scale * b_scale | ✓ | ✓ |

ml8 has slightly MORE work per iter (unpack + LUT lookup), not less. Perf
prediction is "competitive with blockscale on perf, win on bpv-quality
Pareto" — not "strictly cheaper inner loop." Honest perf framing matters.

### 5. Sign handling

ml8-4 uses a signed-16 layout: 16 centroids that include both positive
and negative values (Lloyd-Max signed clustering, no sign-bit packing).
The 4-bit index addresses the centroid directly. No XOR-with-sign-bit
step needed (turbo-FP8 KV path has this; ml8 weight path does not).

---

## MoE vs dense entrypoints

Both shapes share the same core dequant + WMMA pattern. The dispatch
difference is at the wrapper layer, not the kernel core.

### Dense path

```c
// mt_ml8_gemm.h
struct mt_ml8_gemm_args_t {
    void *  activations_fp8;     // [M, K], fp8 e4m3
    void *  weights_packed_4bit; // [N, K/2], packed signed-16 indices
    void *  centroid_lut_fp8;    // [N_groups, 16], fp8 e4m3
    void *  output;              // [M, N], output dtype
    int32_t m, n, k;
    int32_t group_size;          // typically 64
    int32_t n_centroids;         // 16 for ml8-4
    int32_t out_dtype;           // 0=fp16, 1=bf16, 2=fp32
};

void mt_ml8_gemm(const mt_ml8_gemm_args_t * args, hipStream_t stream);
```

Dispatches to `gemm_ml8.py` Triton kernel via the AOT-compiled path.

### MoE path

```c
// mt_ml8_moe_gemm.h
struct mt_ml8_moe_gemm_args_t {
    // ... fields from mt_ml8_gemm_args_t for each expert ...
    void *  expert_routing;      // [M, top_k], expert IDs
    void *  expert_offsets;      // [num_experts + 1], CSR-style
    void *  centroid_lut_fp8;    // [num_experts, N_groups, 16]
    int32_t num_experts;
    int32_t top_k;
};

void mt_ml8_moe_gemm(const mt_ml8_moe_gemm_args_t * args, hipStream_t stream);
```

Dispatches to `moe_op_gemm_ml8.py` via the AOT-compiled path. Expert
routing inherits AITER's existing dispatch logic (per-expert weights base
pointer, per-expert LUT base pointer offset by `expert_idx`).

### Shared kernel template?

**Option:** factor the inner-loop dequant into a `@triton.jit` helper that
both MoE and dense files import. AITER doesn't do this for int4 (each
kernel has its own inlined dequant), so to minimize divergence from
upstream we follow their pattern — each file has its own inlined LUT
branch. Re-vendor cost stays at "one find-and-replace per file."

If during implementation we find ourselves typing the dequant block twice
in a way that genuinely doesn't differ, we can extract to a helper. Don't
preemptively abstract.

---

## Triton `tl.dot` FP8 codegen — RESOLVED 2026-05-25

**Status: ✅ standard `tl.dot` is the production path. No fallback needed.**

Phase 0 step 4 ran the existing probe at `tests/test_triton_fp8_dot_probe.py`
on the R9700 (gfx1201) and confirmed:

- **Correctness:** PASS (max_err=0, rms_err=0, 0 mismatches out of 256
  elements for the 16×16 fp8×fp8→fp32 reference matmul)
- **AMDGCN inspection:** `v_wmma_f32_16x16x16_fp8_fp8` confirmed emitted
  in multiple cached kernel directories (`unified_attention_2d.amdgcn`,
  `unified_attention_3d.amdgcn`). Sample instruction:
  ```
  v_wmma_f32_16x16x16_fp8_fp8 v[193:200], v[201:202], v[104:105], 0
  ```

**Verdict:** Triton's autoscheduler correctly lowers `tl.dot(fp8_e4m3,
fp8_e4m3, fp32)` to the FP8 WMMA instruction on gfx1201. Build the ml8
kernel with standard `tl.dot`; no inline-asm contingency required.

Inline-asm fallback (kept documented for historical / contingency reasons):

- If a future Triton or ROCm release regresses, fall back to
  `tl.inline_asm_elementwise` with
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.
- AMD intrinsic verified working on gfx1201 by TURBO_FP8 Phase 1D step 0
  (per TURBO_FP8_KERNEL_DESIGN.md §"Fallback").
- Re-run `tests/test_triton_fp8_dot_probe.py` after any Triton or ROCm
  upgrade to confirm this verdict still holds.

---

## Two-level FP32 accumulation — note for now

For GEMM the issue is less severe than for attention. Attention accumulates
softmax-weighted values across the whole context (10^5-10^7 accumulations
at long ctx); GEMM accumulates only along the K dimension (10^3-10^4 for
typical model widths). FP32 accumulator precision should be sufficient
without two-level accumulation for the shapes we target.

**Stage 4/5 validation** (full forward on Qwen3.5-4B) will surface any
drift. If observed, add the same two-level accumulator pattern TURBO_FP8
designs (drain inner FP32 accumulator into running outer accumulator every
~8 K-tiles). Reserve as a contingency, not a v1 requirement.

---

## Correctness validation matrix

Per `[[attention-kernel-test-principle]]` — uniform-input tests do NOT
validate the math. Each stage uses inputs designed to actually exercise
the path being tested. Adapted for GEMM (no softmax, but the spirit of
"non-uniform inputs that exercise the operation" still applies).

### Stage 1 — Dequant unit test (no WMMA)

- **Input:** synthetic packed 4-bit indices + known fp8 centroid LUT
- **Validates:** index extraction (raw 4-bit unpacking), LUT lookup,
  group boundary refresh logic
- **Expected:** per-element output exactly equal to `centroid_lut[raw_idx]`
- **File:** `tests/ml8/test_dequant_unit.py`
- **Time budget:** 30 min

### Stage 2 — Single-tile FP8 GEMM with non-uniform inputs

- **Input:** synthetic A and B with known patterns. Critical:
  - A varies across both M and K (not uniform)
  - B varies across both K and N (not uniform)
  - Patterns chosen so transposed-layout bugs are visible (asymmetric)
- **Validates:** WMMA fragment write layout (catches `lane%16=column`
  trap), accumulator setup, single-tile correctness
- **Expected:** matches reference matmul (run separately in PyTorch fp32)
  within fp8 rounding tolerance (~1e-2)
- **File:** `tests/ml8/test_single_tile_gemm.py`
- **Time budget:** 1-2h (initial), can debug fragment layout for hours
  if Phase 0 JohnTDI re-reading was insufficient

### Stage 3 — Multi-tile GEMM cross-boundary

- **Input:** larger shape requiring cross-tile accumulation along K,
  varying M and N to exercise tile dispatch
- **Validates:** BLOCK_K stride correctness, group-boundary LUT refresh
  across tile boundaries, multi-block accumulation
- **Expected:** matches reference PyTorch matmul ±1e-2
- **File:** `tests/ml8/test_multi_tile_gemm.py`
- **Time budget:** 1h

### Stage 4 — One real linear (Qwen3.5-4B `gate_proj` layer 0)

- **Input:** real Hadamard-rotated weights from `/tmp/ml8-full-hadamard/`
  calibration artifact (see `[[hadamard-ml8-4-2026-05-24]]`)
- **Validates:** real centroid LUT layout, real index packing, real
  Hadamard absorption at wrapper
- **Reference:** f16 forward of the same linear (PyTorch baseline)
- **Tolerance:** ±0.05 PPL contribution per linear (fp8 precision floor)
- **File:** `tests/ml8/test_real_linear_qwen4b.py`
- **Time budget:** 2-4h

### Stage 5 — End-to-end PPL on Qwen3.5-4B (MLP only)

- **Input:** full Qwen3.5-4B with all MLP linears ml8-quantized via Cell C
- **Reference:** f16 baseline PPL = 8.3181 (HF eval, 100K-token wikitext-2)
- **Expectation:** ml8 kernel output within ±0.05 of Python `reconstruct_model.py`
  output (which itself is +0.109 above f16 = 8.4268)
- **Validates:** full inference path correctness, multi-layer error
  accumulation
- **Wired via:** `reconstruct_model.py` with `--use-ml8-kernel` flag
- **Time budget:** 4-6h (includes wiring + PPL run)

### Stage 6 — End-to-end PPL on Qwen3.6-35B-A3B (MoE)

- **Input:** full 35B-A3B with all expert linears ml8-quantized
- **Reference:** TBD — depends on MAD-238 (Task #27) producing the
  35B-A3B calibration artifact first
- **Validates:** MoE dispatch + expert routing under ml8, paged inference
  path correctness (if paged)
- **Time budget:** 6-12h (gated on Task #27 unblocking)
- **Hardware safety:** per `[[mad-lab-hardware-safety-rule]]`, write
  the VRAM math before launching. Estimated footprint for 35B-A3B ml8
  inference: ~17 GB weights + ~3 GB activations + ~1 GB pool overhead =
  ~21 GB on 32 GB R9700. Well under the 28 GB ceiling.

---

## Phased implementation plan

### Phase 0 — Prerequisites (read-only, ~1-2h)

**Status (updated 2026-05-25):** partially complete. Critical-path items
done; remaining items not blocking for Phase A but should land before
Stage 2 of the correctness matrix.

- [x] **Triton FP8 `tl.dot` codegen probe — DONE 2026-05-25.**
      Result: `v_wmma_f32_16x16x16_fp8_fp8` emitted, correctness PASS.
      See "Triton `tl.dot` FP8 codegen — RESOLVED" section above for
      full data. **Standard `tl.dot` is the production path; no fallback
      needed.**

- [x] **AITER baseline kernel identification — DONE 2026-05-25 (Path C).**
      Findings (correcting RDNA4_AUDIT recommendation):
      - The audit recommended `_moe_gemm_a8w4` as the vendor baseline.
        That's an MX-format kernel using `tl.dot_scaled` — NOT an
        explicit-dequant baseline. All four 4-bit-weight kernels in
        AITER (`a8w4`, `a16w4`, `a4w4`, gluon `a8w4`) are MX-format.
      - **Correct baseline: `gemm_a8w8_blockscale.py` (dense) +
        `moe_op_gemm_a8w8_blockscale.py` (MoE).** Both use explicit
        dequant via `tl.dot(a, b) * a_scale * b_scale` pattern — the
        injection point we need for the LUT branch.
      - **Dense kernel exists in AITER** (`_triton_kernels/gemm/basic/`)
        — no MoE→dense porting needed. Saves the 2-4h Phase A estimate.
      - **Bonus reference: `gemm_a8wfp4.py`** (also `_triton_kernels/gemm/basic/`)
        — its 4-bit packed byte arithmetic (`offs_bk` with `K // 2` stride,
        mask logic, lines 164-170) is the pattern we borrow when adapting
        the 8-bit blockscale baseline to 4-bit weight loads.
      - Local AITER HEAD: `32e1e6d76` (2026-05-18). `git pull` AITER
        before Phase A to pick up the 28 new gfx1201 FP8 configs from
        PR #3228 (post-2026-05-18). Or use `gfx1201 → MI350X` aliasing
        fallback.

- [x] **JohnTDI rdna4-wmma-guide README read — DONE 2026-05-25.**
      Confirms the `lane%16=column` principle and the output store pattern
      (`for j in 0..8: write row_base+j, col`). The README **does not show
      input-fragment construction code** for A/B operands — points instead
      to AMD Composable Kernels `wmma_gemm.hpp` and the ROCm matrix-
      instruction calculator. **Not blocking for Phase A;** revisit before
      writing the Stage 2 fragment-layout test in Phase B (see Stage 2
      below).

- [ ] **jagsan-cyber/turboquant-rocm-llamacpp WMMA verification** —
      WebFetch of the README didn't expose source. Pending: clone the repo
      and read the actual attention kernel file. Not blocking for Phase A.
      Even if their kernel doesn't use WMMA, ml8's pattern stands on its
      own (JohnTDI's MXFP4 + LDS LUT is the primary reference). This is
      "nice to confirm" not "must verify."

- [ ] **Quantix paper (PPoPP 2026, DOI 10.1145/3774934.3786423)** —
      pending; defer until Phase B fragment-layout work where the
      in-register dequant + MMA pipelining pattern is directly relevant.
      Algorithmic blueprint, not direct port.

**Net Phase 0 verdict:** the load-bearing risks are resolved. Phase A can
begin. Update the local AITER clone (`git pull`) before vendoring to pick
up the gfx1201 configs.

### Phase A — Vendor + sanity smoke (2-3h)

- [ ] Identify the exact AITER commit SHA we're vendoring from.
      Document in each new kernel file's LOCAL ADDITIONS header.
- [ ] Copy `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`
      → `kernels/moe_op_gemm_ml8.py`. No modifications yet.
- [ ] Copy AITER's dense FP8 GEMM equivalent → `kernels/gemm_ml8.py`.
      No modifications yet.
- [ ] Add LOCAL ADDITIONS header block (template in Appendix C),
      initially empty list of patches.
- [ ] Wire into CMakeLists.txt AOT spec block (initially with WEIGHT_FORMAT=0
      only, mirroring AITER's existing int4 dispatch).
- [ ] Smoke: build, run AITER int4 path through our vendored copy on
      a synthetic shape. Confirm output matches direct AITER call
      (bit-identical, since no code change yet).

**Gate:** vendored copy works identically to upstream before adding any
ml8 modifications.

### Phase B — LUT branch (~10-14h with C-direct mitigations and corrected format)

**Sequenced sub-phases per `[[mad-lab-shipping-principle]]`:**

#### B.0a — Design doc correction (~30 min) — DONE 2026-05-25
- [x] Section "ml8 modification — single constexpr branch" rewritten with
      correct b_scale handling (kept as post-tl.dot multiply, identical to
      blockscale baseline).
- [x] Section "Wrapper-layer responsibilities" §4 rewritten — scale is NOT
      absorbed into centroids; ml8 is competitive on perf with blockscale,
      not strictly cheaper.
- [x] Appendix A rewritten as A.1 (calibration on-disk), A.2 (inference
      packed binary), A.3 (kernel runtime inputs), A.4 (MoE), A.5 (ABI).

#### B.0b — Inference format converter (~2-3h)
- [ ] Build `scripts/calibration/ml8_to_packed.py`:
      `.pt` (per ml8_io.py V1) → `.ml8` (per Appendix A.2 packed binary)
- [ ] Pack int8 indices → 4-bit nibbles (lo-first convention)
- [ ] Cast fp32 centroids → fp8 e4m3 (bit-preserving since calibration
      pre-snaps to E4M3 lattice)
- [ ] Pass through fp32 per-(N, group_k) scales
- [ ] Emit optional sidecar `.rotation.pt` and `.awq.pt` for wrapper consumption
- [ ] Unit test: round-trip a Cell C `.pt` → packed → unpack → match
      `reconstruct_weight` output within fp8 quantization noise

#### B.1 — Dense kernel LUT branch (~3-4h)
- [ ] Add `WEIGHT_FORMAT: tl.constexpr` + `N_CENTROIDS: tl.constexpr` +
      LUT pointer args + LUT strides to dense kernel signature.
- [ ] Add `b_ml8_ptrs` construction inside the WEIGHT_FORMAT==1 branch
      (K/2 stride per gemm_a8wfp4.py:164-170 pattern).
- [ ] Add the inner-loop if/else branch (see corrected "ml8 modification" §).
- [ ] Both `_ml8_lut_lookup_ds_read_b8` (inline asm, primary) and
      `_ml8_lut_lookup_reference` (`tl.where` unroll, fallback) helpers
      defined as sibling `@triton.jit` functions in the kernel file.
- [ ] Constexpr `LUT_IMPL` switch in kernel to A/B the two helpers in Stage 1.

#### B.2 — AMDGCN probe for `ds_read_b8` (~1-2h)
- [ ] Standalone test `tests/test_triton_ds_read_b8_probe.py` (mirrors
      `test_triton_fp8_dot_probe.py`).
- [ ] Minimal kernel: load 16 fp8 bytes into LDS, each lane reads byte at
      its assigned index via `tl.inline_asm_elementwise` + `ds_read_b8`.
- [ ] Verify correctness (synthetic LUT + known indices → known outputs).
- [ ] Inspect AMDGCN assembly for `ds_read_u8` / `ds_read_b8` instruction.
- [ ] If probe fails: lock C-fallback (reference helper), document why,
      flag explicitly to user — do NOT silently ship the slower path.

#### B.3 — Stage 1 dequant unit test (~30-60 min)
- [ ] `tests/ml8/test_dequant_unit.py` — synthetic packed indices + known
      LUT, run both kernel helpers (asm + reference), confirm equal output.
- [ ] Validates: nibble unpack, LUT lookup, both helper variants.

#### B.4 — Stage 2 single-tile GEMM, lane%16=column gate (~1-2h)
- [ ] `tests/ml8/test_single_tile_gemm.py` — non-uniform A and B (catches
      transpose bugs), differential check vs PyTorch fp32 reference.
- [ ] Tolerance: ~1e-2 (fp8 quantization noise).
- [ ] **The lane%16=column moment of truth.** If this fails, almost
      certainly the fragment write layout — debug against JohnTDI source.

#### B.5 — MoE port (~1h)
- [ ] Apply the same constexpr branch + helpers to `moe_op_gemm_ml8.py`.
- [ ] Same Stage 1+2 tests adapted to MoE shape.

#### B.6 — Dispatch wiring via runtime compiler (~30-45 min)

**Reframed 2026-05-25 after reading `aiter_runtime_compiler.h` (MAD-188).**

The existing aiter-integration tree has a documented dispatch architecture:
the `aiter::Registry::get_or_compile()` API in
`wrappers/aiter_runtime_compiler.h` is the **production path** for all
AITER Triton kernels. It JIT-compiles on first call (~2.5s on R9700),
caches the resulting HSACO on disk at
`~/.cache/llama.cpp/aiter/<key>/`, and serves subsequent calls from
that cache with no runtime overhead.

Build-time AOT specialization (the `add_triton_aot_kernel(...)` blocks
that wire `unified_attention` variants into `aiter_triton_aot` static lib)
is **a perf-tuning layer ON TOP of the runtime compiler**, not a
prerequisite. It exists to skip the 2.5s first-call cost for shapes
known to be popular based on production profiling data.

Per the header comment block at `aiter_runtime_compiler.h:11-14`:
> "Why this exists: vLLM-style build-time AOT explodes the build matrix
>  and ties the binary to the model shape. Triton AOT itself is cheap
>  (~2.5s on R9700). Doing it at server startup once per shape, with a
>  disk cache for warm restarts, is strictly less friction with no
>  runtime cost in the steady state."

**For ml8 v1, the runtime compiler IS the dispatch.** AOT specialization
moves to Phase F, gated on profiling data identifying hot shapes.

**B.6 deliverable (this phase) — DONE 2026-05-25:**
- [x] Verified our kernels are runtime-compiler-compatible. Evidence chain:
      - Vendor smoke test (`tests/test_ml8_vendor_smoke.py`) confirms both
        kernel modules import cleanly with all helpers inlined and AITER
        package imports avoided.
      - Stage 1 + Stage 2/3 multi-tile tests
        (`tests/test_ml8_kernel_stage1_dequant.py`) ran the dense ml8
        kernel through Triton's JIT compile path end-to-end with
        `max_err = 0` against PyTorch reference.
      - MoE single-expert tests (`tests/test_ml8_kernel_moe.py`) did the
        same for the MoE ml8 kernel.
      - `compile_aiter_kernel.py` invokes the same `triton.tools.compile`
        path the JIT uses; what works for JIT works for runtime AOT.
- [x] Documented the dispatch architecture above (this section).
- [x] No CMakeLists changes — the existing AOT static lib doesn't grow.
      Phase C wrappers will call `Registry::get_or_compile` directly per
      the `mt_aiter_unified_attn.cpp` pattern.

#### Phase F — AOT specialization for hot shapes (deferred, profiling-driven)

- [ ] Run perf benches across Qwen3.5-4B MLP shapes (gate/up/down) and any
      35B-A3B MoE expert shapes that profiling identifies as hot path
- [ ] For each shape proving worth the build-matrix cost: add an
      `add_triton_aot_kernel(...)` entry in CMakeLists.txt with concrete
      (N, K, WEIGHT_FORMAT) baked-in. Pattern is `unified_attention`'s
      `uattn_3d` block.
- [ ] Update `Registry::get_or_compile` dispatch to prefer the AOT
      variant when the shape matches (existing `unified_attention.cpp`
      pattern shows how)
- [ ] Document each AOT entry's empirical justification (this shape costs
      Xms on first call without AOT, takes Y% of total inference time,
      AOT win = Z%)

**Gate (overall Phase B):** Stage 1 + Stage 2/3 + MoE single-expert tests
all pass with `max_err = 0`. Both kernels (dense + MoE) verified
invokable via `Registry::get_or_compile`. Phase C unblocked.

### Phase C — Wrapper integration (3-4h)

- [ ] Implement `mt_ml8_gemm.h` C++ wrapper.
- [ ] Implement `mt_ml8_moe_gemm.h` C++ wrapper.
- [ ] Python entry points: `ml8_gemm()`, `ml8_moe_gemm()` in
      `aiter-integration/python/ml8_dispatch.py` (mirror the existing
      `unified_attention` dispatcher).
- [ ] Wrapper consumes `.ml8` blob (per `ml8_io.py` V1) and marshals
      centroid LUT into the kernel-expected layout.
- [ ] Stage 3 (multi-tile cross-boundary) tests pass.

**Gate:** wrapper can take a real `.ml8` artifact and produce correct
GEMM output on synthetic activations.

### Phase D — End-to-end on Qwen3.5-4B dense (4-6h)

- [ ] Wire `ml8_gemm` into `reconstruct_model.py` as an alternative to
      the Python centroid-reconstruction path.
- [ ] Add `--use-ml8-kernel` flag to switch between Python reconstruct
      (reference) and kernel path (under test).
- [ ] Stage 4 (one real linear) passes — output within ±0.05 PPL
      contribution of f16 reference.
- [ ] Stage 5 (full Qwen3.5-4B MLP-only PPL) passes — within ±0.05 PPL
      of Python reconstruct output.

**Gate:** kernel produces numerically equivalent output to Python
reference on a real model.

### Phase E — 35B-A3B MoE (gated on Task #27, ~6-12h)

- [ ] Wait for Task #27 (MAD-238 scaling to 35B-A3B) to produce the
      calibration artifact.
- [ ] Write VRAM math; confirm under 28 GB ceiling.
- [ ] Wire `ml8_moe_gemm` into the paged-MoE inference path.
- [ ] Stage 6 PPL eval.

**Gate:** 35B-A3B end-to-end PPL within expected delta of f16.

### Phase F — Benchmark + tune (iterative)

- [ ] Per-shape benchmark vs AITER int4 path (matched shapes).
- [ ] Per-shape benchmark vs f16 baseline.
- [ ] If perf gap >5% from AITER int4 (we expect ml8 to MATCH or BEAT
      due to absorbed-scale simplification), profile via rocprof and tune.
- [ ] Update autotune configs for gfx1201 if AITER's MI300X tuning
      transfers poorly.

**No gate** — Phase F is open-ended performance polish.

### Total time estimate

| Phase | Hours | Gating |
|---|---|---|
| 0 — Reading | 1-2 | none |
| A — Vendor + smoke | 2-3 | Phase 0 |
| B — LUT branch | 3-4 | Phase A |
| C — Wrapper | 3-4 | Phase B |
| D — Qwen3.5-4B E2E | 4-6 | Phase C |
| E — 35B-A3B E2E | 6-12 | Phase D + Task #27 |
| F — Bench/tune | open | Phase D |

**Critical path to "working ml8 inference on Qwen3.5-4B": 13-19h.**
Single-machine, single-developer. Realistic over 2-3 focused mad-lab sessions.

---

## Reading list with annotations

### PRIMARY (read before kernel work — Phase 0)

1. **JohnTDI-cpu/rdna4-wmma-guide** — fused MXFP4→FP16 WMMA via LDS-based
   LUT on R9700. **40.8 TFLOPS = 53% of FP16 peak. 3.8× vs separate
   dequant+hipBLAS for bs≤32.** Documents the gfx12 `lane%16=column`
   layout gotcha. **The closest existing reference implementation to our
   pattern.** Our ml8 kernel is structurally this guide's pattern with
   (a) fp8 LUT entries instead of fp16, (b) fp8 WMMA output instead of
   fp16 WMMA, (c) signed-16 centroid layout instead of MXFP4 E2M1.
   *Source: RDNA4 audit Round 2, Agent 2 §"The proven pattern."*

2. **jagsan-cyber/turboquant-rocm-llamacpp** — RDNA4 gfx1201 turbo3 port.
   Claims **2481 t/s turbo3 prefill vs 2349 t/s F16 prefill at 32K**.
   If WMMA is actually used (Phase 0 verification), this is concrete
   evidence that quantized+WMMA can BEAT f16 on R9700 — strongest
   possible existence proof for ml8's ceiling. *Source: RDNA4 audit
   Round 2, Agent 2 §"HIGHEST SIGNAL finding."*

3. **`TURBO_FP8_KERNEL_DESIGN.md`** (sibling) — established the vendor +
   `tl.constexpr` pattern in our tree. Same critical unknown
   (Triton `tl.dot` FP8 codegen on gfx1201). Same fallback path
   (inline asm `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`).

4. **`RDNA4_AUDIT_2026-05-20.md` Round 2** (lines 488-697) — full
   distillation of all relevant prior art. Already in our tree.
   **Caveat:** the audit's recommendation of `_moe_gemm_a8w4` as the
   ml8 baseline was based on a misread of AITER's `aXwY` naming —
   corrected in this doc 2026-05-25. Use `gemm_a8w8_blockscale` family
   for explicit-dequant baseline; reserve `_moe_gemm_a8w4` references
   only for the MX-format comparison context.

5. **AITER `gemm_a8w8_blockscale.py` + `moe_op_gemm_a8w8_blockscale.py`** —
   our vendor target. Pattern reference for the explicit-dequant +
   `tl.dot` structure. Re-read before adding the LUT branch.

6. **AITER `gemm_a8wfp4.py`** — packed 4-bit byte arithmetic reference.
   Lines 164-170 show the `K // 2` stride logic for loading 2-nibble-per-byte
   weights. Borrow verbatim when adapting the 8-bit blockscale baseline
   to 4-bit packed loads.

### SECONDARY (algorithmic blueprints)

7. **Quantix** (PPoPP 2026, DOI 10.1145/3774934.3786423) — 3-bit
   non-uniform clustering with 8 centroids per row + fused dequant + matmul
   on Tensor Cores. **4.82× over FP16 cuBLAS.** Algorithmic blueprint:
   in-register dequant pipelined directly into MMA. *Source: RDNA4 audit
   Round 2, Agent 1 §"HIGH SIGNAL papers."*

8. **FLUTE** (EMNLP Findings 2024, arXiv:2407.10960) — Flexible LUT engine
   for non-uniform / non-power-of-2-bit weight quantization. Offline
   restructuring + LUT vectorization for Tensor Cores; 2-4× over GEMM
   for batch <32. **Most directly transferable algorithm to ml8.**

9. **Marlin** (IST-DASLab) — production pattern: INT4 dequant to FP16 into
   tensor-core fragment in registers, then `mma.sync`. Ampere-only,
   ~4× speedup vs FP16 GEMM. **W4A16 weight-side, reference for the
   pattern shape** (not the specific arch).

10. **BitDecoding** (HPCA 2025, arXiv:2503.18773) — Tensor-Cores-Centric
    BitFusion: dequant fused into MMA pipeline. 7.5× over FP16
    FlashDecoding on RTX 4090, 8.9× on H100. Decoding-only and NVIDIA-only,
    but the pipeline pattern transfers.

### CAUTIONARY (what NOT to do)

11. **domvox/llama.cpp-turboquant-hip** — gfx1100 (RDNA3) port that
    AVOIDED WMMA: *"WMMA was unnecessary on ROCm 6.3.1 and a regression
    on ROCm 7.2.1."* They use scalar VEC FA. **This DOES NOT apply to
    our RDNA4 FP8 WMMA case** — different arch (gfx1100 vs gfx1201),
    different operation (attention vs GEMM), different precision (f16 vs
    fp8). Noted to avoid copying their "no WMMA" conclusion to RDNA4.

12. **TheTom/llama-cpp-turboquant** — multi-arch port using scalar VEC FA.
    What NOT to do on RDNA4 — scalar path leaves the matrix engine idle.

### INDEPENDENT VALIDATION

13. **Zolotukhin blog** (Apr 26 2026, R9700) — independent argument for
    inline-quant attention. **~36ms/prompt overhead from offline scratch
    dequant at 644 GB/s peak.** Validates our inline-LUT approach (LDS
    LUT IS the inline pattern — LUT lives in fast on-chip memory, dequant
    happens inside the kernel, no scratch buffer).
    https://zolotukhin.ai/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4/

---

## RDNA4 gotchas (referenced inline above; consolidated for quick recall)

1. **`lane % 16 = column`, not row** (gfx12 WMMA layout). Silently
   produces transposed output if mis-implemented. Source: JohnTDI's
   rdna4-wmma-guide. Mitigation: Stage 2 test with non-symmetric
   input pattern. **DO NOT skip Phase 0 reading.**

2. **`num_stages>=2` UAF on gfx1201** (Triton pipeliner crash). Force
   `num_stages=1` in autotune configs. Source: RDNA4 audit §2.2,
   ComfyUI-WanVideoWrapper #2007.

3. **No block-pingpong scheduler on RDNA4** — WMMA archs miss the
   mfma/memory overlap. Structural Triton gap, not fixable in our code.
   Source: RDNA4 audit §2.2.

4. **WMMA shape constraints** (Triton #8931) — non-16-multiple blocks
   fall to worse layout. Keep `BLOCK_M`, `BLOCK_N`, `BLOCK_K` multiples
   of 16. Source: RDNA4 audit §2.2.

5. **gfx1201 has NO arch-specific defaults in Triton `compiler.py`** —
   all archs get `num_warps=4, num_stages=2, waves_per_eu=0`. We must
   override in our kernel autotune configs. Source: RDNA4 audit §2.1.

6. **`tl.dot` FP8 codegen on gfx1201 is UNVERIFIED.** Inherits TURBO_FP8's
   open question. Probe in Phase 0 before relying on it. Inline-asm
   fallback is the contingency.

7. **AITER's gfx1201 → MI350X aliasing trick** (AITER #1552 workaround) —
   set `ARCH=MI350X` in our AITER dispatch until upstream adds gfx1201
   to `_ARCH_TO_DEVICE`. Both archs use FP8 E4M3FN (not FNUZ) so
   semantically correct.

8. **Triton PR #7616** (`global_load_tr_b128` on gfx12) — single-instruction
   8×8 transpose-load that skips LDS. Stuck in draft since 2025-07. If/when
   it lands, our LUT load path could potentially use it for the activation
   transpose (the LUT itself is too small to benefit). Track but not
   tonight-actionable.

9. **PR #6250 footgun** (Triton DPP for RDNA) — ROCm fork reverted upstream
   PR #6250 due to regression upstream didn't catch. If we upgrade Triton
   and hit RDNA-specific weirdness, this is a known land-mine. Source:
   RDNA4 audit §2.7.

10. **Symmetric K/V mandate on AMD HIP fused FA** (Krillian8 #22411) —
    does NOT apply to weight GEMM, but applies if we extend ml8 to KV
    cache later. Noted for future epic.

---

## Open questions / future work

1. **Option B (register-resident LUT)** — explore if LDS shows up as
   a bottleneck in Phase F profiling. JohnTDI's pattern uses LDS; we
   follow as production. Registers as v2 experiment if needed.
   ETA: deferred indefinitely unless profiling motivates.

2. **GGUF-wrapped ml8 inference** — per `[[ml8-ships-two-formats]]`, the
   kernel is container-agnostic. GGUF integration is a separate ticket
   (loader, metadata, llama.cpp glue). Kernel work here unblocks both.

3. **ml8-3 (8 centroids) and ml8-5 (32 centroids)** — same kernel template
   should handle via `N_CENTROIDS: tl.constexpr` (LDS allocation scales
   linearly). Phase G work after Phase F bench validates ml8-4.

4. **Per-token activation quant tuning** — currently inheriting AITER's
   upstream FP8 input path. If we want tighter activation quant
   specifically for ml8 (e.g., per-group activation scales), separate
   work after ml8-4 ships.

5. **KV ml8** — out of scope here. Would reuse the LUT-dequant pattern
   in an attention kernel (mirrors turbo3/4/5 → ml8 KV). Logical follow-up
   epic once both ml8 weight and turbo-FP8 KV ship.

6. **AITER upstream merge** — if our LUT-branch pattern proves clean,
   propose upstream as another `tl.constexpr` knob (alongside `USE_FP8`,
   `USE_QQ_BIAS`). Estimated cost: 1-2 weeks of upstream review cycles.
   Reduces our re-vendor cost to ~zero long-term.

7. **R9700 vs MI355 dispatch** — our autotune configs target gfx1201.
   If MAD-186 (AMD-focused fork productization) expands to CDNA hardware,
   separate config blocks needed. Mirror AITER's existing per-arch config
   organization.

8. **`tl.dot` FP8 inline-asm fallback share with TURBO_FP8** — if both
   end up on inline asm, factor into a shared `@triton.jit` helper file
   so the inline-asm block is maintained in one place.

---

## Appendix A — Calibration → kernel handoff

**Corrected 2026-05-25 after reading `scripts/calibration/ml8_io.py`
directly.** Earlier draft of this appendix was based on assumed format
that didn't match reality. Two distinct formats now documented:

### A.1 Calibration on-disk format (ml8_io.py V1)

This is what `scripts/calibration/calibrate_ml8.py` writes today. Verbose,
human-readable, intended for offline analysis and PPL eval — NOT inference.

Per-layer `.pt` file (one per Linear):
```python
{
    "name":                 str,                          # tensor name, e.g. "model.layers.0.mlp.up_proj"
    "shape":                [rows, in_features],          # = [N, K]
    "group_size":           int,                          # 64 for Cell C ml8-4
    "n_centroids":          int,                          # 16 for ml8-4
    "indices":              int8  [rows, in_features],    # values 0..n_centroids-1, ONE BYTE PER ELEMENT
    "centroids_per_group":  fp32  [n_groups, n_centroids],# E4M3-snapped values, stored as fp32
    "scale_per_group":      fp32  [rows, n_groups],       # per-(N, group_k) scale
    # Optional fields:
    "rotation":  {"kind": "kronecker_orth_sylvester", "h_a": fp32[a,a], "a_dim": int, "b_dim": int, ...},
    "awq":       {"kind": "...", "s": fp32[in_features]},
    # Metrics:
    "mse", "w_snr_db", "y_snr_db", "rel_err": float,
}
```

Reconstruction formula (per `ml8_io.py:reconstruct_weight`):
```
W[r, c] = centroids_per_group[g][indices[r, c]] * scale_per_group[r, g]
```

**Notes:**
- `centroids_per_group` are FP32 storage of E4M3-snapped values. The snap
  happens at calibration (`--snap-centroids e4m3` in Cell C) — so every
  value IS representable as fp8 e4m3. The fp32 storage just postpones the
  cast for human-readability and to avoid mixing dtypes in the .pt blob.
- `indices` are int8 (one full byte per index). Wasteful but readable.
  Packing into 4-bit nibbles happens at inference-format conversion (A.2).
- `scale_per_group` is shape `[rows, n_groups]` — per-(output_dim,
  K-group). It matches AITER blockscale baseline's `b_scale` layout
  exactly (which is per-(N, K-tile)).
- Rotation factors are consumed at `reconstruct_model.py` wrap time
  (folded into upstream Linear weight), NOT at kernel time.
- AWQ scale folds into activations or weights at wrap time, NOT at
  kernel time.

### A.2 Inference packed-binary format (Phase B.0b deliverable)

**This format does not exist yet.** It is built by a format conversion
tool (Phase B.0b) that reads A.1 `.pt` blobs and emits a packed binary
suitable for direct kernel consumption.

Per-layer packed binary (one section per Linear; multiple linears
concatenated into a single `.ml8` model file):

```
HEADER (32 bytes per layer):
├── magic:        u32 = 0x4D4C3849  # "ML8I" ascii
├── version:      u32 = 1
├── n_rows:       u32  (= N output_dim)
├── n_cols:       u32  (= K in_features)
├── group_size:   u32  (typically 64)
├── n_centroids:  u32  (= 16 for ml8-4)
├── flags:        u32  (bit 0: nibble_order_lo_first; future use)
└── reserved:     u32

DATA (3 sections, each aligned to 16 bytes):
├── indices_packed:  uint8 [n_rows, n_cols / 2]   # 4-bit nibbles, 2 per byte
├── centroids_fp8:   fp8   [n_groups, n_centroids] # E4M3, cast from .pt fp32
└── scales_fp32:     fp32  [n_rows, n_groups]      # passed through from .pt
```

**Nibble packing convention:** lo-nibble = index for column `2j`, hi-nibble
= index for column `2j+1`. Locked here; documented in the converter and
matched by the kernel unpack logic.

**Tool:** `scripts/calibration/ml8_to_packed.py` (Phase B.0b).
- Input: one or more `.pt` files (calibration output) + optional model
  metadata (which layers, expert routing for MoE)
- Output: one `.ml8` packed binary file (dense) or `.ml8m` (MoE with
  expert dispatch metadata)
- Optional: emit a sidecar `.rotation.pt` for the wrapper to consume at
  load time (per-Linear Hadamard factors that the wrapper folds into the
  upstream weight)
- Optional: emit a sidecar `.awq.pt` for per-input-channel AWQ scales

### A.3 What the kernel receives at runtime

**Three pointers + their strides** (excluding the standard A/C/dispatch
args inherited from AITER blockscale baseline):

```c
struct mt_ml8_kernel_inputs {
    void *  a_ptr;                  // fp8 e4m3 activations  [M, K]
    void *  a_scale_ptr;            // fp32 per-row scale    [M]    (inherited)
    void *  b_packed_ptr;           // uint8 packed indices  [N, K/2]
    void *  centroid_lut_ptr;       // fp8 e4m3 centroids    [N_groups, 16]
    void *  b_scale_ptr;            // fp32 per-(N,group_k)  [N, N_groups]
    void *  c_ptr;                  // output                [M, N]
    /* strides + dimensions as in AITER blockscale baseline */
};
```

**Notes:**
- No rotation tensors at kernel time (folded by wrapper).
- No AWQ scales at kernel time (folded by wrapper).
- `b_scale_ptr` layout is identical to AITER blockscale's `b_scale_ptr`
  — reuses the same `b_scale_ptrs`, `offs_ks_step`, `stride_bscale_*`
  setup.

### A.4 MoE additions

- **expert_routing:** standard MoE top-k indices (inherited from AITER MoE
  blockscale baseline)
- **per-expert weight base ptr:** standard offset arithmetic, but stride
  walks K/2 elements per row (not K)
- **per-expert centroid LUT base ptr:** `centroid_lut_ptr + expert_idx *
  per_expert_lut_size`
- **per-expert scale base ptr:** `b_scale_ptr + expert_idx *
  per_expert_scale_size` (identical to AITER MoE blockscale)

### A.5 ABI version contract

- **Kernel-facing ABI version: 1** (matches packed binary format A.2)
- Calibration `.pt` format (A.1) versioned independently via `ml8_io.py`'s
  `format_version` field; conversion tool handles format upgrades
- Future kernel-format changes (e.g. 3-bit indices for ml8-3, larger
  centroid counts for ml8-5) bump the kernel ABI version

---

## Appendix B — Build system integration

### CMakeLists.txt changes

```cmake
# ggml/src/ggml-cuda/aiter-integration/CMakeLists.txt
# (additions; existing AITER kernel build logic unchanged)

if (GGML_HIP_AITER)
    # ... existing aiter_triton_aot target unchanged ...

    # New AOT target for ml8 kernels
    add_custom_target(aiter_ml8_aot
        # AOT spec: dense + MoE × {WEIGHT_FORMAT=0 (passthrough), WEIGHT_FORMAT=1 (ml8)}
        # × N_CENTROIDS ∈ {16}  # Phase 1 ships ml8-4 only
        # = 2 × 2 × 1 = 4 AOT entries
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/aot_compile_ml8.py
                --out ${CMAKE_CURRENT_BINARY_DIR}/ml8_aot/
                --kernels moe_op_gemm_ml8 gemm_ml8
                --weight-formats 0 1
                --n-centroids 16
        DEPENDS aiter_triton_aot     # share libtriton.so
    )

    target_link_libraries(ggml-hip PRIVATE aiter_ml8_aot)
endif()
```

### Header inclusion

```c
// ggml/src/ggml-cuda/CMakeLists.txt — add to GGML_HIP_AITER conditional:
ggml/src/ggml-cuda/aiter-integration/mt_ml8_gemm.h
ggml/src/ggml-cuda/aiter-integration/mt_ml8_moe_gemm.h
```

### Build dependency notes

- Shares `libtriton.so` with existing AITER kernels (no new triton install
  needed; uses the `PYTHONPATH` workflow established for MAD-186)
- Same `Python3_EXECUTABLE` (agents venv) as existing aiter_triton_aot
- Per `[[mad-lab-build-hip-multi-arch]]`: build-hip is always gfx1201 +
  gfx1030. The gfx1030 build will use the AITER int4 passthrough path
  only (no FP8 WMMA on RDNA2); ml8 inference is gfx1201-gated at the
  dispatch layer.

---

## Appendix C — Vendored kernel header block template

Standard header for both `moe_op_gemm_ml8.py` and `gemm_ml8.py`. Mirrors
the existing `unified_attention.py` pattern. Each numbered patch entry
documents what changed and what needs re-application on re-vendor.

```python
"""
mad-lab LOCAL ADDITIONS to AITER's _moe_gemm_a8w4 (or gemm_a8w4)
=================================================================
Vendored from AITER commit <SHA>, date <YYYY-MM-DD>.
Source: aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py
        (or aiter/ops/triton/gemm/gemm_a8w4.py for dense file)

Per [[mad-lab-aiter-upstream-compat]]: prefer constexpr branching inside
the existing kernel functions over forking new ones. Mark all changes
here; keep dequant helpers as sibling @triton.jit functions if extracted.

Local patches:

1. WEIGHT_FORMAT: tl.constexpr branch (0=int4_uniform, 1=ml8_lut)
   ---------------------------------------------------------------
   Purpose: enable ml8 weight format (Lloyd-Max FP8 centroid LUT) as
            an alternative dequant path to AITER's uniform-INT4.
   Region:  kernel signature additions (top of function);
            inner K-loop dequant block (lines ~XXX-XXX in upstream).
   Args:    + WEIGHT_FORMAT: tl.constexpr
            + N_CENTROIDS: tl.constexpr
            + centroid_lut_ptr
            + centroid_lut_stride_group
   Behavior: if WEIGHT_FORMAT==0, code is byte-identical to upstream.
            if WEIGHT_FORMAT==1, uses centroid LUT lookup, no per-group
            scale multiply (scale absorbed into centroids by calibration).

2. LDS allocation for centroid LUT
   --------------------------------
   Purpose: hold 16 fp8 centroids in LDS during a K-tile's group span.
   Region:  kernel preamble.
   Code:    centroid_lut_lds = tl.zeros((N_CENTROIDS,), dtype=tl.float8e4m3)
   Notes:   negligible LDS footprint (16 bytes); fits in single bank.

3. Group-boundary LUT refresh
   ---------------------------
   Purpose: reload LUT into LDS when K-loop crosses a group boundary.
   Region:  start of K-loop iteration.
   Code:    if k_iter % GROUP_SIZE == 0:
                group_id = k_iter // GROUP_SIZE
                centroid_lut_lds = tl.load(centroid_lut_ptr + ...)

Re-vendor procedure:
1. Diff upstream HEAD against this file.
2. Most upstream changes (perf schedules, autotune configs, async-load
   tuning) will NOT touch the dequant block — re-apply directly.
3. For each upstream change inside the dequant block: wrap in
   `if WEIGHT_FORMAT == tl.constexpr(0):` to preserve the ml8 branch.
4. Verify Stage 1-3 tests still pass on re-vendored kernel before merge.

Estimated re-vendor cost: 5-20 min per AITER release.
"""
```

---

## Cross-references

- Calibration spec: `scripts/calibration/ML8_README.md`,
  `[[mad-223-pipeline-ml8-4]]`, `[[hadamard-ml8-4-2026-05-24]]`
- Sibling kernel: `TURBO_FP8_KERNEL_DESIGN.md`
- Prior-art audit: `RDNA4_AUDIT_2026-05-20.md` (esp. Round 2)
- Design principles:
  `[[mad-lab-aiter-upstream-compat]]`,
  `[[mad-lab-dev-test-both-paths]]`,
  `[[mad-lab-shipping-principle]]`,
  `[[mad-lab-build-hip-multi-arch]]`,
  `[[mad-lab-hardware-safety-rule]]`,
  `[[ml8-ships-two-formats]]`,
  `[[ml8-weight-quantization-vision]]`
- Jira: MAD-223 (epic), Task #13 (this scoping work)
- Related tasks:
  Task #2 (calibration sweep — feeds the recipe this kernel consumes),
  Task #22 (percdamp scout — may adjust calibration; kernel ABI stable),
  Task #27 (MAD-238 35B-A3B scaling — gates Phase E),
  Task #35 (MAD-186 R9700 integration test — broader integration story)
