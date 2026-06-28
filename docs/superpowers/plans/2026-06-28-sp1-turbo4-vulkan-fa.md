# SP1 — turbo4_0 KV on Vulkan contiguous FA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `turbo4_0` (4-bit PolarQuant + RHT) as a symmetric KV cache type on the `ggml-vulkan` contiguous flash-attention path, matching the CUDA implementation numerically.

**Architecture:** Three device pieces — a `GGML_OP_FWHT` Vulkan op (handles all Hadamard rotation: Q/K/V forward, output inverse), a rotation-free f32→turbo4_0 cpy/quantize shader, and a centroid×norm dequant in the FA shader. All rotation is graph-level; the FA and cpy kernels only encode/decode centroids. CUDA is the reference implementation to translate, not reuse (separate backend, GLSL vs CUDA C++).

**Tech Stack:** C++ (Vulkan API host code in `ggml-vulkan.cpp`), GLSL compute shaders → SPIR-V (`ggml/src/ggml-vulkan/vulkan-shaders/`), `test-backend-ops` for per-op correctness, `llama-perplexity` for end-to-end parity.

**Spec:** `docs/superpowers/specs/2026-06-28-sp1-turbo4-vulkan-fa-design.md`

## Global Constraints

- **Target hardware:** RADV Polaris/gfx803 (RX480), **wave64**. Every shader with subgroup ops MUST be subgroup-size-agnostic — use `gl_SubgroupSize`/explicit widths, never assume 32. (This is the GLSL analog of the gfx803 `WARP_SIZE` scatter bugs.)
- **Symmetric only:** turbo4_0 for both K and V. No asymmetric combos, no turbo2/turbo3 in SP1.
- **FA path coverage:** scalar (`flash_attn.comp`) is the gate; `flash_attn_cm1.comp` shares `flash_attn_dequant.glsl` so it comes along free. `flash_attn_cm2.comp` (own decode path) is a parity follow-on, **out of SP1**.
- **Numerical bar:** match CUDA turbo4_0, validated end-to-end (accumulated sub-tolerance drift over many layers/tokens is the known failure mode — judge by PPL, not per-op tolerance alone).
- **Reference constants:** `TURBO_CENTROIDS_4BIT[16]` (`ggml/src/ggml-cuda/turbo-quant.cuh:297`) and `TURBO_MID_4BIT[15]` (`:306`) are the single source of truth — port verbatim, do not re-derive.
- **Block format:** `block_turbo4_0` = 68 bytes = `ggml_half norm` + `ggml_half rnorm`(reserved, unused in 4-bit) + `uint8 qs[64]` (128 nibble-packed indices). `QK_TURBO4 = 128`. (`ggml/src/ggml-common.h:307-316`.)
- **Build:** configure with `-DGGML_VULKAN=ON`; shaders are generated at build time by `vulkan-shaders-gen`. After adding/editing a `.comp`/`.glsl`, a normal `cmake --build` regenerates SPIR-V.

---

## File Structure

**Create:**
- `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp` — Walsh-Hadamard op shader (forward + inverse via push-constant flag).
- `ggml/src/ggml-vulkan/vulkan-shaders/cpy_f32_turbo4_0.comp` — f32→turbo4_0 nearest-centroid quantize.
- `ggml/src/ggml-vulkan/vulkan-shaders/turbo_centroids.glsl` — shared `TURBO_CENTROIDS_4BIT`/`TURBO_MID_4BIT` const tables (included by the cpy + dequant shaders).

**Modify:**
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl` — add `FA_DEQUANT4_TURBO4_0` macro + SSBO view.
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl:6-14,29-51` — add `FA_TYPE_TURBO4_0` + `fa_block_elems`/`fa_block_r` cases.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — pipeline decls/creation, op dispatch (`ggml_vk_build_graph` ~14173), `ggml_backend_vk_device_supports_op` (~16570), FA k/v-type acceptance (~3583), cpy-to-quant wiring (~11116-11128, `pipeline_cpy_f32_quant[]` ~818), FA pipeline selection block-size (~3473).
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — register the two new shaders for SPIR-V generation if not auto-globbed.
- `tests/test-backend-ops.cpp` — add/confirm `FWHT` and turbo4_0 `FLASH_ATTN_EXT`/`CPY` cases.

---

## Task 1: `GGML_OP_FWHT` Vulkan op (forward + inverse)

The linchpin — used 4× (Q/K/V pre-rotate, output inverse). CUDA reference: `ggml/src/ggml-cuda/fwht.cu` (kernel `fwht_cuda<N>`, sizes 64/128/256/512, `scale` param) and `turbo_fwht_128`/`turbo_rotate_forward` in `ggml/src/ggml-cuda/turbo-quant.cuh:88,127`. Note the rotation is `signs1 → FWHT → signs2`; inverse reverses order and normalization.

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (pipeline decl + create; dispatch case in `ggml_vk_build_graph` ~14173; `supports_op` ~16570)
- Test: `tests/test-backend-ops.cpp` (FWHT case)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `ggml_vk_op_fwht(ctx, subctx, src, dst, inverse)` dispatched on `GGML_OP_FWHT`; a `vk_pipeline pipeline_fwht_f32` (or `[2]` for fwd/inverse). Push constant: `{ uint n_rows; uint d; float scale; uint inverse; }`.

- [ ] **Step 1: Add the FWHT test case to test-backend-ops**

In `tests/test-backend-ops.cpp`, confirm a `test_case` exists for `GGML_OP_FWHT` (search `FWHT`). If absent, add one mirroring the CUDA-covered shapes: rows ∈ {1, 32}, d = 128, both `inverse=false/true`. (The harness compares the Vulkan result against the CPU `ggml` reference automatically.)

- [ ] **Step 2: Run it to verify Vulkan FAILS / is unsupported**

Run: `cmake --build build -j && ./build/bin/test-backend-ops -b Vulkan0 -o FWHT`
Expected: FAIL or "not supported" for Vulkan0 (op not implemented yet). CPU/CUDA pass.

- [ ] **Step 3: Write `fwht.comp`**

128-wide normalized Fast Walsh-Hadamard over each row of `d=128`. Translate `turbo_fwht_128` (`turbo-quant.cuh:88`): in-place butterfly across stride `h = 1,2,4,…,64`, normalized by `1/sqrt(128)`. `inverse` push-constant selects the sign-step ordering per `turbo_rotate_forward`/its inverse. **Use a 128-lane shared-memory array, barrier between butterfly stages; do NOT assume subgroup width — index by `gl_LocalInvocationID` over a fixed 128 workgroup, not by subgroup.** One workgroup per row.

- [ ] **Step 4: Wire the pipeline + dispatch + supports_op**

In `ggml-vulkan.cpp`: declare `pipeline_fwht_f32`, create it (pattern of the cpy pipelines ~`:4880`, push-constant struct above, workgroup `{128,1,1}`); add `case GGML_OP_FWHT:` to `ggml_vk_build_graph` (~14173) calling a new `ggml_vk_fwht(...)`; add `case GGML_OP_FWHT: return true;` (with `d` divisible-by-supported-size + f32 guard) in `ggml_backend_vk_device_supports_op` (~16570).

- [ ] **Step 5: Run the test to verify it passes (incl. wave64)**

Run: `./build/bin/test-backend-ops -b Vulkan0 -o FWHT`
Expected: PASS for `inverse=false` and `inverse=true`, rows {1,32}, d=128. (Polaris is wave64 — this is the case that catches subgroup-width bugs.)

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp ggml/src/ggml-vulkan/ggml-vulkan.cpp tests/test-backend-ops.cpp
git commit -m "vulkan: implement GGML_OP_FWHT (forward+inverse) for turbo KV rotation"
```

---

## Task 2: f32→turbo4_0 cpy/quantize shader

Rotation-free nearest-centroid encode. CUDA reference: `quantize_f32_turbo4_0_block` (`turbo-quant.cuh:336`) + `turbo_nearest_centroid_4bit` (the `TURBO_MID_4BIT` ladder, `:306-330`).

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/turbo_centroids.glsl`, `ggml/src/ggml-vulkan/vulkan-shaders/cpy_f32_turbo4_0.comp`
- Modify: `ggml-vulkan.cpp` (`pipeline_cpy_f32_quant[GGML_TYPE_TURBO4_0]` create ~`:818,4880`; cpy dispatch dst-quant path ~`:11128`; `supports_op` CPY ~`:16811`)
- Test: `tests/test-backend-ops.cpp` (CPY f32→turbo4_0 case)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `pipeline_cpy_f32_quant[GGML_TYPE_TURBO4_0]` populated; `turbo_centroids.glsl` exporting `const float TURBO_CENTROIDS_4BIT[16]` and `const float TURBO_MID_4BIT[15]` (reused by Task 3).

- [ ] **Step 1: Port the constant tables**

Create `turbo_centroids.glsl` containing `const float TURBO_CENTROIDS_4BIT[16]` and `const float TURBO_MID_4BIT[15]`, values copied **verbatim** from `ggml/src/ggml-cuda/turbo-quant.cuh:297` and `:306`. No re-derivation.

- [ ] **Step 2: Add the CPY test case**

In `tests/test-backend-ops.cpp`, add a `CPY` case `f32 → GGML_TYPE_TURBO4_0`, shape a multiple of 128 (e.g. `{128, 8}`). Harness compares against the CPU `ggml_compute_forward_dup`/quantize reference for turbo4_0.

- [ ] **Step 3: Run it to verify Vulkan FAILS**

Run: `cmake --build build -j && ./build/bin/test-backend-ops -b Vulkan0 -o CPY`
Expected: the turbo4_0 case FAILs/unsupported on Vulkan0.

- [ ] **Step 4: Write `cpy_f32_turbo4_0.comp`**

`#include "turbo_centroids.glsl"`. One invocation per 128-element block: read 128 f32 (already-rotated upstream), for each value pick the nearest centroid index via the `TURBO_MID_4BIT` ladder (translate `turbo_nearest_centroid_4bit`), nibble-pack into `qs[64]`, write `norm` as the per-block scale (fp16) and leave `rnorm` zero. Match the 68-byte `block_turbo4_0` layout exactly.

- [ ] **Step 5: Register the pipeline + dispatch**

Create `pipeline_cpy_f32_quant[GGML_TYPE_TURBO4_0]` (~`:4880` pattern); ensure the dst-quantized cpy path (~`:11128`) selects it for `dst->type == GGML_TYPE_TURBO4_0`; add `GGML_TYPE_TURBO4_0` to the CPY branch of `supports_op` (~`:16811`).

- [ ] **Step 6: Run the test to verify it passes**

Run: `./build/bin/test-backend-ops -b Vulkan0 -o CPY`
Expected: f32→turbo4_0 case PASS (bit-exact indices, norm within fp16 tolerance vs CPU ref).

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/turbo_centroids.glsl ggml/src/ggml-vulkan/vulkan-shaders/cpy_f32_turbo4_0.comp ggml/src/ggml-vulkan/ggml-vulkan.cpp tests/test-backend-ops.cpp
git commit -m "vulkan: f32->turbo4_0 cpy/quantize shader"
```

---

## Task 3: turbo4_0 dequant on the contiguous FA path

CUDA reference: `turbo4_dequant_element` (`turbo-quant.cuh:348`) = `TURBO_CENTROIDS_4BIT[idx] * norm`.

**Files:**
- Modify: `flash_attn_dequant.glsl` (new SSBO view + `FA_DEQUANT4_TURBO4_0` macro), `flash_attn_base.glsl:6-14,29-51` (`FA_TYPE_TURBO4_0 43u`, `fa_block_elems`→128, `fa_block_r`→1), `ggml-vulkan.cpp` (FA k/v-type acceptance ~`:3583`, FA pipeline block-size table ~`:3473`, FA `supports_op` ~`:16709`)
- Test: `tests/test-backend-ops.cpp` (FLASH_ATTN_EXT turbo4_0/turbo4_0 case)

**Interfaces:**
- Consumes: `TURBO_CENTROIDS_4BIT` from `turbo_centroids.glsl` (Task 2); the cpy from Task 2 to build the reference cache.
- Produces: FA dispatch accepts `k_type == v_type == GGML_TYPE_TURBO4_0`.

- [ ] **Step 1: Add the FA test case**

In `tests/test-backend-ops.cpp`, add/confirm a `FLASH_ATTN_EXT` case with K and V type `GGML_TYPE_TURBO4_0`, `hs=64` (LFM2.5 head dim), small `kv`/`nb`. Reference is the CPU FA path.

- [ ] **Step 2: Run it to verify Vulkan FAILS**

Run: `cmake --build build -j && ./build/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT`
Expected: turbo4_0 case unsupported/FAIL on Vulkan0.

- [ ] **Step 3: Add the FaType + block-size cases**

In `flash_attn_base.glsl`: add `#define FA_TYPE_TURBO4_0 43u` (line ~14), a `case FA_TYPE_TURBO4_0: return uint(QK_TURBO4);` (=128) in `fa_block_elems` (~:35) and `return 1u;` in the `fa_block_r` switch (~:51).

- [ ] **Step 4: Add the dequant macro + SSBO view**

In `flash_attn_dequant.glsl`: add `#include "turbo_centroids.glsl"`, a `K_PACKED_TURBO4_0`/`V_PACKED_TURBO4_0` buffer view matching `block_turbo4_0`, and:

```glsl
#define FA_DEQUANT4_TURBO4_0(BUF) {                                              \
    uint b  = (a_offset + ib);                                                   \
    FLOAT_TYPE nm = FLOAT_TYPE(BUF.data[b].norm);                                \
    uint i0 = iqs;                                                               \
    FLOAT_TYPEV4 c = FLOAT_TYPEV4(                                               \
        TURBO_CENTROIDS_4BIT[(BUF.data[b].qs[(i0  )/2] >> (((i0  )%2)*4)) & 0xF],\
        TURBO_CENTROIDS_4BIT[(BUF.data[b].qs[(i0+1)/2] >> (((i0+1)%2)*4)) & 0xF],\
        TURBO_CENTROIDS_4BIT[(BUF.data[b].qs[(i0+2)/2] >> (((i0+2)%2)*4)) & 0xF],\
        TURBO_CENTROIDS_4BIT[(BUF.data[b].qs[(i0+3)/2] >> (((i0+3)%2)*4)) & 0xF]);\
    return c * nm;                                                               \
}
```
Wire it into the `dequantize4` type switch alongside the existing `FA_DEQUANT4_*` entries for both the K and V binding expansions.

- [ ] **Step 5: Host-side FA acceptance + block size**

In `ggml-vulkan.cpp`: add `GGML_TYPE_TURBO4_0` to the FA k/v-type allowed set (~`:3583`); add its `block_a_size` entry (68 bytes / 128 elems) in the FA block-size switch (~`:3473`); allow it in FA `supports_op` (~`:16709`) for the symmetric case.

- [ ] **Step 6: Run the test to verify it passes**

Run: `./build/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT`
Expected: turbo4_0/turbo4_0 case PASS vs CPU FA reference.

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl ggml/src/ggml-vulkan/ggml-vulkan.cpp tests/test-backend-ops.cpp
git commit -m "vulkan: turbo4_0 dequant on contiguous flash-attention path"
```

---

## Task 4: End-to-end enablement + CUDA parity validation

Make `--cache-type-k/-v turbo4` select turbo4_0 on Vulkan and prove parity.

**Files:**
- Modify: `ggml-vulkan.cpp` (any remaining `ggml_is_quantized`/KV-cache-type guards that exclude turbo4_0 from buffer alloc / `get_to_fp32`/`get_to_fp16` paths so the cache can be created on Vulkan).
- Test: manual `llama-perplexity` parity run (no unit test — this is the integration gate).

**Interfaces:**
- Consumes: Tasks 1–3 (FWHT op, cpy, dequant).
- Produces: a working turbo4_0 KV cache on Vulkan via `--cache-type-k turbo4 --cache-type-v turbo4 --flash-attn on`.

- [ ] **Step 1: Remove remaining type guards**

Build and launch a turbo4 server; fix any assertion/"unsupported type" that blocks KV-cache creation for `GGML_TYPE_TURBO4_0` on Vulkan (mirror how q4_0 KV is permitted). Re-run until it loads:

```bash
cmake --build build -j
CUDA_VISIBLE_DEVICES="" GGML_VK_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -m /home/kmbandy/models/LFM2.5-8B-A1B-UD-Q5_K_S.gguf -ngl 99 --flash-attn on \
  --cache-type-k turbo4 --cache-type-v turbo4 --cache-ram 0 --ctx-checkpoints 0 \
  -c 16384 --host 127.0.0.1 --port 8099 --no-warmup --no-mmap
```
Expected: loads, `/completion` returns coherent output.

- [ ] **Step 2: PPL parity vs CUDA (short + long ctx)**

On the RX480 (Vulkan) and a CUDA box, run identical perplexity at `-c 512` and `-c 16384` with `--cache-type-k turbo4 --cache-type-v turbo4 --flash-attn on` over the same corpus. Record both. Expected: Vulkan PPL within ~1% of CUDA turbo4_0 at both depths (drift must not widen with context).

```bash
CUDA_VISIBLE_DEVICES="" GGML_VK_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity \
  -m /home/kmbandy/models/LFM2.5-8B-A1B-UD-Q5_K_S.gguf -ngl 99 --flash-attn on \
  --cache-type-k turbo4 --cache-type-v turbo4 -c 16384 -f <corpus> --chunks 4
```

- [ ] **Step 3: Perf sanity on the 480**

`llama-bench` tg/pp with turbo4 vs f16 vs q4_0 (symmetric). Expected: turbo4 in the same ballpark as q4_0; note the FWHT op overhead. Record numbers.

- [ ] **Step 4: Commit results note**

```bash
git add docs/superpowers/plans/2026-06-28-sp1-turbo4-vulkan-fa.md
git commit -m "vulkan: enable turbo4 KV end-to-end; record CUDA PPL parity"
```

---

## Self-Review

**Spec coverage:** FWHT op → T1. Quantize → T2. Dequant + FaType + host acceptance → T3. Type registration + PPL/perf validation → T4. Wave64 constraint → Global + T1S5. Symmetric-only / no asymmetric → Global + T3S5. cm2 out-of-scope → Global. All spec sections covered.

**Placeholder scan:** No "TBD/handle edge cases" — each shader cites its exact CUDA reference function+line to translate, constants are pointer-to-source (single source of truth, intentionally not duplicated), the one literal code block (dequant macro) is complete.

**Type consistency:** `FA_TYPE_TURBO4_0 = 43u` matches `GGML_TYPE_TURBO4_0 = 43`. `block_turbo4_0` 68-byte layout used consistently in T2 (write) and T3 (read). `TURBO_CENTROIDS_4BIT`/`TURBO_MID_4BIT` defined once in T2, consumed in T3.

**Known implementation-time confirmations (not blockers):** exact inverse-FWHT sign-step ordering vs the CUDA `scale` param (T1S3); whether `test-backend-ops` already carries FWHT/turbo4_0 cases or they must be added (T1S1/T2S2/T3S1).
