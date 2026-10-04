# SP1 — turbo4_0 KV cache on the Vulkan contiguous flash-attention path

- **Date:** 2026-06-28
- **Status:** Design — pending review
- **Owner:** kmbandy + Claude
- **Parent effort:** Port the custom KV stack (turbo4 + paged-attn + tiering + semantic) to `ggml-vulkan` so the RX480/gfx803 swarm can run on RADV instead of the unmaintained ROCm-6.4-gfx803 port.

## Context

Vulkan/RADV on the RX480 (Polaris/gfx803) was validated as a viable backend: plain f16 FA is numerically correct (PPL within 0.4% of CPU), symmetric quantized KV is stable at depth, and the 8×64k (512k) swarm topology fits VRAM-resident at 4-bit KV (q4_0 proxy: 7594/8181 MiB, GTT flat). The custom feature stack that makes the swarm work on CUDA/HIP does **not** exist in `ggml-vulkan` (separate backend: GLSL/SPIR-V, zero shared code with `ggml-cuda`).

This spec covers **SP1**, the first of four sub-projects:

| Sub-project | Scope | Effort |
|---|---|---|
| **SP1 (this doc)** | turbo4_0 KV on the *contiguous* Vulkan FA path | M |
| SP2 | Paged-attn ops (`PAGED_ATTN_MT`, `PAGED_KV_UPDATE_MT`) + `turbo4_64` | L |
| SP3 | Tiering + semantic (VRAM↔RAM↔SSD block movement) | S–M |
| SP4 | `GGML_VK_STRICT_VRAM` guard (refuse silent GTT spill) | S |

SP1 ships turbo4 quality+headroom on the contiguous path we already proved viable, and isolates the turbo4 GLSL math before the paged complexity of SP2.

## Why turbo4 (not q4_0)

q4_0 already works on Vulkan and fits the 512k topology, but uniform 4-bit degrades faster at long context. turbo4's Lloyd-Max optimal centroids **plus** the random Hadamard transform (RHT) keep accuracy deep into the cache — the whole point is long-range context. RHT is therefore **in scope** (it is what differentiates turbo4 from q4_0).

## Scope

**In:**
- `GGML_TYPE_TURBO4_0` (128-element block, 4-bit PolarQuant, 4.25 bpv) as a Vulkan KV cache type.
- **Symmetric only**: turbo4_0 for both K and V. No asymmetric combos.
- RHT (Hadamard rotation), full parity with the CUDA contiguous path.
- Target: RADV Polaris/gfx803 (wave64), scalar FA path.

**Out (explicitly deferred):**
- `turbo4_64`, paged attention → **SP2**.
- Tiering / semantic → **SP3**.
- `turbo2_0` / `turbo3_0`, asymmetric combos (turbo4/q8_0, …).
- Inner-quant calibration (`innerq`) — env-gated, off by default; not required.

**Coverage decision (open):** the dequant macro is shared by `flash_attn.comp` (scalar) and `flash_attn_cm1.comp` (coopmat1), so cm1 comes nearly free. `flash_attn_cm2.comp` has its own `buffer_reference` decode path and would be additional work. **Proposal: scalar is the SP1 gate (RX480), cm1 included if free, cm2 a parity follow-on.** (Confirm.)

## Architecture — where rotation happens

All RHT rotation is **graph-level** (`llama-graph.cpp`), via the existing `GGML_OP_FWHT` op. The FA kernel and the quantize kernel are **rotation-free** (centroid encode/decode only):

| Tensor | Rotation | Where |
|---|---|---|
| Q | forward FWHT before FA | graph op (`llama-graph.cpp:2769`) |
| K | forward FWHT, then quantized into cache | graph op + cpy-to-cache |
| V | forward FWHT, then quantized into cache | graph op + cpy-to-cache |
| O (FA output) | **inverse** FWHT (because V was rotated) | graph op (`llama-graph.cpp:2339`) |

Confirmed in source: `quantize_f32_turbo4_0_block` does pure nearest-centroid ("expects already-rotated input"); `turbo4_dequant_element` = `TURBO_CENTROIDS_4BIT[idx] * norm`. There is **no double-rotate**: rotation lives only in the FWHT op.

## Components

### 1. `GGML_OP_FWHT` Vulkan op — the linchpin
- Implement the existing core op for the Vulkan backend (CUDA ref: `ggml-cuda/fwht.cu`).
- Normalized Fast Walsh-Hadamard, 128-wide for turbo4_0 (CUDA supports 64/128/256/512).
- Needs **forward and inverse** (inverse = same butterfly, `1/n` normalization + sign-step ordering; CUDA exposes a `scale` param — confirm whether direction is a flag or a second op).
- Used 4 ways: Q pre-rotate, K pre-rotate, V pre-rotate, O inverse-rotate.
- **Must be subgroup-size-agnostic (wave64).** This is the GLSL analog of the gfx803 `WARP_SIZE` scatter bugs — use `gl_SubgroupSize`/explicit widths, never assume 32.

### 2. f32→turbo4_0 cpy/quantize shader
- New Vulkan cpy/dup variant with `turbo4_0` as dst type.
- Per 128-block: nearest-centroid (Lloyd-Max midpoints → 4-bit index), nibble-pack 128 indices into 64 bytes, compute + store the fp16 `norm`. **No rotation** (input pre-rotated by op 1).
- Register in the Vulkan cpy dispatch.

### 3. turbo4_0 dequant on the contiguous FA path
- Add `FA_DEQUANT4_TURBO4_0(BUF)` to `flash_attn_dequant.glsl`: `norm × TURBO_CENTROIDS_4BIT[nibble]` (16-entry constant LUT).
- `FaType` enum value mirroring `GGML_TYPE_TURBO4_0`.
- Host: k_type/v_type acceptance (~`ggml-vulkan.cpp:3583`), block-size table entry (68 B / 128 elems), pipeline creation.

### 4. Host plumbing + constants
- Register `GGML_TYPE_TURBO4_0` as a legal Vulkan KV cache type so `--cache-type-k/-v turbo4` selects it.
- Port the centroid + midpoint constant tables (`TURBO_CENTROIDS_4BIT`, midpoints) verbatim into GLSL.

## Testing

1. **`test-backend-ops`** — FWHT (fwd + inverse) and the turbo4_0 cpy, Vulkan output vs CPU reference, bit-level. Run the FWHT case explicitly at wave64.
2. **PPL parity vs CUDA turbo4_0** (not just f16) on the same corpus, at 512 *and* a long context (≥16k) — proves the RHT+centroid path matches and holds at range. Accumulated sub-tolerance drift is the known failure mode (the gfx803 lesson), so compare end-to-end PPL, not just per-op tolerance.
3. **Perf** — tg/pp on the RX480 vs f16 and vs q4_0; confirm turbo4 is competitive and the FWHT op cost is acceptable on Polaris (no fast WHT path).
4. (Optional) NIAH-style long-context retrieval check.

## Risks / open items

- **FWHT correctness on wave64** — primary risk; the op is reused 4× so a sign/width bug corrupts everything. Mitigate with the explicit `test-backend-ops` wave64 case.
- **Inverse-FWHT parameterization** — confirm forward/inverse is a flag on one op vs two ops before wiring.
- **cm2 coverage** — in SP1 or follow-on (see Coverage decision).
- **Drift accumulation** — validate via end-to-end PPL, not per-op tolerance alone.

## Out-of-scope reminder

SP1 is contiguous-KV only. The 512k swarm's *dynamic* multi-agent block sharing comes with paged attention in **SP2**; SP1 proves turbo4 quality/perf/correctness on Vulkan on the contiguous path first.
