# SP2 — turbo4_0 Paged Attention on Vulkan (Design)

**Date:** 2026-06-29
**Status:** Design — awaiting plan
**Predecessor:** SP1 (turbo4_0 KV-quant on Vulkan FA) — COMPLETE (branch tip `19ef14d23`)
**Successors:** SP3 (turbo4_64-native paged), SP4 (KV tiering + semantic retrieval)

## Goal

Make `GGML_OP_PAGED_ATTN_MT` run on the RX480 / RADV (gfx803, wave64) Vulkan
backend for a **turbo4_0** paged KV cache, numerically matching the GTX1070 /
CUDA paged path. This is the first paged-attention slice and the path the
swarm actually runs (turbo4 KV exclusively; F16 KV is not used and is not a
deliverable). Targets both internal swarm use and eventual public llama.cpp
release.

## Why turbo4_0 first (not F16)

The swarm runs turbo4 KV exclusively — an F16 paged path would be code we
never ship. turbo4_0 is also *not* harder than F16 here:

- **turbo4 K and V are symmetric.** Both use the layout
  `[BLOCK_SIZE, HEAD_SIZE/128]` of `block_turbo4_0` with the *same* load path.
  (F16 is asymmetric: V is head-dim-major, K is X-strided with `K_X=8`.)
- **The WHT rotation lives outside the op.** Per SP1's architecture, Q is
  rotated by a graph-level `GGML_OP_TURBO_WHT` (already ported to Vulkan in
  SP1), K/V rotation is bundled inside the scatter-quantizer (SP1's
  cooperative quantizer), and the read returns rotated `centroid×norm`, so
  `⟨Q_rot, K_rot⟩ = ⟨Q, K⟩` (WHT is orthogonal). The graph wiring is
  backend-agnostic → port-for-free. **SP2's op shaders never touch WHT.**
- **SP1 already built the turbo4_0 pieces.** The cooperative f32→turbo4_0
  quantizer (`cpy_f32_turbo4_0.comp` / `set_rows_f32_turbo4_0.comp`) and the
  turbo4_0 dequant (`centroid×norm`) are validated. SP2 reuses them; the new
  work is *paged plumbing only*.

## Why turbo4_64 is deferred to SP3 (not in SP2)

1. **Zero Vulkan presence.** SP1 ported only turbo4_0. `grep` finds nothing
   turbo4_64 in `ggml/src/ggml-vulkan/`. turbo4_64 paged would first require
   porting a second quantizer + dequant to Vulkan.
2. **Different quant family, not a parameter swap.** turbo4_64 = 64-element
   blocks, struct `block_turbo4_64` (34 B vs 68 B), and its dequant
   (`TURBO_CENTROIDS_4BIT[idx]*norm`) applies **no WHT/RHT rotation**
   ("no-RHT"). Both the scatter-quantizer and graph wiring differ from
   turbo4_0's WHT-coupled path.
3. **Deferring costs ~nothing.** LFM2.5 (head_dim 64) reaches the turbo4_0
   paged path via the existing graph pad-to-128
   (`src/llama-graph.cpp:2663-2676`), so the 480 runs paged turbo4 KV today
   without native turbo4_64. turbo4_64 is a later memory-efficiency
   optimization (no 64→128 pad waste), not a blocker.

## The op (`GGML_OP_PAGED_ATTN_MT`)

Defined in `ggml/src/ggml.c` (`ggml_paged_attn_mt`). It **fuses two stages**
in one op handler:

1. **Scatter:** quantize-write `k_cur`/`v_cur` (F16,
   `[head_dim, n_kv_heads, n_tokens]`) into the paged `k_cache`/`v_cache` at
   positions given by `slot_mapping` (I32, one slot per token). The cache
   tensors are `src[1]`/`src[2]` and are **mutated in place**.
2. **Attention:** for each query, gather K/V through `block_tables` up to
   `context_lens`, compute `softmax(scale · Q·Kᵀ) · V` with causal masking
   and GQA head mapping.

**Sources:** `src[0]=q (F16)`, `src[1]=k_cache`, `src[2]=v_cache`,
`src[3]=block_tables (I32)`, `src[4]=context_lens (I32)`,
`src[5]=q_lens (I32)`, `src[6]=k_cur (F16)`, `src[7]=v_cur (F16)`,
`src[8]=slot_mapping (I32)`.
**op_params:** `[0]=scale (f32)`, `[1]=block_size (i32)`,
`[2]=max_blocks_per_seq (i32 = block_tables->ne[0])`, `[3]=n_kv_heads (i32)`.
**Output:** F16, shape of `q`: `[head_dim, n_heads, sum(q_lens), 1]`.

## Target parameters (SP2 slice)

| Param | SP2 value | Notes |
|---|---|---|
| cache type | `TURBO4_0` (K==V) | F16 = free identity bring-up variant only |
| `HEAD_SIZE` | 128 | turbo4_0 block = 128; 1 block/token/head |
| `BLOCK_SIZE` | 16 | `kv_tier_paged_block_size` default |
| GQA | `n_heads ≥ n_kv_heads` | LFM2.5 uses GQA; must map q_head→kv_head |
| subgroup | wave64-aware | no 32-lane assumptions; shared-mem reductions |

Out of SP2: turbo4_64-native, Q8_0/F16 as products, head_dim 64/256 native,
block_size 32, FP8, tiering+semantic (SP4).

## Cache layout (mirror CUDA `mt_pagedattn_ops.cuh` exactly)

The Vulkan shaders must read/write the **identical byte layout** the CUDA path
uses, or the 1070-vs-480 equivalence test is meaningless.

**turbo4_0**, per `(paged_block, kv_head)`: `[BLOCK_SIZE, HEAD_SIZE/128]` of
`block_turbo4_0`. With `N_QBLK = HEAD_SIZE/128` (= 1 at head 128):

```
element_block_index(paged_block, kv_head, token_in_block, d) =
    (paged_block * n_kv_heads + kv_head) * BLOCK_SIZE * N_QBLK
  + token_in_block * N_QBLK
  + d / 128
iqs (within block) = d % 128
dequant = TURBO_CENTROIDS_4BIT[ qs_nibble(iqs) ] * norm     // rotated domain
```

K and V use the identical index and load path.

**Scatter source** (`k_cur`/`v_cur`, F16 `[head_dim, n_kv_heads, n_tokens]`):
`src_elem = token*n_kv_heads*HEAD_SIZE + kv_head*HEAD_SIZE + d`. Scatter is
**non-redundant** (one quantize-write per `(token, kv_head)`).
`slot = slot_mapping[token]`, `paged_block = slot/BLOCK_SIZE`,
`token_in_block = slot % BLOCK_SIZE`.

## Architecture: cache-type-generic shaders

Mirror CUDA's `paged_cache_ops<TYPE>` abstraction: write the plumbing
(block-table indirection, slot-mapping scatter, attention loop, softmax,
GQA, causal mask, split-K) **once, type-agnostic**, with the per-element
load/store as a compile variant selected by SP1's `DATA_A_TURBO4_0` macro
pattern (and `DATA_A_F16` for the identity bring-up variant). The turbo4_0
specialization plugs in SP1's cooperative quantizer (store) and dequant
(load).

### New shaders (`ggml/src/ggml-vulkan/vulkan-shaders/`)

All wave64-aware (shared-memory reductions; no `gl_SubgroupSize==32`
assumptions; no coopmat — Polaris has no matrix cores).

1. **`paged_scatter_turbo4_0.comp`** — cooperative f16→turbo4_0
   quantize-scatter. Reuses SP1's `cpy_f32_turbo4_0.comp` cooperative WHT-quant
   math (128-thread block, tree-reduced L2 norm, s1/WHT/s2, centroid ladder,
   recon-norm correction, nibble pack) with **paged addressing**: read
   `k_cur`/`v_cur` at `src_elem`, write `block_turbo4_0` at
   `element_block_index(slot)`. One workgroup per `(token, kv_head, qblock)`.
   K and V scattered by two dispatches (or one over a doubled grid).

2. **`paged_attn_turbo4_0.comp`** — general / prefill path. One workgroup per
   `(query_token, q_head)`. Loop KV positions `0..context_len`: map
   `q_head → kv_head` (`q_head / (n_heads/n_kv_heads)`); for each KV position,
   resolve `paged_block = block_tables[seq*max_blocks_per_seq + pos/BLOCK_SIZE]`,
   `token_in_block = pos % BLOCK_SIZE`; dequant K via the cache-ops load;
   accumulate `dot = scale·Σ_d Q[d]·K[d]`; online-softmax (running max +
   denom); dequant V and accumulate weighted output. Causal mask via
   `context_lens` / query position. Shared-memory reduction across the
   head-dim threads.

3. **`paged_attn_decode_turbo4_0.comp`** + **`paged_attn_split_k_reduce.comp`**
   — split-K decode (single-query-per-seq fast path). Mirror
   `mt_pagedattn_decode.cu`: `CHUNK_KV = 128`, `DECODE_NUM_THREADS = 128`,
   one workgroup per `(seq, q_head, kv_chunk)` producing partial
   `(out, max, denom)`; a reduce shader (adapt the existing
   `flash_attn_split_k_reduce.comp`) combines partials per `(seq, q_head)`.
   Number of splits = `ceil(max_context_len / CHUNK_KV)`.

### Op handler (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`)

- `ggml_vk_paged_attn_mt(ctx, dst)`: enqueue scatter-K, scatter-V; pipeline
  **barrier** (attention reads what scatter wrote); then dispatch decode
  (when all `q_lens==1`) or prefill attention; for decode, dispatch the
  split-K reduce. Allocate the split-K partials scratch buffer.
- Push-constant structs for scatter and attention (dims, strides,
  `n_kv_heads`, `n_heads`, `block_size`, `max_blocks_per_seq`, `scale`).
- Pipeline registration (mirror SP1's turbo4_0 pipeline + the FA pipelines).
- `ggml_backend_vk_device_supports_op`: add `GGML_OP_PAGED_ATTN_MT` — require
  `k_cache->type == v_cache->type == TURBO4_0`, `q->type == F16`,
  `head_dim == 128`, `block_size == 16`, index tensors I32. (F16 variant may
  be admitted too for the bring-up test; gated narrowly.)
- Dispatch switch: route `GGML_OP_PAGED_ATTN_MT` to the handler.

### Files

- Create: 4 shaders above.
- Modify: `ggml-vulkan.cpp` (handler, pipelines, push-constants, supports_op,
  dispatch), `vulkan-shaders-gen.cpp` (register the new shader variants),
  `tests/test-backend-ops.cpp` (scatter + equivalence cases).
- Reuse (no change): SP1's `cpy_f32_turbo4_0.comp` math, `types.glsl`
  `block_turbo4_0`, graph-level `GGML_OP_TURBO_WHT` Vulkan op.

## Testing strategy

**Decision (user):** no CPU reference op is built. The numeric oracle is the
**GTX1070 / CUDA** paged path (not ROCm). Everything testable without a
reference is tested on the 480 alone first.

1. **Scatter correctness (480-only, deterministic).** The scatter is a
   permutation + the SP1 quantizer, both deterministic. A `test-backend-ops`
   case writes known F16 `k_cur`/`v_cur` with a fixed `slot_mapping`, runs the
   scatter, reads back the paged buffer, and compares against a host
   computation that applies the turbo4_0 quantizer oracle at the expected
   slots. No GPU reference needed.

2. **Plumbing / support / shapes (480-only).** Op-support admission, pipeline
   load without crash, output shape, block-table indexing edge cases
   (multi-block sequences, `context_len` not a multiple of `BLOCK_SIZE`),
   GQA mapping. The **F16 identity variant** isolates plumbing bugs from quant
   bugs.

3. **Attention numerics (1070-vs-480 equivalence).** A single **capped** build
   with **both** `GGML_CUDA=ON` and `GGML_VULKAN=ON` exposes `CUDA0` (1070)
   and `Vulkan0` (480) in one process. A `test-backend-ops` case (or small
   harness) runs `PAGED_ATTN_MT` with fixed seeded inputs on both backends and
   diffs the F16 output within a turbo4-class tolerance. The earlier host OOM
   was an *uncapped* `-j` CUDA build; this build runs inside the systemd
   `--user` scope (`MemoryMax`/`MemoryHigh`/low `-j`) so any OOM is contained.
   No separate CUDA wrapper, no cross-binary dump-and-diff, no seed-matching.

**Inference gate (unchanged standing constraint):** any end-to-end run
(`llama-cli`/`llama-server`/`llama-perplexity`/`llama-bench`) requires explicit
user go-ahead and is never autonomous. SP2's correctness gate is the
1070-vs-480 op equivalence above, which needs no inference.

**Perf:** per the SP1 lesson, benchmark the paged path (`llama-bench` pp/tg,
behind the inference gate) before declaring SP2 done — op-correctness tests do
not catch perf cliffs. Profile with `GGML_VK_PERF_LOGGER=1`. A cooperative
(not serial-per-thread) shader is mandatory.

## Build

One capped build, CUDA+Vulkan on, via the systemd `--user` scope pattern of
`build-vk.sh` (adaptive `MemoryMax`/`MemoryHigh`, `-j` scaled to free RAM).
The CUDA flag is folded into that build path; never an uncapped `nvcc`.
`MemoryHigh` must exceed the biggest single TU's peak RSS (don't throttle into
swap). Test binary: `build-vk/bin/test-backend-ops`, run with
`-b Vulkan0` (480) and `-b CUDA0` (1070).

## Risks / watch-items

- **WHT graph wiring for the paged branch.** Verify the backend-agnostic graph
  applies `GGML_OP_TURBO_WHT` to Q around the paged op (as it does for the
  non-paged FA path) when the cache is turbo4_0. If it only wires WHT for the
  FA branch, the paged Q would be unrotated and the dot wrong. Confirm in
  `src/llama-graph.cpp` build_attn paged branch. (Plan task 0 / pre-flight.)
- **wave64 reductions.** Polaris is wave64. Use shared-memory or
  subgroup-size-agnostic reductions; do not translate CUDA 32-lane warp
  collectives literally.
- **TURBO_MID_4BIT midpoint delta.** The ~1e-6 centroid-midpoint delta noted in
  SP1 (CUDA-parity) is the first suspect if 1070-vs-480 equivalence drifts.
- **head_dim 64 pad path.** SP2 targets head 128; LFM2.5 head 64 relies on the
  graph pad-to-128. Confirm the padded path produces head-128 turbo4_0 tensors
  the SP2 shaders accept.
- **split-K scratch sizing.** `ceil(max_ctx/CHUNK_KV)` partials per
  `(seq, q_head)`; size the scratch buffer from the op's max context.

## Out of scope (later SPs)

- SP3: turbo4_64-native paged (own Vulkan quant/dequant), turbo4_0 head 64/256.
- SP4: KV tiering (warm host / SSD cold) + semantic retrieval — mostly
  backend-agnostic host code; depends only on Vulkan paged attention existing
  and `ggml_backend_tensor_get/set` being correct/synchronized for Vulkan.
