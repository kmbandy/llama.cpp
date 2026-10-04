# SP2.5 — Multi-bracket paged turbo attention on Vulkan (design)

**Status:** approved (brainstorm 2026-06-30)
**Extends:** [SP2 design](2026-06-29-sp2-turbo4-paged-attn-vulkan-design.md) and [SP2 plan](../plans/2026-06-29-sp2-turbo4-paged-attn-vulkan.md)
**Branch:** `feat/sp1-turbo4-vulkan-fa` (continues SP2). Multi-bracket BASE = SP2 HEAD `c6561f110`.

## Motivation

SP2 built `GGML_OP_PAGED_ATTN_MT` on the Vulkan backend (RX480 / RADV POLARIS10 / wave64)
for a turbo4_0 paged KV cache and validated it at the op level against the GTX1070/CUDA
paged path. SP2 pinned **head_dim == 128 / turbo4_0** (the 128-element quant block).

The end-to-end gate (SP2 Task 6) exposed that the swarm's real models do **not** fit that
single bracket:

| Model | arch | head_dim | native paged KV (CUDA) |
|---|---|---|---|
| LFM2.5-8B-A1B | lfm2moe | **64** | TURBO4_64 (34 B/head) |
| Qwen3.5-4B | qwen35 | **256** | TURBO4_0, 2 blocks/head |
| ornith-9b | qwen35 | 256 | TURBO4_0, 2 blocks/head |

On Vulkan, both off-bracket cases currently abort: `supports_op` rejects `head_dim != 128`,
the scheduler falls the op back to the CPU backend, and the CPU has no `PAGED_ATTN_MT`
kernel (by design — SP2 decision: no CPU op) → `ggml_get_n_tasks: op not implemented`.

A working **interim** exists for head_dim-64 models: `GGML_PAGED_TURBO4_64=0` forces the
legacy path that pads head_dim 64 → 128 and keeps TURBO4_0, which the existing SP2 Vulkan
code already runs end-to-end (verified: LFM2.5 PPL = 28.27 over 4 chunks, Vulkan0, exit 0).
That path is correct but doubles the turbo4 KV footprint (68 vs 34 B/head).

## Goal

Generalize the SP2 paged turbo path so the Vulkan backend natively serves all three brackets
the swarm runs — head_dim **64** (turbo4_64), **128** and **256** (turbo4_0) — each
numerically matching the CUDA paged path, with a real-model end-to-end run per bracket.

## Scope & sequencing

Two additive features on the existing SP2 framework, sequenced (user decision 2026-06-30):

1. **Feature 1 — head_dim generalization (N_QBLK).** Unlocks head_dim 256 (currently broken)
   and formalizes 128. Reuses the existing turbo4_0 / F16 cache-ops.
2. **Feature 2 — native turbo4_64 bracket.** A new 64-element cache-ops branch for head_dim-64
   models (LFM2.5). A footprint optimization (halves turbo4 KV); the padded-128 path is the
   interim until it lands.

Out of scope: KV tiering + semantic eviction (SP4); non-turbo cache types beyond the F16
test variant; head dims that are not a multiple of 64.

## Bracket matrix (target end state)

| Bracket | Cache type | Block | Blocks/head (N_QBLK) | Feature |
|---|---|---|---|---|
| head_dim 64 | turbo4_64 | 34 B, 64-elt | 1 | Feature 2 |
| head_dim 128 | turbo4_0 | 68 B, 128-elt | 1 | SP2 (done) |
| head_dim 256 | turbo4_0 | 68 B, 128-elt | 2 | Feature 1 |

F16 is a per-bracket test-only identity variant (same plumbing, trivial cache-ops).

## Architecture

The op fuses scatter (quantize-write k_cur/v_cur into the paged cache at slot_mapping) and
attention (block-table gather → online softmax → causal/GQA), with a split-K decode fast path
for q_len==1. All of that — block tables, slot mapping, the dual-backend CUDA oracle, the
harness, the capped build — is head_dim/quant-generic already. This work touches only:

- `paged_cache_ops.glsl` — the per-cache-type load/store helpers.
- `paged_attn_scatter.comp`, `paged_attn.comp`, `paged_attn_decode.comp` — the N_QBLK loop.
- `vulkan-shaders-gen.cpp` — new shader variants.
- `ggml-vulkan.cpp` — pipelines, push-constants, supports_op, dispatch grids.
- `tests/test-paged-attn-vk.cpp` — bracket harness cases + turbo4_64 readback oracle.

**RHT-free invariant (carried, load-bearing):** every turbo paged cache-op here is
RHT-free. turbo4_64 quantize = load → L2-norm → normalize → (NO Hadamard) → nearest-centroid
→ recon-norm-correct → nibble-pack; dequant = `centroid · norm` (un-rotated). This matches
the CUDA `mt_scatter_kv_turbo4_64` path and the SP2 turbo4_0 convention. No graph changes.

### Feature 1 — head_dim generalization (Approach A)

The cooperative turbo quantizer is inherently per-128-element block, so the 128-thread
workgroup stays the unit and head_dim 256 maps as two blocks.

- **Cache-ops:** `pa_k_off`/`pa_v_off` already encode `(block_index, intra-block-dim)`;
  callers pass full `head_dim` instead of assuming one block. `element_block_index` already
  computes the `d/128` block term — it stops being assumed-0. `N_QBLK = head_dim / QBE`
  (quant-block elements: 128 for turbo4_0; F16 uses its existing PA_KX=8 striding over the
  full head_dim).
- **Attention (`paged_attn.comp`, `paged_attn_decode.comp`):** thread `t` loops
  `v = 0 .. N_QBLK-1` over dims `v*128 + t`, summing `Q[d]·K[d]` into the shared-mem tree
  reduction and carrying a per-qblock output accumulator `acc_o[v]`. These are the
  `MAX_VEC = 8` `q_reg[]`/`acc_o[]` arrays already present in the shaders (kept "for
  head-size generality"). Online-softmax state (m, l) is per (query, head), shared across
  qblocks.
- **Scatter (`paged_attn_scatter.comp`):** each 128-thread workgroup quantizes exactly one
  block; the dispatch grid y-dim becomes `n_kv_heads * N_QBLK` (the fix the SP2 Task-4
  review flagged), with `gl_WorkGroupID.y` decoding to `(kv_head, qblock)`.
- **Gate:** `head_dim % 128 == 0` (was `== 128`) with `N_QBLK <= MAX_VEC`.

### Feature 2 — native turbo4_64 bracket

- **Block (`block_turbo4_64`, ggml-common.h):** 34 B = `ggml_half norm` (2) + `qs[32]`
  (64 nibbles). **No `rnorm` field** — differs from turbo4_0's 68 B (norm + rnorm + qs[64]).
  64-element block; one head_dim-64 head is exactly one block.
- **Cache-ops (`#ifdef DATA_A_TURBO4_64` in `paged_cache_ops.glsl`):** `d/64` block
  indexing; `element_block_index` uses the 64-element block stride; dequant `= centroid · norm`.
  Buffers bound as `block_turbo4_64[]` (34 B stride; verify std430 layout matches CUDA).
- **Scatter quantizer:** mirrors Task 4's no-RHT cooperative quantizer but reduces L2-norm and
  Σcentroid² over **64** elements, **64 active threads** (clean 1:1 with the 64-element block;
  decision 2026-06-30), writing the 34 B block (`norm` + nibbles; no rnorm). Same WAR-barrier
  discipline.
- **Variants:** register `paged_attn_scatter_turbo4_64`, `paged_attn_turbo4_64`,
  `paged_attn_decode_turbo4_64` in the generator; pipelines indexed by cache type
  (extend the F16/TURBO4_0 indexing to TURBO4_64).
- **Gate:** admit `k_cache->type == v_cache->type == GGML_TYPE_TURBO4_64` with
  `head_dim == 64`. No host changes — `llama-kv-cache-paged.cpp` already remaps TURBO4_0 →
  TURBO4_64 for head_dim-64 models when `GGML_PAGED_TURBO4_64` is on (default).

## supports_op (final state)

Admit `GGML_OP_PAGED_ATTN_MT` when: q and op type F16; all four index tensors I32; k_cur/v_cur
F16; block_size (`op_params[1]`) == 16; and one of:

- `k_cache == v_cache == F16` with `head_dim % 128 == 0` (test variant), or
- `k_cache == v_cache == TURBO4_0` with `head_dim % 128 == 0` and `head_dim/128 <= MAX_VEC`, or
- `k_cache == v_cache == TURBO4_64` with `head_dim == 64`.

(CUDA's gate omits the head_dim/block_size checks entirely; we keep them because the Vulkan
shaders assume these layouts. Anything admitted here must have a real shader path — never
relax the gate past what the shaders support, or the scheduler silently falls back to the
CPU backend, which has no kernel and aborts.)

## Testing

**Op-level (CUDA-equivalence harness, `test-paged-attn-vk.cpp`, the gate per task):**

- Feature 1: turbo4_0 + F16 at **head_dim 256**, prefill and decode (ctx spanning chunk
  boundaries {32,128,200,512}), the multi-qblock (N_QBLK=2) path. Non-degenerate cache fill
  (the SP2 Task-5 fix) so the multi-chunk reduce is genuinely exercised.
- Feature 2: turbo4_64 at **head_dim 64**, prefill and decode, plus the deterministic 480-only
  scatter-readback oracle extended to the 34 B block (host turbo4_64 quantizer; assert per-block
  norm within 1e-3 and all nibbles exact).
- Tolerances unchanged: F16 2e-3, turbo 5e-2.

**End-to-end (inference-gated, per-step user go-ahead; vs CUDA0 in-box, same binary):**

- head_dim 256 → **Qwen3.5-4B** (qwen35, lighter on the RX480 than ornith-9b), Vulkan0 turbo4_0:
  PPL `-c 512` vs CUDA0; `llama-bench` pp512/tg128.
- head_dim 64 → **LFM2.5** native turbo4_64, Vulkan0: PPL + bench vs CUDA0, plus a KV-footprint
  comparison against the padded-128 path (expect ~½).
- head_dim 128 → covered by the padded-LFM2.5 run (already green) + the op-level harness.

## Constraints (carried from SP2 — bind every task)

- **RHT-free** for all turbo paged cache-ops (no Hadamard); dequant = `centroid · norm`.
- **wave64 / Polaris:** shared-memory tree reductions only, never 32-lane subgroup/shuffle ops;
  no coopmat. `barrier()` outside `if (thread<N)` guards (uniform); `barrier()` before reusing a
  broadcast-read shared slot.
- **Build only via the capped wrapper** `WITH_CUDA=1 bash build-vk.sh <target>` (systemd --user
  memory cap). Never uncapped `cmake --build -j` / `ninja` / `nvcc` (OOM-killed the host before).
  Check `free -h` before a build.
- **Numeric oracle = GTX1070/CUDA0**, in-process via `ggml_backend_compare_graph_backend`. No CPU op.
- **Never stage the 4 CUDA WIP files** (`common.cuh`, `mt_pagedattn.cu`, `mt_pagedattn_aiter.cu`,
  `mt_pagedattn_turbo_fp8.cuh`). Stage only named files; never `git add -A`.
- **Inference is gated:** every e2e step needs explicit user go-ahead; never autonomous.
- **TURBO_MID_4BIT** centroid-midpoint delta (~1e-6 at idx 0,6,8,14) is CUDA-parity-correct — if
  turbo equivalence drift is tiny and localized, widen tolerance, don't "fix" it.

## Plan shape

One implementation plan, tasks sequenced: Feature 1 (head_dim generalization → 256 harness →
256 e2e) then Feature 2 (turbo4_64 cache-ops → scatter → harness + readback oracle → e2e), each
op-level task gated by the CUDA-equivalence harness, each e2e step inference-gated. Reuses the
SP2 dual-backend capped build and the `.superpowers/sdd/progress.md` ledger. Final whole-branch
review spans the full SP2 + SP2.5 range.
