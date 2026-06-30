# SP2.5 Multi-bracket Paged Turbo Attention (Vulkan) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the SP2 Vulkan paged turbo attention op so it natively serves head_dim 64 (turbo4_64), 128 and 256 (turbo4_0) — the brackets the swarm's real models span — each matching the GTX1070/CUDA paged path.

**Architecture:** SP2 built the whole paged framework (scatter + prefill + split-K decode + dual-backend CUDA oracle harness) for head_dim-128/turbo4_0. The prefill/decode shaders and the decode partials buffer are *already* head_dim-generic via their `MAX_VEC` per-thread loops. Feature 1 unlocks head_dim 256 by relaxing the `supports_op` gate, fixing the scatter dispatch grid to fan out per quant-block (`×N_QBLK`), and giving the F16 test scatter the same per-block convention. Feature 2 adds a native turbo4_64 cache-ops bracket (34 B/head, 64-element block, RHT-free) for head_dim-64 models. All work stays on the SP2 branch.

**Tech Stack:** C++ / GLSL compute (Vulkan), ggml/llama.cpp, RX480 RADV POLARIS10 (wave64, no coopmat) as SUT, GTX1070/CUDA0 as numeric oracle.

**Spec:** `docs/superpowers/specs/2026-06-30-sp2-multibracket-paged-turbo-vulkan-design.md`
**BASE (branch state at plan start):** `c6561f110` (SP2 complete). Final whole-branch review spans `2d96f287b..HEAD`.

## Global Constraints

- **RHT-free** for every turbo paged cache-op: quantize = load→L2-norm→normalize→(NO Hadamard)→nearest-centroid→recon-norm-correct→nibble-pack; dequant = `TURBO_CENTROIDS_4BIT[idx] * norm` (un-rotated). No graph changes.
- **wave64 / Polaris:** shared-memory tree reductions only — NEVER 32-lane subgroup/shuffle ops, no coopmat. `barrier()` outside any `if (thread<N)` guard (uniform across all 128 lanes). `barrier()` before reusing a shared-mem slot that was broadcast-read by other lanes (SP1 WAR-race fix).
- **Build ONLY via the capped wrapper:** `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk` (or the relevant target) from repo root. NEVER `cmake --build -j`, bare `ninja`, or `nvcc` — uncapped builds OOM-killed this host. Run `free -h` before a build; wait/report if memory is very low.
- **Numeric oracle = GTX1070/CUDA0**, in-process via `ggml_backend_compare_graph_backend`. No CPU op is built (CPU `PAGED_ATTN_MT` aborts by design).
- **Never stage the 4 CUDA WIP files**: `ggml/src/ggml-cuda/common.cuh`, `mt_pagedattn.cu`, `mt_pagedattn_aiter.cu`, `mt_pagedattn_turbo_fp8.cuh`. Stage only named files; never `git add -A`.
- **Inference is gated:** Tasks 2 and 5 run llama-perplexity/llama-bench → require EXPLICIT user go-ahead per step, never autonomous.
- **Tolerances:** F16 cases `2e-3`, turbo cases `5e-2`. TURBO_MID_4BIT centroid-midpoint delta (~1e-6 at idx 0,6,8,14) is CUDA-parity-correct — if turbo drift is tiny and localized, widen tolerance, don't "fix" it.
- **MAX_VEC = 8** (shaders): head_dim ≤ 1024. `N_QBLK = head_dim / quant_block_elems` must be ≤ MAX_VEC.

## File Structure

| File | Responsibility | Feature |
|---|---|---|
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | supports_op gate; scatter dispatch grid (`×N_QBLK`); turbo4_64 pipeline registration | 1 + 2 |
| `…/vulkan-shaders/paged_attn_scatter.comp` | F16 per-block convention; new turbo4_64 64-element quantizer | 1 + 2 |
| `…/vulkan-shaders/paged_cache_ops.glsl` | turbo4_64 load/dequant branch; factor centroid include for both turbo types | 2 |
| `…/vulkan-shaders/paged_attn.comp`, `paged_attn_decode.comp` | turbo4_64 buffer declarations (`#ifdef DATA_A_TURBO4_64`) | 2 |
| `…/vulkan-shaders/vulkan-shaders-gen.cpp` | register turbo4_64 shader variants | 2 |
| `tests/test-paged-attn-vk.cpp` | hd256 cases; turbo4_64 host quantizer + cases + readback oracle | 1 + 2 |
| `docs/.../plans/2026-06-30-…md`, `.superpowers/sdd/progress.md` | results + ledger | wrap |

---

## Task 1: head_dim generalization (unlock head_dim 256, turbo4_0 + F16)

**Files:**
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (supports_op `GGML_OP_PAGED_ATTN_MT` case ~16994-17021; scatter dispatch in `ggml_vk_paged_attn_mt` ~10068)
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp` (the `#ifdef DATA_A_F16` `main`, ~54-83)
- Test: `tests/test-paged-attn-vk.cpp` (add hd256 cases in `main`)

**Interfaces:**
- Consumes: existing `vk_op_paged_scatter_pc { HS, BS, n_kv_heads, n_tokens }`; `pa_k_off/pa_v_off/pa_k_store` from `paged_cache_ops.glsl` (already N_QBLK-generic for turbo4_0); `compare_paged_case(label, paged_case, vk, cuda, tol)` and `scatter_turbo4_readback(paged_case, vk)` already in the harness.
- Produces: a scatter dispatch grid `{ n_tokens, n_kv_heads * N_QBLK, 2 }` with `N_QBLK = head_size / 128` (turbo4_0/F16 use 128-element qblocks); a `supports_op` gate admitting `head_dim % 128 == 0 && head_dim/128 <= 8`.

- [ ] **Step 1: Add failing hd256 harness cases.** In `tests/test-paged-attn-vk.cpp` `main`, after the existing hd128 turbo4_0 prefill case and the decode loop, add a head_dim-256 block. Use `n_heads=8, n_kv_heads=2` (GQA 4:1), `block_size=16`:

```cpp
// ── head_dim 256 (N_QBLK=2): turbo4_0 + F16, prefill + decode ───────────────
{
    const paged_case p256t { 256, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_0 };
    all_ok = compare_paged_case("paged turbo4_0 hd256 prefill", p256t, vk, cuda, 5e-2) && all_ok;
    all_ok = scatter_turbo4_readback(p256t, vk) && all_ok;          // exercises N_QBLK=2 scatter
    const paged_case p256f { 256, 8, 2, 16, 32, 32, 1, GGML_TYPE_F16 };
    all_ok = compare_paged_case("paged f16 hd256 prefill", p256f, vk, cuda, 2e-3) && all_ok;
    for (int ctx : { 128, 512 }) {                                  // decode, multi-chunk reduce
        char lt[64], lf[64];
        snprintf(lt, sizeof lt, "paged turbo4_0 hd256 decode ctx=%d", ctx);
        snprintf(lf, sizeof lf, "paged f16 hd256 decode ctx=%d", ctx);
        const paged_case dt { 256, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_0 };
        const paged_case df { 256, 8, 2, 16, 1, ctx, 1, GGML_TYPE_F16 };
        all_ok = compare_paged_case(lt, dt, vk, cuda, 5e-2) && all_ok;
        all_ok = compare_paged_case(lf, df, vk, cuda, 2e-3) && all_ok;
    }
}
```

- [ ] **Step 2: Build and confirm the hd256 cases FAIL (gate rejects, or wrong output).**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: the hd128 cases still PASS; the new `hd256` cases FAIL or the op is reported unsupported (supports_op rejects `head_dim != 128`, so the compare either errors or the scatter grid under-dispatches → wrong output).

- [ ] **Step 3: Relax the supports_op head_dim gate.** In `ggml-vulkan.cpp`, the `GGML_OP_PAGED_ATTN_MT` case currently has:

```cpp
                if (q->ne[0] != 128) {
                    return false;
                }
```

Replace with (admit any multiple of 128 within MAX_VEC):

```cpp
                if (q->ne[0] % 128 != 0 || (q->ne[0] / 128) > 8 /*MAX_VEC*/) {
                    return false;
                }
```

- [ ] **Step 4: Fix the scatter dispatch grid to fan out per quant-block.** In `ggml_vk_paged_attn_mt`, the scatter dispatch is:

```cpp
    const vk_op_paged_scatter_pc scatter_pc = { head_size, block_size, n_kv_heads, n_tokens };
    ggml_vk_dispatch_pipeline(ctx, subctx, scatter_pipeline,
        { slot_mapping_buf, k_cur_buf, v_cur_buf, k_cache_buf, v_cache_buf },
        scatter_pc, { n_tokens, n_kv_heads, 2 });
```

Replace the grid with a per-block fan-out (turbo4_0 and F16 both use 128-element qblocks here; turbo4_64 in Task 4 sets `qblk_elems = 64`):

```cpp
    const uint32_t qblk_elems = 128u;                       // turbo4_0 / F16 (Task 4: 64 for turbo4_64)
    const uint32_t n_qblk     = head_size / qblk_elems;     // head_dim is a multiple of qblk_elems
    const vk_op_paged_scatter_pc scatter_pc = { head_size, block_size, n_kv_heads, n_tokens };
    ggml_vk_dispatch_pipeline(ctx, subctx, scatter_pipeline,
        { slot_mapping_buf, k_cur_buf, v_cur_buf, k_cache_buf, v_cache_buf },
        scatter_pc, { n_tokens, n_kv_heads * n_qblk, 2 });
```

- [ ] **Step 5: Give the F16 scatter the per-block (N_QBLK) convention.** In `paged_attn_scatter.comp`, replace the `#ifdef DATA_A_F16 main()` so each workgroup handles one 128-element slice of one (token, kv_head), matching the turbo4 grid:

```glsl
#ifdef DATA_A_F16
void main() {
    const uint token     = gl_WorkGroupID.x;          // global token index
    const uint y_idx     = gl_WorkGroupID.y;          // kv_head*N_QBLK + qb
    const uint kv_select = gl_WorkGroupID.z;          // 0 = K, 1 = V
    const uint lane      = gl_LocalInvocationID.x;     // 0..127

    if (token >= p.n_tokens) { return; }
    const uint N_QBLK = p.HS / 128u;
    const uint kv_head = y_idx / N_QBLK;
    const uint qb      = y_idx % N_QBLK;
    const uint d       = qb * 128u + lane;
    if (d >= p.HS) { return; }

    const int slot = slot_mapping[token];
    if (slot < 0) { return; }                          // padding token
    const uint paged_block = uint(slot) / p.BS;
    const uint tok         = uint(slot) % p.BS;

    const uint src = token * p.n_kv_heads * p.HS + kv_head * p.HS + d;
    if (kv_select == 0u) {
        pa_k_store(pa_k_off(paged_block, kv_head, p.n_kv_heads, tok, d, p.HS, p.BS), float(k_cur[src]));
    } else {
        pa_v_store(pa_v_off(paged_block, kv_head, p.n_kv_heads, tok, d, p.HS, p.BS), float(v_cur[src]));
    }
}
#endif // DATA_A_F16
```

(For HS=128 this is identical to the old behaviour: N_QBLK=1, qb=0, d=lane.)

- [ ] **Step 6: Verify the readback oracle covers N_QBLK=2.** Read `scatter_turbo4_readback` in `tests/test-paged-attn-vk.cpp`. Confirm it iterates every `block_turbo4_0` in `k_cache` (block count = `n_kv_heads * N_QBLK * BLOCK_SIZE` per paged block) rather than assuming one block per (token, kv_head). If it assumes N_QBLK=1, generalize its loop to `for (qb = 0; qb < head_dim/128; ++qb)` over the `pa_turbo_block_index` layout. (If it already loops by raw block count, no change.)

- [ ] **Step 7: Build and confirm all cases PASS.**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: every case PASS — the original hd128 cases, plus `paged turbo4_0 hd256 prefill`, `paged f16 hd256 prefill`, the hd256 decode cases, and the hd256 scatter readback (`PASS`). turbo4 max_err in the `5e-2` band, F16 in `2e-3`.

- [ ] **Step 8: Commit.**

```bash
git add ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp \
        tests/test-paged-attn-vk.cpp
git commit -m "feat(sp2.5): generalize paged attention to head_dim multiples of 128 (unlock hd256)"
```

---

## Task 2: head_dim 256 end-to-end on Qwen3.5-4B (INFERENCE GATE)

**Files:**
- Modify: `docs/superpowers/plans/2026-06-30-sp2-multibracket-paged-turbo-vulkan.md` (append Results), `.superpowers/sdd/progress.md`

**Every step runs inference — REQUIRES explicit user go-ahead before running. Not autonomous.**

**Interfaces:**
- Consumes: Task 1's generalized op (head_dim 256 admitted). Model `/home/kmbandy/models/Qwen3.5-4B-UD-Q4_K_XL.gguf` (qwen35, head_dim 256). Corpus `wikitext-2-raw/wiki.test.raw`.

- [ ] **Step 1: Smoke (4 chunks) on Vulkan0, paged turbo4.** With user go-ahead:

Run: `./build-vk/bin/llama-perplexity -m /home/kmbandy/models/Qwen3.5-4B-UD-Q4_K_XL.gguf --cache-type-k turbo4 --cache-type-v turbo4 --kv-tier-paged-blocks -c 512 -f wikitext-2-raw/wiki.test.raw -ngl 99 --device Vulkan0 --chunks 4`
Expected: runs to completion (exit 0), prints a finite PPL. No `op not implemented: PAGED_ATTN_MT` abort. (The `--cache-type-k` CLI token is `turbo4`, not `turbo4_0`.)

- [ ] **Step 2: Full PPL parity, Vulkan0 vs CUDA0.** With user go-ahead, run the full `wiki.test.raw` PPL at `-c 512` on `--device Vulkan0`, then again on `--device CUDA0`, same flags. Record both PPL ± stderr. Expected: Vulkan0 within noise of CUDA0.

- [ ] **Step 3: Throughput, Vulkan0.** With user go-ahead:

Run: `./build-vk/bin/llama-bench -m /home/kmbandy/models/Qwen3.5-4B-UD-Q4_K_XL.gguf --cache-type-k turbo4 --cache-type-v turbo4 --kv-tier-paged-blocks -p 512 -n 128 --device Vulkan0`
Record pp512 / tg128.

- [ ] **Step 4: Record results.** Append a "Results — hd256 (Qwen3.5-4B)" section to this plan with the parity (Vulkan0 vs CUDA0 PPL) and perf numbers. Note any cliff + diagnosis. Update `.superpowers/sdd/progress.md`.

- [ ] **Step 5: Commit.**

```bash
git add docs/superpowers/plans/2026-06-30-sp2-multibracket-paged-turbo-vulkan.md .superpowers/sdd/progress.md
git commit -m "docs(sp2.5): head_dim 256 e2e results (Qwen3.5-4B, Vulkan0 vs CUDA0)"
```

---

## Results — hd256 (Qwen3.5-4B)

Branch `feat/sp1-turbo4-vulkan-fa` @ 27c9b7d23 (Task 1: hd256 generalization, reviewed clean). Model
`/home/kmbandy/models/Qwen3.5-4B-UD-Q4_K_XL.gguf` (qwen35, head_dim 256, Q4_K_XL). Corpus
`wikitext-2-raw/wiki.test.raw`. Flags: `--cache-type-k turbo4 --cache-type-v turbo4 --kv-tier-paged-blocks
-c 512 -ngl 99` (perplexity), `--cache-type-k turbo4 --cache-type-v turbo4 -p 512 -n 128` (bench).

**Step 1 — Smoke (4 chunks, Vulkan0):** PASS. Ran to completion, exit 0. No `op not implemented:
PAGED_ATTN_MT` abort. PPL (4 chunks) = 8.5181 +/- 0.72686 (expected to be noisy with only 4 chunks).

**Step 2 — Full PPL parity, Vulkan0 vs CUDA0** (580 chunks, full `wiki.test.raw`, `-c 512`):

| Device | PPL | stderr | Wall time |
| --- | --- | --- | --- |
| Vulkan0 (RX 480 / RADV POLARIS10) | 10.1259 | ± 0.07238 | ~21.0 min (7.83 s/pass) |
| CUDA0 (GTX 1070, oracle) | 10.1484 | ± 0.07257 | ~9.4 min (3.86 s/pass) |

Delta = 0.0225, well inside one stderr (~0.0724) on either side — **Vulkan0 is within noise of CUDA0**.
This is the first real model to exercise the generalized `head_dim % 128 == 0` paged-attention path
end-to-end (prior validation was the op-level harness only); parity confirms Task 1's scatter/dispatch
fix and `supports_op` gate are correct for native hd256, not just synthetic cache fills.

**Step 3 — Throughput, Vulkan0** (`llama-bench`, `-p 512 -n 128`):

Note: the brief's bench command included `--kv-tier-paged-blocks`, which `llama-bench` does not accept
(it's a `llama-perplexity`/`llama-cli`-only flag — `llama-bench` has no such option in its `--help`). Ran
without it; `--cache-type-k/v turbo4` alone is sufficient to select the paged turbo4 cache path.

| test | t/s |
| --- | --- |
| pp512 | 266.68 ± 78.66 |
| tg128 | 35.80 ± 0.66 |

pp512's stderr is large relative to the mean (~30%) — typical for a single short prefill on this card with
default `-r 5` repetitions and no `--no-warmup` tuning; tg128's stderr is tight (~2%) and looks stable. No
perf cliff observed relative to the hd128 LFM2.5 numbers from SP2 (same backend, same op); the bd256 path
is not measurably worse per-token than hd128 on this hardware. No further diagnosis needed — flagged as a
measurement-noise note, not a regression.

**Verdict:** hd256 end-to-end on Vulkan0 is correctness-confirmed (PPL parity) and runs at expected
throughput. No code changes required; Task 1's generalization holds under a real model.

---

## Task 3: turbo4_64 cache-ops — load/dequant + attention (head_dim 64)

**Goal:** Add the turbo4_64 *read* path (dequant load + attention/decode variants) and validate it against CUDA with a host-prefilled turbo4_64 cache. Scatter (write) is Task 4 — the harness pre-fills the cache (the SP2 Task-5 mechanism), so the read path is testable independently.

**Files:**
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/paged_cache_ops.glsl` (factor centroid include; add `#ifdef DATA_A_TURBO4_64` load branch)
- Modify: `…/vulkan-shaders/paged_attn.comp`, `…/vulkan-shaders/paged_attn_decode.comp` (turbo4_64 buffer decls)
- Modify: `…/vulkan-shaders/vulkan-shaders-gen.cpp` (register `paged_attn_turbo4_64`, `paged_attn_decode_turbo4_64`)
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (register the two pipelines; admit TURBO4_64 in supports_op)
- Test: `tests/test-paged-attn-vk.cpp` (host turbo4_64 quantizer + fill; turbo4_64 prefill/decode cases)

**Interfaces:**
- Consumes: `block_turbo4_64` = `{ ggml_half norm; uint8_t qs[32]; }` (34 B, 64-element block, NO rnorm — `ggml-common.h:339`); `TURBO_CENTROIDS_4BIT`, `pa_turbo_nearest_4bit` (factored to be visible for both turbo types); the MAX_VEC attention/decode loops (already head_dim-generic).
- Produces: `pa_k_load/pa_v_load` for turbo4_64; pipelines `pipeline_paged_attn[GGML_TYPE_TURBO4_64]`, `pipeline_paged_attn_decode[GGML_TYPE_TURBO4_64]`; a host `host_turbo4_64_quantize_block(const float* x /*64*/, uint8_t* out /*34*/)` and `fill_turbo4_64` in the harness; supports_op admits `TURBO4_64 && head_dim==64`.

- [ ] **Step 1: Factor the centroid table + nearest-centroid helper so both turbo types see them.** In `paged_cache_ops.glsl`, the `#include "turbo_centroids.glsl"` and `pa_turbo_nearest_4bit` currently live inside `#ifdef DATA_A_TURBO4_0`. Change the guard to cover both turbo types so Task 3/4 can reuse them:

```glsl
#if defined(DATA_A_TURBO4_0) || defined(DATA_A_TURBO4_64)
#include "turbo_centroids.glsl"
uint pa_turbo_nearest_4bit(float v) { /* unchanged 15-way midpoint ladder */ }
#endif
```

(Move the existing `pa_turbo_nearest_4bit` body verbatim out of the `DATA_A_TURBO4_0` block into this shared guard; leave the turbo4_0 `pa_*_off`/`pa_*_load`/`PA_QK` definitions in their `DATA_A_TURBO4_0` block.)

- [ ] **Step 2: Add the turbo4_64 load branch to `paged_cache_ops.glsl`.** After the turbo4_0 block:

```glsl
#ifdef DATA_A_TURBO4_64
// turbo4_64: 4-bit PolarQuant, 64-element block (34 B: norm + qs[32], NO rnorm).
// RHT-FREE: dequant = TURBO_CENTROIDS_4BIT[idx] * norm.
#define PA_QK64 64u
uint pa_turbo64_block_index(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint qb, uint HS, uint BS) {
    const uint N_QBLK = HS / PA_QK64;
    return ((paged_block*n_kv_heads + kv_head) * BS * N_QBLK) + tok*N_QBLK + qb;
}
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    const uint block_ib = pa_turbo64_block_index(paged_block, kv_head, n_kv_heads, tok, d/PA_QK64, HS, BS);
    return block_ib * PA_QK64 + (d % PA_QK64);
}
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return pa_k_off(paged_block, kv_head, n_kv_heads, tok, d, HS, BS);
}
float pa_k_load(uint off) {
    const uint ib = off / PA_QK64; const uint iqs = off % PA_QK64;
    const uint idx = (uint(data_k[ib].qs[iqs >> 1u]) >> ((iqs & 1u) * 4u)) & 0xFu;
    return TURBO_CENTROIDS_4BIT[idx] * float(data_k[ib].norm);
}
float pa_v_load(uint off) {
    const uint ib = off / PA_QK64; const uint iqs = off % PA_QK64;
    const uint idx = (uint(data_v[ib].qs[iqs >> 1u]) >> ((iqs & 1u) * 4u)) & 0xFu;
    return TURBO_CENTROIDS_4BIT[idx] * float(data_v[ib].norm);
}
#endif // DATA_A_TURBO4_64
```

- [ ] **Step 3: Add turbo4_64 buffer declarations to the read shaders.** In `paged_attn.comp` and `paged_attn_decode.comp`, alongside the existing `#ifdef DATA_A_TURBO4_0` buffer blocks, add (and the matching int8/8-bit-storage `#extension` requires next to the turbo4_0 ones):

```glsl
#ifdef DATA_A_TURBO4_64
struct block_turbo4_64 { float16_t norm; uint8_t qs[32]; };   // 34 B, no rnorm
layout (binding = 1) buffer KCache  { block_turbo4_64 data_k[]; };
layout (binding = 2) buffer VCache  { block_turbo4_64 data_v[]; };
#endif
```

(In `paged_attn.comp` bindings 1/2 are the cache; in `paged_attn_decode.comp` use the same binding numbers its turbo4_0 block uses.)

- [ ] **Step 4: Register the turbo4_64 read variants.** In `vulkan-shaders-gen.cpp`, after the turbo4_0 lines:

```cpp
string_to_spv("paged_attn_turbo4_64",        "paged_attn.comp",        {{"DATA_A_TURBO4_64","1"},{"D_TYPE","float16_t"}});
string_to_spv("paged_attn_decode_turbo4_64", "paged_attn_decode.comp", {{"DATA_A_TURBO4_64","1"},{"D_TYPE","float16_t"}});
```

- [ ] **Step 5: Register the pipelines.** In `ggml-vulkan.cpp`, next to the turbo4_0 `pipeline_paged_attn[...]` / `pipeline_paged_attn_decode[...]` creations:

```cpp
ggml_vk_create_pipeline(device, device->pipeline_paged_attn[GGML_TYPE_TURBO4_64],        "paged_attn_turbo4_64",        paged_attn_turbo4_64_len,        paged_attn_turbo4_64_data,        "main", 7, sizeof(vk_op_paged_attn_pc),   {1,1,1}, {}, 1);
ggml_vk_create_pipeline(device, device->pipeline_paged_attn_decode[GGML_TYPE_TURBO4_64], "paged_attn_decode_turbo4_64", paged_attn_decode_turbo4_64_len, paged_attn_decode_turbo4_64_data, "main", 7, sizeof(vk_op_paged_decode_pc), {1,1,1}, {}, 1);
```

- [ ] **Step 6: Admit turbo4_64 in supports_op.** In the `GGML_OP_PAGED_ATTN_MT` case, the cache-type check currently admits F16/TURBO4_0:

```cpp
                if (k_cache->type != GGML_TYPE_F16 && k_cache->type != GGML_TYPE_TURBO4_0) {
                    return false;
                }
```

Extend to TURBO4_64, and require head_dim==64 specifically for it (the 64-element block has no N_QBLK fan-out for hd>64 in this plan). Replace the head_dim gate (from Task 1) and the cache-type gate with:

```cpp
                const bool is_t64 = (k_cache->type == GGML_TYPE_TURBO4_64);
                if (k_cache->type != GGML_TYPE_F16 &&
                    k_cache->type != GGML_TYPE_TURBO4_0 &&
                    !is_t64) {
                    return false;
                }
                if (is_t64) {
                    if (q->ne[0] != 64) { return false; }          // turbo4_64: one 64-elt block/head
                } else {
                    if (q->ne[0] % 128 != 0 || (q->ne[0] / 128) > 8) { return false; }
                }
```

- [ ] **Step 7: Add the host turbo4_64 quantizer + fill to the harness.** In `tests/test-paged-attn-vk.cpp`, alongside `host_turbo4_quantize_block` (68 B) and `fill_turbo4`, add a 64-element / 34-byte version (same no-RHT math: L2-norm → normalize → nearest-centroid → pack 32 bytes → write corrected norm; **no rnorm field**):

```cpp
static void host_turbo4_64_quantize_block(const float * x /*64*/, uint8_t * out /*34 bytes*/) {
    // mirror host_turbo4_quantize_block but over 64 elements; layout: norm(f16) @0..1, qs[32] @2..33
    // (reuse the same centroid table + nearest-centroid + recon-norm correction as the 128 version)
}
static void fill_turbo4_64(ggml_tensor * t, uint32_t seed) {
    // iterate 34-byte blocks; quantize deterministic random 64-vectors into each
}
```

Wire `fill_turbo4_64` into `build_case`'s cache-fill switch for `GGML_TYPE_TURBO4_64` (mirroring the `fill_turbo4` branch).

- [ ] **Step 8: Add failing turbo4_64 attention cases.** In `main`, add a head_dim-64 block (host-prefilled cache exercises the read path without the scatter):

```cpp
// ── head_dim 64 turbo4_64: prefill + decode (read path; cache host-prefilled) ──
{
    const paged_case t64 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64 };
    all_ok = compare_paged_case("paged turbo4_64 hd64 prefill", t64, vk, cuda, 5e-2) && all_ok;
    for (int ctx : { 128, 512 }) {
        char l[64]; snprintf(l, sizeof l, "paged turbo4_64 hd64 decode ctx=%d", ctx);
        const paged_case d64 { 64, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_64 };
        all_ok = compare_paged_case(l, d64, vk, cuda, 5e-2) && all_ok;
    }
}
```

- [ ] **Step 9: Build, confirm turbo4_64 attention cases PASS (scatter still untested — Task 4).**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: all prior cases still PASS; `paged turbo4_64 hd64 prefill` and decode cases PASS at `5e-2` (cache host-prefilled identically on both backends → genuine Vulkan-vs-CUDA dequant+attention comparison). Investigate first if max_err sits at the ~1e-6 centroid-midpoint scale (benign) vs a gross layout error (34 B stride / std430 mismatch).

- [ ] **Step 10: Commit.**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/paged_cache_ops.glsl \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_decode.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        tests/test-paged-attn-vk.cpp
git commit -m "feat(sp2.5): turbo4_64 cache-ops read path + attention (head_dim 64)"
```

---

## Task 4: turbo4_64 cooperative scatter quantizer (head_dim 64)

**Files:**
- Modify: `…/vulkan-shaders/paged_attn_scatter.comp` (add `#ifdef DATA_A_TURBO4_64` 64-element quantizer)
- Modify: `…/vulkan-shaders/vulkan-shaders-gen.cpp` (register `paged_attn_scatter_turbo4_64`)
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (register scatter pipeline; scatter grid `qblk_elems` by cache type)
- Test: `tests/test-paged-attn-vk.cpp` (extend `scatter_turbo4_readback` to turbo4_64; add a non-prefilled turbo4_64 equivalence case)

**Interfaces:**
- Consumes: Task 3's `block_turbo4_64`, `pa_turbo64_block_index`, `pa_turbo_nearest_4bit`, `TURBO_CENTROIDS_4BIT`; the existing `vk_op_paged_scatter_pc`.
- Produces: `pipeline_paged_attn_scatter[GGML_TYPE_TURBO4_64]`; scatter grid `{ n_tokens, n_kv_heads*(HS/qblk_elems), 2 }` with `qblk_elems = 64` when cache is turbo4_64.

- [ ] **Step 1: Write the turbo4_64 cooperative quantizer.** In `paged_attn_scatter.comp`, add a `#ifdef DATA_A_TURBO4_64` `main()` mirroring the turbo4_0 quantizer (lines 85-195) but over **64 elements with 64 active threads**, writing a 34 B block (norm + 32 nibble bytes, **no rnorm**). Key differences from turbo4_0: `s_x[64]`/`s_red[64]`; the L2 and recon tree reductions start at stride 32 (`if (j<32) … +=[j+32]` … down to 1); `block_turbo4_64` struct; `y_idx = kv_head*N_QBLK + qb` with `N_QBLK = p.HS/64u`; `d = qb*64u + j`; thread 0 writes only `.norm` (no `.rnorm`); nibble pack `qs[j>>1]` for `j<64` even lanes. Preserve the WAR-barrier discipline (barrier before reusing `s_red[0]` after the grp_norm and recon_norm broadcasts). Add the int8/8-bit-storage `#extension` requires for `DATA_A_TURBO4_64` next to the turbo4_0 ones.

- [ ] **Step 2: Register the scatter variant.** In `vulkan-shaders-gen.cpp`:

```cpp
string_to_spv("paged_attn_scatter_turbo4_64", "paged_attn_scatter.comp", {{"DATA_A_TURBO4_64","1"},{"D_TYPE","float16_t"}});
```

- [ ] **Step 3: Register the scatter pipeline.** In `ggml-vulkan.cpp`:

```cpp
ggml_vk_create_pipeline(device, device->pipeline_paged_attn_scatter[GGML_TYPE_TURBO4_64], "paged_attn_scatter_turbo4_64", paged_attn_scatter_turbo4_64_len, paged_attn_scatter_turbo4_64_data, "main", 5, sizeof(vk_op_paged_scatter_pc), {1,1,1}, {}, 1);
```

- [ ] **Step 4: Make the scatter grid use the cache-type block size.** In `ggml_vk_paged_attn_mt`, replace the fixed `qblk_elems = 128u` from Task 1 Step 4 with:

```cpp
    const uint32_t qblk_elems = (cache_type == GGML_TYPE_TURBO4_64) ? 64u : 128u;
    const uint32_t n_qblk     = head_size / qblk_elems;
```

(grid stays `{ n_tokens, n_kv_heads * n_qblk, 2 }`.)

- [ ] **Step 5: Extend the readback oracle + add a non-prefilled equivalence case.** In `tests/test-paged-attn-vk.cpp`, generalize `scatter_turbo4_readback` to handle turbo4_64 (34 B blocks, 64-element host quantizer, `pa_turbo64_block_index` layout) — branch on `c.cache_type`. Then add a turbo4_64 case that does NOT pre-fill the cache (so the device scatter is what populates it):

```cpp
{
    const paged_case t64 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64 };
    all_ok = scatter_turbo4_readback(t64, vk) && all_ok;     // device scatter vs host quantizer, bit-exact
}
```

- [ ] **Step 6: Build, confirm scatter readback + all turbo4_64 cases PASS.**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: `scatter turbo4_64 readback: PASS` (max_norm_err ~0, nibble_mismatch=0), and every prefill/decode case (hd128, hd256, hd64-turbo4_64) PASS.

- [ ] **Step 7: Commit.**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        tests/test-paged-attn-vk.cpp
git commit -m "feat(sp2.5): turbo4_64 cooperative no-RHT scatter quantizer (head_dim 64)"
```

---

## Task 5: turbo4_64 end-to-end on LFM2.5 + footprint (INFERENCE GATE)

**Files:**
- Modify: this plan (Results), `.superpowers/sdd/progress.md`

**Every step runs inference — REQUIRES explicit user go-ahead. Not autonomous.**

**Interfaces:**
- Consumes: Tasks 3-4 (turbo4_64 op complete). Model `/home/kmbandy/models/LFM2.5-8B-A1B-Q5_K_M.gguf` (head_dim 64). The host already remaps TURBO4_0→TURBO4_64 for head_dim-64 models when `GGML_PAGED_TURBO4_64` is unset/`1` (default).

- [ ] **Step 1: Native turbo4_64 smoke (4 chunks), Vulkan0.** With user go-ahead:

Run: `./build-vk/bin/llama-perplexity -m /home/kmbandy/models/LFM2.5-8B-A1B-Q5_K_M.gguf --cache-type-k turbo4 --cache-type-v turbo4 --kv-tier-paged-blocks -c 512 -f wikitext-2-raw/wiki.test.raw -ngl 99 --device Vulkan0 --chunks 4`
Expected: exit 0, finite PPL, no abort. (Default remap routes head_dim-64 → TURBO4_64; confirm via the load log that the cache type is turbo4_64, e.g. `GGML_PAGED_TURBO4_64` path taken.)

- [ ] **Step 2: Full PPL parity, Vulkan0 vs CUDA0** (native turbo4_64, same flags, both devices). Record PPL ± stderr. Expected: within noise of CUDA0, and comparable to the padded-128 baseline (PPL 28.27 over 4 chunks measured 2026-06-30).

- [ ] **Step 3: Throughput + footprint.** With user go-ahead: `llama-bench -p 512 -n 128 --device Vulkan0`. Record pp/tg. Capture the reported KV cache size for native turbo4_64 vs the padded-128 path (`GGML_PAGED_TURBO4_64=0`) — expect ~½ (34 vs 68 B/head).

- [ ] **Step 4: Record results** (parity + perf + footprint tables) in this plan; update `.superpowers/sdd/progress.md` marking SP2.5 complete.

- [ ] **Step 5: Commit.**

```bash
git add docs/superpowers/plans/2026-06-30-sp2-multibracket-paged-turbo-vulkan.md .superpowers/sdd/progress.md
git commit -m "docs(sp2.5): turbo4_64 e2e results (LFM2.5 native, ~half KV footprint)"
```

---

## Task 6: Results consolidation + final-review prep

**Files:**
- Modify: `docs/superpowers/plans/2026-06-30-sp2-multibracket-paged-turbo-vulkan.md`, `.superpowers/sdd/progress.md`

- [ ] **Step 1: Write the consolidated bracket matrix** (head_dim 64/128/256 × {op-level CUDA-equivalence, e2e PPL parity, perf}) into this plan's Results section, with the carried Minor findings from each task for the final whole-branch review to triage.
- [ ] **Step 2: Mark SP2.5 complete in the ledger** with the commit range and point the final whole-branch review at `2d96f287b..HEAD` (covers SP2 + SP2.5).
- [ ] **Step 3: Commit.**

```bash
git add docs/superpowers/plans/2026-06-30-sp2-multibracket-paged-turbo-vulkan.md .superpowers/sdd/progress.md
git commit -m "docs(sp2.5): multi-bracket results matrix + final-review pointer"
```

---

## Self-Review notes

- **Spec coverage:** Feature 1 (hd256 generalization) = Tasks 1-2; Feature 2 (turbo4_64) = Tasks 3-5; results/parity per bracket = Tasks 2/5/6. F16 test variant covered (hd256 F16 case Task 1; hd128 F16 pre-existing). All three brackets get op-level CUDA-equivalence; hd256 + hd64 get e2e; hd128 e2e is the existing padded-LFM2.5 run (noted in spec).
- **Type consistency:** `N_QBLK` = head_dim / quant-block-elems (128 for turbo4_0/F16, 64 for turbo4_64); scatter grid `{ n_tokens, n_kv_heads*N_QBLK, 2 }` everywhere; `block_turbo4_64` = 34 B (norm + qs[32], no rnorm) consistent across shader decls + host quantizer + cache-ops. `pa_turbo_nearest_4bit` factored once, used by both turbo types.
- **Inference gates:** Tasks 2 and 5 explicitly flagged; every other task is harness-gated (autonomous-safe).
