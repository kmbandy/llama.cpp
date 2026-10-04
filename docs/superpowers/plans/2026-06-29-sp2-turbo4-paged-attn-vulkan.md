# SP2 — turbo4_0 Paged Attention on Vulkan — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GGML_OP_PAGED_ATTN_MT` run on the RX480/RADV (gfx803, wave64) Vulkan backend for a turbo4_0 paged KV cache, numerically matching the GTX1070/CUDA paged path.

**Architecture:** A cache-type-generic Vulkan paged-attention shader set (mirroring CUDA's `paged_cache_ops<TYPE>` abstraction). All paged plumbing — block-table indirection, slot-mapping scatter, GQA, causal mask, online softmax, split-K decode — is written once; the per-element load/store is a compile variant selected by the `DATA_A_F16` / `DATA_A_TURBO4_0` macro pattern (SP1's convention). turbo4_0 is the deliverable; F16 is the trivial identity variant used as a bring-up oracle that isolates plumbing bugs from quant bugs. The op is **RHT-free** end-to-end (no WHT on Q, no Hadamard in the scatter; `dequant = centroid·norm ≈ K` directly).

**Tech Stack:** GLSL compute shaders (Vulkan 1.x, `GL_EXT_shader_explicit_arithmetic_types`, `GL_EXT_shader_16bit_storage`/`8bit_storage`), C++ (ggml-vulkan backend), CMake, `test-backend-ops`-style C++ test harness using `ggml_backend_compare_graph_backend`.

**Spec:** `docs/superpowers/specs/2026-06-29-sp2-turbo4-paged-attn-vulkan-design.md`

## Global Constraints

- **Build ONLY via the capped wrapper.** Never run an uncapped `cmake --build -j` or `nvcc`. All builds run inside the `systemd --user --scope` cgroup (`MemoryMax`/`MemoryHigh`/`CPUQuota`) of `build-vk.sh`. The uncapped CUDA build OOM-killed the host on 2026-06-28. `MemoryHigh` must EXCEED the biggest single TU's peak RSS (ggml-vulkan.cpp ≈ 5 GiB), or it throttles into swap-thrash. Always check `free -h` / `/proc/loadavg` before building (the wrapper does this).
- **turbo4_0 paged path is RHT-FREE.** The scatter must NOT apply the Hadamard transform. Mirror `mt_scatter_kv_turbo4_0_kernel` (`ggml/src/ggml-cuda/mt_pagedattn.cu:323-455`), NOT SP1's WHT-bundled `cpy_f32_turbo4_0.comp`. Q is passed un-rotated to the op. Dequant = `TURBO_CENTROIDS_4BIT[idx]·norm`.
- **Cache layout mirrors CUDA `mt_pagedattn_ops.cuh` byte-for-byte.** Any deviation makes the 1070-vs-480 equivalence test meaningless. turbo4_0: per `(paged_block, kv_head)` → `[BLOCK_SIZE, HEAD_SIZE/128]` of `block_turbo4_0`; `element_block_index = (paged_block·n_kv_heads + kv_head)·BLOCK_SIZE·N_QBLK + token_in_block·N_QBLK + d/128`; K and V identical layout/load.
- **Wave64-aware.** Polaris is subgroup-size 64. Use shared-memory tree reductions or subgroup-size-agnostic reductions. Do NOT translate CUDA 32-lane warp collectives (`__shfl_*_sync(..., WARP_SIZE)`) literally — replace with shared-memory reductions over the 128-thread workgroup. No coopmat (Polaris has no matrix cores).
- **No coopmat2 advertising.** Gate any new `supports_op` admission so it does not over-promise on coopmat2 devices the shader doesn't cover (follow SP1's `return !coopmat2` precedent where relevant).
- **Numeric oracle = GTX1070/CUDA paged path** (device `CUDA0`), NOT ROCm. No CPU reference op is built.
- **Inference gate.** Any `llama-cli`/`llama-server`/`llama-perplexity`/`llama-bench` run requires EXPLICIT user go-ahead and is never autonomous. Only Task 6 touches inference, and every step there is gated.
- **Target params:** turbo4_0, `HEAD_SIZE=128`, `BLOCK_SIZE=16`, GQA (`n_heads ≥ n_kv_heads`, `n_heads % n_kv_heads == 0`). F16 variant is test-only.
- **Commits:** stage only named files (never `git add -A`). The pre-existing uncommitted CUDA WIP files (`mt_pagedattn*`, `common.cuh`) are NOT part of SP2 — never stage them.
- **test binary:** `build-vk/bin/test-backend-ops` and `build-vk/bin/test-paged-attn-vk`. Run Vulkan with `CUDA_VISIBLE_DEVICES="" GGML_VK_VISIBLE_DEVICES=0` for `Vulkan0`=RX480. For dual-backend runs, expose both (`CUDA0`=1070, `Vulkan0`=480).
- **Process hygiene:** kill stray processes with `pkill -x <comm>` (exact name), never `pkill -f <pattern>` (matches the controlling shell → exit-144 self-kill). Avoid foreground `sleep`.

---

## File Structure

**New shaders** (`ggml/src/ggml-vulkan/vulkan-shaders/`):
- `paged_attn_scatter.comp` — cooperative quantize/identity scatter of `k_cur`/`v_cur` into the paged cache at `slot_mapping`. Type-generic via `DATA_A_F16` / `DATA_A_TURBO4_0`. turbo4_0 path = no-RHT cooperative quantizer.
- `paged_attn.comp` — prefill/general attention: one workgroup per `(query_token, q_head)`, block-table gather, online softmax, GQA, causal mask. Type-generic load via the cache-ops include.
- `paged_attn_decode.comp` — split-K decode: one workgroup per `(seq, q_head, kv_chunk)`, partial `(out, m, l)`.
- `paged_attn_decode_reduce.comp` — combine split-K partials per `(seq, q_head)`.
- `paged_cache_ops.glsl` — shared include: `element_block_index`, `k_load`, `v_load`, `kv_store` per cache type (F16 + turbo4_0), gated by `DATA_A_*`. This is the GLSL analog of `mt_pagedattn_ops.cuh`.

**Modified C++:**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — pipeline structs + registration, push-constant structs, `ggml_vk_paged_attn_mt` handler, `supports_op`, compute-graph dispatch.
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — register the new shader variants (F16 + turbo4_0 specializations).
- `tests/CMakeLists.txt` — add `test-paged-attn-vk` target.

**New test:**
- `tests/test-paged-attn-vk.cpp` — standalone dual-backend (Vulkan0 vs CUDA0) equivalence harness + 480-only scatter-readback oracle.

**Reused unchanged:** `types.glsl` (`block_turbo4_0`), `turbo_centroids.glsl` (`TURBO_CENTROIDS_4BIT`, `turbo_nearest_centroid_4bit`), `generic_binary_head.glsl` (indexing helpers), SP1's `init_pushconst_fastdiv` / fast-division helpers.

---

## Reference map (read these; do not re-derive)

| What | Where |
|---|---|
| Op construction, op_params, src[] order | `ggml/src/ggml.c` `ggml_paged_attn_mt` |
| CUDA dispatch entry | `ggml/src/ggml-cuda/ggml-cuda.cu:3414` → `mt::ggml_cuda_op_paged_attn_mt` |
| F16/turbo4_0 cache layout (k_load/v_load/element_block_index) | `ggml/src/ggml-cuda/mt_pagedattn_ops.cuh:37-155` |
| **No-RHT** turbo4_0 scatter (mirror this) | `ggml/src/ggml-cuda/mt_pagedattn.cu:323-455` |
| F16 scatter | `mt_scatter_kv_kernel` `mt_pagedattn.cu:174-225` |
| Prefill paged-attn kernel | `mt_paged_attention_kernel` in `mt_pagedattn.cu` |
| Split-K decode + constants (`CHUNK_KV=128`, `DECODE_NUM_THREADS=128`) | `ggml/src/ggml-cuda/mt_pagedattn_decode.cu:65-130` |
| turbo4_0 dequant element | `turbo4_dequant_element` `ggml/src/ggml-cuda/turbo-quant.cuh` |
| SP1 cooperative quantizer structure (the template, WITH WHT — strip the WHT) | `vulkan-shaders/cpy_f32_turbo4_0.comp` |
| SP1 turbo4 pipeline registration | `ggml-vulkan.cpp:4933, 4946` |
| SP1 supports_op turbo4 admission | `ggml-vulkan.cpp:16819, 16886, 16911` |
| FA handler structure to mirror | `ggml_vk_flash_attn` `ggml-vulkan.cpp:9932` |
| Graph paged branch (un-rotated Q, op call) | `src/llama-graph.cpp:2674-2760` (call at 2696) |
| Cross-backend compare primitive | `ggml_backend_compare_graph_backend` (`ggml-backend.h:423`) |

---

## Task 1: Capped CUDA+Vulkan build

**Goal:** A single capped build with both backends, so a test process sees `CUDA0` (1070) and `Vulkan0` (480) at once. This is the infrastructure the numeric oracle depends on.

**Files:**
- Modify: `build-vk.sh` (repo root, git-ignored)

**Interfaces:**
- Produces: a `build-vk/` configured with `-DGGML_VULKAN=ON -DGGML_CUDA=ON`, and `build-vk/bin/test-backend-ops` that enumerates both backends.

- [ ] **Step 1: Add a CUDA-enabling flag to the build wrapper.** Edit `build-vk.sh` so that an env toggle `WITH_CUDA=1` adds `-DGGML_CUDA=ON` to the configure line (keep CUDA OFF by default so the fast shader-only builds stay fast). The configure block becomes:

```bash
CUDA_FLAGS="-DGGML_CUDA=OFF"
if [ "${WITH_CUDA:-0}" = "1" ]; then
  CUDA_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61"   # GTX1070 = sm_61
fi
# ...in the cmake -S ... -B ... configure line, replace `-DGGML_CUDA=OFF` with ${CUDA_FLAGS}
```

Because CUDA TUs are RAM-heavy, when `WITH_CUDA=1` force a conservative job count: after the existing `jobs=` computation add `[ "${WITH_CUDA:-0}" = "1" ] && jobs=2`. Keep `MemoryMax`/`MemoryHigh` as-is (host-protective). If `build-vk/CMakeCache.txt` already exists with CUDA OFF, the step must delete it first when `WITH_CUDA=1` so the backend actually gets enabled:

```bash
if [ "${WITH_CUDA:-0}" = "1" ] && [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "GGML_CUDA:BOOL=ON" "$BUILD/CMakeCache.txt"; then
  echo "[build-vk] reconfiguring for CUDA (wiping stale CMakeCache)…"; rm -f "$BUILD/CMakeCache.txt"
fi
```

- [ ] **Step 2: Pre-flight memory check, then build test-backend-ops with CUDA on.**

Run: `free -h && cat /proc/loadavg`
Expected: ≥ 3000 MiB available (the wrapper aborts otherwise). If low, stop and tell the controller.

Run: `WITH_CUDA=1 bash build-vk.sh test-backend-ops`
Expected: configures with `GGML_CUDA:BOOL=ON`, compiles to completion (exit 0). This is SLOW (CUDA backend, -j2). Memory must stay > ~1 GiB free throughout (watch for thrash; if `MemAvailable` collapses, abort and reduce jobs).

- [ ] **Step 3: Verify both backends enumerate.**

Run: `./build-vk/bin/test-backend-ops -o ADD 2>&1 | grep -iE "Backend [0-9].*(CUDA|Vulkan)|POLARIS|1070"`
Expected: lists at least `CUDA0` = NVIDIA GTX 1070 AND `Vulkan0` = RADV POLARIS10. (`Vulkan1` may also be the 1070 via Vulkan — ignore it; the CUDA reference is `CUDA0`.)

- [ ] **Step 4: Verify ADD passes on both backends** (smoke that the dual-backend compare path is healthy).

Run: `./build-vk/bin/test-backend-ops test -o ADD -b Vulkan0 && ./build-vk/bin/test-backend-ops test -o ADD -b CUDA0`
Expected: both print `OK` for ADD cases.

- [ ] **Step 5: Commit.**

```bash
git add build-vk.sh
git commit -m "build(sp2): optional WITH_CUDA=1 dual-backend capped build for 1070-vs-480 oracle"
```
(Note: `build-vk.sh` is normally git-ignored; force-add is acceptable here since the team treats it as the canonical wrapper. If `git add` reports it ignored, use `git add -f build-vk.sh`.)

---

## Task 2: Dual-backend equivalence harness + turbo4 host oracle

**Goal:** A standalone `test-paged-attn-vk` that (a) builds a `PAGED_ATTN_MT` graph from seeded deterministic inputs, (b) compares `Vulkan0` against `CUDA0` via `ggml_backend_compare_graph_backend`, and (c) provides the 480-only scatter-readback oracle. Written test-first: it must compile and run, correctly reporting the op as **unsupported on Vulkan today** (the failing precondition that Tasks 3-5 turn green), while proving the compare path itself works on an already-supported op.

**Files:**
- Create: `tests/test-paged-attn-vk.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces (C ABI used by later tasks via the test): builds the op with `ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables, context_lens, q_lens, k_cur, v_cur, slot_mapping, block_size=16, n_kv_heads, scale)`.
- Consumes: `ggml_backend_compare_graph_backend` (`ggml-backend.h:423`), `ggml_backend_tensor_get/set` (`ggml-backend.h:92-93`).

- [ ] **Step 1: Add the test target to CMake.** In `tests/CMakeLists.txt`, mirror how `test-backend-ops` is registered. Add:

```cmake
llama_build_and_test(test-paged-attn-vk.cpp)
```
(Use the exact helper/macro the file already uses for `test-backend-ops.cpp`; match its argument style.)

- [ ] **Step 2: Write the harness — device selection + deterministic input gen + compare.** Create `tests/test-paged-attn-vk.cpp`. Complete content:

```cpp
// Dual-backend equivalence + scatter-oracle harness for GGML_OP_PAGED_ATTN_MT.
// Compares Vulkan0 (RX480) against CUDA0 (GTX1070) — the numeric oracle.
// turbo4_0 paged path is RHT-free: dequant = centroid*norm (un-rotated).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <string>

// Find a backend device by reg name ("Vulkan"/"CUDA") and device index within that reg.
static ggml_backend_t init_backend(const char * reg_substr) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (std::string(ggml_backend_reg_name(reg)).find(reg_substr) != std::string::npos) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// Deterministic fill: index-seeded, reproducible across processes/backends.
static void fill_f16(ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n = ggml_nelements(t);
    std::vector<ggml_fp16_t> buf(n);
    for (int64_t i = 0; i < n; ++i) buf[i] = ggml_fp32_to_fp16(dist(rng));
    ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
}
static void fill_i32(ggml_tensor * t, const std::vector<int32_t> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
}
```

- [ ] **Step 3: Add the graph builder.** Append a function that constructs ONE paged-attn case into a backend buffer. Use it for both backends (same seeds → identical inputs). Parameters chosen for the first equivalence case: 1 sequence, `head_dim=128`, `n_heads=8`, `n_kv_heads=2` (GQA 4:1), `block_size=16`, prefill `q_len=32`, `context_len=32`, cache type passed in. Complete content:

```cpp
struct paged_case {
    int head_dim, n_heads, n_kv_heads, block_size, q_len, ctx_len, n_seq;
    ggml_type cache_type;          // GGML_TYPE_F16 or GGML_TYPE_TURBO4_0
};

struct built_graph {
    ggml_context * ctx;
    ggml_cgraph  * gf;
    ggml_tensor  * out;
    ggml_tensor  * k_cache;
    ggml_tensor  * v_cache;
    ggml_backend_buffer_t buf;
};

static built_graph build_case(const paged_case & c, ggml_backend_t backend) {
    const int HD = c.head_dim;
    const int total_tokens = c.q_len * c.n_seq;
    const int max_blocks = (c.ctx_len + c.block_size - 1) / c.block_size;
    const int n_blocks_total = max_blocks * c.n_seq;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q       = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, c.n_heads,    total_tokens);
    ggml_tensor * k_cur   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, c.n_kv_heads, total_tokens);
    ggml_tensor * v_cur   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, c.n_kv_heads, total_tokens);
    // Paged cache: flat blocks of `cache_type`. Element count = n_blocks_total * block_size * n_kv_heads * HD.
    const int64_t cache_elts = (int64_t) n_blocks_total * c.block_size * c.n_kv_heads * HD;
    ggml_tensor * k_cache = ggml_new_tensor_1d(ctx, c.cache_type, cache_elts);
    ggml_tensor * v_cache = ggml_new_tensor_1d(ctx, c.cache_type, cache_elts);
    ggml_tensor * block_tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, c.n_seq);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seq);
    ggml_tensor * q_lens       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seq);
    ggml_tensor * slot_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_tokens);

    const float scale = 1.0f / sqrtf((float) HD);
    ggml_tensor * out = ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables,
                                           context_lens, q_lens, k_cur, v_cur, slot_mapping,
                                           c.block_size, c.n_kv_heads, scale);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Deterministic inputs (identical seeds on both backends).
    fill_f16(q, 1); fill_f16(k_cur, 2); fill_f16(v_cur, 3);
    // Cache must start zeroed (scatter writes only the touched slots; attention reads ctx_len of them).
    {
        std::vector<uint8_t> zeros(ggml_nbytes(k_cache), 0);
        ggml_backend_tensor_set(k_cache, zeros.data(), 0, ggml_nbytes(k_cache));
        ggml_backend_tensor_set(v_cache, zeros.data(), 0, ggml_nbytes(v_cache));
    }
    // Single-seq, contiguous slots: block_tables = [0,1,2,...]; slot_mapping = [0,1,...,q_len-1].
    std::vector<int32_t> bt(max_blocks * c.n_seq);
    for (int s = 0; s < c.n_seq; ++s) for (int b = 0; b < max_blocks; ++b) bt[s*max_blocks + b] = s*max_blocks + b;
    fill_i32(block_tables, bt);
    fill_i32(context_lens, std::vector<int32_t>(c.n_seq, c.ctx_len));
    fill_i32(q_lens,       std::vector<int32_t>(c.n_seq, c.q_len));
    std::vector<int32_t> slots(total_tokens);
    for (int i = 0; i < total_tokens; ++i) slots[i] = i;   // physical slot == logical pos for one seq
    fill_i32(slot_mapping, slots);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    return { ctx, gf, out, k_cache, v_cache, buf };
}
```

- [ ] **Step 4: Add `main` — smoke the compare path on a supported op, then the paged case.** Append:

```cpp
struct cb_state { double max_err = 0.0; bool any = false; };
static bool cmp_cb(int, const char *, const ggml_tensor * t1, const ggml_tensor * t2, void * ud) {
    auto * st = (cb_state *) ud;
    const int64_t n = ggml_nelements(t1);
    std::vector<float> a(n), b(n);
    ggml_backend_tensor_get(t1, a.data(), 0, ggml_nbytes(t1)); // works because test reads via to-float path
    ggml_backend_tensor_get(t2, b.data(), 0, ggml_nbytes(t2));
    for (int64_t i = 0; i < n; ++i) { double e = std::fabs((double)a[i]-(double)b[i]); if (e > st->max_err) st->max_err = e; }
    st->any = true;
    return true;
}

int main() {
    ggml_backend_t vk   = init_backend("Vulkan");
    ggml_backend_t cuda = init_backend("CUDA");
    if (!vk)   { printf("SKIP: no Vulkan backend\n");   return 0; }
    if (!cuda) { printf("SKIP: no CUDA backend (build with WITH_CUDA=1)\n"); return 0; }

    // Paged turbo4_0 prefill equivalence (the real gate — RED until Tasks 3-5 land).
    paged_case c { 128, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_0 };
    if (!ggml_backend_supports_op(vk, build_case(c, vk).out)) {   // pseudo: see note
        printf("EXPECTED-FAIL: PAGED_ATTN_MT not yet supported on Vulkan\n");
        return 0;   // not a hard failure until impl lands; controller treats this as the RED baseline
    }
    built_graph gvk = build_case(c, vk), gcu = build_case(c, cuda);
    cb_state st;
    std::vector<const ggml_tensor *> nodes = { gvk.out };
    bool ok = ggml_backend_compare_graph_backend(vk, cuda, gvk.gf, cmp_cb, &st, nodes.data(), nodes.size());
    const double tol = 5e-2;   // turbo4-class: 4-bit centroid quant. Tighten once measured.
    printf("paged turbo4_0 prefill: max_err=%.6f tol=%.6f %s\n", st.max_err, tol, (ok && st.max_err <= tol) ? "PASS" : "FAIL");
    return (ok && st.max_err <= tol) ? 0 : 1;
}
```

NOTE on the support check: `ggml_backend_supports_op` takes the op tensor and a backend. Build a throwaway op on the Vulkan buffer to query it (the snippet's `build_case(c, vk).out`), or query via `ggml_backend_dev_supports_op(dev, op)`. The implementer picks whichever compiles; the behavior required is: **if Vulkan does not support the op, print EXPECTED-FAIL and return 0; once supported, run the real comparison and return nonzero on mismatch.**

- [ ] **Step 5: Build and run — verify it reports EXPECTED-FAIL cleanly.**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: prints `EXPECTED-FAIL: PAGED_ATTN_MT not yet supported on Vulkan` and exits 0. (This is the RED baseline — the op compiles into the graph, both backends init, and Vulkan correctly declines the op.)

- [ ] **Step 6: Commit.**

```bash
git add tests/test-paged-attn-vk.cpp tests/CMakeLists.txt
git commit -m "test(sp2): dual-backend paged-attn equivalence harness (Vulkan0 vs CUDA0)"
```

---

## Task 3: F16 paged plumbing (handler + cache-ops include + scatter + prefill attention)

**Goal:** Get ALL the paged plumbing correct with the trivial F16 cache type — block-table gather, slot-mapping scatter, GQA, causal mask, online softmax — validated against CUDA's F16 paged path. This isolates plumbing bugs from quant bugs. The attention + scatter shaders are written **type-generic** here; Task 4 only adds the turbo4_0 cache-ops specialization.

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/paged_cache_ops.glsl`
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp`
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn.comp`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

**Interfaces:**
- Consumes: op `src[]` order and `op_params` from `ggml_paged_attn_mt` (`ggml.c`); F16 cache layout from `mt_pagedattn_ops.cuh:37-87`.
- Produces: `ggml_vk_paged_attn_mt(ggml_backend_vk_context*, vk_context&, const ggml_tensor* dst)` dispatched from the compute switch; `paged_cache_ops.glsl` helper functions `pa_k_off`/`pa_v_off`/`pa_k_load`/`pa_v_load`/`pa_k_store`/`pa_v_store` consumed by Task 4 (Task 4 adds the `DATA_A_TURBO4_0` branch of the same function names).

- [ ] **Step 1: Write the F16 cache-ops include.** Create `paged_cache_ops.glsl` with the F16 specialization, mirroring `mt_pagedattn_ops.cuh:37-87` EXACTLY. Provide functions guarded by `#ifdef DATA_A_F16`:

```glsl
// paged_cache_ops.glsl — GLSL analog of mt_pagedattn_ops.cuh. Cache-type load/store.
// Selected by DATA_A_F16 / DATA_A_TURBO4_0 (Task 4 adds the turbo4_0 block).

#ifdef DATA_A_F16
#define PA_KX 8u                                  // 16 / sizeof(f16)
// K: [HEAD_SIZE/KX, BLOCK_SIZE, KX]; off = base + (d/KX)*BS*KX + tok*KX + (d%KX)
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return ((paged_block*n_kv_heads + kv_head) * (HS/PA_KX) * BS * PA_KX)
         + (d/PA_KX)*BS*PA_KX + tok*PA_KX + (d%PA_KX);
}
// V: [HEAD_SIZE, BLOCK_SIZE]; off = base + d*BS + tok
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return ((paged_block*n_kv_heads + kv_head) * HS * BS) + d*BS + tok;
}
float pa_k_load(uint off) { return float(data_k[off]); }   // data_k = f16 cache buffer
float pa_v_load(uint off) { return float(data_v[off]); }
void  pa_k_store(uint off, float val) { data_k[off] = float16_t(val); }
void  pa_v_store(uint off, float val) { data_v[off] = float16_t(val); }
#endif
```

(The exact buffer names `data_k`/`data_v` must match the bindings declared in the shaders below. Adjust names to whatever the shaders bind.)

- [ ] **Step 2: Write the type-generic scatter shader (F16 path first).** Create `paged_attn_scatter.comp`. For F16 it is a pure permutation copy (no cooperative reduction needed — that's only for quant). Mirror `mt_scatter_kv_kernel` (`mt_pagedattn.cu:174-225`). Structure: `local_size_x = HEAD_SIZE` (128), one workgroup per `(token, kv_head)`, `gl_WorkGroupID.z` selects K vs V. Each thread handles one `d`. Read `k_cur`/`v_cur` at `token*n_kv_heads*HS + kv_head*HS + d`, resolve `slot=slot_mapping[token]` (skip if `<0`), write via `pa_k_store(pa_k_off(...))` / `pa_v_store(pa_v_off(...))`. Push constants: `HS, BS, n_kv_heads, n_tokens`. Bindings: `slot_mapping` (i32), `k_cur`/`v_cur` (f16 in), `data_k`/`data_v` (cache out). Include `paged_cache_ops.glsl`.

- [ ] **Step 3: Write the type-generic prefill attention shader (F16 path first).** Create `paged_attn.comp`. Translate `mt_paged_attention_kernel` (`mt_pagedattn.cu`), **replacing 32-lane warp collectives with shared-memory tree reductions over the 128-thread workgroup** (wave64-safe). Structure: `local_size_x = 128`; one workgroup per `(query_index, q_head)` (grid over `sum(q_lens) × n_heads`). Algorithm:
  - Map `q_head → kv_head = q_head / (n_heads / n_kv_heads)`.
  - Determine the query's sequence `s` and its absolute position `q_pos` from `q_lens` prefix sums (compute on the host side into a small `q_seq`/`q_pos` push param OR derive in-shader from `q_lens`; simplest: pass `n_seq==1` first, generalize after).
  - Online softmax over `kv_pos in [0, context_len)`: `logical_block = kv_pos / BS`, `paged_block = block_tables[s*max_blocks_per_seq + logical_block]`, `tok = kv_pos % BS`; load K[d] via `pa_k_load(pa_k_off(...))`; `score = scale * Σ_d Q[d]·K[d]` (shared-mem reduce); causal mask if `kv_pos > q_pos` (skip); update running max `m`, denom `l`, and accumulator `acc[d] += p · V[d]`.
  - Write `out[d] = acc[d]/l` as F16 to `dst` at `(d, q_head, query_index)`.
  - Push constants: `HS, BS, n_heads, n_kv_heads, max_blocks_per_seq, n_seq, scale` + fast-div magics if needed.

- [ ] **Step 4: Register the shader variants in the generator.** In `vulkan-shaders-gen.cpp`, near SP1's turbo4 registrations (~line 806), add the F16 variants:

```cpp
string_to_spv("paged_attn_scatter_f16", "paged_attn_scatter.comp", {{"DATA_A_F16","1"},{"D_TYPE","float16_t"}});
string_to_spv("paged_attn_f16",         "paged_attn.comp",         {{"DATA_A_F16","1"},{"D_TYPE","float16_t"}});
```

- [ ] **Step 5: Wire pipelines, push-constants, handler, supports_op, dispatch in `ggml-vulkan.cpp`.**
  - Declare pipeline members (mirror SP1's `pipeline_set_rows...[GGML_TYPE_TURBO4_0]` at 4946 and the FA pipelines). Add `vk_pipeline pipeline_paged_attn_scatter[2]` and `pipeline_paged_attn[2]` indexed by cache type (F16, TURBO4_0).
  - Define push-constant structs `vk_op_paged_scatter_pc` and `vk_op_paged_attn_pc` with the fields from Steps 2-3.
  - Register pipelines (mirror 4933): `ggml_vk_create_pipeline(... "paged_attn_scatter_f16", paged_attn_scatter_f16_len, paged_attn_scatter_f16_data, "main", <n_bindings>, sizeof(vk_op_paged_scatter_pc), {1,1,1}, {}, 1);` and likewise for `paged_attn_f16`.
  - Write `ggml_vk_paged_attn_mt(ctx, subctx, dst)`: read `op_params` (scale/block_size/max_blocks_per_seq/n_kv_heads); bind all 9 src tensors + dst; dispatch scatter (grid `{total_tokens, n_kv_heads, 2}`); insert a pipeline barrier (`ggml_vk_sync_buffers` / memory barrier so attention sees the scattered cache); dispatch prefill attention (grid `{sum(q_lens), n_heads, 1}`).
  - Add `GGML_OP_PAGED_ATTN_MT` to the compute-graph dispatch switch (near the `GGML_OP_FLASH_ATTN_EXT` case) → call the handler.
  - Add `GGML_OP_PAGED_ATTN_MT` to `ggml_backend_vk_device_supports_op`: require `src[1]->type == src[2]->type` and `== F16` (for this task; Task 4 adds TURBO4_0), `src[0]->type == F16`, `src[0]->ne[0] == 128` (head_dim), `block_size (op_params[1]) == 16`, index tensors I32. Return `!coopmat2` only if the shader path genuinely needs it; otherwise return true for the admitted shapes.

- [ ] **Step 6: Build.**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk`
Expected: compiles to completion (exit 0).

- [ ] **Step 7: Add an F16 equivalence case to the harness and verify it PASSES vs CUDA.** In `tests/test-paged-attn-vk.cpp` `main`, before the turbo4 case, add an F16 case `paged_case cf { 128, 8, 2, 16, 32, 32, 1, GGML_TYPE_F16 }` and run the same compare; tolerance `2e-3` (F16, no quant). Rebuild and run:

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: `paged f16 prefill: max_err=... tol=0.002000 PASS`. (turbo4 case still EXPECTED-FAIL until Task 4.)

- [ ] **Step 8: Commit.**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/paged_cache_ops.glsl \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        tests/test-paged-attn-vk.cpp
git commit -m "feat(sp2): F16 paged attention on Vulkan (plumbing + prefill, matches CUDA)"
```

---

## Task 4: turbo4_0 cache-ops (no-RHT scatter quantizer + dequant load)

**Goal:** Add the turbo4_0 specialization so the deliverable cache type works. The scatter becomes a cooperative no-RHT quantizer; the attention load becomes a dequant. Plumbing is unchanged from Task 3 (same shaders, new cache-ops branch + a cooperative scatter path).

**Files:**
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/paged_cache_ops.glsl`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Modify: `tests/test-paged-attn-vk.cpp`

**Interfaces:**
- Consumes: turbo4_0 layout `mt_pagedattn_ops.cuh:123-155`; no-RHT scatter `mt_pagedattn.cu:323-455`; `TURBO_CENTROIDS_4BIT` / `turbo_nearest_centroid_4bit` from `turbo_centroids.glsl`; `block_turbo4_0` from `types.glsl`.
- Produces: turbo4_0 pipelines `pipeline_paged_attn_scatter[GGML_TYPE_TURBO4_0]`, `pipeline_paged_attn[GGML_TYPE_TURBO4_0]`.

- [ ] **Step 1: Add the turbo4_0 load branch to `paged_cache_ops.glsl`.** Add an `#ifdef DATA_A_TURBO4_0` block mirroring `mt_pagedattn_ops.cuh:123-155`. `N_QBLK = HS/128`; `element_block_index = (paged_block*n_kv_heads + kv_head)*BS*N_QBLK + tok*N_QBLK + d/128`; `iqs = d % 128`; dequant = `TURBO_CENTROIDS_4BIT[nibble(qs, iqs)] * float(blocks[ib].norm)` (un-rotated). K and V identical. Provide `pa_k_load`/`pa_v_load` returning the dequant float; the cache buffers are bound as `block_turbo4_0` arrays.

- [ ] **Step 2: Add the cooperative no-RHT quantize path to the scatter shader.** Under `#ifdef DATA_A_TURBO4_0`, replace the F16 identity store with the cooperative quantizer mirroring `mt_scatter_kv_turbo4_0_kernel` (`mt_pagedattn.cu:323-455`): `local_size_x = 128`, one workgroup per `(token, kv_head, qblock)` with `gl_WorkGroupID.z` = K/V select. Steps (shared-mem, wave64-safe — use SP1's `cpy_f32_turbo4_0.comp` reductions as the structural template but **OMIT the s1/WHT-butterfly/s2 stage entirely**):
  1. Load `x[t] = float(src[src_off])` into shared mem.
  2. Tree-reduce L2 norm → `grp_norm`, `inv_norm`.
  3. Normalize `x[t] *= inv_norm`.
  4. **(NO Hadamard.)**
  5. `idx = turbo_nearest_centroid_4bit(x[t])`.
  6. Nibble-pack `qs[t/2]` cooperatively (even thread writes `lo | (hi<<4)`; barrier before reuse — SP1's WAR-race fix).
  7. Tree-reduce `Σ centroid[idx]²` → `recon_norm`; `corrected_norm = grp_norm / recon_norm`.
  8. Thread 0 writes `blocks[block_ib].norm = float16_t(corrected_norm)`, `.rnorm = 0`.
  Use `element_block_index` from `paged_cache_ops.glsl` for `block_ib`.

- [ ] **Step 3: Register turbo4_0 variants in the generator.** In `vulkan-shaders-gen.cpp` add:

```cpp
string_to_spv("paged_attn_scatter_turbo4_0", "paged_attn_scatter.comp", {{"DATA_A_TURBO4_0","1"},{"D_TYPE","float16_t"}});
string_to_spv("paged_attn_turbo4_0",         "paged_attn.comp",         {{"DATA_A_TURBO4_0","1"},{"D_TYPE","float16_t"}});
```

- [ ] **Step 4: Register turbo4_0 pipelines + admit in supports_op.** In `ggml-vulkan.cpp`: register the two turbo4_0 pipelines (mirror Task 3 Step 5, index `[GGML_TYPE_TURBO4_0]`); extend `supports_op` to also admit `src[1]->type == src[2]->type == GGML_TYPE_TURBO4_0` (same head_dim 128 / block_size 16 constraints). The handler already dispatches by cache type — verify it selects the turbo4_0 pipelines when `dst`'s K-cache is turbo4_0.

- [ ] **Step 5: Add the scatter-readback oracle to the harness (480-only, deterministic).** In `tests/test-paged-attn-vk.cpp`, add a function that, after running the op on Vulkan, reads back `k_cache` via `ggml_backend_tensor_get` and compares against a host computation that applies the no-RHT turbo4 quantizer (L2-norm → normalize → nearest-centroid → recon-norm-correct → pack) to the same `k_cur` at the expected `element_block_index(slot)`. Assert per-block `norm` within `1e-3` and all nibbles exactly equal. This needs no CUDA — pure determinism. Wire it to run for the turbo4 case.

- [ ] **Step 6: Build, run — turbo4 scatter oracle + turbo4 equivalence both green.**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: `scatter turbo4_0 readback: PASS`, `paged f16 prefill: ... PASS`, `paged turbo4_0 prefill: max_err=... tol=0.050000 PASS`. Investigate first if `max_err` is near the centroid-midpoint scale (~1e-6 deltas at idx 0,6,8,14 — the SP1 TURBO_MID_4BIT watch-item) vs a gross plumbing error.

- [ ] **Step 7: Commit.**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/paged_cache_ops.glsl \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_scatter.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        tests/test-paged-attn-vk.cpp
git commit -m "feat(sp2): turbo4_0 paged attention on Vulkan (no-RHT scatter + dequant, matches CUDA)"
```

---

## Task 5: Split-K decode + reduce

**Goal:** Add the split-K decode path (single-query-per-seq fast path) for long-context throughput, gated on equivalence vs CUDA's decode across context lengths. Prefill (Task 3-4) handles `q_len>1`; decode handles `q_len==1`.

**Files:**
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_decode.comp`
- Create: `ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_decode_reduce.comp`
- Modify: `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- Modify: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Modify: `tests/test-paged-attn-vk.cpp`

**Interfaces:**
- Consumes: `mt_pagedattn_decode.cu:65-130` (constants `CHUNK_KV=128`, `DECODE_NUM_THREADS=128`); the same `paged_cache_ops.glsl` loads.
- Produces: decode pipelines + a split-K partials scratch buffer sized `ceil(max_ctx/CHUNK_KV) · n_seq · n_heads · (HEAD_SIZE + 2)` floats.

- [ ] **Step 1: Write `paged_attn_decode.comp`.** Translate the decode kernel from `mt_pagedattn_decode.cu`. `local_size_x = 128`; one workgroup per `(seq, q_head, kv_chunk)` where `kv_chunk in [0, ceil(context_len/CHUNK_KV))`, `CHUNK_KV = 128`. Each workgroup computes a partial `(acc[HEAD_SIZE], m_partial, l_partial)` over its chunk via the same block-table gather + online softmax as prefill, writing partials to the scratch buffer. **Shared-mem reductions, not 32-lane shuffles.** Loads via `paged_cache_ops.glsl` (works for F16 and turbo4_0).

- [ ] **Step 2: Write `paged_attn_decode_reduce.comp`.** Adapt the existing `flash_attn_split_k_reduce.comp` pattern: one workgroup per `(seq, q_head)`, combine the `ceil(context_len/CHUNK_KV)` partials with the standard log-sum-exp merge (`m = max(m_i)`, `l = Σ l_i·exp(m_i−m)`, `out = Σ acc_i·exp(m_i−m) / l`), write F16 to `dst`. Bound the loop by `ceil(context_lens/CHUNK_KV)` (don't scan empty chunks).

- [ ] **Step 3: Register variants + pipelines.** In `vulkan-shaders-gen.cpp` add F16 and turbo4_0 variants of `paged_attn_decode` (the reduce is type-agnostic — one variant). In `ggml-vulkan.cpp` register the pipelines and allocate the partials scratch buffer (size from the op's max context).

- [ ] **Step 4: Route decode vs prefill in the handler.** In `ggml_vk_paged_attn_mt`, after scatter+barrier: if all `q_lens == 1` (decode), dispatch `paged_attn_decode` (grid `{n_seq, n_heads, n_splits}`) then `paged_attn_decode_reduce` (grid `{n_seq, n_heads, 1}`); else dispatch the prefill `paged_attn`. Reading `q_lens` host-side requires it be available — it is an input tensor; read its values via the already-uploaded buffer or pass `max_q_len`/`all_q_len_one` as op-derived dispatch params (compute from `q_lens` tensor data through `ggml_backend_tensor_get` on the input, or infer from `dst->ne[2]` vs `n_seq`).

- [ ] **Step 5: Add decode equivalence cases to the harness.** In `test-paged-attn-vk.cpp`, add decode cases (`q_len=1`) at several context lengths spanning chunk boundaries: `ctx_len ∈ {32, 128, 200, 512}` (200 forces a partial second chunk; 512 forces ≥4 chunks → exercises the reduce), turbo4_0 and F16. Same tolerances as Task 3-4.

- [ ] **Step 6: Build, run — all cases green.**

Run: `WITH_CUDA=1 bash build-vk.sh test-paged-attn-vk && ./build-vk/bin/test-paged-attn-vk`
Expected: every case (F16/turbo4 × prefill/decode × ctx lengths) prints PASS.

- [ ] **Step 7: Commit.**

```bash
git add ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_decode.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/paged_attn_decode_reduce.comp \
        ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp \
        ggml/src/ggml-vulkan/ggml-vulkan.cpp \
        tests/test-paged-attn-vk.cpp
git commit -m "feat(sp2): split-K decode for Vulkan paged attention (matches CUDA across ctx lengths)"
```

---

## Task 6: Perf + end-to-end validation (INFERENCE GATE)

**Goal:** Confirm the Vulkan paged turbo4 path is perf-competitive and coherent end-to-end on the 480. Per the SP1 lesson, op-correctness tests do not catch perf cliffs — benchmark before declaring done.

**EVERY step here runs inference and REQUIRES explicit user go-ahead before running. Do not run autonomously.**

**Files:**
- Modify: `docs/superpowers/plans/2026-06-29-sp2-turbo4-paged-attn-vulkan.md` (append a Results section)
- Modify: `.superpowers/sdd/progress.md` (ledger)

- [ ] **Step 1: Profile the op microbenchmark.** With user go-ahead, run `GGML_VK_PERF_LOGGER=1 ./build-vk/bin/test-paged-attn-vk` and record per-op µs for scatter / prefill / decode / reduce. A cooperative shader should show scatter in the low-µs range (compare to SP1's 9.2 µs set_rows). If any op is 10×+ slower than its CUDA counterpart, treat as a cliff and diagnose (register spill, missing barrier, serial reduction) — do NOT excuse it.

- [ ] **Step 2: End-to-end coherence + PPL.** With user go-ahead, run LFM2.5 (or the agreed model) on `Vulkan0` with the paged turbo4 cache enabled (`--kv-tier-paged-blocks` / the flags that route to `llama_kv_cache_paged` + turbo4), and compare perplexity at `-c 512` against the CUDA/ROCm paged turbo4 baseline. Expected: within noise (SP1 saw 32.73 vs 33.08 f16). Record exact commands and numbers.

- [ ] **Step 3: Throughput.** With user go-ahead, `llama-bench` pp512/tg128 on `Vulkan0`, paged turbo4. Compare to SP1's non-paged turbo4 (pp512 ≈ 560, tg128 ≈ 88) and to the ROCm paged numbers. Record.

- [ ] **Step 4: Write the Results section** (parity table, perf table, any cliff+fix story) into this plan doc, and mark SP2 complete in `.superpowers/sdd/progress.md` with the commit range.

- [ ] **Step 5: Commit.**

```bash
git add docs/superpowers/plans/2026-06-29-sp2-turbo4-paged-attn-vulkan.md .superpowers/sdd/progress.md
git commit -m "docs(sp2): paged attention Vulkan results + ledger (SP2 complete)"
```

---

## Notes for the implementer

- **GLSL shader bodies are translations, not inventions.** Each shader names the exact CUDA reference (and SP1 shader) to translate from. The correctness gate is the CUDA-equivalence harness, not transcription fidelity — when a case fails, diff the math against the named CUDA lines.
- **wave64.** Every reduction is shared-memory tree reduction over 128 threads. If you see `__shfl_*_sync(..., 32)` in the CUDA reference, that is a 32-lane warp op — do NOT port it as a 32-lane subgroup op; reduce across the whole 128-thread block in shared memory. This is the single most common porting bug for this hardware.
- **Barriers.** Insert a `barrier()` before any shared-memory slot is overwritten and re-read (SP1's WAR-race fix). Insert a pipeline/buffer barrier in the handler between scatter and attention.
- **The turbo4 midpoint delta** (~1e-6 at centroid idx 0,6,8,14) is the first suspect for small turbo4 equivalence drift — it is CUDA-parity-correct, so if `max_err` is tiny and localized, widen tolerance rather than "fixing" it.
- **Never stage the pre-existing CUDA WIP files** (`ggml/src/ggml-cuda/mt_pagedattn*`, `common.cuh`) — they are not SP2.
```
