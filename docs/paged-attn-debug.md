# mt:: paged attention — debug investigation notes

This file captures what we know about the current state of the
`--kv-tier-paged-blocks` path and the four interlocking bugs that
prevent it from being deployable for the agent army workload.
Written 2026-05-08 after a session that uncovered all four; first
debug session that picks this up should start here.

## TL;DR — order to fix things

1. **MAD-114**: Paged kernel produces garbage attention output on
   Qwen3.5 hybrid models. ← **start here**, everything else depends
   on a known-good baseline
2. **MAD-116**: Add quantized-KV (turbo4 / Q4_0) support to the kernel
3. **MAD-115**: Chunked online-softmax for ctx > ~16k per attention call
4. **MAD-117**: Re-apply tier-aware pool sizing in `src/llama-model.cpp`

After all four land, the original target (army on 8 GB consumer GPUs
with `--parallel 4 --kv-tiered 25,25,50 -c 524288 --kv-tier-paged-blocks
--cache-type-k turbo4 --cache-type-v turbo4`) should actually fit
and work.

## Repro recipes

### 1. Paged garbage on hybrid (MAD-114)

```bash
./build/bin/llama-server \
  -m /home/kmbandy/Downloads/Qwen3.5-4B-UD-IQ3_XXS.gguf \
  --no-mmap --device ROCm0 --parallel 1 \
  --kv-tier-paged-blocks \
  -c 4096 -ngl 99 --port 18110

curl -s -X POST http://127.0.0.1:18110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Reply with just 4."}],"max_tokens":20,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

→ output is `?????…`. Drop `--kv-tier-paged-blocks` → returns `4`.

### 2. LDS overflow on long ctx (MAD-115)

```bash
./build/bin/llama-server \
  -m <any model that hits paged path> \
  --no-mmap --device ROCm0 --parallel 4 \
  --kv-tier-paged-blocks \
  -c 524288 -ngl 99 --port 18091
```

After the LDS guard added in commit 44301989, this aborts with a
clear log line:

```
mt::paged_attn: requested smem 524444 B exceeds 64 KiB LDS limit
(max_ctx_len=131072). Reduce -c/--parallel so n_ctx_seq * 4 bytes
fits, or drop --kv-tier-paged-blocks.
```

Pre-fix, this manifested as a generic `cudaErrorInvalidArgument`.

### 3. Tier % does NOT shrink VRAM (MAD-117)

Look at `common_memory_breakdown_print` output for any
`--kv-tiered 25,25,50` launch. The "context" column is sized for
the full `n_ctx_seq`, not 25% of it. The tier wrapper currently
adds software-only eviction triggers on top of a fully-allocated
inner cache.

## What we ruled out for the hybrid garbage bug (MAD-114)

| Hypothesis | Verdict | How |
|---|---|---|
| Multi-seq compute_slot_mapping bug from this week's changes | Ruled out | Reproduces with all of tonight's commits reverted in working tree |
| Layer filter (skip recurrent) corrupting K/V allocation | Ruled out | Repros without filter (the filter is correct — `qwen35.cpp:123` only calls `build_attn` for non-recurrent layers; recurrent ones go through `build_layer_attn_linear` and never touch the paged cache) |
| Type parameterization (commit 4d55540) breaking F16 path | Ruled out | Default args = exactly old behavior; reverting that commit also produces garbage |
| LDS / smem overflow | Ruled out | Repros at small ctx (4096) where smem is ~16 KiB |
| Compute-slot-mapping returning wrong slot | Probably ruled out | Same model returns correct output on the standard non-paged path which uses different slot resolution; per-seq smoke at parallel=4 shows expected metadata |
| Recent ggml-cuda upstream changes | Unknown | Worth a `git bisect` between Phase 3.5 (last known good non-hybrid paged) and current |

## Likely culprits for MAD-114, ranked

### 1. mt_reshape_and_cache scatter layout assumption

`ggml/src/ggml-cuda/mt_pagedattn.cu:427` — the kernel comment
documents `k_cur` as `[head_dim, n_kv_heads, n_tokens]` (head_dim
fastest), and indexes:

```cpp
const int src_idx = token_idx * (n_kv_heads * HEAD_SIZE) +
                    head_idx  * HEAD_SIZE +
                    dim_idx;
```

But post-RoPE K from `build_attn` may be `[head_dim, n_tokens,
n_kv_heads]` for hybrid models (different permute order than
Llama-style attention). If the actual shape is the second one, this
indexing reads the wrong source bytes, producing scrambled K cache,
producing scrambled attention output.

**To check**: in `src/llama-graph.cpp` paged dispatch (~line 2335),
log `k_cast->ne[0..3]` and `k_cast->nb[0..3]` for the first batch
on Qwen3.5. Compare against what the kernel assumes.

### 2. Kernel template instantiation for HEAD_SIZE=256, n_kv_heads=4

Phase 3.5's "live smoke against non-hybrid model (paged)" used a
non-hybrid model — likely a Llama-arch with `HEAD_SIZE=128` and
larger `n_kv_heads`. Qwen3.5 uses `HEAD_SIZE=256, n_kv_heads=4`.

The kernel templates on `HEAD_SIZE` and `BLOCK_SIZE`. Look at the
explicit instantiations in the dispatch (`mt_pagedattn.cu:386+`):

```cpp
if (head_size == 128 && block_size == 16) { run(<128, 16>); }
else if (head_size == 64 && block_size == 16) { run(<64, 16>); }
else if (head_size == 256 && block_size == 16) { run(<256, 16>); }
```

The 256 path exists. But — `K_X = 16 / sizeof(__half) = 8`, and
`HEAD_SIZE % K_X == 0` is asserted; 256 % 8 = 0 OK. `VEC_PER_THREAD
= ceil(256 / 128) = 2`. All those align.

Could still be a layout-math bug specific to the 256/4/16 combo.
Worth a focused inspection.

### 3. build_attn dispatch in src/llama-graph.cpp paged path

Line 2335 onward. The `to_f16_cont(k_cur)` cast/contiguous may not
produce the exact shape the kernel expects, because for hybrid
models Q/K/V come from `build_layer_attn` with a different shape
than the standard llama_attn (which Phase 3.5's non-hybrid model
exercised).

## What works (don't break it)

* Multi-seq tiered cache (commit 81e0b3e2a) — `--parallel 4 --kv-tiered`
  WITHOUT `--kv-tier-paged-blocks` works correctly. Smoke this
  session passed: 4 concurrent OAI requests, all 200 OK, slot 1 returned
  the correct content; per-seq tier metadata isolated correctly.
* Multi-seq paged compute_slot_mapping (commit 7ef076e20) — block
  table routes per seq via `ubatch->seq_id[i][0]`. Math-only test
  passes.
* Layer filter + types parameterization (commit 4d55540a0) — paged
  cache allocation drops 4× on hybrid (8/32 attn layers vs all 32
  pre-fix).
* LDS guard (commit 44301989d) — surfaces clear error instead of
  cryptic `cudaErrorInvalidArgument`.

## Tools for tomorrow's debug

1. **CPU reference attention** — there's a stub mention of
   `Phase 3.2: ggml op + dispatch + CPU reference test` in the
   in-conversation task list. Check `tests/test-backend-ops.cpp` or
   adjacent for the CPU reference and use it to diff GPU vs CPU
   for a single Qwen3.5 attention layer. That's the fastest way to
   isolate kernel correctness from graph correctness.

2. **HIP_LAUNCH_BLOCKING=1** + **AMD_LOG_LEVEL=4** environment vars
   to get more diagnostic from ROCm runtime if anything fails inside
   the kernel.

3. **rocgdb** can step into the kernel for the offending model. Slow
   but conclusive.

## Don't waste time on

* Trying to make the existing 64-KiB LDS limit work via clever
  packing — chunked attention is the right fix.
* Reverting the multi-seq commits — they aren't the cause.
* Rewriting from scratch — the existing kernel is mostly right; it's
  one or two layout / shape assumption bugs.

---

## 2026-06-15 session — non-WMMA prefill, gfx803 perf, checkpoint+paged, tool bleed

Investigation on **mad-lab-2026** (GTX 1070 / sm_61 / CUDA `build-army`;
RX 480 / gfx803 / HIP `build-rocm-gfx803` in docker `rx480-army`).

### A. Paged PREFILL was on the slow scalar kernel for ALL non-WMMA GPUs (fixed)

`nsys` on a 2501-tok paged prefill (1070): `mt_paged_attention_kernel`
= **84.7% of GPU time, 623 ms/launch**, grid `n_heads×num_seqs` (NO query
parallelism). Root cause: the fast query-tiled prefill kernel
(`mt_pagedattn_tile.cu`, grid `n_heads×num_seqs×n_q_tiles`) is gated at
`mt_pagedattn.cu` behind `tile_gate_on = wmma_ok && tile_env_on`, and
`wmma_ok = amd_wmma_available(cc)` is **RDNA3/4-only**. So Pascal (1070)
and GCN/gfx803 (480) both fall through to the scalar O(n_q·n_kv) kernel.
Decode was unaffected (`launch_paged_attn_decode` is not WMMA-gated).

**Fix (uncommitted, working tree):**
1. Added a portable FMA `#else` body to `mt_paged_attention_tile_kernel`
   (was `NO_DEVICE_CODE`): QK^T as smem dot-products, scores→`smem_s`,
   scores·V as smem GEMM. Same lane layout as the WMMA path; shfl
   reductions pass explicit `WARP_SIZE` (correct on gfx803 64-lane waves).
   Footprint ~13 KiB @ HS128 (fits Pascal's 48 KiB).
2. Dispatch: `tile_gate_on = tile_env_on` (drop `wmma_ok`); force
   `mw_on=false` on non-WMMA (multi-warp tile still WMMA-only).
3. **Also fixed a latent grid-sizing bug** (same class as the decode
   `93a1e6a11` fix): the tile launches sized `n_q_tiles` from
   `avg_q_len = total_q_tokens/num_seqs`, which floors with idle
   `--parallel` slots (1 active + N idle) → high Q-tiles never launch →
   uncomputed rows → token-salad. Now use `total_q_tokens`. **This also
   fixes the WMMA path and is very likely the core of MAD-288.**

**Verified:** 1070 (CUDA) paged prefill **160 → 328 t/s** (non-paged ref
395), output byte-matches the scalar oracle (`GGML_PAGED_TILE=0`) for
Qwen3.5-9B (HS128) and LFM2.5-8B-A1B (HS64 **padded to 128** in the paged
cache → uses the tile path; no separate HS64 work needed — clarifies
MAD-298). 480 (gfx803): output correct, no crash. Revert: `GGML_PAGED_TILE=0`.

### B. gfx803 (RX 480) ROCm is ~10× slower than Vulkan — and it's NOT the attn kernel

Same card, Qwen3.5-9B, `-p512 -n128`:

| Path | Prefill | Decode |
|---|---|---|
| gfx803 ROCm/HIP, paged tile | 16.6 t/s | — |
| gfx803 ROCm/HIP, paged scalar | 13.8 t/s | — |
| **480 via Vulkan0 (RADV, f16 KV)** | **162 t/s** | 20.4 t/s |
| 1070 via Vulkan | 181 t/s | — |

Tile ≈ scalar on gfx803 (both ~16) → the bottleneck is **gfx803 ROCm/HIP
itself**, not the attention kernel and not the Polaris silicon. Suspects:
turbo4 per-element dequant in `stage_k/v_tile` (`ops::k_load`), paged-block
gather, or `gated_delta_net` hybrid layers under HIP. **Decision (Kurtis):
keep the custom paged/turbo4/tiered features on gfx803, find the ROCm perf
bug — do NOT port to Vulkan** (Vulkan lacks all custom kernels; only viable
for standard dense models that fit VRAM). NEXT: profile the gfx803 paged
prefill (rocprof not installed in `rx480-army` — needs a profiling path).

### C. Checkpoint reuse vs paged cache — block-alignment crash (NOT fixed)

`--ctx-checkpoints>0` would let hybrid/recurrent models reuse the prompt
prefix (the restore path at `server-context.cpp` ~3397 already exists, and
with paged the checkpoints are tiny — **0.282 MiB**, blocks referenced not
copied). But restore sets `n_past` to the checkpoint pos (e.g. 1364), and
`llama_kv_cache_paged::seq_rm` requires **block-aligned (×16)** ranges →
unaligned trim leaves stale partial blocks → `GGML_ASSERT compute_slot_mapping
failed` abort. This is why LFM2.5 runs `--ctx-checkpoints 0` and full-
reprocesses every turn (~2.5 min for ~3700 tok). **Two fix attempts failed:**
round-at-restore is wrong (recurrent state can't be partially rolled back —
must reuse exactly `pos_max+1`); block-aligning the checkpoint batch-break at
creation crashed turn-1 prefill. **Reverted.** Needs a deeper design (paged
`seq_rm`/`compute_slot_mapping` tolerant of the boundary, or a hybrid+paged-
aware checkpoint path). Upstream unmerged references: PRs #20955, #21099, #20428.

### D. Tool-schema bleed (LFM2.5)

"hello!" → model regurgitates tool-schema JSON fragments then hallucinates.
The LFM2.5 jinja template renders the tool list as a **plain-text system
message** `List of tools: [{json}]` (faithful to the official template;
`common/chat.cpp:2226` detects this variant, vs LFM2's native
`<|tool_list_start|>` tokens at :622). Tool *calls* are correct
(grammar-constrained `<|tool_call_start|>`). With `thinking=1` the model
also reasons verbosely about the tools. Fix TBD (limit/disable reasoning on
tool turns, or revisit tool presentation). Single/multi-turn tool calls
otherwise correct on 1070.
