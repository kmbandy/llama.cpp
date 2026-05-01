# Memory-Tier + Weight-Pager Bug Catalog

This document is the reference of bugs *already fixed* in the existing
implementations of the tiered KV cache and the NVMe weight pager. It is
authored before the rewrite (Phase 0 of the rewrite plan) so the rewritten
subsystems can re-pay each fix without re-discovering the bug.

References point to the live code on branch `feat/memory-tier-rewrite-v2`
at HEAD `65fb0a5c3`. After Phase 1.0/2.0 deletes those files, this document
plus the listed commit hashes is the only remaining record.

The plan that drives this work is at
`/home/kmbandy/.claude/plans/proud-stargazing-engelbart.md`.

---

## Tiered KV cache bugs

### B-K1 — Quantized K/V row corruption from wrong stride
**Component:** tier mover (hot ↔ warm copy)
**Symptom:** silent corruption of K/V data for block-quantized cache types
(`turbo4`, `Q4_K`, `Q5_K`, etc.). Manifested as "557056 MiB" misreads and
silent `hipMalloc` failures sized as if every element were one block.
**Root cause:** per-token byte size was computed as
`tensor->ne[0] * ggml_element_size(tensor)`. For block-quantized types,
`ggml_element_size` returns the *block* size, not bytes-per-element, so this
expression returns the size of `ne[0]` *blocks* rather than the size of one
row of elements.
**Fix:** use `tensor->nb[1]` (the actual row stride), which ggml computes
correctly for every type.
**Reference:** `src/llama-kv-cache-tiered.cpp:658-663`
**Introduced/fixed in:** commit `786f5019a`
**Test that locks it in (Phase 2):** unit in `mt-mover-attn` that constructs
a quantized layer and asserts `mover.row_bytes(layer) == layer->nb[1]`.

---

### B-K2 — Warm GPU tier OOM (RAM staging double-allocated)
**Component:** tier warm-buffer allocator
**Symptom:** OOM on host RAM with `--kv-warm-device 1` (warm tier on the
6900XT eGPU). At 65k ctx the unconditional RAM staging buffers cost ~8 GB
per session in addition to the GPU warm buffers that were the actual target.
**Root cause:** the warm-buffer allocator allocated `n_warm * k_bytes` and
`n_warm * v_bytes` of host RAM unconditionally, then *additionally* allocated
the GPU warm buffers when `warm_device >= 0`.
**Fix:** under `GGML_USE_HIP`, skip the RAM staging allocation when
`config.warm_device >= 0`. The GPU path (`warm_k_dev` / `warm_v_dev`) is the
only buffer needed in that mode.
**Reference:** `src/llama-kv-cache-tiered.cpp:680-688`
**Introduced/fixed in:** commit `786f5019a`
**Test that locks it in (Phase 2):** unit in `mt-mover-attn` that asserts
`mover` allocated zero host RAM staging when `warm_device >= 0` and that
`hipMalloc` was called for `warm_k_dev` / `warm_v_dev` instead.

---

### B-K3 — Hybrid model attention KV not detected
**Component:** server-side tier wiring
**Symptom:** for hybrid models (Qwen3.5/Qwen3.5-MoE/Qwen3-Next style:
attention + recurrent), the tier system silently ran in metadata-only mode
because the live memory backend was a `llama_memory_hybrid_iswa`, not a
`llama_kv_cache`. No K/V data movement happened despite `--kv-tiered`.
**Root cause:** `init_slot` did `dynamic_cast<llama_kv_cache*>(mem)`, which
returns null for hybrid wrappers. No fallback existed for hybrid memory
types.
**Fix:** three-way `dynamic_cast` chain — `llama_kv_cache`, then
`llama_memory_hybrid` (Qwen3-27B style; `hybrid->get_mem_attn()`), then
`llama_memory_hybrid_iswa` (97B/REAP style;
`hybrid_iswa->get_mem_attn()->get_base()`).
**Reference:** `tools/server/server-tiered-cache.cpp:195-209`
**Introduced/fixed in:** commit `786f5019a` (initial fix), refined later.
**Replaced in rewrite (Phase 2):** the `dynamic_cast` chain is removed.
A new `get_tier_view()` virtual method on `llama_memory_i` returns a
narrow accessor; each cache subclass overrides it. CI gate: zero
`dynamic_cast` in the new tier code.
**Test that locks it in (Phase 2):** unit asserting that
`llama_memory_hybrid_iswa::get_tier_view()` returns a composite view
referencing both attention and recurrent inner views.

---

### B-K4 — Stale checkpoint restore after context shift
**Component:** server slot checkpoint management
**Symptom:** GPU memory access faults on multi-GPU ROCm setups during
`llama_state_seq_set_data_ext` after a slot's context had shifted.
**Root cause:** `slot.prompt.checkpoints` retained pre-shift positions; a
later checkpoint restore used those stale positions to seed the live KV
cache.
**Fix:** `slot.prompt.checkpoints.clear()` immediately after the
context-shift truncation block.
**Reference:** `tools/server/server-context.cpp:2288` (single line)
**Introduced/fixed in:** commit `b25eddd1b`
**Status in rewrite:** **KEEP** — fix is server-side and orthogonal to
the rewrite. Phase 2 must not regress this. Acceptance: a regression test
that performs a context shift on a long prompt and then triggers a
checkpoint restore on the same slot.

---

### B-K5 — Recurrent state invisible to tier system *(no fix in old code)*
**Component:** tier system, hybrid-model path
**Symptom:** for hybrid models, only attention KV layers participate in
tiering. Recurrent state (gated-delta-net / SSM) stays fully resident in
VRAM. For 97B at 65k hot ctx this is ~149 MiB per session that cannot be
evicted. The user accepts this behavior today but it limits the value of
tiering on hybrid models.
**Root cause:** the tier system iterates layers via `get_layer_k_raw` /
`get_layer_v_raw`, which exist only on `llama_kv_cache`. Recurrent state
is held by `llama_memory_recurrent` and exposes per-sequence `r` / `s`
buffers, not per-token K/V.
**Fix in old code:** none — recurrent layers are silently skipped.
**Status in rewrite:** **NEW WORK** in Phase 2. The plan introduces a
`RecurrentStateMover` with per-sequence (not per-token) granularity and a
new `KVTC_SECTION_RECURRENT` cold-tier section type. Eager-reload is
performed before `init_batch` returns when a tiered-out sequence is
referenced.
**Test that locks it in (Phase 2):** golden test recording per-step
`r`/`s` state for a fixed prompt with no tiering, then replaying with
forced tier migration and asserting byte equality after restore.

---

### B-K6 — Mixed-arch ROCm peer-copy fault (gfx1030 + gfx1201)
**Component:** ggml-cuda cross-device copy paths
**Symptom:** GPU page faults on multi-GPU ROCm setups mixing
RDNA2 (gfx1030, RX 6900 XT) and RDNA4 (gfx1201, R9700) when the scheduler
attempted cross-device tensor copies (split-mode row, MoE expert routing,
KV cache movement, gated-delta-net state cache).
**Root cause:** `hipMemcpyPeerAsync` and `cudaMemcpy3DPeerAsync` use an
internal copy kernel. On a mixed-architecture build, that kernel is not
compiled for both ISAs simultaneously; the runtime selects the wrong
binary or fails asynchronously with `hipErrorNoBinaryForGpu`. ROCm
non-XGMI PCIe P2P additionally uses CPU virtual addresses as GPU peer
addresses, which causes page faults regardless of the kernel selection.
**Fix:** replace all cross-device peer-copy paths with explicit host
staging through a cached `hipHostMalloc` buffer (grow-only, reused across
calls). The HIP path of `ggml_backend_cuda_buffer_cpy_tensor`,
`ggml_backend_cuda_cpy_tensor_async`, `ggml_cuda_cpy_tensor_2d`, and
`ggml_cuda_Memcpy2DPeerAsync` all funnel through this staging path.
P2P remains available behind `GGML_CUDA_P2P=1` for hardware where it
works (XGMI).
**References:** `ggml/src/ggml-cuda/ggml-cuda.cu:712-732` (cpy_tensor),
`ggml/src/ggml-cuda/ggml-cuda.cu:1010-1022` (split_buffer_set_tensor
explicit per-device stream), `ggml/src/ggml-cuda/ggml-cuda.cu:1448-1466`
(cpy_tensor_2d host staging), `ggml/src/ggml-cuda/ggml-cuda.cu:1685-1750`
(`ggml_cuda_Memcpy2DPeerAsync` host-staging path)
**Introduced/fixed in:** commits `4424bbe70` and `65fb0a5c3` (cherry-picks
from branch `rocm-multi-gpu-copy-opt`; original commits `5f88e0861` and
`e379f638c`).
**Status in rewrite:** **KEEP** — fix is in `ggml-cuda.cu`, orthogonal to
both rewritten subsystems. Phase 3 (row-split optimization) must not
disturb this code.

---

### B-K7 — `ggml_cuda_set_device` early-return (suspected 97B regression)
**Component:** ggml-cuda device routing
**Symptom (suspected):** Qwen3.5-REAP-97B-A10B at ctx 131072 generates only
token_id 14 (`/`) from the first decode token. Independent of `--kv-tiered`.
35B (`qwen3moe`, no delta-net) is unaffected.
**Root cause:** unknown. The early-return optimization in
`ggml_cuda_set_device` was removed because `hipGetDevice` returned
unexpected values on threads with uninitialized device context, causing
`hipErrorIllegalAddress` during KV cache checkpoint restore on multi-GPU
ROCm setups. The removal is correct in principle. The 97B failure is
*correlated with* this commit per the prior bisection but the actual
mechanism has not been confirmed.
**Fix in old code:** kept — the early-return removal stays.
**Reference:** `ggml/src/ggml-cuda/ggml-cuda.cu:102-106`
**Introduced/fixed in:** commit `16d66aadf`
**Status in rewrite:** **KEEP**. Phase 0.5 gate (run 97B at 131k with all
fork subsystems disabled) is the test. If 97B is still broken with the
gate disabled, the bug is not in our subsystems; halt and re-investigate
this commit. If 97B is coherent at the gate, the regression is downstream
in our tier/pager code and the rewrite has a chance of fixing it
incidentally.

---

## Weight pager bugs

### B-P1 — View-tensor sentinel from gallocr
**Component:** weight-pager eval callback
**Symptom:** views of paged-in weights read garbage data because their
`tensor->data` pointer was a gallocr sentinel value (`(char*)1 + view_offs`),
not a real address.
**Root cause:** `ggml_gallocr_alloc_graph` initialises view tensors with
`data = (char*)1 + view_offs` because it does not know the real address
yet (paged weights have no static address). Every op that consumes such a
view must have its `data` overwritten with the real per-step address before
the op runs.
**Fix:** in the pager eval callback, walk `t->src[]`; for each src that is
a view of a tracked weight (`src->view_src` matches a paged tensor name),
overwrite `src->data = vram + src->view_offs` and
`src->buffer = pool.ggml_buf` before the op runs.
**Reference:** `src/llama-weight-pager.cpp:39-67` (detection),
`src/llama-weight-pager.cpp:85-108` (rewrite)
**Introduced/fixed in:** commit `ff060471c` (Phases 1-4) + `786f5019a`
(buffer assignment correctness).
**Test that locks it in (Phase 1):** unit in `wp-eval-cb` that builds a
synthetic graph with `ggml_view_2d` of a paged tensor and asserts the view's
`data` is overwritten before `ggml_graph_compute`.

---

### B-P2 — Quantized kernel reads past tensor data (uninitialized padding)
**Component:** weight-pager `page_in`
**Symptom:** silent miscompute on quantized matmul kernels (turbo4, IQ3_*)
when the tensor's byte size is smaller than the pool slot size.
**Root cause:** quantized kernels routinely read past the end of the tensor
into padding bytes. If a slot is reused across pages and the new page is
smaller than the previous one, the tail bytes contain stale data that the
kernel mis-interprets.
**Fix:** after `hipMemcpy(slot, pinned_staging, page.size)`, zero the
remaining `pool.slot_size - page.size` bytes with `hipMemset`.
**Reference:** `src/llama-weight-pager.cpp:368-371`
**Introduced/fixed in:** commit `786f5019a`
**Test that locks it in (Phase 1):** unit in `wp-gpu-transport` that pages
in a small tensor into a slot previously holding a larger one and asserts
the tail bytes are zero.

---

### B-P3 — `O_DIRECT` inherited on dup'd fds
**Component:** weight-pager file-descriptor setup
**Symptom:** silent reads from rounded-down offsets on filesystems that
honour `O_DIRECT` strict alignment. GGUF tensor offsets are not
sector-aligned, so the read returned data from the prior 512-byte boundary
instead of the requested offset.
**Root cause:** the model loader opens GGUF files with `O_DIRECT` (when
direct I/O is available). The pager `dup`s those fds for its own use; `dup`
preserves the file flags including `O_DIRECT`. The pager's `pread` does not
honour sector alignment.
**Fix:** after `dup`, fetch the fd's flags with `fcntl(fd, F_GETFL)` and
clear `O_DIRECT` with `fcntl(fd, F_SETFL, fl & ~O_DIRECT)`.
**Reference:** `src/llama.cpp:152-161`
**Introduced/fixed in:** commit `786f5019a`
**Test that locks it in (Phase 1):** unit in `wp-file-io` that mocks an fd
opened with `O_DIRECT` and asserts the layer clears the flag before issuing
its first read.

---

### B-P4 — `tensor->buffer` null on callback (raw `hipMalloc` pool)
**Component:** weight-pager pool allocator
**Symptom:** segfault inside `ggml_cuda_mul_mat` because it asserted that
`tensor->buffer->buft == ggml_backend_cuda_buffer_type(device)`. The pager
had set `tensor->data` to a raw `hipMalloc` pointer with `tensor->buffer`
unchanged, which on first use was null or stale.
**Root cause:** the pool was a raw `hipMalloc` block; there was no
corresponding `ggml_backend_buffer_t` to assign to `tensor->buffer`.
**Fix:** allocate the pool through ggml's CUDA buffer-type interface
(`ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device),
n_slots * slot_size)`); store the resulting `ggml_backend_buffer_t` in
`pool.ggml_buf`; on every page-in, set `src->buffer = pool.ggml_buf` along
with `src->data`.
**References:** `src/llama-weight-pager.cpp:96-106` (callback assignment),
`src/llama-weight-pager.cpp:206-217` (pool allocation)
**Introduced/fixed in:** commit `786f5019a`
**Test that locks it in (Phase 1):** unit in `wp-pool` asserting
`pool.ggml_buf` is non-null after init and is the buffer-type returned by
`ggml_backend_cuda_buffer_type(device)`.

---

### B-P5 — `GGML_CUDA_DISABLE_GRAPHS` env-var leak *(no fix in old code)*
**Component:** weight-pager initialisation
**Symptom:** if a process loads a model with weight paging and later loads
a different model without paging, hipGraphs remain disabled for the second
model. Performance regression goes unnoticed.
**Root cause:** when `model.weight_pager` exists, the context-init path
calls `setenv("GGML_CUDA_DISABLE_GRAPHS","1",1)` unconditionally and never
restores the prior value.
**Reference:** `src/llama-context.cpp:1207`
**Status in rewrite:** **NEW WORK** in Phase 1. `WeightPager::init` records
prior env state (presence + value); `WeightPager::shutdown` restores it
(`unsetenv` if originally absent, else `setenv` to prior). RAII in
`wp-pager.cpp`.
**Test that locks it in (Phase 1):** unit in `wp-pager` that snapshots
`GGML_CUDA_DISABLE_GRAPHS` before pager construction, exercises the pager
lifecycle, and asserts the env var matches the snapshot after destruction.

---

### B-P6 — Async-prefetch page-index race (currently disabled)
**Component:** weight-pager async prefetch
**Symptom:** wrong pages occasionally loaded into wrong slots when async
prefetch was enabled. Could cause silent miscompute or hangs.
**Root cause:** `submit_prefetch` writes `page_idx` into the io_uring
`user_data` field; `complete_prefetch` reads it back as `completed_idx`.
When multiple in-flight requests complete out of submit order — which
io_uring is permitted to do — the loop processes them in completion order
but uses the per-request slot/dst from `in_flight[completed_idx]`. There
is no inherent race in that single map, but the loop also touches the
*current* `page_idx`'s state via `loaded = (completed_idx == page_idx)`,
and state machine assumptions about completion ordering break down. The
old fix was to disable async prefetch by default
(`async_prefetch = false`) and leave the bug as a TODO.
**References:** `src/llama-weight-pager.cpp:472-518` (broken loop),
`src/llama-weight-pager.h:78-80` (`async_prefetch = false; // TODO`)
**Status in rewrite:** **NEW WORK** in Phase 1. The new `FileIOLayer` API
uses a monotonic `req_id` (not page index) as the io_uring `user_data`. A
`PrefetchScheduler` keeps `unordered_map<req_id, request>` so completions
route correctly regardless of order. The two-stage pipeline (read → copy)
is explicit; each stage records its own completion event.
**Test that locks it in (Phase 1):** stress test that submits N=1000
prefetches with deliberately variable per-request latency (mocked
`FileIOLayer::wait` returns out-of-order) and asserts every page lands in
the correct slot.

---

### B-P7 — Multi-GPU silent miscompute *(no fix in old code)*
**Component:** weight-pager pool allocation
**Symptom:** with `--device ROCmA,ROCmB --weight-paging`, weights placed
on device 1 by the layer split are not paged at all (the pool exists only
on device 0). Either silent miscompute or OOM if device 1 cannot hold its
unpaged layers statically.
**Root cause:** `pool.ggml_buf` is allocated via
`ggml_backend_cuda_buffer_type(0)` — hard-coded to device 0. There is no
guard against multi-device configs.
**Reference:** `src/llama-weight-pager.cpp:210`
**Status in rewrite:** **NEW WORK** in Phase 1. `WeightPager::init` takes
a `devices_used` set; if it has more than one element, init fails with a
specific error pointing at the future per-device-pool work item. Phase 1
is single-device by explicit design; multi-device pools are deferred.
Detection happens in `src/llama-model.cpp` based on `params.tensor_split`
and `params.split_mode` before pager construction.
**Test that locks it in (Phase 1):** negative test running with
`--device ROCm0,ROCm1 --weight-paging` and asserting a clear error message
is printed and the process exits non-zero.

---

## Cross-cutting notes

### CLI flag inventory (current names, before Phase 2 rename)

Tier flags (`common/arg.cpp`):
- `--kv-tiered HOT,WARM,COLD` — line 1329
- `--tier-ssd-path PATH` — line 1348
- `--tier-eviction-policy POLICY` — line 1355
- `--tier-compression TYPE` — line 1362
- `--tier-attention-threshold THRESH` — line 1369
- `--kv-warm-device DEVICE` — line 1376
- `--tier-total-ctx N` (search `kv_tier_total_ctx`)
- `--kv-semantic-index PATH` — line 1383
- `--kv-semantic-threshold F` — line 1390
- `--kv-semantic-topk K` — line 1397

Pager flags (`common/arg.cpp` ~2421-2438):
- `--weight-paging`
- `--weight-paging-slots N`
- `--weight-paging-prefetch`

Phase 2 renames the inconsistent `--tier-*` prefix to `--kv-tier-*` with
deprecation aliases (see plan §2.5). Pager flags are kept verbatim.

### KVTC SSD format (current, version 1)

Magic: `0x4B565443` ("KVTC"). Version: 1. Structure:
`file_header | layer_index_entry[] | data sections`. Defined in
`src/llama-kv-cache-tiered.h:65-145`. Phase 2 bumps the version to 2 and
adds a `KVTC_SECTION_RECURRENT` section type for recurrent state. Old v1
files reject with a clear error.

### Commit hashes (KEEP list, do not revert)
- `4424bbe70` — cached pinned staging in
  `ggml_backend_cuda_split_buffer_set_tensor` (B-K6 part 1)
- `65fb0a5c3` — stage cross-device copies in `ggml_cuda_op_mul_mat` (B-K6
  part 2)
- `b25eddd1b` — server checkpoint clear on context shift (B-K4)
- `16d66aadf` — `ggml_cuda_set_device` early-return removal (B-K7)
- All upstream merges (out of scope)

### Source files retired by the rewrite
The following are deleted by Phase 1.0 (pager) and Phase 2.0 (tier):
- `src/llama-kv-cache-tiered.{h,cpp}` — replaced by `src/memory-tier/`
- `src/llama-eviction-policy.h` — replaced by `src/memory-tier/mt-eviction.{h,cpp}`
- `tools/server/server-tiered-cache.{h,cpp}` — server no longer owns tier
- `src/llama-weight-pager.{h,cpp}` — replaced by `src/weight-pager/`
- `src/llama-io-uring.{h,cpp}` — concept retained, API rewritten in
  `src/weight-pager/wp-file-io.{h,cpp}`

This catalog must be reviewed before Phase 1.0 deletion. Any bug
discovered later that turns out to have a fix in the deleted code adds an
entry here (and a test) before the rewrite proceeds past the failing sub-phase.
