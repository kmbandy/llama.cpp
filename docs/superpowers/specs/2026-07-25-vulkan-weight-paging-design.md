# Weight paging on Vulkan — design (REVISION 2)

**Date:** 2026-07-25
**Repo:** `~/GitHub/llama.cpp` on **mad-lab-2026**, master, tip `7504856a2` + working tree
**Target:** RX 480 (`Vulkan0`, RADV POLARIS10, 8 GB, discrete, `uma: 0`)

> **Revision 2 supersedes revision 1.** Revision 1 scoped only the transport and
> was **incomplete**: it missed that weight paging also needs a *consumption*
> path in the matmul kernel. An implementation built to revision 1 compiles,
> initialises, stages pages into the pool with correct offsets — and outputs
> garbage, because nothing ever reads from the pool. Section 3 is the part that
> was missing. Do not work from revision 1.

## 1. Status: what is already done and proven

The **transport is implemented and is not the defect.** It lives in
`ggml_backend_vk_wp_stage_in` and friends (`ggml/include/ggml-vulkan.h`,
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`) plus a Vulkan branch in
`src/weight-pager/wp-gpu-transport.{h,cpp}`. It:

- recovers the pool's `vk_buffer` from the `ggml_backend_buffer_t` handed to
  `GpuTransport::init`;
- converts `dst` to a buffer offset as `(uintptr_t)dst - (uintptr_t)vk_ptr_base`,
  which is correct — observed `dst=0x1000` for slot 0 and `0xee1000` for slot 1
  at a 14.875 MiB slot size;
- writes the payload and the padding zero, and returns a fence-backed event.

Measured evidence that staging is sound: replacing the async context handling
with the known-good fully-synchronous `ggml_vk_buffer_write` /
`ggml_vk_buffer_memset` produced **byte-identical garbage**. The defect is not in
staging, context management, or the offset arithmetic.

**Keep this work.** The remaining task is section 3.

## 2. The evidence, so the next person does not re-derive it

On `Vulkan0`, `LFM2.5-8B-A1B-UD-Q6_K`, same prompt, `--temp 0`:

| run | output |
|---|---|
| non-paged | coherent prose, byte-matching the CUDA non-paged output |
| paged | `66666666666666666666666666666666` |

The card, the model, and the Vulkan backend are all fine. Paging is what breaks.

## 3. The missing half: consumption

On CUDA/HIP the pager does **not** rely on tensor pointers. Per MoE node, the
eval callback builds a host array of per-expert device pointers, uploads it, and
calls `ggml_cuda_set_routed_expert_ptrs` (`ggml/src/ggml-cuda/mmq.cu`) from
`src/weight-pager/wp-eval-cb.cpp:1080`. The MMQ kernel then reads each expert
from its own pointer. Paged tensors are deliberately left with
`buffer == NULL`, and `wp-eval-cb.cpp` patches `src0->buffer` purely so
`ggml_cuda_mul_mat_id` does not NULL-deref on entry.

**ggml-vulkan has no equivalent.** That is the entire bug.

### 3a. Why Vulkan is an easier target than CUDA here

CUDA needs raw pointers because slots are bare addresses. On Vulkan **every slot
lives in one pool `VkBuffer`**, so the shader needs only a per-expert **offset**,
not a pointer. No `buffer_device_address`, no pointer plumbing.

### 3b. The exact seams — there are TWO, and decode uses the one not named first

`ggml_vk_mul_mat_id` (`ggml-vulkan.cpp:10360`) forks:

```
if (ggml_vk_use_mul_mat_vec_id(cgraph, node_idx))  ggml_vk_mul_mat_vec_id_q_f16(...);
else                                               ggml_vk_mul_mat_id_q_f16(...);
```

and `ggml_vk_use_mul_mat_vec_id` (`:10353`) is `src2->ne[1] <= 8 && (f32|f16|quantized)`.
`src2->ne[1]` is the token count, so **decode takes the vec path and prefill takes
the mm path.** Both must be changed or generation is wrong.

- **decode** — `vulkan-shaders/mul_mat_vec_base.glsl:69-71`:
  `a_offset = expert_id * (p.batch_stride_a / QUANT_K);`
- **prefill** — `vulkan-shaders/mul_mm.comp:248-252`:
  `pos_a = expert_idx * (p.batch_stride_a / LOAD_VEC_A_EFF) + ...`

Both compute the expert base as a **uniform stride** off one consolidated
tensor. Under paging each expert sits at an arbitrary pool slot, so both must
become a per-expert lookup.

### 3c. BLOCKING CONSTRAINT: slot offsets must be quant-block aligned

Note the `/ QUANT_K` and `/ LOAD_VEC_A_EFF` above: `a_offset` / `pos_a` are
**block indices**, not byte offsets. Vulkan's quantized shaders bind `data_a` as
an array of quant-block structs, so an expert's base must be an exact multiple of
the block byte size.

`PoolAllocator` sets `slot_alignment_ = ggml_backend_buft_get_alignment(buft)`
(`wp-pool.cpp:212`). On Vulkan that is a storage-buffer alignment (typically 256).
A Q6_K block is **210 bytes**. 256 is not a multiple of 210 — `lcm(256, 210)` is
26880 — so **slot offsets are not block-aligned and cannot be represented as a
block index at all.** The observed slot size, 15597568 B, is likewise not a
multiple of 210.

This never arises on CUDA/HIP because those pass raw byte pointers, which have no
block-alignment requirement. It is Vulkan-specific and it blocks the whole
approach until fixed.

**Resolution:** align pool slots to `lcm(buft_alignment, block_byte_size)` so
every slot offset is an exact multiple of the block size, then publish per-expert
offsets already divided into block units. At 26880 B against a ~15 MB slot the
waste is about 0.17%, and the change is confined to `PoolAllocator`. Assert
exactness when converting a byte offset to a block index rather than trusting it —
a silent truncation here yields subtly wrong weights, not a crash.

### 3c. What to build

1. A **per-expert offset buffer** (SSBO of 32-bit offsets, or 64-bit if the pool
   can exceed 4 GiB — check, and size the type from the pool, do not assume).
2. A **paged variant of the MUL_MAT_ID path**, selected by a define or
   specialization constant, in which `pos_a` is read from that buffer instead of
   computed from `expert_idx * batch_stride_a`. Keep the non-paged path
   byte-identical; this must be opt-in per dispatch.
3. **Descriptor binding**: with paging active, `data_a` must bind the **pool**
   buffer, not `src0`'s buffer. `ggml_vk_mul_mat_id_q_f16`
   (`ggml-vulkan.cpp:9771`) and `ggml_vk_mul_mat_id` (`:10360`) are the
   integration points.
4. A **side channel** to publish the current node's expert offsets, mirroring
   `ggml_cuda_set_routed_expert_ptrs` in role. Keep it narrow and opaque, in the
   style of the existing `ggml_backend_vk_wp_*` surface.
5. A **Vulkan branch in `wp-eval-cb.cpp`**. The routed-expert block is currently
   `#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)`; the Vulkan path
   computes slot offsets rather than device pointers, so it is a sibling branch,
   not a widened guard.

**Establish which matmul path actually executes** for the test model before
changing shaders. `mul_mm.comp` and `mul_mmq.comp` both include
`mul_mm_id_funcs.glsl`, and a Q6_K model may take the integer-dot MMQ path
rather than the f16 path. Verify by instrumenting or logging the chosen pipeline
— do not assume, and do not change both blind. If both are reachable for models
we care about, both need it, but say which you verified.

## 4. Ordering against compute

Unchanged from revision 1 and already resolved in the implementation: the
transport uses **fence-before-return**, so no `ensure_batch` exit path can return
with a transfer in flight. This is the conservative choice and is correct; a
semaphore handoff is a later optimisation, not required now.

Note the reason it matters, recorded at `wp-gpu-transport.cpp:166-178`: on HIP a
copy on the default stream did *not* auto-serialize with GGML's compute stream,
producing a torn-write race against kernels reading the same slot. The same
hazard exists on Vulkan if the fence is dropped.

## 5. Known dependency, still not done

Pool auto-sizing **does not clamp to free VRAM**, despite the
`--weight-paging-slots` help text claiming `-1 = auto = layer count, capped to
free VRAM`. Measured: it requested **34688 MiB on an 8 GB card**
(`n_slots=2332`) and hard-failed init. The heuristic is calibrated for the
R9700's 32 GB, and the RX 480 is also 8 GB.

Fix the clamp. It was in the previous task and was not done. Until then every
run must pin `--weight-paging-slots` explicitly, and a failure here must not be
mistaken for a Vulkan fault.

## 6. Verification

Correctness only. Throughput is not a goal and must not be reported as one.

1. **`test-backend-ops -b Vulkan0`** still green for `MUL_MAT_ID` — the non-paged
   path must not regress. This is the first gate because it is cheap and it
   guards the code everyone else uses.
2. **Coherent output from a paged run on `Vulkan0`**, with `page_ins > 0` and
   `ensure_batch` actually engaging. Command:
   `./bin/llama-cli -m ~/models/LFM2.5-8B-A1B-UD-Q6_K.gguf --device Vulkan0 -ngl 99
   --weight-paging --weight-paging-slots 200 -st -n 32 --temp 0 -p "The capital of France is"`
   with `WP_HOST_BUDGET_BYTES=4294967296 WP_ENSURE_BATCH=1 WP_ENSURE_BATCH_HOST=1`.
   The current failure signature is `6666...`; the non-paged reference text on
   this exact device begins `The user wrote: "The capital of France is" and
   presumably expects the answer "Paris".`
3. **Perplexity equivalence, paged vs non-paged, same device.** The gate that
   separates "runs" from "correct". Compare **final estimates only**. Do **not**
   run a sign test over the per-chunk `[N]` values — they are running cumulative
   averages, not independent samples, and treating them as independent produced a
   false regression report earlier today.
4. **No regression on CUDA0 or HIP.** `wp-eval-cb.cpp` and the transport are
   shared source. Run the CUDA paged smoke test too.

Build the **full target set** and report real exit status. A stale executable
against a rebuilt `libllama.so.0` segfaults on ABI drift; shared linkage does not
protect against it.

## 7. Machine and safety notes

Build directory is **`build-army`**, the only sanctioned one here.

**A board claim is mandatory before any GPU run** — `board_claim` for
`mad-lab-2026` / `gpu:480`, naming the resource explicitly because `board_check`
for `gpu:0` resolves to the RX 480 and both cards report index 0 in their own
vendor namespaces. The previous task ran no gates and never claimed a card;
that is what let a non-working implementation look finished.

A live `llama-router.service` holds `/dev/dri/renderD*`. Never `pkill`/`pgrep` by
pattern; never run `killsweep.sh llama-server`. Stage only files you touch, by
explicit path; never `git checkout/restore/stash/reset/add -A/commit -a`. The
tree holds unrelated uncommitted work (Vulkan sinkhorn, and Instella specs).
