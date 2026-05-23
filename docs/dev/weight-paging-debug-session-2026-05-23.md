# Weight Paging Debug Session — 2026-05-23

**Branch:** `wip/mad-230-paging-correctness` (HEAD at `04e34d9e1`)
**Jira:** MAD-230
**Standing test command** (locked — apples-to-apples between runs):

```bash
./build-hip/bin/llama-perplexity \
    -m /home/kmbandy/models/Qwen3.5-4B-UD-IQ3_XXS.gguf \
    --weight-paging --weight-paging-slots 128 --device ROCm0 -ngl 99 \
    -f /home/kmbandy/GitHub/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --chunks 4 -c 2048
```
Expected: `PPL = 10.8649 +/- 0.47706`. Baseline (no paging) = 10.8649.

---

## 1. What works (validated)

**Dense + small-MoE pageable models: PPL matches no-paging baseline exactly.**

- Qwen3.5-4B-UD-IQ3_XXS, 128 slots, `--no-mmap` not needed (auto-disabled): **PPL = 10.8649 ✓**
- Same model, 8 slots (heavy eviction stress): **PPL = 10.8649 ✓**
- Original "garbage 248320 PPL" bug fully resolved for this class.

## 2. What's still broken

**Qwen3.6-35B-A3B-UD-Q4_K_XL with `--weight-paging`: GPU memory access fault during first forward pass.**

Fault addresses across runs (all near-null = `nullptr + small_offset`):
- `0x1b000` (110 KiB)
- `0x1d000` (118 KiB)
- `0x41000` (264 KiB)
- `0x47000` (290 KiB)

**Hardware safety:** caused **two full system restarts** today. DO NOT iterate on 35B with `--weight-paging` until we have a higher-confidence hypothesis. See § 9.

## 3. Root cause(s) we fixed (dense path)

Three intertwined bugs we identified and fixed:

### 3.1 Resident weights never allocated/loaded
`token_embd / output_norm / output.weight` are deliberately excluded from the pager catalog (MAD-88: keep them resident, smaller slot stride). The alloc loop in `llama-model.cpp` also skipped them as "weights" via `ml.get_weight(name) != nullptr`. Result: `buffer == NULL`, `data == NULL` forever. Placeholder set in `init_weight_pager` only caught some tensors. Kernel read uninitialized memory → uniform logits → **PPL = vocab_size = 248320**.

**Fix:** `llama-model.cpp` manual per-tensor allocation. For GPU buft with paging on, allocate a real buffer sized for the NON-paged tensors only (resident weights + non-weight tensors). Use `ggml_backend_tensor_alloc` to place each at offsets. Paged tensors stay `buffer == NULL` for the pager. This avoids the alloc-then-orphan trap that wastes 22+ GB for 35B-class models.

### 3.2 TENSOR_DUPLICATED creates two ggml_tensor instances in two ctxs
For tied embeddings (Qwen3.5), `output.weight` is created with `TENSOR_DUPLICATED` flag. `buft_for_tensor` (llama-model-loader.cpp:1094) rewrites `LLM_TENSOR_TOKEN_EMBD + DUPLICATED → LLM_TENSOR_OUTPUT` for buft selection. This puts the duplicate in a DIFFERENT ctx (typically CPU-side instead of GPU-side). GET_ROWS uses one instance, MUL_MAT uses the other. Both need allocation + load.

**Fix:** `llama-model.cpp` calls `load_all_data` on ALL ctxs (host AND GPU). Each ctx's allocator runs independently; both instances get populated.

### 3.3 `load_all_data` null-buffer skip placement
Original skip for "weight has no buffer = paged, skip" was placed BEFORE the mmap-path lazy-alloc check. For mmap-eligible host tensors (`buffer == NULL` at entry, lazily allocated inside `load_all_data` via `ggml_backend_tensor_alloc(buf_mmap, ...)`), the skip killed the lazy allocation.

**Fix:** `llama-model-loader.cpp` skip is `if (cur->buffer == nullptr && cur->data == nullptr && !use_mmap) continue;`. Mmap-path tensors fall through and get properly allocated.

### 3.4 mmap and weight-paging conflict
`mmap` pre-resolves tensor data pointers at graph build; the pager rewrites them per-op; the mmap-path bounds-check at `ggml-backend.cpp:2023` asserts on the duplicated tensor in the CPU ctx.

**Fix:** `llama-model.cpp` — when `params.weight_paging_enabled` is true, force `ml.use_mmap = false` at the top of `load_tensors` with a clear log message. One canonical residency manager, no overlap.

## 4. MoE 35B-A3B crash — what we know

### 4.1 Crash signature (consistent across runs)
- Fault address: `nullptr + small_offset` (0x1b000–0x47000 range)
- Crash timing: **after 4 successful routed MUL_MAT_ID ops** (per `[mmq DIAG]` log)
- 4 ops = blk.0 ffn_gate_exps + ffn_up_exps + ffn_down_exps + blk.1 ffn_gate_exps
- So crashes around the 5th MoE op (blk.1.ffn_up_exps or blk.1.ffn_down_exps)

### 4.2 Diagnostic prints captured
At the routing-active branch in `mmq.cu` (after my gate), every consolidated MoE parent shows:
- `src0=blk.N.ffn_*_exps.weight`
- `data=0x7f...c00000` — **SAME placeholder address across all parents** (pool_buf base)
- `buf=0x55...` — **SAME pool_buf across all**
- `nbytes` per parent = 150-184 MB (full consolidated size, not per-expert)

### 4.3 Fixes already applied (don't fully resolve)
- **mmq.cu padding-clear gated** on `!ggml_cuda_has_routed_expert_ptrs()` — confirmed working (4 routed ops succeed instead of crashing immediately at mmq:144)
- **wp-eval-cb.cpp inactive expert sentinel** — `host_ptrs[inactive] = first_active_slot` instead of nullptr. Defends against `expert_ptrs[inactive_idx]` reads. Didn't change fault behavior.
- **wp-eval-cb.cpp `hipDeviceSynchronize`** before overwriting `s_dev_expert_ptrs` — defends against the static-array race where op N+1's eval_cb overwrites the array while op N's kernel is still reading. Didn't change fault behavior either.

### 4.4 Hypotheses ruled out
- ❌ Slot-reuse eviction race (verified by dense-with-8-slots heavy eviction working)
- ❌ HIP graphs caching pointers (GGML_CUDA_DISABLE_GRAPHS=1 set, no change)
- ❌ ggml-sched cross-backend copy (sched_copy diagnostic never fires for MoE)
- ❌ MMVQ kernel reading `expert_ptrs[inactive] == nullptr` (sentinel fill didn't help)
- ❌ Static `s_dev_expert_ptrs` overwritten while in use (hipDeviceSynchronize didn't help)
- ❌ mmq.cu padding-clear writing to placeholder (gated, got past it)

### 4.5 Hypotheses STILL ON THE TABLE
1. **A different kernel reads `src0->data` directly without going through `expert_ptrs`.** Suspects: dequant kernel (Q4_K_M → fp16 staging), MMVF (mul_mat_vec_f), copy-to-shared-memory paths, MMVQ's `vx` parameter not just `vx_for_channel`. The dispatcher at `ggml-cuda.cu:2830+` picks between MMVQ/MMQ/MMVF based on batch shape — for prefill ne2 size, the chosen kernel may be one we haven't gated.
2. **A non-routing-aware op references a paged tensor.** E.g., a norm op, a gate computation, a scaling op that reads from a paged weight tensor where eval_cb didn't patch correctly.
3. **The `src0->buffer = pool_buf` patch on the consolidated parent (wp-eval-cb.cpp:200-203) misleads some downstream ggml code** into thinking the tensor is properly allocated, when really its data pointer is the placeholder.
4. **The DEQUANT path for Q4_K_M consolidated tensors.** MMQ does in-loop dequant from `src0->data + offset`. If `expert_ptrs` is set but the dequant code reads relative to `src0` (the consolidated parent) rather than per-expert pointer, it reads pool_base + some big offset → far address, not near-null. Doesn't match symptom unless it then misuses that for indexing.
5. **The fault happens NOT in MoE compute but in a non-MoE op that immediately follows** — e.g., the residual add, the norm, or attention setup. We see 4 MoE DIAGs then crash, but the next op might be a non-MoE op that doesn't print a DIAG.

## 5. Key code-comparison findings (hipfire vs llama.cpp)

| Aspect | hipfire (works) | llama.cpp (broken on MoE) |
|---|---|---|
| Per-expert memory | Separate `hipMalloc` per expert → distinct device pointers | Shared slot pool, slot pointers patched per-op |
| Kernel reads | `gpu.embedding_lookup_hfq4g256(&weights.token_embd, ...)` — direct pointer | `src0->data` patched by eval_cb, kernel reads it OR reads via `expert_ptrs` (routing-aware only) |
| Inactive experts | Never accessed; gridDim.y = K (active count) | gridDim covers `n_experts`; relies on `expert_ptrs` to filter (potentially leaky) |
| Synchronization | Explicit `hipEventSynchronize` between H2D and kernel launch | Sync hipMemcpy + stream ordering; routing-aware just added explicit sync, didn't help |
| Slot pinning | LRU touch on `ensure_resident` keeps slot MRU for forward pass | No pinning; pool can evict mid-op (we've worked around this for dense case) |

**Architectural takeaway:** hipfire's design fundamentally avoids the class of bugs we're hitting because there's NO shared placeholder and NO indirection through a pool. Each weight is independently addressable. Llama.cpp's pool+patch design is forced to constantly re-establish pointer correctness per-op, and any path that doesn't honor the routing-aware indirection breaks.

## 6. Files changed on this branch

```
common/arg.cpp                    — accept --weight-paging in llama-perplexity
src/llama.cpp                     — placeholder only on catalog tensors
src/llama-model.cpp               — manual per-tensor alloc; auto-disable mmap
src/llama-model-loader.cpp        — load_all_data null-buffer skip + diag
src/weight-pager/wp-eval-cb.cpp   — diag infrastructure + inactive sentinel + sync
ggml/src/ggml-cuda/mmq.cu         — gated padding-clear + DIAG print
```

Diff stats from main: ~250 insertions / ~70 deletions across 6 files.

## 7. Commits on the branch

```
04e34d9e1  wip(MAD-230): manual per-tensor alloc + MoE crash diagnostics
76bb2cffc  fix(MAD-230): auto-disable mmap when weight paging is enabled
765bfb093  fix(MAD-230): weight pager produces correct PPL — root cause solved
5dba94d0e  wip(MAD-230): weight pager correctness — partial fix + root cause
```

## 8. What to investigate next session

**Don't run 35B-A3B + --weight-paging without strong reason — see § 9.**

In priority order:

1. **Read MMQ's main compute kernel** (after the padding-clear) end-to-end. Find every place that dereferences `src0->data` directly. Identify which paths are gated on `expert_ptrs` and which aren't. Specifically: the dequant inner loop, any "load to shared memory" prologue, any `args.x = src0->data` setup before kernel launch.

2. **Read MMVF (`mul_mat_vec_f`)** with the same lens. Same questions.

3. **Check the dispatcher choice for the 35B-A3B prefill shape.** What's `ne2` (batch size dim 2) for a 7-token prefill × 8 active experts? Cross-reference `MMVQ_MAX_BATCH_SIZE`, `MMVF_MAX_BATCH_SIZE`, `get_mmvq_mmid_max_batch(...)`. Figure out which kernel actually runs.

4. **Find a smaller MoE model** to reproduce on (Mixtral 8x7B? something <16GB?) so we can iterate without risking the box. A model where we can test the same code path with way less hardware risk.

5. **Static analysis + targeted printk** of the EXACT kernel that runs (once 3 confirms). Print `args.x`, `args.expert_ptrs`, every read pointer at the kernel C++ wrapper before launch.

6. **ONE more end-to-end run** only after the printk-localized hypothesis is verified.

## 9. Hardware safety lessons learned today

Three near-OOM events in one session, two requiring full system restart:

1. PPL with `--weight-paging` no slot cap → auto-sized to 29 GB pool, near-OOM (caught + fixed via slot count rule)
2. 35B-A3B `--weight-paging-slots 1200` + alloc-then-orphan → silent OOM (allocated 22.8 GB then orphaned), **system restart #1**
3. 35B-A3B + `hipDeviceSynchronize` in routing path → hard hang, **system restart #2**

**Rules going forward:**
- Calculate expected VRAM footprint BEFORE launching any 20B+ paged test. Present the math to user. Get confirmation if total is within 4 GB of 32 GB.
- After one OOM event on a model, STOP iterating on that model — switch to smaller test or pure static analysis. The failure mode of `--weight-paging` on a model where the bug isn't fully fixed can hard-hang the GPU.
- `hipDeviceSynchronize` in eval_cb is risky when the GPU is in fault state — may itself hang. Prefer event-based or stream-scoped synchronization.
- The R9700 + this Linux install + this model + this branch combination is a finite resource. Treat it accordingly.

## 10. Implications for MAD-223 (ml8-4 calibration)

**Calibration is unblocked for dense / small-model proof-of-concept.** Pager produces correct output on dense paged models. We can validate the `auto_gptq + CentroidQuantizer + pybind11` pipeline end-to-end on Qwen3.5-4B-IQ3_XXS.

**Calibration on 35B-A3B specifically is BLOCKED** on the remaining MoE pager bug. Workarounds for the immediate weekend:
- Validate format + pipeline on 4B first
- Convert 4B / 7B class models to ml8-4 to prove the format
- Defer 35B end-to-end to a later session after the MoE bug lands

## Related
- `docs/dev/memory-tier-bug-catalog.md` — earlier paging bug catalog (B-P1 through B-P7)
- `ggml/src/ggml-cuda/aiter-integration/TURBO_FP8_CALIBRATION_DESIGN.md` — calibration recipe lever-map
- MAD-88 — original weight pager epic (Phase 1-9 history)
- MAD-229 — hipfire v0.5 dma_buf P2P port (independent, doesn't block this)
