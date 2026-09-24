# Upstream sync 2026-09-24

Merge of `upstream/master` (ggml-org/llama.cpp @ `bd4f514`) into the fork (origin/master @ `26e5f58`).

- Merge base: `427291b5b` (metal : add remaining fa-vec tunings for M3 (#28396))
- Upstream commits merged: 335
- Fork commits since base: 1570
- Conflicted files: 32 (~90 hunks)
- Branch: `claude/hello-ky3ivj` (not master; review before merging)

Protocol: same as the earlier syncs. No blind `--theirs` on any customized file. Where upstream added a mechanism that does the same job as fork code, both were compared and one kept (reasons below). Fork lines that the merge dropped were audited afterwards.

Trivial "keep both sides" hunks (independent additions next to each other) are not listed. Everything that needed a decision is.

## Decisions

### common/common.h - `hostname` -> `hostnames`
Upstream replaced `std::string hostname` with `std::vector<std::string> hostnames` (multi-address bind). The conflict only came from our paged-KV / semantic-index params sitting right above it. Kept all fork params, took upstream's `hostnames`. No fork code reads `params.hostname`.

### common/chat.cpp - parsers moved to `common/parsers/` (upstream #27764)
Upstream split the per-model chat parsers out of `chat.cpp` into `common/parsers/*.cpp`. That is the whole 2500-line conflict. Our side had two small fixes inside the moved functions. Took upstream's `chat.cpp` and ported both fixes to the new files:
- `common/parsers/lfm2.cpp`: LFM2.5 optional opening `<think>` (fork `bf90040d7`). Upstream still has the strict `THINK_START + ...` form, so the reasoning-bleed bug would come back without the port.
- `common/parsers/deepseek.cpp`: DeepSeek V4.1 DSML tag names with a leading space (fork `6274e38a7`). Upstream has no V4.1 detection.

### common/speculative.cpp - draft split mode (upstream #28390 vs fork MAD-LAB)
**Overlap: both sides solved the same bug.** A draft model inherits the target's `-sm tensor` and gets its own Meta device wrapper.
- Upstream: force `LLAMA_SPLIT_MODE_LAYER` only when `-devd` names exactly one device.
- Fork: force `LAYER` for every separate draft model.

**Kept: fork.** The fork's version covers upstream's case (one draft device) and also covers multi-device and inherited-device drafts. In this fork the draft context co-schedules on the target's Meta backend (see the MAD-LAB note in llama_context backend init), and two Meta devices in one scheduler abort in `ggml_backend_meta_buffer_simple_tensor()`. Upstream's narrower check would bring that abort back for `-devd A,B` or a draft that inherits the target devices. The fork also avoids a per-layer AllReduce on the latency-critical draft path. Upstream's comment and code were dropped.

### common/speculative.cpp - M-RoPE pinned-image skip in DFlash inject
Upstream added a skip for image rows pinned to one position. Kept it and put it **before** the fork's `DFlash inject` stats and `WP_DSPARK_DEBUG` census, so skipped rows are not logged as injected. Also took upstream's "frist" -> "first" typo fix next to our hash-trace block.

### CUDA/HIP FlashAttention instance selection (upstream #28079 vs fork turbo lists)
**Overlap.** Upstream replaced `GGML_CUDA_FA_ALL_QUANTS` with `GGML_CUDA_FA_QUANTS`: a list of `K-V` pairs, per-pair `GGML_CUDA_FA_<K>_<V>` defines, and a shared `ggml_cuda_fattn_vec_instances()` in `ggml/cmake/common.cmake` used by CUDA, HIP and MUSA. When a pair was not compiled, the dispatcher now falls back to f16 conversion (with a one-time warning) instead of aborting. It also dropped the `K->type != V->type -> NONE` gate.

The fork had three pieces that did the same job: turbo instance files hard-coded into the CUDA and HIP `else()` lists, turbo `FATTN_VEC_CASES_ALL_D` rows placed before `GGML_ABORT`, and a widened `K != V` gate that let mixed turbo/q8_0 pairs through.

**Kept: upstream's mechanism, extended for turbo.**
- `ggml/cmake/common.cmake`: new `FA_TURBO_COMBINATIONS` (the same 15 pairs the fork compiled before). They are always compiled, whatever `GGML_CUDA_FA_QUANTS` is set to, and they get `GGML_CUDA_FA_TURBO*_*=1` defines. They are not added to `FA_TYPES`, because `GGML_CUDA_FA_QUANTS=all` crosses every type with every other and there are no turbo x q4_0 (etc.) instance files. That would be a configure-time FATAL_ERROR.
- `fattn.cu`: the turbo rows moved into `ggml_cuda_get_fattn_vec_case()` with upstream's short type names. The fork's mixed-type gate was dropped because upstream has no gate left to widen.
- CUDA/HIP `CMakeLists.txt`: took upstream (the function call).

Why: upstream's version can be configured and fails soft. A turbo pair we did not compile (for example turbo3-f16) now runs via f16 conversion instead of aborting. MUSA also gets the turbo instances now. Before this it referenced them without compiling them. The fork's `GGML_CUDA_FA_ALL_QUANTS=ON` still works; upstream maps it to `all` with a deprecation warning.

### ggml/src/ggml-cuda/vendors/hip.h - host-alloc aliases
**Overlap.** Upstream added four of the six `cudaHostAlloc*` -> `hipHostMalloc*` aliases that the fork already had for the AllReduce pipeline. Kept the fork's block because it is a superset (it also has `cudaHostAllocDefault` and `cudaDevAttrCanMapHostMemory`) and dropped upstream's duplicate defines.

### src/CMakeLists.txt - unity build (upstream #28091)
Upstream moved the core sources into `LLAMA_CORE_SOURCES` and turned on `UNITY_BUILD` for the rest (the `models/*.cpp` files, in batches of 16). Core sources are excluded. Added the fork's core files (`llama-tp-lockstep`, `pipeline/pipe-tp-msg`, `llama-kv-cache-paged`, `llama-ml8-registry`, `llama-pipeline`, `llama-pipeline-gguf`, `llama-weight-pager`) to `LLAMA_CORE_SOURCES`. Also excluded `weight-pager/*.cpp` and `memory-tier/*.cpp` from the unity build. Those are large, standalone subsystems that were never written for single-TU inclusion. The fork's own `models/*.cpp` now take part in the unity build, the same as upstream's (see the build check at the end).

### tools/server/server-models.cpp - router scheduler (upstream #29217, #28539, #28530)
Upstream rewrote `server_lru_sched`. Every load now goes through the queue (`join` -> `try_claim` -> `load`). `tick()` evicts idle models while queued requests outnumber free slots, and replaces `on_model_idle` / `mark_slot_pending`. `pick_victim` lost its `exclude` argument and skips any model a queued request wants. Requests are no longer admitted into a stopping model. The fork had not changed `ensure_model_ready`, so upstream's version came in unchanged. The fork-specific pieces were merged one at a time:

- **`pick_victim` guards:** kept the fork's **pinned** hard hold (upstream has no pinning) and took upstream's "stopping or queued" filter, which covers the fork's old stopping guard. `tick()` also goes through `pick_victim`, so pinned models are safe from queue-driven eviction too.
- **`unload_lru()`**, overlapping fix: **kept fork** (with the new one-argument `pick_victim`). The fork marks the victim as stopping under the same lock that picked it, returns early when nothing can be evicted, and waits with the find-based `wait()`. Upstream's version waits on `mapping[name]`, which default-constructs a stray entry if the model was erased mid-wait. The fork's version is the more race-safe of the two.
- **Capacity re-check in `load()`**, overlapping: upstream throws "model limit reached" and relies on the queue to prevent the race. The fork evicts and waits (30 s cap). **Kept the fork's loop and added upstream's `opts.mode == SERVER_CHILD_MODE_NORMAL` exemption** so download workers do not take `models_max` slots. On the request path the queue guarantees capacity before `load()` runs, so the loop does nothing there. It still matters for loads that bypass the queue (the `/models/load` API, startup preload, reload), where a 500 is worse than a short wait.
- **Proxy cleanup:** kept the fork's chained cleanup and `last_used` re-stamp at response end (so the idle sweeper does not unload mid-stream). Swapped the removed `sched->on_model_idle()` for upstream's `sched->tick(lk)` under the same lock.
- **`update_status`:** kept `reconcile_gpu_reservation_locked()` and added upstream's `sched->tick(lk)` after it.
- **Preset load:** kept the fork's preset-mtime tracking and took upstream's `SRV_INF` -> `SRV_TRC` log level change.

### src/llama-context.{h,cpp} - graph result cache (upstream #28549 vs fork `WP_GRAPH_RESULT_SLOTS`)
**Overlap: both sides cache more than one graph result so alternating graph shapes (MTP draft vs verify) stop fighting over one slot.**
- Upstream: `gf_res_prev` became `std::array<..., 2>`, picked by `n_outputs > 0`, with a `gf_res_prev_active` pointer. Both arenas share the context's single scheduler, so a ggml graph is only reused when its arena is the one currently allocated. The gain is that the CUDA backend sees stable graph keys per arena and can reuse captured CUDA graphs. Default on.
- Fork: `gf_slots` keyed by `(gtype, n_tokens)`, and **each slot owns its own scheduler**. A hit therefore skips the ggml graph build and allocation, not only the CUDA re-capture. Slot 0 aliases the single `gf_res_prev`. Opt-in via `WP_GRAPH_RESULT_SLOTS=N` (default 1 = legacy).

**Kept: fork.** Removed upstream's 2-arena array, `get_gf_res_prev()` and `gf_res_prev_active`, including the parts that auto-merged cleanly (`sched_reserve`, `process_ubatch`, `opt_epoch_iter`).
Why:
1. The two designs cannot both own `gf_res_prev`. Many fork paths treat it as the single slot-0 result tied to the context `sched`: the TP-overlap rolling loop (`gf_res_prev` / `gf_res_overlap`), the Qwen4exp staged layer-cut path, `output_project`'s max-nodes sizing, and `wp_reset_graph_results()`. Lazily allocating a second arena under them would break the "one sched sees one topology" invariant that the fork's `c7b8001ff` post-mortem (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`) is about.
2. The fork's design is a functional superset. With `WP_GRAPH_RESULT_SLOTS>=2` the verify and draft shapes land in separate slots, each with its own allocator, so there is real graph reuse instead of only CUDA-graph reuse.
3. Upstream's gain depends on CUDA graphs. On this fork's ROCm fleet they only apply with `GGML_HIP_GRAPHS`, which is off by default.
4. The `gf_res_prev_active` guard only exists because upstream's two arenas share one scheduler. The fork's per-slot scheduler already keeps that invariant, and `opt_epoch_iter` still calls `res->reset()`, which makes the next `can_reuse()` fail.

**Needs your review:** if you run NVIDIA with CUDA graphs and default `WP_GRAPH_RESULT_SLOTS=1`, this loses upstream's MTP CUDA-graph gain. Setting `WP_GRAPH_RESULT_SLOTS=2` is the fork-native equivalent.

### src/llama-graph.cpp - HC-arch list
Upstream added `LLM_ARCH_MAPLE` to the DSv4-style hyper-connection arch check and the fork had added `LLM_ARCH_DEEPSEEK41`. Kept both.

### src/llama-model.cpp - meta-split tensor config
Upstream mirrors every tensor for `LLM_ARCH_HRM_TEXT` (aliased cache slots). The fork added NextN/MTP and ml8-sidecar split rules at the same spot. Put upstream's arch check **first** so an HRM model never reaches the fork's per-tensor rules.

### src/models/kimi-k3.cpp - GDN l2 norm
Both sides made the same fix (`ggml_l2_norm` -> `build_gdn_l2_norm`). Took upstream's text (`eps_norm` naming).

### src/models/qwen4exp.cpp - HC grouped-norm fusion (upstream #28896 vs fork `WP_QWEN4EXP_FUSE_HC_NORM`)
**Overlap.** Both sides wanted RMS_NORM and MUL next to each other so the backend can fuse them.
- Fork: opt-in env flag that reshapes the `[hc_dim]` gamma into a 3-D view inside the graph.
- Upstream: loads the gamma as `[n_embd, hc]` (`TENSOR_ALLOW_RESHAPE`) and always multiplies before the reshape. Same math, no env flag, no graph reshape.

**Kept: upstream.** It gives the fused form by default with fewer nodes. It also has to win, because the tensors now load as `[n_embd, hc]`. The fork's fallback branch (reshape to `[hc_dim, nt]`, then multiply) would no longer broadcast. `WP_QWEN4EXP_FUSE_HC_NORM` is gone (it is now always effectively on).
**Extra port:** the fork's MTP/NextN block (not in upstream) created `hc_attn_norm` / `hc_ffn_norm` as `{ hc_dim }`. Changed both to `{ n_embd, hc }` with `TENSOR_ALLOW_RESHAPE`, because they go through the same `build_hc_mix`. Without that, the MTP head would assert on the broadcast.

### src/models/qwen4exp.cpp - sparse FA (upstream #28770)
Upstream turned on the sparse attention hint (`top_k->ne[0]`). The fork added the TurboQuant Q forward-rotation just before that call. Kept both. The top-k value is only an `n_kv_max` hint that the CUDA MMA-f16 kernel uses. Turbo K goes through the vec kernel, which ignores it, and the mask stays authoritative, so correctness does not depend on it.

### src/models/dflash.cpp - Gemma4 DSpark backbone (upstream #29226) and build_arch_graph move (#28934)
- Layer tensors: kept the fork's optional softplus attention gate (Laguna-style drafter) and added upstream's optional post-norms, `out_scale` and `rope_freqs`.
- Upstream moved `build_arch_graph()` below the graph template specialisations. Dropped the fork's copy at the old spot (the moved copy is identical) and kept the fork's `build_dflash_encode()` helper there.
- Embedding scale: upstream added `f_embedding_scale` after `get_rows`. The fork's three-way embedding source (own table / borrowed target table / DSpark services mode, where the driver passes in gathered target rows) was kept. The scale is applied **after** all three, because the services-mode rows are the target's raw table rows, the same thing upstream scales.

### ggml-cuda/mmq.{cu,cuh} - MMQ tile sizing (upstream #28552 vs fork `WP_EXPERT_MM_PIN`)
**Overlap: both change the column count that MMQ picks its J tile against.**
- Upstream: new `mmq_args::ncols_opt`. On RDNA3/RDNA4 the routed-MoE path sizes tiles against the average tokens per expert (`ne12*n_expert_used/ne02`) instead of `ne12`. Everywhere else it is the same as before.
- Fork: when `force_mm_id` is set (the pinned expert mul_mat), tiles are sized against a fixed reference width (`MMQ_FORCE_MM_REFERENCE_TOKENS`, override `GGML_MMQ_PIN_REF_TOKENS`) so results do not depend on batch width.

**Kept: both, with the fork taking priority.** `ncols_tile = force_mm_id && pin_ref_tokens > 0 ? pin_ref_tokens : args.ncols_opt`. The pin is a correctness contract (width-invariant expert outputs, which the wp-expert-worker path relies on). Upstream's gain is performance and still applies to every non-pinned mul_mat. `ncols_opt` goes right after `ncols_max` in the struct, before the fork's defaulted `force_mm_id` / `expert_ptrs`, so the positional brace-init in `mmq.cu` fills the correct field.
Also kept the fork's gfx80x (GCN4 and earlier) decode-only MMQ rule and took upstream's wider Vega/gfx909/gfx90c comment.

### ggml-cuda/ggml-cuda.cu - BF16 cuBLAS fallback (upstream #28846 vs fork `4019c9a7b`)
**Overlap.** Both fall back from BF16 to F32 cuBLAS compute on GPUs without fast BF16.
- Fork: always F32 when `!bf16_mma_hardware_available(cc)`.
- Upstream: F32 only above a batch threshold (AMD `ne11 > 32`, NVIDIA `> 8` from Volta, else `> 128`) when `!fast_bf16_hardware_available(cc)`. Small batches keep BF16 compute.

**Kept: upstream.** It was benchmarked and reviewed upstream (J. Gäßler co-author). It covers the fork's case for prefill-sized batches and avoids a conversion pass for small (decode) batches. Both use the same hardware sets on AMD/NVIDIA and both fall back to F32, not F16, so exactness is the same. Also kept the fork's FP8_B128 AITER unpack at the top of `ggml_cuda_mul_mat_cublas`.
**Needs your review:** on RDNA2/GCN with BF16 weights at batch <= 32, cuBLAS/hipBLAS now runs BF16 compute again instead of F32. If your earlier DS4 measurements showed this was slow at decode sizes, restore the fork's unconditional arm.

### ggml-cuda/allreduce.cu - internal AllReduce on ROCm (upstream #27825 vs fork HIP port)
**Overlap: both sides ported the internal two-GPU AllReduce to HIP.**
- Upstream: the minimum. Drop the `GGML_USE_HIP` exclusion, use `__builtin_amdgcn_s_sleep(4)` in the spin, and change the Volta comment.
- Fork: a full port. `hipHostMalloc(Portable|Mapped|Coherent)` fine-grained staging so device stores reach host memory and not L2; `__hip_atomic_*` at system scope for the arrival handshake; a `cudaDevAttrCanMapHostMemory` capability check with fallback; a wall-clock spin watchdog; plus the duplex transport, codecs and diagnostics built on top.

**Kept: fork** in all 9 hunks. It is a strict superset of upstream's change (same s_sleep idea, wrapped in `ggml_cuda_ar_spin_pause()`), and the coherence and atomic-scope pieces are what make it correct on multi-GPU PCIe/TB3 ROCm setups. Upstream's version leaves device stores to coarse-grained pinned memory with volatile accesses. Upstream's non-conflicting `CUDA_CHECK()` wrapping of teardown calls merged in cleanly and was kept. The MUSA stub keeps the fork's comment (it correctly says HIP *is* supported).

### ggml-vulkan - upstream file split (#28732) plus 30 upstream Vulkan commits
**The largest piece of this sync.** Upstream moved ~5,400 lines out of `ggml-vulkan.cpp` into `ggml-vulkan-types.h`, `ggml-vulkan-push-constants.h`, `ggml-vulkan-common.h`, `ggml-vulkan-buffers.cpp` and `ggml-vulkan-debug.cpp`. Many functions went from `static` to shared (declared in `common.h`), and `ggml_vk_dispatch_pipeline` became an `inline` template in `common.h`. The fork had ~2,800 changed lines spread across those regions. A hunk-by-hunk text merge gave 90 diff3 hunks, most of which were "fork code sitting in a region upstream deleted", so resolving them in place would have silently dropped the moved-out copies.

**Method:** started from upstream's split files and replayed the fork's own patch (`git diff 427291b5b HEAD -- ggml-vulkan.cpp`, 141 hunks). Each hunk was tried against `ggml-vulkan.cpp` and then each new file: 93 applied with exact context, 10 with fuzz 1 (each placement checked by hand), and 38 were ported by hand. Afterwards a line-level audit confirmed every fork-added line exists somewhere in the merged tree, except the lines replaced on purpose below.

Hand-port decisions worth knowing about:
- **Queue quiescence epochs** (`WP_VK_HOST_READ_FASTPATH`): the atomics and helpers moved to `vk_queue_handle` in `types.h`. The `submit_epoch` bump goes into both `submit()` bodies, which are now out-of-line in `.cpp` and also carry upstream's NVIDIA `device_submit_mutex` workaround (#28830). The bump sits after the device lock, as before.
- **Graph plans** (`ggml_backend_vk_graph_plan`, `recording_plan`): the plan struct is in `types.h`. The descriptor pool/set accessors moved to `common.h` as `inline`, because upstream's `ggml_vk_dispatch_pipeline` template now lives there and has to see them. The fork's `dispatch_pipeline` (takes `vk_subbuffer`, records descriptor updates for plans) replaces upstream's copy in `common.h`.
- `ggml_vk_ctx_begin(..., bool one_time = true)`: the default moved to the `common.h` declaration.
- `vk_node_ts_enabled` is a shared global next to upstream's other logger flags in `ggml-vulkan-debug.cpp` (extern in `common.h`). `vk_want_timestamps()` is `inline` in `common.h`.
- Forward declarations the fork added for `ggml_vk_buffer_from_host_ptr` / `ggml_vk_is_empty` were **dropped**. Upstream now has both as shared functions, and a `static` redeclaration would not compile.
- **MUL_MAT pin (`GGML_VK_PIN_REF_TOKENS`, `force_mm`) vs upstream's pipeline maps (#25773):** upstream replaced `ggml_vk_guess_matmul_pipeline(ctx, mmp, ..., type, type)` with `..._map(ctx, *mmp_map, ...)`. The fork's `pipeline_n` override now goes into the map-based calls for both `mul_mat_q_f16` and `mul_mat_id_q_f16`, so the pinned tile choice is unchanged.
- **`ggml_vk_mul_mat_vec_q_f16` signature:** upstream added `bool swap_inputs = false` (#28457, one-row B^T*A trick) and the fork had added `bool pin = false` in the same position. The merged signature is `(..., bool swap_inputs = false, bool pin = false)`, and the fork's only `pin` caller (`ggml_vk_mul_mat_pinned_vec`) now passes `false, true`. Without this, the pinned path would have silently turned on input swapping.
- Upstream's new swap-inputs branch tests `dst->ne[1] > mul_mat_vec_max_cols`. In this fork that constant is the 16-wide array bound, not the working cap, so the check was changed to `ctx->device->mul_mat_vec_max_cols_eff`, to match the fork's per-device GEMV cap.
- `ggml_vk_mul_mat_q_f16`: the fork had moved the buffer lookup block below pipeline selection (graph-plan dynamic subbuffers). The upstream-position copy was removed so it is not declared twice.

### Vulkan shader type ids - `GGML_TYPE_Q2_0`
Upstream added `vulkan-shaders/ggml_type_ids.glsl` with `#define GGML_TYPE_Q2_0 42u` and moved the FA shaders from `FA_TYPE_*` to `GGML_TYPE_*`. In this fork 42 is `TURBO3_0`, and Q2_0 was renumbered to 56 in the 2026-08 sync (see the NOTE(fork) in `ggml.h`). **Changed the shader define to 56u** and added the fork's `TURBO2/3/4_0` ids there. The fork's `FA_TYPE_Q1_0` / `FA_TYPE_TURBO4_0` cases in `fa_types.glsl` / `flash_attn_dequant.glsl` were renamed to the new `GGML_TYPE_*` names. Without this, a Q2_0 matmul on Vulkan would have used the TURBO3_0 branch.

## Found by the post-merge build (fixed in the follow-up commit)

The merge commit was pushed before the compile check finished. Building with clang (CPU + Vulkan + all tests) found these. They are fixed in the commit after the merge.

- **Router deadlock (important; corrects the `unload_lru()` decision above).** Upstream #28555 removed the router's stopper thread and its `cv_stop`. Stops now go through `request_stop()`, which **returns early when the model is already in `stopping_models`**. The fork's code still used "insert into `stopping_models`, then wake the stopper thread", in `unload_lru()` and in both GPU-placement eviction paths. As merged, a victim would be marked stopping, `unload()` -> `request_stop()` would then no-op, and the router would wait forever for a model that was never told to exit. All three sites now call `request_stop(victim, !loading)` under the same lock that picked the victim. The fork's guarantee (a concurrent `unload_lru()` cannot choose a second victim) still holds, because `request_stop()` marks the victim stopping under that same lock.
- `src/models/models.h`: both sides had picked up upstream #28068 (`build_gdn_l2_norm`) at different spots, so git kept two definitions. Removed one.
- `common/speculative.cpp`: upstream renamed `common_speculative_draft_params::n_past` to `pos0` (#28715, "it is a position, not a count"). Updated the fork's `WP_DSPARK_DEBUG` log lines.
- Vulkan: the fork's `ggml_vk_ensure_host_read_staging_buffer` / `ggml_vk_ensure_wp_fused_batch_scratch_buffer` landed in `ggml-vulkan-buffers.cpp` (next to `ensure_sync_staging_buffer`, where upstream moved it) but are called from `ggml-vulkan.cpp`. Made them shared and declared them in `common.h`.
- Vulkan, **two misplaced replay hunks** (identical context text in two places):
  - The fork's `d_X_buf` / `d_Y_buf` dynamic-subbuffer block and the `d_Qy_copy` quantize source belong to `ggml_vk_mul_mat_q_f16` but had landed in `ggml_vk_mul_mat_id_q_f16`. Moved back.
  - `case GGML_TYPE_TURBO4_0:` in `supports_op` belongs in the F32->TURBO4_0 (quantize) CPY case but had landed in TURBO4_0->F32. As merged, the backend would have claimed an op it does not implement. Moved back.
  To catch any others, every fork-modified function was checked with a per-function 3-way merge (base / fork / upstream) against the ported body. After the fixes, the only differences left are the hand-merged functions described above.

### Pre-existing on origin/master (not caused by this merge; not changed)
- `ggml/include/ggml-ml8.h` (from `55bc7c6ea`) declares a C++ overload of `ggml_fp8_quant_rot` inside `extern "C"`. Clang accepts it, **GCC rejects it** ("conflicting declaration of C function"). Every build here used clang, which is what the ROCm toolchain uses anyway.

## Verification

- Build: clang, `-DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON`, Release. All 1014 targets build (libllama including the unity-built `models/*.cpp`, llama-server, the wp-expert-worker/dispatcher tools, all tests). No CUDA/HIP toolchain was available in the sync environment, so **the `.cu` changes (FA instance selection, mmq tile sizing, BF16 cuBLAS fallback, allreduce) are only checked by reading, not compiled.** Build `-DGGML_HIP=ON` before merging to master.
- `ctest -LE model` on CPU (no GPU in the sync environment): 90/106 pass. All 16 failures were checked against a clean build of pre-merge `origin/master` (`26e5f58`) in a separate worktree:
  - 11 fail identically on origin/master, so they are **pre-existing**: test-arg-parser (asserts upstream's `n_outputs_max_per_seq == 1`, which the fork deliberately does not use; see speculative.cpp), test-dsv41-load/-decode, test-recurrent-state-rollback/-dsv4, test-save-load-state, test-paged-decode-oracle, test-wp-expert-worker, test-routed-experts-external, test-quantize-fns, test-ml8-registry.
  - 4 need model downloads (no network in the sync environment): test-download-model, test-eval-callback(-download-model), test-thread-safety.
  - test-barrier timed out only under `-j4` load. It passes when run alone on both trees.
- Not verified here (needs your hardware): a GPU run of test-backend-ops on ROCm/Vulkan, a router run with `--gpus` placement plus `models_max` eviction (to exercise the deadlock fix), and a DSpark/DFlash draft on a Meta-split target.
