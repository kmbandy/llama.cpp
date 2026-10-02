# Upstream sync 2026-10-02

Merge of `upstream/master` (ggml-org/llama.cpp @ `bed0a8566`) into the fork (origin/master @ `9d59f23e7`).

- Previous sync: `docs/dev/2026-09-24-upstream-sync.md` (upstream `bd4f514db`, now the merge base)
- Upstream commits merged: 219
- Conflicted files: 42 (~190 diff3 hunks)
- Branch: `claude/hello-ky3ivj`, restarted from origin/master (the previous sync branch was merged). Not master.

Same protocol as last time. No blind `--theirs` on customized files. Where upstream added something that does the same job as fork code, both versions were compared and one kept, with the reason given. Afterwards, fork-added lines missing from the result were audited, the tree was built with clang (CPU + Vulkan + tests), and failures were compared against a clean build of pre-merge origin/master.

Only hunks that needed a decision are listed. Two independent additions sitting next to each other are not.

## Decisions

### ggml-cuda/mmq.{cuh,cu} - W4A4 `prec_src1` (upstream #24364) vs fork `force_no_stream_k`
Upstream's model-driven precision policy added a `ggml_prec prec_src1 = GGML_PREC_Q8` template parameter to every MMQ kernel, tile processor and launcher, in the same slot where the fork had added `bool force_no_stream_k = false` (pinned expert mul_mat_id: no stream-k, so results do not depend on batch width). These do different jobs, so both are kept. The combined signature is `<type, J, fallback, prec_src1 = GGML_PREC_Q8, force_no_stream_k = false>`. Upstream's parameter goes first, so every upstream call site and the `DECL_MMQ_CASE_W4A4` instantiations work unchanged. Fork call sites pass both. The fork's `mul_mat_q_switch_J` -> `mul_mat_q_switch_J_impl<.., prec_src1, force>` dispatch wrapper now forwards `prec_src1`. Upstream moved the host stream-k decision into `config.stream_k`, so the fork's launch condition is now `force_no_stream_k || !config.stream_k` and its `GGML_MMQ_DEBUG_SYNC` print reads `config.stream_k`. In `mmq.cu`, kept the fork's `expert_ptrs` binding and debug-sync stage next to upstream's new `prec_src1` argument.

### ggml-cuda/norm.cu + ggml-cuda.cu - RMS_NORM -> SCALE fusion (upstream #29393 vs fork `55bc7c6ea`)
**Overlap: both sides fused the GDN q/k L2-norm (`build_gdn_l2_norm` = `scale(rms_norm(x, eps/n), 1/sqrt(n))`) into one launch, and both are bit-identical to the unfused ops.**
- Upstream: a `do_scale` template flag + `scale_out` argument on `rms_norm_f32`, a new `ggml_cuda_op_rms_norm_scale_fused()`, and a `ggml_cuda_can_fuse` branch that **`GGML_ASSERT`s the RMS_NORM input/output are F32**.
- Fork: a runtime `post_scale` argument on the (BF16-templated) `rms_norm_f32` and on the `MT_WIDE_KERNELS` `rms_norm_rows_f32` fast path, and `ggml_cuda_op_rms_norm_fused_scale()`, which **returns false** (falls back to two ops) for bias, non-F32 or `ne00 >= 1024`.

**Kept: fork.** The fork runs BF16 activation streams (`LLAMA_ACT_BF16`). A BF16 RMS_NORM -> SCALE would have hit upstream's assert and aborted, while the fork's path just declines. The fork's version also uses the rows fast path. Upstream's fused function, its declaration and the `do_scale` kernel variant were removed, and the try-fuse call dropped. Upstream's `ggml_cuda_can_fuse` RMS_NORM+SCALE branch was kept, but its two asserts now `return false`, so any other caller is BF16-safe too. Upstream's `RMS_NORM_SCALE` test-backend-ops cases still cover the fused result (the fork's path fuses the same narrow shapes).

### ggml-cuda BF16 elementwise ops (upstream #29675 vs fork "BF16 activation coverage" 2026-09-18)
**Overlap: both sides added BF16 to binbcast/unary/glu/scale.** The kernel changes were the same in substance. In the binbcast dispatchers git kept **both** copies of the BF16xBF16->BF16 and BF16xF32->BF16 branches. Upstream's come first, so the fork's copies were dead and were removed. The fork's extra BF16 src0 -> F32 dst branch was kept and narrowed to `src1 == F32` (it always passed src1 as `float*`). `scale.cu`: kept the fork's version (the same BF16 template plus its `scale_f32_vec16` wide path). unary: kept the fork's assert, which also allows a BF16 dst.
**Bug fixed by upstream that the fork had:** the F32 dispatch branch now requires `src1 != BF16`. Before, F32 src0 + BF16 src1 matched the F32 branch and read BF16 data through `float*`, and the fork's `supports_op` advertised that combination. `supports_op` for ADD/SUB/MUL/DIV now lists exactly what the dispatcher implements: upstream's BF16 rule plus the fork's BF16 src0 -> F32 dst. Kept the fork's `TURBO_WHT` and upstream's BF16 `SCALE` cases.

### ggml-cuda/mmvq.cu - fused shared experts (upstream #29184) vs fork restrict locals + weight paging
Upstream's shared-expert fusion adds one extra grid channel per MoE matvec that reads `fusion.shared_up` and writes `fusion.shared_dst`, by reassigning the `vx`/`dst` parameters. The fork had renamed those parameters to `vx_ptr`/`dst_ptr` and binds `GGML_CUDA_RESTRICT` locals from them after `ggml_cuda_pdl_sync()` (fork `53a55c37c`). Upstream's reassignment now targets `vx_ptr`/`dst_ptr`, so the restrict locals pick up the shared-expert pointers. Kept the fork's clamp fields in the fusion setup.
**Merge-only bug fixed:** the fork's MAD-88 weight-paging hook replaces the weight base with `expert_ptrs[channel_x]`. On the shared-expert channel `channel_x` is 0, so with paging on that channel would have read **routed expert 0's weights in place of the shared expert's**. Both kernels now skip the remap when `shared_expert` is set.

### ggml-cuda/ssm-scan.cu - CUB
The fork enabled CUB (hipcub) on HIP and upstream enabled it on MUSA. Kept both: `HIP || MUSA -> USE_CUB`, otherwise `CUDART >= 11070`.


### ggml-cpu/ggml-cpu.c - upstream tiled MUL_MAT_ID (#27851) replaces iqp
Upstream removed the `iqp` fast path that the fork had guarded with `expert_ptrs == NULL`, and replaced it with `ggml_compute_forward_mul_mat_id_tiled()`. The same guard was moved onto the tiled call: the tiled kernel indexes `src0` by expert slot, so it must not run when the expert worker supplies its own per-expert base pointers (WP gather / BATCH_MMID arena). `CMakeLists.txt` keeps the fork's `wp-gemm.{cpp,h}` and drops `iqp`.

### ggml-backend-meta.cpp - zero-fill
Upstream now zero-fills with `GGML_OP_FILL` plus a `ggml_nelements > 0` guard, where the fork had used a `SCALE` by 0. Took upstream's: FILL does not read its input, so NaN/Inf garbage can't survive as `0 * NaN`.

### common/arg.cpp - `--rpc`
Took upstream's handler (it throws when RPC is not compiled in), plus the fork's `params.rpc_enabled = true`, which the fork's RPC worker startup reads.

### Qwen4Exp MTP - upstream #29761 vs fork's port of #27739
**Overlap: both sides added the Qwen4Exp (Qwen3.8-Flash-Next) MTP draft head, converter and graph.** They differ in three places. The sglang reference implementation (`qwen4_exp_mtp.py`) settled each one:

| | fork | upstream | reference | kept |
|---|---|---|---|---|
| MTP mixer | dropped `mtp.hyper_connection_mixer` in a full conversion and reused the trunk's `hc_head_*` (the fork's own comment: "a separate blk.N.nextn slot is future work") | maps it to `blk.N.nextn.hc_head_{norm,down,up}` | the MTP model is a full `Qwen4ExpModel(is_nextn=True)` with its own mixer | **upstream** |
| `pre_fc_norm_hidden` | per-stream `{n_embd}` norm | one `{n_embd, hc}` norm over the whole hc-wide row | `GemmaRMSNorm(hc_count * hidden_size)`: one norm over the whole row | **upstream's shape**: `rms_norm` over `n_embd*hc`, then the `{n_embd, hc}` gamma |
| MTP attention | dense (compress ratio 0) | QSA (compress ratio = the trunk's) | `layer_types = ["full_attention"]`, which in Qwen4Exp is the QSA layer class | **upstream** (converter writes `ratio`) |

The fork's in-file comment on the full-conversion bug it fixed (the MTP mixer landing on the same `model.hyper_connection_mixer.*` key and overwriting the trunk's) no longer applies: upstream's mapping gives the MTP mixer its own `blk.N.nextn` key.
**Compatibility for GGUFs from the fork's older converter** (already on disk, no `nextn.hc_head_*`, MTP ratio 0):
- `nextn.hc_head_*` load as `TENSOR_NOT_REQUIRED`, and `graph_mtp` falls back to the model-level `hc_head_*` when they are absent. This reproduces the fork's old behaviour.
- The MTP layer's indexer tensors are `TENSOR_NOT_REQUIRED` when its ratio is 0, and `build_layer_attn` already runs dense there.

`fc_embedding`/`fc_hidden`: both sides concatenate them into `eh_proj` (exact, since `A*e + B*h == [A|B]*[e;h]`). Kept upstream's version (in `modify_tensors`) and dropped the fork's duplicate `generate_extra_tensors` pass, along with the fork's second `filter_tensors`, which shadowed upstream's.
Kept from the fork: the PLE row-width derivation for a spine converted without the table (wp-forge sidecar). Upstream's `mtp_only` early-out replaces the fork's equivalent.
**Needs a re-convert to benefit:** a fork-era MTP GGUF still loads and drafts, but on the trunk mixer and dense attention as before. Re-converting gives the reference-faithful draft head.

### src/models/qwen4exp.cpp - rebuilt on upstream's file
Upstream restructured the file (MTP graph, `build_layer_attn` taking the k-pool input, `n_ff_exp(il)`). The fork's changes were replayed onto upstream's file rather than the reverse:
- `WP_QWEN4EXP_LAYER_CUT` staged trunk: the trunk ctor body was split into `build_trunk_layer()` / `build_trunk_head()`, so `build_trunk_staged()` and the whole-graph path share one layer body. The k-pool input is built in both.
- The expert-dispatch `keep_all_rows` / `res_hc` path, `WP_DISPATCH_SPLIT_SHEXP`, and routed-expert-external `TENSOR_SKIP`.
- Per-layer PLE (`build_ple`, with `WP_PLE_SKIP` / `WP_PLE_TRACE`) and the `can_reuse` row check.
- TurboQuant: the Q forward WHT before `build_attn_mha` on the QSA path.
- Dropped as dead: the fork's turbo4 indexer un-rotation. The indexer cache is forced to F16 in `llama-model.cpp`, so it could never trigger.

### gguf-py constants / llama-arch / llama-model.h - NEXTN_HC_HEAD
DS4's MTP head uses `NEXTN_HC_HEAD_{FN,BASE,SCALE}` (fork); Qwen4Exp's uses `NEXTN_HC_HEAD_{NORM,DOWN,UP}` (upstream). They are different tensors, so both were kept. The QWEN4EXP tensor list is upstream's plus the fork's Qwen3Next-style `NEXTN_EMBED_TOKENS` / `SHARED_HEAD_*` slots.

### llama-memory-hybrid-idx - full-context `ctx_idx` stays null (fork divergence kept)
Upstream's ctor for the full memory context now also seeds the QSA k-pool state. The fork deliberately leaves `ctx_idx` null there (MAD-LAB 2026-08-27: reserving the sparse graph costs about 1.8 GiB of compute buffer and stops the spine fitting beside the worker). **Kept the fork's.** It is consistent with upstream's k-pool code: `kpool_track()` requires a non-empty `ns_ubatch`, so it is off in the full context, and the qwen4exp graph builds the k-pool input only when `get_idx()` is non-null.

### llama-memory-hybrid.cpp - state_read rollback
Upstream added "undo the attention restore if the recurrent restore throws". The fork routes attention through `attn_base()` (normal or paged cache). The paged cache has no `state_clear`, so its undo is `seq_rm(seq_id, -1, -1)`.

### llama-batch.cpp
Upstream restructured `llama_batch_allocr::init` around `llama_batch_ext`. The fork's warn-once for the "embeddings required but some tokens not marked as outputs" override (an embedding server hits it on every batch) was moved into upstream's new location.

### llama-graph.cpp
- `build_attn` gained upstream's training bypass, which uses `k_cur`/`v_cur` directly. The fork's TurboQuant Q pre-rotation runs after `k`/`v` are chosen, so it still keys on the cache type.
- Rerank mean-first: upstream generalized `arch == MODERN_BERT` to `pooling_type_cls == MEAN`. Kept `|| arch == LLAMA_EMBED` for the fork's llama-nemotron-rerank support (`899858e64`), whose GGUFs may not carry the key.
- Arch lists gained both `DEEPSEEK41` (fork) and `GLM5_NEXT` (upstream).

### llama_batch_ext migration (upstream #24669 / #29385 / #29601) - the big one
Upstream moved `llama_context::decode/encode` to a new `llama_batch_ext`. `llama_decode(llama_batch)` survives as a compat wrapper that copies into one. It then migrated speculative, server, mtmd and the examples to a C++ `common_batch`, and deleted `common_batch_add` / `common_batch_clear`.

**src/llama-context.cpp - adopted the new entry points and adapted the fork's decode logic to them:**
- **Embd row width.** The fork picks it per batch, and the comment history explains each gate: DSpark contexts take `n_embd_out` rows; DFlash embd-only batches take the encoder width unless `dsv4_hc_mult > 0`; a services-mode draft batch carrying token ids keeps `n_embd_inp`. Upstream picks one width per context type. This logic is now `llama_fork_n_embd()`. The `llama_batch` compat wrapper uses it to size the copy, and `decode(llama_batch_ext)` accepts either upstream's per-context width or the fork's per-batch width.
- **Output flags.** The compat shim defaults a `llama_batch` without logits to "last token only". The fork's sparse-output paths read the flags, so:
  - The decode wrapper marks every row as output for an embeddings context when no logits were passed. This is what the fork's `output_all` produced before.
  - Sparse **encoder** outputs (the DSpark encode) now apply only to a legacy `llama_batch` that carried logits (`encode(batch_ext, sparse_outputs)`). `llama_batch_ext` callers get every row, as upstream does. Otherwise upstream's encoders would have been trimmed to one row.
- **TP mirroring.** Added `tp_mirror_batch(const llama_batch_ext &)`. It flattens the ext batch into the existing token-only wire format, so a leader called through `llama_process()` still mirrors.
- **Layer-input / NextN row order (upstream #29019).** Upstream now restores the original batch order of layer inputs and unmasked NextN rows after the fact (`embd_batch_idxs`), instead of applying the logits swap to them. That swap was wrong for multi-sequence batches, so the fork's own swap of layer inputs was dropped for upstream's mechanism, adapted to the fork's row widths (`n_embd_layer_inp`, `n_embd_nextn`) and to its external (wire-supplied) layer inputs, which stay unpermuted.
- **Narrow sync.** `WP_LAYER_INP_NARROW_SYNC` now also requires that this order map is the identity. Without that, a narrow read could return rows that are still pending the reorder.
- `extract_layer_inputs` keeps the fork's `sched_override` and returns upstream's `bool`. The fork's DS4.1 trimmed-row placement is kept (upstream's `row_floats == n_embd` assert would fire on the fork's hc-wide taps).

**common/speculative.cpp + tools/server/server-context.cpp - kept on `llama_batch`.** These are the fork's most heavily customized files: DSpark/DFlash fused and split injection, MTP prefill pipelining and pinned handoff, the pipeline-streams server, TP and paged tiers. Upstream's migration rewrote batch handling throughout both. Porting thousands of fork-only lines to `common_batch` with no GPU or models available to validate it would risk silent breakage for no functional gain. Instead, each file was rebuilt from the fork's version by replaying every **other** upstream commit 3-way and skipping only the two migration commits:
- probabilistic draft sampling + rejection verify (#27694). This was ported into the fork's MTP and draft-simple, including the `begin()` sampler reset, retune and `result_q`.
- the `/v1/systemone` decision task (#29818). `send_decision` was translated to `llama_batch` and the fork's `stream_slot_idx`.
- the nimble model (#29844), typed-content embeddings (#29556), the mtmd ubatch cap (#29773), RANK pooling splitting (#28876) and the GCC 12 fix (#29325).

`common_batch_add` / `common_batch_clear` were restored in `common/` for these two files. `llama_decode(llama_batch)` remains a supported upstream API, so this is a deferral, not a dead end. **Follow-up worth scheduling:** migrate the two files to `common_batch` in a dedicated change that can be tested on the rig. Until then, every upstream server or speculative change will need this same translation.

### ggml-vulkan - descriptor-set reuse (#29280) vs fork graph plans
Upstream skips `updateDescriptorSets` when a set already holds identical bindings, using a cache indexed by the context's set index. The fork's graph plans record into their own pools and sets (`ggml_vk_descriptor_sets(ctx)`), so that index would name a different set. Reuse now applies only when `recording_plan == nullptr`. A plan writes its sets once, while recording. Kept the fork's array-based binding conversion.

### ggml-vulkan - mul_mat ranges (#28956) and MoE tile selection (#29182)
- **#28956** binds in-place A/B with their strided extent (`ggml_nbytes`), using `ggml_vk_batch_stride()` for the batch stride:
  - mul_mat: the fork builds the A/B subbuffers early (for graph-plan dynamic binding), so the ranges are now computed before that.
  - mul_mat_id: the fork's weight-pager pool binding defaults to `x_range`. The fork's own `src0_arena_stride` block was dropped because `ggml_vk_batch_stride` computes the same value (`nb[2]/type_size*blck_size`), and the arena path asserts `!qx_needs_dequant`.
- **#29182** picks the mul_mat_id tile from rows per expert. The fork already did that for **pinned** experts (`ggml_vk_mul_mat_id_pin_pipeline_n`, floor 9). The unpinned path now uses upstream's `CEIL_DIV(nei0*nei1, n_as)`, and the pinned and env-forced paths are unchanged.

### Smaller unions
- `include/llama.h`: fork TP-follower API + upstream batch_ext API.
- `llama-io.h`: fork `read_tensor_as` + upstream `discard`.
- `llama-kv-cells.h`: fork `for_each_token_in` + upstream `seq_pos_get`.
- `llama-model.h`: fork DS4 nextn head + upstream qwen4exp nextn head; fork ml8/GDN sidecars and pagers + upstream `can_prefetch`; fork `llama_dspark_build_markov_graph` + upstream `llama_prec_policy`.
- `llama-model.cpp`: kept the fork's `done_getting_tensors(partial_load, pipeline_band)`.
- `deepseek4.cpp`: the fork's DSpark tap collapse now calls upstream's renamed `build_hc_mean`.
- `qwen35.cpp`: the fork's `output_tied` + upstream's `cls_out` projection, with the fork's `no_output_head` early return before it.
- `tools/cli/cli.cpp`: upstream's `llama_backend_init` (#29632) moved before the fork's TP-follower branch, which does not init the backend itself.
- `conversion/qwen.py` DFlash: fork `hc_mult` + upstream partial-rotary and value-scale keys.
- `test-backend-ops.cpp`: fork `TURBO_WHT` + upstream W4A8/W4A4 cases.
