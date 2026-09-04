#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

namespace pipe_expert_dispatcher {
class graph_dispatcher;
}

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    // Offer the hash-layer experts of `tokens` to the expert workers ahead of the
    // forward pass that will consume them. See llama_expert_prefetch_hint.
    // Returns hint frames sent; 0 whenever the feature or the dispatcher is off.
    int expert_prefetch_hint(const llama_token * tokens, int n_tokens, int n_certain = -1,
                             const float * conf = nullptr);

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    // MAD-LAB logits-on-head: project already-output_norm'd hidden states
    // ([n_tokens][n_embd], F32, row-major) through the LM head straight into this
    // context's logits buffer. For a dense-segment HEAD whose band graph stops
    // before the LM head but which still holds output.weight. See the definition.
    // `out` (optional) redirects the result into a caller buffer instead of this
    // context's logits, leaving this context's output state untouched.
    bool output_project(const float * hidden, int32_t n_tokens, float * out = nullptr);

    // MAD-LAB speculative services. A sidecar draft (DFlash/DSpark) ships neither a
    // token_embd nor an output.weight and used to borrow the target's through
    // ctx_other. That is impossible when the target is Meta-split, so the two borrowed
    // ops are performed HERE, on the context that owns the tensors, and the results are
    // handed to the draft as plain buffers. See the definitions.

    // gather token_embd rows on THIS context: out is [n_tokens][n_embd] F32 row-major.
    bool token_embed_gather(const llama_token * tokens, int32_t n_tokens, float * out);

    // replay the DSpark Markov/confidence head on THIS (draft) context, over base
    // logits projected elsewhere. base is [n_tokens][n_vocab] row-major, hidden is
    // [n_tokens][n_embd]; the biased logits land in this context's own logits buffer
    // and out_conf ([n_tokens], may be null) receives the acceptance confidences.
    bool dspark_markov_head(const float * base,
                            const llama_token * tokens,
                            const float * hidden,
                            int32_t n_tokens,
                            int32_t n_blocks,
                            float * out_conf);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    // dense-segment head: the nextn buffer is filled from the tail segment's
    // wire sideband instead of the local graph, which never sets the width.
    // The wire owner declares it here so *_ith accessors stride correctly.
    void set_embeddings_nextn_width(uint32_t w) { n_embd_nextn = w; }

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_no_output_head(bool value);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);

    // MAD-LAB interior taps: reserve the host buffer for a layer whose rows this
    // context cannot compute (it lies outside this process's pipeline band) and will be
    // handed over the wire instead. Unlike set_embeddings_layer_inp() this does NOT arm
    // the graph-output path, which would assert on the missing tensor.
    void set_embeddings_layer_inp_external(uint32_t lid, bool enable);
    bool is_embeddings_layer_inp_external(uint32_t lid) const;
    // Install [n_tokens][n_embd] F32 row-major rows, in batch order, for an armed layer.
    bool set_layer_inp_data(uint32_t lid, const float * data, int32_t n_tokens);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    // WP_QWEN4EXP_LAYER_CUT (Stage 3): true only for LLM_ARCH_QWEN4EXP,
    // LLM_GRAPH_TYPE_DEFAULT, a normal (non-draft) context, pooling_type NONE,
    // and ubatch.n_tokens >= 64 -- see the .cpp for the full gate, including the
    // exclusions this file adds beyond the architect's list (output_layer_inp).
    bool layer_cut_eligible(const llama_ubatch & ubatch, llm_graph_type gtype) const;

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    // WP_QWEN4EXP_LAYER_CUT (Stage 3) serial stage-list executor: replaces only
    // the whole-graph execution decision at the graph_compute(res->get_gf(), ...)
    // seam. Each of the 49 stages goes through the existing scheduler/
    // graph_compute path, one at a time, before the next stage is built --
    // no second outstanding dispatch (stage-2's k_max_open_dispatches=1 is
    // unaffected: every stage waits before the next issues).
    llm_graph_result * process_ubatch_staged(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // MAD-LAB: the target context owns the dispatcher; speculative contexts borrow it.
    std::unique_ptr<pipe_expert_dispatcher::graph_dispatcher> expert_dispatch_owned;
    pipe_expert_dispatcher::graph_dispatcher * expert_dispatch = nullptr;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};
    // MAD-LAB: nextn rows follow the graph tensor width, which can vary by graph.
    uint32_t n_embd_nextn = 0;

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    // MAD-LAB logits-on-head: dedicated scheduler for output_project(). Kept
    // separate from `sched` so the decode path's graph reuse is never reset.
    // Created lazily -- nothing allocates it on a non-segment head.
    ggml_backend_sched_ptr sched_proj;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // WP_GRAPH_RESULT_SLOTS=N (default 1 == today's behaviour exactly).
    //
    // WHY. There is exactly ONE cached graph result (gf_res_prev), but the spine
    // runs several DIFFERENT graph shapes through the same llama_context: the
    // n_tokens=4 speculative-verify trunk pass and the n_tokens=1 MTP draft-head
    // passes alternate 4 -> 1 -> 1 -> 1 -> 4 within a single decode step. Because
    // llm_graph_params::allow_reuse() compares ubatch.n_tokens, every
    // verify<->draft transition misses and rebuilds a ~7-9k-node graph. Measured
    // on the g128 spine: "graphs reused = 198" against ~700 decode() calls (28%),
    // even though 63% of verify calls already share the n_tokens=4 shape -- i.e.
    // the binding constraint is the single slot, not the batch shape.
    //
    // A slot holds a graph RESULT (the built topology), keyed by (gtype, n_tokens).
    //
    // EACH SLOT OWNS ITS SCHEDULER. This is not an optimisation, it is the only
    // sound design, and the first attempt (c7b8001ff) crashed the R9700 with
    // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION for want of it.
    //
    // A ggml_backend_sched owns a ggml_gallocr, and BOTH of the guards that decide
    // whether a graph must be re-allocated are POSITIONAL AND SIZE-BASED, never
    // identity-based:
    //   * ggml_backend_sched_alloc_splits() (ggml-backend.cpp) compares
    //     node_backend_ids[i] against prev_node_backend_ids[i] and only flags a
    //     change when the BUFT differs;
    //   * ggml_gallocr_needs_realloc() (ggml-alloc.c) compares n_nodes / n_leafs
    //     and then, per node index i, only asks `talloc->size_max >= node_size`.
    // Neither can tell "the same graph re-run" from "a DIFFERENT graph that happens
    // to have the same node/leaf counts and whose tensors fit the previous
    // offsets". When both guards pass, ggml_gallocr_alloc_graph() runs
    // ggml_gallocr_init_tensor() over the new graph using the OLD graph's
    // node_allocs[i], assigning tensor i the offset AND buffer_id computed for a
    // different tensor i. Going from the 4-token verify graph to the 1-token draft
    // graph every tensor "fits", so nothing trips: tensors silently alias, and one
    // whose true extent exceeds the recorded chunk runs off the end of its buffer.
    // With ~96 CPU splits per layer-set, index i can even be CPU in one graph and
    // ROCm0 in the other, handing a kernel a host pointer.
    //
    // The whole reuse machinery rests on the invariant that ONE sched sees ONE
    // graph topology at a time. Keeping two graph results against one sched breaks
    // that invariant; a per-slot sched restores it, and lets a cross-slot hit skip
    // allocation entirely rather than re-running it.
    //
    // See also the comment above output_project() in llama-context.cpp: it gives
    // the projection head a DEDICATED sched for exactly this reason.
    //
    // COST. One compute buffer per extra slot. It is NOT the +429 MiB worst case:
    // an extra sched is never graph_reserve()d, so ggml_gallocr_reserve_n() sizes
    // it on first use for the shapes that slot actually sees, and a slot keyed to
    // n_tokens=1 (the MTP draft head) is far smaller than the n_ubatch=512
    // worst-case reserve. Slot 0 keeps the context's reserved sched unchanged.
    struct wp_graph_slot {
        // NON-OWNING. Slot 0 points at gf_res_prev (owned by that unique_ptr);
        // slots 1..N-1 point into gf_slots_owned below. Keeping this raw avoids
        // a second unique_ptr claiming ownership of the gf_res_prev object.
        llm_graph_result *   res      = nullptr;
        // NON-OWNING. Slot 0 points at this context's `sched`; slots 1..N-1 point
        // into gf_slots_sched below. Every slot MUST have its own allocator.
        ggml_backend_sched_t sch      = nullptr;
        llm_graph_type       gtype    = LLM_GRAPH_TYPE_DEFAULT;
        uint32_t             n_tokens = 0;
        bool                 keyed    = false;
        uint64_t             last_use = 0;
    };

    // empty when WP_GRAPH_RESULT_SLOTS <= 1; otherwise slot 0 aliases gf_res_prev
    // so that every pre-existing gf_res_prev code path keeps working untouched.
    std::vector<wp_graph_slot> gf_slots;
    // owns only the EXTRA results (slots 1..N-1); slot 0's object is gf_res_prev's
    std::vector<llm_graph_result_ptr> gf_slots_owned;
    // owns only the EXTRA schedulers (slots 1..N-1); slot 0 uses `sched`
    std::vector<ggml_backend_sched_ptr> gf_slots_sched;
    // the scheduler the CURRENT process_ubatch/graph_compute call must drive.
    // Always sched.get() in single-slot mode.
    ggml_backend_sched_t sched_active = nullptr;
    uint64_t                   gf_slot_clock     = 0;
    int32_t                    gf_slot_hits      = 0;
    int32_t                    gf_slot_misses    = 0;

    // number of graph-result slots (>=1). 1 == the single-slot legacy path.
    size_t wp_graph_result_slots() const;
    // select the slot for (gtype, n_tokens); returns null when single-slot.
    wp_graph_slot * wp_pick_graph_slot(llm_graph_type gtype, uint32_t n_tokens);
    // invalidate every cached graph result (all slots + gf_res_prev).
    void wp_reset_graph_results();

    // WP_QWEN4EXP_LAYER_CUT (Stage 3): the one live logical execution slot
    // (plan item 4) -- see llama_context::process_ubatch_staged().
    llm_graph_stage_slot layer_cut_slot;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
