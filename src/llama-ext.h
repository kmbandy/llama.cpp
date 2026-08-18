#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// MAD-LAB logits-on-head. Stop the decode graph after output_norm and never build
// the LM head, so llama_get_logits() is not produced but llama_get_embeddings()
// returns the post-output_norm hidden state. Set on a dense-segment TAIL worker,
// which then ships n_embd floats per token instead of n_vocab and lets the head do
// the projection with llama_output_project(). Changes the graph shape, so it forces
// a graph reserve -- call it once, before the first decode.
LLAMA_API void llama_set_no_output_head(struct llama_context * ctx, bool value);

// MAD-LAB logits-on-head. Project `hidden` ([n_tokens][n_embd] F32, row-major,
// ALREADY passed through output_norm) through this context's LM head and write the
// result into the context's logits buffer, exactly where a local decode would have
// put it -- llama_get_logits()/llama_get_logits_ith() then behave as usual.
//
// For a dense-segment HEAD: its band graph stops mid-model and never builds the LM
// head, but the head stage GGUF still carries output.weight. Returns false (and
// logs) if the model has no LM head, if a LoRA adapter is attached, or if the
// logits buffer is too small for n_tokens.
LLAMA_API bool llama_output_project(struct llama_context * ctx,
                                    const float * hidden,
                                    int32_t n_tokens);

// As llama_output_project(), but writes the result into `out` ([n_tokens][n_vocab] F32
// row-major) and leaves this context's logits buffer and pending output state untouched.
// Used to project a DRAFT hidden state through the TARGET's head without disturbing the
// verification logits the speculative loop is about to read.
LLAMA_API bool llama_output_project_to(struct llama_context * ctx,
                                       const float * hidden,
                                       int32_t n_tokens,
                                       float * out);

// True if this model carries its own LM head (output.weight). False for a sidecar
// speculative draft, which is the signal that the services path below is required.
LLAMA_API bool llama_model_has_output_head(const struct llama_model * model);

// The INCLUSIVE layer band this process actually builds. For a cross-machine dense
// pipeline segment that is the band it owns; otherwise the whole model. Callers that
// need a specific layer's activations must check it lies inside this range -- a layer
// outside it is computed on a different segment and is simply not available here.
LLAMA_API void llama_model_pipeline_band(const struct llama_model * model,
                                         int32_t * first,
                                         int32_t * last);

// MAD-LAB speculative services for a SIDECAR draft (a separate DFlash/DSpark GGUF
// passed with -md). Such a sidecar ships neither token_embd nor output.weight: it is
// trained against the target's embedding space and used to reach into the target's
// tensors through ctx_other. That is impossible once the target is Meta-split
// (-sm tensor), because those tensors are pre-allocated in a buffer type the draft's
// scheduler does not own. So the two borrowed ops are performed on the context that
// owns the tensors and the results are handed across as plain host buffers.

// Gather token_embd rows on THIS context. `out` is [n_tokens][n_embd] F32 row-major.
// Call on the TARGET; feed the result to the draft as the embd half of a batch that
// also carries the token ids (the DSpark Markov head conditions on the ids).
LLAMA_API bool llama_token_embed_gather(struct llama_context * ctx,
                                        const llama_token * tokens,
                                        int32_t n_tokens,
                                        float * out);

// Replay the DSpark Markov/confidence head on THIS context, over base logits that were
// projected elsewhere. Call on the DRAFT, whose model owns markov_w1/w2 and conf_proj.
// `base` is [n_tokens][n_vocab] row-major, `hidden` is [n_tokens][n_embd] (the
// post-output_norm state the draft graph exported), `out_conf` is [n_tokens] and may be
// NULL. n_blocks is the number of speculative blocks in the batch (ubatch n_seqs_unq);
// n_tokens must be a multiple of it.
//
// The biased logits are written into this context's own logits buffer, exactly where a
// local decode would have put them, so llama_get_logits_ith() and the samplers behave
// as usual and no sampling code needs to know the head ran out-of-graph.
LLAMA_API bool llama_dspark_markov_head(struct llama_context * ctx,
                                        const float * base,
                                        const llama_token * tokens,
                                        const float * hidden,
                                        int32_t n_tokens,
                                        int32_t n_blocks,
                                        float * out_conf);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// MAD-LAB INTERIOR TAPS. Arm a layer whose input hidden state this process cannot
// compute -- it lies outside this process's pipeline band -- and will instead be handed
// over the dense-segment wire by the segment that owns it. This reserves the same host
// buffer llama_get_embeddings_layer_inp() reads, so every existing reader (notably the
// DFlash/DSpark draft's target-feature gather) works unchanged and cannot tell the
// difference. It does NOT arm the graph-output path: that asserts the layer's tensor
// exists, which is exactly what a banded graph cannot provide.
LLAMA_API void llama_set_embeddings_layer_inp_external(struct llama_context * ctx, uint32_t lid, bool value);

// True if `lid` was armed by llama_set_embeddings_layer_inp_external(). Lets a caller
// that is about to require a tap distinguish "outside my band and therefore impossible"
// from "outside my band but supplied by a peer".
LLAMA_API bool llama_get_embeddings_layer_inp_external(const struct llama_context * ctx, uint32_t lid);

// Install [n_tokens][n_embd] F32 row-major rows, in BATCH order, for an armed layer.
// Call once per forward, after the remote segments have returned and before anything
// reads the tap. Fails if the layer was not armed or the buffer is too small.
LLAMA_API bool llama_set_layer_inp_data(struct llama_context * ctx, uint32_t lid,
                                        const float * data, int32_t n_tokens);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
// returns the DFlash hyper-connection stream multiplier
LLAMA_API uint32_t        llama_model_dflash_hc_mult    (const struct llama_model * model);
LLAMA_API uint32_t        llama_model_dflash_block_size (const struct llama_model * model);

// Weight-pager draft-as-paging-oracle: after a draft model produces tokens,
// map them through DS4 hash-layer tid2eid and pin/prefetch those expert
// pages across the draft->target gap. n_tokens <= 0 clears the window.
// Returns pages submitted for cold prefetch. No-op if paging disabled.
// NOTE: under WP strip (accept=0), draft token != next input token; prefer
// llama_wp_on_sampled_token for ground-truth hash-layer paging.
LLAMA_API int llama_wp_on_draft_tokens(struct llama_context * ctx,
                                       const llama_token * tokens,
                                       int n_tokens);

// Ground-truth hash-layer oracle: after the target samples token `id`, the
// next forward will *consume* that id as input. tid2eid(id) is exact for
// hash layers 0..H — no draft model required. Call once per sample.
LLAMA_API int llama_wp_on_sampled_token(struct llama_context * ctx,
                                        llama_token id);

// Hash-layer expert prefetch hint. Resolves `tokens` through the DS4 tid2eid
// tables (layers 0..H, a pure token-id lookup with no prediction) and offers the
// resulting expert ids to the cross-machine expert workers, which warm their own
// pools in their idle windows.
//
// DISTINCT FROM llama_wp_on_draft_tokens ABOVE, which drives the IN-PROCESS
// WeightPager. That path is a no-op on the cross-machine layout: the spine runs
// without --weight-paging so model.wp_pager is null, and even with a pager the
// routed experts are TENSOR_SKIP so its catalog holds no expert pages. This one
// needs no pager at all.
//
// CALL IT AS EARLY AS THE TOKENS ARE KNOWN -- the value is entirely in how much
// compute separates the hint from the forward pass that consumes it. Advisory:
// never throws, never blocks, and a dropped hint costs at most a page-in the run
// was going to pay anyway. Returns hint frames sent; 0 unless WP_PREFETCH_HINT=1
// and the context owns (or borrows) an expert dispatcher.
//
// n_certain: how many LEADING tokens the target is already committed to
// processing. The rest are predictions and are sent as a separate frame so the
// worker can hold them on a shorter lease -- a guess that outranks a certainty
// converts a free fetch into a displaced fact. -1 means all of them, which is
// what a caller with no prediction wants.
LLAMA_API int llama_expert_prefetch_hint(struct llama_context * ctx,
                                         const llama_token * tokens,
                                         int n_tokens,
                                         int n_certain);

// Adaptive gate: when false, skip running the draft model this step (pool
// already warm for hash-layer experts). Default adaptive ON; WP_DRAFT_ADAPTIVE=0
// always returns true. No-op / true if paging disabled.
LLAMA_API bool llama_wp_draft_oracle_should_run(struct llama_context * ctx);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// MAD-LAB / WP_DSPARK_DEBUG: read-only census of one sequence's resident KV cells.
//
// Diagnostic only. The public memory API exposes seq_pos_min/seq_pos_max, which cannot
// tell "one clean cell per position" apart from "several cells stacked on one position".
// out_n_dup is the load-bearing number: resident cells minus distinct positions, i.e.
// how many duplicate cells this sequence is carrying. out_n_at_or_above counts cells at
// pos >= pos_thresh (pass the committed prefix to isolate leftover draft-block cells).
// Returns false if mem is not a unified KV cache; all out params are optional.
LLAMA_API bool llama_dspark_kv_census(llama_memory_t mem,
                                      llama_seq_id   seq_id,
                                      llama_pos      pos_thresh,
                                      int32_t      * out_n_cells,
                                      int32_t      * out_n_at_or_above,
                                      int32_t      * out_n_dup,
                                      llama_pos    * out_pos_min,
                                      llama_pos    * out_pos_max);
