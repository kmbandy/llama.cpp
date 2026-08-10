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
