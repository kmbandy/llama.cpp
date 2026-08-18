#include "pipeline-stage.h"

#include <cstring>
#include <vector>

bool llama_pipeline_stage_decode_hidden(
        llama_context * ctx,
        int32_t n_embd,
        const float * activations,
        int32_t n_tokens,
        const int32_t * positions,
        uint32_t n_pos_per_token) {
    if (ctx == nullptr || activations == nullptr || positions == nullptr ||
        n_embd <= 0 || n_tokens <= 0 || n_pos_per_token == 0) {
        return false;
    }

    // An embd (hidden-state) batch is read by llama_batch_allocr with a section
    // stride: udata->pos[j*n_tokens + i] = batch.pos[j*batch.n_tokens + i] for
    // j < n_pos_per_embd (llama-batch.cpp, the `src_off = batch.token ? 0 : ...`
    // branch). Token batches broadcast one position across the M-RoPE sections,
    // but embd batches do not -- so batch.pos MUST hold n_tokens*n_pos_per_embd
    // entries. llama_batch_init only allocates n_tokens of them, so for an
    // M-RoPE model (qwen35 is IMROPE => n_pos_per_embd == 4) the sections above
    // the first would be read out of bounds. Build the position plane here
    // instead, broadcasting like the text-token path does: sections 0..2 carry
    // the position and section 3 is zero.
    const llama_model * model = llama_get_model(ctx);
    const llama_rope_type rope_type = model ? llama_model_rope_type(model) : LLAMA_ROPE_TYPE_NONE;
    const uint32_t n_pos_per_embd =
        (rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    std::vector<llama_pos> pos((size_t) n_tokens * n_pos_per_embd, 0);
    for (uint32_t j = 0; j < n_pos_per_embd; ++j) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            llama_pos p = 0;
            if (j < n_pos_per_token) {
                // the wire carries this section explicitly
                p = positions[(size_t) i * n_pos_per_token + j];
            } else if (j < 3) {
                // text semantics: the first three sections share the position
                p = positions[(size_t) i * n_pos_per_token];
            } // section 3 stays 0
            pos[(size_t) j * n_tokens + i] = p;
        }
    }

    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens; // llama_batch_init only allocates capacity
    std::memcpy(batch.embd, activations, (size_t) n_tokens * n_embd * sizeof(float));
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    // llama_batch_init's pos array is one section wide; swap in the full plane
    // and restore it before llama_batch_free so the right pointer is released.
    llama_pos * pos_owned = batch.pos;
    batch.pos = pos.data();

    const int rc = llama_decode(ctx, batch);

    batch.pos = pos_owned;
    llama_batch_free(batch);
    return rc == 0;
}

bool llama_pipeline_stage_read_hidden(
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        std::vector<float> & out) {
    if (ctx == nullptr || n_tokens <= 0 || n_embd <= 0) {
        return false;
    }
    const float * embd = llama_get_embeddings(ctx);
    if (embd == nullptr) {
        return false;
    }
    out.assign(embd, embd + (size_t) n_tokens * n_embd);
    return true;
}
