#pragma once

// EmbeddingModel — thin wrapper over a small CPU embedding model
// (e.g. bge-small-en-v1.5, nomic-embed-text). Used by the tier system
// to compute L2-normalized fingerprints for semantic chunk retrieval.
//
// Lifecycle is lazy: nothing happens until embed() is called for the
// first time. The model file path is stored at construction; the
// actual llama_model + llama_context come up on first use. This keeps
// the wrapper's startup cheap when --kv-tier-semantic-index is set
// but the user's session doesn't actually exercise semantic restore.
//
// Threadsafe: a single mutex serializes all embed() calls. Embedding
// inference is fast on CPU (a few ms for ~512-token inputs on bge-small)
// so contention isn't a concern at the tier-restore call rate.
//
// Errors: load failures and decode failures log a warning and cause
// embed() to return an empty vector. Caller should treat empty as
// "skip this fingerprint" rather than retrying.

#include "llama.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mt {

// Asymmetric retrieval role. Modern embedders (LFM2.5-Embedding, nomic,
// arctic, bge) are trained with distinct query/passage instruction
// prefixes and silently lose recall if you embed both sides the same way.
// The prefix strings themselves are model-specific and live in mt-embed.cpp.
// Document is the default because all but one call site embed passages.
enum class EmbedRole {
    Document,
    Query,
};

class EmbeddingModel {
public:
    // `parallel` = size of the llama_context pool (values < 1 clamp to 1).
    // The llama_model is loaded ONCE and shared by every context, so an extra
    // context costs only its own KV/compute buffers (a few MB), not another
    // copy of the weights.
    explicit EmbeddingModel(std::string path, int parallel = 1, int n_threads = 0,
                            std::string query_prefix = {}, std::string doc_prefix = {});
    ~EmbeddingModel();

    EmbeddingModel(const EmbeddingModel &)             = delete;
    EmbeddingModel & operator=(const EmbeddingModel &) = delete;

    // Returns L2-normalized embedding of `text`. Empty vector on failure
    // (model failed to load, tokenization produced no tokens, decode
    // returned no embedding). Lazy-initializes the model on first call.
    std::vector<float> embed(const std::string & text, EmbedRole role = EmbedRole::Document);

    // Embed multiple independent texts with multi-sequence batches. Results
    // preserve input order; an empty vector marks an item that failed.
    std::vector<std::vector<float>> embed_batch(const std::vector<std::string> & texts,
                                                EmbedRole role = EmbedRole::Document);

    // Embedding dimensionality of the loaded model. Returns 0 if the
    // model hasn't been initialized yet (no embed() call has succeeded).
    int n_embd() const { return n_embd_; }

    // Diagnostic: has the model been successfully loaded?
    bool ready() const;

private:
    bool ensure_loaded_locked();
    void shutdown_locked();

    // Take a context out of the pool, blocking until one frees up. Returns
    // nullptr if the model failed to load. Every acquire() MUST be matched by
    // a release() or the pool leaks a slot and eventually deadlocks.
    llama_context * acquire_ctx();

    // MAD-348: the model's own retrieval prefix for this role (may be empty).
    const std::string & embed_prefix_for(EmbedRole role) const;
    void            release_ctx(llama_context * ctx);

    mutable std::mutex      mu_;
    std::condition_variable cv_;           // signalled when a context returns to free_
    std::string          path_;
    int                  parallel_ = 1;
    int                  n_threads_ = 0;   // MAD-348: <=0 => ggml default (4)
    std::string          query_prefix_;    // MAD-348: model-specific; empty => none
    std::string          doc_prefix_;
    llama_model    *     model_ = nullptr;
    std::vector<llama_context *> ctxs_;    // owned; parallel_ entries once loaded
    std::vector<llama_context *> free_;    // the currently-available subset of ctxs_
    int                  n_embd_ = 0;
    int                  n_ctx_seq_ = 0;   // per-sequence KV capacity (llama_n_ctx_seq); hard cap per input
    bool                 init_attempted_ = false;
    bool                 init_succeeded_ = false;
};

}  // namespace mt
