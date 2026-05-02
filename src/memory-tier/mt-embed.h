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

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mt {

class EmbeddingModel {
public:
    explicit EmbeddingModel(std::string path);
    ~EmbeddingModel();

    EmbeddingModel(const EmbeddingModel &)             = delete;
    EmbeddingModel & operator=(const EmbeddingModel &) = delete;

    // Returns L2-normalized embedding of `text`. Empty vector on failure
    // (model failed to load, tokenization produced no tokens, decode
    // returned no embedding). Lazy-initializes the model on first call.
    std::vector<float> embed(const std::string & text);

    // Embedding dimensionality of the loaded model. Returns 0 if the
    // model hasn't been initialized yet (no embed() call has succeeded).
    int n_embd() const { return n_embd_; }

    // Diagnostic: has the model been successfully loaded?
    bool ready() const;

private:
    bool ensure_loaded_locked();
    void shutdown_locked();

    mutable std::mutex   mu_;
    std::string          path_;
    llama_model    *     model_ = nullptr;
    llama_context  *     ctx_   = nullptr;
    int                  n_embd_ = 0;
    bool                 init_attempted_ = false;
    bool                 init_succeeded_ = false;
};

}  // namespace mt
