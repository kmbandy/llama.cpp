#include "mt-embed.h"

#include "llama-impl.h"  // LLAMA_LOG_*
#include "llama-vocab.h"

#include <cmath>
#include <cstring>
#include <utility>

namespace mt {

EmbeddingModel::EmbeddingModel(std::string path) : path_(std::move(path)) {}

EmbeddingModel::~EmbeddingModel() {
    std::lock_guard<std::mutex> lk(mu_);
    shutdown_locked();
}

bool EmbeddingModel::ready() const {
    std::lock_guard<std::mutex> lk(mu_);
    return init_succeeded_;
}

bool EmbeddingModel::ensure_loaded_locked() {
    if (init_attempted_) return init_succeeded_;
    init_attempted_ = true;

    if (path_.empty()) {
        LLAMA_LOG_WARN("mt::EmbeddingModel: no path configured\n");
        return false;
    }

    // Default model params: CPU-only embedding model. We deliberately
    // don't offload to GPU — the model is tiny (~30 MiB for bge-small)
    // and contention with the main model's VRAM isn't worth saving a
    // few ms of inference latency.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;        // CPU only
    mparams.use_mmap     = true;     // small model, mmap is fine

    model_ = llama_model_load_from_file(path_.c_str(), mparams);
    if (!model_) {
        LLAMA_LOG_WARN("mt::EmbeddingModel: failed to load model from %s\n", path_.c_str());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 512;                          // bge-small max
    cparams.n_batch      = 512;
    cparams.n_ubatch     = 512;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;      // matches bge training

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        LLAMA_LOG_WARN("mt::EmbeddingModel: failed to create context for %s\n", path_.c_str());
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    n_embd_ = llama_model_n_embd(model_);
    init_succeeded_ = true;
    LLAMA_LOG_INFO("mt::EmbeddingModel: loaded %s (n_embd=%d)\n", path_.c_str(), n_embd_);
    return true;
}

void EmbeddingModel::shutdown_locked() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    init_succeeded_ = false;
}

std::vector<float> EmbeddingModel::embed(const std::string & text) {
    std::lock_guard<std::mutex> lk(mu_);

    if (!ensure_loaded_locked()) return {};
    if (text.empty()) return {};

    const llama_vocab * vocab = llama_model_get_vocab(model_);
    if (!vocab) {
        LLAMA_LOG_WARN("mt::EmbeddingModel::embed: no vocab on model\n");
        return {};
    }

    // Tokenize. add_special=true to include BOS for bge-style models.
    // First call with negative count returns the required token count.
    int n_tokens_max = (int) text.size() + 8;
    std::vector<llama_token> tokens(n_tokens_max);
    int n_tokens = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                  tokens.data(), n_tokens_max,
                                  /*add_special=*/ true,
                                  /*parse_special=*/ false);
    if (n_tokens < 0) {
        // Buffer too small. The negative return is -required_size.
        n_tokens_max = -n_tokens;
        tokens.resize(n_tokens_max);
        n_tokens = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                  tokens.data(), n_tokens_max, true, false);
    }
    if (n_tokens <= 0) {
        LLAMA_LOG_DEBUG("mt::EmbeddingModel::embed: tokenization produced %d tokens\n", n_tokens);
        return {};
    }

    // Cap to context size; bge-small's 512 limit is plenty for chunk
    // fingerprinting (we'd typically embed ~50-200 tokens).
    const int n_ctx = (int) llama_n_ctx(ctx_);
    if (n_tokens > n_ctx) n_tokens = n_ctx;
    tokens.resize(n_tokens);

    // Build a single-sequence batch. The pooling layer will reduce to
    // one vector per seq.
    llama_batch batch = llama_batch_init(n_tokens, /*embd=*/ 0, /*n_seq_max=*/ 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token   [i] = tokens[i];
        batch.pos     [i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i] = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    // Reset cache for this seq before decode (the bge model is
    // single-shot — every call starts fresh).
    llama_memory_clear(llama_get_memory(ctx_), true);

    if (llama_decode(ctx_, batch) != 0) {
        LLAMA_LOG_WARN("mt::EmbeddingModel::embed: llama_decode failed\n");
        llama_batch_free(batch);
        return {};
    }

    const float * raw = llama_get_embeddings_seq(ctx_, 0);
    if (!raw) {
        // Some pooling configs don't produce per-seq output; fall back to
        // last-token embedding via llama_get_embeddings.
        raw = llama_get_embeddings(ctx_);
    }
    if (!raw) {
        LLAMA_LOG_WARN("mt::EmbeddingModel::embed: no embedding output\n");
        llama_batch_free(batch);
        return {};
    }

    std::vector<float> v(raw, raw + n_embd_);
    llama_batch_free(batch);

    // L2 normalize. SemanticIndex expects normalized vectors so cosine
    // similarity reduces to a dot product.
    double norm_sq = 0.0;
    for (float x : v) norm_sq += (double) x * x;
    if (norm_sq > 0.0) {
        const float inv = (float) (1.0 / std::sqrt(norm_sq));
        for (float & x : v) x *= inv;
    }

    return v;
}

}  // namespace mt
