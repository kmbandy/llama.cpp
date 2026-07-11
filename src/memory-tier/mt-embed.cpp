#include "mt-embed.h"

#include "llama-impl.h"  // LLAMA_LOG_*
#include "llama-vocab.h"

#include <cmath>
#include <cstring>
#include <utility>

namespace mt {

static constexpr int EMBED_BATCH_TOKENS = 512;
static constexpr int EMBED_BATCH_SEQS   = 32;

// Model-specific asymmetric retrieval prefixes. LFM2.5-Embedding-350M was
// trained with these exact strings (config_sentence_transformers.json);
// omitting them silently degrades retrieval quality. Revisit on model swap.
static constexpr const char * QUERY_PREFIX    = "query: ";
static constexpr const char * DOCUMENT_PREFIX = "document: ";

static const char * embed_prefix(EmbedRole role) {
    return role == EmbedRole::Query ? QUERY_PREFIX : DOCUMENT_PREFIX;
}

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
    cparams.n_ctx        = 512;                          // query window is capped at 512; KV blocks are 16 tokens
    cparams.n_batch      = 512;
    cparams.n_ubatch     = 512;                          // non-causal encode needs n_ubatch >= n_tokens; 512 covers the largest input
    cparams.n_seq_max    = EMBED_BATCH_SEQS;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_CLS;       // LFM2.5-Embedding-350M uses CLS pooling

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        LLAMA_LOG_WARN("mt::EmbeddingModel: failed to create context for %s\n", path_.c_str());
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    // Use the projected output dim, not the hidden size. Models with a
    // dense head (e.g. LFM2.5-Embedding's dense_2 -> 1024) emit vectors of
    // n_embd_out(); n_embd_out() falls back to n_embd for plain encoders
    // (bge/granite), so this is correct for every embedding model.
    n_embd_ = llama_model_n_embd_out(model_);
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

std::vector<float> EmbeddingModel::embed(const std::string & text, EmbedRole role) {
    auto result = embed_batch({text}, role);
    return result.empty() ? std::vector<float>{} : std::move(result[0]);
}

std::vector<std::vector<float>> EmbeddingModel::embed_batch(const std::vector<std::string> & texts,
                                                           EmbedRole role) {
    std::lock_guard<std::mutex> lk(mu_);

    std::vector<std::vector<float>> result(texts.size());

    if (!ensure_loaded_locked() || texts.empty()) return result;

    const llama_vocab * vocab = llama_model_get_vocab(model_);
    if (!vocab) {
        LLAMA_LOG_WARN("mt::EmbeddingModel::embed_batch: no vocab on model\n");
        return result;
    }

    std::vector<std::vector<llama_token>> tokenized(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        const auto & text = texts[i];
        if (text.empty()) continue;
        // Prepend the model's asymmetric retrieval prefix before tokenizing.
        const std::string prefixed = embed_prefix(role) + text;
        int cap = (int) prefixed.size() + 8;
        tokenized[i].resize(cap);
        int n = llama_tokenize(vocab, prefixed.c_str(), (int) prefixed.size(),
                               tokenized[i].data(), cap, true, false);
        if (n < 0) {
            cap = -n;
            tokenized[i].resize(cap);
            n = llama_tokenize(vocab, prefixed.c_str(), (int) prefixed.size(),
                               tokenized[i].data(), cap, true, false);
        }
        if (n <= 0) {
            tokenized[i].clear();
            continue;
        }
        // Clamp to the batch budget, NOT llama_n_ctx(): embedding models
        // silently inflate n_ctx to their trained max (granite -> 8192,
        // LFM2.5 -> 128000), so llama_n_ctx() is not a safe per-sequence
        // cap. A single sequence over n_ubatch/n_batch aborts either the
        // encode assert (n_ubatch >= n_tokens) or the decode assert
        // (n_tokens_all <= n_batch). EMBED_BATCH_TOKENS == both (512).
        tokenized[i].resize(std::min(n, EMBED_BATCH_TOKENS));
    }

    size_t next = 0;
    while (next < texts.size()) {
        std::vector<size_t> indices;
        int n_batch_tokens = 0;
        while (next < texts.size() && (int) indices.size() < EMBED_BATCH_SEQS) {
            const int n = (int) tokenized[next].size();
            if (n == 0) {
                ++next;
                continue;
            }
            if (!indices.empty() && n_batch_tokens + n > EMBED_BATCH_TOKENS) break;
            indices.push_back(next++);
            n_batch_tokens += n;
        }
        if (indices.empty()) continue;

        llama_batch batch = llama_batch_init(n_batch_tokens, 0, (int) indices.size());
        int ib = 0;
        for (size_t seq = 0; seq < indices.size(); ++seq) {
            const auto & tokens = tokenized[indices[seq]];
            for (size_t pos = 0; pos < tokens.size(); ++pos, ++ib) {
                batch.token   [ib] = tokens[pos];
                batch.pos     [ib] = (llama_pos) pos;
                batch.n_seq_id[ib] = 1;
                batch.seq_id  [ib][0] = (llama_seq_id) seq;
                batch.logits  [ib] = pos + 1 == tokens.size();
            }
        }
        batch.n_tokens = ib;

        llama_memory_clear(llama_get_memory(ctx_), true);
        if (llama_decode(ctx_, batch) != 0) {
            LLAMA_LOG_WARN("mt::EmbeddingModel::embed_batch: llama_decode failed for %zu sequences\n", indices.size());
            llama_batch_free(batch);
            continue;
        }

        for (size_t seq = 0; seq < indices.size(); ++seq) {
            const float * raw = llama_get_embeddings_seq(ctx_, (llama_seq_id) seq);
            if (!raw && indices.size() == 1) raw = llama_get_embeddings(ctx_);
            if (!raw) continue;
            auto & v = result[indices[seq]];
            v.assign(raw, raw + n_embd_);
            double norm_sq = 0.0;
            for (float x : v) norm_sq += (double) x * x;
            if (norm_sq > 0.0) {
                const float inv = (float) (1.0 / std::sqrt(norm_sq));
                for (float & x : v) x *= inv;
            }
        }
        llama_batch_free(batch);
    }

    return result;
}

}  // namespace mt
