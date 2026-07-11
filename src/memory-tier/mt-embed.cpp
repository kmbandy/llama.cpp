#include "mt-embed.h"

#include "llama-impl.h"  // LLAMA_LOG_*
#include "llama-vocab.h"

#include <cmath>
#include <cstring>
#include <utility>

namespace mt {

static constexpr int EMBED_BATCH_TOKENS = 512;   // max tokens per llama_decode (n_batch/n_ubatch) AND per single input
static constexpr int EMBED_BATCH_SEQS   = 16;    // max sequences per decode. With n_ctx = TOKENS*SEQS below, llama.cpp
                                                 // derives n_ctx_seq = GGML_PAD(n_ctx/n_seq_max, 256) = 512 per stream,
                                                 // so each stream can hold a full EMBED_BATCH_TOKENS-length input.

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
    // KV sizing (learned the hard way, 2026-07-11): llama.cpp derives the
    // PER-SEQUENCE capacity as n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256),
    // then re-expands n_ctx = n_ctx_seq * n_seq_max. A single input longer
    // than n_ctx_seq fails llama_decode with "find_slot: n_tokens > size"
    // (non-fatal: returns empty embedding, silently dropping the fingerprint).
    // The old n_ctx=512 + n_seq_max=32 gave n_ctx_seq=256, so 257..512-token
    // QUERY embeds (q_max is capped at 512 upstream) all failed. Setting
    // n_ctx = EMBED_BATCH_TOKENS * EMBED_BATCH_SEQS makes n_ctx_seq exactly
    // EMBED_BATCH_TOKENS (512), so every stream holds a full-length input.
    // Same total cell count as before (8192) — just 16 streams x 512 instead
    // of 32 x 256 — so no extra RAM. Inputs are still hard-clamped to the
    // ACTUAL n_ctx_seq below (n_ctx_seq_) so this can never silently desync.
    cparams.n_ctx        = EMBED_BATCH_TOKENS * EMBED_BATCH_SEQS;  // 8192 -> n_ctx_seq = 512
    cparams.n_batch      = EMBED_BATCH_TOKENS;
    cparams.n_ubatch     = EMBED_BATCH_TOKENS;           // non-causal encode needs n_ubatch >= tokens-per-decode (<= EMBED_BATCH_TOKENS)
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
    // Actual per-sequence KV capacity after llama.cpp's GGML_PAD/re-expand.
    // This is the hard cap on any single input; inputs are clamped to it.
    n_ctx_seq_ = (int) llama_n_ctx_seq(ctx_);
    init_succeeded_ = true;
    LLAMA_LOG_INFO("mt::EmbeddingModel: loaded %s (n_embd=%d, n_ctx_seq=%d)\n", path_.c_str(), n_embd_, n_ctx_seq_);
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
        // Clamp each input to the context's ACTUAL per-sequence KV capacity
        // (n_ctx_seq_ = llama_n_ctx_seq()). Do NOT use llama_n_ctx(): embedding
        // models inflate it to their trained max (granite -> 8192, LFM2.5 ->
        // 128000), and it is the TOTAL across streams, not the per-seq cap.
        // A single input longer than n_ctx_seq_ fails llama_decode with
        // "find_slot: n_tokens > size" -> empty embedding, dropped fingerprint.
        // (Fallback to EMBED_BATCH_TOKENS if n_ctx_seq_ is somehow unset.)
        const int seq_cap = n_ctx_seq_ > 0 ? n_ctx_seq_ : EMBED_BATCH_TOKENS;
        tokenized[i].resize(std::min(n, seq_cap));
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
