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

// MAD-348: retrieval prefixes are a property of the EMBEDDING MODEL, not of this file.
//
// They used to be hardcoded to LFM2.5-Embedding's ("query: " / "document: ") under a
// comment reading "Revisit on model swap". Nobody did -- and the identical failure had
// already happened to kv_semantic_threshold (0.65 was BGE's number; when LFM2.5 arrived,
// whose true-positive cosines top out at 0.63, the gate silently rejected 100% of correct
// matches and the index prefetched NOTHING for months).
//
// Measured 2026-07-14: LFM2.5 wants "query: "/"document: "; BGE wants an instruction on
// the query only; Granite wants NONE. A wrong prefix is silent recall loss, so it now
// comes from config and the preset must state it beside the model path.

EmbeddingModel::EmbeddingModel(std::string path, int parallel, int n_threads,
                               std::string query_prefix, std::string doc_prefix)
    : path_(std::move(path)), parallel_(parallel > 1 ? parallel : 1),
      n_threads_(n_threads > 0 ? n_threads : 0),
      query_prefix_(std::move(query_prefix)), doc_prefix_(std::move(doc_prefix)) {}

EmbeddingModel::~EmbeddingModel() {
    std::lock_guard<std::mutex> lk(mu_);
    shutdown_locked();
}

llama_context * EmbeddingModel::acquire_ctx() {
    std::unique_lock<std::mutex> lk(mu_);
    if (!ensure_loaded_locked()) return nullptr;
    cv_.wait(lk, [this] { return !free_.empty(); });
    llama_context * ctx = free_.back();
    free_.pop_back();
    return ctx;
}

void EmbeddingModel::release_ctx(llama_context * ctx) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push_back(ctx);
    }
    cv_.notify_one();
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

    // CPU-only embedding model. Offloading is not worth VRAM contention with
    // the main model (on the 8 GB cards there is none to contend for).
    //
    // MAD-348 — n_gpu_layers = 0 IS NOT ENOUGH, and believing it was cost us
    // twice. With mparams.devices left unset, llama_init_from_model still builds
    // a GPU backend for this model and schedules ops on it. Two failures traced
    // back to exactly that:
    //
    //   1. On mad-lab-2026 the 480's scout died with `CUDA error: out of memory`
    //      in ggml_cuda_op_mul during warmup — its stray CUDA context landed on
    //      CUDA0 (the 1070), which was already full with its own scout. The
    //      workaround was to drop --kv-tier-semantic-index from the 480 entirely,
    //      giving up eviction quality.
    //   2. Running the fingerprint sweep off the inference thread aborted the
    //      server with `ROCm error: operation would make the legacy stream depend
    //      on a capturing blocking stream` — the embedder's GPU ops raced the main
    //      model's HIP graph capture.
    //
    // Pinning devices to a CPU-only list makes the flag mean what it says: this
    // model touches no GPU, so it can never squat VRAM on another card and can
    // never collide with the main model's stream/graph capture. That in turn is
    // what makes the background (off-inference-thread) sweep safe.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;        // CPU only
    mparams.use_mmap     = true;     // small model, mmap is fine

    static std::vector<ggml_backend_dev_t> cpu_devices = [] {
        std::vector<ggml_backend_dev_t> devs;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                devs.push_back(dev);
            }
        }
        devs.push_back(nullptr);  // llama expects a NULL-terminated list
        return devs;
    }();
    if (cpu_devices.size() > 1) {
        mparams.devices = cpu_devices.data();
    } else {
        LLAMA_LOG_WARN("mt::EmbeddingModel: no CPU backend device found; leaving device "
                       "selection to llama (the embedder may open a GPU context)\n");
    }

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
    // MAD-348: pooling comes from the MODEL, not from us. This was hardcoded to CLS for
    // LFM2.5-Embedding. BGE also happens to use CLS, but a MEAN-pooled model (the previous
    // default was MEAN!) would be silently mis-pooled -- degraded embeddings, no crash, no
    // log line. UNSPECIFIED makes llama read pooling_type from the GGUF metadata, which is
    // the only source that is right for every model.
    cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    // MAD-348: THREAD COUNT. This was never set, so it silently inherited
    // llama_context_default_params()'s GGML_DEFAULT_N_THREADS = 4 (whose own definition
    // in llama-context.cpp is commented "TODO: better default"). On a 12-core host that
    // ran a 350M model on FOUR threads: measured ~360 tok/s of embedding throughput, and
    // an 18k-token prefill fingerprint sweep took 50-68 s, of which a phase probe put
    // 100% inside embed_text_batch. That is not an inherently expensive embedder; it is
    // a 350M model running on 1/6th of the machine.
    //
    // n_threads_ is plumbed from the MAIN MODEL's --threads (common_cpu_get_num_math()
    // by default, operator-overridable per box, and separately settable with
    // --kv-tier-semantic-threads). It is deliberately NOT a hardcoded "physical cores
    // minus one": that formula gives 11 threads on a 12c/24t host but only THREE on a
    // 4c/8t one -- FEWER than the old default of 4, i.e. a REGRESSION on the small box,
    // and on a host whose four cores are already driving two GPUs and the server loop
    // there is no headroom to take anyway.
    //
    // Falling back to 0 keeps ggml's own default rather than inventing a number.
    if (n_threads_ > 0) {
        cparams.n_threads       = n_threads_;
        cparams.n_threads_batch = n_threads_;   // the embed is a BATCH decode
        LLAMA_LOG_INFO("mt::EmbeddingModel: %d threads\n", n_threads_);
    }

    // One llama_model, `parallel_` contexts. Each context carries its own
    // n_ctx = EMBED_BATCH_TOKENS * EMBED_BATCH_SEQS, so n_ctx_seq stays exactly
    // EMBED_BATCH_TOKENS per stream no matter how big the pool gets -- the pool
    // ADDS total capacity (parallel_ * n_ctx) rather than dividing a fixed
    // budget. Sizing the pool by widening a single context's n_seq_max instead
    // would shrink n_ctx_seq (= GGML_PAD(n_ctx/n_seq_max, 256)) and silently
    // start dropping fingerprints for long inputs.
    ctxs_.reserve(parallel_);
    free_.reserve(parallel_);
    for (int i = 0; i < parallel_; ++i) {
        llama_context * ctx = llama_init_from_model(model_, cparams);
        if (!ctx) {
            if (i == 0) {
                LLAMA_LOG_WARN("mt::EmbeddingModel: failed to create context for %s\n", path_.c_str());
                llama_model_free(model_);
                model_ = nullptr;
                return false;
            }
            // Partial pool is still usable -- run with what we got.
            LLAMA_LOG_WARN("mt::EmbeddingModel: only %d/%d contexts created for %s\n",
                           i, parallel_, path_.c_str());
            parallel_ = i;
            break;
        }
        ctxs_.push_back(ctx);
        free_.push_back(ctx);
    }

    // Use the projected output dim, not the hidden size. Models with a
    // dense head (e.g. LFM2.5-Embedding's dense_2 -> 1024) emit vectors of
    // n_embd_out(); n_embd_out() falls back to n_embd for plain encoders
    // (bge/granite), so this is correct for every embedding model.
    n_embd_ = llama_model_n_embd_out(model_);
    // Actual per-sequence KV capacity after llama.cpp's GGML_PAD/re-expand.
    // This is the hard cap on any single input; inputs are clamped to it.
    // MAD-348: log the COUPLED unit -- model + pooling + prefixes belong together, and a
    // mismatch between them is silent (degraded recall, no crash, no error). Twice now a
    // model swap stranded a constant tuned for the previous model. Print them side by side
    // so the next swap cannot hide. kv_semantic_threshold is logged by the server; the
    // measured values are: LFM2.5 0.35, BGE 0.63, Granite 0.84 -- they are NOT portable.
    LLAMA_LOG_WARN("mt::EmbeddingModel: %s | pooling=%d (from GGUF) | query_prefix=%s | doc_prefix=%s | threads=%d\n",
                   path_.c_str(), (int) llama_pooling_type(ctxs_[0]),
                   query_prefix_.empty() ? "(none)" : query_prefix_.c_str(),
                   doc_prefix_.empty()   ? "(none)" : doc_prefix_.c_str(),
                   n_threads_ > 0 ? n_threads_ : 4);

    n_ctx_seq_ = (int) llama_n_ctx_seq(ctxs_[0]);
    if (n_ctx_seq_ < EMBED_BATCH_TOKENS) {
        LLAMA_LOG_WARN("mt::EmbeddingModel: n_ctx_seq=%d < EMBED_BATCH_TOKENS=%d -- inputs will be "
                       "truncated; check the n_ctx/n_seq_max sizing\n", n_ctx_seq_, EMBED_BATCH_TOKENS);
    }
    init_succeeded_ = true;
    LLAMA_LOG_INFO("mt::EmbeddingModel: loaded %s (n_embd=%d, n_ctx_seq=%d, parallel=%d)\n",
                   path_.c_str(), n_embd_, n_ctx_seq_, parallel_);
    return true;
}

void EmbeddingModel::shutdown_locked() {
    for (llama_context * ctx : ctxs_) {
        llama_free(ctx);
    }
    ctxs_.clear();
    free_.clear();
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

const std::string & EmbeddingModel::embed_prefix_for(EmbedRole role) const {
    return role == EmbedRole::Query ? query_prefix_ : doc_prefix_;
}

std::vector<std::vector<float>> EmbeddingModel::embed_batch(const std::vector<std::string> & texts,
                                                           EmbedRole role) {
    std::vector<std::vector<float>> result(texts.size());

    if (texts.empty()) return result;


    // Hold a pooled context for the duration, NOT the class mutex. The sweep
    // runs on the server's single inference thread, so serializing every slot
    // behind one embed lock stalls the batch that carries every other slot's
    // decode token. With parallel_ > 1 the slots embed concurrently instead.
    llama_context * ctx = acquire_ctx();
    if (!ctx) return result;
    struct CtxReturn {
        EmbeddingModel * self;
        llama_context  * ctx;
        ~CtxReturn() { self->release_ctx(ctx); }
    } ctx_return{this, ctx};

    // model_/n_embd_/n_ctx_seq_ are immutable once init_succeeded_, and
    // acquire_ctx() only returns after a successful load, so these are safe
    // to read without the lock.
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
        const std::string prefixed = embed_prefix_for(role) + text;
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

        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            LLAMA_LOG_WARN("mt::EmbeddingModel::embed_batch: llama_decode failed for %zu sequences\n", indices.size());
            llama_batch_free(batch);
            continue;
        }

        for (size_t seq = 0; seq < indices.size(); ++seq) {
            const float * raw = llama_get_embeddings_seq(ctx, (llama_seq_id) seq);
            if (!raw && indices.size() == 1) raw = llama_get_embeddings(ctx);
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
