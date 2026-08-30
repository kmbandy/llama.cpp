#include "speculative.h"
#include <cstdio>

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)
#include "../src/llama-graph.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <cstdlib>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

// MAD-LAB / WP_DSPARK_DEBUG: env-gated DSpark draft instrumentation.
//
// Read once. Off by default and costs one predictable branch per draft call when off.
// Everything behind this gate is read-only: no cache mutation, no sampler state change,
// no effect on which tokens are drafted. A run with the gate on must produce byte-identical
// draft/accept counts to a run with it off -- if it does not, the instrumentation itself is
// perturbing the path and nothing measured under it can be trusted.
static bool wp_dspark_debug() {
    static const bool s_on = [](){
        const char * e = std::getenv("WP_DSPARK_DEBUG");
        return e && e[0] == '1';
    }();
    return s_on;
}

// MAD-LAB / verify-width padding is a separate, opt-in knob from
// WP_DS4_CONST_SHAPE (2026-08-24 split, mirrors tools/server/server-context.cpp
// server_spec_const_width()). WP_DS4_CONST_SHAPE=1 alone no longer defaults
// this to 7 -- padding the drafter's block to a constant width is real added
// compute on the hot verify path, not free dispatch overhead (measured: DS4
// decode cost is ~linear in tokens verified). WP_SPEC_CONST_WIDTH must be set
// explicitly to enable it; deprecated in favor of running const-shape unpadded.
static int32_t wp_ds4_const_shape_width() {
    static const int32_t width = [] {
        const char * value = std::getenv("WP_SPEC_CONST_WIDTH");
        return value != nullptr ? std::atoi(value) : 0;
    }();
    return width;
}

// MAD-LAB / WP_DSPARK_ANCHOR_ABLATE: anchor-sensitivity probe.
//
// Set to a token id to REPLACE the anchor (dp.id_last) in the DSpark noise block with that
// fixed id, every step, while leaving the injected context KV and everything else alone.
//
// This is the controlled experiment the fantasy-shift observation demands. The anchor is the
// ONLY channel through which the target's correction reaches the drafter: the correction sits
// at position n_past and its target features are not injected until the NEXT process() call,
// so at draft time it exists solely as the anchor slot's token embedding. Therefore:
//   corrupt the anchor and acceptance barely moves -> the anchor is provably being ignored,
//     and the defect is somewhere on the anchor's path into the graph
//   corrupt the anchor and acceptance collapses    -> the anchor is working, the drafter is
//     simply self-consistent, and the fantasy-shift has an innocent explanation
// 0 / unset = off.
static llama_token wp_dspark_anchor_ablate() {
    static const llama_token s_tok = [](){
        const char * e = std::getenv("WP_DSPARK_ANCHOR_ABLATE");
        return e ? (llama_token) std::atoi(e) : 0;
    }();
    return s_tok;
}

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

static bool common_speculative_capture_enabled() {
    static const bool enabled = [] {
        const char * path = std::getenv("WP_DRAFT_CAPTURE");
        return path != nullptr && path[0] != '\0';
    }();
    return enabled;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.
    std::vector<double> n_draft_conf_sum;
    std::vector<size_t> n_draft_conf_count;
    std::vector<size_t> n_draft_len_hist;

    // MAD-LAB / multi-sequence-safe: count of ragged (non-block-aligned) ubatches the
    // DSpark Markov head skipped, forcing that call's confidence to an explicit 0 instead
    // of serving it stale. Only common_speculative_impl_draft_dflash ever increments this
    // (via llama_dspark_markov_ragged_skipped_fetch_reset(), src/models/dflash.cpp); every
    // other implementation leaves it at 0. Should stay 0 for a correctly configured
    // DSpark draft context -- see the hard n_ubatch check in that constructor. Printed
    // unconditionally in common_speculative_print_stats() so a nonzero value is visible
    // in the normal stats line, not just in a log grep.
    size_t n_markov_ragged_skipped = 0;

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }

    virtual bool get_draft_capture(
            llama_seq_id /*seq_id*/, const float *& /*embeddings*/, int32_t & /*n_embd*/) const {
        return false;
    }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size
    int32_t hc_mult    = 1;  // target residual streams per tapped layer
    int32_t n_embd_nextn = 0; // row width of the nextn embeddings buffer = n_embd_out

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    // MAD-LAB / WP_DSPARK_DEBUG: instrumentation state only, never read by the decode path.
    // dbg_n_draft counts draft() calls so the expensive per-slot dump can be capped at the
    // first few. dbg_blk_* remember the previous block's position span per seq so process()
    // can report whether it is injecting features over positions the last block drafted.
    int32_t              dbg_n_draft = 0;
    std::vector<int32_t> dbg_blk_pos0;   // anchor position of the previous block, -1 = none
    std::vector<int32_t> dbg_blk_pos1;   // last MASK position of the previous block

    // MAD-LAB: sidecar services mode. True when the draft model carries no LM head of
    // its own, i.e. it used to borrow the target's tok_embd/output through ctx_other.
    // That borrowing is impossible once the target is Meta-split (-sm tensor), so the
    // two borrowed ops run on the target and the results cross as host buffers:
    //   embd_buf  [n_tokens, n_embd_dec]  gathered token embeddings, fed in on the batch
    //   base_buf  [n_tokens, n_vocab]     LM-head projection of the exported hidden state
    //   conf_buf  [n_tokens]              Markov-head acceptance confidences
    bool               services_mode = false;
    int32_t            n_vocab_dft   = 0;
    std::vector<float> embd_buf;
    std::vector<float> hidden_buf;
    std::vector<float> base_buf;
    std::vector<float> conf_buf;

    const bool collect_conf_stats;

    // The previous block's drafted tokens, carried across draft() calls to be
    // hinted at the top of the next one -- see the hint site in draft() for why
    // this block's own tokens cannot buy any lead time.
    std::vector<llama_token> prev_draft_toks;

    // The Markov head's acceptance confidence for each of those tokens, same
    // order and length. Without it the hint site can only say "these tokens
    // might come next"; with it, it can say how likely each one is, which is
    // what the expert-level gate in prefetch_for_tokens spends its budget on.
    std::vector<float> prev_draft_conf;

    std::vector<std::vector<float>> capture_embd;
    int32_t capture_n_embd = 0;

    // WP_SPEC_PREDICT_PREV=0 turns the predicted half off and leaves only
    // id_last, which is ground truth. One binary, both arms, and it isolates
    // exactly the part that can be wrong.
    const bool spec_predict_prev = [] {
        const char * e = std::getenv("WP_SPEC_PREDICT_PREV");
        return e == nullptr || e[0] != '0';
    }();

    // How many of the previous block's tokens to hint. 0 = all.
    const int spec_predict_n = [] {
        const char * e = std::getenv("WP_SPEC_PREDICT_N");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 2;
        return v > 0 ? (int) v : 0;
    }();

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
        , collect_conf_stats(params.draft.conf_mode == COMMON_SPECULATIVE_DRAFT_CONF_MODE_PER_TOKEN || wp_dspark_debug())
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        hc_mult       = (int32_t) llama_model_dflash_hc_mult(model_dft);
        GGML_ASSERT(hc_mult > 0);
        // *** THE ROW WIDTH OF THE nextn BUFFER IS n_embd_out, NOT n_embd. ***
        // llama_get_embeddings_nextn() returns embd.data + j*n_embd_out (llama-context
        // .cpp:1032), and for DS4-Flash n_embd_out = dsv4_hc_mult * n_embd = 4*4096 =
        // 16384 -- the four Manifold-Constrained Hyper-Connection residual streams.
        // Every consumer below used n_embd_dec (4096) to stride that buffer, which is
        // correct ONLY for row 0. Two consequences, both measured on 2026-08-04:
        //   1. the injection memcpy under-copied and the batch was under-allocated, so
        //      a chunk > 512 tokens ran past the end (n_chunk*16384 > n_batch*4096).
        //      That is the segfault at n_ubatch=1024 and the silent corruption at 2048.
        //   2. the conf_min gate read conf[idx*4096] out of 16384-wide rows, i.e. a
        //      quarter into the WRONG row for every idx > 0, so block truncation fired
        //      on arbitrary values. That is why the drafter emitted ~3 of a trained 5
        //      and mean accepted length sat at ~2.0 against a historical 3.5-5.9.
        n_embd_nextn  = llama_model_n_embd_out(model_dft);
        GGML_ASSERT(n_embd_nextn >= n_embd_dec);
        // MAD-LAB: DSpark target taps are collapsed to n_embd_tgt at extraction.
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        const char * block_size_source = "default";
        block_size = 16;
        if (const uint32_t model_block_size = llama_model_dflash_block_size(model_dft); model_block_size > 0) {
            block_size = (int32_t) model_block_size;
            block_size_source = "accessor";
        } else {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
                block_size_source = "metadata-probe";
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        // MAD-LAB: a sidecar GGUF ships no LM head, which is the signal that this draft
        // cannot produce logits in its own graph and that the two borrowed ops
        // (token_embd gather, LM-head projection) must be routed through the target.
        services_mode = !llama_model_has_output_head(model_dft);
        n_vocab_dft   = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        // conf_min at WARN: llama-server default logger threshold is 3; libllama
        // INFO maps to 4 and is filtered, WARN maps to 2 and passes. A gate whose
        // value you cannot see in the log has cost this project multiple
        // retracted measurement sets.
        LOG_WRN("%s: - n_max=%d, n_min=%d, p_min=%.2f, conf_min=%.2f, conf_mode=%s (0=gate off)\n",
                __func__, this->params.n_max, this->params.n_min, this->params.p_min, this->params.conf_min,
                this->params.conf_mode == COMMON_SPECULATIVE_DRAFT_CONF_MODE_PER_TOKEN ? "per-token" : "chain");
        LOG_WRN("%s: - block_size=%d (source=%s), mask_token_id=%d, n_extract=%u, hc_mult=%d, sample_from_anchor=%s\n", __func__, block_size, block_size_source, mask_token_id, target_layer_ids_n, hc_mult, sample_from_anchor ? "true" : "false");
        LOG_WRN("%s: - services_mode=%d (1 = sidecar without an LM head: token_embd gather and head projection run on the target)\n",
                __func__, (int) services_mode);

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }

        // Keep the draft result within the server verify width. The verify
        // batch has one sampled row in addition to the draft rows.
        const int32_t const_shape_width = wp_ds4_const_shape_width();
        if (const_shape_width > 0) {
            const int32_t draft_width = const_shape_width;
            this->params.n_max = std::min(this->params.n_max, draft_width);
            this->params.n_min = std::min(this->params.n_min, draft_width);
        }

        // MAD-LAB / multi-sequence-safe: PREVENT torn DSpark blocks, don't just tolerate
        // them. draft() (below) packs one equal-width block per drafting sequence into a
        // single shared llama_decode(ctx_dft, ...) call -- up to n_seq blocks of
        // n_shape_tokens each, back-to-back in one llama_batch. llama_kv_cache::init_batch()
        // (src/llama-kv-cache.cpp) splits that batch with split_simple() whenever ctx_dft
        // runs a single unified KV stream (n_stream==1, the common --parallel>1 config,
        // inherited from the target's --kv-unified unless overridden), and split_simple()
        // slices strictly by raw token position with NO regard for sequence boundaries. If
        // the worst-case batch (every slot drafting a full-width block at once) is larger
        // than ctx_dft's n_ubatch, split_simple() can cut straight through the middle of a
        // block -- one ubatch ends up holding a partial block from one sequence plus a few
        // leading tokens of the next, which is exactly the shape the DSpark Markov head
        // (src/models/dflash.cpp, llama_dspark_build_markov_graph) cannot recover a correct
        // chained-Markov bias or confidence for.
        //
        // Only the DSpark markov head imposes this block-alignment requirement (plain
        // DFlash denoising is fine split across ubatches -- the KV cache still accumulates
        // correctly), so only enforce it when the draft model actually carries markov
        // weights. Fail HARD at construction, not with a warning that can go unread: the
        // caller (common_speculative_init_from_params, tools/server/server-context.cpp)
        // already catches std::runtime_error from this constructor and disables speculative
        // decoding rather than crashing the server, so this degrades the server to "no
        // speculative decoding" with a clear, actionable message instead of either aborting
        // mid-request (the original crash) or silently degrading the confidence channel.
        if (llama_model_has_dspark_markov(model_dft)) {
            const int32_t n_block_tokens_max = this->params.n_max + (is_dspark ? 0 : 1);
            const int32_t n_shape_tokens_max = const_shape_width > 0 ? const_shape_width + 1 : n_block_tokens_max;
            const int64_t n_ubatch_dft       = llama_n_ubatch(ctx_dft);
            const int64_t worst_case_tokens  = (int64_t) n_seq * n_shape_tokens_max;

            if (worst_case_tokens > n_ubatch_dft) {
                throw std::runtime_error(string_format(
                    "%s: ctx_dft's n_ubatch (%d) is too small for DSpark multi-sequence "
                    "drafting: with n_parallel=%u sequences each drafting a block of up to "
                    "%d tokens, the shared draft batch can reach %" PRId64 " tokens, which "
                    "the KV cache's ubatch splitter can tear mid-block once n_ubatch is "
                    "smaller than that. Raise the draft context's --ubatch-size (-ub, or "
                    "the draft-specific override if this rig has one) to at least %" PRId64
                    ", or reduce --parallel / the draft block width.",
                    __func__, (int) n_ubatch_dft, n_seq, n_shape_tokens_max,
                    worst_case_tokens, worst_case_tokens));
            }
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,            n_seq);
        // n_embd_nextn, not n_embd_dec: the injected rows are n_embd_out wide.
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_nextn, n_seq);

        smpls.resize(n_seq);
        if (common_speculative_capture_enabled()) {
            capture_embd.resize(n_seq);
        }
        dbg_blk_pos0.assign(n_seq, -1);
        dbg_blk_pos1.assign(n_seq, -1);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // MAD-LAB: every target tap must be resident in THIS process.
        //
        // A DFlash/DSpark sidecar conditions on the target's hidden states at fixed layers
        // (dflash.target_layers). Under a cross-machine dense pipeline this head builds
        // only the layers in its own band, so a tap outside the band is never produced:
        // llama_set_embeddings_layer_inp() accepts the id -- it IS a valid layer of the
        // full model -- but t_layer_inp[il] stays null and llm_graph_result::set_outputs
        // aborts on the first real decode, a long way from the cause. Fail here instead,
        // with the numbers that explain it. The server catches this and runs without
        // speculative decoding rather than dying.
        {
            const llama_model * model_tgt = llama_get_model(ctx_tgt);

            int32_t band_first = 0;
            int32_t band_last  = 0;
            llama_model_pipeline_band(model_tgt, &band_first, &band_last);

            // MAD-LAB: a target_layer of n_layer() taps the boundary AFTER the last main
            // layer, not the (nonexistent) input to a layer n_layer() -- see
            // set_layer_boundary_inp(il+1, ...) in src/models/deepseek4.cpp, called once
            // per il from *inside* the per-layer loop, so the tap at n_layer() falls out
            // of the very last loop iteration (il = n_layer()-1) rather than requiring a
            // layer n_layer() to be built. This is exactly how DSpark's nextn head taps
            // the target: target_layer_ids holds n_layer() itself (43 here), meaning
            // "everything the main stack produced", and any process that has computed
            // through the model's last main layer already holds it -- no extra graph
            // output, no cross-process forward needed.
            //
            // band_last = pipeline_layer_last() is the index of the last main layer this
            // process COMPUTES, so the boundary tap immediately after it (band_last + 1)
            // is always available in-band too. Only extend that far when this process
            // owns the model's main layers end-to-end (band_first == 0 and band_last ==
            // n_layer()-1) -- i.e. not pipeline-banded, or banded but the band happens to
            // cover the whole thing. A dense-segment worker that owns only a prefix or
            // middle slice must still get that boundary tap from the manifest, same as
            // before: this does not change behavior for band_last < n_layer()-1.
            const int32_t n_layer_tgt   = llama_model_n_layer(model_tgt);
            const int32_t band_last_eff = (band_first == 0 && band_last == n_layer_tgt - 1)
                ? band_last + 1
                : band_last;

            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const int32_t il = target_layer_ids[k];
                if (il >= band_first && il <= band_last_eff) {
                    continue;
                }
                // Out of band, but a dense-segment peer may be forwarding it. The head
                // arms those before constructing us (tools/server/server-context.cpp),
                // so this is the point where the manifest's declared taps are checked
                // against what the draft actually needs. A manifest that under-declares
                // fails HERE rather than leaving the draft on a stale buffer -- which
                // would change no verified token and so survive every parity test.
                if (llama_get_embeddings_layer_inp_external(ctx_tgt, (uint32_t) il)) {
                    continue;
                }
                throw std::runtime_error(string_format(
                    "%s: the draft taps target layer %d, but this process owns only target layers "
                    "[%d, %d] and no dense segment is forwarding it. Add %d to the owning segment's "
                    "\"tap_layers\" in the manifest (on the head AND the worker), run against a target "
                    "that owns the whole model, or use a draft whose target_layers fit the band.",
                    __func__, il, band_first, band_last, il));
            }
        }

        // turn on extraction of the target layers' input embeddings -- but only for the
        // layers this process actually computes. An externally supplied tap already has
        // its buffer reserved, and arming it here would additionally demand a graph
        // output the banded graph cannot produce.
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const uint32_t il = (uint32_t) target_layer_ids[k];
            if (llama_get_embeddings_layer_inp_external(ctx_tgt, il)) {
                continue;
            }
            llama_set_embeddings_layer_inp(ctx_tgt, il, true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        if (common_speculative_capture_enabled()) {
            llama_set_embeddings_layer_inp(ctx_dft, 0, true);
        }
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // skipping the embedding batches leaves a hole in the draft's cache and
        // the next injection fails to initialize.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // (c) MAD-LAB / WP_DSPARK_DEBUG: encoder/injection census.
            //
            // Reports how many context tokens get feature cells this call and over which
            // positions, plus the draft cache's state BEFORE the injection decode, plus
            // whether any of these positions was drafted by the previous block. The last
            // one is the "wrong-token features at committed positions" suspect: a position
            // the previous block MASK-drafted and the target then rejected must still be
            // injected with the TARGET's features, never left carrying draft-derived ones.
            if (wp_dspark_debug()) {
                const int32_t p_beg = batch_in.pos[i_batch_beg[seq_id]];
                const int32_t p_end = batch_in.pos[i_batch_end[seq_id]];

                int32_t   n_cells = -1, n_ge = -1, n_dup = -1;
                llama_pos p_min   = -1, p_max = -1;
                const bool ok = llama_dspark_kv_census(llama_get_memory(ctx_dft), seq_id,
                        p_beg, &n_cells, &n_ge, &n_dup, &p_min, &p_max);

                const int32_t b0 = dbg_blk_pos0[seq_id];
                const int32_t b1 = dbg_blk_pos1[seq_id];
                const bool overlaps_prev_block = (b0 >= 0) && (p_beg <= b1) && (p_end >= b0);

                SPC_INF("DBG inject seq=%d rows=%d pos=[%d,%d] | pre-inject cache: ok=%d "
                        "cells=%d pos=[%d,%d] at_or_above_%d=%d dup=%d | prev_block=[%d,%d] overlap=%d\n",
                        seq_id, n_rows, p_beg, p_end,
                        (int) ok, n_cells, (int) p_min, (int) p_max, p_beg, n_ge, n_dup,
                        b0, b1, (int) overlaps_prev_block);
            }

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        // MAD-LAB: DSpark taps are collapsed per layer, like EAGLE3.
                        const int32_t n_embd_layer = n_embd_tgt;
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_layer;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_layer;
                        std::memcpy(dst, src, (size_t) n_embd_layer * sizeof(float));
                    }
                }

                // fuse extracted features through DFlash encoder
                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // inject the DFlash decoder K/V cache at the tokens' target positions
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_nextn * sizeof(float));
                {
                    // WP_CAPTURE_DFLASH (read-only, gated): DFlash predictive hidden inp_g[i]
                    // (predicts pos+1) + target position. In the DFlash class process(). Off by default.
                    static const int s_cap_df = [](){ const char* e=std::getenv("WP_CAPTURE_DFLASH"); return (e&&e[0]=='1')?1:0; }();
                    if (s_cap_df) {
                        static FILE* s_df_fp = std::fopen("/home/kmbandy/wp_logs/accounting/dflash_capture.bin","wb");
                        if (s_df_fp) {
                            for (int32_t i = 0; i < n_chunk; ++i) {
                                int32_t hdr[2] = { (int32_t) batch_in.pos[i_batch_beg[seq_id] + offset + i], (int32_t) n_embd_nextn };
                                std::fwrite(hdr, sizeof(hdr), 1, s_df_fp);
                                std::fwrite(inp_g + (size_t) i * n_embd_nextn, sizeof(float), (size_t) n_embd_nextn, s_df_fp);
                            }
                            std::fflush(s_df_fp);
                        }
                    }
                }

                for (int32_t i = 0; i < n_chunk; ++i) {
                    batch_inject.pos[i]       = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        if (common_speculative_capture_enabled()) {
            capture_n_embd = 0;
            for (auto & rows : capture_embd) {
                rows.clear();
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.n_past;

            const int32_t n_draft = params.n_max;

            const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            const int32_t const_shape_width = wp_ds4_const_shape_width();
            const int32_t n_shape_tokens = const_shape_width > 0 ? const_shape_width + 1 : n_block_tokens;
            GGML_ASSERT(n_block_tokens <= n_shape_tokens);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            // MAD-LAB / WP_DSPARK_ANCHOR_ABLATE: normally anchor_id == dp.id_last.
            llama_token anchor_id = dp.id_last;
            if (wp_dspark_anchor_ablate() != 0) {
                anchor_id = wp_dspark_anchor_ablate();
                if (dbg_n_draft < 3) {
                    SPC_WRN("ANCHOR ABLATION ACTIVE: replacing id_last=%d with %d "
                            "(this run's acceptance numbers are a probe, not a measurement)\n",
                            dp.id_last, anchor_id);
                }
            }

            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? anchor_id : mask_token_id, n + i, { seq_id }, true);
            }

            if (n_block_tokens < n_shape_tokens) {
                if (mask_token_id == LLAMA_TOKEN_NULL) {
                    GGML_ABORT("WP_DS4_CONST_SHAPE requires a vocabulary mask token for draft padding");
                }
                for (int32_t i = n_block_tokens; i < n_shape_tokens; ++i) {
                    common_batch_add(batch, mask_token_id, n + i, { seq_id }, true);
                }
            }

            // (a) MAD-LAB / WP_DSPARK_DEBUG: draft-cache census, BEFORE the block decode.
            //
            // This is the measurement that settles the stale-cell question. The drafter is
            // non-causal with no sliding window, so every resident cell of this sequence is
            // visible to every slot of the block regardless of position. Therefore:
            //   dup  == 0 and at_or_above_n == 0  -> cache is clean, pollution ruled OUT
            //   dup  >  0                         -> duplicate cells stacked on positions
            //   at_or_above_n > 0                 -> leftover cells from previous blocks
            // A clean cache here retires the hypothesis for good and explains the zero
            // delta from the seq_rm attempt: it was removing nothing.
            if (wp_dspark_debug()) {
                int32_t   n_cells = -1, n_ge = -1, n_dup = -1;
                llama_pos p_min   = -1, p_max = -1;
                const bool ok = llama_dspark_kv_census(llama_get_memory(ctx_dft), seq_id,
                        n, &n_cells, &n_ge, &n_dup, &p_min, &p_max);

                SPC_INF("DBG census seq=%d call=%d n_past=%d block=[%d,%d] | ok=%d cells=%d "
                        "pos=[%d,%d] at_or_above_%d=%d dup=%d | expect clean: cells==n_past, "
                        "at_or_above==0, dup==0\n",
                        seq_id, dbg_n_draft, n, n, n + n_block_tokens - 1,
                        (int) ok, n_cells, (int) p_min, (int) p_max, n, n_ge, n_dup);
            }

            dbg_blk_pos0[seq_id] = n;
            dbg_blk_pos1[seq_id] = n + n_block_tokens - 1;
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // *** THE PREFETCH HINT THAT ACTUALLY BUYS LEAD TIME. ***
        //
        // dp.id_last is the last ACCEPTED token, so it is the first token of the
        // target's next verify batch -- ground truth, not a prediction. Its DS4
        // hash-layer experts (blocks 0..2) are needed by that verify pass, and
        // between here and there sits the ENTIRE draft decode below: on DSpark,
        // three NextN layers measured at ~12.6 ms each. That is ~38 ms of lead
        // against a ~5 ms cold expert read.
        //
        // Contrast with the post-draft hook further down, and with the per-ubatch
        // hint in llama_context::decode: both fire microseconds before the pass
        // that consumes them, so they cover the right experts with almost no time
        // to fetch them. The whole 2026-07 cross-layer attempt failed for exactly
        // this reason -- a sub-10 ms horizon cannot hide a 5 ms read, and a
        // predictor at 0.973 precision@rank-1 still lost at every width. LEAD
        // TIME, NOT PREDICTION QUALITY, IS THE VARIABLE.
        //
        // Before llama_decode, so no dispatch is in flight on these sockets.
        // Advisory: cannot throw, cannot block, ignores its own failures.
        {
            std::vector<llama_token> known;
            known.reserve(n_seq + prev_draft_toks.size());
            size_t n_certain = 0;
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_block_beg[seq_id] >= 0) {
                    known.push_back(dparams[seq_id].id_last);
                }
            }
            n_certain = known.size();

            // *** THE PREDICTED HALF, AND WHY IT IS HERE AND NOT BELOW. ***
            //
            // id_last above is ONE token of a verify batch that holds about six.
            // The other five are this block's drafted tokens -- ground truth for
            // the verify, known only when the decode below FINISHES, which is
            // microseconds before the pass that consumes them. So the ~38 ms of
            // lead this site owns is currently spent on 1 token in 6. DSpark
            // denoises the whole masked block in a single decode, so there is no
            // earlier moment at which this block's tokens exist -- no amount of
            // re-ordering fixes that.
            //
            // What DOES exist here, with the full window ahead of it, is the
            // PREVIOUS block's tokens. Consecutive tokens share ~2.4 of 6 experts
            // (lag-1 overlap 0.399 against a 0.023 chance baseline, measured
            // 2026-07-19 over 1200 token-steps), so they are a 17x-chance
            // predictor of this block's expert set with real lead time.
            //
            // This is a PREDICTION and it can be wrong -- unlike id_last, which
            // cannot. That is what mispredict counts, and it is why the
            // amplification gate has to be read before this is called a win. The
            // 2026-07 attempts failed on lead time, not on prediction quality;
            // this trades a little of the second for a lot of the first.
            // VOLUME IS A SEPARATE KNOB FROM SIGNAL. The first matrix run hinted
            // the WHOLE previous block: 1222 extra expert ids, which produced 76
            // FEWER used. Not a queueing problem -- spec_dropped was 0 and the
            // queue never exceeded 9 -- but a POOL one. With a lease every
            // speculative page holds a slot for its window, so extra hints and
            // lease occupancy multiply, and the marginal hint displaces a better
            // one already resident.
            //
            // Overlap also decays with distance (lag-1 0.399, lag-2 0.335, lag-3
            // 0.301 against 0.023 chance), so the nearest tokens carry most of
            // the signal and the tail carries most of the cost. WP_SPEC_PREDICT_N
            // takes the first N; 0 means all of them, which is the run above.
            //
            // SIGNAL, NOT JUST VOLUME. WP_SPEC_PREDICT_N cuts the tail by
            // POSITION, which is a proxy: it assumes token 3 is worth less than
            // token 1 because it is further away. The drafter already computed
            // the thing that proxy stands in for -- the acceptance confidence it
            // uses for its own conf_min truncation -- so carry it. Downstream,
            // an expert's confidence is the chance any token wanting it is real,
            // and WP_PREFETCH_CONF_MIN drops the rest. Without this the
            // predicted frame is the union of everything the block touched and
            // the only available cap truncates by expert id.
            std::vector<float> conf(known.size(), 1.0f);   // certain half: 1.0
            if (spec_predict_prev && !prev_draft_toks.empty()) {
                const size_t take = spec_predict_n > 0
                    ? std::min((size_t) spec_predict_n, prev_draft_toks.size())
                    : prev_draft_toks.size();
                known.insert(known.end(), prev_draft_toks.begin(),
                             prev_draft_toks.begin() + (ptrdiff_t) take);
                for (size_t i = 0; i < take; ++i) {
                    conf.push_back(i < prev_draft_conf.size() ? prev_draft_conf[i] : 1.0f);
                }
            }

            if (!known.empty()) {
                // n_certain = the id_last entries added first. Everything after
                // them came from the previous block and is a guess.
                llama_expert_prefetch_hint(this->params.ctx_tgt, known.data(),
                                           (int) known.size(), (int) n_certain,
                                           conf.data());
            }
        }

        // MAD-LAB: services mode -- gather the token embeddings on the TARGET.
        //
        // The draft graph consumes them as the embd half of this batch instead of doing
        // get_rows on a table it does not own. llm_graph_input_embd::set_input fills
        // `tokens` and `embd` from independent branches and llama_batch_allocr carries
        // both through, so the dual carry is well formed. The ids still have to be there:
        // the Markov head conditions on them, not on the embeddings.
        if (services_mode) {
            embd_buf.resize((size_t) batch.n_tokens * n_embd_dec);

            if (!llama_token_embed_gather(this->params.ctx_tgt, batch.token, batch.n_tokens, embd_buf.data())) {
                LOG_ERR("%s: token_embed_gather failed\n", __func__);
                return;
            }

            batch.embd = embd_buf.data();
        }

        // MAD-LAB / multi-sequence-safe: the constructor's hard n_ubatch >=
        // n_seq*n_shape_tokens check (see above) is what actually prevents
        // llama_kv_cache::init_batch()'s split_simple() from ever tearing this batch
        // mid-block, so there is nothing to re-check per call here anymore -- this is
        // just the debug-build tripwire confirming that invariant still holds should
        // this function's batch-sizing math ever drift out of sync with the
        // constructor's.
        assert(batch.n_tokens <= (int32_t) llama_n_ubatch(ctx_dft));

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);

        // Detach before any path can free the batch: llama_batch_free() frees ->embd,
        // and this buffer is owned by embd_buf.
        batch.embd = nullptr;

        // MAD-LAB / multi-sequence-safe: fold in whatever the graph build(s) inside that
        // llama_decode() just tallied. is_dspark is the only type that ever wires up the
        // Markov head, so this stays 0 for every other impl; polled unconditionally
        // (regardless of `ret`) since the counter reflects graph construction, not decode
        // success.
        if (is_dspark) {
            n_markov_ragged_skipped += (size_t) llama_dspark_markov_ragged_skipped_fetch_reset();
        }

        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        // MAD-LAB: services mode -- finish the step the draft graph could not.
        //
        // Graph A stopped after output_norm and exported the hidden state on the ordinary
        // embeddings path. Project it through the TARGET's head -- into our OWN buffer, so
        // the target's verification logits are left exactly as the spec loop expects them
        // -- then replay the Markov head on the draft. That writes the biased logits into
        // ctx_dft's own logits buffer, so every sampler call below is unchanged.
        //
        // One projection per draft step, not one per block position: the head is a single
        // batched mul_mat and the Markov chain conditions only on the sidecar's own w1/w2.
        if (services_mode) {
            const int32_t n_tok = batch.n_tokens;

            int32_t n_blocks_drafting = 0;
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_block_beg[seq_id] >= 0) {
                    n_blocks_drafting++;
                }
            }

            // The decoder exported the post-output_norm hidden state on the nextn channel
            // (see src/models/dflash.cpp). Those rows are n_embd_nextn wide, which equals
            // n_embd_dec only when hc_mult == 1; compact to a tight [n_tok][n_embd_dec]
            // block, which is what both services below expect.
            const float * nextn = llama_get_embeddings_nextn(ctx_dft);
            if (nextn == nullptr) {
                LOG_ERR("%s: draft exported no hidden state on the nextn channel\n", __func__);
                return;
            }

            hidden_buf.resize((size_t) n_tok * n_embd_dec);
            for (int32_t i = 0; i < n_tok; ++i) {
                std::memcpy(hidden_buf.data() + (size_t) i * n_embd_dec,
                            nextn              + (size_t) i * n_embd_nextn,
                            (size_t) n_embd_dec * sizeof(float));
            }

            const float * hidden = hidden_buf.data();

            base_buf.resize((size_t) n_tok * n_vocab_dft);
            if (!llama_output_project_to(this->params.ctx_tgt, hidden, n_tok, base_buf.data())) {
                LOG_ERR("%s: output_project_to(ctx_tgt) failed\n", __func__);
                return;
            }

            conf_buf.assign(n_tok, 1.0f);
            if (!llama_dspark_markov_head(ctx_dft, base_buf.data(), batch.token, hidden,
                        n_tok, n_blocks_drafting, conf_buf.data())) {
                LOG_ERR("%s: dspark_markov_head(ctx_dft) failed\n", __func__);
                return;
            }
        }

        if (common_speculative_capture_enabled()) {
            capture_n_embd = n_embd_dec;
        }
        const float * capture_rows = capture_n_embd > 0 ? llama_get_embeddings_layer_inp(ctx_dft, 0) : nullptr;

        // Parallel to each sequence's `result`: how likely the drafter thinks
        // each token it just proposed is. Carried to the next draft() for the
        // predicted half of the prefetch hint.
        std::vector<std::vector<float>> draft_conf(n_seq);
        std::vector<std::vector<float>> draft_conf_all(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            // (b) MAD-LAB / WP_DSPARK_DEBUG: per-slot dump, first few draft calls only.
            //
            // THE decisive distinction we do not currently have: do deep slots emit
            // plausible-but-wrong continuations (a QUALITY problem -- the head is working,
            // it just is not good enough) or degenerate output (a CORRUPTION problem --
            // repeats of slot 0, the MASK id itself, punctuation/byte junk, or a near-flat
            // top-1/top-2 gap)? Read the top-3 and the logit gap, not just the argmax.
            //
            // Read-only: uses llama_get_logits_ith directly and never touches the sampler,
            // so the tokens actually drafted below are unaffected.
            if (wp_dspark_debug() && dbg_n_draft < 3) {
                // Anchor identity, so the log shows WHICH committed token each call was
                // conditioned on. If call N+1's proposals repeat call N's at the same
                // absolute positions while THIS line changes, the anchor is being ignored.
                SPC_INF("DBG anchor seq=%d call=%d n_past=%d id_last=%d '%s'\n",
                        seq_id, dbg_n_draft, (int) dp.n_past, dp.id_last,
                        common_token_to_piece(ctx_dft, dp.id_last).c_str());

                const float * conf_dbg = services_mode
                    ? (conf_buf.empty() ? nullptr : conf_buf.data())
                    : llama_get_embeddings_nextn(ctx_dft);
                const size_t conf_stride_dbg = services_mode ? 1 : (size_t) n_embd_nextn;

                for (int32_t i = 0; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    const float * lg = llama_get_logits_ith(ctx_dft, idx);
                    if (lg == nullptr) {
                        SPC_INF("DBG slot seq=%d call=%d i=%d: no logits\n", seq_id, dbg_n_draft, i);
                        continue;
                    }

                    // -FLT_MAX rather than -INFINITY: no <cmath> dependency needed here,
                    // and it seeds the top-3 scan identically for any real logit.
                    int32_t t[3] = { -1, -1, -1 };
                    float   v[3] = { -3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f };
                    for (int32_t k = 0; k < n_vocab_dft; ++k) {
                        const float x = lg[k];
                        if (x > v[0]) { v[2]=v[1]; t[2]=t[1]; v[1]=v[0]; t[1]=t[0]; v[0]=x; t[0]=k; }
                        else if (x > v[1]) { v[2]=v[1]; t[2]=t[1]; v[1]=x; t[1]=k; }
                        else if (x > v[2]) { v[2]=x; t[2]=k; }
                    }

                    const float c = conf_dbg ? conf_dbg[(size_t) idx * conf_stride_dbg] : -1.0f;

                    // raw-vs-resolved discriminator for the pos>=2 exact-zero readout:
                    // the raw pointer indexes by batch position; _ith resolves through the
                    // output-row map. Disagreement = layout bug; agreement on 0 = the graph
                    // never wrote the row.
                    float c_ith = -1.0f;
                    if (!services_mode) {
                        const float * row_ith = llama_get_embeddings_nextn_ith(ctx_dft, idx);
                        if (row_ith != nullptr) {
                            c_ith = row_ith[0];
                        }
                    }
                    SPC_INF("DBG confrow seq=%d i=%d raw=%.3e ith=%.3e\n", seq_id, i, c, c_ith);

                    SPC_INF("DBG slot seq=%d call=%d i=%d pos=%d conf=%.3e gap=%.3f | "
                            "top1=%6d (%8.3f) '%s' | top2=%6d (%8.3f) '%s' | top3=%6d (%8.3f) '%s'%s\n",
                            seq_id, dbg_n_draft, i, (int) dp.n_past + i, c, v[0] - v[1],
                            t[0], v[0], common_token_to_piece(ctx_dft, t[0]).c_str(),
                            t[1], v[1], common_token_to_piece(ctx_dft, t[1]).c_str(),
                            t[2], v[2], common_token_to_piece(ctx_dft, t[2]).c_str(),
                            t[0] == mask_token_id ? "  <<< ARGMAX IS THE MASK TOKEN" : "");
                }
            }

            if (is_dspark) {
                // DSpark predicts the next token from position 0 and optionally truncates
                // at the first position below the confidence threshold.
                // MAD-LAB: in services mode the Markov head ran out-of-graph, so its
                // confidences are in conf_buf -- one float per token -- instead of being
                // broadcast across the n_embd_out-wide nextn embeddings buffer. Carry the
                // stride explicitly rather than assuming either layout.
                // Resolved UNCONDITIONALLY now: conf_min decides whether to
                // TRUNCATE the block, but the prefetch hint wants the per-token
                // confidence either way. The gate below stays keyed on conf_min
                // so behaviour with the gate off is unchanged.
                const float * conf        = services_mode
                    ? (conf_buf.empty() ? nullptr : conf_buf.data())
                    : llama_get_embeddings_nextn(ctx_dft);

                // MAD-LAB 2026-08-21: in-graph rows MUST be read through the
                // output-row map. The raw pointer indexes by batch position, but the
                // masked nextn buffer's row order is the OUTPUT order — for this
                // batch shape rows 2+ land elsewhere and the raw read returned
                // literal unwritten 0.0f, silently truncating every draft at
                // length 2 regardless of floor (verified raw=0.000e+00 vs
                // ith=~1.0 on-rig). services_mode conf_buf is per-token dense and
                // keeps the direct read.
                const auto conf_row = [&](int32_t idx) -> float {
                    if (services_mode) {
                        return conf[idx];
                    }
                    const float * row = llama_get_embeddings_nextn_ith(ctx_dft, idx);
                    return row != nullptr ? row[0] : 1.0f;
                };

                // MAD-LAB: per-token mode treats a decreasing head score as a
                // survival score and gates on its conditional ratio.
                const auto gate_conf_at = [&](int32_t i) {
                    const int32_t idx = beg + i;
                    const float raw_conf = conf ? conf_row(idx) : 1.0f;
                    if (params.conf_mode != COMMON_SPECULATIVE_DRAFT_CONF_MODE_PER_TOKEN || i == 0 || !conf) {
                        return raw_conf;
                    }
                    const float prev_conf = conf ? conf_row(idx - 1) : 1.0f;
                    return std::min(1.0f, raw_conf / std::max(prev_conf, 1.0e-6f));
                };

                // bonus-anchor drafts read the mask positions only (upstream #26958)
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;

                if (collect_conf_stats) {
                    for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                        draft_conf_all[seq_id].push_back(gate_conf_at(i));
                    }
                }

                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    const float raw_conf = conf ? conf_row(idx) : 1.0f;
                    const float gate_conf = gate_conf_at(i);

                    // MAD-LAB: chain mode keeps the legacy ungated first position;
                    // per-token mode applies the floor to every predicted position.
                    const bool gate_position = params.conf_mode == COMMON_SPECULATIVE_DRAFT_CONF_MODE_PER_TOKEN || i > 0;
                    if (gate_position && conf && params.conf_min > 0.0f && gate_conf < params.conf_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                    draft_conf[seq_id].push_back(raw_conf);

                    if (capture_n_embd > 0) {
                        const float * row = capture_rows + (size_t) idx * capture_n_embd;
                        capture_embd[seq_id].insert(capture_embd[seq_id].end(), row, row + capture_n_embd);
                    }
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                    draft_conf[seq_id].push_back(cur_p->data[0].p);

                    if (capture_n_embd > 0) {
                        const float * row = capture_rows + (size_t) (beg + i) * capture_n_embd;
                        capture_embd[seq_id].insert(capture_embd[seq_id].end(), row, row + capture_n_embd);
                    }
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
                draft_conf[seq_id].clear();
                if (common_speculative_capture_enabled()) {
                    capture_embd[seq_id].clear();
                }
            }

            if (collect_conf_stats) {
                if (n_draft_len_hist.size() <= result.size()) {
                    n_draft_len_hist.resize(result.size() + 1, 0);
                }
                n_draft_len_hist[result.size()]++;
                for (size_t i = 0; i < draft_conf_all[seq_id].size(); ++i) {
                    if (n_draft_conf_sum.size() <= i) {
                        n_draft_conf_sum.resize(i + 1, 0.0);
                        n_draft_conf_count.resize(i + 1, 0);
                    }
                    n_draft_conf_sum[i] += draft_conf_all[seq_id][i];
                    n_draft_conf_count[i]++;
                }
            }
        }

        // Draft-driven expert prefetch: pass actual draft token ids so the
        // pager can resolve DS4 hash-layer tid2eid experts (cold pages) and
        // pin last-pass actives across the draft->verify gap. Empty clears.
        std::vector<llama_token> draft_toks;
        std::vector<float>       draft_confs;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            const auto & res = *dparams[seq_id].result;
            draft_toks.insert(draft_toks.end(), res.begin(), res.end());
            // Same concatenation order, so draft_confs[i] belongs to
            // draft_toks[i]. A sequence that produced no confidences (a path
            // that pushed tokens without one) pads to 1.0 rather than shifting
            // every later token onto the wrong confidence.
            const auto & cf = draft_conf[seq_id];
            draft_confs.insert(draft_confs.end(), cf.begin(),
                               cf.begin() + (ptrdiff_t) std::min(cf.size(), res.size()));
            draft_confs.resize(draft_toks.size(), 1.0f);
        }
        const int n_sub = draft_toks.empty()
            ? llama_wp_on_draft_tokens(this->params.ctx_tgt, nullptr, 0)
            : llama_wp_on_draft_tokens(this->params.ctx_tgt, draft_toks.data(),
                                       (int) draft_toks.size());
        if (n_sub > 0) {
            LOG_DBG("%s: draft-prefetch submitted %d expert pages (n_draft_toks=%zu)\n",
                    __func__, n_sub, draft_toks.size());
        }

        // Do NOT hint the just-drafted tokens here. Verify is the next thing
        // on the wire, so those pages have ~0 lead: they only lengthen the
        // worker's late list. The hint at the top of draft() (id_last + optional
        // previous block) is the one with real lead. llama_wp_on_draft_tokens
        // above still pins the in-process pager for layouts that have one.

        // Carry this block forward. At the top of the NEXT draft these become the
        // predicted half of the hint, with the whole draft decode as lead.
        prev_draft_toks = draft_toks;
        prev_draft_conf = draft_confs;

        // MAD-LAB / WP_DSPARK_DEBUG: draft-call counter (instrumentation only).
        dbg_n_draft++;
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // Clear draft-window + retain pins after target verify.
        llama_wp_on_draft_tokens(this->params.ctx_tgt, nullptr, 0);
    }

    bool get_draft_capture(llama_seq_id seq_id, const float *& embeddings, int32_t & n_embd) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) capture_embd.size() || capture_n_embd <= 0 || capture_embd[seq_id].empty()) {
            return false;
        }
        embeddings = capture_embd[seq_id].data();
        n_embd = capture_n_embd;
        return true;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // MAD-LAB: a sidecar draft must never inherit the target's split mode.
        //
        // split_mode is a top-level common_params field, not part of
        // common_params_model, so `result = params` above carries the target's
        // -sm through even though result.devices has just been replaced by the
        // draft's own -devd list. Under -sm tensor that makes
        // llama_prepare_model_devices() wrap the draft's devices in a SECOND Meta
        // device -- a degenerate one-device Meta for `-devd ROCm0` -- with its own
        // split-state userdata, distinct from the target's Meta device.
        //
        // Two different Meta devices in one scheduler is not a supported state. The
        // draft's meta buffers hold 1 simple buffer while the target's meta backend
        // indexes 2, so the first draft decode aborts in
        // ggml_backend_meta_buffer_simple_tensor() at ggml-backend-meta.cpp:476.
        //
        // The draft is a small standalone model that wants to sit whole on its own
        // device; tensor-parallelising it would add an AllReduce per layer to a
        // latency-critical path for no bandwidth win. There is also no CLI surface
        // to request it (there is no -smd to pair with -devd). LAYER is the default
        // and the right answer: the borrowed target tensors still run
        // tensor-parallel on the target's Meta backend, which the draft context
        // co-schedules (see the MAD-LAB note in llama_context's backend init).
        result.split_mode = LLAMA_SPLIT_MODE_LAYER;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    // MAD-LAB: reserve one output row per sequence plus the largest speculative block.
    //
    // 2026-08-03: that budget is NOT sufficient. It covers only the decode-time
    // draft block, but the DS4/DSpark PREFILL path requests output rows at prompt
    // positions too -- measured 223 rows on the first prompt-processing call of a
    // 739-token prompt, against a budget of n_parallel*(1+n_max) ~= 28 -- which
    // trips llama-context.cpp:2435 and aborts the server mid-request. It only ever
    // appeared to work because every prior measurement used a ~5-token prompt that
    // fit under the budget by accident.
    //
    // n_batch is the host-buffer / encoder ceiling (output_reserve asserts against
    // it). It is NOT free: sched_reserve sizes the GPU logits tensor to
    // min(n_ubatch, n_outputs_max). Draft graphs now cap that reserve separately
    // (draft_graph_n_outputs) so this ceiling does not materialize
    // n_vocab*n_ubatch*4 of dead logits.
    result.n_outputs_max = params.n_batch;

    // 2026-08-10 upstream sync: upstream sets n_outputs_max_per_seq = 1 here.
    // We deliberately do NOT. That cparam is enforced by a hard abort in
    // llama_context::decode (the seq_output_count check), so 1 would reinstate
    // exactly the failure the block above removed, only relocated from a global
    // ceiling to a per-sequence one: upstream's draft loop samples one token per
    // sequence per step, while DFlash/DSpark draft a whole BLOCK per sequence and
    // the prefill path measured 223 output rows for a single sequence.
    // Leaving it 0 means "no per-seq limit beyond n_outputs_max" (llama-context
    // resolves 0 to cparams.n_outputs_max), which is the pre-sync behaviour.
    result.n_outputs_max_per_seq = 0;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    // MAD-LAB: DSpark may live inside the target GGUF or in a sidecar.
    const bool spec_dspark = std::find(params.speculative.types.begin(),
                                       params.speculative.types.end(),
                                       COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != params.speculative.types.end();
    const bool spec_dspark_self = spec_dspark && !has_draft &&
                                  llama_model_n_layer_nextn(model_tgt) > 0;
    const bool spec_self = spec_mtp || spec_dspark_self;
    GGML_ASSERT(has_draft || spec_self);
    // MAD-LAB: end

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    // Draft decoding emits at most the anchor plus n_max tokens per sequence.
    // Keep prompt processing within the same startup graph reserve.
    if ((has_draft || spec_self) && params.speculative.draft.n_max_explicit) {
        const uint32_t n_draft_tokens = (uint32_t) std::max<int64_t>(
                1, (int64_t) params.speculative.draft.n_max + 1);
        cparams.n_outputs_max_per_seq = n_draft_tokens;
        cparams.n_ubatch = std::min(cparams.n_ubatch,
                std::min(n_draft_tokens, llm_graph_logit_row_cap));
    }

    // MAD-LAB: select the graph for an in-model DSpark context.
    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    } else if (spec_dspark_self) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_DSPARK;
    }
    // MAD-LAB: end

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        // The draft is a whole standalone model: never inherit the target's
        // cross-machine pipeline band. common_base_params_to_speculative() does
        // `result = params` and then overrides only result.model (a
        // common_params_model), but pipeline_layer_first/last are top-level
        // common_params fields, so they survive into the draft's mparams via
        // common_model_params_to_llama(). A banded head then rejects the draft at
        // llama-model.cpp:2514, whose layer range lies outside the target's band.
        //
        // This is also what severs the segment-manifest coupling: the manifest
        // itself is only ever read from params_base in the server, and the sole way
        // it reaches the draft is that the head sets pipeline_layer_first/last from
        // it (server-context.cpp:1675). Clearing them here is the complete fix.
        mparams.pipeline_layer_first = -1;
        mparams.pipeline_layer_last  = -1;

        // NOTE: passing model_path rather than params.model.path is a readability
        // change, NOT a bug fix -- the two are the same string here, because
        // common_base_params_to_speculative() already assigned
        // `result.model = params_spec.mparams` for the has_draft case. Keep them in
        // sync if that assignment ever becomes conditional.
        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        // MAD-LAB: a sidecar draft that ships no LM head cannot produce logits in its own
        // graph. Its decoder stops after output_norm and exports the hidden state, which
        // the driver then projects through the target's head.
        //
        // That export rides the NEXTN channel (res->t_h_nextn), which the DFlash impl
        // already turns on with llama_set_embeddings_nextn(), so nothing extra is needed
        // here. Deliberately NOT cparams.embeddings: build_pooling() gates only on that
        // flag and would then run on this arch's encoder graph, which never sets t_embd.
        if (!llama_model_has_output_head(model_dft)) {
            LOG_INF("%s: draft has no LM head -- hidden state will be exported via the nextn channel (services mode)\n", __func__);

            // The nextn copy in llama_context::decode is guarded on pooling being NONE.
            // The DFlash arm already depends on that for its confidence read, but it was
            // only ever inherited from the defaults; make it explicit, because the hidden
            // state the whole services path is built on now rides the same guard. A draft
            // context never wants pooling, so this is unconditionally right here.
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        }

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    // MAD-LAB: create the second context on the target for MTP or in-model DSpark.
    } else if (spec_self) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
    // MAD-LAB: end
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

static void common_speculative_capture_draft(const common_speculative_impl * impl,
                                              llama_seq_id seq_id,
                                              const llama_tokens & tokens) {
    static FILE * file = []() -> FILE * {
        const char * path = std::getenv("WP_DRAFT_CAPTURE");
        if (path == nullptr || path[0] == '\0') {
            return nullptr;
        }
        FILE * result = std::fopen(path, "ab");
        if (result == nullptr) {
            return nullptr;
        }
        return result;
    }();
    static uint64_t block_id = 0;
    static bool header_written = false;
    if (file == nullptr || impl == nullptr || tokens.empty()) {
        return;
    }
    const float * embeddings = nullptr;
    int32_t n_embd = 0;
    if (!impl->get_draft_capture(seq_id, embeddings, n_embd) || n_embd <= 0 || embeddings == nullptr ||
            tokens.size() * (size_t) n_embd > SIZE_MAX / sizeof(float)) {
        return;
    }
    if (!header_written) {
        std::fseek(file, 0, SEEK_END);
        if (std::ftell(file) == 0) {
            const uint32_t header[4] = { 0x31445057u, 1u, (uint32_t) n_embd, 1u };
            std::fwrite(header, sizeof(header), 1, file);
        }
        header_written = true;
    }
    const uint32_t marker = 0x31445257u;
    const uint64_t id = block_id++;
    const uint32_t n_drafted = (uint32_t) tokens.size();
    std::fwrite(&marker, sizeof(marker), 1, file);
    std::fwrite(&id, sizeof(id), 1, file);
    std::fwrite(&n_drafted, sizeof(n_drafted), 1, file);
    std::fwrite(tokens.data(), sizeof(llama_token), tokens.size(), file);
    std::fwrite(embeddings, sizeof(float), tokens.size() * (size_t) n_embd, file);
    std::fflush(file);
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                    common_speculative_capture_draft(impl.get(), seq_id, result);
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        if (!impl->n_draft_len_hist.empty()) {
            std::ostringstream tmp;
            for (size_t i = 0; i < impl->n_draft_len_hist.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << i << ":" << impl->n_draft_len_hist[i];
            }
            str_stats += ", #draft len hist = (" + tmp.str() + ")";
        }
        if (!impl->n_draft_conf_count.empty()) {
            std::ostringstream tmp;
            // scientific, not fixed: a saturated-sigmoid mean (~1e-5) and a true 0.0
            // are indistinguishable at %.3f, and that distinction is the whole point
            // of this counter on the dspark arm.
            tmp << std::scientific << std::setprecision(3);
            for (size_t i = 0; i < impl->n_draft_conf_count.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_draft_conf_sum[i] / (double) impl->n_draft_conf_count[i];
            }
            str_stats += ", #draft conf/pos = (" + tmp.str() + ")";
        }

        // MAD-LAB / multi-sequence-safe: nonzero here means a ragged ubatch reached the
        // DSpark Markov head despite the draft context's load-time n_ubatch guard (see
        // common_speculative_impl_draft_dflash's constructor) -- that call's confidence
        // was forced to 0 rather than served stale, but the invariant that's supposed to
        // make this impossible was violated somewhere. Treat any nonzero value here as a
        // bug to chase down, not a tuning knob.
        if (impl->n_markov_ragged_skipped > 0) {
            str_stats += ", #markov ragged skipped = " + std::to_string(impl->n_markov_ragged_skipped);
        }

        // Promoted from TRC to INF (2026-08-16), alongside the server-side "acc per pos"
        // line. This counter (n_acc_tokens_per_pos) is tallied independently of the
        // server's n_accepted_per_pos, so having both visible gives a cross-check: if
        // the two per-position curves disagree, the accounting itself is wrong, which is
        // worth knowing before reading anything into either. Prints once per stats call.
        SPC_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
