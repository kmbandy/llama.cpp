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
#include <chrono>
#include <cmath>
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

// WP_SPEC_PREFILL_STATS=1: see common_speculative_wp_prefill_call_stats in
// speculative.h. Read once; when unset, process() takes none of the
// std::chrono timestamps below the gate -- same zero-cost-when-off shape as
// wp_dspark_debug()/wp_spec_hash_trace() above.
static bool wp_spec_prefill_stats_enabled() {
    static const bool s_on = [](){
        const char * e = std::getenv("WP_SPEC_PREFILL_STATS");
        return e && e[0] == '1';
    }();
    return s_on;
}

static bool wp_spec_hash_trace() {
    static const bool s_on = [](){
        const char * e = std::getenv("WP_SPEC_HASH_TRACE");
        return e && e[0] == '1';
    }();
    return s_on;
}

static uint64_t wp_spec_fnv1a(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void wp_spec_fnv1a_update(uint64_t & hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
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
    int32_t n_max; // maximum draft length after implementation-specific limits

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

    // WP_STEP_STATS: number of llama_decode(ctx_dft) calls issued by the
    // MOST RECENT draft(dparams) call. Only common_speculative_impl_draft_mtp
    // sets this (its draft() while(n_drafting>0) loop, one llama_decode per
    // iteration); every other implementation leaves it at 0. Read back via
    // common_speculative_last_n_draft_decodes() so a caller (server-context.cpp)
    // can normalise a decode step's ms by how many draft calls it cost.
    size_t n_decode_calls_last = 0;

    // WP_SPEC_PREFILL_STATS: wall-clock breakdown of the MOST RECENT
    // process() call's draft-mtp hidden-state handoff, in nanoseconds. Only
    // common_speculative_impl_draft_mtp::process() sets these (guarded by
    // wp_spec_prefill_stats_enabled(), zero-cost when unset -- the timers
    // are not taken at all in that case, these fields just stay 0); every
    // other implementation leaves them at 0. Read back via
    // common_speculative_wp_prefill_last_call_stats() so a caller
    // (server-context.cpp) can fold them into a per-prompt accumulator.
    // See common_speculative_wp_prefill_call_stats in speculative.h for
    // what each field measures.
    uint64_t wp_pp_a_sync_ns   = 0;
    uint64_t wp_pp_b_copy_ns   = 0;
    uint64_t wp_pp_c_decode_ns = 0;
    uint32_t wp_pp_c_n_chunks  = 0;
    uint64_t wp_pp_c_n_tokens  = 0;

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual void reset(llama_seq_id /*seq_id*/) {}

    virtual bool process(const llama_batch & batch) = 0;

    // MAD-LAB: prefill-sync pipelining (see draft-sync-cost-0912.txt). Some
    // implementations (currently draft-mtp's single-head path) defer part of
    // a process() call's work by one call so its target-side sync overlaps
    // the NEXT target decode() instead of blocking it from being issued.
    // Must be called once after the LAST process() call for a prompt and
    // before the first draft() call for it, to resolve whatever is still
    // pending. No-op (true) for every implementation that doesn't defer.
    virtual bool flush_pending() { return true; }

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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
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

    // MAD-LAB 2026-09-07 / upstream ggml-org#27310 -- THE INJECTION CONTRACT, made explicit.
    //
    // This one impl class drives two structurally different draft graphs, and they do NOT
    // share an injection width. Which one is in play is decided once, here, from the draft
    // model's metadata -- never inferred per call site:
    //
    //   fused_enc == true   sidecar DFlash / DFlash2 heads (dsv4_hc_mult == 0).
    //                       src/models/dflash.cpp graph<false>. Upstream's FUSED contract:
    //                       batch_inject carries RAW target features, n_embd_enc wide,
    //                       and model.fc + output_norm_enc run inside the injection graph.
    //                       One llama_decode per chunk, no llama_encode, no host round trip.
    //
    //   fused_enc == false  DS4 in-model DSpark head (dsv4_hc_mult > 0).
    //                       src/models/dflash.cpp graph_dsv4. This fork's SPLIT contract,
    //                       unchanged: a separate llama_encode() produces the encoded row,
    //                       it is read back through llama_get_embeddings_nextn() and
    //                       injected already-encoded, n_embd_nextn (= n_embd_out) wide.
    //
    // n_embd_inject is the single width every batch_inject user must go through. The two
    // widths are generally different (25600 vs 5120 on the head of record), which is
    // exactly why the 2026-09-07 attempt 7d30712e7 was wrong: it moved the SHARED width to
    // the fused one while leaving graph_dsv4 on the split contract.
    bool    fused_enc     = false;
    int32_t n_embd_inject = 0;

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

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
        : common_speculative_impl(type, n_seq, params.draft.n_max)
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

        // MAD-LAB 2026-09-07 / #27310: pick the injection contract (see the member decls).
        fused_enc     = llama_model_dsv4_hc_mult(model_dft) == 0;
        n_embd_inject = fused_enc ? n_embd_enc : n_embd_nextn;

        if (fused_enc) {
            // The fused graph declares its input at hparams.n_embd_inp_enc() =
            // n_extract * dflash_hc_mult * n_embd(draft). The gather above produces
            // n_extract * n_embd(target) COLLAPSED columns. These agree exactly when
            // dflash_hc_mult == 1 and the two models share a hidden size -- true for
            // every DFlash-family head on this box -- and disagree otherwise, in which
            // case the fused graph would be handed a row of the wrong stride and the
            // drafter would be conditioned on garbage with no visible error.
            //
            // hc_mult > 1 is therefore REFUSED, loudly, at init. The caller
            // (common_speculative_init_from_params / server-context.cpp) catches
            // std::runtime_error here and disables speculative decoding rather than
            // serving a silently mis-shaped drafter. Landing hc_mult > 1 means deciding
            // whether the gather must emit hc_mult streams per tap or the model must
            // size the encoder input collapsed -- that needs the head loaded to answer.
            const int32_t n_embd_inp_enc = (int32_t) llama_model_n_embd_inp_enc(model_dft);
            if (n_embd_inject != n_embd_inp_enc) {
                throw std::runtime_error(string_format(
                    "%s: DFlash fused-encoder width mismatch: the host gathers %d "
                    "(n_extract=%u x n_embd_tgt=%d, collapsed taps) but the draft graph's "
                    "encoder input is %d (n_extract x dflash.hc_mult=%d x n_embd_dft=%d). "
                    "Only hc_mult == 1 with matching hidden sizes is supported on the "
                    "fused path; refusing to inject a mis-strided feature row.",
                    __func__, n_embd_inject, target_layer_ids_n, n_embd_tgt,
                    n_embd_inp_enc, hc_mult, n_embd_dec));
            }
        }

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
        }

        // MAD-LAB 2026-09-07: upstream reads all three keys unconditionally; the fork's
        // block_size accessor short-circuit had buried sample_from_anchor and
        // attention.causal inside the metadata-probe branch, so a head whose block_size
        // came from the accessor silently ran with the DEFAULTS for both -- non-causal
        // block attention on a head that asks for causal, for instance.
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        // MAD-LAB: a sidecar GGUF ships no LM head, which is the signal that this draft
        // cannot produce logits in its own graph and that the two borrowed ops
        // (token_embd gather, LM-head projection) must be routed through the target.
        //
        // MAD-LAB 2026-09-07: ...but that is true of upstream's DFlash/DFlash2 heads too,
        // and those borrow the target's tok_embd/output through ctx_other exactly as
        // upstream does (src/models/dflash.cpp, graph<false>). Only the DSpark sidecar --
        // the one carrying markov_w1, whose target may be Meta-split under -sm tensor --
        // needs the out-of-graph gather/projection/markov replay. Gating on "no output
        // tensor" alone routed every upstream-format DFlash2 head into the services path,
        // where llama_dspark_markov_head() failed on a model that has no Markov head and
        // draft() returned before it could ever read the DFlash2 selector lattice.
        services_mode = !llama_model_has_output_head(model_dft) && llama_model_has_dspark_markov(model_dft);
        n_vocab_dft   = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        // conf_min at WARN: llama-server default logger threshold is 3; libllama
        // INFO maps to 4 and is filtered, WARN maps to 2 and passes. A gate whose
        // value you cannot see in the log has cost this project multiple
        // retracted measurement sets.
        LOG_WRN("%s: - n_max=%d, n_min=%d, p_min=%.2f, conf_min=%.2f, conf_mode=%s (0=gate off)\n",
                __func__, this->params.n_max, this->params.n_min, this->params.p_min, this->params.conf_min,
                this->params.conf_mode == COMMON_SPECULATIVE_DRAFT_CONF_MODE_PER_TOKEN ? "per-token" : "chain");
        LOG_WRN("%s: - block_size=%d (source=%s), mask_token_id=%d, n_extract=%u, hc_mult=%d, sample_from_anchor=%s, causal_attn=%s\n", __func__, block_size, block_size_source, mask_token_id, target_layer_ids_n, hc_mult, sample_from_anchor ? "true" : "false", causal_attn ? "true" : "false");
        // MAD-LAB 2026-09-07: causal_attn and the tap list are the two head-metadata
        // values that silently ran on defaults before the 2026-09-07 probe hoist, and
        // an acceptance regression from either is invisible without them in the log.
        {
            std::string taps;
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                taps += (k ? "," : "") + std::to_string(target_layer_ids[k]);
            }
            LOG_WRN("%s: - target_layers=[%s], n_embd_tgt=%d, n_embd_enc=%d, n_embd_nextn=%d, n_embd_dec=%d\n",
                    __func__, taps.c_str(), n_embd_tgt, n_embd_enc, n_embd_nextn, n_embd_dec);
            // MAD-LAB / #27310: the injection contract, printed so a width mismatch is
            // visible in the log even when the init-time refusal does not fire.
            LOG_WRN("%s: - fused_enc=%d (1 = upstream #27310 fused fc+output_norm_enc in graph<false>; 0 = split encoder + graph_dsv4), n_embd_inject=%d, n_embd_inp_enc=%u, dflash_hc_mult=%d, dsv4_hc_mult=%u\n",
                    __func__, (int) fused_enc, n_embd_inject,
                    llama_model_n_embd_inp_enc(model_dft), hc_mult,
                    llama_model_dsv4_hc_mult(model_dft));
        }
        LOG_WRN("%s: - is_dflash2=%d, selector_top_k=%d, has_markov=%d, has_output_head=%d\n", __func__,
                (int) is_dflash2, selector_top_k, (int) llama_model_has_dspark_markov(model_dft),
                (int) llama_model_has_output_head(model_dft));
        LOG_WRN("%s: - services_mode=%d (1 = DSpark sidecar without an LM head: token_embd gather and head projection run on the target; 0 = upstream path, tok_embd/output borrowed via ctx_other when absent)\n",
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
        this->n_max = this->params.n_max;

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

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,             n_seq);
        // MAD-LAB / #27310: n_embd_inject, per the contract picked above -- n_embd_enc on
        // the fused sidecar path, n_embd_nextn (= n_embd_out, NOT n_embd_dec) on the DS4
        // split path. process() chunks strictly by n_ubatch, so on the fused path n_ubatch
        // rows is exactly what has to fit and n_batch rows would waste
        // (n_batch-n_ubatch)*n_embd_enc*4 bytes; the split path keeps its n_batch
        // allocation byte-for-byte as it was.
        batch_inject = llama_batch_init(fused_enc ? llama_n_ubatch(ctx_dft)
                                                  : llama_n_batch(ctx_dft), n_embd_inject, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

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
        if (this->params.backend_sampling && !is_dflash2) {
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

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2);
        if (common_speculative_capture_enabled()) {
            llama_set_embeddings_layer_inp(ctx_dft, 0, true);
        }
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
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

                // MAD-LAB 2026-09-07 / upstream ggml-org#27310: gather this chunk's target
                // features, interleaved by extract layer. On the FUSED path they go
                // straight into batch_inject.embd and llama_decode() below runs
                // model.fc + output_norm_enc inside the draft graph -- one graph, no
                // llama_encode, no host round trip through the embd_nextn channel whose
                // row width and row ORDER are re-derived per graph. On the DS4 SPLIT path
                // (graph_dsv4) they are staged in features_buf and encoded separately,
                // exactly as before.
                batch_inject.n_tokens = n_chunk;

                float * features = batch_inject.embd;
                if (!fused_enc) {
                    features_buf.resize((size_t) n_chunk * n_embd_enc);
                    features = features_buf.data();
                }

                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        // MAD-LAB: DSpark taps are collapsed per layer, like EAGLE3.
                        const int32_t n_embd_layer = n_embd_tgt;
                        float       * dst = features + (size_t) i * n_embd_enc + k * (size_t) n_embd_layer;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_layer;
                        std::memcpy(dst, src, (size_t) n_embd_layer * sizeof(float));
                    }
                }

                if (!fused_enc) {
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

                    const int32_t rc_enc = llama_encode(ctx_dft, enc_batch);
                    if (rc_enc != 0) {
                        LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc_enc, (int) n_chunk, (int) offset);
                        return false;
                    }

                    const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                    GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                    // inject the DFlash decoder K/V cache at the tokens' target positions
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
                }

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        batch_inject.pos[1 * n_chunk + i] = p;
                        batch_inject.pos[2 * n_chunk + i] = p;
                        batch_inject.pos[3 * n_chunk + i] = 0;
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                const int32_t rc_dec = llama_decode(ctx_dft, batch_inject);
                if (rc_dec != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc_dec, (int) n_chunk, (int) offset);
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
                common_batch_add(batch, i == 0 ? anchor_id : mask_token_id, n + i, { seq_id }, !is_dflash2);
            }

            if (n_block_tokens < n_shape_tokens) {
                if (mask_token_id == LLAMA_TOKEN_NULL) {
                    GGML_ABORT("WP_DS4_CONST_SHAPE requires a vocabulary mask token for draft padding");
                }
                for (int32_t i = n_block_tokens; i < n_shape_tokens; ++i) {
                    common_batch_add(batch, mask_token_id, n + i, { seq_id }, !is_dflash2);
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

            if (is_dflash2) {
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                int32_t predecessor = 0;
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const float * scores = row + selector_top_k + (size_t) predecessor * selector_top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + selector_top_k));
                    if (params.p_min > 0.0f) {
                        // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                        float sum = 0.0f;
                        for (int32_t k = 0; k < selector_top_k; ++k) {
                            sum += std::exp(scores[k] - scores[predecessor]);
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                }
                continue;
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

    // MAD-LAB: prefill-sync pipelining (see draft-sync-cost-0912.txt).
    // Scoped STRICTLY to the single-head, non-shared-KV case (n_mtp_layers
    // == 1, i.e. !chain_heads -- qwen35/qwen35moe's one trained MTP head,
    // the setup this was measured on). is_mem_shared (gemma4) already skips
    // the whole catch-up decode and never sets pipeline_enabled; chain_heads
    // models (multi-head MTP) always take the original, byte-for-byte
    // unmodified path in process() below -- pipelining is wired in only
    // through the two `if (!chain_heads)` branches there, so nothing about
    // their control flow, KV writes, or per-head bookkeeping changes.
    //
    // process() for chunk K+1 captures chunk K+1's batch (tokens/pos/seq_id,
    // a cheap CPU-only copy -- NOT the sync) and, if a PRIOR chunk (K) is
    // still pending, resolves it now: sync + read ctx_tgt's *staged* nextn
    // rows for chunk K (llama_get_embeddings_nextn_staged_at(), by the
    // EXPLICIT stage slot capture_pending() recorded for chunk K via
    // llama_get_embd_nextn_stage_index() right after chunk K's own
    // decode() call -- NOT "the other slot from whatever's current now",
    // which is only correct when exactly one more decode() has happened in
    // between and breaks at a flush with nothing left to overlap; see
    // draft-sync-cost-0912.txt) +
    // the shift-copy + the draft's own catch-up llama_decode(ctx_dft, ...)
    // for chunk K + the verify_h/pending_h bookkeeping chunk K's tail needs.
    // Because chunk K+1's own llama_decode(ctx_tgt, ...) was already issued
    // by the caller (server-context.cpp's decode()) before this
    // process(batch_in) call runs, that sync drains chunk K+1's freshly
    // enqueued GPU work too -- i.e. it's no longer wasted idle wait time in
    // front of chunk K+1's issuance, it overlaps with chunk K+1's own
    // compute. See draft-sync-cost-0912.txt for the full mechanism and the
    // wall-time argument for why this is a real reduction, not just a
    // relabeling.
    //
    // flush_pending() (called once by the caller, right before the first
    // draft() for this prompt -- see common_speculative_flush_prefill())
    // resolves whatever chunk is still pending after the LAST process()
    // call, exactly like every prior resolve, just with nothing left to
    // overlap it with (same cost as today's per-chunk sync, paid once
    // instead of N times).
    bool pipeline_enabled = false;
    bool pipeline_pending = false;

    std::vector<llama_token>  pend_token;
    std::vector<llama_pos>    pend_pos;
    std::vector<llama_seq_id> pend_seq_id;
    int32_t pend_n_tokens = 0;
    std::vector<int32_t> pend_i_batch_beg;
    std::vector<int32_t> pend_i_batch_end;
    // MAD-LAB: which nextn_stage slot (0/1) THIS pending chunk's decode()
    // call wrote -- recorded explicitly at capture time via
    // llama_get_embd_nextn_stage_index(), NOT re-derived at resolve time.
    // "the other slot from whatever's current now" is only correct when
    // exactly one more decode() call has happened since capture, which is
    // false at a flush with nothing left to overlap (a prompt that fits in
    // one chunk, or any prompt's LAST pending chunk) -- see
    // draft-sync-cost-0912.txt for the have=0/need=N bug this fixes.
    int pend_stage_slot = -1;

    uint64_t hash_trace_step = 0;
    std::vector<llama_token> hash_trace_tokens;
    std::vector<int32_t> hash_trace_i_batch;

    // Per-call scratch. process() and draft() rebuild these before reading them.
    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
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

        // MAD-LAB: prefill-sync pipelining (see draft-sync-cost-0912.txt).
        // MUST happen here, at construction, NOT lazily on the first
        // process() call -- server-context.cpp's decode() always issues
        // ctx_tgt's llama_decode() for a prompt's FIRST chunk before this
        // spec's process() is ever called, so enabling staging inside
        // process() would be one decode() call too late: that first
        // chunk's own extraction would already have run with
        // nextn_stage_enabled still false and never write anything into
        // the stage buffer, leaving resolve_pending() to find nothing
        // there when it later tries to read it (have=0). Enabling here,
        // before ANY decode() on ctx_tgt has happened for this spec, means
        // even the very first chunk's extraction sees staging on. Scoped
        // to the case that actually uses it (see the pipeline_* member
        // comments above); gemma4/chain_heads never call
        // get_embeddings_nextn_staged_at() so there is no reason to pay
        // the extra per-ubatch copy for them.
        if (!is_mem_shared && !chain_heads) {
            llama_enable_embd_nextn_staging(ctx_tgt);
            pipeline_enabled = true;
        }

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        pend_i_batch_beg.assign(n_seq, -1);
        pend_i_batch_end.assign(n_seq, -1);
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

    void reset(llama_seq_id seq_id) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // MAD-LAB: if a chunk is still pending (prefill-sync pipelining --
        // see process()/resolve_pending()), resolve it now, BEFORE clearing
        // this seq's state below. Otherwise a later resolve_pending() /
        // flush_pending() would silently resurrect pending_h/verify_h for a
        // sequence that was just reset. Cheap no-op when nothing is
        // pending. May touch OTHER seq_ids present in the same pending
        // chunk too -- harmless, it's the same work they'd need resolved
        // eventually anyway, just done a little earlier.
        if (pipeline_pending) {
            resolve_pending();
        }

        std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
        verify_h[seq_id].clear();
        verify_h_rows[seq_id] = 0;
        i_last[seq_id] = -1;
        i_batch_beg[seq_id] = -1;
        i_batch_end[seq_id] = -1;
        if (chain_heads) {
            chain_h[seq_id].clear();
        }
        common_sampler_reset(smpls[seq_id].get());
        if (backend_chains[seq_id]) {
            llama_sampler_reset(backend_chains[seq_id]);
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

        if (wp_spec_hash_trace()) {
            hash_trace_tokens.assign(batch_in.token, batch_in.token + n_tokens);
            hash_trace_i_batch.resize(n_tokens);
            for (int32_t i = 0; i < n_tokens; ++i) {
                hash_trace_i_batch[i] = i;
            }
        }

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

        // if kv is shared with target (e.g Gemma4), then we can skip the
        // catch-up decode entirely -- ORIGINAL, unmodified path.
        if (is_mem_shared) {
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

        if (chain_heads) {
            // ORIGINAL, unmodified multi-head path -- MAD-LAB prefill-sync
            // pipelining (see draft-sync-cost-0912.txt) is scoped to the
            // single-head case only; chain models keep this exact
            // byte-for-byte behavior, sync included, every call.
            const bool wp_pp_stats = wp_spec_prefill_stats_enabled();
            if (wp_pp_stats) {
                wp_pp_a_sync_ns   = 0;
                wp_pp_b_copy_ns   = 0;
                wp_pp_c_decode_ns = 0;
                wp_pp_c_n_chunks  = 0;
                wp_pp_c_n_tokens  = 0;
            }

            const auto wp_pp_t_b0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

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
                const auto wp_pp_t_a0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                if (wp_pp_stats) {
                    wp_pp_a_sync_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wp_pp_t_a0).count();
                }
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            if (wp_pp_stats) {
                wp_pp_b_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wp_pp_t_b0).count() - wp_pp_a_sync_ns;
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (i_batch_beg[seq_id] < 0) {
                        continue;
                    }
                    llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                }
                llama_set_nextn_layer_offset(ctx_dft, head);

                const auto wp_pp_t_c0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
                const int32_t rc = llama_decode(ctx_dft, batch);
                if (wp_pp_stats) {
                    wp_pp_c_decode_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wp_pp_t_c0).count();
                    wp_pp_c_n_chunks += 1;
                    wp_pp_c_n_tokens += (uint64_t) n_tokens;
                }
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            if (!ok) {
                return false;
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

        // Single-head case (!is_mem_shared && !chain_heads, i.e.
        // n_mtp_layers == 1 -- qwen35/qwen35moe). Staging is enabled once,
        // at construction time (see the ctor, above) -- NOT lazily here.
        // It must be on before the very first decode() call this spec ever
        // sees on ctx_tgt, which already happened by the time this
        // process() call runs (server-context.cpp issues
        // llama_decode(ctx_tgt, ...) before calling process()); enabling it
        // lazily here would be one decode() call too late for chunk 1.
        // GGML_ASSERT, not a silent bail: if this ever fires, staging
        // wasn't enabled where it needs to be and resolve_pending() would
        // otherwise fail confusingly later with "have=0".
        GGML_ASSERT(pipeline_enabled && "nextn staging must be enabled in the ctor, not lazily in process()");

        // MAD-LAB: this SAME call site (server-context.cpp decode(), the
        // ONE place that calls common_speculative_process() for real token
        // batches) fires for BOTH prompt prefill chunks AND every
        // generation-time verify step -- see draft-sync-cost-0912.txt's
        // second bugfix. Only prefill chunks may be deferred: draft(),
        // called immediately after a verify-step process() call, needs
        // pending_h/verify_h/i_last refreshed for THAT call right now, not
        // one process() call later, and a verify call never gets a later
        // flush_pending() to resolve it (that only fires once per prompt,
        // at prompt-done) -- deferring one would leave it pending forever,
        // corrupting the NEXT thing that finally does drain it (this is
        // exactly what produced the observed "pos=8017 rc=-1" crash: a
        // stale, wrong-phase batch sitting in pend_* got resolved against
        // the wrong KV state at the next prompt's flush).
        //
        // A verify batch is at most n_max+1 tokens (the last accepted/
        // sampled token plus up to n_max drafted tokens); a real prefill
        // chunk is the server's whole n_batch-bounded slab, always far
        // larger for any config that would benefit from pipelining at all.
        // Using n_max+1 as the cutoff (rather than an arbitrary constant)
        // means a prompt/chunk that happens to be small enough to be
        // ambiguous just falls back to the always-correct synchronous
        // path instead of guessing wrong -- see process_single_head_sync().
        const bool is_prefill_chunk = n_tokens > (this->n_max + 1);

        if (!is_prefill_chunk) {
            // Generation-time verify call: byte-for-byte the ORIGINAL
            // synchronous single-head behavior (process_single_head_sync()
            // below), never deferred.
            if (pipeline_pending) {
                // Defensive only: flush_pending() is supposed to have
                // already resolved the last prefill chunk before
                // generation starts, so this should never fire. If it
                // ever does, resolve it now rather than silently
                // dropping/misordering state.
                if (!resolve_pending()) {
                    return false;
                }
            }
            return process_single_head_sync(batch_in, n_tokens);
        }

        // Prefill chunk: deferred/pipelined path. See draft-sync-cost-0912.txt
        // and the pipeline_* member comments above for the mechanism.
        bool ok = true;
        if (pipeline_pending) {
            // Resolve the PRIOR chunk now. ctx_tgt's llama_decode() for
            // THIS chunk (batch_in) was already issued by the caller before
            // this process(batch_in) call (see server-context.cpp decode():
            // llama_decode(ctx_tgt, ...) always precedes
            // common_speculative_process()), so resolve_pending()'s sync
            // overlaps that already-enqueued GPU work instead of blocking
            // its issuance.
            ok = resolve_pending();
        }

        capture_pending(batch_in, n_tokens);
        pipeline_pending = true;

        return ok;
    }

    // MAD-LAB: byte-for-byte the ORIGINAL (pre-pipelining) single-head
    // process() body -- builds the draft batch, reads ctx_tgt's nextn rows
    // via the LIVE (not staged) accessor (an immediate, un-overlapped
    // synchronize() -- correct and necessary here: this path is only used
    // for generation-time verify calls, one at a time, with nothing later
    // to overlap the wait with), runs the single catch-up
    // llama_decode(ctx_dft, ...), and fills verify_h/pending_h. Used for
    // (a) every generation-time verify call in the pipelined single-head
    // case, and (b) any prefill chunk small enough to be classified as a
    // verify call by the n_max+1 heuristic above (safe fallback, just
    // forgoes pipelining for that one chunk).
    bool process_single_head_sync(const llama_batch & batch_in, int32_t n_tokens) {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        const bool wp_pp_stats = wp_spec_prefill_stats_enabled();
        if (wp_pp_stats) {
            wp_pp_a_sync_ns   = 0;
            wp_pp_b_copy_ns   = 0;
            wp_pp_c_decode_ns = 0;
            wp_pp_c_n_chunks  = 0;
            wp_pp_c_n_tokens  = 0;
        }

        const auto wp_pp_t_b0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

        common_batch_clear(batch);
        for (int k = 0; k < n_tokens; ++k) {
            common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
        }

        {
            const auto wp_pp_t_a0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            if (wp_pp_stats) {
                wp_pp_a_sync_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wp_pp_t_a0).count();
            }
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
        }

        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
        }

        if (wp_pp_stats) {
            wp_pp_b_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wp_pp_t_b0).count() - wp_pp_a_sync_ns;
        }

        const auto wp_pp_t_c0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const int32_t rc = llama_decode(ctx_dft, batch);
        if (wp_pp_stats) {
            wp_pp_c_decode_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wp_pp_t_c0).count();
            wp_pp_c_n_chunks += 1;
            wp_pp_c_n_tokens += (uint64_t) n_tokens;
        }
        if (rc != 0) {
            SPC_ERR("llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", (int) rc, (int) batch_in.pos[0]);
            return false;
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

    // MAD-LAB: cheap CPU-only snapshot of the just-decoded chunk (tokens,
    // positions, single seq_id per token, and this call's i_batch_beg/end),
    // so resolve_pending() can replay it one process() call later without
    // touching ctx_tgt again. No sync, no GPU work -- see process() above.
    void capture_pending(const llama_batch & batch_in, int32_t n_tokens) {
        pend_token.assign(batch_in.token, batch_in.token + n_tokens);
        pend_pos.assign(batch_in.pos, batch_in.pos + n_tokens);
        pend_seq_id.resize(n_tokens);
        for (int32_t k = 0; k < n_tokens; ++k) {
            pend_seq_id[k] = batch_in.seq_id[k][0];
        }
        pend_n_tokens    = n_tokens;
        pend_i_batch_beg = i_batch_beg;
        pend_i_batch_end = i_batch_end;
        // Record which stage slot THIS chunk's already-completed decode()
        // call wrote -- ctx_tgt's decode() for batch_in ran before this
        // process(batch_in) call (see server-context.cpp decode()), so the
        // index is already fixed by the time we read it here.
        pend_stage_slot  = llama_get_embd_nextn_stage_index(this->params.ctx_tgt);
        if (wp_spec_prefill_stats_enabled()) {
            SPC_INF("captured pending chunk: tokens=%d stage_slot=%d (pair with the "
                    "\"nextn stage write ... slot=%d\" line llama-context.cpp just logged)\n",
                    n_tokens, pend_stage_slot, pend_stage_slot);
        }
    }

    // MAD-LAB: resolves whatever chunk capture_pending() last captured --
    // the deferred half of process()'s original single-head body: sync +
    // read the STAGED (not live) nextn rows for that chunk + the shift-copy
    // + the draft's own catch-up llama_decode(ctx_dft, ...) + the
    // verify_h/pending_h bookkeeping. Only called for !is_mem_shared &&
    // !chain_heads (see process() / flush_pending()).
    bool resolve_pending() {
        if (!pipeline_pending) {
            return true;
        }
        pipeline_pending = false;

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        const int32_t  n_tokens  = pend_n_tokens;
        const size_t   row_bytes = (size_t) n_embd * sizeof(float);

        const bool wp_pp_stats = wp_spec_prefill_stats_enabled();
        if (wp_pp_stats) {
            wp_pp_a_sync_ns   = 0;
            wp_pp_b_copy_ns   = 0;
            wp_pp_c_decode_ns = 0;
            wp_pp_c_n_chunks  = 0;
            wp_pp_c_n_tokens  = 0;
        }

        const auto wp_pp_t_b0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

        common_batch_clear(batch);
        for (int32_t k = 0; k < n_tokens; ++k) {
            common_batch_add(batch, pend_token[k], pend_pos[k], { pend_seq_id[k] }, 0);
        }

        const auto wp_pp_t_a0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        uint32_t n_staged = 0;
        const float * h_tgt = llama_get_embeddings_nextn_staged_at(ctx_tgt, pend_stage_slot, &n_staged);
        if (wp_pp_stats) {
            wp_pp_a_sync_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wp_pp_t_a0).count();
        }
        if (h_tgt == nullptr || (int32_t) n_staged < n_tokens) {
            SPC_ERR("draft-mtp: staged nextn rows missing/short for pending chunk "
                    "(have=%u need=%d stage_slot=%d) -- staging must be enabled before "
                    "the chunk's own decode() call; see draft-sync-cost-0912.txt\n",
                    n_staged, n_tokens, pend_stage_slot);
            return false;
        }

        std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens - 1));

        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (pend_i_batch_beg[seq_id] < 0) {
                continue;
            }
            set_h(pend_i_batch_beg[seq_id], pending_h[seq_id].data());
        }

        if (wp_pp_stats) {
            wp_pp_b_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wp_pp_t_b0).count() - wp_pp_a_sync_ns;
        }

        const auto wp_pp_t_c0 = wp_pp_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
        const int32_t rc = llama_decode(ctx_dft, batch);
        if (wp_pp_stats) {
            wp_pp_c_decode_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wp_pp_t_c0).count();
            wp_pp_c_n_chunks += 1;
            wp_pp_c_n_tokens += (uint64_t) n_tokens;
        }
        if (rc != 0) {
            SPC_ERR("llama_decode(ctx_dft) pending-chunk failed rc=%d (pos=%d)\n",
                    (int) rc, pend_n_tokens > 0 ? (int) pend_pos[0] : -1);
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (pend_i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = pend_i_batch_end[seq_id] - pend_i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = h_tgt + (size_t) (pend_i_batch_beg[seq_id] + i) * n_embd;
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    bool flush_pending() override {
        // is_mem_shared / chain_heads never set pipeline_pending, so this is
        // a no-op for them (matches the base class default).
        return resolve_pending();
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        n_decode_calls_last = 0; // WP_STEP_STATS: see the field's declaration

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

        if (wp_spec_hash_trace() && n_drafting > 0) {
            uint64_t vh = UINT64_C(14695981039346656037);
            uint64_t ph = UINT64_C(14695981039346656037);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }
                const auto & verify = verify_h[seq_id];
                const auto & pending = pending_h[seq_id];
                wp_spec_fnv1a_update(vh, verify.data(), verify.size() * sizeof(float));
                wp_spec_fnv1a_update(ph, pending.data(), pending.size() * sizeof(float));
            }

            const uint64_t de = wp_spec_fnv1a(batch.embd, (size_t) batch.n_tokens * n_embd * sizeof(float));
            std::fprintf(stderr, "SPECHASH step=%" PRIu64 " width=%d toks=", hash_trace_step++, (int) hash_trace_tokens.size());
            for (size_t i = 0; i < hash_trace_tokens.size(); ++i) {
                std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int) hash_trace_tokens[i]);
            }
            std::fprintf(stderr, " spec_i_batch=");
            for (size_t i = 0; i < hash_trace_i_batch.size(); ++i) {
                std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", hash_trace_i_batch[i]);
            }
            std::fprintf(stderr, " vh=%016" PRIx64 " ph=%016" PRIx64 " de=%016" PRIx64 "\n", vh, ph, de);
            std::fflush(stderr);
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

            // WP_DRAFT_STATS=1: split "wall ms of llama_decode" from "ms spent
            // waiting for the copy + sampling" so the 16 ms/draft-token figure
            // can be attributed instead of guessed at. The decode-side graph
            // shape/HIP-graph-counter half of this lives in
            // src/llama-context.cpp (process_ubatch, same env var, gated on
            // is_draft_ctx()) -- that is the only place with the ggml_cgraph
            // and backend-scheduler handles needed for those two numbers.
            // Unset: one getenv() + one bool check, no timers taken.
            static const char * wp_draft_env    = getenv("WP_DRAFT_STATS");
            static const bool   wp_draft_stats  = wp_draft_env != nullptr && wp_draft_env[0] == '1';
            static uint64_t wp_draft_decode_ns  = 0;
            static uint64_t wp_draft_wait_ns    = 0;
            static uint64_t wp_draft_sample_ns  = 0;
            static uint64_t wp_draft_calls      = 0;

            const auto wp_t0 = wp_draft_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }
            ++n_decode_calls_last; // WP_STEP_STATS

            const auto wp_t1 = wp_draft_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
            if (wp_draft_stats) {
                // Isolate the "copy/wait" component explicitly: without this,
                // that wait happens implicitly inside the first
                // common_sampler_sample() call below (llama_synchronize() at
                // sampling.cpp:594) and gets folded into "sampling" instead.
                llama_synchronize(ctx_dft);
            }
            const auto wp_t2 = wp_draft_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

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

            if (wp_draft_stats) {
                const auto wp_t3 = std::chrono::steady_clock::now();
                wp_draft_decode_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(wp_t1 - wp_t0).count();
                wp_draft_wait_ns   += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(wp_t2 - wp_t1).count();
                wp_draft_sample_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(wp_t3 - wp_t2).count();
                ++wp_draft_calls;

                if (wp_draft_calls % 256 == 0) {
                    // Plain LOG_WRN, not SPC_WRN: the latter's "spec %12.*s: "
                    // prefix right-justifies __func__ in a 12-wide field
                    // ("spec        draft: wp draft-stats: ..."), so a
                    // straight substring grep for "wp draft-stats: sample"
                    // still matches, but a grep anchored on "spec draft:"
                    // (single space) does not -- likely why an earlier probe
                    // of this line reported zero matches even though it was
                    // firing. Dropping the prefix here removes the ambiguity
                    // and matches the plain style of the other two
                    // "wp draft-stats: ..." banners (llama-context.cpp).
                    LOG_WRN("wp draft-stats: sample n=%" PRIu64 " decode=%.3f wait=%.3f sample=%.3f "
                            "ms/call (mean over %" PRIu64 " calls)\n",
                            wp_draft_calls,
                            wp_draft_decode_ns / 1e6 / (double) wp_draft_calls,
                            wp_draft_wait_ns   / 1e6 / (double) wp_draft_calls,
                            wp_draft_sample_ns / 1e6 / (double) wp_draft_calls,
                            wp_draft_calls);
                }
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

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || n_embd <= 0) {
            return false;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        const auto & pending = pending_h[seq_id];
        const auto & verify  = verify_h[seq_id];
        if (n_rows < 0 || pending.size() != (size_t) n_embd ||
                verify.size() != (size_t) n_rows * (size_t) n_embd) {
            return false;
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        data.resize(sizeof(n_rows) + row_bytes + verify.size() * sizeof(float));

        size_t offset = 0;
        std::memcpy(data.data() + offset, &n_rows, sizeof(n_rows));
        offset += sizeof(n_rows);
        std::memcpy(data.data() + offset, pending.data(), row_bytes);
        offset += row_bytes;
        if (!verify.empty()) {
            std::memcpy(data.data() + offset, verify.data(), verify.size() * sizeof(float));
        }
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || n_embd <= 0) {
            return;
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        if (data.size() < sizeof(int32_t) + row_bytes) {
            return;
        }

        int32_t n_rows = 0;
        std::memcpy(&n_rows, data.data(), sizeof(n_rows));
        if (n_rows < 0) {
            return;
        }

        const size_t verify_size = (size_t) n_rows * (size_t) n_embd;
        if (data.size() != sizeof(n_rows) + row_bytes + verify_size * sizeof(float)) {
            return;
        }

        pending_h[seq_id].resize(n_embd);
        std::memcpy(pending_h[seq_id].data(), data.data() + sizeof(n_rows), row_bytes);

        verify_h[seq_id].resize(verify_size);
        if (!verify_h[seq_id].empty()) {
            std::memcpy(verify_h[seq_id].data(), data.data() + sizeof(n_rows) + row_bytes, verify_h[seq_id].size() * sizeof(float));
        }
        verify_h_rows[seq_id] = n_rows;
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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
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
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
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

    std::vector<double> synth_probs;
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

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
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
        // MAD-LAB: sidecars belong to the target GGUF; a separate draft model
        // must not pick up the target's Engram tables.
        result.model_sidecars.clear();
        result.model_sidecar_ptrs.clear();
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
    // Cap the graph's output-row budget (n_outputs_max_per_seq) to that --
    // is_draft_ctx()'s draft_graph_n_tokens()/draft_graph_n_outputs() in
    // llama-context.cpp already derive the *reserved* graph size (and the
    // live LM-head row count) from n_outputs_max_per_seq, clamped again to
    // llm_graph_logit_row_cap (32). That reserve does NOT read cparams.n_ubatch
    // directly -- see llama-context.cpp:1174-1175 -- so it is unaffected by
    // whatever n_ubatch is left at below.
    //
    // 2026-09-12 WP_MTP_PREFILL_COST: cparams.n_ubatch here used to be clamped
    // to the same n_draft_tokens (<= 5 for --spec-draft-n-max 4), on the
    // reasoning above ("keep prompt processing within the same startup graph
    // reserve") -- but the reserve was already independent of n_ubatch, so that
    // clamp bought nothing and cost everything: cparams.n_ubatch is also the
    // chunk size llama_context::decode() uses to split an incoming batch
    // (memory->init_batch(*balloc, cparams.n_ubatch, ...), llama-context.cpp
    // ~3603-3614). common_speculative_impl_draft_mtp::process() feeds the
    // draft context ctx_tgt's per-target-ubatch catch-up batch (up to the
    // TARGET's n_ubatch, e.g. 2048) in one llama_decode(ctx_dft, batch) call;
    // with ctx_dft's n_ubatch clamped to 5 that call was silently exploded
    // into ceil(n_tokens/5) sub-batches -- ~410 per 2048-token target ubatch,
    // ~3200 for a 16k prompt -- each a full graph build + backend-sched launch
    // for a chunk almost entirely below the wavefront/launch-overhead floor.
    // That serialized, launch-bound loop (not the actual one-layer compute,
    // which is cheap) is what was measured as a 26-31% prefill throughput hit
    // on Qwen3.8-27B draft-mtp vs. no-spec. Leaving n_ubatch at its normal
    // (target-matching) value lets that catch-up batch run as ~1 big ubatch
    // instead of hundreds of tiny ones. KV contents and accepted-token
    // selection are unchanged: the decode-time draft loop
    // (common_speculative_impl_draft_mtp::draft()) still only ever submits
    // n_max+1 tokens per call regardless of how large n_ubatch is (larger
    // capacity, not larger use), and the LM-head/output-row cap governing what
    // draft() reads back is n_outputs_max_per_seq, set below exactly as before.
    if ((has_draft || spec_self) && params.speculative.draft.n_max_explicit) {
        const uint32_t n_draft_tokens = (uint32_t) std::max<int64_t>(
                1, (int64_t) params.speculative.draft.n_max + 1);
        cparams.n_outputs_max_per_seq = n_draft_tokens;
    }

    // 2026-09-12 scratch-bound-and-draft-vram-0912 (fixes the load-time OOM
    // 38a88719e introduced): 38a88719e removed the (n_max+1)-token n_ubatch
    // clamp above to stop common_speculative_impl_draft_mtp::process()'s
    // prompt catch-up (up to the TARGET's n_ubatch tokens, e.g. 2048, in one
    // llama_decode(ctx_dft, batch) call) from being silently exploded into
    // ceil(n_tokens/5) ~410 tiny sub-batch graph builds per target ubatch --
    // that fixed the 26-31% prefill regression (mtp-draft-prefill-cost-0912.txt)
    // but broke a load-time invariant: is_draft_ctx()'s draft_graph_n_tokens()
    // (llama-context.cpp, used by sched_reserve()) sizes the draft ctx's
    // RESERVED compute-buffer graph WIDTH (attention scratch, FFN
    // intermediate -- not just logit rows, those stay capped separately via
    // n_outputs_max_per_seq / cap_lm_head_rows) from n_outputs_max_per_seq
    // (5), on the assumption process_ubatch() would never see more than
    // n_max+1 tokens in one call. With the clamp gone, the catch-up call
    // really does build/execute a graph up to cparams.n_ubatch tokens wide
    // (2048) -- ~400x the reserved width -- so the backend scheduler has to
    // grow the compute buffer on the fly the first time that happens. On the
    // tight card in this alias (6900XT, ~100 MiB free once the target's KV +
    // weight split lands) that on-the-fly growth is exactly the observed
    // "allocating 488.00 MiB on device 1: cudaMalloc failed" load abort.
    //
    // Fix: reintroduce an n_ubatch cap for the draft ctx -- big enough to
    // keep the catch-up sub-batch count sane (still far fewer than the
    // pre-38a88719e ~410/target-ubatch) while keeping the RESERVED compute
    // buffer small enough to fit the tight card. draft_graph_n_tokens() is
    // changed alongside this (llama-context.cpp) to size the reserve from
    // cparams.n_ubatch instead of n_outputs_max_per_seq, so the reserve and
    // the live graph width agree again -- no runtime buffer growth, no OOM.
    // Configurable via WP_MTP_DRAFT_UBATCH so it can be measured; default 256
    // (8 catch-up sub-batches per 2048-token target ubatch, vs. ~410
    // pre-38a88719e and 1 -- but OOM-prone -- immediately post-38a88719e).
    // Decode-time draft width (n_max+1 = 5 tokens/step) and draft KV contents
    // are untouched: cparams.n_ubatch only bounds catch-up chunk size and
    // reserve sizing, never the per-step draft() call.
    if (has_draft || spec_self) {
        static const uint32_t wp_mtp_draft_ubatch = [] {
            const char * env = std::getenv("WP_MTP_DRAFT_UBATCH");
            if (env != nullptr && env[0] != '\0') {
                const long v = std::strtol(env, nullptr, 10);
                if (v > 0) {
                    return (uint32_t) v;
                }
            }
            return (uint32_t) 256;
        }();
        cparams.n_ubatch = std::min(cparams.n_ubatch, wp_mtp_draft_ubatch);
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
            LOG_INF("%s: draft has no LM head -- it will borrow the target's output projection via ctx_other, "
                    "or (DSpark sidecar only) export the hidden state via the nextn channel\n", __func__);

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

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
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

void common_speculative_reset(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr || seq_id < 0 || seq_id >= (llama_seq_id) spec->dparams.size()) {
        return;
    }

    spec->dparams[seq_id] = {};
    spec->impl_last[seq_id] = nullptr;

    for (auto & impl : spec->impls) {
        impl->reset(seq_id);
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

bool common_speculative_flush_prefill(common_speculative * spec) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->flush_pending();
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

size_t common_speculative_last_n_draft_decodes(const common_speculative * spec) {
    if (spec == nullptr) {
        return 0;
    }

    size_t total = 0;
    for (auto & impl : spec->impls) {
        total += impl->n_decode_calls_last;
    }
    return total;
}

common_speculative_wp_prefill_call_stats common_speculative_wp_prefill_last_call_stats(const common_speculative * spec) {
    common_speculative_wp_prefill_call_stats result;
    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result.a_sync_ns   += impl->wp_pp_a_sync_ns;
        result.b_copy_ns   += impl->wp_pp_b_copy_ns;
        result.c_decode_ns += impl->wp_pp_c_decode_ns;
        result.c_n_chunks  += impl->wp_pp_c_n_chunks;
        result.c_n_tokens  += impl->wp_pp_c_n_tokens;
    }
    return result;
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
