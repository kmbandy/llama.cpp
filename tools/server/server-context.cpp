#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "../src/llama-model.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache-paged.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/memory-tier/mt-tiered.h"
#include "../src/weight-pager/wp-pager.h"
#include "../src/weight-pager/wp-pager-set.h"
#include "../src/pipeline/pipe-dense-segment-client.h"
#include "../src/pipeline/pipe-dense-segment-manifest.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "reasoning-budget.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <filesystem>
#include <mutex>
#include <thread>
#include <utility>
#include <fstream>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static bool server_ds4_const_shape_enabled() {
    const char * value = std::getenv("WP_DS4_CONST_SHAPE");
    return value != nullptr && value[0] == '1';
}

// MAD-LAB / verify-width padding is a separate, opt-in knob from
// WP_DS4_CONST_SHAPE (2026-08-24 split). WP_DS4_CONST_SHAPE=1 alone no
// longer implies a default width of 7: measured live, that implicit pad
// drove every verify step to worst-case-width dense compute (decode
// 2.56->1.39 t/s, -45%) because DS4 verify cost scales with tokens
// verified (see server_spec_const_width() call sites below) -- padding is
// not free dispatch overhead, it is real additional expert routing/compute.
// WP_SPEC_CONST_WIDTH must be set explicitly to pad the verify batch; it
// is deprecated in favor of running const-shape unpadded. Neither var set,
// or WP_SPEC_CONST_WIDTH=0 -> width 0 -> byte-identical to the pre-existing
// off path either way.
static int32_t server_spec_const_width() {
    static const int32_t width = [] {
        const char * value = std::getenv("WP_SPEC_CONST_WIDTH");
        return value != nullptr ? std::atoi(value) : 0;
    }();
    return width;
}

// MAD-125: walk the active memory pointer chain and return the
// llama_kv_cache_paged at the bottom (if any). Three nestings to cover:
//   - paged is the raw active memory (rare standalone test case)
//   - paged is the attention member of llama_memory_hybrid (hybrid models)
//   - llama_memory_hybrid is wrapped by mt::llama_memory_tiered (the
//     tiered + paged + hybrid stack — Qwen3.x family with --kv-tiered
//     and --kv-tier-paged-blocks).
// Returns nullptr when the active memory isn't running paged-blocks.
static llama_kv_cache_paged * mt_get_paged_cache(llama_memory_i * mem) {
    if (!mem) return nullptr;
    if (auto * p = dynamic_cast<llama_kv_cache_paged *>(mem)) return p;
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem))  return h->get_mem_attn_paged();
    if (auto * t = dynamic_cast<mt::llama_memory_tiered *>(mem)) {
        return mt_get_paged_cache(t->inner_for_test());
    }
    return nullptr;
}

// MAD-122/125: write fingerprints for the token range [p0, p1). When a
// paged_cache is supplied we emit one BGE-small embedding per logical
// block via llama_kv_cache_paged::record_paged_block_fingerprint so
// query-time semantic prefetch can score at block granularity. When
// paged_cache is null we fall back to the legacy chunk-level path
// (one embedding for the whole range, position-keyed) on the tier
// wrapper — the dispatch happens at the call site so this function
// stays a single helper.
//
// mt_tier is required either way: it owns the embed_text / bge-small
// model. paged_cache is the destination; null = legacy path.
//
// Returns the number of fingerprints actually recorded.
static int mt_record_fingerprints_for_range(
        mt::llama_memory_tiered * mt_tier,
        llama_kv_cache_paged    * paged_cache,
        llama_context           * ctx,
        llama_seq_id              seq_id,
        const llama_tokens      & toks,
        int                       p0,
        int                       p1,
        uint32_t                  block_size) {
    if (!mt_tier || p1 <= p0) return 0;
    const int hi = std::min<int>(p1, (int) toks.size());
    if (hi <= p0) return 0;

    if (!paged_cache) {
        llama_tokens chunk(toks.begin() + p0, toks.begin() + hi);
        const std::string text = common_detokenize(ctx, chunk, /*special=*/ false);
        const auto emb = mt_tier->embed_text(text);
        if (emb.empty()) return 0;
        std::vector<llama_pos> positions;
        positions.reserve(hi - p0);
        for (int i = p0; i < hi; ++i) positions.push_back((llama_pos) i);
        mt_tier->record_chunk_fingerprint(
            std::move(positions), emb, mt::SemanticIndex::Tier::Warm);
        return 1;
    }

    // Paged path: walk in block_size strides starting from a block-aligned
    // floor. paged_backup is itself block-aligned so p0 should already be
    // on a boundary, but rounding down keeps the lblock arithmetic clean
    // even if a future caller passes an unaligned range.
    const uint32_t bsize = block_size > 0 ? block_size : 16u;
    const int aligned_p0 = (int)((uint32_t) p0 / bsize) * (int) bsize;
    std::vector<uint32_t> lblocks;
    std::vector<std::string> texts;
    for (int b = aligned_p0; b < hi; b += (int) bsize) {
        const int chunk_hi = std::min(b + (int) bsize, hi);
        if (chunk_hi <= b) continue;
        llama_tokens chunk(toks.begin() + b, toks.begin() + chunk_hi);
        const std::string text = common_detokenize(ctx, chunk, /*special=*/ false);
        texts.push_back(text);
        lblocks.push_back((uint32_t) b / bsize);
    }
    const auto embeddings = mt_tier->embed_text_batch(texts);
    int n_recorded = 0;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (embeddings[i].empty()) continue;
        paged_cache->record_paged_block_fingerprint(
            seq_id, lblocks[i], embeddings[i], mt::SemanticIndex::Tier::Warm);
        ++n_recorded;
    }
    return n_recorded;
}

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    // Any speculative run may need output rows at PREFILL positions, not just for
    // the decode-time draft block. DS4/DSpark asked for 223 output rows on the
    // first prompt-processing call of a 739-token prompt, against the
    // n_parallel*(1+n_max) budget below (~28), and tripped
    //   llama-context.cpp:2435 GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)
    // killing the server mid-request. It only ever worked because every previous
    // measurement used a ~5-token prompt, which fit under the budget by accident.
    //
    // n_batch is the true upper bound on outputs in a batch, and this cap is only
    // an assert ceiling -- output_reserve() allocates for the REQUESTED count, not
    // for the cap -- so raising it costs no memory until the rows are really used.
    //
    // 2026-08-10 upstream sync: common_speculative_get_output_limits() below
    // returns precisely the n_parallel*(1+n_max) budget this comment documents as
    // insufficient, and now also caps PER-SEQUENCE outputs at 1+n_max. That
    // per-seq cap is enforced by its own abort in llama_context::decode, and one
    // DFlash sequence drafting a block exceeds it unaided. Hold both at n_batch.
    if (!params.speculative.types.empty()) {
        return { (int32_t) params.n_batch, (int32_t) params.n_batch };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_all_output() {
        for (auto & token : tokens) {
            token.output = true;
        }
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    // MAD-LAB DS4-Flash pipeline-streams: this slot's index WITHIN its
    // stream (0..stream.slots.size()-1), NOT the global slot id. Every
    // per-stream resource -- spec/spec2's dparams array (sized by that
    // stream's slot count via common_speculative_init(..., n_slots_a-or-b)),
    // and each llama_context's own KV/seq-id space (sequence ids for a
    // context must be dense from 0 for THAT context, not globally unique
    // across both contexts) -- is indexed by this, never by `id`. Set once
    // at slot construction in init(): stream_slot_idx == id for stream A
    // (slots [0, n_slots_a) happen to line up with their own local index,
    // which is exactly why a stream-B bug here went undetected until a live
    // run actually exercised stream B -- see the pipeline-streams stage-2
    // post-mortem). `id` remains the GLOBAL slot id and stays the only
    // thing used for task routing (pop_deferred_task, get_slot_by_id),
    // logging (SLT_*), and response id_slot fields -- never for indexing
    // into a per-stream resource.
    int stream_slot_idx = 0;

    // Invalidates fingerprint jobs when this slot is released or reused.
    uint64_t fp_epoch = 0;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // High-water mark of positions backed up to warm by the proactive eviction
    // trigger. The next eviction starts here so we don't repeatedly try to
    // back up positions that are already in warm (which would be a no-op and
    // would leave the inner cache to overflow on long generations). Reset
    // when the slot's KV gets cleared (prompt_clear / reset).
    llama_pos kv_evict_through = 0;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    // MAD-141: counts how many consecutive update_slots() iterations have
    // preempted this slot via the MAD-120 admission gate without making
    // any progress (evict_seq returned 0). Once it crosses
    // kPagedPreemptDeadlockThreshold, the slot's request is unservable
    // (the prompt simply doesn't fit alongside the rest of the live
    // workload) and we fail it with a 503-equivalent rather than spin
    // until the upstream "n_empty_consecutive > 3" safety abort fires.
    // Reset on slot.reset() and any iteration where the slot DOES make
    // progress through the prefill path.
    int32_t paged_preempt_no_progress_count = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        // MAD-LAB DS4-Flash pipeline-streams: stream-local seq id, see
        // server_slot::stream_slot_idx.
        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_NONE); // MAD-LAB
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_NONE); // MAD-LAB
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, stream_slot_idx); // MAD-LAB: stream-local seq id
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(stream_slot_idx, -1, -1); // MAD-LAB: stream-local seq id

        prompt.clear();
        kv_evict_through = 0;
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        paged_preempt_no_progress_count    = 0;  // MAD-141

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, stream_slot_idx, nullptr); // MAD-LAB: stream-local seq id

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    // The trunk needs to emit logits at every prefill position when either:
    //  - the task asked for embeddings, or
    //  - any speculative impl needs target embeddings (MTP's hook reads
    //    h_pre_norm at every prompt position; common_speculative_need_embd
    //    queries all impls).
    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        const int32_t spec_const_width = server_spec_const_width();
        if (spec_const_width > 0) {
            n_draft_max = std::min(n_draft_max, spec_const_width);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // The single-token path is ordinary decode/NLL-gate scoring, not a verify batch.
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(stream_slot_idx, inp_embd, prompt.tokens.pos_next(), true, false); // MAD-LAB: stream-local seq id, not global slot id
            } else {
                add_ok &= batch.add(stream_slot_idx, sampled, prompt.tokens.pos_next(), true, false); // MAD-LAB: stream-local seq id, not global slot id
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(stream_slot_idx, sampled, pos0++, true, false); // MAD-LAB: stream-local seq id
            for (auto token : spec_draft) {
                add_ok &= batch.add(stream_slot_idx, token, pos0++, true, false); // MAD-LAB: stream-local seq id
            }

            // MAD-LAB / WP_SPEC_CONST_WIDTH (HIP-graph shape invariance):
            // Every tensor leading dim in the DSV4 decode graph is ubatch.n_tokens
            // (comp/lid kq_mask ne[1], lid_top_k, raw k_idxs, the top-k SET_ROWS
            // index/value counts -- the whole (a)/(b)/(c)/(d) set from the shape
            // map). A spec verify batch is 1 + (accepted-forward drafts), and that
            // draft count varies 0..n_max EVERY step, so n_tokens churns and the
            // HIP-graph capture key never repeats -> capture never holds. Pad the
            // draft portion up to a CONSTANT width with masked phantom tokens so
            // every spec verify step submits exactly the same token count.
            //
            // Phantoms are pure batch padding: SAME sequence (keeps n_seqs_unq==1 ->
            // plan.n_stream constant), output=true (keeps the LM-head row count
            // constant), and they are deliberately NOT recorded in spec_i_batch and NOT pushed into
            // prompt.tokens -- so they are invisible to sampling/acceptance and to
            // the prompt. They occupy the positions a full-length draft would have
            // used; real tokens never attend to them (causal mask, strictly higher
            // positions); their KV is discarded by the unconditional
            // seq_rm(slot.id, pos_next, -1) that runs after every verify step (see
            // the accept path).
            //
            // Config, not hardcode: default off preserves current serving; set it to
            // the configured spec draft n-max and enable alongside WP_HIP_GRAPHS for
            // the capture run.
            //
            // MAD-LAB / WP_DS4_CONST_SHAPE (2026-08-24: DECOUPLED from this pad).
            // WP_DS4_CONST_SHAPE=1 no longer implies a default width here -- it used
            // to default the pin to 7 (spec-draft-n-max) when WP_SPEC_CONST_WIDTH
            // was unset, but that made every verify step pay dense compute at the
            // worst-case width even when the real accepted-draft count was small:
            // DS4 decode cost is ~linear in tokens verified (each extra verify
            // position routes several more experts), so padding is not free
            // dispatch overhead, it is real added compute -- measured decode
            // 2.56->1.39 t/s (-45%) with const-shape alone. WP_DS4_CONST_SHAPE keeps
            // the OTHER canonicalization axes constant (n_stream, index-vector rank,
            // indexer/CSA top-k KV-length padding -- see ds4_const_shape_enabled()
            // call sites in src/models/deepseek4.cpp) which is where the measured
            // expert-fingerprint stability comes from; it does NOT touch verify
            // width. Width padding is now WP_SPEC_CONST_WIDTH's decision alone,
            // independent of WP_DS4_CONST_SHAPE, and is a manual opt-in for capture
            // experiments (effectively deprecated). Unset/0 -> width 0 -> padding
            // branch below never fires -> byte-identical to the pre-existing off
            // path.
            const int32_t spec_const_width = server_spec_const_width();
            if (server_ds4_const_shape_enabled() && spec_const_width > 0) {
                GGML_ASSERT((int32_t) spec_draft.size() <= spec_const_width);
            }
            if (spec_const_width > 0 && (int32_t) spec_draft.size() < spec_const_width) {
                const llama_token mask_tok =
                    llama_vocab_mask(llama_model_get_vocab(llama_get_model(ctx_tgt)));
                if (mask_tok != LLAMA_TOKEN_NULL) {
                    for (int32_t i = (int32_t) spec_draft.size(); i < spec_const_width; ++i) {
                        add_ok &= batch.add(stream_slot_idx, mask_tok, pos0++, /*output=*/ true, false); // MAD-LAB: stream-local seq id
                    }
                } else if (server_ds4_const_shape_enabled()) {
                    GGML_ABORT("WP_DS4_CONST_SHAPE requires a vocabulary mask token for verify padding");
                }
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            ++fp_epoch;

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            // Promoted from TRC to INF (2026-08-16). This is a cumulative SURVIVAL curve
            // -- n_accepted_per_pos[i] counts verification steps that accepted at least
            // i+1 tokens, so entry i is P(accepted length > i) and the series is
            // monotonically non-increasing. It is the only line that distinguishes "the
            // drafter is weak everywhere" from "position 0 is fine and the chain dies
            // after it", and a conf gate that clamps drafting to ~1 token/block hides
            // that difference from every aggregate number, including mean length.
            //
            // It prints once per slot at task completion, not per token, so making it
            // visible costs nothing.
            SLT_INF(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        // MAD-LAB DS4-Flash pipeline-streams: `mem` here is THIS slot's own
        // memory wrapper (bound to this->ctx_tgt/ctx_dft), so seq_cp only
        // makes sense if `other` shares the same underlying KV memory --
        // i.e. the same stream. That is also exactly why the ids below must
        // be stream_slot_idx (dense per-context seq space), not global id.
        // Parent/child slot assignment is not currently stream-aware (see
        // activate_parent_child_tasks()'s comment); if a parent and its
        // child ever land in different streams this call is wrong
        // regardless of which id flavor is used (seq_cp cannot copy KV
        // across two separate llama_context instances) -- a pre-existing,
        // separately-flagged gap this fix does not close.
        GGML_ASSERT(ctx_tgt == other.ctx_tgt && "copy_state_to: parent/child slots must share the same stream's context");
        mem.seq_rm(other.stream_slot_idx,                   -1, -1);
        mem.seq_cp(stream_slot_idx, other.stream_slot_idx,  -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

// MAD-LAB DS4-Flash pipeline-streams: bundles every piece of per-stream
// state that pre_decode()/decode()/post_decode() used to read off
// server_context_impl directly (ctx_tgt, ctx_dft, spec, the shared `batch`,
// the shared `slots`, and the previously function-local-static
// paged_admit_rotor). STAGE 1 (this struct's introduction): exactly one
// instance exists, built from stream A's values (ctx_tgt/ctx_dft/spec/batch
// unchanged from today), and pre_decode()/decode()/post_decode() are
// converted to take it as a parameter instead of reading `this->` members
// -- called from update_slots() unconditionally, at both
// n_pipeline_streams==1 and >=2, so the default path is byte-identical by
// construction (there is only one server_stream either way in this stage).
// STAGE 2 (not yet implemented) adds a second instance wired to
// ctx_tgt2/spec2/batch_b and a second call, driven on its own thread.
//
// mem_for_admit is deliberately NOT a field here: today's code computes it
// fresh each tick via llama_get_memory(ctx_tgt) as a pre_decode() LOCAL, not
// a stored member, so pre_decode() keeps doing exactly that off
// stream.ctx_tgt -- storing a second copy here would just be a second,
// potentially-stale, source of truth for the same pointer.
//
// alora_scale/alora_disabled_id are NOT fields here either: they already
// live on server_batch (batch.alora_scale / batch.alora_disabled_id, see
// server_batch above), so routing pre_decode()/update_slots() through
// stream.batch already makes them per-stream for free.
//
// n_swa is NOT a field here: it is llama_model_n_swa(model), a property of
// the shared model_tgt weights, identical for every stream by construction
// (all streams share one loaded model) -- there is nothing to bundle.
struct server_stream {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;
    llama_model   * model_dft = nullptr;
    common_speculative * spec = nullptr;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // this stream's server_batch (server_context_impl::batch for stream A,
    // ::batch_b for stream B in stage 2) -- pointer because server_batch
    // owns a llama_batch with a non-trivial destructor and is not meant to
    // be copied.
    server_batch * batch = nullptr;

    // this stream's slots, in ascending slot-id order. At
    // n_pipeline_streams==1 this is every slot, in the same order
    // std::vector<server_slot>::begin()..end() would give -- so iterate()
    // over this list is byte-identical to iterate(slots, ...) over the raw
    // member today.
    std::vector<server_slot *> slots;

    // was a function-local `static uint64_t paged_admit_rotor` inside
    // pre_decode() -- moved here because a static local is ONE counter
    // shared by every call, which is correct for a single stream but wrong
    // once stage 2 calls pre_decode() twice (once per stream) with
    // genuinely separate paged-KV pools (server_context_impl::ctx_tgt vs
    // ctx_tgt2) and no shared admission budget between them.
    uint64_t paged_admit_rotor = 0;

    // was server_context_impl::n_empty_consecutive -- same reasoning: an
    // impl-level member would be shared across streams once there are two,
    // conflating "stream A produced 4 empty batches in a row" with "stream
    // B did," which is not what the upstream safety abort (decode(),
    // ++n_empty_consecutive > 3) means to detect.
    int32_t n_empty_consecutive = 0;
};

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;
    friend struct server_routes;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    // MAD-LAB DS4-Flash pipeline-streams: stream B's context (nullptr unless
    // params_base.n_pipeline_streams >= 2). Built directly off model_tgt via
    // llama_init_from_model() (NOT common_init_from_params(), which would
    // reload the model) so it shares model_tgt's already-loaded weights but
    // gets its own KV cache and -- because llama_context_params::ctx_other
    // is left null, see src/llama-context.cpp:622-660 -- its own expert-
    // dispatch connection (design (b), see update_slots() PIPELINE-STREAMS
    // comment). Owned here; freed in destroy() with llama_free().
    llama_context * ctx_tgt2 = nullptr;

    server_batch batch;

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): stream B's own batch,
    // init()-sized identically to `batch` (see the stream_b population
    // block in init()). Never touched by the main thread while the
    // stream-B thread is running this tick's decode.
    server_batch batch_b;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    // MAD-LAB DS4-Flash pipeline-streams: stream B's own draft/MTP context,
    // bound to ctx_tgt2 (never ctx_tgt) -- see the spec2 construction block
    // in init(), a straight mirror of the model_dft/ctx_dft block below but
    // targeting ctx_tgt2. For MTP/DSpark-self speculative (the dflash
    // config) this taps model_tgt's already-loaded weights, same as
    // model_dft/ctx_dft; for an external --model-draft it loads a second
    // copy of the draft model (documented, not silently duplicated).
    llama_model   * model_dft2 = nullptr;
    llama_context * ctx_dft2   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    // MAD-LAB DS4-Flash pipeline-streams: stream B's own common_speculative
    // instance (item 2 of the pipeline-streams task -- today spec/spec_init
    // above is ONE shared instance for all slots, a known compromise; a
    // second stream gets its own draft state instead of sharing stream A's).
    // Built the same way spec_init/spec are built below, against ctx_tgt2.
    // nullptr unless params_base.n_pipeline_streams >= 2 AND speculative
    // decoding is configured.
    common_speculative_init_result_ptr spec_init2;
    common_speculative_ptr spec2;

    std::unique_ptr<pipe_dense_segment_client::client> segment_client;
    uint64_t segment_session_id = 1;
    uint64_t segment_cache_epoch = 1;
    uint64_t segment_next_seq_id = 1;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    struct fp_job {
        llama_seq_id seq;
        uint32_t     lblock;
        uint64_t     epoch;
        std::string  text;
    };

    struct fp_result {
        llama_seq_id     seq;
        uint32_t         lblock;
        uint64_t         epoch;
        std::vector<float> emb;
    };

    std::thread                fp_worker_;
    std::mutex                 fp_job_mtx_;
    std::condition_variable    fp_job_cv_;
    std::deque<fp_job>         fp_jobs_;
    std::mutex                 fp_res_mtx_;
    std::deque<fp_result>      fp_results_;
    std::atomic<bool>          fp_worker_stop_{false};
    mt::llama_memory_tiered *  fp_embedder_ = nullptr;

    int trace = 0;        // env: LLAMA_TRACE
    int spec_phase = 0;   // WP_SPEC_PHASE: per-block phase timing for speculative decode
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    // MAD-LAB DS4-Flash pipeline-streams (stage 1): stream A's bundle --
    // replaces the standalone n_empty_consecutive member and the
    // function-local `static uint64_t paged_admit_rotor` that used to live
    // inside pre_decode(). Populated once slots/spec/batch exist, at the
    // end of init() (see the "pipeline-streams: build stream A's
    // server_stream" block). pre_decode()/decode()/post_decode() are called
    // with this unconditionally today (n_pipeline_streams==1 and >=2 alike)
    // -- there is exactly one server_stream in play until stage 2 adds a
    // second, so this is a pure representation change, not a behavior one.
    server_stream stream_a;

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): stream B's bundle,
    // populated in init() only when n_pipeline_streams >= 2 (wired to
    // ctx_tgt2/ctx_dft2/model_dft2/spec2/batch_b and the second half of
    // `slots`). Left default-constructed (all null/empty) otherwise --
    // never read in that case because stream_b_thread_ is never spawned and
    // the tick logic below skips it.
    server_stream stream_b;

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): persistent thread for
    // stream B, spawned once in init() when n_pipeline_streams >= 2 (never
    // otherwise -- the n_pipeline_streams==1 path never touches any of
    // this). Parked on stream_b_cv_ between ticks; woken by
    // run_stream_b_tick_if_any() setting stream_b_has_work_ and notifying;
    // signals completion by clearing stream_b_has_work_ and notifying
    // stream_b_done_cv_, which update_slots() waits on before returning.
    // See run_stream_tick()/update_slots() for the per-tick protocol and
    // the PIPELINE-STREAMS comment there for why this is a per-tick join
    // (not fully independent cross-tick ticking) as a first cut.
    std::thread             stream_b_thread_;
    std::mutex               stream_b_mtx_;
    std::condition_variable  stream_b_cv_;       // main thread -> stream-B thread: "you have work"
    std::condition_variable  stream_b_done_cv_;   // stream-B thread -> main thread: "done"
    bool                     stream_b_has_work_ = false;
    bool                     stream_b_done_ = true; // starts "done" (idle)
    std::atomic<bool>        stream_b_thread_stop_{false};

    // MAD-LAB DS4-Flash pipeline-streams (stage 2) metrics decision: a
    // narrow lock around the handful of server_metrics calls inside
    // decode()/post_decode() (on_decoded/on_prompt_eval/on_prediction),
    // rather than deferring all metrics collection to after the join.
    // Chosen over "main-thread-after-join" because those calls are already
    // interleaved token-by-token with per-slot sampling logic throughout
    // post_decode() -- buffering and merging counts after the fact would
    // mean threading a second, parallel bookkeeping structure through the
    // same body this stage just finished de-duplicating via `stream`.
    // server_metrics's own counters are plain (non-atomic) members, so
    // without this lock two threads calling on_decoded/on_prompt_eval/
    // on_prediction concurrently would race on ++/max updates -- narrow
    // enough (increment-sized critical sections) not to meaningfully
    // serialize the two streams' actual decode work, which is what this
    // whole design is for.
    std::mutex metrics_mtx_;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void start_fp_worker(mt::llama_memory_tiered * mt_tier) {
        if (fp_worker_.joinable()) {
            return;
        }

        fp_embedder_ = mt_tier;
        fp_worker_stop_.store(false);
        fp_worker_ = std::thread([this] {
            for (;;) {
                std::vector<fp_job> batch;
                {
                    std::unique_lock<std::mutex> lock(fp_job_mtx_);
                    fp_job_cv_.wait(lock, [this] {
                        return fp_worker_stop_.load() || !fp_jobs_.empty();
                    });
                    if (fp_worker_stop_.load() && fp_jobs_.empty()) {
                        return;
                    }
                    while (!fp_jobs_.empty() && batch.size() < 16) {
                        batch.push_back(std::move(fp_jobs_.front()));
                        fp_jobs_.pop_front();
                    }
                }

                std::vector<std::string> texts;
                texts.reserve(batch.size());
                for (const fp_job & job : batch) {
                    texts.push_back(job.text);
                }

                const auto embeddings = fp_embedder_->embed_text_batch(texts);
                {
                    std::lock_guard<std::mutex> lock(fp_res_mtx_);
                    for (size_t i = 0; i < batch.size(); ++i) {
                        if (i >= embeddings.size() || embeddings[i].empty()) {
                            continue;
                        }
                        fp_results_.push_back({
                            batch[i].seq,
                            batch[i].lblock,
                            batch[i].epoch,
                            embeddings[i],
                        });
                    }
                }
            }
        });
    }

    void stop_fp_worker() {
        fp_worker_stop_.store(true);
        fp_job_cv_.notify_all();
        if (fp_worker_.joinable()) {
            fp_worker_.join();
        }

        fp_embedder_ = nullptr;
        {
            std::lock_guard<std::mutex> lock(fp_job_mtx_);
            fp_jobs_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(fp_res_mtx_);
            fp_results_.clear();
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): takes the stream's own
    // ctx_tgt now (was the impl member, i.e. always stream A's). Note the
    // fp_worker_/start_fp_worker() background embedder below this point is
    // NOT per-stream -- it is one shared worker/queue for the whole
    // server, and start_fp_worker() is a no-op once any mt_tier has
    // claimed it (first caller wins, see its "if (fp_worker_.joinable())
    // return;" guard). If --kv-tier-semantic-index is ever combined with
    // --pipeline-streams >= 2, whichever stream's mt_tier calls this first
    // gets fingerprinting; the other stream's fingerprint jobs are silently
    // never embedded. Not a crash and not decode-result corruption --
    // fingerprinting is already documented as best-effort/droppable
    // elsewhere in this function -- but it is a real, un-fixed limitation
    // of this stage, flagged rather than silently accepted.
    void drain_paged_fingerprints(llama_context * ctx_tgt) {
        if (!params_base.kv_tier_paged_blocks) {
            return;
        }

        auto * paged_cache = mt_get_paged_cache(llama_get_memory(ctx_tgt));
        if (!paged_cache) {
            return;
        }

        // Always drain the eviction queue on the inference thread, even when the
        // semantic index is off, so pending_fp_evicted_ cannot grow without bound
        // on a tiered-but-nosem run that evicts (the evict_block_to_warm hook
        // records unconditionally).
        auto evicted = paged_cache->take_pending_fp_evicted();

        if (params_base.kv_semantic_index.empty()) {
            return;  // semantic off: nothing to fingerprint, queue already cleared
        }

        auto * mt_tier = dynamic_cast<mt::llama_memory_tiered *>(llama_get_memory(ctx_tgt));
        if (!mt_tier) {
            return;
        }

        start_fp_worker(mt_tier);

        if (!evicted.empty()) {
            // Backpressure: eviction can outrun the ~285 ms CPU embed. If the
            // worker is already far behind, drop new jobs rather than grow the
            // queue (and the detokenize cost) unboundedly. Fingerprinting is
            // best-effort — a dropped block simply isn't semantically restorable
            // until it is re-evicted with the worker caught up.
            constexpr size_t kMaxPendingJobs = 512;
            size_t backlog;
            {
                std::lock_guard<std::mutex> lock(fp_job_mtx_);
                backlog = fp_jobs_.size();
            }
            if (backlog < kMaxPendingJobs) {
                const uint32_t bsize = (uint32_t) std::max(1, params_base.kv_tier_paged_block_size);
                std::vector<fp_job> jobs;
                jobs.reserve(evicted.size());
                for (const auto & [seq, lblock] : evicted) {
                    server_slot * slot = get_slot_by_id(seq);
                    if (!slot) {
                        continue;
                    }

                    const auto & toks = slot->prompt.tokens.get_text_tokens();
                    const int p0 = (int) lblock * (int) bsize;
                    const int p1 = p0 + (int) bsize;
                    if (p1 > (int) toks.size()) {
                        continue;
                    }

                    llama_tokens chunk(toks.begin() + p0, toks.begin() + p1);
                    jobs.push_back({
                        seq,
                        lblock,
                        slot->fp_epoch,
                        common_detokenize(ctx_tgt, chunk, /*special=*/ false),
                    });
                }

                if (!jobs.empty()) {
                    std::lock_guard<std::mutex> lock(fp_job_mtx_);
                    for (fp_job & job : jobs) {
                        fp_jobs_.push_back(std::move(job));
                    }
                    fp_job_cv_.notify_one();
                }
            }
        }

        std::deque<fp_result> done;
        {
            std::lock_guard<std::mutex> lock(fp_res_mtx_);
            done.swap(fp_results_);
        }
        for (fp_result & result : done) {
            server_slot * slot = get_slot_by_id(result.seq);
            if (!slot || slot->fp_epoch != result.epoch) {
                continue;
            }
            paged_cache->record_paged_block_fingerprint(
                result.seq, result.lblock, std::move(result.emb), mt::SemanticIndex::Tier::Warm);
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): stop stream_b_thread_
    // before anything it might touch (ctx_tgt2, stream_b.*) gets freed.
    // Safe to call even if the thread was never spawned (join is only
    // attempted if joinable).
    void stop_stream_b_thread() {
        stream_b_thread_stop_.store(true);
        {
            std::lock_guard<std::mutex> lock(stream_b_mtx_);
            stream_b_cv_.notify_all();
        }
        if (stream_b_thread_.joinable()) {
            stream_b_thread_.join();
        }
    }

    void destroy() {
        stop_stream_b_thread();
        stop_fp_worker();

        spec.reset();
        spec_init.reset();
        spec2.reset();
        spec_init2.reset();
        segment_client.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;
        ctx_dft2   = nullptr; // MAD-LAB: owned by spec_init2, just reset above
        model_dft2 = nullptr;
        // MAD-LAB: stream_a/stream_b hold copies of these pointers; clear
        // them too so nothing downstream can read a freed context between
        // destroy() and the next init() (relevant on the sleep/wake path).
        stream_a = server_stream{};
        stream_b = server_stream{};

        // ctx_tgt2 is NOT owned by llama_init (unlike ctx_tgt) -- it was
        // built directly via llama_init_from_model(), so it must be freed
        // explicitly, and before llama_init.reset() destroys model_tgt.
        if (ctx_tgt2 != nullptr) {
            llama_free(ctx_tgt2);
            ctx_tgt2 = nullptr;
        }

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        // MAD-134: validate tier-related config BEFORE model load so
        // operators get fast clear failure instead of late crashes.
        if (!params.kv_semantic_index.empty()) {
            struct stat st;
            if (::stat(params.kv_semantic_index.c_str(), &st) != 0) {
                SRV_ERR("--kv-tier-semantic-index '%s' does not exist or is not "
                        "readable. Either provide a valid bge-small / nomic-embed gguf "
                        "file, or omit the flag to disable semantic prefetch.\n",
                        params.kv_semantic_index.c_str());
                return false;
            }
        }
        if (params.kv_tiered_enabled && params.kv_tier_cold_pct > 0.0f &&
            !params.kv_tier_ssd_path.empty()) {
            // Try to mkdir + create a test file to confirm writability.
            const std::string test_dir = params.kv_tier_ssd_path;
            (void) ::mkdir(test_dir.c_str(), 0700);  // ok if exists
            const std::string test_path = test_dir + "/.write_test_" +
                                          std::to_string(::getpid());
            int fd = ::open(test_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (fd < 0) {
                SRV_ERR("--kv-tier-ssd-path '%s' is not writable (errno %d: %s). "
                        "Pick a different path or fix permissions.\n",
                        test_dir.c_str(), errno, strerror(errno));
                return false;
            }
            ::close(fd);
            ::unlink(test_path.c_str());
        }

        params_base = params;

        // MAD-LAB DS4-Flash pipeline-streams: split --ctx-size evenly across
        // streams up front, before ctx_tgt (stream A) is built below, so
        // both/all streams are sized symmetrically and stream A's
        // construction goes through the exact same common_init_from_params()
        // path it always has (no separate code path for stream A). At
        // n_pipeline_streams==1 this is a no-op division by 1 -- byte-
        // identical to today per the campaign's exactness rule.
        //
        // Chosen sizing: params.n_ctx (the existing --ctx-size /
        // LLAMA_ARG_CTX_SIZE knob) / n_pipeline_streams, floor-divided, with
        // n_ctx==0 (== "use the model's trained context") left untouched --
        // splitting an auto-sized ctx here would require knowing the
        // model's default ctx before common_init_from_params() computes it,
        // which is not available yet. No new --ctx-size-per-stream flag was
        // added; if a use case needs asymmetric per-stream sizing later,
        // that is the natural extension point.
        if (params_base.n_pipeline_streams >= 2) {
            if (!params_base.segment_manifest.empty()) {
                SRV_ERR("%s", "--pipeline-streams >= 2 is not supported together with "
                               "--segment-manifest (dense segment client is not yet "
                               "duplicated per stream; unset one of the two)\n");
                return false;
            }
            if (params_base.n_ctx > 0) {
                const int32_t split = params_base.n_ctx / params_base.n_pipeline_streams;
                SRV_INF("pipeline-streams: splitting --ctx-size %d across %d streams -> %d per stream\n",
                        params_base.n_ctx, params_base.n_pipeline_streams, split);
                params_base.n_ctx = split;
            } else {
                SRV_WRN("%s", "pipeline-streams: --ctx-size is 0 (model default); "
                               "each stream will independently get the model's full "
                               "trained context instead of an even split -- pass an "
                               "explicit --ctx-size to size streams predictably\n");
            }
        }

        const int32_t spec_const_width = server_spec_const_width();
        if (server_ds4_const_shape_enabled() && spec_const_width > 0 &&
                params_base.speculative.draft.n_max > spec_const_width) {
            SRV_WRN("ds4: WP_DS4_CONST_SHAPE=1 clamps spec-draft-n-max from %d to %d\n",
                    params_base.speculative.draft.n_max, spec_const_width);
            params_base.speculative.draft.n_max = spec_const_width;
        }
        if (!params_base.segment_manifest.empty() && params_base.n_cache_reuse > 0) {
            SRV_WRN("%s", "dense segments do not support shifted prompt reuse; disabling --cache-reuse\n");
            params_base.n_cache_reuse = 0;
        }
        if (params_base.kv_tiered_enabled && params_base.kv_tier_hot_pct > 0.0f && params_base.kv_tier_hot_pct < 100.0f) {
            // Save the original n_ctx so the cache layer can size pools
            // based on the full ctx budget, not just the hot slice.
            params_base.kv_tier_total_ctx = params_base.n_ctx;

            if (params_base.kv_tier_paged_blocks) {
                // MAD-120: with paged-blocks the cache itself handles the
                // hot↔warm tiering — model sees full n_ctx, paged cache
                // evicts/restores blocks transparently. NO pre-shrink.
                SRV_INF("tiered KV (paged): total ctx=%d (model sees full); cache hot=%.0f%% warm=%.0f%% cold=%.0f%% — paged cache handles tier movement\n",
                        params_base.kv_tier_total_ctx,
                        params_base.kv_tier_hot_pct,
                        params_base.kv_tier_warm_pct,
                        params_base.kv_tier_cold_pct);
            } else {
                // Legacy non-paged tiered cache: model sees only hot-tier
                // worth of ctx; cache stores the rest transparently via
                // migrate_tokens. Pre-shrink keeps the model in-bounds.
                params_base.n_ctx = std::max(512, (int)(params_base.n_ctx * params_base.kv_tier_hot_pct / 100.0f));
                SRV_INF("tiered KV: total ctx=%d, hot ctx=%d (%.0f%%)\n",
                        params_base.kv_tier_total_ctx,
                        params_base.n_ctx,
                        params_base.kv_tier_hot_pct);
            }
        }
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        // MAD-LAB: include in-model DSpark in the server speculative gate.
        const bool spec_dspark = std::find(params_base.speculative.types.begin(),
                                           params_base.speculative.types.end(),
                                           COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != params_base.speculative.types.end();
        const bool spec_dspark_self = spec_dspark && !has_draft;
        const bool has_spec = has_draft || spec_mtp || spec_dspark_self;
        // MAD-LAB: end

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        // MAD-LAB DS4-Flash pipeline-streams: build stream B's context.
        // Deliberately NOT common_init_from_params() again -- that reloads
        // the model from disk into a second set of weights. Instead go
        // straight to llama_init_from_model() with model_tgt (already
        // loaded) and a llama_context_params derived from the SAME
        // (already ctx-split, see above) params_base used for ctx_tgt.
        // Because llama_context_params::ctx_other is left at its default
        // (null) here, the llama_context constructor takes the "not
        // borrowed" branch at src/llama-context.cpp:642-660 and opens its
        // OWN expert-dispatch connection -- this is the entirety of design
        // (b) (two independent dispatchers) from the pipeline-streams task;
        // no change to src/llama-context.cpp was needed. Contrast with the
        // existing MTP/draft context further below, which sets
        // params_base.speculative.draft.ctx_tgt = ctx_tgt (see
        // common_speculative_init_from_params(), which threads that through
        // to llama_context_params::ctx_other) to deliberately borrow
        // (design (a)) -- appropriate there because the draft context's
        // expert traffic is meant to ride the SAME worker connection as the
        // target's, not a design to copy for pipeline-streams.
        if (params_base.n_pipeline_streams >= 2) {
            llama_context_params cparams2 = common_context_params_to_llama(params_base);
            ctx_tgt2 = llama_init_from_model(model_tgt, cparams2);
            if (ctx_tgt2 == nullptr) {
                SRV_ERR("%s", "pipeline-streams: failed to create stream B context "
                               "(second expert-dispatch connection likely refused by "
                               "the worker -- check WP_WORKER_MULTI_CONN on the worker "
                               "side)\n");
                return false;
            }
            SRV_INF("pipeline-streams: stream B context constructed with its own "
                    "expert-dispatch connection (n_ctx=%d)\n", llama_n_ctx(ctx_tgt2));
            SRV_WRN("%s", "pipeline-streams: serving loop is not yet dual-stream -- "
                          "all slots still run on stream A (see the PIPELINE-STREAMS "
                          "comment at update_slots()); stream B's context/dispatcher "
                          "is constructed and idle\n");
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (!params_base.segment_manifest.empty()) {
            const auto manifest = pipe_dense_segment::load_manifest(params_base.segment_manifest);
            const auto & head = manifest.segments.front();
            if (head.layer_first != params_base.pipeline_layer_first ||
                head.layer_last != params_base.pipeline_layer_last ||
                llama_model_n_layer(model_tgt) != manifest.n_layer ||
                llama_model_n_embd(model_tgt) != manifest.n_embd) {
                SRV_ERR("%s", "dense segment manifest does not match the local head stage\n");
                return false;
            }
            // NEXTN SIDEBAND NEED. Only draft-mtp reads the target's nextn -- see the
            // consume site further down, which gates on
            // common_speculative_need_embd_nextn(), a predicate only
            // common_speculative_impl_draft_mtp answers true. Declaring it here lets the
            // tail skip serializing an n_embd f32 run per token on every other arm,
            // including the production draft-dspark default.
            //
            // spec_mtp is the right source rather than the built `spec`: the client
            // performs its HELLO in this constructor, well before the speculative stack
            // exists, and the spec TYPE is known from params from the start.
            //
            // WP_SEGMENT_PROJ_CHECK also reads terminal.nextn (it projects the tail's
            // hidden state on the head and diffs against the tail's own logits), so it
            // has to keep the sideband alive even without draft-mtp -- otherwise the
            // diagnostic would quietly measure nothing.
            const bool segment_proj_check = [] {
                const char * v = std::getenv("WP_SEGMENT_PROJ_CHECK");
                return v != nullptr && v[0] == '1';
            }();
            segment_client = std::make_unique<pipe_dense_segment_client::client>(
                manifest, (uint32_t) llama_vocab_n_tokens(vocab), spec_mtp || segment_proj_check);
            if (segment_client->has_remote_segments()) {
                // Logits-on-head: under PIPE_SEGMENT_TERMINAL_HIDDEN the terminal
                // response is the post-output_norm hidden state (n_embd), and this
                // server finishes the LM head locally with llama_output_project().
                // Under the legacy LOGITS arm it is n_vocab and is memcpy'd
                // straight into the logits buffer as before.
                const bool terminal_hidden =
                    segment_client->terminal_kind() == PIPE_SEGMENT_TERMINAL_HIDDEN;
                const uint32_t expect_width = terminal_hidden
                    ? (uint32_t) llama_model_n_embd(model_tgt)
                    : (uint32_t) llama_vocab_n_tokens(vocab);
                if (segment_client->terminal_width() != expect_width) {
                    SRV_ERR("dense segment terminal width %u does not match the expected %u\n",
                            segment_client->terminal_width(), expect_width);
                    return false;
                }
                // Fail here, not on the first token: under logits-on-head the head
                // stage GGUF MUST carry output.weight, because nothing else in the
                // pipeline computes the projection any more.
                if (terminal_hidden && model_tgt->output == nullptr) {
                    SRV_ERR("%s", "logits-on-head requires output.weight in the head stage; "
                                  "rerun with WP_SEGMENT_TAIL_LOGITS=1 on head and tail\n");
                    return false;
                }
                SRV_INF("dense segment terminal payload = %s (%u f32/token)\n",
                        terminal_hidden ? "HIDDEN (logits-on-head)" : "LOGITS", expect_width);
                // Make the negotiated sideband visible at startup: this is the number
                // the bytes-on-wire measurement should be read against.
                SRV_INF("dense segment nextn sideband = %s (%u f32/token)\n",
                        segment_client->nextn_width() != 0 ? "NEEDED" : "not needed",
                        segment_client->nextn_width());
            }
            // INTERIOR TAPS. Arm the head's host buffers for every layer a remote
            // segment will forward, BEFORE the speculative stack is constructed at the
            // bottom of load_model: the draft's band check consults these to tell
            // "outside my band, impossible" from "outside my band, supplied by a peer".
            //
            // Arming is driven by the manifest rather than by the draft's
            // target_layers because the segment client performs its HELLO while
            // constructing, above, and the draft model is not loaded until later --
            // there is no point at which the draft's requirements are known early
            // enough to negotiate. The two are cross-checked in the draft instead, so a
            // manifest that under-declares fails loudly there.
            for (size_t i = 1; i < segment_client->manifest().segments.size(); ++i) {
                for (const uint32_t lid : segment_client->manifest().segments[i].tap_layers) {
                    llama_set_embeddings_layer_inp_external(ctx_tgt, lid, true);
                    SRV_INF("dense segment %u forwards interior tap for layer %u\n",
                            segment_client->manifest().segments[i].id, lid);
                }
            }

            segment_client->reset(segment_session_id, segment_cache_epoch);
            if (spec_mtp && segment_client->has_remote_segments()) {
                const uint32_t depth = segment_client->recurrent_snapshots();
                const uint32_t requested = (uint32_t) std::max(0, params_base.speculative.draft.n_max);
                if (depth == 0) {
                    SRV_ERR("%s", "dense segments report no recurrent snapshots for draft-mtp\n");
                    return false;
                }
                if (requested > depth) {
                    SRV_WRN("dense segments have %u recurrent snapshots; clamping --spec-draft-n from %u\n",
                            depth, requested);
                    params_base.speculative.draft.n_max = (int32_t) depth;
                }
            }
        }

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                // NOTE(fork): the 2026-07-31 upstream sync moved draft/MTP context
                // creation into common_speculative_init_from_params, so these two fork
                // behaviours now have to be expressed on params_dft rather than on a
                // hand-built cparams_mtp.

                // Optional per-MTP context-size cap via --spec-draft-n-ctx.
                // 0 = inherit target ctx (default). Capped at target ctx.
                if (params_base.speculative.draft.n_ctx > 0) {
                    const uint32_t tgt_n_ctx   = llama_n_ctx_seq(ctx_tgt);
                    const int32_t  draft_n_ctx = params_base.speculative.draft.n_ctx;
                    params_dft.n_ctx = (int32_t) (((uint32_t) draft_n_ctx < tgt_n_ctx)
                        ? (uint32_t) draft_n_ctx
                        : tgt_n_ctx);
                }

                // Disable tier/paged KV on the MTP draft context — it's a small
                // re-eval context with no working-set pressure, and at full
                // ctx_size the paged block pool would try to allocate enormous
                // amounts on the MTP-only layer. Tier features stay on for ctx_tgt.
                // paged_blocks_explicit suppresses the MAD-134 auto-default so this
                // really is off rather than re-enabled by the heuristic.
                params_dft.kv_tiered_enabled             = false;
                params_dft.kv_tier_paged_blocks          = false;
                params_dft.kv_tier_paged_blocks_explicit = true;
                params_dft.kv_tier_total_ctx             = 0;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);

            // MAD-LAB DS4-Flash pipeline-streams (item 2): stream B's own
            // draft/MTP context, bound to ctx_tgt2. Straight mirror of the
            // block above, with every ctx_tgt below replaced by ctx_tgt2 and
            // every *_dft replaced by *_dft2, so stream B's draft state can
            // never touch stream A's target context or KV. Gated separately
            // from `has_spec` so a pipeline-streams==1 run (or a streams>=2
            // run with segment_manifest set, which already refused to start
            // above) takes exactly the code path it always has.
            if (params_base.n_pipeline_streams >= 2 && ctx_tgt2 != nullptr) {
                common_params params_dft2 = common_base_params_to_speculative(params_base);

                params_dft2.load_progress_callback           = load_progress_callback;
                params_dft2.load_progress_callback_user_data = &load_progress_spec;

                if (params_base.speculative.draft.n_ctx > 0) {
                    const uint32_t tgt_n_ctx   = llama_n_ctx_seq(ctx_tgt2);
                    const int32_t  draft_n_ctx = params_base.speculative.draft.n_ctx;
                    params_dft2.n_ctx = (int32_t) (((uint32_t) draft_n_ctx < tgt_n_ctx)
                        ? (uint32_t) draft_n_ctx
                        : tgt_n_ctx);
                }

                params_dft2.kv_tiered_enabled             = false;
                params_dft2.kv_tier_paged_blocks          = false;
                params_dft2.kv_tier_paged_blocks_explicit = true;
                params_dft2.kv_tier_total_ctx             = 0;

                spec_init2 = common_speculative_init_from_params(params_dft2, model_tgt, ctx_tgt2);
                model_dft2 = spec_init2->model();
                ctx_dft2   = spec_init2->context();

                if (has_draft && model_dft2 == nullptr) {
                    SRV_ERR("%s", "pipeline-streams: failed to load stream B's draft model\n");
                    return false;
                }

                if (ctx_dft2 == nullptr) {
                    SRV_ERR("%s", "pipeline-streams: failed to create stream B's MTP/draft context\n");
                    return false;
                }

                if (has_draft) {
                    SRV_WRN("%s", "pipeline-streams: stream B loaded its OWN copy of "
                                  "the external draft model (--model-draft) -- this "
                                  "doubles draft-model VRAM; MTP/DSpark-self speculative "
                                  "does not hit this path (no separate draft weights)\n");
                }

                SRV_INF("%s", "pipeline-streams: stream B draft/MTP context constructed, "
                              "bound to ctx_tgt2\n");
            }
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                if (params_base.kv_tier_paged_blocks) {
                    // MAD-128: paged-blocks deliberately doesn't support
                    // position-shift in-place (would require re-indexing
                    // every block_table entry + per-block reindex of the
                    // GPU K/V layout). Instead: when a slot hits its
                    // context limit, the server stops it with
                    // STOP_TYPE_LIMIT and the caller resubmits as a fresh
                    // request — paged's prompt-cache + semantic prefetch
                    // recover most of the prefix on the next prefill.
                    SRV_WRN("%s\n", "ctx_shift disabled: --kv-tier-paged-blocks "
                            "is set; paged attention manages context via "
                            "tier movement, not in-place shift. Slots will "
                            "stop at n_ctx; clients should re-submit fresh "
                            "requests (prompt cache + semantic prefetch "
                            "will recover the prefix). Plan capacity "
                            "accordingly.");
                } else {
                    SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
                }
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        // The kv_tier_total_ctx override exists ONLY to undo the legacy non-paged tiered
        // path's pre-shrink of params_base.n_ctx (see the hot-tier branch above, which does
        // n_ctx = n_ctx * hot_pct/100). There llama_n_ctx_seq() reports the shrunken hot-tier
        // size and the slot must be told the full logical window instead.
        //
        // The PAGED-blocks path deliberately does NOT pre-shrink -- the model already sees
        // the full n_ctx and the paged cache moves blocks between tiers underneath it. So
        // llama_n_ctx_seq() is already correct there, and applying the override anyway
        // replaced the per-slot share (n_ctx / n_parallel) with the UNDIVIDED total, i.e.
        // every slot got the whole context and the real KV allocation was n_parallel times
        // what --ctx-size asked for. Stock llama.cpp semantics are that --ctx-size is the
        // total, divided among --parallel slots; keep them.
        const bool tiered_ctx_override = params_base.kv_tiered_enabled
                                      && params_base.kv_tier_total_ctx > 0
                                      && !params_base.kv_tier_paged_blocks;

        int n_ctx_slot = tiered_ctx_override
                         ? params_base.kv_tier_total_ctx
                         : llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - using rope scaling to extend\n", n_ctx_slot, n_ctx_train);
            // Do not cap: caller has configured rope scaling (--rope-scale / --rope-scaling yarn) to handle extended context.
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        // MAD-LAB DS4-Flash pipeline-streams (item 3): build spec2 -- the
        // stream-B mirror of spec above -- now that ctx_tgt2/ctx_dft2 exist
        // (built earlier in this function, gated on
        // params_base.n_pipeline_streams >= 2). common_speculative_init()
        // reads its target/draft contexts off the common_params_speculative
        // it is given (params.draft.ctx_tgt / params.draft.ctx_dft), so a
        // shallow copy of params_base.speculative with those two fields
        // overridden to ctx_tgt2/ctx_dft2 is enough to get a fully
        // independent instance -- it does NOT share any state with `spec`.
        common_context_seq_rm_type ctx_tgt2_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        common_context_seq_rm_type ctx_dft2_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        if (ctx_tgt2 != nullptr) {
            ctx_tgt2_seq_rm_type = common_context_can_seq_rm(ctx_tgt2);
            if (ctx_tgt2_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                common_params_speculative speculative_b = params_base.speculative;
                speculative_b.draft.ctx_tgt = ctx_tgt2;
                speculative_b.draft.ctx_dft = ctx_dft2;
                try {
                    // n_seq sized for however many slots stream B ends up
                    // with -- computed the same way the slot split below
                    // computes n_slots_b, kept in sync deliberately (both
                    // read params_base.n_parallel and n_pipeline_streams).
                    const int n_slots_b = params_base.n_parallel / params_base.n_pipeline_streams;
                    spec2.reset(common_speculative_init(speculative_b, std::max(1, n_slots_b)));
                } catch (const std::exception & e) {
                    SRV_ERR("pipeline-streams: failed to initialize stream B speculative decoding: %s\n", e.what());
                }
            }
            if (ctx_dft2) {
                ctx_dft2_seq_rm_type = common_context_can_seq_rm(ctx_dft2);
            }
            if (!spec2) {
                spec_init2.reset();
                ctx_dft2   = nullptr;
                model_dft2 = nullptr;
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        // MAD-LAB DS4-Flash pipeline-streams (item 3): assign slots to
        // streams. Stream B gets slots [n_slots_a, n_parallel), stream A
        // gets [0, n_slots_a) -- a contiguous split rather than interleaved
        // parity so slot-id-based logging/debugging stays easy to read
        // ("slots 0..N/2-1 are stream A"). At n_pipeline_streams==1,
        // n_slots_a == n_parallel and every slot is stream A, unchanged
        // from today.
        const int n_slots_b = (params_base.n_pipeline_streams >= 2)
            ? params_base.n_parallel / params_base.n_pipeline_streams
            : 0;
        const int n_slots_a = params_base.n_parallel - n_slots_b;

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            const bool is_stream_b = i >= n_slots_a && ctx_tgt2 != nullptr;

            slot.id      = i;
            // MAD-LAB DS4-Flash pipeline-streams: stream-local index, dense
            // from 0 within EACH stream -- stream A is slots [0, n_slots_a)
            // so stream_slot_idx == id there (that equality is exactly what
            // let a missing conversion go undetected for stream A); stream
            // B is slots [n_slots_a, n_parallel), so its stream_slot_idx is
            // id - n_slots_a, i.e. also dense from 0.
            slot.stream_slot_idx = is_stream_b ? (i - n_slots_a) : i;
            slot.ctx_tgt = is_stream_b ? ctx_tgt2 : ctx_tgt;
            slot.ctx_dft = is_stream_b ? ctx_dft2 : ctx_dft;
            slot.mem.init(slot.ctx_tgt, slot.ctx_dft);
            slot.spec    = is_stream_b ? spec2.get() : spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d, stream = %s\n", slot.n_ctx, is_stream_b ? "B" : "A");

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        if (params_base.n_pipeline_streams >= 2) {
            SRV_INF("pipeline-streams: %d slots on stream A, %d slots on stream B "
                    "(ctx_tgt2_seq_rm=%d, ctx_dft2_seq_rm=%d)\n",
                    n_slots_a, n_slots_b, (int) ctx_tgt2_seq_rm_type, (int) ctx_dft2_seq_rm_type);
            // MAD-LAB DS4-Flash pipeline-streams: the hard refusal that used
            // to be here is gone -- update_slots()/decode()/post_decode()
            // now route every slot's decode through its OWN stream's
            // context end to end (server_stream, run_stream_decode_loop(),
            // the transitive-helper audit converting try_clear_idle_slots/
            // launch_slot_with_task/process_token/populate_token_probs/
            // send_final_response/send_rerank/drain_paged_fingerprints to
            // slot.*/stream.* reads). stream_b + batch_b + stream_b_thread_
            // are wired up just below.
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * WP_SPEC_PHASE = getenv("WP_SPEC_PHASE");
            spec_phase = WP_SPEC_PHASE ? atoi(WP_SPEC_PHASE) : 0;

            if (spec_phase) {
                SRV_WRN("WP_SPEC_PHASE = %d\n", spec_phase);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, n_slots_a), n_embd);
        }

        // MAD-LAB DS4-Flash pipeline-streams (stage 2): populate stream_a
        // with just its own slots -- [0, n_slots_a), not "every slot"
        // anymore now that a stream B can exist.
        {
            stream_a.ctx_tgt   = ctx_tgt;
            stream_a.ctx_dft   = ctx_dft;
            stream_a.model_dft = model_dft;
            stream_a.spec      = spec.get();
            stream_a.ctx_tgt_seq_rm_type = ctx_tgt_seq_rm_type;
            stream_a.ctx_dft_seq_rm_type = ctx_dft_seq_rm_type;
            stream_a.batch     = &batch;
            stream_a.slots.clear();
            stream_a.slots.reserve((size_t) n_slots_a);
            for (int i = 0; i < n_slots_a; i++) {
                stream_a.slots.push_back(&slots[i]);
            }
            stream_a.paged_admit_rotor    = 0;
            stream_a.n_empty_consecutive  = 0;
        }

        // MAD-LAB DS4-Flash pipeline-streams (stage 2): batch_b + stream_b +
        // stream_b_thread_, only when n_pipeline_streams >= 2. batch_b is
        // init()-sized the same way `batch` is above, just against
        // n_slots_b instead of n_slots_a. stream_b_thread_ is spawned here
        // (once, for the lifetime of this init()) rather than per-tick --
        // see stream_b_thread_main()/run_stream_decode_loop() for its body
        // and update_slots() for the per-tick dispatch/join protocol.
        if (params_base.n_pipeline_streams >= 2) {
            const int32_t n_batch = llama_n_batch(ctx_tgt2);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch_b.init(std::max(n_batch, n_slots_b), n_embd);

            stream_b.ctx_tgt   = ctx_tgt2;
            stream_b.ctx_dft   = ctx_dft2;
            stream_b.model_dft = model_dft2;
            stream_b.spec      = spec2.get();
            stream_b.ctx_tgt_seq_rm_type = ctx_tgt2_seq_rm_type;
            stream_b.ctx_dft_seq_rm_type = ctx_dft2_seq_rm_type;
            stream_b.batch     = &batch_b;
            stream_b.slots.clear();
            stream_b.slots.reserve((size_t) n_slots_b);
            for (int i = n_slots_a; i < params_base.n_parallel; i++) {
                stream_b.slots.push_back(&slots[i]);
            }
            stream_b.paged_admit_rotor    = 0;
            stream_b.n_empty_consecutive  = 0;

            stream_b_thread_stop_.store(false);
            {
                std::lock_guard<std::mutex> lock(stream_b_mtx_);
                stream_b_has_work_ = false;
                stream_b_done_ = true;
            }
            stream_b_thread_ = std::thread([this] { stream_b_thread_main(); });
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    // MAD-LAB DS4-Flash pipeline-streams (stage 2): scoped to stream.slots
    // -- purging a slot only helps if it frees KV space in the STREAM
    // that just failed to decode; a stream-B slot's KV lives in ctx_tgt2,
    // which does nothing for a stream-A retry (and vice versa) now that
    // the two streams have separate paged-KV pools.
    bool try_clear_idle_slots(server_stream & stream) {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto * slot_ptr : stream.slots) {
            auto & slot = *slot_ptr;
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(slot.ctx_tgt)) { // MAD-LAB: this slot's own stream's context
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(slot.ctx_tgt, slot.stream_slot_idx, common_sampler_get(slot.smpl.get())); // MAD-LAB: stream-local seq id
            } else {
                llama_set_sampler(slot.ctx_tgt, slot.stream_slot_idx, nullptr); // MAD-LAB: stream-local seq id
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            {
                const auto * rbudget  = common_sampler_get_rbudget(slot.smpl.get());
                const auto   rb_state = common_reasoning_budget_get_state(rbudget);
                SLT_INF(slot, "reasoning budget: %s initial_state=%d budget_tokens=%d\n",
                        rbudget ? "active" : "inactive", (int)rb_state,
                        task.params.sampling.reasoning_budget_tokens);
            }
        } else {
            slot.smpl.reset();
        }

        ++slot.fp_epoch;
        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        // MAD-LAB DS4-Flash pipeline-streams (stage 2): resets THIS slot's
        // own stream's counter, not always stream_a.
        stream_for_slot(slot).n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        SLT_DBG(slot, "token: id=%d '%s' rbudget_state=%d\n",
                (int)result.tok, token_str.c_str(),
                (int)common_reasoning_budget_get_state(common_sampler_get_rbudget(slot.smpl.get())));

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // Proactive tiered cache eviction: when hot tier reaches 80% capacity,
        // back up oldest tokens to warm/cold so the read side can restore
        // them later. Works for all architectures including hybrids that
        // auto-disable ctx_shift (Qwen3.6, DeepSeek V4) — for those, this
        // is the ONLY path that populates mt::'s warm + cold tiers
        // (otherwise the seq_rm-time backup hook never fires).
        //
        // **Threshold capacity**: use the physical attention cache cell
        // count, NOT slot.n_ctx. For hybrid models (Qwen3.6, DeepSeek V4)
        // the attention KV cache is sized for a sliding window — much
        // smaller than the user-facing context (recurrent layers carry
        // the long context). Comparing against slot.n_ctx makes the
        // trigger fire too late and the inner cache 500s with "failed
        // to find free space in the KV cache" before we ever get a
        // chance to evict.
        if (params_base.kv_tiered_enabled) {
            const int n_tokens = slot.prompt.n_tokens();

            auto * mt_tier = dynamic_cast<mt::llama_memory_tiered *>(llama_get_memory(slot.ctx_tgt)); // MAD-LAB: this slot's own stream's context
            uint32_t cap = (uint32_t)slot.n_ctx;
            if (mt_tier) {
                const uint32_t phys = mt_tier->physical_attn_cells();
                if (phys > 0) cap = phys;
            }

            const int evict_threshold = (int)(cap * 0.80f);

            // If the slot was truncated (new task with no shared prefix
            // wipes the inner cache via memory_seq_rm), kv_evict_through
            // is stale — it points past the now-much-smaller live cache.
            // Reset to start eviction from scratch on this fresh slot.
            // Self-heals without having to instrument every memory_seq_rm
            // site in the server.
            if ((int)slot.kv_evict_through > n_tokens) {
                slot.kv_evict_through = 0;
            }

            // Live-in-hot count = total positions minus what we've already
            // backed up. Without subtracting, the trigger keeps firing on
            // every step once n_tokens passes the threshold but does nothing
            // (the [0, n_evict) range is already in warm) and the inner
            // cache overflows on long generations.
            const int n_live_hot = n_tokens - (int)slot.kv_evict_through;
            if (n_live_hot >= evict_threshold) {
                const int n_evict = std::max(1, (int)(cap * 0.20f));

                if (mt_tier) {
                    // Eviction window starts at the cursor — the next
                    // not-yet-backed-up positions. Cursor advances by the
                    // RANGE WIDTH on success, NOT the count of positions
                    // returned. For the position-keyed path these match;
                    // for the paged-blocks path they don't (block-aligned
                    // backups can return fewer positions than the
                    // requested range when n_evict isn't a multiple of
                    // block_size). Advancing by range width keeps
                    // subsequent triggers from re-asking for partial
                    // ranges that already got a block-rounded sweep.
                    // (If backup returns 0, we DO NOT advance — that
                    // means total failure and we want the next trigger
                    // to retry the same range.)
                    const llama_pos p0 = slot.kv_evict_through;
                    const llama_pos p1 = p0 + (llama_pos)n_evict;
                    const uint32_t backed_up = mt_tier->backup_proactive(
                        slot.stream_slot_idx, p0, p1); // MAD-LAB: stream-local seq id
                    if (backed_up > 0) {
                        // Semantic fingerprint: embed the detokenized chunk
                        // so a future query (likely a separate request with
                        // no shared prefix) can find this chunk via cosine
                        // similarity and prefetch it back to hot before
                        // attention runs. Mirrors the context-shift-time
                        // fingerprinting; only fires when --kv-tier-semantic-index
                        // is set and the chunk isn't multimodal.
                        //
                        // MAD-122: under --kv-tier-paged-blocks the helper
                        // emits one fingerprint per logical block instead of
                        // one for the whole chunk, so query-time prefetch can
                        // score at block granularity.
                        if (!params_base.kv_semantic_index.empty() && !slot.prompt.tokens.has_mtmd) {
                            const auto & toks = slot.prompt.tokens.get_text_tokens();
                            llama_kv_cache_paged * paged_cache = params_base.kv_tier_paged_blocks
                                ? mt_get_paged_cache(llama_get_memory(slot.ctx_tgt)) : nullptr; // MAD-LAB
                            const int n_fp = mt_record_fingerprints_for_range(
                                mt_tier, paged_cache, slot.ctx_tgt, slot.stream_slot_idx, toks, p0, p1, // MAD-LAB: stream-local seq id
                                (uint32_t) params_base.kv_tier_paged_block_size);
                            if (n_fp > 0) {
                                SLT_INF(slot, "tier semantic: %d %s fingerprint(s) [%d,%d) for proactive backup\n",
                                        n_fp,
                                        paged_cache ? "paged-block" : "chunk",
                                        p0, p1);
                            }
                        }
                        // Advance to the requested range end, not the count
                        // of positions actually backed up. See the comment
                        // above this branch — block-aligned paged backups
                        // can return fewer positions than requested when
                        // n_evict isn't a clean multiple of block_size.
                        slot.kv_evict_through = p1;
                        SLT_DBG(slot, "proactive mt:: backup: %u/%d positions [%d,%d) at %d live / %u hot capacity\\n",
                                backed_up, n_evict, p0, p1, n_live_hot, cap);
                    }
                }
            }
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            // MAD-128: clear log when paged hits the limit — the operator
            // needs to know this is a "resubmit and rely on prompt cache"
            // situation, not a hard failure. SLT_INF (not DBG) so it lands
            // in the default log level.
            if (params_base.kv_tier_paged_blocks) {
                SLT_INF(slot, "paged: hit n_ctx limit (n_tokens=%d, n_ctx=%d) — "
                        "stopping slot. Client should re-submit as a fresh "
                        "request; the prompt cache + semantic prefetch will "
                        "recover the prefix.\n",
                        slot.prompt.n_tokens(), slot.n_ctx);
            } else {
                SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                        slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
            }
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(slot.ctx_tgt, cur_p->data[i].id, special), // MAD-LAB
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(slot.ctx_tgt, idx, n_probs_request); // MAD-LAB
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(slot.ctx_tgt, cur[i].id, special), // MAD-LAB
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(slot.ctx_tgt, true); // MAD-LAB: this slot's own stream context
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(slot.ctx_tgt, slot.stopping_word, false); // MAD-LAB

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.stream_slot_idx) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.stream_slot_idx) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]); // MAD-LAB
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i); // MAD-LAB
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        // MAD-LAB DS4-Flash pipeline-streams: create_checkpoint() isn't
        // parameterized by server_stream (it doesn't need to be) -- slot.
        // ctx_tgt/ctx_dft/spec are already this slot's own stream's context
        // (wired at slot construction in init(), see the earlier
        // pipeline-streams slot-split work), which for any slot in
        // stream.slots is by construction the same pointer as
        // stream.ctx_tgt/ctx_dft/spec. Using the per-slot fields here avoids
        // adding a parameter to a function whose only per-tick input is
        // already the slot.
        cur.update_tgt(slot.ctx_tgt, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); // MAD-LAB: stream-local seq id
        cur.update_dft(slot.ctx_dft, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY); // MAD-LAB: stream-local seq id
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(slot.spec, slot.stream_slot_idx, cur.data_spec); // MAD-LAB: stream-local seq id (spec's dparams sized per-stream)

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

            // MAD-133: Add paged-tier metrics when --kv-tier-paged-blocks
            // is on. Reads counters directly from the live cache (single-
            // thread contract — main thread is the only mutator; metrics
            // endpoint runs on the HTTP thread but only READS volatile
            // uint64s, which is safe-ish on x86/ARM64 for monotonic counters).
            if (ctx_tgt) {
                llama_kv_cache_paged * paged_cache = mt_get_paged_cache(llama_get_memory(ctx_tgt));
                if (paged_cache) {
                    auto add_counter = [&](const char * name, const char * help, uint64_t value) {
                        res->extra_counters.push_back({ name, help, (double) value });
                    };
                    auto add_gauge = [&](const char * name, const char * help, uint64_t value) {
                        res->extra_gauges.push_back({ name, help, (double) value });
                    };

                    add_counter("paged_evict_hot_to_warm_total",   "Hot→warm evictions",            paged_cache->evict_h2w_total());
                    add_counter("paged_evict_warm_to_cold_total",  "Warm→cold evictions",           paged_cache->evict_w2c_total());
                    add_counter("paged_evict_cold_to_drop_total",  "Cold-block drops (no recovery)", paged_cache->evict_c2drop_total());
                    add_counter("paged_restore_warm_to_hot_total", "Warm→hot restores",             paged_cache->restore_w2h_total());
                    add_counter("paged_restore_cold_to_hot_total", "Cold→hot restores",             paged_cache->restore_c2h_total());
                    add_counter("paged_seq_preempt_total",         "MAD-120 whole-seq preemptions", paged_cache->seq_preempt_total());
                    add_counter("paged_seq_restore_total",         "MAD-120 whole-seq restores",    paged_cache->seq_restore_total());

                    add_counter("paged_semantic_attempts_total",        "MAD-129 semantic restore attempts",         paged_cache->semantic_attempts_total());
                    add_counter("paged_semantic_hits_total",            "MAD-129 semantic restore attempts that restored ≥1 block", paged_cache->semantic_hits_total());
                    add_counter("paged_semantic_blocks_restored_total", "MAD-129 total blocks restored via semantic", paged_cache->semantic_blocks_restored_total());

                    add_gauge("paged_blocks_capacity_gpu",   "GPU pool size (blocks)",   paged_cache->n_blocks_total());
                    add_gauge("paged_blocks_capacity_warm",  "Warm pool size (blocks)",  paged_cache->n_warm_blocks());
                    add_gauge("paged_blocks_capacity_cold",  "Cold pool size (blocks)",  paged_cache->n_cold_blocks());
                    add_gauge("paged_fingerprints",          "MAD-129 paged-block fingerprints currently held", paged_cache->n_paged_fingerprints());
                }
            }

            // Add weight pager metrics if enabled
            if (model_tgt && model_tgt->wp_pager) {
                // Multi-device paging: these gauges describe the PROCESS, so sum across
                // every pager. Reporting primary() alone would compile and quietly
                // under-report whenever more than one device is paging.
                auto * pagers = model_tgt->wp_pager.get();
                double wp_n_pages = 0, wp_loaded = 0, wp_pending = 0, wp_async = 0;
                double wp_page_ins = 0, wp_evictions = 0, wp_prefetch_hits = 0;
                double wp_prefetch_misses = 0, wp_sync_fallbacks = 0, wp_io_bytes = 0;
                double wp_io_seconds = 0, wp_lru_hot = 0, wp_lru_pinned = 0;
                double wp_dense_prefetch = 0, wp_xlayer_sub = 0, wp_xlayer_hit = 0;
                double wp_rp_set = 0, wp_rp_consumed = 0, wp_rp_discarded = 0;
                for (const auto & wp_entry : pagers->entries()) {
                    const auto * wp_p = wp_entry.pager.get();
                    if (wp_p == nullptr) {
                        continue;
                    }
                    const auto & s = wp_p->stats();
                    wp_n_pages          += (double) wp_p->n_pages();
                    wp_loaded           += (double) wp_p->loaded_pages();
                    wp_pending          += (double) wp_p->pending_prefetches();
                    wp_async             = wp_p->async_prefetch_enabled() ? 1.0 : wp_async;
                    wp_page_ins         += (double) s.page_ins;
                    wp_evictions        += (double) s.evictions;
                    wp_prefetch_hits    += (double) s.prefetch_hits;
                    wp_prefetch_misses  += (double) s.prefetch_misses;
                    wp_sync_fallbacks   += (double) s.sync_fallbacks;
                    wp_io_bytes         += (double) s.io_bytes;
                    wp_io_seconds       += s.io_seconds;
                    wp_lru_hot          += (double) s.lru_walk_hot_skips;
                    wp_lru_pinned       += (double) s.lru_walk_pinned_skips;
                    wp_dense_prefetch   += (double) s.dense_prefetch_submitted;
                    wp_xlayer_sub       += (double) s.cross_layer_prefetch_submitted;
                    wp_xlayer_hit       += (double) s.cross_layer_hit_in_ensure;
                    wp_rp_set           += (double) s.routing_ptrs_set;
                    wp_rp_consumed      += (double) s.routing_ptrs_consumed;
                    wp_rp_discarded     += (double) s.routing_ptrs_discarded_unconsumed;
                }
                const double io_gb = wp_io_bytes / 1000000000.0;
                // Summed device-seconds: with concurrent pagers this is aggregate I/O
                // time, not wall time, so the derived rate is per-device-second.
                const double io_gbps = wp_io_seconds > 0.0 ? io_gb / wp_io_seconds : 0.0;
                auto add_wp_gauge = [&](const char * name, const char * help, double value) {
                    res->extra_gauges.push_back({ name, help, value });
                };

                add_wp_gauge("llama_weight_pager_pages_total", "Total number of weight pages tracked", wp_n_pages);
                add_wp_gauge("llama_weight_pager_loaded_pages", "Number of pages currently loaded in VRAM", wp_loaded);
                add_wp_gauge("llama_weight_pager_in_flight_prefetches", "Number of in-flight prefetch requests", wp_pending);
                add_wp_gauge("llama_weight_pager_async_prefetch_enabled", "Async prefetch enabled (1=true, 0=false)", wp_async);
                add_wp_gauge("llama_weight_pager_page_ins_total", "Total weight pages read into VRAM", wp_page_ins);
                add_wp_gauge("llama_weight_pager_evictions_total", "Total weight pager pool evictions", wp_evictions);
                add_wp_gauge("llama_weight_pager_prefetch_hits_total", "Total ensure calls where prefetch was already complete", wp_prefetch_hits);
                add_wp_gauge("llama_weight_pager_prefetch_misses_total", "Total ensure calls where prefetch was missing or incomplete", wp_prefetch_misses);
                add_wp_gauge("llama_weight_pager_sync_fallbacks_total", "Total ensure calls that used synchronous page-in fallback", wp_sync_fallbacks);
                add_wp_gauge("llama_weight_pager_io_bytes_total", "Total weight pager bytes read", wp_io_bytes);
                add_wp_gauge("llama_weight_pager_io_seconds_total", "Total measured weight pager IO seconds", wp_io_seconds);
                add_wp_gauge("llama_weight_pager_io_effective_gb_s", "Effective weight pager read bandwidth in GB/s", io_gbps);
                add_wp_gauge("llama_weight_pager_lru_walk_hot_skips_total", "Total LRU walk skips of hot slots", wp_lru_hot);
                add_wp_gauge("llama_weight_pager_lru_walk_pinned_skips_total", "Total LRU walk skips of pinned slots", wp_lru_pinned);
                if (wp_dense_prefetch > 0) {
                    add_wp_gauge("llama_weight_pager_dense_prefetch_submitted_total", "Total successful dense forward-prefetch submissions", wp_dense_prefetch);
                }
                add_wp_gauge("llama_weight_pager_cross_layer_prefetch_submitted_total", "Total successful cross-layer prefetch submissions", wp_xlayer_sub);
                add_wp_gauge("llama_weight_pager_cross_layer_hit_in_ensure_total", "Total ensure-time hits from cross-layer prefetch candidates", wp_xlayer_hit);
                add_wp_gauge("llama_weight_pager_routing_ptrs_set_total", "Total routed expert pointer arrays armed", wp_rp_set);
                add_wp_gauge("llama_weight_pager_routing_ptrs_consumed_total", "Total routed expert pointer arrays consumed by MMQ/MMVQ", wp_rp_consumed);
                add_wp_gauge("llama_weight_pager_routing_ptrs_discarded_unconsumed_total", "Total routed expert pointer arrays discarded before MMQ/MMVQ consumed them", wp_rp_discarded);
            }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        slot->ctx_tgt, filepath.c_str(), slot->stream_slot_idx, // MAD-LAB: stream-local seq id, this slot's own stream context
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(slot->ctx_tgt, filepath.c_str(), slot->stream_slot_idx, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(slot->ctx_tgt, filepath.c_str(), slot->stream_slot_idx, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(slot->ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): scoped variant, used
    // from within a per-stream tick (pre_decode()/run_stream_tick()
    // exception handling) so a stream-A failure cannot abort stream-B's
    // in-flight slots (separate contexts, separate failure domains) and
    // vice versa. The unscoped abort_all_slots() above is kept for the
    // few call sites that are genuinely global (e.g. shutdown).
    void abort_all_slots(server_stream & stream, const std::string & reason) {
        for (auto * slot_ptr : stream.slots) {
            auto & slot = *slot_ptr;
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): maps a slot to its
    // stream by comparing slot.ctx_tgt against each stream's ctx_tgt --
    // slot.ctx_tgt was wired to the correct stream's context at slot
    // construction in init() and never changes afterward, so this is a
    // safe, cheap way for code that only has a `server_slot&` (not a
    // `server_stream&`) to find the right stream. Falls back to stream_a
    // if stream_b was never constructed (n_pipeline_streams < 2) or if the
    // slot somehow doesn't match either (should not happen; defensive).
    server_stream & stream_for_slot(server_slot & slot) {
        if (stream_b.ctx_tgt != nullptr && slot.ctx_tgt == stream_b.ctx_tgt) {
            return stream_b;
        }
        return stream_a;
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // PIPELINE-STREAMS (MAD-LAB DS4-Flash, NOT YET WIRED): this function is
    // still single-context / single-thread regardless of
    // params_base.n_pipeline_streams. All slots are pinned to slot.ctx_tgt
    // == ctx_tgt (stream A) at slot construction above; decode() below
    // always calls llama_decode() against the server_context_impl::ctx_tgt
    // MEMBER (not slot.ctx_tgt), and the batch built at the top of this
    // function mixes every ready slot's tokens into ONE llama_batch/
    // llama_decode() call. ctx_tgt2 (stream B, constructed in init() with
    // its own KV cache and its own expert-dispatch connection -- see the
    // comment there) exists and is live, but nothing currently drives it.
    //
    // What full wiring needs, concretely:
    //   1. Split `slots` into a stream-A group and a stream-B group (e.g.
    //      by slot.id parity, or a new slot.stream_id), each with its
    //      slot.ctx_tgt/slot.spec pointing at its own stream's ctx_tgt(2)/
    //      spec(2).
    //   2. Build TWO independent llama_batch views (one per group) instead
    //      of the single `batch` below, and run pre_decode/decode/
    //      post_decode against each group's OWN context.
    //   3. Run stream B's decode() on a second OS thread (per the task's
    //      "smallest honest seam") so its blocking llama_decode() call can
    //      overlap stream A's -- join before this function returns so a
    //      "tick" still completes both streams before the task-queue loop
    //      advances. decode()/post_decode() touch slot state and call
    //      queue_results.send(...); confirm server_response's internal
    //      locking (tools/server/server-*.h, class server_response) is
    //      actually safe for two concurrent senders before relying on it --
    //      it was written for the existing single-decode-thread model
    //      where callers are otherwise serialized.
    //   4. Everything downstream of a batch/decode (post_decode, sampling,
    //      speculative harvest at ~:4990-ish) already reads ctx_tgt off the
    //      slot or off `batch_view` sizes, so it should generalize once (1)
    //      and (2) are in place; audit rather than assume.
    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        drain_paged_fingerprints(ctx_tgt);
        if (stream_b.ctx_tgt != nullptr) {
            drain_paged_fingerprints(stream_b.ctx_tgt);
        }

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        // MAD-LAB DS4-Flash pipeline-streams (stage 2) PIPELINE-STREAMS
        // threading model:
        //   1. pre_decode() + render() run SEQUENTIALLY on this (the main/
        //      task-queue) thread for BOTH streams, one after the other,
        //      before either stream's decode starts. This is deliberate,
        //      not a missed overlap opportunity: pre_decode() is where all
        //      the KV-tier/checkpoint/cache-reuse/paged-admission state
        //      mutation happens, and stream_a/stream_b touch fully
        //      disjoint state there (separate rotor, separate slots,
        //      separate batch) so nothing stops them running concurrently
        //      too -- but that state is also the least-audited part of
        //      this refactor, so the first cut keeps it single-threaded
        //      and puts the persistent thread ONLY around the part that
        //      actually blocks on the GPU/dispatcher (decode()) and the
        //      per-token sampling after it (post_decode()), which is where
        //      the real wall-clock overlap this design exists for comes
        //      from.
        //   2. run_stream_decode_loop(stream_b) is dispatched to
        //      stream_b_thread_ ONLY if pre_decode(stream_b) actually
        //      produced work (batch_b.size() > 0) -- idle-skip: the thread
        //      is never woken for an empty tick, so it never holds
        //      anything a busy stream A needs, and the "join" below is a
        //      no-op wait on a condition already satisfied.
        //   3. This thread then runs run_stream_decode_loop(stream_a)
        //      directly -- this is where stream A's llama_decode()/GPU
        //      submission can overlap stream B's (running concurrently on
        //      stream_b_thread_).
        //   4. join_stream_b_tick() blocks until stream B's dispatched tick
        //      (if any) completes, before this function returns.
        // This is a PER-TICK JOIN (first cut), not fully independent
        // cross-tick ticking: the surrounding harness (server_queue::
        // start_loop(), on which see the earlier per-round audit) is built
        // around exactly one update_slots() call representing "this tick,
        // all slots," and fully decoupling stream B onto its own cadence
        // would mean splitting start_loop() itself into two independent
        // task-admission/dispatch loops -- a materially bigger change than
        // giving stream B's blocking llama_decode() its own thread. If a
        // live gate shows the per-tick join is costing real overlap (stream
        // A regularly finishing well before stream B, or vice versa, and
        // waiting idle), that cross-tick decoupling is the known next step.
        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode(stream_a);
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots(stream_a, "pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        bool stream_b_dispatched = false;
        if (stream_b.ctx_tgt != nullptr) {
            try {
                scoped_timer t(t_pre_decode, n_pre_decode);
                pre_decode(stream_b);
                batch_b.render();
            } catch (const std::exception & e) {
                SRV_ERR("pre_decode() (stream B) failed: %s\n", e.what());
                abort_all_slots(stream_b, "pre_decode() failed: " + std::string(e.what()));
            }

            if (batch_b.size() > 0) {
                stream_b_dispatched = true;
                std::lock_guard<std::mutex> lock(stream_b_mtx_);
                stream_b_has_work_ = true;
                stream_b_done_ = false;
                stream_b_cv_.notify_one();
            }
        }

        run_stream_decode_loop(stream_a);

        if (stream_b_dispatched) {
            std::unique_lock<std::mutex> lock(stream_b_mtx_);
            stream_b_done_cv_.wait(lock, [this] { return stream_b_done_; });
        }

        // MAD-LAB DS4-Flash pipeline-streams: the n_cmpl>1 parent/child
        // activation walk needs to see every slot regardless of stream, so
        // it runs once here, in the single-threaded part of the tick, after
        // stream A's (and stream B's already-joined) decode/post_decode
        // work for this tick is done -- see activate_parent_child_tasks()
        // for why it isn't parameterized.
        activate_parent_child_tasks();
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): the chunked decode()/
    // post_decode() loop, extracted out of update_slots() so it can run for
    // either stream -- stream A on the calling (main) thread, stream B on
    // stream_b_thread_. Assumes stream.batch has already been rendered by a
    // preceding pre_decode(stream) + stream.batch->render() (done in
    // update_slots(), sequentially for both streams -- see the threading
    // model comment there for why pre_decode is not itself dual-threaded in
    // this stage). Also applies the lora/embeddings-per-batch setup that
    // used to live inline in update_slots(), now per-stream (was reading
    // the impl-level `ctx_tgt`/`batch` members, i.e. always stream A's).
    void run_stream_decode_loop(server_stream & stream) {
        server_batch & sbatch = *stream.batch;

        GGML_ASSERT(sbatch.slot_batched || sbatch.size() == 0);

        if (sbatch.slot_batched) {
            auto & slot_batched      = sbatch.slot_batched;
            auto & alora_scale       = sbatch.alora_scale;
            auto & alora_disabled_id = sbatch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(stream.ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            // MAD-LAB: segment_client is guaranteed null whenever
            // n_pipeline_streams >= 2 (init() refuses --segment-manifest +
            // --pipeline-streams together), so this is dead-but-correct at
            // N=2 and unchanged at N=1.
            llama_set_embeddings(stream.ctx_tgt, slot_batched->need_embd() ||
                (segment_client && segment_client->has_remote_segments()));
        }

        if (segment_client && segment_client->has_remote_segments()) {
            sbatch.set_all_output();
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(stream.ctx_tgt);
        for (int32_t off = 0; off < sbatch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, sbatch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = sbatch.get_view(off, n_tokens);
                bool ok = decode(stream, n_batch, off, batch_view);
                drain_paged_fingerprints(stream.ctx_tgt);
#ifdef DEBUG_TIMINGS
                llama_synchronize(stream.ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(stream.ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots(stream, "decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(stream, n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots(stream, "post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 2): body of stream_b_thread_.
    // Parked on stream_b_cv_ waiting for stream_b_has_work_; runs exactly
    // run_stream_decode_loop(stream_b) (pre_decode/render for stream B
    // already happened on the main thread before this was signaled -- see
    // update_slots()); signals stream_b_done_cv_ when finished. Exits when
    // stream_b_thread_stop_ is set (destroy()).
    void stream_b_thread_main() {
        for (;;) {
            std::unique_lock<std::mutex> lock(stream_b_mtx_);
            stream_b_cv_.wait(lock, [this] {
                return stream_b_thread_stop_.load() || stream_b_has_work_;
            });
            if (stream_b_thread_stop_.load() && !stream_b_has_work_) {
                return;
            }
            stream_b_has_work_ = false;
            lock.unlock();

            run_stream_decode_loop(stream_b);

            lock.lock();
            stream_b_done_ = true;
            lock.unlock();
            stream_b_done_cv_.notify_one();
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 1): pre_decode() reads ONLY
    // through `stream` now, via the shadow locals right below -- no member
    // access to ctx_tgt/ctx_dft/model_dft/spec/ctx_tgt_seq_rm_type/
    // ctx_dft_seq_rm_type/batch/slots/paged_admit_rotor survives past this
    // point in the function; every one of those names is redeclared here to
    // alias the struct instead, so ordinary C++ name lookup routes every
    // existing reference in the ~800-line body below to `stream` with zero
    // further textual changes, EXCEPT the handful the compiler forces
    // because the shadow's type differs from the member's (see the two
    // `slots` fixups below, and the `spec.get()` -> `spec` fixups, both
    // marked MAD-LAB inline). This is a pure parameterization: no branch,
    // no reordering, no new logic -- called once, unconditionally, with
    // stream_a, whether n_pipeline_streams is 1 or >= 2.
    void pre_decode(server_stream & stream) {
        llama_context * const ctx_tgt   = stream.ctx_tgt;
        llama_context * const ctx_dft   = stream.ctx_dft;
        common_speculative * const spec = stream.spec;
        const common_context_seq_rm_type ctx_tgt_seq_rm_type = stream.ctx_tgt_seq_rm_type;
        const common_context_seq_rm_type ctx_dft_seq_rm_type = stream.ctx_dft_seq_rm_type;
        server_batch & batch = *stream.batch;
        uint64_t & paged_admit_rotor = stream.paged_admit_rotor;
        int32_t & n_empty_consecutive = stream.n_empty_consecutive;
        // MAD-LAB: was `std::vector<server_slot> & slots` (the impl
        // member); now this stream's slot list. Type change is
        // deliberate -- iterate(slots, ...) below auto-resolves to the
        // std::vector<server_slot*> overload (same callback signature, zero
        // body change); any bare `for (auto & slot : slots)` would instead
        // fail to compile (slot becomes server_slot*), which is exactly how
        // the two direct-index sites below were found and fixed.
        std::vector<server_slot *> & slots = stream.slots;

        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                // MAD-122/125: capture a semantic fingerprint of the tokens
                // about to be shifted out so a future query can pull the chunk
                // back via restore_semantic. Fires only when
                // --kv-tier-semantic-index is set AND the active memory is the
                // tier wrapper. The wrapper's seq_rm-time K/V backup runs
                // separately on the seq_rm call below; this attaches the
                // fingerprint to that backup.
                if (auto * mt_tier = dynamic_cast<mt::llama_memory_tiered *>(llama_get_memory(ctx_tgt))) {
                    if (!params_base.kv_semantic_index.empty() && !slot.prompt.tokens.has_mtmd && n_keep >= 0) {
                        const auto & toks = slot.prompt.tokens.get_text_tokens();
                        llama_kv_cache_paged * paged_cache = params_base.kv_tier_paged_blocks
                            ? mt_get_paged_cache(llama_get_memory(ctx_tgt)) : nullptr;
                        const int n_fp = mt_record_fingerprints_for_range(
                            mt_tier, paged_cache, ctx_tgt, slot.stream_slot_idx, toks, n_keep, n_keep + n_discard, // MAD-LAB: stream-local seq id
                            (uint32_t) params_base.kv_tier_paged_block_size);
                        if (n_fp > 0) {
                            SLT_INF(slot, "tier semantic: %d %s fingerprint(s) [%d,%d) for context shift\n",
                                    n_fp,
                                    paged_cache ? "paged-block" : "chunk",
                                    n_keep, n_keep + n_discard);
                        }
                    }
                }

                // slot.mem covers both the target and draft contexts (upstream 2026-07-31).
                slot.mem.seq_rm (slot.stream_slot_idx, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.stream_slot_idx, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.prompt.checkpoints.clear();
                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        // MAD-120 Phase 2: paged-attn admission control. Track which seq
        // ids have already been admitted to this iteration's batch. Each
        // slot we want to add is gated by llama_memory_paged_can_admit;
        // if hot can't fit it alongside the already-admitted set, the
        // slot is preempted (whole-slot evict to warm) and skipped this
        // iteration. A non-paged backend short-circuits can_admit to
        // true, so this is a no-op there. paged_evicted_this_iter tracks
        // anyone we sent to warm so we don't try to admit them right
        // back in the prefill loop below.
        //
        // Anti-starvation: rotate which slot is iterated first across
        // admission decisions. Without this, with 4 slots and hot fitting
        // only 3, slot 3 always loses the admission race because it's
        // evaluated last and the first three fill the budget. Applied to
        // the admission pass only (where slots compete for hot); the
        // draft-params collection pass is read-only state gathering and
        // runs in natural order.
        llama_memory_t mem_for_admit = llama_get_memory(ctx_tgt);
        std::vector<llama_seq_id> paged_admitted;
        std::vector<llama_seq_id> paged_evicted_this_iter;
        paged_admitted.reserve(slots.size());

        // MAD-LAB: paged_admit_rotor is stream.paged_admit_rotor now (shadowed
        // above), not a function-local static -- see the server_stream comment.
        const size_t    n_slots_total     = slots.size();
        const size_t    rotor_off         = (n_slots_total > 0)
                                                ? (size_t)(paged_admit_rotor % n_slots_total)
                                                : 0;
        ++paged_admit_rotor;

        // MAD-LAB: slots[i] is now server_slot* (stream-scoped vector), so
        // this needs a dereference where it didn't before -- the compiler
        // caught this one (return type server_slot& vs the pointer the old
        // body would have produced).
        auto rotated_slot = [&](size_t i) -> server_slot & {
            return *slots[(rotor_off + i) % n_slots_total];
        };

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;


        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec, slot.stream_slot_idx).drafting = false; // MAD-LAB: spec is now a raw ptr

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.stream_slot_idx),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.stream_slot_idx));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec, slot.stream_slot_idx) = { // MAD-LAB: spec is now a raw ptr
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            const int64_t t0 = spec_phase ? ggml_time_us() : 0;
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec); // MAD-LAB: spec is now a raw ptr
            });
            if (spec_phase) {
                size_t n_drafted = 0;
                for (const auto * s : drafting) {
                    n_drafted += s->spec_draft.size();
                }
                SRV_INF("SPECPHASE draft_us=%" PRId64 " n_drafted=%zu\n", ggml_time_us() - t0, n_drafted);
            }
        }

        // make checkpoints if needed
        const int64_t t_ckpt0 = spec_phase ? ggml_time_us() : 0;
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft_full = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft_full) {
                    ckpt.load_dft(ctx_dft, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.stream_slot_idx, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.stream_slot_idx);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });
        if (spec_phase && !drafting.empty()) {
            SRV_INF("SPECPHASE ckpt_us=%" PRId64 "\n", ggml_time_us() - t_ckpt0);
        }

        // update the batch with the sampled/drafted tokens.
        // MAD-120: this is the admission pass — gate handle_last_sampled_token on
        // can_admit and iterate with anti-starvation rotation over the generating set.
        const size_t n_generating = generating.size();
        const size_t admit_rotor_off = (n_generating > 0)
                                           ? (size_t)(paged_admit_rotor % n_generating)
                                           : 0;
        const int32_t spec_const_width = server_spec_const_width();
        for (size_t _i = 0; _i < n_generating; ++_i) {
            auto & slot = *generating[(admit_rotor_off + _i) % n_generating];

            const int32_t n_verify_tokens = slot.spec_draft.empty()
                ? 1
                : 1 + std::max(spec_const_width, (int32_t) slot.spec_draft.size());
            if (spec_const_width > 0 && !slot.spec_draft.empty() &&
                    batch.size() + n_verify_tokens > batch.n_tokens_alloc) {
                SLT_DBG(slot, "deferring constant-shape verify: need %d batch rows, have %d\n",
                        n_verify_tokens, batch.n_tokens_alloc - batch.size());
                continue;
            }

            // MAD-120: admission. Reserve the complete padded verify batch when enabled.
            if (!llama_memory_paged_can_admit(
                        mem_for_admit, slot.stream_slot_idx,
                        spec_const_width > 0 && !slot.spec_draft.empty() ? n_verify_tokens : 1,
                        paged_admitted.data(), paged_admitted.size())) {
                int n_evicted = llama_memory_paged_evict_seq(mem_for_admit, slot.stream_slot_idx);
                SLT_INF(slot, "MAD-120 preempt (decode): hot pool full, "
                              "evicted %d block(s) to warm; will retry next iter\n",
                              n_evicted);
                paged_evicted_this_iter.push_back(slot.stream_slot_idx);
                continue;
            }

            slot.handle_last_sampled_token(batch);
            paged_admitted.push_back(slot.stream_slot_idx);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            // Preserve anti-starvation rotation: feed iterate() a snapshot of
            // slots in rotated order (rotated_slot/n_slots_total, defined above)
            // instead of slots' natural order, so one client can't perpetually
            // win first dibs on batch capacity.
            std::vector<server_slot *> rotated_slots;
            rotated_slots.reserve(n_slots_total);
            for (size_t _j = 0; _j < n_slots_total; ++_j) {
                rotated_slots.push_back(&rotated_slot(_j));
            }

            iterate(rotated_slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }


                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // MAD-120 Phase 2: paged-attn admission control for prefill.
                // We use the slot's full prompt size as a conservative
                // estimate of the new tokens this slot wants to add. If
                // the candidate's working set wouldn't fit alongside the
                // already-admitted seqs, evict the candidate and skip it
                // this iteration.
                if (slot.is_processing() && (slot.state == SLOT_STATE_PROCESSING_PROMPT ||
                                              slot.state == SLOT_STATE_STARTED)) {
                    const uint32_t n_new_est = slot.task ? (uint32_t) slot.task->n_tokens() : 0;
                    if (!llama_memory_paged_can_admit(
                                mem_for_admit, slot.stream_slot_idx, n_new_est,
                                paged_admitted.data(), paged_admitted.size())) {
                        // Already evicted above? skip the duplicate evict.
                        bool already = false;
                        for (auto sid : paged_evicted_this_iter) {
                            if (sid == slot.stream_slot_idx) { already = true; break; }
                        }
                        int n_evicted = 0;
                        if (!already) {
                            n_evicted = llama_memory_paged_evict_seq(mem_for_admit, slot.stream_slot_idx);
                            SLT_INF(slot, "MAD-120 preempt (prefill): hot pool full, "
                                          "evicted %d block(s) to warm; will retry next iter\n",
                                          n_evicted);
                            paged_evicted_this_iter.push_back(slot.stream_slot_idx);
                        }

                        // MAD-141: deadlock break. evict_seq returns 0 when
                        // the slot has no GPU-resident blocks to give back —
                        // typically because nothing has been prefilled yet
                        // and the prompt itself is too large for the hot
                        // budget. Without this guard the slot loops forever
                        // until the upstream n_empty_consecutive safety
                        // abort fires and crashes the server. Track
                        // consecutive no-progress preempt iterations and
                        // fail the request cleanly once it's clearly stuck.
                        constexpr int32_t kPagedPreemptDeadlockThreshold = 4;
                        if (n_evicted <= 0) {
                            ++slot.paged_preempt_no_progress_count;
                            if (slot.paged_preempt_no_progress_count >= kPagedPreemptDeadlockThreshold) {
                                SLT_ERR(slot,
                                        "MAD-141: paged admission stuck for %d iters with no eviction "
                                        "progress (n_new_est=%u). Prompt does not fit alongside the "
                                        "live workload. Failing request.\n",
                                        slot.paged_preempt_no_progress_count, n_new_est);
                                send_error(slot,
                                           string_format(
                                               "paged KV admission could not fit a %u-token request "
                                               "alongside the active workload after %d retries. "
                                               "Reduce the prompt or wait for slots to drain.",
                                               n_new_est, slot.paged_preempt_no_progress_count),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                // Releasing a deadlocked slot IS progress — reset
                                // the empty-batch streak so the upstream safety
                                // abort doesn't fire on this same iteration just
                                // because we haven't built a token batch yet.
                                n_empty_consecutive = 0;
                                return;
                            }
                        }
                        return;
                    }
                    // Slot was admitted — any prior no-progress streak ends here.
                    slot.paged_preempt_no_progress_count = 0;
                    paged_admitted.push_back(slot.stream_slot_idx);
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // mt:: tier semantic restore: BEFORE prefix-match,
                            // try to pull any semantically-similar warm-tier
                            // chunks back into hot. The restored content gets
                            // tagged with its original positions, so when
                            // get_common_prefix runs next it can extend its
                            // match if the new prompt happens to overlap
                            // with what was just restored. Fires only when
                            // --kv-tier-semantic-index is set and the wrapper
                            // is the active memory.
                            if (auto * mt_tier = dynamic_cast<mt::llama_memory_tiered *>(llama_get_memory(ctx_tgt))) {
                                if (!params_base.kv_semantic_index.empty() && !input_tokens.has_mtmd) {
                                    // MAD-122/125: paged-blocks routes through
                                    // llama_kv_cache_paged (the active tier
                                    // layer for hybrid+paged). Non-paged falls
                                    // back to the chunk-level wrapper path.
                                    llama_kv_cache_paged * paged_cache = params_base.kv_tier_paged_blocks
                                        ? mt_get_paged_cache(llama_get_memory(ctx_tgt)) : nullptr;

                                    // MAD-348: do NOT pay an embedder forward pass to
                                    // build a query vector for a search that provably
                                    // cannot return anything. restore_semantic_paged()
                                    // only acts on fingerprinted blocks that are mapped
                                    // and NOT already hot; if every block is resident,
                                    // it returns 0 no matter what the query vector is.
                                    //
                                    // This is not a micro-optimisation. The embed runs
                                    // SYNCHRONOUSLY on the single-threaded server loop
                                    // (measured: ~285 ms/call, CPU embedder), so every
                                    // wasted call stalls the batch carrying every other
                                    // slot's decode token. On a short-context, high
                                    // fan-out workload nothing is ever evicted, so this
                                    // fired on every request and restored nothing.
                                    const bool can_restore = paged_cache
                                        ? paged_cache->has_restorable_blocks(slot.stream_slot_idx)
                                        : true;  // chunk path has no cheap predicate; unchanged


                                    if (can_restore) {
                                        // Embed the new prompt (or its leading window).
                                        // bge-small caps at ~512 tokens; truncate the
                                        // prompt to that for the query embedding.
                                        const auto & qtoks = input_tokens.get_text_tokens();
                                        const int q_max = std::min<int>(512, (int) qtoks.size());
                                        if (q_max > 0) {
                                            llama_tokens q(qtoks.begin(), qtoks.begin() + q_max);
                                            const std::string qtext = common_detokenize(ctx_tgt, q, /*special=*/ false);
                                            const auto qemb = mt_tier->embed_text(qtext, mt::EmbedRole::Query);
                                            if (!qemb.empty()) {
                                                const uint32_t restored = paged_cache
                                                    ? paged_cache->restore_semantic_paged(
                                                          slot.stream_slot_idx, qemb,
                                                          params_base.kv_semantic_top_k,
                                                          params_base.kv_semantic_threshold)
                                                    : mt_tier->restore_semantic(
                                                          slot.stream_slot_idx, qemb,
                                                          params_base.kv_semantic_top_k,
                                                          params_base.kv_semantic_threshold);
                                                if (restored > 0) {
                                                    SLT_INF(slot, "tier semantic: restored %u positions/blocks via cosine search (%s path)\n",
                                                            restored,
                                                            paged_cache ? "paged-block" : "chunk");
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.stream_slot_idx, head_p, head_c);
                                            slot.mem.seq_add(slot.stream_slot_idx, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            if (segment_client && segment_client->has_remote_segments()) {
                                // v1: every request starts from a clean slate on every segment.
                                // Cross-segment prompt reuse is a later optimization; the
                                // reuse negotiation left head/segment state inconsistent and
                                // cost determinism (2026-08-15).
                                if (segment_cache_epoch == UINT64_MAX) {
                                    throw std::runtime_error("dense segment cache epoch is exhausted");
                                }
                                segment_client->reset(segment_session_id, ++segment_cache_epoch);
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.stream_slot_idx);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.stream_slot_idx, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec, slot.stream_slot_idx, it->data_spec); // MAD-LAB: spec is now a raw ptr

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.stream_slot_idx, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.stream_slot_idx,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                        SLT_INF(slot, "prompt processing done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.size());
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.stream_slot_idx);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.stream_slot_idx);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 1): decode() reads ONLY
    // through `stream` now -- same shadowing approach as pre_decode() above.
    // The n_cmpl>1 parent/child activation loop that used to live at the
    // end of this function has been HOISTED to update_slots() (see
    // activate_parent_child_tasks()) instead of parameterized: unlike
    // everything else here, it genuinely needs to see every slot regardless
    // of stream (a child could in principle land on a different stream's
    // slot than its parent), so it belongs in the single-threaded part of
    // the tick, not inside a per-stream call that will eventually run on a
    // per-stream thread.
    //
    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(server_stream & stream, int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        llama_context * const ctx_tgt   = stream.ctx_tgt;
        common_speculative * const spec = stream.spec;
        server_batch & batch = *stream.batch;
        int32_t & n_empty_consecutive = stream.n_empty_consecutive;
        std::vector<server_slot *> & slots = stream.slots;

        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            // MAD-LAB: model_dft/model_tgt are NOT stream fields -- model_tgt
            // is the one shared model every stream decodes against, and
            // stream.model_dft (added for exactly this check) is this
            // stream's own draft/MTP model, mirroring model_dft/model_tgt today.
            if (llama_model_n_embd_inp(stream.model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
        }

        const int64_t t_local0 = spec_phase ? ggml_time_us() : 0;

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (spec_phase) {
            SRV_INF("SPECPHASE local_decode_us=%" PRId64 " n_tokens=%d\n", ggml_time_us() - t_local0, batch_view.n_tokens);
        }

        // MAD-LAB DS4-Flash pipeline-streams (stage 2): narrow lock -- see
        // metrics_mtx_'s declaration for why. server_metrics's counters
        // are not otherwise thread-safe, and this can now be called from
        // either stream's thread.
        {
            std::lock_guard<std::mutex> lock(metrics_mtx_);
            metrics.n_decode++;
            for (const auto * slot : slots) {
                if (slot->is_processing()) {
                    metrics.n_busy_slots++;
                }
                metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot->prompt.n_tokens());
            }
        }

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    // MAD-LAB: scoped to stream.slots -- a decode() failure
                    // on this stream's context must not touch the other
                    // stream's slots (separate contexts, separate KV pools).
                    for (auto * slot_ptr : slots) {
                        auto & slot = *slot_ptr;
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots(stream)) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        if (segment_client && segment_client->has_remote_segments()) {
            const int32_t n_embd = llama_model_n_embd(model_tgt);
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            const float * hidden = llama_get_embeddings(ctx_tgt);
            if (hidden == nullptr) {
                throw std::runtime_error("local dense segment did not return hidden activations");
            }
            std::vector<float> activations(hidden, hidden + (size_t) batch_view.n_tokens * n_embd);
            std::vector<int32_t> positions(batch_view.pos, batch_view.pos + batch_view.n_tokens);
            if (segment_next_seq_id == 0 || segment_next_seq_id == UINT64_MAX) {
                throw std::runtime_error("dense segment sequence space is exhausted");
            }
            const int64_t t_seg0 = spec_phase ? ggml_time_us() : 0;
            segment_client->begin_forward(
                segment_session_id, segment_next_seq_id++, (uint32_t) batch_view.n_tokens,
                positions, activations);
            const pipe_segment_fwd_resp terminal = segment_client->finish_forward();
            if (spec_phase) {
                SRV_INF("SPECPHASE seg_fwd_us=%" PRId64 " n_tokens=%d terminal_bytes=%zu nextn_bytes=%zu\n",
                        ggml_time_us() - t_seg0, batch_view.n_tokens,
                        terminal.activations.size() * sizeof(float),
                        terminal.nextn.size() * sizeof(float));
            }

            if (segment_client->terminal_kind() == PIPE_SEGMENT_TERMINAL_HIDDEN) {
                // LOGITS-ON-HEAD. terminal.activations is the tail's
                // post-output_norm hidden state ([n_tokens][n_embd] f32) -- exactly
                // the tensor the tail's LM head would have consumed. Finish the
                // projection here, straight into ctx_tgt's logits buffer, so the
                // sampler and common_speculative_process see the same thing they
                // always did. 5120 vs 248320 floats per token on the wire.
                if (terminal.activations.size() != (size_t) batch_view.n_tokens * n_embd) {
                    throw std::runtime_error("dense segment terminal hidden state has an invalid width");
                }
                const int64_t t_proj0 = spec_phase ? ggml_time_us() : 0;
                if (!llama_output_project(ctx_tgt, terminal.activations.data(), batch_view.n_tokens)) {
                    throw std::runtime_error("head-side LM head projection failed");
                }
                if (spec_phase) {
                    SRV_INF("SPECPHASE head_project_us=%" PRId64 " n_tokens=%d\n",
                            ggml_time_us() - t_proj0, batch_view.n_tokens);
                }
            } else {
                float * logits = ctx_tgt->get_logits();
                if (terminal.activations.size() != (size_t) batch_view.n_tokens * n_vocab || logits == nullptr) {
                    throw std::runtime_error("dense segment terminal logits have an invalid width");
                }
                std::memcpy(logits, terminal.activations.data(),
                            terminal.activations.size() * sizeof(float));

                // WP_SEGMENT_PROJ_CHECK=1: measure the head-side projection against
                // the tail's, IN THE SAME PROCESS AND THE SAME STEP, without changing
                // what the sampler consumes.
                //
                // This works because terminal.nextn is ALREADY the post-output_norm
                // hidden state for every token (qwen35.cpp sets t_h_nextn from the
                // output_norm output, and the tail runs unmasked so it covers all
                // rows) -- the exact input the tail's own LM head consumed. So we can
                // project it here and diff against the logits the tail computed, with
                // the tail's arm still driving generation. That quantifies the
                // Vulkan-vs-HIP/split-matmul deviation on real activations BEFORE
                // switching the wire over, and reports argmax agreement, which is the
                // only thing temp-0 token parity actually depends on.
                static const bool proj_check = [] {
                    const char * v = std::getenv("WP_SEGMENT_PROJ_CHECK");
                    return v != nullptr && v[0] == '1';
                }();
                if (proj_check && !terminal.nextn.empty() &&
                    terminal.nextn.size() == (size_t) batch_view.n_tokens * n_embd &&
                    model_tgt->output != nullptr) {
                    std::vector<float> reference(terminal.activations);
                    if (llama_output_project(ctx_tgt, terminal.nextn.data(), batch_view.n_tokens)) {
                        double max_abs = 0.0;
                        double max_rel = 0.0;
                        int    n_argmax_diff = 0;
                        for (int t = 0; t < batch_view.n_tokens; ++t) {
                            const float * a = reference.data() + (size_t) t * n_vocab; // tail
                            const float * b = logits          + (size_t) t * n_vocab; // head
                            int am_a = 0;
                            int am_b = 0;
                            for (int v = 0; v < n_vocab; ++v) {
                                const double d = std::fabs((double) a[v] - (double) b[v]);
                                max_abs = std::max(max_abs, d);
                                const double m = std::fabs((double) a[v]);
                                if (m > 1e-3) {
                                    max_rel = std::max(max_rel, d / m);
                                }
                                if (a[v] > a[am_a]) am_a = v;
                                if (b[v] > b[am_b]) am_b = v;
                            }
                            if (am_a != am_b) {
                                ++n_argmax_diff;
                                SRV_WRN("PROJCHECK ARGMAX FLIP tok=%d tail=%d head=%d "
                                        "tail_logit=%.6f head_logit=%.6f\n",
                                        t, am_a, am_b, a[am_a], b[am_b]);
                            }
                        }
                        SRV_INF("PROJCHECK n_tokens=%d max_abs=%.6e max_rel=%.6e argmax_flips=%d\n",
                                batch_view.n_tokens, max_abs, max_rel, n_argmax_diff);
                    }
                    // restore the tail's logits: the LOGITS arm must remain the one
                    // driving generation, so the check never perturbs the trajectory
                    std::memcpy(logits, reference.data(), reference.size() * sizeof(float));
                }
            }
            // Consume the tail's NextN sideband only if the speculative arm actually
            // reads it. Only draft-mtp does -- see
            // common_speculative_impl_draft_mtp::need_embd_nextn(); every other impl
            // inherits the base `false`. A DFlash/DSpark SIDECAR conditions on interior
            // layer taps plus its OWN draft context's nextn, and never on the target's.
            //
            // The tail ships this payload unconditionally (nextn_width is decided at the
            // worker's load time and is not negotiated), so on the dspark arm it arrives
            // for a context that never allocated the receiving buffer: only draft-mtp
            // calls llama_set_embeddings_nextn(ctx_tgt, ...). Keying on presence rather
            // than on need therefore tripped the null check on a payload nobody wanted.
            //
            // Pairs with the need && empty check below: need+present consumes,
            // need+absent errors, !need+present is ignored.
            if (spec && common_speculative_need_embd_nextn(spec) && !terminal.nextn.empty()) { // MAD-LAB: spec is a raw ptr
                float * nextn = ctx_tgt->get_embeddings_nextn();
                if (terminal.nextn.size() != (size_t) batch_view.n_tokens * n_embd || nextn == nullptr) {
                    throw std::runtime_error("dense segment terminal NextN activations have an invalid width");
                }
                std::memcpy(nextn, terminal.nextn.data(),
                            terminal.nextn.size() * sizeof(float));
                // the local graph never produces the tap on a banded head, so
                // declare the wire width for the *_ith accessors
                ctx_tgt->set_embeddings_nextn_width((uint32_t) n_embd);
            }

            // INTERIOR TAPS: install every layer hidden state the remote segments
            // forwarded, so llama_get_embeddings_layer_inp() returns it as if this
            // process had computed it. Safe here because llama_decode() has already
            // completed for this batch_view -- extract_layer_inputs() has run and was a
            // no-op for these layers -- and nothing reads the taps until
            // common_speculative_process() below.
            for (const auto & tap : segment_client->taps()) {
                if (tap.width != (uint32_t) n_embd ||
                    tap.rows.size() != (size_t) batch_view.n_tokens * n_embd) {
                    throw std::runtime_error("dense segment interior tap has an invalid width");
                }
                if (!llama_set_layer_inp_data(ctx_tgt, tap.layer, tap.rows.data(), batch_view.n_tokens)) {
                    throw std::runtime_error("failed to install a forwarded interior tap");
                }
            }
            if (spec && common_speculative_need_embd_nextn(spec) && terminal.nextn.empty()) { // MAD-LAB: spec is a raw ptr
                throw std::runtime_error("dense segment terminal did not return NextN activations");
            }
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        const int64_t t_proc0 = spec_phase ? ggml_time_us() : 0;
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec, batch_view); // MAD-LAB: spec is a raw ptr
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }
        if (spec_phase && spec) {
            SRV_INF("SPECPHASE process_us=%" PRId64 "\n", ggml_time_us() - t_proc0);
        }

        // MAD-LAB DS4-Flash pipeline-streams: the n_cmpl>1 parent/child
        // activation loop that used to live here has been hoisted to
        // activate_parent_child_tasks(), called once from update_slots()
        // after every stream's decode() has run this tick -- see that
        // function for why.

        return true;
    }

    // MAD-LAB DS4-Flash pipeline-streams: hoisted out of decode() (see the
    // comment there). Handles `n_cmpl > 1` tasks -- when a parent's prompt
    // finishes, copy its state to every child slot waiting on it. This
    // genuinely needs to see every slot regardless of stream (a child could
    // land on a different stream's slot than its parent -- slot->stream
    // assignment is not request-aware), so unlike the rest of decode()/
    // post_decode() it is NOT parameterized by server_stream and stays a
    // single call over the full `slots` member, run from the
    // single-threaded part of the tick (i.e. after any per-stream threads
    // for this tick have already been joined).
    void activate_parent_child_tasks() {
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }
    }

    // MAD-LAB DS4-Flash pipeline-streams (stage 1): post_decode() reads
    // ONLY through `stream` now. Most of this function already used
    // slot.ctx_tgt/slot.ctx_dft/slot.spec (per-slot fields wired to the
    // right stream since the earlier pipeline-streams slot-split work) --
    // the shadows below only matter for the handful of bare
    // ctx_tgt_seq_rm_type/ctx_tgt/spec.get() reads that weren't already
    // going through the slot.
    void post_decode(server_stream & stream, int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        llama_context * const ctx_tgt   = stream.ctx_tgt;
        common_speculative * const spec = stream.spec;
        const common_context_seq_rm_type ctx_tgt_seq_rm_type = stream.ctx_tgt_seq_rm_type;
        std::vector<server_slot *> & slots = stream.slots;

        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec, slot.stream_slot_idx, slot.prompt.tokens.get_text_tokens()); // MAD-LAB: spec is a raw ptr
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            bool segment_trim_required = false;
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
                segment_trim_required = n_rollback > 0;

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.stream_slot_idx, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        if (segment_client && segment_client->has_remote_segments()) {
                            segment_client->trim(segment_session_id, segment_cache_epoch,
                                                 (uint32_t) (ckpt.pos_max + 1));
                        }
                        slot.mem.seq_rm(slot.stream_slot_idx, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec, slot.stream_slot_idx, accepted.size() - 1); // MAD-LAB: spec is a raw ptr

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            if (segment_trim_required && segment_client && segment_client->has_remote_segments()) {
                const int64_t t_trim0 = spec_phase ? ggml_time_us() : 0;
                segment_client->trim(segment_session_id, segment_cache_epoch,
                                     (uint32_t) slot.prompt.tokens.pos_next());
                if (spec_phase) {
                    SRV_INF("SPECPHASE trim_us=%" PRId64 "\n", ggml_time_us() - t_trim0);
                }
            }
            slot.mem.seq_rm(slot.stream_slot_idx, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (slot.stats.n_gen == 1) {
                    slot.stats.update_prompt_last();
                }

                slot.stats.update_gen_last();

                // MAD-LAB: periodic generation trace (reasoning-budget state included)
                if (slot.stats.n_gen % 64 == 0 || slot.stats.n_gen == 1) {
                    const auto * rbudget  = common_sampler_get_rbudget(slot.smpl.get());
                    const int32_t n_think = common_reasoning_budget_get_n_thinking(rbudget);
                    const auto    rb_st   = common_reasoning_budget_get_state(rbudget);
                    SLT_INF(slot, "generate: n_gen=%d n_past=%d t/s=%.2f rbudget_state=%d n_thinking=%d token_id=%d\n",
                            (int) slot.stats.n_gen, slot.prompt.n_tokens(), slot.stats.n_gen_tps(),
                            (int) rb_st, n_think, (int) ids[i]);
                }

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        // MAD-133: enrich each slot's JSON with paged-tier residency
        // (blocks_hot / blocks_warm / blocks_cold / fingerprints).
        // Skipped when paged isn't on or the slot's id is missing.
        json slots_out = res_task->to_json();
        if (slots_out.is_array() && ctx_server.ctx_tgt) {
            llama_kv_cache_paged * paged_cache = mt_get_paged_cache(llama_get_memory(ctx_server.ctx_tgt));
            if (paged_cache) {
                for (auto & slot : slots_out) {
                    if (!slot.contains("id")) continue;
                    const llama_seq_id sid = slot["id"].get<llama_seq_id>();
                    if (sid < 0) continue;
                    slot["tier"] = {
                        {"blocks_hot",   paged_cache->n_gpu_blocks_for(sid)},
                        {"blocks_warm",  paged_cache->n_warm_blocks_for(sid)},
                        {"blocks_cold",  paged_cache->n_blocks_cold_for(sid)},
                        {"fingerprints", paged_cache->n_fingerprints_for_seq(sid)},
                    };
                }
            }
        }

        res->ok(slots_out);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
