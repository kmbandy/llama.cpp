#pragma once

#include "llama.h"
#include "common.h"

#include <thread>

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

// MAD-LAB: DSpark speculative sampling. The draft distribution of a stochastically drafted block, kept
// next to the draft tokens (it must survive until the draft is verified) so the verifier can run the
// accept/reject test. Row i (n_vocab floats) holds the draft logits that token i was sampled from as
// softmax(logits / temp).
struct common_speculative_draft_q {
    float              temp    = 0.0f;
    int32_t            n_vocab = 0;
    std::vector<float> logits;

    // WP_DSPARK_Q_PRECOMPUTE=1: per-row normaliser stats (max, Z) of the stochastic-accept q_prob, computed by
    // a worker thread started once logits is complete. Values come from common_spec_q_norm_stats(), the very
    // function the lazy path uses, so they are bit-identical. pre_start() / pre_join() / clear() manage it.
    std::vector<double> pre_max;
    std::vector<double> pre_z;
    std::thread         pre_th;

    common_speculative_draft_q() = default;
    common_speculative_draft_q(const common_speculative_draft_q &) = delete;
    common_speculative_draft_q & operator=(const common_speculative_draft_q &) = delete;
    // moves are safe with a running worker: it only holds raw pointers into the vectors' heap buffers, which a
    // vector move keeps in place
    common_speculative_draft_q(common_speculative_draft_q &&) noexcept = default;
    common_speculative_draft_q & operator=(common_speculative_draft_q && o) noexcept {
        if (this != &o) {
            pre_join();
            temp = o.temp; n_vocab = o.n_vocab;
            logits = std::move(o.logits); pre_max = std::move(o.pre_max); pre_z = std::move(o.pre_z);
            pre_th = std::move(o.pre_th);
        }
        return *this;
    }
    ~common_speculative_draft_q() { pre_join(); }

    void pre_join() { if (pre_th.joinable()) { pre_th.join(); } }
    bool pre_ready() const { return !pre_th.joinable() && n_vocab > 0 && pre_z.size() == n() && !pre_z.empty(); }
    void pre_start();   // speculative.cpp / sampling.cpp: launches the worker over the current logits

    void   clear()       { pre_join(); pre_max.clear(); pre_z.clear(); temp = 0.0f; n_vocab = 0; logits.clear(); }
    size_t n() const     { return n_vocab > 0 ? logits.size() / (size_t) n_vocab : 0; }
};

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // candidate distribution per drafted token; set it to make draft-simple and draft-mtp sample
    std::vector<std::vector<llama_token_data>> * result_q = nullptr;

    // the target's temp and seed, read only when the drafter samples probabilistically
    float    temp = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;

    // MAD-LAB: DSpark speculative sampling (only honored with --spec-draft-sampling stochastic).
    // dspark_temp > 0 asks the drafter to SAMPLE its chain at that temperature, seeded by noise_seed, and to
    // fill *dspark_q with the draft logits; dspark_temp <= 0 or dspark_q == nullptr keeps the greedy chain. dspark_q is cleared by
    // common_speculative_draft() for every drafting sequence and is only left non-empty when the
    // returned draft is a stochastic one.
    float                         dspark_temp = 0.0f;
    uint64_t                      noise_seed  = 0;
    common_speculative_draft_q *  dspark_q    = nullptr;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// reset per-sequence state before starting an uncached generation
void common_speculative_reset(common_speculative * spec, llama_seq_id seq_id);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// fork: the speculative implementations still consume llama_batch; this overload lets upstream's
// common_batch callers through by flattening the batch
bool common_speculative_process(common_speculative * spec, const common_batch & batch);

// MAD-LAB: prefill-sync pipelining (see draft-sync-cost-0912.txt). Resolves
// whatever a prior common_speculative_process() call deferred (currently
// only draft-mtp's single-head path defers anything; every other
// implementation's flush is a no-op). MUST be called once after the LAST
// common_speculative_process() call for a prompt and before the first
// common_speculative_draft() call for it -- draft() reads per-sequence
// state (pending_h/i_last/chain_h) that a deferred process() call has not
// written yet. Safe to call even when nothing is pending (no-op then too).
bool common_speculative_flush_prefill(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// WP_STEP_STATS: number of llama_decode(ctx_dft) calls issued by the most
// recent common_speculative_draft(spec) call, summed across every
// implementation in spec->impls (only common_speculative_impl_draft_mtp
// currently sets its own counter; every other impl contributes 0). 0 if
// spec is null.
size_t common_speculative_last_n_draft_decodes(const common_speculative * spec);

// WP_SPEC_PREFILL_STATS: wall-clock breakdown of the MOST RECENT
// common_speculative_process(spec, batch) call's draft-mtp hidden-state
// handoff, in nanoseconds. Only set (nonzero) by
// common_speculative_impl_draft_mtp's process() override, in its
// non-mem-shared (catch-up decode) branch; every other implementation, and
// the mem-shared branch (e.g. Gemma4), leaves these at 0. All fields are 0
// unless WP_SPEC_PREFILL_STATS=1 -- the timers are not taken at all when
// unset, so this is a zero-cost accessor returning zeroes.
//   a_sync_ns   -- time inside llama_get_embeddings_nextn(ctx_tgt), which is
//                  ctx_tgt->synchronize() (the dual-GPU meta-backend drain)
//                  plus the embeddings_nextn pointer fetch.
//   b_copy_ns   -- host-side work around that: the draft llama_batch build
//                  (common_batch_clear/common_batch_add loop) plus the
//                  memcpy of h_tgt into batch.embd and the pending/prior
//                  hidden-state fill (set_h()).
//   c_decode_ns -- sum of llama_decode(ctx_dft, batch) calls (the per-MTP-
//                  layer catch-up decode loop).
//   c_n_chunks  -- number of llama_decode(ctx_dft, ...) calls summed into
//                  c_decode_ns (== n_mtp_layers on success).
//   c_n_tokens  -- sum of n_tokens across those calls.
struct common_speculative_wp_prefill_call_stats {
    uint64_t a_sync_ns   = 0;
    uint64_t b_copy_ns   = 0;
    uint64_t c_decode_ns = 0;
    uint32_t c_n_chunks  = 0;
    uint64_t c_n_tokens  = 0;
};
common_speculative_wp_prefill_call_stats common_speculative_wp_prefill_last_call_stats(const common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// MAD-LAB: true if any impl needs the target to extract PRE-norm (NextN) embeddings.
// Only draft-mtp answers true; the dense-segment terminal contract depends on it
// (see src/pipeline/pipe-protocol.h).
bool common_speculative_need_embd_nextn(common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
