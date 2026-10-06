#pragma once

#include "llama.h"

#include "common.h"

#include <functional>
#include <random>
#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// as above, but verifies by rejection sampling; draft_q holds the draft's candidates per token
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

// MAD-LAB: speculative SAMPLING (Leviathan/Chen et al.) verification, the stochastic counterpart of
// common_sampler_sample_and_accept_n(). The draft tokens x_i were sampled from q_i; x_i is accepted
// with probability min(1, p_i(x_i)/q_i(x_i)) where p_i is the target's FINAL sampling distribution at
// position i (the candidates the full sampler chain leaves, normalized -- penalties, top-k/top-p/min-p,
// temperature, ...). The first rejection resamples from norm(max(0, p_i - q_i)) and stops; if every
// draft token is accepted a bonus token is sampled from p_n. The output distribution equals the
// target's exactly, for any q.
//
// Pure math core, no llama context needed (also used by tests/test-spec-sampling):
//   target_p(i, cand)  fill cand with the target distribution at position i, i in [0, n_draft], given
//                      that tokens 0..i-1 of the result were already reported through on_token
//   q_prob(i, tok)     q_i(tok), the draft distribution the token at position i was sampled from
//   on_token(i, tok)   called for every returned token, in order, before target_p(i + 1, ...)
// accept_probs (optional) receives min(1, p/q) of every draft position that was tested.
std::vector<llama_token> common_spec_verify_stochastic(
        size_t                        n_draft,
        const llama_token           * draft,
        const std::function<void(size_t, std::vector<llama_token_data> &)> & target_p,
        const std::function<double(size_t, llama_token)>                    & q_prob,
        const std::function<void(size_t, llama_token)>                      & on_token,
        std::mt19937                & rng,
        std::vector<float>          * accept_probs = nullptr);

// true if this sampler's final distribution is a plain softmax over its post-chain candidates, i.e. the
// stochastic verification above is exact for it: temp > 0, no mirostat / XTC / adaptive-p, no grammar.
bool common_sampler_spec_sampling_ok(const struct common_sampler * gsmpl);

// Sampler-chain driven stochastic verification. q_logits is [draft.size()][n_vocab_q] row-major draft
// logits; the draft token at row i was sampled from softmax(q_logits[i] / q_temp). Falls back to
// common_sampler_sample_and_accept_n() (exact match) when the sampler is not eligible, backend sampling
// already picked the target tokens, or q_logits is missing.
// is_replay: the draft is the already-accepted output of an earlier stochastic round (restored from a
// checkpoint) -- accept it unconditionally and only draw the bonus token.
std::vector<llama_token> common_sampler_sample_and_accept_n_stochastic(
        struct common_sampler * gsmpl,
        struct llama_context  * ctx,
        const std::vector<int> & idxs,
        const llama_tokens    & draft,
        const float           * q_logits,
        int32_t                 n_vocab_q,
        float                   q_temp,
        std::mt19937          & rng,
        bool                    is_replay,
        std::vector<float>    * accept_probs = nullptr);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// Returns the internal reasoning-budget llama_sampler (nullptr if not active).
const struct llama_sampler * common_sampler_get_rbudget(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
