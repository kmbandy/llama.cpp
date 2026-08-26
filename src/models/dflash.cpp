#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-ext.h"

#include <atomic>
#include <cinttypes>
#include <unordered_set>

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    ml.get_key(LLM_KV_DFLASH_HC_MULT, dflash_hc_mult, false);
    if (dflash_hc_mult == 0) {
        throw std::runtime_error("DFlash model has invalid 'dflash.hc_mult' metadata");
    }
    ml.get_key(LLM_KV_BLOCK_SIZE, dflash_block_size, false);

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * dflash_hc_mult * hparams.n_embd;

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());
    LLAMA_LOG_INFO("%s: DFlash hc_mult = %u\n", __func__, dflash_hc_mult);

    // DeepSeek-V4 DSpark backbone: stages are full DSV4 blocks, uniform sliding window (the draft KV ring)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,                hparams.n_lora_q);
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,             hparams.n_swa);
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,           hparams.n_ff_exp);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                  hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,                 hparams.expert_weights_scale);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,                  hparams.expert_weights_norm);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                   hparams.expert_gating_func);
        ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,              hparams.swiglu_clamp_exp, hparams.n_layer_all);
        if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,       hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
            hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
        }
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
        ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
        ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS,            hparams.dsv4_compress_ratios, false);

        GGML_ASSERT(hparams.dsv4_o_group_count > 0); // avoid div by zero

        if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            throw std::runtime_error("DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (hparams.dsv4_compress_ratios[il] != 0) {
                throw std::runtime_error("DSpark DSV4 draft expects uncompressed attention on all stages");
            }
        }

        GGML_ASSERT(hparams.n_swa > 0);
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_swa_impl[il] = true;
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // optional interleaved sliding-window attention with per-layer pattern array.
    // DFlash has a single rope, so the SWA rope == main rope.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    } else {
        // Some DFlash exports omit the sliding-window key even though the drafter was
        // trained with SWA. That loads cleanly and silently drafts with full attention,
        // which only shows up as a lower acceptance rate -- so say so out loud.
        LLAMA_LOG_WARN("%s: DFlash export declares no sliding window; the drafter will use "
                       "FULL attention. If this speculator was trained with SWA, acceptance "
                       "will be degraded.\n", __func__);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    } else {
        d2t = nullptr;
        LLAMA_LOG_INFO("%s: DFlash without d2t (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
    //
    // TODO: only Qwen3-style backbones are supported for now; other backbones (e.g. Gemma4)
    //       need their own conversion path and graph tweaks
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab_draft }, 0);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    if (hparams.dsv4_hc_mult > 0) {
        const int64_t q_lora_rank     = hparams.n_lora_q;
        const int64_t n_ff_exp        = hparams.n_ff_exp;
        const int64_t n_expert_shared = hparams.n_expert_shared;
        const int64_t n_embd_head     = hparams.n_embd_head_k();
        const int64_t o_groups        = hparams.dsv4_o_group_count;
        const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
        const int64_t hc_mult         = hparams.dsv4_hc_mult;
        const int64_t hc_dim          = hc_mult * n_embd;
        const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

        hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
        hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
        hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
            layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
            layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
            layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
            layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        }
        return;
    }

    // Optional per-aux-layer norm applied to each target-layer hidden slice BEFORE
    // the fc fusion. Present on the Laguna-generation DFlash export, absent on the
    // original (DS4) one, which normalises only after fc via output_norm_enc.
    // NOTE: deliberately AFTER upstream's dsv4 branch, which returns early -- the
    // DSV4 DSpark draft borrows tok_embd/output from the target via ctx_other and
    // must leave them null, and aux_norm_enc is Laguna-only.
    const int64_t n_aux = (int64_t) target_layer_ids.size();
    aux_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_AUX_NORM, "weight"), { n_embd, n_aux }, TENSOR_NOT_REQUIRED);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        // Optional softplus attention-output gate. Present when the drafter is built
        // from a gated decoder block (dflash.decoder_arch = "laguna"); absent on the
        // plain qwen3-style block. Width selects the layout exactly as in laguna.cpp
        // -- never guess between the two.
        const ggml_tensor * gate_meta = ml->get_tensor_meta(tn(LLM_TENSOR_ATTN_GATE, "weight", i).str().c_str());
        if (gate_meta != nullptr) {
            const int64_t n_gate_per_head = n_head;
            const int64_t n_gate_per_elem = n_embd_head_k * n_head;
            const int64_t n_gate_out      = gate_meta->ne[1];
            if (n_gate_out != n_gate_per_head && n_gate_out != n_gate_per_elem) {
                GGML_ABORT("DFlash: unexpected attention gate width %lld at layer %d "
                           "(expected %lld per-head or %lld per-element)",
                           (long long) n_gate_out, i, (long long) n_gate_per_head, (long long) n_gate_per_elem);
            }
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, n_gate_out }, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.dsv4_hc_mult > 0) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    // Per-aux-layer norm, applied to each target-layer slice before fusion.
    // cur is [n_embd*n_aux, n_tokens] with each slice contiguous along ne[0], so
    // viewing it as [n_embd, n_aux, n_tokens] makes ggml_rms_norm (which reduces
    // over ne[0]) normalise every slice independently -- one op for all of them.
    // The weight is [n_embd, n_aux] and broadcasts over the token axis.
    if (model.aux_norm_enc) {
        const int64_t n_aux = model.aux_norm_enc->ne[1];

        GGML_ASSERT(cur->ne[0] == model.aux_norm_enc->ne[0] * n_aux);

        cur = ggml_reshape_3d(ctx0, cur, model.aux_norm_enc->ne[0], n_aux, n_tokens);
        cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.aux_norm_enc);
        cur = ggml_reshape_2d(ctx0, cur, hparams.n_embd_inp_enc(), n_tokens);
        cb(cur, "aux_norm_out", -1);
    }

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;
    if (cparams.embeddings_layer_inp[0]) {
        res->t_layer_inp[0] = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
// MAD-LAB: pure-ggml core of the DSpark Markov/confidence head.
//
// Split out of build_dspark_markov_head() so it can also be built into a STANDALONE
// graph by llama_context::dspark_markov_head(). A sidecar draft whose target is
// Meta-split cannot compute the LM head in its own graph, so the driver projects the
// hidden state through the target context and then replays just this head here, with
// `base` supplied as a plain input instead of being produced upstream in the graph.
//
// Takes no llm_graph_context on purpose: `base` and `conf_inp` are parameters rather
// than res->t_logits / res->t_embd, and nothing is expanded into a graph -- the caller
// owns gf and decides what to build. Returns false if the ubatch carries more drafts
// per block than the head was trained for, OR if the ubatch's blocks are not all the
// same width (a torn block from ubatch splitting -- see the multi-sequence-safe note
// below), in which case the caller keeps `base` unbiased (the same early-out the
// in-graph path has always had).
// MAD-LAB / multi-sequence-safe: how many times a ragged (non-block-aligned) ubatch
// reached the DSpark Markov head and was skipped, across both the in-graph and the
// standalone (services-mode) call paths. Prevention (the draft context's hard
// n_ubatch >= n_seq*block_width check at construction, see common/speculative.cpp)
// should make this permanently zero for every real drafting decode; a nonzero value
// means that invariant was violated somewhere and the affected block's confidence was
// forced to 0 rather than served stale -- see llama_dspark_markov_ragged_skipped_inc()
// below and its counterpart read in common_speculative_print_stats().
static std::atomic<int64_t> g_dspark_markov_ragged_skipped{0};

// MAD-LAB / multi-sequence-safe: rate-limited diagnostic for the ragged path, added
// 2026-08-24 because the recount fix (dspark_count_seqs_with_tokens(), below) and its
// reserve-ubatch fallback did NOT bring a live --parallel 2 rig's counter to 0 (it read 4
// at the first stats print, same as before both fixes), so the exact call shape needs to
// be captured from the log on the next live cycle instead of guessed at from stats alone.
// Capped at 8 total lines across both increment sites so a pathological run can't flood
// the log; the counter itself (unrate-limited) is still exact.
//
// Grep for "DSPARK_RAGGED" to find these.
static std::atomic<int32_t> g_dspark_ragged_log_budget{8};

// site: "in-graph" (build_dspark_markov_head(), the path this rig's services_mode=0
//   config actually exercises) or "standalone" (llama_dspark_build_markov_graph()'s own
//   check, reached directly by the services-mode replay in src/llama-context.cpp /
//   common/speculative.cpp, which has no llama_ubatch to inspect -- ubatch_n_seqs_unq and
//   used_fallback are reported as -1 there, meaning "not applicable").
// n_tok / n_blocks: the exact values the divisibility check failed on.
// ubatch_n_tokens: g.ubatch.n_tokens, logged alongside n_tok (g.res->t_logits->ne[1]) so a
//   divergence between the two -- which would mean the LM head output rows don't match
//   the ubatch's own token count -- shows up directly instead of being assumed away.
// ubatch_n_seqs_unq: the raw (possibly wrong) ubatch.n_seqs_unq field, for comparison
//   against the corrected n_blocks.
// used_fallback: 1 if the per-token seq_id scan came back empty and this call fell back
//   to trusting ubatch.n_seqs_unq (the synthetic graph_reserve()-ubatch case); 0 if the
//   scan found real per-token sequence data; -1 = not applicable (standalone site).
static void llama_dspark_markov_ragged_skipped_inc(
        const char * site,
        int64_t      n_tok,
        int64_t      n_blocks,
        int64_t      ubatch_n_tokens,
        int64_t      ubatch_n_seqs_unq,
        int          used_fallback) {
    g_dspark_markov_ragged_skipped.fetch_add(1, std::memory_order_relaxed);

    int32_t budget = g_dspark_ragged_log_budget.load(std::memory_order_relaxed);
    while (budget > 0 &&
           !g_dspark_ragged_log_budget.compare_exchange_weak(budget, budget - 1, std::memory_order_relaxed)) {
        // retry with the freshly observed `budget`
    }
    if (budget > 0) {
        LLAMA_LOG_WARN("%s: DSPARK_RAGGED site=%s n_tok=%" PRId64 " n_blocks=%" PRId64
                       " ubatch.n_tokens=%" PRId64 " ubatch.n_seqs_unq=%" PRId64 " used_fallback=%d\n",
                       __func__, site, n_tok, n_blocks, ubatch_n_tokens, ubatch_n_seqs_unq, used_fallback);
    }
}

int64_t llama_dspark_markov_ragged_skipped_fetch_reset(void) {
    return g_dspark_markov_ragged_skipped.exchange(0, std::memory_order_relaxed);
}

bool llama_dspark_build_markov_graph(
        ggml_context      * ctx0,
        const llama_model & model,
        ggml_tensor       * tokens,    // I32 [n_tok]
        ggml_tensor       * base,      // F32 [n_vocab, n_tok]
        ggml_tensor       * conf_inp,  // F32 [n_embd, n_tok]
        int64_t             n_blocks,
        ggml_tensor      ** out_logits,
        ggml_tensor      ** out_conf) {
    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && model.dspark_conf_proj && "DSpark markov/confidence weights not loaded");

    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    // MAD-LAB: use the parsed arch-prefixed value, with a sidecar fallback.
    int64_t block_size = model.dflash_block_size;
    if (block_size == 0) {
        const auto it = model.gguf_kv.find("dflash.block_size");
        if (it != model.gguf_kv.end()) {
            block_size = std::stoi(it->second);
        }
    }
    GGML_ASSERT(block_size > 0 && "DSpark draft requires a valid block_size in GGUF metadata");
    // MAD-LAB: end

    // bonus anchor (SpecForge exports): slot 0 is a bonus token, not a prediction slot
    const auto it_anchor          = model.gguf_kv.find("dflash.sample_from_anchor");
    const bool sample_from_anchor = it_anchor == model.gguf_kv.end() || it_anchor->second == "true";
    const int64_t i_draft_beg    = sample_from_anchor ? 0 : 1;

    // MAD-LAB / multi-sequence-safe: this head assumes the ubatch holds exactly one
    // equal-width DSpark block per drafting sequence -- true when draft() (common/
    // speculative.cpp) builds every active sequence's block with the same width AND the
    // whole batch survives as one ubatch. That second half is not guaranteed once
    // --parallel > 1: llama_kv_cache::init_batch() picks split_simple() whenever this
    // context runs a single unified KV stream (n_stream == 1, the common multi-slot
    // config), and split_simple() slices strictly by raw token position with no
    // awareness of sequence boundaries. If n_ubatch is small enough to fall inside a
    // block, one ubatch can end up holding a partial block from one sequence plus a few
    // leading tokens of the next, and n_blocks (the caller's g.ubatch.n_seqs_unq) no
    // longer divides n_tok evenly. There is no way to recover the chained-Markov bias
    // for a torn block from inside a single graph build (the missing block positions
    // simply are not present here), so bail out the same way dsv4_build_dspark_head()
    // (src/models/deepseek4.cpp) already does for the identical shape mismatch: skip
    // biasing and let the caller keep `base` unbiased, instead of aborting the process
    // (previously a GGML_ASSERT here -- see the DS4-Flash --parallel 2 crash at this
    // line, common/speculative.cpp's draft() batches every drafting sequence's block
    // into one shared decode).
    //
    // n_blocks == 1 (the --parallel 1 / single-sequence case) always divides evenly, so
    // this bailout is unreachable there and the computation below is unchanged.
    //
    // This should be unreachable for a real drafting decode: the draft context's
    // constructor (common/speculative.cpp) hard-requires n_ubatch >= n_seq*block_width so
    // split_simple() can never tear a block across ubatches. If it fires anyway, count it
    // -- build_dspark_markov_head() (below) forces the confidence channel to an explicit,
    // honest 0 for this call rather than leaving it unset/stale, and the standalone
    // (services-mode) caller drops the whole draft round; neither path serves a biased
    // logit next to a garbage confidence.
    if (n_blocks <= 0 || n_tok % n_blocks != 0) {
        llama_dspark_markov_ragged_skipped_inc("standalone", n_tok, n_blocks, /*ubatch_n_tokens=*/-1,
                                               /*ubatch_n_seqs_unq=*/-1, /*used_fallback=*/-1);
        return false;
    }
    // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return false;
    }

    // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    if (!sample_from_anchor) {
        // bonus anchor slot: pass the logits through unbiased, pad the (unread) confidence column
        cat      = ggml_cont(ctx0, ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, 0));
        cat_conf = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, base, 1, n_blocks, base_stride, 0)));
    }

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    for (int64_t i = i_draft_beg; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev);   // [R, n_blocks]
        ggml_tensor * bias    = ggml_mul_mat(ctx0, w2, w1_prev); // [n_vocab_draft, n_blocks]
        if (model.d2t) {
            // reduced draft vocab: scatter the bias to the target rows (base is -inf on the others)
            const int64_t n_draft_vocab = bias->ne[0];
            ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_blocks), 0.0f);
            bias = ggml_set_rows(ctx0, full,
                    ggml_reshape_3d(ctx0, bias,      1,             n_draft_vocab, n_blocks),
                    ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
            bias = ggml_reshape_2d(ctx0, bias, n_vocab, n_blocks);
        }

        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
        ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
        ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), w1_prev, 0);
        ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
        if (model.dspark_conf_proj_b) {
            conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
        }
        conf = ggml_sigmoid(ctx0, conf);

        cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);

        // note: returned as [1, n_tok]. The in-graph wrapper broadcasts to n_embd-wide
        // rows so it can reuse `llama_get_embeddings_nextn`; the standalone path wants
        // the compact form and would otherwise read back n_embd copies of every value.
        *out_conf = conf;
    }

    *out_logits = out;

    return true;
}

// MAD-LAB / multi-sequence-safe: the number of sequences that actually own a token in
// this ubatch, i.e. the number of DSpark blocks really present here -- computed directly
// from the per-token seq_id assignment rather than trusted from ubatch.n_seqs_unq.
//
// ubatch.n_seqs_unq is *supposed* to be exactly this (llama_batch_allocr::ubatch_add(),
// src/llama-batch.cpp, derives it from a bitset of the seq_id values actually seen on the
// tokens it copied in). But live-rig evidence at --parallel 2 (2026-08-2x, DS4-Flash
// spine, n_ubatch=2048 so tearing is provably not the cause) shows the two disagreeing:
// with exactly ONE sequence drafting, n_tok never divided evenly by ubatch.n_seqs_unq,
// on every single draft() call -- consistent with n_seqs_unq reporting the 2-slot
// context/cache width rather than the 1 sequence whose tokens are actually in this
// specific decode. The exact upstream mechanism producing that stale/wrong count is not
// yet isolated, but this head does not need to trust it: recomputing straight from
// ubatch.seq_id is self-correcting regardless of the cause, and it is the more honest
// question anyway -- not "how many sequences does the allocr/cache believe are live" but
// "how many sequences' block tokens are in THIS ubatch".
//
// This is not just a crash-avoidance fix: had n_blocks (wrongly) still divided n_tok
// evenly by coincidence (e.g. an 8-token single-sequence block miscounted as 2 blocks of
// 4), llama_dspark_build_markov_graph() would have computed WRONG strided views --
// treating the single sequence's own later positions as a second block's anchor -- and
// silently corrupted the Markov chain/confidence without ever tripping the ragged check
// at all. Recomputing here fixes that class of error too, not only the assert/skip case.
static int64_t dspark_count_seqs_with_tokens(const llama_ubatch & ubatch, bool * out_used_fallback = nullptr) {
    std::unordered_set<llama_seq_id> seen;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            seen.insert(ubatch.seq_id[i][s]);
        }
    }
    if (!seen.empty()) {
        if (out_used_fallback) {
            *out_used_fallback = false;
        }
        return (int64_t) seen.size();
    }

    if (out_used_fallback) {
        *out_used_fallback = true;
    }

    // MAD-LAB / multi-sequence-safe: the scan above found nothing, which is NOT the same
    // thing as "zero sequences drafted". llama_batch_allocr::ubatch_reserve()
    // (src/llama-batch.cpp:400) builds synthetic memory-sizing probe ubatches for
    // graph_reserve()/sched_reserve() -- run once per distinct shape this context hasn't
    // built a graph for yet, which includes the first few real draft() calls before the
    // shape settles (matches the live evidence: ragged fires a handful of times at
    // startup, then goes flat for hundreds of real calls). ubatch_reserve() leaves
    // n_seq_id/seq_id value-initialized (all 0 / all null) -- it only sets
    // seq_id_unq/n_seqs_unq, deliberately, because a reserve build's outputs are never
    // read. A real drafting ubatch always has n_seq_id[i] >= 1 for every token
    // (common_batch_add() never emits an empty seq_id list), so an empty `seen` here can
    // only mean this is one of those synthetic probes, not an empty real one.
    //
    // Reserve ubatches ARE always well-formed by construction: ubatch_reserve() builds
    // exactly n_seq_tokens*n_seqs tokens, so n_tokens is trivially divisible by n_seqs.
    // ubatch.n_seqs_unq is authoritative for them (it IS n_seqs, set directly, not
    // derived from per-token data) -- fall back to it only in this specific,
    // unambiguous case, not for real drafting ubatches (see the long comment above,
    // where n_seqs_unq was shown to be the unreliable one).
    return (int64_t) ubatch.n_seqs_unq;
}

// In-graph wrapper: used by the DFlash/DSV4 decoders when the LM head IS reachable
// from the draft context, so `base` is res->t_logits produced upstream in the same
// graph and `conf_inp` is res->t_embd. Keeps one implementation of the head shared
// with the standalone (services-mode) path.
static void build_dspark_markov_head(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    // MAD-LAB / multi-sequence-safe: root-caused 2026-08-24 via the DSPARK_RAGGED
    // instrumentation below -- the 4 residual ragged events (site=in-graph,
    // used_fallback=0, n_tok=31, ubatch.n_tokens=32, ubatch.n_seqs_unq=2) were NOT real
    // drafting calls and NOT a tearing/miscounting problem at all. They were the "pp"
    // worst-case buffer-sizing reserve pass (sched_reserve(), src/llama-context.cpp,
    // called twice at startup -- hence 4 = 2 call sites x 2 sched_reserve() runs,
    // matching the two observed timestamp pairs). draft_graph_n_tokens() clamps that
    // pass's ubatch to k_draft_graph_tokens=32 tokens over n_seq_max=2 sequences, but
    // reserve_graph_n_outputs() (src/llama-context.cpp) DELIBERATELY requests only 31
    // output rows -- "Keep n_outputs strictly below n_tokens so get_rows stays on", i.e.
    // it intentionally exercises the non-identity build_inp_out_ids() path while sizing
    // compute buffers, by forcing n_outputs = n_tokens-1 whenever the natural cap would
    // otherwise equal n_tokens. That is where res->t_logits ends up with 31 rows against
    // a 32-token, 2-sequence ubatch: 31 is not a divisibility bug, it is n_tokens-1 by
    // deliberate design, and 31 was never going to divide evenly by any n_blocks > 1.
    //
    // A real drafting decode never does this: draft() (common/speculative.cpp) requests
    // logits=true for every token in the block, so n_outputs == n_tokens always there,
    // and build_inp_out_ids() takes the identity path. So "res->t_logits->ne[1] !=
    // ubatch.n_tokens" is a clean, precise signal for "this is the reserve/buffer-sizing
    // probe, not a real block" -- more precise than inferring it from a shape mismatch
    // that (as this exact case proved) can also occur for reasons that have nothing to
    // do with sequence tearing or miscounting.
    //
    // The reserve pass's output is provably never consumed (ggml_backend_sched_reserve
    // only needs the graph's node topology/shapes, not real values), and the OTHER
    // reserve pass in the same sched_reserve() call (tg: graph_reserve(n_seqs, n_seqs,
    // n_seqs, ...) at src/llama-context.cpp -- n_outputs == n_tokens there, not run
    // through reserve_graph_n_outputs at all) already exercises this head with a valid,
    // divisible shape (n_tokens=2, n_seqs_unq=2) for worst-case memory sizing purposes.
    // So there is nothing to size here that isn't already sized elsewhere: skip cleanly,
    // don't touch t_h_nextn (nothing will ever read it for this pass), and don't count
    // it -- this is expected, not a violated invariant.
    if (g.res->t_logits->ne[1] != (int64_t) g.ubatch.n_tokens) {
        return;
    }

    // MAD-LAB / multi-sequence-safe: detect the ragged-block case (a ubatch whose
    // (corrected) block count does not evenly divide its token count -- see the long
    // comment in llama_dspark_build_markov_graph() above) BEFORE calling into the shared
    // head, so we can respond to it differently from that function's *other*
    // false-return reason (block_drafts > block_size, an unrelated, pre-existing,
    // opt-in-only case reached via WP_DS4_CONST_SHAPE that must keep its old "leave
    // everything alone" behavior unchanged for parallel=1 bit-identity).
    //
    // t_h_nextn feeds llama_get_embeddings_nextn(), which conf_min gating and the
    // dispatch/prefetch hint both read as this block's acceptance confidence. Silently
    // `return`-ing here (the old behavior) leaves it whatever it was before this graph was
    // built -- unset, or a stale tensor from a prior ubatch under graph reuse -- so a
    // confidence-gate consumer could accept a torn, unverified block because its garbage
    // confidence happened to read high. That is strictly worse than the crash this used to
    // be. Force an explicit, honest 0 instead: the gate then rejects every draft position
    // this call produced, which is the safe direction to fail in.
    bool used_fallback = false;
    const int64_t n_blocks_chk = dspark_count_seqs_with_tokens(g.ubatch, &used_fallback);
    const int64_t n_tok_chk    = g.res->t_logits->ne[1];
    if (n_blocks_chk <= 0 || n_tok_chk % n_blocks_chk != 0) {
        llama_dspark_markov_ragged_skipped_inc("in-graph", n_tok_chk, n_blocks_chk,
                                               (int64_t) g.ubatch.n_tokens, (int64_t) g.ubatch.n_seqs_unq,
                                               used_fallback ? 1 : 0);

        ggml_tensor * zero_conf = ggml_fill(g.ctx0,
                ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, g.res->t_embd->ne[0], g.res->t_embd->ne[1]),
                0.0f);
        g.res->t_h_nextn = zero_conf;
        ggml_build_forward_expand(g.gf, zero_conf);
        // `base` (g.res->t_logits) is left exactly as the LM head produced it -- unbiased,
        // never stale -- same as every other early-out in this head.
        return;
    }

    ggml_tensor * out  = nullptr;
    ggml_tensor * conf = nullptr;

    if (!llama_dspark_build_markov_graph(g.ctx0, model, tokens,
                g.res->t_logits, g.res->t_embd, n_blocks_chk, &out, &conf)) {
        // Only the block_drafts > block_size case can reach here now (the ragged shape was
        // already handled above) -- unchanged behavior: base stays unbiased, conf is left
        // alone. This is the same pre-existing, opt-in-only (WP_DS4_CONST_SHAPE) early-out
        // this function has always had.
        return;
    }

    // broadcast the [1, n_tok] confidences to n_embd-wide rows to reuse `llama_get_embeddings_nextn`
    conf = ggml_repeat(g.ctx0, conf, g.res->t_embd);

    g.res->t_h_nextn = conf;
    ggml_build_forward_expand(g.gf, conf);

    g.res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    // optional iSWA: pick the matching attention input
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // KV cache injection
    //
    // MAD-LAB: `!ubatch.token`, not just `ubatch.embd`. This dual-mode switch used to key
    // on embd alone, which was unambiguous while an embd batch could only ever be target
    // features for injection -- llama_batch_init(ctx, n_embd_enc, n_seq) allocates embd
    // and leaves token null, so an injection batch never carries ids. A services-mode
    // draft batch carries BOTH: precomputed token embeddings (because this model has no
    // embedding table of its own) AND the ids (because the driver needs them for the
    // out-of-graph Markov head). Without this guard that batch would be misrouted into
    // the injection path and the draft body would never run.
    if (ubatch.embd && !ubatch.token) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, inp_g);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                // rotate K/V into the cache's rotated space
                ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;
                ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;
                if (k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                // rotate K/V into the cache's rotated space
                if (inp_attn->self_k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
                }
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;
        if (!cparams.embeddings_layer_inp.empty() && cparams.embeddings_layer_inp[0]) {
            res->t_layer_inp[0] = inp_g;
        }

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    // MAD-LAB: token embeddings, own table or supplied by the driver.
    //
    // A sidecar DFlash/DSpark GGUF ships no token_embd of its own -- it is trained
    // against the target's embedding space and used to borrow model_other->tok_embd
    // through ctx_other. That cannot work when the target is Meta-split (-sm tensor):
    // the borrowed tensor is pre-allocated in the target's Meta buffer, and a draft
    // scheduler that does not own that buffer type cannot schedule it. Co-scheduling
    // the target's Meta backend into this context was tried and abandoned -- the meta
    // backend's split-state algebra recurses into every src and has no representation
    // for "resident on one foreign device", so mixed meta/simple graphs fail in ways
    // that get progressively harder to detect.
    //
    // Instead the driver gathers the rows through the TARGET context
    // (llama_token_embed_gather) and hands them in on the SAME ubatch that carries the
    // token ids. llm_graph_input_embd::set_input fills `tokens` and `embd` from
    // independent branches, and llama_batch_allocr propagates both independently, so a
    // dual-carry batch is well formed. We still need the token ids: the DSpark Markov
    // head conditions on them, not on the embeddings.
    ggml_tensor * inpL;
    if (model.tok_embd != nullptr) {
        inpL = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    } else {
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);
        inpL = inp->embd;

        // Nothing in THIS graph consumes the ids: the embeddings arrive precomputed and
        // the Markov head that used to read the ids now runs out-of-graph. But an input
        // leaf that never reaches the graph is never seen by ggml-alloc, so it would be
        // handed to llm_graph_input_embd::set_input with a null buffer -- and set_input
        // writes `tokens` for ANY ubatch carrying ids, including the token-only probe
        // decode that common_context_can_seq_rm() runs against this context at load time
        // (tools/server/server-context.cpp). Keep the leaf in the graph so it is still
        // allocated; it stays a dead input, which costs n_tokens*4 bytes.
        //
        // The mirror case is safe by construction: on that probe the ubatch carries no
        // embd, so inp->embd is allocated but unwritten and the body computes from
        // uninitialised memory -- which is fine, because the probe only checks the
        // return code and discards the result.
        ggml_build_forward_expand(gf, inp->tokens);
    }
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, noise_norm);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // Softplus output gate (gated decoder block, e.g. dflash.decoder_arch =
        // "laguna"). Projected from the *pre-attention* hidden state, matching the
        // reference, and applied to the attention output before o_proj -- so wo is
        // deferred out of build_attn when a gate is present.
        ggml_tensor * gate = layer.wqkv_gate
            ? build_lora_mm(layer.wqkv_gate, noise_norm)
            : nullptr;
        if (gate) {
            cb(gate, "attn_gate_proj", il);
        }

        // cache-aware, non-causal attention
        ggml_tensor * wo_deferred = gate ? nullptr : layer.wo;

        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, wo_deferred, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(inp_attn,      wo_deferred, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        if (gate) {
            gate = ggml_softplus(ctx0, gate);
            cb(gate, "attn_gate_softplus", il);

            const int64_t n_gate_tokens = cur->ne[1];

            if (layer.wqkv_gate->ne[1] == n_head) {
                // per-head: broadcast one scalar per head across head_dim
                cur  = ggml_reshape_3d(ctx0, cur,  n_embd_head, n_head, n_gate_tokens);
                gate = ggml_reshape_3d(ctx0, gate, 1,           n_head, n_gate_tokens);
                cur  = ggml_mul(ctx0, cur, gate);
                cur  = ggml_reshape_2d(ctx0, cur, n_embd_head * n_head, n_gate_tokens);
            } else {
                // per-element: gate spans the full attention output
                cur = ggml_mul(ctx0, cur, gate);
            }
            cb(cur, "attn_gated", il);

            cur = build_lora_mm(layer.wo, cur);
            cb(cur, "attn_o_proj", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;
    if (cparams.embeddings_layer_inp[0]) {
        res->t_layer_inp[0] = cur;
        cb(cur, "result_embd_capture", -1);
        ggml_build_forward_expand(gf, cur);
    }

    // MAD-LAB: services mode -- the LM head is not reachable from this context.
    //
    // A sidecar ships no output.weight and used to borrow model_other->output here.
    // Under -sm tensor that tensor lives in the target's Meta buffer, which this
    // scheduler cannot own (see the note on the embedding path above). So graph A
    // ENDS at the post-norm hidden state, which res->t_embd already exposes through
    // the ordinary embeddings output path -- no new export plumbing needed.
    //
    // The driver completes the step in two more calls:
    //   1. llama_output_project(ctx_tgt, hidden, n_tokens)  -> base logits, computed
    //      tensor-parallel on the target where the head actually lives.
    //   2. llama_dspark_markov_head(ctx_dft, base, tokens, hidden, ...) -> final
    //      logits + confidences, using the sidecar's OWN markov weights on this device.
    //
    // Splitting here is cheap because the head is ONE batched mul_mat over all block
    // positions: build_dspark_markov_head consumes res->t_logits only through
    // ggml_view_2d slices, and its argmax chain feeds ggml_get_rows(w1, prev) -- the
    // sidecar's weights -- not the head. So this costs one projection per draft step,
    // not one per block position.
    if (model.output == nullptr) {
        // Export through the NEXTN channel, not the embeddings one.
        //
        // res->t_embd is set just above, but reading it would mean turning on
        // cparams.embeddings, and llm_graph_context::build_pooling() gates ONLY on that
        // flag -- not on pooling_type -- so enabling it makes pooling run on this arch's
        // ENCODER graph too, which sets t_h_nextn and deliberately never sets t_embd.
        // That aborts at llama-graph.cpp:4252 on a null pooling input.
        //
        // t_h_nextn is already the channel this arch exports on (see the encoder), it is
        // gated by cparams.embeddings_nextn which the driver already enables, and in
        // services mode it is otherwise unused because the Markov head that used to write
        // it now runs out-of-graph. So the hidden state rides a path that already works.
        res->t_h_nextn = cur;

        ggml_build_forward_expand(gf, cur);
        return;
    }

    ggml_tensor * output = model.output;

    cur = cap_lm_head_rows(cur);
    cur = build_lora_mm(output, cur, model.output_s);
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // DSpark: bias the draft logits with the Markov head
    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
}

// DSV4 DSpark decoder, dual-mode by batch type (see the DFlash decoder above):
//   * embd batch  -> project main_x through each stage's wkv and inject K into the ring cache
//   * token batch -> noise block through 3 full DSV4 stages (hc + MLA + MoE), markov + confidence heads
// MAD-LAB: reuse the sidecar DSV4 graph for stages embedded in the target model.
llama_model_dflash::graph_dsv4::graph_dsv4(const llama_model & model, const llm_graph_params & params,
                                           int stage_base, int n_stages) :
    llama_model_deepseek4::graph(params) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();
    const int n_st = n_stages > 0 ? n_stages : n_layer;

    // KV cache injection: fused target features from the encoder
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_st; ++il) {
            const int il_m = stage_base + il;
            const auto & layer = model.layers[il_m];

            // main-track KV: kv_norm(wkv(main_x)) with rope on the trailing dims, same
            // rope parameters as the uncompressed layers in build_attention_impl
            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il_m);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_rope_set_offset(kv, n_embd_head_nope);
            cb(kv, "kv_injected", il_m);

            if (inp_attn->self_k_rot_swa) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il_m));
        }

        res->t_embd = inp_g;
        if (cparams.embeddings_layer_inp[0]) {
            res->t_layer_inp[0] = inp_g;
        }

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    const int64_t hc = hparams.dsv4_hc_mult;
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_st; ++il) {
        const int il_m = stage_base + il;
        const auto & layer = model.layers[il_m];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * cur = build_hc_pre(inpL,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                &post, &comb, il_m);
        cb(cur, "hc_attn_pre", il_m);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il_m);
        cb(cur, "attn_norm", il_m);

        cur = build_attention(model, inp_attn, cur, inp_pos, il_m);

        inpL = build_hc_post(cur, residual, post, comb, il_m);
        cb(inpL, "hc_attn_post", il_m);

        residual = inpL;
        cur = build_hc_pre(inpL,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                &post, &comb, il_m);
        cb(cur, "hc_ffn_pre", il_m);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il_m);
        cb(cur, "ffn_norm", il_m);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il_m);
        cb(moe_out, "ffn_moe_out", il_m);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il_m);
        cb(ffn_shexp, "ffn_shexp", il_m);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il_m);

        inpL = build_hc_post(cur, residual, post, comb, il_m);
        cb(inpL, "l_out", il_m);
    }

    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    // confidence head input: the reference scores the pre-norm collapsed hidden state
    res->t_embd = cur;
    if (!cparams.embeddings_layer_inp.empty() && cparams.embeddings_layer_inp[0]) {
        res->t_layer_inp[0] = cur;
        cb(cur, "layer_inp", 0);
        ggml_build_forward_expand(gf, cur);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // Slice to output rows BEFORE the vocab projection. Draft only needs logits
    // for the speculative block; prefill injects via the embd path (no logits).
    // Unsliced this is n_vocab * n_tokens * 4 — 1.01 GiB at 129280 x 2048 —
    // for a 3-layer head that never fills those rows.
    {
        ggml_tensor * out_ids = build_inp_out_ids();
        const int64_t nt  = n_tokens;
        const int64_t row = ggml_nelements(cur) / nt;
        GGML_ASSERT(row * nt == ggml_nelements(cur));
        cur = ggml_reshape_2d(ctx0, cur, row, nt);
        cur = ggml_get_rows(ctx0, cur, out_ids);
        cb(cur, "result_out_ids", -1);
    }

    // lm_head from the target model (shared via ctx_other)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = cap_lm_head_rows(cur);
    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (model.dspark_markov_w1) {
        build_dspark_markov_head(*this, model, inp_tokens);
    }
    // MAD-LAB: end
}
