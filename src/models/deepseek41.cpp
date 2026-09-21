#include "models.h"

#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static bool dsv41_has_dspark_head(const llama_model_loader & ml) {
    return ml.get_weight("markov_w1.weight") != nullptr;
}

// V4.1's MTP blocks route over 128 experts while the main stack has 384, and
// the GGUF records only the main count. Read a block's real width off its
// router so the MTP tensors are created at the shape the file actually has.
// Returns n_expert when the router is absent (skeleton / trunk-less loads).
static int64_t dsv41_layer_n_expert(const llama_model_loader & ml, const llama_hparams & hparams, int il) {
    const std::string name = "blk." + std::to_string(il) + ".ffn_gate_inp.weight";
    const ggml_tensor * router = ml.get_tensor_meta(name.c_str());
    if (router != nullptr && router->ne[1] > 0) {
        return router->ne[1];
    }
    return hparams.n_expert;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    // Same MTP-first rule as V4: n_layer() is n_layer_all - n_layer_nextn.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn > 0 && hparams.n_layer_nextn < hparams.n_layer_all) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr && !dsv41_has_dspark_head(ml)) {
            hparams.n_layer_nextn = 0;
        }
    }
    LLAMA_LOG_WARN("%s: head=%s n_layer_nextn=%u n_layer=%u n_layer_all=%u\n", __func__,
            dsv41_has_dspark_head(ml) ? "DSpark" : "MTP",
            hparams.n_layer_nextn, hparams.n_layer(), hparams.n_layer_all);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");
    hparams.n_layer_kv_from_start = hparams.n_layer_all - hparams.n_layer_nextn;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);

    uint32_t n_expert_used_nextn = hparams.n_expert_used();
    ml.get_key(LLM_KV_NEXTN_EXPERT_USED_COUNT, n_expert_used_nextn, false);
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.n_expert_used_arr[il] = n_expert_used_nextn;
    }

    uint32_t n_clamp = hparams.n_layer_all;
    {
        uint32_t n = 0;
        ml.get_arr_n(LLM_KV_SWIGLU_CLAMP_EXP, n, false);
        if (n > 0 && n < hparams.n_layer_all) {
            n_clamp = n;
        }
    }
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, n_clamp);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, n_clamp, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }
    if (n_clamp > 0 && n_clamp < hparams.n_layer_all) {
        for (uint32_t il = n_clamp; il < hparams.n_layer_all; ++il) {
            hparams.swiglu_clamp_exp[il]   = hparams.swiglu_clamp_exp[n_clamp - 1];
            hparams.swiglu_clamp_shexp[il] = hparams.swiglu_clamp_shexp[n_clamp - 1];
        }
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count, false);

    // Single-Pass mHC: no output_hc_*. Keep n_embd_out at n_embd.
    hparams.n_embd_out_impl = 0;

    if (ml.get_tensor_meta("fc.weight") != nullptr) {
        if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false) || target_layer_ids.empty()) {
            throw std::runtime_error("DeepSeek-V4.1 DSpark model has no target_layers metadata");
        }
        hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;
        ml.get_key(LLM_KV_BLOCK_SIZE, dflash_block_size);
        dflash_hc_mult = hparams.dsv4_hc_mult;
    }

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4.1 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);
    // CSA2 uses 0/1/2 (Full / Reindex / Reuse). Do not reuse V4's 0/4/128 check.

    {
        std::vector<uint32_t> engram_ids;
        if (ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS, engram_ids, false) && !engram_ids.empty()) {
            if (engram_ids.size() > LLAMA_MAX_ENGRAM_LAYERS) {
                throw std::runtime_error("DeepSeek-V4.1 engram.layer_ids is longer than LLAMA_MAX_ENGRAM_LAYERS");
            }
            hparams.dsv41_n_engram_layers = (uint32_t) engram_ids.size();
            for (uint32_t i = 0; i < hparams.dsv41_n_engram_layers; ++i) {
                hparams.dsv41_engram_layer_ids[i] = engram_ids[i];
                if (engram_ids[i] < LLAMA_MAX_LAYERS) {
                    hparams.is_engram_impl[engram_ids[i]] = true;
                }
            }
            ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,            hparams.dsv41_engram_n_heads);
            ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,            hparams.dsv41_engram_head_dim);
            ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE,        hparams.dsv41_engram_max_ngram, false);
            ml.get_key(LLM_KV_ENGRAM_VOCAB_SIZE,            hparams.dsv41_engram_vocab_size, false);
            ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,          hparams.dsv41_engram_pad_token_id, false);
            ml.get_key(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE, hparams.dsv41_engram_compressed_vocab_size, false);
        }
    }
    {
        std::vector<uint32_t> kv_src;
        if (ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYER_IDS, kv_src, false) && !kv_src.empty()) {
            if (kv_src.size() > LLAMA_MAX_KV_SOURCES) {
                throw std::runtime_error("DeepSeek-V4.1 kv_source_layer_ids is longer than LLAMA_MAX_KV_SOURCES");
            }
            hparams.dsv41_n_kv_sources = (uint32_t) kv_src.size();
            for (uint32_t i = 0; i < hparams.dsv41_n_kv_sources; ++i) {
                hparams.dsv41_kv_source_layer_ids[i] = kv_src[i];
            }
        }
        std::vector<uint32_t> idx_src;
        if (ml.get_arr(LLM_KV_ATTENTION_INDEX_SOURCE_LAYER_IDS, idx_src, false) && !idx_src.empty()) {
            if (idx_src.size() > LLAMA_MAX_INDEX_SOURCES) {
                throw std::runtime_error("DeepSeek-V4.1 index_source_layer_ids is longer than LLAMA_MAX_INDEX_SOURCES");
            }
            hparams.dsv41_n_index_sources = (uint32_t) idx_src.size();
            for (uint32_t i = 0; i < hparams.dsv41_n_index_sources; ++i) {
                hparams.dsv41_index_source_layer_ids[i] = idx_src[i];
            }
        }
        ml.get_key(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER_ID, hparams.dsv41_candidate_source_layer, false);
        ml.get_key(LLM_KV_ATTENTION_CANDIDATE_TOPK_BLOCKS,     hparams.dsv41_candidate_topk_blocks, false);
        ml.get_key(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,      hparams.dsv41_candidate_block_size, false);
    }

    if (hparams.dsv41_n_engram_layers > 0) {
        engram.max_ngram = hparams.dsv41_engram_max_ngram ? hparams.dsv41_engram_max_ngram : 4;
        engram.n_heads   = hparams.dsv41_engram_n_heads   ? hparams.dsv41_engram_n_heads   : 8;
        engram.n_layers  = hparams.dsv41_n_engram_layers;

        ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,        engram.token_map,   false);
        ml.get_arr(LLM_KV_ENGRAM_HASH_MULTIPLIERS, engram.multipliers, false);
        if (!engram.multipliers.empty() &&
            engram.multipliers.size() != (size_t) engram.n_layers * engram.max_ngram) {
            throw std::runtime_error("DeepSeek-V4.1 engram.hash_multipliers is not [n_engram_layers * max_ngram_size]");
        }
        engram.pad_id = engram.compress((int32_t) hparams.dsv41_engram_pad_token_id);

        // EngramLayout.from_args: per (layer, n-gram size, head) the next prime
        // above engram_vocab_size-1 not handed out yet; the ranges stay disjoint.
        const uint32_t ngram_kinds = engram.max_ngram > 1 ? engram.max_ngram - 1 : 1;
        const int64_t  vocab       = hparams.dsv41_engram_vocab_size ? (int64_t) hparams.dsv41_engram_vocab_size : 1;
        auto is_prime = [](int64_t n) {
            if (n < 2) return false;
            if (n % 2 == 0) return n == 2;
            for (int64_t d = 3; d * d <= n; d += 2) {
                if (n % d == 0) return false;
            }
            return true;
        };
        std::set<int64_t> seen;
        engram.primes.clear();
        engram.offsets.clear();
        for (uint32_t layer = 0; layer < engram.n_layers; ++layer) {
            int64_t offset = 0;
            for (uint32_t ng = 0; ng < ngram_kinds; ++ng) {
                int64_t current = vocab - 1;
                for (uint32_t h = 0; h < engram.n_heads; ++h) {
                    int64_t candidate = current + 1;
                    while (!is_prime(candidate) || seen.count(candidate)) {
                        ++candidate;
                    }
                    seen.insert(candidate);
                    current = candidate;
                    engram.primes.push_back(candidate);
                    engram.offsets.push_back(offset);
                    offset += candidate;
                }
            }
        }
        if (engram.multipliers.empty()) {
            LLAMA_LOG_WARN("%s: engram.hash_multipliers missing from GGUF; Engram hashes will NOT match the reference (reconvert with tokenizer.json present)\n", __func__);
        }
        if (engram.token_map.empty()) {
            LLAMA_LOG_WARN("%s: engram.token_map missing from GGUF; hashing raw token ids\n", __func__);
        }
    }

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4.1 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_FULL;
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only    ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    if (!engram.token_map.empty() && engram.token_map.size() != (size_t) n_vocab) {
        throw std::runtime_error("DeepSeek-V4.1 engram.token_map length does not match the vocab");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // V4.1 Single-Pass mHC does not ship output_hc_*. Leave hc_head_* null.

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            // CSA2 has compressor wkv/gate/norm; no ape, no indexer compressor.
            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, flags | TENSOR_NOT_REQUIRED);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, flags | TENSOR_NOT_REQUIRED);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags | TENSOR_NOT_REQUIRED);
        }

        const int64_t n_embd_indexer = hparams.indexer_head_size;
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags | TENSOR_NOT_REQUIRED);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags | TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, flags | TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd_head, n_embd_indexer}, flags | TENSOR_NOT_REQUIRED);

        if (hparams.is_engram((uint32_t) i)) {
            const int64_t engram_dim = hparams.dsv41_engram_head_dim ? hparams.dsv41_engram_head_dim : 256;
            const std::string embd_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i).str();
            int64_t n_rows = 0;
            int64_t n_cols = engram_dim;
            if (const auto * w = ml.get_weight(embd_name.c_str())) {
                n_cols = w->tensor->ne[0];
                n_rows = w->tensor->ne[1];
            }
            if (n_rows <= 0) {
                throw std::runtime_error("DeepSeek-V4.1 missing Engram table " + embd_name);
            }
            if (engram.ready()) {
                // The only evidence the derived bucket layout is the trained one:
                // the table has exactly sum(primes) rows.
                uint32_t ordinal = 0;
                while (ordinal < engram.n_layers && (int) hparams.dsv41_engram_layer_ids[ordinal] != i) {
                    ++ordinal;
                }
                const size_t per_layer = engram.n_cols();
                int64_t want_rows = 0;
                for (size_t c = 0; ordinal < engram.n_layers && c < per_layer; ++c) {
                    want_rows += engram.primes[ordinal * per_layer + c];
                }
                if (ordinal >= engram.n_layers || want_rows != n_rows) {
                    throw std::runtime_error("DeepSeek-V4.1 Engram table " + embd_name + " has " +
                            std::to_string(n_rows) + " rows but the derived bucket layout needs " +
                            std::to_string(want_rows));
                }
            }
            const int64_t n_hash_cols = std::max<int64_t>(1,
                    (int64_t) (hparams.dsv41_engram_max_ngram ? hparams.dsv41_engram_max_ngram - 1 : 0) *
                    (int64_t) hparams.dsv41_engram_n_heads);
            layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {n_cols, n_rows}, flags | TENSOR_READ_LAZY);
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, flags | TENSOR_NOT_REQUIRED);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, flags | TENSOR_NOT_REQUIRED);
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i),
                    {n_hash_cols * engram_dim, n_embd * (hc_mult + 1)}, flags | TENSOR_NOT_REQUIRED);
        }

        const int64_t n_expert_il = i < n_layer ? n_expert : dsv41_layer_n_expert(ml, hparams, i);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert_il}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {hparams.n_expert_used(i), n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert_il}, flags);
        }
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert_il}, flags | TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        const int expert_flags = routed_experts_external ? TENSOR_SKIP | TENSOR_NOT_REQUIRED : 0;
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert_il}, expert_flags | flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert_il}, expert_flags | flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert_il}, expert_flags | flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, TENSOR_NOT_REQUIRED | flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
            if (i == n_layer_all - 1) {
                layer.nextn.hc_head_fn    = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_FN,    "weight", i), {hc_dim, hc_mult}, TENSOR_NOT_REQUIRED | flags);
                layer.nextn.hc_head_base  = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_BASE,  "weight", i), {hc_mult},         TENSOR_NOT_REQUIRED | flags);
                layer.nextn.hc_head_scale = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_SCALE, "weight", i), {1},               TENSOR_NOT_REQUIRED | flags);
            }
        }
    }

    if (ml.get_tensor_meta("fc.weight") != nullptr) {
        const int64_t n_target = target_layer_ids.size();
        fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), {n_target*n_embd, n_embd}, 0);
        output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);

        const auto * markov = ml.get_tensor_meta(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight").str().c_str());
        GGML_ASSERT(markov != nullptr);
        const int64_t rank = markov->ne[0];
        dspark_markov_w1   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {rank, n_vocab}, 0);
        dspark_markov_w2   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), {rank, n_vocab}, 0);
        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), {n_embd + rank, 1}, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   {1}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

static int32_t ds41_n_expert_used(const llama_hparams & hparams, int il) {
    const int32_t trained = (int32_t) hparams.n_expert_used(il >= 0 ? (uint32_t) il : 0);
    static const int32_t override = [] {
        const char * e = std::getenv("WP_N_EXPERT_USED");
        if (e == nullptr || e[0] == '\0') {
            return 0;
        }
        return std::atoi(e);
    }();
    if (override > 0 && override < trained) {
        return override;
    }
    return trained;
}

static float dsv41_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }
    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek41::graph::build_dspark_encoder(const llama_model & model) {
    const int64_t n_target = model.target_layer_ids.size();
    auto inp = std::make_unique<llm_graph_input_embd>(n_target*n_embd);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, inp->n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * cur = build_lora_mm(model.fc, inp->embd);
    cur = build_norm(cur, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    res->t_h_nextn = cur;
    if (!cparams.embeddings_layer_inp.empty() && cparams.embeddings_layer_inp[0]) {
        res->t_layer_inp[0] = cur;
    }

    res->add_input(std::move(inp));
    ggml_build_forward_expand(gf, cur);
}

void llama_model_deepseek41::graph::build_dspark_stages(const llama_model & model) {
    const int stage_base = hparams.n_layer();
    const int n_stages = hparams.n_layer_nextn;
    GGML_ASSERT(n_stages == 3 && "DeepSeek-V4.1 DSpark expects three stages");

    {
        const char * e = std::getenv("WP_DISPATCH_SPLIT_SHEXP");
        const bool split_on = (e == nullptr) || (e[0] != '0');
        moe_dispatch_split_shexp = (expert_dispatch != nullptr) && split_on;
    }

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);
        ggml_tensor * inp_g = inp->embd;
        res->add_input(std::move(inp));

        // Split encoder contract: the input is already fc + output_norm_enc encoded.
        for (int il = 0; il < n_stages; ++il) {
            const int il_m = stage_base + il;
            const auto & layer = model.layers[il_m];

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
            ggml_build_forward_expand(gf,
                    inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il_m));
        }

        res->t_embd = inp_g;
        if (!cparams.embeddings_layer_inp.empty() && cparams.embeddings_layer_inp[0]) {
            res->t_layer_inp[0] = inp_g;
        }
        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr &&
                "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    ggml_tensor * inp_tokens = inp->tokens;
    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    cb(inpL, "inp_noise_embd", -1);
    res->add_input(std::move(inp));

    const int64_t hc = hparams.dsv4_hc_mult;
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * pre_mix = identity_pre_mix();
    cb(pre_mix, "hc_pre_mix_init", -1);

    for (int il = 0; il < n_stages; ++il) {
        const int il_m = stage_base + il;
        const auto & layer = model.layers[il_m];

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * attn_post = nullptr;
        ggml_tensor * attn_comb = nullptr;
        build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &attn_pre, &attn_post, &attn_comb, il_m);

        ggml_tensor * cur = build_hc_pre(inpL, pre_mix, il_m);
        cb(cur, "hc_attn_pre", il_m);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il_m);
        cb(cur, "attn_norm", il_m);
        cur = build_attention(model, inp_attn, cur, inp_pos, il_m);

        inpL = build_hc_post(cur, residual, attn_post, attn_comb, il_m);
        cb(inpL, "hc_attn_post", il_m);

        residual = inpL;
        ggml_tensor * ffn_pre = nullptr;
        ggml_tensor * ffn_post = nullptr;
        ggml_tensor * ffn_comb = nullptr;
        build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &ffn_pre, &ffn_post, &ffn_comb, il_m);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_post);
        ggml_build_forward_expand(gf, ffn_comb);

        cur = build_hc_pre(inpL, attn_pre, il_m);
        cb(cur, "hc_ffn_pre", il_m);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il_m);
        cb(cur, "ffn_norm", il_m);

        const int32_t n_expert_layer = (int32_t) layer.ffn_gate_inp->ne[1];
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                layer.ffn_exp_probs_b, n_expert_layer, ds41_n_expert_used(hparams, il_m),
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, il_m);
        cb(moe_out, "ffn_moe_out", il_m);

        ggml_tensor * ffn_shexp = build_ffn(shexp_after_issue(cur, il_m),
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il_m);
        cb(ffn_shexp, "ffn_shexp", il_m);

        cur = complete_moe_dispatch(moe_out, ffn_shexp, il_m);
        inpL = build_hc_post(cur, residual, ffn_post, ffn_comb, il_m);
        inpL = build_cvec(inpL, il_m);
        cb(inpL, "l_last", il_m);
        pre_mix = ffn_pre;
    }

    ggml_tensor * cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_collapse", -1);
    res->t_embd = cur;
    if (!cparams.embeddings_layer_inp.empty() && cparams.embeddings_layer_inp[0]) {
        res->t_layer_inp[0] = cur;
        cb(cur, "layer_inp", 0);
        ggml_build_forward_expand(gf, cur);
    }

    const auto & last = model.layers[hparams.n_layer_all - 1];
    cur = build_norm(cur, last.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    ggml_tensor * out_ids = build_inp_out_ids();
    const int64_t nt = n_tokens;
    const int64_t row = ggml_nelements(cur) / nt;
    GGML_ASSERT(row * nt == ggml_nelements(cur));
    cur = ggml_reshape_2d(ctx0, cur, row, nt);
    cur = ggml_get_rows(ctx0, cur, out_ids);
    cb(cur, "result_out_ids", -1);

    auto * output = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr &&
                "DSpark decoder requires the target model's output projection");
        output = model_other->output;
        output_s = model_other->output_s;
    }

    cur = cap_lm_head_rows(cur);
    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

struct llm_graph_input_csa2 : public llm_graph_input_i {
    struct src_io {
        int      il     = -1;
        uint32_t ratio  = 1;
        uint32_t n_latents = 1;
        uint32_t ring_size = 0;
        ggml_tensor * ring_write_idxs = nullptr; // I64 [n_tokens]
        ggml_tensor * ring_read_idxs  = nullptr; // I32 [ratio*n_latents]
        ggml_tensor * write_idxs = nullptr; // I64 [n_latents]
        ggml_tensor * group_pos  = nullptr; // I32 [n_latents]
        ggml_tensor * pool_idxs = nullptr; // I64 [n_tokens]
        ggml_tensor * pool_mask = nullptr; // F32 [1, ratio*n_latents]
        ggml_tensor * kq_mask    = nullptr; // [n_csa, n_tokens, 1, 1]
        uint32_t candidate_block_size = 0;
        ggml_tensor * candidate_pin = nullptr; // F32 [n_blocks, n_tokens]
    };

    std::vector<src_io> sources;
    uint32_t n_csa = 0;

    const src_io * find(int il) const {
        for (const auto & s : sources) {
            if (s.il == il) {
                return &s;
            }
        }
        return nullptr;
    }

    void set_input(const llama_ubatch * ubatch) override {
        if (ubatch == nullptr || ubatch->pos == nullptr) {
            return;
        }
        const uint32_t n_tok = ubatch->n_tokens;
        if (n_tok == 0) {
            return;
        }
        for (const auto & s : sources) {
            // Single-sequence ubatches with contiguous positions are assumed here.
            const int32_t pos0 = ubatch->pos[0];
            const int64_t ratio = s.ratio > 0 ? (int64_t) s.ratio : 1;
            const int64_t base = pos0 / ratio;
            const int64_t last_pos = (int64_t) pos0 + n_tok - 1;
            if (s.ring_write_idxs && s.ring_write_idxs->data) {
                int64_t * idxs = (int64_t *) s.ring_write_idxs->data;
                const int64_t ring_size = s.ring_size > 0 ? (int64_t) s.ring_size : 1;
                for (uint32_t i = 0; i < n_tok && i < (uint32_t) s.ring_write_idxs->ne[0]; ++i) {
                    idxs[i] = (int64_t) ubatch->pos[i] % ring_size;
                }
            }
            if (s.ring_read_idxs && s.ring_read_idxs->data) {
                int32_t * idxs = (int32_t *) s.ring_read_idxs->data;
                const int64_t ring_size = s.ring_size > 0 ? (int64_t) s.ring_size : 1;
                for (uint32_t g = 0; g < s.n_latents; ++g) {
                    for (uint32_t i = 0; i < s.ratio; ++i) {
                        const int64_t pos = (base + g) * ratio + i;
                        idxs[g*s.ratio + i] = pos < pos0 ? (int32_t) (pos % ring_size) : 0;
                    }
                }
            }
            if (s.write_idxs && s.write_idxs->data) {
                int64_t * wr = (int64_t *) s.write_idxs->data;
                for (uint32_t i = 0; i < s.n_latents; ++i) {
                    wr[i] = base + i;
                }
                if (s.group_pos && s.group_pos->data) {
                    int32_t * gp = (int32_t *) s.group_pos->data;
                    for (uint32_t i = 0; i < s.n_latents; ++i) {
                        gp[i] = (int32_t) (wr[i] * ratio);
                    }
                }
            }
            if (s.pool_idxs && s.pool_idxs->data) {
                int64_t * idxs = (int64_t *) s.pool_idxs->data;
                for (uint32_t i = 0; i < n_tok && i < (uint32_t) s.pool_idxs->ne[0]; ++i) {
                    const int64_t pos = ubatch->pos[i];
                    idxs[i] = (pos / ratio - base) * ratio + pos % ratio;
                }
            }
            if (s.pool_mask && s.pool_mask->data) {
                float * mask = (float *) s.pool_mask->data;
                for (uint32_t g = 0; g < s.n_latents; ++g) {
                    for (uint32_t i = 0; i < s.ratio; ++i) {
                        const int64_t pos = (base + g) * ratio + i;
                        const bool in_ubatch = pos >= pos0 && pos <= last_pos;
                        const bool in_state = g == 0 && pos < pos0;
                        const bool dummy = g + 1 == s.n_latents && !in_ubatch && !in_state;
                        mask[g*s.ratio + i] = in_ubatch || in_state || dummy ? 0.0f : -INFINITY;
                    }
                }
            }
            if (s.kq_mask && s.kq_mask->data) {
                const int64_t n_kv = s.kq_mask->ne[0];
                const int64_t n_q  = s.kq_mask->ne[1];
                if (s.kq_mask->type == GGML_TYPE_F32) {
                    float * dst = (float *) s.kq_mask->data;
                    for (int64_t q = 0; q < n_q; ++q) {
                        const int32_t pos = q < n_tok ? ubatch->pos[q] : -1;
                        for (int64_t j = 0; j < n_kv; ++j) {
                            const bool vis = pos >= 0 && s.ratio > 0 &&
                                    ((int32_t) ((j + 1) * s.ratio) - 1) <= pos;
                            dst[q*n_kv + j] = vis ? 0.0f : -INFINITY;
                        }
                    }
                }
            }
            if (s.candidate_pin && s.candidate_pin->data) {
                const int64_t n_blocks = s.candidate_pin->ne[0];
                const int64_t n_q = s.candidate_pin->ne[1];
                const int64_t block_size = s.candidate_block_size > 0 ?
                        (int64_t) s.candidate_block_size : 1;
                float * pin = (float *) s.candidate_pin->data;
                std::fill(pin, pin + n_blocks*n_q, 0.0f);
                for (int64_t q = 0; q < n_q && q < (int64_t) n_tok; ++q) {
                    const int64_t pos = ubatch->pos[q];
                    const int64_t compress_len = pos >= 0 ? (pos + 1) / ratio : 0;
                    const int64_t last = (compress_len - 1) / block_size;
                    if (last >= 0 && last < n_blocks) {
                        pin[q*n_blocks + last] = 1e30f;
                    }
                }
            }
        }
    }
};

struct llm_graph_input_engram : public llm_graph_input_i {
    // [n_cols, n_tokens, n_engram_layers] I32 -- one contiguous [n_cols*n_tokens]
    // block per engram layer (ordinal order), so build_engram can take a 1-D view.
    ggml_tensor * hash_ids = nullptr;
    const llama_model_deepseek41::engram_hasher * hasher = nullptr;
    std::vector<int32_t> hist; // compressed tokens by position, seq 0

    void set_input(const llama_ubatch * ubatch) override {
        if (hash_ids == nullptr || hash_ids->data == nullptr || ubatch == nullptr || hasher == nullptr) {
            return;
        }
        const int64_t n_cols   = hash_ids->ne[0];
        const int64_t n_tok    = hash_ids->ne[1];
        const int64_t n_layers = hash_ids->ne[2];
        int32_t * out = (int32_t *) hash_ids->data;
        std::fill(out, out + n_cols * n_tok * n_layers, 0);

        if (ubatch->token == nullptr || ubatch->pos == nullptr) {
            return;
        }
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            const int32_t pos = ubatch->pos[i];
            if (pos < 0) {
                continue;
            }
            if ((size_t) pos >= hist.size()) {
                hist.resize((size_t) pos + 1, hasher->pad_id);
            }
            hist[(size_t) pos] = hasher->compress((int32_t) ubatch->token[i]);
        }

        for (int64_t t = 0; t < n_tok && t < (int64_t) ubatch->n_tokens; ++t) {
            for (int64_t layer = 0; layer < n_layers; ++layer) {
                hasher->hash_position(hist, ubatch->pos[t], (uint32_t) layer, out + (layer * n_tok + t) * n_cols);
            }
        }
    }
};

// NgramHashState.forward: tokens[shift] = compressed id at pos-shift (pad once
// the lookback runs off the start); products = tokens * multipliers[layer];
// rolling XOR over lookbacks so the value after step i is the (i+1)-gram
// hash; each (n-gram, head) lands in its own prime-sized bucket range at its
// offset. Products cannot overflow: multipliers are bounded by
// INT64_MAX / compressed_vocab / 2.
void llama_model_deepseek41::engram_hasher::hash_position(
        const std::vector<int32_t> & hist, int32_t pos, uint32_t layer, int32_t * row) const {
    const uint32_t ngram_kinds = max_ngram > 1 ? max_ngram - 1 : 1;
    const bool     have_mult   = !multipliers.empty();
    int64_t products[16];
    GGML_ASSERT(max_ngram <= 16);
    for (uint32_t shift = 0; shift < max_ngram; ++shift) {
        const int32_t src_pos = pos - (int32_t) shift;
        int32_t tok = pad_id;
        if (src_pos >= 0 && (size_t) src_pos < hist.size()) {
            tok = hist[(size_t) src_pos];
        }
        const int64_t mult = have_mult ? multipliers[(size_t) layer * max_ngram + shift] : 1;
        products[shift] = (int64_t) tok * mult;
    }
    int64_t rolling = products[0];
    uint32_t col = 0;
    for (uint32_t i = 1; i < max_ngram; ++i) {
        rolling ^= products[i];
        for (uint32_t h = 0; h < n_heads; ++h, ++col) {
            const size_t k = ((size_t) layer * ngram_kinds + (i - 1)) * n_heads + h;
            const int64_t prime = primes[k];
            int64_t id = rolling % prime;
            if (id < 0) {
                id += prime;
            }
            row[col] = (int32_t) (id + offsets[k]);
        }
    }
}

bool llama_model_deepseek41::graph::is_kv_source(int il) const {
    for (uint32_t i = 0; i < hparams.dsv41_n_kv_sources; ++i) {
        if ((int32_t) hparams.dsv41_kv_source_layer_ids[i] == il) {
            return true;
        }
    }
    return false;
}

bool llama_model_deepseek41::graph::is_index_source(int il) const {
    for (uint32_t i = 0; i < hparams.dsv41_n_index_sources; ++i) {
        if ((int32_t) hparams.dsv41_index_source_layer_ids[i] == il) {
            return true;
        }
    }
    return false;
}

int32_t llama_model_deepseek41::graph::kv_source_for(int il) const {
    int32_t src = -1;
    for (uint32_t i = 0; i < hparams.dsv41_n_kv_sources; ++i) {
        const int32_t s = (int32_t) hparams.dsv41_kv_source_layer_ids[i];
        if (s <= il) {
            src = s;
        }
    }
    return src;
}

ggml_tensor * llama_model_deepseek41::graph::identity_pre_mix() const {
    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * ones = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ones = ggml_fill(ctx0, ones, 1.0f);
    if (hc <= 1) {
        return ones;
    }
    ggml_tensor * zeros = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    return ggml_concat(ctx0, ones, zeros, 0);
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * hash_ids,
        int il) const {
    const auto & layer = model.layers[il];
    if (layer.engram_embd == nullptr || layer.engram_wkv == nullptr || hash_ids == nullptr) {
        return x;
    }

    const int64_t hc          = hparams.dsv4_hc_mult;
    const int64_t n_hash_cols = hash_ids->ne[0];
    const int64_t head_dim    = layer.engram_embd->ne[0];
    const int64_t nt          = n_tokens;

    int64_t ordinal = 0;
    while (ordinal < (int64_t) hparams.dsv41_n_engram_layers && (int) hparams.dsv41_engram_layer_ids[ordinal] != il) {
        ++ordinal;
    }
    GGML_ASSERT(ordinal < hash_ids->ne[2] && "engram layer has no hash block");
    ggml_tensor * ids = ggml_view_1d(ctx0, hash_ids, n_hash_cols * nt, ordinal * hash_ids->nb[2]);
    ggml_tensor * rows = ggml_get_rows(ctx0, layer.engram_embd, ids);
    rows = ggml_reshape_2d(ctx0, rows, n_hash_cols * head_dim, nt);
    cb(rows, "engram_rows", il);

    ggml_tensor * kv = build_lora_mm(layer.engram_wkv, rows);
    cb(kv, "engram_wkv", il);

    // kv rows are [n_embd*hc | n_embd] per token: view the key and value halves
    // as strided 3-D tensors (a 2-D view + reshape is only contiguous at nt == 1);
    // the casts below materialise them contiguously.
    ggml_tensor * key   = ggml_view_3d(ctx0, kv, n_embd, hc, nt, n_embd * kv->nb[0], kv->nb[1], 0);
    ggml_tensor * value = ggml_view_3d(ctx0, kv, n_embd, 1,  nt, n_embd * kv->nb[0], kv->nb[1], n_embd * hc * kv->nb[0]);

    ggml_tensor * h = ggml_cast(ctx0, x,   GGML_TYPE_F32);
    ggml_tensor * k = ggml_cast(ctx0, key, GGML_TYPE_F32);
    ggml_tensor * v = ggml_cast(ctx0, value, GGML_TYPE_F32);

    ggml_tensor * prod = ggml_mul(ctx0, h, k);
    if (layer.engram_q && layer.engram_k) {
        ggml_tensor * weight = ggml_mul(ctx0, layer.engram_q, layer.engram_k);
        weight = ggml_reshape_3d(ctx0, weight, n_embd, hc, 1);
        prod = ggml_mul(ctx0, prod, weight);
    }

    const float eps = hparams.f_norm_rms_eps;
    ggml_tensor * h_ms = ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, h)), 1.0f / (float) n_embd);
    ggml_tensor * k_ms = ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, k)), 1.0f / (float) n_embd);
    // sqrt((h_ms + eps) * (k_ms + eps)) without scalar constant tensors: the
    // graph context is no_alloc, so ggml_new_f32 cannot be used here.
    ggml_tensor * denom = ggml_sqrt(ctx0, ggml_mul(ctx0,
            ggml_scale_bias(ctx0, h_ms, 1.0f, eps),
            ggml_scale_bias(ctx0, k_ms, 1.0f, eps)));

    ggml_tensor * dot = ggml_sum_rows(ctx0, prod);
    dot = ggml_div(ctx0, dot, denom);
    dot = ggml_scale(ctx0, dot, 1.0f / sqrtf((float) n_embd));
    ggml_tensor * mag  = ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, dot), ggml_sqrt(ctx0, mag)));
    gate = ggml_reshape_3d(ctx0, gate, 1, hc, nt);

    // delta[n_embd, hc, nt] = value (shared across the hc streams) * per-stream gate
    ggml_tensor * delta = ggml_mul(ctx0, ggml_repeat(ctx0, v, h), gate);
    ggml_tensor * out = ggml_add(ctx0, h, delta);
    return ggml_cast(ctx0, out, x->type);
}

static ggml_tensor * dsv41_pool_ratio(
        ggml_context * ctx0,
        ggml_tensor * kv,    // [head_dim, n]
        ggml_tensor * score, // [head_dim, n]
        int64_t ratio,
        int64_t n_groups) {
    const int64_t head_dim = kv->ne[0];
    ggml_tensor * kv3 = ggml_reshape_3d(ctx0, kv, head_dim, ratio, n_groups);
    kv3 = ggml_cont(ctx0, ggml_permute(ctx0, kv3, 1, 0, 2, 3));
    ggml_tensor * sc3 = ggml_reshape_3d(ctx0, score, head_dim, ratio, n_groups);
    sc3 = ggml_cont(ctx0, ggml_permute(ctx0, sc3, 1, 0, 2, 3));
    ggml_tensor * kv2 = ggml_reshape_2d(ctx0, kv3, ratio, head_dim * n_groups);
    ggml_tensor * sc2 = ggml_reshape_2d(ctx0, sc3, ratio, head_dim * n_groups);
    ggml_tensor * w = ggml_soft_max(ctx0, sc2);
    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, kv2, w));
    return ggml_reshape_2d(ctx0, pooled, head_dim, n_groups);
}

// DSpark layer-input taps collapse the hyper-connection residual
// [n_embd, hc, n_tokens] to ONE [n_embd, n_tokens] row by a plain mean over the
// hc copies (reference: Transformer.forward does h.mean(dim=2)). dsv4_hc_mean in
// deepseek4.cpp is file-static, so the equivalent lives here. The result is a
// real op (adds + scale), NOT a bare ggml_reshape_2d view: pure views often have
// no sched backend and extract_layer_inputs asserts
// ggml_backend_sched_get_tensor_backend != null.
static ggml_tensor * dsv41_hc_mean(ggml_context * ctx0, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx0, acc, 1.0f/hc);
}

ggml_tensor * llama_model_deepseek41::graph::build_attention_csa2(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group * n_embd_head;
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = ratio != 0;
    const float freq_base_l    = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l   = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l   = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l  = dsv41_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l    = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l    = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = ggml_rope_set_offset(q, n_embd_head_nope);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);
    ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = inp_attn->mctx->get_k(ctx0, il);
    ggml_tensor * k_all = raw_k;
    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    const int32_t src_il = kv_source_for(il);
    const csa2_src_io * src_io = nullptr;
    for (const auto & s : csa2_sources) {
        if (s.il == src_il) {
            src_io = &s;
            break;
        }
    }

    if (ratio > 0 && src_il >= 0 && src_io != nullptr && layer.attn_comp_wkv != nullptr) {
        auto * csa_ctx = inp_dsv4->mctx->get_csa();
        auto * grp     = inp_dsv4->mctx->get_csa_state();

        if (is_kv_source(il) && grp != nullptr) {
            ggml_tensor * comp_in = cur;
            ggml_tensor * lat_kv = build_lora_mm(layer.attn_comp_wkv, comp_in);
            auto write_lat = [&](ggml_tensor * lat, int64_t n_lat) {
                // Indexer keys are derived from the pre-RoPE latent (official Compressor).
                if (layer.indexer_attn_k) {
                    ggml_tensor * ik = build_lora_mm(layer.indexer_attn_k, lat);
                    if (layer.indexer_k_norm) {
                        ik = build_norm(ik, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
                    }
                    const int64_t idx_head = hparams.indexer_head_size;
                    const int64_t idx_rope = std::min(n_embd_head_rope, idx_head);
                    ik = ggml_reshape_3d(ctx0, ik, idx_head, 1, n_lat);
                    ik = ggml_rope_ext(ctx0, ik, src_io->group_pos, nullptr, (int) idx_rope, rope_type, n_ctx_orig_l,
                            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
                    ik = ggml_rope_set_offset(ik, idx_head - idx_rope);
                    ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0, ik, src_io->write_idxs, il));
                }
                lat = ggml_reshape_3d(ctx0, lat, n_embd_head, 1, n_lat);
                if (src_io->group_pos) {
                    lat = ggml_rope_ext(ctx0, lat, src_io->group_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
                    lat = ggml_rope_set_offset(lat, n_embd_head_nope);
                }
                ggml_build_forward_expand(gf, csa_ctx->cpy_k(ctx0, lat, src_io->write_idxs, il));
            };
            if (ratio == 1) {
                ggml_tensor * lat = build_norm(lat_kv, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                write_lat(lat, nt);
            } else {
                ggml_tensor * lat_sc = build_lora_mm(layer.attn_comp_wgate, comp_in);
                const int64_t n_latents = src_io->write_idxs->ne[0];
                // Read the ring snapshot before the current ubatch writes its rows.
                ggml_tensor * ring_kv = ggml_cont(ctx0, grp->get_kv(ctx0, il));
                ggml_tensor * ring_sc = ggml_cont(ctx0, grp->get_score(ctx0, il));
                ggml_tensor * pool_kv = ggml_get_rows(ctx0, ring_kv, src_io->ring_read_idxs);
                ggml_tensor * pool_sc = ggml_get_rows(ctx0, ring_sc, src_io->ring_read_idxs);
                pool_kv = ggml_set_rows(ctx0, pool_kv, lat_kv, src_io->pool_idxs);
                pool_sc = ggml_set_rows(ctx0, pool_sc, lat_sc, src_io->pool_idxs);
                pool_sc = ggml_add(ctx0, pool_sc, src_io->pool_mask);

                ggml_tensor * lat = dsv41_pool_ratio(ctx0, pool_kv, pool_sc, ratio, n_latents);
                if (layer.attn_comp_norm) {
                    lat = build_norm(lat, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                }
                write_lat(lat, n_latents);

                ggml_build_forward_expand(gf, ggml_set_rows(ctx0,
                        grp->get_kv(ctx0, il), lat_kv, src_io->ring_write_idxs));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx0,
                        grp->get_score(ctx0, il), lat_sc, src_io->ring_write_idxs));
            }
        }

        ggml_tensor * csa_k = csa_ctx->get_k(ctx0, src_il);
        const int64_t n_csa = src_io->kq_mask->ne[0];
        csa_k = ggml_view_4d(ctx0, csa_k,
                csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
                csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
        if (csa_k->type != raw_k->type) {
            csa_k = ggml_cast(ctx0, csa_k, raw_k->type);
        }
        k_all = ggml_concat(ctx0, raw_k, csa_k, 2);

        ggml_tensor * csa_mask = src_io->kq_mask;
        ggml_tensor * top_k = nullptr;
        if (is_index_source(il) && layer.indexer_attn_q_b && layer.indexer_proj) {
            const int64_t idx_head  = hparams.indexer_head_size;
            const int64_t idx_heads = hparams.indexer_n_head;
            const int64_t idx_rope  = std::min(n_embd_head_rope, idx_head);
            ggml_tensor * iq = build_lora_mm(layer.indexer_attn_q_b, qr);
            iq = ggml_reshape_3d(ctx0, iq, idx_head, idx_heads, nt);
            iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, (int) idx_rope, rope_type, n_ctx_orig_l,
                    freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
            iq = ggml_rope_set_offset(iq, idx_head - idx_rope);
            ggml_tensor * iw = build_lora_mm(layer.indexer_proj, cur);
            iw = ggml_scale(ctx0, iw, 1.0f / sqrtf((float) (idx_head * idx_heads)));
            ggml_tensor * ik = inp_dsv4->mctx->get_lid()->get_k(ctx0, src_il);
            ik = ggml_view_4d(ctx0, ik, ik->ne[0], ik->ne[1], n_csa, ik->ne[3],
                    ik->nb[1], ik->nb[2], ik->nb[3], 0);
            // split the token axis of q / weights per KV stream so the mul_mat
            // broadcasts over ik's stream dimension (same as deepseek4)
            const int64_t n_stream = ik->ne[3];
            iq = ggml_view_4d(ctx0, iq,
                    iq->ne[0], iq->ne[1], iq->ne[2] / n_stream, n_stream,
                    iq->nb[1], iq->nb[2], iq->nb[3] / n_stream, 0);
            iw = ggml_view_4d(ctx0, iw,
                    iw->ne[0], iw->ne[1] / n_stream, iw->ne[2], n_stream,
                    iw->nb[1], iw->nb[2] / n_stream, iw->nb[3] / n_stream, 0);
            iq = ggml_permute(ctx0, iq, 0, 2, 1, 3);
            ik = ggml_permute(ctx0, ik, 0, 2, 1, 3);
            ggml_tensor * kq = ggml_mul_mat(ctx0, ik, iq);
            kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
            ggml_tensor * score = ggml_mul(ctx0, ggml_relu(ctx0, kq), iw);
            score = ggml_sum_rows(ctx0, score);
            score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));
            if (csa_mask->type != score->type) {
                csa_mask = ggml_cast(ctx0, csa_mask, score->type);
            }
            score = ggml_add(ctx0, score, csa_mask);
            const uint32_t cand_bs = hparams.dsv41_candidate_block_size;
            const uint32_t cand_k  = hparams.dsv41_candidate_topk_blocks;
            const int32_t  cand_il = (int32_t) hparams.dsv41_candidate_source_layer;
            if (cand_bs > 0 && cand_k > 0) {
                if (il == cand_il) {
                    const int64_t n_blocks = (n_csa + cand_bs - 1) / cand_bs;
                    const int64_t padded_n_csa = n_blocks * cand_bs;
                    ggml_tensor * score_padded = score;
                    ggml_tensor * reach_padded = src_io->kq_mask;
                    if (padded_n_csa != n_csa) {
                        const int64_t pad_ne[GGML_MAX_DIMS] = {
                            padded_n_csa - n_csa, score->ne[1], score->ne[2], score->ne[3]
                        };
                        score_padded = ggml_concat(ctx0, score,
                                get_constant(score->type, pad_ne, -INFINITY), 0);
                        reach_padded = ggml_concat(ctx0, src_io->kq_mask,
                                get_constant(src_io->kq_mask->type, pad_ne, -INFINITY), 0);
                    }
                    ggml_tensor * pooled = ggml_pool_1d(ctx0, score_padded,
                            GGML_OP_POOL_MAX, (int) cand_bs, (int) cand_bs, 0);
                    ggml_tensor * block_reach = ggml_pool_1d(ctx0, reach_padded,
                            GGML_OP_POOL_MAX, (int) cand_bs, (int) cand_bs, 0);
                    pooled = ggml_add(ctx0, pooled, src_io->candidate_pin);
                    const uint32_t n_bk = (uint32_t) std::min<int64_t>(cand_k, n_blocks);
                    ggml_tensor * blk_top = ggml_cont(ctx0, ggml_top_k(ctx0, pooled, n_bk));
                    ggml_tensor * blk_mask = build_top_k_mask(block_reach, blk_top, "csa2_cand_blocks", il);
                    // repeat each block score across cand_bs compressed positions
                    blk_mask = ggml_reshape_3d(ctx0, blk_mask, 1, n_blocks, nt);
                    blk_mask = ggml_repeat_4d(ctx0, blk_mask, cand_bs, n_blocks, nt, 1);
                    blk_mask = ggml_reshape_4d(ctx0, blk_mask, padded_n_csa, nt, 1, 1);
                    if (padded_n_csa != n_csa) {
                        blk_mask = ggml_view_4d(ctx0, blk_mask, n_csa, nt, 1, 1,
                                blk_mask->nb[1], blk_mask->nb[2], blk_mask->nb[3], 0);
                    }
                    csa2_candidates = blk_mask;
                    cb(csa2_candidates, "csa2_candidates", il);
                } else if (cand_il >= 0 && il > cand_il && csa2_candidates) {
                    score = ggml_add(ctx0, score, csa2_candidates);
                }
            }
            const uint32_t n_top = (uint32_t) std::min<int64_t>(hparams.indexer_top_k, score->ne[0]);
            top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top));
            csa2_shared_topk = top_k;
            cb(top_k, "csa2_top_k", il);
        } else if (csa2_shared_topk) {
            top_k = csa2_shared_topk;
        }
        if (top_k) {
            csa_mask = build_top_k_mask(src_io->kq_mask, top_k, "csa2_top_k_mask", il);
        }
        if (csa_mask->type != kq_mask->type) {
            csa_mask = ggml_cast(ctx0, csa_mask, kq_mask->type);
        }
        kq_mask = ggml_concat(ctx0, kq_mask, csa_mask, 0);
        cb(k_all, "csa2_k_all", il);
    }

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, 0,
            1.0f/sqrtf((float) n_embd_head), il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank * n_groups, nt);
    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);
    return out;
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llama_model_deepseek4::graph(params) {
    if (params.gtype == LLM_GRAPH_TYPE_ENCODER && model.fc) {
        build_dspark_encoder(model);
        return;
    }
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_DSPARK && model.fc) {
        build_dspark_stages(model);
        return;
    }

    {
        const char * e = std::getenv("WP_DISPATCH_SPLIT_SHEXP");
        const bool split_on = (e == nullptr) || (e[0] != '0');
        moe_dispatch_split_shexp = (expert_dispatch != nullptr) && split_on;
    }

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_kq_mask);

    if (hparams.dsv41_n_kv_sources > 0) {
        auto inp_csa2 = std::make_unique<llm_graph_input_csa2>();
        inp_csa2->n_csa = inp_dsv4->mctx->get_csa()->get_n_kv();
        const int32_t candidate_src_il = kv_source_for((int) hparams.dsv41_candidate_source_layer);
        for (uint32_t i = 0; i < hparams.dsv41_n_kv_sources; ++i) {
            llm_graph_input_csa2::src_io io;
            io.il = (int) hparams.dsv41_kv_source_layer_ids[i];
            io.ratio = io.il >= 0 ? hparams.dsv4_compress_ratios[io.il] : 1;
            if (io.ratio == 0) {
                io.ratio = 1;
            }
            io.ring_size = inp_dsv4->mctx->get_csa_state()->get_state_size();
            io.n_latents = io.ratio == 1 ? n_tokens : std::max<uint32_t>(1,
                    ((uint32_t) n_tokens + io.ratio - 1) / io.ratio + 1);
            io.write_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, io.n_latents);
            io.group_pos  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, io.n_latents);
            if (io.ratio > 1) {
                io.ring_write_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
                io.ring_read_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, io.ratio * io.n_latents);
                io.pool_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
                io.pool_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, io.ratio * io.n_latents);
            }
            const uint32_t cand_bs = hparams.dsv41_candidate_block_size;
            if (cand_bs > 0 && hparams.dsv41_candidate_topk_blocks > 0 && io.il == candidate_src_il) {
                const int64_t n_blocks = (inp_csa2->n_csa + cand_bs - 1) / cand_bs;
                io.candidate_block_size = cand_bs;
                io.candidate_pin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blocks, n_tokens);
            }
            io.kq_mask    = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, inp_csa2->n_csa, n_tokens, 1, 1);
            ggml_set_input(io.write_idxs);
            ggml_set_input(io.group_pos);
            if (io.ratio > 1) {
                ggml_set_input(io.ring_write_idxs);
                ggml_set_input(io.ring_read_idxs);
                ggml_set_input(io.pool_idxs);
                ggml_set_input(io.pool_mask);
            }
            if (io.candidate_pin) {
                ggml_set_input(io.candidate_pin);
            }
            ggml_set_input(io.kq_mask);
            csa2_sources.push_back({ io.il, io.ratio, io.ring_write_idxs, io.ring_read_idxs,
                    io.write_idxs, io.group_pos, io.pool_idxs, io.pool_mask, io.kq_mask,
                    io.candidate_pin });
            inp_csa2->sources.push_back(io);
        }
        res->add_input(std::move(inp_csa2));
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * pre_mix = identity_pre_mix();
    cb(pre_mix, "hc_pre_mix_init", -1);

    ggml_tensor * hash_ids = nullptr;
    if (hparams.dsv41_n_engram_layers > 0 && hparams.dsv41_engram_n_heads > 0) {
        const int64_t n_hash_cols = std::max<int64_t>(1,
                (int64_t) (hparams.dsv41_engram_max_ngram ? hparams.dsv41_engram_max_ngram - 1 : 0) *
                (int64_t) hparams.dsv41_engram_n_heads);
        auto inp_engram = std::make_unique<llm_graph_input_engram>();
        inp_engram->hash_ids = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, n_hash_cols, n_tokens, hparams.dsv41_n_engram_layers);
        inp_engram->hasher   = &static_cast<const llama_model_deepseek41 &>(model).engram;
        ggml_set_input(inp_engram->hash_ids);
        hash_ids = inp_engram->hash_ids;
        res->add_input(std::move(inp_engram));
    }

    for (int il = 0; il < n_layer; ++il) {
        if (hparams.is_engram((uint32_t) il)) {
            inpL = build_engram(model, inpL, hash_ids, il);
            cb(inpL, "engram", il);
        }

        // DSpark layer-input taps: publish the collapsed residual ENTERING this
        // layer, mirroring the reference ordering (Engram added, then
        // main_hiddens.append(h.mean(dim=2)), then the layer runs) -- so the tap
        // sits post-Engram, pre-hc-mix/attn_norm. Plain mean over hc (the
        // V4.1 reference is unambiguous and has no hc_head tensors), real op
        // result, not a bare ggml_reshape_2d view.
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * attn_post = nullptr;
        ggml_tensor * attn_comb = nullptr;
        build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &attn_pre, &attn_post, &attn_comb, il);

        ggml_tensor * cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        const int64_t ratio = hparams.dsv4_compress_ratios[il];
        if (ratio == 0 || csa2_sources.empty()) {
            cur = build_attention(model, inp_dsv4, cur, inp_pos, il);
        } else {
            cur = build_attention_csa2(model, inp_dsv4, cur, inp_pos, il);
        }
        inpL = build_hc_post(cur, residual, attn_post, attn_comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        ggml_tensor * ffn_pre = nullptr;
        ggml_tensor * ffn_post = nullptr;
        ggml_tensor * ffn_comb = nullptr;
        build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &ffn_pre, &ffn_post, &ffn_comb, il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, ffn_post);
        ggml_build_forward_expand(gf, ffn_comb);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                layer.ffn_exp_probs_b, n_expert, ds41_n_expert_used(hparams, il),
                LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func, il);
        ggml_tensor * ffn_shexp = build_ffn(shexp_after_issue(cur, il),
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = complete_moe_dispatch(moe_out, ffn_shexp, il);
        inpL = build_hc_post(cur, residual, ffn_post, ffn_comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
        pre_mix = ffn_pre;
    }

    // Final boundary (index n_layer): the residual entering the head, same
    // collapsed-mean form as the per-layer taps.
    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    ggml_tensor * cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_collapse", -1);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = cap_lm_head_rows(cur);
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
