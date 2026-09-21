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

// DeepSeek-V4.1.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache (llama_model_deepseek4::graph, deepseek4.cpp). Three
// things differ, mirroring the vcruz/runtime/deepseek41 upstream port:
//
// 1. The hyper-connection coefficients lag by one sublayer (see build_hc_mixes/build_hc_pre/
//    build_hc_post in deepseek4.cpp, which already implement this for both V4 and V4.1).
// 2. There is no learned hyper-connection head; the last layer's FFN mix collapses the copies
//    instead (identity_pre_mix() below stands in for layer 0's missing predecessor mix).
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// The sparse attention also differs from V4: V4 compresses KV on every layer at one of two fixed
// ratios; V4.1 compresses on a few source layers and the layers after each source read the same
// rows via the KV cache's reuse-callback aliasing (see the is_v41 branch in
// llama_kv_cache_dsv4::llama_kv_cache_dsv4, llama-kv-cache-dsv4.cpp), and it derives index keys
// from that shared latent rather than from a second compressor. See build_attention_v41().
//
// WP-specific additions kept on top of the upstream shape: DSpark speculative-decoding head
// (build_dspark_encoder/build_dspark_stages, MTP/nextn tensors), routed_experts_external /
// WP_N_EXPERT_USED weight-paging hooks, and the moe_dispatch_split_shexp / complete_moe_dispatch
// pipelined-expert-dispatch skeleton.

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

int llama_model_deepseek41::engram_index(int il) const {
    for (uint32_t e = 0; e < hparams.dsv41_n_engram_layers; ++e) {
        if ((int) hparams.dsv41_engram_layer_ids[e] == il) {
            return (int) e;
        }
    }
    return -1;
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
    // V4.1 uses at most two nonzero ratios (a source layer and the layers that read it); 0
    // means "no compressed stream, pure sliding window". Do not reuse V4's 0/4/128 check.

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

    // Derive the O(1) per-layer "which layer published what I read" lookups (dsv41_kv_source[il]
    // etc, consumed by the KV cache's is_v41 reuse-callback scheme) from the explicit lists just
    // read above. The KV cache is constructed before any tensor exists, so this cannot wait for
    // load_arch_tensors the way theirs' tensor-presence walk does.
    hparams.dsv41_derive_stream_roles();

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

        // Only the KV source layers carry a compressor, and only those with a ratio above 1 pool
        // with a gate, so both are optional rather than keyed off the ratio the way V4 does it
        // (a reader layer shares its source's ratio but builds no compressor tensors of its own).
        layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, flags | TENSOR_NOT_REQUIRED);
        layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, flags | TENSOR_NOT_REQUIRED);
        layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags | TENSOR_NOT_REQUIRED);

        // An index source scores queries against shared index keys. Only a layer that also
        // compresses its own KV builds those keys; the rest read what an earlier layer published.
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
            // build_engram (below) always applies engram_k/engram_q, so unlike the compressor
            // tensors above these are required wherever the table itself is required.
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, flags);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, flags);
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i),
                    {n_hash_cols * engram_dim, n_embd * (hc_mult + 1)}, flags);
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

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence (get_prev_tokens reports LLAMA_TOKEN_NULL there), and the compressed
// token map folds case and accents together first.
//
// The predecessor tokens now come from the attention KV cells (ext.tok), via
// llama_kv_cache_dsv4_raw_context::get_prev_tokens -- the same generic mechanism PLE uses --
// rather than a graph-local buffer. A graph-local buffer is only ever populated for the ubatch
// currently being built (llm_graph_input_i objects are not kept alive across decode steps unless
// they override can_reuse(), which the original version of this file did not do), so any lookback
// past the start of the current ubatch silently read the pad id on every step after the first.
// The KV cache persists across steps, so this is correct for both prefill and incremental decode.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(
            const llama_model_deepseek41::engram_hasher & hasher,
            const llama_kv_cache_dsv4_raw_context * mctx,
            int ordinal) : hasher(hasher), mctx(mctx), ordinal(ordinal) {}
    ~llm_graph_input_engram() override = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (int64_t) hasher.n_cols();
        return rows != nullptr && rows->ne[0] == n_cols * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr; // I32 [n_cols * n_tokens]

    const llama_model_deepseek41::engram_hasher & hasher;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is (ordinal in dsv41_engram_layer_ids), so the right
    // multipliers/primes/offsets rows are used
    const int ordinal;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    if (rows == nullptr || rows->data == nullptr || ubatch == nullptr) {
        return;
    }

    const int64_t  n_tok     = ubatch->n_tokens;
    const uint32_t max_ngram = hasher.max_ngram;
    const uint32_t n_prev    = max_ngram > 1 ? max_ngram - 1 : 0;
    const int64_t  n_cols    = (int64_t) hasher.n_cols();

    GGML_ASSERT(mctx != nullptr);
    GGML_ASSERT(max_ngram >= 1 && max_ngram <= 16);

    for (int64_t i = 0; i < n_tok; ++i) {
        // the preceding tokens would be ambiguous; see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    std::vector<llama_token> prev;
    if (n_prev > 0) {
        mctx->get_prev_tokens(*ubatch, n_prev, prev);
    }

    std::vector<int32_t> idx((size_t) (n_cols * n_tok));
    std::vector<int32_t> ctx_ids(max_ngram);

    for (int64_t i = 0; i < n_tok; ++i) {
        // an image arrives as an embd batch, so ubatch->token is null; the reference gives those
        // positions no engram contribution at all, which the padding token stands in for here
        ctx_ids[0] = ubatch->token ? hasher.compress((int32_t) ubatch->token[i]) : hasher.pad_id;

        // look-back stops at the start of the sequence; everything from there on reads as padding
        bool blocked = false;
        for (uint32_t s = 1; s < max_ngram; ++s) {
            const llama_token t = (blocked || n_prev == 0)
                ? LLAMA_TOKEN_NULL
                : prev[(size_t) i * n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx_ids[s] = blocked ? hasher.pad_id : hasher.compress((int32_t) t);
        }

        hasher.hash_position(ctx_ids.data(), (uint32_t) ordinal, idx.data() + i * n_cols);
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size() * ggml_element_size(rows));
}

void llama_model_deepseek41::engram_hasher::hash_position(
        const int32_t * ctx_ids, uint32_t ordinal, int32_t * row) const {
    const uint32_t ngram_kinds = max_ngram > 1 ? max_ngram - 1 : 1;
    const bool     have_mult   = !multipliers.empty();
    int64_t products[16];
    GGML_ASSERT(max_ngram <= 16);
    for (uint32_t shift = 0; shift < max_ngram; ++shift) {
        const int32_t tok  = ctx_ids[shift];
        const int64_t mult = have_mult ? multipliers[(size_t) ordinal * max_ngram + shift] : 1;
        products[shift] = (int64_t) tok * mult;
    }
    int64_t rolling = products[0];
    uint32_t col = 0;
    for (uint32_t i = 1; i < max_ngram; ++i) {
        rolling ^= products[i];
        for (uint32_t h = 0; h < n_heads; ++h, ++col) {
            const size_t k = ((size_t) ordinal * ngram_kinds + (i - 1)) * n_heads + h;
            const int64_t prime = primes[k];
            int64_t id = rolling % prime;
            if (id < 0) {
                id += prime;
            }
            row[col] = (int32_t) (id + offsets[k]);
        }
    }
}

ggml_tensor * llama_model_deepseek41::graph::build_inp_engram(
        const llama_model & model,
        int il) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    const int64_t n_cols  = (int64_t) pmodel.engram.n_cols();
    const int64_t key_len = pmodel.hparams.dsv41_engram_head_dim ? pmodel.hparams.dsv41_engram_head_dim : 256;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_engram>(pmodel.engram, mctx_cur->get_raw(), pmodel.engram_index(il));

    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_set_input(inp->rows);
    ggml_tensor * rows = inp->rows;
    res->add_input(std::move(inp));

    // gather then flatten, laying the buckets out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.layers[il].engram_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // The gate scales reach ggml_mul, which takes only f32, and a file quantized before
    // llama-quant.cpp learned to skip them carries them quantized. get_rows dequantizes.
    auto as_f32 = [&](ggml_tensor * w) {
        if (w->type == GGML_TYPE_F32) {
            return w;
        }
        ggml_tensor * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) w->ne[1], 1.0f), GGML_TYPE_I32);
        return ggml_get_rows(ctx0, w, ids);
    };

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference
    // keeps engram_q and engram_k apart but only ever uses their product, so applying one to each
    // side of the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, ggml_reshape_2d(ctx0, w, hc_dim, 1));
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };

    ggml_tensor * k = grouped_norm(key, as_f32(model.layers[il].engram_k));
    ggml_tensor * q = grouped_norm(x,   as_f32(model.layers[il].engram_q));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

// Rope settings for one layer. A layer that reads a compressed stream rotates with the
// compressor's base and YaRN; a plain sliding window layer rotates with the model's.
struct dsv41_rope_cfg {
    float   base;
    float   scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t n_ctx_orig;
};

dsv41_rope_cfg llama_model_deepseek41::graph::rope_cfg(int il) const {
    if (hparams.dsv4_compress_ratios[il] == 0) {
        return { freq_base, 1.0f, 0.0f, dsv4_rope_attn_factor(1.0f, 0.0f), 0.0f, 0.0f, 0 };
    }

    return {
        hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
        dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow, n_ctx_orig,
    };
}

// Undo the rotation the query carried into attention, then the grouped output projection. wo_a is
// block diagonal over groups, each projecting only its own heads, hence a batched mul_mat.
ggml_tensor * llama_model_deepseek41::graph::build_attention_tail(
        const llama_model & model,
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = (n_head/n_groups)*n_embd_head;

    const dsv41_rope_cfg rc = rope_cfg(il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);

    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);

    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

// Score this layer's queries against the shared index keys and keep the best compressed positions.
// The keys were published by an earlier layer, so this only builds the query side.
ggml_tensor * llama_model_deepseek41::graph::build_indexer_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head      = hparams.indexer_n_head;
    const int64_t n_idx_head_dim  = hparams.indexer_head_size;
    const int64_t n_idx_head_rope = hparams.n_rot();
    const int64_t n_idx_head_nope = n_idx_head_dim - n_idx_head_rope;
    const int64_t nt              = cur->ne[1];

    GGML_ASSERT(inp_comp.kq_mask);
    GGML_ASSERT(n_idx_head_dim >= n_idx_head_rope);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_head_dim, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_idx_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    idx_q = ggml_rope_set_offset(idx_q, n_idx_head_nope);
    cb(idx_q, "idx_q", il);

    ggml_tensor * idx_k_rot = inp_dsv4->get_lid().k_rot;
    if (idx_k_rot) {
        idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);
        cb(idx_q, "idx_q_rot", il);
    }

    // one weight per head, scaled so the score matches the reference's
    // softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head)));
    cb(idx_w, "idx_weights", il);

    ggml_tensor * idx_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "idx_k", il);

    const int64_t n_stream = idx_k->ne[3];
    idx_q = ggml_view_4d(ctx0, idx_q,
            idx_q->ne[0], idx_q->ne[1], idx_q->ne[2]/n_stream, n_stream,
            idx_q->nb[1], idx_q->nb[2], idx_q->nb[3]/n_stream, 0);
    idx_w = ggml_view_4d(ctx0, idx_w,
            idx_w->ne[0], idx_w->ne[1]/n_stream, idx_w->ne[2], n_stream,
            idx_w->nb[1], idx_w->nb[2]/n_stream, idx_w->nb[3]/n_stream, 0);

    idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
    idx_k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

    ggml_tensor * score = ggml_mul_mat(ctx0, idx_k, idx_q);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

    score = ggml_relu(ctx0, score);
    score = ggml_mul(ctx0, score, idx_w);
    score = ggml_sum_rows(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

    // the attention mask is F16 when flash attention is on, and this score is F32. the mask only
    // ever holds 0 or -inf, so widening it is exact.
    ggml_tensor * mask = inp_comp.kq_mask;
    if (mask->type != score->type) {
        mask = ggml_cast(ctx0, mask, score->type);
    }

    score = ggml_add(ctx0, score, mask);
    cb(score, "idx_score", il);

    const uint32_t n_top_k = score->ne[0] < hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
    cb(top_k, "idx_top_k", il);

    return top_k;
}

// DeepSeek-V4.1 attention: a sliding window of raw KV, plus, where the layer uses one, the
// compressed positions the indexer picked, concatenated into a single masked attention.
//
// Only a source layer compresses. The layers after it read the same rows, which the KV cache
// hands them through the reuse callback (llama_kv_cache_dsv4's is_v41 ctor branch), so a reader
// builds no compressor at all.
ggml_tensor * llama_model_deepseek41::graph::build_attention_v41(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const dsv41_rope_cfg rc = rope_cfg(il);

    // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    // the sliding window KV, which every layer keeps for itself
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = nullptr;

    if (ratio == 0) {
        // no compressed stream, so this layer sees only its own window
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);

        return build_attention_tail(model, out, inp_pos, nt, il);
    }

    // The plan slot follows the ratio, since a plan encodes how many tokens make a row. The rows
    // themselves always live in the CSA cache, and the index keys in the indexer cache, whichever
    // plan produced them.
    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    const auto & inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();

    const llama_dsv4_comp_state * comp_state = use_csa
        ? inp_dsv4->mctx->get_csa_state()
        : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(inp_comp.kq_mask && "a compressed layer needs a plan for its ratio");

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(state_kv, "comp_state_kv", il);

        // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below
        // then runs over a single element and returns 1.0 whatever the score holds, so the
        // values reach the cache unweighted, which is what a plain projection means.
        ggml_tensor * state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, cur)
            : state_kv;
        cb(state_score, "comp_state_score", il);

        const dsv4_state_tensors restored = dsv4_build_state_restore(ctx0, inp_comp, comp_state, il);

        ggml_tensor * base_kv = dsv4_view_2d(
                ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
        ggml_tensor * base_score = dsv4_view_2d(
                ctx0, restored.score, restored.score->ne[0], comp_state->get_n_rows(), 0);

        ggml_tensor * source_kv    = ggml_concat(ctx0, base_kv,    state_kv,    1);
        ggml_tensor * source_score = ggml_concat(ctx0, base_score, state_score, 1);

        // the indexer reads the latent before it is rotated, so ask for both forms at once
        ggml_tensor * latent_pre = nullptr;

        ggml_tensor * latent = build_hca_compressed_kv_from_state(
                source_kv,
                source_score,
                inp_comp.state_read_idxs,
                inp_comp.state_write_pos,
                layer.attn_comp_norm,
                ratio,
                n_embd_head,
                "comp_kv",
                il,
                &latent_pre);

        if (hparams.dsv41_owns_index_k(il)) {
            const int64_t n_idx_head_dim  = hparams.indexer_head_size;
            const int64_t n_idx_head_rope = hparams.n_rot();

            ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, latent_pre);
            idx_k = build_norm(idx_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
            idx_k = ggml_rope_ext(ctx0, idx_k, inp_comp.state_write_pos, nullptr, n_idx_head_rope,
                    rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                    dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
            idx_k = ggml_rope_set_offset(idx_k, n_idx_head_dim - n_idx_head_rope);
            cb(idx_k, "idx_k_new", il);

            if (inp_dsv4->get_lid().k_rot) {
                idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);
            }

            ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(
                        ctx0, idx_k, inp_comp.state_write_idxs, il));
        }

        if (inp_dsv4->get_csa().k_rot) {
            latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
            cb(latent, "comp_kv_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(
                    ctx0, latent, inp_comp.state_write_idxs, il));

        // carry whatever did not complete a row into the next ubatch
        ggml_tensor * snapshot_kv    = ggml_concat(ctx0, restored.kv,    state_kv,    1);
        ggml_tensor * snapshot_score = ggml_concat(ctx0, restored.score, state_score, 1);

        const dsv4_state_tensors snapshot = dsv4_build_state_snapshot(
                ctx0, inp_comp, comp_state, snapshot_kv, snapshot_score, il);
        if (snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, snapshot.kv);
        }
        if (snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, snapshot.score);
        }

        ggml_tensor * persist_kv = ggml_get_rows(ctx0, state_kv, inp_comp.state_persist_src_idxs);
        ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp_comp.state_persist_src_idxs);

        ggml_build_forward_expand(gf, comp_state->cpy_kv(
                    ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
        ggml_build_forward_expand(gf, comp_state->cpy_score(
                    ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
    }

    // an index source picks the positions; the layers in between reuse what it picked
    ggml_tensor * top_k = nullptr;
    if (hparams.dsv41_is_index_source(il)) {
        top_k = build_indexer_top_k(model, inp_dsv4, inp_comp, qr, cur, inp_pos, il);
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
    cb(k_all, "k_all", il);

    ggml_tensor * raw_mask  = inp_attn->get_kq_mask();
    ggml_tensor * comp_mask = top_k
        ? build_top_k_mask(inp_comp.kq_mask, top_k, "comp_top_k_mask", il)
        : inp_comp.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
    cb(kq_mask, "kq_mask", il);

    const int64_t n_kv_max = top_k
        ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0]
        : 0;

    out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
            nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(model, out, inp_pos, nt, il);
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

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = identity_pre_mix();
    cb(pre_mix, "hc_pre_mix_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // the engram sits before the block and writes straight into the stream
        if (static_cast<const llama_model_deepseek41 &>(model).engram_index(il) >= 0) {
            inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
            cb(inpL, "engram_out", il);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);

        ggml_tensor * cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl if the input is from mtmd
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, ds41_n_expert_used(hparams, il),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(shexp_after_issue(cur, il),
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = complete_moe_dispatch(moe_out, ffn_shexp, il);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    // This is what a learned hyper-connection head does in V4, which is why this model has none.
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
