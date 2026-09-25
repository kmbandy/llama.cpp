#include "models.h"

#include "ggml-backend.h"
#include "ggml-ml8.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-kv-cache-dsv4.h"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
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

    // ml8-4 / ml8-fp8 / fp8_b128 sidecar registration (2026-09-22), mirroring
    // llama_model_qwen35::load_arch_tensors's identical lambda block
    // (src/models/qwen35.cpp ~line 64-191) verbatim except for one addition:
    // an optional `xid_` parameter threaded through every tn() call, needed
    // for LLM_TENSOR_ATTN_OUT_A_SPLIT's group index (wo_a's split tensors,
    // "blk.%d.attn_output_a.g%d") -- every other DS4.1 target tensor passes
    // xid_=-1 (tn()'s default, meaning "no second index", same as qwen35's
    // calls). build_ml8_or_mul_mat (src/llama-ml8-registry.h) itself is
    // arch-agnostic; this registration is the piece that was NOT automatic
    // (scoping note: earlier assumed "likely automatic", which turned out
    // to be wrong -- register_weight() is only ever called from model-arch
    // code, once per weight, at load time).
    auto is_ml8_sidecar_weight = [](const struct ggml_tensor * w) {
        return w && (w->type == GGML_TYPE_ML8_4 || w->type == GGML_TYPE_ML8_FP8 || w->type == GGML_TYPE_FP8_B128);
    };

    auto load_ml8_sidecars = [&](
            struct ggml_tensor * weight,
            llm_tensor tensor_id,
            int il_,
            int64_t k_dim,
            struct ggml_tensor ** out_centroids,
            struct ggml_tensor ** out_rotation_h_a,
            struct ggml_tensor ** out_rotation_meta,
            struct ggml_tensor ** out_awq_scale,
            int xid_ = -1) {
        if (!is_ml8_sidecar_weight(weight)) {
            return;
        }
        if (weight->type == GGML_TYPE_ML8_4) {
            *out_centroids = create_tensor(tn(tensor_id, "centroids", il_, xid_),
                                           { 16, k_dim / 64 }, TENSOR_NOT_REQUIRED);
            *out_awq_scale = create_tensor(tn(tensor_id, "awq_scale", il_, xid_),
                                           { k_dim }, TENSOR_NOT_REQUIRED);
        }
        const auto * h_a_meta = ml.get_tensor_meta(tn(tensor_id, "rotation_h_a", il_, xid_).str().c_str());
        if (h_a_meta != nullptr) {
            const int64_t a = h_a_meta->ne[0];
            *out_rotation_h_a  = create_tensor(tn(tensor_id, "rotation_h_a",  il_, xid_),
                                               { a, a }, TENSOR_NOT_REQUIRED);
        }
        // block_hadamard weights carry rotation_meta with NO rotation_h_a, so
        // this is checked independently rather than nested under h_a_meta.
        const auto * meta_meta = ml.get_tensor_meta(tn(tensor_id, "rotation_meta", il_, xid_).str().c_str());
        if (meta_meta != nullptr) {
            *out_rotation_meta = create_tensor(tn(tensor_id, "rotation_meta", il_, xid_),
                                               { 4 }, TENSOR_NOT_REQUIRED);
        }
    };

    auto read_rotation_meta = [&](llm_tensor tensor_id, int il_, int32_t (&out)[4], int xid_ = -1) -> bool {
        const std::string name = tn(tensor_id, "rotation_meta", il_, xid_).str();
        const auto * w = ml.get_weight(name.c_str());
        if (w == nullptr) {
            return false;
        }
        GGML_ASSERT(w->idx < ml.files.size());
        ml.files.at(w->idx)->seek(w->offs, SEEK_SET);
        ml.files.at(w->idx)->read_raw(out, sizeof(out));
        return true;
    };

    auto fill_rotation_meta = [&](ml8_sidecars & sc, struct ggml_tensor * rotation_meta,
                                  llm_tensor tensor_id, int il_, int xid_ = -1) {
        if (rotation_meta == nullptr) {
            return;
        }
        int32_t meta[4];
        if (!read_rotation_meta(tensor_id, il_, meta, xid_)) {
            return;
        }
        const int32_t kind_id = meta[3];
        GGML_ASSERT((kind_id == GGML_ML8_ROTATION_KIND_KRONECKER_ORTH_SYLVESTER ||
                     kind_id == GGML_ML8_ROTATION_KIND_BLOCK_HADAMARD) &&
                    "rotation_meta: unrecognized kind_id");
        GGML_ASSERT((kind_id == GGML_ML8_ROTATION_KIND_KRONECKER_ORTH_SYLVESTER) == (sc.rotation_h_a != nullptr) &&
                    "rotation_meta kind_id / rotation_h_a presence mismatch");
        sc.rotation_b_dim          = meta[1];
        sc.rotation_block_hadamard = (kind_id == GGML_ML8_ROTATION_KIND_BLOCK_HADAMARD);
    };

    // k_dim is the weight's input feature count (ne[0] / K) — the same value
    // the weight's create_tensor used for its leading dim.
    auto register_ml8_weight = [&](struct ggml_tensor * weight,
                                   llm_tensor tensor_id, int il_, int64_t k_dim, int xid_ = -1) {
        if (!is_ml8_sidecar_weight(weight)) {
            return;
        }
        struct ggml_tensor * centroids    = nullptr;
        struct ggml_tensor * rotation_h_a = nullptr;
        struct ggml_tensor * rotation_meta = nullptr;
        struct ggml_tensor * awq_scale    = nullptr;
        load_ml8_sidecars(weight, tensor_id, il_, k_dim,
                          &centroids, &rotation_h_a, &rotation_meta, &awq_scale, xid_);
        ml8_sidecars sc{ centroids, rotation_h_a, awq_scale };
        fill_rotation_meta(sc, rotation_meta, tensor_id, il_, xid_);
        ml8_reg.register_weight(weight, sc);
    };

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
        // Not required: an ml8-4 spine ships only the split wo_a_g below. The
        // assert after the split loop requires exactly one of the two forms.
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE | TENSOR_NOT_REQUIRED);
        // ml8-4 data-free conversion (Task 2, 2026-09-22): optional split wo_a,
        // o_groups separate 2D tensors instead of the flat block-diagonal-over-
        // groups layout above (rotating that flat tensor as one GEMM would mix
        // groups -- see scripts/calibration/convert_fp8_rotated.py's wo_a
        // split). Present only in a converted ml8-4 spine; a Q8_0 spine has
        // none of these (TENSOR_NOT_REQUIRED), wo_a_g[0] stays nullptr, and
        // build_attention_tail below keeps using the batched-mul_mat path on
        // layer.wo_a unchanged.
        GGML_ASSERT(o_groups <= (int64_t) (sizeof(layer.wo_a_g) / sizeof(layer.wo_a_g[0])) &&
                "wo_a_g is fixed-size; dsv4_o_group_count grew past what the loader supports");
        for (int64_t g = 0; g < o_groups; ++g) {
            layer.wo_a_g[g] = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A_SPLIT, "weight", i, (int) g),
                    {n_head * n_embd_head / o_groups, o_lora_rank}, flags | TENSOR_NOT_REQUIRED);
        }
        if (!(flags & TENSOR_SKIP)) { // MTP layers skipped at load have neither form, by design
            int64_t n_split = 0;
            for (int64_t g = 0; g < o_groups; ++g) {
                n_split += layer.wo_a_g[g] != nullptr;
            }
            if ((layer.wo_a == nullptr) == (n_split == 0) || (n_split != 0 && n_split != o_groups)) {
                throw std::runtime_error(format("layer %d: need either attn_output_a.weight or all %d attn_output_a.g* splits (have flat=%d, splits=%d)",
                                                i, (int) o_groups, layer.wo_a != nullptr, (int) n_split));
            }
        }
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

        // ml8-4 / ml8-fp8 / fp8_b128 registry registration for the 9 (+8 wo_a
        // split groups) DS4.1 attention/indexer/compressor GEMM roles
        // convert_fp8_rotated.py's ml8_4 format targets (2026-09-22). Every
        // call is a no-op unless the weight's actual GGUF type is one of the
        // three ml8-family types (is_ml8_sidecar_weight), so this is safe on
        // an unconverted Q8_0 spine -- registry stays empty, build_lora_mm
        // falls back to plain mul_mat exactly as before this change.
        register_ml8_weight(layer.wq_a,          LLM_TENSOR_ATTN_Q_A,          i, n_embd);
        register_ml8_weight(layer.wq_b,          LLM_TENSOR_ATTN_Q_B,          i, q_lora_rank);
        register_ml8_weight(layer.wkv,           LLM_TENSOR_ATTN_KV,           i, n_embd);
        register_ml8_weight(layer.wo_b,          LLM_TENSOR_ATTN_OUT_B,        i, o_groups * o_lora_rank);
        for (int64_t g = 0; g < o_groups; ++g) {
            register_ml8_weight(layer.wo_a_g[g], LLM_TENSOR_ATTN_OUT_A_SPLIT, i,
                                n_head * n_embd_head / o_groups, (int) g);
        }
        register_ml8_weight(layer.attn_comp_wkv,   LLM_TENSOR_ATTN_COMPRESSOR_WKV,   i, n_embd);
        register_ml8_weight(layer.attn_comp_wgate, LLM_TENSOR_ATTN_COMPRESSOR_WGATE, i, n_embd);
        register_ml8_weight(layer.indexer_proj,     LLM_TENSOR_INDEXER_PROJ,     i, n_embd);
        register_ml8_weight(layer.indexer_attn_q_b, LLM_TENSOR_INDEXER_ATTN_Q_B, i, q_lora_rank);
        register_ml8_weight(layer.indexer_attn_k,   LLM_TENSOR_INDEXER_ATTN_K,   i, n_embd_head);

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
    // WP_DSV41_ENGRAM=0: diagnostic ablation -- the engram layers become a
    // pass-through so the n-gram lookup can be isolated from the rest of the
    // graph on a live serve. Default on.
    static const bool engram_enabled = [] {
        const char * e = std::getenv("WP_DSV41_ENGRAM");
        return e == nullptr || e[0] != '0';
    }();
    if (!engram_enabled) {
        return x;
    }
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

    ggml_tensor * oa;
    if (layer.wo_a_g[0] != nullptr) {
        // ml8-4 data-free conversion (Task 2): wo_a was split at conversion
        // time into `n_groups` separate 2D tensors (see load_arch_tensors and
        // scripts/calibration/convert_fp8_rotated.py's wo_a split), each
        // projecting only its own group's heads -- o_groups build_lora_mm
        // calls (so ml8-4 dispatch via the registry applies per group) whose
        // [o_lora_rank, 1, nt] outputs are concatenated along the group axis
        // to reproduce the batched path's [o_lora_rank, n_groups, nt] shape
        // below, bit-for-bit the same layout the non-split path's permute +
        // cont_2d produces.
        oa = nullptr;
        for (int64_t g = 0; g < n_groups; ++g) {
            ggml_tensor * out_g = ggml_view_2d(ctx0, out, o_group_dim, nt, out->nb[2], g*out->nb[1]);
            out_g = ggml_cont(ctx0, out_g);
            ggml_tensor * oa_g = build_lora_mm(layer.wo_a_g[g], out_g);
            oa_g = ggml_reshape_3d(ctx0, oa_g, o_lora_rank, 1, nt);
            oa = oa == nullptr ? oa_g : ggml_concat(ctx0, oa, oa_g, 1);
        }
        cb(oa, "attn_wo_a_split", il);
    } else {
        out = ggml_permute(ctx0, out, 0, 2, 1, 3);
        oa = ggml_mul_mat(ctx0, layer.wo_a, out);
        cb(oa, "attn_wo_a", il);
        oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    }
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

    ggml_tensor * score = nullptr;
    if (cparams.fused_lid) {
        // ggml_lightning_indexer wants q/k/weights in their pre-permute layout
        // (see the shape doc on the op) and the mask as F16; the compressed
        // mask is only F16 when flash attention is on (dsv4_build_comp_inputs),
        // so cast it up front, the same widening the unfused path below does
        // to F32 (the mask only ever holds 0 or -inf, so either cast is exact).
        ggml_tensor * mask = inp_comp.kq_mask;
        if (mask->type != GGML_TYPE_F16) {
            mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        }
        score = ggml_lightning_indexer(ctx0, idx_q, idx_k, idx_w, mask);
        cb(score, "idx_score", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
        idx_k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

        score = ggml_mul_mat(ctx0, idx_k, idx_q);
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
    }

    const uint32_t n_top_k = score->ne[0] < hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
    cb(top_k, "idx_top_k", il);

    return top_k;
}

// CED prefill trim (WP_DSV41_CED_PREFILL), gate + trunk-loop helpers. See the
// long comment in graph::graph() (where the gate is evaluated) for the
// architecture rationale; these are just the small pieces it and
// build_attention_v41 share.

// see the HAZARD check in graph::graph(); one sequence per server here
static llama_pos dsv41_ced_skipped_decoder_end = -1;

static bool dsv41_ced_prefill_env_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("WP_DSV41_CED_PREFILL");
        return e != nullptr && e[0] == '1'; // default OFF; require exact "1"
    }();
    return enabled;
}

// WP_DSV41_SPARSE_ATTN: route DS4.1's attention through AITER's native
// sparse kernels (ggml_sparse_attn_dsv4 / GGML_OP_SPARSE_ATTN_DSV4, HIP-only)
// instead of the dense build_attn_mha scan. Default OFF; the dense path is
// always the fallback (env off, op unsupported, or the "no top_k" case
// below where n_comp is unbounded).
static bool dsv41_sparse_attn_env_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("WP_DSV41_SPARSE_ATTN");
        return e != nullptr && e[0] == '1'; // default OFF; require exact "1"
    }();
    return enabled;
}

// WP_DSV41_TOPK_REUSE=1: non-source compressed layers reuse their index
// source's top-k (as the reference does) instead of attending to every
// compressed position. Default off until A/B'd.
static bool dsv41_topk_reuse_env_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("WP_DSV41_TOPK_REUSE");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

static bool dsv41_sparse_attn_check_env_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("WP_DSV41_SPARSE_ATTN_CHECK");
        return e != nullptr && e[0] == '1'; // default OFF; require exact "1"
    }();
    return enabled;
}

// WP_DSV41_CED_W: bisection knob, active only when WP_DSV41_CED_PREFILL=1.
// Overrides the replay width W (default hparams.n_swa) with any value the
// caller asks for; the only clamp applied is the structural one every other
// eligibility condition already enforces (w >= 1, and the gate's own
// `n_tokens > w` check refuses anything that isn't strictly narrower than
// the prompt). Deliberately does NOT require w == n_swa: that equality was
// a correctness assumption for the *shipped* semantics (SWA Bounded
// Replay), not a structural necessity of the narrowing mechanism itself --
// this knob exists precisely to let W vary so the two can be told apart.
// Memoized like the other env parsers; -1 means "not set, use the default".
static int64_t dsv41_ced_w_override() {
    static const int64_t w = []() -> int64_t {
        const char * e = std::getenv("WP_DSV41_CED_W");
        if (e == nullptr || e[0] == '\0') {
            return -1;
        }
        char * end = nullptr;
        const long long v = std::strtoll(e, &end, 10);
        if (end == e || v < 1) {
            return -1; // unparseable or structurally invalid -- fall back to the default
        }
        return (int64_t) v;
    }();
    return w;
}

enum class dsv41_ced_scope {
    ALL,  // default: today's behaviour -- attention AND the FFN/MoE both narrow at the seam
    FFN,  // narrow only the FFN/MoE path; the seam layer's own attention runs full width
    ATTN, // narrow only attention; see the scope-coupling note in graph::graph()
};

// WP_DSV41_CED_SCOPE: bisection knob, active only when WP_DSV41_CED_PREFILL=1.
// "all" (default), "ffn", or "attn" (case-sensitive, exact match); anything
// else falls back to "all". See graph::graph() for how each is implemented
// and the one place ATTN and ALL are provably the same code path (a
// structural coupling, not an oversight -- documented there, not hidden
// here).
static dsv41_ced_scope dsv41_ced_scope_override() {
    static const dsv41_ced_scope scope = []() {
        const char * e = std::getenv("WP_DSV41_CED_SCOPE");
        if (e != nullptr && std::strcmp(e, "ffn") == 0) {
            return dsv41_ced_scope::FFN;
        }
        if (e != nullptr && std::strcmp(e, "attn") == 0) {
            return dsv41_ced_scope::ATTN;
        }
        return dsv41_ced_scope::ALL;
    }();
    return scope;
}

static const char * dsv41_ced_scope_name(dsv41_ced_scope scope) {
    switch (scope) {
        case dsv41_ced_scope::FFN:  return "ffn";
        case dsv41_ced_scope::ATTN: return "attn";
        default:                    return "all";
    }
}

// WP_DSV41_CED_SKIP_NONFINAL: bisection knob, active only when
// WP_DSV41_CED_PREFILL=1. A non-final ubatch (n_outputs==0) has no row any
// caller will ever read: the trim above already narrows its decoder-range
// compute to the trailing W rows of THAT ubatch, but even that is wasted
// work when nothing downstream looks at it. Default ON (any value other than
// exact "0") once the trim itself is on, so enabling the trim gets this too
// unless the caller opts out to restore today's per-ubatch trailing-W
// behaviour -- see graph::graph()'s ced_skip_layer for what it changes.
static bool dsv41_ced_skip_nonfinal_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("WP_DSV41_CED_SKIP_NONFINAL");
        return e == nullptr || e[0] != '0';
    }();
    return enabled;
}

// Observability for the gate above: with the env var on, we still build a
// graph per ubatch (many per prefill, one per decode token) and log at WARN
// -- same facility/threshold as "expert dispatch gate"/"expert dispatch
// deferral" in llama-graph.cpp / pipe-expert-dispatch-graph.cpp -- but a line
// per ubatch would be hundreds of lines for one prompt. Only emit when the
// outcome (applied vs. refused, and if refused, which condition) actually
// differs from the last line logged, so a request produces a handful of
// lines: one at the first decision, and one each time it changes. Only ever
// called from code gated on dsv41_ced_prefill_env_enabled(), which the
// env-var-off path never enters, so this is silent (and these statics are
// never touched) when the feature is off.
//
// `channel` keeps independent dedup state per kind of line -- the gate
// decision (0), k_idxs bounds-check trips (1), shape logging (2), and the
// skip-nonfinal short-final-ubatch hazard (3) -- so e.g. a bounds-check line
// firing doesn't reset the gate-decision line's "already logged this" state
// and cause it to reprint on the next ubatch.
static void dsv41_ced_log_if_changed(const std::string & msg, int channel = 0) {
    static std::string last[4];
    GGML_ASSERT(channel >= 0 && channel < 4);
    if (msg != last[channel]) {
        last[channel] = msg;
        LLAMA_LOG_WARN("%s\n", msg.c_str());
    }
}

// Every requested output row must resolve inside the trailing `w` tokens the
// trim actually computes; a row outside that window was never produced by
// the trimmed graph and reading it would be a stale/uninitialized-buffer
// read, not merely a wrong answer. O(n_tokens - w), not O(n_tokens): this
// only has to find one disqualifying row outside the kept window.
//
// n_outputs==0 (no row of THIS ubatch requested) USED TO be refused
// unconditionally (round-1 fix, see scratchpad/ced-multiubatch-fix-report.md):
// llama-server requests outputs only on the ubatch that actually contains
// the token(s) the caller wants (normally just the last ubatch of a
// multi-ubatch prefill), so every earlier ubatch had n_outputs==0. Trimming
// decoder layers on THAT ubatch narrows their query/KV-write compute (see
// graph::graph() below) to its own local trailing w rows; find_slot()/
// apply_ubatch() -- CED-unaware -- still mark cells used for the ubatch's
// FULL n_tokens, so the untrimmed rows' decoder-range raw-KV cells are
// marked valid/in-range but never written by anything. The confirmed root
// cause (WP_CED_NAN_TRACE=1 on a reliably-crashing 7010-token, ubatch-2048
// prompt): a non-seam decoder layer's raw attention passed n_kv_max=0 ("no
// explicit cap", the plain dense-mask fallback every non-index-source V4/
// V4.1 layer already used before CED existed), so the flash-attention
// kernel scanned every physical raw-KV column regardless of the mask;
// ced_replay_floor's masking (llama_kv_cache::set_input_kq_mask) correctly
// marks the never-written cells -inf, but IEEE-754 -inf + NaN = NaN, so a
// masked cell that happens to hold a NaN bit pattern still poisons the
// whole row's softmax -- the mask value alone doesn't suppress it when the
// kernel isn't told to skip the entry.
//
// round-3 fix: rather than refuse n_outputs==0 outright, `n_kv_max` (see its
// own doc comment where it's computed in build_attention_v41) now bounds
// EVERY decoder-range layer's scan to the true finite-entry count (SWA-
// visible raw span + every comp entry, the comp side always being fully
// written regardless of trim -- see cur_kv), not the dense "scan everything"
// fallback. That directly closes the mechanism above: a masked, possibly-
// NaN raw cell from a skipped (non-final-ubatch) row is simply never
// touched by the kernel once n_kv_max excludes it, independent of whether
// the mask value alone would have suppressed it. With that fix in place,
// n_outputs==0 is exactly as safe to trim as n_outputs>0 -- every eligible
// ubatch (final or not) narrows its own decoder-range compute to its own
// trailing w rows, writing real raw KV only for those rows (still finite,
// still correctly masked for whichever LATER query, if any, needs them --
// see the "final ubatch shorter than w" note below for the one case this
// doesn't cover, which the w > 0 && n_tokens > w gate already refuses).
static bool dsv41_ced_outputs_in_window(const llama_ubatch & ubatch, uint32_t n_outputs, int64_t n_tokens, int64_t w) {
    if ((int64_t) n_outputs == n_tokens) {
        // logits_all: every row is requested. A trimmed graph only ever
        // computes the last `w` of them, so this can never be satisfied.
        return false;
    }
    if (n_outputs == 0) {
        // No row of THIS ubatch is requested -- fine to trim now (see the
        // function comment above): every eligible ubatch narrows to its own
        // trailing w rows regardless, and n_kv_max keeps that safe. Skip
        // straight to "trim it" rather than falling into the loop below,
        // which is scanning ubatch.output for rows OUTSIDE the window --
        // moot when nothing is requested at all, and ubatch.output may not
        // even be populated in that case.
        return true;
    }
    if (!ubatch.output) {
        return false; // can't verify which rows are wanted; refuse rather than guess
    }
    for (int64_t i = 0; i < n_tokens - w; ++i) {
        if (ubatch.output[i]) {
            return false;
        }
    }
    return true;
}

// WP_DSV41_CED_CONT: bisection/diagnostic knob, active only when
// WP_DSV41_CED_PREFILL=1. When set, every dsv41_ced_trailing_2d/3d call
// below materializes its trailing view into a genuine contiguous copy
// (ggml_cont) before returning it, instead of handing callers a strided view
// that aliases the pre-narrowing tensor's own buffer. Costs one small memcpy
// per narrowed tensor per seam-layer call (a handful of times per prefill,
// not per token) -- negligible against the dispatch cost this trim is
// chasing.
//
// Purpose: isolate whether the view itself -- offset arithmetic, stride
// preservation, aliasing with the parent's buffer -- is implicated, as
// opposed to something downstream of it. If turning this on makes an
// otherwise-reproducing fault disappear, the view/aliasing mechanism is
// proven at fault (a real GGML op reading through a narrowed inpL view is
// doing something a genuinely materialized, ordinarily-allocated tensor of
// the same shape would not); if the fault persists identically with this on,
// the view construction itself is exonerated and whatever narrow inpL feeds
// downstream (build_hc_mixes/build_hc_pre/build_moe_ffn/expert dispatch) is
// implicated instead, independent of whether its input happens to be a view.
static bool dsv41_ced_cont_enabled() {
    static const bool enabled = []() {
        const char * e = std::getenv("WP_DSV41_CED_CONT");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

// Trailing-token views of tensors the trunk loop just built itself (inpL/
// residual, and build_hc_mixes' attn_pre/post/comb outputs) at the seam layer.
// All of these are freshly allocated, contiguous tensors going into the seam
// (see the seam-layer comment in graph::graph()), and hc_pre/hc_post/hc_mixes
// are strictly per-token/row-local (no cross-token mixing), so slicing their
// OUTPUT to the last `w` tokens here is exactly the tensor those ops would
// have produced had they been fed an already-trimmed input -- not an
// approximation, just done after the (cheap) full-width compute instead of
// before it, matching build_hc_mixes' own existing "compute full, slice at
// point of use" pattern for its outputs.
static ggml_tensor * dsv41_ced_trailing_2d(ggml_context * ctx0, ggml_tensor * x, int64_t w, int64_t offset) {
    ggml_tensor * v = ggml_view_2d(ctx0, x, x->ne[0], w, x->nb[1], offset * x->nb[1]);
    return dsv41_ced_cont_enabled() ? ggml_cont(ctx0, v) : v;
}

static ggml_tensor * dsv41_ced_trailing_3d(ggml_context * ctx0, ggml_tensor * x, int64_t w, int64_t offset) {
    ggml_tensor * v = ggml_view_3d(ctx0, x, x->ne[0], x->ne[1], w, x->nb[1], x->nb[2], offset * x->nb[2]);
    return dsv41_ced_cont_enabled() ? ggml_cont(ctx0, v) : v;
}

// CED prefill trim (WP_DSV41_CED_PREFILL): every DSV4 raw/compressed kq_mask
// is [n_kv, n_query_tokens/n_stream, 1, n_stream] and every k_idxs is
// [n_query_tokens] (see llm_graph_input_dsv4_raw / comp_input in
// llama-graph.h) -- the query-token axis, not the graph's n_tokens. A trimmed
// decoder-layer `cur` (see graph::graph()) carries fewer query tokens than
// these graph-wide inputs were built for, so a trailing view of that axis is
// needed wherever they are read. These helpers are no-ops (return the input
// unchanged) whenever the axis already matches, so every call site below
// behaves exactly as before when the trim is off -- there is no separate
// "trim active" flag threaded through this file's attention path at all.
// mask is a graph-wide INPUT LEAF (built once by build_inp_dsv4(), filled by
// llama_kv_cache::set_input_kq_mask), not a computed tensor -- unlike inpL/
// cur/attn_pre/post/comb, which are freshly computed by this file's own hc_*
// helpers every layer. ggml_cont here materializes the narrowed view into an
// ordinary computed tensor before any GPU consumer (build_attn_mha) reads it,
// so the same input-leaf-view hazard the raw k_idxs view turned out to have
// (see dsv41_ced_raw_k_idxs_for_nt below, and the report) cannot apply here
// too: the attention kernel that reads this mask sees a normal intermediate
// tensor, not a view whose data pointer is offset into a host-resident leaf's
// own buffer. Costs one small copy per narrowed layer, same trade discussed
// for WP_DSV41_CED_CONT.
static ggml_tensor * dsv41_ced_mask_for_nt(ggml_context * ctx0, ggml_tensor * mask, ggml_tensor * trailing, int64_t nt) {
    if (mask == nullptr || mask->ne[1] == nt) {
        return mask;
    }
    // NaN fix (2026-09-22): prefer the dedicated, directly host-filled
    // ced_kq_mask_trailing tensor graph::graph() builds whenever the trim is
    // active (see llm_graph_input_dsv4_raw::ced_kq_mask_trailing's doc
    // comment) -- it is filled by a host-to-host memcpy in set_input(), not
    // a GPU op reading a view of the graph-wide leaf, which is what this
    // function used to do below and what turned out to corrupt some rows of
    // the narrowed copy (see the nan-fix report).
    if (trailing != nullptr) {
        GGML_ASSERT(trailing->ne[1] == nt &&
                "CED prefill trim: dedicated trailing mask tensor was built at the wrong width");
        return trailing;
    }
    // Fallback only: every call site below passes a non-null `trailing`
    // whenever mask->ne[1] != nt can happen (i.e. whenever the trim is
    // active), so this GGML_ASSERT should never actually fire in practice --
    // kept as a loud failure instead of silently falling back to the known-
    // bad view+cont path if some future call site forgets to build one.
    GGML_ASSERT(false && "CED prefill trim: mask needs narrowing but no dedicated trailing tensor was built");
    return nullptr;
}

// Selects which k_idxs tensor a raw-window cpy_k call should use: the full,
// unmodified graph-wide leaf when this layer isn't narrowed (nt == the
// leaf's own width -- every encoder-range layer, and the whole graph when
// the trim is off), or the SEPARATE, dedicated, already-correctly-sized
// ced_k_idxs_trailing tensor (built by graph::graph(), see
// llm_graph_input_dsv4_raw's doc comment) when it is. No ggml_view_1d of
// the leaf anywhere -- that view is exactly what corrupted this write (see
// the report): self_k_idxs is host-resident (set_input_k_idxs requires
// ggml_backend_buffer_is_host), and a per-layer, freshly-built view of it
// feeding a GPU-side ggml_set_rows gave the backend scheduler a new,
// uncached cross-backend materialization to get right on every narrowed
// layer, which it did not. ced_k_idxs_trailing sidesteps the question
// entirely: it is its own ordinary graph input, filled directly at its own
// correct width, never a view of anything.
static ggml_tensor * dsv41_ced_raw_k_idxs_for_nt(llm_graph_input_dsv4_raw * inp_attn, int64_t nt) {
    ggml_tensor * full = inp_attn->get_k_idxs();
    if (full == nullptr || full->ne[0] == nt) {
        return full;
    }
    GGML_ASSERT(inp_attn->ced_k_idxs_trailing != nullptr && inp_attn->ced_k_idxs_trailing->ne[0] == nt &&
            "CED prefill trim: narrowed raw-window write requested but ced_k_idxs_trailing wasn't built at this width");
    return inp_attn->ced_k_idxs_trailing;
}

// CED prefill trim observability: a runtime bounds check on the actual index
// *values* handed to a cpy_k WRITE, or a set_rows/get_rows-style GATHER,
// gated behind WP_DSV41_CED_PREFILL the same way as the trim itself.
//
// Covers two distinct hazards this file's own bounds no longer guarantee once
// the trim is active:
//  - write indices (k_idxs into a cpy_k call): checks that skipping the
//    writes for the tokens a decoder layer doesn't process under the trim
//    never leaves a k_idxs value pointing outside the cache's actual
//    scatter-destination extent. `limit` here is get_write_capacity() --
//    the destination's own row count (physical size times stream count),
//    NOT get_n_kv() -- an earlier version of this check used get_n_kv() and
//    it is the wrong bound: get_n_kv() is a separately-computed masking/
//    attention-read-window size that is not guaranteed to be >= every valid
//    physical write-slot index (see the crash-report investigation, which
//    also found the CSA compressed-state write was calling cpy_k completely
//    unguarded by any check at all).
//  - gather indices (the indexer's top-k output, used by build_top_k_mask,
//    deepseek4.cpp, to select which compressed positions a query is allowed
//    to see): `limit` is n_comp (inp_comp.kq_mask->ne[0]), the exact size
//    build_top_k_mask's own scatter destination is built to -- a graph-local
//    temporary, not a persistent cache, so no get_n_kv()-vs-capacity
//    distinction applies here. If anything upstream of the indexer score
//    produces NaNs, ggml_top_k's comparison-based selection over NaN is not
//    guaranteed to stay in range, and a garbage index handed to a
//    set_rows-style scatter is exactly the shape of an async out-of-bounds
//    memory fault.
//
// Runs on every call site while the env var is on, not just the ones the
// trim actually narrows -- mirrors dsv41_ced_mask_for_nt/_k_idxs_for_nt's own
// no-separate-"trim active"-flag design above -- and is cheap (an O(n) scan
// of at most a few hundred indices) and silent unless a value actually trips
// it, using the same change-only logging discipline as the gate decision
// (dsv41_ced_log_if_changed, channel 1) so a whole prefill costs at most a
// handful of extra log lines even if something is wrong on every ubatch.
struct dsv41_ced_idx_check_ctx {
    int64_t      limit;
    int32_t      il;
    const char * label; // always a string literal (e.g. "raw"/"lid"/"top_k") -- static storage, no lifetime concern
};

static void dsv41_ced_check_idxs_cb(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return; // n_tasks == 1 below, but guard regardless of how a backend schedules it
    }

    const auto * cx = (const dsv41_ced_idx_check_ctx *) userdata;
    const int64_t n = ggml_nelements(a);

    // Both k_idxs (I64, cpy_k) and ggml_top_k's output (I32, the indexer
    // gather) reach here -- ggml_set_rows itself accepts either, so this
    // mirrors that rather than assuming one width.
    for (int64_t i = 0; i < n; ++i) {
        int64_t v;
        if (a->type == GGML_TYPE_I64) {
            v = ((const int64_t *) a->data)[i];
        } else {
            GGML_ASSERT(a->type == GGML_TYPE_I32 && "CED prefill trim: index bounds check expects an I32 or I64 index tensor");
            v = ((const int32_t *) a->data)[i];
        }

        if (v < 0 || v >= cx->limit) {
            char msg[224];
            std::snprintf(msg, sizeof(msg),
                    "CED prefill trim: index OOB cache=%s il=%d row=%lld value=%lld limit=%lld over_by=%lld",
                    cx->label, cx->il, (long long) i, (long long) v, (long long) cx->limit,
                    (long long) (v >= cx->limit ? (v - cx->limit + 1) : -(v + 1)));
            dsv41_ced_log_if_changed(msg, 1);
            break; // one report per triggering call is enough to localise it
        }
    }
}

// Checks `idxs` as a wholly SEPARATE, forward-expanded side node -- idxs
// itself is returned to callers unmodified and untouched, never spliced into
// its own consumer's data path. This is deliberately NOT what an earlier
// version of this function did: that version used ggml_map_custom1_INPLACE
// and returned the wrapped view for the caller to hand to cpy_k in idxs'
// place. That turned out to be a real bug, found via WP_NODE_TRACE_LIVE, not
// a theory -- see the report. GGML_OP_MAP_CUSTOM1 is not a CUDA-supported op
// (ggml-cuda.cu's device-support switch has no case for it, confirmed by
// reading it directly), so the scheduler must run that node on the CPU
// backend. Every layer's raw-window cpy_k reads the SAME shared host input
// leaf (self_k_idxs) directly in the unmodified codebase, which lets the
// backend scheduler's cross-backend-copy machinery materialize ONE GPU-side
// copy of it and reuse that copy for every layer's cpy_k. Splicing in a
// freshly-built, per-layer INPLACE custom-op node ahead of each layer's
// cpy_k instead handed each one a DIFFERENT tensor object aliasing the same
// buffer -- defeating that reuse and forcing ~40 new CPU/GPU transition
// points where there used to be effectively one, each one a fresh
// opportunity for the copy the scheduler generates to be stale, wrong-sized,
// or simply never populated correctly for that specific split. The node
// WP_NODE_TRACE_LIVE caught faulting (an ordinary layer-2 raw-window cpy_k,
// nowhere near anything CED narrows) is consistent with exactly that: a
// scheduler-generated cross-backend copy feeding the real consumer with
// garbage, while this check's own callback -- reading the original,
// correctly-filled host buffer via the same alias -- saw valid data and
// never had anything to report. Keeping the check itself (it is not useless:
// it still verifies the CPU-side values are correct) but no longer letting
// it sit on the path to the actual scatter.
static void dsv41_ced_check_indices(ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * idxs, int64_t limit, int il, const char * label) {
    if (!dsv41_ced_prefill_env_enabled() || idxs == nullptr) {
        return;
    }

    // Small, deliberately leaked per call: ggml_map_custom1's own op_params
    // slot is already spoken for internally (ggml_map_custom1_impl stores the
    // {fun, n_tasks, userdata} triple there itself to dispatch the callback --
    // writing into it ourselves would corrupt that), so `userdata` is the
    // only channel this API gives a callback to receive per-call context
    // through, and it must outlive graph build (this call returning) until
    // graph execution (when the callback actually runs). At most a few dozen
    // of these are allocated per graph build, only while
    // WP_DSV41_CED_PREFILL=1, so the leak is bounded -- this is the same
    // userdata-ownership shape already used for CPU-side custom ops in
    // pipe-expert-dispatch-graph.cpp (there the context is long-lived and
    // reused instead of leaked; for a value that is never freed, the effect
    // is the same).
    auto * cx = new dsv41_ced_idx_check_ctx{ limit, il, label };

    // Non-inplace: a fresh, independent tensor, not a view of idxs. Nothing
    // else in the graph reads it -- it exists purely so the scheduler has a
    // reason to execute the callback -- so it must be forward-expanded
    // explicitly, unlike the old inplace version, which got pulled in
    // automatically as cpy_k's own input dependency.
    ggml_tensor * checked = ggml_map_custom1(ctx0, idxs, dsv41_ced_check_idxs_cb, 1, cx);
    ggml_build_forward_expand(gf, checked);
}

// CED prefill trim observability: log a tensor's ne[]/nb[]/type, using the
// same change-only discipline as the rest of this feature's logging
// (dsv41_ced_log_if_changed, channel 2, independent of the gate-decision and
// bounds-check channels). The caller decides which layers/tensors this is
// worth calling for (graph::graph() only asks build_attention_v41 to do this
// for the seam layer and the first decoder layer past it); a null tensor is
// silently skipped.
static void dsv41_ced_log_shape(int il, const char * label, const ggml_tensor * t) {
    if (t == nullptr) {
        return;
    }
    char msg[224];
    std::snprintf(msg, sizeof(msg),
            "CED prefill trim: shape il=%d %s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] type=%d",
            il, label,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], (int) t->type);
    dsv41_ced_log_if_changed(msg, 2);
}

// WP_DSV41_SPARSE_ATTN index construction. Builds the fixed-width per-query
// kv_indices [n_idx, nt] I32 (-1 = pad/skip) that ggml_sparse_attn_dsv4 needs,
// from the SAME (dedicated, non-view) raw_mask tensor the dense path already
// uses for this call -- not a re-derivation of the causal/SWA/replay-floor
// logic, so it automatically honours the CED trim's trailing-row narrowing
// and replay floor exactly as the dense path sees them.
//
// Window part: ggml_top_k(raw_mask, k_win) picks, per query row, the k_win
// highest-valued raw_mask entries (admitted cells are 0.0, others -INFINITY,
// so top_k always prefers every admitted cell first). When a row's true
// window is shorter than k_win (only possible for the first few rows of a
// sequence), the extra picks land on -inf cells; gathering raw_mask's own
// value back at each picked index (via ggml_get_rows on a n_embd=1 reshape)
// and folding "is this pick's mask value -inf?" into the index itself turns
// those extra picks into -1, which both AITER kernels vendored under
// aiter-integration/kernels/{sparse_attention_dsv4,pa_decode_sparse}.py skip
// (kv_indices_ptr entries are compared `slot >= 0`, `other=-1` on the
// index-tile load) -- verified against the vendored kernel bodies, not
// assumed.
//
// Comp part: top_k (the indexer's own picks, when this layer is an index
// source) is already an explicit, fixed-width index list into the n_comp
// compressed positions -- appended verbatim, offset by raw_k_len (k_all's
// layout is raw-then-comp, ggml_concat(raw_k, comp_k, 2)). No layer here has
// a "no top_k but small n_comp" case in DS4.1's actual hparams (every
// ratio!=0 layer is either an index source or a decoder-range reader with an
// unbounded-by-top-k n_comp), so returns nullptr (dense fallback) whenever
// top_k is null -- the conservative half of the coordinator-approved
// "append top_k ... or fall back to dense" instruction, not implemented via
// a size-threshold heuristic.
// Turn candidate indices into kernel indices: gather `mask_f32` (0 = admitted,
// -inf = masked, [n_cells, nt]) at each pick and replace masked picks with -1
// (both AITER kernels skip slot < 0), then add `offset` to the kept ones.
static ggml_tensor * dsv41_sparse_attn_mask_picks(
        ggml_context * ctx0,
        ggml_tensor * mask_f32,
        ggml_tensor * picks,
        int64_t offset) {
    const int64_t nt = mask_f32->ne[1];
    const int64_t k  = picks->ne[0];
    ggml_tensor * mask_1 = ggml_reshape_4d(ctx0, mask_f32, 1, mask_f32->ne[0], nt, 1);
    ggml_tensor * gathered = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, mask_1, picks), k, nt);
    ggml_tensor * valid = ggml_scale_bias(ctx0, ggml_clamp(ctx0, gathered, -1.0f, 0.0f), 1.0f, 1.0f); // 1 kept, 0 masked
    ggml_tensor * picks_f = ggml_scale_bias(ctx0, ggml_cast(ctx0, picks, GGML_TYPE_F32), 1.0f, (float) offset + 1.0f);
    ggml_tensor * out_f = ggml_scale_bias(ctx0, ggml_mul(ctx0, picks_f, valid), 1.0f, -1.0f); // pick+offset or -1
    return ggml_cast(ctx0, out_f, GGML_TYPE_I32);
}

// Window half of a sparse index list: the k_win = min(n_raw, n_swa) raw cells
// raw_mask admits per row, padding picks (-inf cells) turned into -1.
static ggml_tensor * dsv41_sparse_attn_window_indices(
        ggml_context * ctx0,
        ggml_tensor * raw_mask,
        int64_t n_swa) {
    GGML_ASSERT(raw_mask->ne[2] == 1 && raw_mask->ne[3] == 1 &&
            "sparse_attn index build assumes a plain 2D [n_raw, nt] mask");

    const int64_t k_win = std::min<int64_t>(raw_mask->ne[0], n_swa);
    GGML_ASSERT(k_win > 0);

    // ggml_cuda_op_top_k only accepts F32 (ggml-cuda/top-k.cu:220); kq_mask
    // is F16 under flash_attn (matches build_top_k_mask's own
    // `cparams.flash_attn ? F16 : F32` convention elsewhere in this file) --
    // cast defensively rather than assume. get_rows tolerates the source
    // dtype fine on its own, but top_k does not, so this must happen before
    // the top_k call, not just before the later float arithmetic.
    ggml_tensor * raw_mask_f32 = raw_mask->type == GGML_TYPE_F32
        ? raw_mask : ggml_cast(ctx0, raw_mask, GGML_TYPE_F32);

    ggml_tensor * win_idx = ggml_top_k(ctx0, raw_mask_f32, (int) k_win); // I32 [k_win, nt]
    return dsv41_sparse_attn_mask_picks(ctx0, raw_mask_f32, win_idx, 0);
}

static ggml_tensor * dsv41_sparse_attn_build_indices(
        ggml_context * ctx0,
        ggml_tensor * raw_mask,
        ggml_tensor * comp_mask,
        ggml_tensor * top_k,
        int64_t n_swa,
        int64_t raw_k_len) {
    if (!top_k) {
        return nullptr; // dense fallback: no bounded index list to build
    }
    const int64_t nt = raw_mask->ne[1];
    ggml_tensor * win_idx_masked = dsv41_sparse_attn_window_indices(ctx0, raw_mask, n_swa);

    // Compressed half: the dense path admits a top-k pick only where the causal
    // compressed mask is also 0 (build_top_k_mask adds kq_mask), so a pick of a
    // not-yet-visible position -- top_k over fewer than indexer_top_k visible
    // entries -- must be dropped here too, not attended.
    GGML_ASSERT(comp_mask->ne[1] == nt && comp_mask->ne[2] == 1 && comp_mask->ne[3] == 1);
    ggml_tensor * comp_mask_f32 = comp_mask->type == GGML_TYPE_F32
        ? comp_mask : ggml_cast(ctx0, comp_mask, GGML_TYPE_F32);
    ggml_tensor * comp_idx = dsv41_sparse_attn_mask_picks(ctx0, comp_mask_f32, top_k, raw_k_len); // offset into k_all

    return ggml_concat(ctx0, win_idx_masked, comp_idx, 0); // [k_win + top_k->ne[0], nt] I32
}

// WP_DSV41_SPARSE_ATTN_CHECK: index-equivalence proof. Reads back
// kv_indices (the sparse op's constructed index set) and raw_mask (the SAME
// dense mask the non-sparse path admits columns from) on the CPU and checks,
// for a small sample of rows, that the window half of kv_indices names
// exactly the raw_mask-admitted cells for that row -- no more (every -1 in
// [0,k_win) really is a padding slot, verified by checking raw_mask at that
// same row/column is -inf) and no fewer (the count of non-(-1) entries in
// [0,k_win) equals the count of admitted (0-valued) raw_mask entries in that
// row). Logged via the same change-only discipline as the CED checks above.
struct dsv41_sparse_attn_check_ctx {
    int32_t il;
    int64_t k_win;
    int64_t n_raw;
};

static void dsv41_sparse_attn_check_cb(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
        int ith, int nth, void * userdata) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const auto * cx = (const dsv41_sparse_attn_check_ctx *) userdata;
    const ggml_tensor * kv_indices = a; // I32 [n_idx, nt]
    const ggml_tensor * raw_mask   = b; // F16/F32 [n_raw, nt]
    const int64_t nt = kv_indices->ne[1];

    auto mask_admits = [&](int64_t kv, int64_t t) -> bool {
        const char * row = (const char *) raw_mask->data + t * raw_mask->nb[1];
        if (raw_mask->type == GGML_TYPE_F16) {
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) row)[kv]) == 0.0f;
        }
        GGML_ASSERT(raw_mask->type == GGML_TYPE_F32);
        return ((const float *) row)[kv] == 0.0f;
    };

    // Sample the first and last query row of this ubatch -- enough to catch
    // a systematic construction bug without an O(n_idx * nt) scan every call.
    const int64_t rows[2] = { 0, nt - 1 };
    for (int64_t ri = 0; ri < (nt > 1 ? 2 : 1); ++ri) {
        const int64_t t = rows[ri];
        const int32_t * idx_row = (const int32_t *) ((const char *) kv_indices->data + t * kv_indices->nb[1]);

        int64_t admitted_in_mask = 0;
        for (int64_t kv = 0; kv < cx->n_raw; ++kv) {
            if (mask_admits(kv, t)) admitted_in_mask++;
        }

        int64_t admitted_in_idx = 0;
        int64_t bad_picks = 0;
        for (int64_t i = 0; i < cx->k_win; ++i) {
            const int32_t v = idx_row[i];
            if (v < 0) continue;
            admitted_in_idx++;
            if (v >= cx->n_raw || !mask_admits(v, t)) {
                bad_picks++;
            }
        }

        char msg[256];
        if (admitted_in_idx != admitted_in_mask || bad_picks != 0) {
            std::snprintf(msg, sizeof(msg),
                    "SPARSE_ATTN_CHECK MISMATCH il=%d row=%lld admitted_mask=%lld admitted_idx=%lld bad_picks=%lld",
                    cx->il, (long long) t, (long long) admitted_in_mask, (long long) admitted_in_idx, (long long) bad_picks);
            dsv41_ced_log_if_changed(msg, 3);
        } else {
            std::snprintf(msg, sizeof(msg),
                    "SPARSE_ATTN_CHECK ok il=%d row=%lld admitted=%lld/%lld",
                    cx->il, (long long) t, (long long) admitted_in_idx, (long long) cx->k_win);
            dsv41_ced_log_if_changed(msg, 3);
        }
    }
}

// WP_DSV41_SPARSE_ATTN_DIFF=1 (debug, needs WP_DSV41_SPARSE_ATTN=1): per layer,
// log the sparse output's error against the dense output for the first few
// prefill and decode calls. a = sparse, b = dense, both F32 [512*64, nt].
static bool dsv41_sparse_attn_diff_env_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("WP_DSV41_SPARSE_ATTN_DIFF");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

struct dsv41_sparse_attn_diff_ctx {
    int     il;
    int     kind;  // 0 = prefill kernel call (nt >= 32), 1 = decode kernel
    int64_t nt;    // full call width
    int64_t keep;  // trailing rows handed to the callback
    float   scale;
};

// src: [0] sparse tail F32 [512*64, keep], [1] dense tail (same), [2] q tail [512, 64, keep],
// [3] k_all F16 [512, 1, n_kv], [4] kv_indices tail I32 [n_idx, keep], [5] sinks F32 [64].
// Logs each path's error against a float64 attention over the same indices.
static void dsv41_sparse_attn_diff_cb(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const auto * cx = (const dsv41_sparse_attn_diff_ctx *) userdata;
    static std::atomic<int> n_logged[LLAMA_MAX_LAYERS][2];
    if (n_logged[cx->il][cx->kind].fetch_add(1) >= 3) {
        return;
    }
    const ggml_tensor * sp  = dst->src[0];
    const ggml_tensor * dn  = dst->src[1];
    const ggml_tensor * q   = dst->src[2];
    const ggml_tensor * k   = dst->src[3];
    const ggml_tensor * idx = dst->src[4];
    const ggml_tensor * snk = dst->src[5];
    GGML_ASSERT(sp->type == GGML_TYPE_F32 && dn->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16);
    GGML_ASSERT(idx->type == GGML_TYPE_I32 && snk->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16);

    const int64_t D = 512, H = 64, row = D * H;
    const int64_t n_idx = idx->ne[0];
    auto qv = [&](int64_t t, int64_t h, int64_t d) -> double {
        const char * p = (const char *) q->data + t * q->nb[2] + h * q->nb[1] + d * q->nb[0];
        return q->type == GGML_TYPE_F32 ? *(const float *) p : ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
    };
    auto kv = [&](int64_t n, int64_t d) -> double {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) ((const char *) k->data + n * k->nb[2] + d * k->nb[0]));
    };

    // sparse vs dense over every handed row, and both vs float64 on the last few
    double sd2 = 0.0, dd2 = 0.0, se2 = 0.0, de2 = 0.0, r2 = 0.0;
    const int64_t n_ref = std::min<int64_t>(cx->keep, 4);
    std::vector<double> ref(D), sc;
    std::vector<int32_t> picks;
    for (int64_t r = 0; r < cx->keep; ++r) {
        const float * xs = (const float *) sp->data + r * row;
        const float * xd = (const float *) dn->data + r * row;
        for (int64_t i = 0; i < row; ++i) {
            const double d = (double) xs[i] - (double) xd[i];
            sd2 += d * d;
            dd2 += (double) xd[i] * xd[i];
        }
        if (r < cx->keep - n_ref) {
            continue;
        }
        const int64_t t = r; // q and kv_indices arrive as the same trailing rows
        const int32_t * ir = (const int32_t *) ((const char *) idx->data + t * idx->nb[1]);
        picks.clear();
        for (int64_t j = 0; j < n_idx; ++j) {
            if (ir[j] >= 0) {
                picks.push_back(ir[j]);
            }
        }
        sc.resize(picks.size());
        for (int64_t h = 0; h < H; ++h) {
            double m = ((const float *) snk->data)[h];
            for (size_t j = 0; j < picks.size(); ++j) {
                double s = 0.0;
                for (int64_t d = 0; d < D; ++d) {
                    s += qv(t, h, d) * kv(picks[j], d);
                }
                sc[j] = s * cx->scale;
                m = std::max(m, sc[j]);
            }
            double den = std::exp(((const float *) snk->data)[h] - m);
            std::fill(ref.begin(), ref.end(), 0.0);
            for (size_t j = 0; j < picks.size(); ++j) {
                const double w = std::exp(sc[j] - m);
                den += w;
                for (int64_t d = 0; d < D; ++d) {
                    ref[d] += w * kv(picks[j], d);
                }
            }
            for (int64_t d = 0; d < D; ++d) {
                const double y = ref[d] / den;
                const double es = xs[h * D + d] - y;
                const double ed = xd[h * D + d] - y;
                se2 += es * es;
                de2 += ed * ed;
                r2  += y * y;
            }
        }
    }
    LLAMA_LOG_WARN("SPARSE_ATTN_DIFF il=%d %s nt=%lld sparse_vs_dense=%.3e | vs_f64 (last %lld rows): sparse=%.3e dense=%.3e\n",
            cx->il, cx->kind == 0 ? "prefill" : "decode", (long long) cx->nt,
            dd2 > 0.0 ? std::sqrt(sd2 / dd2) : 0.0, (long long) n_ref,
            r2 > 0.0 ? std::sqrt(se2 / r2) : 0.0, r2 > 0.0 ? std::sqrt(de2 / r2) : 0.0);
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
        int il,
        ggml_tensor * cur_state,
        bool ced_log_shapes,
        bool publish_only) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    // CED prefill trim: `cur` may be narrowed to the trailing W tokens (see
    // graph::graph()); the kv-source/indexer-publish block below always reads
    // `cur_kv` instead, which defaults to `cur` itself (every layer except the
    // trim's seam layer) so nothing changes when the trim is inactive.
    ggml_tensor * cur_kv = cur_state ? cur_state : cur;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);
    // graph::graph()'s gate already refuses the whole trim for a ratio==0
    // decoder-range layer, so publish_only (skip-nonfinal) never reaches here
    // with no compressed stream to publish.
    GGML_ASSERT((!publish_only || ratio != 0) &&
            "CED prefill trim: publish_only requires a compressed-stream (ratio != 0) layer");

    const dsv41_rope_cfg rc = rope_cfg(il);
    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // CED prefill trim (WP_DSV41_CED_SKIP_NONFINAL): publish_only means this
    // ubatch is non-final (n_outputs==0) and only the kv-source/indexer
    // publish block below is needed -- nothing reads this layer's own query
    // output on this ubatch. Skip the RoPE'd q/kv projections, the raw-window
    // write and the attention/top-k below entirely; qr stays null and is
    // never read (build_indexer_top_k, which uses it, is also skipped).
    ggml_tensor * qr = nullptr;
    ggml_tensor * q  = nullptr;
    ggml_tensor * kv = nullptr;
    if (!publish_only) {
        // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
        qr = build_lora_mm(layer.wq_a, cur);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "qr", il);

        q = build_lora_mm(layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
                rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
        cb(q, "q", il);

        // the sliding window KV, which every layer keeps for itself
        kv = build_lora_mm(layer.wkv, cur);
        kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
        kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
                rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
        cb(kv, "kv", il);

        if (ratio == 0) {
            // no compressed stream, so this layer sees only its own window
            ggml_tensor * out = nullptr;
            // WP_DSV41_SPARSE_ATTN: the dense path scans the whole raw cache to use n_swa cells;
            // hand the sparse kernel just the window instead.
            const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;
            if (dsv41_sparse_attn_env_enabled() && n_embd_head == 512 && n_head == 64 &&
                    inp_attn->self_k_rot == nullptr && mctx_raw->get_k(ctx0, il)->type == GGML_TYPE_F16) {
                ggml_build_forward_expand(gf, q);
                ggml_build_forward_expand(gf, kv);
                ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

                ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
                if (!ggml_is_contiguous(raw_k)) {
                    raw_k = ggml_cont(ctx0, raw_k);
                }
                ggml_tensor * kv_indices = dsv41_sparse_attn_window_indices(ctx0, inp_attn->get_kq_mask(), hparams.n_swa);
                ggml_tensor * q_f16 = q->type == GGML_TYPE_F16 ? q : ggml_cast(ctx0, q, GGML_TYPE_F16);

                ggml_tensor * kv_indptr = ggml_arange(ctx0, 0.0f, (float) (nt + 1), 1.0f);
                kv_indptr = ggml_scale(ctx0, kv_indptr, (float) kv_indices->ne[0]);
                kv_indptr = ggml_cast(ctx0, kv_indptr, GGML_TYPE_I32);
                out = ggml_sparse_attn_dsv4(ctx0, q_f16, raw_k, kv_indices, kv_indptr, layer.attn_sinks, kq_scale);
                out = ggml_cast(ctx0, out, GGML_TYPE_F32);
                cb(out, "attn_raw_sparse", il);
            }
            if (!out) {
                out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
            }

            return build_attention_tail(model, out, inp_pos, nt, il);
        }
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

    // CED prefill trim: everything below this point that reads inp_comp.kq_mask
    // is query-side (this layer's own candidate selection / attention), so it
    // uses the trailing-window view when `cur` was narrowed; the kv-source
    // block above already read cur_kv/inp_comp directly and is unaffected.
    // Skipped under publish_only: no query side to build a mask for.
    llm_graph_input_dsv4::comp_input inp_comp_q = inp_comp;
    if (!publish_only) {
        inp_comp_q.kq_mask = dsv41_ced_mask_for_nt(ctx0, inp_comp.kq_mask, inp_comp.ced_kq_mask_trailing, nt);
    }

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur_kv);
        cb(state_kv, "comp_state_kv", il);

        // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below
        // then runs over a single element and returns 1.0 whatever the score holds, so the
        // values reach the cache unweighted, which is what a plain projection means.
        ggml_tensor * state_score = layer.attn_comp_wgate
            ? build_lora_mm(layer.attn_comp_wgate, cur_kv)
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

            // idxs are the ORIGINAL state_write_idxs tensor, unwrapped and
            // unmodified -- see dsv41_ced_check_indices's comment for why this
            // must not splice a graph node into cpy_k's own index argument.
            dsv41_ced_check_indices(ctx0, gf, inp_comp.state_write_idxs,
                    (int64_t) inp_dsv4->mctx->get_lid()->get_write_capacity(), il, "lid");
            // Build-time width check (see the crash-report investigation): if
            // the seam split ever left idx_k narrowed while state_write_idxs
            // stayed full-width, or vice versa, llama_kv_cache::cpy_k reads
            // idx_k->ne[2] as its row count -- assert that against the index
            // count directly, at graph-build time, rather than waiting on a
            // runtime value check. Gated on the env var like every other CED
            // check; a real mismatch would GGML_ABORT here, immediately, with
            // a stack that names this exact line.
            if (dsv41_ced_prefill_env_enabled()) {
                GGML_ASSERT(idx_k->ne[2] == inp_comp.state_write_idxs->ne[0] &&
                        "CED prefill trim: LID cpy_k source row count (idx_k->ne[2]) != index count (state_write_idxs->ne[0])");
            }
            ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(
                        ctx0, idx_k, inp_comp.state_write_idxs, il));
        }

        if (inp_dsv4->get_csa().k_rot) {
            latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
            cb(latent, "comp_kv_rot", il);
        }

        dsv41_ced_check_indices(ctx0, gf, inp_comp.state_write_idxs,
                (int64_t) inp_dsv4->mctx->get_csa()->get_write_capacity(), il, "csa");
        if (dsv41_ced_prefill_env_enabled()) {
            GGML_ASSERT(latent->ne[2] == inp_comp.state_write_idxs->ne[0] &&
                    "CED prefill trim: CSA cpy_k source row count (latent->ne[2]) != index count (state_write_idxs->ne[0])");
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

    if (publish_only) {
        // CED prefill trim (WP_DSV41_CED_SKIP_NONFINAL): the publish above is
        // all this non-final ubatch needs from this layer -- no query, no
        // raw-window write, no attention/top-k/MoE. The caller (graph::graph()'s
        // ced_skip_layer) discards the return value.
        return nullptr;
    }

    // an index source picks the positions; the layers in between reuse what it picked
    ggml_tensor * top_k = nullptr;
    if (hparams.dsv41_is_index_source(il)) {
        top_k = build_indexer_top_k(model, inp_dsv4, inp_comp_q, qr, cur, inp_pos, il);
        dsv41_shared_top_k     = top_k;
        dsv41_shared_top_k_src = il;
    } else if (dsv41_topk_reuse_env_enabled() &&
               hparams.dsv41_topk_source[il] >= 0 &&
               hparams.dsv41_topk_source[il] == dsv41_shared_top_k_src &&
               dsv41_shared_top_k != nullptr &&
               dsv41_shared_top_k->ne[1] == nt) {
        // Reference model.py: a compressed layer that is not an index source
        // attends to the positions its source picked (shared_attn.topk_idxs),
        // not to every compressed position. The source's comp rows are the
        // same ones this layer reads (kv-cache reuse aliasing), so the indices
        // are valid here unchanged.
        top_k = dsv41_shared_top_k;
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    // The value cpy_k actually receives -- the full graph-wide leaf, or the
    // dedicated ced_k_idxs_trailing tensor, never a view of the leaf (see
    // dsv41_ced_raw_k_idxs_for_nt's comment). The bounds check below reads
    // this SAME tensor as a separate side node -- it must not wrap or
    // replace it (see dsv41_ced_check_indices's comment).
    ggml_tensor * raw_write_idxs = dsv41_ced_raw_k_idxs_for_nt(inp_attn, nt);
    dsv41_ced_check_indices(ctx0, gf, raw_write_idxs, (int64_t) mctx_raw->get_write_capacity(), il, "raw");
    if (dsv41_ced_prefill_env_enabled()) {
        GGML_ASSERT(kv->ne[2] == raw_write_idxs->ne[0] &&
                "CED prefill trim: raw-window cpy_k source row count (kv->ne[2]) != index count (k_idxs->ne[0])");
    }
    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, raw_write_idxs, il));

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

    ggml_tensor * raw_mask = dsv41_ced_mask_for_nt(ctx0, inp_attn->get_kq_mask(), inp_attn->ced_kq_mask_trailing, nt);

    // Gather-bounds check on the indexer's top-k output: these values select
    // which of the n_comp compressed positions build_top_k_mask (deepseek4.cpp)
    // scatters into the comp mask below. See dsv41_ced_check_indices's
    // comment -- this is the coordinator's leading hypothesis for the
    // original crash, not just hardening: if anything upstream of the
    // indexer score is NaN-contaminated (e.g. by the raw-window replay gap
    // closed at the source in llama_kv_cache::set_input_kq_mask's
    // ced_replay_floor -- see graph::graph() below), ggml_top_k's selection
    // over NaN is not guaranteed to stay in range, and a garbage index
    // feeding a set_rows-style scatter is exactly the shape of an async OOB
    // memory fault.
    dsv41_ced_check_indices(ctx0, gf, top_k, n_comp, il, "top_k");

    ggml_tensor * comp_mask = top_k
        ? build_top_k_mask(inp_comp_q.kq_mask, top_k, "comp_top_k_mask", il)
        : inp_comp_q.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
    cb(kq_mask, "kq_mask", il);

    // CED prefill trim, multi-ubatch fix (round 3): a non-index-source
    // decoder layer has no top_k cap, so it used to pass n_kv_max=0 ("no
    // explicit cap", the ordinary dense-mask fallback every non-index-source
    // V4/V4.1 layer already used before CED existed) -- safe normally
    // because every physical raw-KV cell the dense kernel scans is finite
    // (the untrimmed path writes all of them). Under the trim, a non-final
    // ubatch's decoder-range layers only write raw KV for their own
    // trailing W rows (see dsv41_ced_outputs_in_window below); cells outside
    // that range are never written for THAT ubatch, and per IEEE-754
    // -inf + NaN = NaN, a masked-but-NaN cell the dense (n_kv_max=0) kernel
    // still touches poisons the whole row's softmax even though the mask
    // marks it invisible (this was the confirmed root cause of the original
    // multi-ubatch crash, src/models/deepseek41.cpp's dsv41_ced_outputs_in_window
    // doc comment / scratchpad/ced-multiubatch-fix-report.md). The compressed
    // side has no such gap -- the seam layer's kv-source/indexer-publish
    // block always runs full-width regardless of trim (see cur_kv above), so
    // every one of the n_comp compressed positions any row could see is
    // always finite. So: give every layer a real n_kv_max bound instead of
    // falling back to the dense scan -- min(raw_mask->ne[0], n_swa) raw
    // entries (the true SWA-visible count, always <= what's actually
    // written, trim or not) plus either top_k's exact count or, when there
    // is no top-k cap, n_comp itself (every comp entry, which is always
    // finite). This is the SAME true finite-entry count the dense fallback
    // was already relying on implicitly; the only change is telling the
    // kernel the bound explicitly, so it uses the sparse path (which looks
    // up finite entries directly, per ggml.h's contract) instead of touching
    // every column unconditionally. Gated on the CED env var, matching every
    // other change in this file's discipline: zero behavior change with the
    // trim off, and even with the trim on this is a strict tightening (a
    // real bound instead of "no bound"), never a relaxation.
    const int64_t n_kv_max = dsv41_ced_prefill_env_enabled()
        ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + (top_k ? top_k->ne[0] : n_comp)
        : (top_k ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0] : 0);

    if (ced_log_shapes) {
        dsv41_ced_log_shape(il, "q", q);
        dsv41_ced_log_shape(il, "k_all", k_all);
        dsv41_ced_log_shape(il, "kq_mask", kq_mask);
        char msg[96];
        std::snprintf(msg, sizeof(msg), "CED prefill trim: shape il=%d n_kv_max=%lld", il, (long long) n_kv_max);
        dsv41_ced_log_if_changed(msg, 2);
    }

    // WP_DSV41_SPARSE_ATTN: route through AITER's native sparse kernels
    // (ggml_sparse_attn_dsv4, HIP-only -- see ggml-cuda/mt_sparse_attn_dsv4.{cuh,cu}
    // and aiter-integration/wrappers/mt_sparse_attn_dsv4.{h,cpp}) instead of
    // the dense build_attn_mha scan below. Gated on: the knob being on, the
    // baked AOT shape assumptions matching DS4.1's own fixed hparams
    // (n_embd_head==512, n_head==64 -- true for every DS4.1 layer today;
    // this is a "should never fail" defensive check, not a per-call runtime
    // fallback), and top_k being non-null (see dsv41_sparse_attn_build_indices's
    // doc comment for why the no-top_k case always falls back to dense).
    // ggml_backend_cuda_supports_op independently re-checks tensor dtype/
    // contiguity at schedule time and would refuse the op if this graph ever
    // built one it can't actually run -- this is belt-and-suspenders, not the
    // only gate.
    // 512/64 mirror MT_SPARSE_ATTN_DSV4_HEAD_DIM/MT_SPARSE_ATTN_DSV4_NUM_HEADS
    // in aiter-integration/wrappers/mt_sparse_attn_dsv4.h (not included here --
    // that header pulls in hip_runtime_api.h, only available in the HIP-only
    // ggml-cuda/aiter-integration translation units, not this host-side file).
    ggml_tensor * out = nullptr;
    ggml_tensor * sparse_idx = nullptr; // kept for WP_DSV41_SPARSE_ATTN_DIFF's reference
    ggml_tensor * sparse_k   = nullptr;
    if (dsv41_sparse_attn_env_enabled() &&
            n_embd_head == 512 &&
            n_head       == 64) {
        ggml_tensor * kv_indices = dsv41_sparse_attn_build_indices(
                ctx0, raw_mask, inp_comp_q.kq_mask, top_k, hparams.n_swa, raw_k->ne[2]);
        if (kv_indices) {
            ggml_tensor * q_f16 = q->type == GGML_TYPE_F16 ? q : ggml_cast(ctx0, q, GGML_TYPE_F16);
            ggml_tensor * k_all_f16 = k_all->type == GGML_TYPE_F16 ? k_all : ggml_cast(ctx0, k_all, GGML_TYPE_F16);

            if (dsv41_sparse_attn_check_env_enabled()) {
                // Small, deliberately leaked per call -- same userdata-ownership
                // shape as dsv41_ced_check_indices above (must outlive graph
                // build until graph execution runs the callback).
                auto * cx = new dsv41_sparse_attn_check_ctx{
                    il, std::min<int64_t>(raw_mask->ne[0], hparams.n_swa), raw_mask->ne[0] };
                ggml_tensor * checked = ggml_map_custom2(ctx0, kv_indices, raw_mask,
                        dsv41_sparse_attn_check_cb, 1, cx);
                ggml_build_forward_expand(gf, checked);
            }

            // Uniform row offsets [0, n_idx, ..., nt*n_idx] built in-graph (exact in
            // F32 for nt*n_idx < 2^24), so the op needs no host copy or stream sync
            // and stays safe under HIP graph capture.
            const int64_t nt_q = q->ne[2];
            ggml_tensor * kv_indptr = ggml_arange(ctx0, 0.0f, (float) (nt_q + 1), 1.0f);
            kv_indptr = ggml_scale(ctx0, kv_indptr, (float) kv_indices->ne[0]);
            kv_indptr = ggml_cast(ctx0, kv_indptr, GGML_TYPE_I32);
            out = ggml_sparse_attn_dsv4(ctx0, q_f16, k_all_f16, kv_indices, kv_indptr, layer.attn_sinks, kq_scale);
            sparse_idx = kv_indices;
            sparse_k   = k_all_f16;
            // ggml_flash_attn_ext (the dense path below) always returns
            // GGML_TYPE_F32 (ggml.c:5723) -- downstream consumers of `out`
            // (build_attention_tail, then whatever FFN-side op reads it next,
            // e.g. the ml8-4 fp8_quant_rot path) are built against that
            // contract. The AITER kernel itself genuinely writes F16 bytes
            // (matches its real output buffer, verified in Stage 2), so the
            // op's tensor stays F16 -- this cast is a real, separate
            // conversion node, not a type-tag mismatch. Its absence was the
            // root cause of a `ggml-ml8.c:425 GGML_ASSERT(x->type == F32 ||
            // x->type == BF16)` abort hit during this turn's first load
            // attempt (isolated by reloading with the knob off, which loaded
            // clean -- confirming this path, not something pre-existing).
            out = ggml_cast(ctx0, out, GGML_TYPE_F32);
            cb(out, "attn_out_sparse", il);
        }
    }
    if (out && dsv41_sparse_attn_diff_env_enabled()) {
        // Debug: build the dense path too, log how far the sparse output is from
        // it (last 64 token rows only, so prefill stays cheap), and carry the
        // dense output forward so every layer compares on the same inputs.
        ggml_tensor * dense = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
                nullptr, n_kv_max, kq_scale, il);
        GGML_ASSERT(ggml_nelements(dense) == ggml_nelements(out));
        const int64_t row  = n_embd_head * n_head;
        const int64_t keep = std::min<int64_t>(nt, 64);
        ggml_tensor * out_2d   = ggml_reshape_2d(ctx0, out, row, nt);
        ggml_tensor * dense_2d = ggml_reshape_2d(ctx0, dense, row, nt);
        ggml_tensor * out_tail   = ggml_view_2d(ctx0, out_2d, row, keep, out_2d->nb[1], (nt - keep) * out_2d->nb[1]);
        ggml_tensor * dense_tail = ggml_view_2d(ctx0, dense_2d, row, keep, dense_2d->nb[1], (nt - keep) * dense_2d->nb[1]);
        // the kind (prefill vs decode) is judged on the full call, so pass nt along
        ggml_tensor * args[] = {
            ggml_cont(ctx0, out_tail), ggml_cont(ctx0, dense_tail),
            ggml_cont(ctx0, ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], keep, q->nb[1], q->nb[2], (nt - keep) * q->nb[2])),
            sparse_k,
            ggml_cont(ctx0, ggml_view_2d(ctx0, sparse_idx, sparse_idx->ne[0], keep, sparse_idx->nb[1], (nt - keep) * sparse_idx->nb[1])),
            layer.attn_sinks,
        };
        auto * dcx = new dsv41_sparse_attn_diff_ctx{ il, nt >= 32 ? 0 : 1, nt, keep, kq_scale };
        ggml_tensor * diffed = ggml_custom_4d(ctx0, GGML_TYPE_F32, 1, 1, 1, 1, args, 6,
                dsv41_sparse_attn_diff_cb, 1, dcx);
        ggml_build_forward_expand(gf, diffed);
        out = dense;
    }
    if (!out) {
        out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
                nullptr, n_kv_max, kq_scale, il);
    }
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

    // CED prefill trim (WP_DSV41_CED_PREFILL, default OFF -- with it unset the
    // graph built below is identical to before this feature existed). DeepSeek-
    // V4.1 is a Causal Encoder-Decoder: layers [0, ced_seam) are the encoder,
    // [ced_seam, n_layer) the decoder, where ced_seam is the LAST kv-source
    // layer (dsv41_kv_source_layer_ids -- derived per-model, not hardcoded to
    // "20"). Per the model's own card, most prompt tokens are only ever
    // processed by the encoder: the decoder's compressed/global KV is
    // projected once from the encoder's final hidden state (the seam layer's
    // kv-source/indexer-publish block, which always runs over every token --
    // see cur_kv in build_attention_v41), and the decoder layers' own sliding-
    // window KV is exactly what decode itself would rebuild by replaying only
    // the last n_swa tokens ("SWA Bounded Replay"). Prefill currently pays the
    // full decoder cost over the whole prompt; this trims the decoder layers'
    // QUERY-side compute -- their own attention/MoE, NOT the kv-source/index-
    // key publish -- down to that same trailing window, matching what a
    // subsequent decode step would already have rebuilt.
    //
    // Every condition below is a REFUSAL, decided fresh per ubatch: any doubt
    // and this silently falls back to the untrimmed path, which is what makes
    // WP_DSV41_CED_PREFILL=1 safe to leave on in production while A/B testing.
    int     ced_seam   = -1;   // first decoder layer, i.e. last kv-source layer id
    bool    ced_trim   = false;
    int64_t ced_w      = 0;    // trim window width; hparams.n_swa unless WP_DSV41_CED_W overrides it
    int64_t ced_offset = 0;    // n_tokens - ced_w
    // WP_DSV41_CED_SKIP_NONFINAL: true when this ubatch is non-final
    // (n_outputs==0, nothing reads its decoder-range output) and the trim is
    // applied -- see dsv41_ced_skip_nonfinal_enabled() and the ced_skip_layer
    // use below. false whenever ced_trim is false, so it is always safe to
    // read regardless of the env var.
    bool    ced_skip_nonfinal = false;

    // ced_reason names the single condition that refused the trim for this
    // ubatch (or "applied"), for the log line below -- see
    // dsv41_ced_log_if_changed(). Never read/written when the env var is off.
    char ced_reason[160] = "";

    if (dsv41_ced_prefill_env_enabled()) {
        const bool gtype_ok        = params.gtype != LLM_GRAPH_TYPE_ENCODER && params.gtype != LLM_GRAPH_TYPE_DECODER_DSPARK;
        const bool const_shape_off = !ds4_const_shape_enabled();             // mutually exclusive with topology pinning
        const bool tokens_only     = ubatch.embd == nullptr;                 // tokens only, no VL/embd input
        // ubatch.n_seqs_unq is the count of DISTINCT sequence ids actually present
        // in the ubatch (llama_batch_allocr::ubatch_add derives it from seq_id_unq,
        // regardless of split mode) -- this is what the KV cache/graph code
        // elsewhere treats as the stream count (dsv4_comp_graph_n_stream,
        // llama-graph.cpp's n_stream, deepseek4.cpp's n_blocks). ubatch.n_seqs is
        // NOT a stream count: it is "sequence SETS in the ubatch", and for the
        // split_simple() ubatches DS4.1 always builds (llama-kv-cache-dsv4.cpp,
        // b_equal_seqs=false), each token is its own set, so n_seqs == n_tokens for
        // any multi-token single-sequence prefill -- requiring n_seqs == 1 as well
        // made this refuse every real prefill. Dropped.
        const bool single_stream   = ubatch.n_seqs_unq == 1;
        const bool pos_flat        = hparams.n_pos_per_embd() == 1;          // a flat trailing view of inp_pos must be valid
        const bool has_kv_sources  = hparams.dsv41_n_kv_sources > 0;
        const bool multi_token     = n_tokens > 1;

        // NOTE: whether the DSpark HEAD WEIGHTS are loaded (model.fc != nullptr) is
        // deliberately NOT checked here. The DeepSeek-V4.1 spine GGUF ships those
        // tensors unconditionally, so model.fc != nullptr for every DS4.1 model this
        // trim will ever run against, loaded or not, speculative decoding on or off --
        // gating on it made the trim permanently dead. The actual hazard (DSpark's
        // drafter reading a decoder-range hidden-state tap computed only over the
        // trimmed trailing window) is already caught below, correctly, by the
        // embeddings_layer_inp scan over [ced_seam, n_layer]: common_speculative's
        // DFlash/EAGLE3/DSpark implementations (common/speculative.cpp) all arm
        // cparams.embeddings_layer_inp[il] on ctx_tgt -- this same context -- via
        // llama_set_embeddings_layer_inp() at spec-init time, for exactly the
        // target_layer_ids they tap, and ONLY when a drafter is actually configured
        // (i.e. only when --spec-* wiring runs common_speculative_init(); the array
        // stays all-false for the lifetime of a context that never gets one). That is
        // the genuine "DSpark is live for this graph" signal; model.fc is not.
        if (!gtype_ok) {
            std::snprintf(ced_reason, sizeof(ced_reason), "side-graph (gtype=%d, not a plain prefill/decode build)", (int) params.gtype);
        } else if (!const_shape_off) {
            std::snprintf(ced_reason, sizeof(ced_reason), "WP_DS4_CONST_SHAPE-active (mutually exclusive with topology pinning)");
        } else if (!tokens_only) {
            std::snprintf(ced_reason, sizeof(ced_reason), "embd-input (VL/embd ubatch, not plain tokens)");
        } else if (!single_stream) {
            std::snprintf(ced_reason, sizeof(ced_reason), "multi-stream ubatch (n_seqs_unq=%u n_seqs=%u)", ubatch.n_seqs_unq, ubatch.n_seqs);
        } else if (!pos_flat) {
            std::snprintf(ced_reason, sizeof(ced_reason), "n_pos_per_embd=%d (multi-axis positions, trailing view unsafe)", hparams.n_pos_per_embd());
        } else if (!has_kv_sources) {
            std::snprintf(ced_reason, sizeof(ced_reason), "no-kv-source-layers (hparams.dsv41_n_kv_sources=0)");
        } else if (!multi_token) {
            std::snprintf(ced_reason, sizeof(ced_reason), "single-token ubatch (n_tokens=%lld, nothing to trim)", (long long) n_tokens);
        } else {
            for (uint32_t i = 0; i < hparams.dsv41_n_kv_sources; ++i) {
                ced_seam = std::max(ced_seam, (int) hparams.dsv41_kv_source_layer_ids[i]);
            }

            bool layers_ok = ced_seam >= 0 && ced_seam < (int) n_layer;
            if (!layers_ok) {
                std::snprintf(ced_reason, sizeof(ced_reason), "no-decoder-layers (ced_seam=%d n_layer=%d)", ced_seam, (int) n_layer);
            }

            for (int il = ced_seam; layers_ok && il < (int) n_layer; ++il) {
                // The ratio==0 "pure SWA, no compressed stream" path
                // (build_raw_attention, deepseek4.cpp) reads the raw mask/k_idxs
                // directly off inp_attn without going through any of the trailing-
                // view helpers above, and that function is shared with plain V4 --
                // rather than teach shared code about a V4.1-only trim, refuse it
                // here. No layer in the decoder range is expected to be ratio==0
                // in a trained config (every one of them reads ced_seam's
                // compressed stream), so this should never actually fire, but it
                // is cheap to check and load-bearing if that assumption is wrong.
                if (hparams.dsv4_compress_ratios[il] == 0) {
                    layers_ok = false;
                    std::snprintf(ced_reason, sizeof(ced_reason), "ratio==0 decoder layer (il=%d has no compressed stream)", il);
                }
                // build_inp_engram() sizes its graph input off the FULL n_tokens
                // (see build_inp_engram/build_engram below), not the trimmed
                // window; an engram layer inside the decoder range would silently
                // mismatch a trimmed inpL. Not observed in the current config
                // (engram layers are encoder-side), but checked directly rather
                // than assumed.
                if (static_cast<const llama_model_deepseek41 &>(model).engram_index(il) >= 0) {
                    layers_ok = false;
                    std::snprintf(ced_reason, sizeof(ced_reason), "engram layer in decoder range (il=%d)", il);
                }
            }

            // No decoder-range layer-input/embedding tap (distillation probes,
            // DSpark's tap-population path -- see build_dspark_stages and the
            // file-header comment) may be requested: a tap at or past the seam
            // would publish a W-row tensor where the caller expects n_tokens rows.
            // Checked through the post-loop tap too (il == n_layer).
            for (int il = ced_seam; layers_ok && il <= (int) n_layer; ++il) {
                if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
                    layers_ok = false;
                    std::snprintf(ced_reason, sizeof(ced_reason), "embeddings_layer_inp tap in decoder range (il=%d)", il);
                }
            }

            // WP_DSV41_CED_W (bisection knob): overrides W away from n_swa.
            // Safe to do unconditionally now that llama_kv_cache::
            // set_input_kq_mask's ced_replay_floor (see graph::graph() below)
            // masks out every cell earlier than the first replayed token
            // directly, regardless of W vs n_swa -- the old
            // "W must equal n_swa or decode-boundary rows read unwritten
            // cells" hazard this used to guard against is exactly the gap
            // that fix closed at the source, so a caller-chosen W no longer
            // depends on it for correctness.
            const int64_t w_override = dsv41_ced_w_override();
            const int64_t w = w_override >= 0 ? w_override : (int64_t) hparams.n_swa;

            if (layers_ok && !(w > 0 && n_tokens > w)) {
                std::snprintf(ced_reason, sizeof(ced_reason), "window-not-smaller-than-prompt (n_tokens=%lld <= W=%lld)", (long long) n_tokens, (long long) w);

                // Graph-side guard (WP_DSV41_CED_SKIP_NONFINAL hazard): this
                // ubatch falls back to the untrimmed decoder path below,
                // which assumes every raw-KV cell its attention window can
                // reach was already written by a prior ubatch. That is true
                // under today's (non-skip) trim -- every eligible ubatch,
                // final or not, writes its own trailing W raw-KV rows -- but
                // NOT if a preceding, non-final ubatch of this same prompt
                // had its decoder-range work skipped entirely (skip_nonfinal
                // on) and this one is shorter than W: part of its replay
                // window then falls on cells nothing ever wrote. n_outputs>0
                // (this is a final ubatch, otherwise nothing reads it) and
                // ubatch.pos[0]>0 (a prior ubatch of this sequence exists) are
                // the two facts observable from right here; there is no way
                // to recover the missing KV at this point (that would mean
                // redoing the previous ubatch), so this can only log loudly
                // and fall through to the same untrimmed path it would have
                // taken anyway -- the actual fix is at the source: the
                // server's prompt batching (WP_PREFILL_TAIL_MIN) must never
                // hand this graph a final ubatch shorter than W when
                // skip_nonfinal is in play. See the report for how the two
                // halves of this fix compose.
                if (dsv41_ced_skip_nonfinal_enabled() && n_outputs > 0 && ubatch.pos[0] > 0 &&
                        dsv41_ced_skipped_decoder_end == ubatch.pos[0]) {
                    char haz[256];
                    std::snprintf(haz, sizeof(haz),
                            "CED prefill trim: HAZARD final ubatch shorter than window (n_tokens=%lld < W=%lld, "
                            "pos0=%lld) with WP_DSV41_CED_SKIP_NONFINAL enabled -- a preceding ubatch may be "
                            "missing decoder KV; ensure WP_PREFILL_TAIL_MIN>=W at the server",
                            (long long) n_tokens, (long long) w, (long long) ubatch.pos[0]);
                    dsv41_ced_log_if_changed(haz, 3);
                }
            } else if (layers_ok && w > 0 && n_tokens > w && !dsv41_ced_outputs_in_window(ubatch, n_outputs, n_tokens, w)) {
                if ((int64_t) n_outputs == n_tokens) {
                    std::snprintf(ced_reason, sizeof(ced_reason),
                            "logits_all (all %lld rows requested; trim only computes the trailing W=%lld)",
                            (long long) n_tokens, (long long) w);
                } else {
                    // n_outputs==0 no longer refuses (round-3 multi-ubatch
                    // fix -- see dsv41_ced_outputs_in_window's comment), so
                    // reaching here with n_outputs==0 can't happen; this is
                    // the "some requested row is outside the trailing
                    // window" case (ubatch.output null, or a wanted row <
                    // n_tokens-W).
                    std::snprintf(ced_reason, sizeof(ced_reason),
                            "output row outside trailing window (a requested row is < n_tokens-W=%lld, or ubatch.output is unavailable)",
                            (long long) (n_tokens - w));
                }
            }

            ced_trim = layers_ok && w > 0 && n_tokens > w &&
                dsv41_ced_outputs_in_window(ubatch, n_outputs, n_tokens, w);

            if (ced_trim) {
                // Structural requirement only (see dsv41_ced_w_override's
                // comment): w >= 1 is already implied by `w > 0` above, and
                // `n_tokens > w` was just checked too -- restated here as a
                // belt-and-braces assert on the value actually committed,
                // since WP_DSV41_CED_W can set w to anything a caller likes.
                GGML_ASSERT(w >= 1 && n_tokens > w &&
                        "CED prefill trim: committed window width must be structurally valid (1 <= w < n_tokens)");
                ced_w      = w;
                ced_offset = n_tokens - w;
                std::snprintf(ced_reason, sizeof(ced_reason), "applied");

                // WP_DSV41_CED_SKIP_NONFINAL: n_outputs==0 means no row of
                // this ubatch is ever read (see dsv41_ced_outputs_in_window),
                // so a non-final ubatch can skip the decoder-range compute
                // entirely rather than just narrowing it to W rows -- see
                // graph::graph()'s ced_skip_layer below.
                ced_skip_nonfinal = dsv41_ced_skip_nonfinal_enabled() && n_outputs == 0;
            }
        }

        // Where the last ubatch that skipped its decoder ended, so the HAZARD
        // above fires only when this ubatch directly follows one (the server's
        // WP_PREFILL_TAIL_MIN marks the batch before a short tail as final).
        dsv41_ced_skipped_decoder_end = ced_skip_nonfinal ? (llama_pos) (ubatch.pos[0] + n_tokens) : -1;

        char ced_log[256];
        if (ced_trim) {
            std::snprintf(ced_log, sizeof(ced_log),
                    "CED prefill trim: applied seam_il=%d W=%lld scope=%s ubatch_n_tokens=%lld decoder_n_tokens=%lld skip_decoder=%s",
                    ced_seam, (long long) ced_w, dsv41_ced_scope_name(dsv41_ced_scope_override()),
                    (long long) n_tokens, (long long) ced_w, ced_skip_nonfinal ? "nonfinal" : "none");
        } else {
            std::snprintf(ced_log, sizeof(ced_log),
                    "CED prefill trim: refused reason=%s ubatch_n_tokens=%lld",
                    ced_reason, (long long) n_tokens);
        }
        dsv41_ced_log_if_changed(ced_log);
    }

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids(ced_trim ? ced_offset : 0);
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_kq_mask);

    // CED prefill trim: the raw-window mask restriction, applied at the
    // source in llama_kv_cache::set_input_kq_mask (llama-kv-cache.cpp) --
    // see llm_graph_input_dsv4_raw::ced_replay_floor's doc comment. SWA
    // Bounded Replay ("truncat[es] attention to that segment", per the DS4.1
    // paper) means every decoder-range query under the trim may attend only
    // to positions actually replayed through that layer; the mask builder
    // now enforces that directly from cell positions, in the one place they
    // are known, rather than via a graph-space reconstruction (which turned
    // out to be the actual crash: raw_write_idxs carries physical cache-slot
    // addresses, not mask-column indices, and the two spaces are not
    // interchangeable -- see the report). ced_offset indexes into THIS
    // ubatch, so ubatch.pos[ced_offset] is always the first replayed
    // token's own absolute position, never a cell from a prior ubatch.
    // -1 (unset, the default) when the trim is inactive for this ubatch --
    // set_input_kq_mask then behaves exactly as it did before this existed.
    if (ced_trim) {
        inp_dsv4->get_raw()->ced_replay_floor = ubatch.pos[ced_offset];

        // CED prefill trim: a dedicated, directly-filled I64 input tensor
        // for the raw-window write's trailing ced_w indices -- see
        // llm_graph_input_dsv4_raw::ced_k_idxs_trailing's doc comment. Built
        // once, graph-wide, exactly like the self_k_idxs leaf it replaces
        // for the decoder-range layers, since every decoder-range layer's
        // raw-window write targets the same ced_w physical cells.
        inp_dsv4->get_raw()->ced_k_idxs_trailing =
            inp_dsv4->get_raw()->mctx->build_input_k_idxs_trailing(ctx0, (uint32_t) ced_w);
        inp_dsv4->get_raw()->ced_k_idxs_trailing_offset = (uint32_t) ced_offset;

        // NaN fix (2026-09-22): dedicated, directly host-filled trailing-
        // width copies of the raw-window and compressed kq_mask inputs --
        // see llm_graph_input_dsv4_raw::ced_kq_mask_trailing's doc comment.
        // Shapes are derived from each already-built full-width leaf's own
        // ne[]/type (self_kq_mask / inp_csa.kq_mask / inp_hca.kq_mask were
        // all built earlier by build_inp_dsv4(), before this gate runs), not
        // re-derived from the comp plan -- ne[0] (n_kv) and ne[2]/ne[3]
        // (stream shape) are unaffected by the CED narrowing, only ne[1]
        // (the query-token axis) changes.
        {
            ggml_tensor * full = inp_dsv4->get_raw()->self_kq_mask;
            GGML_ASSERT(full != nullptr && "CED prefill trim: raw self_kq_mask must exist once the gate is eligible");
            ggml_tensor * trailing = ggml_new_tensor_4d(ctx0, full->type, full->ne[0], ced_w, full->ne[2], full->ne[3]);
            ggml_set_input(trailing);
            ggml_set_name(trailing, "attn_inp_kq_mask_ced_trailing");
            inp_dsv4->get_raw()->ced_kq_mask_trailing = trailing;
        }

        inp_dsv4->ced_mask_trailing_offset = (uint32_t) ced_offset;
        for (llm_graph_input_dsv4::comp_input * comp : { &inp_dsv4->inp_csa, &inp_dsv4->inp_hca }) {
            if (comp->kq_mask == nullptr) {
                continue;
            }
            ggml_tensor * full = comp->kq_mask;
            ggml_tensor * trailing = ggml_new_tensor_4d(ctx0, full->type, full->ne[0], ced_w, full->ne[2], full->ne[3]);
            ggml_set_input(trailing);
            ggml_set_name(trailing, "attn_inp_comp_kq_mask_ced_trailing");
            comp->ced_kq_mask_trailing = trailing;
        }
    }

    // CED prefill trim fix (round 2): a dedicated, directly host-filled I32
    // input tensor for the decoder-range layers' RoPE positions -- see
    // llm_graph_input_dsv4_raw::ced_pos_trailing's doc comment. Was
    // previously `ggml_view_1d(ctx0, inp_pos, ced_w, ced_offset * ...)`, a
    // GPU-side view of the graph-wide `inp_pos` input LEAF fed straight into
    // ggml_rope_ext/ggml_rope_ext_back at every decoder-range layer -- the
    // same "view of a host-resident input leaf feeding a GPU op" shape
    // already found (and fixed, for k_idxs and the mask) to produce wrong
    // data; this was the one instance of that shape the prior fix round
    // missed. hparams.n_pos_per_embd() == 1 is required by the gate above,
    // so a flat I32 leaf is the right shape (matches inp_pos's own layout in
    // that case).
    ggml_tensor * inp_pos_trim = nullptr;
    if (ced_trim) {
        inp_pos_trim = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ced_w);
        ggml_set_input(inp_pos_trim);
        ggml_set_name(inp_pos_trim, "attn_inp_pos_ced_trailing");
        inp_dsv4->get_raw()->ced_pos_trailing        = inp_pos_trim;
        inp_dsv4->get_raw()->ced_pos_trailing_offset = (uint32_t) ced_offset;
    }

    // WP_DSV41_CED_SCOPE (bisection knob): read once per graph build. Only
    // ever consulted below when ced_trim is true; dsv41_ced_scope_override()
    // itself is inert (never called) when the env var is off, matching every
    // other CED knob's discipline.
    const dsv41_ced_scope ced_scope = ced_trim ? dsv41_ced_scope_override() : dsv41_ced_scope::ALL;

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = identity_pre_mix();
    cb(pre_mix, "hc_pre_mix_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        // CED prefill trim (WP_DSV41_CED_SKIP_NONFINAL): on a non-final
        // ubatch (nothing reads this graph's output), every decoder-range
        // layer past the seam contributes nothing at all -- whatever it
        // would compute was already skipped at il==ced_seam below, so there
        // is nothing left to inherit. Layer-input/embedding taps and engram
        // layers cannot occur in the decoder range while ced_trim is active
        // (the gate above refuses the trim otherwise), so skipping them here
        // too is a no-op, not a behavior change.
        if (ced_skip_nonfinal && il > ced_seam) {
            continue;
        }

        if (ced_skip_nonfinal && il == ced_seam) {
            // Build only what the seam's kv-source/indexer publish needs --
            // the attn_norm'd, full-width hidden state -- and stop there.
            // No hc_mixes/hc_post/FFN/MoE for this layer, and (per the skip
            // above) none at all for the decoder layers after it. inpL/
            // pre_mix are left exactly as the encoder produced them; the
            // trunk loop's post-loop code and hc_collapse below run on that
            // stale-but-shape-valid stream, which is fine since n_outputs==0
            // is exactly the condition that guarantees nothing reads it.
            ggml_tensor * cur = build_hc_pre(inpL, pre_mix, il);
            cb(cur, "hc_attn_pre", il);

            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            build_attention_v41(model, inp_dsv4, cur, inp_pos, il,
                    /* cur_state = */ cur, /* ced_log_shapes = */ true, /* publish_only = */ true);
            continue;
        }

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

        ggml_tensor * cur_state    = nullptr;
        ggml_tensor * attn_inp_pos = inp_pos;

        // WP_DSV41_CED_SCOPE (bisection knob): which half of the seam layer's
        // work gets narrowed before it runs. Only meaningful at il==ced_seam --
        // by il==ced_seam+1, inpL is already whatever width the seam layer's
        // own FFN-side hc_post left it at (see ced_narrow_after_attn below),
        // and every later decoder-range layer just inherits that, regardless
        // of scope; there is no width left to choose independently for them.
        // This is a real structural coupling, not an oversight -- see the
        // report.
        const bool ced_narrow_before_attn = ced_trim && il == ced_seam && ced_scope != dsv41_ced_scope::FFN;
        const bool ced_narrow_after_attn  = ced_trim && il == ced_seam && ced_scope == dsv41_ced_scope::FFN;

        if (ced_narrow_before_attn) {
            // scope=all (default) or scope=attn: today's behaviour. cur_state
            // carries the full-width `cur` into build_attention_v41's
            // kv-source/indexer-publish block (every token, per the
            // architecture comment above); everything else -- this layer's
            // own query/window-KV/attention/MoE, and the whole of the decoder
            // layers after it -- only ever needs the trailing ced_w rows from
            // here on, so cur/residual/attn_pre/post/comb are all narrowed to
            // that window now (see dsv41_ced_trailing_2d/3d).
            //
            // scope=attn and scope=all are the SAME code path here: nothing
            // in this file independently narrows the FFN/MoE half at all --
            // its width has only ever been an inherited CONSEQUENCE of
            // attention's own narrowing (inpL flows straight from attention's
            // hc_post into the FFN's build_hc_mixes/build_hc_pre with no
            // separate narrowing step of its own). "Leave the FFN full width"
            // while attention narrows first is not achievable without
            // recomputing the dropped rows' hidden state some other way, so
            // scope=attn documents this coupling rather than papering over it
            // with a second, cosmetic narrowing call that would do nothing
            // different from scope=all.
            cur_state = cur;
            cur       = dsv41_ced_trailing_2d(ctx0, cur,      ced_w, ced_offset);
            residual  = dsv41_ced_trailing_3d(ctx0, residual, ced_w, ced_offset);
            attn_pre  = dsv41_ced_trailing_2d(ctx0, attn_pre, ced_w, ced_offset);
            post      = dsv41_ced_trailing_2d(ctx0, post,     ced_w, ced_offset);
            comb      = dsv41_ced_trailing_3d(ctx0, comb,     ced_w, ced_offset);
            attn_inp_pos = inp_pos_trim;
        } else if (ced_trim && il > ced_seam) {
            // inpL (hence residual/attn_pre/post/comb, all derived from it by
            // build_hc_mixes/build_hc_post above) is already ced_w rows here --
            // it became that width when layer ced_seam's hc_post ran (scope=all/
            // attn) or its FFN-side hc_post ran (scope=ffn, see
            // ced_narrow_after_attn below) -- either way, every layer past the
            // seam inherits a narrow stream regardless of scope. Only inp_pos
            // is a single graph-wide input untouched by that, so it is the one
            // thing still needing the trailing view every layer.
            attn_inp_pos = inp_pos_trim;
        }
        // scope=ffn at il==ced_seam: neither branch above runs. cur_state stays
        // null and attn_inp_pos stays the full inp_pos, so build_attention_v41
        // sees exactly what an encoder layer would -- nt == n_tokens, every
        // trailing-view helper it calls is a no-op, every scatter (raw-window
        // cpy_k, mask, k_idxs) runs at full width, identical to the off path
        // for this one layer.

        // CED prefill trim observability: log q/k_all/kq_mask/n_kv_max shapes
        // for the seam layer (where cur/etc. just got narrowed, or -- scope=ffn
        // -- is about to run at full width) and the first decoder layer past
        // it (where the narrowing has already propagated through inpL with no
        // explicit view left in this loop to point at) -- enough to see the
        // shapes at both ends of the seam without a line per layer for the
        // whole decoder range.
        const bool ced_log_shapes = ced_trim && (il == ced_seam || il == ced_seam + 1);

        cur = build_attention_v41(model, inp_dsv4, cur, attn_inp_pos, il, cur_state, ced_log_shapes);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        if (ced_narrow_after_attn) {
            // scope=ffn: attention just ran at full width (898 rows in, 898
            // rows out); narrow the stream HERE instead, right before the
            // FFN/MoE portion of this same layer. Everything computed FROM
            // this point on -- the FFN-side build_hc_mixes/build_hc_pre
            // below, residual, build_moe_ffn, build_ffn, this layer's own
            // second hc_post, and every subsequent decoder layer, which
            // inherits inpL as its own input -- derives its row count from
            // inpL's own shape, so one narrow of inpL here covers all of
            // that with no separate call needed.
            //
            // attn_pre is the one exception: it was computed at the TOP of
            // this same loop iteration (build_hc_mixes(inpL, hc_attn_fn, ...,
            // &attn_pre, ...), before either narrow point exists), at
            // whatever width inpL entered this layer with -- full 898 here,
            // since scope=ffn never narrows anything before attention. It is
            // then read by `cur = build_hc_pre(inpL, attn_pre, il)` below,
            // AFTER this narrow, against the now-narrow inpL. Left
            // unnarrowed, build_hc_pre's manual (non-fused) path would view
            // attn_pre's LEADING ced_w rows (dsv4_view_2d's extent comes from
            // x's nt, offset 0) instead of the TRAILING ones inpL's own view
            // actually represents -- wrong data, not an out-of-bounds read
            // (the view still fits inside attn_pre's real 898-row buffer),
            // but wrong regardless, and the fused path
            // (cparams.fused_dsv4_hc_pre) would instead hit a GGML_ASSERT
            // shape mismatch at graph-build time. Narrow it the same way here.
            inpL     = dsv41_ced_trailing_3d(ctx0, inpL,     ced_w, ced_offset);
            attn_pre = dsv41_ced_trailing_2d(ctx0, attn_pre, ced_w, ced_offset);
            cb(inpL, "hc_attn_post_ffn_scope_narrow", il);
        }

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
        // Row count comes from inpL itself, not the graph-level n_tokens: under
        // the CED prefill trim inpL only ever has ced_w rows by this point (see
        // the trunk loop above), and inp_out_ids' values were already rebased
        // by build_inp_out_ids(row_offset) to address that narrower stream --
        // using n_tokens here would reshape into the wrong row count and then
        // read out of bounds. Using inpL->ne[2] is exactly correct (and a
        // no-op change) whether or not the trim is active.
        const int64_t nt_cur = inpL->ne[2];
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, nt_cur);
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
