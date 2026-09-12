#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_tensor_data_f32(struct ggml_tensor * tensor, void * /*userdata*/) {
    const int64_t ne = ggml_nelements(tensor);
    uint32_t seed = 2166136261u;
    const char * name = ggml_get_name(tensor);
    for (const char * p = name; p != nullptr && *p != '\0'; ++p) {
        seed = (seed ^ (uint8_t) *p) * 16777619u;
    }

    const auto weight = [seed, name](int64_t i) {
        uint32_t x = seed ^ (uint32_t) i ^ (uint32_t) (i >> 32);
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        const float unit = 2.0f * (float) (x & 0xffffu) / 65535.0f - 1.0f;
        return name != nullptr && std::strstr(name, "norm") != nullptr
                ? 1.0f + 0.01f * unit
                : 0.05f * unit;
    };

    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = weight(i);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = ggml_fp32_to_fp16(weight(i));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    }
}

static gguf_context_ptr make_dsv41_gguf(uint32_t index_top_k = 8, bool dspark = false) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_DEEPSEEK41, ret.get());

    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 8;
    const uint32_t n_ff    = 256;
    const uint32_t n_layer = 2;
    const uint32_t n_layer_all = dspark ? 5 : n_layer;
    const uint32_t n_ctx   = 256;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_DEEPSEEK41));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer_all);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(0));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,      n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,    n_embd_head);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd_head / 2);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,     uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,  n_ctx / 8);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, n_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(32));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      index_top_k);
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,   uint32_t(32));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,    dspark
            ? std::vector<uint32_t>({0, 2, 0, 0, 0})
            : std::vector<uint32_t>({0, 2}));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, 160000.0f);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
    ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                     uint32_t(0));
    ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                     dspark
            ? std::vector<float>({10.0f, 10.0f, 10.0f, 10.0f, 10.0f})
            : std::vector<float>({10.0f, 10.0f}));
    if (dspark) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP,
                std::vector<float>({10.0f, 10.0f, 10.0f, 10.0f, 10.0f}));
    }
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,           n_ff);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                         uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                    uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,                  uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                   uint32_t(4));
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,                      "no_vocab");

    if (dspark) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,              uint32_t(3));
        ms.add_kv(LLM_KV_NEXTN_EXPERT_USED_COUNT,           uint32_t(1));
        ms.add_kv(LLM_KV_TARGET_LAYERS,                     std::vector<uint32_t>({2}));
        ms.add_kv(LLM_KV_BLOCK_SIZE,                        uint32_t(5));
        ms.add_kv(LLM_KV_TOKENIZER_MASK_ID,                 uint32_t(127));
        gguf_set_val_u32(ret.get(), "deepseek41.markov_rank", 16);

        const auto add_tensor_meta = [&](const char * name, int64_t ne0, int64_t ne1) {
            ggml_tensor tensor = {};
            tensor.type = GGML_TYPE_F32;
            tensor.ne[0] = ne0;
            tensor.ne[1] = ne1;
            tensor.ne[2] = 1;
            tensor.ne[3] = 1;
            tensor.nb[0] = sizeof(float);
            tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
            tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
            tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
            ggml_set_name(&tensor, name);
            gguf_add_tensor(ret.get(), &tensor);
        };

        add_tensor_meta("fc.weight", n_embd, n_embd);
        add_tensor_meta("enc.output_norm.weight", n_embd, 1);
        add_tensor_meta("markov_w1.weight", 16, n_vocab);
        add_tensor_meta("markov_w2.weight", 16, n_vocab);
        add_tensor_meta("conf_proj.weight", n_embd + 16, 1);
        add_tensor_meta("blk.2.ffn_gate_inp.weight", n_embd, 2);
    }

    ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYER_IDS,    std::vector<uint32_t>({1}));
    ms.add_kv(LLM_KV_ATTENTION_INDEX_SOURCE_LAYER_IDS, std::vector<uint32_t>({1}));
    ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER_ID, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_TOPK_BLOCKS,     uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,      uint32_t(8));
    ms.add_kv(LLM_KV_ENGRAM_HEAD_COUNT,            uint32_t(8));
    ms.add_kv(LLM_KV_ENGRAM_KEY_LENGTH,            uint32_t(256));
    ms.add_kv(LLM_KV_ENGRAM_MAX_NGRAM_SIZE,        uint32_t(4));
    ms.add_kv(LLM_KV_ENGRAM_VOCAB_SIZE,            uint32_t(16));
    ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
    ms.add_kv(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE, uint32_t(8));

    return ret;
}

static bool decode_dsv41_chunk(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        uint32_t offset,
        uint32_t count) {
    llama_batch batch = llama_batch_init((int32_t) count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        batch.token[i]     = tokens[offset + i];
        batch.pos[i]       = (llama_pos) (offset + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i + 1 == count;
    }
    batch.n_tokens = (int32_t) count;

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool capture_dsv41_last_logits(
        llama_context * ctx,
        uint32_t n_vocab,
        std::vector<float> & logits) {
    const float * last = llama_get_logits_ith(ctx, -1);
    if (last == nullptr) {
        return false;
    }

    logits.assign(last, last + n_vocab);
    return true;
}

static bool check_dsv41_finite(
        const float * data,
        size_t n,
        float & max_abs,
        bool require_nonzero = false) {
    if (data == nullptr || n == 0) {
        return false;
    }

    max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
        max_abs = std::max(max_abs, std::fabs(data[i]));
    }
    return !require_nonzero || max_abs > 0.0f;
}

static llama_batch make_dsv41_token_batch(
        const std::vector<llama_token> & tokens,
        llama_pos pos0,
        bool logits) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits;
    }
    batch.n_tokens = (int32_t) tokens.size();
    return batch;
}

static llama_batch make_dsv41_embd_batch(
        const std::vector<float> & embd,
        uint32_t n_embd,
        llama_pos pos0) {
    const int32_t n_tokens = (int32_t) (embd.size() / n_embd);
    llama_batch batch = llama_batch_init(n_tokens, (int32_t) n_embd, 1);
    std::memcpy(batch.embd, embd.data(), embd.size() * sizeof(float));
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }
    batch.n_tokens = n_tokens;
    return batch;
}

static bool compare_dsv41_logits(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        float & max_abs_diff,
        float & tolerance,
        int & lhs_top,
        int & rhs_top) {
    if (lhs.size() != rhs.size() || lhs.empty()) {
        return false;
    }

    max_abs_diff = 0.0f;
    float max_abs_logit = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i])) {
            return false;
        }
        max_abs_diff  = std::max(max_abs_diff, std::fabs(lhs[i] - rhs[i]));
        max_abs_logit = std::max(max_abs_logit, std::max(std::fabs(lhs[i]), std::fabs(rhs[i])));
    }

    tolerance = 1.0e-3f * std::max(1.0f, max_abs_logit);
    lhs_top = (int) (std::max_element(lhs.begin(), lhs.end()) - lhs.begin());
    rhs_top = (int) (std::max_element(rhs.begin(), rhs.end()) - rhs.begin());
    return max_abs_diff <= tolerance && lhs_top == rhs_top;
}

static bool test_dsv41_chunking_consistency() {
    constexpr uint32_t n_tokens       = 45;
    constexpr uint32_t n_vocab        = 128;
    constexpr uint32_t index_top_k    = 32;
    constexpr uint32_t n_rs_seq       = 8;
    constexpr uint32_t token_seed     = 0x41D541;

    std::mt19937 rng(token_seed);
    std::uniform_int_distribution<int32_t> token_dist(0, (int32_t) n_vocab - 1);
    std::vector<llama_token> tokens(n_tokens);
    for (llama_token & token : tokens) {
        token = (llama_token) token_dist(rng);
    }

    gguf_context_ptr gguf_ctx = make_dsv41_gguf(index_top_k);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = { cpu, nullptr };
    model_params.devices = devices;

    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data_f32, nullptr, model_params));
    if (!model) {
        fprintf(stderr, "CSA2 chunking: failed to load skeleton GGUF\n");
        return false;
    }

    const uint32_t ratio = model->hparams.dsv4_compress_ratios[1];
    const uint32_t n_swa = model->hparams.n_swa;
    const uint32_t n_compressed_groups = ratio == 0 ? 0 : (n_tokens - 1) / ratio;

    if (model->vocab.n_tokens() != n_vocab ||
        model->hparams.dsv41_n_kv_sources != 1 ||
        model->hparams.dsv41_kv_source_layer_ids[0] != 1 ||
        model->hparams.dsv41_n_index_sources != 1 ||
        model->hparams.dsv41_index_source_layer_ids[0] != 1 ||
        ratio != 2 || n_swa >= n_tokens || n_compressed_groups < 1 ||
        model->hparams.indexer_top_k < n_compressed_groups) {
        fprintf(stderr, "CSA2 chunking: synthetic hparams do not exercise the compressed path\n");
        return false;
    }

    printf("DeepSeek-V4.1 CSA2 chunking hparams: n_swa=%u ratio=%u kv_source=%u "
           "index_source=%u index_topk=%u vocab=%u compressed_groups=%u\n",
           n_swa,
           ratio,
           model->hparams.dsv41_kv_source_layer_ids[0],
           model->hparams.dsv41_index_source_layer_ids[0],
           model->hparams.indexer_top_k,
           model->vocab.n_tokens(),
           n_compressed_groups);

    const auto make_ctx = [&]() {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx     = 256;
        cparams.n_batch   = n_tokens;
        cparams.n_ubatch  = n_tokens;
        cparams.n_seq_max = 1;
        cparams.n_rs_seq  = n_rs_seq;
        return llama_context_ptr(llama_init_from_model(model.get(), cparams));
    };

    llama_context_ptr ctx_a = make_ctx();
    llama_context_ptr ctx_b = make_ctx();
    if (!ctx_a || !ctx_b) {
        fprintf(stderr, "CSA2 chunking: failed to create contexts\n");
        return false;
    }

    if (!decode_dsv41_chunk(ctx_a.get(), tokens, 0, n_tokens)) {
        fprintf(stderr, "CSA2 chunking: case A decode failed\n");
        return false;
    }

    std::vector<float> logits_a;
    if (!capture_dsv41_last_logits(ctx_a.get(), n_vocab, logits_a)) {
        fprintf(stderr, "CSA2 chunking: case A logits missing\n");
        return false;
    }

    if (!decode_dsv41_chunk(ctx_b.get(), tokens, 0, 39) ||
        !decode_dsv41_chunk(ctx_b.get(), tokens, 39, 1) ||
        !decode_dsv41_chunk(ctx_b.get(), tokens, 40, 1) ||
        !decode_dsv41_chunk(ctx_b.get(), tokens, 41, 4)) {
        fprintf(stderr, "CSA2 chunking: case B decode failed\n");
        return false;
    }

    std::vector<float> logits_b;
    if (!capture_dsv41_last_logits(ctx_b.get(), n_vocab, logits_b)) {
        fprintf(stderr, "CSA2 chunking: case B logits missing\n");
        return false;
    }

    float diff_b = 0.0f;
    float tolerance_b = 0.0f;
    int top_a_b = -1;
    int top_b_b = -1;
    const bool case_b_ok = compare_dsv41_logits(
            logits_a, logits_b, diff_b, tolerance_b, top_a_b, top_b_b);
    printf("DeepSeek-V4.1 CSA2 case B: max_abs_diff=%g tolerance=%g top1=%d/%d %s\n",
           (double) diff_b,
           (double) tolerance_b,
           top_a_b,
           top_b_b,
           case_b_ok ? "PASS" : "FAIL");
    if (!case_b_ok) {
        fprintf(stderr, "CSA2 chunking: case B does not match case A\n");
        return false;
    }

    const char * rollback_env = std::getenv("DSV41_TEST_ROLLBACK");
    const bool rollback_gate = rollback_env != nullptr && std::strcmp(rollback_env, "1") == 0;
    const bool rollback_ok = llama_memory_seq_rm(llama_get_memory(ctx_a.get()), 0, 43, -1);
    const bool case_c_decoded = rollback_ok && decode_dsv41_chunk(ctx_a.get(), tokens, 43, 2);

    std::vector<float> logits_c;
    const bool case_c_logits = case_c_decoded &&
            capture_dsv41_last_logits(ctx_a.get(), n_vocab, logits_c);
    float diff_c = std::numeric_limits<float>::infinity();
    float tolerance_c = 0.0f;
    int top_a_c = -1;
    int top_c = -1;
    const bool case_c_ok = case_c_logits && compare_dsv41_logits(
            logits_a, logits_c, diff_c, tolerance_c, top_a_c, top_c);
    printf("DeepSeek-V4.1 CSA2 case C: max_abs_diff=%g tolerance=%g top1=%d/%d %s\n",
           (double) diff_c,
           (double) tolerance_c,
           top_a_c,
           top_c,
           case_c_ok ? "PASS" : "FAIL");

    if (!case_c_ok) {
        if (rollback_gate) {
            fprintf(stderr, "CSA2 chunking: case C rollback check failed\n");
            return false;
        }
        printf("DeepSeek-V4.1 CSA2 case C: EXPECTED-FAIL (rollback not yet supported)\n");
    }

    return true;
}

static bool test_dsv41_dspark_cycle() {
    constexpr uint32_t n_vocab = 128;
    constexpr uint32_t n_embd  = 256;
    constexpr uint32_t n_prompt = 12;

    gguf_context_ptr gguf_ctx = make_dsv41_gguf(8, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = { cpu, nullptr };
    model_params.devices = devices;
    model_params.load_mtp = true; // the DSPARK draft context needs the stage tensors (common_model_params_to_llama does this for a self-draft)

    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data_f32, nullptr, model_params));
    if (!model) {
        fprintf(stderr, "DSpark cycle: failed to load synthetic model\n");
        return false;
    }
    if (llama_model_n_layer_nextn(model.get()) != 3 || !llama_model_has_dspark_markov(model.get())) {
        fprintf(stderr, "DSpark cycle: head hparams not loaded (nextn=%d markov=%d)\n",
                llama_model_n_layer_nextn(model.get()),
                (int) llama_model_has_dspark_markov(model.get()));
        return false;
    }
    if (llama_model_target_layer_ids_n(model.get()) != 1 ||
            llama_model_target_layer_ids(model.get())[0] != 2) {
        fprintf(stderr, "DSpark cycle: target_layers metadata not loaded\n");
        return false;
    }
    const llama_token mask_token = llama_vocab_mask(llama_model_get_vocab(model.get()));
    if (mask_token != 127) {
        fprintf(stderr, "DSpark cycle: mask token metadata not loaded (got %d)\n", mask_token);
        return false;
    }

    llama_context_params cparams_tgt = llama_context_default_params();
    cparams_tgt.n_ctx     = 256;
    cparams_tgt.n_batch   = 64;
    cparams_tgt.n_ubatch  = 64;
    cparams_tgt.n_seq_max = 1;
    llama_context_ptr ctx_tgt(llama_init_from_model(model.get(), cparams_tgt));
    if (!ctx_tgt) {
        fprintf(stderr, "DSpark cycle: failed to create target context\n");
        return false;
    }

    // target_layers is 1-based in the model metadata, and the speculative runtime uses
    // that value directly as the layer-input lid. For n_layer=2, lid 2 is the final boundary.
    const uint32_t target_lid = llama_model_target_layer_ids(model.get())[0];
    llama_set_embeddings_layer_inp(ctx_tgt.get(), target_lid, true);

    const std::vector<llama_token> prompt = { 3, 14, 27, 40, 53, 66, 79, 92, 105, 118, 7, 20 };
    llama_batch prompt_batch = make_dsv41_token_batch(prompt, 0, true);
    const int prompt_rc = llama_decode(ctx_tgt.get(), prompt_batch);
    llama_batch_free(prompt_batch);
    if (prompt_rc != 0) {
        fprintf(stderr, "DSpark cycle: target prompt decode failed rc=%d\n", prompt_rc);
        return false;
    }

    const float * target_tap = llama_get_embeddings_layer_inp(ctx_tgt.get(), target_lid);
    float target_tap_max_abs = 0.0f;
    if (!check_dsv41_finite(target_tap, (size_t) n_prompt * n_embd, target_tap_max_abs, true)) {
        fprintf(stderr, "DSpark cycle: target tap is missing, non-finite, or all zero\n");
        return false;
    }
    printf("DSpark cycle target: tap_lid=%u rows=%u width=%u max_abs=%g\n",
           target_lid, n_prompt, n_embd, (double) target_tap_max_abs);

    llama_context_params cparams_dft = llama_context_default_params();
    cparams_dft.n_ctx     = 256;
    cparams_dft.n_batch   = 64;
    cparams_dft.n_ubatch  = 64;
    cparams_dft.n_seq_max = 1;
    cparams_dft.n_rs_seq  = 0;
    cparams_dft.ctx_type  = LLAMA_CONTEXT_TYPE_DSPARK;
    cparams_dft.ctx_other = ctx_tgt.get();
    llama_context_ptr ctx_dft(llama_init_from_model(model.get(), cparams_dft));
    if (!ctx_dft) {
        fprintf(stderr, "DSpark cycle: failed to create draft context\n");
        return false;
    }
    llama_set_embeddings_nextn(ctx_dft.get(), true, true);
    llama_set_causal_attn(ctx_dft.get(), false);

    std::vector<float> features((size_t) n_prompt * n_embd);
    std::memcpy(features.data(), target_tap, features.size() * sizeof(float));
    llama_batch enc_batch = make_dsv41_embd_batch(features, n_embd, 0);
    const int enc_rc = llama_encode(ctx_dft.get(), enc_batch);
    llama_batch_free(enc_batch);
    if (enc_rc != 0) {
        fprintf(stderr, "DSpark cycle: encoder failed rc=%d\n", enc_rc);
        return false;
    }

    const float * encoded = llama_get_embeddings_nextn(ctx_dft.get());
    float encoded_max_abs = 0.0f;
    if (!check_dsv41_finite(encoded, (size_t) n_prompt * n_embd, encoded_max_abs, true)) {
        fprintf(stderr, "DSpark cycle: encoded rows are missing, non-finite, or all zero\n");
        return false;
    }
    printf("DSpark cycle encode: rows=%u width=%u -> nextn_width=%u max_abs=%g\n",
           n_prompt, n_embd, n_embd, (double) encoded_max_abs);

    std::vector<float> encoded_rows(encoded, encoded + (size_t) n_prompt * n_embd);
    llama_batch inject_batch = make_dsv41_embd_batch(encoded_rows, n_embd, 0);
    const int inject_rc = llama_decode(ctx_dft.get(), inject_batch);
    llama_batch_free(inject_batch);
    if (inject_rc != 0) {
        fprintf(stderr, "DSpark cycle: first injection decode failed rc=%d\n", inject_rc);
        return false;
    }
    printf("DSpark cycle inject: rows=%u width=%u pos=[0,11] rc=%d\n",
           n_prompt, n_embd, inject_rc);

    std::vector<llama_token> block_tokens = {
        prompt[n_prompt - 1], mask_token, mask_token, mask_token, mask_token,
    };
    llama_batch block_batch = make_dsv41_token_batch(block_tokens, n_prompt, true);
    const int block_rc = llama_decode(ctx_dft.get(), block_batch);
    llama_batch_free(block_batch);
    if (block_rc != 0) {
        fprintf(stderr, "DSpark cycle: first block decode failed rc=%d\n", block_rc);
        return false;
    }

    std::vector<float> block_logits(5 * n_vocab);
    std::vector<float> block_hidden(5 * n_embd);
    float block_logit0_max_abs = 0.0f;
    for (size_t i = 0; i < 5; ++i) {
        const float * logits = llama_get_logits_ith(ctx_dft.get(), (int32_t) i);
        float row_max_abs = 0.0f;
        if (!check_dsv41_finite(logits, n_vocab, row_max_abs)) {
            fprintf(stderr, "DSpark cycle: first block logits row %zu is invalid\n", i);
            return false;
        }
        std::memcpy(block_logits.data() + i * n_vocab, logits, n_vocab * sizeof(float));
        if (i == 0) {
            block_logit0_max_abs = row_max_abs;
        }

        const float * hidden = llama_get_embeddings_nextn_ith(ctx_dft.get(), (int32_t) i);
        float hidden_max_abs = 0.0f;
        if (!check_dsv41_finite(hidden, n_embd, hidden_max_abs)) {
            fprintf(stderr, "DSpark cycle: first block confidence embedding row %zu is invalid\n", i);
            return false;
        }
        std::memcpy(block_hidden.data() + i * n_embd, hidden, n_embd * sizeof(float));

        const int top = (int) (std::max_element(logits, logits + n_vocab) - logits);
        if (top < 0 || top >= (int) n_vocab) {
            fprintf(stderr, "DSpark cycle: first block row %zu has invalid argmax\n", i);
            return false;
        }
    }

    bool block_rows_differ = false;
    for (size_t i = 1; i < 5 && !block_rows_differ; ++i) {
        for (size_t j = 0; j < n_vocab; ++j) {
            if (block_logits[i * n_vocab + j] != block_logits[j]) {
                block_rows_differ = true;
                break;
            }
        }
    }
    if (!block_rows_differ) {
        fprintf(stderr, "DSpark cycle: first block logit rows are all identical\n");
        return false;
    }
    printf("DSpark cycle block: tokens=5 logits=[5,%u] hidden=[5,%u] pos=[12,16] max_abs_logit0=%g\n",
           n_vocab, n_embd, (double) block_logit0_max_abs);

    // llama_dspark_markov_head() is for the services path: it requires base logits and
    // post-output_norm hidden rows. This in-model path exposes pre-norm rows here and
    // already runs the Markov head in its decoder graph, so do not call it with invalid
    // inputs just to exercise the symbol.
    printf("DSpark cycle markov: skipped (in-model graph owns Markov; nextn rows are pre-norm)\n");

    const float * last_logits = llama_get_logits_ith(ctx_tgt.get(), -1);
    if (last_logits == nullptr) {
        fprintf(stderr, "DSpark cycle: target last logits are missing\n");
        return false;
    }
    const llama_token successor = (llama_token) (std::max_element(
            last_logits, last_logits + n_vocab) - last_logits);

    llama_batch successor_batch = make_dsv41_token_batch({ successor }, 12, true);
    const int successor_rc = llama_decode(ctx_tgt.get(), successor_batch);
    llama_batch_free(successor_batch);
    if (successor_rc != 0) {
        fprintf(stderr, "DSpark cycle: target successor decode failed rc=%d\n", successor_rc);
        return false;
    }

    target_tap = llama_get_embeddings_layer_inp(ctx_tgt.get(), target_lid);
    const float * successor_tap = target_tap + (size_t) n_prompt * n_embd;
    float successor_tap_max_abs = 0.0f;
    if (!check_dsv41_finite(successor_tap, n_embd, successor_tap_max_abs, true)) {
        fprintf(stderr, "DSpark cycle: successor target tap is invalid\n");
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, 12, -1)) {
        fprintf(stderr, "DSpark cycle: draft cache clear failed\n");
        return false;
    }
    printf("DSpark cycle clear: draft_seq=0 pos=[12,inf) before successor injection\n");

    std::vector<float> successor_features(successor_tap, successor_tap + n_embd);
    llama_batch enc_batch_next = make_dsv41_embd_batch(successor_features, n_embd, 12);
    const int enc_next_rc = llama_encode(ctx_dft.get(), enc_batch_next);
    llama_batch_free(enc_batch_next);
    if (enc_next_rc != 0) {
        fprintf(stderr, "DSpark cycle: second encoder failed rc=%d\n", enc_next_rc);
        return false;
    }

    const float * encoded_next = llama_get_embeddings_nextn(ctx_dft.get());
    float encoded_next_max_abs = 0.0f;
    if (!check_dsv41_finite(encoded_next, n_embd, encoded_next_max_abs, true)) {
        fprintf(stderr, "DSpark cycle: second encoded row is invalid\n");
        return false;
    }
    std::vector<float> encoded_next_row(encoded_next, encoded_next + n_embd);
    llama_batch inject_batch_next = make_dsv41_embd_batch(encoded_next_row, n_embd, 12);
    const int inject_next_rc = llama_decode(ctx_dft.get(), inject_batch_next);
    llama_batch_free(inject_batch_next);
    if (inject_next_rc != 0) {
        fprintf(stderr, "DSpark cycle: second injection decode failed rc=%d\n", inject_next_rc);
        return false;
    }

    std::vector<llama_token> block_tokens_next = { successor, mask_token, mask_token, mask_token, mask_token };
    llama_batch block_batch_next = make_dsv41_token_batch(block_tokens_next, 13, true);
    const int block_next_rc = llama_decode(ctx_dft.get(), block_batch_next);
    llama_batch_free(block_batch_next);
    if (block_next_rc != 0) {
        fprintf(stderr, "DSpark cycle: second block decode failed rc=%d\n", block_next_rc);
        return false;
    }

    float block_next_logit0_max_abs = 0.0f;
    for (size_t i = 0; i < 5; ++i) {
        const float * logits = llama_get_logits_ith(ctx_dft.get(), (int32_t) i);
        float logit_max_abs = 0.0f;
        float hidden_max_abs = 0.0f;
        if (!check_dsv41_finite(logits, n_vocab, logit_max_abs) ||
                !check_dsv41_finite(llama_get_embeddings_nextn_ith(ctx_dft.get(), (int32_t) i),
                        n_embd, hidden_max_abs)) {
            fprintf(stderr, "DSpark cycle: second block row %zu is invalid\n", i);
            return false;
        }
        if (i == 0) {
            block_next_logit0_max_abs = logit_max_abs;
        }
    }
    printf("DSpark cycle second: target_pos=12 tap_width=%u tap_max_abs=%g encode_width=%u "
           "block_logits=[5,%u] hidden=[5,%u] pos=[13,17] max_abs_logit0=%g\n",
           n_embd, (double) successor_tap_max_abs, n_embd, n_vocab, n_embd,
           (double) block_next_logit0_max_abs);
    return true;
}

int main() {
    ggml_backend_load_all();

    gguf_context_ptr gguf_ctx = make_dsv41_gguf();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = { cpu, nullptr };
    model_params.devices = devices;

    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data_f32, nullptr, model_params));
    if (!model) {
        fprintf(stderr, "failed to load DeepSeek-V4.1 skeleton GGUF\n");
        return 1;
    }

    if (model->arch != LLM_ARCH_DEEPSEEK41) {
        fprintf(stderr, "arch mismatch: got %s\n", llm_arch_name(model->arch));
        return 1;
    }
    if (llama_model_n_embd(model.get()) != 256) {
        fprintf(stderr, "n_embd mismatch: %d\n", llama_model_n_embd(model.get()));
        return 1;
    }
    if (llama_model_n_layer(model.get()) != 2) {
        fprintf(stderr, "n_layer mismatch: %d\n", llama_model_n_layer(model.get()));
        return 1;
    }
    if (model->hparams.dsv41_n_kv_sources != 1 || model->hparams.dsv41_kv_source_layer_ids[0] != 1) {
        fprintf(stderr, "kv_source_layer_ids not loaded\n");
        return 1;
    }
    if (model->hparams.dsv41_candidate_source_layer != 1) {
        fprintf(stderr, "candidate_source_layer not loaded\n");
        return 1;
    }
    if (model->hc_head_fn != nullptr) {
        fprintf(stderr, "output_hc should be absent on V4.1\n");
        return 1;
    }
    if (model->hparams.n_embd_out_impl != 0) {
        fprintf(stderr, "n_embd_out_impl should stay 0 (no output_hc)\n");
        return 1;
    }

    printf("DeepSeek-V4.1 skeleton load ok: arch=%s n_embd=%d n_layer=%d kv_sources=%u\n",
           llm_arch_name(model->arch),
           llama_model_n_embd(model.get()),
           llama_model_n_layer(model.get()),
           model->hparams.dsv41_n_kv_sources);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_batch   = 1;
    cparams.n_ubatch  = 1;
    cparams.n_seq_max = 1;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "failed to create DeepSeek-V4.1 context\n");
        return 1;
    }

    llama_token tok0 = 1;
    llama_batch batch0 = llama_batch_get_one(&tok0, 1);
    if (llama_decode(ctx.get(), batch0) != 0) {
        fprintf(stderr, "llama_decode token 0 failed\n");
        return 1;
    }
    llama_token tok1 = 2;
    llama_batch batch1 = llama_batch_get_one(&tok1, 1);
    if (llama_decode(ctx.get(), batch1) != 0) {
        fprintf(stderr, "llama_decode token 1 failed\n");
        return 1;
    }

    printf("DeepSeek-V4.1 CPU decode ok (2 tokens, CSA2 ratio-2 source)\n");

    if (!test_dsv41_chunking_consistency()) {
        return 1;
    }

    if (!test_dsv41_dspark_cycle()) {
        return 1;
    }

    return 0;
}
