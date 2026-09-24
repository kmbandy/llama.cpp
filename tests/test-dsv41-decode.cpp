// DeepSeek-V4.1 multi-token CPU decode test.
//
// test-dsv41-load.cpp decodes exactly one/two tokens at a time and never
// builds a batch bigger than the ubatch size, so it cannot see:
//   - shape bugs that only appear when n_tokens > 1 in a single llama_decode
//   - state bugs from lookback across ubatch boundaries (Engram n-gram
//     history, SWA window eviction, CSA2 ratio>=2 pooling)
//
// This file exercises the PUBLIC llama API only (llama_model_init_from_user
// is part of that public surface for building synthetic models; the GGUF
// fixture writer below duplicates the technique in test-dsv41-load.cpp and
// necessarily uses the same private headers that writer does -- it is not
// part of the surface under test). It does not touch src/ or
// test-dsv41-load.cpp.
//
// Expected RED today: the Engram n-gram history is graph-local, so any
// lookback into a previous ubatch reads the pad token instead of real
// history. Check 2 (ubatch invariance) is expected to fail until the port
// lands.

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Fixture: a small but shape-realistic DeepSeek-V4.1 skeleton GGUF.
// ---------------------------------------------------------------------------

namespace dsv41_decode_fixture {

// n_embd_head = 64, hc_mult = 4, engram_dim = 16, n_hash_cols = 6 (below) --
// keep every tensor tiny but every architectural path (CSA2 ratio-1 and
// ratio-2 layers, a KV/index source + reader, Engram, SWA) exercised.
//
// n_embd_head is decoupled from n_embd / n_head (DeepSeek's MLA-style
// attention already does this: Q/K/V go through LoRA-projected head dims,
// not n_embd/n_head) and pinned at 64: the DeepSeek lightning-indexer
// Hadamard rotation (forced on whenever n_embd_head_k_full ==
// indexer_head_size, which this fixture sets up on purpose) only
// precomputes rotation matrices for sizes >= 64
// (llama_kv_cache::llama_kv_cache's `for (n = 64; n <= ...; n *= 2)` loop
// in src/llama-kv-cache.cpp), so a smaller head dim aborts in
// llama_kv_cache::set_input_k_rot. See test-dsv41-load.cpp's make_dsv41_gguf
// for the same fixture constraint.
constexpr uint32_t n_vocab     = 128;
constexpr uint32_t n_embd      = 64;
constexpr uint32_t n_head      = 4;
constexpr uint32_t n_embd_head = 64;
constexpr uint32_t n_ff        = 64;
constexpr uint32_t n_layer     = 6;
constexpr uint32_t n_ctx       = 512;
constexpr uint32_t n_swa       = 128;
constexpr uint32_t hc_mult     = 4;

// compress_ratios mirror the real V4.1-Flash file: 0 = raw window attention,
// then a ratio-2 compressed block whose FIRST layer is the stream's KV/index
// source, then ratio-1 layers that share a later source. Real file:
// [0,0,2,...,2,1,...] with kv/index sources at the block starts (2, 8, 14, 20).
const std::vector<uint32_t> compress_ratios = { 0, 0, 2, 2, 1, 1 };

// Sources: layer 2 (ratio-2 block: readers 2,3) and layer 4 (ratio-1 block:
// readers 4,5). Layer 3 and 5 alias their source's storage.
const std::vector<uint32_t> kv_source_layers    = { 2, 4 };
const std::vector<uint32_t> index_source_layers = { 2, 4 };
constexpr uint32_t candidate_source_layer = 2;

// Engram lives on layer 4 (a reader layer, well past prefill start) with a
// tiny table so the ~200-token history it needs to look back through stays
// cheap while still being non-trivial (max_ngram_size 4 means 3-gram lookback).
constexpr uint32_t engram_layer      = 4;
constexpr uint32_t engram_n_heads    = 2;
constexpr uint32_t engram_head_dim   = 16;
constexpr uint32_t engram_max_ngram  = 4;
constexpr uint32_t engram_vocab_size = 16;
constexpr uint32_t engram_pad_token  = 2;
constexpr uint32_t engram_compressed_vocab_size = 8;

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

static void add_tensor_meta(struct gguf_context * ctx, const char * name, int64_t ne0, int64_t ne1) {
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
    gguf_add_tensor(ctx, &tensor);
}

static gguf_context_ptr make_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_DEEPSEEK41, ret.get());

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_DEEPSEEK41));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(0));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,      n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,    n_embd_head);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd_head / 2);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,     uint32_t(32));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,  n_swa);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, n_head);
    // matches n_embd_head so n_embd_head_k_full == indexer_head_size (see n_embd_head comment above)
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(32));
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,   uint32_t(16));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,    compress_ratios);
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, 160000.0f);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               hc_mult);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
    ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                     uint32_t(0));
    ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,
            std::vector<float>(n_layer, 10.0f));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,           n_ff);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                         uint32_t(4));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                    uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,                  uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                   uint32_t(4)); // SQRT_SOFTPLUS
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                 1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                  true);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,                      "no_vocab");

    ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYER_IDS,    kv_source_layers);
    ms.add_kv(LLM_KV_ATTENTION_INDEX_SOURCE_LAYER_IDS, index_source_layers);
    ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_SOURCE_LAYER_ID, candidate_source_layer);
    ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_TOPK_BLOCKS,     uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_CANDIDATE_BLOCK_SIZE,      uint32_t(8));

    ms.add_kv(LLM_KV_ENGRAM_LAYER_IDS,             std::vector<uint32_t>({ engram_layer }));
    ms.add_kv(LLM_KV_ENGRAM_HEAD_COUNT,            engram_n_heads);
    ms.add_kv(LLM_KV_ENGRAM_KEY_LENGTH,            engram_head_dim);
    ms.add_kv(LLM_KV_ENGRAM_MAX_NGRAM_SIZE,        engram_max_ngram);
    ms.add_kv(LLM_KV_ENGRAM_VOCAB_SIZE,            engram_vocab_size);
    ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          engram_pad_token);
    ms.add_kv(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE, engram_compressed_vocab_size);

    // Engram's embd/wkv tensors are the only ones the loader requires by
    // name lookup before create_tensor() runs (see load_arch_tensors() /
    // build_engram() in src/models/deepseek41.cpp: engram_embd and
    // engram_wkv both being non-null is what turns the Engram path on).
    // engram_k/engram_q are left absent (optional, both-or-neither).
    const uint32_t ngram_kinds  = engram_max_ngram > 1 ? engram_max_ngram - 1 : 1;
    const uint32_t n_hash_cols  = ngram_kinds * engram_n_heads; // (4-1)*2 = 6

    // load_arch_hparams() derives per-(ngram-kind, head) prime buckets from
    // engram.vocab_size unconditionally (it does not require hash_multipliers
    // to be present -- see the "if (hparams.dsv41_n_engram_layers > 0)" block
    // in src/models/deepseek41.cpp), and build_engram()'s ggml_get_rows()
    // indexes engram_embd with ids up to sum(primes). Replicate that same
    // search here so the fixture's row count is exactly right instead of a
    // guess (an under-sized table reads out of bounds at runtime).
    const auto is_prime = [](int64_t n) {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (int64_t d = 3; d * d <= n; d += 2) {
            if (n % d == 0) return false;
        }
        return true;
    };
    uint32_t engram_rows = 0;
    {
        std::set<int64_t> seen;
        int64_t offset = 0;
        for (uint32_t ng = 0; ng < ngram_kinds; ++ng) {
            int64_t current = (int64_t) engram_vocab_size - 1;
            for (uint32_t h = 0; h < engram_n_heads; ++h) {
                int64_t candidate = current + 1;
                while (!is_prime(candidate) || seen.count(candidate)) {
                    ++candidate;
                }
                seen.insert(candidate);
                current = candidate;
                offset += candidate;
            }
        }
        engram_rows = (uint32_t) offset; // exact row count for one engram layer's table
    }
    const std::string embd_name = "blk." + std::to_string(engram_layer) + ".engram_embd.weight";
    const std::string wkv_name  = "blk." + std::to_string(engram_layer) + ".engram_wkv.weight";
    add_tensor_meta(ret.get(), embd_name.c_str(), engram_head_dim, engram_rows);
    add_tensor_meta(ret.get(), wkv_name.c_str(), n_hash_cols * engram_head_dim, n_embd * (hc_mult + 1));

    return ret;
}

static llama_model_ptr load_model() {
    gguf_context_ptr gguf_ctx = make_gguf();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devices[] = { cpu, nullptr };
    model_params.devices = devices;

    return llama_model_ptr(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data_f32, nullptr, model_params));
}

} // namespace dsv41_decode_fixture

// ---------------------------------------------------------------------------
// Test helpers (public API only from here down).
// ---------------------------------------------------------------------------

namespace {

using dsv41_decode_fixture::n_vocab;
using dsv41_decode_fixture::n_ctx;
using dsv41_decode_fixture::n_swa;

llama_context_ptr make_context(llama_model * model, uint32_t n_batch, uint32_t n_ubatch) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = n_batch;
    cparams.n_ubatch  = n_ubatch;
    cparams.n_seq_max = 1;
    cparams.n_threads       = 8;
    cparams.n_threads_batch = 8;
    // The CPU flash-attn op picks a tiled or a per-row kernel by query-row
    // count, so its rounding depends on how a prompt was split into ubatches.
    // That is a property of our CPU fork's kernels, not of this model: run the
    // non-fused attention path so the checks below are bit-exact on the graph.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_context_ptr(llama_init_from_model(model, cparams));
}

std::vector<llama_token> make_tokens(uint32_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> token_dist(0, (int32_t) n_vocab - 1);
    std::vector<llama_token> tokens(count);
    for (llama_token & tok : tokens) {
        tok = (llama_token) token_dist(rng);
    }
    return tokens;
}

bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i + 1 == tokens.size());
    }
    batch.n_tokens = (int32_t) tokens.size();
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// Position is tracked automatically by llama_decode() for a get_one() batch,
// continuing from wherever this sequence's decode left off.
bool decode_one(llama_context * ctx, llama_token token) {
    llama_batch batch = llama_batch_get_one(&token, 1);
    return llama_decode(ctx, batch) == 0;
}

bool capture_last_logits(llama_context * ctx, std::vector<float> & out) {
    const float * last = llama_get_logits_ith(ctx, -1);
    if (last == nullptr) {
        return false;
    }
    out.assign(last, last + n_vocab);
    return true;
}

bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

llama_token argmax(const std::vector<float> & logits) {
    return (llama_token) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

} // namespace

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

// Check 1: single-decode shape. Prefill N=200 tokens in ONE llama_decode
// (n_batch == n_ubatch == 256, so no internal ubatch splitting happens);
// every logit must be finite.
static bool check_shape(llama_model * model, const std::vector<llama_token> & tokens200, std::vector<float> & out_logits) {
    llama_context_ptr ctx = make_context(model, 256, 256);
    if (!ctx) {
        fprintf(stderr, "check1 (shape): failed to create context\n");
        return false;
    }
    if (!decode_tokens(ctx.get(), tokens200, 0)) {
        fprintf(stderr, "check1 (shape): decode failed\n");
        return false;
    }
    if (!capture_last_logits(ctx.get(), out_logits)) {
        fprintf(stderr, "check1 (shape): logits missing\n");
        return false;
    }
    if (!all_finite(out_logits)) {
        fprintf(stderr, "check1 (shape): FAIL non-finite logit found among %u\n", n_vocab);
        return false;
    }
    printf("check1 (shape): PASS n_tokens=%zu all %u logits finite\n", tokens200.size(), n_vocab);
    return true;
}

// Check 2: ubatch invariance. Same 200 tokens, fresh context, n_ubatch=32
// (7 ubatches). Final-position logits must equal check 1 within 1e-4
// max-abs. This is the one that catches cross-ubatch lookback bugs (Engram
// history is graph-local today, so it reads the pad token whenever the
// lookback crosses a ubatch boundary).
static bool check_ubatch_invariance(
        llama_model * model,
        const std::vector<llama_token> & tokens200,
        const std::vector<float> & reference_logits) {
    llama_context_ptr ctx = make_context(model, 256, 32);
    if (!ctx) {
        fprintf(stderr, "check2 (ubatch invariance): failed to create context\n");
        return false;
    }
    if (!decode_tokens(ctx.get(), tokens200, 0)) {
        fprintf(stderr, "check2 (ubatch invariance): decode failed\n");
        return false;
    }
    std::vector<float> logits;
    if (!capture_last_logits(ctx.get(), logits)) {
        fprintf(stderr, "check2 (ubatch invariance): logits missing\n");
        return false;
    }
    if (!all_finite(logits)) {
        fprintf(stderr, "check2 (ubatch invariance): FAIL non-finite logit\n");
        return false;
    }
    const float diff = max_abs_diff(reference_logits, logits);
    const bool ok = diff <= 1.0e-4f;
    printf("check2 (ubatch invariance): n_ubatch=32 (7 ubatches) max_abs_diff=%g tol=1e-4 %s\n",
            (double) diff, ok ? "PASS" : "FAIL");
    if (!ok) {
        fprintf(stderr,
                "check2 (ubatch invariance): FAIL -- final-position logits diverge across ubatch "
                "boundaries (expected: Engram history is graph-local, so lookback into a previous "
                "ubatch reads the pad token instead of real history)\n");
    }
    return ok;
}

// Check 3: decode continuity. Prefill 200 tokens with n_ubatch=32 (same
// ubatch-split shape as check 2, but its own fresh context -- independent of
// check 2's outcome so all four checks always run), then decode 4 more
// tokens one at a time (positions 200..203, argmax feed-forward); compare
// against a fresh context that prefills all 204 tokens at once: logits at
// position 203 within 1e-4. SWA window 128 and CSA2 ratio-2 pooling must be
// crossed by these positions.
static bool check_decode_continuity(llama_model * model, const std::vector<llama_token> & tokens200) {
    if (!(tokens200.size() > n_swa)) {
        fprintf(stderr, "check3 (decode continuity): fixture error -- N (%zu) must exceed the SWA window (%u)\n",
                tokens200.size(), n_swa);
        return false;
    }

    llama_context_ptr ctx_running = make_context(model, 256, 32);
    if (!ctx_running) {
        fprintf(stderr, "check3 (decode continuity): failed to create running context\n");
        return false;
    }
    if (!decode_tokens(ctx_running.get(), tokens200, 0)) {
        fprintf(stderr, "check3 (decode continuity): running prefill decode failed\n");
        return false;
    }

    std::vector<llama_token> tail_tokens;
    std::vector<float> last_logits;
    if (!capture_last_logits(ctx_running.get(), last_logits)) {
        fprintf(stderr, "check3 (decode continuity): missing logits after prefill\n");
        return false;
    }

    llama_token next = argmax(last_logits);
    for (int step = 0; step < 4; ++step) {
        tail_tokens.push_back(next);
        const llama_pos pos = 200 + step;
        if (!decode_one(ctx_running.get(), next)) {
            fprintf(stderr, "check3 (decode continuity): running decode failed at pos=%d\n", pos);
            return false;
        }
        if (!capture_last_logits(ctx_running.get(), last_logits)) {
            fprintf(stderr, "check3 (decode continuity): logits missing at pos=%d\n", pos);
            return false;
        }
        next = argmax(last_logits);
    }
    const std::vector<float> running_logits_203 = last_logits;

    std::vector<llama_token> tokens204 = tokens200;
    tokens204.insert(tokens204.end(), tail_tokens.begin(), tail_tokens.end());
    if (tokens204.size() != 204) {
        fprintf(stderr, "check3 (decode continuity): internal error building 204-token sequence\n");
        return false;
    }

    llama_context_ptr ctx_fresh = make_context(model, 256, 256);
    if (!ctx_fresh) {
        fprintf(stderr, "check3 (decode continuity): failed to create fresh context\n");
        return false;
    }
    if (!decode_tokens(ctx_fresh.get(), tokens204, 0)) {
        fprintf(stderr, "check3 (decode continuity): fresh full-prefill decode failed\n");
        return false;
    }
    std::vector<float> fresh_logits_203;
    if (!capture_last_logits(ctx_fresh.get(), fresh_logits_203)) {
        fprintf(stderr, "check3 (decode continuity): fresh-context logits missing\n");
        return false;
    }

    if (!all_finite(running_logits_203) || !all_finite(fresh_logits_203)) {
        fprintf(stderr, "check3 (decode continuity): FAIL non-finite logit\n");
        return false;
    }

    const float diff = max_abs_diff(running_logits_203, fresh_logits_203);
    const bool ok = diff <= 1.0e-4f;
    printf("check3 (decode continuity): pos=203 (SWA=%u crossed) max_abs_diff=%g tol=1e-4 %s\n",
            n_swa, (double) diff, ok ? "PASS" : "FAIL");
    return ok;
}

// Check 4: memory clear + repeat. llama_memory_clear, redo check 1,
// identical logits (bitwise).
static bool check_memory_clear_repeat(
        llama_model * model,
        const std::vector<llama_token> & tokens200,
        const std::vector<float> & reference_logits) {
    llama_context_ptr ctx = make_context(model, 256, 256);
    if (!ctx) {
        fprintf(stderr, "check4 (memory clear): failed to create context\n");
        return false;
    }
    if (!decode_tokens(ctx.get(), tokens200, 0)) {
        fprintf(stderr, "check4 (memory clear): first decode failed\n");
        return false;
    }
    std::vector<float> first_logits;
    if (!capture_last_logits(ctx.get(), first_logits)) {
        fprintf(stderr, "check4 (memory clear): first logits missing\n");
        return false;
    }

    llama_memory_clear(llama_get_memory(ctx.get()), /*data*/ true);

    if (!decode_tokens(ctx.get(), tokens200, 0)) {
        fprintf(stderr, "check4 (memory clear): second decode failed\n");
        return false;
    }
    std::vector<float> second_logits;
    if (!capture_last_logits(ctx.get(), second_logits)) {
        fprintf(stderr, "check4 (memory clear): second logits missing\n");
        return false;
    }

    const bool bitwise_ok = first_logits.size() == second_logits.size() &&
            std::memcmp(first_logits.data(), second_logits.data(), first_logits.size() * sizeof(float)) == 0;
    const float diff_vs_check1 = max_abs_diff(reference_logits, first_logits);
    printf("check4 (memory clear): bitwise_repeat=%s diff_vs_check1=%g\n",
            bitwise_ok ? "PASS" : "FAIL", (double) diff_vs_check1);
    if (!bitwise_ok) {
        fprintf(stderr, "check4 (memory clear): FAIL logits after clear+redecode are not bit-identical\n");
        return false;
    }
    return true;
}

int main() {
    ggml_backend_load_all();

    llama_model_ptr model = dsv41_decode_fixture::load_model();
    if (!model) {
        fprintf(stderr, "failed to load synthetic DeepSeek-V4.1 GGUF\n");
        return 1;
    }
    printf("DeepSeek-V4.1 decode fixture loaded: n_embd=%d n_vocab=%u n_layer=%u compress_ratios=[0,0,2,2,1,1] "
            "kv_sources=[2,4] index_sources=[2,4] engram_layer=%u n_swa=%u\n",
            llama_model_n_embd(model.get()),
            dsv41_decode_fixture::n_vocab,
            dsv41_decode_fixture::n_layer,
            dsv41_decode_fixture::engram_layer,
            dsv41_decode_fixture::n_swa);

    const std::vector<llama_token> tokens200 = make_tokens(200, 0x4157D1u);

    bool all_ok = true;

    std::vector<float> logits_check1;
    const bool ok1 = check_shape(model.get(), tokens200, logits_check1);
    all_ok = all_ok && ok1;

    bool ok2 = false;
    if (ok1) {
        ok2 = check_ubatch_invariance(model.get(), tokens200, logits_check1);
        all_ok = all_ok && ok2;
    } else {
        fprintf(stderr, "check2 (ubatch invariance): SKIPPED (check1 did not produce a reference)\n");
        all_ok = false;
    }

    // Independent of check1/check2: check3 builds its own context and its
    // own 200-token ubatch-split prefill, so it always runs.
    const bool ok3 = check_decode_continuity(model.get(), tokens200);
    all_ok = all_ok && ok3;

    bool ok4 = false;
    if (ok1) {
        ok4 = check_memory_clear_repeat(model.get(), tokens200, logits_check1);
        all_ok = all_ok && ok4;
    } else {
        fprintf(stderr, "check4 (memory clear): SKIPPED (check1 did not produce a reference)\n");
        all_ok = false;
    }

    printf("DeepSeek-V4.1 decode summary: check1=%s check2=%s check3=%s check4=%s\n",
            ok1 ? "PASS" : "FAIL",
            ok2 ? "PASS" : "FAIL",
            ok3 ? "PASS" : "FAIL",
            ok4 ? "PASS" : "FAIL");

    return all_ok ? 0 : 1;
}
