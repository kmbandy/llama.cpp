// pagedattn-dump: one-off diagnostic tool for SP2.5 turbo4_64 paged-attention
// cross-backend divergence investigation (LFM2.5-8B-A1B, Vulkan0 vs CUDA0).
//
// Runs a single short deterministic prompt through prefill (one llama_decode
// call) and, via cb_eval, intercepts every GGML_OP_PAGED_ATTN_MT tensor's
// OUTPUT (dst) right after it is computed, dumping its raw bytes (F16) to
// <dump-dir>/layer_<N>.bin where N increments once per invocation (0-based).
//
// Not part of the normal build; not intended to be committed permanently.
// Output directory can be overridden with the PAGEDATTN_DUMP_DIR env var
// (default: /tmp/pagedattn_dump).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

struct pagedattn_dump_ctx {
    std::string dump_dir;
    int         counter = 0;
};

static bool ensure_dir(const std::string & path) {
    // best-effort mkdir -p for a single-level (or already-existing) directory
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

static bool is_paged_qkv_input(const struct ggml_tensor * t) {
    return strncmp(t->name, "paged_q_cast_l", 14) == 0 ||
           strncmp(t->name, "paged_k_cast_l", 14) == 0 ||
           strncmp(t->name, "paged_v_cast_l", 14) == 0 ||
           strncmp(t->name, "model.layers.{}.conv.", 21) == 0;
}

static bool pagedattn_dump_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (pagedattn_dump_ctx *) user_data;

    const bool is_op_of_interest = (t->op == GGML_OP_PAGED_ATTN_MT) || is_paged_qkv_input(t);

    if (!is_op_of_interest) {
        return false; // don't request data for anything else
    }

    if (ask) {
        return true; // yes, make this tensor's data host-readable
    }

    const size_t n_bytes = ggml_nbytes(t);

    std::vector<uint8_t> host_buf(n_bytes);
    ggml_backend_tensor_get(t, host_buf.data(), 0, n_bytes);

    const std::string name = (t->op == GGML_OP_PAGED_ATTN_MT)
        ? ("layer_" + std::to_string(cb->counter))
        : std::string(t->name);
    const std::string path = cb->dump_dir + "/" + name + ".bin";

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, path.c_str());
        return true;
    }
    fwrite(host_buf.data(), 1, n_bytes, f);
    fclose(f);

    LOG_INF("%s: dumped %s -> %s  (op=%s, type=%s, ne=[%lld,%lld,%lld,%lld], nbytes=%zu)\n",
        __func__, name.c_str(), path.c_str(), ggml_op_name(t->op), ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], n_bytes);

    if (t->op == GGML_OP_PAGED_ATTN_MT) {
        cb->counter++;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    // sensible defaults for this diagnostic; still overridable on the CLI
    params.prompt = "The quick brown fox jumps over the lazy dog and runs through the forest.";
    params.n_ctx  = 512;

    // LLAMA_EXAMPLE_PERPLEXITY (not COMMON) so that --kv-tier-paged-blocks etc.
    // are recognized (that flag is restricted to SERVER/CLI/PERPLEXITY/BENCH).
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    pagedattn_dump_ctx cb_data;
    cb_data.dump_dir = std::getenv("PAGEDATTN_DUMP_DIR") ? std::getenv("PAGEDATTN_DUMP_DIR") : "/tmp/pagedattn_dump";

    if (!ensure_dir(cb_data.dump_dir)) {
        LOG_ERR("%s: could not create dump dir '%s'\n", __func__, cb_data.dump_dir.c_str());
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = pagedattn_dump_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("%s: dump dir = %s\n", __func__, cb_data.dump_dir.c_str());

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool           add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s: empty tokenized prompt\n", __func__);
        return 1;
    }

    LOG_INF("%s: prompt = \"%s\"\n", __func__, params.prompt.c_str());
    LOG_INF("%s: n_tokens = %zu\n", __func__, tokens.size());

    cb_data.counter = 0;

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    llama_synchronize(ctx);

    LOG_INF("%s: done. %d paged-attn tensors dumped to %s\n", __func__, cb_data.counter, cb_data.dump_dir.c_str());

    llama_backend_free();

    return 0;
}
