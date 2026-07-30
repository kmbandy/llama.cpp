#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama-arch.h"
#include "llama-model-saver.h"
#include "llama-model.h"
#include "llama.h"
#include "weight-pager/wp-router.h"
#include "wp-dense-extract-lib.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char * EXTERNAL_KEY = "weight_pager.routed_experts_external";

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        const fs::path base = fs::temp_directory_path();
        for (unsigned i = 0; i < 1000; ++i) {
            path_ = base / ("routed-experts-external-" + std::to_string(std::rand()) + "-" + std::to_string(i));
            std::error_code ec;
            if (fs::create_directory(path_, ec)) {
                return;
            }
        }
        throw std::runtime_error("failed to create temporary directory");
    }

    ~temporary_directory() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    const fs::path & path() const { return path_; }

  private:
    fs::path path_;
};

bool set_environment(const char * name, const char * value) {
#if defined(_WIN32)
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

void clear_environment(const char * name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class environment_override {
  public:
    environment_override(const char * name, const char * value) : name_(name) {
        if (const char * current = std::getenv(name)) {
            old_value_ = current;
            had_old_value_ = true;
        }
        if (!set_environment(name, value)) {
            throw std::runtime_error("failed to set environment variable");
        }
    }

    ~environment_override() {
        if (had_old_value_) {
            set_environment(name_.c_str(), old_value_.c_str());
        } else {
            clear_environment(name_.c_str());
        }
    }

  private:
    std::string name_;
    std::string old_value_;
    bool had_old_value_ = false;
};

class log_capture {
  public:
    log_capture() {
        llama_log_get(&old_callback_, &old_user_data_);
        llama_log_set([](ggml_log_level, const char * text, void * user_data) {
            static_cast<log_capture *>(user_data)->text_ += text;
        }, this);
    }

    ~log_capture() {
        llama_log_set(old_callback_, old_user_data_);
    }

    const std::string & text() const { return text_; }

  private:
    ggml_log_callback old_callback_ = nullptr;
    void * old_user_data_ = nullptr;
    std::string text_;
};

gguf_context_ptr make_glm_dsa_metadata() {
    gguf_context_ptr gguf(gguf_init_empty());
    require(gguf != nullptr, "failed to create GGUF metadata");

    llama_model_saver saver(LLM_ARCH_GLM_DSA, gguf.get());
    saver.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      "glm-dsa");
    saver.add_kv(LLM_KV_GENERAL_NAME,              "routed-experts-external-test");
    saver.add_kv(LLM_KV_VOCAB_SIZE,                uint32_t(32));
    saver.add_kv(LLM_KV_CONTEXT_LENGTH,            uint32_t(32));
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH,          uint32_t(32));
    saver.add_kv(LLM_KV_BLOCK_COUNT,               uint32_t(2));
    saver.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       uint32_t(48));

    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        uint32_t(1));
    saver.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     uint32_t(1));
    saver.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,        uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,      uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    saver.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,       uint32_t(16));
    saver.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,      uint32_t(8));
    saver.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    saver.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(8));
    saver.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));

    saver.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,  uint32_t(8));
    saver.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({2, 2, 2, 2}));
    saver.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");

    saver.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(48));
    saver.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
    saver.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
    saver.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
    saver.add_kv(LLM_KV_EXPERT_GROUP_COUNT,         uint32_t(1));
    saver.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT,    uint32_t(1));
    saver.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2));
    saver.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
    saver.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);

    return gguf;
}

void set_tensor_data(ggml_tensor * tensor, void *) {
    std::vector<unsigned char> zeros(ggml_nbytes(tensor), 0);
    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
}

llama_model_params model_params() {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    params.progress_callback = [](float, void *) { return true; };
    return params;
}

bool has_tensor(const llama_model * model, const char * name) {
    for (const auto & entry : llama_internal_get_tensor_map(model)) {
        if (entry.first == name) {
            return true;
        }
    }
    return false;
}

size_t routed_tensor_count(const llama_model * model) {
    size_t result = 0;
    for (const auto & entry : llama_internal_get_tensor_map(model)) {
        if (wp::is_routed_expert_name(entry.first.c_str())) {
            ++result;
        }
    }
    return result;
}

void require_router_and_shared_experts(const llama_model * model) {
    require(has_tensor(model, "blk.1.ffn_gate_inp.weight"), "router tensor is missing");
    require(has_tensor(model, "blk.1.exp_probs_b.bias"), "router bias is missing");
    require(has_tensor(model, "blk.1.ffn_gate_shexp.weight"), "shared gate tensor is missing");
    require(has_tensor(model, "blk.1.ffn_up_shexp.weight"), "shared up tensor is missing");
    require(has_tensor(model, "blk.1.ffn_down_shexp.weight"), "shared down tensor is missing");
}

void run_tests() {
    temporary_directory temporary;
    const fs::path complete_path = temporary.path() / "complete.gguf";
    const fs::path external_path = temporary.path() / "external.gguf";

    gguf_context_ptr metadata = make_glm_dsa_metadata();
    model_ptr generated(
        llama_model_init_from_user(metadata.get(), set_tensor_data, nullptr, model_params()),
        llama_model_free);
    require(generated != nullptr, "failed to generate complete synthetic model");
    const size_t generated_routed = routed_tensor_count(generated.get());
    if (generated_routed != 9) {
        throw std::runtime_error(
            "complete synthetic model created " + std::to_string(generated_routed) +
            " routed-expert tensors instead of 9");
    }
    require_router_and_shared_experts(generated.get());
    llama_model_save_to_file(generated.get(), complete_path.c_str());
    generated.reset();

    model_ptr complete(llama_model_load_from_file(complete_path.c_str(), model_params()), llama_model_free);
    require(complete != nullptr, "complete MoE model failed to reload");
    require(!complete->routed_experts_external, "complete model was marked external");
    require(routed_tensor_count(complete.get()) == 9, "complete MoE model lost routed experts");
    require_router_and_shared_experts(complete.get());
    complete.reset();

    const wp_dense_extract::result extraction =
        wp_dense_extract::extract(complete_path.string(), external_path.string(), true);
    require(extraction.routed_tensor_count == 9, "extractor removed the wrong routed tensor count");

    {
        llama_model_kv_override overrides[2] = {};
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        std::snprintf(overrides[0].key, sizeof(overrides[0].key), "%s", EXTERNAL_KEY);
        overrides[0].val_bool = false;

        llama_model_params params = model_params();
        params.kv_overrides = overrides;
        log_capture logs;
        model_ptr unmarked(llama_model_load_from_file(external_path.c_str(), params), llama_model_free);
        require(unmarked == nullptr, "unmarked missing routed experts were accepted");
        require(logs.text().find("missing tensor 'blk.1.ffn_gate_exps.weight'") != std::string::npos,
                "unmarked missing routed experts did not report the missing tensor");
    }

    environment_override resident_dense("WP_RESIDENT_DENSE", "1");
    llama_model_params external_params = model_params();
    external_params.weight_paging_enabled = true;
    external_params.pipeline_layer_first = 0;
    external_params.pipeline_layer_last = 1;

    model_ptr external(
        llama_model_load_from_file(external_path.c_str(), external_params),
        llama_model_free);
    require(external != nullptr, "external-expert model failed to load");
    require(external->routed_experts_external, "external metadata signal was not recorded");
    require(routed_tensor_count(external.get()) == 0, "external model created routed-expert tensors");
    require_router_and_shared_experts(external.get());
    require(external->weight_pager != nullptr, "weight pager carrier was not created");
    require(external->weight_pager->weight_tensor_ptrs.empty(), "empty expert catalog collected tensor pointers");

    {
        llama_context_params params = llama_context_default_params();
        log_capture logs;
        llama_context * context = llama_init_from_model(external.get(), params);
        require(context == nullptr, "external experts were accepted without --expert-dispatch");
        require(logs.text().find(EXTERNAL_KEY) != std::string::npos,
                "missing-dispatch error did not name the metadata key");
        require(logs.text().find("--expert-dispatch") != std::string::npos,
                "missing-dispatch error did not name --expert-dispatch");
    }
}

}  // namespace

int main() {
    llama_backend_init();
    try {
        run_tests();
        llama_backend_free();
        std::puts("test-routed-experts-external: all tests passed");
        return 0;
    } catch (const std::exception & error) {
        llama_backend_free();
        std::fprintf(stderr, "test-routed-experts-external: %s\n", error.what());
        return 1;
    }
}
