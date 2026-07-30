#include "ggml.h"
#include "gguf.h"
#include "weight-pager/wp-router.h"
#include "wp-dense-extract-lib.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                          \
        }                                                                        \
    } while (0)

namespace {

namespace fs = std::filesystem;

float payload_value(const std::string & name, int64_t i) {
    uint32_t hash = 0;
    for (char c : name) {
        hash = hash * 31u + static_cast<uint32_t>(c);
    }
    return static_cast<float>(static_cast<int>(hash % 1000) - 500) + static_cast<float>(i);
}

void zeros(std::ostream & output, size_t n) {
    static const char zero[32] = {};
    while (n > 0) {
        const size_t chunk = std::min(n, sizeof(zero));
        output.write(zero, static_cast<std::streamsize>(chunk));
        n -= chunk;
    }
}

void write_shard(const std::string &              path,
                 int                              split_no,
                 int                              split_count,
                 int                              tensor_count,
                 bool                             model_metadata,
                 const std::vector<std::string> & names) {
    gguf_context *   gguf   = gguf_init_empty();
    ggml_init_params params = {
        /*.mem_size   =*/names.size() * ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * tensors = ggml_init(params);
    if (gguf == nullptr || tensors == nullptr) {
        throw std::runtime_error("failed to create synthetic GGUF");
    }

    if (model_metadata) {
        gguf_set_val_str(gguf, "general.architecture", "glm-dsa");
        gguf_set_val_str(gguf, "general.name", "wp-dense-extract-test");
        gguf_set_val_u32(gguf, "glm-dsa.block_count", 3);
        const int32_t ints[] = { 7, -3, 99 };
        gguf_set_arr_data(gguf, "test.int_array", GGUF_TYPE_INT32, ints, 3);
        const char * strings[] = { "alpha", "beta" };
        gguf_set_arr_str(gguf, "test.string_array", strings, 2);
    }
    gguf_set_val_u16(gguf, "split.no", static_cast<uint16_t>(split_no));
    gguf_set_val_u16(gguf, "split.count", static_cast<uint16_t>(split_count));
    gguf_set_val_i32(gguf, "split.tensors.count", tensor_count);

    for (const std::string & name : names) {
        ggml_tensor * tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 4, 3);
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(gguf, tensor);
    }

    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    std::vector<uint8_t> metadata(gguf_get_meta_size(gguf));
    gguf_get_meta_data(gguf, metadata.data());
    output.write(reinterpret_cast<const char *>(metadata.data()), static_cast<std::streamsize>(metadata.size()));
    for (const std::string & name : names) {
        const ggml_tensor * tensor = ggml_get_tensor(tensors, name.c_str());
        std::vector<float>  data(ggml_nelements(tensor));
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            data[i] = payload_value(name, i);
        }
        output.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(ggml_nbytes(tensor)));
        zeros(output, GGML_PAD(ggml_nbytes(tensor), GGUF_DEFAULT_ALIGNMENT) - ggml_nbytes(tensor));
    }
    output.close();
    ggml_free(tensors);
    gguf_free(gguf);
}

std::set<std::string> read_and_check_output(const std::string & path) {
    ggml_context *   tensors = nullptr;
    gguf_init_params params  = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&tensors,
    };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (gguf == nullptr || tensors == nullptr) {
        throw std::runtime_error("failed to read extracted GGUF");
    }

    std::set<std::string> names;
    std::ifstream         input(path, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        names.insert(name);
        CHECK(!wp::is_routed_expert_name(name));

        const ggml_tensor * tensor = ggml_get_tensor(tensors, name);
        std::vector<float>  data(ggml_nelements(tensor));
        const size_t        offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
        input.seekg(static_cast<std::streamoff>(offset));
        input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(ggml_nbytes(tensor)));
        for (int64_t j = 0; j < ggml_nelements(tensor); ++j) {
            CHECK(data[j] == payload_value(name, j));
        }
    }

    const int64_t marker = gguf_find_key(gguf, wp_dense_extract::ROUTED_EXPERTS_EXTERNAL_KEY);
    CHECK(marker >= 0);
    CHECK(gguf_get_kv_type(gguf, marker) == GGUF_TYPE_BOOL);
    CHECK(gguf_get_val_bool(gguf, marker));
    CHECK(gguf_find_key(gguf, "split.no") < 0);
    CHECK(gguf_find_key(gguf, "split.count") < 0);
    CHECK(gguf_find_key(gguf, "split.tensors.count") < 0);
    CHECK(std::strcmp(gguf_get_val_str(gguf, gguf_find_key(gguf, "general.name")), "wp-dense-extract-test") == 0);

    const int64_t int_array = gguf_find_key(gguf, "test.int_array");
    CHECK(int_array >= 0);
    CHECK(gguf_get_arr_type(gguf, int_array) == GGUF_TYPE_INT32);
    CHECK(gguf_get_arr_n(gguf, int_array) == 3);
    const int32_t * ints = static_cast<const int32_t *>(gguf_get_arr_data(gguf, int_array));
    CHECK(ints[0] == 7 && ints[1] == -3 && ints[2] == 99);

    const int64_t string_array = gguf_find_key(gguf, "test.string_array");
    CHECK(string_array >= 0);
    CHECK(gguf_get_arr_n(gguf, string_array) == 2);
    CHECK(std::strcmp(gguf_get_arr_str(gguf, string_array, 0), "alpha") == 0);
    CHECK(std::strcmp(gguf_get_arr_str(gguf, string_array, 1), "beta") == 0);

    gguf_free(gguf);
    ggml_free(tensors);
    return names;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "wp-dense-extract-test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const std::string first  = (dir / "model-00001-of-00002.gguf").string();
    const std::string second = (dir / "model-00002-of-00002.gguf").string();
    const std::string output = (dir / "dense.gguf").string();

    const std::vector<std::string> first_names = {
        "token_embd.weight",          "blk.0.attn_q.weight",       "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_exp_probs_b.bias", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
    };
    const std::vector<std::string> second_names = {
        "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_down_exps.weight",  "blk.2.nextn_eh_proj.weight",
        "blk.2.ffn_gate_inp.weight",   "blk.2.ffn_up_exps.weight",
        "output_norm.weight",          "output.weight",
    };
    const int tensor_count = static_cast<int>(first_names.size() + second_names.size());
    write_shard(first, 0, 2, tensor_count, true, first_names);
    write_shard(second, 1, 2, tensor_count, false, second_names);

    const wp_dense_extract::result result = wp_dense_extract::extract(first, output, true);
    CHECK(result.verified);
    CHECK(result.tensor_count == 11);
    CHECK(result.routed_tensor_count == 4);
    CHECK(result.tensor_bytes == 11 * 4 * 3 * sizeof(float));
    CHECK(result.routed_tensor_bytes == 4 * 4 * 3 * sizeof(float));
    CHECK(result.file_bytes == fs::file_size(output));

    const std::set<std::string> expected = {
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_exp_probs_b.bias",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.2.nextn_eh_proj.weight",
        "blk.2.ffn_gate_inp.weight",
        "output_norm.weight",
        "output.weight",
    };
    const std::set<std::string> names = read_and_check_output(output);
    CHECK(names == expected);

    for (const std::string & name : first_names) {
        CHECK(names.count(name) == static_cast<size_t>(!wp::is_routed_expert_name(name.c_str())));
    }
    for (const std::string & name : second_names) {
        CHECK(names.count(name) == static_cast<size_t>(!wp::is_routed_expert_name(name.c_str())));
    }
    CHECK(names.count("blk.0.ffn_up_shexp.weight") == 1);
    CHECK(names.count("blk.0.ffn_gate_shexp.weight") == 1);
    CHECK(names.count("blk.0.ffn_down_shexp.weight") == 1);

    fs::remove_all(dir);
    if (g_failed == 0) {
        std::printf("OK: all wp-dense-extract tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d check(s) failed\n", g_failed);
    return 1;
}
