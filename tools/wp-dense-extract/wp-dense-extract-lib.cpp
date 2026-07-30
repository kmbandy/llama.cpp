#include "wp-dense-extract-lib.h"

#include "ggml.h"
#include "gguf.h"
#include "weight-pager/wp-router.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wp_dense_extract {

namespace {

constexpr size_t COPY_BUFFER_SIZE = 16u * 1024u * 1024u;

namespace fs = std::filesystem;

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

struct input_shard {
    std::string path;
    ggml_ptr    tensors;
    gguf_ptr    gguf;
};

struct tensor_source {
    size_t  shard_idx;
    int64_t tensor_idx;
};

class temporary_output {
  public:
    explicit temporary_output(std::string path) : path_(std::move(path)) {}

    ~temporary_output() {
        if (!released_) {
            std::remove(path_.c_str());
        }
    }

    const std::string & path() const { return path_; }

    void release() { released_ = true; }

  private:
    std::string path_;
    bool        released_ = false;
};

input_shard load_shard(const std::string & path) {
    ggml_context *         raw_tensors = nullptr;
    const gguf_init_params params      = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&raw_tensors,
    };
    gguf_context * raw_gguf = gguf_init_from_file(path.c_str(), params);
    ggml_ptr       tensors(raw_tensors, ggml_free);
    gguf_ptr       gguf(raw_gguf, gguf_free);
    if (gguf == nullptr || tensors == nullptr) {
        throw std::runtime_error("failed to load GGUF metadata from " + path);
    }
    return { path, std::move(tensors), std::move(gguf) };
}

uint16_t optional_u16(const gguf_context * ctx, const char * key, uint16_t fallback) {
    const int64_t key_id = gguf_find_key(ctx, key);
    return key_id < 0 ? fallback : gguf_get_val_u16(ctx, key_id);
}

std::vector<input_shard> load_shards(const std::string & input) {
    const std::string first = fs::canonical(fs::path(input)).string();

    std::vector<input_shard> shards;
    shards.push_back(load_shard(first));

    const uint16_t split_count = optional_u16(shards.front().gguf.get(), "split.count", 1);
    if (split_count <= 1) {
        return shards;
    }

    const uint16_t split_no = optional_u16(shards.front().gguf.get(), "split.no", UINT16_MAX);
    if (split_no != 0) {
        throw std::runtime_error("split GGUF input must be the first shard: " + first);
    }

    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), "-%05u-of-%05u.gguf", static_cast<unsigned int>(split_no + 1),
                  static_cast<unsigned int>(split_count));
    const std::string expected_suffix(suffix);
    if (first.size() <= expected_suffix.size() ||
        first.compare(first.size() - expected_suffix.size(), expected_suffix.size(), expected_suffix) != 0) {
        throw std::runtime_error("invalid split GGUF file name: " + first);
    }
    const std::string prefix = first.substr(0, first.size() - expected_suffix.size());

    shards.reserve(split_count);
    for (uint16_t i = 1; i < split_count; ++i) {
        char split_suffix[64];
        std::snprintf(split_suffix, sizeof(split_suffix), "-%05u-of-%05u.gguf", static_cast<unsigned int>(i + 1),
                      static_cast<unsigned int>(split_count));
        const std::string path  = fs::canonical(fs::path(prefix + split_suffix)).string();
        input_shard       shard = load_shard(path);
        if (optional_u16(shard.gguf.get(), "split.no", UINT16_MAX) != i ||
            optional_u16(shard.gguf.get(), "split.count", UINT16_MAX) != split_count) {
            throw std::runtime_error("inconsistent split metadata: " + path);
        }
        shards.push_back(std::move(shard));
    }
    return shards;
}

void zeros(std::ostream & file, size_t n) {
    static const char zeros_buf[4096] = {};
    while (n > 0) {
        const size_t chunk = std::min(n, sizeof(zeros_buf));
        file.write(zeros_buf, chunk);
        n -= chunk;
    }
}

bool is_split_key(const char * key) {
    return std::strcmp(key, "split.no") == 0 || std::strcmp(key, "split.count") == 0 ||
           std::strcmp(key, "split.tensors.count") == 0;
}

size_t value_type_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
            break;
    }
    throw std::runtime_error("invalid numeric GGUF metadata type");
}

bool kv_values_equal(const gguf_context * lhs, int64_t lhs_id, const gguf_context * rhs, int64_t rhs_id) {
    const gguf_type type = gguf_get_kv_type(lhs, lhs_id);
    if (type != gguf_get_kv_type(rhs, rhs_id)) {
        return false;
    }
    if (type == GGUF_TYPE_STRING) {
        return std::strcmp(gguf_get_val_str(lhs, lhs_id), gguf_get_val_str(rhs, rhs_id)) == 0;
    }
    if (type != GGUF_TYPE_ARRAY) {
        return std::memcmp(gguf_get_val_data(lhs, lhs_id), gguf_get_val_data(rhs, rhs_id), value_type_size(type)) == 0;
    }

    const gguf_type arr_type = gguf_get_arr_type(lhs, lhs_id);
    const size_t    arr_n    = gguf_get_arr_n(lhs, lhs_id);
    if (arr_type != gguf_get_arr_type(rhs, rhs_id) || arr_n != gguf_get_arr_n(rhs, rhs_id)) {
        return false;
    }
    if (arr_type == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < arr_n; ++i) {
            if (std::strcmp(gguf_get_arr_str(lhs, lhs_id, i), gguf_get_arr_str(rhs, rhs_id, i)) != 0) {
                return false;
            }
        }
        return true;
    }
    return std::memcmp(gguf_get_arr_data(lhs, lhs_id), gguf_get_arr_data(rhs, rhs_id),
                       arr_n * value_type_size(arr_type)) == 0;
}

void verify_metadata(const gguf_context * source, const gguf_context * output) {
    if (gguf_get_version(source) != gguf_get_version(output)) {
        throw std::runtime_error("verification failed: GGUF version differs");
    }
    if (gguf_get_alignment(source) != gguf_get_alignment(output)) {
        throw std::runtime_error("verification failed: GGUF alignment differs");
    }

    int64_t expected_kv = 1;
    for (int64_t i = 0; i < gguf_get_n_kv(source); ++i) {
        const char * key = gguf_get_key(source, i);
        if (is_split_key(key)) {
            continue;
        }
        ++expected_kv;
        const int64_t out_id = gguf_find_key(output, key);
        if (out_id < 0) {
            throw std::runtime_error("verification failed: output metadata is missing " + std::string(key));
        }
        if (!kv_values_equal(source, i, output, out_id)) {
            throw std::runtime_error("verification failed: metadata differs for " + std::string(key));
        }
    }
    if (gguf_get_n_kv(output) != expected_kv) {
        throw std::runtime_error("verification failed: output has unexpected metadata keys");
    }

    const int64_t marker_id = gguf_find_key(output, ROUTED_EXPERTS_EXTERNAL_KEY);
    if (marker_id < 0 || gguf_get_kv_type(output, marker_id) != GGUF_TYPE_BOOL ||
        !gguf_get_val_bool(output, marker_id)) {
        throw std::runtime_error("verification failed: output does not declare external routed experts");
    }
    if (gguf_find_key(output, "split.no") >= 0 || gguf_find_key(output, "split.count") >= 0 ||
        gguf_find_key(output, "split.tensors.count") >= 0) {
        throw std::runtime_error("verification failed: split metadata leaked into single-file output");
    }
}

void verify_tensor_shape(const ggml_tensor * source, const ggml_tensor * output, const char * name) {
    if (source->type != output->type || ggml_n_dims(source) != ggml_n_dims(output) ||
        ggml_nbytes(source) != ggml_nbytes(output)) {
        throw std::runtime_error("verification failed: tensor metadata differs for " + std::string(name));
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source->ne[i] != output->ne[i]) {
            throw std::runtime_error("verification failed: tensor shape differs for " + std::string(name));
        }
    }
}

void compare_tensor_bytes(std::ifstream &     source,
                          uint64_t            source_offset,
                          std::ifstream &     output,
                          uint64_t            output_offset,
                          uint64_t            n_bytes,
                          const char *        name,
                          std::vector<char> & source_buf,
                          std::vector<char> & output_buf) {
    source.seekg(static_cast<std::streamoff>(source_offset));
    output.seekg(static_cast<std::streamoff>(output_offset));

    uint64_t remaining = n_bytes;
    uint64_t checked   = 0;
    while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, source_buf.size()));
        source.read(source_buf.data(), static_cast<std::streamsize>(chunk));
        output.read(output_buf.data(), static_cast<std::streamsize>(chunk));
        if (std::memcmp(source_buf.data(), output_buf.data(), chunk) != 0) {
            throw std::runtime_error("verification failed: tensor bytes differ for " + std::string(name) + " at byte " +
                                     std::to_string(checked));
        }
        remaining -= chunk;
        checked += chunk;
    }
}

void verify_output(const std::vector<input_shard> &   shards,
                   const std::vector<tensor_source> & selected,
                   const std::string &                output_path,
                   uint64_t                           expected_tensor_bytes) {
    input_shard output = load_shard(output_path);
    verify_metadata(shards.front().gguf.get(), output.gguf.get());

    if (gguf_get_n_tensors(output.gguf.get()) != static_cast<int64_t>(selected.size())) {
        throw std::runtime_error("verification failed: output tensor count differs");
    }

    std::set<std::string> source_names;
    for (const input_shard & shard : shards) {
        for (int64_t i = 0; i < gguf_get_n_tensors(shard.gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(shard.gguf.get(), i);
            source_names.insert(name);
            const int64_t out_id = gguf_find_tensor(output.gguf.get(), name);
            if (wp::is_routed_expert_name(name)) {
                if (out_id >= 0) {
                    throw std::runtime_error("verification failed: routed-expert tensor is present: " +
                                             std::string(name));
                }
            } else if (out_id < 0) {
                throw std::runtime_error("verification failed: non-expert tensor is missing: " + std::string(name));
            }
        }
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(output.gguf.get()); ++i) {
        const char * name = gguf_get_tensor_name(output.gguf.get(), i);
        if (wp::is_routed_expert_name(name) || source_names.count(name) == 0) {
            throw std::runtime_error("verification failed: unexpected output tensor: " + std::string(name));
        }
    }

    std::vector<std::unique_ptr<std::ifstream>> inputs;
    inputs.reserve(shards.size());
    for (const input_shard & shard : shards) {
        inputs.push_back(std::make_unique<std::ifstream>(shard.path, std::ios::binary));
        inputs.back()->exceptions(std::ifstream::failbit | std::ifstream::badbit);
    }
    std::ifstream output_file(output_path, std::ios::binary);
    output_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    std::vector<char> source_buf(COPY_BUFFER_SIZE);
    std::vector<char> output_buf(COPY_BUFFER_SIZE);
    uint64_t          verified_bytes = 0;
    for (const tensor_source & entry : selected) {
        const input_shard & shard  = shards[entry.shard_idx];
        const char *        name   = gguf_get_tensor_name(shard.gguf.get(), entry.tensor_idx);
        const int64_t       out_id = gguf_find_tensor(output.gguf.get(), name);
        if (out_id < 0) {
            throw std::runtime_error("verification failed: selected tensor is missing: " + std::string(name));
        }

        const ggml_tensor * source_tensor = ggml_get_tensor(shard.tensors.get(), name);
        const ggml_tensor * output_tensor = ggml_get_tensor(output.tensors.get(), name);
        verify_tensor_shape(source_tensor, output_tensor, name);

        const uint64_t n_bytes = ggml_nbytes(source_tensor);
        const uint64_t source_offset =
            gguf_get_data_offset(shard.gguf.get()) + gguf_get_tensor_offset(shard.gguf.get(), entry.tensor_idx);
        const uint64_t output_offset =
            gguf_get_data_offset(output.gguf.get()) + gguf_get_tensor_offset(output.gguf.get(), out_id);
        compare_tensor_bytes(*inputs[entry.shard_idx], source_offset, output_file, output_offset, n_bytes, name,
                             source_buf, output_buf);
        verified_bytes += n_bytes;
    }
    if (verified_bytes != expected_tensor_bytes) {
        throw std::runtime_error("verification failed: verified tensor byte total differs");
    }

    uint64_t expected_file_bytes = gguf_get_data_offset(output.gguf.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(output.gguf.get()); ++i) {
        expected_file_bytes += GGML_PAD(gguf_get_tensor_size(output.gguf.get(), i), GGUF_DEFAULT_ALIGNMENT);
    }
    if (fs::file_size(output_path) != expected_file_bytes) {
        throw std::runtime_error("verification failed: output file has trailing or missing bytes");
    }
}

}  // namespace

result extract(const std::string & model_path, const std::string & output_path, bool verify) {
    if (output_path.empty()) {
        throw std::runtime_error("output path is empty");
    }
    if (fs::exists(output_path)) {
        throw std::runtime_error("output file already exists: " + output_path);
    }

    std::vector<input_shard> shards = load_shards(model_path);
    for (const input_shard & shard : shards) {
        if (gguf_get_alignment(shard.gguf.get()) != GGUF_DEFAULT_ALIGNMENT) {
            throw std::runtime_error("source uses unsupported GGUF alignment " +
                                     std::to_string(gguf_get_alignment(shard.gguf.get())) + ": " + shard.path);
        }
    }

    result                     res;
    std::vector<tensor_source> selected;
    std::set<std::string>      tensor_names;
    for (size_t shard_idx = 0; shard_idx < shards.size(); ++shard_idx) {
        const input_shard & shard = shards[shard_idx];
        for (int64_t tensor_idx = 0; tensor_idx < gguf_get_n_tensors(shard.gguf.get()); ++tensor_idx) {
            const char * name = gguf_get_tensor_name(shard.gguf.get(), tensor_idx);
            if (name == nullptr) {
                throw std::runtime_error("GGUF tensor has no name: " + shard.path);
            }
            if (!tensor_names.insert(name).second) {
                throw std::runtime_error("duplicate tensor across GGUF shards: " + std::string(name));
            }
            const ggml_tensor * tensor = ggml_get_tensor(shard.tensors.get(), name);
            if (tensor == nullptr) {
                throw std::runtime_error("GGUF tensor metadata is missing: " + std::string(name));
            }

            const uint64_t n_bytes = ggml_nbytes(tensor);
            if (wp::is_routed_expert_name(name)) {
                ++res.routed_tensor_count;
                res.routed_tensor_bytes += n_bytes;
                continue;
            }
            selected.push_back({ shard_idx, tensor_idx });
            ++res.tensor_count;
            res.tensor_bytes += n_bytes;
        }
    }
    if (selected.empty()) {
        throw std::runtime_error("source contains no non-expert tensors");
    }
    if (res.routed_tensor_count == 0) {
        throw std::runtime_error("source contains no routed-expert tensors");
    }

    gguf_ptr output_gguf(gguf_init_empty(), gguf_free);
    if (output_gguf == nullptr) {
        throw std::runtime_error("failed to create output GGUF metadata");
    }
    gguf_set_kv(output_gguf.get(), shards.front().gguf.get());
    gguf_remove_key(output_gguf.get(), "split.no");
    gguf_remove_key(output_gguf.get(), "split.count");
    gguf_remove_key(output_gguf.get(), "split.tensors.count");
    gguf_set_val_bool(output_gguf.get(), ROUTED_EXPERTS_EXTERNAL_KEY, true);
    for (const tensor_source & entry : selected) {
        const input_shard & shard = shards[entry.shard_idx];
        const char *        name  = gguf_get_tensor_name(shard.gguf.get(), entry.tensor_idx);
        gguf_add_tensor(output_gguf.get(), ggml_get_tensor(shard.tensors.get(), name));
    }

    const std::string temporary_path = output_path + ".tmp";
    if (fs::exists(temporary_path)) {
        throw std::runtime_error("temporary output file already exists: " + temporary_path);
    }
    temporary_output temporary(temporary_path);

    std::vector<std::unique_ptr<std::ifstream>> inputs;
    inputs.reserve(shards.size());
    for (const input_shard & shard : shards) {
        inputs.push_back(std::make_unique<std::ifstream>(shard.path, std::ios::binary));
        inputs.back()->exceptions(std::ifstream::failbit | std::ifstream::badbit);
    }

    std::ofstream output_file(temporary.path(), std::ios::binary);
    output_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    std::vector<uint8_t> metadata(gguf_get_meta_size(output_gguf.get()));
    gguf_get_meta_data(output_gguf.get(), metadata.data());
    output_file.write(reinterpret_cast<const char *>(metadata.data()), static_cast<std::streamsize>(metadata.size()));

    std::vector<char> copy_buf(COPY_BUFFER_SIZE);
    for (const tensor_source & entry : selected) {
        const input_shard & shard   = shards[entry.shard_idx];
        const char *        name    = gguf_get_tensor_name(shard.gguf.get(), entry.tensor_idx);
        const ggml_tensor * tensor  = ggml_get_tensor(shard.tensors.get(), name);
        const uint64_t      n_bytes = ggml_nbytes(tensor);
        const uint64_t      offset =
            gguf_get_data_offset(shard.gguf.get()) + gguf_get_tensor_offset(shard.gguf.get(), entry.tensor_idx);

        std::ifstream & input_file = *inputs[entry.shard_idx];
        input_file.seekg(static_cast<std::streamoff>(offset));
        uint64_t remaining = n_bytes;
        while (remaining > 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, copy_buf.size()));
            input_file.read(copy_buf.data(), static_cast<std::streamsize>(chunk));
            output_file.write(copy_buf.data(), static_cast<std::streamsize>(chunk));
            remaining -= chunk;
        }
        zeros(output_file, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
    }
    output_file.close();

    if (verify) {
        verify_output(shards, selected, temporary.path(), res.tensor_bytes);
        res.verified = true;
    }
    res.file_bytes = fs::file_size(temporary.path());
    fs::rename(temporary.path(), output_path);
    temporary.release();
    return res;
}

}  // namespace wp_dense_extract
