#include "wp-stage-split-lib.h"

#include "ggml.h"
#include "gguf.h"
#include "llama-pipeline.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wp_stage_split {

namespace {

constexpr size_t COPY_BUFFER_SIZE = 16u*1024u*1024u;

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
        const std::string path = fs::canonical(fs::path(prefix + split_suffix)).string();
        shards.push_back(load_shard(path));
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

} // namespace

static int32_t model_n_layer(::gguf_context * ctx, int32_t * n_layer_nextn) {
    const int64_t key_arch = gguf_find_key(ctx, "general.architecture");
    if (key_arch < 0) {
        throw std::runtime_error("metadata is missing general.architecture");
    }
    const std::string arch = gguf_get_val_str(ctx, key_arch);

    const std::string key_block_count = arch + ".block_count";
    const int64_t key_bc = gguf_find_key(ctx, key_block_count.c_str());
    if (key_bc < 0) {
        throw std::runtime_error("metadata is missing " + key_block_count);
    }
    const int32_t n_layer_all = (int32_t) gguf_get_val_u32(ctx, key_bc);

    int32_t n_nextn = 0;
    const std::string key_nextn = arch + ".nextn_predict_layers";
    const int64_t key_nn = gguf_find_key(ctx, key_nextn.c_str());
    if (key_nn >= 0) {
        n_nextn = (int32_t) gguf_get_val_u32(ctx, key_nn);
    }
    *n_layer_nextn = n_nextn;

    const int32_t n_layer = n_layer_all - n_nextn;
    if (n_layer <= 0) {
        throw std::runtime_error("bad layer count: block_count=" + std::to_string(n_layer_all) +
                                 " nextn=" + std::to_string(n_nextn));
    }
    return n_layer;
}

result split_stage(const std::string & model_path, const std::string & out_path,
                   int32_t first, int32_t last, bool dry_run) {
    std::vector<input_shard> shards = load_shards(model_path);
    for (const input_shard & shard : shards) {
        if (gguf_get_alignment(shard.gguf.get()) != GGUF_DEFAULT_ALIGNMENT) {
            throw std::runtime_error("source uses a non-default GGUF alignment (" +
                                     std::to_string(gguf_get_alignment(shard.gguf.get())) + "): " + shard.path +
                                     "; not supported yet");
        }
    }

    result  res;
    int32_t n_layer_nextn = 0;
    res.n_layer = model_n_layer(shards.front().gguf.get(), &n_layer_nextn);

    const llama_pipeline_stage band = llama_pipeline_resolve_band(first, last, res.n_layer);
    res.first = band.first;
    res.last  = band.last;

    std::vector<tensor_source> all_tensors;
    std::set<std::string>      tensor_names;
    bool                       has_output = false;
    for (size_t shard_idx = 0; shard_idx < shards.size(); ++shard_idx) {
        const input_shard & shard     = shards[shard_idx];
        const int64_t       n_tensors = gguf_get_n_tensors(shard.gguf.get());
        for (int64_t tensor_idx = 0; tensor_idx < n_tensors; ++tensor_idx) {
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

            all_tensors.push_back({ shard_idx, tensor_idx });
            ++res.n_tensors_in;
            res.bytes_in += (int64_t) ggml_nbytes(tensor);
            has_output = has_output || std::strcmp(name, "output.weight") == 0;
        }
    }

    // a tail with tied embeddings needs token_embd as its lm_head
    const bool tail_tied = band.last == res.n_layer - 1 && !has_output;

    // select the stage's tensors with the same predicate the loader uses
    std::vector<tensor_source> selected;
    for (const tensor_source & source : all_tensors) {
        const input_shard & shard   = shards[source.shard_idx];
        const char *        name    = gguf_get_tensor_name(shard.gguf.get(), source.tensor_idx);
        const ggml_tensor * tensor  = ggml_get_tensor(shard.tensors.get(), name);
        const int64_t       n_bytes = (int64_t) ggml_nbytes(tensor);
        if (llama_pipeline_owns_tensor(
                band.first, band.last, res.n_layer, n_layer_nextn, name, tail_tied)) {
            selected.push_back(source);
            res.bytes_out += n_bytes;
            res.tensor_names.emplace_back(name);
        }
    }
    res.n_tensors_out = (int64_t) selected.size();
    if (selected.empty()) {
        throw std::runtime_error("band selects zero tensors; refusing to write an empty stage");
    }

    if (dry_run) {
        return res;
    }

    if (std::ifstream(out_path)) {
        throw std::runtime_error("output file already exists: " + out_path);
    }

    // all KV metadata unchanged, plus the band markers
    gguf_ptr ctx_out(gguf_init_empty(), gguf_free);
    if (ctx_out == nullptr) {
        throw std::runtime_error("failed to create output GGUF metadata");
    }
    gguf_set_kv(ctx_out.get(), shards.front().gguf.get());
    if (shards.size() > 1) {
        gguf_remove_key(ctx_out.get(), "split.no");
        gguf_remove_key(ctx_out.get(), "split.count");
        gguf_remove_key(ctx_out.get(), "split.tensors.count");
    }

    gguf_set_val_i32(ctx_out.get(), "pipeline.layer_first", band.first);
    gguf_set_val_i32(ctx_out.get(), "pipeline.layer_last", band.last);

    for (const tensor_source & source : selected) {
        const input_shard & shard = shards[source.shard_idx];
        const char *        name  = gguf_get_tensor_name(shard.gguf.get(), source.tensor_idx);
        gguf_add_tensor(ctx_out.get(), ggml_get_tensor(shard.tensors.get(), name));
    }

    // write metadata, then tensor data in info order (gguf-split idiom)
    std::vector<std::unique_ptr<std::ifstream>> inputs;
    inputs.reserve(shards.size());
    for (const input_shard & shard : shards) {
        inputs.push_back(std::make_unique<std::ifstream>(shard.path, std::ios::binary));
        inputs.back()->exceptions(std::ifstream::failbit | std::ifstream::badbit);
    }

    std::ofstream fout(out_path, std::ios::binary);
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    std::vector<uint8_t> meta(gguf_get_meta_size(ctx_out.get()));
    gguf_get_meta_data(ctx_out.get(), meta.data());
    fout.write((const char *) meta.data(), meta.size());

    std::vector<char> copy_buf(COPY_BUFFER_SIZE);
    for (const tensor_source & source : selected) {
        const input_shard & shard   = shards[source.shard_idx];
        const char *        name    = gguf_get_tensor_name(shard.gguf.get(), source.tensor_idx);
        const ggml_tensor * tensor  = ggml_get_tensor(shard.tensors.get(), name);
        const size_t        n_bytes = ggml_nbytes(tensor);

        const size_t offset =
            gguf_get_data_offset(shard.gguf.get()) + gguf_get_tensor_offset(shard.gguf.get(), source.tensor_idx);
        std::ifstream & fin = *inputs[source.shard_idx];
        fin.seekg(offset);

        size_t remaining = n_bytes;
        while (remaining > 0) {
            const size_t chunk = std::min(remaining, copy_buf.size());
            fin.read(copy_buf.data(), chunk);
            fout.write(copy_buf.data(), chunk);
            remaining -= chunk;
        }
        zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
    }

    fout.close();
    return res;
}

} // namespace wp_stage_split
