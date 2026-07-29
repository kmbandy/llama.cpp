#include "wp-stage-split-lib.h"

#include "ggml.h"
#include "gguf.h"
#include "llama-pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace wp_stage_split {

namespace {

constexpr size_t COPY_BUFFER_SIZE = 16u*1024u*1024u;

void zeros(std::ostream & file, size_t n) {
    static const char zeros_buf[4096] = {};
    while (n > 0) {
        const size_t chunk = std::min(n, sizeof(zeros_buf));
        file.write(zeros_buf, chunk);
        n -= chunk;
    }
}

} // namespace

int32_t model_n_layer(::gguf_context * ctx) {
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

    const int32_t n_layer = n_layer_all - n_nextn;
    if (n_layer <= 0) {
        throw std::runtime_error("bad layer count: block_count=" + std::to_string(n_layer_all) +
                                 " nextn=" + std::to_string(n_nextn));
    }
    return n_layer;
}

result split_stage(const std::string & model_path, const std::string & out_path,
                   int32_t first, int32_t last, bool dry_run) {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params iparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx_meta,
    };
    gguf_context * ctx_in = gguf_init_from_file(model_path.c_str(), iparams);
    if (ctx_in == nullptr) {
        throw std::runtime_error("failed to load GGUF metadata from " + model_path);
    }

    result res;
    res.n_layer = model_n_layer(ctx_in);

    const llama_pipeline_stage band = llama_pipeline_resolve_band(first, last, res.n_layer);
    res.first = band.first;
    res.last  = band.last;

    // a tail with tied embeddings needs token_embd as its lm_head
    const bool has_output = gguf_find_tensor(ctx_in, "output.weight") >= 0;
    const bool tail_tied  = band.last == res.n_layer - 1 && !has_output;

    // select the stage's tensors with the same predicate the loader uses
    std::vector<int64_t> selected;
    const int64_t n_tensors = gguf_get_n_tensors(ctx_in);
    res.n_tensors_in = n_tensors;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        const ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        const int64_t n_bytes = (int64_t) ggml_nbytes(t);
        res.bytes_in += n_bytes;
        if (llama_pipeline_owns_tensor(band.first, band.last, res.n_layer, name, tail_tied)) {
            selected.push_back(i);
            res.bytes_out += n_bytes;
            res.tensor_names.emplace_back(name);
        }
    }
    res.n_tensors_out = (int64_t) selected.size();
    if (selected.empty()) {
        gguf_free(ctx_in);
        ggml_free(ctx_meta);
        throw std::runtime_error("band selects zero tensors; refusing to write an empty stage");
    }

    if (dry_run) {
        gguf_free(ctx_in);
        ggml_free(ctx_meta);
        return res;
    }

    if (std::ifstream(out_path)) {
        gguf_free(ctx_in);
        ggml_free(ctx_meta);
        throw std::runtime_error("output file already exists: " + out_path);
    }

    // all KV metadata unchanged, plus the band markers
    gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    // gguf_init_empty fixes ctx->alignment at GGUF_DEFAULT_ALIGNMENT and both
    // the tensor offsets (gguf_add_tensor) and the data pads below use it. A
    // source file with a non-default alignment KV would be copied verbatim by
    // gguf_set_kv but interpreted with the default -- refuse rather than
    // write a corrupt stage.
    if (gguf_get_alignment(ctx_in) != gguf_get_alignment(ctx_out)) {
        gguf_free(ctx_out);
        gguf_free(ctx_in);
        ggml_free(ctx_meta);
        throw std::runtime_error(
            "source uses a non-default GGUF alignment (" +
            std::to_string(gguf_get_alignment(ctx_in)) + "); not supported yet");
    }

    gguf_set_val_i32(ctx_out, "pipeline.layer_first", band.first);
    gguf_set_val_i32(ctx_out, "pipeline.layer_last",  band.last);

    for (int64_t i : selected) {
        gguf_add_tensor(ctx_out, ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_in, i)));
    }

    // write metadata, then tensor data in info order (gguf-split idiom)
    std::ifstream fin(model_path, std::ios::binary);
    fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    std::ofstream fout(out_path, std::ios::binary);
    fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    std::vector<uint8_t> meta(gguf_get_meta_size(ctx_out));
    gguf_get_meta_data(ctx_out, meta.data());
    fout.write((const char *) meta.data(), meta.size());

    std::vector<char> copy_buf(COPY_BUFFER_SIZE);
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
        const char * name = gguf_get_tensor_name(ctx_out, i);
        const int64_t i_in = gguf_find_tensor(ctx_in, name);
        const ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        const size_t n_bytes = ggml_nbytes(t);

        const size_t offset = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, i_in);
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
    fin.close();

    gguf_free(ctx_out);
    gguf_free(ctx_in);
    ggml_free(ctx_meta);
    return res;
}

} // namespace wp_stage_split
