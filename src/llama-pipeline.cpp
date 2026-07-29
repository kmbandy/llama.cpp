#include "llama-pipeline.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

static std::string band_err(const char * fmt, int32_t a, int32_t b, int32_t c) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), fmt, a, b, c);
    return std::string(buf);
}

bool llama_pipeline_band_enabled(int32_t first, int32_t last) {
    return first != -1 || last != -1;
}

llama_pipeline_stage llama_pipeline_resolve_band(int32_t first, int32_t last, int32_t n_layer) {
    if (!llama_pipeline_band_enabled(first, last)) {
        return {0, n_layer - 1};
    }
    if (first == -1 || last == -1) {
        throw std::runtime_error(band_err(
            "pipeline: layer band must set both bounds or neither (got first=%d, last=%d)",
            first, last, 0));
    }
    if (n_layer <= 0) {
        throw std::runtime_error(band_err(
            "pipeline: model reports n_layer=%d, cannot resolve a layer band", n_layer, 0, 0));
    }
    if (first < 0 || last >= n_layer) {
        throw std::runtime_error(band_err(
            "pipeline: layer band [%d, %d] is outside the model's layer range [0, %d]",
            first, last, n_layer - 1));
    }
    if (first > last) {
        throw std::runtime_error(band_err(
            "pipeline: layer band [%d, %d] is empty (first > last)", first, last, 0));
    }
    return {first, last};
}

int32_t llama_pipeline_tensor_block_index(const char * name) {
    if (name == nullptr) {
        return -1;
    }
    // per-layer tensors carry "blk.N." somewhere in the name ("blk.N.*" for
    // decoder-only models, "enc.blk.N.*"/"dec.blk.N.*" for encoder-decoder)
    const char * p = std::strstr(name, "blk.");
    if (p == nullptr) {
        return -1;
    }
    p += 4;
    if (*p < '0' || *p > '9') {
        return -1;
    }
    int32_t idx = 0;
    while (*p >= '0' && *p <= '9') {
        idx = idx*10 + (*p - '0');
        ++p;
    }
    if (*p != '.') {
        return -1;
    }
    return idx;
}

bool llama_pipeline_owns_tensor(int32_t first, int32_t last, int32_t n_layer, const char * name, bool duplicated_embd) {
    if (name == nullptr) {
        return false;
    }
    const int32_t blk = llama_pipeline_tensor_block_index(name);
    if (blk >= 0) {
        if (blk >= n_layer) {
            // NextN/MTP tensors sit past the last real layer. They belong to
            // the tail: that is where the base model's final hidden state
            // (their input) exists.
            return last == n_layer - 1;
        }
        return blk >= first && blk <= last;
    }
    if (std::strncmp(name, "token_embd.", 11) == 0) {
        // a tail with tied embeddings loads token_embd AS its output tensor
        // (duplicated_embd is the loader's TENSOR_DUPLICATED fallback); a
        // head or middle stage must never claim token_embd through it
        return first == 0 || (duplicated_embd && last == n_layer - 1);
    }
    if (std::strncmp(name, "output_norm.", 12) == 0) {
        return last == n_layer - 1;
    }
    if (std::strncmp(name, "output.", 7) == 0) {
        return last == n_layer - 1;
    }
    // small global tensors (no block index, not one of the gated shared
    // tensors) are owned by every stage that loads them
    return true;
}

void llama_pipeline_validate_stages(const std::vector<llama_pipeline_stage> & stages, int32_t n_layer) {
    if (stages.empty()) {
        throw std::runtime_error("pipeline: no stages defined");
    }
    if (n_layer <= 0) {
        throw std::runtime_error(band_err(
            "pipeline: model reports n_layer=%d, cannot validate stages", n_layer, 0, 0));
    }

    int32_t expected_first = 0;
    for (size_t i = 0; i < stages.size(); ++i) {
        const llama_pipeline_stage band =
            llama_pipeline_resolve_band(stages[i].first, stages[i].last, n_layer);

        if (band.first != expected_first) {
            if (band.first < expected_first) {
                throw std::runtime_error(band_err(
                    "pipeline: stage %d starts at layer %d, overlapping a layer already owned "
                    "(two stages would compute the same blk.* tensors)", (int32_t) i, band.first, 0));
            }
            throw std::runtime_error(band_err(
                "pipeline: stage %d starts at layer %d but layer %d is unowned "
                "(gap: nobody computes those blk.* tensors)", (int32_t) i, band.first, expected_first));
        }
        expected_first = band.last + 1;
    }
    if (expected_first != n_layer) {
        throw std::runtime_error(band_err(
            "pipeline: stages end at layer %d but the model has %d layers "
            "(tail missing: nobody owns output_norm/output)", expected_first - 1, n_layer, 0));
    }
}
