#pragma once

// wp-stage-split library: the GGUF stage-splitting core, shared by the
// wp-stage-split tool and its unit test.
// See docs/superpowers/specs/2026-07-28-cross-machine-pipeline-parallelism.md
// (Phase 1b).

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context; // from gguf.h; forward-declared to keep the header light

namespace wp_stage_split {

// One split's outcome, for logging and tests.
struct result {
    int32_t first = -1;
    int32_t last  = -1;
    int32_t n_layer = 0;

    int64_t n_tensors_in  = 0;
    int64_t n_tensors_out = 0;
    int64_t bytes_in  = 0;
    int64_t bytes_out = 0;

    std::vector<std::string> tensor_names; // selected, in file order
};

// The model's real (non-NextN) layer count from GGUF metadata:
// n_layer = <arch>.block_count - <arch>.nextn_predict_layers.
// Throws std::runtime_error on missing/inconsistent metadata.
int32_t model_n_layer(::gguf_context * ctx);

// Split the stage band [first, last] out of `model_path` into `out_path`.
// With dry_run, only computes the selection (no file is written).
// Throws std::runtime_error on any inconsistency (bad band, empty selection,
// existing output, unsupported alignment).
result split_stage(const std::string & model_path, const std::string & out_path,
                   int32_t first, int32_t last, bool dry_run);

} // namespace wp_stage_split
