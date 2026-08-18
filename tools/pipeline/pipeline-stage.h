#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// Decode F32 hidden activations through one loaded pipeline band. All input
// tokens use sequence 0 and request an output row.
bool llama_pipeline_stage_decode_hidden(
        llama_context * ctx,
        int32_t n_embd,
        const float * activations,
        int32_t n_tokens,
        const int32_t * positions,
        uint32_t n_pos_per_token);

// Read the requested embedding rows after llama_pipeline_stage_decode_hidden.
bool llama_pipeline_stage_read_hidden(
        llama_context * ctx,
        int32_t n_tokens,
        int32_t n_embd,
        std::vector<float> & out);
