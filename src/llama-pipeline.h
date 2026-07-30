#pragma once

// Cross-machine pipeline parallelism: per-process contiguous layer bands.
// See docs/superpowers/specs/2026-07-28-cross-machine-pipeline-parallelism.md.
//
// A band [first, last] names the transformer block indices a single process
// owns. Both bounds -1 means "own everything" and MUST reproduce the
// pre-pipeline behaviour exactly -- every call site checks
// llama_pipeline_band_enabled() first and takes the legacy path when false.
//
// Absolute layer indices are preserved end to end: a tensor named blk.57.* is
// layer 57 on every machine. hparams.n_layer stays GLOBAL; bands never
// renumber.

#include <cstdint>
#include <vector>

struct llama_pipeline_stage {
    int32_t first = -1;
    int32_t last  = -1;
};

// true when a band was requested (both bounds set)
bool llama_pipeline_band_enabled(int32_t first, int32_t last);

// Validate a requested band against a model with n_layer real (non-NextN)
// layers. Returns the resolved band: with both bounds unset this is the full
// range [0, n_layer-1]. Throws std::runtime_error on anything that would
// produce a model that runs and emits garbage:
//   - exactly one bound set
//   - empty band (first > last)
//   - band outside [0, n_layer-1]
// Bands are contiguous by construction (a single [first, last] interval), so
// discontinuity cannot pass validation here.
llama_pipeline_stage llama_pipeline_resolve_band(int32_t first, int32_t last, int32_t n_layer);

// Does the stage [first, last] of an n_layer model own the GGUF tensor
// `name`? Names follow the GGUF convention: "blk.N.<...>" for per-layer
// tensors, "token_embd.<...>", "output_norm.<...>", "output.<...>" for the
// shared tensors.
//   - blk.N.*            -> first <= N <= last
//   - blk.N.*, N >= n_layer (NextN/MTP) -> the head when
//                           n_layer_nextn > 0, otherwise the tail
//   - token_embd.*       -> first == 0 (the head owns embeddings), unless
//                           duplicated_embd is set (a tail with tied
//                           embeddings loads token_embd as its lm_head)
//   - output_norm.*/output.* -> the tail, plus the head when
//                           n_layer_nextn > 0
//   - anything else      -> owned (small global tensors: biases, norms that
//                           do not follow the blk convention, etc.)
// This single predicate is used both by the loader (which tensors to create)
// and by wp-stage-split (which tensors to write), so the two can never
// disagree about stage contents.
bool llama_pipeline_owns_tensor(
    int32_t first, int32_t last, int32_t n_layer, int32_t n_layer_nextn, const char * name, bool duplicated_embd);

// Validate a complete pipeline: the stages, in order, must cover [0, n_layer-1]
// exactly once -- no gaps (continuity), no overlaps (two stages owning one
// layer). This implies exactly one head (first == 0, owns token_embd) and
// exactly one tail (last == n_layer-1, owns output_norm/output). Throws
// std::runtime_error describing the first inconsistency found.
void llama_pipeline_validate_stages(const std::vector<llama_pipeline_stage> & stages, int32_t n_layer);

// Parse the block index out of a GGUF tensor name ("blk.N.<...>", also
// "enc.blk.N."/"dec.blk.N."). Returns -1 when the name carries no block
// index. Exposed for the pager-catalog band assertion.
int32_t llama_pipeline_tensor_block_index(const char * name);
