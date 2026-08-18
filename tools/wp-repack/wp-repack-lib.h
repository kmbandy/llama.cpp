#pragma once

#include "weight-pager/wp-page-catalog.h"

#include <cstdint>
#include <string>
#include <vector>

namespace wp_repack {

struct ExpertMember {
    uint8_t     role_mask   = 0;
    uint16_t    file_idx    = 0;
    uint64_t    file_offset = 0;
    uint64_t    size        = 0;
    std::string catalog_name;
    std::string source_tensor_name;
};

struct ExpertGroup {
    int                       block_idx  = -1;
    int                       expert_idx = -1;
    uint64_t                  size       = 0;
    std::vector<ExpertMember> members;
};

struct LayerRange {
    int first = -1;
    int last  = -1;
};

struct ShardPlan {
    int                 layer_first = -1;
    int                 layer_last  = -1;
    uint64_t            size        = 0;
    std::vector<size_t> group_indices;
};

// ---------------------------------------------------------------------------
// Expert slicing (repack format v2).
//
// NOTE ON WORDS. "Shard" in this tool has always meant an OUTPUT FILE holding a
// contiguous range of expert LAYERS. That meaning is unchanged. The v2 concept
// is a "slice": a contiguous range of the FFN INTERMEDIATE dimension of a single
// expert, held by one GPU. A slice is a strict subdivision of an expert; shards
// and slices are orthogonal and both appear in a v2 output set.
//
// A slice [a, b) of expert E is:
//   ffn_up_exps   rows    [a, b)   (contiguous in the source tensor)
//   ffn_gate_exps rows    [a, b)   (contiguous in the source tensor)
//   ffn_down_exps columns [a, b)   (STRIDED in the source tensor; gathered here)
//
// which is self-contained: it consumes the full-width activation and emits a
// full-width n_embd output vector, so the per-slice outputs simply sum.
// ---------------------------------------------------------------------------

struct SliceSpec {
    // Widths along the FFN intermediate dimension, in elements. Sums to n_ff_exp.
    std::vector<int64_t> widths;
    // The spec exactly as the user typed it, recorded in the index for provenance.
    std::string          text;
    // True when `text` was a ratio list and `widths` were solved from it.
    bool                 from_ratios = false;
    // Only set when from_ratios: the parsed ratio weights, recorded for provenance.
    std::vector<int64_t> ratios;
};

// Accepts either an explicit width list ("1024,512,256,256") or a ratio list
// ("4:2:1:1"). A ratio list leaves `widths` empty until resolve_slice_widths is
// called with the model's real n_ff_exp; an explicit list is returned ready.
// Throws std::invalid_argument on anything malformed, zero, or negative.
SliceSpec parse_slice_spec(const std::string & text);

// Fill in / validate `spec.widths` against the model's FFN intermediate size and
// the quantization block size of the expert tensors.
//
// The alignment rule is the whole ballgame: every slice boundary must land on a
// quant-block boundary, or a slice would cut a block in half and its bytes would
// no longer be the official release bytes. So every width must be a positive
// multiple of `blck`, and the widths must sum to EXACTLY n_ff. Ratios are solved
// in units of `blck` by largest-remainder, which keeps the split deterministic
// and guarantees the sum lands on n_ff with no leftover.
//
// Throws std::invalid_argument if n_ff is not a multiple of blck, if a resolved
// or explicit width is not a multiple of blck, if any width is <= 0, if the
// widths do not sum to n_ff, or if there are more slices than blocks to go round.
void resolve_slice_widths(SliceSpec & spec, int64_t n_ff, int64_t blck);

// Half-open [first, last) range along the FFN intermediate dimension.
struct SliceRange {
    int     index = -1;
    int64_t first = 0;
    int64_t last  = 0;

    int64_t width() const { return last - first; }
};

// Prefix-sum `widths` into half-open ranges. widths must already be resolved.
std::vector<SliceRange> slice_ranges(const std::vector<int64_t> & widths);

std::vector<ExpertGroup> build_expert_groups(const wp::PageCatalog & catalog);

std::vector<LayerRange> parse_layer_ranges(const std::string & text);

std::vector<ShardPlan> plan_shards_by_layer(const std::vector<ExpertGroup> & groups);

std::vector<ShardPlan> plan_shards_max_bytes(const std::vector<ExpertGroup> & groups, uint64_t max_shard_bytes);

// Explicit layer ranges. Any expert group whose layer falls outside `ranges` is
// NOT written to the shard set. That is occasionally wanted (repacking a subset
// for a test) but is a silent-data-loss footgun for the primary use case, where
// the ranges describe a machine split and a mistyped boundary would drop a whole
// layer while still reporting success -- and --verify cannot catch it, because it
// validates what the index claims, not what the model contains.
//
// So incomplete coverage THROWS unless allow_partial is set. The message names the
// uncovered layers.
std::vector<ShardPlan> plan_shards_for_ranges(const std::vector<ExpertGroup> & groups,
                                              const std::vector<LayerRange> &  ranges,
                                              bool                             allow_partial = false);

// Layers present in `groups` that no range covers. Empty == full coverage.
// Exposed separately so callers can report coverage without triggering the throw.
std::vector<int> uncovered_layers(const std::vector<ExpertGroup> & groups,
                                  const std::vector<LayerRange> &  ranges);

}  // namespace wp_repack
