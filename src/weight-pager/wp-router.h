#pragma once

#include "llama.h"          // llama_model_tensor_buft_override
#include "ggml-backend.h"   // ggml_backend_buffer_type_t

#include <cstdint>
#include <string>
#include <vector>

namespace wp {

// Routed-expert tensor-name regex (consolidated MoE experts). MUST stay
// identical to the paging catalog / is_paged_weight filter.
extern const char * const ROUTER_EXPERT_PATTERN;

// Shared expert (shexp) - always-resident on the *paging* GPU so the eGPU
// resident card is free for attention+draft; not paged (stays in VRAM).
extern const char * const ROUTER_SHEXP_PATTERN;

// FFN-side dense that should co-locate with experts on the paging GPU so
// the residual only crosses TB3 at attention boundaries (T4 fewer crossings).
// Covers: ffn_norm, router, exp bias, tid2eid tables, hyper-connection FFN.
extern const char * const ROUTER_FFN_ISLAND_PATTERN;

// Token embeddings - host/CPU (row gather only).
extern const char * const ROUTER_TOKEN_EMBD_PATTERN;

// Dense catch-all regex (".*") used to pin every non-expert, non-user-overridden
// tensor to the resident GPU buffer instead of a host/CPU fallback.
extern const char * const ROUTER_DENSE_PATTERN;

// ---------------------------------------------------------------------------
// Resident expert layers
// ---------------------------------------------------------------------------
//
// By default every routed-expert tensor is paged (ROUTER_EXPERT_PATTERN ->
// paging device). When a second GPU has VRAM to spare, whole transformer
// blocks' routed experts can instead be held RESIDENT on it: allocated and
// loaded by the normal allocator, computed in place, never paged.
//
// This is deliberately whole-BLOCK granularity. Routed experts are a single
// fused 3D tensor {n_embd, n_ff_exp, n_expert} that mul_mat_id indexes into,
// so a buffer-type override cannot place individual experts on different
// devices -- that would need the matmul split across devices plus a
// cross-device partial-sum reduction. Whole blocks need neither.
//
// Nothing here is model-specific: the block set is derived at load time from
// the measured per-block expert bytes of whatever model is being loaded and a
// device VRAM budget.

// Measured routed-expert footprint of one transformer block.
//
// `bytes` is the whole block: all three role tensors summed, which is what a
// resident block actually costs on the island device.
//
// `max_page_bytes` is the largest single PAGE the block contributes to the
// pager. The catalog splits each consolidated role tensor into one sub-page per
// expert (wp-page-catalog.cpp: per_expert_size = tensor_bytes / n_experts), so
// one expert is three pages and the pool's uniform slot stride is the max over
// all of them. Block totals are NOT a proxy for this: a (10,10,10) block totals
// 30 with a max page of 10, while (5,5,18) totals 28 with a max page of 18.
struct LayerExpertBytes {
    int      block_idx      = -1;
    uint64_t bytes          = 0;
    uint64_t max_page_bytes = 0;
};

// True iff `name` is a consolidated routed-expert tensor (ffn_{up,gate,down}_exps).
// Shared experts (ffn_*_shexp) and the router (ffn_gate_inp) are NOT routed experts.
bool is_routed_expert_name(const char * name);

// Parse the leading "blk.<N>." of a tensor name. Returns false if absent.
bool parse_block_index(const char * name, int & block_idx);

// The set of blocks whose routed experts are resident rather than paged.
//
// Empty by default, which reproduces the previous behaviour exactly.
class ResidentExpertPlan {
public:
    bool                     empty()    const { return layers_.empty(); }
    const std::vector<int> & layers()   const { return layers_; }
    uint64_t                 bytes()    const { return bytes_; }

    bool covers_block(int block_idx) const;

    // True iff `name` is a routed-expert tensor belonging to a covered block.
    // This is THE predicate: the buft override, is_paged_weight() and the
    // pager-catalog filter must all consult it, or a tensor ends up
    // allocated-but-never-loaded (garbage) or paged-but-never-allocated.
    bool covers_tensor(const char * name) const;

    // Regex matching exactly the covered routed-expert tensors, e.g.
    // "blk\.(3|4|17)\.ffn_(up|gate|down)_exps\.". Empty when the plan is empty.
    // Stable storage: the returned string outlives any override list built from
    // it only as long as this object does.
    const std::string & pattern() const { return pattern_; }

    std::string describe() const;


    // Explicit blocks, e.g. for a deliberate cross-machine split where the
    // range must be contiguous rather than whichever blocks happen to be
    // biggest. Blocks with no routed experts in `per_layer` are ignored.
    static ResidentExpertPlan from_blocks(const std::vector<LayerExpertBytes> & per_layer,
                                          const std::vector<int> &              blocks);

private:
    void rebuild_pattern();

    std::vector<int> layers_;   // sorted, unique
    uint64_t         bytes_ = 0;
    std::string      pattern_;
};

// Parse the --weight-paging-resident-experts value.
//
//   ""/"off"/"none"  -> disabled
//   "<ranges>"       -> blocks = expanded ranges ("0-6,20-22" or "5")
// "auto" and byte sizes are deliberately REJECTED -- see ResidentExpertPlan.
//
// Throws std::invalid_argument on malformed input.
struct ResidentExpertRequest {
    bool             enabled      = false;
    std::vector<int> blocks;
};
ResidentExpertRequest parse_resident_expert_request(const char * value);

// Explicit paged layer bands. Syntax:
//   "ROCm0:0-37;ROCm1:38-74"
// Every group names one backend device and one non-empty block range list.
// There is deliberately no auto or budget form.
struct DeviceLayerRequest {
    std::string      device;
    std::vector<int> blocks;
};

std::vector<DeviceLayerRequest> parse_device_layer_request(const char * value);

class DeviceLayerPlan {
public:
    struct Entry {
        std::string                    device;
        std::vector<int>               blocks;
        std::string                    pattern;
        ggml_backend_buffer_type_t     buft = nullptr;
    };

    void add(
        std::string device, std::vector<int> blocks,
        ggml_backend_buffer_type_t buft);

    bool empty() const { return entries_.empty(); }
    const std::vector<Entry> & entries() const { return entries_; }
    ggml_backend_buffer_type_t buft_for_tensor(
        const char * name, ggml_backend_buffer_type_t fallback) const;
    bool covers_block(int block_idx) const;

private:
    std::vector<Entry> entries_;
};

// Build the tensor_buft_override list for the hetero resident-dense router:
//   0. resident-expert blocks -> island GPU (resident, NOT paged) when the plan
//                         is non-empty. Must precede rule 1, which is a
//                         superset pattern and would otherwise shadow it.
//   1. routed experts  -> paging GPU (paged pool)
//   2. shexp           -> paging GPU (always-resident, not paged), or island GPU
//   3. FFN island      -> paging GPU (norm/router/hc_ffn; cuts TB3 intermediates),
//                         or island GPU
//   4. token_embd      -> CPU
//   5. <user overrides>
//   6. .*              -> resident GPU (attention island + lm_head + ...)
// First match wins. Patterns are static string literals; user patterns
// are borrowed from `user_overrides` and must outlive the result.
// cpu_buft may be null: token_embd then falls through to resident (single-GPU).
// emit_dense_catch_all is false when layer-home allocation spans residents.
// island_buft may be null (default): shexp and FFN island route to paging_buft
// as before. When non-null, shexp and FFN island route to island_buft instead
// (FFN-island device role: shared experts + router live on a second GPU).
std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        ggml_backend_buffer_type_t resident_buft,
        ggml_backend_buffer_type_t cpu_buft,
        const llama_model_tensor_buft_override * user_overrides,
        bool emit_dense_catch_all = true,
        ggml_backend_buffer_type_t island_buft = nullptr,
        const ResidentExpertPlan * resident_experts = nullptr,
        const DeviceLayerPlan * device_layers = nullptr);

} // namespace wp
