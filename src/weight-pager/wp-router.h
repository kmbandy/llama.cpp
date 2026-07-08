#pragma once

#include "llama.h"          // llama_model_tensor_buft_override
#include "ggml-backend.h"   // ggml_backend_buffer_type_t

#include <vector>

namespace wp {

// Routed-expert tensor-name regex (consolidated MoE experts). MUST stay
// identical to the paging catalog / is_paged_weight filter.
extern const char * const ROUTER_EXPERT_PATTERN;

// Build the tensor_buft_override list for the resident-dense device router.
// Routes ONLY routed-expert tensors to `paging_buft`; every other tensor is
// left with no override so it defaults to its layer-home device. Any
// caller-supplied user overrides (nullptr-terminated array, may be null) are
// appended AFTER the expert entry, then a {nullptr,nullptr} terminator.
// The expert entry's .pattern is a static string literal (stable); user
// patterns are borrowed from `user_overrides` and must outlive the result.
std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        const llama_model_tensor_buft_override * user_overrides);

} // namespace wp
