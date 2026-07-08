#pragma once

#include "llama.h"          // llama_model_tensor_buft_override
#include "ggml-backend.h"   // ggml_backend_buffer_type_t

#include <vector>

namespace wp {

// Routed-expert tensor-name regex (consolidated MoE experts). MUST stay
// identical to the paging catalog / is_paged_weight filter.
extern const char * const ROUTER_EXPERT_PATTERN;

// Dense catch-all regex (".*") used to pin every non-expert, non-user-overridden
// tensor to the resident GPU buffer instead of a host/CPU fallback.
extern const char * const ROUTER_DENSE_PATTERN;

// Build the tensor_buft_override list for the resident-dense device router:
//   [ {experts -> paging}, <user overrides...>, {".*" -> resident}, {null,null} ]
// experts match first (win over the ".*"); user overrides come before the ".*"
// catch-all so user intent is never shadowed; the trailing ".*" pins all
// remaining dense tensors to the resident GPU buffer. The expert/dense .pattern
// values are static string literals (stable); user patterns are borrowed from
// `user_overrides` and must outlive the result.
std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        ggml_backend_buffer_type_t resident_buft,
        const llama_model_tensor_buft_override * user_overrides);

} // namespace wp
