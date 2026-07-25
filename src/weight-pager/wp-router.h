#pragma once

#include "llama.h"          // llama_model_tensor_buft_override
#include "ggml-backend.h"   // ggml_backend_buffer_type_t

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

// Build the tensor_buft_override list for the hetero resident-dense router:
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
        ggml_backend_buffer_type_t island_buft = nullptr);

} // namespace wp
