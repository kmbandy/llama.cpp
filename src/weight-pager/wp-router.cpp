#include "weight-pager/wp-router.h"

namespace wp {

const char * const ROUTER_EXPERT_PATTERN = "ffn_(up|gate|down)_exps\\.";
const char * const ROUTER_DENSE_PATTERN  = ".*";

std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        ggml_backend_buffer_type_t resident_buft,
        const llama_model_tensor_buft_override * user_overrides) {
    std::vector<llama_model_tensor_buft_override> out;
    out.push_back({ ROUTER_EXPERT_PATTERN, paging_buft });
    // Offload the two large non-attention dense tensors to the paging card to
    // free resident-card VRAM (they stay resident there, not paged).
    out.push_back({ "token_embd\\.",  paging_buft });
    out.push_back({ "output\\.weight", paging_buft });
    if (user_overrides != nullptr) {
        for (const auto * o = user_overrides; o->pattern != nullptr; ++o) {
            out.push_back(*o);
        }
    }
    out.push_back({ ROUTER_DENSE_PATTERN, resident_buft });
    out.push_back({ nullptr, nullptr });
    return out;
}

} // namespace wp
