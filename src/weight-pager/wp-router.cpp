#include "weight-pager/wp-router.h"

namespace wp {

const char * const ROUTER_EXPERT_PATTERN     = "ffn_(up|gate|down)_exps\\.";
const char * const ROUTER_SHEXP_PATTERN      = "ffn_(up|gate|down)_shexp\\.";
// FFN island on paging GPU: keeps MoE block intra-device (R9700). Residual then
// only crosses TB3 into/out of the attention island, not per-op mid-FFN.
const char * const ROUTER_FFN_ISLAND_PATTERN =
        "(ffn_norm\\.|ffn_gate_inp\\.|ffn_exp_probs_b\\.|ffn_gate_tid2eid\\.|hc_ffn_)";
const char * const ROUTER_TOKEN_EMBD_PATTERN = "token_embd\\.";
const char * const ROUTER_DENSE_PATTERN      = ".*";

std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        ggml_backend_buffer_type_t resident_buft,
        ggml_backend_buffer_type_t cpu_buft,
        const llama_model_tensor_buft_override * user_overrides,
        bool emit_dense_catch_all) {
    std::vector<llama_model_tensor_buft_override> out;

    // 1) Routed experts -> paging device (catalog/paged pool).
    out.push_back({ ROUTER_EXPERT_PATTERN, paging_buft });

    // 2) Shared expert -> paging device, always-resident (not in paged set).
    out.push_back({ ROUTER_SHEXP_PATTERN, paging_buft });

    // 3) FFN island dense -> paging (T4: fewer TB3 intermediate activations).
    out.push_back({ ROUTER_FFN_ISLAND_PATTERN, paging_buft });

    // 4) token_embd -> CPU when available (row gather; frees eGPU for draft/attn).
    if (cpu_buft != nullptr) {
        out.push_back({ ROUTER_TOKEN_EMBD_PATTERN, cpu_buft });
    }

    // 5) User overrides before dense catch-all so they are never shadowed.
    if (user_overrides != nullptr) {
        for (const auto * o = user_overrides; o->pattern != nullptr; ++o) {
            out.push_back(*o);
        }
    }

    // 6) Everything else dense (attention, lm_head, attn norms, ...)
    //    -> resident / attention-island GPU.
    if (emit_dense_catch_all) {
        out.push_back({ ROUTER_DENSE_PATTERN, resident_buft });
    }
    out.push_back({ nullptr, nullptr });
    return out;
}

} // namespace wp
