#include "mt-context.h"
#include "mt-tiered.h"

#include "llama-batch.h"  // llama_ubatch (full definition)

namespace mt {

namespace {
// Sentinel ubatch used when the inner context is null. The contract
// expects a reference, so we point at a static empty instance.
const llama_ubatch & sentinel_ubatch() {
    static const llama_ubatch s{};
    return s;
}
}  // namespace

llama_memory_tiered_context::llama_memory_tiered_context(llama_memory_tiered * mem,
                                                         llama_memory_context_ptr inner)
    : mem_(mem),
      inner_ctx_(std::move(inner)) {
}

bool llama_memory_tiered_context::next() {
    return inner_ctx_ ? inner_ctx_->next() : false;
}

bool llama_memory_tiered_context::apply() {
    // Phase 2d-pt: pure passthrough. Future sub-iterations hook tier
    // eviction here (after the inner apply succeeds, before the next
    // ubatch begins).
    return inner_ctx_ ? inner_ctx_->apply() : false;
}

const llama_ubatch & llama_memory_tiered_context::get_ubatch() const {
    return inner_ctx_ ? inner_ctx_->get_ubatch() : sentinel_ubatch();
}

llama_memory_status llama_memory_tiered_context::get_status() const {
    if (!inner_ctx_) return LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    return inner_ctx_->get_status();
}

ggml_tensor * llama_memory_tiered_context::get_turbo_rot_forward() const {
    return inner_ctx_ ? inner_ctx_->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_tiered_context::get_turbo_rot_inverse() const {
    return inner_ctx_ ? inner_ctx_->get_turbo_rot_inverse() : nullptr;
}

ggml_tensor * llama_memory_tiered_context::get_turbo_innerq_scale_inv() const {
    return inner_ctx_ ? inner_ctx_->get_turbo_innerq_scale_inv() : nullptr;
}

}  // namespace mt
