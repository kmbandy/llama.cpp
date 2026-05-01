#pragma once

// llama_memory_tiered_context — wraps llama_memory_context_i for the
// tier system. Phase 2d-pt is pure passthrough: every method
// delegates to the inner context. Tier eviction will hook into
// apply() in subsequent sub-iterations (apply() is the only mutating
// call point in the memory_context contract).

#include "llama-memory.h"

#include <memory>

namespace mt {

class llama_memory_tiered;  // fwd

class llama_memory_tiered_context : public llama_memory_context_i {
public:
    // Wrap an inner context. The wrapper owns it.
    llama_memory_tiered_context(llama_memory_tiered * mem,
                                llama_memory_context_ptr inner);

    ~llama_memory_tiered_context() override = default;

    // ---- llama_memory_context_i ----
    bool                 next()        override;
    bool                 apply()       override;
    const llama_ubatch & get_ubatch()  const override;
    llama_memory_status  get_status()  const override;

    ggml_tensor * get_turbo_rot_forward()      const override;
    ggml_tensor * get_turbo_rot_inverse()      const override;
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

private:
    llama_memory_tiered     * mem_;        // borrowed; never null after construction
    llama_memory_context_ptr  inner_ctx_;
};

}  // namespace mt
