#include "mt-mover-recurrent.h"

#include "ggml.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <cstring>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace mt {

#if defined(GGML_USE_HIP)

namespace {
// Returns the device-side byte address of seq `slot`'s slice in `t`,
// using nb[1] as the per-seq stride.
inline const uint8_t * seq_slice_ptr(const ggml_tensor * t, int slot) {
    return (const uint8_t *) t->data + (size_t) slot * t->nb[1];
}
inline uint8_t * seq_slice_ptr_mut(ggml_tensor * t, int slot) {
    return (uint8_t *) t->data + (size_t) slot * t->nb[1];
}
}  // namespace

bool RecurrentStateMover::evict_seq(const RecurrentStateView & seq, void * dst_host) const {
    if (dst_host == nullptr || seq.seq_slot < 0 || seq.layers.empty()) {
        return false;
    }

    uint8_t * dst = (uint8_t *) dst_host;

    // Pass 1: r tensors of all layers, in order.
    for (size_t i = 0; i < seq.layers.size(); ++i) {
        const auto & L = seq.layers[i];
        if (L.r == nullptr || L.r_bytes_per_seq == 0) continue;
        const uint8_t * src = seq_slice_ptr(L.r, seq.seq_slot);
        hipError_t err = hipMemcpy(dst, src, L.r_bytes_per_seq, hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("mt::RecurrentStateMover::evict_seq: r layer %zu hipMemcpy failed: %s\n",
                           i, hipGetErrorString(err));
            return false;
        }
        dst += L.r_bytes_per_seq;
    }

    // Pass 2: s tensors.
    for (size_t i = 0; i < seq.layers.size(); ++i) {
        const auto & L = seq.layers[i];
        if (L.s == nullptr || L.s_bytes_per_seq == 0) continue;
        const uint8_t * src = seq_slice_ptr(L.s, seq.seq_slot);
        hipError_t err = hipMemcpy(dst, src, L.s_bytes_per_seq, hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("mt::RecurrentStateMover::evict_seq: s layer %zu hipMemcpy failed: %s\n",
                           i, hipGetErrorString(err));
            return false;
        }
        dst += L.s_bytes_per_seq;
    }

    return true;
}

bool RecurrentStateMover::restore_seq(const RecurrentStateView & seq, const void * src_host) const {
    if (src_host == nullptr || seq.seq_slot < 0 || seq.layers.empty()) {
        return false;
    }

    const uint8_t * src = (const uint8_t *) src_host;

    for (size_t i = 0; i < seq.layers.size(); ++i) {
        const auto & L = seq.layers[i];
        if (L.r == nullptr || L.r_bytes_per_seq == 0) continue;
        uint8_t * dst = seq_slice_ptr_mut(L.r, seq.seq_slot);
        hipError_t err = hipMemcpy(dst, src, L.r_bytes_per_seq, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("mt::RecurrentStateMover::restore_seq: r layer %zu hipMemcpy failed: %s\n",
                           i, hipGetErrorString(err));
            return false;
        }
        src += L.r_bytes_per_seq;
    }

    for (size_t i = 0; i < seq.layers.size(); ++i) {
        const auto & L = seq.layers[i];
        if (L.s == nullptr || L.s_bytes_per_seq == 0) continue;
        uint8_t * dst = seq_slice_ptr_mut(L.s, seq.seq_slot);
        hipError_t err = hipMemcpy(dst, src, L.s_bytes_per_seq, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("mt::RecurrentStateMover::restore_seq: s layer %zu hipMemcpy failed: %s\n",
                           i, hipGetErrorString(err));
            return false;
        }
        src += L.s_bytes_per_seq;
    }

    return true;
}

#else  // !GGML_USE_HIP

bool RecurrentStateMover::evict_seq(const RecurrentStateView &, void *) const         { return false; }
bool RecurrentStateMover::restore_seq(const RecurrentStateView &, const void *) const { return false; }

#endif  // GGML_USE_HIP

}  // namespace mt
