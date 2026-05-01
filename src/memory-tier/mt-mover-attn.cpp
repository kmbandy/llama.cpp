#include "mt-mover-attn.h"

#include "ggml.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <cstring>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace mt {

#if defined(GGML_USE_HIP)

namespace {
inline bool valid_layer(const AttentionLayerView & layer) {
    return layer.k != nullptr && layer.v != nullptr
        && layer.k_row_bytes > 0 && layer.v_row_bytes > 0
        && layer.kv_size > 0;
}
}  // namespace

bool AttentionMover::evict_k(const AttentionLayerView & layer,
                              int64_t                    pos,
                              void *                     dst) const {
    if (!valid_layer(layer) || dst == nullptr || pos < 0 || pos >= layer.kv_size) {
        return false;
    }
    const uint8_t * src = (const uint8_t *) layer.k->data + (size_t) pos * layer.k_row_bytes;
    hipError_t err = hipMemcpy(dst, src, layer.k_row_bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("mt::AttentionMover::evict_k: hipMemcpy failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool AttentionMover::restore_k(const AttentionLayerView & layer,
                                const void *               src,
                                int64_t                    pos) const {
    if (!valid_layer(layer) || src == nullptr || pos < 0 || pos >= layer.kv_size) {
        return false;
    }
    uint8_t * dst = (uint8_t *) layer.k->data + (size_t) pos * layer.k_row_bytes;
    hipError_t err = hipMemcpy(dst, src, layer.k_row_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("mt::AttentionMover::restore_k: hipMemcpy failed: %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool AttentionMover::evict_v(const AttentionLayerView & layer,
                              int64_t                    pos,
                              void *                     dst) const {
    if (!valid_layer(layer) || dst == nullptr || pos < 0 || pos >= layer.kv_size) {
        return false;
    }

    if (!layer.v_trans) {
        const uint8_t * src = (const uint8_t *) layer.v->data + (size_t) pos * layer.v_row_bytes;
        hipError_t err = hipMemcpy(dst, src, layer.v_row_bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("mt::AttentionMover::evict_v(contig): hipMemcpy failed: %s\n",
                           hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // V transposed: gather column `pos` from [kv_size, n_embd_v] storage
    // into a contiguous buffer. Element/block size is v_row_bytes / ne[0].
    // We let ggml_element_size give us the per-element-or-block stride
    // — for quantized types this is the BLOCK size, which is correct
    // because the transposed V layout addresses blocks per row.
    const size_t elem_v   = ggml_element_size(layer.v);
    const size_t n_embd_v = (size_t) layer.v->ne[0];

    if (elem_v == 0 || n_embd_v == 0) {
        LLAMA_LOG_WARN("mt::AttentionMover::evict_v(trans): degenerate v shape elem=%zu ne0=%zu\n",
                       elem_v, n_embd_v);
        return false;
    }

    const uint8_t * src = (const uint8_t *) layer.v->data + (size_t) pos * elem_v;
    hipError_t err = hipMemcpy2D(dst,                                // dst (contiguous)
                                  elem_v,                             // dst pitch
                                  src,                                // src (column pos)
                                  (size_t) layer.kv_size * elem_v,    // src pitch
                                  elem_v,                             // width per row
                                  n_embd_v,                           // number of rows
                                  hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("mt::AttentionMover::evict_v(trans): hipMemcpy2D failed: %s\n",
                       hipGetErrorString(err));
        return false;
    }
    return true;
}

bool AttentionMover::restore_v(const AttentionLayerView & layer,
                                const void *               src,
                                int64_t                    pos) const {
    if (!valid_layer(layer) || src == nullptr || pos < 0 || pos >= layer.kv_size) {
        return false;
    }

    if (!layer.v_trans) {
        uint8_t * dst = (uint8_t *) layer.v->data + (size_t) pos * layer.v_row_bytes;
        hipError_t err = hipMemcpy(dst, src, layer.v_row_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("mt::AttentionMover::restore_v(contig): hipMemcpy failed: %s\n",
                           hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // Scatter: contiguous src -> column `pos` of transposed V.
    const size_t elem_v   = ggml_element_size(layer.v);
    const size_t n_embd_v = (size_t) layer.v->ne[0];

    if (elem_v == 0 || n_embd_v == 0) {
        return false;
    }

    uint8_t * dst = (uint8_t *) layer.v->data + (size_t) pos * elem_v;
    hipError_t err = hipMemcpy2D(dst,
                                  (size_t) layer.kv_size * elem_v,    // dst pitch
                                  src,                                // src (contiguous)
                                  elem_v,                             // src pitch
                                  elem_v,                             // width
                                  n_embd_v,                           // height
                                  hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("mt::AttentionMover::restore_v(trans): hipMemcpy2D failed: %s\n",
                       hipGetErrorString(err));
        return false;
    }
    return true;
}

#else  // !GGML_USE_HIP

// Stub: HIP not compiled in. The tier system is currently HIP-only.
// CUDA support is the same pattern with cuda* names; deferred.

bool AttentionMover::evict_k(const AttentionLayerView &, int64_t, void *) const            { return false; }
bool AttentionMover::restore_k(const AttentionLayerView &, const void *, int64_t) const    { return false; }
bool AttentionMover::evict_v(const AttentionLayerView &, int64_t, void *) const            { return false; }
bool AttentionMover::restore_v(const AttentionLayerView &, const void *, int64_t) const    { return false; }

#endif  // GGML_USE_HIP

}  // namespace mt
