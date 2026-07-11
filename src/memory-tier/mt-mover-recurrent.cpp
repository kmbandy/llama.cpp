#include "mt-mover-recurrent.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstring>

namespace mt {

namespace {
inline bool valid_slice(const ggml_tensor * t, int slot, size_t size) {
    if (t == nullptr || slot < 0 || size == 0) return false;
    const size_t offset = (size_t) slot * t->nb[1];
    return offset <= ggml_nbytes(t) && size <= ggml_nbytes(t) - offset;
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
        if (!valid_slice(L.r, seq.seq_slot, L.r_bytes_per_seq)) return false;
        ggml_backend_tensor_get(L.r, dst, (size_t) seq.seq_slot * L.r->nb[1], L.r_bytes_per_seq);
        dst += L.r_bytes_per_seq;
    }

    // Pass 2: s tensors.
    for (size_t i = 0; i < seq.layers.size(); ++i) {
        const auto & L = seq.layers[i];
        if (L.s == nullptr || L.s_bytes_per_seq == 0) continue;
        if (!valid_slice(L.s, seq.seq_slot, L.s_bytes_per_seq)) return false;
        ggml_backend_tensor_get(L.s, dst, (size_t) seq.seq_slot * L.s->nb[1], L.s_bytes_per_seq);
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
        if (!valid_slice(L.r, seq.seq_slot, L.r_bytes_per_seq)) return false;
        ggml_backend_tensor_set(L.r, src, (size_t) seq.seq_slot * L.r->nb[1], L.r_bytes_per_seq);
        src += L.r_bytes_per_seq;
    }

    for (size_t i = 0; i < seq.layers.size(); ++i) {
        const auto & L = seq.layers[i];
        if (L.s == nullptr || L.s_bytes_per_seq == 0) continue;
        if (!valid_slice(L.s, seq.seq_slot, L.s_bytes_per_seq)) return false;
        ggml_backend_tensor_set(L.s, src, (size_t) seq.seq_slot * L.s->nb[1], L.s_bytes_per_seq);
        src += L.s_bytes_per_seq;
    }

    return true;
}

}  // namespace mt
