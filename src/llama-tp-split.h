#pragma once

// The row-split arithmetic behind LLAMA_SPLIT_MODE_TENSOR, factored out of
// llama_meta_device_get_split_state so it can be unit-tested without a model, a GPU or a network.
//
// This is the function that decides, for one segment of one tensor, how many rows along the split
// axis each device in the WORLD receives. It is deliberately world-only: it never looks at which
// devices are local, so every rank of a cross-host tensor-parallel world computes an identical
// global row map from the same tensor_split, and each rank then keeps the window it owns.

#include <cstddef>
#include <cstdint>
#include <vector>

// Distribute `ne_s` rows over `n_devices_eff` of `n_devices` devices.
//
//   ne_s          rows in this segment
//   granularity   every device's row count is a multiple of it, except the last, which takes the
//                 remainder (so the segment always adds up exactly)
//   tensor_split  per-device proportions, length >= n_devices. nullptr, or all zero, means "even"
//   n_devices     size of the world; also the stride of the ne array this writes into
//   n_devices_eff number of leading devices allowed to receive rows (<= n_devices). Devices at or
//                 beyond it get zero. Used to keep the LM head on rank 0 only.
//   rotation      which effective device the distribution starts at, so that consecutive layers
//                 hand the rounding remainder to different devices
//   ne_out        length n_devices, fully overwritten
inline void llama_tp_split_segment(
        int64_t      ne_s,
        int64_t      granularity,
        const float * tensor_split,
        size_t       n_devices,
        size_t       n_devices_eff,
        size_t       rotation,
        int64_t *    ne_out) {
    for (size_t j = 0; j < n_devices; j++) {
        ne_out[j] = 0;
    }
    if (n_devices_eff == 0 || n_devices_eff > n_devices) {
        return;
    }

    // cumulative scan of the proportions, in rotated device order
    std::vector<float> scan;
    scan.reserve(n_devices_eff);
    for (size_t j = 0; j < n_devices_eff; j++) {
        scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + rotation) % n_devices_eff]);
        if (j > 0) {
            scan[j] += scan[j - 1];
        }
    }

    int64_t low = 0;
    size_t  j   = 0;
    for (; j + 1 < n_devices_eff; j++) {
        int64_t high = scan.back() == 0.0f ?
            ne_s * (int64_t) (j + 1) / (int64_t) n_devices_eff :
            (int64_t) (ne_s * scan[j] / scan.back());
        if (granularity != 0 && high % granularity != 0) {
            high -= high % granularity;
        }
        ne_out[(j + rotation) % n_devices_eff] = high - low;
        low = high;
    }
    ne_out[(j + rotation) % n_devices_eff] = ne_s - low;
}

// Half-open row range [first, last) that world device `jw` owns in a segment laid out by
// llama_tp_split_segment. Rows are laid out in DEVICE-INDEX order (device 0's rows first), which is
// how ggml_backend_meta_buffer_set_tensor walks the source buffer.
inline void llama_tp_split_row_range(
        const int64_t * ne, size_t n_devices, size_t jw, int64_t * first, int64_t * last) {
    int64_t off = 0;
    for (size_t j = 0; j < jw && j < n_devices; j++) {
        off += ne[j];
    }
    *first = off;
    *last  = off + (jw < n_devices ? ne[jw] : 0);
}
