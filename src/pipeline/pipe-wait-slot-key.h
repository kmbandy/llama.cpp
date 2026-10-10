#pragma once

// Pinned wait-slot key for WP_WAIT_ZEROCOPY / WP_WAIT_GPU_FLAG.
//
// Legacy contexts (group < 0, whole layer) keep the layer index as the key, so
// the legacy path is unchanged. Row-group contexts (WP_SPINE_ROW_GROUPS) get a
// per-(layer, group) key in the negative range, so A(l) and B(l) never share a
// slot and neither aliases a legacy layer key (layer >= 0).

#include <cstdint>

inline int32_t wait_slot_key(int32_t layer, int32_t group) {
    return group < 0 ? layer : -(layer * 2 + group) - 1;
}
