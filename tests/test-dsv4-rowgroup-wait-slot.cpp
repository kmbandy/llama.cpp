// WP_SPINE_ROW_GROUPS + WP_WAIT_ZEROCOPY: pinned wait-slot keys. CPU only.
//
// The dispatcher keys its pinned wait slots by wait_slot_key(layer, group).
// Checked: legacy (group -1) keys equal the layer index (legacy path unchanged),
// every (layer, group in {0,1}) key is distinct across all layers, no row-group
// key aliases a legacy layer key, and A(l) / B(l) never share a slot.

#include "pipe-wait-slot-key.h"

#include <cstdint>
#include <cstdio>
#include <set>

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char * what) {
        if (!ok) {
            std::fprintf(stderr, "FAIL: %s\n", what);
            ++failures;
        }
    };

    const int32_t n_layer = 256;

    bool legacy_identity = true;
    for (int32_t l = 0; l < n_layer; ++l) {
        legacy_identity = legacy_identity && wait_slot_key(l, -1) == l;
    }
    check(legacy_identity, "legacy group -1 key must equal the layer index");

    std::set<int32_t> legacy_keys;
    std::set<int32_t> group_keys;
    for (int32_t l = 0; l < n_layer; ++l) {
        legacy_keys.insert(wait_slot_key(l, -1));
        for (int32_t g = 0; g < 2; ++g) {
            group_keys.insert(wait_slot_key(l, g));
        }
    }
    check(group_keys.size() == (size_t) (2 * n_layer), "row-group keys must be unique per (layer, group)");

    bool disjoint = true;
    for (int32_t k : group_keys) {
        disjoint = disjoint && legacy_keys.count(k) == 0;
    }
    check(disjoint, "row-group keys must not alias legacy layer keys");

    bool a_ne_b = true;
    for (int32_t l = 0; l < n_layer; ++l) {
        a_ne_b = a_ne_b && wait_slot_key(l, 0) != wait_slot_key(l, 1);
    }
    check(a_ne_b, "A(l) and B(l) must use different slots");

    // Consecutive layers: A(l+1) and B(l) are both in flight in the offset schedule.
    bool a_next_ne_b = true;
    for (int32_t l = 0; l + 1 < n_layer; ++l) {
        a_next_ne_b = a_next_ne_b && wait_slot_key(l + 1, 0) != wait_slot_key(l, 1);
    }
    check(a_next_ne_b, "A(l+1) and B(l) must use different slots");

    if (failures == 0) {
        std::printf("test-dsv4-rowgroup-wait-slot: OK\n");
    }
    return failures == 0 ? 0 : 1;
}
