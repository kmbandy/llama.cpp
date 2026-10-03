#pragma once

// Pure policy decisions for the router: which residents may be evicted, and whether a
// child's final environment is acceptable. No router state and no /proc access here, so
// both are unit-testable with literals (the one filesystem touch is the temp-dir check).

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct evict_resident {
    std::string name;
    int64_t     last_used = 0;
    bool        pinned    = false;
    int         req_count = 0; // in-flight requests; > 0 means busy
};

// Residents that may be evicted, least recently used first (ties by name). Pinned and busy
// residents are never returned: killing a child with in-flight requests drops them.
std::vector<evict_resident> evict_pick_lru(const std::vector<evict_resident> & residents);

struct evict_blocker {
    std::string name;
    bool        pinned = false; // true: pinned; false: busy
};

// For exclusive placement, where EVERY overlapping resident has to go: the first resident
// that cannot be evicted (pinned or busy), if any. The load must then be refused.
std::optional<evict_blocker> evict_find_blocker(const std::vector<evict_resident> & residents);

// Checks TEMP / TMP / TMPDIR of a final child environment ("KEY=VALUE" entries; the last
// definition of a key wins). Returns the offending entry ("TMPDIR=/etc/passwd") when a
// variable is set, non-empty, and does not name an existing directory; "" when fine.
// Unset and empty both count as unset.
std::string env_temp_dir_violation(const std::vector<std::string> & env);
