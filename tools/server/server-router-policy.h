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
    bool        held      = false; // under a hold lease: never a victim, like pinned
};

// Residents that may be evicted, least recently used first (ties by name). Pinned and held
// residents are never returned. Busy ones (in-flight requests, which killing the child drops)
// are returned only with allow_busy (a `highest` admission), and then after every idle one.
std::vector<evict_resident> evict_pick_lru(const std::vector<evict_resident> & residents, bool allow_busy = false);

struct evict_blocker {
    std::string name;
    bool        pinned = false; // true: pinned
    bool        held   = false; // true: held (and not pinned); both false: busy
};

// For exclusive placement, where EVERY overlapping resident has to go: the first resident
// that cannot be evicted (pinned, held, or busy unless allow_busy), if any.
std::optional<evict_blocker> evict_find_blocker(const std::vector<evict_resident> & residents, bool allow_busy = false);

// What the idle sweeper knows about a resident that is up (loaded or sleeping).
struct idle_resident {
    bool    stopping  = false;
    bool    pinned    = false;
    bool    held      = false; // hold lease
    int     req_count = 0;
    int64_t last_used = 0;     // ms; <= 0 = never served a request
    int     timeout_s = 0;     // effective idle timeout; <= 0 = never idle-unload
};

// Whether the idle sweeper unloads it now: idle past its timeout, nothing in flight, and
// neither pinned nor held (a human / an orchestrator asked to keep it).
bool idle_unload_due(const idle_resident & r, int64_t now_ms);

// Checks TEMP / TMP / TMPDIR of a final child environment ("KEY=VALUE" entries; the last
// definition of a key wins). Returns the offending entry ("TMPDIR=/etc/passwd") when a
// variable is set, non-empty, and does not name an existing directory; "" when fine.
// Unset and empty both count as unset.
std::string env_temp_dir_violation(const std::vector<std::string> & env);
