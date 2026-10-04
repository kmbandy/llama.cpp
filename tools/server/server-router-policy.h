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

//
// bounded waits for a child to leave "running"
//

enum router_child_wait {
    ROUTER_CHILD_WAIT_PENDING,  // keep waiting
    ROUTER_CHILD_WAIT_EXITED,   // every child is down
    ROUTER_CHILD_WAIT_OFFLINE,  // what is left runs on machines whose node is offline
    ROUTER_CHILD_WAIT_SHUTDOWN, // the router is shutting down (only when the caller asked to honour it)
    ROUTER_CHILD_WAIT_TIMEOUT,  // the bound passed with online children still up
};

// One wait step. `remaining` children are still running, `remaining_online` of them on a machine
// that is online (the local one always is). Order: all exited wins, then shutdown (when honoured),
// then "only offline machines are left" (nothing will ever exit it), then the deadline.
router_child_wait router_child_wait_decide(int remaining, int remaining_online, bool shutting_down, bool timed_out);

// The bound of such a wait, in ms: the longest stop-timeout among the children plus the SIGKILL
// grace; a group adds its workers' quiesce + grace (what a stop of the group takes).
int64_t router_child_wait_bound_ms(int max_stop_timeout_s, bool has_group, int64_t kill_grace_ms, int64_t group_stop_ms);

// A pool replica is not addressable by name: it answers like an unknown model.
inline bool router_name_addressable(const std::string & replica_of) {
    return replica_of.empty();
}

// A hold's lease length: negative becomes 0, and the cap keeps `ttl_s * 1000` from overflowing.
static constexpr int64_t ROUTER_HOLD_TTL_MAX_S = 7 * 24 * 3600;
int64_t router_hold_ttl_clamp(int64_t ttl_s);
