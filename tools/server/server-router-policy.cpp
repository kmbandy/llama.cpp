#include "server-router-policy.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

std::vector<evict_resident> evict_pick_lru(const std::vector<evict_resident> & residents, bool allow_busy) {
    std::vector<evict_resident> out;
    for (const auto & r : residents) {
        if (!r.pinned && !r.held && (allow_busy || r.req_count <= 0)) {
            out.push_back(r);
        }
    }
    // idle before busy, then least recently used, then by name
    std::sort(out.begin(), out.end(), [](const evict_resident & a, const evict_resident & b) {
        const bool a_busy = a.req_count > 0;
        const bool b_busy = b.req_count > 0;
        if (a_busy != b_busy) {
            return !a_busy;
        }
        return a.last_used != b.last_used ? a.last_used < b.last_used : a.name < b.name;
    });
    return out;
}

bool idle_unload_due(const idle_resident & r, int64_t now_ms) {
    if (r.stopping || r.pinned || r.held || r.req_count > 0) {
        return false;
    }
    if (r.last_used <= 0 || r.timeout_s <= 0) {
        return false;
    }
    return now_ms - r.last_used >= (int64_t) r.timeout_s * 1000;
}

std::optional<evict_blocker> evict_find_blocker(const std::vector<evict_resident> & residents, bool allow_busy) {
    for (const auto & r : residents) {
        if (r.pinned) {
            return evict_blocker{ r.name, true, false };
        }
        if (r.held) {
            return evict_blocker{ r.name, false, true };
        }
        if (!allow_busy && r.req_count > 0) {
            return evict_blocker{ r.name, false, false };
        }
    }
    return std::nullopt;
}

std::string env_temp_dir_violation(const std::vector<std::string> & env) {
    static const char * const keys[] = { "TEMP", "TMP", "TMPDIR" };
    for (const char * key : keys) {
        const std::string prefix = std::string(key) + "=";
        const std::string * last = nullptr;
        for (const auto & entry : env) {
            if (entry.compare(0, prefix.size(), prefix) == 0) {
                last = &entry;
            }
        }
        if (!last) {
            continue;
        }
        const std::string value = last->substr(prefix.size());
        if (value.empty()) {
            continue;
        }
        std::error_code ec;
        if (!fs::is_directory(fs::path(value), ec)) {
            return *last;
        }
    }
    return "";
}

router_child_wait router_child_wait_decide(int remaining, int remaining_online, bool shutting_down, bool timed_out) {
    if (remaining <= 0) {
        return ROUTER_CHILD_WAIT_EXITED;
    }
    if (shutting_down) {
        return ROUTER_CHILD_WAIT_SHUTDOWN;
    }
    if (remaining_online <= 0) {
        return ROUTER_CHILD_WAIT_OFFLINE;
    }
    return timed_out ? ROUTER_CHILD_WAIT_TIMEOUT : ROUTER_CHILD_WAIT_PENDING;
}

int64_t router_child_wait_bound_ms(int max_stop_timeout_s, bool has_group, int64_t kill_grace_ms, int64_t group_stop_ms) {
    return (int64_t) std::max(1, max_stop_timeout_s) * 1000 + kill_grace_ms + (has_group ? group_stop_ms : 0);
}

int64_t router_hold_ttl_clamp(int64_t ttl_s) {
    return std::min<int64_t>(std::max<int64_t>(ttl_s, 0), ROUTER_HOLD_TTL_MAX_S);
}
