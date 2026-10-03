#include "server-router-policy.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

std::vector<evict_resident> evict_pick_lru(const std::vector<evict_resident> & residents) {
    std::vector<evict_resident> out;
    for (const auto & r : residents) {
        if (!r.pinned && r.req_count <= 0) {
            out.push_back(r);
        }
    }
    std::sort(out.begin(), out.end(), [](const evict_resident & a, const evict_resident & b) {
        return a.last_used != b.last_used ? a.last_used < b.last_used : a.name < b.name;
    });
    return out;
}

std::optional<evict_blocker> evict_find_blocker(const std::vector<evict_resident> & residents) {
    for (const auto & r : residents) {
        if (r.pinned) {
            return evict_blocker{ r.name, true };
        }
        if (r.req_count > 0) {
            return evict_blocker{ r.name, false };
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
