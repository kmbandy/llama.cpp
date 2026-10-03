// Tests for the router's pure eviction/env policy (tools/server/server-router-policy.cpp).

#include "server-router-policy.h"

#undef NDEBUG
#include <cassert>
#include <filesystem>
#include <fstream>

static bool has(const std::vector<evict_resident> & v, const std::string & name) {
    for (const auto & r : v) {
        if (r.name == name) {
            return true;
        }
    }
    return false;
}

int main() {
    // busy model is never a victim; the idle one is
    {
        const std::vector<evict_resident> residents = {
            { "busy", 1, false, 2 },
            { "idle", 5, false, 0 },
        };
        const auto picked = evict_pick_lru(residents);
        assert(picked.size() == 1);
        assert(picked[0].name == "idle");
        assert(!has(picked, "busy"));
    }
    // pinned is excluded too; LRU order among the rest
    {
        const std::vector<evict_resident> residents = {
            { "newer",  9, false, 0 },
            { "pinned", 1, true,  0 },
            { "older",  3, false, 0 },
            { "busy",   2, false, 1 },
        };
        const auto picked = evict_pick_lru(residents);
        assert(picked.size() == 2);
        assert(picked[0].name == "older");
        assert(picked[1].name == "newer");
    }
    assert(evict_pick_lru({}).empty());

    // exclusive placement: a busy overlapping resident blocks the load, as does a pinned one
    {
        assert(!evict_find_blocker({ { "idle", 1, false, 0 } }).has_value());
        const auto busy = evict_find_blocker({ { "idle", 1, false, 0 }, { "busy", 2, false, 3 } });
        assert(busy.has_value() && busy->name == "busy" && !busy->pinned);
        const auto pinned = evict_find_blocker({ { "p", 1, true, 0 } });
        assert(pinned.has_value() && pinned->name == "p" && pinned->pinned);
    }

    // held is never a victim; allow_busy (a `highest` admission) returns busy after idle
    {
        const std::vector<evict_resident> residents = {
            { "busy-old", 1, false, 2, false },
            { "held",     2, false, 0, true  },
            { "idle-new", 8, false, 0, false },
            { "idle-old", 4, false, 0, false },
        };
        const auto idle_only = evict_pick_lru(residents);
        assert(idle_only.size() == 2 && idle_only[0].name == "idle-old" && idle_only[1].name == "idle-new");
        const auto with_busy = evict_pick_lru(residents, /*allow_busy=*/true);
        assert(with_busy.size() == 3);
        assert(with_busy[0].name == "idle-old" && with_busy[1].name == "idle-new" && with_busy[2].name == "busy-old");
        assert(!has(with_busy, "held"));

        const auto held = evict_find_blocker({ { "h", 1, false, 0, true } }, true);
        assert(held.has_value() && held->name == "h" && held->held && !held->pinned);
        assert(!evict_find_blocker({ { "busy", 2, false, 3, false } }, true).has_value());
    }

    // temp-dir env scrub
    {
        namespace fs = std::filesystem;
        const std::string dir = fs::temp_directory_path().string();
        assert(fs::is_directory(dir));

        // rejects a regular file (/etc/passwd) and a missing path, naming the variable
        assert(env_temp_dir_violation({ "TMPDIR=/etc/passwd" }) == "TMPDIR=/etc/passwd");
        assert(env_temp_dir_violation({ "TEMP=/nonexistent/router-test-dir" }) == "TEMP=/nonexistent/router-test-dir");
        assert(env_temp_dir_violation({ "TMP=/etc/passwd" }) == "TMP=/etc/passwd");

        // accepts a real directory
        assert(env_temp_dir_violation({ "TMPDIR=" + dir }).empty());
        assert(env_temp_dir_violation({ "TEMP=" + dir, "TMP=" + dir, "TMPDIR=" + dir }).empty());

        // accepts unset, empty, and unrelated variables
        assert(env_temp_dir_violation({}).empty());
        assert(env_temp_dir_violation({ "PATH=/usr/bin", "TMPDIRX=/etc/passwd" }).empty());
        assert(env_temp_dir_violation({ "TMPDIR=" }).empty());

        // last definition of a key wins
        assert(env_temp_dir_violation({ "TMPDIR=/etc/passwd", "TMPDIR=" + dir }).empty());
        assert(env_temp_dir_violation({ "TMPDIR=" + dir, "TMPDIR=/etc/passwd" }) == "TMPDIR=/etc/passwd");
    }

    return 0;
}
