// Tests for the per-alias slot store helper (tools/server/server-slot-store.cpp).

#include "server-slot-store.h"

#undef NDEBUG
#include <cassert>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void write_bytes(const fs::path & p, size_t n) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    const std::string data(n, 'x');
    f.write(data.data(), (std::streamsize) n);
}

// Creates <dir>/.last_used (empty) and sets its mtime, so the LRU order is controlled by the test.
static void mark_used(const fs::path & dir, fs::file_time_type t) {
    fs::create_directories(dir);
    const fs::path m = dir / ".last_used";
    { std::ofstream f(m); }
    fs::last_write_time(m, t);
}

int main() {
    const fs::path root_base = fs::temp_directory_path() /
        ("slot-store-test-" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::remove_all(root_base, ec);
    fs::create_directories(root_base);
    const fs::path root = root_base / "root";
    fs::create_directories(root);

    const fs::file_time_type base = fs::file_time_type::clock::now();

    // file naming
    assert(slot_store_file("/x/alias", 3)  == (fs::path("/x/alias") / "slot-3.wpslot").string());
    assert(slot_store_file("/x/alias", 12) == (fs::path("/x/alias") / "slot-12.wpslot").string());

    // listing: ascending ids; .tmp files, junk, non-canonical names, directories and
    // symlinks with a matching name are all ignored; a missing dir yields nothing
    {
        const fs::path d = root / "list";
        write_bytes(d / "slot-10.wpslot", 4);
        write_bytes(d / "slot-0.wpslot", 4);
        write_bytes(d / "slot-2.wpslot", 4);
        write_bytes(d / "slot-3.wpslot.tmp", 4);
        write_bytes(d / "slot-4.tmp", 4);
        write_bytes(d / "slot-5.wpslot.partial", 4);
        write_bytes(d / "slot-07.wpslot", 4);
        write_bytes(d / "slotx.wpslot", 4);
        write_bytes(d / "junk.txt", 4);
        write_bytes(d / ".last_used", 0);
        fs::create_directories(d / "slot-6.wpslot");
        fs::create_symlink(d / "slot-0.wpslot", d / "slot-8.wpslot");

        const std::vector<int> ids = slot_store_list(d.string());
        assert((ids == std::vector<int>{ 0, 2, 10 }));

        assert(slot_store_list((root / "no-such-dir").string()).empty());
    }

    // touch: creates the dir and a regular .last_used, and re-stamps it with a newer mtime
    {
        const fs::path d = root / "touch" / "alias";
        slot_store_touch(d.string());
        assert(fs::is_regular_file(d / ".last_used"));

        fs::last_write_time(d / ".last_used", base - std::chrono::hours(24));
        slot_store_touch(d.string());
        assert(fs::last_write_time(d / ".last_used") > base - std::chrono::hours(1));
    }

    // prune order follows .last_used mtime, not name; a dir without .last_used is oldest.
    // Four 1000-byte dirs, budget 2500: evict "b" (unmarked), then "z" (oldest mark). Stop at 2000.
    {
        const fs::path r = root / "order";
        write_bytes(r / "z" / "slot-0.wpslot", 1000);
        mark_used(r / "z", base + std::chrono::seconds(10));
        write_bytes(r / "a" / "slot-0.wpslot", 1000);
        mark_used(r / "a", base + std::chrono::seconds(20));
        write_bytes(r / "m" / "slot-0.wpslot", 1000);
        mark_used(r / "m", base + std::chrono::seconds(30));
        write_bytes(r / "b" / "slot-0.wpslot", 1000);

        const uint64_t freed = slot_store_prune(r.string(), (r / "none").string(), 2500);
        assert(freed == 2000);
        assert(!fs::exists(r / "b"));
        assert(!fs::exists(r / "z"));
        assert(fs::exists(r / "a" / "slot-0.wpslot"));
        assert(fs::exists(r / "m" / "slot-0.wpslot"));
    }

    // keep_dir is never deleted, even when it is the oldest and is the only thing over budget
    {
        const fs::path r = root / "keep";
        write_bytes(r / "cur" / "slot-0.wpslot", 1000);
        mark_used(r / "cur", base + std::chrono::seconds(10));
        write_bytes(r / "old" / "slot-0.wpslot", 1000);
        mark_used(r / "old", base + std::chrono::seconds(50));

        const uint64_t freed = slot_store_prune(r.string(), (r / "cur").string(), 500);
        assert(freed == 1000);
        assert(!fs::exists(r / "old"));
        assert(fs::exists(r / "cur" / "slot-0.wpslot"));
    }
    {
        const fs::path r = root / "keep-alone";
        write_bytes(r / "cur" / "slot-0.wpslot", 1000);
        mark_used(r / "cur", base);

        const uint64_t freed = slot_store_prune(r.string(), (r / "cur").string(), 500);
        assert(freed == 0);
        assert(fs::exists(r / "cur" / "slot-0.wpslot"));
    }

    // budget 0 is unlimited: nothing is deleted
    {
        const fs::path r = root / "zero";
        write_bytes(r / "a" / "slot-0.wpslot", 1000);
        mark_used(r / "a", base);

        assert(slot_store_prune(r.string(), (r / "none").string(), 0) == 0);
        assert(fs::exists(r / "a" / "slot-0.wpslot"));
    }

    // symlinks: a symlink to a dir outside root is neither followed nor deleted, and its
    // target's bytes are not counted. Only the 1000-byte alias dir is freed.
    {
        const fs::path outside = root_base / "outside";
        write_bytes(outside / "victim.bin", 5000);

        const fs::path r = root / "sym";
        write_bytes(r / "s" / "slot-0.wpslot", 1000);
        fs::create_directory_symlink(outside, r / "s" / "link");
        fs::create_directory_symlink(outside, r / "evil");

        const uint64_t freed = slot_store_prune(r.string(), (r / "none").string(), 1);
        assert(freed == 1000);
        assert(!fs::exists(r / "s"));
        assert(fs::is_symlink(r / "evil"));
        assert(fs::exists(outside / "victim.bin"));
    }

    // symlinked alias dir: touch refuses and writes nothing outside root; list returns {}
    {
        const fs::path outside = root_base / "outside-alias";
        fs::create_directories(outside);
        write_bytes(outside / "slot-0.wpslot", 10);

        const fs::path r = root / "alias-link";
        fs::create_directories(root);
        fs::create_directory_symlink(outside, r);

        slot_store_touch(r.string());
        assert(!fs::exists(outside / ".last_used"));
        assert(fs::is_symlink(r));
        assert(slot_store_list(r.string()).empty());
    }

    // negative slot ids have no file name
    assert(slot_store_file("/x/alias", -1).empty());

    fs::remove_all(root_base, ec);
    return 0;
}
