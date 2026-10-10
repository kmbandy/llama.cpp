#include "server-slot-store.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

const char * const SLOT_PREFIX = "slot-";
const char * const SLOT_SUFFIX = ".wpslot";
const char * const LAST_USED   = ".last_used";

// slot-<id>.wpslot with a canonical decimal id (no sign, no leading zeros, at most 9 digits)
bool parse_slot_name(const std::string & name, int & id) {
    const std::string pre = SLOT_PREFIX;
    const std::string suf = SLOT_SUFFIX;
    if (name.size() <= pre.size() + suf.size()) {
        return false;
    }
    if (name.compare(0, pre.size(), pre) != 0) {
        return false;
    }
    if (name.compare(name.size() - suf.size(), suf.size(), suf) != 0) {
        return false;
    }
    const std::string digits = name.substr(pre.size(), name.size() - pre.size() - suf.size());
    if (digits.size() > 9) {
        return false;
    }
    for (const char c : digits) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    const int v = std::stoi(digits);
    if (std::to_string(v) != digits) {
        return false;
    }
    id = v;
    return true;
}

// Regular file, judged without following a symlink.
bool is_regular_nofollow(const fs::path & p) {
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(p, ec);
    return !ec && fs::is_regular_file(st);
}

// True for a filesystem root ("/") or an empty path. A lexically normal path whose parent is
// itself is a root; a root must never be used as an alias dir or as a prune root.
bool is_fs_root(const fs::path & p) {
    fs::path q = p.lexically_normal();
    while (q.filename().empty() && q.parent_path() != q) {
        q = q.parent_path();
    }
    return q.empty() || q.parent_path() == q;
}

// Normalises a caller-supplied alias dir before any symlink check or use. lexically_normal()
// folds "." components ("a/link/." -> "a/link/"), then trailing separators are stripped so
// symlink_status() judges the entry itself: symlink_status("link/") resolves the link and
// would report a directory. Returns false (caller refuses) when the result still has a ".."
// component (e.g. "../x"), or is empty or "." (no alias name), since those could escape root.
bool normalise_alias_dir(const fs::path & in, fs::path & out) {
    fs::path p = in.lexically_normal();
    for (const fs::path & part : p) {
        if (part == "..") {
            return false;
        }
    }
    while (p.filename().empty() && p.parent_path() != p) {
        p = p.parent_path();
    }
    if (p.empty() || p.filename() == "." || is_fs_root(p)) {
        return false;
    }
    out = p;
    return true;
}

// Symlink itself (not its target), judged without following it. Callers pass a path that
// normalise_alias_dir() has already processed.
bool is_symlink_nofollow(const fs::path & p) {
    std::error_code ec;
    return fs::is_symlink(fs::symlink_status(p, ec));
}

// Real directory, judged without following a symlink.
bool is_dir_nofollow(const fs::path & p) {
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(p, ec);
    return !ec && fs::is_directory(st);
}

// Bytes of regular files under dir. Symlinks are neither followed nor counted.
uint64_t tree_bytes(const fs::path & dir) {
    uint64_t total = 0;
    std::error_code ec;
    fs::recursive_directory_iterator it(dir, fs::directory_options::none, ec);
    if (ec) {
        return 0;
    }
    const fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            LOG_WRN("slot-store: walk of %s stopped: %s\n", dir.string().c_str(), ec.message().c_str());
            break;
        }
        std::error_code sec;
        const fs::file_status st = it->symlink_status(sec);
        if (sec || !fs::is_regular_file(st)) {
            continue;
        }
        std::error_code fec;
        const uintmax_t sz = fs::file_size(it->path(), fec);
        if (!fec) {
            total += sz;
        }
    }
    return total;
}

struct alias_dir {
    fs::path path;
    bool has_mark = false;
    fs::file_time_type last_used{};
    uint64_t bytes = 0;
};

// Oldest first: unmarked dirs, then by .last_used mtime, ties broken by name.
bool older_first(const alias_dir & a, const alias_dir & b) {
    if (a.has_mark != b.has_mark) {
        return !a.has_mark;
    }
    if (a.has_mark && a.last_used != b.last_used) {
        return a.last_used < b.last_used;
    }
    return a.path.filename() < b.path.filename();
}

// slot-<id>.wpslot.tmp: the temp name a slot save writes before its rename.
bool is_slot_tmp_name(const std::string & name) {
    const std::string tmp = ".tmp";
    if (name.size() <= tmp.size() || name.compare(name.size() - tmp.size(), tmp.size(), tmp) != 0) {
        return false;
    }
    int id = 0;
    return parse_slot_name(name.substr(0, name.size() - tmp.size()), id);
}

// A slot store is a dir in which EVERY entry is a regular file named .last_used,
// slot-<id>.wpslot or slot-<id>.wpslot.tmp, and which holds at least one such entry.
// A dir with any other entry (foreign file, subdir, symlink, other *.tmp) is foreign: it is
// never deleted and never counted. Ruling 8.
bool is_slot_store_dir(const fs::path & p, bool & marked, fs::file_time_type & last_used) {
    marked = false;
    std::error_code ec;
    fs::directory_iterator it(p, ec);
    if (ec) {
        return false;
    }
    size_t members = 0;
    const fs::directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            return false;
        }
        const fs::path e = it->path();
        if (!is_regular_nofollow(e)) {
            return false;
        }
        const std::string name = e.filename().string();
        if (name == LAST_USED) {
            std::error_code mec;
            const fs::file_time_type t = fs::last_write_time(e, mec);
            if (mec) {
                return false;
            }
            marked = true;
            last_used = t;
        } else {
            int id = 0;
            if (!parse_slot_name(name, id) && !is_slot_tmp_name(name)) {
                return false;
            }
        }
        members++;
    }
    return members > 0;
}

fs::path canon(const fs::path & p) {
    std::error_code ec;
    fs::path c = fs::weakly_canonical(p, ec);
    return ec ? p.lexically_normal() : c;
}

} // namespace

std::string slot_store_file(const std::string & dir, int slot_id) {
    if (slot_id < 0) {
        return "";
    }
    return (fs::path(dir) / (std::string(SLOT_PREFIX) + std::to_string(slot_id) + SLOT_SUFFIX)).string();
}

std::vector<int> slot_store_list(const std::string & dir) {
    std::vector<int> ids;
    fs::path d;
    if (!normalise_alias_dir(fs::path(dir), d)) {
        LOG_WRN("slot-store: refusing to list %s: alias dir is not a plain name under root\n", dir.c_str());
        return ids;
    }
    if (is_symlink_nofollow(d)) {
        LOG_WRN("slot-store: refusing to list %s: alias dir is a symlink\n", dir.c_str());
        return ids;
    }
    try {
        std::error_code ec;
        fs::directory_iterator it(d, ec);
        const fs::directory_iterator end;
        if (ec) {
            if (fs::exists(d, ec)) {
                LOG_WRN("slot-store: cannot list %s: %s\n", dir.c_str(), ec.message().c_str());
            }
            return ids;
        }
        for (; it != end; it.increment(ec)) {
            if (ec) {
                LOG_WRN("slot-store: listing %s stopped: %s\n", dir.c_str(), ec.message().c_str());
                break;
            }
            int id = 0;
            if (parse_slot_name(it->path().filename().string(), id) && is_regular_nofollow(it->path())) {
                ids.push_back(id);
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("slot-store: listing %s failed: %s\n", dir.c_str(), e.what());
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

void slot_store_touch(const std::string & dir) {
    fs::path d;
    if (!normalise_alias_dir(fs::path(dir), d)) {
        LOG_WRN("slot-store: refusing to touch %s: alias dir is not a plain name under root\n", dir.c_str());
        return;
    }
    // an alias dir that is a symlink could point outside root: refuse before any write
    if (is_symlink_nofollow(d)) {
        LOG_WRN("slot-store: refusing to touch %s: alias dir is a symlink\n", dir.c_str());
        return;
    }
    try {
        std::error_code ec;
        fs::create_directories(d, ec);
        if (ec) {
            LOG_WRN("slot-store: cannot create %s: %s\n", dir.c_str(), ec.message().c_str());
            return;
        }

        const fs::path marker = d / LAST_USED;
        // never write through a symlink planted in the alias dir
        if (fs::is_symlink(fs::symlink_status(marker, ec))) {
            fs::remove(marker, ec);
        }

        {
            std::ofstream f(marker);
            if (!f) {
                LOG_WRN("slot-store: cannot write %s\n", marker.string().c_str());
                return;
            }
            const auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            f << secs << "\n";
        }

        std::error_code tec;
        fs::last_write_time(marker, fs::file_time_type::clock::now(), tec);
        if (tec) {
            LOG_WRN("slot-store: cannot stamp %s: %s\n", marker.string().c_str(), tec.message().c_str());
        }
    } catch (const std::exception & e) {
        LOG_WRN("slot-store: touch of %s failed: %s\n", dir.c_str(), e.what());
    }
}

bool slot_store_alias_dir_ok(const std::string & dir) {
    fs::path d;
    if (!normalise_alias_dir(fs::path(dir), d)) {
        return false;
    }
    return !is_symlink_nofollow(d);
}

uint64_t slot_store_prune(const std::string & root, const std::string & keep_dir, uint64_t budget_bytes) {
    if (budget_bytes == 0) {
        return 0;
    }
    // a prune root of "/" (or empty) would scan the whole filesystem: refuse
    if (is_fs_root(fs::path(root))) {
        LOG_WRN("slot-store: refusing to prune root '%s': not a usable directory\n", root.c_str());
        return 0;
    }

    uint64_t freed = 0;
    try {
        const fs::path r(root);
        std::error_code ec;
        if (!fs::is_directory(r, ec)) {
            return 0;
        }

        // Only slot-store dirs are candidates; foreign dirs are never deleted and never
        // counted, so the budget applies to slot-store bytes only.
        std::vector<alias_dir> dirs;
        fs::directory_iterator it(r, ec);
        const fs::directory_iterator end;
        if (ec) {
            LOG_WRN("slot-store: cannot list %s: %s\n", root.c_str(), ec.message().c_str());
            return 0;
        }
        for (; it != end; it.increment(ec)) {
            if (ec) {
                LOG_WRN("slot-store: listing %s stopped: %s\n", root.c_str(), ec.message().c_str());
                break;
            }
            const fs::path p = it->path();
            if (!is_dir_nofollow(p)) {
                continue;
            }

            alias_dir a;
            a.path = p;
            if (!is_slot_store_dir(p, a.has_mark, a.last_used)) {
                LOG_WRN("slot-store: skipping %s: not a slot store\n", p.string().c_str());
                continue;
            }
            a.bytes = tree_bytes(p);
            dirs.push_back(std::move(a));
        }

        uint64_t total = 0;
        for (const alias_dir & a : dirs) {
            total += a.bytes;
        }
        if (total <= budget_bytes) {
            return 0;
        }

        std::sort(dirs.begin(), dirs.end(), older_first);

        const bool has_keep = !keep_dir.empty();
        const fs::path keep = has_keep ? canon(fs::path(keep_dir)) : fs::path();

        for (const alias_dir & a : dirs) {
            if (total <= budget_bytes) {
                break;
            }
            if (has_keep && canon(a.path) == keep) {
                continue;
            }

            std::error_code rec;
            fs::remove_all(a.path, rec);
            if (rec) {
                LOG_WRN("slot-store: cannot remove %s: %s\n", a.path.string().c_str(), rec.message().c_str());
                continue;
            }

            total = a.bytes >= total ? 0 : total - a.bytes;
            freed += a.bytes;
            LOG_INF("slot-store: pruned %s (%llu bytes)\n",
                    a.path.string().c_str(), (unsigned long long) a.bytes);
        }
    } catch (const std::exception & e) {
        LOG_WRN("slot-store: prune of %s failed: %s\n", root.c_str(), e.what());
    }
    return freed;
}
