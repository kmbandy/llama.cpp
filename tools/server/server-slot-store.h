#pragma once

// Per-alias slot state files for KV-preserving evict/reload. Pure filesystem helpers:
// no model, no server state. Every filesystem error is logged and swallowed; these
// functions never throw to the caller.

#include <cstdint>
#include <string>
#include <vector>

// Path of one slot's file in an alias directory: <dir>/slot-<id>.wpslot
std::string slot_store_file(const std::string & dir, int slot_id);

// Slot ids that have a regular file slot-<id>.wpslot in dir, ascending. Anything else
// (*.tmp, symlinks, directories, non-canonical ids, junk) is ignored. A missing dir yields {}.
std::vector<int> slot_store_list(const std::string & dir);

// Marks dir as recently used: creates dir if needed and (re)writes dir/.last_used,
// whose mtime is the LRU key.
void slot_store_touch(const std::string & dir);

// Deletes whole alias directories directly under root, least recently used first, until
// the bytes of regular files under root are <= budget_bytes. A directory without .last_used
// counts as oldest. Symlinks are never followed or deleted. keep_dir is never deleted.
// budget_bytes == 0 means unlimited (no-op). Returns the bytes freed.
uint64_t slot_store_prune(const std::string & root, const std::string & keep_dir, uint64_t budget_bytes);
