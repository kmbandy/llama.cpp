#pragma once

// Per-alias slot state files for KV-preserving evict/reload. Pure filesystem helpers:
// no model, no server state. Every filesystem error is logged and swallowed; these
// functions never throw to the caller.

#include <cstdint>
#include <string>
#include <vector>

// Path of one slot's file in an alias directory: <dir>/slot-<id>.wpslot.
// A negative slot_id has no valid file name and yields "".
std::string slot_store_file(const std::string & dir, int slot_id);

// Slot ids that have a regular file slot-<id>.wpslot in dir, ascending. Anything else
// (*.tmp, symlinks, directories, non-canonical ids, junk) is ignored. A missing dir, or a dir
// that is itself a symlink, yields {}.
std::vector<int> slot_store_list(const std::string & dir);

// Marks dir as recently used: creates dir if needed and (re)writes dir/.last_used,
// whose mtime is the LRU key. dir is lexically normalised before the symlink check, so
// spellings such as "<link>/." or "<root>/x/../<link>" cannot bypass it. Refuses (logs,
// writes nothing) when dir is a symlink or still has a ".." component after normalisation,
// so an alias dir can never redirect writes outside root.
void slot_store_touch(const std::string & dir);

// True when dir is usable as an alias dir for writes: it normalises (same rules as touch/list,
// so "<link>/." and trailing separators are judged on the real entry), is not "/" or empty, and
// is not a symlink. The save path refuses (logs, writes nothing) when this is false.
bool slot_store_alias_dir_ok(const std::string & dir);

// Makes dir exist as a plain directory before a save writes into it. Refuses (logs, creates
// nothing) when slot_store_alias_dir_ok(dir) is false, creates the directory and all missing
// parents, then re-runs the same check on the created dir in case a race planted a symlink.
// Returns true only when dir is a plain directory afterwards. Idempotent on an existing dir.
// Never throws.
bool slot_store_ensure_dir(const std::string & dir);

// Deletes slot-store dirs directly under root, least recently used first, until the bytes of
// slot-store dirs under root are <= budget_bytes. A slot-store dir is one in which EVERY entry is
// a regular file named .last_used, slot-<id>.wpslot or slot-<id>.wpslot.tmp, holding at least one
// such entry. Any other dir (a foreign file, subdir, symlink, other *.tmp, or empty) is foreign:
// never deleted and not counted (logged as skipped). An unmarked slot-store dir counts as oldest.
// Symlinks are never followed or deleted. keep_dir is never deleted. A root that is "/" or empty
// is refused. budget_bytes == 0 means unlimited (no-op). Returns the bytes freed.
uint64_t slot_store_prune(const std::string & root, const std::string & keep_dir, uint64_t budget_bytes);
