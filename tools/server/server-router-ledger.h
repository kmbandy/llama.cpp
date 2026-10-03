#pragma once

// Pure accounting for the router's VRAM + host-RAM ledger. No I/O here: the
// caller (server-models.cpp) reads /proc and /sys through server-router-probe
// and feeds the numbers in, so everything below is unit-testable with literals.

#include "server-router-probe.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

struct ledger_slot {
    std::string id;              // "<machine>/<dev>" (just "<dev>" while local-only)
    std::string pdev;            // PCI address; "" = unknown / NVML (whole-card, no per-PID view)
    int64_t     total;           // bytes; probe/sysfs total unless overridden in the slot spec
    int64_t     router_reserved; // bytes reserved by the router's own children
};

// The one place slot ids are built, so a later phase can prefix them with a machine name.
std::string ledger_slot_id(const std::string & machine, const std::string & dev);

// Sum of VRAM on `pdev` held by PIDs that are NOT in `router_pids`. Router children are
// accounted through router_reserved and must never be counted a second time here.
// An empty pdev yields 0 (no per-PID view of that slot).
int64_t ledger_foreign_vram(const std::vector<proc_vram> & usage, const std::set<int> & router_pids, const std::string & pdev);

// Free VRAM for a slot: min(total - router_reserved - foreign, total - sysfs_used), clamped
// at 0. sysfs_used < 0 means the card-wide reading failed and only the ledger term applies.
int64_t ledger_free_vram(const ledger_slot & slot, int64_t foreign_bytes, int64_t sysfs_used);

// Free host RAM: MemAvailable minus headroom, clamped at 0. mem_available < 0 (unreadable)
// yields -1, which callers treat as "unknown, do not gate on RAM".
int64_t ledger_free_ram(int64_t mem_available, int64_t headroom);

// Whether a load needing `need` bytes fits in `free_ram`. Unknown free (< 0) always fits.
bool ledger_ram_fits(int64_t free_ram, int64_t need);
