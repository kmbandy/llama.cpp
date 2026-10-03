#include "server-router-ledger.h"

#include <algorithm>

std::string ledger_slot_id(const std::string & machine, const std::string & dev) {
    return machine.empty() ? dev : machine + "/" + dev;
}

int64_t ledger_foreign_vram(const std::vector<proc_vram> & usage, const std::set<int> & router_pids, const std::string & pdev) {
    if (pdev.empty()) {
        return 0;
    }
    int64_t sum = 0;
    for (const auto & u : usage) {
        if (u.pdev == pdev && router_pids.find(u.pid) == router_pids.end() && u.vram_bytes > 0) {
            sum += u.vram_bytes;
        }
    }
    return sum;
}

int64_t ledger_free_vram(const ledger_slot & slot, int64_t foreign_bytes, int64_t sysfs_used) {
    int64_t free = slot.total - slot.router_reserved - std::max<int64_t>(0, foreign_bytes);
    if (sysfs_used >= 0) {
        free = std::min(free, slot.total - sysfs_used);
    }
    return std::max<int64_t>(0, free);
}

int64_t ledger_free_ram(int64_t mem_available, int64_t headroom) {
    if (mem_available < 0) {
        return -1;
    }
    return std::max<int64_t>(0, mem_available - std::max<int64_t>(0, headroom));
}

bool ledger_ram_fits(int64_t free_ram, int64_t need) {
    return free_ram < 0 || need <= free_ram;
}
