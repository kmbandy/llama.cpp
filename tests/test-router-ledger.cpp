// Tests for the router's pure VRAM/RAM accounting (tools/server/server-router-ledger.cpp).

#include "server-router-ledger.h"

#undef NDEBUG
#include <cassert>

static constexpr int64_t MB = 1024LL * 1024LL;

int main() {
    const std::string PD = "0000:42:00.0";

    // ledger_slot_id: unprefixed while local-only, "<machine>/<dev>" otherwise
    assert(ledger_slot_id("", "cuda0") == "cuda0");
    assert(ledger_slot_id("mad-lab-main", "cuda0") == "mad-lab-main/cuda0");

    // foreign sums only non-router PIDs on the matching pdev
    {
        const std::vector<proc_vram> usage = {
            { 100, PD,             3000 * MB },  // router child
            { 200, PD,             1000 * MB },  // foreign
            { 300, PD,              500 * MB },  // foreign
            { 400, "0000:03:00.0", 7000 * MB },  // other card
        };
        const std::set<int> router = { 100 };
        assert(ledger_foreign_vram(usage, router, PD) == 1500 * MB);
        // router child not double-counted: with every PID being a router child, foreign is 0
        assert(ledger_foreign_vram(usage, { 100, 200, 300 }, PD) == 0);
        // unknown pdev (NVML / unresolved): no per-PID view
        assert(ledger_foreign_vram(usage, router, "") == 0);
        assert(ledger_foreign_vram({}, router, PD) == 0);
    }

    // formula table: free = min(total - reserved - foreign, total - sysfs_used)
    {
        ledger_slot s = { "cuda0", PD, 16000 * MB, 4000 * MB };

        // nothing foreign, sysfs agrees with the ledger
        assert(ledger_free_vram(s, 0, 4000 * MB) == 12000 * MB);

        // foreign usage reduces free (sysfs also sees it: 4000 router + 2000 foreign)
        assert(ledger_free_vram(s, 2000 * MB, 6000 * MB) == 10000 * MB);

        // foreign usage reduces free even when sysfs reading is lower/stale
        assert(ledger_free_vram(s, 2000 * MB, 0) == 10000 * MB);

        // sysfs stricter than ledger wins (untracked usage: 9000 used vs 4000 reserved)
        assert(ledger_free_vram(s, 0, 9000 * MB) == 7000 * MB);

        // router child not double-counted: its 4000 MB shows up in sysfs_used AND in
        // router_reserved; free must be 12000, not 8000
        assert(ledger_free_vram(s, 0, 4000 * MB) == 12000 * MB);

        // sysfs unreadable: ledger term only
        assert(ledger_free_vram(s, 1000 * MB, -1) == 11000 * MB);

        // clamp at zero
        assert(ledger_free_vram(s, 20000 * MB, 0) == 0);
        assert(ledger_free_vram(s, 0, 20000 * MB) == 0);

        // negative foreign is treated as 0
        assert(ledger_free_vram(s, -5 * MB, 4000 * MB) == 12000 * MB);
    }

    // RAM: MemAvailable minus headroom
    {
        const int64_t headroom = 4096 * MB;
        assert(ledger_free_ram(40000 * MB, headroom) == 40000 * MB - headroom);
        assert(ledger_free_ram(1000 * MB, headroom) == 0);  // below headroom clamps
        assert(ledger_free_ram(-1, headroom) == -1);        // unreadable => unknown
    }

    // RAM fit refusal + headroom respected
    {
        const int64_t headroom = 4096 * MB;
        const int64_t avail    = 20000 * MB;
        const int64_t free_ram = ledger_free_ram(avail, headroom); // 15904 MB
        assert(ledger_ram_fits(free_ram, 15904 * MB));             // exactly fits
        assert(!ledger_ram_fits(free_ram, 15905 * MB));            // 1 MB into headroom: refused
        assert(!ledger_ram_fits(free_ram, 20000 * MB));            // would fit raw MemAvailable only
        assert(ledger_ram_fits(free_ram, 0));
        // a larger headroom shrinks what fits
        assert(!ledger_ram_fits(ledger_free_ram(avail, 8192 * MB), 15000 * MB));
        // unknown free RAM never blocks
        assert(ledger_ram_fits(-1, 999999 * MB));
    }

    return 0;
}
