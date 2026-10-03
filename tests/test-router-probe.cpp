// Tests for the router's /proc and /sys probes (tools/server/server-router-probe.cpp),
// run against the fixture tree tests/router-fixtures/proc-sys-basic.
// Run from the repo root (WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}).

#include "server-router-probe.h"

#include <filesystem>
#include <fstream>
#include <string>

#undef NDEBUG
#include <cassert>

namespace fs = std::filesystem;

static const std::string ROOT = "tests/router-fixtures/proc-sys-basic";

static int64_t find_vram(const std::vector<proc_vram> & v, int pid, const std::string & pdev) {
    for (const auto & e : v) {
        if (e.pid == pid && e.pdev == pdev) {
            return e.vram_bytes;
        }
    }
    return -1;
}

int main() {
    // fdinfo: pid 100 has fds 3 and 4 sharing client 1 (1024 KiB, must count once)
    // plus fd 5 client 2 (2 MiB) on 0000:42:00.0 => 1 MiB + 2 MiB = 3145728.
    // pid 200 has client 7 (512 KiB) and client 8 (bare 4096 bytes) on 0000:03:00.0,
    // plus a non-DRM fd that must be ignored. pid 300 has no fdinfo (vanished).
    {
        const auto v = probe_fdinfo_vram(ROOT);
        assert(v.size() == 2);
        assert(find_vram(v, 100, "0000:42:00.0") == 3145728);
        assert(find_vram(v, 200, "0000:03:00.0") == 524288 + 4096);
        assert(find_vram(v, 300, "0000:42:00.0") == -1);
        // sorted by (pid, pdev)
        assert(v[0].pid == 100 && v[1].pid == 200);
    }

    // a root with no /proc at all yields an empty result, not an error
    {
        assert(probe_fdinfo_vram(ROOT + "/does-not-exist").empty());
    }

    // proc status: RssAnon 123456 kB, RssShmem 789 kB
    {
        const auto m = probe_proc_mem(ROOT, 100);
        assert(m.has_value());
        assert(m->rss_anon == 123456LL * 1024);
        assert(m->rss_shmem == 789LL * 1024);
        assert(!probe_proc_mem(ROOT, 300).has_value()); // vanished: no status
        assert(!probe_proc_mem(ROOT, 4242).has_value());
    }

    // meminfo
    {
        assert(probe_mem_available(ROOT) == 40000000LL * 1024);
        assert(probe_mem_available(ROOT + "/does-not-exist") == -1);
    }

    // sysfs vram used
    {
        assert(probe_sysfs_vram_used(ROOT, "0000:42:00.0") == 31234560LL);
        assert(probe_sysfs_vram_used(ROOT, "0000:ff:00.0") == -1);
    }

    // pdev_for_probe: <tmp>/class/drm/card1/device -> symlink to <tmp>/pci/0000:42:00.0
    {
        const fs::path tmp = fs::temp_directory_path() / "test-router-probe-pdev";
        std::error_code ec;
        fs::remove_all(tmp, ec);
        fs::create_directories(tmp / "pci" / "0000:42:00.0");
        fs::create_directories(tmp / "card1");
        fs::create_directory_symlink(tmp / "pci" / "0000:42:00.0", tmp / "card1" / "device");
        { std::ofstream((tmp / "pci" / "0000:42:00.0" / "mem_info_vram_used").string()) << "1\n"; }

        assert(pdev_for_probe((tmp / "card1" / "device" / "mem_info_vram_used").string()) == "0000:42:00.0");
        assert(pdev_for_probe("nvml:0").empty());
        assert(pdev_for_probe("/nonexistent/card9/device/mem_info_vram_used").empty());
        fs::remove_all(tmp, ec);
    }

    return 0;
}
