#pragma once

// Pure readers of /proc and /sys used by the router's VRAM/RAM ledger.
// Every function takes a `root` prefix ("" for the real filesystem) so tests
// can point at a fixture tree laid out as <root>/proc/... and <root>/sys/...

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct proc_vram {
    int         pid;
    std::string pdev;       // PCI address, e.g. "0000:42:00.0"
    int64_t     vram_bytes;
};

struct proc_mem {
    int64_t rss_anon;
    int64_t rss_shmem;
};

// VRAM per (pid, pdev) from /proc/<pid>/fdinfo/*, deduplicated by drm-client-id.
// Unreadable PIDs are skipped. Result is sorted by (pid, pdev).
std::vector<proc_vram> probe_fdinfo_vram(const std::string & root);

// RssAnon / RssShmem of a PID in bytes; nullopt if /proc/<pid>/status is unreadable.
std::optional<proc_mem> probe_proc_mem(const std::string & root, int pid);

// MemAvailable from /proc/meminfo in bytes; -1 if unreadable.
int64_t probe_mem_available(const std::string & root);

// mem_info_vram_used of a PCI device in bytes; -1 if unreadable.
int64_t probe_sysfs_vram_used(const std::string & root, const std::string & pdev);

// PCI address of the GPU a router slot's vram_probe path refers to, e.g.
// "/sys/class/drm/card1/device/mem_info_vram_used" -> "0000:42:00.0".
// This is the same probe path read_physical_free_bytes() reads. Returns ""
// for non-sysfs probes (e.g. "nvml:0") or if the path does not resolve.
std::string pdev_for_probe(const std::string & vram_probe);
