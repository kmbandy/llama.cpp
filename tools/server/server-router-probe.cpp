#include "server-router-probe.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

// Parses "<number>[ <unit>]" with unit KiB/MiB/GiB/kB/B (or none = bytes) into bytes.
static int64_t parse_size_bytes(const std::string & s) {
    char * end = nullptr;
    const long long n = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str()) {
        return -1;
    }
    while (*end == ' ' || *end == '\t') {
        end++;
    }
    const std::string unit(end);
    int64_t mult = 1;
    if (unit.rfind("KiB", 0) == 0 || unit.rfind("kB", 0) == 0 || unit.rfind("KB", 0) == 0) {
        mult = 1024;
    } else if (unit.rfind("MiB", 0) == 0) {
        mult = 1024 * 1024;
    } else if (unit.rfind("GiB", 0) == 0) {
        mult = 1024LL * 1024 * 1024;
    }
    return (int64_t) n * mult;
}

// Splits "key:<ws>value" lines; returns false if there is no colon.
static bool split_kv(const std::string & line, std::string & key, std::string & val) {
    const size_t c = line.find(':');
    if (c == std::string::npos) {
        return false;
    }
    key = line.substr(0, c);
    size_t b = c + 1;
    while (b < line.size() && (line[b] == ' ' || line[b] == '\t')) {
        b++;
    }
    size_t e = line.size();
    while (e > b && std::isspace((unsigned char) line[e - 1])) {
        e--;
    }
    val = line.substr(b, e - b);
    return true;
}

static bool is_all_digits(const std::string & s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; });
}

std::vector<proc_vram> probe_fdinfo_vram(const std::string & root) {
    std::map<std::pair<int, std::string>, int64_t> sums;

    std::error_code ec;
    for (fs::directory_iterator it(root + "/proc", ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (!is_all_digits(name)) {
            continue;
        }
        const int pid = std::atoi(name.c_str());

        std::error_code ec2;
        fs::directory_iterator fit(it->path() / "fdinfo", ec2);
        if (ec2) {
            continue; // exited or permission denied
        }

        std::set<std::pair<std::string, std::string>> seen; // (pdev, client-id)
        for (fs::directory_iterator end2; !ec2 && fit != end2; fit.increment(ec2)) {
            std::ifstream f(fit->path());
            if (!f) {
                continue;
            }
            std::string line, key, val, driver, pdev, client;
            int64_t vram = 0;
            while (std::getline(f, line)) {
                if (!split_kv(line, key, val)) {
                    continue;
                }
                if (key == "drm-driver") {
                    driver = val;
                } else if (key == "drm-pdev") {
                    pdev = val;
                } else if (key == "drm-client-id") {
                    client = val;
                } else if (key == "drm-memory-vram") {
                    vram = std::max<int64_t>(0, parse_size_bytes(val));
                }
            }
            if (driver.empty() || pdev.empty() || client.empty()) {
                continue; // not a DRM fd
            }
            if (!seen.insert({ pdev, client }).second) {
                continue; // another fd of an already-counted client
            }
            sums[{ pid, pdev }] += vram;
        }
    }

    std::vector<proc_vram> out;
    out.reserve(sums.size());
    for (const auto & kv : sums) {
        out.push_back({ kv.first.first, kv.first.second, kv.second });
    }
    return out;
}

std::optional<proc_mem> probe_proc_mem(const std::string & root, int pid) {
    std::ifstream f(root + "/proc/" + std::to_string(pid) + "/status");
    if (!f) {
        return std::nullopt;
    }
    proc_mem m = { 0, 0 };
    std::string line, key, val;
    while (std::getline(f, line)) {
        if (!split_kv(line, key, val)) {
            continue;
        }
        if (key == "RssAnon") {
            m.rss_anon = std::max<int64_t>(0, parse_size_bytes(val));
        } else if (key == "RssShmem") {
            m.rss_shmem = std::max<int64_t>(0, parse_size_bytes(val));
        }
    }
    return m;
}

int64_t probe_mem_available(const std::string & root) {
    std::ifstream f(root + "/proc/meminfo");
    std::string line, key, val;
    while (std::getline(f, line)) {
        if (split_kv(line, key, val) && key == "MemAvailable") {
            return parse_size_bytes(val);
        }
    }
    return -1;
}

int64_t probe_sysfs_vram_used(const std::string & root, const std::string & pdev) {
    std::ifstream f(root + "/sys/bus/pci/devices/" + pdev + "/mem_info_vram_used");
    long long used = 0;
    if (f >> used) {
        return used;
    }
    return -1;
}

std::string pdev_for_probe(const std::string & vram_probe) {
    if (vram_probe.empty() || vram_probe.rfind("nvml:", 0) == 0) {
        return "";
    }
    std::error_code ec;
    const fs::path dev = fs::canonical(fs::path(vram_probe).parent_path(), ec);
    if (ec) {
        return "";
    }
    return dev.filename().string();
}
