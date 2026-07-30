#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>

namespace wp_expert_shard {

struct Options {
    std::filesystem::path              src_manifest;
    std::filesystem::path              out_base;
    int                                expert_first = -1;
    int                                expert_last  = -1;
    std::optional<std::pair<int, int>> layers;
    bool                               verify        = false;
    bool                               manifest_only = false;
};

struct RunStats {
    uint64_t shards  = 0;
    uint64_t groups  = 0;
    uint64_t members = 0;
    uint64_t bytes   = 0;
};

std::filesystem::path output_manifest_path(const std::filesystem::path & out_base);
RunStats              run(const Options & options);

}  // namespace wp_expert_shard
