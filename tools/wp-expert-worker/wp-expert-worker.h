#pragma once

#include <filesystem>
#include <string>

namespace wp_expert_worker {

struct Options {
    std::filesystem::path shard_manifest;
    std::filesystem::path descriptor;
    std::string           device;
    std::string           listen_host;
    int                   listen_port = 0;
    int                   slots       = 0;
    bool                  once        = false;
};

// Serve expert dispatch connections. With once=true, accept one connection
// and return after it closes; this is used by the CPU integration test.
int run(const Options & options);

} // namespace wp_expert_worker
