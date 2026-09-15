#pragma once

#include <filesystem>

namespace fs = std::filesystem;

// Shared with the test target: the .cpp includes this header and keeps its
// run() at global scope. The executable's main() is excluded when
// WP_EXPERT_DESCRIPTOR_NO_MAIN is defined (the test links this translation
// unit into its own main).

struct Options {
    fs::path model;
    fs::path manifest;
    fs::path output;
};

// Builds the expert descriptor described by the shard manifest in
// `options.manifest` and writes it to `options.output`.
// Paths are canonicalized internally. Throws std::runtime_error on any
// validation failure. Returns 0 on success.
int run(const Options & options);
