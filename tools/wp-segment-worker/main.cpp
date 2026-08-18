#include "arg.h"
#include "common.h"
#include "wp-segment-worker.h"

#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

uint32_t parse_u32(const std::string & value, const char * name) {
    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument(std::string(name) + " requires an unsigned integer");
    }
    return result;
}

wp_segment_worker::options parse_worker_args(int & argc, char ** argv) {
    wp_segment_worker::options options;
    bool has_manifest = false;
    bool has_segment = false;
    int write = 1;
    for (int read = 1; read < argc; ++read) {
        const std::string arg = argv[read];
        auto take = [&]() -> std::string {
            if (++read >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            return argv[read];
        };
        if (arg == "--manifest") {
            options.manifest_path = take();
            has_manifest = true;
        } else if (arg == "--segment") {
            options.segment_id = parse_u32(take(), "--segment");
            has_segment = true;
        } else if (arg == "--segment-rs-snapshots") {
            options.recurrent_snapshots = parse_u32(take(), "--segment-rs-snapshots");
            if (options.recurrent_snapshots == 0) {
                throw std::invalid_argument("--segment-rs-snapshots must be positive");
            }
        } else {
            argv[write++] = argv[read];
        }
    }
    argc = write;
    if (!has_manifest || !has_segment) {
        throw std::invalid_argument("--manifest and --segment are required");
    }
    return options;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const wp_segment_worker::options options = parse_worker_args(argc, argv);
        const wp_segment_worker::resolved_segment resolved = wp_segment_worker::resolve_segment(options);
        common_params params;
        params.model.path = resolved.stage_gguf.string();
        params.fit_params = false;
        common_init();
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, nullptr)) {
            return 1;
        }
        return wp_segment_worker::run(options, params);
    } catch (const std::exception & error) {
        std::cerr << "wp-segment-worker: " << error.what() << '\n';
        return 1;
    }
}
