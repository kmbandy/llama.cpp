#include "wp-expert-shard-lib.h"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

std::pair<int, int> parse_range(const std::string & text, const char * option) {
    const size_t dash = text.find('-');
    if (dash == std::string::npos || dash == 0 || dash + 1 == text.size() ||
        text.find('-', dash + 1) != std::string::npos) {
        throw std::invalid_argument(std::string(option) + " expects FIRST-LAST");
    }

    int          first        = -1;
    int          last         = -1;
    const char * first_begin  = text.data();
    const char * first_end    = text.data() + dash;
    const char * last_begin   = first_end + 1;
    const char * last_end     = text.data() + text.size();
    const auto   first_result = std::from_chars(first_begin, first_end, first);
    const auto   last_result  = std::from_chars(last_begin, last_end, last);
    if (first_result.ec != std::errc() || first_result.ptr != first_end || last_result.ec != std::errc() ||
        last_result.ptr != last_end || first < 0 || last < first) {
        throw std::invalid_argument(std::string(option) + " expects a non-negative inclusive FIRST-LAST");
    }
    return { first, last };
}

void print_usage(const char * argv0) {
    std::cout << "usage:\n"
              << "  " << argv0 << " --src-manifest PATH --out-base PATH --experts FIRST-LAST [options]\n\n"
              << "options:\n"
              << "  --layers FIRST-LAST  emit only these source layers in this invocation\n"
              << "  --verify              re-read output and compare every member byte with the source\n"
              << "  --manifest-only       write the complete manifest without emitting blob files\n"
              << "  -h, --help            show this help\n\n"
              << "Data emission writes blobs and sidecars only. Run --manifest-only separately and\n"
              << "copy that manifest after all per-layer files have reached their destination.\n"
              << "Existing output files are never overwritten.\n";
}

wp_expert_shard::Options parse_cli(int argc, char ** argv) {
    wp_expert_shard::Options options;
    bool                     have_src     = false;
    bool                     have_out     = false;
    bool                     have_experts = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--src-manifest") {
            if (++i >= argc) {
                throw std::invalid_argument("--src-manifest requires a path");
            }
            options.src_manifest = argv[i];
            have_src             = true;
        } else if (arg == "--out-base") {
            if (++i >= argc) {
                throw std::invalid_argument("--out-base requires a path");
            }
            options.out_base = argv[i];
            have_out         = true;
        } else if (arg == "--experts") {
            if (++i >= argc) {
                throw std::invalid_argument("--experts requires FIRST-LAST");
            }
            const auto range     = parse_range(argv[i], "--experts");
            options.expert_first = range.first;
            options.expert_last  = range.second;
            have_experts         = true;
        } else if (arg == "--layers") {
            if (++i >= argc) {
                throw std::invalid_argument("--layers requires FIRST-LAST");
            }
            options.layers = parse_range(argv[i], "--layers");
        } else if (arg == "--verify") {
            options.verify = true;
        } else if (arg == "--manifest-only") {
            options.manifest_only = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (!have_src || !have_out || !have_experts) {
        throw std::invalid_argument("--src-manifest, --out-base, and --experts are required");
    }
    if (options.manifest_only && options.layers.has_value()) {
        throw std::invalid_argument("--layers is not valid with --manifest-only");
    }
    if (options.manifest_only && options.verify) {
        throw std::invalid_argument("--verify is not valid with --manifest-only");
    }
    return options;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const wp_expert_shard::Options options = parse_cli(argc, argv);
        wp_expert_shard::run(options);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
