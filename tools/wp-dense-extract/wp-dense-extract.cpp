#include "wp-dense-extract-lib.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace {

struct options {
    std::string model;
    std::string output;
    bool        verify = false;
};

[[noreturn]] void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --model <first.gguf> --output <dense.gguf> [--verify]\n", argv0);
    std::exit(1);
}

options parse_args(int argc, const char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg  = argv[i];
        auto              next = [&](const char * name) -> std::string {
            if (++i >= argc) {
                std::fprintf(stderr, "%s: missing value for %s\n", argv[0], name);
                usage(argv[0]);
            }
            return argv[i];
        };
        if (arg == "--model") {
            opt.model = next("--model");
        } else if (arg == "--output") {
            opt.output = next("--output");
        } else if (arg == "--verify") {
            opt.verify = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        } else {
            std::fprintf(stderr, "%s: unknown argument '%s'\n", argv[0], arg.c_str());
            usage(argv[0]);
        }
    }
    if (opt.model.empty() || opt.output.empty()) {
        usage(argv[0]);
    }
    return opt;
}

}  // namespace

int main(int argc, const char ** argv) {
    const options opt = parse_args(argc, argv);
    try {
        const wp_dense_extract::result res = wp_dense_extract::extract(opt.model, opt.output, opt.verify);
        std::fprintf(stderr,
                     "wp-dense-extract: wrote %lld tensors, %llu bytes (file %llu bytes); "
                     "excluded %lld routed-expert tensors, %llu bytes; verify %s\n",
                     static_cast<long long>(res.tensor_count), static_cast<unsigned long long>(res.tensor_bytes),
                     static_cast<unsigned long long>(res.file_bytes), static_cast<long long>(res.routed_tensor_count),
                     static_cast<unsigned long long>(res.routed_tensor_bytes), res.verified ? "PASS" : "not requested");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "wp-dense-extract: error: %s\n", error.what());
        return 1;
    }
    return 0;
}
