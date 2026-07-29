// wp-stage-split: emit a per-stage GGUF for cross-machine pipeline
// parallelism (docs/superpowers/specs/2026-07-28-cross-machine-pipeline-
// parallelism.md, Phase 1b).
//
// Each pipeline stage needs a LOADABLE GGUF holding its own layer band plus
// the shared tensors it owns -- both machines loading the full model is
// impossible when a stage's disk budget is smaller than the model. The split
// logic lives in wp-stage-split-lib; this is only the CLI wrapper.
//
// Usage:
//   wp-stage-split --model in.gguf --out head.gguf --first 0  --last 56
//   wp-stage-split --model in.gguf --out tail.gguf --first 57 --last 77
//   wp-stage-split --model in.gguf --first 57 --last 77 --dry-run

#include "wp-stage-split-lib.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

struct options {
    std::string model;
    std::string out;
    int32_t     first   = -1;
    int32_t     last    = -1;
    bool        dry_run = false;
};

[[noreturn]] void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <in.gguf> [--out <stage.gguf>] --first N --last M [--dry-run]\n", argv0);
    std::exit(1);
}

options parse_args(int argc, const char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (++i >= argc) {
                std::fprintf(stderr, "%s: missing value for %s\n", argv[0], name);
                usage(argv[0]);
            }
            return argv[i];
        };
        if (arg == "--model") {
            opt.model = next("--model");
        } else if (arg == "--out") {
            opt.out = next("--out");
        } else if (arg == "--first") {
            opt.first = std::stoi(next("--first"));
        } else if (arg == "--last") {
            opt.last = std::stoi(next("--last"));
        } else if (arg == "--dry-run") {
            opt.dry_run = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        } else {
            std::fprintf(stderr, "%s: unknown argument '%s'\n", argv[0], arg.c_str());
            usage(argv[0]);
        }
    }
    if (opt.model.empty() || (!opt.dry_run && opt.out.empty())) {
        usage(argv[0]);
    }
    return opt;
}

} // namespace

int main(int argc, const char ** argv) {
    const options opt = parse_args(argc, argv);

    try {
        const wp_stage_split::result res =
            wp_stage_split::split_stage(opt.model, opt.out, opt.first, opt.last, opt.dry_run);

        std::fprintf(stderr,
            "wp-stage-split: layers [%d, %d] of %d (%s%s): %lld of %lld tensors, "
            "%.2f of %.2f GiB\n",
            res.first, res.last, res.n_layer,
            res.first == 0 ? "head " : "",
            res.last == res.n_layer - 1 ? "tail " : "",
            (long long) res.n_tensors_out, (long long) res.n_tensors_in,
            res.bytes_out / 1024.0 / 1024.0 / 1024.0,
            res.bytes_in  / 1024.0 / 1024.0 / 1024.0);

        if (opt.dry_run) {
            for (const std::string & name : res.tensor_names) {
                std::printf("%s\n", name.c_str());
            }
        } else {
            std::fprintf(stderr, "wp-stage-split: wrote %s\n", opt.out.c_str());
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "wp-stage-split: error: %s\n", e.what());
        return 1;
    }

    return 0;
}
