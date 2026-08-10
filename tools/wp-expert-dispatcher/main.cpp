#include "ggml.h"
#include "pipe-expert-dispatcher.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct options {
    std::vector<pipe_expert_dispatcher::endpoint> workers;
    std::vector<pipe_expert_assignment>           assignments;
    std::string                                   activation_path;
    int32_t                                       layer    = -1;
    uint32_t                                      n_tokens = 0;
    uint64_t                                      seq_id   = 1;
};

void print_usage(const char * argv0) {
    std::cout << "usage: " << argv0 << " --worker [MACHINE@]HOST:PORT ...\n"
              << "       " << argv0 << " --worker ... --layer N --tokens N --activation-f32 PATH"
              << " --expert ID:W0[,W1...] ... [--seq N]\n\n"
              << "With only --worker arguments, validates HELLOs and full coverage.\n"
              << "Dispatch mode reads exactly tokens*n_embd native F32 values and prints"
              << " the reduced F32 rows.\n";
}

template <typename T> T parse_integer(const std::string & text, const char * option) {
    T          value  = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string(option) + " requires an integer");
    }
    return value;
}

pipe_expert_dispatcher::endpoint parse_endpoint(const std::string & text) {
    pipe_expert_dispatcher::endpoint result;
    std::string                      address = text;
    const size_t                     at      = address.find('@');
    if (at != std::string::npos) {
        if (at == 0 || at + 1 == address.size()) {
            throw std::invalid_argument("--worker expects [MACHINE@]HOST:PORT");
        }
        result.machine = address.substr(0, at);
        address        = address.substr(at + 1);
    }
    const size_t colon = address.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == address.size()) {
        throw std::invalid_argument("--worker expects [MACHINE@]HOST:PORT");
    }
    result.host = address.substr(0, colon);
    result.port = parse_integer<int>(address.substr(colon + 1), "--worker");
    return result;
}

pipe_expert_assignment parse_assignment(const std::string & text) {
    const size_t colon = text.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == text.size()) {
        throw std::invalid_argument("--expert expects ID:W0[,W1...]");
    }
    pipe_expert_assignment result;
    result.expert_id = parse_integer<int32_t>(text.substr(0, colon), "--expert");
    size_t begin     = colon + 1;
    while (begin < text.size()) {
        const size_t      comma    = text.find(',', begin);
        const std::string value    = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        size_t            consumed = 0;
        const float       weight   = std::stof(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("--expert contains an invalid weight");
        }
        result.weights.push_back(weight);
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return result;
}

options parse_cli(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg  = argv[i];
        auto              take = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            return argv[i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--worker") {
            result.workers.push_back(parse_endpoint(take()));
        } else if (arg == "--layer") {
            result.layer = parse_integer<int32_t>(take(), "--layer");
        } else if (arg == "--tokens") {
            result.n_tokens = parse_integer<uint32_t>(take(), "--tokens");
        } else if (arg == "--seq") {
            result.seq_id = parse_integer<uint64_t>(take(), "--seq");
        } else if (arg == "--activation-f32") {
            result.activation_path = take();
        } else if (arg == "--expert") {
            result.assignments.push_back(parse_assignment(take()));
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (result.workers.empty()) {
        throw std::invalid_argument("at least one --worker is required");
    }
    const bool any_dispatch =
        result.layer >= 0 || result.n_tokens != 0 || !result.activation_path.empty() || !result.assignments.empty();
    const bool full_dispatch =
        result.layer >= 0 && result.n_tokens != 0 && !result.activation_path.empty() && !result.assignments.empty();
    if (any_dispatch != full_dispatch) {
        throw std::invalid_argument("dispatch mode requires --layer, --tokens, --activation-f32, and --expert");
    }
    return result;
}

std::vector<float> read_activations(const std::string & path, size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open activation file " + path);
    }
    std::vector<float> values(count);
    input.read(reinterpret_cast<char *>(values.data()), (std::streamsize) (values.size() * sizeof(float)));
    if ((size_t) input.gcount() != values.size() * sizeof(float) || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("activation file size does not equal tokens*n_embd F32 values");
    }
    return values;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const options                      args = parse_cli(argc, argv);
        pipe_expert_dispatcher::dispatcher dispatcher(args.workers);
        std::cout << "connected workers=" << dispatcher.workers().size() << " n_embd=" << dispatcher.n_embd()
                  << " n_expert=" << dispatcher.n_expert() << " model=" << dispatcher.model_identity() << '\n';
        if (args.layer < 0) {
            return 0;
        }
        for (const pipe_expert_assignment & assignment : args.assignments) {
            if (assignment.weights.size() != args.n_tokens) {
                throw std::invalid_argument("each --expert must provide one weight per token");
            }
        }
        const size_t                count       = (size_t) args.n_tokens * (size_t) dispatcher.n_embd();
        const std::vector<float> activations = read_activations(args.activation_path, count);
        const std::vector<float>    result =
            dispatcher.dispatch(args.layer, args.seq_id, args.n_tokens, activations, args.assignments, 0.0f);
        for (uint32_t token = 0; token < args.n_tokens; ++token) {
            std::cout << "token " << token;
            const size_t base = (size_t) token * dispatcher.n_embd();
            for (int32_t i = 0; i < dispatcher.n_embd(); ++i) {
                std::cout << " " << result[base + (size_t) i];
            }
            std::cout << '\n';
        }
        const pipe_expert_dispatcher::dispatch_stats & stats = dispatcher.last_dispatch_stats();
        std::cout << "workers_used=" << stats.workers_used << " first_await_in_flight=" << stats.first_await_in_flight
                  << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
