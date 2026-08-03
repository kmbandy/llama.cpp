#include "wp-expert-worker.h"

#include "weight-pager/wp-router.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

void print_usage(const char * argv0) {
    std::cout
        << "usage: " << argv0
        << " --shard-manifest PATH --descriptor PATH --device DEVICE"
        << " --listen HOST:PORT --slots N [--host-budget-bytes N]"
        << " [--host-victim-bytes N]"
        << " [--weight-paging-resident-experts BLOCKS]\n"
        << "       --slots is the device budget in largest-page equivalents\n"
        << "       staging defaults to up to 16 largest-page buffers\n"
        << "       WP_EXPERT_HOST_BUDGET_BYTES supplies the same optional staging budget\n"
        << "       WP_EXPERT_HOST_VICTIM_BYTES supplies the optional VRAM victim tier\n"
        << "       WP_EXPERT_RESIDENT_EXPERTS supplies resident block ranges\n";
}

int parse_positive_int(const std::string & text, const char * option) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size() || value <= 0) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return value;
}

uint64_t parse_positive_u64(const std::string & text, const char * option) {
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size() || value == 0) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return value;
}

void parse_endpoint(const std::string & text, std::string & host, int & port) {
    const size_t colon = text.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == text.size()) {
        throw std::invalid_argument("--listen expects HOST:PORT");
    }
    host = text.substr(0, colon);
    port = parse_positive_int(text.substr(colon + 1), "--listen");
    if (port > 65535) {
        throw std::invalid_argument("--listen port is out of range");
    }
}

wp_expert_worker::Options parse_cli(int argc, char ** argv) {
    wp_expert_worker::Options options;
    std::string endpoint;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            return argv[i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--shard-manifest") {
            options.shard_manifest = take();
        } else if (arg == "--descriptor") {
            options.descriptor = take();
        } else if (arg == "--device") {
            options.device = take();
        } else if (arg == "--listen") {
            endpoint = take();
        } else if (arg == "--slots") {
            options.slots = parse_positive_int(take(), "--slots");
        } else if (arg == "--host-budget-bytes") {
            options.host_budget_bytes =
                parse_positive_u64(take(), "--host-budget-bytes");
        } else if (arg == "--host-victim-bytes") {
            options.host_victim_bytes =
                parse_positive_u64(take(), "--host-victim-bytes");
        } else if (arg == "--weight-paging-resident-experts") {
            const wp::ResidentExpertRequest request =
                wp::parse_resident_expert_request(take().c_str());
            options.resident_expert_blocks = request.blocks;
            options.resident_expert_blocks_set = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.shard_manifest.empty() || options.descriptor.empty() ||
        options.device.empty() || endpoint.empty() || options.slots <= 0) {
        throw std::invalid_argument(
            "--shard-manifest, --descriptor, --device, --listen, and --slots are required");
    }
    parse_endpoint(endpoint, options.listen_host, options.listen_port);
    if (options.host_budget_bytes == 0) {
        const char * value = std::getenv("WP_EXPERT_HOST_BUDGET_BYTES");
        if (value != nullptr && value[0] != '\0') {
            options.host_budget_bytes =
                parse_positive_u64(value, "WP_EXPERT_HOST_BUDGET_BYTES");
        }
    }
    if (options.host_victim_bytes == 0) {
        const char * value = std::getenv("WP_EXPERT_HOST_VICTIM_BYTES");
        if (value != nullptr && value[0] != '\0') {
            options.host_victim_bytes =
                parse_positive_u64(value, "WP_EXPERT_HOST_VICTIM_BYTES");
        }
    }
    if (!options.resident_expert_blocks_set && options.resident_expert_blocks.empty()) {
        const char * value = std::getenv("WP_EXPERT_RESIDENT_EXPERTS");
        if (value != nullptr && value[0] != '\0') {
            const wp::ResidentExpertRequest request =
                wp::parse_resident_expert_request(value);
            options.resident_expert_blocks = request.blocks;
            options.resident_expert_blocks_set = true;
        }
    }
    return options;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return wp_expert_worker::run(parse_cli(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
