#include "wp-expert-worker.h"

#include "weight-pager/wp-router.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <cerrno>

namespace {

void print_usage(const char * argv0) {
    std::cout
        << "usage: " << argv0
        << " --shard-manifest PATH --descriptor PATH --device DEVICE"
        << " --listen HOST:PORT --slots N [--host-budget-bytes N]"
        << " [--host-tier-bytes N]"
        << " [--weight-paging-resident-experts BLOCKS]"
        << " [--expert-reserve-blocks BLOCKS --expert-reserve-bytes SIZE]"
        << " [--layer-device RANGE=DEVICE[,RANGE=DEVICE]]\n"
        << "       --slots is the device budget in largest-page equivalents\n"
        << "       --device NAME1,NAME2 with --slots N1,N2 selects multiple devices\n"
        << "       --host-budget-bytes bounds in-flight reads (default 16 largest-page entries)\n"
        << "       WP_EXPERT_HOST_BUDGET_BYTES supplies the same optional in-flight budget\n"
        << "       --host-tier-bytes / WP_EXPERT_HOST_TIER_BYTES: host RAM retained for evicted\n"
        << "       and predicted pages (0 = retain nothing); both accept K/M/G suffixes (e.g. 32G)\n"
        << "       WP_EXPERT_RESIDENT_EXPERTS supplies resident block ranges\n"
        << "       WP_EXPERT_RESERVE_BLOCKS and WP_EXPERT_RESERVE_BYTES supply the reserved partition\n"
        << "       --layer-device 43-45=CPU forces those layers onto a named --device entry\n"
        << "       WP_EXPERT_LAYER_DEVICE supplies the same layer/device map\n";
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

uint64_t parse_size(const std::string & text, const char * option) {
    if (text.empty()) throw std::invalid_argument(std::string(option) + " requires a size");
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str()) throw std::invalid_argument(std::string(option) + " requires a size");
    std::string suffix(end);
    for (char & c : suffix) c = (char) std::tolower((unsigned char) c);
    uint64_t multiplier = 1;
    if (suffix == "kib" || suffix == "kb") multiplier = 1ull << 10;
    else if (suffix == "mib" || suffix == "mb") multiplier = 1ull << 20;
    else if (suffix == "gib" || suffix == "gb" || suffix == "g") multiplier = 1ull << 30;
    else if (suffix == "m") multiplier = 1ull << 20;
    else if (suffix == "k") multiplier = 1ull << 10;
    else if (!suffix.empty()) throw std::invalid_argument(std::string(option) + " has an invalid suffix");
    if (value == 0 || value > UINT64_MAX / multiplier) throw std::invalid_argument(std::string(option) + " is out of range");
    return (uint64_t) value * multiplier;
}

std::vector<std::string> parse_devices(const std::string & text) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(',', begin);
        const std::string device = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (device.empty()) {
            throw std::invalid_argument("--device requires non-empty device names");
        }
        result.push_back(device);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

std::vector<int> parse_slots(const std::string & text) {
    std::vector<int> result;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(',', begin);
        const std::string value = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        if (value.empty()) {
            throw std::invalid_argument("--slots requires positive integers");
        }
        result.push_back(parse_positive_int(value, "--slots"));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

// "43-45=CPU" or "0-6,20-22=CPU,0-2=CUDA0". Entries are comma-separated and the
// range half reuses the resident-experts grammar, which is ALSO comma-separated,
// so a token without '=' is a continuation of the range list for the next entry.
std::vector<std::pair<std::vector<int>, std::string>> parse_layer_device(
        const std::string & text, const char * option) {
    std::vector<std::pair<std::vector<int>, std::string>> result;
    std::string ranges;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t end = text.find(',', begin);
        const std::string token = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        const size_t equals = token.find('=');
        const std::string head = token.substr(0, equals);
        if (head.empty()) {
            throw std::invalid_argument(std::string(option) + " has an empty layer range");
        }
        ranges += ranges.empty() ? head : ("," + head);
        if (equals != std::string::npos) {
            const std::string device = token.substr(equals + 1);
            if (device.empty()) {
                throw std::invalid_argument(std::string(option) + " expects RANGE=DEVICE");
            }
            const wp::ResidentExpertRequest request =
                wp::parse_resident_expert_request(ranges.c_str());
            if (request.blocks.empty()) {
                throw std::invalid_argument(
                    std::string(option) + " requires a non-empty layer range before '='");
            }
            result.emplace_back(request.blocks, device);
            ranges.clear();
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (!ranges.empty() || result.empty()) {
        throw std::invalid_argument(
            std::string(option) + " expects RANGE=DEVICE, for example 43-45=CPU");
    }
    return result;
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
            options.devices = parse_devices(options.device);
        } else if (arg == "--listen") {
            endpoint = take();
        } else if (arg == "--slots") {
            const std::string value = take();
            options.device_slots = parse_slots(value);
            options.slots = options.device_slots.size() == 1
                ? options.device_slots.front() : 0;
        } else if (arg == "--host-budget-bytes") {
            options.host_budget_bytes =
                parse_size(take(), "--host-budget-bytes");
        } else if (arg == "--host-tier-bytes") {
            options.host_tier_bytes =
                parse_size(take(), "--host-tier-bytes");
        } else if (arg == "--weight-paging-resident-experts") {
            const wp::ResidentExpertRequest request =
                wp::parse_resident_expert_request(take().c_str());
            options.resident_expert_blocks = request.blocks;
            options.resident_expert_blocks_set = true;
        } else if (arg == "--expert-reserve-blocks") {
            const wp::ResidentExpertRequest request = wp::parse_resident_expert_request(take().c_str());
            options.expert_reserve_blocks = request.blocks;
            options.expert_reserve_blocks_set = true;
        } else if (arg == "--expert-reserve-bytes") {
            options.expert_reserve_bytes = parse_size(take(), "--expert-reserve-bytes");
        } else if (arg == "--layer-device") {
            options.layer_device = parse_layer_device(take(), "--layer-device");
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.shard_manifest.empty() || options.descriptor.empty() ||
        options.device.empty() || endpoint.empty() || options.device_slots.empty()) {
        throw std::invalid_argument(
            "--shard-manifest, --descriptor, --device, --listen, and --slots are required");
    }
    if (options.devices.size() != options.device_slots.size()) {
        throw std::invalid_argument(
            "--device and --slots must contain the same number of comma-separated values");
    }
    parse_endpoint(endpoint, options.listen_host, options.listen_port);
    if (options.host_budget_bytes == 0) {
        const char * value = std::getenv("WP_EXPERT_HOST_BUDGET_BYTES");
        if (value != nullptr && value[0] != '\0') {
            options.host_budget_bytes =
                parse_size(value, "WP_EXPERT_HOST_BUDGET_BYTES");
        }
    }
    if (options.host_tier_bytes == 0) {
        const char * value = std::getenv("WP_EXPERT_HOST_TIER_BYTES");
        if (value != nullptr && value[0] != '\0') {
            options.host_tier_bytes =
                parse_size(value, "WP_EXPERT_HOST_TIER_BYTES");
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
    if (!options.expert_reserve_blocks_set) {
        const char * value = std::getenv("WP_EXPERT_RESERVE_BLOCKS");
        if (value != nullptr && value[0] != '\0') {
            const wp::ResidentExpertRequest request = wp::parse_resident_expert_request(value);
            options.expert_reserve_blocks = request.blocks;
            options.expert_reserve_blocks_set = true;
        }
    }
    if (options.expert_reserve_bytes == 0) {
        const char * value = std::getenv("WP_EXPERT_RESERVE_BYTES");
        if (value != nullptr && value[0] != '\0') options.expert_reserve_bytes = parse_size(value, "WP_EXPERT_RESERVE_BYTES");
    }
    if (options.layer_device.empty()) {
        const char * value = std::getenv("WP_EXPERT_LAYER_DEVICE");
        if (value != nullptr && value[0] != '\0') {
            options.layer_device = parse_layer_device(value, "WP_EXPERT_LAYER_DEVICE");
        }
    }
    // A device name that is not in --device looks configured and is not, so it
    // is fatal here rather than a silent fall back to device 0.
    for (const auto & entry : options.layer_device) {
        if (std::find(options.devices.begin(), options.devices.end(), entry.second) ==
                options.devices.end()) {
            std::string names;
            for (const std::string & name : options.devices) {
                names += names.empty() ? name : ("," + name);
            }
            throw std::invalid_argument(
                "--layer-device names device '" + entry.second +
                "' which is not one of --device " + names);
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
