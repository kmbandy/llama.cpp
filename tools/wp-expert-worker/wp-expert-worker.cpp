#include "wp-expert-worker.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "pipe-protocol.h"
#include "pipe-transport.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace wp_expert_worker {

static constexpr uint64_t DEFAULT_STAGING_BUFFERS = 16;

ResourcePlan plan_resources(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes) {
    if (pages.empty() || requested_slots <= 0) {
        throw std::invalid_argument("invalid expert resource plan dimensions");
    }

    std::map<uint64_t, int> histogram;
    std::map<uint64_t, std::map<int, int>> layer_counts;
    uint64_t max_page_size = 0;
    for (const ResourcePage & page : pages) {
        if (page.layer < 0 || page.size == 0 ||
            histogram[page.size] == std::numeric_limits<int>::max() ||
            layer_counts[page.size][page.layer] == std::numeric_limits<int>::max()) {
            throw std::invalid_argument("invalid expert resource page");
        }
        ++histogram[page.size];
        ++layer_counts[page.size][page.layer];
        max_page_size = std::max(max_page_size, page.size);
    }
    if (max_page_size >
        std::numeric_limits<uint64_t>::max() / (uint64_t) requested_slots) {
        throw std::overflow_error("expert device budget overflows");
    }

    ResourcePlan result;
    result.requested_slots      = requested_slots;
    result.device_budget_bytes =
        max_page_size * (uint64_t) requested_slots;

    uint64_t    total_pages   = 0;
    long double weighted_bytes = 0.0;
    uint64_t    floor_bytes   = 0;
    std::vector<SlotClass> classes;
    classes.reserve(histogram.size());
    for (const auto & item : histogram) {
        int floor = 0;
        for (const auto & layer : layer_counts.at(item.first)) {
            floor = std::max(floor, layer.second);
        }
        if ((uint64_t) floor >
            (std::numeric_limits<uint64_t>::max() - floor_bytes) / item.first) {
            throw std::overflow_error("expert pin floor overflows");
        }
        floor_bytes += item.first * (uint64_t) floor;
        total_pages += (uint64_t) item.second;
        weighted_bytes +=
            (long double) item.first * (long double) item.second;
        classes.push_back({ item.first, 0, floor, item.second });
    }

    bool use_size_classes = floor_bytes <= result.device_budget_bytes;
    if (use_size_classes) {
        const long double average =
            weighted_bytes / (long double) total_pages;
        const long double remaining =
            (long double) (result.device_budget_bytes - floor_bytes);
        const long double remaining_slots = remaining / average;
        for (SlotClass & slot_class : classes) {
            const long double fraction =
                (long double) slot_class.pages / (long double) total_pages;
            long long count =
                (long long) slot_class.pin_floor +
                std::llround(fraction * remaining_slots);
            count = std::max<long long>(1, count);
            count = std::max<long long>(slot_class.pin_floor, count);
            count = std::min<long long>(slot_class.pages, count);
            slot_class.slots = (int) count;
        }

        auto planned_bytes = [&]() {
            uint64_t bytes = 0;
            for (const SlotClass & slot_class : classes) {
                bytes += slot_class.size * (uint64_t) slot_class.slots;
            }
            return bytes;
        };
        while (planned_bytes() > result.device_budget_bytes) {
            SlotClass * trim = nullptr;
            for (SlotClass & slot_class : classes) {
                const int keep = std::max(1, slot_class.pin_floor);
                if (slot_class.slots > keep &&
                    (trim == nullptr || slot_class.size > trim->size)) {
                    trim = &slot_class;
                }
            }
            if (trim == nullptr) {
                use_size_classes = false;
                break;
            }
            --trim->slots;
        }
    }

    if (!use_size_classes) {
        int max_layer_pages = 0;
        std::map<int, int> pages_by_layer;
        for (const ResourcePage & page : pages) {
            max_layer_pages =
                std::max(max_layer_pages, ++pages_by_layer[page.layer]);
        }
        if (requested_slots < max_layer_pages) {
            throw std::invalid_argument(
                "expert slot budget is smaller than the largest layer request");
        }
        classes.clear();
        classes.push_back({
            max_page_size, requested_slots, max_layer_pages, (int) pages.size()
        });
    }

    result.size_classes = use_size_classes;
    result.slot_classes = std::move(classes);
    for (const SlotClass & slot_class : result.slot_classes) {
        if (slot_class.slots >
            std::numeric_limits<int>::max() - result.slot_count) {
            throw std::overflow_error("expert slot count overflows");
        }
        result.slot_count += slot_class.slots;
        result.device_bytes +=
            slot_class.size * (uint64_t) slot_class.slots;
    }

    if (max_page_size >
        std::numeric_limits<uint64_t>::max() / DEFAULT_STAGING_BUFFERS) {
        throw std::overflow_error("default expert host budget overflows");
    }
    result.host_budget_bytes = host_budget_bytes == 0
        ? max_page_size * std::min<uint64_t>(
              (uint64_t) result.slot_count, DEFAULT_STAGING_BUFFERS)
        : host_budget_bytes;
    if (result.host_budget_bytes < max_page_size) {
        throw std::invalid_argument(
            "expert host budget is smaller than the largest page");
    }
    const uint64_t staging_count =
        std::min<uint64_t>(
            (uint64_t) result.slot_count,
            result.host_budget_bytes / max_page_size);
    if (staging_count == 0 ||
        staging_count > (uint64_t) std::numeric_limits<int>::max()) {
        throw std::overflow_error("invalid expert staging buffer count");
    }
    result.staging_buffers      = (int) staging_count;
    result.staging_buffer_bytes = max_page_size;
    result.staging_bytes        = max_page_size * staging_count;
    return result;
}

namespace {

static constexpr const char * MANIFEST_FORMAT =
    "llama.cpp.weight-pager.expert-shard-manifest";
static constexpr const char * INDEX_FORMAT =
    "llama.cpp.weight-pager.expert-shard-index";
static constexpr const char * DESCRIPTOR_FORMAT =
    "llama.cpp.weight-pager.expert-descriptor";
static constexpr size_t DIRECT_ALIGNMENT = 4096;

struct RoleSpec {
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t        ne0 = 0;
    int64_t        ne1 = 0;
    uint64_t       bytes = 0;
    std::string    source_tensor_name;
};

struct HParams {
    int n_layer       = 0;
    int n_embd        = 0;
    int n_ff_exp      = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
};

struct Descriptor {
    HParams                                      hparams;
    int                                          expert_first = -1;
    int                                          expert_last  = -1;
    std::string                                  input_model;
    std::string                                  identity_algorithm;
    std::string                                  identity_value;
    std::vector<std::string>                     model_files;
    std::map<int, std::map<std::string, RoleSpec>> layers;
};

struct MemberSpan {
    uint64_t offset = 0;
    uint64_t size   = 0;
};

struct ExpertPage {
    int                               layer  = -1;
    int                               expert = -1;
    fs::path                          blob;
    uint64_t                          offset = 0;
    uint64_t                          size   = 0;
    std::map<std::string, MemberSpan> roles;
};

struct Catalog {
    Descriptor                                      descriptor;
    std::map<std::pair<int, int>, ExpertPage>       pages;
    std::vector<int>                                layers;
    uint64_t                                        max_page_size = 0;
};

struct backend_deleter {
    void operator()(ggml_backend * backend) const {
        ggml_backend_free(backend);
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        ggml_backend_buffer_free(buffer);
    }
};

struct context_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using buffer_ptr  = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;

json read_json(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    json value;
    try {
        input >> value;
    } catch (const json::exception & error) {
        throw std::runtime_error("failed to parse " + path.string() + ": " + error.what());
    }
    return value;
}

template <typename T>
T get_value(const json & value, const char * key, const fs::path & path) {
    try {
        return value.at(key).get<T>();
    } catch (const json::exception & error) {
        throw std::runtime_error(path.string() + ": invalid " + key + ": " + error.what());
    }
}

const json & get_array(const json & value, const char * key, const fs::path & path) {
    try {
        const json & result = value.at(key);
        if (!result.is_array()) {
            throw std::runtime_error(path.string() + ": " + key + " is not an array");
        }
        return result;
    } catch (const json::exception & error) {
        throw std::runtime_error(path.string() + ": invalid " + key + ": " + error.what());
    }
}

void check_format(const json & value, const char * expected, const fs::path & path) {
    if (get_value<std::string>(value, "format", path) != expected ||
        get_value<int>(value, "version", path) != 1) {
        throw std::runtime_error(path.string() + ": unsupported format or version");
    }
}

int checked_int(uint64_t value, const char * name) {
    if (value == 0 || value > INT32_MAX) {
        throw std::runtime_error(std::string(name) + " is out of range");
    }
    return (int) value;
}

void sha_update_u64(sha256_t & hash, uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = (unsigned char) ((value >> (i * 8)) & 0xffu);
    }
    sha256_update(&hash, bytes.data(), bytes.size());
}

void sha_update_string(sha256_t & hash, const std::string & value) {
    sha_update_u64(hash, value.size());
    sha256_update(
        &hash, reinterpret_cast<const unsigned char *>(value.data()), value.size());
}

std::string source_model_identity(
        const std::string & input_model,
        const std::vector<std::string> & model_files) {
    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-expert.source-model.v1");
    sha_update_string(hash, input_model);
    sha_update_u64(hash, model_files.size());
    for (const std::string & model_file : model_files) {
        sha_update_string(hash, model_file);
    }

    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_final(&hash, digest.data());
    std::ostringstream result;
    result << "sha256:" << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        result << std::setw(2) << (unsigned int) byte;
    }
    return result.str();
}

RoleSpec parse_role(const json & value, const fs::path & path) {
    RoleSpec role;
    const int type = get_value<int>(value, "ggml_type", path);
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        throw std::runtime_error(path.string() + ": ggml_type is out of range");
    }
    role.type = (enum ggml_type) type;
    const std::string type_name = get_value<std::string>(value, "ggml_type_name", path);
    if (type_name != ggml_type_name(role.type)) {
        throw std::runtime_error(path.string() + ": ggml type name and enum disagree");
    }
    const json & shape = get_array(value, "shape", path);
    if (shape.size() != 2) {
        throw std::runtime_error(path.string() + ": expert role shape is not 2D");
    }
    role.ne0   = shape.at(0).get<int64_t>();
    role.ne1   = shape.at(1).get<int64_t>();
    role.bytes = get_value<uint64_t>(value, "bytes_per_expert", path);
    role.source_tensor_name =
        get_value<std::string>(value, "source_tensor_name", path);
    if (role.ne0 <= 0 || role.ne1 <= 0 ||
        role.source_tensor_name.empty() ||
        ggml_row_size(role.type, role.ne0) * (uint64_t) role.ne1 != role.bytes) {
        throw std::runtime_error(path.string() + ": expert role shape/type byte count is invalid");
    }
    return role;
}

Descriptor load_descriptor(const fs::path & path) {
    const json value = read_json(path);
    check_format(value, DESCRIPTOR_FORMAT, path);
    Descriptor result;

    const json & hparams = value.at("hparams");
    result.hparams.n_layer =
        checked_int(get_value<uint64_t>(hparams, "n_layer", path), "n_layer");
    result.hparams.n_embd =
        checked_int(get_value<uint64_t>(hparams, "n_embd", path), "n_embd");
    result.hparams.n_ff_exp =
        checked_int(get_value<uint64_t>(hparams, "n_ff_exp", path), "n_ff_exp");
    result.hparams.n_expert =
        checked_int(get_value<uint64_t>(hparams, "n_expert", path), "n_expert");
    result.hparams.n_expert_used =
        checked_int(get_value<uint64_t>(hparams, "n_expert_used", path), "n_expert_used");
    if (result.hparams.n_expert_used > result.hparams.n_expert ||
        get_value<std::string>(hparams, "activation", path) != "silu") {
        throw std::runtime_error(path.string() + ": unsupported expert hparams");
    }

    const json & range = value.at("retained_expert_range");
    result.expert_first = get_value<int>(range, "first", path);
    result.expert_last  = get_value<int>(range, "last", path);
    if (result.expert_first < 0 || result.expert_last < result.expert_first ||
        result.expert_last >= result.hparams.n_expert) {
        throw std::runtime_error(path.string() + ": invalid retained expert range");
    }

    const json & identity = value.at("shard_manifest_identity");
    result.identity_algorithm = get_value<std::string>(identity, "algorithm", path);
    result.identity_value     = get_value<std::string>(identity, "value", path);
    if (result.identity_algorithm.empty() || result.identity_value.empty()) {
        throw std::runtime_error(path.string() + ": empty shard manifest identity");
    }
    const json & source_model = value.at("source_model");
    result.input_model = get_value<std::string>(source_model, "input_model", path);
    for (const json & model_file : get_array(source_model, "model_files", path)) {
        result.model_files.push_back(model_file.get<std::string>());
    }
    if (result.input_model.empty() || result.model_files.empty()) {
        throw std::runtime_error(path.string() + ": descriptor has no source model");
    }

    const json & layers = get_array(value, "layers", path);
    for (const json & layer_value : layers) {
        const int layer = get_value<int>(layer_value, "layer", path);
        if (layer < 0 || layer >= result.hparams.n_layer || result.layers.count(layer) != 0) {
            throw std::runtime_error(path.string() + ": invalid or repeated layer descriptor");
        }
        const json & roles = layer_value.at("roles");
        std::map<std::string, RoleSpec> parsed;
        for (const std::string name : { "gate", "up", "down" }) {
            parsed.emplace(name, parse_role(roles.at(name), path));
        }
        if (parsed.at("gate").ne0 != result.hparams.n_embd ||
            parsed.at("gate").ne1 != result.hparams.n_ff_exp ||
            parsed.at("up").ne0 != result.hparams.n_embd ||
            parsed.at("up").ne1 != result.hparams.n_ff_exp ||
            parsed.at("down").ne0 != result.hparams.n_ff_exp ||
            parsed.at("down").ne1 != result.hparams.n_embd) {
            throw std::runtime_error(path.string() + ": descriptor role shapes disagree with hparams");
        }
        result.layers.emplace(layer, std::move(parsed));
    }
    if (result.layers.empty()) {
        throw std::runtime_error(path.string() + ": descriptor has no layers");
    }
    return result;
}

std::string role_from_mask(uint64_t mask) {
    switch (mask) {
        case 1: return "up";
        case 2: return "gate";
        case 4: return "down";
        default:
            throw std::runtime_error("sidecar has an unknown role mask " + std::to_string(mask));
    }
}

Catalog load_catalog(const fs::path & manifest_path, const fs::path & descriptor_path) {
    Catalog result;
    result.descriptor = load_descriptor(descriptor_path);

    const json manifest = read_json(manifest_path);
    check_format(manifest, MANIFEST_FORMAT, manifest_path);
    if (get_value<std::string>(manifest, "sharding_mode", manifest_path) !=
        "expert-index-range") {
        throw std::runtime_error("worker requires an expert-index-range shard manifest");
    }
    const json & identity = manifest.at("content_hash");
    if (get_value<std::string>(identity, "algorithm", manifest_path) !=
            result.descriptor.identity_algorithm ||
        get_value<std::string>(identity, "value", manifest_path) !=
            result.descriptor.identity_value) {
        throw std::runtime_error("descriptor does not match shard manifest identity");
    }
    if (get_array(manifest, "model_files", manifest_path) !=
        json(result.descriptor.model_files)) {
        throw std::runtime_error("descriptor and shard manifest model files disagree");
    }
    if (get_value<std::string>(manifest, "input_model", manifest_path) !=
        result.descriptor.input_model) {
        throw std::runtime_error("descriptor and shard manifest input models disagree");
    }
    const json & range = manifest.at("retained_expert_range");
    if (get_value<int>(range, "first", manifest_path) != result.descriptor.expert_first ||
        get_value<int>(range, "last", manifest_path) != result.descriptor.expert_last) {
        throw std::runtime_error("descriptor and shard manifest expert ranges disagree");
    }

    const json & shards = get_array(manifest, "shards", manifest_path);
    if (shards.size() != get_value<uint64_t>(manifest, "shard_count", manifest_path)) {
        throw std::runtime_error("manifest shard count mismatch");
    }
    std::set<int> seen_layers;
    std::set<int> seen_shard_indices;
    uint64_t total_groups = 0;
    uint64_t total_bytes  = 0;
    for (const json & shard : shards) {
        const fs::path index_path = manifest_path.parent_path() /
            get_value<std::string>(shard, "index_file", manifest_path);
        const json index = read_json(index_path);
        check_format(index, INDEX_FORMAT, index_path);
        const int shard_index = get_value<int>(index, "shard_index", index_path);
        const int layer_first = get_value<int>(index, "layer_first", index_path);
        const int layer_last  = get_value<int>(index, "layer_last", index_path);
        if (shard_index < 0 || (uint64_t) shard_index >= shards.size() ||
            !seen_shard_indices.insert(shard_index).second ||
            get_value<uint64_t>(index, "shard_count", index_path) != shards.size() ||
            get_value<int>(shard, "shard_index", manifest_path) != shard_index ||
            get_value<int>(shard, "layer_first", manifest_path) != layer_first ||
            get_value<int>(shard, "layer_last", manifest_path) != layer_last ||
            get_value<uint64_t>(shard, "group_count", manifest_path) !=
                get_value<uint64_t>(index, "group_count", index_path) ||
            get_value<uint64_t>(shard, "blob_bytes", manifest_path) !=
                get_value<uint64_t>(index, "blob_bytes", index_path) ||
            get_value<std::string>(shard, "blob_file", manifest_path) !=
                get_value<std::string>(index, "blob_file", index_path) ||
            get_array(index, "model_files", index_path) !=
                get_array(manifest, "model_files", manifest_path) ||
            layer_first != layer_last || !seen_layers.insert(layer_first).second ||
            result.descriptor.layers.count(layer_first) == 0) {
            throw std::runtime_error(index_path.string() + ": invalid or undescribed layer");
        }

        const fs::path blob_path = manifest_path.parent_path() /
            get_value<std::string>(index, "blob_file", index_path);
        std::error_code size_error;
        const uint64_t actual_size = fs::file_size(blob_path, size_error);
        const uint64_t blob_bytes  = get_value<uint64_t>(index, "blob_bytes", index_path);
        if (size_error || actual_size != blob_bytes) {
            throw std::runtime_error(blob_path.string() + ": blob size does not match sidecar");
        }

        const json & groups = get_array(index, "groups", index_path);
        if (groups.size() != get_value<uint64_t>(index, "group_count", index_path)) {
            throw std::runtime_error(index_path.string() + ": group count mismatch");
        }
        uint64_t next_offset = 0;
        int expected_expert = result.descriptor.expert_first;
        for (const json & group : groups) {
            ExpertPage page;
            page.layer  = get_value<int>(group, "block_idx", index_path);
            page.expert = get_value<int>(group, "expert_idx", index_path);
            page.blob   = blob_path;
            page.offset = next_offset;
            if (page.layer != layer_first || page.expert != expected_expert++) {
                throw std::runtime_error(index_path.string() + ": expert groups are not dense and ordered");
            }
            const json & members = get_array(group, "members", index_path);
            if (members.size() != 3 ||
                get_value<uint64_t>(group, "member_count", index_path) != 3) {
                throw std::runtime_error(index_path.string() + ": expert group does not have three members");
            }
            const auto & role_specs = result.descriptor.layers.at(page.layer);
            for (const json & member : members) {
                const std::string role =
                    role_from_mask(get_value<uint64_t>(member, "role_mask", index_path));
                const uint64_t offset = get_value<uint64_t>(member, "offset", index_path);
                const uint64_t size   = get_value<uint64_t>(member, "size", index_path);
                const std::string source_tensor_name =
                    get_value<std::string>(member, "source_tensor_name", index_path);
                if (offset != next_offset || size != role_specs.at(role).bytes ||
                    source_tensor_name != role_specs.at(role).source_tensor_name ||
                    page.roles.count(role) != 0) {
                    throw std::runtime_error(
                        index_path.string() + ": member spans disagree with descriptor");
                }
                page.roles.emplace(role, MemberSpan{ offset - page.offset, size });
                next_offset += size;
                page.size += size;
            }
            if (page.offset % DIRECT_ALIGNMENT != 0 ||
                page.size % DIRECT_ALIGNMENT != 0) {
                throw std::runtime_error(
                    index_path.string() +
                    ": expert page is not aligned for one O_DIRECT read");
            }
            result.max_page_size = std::max(result.max_page_size, page.size);
            if (!result.pages.emplace(
                    std::make_pair(page.layer, page.expert), std::move(page)).second) {
                throw std::runtime_error(index_path.string() + ": repeated expert page");
            }
            ++total_groups;
        }
        if (next_offset != blob_bytes) {
            throw std::runtime_error(index_path.string() + ": groups do not cover blob");
        }
        total_bytes += blob_bytes;
    }
    if (seen_layers.size() != result.descriptor.layers.size()) {
        throw std::runtime_error("descriptor and shard manifest layer sets disagree");
    }
    if (total_groups != get_value<uint64_t>(manifest, "total_group_count", manifest_path) ||
        total_bytes != get_value<uint64_t>(manifest, "total_blob_bytes", manifest_path)) {
        throw std::runtime_error("shard catalog totals disagree with manifest");
    }
    result.layers.assign(seen_layers.begin(), seen_layers.end());
    return result;
}

backend_ptr init_backend(const std::string & device) {
    std::string lower = device;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    ggml_backend_t backend = nullptr;
    if (lower == "cpu") {
        backend = ggml_backend_cpu_init();
        if (backend != nullptr) {
            const unsigned int hw = std::thread::hardware_concurrency();
            ggml_backend_cpu_set_n_threads(backend, hw == 0 ? 1 : (int) hw);
        }
    } else {
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name(device.c_str(), nullptr);
    }
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize device " + device);
    }
    return backend_ptr(backend);
}

std::vector<ResourcePage> resource_pages(const Catalog & catalog) {
    std::vector<ResourcePage> result;
    result.reserve(catalog.pages.size());
    for (const auto & item : catalog.pages) {
        result.push_back({ item.second.layer, item.second.size });
    }
    return result;
}

struct free_deleter {
    void operator()(void * p) const {
        std::free(p);
    }
};

class StagingPool {
public:
    explicit StagingPool(const ResourcePlan & resources) :
        buffer_bytes_(resources.staging_buffer_bytes) {
        buffers_.reserve((size_t) resources.staging_buffers);
        available_.reserve((size_t) resources.staging_buffers);
        for (int i = 0; i < resources.staging_buffers; ++i) {
            void * raw = nullptr;
            if (posix_memalign(
                    &raw, DIRECT_ALIGNMENT, (size_t) buffer_bytes_) != 0) {
                throw std::runtime_error(
                    "failed to allocate aligned O_DIRECT staging buffer");
            }
            buffers_.emplace_back(raw);
            available_.push_back(raw);
        }
    }

    class Lease {
    public:
        Lease(Lease && other) noexcept :
            owner_(other.owner_), data_(other.data_) {
            other.owner_ = nullptr;
            other.data_  = nullptr;
        }

        ~Lease() {
            if (owner_ != nullptr) {
                owner_->release(data_);
            }
        }

        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;
        Lease & operator=(Lease &&) = delete;

        void * get() const {
            return data_;
        }

    private:
        friend class StagingPool;

        Lease(StagingPool * owner, void * data) :
            owner_(owner), data_(data) {
        }

        StagingPool * owner_ = nullptr;
        void *        data_  = nullptr;
    };

    Lease borrow() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_cv_.wait(lock, [&]() { return !available_.empty(); });
        void * result = available_.back();
        available_.pop_back();
        return Lease(this, result);
    }

    int buffer_count() const {
        return (int) buffers_.size();
    }

    uint64_t buffer_bytes() const {
        return buffer_bytes_;
    }

private:
    void release(void * data) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            available_.push_back(data);
        }
        available_cv_.notify_one();
    }

    uint64_t                                        buffer_bytes_ = 0;
    std::vector<std::unique_ptr<void, free_deleter>> buffers_;
    std::vector<void *>                             available_;
    std::mutex                                      mutex_;
    std::condition_variable                         available_cv_;
};

class ExpertSlotPool {
private:
    struct Miss {
        size_t             entry_index = 0;
        size_t             slot_index  = 0;
        const ExpertPage * page        = nullptr;
        int                fd          = -1;
    };

    struct ReadResult {
        size_t                              miss_index = 0;
        std::unique_ptr<StagingPool::Lease> staging;
        std::exception_ptr                  error;
    };

    struct BatchState {
        std::vector<Miss>                       misses;
        std::atomic<size_t>                     next{0};
        std::mutex                              mutex;
        std::condition_variable                 cv;
        std::deque<std::unique_ptr<ReadResult>> ready;
        bool                                    start  = false;
        bool                                    cancel = false;
    };

public:
    ExpertSlotPool(
            ggml_backend_t backend, ResourcePlan resources,
            TestHooks * test_hooks) :
        backend_(backend),
        resources_(std::move(resources)),
        staging_(resources_),
        test_hooks_(test_hooks) {
        if (resources_.slot_count <= 0 ||
            resources_.staging_buffer_bytes == 0 ||
            resources_.staging_buffer_bytes >
                (uint64_t) std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("invalid expert slot pool dimensions");
        }
        slots_.reserve((size_t) resources_.slot_count);
        resources_.device_bytes = 0;
        for (const SlotClass & slot_class : resources_.slot_classes) {
            for (int i = 0; i < slot_class.slots; ++i) {
                slots_.push_back(make_slot(slot_class.size));
                resources_.device_bytes +=
                    ggml_backend_buffer_get_size(slots_.back().buffer.get());
            }
        }
        resources_.staging_buffers      = staging_.buffer_count();
        resources_.staging_buffer_bytes = staging_.buffer_bytes();
        resources_.staging_bytes =
            resources_.staging_buffer_bytes *
            (uint64_t) resources_.staging_buffers;
    }

    ~ExpertSlotPool() {
#if defined(__linux__)
        for (const auto & item : fds_) {
            close(item.second);
        }
#endif
    }

    struct Loaded {
        ggml_backend_buffer_t buffer = nullptr;
        void *                base   = nullptr;
    };

    class Batch {
    public:
        Batch(Batch && other) noexcept;
        ~Batch();

        Batch(const Batch &) = delete;
        Batch & operator=(const Batch &) = delete;
        Batch & operator=(Batch &&) = delete;

        bool is_hit(size_t index) const {
            return entries_.at(index).hit;
        }

        const Loaded & loaded(size_t index) const {
            const Entry & entry = entries_.at(index);
            if (!entry.ready) {
                throw std::logic_error("expert batch entry is not ready");
            }
            return entry.loaded;
        }

        void complete();

    private:
        friend class ExpertSlotPool;

        struct Entry {
            Loaded loaded;
            size_t slot_index = std::numeric_limits<size_t>::max();
            bool   hit        = false;
            bool   ready      = false;
        };

        explicit Batch(ExpertSlotPool * owner, size_t count) :
            owner_(owner), entries_(count) {
        }

        ExpertSlotPool *           owner_ = nullptr;
        std::vector<Entry>         entries_;
        std::shared_ptr<BatchState> state_;
        std::vector<std::thread>   workers_;
        bool                       completed_ = false;
    };

    Batch ensure_batch(const std::vector<const ExpertPage *> & pages) {
        Batch batch(this, pages.size());
        try {
            std::vector<size_t> misses;
            misses.reserve(pages.size());

            // Resolve and pin every hit before selecting a victim. A hit is
            // immediately usable while sibling misses are read.
            for (size_t i = 0; i < pages.size(); ++i) {
                const ExpertPage & page = *pages[i];
                const std::pair<int, int> key(page.layer, page.expert);
                size_t slot_index = slots_.size();
                for (size_t j = 0; j < slots_.size(); ++j) {
                    if (slots_[j].valid && slots_[j].key == key) {
                        slot_index = j;
                        break;
                    }
                }
                if (slot_index == slots_.size()) {
                    misses.push_back(i);
                    continue;
                }
                Slot & slot = slots_[slot_index];
                ++slot.pin_count;
                slot.tick = ++tick_;
                batch.entries_[i] = {
                    { slot.buffer.get(),
                      ggml_backend_buffer_get_base(slot.buffer.get()) },
                    slot_index,
                    true,
                    true,
                };
            }

            // Reserve and pin all miss slots before starting any read. Later
            // allocations in this request cannot select an earlier miss.
            for (size_t entry_index : misses) {
                const ExpertPage & page = *pages[entry_index];
                const size_t slot_index = select_victim(page.size);
                if (slot_index == slots_.size()) {
                    throw std::runtime_error(
                        "no expert slot can hold requested page");
                }
                ++slots_[slot_index].pin_count;
                batch.entries_[entry_index].slot_index = slot_index;
            }

            batch.state_ = std::make_shared<BatchState>();
            batch.state_->misses.reserve(misses.size());
            for (size_t entry_index : misses) {
                const ExpertPage & page = *pages[entry_index];
                const size_t slot_index =
                    batch.entries_[entry_index].slot_index;
                Slot & slot = slots_[slot_index];
                slot.valid = false;
                batch.state_->misses.push_back({
                    entry_index, slot_index, &page, fd_for(page.blob)
                });
                if (test_hooks_ != nullptr &&
                    test_hooks_->slot_reserved) {
                    test_hooks_->slot_reserved(
                        page.layer, page.expert, (int) slot_index);
                }
            }

            if (misses.empty()) {
                batch.completed_ = true;
                return batch;
            }

            const size_t worker_count = std::min<size_t>(
                misses.size(), (size_t) staging_.buffer_count());
            batch.workers_.reserve(worker_count);
            for (size_t i = 0; i < worker_count; ++i) {
                batch.workers_.emplace_back(
                    [this, state = batch.state_]() {
                        read_worker(state);
                    });
            }
            {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                batch.state_->start = true;
            }
            batch.state_->cv.notify_all();
            return batch;
        } catch (...) {
            cancel_workers(batch);
            release_pins(batch);
            throw;
        }
    }

    const ResourcePlan & resources() const {
        return resources_;
    }

private:
    struct Slot {
        context_ptr         ctx;
        buffer_ptr          buffer;
        ggml_tensor *       raw       = nullptr;
        std::pair<int, int> key;
        uint64_t            capacity  = 0;
        uint64_t            tick      = 0;
        int                 pin_count = 0;
        bool                valid     = false;
    };

    size_t select_victim(uint64_t page_size) const {
        size_t victim = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pin_count != 0 || slot.valid ||
                page_size > slot.capacity) {
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity) {
                victim = i;
            }
        }
        if (victim != slots_.size()) {
            return victim;
        }

        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pin_count != 0 || !slot.valid ||
                page_size > slot.capacity) {
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity ||
                (slot.capacity == slots_[victim].capacity &&
                 slot.tick < slots_[victim].tick)) {
                victim = i;
            }
        }
        return victim;
    }

    void read_worker(const std::shared_ptr<BatchState> & state) {
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&]() {
                return state->start || state->cancel;
            });
            if (state->cancel) {
                return;
            }
        }

        while (true) {
            const size_t miss_index =
                state->next.fetch_add(1, std::memory_order_relaxed);
            if (miss_index >= state->misses.size()) {
                return;
            }
            const Miss & miss = state->misses[miss_index];
            auto result = std::make_unique<ReadResult>();
            result->miss_index = miss_index;
            bool read_started = false;
            try {
                result->staging = std::make_unique<StagingPool::Lease>(
                    staging_.borrow());
                if (test_hooks_ != nullptr &&
                    test_hooks_->staging_borrowed) {
                    test_hooks_->staging_borrowed();
                }
                read_started = true;
                if (test_hooks_ != nullptr &&
                    test_hooks_->read_started) {
                    test_hooks_->read_started(
                        miss.page->layer, miss.page->expert);
                }
                read_page(*miss.page, miss.fd, result->staging->get());
            } catch (...) {
                result->error = std::current_exception();
            }
            if (read_started && test_hooks_ != nullptr &&
                test_hooks_->read_finished) {
                try {
                    test_hooks_->read_finished(
                        miss.page->layer, miss.page->expert);
                } catch (...) {
                    if (result->error == nullptr) {
                        result->error = std::current_exception();
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->ready.push_back(std::move(result));
            }
            state->cv.notify_one();
        }
    }

    void complete_batch(Batch & batch) {
        if (batch.completed_) {
            return;
        }

        std::exception_ptr first_error;
        size_t received = 0;
        while (received < batch.state_->misses.size()) {
            std::unique_ptr<ReadResult> result;
            {
                std::unique_lock<std::mutex> lock(batch.state_->mutex);
                batch.state_->cv.wait(lock, [&]() {
                    return !batch.state_->ready.empty();
                });
                result = std::move(batch.state_->ready.front());
                batch.state_->ready.pop_front();
            }

            const Miss & miss =
                batch.state_->misses[result->miss_index];
            if (result->error != nullptr) {
                if (first_error == nullptr) {
                    first_error = result->error;
                }
            } else {
                Slot & slot = slots_[miss.slot_index];
                ggml_backend_tensor_set(
                    slot.raw, result->staging->get(), 0,
                    (size_t) miss.page->size);
                slot.valid = true;
                slot.key   = {
                    miss.page->layer, miss.page->expert
                };
                slot.tick  = ++tick_;
                Batch::Entry & entry =
                    batch.entries_[miss.entry_index];
                entry.loaded = {
                    slot.buffer.get(),
                    ggml_backend_buffer_get_base(slot.buffer.get())
                };
                entry.ready = true;
            }
            result.reset();
            ++received;
        }

        for (std::thread & worker : batch.workers_) {
            worker.join();
        }
        batch.workers_.clear();
        batch.completed_ = true;
        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
        }
    }

    void cancel_workers(Batch & batch) {
        if (batch.state_ != nullptr && !batch.workers_.empty()) {
            {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                batch.state_->cancel = true;
            }
            batch.state_->cv.notify_all();
            for (std::thread & worker : batch.workers_) {
                worker.join();
            }
            batch.workers_.clear();
        }
    }

    void release_pins(Batch & batch) noexcept {
        for (const Batch::Entry & entry : batch.entries_) {
            if (entry.slot_index == std::numeric_limits<size_t>::max()) {
                continue;
            }
            Slot & slot = slots_[entry.slot_index];
            if (slot.pin_count > 0) {
                --slot.pin_count;
            }
        }
        batch.entries_.clear();
    }

    void abandon_batch(Batch & batch) noexcept {
        if (!batch.completed_) {
            try {
                complete_batch(batch);
            } catch (...) {
            }
        }
        release_pins(batch);
    }

    Slot make_slot(uint64_t capacity) {
        if (capacity == 0 ||
            capacity > (uint64_t) std::numeric_limits<size_t>::max() ||
            capacity > (uint64_t) std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error("invalid expert slot capacity");
        }

        Slot slot;
        slot.capacity = capacity;
        slot.buffer.reset(ggml_backend_alloc_buffer(backend_, (size_t) capacity));
        if (!slot.buffer) {
            throw std::runtime_error("failed to allocate device expert slot");
        }
        ggml_backend_buffer_set_usage(slot.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 2,
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        slot.ctx.reset(ggml_init(params));
        if (!slot.ctx) {
            throw std::runtime_error("failed to allocate expert slot metadata");
        }
        slot.raw = ggml_new_tensor_1d(
            slot.ctx.get(), GGML_TYPE_I8, (int64_t) capacity);
        slot.raw->buffer = slot.buffer.get();
        slot.raw->data   = ggml_backend_buffer_get_base(slot.buffer.get());
        if (slot.raw->data == nullptr ||
            ggml_backend_buffer_init_tensor(slot.buffer.get(), slot.raw) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize expert slot tensor");
        }
        return slot;
    }

    int fd_for(const fs::path & path) {
#if defined(__linux__)
        const std::string key = path.string();
        auto it = fds_.find(key);
        if (it != fds_.end()) {
            return it->second;
        }
        const int fd = open(key.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "failed to open O_DIRECT shard " + key + ": " + std::strerror(errno));
        }
        fds_.emplace(key, fd);
        return fd;
#else
        (void) path;
        throw std::runtime_error("O_DIRECT expert slots require Linux");
#endif
    }

    void read_page(const ExpertPage & page, int fd, void * dst) {
#if defined(__linux__)
        ssize_t n = -1;
        do {
            n = pread(fd, dst, (size_t) page.size, (off_t) page.offset);
        } while (n < 0 && errno == EINTR);
        if (n < 0 || (uint64_t) n != page.size) {
            throw std::runtime_error(
                "short O_DIRECT expert read from " + page.blob.string() +
                ": got " + std::to_string(n) + " want " + std::to_string(page.size));
        }
#else
        (void) page;
        (void) fd;
        (void) dst;
#endif
    }

    ggml_backend_t             backend_ = nullptr;
    ResourcePlan               resources_;
    StagingPool                staging_;
    TestHooks *                test_hooks_ = nullptr;
    uint64_t                   tick_ = 0;
    std::vector<Slot>          slots_;
    std::map<std::string, int> fds_;
};

ExpertSlotPool::Batch::Batch(Batch && other) noexcept :
    owner_(other.owner_),
    entries_(std::move(other.entries_)),
    state_(std::move(other.state_)),
    workers_(std::move(other.workers_)),
    completed_(other.completed_) {
    other.owner_ = nullptr;
}

ExpertSlotPool::Batch::~Batch() {
    if (owner_ != nullptr) {
        owner_->abandon_batch(*this);
    }
}

void ExpertSlotPool::Batch::complete() {
    if (owner_ == nullptr) {
        throw std::logic_error("expert batch has no owner");
    }
    owner_->complete_batch(*this);
}

void attach_weight(
        ggml_tensor * tensor, ggml_backend_buffer_t buffer, void * base,
        uint64_t offset) {
    tensor->buffer = buffer;
    tensor->data   = (uint8_t *) base + offset;
    if (ggml_backend_buffer_init_tensor(buffer, tensor) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("failed to attach expert weight tensor to slot");
    }
}

std::vector<float> compute_expert(
        ggml_backend_t backend,
        const Descriptor & descriptor,
        const ExpertPage & page,
        const ExpertSlotPool::Loaded & loaded,
        const std::vector<float> & activation,
        uint32_t n_tokens) {
    const int n_embd   = descriptor.hparams.n_embd;
    const auto & specs = descriptor.layers.at(page.layer);
    const size_t tensor_count = 24;
    const ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                          ggml_graph_overhead_custom(64, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    context_ptr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate expert graph metadata");
    }

    auto make_weight = [&](const std::string & role) {
        const RoleSpec & spec = specs.at(role);
        ggml_tensor * tensor =
            ggml_new_tensor_2d(ctx.get(), spec.type, spec.ne0, spec.ne1);
        attach_weight(tensor, loaded.buffer, loaded.base, page.roles.at(role).offset);
        return tensor;
    };

    ggml_tensor * gate_weight = make_weight("gate");
    ggml_tensor * up_weight   = make_weight("up");
    ggml_tensor * down_weight = make_weight("down");
    ggml_tensor * input =
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(input);

    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_weight, input);
    ggml_tensor * up   = ggml_mul_mat(ctx.get(), up_weight, input);
    ggml_tensor * hidden = ggml_swiglu_split(ctx.get(), gate, up);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), down_weight, hidden);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    buffer_ptr compute_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!compute_buffer) {
        throw std::runtime_error("failed to allocate expert compute graph");
    }
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("expert backend graph compute failed");
    }
    std::vector<float> result((size_t) n_embd * n_tokens);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    return result;
}

class Worker {
public:
    Worker(
            Catalog catalog,
            const std::string & device,
            int slots,
            uint64_t host_budget_bytes,
            TestHooks * test_hooks) :
        catalog_(std::move(catalog)),
        backend_(init_backend(device)),
        pool_(
            backend_.get(),
            plan_resources(
                resource_pages(catalog_), slots, host_budget_bytes),
            test_hooks),
        slots_(pool_.resources().slot_count) {
    }

    pipe_expert_hello hello() const {
        pipe_expert_hello hello;
        hello.role          = PIPE_EXPERT_ROLE_WORKER;
        hello.hidden_type   = PIPE_HIDDEN_F16;
        hello.n_embd        = catalog_.descriptor.hparams.n_embd;
        hello.n_ff_exp      = catalog_.descriptor.hparams.n_ff_exp;
        hello.n_expert      = catalog_.descriptor.hparams.n_expert;
        hello.n_expert_used = catalog_.descriptor.hparams.n_expert_used;
        hello.expert_first  = catalog_.descriptor.expert_first;
        hello.expert_last   = catalog_.descriptor.expert_last;
        hello.n_slots       = (uint32_t) slots_;
        hello.layers        = catalog_.layers;
        hello.model_identity = source_model_identity(
            catalog_.descriptor.input_model, catalog_.descriptor.model_files);
        hello.shard_identity =
            catalog_.descriptor.identity_algorithm + ":" +
            catalog_.descriptor.identity_value;
        return hello;
    }

    pipe_expert_partial dispatch(const pipe_expert_dispatch_req & request) {
        if (!std::binary_search(catalog_.layers.begin(), catalog_.layers.end(), request.layer)) {
            throw pipe_protocol_error(
                PIPE_ERR_EXPERT_LAYER,
                "worker does not serve layer " + std::to_string(request.layer));
        }
        if (request.assignments.size() >
            (size_t) catalog_.descriptor.hparams.n_expert) {
            throw pipe_protocol_error(
                PIPE_ERR_BAD_FRAME,
                "expert dispatch has more assignments than model experts");
        }
        std::set<int32_t> seen_experts;
        for (const pipe_expert_assignment & assignment : request.assignments) {
            if (!seen_experts.insert(assignment.expert_id).second) {
                throw pipe_protocol_error(
                    PIPE_ERR_BAD_FRAME,
                    "expert dispatch repeats expert " + std::to_string(assignment.expert_id));
            }
            if (assignment.expert_id < catalog_.descriptor.expert_first ||
                assignment.expert_id > catalog_.descriptor.expert_last ||
                catalog_.pages.count({ request.layer, assignment.expert_id }) == 0) {
                throw pipe_protocol_error(
                    PIPE_ERR_EXPERT_RANGE,
                    "worker does not serve expert " + std::to_string(assignment.expert_id));
            }
        }

        std::vector<float> activation(request.activations.size());
        for (size_t i = 0; i < request.activations.size(); ++i) {
            activation[i] = ggml_fp16_to_fp32((ggml_fp16_t) request.activations[i]);
        }
        std::vector<const ExpertPage *> pages;
        pages.reserve(request.assignments.size());
        for (const pipe_expert_assignment & assignment : request.assignments) {
            pages.push_back(&catalog_.pages.at({
                request.layer, assignment.expert_id
            }));
        }
        ExpertSlotPool::Batch batch = pool_.ensure_batch(pages);

        std::vector<std::vector<float>> values(request.assignments.size());
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            if (!batch.is_hit(i)) {
                continue;
            }
            values[i] = compute_expert(
                backend_.get(), catalog_.descriptor, *pages[i],
                batch.loaded(i), activation, request.n_tokens);
        }
        batch.complete();
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            if (batch.is_hit(i)) {
                continue;
            }
            values[i] = compute_expert(
                backend_.get(), catalog_.descriptor, *pages[i],
                batch.loaded(i), activation, request.n_tokens);
        }

        std::vector<float> sum(
            (size_t) request.n_tokens * catalog_.descriptor.hparams.n_embd, 0.0f);
        for (size_t j = 0; j < request.assignments.size(); ++j) {
            const pipe_expert_assignment & assignment =
                request.assignments[j];
            const std::vector<float> & value = values[j];
            for (uint32_t token = 0; token < request.n_tokens; ++token) {
                const float weight = assignment.weights[token];
                const size_t base =
                    (size_t) token * catalog_.descriptor.hparams.n_embd;
                for (int i = 0; i < catalog_.descriptor.hparams.n_embd; ++i) {
                    sum[base + (size_t) i] += weight * value[base + (size_t) i];
                }
            }
        }

        pipe_expert_partial response;
        response.layer    = request.layer;
        response.n_tokens = request.n_tokens;
        response.partial.resize(sum.size());
        for (size_t i = 0; i < sum.size(); ++i) {
            response.partial[i] = (uint16_t) ggml_fp32_to_fp16(sum[i]);
        }
        return response;
    }

    const ResourcePlan & resources() const {
        return pool_.resources();
    }

private:
    Catalog        catalog_;
    backend_ptr    backend_;
    ExpertSlotPool pool_;
    int            slots_ = 0;
};

bool validate_client_hello(
        const pipe_expert_hello & client, const pipe_expert_hello & worker,
        std::string & error) {
    if (client.role != PIPE_EXPERT_ROLE_CLIENT) {
        error = "peer did not identify as an expert client";
    } else if (client.hidden_type != worker.hidden_type ||
               client.n_embd != worker.n_embd ||
               client.n_ff_exp != worker.n_ff_exp ||
               client.n_expert != worker.n_expert ||
               client.n_expert_used != worker.n_expert_used) {
        error = "expert HELLO hparams mismatch";
    } else if (client.model_identity != worker.model_identity) {
        error = "expert HELLO model identity mismatch";
    } else if (client.shard_identity != worker.shard_identity) {
        error = "expert HELLO shard identity mismatch";
    }
    return error.empty();
}

bool send_hello_ack(
        pipe_socket_t & socket, bool accepted, const std::string & reason) {
    const std::vector<uint8_t> payload =
        pipe_encode_expert_hello_ack({ accepted, reason });
    return pipe_send_frame(
        socket, PIPE_EXPERT_HELLO_ACK, 0, payload.data(), payload.size());
}

int serve_connection(pipe_socket_t & socket, Worker & worker) {
    const pipe_expert_hello mine = worker.hello();
    const std::vector<uint8_t> hello_payload = pipe_encode_expert_hello(mine);
    if (!pipe_send_frame(
            socket, PIPE_HELLO, 0, hello_payload.data(), hello_payload.size())) {
        return 1;
    }

    pipe_frame_type type;
    uint64_t seq_id = 0;
    std::vector<uint8_t> payload;
    if (!pipe_recv_frame(socket, type, seq_id, payload)) {
        return 1;
    }
    if (type != PIPE_HELLO || seq_id != 0) {
        send_hello_ack(socket, false, "expected expert HELLO");
        return 1;
    }
    try {
        const pipe_expert_hello client =
            pipe_decode_expert_hello(payload.data(), payload.size());
        std::string error;
        if (!validate_client_hello(client, mine, error)) {
            send_hello_ack(socket, false, error);
            return 1;
        }
    } catch (const pipe_protocol_error & error) {
        send_hello_ack(socket, false, error.what());
        return 1;
    }
    if (!send_hello_ack(socket, true, "")) {
        return 1;
    }

    while (pipe_recv_frame(socket, type, seq_id, payload)) {
        if (type == PIPE_PING) {
            if (!pipe_send_frame(socket, PIPE_PONG, seq_id, nullptr, 0)) {
                return 1;
            }
            continue;
        }
        if (type != PIPE_EXPERT_DISPATCH_REQ) {
            pipe_send_error(socket, seq_id, PIPE_ERR_BAD_FRAME, "expected expert dispatch request");
            return 1;
        }
        try {
            const pipe_expert_dispatch_req request =
                pipe_decode_expert_dispatch_req(
                    payload.data(), payload.size(), mine.n_embd);
            const pipe_expert_partial response = worker.dispatch(request);
            const std::vector<uint8_t> encoded =
                pipe_encode_expert_partial(response);
            if (!pipe_send_frame(
                    socket, PIPE_EXPERT_PARTIAL, seq_id,
                    encoded.data(), encoded.size())) {
                return 1;
            }
        } catch (const pipe_protocol_error & error) {
            if (!pipe_send_error(socket, seq_id, error.code, error.what())) {
                return 1;
            }
        } catch (const std::exception & error) {
            pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
            return 1;
        }
    }
    return 0;
}

} // namespace

ResourcePlan inspect_resources(const Options & options) {
    if (options.slots <= 0 || options.device.empty()) {
        throw std::invalid_argument("invalid expert worker resource options");
    }
    const fs::path manifest   = fs::canonical(options.shard_manifest);
    const fs::path descriptor = fs::canonical(options.descriptor);
    Worker worker(
        load_catalog(manifest, descriptor),
        options.device,
        options.slots,
        options.host_budget_bytes,
        options.test_hooks);
    return worker.resources();
}

int run(const Options & options) {
    if (options.slots <= 0 || options.listen_host.empty() ||
        options.listen_port <= 0 || options.listen_port > 65535 ||
        options.device.empty()) {
        throw std::invalid_argument("invalid expert worker options");
    }
    const fs::path manifest = fs::canonical(options.shard_manifest);
    const fs::path descriptor = fs::canonical(options.descriptor);
    Worker worker(
        load_catalog(manifest, descriptor),
        options.device,
        options.slots,
        options.host_budget_bytes,
        options.test_hooks);
    const pipe_expert_hello advertised = worker.hello();
    const ResourcePlan & resources = worker.resources();

    pipe_socket_ptr server =
        pipe_socket_t::create_server(options.listen_host.c_str(), options.listen_port);
    if (!server) {
        throw std::runtime_error(
            "failed to listen on " + options.listen_host + ":" +
            std::to_string(options.listen_port));
    }
    std::cout << "expert worker listening on " << options.listen_host << ":"
              << options.listen_port << " device=" << options.device
              << " experts=" << advertised.expert_first << "-"
              << advertised.expert_last << " layers=" << advertised.layers.size()
              << " slots=" << advertised.n_slots
              << " requested_slots=" << resources.requested_slots
              << " device_bytes=" << resources.device_bytes
              << " size_classes=" << (resources.size_classes ? 1 : 0)
              << " staging=" << resources.staging_buffers << "x"
              << resources.staging_buffer_bytes
              << " host_budget=" << resources.host_budget_bytes << '\n';
    for (const SlotClass & slot_class : resources.slot_classes) {
        std::cout << "expert slot class bytes=" << slot_class.size
                  << " slots=" << slot_class.slots
                  << " pin_floor=" << slot_class.pin_floor
                  << " pages=" << slot_class.pages << '\n';
    }

    int result = 0;
    do {
        pipe_socket_ptr client = server->accept();
        if (!client) {
            return 1;
        }
        result = serve_connection(*client, worker);
    } while (!options.once);
    return result;
}

} // namespace wp_expert_worker
