#include "wp-expert-shard-lib.h"

#include "nlohmann/json.hpp"

extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

namespace wp_expert_shard {
namespace {

constexpr const char * INDEX_FORMAT     = "llama.cpp.weight-pager.expert-shard-index";
constexpr const char * MANIFEST_FORMAT  = "llama.cpp.weight-pager.expert-shard-manifest";
constexpr int          FORMAT_VERSION   = 1;
constexpr size_t       COPY_BUFFER_SIZE = 8u * 1024u * 1024u;

struct ShardPaths {
    fs::path blob;
    fs::path index;
};

struct PlannedShard {
    fs::path          source_blob;
    int               layer        = -1;
    size_t            output_index = 0;
    uint64_t          blob_bytes   = 0;
    std::vector<json> groups;
    std::string       content_hash;
};

struct Plan {
    fs::path                  out_base;
    json                      source_manifest;
    std::vector<PlannedShard> shards;
};

json read_json(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    try {
        json value;
        input >> value;
        return value;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to parse " + path.string() + ": " + error.what());
    }
}

template <typename T> T get_value(const json & value, const char * key, const fs::path & path) {
    const auto it = value.find(key);
    if (it == value.end()) {
        throw std::runtime_error(path.string() + " is missing required field '" + key + "'");
    }
    try {
        return it->get<T>();
    } catch (const std::exception & error) {
        throw std::runtime_error(path.string() + " field '" + key + "' has the wrong type: " + error.what());
    }
}

const json & get_array(const json & value, const char * key, const fs::path & path) {
    const auto it = value.find(key);
    if (it == value.end() || !it->is_array()) {
        throw std::runtime_error(path.string() + " is missing array field '" + key + "'");
    }
    return *it;
}

void check_format(const json & value, const char * format, const fs::path & path) {
    if (get_value<std::string>(value, "format", path) != format ||
        get_value<int>(value, "version", path) != FORMAT_VERSION) {
        throw std::runtime_error("unsupported or invalid weight-pager metadata format in " + path.string());
    }
}

uint64_t checked_add(uint64_t left, uint64_t right, const std::string & what) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error(what + " byte count overflows uint64");
    }
    return left + right;
}

void sha_update_u64(sha256_t & hash, uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (i * 8)) & 0xffu);
    }
    sha256_update(&hash, bytes.data(), bytes.size());
}

void sha_update_string(sha256_t & hash, const std::string & value) {
    sha_update_u64(hash, value.size());
    sha256_update(&hash, reinterpret_cast<const unsigned char *>(value.data()), value.size());
}

void sha_update_group(sha256_t & hash, const json & group) {
    sha_update_u64(hash, static_cast<uint64_t>(group.at("block_idx").get<int>()));
    sha_update_u64(hash, static_cast<uint64_t>(group.at("expert_idx").get<int>()));
    const json & members = group.at("members");
    sha_update_u64(hash, members.size());
    for (const json & member : members) {
        sha_update_u64(hash, member.at("role_mask").get<uint64_t>());
        sha_update_u64(hash, member.at("size").get<uint64_t>());
        sha_update_string(hash, member.at("catalog_name").get<std::string>());
        sha_update_string(hash, member.at("source_tensor_name").get<std::string>());
    }
}

std::string finish_sha(sha256_t & hash) {
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_final(&hash, digest.data());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

std::string hash_groups(const std::vector<json> & groups) {
    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-repack.identity.v1");
    sha_update_u64(hash, groups.size());
    for (const json & group : groups) {
        sha_update_group(hash, group);
    }
    return finish_sha(hash);
}

std::string hash_plan(const std::vector<PlannedShard> & shards) {
    uint64_t group_count = 0;
    for (const PlannedShard & shard : shards) {
        group_count = checked_add(group_count, shard.groups.size(), "manifest group");
    }

    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-repack.identity.v1");
    sha_update_u64(hash, group_count);
    for (const PlannedShard & shard : shards) {
        for (const json & group : shard.groups) {
            sha_update_group(hash, group);
        }
    }
    return finish_sha(hash);
}

std::string numbered_name(const std::string & base, size_t index, size_t total) {
    std::ostringstream name;
    name << base << "-experts-" << std::setw(5) << std::setfill('0') << index + 1 << "-of-" << std::setw(5)
         << std::setfill('0') << total;
    return name.str();
}

ShardPaths shard_paths(const fs::path & out_base, size_t index, size_t total) {
    const std::string prefix = numbered_name(out_base.string(), index, total);
    return { fs::path(prefix + ".wpb"), fs::path(prefix + ".wpi.json") };
}

void require_absent(const fs::path & path) {
    if (fs::exists(path)) {
        throw std::runtime_error("refusing to overwrite existing output: " + path.string());
    }
}

void write_json(const fs::path & path, const json & value) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create " + path.string());
    }
    output << value.dump(2) << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

std::string serialize_json(const json & value) {
    return value.dump(2) + '\n';
}

void verify_existing_json(const fs::path & path, const json & value) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (input.bad()) {
        throw std::runtime_error("failed to read " + path.string());
    }
    if (contents.str() != serialize_json(value)) {
        throw std::runtime_error("refusing to overwrite existing output that differs from planned metadata: " +
                                 path.string());
    }
}

void write_or_verify_json(const fs::path & path, const json & value) {
    const fs::path temp_path = fs::path(path.string() + ".tmp");
    require_absent(temp_path);
    if (fs::exists(path)) {
        verify_existing_json(path, value);
        return;
    }

    try {
        write_json(temp_path, value);
        fs::rename(temp_path, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temp_path, ignored);
        throw;
    }
}

Plan build_plan(const Options & options) {
    const fs::path source_manifest_path = fs::canonical(options.src_manifest);
    const fs::path source_dir           = source_manifest_path.parent_path();
    json           manifest             = read_json(source_manifest_path);
    check_format(manifest, MANIFEST_FORMAT, source_manifest_path);

    get_value<std::string>(manifest, "input_model", source_manifest_path);
    const json & model_files = get_array(manifest, "model_files", source_manifest_path);
    if (model_files.empty()) {
        throw std::runtime_error(source_manifest_path.string() + " has no model_files");
    }

    const json & source_shards = get_array(manifest, "shards", source_manifest_path);
    const size_t source_count  = get_value<uint64_t>(manifest, "shard_count", source_manifest_path);
    if (source_shards.empty() || source_shards.size() != source_count) {
        throw std::runtime_error("source manifest shard count mismatch");
    }

    std::vector<std::optional<PlannedShard>> selected_by_source(source_count);
    std::vector<bool>                        seen_source_indices(source_count, false);
    std::set<std::pair<int, int>>            seen_groups;
    std::set<int>                            seen_layers;
    uint64_t                                 source_total_groups = 0;
    uint64_t                                 source_total_bytes  = 0;

    for (const json & source_shard : source_shards) {
        const int source_index = get_value<int>(source_shard, "shard_index", source_manifest_path);
        if (source_index < 0 || static_cast<size_t>(source_index) >= source_count ||
            seen_source_indices[source_index]) {
            throw std::runtime_error("source manifest has an invalid or repeated shard_index");
        }
        seen_source_indices[source_index] = true;

        const fs::path index_path =
            source_dir / get_value<std::string>(source_shard, "index_file", source_manifest_path);
        const fs::path blob_path = source_dir / get_value<std::string>(source_shard, "blob_file", source_manifest_path);
        const json     index     = read_json(index_path);
        check_format(index, INDEX_FORMAT, index_path);

        const int layer_first = get_value<int>(index, "layer_first", index_path);
        const int layer_last  = get_value<int>(index, "layer_last", index_path);
        if (layer_first < 0 || layer_first != layer_last) {
            throw std::runtime_error("expert-index sharding requires one source layer per shard: " +
                                     index_path.string());
        }
        if (!seen_layers.insert(layer_first).second) {
            throw std::runtime_error("source manifest repeats layer " + std::to_string(layer_first));
        }

        const uint64_t group_count = get_value<uint64_t>(index, "group_count", index_path);
        const uint64_t blob_bytes  = get_value<uint64_t>(index, "blob_bytes", index_path);
        if (get_value<int>(index, "shard_index", index_path) != source_index ||
            get_value<uint64_t>(index, "shard_count", index_path) != source_count ||
            get_value<std::string>(index, "blob_file", index_path) != blob_path.filename().string() ||
            get_value<int>(source_shard, "layer_first", source_manifest_path) != layer_first ||
            get_value<int>(source_shard, "layer_last", source_manifest_path) != layer_last ||
            get_value<uint64_t>(source_shard, "group_count", source_manifest_path) != group_count ||
            get_value<uint64_t>(source_shard, "blob_bytes", source_manifest_path) != blob_bytes) {
            throw std::runtime_error("source manifest and sidecar disagree for " + index_path.string());
        }
        if (get_array(index, "model_files", index_path) != model_files) {
            throw std::runtime_error("source manifest and sidecar model_files disagree for " + index_path.string());
        }

        std::error_code size_error;
        const uint64_t  actual_blob_bytes = fs::file_size(blob_path, size_error);
        if (size_error || actual_blob_bytes != blob_bytes) {
            throw std::runtime_error("source blob size mismatch for " + blob_path.string());
        }

        const json & groups = get_array(index, "groups", index_path);
        if (groups.size() != group_count) {
            throw std::runtime_error("source group count mismatch in " + index_path.string());
        }

        PlannedShard planned;
        planned.source_blob         = blob_path;
        planned.layer               = layer_first;
        uint64_t source_next_offset = 0;

        for (const json & group : groups) {
            const int block_idx  = get_value<int>(group, "block_idx", index_path);
            const int expert_idx = get_value<int>(group, "expert_idx", index_path);
            if (block_idx != layer_first || expert_idx < 0 || !seen_groups.insert({ block_idx, expert_idx }).second) {
                throw std::runtime_error("invalid or repeated source expert group in " + index_path.string());
            }

            const json & members = get_array(group, "members", index_path);
            if (members.size() != 3 || get_value<uint64_t>(group, "member_count", index_path) != members.size()) {
                throw std::runtime_error("source expert group does not contain exactly three members in " +
                                         index_path.string());
            }

            uint64_t group_bytes = 0;
            for (const json & member : members) {
                const uint64_t offset = get_value<uint64_t>(member, "offset", index_path);
                const uint64_t size   = get_value<uint64_t>(member, "size", index_path);
                get_value<uint64_t>(member, "role_mask", index_path);
                get_value<std::string>(member, "catalog_name", index_path);
                get_value<std::string>(member, "source_tensor_name", index_path);
                get_value<uint64_t>(member, "source_file_idx", index_path);
                get_value<uint64_t>(member, "source_file_offset", index_path);
                if (size == 0 || offset != source_next_offset) {
                    throw std::runtime_error("source groups are not gapless and contiguous in " + index_path.string());
                }
                source_next_offset = checked_add(source_next_offset, size, "source blob");
                group_bytes        = checked_add(group_bytes, size, "source group");
            }

            if (expert_idx >= options.expert_first && expert_idx <= options.expert_last) {
                planned.groups.push_back(group);
                planned.blob_bytes = checked_add(planned.blob_bytes, group_bytes, "output shard");
            }
        }

        if (source_next_offset != blob_bytes) {
            throw std::runtime_error("source groups do not cover the recorded blob size in " + index_path.string());
        }
        source_total_groups = checked_add(source_total_groups, groups.size(), "source manifest group");
        source_total_bytes  = checked_add(source_total_bytes, source_next_offset, "source manifest");

        if (!planned.groups.empty()) {
            selected_by_source[source_index] = std::move(planned);
        } else {
            std::cout << "skipping layer " << layer_first << ": no groups in retained expert range\n";
        }
    }

    if (get_value<uint64_t>(manifest, "total_group_count", source_manifest_path) != source_total_groups ||
        get_value<uint64_t>(manifest, "total_blob_bytes", source_manifest_path) != source_total_bytes) {
        throw std::runtime_error("source manifest totals do not match its sidecars");
    }

    Plan plan;
    plan.out_base        = fs::absolute(options.out_base).lexically_normal();
    plan.source_manifest = std::move(manifest);
    for (std::optional<PlannedShard> & selected : selected_by_source) {
        if (selected.has_value()) {
            selected->output_index = plan.shards.size();
            selected->content_hash = hash_groups(selected->groups);
            plan.shards.push_back(std::move(*selected));
        }
    }
    if (plan.shards.empty()) {
        throw std::runtime_error("the requested expert range retains no groups");
    }
    return plan;
}

json build_output_index(const Plan & plan, const PlannedShard & shard) {
    const ShardPaths paths = shard_paths(plan.out_base, shard.output_index, plan.shards.size());
    json             index = {
        { "format",       INDEX_FORMAT                           },
        { "version",      FORMAT_VERSION                         },
        { "blob_file",    paths.blob.filename().string()         },
        { "shard_index",  shard.output_index                     },
        { "shard_count",  plan.shards.size()                     },
        { "layer_first",  shard.layer                            },
        { "layer_last",   shard.layer                            },
        { "group_count",  shard.groups.size()                    },
        { "blob_bytes",   shard.blob_bytes                       },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", shard.content_hash },
          }                                                      },
        { "model_files",  plan.source_manifest.at("model_files") },
        { "groups",       json::array()                          },
    };

    uint64_t output_offset = 0;
    for (const json & source_group : shard.groups) {
        json output_group       = source_group;
        output_group["members"] = json::array();
        for (const json & source_member : source_group.at("members")) {
            json output_member      = source_member;
            output_member["offset"] = output_offset;
            output_offset = checked_add(output_offset, source_member.at("size").get<uint64_t>(), "output blob");
            output_group["members"].push_back(std::move(output_member));
        }
        index["groups"].push_back(std::move(output_group));
    }
    if (output_offset != shard.blob_bytes) {
        throw std::runtime_error("internal output byte count mismatch");
    }
    return index;
}

void copy_member(std::ifstream &     source,
                 uint64_t            source_offset,
                 uint64_t            size,
                 std::ofstream &     output,
                 std::vector<char> & buffer) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(source_offset));
    if (!source) {
        throw std::runtime_error("failed to seek source blob");
    }

    uint64_t remaining = size;
    while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
        source.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (source.gcount() != static_cast<std::streamsize>(chunk)) {
            throw std::runtime_error("short read from source blob");
        }
        output.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!output) {
            throw std::runtime_error("failed to write output blob");
        }
        remaining -= chunk;
    }
}

void compare_member_bytes(std::ifstream &     source,
                          uint64_t            source_offset,
                          std::ifstream &     output,
                          uint64_t            output_offset,
                          uint64_t            size,
                          const std::string & name,
                          std::vector<char> & source_buffer,
                          std::vector<char> & output_buffer) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(source_offset));
    output.clear();
    output.seekg(static_cast<std::streamoff>(output_offset));
    if (!source || !output) {
        throw std::runtime_error("failed to seek while verifying " + name);
    }

    uint64_t remaining = size;
    uint64_t compared  = 0;
    while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, source_buffer.size()));
        source.read(source_buffer.data(), static_cast<std::streamsize>(chunk));
        output.read(output_buffer.data(), static_cast<std::streamsize>(chunk));
        if (source.gcount() != static_cast<std::streamsize>(chunk) ||
            output.gcount() != static_cast<std::streamsize>(chunk)) {
            throw std::runtime_error("short read while verifying " + name);
        }
        if (std::memcmp(source_buffer.data(), output_buffer.data(), chunk) != 0) {
            throw std::runtime_error("payload byte mismatch for " + name + " at member byte " +
                                     std::to_string(compared));
        }
        remaining -= chunk;
        compared += chunk;
    }
}

RunStats verify_output(const PlannedShard & shard,
                       const json &         expected_index,
                       const fs::path &     index_path,
                       const fs::path &     blob_path) {
    const json actual_index = read_json(index_path);
    if (actual_index != expected_index) {
        throw std::runtime_error("emitted sidecar differs from the planned metadata");
    }

    std::error_code size_error;
    const uint64_t  actual_size = fs::file_size(blob_path, size_error);
    if (size_error || actual_size != shard.blob_bytes || actual_index.at("blob_bytes").get<uint64_t>() != actual_size) {
        throw std::runtime_error("emitted blob_bytes does not match the output file size");
    }

    std::ifstream source(shard.source_blob, std::ios::binary);
    std::ifstream output(blob_path, std::ios::binary);
    if (!source || !output) {
        throw std::runtime_error("failed to reopen blobs for verification");
    }

    RunStats          stats;
    std::vector<char> source_buffer(COPY_BUFFER_SIZE);
    std::vector<char> output_buffer(COPY_BUFFER_SIZE);
    uint64_t          output_offset = 0;
    const json &      output_groups = actual_index.at("groups");
    for (size_t group_index = 0; group_index < shard.groups.size(); ++group_index) {
        const json & source_group   = shard.groups[group_index];
        const json & output_group   = output_groups.at(group_index);
        const json & source_members = source_group.at("members");
        const json & output_members = output_group.at("members");
        if (source_members.size() != output_members.size()) {
            throw std::runtime_error("emitted member count mismatch");
        }

        for (size_t member_index = 0; member_index < source_members.size(); ++member_index) {
            const json &   source_member = source_members.at(member_index);
            const json &   output_member = output_members.at(member_index);
            const uint64_t size          = source_member.at("size").get<uint64_t>();
            if (output_member.at("offset").get<uint64_t>() != output_offset ||
                output_member.at("size").get<uint64_t>() != size ||
                output_member.at("catalog_name") != source_member.at("catalog_name")) {
                throw std::runtime_error("emitted member metadata or contiguity mismatch");
            }
            compare_member_bytes(source, source_member.at("offset").get<uint64_t>(), output, output_offset, size,
                                 source_member.at("catalog_name").get<std::string>(), source_buffer, output_buffer);
            output_offset = checked_add(output_offset, size, "verified output");
            ++stats.members;
            stats.bytes += size;
        }
        ++stats.groups;
    }
    if (output_offset != actual_size) {
        throw std::runtime_error("emitted groups do not cover the output blob");
    }
    stats.shards = 1;
    return stats;
}

void add_stats(RunStats & total, const RunStats & value) {
    total.shards += value.shards;
    total.groups += value.groups;
    total.members += value.members;
    total.bytes += value.bytes;
}

RunStats write_shard(const Plan & plan, const PlannedShard & shard, bool verify) {
    const ShardPaths paths      = shard_paths(plan.out_base, shard.output_index, plan.shards.size());
    const fs::path   temp_blob  = fs::path(paths.blob.string() + ".tmp");
    const fs::path   temp_index = fs::path(paths.index.string() + ".tmp");
    require_absent(paths.blob);
    require_absent(temp_blob);
    require_absent(temp_index);

    const json index        = build_output_index(plan, shard);
    const bool index_exists = fs::exists(paths.index);
    if (index_exists) {
        verify_existing_json(paths.index, index);
    }
    try {
        std::ifstream source(shard.source_blob, std::ios::binary);
        std::ofstream output(temp_blob, std::ios::binary);
        if (!source || !output) {
            throw std::runtime_error("failed to open source or output blob for layer " + std::to_string(shard.layer));
        }

        std::vector<char> buffer(COPY_BUFFER_SIZE);
        for (const json & group : shard.groups) {
            for (const json & member : group.at("members")) {
                copy_member(source, member.at("offset").get<uint64_t>(), member.at("size").get<uint64_t>(), output,
                            buffer);
            }
        }
        output.close();
        if (!output || fs::file_size(temp_blob) != shard.blob_bytes) {
            throw std::runtime_error("failed to finish output blob for layer " + std::to_string(shard.layer));
        }
        if (!index_exists) {
            write_json(temp_index, index);
        }

        RunStats stats;
        if (verify) {
            stats = verify_output(shard, index, index_exists ? paths.index : temp_index, temp_blob);
        } else {
            stats.shards  = 1;
            stats.groups  = shard.groups.size();
            stats.members = shard.groups.size() * 3;
            stats.bytes   = shard.blob_bytes;
        }

        fs::rename(temp_blob, paths.blob);
        if (!index_exists) {
            fs::rename(temp_index, paths.index);
        }
        if (verify) {
            std::cout << "verify PASS: layer=" << shard.layer << " groups=" << stats.groups
                      << " members=" << stats.members << " bytes=" << stats.bytes << '\n';
        }
        std::cout << "wrote layer " << shard.layer << ": groups=" << stats.groups << " bytes=" << stats.bytes
                  << " blob=" << paths.blob.string() << '\n';
        return stats;
    } catch (...) {
        std::error_code ignored;
        fs::remove(temp_blob, ignored);
        fs::remove(temp_index, ignored);
        throw;
    }
}

RunStats plan_stats(const Plan & plan) {
    RunStats stats;
    stats.shards = plan.shards.size();
    for (const PlannedShard & shard : plan.shards) {
        stats.groups += shard.groups.size();
        stats.members += shard.groups.size() * 3;
        stats.bytes += shard.blob_bytes;
    }
    return stats;
}

RunStats write_manifest(const Plan & plan, const Options & options) {
    const fs::path manifest_path = output_manifest_path(plan.out_base);

    const RunStats stats    = plan_stats(plan);
    json           manifest = {
        { "format",                plan.source_manifest.at("format")      },
        { "version",               plan.source_manifest.at("version")     },
        { "input_model",           plan.source_manifest.at("input_model") },
        { "model_files",           plan.source_manifest.at("model_files") },
        { "sharding_mode",         "expert-index-range"                   },
        { "retained_expert_range",
         {
              { "first", options.expert_first },
              { "last", options.expert_last },
          }                                                               },
        { "total_group_count",     stats.groups                           },
        { "total_blob_bytes",      stats.bytes                            },
        { "shard_count",           stats.shards                           },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_plan(plan.shards) },
          }                                                               },
        { "shards",                json::array()                          },
    };

    for (const PlannedShard & shard : plan.shards) {
        const ShardPaths paths = shard_paths(plan.out_base, shard.output_index, plan.shards.size());
        manifest["shards"].push_back({
            { "blob_file",    paths.blob.filename().string()  },
            { "index_file",   paths.index.filename().string() },
            { "shard_index",  shard.output_index              },
            { "layer_first",  shard.layer                     },
            { "layer_last",   shard.layer                     },
            { "group_count",  shard.groups.size()             },
            { "blob_bytes",   shard.blob_bytes                },
            { "content_hash",
             {
                  { "algorithm", "sha256" },
                  { "value", shard.content_hash },
              }                                               },
        });
    }

    for (const PlannedShard & shard : plan.shards) {
        const ShardPaths paths = shard_paths(plan.out_base, shard.output_index, plan.shards.size());
        write_or_verify_json(paths.index, build_output_index(plan, shard));
    }
    write_or_verify_json(manifest_path, manifest);

    std::cout << "manifest complete: shards=" << stats.shards << " groups=" << stats.groups << " bytes=" << stats.bytes
              << " manifest=" << manifest_path.string() << '\n';
    return stats;
}

}  // namespace

fs::path output_manifest_path(const fs::path & out_base) {
    return fs::path(fs::absolute(out_base).lexically_normal().string() + "-experts-manifest.json");
}

RunStats run(const Options & options) {
    if (options.expert_first < 0 || options.expert_last < options.expert_first) {
        throw std::invalid_argument("invalid retained expert range");
    }
    if (options.manifest_only && (options.verify || options.layers.has_value())) {
        throw std::invalid_argument("--manifest-only cannot be combined with --verify or --layers");
    }

    Plan plan = build_plan(options);
    if (!plan.out_base.parent_path().empty()) {
        fs::create_directories(plan.out_base.parent_path());
    }
    if (options.manifest_only) {
        return write_manifest(plan, options);
    }

    RunStats emitted;
    for (const PlannedShard & shard : plan.shards) {
        if (options.layers.has_value() &&
            (shard.layer < options.layers->first || shard.layer > options.layers->second)) {
            continue;
        }
        add_stats(emitted, write_shard(plan, shard, options.verify));
    }
    if (emitted.shards == 0) {
        throw std::runtime_error("the requested layer range retains no expert groups");
    }
    std::cout << "shard emission complete: shards=" << emitted.shards << " groups=" << emitted.groups
              << " bytes=" << emitted.bytes << "; manifest not written (run --manifest-only separately)\n";
    return emitted;
}

}  // namespace wp_expert_shard
