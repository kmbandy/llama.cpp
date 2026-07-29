/*
 * Expert-major weight-pager repack format, version 1.
 *
 * Each .wpb file is a headerless concatenation of expert groups. A group is
 * every PageCatalog entry with is_expert=true, is_consolidated=false, and
 * is_pinned=false for one (block_idx, expert_idx), with members ordered by
 * expert_role_mask. Groups and layers are never split between blobs.
 *
 * Every blob has a self-sufficient .wpi.json sidecar. It records the format
 * version, source model files, layer range, structural SHA-256, and, for every
 * group, block/expert IDs plus each member's role mask, byte size, blob offset,
 * catalog name, source tensor name, source file index, and source file offset.
 * The global -experts-manifest.json lists the complete shard set and its
 * structural SHA-256. Hashes cover canonical group identity (IDs, role masks,
 * sizes, and names), while --verify also compares every payload byte.
 */

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include "wp-repack-lib.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

namespace {

constexpr int          FORMAT_VERSION   = 1;
constexpr const char * INDEX_FORMAT     = "llama.cpp.weight-pager.expert-shard-index";
constexpr const char * MANIFEST_FORMAT  = "llama.cpp.weight-pager.expert-shard-manifest";
constexpr size_t       COPY_BUFFER_SIZE = 8u * 1024u * 1024u;

struct ModelCatalog {
    wp::PageCatalog          catalog;
    std::vector<std::string> files;
};

struct CliOptions {
    bool                               verify          = false;
    bool                               allow_partial   = false;
    uint64_t                           max_shard_bytes = 0;
    std::vector<wp_repack::LayerRange> layer_ranges;
    std::string                        model;
    std::string                        output;
};

struct ShardPaths {
    fs::path blob;
    fs::path index;
};

struct VerifyCounts {
    uint64_t shards  = 0;
    uint64_t groups  = 0;
    uint64_t members = 0;
    uint64_t bytes   = 0;
};

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

void print_usage(const char * argv0) {
    std::cout << "usage:\n"
              << "  " << argv0 << " [sharding options] MODEL OUTPUT_BASE\n"
              << "  " << argv0 << " --verify MODEL OUTPUT_BASE_OR_MANIFEST_OR_INDEX\n\n"
              << "options:\n"
              << "  --shard-by-layer       one output shard per expert layer (default)\n"
              << "  --max-shard-bytes N    coalesce adjacent layers up to N bytes\n"
              << "                         suffixes K, M, G, KiB, MiB, GiB are accepted\n"
              << "  --layer-ranges RANGES  explicit shards, e.g. 0-32,33-46\n"
              << "                         must cover every expert layer unless --allow-partial\n"
              << "  --allow-partial        permit --layer-ranges that omit layers (subset repack)\n"
              << "  --verify               compare indexes and blob bytes with MODEL\n"
              << "  -h, --help             show this help\n\n"
              << "Existing output files are never overwritten.\n";
}

uint64_t parse_bytes(const std::string & text) {
    if (text.empty()) {
        throw std::invalid_argument("byte count cannot be empty");
    }

    size_t                   used  = 0;
    const unsigned long long value = std::stoull(text, &used, 10);
    if (value == 0) {
        throw std::invalid_argument("byte count must be positive");
    }

    const std::string suffix     = text.substr(used);
    uint64_t          multiplier = 1;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1;
    } else if (suffix == "K") {
        multiplier = 1000ull;
    } else if (suffix == "M") {
        multiplier = 1000ull * 1000ull;
    } else if (suffix == "G") {
        multiplier = 1000ull * 1000ull * 1000ull;
    } else if (suffix == "KiB") {
        multiplier = 1024ull;
    } else if (suffix == "MiB") {
        multiplier = 1024ull * 1024ull;
    } else if (suffix == "GiB") {
        multiplier = 1024ull * 1024ull * 1024ull;
    } else {
        throw std::invalid_argument("unsupported byte-count suffix: " + suffix);
    }
    if (value > std::numeric_limits<uint64_t>::max() / multiplier) {
        throw std::overflow_error("byte count is too large");
    }
    return static_cast<uint64_t>(value) * multiplier;
}

CliOptions parse_cli(int argc, char ** argv) {
    CliOptions               options;
    std::vector<std::string> positional;
    int                      sharding_modes = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--verify") {
            options.verify = true;
        } else if (arg == "--allow-partial") {
            options.allow_partial = true;
        } else if (arg == "--shard-by-layer") {
            ++sharding_modes;
        } else if (arg == "--max-shard-bytes") {
            if (++i >= argc) {
                throw std::invalid_argument("--max-shard-bytes requires a value");
            }
            options.max_shard_bytes = parse_bytes(argv[i]);
            ++sharding_modes;
        } else if (arg == "--layer-ranges") {
            if (++i >= argc) {
                throw std::invalid_argument("--layer-ranges requires a value");
            }
            options.layer_ranges = wp_repack::parse_layer_ranges(argv[i]);
            ++sharding_modes;
        } else if (arg.size() > 1 && arg[0] == '-') {
            throw std::invalid_argument("unknown option: " + arg);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 2) {
        throw std::invalid_argument("expected MODEL and OUTPUT_BASE_OR_INDEX");
    }
    if (sharding_modes > 1) {
        throw std::invalid_argument("choose only one sharding mode");
    }
    if (options.verify && sharding_modes != 0) {
        throw std::invalid_argument("sharding options are not valid with --verify");
    }

    options.model  = positional[0];
    options.output = positional[1];
    return options;
}

gguf_ptr load_gguf(const std::string & path, ggml_ptr & tensors) {
    ggml_context *         raw_ctx = nullptr;
    const gguf_init_params params  = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&raw_ctx,
    };
    gguf_context * raw_gguf = gguf_init_from_file(path.c_str(), params);
    gguf_ptr       result(raw_gguf, gguf_free);
    tensors.reset(raw_ctx);
    if (result == nullptr || raw_ctx == nullptr) {
        throw std::runtime_error("failed to read GGUF metadata: " + path);
    }
    return result;
}

uint16_t optional_u16(const gguf_context * ctx, const char * key, uint16_t fallback) {
    const int key_id = gguf_find_key(ctx, key);
    return key_id < 0 ? fallback : gguf_get_val_u16(ctx, key_id);
}

std::vector<std::string> discover_model_files(const std::string & input) {
    const std::string first = fs::canonical(fs::path(input)).string();
    ggml_ptr          tensor_ctx(nullptr, ggml_free);
    gguf_ptr          gguf = load_gguf(first, tensor_ctx);

    const uint16_t split_count = optional_u16(gguf.get(), "split.count", 1);
    if (split_count <= 1) {
        return { first };
    }

    const uint16_t split_no = optional_u16(gguf.get(), "split.no", UINT16_MAX);
    if (split_no != 0) {
        throw std::runtime_error("split GGUF input must be the first shard: " + first);
    }

    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), "-%05u-of-%05u.gguf", static_cast<unsigned int>(split_no + 1),
                  static_cast<unsigned int>(split_count));
    const std::string expected_suffix(suffix);
    if (first.size() <= expected_suffix.size() ||
        first.compare(first.size() - expected_suffix.size(), expected_suffix.size(), expected_suffix) != 0) {
        throw std::runtime_error("invalid split GGUF file name: " + first);
    }
    const std::string prefix = first.substr(0, first.size() - expected_suffix.size());

    std::vector<std::string> files;
    files.reserve(split_count);
    for (uint16_t i = 0; i < split_count; ++i) {
        char split_suffix[64];
        std::snprintf(split_suffix, sizeof(split_suffix), "-%05u-of-%05u.gguf", static_cast<unsigned int>(i + 1),
                      static_cast<unsigned int>(split_count));
        files.push_back(fs::canonical(fs::path(prefix + split_suffix)).string());
    }
    return files;
}

ModelCatalog build_catalog(const std::string & input) {
    ModelCatalog result;
    result.files = discover_model_files(input);
    if (result.files.size() > UINT16_MAX) {
        throw std::runtime_error("too many GGUF shards");
    }

    std::set<std::string> tensor_names;
    for (size_t file_idx = 0; file_idx < result.files.size(); ++file_idx) {
        ggml_ptr       tensor_ctx(nullptr, ggml_free);
        gguf_ptr       gguf        = load_gguf(result.files[file_idx], tensor_ctx);
        const uint64_t data_offset = gguf_get_data_offset(gguf.get());
        const int64_t  n_tensors   = gguf_get_n_tensors(gguf.get());

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * raw_name = gguf_get_tensor_name(gguf.get(), i);
            if (raw_name == nullptr) {
                throw std::runtime_error("GGUF tensor has no name");
            }
            const std::string name(raw_name);
            if (!tensor_names.insert(name).second) {
                throw std::runtime_error("duplicate tensor across GGUF shards: " + name);
            }

            const uint64_t file_offset = data_offset + gguf_get_tensor_offset(gguf.get(), i);
            const size_t   size        = gguf_get_tensor_size(gguf.get(), i);

            wp::PageCatalog classifier;
            const int       classified_idx  = classifier.add(name, static_cast<uint16_t>(file_idx), file_offset, size);
            const wp::PageMeta & classified = classifier.at(classified_idx);

            int n_experts = 1;
            if (classified.is_expert && classified.is_consolidated) {
                const ggml_tensor * tensor = ggml_get_tensor(tensor_ctx.get(), name.c_str());
                if (tensor == nullptr) {
                    throw std::runtime_error("missing GGML tensor metadata: " + name);
                }
                if (tensor->ne[2] > 1 && tensor->ne[2] <= INT_MAX) {
                    n_experts = static_cast<int>(tensor->ne[2]);
                } else if (tensor->ne[3] > 1 && tensor->ne[3] <= INT_MAX) {
                    n_experts = static_cast<int>(tensor->ne[3]);
                }
            }

            if (n_experts > 1) {
                result.catalog.add_consolidated_experts(name, static_cast<uint16_t>(file_idx), file_offset, size,
                                                        n_experts);
            } else {
                result.catalog.add(name, static_cast<uint16_t>(file_idx), file_offset, size);
            }
        }
    }
    return result;
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

std::string finish_sha(sha256_t & hash) {
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_final(&hash, digest.data());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return out.str();
}

std::string hash_groups(const std::vector<wp_repack::ExpertGroup> & groups, const std::vector<size_t> & indices) {
    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-repack.identity.v1");
    sha_update_u64(hash, indices.size());

    for (size_t index : indices) {
        if (index >= groups.size()) {
            throw std::runtime_error("internal group index is out of range");
        }
        const wp_repack::ExpertGroup & group = groups[index];
        sha_update_u64(hash, static_cast<uint64_t>(group.block_idx));
        sha_update_u64(hash, static_cast<uint64_t>(group.expert_idx));
        sha_update_u64(hash, group.members.size());
        for (const wp_repack::ExpertMember & member : group.members) {
            sha_update_u64(hash, member.role_mask);
            sha_update_u64(hash, member.size);
            sha_update_string(hash, member.catalog_name);
            sha_update_string(hash, member.source_tensor_name);
        }
    }
    return finish_sha(hash);
}

std::vector<size_t> flatten_indices(const std::vector<wp_repack::ShardPlan> & shards) {
    std::vector<size_t> indices;
    for (const wp_repack::ShardPlan & shard : shards) {
        indices.insert(indices.end(), shard.group_indices.begin(), shard.group_indices.end());
    }
    return indices;
}

std::string numbered_name(const std::string & base, size_t index, size_t total) {
    std::ostringstream name;
    name << base << "-experts-" << std::setw(5) << std::setfill('0') << index + 1 << "-of-" << std::setw(5)
         << std::setfill('0') << total;
    return name.str();
}

ShardPaths shard_paths(const fs::path & output_base, size_t index, size_t total) {
    const std::string prefix = numbered_name(output_base.string(), index, total);
    return { fs::path(prefix + ".wpb"), fs::path(prefix + ".wpi.json") };
}

fs::path manifest_path(const fs::path & output_base) {
    return fs::path(output_base.string() + "-experts-manifest.json");
}

void ensure_outputs_absent(const fs::path & output_base, const std::vector<wp_repack::ShardPlan> & shards) {
    std::vector<fs::path> paths{ manifest_path(output_base) };
    for (size_t i = 0; i < shards.size(); ++i) {
        const ShardPaths names = shard_paths(output_base, i, shards.size());
        paths.push_back(names.blob);
        paths.push_back(names.index);
    }
    for (const fs::path & path : paths) {
        if (fs::exists(path)) {
            throw std::runtime_error("refusing to overwrite existing output: " + path.string());
        }
    }
}

void write_json(const fs::path & path, const json & value) {
    const fs::path temp(path.string() + ".tmp");
    std::ofstream  out(temp);
    if (!out) {
        throw std::runtime_error("failed to create " + temp.string());
    }
    out << value.dump(2) << '\n';
    out.close();
    if (!out) {
        throw std::runtime_error("failed to write " + temp.string());
    }
    fs::rename(temp, path);
}

void copy_member(std::ifstream &     source,
                 uint64_t            source_offset,
                 uint64_t            size,
                 std::ofstream &     output,
                 std::vector<char> & buffer) {
    source.clear();
    source.seekg(static_cast<std::streamoff>(source_offset));
    if (!source) {
        throw std::runtime_error("failed to seek source tensor bytes");
    }

    uint64_t remaining = size;
    while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
        source.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (source.gcount() != static_cast<std::streamsize>(chunk)) {
            throw std::runtime_error("short read from source GGUF");
        }
        output.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!output) {
            throw std::runtime_error("failed to write expert blob");
        }
        remaining -= chunk;
    }
}

json write_shard(const fs::path &                            output_base,
                 size_t                                      shard_index,
                 size_t                                      shard_count,
                 const wp_repack::ShardPlan &                shard,
                 const std::vector<wp_repack::ExpertGroup> & groups,
                 const std::vector<std::string> &            model_files,
                 std::vector<std::ifstream> &                sources) {
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
    const fs::path   temp_blob(paths.blob.string() + ".tmp");
    std::ofstream    blob(temp_blob, std::ios::binary);
    if (!blob) {
        throw std::runtime_error("failed to create " + temp_blob.string());
    }

    json index = {
        { "format",       INDEX_FORMAT                   },
        { "version",      FORMAT_VERSION                 },
        { "blob_file",    paths.blob.filename().string() },
        { "shard_index",  shard_index                    },
        { "shard_count",  shard_count                    },
        { "layer_first",  shard.layer_first              },
        { "layer_last",   shard.layer_last               },
        { "group_count",  shard.group_indices.size()     },
        { "blob_bytes",   shard.size                     },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups(groups, shard.group_indices) },
          }                                              },
        { "model_files",  model_files                    },
        { "groups",       json::array()                  },
    };

    uint64_t          blob_offset = 0;
    std::vector<char> buffer(COPY_BUFFER_SIZE);
    for (size_t group_index : shard.group_indices) {
        const wp_repack::ExpertGroup & group      = groups.at(group_index);
        json                           group_json = {
            { "block_idx",    group.block_idx      },
            { "expert_idx",   group.expert_idx     },
            { "member_count", group.members.size() },
            { "members",      json::array()        },
        };

        for (const wp_repack::ExpertMember & member : group.members) {
            if (member.file_idx >= sources.size()) {
                throw std::runtime_error("catalog source file index is out of range");
            }
            group_json["members"].push_back({
                { "role_mask",          member.role_mask          },
                { "size",               member.size               },
                { "offset",             blob_offset               },
                { "catalog_name",       member.catalog_name       },
                { "source_tensor_name", member.source_tensor_name },
                { "source_file_idx",    member.file_idx           },
                { "source_file_offset", member.file_offset        },
            });
            copy_member(sources[member.file_idx], member.file_offset, member.size, blob, buffer);
            blob_offset += member.size;
        }
        index["groups"].push_back(std::move(group_json));
    }

    blob.close();
    if (!blob) {
        throw std::runtime_error("failed to finish " + temp_blob.string());
    }
    if (blob_offset != shard.size) {
        throw std::runtime_error("internal shard byte count mismatch");
    }
    fs::rename(temp_blob, paths.blob);
    write_json(paths.index, index);

    return {
        { "blob_file",    paths.blob.filename().string()  },
        { "index_file",   paths.index.filename().string() },
        { "shard_index",  shard_index                     },
        { "layer_first",  shard.layer_first               },
        { "layer_last",   shard.layer_last                },
        { "group_count",  shard.group_indices.size()      },
        { "blob_bytes",   shard.size                      },
        { "content_hash", index["content_hash"]           },
    };
}

void repack(const CliOptions & options) {
    ModelCatalog                              model  = build_catalog(options.model);
    const std::vector<wp_repack::ExpertGroup> groups = wp_repack::build_expert_groups(model.catalog);
    if (groups.empty()) {
        throw std::runtime_error("PageCatalog found no slottable expert groups");
    }

    std::vector<wp_repack::ShardPlan> shards;
    std::string                       sharding_mode;
    if (!options.layer_ranges.empty()) {
        shards        = wp_repack::plan_shards_for_ranges(groups, options.layer_ranges, options.allow_partial);
        sharding_mode = "layer-ranges";
        if (options.allow_partial) {
            // Deliberate subset. Say so loudly and name what is missing, so a partial
            // artifact is never mistaken for a complete one further down the line.
            const std::vector<int> missing = wp_repack::uncovered_layers(groups, options.layer_ranges);
            if (!missing.empty()) {
                std::fprintf(stderr, "wp-repack: WARNING partial repack, %zu expert layer(s) omitted:",
                             missing.size());
                for (const int layer : missing) {
                    std::fprintf(stderr, " %d", layer);
                }
                std::fprintf(stderr, "\n");
            }
        }
    } else if (options.max_shard_bytes != 0) {
        shards        = wp_repack::plan_shards_max_bytes(groups, options.max_shard_bytes);
        sharding_mode = "max-shard-bytes";
    } else {
        shards        = wp_repack::plan_shards_by_layer(groups);
        sharding_mode = "shard-by-layer";
    }

    shards.erase(std::remove_if(shards.begin(), shards.end(),
                                [](const wp_repack::ShardPlan & shard) { return shard.group_indices.empty(); }),
                 shards.end());
    if (shards.empty()) {
        throw std::runtime_error("selected layer range contains no expert groups");
    }

    const fs::path output_base = fs::absolute(fs::path(options.output)).lexically_normal();
    if (!output_base.parent_path().empty()) {
        fs::create_directories(output_base.parent_path());
    }
    ensure_outputs_absent(output_base, shards);

    std::vector<std::ifstream> sources;
    sources.reserve(model.files.size());
    for (const std::string & path : model.files) {
        sources.emplace_back(path, std::ios::binary);
        if (!sources.back()) {
            throw std::runtime_error("failed to open source model file: " + path);
        }
    }

    const std::vector<size_t> selected = flatten_indices(shards);
    json                      manifest = {
        { "format",            MANIFEST_FORMAT                                 },
        { "version",           FORMAT_VERSION                                  },
        { "input_model",       fs::canonical(fs::path(options.model)).string() },
        { "model_files",       model.files                                     },
        { "sharding_mode",     sharding_mode                                   },
        { "total_group_count", selected.size()                                 },
        { "total_blob_bytes",  0                                               },
        { "shard_count",       shards.size()                                   },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups(groups, selected) },
          }                                                                    },
        { "shards",            json::array()                                   },
    };

    uint64_t total_bytes = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "writing shard " << i + 1 << "/" << shards.size() << " layers " << shards[i].layer_first << "-"
                  << shards[i].layer_last << " groups " << shards[i].group_indices.size() << " bytes " << shards[i].size
                  << '\n';
        manifest["shards"].push_back(
            write_shard(output_base, i, shards.size(), shards[i], groups, model.files, sources));
        total_bytes += shards[i].size;
    }
    manifest["total_blob_bytes"] = total_bytes;
    const fs::path out_manifest  = manifest_path(output_base);
    write_json(out_manifest, manifest);

    std::cout << "repack complete: shards=" << shards.size() << " groups=" << selected.size()
              << " bytes=" << total_bytes << " manifest=" << out_manifest.string() << '\n';
}

json read_json(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    json value;
    input >> value;
    return value;
}

void require_format(const json & value, const char * format) {
    if (value.value("format", "") != format || value.value("version", 0) != FORMAT_VERSION) {
        throw std::runtime_error("unsupported or invalid repack metadata format");
    }
}

std::vector<size_t> indices_for_range(const std::vector<wp_repack::ExpertGroup> & groups, int first, int last) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].block_idx >= first && groups[i].block_idx <= last) {
            indices.push_back(i);
        }
    }
    return indices;
}

void compare_bytes(std::ifstream &     blob,
                   uint64_t            blob_offset,
                   std::ifstream &     source,
                   uint64_t            source_offset,
                   uint64_t            size,
                   std::vector<char> & blob_buffer,
                   std::vector<char> & source_buffer) {
    blob.clear();
    blob.seekg(static_cast<std::streamoff>(blob_offset));
    source.clear();
    source.seekg(static_cast<std::streamoff>(source_offset));
    if (!blob || !source) {
        throw std::runtime_error("failed to seek while verifying bytes");
    }

    uint64_t remaining = size;
    uint64_t compared  = 0;
    while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, blob_buffer.size()));
        blob.read(blob_buffer.data(), static_cast<std::streamsize>(chunk));
        source.read(source_buffer.data(), static_cast<std::streamsize>(chunk));
        if (blob.gcount() != static_cast<std::streamsize>(chunk) ||
            source.gcount() != static_cast<std::streamsize>(chunk)) {
            throw std::runtime_error("short read while verifying bytes");
        }
        if (std::memcmp(blob_buffer.data(), source_buffer.data(), chunk) != 0) {
            throw std::runtime_error("payload byte mismatch at member byte " + std::to_string(compared));
        }
        remaining -= chunk;
        compared += chunk;
    }
}

VerifyCounts verify_index(const fs::path &                            index_path,
                          const json &                                index,
                          const std::vector<wp_repack::ExpertGroup> & groups,
                          std::vector<std::ifstream> &                sources) {
    require_format(index, INDEX_FORMAT);
    const int layer_first = index.at("layer_first").get<int>();
    const int layer_last  = index.at("layer_last").get<int>();
    if (layer_first < 0 || layer_last < layer_first) {
        throw std::runtime_error("invalid shard layer range");
    }

    const std::vector<size_t> expected_indices = indices_for_range(groups, layer_first, layer_last);
    const json &              indexed_groups   = index.at("groups");
    if (!indexed_groups.is_array() || indexed_groups.size() != expected_indices.size() ||
        index.at("group_count").get<uint64_t>() != expected_indices.size()) {
        throw std::runtime_error("group count mismatch");
    }

    const std::string expected_hash = hash_groups(groups, expected_indices);
    if (index.at("content_hash").at("algorithm").get<std::string>() != "sha256" ||
        index.at("content_hash").at("value").get<std::string>() != expected_hash) {
        throw std::runtime_error("shard structural content hash mismatch");
    }

    const fs::path blob_path = index_path.parent_path() / index.at("blob_file").get<std::string>();
    std::ifstream  blob(blob_path, std::ios::binary);
    if (!blob) {
        throw std::runtime_error("failed to open expert blob: " + blob_path.string());
    }

    uint64_t     next_offset = 0;
    VerifyCounts counts;
    counts.shards = 1;
    std::vector<char> blob_buffer(COPY_BUFFER_SIZE);
    std::vector<char> source_buffer(COPY_BUFFER_SIZE);

    for (size_t i = 0; i < expected_indices.size(); ++i) {
        const wp_repack::ExpertGroup & expected = groups[expected_indices[i]];
        const json &                   actual   = indexed_groups.at(i);
        if (actual.at("block_idx").get<int>() != expected.block_idx ||
            actual.at("expert_idx").get<int>() != expected.expert_idx) {
            throw std::runtime_error("group identity mismatch");
        }
        const json & members = actual.at("members");
        if (!members.is_array() || actual.at("member_count").get<uint64_t>() != expected.members.size() ||
            members.size() != expected.members.size()) {
            throw std::runtime_error("member count mismatch");
        }

        for (size_t j = 0; j < expected.members.size(); ++j) {
            const wp_repack::ExpertMember & expected_member = expected.members[j];
            const json &                    actual_member   = members.at(j);
            const uint64_t                  actual_offset   = actual_member.at("offset").get<uint64_t>();
            if (actual_member.at("role_mask").get<uint8_t>() != expected_member.role_mask ||
                actual_member.at("size").get<uint64_t>() != expected_member.size ||
                actual_member.at("catalog_name").get<std::string>() != expected_member.catalog_name ||
                actual_member.at("source_tensor_name").get<std::string>() != expected_member.source_tensor_name ||
                actual_member.at("source_file_idx").get<uint16_t>() != expected_member.file_idx ||
                actual_member.at("source_file_offset").get<uint64_t>() != expected_member.file_offset) {
                throw std::runtime_error("member identity or size mismatch for " + expected_member.catalog_name);
            }
            if (actual_offset != next_offset) {
                throw std::runtime_error("blob offsets are not contiguous");
            }
            if (expected_member.file_idx >= sources.size()) {
                throw std::runtime_error("fresh catalog source file index is out of range");
            }

            compare_bytes(blob, actual_offset, sources[expected_member.file_idx], expected_member.file_offset,
                          expected_member.size, blob_buffer, source_buffer);
            next_offset += expected_member.size;
            ++counts.members;
            counts.bytes += expected_member.size;
        }
        ++counts.groups;
    }

    if (index.at("blob_bytes").get<uint64_t>() != next_offset || fs::file_size(blob_path) != next_offset) {
        throw std::runtime_error("blob byte size mismatch");
    }
    return counts;
}

VerifyCounts verify_index_path(const fs::path &                            index_path,
                               const std::vector<wp_repack::ExpertGroup> & groups,
                               std::vector<std::ifstream> &                sources) {
    const json index = read_json(index_path);
    return verify_index(index_path, index, groups, sources);
}

void add_counts(VerifyCounts & total, const VerifyCounts & value) {
    total.shards += value.shards;
    total.groups += value.groups;
    total.members += value.members;
    total.bytes += value.bytes;
}

VerifyCounts verify_manifest(const fs::path &                            path,
                             const std::vector<wp_repack::ExpertGroup> & groups,
                             std::vector<std::ifstream> &                sources) {
    const json manifest = read_json(path);
    require_format(manifest, MANIFEST_FORMAT);
    const json & shards = manifest.at("shards");
    if (!shards.is_array() || manifest.at("shard_count").get<uint64_t>() != shards.size()) {
        throw std::runtime_error("manifest shard count mismatch");
    }

    VerifyCounts                  total;
    std::vector<size_t>           all_indices;
    std::set<std::pair<int, int>> seen_groups;
    for (size_t shard_pos = 0; shard_pos < shards.size(); ++shard_pos) {
        const json &              shard         = shards.at(shard_pos);
        const int                 first         = shard.at("layer_first").get<int>();
        const int                 last          = shard.at("layer_last").get<int>();
        const std::vector<size_t> shard_indices = indices_for_range(groups, first, last);
        for (size_t index : shard_indices) {
            const auto key = std::make_pair(groups[index].block_idx, groups[index].expert_idx);
            if (!seen_groups.insert(key).second) {
                throw std::runtime_error("manifest layer ranges duplicate an expert group");
            }
        }
        all_indices.insert(all_indices.end(), shard_indices.begin(), shard_indices.end());

        const fs::path index_path = path.parent_path() / shard.at("index_file").get<std::string>();
        const json     index      = read_json(index_path);
        require_format(index, INDEX_FORMAT);
        if (shard.at("shard_index").get<uint64_t>() != shard_pos ||
            index.at("shard_index").get<uint64_t>() != shard_pos ||
            index.at("shard_count").get<uint64_t>() != shards.size() || index.at("layer_first").get<int>() != first ||
            index.at("layer_last").get<int>() != last || index.at("group_count") != shard.at("group_count") ||
            index.at("blob_bytes") != shard.at("blob_bytes") || index.at("blob_file") != shard.at("blob_file") ||
            index.at("content_hash") != shard.at("content_hash")) {
            throw std::runtime_error("manifest and shard index metadata disagree");
        }
        add_counts(total, verify_index(index_path, index, groups, sources));
    }

    if (manifest.at("total_group_count").get<uint64_t>() != total.groups ||
        manifest.at("total_blob_bytes").get<uint64_t>() != total.bytes) {
        throw std::runtime_error("manifest totals mismatch");
    }
    if (manifest.at("content_hash").at("algorithm").get<std::string>() != "sha256" ||
        manifest.at("content_hash").at("value").get<std::string>() != hash_groups(groups, all_indices)) {
        throw std::runtime_error("manifest structural content hash mismatch");
    }
    return total;
}

void verify(const CliOptions & options) {
    ModelCatalog                              model  = build_catalog(options.model);
    const std::vector<wp_repack::ExpertGroup> groups = wp_repack::build_expert_groups(model.catalog);
    if (groups.empty()) {
        throw std::runtime_error("PageCatalog found no slottable expert groups");
    }

    std::vector<std::ifstream> sources;
    sources.reserve(model.files.size());
    for (const std::string & source : model.files) {
        sources.emplace_back(source, std::ios::binary);
        if (!sources.back()) {
            throw std::runtime_error("failed to open source model file: " + source);
        }
    }

    fs::path     target = fs::absolute(fs::path(options.output)).lexically_normal();
    VerifyCounts counts;
    if (target.extension() == ".json" && target.filename().string().find(".wpi.json") != std::string::npos) {
        counts = verify_index_path(target, groups, sources);
    } else {
        if (!(target.extension() == ".json")) {
            target = manifest_path(target);
        }
        counts = verify_manifest(target, groups, sources);
    }

    std::cout << "verify PASS: shards=" << counts.shards << " groups=" << counts.groups << " members=" << counts.members
              << " bytes=" << counts.bytes << '\n';
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.verify) {
            try {
                verify(options);
            } catch (const std::exception & error) {
                std::cerr << "verify FAIL: " << error.what() << '\n';
                return 1;
            }
        } else {
            repack(options);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
