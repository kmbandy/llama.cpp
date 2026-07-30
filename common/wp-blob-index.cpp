#include "wp-blob-index.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;
namespace fs = std::filesystem;

static constexpr const char * WP_MANIFEST_FORMAT = "llama.cpp.weight-pager.expert-shard-manifest";
static constexpr const char * WP_INDEX_FORMAT    = "llama.cpp.weight-pager.expert-shard-index";
static constexpr int          WP_FORMAT_VERSION  = 1;

static json wp_read_json(const fs::path & path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("wp-repack blobs: cannot open '" + path.string() + "'");
    }
    try {
        json value;
        ifs >> value;
        return value;
    } catch (const std::exception & e) {
        throw std::runtime_error("wp-repack blobs: '" + path.string() +
                                 "' is not valid JSON: " + e.what());
    }
}

// Field access with a message that names the file, so a malformed descriptor
// is diagnosable without reading the parser.
template <typename T>
static T wp_get(const json & obj, const char * key, const fs::path & where) {
    const auto it = obj.find(key);
    if (it == obj.end()) {
        throw std::runtime_error("wp-repack blobs: '" + where.string() +
                                 "' is missing required field '" + key + "'");
    }
    try {
        return it->get<T>();
    } catch (const std::exception & e) {
        throw std::runtime_error("wp-repack blobs: '" + where.string() + "' field '" +
                                 key + "' has the wrong type: " + e.what());
    }
}

static void wp_check_format(const json & obj, const char * expect_format,
                            const fs::path & where) {
    const auto format = wp_get<std::string>(obj, "format", where);
    if (format != expect_format) {
        throw std::runtime_error("wp-repack blobs: '" + where.string() + "' has format '" +
                                 format + "', expected '" + expect_format + "'");
    }
    const auto version = wp_get<int>(obj, "version", where);
    if (version != WP_FORMAT_VERSION) {
        throw std::runtime_error("wp-repack blobs: '" + where.string() + "' is version " +
                                 std::to_string(version) + ", this build understands version " +
                                 std::to_string(WP_FORMAT_VERSION));
    }
}

common_wp_blob_index common_wp_blob_index_load(const std::string & manifest_path,
                                               const std::string & model_path) {
    const fs::path manifest_file = fs::path(manifest_path);
    const fs::path dir           = manifest_file.parent_path();

    const json manifest = wp_read_json(manifest_file);
    wp_check_format(manifest, WP_MANIFEST_FORMAT, manifest_file);

    // The set must belong to this model. Compare filenames rather than full
    // paths: a blob set stays valid when the model directory is moved or
    // mounted elsewhere, but must not be paired with a different model or a
    // different quantisation, whose tensors would differ in size and content.
    const auto input_model = wp_get<std::string>(manifest, "input_model", manifest_file);
    const std::string want = fs::path(input_model).filename().string();
    const std::string have = fs::path(model_path).filename().string();
    if (want != have) {
        throw std::runtime_error(
            "wp-repack blobs: this set was built from '" + want + "' but the model being "
            "loaded is '" + have + "'. Refusing to page one model's experts out of "
            "another's blobs.");
    }

    const auto & shards = manifest.at("shards");
    if (!shards.is_array() || shards.empty()) {
        throw std::runtime_error("wp-repack blobs: manifest lists no shards");
    }

    common_wp_blob_index out;
    out.blob_files.resize(shards.size());

    // Pass 1: resolve and validate every blob, and count members so the name
    // storage can be sized exactly once (the entry name pointers must stay
    // stable, so it must never reallocate afterwards).
    std::vector<fs::path> index_files(shards.size());
    size_t n_members_total = 0;
    for (const auto & shard : shards) {
        const auto shard_index = wp_get<int>(shard, "shard_index", manifest_file);
        if (shard_index < 0 || (size_t) shard_index >= shards.size()) {
            throw std::runtime_error("wp-repack blobs: manifest has out-of-range shard_index " +
                                     std::to_string(shard_index));
        }
        if (!out.blob_files[shard_index].empty()) {
            throw std::runtime_error("wp-repack blobs: manifest repeats shard_index " +
                                     std::to_string(shard_index));
        }

        const fs::path blob  = dir / wp_get<std::string>(shard, "blob_file",  manifest_file);
        const fs::path index = dir / wp_get<std::string>(shard, "index_file", manifest_file);

        std::error_code ec;
        const auto on_disk = fs::file_size(blob, ec);
        if (ec) {
            throw std::runtime_error("wp-repack blobs: cannot stat blob '" + blob.string() +
                                     "': " + ec.message());
        }
        const auto expect = wp_get<uint64_t>(shard, "blob_bytes", manifest_file);
        if (on_disk != expect) {
            throw std::runtime_error(
                "wp-repack blobs: '" + blob.string() + "' is " + std::to_string(on_disk) +
                " bytes but the manifest records " + std::to_string(expect) +
                " -- the blob is truncated or was modified after packing");
        }

        out.blob_files[shard_index] = blob.string();
        index_files[shard_index]    = index;
        n_members_total += (size_t) wp_get<int>(shard, "group_count", manifest_file) * 3;
    }

    out.entry_names.reserve(n_members_total);
    out.entries.reserve(n_members_total);

    // Pass 2: read each sidecar and flatten its groups into entries.
    for (size_t blob_idx = 0; blob_idx < index_files.size(); ++blob_idx) {
        const fs::path & index_file = index_files[blob_idx];
        const json index = wp_read_json(index_file);
        wp_check_format(index, WP_INDEX_FORMAT, index_file);

        const auto & groups = index.at("groups");
        if (!groups.is_array()) {
            throw std::runtime_error("wp-repack blobs: '" + index_file.string() +
                                     "' has no groups array");
        }
        const auto blob_bytes = wp_get<uint64_t>(index, "blob_bytes", index_file);

        for (const auto & group : groups) {
            for (const auto & member : group.at("members")) {
                const auto name   = wp_get<std::string>(member, "catalog_name", index_file);
                const auto offset = wp_get<uint64_t>(member, "offset", index_file);
                const auto size   = wp_get<uint64_t>(member, "size",   index_file);

                // A member that runs past the blob would read adjacent
                // experts' bytes, or short-read at the tail.
                if (size == 0 || offset > blob_bytes || size > blob_bytes - offset) {
                    throw std::runtime_error(
                        "wp-repack blobs: '" + name + "' spans [" + std::to_string(offset) +
                        ", " + std::to_string(offset + size) + ") which does not fit in the " +
                        std::to_string(blob_bytes) + "-byte blob");
                }

                if (out.entry_names.size() == out.entry_names.capacity()) {
                    // Would reallocate and dangle every name pointer emitted
                    // so far. Means the manifest's group_count disagrees with
                    // the sidecars, which is a corrupt set.
                    throw std::runtime_error(
                        "wp-repack blobs: sidecars contain more members than the manifest's "
                        "group counts account for -- the set is inconsistent");
                }
                out.entry_names.push_back(name);

                llama_wp_blob_entry entry {};
                entry.name        = out.entry_names.back().c_str();
                entry.blob_idx    = (uint32_t) blob_idx;
                entry.blob_offset = offset;
                entry.size        = size;
                out.entries.push_back(entry);
            }
        }
    }

    if (out.entries.empty()) {
        throw std::runtime_error("wp-repack blobs: the set describes no expert pages");
    }

    out.blob_file_ptrs.reserve(out.blob_files.size());
    for (const std::string & path : out.blob_files) {
        out.blob_file_ptrs.push_back(path.c_str());
    }

    return out;
}
