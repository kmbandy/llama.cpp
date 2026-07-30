// Expert-index shard builder acceptance test.
//
// The fixture is a complete two-layer wp-repack blob set with three experts
// per layer. The test retains experts 0-1 and verifies both the file layout
// and every copied byte without loading a model or using a GPU.

#include "nlohmann/json.hpp"
#include "wp-blob-index.h"
#include "wp-expert-shard-lib.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

static int g_fail = 0;

static void check(bool ok, const std::string & what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        ++g_fail;
    }
}

static void write_json(const fs::path & path, const json & value) {
    std::ofstream output(path);
    output << value.dump(2) << '\n';
}

static std::vector<char> read_bytes(const fs::path & path, uint64_t offset, uint64_t size) {
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<char> bytes(static_cast<size_t>(size));
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

struct Fixture {
    fs::path dir;
    fs::path source_manifest;
    fs::path out_base;

    Fixture() {
        dir = fs::temp_directory_path() / "wp-expert-shard-test";
        std::error_code ignored;
        fs::remove_all(dir, ignored);
        fs::create_directories(dir);
        source_manifest = dir / "source-experts-manifest.json";
        out_base        = dir / "cut";
        write_source();
    }

    ~Fixture() {
        std::error_code ignored;
        fs::remove_all(dir, ignored);
    }

    static std::string numbered(const std::string & extension, int shard_index) {
        char name[128];
        std::snprintf(name, sizeof(name), "source-experts-%05d-of-00002.%s", shard_index + 1, extension.c_str());
        return name;
    }

    json write_source_shard(int shard_index, int layer) {
        const fs::path blob_path  = dir / numbered("wpb", shard_index);
        const fs::path index_path = dir / numbered("wpi.json", shard_index);
        std::ofstream  blob(blob_path, std::ios::binary);

        json     groups = json::array();
        uint64_t offset = 0;
        for (int expert = 0; expert < 3; ++expert) {
            json group = {
                { "block_idx",    layer         },
                { "expert_idx",   expert        },
                { "member_count", 3             },
                { "members",      json::array() },
            };

            const std::array<const char *, 3> roles = { "up", "gate", "down" };
            const std::array<int, 3>          masks = { 1, 2, 4 };
            const std::array<uint64_t, 3>     sizes = {
                static_cast<uint64_t>(5 + expert),
                static_cast<uint64_t>(7 + expert),
                static_cast<uint64_t>(9 + expert),
            };
            for (size_t member_index = 0; member_index < roles.size(); ++member_index) {
                const std::string tensor =
                    "blk." + std::to_string(layer) + ".ffn_" + roles[member_index] + "_exps.weight";
                group["members"].push_back({
                    { "role_mask",          masks[member_index]                          },
                    { "size",               sizes[member_index]                          },
                    { "offset",             offset                                       },
                    { "catalog_name",       tensor + "#expert." + std::to_string(expert) },
                    { "source_tensor_name", tensor                                       },
                    { "source_file_idx",    0                                            },
                    { "source_file_offset", 100000 + offset                              },
                });

                for (uint64_t byte = 0; byte < sizes[member_index]; ++byte) {
                    const char value = static_cast<char>(
                        (layer * 31 + expert * 17 + static_cast<int>(member_index) * 7 + byte) & 0xff);
                    blob.write(&value, 1);
                }
                offset += sizes[member_index];
            }
            groups.push_back(std::move(group));
        }
        blob.close();

        const json index = {
            { "format",       "llama.cpp.weight-pager.expert-shard-index"           },
            { "version",      1                                                     },
            { "blob_file",    blob_path.filename().string()                         },
            { "shard_index",  shard_index                                           },
            { "shard_count",  2                                                     },
            { "layer_first",  layer                                                 },
            { "layer_last",   layer                                                 },
            { "group_count",  groups.size()                                         },
            { "blob_bytes",   offset                                                },
            { "content_hash", { { "algorithm", "sha256" }, { "value", "fixture" } } },
            { "model_files",  { "/models/toy-model.gguf" }                          },
            { "groups",       std::move(groups)                                     },
        };
        write_json(index_path, index);

        return {
            { "blob_file",    blob_path.filename().string()                         },
            { "index_file",   index_path.filename().string()                        },
            { "shard_index",  shard_index                                           },
            { "layer_first",  layer                                                 },
            { "layer_last",   layer                                                 },
            { "group_count",  3                                                     },
            { "blob_bytes",   offset                                                },
            { "content_hash", { { "algorithm", "sha256" }, { "value", "fixture" } } },
        };
    }

    void write_source() {
        json shards = json::array();
        shards.push_back(write_source_shard(0, 3));
        shards.push_back(write_source_shard(1, 4));
        const uint64_t total_bytes =
            shards.at(0).at("blob_bytes").get<uint64_t>() + shards.at(1).at("blob_bytes").get<uint64_t>();

        const json manifest = {
            { "format",            "llama.cpp.weight-pager.expert-shard-manifest"        },
            { "version",           1                                                     },
            { "input_model",       "/models/toy-model.gguf"                              },
            { "model_files",       { "/models/toy-model.gguf" }                          },
            { "sharding_mode",     "shard-by-layer"                                      },
            { "total_group_count", 6                                                     },
            { "total_blob_bytes",  total_bytes                                           },
            { "shard_count",       2                                                     },
            { "content_hash",      { { "algorithm", "sha256" }, { "value", "fixture" } } },
            { "shards",            std::move(shards)                                     },
        };
        write_json(source_manifest, manifest);
    }
};

static const json & find_source_group(const json & groups, int expert_idx) {
    for (const json & group : groups) {
        if (group.at("expert_idx").get<int>() == expert_idx) {
            return group;
        }
    }
    throw std::runtime_error("fixture source group not found");
}

static void test_expert_shard() {
    std::printf("expert-index shard\n");
    Fixture fixture;

    const fs::path planned_out_base = fixture.dir / "planned" / "cut";

    wp_expert_shard::Options planned;
    planned.src_manifest                          = fixture.source_manifest;
    planned.out_base                              = planned_out_base;
    planned.expert_first                          = 0;
    planned.expert_last                           = 1;
    planned.manifest_only                         = true;
    const wp_expert_shard::RunStats planned_stats = wp_expert_shard::run(planned);

    const fs::path planned_manifest_path = wp_expert_shard::output_manifest_path(planned_out_base);
    std::ifstream  planned_manifest_input(planned_manifest_path);
    json           planned_manifest;
    planned_manifest_input >> planned_manifest;

    bool                           planned_sidecars_exist = true;
    bool                           planned_blobs_absent   = true;
    std::vector<std::vector<char>> planned_sidecars;
    for (const json & shard : planned_manifest.at("shards")) {
        const fs::path sidecar = planned_out_base.parent_path() / shard.at("index_file").get<std::string>();
        const fs::path blob    = planned_out_base.parent_path() / shard.at("blob_file").get<std::string>();
        planned_sidecars_exist = planned_sidecars_exist && fs::exists(sidecar);
        planned_blobs_absent   = planned_blobs_absent && !fs::exists(blob);
        planned_sidecars.push_back(read_bytes(sidecar, 0, fs::file_size(sidecar)));
    }
    check(planned_stats.shards == 2 && planned_sidecars_exist && planned_blobs_absent,
          "manifest-only writes every sidecar without writing blobs");

    wp_expert_shard::Options planned_emit           = planned;
    planned_emit.manifest_only                      = false;
    planned_emit.verify                             = true;
    planned_emit.layers                             = std::make_pair(3, 3);
    const wp_expert_shard::RunStats planned_layer_3 = wp_expert_shard::run(planned_emit);
    const fs::path                  planned_layer_3_sidecar =
        planned_out_base.parent_path() / planned_manifest.at("shards").at(0).at("index_file").get<std::string>();
    check(planned_layer_3.shards == 1 &&
              read_bytes(planned_layer_3_sidecar, 0, fs::file_size(planned_layer_3_sidecar)) == planned_sidecars.at(0),
          "data emission accepts and preserves a matching metadata-only sidecar");

    wp_expert_shard::Options emit;
    emit.src_manifest                       = fixture.source_manifest;
    emit.out_base                           = fixture.out_base;
    emit.expert_first                       = 0;
    emit.expert_last                        = 1;
    emit.verify                             = true;
    emit.layers                             = std::make_pair(3, 3);
    const wp_expert_shard::RunStats layer_3 = wp_expert_shard::run(emit);
    emit.layers                             = std::make_pair(4, 4);
    const wp_expert_shard::RunStats layer_4 = wp_expert_shard::run(emit);
    check(layer_3.shards == 1 && layer_3.groups == 2 && layer_3.members == 6 && layer_4.shards == 1 &&
              layer_4.groups == 2 && layer_4.members == 6,
          "separate layer invocations emit deterministic, non-overlapping files");

    bool                           sidecars_byte_identical = true;
    std::vector<std::vector<char>> emitted_sidecars;
    for (size_t shard_index = 0; shard_index < planned_manifest.at("shards").size(); ++shard_index) {
        const fs::path emitted_sidecar =
            fixture.dir / planned_manifest.at("shards").at(shard_index).at("index_file").get<std::string>();
        emitted_sidecars.push_back(read_bytes(emitted_sidecar, 0, fs::file_size(emitted_sidecar)));
        sidecars_byte_identical =
            sidecars_byte_identical && emitted_sidecars.back() == planned_sidecars.at(shard_index);
    }
    check(sidecars_byte_identical, "manifest-only sidecars are byte-identical to data-run sidecars");

    wp_expert_shard::Options manifest_options = emit;
    manifest_options.verify                   = false;
    manifest_options.layers.reset();
    manifest_options.manifest_only            = true;
    const wp_expert_shard::RunStats described = wp_expert_shard::run(manifest_options);
    check(described.shards == layer_3.shards + layer_4.shards && described.groups == layer_3.groups + layer_4.groups &&
              described.bytes == layer_3.bytes + layer_4.bytes,
          "manifest-only describes the independently emitted files");
    bool emitted_sidecars_unchanged = true;
    for (size_t shard_index = 0; shard_index < planned_manifest.at("shards").size(); ++shard_index) {
        const fs::path emitted_sidecar =
            fixture.dir / planned_manifest.at("shards").at(shard_index).at("index_file").get<std::string>();
        emitted_sidecars_unchanged =
            emitted_sidecars_unchanged &&
            read_bytes(emitted_sidecar, 0, fs::file_size(emitted_sidecar)) == emitted_sidecars.at(shard_index);
    }
    check(emitted_sidecars_unchanged, "manifest-only verifies matching data-run sidecars without rewriting them");

    const fs::path                  manifest_path  = wp_expert_shard::output_manifest_path(fixture.out_base);
    const std::vector<char>         manifest_bytes = read_bytes(manifest_path, 0, fs::file_size(manifest_path));
    const wp_expert_shard::RunStats repeated       = wp_expert_shard::run(manifest_options);
    check(repeated.shards == described.shards &&
              read_bytes(manifest_path, 0, fs::file_size(manifest_path)) == manifest_bytes,
          "repeated manifest-only verifies matching metadata without rewriting it");

    const fs::path mismatched_sidecar =
        planned_out_base.parent_path() / planned_manifest.at("shards").at(1).at("index_file").get<std::string>();
    {
        std::ofstream output(mismatched_sidecar, std::ios::binary | std::ios::app);
        output << ' ';
    }
    const std::vector<char> mismatched_bytes  = read_bytes(mismatched_sidecar, 0, fs::file_size(mismatched_sidecar));
    bool                    mismatch_rejected = false;
    try {
        wp_expert_shard::run(planned);
    } catch (const std::runtime_error &) {
        mismatch_rejected = true;
    }
    check(mismatch_rejected && read_bytes(mismatched_sidecar, 0, fs::file_size(mismatched_sidecar)) == mismatched_bytes,
          "manifest-only rejects a mismatched existing sidecar without overwriting it");

    std::ifstream manifest_input(manifest_path);
    json          manifest;
    manifest_input >> manifest;
    check(manifest.at("format") == "llama.cpp.weight-pager.expert-shard-manifest" && manifest.at("version") == 1,
          "preserves the loader format and version");
    check(manifest.at("model_files") == json({ "/models/toy-model.gguf" }), "preserves model_files");
    check(manifest.at("retained_expert_range").at("first") == 0 && manifest.at("retained_expert_range").at("last") == 1,
          "records the retained expert range");

    bool all_bytes_exact = true;
    bool all_contiguous  = true;
    bool all_sizes_match = true;
    bool found_expert_2  = false;
    for (const json & shard : manifest.at("shards")) {
        const fs::path output_blob       = fixture.dir / shard.at("blob_file").get<std::string>();
        const fs::path output_index_path = fixture.dir / shard.at("index_file").get<std::string>();
        std::ifstream  output_index_input(output_index_path);
        json           output_index;
        output_index_input >> output_index;

        const int      layer              = output_index.at("layer_first").get<int>();
        const int      source_shard_index = layer - 3;
        const fs::path source_index_path  = fixture.dir / Fixture::numbered("wpi.json", source_shard_index);
        const fs::path source_blob        = fixture.dir / Fixture::numbered("wpb", source_shard_index);
        std::ifstream  source_index_input(source_index_path);
        json           source_index;
        source_index_input >> source_index;

        uint64_t next_offset = 0;
        for (const json & output_group : output_index.at("groups")) {
            const int expert_idx      = output_group.at("expert_idx").get<int>();
            found_expert_2            = found_expert_2 || expert_idx == 2;
            const json & source_group = find_source_group(source_index.at("groups"), expert_idx);
            for (size_t member_index = 0; member_index < output_group.at("members").size(); ++member_index) {
                const json &   output_member = output_group.at("members").at(member_index);
                const json &   source_member = source_group.at("members").at(member_index);
                const uint64_t size          = output_member.at("size").get<uint64_t>();
                all_contiguous  = all_contiguous && output_member.at("offset").get<uint64_t>() == next_offset;
                all_bytes_exact = all_bytes_exact &&
                                  output_member.at("catalog_name") == source_member.at("catalog_name") &&
                                  read_bytes(output_blob, next_offset, size) ==
                                      read_bytes(source_blob, source_member.at("offset").get<uint64_t>(), size);
                next_offset += size;
            }
        }
        all_sizes_match = all_sizes_match && output_index.at("blob_bytes").get<uint64_t>() == next_offset &&
                          shard.at("blob_bytes").get<uint64_t>() == next_offset &&
                          fs::file_size(output_blob) == next_offset;
    }

    check(all_bytes_exact, "every retained member is byte-exact");
    check(all_contiguous, "retained groups and members are gapless and contiguous");
    check(all_sizes_match, "blob_bytes equals the actual output file size");
    check(!found_expert_2, "out-of-range expert is absent");

    const common_wp_blob_index loaded = common_wp_blob_index_load(manifest_path.string(), "/relocated/toy-model.gguf");
    check(loaded.blob_files.size() == 2 && loaded.entries.size() == 12, "existing loader accepts the emitted blob set");
}

int main() {
    test_expert_shard();
    if (g_fail != 0) {
        std::printf("\n%d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
