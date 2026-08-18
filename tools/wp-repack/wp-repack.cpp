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
#include <map>
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
constexpr int          FORMAT_VERSION_SLICED = 2;
constexpr const char * INDEX_FORMAT     = "llama.cpp.weight-pager.expert-shard-index";
constexpr const char * MANIFEST_FORMAT  = "llama.cpp.weight-pager.expert-shard-manifest";
constexpr size_t       COPY_BUFFER_SIZE = 8u * 1024u * 1024u;

// v2 output files get their own base suffix so a sliced set can never collide
// with, overwrite, or be mistaken for the v1 set built from the same model.
constexpr const char * SLICED_BASE_SUFFIX = "-eslice";

// Per-expert-tensor geometry, captured while the catalog is built. v1 never
// needed this -- it copies whole tensors and only ever touches byte offsets --
// but slicing has to know which axis the FFN intermediate dimension is on and
// where the quant block boundaries fall.
struct TensorGeom {
    int64_t   ne0       = 0;
    int64_t   ne1       = 0;
    int64_t   n_expert  = 1;
    ggml_type type      = GGML_TYPE_COUNT;
    int64_t   blck      = 0;  // elements per quant block along ne0
    size_t    type_size = 0;  // bytes per quant block
    uint64_t  row_bytes = 0;  // bytes for one full ne0 row = ne0 / blck * type_size
};

struct ModelCatalog {
    wp::PageCatalog                   catalog;
    std::vector<std::string>          files;
    std::map<std::string, TensorGeom> geom;  // source tensor name -> geometry
};

struct CliOptions {
    bool                               verify          = false;
    bool                               allow_partial   = false;
    uint64_t                           max_shard_bytes = 0;
    std::vector<wp_repack::LayerRange> layer_ranges;
    bool                               sliced          = false;
    wp_repack::SliceSpec               slice_spec;
    std::string                        model;
    std::string                        output;
};

// The shared shape of one expert across the three FFN roles, in the form the
// slicer needs. Derived once and then checked to be identical for every expert.
struct SliceGeometry {
    int64_t   n_ff           = 0;  // FFN intermediate size -- the axis we slice
    int64_t   n_embd         = 0;  // model width -- untouched, every slice is full width
    int64_t   blck           = 0;
    size_t    type_size      = 0;
    ggml_type type           = GGML_TYPE_COUNT;
    uint64_t  gate_row_bytes = 0;  // up/gate: bytes of one FFN row      (contiguous run)
    uint64_t  down_row_bytes = 0;  // down:    bytes of one n_embd row   (we take a sub-run)

    // Bytes one slice of width `w` contributes, per role. Identical for all three
    // roles, because every role holds exactly n_embd * n_ff elements per expert.
    uint64_t role_slice_bytes(int64_t w) const {
        return static_cast<uint64_t>(w / blck) * static_cast<uint64_t>(n_embd) * type_size;
    }
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
              << "expert slicing (format v2):\n"
              << "  --expert-slices SPEC   also split every expert across N slices of the FFN\n"
              << "                         intermediate dimension. SPEC is either explicit widths\n"
              << "                         in elements (\"1024,512,256,256\") or bandwidth ratios\n"
              << "                         (\"4:2:1:1\"), which are solved against the model's real\n"
              << "                         n_ff_exp. Enables format v2; v1 is unaffected.\n\n"
              << "Existing output files are never overwritten.\n"
              << "\n"
              << "FORMAT v1 (default)\n"
              << "  BASE-experts-NNNNN-of-MMMMM.wpb        headerless expert-major blob\n"
              << "  BASE-experts-NNNNN-of-MMMMM.wpi.json   per-blob index\n"
              << "  BASE-experts-manifest.json             set manifest\n"
              << "  A .wpb is a concatenation of expert groups. A group is one (block_idx,\n"
              << "  expert_idx) and holds its three role members (up, gate, down) whole, in\n"
              << "  role_mask order, byte-tight with no padding.\n"
              << "\n"
              << "FORMAT v2 (--expert-slices)\n"
              << "  BASE" << SLICED_BASE_SUFFIX << "-experts-NNNNN-of-MMMMM.wpb        sliced expert-major blob\n"
              << "  BASE" << SLICED_BASE_SUFFIX << "-experts-NNNNN-of-MMMMM.wpi.json   per-blob index\n"
              << "  BASE" << SLICED_BASE_SUFFIX << "-experts-manifest.json             set manifest\n"
              << "  Written alongside v1, never instead of it. The distinct base suffix means a\n"
              << "  v2 run can never overwrite or be confused with a v1 set from the same model.\n"
              << "\n"
              << "  A SLICE is a contiguous range [a, b) of the FFN intermediate dimension of a\n"
              << "  single expert. It is self-contained: it consumes the full-width activation\n"
              << "  and emits a full-width n_embd output, so per-slice outputs are summed by the\n"
              << "  consumer with no cross-slice traffic. A slice holds\n"
              << "\n"
              << "      ffn_up_exps   rows    [a, b)   -- contiguous in the source tensor\n"
              << "      ffn_gate_exps rows    [a, b)   -- contiguous in the source tensor\n"
              << "      ffn_down_exps columns [a, b)   -- STRIDED in the source; gathered here\n"
              << "\n"
              << "  down_exps is stored [n_ff_exp, n_embd] with n_ff_exp on the contiguous axis,\n"
              << "  so slicing it on the FFN dimension is a column cut. This tool performs that\n"
              << "  gather on the CPU at repack time and writes the result contiguously, so the\n"
              << "  runtime still does one flat read per slice. The gathered slice is a valid\n"
              << "  [b-a, n_embd] tensor of the SAME ggml type.\n"
              << "\n"
              << "  NO REQUANTIZATION HAPPENS, EVER. Every output byte is a verbatim copy of a\n"
              << "  source byte; only their order changes. That holds because slice boundaries\n"
              << "  are required to be multiples of the expert tensor's quant block size, so a\n"
              << "  quant block is never cut and its scale never has to be recomputed. A width\n"
              << "  that is not a block multiple is a hard error, not a rounding.\n"
              << "\n"
              << "  Blob layout is expert-major, then slice-major, then role-major:\n"
              << "      group 0: [slice 0: up|gate|down] [slice 1: up|gate|down] ...\n"
              << "      group 1: ...\n"
              << "  so each (group, slice) is ONE contiguous byte range -- the page-in unit for\n"
              << "  the GPU that owns that slice: a single flat read, no scatter at runtime.\n"
              << "\n"
              << "  Per-role bytes for a slice of width w are identical across all three roles:\n"
              << "      w / blck * n_embd * type_size\n"
              << "  so a slice costs exactly 3x that, and the widths alone determine the split.\n"
              << "\n"
              << "  The v2 index adds, per group, a \"slices\" array giving each slice's index,\n"
              << "  ff_first, ff_last (exclusive), blob offset and byte size, with the same\n"
              << "  per-member records as v1 plus each member's sliced shape. The v2 manifest\n"
              << "  adds a top-level \"expert_slicing\" block recording the spec text, the ratios\n"
              << "  if any, the resolved widths, n_ff_exp, n_embd, the ggml type and its block\n"
              << "  size -- enough for a consumer to validate the geometry without the model.\n";
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
        } else if (arg == "--expert-slices") {
            if (++i >= argc) {
                throw std::invalid_argument("--expert-slices requires a value");
            }
            // Slicing is orthogonal to how layers are grouped into output files,
            // so it deliberately does NOT count as a sharding mode.
            options.slice_spec = wp_repack::parse_slice_spec(argv[i]);
            options.sliced     = true;
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
    if (options.verify && options.sliced) {
        // --verify reads the format version out of the metadata it is handed, so
        // it needs no hint. Point it at the v2 manifest, or at the "BASE-eslice"
        // base; taking a spec here would let it disagree with what was written.
        throw std::invalid_argument("--expert-slices is not valid with --verify; pass the v2 manifest or output base");
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

            // Record geometry for every tensor. v1 ignores it; the slicer needs
            // ne0/ne1 and the quant block size to know where it may legally cut.
            if (const ggml_tensor * shape = ggml_get_tensor(tensor_ctx.get(), name.c_str())) {
                TensorGeom geom;
                geom.ne0       = shape->ne[0];
                geom.ne1       = shape->ne[1];
                geom.n_expert  = shape->ne[2] > 1 ? shape->ne[2] : (shape->ne[3] > 1 ? shape->ne[3] : 1);
                geom.type      = shape->type;
                geom.blck      = ggml_blck_size(shape->type);
                geom.type_size = ggml_type_size(shape->type);
                if (geom.blck > 0 && geom.ne0 % geom.blck == 0) {
                    geom.row_bytes = static_cast<uint64_t>(geom.ne0 / geom.blck) * geom.type_size;
                }
                result.geom.emplace(name, geom);
            }

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

// ---------------------------------------------------------------------------
// Format v2 -- expert slicing.
// ---------------------------------------------------------------------------

const wp_repack::ExpertMember & member_for_role(const wp_repack::ExpertGroup & group, uint8_t role, const char * what) {
    const wp_repack::ExpertMember * found = nullptr;
    for (const wp_repack::ExpertMember & member : group.members) {
        if (member.role_mask == role) {
            if (found != nullptr) {
                throw std::runtime_error(std::string("expert group has two ") + what + " members: blk " +
                                         std::to_string(group.block_idx) + " expert " +
                                         std::to_string(group.expert_idx));
            }
            found = &member;
        }
    }
    if (found == nullptr) {
        throw std::runtime_error(std::string("expert group has no ") + what + " member: blk " +
                                 std::to_string(group.block_idx) + " expert " + std::to_string(group.expert_idx));
    }
    return *found;
}

const TensorGeom & geom_for(const std::map<std::string, TensorGeom> & geom, const std::string & name) {
    const auto it = geom.find(name);
    if (it == geom.end()) {
        throw std::runtime_error("no tensor geometry recorded for " + name);
    }
    return it->second;
}

// Derive the one expert shape the whole model must share, and refuse anything
// this slicer cannot cut byte-exactly. Every check here is a refusal to guess:
// if the layout is not the (up/gate = [n_embd, n_ff], down = [n_ff, n_embd])
// arrangement, a "slice" would silently mean something else.
SliceGeometry derive_slice_geometry(const std::vector<wp_repack::ExpertGroup> & groups,
                                    const std::map<std::string, TensorGeom> &   geom_map) {
    SliceGeometry out;
    bool          first = true;

    for (const wp_repack::ExpertGroup & group : groups) {
        if (group.members.size() != 3) {
            throw std::runtime_error("expert slicing needs exactly three role members (up, gate, down); blk " +
                                     std::to_string(group.block_idx) + " expert " +
                                     std::to_string(group.expert_idx) + " has " +
                                     std::to_string(group.members.size()));
        }

        const wp_repack::ExpertMember & up   = member_for_role(group, wp::ROLE_UP, "up");
        const wp_repack::ExpertMember & gate = member_for_role(group, wp::ROLE_GATE, "gate");
        const wp_repack::ExpertMember & down = member_for_role(group, wp::ROLE_DOWN, "down");

        const TensorGeom & g_up   = geom_for(geom_map, up.source_tensor_name);
        const TensorGeom & g_gate = geom_for(geom_map, gate.source_tensor_name);
        const TensorGeom & g_down = geom_for(geom_map, down.source_tensor_name);

        const std::string where =
            "blk " + std::to_string(group.block_idx) + " expert " + std::to_string(group.expert_idx);

        if (g_up.type != g_gate.type || g_up.type != g_down.type) {
            throw std::runtime_error("expert roles do not share one ggml type at " + where +
                                     "; slicing all three on one boundary would need one block size");
        }
        if (g_up.blck <= 0 || g_up.type_size == 0) {
            throw std::runtime_error("expert tensor type has no usable quant block geometry at " + where);
        }
        if (g_up.ne0 != g_gate.ne0 || g_up.ne1 != g_gate.ne1) {
            throw std::runtime_error("ffn_up and ffn_gate shapes disagree at " + where);
        }
        // The load-bearing layout assumption, stated rather than assumed:
        // up/gate are [n_embd, n_ff] and down is [n_ff, n_embd].
        if (g_down.ne0 != g_up.ne1 || g_down.ne1 != g_up.ne0) {
            throw std::runtime_error("ffn_down is not the transpose-shaped partner of ffn_up/ffn_gate at " + where +
                                     " (expected down=[n_ff, n_embd] against up=[n_embd, n_ff]); this tool cannot "
                                     "identify the FFN intermediate axis for that layout");
        }

        const int64_t n_embd = g_up.ne0;
        const int64_t n_ff   = g_up.ne1;
        if (g_up.ne0 % g_up.blck != 0 || g_down.ne0 % g_down.blck != 0) {
            throw std::runtime_error("expert tensor rows are not a whole number of quant blocks at " + where);
        }

        const uint64_t gate_row_bytes = static_cast<uint64_t>(n_embd / g_up.blck) * g_up.type_size;
        const uint64_t down_row_bytes = static_cast<uint64_t>(n_ff / g_down.blck) * g_down.type_size;
        const uint64_t up_bytes       = gate_row_bytes * static_cast<uint64_t>(n_ff);
        const uint64_t down_bytes     = down_row_bytes * static_cast<uint64_t>(n_embd);

        // The catalog derived per-expert sizes by dividing the consolidated
        // tensor; if that disagrees with the shape math, the expert axis is not
        // where we think it is and every offset below would be wrong.
        if (up.size != up_bytes || gate.size != up_bytes || down.size != down_bytes) {
            throw std::runtime_error("per-expert byte size disagrees with tensor geometry at " + where);
        }

        if (first) {
            out.n_ff           = n_ff;
            out.n_embd         = n_embd;
            out.blck           = g_up.blck;
            out.type_size      = g_up.type_size;
            out.type           = g_up.type;
            out.gate_row_bytes = gate_row_bytes;
            out.down_row_bytes = down_row_bytes;
            first              = false;
        } else if (out.n_ff != n_ff || out.n_embd != n_embd || out.type != g_up.type) {
            // A per-layer geometry would need per-layer widths, and the whole
            // point of the widths is that they are one fixed per-GPU split.
            throw std::runtime_error("expert geometry is not uniform across the model (differs at " + where +
                                     "); this slicer requires one shape for every expert");
        }
    }

    if (first) {
        throw std::runtime_error("no expert groups to derive slice geometry from");
    }
    return out;
}

void read_member(std::ifstream & source, uint64_t offset, uint64_t size, std::vector<char> & buffer) {
    buffer.resize(static_cast<size_t>(size));
    source.clear();
    source.seekg(static_cast<std::streamoff>(offset));
    if (!source) {
        throw std::runtime_error("failed to seek source tensor bytes");
    }
    source.read(buffer.data(), static_cast<std::streamsize>(size));
    if (source.gcount() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error("short read from source GGUF");
    }
}

// Append one role's contribution to one slice. up/gate are a single contiguous
// run of whole rows; down is a column cut, so it is gathered row by row. Both
// paths copy source bytes verbatim -- never a value is recomputed.
void append_role_slice(uint8_t                      role,
                       const std::vector<char> &    role_bytes,
                       const SliceGeometry &        geom,
                       const wp_repack::SliceRange & range,
                       std::vector<char> &          out) {
    const uint64_t expected = geom.role_slice_bytes(range.width());
    const size_t   before   = out.size();

    if (role == wp::ROLE_UP || role == wp::ROLE_GATE) {
        const uint64_t begin = static_cast<uint64_t>(range.first) * geom.gate_row_bytes;
        if (begin + expected > role_bytes.size()) {
            throw std::runtime_error("up/gate slice runs past the expert payload");
        }
        out.insert(out.end(), role_bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                   role_bytes.begin() + static_cast<std::ptrdiff_t>(begin + expected));
    } else if (role == wp::ROLE_DOWN) {
        const uint64_t skip = static_cast<uint64_t>(range.first / geom.blck) * geom.type_size;
        const uint64_t run  = static_cast<uint64_t>(range.width() / geom.blck) * geom.type_size;
        for (int64_t row = 0; row < geom.n_embd; ++row) {
            const uint64_t begin = static_cast<uint64_t>(row) * geom.down_row_bytes + skip;
            if (begin + run > role_bytes.size()) {
                throw std::runtime_error("down slice runs past the expert payload");
            }
            out.insert(out.end(), role_bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                       role_bytes.begin() + static_cast<std::ptrdiff_t>(begin + run));
        }
    } else {
        throw std::runtime_error("unknown expert role mask " + std::to_string(role));
    }

    if (out.size() - before != expected) {
        throw std::runtime_error("internal slice byte count mismatch");
    }
}

// Materialize every byte of one expert group, laid out slice-major then
// role-major. Shared by the writer and the verifier so they can never drift.
void build_group_slices(const wp_repack::ExpertGroup &                group,
                        const SliceGeometry &                         geom,
                        const std::vector<wp_repack::SliceRange> &    ranges,
                        std::vector<std::ifstream> &                  sources,
                        std::vector<char> &                           scratch,
                        std::vector<char> &                           out) {
    const wp_repack::ExpertMember * roles[3] = {
        &member_for_role(group, wp::ROLE_UP, "up"),
        &member_for_role(group, wp::ROLE_GATE, "gate"),
        &member_for_role(group, wp::ROLE_DOWN, "down"),
    };
    const uint8_t role_masks[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };

    // Hold all three roles at once: the output is slice-major, so every role is
    // revisited once per slice. One expert is ~13 MB, which is cheap next to
    // re-reading each role N_slices times from disk.
    std::vector<std::vector<char>> role_bytes(3);
    for (size_t r = 0; r < 3; ++r) {
        if (roles[r]->file_idx >= sources.size()) {
            throw std::runtime_error("catalog source file index is out of range");
        }
        read_member(sources[roles[r]->file_idx], roles[r]->file_offset, roles[r]->size, scratch);
        role_bytes[r].swap(scratch);
    }

    out.clear();
    for (const wp_repack::SliceRange & range : ranges) {
        for (size_t r = 0; r < 3; ++r) {
            append_role_slice(role_masks[r], role_bytes[r], geom, range, out);
        }
    }
}

std::string hash_groups_sliced(const std::vector<wp_repack::ExpertGroup> & groups,
                               const std::vector<size_t> &                 indices,
                               const SliceGeometry &                       geom,
                               const std::vector<int64_t> &                widths) {
    sha256_t hash;
    sha256_init(&hash);
    // Domain-separated from v1: identical experts sliced differently must not
    // hash the same, or a mismatched blob set could pass --verify.
    sha_update_string(hash, "llama.cpp.wp-repack.identity.v2");
    sha_update_u64(hash, static_cast<uint64_t>(geom.n_ff));
    sha_update_u64(hash, static_cast<uint64_t>(geom.n_embd));
    sha_update_u64(hash, static_cast<uint64_t>(geom.blck));
    sha_update_u64(hash, static_cast<uint64_t>(geom.type));
    sha_update_u64(hash, widths.size());
    for (const int64_t w : widths) {
        sha_update_u64(hash, static_cast<uint64_t>(w));
    }
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

json slice_member_json(const wp_repack::ExpertGroup &                group,
                       uint8_t                                       role,
                       const char *                                  what,
                       const SliceGeometry &                         geom,
                       const wp_repack::SliceRange &                 range,
                       uint64_t                                      offset) {
    const wp_repack::ExpertMember & member = member_for_role(group, role, what);
    const bool                      is_down = role == wp::ROLE_DOWN;
    return {
        { "role_mask",          member.role_mask                                                  },
        { "size",               geom.role_slice_bytes(range.width())                              },
        { "offset",             offset                                                            },
        { "catalog_name",       member.catalog_name                                               },
        { "source_tensor_name", member.source_tensor_name                                         },
        { "source_file_idx",    member.file_idx                                                   },
        { "source_file_offset", member.file_offset                                                },
        // The shape this slice's bytes actually form, so a consumer can build the
        // tensor without re-deriving the cut. down is the gathered column slice.
        { "slice_shape",        json::array({ is_down ? range.width() : geom.n_embd,
                                              is_down ? geom.n_embd  : range.width() })           },
        { "contiguous_in_source", !is_down                                                        },
    };
}

// The slicing geometry block. Written into BOTH the manifest and every index,
// because a v1 index is documented as self-sufficient and a v2 one has more to
// be self-sufficient about: without the widths you cannot read the blob at all.
json slicing_json(const wp_repack::SliceSpec & spec, const SliceGeometry & geom) {
    json out = {
        { "spec",               spec.text                     },
        { "from_ratios",        spec.from_ratios              },
        { "ratios",             spec.ratios                   },
        { "widths",             spec.widths                   },
        { "slice_count",        spec.widths.size()            },
        { "n_ff_exp",           geom.n_ff                     },
        { "n_embd",             geom.n_embd                   },
        { "ggml_type",          static_cast<int>(geom.type)   },
        { "ggml_type_name",     ggml_type_name(geom.type)     },
        { "quant_block_size",   geom.blck                     },
        { "quant_block_bytes",  geom.type_size                },
        { "bytes_per_slice_per_role", json::array()           },
    };
    for (const int64_t w : spec.widths) {
        out["bytes_per_slice_per_role"].push_back(geom.role_slice_bytes(w));
    }
    return out;
}

json write_shard_sliced(const fs::path &                            output_base,
                        size_t                                      shard_index,
                        size_t                                      shard_count,
                        const wp_repack::ShardPlan &                shard,
                        const std::vector<wp_repack::ExpertGroup> & groups,
                        const std::vector<std::string> &            model_files,
                        const SliceGeometry &                       geom,
                        const wp_repack::SliceSpec &                spec,
                        const std::vector<wp_repack::SliceRange> &  ranges,
                        std::vector<std::ifstream> &                sources) {
    const std::vector<int64_t> & widths = spec.widths;
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
    const fs::path   temp_blob(paths.blob.string() + ".tmp");
    std::ofstream    blob(temp_blob, std::ios::binary);
    if (!blob) {
        throw std::runtime_error("failed to create " + temp_blob.string());
    }

    json index = {
        { "format",       INDEX_FORMAT                                        },
        { "version",      FORMAT_VERSION_SLICED                               },
        { "blob_file",    paths.blob.filename().string()                      },
        { "shard_index",  shard_index                                         },
        { "shard_count",  shard_count                                         },
        { "layer_first",  shard.layer_first                                   },
        { "layer_last",   shard.layer_last                                    },
        { "group_count",  shard.group_indices.size()                          },
        { "expert_slicing", slicing_json(spec, geom)                          },
        { "blob_bytes",   0                                                   },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups_sliced(groups, shard.group_indices, geom, widths) },
          }                                                                   },
        { "model_files",  model_files                                         },
        { "groups",       json::array()                                       },
    };

    uint64_t          blob_offset = 0;
    std::vector<char> scratch;
    std::vector<char> payload;
    for (size_t group_index : shard.group_indices) {
        const wp_repack::ExpertGroup & group = groups.at(group_index);
        build_group_slices(group, geom, ranges, sources, scratch, payload);

        json group_json = {
            { "block_idx",  group.block_idx      },
            { "expert_idx", group.expert_idx     },
            { "slices",     json::array()        },
        };

        uint64_t cursor = blob_offset;
        for (const wp_repack::SliceRange & range : ranges) {
            const uint64_t role_bytes  = geom.role_slice_bytes(range.width());
            const uint64_t slice_bytes = role_bytes * 3;
            group_json["slices"].push_back({
                { "slice_idx", range.index                              },
                { "ff_first",  range.first                              },
                { "ff_last",   range.last                               },
                { "width",     range.width()                            },
                { "offset",    cursor                                   },
                { "bytes",     slice_bytes                              },
                { "members",   json::array({
                       slice_member_json(group, wp::ROLE_UP,   "up",   geom, range, cursor),
                       slice_member_json(group, wp::ROLE_GATE, "gate", geom, range, cursor + role_bytes),
                       slice_member_json(group, wp::ROLE_DOWN, "down", geom, range, cursor + role_bytes * 2),
                   })                                                   },
            });
            cursor += slice_bytes;
        }
        if (cursor - blob_offset != payload.size()) {
            throw std::runtime_error("internal group payload size mismatch");
        }

        blob.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (!blob) {
            throw std::runtime_error("failed to write sliced expert blob");
        }
        blob_offset = cursor;
        index["groups"].push_back(std::move(group_json));
    }

    index["blob_bytes"] = blob_offset;

    blob.close();
    if (!blob) {
        throw std::runtime_error("failed to finish " + temp_blob.string());
    }
    // Slicing reorders bytes but never adds or drops any, so the sliced set must
    // weigh exactly what the v1 set would have.
    if (blob_offset != shard.size) {
        throw std::runtime_error("sliced blob byte count " + std::to_string(blob_offset) +
                                 " does not match the unsliced expert bytes " + std::to_string(shard.size));
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
        { "blob_bytes",   blob_offset                     },
        { "content_hash", index["content_hash"]           },
    };
}

void repack_sliced(const CliOptions &                          options,
                   const ModelCatalog &                        model,
                   const std::vector<wp_repack::ExpertGroup> & groups,
                   const std::vector<wp_repack::ShardPlan> &   shards,
                   const std::string &                         sharding_mode,
                   std::vector<std::ifstream> &                sources) {
    const SliceGeometry  geom = derive_slice_geometry(groups, model.geom);
    wp_repack::SliceSpec spec = options.slice_spec;
    wp_repack::resolve_slice_widths(spec, geom.n_ff, geom.blck);
    const std::vector<wp_repack::SliceRange> ranges = wp_repack::slice_ranges(spec.widths);

    const fs::path output_base =
        fs::absolute(fs::path(options.output + SLICED_BASE_SUFFIX)).lexically_normal();
    if (!output_base.parent_path().empty()) {
        fs::create_directories(output_base.parent_path());
    }
    ensure_outputs_absent(output_base, shards);

    const std::vector<size_t> selected = flatten_indices(shards);
    json                      manifest = {
        { "format",            MANIFEST_FORMAT                                 },
        { "version",           FORMAT_VERSION_SLICED                           },
        { "input_model",       fs::canonical(fs::path(options.model)).string() },
        { "model_files",       model.files                                     },
        { "sharding_mode",     sharding_mode                                   },
        { "total_group_count", selected.size()                                 },
        { "total_blob_bytes",  0                                               },
        { "shard_count",       shards.size()                                   },
        { "expert_slicing",    slicing_json(spec, geom)                        },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups_sliced(groups, selected, geom, spec.widths) },
          }                                                                    },
        { "shards",            json::array()                                   },
    };

    std::cout << "expert slicing: n_ff_exp=" << geom.n_ff << " n_embd=" << geom.n_embd << " type="
              << ggml_type_name(geom.type) << " blck=" << geom.blck << " slices=" << spec.widths.size() << '\n';
    for (const wp_repack::SliceRange & range : ranges) {
        std::cout << "  slice " << range.index << " ff [" << range.first << "," << range.last << ") width "
                  << range.width() << " bytes " << geom.role_slice_bytes(range.width()) * 3 << " per expert\n";
    }

    uint64_t total_bytes = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "writing sliced shard " << i + 1 << "/" << shards.size() << " layers " << shards[i].layer_first
                  << "-" << shards[i].layer_last << " groups " << shards[i].group_indices.size() << " bytes "
                  << shards[i].size << '\n';
        manifest["shards"].push_back(write_shard_sliced(output_base, i, shards.size(), shards[i], groups, model.files,
                                                        geom, spec, ranges, sources));
        total_bytes += shards[i].size;
    }
    manifest["total_blob_bytes"] = total_bytes;
    const fs::path out_manifest  = manifest_path(output_base);
    write_json(out_manifest, manifest);

    std::cout << "sliced repack complete: shards=" << shards.size() << " groups=" << selected.size()
              << " slices=" << spec.widths.size() << " bytes=" << total_bytes << " manifest=" << out_manifest.string()
              << '\n';
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

    std::vector<std::ifstream> sources;
    sources.reserve(model.files.size());
    for (const std::string & path : model.files) {
        sources.emplace_back(path, std::ios::binary);
        if (!sources.back()) {
            throw std::runtime_error("failed to open source model file: " + path);
        }
    }

    if (options.sliced) {
        // v2 writes to its own output base and returns. The v1 path below is not
        // touched, and no v1 artifact is read, written, or overwritten -- run
        // without --expert-slices to produce a v1 set from the same model.
        repack_sliced(options, model, groups, shards, sharding_mode, sources);
        return;
    }

    const fs::path output_base = fs::absolute(fs::path(options.output)).lexically_normal();
    if (!output_base.parent_path().empty()) {
        fs::create_directories(output_base.parent_path());
    }
    ensure_outputs_absent(output_base, shards);

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

// Read the format version out of metadata the caller has not yet committed to a
// version. Used by --verify, which is handed a path and must work out for itself
// whether it is looking at a v1 or a v2 set.
int detect_version(const json & value, const char * format) {
    if (value.value("format", "") != format) {
        throw std::runtime_error("unsupported or invalid repack metadata format");
    }
    const int version = value.value("version", 0);
    if (version != FORMAT_VERSION && version != FORMAT_VERSION_SLICED) {
        throw std::runtime_error("unsupported repack format version " + std::to_string(version));
    }
    return version;
}

void require_sliced_format(const json & value, const char * format) {
    if (value.value("format", "") != format || value.value("version", 0) != FORMAT_VERSION_SLICED) {
        throw std::runtime_error("unsupported or invalid sliced repack metadata format");
    }
}

// Rebuild the SliceSpec a v2 file was written with, and cross-check the geometry
// it claims against the geometry the model actually has. A blob whose sidecar
// describes a different model's shape is exactly the failure --verify exists for.
wp_repack::SliceSpec slicing_from_json(const json & value, const SliceGeometry & geom) {
    const json & block = value.at("expert_slicing");

    if (block.at("n_ff_exp").get<int64_t>() != geom.n_ff || block.at("n_embd").get<int64_t>() != geom.n_embd ||
        block.at("quant_block_size").get<int64_t>() != geom.blck ||
        block.at("quant_block_bytes").get<uint64_t>() != geom.type_size ||
        block.at("ggml_type").get<int>() != static_cast<int>(geom.type)) {
        throw std::runtime_error("recorded slice geometry disagrees with the model's expert tensors");
    }

    wp_repack::SliceSpec spec;
    spec.text        = block.at("spec").get<std::string>();
    spec.from_ratios = block.at("from_ratios").get<bool>();
    spec.ratios      = block.at("ratios").get<std::vector<int64_t>>();
    spec.widths      = block.at("widths").get<std::vector<int64_t>>();
    if (block.at("slice_count").get<uint64_t>() != spec.widths.size()) {
        throw std::runtime_error("recorded slice_count disagrees with the recorded widths");
    }
    // Re-run the same validation the writer ran. A hand-edited width list that
    // still sums to n_ff but cuts a quant block must not survive verification.
    wp_repack::SliceSpec check = spec;
    check.from_ratios          = false;
    wp_repack::resolve_slice_widths(check, geom.n_ff, geom.blck);
    return spec;
}

// Defined below with the v1 verifier; the sliced verifier reuses them unchanged.
std::vector<size_t> indices_for_range(const std::vector<wp_repack::ExpertGroup> & groups, int first, int last);
void                add_counts(VerifyCounts & total, const VerifyCounts & value);

VerifyCounts verify_index_sliced(const fs::path &                            index_path,
                                 const json &                                index,
                                 const std::vector<wp_repack::ExpertGroup> & groups,
                                 const SliceGeometry &                       geom,
                                 std::vector<std::ifstream> &                sources) {
    require_sliced_format(index, INDEX_FORMAT);
    const int layer_first = index.at("layer_first").get<int>();
    const int layer_last  = index.at("layer_last").get<int>();
    if (layer_first < 0 || layer_last < layer_first) {
        throw std::runtime_error("invalid shard layer range");
    }

    const wp_repack::SliceSpec               spec   = slicing_from_json(index, geom);
    const std::vector<wp_repack::SliceRange> ranges = wp_repack::slice_ranges(spec.widths);

    const std::vector<size_t> expected_indices = indices_for_range(groups, layer_first, layer_last);
    const json &              indexed_groups   = index.at("groups");
    if (!indexed_groups.is_array() || indexed_groups.size() != expected_indices.size() ||
        index.at("group_count").get<uint64_t>() != expected_indices.size()) {
        throw std::runtime_error("group count mismatch");
    }

    if (index.at("content_hash").at("algorithm").get<std::string>() != "sha256" ||
        index.at("content_hash").at("value").get<std::string>() !=
            hash_groups_sliced(groups, expected_indices, geom, spec.widths)) {
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
    std::vector<char> scratch;
    std::vector<char> expected_payload;
    std::vector<char> actual_payload;

    for (size_t i = 0; i < expected_indices.size(); ++i) {
        const wp_repack::ExpertGroup & expected = groups[expected_indices[i]];
        const json &                   actual   = indexed_groups.at(i);
        if (actual.at("block_idx").get<int>() != expected.block_idx ||
            actual.at("expert_idx").get<int>() != expected.expert_idx) {
            throw std::runtime_error("group identity mismatch");
        }

        const json & slices = actual.at("slices");
        if (!slices.is_array() || slices.size() != ranges.size()) {
            throw std::runtime_error("slice count mismatch for blk " + std::to_string(expected.block_idx) +
                                     " expert " + std::to_string(expected.expert_idx));
        }

        // Regenerate the payload from the source GGUF using the same gather the
        // writer used, then compare it byte for byte with what is on disk. This
        // is what makes the down-column gather trustworthy: a wrong stride shows
        // up here rather than as quiet garbage at inference time.
        build_group_slices(expected, geom, ranges, sources, scratch, expected_payload);

        const uint64_t group_offset = next_offset;
        uint64_t       cursor       = group_offset;
        for (size_t s = 0; s < ranges.size(); ++s) {
            const json &                  slice_json  = slices.at(s);
            const wp_repack::SliceRange & range       = ranges[s];
            const uint64_t                role_bytes  = geom.role_slice_bytes(range.width());
            const uint64_t                slice_bytes = role_bytes * 3;
            if (slice_json.at("slice_idx").get<int>() != range.index ||
                slice_json.at("ff_first").get<int64_t>() != range.first ||
                slice_json.at("ff_last").get<int64_t>() != range.last ||
                slice_json.at("width").get<int64_t>() != range.width() ||
                slice_json.at("bytes").get<uint64_t>() != slice_bytes) {
                throw std::runtime_error("slice descriptor mismatch");
            }
            if (slice_json.at("offset").get<uint64_t>() != cursor) {
                throw std::runtime_error("slice blob offsets are not contiguous");
            }

            const json & members = slice_json.at("members");
            if (!members.is_array() || members.size() != 3) {
                throw std::runtime_error("a slice must have exactly three role members");
            }
            const uint8_t order[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };
            const char *  names[3] = { "up", "gate", "down" };
            for (size_t r = 0; r < 3; ++r) {
                const wp_repack::ExpertMember & src = member_for_role(expected, order[r], names[r]);
                const json &                    m   = members.at(r);
                if (m.at("role_mask").get<uint8_t>() != src.role_mask || m.at("size").get<uint64_t>() != role_bytes ||
                    m.at("offset").get<uint64_t>() != cursor + role_bytes * r ||
                    m.at("catalog_name").get<std::string>() != src.catalog_name ||
                    m.at("source_tensor_name").get<std::string>() != src.source_tensor_name ||
                    m.at("source_file_idx").get<uint16_t>() != src.file_idx ||
                    m.at("source_file_offset").get<uint64_t>() != src.file_offset) {
                    throw std::runtime_error("slice member identity or size mismatch for " + src.catalog_name);
                }
                ++counts.members;
            }

            cursor += slice_bytes;
            counts.bytes += slice_bytes;
        }

        if (cursor - group_offset != expected_payload.size()) {
            throw std::runtime_error("indexed group size disagrees with the regenerated payload");
        }

        actual_payload.resize(expected_payload.size());
        blob.clear();
        blob.seekg(static_cast<std::streamoff>(group_offset));
        if (!blob) {
            throw std::runtime_error("failed to seek while verifying bytes");
        }
        blob.read(actual_payload.data(), static_cast<std::streamsize>(actual_payload.size()));
        if (blob.gcount() != static_cast<std::streamsize>(actual_payload.size())) {
            throw std::runtime_error("short read while verifying bytes");
        }
        if (std::memcmp(actual_payload.data(), expected_payload.data(), expected_payload.size()) != 0) {
            throw std::runtime_error("payload byte mismatch in blk " + std::to_string(expected.block_idx) +
                                     " expert " + std::to_string(expected.expert_idx));
        }

        next_offset = cursor;
        ++counts.groups;
    }

    if (index.at("blob_bytes").get<uint64_t>() != next_offset || fs::file_size(blob_path) != next_offset) {
        throw std::runtime_error("blob byte size mismatch");
    }
    return counts;
}

VerifyCounts verify_manifest_sliced(const fs::path &                            path,
                                    const json &                                manifest,
                                    const std::vector<wp_repack::ExpertGroup> & groups,
                                    const SliceGeometry &                       geom,
                                    std::vector<std::ifstream> &                sources) {
    require_sliced_format(manifest, MANIFEST_FORMAT);
    const json & shards = manifest.at("shards");
    if (!shards.is_array() || manifest.at("shard_count").get<uint64_t>() != shards.size()) {
        throw std::runtime_error("manifest shard count mismatch");
    }

    const wp_repack::SliceSpec spec = slicing_from_json(manifest, geom);

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
        require_sliced_format(index, INDEX_FORMAT);
        if (shard.at("shard_index").get<uint64_t>() != shard_pos ||
            index.at("shard_index").get<uint64_t>() != shard_pos ||
            index.at("shard_count").get<uint64_t>() != shards.size() || index.at("layer_first").get<int>() != first ||
            index.at("layer_last").get<int>() != last || index.at("group_count") != shard.at("group_count") ||
            index.at("blob_bytes") != shard.at("blob_bytes") || index.at("blob_file") != shard.at("blob_file") ||
            index.at("content_hash") != shard.at("content_hash")) {
            throw std::runtime_error("manifest and shard index metadata disagree");
        }
        if (index.at("expert_slicing").at("widths").get<std::vector<int64_t>>() != spec.widths) {
            throw std::runtime_error("shard index slice widths disagree with the manifest");
        }
        add_counts(total, verify_index_sliced(index_path, index, groups, geom, sources));
    }

    if (manifest.at("total_group_count").get<uint64_t>() != total.groups ||
        manifest.at("total_blob_bytes").get<uint64_t>() != total.bytes) {
        throw std::runtime_error("manifest totals mismatch");
    }
    if (manifest.at("content_hash").at("algorithm").get<std::string>() != "sha256" ||
        manifest.at("content_hash").at("value").get<std::string>() !=
            hash_groups_sliced(groups, all_indices, geom, spec.widths)) {
        throw std::runtime_error("manifest structural content hash mismatch");
    }
    return total;
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

    fs::path   target    = fs::absolute(fs::path(options.output)).lexically_normal();
    const bool is_index  = target.extension() == ".json" &&
                          target.filename().string().find(".wpi.json") != std::string::npos;
    if (!is_index && target.extension() != ".json") {
        target = manifest_path(target);
    }

    // The version lives in the metadata, so --verify needs no flag to tell a v1
    // set from a v2 one; it reads what it was pointed at and checks accordingly.
    const json   root    = read_json(target);
    const int    version = detect_version(root, is_index ? INDEX_FORMAT : MANIFEST_FORMAT);
    VerifyCounts counts;

    if (version == FORMAT_VERSION_SLICED) {
        const SliceGeometry geom = derive_slice_geometry(groups, model.geom);
        counts = is_index ? verify_index_sliced(target, root, groups, geom, sources)
                          : verify_manifest_sliced(target, root, groups, geom, sources);
    } else {
        counts = is_index ? verify_index(target, root, groups, sources)
                          : verify_manifest(target, groups, sources);
    }

    std::cout << "verify PASS: format=v" << version << " shards=" << counts.shards << " groups=" << counts.groups
              << " members=" << counts.members << " bytes=" << counts.bytes << '\n';
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
