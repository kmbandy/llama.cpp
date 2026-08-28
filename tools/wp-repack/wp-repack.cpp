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
 * sizes, and names), while --verify also compares every payload byte. With
 * --slice-output-split, each slice has its own complete set of these files.
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
    bool                               manifest_only   = false;
    bool                               allow_partial   = false;
    bool                               slice_output_split = false;
    uint64_t                           max_shard_bytes = 0;
    std::vector<wp_repack::LayerRange> layer_ranges;
    bool                               sliced          = false;
    wp_repack::SliceSpec               slice_spec;
    std::string                        model;
    std::string                        output;
};

struct SliceRoleGeometry {
    ggml_type type       = GGML_TYPE_COUNT;
    int64_t   blck       = 0;  // elements per quant block along ne0
    int64_t   slice_blck = 1;  // elements per quant block along the sliced axis
    size_t    type_size  = 0;
    uint64_t  row_bytes  = 0;  // bytes for one full ne0 row
    bool      ffn_ne0    = false;
};

struct SliceGeometryVariant {
    std::array<SliceRoleGeometry, 3> roles;

    const SliceRoleGeometry & role_geometry(uint8_t role) const {
        if (role == wp::ROLE_UP) {
            return roles[0];
        }
        if (role == wp::ROLE_GATE) {
            return roles[1];
        }
        if (role == wp::ROLE_DOWN) {
            return roles[2];
        }
        throw std::runtime_error("unknown expert role mask " + std::to_string(role));
    }

    uint64_t role_slice_bytes(uint8_t role, int64_t w, int64_t n_embd) const {
        const SliceRoleGeometry & role_geom = role_geometry(role);
        if (role_geom.ffn_ne0) {
            return static_cast<uint64_t>(w / role_geom.slice_blck) * static_cast<uint64_t>(n_embd) *
                   role_geom.type_size;
        }
        return static_cast<uint64_t>(w) * role_geom.row_bytes;
    }
};

// Shape and alignment are shared by all experts. Quantization details are kept
// per group because dynamic quantization may change them from layer to layer.
struct SliceGeometry {
    int64_t                         n_ff = 0;  // FFN intermediate size -- the axis we slice
    int64_t                         n_embd = 0;  // model width -- untouched, every slice is full width
    int64_t                         blck = 0;  // binding block size for ratio solving
    std::vector<SliceGeometryVariant> variants;
    std::vector<size_t>             group_variants;

    const SliceGeometryVariant & variant_for_group(size_t group_index) const {
        if (group_index >= group_variants.size() || group_variants[group_index] >= variants.size()) {
            throw std::runtime_error("internal group geometry index is out of range");
        }
        return variants[group_variants[group_index]];
    }
};

bool same_slice_role_geometry(const SliceRoleGeometry & a, const SliceRoleGeometry & b) {
    return a.type == b.type && a.blck == b.blck && a.slice_blck == b.slice_blck && a.type_size == b.type_size &&
           a.row_bytes == b.row_bytes && a.ffn_ne0 == b.ffn_ne0;
}

bool same_slice_geometry_variant(const SliceGeometryVariant & a, const SliceGeometryVariant & b) {
    for (size_t r = 0; r < a.roles.size(); ++r) {
        if (!same_slice_role_geometry(a.roles[r], b.roles[r])) {
            return false;
        }
    }
    return true;
}

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
              << "  --manifest-only       write manifest and indexes without emitting blob files\n"
              << "  -h, --help             show this help\n\n"
              << "expert slicing (format v2):\n"
              << "  --expert-slices SPEC   also split every expert across N slices of the FFN\n"
              << "                         intermediate dimension. SPEC is either explicit widths\n"
              << "                         in elements (\"1024,512,256,256\") or bandwidth ratios\n"
              << "                         (\"4:2:1:1\"), which are solved against the model's real\n"
              << "                         n_ff_exp. Enables format v2; v1 is unaffected.\n\n"
              << "  --slice-output-split  with --expert-slices, write one self-contained v1\n"
              << "                         blob/index/manifest set per slice\n\n"
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
              << "  are required to be multiples of the quant block size for each role sliced on\n"
              << "  ne0, so a quant block is never cut and its scale never has to be recomputed.\n"
              << "  A width that violates any such role constraint is a hard error, not a rounding.\n"
              << "\n"
              << "  Blob layout is expert-major, then slice-major, then role-major:\n"
              << "      group 0: [slice 0: up|gate|down] [slice 1: up|gate|down] ...\n"
              << "      group 1: ...\n"
              << "  so each (group, slice) is ONE contiguous byte range -- the page-in unit for\n"
              << "  the GPU that owns that slice: a single flat read, no scatter at runtime.\n"
              << "\n"
              << "  Per-role bytes are calculated from each role's own type and shape. A role\n"
              << "  sliced on ne1 contributes w * row_bytes; a role sliced on ne0 contributes\n"
              << "      w / role_blck * n_embd * role_type_size\n"
              << "  A slice costs the sum of those three role sizes. Widths must satisfy every\n"
              << "  role whose FFN dimension is ne0; roles sliced on ne1 impose no alignment.\n"
              << "\n"
              << "  The v2 index adds, per group, a \"slices\" array giving each slice's index,\n"
              << "  ff_first, ff_last (exclusive), blob offset and byte size, with the same\n"
              << "  per-member records as v1 plus each member's sliced shape. The v2 manifest\n"
              << "  adds a top-level \"expert_slicing\" block recording the spec text, the ratios\n"
              << "  if any, the resolved widths, n_ff_exp, n_embd, and distinct per-group role\n"
              << "  geometry variants with their group assignments -- enough for a consumer to\n"
              << "  validate per-layer byte sizes without the model.\n"
              << "\n"
              << "FORMAT v1 split (--expert-slices --slice-output-split)\n"
              << "  BASE" << SLICED_BASE_SUFFIX << "-slice-NNNNN-experts-NNNNN-of-MMMMM.wpb\n"
              << "  BASE" << SLICED_BASE_SUFFIX << "-slice-NNNNN-experts-NNNNN-of-MMMMM.wpi.json\n"
              << "  BASE" << SLICED_BASE_SUFFIX << "-slice-NNNNN-experts-manifest.json\n"
              << "  NNNNN is the zero-based slice index. Each set uses the canonical\n"
              << "  expert-slice manifest and flat per-slice index shape.\n";
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
        } else if (arg == "--manifest-only") {
            options.manifest_only = true;
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
        } else if (arg == "--slice-output-split") {
            options.slice_output_split = true;
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
    if (options.manifest_only && options.verify) {
        throw std::invalid_argument("--manifest-only cannot be combined with --verify");
    }
    if (options.verify && options.sliced) {
        // --verify reads the format version out of the metadata it is handed, so
        // it needs no hint. Point it at the v2 manifest, or at the "BASE-eslice"
        // base; taking a spec here would let it disagree with what was written.
        throw std::invalid_argument("--expert-slices is not valid with --verify; pass the v2 manifest or output base");
    }
    if (options.slice_output_split && !options.sliced) {
        throw std::invalid_argument("--slice-output-split requires --expert-slices");
    }
    if (options.verify && options.slice_output_split) {
        throw std::invalid_argument("--slice-output-split is not valid with --verify");
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

fs::path slice_output_base(const fs::path & output_base, const wp_repack::SliceRange & range) {
    std::ostringstream suffix;
    suffix << output_base.string() << "-slice-" << std::setw(5) << std::setfill('0') << range.index;
    return fs::path(suffix.str());
}

fs::path manifest_path(const fs::path & output_base) {
    return fs::path(output_base.string() + "-experts-manifest.json");
}

void ensure_outputs_absent(const fs::path &                              output_base,
                           const std::vector<wp_repack::ShardPlan> &    shards,
                           bool                                         include_blobs = true) {
    std::vector<fs::path> paths{ manifest_path(output_base) };
    for (size_t i = 0; i < shards.size(); ++i) {
        const ShardPaths names = shard_paths(output_base, i, shards.size());
        if (include_blobs) {
            paths.push_back(names.blob);
        }
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

json build_shard_index(const fs::path &                            output_base,
                       size_t                                      shard_index,
                       size_t                                      shard_count,
                       const wp_repack::ShardPlan &                shard,
                       const std::vector<wp_repack::ExpertGroup> & groups,
                       const std::vector<std::string> &            model_files) {
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
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

    uint64_t blob_offset = 0;
    for (size_t group_index : shard.group_indices) {
        const wp_repack::ExpertGroup & group      = groups.at(group_index);
        json                           group_json = {
            { "block_idx",    group.block_idx      },
            { "expert_idx",   group.expert_idx     },
            { "member_count", group.members.size() },
            { "members",      json::array()        },
        };

        for (const wp_repack::ExpertMember & member : group.members) {
            if (member.file_idx >= model_files.size()) {
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
            blob_offset += member.size;
        }
        index["groups"].push_back(std::move(group_json));
    }

    if (blob_offset != shard.size) {
        throw std::runtime_error("internal shard byte count mismatch");
    }
    return index;
}

json shard_manifest_entry(const ShardPaths & paths, const json & index) {
    return {
        { "blob_file",    paths.blob.filename().string()  },
        { "index_file",   paths.index.filename().string() },
        { "shard_index",  index["shard_index"]             },
        { "layer_first",  index["layer_first"]             },
        { "layer_last",   index["layer_last"]              },
        { "group_count",  index["group_count"]             },
        { "blob_bytes",   index["blob_bytes"]              },
        { "content_hash", index["content_hash"]            },
    };
}

json write_shard(const fs::path &                            output_base,
                 size_t                                      shard_index,
                 size_t                                      shard_count,
                 const wp_repack::ShardPlan &                shard,
                 const std::vector<wp_repack::ExpertGroup> & groups,
                 const std::vector<std::string> &            model_files,
                 std::vector<std::ifstream> &                sources,
                 bool                                        manifest_only) {
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
    const json        index = build_shard_index(output_base, shard_index, shard_count, shard, groups, model_files);

    if (!manifest_only) {
        const fs::path temp_blob(paths.blob.string() + ".tmp");
        std::ofstream  blob(temp_blob, std::ios::binary);
        if (!blob) {
            throw std::runtime_error("failed to create " + temp_blob.string());
        }

        uint64_t          blob_offset = 0;
        std::vector<char> buffer(COPY_BUFFER_SIZE);
        for (size_t group_index : shard.group_indices) {
            const wp_repack::ExpertGroup & group = groups.at(group_index);
            for (const wp_repack::ExpertMember & member : group.members) {
                if (member.file_idx >= sources.size()) {
                    throw std::runtime_error("catalog source file index is out of range");
                }
                copy_member(sources[member.file_idx], member.file_offset, member.size, blob, buffer);
                blob_offset += member.size;
            }
        }

        blob.close();
        if (!blob) {
            throw std::runtime_error("failed to finish " + temp_blob.string());
        }
        if (blob_offset != shard.size || fs::file_size(temp_blob) != shard.size) {
            throw std::runtime_error("internal shard byte count mismatch");
        }
        fs::rename(temp_blob, paths.blob);
    }
    write_json(paths.index, index);

    return shard_manifest_entry(paths, index);
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

// Derive the shared expert shape and per-group quantization details, and refuse
// anything this slicer cannot cut byte-exactly. Every check here is a refusal to guess:
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
        const TensorGeom * role_geom_sources[3] = { &g_up, &g_gate, &g_down };
        for (const TensorGeom * role_geom : role_geom_sources) {
            if (role_geom->blck <= 0 || role_geom->type_size == 0 || role_geom->ne0 % role_geom->blck != 0) {
                throw std::runtime_error("expert tensor type has no usable quant block geometry at " + where);
            }
        }

        SliceGeometryVariant variant;
        // Assigning to a std::array from a braced list whose elements are themselves
        // braced aggregate initialisers needs the array type spelled out; the inner
        // braces are otherwise read as arguments to array's own operator=.
        variant.roles = std::array<SliceRoleGeometry, 3>{ {
            { g_up.type, g_up.blck, 1, g_up.type_size, g_up.row_bytes, false },
            { g_gate.type, g_gate.blck, 1, g_gate.type_size, g_gate.row_bytes, false },
            { g_down.type, g_down.blck, g_down.blck, g_down.type_size, g_down.row_bytes, true },
        } };
        const uint64_t role_bytes[3] = {
            static_cast<uint64_t>(n_ff) * variant.roles[0].row_bytes,
            static_cast<uint64_t>(n_ff) * variant.roles[1].row_bytes,
            static_cast<uint64_t>(n_embd) * static_cast<uint64_t>(n_ff / variant.roles[2].slice_blck) *
                variant.roles[2].type_size,
        };

        // The catalog derived per-expert sizes by dividing the consolidated
        // tensor; if that disagrees with the shape math, the expert axis is not
        // where we think it is and every offset below would be wrong.
        if (up.size != role_bytes[0] || gate.size != role_bytes[1] || down.size != role_bytes[2]) {
            throw std::runtime_error("per-expert byte size disagrees with tensor geometry at " + where);
        }

        if (first) {
            out.n_ff   = n_ff;
            out.n_embd = n_embd;
            out.blck   = std::max({ variant.roles[0].slice_blck, variant.roles[1].slice_blck,
                                    variant.roles[2].slice_blck });
            first     = false;
        } else if (out.n_ff != n_ff || out.n_embd != n_embd) {
            throw std::runtime_error("expert geometry is not uniform across the model (differs at " + where +
                                     "); n_ff and n_embd must be uniform across every expert");
        } else if (out.blck != std::max({ variant.roles[0].slice_blck, variant.roles[1].slice_blck,
                                          variant.roles[2].slice_blck })) {
            throw std::runtime_error("expert slice alignment is not uniform across the model (differs at " + where +
                                     "); one width set must be legal for every expert");
        }

        size_t variant_index = 0;
        for (; variant_index < out.variants.size(); ++variant_index) {
            if (same_slice_geometry_variant(out.variants[variant_index], variant)) {
                break;
            }
        }
        if (variant_index == out.variants.size()) {
            out.variants.push_back(variant);
        }
        out.group_variants.push_back(variant_index);
    }

    if (first) {
        throw std::runtime_error("no expert groups to derive slice geometry from");
    }
    return out;
}

void validate_slice_ranges(const std::vector<wp_repack::SliceRange> & ranges, const SliceGeometry & geom) {
    for (size_t v = 0; v < geom.variants.size(); ++v) {
        const SliceGeometryVariant & variant = geom.variants[v];
        for (size_t r = 0; r < variant.roles.size(); ++r) {
            const SliceRoleGeometry & role = variant.roles[r];
            if (!role.ffn_ne0) {
                continue;
            }
            for (const wp_repack::SliceRange & range : ranges) {
                if (range.first % role.slice_blck != 0 || range.last % role.slice_blck != 0) {
                    throw std::invalid_argument("slice " + std::to_string(range.index) + " boundary [" +
                                                std::to_string(range.first) + "," + std::to_string(range.last) +
                                                ") is not aligned to geometry variant " + std::to_string(v) +
                                                " role " + std::to_string(r) + " quant block size " +
                                                std::to_string(role.slice_blck));
                }
            }
        }
    }
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

void read_group_roles(const wp_repack::ExpertGroup &                group,
                      std::vector<std::ifstream> &                  sources,
                      std::vector<char> &                           scratch,
                      std::vector<std::vector<char>> &              role_bytes) {
    const wp_repack::ExpertMember * roles[3] = {
        &member_for_role(group, wp::ROLE_UP, "up"),
        &member_for_role(group, wp::ROLE_GATE, "gate"),
        &member_for_role(group, wp::ROLE_DOWN, "down"),
    };
    role_bytes.resize(3);
    for (size_t r = 0; r < 3; ++r) {
        if (roles[r]->file_idx >= sources.size()) {
            throw std::runtime_error("catalog source file index is out of range");
        }
        read_member(sources[roles[r]->file_idx], roles[r]->file_offset, roles[r]->size, scratch);
        role_bytes[r].swap(scratch);
    }
}

// Append one role's contribution to one slice. up/gate are a single contiguous
// run of whole rows; down is a column cut, so it is gathered row by row. Both
// paths copy source bytes verbatim -- never a value is recomputed.
void append_role_slice(uint8_t                      role,
                       const std::vector<char> &    role_bytes,
                       const SliceGeometryVariant & variant,
                       int64_t                       n_embd,
                       const wp_repack::SliceRange & range,
                       std::vector<char> &          out) {
    const SliceRoleGeometry & role_geom = variant.role_geometry(role);
    const uint64_t            expected  = variant.role_slice_bytes(role, range.width(), n_embd);
    const size_t              before    = out.size();

    if (role == wp::ROLE_UP || role == wp::ROLE_GATE) {
        const uint64_t begin = static_cast<uint64_t>(range.first) * role_geom.row_bytes;
        if (begin + expected > role_bytes.size()) {
            throw std::runtime_error("up/gate slice runs past the expert payload");
        }
        out.insert(out.end(), role_bytes.begin() + static_cast<std::ptrdiff_t>(begin),
                   role_bytes.begin() + static_cast<std::ptrdiff_t>(begin + expected));
    } else if (role == wp::ROLE_DOWN) {
        const uint64_t skip = static_cast<uint64_t>(range.first / role_geom.blck) * role_geom.type_size;
        const uint64_t run  = static_cast<uint64_t>(range.width() / role_geom.blck) * role_geom.type_size;
        for (int64_t row = 0; row < n_embd; ++row) {
            const uint64_t begin = static_cast<uint64_t>(row) * role_geom.row_bytes + skip;
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
                        const SliceGeometryVariant &                  variant,
                        int64_t                                        n_embd,
                        const std::vector<wp_repack::SliceRange> &    ranges,
                        std::vector<std::ifstream> &                  sources,
                        std::vector<char> &                           scratch,
                        std::vector<char> &                           out) {
    const uint8_t role_masks[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };

    // Hold all three roles at once: the output is slice-major, so every role is
    // revisited once per slice. One expert is ~13 MB, which is cheap next to
    // re-reading each role N_slices times from disk.
    std::vector<std::vector<char>> role_bytes(3);
    read_group_roles(group, sources, scratch, role_bytes);

    out.clear();
    for (const wp_repack::SliceRange & range : ranges) {
        for (size_t r = 0; r < 3; ++r) {
            append_role_slice(role_masks[r], role_bytes[r], variant, n_embd, range, out);
        }
    }
}

std::string hash_groups_sliced(const std::vector<wp_repack::ExpertGroup> & groups,
                               const std::vector<size_t> &                 indices,
                               const SliceGeometry &                       geom,
                               const std::vector<int64_t> &                widths,
                               const wp_repack::SliceRange *               output_range = nullptr) {
    sha256_t hash;
    sha256_init(&hash);
    // Domain-separated from v1: identical experts sliced differently must not
    // hash the same, or a mismatched blob set could pass --verify.
    sha_update_string(hash, "llama.cpp.wp-repack.identity.v2");
    sha_update_u64(hash, static_cast<uint64_t>(geom.n_ff));
    sha_update_u64(hash, static_cast<uint64_t>(geom.n_embd));
    sha_update_u64(hash, static_cast<uint64_t>(geom.blck));
    sha_update_u64(hash, widths.size());
    for (const int64_t w : widths) {
        sha_update_u64(hash, static_cast<uint64_t>(w));
    }
    if (output_range != nullptr) {
        sha_update_string(hash, "llama.cpp.wp-repack.identity.v2.slice-output-split");
        sha_update_u64(hash, static_cast<uint64_t>(output_range->index));
        sha_update_u64(hash, static_cast<uint64_t>(output_range->first));
        sha_update_u64(hash, static_cast<uint64_t>(output_range->last));
    }
    sha_update_u64(hash, indices.size());

    for (size_t index : indices) {
        if (index >= groups.size()) {
            throw std::runtime_error("internal group index is out of range");
        }
        const wp_repack::ExpertGroup & group = groups[index];
        const SliceGeometryVariant & variant = geom.variant_for_group(index);
        sha_update_u64(hash, static_cast<uint64_t>(group.block_idx));
        sha_update_u64(hash, static_cast<uint64_t>(group.expert_idx));
        for (const SliceRoleGeometry & role : variant.roles) {
            sha_update_u64(hash, static_cast<uint64_t>(role.type));
            sha_update_u64(hash, static_cast<uint64_t>(role.blck));
            sha_update_u64(hash, static_cast<uint64_t>(role.slice_blck));
            sha_update_u64(hash, static_cast<uint64_t>(role.type_size));
            sha_update_u64(hash, role.row_bytes);
            sha_update_u64(hash, role.ffn_ne0 ? 1 : 0);
        }
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

std::string hash_groups_flat_sliced(const std::vector<wp_repack::ExpertGroup> & groups,
                                    const std::vector<size_t> &                 indices,
                                    const SliceGeometry &                       geom,
                                    const wp_repack::SliceRange &               range) {
    // This is the v1 hash used by llama-wp-expert-shard for flat groups.
    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-repack.identity.v1");
    sha_update_u64(hash, indices.size());
    for (size_t index : indices) {
        if (index >= groups.size()) {
            throw std::runtime_error("internal group index is out of range");
        }
        const wp_repack::ExpertGroup & group = groups[index];
        const SliceGeometryVariant & variant = geom.variant_for_group(index);
        sha_update_u64(hash, static_cast<uint64_t>(group.block_idx));
        sha_update_u64(hash, static_cast<uint64_t>(group.expert_idx));
        sha_update_u64(hash, static_cast<uint64_t>(range.index));
        sha_update_u64(hash, static_cast<uint64_t>(range.first));
        sha_update_u64(hash, static_cast<uint64_t>(range.last));
        sha_update_u64(hash, group.members.size());
        for (const wp_repack::ExpertMember & member : group.members) {
            sha_update_u64(hash, member.role_mask);
            sha_update_u64(hash, variant.role_slice_bytes(member.role_mask, range.width(), geom.n_embd));
            sha_update_string(hash, member.catalog_name);
            sha_update_string(hash, member.source_tensor_name);
        }
    }
    return finish_sha(hash);
}

json slice_member_json(const wp_repack::ExpertGroup &                group,
                       uint8_t                                       role,
                       const char *                                  what,
                       const SliceGeometryVariant &                  variant,
                       int64_t                                        n_embd,
                       const wp_repack::SliceRange &                 range,
                       uint64_t                                      offset) {
    const wp_repack::ExpertMember & member = member_for_role(group, role, what);
    const bool                      is_down = role == wp::ROLE_DOWN;
    return {
        { "role_mask",          member.role_mask                         },
        { "size",               variant.role_slice_bytes(role, range.width(), n_embd) },
        { "offset",             offset                                   },
        { "catalog_name",       member.catalog_name                      },
        { "source_tensor_name", member.source_tensor_name                },
        { "source_file_idx",    member.file_idx                          },
        { "source_file_offset", member.file_offset                       },
        // The shape this slice's bytes actually form, so a consumer can build the
        // tensor without re-deriving the cut. down is the gathered column slice.
        { "slice_shape",        json::array({ is_down ? range.width() : n_embd,
                                              is_down ? n_embd : range.width() })                  },
        { "contiguous_in_source", !is_down                                                        },
    };
}

// The slicing geometry block. Written into BOTH the manifest and every index,
// because a v1 index is documented as self-sufficient and a v2 one has more to
// be self-sufficient about: without the widths you cannot read the blob at all.
json slicing_json(const wp_repack::SliceSpec &                         spec,
                  const SliceGeometry &                               geom,
                  const std::vector<wp_repack::ExpertGroup> &         groups) {
    const uint8_t role_masks[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };
    const char * role_names[3]  = { "up", "gate", "down" };
    json out = {
        { "spec",              spec.text                },
        { "from_ratios",       spec.from_ratios         },
        { "ratios",            spec.ratios              },
        { "widths",            spec.widths              },
        { "slice_count",       spec.widths.size()       },
        { "n_ff_exp",          geom.n_ff                },
        { "n_embd",            geom.n_embd              },
        { "slice_alignment",   geom.blck                },
        { "geometry_variants", json::array()            },
    };

    for (size_t v = 0; v < geom.variants.size(); ++v) {
        const SliceGeometryVariant & variant = geom.variants[v];
        json variant_json = {
            { "variant_idx",               v                 },
            { "groups",                     json::array()     },
            { "role_geometry",              json::object()   },
            { "bytes_per_slice_per_role",   json::array()     },
        };
        for (size_t i = 0; i < groups.size(); ++i) {
            if (geom.group_variants[i] == v) {
                variant_json["groups"].push_back({
                    { "block_idx",  groups[i].block_idx  },
                    { "expert_idx", groups[i].expert_idx },
                });
            }
        }
        for (size_t r = 0; r < 3; ++r) {
            const SliceRoleGeometry & role = variant.roles[r];
            variant_json["role_geometry"][role_names[r]] = {
                { "ggml_type",        static_cast<int>(role.type) },
                { "ggml_type_name",   ggml_type_name(role.type)   },
                { "quant_block_size", role.blck                },
                { "slice_block_size", role.slice_blck          },
                { "quant_block_bytes", role.type_size          },
                { "row_bytes",         role.row_bytes           },
                { "ffn_axis",          role.ffn_ne0 ? 0 : 1      },
            };
        }
        for (const int64_t w : spec.widths) {
            json role_bytes = json::array();
            for (const uint8_t role : role_masks) {
                role_bytes.push_back(variant.role_slice_bytes(role, w, geom.n_embd));
            }
            variant_json["bytes_per_slice_per_role"].push_back(std::move(role_bytes));
        }
        out["geometry_variants"].push_back(std::move(variant_json));
    }
    return out;
}

json build_sliced_index(const fs::path &                            output_base,
                        size_t                                      shard_index,
                        size_t                                      shard_count,
                        const wp_repack::ShardPlan &                shard,
                        const std::vector<wp_repack::ExpertGroup> & groups,
                        const std::vector<std::string> &            model_files,
                        const SliceGeometry &                       geom,
                        const wp_repack::SliceSpec &                spec,
                        const std::vector<wp_repack::SliceRange> &  ranges) {
    const std::vector<int64_t> & widths = spec.widths;
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
    json index = {
        { "format",       INDEX_FORMAT                                        },
        { "version",      FORMAT_VERSION_SLICED                               },
        { "blob_file",    paths.blob.filename().string()                      },
        { "shard_index",  shard_index                                         },
        { "shard_count",  shard_count                                         },
        { "layer_first",  shard.layer_first                                   },
        { "layer_last",   shard.layer_last                                    },
        { "group_count",  shard.group_indices.size()                          },
        { "expert_slicing", slicing_json(spec, geom, groups)                 },
        { "blob_bytes",   0                                                   },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups_sliced(groups, shard.group_indices, geom, widths) },
          }                                                                   },
        { "model_files",  model_files                                         },
        { "groups",       json::array()                                       },
    };

    uint64_t blob_offset = 0;
    for (size_t group_index : shard.group_indices) {
        const wp_repack::ExpertGroup & group = groups.at(group_index);
        const SliceGeometryVariant & variant = geom.variant_for_group(group_index);

        json group_json = {
            { "block_idx",        group.block_idx       },
            { "expert_idx",       group.expert_idx      },
            { "geometry_variant", geom.group_variants[group_index] },
            { "slices",            json::array()         },
        };

        uint64_t cursor = blob_offset;
        for (const wp_repack::SliceRange & range : ranges) {
            const uint64_t role_bytes[3] = {
                variant.role_slice_bytes(wp::ROLE_UP, range.width(), geom.n_embd),
                variant.role_slice_bytes(wp::ROLE_GATE, range.width(), geom.n_embd),
                variant.role_slice_bytes(wp::ROLE_DOWN, range.width(), geom.n_embd),
            };
            const uint64_t slice_bytes = role_bytes[0] + role_bytes[1] + role_bytes[2];
            group_json["slices"].push_back({
                { "slice_idx", range.index                              },
                { "ff_first",  range.first                              },
                { "ff_last",   range.last                               },
                { "width",     range.width()                            },
                { "offset",    cursor                                   },
                { "bytes",     slice_bytes                              },
                { "members",   json::array({
                       slice_member_json(group, wp::ROLE_UP,   "up",   variant, geom.n_embd, range, cursor),
                       slice_member_json(group, wp::ROLE_GATE, "gate", variant, geom.n_embd, range, cursor + role_bytes[0]),
                       slice_member_json(group, wp::ROLE_DOWN, "down", variant, geom.n_embd, range,
                                         cursor + role_bytes[0] + role_bytes[1]),
                   })                                                   },
            });
            cursor += slice_bytes;
        }
        blob_offset = cursor;
        index["groups"].push_back(std::move(group_json));
    }

    // Slicing reorders bytes but never adds or drops any, so the sliced set must
    // weigh exactly what the v1 set would have.
    if (blob_offset != shard.size) {
        throw std::runtime_error("sliced blob byte count " + std::to_string(blob_offset) +
                                 " does not match the unsliced expert bytes " + std::to_string(shard.size));
    }
    index["blob_bytes"] = blob_offset;
    return index;
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
                        std::vector<std::ifstream> &                sources,
                        bool                                        manifest_only) {
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
    const json        index = build_sliced_index(output_base, shard_index, shard_count, shard, groups, model_files,
                                                 geom, spec, ranges);

    if (!manifest_only) {
        const fs::path temp_blob(paths.blob.string() + ".tmp");
        std::ofstream  blob(temp_blob, std::ios::binary);
        if (!blob) {
            throw std::runtime_error("failed to create " + temp_blob.string());
        }

        uint64_t          blob_offset = 0;
        std::vector<char> scratch;
        std::vector<char> payload;
        for (size_t group_position = 0; group_position < shard.group_indices.size(); ++group_position) {
            const size_t                    group_index = shard.group_indices[group_position];
            const wp_repack::ExpertGroup &  group = groups.at(group_index);
            const SliceGeometryVariant &    variant = geom.variant_for_group(group_index);
            build_group_slices(group, variant, geom.n_embd, ranges, sources, scratch, payload);

            const json & group_json = index["groups"].at(group_position);
            const json & last_slice = group_json["slices"].back();
            const uint64_t expected_offset = group_json["slices"].front()["offset"].get<uint64_t>();
            const uint64_t expected_end = last_slice["offset"].get<uint64_t>() + last_slice["bytes"].get<uint64_t>();
            if (expected_offset != blob_offset || expected_end - blob_offset != payload.size()) {
                throw std::runtime_error("internal group payload size mismatch");
            }

            blob.write(payload.data(), static_cast<std::streamsize>(payload.size()));
            if (!blob) {
                throw std::runtime_error("failed to write sliced expert blob");
            }
            blob_offset = expected_end;
        }

        blob.close();
        if (!blob) {
            throw std::runtime_error("failed to finish " + temp_blob.string());
        }
        if (blob_offset != index["blob_bytes"].get<uint64_t>() || fs::file_size(temp_blob) != index["blob_bytes"].get<uint64_t>()) {
            throw std::runtime_error("internal sliced shard byte count mismatch");
        }
        fs::rename(temp_blob, paths.blob);
    }
    write_json(paths.index, index);

    return shard_manifest_entry(paths, index);
}

struct SliceSplitShardOutput {
    ShardPaths                  paths;
    fs::path                    temp_blob;
    std::unique_ptr<std::ofstream> blob;
    json                        index;
    uint64_t                    blob_offset = 0;
};

json build_split_index(const fs::path &                            output_base,
                       size_t                                      shard_index,
                       size_t                                      shard_count,
                       const wp_repack::ShardPlan &                shard,
                       const std::vector<wp_repack::ExpertGroup> & groups,
                       const std::vector<std::string> &            model_files,
                       const SliceGeometry &                       geom,
                       const wp_repack::SliceSpec &                spec,
                       const wp_repack::SliceRange &                range) {
    const ShardPaths paths = shard_paths(output_base, shard_index, shard_count);
    json              expert_slicing = slicing_json(spec, geom, groups);
    expert_slicing["selected_slice"] = range.index;
    json index = {
        { "format",              INDEX_FORMAT                                        },
        { "version",             FORMAT_VERSION                                      },
        { "blob_file",           paths.blob.filename().string()                      },
        { "shard_index",         shard_index                                         },
        { "shard_count",         shard_count                                         },
        { "layer_first",         shard.layer_first                                   },
        { "layer_last",          shard.layer_last                                    },
        { "group_count",         shard.group_indices.size()                          },
        { "blob_bytes",          0                                                    },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups_sliced(groups, shard.group_indices, geom, spec.widths, &range) },
          }                                                                          },
        { "model_files",         model_files                                         },
        { "groups",              json::array()                                       },
    };
    index["expert_slicing"] = std::move(expert_slicing);

    uint64_t blob_offset = 0;
    for (size_t group_index : shard.group_indices) {
        const wp_repack::ExpertGroup & group = groups.at(group_index);
        const SliceGeometryVariant & variant = geom.variant_for_group(group_index);
        const uint64_t role_bytes[3] = {
            variant.role_slice_bytes(wp::ROLE_UP, range.width(), geom.n_embd),
            variant.role_slice_bytes(wp::ROLE_GATE, range.width(), geom.n_embd),
            variant.role_slice_bytes(wp::ROLE_DOWN, range.width(), geom.n_embd),
        };
        const uint64_t slice_bytes = role_bytes[0] + role_bytes[1] + role_bytes[2];

        index["groups"].push_back({
            { "block_idx",    group.block_idx                      },
            { "expert_idx",   group.expert_idx                     },
            { "member_count", 3                                   },
            { "slice_idx",    range.index                         },
            { "ff_first",     range.first                         },
            { "ff_last",      range.last                          },
            { "width",        range.width()                       },
            { "members",      json::array({
                   slice_member_json(group, wp::ROLE_UP,   "up",   variant, geom.n_embd, range, blob_offset),
                   slice_member_json(group, wp::ROLE_GATE, "gate", variant, geom.n_embd, range,
                                     blob_offset + role_bytes[0]),
                   slice_member_json(group, wp::ROLE_DOWN, "down", variant, geom.n_embd, range,
                                     blob_offset + role_bytes[0] + role_bytes[1]),
               })                                                    },
        });
        blob_offset += slice_bytes;
    }

    index["blob_bytes"] = blob_offset;
    return index;
}

void repack_sliced_split(const CliOptions &                          options,
                         const ModelCatalog &                        model,
                         const std::vector<wp_repack::ExpertGroup> & groups,
                         const std::vector<wp_repack::ShardPlan> &   shards,
                         const fs::path &                            output_base,
                         const SliceGeometry &                       geom,
                         const wp_repack::SliceSpec &                spec,
                         const std::vector<wp_repack::SliceRange> &  ranges,
                         std::vector<std::ifstream> &                sources) {
    const std::vector<size_t> selected = flatten_indices(shards);
    const size_t               slice_count = ranges.size();
    int                         expert_first = INT_MAX;
    int                         expert_last  = INT_MIN;
    for (size_t group_index : selected) {
        expert_first = std::min(expert_first, groups.at(group_index).expert_idx);
        expert_last  = std::max(expert_last, groups.at(group_index).expert_idx);
    }
    std::vector<fs::path>      slice_bases(slice_count);
    std::vector<SliceSplitShardOutput> outputs;
    outputs.reserve(slice_count * shards.size());

    for (size_t s = 0; s < slice_count; ++s) {
        slice_bases[s] = slice_output_base(output_base, ranges[s]);
        if (!slice_bases[s].parent_path().empty()) {
            fs::create_directories(slice_bases[s].parent_path());
        }
        ensure_outputs_absent(slice_bases[s], shards, !options.manifest_only);

        for (size_t i = 0; i < shards.size(); ++i) {
            SliceSplitShardOutput output;
            output.paths = shard_paths(slice_bases[s], i, shards.size());
            output.temp_blob = fs::path(output.paths.blob.string() + ".tmp");
            output.index = build_split_index(slice_bases[s], i, shards.size(), shards[i], groups, model.files, geom,
                                             spec, ranges[s]);
            if (!options.manifest_only) {
                output.blob = std::make_unique<std::ofstream>(output.temp_blob, std::ios::binary);
                if (!*output.blob) {
                    throw std::runtime_error("failed to create " + output.temp_blob.string());
                }
            }
            outputs.push_back(std::move(output));
        }
    }

    if (!options.manifest_only) {
        std::vector<char>              scratch;
        std::vector<char>              payload;
        std::vector<std::vector<char>> role_bytes;
        const uint8_t                  role_masks[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };

        for (size_t i = 0; i < shards.size(); ++i) {
            for (size_t group_position = 0; group_position < shards[i].group_indices.size(); ++group_position) {
                const size_t                   group_index = shards[i].group_indices[group_position];
                const wp_repack::ExpertGroup & group = groups.at(group_index);
                const SliceGeometryVariant &   variant = geom.variant_for_group(group_index);
                read_group_roles(group, sources, scratch, role_bytes);

                for (size_t s = 0; s < slice_count; ++s) {
                    SliceSplitShardOutput & output = outputs[s * shards.size() + i];
                    const wp_repack::SliceRange & range = ranges[s];
                    const uint64_t role_bytes_for_slice[3] = {
                        variant.role_slice_bytes(wp::ROLE_UP, range.width(), geom.n_embd),
                        variant.role_slice_bytes(wp::ROLE_GATE, range.width(), geom.n_embd),
                        variant.role_slice_bytes(wp::ROLE_DOWN, range.width(), geom.n_embd),
                    };
                    const uint64_t slice_bytes = role_bytes_for_slice[0] + role_bytes_for_slice[1] +
                                                  role_bytes_for_slice[2];

                    payload.clear();
                    for (size_t r = 0; r < 3; ++r) {
                        append_role_slice(role_masks[r], role_bytes[r], variant, geom.n_embd, range, payload);
                    }
                    if (payload.size() != slice_bytes) {
                        throw std::runtime_error("internal split slice payload size mismatch");
                    }

                    const json & group_json = output.index["groups"].at(group_position);
                    const uint64_t expected_offset = group_json["members"].front()["offset"].get<uint64_t>();
                    if (expected_offset != output.blob_offset) {
                        throw std::runtime_error("internal split slice offset mismatch");
                    }
                    output.blob->write(payload.data(), static_cast<std::streamsize>(payload.size()));
                    if (!*output.blob) {
                        throw std::runtime_error("failed to write split sliced expert blob");
                    }
                    output.blob_offset += slice_bytes;
                }
            }
        }
    }

    for (size_t s = 0; s < slice_count; ++s) {
        json expert_slicing = slicing_json(spec, geom, groups);
        expert_slicing["selected_slice"] = ranges[s].index;
        json manifest = {
            { "format",              MANIFEST_FORMAT                                 },
            { "version",             FORMAT_VERSION                                   },
            { "input_model",         fs::canonical(fs::path(options.model)).string() },
            { "model_files",         model.files                                     },
            { "sharding_mode",       "expert-slice"                                  },
            { "retained_expert_range",
             {
                  { "first", expert_first },
                  { "last",  expert_last  },
              }                                                                    },
            { "total_group_count",   selected.size()                                 },
            { "total_blob_bytes",    0                                               },
            { "shard_count",         shards.size()                                   },
            { "content_hash",
             {
                  { "algorithm", "sha256" },
                  { "value", hash_groups_sliced(groups, selected, geom, spec.widths, &ranges[s]) },
              }                                                                        },
            { "shards",              json::array()                                   },
        };
        manifest["expert_slicing"] = std::move(expert_slicing);

        uint64_t total_bytes = 0;
        for (size_t i = 0; i < shards.size(); ++i) {
            SliceSplitShardOutput & output = outputs[s * shards.size() + i];
            if (!options.manifest_only) {
                output.blob->close();
                if (!*output.blob) {
                    throw std::runtime_error("failed to finish " + output.temp_blob.string());
                }
                if (output.blob_offset != output.index["blob_bytes"].get<uint64_t>() ||
                    fs::file_size(output.temp_blob) != output.index["blob_bytes"].get<uint64_t>()) {
                    throw std::runtime_error("internal split sliced shard byte count mismatch");
                }
                fs::rename(output.temp_blob, output.paths.blob);
            }
            write_json(output.paths.index, output.index);

            manifest["shards"].push_back({
                { "blob_file",    output.paths.blob.filename().string() },
                { "index_file",   output.paths.index.filename().string() },
                { "shard_index",  i                                    },
                { "layer_first",  shards[i].layer_first                },
                { "layer_last",   shards[i].layer_last                 },
                { "group_count",  shards[i].group_indices.size()       },
                { "blob_bytes",   output.index["blob_bytes"]             },
                { "content_hash", output.index["content_hash"]          },
            });
            total_bytes += output.index["blob_bytes"].get<uint64_t>();
        }

        manifest["total_blob_bytes"] = total_bytes;
        write_json(manifest_path(slice_bases[s]), manifest);
        std::cout << "split sliced repack complete: slice=" << ranges[s].index << " ff [" << ranges[s].first << ","
                  << ranges[s].last << ") shards=" << shards.size() << " groups=" << selected.size()
                  << " bytes=" << total_bytes << " manifest=" << manifest_path(slice_bases[s]).string() << '\n';
    }
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
    validate_slice_ranges(ranges, geom);

    const fs::path output_base =
        fs::absolute(fs::path(options.output + SLICED_BASE_SUFFIX)).lexically_normal();
    if (!output_base.parent_path().empty()) {
        fs::create_directories(output_base.parent_path());
    }

    const std::vector<size_t> selected = flatten_indices(shards);

    if (options.slice_output_split) {
        repack_sliced_split(options, model, groups, shards, output_base, geom, spec, ranges, sources);
        return;
    }

    ensure_outputs_absent(output_base, shards, !options.manifest_only);

    json                      manifest = {
        { "format",            MANIFEST_FORMAT                                 },
        { "version",           FORMAT_VERSION_SLICED                           },
        { "input_model",       fs::canonical(fs::path(options.model)).string() },
        { "model_files",       model.files                                     },
        { "sharding_mode",     sharding_mode                                   },
        { "total_group_count", selected.size()                                 },
        { "total_blob_bytes",  0                                               },
        { "shard_count",       shards.size()                                   },
        { "expert_slicing",    slicing_json(spec, geom, groups)               },
        { "content_hash",
         {
              { "algorithm", "sha256" },
              { "value", hash_groups_sliced(groups, selected, geom, spec.widths) },
          }                                                                    },
        { "shards",            json::array()                                   },
    };

    std::cout << "expert slicing: n_ff_exp=" << geom.n_ff << " n_embd=" << geom.n_embd
              << " alignment=" << geom.blck << " slices=" << spec.widths.size()
              << " geometry_variants=" << geom.variants.size() << '\n';
    for (size_t v = 0; v < geom.variants.size(); ++v) {
        const SliceGeometryVariant & variant = geom.variants[v];
        std::cout << "  geometry variant " << v << " roles " << ggml_type_name(variant.roles[0].type) << "/"
                  << ggml_type_name(variant.roles[1].type) << "/" << ggml_type_name(variant.roles[2].type) << '\n';
        for (const wp_repack::SliceRange & range : ranges) {
            const uint64_t role_bytes[3] = {
                variant.role_slice_bytes(wp::ROLE_UP, range.width(), geom.n_embd),
                variant.role_slice_bytes(wp::ROLE_GATE, range.width(), geom.n_embd),
                variant.role_slice_bytes(wp::ROLE_DOWN, range.width(), geom.n_embd),
            };
            std::cout << "    slice " << range.index << " ff [" << range.first << "," << range.last << ") width "
                      << range.width() << " bytes " << role_bytes[0] + role_bytes[1] + role_bytes[2]
                      << " per expert (variant " << v << ")\n";
        }
    }

    uint64_t total_bytes = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "writing sliced shard " << i + 1 << "/" << shards.size() << " layers " << shards[i].layer_first
                  << "-" << shards[i].layer_last << " groups " << shards[i].group_indices.size() << " bytes "
                  << shards[i].size << '\n';
        manifest["shards"].push_back(write_shard_sliced(output_base, i, shards.size(), shards[i], groups, model.files,
                                                        geom, spec, ranges, sources, options.manifest_only));
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
    if (!options.manifest_only) {
        sources.reserve(model.files.size());
        for (const std::string & path : model.files) {
            sources.emplace_back(path, std::ios::binary);
            if (!sources.back()) {
                throw std::runtime_error("failed to open source model file: " + path);
            }
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
    ensure_outputs_absent(output_base, shards, !options.manifest_only);

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
            write_shard(output_base, i, shards.size(), shards[i], groups, model.files, sources, options.manifest_only));
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

// A set is SLICED because it carries expert_slicing, not because of its version
// number. Combined v2 sets are version 2; canonical per-slice sets are version 1,
// because that is the version the expert worker's load_catalog requires. Keying
// "is this sliced?" off the version therefore routes a per-slice set to the v1
// verifier, which hashes with hash_groups() instead of hash_groups_sliced() and
// reports a structural hash mismatch on a store that is perfectly correct.
bool is_sliced_metadata(const json & value) {
    return value.contains("expert_slicing");
}

void require_sliced_format(const json & value, const char * format) {
    const int version = value.value("version", 0);
    const bool version_ok = version == FORMAT_VERSION_SLICED ||
                            (version == FORMAT_VERSION && is_sliced_metadata(value));
    if (value.value("format", "") != format || !version_ok) {
        throw std::runtime_error("unsupported or invalid sliced repack metadata format");
    }
}

// Rebuild the SliceSpec a v2 file was written with, and cross-check the geometry
// it claims against the geometry the model actually has. A blob whose sidecar
// describes a different model's shape is exactly the failure --verify exists for.
wp_repack::SliceSpec slicing_from_json(const json &                               value,
                                       const SliceGeometry &                     geom,
                                       const std::vector<wp_repack::ExpertGroup> & groups) {
    const json & block = value.at("expert_slicing");

    if (block.at("n_ff_exp").get<int64_t>() != geom.n_ff || block.at("n_embd").get<int64_t>() != geom.n_embd ||
        block.at("slice_alignment").get<int64_t>() != geom.blck) {
        throw std::runtime_error("recorded slice geometry disagrees with the model's expert tensors");
    }

    const json & variants = block.at("geometry_variants");
    if (!variants.is_array() || variants.size() != geom.variants.size() || geom.group_variants.size() != groups.size()) {
        throw std::runtime_error("recorded slice geometry variants disagree with the model's expert tensors");
    }
    const char *  role_names[3]  = { "up", "gate", "down" };
    const uint8_t role_masks[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };
    for (size_t v = 0; v < geom.variants.size(); ++v) {
        const SliceGeometryVariant & expected_variant = geom.variants[v];
        const json &                 recorded_variant = variants.at(v);
        if (recorded_variant.at("variant_idx").get<size_t>() != v) {
            throw std::runtime_error("recorded slice geometry variant indexes are invalid");
        }

        const json & recorded_groups = recorded_variant.at("groups");
        size_t       expected_group_count = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
            if (geom.group_variants[i] == v) {
                ++expected_group_count;
            }
        }
        if (!recorded_groups.is_array() || recorded_groups.size() != expected_group_count) {
            throw std::runtime_error("recorded slice geometry group assignments are invalid");
        }
        size_t recorded_group = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
            if (geom.group_variants[i] != v) {
                continue;
            }
            const json & group = recorded_groups.at(recorded_group++);
            if (group.at("block_idx").get<int>() != groups[i].block_idx ||
                group.at("expert_idx").get<int>() != groups[i].expert_idx) {
                throw std::runtime_error("recorded slice geometry group assignments disagree with the model");
            }
        }

        const json & role_geometry = recorded_variant.at("role_geometry");
        for (size_t r = 0; r < 3; ++r) {
            const SliceRoleGeometry & role = expected_variant.roles[r];
            const json &              recorded = role_geometry.at(role_names[r]);
            if (recorded.at("ggml_type").get<int>() != static_cast<int>(role.type) ||
                recorded.at("ggml_type_name").get<std::string>() != ggml_type_name(role.type) ||
                recorded.at("quant_block_size").get<int64_t>() != role.blck ||
                recorded.at("slice_block_size").get<int64_t>() != role.slice_blck ||
                recorded.at("quant_block_bytes").get<uint64_t>() != role.type_size ||
                recorded.at("row_bytes").get<uint64_t>() != role.row_bytes ||
                recorded.at("ffn_axis").get<int>() != (role.ffn_ne0 ? 0 : 1)) {
                throw std::runtime_error("recorded slice geometry disagrees with the model's expert tensors");
            }
        }

        const json & recorded_role_bytes = recorded_variant.at("bytes_per_slice_per_role");
        if (!recorded_role_bytes.is_array() || recorded_role_bytes.size() != block.at("widths").size()) {
            throw std::runtime_error("recorded per-variant slice byte counts disagree with the slice widths");
        }
        for (size_t s = 0; s < recorded_role_bytes.size(); ++s) {
            const json & recorded = recorded_role_bytes.at(s);
            if (!recorded.is_array() || recorded.size() != 3) {
                throw std::runtime_error("recorded per-variant slice byte counts are invalid");
            }
            const int64_t width = block.at("widths").at(s).get<int64_t>();
            for (size_t r = 0; r < 3; ++r) {
                if (recorded.at(r).get<uint64_t>() !=
                    expected_variant.role_slice_bytes(role_masks[r], width, geom.n_embd)) {
                    throw std::runtime_error("recorded per-variant slice byte counts disagree with the model's tensors");
                }
            }
        }
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
    validate_slice_ranges(wp_repack::slice_ranges(check.widths), geom);
    return spec;
}

bool split_slice_from_json(const json &                              value,
                           const std::vector<wp_repack::SliceRange> & ranges,
                           wp_repack::SliceRange &                   output_range) {
    // A per-slice set is identified by expert_slicing.selected_slice, which is
    // the canonical marker the expert worker also keys on. It replaced an
    // ad-hoc set of top-level slice_* keys; those are still accepted so a store
    // written before that change still verifies, but new metadata does not
    // carry them and must not be required to.
    int slice_idx = -1;
    if (value.contains("expert_slicing") && value.at("expert_slicing").contains("selected_slice")) {
        slice_idx = value.at("expert_slicing").at("selected_slice").get<int>();
    } else if (value.value("slice_output_split", false)) {
        if (!value.contains("slice_idx") || !value.contains("slice_ff_first") ||
            !value.contains("slice_ff_last") || !value.contains("slice_width")) {
            throw std::runtime_error("split sliced metadata is missing its slice range");
        }
        slice_idx = value.at("slice_idx").get<int>();
    } else {
        return false;
    }
    if (slice_idx < 0 || static_cast<size_t>(slice_idx) >= ranges.size()) {
        throw std::runtime_error("split sliced metadata has an invalid slice index");
    }
    output_range = ranges[slice_idx];
    // Legacy metadata restated the range alongside the index; cross-check it
    // when present. Canonical metadata derives the range from the widths and
    // the selected slice, so there is nothing to disagree with.
    if (value.contains("slice_ff_first")) {
        if (value.at("slice_ff_first").get<int64_t>() != output_range.first ||
            value.at("slice_ff_last").get<int64_t>() != output_range.last ||
            value.at("slice_width").get<int64_t>() != output_range.width()) {
            throw std::runtime_error("split sliced metadata has an invalid slice range");
        }
    }
    return true;
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

    const wp_repack::SliceSpec               spec   = slicing_from_json(index, geom, groups);
    const std::vector<wp_repack::SliceRange> ranges = wp_repack::slice_ranges(spec.widths);
    wp_repack::SliceRange                    output_range;
    const bool                               split = split_slice_from_json(index, ranges, output_range);
    std::vector<wp_repack::SliceRange>       verify_ranges;
    if (split) {
        verify_ranges.push_back(output_range);
    } else {
        verify_ranges = ranges;
    }

    const std::vector<size_t> expected_indices = indices_for_range(groups, layer_first, layer_last);
    const json &              indexed_groups   = index.at("groups");
    if (!indexed_groups.is_array() || indexed_groups.size() != expected_indices.size() ||
        index.at("group_count").get<uint64_t>() != expected_indices.size()) {
        throw std::runtime_error("group count mismatch");
    }

    const bool flat_groups = !indexed_groups.empty() && indexed_groups.front().contains("members") &&
                             !indexed_groups.front().contains("slices");
    const std::string expected_hash = hash_groups_sliced(groups, expected_indices, geom, spec.widths,
                                                         split ? &output_range : nullptr);
    const std::string flat_hash = flat_groups && split ?
        hash_groups_flat_sliced(groups, expected_indices, geom, output_range) : "";
    if (index.at("content_hash").at("algorithm").get<std::string>() != "sha256" ||
        (index.at("content_hash").at("value").get<std::string>() != expected_hash &&
         (flat_hash.empty() || index.at("content_hash").at("value").get<std::string>() != flat_hash))) {
        // Report both values: a bare "mismatch" here cost real time, because the
        // interesting question is always WHICH of the two is wrong.
        throw std::runtime_error(
            "shard structural content hash mismatch: recorded=" +
            index.at("content_hash").at("value").get<std::string>() + " recomputed=" +
            expected_hash +
            " split=" + std::to_string(split ? 1 : 0) +
            " groups=" + std::to_string(expected_indices.size()) +
            " slice=" + std::to_string(output_range.index));
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
    bool              group_shape_set = false;
    bool              flat_group_shape = false;

    for (size_t i = 0; i < expected_indices.size(); ++i) {
        const wp_repack::ExpertGroup & expected = groups[expected_indices[i]];
        const size_t                  group_index = expected_indices[i];
        const SliceGeometryVariant &  variant = geom.variant_for_group(group_index);
        const json &                   actual   = indexed_groups.at(i);
        if (actual.at("block_idx").get<int>() != expected.block_idx ||
            actual.at("expert_idx").get<int>() != expected.expert_idx) {
            throw std::runtime_error("group identity mismatch");
        }

        const bool flat = actual.contains("members");
        const bool nested = actual.contains("slices");
        if (flat == nested) {
            throw std::runtime_error("expert group must use exactly one slice metadata shape");
        }
        if (group_shape_set && flat != flat_group_shape) {
            throw std::runtime_error("expert groups use different slice metadata shapes");
        }
        flat_group_shape = flat;
        group_shape_set = true;
        if (flat) {
            if (!split || verify_ranges.size() != 1) {
                throw std::runtime_error("flat expert group is missing a selected slice");
            }
        } else {
            if (actual.at("geometry_variant").get<size_t>() != geom.group_variants[group_index]) {
                throw std::runtime_error("group identity mismatch");
            }
            const json & slices = actual.at("slices");
            if (!slices.is_array() || slices.size() != verify_ranges.size()) {
                throw std::runtime_error("slice count mismatch for blk " + std::to_string(expected.block_idx) +
                                         " expert " + std::to_string(expected.expert_idx));
            }
        }

        // Regenerate the payload from the source GGUF using the same gather the
        // writer used, then compare it byte for byte with what is on disk. This
        // is what makes the down-column gather trustworthy: a wrong stride shows
        // up here rather than as quiet garbage at inference time.
        build_group_slices(expected, variant, geom.n_embd, verify_ranges, sources, scratch, expected_payload);

        const json * flat_members = nullptr;
        uint64_t group_offset = next_offset;
        if (flat) {
            const json & members = actual.at("members");
            if (!members.is_array() || actual.at("member_count").get<uint64_t>() != 3 || members.size() != 3) {
                throw std::runtime_error("flat expert group must have exactly three role members");
            }
            flat_members = &members;
            group_offset = members.front().at("offset").get<uint64_t>();
            if (group_offset != next_offset) {
                throw std::runtime_error("flat group blob offsets are not contiguous");
            }
        }
        uint64_t       cursor       = group_offset;
        for (size_t s = 0; s < verify_ranges.size(); ++s) {
            const wp_repack::SliceRange & range       = verify_ranges[s];
            const uint64_t                role_bytes[3] = {
                variant.role_slice_bytes(wp::ROLE_UP, range.width(), geom.n_embd),
                variant.role_slice_bytes(wp::ROLE_GATE, range.width(), geom.n_embd),
                variant.role_slice_bytes(wp::ROLE_DOWN, range.width(), geom.n_embd),
            };
            const uint64_t                slice_bytes = role_bytes[0] + role_bytes[1] + role_bytes[2];
            const json * slice_json = nullptr;
            const json * members_json = flat_members;
            if (flat) {
                if (s != 0 || actual.at("slice_idx").get<int>() != range.index ||
                    actual.at("ff_first").get<int64_t>() != range.first ||
                    actual.at("ff_last").get<int64_t>() != range.last ||
                    actual.at("width").get<int64_t>() != range.width()) {
                    throw std::runtime_error("slice descriptor mismatch");
                }
            } else {
                const json & slices = actual.at("slices");
                slice_json = &slices.at(s);
                if (slice_json->at("slice_idx").get<int>() != range.index ||
                    slice_json->at("ff_first").get<int64_t>() != range.first ||
                    slice_json->at("ff_last").get<int64_t>() != range.last ||
                    slice_json->at("width").get<int64_t>() != range.width() ||
                    slice_json->at("bytes").get<uint64_t>() != slice_bytes) {
                    throw std::runtime_error("slice descriptor mismatch");
                }
                if (slice_json->at("offset").get<uint64_t>() != cursor) {
                    throw std::runtime_error("slice blob offsets are not contiguous");
                }
                members_json = &slice_json->at("members");
            }

            const json & members = *members_json;
            if (!members.is_array() || members.size() != 3) {
                throw std::runtime_error("a slice must have exactly three role members");
            }
            const uint8_t order[3] = { wp::ROLE_UP, wp::ROLE_GATE, wp::ROLE_DOWN };
            const char *  names[3] = { "up", "gate", "down" };
            for (size_t r = 0; r < 3; ++r) {
                const wp_repack::ExpertMember & src = member_for_role(expected, order[r], names[r]);
                const json &                    m   = members.at(r);
                const uint64_t member_offset = cursor + (r == 0 ? 0 : role_bytes[0]) +
                                               (r == 2 ? role_bytes[1] : 0);
                if (m.at("role_mask").get<uint8_t>() != src.role_mask || m.at("size").get<uint64_t>() != role_bytes[r] ||
                    m.at("offset").get<uint64_t>() != member_offset ||
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

    const wp_repack::SliceSpec spec = slicing_from_json(manifest, geom, groups);
    const std::vector<wp_repack::SliceRange> ranges = wp_repack::slice_ranges(spec.widths);
    wp_repack::SliceRange output_range;
    const bool split = split_slice_from_json(manifest, ranges, output_range);

    VerifyCounts                  total;
    std::vector<size_t>           all_indices;
    std::set<std::pair<int, int>> seen_groups;
    bool                           have_flat_groups = false;
    bool                           group_shape_set  = false;
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
        const json & index_groups = index.at("groups");
        const bool index_flat_groups = index_groups.is_array() && !index_groups.empty() &&
                                       index_groups.front().contains("members") &&
                                       !index_groups.front().contains("slices");
        if (group_shape_set && index_flat_groups != have_flat_groups) {
            throw std::runtime_error("manifest shards use different expert group shapes");
        }
        have_flat_groups = index_flat_groups;
        group_shape_set = true;
        if (shard.at("shard_index").get<uint64_t>() != shard_pos ||
            index.at("shard_index").get<uint64_t>() != shard_pos ||
            index.at("shard_count").get<uint64_t>() != shards.size() || index.at("layer_first").get<int>() != first ||
            index.at("layer_last").get<int>() != last || index.at("group_count") != shard.at("group_count") ||
            index.at("blob_bytes") != shard.at("blob_bytes") || index.at("blob_file") != shard.at("blob_file") ||
            index.at("content_hash") != shard.at("content_hash")) {
            throw std::runtime_error("manifest and shard index metadata disagree");
        }
        // The index must describe the SAME slice the manifest does. Comparing
        // the legacy slice_output_split flag would silently pass on canonical
        // metadata, which does not carry it -- derive both sides instead.
        wp_repack::SliceRange index_range;
        const bool index_split = split_slice_from_json(index, ranges, index_range);
        if (index_split != split ||
            (split && (index_range.index != output_range.index ||
                       index_range.first != output_range.first ||
                       index_range.last  != output_range.last))) {
            throw std::runtime_error("manifest and shard index describe different slices");
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
    const std::string expected_hash =
        hash_groups_sliced(groups, all_indices, geom, spec.widths, split ? &output_range : nullptr);
    const std::string flat_hash = have_flat_groups && split ?
        hash_groups_flat_sliced(groups, all_indices, geom, output_range) : "";
    if (manifest.at("content_hash").at("algorithm").get<std::string>() != "sha256" ||
        (manifest.at("content_hash").at("value").get<std::string>() != expected_hash &&
         (flat_hash.empty() || manifest.at("content_hash").at("value").get<std::string>() != flat_hash))) {
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

    if (version == FORMAT_VERSION_SLICED || is_sliced_metadata(root)) {
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
