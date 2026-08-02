#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

static constexpr const char * MANIFEST_FORMAT =
    "llama.cpp.weight-pager.expert-shard-manifest";
static constexpr const char * INDEX_FORMAT =
    "llama.cpp.weight-pager.expert-shard-index";
static constexpr const char * DESCRIPTOR_FORMAT =
    "llama.cpp.weight-pager.expert-descriptor";

struct gguf_deleter {
    void operator()(gguf_context * ctx) const {
        gguf_free(ctx);
    }
};

struct ggml_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

struct Options {
    fs::path model;
    fs::path manifest;
    fs::path output;
};

struct RoleDesc {
    std::string    role;
    std::string    source_tensor_name;
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t        ne0 = 0;
    int64_t        ne1 = 0;
    uint64_t       bytes = 0;
};

using LayerRoles = std::map<std::string, RoleDesc>;

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

std::string required_string(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF is missing string metadata " + key);
    }
    return gguf_get_val_str(ctx, id);
}

uint64_t required_uint(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        throw std::runtime_error("GGUF is missing integer metadata " + key);
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT8: {
            const int64_t v = gguf_get_val_i8(ctx, id);
            if (v >= 0) return (uint64_t) v;
        } break;
        case GGUF_TYPE_INT16: {
            const int64_t v = gguf_get_val_i16(ctx, id);
            if (v >= 0) return (uint64_t) v;
        } break;
        case GGUF_TYPE_INT32: {
            const int64_t v = gguf_get_val_i32(ctx, id);
            if (v >= 0) return (uint64_t) v;
        } break;
        case GGUF_TYPE_INT64: {
            const int64_t v = gguf_get_val_i64(ctx, id);
            if (v >= 0) return (uint64_t) v;
        } break;
        default:
            break;
    }
    throw std::runtime_error("GGUF metadata " + key + " is not a non-negative integer");
}

std::pair<gguf_ptr, ggml_ptr> load_gguf(const fs::path & path) {
    ggml_context * raw_tensors = nullptr;
    const gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &raw_tensors,
    };
    gguf_ptr gguf(gguf_init_from_file(path.string().c_str(), params));
    ggml_ptr tensors(raw_tensors);
    if (!gguf || !tensors) {
        throw std::runtime_error("failed to read GGUF metadata: " + path.string());
    }
    return { std::move(gguf), std::move(tensors) };
}

bool parse_expert_tensor_name(
        const std::string & name, int & layer, std::string & role) {
    static const std::string prefix = "blk.";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    const size_t dot = name.find('.', prefix.size());
    if (dot == std::string::npos || dot == prefix.size()) {
        return false;
    }
    const std::string layer_text = name.substr(prefix.size(), dot - prefix.size());
    if (!std::all_of(layer_text.begin(), layer_text.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return false;
    }
    const long parsed = std::strtol(layer_text.c_str(), nullptr, 10);
    if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }

    const std::string suffix = name.substr(dot + 1);
    if (suffix == "ffn_gate_exps.weight") {
        role = "gate";
    } else if (suffix == "ffn_up_exps.weight") {
        role = "up";
    } else if (suffix == "ffn_down_exps.weight") {
        role = "down";
    } else {
        return false;
    }
    layer = (int) parsed;
    return true;
}

std::string role_from_mask(uint64_t mask) {
    switch (mask) {
        case 1: return "up";
        case 2: return "gate";
        case 4: return "down";
        default: throw std::runtime_error("sidecar member has unknown expert role mask " + std::to_string(mask));
    }
}

json role_to_json(const RoleDesc & role) {
    return {
        { "ggml_type",         (int) role.type       },
        { "ggml_type_name",    ggml_type_name(role.type) },
        { "shape",             { role.ne0, role.ne1 } },
        { "bytes_per_expert",  role.bytes           },
        { "source_tensor_name", role.source_tensor_name },
    };
}

void validate_role_shape(
        const RoleDesc & role, int64_t n_embd, int64_t n_ff_exp) {
    const int64_t want0 = role.role == "down" ? n_ff_exp : n_embd;
    const int64_t want1 = role.role == "down" ? n_embd : n_ff_exp;
    if (role.ne0 != want0 || role.ne1 != want1) {
        throw std::runtime_error(
            role.source_tensor_name + ": per-expert shape [" +
            std::to_string(role.ne0) + ", " + std::to_string(role.ne1) +
            "] does not match expected [" + std::to_string(want0) + ", " +
            std::to_string(want1) + "]");
    }
}

void print_usage(const char * argv0) {
    std::cout
        << "usage: " << argv0
        << " --model FIRST.gguf --shard-manifest MANIFEST [--output DESCRIPTOR]\n\n"
        << "The output defaults to MANIFEST.expert-descriptor.json in the manifest directory.\n";
}

Options parse_cli(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take = [&](fs::path & dst) {
            if (++i >= argc) {
                throw std::invalid_argument(arg + " requires a path");
            }
            dst = argv[i];
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model" || arg == "-m") {
            take(options.model);
        } else if (arg == "--shard-manifest") {
            take(options.manifest);
        } else if (arg == "--output") {
            take(options.output);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.model.empty() || options.manifest.empty()) {
        throw std::invalid_argument("--model and --shard-manifest are required");
    }
    options.model    = fs::canonical(options.model);
    options.manifest = fs::canonical(options.manifest);
    if (options.output.empty()) {
        options.output = options.manifest.parent_path() /
            (options.manifest.stem().string() + ".expert-descriptor.json");
    } else {
        options.output = fs::absolute(options.output).lexically_normal();
    }
    if (fs::exists(options.output)) {
        throw std::runtime_error("refusing to overwrite existing output: " + options.output.string());
    }
    return options;
}

int run(const Options & options) {
    const json manifest = read_json(options.manifest);
    check_format(manifest, MANIFEST_FORMAT, options.manifest);
    if (get_value<std::string>(manifest, "sharding_mode", options.manifest) !=
        "expert-index-range") {
        throw std::runtime_error("descriptor requires an expert-index-range shard manifest");
    }
    const json & retained = manifest.at("retained_expert_range");
    const int expert_first = get_value<int>(retained, "first", options.manifest);
    const int expert_last  = get_value<int>(retained, "last", options.manifest);
    if (expert_first < 0 || expert_last < expert_first) {
        throw std::runtime_error("manifest has an invalid retained expert range");
    }

    const json & model_files_json = get_array(manifest, "model_files", options.manifest);
    if (model_files_json.empty()) {
        throw std::runtime_error("manifest has no model files");
    }
    std::vector<fs::path> model_files;
    for (const json & item : model_files_json) {
        model_files.push_back(fs::canonical(item.get<std::string>()));
    }
    if (model_files.front() != options.model) {
        throw std::runtime_error("--model is not the manifest's first model file");
    }

    auto first_loaded = load_gguf(model_files.front());
    const std::string architecture = required_string(first_loaded.first.get(), "general.architecture");
    const std::string model_name   = required_string(first_loaded.first.get(), "general.name");
    const uint64_t n_layer = required_uint(first_loaded.first.get(), architecture + ".block_count");
    const uint64_t n_embd = required_uint(first_loaded.first.get(), architecture + ".embedding_length");
    const uint64_t n_ff_exp =
        required_uint(first_loaded.first.get(), architecture + ".expert_feed_forward_length");
    const uint64_t n_expert = required_uint(first_loaded.first.get(), architecture + ".expert_count");
    const uint64_t n_expert_used =
        required_uint(first_loaded.first.get(), architecture + ".expert_used_count");
    if (n_layer > INT32_MAX || n_embd > INT32_MAX || n_ff_exp > INT32_MAX ||
        n_expert > INT32_MAX || n_expert_used > INT32_MAX) {
        throw std::runtime_error("model hparams exceed descriptor integer range");
    }
    if (expert_last >= (int) n_expert) {
        throw std::runtime_error("manifest retained range exceeds model expert count");
    }

    std::string activation;
    const int64_t activation_id =
        gguf_find_key(first_loaded.first.get(), (architecture + ".hidden_activation").c_str());
    if (activation_id >= 0) {
        activation = required_string(first_loaded.first.get(), architecture + ".hidden_activation");
    } else if (architecture == "glm-dsa" || architecture == "deepseek4") {
        // NOTE 2026-07-31: this was a one-entry allowlist ("glm-dsa"), so every
        // other SwiGLU model failed here with a message implying the MODEL was
        // deficient rather than this list. deepseek4 added on evidence, not
        // assumption: the GGUF carries deepseek4.swiglu_clamp_exp and
        // .swiglu_clamp_shexp, which only a SwiGLU FFN emits.
        // TODO: derive this from the presence of a swiglu_clamp key (or an
        // explicit activation KV) instead of naming architectures.
        activation = "silu";
    } else {
        throw std::runtime_error(
            "GGUF does not declare hidden_activation and architecture is not a known SwiGLU model");
    }
    if (activation != "silu") {
        throw std::runtime_error("expert worker only supports silu/SwiGLU, model declares " + activation);
    }

    std::map<int, LayerRoles> layers;
    std::set<std::string> tensor_names;
    for (size_t file_index = 0; file_index < model_files.size(); ++file_index) {
        std::pair<gguf_ptr, ggml_ptr> loaded =
            file_index == 0 ? std::move(first_loaded) : load_gguf(model_files[file_index]);
        const int64_t n_tensors = gguf_get_n_tensors(loaded.first.get());
        for (int64_t tensor_index = 0; tensor_index < n_tensors; ++tensor_index) {
            const char * raw_name = gguf_get_tensor_name(loaded.first.get(), tensor_index);
            if (raw_name == nullptr) {
                throw std::runtime_error("GGUF tensor has no name");
            }
            int         layer = -1;
            std::string role_name;
            if (!parse_expert_tensor_name(raw_name, layer, role_name)) {
                continue;
            }
            if (!tensor_names.insert(raw_name).second) {
                throw std::runtime_error("duplicate expert tensor across GGUF shards: " + std::string(raw_name));
            }
            const ggml_tensor * tensor = ggml_get_tensor(loaded.second.get(), raw_name);
            if (tensor == nullptr) {
                throw std::runtime_error("missing GGML tensor metadata for " + std::string(raw_name));
            }
            if (tensor->ne[2] != (int64_t) n_expert || tensor->ne[3] != 1) {
                throw std::runtime_error(
                    std::string(raw_name) + ": expert dimension does not match model expert count");
            }
            RoleDesc role;
            role.role               = role_name;
            role.source_tensor_name = raw_name;
            role.type               = tensor->type;
            role.ne0                = tensor->ne[0];
            role.ne1                = tensor->ne[1];
            role.bytes              = ggml_row_size(role.type, role.ne0) * (uint64_t) role.ne1;
            if (ggml_nbytes(tensor) % n_expert != 0 ||
                ggml_nbytes(tensor) / n_expert != role.bytes) {
                throw std::runtime_error(
                    std::string(raw_name) + ": tensor bytes do not divide into uniform experts");
            }
            validate_role_shape(role, n_embd, n_ff_exp);
            if (!layers[layer].emplace(role_name, std::move(role)).second) {
                throw std::runtime_error("duplicate role for expert layer " + std::to_string(layer));
            }
        }
    }

    const json & shards = get_array(manifest, "shards", options.manifest);
    const uint64_t shard_count = get_value<uint64_t>(manifest, "shard_count", options.manifest);
    if (shards.size() != shard_count) {
        throw std::runtime_error("manifest shard count does not match shards array");
    }

    std::set<int> served_layers;
    std::set<int> seen_shard_indices;
    uint64_t checked_groups = 0;
    uint64_t checked_members = 0;
    uint64_t checked_bytes = 0;
    for (const json & shard : shards) {
        const fs::path index_path = options.manifest.parent_path() /
            get_value<std::string>(shard, "index_file", options.manifest);
        const json index = read_json(index_path);
        check_format(index, INDEX_FORMAT, index_path);
        const int shard_index = get_value<int>(index, "shard_index", index_path);
        const int layer_first = get_value<int>(index, "layer_first", index_path);
        const int layer_last  = get_value<int>(index, "layer_last", index_path);
        if (shard_index < 0 || (uint64_t) shard_index >= shard_count ||
            !seen_shard_indices.insert(shard_index).second ||
            get_value<uint64_t>(index, "shard_count", index_path) != shard_count ||
            get_value<int>(shard, "shard_index", options.manifest) != shard_index ||
            get_value<int>(shard, "layer_first", options.manifest) != layer_first ||
            get_value<int>(shard, "layer_last", options.manifest) != layer_last ||
            get_value<uint64_t>(shard, "group_count", options.manifest) !=
                get_value<uint64_t>(index, "group_count", index_path) ||
            get_value<uint64_t>(shard, "blob_bytes", options.manifest) !=
                get_value<uint64_t>(index, "blob_bytes", index_path) ||
            get_value<std::string>(shard, "blob_file", options.manifest) !=
                get_value<std::string>(index, "blob_file", index_path) ||
            get_array(index, "model_files", index_path) != model_files_json ||
            layer_first != layer_last || !served_layers.insert(layer_first).second) {
            throw std::runtime_error(index_path.string() + ": expected one unique layer");
        }
        auto layer_it = layers.find(layer_first);
        if (layer_it == layers.end() || layer_it->second.size() != 3) {
            throw std::runtime_error(
                "GGUF does not contain all expert roles for shard layer " + std::to_string(layer_first));
        }

        const json & groups = get_array(index, "groups", index_path);
        if (groups.size() != get_value<uint64_t>(index, "group_count", index_path) ||
            groups.size() != (uint64_t) (expert_last - expert_first + 1)) {
            throw std::runtime_error(index_path.string() + ": group count does not match retained range");
        }
        uint64_t next_offset = 0;
        int expected_expert = expert_first;
        for (const json & group : groups) {
            const int layer = get_value<int>(group, "block_idx", index_path);
            const int expert = get_value<int>(group, "expert_idx", index_path);
            if (layer != layer_first || expert != expected_expert++) {
                throw std::runtime_error(index_path.string() + ": expert groups are not dense and ordered");
            }
            const json & members = get_array(group, "members", index_path);
            if (members.size() != 3 ||
                get_value<uint64_t>(group, "member_count", index_path) != 3) {
                throw std::runtime_error(index_path.string() + ": expert group does not have three members");
            }
            std::set<std::string> roles_seen;
            for (const json & member : members) {
                const std::string role_name =
                    role_from_mask(get_value<uint64_t>(member, "role_mask", index_path));
                if (!roles_seen.insert(role_name).second) {
                    throw std::runtime_error(index_path.string() + ": expert group repeats role " + role_name);
                }
                const RoleDesc & role = layer_it->second.at(role_name);
                const uint64_t offset = get_value<uint64_t>(member, "offset", index_path);
                const uint64_t size   = get_value<uint64_t>(member, "size", index_path);
                const std::string source_name =
                    get_value<std::string>(member, "source_tensor_name", index_path);
                if (offset != next_offset || size != role.bytes ||
                    source_name != role.source_tensor_name) {
                    throw std::runtime_error(
                        index_path.string() + ": layer " + std::to_string(layer_first) +
                        " expert " + std::to_string(expert) + " role " + role_name +
                        " disagrees with GGUF shape/type bytes");
                }
                next_offset += size;
                checked_bytes += size;
                ++checked_members;
            }
            ++checked_groups;
        }
        if (next_offset != get_value<uint64_t>(index, "blob_bytes", index_path)) {
            throw std::runtime_error(index_path.string() + ": member spans do not cover blob_bytes");
        }
    }

    if (served_layers.empty()) {
        throw std::runtime_error("manifest serves no layers");
    }
    if (checked_groups != get_value<uint64_t>(manifest, "total_group_count", options.manifest) ||
        checked_bytes != get_value<uint64_t>(manifest, "total_blob_bytes", options.manifest)) {
        throw std::runtime_error("descriptor cross-check totals disagree with manifest");
    }

    const json & content_hash = manifest.at("content_hash");
    const std::string hash_algorithm =
        get_value<std::string>(content_hash, "algorithm", options.manifest);
    const std::string hash_value =
        get_value<std::string>(content_hash, "value", options.manifest);
    if (hash_algorithm.empty() || hash_value.empty()) {
        throw std::runtime_error("manifest content hash is empty");
    }

    json descriptor = {
        { "format",  DESCRIPTOR_FORMAT },
        { "version", 1 },
        { "source_model",
          {
              { "input_model",  get_value<std::string>(manifest, "input_model", options.manifest) },
              { "model_files",  model_files_json },
              { "architecture", architecture },
              { "name",         model_name },
          } },
        { "shard_manifest_identity",
          {
              { "algorithm", hash_algorithm },
              { "value",     hash_value },
          } },
        { "retained_expert_range",
          {
              { "first", expert_first },
              { "last",  expert_last },
          } },
        { "hparams",
          {
              { "n_layer",       n_layer },
              { "n_embd",        n_embd },
              { "n_ff_exp",      n_ff_exp },
              { "n_expert",      n_expert },
              { "n_expert_used", n_expert_used },
              { "activation",    activation },
          } },
        { "layers", json::array() },
    };

    std::map<std::string, std::map<std::pair<int, std::string>, int>> distribution;
    for (int layer : served_layers) {
        const LayerRoles & roles = layers.at(layer);
        json layer_json = {
            { "layer", layer },
            { "roles",
              {
                  { "gate", role_to_json(roles.at("gate")) },
                  { "up",   role_to_json(roles.at("up"))   },
                  { "down", role_to_json(roles.at("down")) },
              } },
        };
        descriptor["layers"].push_back(std::move(layer_json));
        for (const auto & item : roles) {
            distribution[item.first][{ (int) item.second.type, ggml_type_name(item.second.type) }]++;
        }
    }

    std::ofstream output(options.output);
    if (!output) {
        throw std::runtime_error("failed to create " + options.output.string());
    }
    output << descriptor.dump(2) << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error("failed to write " + options.output.string());
    }

    std::cout << "descriptor complete: layers=" << served_layers.size()
              << " groups=" << checked_groups
              << " members=" << checked_members
              << " bytes=" << checked_bytes
              << " output=" << options.output.string() << '\n';
    for (const std::string role_name : { "gate", "up", "down" }) {
        std::cout << role_name << " type distribution:";
        for (const auto & item : distribution.at(role_name)) {
            std::cout << " " << item.first.second << "(" << item.first.first << ")=" << item.second;
        }
        std::cout << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_cli(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
