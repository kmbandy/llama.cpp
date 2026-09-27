// wp-expert-descriptor: width-sliced + layer-partial (spine-only) manifest.
//
// The fixture is a spine GGUF (no routed-expert tensors) plus a manifest that
// is BOTH width-sliced (expert_slicing) AND layer-partial (layer_ranges +
// expert_ggml_type), whose model_files list only the spine. The per-layer
// index files describe groups of already-sliced members, so the descriptor's
// role geometry must come from the spine hparams + expert_ggml_type, not from
// a GGUF expert-tensor walk.
//
// Negative case: the same manifest without expert_ggml_type must throw,
// because then the GGUF walk runs against a spine that holds no expert roles.
//
// No model, no GPU. The worker's descriptor loader is NOT tested here: its
// Descriptor struct lives in an anonymous namespace in
// tools/wp-expert-worker/wp-expert-worker.cpp and is not linkable.

#include "wp-expert-descriptor.h"

#include "ggml.h"
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using json = nlohmann::ordered_json;

static int g_fail = 0;

static void check(bool ok, const std::string & what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        ++g_fail;
    }
}

static void write_json(const std::filesystem::path & path, const json & value) {
    std::ofstream output(path);
    output << value.dump(2) << '\n';
}

struct Fixture {
    std::filesystem::path dir;
    std::filesystem::path spine;
    std::filesystem::path manifest;
    std::filesystem::path output;

    Fixture() {
        dir = std::filesystem::temp_directory_path() / "wp-expert-descriptor-test";
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
        std::filesystem::create_directories(dir);
        spine    = dir / "spine.gguf";
        manifest = dir / "slice-layered-manifest.json";
        output   = dir / "slice-layered-manifest.expert-descriptor.json";
        write_spine();
    }

    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
    }

    // Spine GGUF: architecture + hparams only, one small f32 tensor. No
    // blk.*.ffn_*_exps.weight tensors, so the descriptor's GGUF walk finds no
    // expert roles in it.
    void write_spine() {
        // one small allocated tensor: gguf_write_to_file copies tensor->data
        const struct ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() + 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ggml_ctx = ggml_init(params);
        struct ggml_tensor *  dummy    = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, 4);
        ggml_set_name(dummy, "tok_embd.weight");

        struct gguf_context * gguf_ctx = gguf_init_empty();
        gguf_set_val_str(gguf_ctx, "general.architecture", "deepseek41");
        gguf_set_val_str(gguf_ctx, "general.name", "wp-expert-descriptor-test");
        gguf_set_val_u32 (gguf_ctx, "deepseek41.block_count", 2);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.embedding_length", 32);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.expert_feed_forward_length", 64);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.expert_count", 4);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.expert_used_count", 2);
        gguf_add_tensor(gguf_ctx, dummy);
        bool ok = gguf_write_to_file(gguf_ctx, spine.string().c_str(), false);
        if (!ok) {
            std::fprintf(stderr, "FAIL writing spine gguf\n");
            std::exit(1);
        }
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
    }
};

// Sliced member sizes at the selected slice (widths [32,32], slice 1):
// gate/up: ne0 = n_embd = 32, ne1 = slice_width = 32
// down:    ne0 = slice_width = 32, ne1 = n_embd = 32
// All q8_0, so each role member is row_size * ne1 bytes.
static const uint64_t ROW  = ggml_row_size(GGML_TYPE_Q8_0, 32);
static const uint64_t SIZE = ROW * 32; // gate, up and down are square here

static json make_index(const std::filesystem::path & dir, const int layer, const uint64_t group_count) {
    json groups = json::array();
    uint64_t offset = 0;
    for (int expert = 0; expert < (int) group_count; ++expert) {
        json members = json::array();
        const std::vector<std::pair<std::string, uint64_t>> roles = {
            { "up", 1 }, { "gate", 2 }, { "down", 4 },
        };
        for (const auto & role : roles) {
            members.push_back({
                { "role_mask",          role.second },
                { "offset",             offset },
                { "size",               SIZE },
                { "slice_shape",        { 32, 32 } },
                { "source_tensor_name", "blk." + std::to_string(layer) +
                    ".ffn_" + role.first + "_exps.weight" },
            });
            offset += SIZE;
        }
        groups.push_back({
            { "block_idx",    layer },
            { "expert_idx",   expert },
            { "slice_idx",    1 },
            { "ff_first",     32 },
            { "ff_last",      64 },
            { "member_count", 3 },
            { "members",      members },
        });
    }
    return {
        { "format",      "llama.cpp.weight-pager.expert-shard-index" },
        { "version",     1 },
        { "shard_index", layer },
        { "shard_count", 2 },
        { "layer_first", layer },
        { "layer_last",  layer },
        { "group_count", (uint64_t) group_count },
        { "blob_bytes",  offset },
        { "blob_file",   "blob-" + std::to_string(layer) },
        { "model_files", { (dir / "spine.gguf").string() } },  // must equal the manifest's list
        { "groups",      groups },
    };
}

static json make_manifest(const std::filesystem::path & dir, const bool with_expert_type) {
    const json slice = {
        { "widths",          { 32, 32 } },
        { "slice_count",     2 },
        { "n_ff_exp",        64 },
        { "n_embd",          32 },
        { "selected_slice",  1 },
        { "slice_alignment", 32 },
    };
    json manifest = {
        { "format",       "llama.cpp.weight-pager.expert-shard-manifest" },
        { "version",      1 },
        { "input_model",  "wp-expert-descriptor-test" },
        { "sharding_mode", "expert-slice" },
        { "layer_ranges",  { "0-1" } },
        { "allow_partial", true },
        { "model_files",   { (dir / "spine.gguf").string() } },
        { "retained_expert_range", { { "first", 0 }, { "last", 3 } } },
        { "shard_count",   2 },
        { "expert_slicing", slice },
        { "shards", json::array({
            {
                { "shard_index", 0 }, { "layer_first", 0 }, { "layer_last", 0 },
                { "group_count", (uint64_t) 4 },
                { "blob_bytes",  4 * 3 * SIZE },
                { "blob_file",   "blob-0" },
                { "index_file",  "index-0.json" },
            },
            {
                { "shard_index", 1 }, { "layer_first", 1 }, { "layer_last", 1 },
                { "group_count", (uint64_t) 4 },
                { "blob_bytes",  4 * 3 * SIZE },
                { "blob_file",   "blob-1" },
                { "index_file",  "index-1.json" },
            },
        }) },
        { "total_group_count", (uint64_t) 8 },
        { "total_blob_bytes",  2 * 4 * 3 * SIZE },
        { "content_hash",  { { "algorithm", "sha256" }, { "value", "test-value" } } },
    };
    if (with_expert_type) {
        manifest["expert_ggml_type"] = "q8_0";
    }
    return manifest;
}

static int run_descriptor(const Fixture & fixture, const json & manifest) {
    Options options;
    options.model    = std::filesystem::canonical(fixture.spine);
    options.manifest = std::filesystem::canonical(fixture.manifest);
    options.output   = fixture.output;  // does not exist yet; run() refuses to overwrite
    return run(options);
}

static void test_slice_and_layered(const Fixture & fixture) {
    write_json(fixture.dir / "index-0.json", make_index(fixture.dir, 0, 4));
    write_json(fixture.dir / "index-1.json", make_index(fixture.dir, 1, 4));
    write_json(fixture.manifest, make_manifest(fixture.dir, true));

    int rc = 0;
    std::string error;
    try {
        rc = run_descriptor(fixture, make_manifest(fixture.dir, true));
    } catch (const std::exception & e) {
        error = e.what();
    }
    check(rc == 0 && error.empty(), "descriptor run() succeeds on sliced + layer-partial manifest");
    if (!error.empty()) {
        std::printf("    exception: %s\n", error.c_str());
    }

    if (!error.empty()) {
        return;  // no output to inspect; the FAIL above is the verdict
    }
    std::ifstream input(fixture.output);
    json descriptor;
    input >> descriptor;

    check(descriptor["sharding_mode"] == "expert-slice", "descriptor sharding_mode is expert-slice");
    check(descriptor["layer_ranges"] == json({ "0-1" }), "descriptor copies manifest layer_ranges");
    check(descriptor["expert_slicing"]["selected_slice"] == 1,
          "descriptor keeps expert_slicing.selected_slice");

    for (int layer = 0; layer < 2; ++layer) {
        const json & found = descriptor["layers"][layer];
        check(found["layer"] == layer, "layer " + std::to_string(layer) + " present in descriptor");
        const json & roles = found["roles"];
        check(roles["gate"]["shape"] == json({ 32, 32 }) &&
              roles["gate"]["bytes_per_expert"] == (uint64_t) SIZE,
              "layer " + std::to_string(layer) + " gate has sliced shape/bytes");
        check(roles["up"]["shape"] == json({ 32, 32 }) &&
              roles["up"]["bytes_per_expert"] == (uint64_t) SIZE,
              "layer " + std::to_string(layer) + " up has sliced shape/bytes");
        check(roles["down"]["shape"] == json({ 32, 32 }) &&
              roles["down"]["bytes_per_expert"] == (uint64_t) SIZE,
              "layer " + std::to_string(layer) + " down has sliced shape/bytes");
    }
}

static void test_negative_without_expert_type(const Fixture & fixture) {
    std::error_code ignored;
    std::filesystem::remove(fixture.output, ignored);
    write_json(fixture.manifest, make_manifest(fixture.dir, false));

    bool threw = false;
    try {
        run_descriptor(fixture, make_manifest(fixture.dir, false));
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "manifest without expert_ggml_type throws (spine has no expert roles)");
}

// ---------------------------------------------------------------------------
// ML8_4 LUT bytes + ml8_rotation: a single unsliced layer-ranges layer,
// n_embd = n_ff_exp = 64 (one QK_ML8 block), expert_ggml_type = ml8_4.
// ---------------------------------------------------------------------------

static const uint64_t ML8_ROW  = ggml_row_size(GGML_TYPE_ML8_4, 64); // 36 bytes/row-block * 1 block
static const uint64_t ML8_SIZE = ML8_ROW * 64;                       // weight-only bytes/expert (square 64x64)
static const uint64_t ML8_LUT  = 16 * (64 / 64);                     // 16 bytes/expert

struct Ml8Fixture {
    std::filesystem::path dir;
    std::filesystem::path spine;
    std::filesystem::path manifest;
    std::filesystem::path index;
    std::filesystem::path output;

    Ml8Fixture() {
        dir = std::filesystem::temp_directory_path() / "wp-expert-descriptor-ml8-test";
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
        std::filesystem::create_directories(dir);
        spine    = dir / "spine.gguf";
        manifest = dir / "manifest.json";
        index    = dir / "index-0.json";
        output   = dir / "manifest.expert-descriptor.json";
        write_spine();
    }

    ~Ml8Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(dir, ignored);
    }

    void write_spine() {
        const struct ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() + 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * ggml_ctx = ggml_init(params);
        struct ggml_tensor *  dummy    = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, 4);
        ggml_set_name(dummy, "tok_embd.weight");

        struct gguf_context * gguf_ctx = gguf_init_empty();
        gguf_set_val_str(gguf_ctx, "general.architecture", "deepseek41");
        gguf_set_val_str(gguf_ctx, "general.name", "wp-expert-descriptor-ml8-test");
        gguf_set_val_u32 (gguf_ctx, "deepseek41.block_count", 1);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.embedding_length", 64);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.expert_feed_forward_length", 64);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.expert_count", 2);
        gguf_set_val_u32 (gguf_ctx, "deepseek41.expert_used_count", 1);
        gguf_add_tensor(gguf_ctx, dummy);
        bool ok = gguf_write_to_file(gguf_ctx, spine.string().c_str(), false);
        if (!ok) {
            std::fprintf(stderr, "FAIL writing ml8 spine gguf\n");
            std::exit(1);
        }
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
    }
};

// kind: GGML_ML8_ROTATION_KIND_KRONECKER_ORTH_SYLVESTER (1) with a=8,b=8,k=64
// carries h_a (64 floats); block_hadamard (2), a=1,b=64,k=64, no h_a.
static json make_rotation_entry(bool kronecker) {
    if (kronecker) {
        json h_a = json::array();
        for (int i = 0; i < 64; ++i) {
            h_a.push_back(0.0f);
        }
        return { { "kind", 1 }, { "a", 8 }, { "b", 8 }, { "k", 64 }, { "h_a", h_a } };
    }
    return { { "kind", 2 }, { "a", 1 }, { "b", 64 }, { "k", 64 } };
}

// mutate_member: index into (expert, role-order-within-group) so tests can
// perturb a single member's lut_bytes/size in an otherwise-valid index.
static json make_ml8_index(uint64_t lut_bytes, int mutate_expert, const std::string & mutate_role,
                            uint64_t mutate_lut_bytes) {
    json groups = json::array();
    uint64_t offset = 0;
    for (int expert = 0; expert < 2; ++expert) {
        json members = json::array();
        const std::vector<std::pair<std::string, uint64_t>> roles = {
            { "up", 1 }, { "gate", 2 }, { "down", 4 },
        };
        for (const auto & role : roles) {
            uint64_t lut = lut_bytes;
            if (expert == mutate_expert && role.first == mutate_role) {
                lut = mutate_lut_bytes;
            }
            json member = {
                { "role_mask",          role.second },
                { "offset",             offset },
                { "size",               ML8_SIZE + lut },
                { "source_tensor_name", "blk.0.ffn_" + role.first + "_exps.weight" },
            };
            if (lut != 0) {
                member["lut_bytes"] = lut;
            }
            members.push_back(member);
            offset += ML8_SIZE + lut;
        }
        groups.push_back({
            { "block_idx",    0 },
            { "expert_idx",   expert },
            { "member_count", 3 },
            { "members",      members },
        });
    }
    json index = {
        { "format",      "llama.cpp.weight-pager.expert-shard-index" },
        { "version",     1 },
        { "shard_index", 0 },
        { "shard_count", 1 },
        { "layer_first", 0 },
        { "layer_last",  0 },
        { "group_count", (uint64_t) 2 },
        { "blob_bytes",  offset },
        { "blob_file",   "blob-0" },
        { "model_files", json::array() },  // filled in by caller
        { "groups",      groups },
    };
    return index;
}

static json make_ml8_manifest(const std::filesystem::path & dir, uint64_t blob_bytes) {
    return {
        { "format",       "llama.cpp.weight-pager.expert-shard-manifest" },
        { "version",      1 },
        { "input_model",  "wp-expert-descriptor-ml8-test" },
        { "sharding_mode", "layer-ranges" },
        { "layer_ranges",  { "0-0" } },
        { "allow_partial", true },
        { "model_files",   { (dir / "spine.gguf").string() } },
        { "retained_expert_range", { { "first", 0 }, { "last", 1 } } },
        { "shard_count",   1 },
        { "expert_ggml_type", "ml8_4" },
        { "shards", json::array({
            {
                { "shard_index", 0 }, { "layer_first", 0 }, { "layer_last", 0 },
                { "group_count", (uint64_t) 2 },
                { "blob_bytes",  blob_bytes },
                { "blob_file",   "blob-0" },
                { "index_file",  "index-0.json" },
            },
        }) },
        { "total_group_count", (uint64_t) 2 },
        { "total_blob_bytes",  blob_bytes },
        { "content_hash",  { { "algorithm", "sha256" }, { "value", "test-value" } } },
    };
}

static int run_ml8_descriptor(const Ml8Fixture & fixture) {
    Options options;
    options.model    = std::filesystem::canonical(fixture.spine);
    options.manifest = std::filesystem::canonical(fixture.manifest);
    options.output   = fixture.output;
    return run(options);
}

static void write_ml8_index_and_manifest(const Ml8Fixture & fixture, json index) {
    index["model_files"] = json::array({ fixture.spine.string() });
    write_json(fixture.index, index);
    const uint64_t blob_bytes = index.at("blob_bytes").get<uint64_t>();
    write_json(fixture.manifest, make_ml8_manifest(fixture.dir, blob_bytes));
}

static void test_ml8_lut_and_rotation() {
    Ml8Fixture fixture;
    json index = make_ml8_index(ML8_LUT, /*mutate_expert=*/-1, "", 0);
    json rot_layer = json::object();
    rot_layer["gate"] = make_rotation_entry(/*kronecker=*/true);
    rot_layer["up"]   = make_rotation_entry(/*kronecker=*/true);
    rot_layer["down"] = make_rotation_entry(/*kronecker=*/false);
    index["ml8_rotation"] = { { "0", rot_layer } };
    write_ml8_index_and_manifest(fixture, index);

    std::string error;
    int rc = 0;
    try {
        rc = run_ml8_descriptor(fixture);
    } catch (const std::exception & e) {
        error = e.what();
    }
    check(rc == 0 && error.empty(), "ml8 descriptor run() succeeds with lut_bytes + rotation");
    if (!error.empty()) {
        std::printf("    exception: %s\n", error.c_str());
        return;
    }

    std::ifstream input(fixture.output);
    json descriptor;
    input >> descriptor;
    const json & roles = descriptor["layers"][0]["roles"];
    check(roles["gate"]["lut_bytes_per_expert"] == ML8_LUT, "gate lut_bytes_per_expert == 16*K/64");
    check(roles["up"]["lut_bytes_per_expert"]   == ML8_LUT, "up lut_bytes_per_expert == 16*K/64");
    check(roles["down"]["lut_bytes_per_expert"] == ML8_LUT, "down lut_bytes_per_expert == 16*K/64");
    check(roles["gate"].contains("rotation") && roles["gate"]["rotation"]["kind"] == 1 &&
              roles["gate"]["rotation"]["a"] == 8 && roles["gate"]["rotation"]["b"] == 8 &&
              roles["gate"]["rotation"]["k"] == 64 && roles["gate"]["rotation"]["h_a"].size() == 64,
          "gate rotation carries kronecker kind/a/b/k/h_a");
    check(roles["up"]["rotation"] == roles["gate"]["rotation"], "up rotation identical to gate");
    check(roles["down"].contains("rotation") && roles["down"]["rotation"]["kind"] == 2 &&
              !roles["down"]["rotation"].contains("h_a"),
          "down rotation is block_hadamard with no h_a");
}

static void test_ml8_lut_mismatch_across_experts() {
    Ml8Fixture fixture;
    json index = make_ml8_index(ML8_LUT, /*mutate_expert=*/1, "gate", ML8_LUT * 2);
    write_ml8_index_and_manifest(fixture, index);

    bool threw = false;
    try {
        run_ml8_descriptor(fixture);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "lut_bytes differing across experts of the same role throws");
}

static void test_ml8_missing_lut_throws() {
    Ml8Fixture fixture;
    // Every ML8_4 gate member gets lut_bytes = 0 (i.e. omitted): violates the
    // ML8_4 => lut_bytes == 16*K/64 contract.
    json index = make_ml8_index(ML8_LUT, /*mutate_expert=*/0, "gate", 0);
    // mutate only touches expert 0's gate; force expert 1's gate to match so
    // the failure is specifically the "ML8_4 role requires a LUT" check, not
    // the cross-expert consistency check.
    for (auto & group : index["groups"]) {
        if (group["expert_idx"] == 1) {
            for (auto & member : group["members"]) {
                if (member["role_mask"] == 2) { // gate
                    member.erase("lut_bytes");
                    member["size"] = ML8_SIZE;
                }
            }
        }
    }
    // recompute offsets/blob_bytes after removing expert 1's gate LUT bytes
    uint64_t offset = 0;
    for (auto & group : index["groups"]) {
        for (auto & member : group["members"]) {
            member["offset"] = offset;
            offset += member["size"].get<uint64_t>();
        }
    }
    index["blob_bytes"] = offset;
    write_ml8_index_and_manifest(fixture, index);

    bool threw = false;
    std::string error;
    try {
        run_ml8_descriptor(fixture);
    } catch (const std::exception & e) {
        threw = true;
        error = e.what();
    }
    check(threw, "ML8_4 role with no LUT (lut_bytes=0) throws");
}

static void test_ml8_gate_up_rotation_mismatch() {
    Ml8Fixture fixture;
    json index = make_ml8_index(ML8_LUT, /*mutate_expert=*/-1, "", 0);
    json rot_layer = json::object();
    rot_layer["gate"] = make_rotation_entry(/*kronecker=*/true);
    rot_layer["up"]   = make_rotation_entry(/*kronecker=*/false); // deliberately different from gate
    index["ml8_rotation"] = { { "0", rot_layer } };
    write_ml8_index_and_manifest(fixture, index);

    bool threw = false;
    try {
        run_ml8_descriptor(fixture);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "gate and up rotations differing throws (they share the input)");
}

static void test_ml8_rotation_k_mismatch() {
    Ml8Fixture fixture;
    json index = make_ml8_index(ML8_LUT, /*mutate_expert=*/-1, "", 0);
    json bad_rotation = make_rotation_entry(/*kronecker=*/false);
    bad_rotation["k"] = 32; // does not match down's ne0 == 64
    bad_rotation["a"] = 1;
    bad_rotation["b"] = 32;
    json rot_layer = json::object();
    rot_layer["down"] = bad_rotation;
    index["ml8_rotation"] = { { "0", rot_layer } };
    write_ml8_index_and_manifest(fixture, index);

    bool threw = false;
    try {
        run_ml8_descriptor(fixture);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "rotation k != role ne0 throws");
}

int main() {
    Fixture fixture;
    test_slice_and_layered(fixture);
    test_negative_without_expert_type(fixture);
    test_ml8_lut_and_rotation();
    test_ml8_lut_mismatch_across_experts();
    test_ml8_missing_lut_throws();
    test_ml8_gate_up_rotation_mismatch();
    test_ml8_rotation_k_mismatch();
    if (g_fail != 0) {
        std::printf("%d test(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
