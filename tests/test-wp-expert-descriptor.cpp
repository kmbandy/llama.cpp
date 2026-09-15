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

int main() {
    Fixture fixture;
    test_slice_and_layered(fixture);
    test_negative_without_expert_type(fixture);
    if (g_fail != 0) {
        std::printf("%d test(s) FAILED\n", g_fail);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
