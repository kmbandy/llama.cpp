// wp-repack blob-set reader: acceptance and, more importantly, rejection.
//
// The reader's job is to redirect routed-expert pages at an expert-major blob.
// Getting that WRONG is silent: the pager would read well-formed bytes from the
// wrong place and the model would produce fluent, subtly incorrect output
// rather than crashing. So most of what follows checks that a bad set is
// refused rather than adopted.
//
// Fixtures are synthesized here (a few KB) -- no model, no GPU, no real blobs.

#include "wp-blob-index.h"
#include "wp-page-catalog.h"

#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        g_fail++;
    }
}

// Run `fn`, expecting it to throw with a message containing `needle`.
static void check_throws(const std::string & what, const std::string & needle,
                         const std::function<void()> & fn) {
    try {
        fn();
    } catch (const std::exception & e) {
        const bool matched = std::string(e.what()).find(needle) != std::string::npos;
        check(matched, what + (matched ? "" : " [message was: " + std::string(e.what()) + "]"));
        return;
    }
    check(false, what + " [no exception thrown]");
}

struct Fixture {
    fs::path dir;
    fs::path manifest;
    std::string model_name = "toy-model-00001-of-00001.gguf";

    // One shard, one group, three members (up/gate/down) of `member_size`
    // bytes each, laid out contiguously from offset 0.
    static constexpr uint64_t MEMBER = 64;
    static constexpr uint64_t BLOB   = MEMBER * 3;

    explicit Fixture(const std::string & tag) {
        dir = fs::temp_directory_path() / ("wp-blob-index-test-" + tag);
        fs::remove_all(dir);
        fs::create_directories(dir);
        manifest = dir / "toy-experts-manifest.json";
    }
    ~Fixture() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    void write_blob(uint64_t bytes) const {
        std::ofstream ofs(dir / "toy-experts-00001-of-00001.wpb", std::ios::binary);
        const std::vector<char> zeros((size_t) bytes, 0);
        ofs.write(zeros.data(), (std::streamsize) zeros.size());
    }

    void write_manifest(const std::string & format, int version,
                        uint64_t blob_bytes_recorded) const {
        std::ofstream ofs(manifest);
        ofs << "{\n"
            << "  \"format\": \"" << format << "\",\n"
            << "  \"version\": " << version << ",\n"
            << "  \"input_model\": \"/somewhere/else/" << model_name << "\",\n"
            << "  \"shards\": [ {\n"
            << "    \"blob_file\": \"toy-experts-00001-of-00001.wpb\",\n"
            << "    \"index_file\": \"toy-experts-00001-of-00001.wpi.json\",\n"
            << "    \"shard_index\": 0,\n"
            << "    \"group_count\": 1,\n"
            << "    \"blob_bytes\": " << blob_bytes_recorded << "\n"
            << "  } ]\n"
            << "}\n";
    }

    // `down_size` lets a member be made to overrun the blob.
    void write_index(uint64_t down_size = MEMBER) const {
        std::ofstream ofs(dir / "toy-experts-00001-of-00001.wpi.json");
        ofs << "{\n"
            << "  \"format\": \"llama.cpp.weight-pager.expert-shard-index\",\n"
            << "  \"version\": 1,\n"
            << "  \"blob_file\": \"toy-experts-00001-of-00001.wpb\",\n"
            << "  \"blob_bytes\": " << BLOB << ",\n"
            << "  \"groups\": [ {\n"
            << "    \"block_idx\": 3, \"expert_idx\": 0, \"member_count\": 3,\n"
            << "    \"members\": [\n"
            << "      { \"role_mask\": 1, \"offset\": 0,   \"size\": " << MEMBER
            << ", \"catalog_name\": \"blk.3.ffn_up_exps.weight#expert.0\" },\n"
            << "      { \"role_mask\": 2, \"offset\": " << MEMBER << ", \"size\": " << MEMBER
            << ", \"catalog_name\": \"blk.3.ffn_gate_exps.weight#expert.0\" },\n"
            << "      { \"role_mask\": 4, \"offset\": " << 2 * MEMBER << ", \"size\": " << down_size
            << ", \"catalog_name\": \"blk.3.ffn_down_exps.weight#expert.0\" }\n"
            << "    ]\n"
            << "  } ]\n"
            << "}\n";
    }

    // A well-formed set.
    void write_good() const {
        write_blob(BLOB);
        write_manifest("llama.cpp.weight-pager.expert-shard-manifest", 1, BLOB);
        write_index();
    }
};

static void test_happy_path() {
    printf("happy path\n");
    Fixture f("happy");
    f.write_good();

    const common_wp_blob_index idx =
        common_wp_blob_index_load(f.manifest.string(), "/models/" + f.model_name);

    check(idx.blob_files.size() == 1, "one blob file");
    check(idx.blob_file_ptrs.size() == 1, "one blob file pointer");
    check(idx.entries.size() == 3, "three expert pages");

    // Blob paths resolve relative to the manifest, not the cwd.
    check(fs::path(idx.blob_files[0]).parent_path() == f.dir,
          "blob path resolved against the manifest directory");

    // Names must survive the return-by-value move: this is the whole reason
    // the type forbids copying.
    check(std::string(idx.entries[0].name) == "blk.3.ffn_up_exps.weight#expert.0",
          "entry name survives the move out of the loader");
    check(std::string(idx.entries[2].name) == "blk.3.ffn_down_exps.weight#expert.0",
          "last entry name intact");

    // Contiguity is the point of the whole exercise: one expert's three
    // members must form a single unbroken run.
    bool contiguous = true;
    uint64_t expect_off = 0;
    for (const llama_wp_blob_entry & e : idx.entries) {
        contiguous = contiguous && e.blob_offset == expect_off && e.blob_idx == 0;
        expect_off += e.size;
    }
    check(contiguous, "gate/up/down are contiguous within the blob");
}

static void test_rejections() {
    printf("rejections\n");

    {   // A blob set built from a different model must never be adopted:
        // its experts are the wrong weights, and every size could still match.
        Fixture f("wrongmodel");
        f.write_good();
        check_throws("refuses a set built from a different model", "Refusing", [&] {
            common_wp_blob_index_load(f.manifest.string(), "/models/some-other-model.gguf");
        });
    }
    {   // Truncated or rewritten blob: sizes are the cheap guard that a load
        // does not read past the data that was actually packed.
        Fixture f("truncated");
        f.write_manifest("llama.cpp.weight-pager.expert-shard-manifest", 1, Fixture::BLOB);
        f.write_index();
        f.write_blob(Fixture::BLOB - 1);
        check_throws("refuses a truncated blob", "truncated", [&] {
            common_wp_blob_index_load(f.manifest.string(), "/models/" + f.model_name);
        });
    }
    {   // A member that runs past the end of its blob would short-read or
        // silently pick up the neighbouring expert's bytes.
        Fixture f("overrun");
        f.write_blob(Fixture::BLOB);
        f.write_manifest("llama.cpp.weight-pager.expert-shard-manifest", 1, Fixture::BLOB);
        f.write_index(/*down_size =*/ Fixture::MEMBER + 1);
        check_throws("refuses a member that overruns the blob", "does not fit", [&] {
            common_wp_blob_index_load(f.manifest.string(), "/models/" + f.model_name);
        });
    }
    {   // Format/version are how a future layout change announces itself.
        Fixture f("format");
        f.write_good();
        f.write_manifest("llama.cpp.weight-pager.something-else", 1, Fixture::BLOB);
        check_throws("refuses an unknown format", "expected", [&] {
            common_wp_blob_index_load(f.manifest.string(), "/models/" + f.model_name);
        });
    }
    {
        Fixture f("version");
        f.write_good();
        f.write_manifest("llama.cpp.weight-pager.expert-shard-manifest", 2, Fixture::BLOB);
        check_throws("refuses a future version", "version", [&] {
            common_wp_blob_index_load(f.manifest.string(), "/models/" + f.model_name);
        });
    }
    {
        Fixture f("missing");
        check_throws("reports a missing manifest", "cannot open", [&] {
            common_wp_blob_index_load((f.dir / "nope.json").string(), "/models/" + f.model_name);
        });
    }
}

// The catalog side of the same contract: a remap must land on a real, pageable
// page whose size agrees, and must leave everything else about it alone.
static void test_catalog_remap() {
    printf("catalog remap\n");
    wp::PageCatalog cat;
    const int dense = cat.add("blk.0.attn_q.weight", /*file_idx =*/ 0,
                              /*file_offset =*/ 4096, /*size =*/ 256);
    // 2 experts x 512 bytes -> sub-pages of 256 bytes each.
    cat.add_consolidated_experts("blk.3.ffn_up_exps.weight", 0, 8192, 512, 2);

    const int sub = cat.find("blk.3.ffn_up_exps.weight#expert.1");
    check(sub >= 0, "consolidated tensor produced the synthetic sub-expert name");
    const size_t sub_size = cat.at(sub).size;

    check(cat.remap_source("blk.3.ffn_up_exps.weight#expert.1", 7, 999, sub_size) ==
              wp::PageCatalog::RemapStatus::Ok,
          "remaps an expert sub-page");
    check(cat.at(sub).file_idx == 7 && cat.at(sub).file_offset == 999,
          "sub-page now points into the blob");
    check(cat.at(sub).size == sub_size && cat.at(sub).is_sub_expert &&
              cat.at(sub).block_idx == 3 && cat.at(sub).expert_idx == 1,
          "remap changes only the source location, not the page's identity");

    // The untouched dense page must be exactly where it was.
    check(cat.at(dense).file_idx == 0 && cat.at(dense).file_offset == 4096,
          "pages with no blob entry still read from the original GGUF");

    check(cat.remap_source("blk.9.does_not_exist.weight", 7, 0, 256) ==
              wp::PageCatalog::RemapStatus::NotFound,
          "unknown page reports NotFound rather than being created");
    check(cat.remap_source("blk.3.ffn_up_exps.weight#expert.0", 7, 0, sub_size + 1) ==
              wp::PageCatalog::RemapStatus::SizeMismatch,
          "a size disagreement is refused, not rounded");

    // The consolidated parent holds no bytes of its own; its children do.
    check(cat.remap_source("blk.3.ffn_up_exps.weight", 7, 0,
                           cat.at(cat.find("blk.3.ffn_up_exps.weight")).size) ==
              wp::PageCatalog::RemapStatus::NotPageable,
          "the consolidated parent cannot be remapped");

    int pinned_val = 0;
    cat.add_pinned("token_embd.weight", &pinned_val, 128);
    check(cat.remap_source("token_embd.weight", 7, 0, 128) ==
              wp::PageCatalog::RemapStatus::NotPageable,
          "a pinned page cannot be remapped");
}

int main() {
    test_happy_path();
    test_rejections();
    test_catalog_remap();

    if (g_fail != 0) {
        printf("\n%d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
