#include "pipe-prefetch-hints.h"

#include <algorithm>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace pipe_expert_dispatcher;

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_u16(std::ofstream & output, uint16_t value) {
    output.put((char) value);
    output.put((char) (value >> 8));
}

void write_u32(std::ofstream & output, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        output.put((char) (value >> shift));
    }
}

void write_u64(std::ofstream & output, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        output.put((char) (value >> shift));
    }
}

struct table_entry {
    uint16_t expert;
    uint32_t count;
};

void write_row(std::ofstream & output, uint32_t total, const std::vector<table_entry> & entries) {
    write_u32(output, total);
    write_u16(output, (uint16_t) entries.size());
    write_u16(output, 0);
    for (const table_entry & entry : entries) {
        write_u16(output, entry.expert);
        write_u32(output, entry.count);
    }
}

void write_test_table(const fs::path & path) {
    std::ofstream output(path, std::ios::binary);
    require((bool) output, "failed to create n-gram test table");
    output.write("WPNGRAM\0", 8);
    write_u32(output, 1);
    write_u32(output, 2);
    write_u32(output, 4);
    write_u32(output, 4);
    write_u64(output, 2);

    write_row(output, 100,
              {
                  { 3, 40 },
                  { 2, 30 },
                  { 1, 20 },
                  { 0, 10 }
    });
    write_row(output, 10,
              {
                  { 0, 4 },
                  { 1, 3 },
                  { 2, 2 },
                  { 3, 1 }
    });

    write_u32(output, 10);
    write_u16(output, 0);
    write_u16(output, 0);
    write_row(output, 100,
              {
                  { 0, 60 },
                  { 1, 20 }
    });

    write_u32(output, 20);
    write_u16(output, 0);
    write_u16(output, 0);
    write_row(output, 10,
              {
                  { 2, 9 },
                  { 1, 1 }
    });
}

void test_router2_per_token_union() {
    // Token 0 wants e0 then e2; token 1 wants e1 then e2. Max-pool-then-top-2
    // kept {0,1} and dropped e2, which is in BOTH tokens' top-2 -- the set the
    // target actually dispatches. Per-token top-2 union is {0,1,2}.
    const float weights[] = {
        4.0f, 0.0f, 0.0f, 4.0f, 3.0f, 3.0f, 0.0f, 0.0f,
    };
    const float bias[]        = { 0.0f, 0.0f, 0.0f, 0.0f };
    const float activations[] = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const std::vector<int32_t> top = router2_top_experts(weights, bias, activations, 2, 4, 2, 2);
    require(top == std::vector<int32_t>({ 0, 1, 2 }),
            "router2 must union each token's top-M, not max-pool then top-M");
}

void test_router2_confidence_gate() {
    // 4 experts, 2 dims. Expert 0 is strongly aligned with the activation,
    // expert 1 weakly, experts 2 and 3 not at all -- a PEAKED layer.
    const float weights[] = {
        8.0f, 0.0f,   // e0 . h = 8
        2.0f, 0.0f,   // e1 . h = 2
        0.0f, 0.0f,   // e2 . h = 0
        0.0f, 0.0f,   // e3 . h = 0
    };
    const float bias[]        = { 0.0f, 0.0f, 0.0f, 0.0f };
    const float activations[] = { 1.0f, 0.0f };

    const std::vector<int32_t> ungated =
        router2_top_experts(weights, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.0f);
    require(ungated.size() == 4, "ungated router2 must still emit the full top-M");

    // All-or-nothing: best expert clears 0.2, so the WHOLE top-M is emitted.
    // Truncating to only the ids above the floor leaves a layer partially
    // covered, which still demand-pages.
    const std::vector<int32_t> gated =
        router2_top_experts(weights, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.2f);
    require(gated == ungated, "peaked layer that clears the floor must emit the full top-M");
    require(std::is_sorted(gated.begin(), gated.end()),
            "gated router2 output must stay ascending for the wire");

    // A FLAT layer -- the router is undecided, every expert scores the same, so
    // no expert can clear a floor above 1/n_expert. Emitting nothing here is the
    // entire point: an undecided layer is where speculative reads are wasted.
    const float flat_w[] = {
        1.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
    };
    const std::vector<int32_t> flat =
        router2_top_experts(flat_w, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.5f);
    require(flat.empty(), "confidence gate emitted experts on a layer with no signal");

    const std::vector<int32_t> flat_ungated =
        router2_top_experts(flat_w, bias, activations, 1, 4, 2, 4, /*min_conf=*/0.0f);
    require(flat_ungated.size() == 4, "flat layer must emit when the gate is off");
}

void test_ngram_format_and_scoring(const fs::path & path) {
    write_test_table(path);
    const ngram_hint_table table(path.string());
    require(table.n_layers() == 2, "n-gram layer count is wrong");
    require(table.n_experts() == 4, "n-gram expert count is wrong");
    require(table.row_width() == 4, "n-gram row width is wrong");
    require(table.row_count() == 2, "n-gram token row count is wrong");

    const int32_t tokens[] = { 10, 20 };
    require(table.top_experts(tokens, 2, 0, 2) == std::vector<int32_t>({ 0, 2 }),
            "n-gram rows were not normalized per token before summing");

    const int32_t missing[] = { 999 };
    require(table.top_experts(missing, 1, 0, 2) == std::vector<int32_t>({ 2, 3 }),
            "n-gram popularity fallback is wrong");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2) {
        const ngram_hint_table table(argv[1]);
        require(table.n_layers() == 43, "built DS4 table has the wrong layer count");
        require(table.n_experts() == 256, "built DS4 table has the wrong expert count");
        require(table.row_width() == 16, "built DS4 table has the wrong row width");
        require(table.row_count() > 0, "built DS4 table has no token rows");
        return 0;
    }
    require(argc == 1, "usage: test-wp-prefetch-hints [table]");
    const fs::path path = fs::temp_directory_path() / ("wp-prefetch-hints-" + std::to_string((long) getpid()) + ".bin");
    try {
        test_router2_per_token_union();
        test_router2_confidence_gate();
        test_ngram_format_and_scoring(path);
        std::error_code ignored;
        fs::remove(path, ignored);
        return 0;
    } catch (...) {
        std::error_code ignored;
        fs::remove(path, ignored);
        throw;
    }
}
