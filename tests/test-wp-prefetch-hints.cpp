#include "pipe-prefetch-hints.h"

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

void test_router2_max_pool() {
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
    require(top == std::vector<int32_t>({ 0, 1 }), "router2 did not max-pool token scores before top-M");
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
        test_router2_max_pool();
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
