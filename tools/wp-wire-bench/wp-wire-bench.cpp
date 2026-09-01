#include "pipe-protocol.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void usage(const char * program) {
    std::fprintf(stderr, "usage: %s raw-f32-dump [n_embd]\n", program);
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        usage(argv[0]);
        return 2;
    }

    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    const std::streamoff end = input.tellg();
    if (end < 0 || (uint64_t) end % sizeof(float) != 0) {
        std::fprintf(stderr, "input size is not a multiple of 4 bytes\n");
        return 1;
    }
    const size_t size = (size_t) end;
    input.seekg(0);
    std::vector<uint8_t> source(size);
    if (size != 0 && !input.read((char *) source.data(), (std::streamsize) size)) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }

    if (argc == 3) {
        char * parse_end = nullptr;
        const unsigned long long n_embd = std::strtoull(argv[2], &parse_end, 10);
        if (*argv[2] == '\0' || parse_end == argv[2] || *parse_end != '\0' ||
            n_embd == 0 || (size / sizeof(float)) % n_embd != 0) {
            std::fprintf(stderr, "n_embd must divide the number of f32 values\n");
            return 1;
        }
        std::printf("input: %zu bytes, %llu f32 rows of %llu\n",
                    size, (unsigned long long) ((size / sizeof(float)) / n_embd), n_embd);
    } else {
        std::printf("input: %zu bytes\n", size);
    }

    const pipe_wire_compress_mode modes[] = {
        PIPE_WIRE_COMPRESS_RAW_LZ4,
        PIPE_WIRE_COMPRESS_SHUFFLE,
        PIPE_WIRE_COMPRESS_SELECTIVE,
    };
    const char * names[] = { "raw-lz4", "shuffle-lz4", "selective" };
    const int iterations = 3;
    const double input_mb = (double) size / (1024.0 * 1024.0);
    for (size_t i = 0; i < 3; ++i) {
        std::vector<uint8_t> encoded;
        std::vector<uint8_t> decoded;
        const auto compress_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            pipe_wire_compress_payload(source.data(), source.size(), modes[i], encoded);
        }
        const double compress_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - compress_start).count() / iterations;

        const auto decompress_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            pipe_wire_decompress_payload(encoded.data(), encoded.size(), modes[i], decoded);
        }
        const double decompress_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decompress_start).count() / iterations;
        if (decoded != source) {
            std::fprintf(stderr, "%s failed byte-identical round-trip\n", names[i]);
            return 1;
        }

        std::printf("mode %u (%s): ratio %.4f, compress %.1f MB/s, decompress %.1f MB/s\n",
                    (unsigned) modes[i], names[i],
                    size == 0 ? 0.0 : (double) encoded.size() / size,
                    compress_seconds == 0.0 ? 0.0 : input_mb / compress_seconds,
                    decompress_seconds == 0.0 ? 0.0 : input_mb / decompress_seconds);
    }
    return 0;
}
