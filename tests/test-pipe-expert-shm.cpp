#include "pipe-expert-shm.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#  include <unistd.h>
#endif

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-pipe-expert-shm: %s\n", message);
    }
    return condition;
}

int main() {
#if !defined(__linux__)
    return 0;
#else
    const std::string name = "/llama_test_pipe_expert_shm_" + std::to_string((long long) getpid());
    std::unique_ptr<pipe_expert_shm_ring> ring =
        pipe_expert_shm_ring::create_for_test(name, 128, 64);
    if (!check(ring != nullptr && ring->valid(), "failed to create test ring")) {
        return 1;
    }

    const std::vector<uint8_t> first(40, 0x11);
    const std::vector<uint8_t> second(40, 0x22);
    const std::vector<uint8_t> wrapped(40, 0x33);
    pipe_expert_shm_ref ref{};
    const uint8_t * view = nullptr;
    if (!check(ring->write_request(first.data(), first.size(), ref), "first write failed") ||
        !check(ring->read_request(ref, &view), "first read failed") ||
        !check(std::memcmp(view, first.data(), first.size()) == 0, "first data changed") ||
        !check(ring->consume_request(ref), "first consume failed")) {
        return 1;
    }
    if (!check(ring->write_request(second.data(), second.size(), ref), "second write failed") ||
        !check(ring->read_request(ref, &view), "second read failed") ||
        !check(std::memcmp(view, second.data(), second.size()) == 0, "second data changed") ||
        !check(ring->consume_request(ref), "second consume failed")) {
        return 1;
    }
    // The next record cannot fit at the end of the data area. It must emit a
    // wrap marker and be readable at the beginning without corrupting bytes.
    if (!check(ring->write_request(wrapped.data(), wrapped.size(), ref), "wrapped write failed") ||
        !check(ring->read_request(ref, &view), "wrapped read failed") ||
        !check(std::memcmp(view, wrapped.data(), wrapped.size()) == 0, "wrapped data changed") ||
        !check(ring->consume_request(ref), "wrapped consume failed")) {
        return 1;
    }

    const std::vector<uint8_t> response(32, 0x44);
    if (!check(ring->write_response(response.data(), response.size(), ref), "response write failed") ||
        !check(ring->read_response(ref, &view), "response read failed") ||
        !check(std::memcmp(view, response.data(), response.size()) == 0, "response data changed") ||
        !check(ring->consume_response(ref), "response consume failed")) {
        return 1;
    }
    return 0;
#endif
}
