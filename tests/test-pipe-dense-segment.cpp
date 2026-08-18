#include "pipe-dense-segment-manifest.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

static int g_failed = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

static const char * MODEL_SHA =
    "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
static const char * ARTIFACT_A_SHA =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static const char * ARTIFACT_B_SHA =
    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

static std::string resign(std::string value) {
    const std::string checksum = pipe_dense_segment::manifest_checksum(value);
    const std::string needle = "\"checksum\":\"";
    const size_t begin = value.find(needle);
    if (begin == std::string::npos) {
        throw std::runtime_error("missing checksum field in test fixture");
    }
    const size_t value_begin = begin + needle.size();
    const size_t value_end = value.find('"', value_begin);
    if (value_end == std::string::npos) {
        throw std::runtime_error("invalid checksum field in test fixture");
    }
    value.replace(value_begin, value_end - value_begin, checksum);
    return value;
}

static std::string valid_manifest() {
    std::string value =
        "{\"format\":\"llama.cpp.dense-segment-manifest\",\"version\":1,"
        "\"model_identity_sha256\":\"" + std::string(MODEL_SHA) + "\","
        "\"n_layer\":8,\"n_embd\":4096,\"wire_precision\":\"f32\","
        "\"segments\":["
        "{\"id\":4,\"layer_first\":4,\"layer_last\":7,\"host\":\"gpu1\",\"port\":9101,"
        "\"stage_gguf\":\"stage-4.gguf\",\"device\":\"HIP1\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_B_SHA) + "\"},"
        "{\"id\":2,\"layer_first\":0,\"layer_last\":3,\"host\":\"gpu0\",\"port\":9100,"
        "\"stage_gguf\":\"stage-2.gguf\",\"device\":\"HIP0\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_A_SHA) + "\"}],"
        "\"checksum\":\"pending\"}";
    return resign(std::move(value));
}

static void expect_rejected(const std::function<void()> & fn) {
    try {
        fn();
        CHECK(false);
    } catch (const std::runtime_error &) {
    }
}

int main() {
    const std::string valid = valid_manifest();
    const pipe_dense_segment::manifest parsed =
        pipe_dense_segment::parse_manifest(valid, MODEL_SHA);
    CHECK(parsed.model_identity_sha256 == MODEL_SHA);
    CHECK(parsed.n_layer == 8);
    CHECK(parsed.n_embd == 4096);
    CHECK(parsed.wire_precision == "f32");
    CHECK(parsed.segments.size() == 2);
    CHECK(parsed.segments[0].id == 2);
    CHECK(parsed.segments[0].layer_first == 0);
    CHECK(parsed.segments[0].stage_gguf == "stage-2.gguf");
    CHECK(parsed.segments[1].id == 4);
    CHECK(parsed.segments[1].layer_last == 7);

    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(valid, ARTIFACT_A_SHA);
    });

    std::string bad_checksum = valid;
    const size_t checksum_pos = bad_checksum.find("fnv1a64:");
    CHECK(checksum_pos != std::string::npos);
    if (checksum_pos != std::string::npos) {
        bad_checksum[checksum_pos + 8] = bad_checksum[checksum_pos + 8] == '0' ? '1' : '0';
    }
    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(bad_checksum);
    });

    std::string unknown_format = valid;
    unknown_format.replace(unknown_format.find("llama.cpp.dense-segment-manifest"),
                           std::string("llama.cpp.dense-segment-manifest").size(),
                           "unknown.format");
    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(resign(std::move(unknown_format)));
    });

    std::string unknown_version = valid;
    unknown_version.replace(unknown_version.find("\"version\":1"), std::string("\"version\":1").size(),
                            "\"version\":2");
    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(resign(std::move(unknown_version)));
    });

    std::string duplicate_id = valid;
    const size_t id_pos = duplicate_id.find("\"id\":4");
    CHECK(id_pos != std::string::npos);
    if (id_pos != std::string::npos) {
        duplicate_id.replace(id_pos, std::string("\"id\":4").size(), "\"id\":2");
    }
    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(resign(std::move(duplicate_id)));
    });

    std::string gap = valid;
    const size_t gap_pos = gap.find("\"layer_first\":4");
    CHECK(gap_pos != std::string::npos);
    if (gap_pos != std::string::npos) {
        gap.replace(gap_pos, std::string("\"layer_first\":4").size(), "\"layer_first\":5");
    }
    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(resign(std::move(gap)));
    });

    std::string overlap = valid;
    const size_t overlap_pos = overlap.find("\"layer_first\":4");
    CHECK(overlap_pos != std::string::npos);
    if (overlap_pos != std::string::npos) {
        overlap.replace(overlap_pos, std::string("\"layer_first\":4").size(), "\"layer_first\":3");
    }
    expect_rejected([&]() {
        pipe_dense_segment::parse_manifest(resign(std::move(overlap)));
    });

    if (g_failed == 0) {
        std::printf("test-pipe-dense-segment: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-pipe-dense-segment: %d check(s) failed\n", g_failed);
    return 1;
}
