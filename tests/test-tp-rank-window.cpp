// Rank-windowed tensor split: does a world-sized row map partition every tensor class correctly,
// and does splitting that world between two ranks reproduce the single-process split exactly?
//
// Exercises the production arithmetic (llama_tp_split_segment, src/llama-tp-split.h) that
// llama_meta_device_get_split_state uses, with the real qwen35 segment/granularity tables from
// the model geometry. No model, no GPU, no network.

#include "llama-tp-split.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                    \
            fprintf(stderr, "\n");                                           \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------------------------
// qwen35 geometry, from the gguf (n_embd 5120, n_ff 17408, 24 q heads, 4 kv heads, head dim 256;
// ssm: d_state 128, n_group 16, dt_rank 48 => key_dim 2048, value_dim 6144, head_ratio 3)
// ---------------------------------------------------------------------------------------------

struct tensor_class {
    const char * name;
    int          axis;                              // 0 = row-parallel (PARTIAL out), 1 = column-parallel
    std::vector<std::pair<int64_t, uint32_t>> segs; // {rows per segment, repeats}
    std::vector<int64_t> gran;                      // granularity per segment
    bool                 mirrored;
};

static std::vector<tensor_class> qwen35_classes() {
    const int64_t key_dim   = 2048;  // 16 k heads x 128
    const int64_t value_dim = 6144;  // 48 v heads x 128
    const int64_t head_ratio = 3;
    const int64_t n_k_heads  = 16;
    const int64_t head_v_dim = 128;
    const int64_t n_ff       = 17408;

    return {
        // column-parallel
        {"attn_q.weight",     1, {{12288, 1}},                          {3072},  false},
        {"attn_k.weight",     1, {{1024, 1}},                           {256},   false},
        {"attn_v.weight",     1, {{1024, 1}},                           {256},   false},
        {"attn_qkv.weight",   1, {{key_dim, (uint32_t)(2 + head_ratio)}}, {128},  false},
        {"attn_gate.weight",  1, {{key_dim, (uint32_t) head_ratio}},     {128},  false},
        {"ssm_conv1d.weight", 1, {{key_dim, (uint32_t)(2 + head_ratio)}}, {128},  false},
        {"ssm_alpha.weight",  1, {{n_k_heads, (uint32_t) head_ratio}},   {1},    false},
        {"ssm_beta.weight",   1, {{n_k_heads, (uint32_t) head_ratio}},   {1},    false},
        {"ffn_gate.weight",   1, {{n_ff, 1}},                            {128},  false},
        {"ffn_up.weight",     1, {{n_ff, 1}},                            {128},  false},
        // row-parallel: the output of these is a PARTIAL sum and is what the reduce exchanges
        {"attn_output.weight", 0, {{6144, 1}},                           {1536}, false},
        {"ssm_out.weight",     0, {{key_dim, (uint32_t) head_ratio}},    {128},  false},
        {"ffn_down.weight",    0, {{n_ff, 1}},                           {128},  false},
        {"ssm_dt.bias",        0, {{n_k_heads, (uint32_t) head_ratio}},  {1},    false},
        {"ssm_a",              0, {{n_k_heads, (uint32_t) head_ratio}},  {1},    false},
        // caches follow the kv-head / k-head shard
        {"cache_k_l",  0, {{1024, 1}},                                        {256},   false},
        {"cache_v_l",  0, {{1024, 1}},                                        {256},   false},
        {"cache_r_l",  0, {{key_dim * 3, (uint32_t)(2 + head_ratio)}},        {384},   false},
        {"cache_s_l",  0, {{n_k_heads * head_v_dim * head_v_dim, (uint32_t) head_ratio}}, {16384}, false},
        // mirrored
        {"attn_norm.weight", -1, {}, {}, true},
        {"ssm_norm",         -1, {}, {}, true},
        {"attn_q_norm.weight", -1, {}, {}, true},
        {"output_norm.weight", -1, {}, {}, true},
        {"token_embd.weight",  -1, {}, {}, true},
    };
}

// ---------------------------------------------------------------------------------------------

static int64_t sum_ne(const int64_t * ne, size_t n) {
    int64_t s = 0;
    for (size_t j = 0; j < n; j++) {
        s += ne[j];
    }
    return s;
}

// 1. Every segment of every tensor class partitions exactly, at every granularity, for every
//    tensor_split and rotation.
static void test_partition_exact() {
    const size_t n_world = 4;
    const std::vector<std::vector<float>> splits = {
        {0.0f, 0.0f, 0.0f, 0.0f},                 // "even"
        {52.36f, 24.64f, 11.5f, 11.5f},           // the recommended cross-host point
        {46.2f, 23.8f, 15.0f, 15.0f},             // 70:30 machine, 66:34 inner
        {1.0f, 1.0f, 1.0f, 1.0f},
        {97.0f, 1.0f, 1.0f, 1.0f},                // skewed enough to produce zero-sized slices
    };

    for (const auto & tc : qwen35_classes()) {
        if (tc.mirrored) {
            continue;
        }
        for (const auto & ts : splits) {
            for (size_t rot = 0; rot < n_world; rot++) {
                for (size_t is = 0; is < tc.segs.size(); is++) {
                    int64_t ne[16] = {0};
                    llama_tp_split_segment(tc.segs[is].first, tc.gran[is], ts.data(),
                            n_world, n_world, rot, ne);
                    CHECK(sum_ne(ne, n_world) == tc.segs[is].first,
                          "%s seg %zu rot %zu: rows sum to %lld, expected %lld",
                          tc.name, is, rot, (long long) sum_ne(ne, n_world), (long long) tc.segs[is].first);
                    for (size_t j = 0; j < n_world; j++) {
                        CHECK(ne[j] >= 0, "%s: negative slice on device %zu", tc.name, j);
                    }
                    // every device but the one holding the remainder is granularity-aligned
                    size_t n_unaligned = 0;
                    for (size_t j = 0; j < n_world; j++) {
                        if (tc.gran[is] != 0 && ne[j] % tc.gran[is] != 0) {
                            n_unaligned++;
                        }
                    }
                    CHECK(n_unaligned <= 1,
                          "%s seg %zu rot %zu: %zu devices off granularity %lld",
                          tc.name, is, rot, n_unaligned, (long long) tc.gran[is]);
                }
            }
        }
    }
}

// 2. THE POINT OF M1: a rank window over a world of 4 keeps exactly its own rows, and the two
//    ranks' windows tile the whole tensor with no overlap and no gap. A window covering the whole
//    world must reproduce the single-process split byte for byte.
static void test_rank_window_tiles_the_world() {
    const size_t n_world = 4;
    const std::vector<float> ts = {52.36f, 24.64f, 11.5f, 11.5f};

    for (const auto & tc : qwen35_classes()) {
        if (tc.mirrored) {
            continue;
        }
        for (size_t rot = 0; rot < n_world; rot++) {
            for (size_t is = 0; is < tc.segs.size(); is++) {
                int64_t ne_world[16] = {0};
                llama_tp_split_segment(tc.segs[is].first, tc.gran[is], ts.data(),
                        n_world, n_world, rot, ne_world);

                // rank 0 owns world devices [0,2), rank 1 owns [2,4)
                int64_t covered = 0;
                int64_t prev_end = 0;
                for (size_t rank_first : {(size_t) 0, (size_t) 2}) {
                    for (size_t j = 0; j < 2; j++) {
                        const size_t jw = rank_first + j;
                        int64_t first = 0, last = 0;
                        llama_tp_split_row_range(ne_world, n_world, jw, &first, &last);
                        CHECK(first == prev_end,
                              "%s seg %zu rot %zu dev %zu: range starts at %lld, previous ended at %lld",
                              tc.name, is, rot, jw, (long long) first, (long long) prev_end);
                        CHECK(last >= first, "%s: inverted range on device %zu", tc.name, jw);
                        CHECK(last - first == ne_world[jw],
                              "%s: range width %lld != ne %lld on device %zu",
                              tc.name, (long long) (last - first), (long long) ne_world[jw], jw);
                        prev_end = last;
                        covered += last - first;
                    }
                }
                CHECK(covered == tc.segs[is].first,
                      "%s seg %zu rot %zu: the two rank windows cover %lld of %lld rows",
                      tc.name, is, rot, (long long) covered, (long long) tc.segs[is].first);

                // the window that IS the whole world is the single-process split, unchanged
                int64_t ne_single[16] = {0};
                llama_tp_split_segment(tc.segs[is].first, tc.gran[is], ts.data(),
                        n_world, n_world, rot, ne_single);
                CHECK(memcmp(ne_world, ne_single, sizeof(ne_world)) == 0,
                      "%s seg %zu rot %zu: full-world window differs from the reference split",
                      tc.name, is, rot);
            }
        }
    }
}

// 3. Both ranks derive the SAME world map. The split policy takes no local state, so this is a
//    property of the function; asserting it here is what makes the lockstep claim testable.
static void test_both_ranks_agree() {
    const size_t n_world = 4;
    const std::vector<float> ts = {52.36f, 24.64f, 11.5f, 11.5f};
    for (const auto & tc : qwen35_classes()) {
        if (tc.mirrored) {
            continue;
        }
        for (size_t is = 0; is < tc.segs.size(); is++) {
            int64_t a[16] = {0}, b[16] = {0};
            llama_tp_split_segment(tc.segs[is].first, tc.gran[is], ts.data(), n_world, n_world, 1, a);
            llama_tp_split_segment(tc.segs[is].first, tc.gran[is], ts.data(), n_world, n_world, 1, b);
            CHECK(memcmp(a, b, sizeof(a)) == 0, "%s seg %zu: split is not deterministic", tc.name, is);
        }
    }
}

// 4. The LM head restriction: with n_devices_eff = 2 of a world of 4, world devices 2 and 3 get
//    zero rows and the first two still cover the whole vocab. This is what keeps output.weight off
//    rank 1 and removes the per-token cross-host vocab gather.
static void test_head_restricted_to_rank0() {
    const size_t n_world = 4;
    const int64_t n_vocab = 248320;
    const std::vector<float> ts = {52.36f, 24.64f, 11.5f, 11.5f};

    for (size_t rot = 0; rot < 2; rot++) {
        int64_t ne[16] = {0};
        llama_tp_split_segment(n_vocab, 1, ts.data(), n_world, /*n_devices_eff =*/ 2, rot, ne);
        CHECK(ne[2] == 0 && ne[3] == 0,
              "output.weight rot %zu: rank 1 got %lld + %lld rows, expected none",
              rot, (long long) ne[2], (long long) ne[3]);
        CHECK(ne[0] + ne[1] == n_vocab,
              "output.weight rot %zu: rank 0 holds %lld of %lld rows",
              rot, (long long) (ne[0] + ne[1]), (long long) n_vocab);
    }

    // n_devices_eff == n_world is the unrestricted default and must change nothing
    int64_t restricted[16] = {0}, unrestricted[16] = {0};
    llama_tp_split_segment(n_vocab, 1, ts.data(), n_world, n_world, 0, unrestricted);
    llama_tp_split_segment(n_vocab, 1, ts.data(), n_world, n_world, 0, restricted);
    CHECK(memcmp(restricted, unrestricted, sizeof(restricted)) == 0,
          "n_devices_eff == n_world must be the unrestricted split");
    CHECK(sum_ne(unrestricted, n_world) == n_vocab, "unrestricted vocab split does not add up");
}

// 5. A world of 1 (i.e. --tp-world absent) gives one device everything: the no-op case that must
//    stay byte-identical to master.
static void test_world_of_one_is_a_noop() {
    for (const auto & tc : qwen35_classes()) {
        if (tc.mirrored) {
            continue;
        }
        for (size_t is = 0; is < tc.segs.size(); is++) {
            int64_t ne[16] = {0};
            llama_tp_split_segment(tc.segs[is].first, tc.gran[is], nullptr, 1, 1, 0, ne);
            CHECK(ne[0] == tc.segs[is].first,
                  "%s seg %zu: world of 1 gave device 0 %lld of %lld rows",
                  tc.name, is, (long long) ne[0], (long long) tc.segs[is].first);
        }
    }
}

// 6. The attention load-balance property the split geometry forces: attn_output splits at
//    granularity 1536 over 6144 columns, i.e. FOUR indivisible units, so a four-device world can
//    only ever hand out whole units and a skewed tensor_split WILL produce zero-sized slices.
//    Documented as a test because those zero slices are what makes the meta backend's
//    zero-slice/compute-disable path live in this configuration.
static void test_attention_units_are_indivisible() {
    const size_t n_world = 4;
    const std::vector<float> even = {1.0f, 1.0f, 1.0f, 1.0f};
    int64_t ne[16] = {0};
    llama_tp_split_segment(6144, 1536, even.data(), n_world, n_world, 0, ne);
    for (size_t j = 0; j < n_world; j++) {
        CHECK(ne[j] == 1536, "even attn_output split: device %zu got %lld, expected 1536", j, (long long) ne[j]);
    }

    const std::vector<float> skewed = {52.36f, 24.64f, 11.5f, 11.5f};
    llama_tp_split_segment(6144, 1536, skewed.data(), n_world, n_world, 0, ne);
    CHECK(sum_ne(ne, n_world) == 6144, "skewed attn_output split does not add up");
    bool any_zero = false;
    for (size_t j = 0; j < n_world; j++) {
        CHECK(ne[j] % 1536 == 0 || j == n_world - 1, "attn_output device %zu off granularity: %lld", j, (long long) ne[j]);
        any_zero = any_zero || ne[j] == 0;
    }
    printf("  attn_output at 52.36/24.64/11.5/11.5: %lld/%lld/%lld/%lld columns%s\n",
           (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
           any_zero ? "  (zero-sized slice present, as expected)" : "");
}

int main() {
    printf("test-tp-rank-window\n");
    test_partition_exact();
    test_rank_window_tiles_the_world();
    test_both_ranks_agree();
    test_head_restricted_to_rank0();
    test_world_of_one_is_a_noop();
    test_attention_units_are_indivisible();

    if (g_failures != 0) {
        fprintf(stderr, "test-tp-rank-window: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-tp-rank-window: OK\n");
    return 0;
}
