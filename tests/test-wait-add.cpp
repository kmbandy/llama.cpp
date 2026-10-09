// CPU-only test of GGML_OP_WAIT_ADD: the pinned-slot / epoch-flag handshake and
// bit-exactness against ggml_add. Nothing here touches a GPU: the op runs on the
// CPU reference implementation (the HIP kernel implements the same contract).
//
// Covers:
//   1. zerocopy mode (ctrl == NULL): dst == ggml_add(a, slot), bitwise.
//   2. flag mode: the consumer graph is started BEFORE the data exist; a producer
//      thread folds three worker partials in fixed order, writes the slot,
//      release-fences and release-stores the epoch. Re-running the SAME graph
//      with a new epoch each iteration (= graph replay) must stay bit-exact.
//   3. timeout: a flag that is never published trips ctrl->fail and the op
//      contributes 0 instead of hanging.
#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); std::exit(1); } } while (0)

static const int N = 4096 * 4;

static void fill(std::vector<float> & v, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    for (float & x : v) {
        x = d(rng);
    }
}

// Reference: ggml_add(a, b) through the same CPU backend.
static void ref_add(const std::vector<float> & a, const std::vector<float> & b, std::vector<float> & out) {
    ggml_init_params ip = { 1u << 22, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    std::memcpy(ta->data, a.data(), N * sizeof(float));
    std::memcpy(tb->data, b.data(), N * sizeof(float));
    ggml_tensor * r = ggml_add(ctx, ta, tb);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    CHECK(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
    out.assign((float *) r->data, (float *) r->data + N);
    ggml_free(ctx);
}

// The dispatcher's fold: partials summed in FIXED worker order.
static void fold(const std::vector<float> (&w)[3], float * out) {
    for (int i = 0; i < N; ++i) {
        out[i] = (w[0][i] + w[1][i]) + w[2][i];
    }
}

int main() {
    std::mt19937 rng(1234);
    std::vector<float> a(N), ref, w[3] = { std::vector<float>(N), std::vector<float>(N), std::vector<float>(N) };

    ggml_init_params ip = { 1u << 22, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);

    // pinned-slot stand-ins: 64-byte aligned like the real allocation
    ggml_wait_ctrl * ctrl = (ggml_wait_ctrl *) std::aligned_alloc(64, sizeof(ggml_wait_ctrl));
    float *          slot = (float *) std::aligned_alloc(64, N * sizeof(float));
    std::memset(ctrl, 0, sizeof(*ctrl));

    // ---- 1. zerocopy (no device wait) ------------------------------------
    {
        fill(a, rng);
        for (auto & v : w) fill(v, rng);
        std::memcpy(ta->data, a.data(), N * sizeof(float));
        fold(w, slot);
        ggml_tensor * dep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);   // ordering edge only
        ggml_tensor * r = ggml_wait_add(ctx, ta, dep, slot, nullptr, 1000);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        CHECK(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
        std::vector<float> folded(slot, slot + N);
        ref_add(a, folded, ref);
        CHECK(std::memcmp(r->data, ref.data(), N * sizeof(float)) == 0);
        std::printf("zerocopy: bit-exact vs ggml_add\n");
    }

    // ---- 2. flag mode, one graph replayed across epochs --------------------
    {
        ggml_tensor * r = ggml_wait_add(ctx, ta, nullptr, slot, ctrl, 10000);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        uint64_t seq = 0;
        for (int it = 0; it < 500; ++it) {
            fill(a, rng);
            for (auto & v : w) fill(v, rng);
            std::memcpy(ta->data, a.data(), N * sizeof(float));

            // dispatch thread: `issue` assigns the epoch before the consumer starts
            const uint64_t epoch = ++seq;
            __atomic_store_n(&ctrl->expect, epoch, __ATOMIC_RELEASE);

            std::atomic<int> consumer_started{0};
            ggml_status st = GGML_STATUS_FAILED;
            std::thread consumer([&] {           // "GPU stream": enqueued before the data exist
                consumer_started.store(1);
                st = ggml_graph_compute_with_ctx(ctx, gf, 1);
            });
            while (!consumer_started.load()) {}
            std::this_thread::sleep_for(std::chrono::microseconds(rng() % 300));

            // background fold thread: fold, write slot, release fence, release-store flag
            std::thread producer([&] {
                fold(w, slot);
                std::atomic_thread_fence(std::memory_order_release);
                __atomic_store_n(&ctrl->flag, epoch, __ATOMIC_RELEASE);
            });
            producer.join();
            consumer.join();
            CHECK(st == GGML_STATUS_SUCCESS);

            std::vector<float> folded(slot, slot + N);
            ref_add(a, folded, ref);
            CHECK(std::memcmp(r->data, ref.data(), N * sizeof(float)) == 0);
            CHECK(__atomic_load_n(&ctrl->fail, __ATOMIC_ACQUIRE) == 0);
        }
        std::printf("flag: 500 replays with fresh epochs, bit-exact, no spurious failure\n");
    }

    // ---- 3. timeout -> failure marker, no hang -----------------------------
    {
        ggml_tensor * r = ggml_wait_add(ctx, ta, nullptr, slot, ctrl, 30);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        __atomic_store_n(&ctrl->expect, (uint64_t) 999999, __ATOMIC_RELEASE);   // never published
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ms >= 30 && ms < 5000);
        CHECK(__atomic_load_n(&ctrl->fail, __ATOMIC_ACQUIRE) == 1);
        CHECK(std::memcmp(r->data, ta->data, N * sizeof(float)) == 0);   // partial contributed 0
        std::printf("timeout: failure marker set after %lld ms, partial dropped\n", (long long) ms);
    }

    std::free(ctrl);
    std::free(slot);
    ggml_free(ctx);
    std::printf("OK\n");
    return 0;
}
