// CPU benchmark + byte-identity check for the spine-side MoE "plan + encode"
// hot path (2026-09-25 profiling task). Builds a synthetic, realistic plan
// input matching the live DeepSeek-V4.1 config of record read from
// ~/models/dsv41-spine-ml84.gguf's own metadata (deepseek41.embedding_length
// = 5120, deepseek41.expert_count = 384, deepseek41.expert_used_count = 6):
// 8192-token ubatch (WP_DISPATCH_STREAM prefill ubatch size), top-6 routing
// over 384 experts, two workers with the live w21 split (expert 0-255 /
// 256-383), and times:
//
//   "plan"   -- classifying each routed (expert, per-token-weight) assignment
//               to the worker that owns it. For this disjoint-range layout
//               that is choose_worker()'s WP_DISPATCH_STATIC_ASSIGN path
//               collapsed to candidates[0] (see pipe-expert-dispatcher.cpp's
//               choose_worker comment); this file reimplements exactly that
//               rule rather than calling the dispatcher's private
//               plan_requests(), which requires a live two-worker TCP session
//               to construct. The thing actually being benchmarked -- the
//               classification loop's per-thread partition + order-preserving
//               merge -- is the SAME code shape added to plan_requests() in
//               pipe-expert-dispatcher.cpp, using the same pipe_parallel_for.
//   "encode" -- pipe_expert_wire_pack_matrix() (pack once, ml8_4 wire) plus
//               pipe_encode_expert_dispatch_req_prepacked() per worker
//               (header + per-token weights + packed activation slice) --
//               called directly, unmodified production code from
//               pipe-protocol.cpp.
//
// WP_EXPERT_PLAN_THREADS / WP_EXPERT_ENCODE_THREADS are latched ONCE per
// process (see their accessors), so comparing thread counts within one run
// means one process per count: this binary re-execs itself (fork+exec) with
// WP_BENCH_CHILD=1 and the two knobs set, once per thread count, and the
// parent compares each child's SHA-256 of (a) the two workers' final
// assignment-order (expert-id sequence) and (b) their encoded wire payloads.
// Byte-identity across thread counts on both hashes is the correctness gate
// item 3 of the profiling task asks for.

#include "pipe-protocol.h"
#include "pipe-thread-pool.h"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// Config of record dimensions (see file header).
constexpr int32_t  k_n_embd    = 5120;
constexpr int32_t  k_n_expert  = 384;
constexpr int32_t  k_top_k     = 6;
constexpr uint32_t k_n_tokens  = 8192;
constexpr int32_t  k_layer     = 0;
constexpr float    k_clamp     = 1.0f;
// Live w21 split: main (loopback) owns 0-255, mad-lab-2026 (remote) owns 256-383.
constexpr int32_t  k_worker0_last = 255;

std::string to_hex(const unsigned char * digest) {
    static const char * hexd = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 32; ++i) {
        out[2 * i]     = hexd[(digest[i] >> 4) & 0xF];
        out[2 * i + 1] = hexd[digest[i] & 0xF];
    }
    return out;
}

// Deterministic synthetic router output: for each of k_n_tokens tokens, pick
// k_top_k distinct experts (seeded PRNG, same seed every run so every thread
// count sees the IDENTICAL input) and a softmax-like positive weight. Returns
// one pipe_expert_assignment per expert that received at least one token
// (almost certainly all k_n_expert of them at this width), ordered by
// ascending expert id -- exactly the order plan_requests() iterates its
// `assignments` parameter in.
std::vector<pipe_expert_assignment> build_synthetic_routing() {
    std::mt19937_64 rng(0xD54100C0FFEEULL);
    std::vector<std::vector<float>> weights_by_expert(
        (size_t) k_n_expert, std::vector<float>((size_t) k_n_tokens, 0.0f));
    std::vector<int32_t> pool((size_t) k_n_expert);
    for (int32_t e = 0; e < k_n_expert; ++e) pool[e] = e;
    std::uniform_real_distribution<float> wdist(0.05f, 1.0f);
    for (uint32_t t = 0; t < k_n_tokens; ++t) {
        std::shuffle(pool.begin(), pool.end(), rng);
        float wsum = 0.0f;
        float picked[k_top_k];
        for (int k = 0; k < k_top_k; ++k) {
            picked[k] = wdist(rng);
            wsum += picked[k];
        }
        for (int k = 0; k < k_top_k; ++k) {
            weights_by_expert[(size_t) pool[k]][t] = picked[k] / wsum;
        }
    }
    std::vector<pipe_expert_assignment> out;
    out.reserve((size_t) k_n_expert);
    for (int32_t e = 0; e < k_n_expert; ++e) {
        bool any = false;
        for (float w : weights_by_expert[(size_t) e]) {
            if (w != 0.0f) { any = true; break; }
        }
        if (!any) continue; // not expected at this width, but stay honest
        pipe_expert_assignment a;
        a.expert_id = e;
        a.weights   = std::move(weights_by_expert[(size_t) e]);
        out.push_back(std::move(a));
    }
    return out;
}

std::vector<float> build_synthetic_activations() {
    std::mt19937_64 rng(0xACF1FEEDULL);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> acts((size_t) k_n_tokens * (size_t) k_n_embd);
    for (float & v : acts) v = dist(rng);
    return acts;
}

// The "plan" phase: classify `all` into two disjoint worker buckets, using
// pipe_parallel_for the same way plan_requests()'s WP_EXPERT_PLAN_THREADS
// path does (chunk, classify+copy per chunk, merge in original chunk order --
// see that function's comment in pipe-expert-dispatcher.cpp for why this is
// order-preserving and therefore byte-identical across thread counts).
void plan_classify(const std::vector<pipe_expert_assignment> & all, int plan_threads,
                   std::vector<pipe_expert_assignment> & by_worker0,
                   std::vector<pipe_expert_assignment> & by_worker1) {
    by_worker0.clear();
    by_worker1.clear();
    const auto choose = [](int32_t expert_id) { return expert_id <= k_worker0_last ? 0 : 1; };
    if (plan_threads <= 1 || all.size() < 64) {
        for (const pipe_expert_assignment & a : all) {
            (choose(a.expert_id) == 0 ? by_worker0 : by_worker1).push_back(a);
        }
        return;
    }
    const size_t per_chunk = (all.size() + (size_t) plan_threads - 1) / (size_t) plan_threads;
    const size_t n_chunks  = (all.size() + per_chunk - 1) / per_chunk;
    std::vector<std::vector<pipe_expert_assignment>> chunk0(n_chunks), chunk1(n_chunks);
    pipe_parallel_for(all.size(), (size_t) plan_threads, [&](size_t a0, size_t a1) {
        const size_t chunk_idx = a0 / per_chunk;
        for (size_t i = a0; i < a1; ++i) {
            (choose(all[i].expert_id) == 0 ? chunk0[chunk_idx] : chunk1[chunk_idx]).push_back(all[i]);
        }
    });
    for (size_t c = 0; c < n_chunks; ++c) {
        for (pipe_expert_assignment & a : chunk0[c]) by_worker0.push_back(std::move(a));
        for (pipe_expert_assignment & a : chunk1[c]) by_worker1.push_back(std::move(a));
    }
}

struct trial_result {
    double   plan_ms   = 0.0;
    double   encode_ms = 0.0;
    std::string order_hash;   // hash of (worker0 expert-id sequence, worker1 expert-id sequence)
    std::string payload_hash; // hash of (worker0 encoded payload, worker1 encoded payload)
};

trial_result run_one_trial() {
    setenv("WP_EXPERT_WIRE", "ml8_4", 1);

    const std::vector<pipe_expert_assignment> all = build_synthetic_routing();
    const std::vector<float>                  acts = build_synthetic_activations();

    const int plan_threads = std::atoi(getenv("WP_EXPERT_PLAN_THREADS") ? getenv("WP_EXPERT_PLAN_THREADS") : "1");

    std::vector<pipe_expert_assignment> by_worker0, by_worker1;
    const auto plan_t0 = std::chrono::steady_clock::now();
    plan_classify(all, plan_threads, by_worker0, by_worker1);
    const auto plan_t1 = std::chrono::steady_clock::now();

    const uint64_t row_bytes = pipe_expert_wire_row_bytes(k_n_embd);
    std::vector<uint8_t> packed((size_t) k_n_tokens * row_bytes);

    const auto enc_t0 = std::chrono::steady_clock::now();
    pipe_expert_wire_pack_matrix(packed.data(), acts.data(), k_n_tokens, k_n_embd);
    std::vector<uint8_t> payload0 = pipe_encode_expert_dispatch_req_prepacked(
        k_layer, k_n_tokens, k_n_embd, by_worker0, k_clamp, packed.data(), packed.size());
    std::vector<uint8_t> payload1 = pipe_encode_expert_dispatch_req_prepacked(
        k_layer, k_n_tokens, k_n_embd, by_worker1, k_clamp, packed.data(), packed.size());
    const auto enc_t1 = std::chrono::steady_clock::now();

    sha256_t sh;
    unsigned char digest[32];
    sha256_init(&sh);
    for (const pipe_expert_assignment & a : by_worker0) {
        sha256_update(&sh, reinterpret_cast<const unsigned char *>(&a.expert_id), sizeof(a.expert_id));
    }
    for (const pipe_expert_assignment & a : by_worker1) {
        sha256_update(&sh, reinterpret_cast<const unsigned char *>(&a.expert_id), sizeof(a.expert_id));
    }
    sha256_final(&sh, digest);

    sha256_t sh2;
    unsigned char digest2[32];
    sha256_init(&sh2);
    sha256_update(&sh2, payload0.data(), payload0.size());
    sha256_update(&sh2, payload1.data(), payload1.size());
    sha256_final(&sh2, digest2);

    trial_result r;
    r.plan_ms      = std::chrono::duration<double, std::milli>(plan_t1 - plan_t0).count();
    r.encode_ms    = std::chrono::duration<double, std::milli>(enc_t1 - enc_t0).count();
    r.order_hash   = to_hex(digest);
    r.payload_hash = to_hex(digest2);
    return r;
}

// Re-exec argv[0] with WP_BENCH_CHILD=1 and the given thread counts (both
// knobs are latched once per process -- see file header -- so a fresh process
// is the only way to change them between trials). Child prints one line:
// "<plan_ms> <encode_ms> <order_hash> <payload_hash>".
bool run_child(const char * self_path, int threads, trial_result & out) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        setenv("WP_BENCH_CHILD", "1", 1);
        const std::string threads_str = std::to_string(threads);
        setenv("WP_EXPERT_PLAN_THREADS", threads_str.c_str(), 1);
        setenv("WP_EXPERT_ENCODE_THREADS", threads_str.c_str(), 1);
        execl(self_path, self_path, (char *) nullptr);
        _exit(127);
    }
    close(pipefd[1]);
    std::string buf;
    char tmp[4096];
    ssize_t n;
    while ((n = read(pipefd[0], tmp, sizeof(tmp))) > 0) {
        buf.append(tmp, (size_t) n);
    }
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "bench: child (threads=%d) exited abnormally\n", threads);
        return false;
    }
    double plan_ms = 0.0, encode_ms = 0.0;
    char order_hash[128] = {0};
    char payload_hash[128] = {0};
    if (std::sscanf(buf.c_str(), "%lf %lf %127s %127s", &plan_ms, &encode_ms, order_hash, payload_hash) != 4) {
        std::fprintf(stderr, "bench: could not parse child output: %s\n", buf.c_str());
        return false;
    }
    out.plan_ms      = plan_ms;
    out.encode_ms    = encode_ms;
    out.order_hash   = order_hash;
    out.payload_hash = payload_hash;
    return true;
}

}  // namespace

int main(int /* argc */, char ** argv) {
    if (getenv("WP_BENCH_CHILD") != nullptr) {
        const trial_result r = run_one_trial();
        std::printf("%.3f %.3f %s %s\n", r.plan_ms, r.encode_ms, r.order_hash.c_str(), r.payload_hash.c_str());
        return 0;
    }

    // Driver: default thread counts are small so `ctest` stays fast; set
    // WP_BENCH_FULL=1 to also try 16 and 24 (the numbers the profiling
    // report cites), which cost real wall time on a loaded box.
    std::vector<int> thread_counts = {1, 4};
    if (getenv("WP_BENCH_FULL") != nullptr) {
        thread_counts = {1, 4, 16, 24};
    }

    std::vector<trial_result> results;
    for (int nt : thread_counts) {
        trial_result r;
        if (!run_child(argv[0], nt, r)) {
            std::fprintf(stderr, "bench: trial threads=%d FAILED to run\n", nt);
            return 1;
        }
        results.push_back(r);
        std::printf("threads=%2d  plan=%8.3f ms  encode=%8.3f ms  total=%8.3f ms  order_hash=%s payload_hash=%s\n",
                    nt, r.plan_ms, r.encode_ms, r.plan_ms + r.encode_ms,
                    r.order_hash.c_str(), r.payload_hash.c_str());
    }

    // Byte-identity gate: every trial's order_hash and payload_hash must match
    // the threads=1 trial's. This is the correctness requirement from the
    // profiling task -- multithreading plan/encode must not change the wire
    // bytes a worker computes on.
    bool ok = true;
    for (size_t i = 1; i < results.size(); ++i) {
        if (results[i].order_hash != results[0].order_hash) {
            std::fprintf(stderr, "bench: FAIL order_hash mismatch at threads=%d (%s vs threads=1's %s)\n",
                        thread_counts[i], results[i].order_hash.c_str(), results[0].order_hash.c_str());
            ok = false;
        }
        if (results[i].payload_hash != results[0].payload_hash) {
            std::fprintf(stderr, "bench: FAIL payload_hash mismatch at threads=%d (%s vs threads=1's %s)\n",
                        thread_counts[i], results[i].payload_hash.c_str(), results[0].payload_hash.c_str());
            ok = false;
        }
    }
    if (ok) {
        std::printf("bench: byte-identity OK across thread counts {");
        for (size_t i = 0; i < thread_counts.size(); ++i) {
            std::printf("%s%d", i ? "," : "", thread_counts[i]);
        }
        std::printf("}\n");
    }
    return ok ? 0 : 1;
}
