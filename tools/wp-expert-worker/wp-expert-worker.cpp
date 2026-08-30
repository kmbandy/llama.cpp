#include "wp-expert-worker.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "pipe-expert-dispatcher.h"
#include "pipe-protocol.h"
#include "pipe-transport.h"
#include "weight-pager/wp-host-tier.h"

extern "C" {
#include "sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// poll() for the keepalive pump. POSIX only; the worker is already POSIX-only
// (O_DIRECT, posix_memalign) so this adds no new portability constraint.
#include <poll.h>            // ppoll needs _GNU_SOURCE, which glibc sets via -std=gnu++

#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
// WP_CPU_TIER_PIN: sched_setaffinity / sched_setscheduler for the ONE-SHOT pin
// of the CPU expert tier's executor thread. Needed because ggml only applies a
// threadpool's cpumask/prio from inside its `#pragma omp parallel`, which it
// skips entirely at n_threads == 1 -- see wp_cpu_tier_pin_self().
#  include <sched.h>
// WP_DISPATCH_DEDUP_ACTIVATIONS: POSIX shm (shm_open/mmap) for the
// same-machine activation rendezvous. Same portability tier as the O_DIRECT /
// posix_memalign / poll() usage already gated behind __linux__ above.
#  include <sys/mman.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

bool ggml_backend_cuda_wp_copy_stream_enabled(ggml_backend_t)
    __attribute__((weak));
bool ggml_backend_cuda_wp_copy_tensor_async(ggml_backend_t, ggml_tensor *,
                                                        const void *, size_t, size_t)
    __attribute__((weak));
bool ggml_backend_cuda_wp_copy_stream_record_event(ggml_backend_t,
                                                               ggml_backend_event_t)
    __attribute__((weak));
// WP_READER_H2D: reader-thread H2D on a dedicated non-blocking per-device
// stream (see the long comment on the definition in ggml-cuda.cu for why
// this exists instead of calling ggml_backend_tensor_set from a reader
// thread -- the MAD-114/gfx1201 legacy-stream capture hazard). Same
// weak-symbol pattern as the three declarations above: null on any build
// that doesn't link ggml-cuda's wp extensions (e.g. a pure-Vulkan worker),
// so every call site must check the pointer before calling.
bool ggml_backend_cuda_wp_reader_copy(ggml_backend_t, ggml_tensor *,
                                      const void *, size_t, size_t)
    __attribute__((weak));

namespace wp_expert_worker {

int parse_gather_min_tokens(const char * env) {
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return 2;
    }
    const int v = std::atoi(env);
    return v < 1 ? 1 : v;
}

bool parse_env_default_on(const char * env) {
    return env == nullptr || env[0] == '\0' || env[0] != '0';
}

bool parse_env_default_off(const char * env) {
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool wp_persistent_graphs_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("WP_PERSISTENT_GRAPHS");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

static bool wp_persistent_cuda_graphs_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("WP_PERSISTENT_CUDA_GRAPHS");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

static bool wp_hip_graphs_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("WP_HIP_GRAPHS");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return enabled || wp_persistent_cuda_graphs_enabled();
}

bool use_expert_gather(uint32_t n_tokens, bool force_dense, int min_tokens, bool gather_enabled) {
    return gather_enabled && !force_dense && (int64_t) n_tokens >= (int64_t) min_tokens;
}

CompactRouting compact_routing_rows(const std::vector<float> & wv) {
    CompactRouting out;
    out.idx.reserve(wv.size());
    out.weights.reserve(wv.size());
    for (size_t t = 0; t < wv.size(); ++t) {
        if (wv[t] != 0.0f) {
            out.idx.push_back((int32_t) t);
            out.weights.push_back(wv[t]);
        }
    }
    if (out.idx.empty()) {
        // Same dummy as compute_batch: skip-the-expert is wrong if every
        // selected expert is empty — sum stays nullptr and ggml_cpy blows up.
        out.idx.push_back(0);
        out.weights.push_back(0.0f);
    }
    return out;
}

ggml_tensor * scatter_add_compact_rows(
        struct ggml_context * ctx,
        struct ggml_tensor * dest,
        struct ggml_tensor * compact,
        struct ggml_tensor * idx) {
    // dest[idx] += compact. get_rows / add / set_rows are all O(n_sel).
    // dest is the io result (already allocated) so gallocr does not grow a
    // second [n_embd, n_tokens] workspace — that 127 MiB cudaMalloc is what
    // OOM'd the 1070 on a 2048-wide prefill.
    ggml_tensor * prev = ggml_get_rows(ctx, dest, idx);
    ggml_tensor * acc  = ggml_add(ctx, prev, compact);
    return ggml_set_rows(ctx, dest, acc, idx);
}

ggml_tensor * scatter_compact_rows(
        struct ggml_context * ctx,
        struct ggml_tensor * compact,
        struct ggml_tensor * idx,
        struct ggml_tensor * full_shape) {
    // Standalone zeros dest, for the byte-match test vs get_rows_back.
    ggml_tensor * dest = ggml_scale(ctx, full_shape, 0.0f);
    return ggml_set_rows(ctx, dest, compact, idx);
}

static constexpr uint64_t DEFAULT_STAGING_BUFFERS = 16;
static constexpr size_t DIRECT_ALIGNMENT = 4096;

// WP_EXPERT_STAGING_BUFFERS=<n> changes the staging depth only when the
// caller did not set --host-budget.  Default-off: an unset, empty, zero, or
// invalid value retains the historical 16-buffer host-budget default.  An
// explicit --host-budget remains an upper bound on the number of whole-page
// buffers; this knob must not silently allocate beyond an operator's budget.
uint64_t staging_buffers_from_env() {
    const char * env = std::getenv("WP_EXPERT_STAGING_BUFFERS");
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return DEFAULT_STAGING_BUFFERS;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' || parsed == 0 ||
            parsed > std::numeric_limits<size_t>::max()) {
        return DEFAULT_STAGING_BUFFERS;
    }
    return (uint64_t) parsed;
}

static size_t read_inflight_from_env() {
    const char * env = std::getenv("WP_READ_INFLIGHT");
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return 0;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' || parsed == 0 ||
            parsed > std::numeric_limits<size_t>::max()) {
        return 0;
    }
    return (size_t) parsed;
}

static size_t read_chunk_bytes_from_env() {
    const char * env = std::getenv("WP_READ_CHUNK_BYTES");
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return 0;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' || parsed < DIRECT_ALIGNMENT ||
            parsed > std::numeric_limits<size_t>::max()) {
        return 0;
    }
    const size_t chunk = (size_t) parsed;
    return chunk / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
}

static bool read_direct_from_env() {
    const char * env = std::getenv("WP_READ_DIRECT");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

static size_t read_workers_from_env() {
    const char * env = std::getenv("WP_EXPERT_READ_WORKERS");
    const long parsed =
        (env != nullptr && env[0] != '\0') ? std::strtol(env, nullptr, 10) : 0;
    return parsed > 0 ? (size_t) parsed : (size_t) 4;
}

// WP_READ_STATS_INTERVAL_MS controls the live read-counter dump. Default 5000;
// zero disables read timing and counters.
static uint64_t read_stats_interval_ms_from_env() {
    static constexpr uint64_t default_interval_ms = 5000;
    const char * env = std::getenv("WP_READ_STATS_INTERVAL_MS");
    if (env == nullptr || env[0] == '\0') {
        return default_interval_ms;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' ||
            parsed > std::numeric_limits<uint64_t>::max() / 1000000ull) {
        return default_interval_ms;
    }
    return (uint64_t) parsed;
}

// WP_EXPERT_LFU_PLACEMENT controls frequency-aware multi-device placement.
// DEFAULT OFF, deliberately. This policy has never been measured on hardware,
// and an unmeasured lever that is on by default stops a bare run from being the
// config of record -- every subsequent comparison would silently include it.
// That exact trap (a harness whose defaults were not the record) already cost
// this project a day of invalid A/Bs. Turn it on explicitly, as the ONE
// variable, and flip this default once it has numbers.
// WP_EXPERT_LFU_PLACEMENT=1 enables it; anything else keeps the static map.
static bool lfu_placement_from_env() {
    const char * env = std::getenv("WP_EXPERT_LFU_PLACEMENT");
    return env != nullptr && env[0] == '1';
}

// Maximum successful page migrations per dispatch request.
static size_t lfu_migration_cap_from_env() {
    const char * env = std::getenv("WP_EXPERT_LFU_MIGRATION_CAP");
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return 2;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' ||
            parsed > std::numeric_limits<size_t>::max()) {
        return 2;
    }
    return (size_t) parsed;
}

// Percentage of the boundary count used by the LFU placement hysteresis.
static uint64_t lfu_hysteresis_pct_from_env() {
    const char * env = std::getenv("WP_EXPERT_LFU_HYSTERESIS_PCT");
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return 25;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' || parsed == 0) {
        return 25;
    }
    return std::min<uint64_t>(parsed, 100);
}

static uint64_t placement_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool cpu_on_arrival_enabled_from_env() {
    const char * env = std::getenv("WP_EXPERT_CPU_ON_ARRIVAL");
    return env != nullptr && env[0] == '1';
}

static size_t cpu_on_arrival_cap_from_env() {
    const char * env = std::getenv("WP_EXPERT_CPU_ON_ARRIVAL_MAX");
    if (env == nullptr || env[0] == '\0' || env[0] == '-') {
        return 2;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (errno == ERANGE || end == env || *end != '\0' ||
            parsed > std::numeric_limits<size_t>::max()) {
        return 2;
    }
    return (size_t) parsed;
}

static std::atomic<bool> g_read_direct_fallback{false};

class ReadPathStats {
public:
    ReadPathStats() :
        interval_ms_(read_stats_interval_ms_from_env()),
        interval_ns_(interval_ms_ * 1000000ull),
        next_report_ns_(interval_ns_ == 0 ? std::numeric_limits<uint64_t>::max() :
                                             read_now_ns()) {
        for (auto & bucket : latency_buckets_) {
            bucket.store(0, std::memory_order_relaxed);
        }
    }

    bool enabled() const {
        return interval_ns_ != 0;
    }

    uint64_t interval_ms() const {
        return interval_ms_;
    }

    void record(size_t bytes, uint64_t ns, uint64_t started_ns, uint64_t finished_ns) {
        if (!enabled()) {
            return;
        }
        bytes_.fetch_add((uint64_t) bytes, std::memory_order_relaxed);
        ns_.fetch_add(ns, std::memory_order_relaxed);
        update_min(first_read_ns_, started_ns);
        update_max(last_read_ns_, finished_ns);
        update_min(min_ns_, ns);
        update_max(max_ns_, ns);
        latency_buckets_[latency_bucket(ns)].fetch_add(1, std::memory_order_relaxed);
        n_reads_.fetch_add(1, std::memory_order_relaxed);
        maybe_report();
    }

    void record_cpu_on_arrival(uint64_t ns) {
        if (!enabled()) {
            return;
        }
        n_cpu_on_arrival_.fetch_add(1, std::memory_order_relaxed);
        ns_cpu_on_arrival_.fetch_add(ns, std::memory_order_relaxed);
        maybe_report();
    }

    void record_cpu_on_arrival_fallback(uint64_t count) {
        if (!enabled() || count == 0) {
            return;
        }
        n_cpu_on_arrival_fallback_.fetch_add(count, std::memory_order_relaxed);
        maybe_report();
    }

private:
    static constexpr size_t latency_bucket_count = 64;

    using clock = std::chrono::steady_clock;

    static uint64_t read_now_ns() {
        return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock::now().time_since_epoch()).count();
    }

    static size_t latency_bucket(uint64_t ns) {
        size_t bucket = 0;
        while (ns > 1 && bucket + 1 < latency_bucket_count) {
            ns >>= 1;
            ++bucket;
        }
        return bucket;
    }

    static void update_min(std::atomic<uint64_t> & value, uint64_t candidate) {
        uint64_t current = value.load(std::memory_order_relaxed);
        while (candidate < current &&
               !value.compare_exchange_weak(
                   current, candidate, std::memory_order_relaxed)) {
        }
    }

    static void update_max(std::atomic<uint64_t> & value, uint64_t candidate) {
        uint64_t current = value.load(std::memory_order_relaxed);
        while (candidate > current &&
               !value.compare_exchange_weak(
                   current, candidate, std::memory_order_relaxed)) {
        }
    }

    uint64_t percentile_ns(uint64_t rank) const {
        uint64_t seen = 0;
        for (size_t i = 0; i < latency_bucket_count; ++i) {
            seen += latency_buckets_[i].load(std::memory_order_relaxed);
            if (seen > rank) {
                return i + 1 < 64 ? (1ull << (i + 1)) - 1 : UINT64_MAX;
            }
        }
        return max_ns_.load(std::memory_order_relaxed);
    }

    void maybe_report() {
        const uint64_t now = read_now_ns();
        uint64_t next = next_report_ns_.load(std::memory_order_relaxed);
        if (now < next || !next_report_ns_.compare_exchange_strong(
                next, now > UINT64_MAX - interval_ns_ ? UINT64_MAX :
                    now + interval_ns_, std::memory_order_relaxed)) {
            return;
        }
        report();
    }

    void report() const {
        const uint64_t n = n_reads_.load(std::memory_order_relaxed);
        if (n == 0) {
            return;
        }
        const uint64_t ns = ns_.load(std::memory_order_relaxed);
        const uint64_t bytes = bytes_.load(std::memory_order_relaxed);
        const uint64_t first_ns = first_read_ns_.load(std::memory_order_relaxed);
        const uint64_t last_ns = last_read_ns_.load(std::memory_order_relaxed);
        const uint64_t wall_ns = last_ns >= first_ns ? last_ns - first_ns : 0;
        const uint64_t min_ns = min_ns_.load(std::memory_order_relaxed);
        const uint64_t max_ns = max_ns_.load(std::memory_order_relaxed);
        const uint64_t p50_ns = percentile_ns((n - 1) * 50 / 100);
        const uint64_t p95_ns = percentile_ns((n - 1) * 95 / 100);
        const double bandwidth_gb_s = wall_ns != 0 ? (double) bytes / (double) wall_ns : 0.0;
        std::fprintf(stderr,
                     "wp expert worker read stats n_reads=%llu bytes=%llu "
                     "read_ns=%llu read_wall_ns=%llu bandwidth_gb_s=%.3f mean_us=%.3f "
                     "min_us=%.3f p50_us=%.3f p95_us=%.3f max_us=%.3f "
                     "direct_fallback=%d\n",
                     (unsigned long long) n,
                     (unsigned long long) bytes,
                     (unsigned long long) ns,
                     (unsigned long long) wall_ns,
                     bandwidth_gb_s,
                     (double) ns / (double) n / 1000.0,
                     (double) min_ns / 1000.0,
                     (double) p50_ns / 1000.0,
                     (double) p95_ns / 1000.0,
                     (double) max_ns / 1000.0,
                     g_read_direct_fallback.load(std::memory_order_relaxed) ? 1 : 0);
        std::fprintf(stderr,
                     "wp expert worker cpu-on-arrival n_experts=%llu "
                     "cpu_ns=%llu cap_fallback=%llu\n",
                     (unsigned long long) n_cpu_on_arrival_.load(std::memory_order_relaxed),
                     (unsigned long long) ns_cpu_on_arrival_.load(std::memory_order_relaxed),
                     (unsigned long long) n_cpu_on_arrival_fallback_.load(
                         std::memory_order_relaxed));
        std::fflush(stderr);
    }

    const uint64_t interval_ms_;
    const uint64_t interval_ns_;
    std::atomic<uint64_t> next_report_ns_;
    std::atomic<uint64_t> n_reads_{0};
    std::atomic<uint64_t> bytes_{0};
    std::atomic<uint64_t> ns_{0};
    std::atomic<uint64_t> first_read_ns_{UINT64_MAX};
    std::atomic<uint64_t> last_read_ns_{0};
    std::atomic<uint64_t> min_ns_{UINT64_MAX};
    std::atomic<uint64_t> max_ns_{0};
    std::atomic<uint64_t> n_cpu_on_arrival_{0};
    std::atomic<uint64_t> ns_cpu_on_arrival_{0};
    std::atomic<uint64_t> n_cpu_on_arrival_fallback_{0};
    std::array<std::atomic<uint64_t>, latency_bucket_count> latency_buckets_{};
};

static ReadPathStats g_read_path_stats;

struct RequestStats {
    uint64_t ns_lookup  = 0;
    uint64_t ns_read    = 0;
    uint64_t ns_compute = 0;
    // Worker::dispatch() entry -> return, ALL exits (RAII). ns_compute starts
    // only at compute_started, so everything before it (lookup, ensure_batch,
    // page-in issue) and after it (response build) was invisible. Added
    // 2026-08-29 to close the ~890 us/layer the spine waits but the worker
    // does not account for.
    uint64_t ns_dispatch_total = 0;
    // The serve_connection segments OUTSIDE dispatch(). Measured 2026-08-29:
    // ns_dispatch_total (834 us) lands on ns_compute (819), yet frame residency
    // is ~1809 us against a spine wait of 1680 -- so ~975 us/layer is in this
    // loop and NOT in dispatch(). These five split it. Pre-send segments are the
    // only ones that can explain the spine's wait; ns_post_send cannot.
    uint64_t ns_lock_wait   = 0;   // recv done -> g_worker_gpu_mutex held
    uint64_t ns_decode_req  = 0;   // pipe_decode_expert_dispatch_req
    uint64_t ns_pre_dispatch= 0;   // ref_log + log_reference, up to dispatch()
    uint64_t ns_encode_send = 0;   // encode + unlock + send + relock
    uint64_t ns_post_send   = 0;   // record_stats, req_log, spec_pagein_after_dispatch
    uint64_t ns_send    = 0;
    // *** THE PRE-RECV BLIND SPOT (added 2026-08-29). ***
    // The five segments above all begin at t_frame, which is stamped AFTER
    // pipe_recv_frame returns -- i.e. after the entire body has landed. The
    // request-carrying frame on the live split path (ACTS) carries the full
    // activation block, so "waiting for the body to arrive" is a real,
    // potentially large interval that sits inside the spine's wait and outside
    // every worker counter. ns_recv is the counter that has printed
    // "unavailable" forever; these three replace it with measured segments,
    // recorded on the ACTS-family branches (the live path) as well as the plain
    // DISPATCH_REQ one.
    //
    //   ns_recv_body  header recv returned -> pipe_recv_frame returned.
    //                 Body transfer + the payload.resize() allocation. Does NOT
    //                 include time the frame sat in the socket buffer before
    //                 this thread reached the recv (that is upstream of the
    //                 first stamp and still unmeasured).
    //   ns_req_decode the request decode for THIS frame -- ACTS/ACTS_REF/
    //                 ACTS_PUBLISH payload decode, or the DISPATCH_REQ decode.
    //                 Distinct from ns_decode_req, which is only ever set on
    //                 the plain DISPATCH_REQ branch and reads 0 on the split
    //                 path that is actually live.
    //   ns_resp_send  encode + unlock + send + relock, on every response-
    //                 bearing branch. The split branches previously folded this
    //                 into ns_send with no separate encode/send split; this is
    //                 the same window, named for the direction it measures, and
    //                 is the worker->spine leg the LAN coalesce is aimed at.
    uint64_t ns_recv_body  = 0;
    uint64_t ns_req_decode = 0;
    uint64_t ns_resp_send  = 0;
    uint64_t n_resident      = 0;
    uint64_t n_pagein     = 0;
    uint64_t n_host_hit = 0;
    uint64_t n_host_demote = 0;
    uint64_t bytes_read = 0;
    uint64_t n_pagein_reserved = 0;
    uint64_t n_pagein_general = 0;
    uint64_t ns_host_get = 0;
    uint64_t host_bytes = 0;
    uint64_t n_graph_submits = 0;
    uint64_t n_device_allocs = 0;
    uint64_t ns_graph_build = 0;
    uint64_t ns_submit = 0;
    uint64_t ns_final_sync = 0;
    uint64_t ns_readback = 0;
    // Vulkan-only nested timers. ns_vk_compute_path covers compute_batch;
    // the other timers identify work inside that span and may overlap it.
    uint64_t ns_vk_compute_path = 0;
    uint64_t ns_vk_dispatch_path = 0;
    uint64_t ns_vk_wait = 0;
    uint64_t ns_vk_cache_lookup = 0;
    uint64_t ns_vk_graph_compute = 0;
    uint64_t ns_vk_params_set = 0;
    uint64_t ns_vk_fold = 0;
    uint64_t ns_vk_sync = 0;
    uint64_t ns_vk_readback = 0;
    // *** ACCOUNTING TIMERS ADDED 2026-08-29 (Vulkan decode gap). ***
    // The 51cae31f9 banner explained only ~64% of the RX 480's ns_compute
    // (dispatch 839 us; graph_compute 274 + readback 62 + cache_lookup 1 the
    // only named costs). These five close the two holes: the untimed
    // dispatch() prologue between ns_compute's start and the first lap(), and
    // the untimed head/tail of compute_batch's D2 fast path.
    //
    // ns_prologue is recorded on EVERY backend (it is a plain host cost that
    // the 1070 pays too and the banner never showed); the ns_vk_* ones stay
    // Vulkan-gated like their neighbours.
    uint64_t ns_prologue = 0;      // ns_compute start -> the phase-lap origin
    uint64_t ns_arena_probe = 0;   // arena_id_eligible()/grouped_gemv_eligible()
    uint64_t ns_vk_arena_probe = 0;
    uint64_t ns_vk_setup = 0;      // compute_batch entry -> D2 cache lookup
    uint64_t ns_vk_rebind = 0;     // D2 hit: attach_weight + routing repack
    uint64_t ns_vk_layer_ahead = 0;// submit_prefill_layer_ahead
    uint64_t ns_prep = 0;
    uint64_t ns_prep_setup = 0;   // ggml_init + new_tensor + buft queries
    uint64_t ns_prep_grow = 0;    // grow_io_buffer (device alloc when it grows)
    uint64_t ns_prep_attach = 0;  // attach_weight -> buffer_init_tensor
    uint64_t ns_prep_set = 0;     // the activation upload itself          // prepare_io: activation upload + io buffer growth
    uint64_t ns_hits = 0;          // compute_batch(hits) end to end
    uint64_t ns_wait = 0;          // batch.complete(): reader-thread join + I/O + H2D
    uint64_t ns_pagein_compute = 0;  // compute_batch(pageins) end to end
    uint64_t ns_result = 0;        // read_result
    uint64_t ns_encode = 0;        // fp32 -> fp16 of the reply
    uint64_t ns_h2d = 0;
    uint64_t bytes_h2d = 0;
    uint64_t n_reader_h2d = 0;  // WP_READER_H2D: pages uploaded by a reader thread
    // ROUTING DENSITY (2026-08-04). compute_batch runs the FULL FFN for every
    // assigned expert over ALL request.n_tokens and then multiplies by a
    // per-token router weight that is ZERO for tokens not routed to that expert
    // (pipe-protocol.h: "one final router weight per token"). So the useful
    // fraction of the expert FLOPs is exactly the nonzero fraction of those
    // weights. Counting it directly rather than inferring it from
    // n_expert_used/n_expert, because the assignment list only contains experts
    // that got at least one token, which biases the naive estimate.
    uint64_t n_weight_nonzero = 0;   // token-expert pairs actually routed
    uint64_t n_weight_total   = 0;   // token-expert pairs actually COMPUTED
    // D2 (2026-08-07): shape-keyed graph cache traffic.
    uint64_t n_gcache_hit  = 0;
    uint64_t n_gcache_miss = 0;
    uint64_t n_arena_hit   = 0;
    uint64_t n_arena_groups = 0;
    uint64_t n_arena_build = 0;
    uint64_t n_hipgraph_capture = 0;
    uint64_t n_hipgraph_replay  = 0;
    uint64_t n_d3_collapse = 0;
    uint64_t n_d3_typed    = 0;
    uint64_t n_d3_bounce   = 0;
    bool d3_counted = false;
    // D1 (2026-08-07): the coalesced routing-weight/gather-idx blob upload.
    // The per-tensor uploads it replaces were never timed anywhere -- they sat
    // in the gap between ns_graph_build and ns_submit (the "accounting hole"),
    // so this counter is NEW time made visible, not time moved from another
    // column. Zero when WP_EXPERT_PARAMS_COALESCE is off.
    uint64_t ns_params_set = 0;
    // Time inside ensure_batch AFTER ns_lookup is snapshotted: sync host-victim
    // demote D2H + host-tier restore + reader-thread spawn. This is the gap
    // that made 40s walls look unaccounted on 1547/1752 (lookup ends before
    // demote_slot). ns_demote is the D2H subset; ns_host_get is the restore.
    uint64_t ns_demote = 0;
    uint64_t ns_ensure_post = 0;
    uint64_t n_read_inflight_max = 0;
    uint64_t ns_read_issue = 0;
    uint64_t ns_read_complete = 0;
    uint64_t n_cpu_on_arrival = 0;
    uint64_t ns_cpu_on_arrival = 0;
    uint64_t n_cpu_on_arrival_fallback = 0;
};

// Forward declarations: the probe itself is defined further down, next to
// run_self_bench, but WorkerStats::report() needs to read it.
void self_bench_tick(ggml_backend_t backend);
bool self_bench_stats(uint64_t & n, uint64_t & min_us, uint64_t & mean_us);

// WP_WORKER_MULTI_CONN=N (N>=2) -- lever-queue item #9, "worker
// double-buffering". Declared up here (rather than next to serve_connection,
// where the reasoning for it and its sibling g_worker_gpu_mutex live) purely
// because WorkerStats::report() below needs to see this vector; see the
// long comment above serve_connection() for the full design.
//
// Per-connection live request counters, purely for the WP_WORKER_STATS
// report line. Sized to multi_conn_n by run() BEFORE any connection thread
// starts and never resized after -- each connection thread only ever
// increments the ONE cell at its own index (std::thread's constructor
// happens-before covers publishing the sized vector to every thread), so
// the increments themselves need no lock, and report()'s read of each cell
// is an independent atomic load -- safe regardless of what any other
// thread's increment is doing concurrently.
std::vector<std::atomic<uint64_t>> g_worker_conn_request_counts;

// Per-connection outstanding STAGING LEASE counts, for the WP_WORKER_STATS
// report line only -- see the StagingPool per-connection quota (the
// 2026-08-25 deadlock fix comment on StagingPool) for what actually enforces
// the cap. Sized identically to g_worker_conn_request_counts, same
// construction-happens-before-any-thread argument applies: each StagingPool
// borrow()/release() only ever touches the ONE cell for its own conn_index,
// so no lock is needed for either the increments/decrements or report()'s
// read. This is exactly the diagnostic that would have made the 2026-08-25
// wedge visible at a glance (one connection's staging_held pinned at its
// quota while the other sits at 0, instead of two threads just... not
// making progress).
std::vector<std::atomic<int64_t>> g_worker_staging_held;

// Fires once per BEGIN-frame ensure_batch() call that took the unlocked
// read-issue path added 2026-08-25 (see the "READ-ISSUE UNLOCKED FROM
// g_worker_gpu_mutex" comment in ExpertSlotPool::ensure_batch): planning
// (slot hit/victim resolution, pins, host-tier fill) ran under
// g_worker_gpu_mutex as before, but spawning the page-in reader threads and
// notifying them to start ran with the mutex released, so the other
// connection's compute is not blocked waiting for this connection's reads
// to be issued. Global, not per-connection: this is a "did the path fire at
// all" counter for the controller, not a per-stream breakdown like
// conn_reqs. Stays 0 forever outside multi-conn mode (the unlock is gated
// on gpu_lock->owns_lock(), which is never true there).
std::atomic<uint64_t> g_worker_n_begin_unlocked_reads{0};

// WP_READER_H2D_VERIFY=1 tripwire counter: mismatches found by
// tensor_verify_page_range, incremented from reader threads (concurrent,
// hence atomic) after a WP_READER_H2D upload. 0 whenever the verify knob
// is off, same as every other WP_* diagnostic.
std::atomic<uint64_t> g_worker_n_reader_h2d_verify_fail{0};

class WorkerStats {
public:
    WorkerStats() :
        enabled_(std::getenv("WP_WORKER_STATS") != nullptr &&
                 std::strcmp(std::getenv("WP_WORKER_STATS"), "1") == 0),
        next_report_(clock::now() + std::chrono::seconds(5)) {
    }

    // Without this the multi-device worker emits two IDENTICAL stats lines per
    // machine (one per DeviceWorker) with nothing to tell ROCm0 from ROCm1 --
    // which makes the per-device balance question unanswerable from the logs.
    void set_device(const std::string & d) { device_ = d; }

    bool enabled() const {
        return enabled_;
    }

    // Set once at startup from ExpertSlotPool::staging_kind() -- which
    // allocation path the staging pool actually used, not what was requested.
    void set_probe_backend(ggml_backend_t b) { probe_backend_ = b; }

    void set_staging_kind(const char * kind) {
        staging_kind_ = kind;
    }

    void set_shield_stats(uint64_t hits, uint64_t exhausted) {
        n_shield_hits_      = hits;
        n_shield_exhausted_ = exhausted;
    }

    void set_layerahead_stats(uint64_t hints, uint64_t pageins, uint64_t hits) {
        n_layerahead_hints_   = hints;
        n_layerahead_pageins_ = pageins;
        n_layerahead_hits_    = hits;
    }

    void set_pin_stats(size_t n_pinned, uint64_t demand_hits) {
        n_pinned_ = n_pinned;
        n_pinned_demand_hits_ = demand_hits;
    }

    ~WorkerStats() {
        report();
    }

    void record(const RequestStats & request, size_t n_experts) {
        if (!enabled_) {
            return;
        }

        ns_lookup_ += request.ns_lookup;
        ns_read_ += request.ns_read;
        ns_compute_ += request.ns_compute;
        ns_send_ += request.ns_send;
        n_resident_ += request.n_resident;
        n_pagein_ += request.n_pagein;
        n_pagein_reserved_ += request.n_pagein_reserved;
        n_pagein_general_ += request.n_pagein_general;
        n_host_hit_ += request.n_host_hit;
        n_host_demote_ += request.n_host_demote;
        bytes_read_ += request.bytes_read;
        ns_host_get_ += request.ns_host_get;
        host_bytes_ = request.host_bytes;
        n_graph_submits_ += request.n_graph_submits;
        n_device_allocs_ += request.n_device_allocs;
        ns_graph_build_ += request.ns_graph_build;
        ns_submit_ += request.ns_submit;
        ns_final_sync_ += request.ns_final_sync;
        ns_vk_compute_path_ += request.ns_vk_compute_path;
        ns_vk_dispatch_path_ += request.ns_vk_dispatch_path;
        ns_vk_wait_ += request.ns_vk_wait;
        ns_vk_cache_lookup_ += request.ns_vk_cache_lookup;
        ns_vk_graph_compute_ += request.ns_vk_graph_compute;
        ns_vk_params_set_ += request.ns_vk_params_set;
        ns_vk_fold_ += request.ns_vk_fold;
        ns_vk_sync_ += request.ns_vk_sync;
        ns_vk_readback_ += request.ns_vk_readback;
        ns_prologue_ += request.ns_prologue;
        ns_arena_probe_ += request.ns_arena_probe;
        ns_vk_arena_probe_ += request.ns_vk_arena_probe;
        ns_vk_setup_ += request.ns_vk_setup;
        ns_vk_rebind_ += request.ns_vk_rebind;
        ns_vk_layer_ahead_ += request.ns_vk_layer_ahead;
        n_gcache_hit_ += request.n_gcache_hit;
        n_gcache_miss_ += request.n_gcache_miss;
        n_arena_hit_ += request.n_arena_hit;
        n_arena_groups_ += request.n_arena_groups;
        n_arena_build_ += request.n_arena_build;
        n_hipgraph_capture_ += request.n_hipgraph_capture;
        n_hipgraph_replay_ += request.n_hipgraph_replay;
        n_d3_collapse_ += request.n_d3_collapse;
        n_d3_typed_ += request.n_d3_typed;
        n_d3_bounce_ += request.n_d3_bounce;
        // PER-REQUEST DISTRIBUTION, not just the total. The cumulative ns_submit
        // cannot distinguish "every request costs 2.1 ms" from "most cost 0.2 ms
        // and a few cost 50 ms", and those have completely different fixes. An
        // isolated rebuild of this exact graph at this exact shape measures
        // ~190 us/expert on Vulkan0 and ~150 on CUDA, against ~1510 and ~254
        // in the worker -- so the shape of this histogram is the open question.
        if (request.n_graph_submits > 0) {
            const uint64_t per = request.ns_submit / request.n_graph_submits;
            if (per < submit_min_ns_) submit_min_ns_ = per;
            if (per > submit_max_ns_) submit_max_ns_ = per;
            // log2-ish buckets in us: <125, <250, <500, <1k, <2k, <4k, <8k, >=8k
            size_t b = 0;
            for (uint64_t edge = 125000; b < 7 && per >= edge; edge *= 2) ++b;
            ++submit_bucket_[b];
        }
        ns_dispatch_total_ += request.ns_dispatch_total;
        ns_lock_wait_    += request.ns_lock_wait;
        ns_decode_req_   += request.ns_decode_req;
        ns_pre_dispatch_ += request.ns_pre_dispatch;
        ns_encode_send_  += request.ns_encode_send;
        ns_post_send_    += request.ns_post_send;
        ns_recv_body_    += request.ns_recv_body;
        ns_req_decode_   += request.ns_req_decode;
        ns_resp_send_    += request.ns_resp_send;
        ns_readback_ += request.ns_readback;
        ns_prep_ += request.ns_prep;
        ns_prep_setup_ += request.ns_prep_setup;
        ns_prep_grow_ += request.ns_prep_grow;
        ns_prep_attach_ += request.ns_prep_attach;
        ns_prep_set_ += request.ns_prep_set;
        ns_hits_ += request.ns_hits;
        ns_wait_ += request.ns_wait;
        ns_pagein_compute_ += request.ns_pagein_compute;
        ns_result_ += request.ns_result;
        ns_encode_ += request.ns_encode;
        n_weight_nonzero_ += request.n_weight_nonzero;
        n_weight_total_   += request.n_weight_total;
        ns_h2d_ += request.ns_h2d;
        bytes_h2d_ += request.bytes_h2d;
        n_reader_h2d_ += request.n_reader_h2d;
        ns_demote_ += request.ns_demote;
        ns_ensure_post_ += request.ns_ensure_post;
        n_read_inflight_max_ = std::max(n_read_inflight_max_, request.n_read_inflight_max);
        ns_read_issue_ += request.ns_read_issue;
        ns_read_complete_ += request.ns_read_complete;
        n_cpu_on_arrival_ += request.n_cpu_on_arrival;
        ns_cpu_on_arrival_ += request.ns_cpu_on_arrival;
        n_cpu_on_arrival_fallback_ += request.n_cpu_on_arrival_fallback;
        ++n_requests_;
        n_experts_ += n_experts;

        // WP_SELF_BENCH_EVERY=N: re-time the static graph every N requests while
        // serving. Costs one extra compute per N requests; default off.
        if (self_bench_every_ > 0 && probe_backend_ != nullptr &&
            n_requests_ % self_bench_every_ == 0) {
            self_bench_tick(probe_backend_);
        }

        const clock::time_point now = clock::now();
        if (now < next_report_) {
            return;
        }
        next_report_ = now + std::chrono::seconds(5);
        report();
    }

    void record_wire(const RequestStats & request) {
        if (!enabled_) {
            return;
        }

        ns_send_ += request.ns_send;
        ns_recv_body_ += request.ns_recv_body;
        ns_req_decode_ += request.ns_req_decode;
        ns_resp_send_ += request.ns_resp_send;
    }

private:
    using clock = std::chrono::steady_clock;

    void report() const {
        if (!enabled_ || n_requests_ == 0) {
            return;
        }
        // *** SERIALISE THE WHOLE BANNER. ***
        // It is dozens of separate << calls into std::cout, and under the
        // default-on WP_DEVICE_PARALLEL path device threads report
        // CONCURRENTLY. On 2026-08-29 two banners interleaved into one torn
        // line:
        //   device=ROCm0 n_requests=12865 ... device=ROCm1 n_requests=0 ...
        // The torn device parsed as n_requests=0, the analysis computed a
        // non-positive delta for it and SILENTLY DROPPED that device, and I
        // then drew conclusions from a two-device view of a three-device
        // worker. A corrupt instrument is worse than none: it fails quietly and
        // still looks like data. Function-local static => one mutex shared by
        // every WorkerStats instance, which is the scope required.
        static std::mutex report_mu;
        std::lock_guard<std::mutex> report_lock(report_mu);
        std::cout << "wp expert worker stats"
                  << " device=" << (device_.empty() ? std::string("?") : device_)
                  << " n_requests=" << n_requests_
                  << " n_experts=" << n_experts_
                  << " n_resident=" << n_resident_
                  << " n_pagein=" << n_pagein_
                  << " n_pagein_reserved=" << n_pagein_reserved_
                  << " n_pagein_general=" << n_pagein_general_
                  << " n_shield_hits=" << n_shield_hits_
                  << " n_shield_exhausted=" << n_shield_exhausted_
                  << " n_layerahead_hints=" << n_layerahead_hints_
                  << " n_layerahead_pageins=" << n_layerahead_pageins_
                  << " n_layerahead_hits=" << n_layerahead_hits_
                  << " n_pinned=" << n_pinned_
                  << " n_pinned_demand_hits=" << n_pinned_demand_hits_
                  << " n_host_hit=" << n_host_hit_
                  << " n_host_demote=" << n_host_demote_
                  << " bytes_read=" << bytes_read_
                  << " ns_recv=unavailable"
                  << " ns_lookup=" << ns_lookup_
                  << " ns_read=" << ns_read_
                  << " ns_read_issue=" << ns_read_issue_
                  << " ns_read_complete=" << ns_read_complete_
                  << " n_read_inflight_max=" << n_read_inflight_max_
                  << " n_cpu_on_arrival=" << n_cpu_on_arrival_
                  << " ns_cpu_on_arrival=" << ns_cpu_on_arrival_
                  << " n_cpu_on_arrival_fallback=" << n_cpu_on_arrival_fallback_
                  << " read_bytes_per_s=" << (ns_read_complete_ == 0 ? 0.0 :
                        (double) bytes_read_ * 1000000000.0 / (double) ns_read_complete_)
                  << " ns_h2d=" << ns_h2d_
                  << " bytes_h2d=" << bytes_h2d_
                  << " gb_s_h2d=" << (ns_h2d_ == 0 ? 0.0 :
                        (double) bytes_h2d_ / (double) ns_h2d_)
                  << " n_reader_h2d=" << n_reader_h2d_
                  << " staging_kind=" << staging_kind_
                  << " ns_host_get=" << ns_host_get_
                  << " ns_demote=" << ns_demote_
                  << " ns_ensure_post=" << ns_ensure_post_
                  << " ns_compute=" << ns_compute_
                  << " ns_dispatch_total=" << ns_dispatch_total_
                  << " ns_lock_wait=" << ns_lock_wait_
                  << " ns_decode_req=" << ns_decode_req_
                  << " ns_pre_dispatch=" << ns_pre_dispatch_
                  << " ns_encode_send=" << ns_encode_send_
                  << " ns_post_send=" << ns_post_send_
                  << " ns_recv_body=" << ns_recv_body_
                  << " ns_req_decode=" << ns_req_decode_
                  << " ns_resp_send=" << ns_resp_send_
                  << " n_graph_submits=" << n_graph_submits_
                  << " n_device_allocs=" << n_device_allocs_
                  << " ns_graph_build=" << ns_graph_build_
                  << " ns_submit=" << ns_submit_
                  << " ns_final_sync=" << ns_final_sync_
                  << " ns_vk_compute_path=" << ns_vk_compute_path_
                  << " ns_vk_dispatch_path=" << ns_vk_dispatch_path_
                  << " ns_vk_wait=" << ns_vk_wait_
                  << " ns_vk_cache_lookup=" << ns_vk_cache_lookup_
                  << " ns_vk_graph_compute=" << ns_vk_graph_compute_
                  << " ns_vk_params_set=" << ns_vk_params_set_
                  << " ns_vk_fold=" << ns_vk_fold_
                  << " ns_vk_sync=" << ns_vk_sync_
                  << " ns_vk_readback=" << ns_vk_readback_
                  << " ns_prologue=" << ns_prologue_
                  << " ns_arena_probe=" << ns_arena_probe_
                  << " ns_vk_arena_probe=" << ns_vk_arena_probe_
                  << " ns_vk_setup=" << ns_vk_setup_
                  << " ns_vk_rebind=" << ns_vk_rebind_
                  << " ns_vk_layer_ahead=" << ns_vk_layer_ahead_
                  << " gcache_hit=" << n_gcache_hit_
                  << " gcache_miss=" << n_gcache_miss_
                  << " n_arena_hit=" << n_arena_hit_
                  << " n_arena_groups=" << n_arena_groups_
                  << " n_arena_build=" << n_arena_build_
                  << " n_hipgraph_capture=" << n_hipgraph_capture_
                  << " n_hipgraph_replay=" << n_hipgraph_replay_
                  << " n_d3_collapse=" << n_d3_collapse_
                  << " n_d3_typed=" << n_d3_typed_
                  << " n_d3_bounce=" << n_d3_bounce_
                  << " ns_readback=" << ns_readback_
                  << " ns_send=" << ns_send_
                  << " host_bytes=" << host_bytes_
                  << " ns_prep=" << ns_prep_
                  << " ns_prep_setup=" << ns_prep_setup_
                  << " ns_prep_grow=" << ns_prep_grow_
                  << " ns_prep_attach=" << ns_prep_attach_
                  << " ns_prep_set=" << ns_prep_set_
                  << " ns_hits=" << ns_hits_
                  << " ns_wait=" << ns_wait_
                  << " ns_pagein_compute=" << ns_pagein_compute_
                  << " ns_result=" << ns_result_
                  << " ns_encode=" << ns_encode_
                  << " n_weight_nonzero=" << n_weight_nonzero_
                  << " n_weight_total=" << n_weight_total_
                  << " submit_us_min=" << (submit_min_ns_ == UINT64_MAX ? 0 : submit_min_ns_ / 1000)
                  << " submit_us_max=" << (submit_max_ns_ / 1000)
                  << " submit_hist_us[<125,<250,<500,<1k,<2k,<4k,<8k,>=8k]="
                  << submit_bucket_[0] << ',' << submit_bucket_[1] << ','
                  << submit_bucket_[2] << ',' << submit_bucket_[3] << ','
                  << submit_bucket_[4] << ',' << submit_bucket_[5] << ','
                  << submit_bucket_[6] << ',' << submit_bucket_[7];
        {
            uint64_t pn = 0, pmin = 0, pmean = 0;
            if (self_bench_stats(pn, pmin, pmean)) {
                std::cout << " probe_n=" << pn
                          << " probe_static_us_min=" << pmin
                          << " probe_static_us_mean=" << pmean;
            }
        }
        // WP_WORKER_MULTI_CONN: per-connection request counts, so a live N-conn
        // run visibly shows every stream making progress rather than just one
        // combined total that could be hiding a stalled connection. Empty
        // outside multi-conn mode (g_worker_conn_request_counts stays
        // default-sized 0), so this is a no-op on the default path.
        if (!g_worker_conn_request_counts.empty()) {
            std::cout << " conn_reqs=[";
            for (size_t i = 0; i < g_worker_conn_request_counts.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << g_worker_conn_request_counts[i].load(std::memory_order_relaxed);
            }
            std::cout << ']';
        }
        // Per-connection outstanding staging leases -- see StagingPool's
        // quota (2026-08-25 deadlock fix). A connection pinned at its quota
        // while another sits idle is the signature of the wedge this exists
        // to diagnose; empty outside multi-conn mode, same as conn_reqs.
        if (!g_worker_staging_held.empty()) {
            std::cout << " staging_held=[";
            for (size_t i = 0; i < g_worker_staging_held.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << g_worker_staging_held[i].load(std::memory_order_relaxed);
            }
            std::cout << ']';
        }
        std::cout << " n_begin_unlocked_reads="
                  << g_worker_n_begin_unlocked_reads.load(std::memory_order_relaxed);
        std::cout << " n_reader_h2d_verify_fail="
                  << g_worker_n_reader_h2d_verify_fail.load(std::memory_order_relaxed);
        std::cout << std::endl;
    }

    bool              enabled_ = false;
    clock::time_point next_report_;
    uint64_t          self_bench_every_ =
        std::getenv("WP_SELF_BENCH_EVERY") ? (uint64_t) atoll(std::getenv("WP_SELF_BENCH_EVERY")) : 0;
    ggml_backend_t    probe_backend_ = nullptr;
    uint64_t          submit_min_ns_ = UINT64_MAX;
    uint64_t          submit_max_ns_ = 0;
    uint64_t          submit_bucket_[8] = {0,0,0,0,0,0,0,0};
    uint64_t          ns_lookup_  = 0;
    uint64_t          ns_read_    = 0;
    uint64_t          ns_compute_ = 0;
    uint64_t          ns_send_    = 0;
    uint64_t          n_resident_      = 0;
    uint64_t          n_pagein_     = 0;
    uint64_t          n_pagein_reserved_ = 0;
    uint64_t          n_pagein_general_ = 0;
    uint64_t          n_shield_hits_ = 0;
    uint64_t          n_shield_exhausted_ = 0;
    uint64_t          n_layerahead_hints_   = 0;
    uint64_t          n_layerahead_pageins_ = 0;
    uint64_t          n_layerahead_hits_    = 0;
    size_t             n_pinned_ = 0;
    uint64_t           n_pinned_demand_hits_ = 0;
    uint64_t          n_host_hit_ = 0;
    uint64_t          n_host_demote_ = 0;
    uint64_t          bytes_read_ = 0;
    uint64_t          ns_host_get_ = 0;
    uint64_t          host_bytes_ = 0;
    uint64_t          n_graph_submits_ = 0;
    uint64_t          n_device_allocs_ = 0;
    uint64_t          ns_graph_build_ = 0;
    uint64_t          ns_submit_ = 0;
    uint64_t          ns_final_sync_ = 0;
    uint64_t          ns_vk_compute_path_ = 0;
    uint64_t          ns_vk_dispatch_path_ = 0;
    uint64_t          ns_vk_wait_ = 0;
    uint64_t          ns_vk_cache_lookup_ = 0;
    uint64_t          ns_vk_graph_compute_ = 0;
    uint64_t          ns_vk_params_set_ = 0;
    uint64_t          ns_vk_fold_ = 0;
    uint64_t          ns_vk_sync_ = 0;
    uint64_t          ns_vk_readback_ = 0;
    uint64_t          ns_prologue_ = 0;
    uint64_t          ns_arena_probe_ = 0;
    uint64_t          ns_vk_arena_probe_ = 0;
    uint64_t          ns_vk_setup_ = 0;
    uint64_t          ns_vk_rebind_ = 0;
    uint64_t          ns_vk_layer_ahead_ = 0;
    uint64_t          n_gcache_hit_ = 0;
    uint64_t          n_gcache_miss_ = 0;
    uint64_t          n_arena_hit_ = 0;
    uint64_t          n_arena_groups_ = 0;
    uint64_t          n_arena_build_ = 0;
    uint64_t          n_hipgraph_capture_ = 0;
    uint64_t          n_hipgraph_replay_ = 0;
    uint64_t          n_d3_collapse_ = 0;
    uint64_t          n_d3_typed_ = 0;
    uint64_t          n_d3_bounce_ = 0;
    uint64_t          ns_readback_ = 0;
    uint64_t          ns_dispatch_total_ = 0;
    uint64_t          ns_lock_wait_    = 0;
    uint64_t          ns_decode_req_   = 0;
    uint64_t          ns_pre_dispatch_ = 0;
    uint64_t          ns_encode_send_  = 0;
    uint64_t          ns_post_send_    = 0;
    uint64_t          ns_recv_body_    = 0;
    uint64_t          ns_req_decode_   = 0;
    uint64_t          ns_resp_send_    = 0;
    uint64_t          ns_prep_ = 0;
    uint64_t          ns_prep_setup_ = 0;
    uint64_t          ns_prep_grow_ = 0;
    uint64_t          ns_prep_attach_ = 0;
    uint64_t          ns_prep_set_ = 0;
    uint64_t          ns_hits_ = 0;
    uint64_t          ns_wait_ = 0;
    uint64_t          ns_pagein_compute_ = 0;
    uint64_t          ns_result_ = 0;
    uint64_t          ns_encode_ = 0;
    uint64_t          n_weight_nonzero_ = 0;
    uint64_t          n_weight_total_ = 0;
    uint64_t          ns_h2d_     = 0;
    uint64_t          ns_demote_ = 0;
    uint64_t          ns_ensure_post_ = 0;
    uint64_t          n_read_inflight_max_ = 0;
    uint64_t          ns_read_issue_ = 0;
    uint64_t          ns_read_complete_ = 0;
    uint64_t          n_cpu_on_arrival_ = 0;
    uint64_t          ns_cpu_on_arrival_ = 0;
    uint64_t          n_cpu_on_arrival_fallback_ = 0;
    std::string       device_;
    uint64_t          bytes_h2d_  = 0;
    uint64_t          n_reader_h2d_ = 0;
    std::string       staging_kind_ = "unknown";
    uint64_t          n_requests_ = 0;
    uint64_t          n_experts_  = 0;
};

static ResourcePlan plan_resources_impl(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes,
        uint64_t pinned_bytes,
        const std::vector<int> & reserve_blocks,
        uint64_t reserve_bytes,
        uint64_t arena_alignment) {
    if (arena_alignment == 0) {
        arena_alignment = 1;
    }
    if (requested_slots <= 0) {
        throw std::invalid_argument("invalid expert resource plan dimensions");
    }

    std::map<uint64_t, int> histogram;
    std::map<uint64_t, std::map<int, int>> layer_counts;
    uint64_t max_page_size = 0;
    uint64_t max_staging_size = 0;
    for (const ResourcePage & page : pages) {
        if (page.layer < 0 || page.size == 0 ||
            (!page.pinned && (histogram[page.size] == std::numeric_limits<int>::max() ||
             layer_counts[page.size][page.layer] == std::numeric_limits<int>::max()))) {
            throw std::invalid_argument("invalid expert resource page");
        }
        if (!page.pinned) {
            ++histogram[page.size];
            ++layer_counts[page.size][page.layer];
        }
        max_page_size = std::max(max_page_size, page.size);
        max_staging_size = std::max(
            max_staging_size, page.staging_size == 0 ? page.size : page.staging_size);
    }
    if (max_page_size >
        std::numeric_limits<uint64_t>::max() / (uint64_t) requested_slots) {
        throw std::overflow_error("expert device budget overflows");
    }

    ResourcePlan result;
    result.requested_slots      = requested_slots;
    result.device_budget_bytes =
        max_page_size * (uint64_t) requested_slots;
    result.pinned_bytes = pinned_bytes;
    if (pinned_bytes > result.device_budget_bytes) {
        throw std::invalid_argument("resident expert bytes exceed device budget");
    }
    result.slot_budget_bytes = result.device_budget_bytes - pinned_bytes;

    if (histogram.empty()) {
        result.host_budget_bytes = host_budget_bytes;
        result.staging_buffers = 0;
        return result;
    }

    uint64_t    total_pages   = 0;
    long double weighted_bytes = 0.0;
    uint64_t    floor_bytes   = 0;
    std::vector<SlotClass> classes;
    classes.reserve(histogram.size());
    for (const auto & item : histogram) {
        int floor = 0;
        for (const auto & layer : layer_counts.at(item.first)) {
            floor = std::max(floor, layer.second);
        }
        if ((uint64_t) floor >
            (std::numeric_limits<uint64_t>::max() - floor_bytes) / item.first) {
            throw std::overflow_error("expert pin floor overflows");
        }
        floor_bytes += item.first * (uint64_t) floor;
        total_pages += (uint64_t) item.second;
        weighted_bytes +=
            (long double) item.first * (long double) item.second;
        classes.push_back({ item.first, 0, floor, item.second });
    }

    bool use_size_classes = floor_bytes <= result.slot_budget_bytes;
    if (use_size_classes) {
        const long double average =
            weighted_bytes / (long double) total_pages;
        const long double remaining =
            (long double) (result.slot_budget_bytes - floor_bytes);
        const long double remaining_slots = remaining / average;
        for (SlotClass & slot_class : classes) {
            const long double fraction =
                (long double) slot_class.pages / (long double) total_pages;
            long long count =
                (long long) slot_class.pin_floor +
                std::llround(fraction * remaining_slots);
            count = std::max<long long>(1, count);
            count = std::max<long long>(slot_class.pin_floor, count);
            count = std::min<long long>(slot_class.pages, count);
            slot_class.slots = (int) count;
        }

        auto planned_bytes = [&]() {
            uint64_t bytes = 0;
            for (const SlotClass & slot_class : classes) {
                bytes += slot_class.size * (uint64_t) slot_class.slots;
            }
            return bytes;
        };

        auto planned_arena_bytes = [&]() {
            uint64_t bytes = 0;
            for (const SlotClass & slot_class : classes) {
                if (slot_class.size > UINT64_MAX - (arena_alignment - 1)) {
                    throw std::overflow_error("expert arena size overflows");
                }
                const uint64_t stride =
                    (slot_class.size + arena_alignment - 1) / arena_alignment * arena_alignment;
                if ((uint64_t) slot_class.slots > UINT64_MAX / stride ||
                        bytes > UINT64_MAX - stride * (uint64_t) slot_class.slots) {
                    throw std::overflow_error("expert arena size overflows");
                }
                bytes += stride * (uint64_t) slot_class.slots;
            }
            return bytes;
        };

        while (planned_bytes() > result.slot_budget_bytes ||
                (classes.size() > 1 &&
                 planned_arena_bytes() > result.slot_budget_bytes)) {
            SlotClass * trim = nullptr;
            for (SlotClass & slot_class : classes) {
                const int keep = std::max(1, slot_class.pin_floor);
                if (slot_class.slots > keep &&
                    (trim == nullptr || slot_class.size > trim->size)) {
                    trim = &slot_class;
                }
            }
            if (trim == nullptr) {
                use_size_classes = false;
                break;
            }
            --trim->slots;
        }
    }

    if (!use_size_classes) {
        int max_layer_pages = 0;
        std::map<int, int> pages_by_layer;
        for (const ResourcePage & page : pages) {
            if (page.pinned) {
                continue;
            }
            max_layer_pages =
                std::max(max_layer_pages, ++pages_by_layer[page.layer]);
        }
        if (result.slot_budget_bytes / max_page_size < (uint64_t) max_layer_pages) {
            throw std::invalid_argument(
                "expert slot budget is smaller than the largest layer request");
        }
        classes.clear();
        classes.push_back({
            max_page_size, (int) (result.slot_budget_bytes / max_page_size),
            max_layer_pages, (int) total_pages
        });
    }

    result.size_classes = use_size_classes;
    result.slot_classes = std::move(classes);
    for (const SlotClass & slot_class : result.slot_classes) {
        if (slot_class.slots >
            std::numeric_limits<int>::max() - result.slot_count) {
            throw std::overflow_error("expert slot count overflows");
        }
        result.slot_count += slot_class.slots;
        result.device_bytes +=
            slot_class.size * (uint64_t) slot_class.slots;
    }
    if (reserve_bytes != 0 && !reserve_blocks.empty()) {
        result.requested_reserved_bytes = reserve_bytes;
        uint64_t named_bytes = 0;
        for (const ResourcePage & page : pages) {
            if (!page.pinned && std::binary_search(reserve_blocks.begin(), reserve_blocks.end(), page.layer)) {
                if (named_bytes > UINT64_MAX - page.size) named_bytes = UINT64_MAX;
                else named_bytes += page.size;
            }
        }
        const uint64_t target = std::min(reserve_bytes, named_bytes);
        result.named_reservable_bytes = named_bytes;
        uint64_t remaining = target;
        int slot_index = 0;
        for (const SlotClass & slot_class : result.slot_classes) {
            const int class_start = slot_index;
            for (int i = 0; i < slot_class.slots && remaining >= slot_class.size; ++i) {
                result.reserved_slot_indices.push_back(class_start + i);
                result.reserved_bytes += slot_class.size;
                remaining -= slot_class.size;
            }
            slot_index += slot_class.slots;
        }
        result.reserved_slot_count = (int) result.reserved_slot_indices.size();
        result.general_slot_count = result.slot_count - result.reserved_slot_count;
        if (target != 0 && result.reserved_bytes == 0) {
            result.reserved_slot_indices.clear();
            result.reserved_slot_count = 0;
            result.general_slot_count = result.slot_count;
        }
    }
    if (result.general_slot_count == 0 && result.reserved_slot_count == 0) {
        result.general_slot_count = result.slot_count;
    }

    const uint64_t requested_staging_buffers = staging_buffers_from_env();
    const uint64_t default_staging_buffers =
        std::min<uint64_t>((uint64_t) result.slot_count, requested_staging_buffers);
    if (default_staging_buffers != 0 &&
        max_staging_size > std::numeric_limits<uint64_t>::max() / default_staging_buffers) {
        throw std::overflow_error("default expert host budget overflows");
    }
    result.host_budget_bytes = host_budget_bytes == 0
        ? max_staging_size * default_staging_buffers
        : host_budget_bytes;
    if (result.host_budget_bytes < max_staging_size) {
        throw std::invalid_argument(
            "expert host budget is smaller than the largest page");
    }
    const uint64_t staging_count =
        std::min<uint64_t>(
            (uint64_t) result.slot_count,
            result.host_budget_bytes / max_staging_size);
    if (staging_count == 0 ||
        staging_count > (uint64_t) std::numeric_limits<int>::max()) {
        throw std::overflow_error("invalid expert staging buffer count");
    }
    result.staging_buffers      = (int) staging_count;
    result.staging_buffer_bytes = max_staging_size;
    result.staging_bytes        = max_staging_size * staging_count;
    return result;
}

ResourcePlan plan_resources(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes,
        uint64_t pinned_bytes,
        const std::vector<int> & reserve_blocks,
        uint64_t reserve_bytes) {
    return plan_resources_impl(
        pages, requested_slots, host_budget_bytes, pinned_bytes,
        reserve_blocks, reserve_bytes, 1);
}

static ResourcePlan plan_resources_for_backend(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes,
        uint64_t pinned_bytes,
        const std::vector<int> & reserve_blocks,
        uint64_t reserve_bytes,
        ggml_backend_t backend) {
    const ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(backend);
    return plan_resources_impl(
        pages, requested_slots, host_budget_bytes, pinned_bytes,
        reserve_blocks, reserve_bytes,
        (uint64_t) ggml_backend_buft_get_alignment(buft));
}

static size_t expert_pin_class_pct_from_env() {
    const char * env = std::getenv("WP_EXPERT_PIN_CLASS_PCT");
    if (env == nullptr || env[0] == '\0') {
        return 85;
    }
    const long parsed = std::strtol(env, nullptr, 10);
    return parsed >= 0 ? std::min<long>(parsed, 100) : 85;
}

static size_t expert_pin_class_index(const ResourcePlan & resources, uint64_t page_size) {
    for (size_t i = 0; i < resources.slot_classes.size(); ++i) {
        const SlotClass & slot_class = resources.slot_classes[i];
        if ((resources.size_classes && slot_class.size == page_size) ||
                (!resources.size_classes && slot_class.size >= page_size)) {
            return i;
        }
    }
    return resources.slot_classes.size();
}

std::vector<DeviceMemberLayout> plan_device_member_layout(
        const std::vector<uint64_t> & sizes, uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument("invalid device member alignment");
    }

    std::vector<DeviceMemberLayout> result;
    result.reserve(sizes.size());
    uint64_t offset = 0;
    for (const uint64_t size : sizes) {
        if (size == 0 || offset > UINT64_MAX - (alignment - 1)) {
            throw std::overflow_error("expert member layout overflows");
        }
        offset = GGML_PAD(offset, alignment);
        if (offset > UINT64_MAX - size) {
            throw std::overflow_error("expert member layout overflows");
        }
        result.push_back({ offset, size });
        offset += size;
    }
    return result;
}

namespace {

static constexpr const char * MANIFEST_FORMAT =
    "llama.cpp.weight-pager.expert-shard-manifest";
static constexpr const char * INDEX_FORMAT =
    "llama.cpp.weight-pager.expert-shard-index";
static constexpr const char * DESCRIPTOR_FORMAT =
    "llama.cpp.weight-pager.expert-descriptor";
struct RoleSpec {
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t        ne0 = 0;
    int64_t        ne1 = 0;
    uint64_t       bytes = 0;
    std::string    source_tensor_name;
};

struct HParams {
    int n_layer       = 0;
    int n_embd        = 0;
    int n_ff_exp      = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
};

struct Descriptor {
    HParams                                      hparams;
    int                                          expert_first = -1;
    int                                          expert_last  = -1;
    bool                                         sliced = false;
    int64_t                                      slice_first = 0;
    int64_t                                      slice_last  = 0;
    json                                         expert_slicing;
    std::string                                  input_model;
    std::string                                  identity_algorithm;
    std::string                                  identity_value;
    std::vector<std::string>                     model_files;
    std::map<int, std::map<std::string, RoleSpec>> layers;
};

struct MemberSpan {
    uint64_t offset = 0;
    uint64_t size   = 0;
    uint64_t device_offset = 0;
    uint64_t device_bytes = 0;
};

struct ExpertPage {
    int                               cache_id = -1;
    int                               layer  = -1;
    int                               expert = -1;
    fs::path                          blob;
    uint64_t                          offset = 0;
    uint64_t                          size   = 0;
    uint64_t                          device_size = 0;
    std::map<std::string, MemberSpan> roles;
    bool                              is_resident = false;
    ggml_backend_buffer_t             resident_buffer = nullptr;
    void *                            resident_base = nullptr;
};

struct Catalog {
    Descriptor                                      descriptor;
    std::map<std::pair<int, int>, ExpertPage>       pages;
    std::vector<int>                                layers;
    uint64_t                                        max_page_size = 0;
};

struct backend_deleter {
    void operator()(ggml_backend * backend) const {
        ggml_backend_free(backend);
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        ggml_backend_buffer_free(buffer);
    }
};

struct context_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

struct galloc_deleter {
    void operator()(ggml_gallocr * galloc) const {
        ggml_gallocr_free(galloc);
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using buffer_ptr  = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using galloc_ptr  = std::unique_ptr<ggml_gallocr, galloc_deleter>;

json read_json(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    json value;
    try {
        input >> value;
    } catch (const json::exception & error) {
        throw std::runtime_error("failed to parse " + path.string() + ": " + error.what());
    }
    return value;
}

template <typename T>
T get_value(const json & value, const char * key, const fs::path & path) {
    try {
        return value.at(key).get<T>();
    } catch (const json::exception & error) {
        throw std::runtime_error(path.string() + ": invalid " + key + ": " + error.what());
    }
}

const json & get_array(const json & value, const char * key, const fs::path & path) {
    try {
        const json & result = value.at(key);
        if (!result.is_array()) {
            throw std::runtime_error(path.string() + ": " + key + " is not an array");
        }
        return result;
    } catch (const json::exception & error) {
        throw std::runtime_error(path.string() + ": invalid " + key + ": " + error.what());
    }
}

void check_format(const json & value, const char * expected, const fs::path & path) {
    if (get_value<std::string>(value, "format", path) != expected ||
        get_value<int>(value, "version", path) != 1) {
        throw std::runtime_error(path.string() + ": unsupported format or version");
    }
}

int checked_int(uint64_t value, const char * name) {
    if (value == 0 || value > INT32_MAX) {
        throw std::runtime_error(std::string(name) + " is out of range");
    }
    return (int) value;
}

void sha_update_u64(sha256_t & hash, uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = (unsigned char) ((value >> (i * 8)) & 0xffu);
    }
    sha256_update(&hash, bytes.data(), bytes.size());
}

void sha_update_string(sha256_t & hash, const std::string & value) {
    sha_update_u64(hash, value.size());
    sha256_update(
        &hash, reinterpret_cast<const unsigned char *>(value.data()), value.size());
}

std::string source_model_identity(
        const std::string & input_model,
        const std::vector<std::string> & model_files) {
    sha256_t hash;
    sha256_init(&hash);
    sha_update_string(hash, "llama.cpp.wp-expert.source-model.v1");
    sha_update_string(hash, input_model);
    sha_update_u64(hash, model_files.size());
    for (const std::string & model_file : model_files) {
        sha_update_string(hash, model_file);
    }

    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_final(&hash, digest.data());
    std::ostringstream result;
    result << "sha256:" << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        result << std::setw(2) << (unsigned int) byte;
    }
    return result.str();
}

RoleSpec parse_role(const json & value, const fs::path & path) {
    RoleSpec role;
    const int type = get_value<int>(value, "ggml_type", path);
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        throw std::runtime_error(path.string() + ": ggml_type is out of range");
    }
    role.type = (enum ggml_type) type;
    const std::string type_name = get_value<std::string>(value, "ggml_type_name", path);
    if (type_name != ggml_type_name(role.type)) {
        throw std::runtime_error(path.string() + ": ggml type name and enum disagree");
    }
    const json & shape = get_array(value, "shape", path);
    if (shape.size() != 2) {
        throw std::runtime_error(path.string() + ": expert role shape is not 2D");
    }
    role.ne0   = shape.at(0).get<int64_t>();
    role.ne1   = shape.at(1).get<int64_t>();
    role.bytes = get_value<uint64_t>(value, "bytes_per_expert", path);
    role.source_tensor_name =
        get_value<std::string>(value, "source_tensor_name", path);
    if (role.ne0 <= 0 || role.ne1 <= 0 ||
        role.source_tensor_name.empty() ||
        ggml_row_size(role.type, role.ne0) * (uint64_t) role.ne1 != role.bytes) {
        throw std::runtime_error(path.string() + ": expert role shape/type byte count is invalid");
    }
    return role;
}

Descriptor load_descriptor(const fs::path & path) {
    const json value = read_json(path);
    check_format(value, DESCRIPTOR_FORMAT, path);
    Descriptor result;

    const json & hparams = value.at("hparams");
    result.hparams.n_layer =
        checked_int(get_value<uint64_t>(hparams, "n_layer", path), "n_layer");
    result.hparams.n_embd =
        checked_int(get_value<uint64_t>(hparams, "n_embd", path), "n_embd");
    result.hparams.n_ff_exp =
        checked_int(get_value<uint64_t>(hparams, "n_ff_exp", path), "n_ff_exp");
    result.hparams.n_expert =
        checked_int(get_value<uint64_t>(hparams, "n_expert", path), "n_expert");
    result.hparams.n_expert_used =
        checked_int(get_value<uint64_t>(hparams, "n_expert_used", path), "n_expert_used");
    if (result.hparams.n_expert_used > result.hparams.n_expert ||
        get_value<std::string>(hparams, "activation", path) != "silu") {
        throw std::runtime_error(path.string() + ": unsupported expert hparams");
    }

    const json & range = value.at("retained_expert_range");
    result.expert_first = get_value<int>(range, "first", path);
    result.expert_last  = get_value<int>(range, "last", path);
    if (result.expert_first < 0 || result.expert_last < result.expert_first ||
        result.expert_last >= result.hparams.n_expert) {
        throw std::runtime_error(path.string() + ": invalid retained expert range");
    }
    if (value.contains("expert_slicing")) {
        const json & slicing = value.at("expert_slicing");
        const int slice_index = get_value<int>(slicing, "selected_slice", path);
        const json & widths = get_array(slicing, "widths", path);
        if (get_value<int64_t>(slicing, "n_ff_exp", path) != result.hparams.n_ff_exp ||
            get_value<int64_t>(slicing, "n_embd", path) != result.hparams.n_embd ||
            slice_index < 0 || slice_index >= (int) widths.size()) {
            throw std::runtime_error(path.string() + ": invalid expert slice geometry");
        }
        result.sliced = true;
        result.expert_slicing = slicing;
        for (int i = 0; i < slice_index; ++i) {
            result.slice_first += widths.at(i).get<int64_t>();
        }
        result.slice_last = result.slice_first + widths.at(slice_index).get<int64_t>();
        if (result.slice_first < 0 || result.slice_last <= result.slice_first ||
            result.slice_last > result.hparams.n_ff_exp) {
            throw std::runtime_error(path.string() + ": invalid selected expert slice");
        }
    }

    const json & identity = value.at("shard_manifest_identity");
    result.identity_algorithm = get_value<std::string>(identity, "algorithm", path);
    result.identity_value     = get_value<std::string>(identity, "value", path);
    if (result.identity_algorithm.empty() || result.identity_value.empty()) {
        throw std::runtime_error(path.string() + ": empty shard manifest identity");
    }
    const json & source_model = value.at("source_model");
    result.input_model = get_value<std::string>(source_model, "input_model", path);
    for (const json & model_file : get_array(source_model, "model_files", path)) {
        result.model_files.push_back(model_file.get<std::string>());
    }
    if (result.input_model.empty() || result.model_files.empty()) {
        throw std::runtime_error(path.string() + ": descriptor has no source model");
    }

    const json & layers = get_array(value, "layers", path);
    for (const json & layer_value : layers) {
        const int layer = get_value<int>(layer_value, "layer", path);
        if (layer < 0 || layer >= result.hparams.n_layer || result.layers.count(layer) != 0) {
            throw std::runtime_error(path.string() + ": invalid or repeated layer descriptor");
        }
        const json & roles = layer_value.at("roles");
        std::map<std::string, RoleSpec> parsed;
        for (const std::string name : { "gate", "up", "down" }) {
            parsed.emplace(name, parse_role(roles.at(name), path));
        }
        const int64_t n_ff = result.sliced ? result.slice_last - result.slice_first : result.hparams.n_ff_exp;
        if (parsed.at("gate").ne0 != result.hparams.n_embd ||
            parsed.at("gate").ne1 != n_ff ||
            parsed.at("up").ne0 != result.hparams.n_embd ||
            parsed.at("up").ne1 != n_ff ||
            parsed.at("down").ne0 != n_ff ||
            parsed.at("down").ne1 != result.hparams.n_embd) {
            throw std::runtime_error(path.string() + ": descriptor role shapes disagree with hparams");
        }
        result.layers.emplace(layer, std::move(parsed));
    }
    if (result.layers.empty()) {
        throw std::runtime_error(path.string() + ": descriptor has no layers");
    }
    return result;
}

std::string role_from_mask(uint64_t mask) {
    switch (mask) {
        case 1: return "up";
        case 2: return "gate";
        case 4: return "down";
        default:
            throw std::runtime_error("sidecar has an unknown role mask " + std::to_string(mask));
    }
}

Catalog load_catalog(const fs::path & manifest_path, const fs::path & descriptor_path) {
    Catalog result;
    result.descriptor = load_descriptor(descriptor_path);

    const json manifest = read_json(manifest_path);
    check_format(manifest, MANIFEST_FORMAT, manifest_path);
    const std::string sharding_mode = get_value<std::string>(manifest, "sharding_mode", manifest_path);
    if ((result.descriptor.sliced && sharding_mode != "expert-slice") ||
        (!result.descriptor.sliced && sharding_mode != "expert-index-range")) {
        throw std::runtime_error("worker descriptor and shard manifest sharding modes disagree");
    }
    if (result.descriptor.sliced && manifest.at("expert_slicing") != result.descriptor.expert_slicing) {
        throw std::runtime_error("descriptor and shard manifest expert slicing disagree");
    }
    const json & identity = manifest.at("content_hash");
    if (get_value<std::string>(identity, "algorithm", manifest_path) !=
            result.descriptor.identity_algorithm ||
        get_value<std::string>(identity, "value", manifest_path) !=
            result.descriptor.identity_value) {
        throw std::runtime_error("descriptor does not match shard manifest identity");
    }
    if (get_array(manifest, "model_files", manifest_path) !=
        json(result.descriptor.model_files)) {
        throw std::runtime_error("descriptor and shard manifest model files disagree");
    }
    if (get_value<std::string>(manifest, "input_model", manifest_path) !=
        result.descriptor.input_model) {
        throw std::runtime_error("descriptor and shard manifest input models disagree");
    }
    const json & range = manifest.at("retained_expert_range");
    if (get_value<int>(range, "first", manifest_path) != result.descriptor.expert_first ||
        get_value<int>(range, "last", manifest_path) != result.descriptor.expert_last) {
        throw std::runtime_error("descriptor and shard manifest expert ranges disagree");
    }

    const json & shards = get_array(manifest, "shards", manifest_path);
    if (shards.size() != get_value<uint64_t>(manifest, "shard_count", manifest_path)) {
        throw std::runtime_error("manifest shard count mismatch");
    }
    std::set<int> seen_layers;
    std::set<int> seen_shard_indices;
    uint64_t total_groups = 0;
    uint64_t total_bytes  = 0;
    for (const json & shard : shards) {
        const fs::path index_path = manifest_path.parent_path() /
            get_value<std::string>(shard, "index_file", manifest_path);
        const json index = read_json(index_path);
        check_format(index, INDEX_FORMAT, index_path);
        if (result.descriptor.sliced && index.at("expert_slicing") != result.descriptor.expert_slicing) {
            throw std::runtime_error(index_path.string() + ": expert slicing disagrees with descriptor");
        }
        const int shard_index = get_value<int>(index, "shard_index", index_path);
        const int layer_first = get_value<int>(index, "layer_first", index_path);
        const int layer_last  = get_value<int>(index, "layer_last", index_path);
        if (shard_index < 0 || (uint64_t) shard_index >= shards.size() ||
            !seen_shard_indices.insert(shard_index).second ||
            get_value<uint64_t>(index, "shard_count", index_path) != shards.size() ||
            get_value<int>(shard, "shard_index", manifest_path) != shard_index ||
            get_value<int>(shard, "layer_first", manifest_path) != layer_first ||
            get_value<int>(shard, "layer_last", manifest_path) != layer_last ||
            get_value<uint64_t>(shard, "group_count", manifest_path) !=
                get_value<uint64_t>(index, "group_count", index_path) ||
            get_value<uint64_t>(shard, "blob_bytes", manifest_path) !=
                get_value<uint64_t>(index, "blob_bytes", index_path) ||
            get_value<std::string>(shard, "blob_file", manifest_path) !=
                get_value<std::string>(index, "blob_file", index_path) ||
            get_array(index, "model_files", index_path) !=
                get_array(manifest, "model_files", manifest_path) ||
            layer_first != layer_last || !seen_layers.insert(layer_first).second ||
            result.descriptor.layers.count(layer_first) == 0) {
            throw std::runtime_error(index_path.string() + ": invalid or undescribed layer");
        }

        const fs::path blob_path = manifest_path.parent_path() /
            get_value<std::string>(index, "blob_file", index_path);
        std::error_code size_error;
        const uint64_t actual_size = fs::file_size(blob_path, size_error);
        const uint64_t blob_bytes  = get_value<uint64_t>(index, "blob_bytes", index_path);
        if (size_error || actual_size != blob_bytes) {
            throw std::runtime_error(blob_path.string() + ": blob size does not match sidecar");
        }

        const json & groups = get_array(index, "groups", index_path);
        if (groups.size() != get_value<uint64_t>(index, "group_count", index_path)) {
            throw std::runtime_error(index_path.string() + ": group count mismatch");
        }
        uint64_t next_offset = 0;
        int expected_expert = result.descriptor.expert_first;
        for (const json & group : groups) {
            ExpertPage page;
            if (total_groups > (uint64_t) INT32_MAX) {
                throw std::runtime_error("expert page cache index is out of range");
            }
            page.cache_id = (int) total_groups;
            page.layer  = get_value<int>(group, "block_idx", index_path);
            page.expert = get_value<int>(group, "expert_idx", index_path);
            page.blob   = blob_path;
            page.offset = next_offset;
            if (page.layer != layer_first || page.expert != expected_expert++) {
                throw std::runtime_error(index_path.string() + ": expert groups are not dense and ordered");
            }
            if (result.descriptor.sliced &&
                (get_value<int>(group, "slice_idx", index_path) !=
                     get_value<int>(result.descriptor.expert_slicing, "selected_slice", descriptor_path) ||
                 get_value<int64_t>(group, "ff_first", index_path) != result.descriptor.slice_first ||
                 get_value<int64_t>(group, "ff_last", index_path) != result.descriptor.slice_last)) {
                throw std::runtime_error(index_path.string() + ": expert group slice span disagrees with descriptor");
            }
            const json & members = get_array(group, "members", index_path);
            if (members.size() != 3 ||
                get_value<uint64_t>(group, "member_count", index_path) != 3) {
                throw std::runtime_error(index_path.string() + ": expert group does not have three members");
            }
            const auto & role_specs = result.descriptor.layers.at(page.layer);
            for (const json & member : members) {
                const std::string role =
                    role_from_mask(get_value<uint64_t>(member, "role_mask", index_path));
                const uint64_t offset = get_value<uint64_t>(member, "offset", index_path);
                const uint64_t size   = get_value<uint64_t>(member, "size", index_path);
                const std::string source_tensor_name =
                    get_value<std::string>(member, "source_tensor_name", index_path);
                if (offset != next_offset || size != role_specs.at(role).bytes ||
                    source_tensor_name != role_specs.at(role).source_tensor_name ||
                    page.roles.count(role) != 0) {
                    throw std::runtime_error(
                        index_path.string() + ": member spans disagree with descriptor");
                }
                if (result.descriptor.sliced && get_array(member, "slice_shape", index_path) !=
                        json({ role_specs.at(role).ne0, role_specs.at(role).ne1 })) {
                    throw std::runtime_error(index_path.string() + ": member slice shape disagrees with descriptor");
                }
                page.roles.emplace(role, MemberSpan{ offset - page.offset, size, offset - page.offset });
                next_offset += size;
                page.size += size;
            }
            // A sliced page is padded with zeros after its last role member so
            // its size is a whole number of O_DIRECT blocks. The member spans
            // only cover the PAYLOAD, so page.size summed from them is short by
            // the padding -- read the recorded padding and extend the page, or
            // an otherwise valid store is rejected as unaligned. Member offsets
            // are unaffected: padding sits after every member.
            const uint64_t page_padding = group.value("padding_bytes", (uint64_t) 0);
            page.size    += page_padding;
            next_offset  += page_padding;
            if (page.offset % DIRECT_ALIGNMENT != 0 ||
                page.size % DIRECT_ALIGNMENT != 0) {
                throw std::runtime_error(
                    index_path.string() +
                    ": expert page is not aligned for one O_DIRECT read");
            }
            page.device_size = page.size;
            result.max_page_size = std::max(result.max_page_size, page.size);
            if (!result.pages.emplace(
                    std::make_pair(page.layer, page.expert), std::move(page)).second) {
                throw std::runtime_error(index_path.string() + ": repeated expert page");
            }
            ++total_groups;
        }
        if (next_offset != blob_bytes) {
            throw std::runtime_error(index_path.string() + ": groups do not cover blob");
        }
        total_bytes += blob_bytes;
    }
    if (seen_layers.size() != result.descriptor.layers.size()) {
        throw std::runtime_error("descriptor and shard manifest layer sets disagree");
    }
    if (total_groups != get_value<uint64_t>(manifest, "total_group_count", manifest_path) ||
        total_bytes != get_value<uint64_t>(manifest, "total_blob_bytes", manifest_path)) {
        throw std::runtime_error("shard catalog totals disagree with manifest");
    }
    result.layers.assign(seen_layers.begin(), seen_layers.end());
    return result;
}

void run_self_bench(ggml_backend_t backend, uint32_t n_embd, uint32_t n_ff);
void run_self_bench_early(ggml_backend_t backend) {
    // DS4-Flash expert shape, hardcoded: init_backend has no catalog yet and
    // this path is diagnostic-only (WP_SELF_BENCH=2).
    const char * saved = std::getenv("WP_SELF_BENCH");
    (void) saved;
    setenv("WP_SELF_BENCH", "1", 1);   // run_self_bench gates on '1'
    run_self_bench(backend, 4096, 2048);
    setenv("WP_SELF_BENCH", "2", 1);
}

// Thread count for the CPU expert backend. ggml_backend_cpu_set_n_threads()
// bakes an EXPLICIT num_threads(n) clause onto every OpenMP region, which
// OVERRIDES OMP_NUM_THREADS -- so passing hardware_concurrency() (24 SMT on the
// 3900X) forced 24 threads no matter the env. Measured 2026-08-05: 24 threads
// is ~2x SLOWER than 8 on ns_submit (raw matmul), because the 3900X is
// 12-core/24-thread and also runs the spine + R9700 worker + desktop, so 24
// oversubscribes and every tiny per-request graph pays full OpenMP barrier cost
// on threads mostly waiting on each other. The DSPARK_OMP=8 fix used to live
// only in the launch harness (OMP_THREAD_LIMIT), silently lost on any other
// launcher. Default to the measured optimum in the worker itself; WP_CPU_THREADS
// overrides (clamped to [1, hw]).
int cpu_worker_n_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    const unsigned int hw_clamped = hw == 0 ? 1 : hw;
    unsigned int n = hw_clamped > 8 ? 8u : hw_clamped;   // measured optimum
    if (const char * e = std::getenv("WP_CPU_THREADS")) {
        const long parsed = std::strtol(e, nullptr, 10);
        if (parsed > 0) {
            n = (unsigned int) std::min<long>(parsed, (long) hw_clamped);
        }
    }
    return (int) n;
}

// ---------------------------------------------------------------------------
// *** WP_CPU_TIER_OVERLAP -- LET THE CPU EXPERT TIER RUN ALONGSIDE THE GPUs. ***
//
// DEFAULT OFF. Unset (or "0") and every function below is inert: the CPU tier
// keeps its own serial phase after the GPUs, exactly as measured and shipped.
//
// THE PROBLEM THIS SOLVES. See the long comment on the WP_DEVICE_PARALLEL
// dispatch loop: running the CPU expert tier concurrently with the GPU tiers
// was tried and REGRESSED everything, because the CPU tier's threads spin at
// their barriers and deschedule the GPU backends' host/submission threads.
// Serialising it costs a measured 0.42 ms per layer-request on 2026 and 0.32 ms
// on main; at 48 layer RPCs per token that is ~15-20 ms/tok on a ~116 ms/tok
// decode. Recovering it needs the CPU tier to overlap WITHOUT being allowed to
// eat every core.
//
// WHY THE ggml `poll` KNOB IS NOT THE ANSWER HERE. ggml_threadpool_params::poll
// (poll=0 => blocking wait) only governs ggml's OWN worker threads, which exist
// only in the `#ifndef GGML_USE_OPENMP` build. This tree builds ggml-cpu with
// GGML_OPENMP=ON (GGML_OPENMP:BOOL=ON in every build-*/CMakeCache.txt here), so
// ggml_graph_compute() runs the graph inside `#pragma omp parallel` and
// ggml_barrier() is `#pragma omp barrier`. The spinning is libgomp's, tuned by
// GOMP_SPINCOUNT/OMP_WAIT_POLICY, which libgomp latches in a constructor before
// main() -- setenv() from inside the worker is too late to change it. So `poll`
// is dead config on this build and env-poking is unreliable.
//
// WHAT WE DO INSTEAD -- CONFINE, DON'T UNSPIN. Under OpenMP,
// ggml_threadpool_new() spawns NO threads: it is purely a carrier for per-thread
// cpumasks and a scheduling priority, and ggml_graph_compute() applies BOTH
// FROM INSIDE the parallel region, to whichever OpenMP threads it actually got
// (ggml-cpu.c: ggml_thread_apply_priority(threadpool->prio) then
// ggml_thread_apply_affinity(workers[ith].cpumask)). Attaching such a
// threadpool to the CPU tier's backend therefore gives us, per graph:
//
//   1. AFFINITY. The tier's OpenMP team is pinned to a fixed subset of logical
//      CPUs. It can still spin all it likes -- it simply no longer HAS the
//      cores the Vulkan/HIP/CUDA submission threads need. Starvation stops
//      being possible rather than being merely discouraged.
//   2. PRIORITY. GGML_SCHED_PRIO_LOW is SCHED_BATCH on Linux (see
//      ggml_thread_apply_priority). SCHED_BATCH marks a thread as CPU-bound and
//      REMOVES ITS WAKEUP PREEMPTION CREDIT, so a GPU submit thread that wakes
//      on the same runqueue preempts the spinner immediately instead of waiting
//      out a scheduling slice. This is the cheap half of the fix and it costs
//      nothing when there is no contention. It is also unprivileged-safe:
//      SCHED_OTHER <-> SCHED_BATCH needs no capability.
//
// Because both are applied INSIDE the parallel region, they land on the real
// team every time and do not depend on which thread created the team.
// ---------------------------------------------------------------------------

bool cpu_tier_overlap_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("WP_CPU_TIER_OVERLAP");
        return e != nullptr && std::strcmp(e, "0") != 0 && e[0] != '\0';
    }();
    return on;
}

// Thread count for the CPU tier WHEN OVERLAPPING. The serial-phase optimum is
// not the overlapped optimum: serialised, the tier owns the machine and wants
// every core it can use; overlapped, it must leave the GPU submission threads
// somewhere to run. WP_CPU_TIER_THREADS overrides; unset it defaults to
// cpu_worker_n_threads() so turning the knob on changes ONE thing at a time.
int cpu_tier_overlap_n_threads() {
    static const int n = [] {
        const unsigned int hw         = std::thread::hardware_concurrency();
        const unsigned int hw_clamped = hw == 0 ? 1 : hw;
        int v = cpu_worker_n_threads();
        if (const char * e = std::getenv("WP_CPU_TIER_THREADS")) {
            const long parsed = std::strtol(e, nullptr, 10);
            if (parsed > 0) {
                v = (int) std::min<long>(parsed, (long) hw_clamped);
            }
        }
        return v;
    }();
    return n;
}

// Parse a Linux-style CPU list: "6,7", "16-23", "0-3,8-11". Returns false if
// nothing was set, so callers can fall back to the derived default.
bool parse_cpu_list(const char * spec, bool * mask /* GGML_MAX_N_THREADS */) {
    bool any = false;
    const char * p = spec;
    while (*p != '\0') {
        while (*p == ',' || *p == ' ') { ++p; }
        if (*p == '\0') { break; }
        char * end = nullptr;
        const long lo = std::strtol(p, &end, 10);
        if (end == p) { break; }
        long hi = lo;
        p = end;
        if (*p == '-') {
            ++p;
            hi = std::strtol(p, &end, 10);
            if (end == p) { break; }
            p = end;
        }
        for (long c = lo; c <= hi; ++c) {
            if (c >= 0 && c < GGML_MAX_N_THREADS) {
                mask[c] = true;
                any = true;
            }
        }
    }
    return any;
}

// The CPU set the overlapped tier is confined to.
//
// WP_CPU_TIER_CPUS takes a CPU list and is the knob you actually want to set
// per box, because the right answer is a TOPOLOGY question this code cannot
// answer portably: logical-CPU numbering differs between the 4c/8t i7-6700K
// (2026) and the 12c/24t 3900X (main), and whether you want whole physical
// cores or SMT siblings depends on how much of the box the GPU drivers need.
//
// The DEFAULT, when the knob is unset, is the last N logical CPUs, N = the
// tier's thread count. That is deliberately the conservative choice: on both
// boxes the high-numbered logical CPUs are the second SMT thread of a physical
// core, so the tier lands on hyperthreads and the primary siblings stay
// available to the GPU submission threads. It is a floor, not an optimum --
// measure, then pin explicitly.
void cpu_tier_cpumask(bool * mask /* GGML_MAX_N_THREADS */) {
    std::memset(mask, 0, GGML_MAX_N_THREADS);
    if (const char * e = std::getenv("WP_CPU_TIER_CPUS")) {
        if (parse_cpu_list(e, mask)) {
            return;
        }
        std::memset(mask, 0, GGML_MAX_N_THREADS);
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) { return; }   // unknown topology: leave the mask empty (= no affinity)
    const int n  = std::min<int>(cpu_tier_overlap_n_threads(), (int) hw);
    const int lo = std::max<int>(0, (int) hw - n);
    for (int c = lo; c < (int) hw && c < GGML_MAX_N_THREADS; ++c) {
        mask[c] = true;
    }
}

// WP_CPU_TIER_STRICT=0|1 (default 1). Strict gives each OpenMP thread ONE
// distinct CPU out of the mask (ggml_thread_cpumask_next), i.e. a hard 1:1 pin;
// non-strict gives every thread the whole mask and lets the scheduler move them
// inside it. Strict is the default because the whole point here is to make the
// tier's footprint deterministic, and it also stops two of the tier's own
// spinning threads from landing on one core.
bool cpu_tier_strict() {
    static const bool strict = [] {
        const char * e = std::getenv("WP_CPU_TIER_STRICT");
        return e == nullptr || (std::strcmp(e, "0") != 0 && e[0] != '\0');
    }();
    return strict;
}

// WP_CPU_TIER_PRIO=low|normal (default low). "low" is SCHED_BATCH on Linux, the
// wakeup-preemption fix described above. Set it to "normal" if the build ever
// starts printing "failed to set thread priority" (that warning would fire once
// per thread per graph, so it would be loud) or if SCHED_BATCH is unavailable.
enum ggml_sched_priority cpu_tier_prio() {
    static const enum ggml_sched_priority prio = [] {
        const char * e = std::getenv("WP_CPU_TIER_PRIO");
        if (e != nullptr && std::strcmp(e, "normal") == 0) {
            return GGML_SCHED_PRIO_NORMAL;
        }
        return GGML_SCHED_PRIO_LOW;
    }();
    return prio;
}

// ---------------------------------------------------------------------------
// WP_CPU_TIER_PIN=1 (DEFAULT OFF) -- ONE-SHOT pin of the CPU expert tier's
// executor thread.
//
// WHY THIS EXISTS AND WHY configure_cpu_backend() IS NOT ENOUGH. ggml applies a
// threadpool's cpumask and priority from INSIDE `#pragma omp parallel`, and
// ggml_graph_compute() only enters that region when n_threads > 1:
//
//     if (n_threads > 1) { #pragma omp parallel ... apply prio/affinity ... }
//     else               { ggml_graph_compute_thread(&workers[0]); }
//
// So at n_threads == 1 -- which is the configuration that makes CPU-tier
// overlap viable on a 4-core box at all, because it means libgomp never forms a
// team and therefore never leaves a worker spinning out GOMP_SPINCOUNT between
// graphs -- the threadpool's cpumask and SCHED_BATCH are DEAD CONFIG. This
// applies them directly to the tier's own DeviceExecutor thread instead.
//
// It is also cheaper where both paths work: ggml re-applies affinity and
// priority on EVERY graph, for every thread in the team (two syscalls per
// thread per graph, and this worker runs 48 layer RPCs per token). This is two
// syscalls for the process lifetime.
//
// Safe to combine with the threadpool path: at n_threads > 1 ggml will simply
// re-apply the same mask and priority over the top of ours.
//
// The mask and priority come from WP_CPU_TIER_CPUS / WP_CPU_TIER_PRIO, exactly
// as for the threadpool, so one topology answer configures both.
bool cpu_tier_pin_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("WP_CPU_TIER_PIN");
        return e != nullptr && std::strcmp(e, "0") != 0 && e[0] != '\0';
    }();
    return on;
}

// Called ON the tier's executor thread, once, at thread start.
void wp_cpu_tier_pin_self() {
#if defined(__linux__)
    if (!cpu_tier_pin_enabled()) {
        return;
    }
    bool mask[GGML_MAX_N_THREADS];
    cpu_tier_cpumask(mask);

    cpu_set_t set;
    CPU_ZERO(&set);
    int n = 0;
    std::string cpus;
    for (int c = 0; c < GGML_MAX_N_THREADS && c < CPU_SETSIZE; ++c) {
        if (mask[c]) {
            CPU_SET(c, &set);
            ++n;
            if (!cpus.empty()) { cpus += ","; }
            cpus += std::to_string(c);
        }
    }
    // An empty mask means "unknown topology" upstream; do not pin to nothing.
    // NOTE: unlike ggml's strict_cpu, the WHOLE mask is given to the single
    // thread. WP_CPU_TIER_CPUS=3,7 therefore hands the tier one entire physical
    // core (both SMT siblings) and lets the scheduler pick, which is what you
    // want for a one-thread tier -- strict 1:1 pinning only matters when there
    // are several tier threads to keep off each other.
    if (n > 0 && sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::fprintf(stderr,
            "wp: WP_CPU_TIER_PIN: sched_setaffinity([%s]) failed: %s\n",
            cpus.c_str(), std::strerror(errno));
    }

    // SCHED_BATCH removes this thread's wakeup preemption credit, so a GPU
    // submission thread waking on the same runqueue preempts it immediately
    // instead of waiting out a slice. Unprivileged-safe (SCHED_OTHER <->
    // SCHED_BATCH needs no capability); nice value is untouched.
    if (cpu_tier_prio() == GGML_SCHED_PRIO_LOW) {
        struct sched_param param;
        std::memset(&param, 0, sizeof(param));
        if (sched_setscheduler(0, SCHED_BATCH, &param) != 0) {
            std::fprintf(stderr,
                "wp: WP_CPU_TIER_PIN: sched_setscheduler(SCHED_BATCH) failed: %s\n",
                std::strerror(errno));
        }
    }

    std::fprintf(stderr,
        "wp: WP_CPU_TIER_PIN=1: CPU expert tier executor thread pinned; "
        "cpus=[%s] prio=%s\n",
        n > 0 ? cpus.c_str() : "unpinned",
        cpu_tier_prio() == GGML_SCHED_PRIO_LOW ? "low(SCHED_BATCH)" : "normal");
#endif
}

// Configure a CPU backend that will carry the CPU EXPERT TIER.
//
// `tier` must be true ONLY for the backend of the "CPU" expert device -- the one
// driven by its own DeviceExecutor thread. It must be FALSE for the per-device
// cpu_backend_ used by compute_cpu_on_arrival(), which runs on a GPU device's
// executor thread: ggml applies prio/affinity to the OpenMP team INCLUDING
// thread 0, which under libgomp IS the calling thread, and nothing restores it
// afterwards. Confining a GPU device's submission thread to the CPU tier's
// cores and dropping it to SCHED_BATCH -- permanently, from a fallback path --
// is exactly the starvation we are trying to remove.
void configure_cpu_backend(ggml_backend_t backend, bool tier) {
    if (backend == nullptr) { return; }
    if (!tier || !cpu_tier_overlap_enabled()) {
        ggml_backend_cpu_set_n_threads(backend, cpu_worker_n_threads());
        return;
    }
    const int n_threads = cpu_tier_overlap_n_threads();
    // n_threads MUST match what the backend asks ggml_graph_compute() for:
    // under OpenMP the team is sized from cplan->n_threads but indexes
    // threadpool->workers[omp_get_thread_num()], so a smaller pool is an
    // out-of-bounds read.
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    // The pool is intentionally never freed: it belongs to a process-lifetime
    // backend, and under OpenMP it owns no threads -- it is a params carrier
    // ggml_graph_compute() reads on every graph, so it must outlive all of them.
    ggml_threadpool_params params;
    ggml_threadpool_params_init(&params, n_threads);
    cpu_tier_cpumask(params.cpumask);
    params.strict_cpu = cpu_tier_strict();
    params.prio       = cpu_tier_prio();
    params.paused     = false;
    ggml_threadpool_t tp = ggml_threadpool_new(&params);
    if (tp == nullptr) {
        std::fprintf(stderr,
            "wp: WP_CPU_TIER_OVERLAP: failed to create the confined CPU-tier "
            "threadpool; falling back to the default (unconfined) pool\n");
        return;
    }
    ggml_backend_cpu_set_threadpool(backend, tp);

    std::string cpus;
    for (int c = 0; c < GGML_MAX_N_THREADS; ++c) {
        if (params.cpumask[c]) {
            if (!cpus.empty()) { cpus += ","; }
            cpus += std::to_string(c);
        }
    }
    std::fprintf(stderr,
        "wp: WP_CPU_TIER_OVERLAP=1: CPU expert tier overlaps the GPU tiers; "
        "threads=%d cpus=[%s] strict=%d prio=%s\n",
        n_threads, cpus.empty() ? "default" : cpus.c_str(),
        (int) params.strict_cpu,
        params.prio == GGML_SCHED_PRIO_LOW ? "low(SCHED_BATCH)" : "normal");
}

// WP_EXPERT_PARTIAL_DTYPE=f32|f16 (default f32, CONFIG NOT HARDCODE). Wire
// encoding of the PARTIAL this worker sends back to the spine after summing
// its own subset of a layer's routed experts -- see the dtype note on
// pipe_expert_partial in pipe-protocol.h for the full history and the wire
// format. THE WORKER DECIDES: this is read once, here, at worker startup, and
// stamped onto every response.dtype in Worker::dispatch() below. The spine
// never needs its own copy of this knob -- pipe_encode_expert_partial() tags
// the frame with whatever this resolves to, and pipe_decode_expert_partial()
// on the spine side decodes strictly from that tag, so a worker that sets
// WP_EXPERT_PARTIAL_DTYPE=f16 and a spine that has never heard of the env var
// still interoperate correctly. That self-describing tag is also why this is
// safe to flip on PER WORKER: two workers on the same layer's expert split can
// disagree about dtype (e.g. the two remote 1GbE workers run f16, the local one
// stays f32) and the spine's scatter_add still sums correct f32 values from
// both, because each partial is converted to f32 on receipt regardless of how
// it arrived.
//
// WHY THIS IS OPT-IN AND DEFAULT OFF: f16 partials reintroduce the same
// quantization the 2026-08-04 f32 fix (see pipe_expert_partial) was written to
// remove -- rounding a worker's subtotal to an 11-bit mantissa (~5e-4 relative)
// AT THE EXPERT-TO-WORKER PARTITION BOUNDARY, which the hyper-connection gates
// and the discontinuous router top-k can amplify into a different generated
// token at temperature 0. That risk does not go away with the dtype tag; the
// tag only makes it IMPOSSIBLE for the wire to silently misinterpret bytes,
// not impossible for f16 rounding to perturb a sum. Acceptable to trade for
// halved bytes on a bandwidth-bound 1GbE hop under f16 KV; risky under turbo4
// KV, where the same amplification mechanism that motivated the f32 default
// lives. Hence: opt-in, per-worker, and off unless a human explicitly asks for
// it on the specific link that needs it.
// WP_EXPERT_PARTIAL_DTYPE REMOVED 2026-08-19. The knob let a worker send its
// expert partial as f16, rounding the subtotal to an 11-bit mantissa AT THE
// EXPERT->WORKER PARTITION BOUNDARY -- and which expert lands on which worker
// moves with batch width, so the model's output depended on the assignment. It
// changed generated text at temperature 0. The self-describing dtype tag made
// that DETECTABLE, never SAFE. It bought ~16 KiB per layer per remote worker
// (~0.13 ms) and measured -10% decode besides. DO NOT REINTRODUCE.
//
// The WIRE FORMAT is unchanged and PIPE_VERSION is NOT bumped: the spine still
// decodes an f16 partial correctly, because a worker built from an older commit
// may still be sending one during a rolling restart. What is gone is this
// process's ability to PRODUCE one.

backend_ptr init_backend(const std::string & device) {
    std::string lower = device;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    ggml_backend_t backend = nullptr;
    if (lower == "cpu") {
        backend = ggml_backend_cpu_init();
        // tier=true: this IS the CPU expert device's backend, the one
        // WP_CPU_TIER_OVERLAP confines. See configure_cpu_backend().
        configure_cpu_backend(backend, /* tier = */ true);
    } else {
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name(device.c_str(), nullptr);
    }
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize device " + device);
    }
    // WP_SELF_BENCH=2 benchmarks HERE -- backend freshly created, no slot pool,
    // no staging pool, nothing else in the process. WP_SELF_BENCH=1 benchmarks
    // in the Worker ctor AFTER both pools exist. The pair bisects the ctor:
    // fast here + slow there => a pool is responsible; slow in both => the
    // backend/process is slow from birth. Standalone reference is ~190 us on
    // Vulkan0, and the post-pool measurement is 1327 us (2026-08-01).
    {
        const char * mode = std::getenv("WP_SELF_BENCH");
        if (mode != nullptr && mode[0] == '2') {
            run_self_bench_early(backend);
        }
    }
    return backend_ptr(backend);
}


// WP_SELF_BENCH=1: time a STATIC pre-built expert graph inside THIS process,
// on THIS backend instance, before serving any request.
//
// WHY THIS EXISTS. A standalone benchmark rebuilding the same graph at the same
// shape measures ~190 us/compute on the RX 480, but the worker's FASTEST of
// 1996 live submits is 618 us and its median is ~1800 us (2026-08-01). Twelve
// reconstructed differences were tested and none reproduced the gap: shader,
// per-graph overhead, buffer layout, 400 live buffers, graph rotation, gallocr
// realloc per request, readback, H2D (per-BYTE, ~36 us weighted), backend
// construction, within-process degradation, second-GPU and CPU contention.
// So the remaining difference is either the worker's PER-REQUEST GRAPH or its
// PROCESS/BACKEND STATE, and a standalone binary cannot tell those apart.
// This does: same process, same backend, same device, static graph.
//   ~190 us  -> the backend is fine; the per-request graph path is the cost.
//   ~618 us  -> the graph is fine; something about this process/backend is.
// Persistent handle for the periodic in-serving probe (WP_SELF_BENCH_EVERY=N).
// The static graph is built once and re-timed every N requests WHILE the worker
// serves real traffic. It separates two things the live ns_submit cannot:
//   static stays ~190 us while real requests cost ~647 us => the difference is
//     the worker's per-request GRAPH CONTENT (cpy into io buffer, gallocr-owned
//     routing weights, attach_weight into slot buffers).
//   static ALSO degrades to ~647 us => it is process/backend state under load
//     (in-flight transfers, reader threads, queue depth), not the graph.
struct SelfBenchProbe {
    ggml_backend_t       backend = nullptr;
    ggml_context *        wctx  = nullptr;
    ggml_backend_buffer_t wbuf  = nullptr;
    ggml_context *        gctx  = nullptr;
    ggml_cgraph *         graph = nullptr;
    ggml_gallocr_t        ga    = nullptr;
    uint64_t              min_ns = UINT64_MAX;
    uint64_t              total_ns = 0;
    uint64_t              n = 0;
    bool                  ready = false;
};
static SelfBenchProbe g_probe;

void run_self_bench(ggml_backend_t backend, uint32_t n_embd, uint32_t n_ff) {
    const char * env = std::getenv("WP_SELF_BENCH");
    if (env == nullptr || env[0] != '1') {
        return;
    }
    const ggml_init_params wp_params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 16,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * wctx = ggml_init(wp_params);
    ggml_tensor * gate  = ggml_new_tensor_2d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff);
    ggml_tensor * up    = ggml_new_tensor_2d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff);
    ggml_tensor * down  = ggml_new_tensor_2d(wctx, GGML_TYPE_MXFP4, n_ff,  n_embd);
    ggml_tensor * input = ggml_new_tensor_2d(wctx, GGML_TYPE_F32,   n_embd, 1);
    ggml_tensor * rw    = ggml_new_tensor_2d(wctx, GGML_TYPE_F32,   1,      1);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (wbuf == nullptr) {
        std::cout << "wp self-bench: alloc failed" << std::endl;
        ggml_free(wctx);
        return;
    }
    {
        std::vector<uint8_t> junk(ggml_nbytes(gate), 0x5a);
        ggml_backend_tensor_set(gate, junk.data(), 0, ggml_nbytes(gate));
        ggml_backend_tensor_set(up,   junk.data(), 0, ggml_nbytes(up));
        ggml_backend_tensor_set(down, junk.data(), 0, ggml_nbytes(down));
        std::vector<float> f(n_embd, 0.01f);
        ggml_backend_tensor_set(input, f.data(), 0, ggml_nbytes(input));
        const float one = 0.5f;
        ggml_backend_tensor_set(rw, &one, 0, sizeof(float));
    }
    const size_t nodes = 32;
    const ggml_init_params gp = {
        /* .mem_size   = */ ggml_tensor_overhead() * nodes
                            + ggml_graph_overhead_custom(nodes, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * g = ggml_mul_mat(gctx, gate, input);
    ggml_tensor * u = ggml_mul_mat(gctx, up,   input);
    ggml_tensor * h = ggml_swiglu_split(gctx, g, u);
    ggml_tensor * o = ggml_mul_mat(gctx, down, h);
    ggml_tensor * w = ggml_mul(gctx, o, rw);
    ggml_cgraph * graph = ggml_new_graph_custom(gctx, nodes, false);
    ggml_build_forward_expand(graph, w);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, graph)) {
        std::cout << "wp self-bench: graph alloc failed" << std::endl;
        ggml_gallocr_free(ga); ggml_free(gctx);
        ggml_backend_buffer_free(wbuf); ggml_free(wctx);
        return;
    }
    for (int i = 0; i < 5; ++i) {          // warm up: first Vulkan submit compiles pipelines
        ggml_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);
    uint64_t best = UINT64_MAX, total = 0;
    const int iters = 300;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (ns < best) best = ns;
        total += ns;
    }
    std::cout << "wp self-bench: static 1-expert graph on this backend"
              << " iters=" << iters
              << " min_us=" << best / 1000
              << " mean_us=" << total / iters / 1000
              << std::endl;
    if (std::getenv("WP_SELF_BENCH_EVERY") != nullptr) {
        g_probe.backend = backend;
        g_probe.wctx = wctx; g_probe.wbuf = wbuf;
        g_probe.gctx = gctx; g_probe.graph = graph; g_probe.ga = ga;
        g_probe.ready = true;
        return;   // deliberately leaked for the process lifetime; diagnostic only
    }
    ggml_gallocr_free(ga);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
}

// One timed compute of the static graph, called from the serving path.

std::vector<ResourcePage> resource_pages(
        const Catalog & catalog,
        const std::function<bool(int, int)> & page_owner = {}) {
    std::vector<ResourcePage> result;
    result.reserve(catalog.pages.size());
    for (const auto & item : catalog.pages) {
        if (!page_owner || page_owner(item.second.layer, item.second.expert)) {
            result.push_back({ item.second.layer, item.second.device_size,
                               item.second.is_resident, item.second.size });
        }
    }
    return result;
}

Catalog & layout_sliced_pages(
        Catalog & catalog, ggml_backend_buffer_type_t buft) {
    // Whole-expert pages used to skip this and keep device_size == blob size.
    // CUDA MMQ over-reads the last quantized row (qwen4exp down Q5_1 ne0=640)
    // into MATRIX_ROW_PADDING; without slack that is the next slot and dim
    // 2559 becomes NaN. Apply the same layout to unsliced catalogs.

    const uint64_t alignment = ggml_backend_buft_get_alignment(buft);
    if (alignment == 0) {
        throw std::runtime_error("invalid expert slice device alignment");
    }
    uint64_t slot_alignment = alignment;
    const char * const arena_env = std::getenv("WP_EXPERT_ARENA_ID");
    const bool arena_requested =
        arena_env != nullptr && std::strtol(arena_env, nullptr, 10) == 1;
    if (arena_requested) {
        // CUDA/HIP converts quantized nb[2] from bytes to blocks. Keep that conversion exact.
        for (const auto & layer : catalog.descriptor.layers) {
            for (const auto & role : layer.second) {
                const uint64_t type_size = ggml_type_size(role.second.type);
                if (type_size == 0) {
                    throw std::runtime_error("invalid expert arena role type size");
                }
                const uint64_t divisor = std::gcd(slot_alignment, type_size);
                if (slot_alignment > UINT64_MAX / (type_size / divisor)) {
                    throw std::overflow_error("expert arena slot alignment overflows");
                }
                slot_alignment *= type_size / divisor;
            }
        }
    }
    for (auto & item : catalog.pages) {
        ExpertPage & page = item.second;
        const auto & specs = catalog.descriptor.layers.at(page.layer);
        std::vector<std::pair<std::string, MemberSpan *>> members;
        members.reserve(page.roles.size());
        for (auto & role : page.roles) {
            members.emplace_back(role.first, &role.second);
        }
        std::sort(members.begin(), members.end(),
                  [](const auto & a, const auto & b) {
                      return a.second->offset < b.second->offset;
                  });

        context_ptr ctx(ggml_init({
            /* .mem_size = */ ggml_tensor_overhead() * members.size(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc = */ true,
        }));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert slice layout metadata");
        }
        std::vector<uint64_t> allocation_sizes;
        allocation_sizes.reserve(members.size());
        for (const auto & member : members) {
            const RoleSpec & spec = specs.at(member.first);
            ggml_tensor * tensor =
                ggml_new_tensor_2d(ctx.get(), spec.type, spec.ne0, spec.ne1);
            size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
            if (alloc_size < member.second->size) {
                throw std::runtime_error("invalid expert slice device allocation size");
            }
            // MAD-LAB 2026-08-26: RESERVE QUANTIZED ROW SLACK ON EVERY BACKEND.
            //
            // CUDA/ROCm pads a quantized tensor up to MATRIX_ROW_PADDING (512)
            // elements precisely "to avoid out-of-bounds memory accesses" from
            // the quantized matmul kernels. Vulkan's
            // ggml_backend_vk_buffer_type_get_alloc_size returns a bare
            // ggml_nbytes and reserves NO slack at all -- so on Vulkan an
            // over-reading kernel walks straight into the NEXT expert packed
            // behind it in the slot, decodes those bytes as f16 block scales,
            // and an exponent-all-ones pattern there is NaN/Inf. That is the
            // RX 480 (:8804) returning non-finite partials while the same
            // shard bytes are provably byte-identical to the source GGUF.
            //
            // Reserve the slack ourselves rather than trusting the backend to.
            // The buffer is zeroed once at allocation and nothing ever writes
            // this tail, so an over-read now lands in zeros -- which is the
            // guarantee CUDA's padding was already providing.
            if (ggml_is_quantized(spec.type) && spec.ne0 % 512 != 0) {
                alloc_size += ggml_row_size(spec.type, 512 - (spec.ne0 % 512));
            }
            allocation_sizes.push_back((uint64_t) alloc_size);
        }
        const std::vector<DeviceMemberLayout> layout =
            plan_device_member_layout(allocation_sizes, alignment);
        uint64_t device_size = 0;
        for (size_t i = 0; i < members.size(); ++i) {
            members[i].second->device_offset = layout[i].offset;
            members[i].second->device_bytes  = layout[i].size;
            if (layout[i].offset > UINT64_MAX - layout[i].size) {
                throw std::overflow_error("expert slice device layout overflows");
            }
            device_size = std::max(device_size, layout[i].offset + layout[i].size);
        }
        if (arena_requested) {
            if (device_size > UINT64_MAX - (slot_alignment - 1)) {
                throw std::overflow_error("expert slice device size overflows");
            }
            device_size = (device_size + slot_alignment - 1) / slot_alignment * slot_alignment;
        }
        // The slot has to hold what the READ delivers, and a sliced page is
        // padded to the O_DIRECT alignment -- page.size includes that padding
        // while the layout above is computed purely from the role tensors and
        // does not. Without this clamp the slot is sized to the aligned payload
        // (e.g. 2,508,936) while the read is the padded page (2,510,848), and
        // select_victim finds nothing that fits: "no expert slot can hold
        // requested page", on exactly the 5 layers that carry padding.
        page.device_size = std::max(device_size, page.size);
    }
    return catalog;
}

template <typename F>
void for_each_page_chunk(
        const ExpertPage & page, size_t page_offset, size_t size, F && fn) {
    if (page_offset > page.size || size > page.size - page_offset) {
        throw std::runtime_error("expert page transfer is outside the blob page");
    }
    const uint64_t range_end = (uint64_t) page_offset + size;
    for (const auto & role : page.roles) {
        const MemberSpan & member = role.second;
        const uint64_t member_end = member.offset + member.size;
        const uint64_t begin = std::max<uint64_t>(page_offset, member.offset);
        const uint64_t end = std::min(range_end, member_end);
        if (begin < end) {
            fn((size_t) (begin - page_offset),
               (size_t) (member.device_offset + begin - member.offset),
               (size_t) (end - begin));
        }
    }
}

void zero_quantized_member_padding(ggml_tensor * slot_raw, const ExpertPage & page) {
    // CUDA MMQ over-reads the last quantized row into MATRIX_ROW_PADDING.
    // slot_raw is I8 covering the whole slot, so this write is in-bounds
    // even though the weight tensor's ggml_nbytes does not include the pad.
    if (slot_raw == nullptr) {
        return;
    }
    for (const auto & role : page.roles) {
        const MemberSpan & member = role.second;
        if (member.device_bytes <= member.size) {
            continue;
        }
        const size_t pad_off = (size_t) (member.device_offset + member.size);
        const size_t pad_len = (size_t) (member.device_bytes - member.size);
        ggml_backend_tensor_memset(slot_raw, 0, pad_off, pad_len);
    }
}

void tensor_set_page_range(
        ggml_tensor * tensor, const ExpertPage & page, const void * source,
        size_t page_offset, size_t size) {
    for_each_page_chunk(page, page_offset, size,
                        [tensor, source](size_t source_offset, size_t device_offset, size_t n) {
        ggml_backend_tensor_set(
            tensor, (const char *) source + source_offset, device_offset, n);
    });
    zero_quantized_member_padding(tensor, page);
}

void tensor_get_page(
        ggml_tensor * tensor, const ExpertPage & page, void * destination) {
    for_each_page_chunk(page, 0, (size_t) page.size,
                        [tensor, destination](size_t destination_offset, size_t device_offset, size_t n) {
        ggml_backend_tensor_get(
            tensor, (char *) destination + destination_offset, device_offset, n);
    });
}

// WP_READER_H2D: same chunking as tensor_set_page_range (for_each_page_chunk,
// so bytes/layout are byte-identical), but through the dedicated
// non-blocking-stream weak symbol instead of ggml_backend_tensor_set --
// see the ggml_backend_cuda_wp_reader_copy declaration and its definition
// in ggml-cuda.cu for why calling ggml_backend_tensor_set from a reader
// thread is unsafe (the gfx1201/MAD-114 legacy-stream capture hazard).
// Throws if the symbol is unresolved (non-CUDA/HIP build -- should never
// be reached; the reader_h2d_enabled_ gate already checked the backend) or
// if the copy itself fails, exactly like a read error, so the caller
// (reader_h2d_upload) can carry it in ReadResult::error.
void tensor_set_page_range_reader(
        ggml_backend_t backend, ggml_tensor * tensor, const ExpertPage & page,
        const void * source, size_t page_offset, size_t size) {
    if (ggml_backend_cuda_wp_reader_copy == nullptr) {
        throw std::runtime_error("wp reader H2D: ggml_backend_cuda_wp_reader_copy is unresolved");
    }
    for_each_page_chunk(page, page_offset, size,
                        [&](size_t source_offset, size_t device_offset, size_t n) {
        if (!ggml_backend_cuda_wp_reader_copy(
                backend, tensor, (const char *) source + source_offset, device_offset, n)) {
            throw std::runtime_error("wp reader H2D: reader-stream copy failed");
        }
    });
    zero_quantized_member_padding(tensor, page);
}

// WP_READER_H2D_VERIFY=1 tripwire: after a successful reader-thread upload,
// D2H-read a small sample back and memcmp it against the staging bytes that
// were just uploaded. Uses plain ggml_backend_tensor_get -- NOT a new HIP
// helper -- because get_tensor's CUDA/HIP implementation
// (ggml_backend_cuda_buffer_get_tensor) always uses cudaStreamPerThread,
// never the legacy stream, on every arch including gfx1201: there is no
// MAD-114 branch on the get side, so it is already capture-safe to call
// from a reader thread with no changes. Cheap and approximate by design
// (a diagnostic, not an exhaustive check): samples only the first and last
// up to 4 KiB of the PAGE range, and assumes that range does not itself
// straddle a page-member boundary whose device offset diverges from its
// source offset (true for the common single-member/contiguous page; a
// scattered multi-member page could under-sample near a boundary, which
// only weakens the tripwire, never produces a false positive against
// correctly-uploaded bytes).
void tensor_verify_page_range(
        ggml_tensor * tensor, const ExpertPage & page, const void * source,
        size_t page_offset, size_t size, uint64_t & mismatches) {
    for_each_page_chunk(page, page_offset, size,
                        [&](size_t source_offset, size_t device_offset, size_t n) {
        std::vector<uint8_t> readback(n);
        ggml_backend_tensor_get(tensor, readback.data(), device_offset, n);
        if (std::memcmp(readback.data(), (const char *) source + source_offset, n) != 0) {
            ++mismatches;
        }
    });
}

struct free_deleter {
    void operator()(void * p) const {
        std::free(p);
    }
};

// WP_STAGING_PINNED=1 opts IN to page-locked staging via
// ggml_backend_dev_host_buffer_type. DEFAULT IS OFF (posix_memalign) because
// measurement on 2026-07-31 showed pinning is INERT AND UNSAFE:
//   - inert: 1070 gb_s_h2d 2.971 pinned vs 3.006 pageable; R9700 14.99 vs
//     16.10; throughput 0.865 vs 0.889 tok/s. All inside the +/-3% band. The
//     "1.29 GB/s H2D" that motivated this was DERIVED from a subtraction, never
//     measured; the first real ns_h2d says the 1070 was always at ~85% of its
//     gen3 x4 ceiling. There was no bounce-copy cost to recover.
//   - unsafe: Vulkan's host buffer type returns memory that is 4096-aligned but
//     NOT O_DIRECT-readable (host-visible device/BAR memory, not host RAM).
//     read() returns -1 and prefill dies at layer 3. The alignment and
//     buffer-type checks below both PASS on it, so the pool reports
//     staging_kind=pinned truthfully and then fails on first read.
// Read at startup only (not a struct Options field) so it cannot reintroduce
// the ABI mismatch that broke every worker on 2026-07-30 -- and so it stays
// settable per worker process, which is what made the A/B above possible.
bool staging_pinned_env_enabled() {
    const char * env = std::getenv("WP_STAGING_PINNED");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

class StagingPool {
public:
    StagingPool(const ResourcePlan & resources, ggml_backend_t backend) :
        buffer_bytes_(resources.staging_buffer_bytes),
        buffer_count_(resources.staging_buffers) {
        buffers_.reserve((size_t) resources.staging_buffers);
        host_buffers_.reserve((size_t) resources.staging_buffers);
        available_.reserve((size_t) resources.staging_buffers);

        ggml_backend_buffer_type_t host_buft = nullptr;
        bool try_pinned = staging_pinned_env_enabled();
        if (try_pinned && backend != nullptr) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev != nullptr) {
                // Portable entry point (ggml-backend.h:187). Never call a
                // backend-specific host-alloc symbol here: this worker binary
                // serves ROCm, CUDA and Vulkan, and #if GGML_USE_* branching
                // in this path is the exact bug class that has recurred six
                // times in this codebase.
                host_buft = ggml_backend_dev_host_buffer_type(dev);
            }
        }

        // Some backends provide no host buffer type; the allocation can also
        // fail or come back misaligned. The O_DIRECT reads need 4096-byte
        // alignment, so verify it now rather than discovering EINVAL at
        // read() time. Any of these cases falls back to posix_memalign for
        // the whole pool.
        //
        // Also verify the returned buffer's actual type matches host_buft.
        // ggml_backend_cuda_host_buffer_type_alloc_buffer silently falls back
        // to a plain CPU buffer (ggml_backend_cpu_buffer_from_ptr) when
        // cudaHostAlloc fails (e.g. GGML_CUDA_NO_PINNED, or the allocator is
        // just out of pinned memory) -- and only stamps buffer->buft with the
        // host type on the success path, so the fallback buffer keeps
        // ggml_backend_cpu_buffer_type(). That buffer is non-null, has a
        // valid base pointer, and is 4096-aligned, so the checks above all
        // pass on it even though the memory is pageable. Without this check
        // we would report staging_kind=pinned while actually running
        // pageable, and derive gb_s_h2d against a mechanism that never ran.
        bool pinned = try_pinned && host_buft != nullptr;
        if (pinned) {
            for (int i = 0; i < resources.staging_buffers; ++i) {
                buffer_ptr buf(ggml_backend_buft_alloc_buffer(
                    host_buft, (size_t) buffer_bytes_));
                void * raw = buf ? ggml_backend_buffer_get_base(buf.get()) : nullptr;
                if (!buf || raw == nullptr ||
                        (reinterpret_cast<uintptr_t>(raw) % DIRECT_ALIGNMENT) != 0 ||
                        ggml_backend_buffer_get_type(buf.get()) != host_buft) {
                    pinned = false;
                    break;
                }
                host_buffers_.push_back(std::move(buf));
            }
            if (!pinned) {
                host_buffers_.clear();
            }
        }

        if (pinned) {
            for (const buffer_ptr & buf : host_buffers_) {
                available_.push_back(ggml_backend_buffer_get_base(buf.get()));
            }
        } else {
            for (int i = 0; i < resources.staging_buffers; ++i) {
                void * raw = nullptr;
                if (posix_memalign(
                        &raw, DIRECT_ALIGNMENT, (size_t) buffer_bytes_) != 0) {
                    throw std::runtime_error(
                        "failed to allocate aligned O_DIRECT staging buffer");
                }
                buffers_.emplace_back(raw);
                available_.push_back(raw);
            }
        }

        pinned_ = pinned;
        backend_ = backend;
        device_  = backend_ != nullptr ? ggml_backend_get_device(backend_) : nullptr;
        copy_stream_h2d_ = backend_ != nullptr && device_ != nullptr &&
            ggml_backend_cuda_wp_copy_stream_enabled != nullptr &&
            ggml_backend_cuda_wp_copy_stream_enabled(backend_);
        // WP_EXPERT_ASYNC_H2D=1 -- opt IN. Default OFF: 2026-08-07 it lost
        // decode (3.82/3.67 vs 3.94) because a copy on the COMPUTE stream
        // cannot overlap the graph. The dedicated-stream path is
        // WP_EXPERT_COPY_STREAM=1. Vulkan stays OFF even when =1: it killed
        // :8804 mid-prefill. Drain uses tensor_set_async on the compute
        // stream; mark_in_flight / borrow() fence staging reuse.
        const char * async_env = std::getenv("WP_EXPERT_ASYNC_H2D");
        if (async_env != nullptr && async_env[0] != '\0' && async_env[0] != '0' &&
                backend != nullptr) {
            // RUNTIME name check, not a GGML_USE_* macro (that bug class has
            // recurred six times here). Vulkan ADVERTISES events, so the probe
            // below passes -- but arming this path on the RX 480 killed the
            // worker mid-prefill on 2026-08-07 (spine: "worker :8804 died
            // while computing", llama_decode ret=-3). Until the vk async-set +
            // event-fence semantics are understood, Vulkan stays on the sync
            // path by name.
            const char * bname = ggml_backend_name(backend);
            const bool vulkan =
                bname != nullptr && std::strstr(bname, "Vulkan") != nullptr;
            if (!vulkan) {
                ggml_backend_event_t probe =
                    device_ != nullptr ? ggml_backend_event_new(device_) : nullptr;
                if (probe != nullptr) {
                    ggml_backend_event_free(probe);
                    async_h2d_ = true;
                }
            }
        }
        // Logged once: one StagingPool is constructed per worker process.
        std::cerr << "wp expert worker: staging_kind="
                  << (pinned_ ? "pinned" : "pageable")
                  << " async_h2d=" << (async_h2d_ ? "on" : "off")
                  << " copy_stream=" << (copy_stream_h2d_ ? "on" : "off") << std::endl;
    }

    ~StagingPool() {
        for (auto & kv : events_) {
            ggml_backend_event_free(kv.second);
        }
    }

    // True when drain paths should use tensor_set_async + mark_in_flight.
    bool async_h2d() const { return async_h2d_; }

    bool copy_stream_h2d() const { return copy_stream_h2d_; }

    ggml_backend_event_t new_copy_event() const {
        return copy_stream_h2d_ && device_ != nullptr ? ggml_backend_event_new(device_) : nullptr;
    }

    bool record_copy_event(ggml_backend_event_t event) const {
        return event != nullptr && ggml_backend_cuda_wp_copy_stream_record_event != nullptr &&
               ggml_backend_cuda_wp_copy_stream_record_event(backend_, event);
    }

    // Record "an async copy out of this staging buffer is in flight" on the
    // backend stream. Dispatch-thread only (same thread as the async issue).
    void mark_in_flight(void * data) {
        if (!async_h2d_ && !copy_stream_h2d_) {
            return;
        }
        if (backend_ == nullptr || device_ == nullptr) {
            async_h2d_ = false;
            copy_stream_h2d_ = false;
            return;
        }
        ggml_backend_event_t ev = nullptr;
        {
            // The map is shared with borrow(); the record itself is not.
            std::lock_guard<std::mutex> lock(mutex_);
            ggml_backend_event_t & slot = events_[data];
            if (slot == nullptr) {
                slot = ggml_backend_event_new(device_);
            }
            ev = slot;
        }
        if (ev == nullptr) {
            // Device refused an event at runtime -- the copy already issued,
            // so the only safe fallback is a full backend sync now, and the
            // feature disarms for the rest of the process.
            async_h2d_ = false;
            ggml_backend_synchronize(backend_);
            return;
        }
        if (copy_stream_h2d_ && ggml_backend_cuda_wp_copy_stream_record_event != nullptr) {
            if (!ggml_backend_cuda_wp_copy_stream_record_event(backend_, ev)) {
                copy_stream_h2d_ = false;
                ggml_backend_synchronize(backend_);
            }
        } else {
            ggml_backend_event_record(ev, backend_);
        }
    }

    // *** PER-CONNECTION STAGING QUOTA -- THE 2026-08-25 MULTI-CONN DEADLOCK
    // FIX. ***
    //
    // Live stacks that day (WP_WORKER_MULTI_CONN=2, WP_EXPERT_STAGING_BUFFERS=32)
    // showed:
    //   - connection thread X: holding g_worker_gpu_mutex inside
    //     finish_split_dispatch -> Worker::dispatch -> drain_one_read, parked
    //     in batch.state_->cv.wait() for a ReadResult.
    //   - connection thread Y: parked in pthread_mutex_lock at the top of
    //     serve_connection's loop, waiting for the SAME g_worker_gpu_mutex.
    //   - every reader thread (ensure_batch's read_worker/stripe_read_worker
    //     pool): parked in borrow() below, because available_ was EMPTY.
    // Mechanism: a Lease travels inside ReadResult::staging and is only
    // released once drain_one_read performs its H2D upload, which runs under
    // g_worker_gpu_mutex. Connection Y's earlier BEGIN had already spawned
    // readers that finished their reads and queued them on Y's batch, each
    // holding a lease -- but Y cannot drain them (and free those leases)
    // until it gets the mutex, which X holds while X's OWN readers are stuck
    // in borrow() because Y's undrained results have pinned every buffer.
    // Hold-and-wait across two independent transactions sharing one bounded
    // pool -- classic deadlock, and impossible in single-connection mode
    // (there is only ever one transaction).
    //
    // THE INVARIANT: in multi-conn mode (quota_ < buffer_count_), each
    // connection may hold at most `quota_` leases at once --
    // floor(buffer_count_ / N) for N connections. That bounds how many
    // buffers ANY one connection's undrained results can pin, so the other
    // connection's readers can never be starved down to zero -- there is
    // always at least one buffer outside every other connection's quota for
    // this connection's own readers to make progress with, which lets this
    // connection's own drain (the only thing that can free ITS leases) always
    // run to completion and release them. Single-connection mode
    // (conn_index == -1, or quota_ == buffer_count_) is untouched: quota_
    // defaults to buffer_count_, so the wait predicate is exactly
    // `!available_.empty()`, byte-identical to before this fix.
    class Lease {
    public:
        Lease(Lease && other) noexcept :
            owner_(other.owner_), data_(other.data_), conn_index_(other.conn_index_) {
            other.owner_ = nullptr;
            other.data_  = nullptr;
        }

        ~Lease() {
            if (owner_ != nullptr) {
                owner_->release(data_, conn_index_);
            }
        }

        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;
        Lease & operator=(Lease &&) = delete;

        void * get() const {
            return data_;
        }

    private:
        friend class StagingPool;

        Lease(StagingPool * owner, void * data, int conn_index) :
            owner_(owner), data_(data), conn_index_(conn_index) {
        }

        StagingPool * owner_ = nullptr;
        void *        data_  = nullptr;
        int           conn_index_ = -1;
    };

    // conn_index: -1 ("none") is the reserved value for a borrow issued
    // outside any connection's transaction (single-connection default path,
    // and the idle-pump host-landing/prefetch paths -- see the call sites at
    // spec_host_submit and spec_pagein_submit). It always draws from the
    // global pool with no quota, same as every borrow() before this fix.
    Lease borrow(int conn_index = -1) {
        void * result = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            available_cv_.wait(lock, [&]() {
                if (available_.empty()) {
                    return false;
                }
                if (conn_index < 0 || quota_ >= buffer_count_) {
                    return true;
                }
                return held_by_conn_[conn_index] < quota_;
            });
            result = available_.back();
            available_.pop_back();
            if (conn_index >= 0 && quota_ < buffer_count_) {
                ++held_by_conn_[conn_index];
                if ((size_t) conn_index < g_worker_staging_held.size()) {
                    g_worker_staging_held[(size_t) conn_index].fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        }
        // Async H2D: wait for any in-flight copy OUT of this buffer before a
        // reader thread refills it. Outside the pool lock -- the wait is on
        // the GPU, and holding the lock would serialize every other borrower
        // behind it.
        if (async_h2d_ || copy_stream_h2d_) {
            ggml_backend_event_t ev = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = events_.find(result);
                if (it != events_.end()) {
                    ev = it->second;
                }
            }
            if (ev != nullptr) {
                ggml_backend_event_synchronize(ev);
            }
        }
        return Lease(this, result, conn_index);
    }

    int buffer_count() const {
        return buffer_count_;
    }

    size_t cpu_lease_cap() {
        std::lock_guard<std::mutex> lock(mutex_);
        return quota_ > 0 ? (size_t) quota_ - 1 : 0;
    }

    uint64_t buffer_bytes() const {
        return buffer_bytes_;
    }

    // Which allocation path actually ran, not what was requested.
    bool pinned() const {
        return pinned_;
    }

    // Called once from run(), after WP_WORKER_MULTI_CONN=N is parsed and
    // BEFORE any connection thread starts (so no borrow() can race the quota
    // change). n<=1 (including the untouched default single-connection path)
    // restores quota_ == buffer_count_, i.e. no cap -- byte-identical to
    // before this fix. n>buffer_count_ cannot give every connection even one
    // guaranteed buffer, so rather than assert (and kill a worker over a
    // config typo) this falls back to the same uncapped N=1 semantics, loudly:
    // multi-conn is still safe (single connection ever ran that starting
    // config, whatever it was), the flag just does not get its double-
    // buffering behaviour until reconfigured.
    void set_multi_conn(int n) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (n <= 1) {
            quota_ = buffer_count_;
            return;
        }
        if (n > buffer_count_) {
            std::cerr << "wp expert worker: WARNING WP_WORKER_MULTI_CONN=" << n
                      << " exceeds staging buffer_count=" << buffer_count_
                      << "; per-connection staging quota disabled (falling back to "
                         "N=1 semantics -- see the 2026-08-25 deadlock comment on "
                         "StagingPool)" << std::endl;
            quota_ = buffer_count_;
            return;
        }
        quota_ = buffer_count_ / n;
    }

private:
    void release(void * data, int conn_index) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            available_.push_back(data);
            if (conn_index >= 0 && quota_ < buffer_count_) {
                auto it = held_by_conn_.find(conn_index);
                if (it != held_by_conn_.end() && it->second > 0) {
                    --it->second;
                }
                if ((size_t) conn_index < g_worker_staging_held.size()) {
                    g_worker_staging_held[(size_t) conn_index].fetch_sub(
                        1, std::memory_order_relaxed);
                }
            }
        }
        // notify_all, not notify_one: with a per-connection quota, waiters
        // block on DIFFERENT predicates (different conn_index's `held_by_conn_
        // < quota_`). A released buffer that satisfies waiter B's predicate
        // but not waiter A's must not be swallowed by waking only A, which
        // would then just go back to sleep while B stays parked forever.
        available_cv_.notify_all();
    }

    uint64_t                                        buffer_bytes_ = 0;
    int                                              buffer_count_ = 0;
    bool                                             pinned_ = false;
    std::vector<std::unique_ptr<void, free_deleter>> buffers_;
    std::vector<buffer_ptr>                         host_buffers_;
    std::vector<void *>                             available_;
    std::mutex                                      mutex_;
    std::condition_variable                         available_cv_;
    // Per-connection outstanding-lease counts backing the quota above.
    // Guarded by mutex_; quota_ defaults to buffer_count_ (no cap) until
    // set_multi_conn() runs. int, not size_t: it is compared directly
    // against quota_ (also int) in the wait predicate.
    std::unordered_map<int, int>                    held_by_conn_;
    int                                              quota_ = buffer_count_;
    // Async H2D (WP_EXPERT_ASYNC_H2D). events_ maps a staging buffer's base
    // pointer to the event that fences its last in-flight copy; guarded by
    // mutex_. async_h2d_ is atomic only because the runtime-disarm path in
    // mark_in_flight (dispatch thread) races borrow()'s read (reader threads)
    // -- either value is safe to observe there.
    std::atomic<bool>                               async_h2d_{false};
    std::atomic<bool>                               copy_stream_h2d_{false};
    ggml_backend_t                                  backend_ = nullptr;
    ggml_backend_dev_t                              device_  = nullptr;
    std::unordered_map<void *, ggml_backend_event_t> events_;
};

class ResidentExpertPool {
public:
    ResidentExpertPool(ggml_backend_t backend, Catalog & catalog,
                       const std::vector<int> & blocks,
                       const std::function<bool(int, int)> & page_owner = {}) :
        backend_(backend) {
        for (auto & item : catalog.pages) {
            ExpertPage & page = item.second;
            if (!std::binary_search(blocks.begin(), blocks.end(), page.layer) ||
                    (page_owner && !page_owner(page.layer, page.expert))) {
                continue;
            }
            Allocation allocation;
            allocation.buffer.reset(ggml_backend_alloc_buffer(
                backend_, (size_t) page.device_size));
            if (!allocation.buffer) {
                throw std::runtime_error("failed to allocate resident expert page");
            }
            ggml_backend_buffer_set_usage(
                allocation.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // MAD-LAB 2026-08-26: ZERO WEIGHT MEMORY ON ALLOCATION.
            // A quantized expert tensor is allocated at the backend's PADDED
            // size (ggml_backend_buft_get_alloc_size), but only its real
            // ggml_nbytes are ever written from the shard. ggml itself warns
            // the padding must be zeroed "to avoid possible NaN values"
            // (ggml-cuda.cu, ggml_backend_cuda_buffer_init_tensor) -- the
            // quantized matmul reads the padded tail, and garbage there is
            // decoded as f16 block scales, where a random exponent-all-ones
            // pattern is literally NaN/Inf. This worker never initialised ANY
            // device memory (no memset / no buffer_clear anywhere in the
            // file), and the one mechanism that would have zeroed the padding
            // -- init_tensor's memset inside attach_weight -- is not reliable
            // here: it aborts with "invalid argument" on this very allocation.
            // Only `down` is affected in practice (ne0=640, 640 %% 512 = 128,
            // so it pads; gate/up have ne0=2560 and do not), which is exactly
            // the role in that abort's stack.
            ggml_backend_buffer_clear(allocation.buffer.get(), 0);
            allocation.ctx.reset(ggml_init({
                /* .mem_size = */ ggml_tensor_overhead() * 2,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc = */ true,
            }));
            if (!allocation.ctx) {
                throw std::runtime_error("failed to allocate resident expert metadata");
            }
            allocation.raw = ggml_new_tensor_1d(
                allocation.ctx.get(), GGML_TYPE_I8, (int64_t) page.device_size);
            allocation.raw->buffer = allocation.buffer.get();
            allocation.raw->data = ggml_backend_buffer_get_base(allocation.buffer.get());
            if (allocation.raw->data == nullptr ||
                ggml_backend_buffer_init_tensor(
                    allocation.buffer.get(), allocation.raw) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("failed to initialize resident expert page");
            }

            void * host = nullptr;
            if (posix_memalign(&host, DIRECT_ALIGNMENT, (size_t) page.size) != 0) {
                throw std::runtime_error("failed to allocate resident expert staging");
            }
            try {
                read_once(page, host);
                tensor_set_page_range(
                    allocation.raw, page, host, 0, (size_t) page.size);
            } catch (...) {
                std::free(host);
                throw;
            }
            std::free(host);

            page.is_resident = true;
            page.resident_buffer = allocation.buffer.get();
            page.resident_base = allocation.raw->data;
            pinned_bytes_ += page.device_size;
            ++pinned_pages_;
            allocations_.push_back(std::move(allocation));
        }
        if (pinned_pages_ != 0) {
            ggml_backend_synchronize(backend_);
        }
    }

    uint64_t pinned_bytes() const { return pinned_bytes_; }
    int pinned_pages() const { return pinned_pages_; }

private:
    struct Allocation {
        buffer_ptr buffer;
        context_ptr ctx;
        ggml_tensor * raw = nullptr;
    };

    static void read_once(const ExpertPage & page, void * dst) {
#if defined(__linux__)
        const int fd = open(page.blob.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "failed to open resident expert shard " + page.blob.string() +
                ": " + std::strerror(errno));
        }
        ssize_t n = -1;
        do {
            n = pread(fd, dst, (size_t) page.size, (off_t) page.offset);
        } while (n < 0 && errno == EINTR);
        const int saved_errno = errno;
        close(fd);
        if (n < 0 || (uint64_t) n != page.size) {
            throw std::runtime_error(
                "short resident expert read from " + page.blob.string() +
                ": got " + std::to_string(n) + " want " +
                std::to_string(page.size) + " (" + std::strerror(saved_errno) + ")");
        }
#else
        (void) page;
        (void) dst;
        throw std::runtime_error("resident expert loading requires Linux");
#endif
    }

    ggml_backend_t backend_ = nullptr;
    std::vector<Allocation> allocations_;
    uint64_t pinned_bytes_ = 0;
    int pinned_pages_ = 0;
};

class ExpertSlotPool;
thread_local ExpertSlotPool * g_host_reader_pool = nullptr;

struct WorkerLogFiles {
    WorkerLogFiles() {
        if (const char * path = std::getenv("WP_PAGEIN_LOG")) {
            if (path[0] != '\0') {
                pagein = fopen(path, "w");
            }
        }
        if (const char * path = std::getenv("WP_HINT_LOG")) {
            if (path[0] != '\0') {
                hint = fopen(path, "w");
            }
        }
    }

    ~WorkerLogFiles() {
        if (pagein != nullptr) {
            fclose(pagein);
        }
        if (hint != nullptr) {
            fclose(hint);
        }
    }

    FILE * pagein = nullptr;
    FILE * hint = nullptr;
    std::mutex mutex;
};

class ExpertSlotPool {
private:
    struct PageIn {
        size_t             entry_index = 0;
        size_t             slot_index  = 0;
        const ExpertPage * page        = nullptr;
        int                fd          = -1;
        // Destination tensor for this page-in's H2D, captured HERE, at plan
        // time, under g_worker_gpu_mutex (see the pageins.push_back() call
        // site in ensure_batch). WP_READER_H2D reader threads copy through
        // this pointer instead of looking the slot up in slots_[slot_index]
        // -- slots_ is pool-wide state shared with the other connection and
        // must not be touched off the lock. raw is stable for the slot's
        // whole lifetime (arena-backed, non-owning -- see Slot::raw), so the
        // pointer stays valid for as long as the pin from ensure_batch keeps
        // this slot from being reassigned.
        ggml_tensor *      raw         = nullptr;
        // WP_READER_H2D eligibility for this page-in, decided ONCE per
        // ensure_batch() call (see the "reader_h2d_this_batch" local right
        // before batch.state_->pageins.push_back below) and copied onto
        // every PageIn of that batch. Deciding it once, at plan time, rather
        // than re-reading staging_.async_h2d()/copy_stream_h2d() from each
        // reader thread as it processes each stripe, guarantees every
        // stripe of one page agrees on whether the reader or drain_one_read
        // performs its H2D -- a stripe-by-stripe re-check could disagree
        // mid-page if the async knobs are runtime-disarmed between stripes,
        // silently under-uploading a page that still gets marked valid.
        bool               reader_h2d  = false;
        bool               cpu_on_arrival = false;
    };

    // One STRIPE of one page-in. A page is read in WP_EXPERT_READ_STRIPES
    // aligned pieces so the dispatch thread can upload stripe k while the
    // reader thread is still reading stripe k+1. With one stripe this is
    // exactly the old whole-page result.
    //
    // WHY THE LEASE IS SHARED: every stripe of a page reads into a different
    // offset of the SAME staging buffer, so the lease must outlive the last
    // stripe's upload, not the first.
    struct ReadResult {
        size_t                              pagein_indexb = 0;
        std::shared_ptr<StagingPool::Lease> staging;
        size_t                              offset = 0;   // byte offset within the page
        size_t                              len    = 0;   // bytes in this stripe
        bool                                last   = true; // last stripe of this page
        std::exception_ptr                  error;
        std::chrono::steady_clock::time_point read_started;
        std::chrono::steady_clock::time_point read_finished;
        bool                                read_timed = false;
        // WP_READER_H2D: true if this result's H2D was already performed
        // (or deliberately skipped as a no-op -- see below) on the reader
        // thread, so drain_one_read must not copy it again. Set on EVERY
        // stripe of a page once pagein.reader_h2d is true for that page,
        // not just the last one: a non-last stripe carries no bytes of its
        // own to upload (the page's bytes only become a complete, copyable
        // range once the LAST stripe lands), so it is marked uploaded with
        // h2d_bytes/h2d_ns left at 0 -- a pure no-op for drain_one_read,
        // which already skips its "if (result->last)" publish block for a
        // non-last result. Only the last stripe's result carries the real
        // whole-page copy (h2d_bytes == the page size) and its timing.
        bool                                uploaded = false;
        uint64_t                            h2d_ns    = 0;
        uint64_t                            h2d_bytes = 0;
    };

    // WP_EXPERT_STRIPE_PARALLEL=1: the read work unit becomes the STRIPE, not
    // the page, so several reader threads pull ONE page's stripes concurrently
    // (QD>1 against a drive that measured 6.2 GB/s sustained but ~3 GB/s at
    // QD1 -- the decode request's 4.5 ms ns_read is 1-2 pages read serially).
    // Stripes of one page share one staging lease; the LAST STRIPE TO COMPLETE
    // (an atomic countdown, not the last index) carries `last`, preserving the
    // exactly-once publish invariant in drain_one_read.
    struct StripeJob {
        size_t pagein_indexb = 0;
        size_t offset        = 0;
        size_t len           = 0;
    };
    struct PageShared {
        std::shared_ptr<StagingPool::Lease> lease;
        std::mutex                          lease_mutex;
        std::atomic<size_t>                 remaining{0};
        // Any stripe failed. The LAST completer inherits it so the page's
        // final ReadResult carries an error and drain never publishes a
        // half-read slot -- the serial path got this via failed-stripe-ends-
        // the-page, which parallel stripes cannot do.
        std::atomic<bool>                   failed{false};
    };

    struct BatchState {
        std::vector<PageIn>                       pageins;
        std::atomic<size_t>                     next{0};
        // Which connection's transaction this batch belongs to (-1 = none:
        // single-connection default path, or a speculative/prefetch batch
        // with no connection context). Threaded into every staging_.borrow()
        // call the reader threads make (read_worker/stripe_read_worker) so
        // the per-connection staging quota (StagingPool's 2026-08-25 deadlock
        // fix) can be enforced. Set once in ensure_batch, read-only after.
        int                                      conn_index = -1;
        // Stripe-parallel mode only; empty otherwise. Built once in
        // ensure_batch before the workers start, read-only afterwards.
        std::vector<StripeJob>                  stripe_jobs;
        std::vector<std::unique_ptr<PageShared>> page_shared;
        std::mutex                              mutex;
        std::condition_variable                 cv;
        std::deque<std::unique_ptr<ReadResult>> ready;
        bool                                    start  = false;
        bool                                    cancel = false;
        bool                                    measure = false;
        std::atomic<size_t>                     read_inflight{0};
        std::atomic<size_t>                     read_inflight_max{0};
        // True for a batch submitted by spec_pagein_submit. Its page-ins are
        // logged as "S" AT SUBMIT; drain_one_read must NOT also log them as
        // "D", or every harvested speculative read masquerades as the demand
        // read it exists to prevent. Under the async path the harvest runs
        // INSIDE ensure_batch -- i.e. AFTER the current request's "R" line --
        // so the classifier saw S..R..D for the same page and filed a USED
        // page as LATE. That artifact alone is what made the asynchronous
        // rewrite look like it had made the used-rate worse (686 -> 431).
        // Only the dispatch thread reads/writes it, and always after submit.
        bool                                    speculative = false;
        // Decode/verify demand reads land a host-tier copy from staging so
        // eviction does not need a sync D2H. Prefill (n_tokens>8) stays out
        // so a 108-pagein ubatch cannot wipe the 3 GiB decode working set.
        bool                                    admit_host_on_read = false;
    };

public:
    ExpertSlotPool(
            ggml_backend_t backend, ResourcePlan resources,
            uint64_t host_victim_bytes, TestHooks * test_hooks,
            const std::vector<int> & reserve_blocks, size_t page_count = 0,
            wp::HostTier * shared_host_tier = nullptr,
            WorkerLogFiles * logs = nullptr) :
        backend_(backend),
        resources_(std::move(resources)),
        staging_(resources_, backend),
        test_hooks_(test_hooks),
        cpu_on_arrival_enabled_(cpu_on_arrival_enabled_from_env()),
        cpu_on_arrival_cap_(cpu_on_arrival_cap_from_env()),
        owned_host_tier_(shared_host_tier == nullptr
            ? std::make_unique<wp::HostTier>() : nullptr),
        host_tier_(shared_host_tier != nullptr ? shared_host_tier : owned_host_tier_.get()) {
        reserve_blocks_ = reserve_blocks;
        logs_ = logs;
        pagein_log_ = logs_ != nullptr ? logs_->pagein : nullptr;
        std::sort(reserve_blocks_.begin(), reserve_blocks_.end());
        if (lfu_history_enabled_) {
            lfu_history_.assign(page_count, 0);
        }
        if (resources_.slot_count < 0 ||
            (resources_.slot_count == 0 && resources_.pinned_bytes == 0) ||
            (resources_.slot_count > 0 && resources_.staging_buffer_bytes == 0) ||
            resources_.staging_buffer_bytes >
                (uint64_t) std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("invalid expert slot pool dimensions");
        }
        // WP_SELF_BENCH=3 benchmarks HERE: staging_ is already built (member
        // init list) but NO slot buffers exist yet. With =2 (before everything)
        // measuring 183 us and =1 (after everything) measuring 1341 us, this
        // splits the constructor: fast here => the SLOT buffers do it; slow here
        // => the STAGING pool does it. Staging is a fixed 16 buffers regardless
        // of --slots, which matches the observed flatness across 100..400 slots.
        {
            const char * mode = std::getenv("WP_SELF_BENCH");
            if (mode != nullptr && mode[0] == '3') {
                setenv("WP_SELF_BENCH", "1", 1);
                run_self_bench(backend_, 4096, 2048);
                setenv("WP_SELF_BENCH", "3", 1);
            }
        }
        slots_.reserve((size_t) resources_.slot_count);
        resources_.device_bytes = 0;
        std::set<int> reserved_indices(resources_.reserved_slot_indices.begin(),
                                       resources_.reserved_slot_indices.end());
        allocate_slot_arenas();
        // *** PIN THE CPU EXPERT TIER INTO RAM -- NEVER SWAP IT. ***
        // The CPU device's slot arenas ARE host RAM (a CPU backend buffer is a
        // plain malloc). Under vm.swappiness=100 + zram the kernel was
        // compressing this whole ~3.3 GB tier into zram and decompressing it on
        // every access -- the CPU expert tier is supposed to be the fast RAM
        // fallback below NVMe, and swapping it defeats its entire purpose (a
        // zstd decompress of a 2.1 MB page is the same order as reading it back
        // off the SN850X). Every OTHER RAM tier here is already mlocked
        // (wp::HostTier under WP_PIN_HOST); this one simply never was.
        // mlock, not madvise: MADV_DONTNEED/COLD would be the wrong direction,
        // and only mlock is a hard guarantee the pages stay resident. Applied
        // to CPU backends only -- a GPU arena is VRAM and mlock does not apply.
        mlock_cpu_arenas();
        size_t arena_index  = 0;
        uint64_t arena_used = 0;
        for (size_t class_index = 0;
                class_index < resources_.slot_classes.size(); ++class_index) {
            const SlotClass & slot_class = resources_.slot_classes[class_index];
            if (resources_.slot_classes.size() > 1) {
                arena_index = arena_class_starts_.at(class_index);
                arena_used = 0;
            }
            for (int i = 0; i < slot_class.slots; ++i) {
                const uint64_t need = arena_slot_stride(slot_class.size);
                if (arena_index >= arenas_.size()) {
                    throw std::runtime_error("expert slot arenas exhausted");
                }
                uint64_t cap = (uint64_t) ggml_backend_buffer_get_size(arenas_[arena_index].get());
                if (arena_used + need > cap) {
                    ++arena_index;
                    arena_used = 0;
                    if (arena_index >= arenas_.size()) {
                        throw std::runtime_error("expert slot arenas exhausted");
                    }
                    cap = (uint64_t) ggml_backend_buffer_get_size(arenas_[arena_index].get());
                }
                if (resources_.slot_classes.size() > 1 &&
                        (need > cap || arena_used > cap - need)) {
                    throw std::runtime_error("expert slot arenas exhausted");
                }
                slots_.push_back(
                    make_slot_in(arenas_[arena_index].get(), arena_used, slot_class.size));
                arena_used += need;
                slots_.back().reserved = reserved_indices.count((int) slots_.size() - 1) != 0;
                // Keep uniform-page accounting byte-identical; class-local
                // arenas report the allocated stride.
                resources_.device_bytes += resources_.slot_classes.size() == 1
                    ? slot_class.size : need;
            }
        }
        // Every input to compute_arena_layout() is now final and never changes
        // again (slot.buffer/offset/capacity are write-once in make_slot_in()).
        // Compute it here, once, instead of rescanning ~6700 slots two to three
        // times per decode request from arena_id_eligible().
        arena_layout_ = compute_arena_layout();
        resources_.staging_buffers      = staging_.buffer_count();
        resources_.staging_buffer_bytes = staging_.buffer_bytes();
        resources_.staging_bytes =
            resources_.staging_buffer_bytes *
            (uint64_t) resources_.staging_buffers;

        // *** WP_EXPERT_HOST_SPEC_BYTES: RAM RESERVED FOR PREFETCH LANDINGS. ***
        //
        // Added ON TOP of the victim budget, so the two knobs mean exactly what
        // they say: WP_EXPERT_HOST_VICTIM_BYTES is what evicted VRAM pages may
        // hold, this is what speculative landings may hold, and the arena is
        // their sum. Unset (0) keeps one shared pool and the historic behaviour.
        //
        // WHY A HARD RESERVATION AND NOT JUST EVICTION ORDER: set_speculative_tier
        // below already drains guesses before victims, but ORDER cannot help when
        // the arena is already full of victims -- a landing then has to evict a
        // page the GPU actually used, or fail. On the sliced rig it failed, every
        // time: host_landed sat at 0 against a 3 GiB tier saturated by demand
        // experts, so the whole prefetch path was measuring a tier it could never
        // get into. Reserved bytes cannot be taken by the demand path at all.
        static const uint64_t host_spec_bytes = [] {
            const char * e = std::getenv("WP_EXPERT_HOST_SPEC_BYTES");
            if (e == nullptr || e[0] == '\0') {
                return (uint64_t) 0;
            }
            const long long v = atoll(e);
            return v > 0 ? (uint64_t) v : (uint64_t) 0;
        }();

        if (host_victim_bytes != 0) {
            const uint64_t arena_bytes = host_victim_bytes + host_spec_bytes;
            if (arena_bytes >
                (uint64_t) std::numeric_limits<size_t>::max() ||
                arena_bytes < host_victim_bytes ||   // overflow
                (!host_tier_->is_initialized() &&
                 !host_tier_->init((size_t) arena_bytes, 0))) {
                throw std::runtime_error("failed to initialize host victim tier");
            }
            if (host_spec_bytes != 0) {
                host_tier_->set_spec_budget((size_t) host_spec_bytes);
            }
            host_tier_->set_device_reader(
                [](void * dst_host, const void * src_device, size_t n, int page_idx) {
                    return g_host_reader_pool != nullptr &&
                        g_host_reader_pool->read_device_page(
                            dst_host, src_device, n, page_idx);
                });
            host_victim_enabled_ = true;
            // Arm HostTier's Pass 0 so an unconfirmed prediction is drained
            // before anything VRAM actually touched. Without it spec_tier_ is
            // false and a guess competes with a known-good victim -- the tier's
            // own comment calls that "prefetch actively degrading the tier it is
            // meant to fill". This line silently failed to apply once already:
            // a str.replace with the wrong indentation is a no-op, not an error.
            host_tier_->set_speculative_tier(true);
            fprintf(stderr,
                    "wp::HostTier: victim=%llu MiB spec_reserved=%llu MiB arena=%llu MiB "
                    "fill_on_read=%d (decode n_tokens<=8) demote_d2h=%d\n",
                    (unsigned long long) (host_victim_bytes >> 20),
                    (unsigned long long) (host_spec_bytes   >> 20),
                    (unsigned long long) ((host_victim_bytes + host_spec_bytes) >> 20),
                    (int) fill_host_on_read_, (int) demote_d2h_);
        }
    }

    ~ExpertSlotPool() {
        // DESTRUCTION-ORDER LANDMINE, DEFUSED EXPLICITLY.
        //
        // spec_batches_ is declared ABOVE slots_/slot_index_ in this class, so
        // C++'s reverse-declaration-order teardown would destroy slots_ and
        // slot_index_ FIRST and spec_batches_ LAST. A live (still-reading)
        // entry's ~Batch() -> abandon_batch() -> complete_batch() drains the
        // read and calls drain_one_read(), which looks the landed page up in
        // slot_index_ -- a hash-map access into an already-destroyed member.
        // Reordering the member declarations would also fix this, but doing it
        // HERE, explicitly, matches the host_threads_ pattern right below (a
        // resource that must be torn down before whatever it touches goes
        // away) and needs no reader to cross-reference two distant member
        // declarations to see why it is safe. clear() drives every live entry
        // through the exact same abandon/complete/join/release path normal
        // retirement uses, just earlier -- while slots_ and slot_index_ are
        // still alive to be looked up.
        spec_batches_.clear();
        // A landing thread outliving the pool would std::terminate on a joinable
        // thread, and it holds raw pointers into catalog pages and the staging
        // arena. Join every one of them -- possibly several now -- before
        // anything they touch goes away. No detached threads, ever.
        for (auto & w : host_threads_) {
            if (w.thread.joinable()) {
                w.thread.join();
            }
        }
#if defined(__linux__)
        for (const auto & item : fds_) {
            close(item.second);
        }
#endif
    }

    size_t pin_pages(const std::vector<const ExpertPage *> & pages);

    struct Loaded {
        ggml_backend_buffer_t buffer = nullptr;
        void *                base   = nullptr;
    };

    struct ArenaLayout {
        struct Arena {
            ggml_backend_buffer_t buffer = nullptr;
            void *                base = nullptr;
            uint64_t              capacity = 0;
            uint64_t              stride = 0;
            size_t                first_slot = 0;
            size_t                n_slots = 0;
        };

        ggml_backend_buffer_t buffer = nullptr;
        void *                base = nullptr;
        uint64_t              slot_stride = 0;
        size_t                n_slots = 0;
        std::vector<Arena>    arenas;

        const Arena * arena_for_slot(size_t slot) const {
            for (const Arena & arena : arenas) {
                if (slot >= arena.first_slot &&
                        slot - arena.first_slot < arena.n_slots) {
                    return &arena;
                }
            }
            return nullptr;
        }
    };

    class Batch {
    public:
        Batch(Batch && other) noexcept;
        ~Batch();

        Batch(const Batch &) = delete;
        Batch & operator=(const Batch &) = delete;
        Batch & operator=(Batch &&) = delete;

        bool is_resident(size_t index) const {
            return entries_.at(index).hit;
        }

        bool is_cpu_on_arrival(size_t index) const {
            return entries_.at(index).cpu_on_arrival;
        }

        const void * cpu_staging(size_t index) const {
            const Entry & entry = entries_.at(index);
            if (!entry.ready || !entry.cpu_staging) {
                throw std::logic_error("CPU-on-arrival page is not ready");
            }
            return entry.cpu_staging->get();
        }

        void release_cpu_staging(size_t index) {
            entries_.at(index).cpu_staging.reset();
        }

        // Slot this entry landed in, or SIZE_MAX for a pinned-resident page that
        // occupies no pool slot. Used by the speculative page-in path to re-stamp the LRU tick.
        size_t slot_index(size_t index) const {
            return entries_.at(index).slot_index;
        }

        const Loaded & loaded(size_t index) const {
            const Entry & entry = entries_.at(index);
            if (!entry.ready) {
                throw std::logic_error("expert batch entry is not ready");
            }
            return entry.loaded;
        }

        void complete();

        void wait_copy_event(ggml_backend_t backend) const {
            if (copy_event_ != nullptr) {
                ggml_backend_event_wait(backend, copy_event_);
            }
        }

        // Drain reads until every entry with index < entry_end that needs a
        // page-in has landed, leaving the rest in flight. Lets the caller
        // compute a leading chunk of experts while the tail is still being read
        // -- see WP_EXPERT_COMPUTE_CHUNKS at the dispatch site. complete() must
        // still be called afterwards to join the reader threads and finalise
        // the read timing, and remains safe to call after any number of these.
        void complete_upto(size_t entry_end);

        uint64_t lookup_ns() const {
            return ns_lookup_;
        }

        uint64_t read_ns() const {
            return ns_read_;
        }

        uint64_t ns_h2d() const {
            return ns_h2d_;
        }

        uint64_t bytes_h2d() const {
            return bytes_h2d_;
        }

        // WP_READER_H2D: pages whose H2D was performed on a reader thread
        // instead of here in drain_one_read. 0 whenever the knob is off (or
        // the backend doesn't support it), same as every other WP_* stat.
        uint64_t n_reader_h2d() const {
            return n_reader_h2d_;
        }

        uint64_t n_read_inflight_max() const {
            return n_read_inflight_max_;
        }

        uint64_t ns_read_issue() const {
            return ns_read_issue_;
        }

        uint64_t ns_read_complete() const {
            return ns_read_complete_;
        }

        uint64_t n_cpu_on_arrival() const {
            return n_cpu_on_arrival_;
        }

        uint64_t n_cpu_on_arrival_fallback() const {
            return n_cpu_on_arrival_fallback_;
        }

        uint64_t n_resident() const {
            return n_resident_;
        }

        uint64_t n_pagein() const {
            return n_pagein_;
        }

        uint64_t n_pagein_reserved() const { return n_pagein_reserved_; }
        uint64_t n_pagein_general() const { return n_pagein_general_; }

        uint64_t bytes_read() const {
            return bytes_read_;
        }

        uint64_t n_host_hit() const {
            return n_host_hit_;
        }

        uint64_t n_host_demote() const {
            return n_host_demote_;
        }

        uint64_t ns_host_get() const {
            return ns_host_get_;
        }

        uint64_t ns_demote() const {
            return ns_demote_;
        }

        uint64_t ns_ensure_post() const {
            return ns_ensure_post_;
        }

        uint64_t host_bytes() const {
            return host_bytes_;
        }

    private:
        friend class ExpertSlotPool;

        struct Entry {
            Loaded loaded;
            size_t slot_index = std::numeric_limits<size_t>::max();
            bool   hit        = false;
            bool   ready      = false;
            bool   cpu_on_arrival = false;
            std::shared_ptr<StagingPool::Lease> cpu_staging;
        };

        explicit Batch(ExpertSlotPool * owner, size_t count) :
            owner_(owner), entries_(count) {
        }

        ExpertSlotPool *           owner_ = nullptr;
        std::vector<Entry>         entries_;
        std::shared_ptr<BatchState> state_;
        std::vector<std::thread>   workers_;
        bool                       completed_ = false;
        uint64_t                   ns_lookup_ = 0;
        uint64_t                   ns_read_   = 0;
        uint64_t                   n_resident_     = 0;
        uint64_t                   n_pagein_    = 0;
        uint64_t                   n_pagein_reserved_ = 0;
        uint64_t                   n_pagein_general_ = 0;
        uint64_t                   bytes_read_ = 0;
        uint64_t                   n_host_hit_ = 0;
        uint64_t                   n_host_demote_ = 0;
        uint64_t                   ns_host_get_ = 0;
        uint64_t                   ns_demote_ = 0;
        uint64_t                   ns_ensure_post_ = 0;
        uint64_t                   host_bytes_ = 0;
        uint64_t                   ns_h2d_    = 0;
        uint64_t                   bytes_h2d_ = 0;
        uint64_t                   n_reader_h2d_ = 0;
        uint64_t                   n_read_inflight_max_ = 0;
        uint64_t                   ns_read_issue_ = 0;
        uint64_t                   ns_read_complete_ = 0;
        uint64_t                   n_cpu_on_arrival_ = 0;
        uint64_t                   n_cpu_on_arrival_fallback_ = 0;
        ggml_backend_event_t       copy_event_ = nullptr;
        // Drain state. Lives on the Batch rather than in complete_batch's frame
        // so a drain can stop part-way (complete_upto) and be resumed. A read
        // that FAILED never sets entry.ready, so every drain loop is bounded by
        // received_ < pageins.size(), never by readiness alone.
        size_t                     received_  = 0;
        std::exception_ptr         first_error_;
        std::chrono::steady_clock::time_point first_read_;
        std::chrono::steady_clock::time_point last_read_;
        std::chrono::steady_clock::time_point first_read_issue_;
        std::chrono::steady_clock::time_point last_read_issue_;
        std::chrono::steady_clock::time_point first_read_complete_;
        std::chrono::steady_clock::time_point last_read_complete_;
        bool                       have_read_time_ = false;
    };

    // *** MEMOISED. arena_layout() USED TO RESCAN EVERY SLOT ON EVERY CALL. ***
    // MEASURED 2026-08-29 by inspection of the Vulkan decode path: the body
    // below walks slots_ once end to end (the `while (first_slot + n_slots <
    // slots_.size())` run-length scan) and heap-allocates layout.arenas. The
    // Vulkan device is started with --slots 6700, so that is 6700 Slot records
    // -- ~870 KiB of L3-resident struct -- traversed per call. dispatch() calls
    // it two to three times per request (grouped_gemv_request's
    // arena_id_eligible, compute_batch's arena_id_eligible, and
    // compute_batch_arena on the arena arm), on the decode critical path,
    // AND IT DOES SO EVEN WHEN WP_EXPERT_ARENA_ID IS UNSET -- arena_id_eligible
    // materialised the layout BEFORE testing `enabled`.
    //
    // WHY MEMOISING IS SAFE (not a cache-invalidation problem): every field the
    // scan reads -- slot.buffer, slot.offset, slot.capacity, and the arena
    // buffers themselves -- is written exactly once, in make_slot_in() /
    // allocate_slot_arenas(), during construction. Nothing in serving mutates
    // them; page residency lives in slot.size/page/key, which this function
    // never looks at. So the layout is a pure function of immutable state and
    // is computed once, eagerly, at the end of the constructor -- which also
    // means no lazy-init race with the reader threads.
    //
    // Returned BY REFERENCE: the old by-value return copied a
    // std::optional<ArenaLayout> (and its std::vector<Arena>) into every
    // caller, i.e. a heap allocation per call on top of the scan.
    const std::optional<ArenaLayout> & arena_layout() const {
        return arena_layout_;
    }

    std::optional<ArenaLayout> compute_arena_layout() const {
        if (slots_.empty() || arenas_.empty()) {
            return std::nullopt;
        }
        // One stride per arena buffer. Qwen's UD mix (Q4_K vs Q5_K/Q8 layer 2)
        // is more than one slot class, so a single-class check left 480/9700
        // with arena_ready=0 and decode n=1 on the 80-kernel per-expert graph.
        // Each buffer is still a regular packed run (offset = i * stride).
        ArenaLayout layout;
        layout.n_slots = slots_.size();
        size_t first_slot = 0;
        uint64_t common_stride = 0;
        bool stride_uniform = true;
        for (const buffer_ptr & buffer : arenas_) {
            const uint64_t capacity = (uint64_t) ggml_backend_buffer_get_size(buffer.get());
            if (capacity == 0 || first_slot >= slots_.size()) {
                return std::nullopt;
            }
            const Slot & head = slots_[first_slot];
            if (head.buffer != buffer.get() || head.offset != 0) {
                return std::nullopt;
            }
            // Slot::size is bytes currently occupied (0 until a page lands).
            // Stride is the allocated capacity, padded to backend alignment.
            const uint64_t stride = arena_slot_stride(head.capacity);
            if (common_stride == 0) {
                common_stride = stride;
            } else if (stride != common_stride) {
                stride_uniform = false;
            }
            size_t n_slots = 0;
            while (first_slot + n_slots < slots_.size()) {
                const Slot & slot = slots_[first_slot + n_slots];
                if (slot.buffer != buffer.get() ||
                        arena_slot_stride(slot.capacity) != stride ||
                        slot.offset != (uint64_t) n_slots * stride) {
                    break;
                }
                ++n_slots;
            }
            if (n_slots == 0 ||
                    (uint64_t) n_slots > UINT64_MAX / stride ||
                    (uint64_t) n_slots * stride > capacity) {
                return std::nullopt;
            }
            ArenaLayout::Arena arena;
            arena.buffer = buffer.get();
            arena.base = ggml_backend_buffer_get_base(buffer.get());
            arena.capacity = capacity;
            arena.stride = stride;
            arena.first_slot = first_slot;
            arena.n_slots = n_slots;
            layout.arenas.push_back(arena);
            first_slot += n_slots;
        }
        if (first_slot != slots_.size()) {
            return std::nullopt;
        }
        layout.slot_stride = stride_uniform ? common_stride : 0;
        if (layout.arenas.size() == 1) {
            layout.buffer = layout.arenas[0].buffer;
            layout.base = layout.arenas[0].base;
            layout.slot_stride = layout.arenas[0].stride;
        }
        return layout;
    }

    // gpu_lock: the caller's g_worker_gpu_mutex unique_lock, or nullptr.
    // Passed through ONLY by the WP_WORKER_MULTI_CONN BEGIN-frame path
    // (Worker::begin_split_dispatch, from serve_connection's PIPE_EXPERT_
    // DISPATCH_BEGIN branch) -- every other caller (the non-split dispatch
    // fallback at line ~5802, the speculative submit path at line ~3086)
    // leaves it null, so for them this function's shared-state footprint is
    // unchanged: still one lock, held start to finish. See the "READ-ISSUE
    // UNLOCKED" block below for what the parameter actually does.
    Batch ensure_batch(
            const std::vector<const ExpertPage *> & pages,
            bool measure,
            std::chrono::steady_clock::time_point lookup_started,
            uint32_t n_tokens = 0,
            int conn_index = -1,
            std::unique_lock<std::mutex> * gpu_lock = nullptr,
            bool count_demand = true) {
        // Take anything the reader threads already landed -- free residency for
        // this request, and it frees the pins. Non-blocking. spec_any_in_flight
        // ("is anything live"), not spec_in_flight ("at the WP_EXPERT_SPEC_MAX_
        // INFLIGHT cap") -- this interlock must run whenever even one batch is
        // outstanding, regardless of whether the pool has room for another.
        //
        // HARDENED AFTER THE s0 INVESTIGATION (2026-08-20): neither
        // spec_recursion_'s reset NOR the poll calls below were exception-safe.
        // spec_recursion_ = false ran only on the FALL-THROUGH path, so a
        // single exception escaping spec_pagein_poll (a backend H2D throw, an
        // allocation failure mid-drain, anything) left spec_recursion_ stuck
        // true FOREVER -- every later ensure_batch call on this pool would then
        // skip the interlock silently, AND the batch spec_pagein_poll was
        // draining when it threw is never retired (retire_spec_batch never
        // runs), so its slots stay pinned forever too: a permanent pin leak
        // that eventually starves select_victim on every subsequent request,
        // with no exception string anywhere (serve_connection's catch, if it
        // even still owns a live socket, only ever saw the FIRST occurrence).
        // That shape -- listens fine, then every request fails, no coredump,
        // no logged exception -- matches the s0 production symptom exactly, so
        // even without a confirmed trigger this is cheap insurance: never let
        // a harvest hiccup here escalate into "silently wedged for the rest of
        // the process". A failure here degrades to "re-read the page on the
        // normal demand path below", which is always correct, just not free.
        if (spec_any_in_flight() && !spec_recursion_) {
            struct RecursionGuard {
                bool & flag;
                ~RecursionGuard() { flag = false; }
            } recursion_guard{ spec_recursion_ };
            spec_recursion_ = true;
            try {
                spec_pagein_poll(false);
                // An in-flight speculative slot is not yet valid, so find_slot
                // below cannot see it and this request would issue a SECOND
                // read of the same page. Wait for the bounded read already in
                // flight instead -- targeted at just the batch holding this
                // page (see spec_pagein_poll's wait_for comment), so an
                // unrelated in-flight batch cannot stall this request once
                // WP_EXPERT_SPEC_MAX_INFLIGHT > 1.
                if (spec_any_in_flight()) {
                    for (const ExpertPage * page : pages) {
                        if (page != nullptr && spec_in_flight_for(*page)) {
                            // cap=1: identical to the pre-MAX_INFLIGHT path
                            // (spec_pagein_poll(true) on the one live batch).
                            // cap>1: block only the batch that holds this page
                            // so an unrelated in-flight read cannot stall demand.
                            if (spec_max_inflight_ <= 1) {
                                spec_pagein_poll(true);
                            } else {
                                spec_pagein_poll(false, page);
                            }
                            break;
                        }
                    }
                }
            } catch (const std::exception & error) {
                // LOGGED, NOT SWALLOWED SILENTLY: this is the one place a spec-
                // harvest failure could otherwise vanish with no trace. Printed
                // even without WP_HINT_LOG so it lands in the worker's own
                // stderr log unconditionally.
                std::fprintf(stderr,
                             "W ensure_batch: speculative harvest failed, "
                             "continuing on the demand path: %s\n",
                             error.what());
                ++spec_errors_;
            } catch (...) {
                std::fprintf(stderr,
                             "W ensure_batch: speculative harvest failed with a "
                             "non-standard exception, continuing on the demand "
                             "path\n");
                ++spec_errors_;
            }
        }
        Batch batch(this, pages.size());
        try {
            std::vector<size_t> pageins;
            pageins.reserve(pages.size());
            std::vector<bool> free_slot_claimed(
                cpu_on_arrival_enabled_ ? slots_.size() : 0, false);
            size_t cpu_on_arrival_count = 0;
            const size_t cpu_on_arrival_max = cpu_on_arrival_enabled_
                ? cpu_on_arrival_limit() : 0;

            // Resolve and pin every hit before selecting a victim. A hit is
            // immediately usable while sibling pageins are read.
            for (size_t i = 0; i < pages.size(); ++i) {
                const ExpertPage & page = *pages[i];
                if (count_demand) {
                    note_demand_reference(page);
                }
                const size_t slot_index = find_slot(page);
                if (slot_index == slots_.size()) {
                    if (page.is_resident) {
                        batch.entries_[i] = {
                            { page.resident_buffer, page.resident_base },
                            std::numeric_limits<size_t>::max(),
                            true,
                            true,
                        };
                        ++batch.n_resident_;
                        continue;
                    }
                    const bool host_hit = cpu_on_arrival_enabled_ && host_victim_enabled_ &&
                        host_tier_->contains(page.cache_id);
                    const size_t free_slot = cpu_on_arrival_enabled_
                        ? free_slot_for(page, free_slot_claimed) : slots_.size();
                    if (free_slot != slots_.size()) {
                        free_slot_claimed[free_slot] = true;
                    } else if (cpu_on_arrival_enabled_ && !host_hit) {
                        if (cpu_on_arrival_count < cpu_on_arrival_max) {
                            batch.entries_[i].cpu_on_arrival = true;
                            ++cpu_on_arrival_count;
                        } else {
                            ++batch.n_cpu_on_arrival_fallback_;
                        }
                    }
                    pageins.push_back(i);
                    continue;
                }
                Slot & slot = slots_[slot_index];
                ++slot.pin_count;
                slot.tick = ++tick_;
                ++slot.uses;
                if (slot.pinned && count_demand) {
                    ++n_pinned_demand_hits_;
                }
                // Promotion: a demand hit on a still-unconfirmed speculative
                // slot confirms the guess. It stops counting against
                // WP_EXPERT_SPEC_MAX_SLOTS from this point on, same as the
                // pager's speculative_[slot]=0 on a demand hit (wp-pool.cpp).
                if (slot.spec_pending) {
                    if (slot.layer_ahead) {
                        ++n_layerahead_hits_;
                        slot.layer_ahead = false;
                    }
                    slot.spec_pending = false;
                    --n_spec_pending_;
                }
                batch.entries_[i] = {
                    // raw->data, NOT the buffer base: one arena backs many slots,
                    // so the buffer base is slot 0's address for every slot.
                    { slot.buffer, slot.raw->data },
                    slot_index,
                    true,
                    true,
                };
                ++batch.n_resident_;
            }

            // Reserve and pin all pagein slots before starting any read. Later
            // allocations in this request cannot select an earlier pagein.
            for (size_t entry_index : pageins) {
                const ExpertPage & page = *pages[entry_index];
                if (batch.entries_[entry_index].cpu_on_arrival) {
                    continue;
                }
                const size_t slot_index = select_victim(page);
                if (slot_index == slots_.size()) {
                    throw std::runtime_error(
                        "no expert slot can hold requested page");
                }
                // The floor rises with what we are actually throwing away, so a
                // page admitted now starts level with the pool instead of at the
                // bottom of it. This is the only thing keeping old counts from
                // locking newcomers out forever, and it needs no tuned constant.
                if (slots_[slot_index].valid) {
                    evict_age_ = slots_[slot_index].uses;
                    ++evictions_;
                }
                slots_[slot_index].lease_until = 0;
                ++slots_[slot_index].pin_count;
                batch.entries_[entry_index].slot_index = slot_index;
            }

            if (measure) {
                batch.ns_lookup_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - lookup_started).count();
            }
            const auto ensure_post_t0 = std::chrono::steady_clock::now();
            auto stamp_ensure_post = [&]() {
                if (measure) {
                    batch.ns_ensure_post_ =
                        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - ensure_post_t0).count();
                }
            };

            struct HostHit {
                const void *                   src = nullptr;
                wp::HostTier::BorrowHandle borrow =
                    wp::HostTier::kInvalidBorrowHandle;
            };
            std::vector<HostHit> host_hits(pages.size());
            if (host_victim_enabled_) {
                for (size_t entry_index : pageins) {
                    const ExpertPage & page = *pages[entry_index];
                    if (host_tier_->borrow(
                            page.cache_id, &host_hits[entry_index].src,
                            (size_t) page.size, &host_hits[entry_index].borrow)) {
                        continue;
                    }
                }
            }

            auto release_host_hits = [&]() {
                for (size_t entry_index : pageins) {
                    HostHit & host_hit = host_hits[entry_index];
                    if (host_hit.borrow != wp::HostTier::kInvalidBorrowHandle) {
                        host_tier_->release(
                            pages[entry_index]->cache_id, host_hit.borrow);
                        host_hit.borrow = wp::HostTier::kInvalidBorrowHandle;
                    }
                }
            };

            // WP_EXPERT_OFFSET_SORT=1 -- read page-ins in (blob, offset) order
            // instead of assignment order. Slots are already assigned above and
            // compute order stays assignment order regardless of read order, so
            // output is byte-identical by construction. Prefill batches average
            // ~31 page-ins; sequentializing the seeks is Kimi's prefill lever,
            // aimed at the 2026 SN750 (3.1 GB/s, the slower shard) most of all.
            static const bool s_offset_sort = [] {
                const char * e = std::getenv("WP_EXPERT_OFFSET_SORT");
                return e != nullptr && e[0] != '\0' && e[0] != '0';
            }();
            if (s_offset_sort) {
                std::sort(pageins.begin(), pageins.end(),
                          [&pages](size_t a, size_t b) {
                              const ExpertPage & pa = *pages[a];
                              const ExpertPage & pb = *pages[b];
                              if (pa.blob != pb.blob) { return pa.blob < pb.blob; }
                              return pa.offset < pb.offset;
                          });
            }
            batch.state_ = std::make_shared<BatchState>();
            batch.state_->conn_index = conn_index;
            batch.state_->admit_host_on_read =
                host_victim_enabled_ && fill_host_on_read_ && !demote_d2h_ &&
                n_tokens > 0 && n_tokens <= 8;
            batch.copy_event_ = staging_.new_copy_event();
            batch.state_->pageins.reserve(pageins.size());
            // WP_READER_H2D: decide once, here, under the lock, for every
            // page-in this ensure_batch() call is about to plan -- see the
            // PageIn::reader_h2d comment for why a single per-batch decision
            // (not a per-stripe re-check on the reader thread) is required
            // for correctness. async_h2d() is read here rather than cached,
            // so a runtime disarm takes effect on the next batch exactly
            // like it does for drain_one_read today; reader_h2d_enabled_
            // itself is fixed for the process (backend + WP_READER_H2D are
            // both decided at construction). copy_stream_h2d() is NOT
            // checked: it defaults ON for every real CUDA/HIP backend
            // (ggml_cuda_wp_copy_requested in ggml-cuda.cu, opt-out only via
            // WP_EXPERT_COPY_STREAM=0), so gating on it left this path dead
            // on every ROCm/CUDA worker that hadn't explicitly disabled the
            // copy stream. It is safe to ignore here because reader_h2d_upload
            // always issues tensor_set_page_range_reader -- a SYNCHRONOUS
            // copy (by the time it returns) on its own dedicated
            // non-blocking reader stream (ggml_backend_cuda_wp_reader_copy
            // in ggml-cuda.cu), never StagingPool's copy_stream_h2d_ stream
            // -- so it never needs the copy-stream event bookkeeping
            // (StagingPool::mark_in_flight / events_) that copy_stream_h2d()
            // exists to gate in drain_one_read.
            const bool reader_h2d_this_batch =
                reader_h2d_enabled_ && !staging_.async_h2d();
            try {
                for (size_t entry_index : pageins) {
                    const ExpertPage & page = *pages[entry_index];
                    if (batch.entries_[entry_index].cpu_on_arrival) {
                        batch.state_->pageins.push_back({
                            entry_index, std::numeric_limits<size_t>::max(), &page,
                            fd_for(page.blob), nullptr, false, true
                        });
                        ++batch.n_cpu_on_arrival_;
                        ++batch.n_pagein_;
                        if (std::binary_search(reserve_blocks_.begin(), reserve_blocks_.end(), page.layer)) {
                            ++batch.n_pagein_reserved_;
                        } else {
                            ++batch.n_pagein_general_;
                        }
                        batch.bytes_read_ += page.size;
                        continue;
                    }
                    const size_t slot_index =
                        batch.entries_[entry_index].slot_index;
                    Slot & slot = slots_[slot_index];
                    if (slot.valid) {
                        const auto demote_t0 = measure ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
                        if (demote_slot(slot)) {
                            ++batch.n_host_demote_;
                        }
                        if (measure) {
                            batch.ns_demote_ +=
                                (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - demote_t0).count();
                        }
                        // The slot is about to hold a different page; drop the
                        // OLD key from the index now, or find_slot(old key) would
                        // keep returning this slot until something overwrites the
                        // map entry (the defensive check in find_slot catches a
                        // stale hit, but there is no reason to rely on it here).
                        slot_index_.erase(slot_key(slot.key.first, slot.key.second));
                        // Evicted before a demand hit ever confirmed it: the
                        // unconfirmed-speculative occupancy this slot held ends
                        // here, whether it is being evicted to serve a demand
                        // page-in or another speculative one (WP_EXPERT_SPEC_MAX_SLOTS).
                        if (slot.spec_pending) {
                            slot.spec_pending = false;
                            slot.layer_ahead = false;
                            --n_spec_pending_;
                        }
                    }
                    slot.valid = false;

                    HostHit & host_hit = host_hits[entry_index];
                    if (host_hit.borrow != wp::HostTier::kInvalidBorrowHandle) {
                        // WP_EXPERT_TIER_VERIFY=1 -- re-read every tier-restored
                        // page from its blob and memcmp against what the tier
                        // returned. Diagnostic for the 2026-08-06 finding that
                        // tier-on runs produce divergent text: discriminates
                        // "restore path corrupts pages" from "timing-induced
                        // reassociation". Costs an extra buffered read per host
                        // hit; never enable in a measured arm.
                        static const bool tier_verify = [] {
                            const char * e = std::getenv("WP_EXPERT_TIER_VERIFY");
                            return e != nullptr && e[0] == '1';
                        }();
                        if (tier_verify) {
                            static uint64_t n_verified = 0, n_mismatch = 0, n_readfail = 0;
                            std::vector<uint8_t> disk((size_t) page.size);
                            bool read_ok = false;
                            const int vfd = ::open(page.blob.c_str(), O_RDONLY);
                            if (vfd >= 0) {
                                read_ok = ::pread(vfd, disk.data(), (size_t) page.size,
                                                  (off_t) page.offset) == (ssize_t) page.size;
                                ::close(vfd);
                            }
                            ++n_verified;
                            if (!read_ok) {
                                ++n_readfail;
                                fprintf(stderr,
                                        "W tier-verify REREAD-FAIL layer=%d expert=%d\n",
                                        page.layer, page.expert);
                            } else if (memcmp(disk.data(), host_hit.src,
                                              (size_t) page.size) != 0) {
                                size_t first = 0;
                                const uint8_t * t = (const uint8_t *) host_hit.src;
                                while (first < (size_t) page.size && disk[first] == t[first]) {
                                    ++first;
                                }
                                ++n_mismatch;
                                fprintf(stderr,
                                        "W tier-verify MISMATCH layer=%d expert=%d size=%llu "
                                        "first_diff_byte=%zu disk=%02x tier=%02x\n",
                                        page.layer, page.expert,
                                        (unsigned long long) page.size, first,
                                        disk[first], t[first]);
                            }
                            fprintf(stderr,
                                    "W tier-verify totals verified=%llu mismatch=%llu readfail=%llu\n",
                                    (unsigned long long) n_verified,
                                    (unsigned long long) n_mismatch,
                                    (unsigned long long) n_readfail);
                        }
                        const std::chrono::steady_clock::time_point host_get_started =
                            measure ? std::chrono::steady_clock::now() :
                                      std::chrono::steady_clock::time_point();
                        tensor_set_page_range(
                            slot.raw, page, host_hit.src, 0, (size_t) page.size);
                        host_tier_->release(page.cache_id, host_hit.borrow);
                        host_hit.borrow = wp::HostTier::kInvalidBorrowHandle;
                        if (!fill_host_on_read_ || demote_d2h_) {
                            host_tier_->erase(page.cache_id);
                        }
                        slot.valid    = true;
                        slot.key      = { page.layer, page.expert };
                        slot_index_[slot_key(page.layer, page.expert)] = slot_index;
                        slot.cache_id = page.cache_id;
                        slot.page     = &page;
                        slot.size     = page.size;
                        slot.tick     = ++tick_;
                        // Admit at the age of the last page we evicted, not at
                        // 1. Otherwise a genuinely hot expert is thrown out
                        // before it can ever prove itself, and stale pages that
                        // were hot early squat forever. Those two are the only
                        // reasons plain use-counting loses to LRU.
                        slot.uses     = lfu_history_enabled_ ? history_uses(page) :
                                         evict_age_ + 1;
                        batch.entries_[entry_index].loaded = {
                            slot.buffer, slot.raw->data
                        };
                        batch.entries_[entry_index].hit   = true;
                        batch.entries_[entry_index].ready = true;
                        ++batch.n_host_hit_;
                        if (measure) {
                            batch.ns_host_get_ +=
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - host_get_started).count();
                        }
                    } else {
                        batch.state_->pageins.push_back({
                            entry_index, slot_index, &page, fd_for(page.blob),
                            slot.raw, reader_h2d_this_batch
                        });
                        ++batch.n_pagein_;
                        if (std::binary_search(reserve_blocks_.begin(), reserve_blocks_.end(), page.layer)) {
                            ++batch.n_pagein_reserved_;
                        } else {
                            ++batch.n_pagein_general_;
                        }
                        batch.bytes_read_ += page.size;
                        if (test_hooks_ != nullptr &&
                            test_hooks_->slot_reserved) {
                            test_hooks_->slot_reserved(
                                page.layer, page.expert, (int) slot_index);
                        }
                    }
                }
            } catch (...) {
                release_host_hits();
                throw;
            }
            g_read_path_stats.record_cpu_on_arrival_fallback(
                batch.n_cpu_on_arrival_fallback_);
            batch.host_bytes_ = host_victim_enabled_ ? host_tier_->used_bytes() : 0;

            if (batch.state_->pageins.empty()) {
                batch.completed_ = true;
                stamp_ensure_post();
                return batch;
            }

            batch.state_->measure = measure;

            // *** READ-ISSUE UNLOCKED FROM g_worker_gpu_mutex (2026-08-25) ***
            //
            // Everything above this point -- hit resolution + pin, victim
            // selection + pin, demote_slot on evicted valid slots, the
            // synchronous host-tier fill for host hits, fd_for() resolution
            // for real page-ins -- mutates slots_, slot_index_, tick_,
            // evict_age_/evictions_, n_spec_pending_/n_layerahead_hits_,
            // host_tier_, and fds_: state shared with the OTHER connection's
            // thread, so it MUST run under g_worker_gpu_mutex (gpu_lock, held
            // by the caller since before this call) exactly as before.
            //
            // What follows -- stripe-job planning, spawning the reader
            // threads, and notify_all() -- touches only batch.state_ (this
            // Batch's own BatchState: pageins/stripe_jobs/page_shared were
            // just built above and are, per the BatchState comment, "read-
            // only after"; start/cancel/mutex/cv are private to this batch)
            // and staging_ (StagingPool has owned its own mutex plus the
            // 2026-08-25 per-connection quota since the earlier deadlock fix,
            // so borrow()/release() from any thread, any connection, is
            // already safe without g_worker_gpu_mutex). None of it touches
            // slots_ or any other pool-wide structure, so it is safe to run
            // with the lock released.
            //
            // WHY THE OTHER CONNECTION CANNOT INVALIDATE THIS: every slot
            // this batch will read from or write into was PINNED above,
            // still under the lock (++slot.pin_count on both the hit loop
            // and the pagein-reservation loop). select_victim_impl skips any
            // slot with pin_count != 0, so the other connection's own
            // planning -- which still requires the lock we are about to
            // release, and will therefore itself run strictly after this
            // unlocked window closes or strictly before this window opened
            // -- cannot select, evict, or reassign a slot this batch is
            // about to read into. The pin is exactly the reservation
            // mechanism release_pins()/abandon_batch() later undo; nothing
            // new was invented here.
            //
            // Only the BEGIN path (Worker::begin_split_dispatch, via
            // serve_connection's PIPE_EXPERT_DISPATCH_BEGIN branch) passes a
            // non-null gpu_lock, and only when g_worker_gpu_mutex != nullptr
            // (WP_WORKER_MULTI_CONN >= 2) -- gpu_lock->owns_lock() is then
            // guaranteed true (serve_connection just locked it for this
            // iteration). Every other caller passes gpu_lock == nullptr, so
            // owns_unlock stays false and this whole block is a no-op there
            // regardless of the env knob below: single-connection and
            // non-BEGIN callers keep taking the lock for the full duration of
            // ensure_batch, byte-identical to before this change.
            //
            // WP_BEGIN_UNLOCKED_READS=1: opt IN to the unlock above. Unset or
            // any other value: stay fully locked through read-issue, i.e. the
            // exact pre-2026-08-25 behaviour, for an A/B pair off one binary.
            // Read once (static, same idiom as WP_EXPERT_READ_WORKERS above
            // and every other WP_* knob in this file) rather than on every
            // call -- getenv() is not free and the value cannot change once
            // the process is up.
            static const bool s_begin_unlocked_reads = [] {
                const char * e = std::getenv("WP_BEGIN_UNLOCKED_READS");
                return e != nullptr && e[0] == '1';
            }();
            const bool owns_unlock = s_begin_unlocked_reads &&
                                     gpu_lock != nullptr && gpu_lock->owns_lock();
            if (owns_unlock) {
                gpu_lock->unlock();
                g_worker_n_begin_unlocked_reads.fetch_add(1, std::memory_order_relaxed);
            }
            // RAII, not a manual relock at the bottom of the try: a thread
            // spawn below (std::thread's constructor) can throw
            // std::system_error, and if it does the outer catch(...) calls
            // cancel_workers()/release_pins(), both of which touch slots_
            // (shared) and therefore need the lock back FIRST. A destructor
            // that always relocks on scope exit -- success or exception --
            // is what makes that true unconditionally, without duplicating
            // the relock at every return/throw site.
            struct RelockOnExit {
                std::unique_lock<std::mutex> * lock;
                bool                           owns;
                ~RelockOnExit() { if (owns) lock->lock(); }
            } relock_guard{ gpu_lock, owns_unlock };

            // Stripe-parallel: pre-plan every (page, stripe) so reader threads
            // claim stripes, not pages. Page-major order keeps the number of
            // concurrently-open pages (= staging leases) near
            // ceil(threads / stripes-per-page), so borrow() never deadlocks:
            // a thread blocked on borrow holds no lease itself, and earlier
            // pages retire to free buffers.
            if (stripe_parallel_ || read_chunk_bytes_ != 0) {
                auto & st = *batch.state_;
                st.page_shared.reserve(st.pageins.size());
                for (size_t pi = 0; pi < st.pageins.size(); ++pi) {
                    const auto plan =
                        stripe_plan(st.pageins[pi].page->size, st.pageins.size());
                    auto ps = std::make_unique<PageShared>();
                    ps->remaining.store(plan.size(), std::memory_order_relaxed);
                    st.page_shared.push_back(std::move(ps));
                    for (const auto & part : plan) {
                        st.stripe_jobs.push_back({ pi, part.first, part.second });
                    }
                }
            }
            // Stripe mode keeps the serial path's invariant: concurrent reader
            // threads <= staging buffers, so outstanding borrows can never
            // exceed the pool and a blocked borrow always has a draining page
            // ahead of it.
            // WP_EXPERT_READ_WORKERS=<n>: cap on concurrent stripe reader
            // threads (default 4, the value that was hardcoded here). Still
            // clamped to the staging pool so the borrow invariant above holds.
            const size_t requested_workers = read_inflight_ != 0
                ? read_inflight_
                : (stripe_parallel_ || read_chunk_bytes_ != 0 ? read_workers_from_env()
                                                               : staging_.buffer_count());
            const size_t worker_count = !batch.state_->stripe_jobs.empty()
                ? std::min<size_t>(batch.state_->stripe_jobs.size(),
                                   std::min<size_t>(requested_workers,
                                                    (size_t) staging_.buffer_count()))
                : std::min<size_t>(
                      batch.state_->pageins.size(),
                      std::min<size_t>(requested_workers, (size_t) staging_.buffer_count()));
            batch.workers_.reserve(worker_count);
            for (size_t i = 0; i < worker_count; ++i) {
                batch.workers_.emplace_back(
                    [this, state = batch.state_]() {
                        read_worker(state);
                    });
            }
            {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                batch.state_->start = true;
            }
            batch.state_->cv.notify_all();
            // relock_guard's destructor fires here (end of scope, normal
            // return path) and reacquires gpu_lock before we hand control
            // back to begin_split_dispatch, which still touches Worker-wide
            // state (split_pending_by_conn_, submit_prefill_layer_ahead)
            // after this call returns and needs the lock held for that, same
            // as before this change.
            stamp_ensure_post();
            return batch;
        } catch (...) {
            cancel_workers(batch);
            release_pins(batch);
            throw;
        }
    }

    // Read `pages` into slots WITHOUT building a compute batch and WITHOUT
    // promoting anything in the LRU. This is the SPECULATIVE PAGE-IN path.
    //
    // IT REUSES ensure_batch + complete_batch DELIBERATELY. A second read path
    // would be a second place for O_DIRECT alignment, striping, staging leases
    // and slot bookkeeping to drift, and the whole point of a prefetch is that
    // the page it leaves behind is indistinguishable from a demand-paged one.
    //
    // *** THE PREFETCH LRU BAND IS THE READ-AMPLIFICATION GUARD. ***
    // select_victim evicts invalid slots first, then the lowest tick. A speculatively paged-in
    // slot must be valid or the next request re-reads it -- but if it also took
    // a FRESH tick it would outrank pages demand actually touched, and an unused
    // prefetch would evict a hot page. That is pool pollution, and it is exactly
    // what made the 2026-07 cross-layer attempt cost 2.7-3.1x the bytes at every
    // width, with a 0.973-precision predictor. So every page read here is
    // stamped from the prefetch band (see kDemandTickBase): strictly older than
    // anything demand has ever touched, hence the first victim if the guess was
    // wrong. A speculative page that is then actually USED gets ++tick_ from
    // ensure_batch's hit path and is promoted into the demand band like any
    // other. A wrong guess therefore costs one read and nothing else.
    //
    // *** SPECULATIVE READS ARE ASYNCHRONOUS. THIS IS THE WHOLE POINT. ***
    //
    // The first version of this called batch.complete() right here -- a blocking
    // join on the reader thread -- from the keepalive pump, which only runs when
    // NO request is pending. So a speculative read could never overlap with
    // compute, which is the only thing prefetching is for, and a request that
    // arrived mid-read waited for it. That is not a prefetch, it is a stall with
    // extra steps, and every speculation measurement taken against it is void.
    //
    // Now: submit and return. The reader threads carry the read while the
    // dispatch thread goes back to poll() and, when work arrives, computes. The
    // pages land underneath the request.
    //
    // SLOT SAFETY: ensure_batch reserves AND PINS every page-in slot before any
    // read is issued, and the pins are held for as long as we hold the Batch. An
    // in-flight speculative slot therefore cannot be recycled out from under its
    // own read -- which is exactly the corruption found on 2026-07-21, when the
    // speculative path allocated slots without pinning them.
    //
    // Returns the number of pages submitted (0 if they were all already present
    // or a batch is still in flight).
    // layer_ahead: one extra in-flight batch beyond WP_EXPERT_SPEC_MAX_INFLIGHT
    // so the prefill whole-next-layer path is not serialized behind the decode
    // spec pump's default cap of 1. Decode spec_pagein_step still passes false.
    size_t spec_pagein_submit(const std::vector<const ExpertPage *> & pages,
                              const std::vector<uint64_t> & leases,
                              bool layer_ahead = false) {
        // WP_EXPERT_SPEC_MAX_INFLIGHT: refuse once spec_batches_ is already at
        // capacity, same shape as the old "one speculative batch in flight at a
        // time" refusal but against a configurable cap instead of a hardcoded 1.
        const size_t cap = (size_t) spec_max_inflight_ + (layer_ahead ? 1 : 0);
        if (pages.empty() || spec_batches_.size() >= cap) {
            return 0;
        }
        std::vector<const ExpertPage *> cold;
        cold.reserve(pages.size());
        std::vector<uint64_t> cold_leases;
        cold_leases.reserve(pages.size());
        for (size_t i = 0; i < pages.size(); ++i) {
            const ExpertPage * page = pages[i];
            if (page == nullptr || page->is_resident || find_slot(*page) != slots_.size() ||
                spec_in_flight_for(*page)) {
                // pinned resident, already in a slot, or already being read by
                // another in-flight speculative batch: nothing to do. The last
                // check only matters once spec_max_inflight_ > 1 -- with a
                // single batch spec_in_flight_for can never be true here
                // because nothing else could be in flight to submit against.
                continue;
            }
            cold.push_back(page);
            cold_leases.push_back(i < leases.size() ? leases[i] : spec_lease_);
        }
        if (cold.empty()) {
            return 0;
        }
        // WP_EXPERT_SPEC_MAX_SLOTS: cap CONCURRENT occupancy of unconfirmed
        // speculative pages against the pool, ported from wp-pager.cpp:860's
        // xlayer_max_slots_/n_speculative() budget check. Clamp the chunk to
        // the remaining budget HERE, before any read is issued -- not submit
        // then free, which would pay for a read only to throw it away.
        // Default 0 = uncapped = byte-identical to today.
        if (spec_max_slots_ > 0) {
            const size_t cap = (size_t) spec_max_slots_;
            const size_t budget = cap > n_spec_pending_ ? cap - n_spec_pending_ : 0;
            if (budget == 0) {
                ++spec_blocked_budget_;
                return 0;
            }
            if (cold.size() > budget) {
                cold.resize(budget);
                cold_leases.resize(budget);
            }
        }
        SpecBatch entry;
        try {
            // measure=false: a speculative read's cost belongs to the spec
            // counters, not to a request's phase timers. Mixing them would make
            // ns_read on the dispatch path stop meaning "time this request spent
            // reading".
            // conn_index -1 ("none"): speculative page-in runs off the idle
            // pump, not inside any connection's transaction -- see the
            // StagingPool quota comment. Uncapped, same as before this fix.
            entry.batch = std::make_unique<Batch>(
                ensure_batch(cold, false, {}, 0, -1, nullptr, false));
            // Safe to set after the fact: reads land on reader threads, but the
            // flag is only consulted by drain_one_read, which runs exclusively
            // on THIS thread and cannot run before submit returns.
            entry.batch->state_->speculative = true;
            if (entry.batch->copy_event_ != nullptr) {
                ggml_backend_event_free(entry.batch->copy_event_);
                entry.batch->copy_event_ = nullptr;
            }
        } catch (const std::exception &) {
            // Advisory: a failed speculative read must not fail the worker. The
            // same error will surface on the demand path, where it belongs.
            ++spec_errors_;
            return 0;
        }
        entry.inflight    = std::move(cold);
        entry.leases      = std::move(cold_leases);
        entry.layer_ahead = layer_ahead;
        // LOG AT SUBMIT, NOT AT HARVEST. The read is issued here, so this is when
        // the cost is paid and when the position in the stream is meaningful.
        // Logging at harvest inverts the order against R: an async batch can be
        // harvested INSIDE ensure_batch, i.e. after the dispatch's reference line
        // has already been written, and the classifier -- which matches S to the
        // next R -- then cannot credit the page to the request that used it. That
        // alone moved USED from 686 to 424 with no change in behaviour.
        if (spec_log_ != nullptr) {
            std::lock_guard<std::mutex> lock(*log_mutex_);
            for (const ExpertPage * page : entry.inflight) {
                fprintf(spec_log_, "S %d %d\n", page->layer, page->expert);
            }
            fflush(spec_log_);
        }
        const size_t n = entry.inflight.size();
        spec_batches_.push_back(std::move(entry));
        return n;
    }

    // AT CAPACITY, not "any batch live" -- this is the pump gate
    // (spec_pagein_step) and the submit-availability check (has_spec_submit_
    // work). Callers that mean "is there ANY speculative work outstanding"
    // (the demand-path interlock in ensure_batch, the prefill-gate harvest,
    // has_spec_work) must use spec_any_in_flight() instead -- see its comment
    // for why the two are NOT the same once WP_EXPERT_SPEC_MAX_INFLIGHT > 1.
    bool spec_in_flight() const {
        return spec_batches_.size() >= (size_t) spec_max_inflight_;
    }

    // ANY speculative batch live, regardless of the WP_EXPERT_SPEC_MAX_INFLIGHT
    // cap. At the default cap of 1 this is identical to spec_in_flight(); once
    // the cap is raised they diverge -- e.g. 1 of 4 slots occupied is "not at
    // capacity" (spec_in_flight()==false, keep submitting) but still "work
    // outstanding" (spec_any_in_flight()==true, keep polling/waiting on it).
    bool spec_any_in_flight() const { return !spec_batches_.empty(); }

    // Outstanding DEMAND page reads -- the Worker pump's demand-first gate
    // (WP_EXPERT_SPEC_DEMAND_FIRST) consults this before submitting new
    // speculative read chunks. Same atomic the host landing thread yields on.
    bool demand_reads_outstanding() const {
        return demand_reads_pending_.load(std::memory_order_relaxed) > 0;
    }

    // Is there anywhere for a predicted page to land? Without the host tier the
    // answer is no and the caller must keep it on the VRAM path.
    bool host_landing_available() const { return host_victim_enabled_; }

    // Demand-serving gate for preemptible landings. Cheap enough to set
    // unconditionally; only consulted when WP_EXPERT_SPEC_PREEMPT=1.
    void demand_serving(bool v) { demand_serving_.store(v, std::memory_order_relaxed); }

    // *** LAND A GUESS IN HOST RAM, NOT IN VRAM. ***
    //
    // A predicted page in a VRAM slot is a slot a CERTAIN page cannot have, and
    // no lease fixes that -- a short lease only lets the guess give the slot up
    // AFTER it has taken it. Landing in the host arena costs no slot at all,
    // and if the guess turns out right the demand path promotes it over PCIe
    // (~1-2 ms) instead of re-reading NVMe (~5 ms) via the borrow path that
    // already exists in ensure_batch.
    //
    // Runs on a reader thread, not the dispatch thread: there is no GPU work
    // here, so the Vulkan command-pool affinity that constrains every other read
    // path does not apply. HostTier is mutex-guarded throughout.
    //
    // Returns pages queued. Requires the host tier -- without it there is
    // nowhere to land and the caller falls back to the VRAM path.
    size_t spec_host_submit(const std::vector<const ExpertPage *> & pages) {
        if (!host_victim_enabled_ || pages.empty()) {
            return 0;
        }
        // Free any landing threads that finished since the last check. Capacity
        // below is measured against host_threads_.size(), so a finished-but-
        // unreaped entry would wrongly look like it is still holding a slot.
        spec_host_reap();
        if (host_threads_.size() >= host_thread_cap_) {
            return 0;   // at capacity -- caller retries on the next pump tick
        }
        // (page, fd). The fd is resolved HERE, on the dispatch thread: fd_for
        // mutates the unguarded fds_ map, and the landing thread below would
        // otherwise race it against ensure_batch's own fd_for on this thread.
        std::vector<std::pair<const ExpertPage *, int>> cold;
        cold.reserve(pages.size());
        for (const ExpertPage * page : pages) {
            // Counted, not lumped. host_landed=0 with host_errors=0 says the
            // filter ate everything and nothing about WHICH condition did it.
            if (page == nullptr || page->cache_id < 0) { ++host_skip_bad_;  continue; }
            if (page->is_resident)                     { ++host_skip_pin_;  continue; }
            if (find_slot(*page) != slots_.size())     { ++host_skip_vram_; continue; }
            if (host_tier_->contains(page->cache_id))   { ++host_skip_tier_; continue; }
            try {
                cold.emplace_back(page, fd_for(page->blob));
            } catch (const std::exception &) {
                // Advisory, like every failure on this path: the same open
                // error surfaces on the demand path, where it belongs.
                host_errors_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (cold.empty()) {
            return 0;
        }
        host_pending_.fetch_add(cold.size(), std::memory_order_release);
        // 'finished' is set by the thread itself, just before it returns, so
        // spec_host_reap() can tell a completed thread from a running one and
        // join() only ever costs the few instructions between that store and
        // the actual return -- never a real wait. Same non-blocking convention
        // the original single-thread code relied on via host_pending_==0.
        auto finished = std::make_shared<std::atomic<bool>>(false);
        std::thread landing_thread([this, cold, finished]() {
            for (const auto & [page, fd] : cold) {
                try {
                    if (spec_preempt_before_borrow_) {
                        while (demand_reads_pending_.load(std::memory_order_relaxed) > 0) {
                            std::this_thread::sleep_for(std::chrono::microseconds(200));
                        }
                    }
                    // conn_index -1 ("none"): a host-landing thread runs off
                    // the idle pump, with no connection transaction of its
                    // own -- see the idle-pump note on the StagingPool quota
                    // comment. Draws from the global pool, uncapped.
                    StagingPool::Lease lease = staging_.borrow(-1);
                    // Fire the same hooks as every other read path. A host
                    // landing IS a read -- instrumentation that cannot see it
                    // would under-report exactly the bytes this feature spends.
                    if (test_hooks_ != nullptr && test_hooks_->read_started) {
                        test_hooks_->read_started(page->layer, page->expert);
                    }
                    if (spec_preempt_) {
                        // Preemptible: slice the read and yield to demand
                        // between slices. A page abandoned mid-read is fine --
                        // the tier store below never sees it, and the partial
                        // work cost only idle bandwidth.
                        //
                        // ANTI-STARVATION (WP_EXPERT_SPEC_PREEMPT_DEADLINE=1):
                        // under continuous decode the demand path is almost
                        // never idle (measured 2026-08-19: the host reader was
                        // in-flight 63% of the time yet completed only 660
                        // reads over 128 tokens -- ~43 ms to land a page that
                        // should take ~3 ms, because every one of a 9 MB
                        // page's 9 slices waits for demand_reads_pending_ to
                        // hit zero, and it almost never does). That starved
                        // prefetch to under 2% of the paging the demand path
                        // does, so it could not move throughput no matter how
                        // good the predictions are (~99% precise). The
                        // preemption intent is still correct -- a speculative
                        // read must never make a demand read WAIT -- what was
                        // missing is a bound on how long a slice itself can be
                        // starved. Track how long the CURRENT slice has been
                        // waiting; once it exceeds
                        // WP_EXPERT_SPEC_PREEMPT_MAX_WAIT_US (default 2000 us)
                        // proceed with that one slice anyway, then resume
                        // yielding for the next. DEFAULT OFF -- gated so this
                        // is exactly today's unbounded-yield behaviour unless
                        // opted in.
                        size_t off = 0;
                        while (off < (size_t) page->size) {
                            if (spec_preempt_deadline_) {
                                const auto wait_start = std::chrono::steady_clock::now();
                                while (demand_reads_pending_.load(std::memory_order_relaxed) > 0) {
                                    const auto waited =
                                        std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - wait_start).count();
                                    if ((uint64_t) waited >= spec_preempt_max_wait_us_) {
                                        // Waited long enough -- take this one
                                        // slice even though a demand read is
                                        // still outstanding, then go back to
                                        // yielding for the next slice.
                                        break;
                                    }
                                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                                }
                            } else {
                                while (demand_reads_pending_.load(std::memory_order_relaxed) > 0) {
                                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                                }
                            }
                            const size_t n =
                                std::min(spec_subread_, (size_t) page->size - off);
                            read_page_range(*page, fd, (char *) lease.get() + off, off, n);
                            off += n;
                        }
                    } else {
                        read_page_range(*page, fd, (char *) lease.get(),
                                        0, (size_t) page->size);
                    }
                    if (test_hooks_ != nullptr && test_hooks_->read_finished) {
                        test_hooks_->read_finished(page->layer, page->expert);
                    }
                    if (host_tier_->store(page->cache_id, lease.get(),
                                         (size_t) page->size, /*speculative=*/true)) {
                        host_landed_.fetch_add(1, std::memory_order_relaxed);
                        host_bytes_.fetch_add(page->size, std::memory_order_relaxed);
                    }
                } catch (const std::exception &) {
                    // Advisory, exactly like the VRAM speculative path: a failed
                    // guess must not fail the worker. The same error surfaces on
                    // the demand path, which is where it belongs.
                    host_errors_.fetch_add(1, std::memory_order_relaxed);
                }
                // Per-page, not per-batch: with multiple concurrent landing
                // threads host_pending_ has to be a real count of pages still
                // in flight across ALL of them, not a single thread's
                // all-or-nothing flag. (With the default cap of 1 thread and
                // chunk of 1 page this decrements exactly once, at the same
                // moment the old store(0) did -- byte-identical externally.)
                host_pending_.fetch_sub(1, std::memory_order_release);
            }
            finished->store(true, std::memory_order_release);
        });
        host_threads_.push_back(HostLandingThread{std::move(landing_thread), finished});
        return cold.size();
    }

    // Is ANY landing thread still working? Used to keep the pump alive
    // (has_spec_work) even when there is nothing left to submit.
    bool spec_host_in_flight() const {
        return host_pending_.load(std::memory_order_acquire) != 0;
    }

    // Is the landing pool at capacity right now? Distinct from
    // spec_host_in_flight(): with host_thread_cap_ > 1, several threads can be
    // busy landing pages while there is STILL room to submit another chunk --
    // that overlap is the entire point of allowing concurrency here. Read-only
    // on host_threads_, which (like spec_host_submit/spec_host_reap) only ever
    // mutates on the dispatch thread.
    bool spec_host_busy() const {
        return host_threads_.size() >= host_thread_cap_;
    }

    // Join any landing threads that have finished so their capacity slot can
    // be reused. Non-blocking: only ever joins a thread whose 'finished' flag
    // is already set, so join() costs at most the instructions between that
    // store and the thread's actual return -- never a real wait. Safe to call
    // from the dispatch thread on every pump tick.
    void spec_host_reap() {
        for (auto it = host_threads_.begin(); it != host_threads_.end(); ) {
            if (it->finished->load(std::memory_order_acquire)) {
                it->thread.join();
                it = host_threads_.erase(it);
            } else {
                ++it;
            }
        }
    }

    uint64_t host_landed() const { return host_landed_.load(std::memory_order_relaxed); }
    uint64_t host_spec_bytes() const { return host_bytes_.load(std::memory_order_relaxed); }
    uint64_t host_spec_errors() const { return host_errors_.load(std::memory_order_relaxed); }
    uint64_t host_spec_promotions() const { return host_tier_->speculative_promotions(); }
    uint64_t host_spec_wasted() const { return host_tier_->speculative_evicted_unused(); }
    uint64_t host_skip_bad()   const { return host_skip_bad_; }
    uint64_t host_skip_pin()   const { return host_skip_pin_; }
    uint64_t host_skip_vram()  const { return host_skip_vram_; }
    uint64_t host_skip_tier()  const { return host_skip_tier_; }

    // Is `page` currently being read speculatively? The demand path has to ask,
    // because an in-flight slot is not yet valid, so find_slot cannot see it and
    // the request would issue a SECOND read of the same page. Searches every
    // live batch -- with WP_EXPERT_SPEC_MAX_INFLIGHT > 1 the page could be in
    // any of them, not just "the" batch.
    bool spec_in_flight_for(const ExpertPage & page) const {
        for (const SpecBatch & entry : spec_batches_) {
            for (const ExpertPage * p : entry.inflight) {
                if (p->layer == page.layer && p->expert == page.expert) {
                    return true;
                }
            }
        }
        return false;
    }

    // Harvest whatever has landed, WITHOUT BLOCKING BY DEFAULT. Walks every
    // live batch (not just "the" one) and retires whichever have fully landed.
    // Returns true if at least one batch finished and was retired. Safe to
    // call from the idle pump and from the dispatch path; both run on the
    // dispatch thread, which is required because drain_one_read performs the
    // H2D upload and Vulkan command pools have thread affinity.
    //
    // `wait_for`, when non-null, force-drains (blocks on) only the ONE batch
    // -- if any -- that is reading that specific page; every OTHER live batch
    // is still polled non-blockingly. This is what lets the demand-path
    // interlock in ensure_batch wait for the read it actually needs without
    // stalling on unrelated in-flight batches once WP_EXPERT_SPEC_MAX_INFLIGHT
    // > 1 (at the default cap of 1 there is at most one batch to begin with,
    // so this is byte-identical to the old single-batch blocking wait).
    bool spec_pagein_poll(bool block, const ExpertPage * wait_for = nullptr) {
        bool did_work = false;
        for (size_t i = 0; i < spec_batches_.size(); ) {
            SpecBatch & entry = spec_batches_[i];
            bool must_block = block;
            if (!must_block && wait_for != nullptr) {
                for (const ExpertPage * p : entry.inflight) {
                    if (p->layer == wait_for->layer && p->expert == wait_for->expert) {
                        must_block = true;
                        break;
                    }
                }
            }
            Batch & batch = *entry.batch;
            bool landed = true;
            while (batch.received_ < batch.state_->pageins.size()) {
                if (!must_block) {
                    std::lock_guard<std::mutex> lock(batch.state_->mutex);
                    if (batch.state_->ready.empty()) {
                        landed = false;   // still in flight; come back later
                        break;
                    }
                }
                drain_one_read(batch);
            }
            if (landed) {
                retire_spec_batch(i);   // erases spec_batches_[i]; do not advance i
                did_work = true;
            } else {
                ++i;
            }
        }
        return did_work;
    }

    // Called when a demand request needs a page some batch is reading. Bounded:
    // at most WP_EXPERT_SPEC_CHUNK pages, and the read is already in flight.
    void spec_pagein_finish() { spec_pagein_poll(true); }

private:
    // Retires spec_batches_[i]: joins its reads, stamps the landed slots, frees
    // its pins, and erases it from the live list. `i` must be a valid index;
    // every submitted batch reaches this exactly once, either here or via the
    // catch below -- no path leaves a completed batch unretired (see the
    // free_q=0 / orphaned-Done comment elsewhere in this file for why that
    // invariant matters).
    void retire_spec_batch(size_t i) {
        SpecBatch & entry = spec_batches_[i];
        Batch & batch = *entry.batch;
        uint64_t n_read = 0;
        try {
            complete_batch(batch);
            n_read = batch.n_pagein();
            spec_bytes_ += batch.bytes_read();

            // Stamp by CHECKING THE SLOT, not by trusting an entry flag. `hit`
            // is only set for pages already present or from the host tier -- a
            // fresh disk page-in sets `ready`, never `hit` -- so keying off
            // is_resident() here would silently skip exactly the pages this path
            // exists to stamp, leaving every speculative page with a FRESH tick
            // and reintroducing the pollution the stale tick prevents.
            for (size_t j = 0; j < entry.inflight.size(); ++j) {
                const size_t slot_index = batch.slot_index(j);
                if (slot_index >= slots_.size()) {
                    continue;
                }
                Slot & slot = slots_[slot_index];
                if (slot.valid &&
                    slot.key == std::pair<int, int>(entry.inflight[j]->layer,
                                                    entry.inflight[j]->expert)) {
                    slot.tick        = ++spec_tick_;
                    slot.uses        = 0;
                    slot.lease_until = evictions_ +
                        (j < entry.leases.size() ? entry.leases[j] : spec_lease_);
                    // This landed page has not been touched by any demand
                    // request yet -- mark it unconfirmed-speculative for
                    // WP_EXPERT_SPEC_MAX_SLOTS. Guarded on !spec_pending so a
                    // slot can never be double-counted.
                    if (!slot.spec_pending) {
                        slot.spec_pending = true;
                        slot.layer_ahead  = entry.layer_ahead;
                        ++n_spec_pending_;
                    }
                }
            }
        } catch (const std::exception &) {
            ++spec_errors_;
        }
        spec_pageins_ += n_read;
        release_pins(batch);
        spec_batches_.erase(spec_batches_.begin() + (ptrdiff_t) i);
    }

public:
    uint64_t spec_pageins()  const { return spec_pageins_; }
    uint64_t spec_bytes()  const { return spec_bytes_; }
    uint64_t spec_errors() const { return spec_errors_; }
    uint64_t n_shield_hits() const { return n_shield_hits_; }
    uint64_t n_shield_exhausted() const { return n_shield_exhausted_; }
    size_t n_pinned() const { return n_pinned_; }
    uint64_t n_pinned_demand_hits() const { return n_pinned_demand_hits_; }

    // Hint frames and victim selection are both owned by the dispatch thread.
    // Keep only the configured number of hint frames, with counts so a page
    // repeated in two frames remains shielded until both frames expire.
    void note_hint_frame(
            uint32_t provenance,
            const std::vector<std::pair<int32_t, int32_t>> & pages) {
        if (hint_shield_depth_ == 0) {
            return;
        }
        std::vector<uint64_t> keys;
        if (provenance != PIPE_HINT_PREDICTED || hint_shield_predicted_) {
            keys.reserve(pages.size());
            for (const auto & page : pages) {
                const uint64_t key = slot_key(page.first, page.second);
                keys.push_back(key);
                ++hint_shield_counts_[key];
            }
        }
        hint_shield_history_.push_back(std::move(keys));
        while (hint_shield_history_.size() > hint_shield_depth_) {
            const std::vector<uint64_t> & expired = hint_shield_history_.front();
            for (uint64_t key : expired) {
                const auto it = hint_shield_counts_.find(key);
                if (it == hint_shield_counts_.end()) {
                    continue;
                }
                if (--it->second == 0) {
                    hint_shield_counts_.erase(it);
                }
            }
            hint_shield_history_.pop_front();
        }
    }

    // Live occupancy against WP_EXPERT_SPEC_MAX_SLOTS, and how many times
    // submission was blocked or shrunk by it.
    size_t   n_spec_pending()     const { return n_spec_pending_; }
    uint64_t spec_blocked_budget() const { return spec_blocked_budget_; }
    // Live in-flight speculative BATCH count and the WP_EXPERT_SPEC_MAX_INFLIGHT
    // cap it is checked against -- surfaced on the WP_HINT_LOG counter line
    // (spec_inflight[live/cap]) so a run proves its own configuration rather
    // than requiring the operator to trust an env var was read correctly.
    size_t   spec_inflight_live() const { return spec_batches_.size(); }
    size_t   spec_inflight_cap()  const { return (size_t) spec_max_inflight_; }
    uint64_t n_layerahead_hits()  const { return n_layerahead_hits_; }

    // Unpinned slots are the ones select_victim can take without stealing a
    // live demand or in-flight spec pin. Layer-ahead uses this as the silent
    // fallback budget: fetch min(layer, unpinned), never error.
    size_t unpinned_slots() const {
        size_t n = 0;
        for (const Slot & slot : slots_) {
            if (!slot.pinned && slot.pin_count <= 0) {
                ++n;
            }
        }
        return n;
    }

    // BORROWED, NOT OWNED -- WorkerLogFiles outlives the pool.
    //
    // The hint log is an ORDERED EVENT STREAM, and the order is the entire point.
    // "Speculatively read, then evicted, then demand-read again" and "speculated
    // on an expert that was never selected" are different failures with different
    // fixes, and the ONLY thing that separates them is whether the demand read
    // came after the speculative one. Two log files have no shared clock, so
    // pool-side page-ins and worker-side hints must go through the SAME handle.
    void set_spec_log(FILE * f, std::mutex * mutex) {
        spec_log_ = f;
        log_mutex_ = mutex;
    }

    // The lease a page gets by provenance. Read by the Worker, which owns the
    // hint queue and resolves provenance to a lease at enqueue time.
    uint64_t spec_lease()           const { return spec_lease_; }
    uint64_t spec_lease_predicted() const { return spec_lease_pred_; }

    const ResourcePlan & resources() const {
        return resources_;
    }

    // "pinned" or "pageable" -- whichever path the staging pool actually
    // used, not whichever was requested.
    const char * staging_kind() const {
        return staging_.pinned() ? "pinned" : "pageable";
    }

    size_t read_inflight() const {
        const size_t requested = read_inflight_ != 0
            ? read_inflight_
            : (stripe_parallel_ || read_chunk_bytes_ != 0
                   ? read_workers_from_env() : staging_.buffer_count());
        return std::min<size_t>(requested, (size_t) staging_.buffer_count());
    }
    size_t read_chunk_bytes() const { return read_chunk_bytes_; }
    bool read_direct() const { return read_direct_; }
    bool read_direct_fallback() const {
        return g_read_direct_fallback.load(std::memory_order_relaxed);
    }

    // Delegates to StagingPool::set_multi_conn -- see that method and the
    // StagingPool class comment (2026-08-25 deadlock fix) for the quota
    // formula and why this must run before any connection thread starts.
    void set_staging_multi_conn(int n) {
        staging_.set_multi_conn(n);
    }

    bool read_device_page(void * dst_host, const void * src_device, size_t n,
                          int page_idx) const {
        (void) src_device;
        for (const Slot & slot : slots_) {
            if (slot.valid && slot.raw != nullptr && slot.page != nullptr &&
                    slot.cache_id == page_idx && n == slot.page->size) {
                tensor_get_page(slot.raw, *slot.page, dst_host);
                return true;
            }
        }
        return false;
    }

private:
    struct Slot {
        context_ptr         ctx;
        // NON-OWNING. Slots are carved out of a few large ARENA buffers owned by
        // arenas_, not allocated one-per-slot, because one backend allocation per
        // slot walks straight into the allocators' per-allocation limits:
        //   RX 480 / RADV : maxMemoryAllocationCount is 4096, and ggml's fallback
        //     list ends in HostVisible|HostCoherent, so allocation 4097 SILENTLY
        //     lands in GTT (system RAM) instead of failing. Measured 2026-08-16:
        //     4096 slots -> 6526 MiB in VRAM, 4400 -> 6528 MiB in VRAM plus
        //     574 MiB in GTT. 1.6 GB of an 8 GB card unreachable.
        //   GTX 1070 / CUDA: rounds every allocation up to a 2 MiB granule, so a
        //     1.594 MiB slice page wastes 25%. 3800 slots consumed 7600 MiB of
        //     VRAM to hold 6057 MiB of experts.
        // Both are consequences of the SLICED page being 1.594 MiB where the old
        // whole-expert page was 12.75 MiB -- 8x the slot count at the same bytes.
        // An arena is one allocation per ~GB, under both limits, with no rounding.
        ggml_backend_buffer_t buffer  = nullptr;
        // Byte offset of this slot within `buffer`. raw->data is the authoritative
        // slot address; this is kept for accounting and assertions.
        uint64_t            offset    = 0;
        ggml_tensor *       raw       = nullptr;
        const ExpertPage *   page      = nullptr;
        std::pair<int, int> key;
        int                 cache_id  = -1;
        uint64_t            capacity  = 0;
        uint64_t            size      = 0;
        uint64_t            tick      = 0;
        // How many times this page has been asked for. Ranking by USE COUNT is
        // the whole policy; tick is only the tie-break. Offline on the reference
        // stream this beats LRU by 3.2-4.0% of page-ins, against +0.2-1.7% for
        // ARC and negative for 2Q -- counting wins, and it is ten lines.
        uint64_t            uses      = 0;
        // Eviction-counter value at which this page stops being protected.
        // A SPECULATIVE PAGE IS USELESS IF IT DIES BEFORE ITS LAYER ARRIVES, and
        // measured 90% of them did. uses=0 made them the first victim by
        // construction -- which is what kept demand pages safe (layers 3+ were
        // identical to the digit across every arm) and also what guaranteed they
        // never survived long enough to pay. The lease is the bounded middle:
        // protected for a fixed number of evictions, then ordinary.
        uint64_t            lease_until = 0;
        int                 pin_count = 0;
        bool                reserved  = false;
        bool                valid     = false;
        // Set true ONLY by retire_spec_batch, when a speculative page-in lands
        // and has not yet been confirmed by a demand hit. Cleared (with the
        // matching n_spec_pending_ decrement) at exactly the two places that end
        // that state: the demand-hit path in ensure_batch (promotion -- the page
        // was actually wanted) and the victim-eviction path in ensure_batch
        // (the slot's content is discarded before it was ever confirmed). Backs
        // WP_EXPERT_SPEC_MAX_SLOTS -- see n_spec_pending_ for why this must be
        // exact rather than inferred from lease_until (a lease can expire while
        // the page is still sitting there unconfirmed, and that must still count).
        bool                spec_pending = false;
        // Set with spec_pending when the landing came from WP_PREFILL_LAYER_AHEAD.
        // Demand hit counts n_layerahead_hits_ and clears both.
        bool                layer_ahead  = false;
        bool                pinned      = false;
    };

    // (layer, expert) -> uint64 for slot_index_. Same packing as SPINE's
    // residency_key (pipe-expert-dispatcher.cpp) -- no reason for the two caches
    // to disagree about what identifies a page.
    static uint64_t slot_key(int layer, int expert) {
        return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    }

    // Index of the slot already holding `page`, or slots_.size(). ONE definition:
    // the speculative path must agree with ensure_batch about what "already here" means,
    // or a prefetch re-reads a page that is sitting in a slot and the extra bytes
    // land in the speculative-read counter as if they were a real page-in.
    //
    // O(1) VIA slot_index_, NOT A SCAN. At ~2200 slots and ~40 assignments/layer *
    // 43 layers/token this was tens of thousands of key compares per token on the
    // serial dispatch thread, ahead of any disk read. slot_index_ is maintained at
    // every site that changes what a slot holds (assign, invalidate, reset) -- see
    // the comment on slot_index_'s declaration for the full list.
    size_t find_slot(const ExpertPage & page) const {
        const auto it = slot_index_.find(slot_key(page.layer, page.expert));
        if (it == slot_index_.end()) {
            return slots_.size();
        }
        const size_t i = it->second;
        // Defensive, not load-bearing: if every mutation site keeps slot_index_ in
        // sync this is always true. Cheap enough (one extra compare on a hit) to
        // leave in unconditionally rather than gate it behind a debug flag.
        if (i >= slots_.size() || !slots_[i].valid ||
            slots_[i].key != std::make_pair(page.layer, page.expert)) {
            return slots_.size();
        }
        return i;
    }

    size_t free_slot_for(
            const ExpertPage & page, const std::vector<bool> & claimed) const {
        const bool wants_reserved = std::binary_search(
            reserve_blocks_.begin(), reserve_blocks_.end(), page.layer);
        const auto find = [&](bool reserved) {
            size_t result = slots_.size();
            for (size_t i = 0; i < slots_.size(); ++i) {
                const Slot & slot = slots_[i];
                if (claimed[i] || slot.pinned || slot.pin_count != 0 || slot.valid ||
                        slot.reserved != reserved || page.size > slot.capacity) {
                    continue;
                }
                if (result == slots_.size() || slot.capacity < slots_[result].capacity) {
                    result = i;
                }
            }
            return result;
        };
        const size_t result = find(wants_reserved);
        return result != slots_.size() || !wants_reserved ? result : find(false);
    }

    size_t cpu_on_arrival_limit() {
        return std::min(cpu_on_arrival_cap_, staging_.cpu_lease_cap());
    }

    bool demote_slot(const Slot & slot) {
        if (slot.pinned || !host_victim_enabled_ || !slot.valid ||
            slot.cache_id < 0 || slot.size == 0) {
            return false;
        }
        if (fill_host_on_read_ && !demote_d2h_ && host_tier_->contains(slot.cache_id)) {
            return true;
        }
        const auto previous = g_host_reader_pool;
        g_host_reader_pool = this;
        const bool stored = slot.page != nullptr && host_tier_->store_from_device(
            slot.cache_id, slot.raw->data, (size_t) slot.size);
        g_host_reader_pool = previous;
        return stored;
    }

    bool hint_shielded(const Slot & slot) const {
        return hint_shield_depth_ != 0 && slot.valid &&
            hint_shield_counts_.count(slot_key(slot.key.first, slot.key.second)) != 0;
    }

    size_t select_victim_impl(
            const ExpertPage & page, bool skip_shielded, bool * shielded_seen) const {
        const uint64_t page_size = page.size;
        const bool wants_reserved = std::binary_search(reserve_blocks_.begin(), reserve_blocks_.end(), page.layer);
        size_t victim = slots_.size();
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pinned || slot.pin_count != 0 || slot.valid ||
                (!wants_reserved && slot.reserved) ||
                (wants_reserved && !slot.reserved) ||
                page_size > slot.capacity) {
                continue;
            }
            if (skip_shielded && hint_shielded(slot)) {
                if (shielded_seen != nullptr) {
                    *shielded_seen = true;
                }
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity) {
                victim = i;
            }
        }
        if (victim != slots_.size()) {
            return victim;
        }

        if (wants_reserved) {
            for (size_t i = 0; i < slots_.size(); ++i) {
                const Slot & slot = slots_[i];
                if (slot.pinned || slot.pin_count != 0 || slot.valid ||
                    slot.reserved || page_size > slot.capacity) continue;
                if (skip_shielded && hint_shielded(slot)) {
                    if (shielded_seen != nullptr) {
                        *shielded_seen = true;
                    }
                    continue;
                }
                if (victim == slots_.size() || slot.capacity < slots_[victim].capacity) victim = i;
            }
            if (victim != slots_.size()) return victim;
        }

        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            if (slot.pinned || slot.pin_count != 0 || !slot.valid ||
                (!wants_reserved && slot.reserved) ||
                (wants_reserved && !slot.reserved) ||
                page_size > slot.capacity) {
                continue;
            }
            if (skip_shielded && hint_shielded(slot)) {
                if (shielded_seen != nullptr) {
                    *shielded_seen = true;
                }
                continue;
            }
            if (victim == slots_.size() ||
                slot.capacity < slots_[victim].capacity ||
                (slot.capacity == slots_[victim].capacity &&
                 rank_less(slot, slots_[victim]))) {
                victim = i;
            }
        }
        if (wants_reserved && victim == slots_.size()) {
            for (size_t i = 0; i < slots_.size(); ++i) {
                const Slot & slot = slots_[i];
                if (slot.pinned || slot.pin_count != 0 || !slot.valid ||
                    page_size > slot.capacity) continue;
                if (skip_shielded && hint_shielded(slot)) {
                    if (shielded_seen != nullptr) {
                        *shielded_seen = true;
                    }
                    continue;
                }
                if (victim == slots_.size() || slot.capacity < slots_[victim].capacity ||
                    (slot.capacity == slots_[victim].capacity &&
                     rank_less(slot, slots_[victim]))) victim = i;
            }
        }
        return victim;
    }

    size_t select_victim(const ExpertPage & page) {
        if (hint_shield_depth_ == 0) {
            return select_victim_impl(page, false, nullptr);
        }
        bool shielded_seen = false;
        const size_t victim = select_victim_impl(page, true, &shielded_seen);
        if (victim != slots_.size()) {
            if (shielded_seen) {
                ++n_shield_hits_;
            }
            return victim;
        }
        const size_t fallback = select_victim_impl(page, false, nullptr);
        if (fallback != slots_.size() && shielded_seen) {
            ++n_shield_exhausted_;
        }
        return fallback;
    }

    // Stripe-claiming reader (WP_EXPERT_STRIPE_PARALLEL=1). Work unit = one
    // stripe; the first claimer of a page borrows its staging lease; the last
    // COMPLETER (atomic countdown) publishes `last`, inheriting any stripe's
    // failure so drain never sees a half-read page as ready.
    // WP_READER_H2D: perform the whole-page H2D for `pagein` from a reader
    // thread, using the staging buffer `result.staging`, and stamp `result`
    // for drain_one_read to skip its own copy. Called only once every
    // stripe of the page has landed in that (shared) staging buffer -- see
    // the call sites in stripe_read_worker/read_worker, both of which only
    // reach this on the page's LAST stripe. Touches no pool state (slots_,
    // slot_index_, tick_): pagein.raw was captured at plan time under the
    // lock precisely so this can run off it. On failure the exception is
    // carried in result.error exactly like a read error, so drain_one_read
    // -- which checks result->error before ever looking at result->uploaded
    // -- never publishes a half-uploaded slot.
    // WP_READER_H2D_VERIFY=1: cheap post-upload tripwire -- see
    // tensor_verify_page_range. Default off; only ever enabled for
    // diagnosis (adds a D2H read-back + memcmp per uploaded page).
    static bool reader_h2d_verify_enabled() {
        static const bool enabled = [] {
            const char * e = std::getenv("WP_READER_H2D_VERIFY");
            return e != nullptr && e[0] == '1';
        }();
        return enabled;
    }

    void reader_h2d_upload(const PageIn & pagein, bool measure, ReadResult & result) {
        const std::chrono::steady_clock::time_point t0 =
            measure ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
        try {
            // NOT tensor_set_page_range / ggml_backend_tensor_set: on
            // gfx1201 that takes the MAD-114 legacy-stream synchronous
            // hipMemcpy, which is illegal to issue from this reader thread
            // while a connection thread may be mid HIP-graph capture (the
            // 2026-08-25 SIGABRT: "operation would make the legacy stream
            // depend on a capturing blocking stream"). Route through the
            // dedicated non-blocking reader stream instead -- see
            // ggml_backend_cuda_wp_reader_copy in ggml-cuda.cu for the full
            // ordering argument (a host-synchronized DMA that has fully
            // retired before any later kernel launch is a stronger
            // guarantee than the same-stream kernel ordering MAD-114 is
            // actually about).
            tensor_set_page_range_reader(
                backend_, pagein.raw, *pagein.page, result.staging->get(),
                0, (size_t) pagein.page->size);
        } catch (...) {
            result.error = std::current_exception();
            return;
        }
        result.uploaded  = true;
        result.h2d_bytes = (uint64_t) pagein.page->size;
        if (measure) {
            result.h2d_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
        }
        if (reader_h2d_verify_enabled()) {
            const size_t page_size = (size_t) pagein.page->size;
            const size_t sample    = std::min<size_t>(4096, page_size);
            uint64_t mismatches = 0;
            tensor_verify_page_range(
                pagein.raw, *pagein.page, result.staging->get(), 0, sample, mismatches);
            if (page_size > sample) {
                tensor_verify_page_range(
                    pagein.raw, *pagein.page, result.staging->get(),
                    page_size - sample, sample, mismatches);
            }
            if (mismatches > 0) {
                g_worker_n_reader_h2d_verify_fail.fetch_add(
                    mismatches, std::memory_order_relaxed);
            }
        }
    }

    void stripe_read_worker(const std::shared_ptr<BatchState> & state) {
        while (true) {
            const size_t j = state->next.fetch_add(1, std::memory_order_relaxed);
            if (j >= state->stripe_jobs.size()) {
                return;
            }
            const StripeJob & job    = state->stripe_jobs[j];
            const PageIn &    pagein = state->pageins[job.pagein_indexb];
            PageShared &      shared = *state->page_shared[job.pagein_indexb];
            auto result = std::make_unique<ReadResult>();
            result->pagein_indexb = job.pagein_indexb;
            result->offset        = job.offset;
            result->len           = job.len;
            // Same demand accounting as the serial path, per stripe: the
            // preemption gate (demand_reads_pending_) must see these reads or
            // host landings would run concurrently with stripe-mode demand.
            const bool count_demand = !state->speculative;
            if (count_demand) {
                demand_reads_pending_.fetch_add(1, std::memory_order_relaxed);
            }
            try {
                // The lease pointer must be COPIED UNDER THE MUTEX: another
                // thread may be assigning shared.lease inside its own critical
                // section, and a concurrent unguarded read of a shared_ptr is a
                // data race (this exact line, read outside the lock, hung the
                // 1070 worker on its first stripe-parallel page-in batch,
                // 2026-08-07 sp1).
                std::shared_ptr<StagingPool::Lease> lease_local;
                {
                    std::lock_guard<std::mutex> lock(shared.lease_mutex);
                    if (!shared.lease) {
                        shared.lease = std::make_shared<StagingPool::Lease>(
                            staging_.borrow(state->conn_index));
                        if (test_hooks_ != nullptr && test_hooks_->staging_borrowed) {
                            test_hooks_->staging_borrowed();
                        }
                        if (test_hooks_ != nullptr && test_hooks_->read_started) {
                            test_hooks_->read_started(pagein.page->layer, pagein.page->expert);
                        }
                    }
                    lease_local = shared.lease;
                }
                result->staging = std::move(lease_local);
                if (state->measure) {
                    result->read_started = std::chrono::steady_clock::now();
                    result->read_timed   = true;
                }
                read_page_range(*pagein.page, pagein.fd,
                                (char *) result->staging->get() + job.offset,
                                job.offset, job.len,
                                state->measure ? state.get() : nullptr);
                if (state->measure && result->read_timed) {
                    result->read_finished = std::chrono::steady_clock::now();
                }
            } catch (...) {
                result->error = std::current_exception();
                shared.failed.store(true, std::memory_order_release);
            }
            if (count_demand) {
                demand_reads_pending_.fetch_sub(1, std::memory_order_relaxed);
            }
            const bool is_last =
                shared.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1;
            result->last = is_last;
            if (is_last) {
                // *** RELEASE THE PAGE'S LEASE PIN. THIS IS THE sp1/sp1r DEADLOCK
                // FIX. *** PageShared holds a shared_ptr copy of the lease; left
                // in place it pins the staging buffer until the BATCH is
                // destroyed, and a prefill batch (~31 page-ins) pins more pages
                // than the pool has buffers (16) -- the readers exhaust the pool
                // and block in borrow() forever while the dispatch thread waits
                // for results that can never be read. Resetting here leaves only
                // the in-flight ReadResults holding the buffer, which is exactly
                // the serial path's lifetime: the buffer recycles as soon as the
                // page's stripes drain.
                std::lock_guard<std::mutex> lock(shared.lease_mutex);
                shared.lease.reset();
            }
            if (is_last) {
                if (result->error == nullptr &&
                    shared.failed.load(std::memory_order_acquire)) {
                    // Another stripe of this page failed; the page must not
                    // publish. Carry a failure on the final result.
                    try {
                        throw std::runtime_error("stripe-parallel: sibling stripe failed");
                    } catch (...) {
                        result->error = std::current_exception();
                    }
                }
                if (result->error == nullptr && test_hooks_ != nullptr &&
                    test_hooks_->read_finished) {
                    try {
                        test_hooks_->read_finished(pagein.page->layer, pagein.page->expert);
                    } catch (...) {
                        result->error = std::current_exception();
                    }
                }
                // WP_READER_H2D: every stripe of this page landed in
                // result->staging (the acq_rel fetch_sub above is the
                // synchronizes-with edge that makes every sibling thread's
                // writes visible here -- see PageShared's remaining/failed
                // comment), so the whole page can be uploaded now, off the
                // connection thread. pagein.reader_h2d was decided once for
                // the whole batch at plan time (see PageIn::reader_h2d), so
                // every non-last stripe of this same page already took the
                // no-op branch below instead of drain_one_read's copy.
                if (result->error == nullptr && pagein.reader_h2d) {
                    reader_h2d_upload(pagein, state->measure, *result);
                }
            } else if (result->error == nullptr && pagein.reader_h2d) {
                // Not the last stripe: this stripe's bytes are not yet a
                // complete, copyable page range on their own (the LAST
                // stripe above does the single whole-page copy once every
                // stripe has landed). Mark uploaded with zero bytes/ns so
                // drain_one_read's "if (result->uploaded)" branch treats
                // this as a pure no-op instead of copying a partial stripe.
                result->uploaded = true;
            }
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->ready.push_back(std::move(result));
            }
            state->cv.notify_one();
        }
    }

    void read_worker(const std::shared_ptr<BatchState> & state) {
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&]() {
                return state->start || state->cancel;
            });
            if (state->cancel) {
                return;
            }
        }

        if (!state->stripe_jobs.empty()) {
            stripe_read_worker(state);
            return;
        }

        while (true) {
            const size_t pagein_indexb =
                state->next.fetch_add(1, std::memory_order_relaxed);
            if (pagein_indexb >= state->pageins.size()) {
                return;
            }
            const PageIn & pagein = state->pageins[pagein_indexb];
            // Captured per page: the speculative flag is set on the batch just
            // after submit, so the first page of a spec batch may briefly count
            // as demand. Sub-millisecond and harmless; the flag never flips the
            // other way.
            const bool count_demand = !state->speculative;
            if (count_demand) {
                demand_reads_pending_.fetch_add(1, std::memory_order_relaxed);
            }
            bool read_started = false;
            std::shared_ptr<StagingPool::Lease> staging;
            std::exception_ptr fatal;
            try {
                staging = std::make_shared<StagingPool::Lease>(
                    staging_.borrow(state->conn_index));
                if (test_hooks_ != nullptr &&
                    test_hooks_->staging_borrowed) {
                    test_hooks_->staging_borrowed();
                }
                read_started = true;
                if (test_hooks_ != nullptr &&
                    test_hooks_->read_started) {
                    test_hooks_->read_started(
                        pagein.page->layer, pagein.page->expert);
                }
            } catch (...) {
                fatal = std::current_exception();
            }

            const auto plan = fatal ? std::vector<std::pair<size_t, size_t>>{{0, 0}}
                                    : stripe_plan(pagein.page->size,
                                                  state->pageins.size());
            // Publish each stripe as soon as it lands so the dispatch thread can
            // upload it while the next one is still being read. The LAST stripe
            // is what flips entry.ready and advances received_, so a page is
            // never visible to compute half-uploaded.
            for (size_t s = 0; s < plan.size(); ++s) {
                auto result = std::make_unique<ReadResult>();
                result->pagein_indexb = pagein_indexb;
                result->staging       = staging;
                result->offset        = plan[s].first;
                result->len           = plan[s].second;
                result->last          = (s + 1 == plan.size());
                result->error         = fatal;
                if (!fatal) {
                    try {
                        if (state->measure) {
                            result->read_started = std::chrono::steady_clock::now();
                            result->read_timed = true;
                        }
                        read_page_range(
                            *pagein.page, pagein.fd,
                            (char *) staging->get() + result->offset,
                            result->offset, result->len,
                            state->measure ? state.get() : nullptr);
                    } catch (...) {
                        result->error = std::current_exception();
                    }
                    if (state->measure && result->read_timed) {
                        result->read_finished = std::chrono::steady_clock::now();
                    }
                }
                // A failed stripe ends the page: mark it last so the drain still
                // accounts for it, and stop reading the rest.
                const bool failed = result->error != nullptr;
                if (failed) {
                    result->last = true;
                }
                if (result->last && read_started && test_hooks_ != nullptr &&
                    test_hooks_->read_finished) {
                    try {
                        test_hooks_->read_finished(
                            pagein.page->layer, pagein.page->expert);
                    } catch (...) {
                        if (result->error == nullptr) {
                            result->error = std::current_exception();
                        }
                    }
                }
                // WP_READER_H2D: the serial reader reads every stripe of a
                // page in program order on this one thread, so by the time
                // the LAST stripe (or a failed stripe, which forces
                // result->last above) is reached, every earlier stripe's
                // bytes are already in `staging` -- same-thread program
                // order needs no extra synchronization, unlike the
                // cross-thread stripe_parallel_ case. pagein.reader_h2d was
                // decided once for the whole batch at plan time (see
                // PageIn::reader_h2d).
                if (result->last && result->error == nullptr && pagein.reader_h2d) {
                    reader_h2d_upload(pagein, state->measure, *result);
                } else if (!result->last && result->error == nullptr && pagein.reader_h2d) {
                    // Not the last stripe: no complete page range to copy
                    // yet (see the whole-page-at-last-stripe design above).
                    // Mark uploaded with zero bytes/ns so drain_one_read
                    // treats this as a pure no-op instead of copying a
                    // partial stripe.
                    result->uploaded = true;
                }
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->ready.push_back(std::move(result));
                }
                state->cv.notify_one();
                if (failed) {
                    break;
                }
            }
            if (count_demand) {
                demand_reads_pending_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    // Process exactly ONE completed read: H2D it into its slot and mark the
    // entry ready. Split out of complete_batch so a drain can stop part-way.
    void drain_one_read(Batch & batch) {
        {
            std::unique_ptr<ReadResult> result;
            {
                std::unique_lock<std::mutex> lock(batch.state_->mutex);
                batch.state_->cv.wait(lock, [&]() {
                    return !batch.state_->ready.empty();
                });
                result = std::move(batch.state_->ready.front());
                batch.state_->ready.pop_front();
            }

            const PageIn & pagein =
                batch.state_->pageins[result->pagein_indexb];
            if (result->read_timed) {
                if (!batch.have_read_time_ || result->read_started < batch.first_read_) {
                    batch.first_read_ = result->read_started;
                }
                if (!batch.have_read_time_ || result->read_finished > batch.last_read_) {
                    batch.last_read_ = result->read_finished;
                }
                if (!batch.have_read_time_ || result->read_started < batch.first_read_issue_) {
                    batch.first_read_issue_ = result->read_started;
                }
                if (!batch.have_read_time_ || result->read_started > batch.last_read_issue_) {
                    batch.last_read_issue_ = result->read_started;
                }
                if (!batch.have_read_time_ || result->read_finished < batch.first_read_complete_) {
                    batch.first_read_complete_ = result->read_finished;
                }
                if (!batch.have_read_time_ || result->read_finished > batch.last_read_complete_) {
                    batch.last_read_complete_ = result->read_finished;
                }
                batch.have_read_time_ = true;
            }
            if (result->error != nullptr) {
                if (batch.first_error_ == nullptr) {
                    batch.first_error_ = result->error;
                }
            } else if (pagein.cpu_on_arrival) {
                if (result->last) {
                    if (pagein_log_ != nullptr) {
                        std::lock_guard<std::mutex> lock(*log_mutex_);
                        fprintf(pagein_log_, "%d %d\n", pagein.page->layer, pagein.page->expert);
                        fflush(pagein_log_);
                    }
                    if (spec_log_ != nullptr && !batch.state_->speculative) {
                        std::lock_guard<std::mutex> lock(*log_mutex_);
                        fprintf(spec_log_, "D %d %d\n", pagein.page->layer, pagein.page->expert);
                        fflush(spec_log_);
                    }
                    if (batch.state_->admit_host_on_read &&
                            result->staging && pagein.page->cache_id >= 0) {
                        host_tier_->store(pagein.page->cache_id, result->staging->get(),
                                          (size_t) pagein.page->size);
                    }
                    Batch::Entry & entry = batch.entries_[pagein.entry_index];
                    entry.cpu_staging = std::move(result->staging);
                    entry.ready = true;
                }
            } else {
                Slot & slot = slots_[pagein.slot_index];
                if (result->uploaded) {
                    // WP_READER_H2D already performed this page's H2D on the
                    // reader thread (the last-stripe branch in
                    // stripe_read_worker/read_worker), or this result is a
                    // non-last stripe's deliberate no-op (see
                    // ReadResult::uploaded and PageIn::reader_h2d) -- either
                    // way there is nothing left to copy here. Only account
                    // for the real copy; the no-op carries h2d_bytes == 0
                    // and must not be double-counted as a page.
                    if (result->h2d_bytes > 0) {
                        if (batch.state_->measure) {
                            batch.ns_h2d_    += result->h2d_ns;
                            batch.bytes_h2d_ += result->h2d_bytes;
                        }
                        ++batch.n_reader_h2d_;
                    }
                } else {
                const bool measure_h2d = batch.state_->measure;
                const std::chrono::steady_clock::time_point h2d_started =
                    measure_h2d ? std::chrono::steady_clock::now() :
                                  std::chrono::steady_clock::time_point();
                // Upload ONLY this stripe, at its own offset. With one stripe
                // this is the original whole-page (0, page.size) call.
                // Async mode: issue on the compute stream and fence the staging
                // buffer (see StagingPool::mark_in_flight). Compute correctness
                // needs no fence -- the graph runs on the same stream, after
                // this copy. NOTE ns_h2d then measures ISSUE time, not copy
                // time; the copy overlaps reads/submit and the A/B metric is
                // the request wall.
                if (pagein.page->device_size != pagein.page->size) {
                    bool async_copy = false;
                    for_each_page_chunk(
                        *pagein.page, result->offset, result->len,
                        [&](size_t source_offset, size_t device_offset, size_t n) {
                        const char * source =
                            (const char *) result->staging->get() + result->offset + source_offset;
                        if (!batch.state_->speculative && staging_.copy_stream_h2d() &&
                            ggml_backend_cuda_wp_copy_tensor_async != nullptr &&
                            ggml_backend_cuda_wp_copy_tensor_async(
                                backend_, slot.raw, source, device_offset, n)) {
                            async_copy = true;
                        } else if (!batch.state_->speculative && staging_.async_h2d()) {
                            ggml_backend_tensor_set_async(
                                backend_, slot.raw, source, device_offset, n);
                            async_copy = true;
                        } else {
                            ggml_backend_tensor_set(slot.raw, source, device_offset, n);
                        }
                    });
                    zero_quantized_member_padding(slot.raw, *pagein.page);
                    if (async_copy) {
                        staging_.mark_in_flight(result->staging->get());
                    }
                } else if (!batch.state_->speculative && staging_.copy_stream_h2d() &&
                           ggml_backend_cuda_wp_copy_tensor_async != nullptr &&
                           ggml_backend_cuda_wp_copy_tensor_async(
                               backend_, slot.raw, (const char *) result->staging->get() + result->offset,
                               result->offset, result->len)) {
                    staging_.mark_in_flight(result->staging->get());
                } else if (!batch.state_->speculative && staging_.async_h2d()) {
                    ggml_backend_tensor_set_async(
                        backend_,
                        slot.raw, (const char *) result->staging->get() + result->offset,
                        result->offset, result->len);
                    staging_.mark_in_flight(result->staging->get());
                } else {
                    ggml_backend_tensor_set(
                        slot.raw, (const char *) result->staging->get() + result->offset,
                        result->offset, result->len);
                }
                if (measure_h2d) {
                    batch.ns_h2d_ +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - h2d_started).count();
                    batch.bytes_h2d_ += result->len;
                }
                }
                // Everything below publishes the page to the compute path and so
                // must happen EXACTLY ONCE, on the final stripe -- otherwise a
                // half-uploaded slot becomes visible, and the page-in log and LRU
                // tick would fire once per stripe.
                if (result->last) {
                    // WP_PAGEIN_LOG=path: append "<layer> <expert>" for every page
                    // actually READ from disk. Intersecting the two 2026 workers'
                    // logs measures whether residency-affinity routing keeps their
                    // caches disjoint, or whether they fetch the same pages twice.
                    // Default off; one fprintf per pagein, negligible against a
                    // 13.37 MB O_DIRECT read.
                    if (pagein_log_ != nullptr) {
                        std::lock_guard<std::mutex> lock(*log_mutex_);
                        fprintf(pagein_log_, "%d %d\n", pagein.page->layer, pagein.page->expert);
                        // The harness SIGKILLs workers at teardown, so a buffered
                        // stream is lost entirely -- the first run produced two
                        // 0-byte logs. One fflush per 13.37 MB O_DIRECT read is free.
                        fflush(pagein_log_);
                    }
                    // The same event into the ordered stream, as a DEMAND read.
                    // Deliberately duplicated rather than joined against
                    // WP_PAGEIN_LOG after the fact: what makes this line useful is
                    // its POSITION relative to the S line for the same page, and a
                    // separate file cannot express that.
                    // NEVER for a speculative batch: those pages were already
                    // logged "S" at submit, and a second line here would claim
                    // speculation provoked the very demand read it prevented.
                    if (spec_log_ != nullptr && !batch.state_->speculative) {
                        std::lock_guard<std::mutex> lock(*log_mutex_);
                        fprintf(spec_log_, "D %d %d\n", pagein.page->layer, pagein.page->expert);
                        fflush(spec_log_);
                    }
                    // Land the full page (all stripes share one staging lease)
                    // in the host tier so the next evict skips D2H. CPU memcpy
                    // of 12.75 MiB; not a GPU sync.
                    if (batch.state_->admit_host_on_read &&
                        result->staging && pagein.page->cache_id >= 0) {
                        host_tier_->store(pagein.page->cache_id,
                                         result->staging->get(),
                                         (size_t) pagein.page->size);
                    }
                    slot.valid = true;
                    slot.key   = {
                        pagein.page->layer, pagein.page->expert
                    };
                    slot_index_[slot_key(pagein.page->layer, pagein.page->expert)] =
                        pagein.slot_index;
                    slot.cache_id = pagein.page->cache_id;
                    slot.page     = pagein.page;
                    slot.size     = pagein.page->size;
                    slot.tick  = ++tick_;
                    slot.uses  = lfu_history_enabled_ ? history_uses(*pagein.page) :
                                  evict_age_ + 1;
                    Batch::Entry & entry =
                        batch.entries_[pagein.entry_index];
                    entry.loaded = {
                        slot.buffer, slot.raw->data
                    };
                    entry.ready = true;
                }
            }
            // received_ counts PAGES, not stripes -- complete_batch's loop is
            // bounded by pageins.size().
            const bool page_done = result->last;
            result.reset();
            if (page_done) {
                ++batch.received_;
            }
        }
    }

    // Drain until every entry below entry_end that needs a page-in is ready.
    // Entries at or above entry_end whose reads happen to land first are still
    // processed -- their H2D is done here, so nothing is dropped or re-done.
    // Does NOT join the reader threads or finalise timings; complete_batch does.
    void complete_batch_upto(Batch & batch, size_t entry_end) {
        if (batch.completed_) {
            return;
        }
        const auto range_pending = [&]() {
            for (const PageIn & pagein : batch.state_->pageins) {
                if (pagein.entry_index < entry_end &&
                    !batch.entries_[pagein.entry_index].ready) {
                    return true;
                }
            }
            return false;
        };
        // received_ bounds the loop: a FAILED read never sets ready, so waiting
        // on readiness alone would hang on exactly the error path.
        while (batch.received_ < batch.state_->pageins.size() && range_pending()) {
            drain_one_read(batch);
        }
        // A partial drain may have issued H2D copies on the dedicated copy
        // stream. Fence those copies before compute starts; complete() records
        // the final fence too, but that is after the overlapped compute point.
        if (batch.copy_event_ != nullptr &&
                !staging_.record_copy_event(batch.copy_event_)) {
            ggml_backend_event_free(batch.copy_event_);
            batch.copy_event_ = nullptr;
        }
    }

    void complete_batch(Batch & batch) {
        if (batch.completed_) {
            return;
        }
        // A Batch can reach here with no state_: ensure_batch sets owner_
        // before it builds the state, so if it throws in between, its catch
        // releases pins and rethrows, and THEN ~Batch() runs abandon_batch()
        // on a half-built object. Dereferencing state_ there segfaults, which
        // kills the worker and destroys the real exception -- the spine only
        // ever sees "worker died while computing". Nothing was queued without
        // a state, so there is nothing to drain.
        if (batch.state_ == nullptr) {
            batch.completed_ = true;
            return;
        }

        while (batch.received_ < batch.state_->pageins.size()) {
            drain_one_read(batch);
        }

        for (std::thread & worker : batch.workers_) {
            worker.join();
        }
        batch.workers_.clear();
        batch.completed_ = true;
        if (batch.copy_event_ != nullptr && !staging_.record_copy_event(batch.copy_event_)) {
            ggml_backend_event_free(batch.copy_event_);
            batch.copy_event_ = nullptr;
        }
        if (batch.have_read_time_) {
            batch.ns_read_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                batch.last_read_ - batch.first_read_).count();
            batch.ns_read_issue_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                batch.last_read_issue_ - batch.first_read_issue_).count();
            batch.ns_read_complete_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                batch.last_read_complete_ - batch.first_read_issue_).count();
        }
        batch.n_read_inflight_max_ = batch.state_->read_inflight_max.load(
            std::memory_order_relaxed);
        if (batch.first_error_ != nullptr) {
            std::rethrow_exception(batch.first_error_);
        }
    }

    void cancel_workers(Batch & batch) {
        if (batch.state_ != nullptr && !batch.workers_.empty()) {
            {
                std::lock_guard<std::mutex> lock(batch.state_->mutex);
                batch.state_->cancel = true;
            }
            batch.state_->cv.notify_all();
            for (std::thread & worker : batch.workers_) {
                worker.join();
            }
            batch.workers_.clear();
        }
    }

    void release_pins(Batch & batch) noexcept {
        for (const Batch::Entry & entry : batch.entries_) {
            if (entry.slot_index == std::numeric_limits<size_t>::max()) {
                continue;
            }
            Slot & slot = slots_[entry.slot_index];
            if (slot.pin_count > 0) {
                --slot.pin_count;
            }
        }
        batch.entries_.clear();
    }

    void abandon_batch(Batch & batch) noexcept {
        if (!batch.completed_) {
            try {
                complete_batch(batch);
            } catch (...) {
            }
        }
        release_pins(batch);
    }

    WorkerLogFiles * logs_ = nullptr;
    FILE * pagein_log_ = nullptr;
    FILE * spec_log_ = nullptr;
    std::mutex * log_mutex_ = nullptr;

    // Per-slot stride inside an arena. layout_sliced_pages also makes the page
    // divisible by each role's type size when arena ids are requested.
    uint64_t arena_slot_stride(uint64_t size) const {
        const size_t align =
            ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend_));
        const uint64_t a = align == 0 ? 1 : (uint64_t) align;
        if (size > UINT64_MAX - (a - 1)) {
            throw std::overflow_error("expert arena slot stride overflows");
        }
        return ((size + a - 1) / a) * a;
    }

    // Allocate the arena buffers that back every slot. A few large allocations
    // instead of one per slot -- see the comment on Slot::buffer for why.
    // Pin every CPU slot arena into RAM so the CPU expert tier can never be
    // swapped/compressed out. No-op on GPU backends (VRAM) and on non-Linux.
    // WP_PIN_CPU_TIER=0 opts out (mirrors WP_PIN_HOST's escape hatch); default
    // ON, because a swapped CPU tier is a correctness-of-purpose bug, not a
    // tunable. Reports the total pinned and any failure loudly -- an EPERM here
    // (RLIMIT_MEMLOCK too low) means the tier is silently still swappable, and
    // that must not pass unnoticed.
    void mlock_cpu_arenas() {
#if defined(__linux__)
        const char * const backend_name = ggml_backend_name(backend_);
        if (backend_name == nullptr || std::strstr(backend_name, "CPU") == nullptr) {
            return;   // GPU arena is device memory; mlock does not apply.
        }
        if (const char * e = std::getenv("WP_PIN_CPU_TIER"); e != nullptr && e[0] == '0') {
            std::fprintf(stderr, "wp expert worker: WP_PIN_CPU_TIER=0 -- CPU expert tier "
                         "left swappable (%zu arenas)\n", arenas_.size());
            return;
        }
        size_t pinned = 0, failed = 0;
        for (const buffer_ptr & arena : arenas_) {
            void * base = ggml_backend_buffer_get_base(arena.get());
            const size_t bytes = ggml_backend_buffer_get_size(arena.get());
            if (base == nullptr || bytes == 0) {
                continue;
            }
            if (mlock(base, bytes) == 0) {
                pinned += bytes;
            } else {
                const int err = errno;
                ++failed;
                std::fprintf(stderr, "wp expert worker: mlock(%zu) on CPU expert arena FAILED "
                             "(%s) -- this arena stays swappable. "
                             "Raise RLIMIT_MEMLOCK (ulimit -l unlimited) if pinning is required.\n",
                             bytes, std::strerror(err));
            }
        }
        if (pinned > 0) {
            std::fprintf(stderr, "wp expert worker: pinned CPU expert tier into RAM, "
                         "%.1f MiB across %zu arenas (mlock)%s\n",
                         (double) pinned / 1048576.0, arenas_.size(),
                         failed ? " -- SOME ARENAS FAILED, tier partially swappable" : "");
        }
#endif
    }

    void allocate_slot_arenas() {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
        uint64_t total = 0;
        for (const SlotClass & slot_class : resources_.slot_classes) {
            total += arena_slot_stride(slot_class.size) * (uint64_t) slot_class.slots;
        }
        if (total == 0) {
            return;
        }
        // Arena ids need one base+stride address space because a ggml tensor cannot cross backend buffers.
        const char * const arena_env = std::getenv("WP_EXPERT_ARENA_ID");
        const char * const backend_name = ggml_backend_name(backend_);
        const bool single_id_arena =
            arena_env != nullptr && std::strtol(arena_env, nullptr, 10) == 1 &&
            backend_name != nullptr &&
            (std::strstr(backend_name, "ROCm") != nullptr ||
             std::strstr(backend_name, "CUDA") != nullptr ||
             std::strstr(backend_name, "Vulkan") != nullptr) &&
            resources_.slot_classes.size() == 1;
        if (single_id_arena && total > (uint64_t) SIZE_MAX) {
            throw std::overflow_error("expert slot arena is too large");
        }
        // Respect the backend's max single-allocation size; split across as many
        // arenas as that requires. Vulkan reports a real cap here (and a 4 GB
        // buffer would exceed maxStorageBufferRange on Polaris anyway).
        size_t max_buf = ggml_backend_buft_get_max_size(buft);
        if (single_id_arena) {
            if (max_buf == 0 || max_buf == SIZE_MAX || total <= max_buf) {
                buffer_ptr buf(ggml_backend_buft_alloc_buffer(buft, (size_t) total));
                if (buf) {
                    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    // see the zeroing note on the resident-page allocation above
                    ggml_backend_buffer_clear(buf.get(), 0);
                    arenas_.push_back(std::move(buf));
                    return;
                }
            }
            std::fprintf(stderr, "WARN wp expert worker: WP_EXPERT_ARENA_ID single allocation failed; using split arenas\n");
        }
        if (max_buf == 0 || max_buf > SIZE_MAX / 2) {
            max_buf = (size_t) 1 << 30;
        }
        if (resources_.slot_classes.size() > 1) {
            // Keep each size class in its own arena set. Invariant: the sum of
            // allocated arena bytes must not exceed the requested slot budget,
            // and every planned slot must have a home.
            arena_class_starts_.reserve(resources_.slot_classes.size());
            uint64_t allocated = 0;
            for (const SlotClass & slot_class : resources_.slot_classes) {
                const uint64_t stride = arena_slot_stride(slot_class.size);
                arena_class_starts_.push_back(arenas_.size());
                if ((uint64_t) slot_class.slots > UINT64_MAX / stride) {
                    throw std::overflow_error("expert arena size overflows");
                }
                const uint64_t class_total = stride * (uint64_t) slot_class.slots;
                if (class_total > resources_.slot_budget_bytes - allocated) {
                    throw std::runtime_error("expert slot arena plan exceeds slot budget");
                }
                uint64_t arena_bytes = (uint64_t) max_buf / stride * stride;
                if (arena_bytes == 0) {
                    arena_bytes = stride;
                }
                uint64_t remaining = class_total;
                while (remaining > 0) {
                    const uint64_t want = std::min(arena_bytes, remaining);
                    buffer_ptr buf(ggml_backend_buft_alloc_buffer(buft, (size_t) want));
                    if (!buf) {
                        throw std::runtime_error(
                            "failed to allocate expert slot arena of " + std::to_string(want) + " bytes");
                    }
                    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    // see the zeroing note on the resident-page allocation above
                    ggml_backend_buffer_clear(buf.get(), 0);
                    arenas_.push_back(std::move(buf));
                    remaining -= want;
                    allocated += want;
                }
            }
            return;
        }
        // Uniform-class path: cap each arena at a whole number of the stride.
        uint64_t max_stride = 0;
        for (const SlotClass & slot_class : resources_.slot_classes) {
            max_stride = std::max(max_stride, arena_slot_stride(slot_class.size));
        }
        if (max_stride == 0) {
            return;
        }
        uint64_t arena_bytes = (uint64_t) max_buf / max_stride * max_stride;
        if (arena_bytes == 0) {
            arena_bytes = max_stride;
        }
        uint64_t remaining = total;
        while (remaining > 0) {
            const uint64_t want = std::min(arena_bytes, remaining);
            buffer_ptr buf(ggml_backend_buft_alloc_buffer(buft, (size_t) want));
            if (!buf) {
                throw std::runtime_error(
                    "failed to allocate expert slot arena of " + std::to_string(want) + " bytes");
            }
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // see the zeroing note on the resident-page allocation above
            ggml_backend_buffer_clear(buf.get(), 0);
            arenas_.push_back(std::move(buf));
            remaining -= want;
        }
    }

    // Carve one slot out of an already-allocated arena buffer at `offset`.
    // The slot does NOT own the buffer; arenas_ does.
    Slot make_slot_in(ggml_backend_buffer_t arena, uint64_t offset, uint64_t capacity) {
        if (capacity == 0 ||
            capacity > (uint64_t) std::numeric_limits<size_t>::max() ||
            capacity > (uint64_t) std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error("invalid expert slot capacity");
        }
        if (arena == nullptr) {
            throw std::runtime_error("expert slot arena is null");
        }
        // The arena must actually contain the slot. A silent overrun here would
        // scribble into the neighbouring slot's page, which is exactly the
        // failure mode the CUDA slot-padding fix existed to end -- so it is a
        // hard check, not an assert compiled out in release.
        const uint64_t arena_bytes = (uint64_t) ggml_backend_buffer_get_size(arena);
        if (offset > arena_bytes || capacity > arena_bytes - offset) {
            throw std::runtime_error("expert slot does not fit its arena");
        }

        Slot slot;
        slot.capacity = capacity;
        slot.buffer   = arena;
        slot.offset   = offset;

        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 2,
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        slot.ctx.reset(ggml_init(params));
        if (!slot.ctx) {
            throw std::runtime_error("failed to allocate expert slot metadata");
        }
        slot.raw = ggml_new_tensor_1d(
            slot.ctx.get(), GGML_TYPE_I8, (int64_t) capacity);
        slot.raw->buffer = arena;
        uint8_t * const base = (uint8_t *) ggml_backend_buffer_get_base(arena);
        if (base == nullptr) {
            throw std::runtime_error("expert slot arena has no base pointer");
        }
        slot.raw->data = base + offset;
        if (ggml_backend_buffer_init_tensor(arena, slot.raw) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize expert slot tensor");
        }
        return slot;
    }

    int fd_for(const fs::path & path) {
#if defined(__linux__)
        const std::string key = path.string();
        auto it = fds_.find(key);
        if (it != fds_.end()) {
            return it->second;
        }
        int fd = -1;
        if (read_direct_) {
            fd = open(key.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
            if (fd < 0) {
                g_read_direct_fallback.store(true, std::memory_order_relaxed);
            }
        }
        if (fd < 0) {
            fd = open(key.c_str(), O_RDONLY | O_CLOEXEC);
        }
        if (fd < 0) {
            throw std::runtime_error(
                "failed to open expert shard " + key + ": " + std::strerror(errno));
        }
        fds_.emplace(key, fd);
        return fd;
#else
        (void) path;
        throw std::runtime_error("O_DIRECT expert slots require Linux");
#endif
    }

    // Read [offset, offset+len) of a page. Direct reads are opt-in; an
    // alignment refusal retries through a buffered fd.
    void read_page_range(const ExpertPage & page, int fd, void * dst,
                         size_t offset, size_t len, BatchState * state = nullptr) {
#if defined(__linux__)
        struct ReadInflightGuard {
            std::atomic<size_t> * count = nullptr;
            ~ReadInflightGuard() {
                if (count != nullptr) {
                    count->fetch_sub(1, std::memory_order_relaxed);
                }
            }
        } guard;
        if (state != nullptr) {
            const size_t active = state->read_inflight.fetch_add(
                1, std::memory_order_relaxed) + 1;
            size_t previous = state->read_inflight_max.load(std::memory_order_relaxed);
            while (previous < active &&
                   !state->read_inflight_max.compare_exchange_weak(
                       previous, active, std::memory_order_relaxed)) {
            }
            guard.count = &state->read_inflight;
        }
        ssize_t n = -1;
        const bool measure_read = g_read_path_stats.enabled();
        const std::chrono::steady_clock::time_point read_started =
            measure_read ? std::chrono::steady_clock::now() :
                            std::chrono::steady_clock::time_point();
        do {
            n = pread(fd, dst, len, (off_t) (page.offset + (uint64_t) offset));
        } while (n < 0 && errno == EINTR);
        if (n < 0 && read_direct_ && errno == EINVAL) {
            g_read_direct_fallback.store(true, std::memory_order_relaxed);
            const int buffered_fd = open(page.blob.c_str(), O_RDONLY | O_CLOEXEC);
            if (buffered_fd >= 0) {
                do {
                    n = pread(buffered_fd, dst, len,
                              (off_t) (page.offset + (uint64_t) offset));
                } while (n < 0 && errno == EINTR);
                close(buffered_fd);
            }
        }
        if (measure_read) {
            const std::chrono::steady_clock::time_point read_finished =
                std::chrono::steady_clock::now();
            const uint64_t elapsed = (uint64_t) std::chrono::duration_cast<
                std::chrono::nanoseconds>(read_finished - read_started).count();
            const uint64_t started_ns = (uint64_t) std::chrono::duration_cast<
                std::chrono::nanoseconds>(read_started.time_since_epoch()).count();
            const uint64_t finished_ns = (uint64_t) std::chrono::duration_cast<
                std::chrono::nanoseconds>(read_finished.time_since_epoch()).count();
            g_read_path_stats.record(
                n > 0 ? std::min<size_t>((size_t) n, len) : 0,
                elapsed, started_ns, finished_ns);
        }
        if (n < 0 || (size_t) n != len) {
            throw std::runtime_error(
                "short expert read from " + page.blob.string() +
                ": got " + std::to_string(n) + " want " + std::to_string(len) +
                " at +" + std::to_string(offset));
        }
#else
        (void) page;
        (void) fd;
        (void) dst;
        (void) offset;
        (void) len;
        (void) state;
#endif
    }

    // Split a page into aligned stripes. Every stripe but the last is a
    // multiple of 4096 (the alignment the staging pool already guarantees and
    // O_DIRECT requires); the last absorbs the remainder. Returns a single
    // whole-page stripe when striping is off or the page is too small to be
    // worth splitting -- that path is byte-for-byte the old behaviour.
    std::vector<std::pair<size_t, size_t>> stripe_plan(uint64_t page_size,
                                                       size_t   n_pageins) const {
        std::vector<std::pair<size_t, size_t>> out;
        const size_t total = (size_t) page_size;
        constexpr size_t kAlign   = 4096;
        // kMinPart: don't split a stripe below this many bytes. THIS VALUE MUST
        // TRACK THE PAGE SIZE OF THE RIG, NOT BE A FIXED CONSTANT -- see
        // stripe_min_part_ below for why a 1 MiB floor (correct for the OLD
        // 12.75 MiB whole-expert page) silently disables striping on the sliced
        // rig's ~1.5-9 MiB width-slice pages.
        const size_t kMinPart = stripe_min_part_;
        if (read_chunk_bytes_ != 0) {
            const size_t part = read_chunk_bytes_ < total ? read_chunk_bytes_ : total;
            if (part < total) {
                size_t off = 0;
                while (off + part < total) {
                    out.emplace_back(off, part);
                    off += part;
                }
                out.emplace_back(off, total - off);
            } else {
                out.emplace_back(0, total);
            }
            if (test_hooks_ != nullptr && test_hooks_->stripe_planned) {
                test_hooks_->stripe_planned(page_size, n_pageins, out.size());
            }
            return out;
        }
        size_t n = read_stripes_;
        // *** GATE ON THE BATCH BEING READ-SPARSE. ***
        // Striping only pays when a page has no OTHER page to overlap against.
        // With many page-ins in one batch, drain_one_read is already uploading
        // page N while the reader threads pull N+1, so splitting each page just
        // multiplies preads and tensor_set calls.
        // MEASURED 2026-08-05, identical block counts both arms: decode-side
        // dispatch wait 72.62 -> 66.04 s (-9.1%, and consistent in sign across
        // decode/verify-43/verify-3), but PREFILL wait 16.50 -> 18.26 s
        // (+10.7%). Prefill averages 31.4 page-ins per request; decode averages
        // 0.212 and verify 0.465. So stripe the sparse batches and leave the
        // dense ones whole.
        if (n_pageins > stripe_max_pageins_) {
            n = 1;
        }
        if (n > 1 && total / n < kMinPart) {
            n = total / kMinPart;
        }
        if (n <= 1) {
            out.emplace_back(0, total);
        } else {
            const size_t part = ((total / n) / kAlign) * kAlign;
            if (part == 0) {
                out.emplace_back(0, total);
            } else {
                size_t off = 0;
                while (off + part < total) {
                    out.emplace_back(off, part);
                    off += part;
                }
                out.emplace_back(off, total - off);   // remainder tail
            }
        }
        // Test-only observation point: reports how many stripes THIS call
        // actually chose for (page_size, n_pageins), independent of the
        // per-page read_started/read_finished hooks (which fire once per
        // page, not once per stripe, and so cannot tell a test whether the
        // sliced-page-size fix above actually engaged striping).
        if (test_hooks_ != nullptr && test_hooks_->stripe_planned) {
            test_hooks_->stripe_planned(page_size, n_pageins, out.size());
        }
        return out;
    }

    // WP_EXPERT_READ_STRIPES=<n>: read each expert page in n aligned pieces so
    // the dispatch thread uploads stripe k while the reader thread reads k+1.
    // 1 = the original whole-page read, byte-for-byte.
    //
    // WHAT THIS DOES AND DOES NOT FIX. Across pages, read and h2d ALREADY
    // overlap: drain_one_read uploads page N on the dispatch thread while the
    // reader threads pull page N+1. So this only bites when there are too few
    // pages in flight to overlap each other -- i.e. DECODE, where the RX 480
    // averages 0.212 page-ins per request and a lone page really is read-then-
    // upload serial. At prefill it averages 31.4 page-ins per request and is
    // already deeply pipelined, so expect ~nothing there.
    // Measured per page-in on the 480 at decode: read ~5.4 ms, h2d ~3.63 ms.
    // NOTE ns_read is a SPAN and ns_h2d is a SUM, so those two do not simply add
    // -- do not size this change by adding them.
    //
    // 2026-08-17: on the DS4-Flash SLICED rig this knob alone stopped being
    // enough -- see stripe_min_part_from_env() below. The old whole-expert
    // page (12.75 MiB) was large enough that n=4 always cleared the old 1 MiB
    // floor; the sliced page (~1.5-9 MiB, e.g. the 1.594 MiB slice page noted
    // on ResidentExpertPool::Slot) is small enough that it often did not, so
    // stripe_plan silently fell back to n=1 (no striping) on exactly the
    // small lone-miss decode pages this knob exists to pipeline.
    static size_t read_stripes_from_env() {
        const char * e = std::getenv("WP_EXPERT_READ_STRIPES");
        if (e == nullptr || e[0] == '\0') {
            return 4;
        }
        const long parsed = std::strtol(e, nullptr, 10);
        return parsed > 0 ? (size_t) parsed : (size_t) 1;
    }

    // WP_EXPERT_STRIPE_MAX_PAGEINS=<n>: only stripe batches with at most n
    // page-ins. Above that the pages already overlap each other. Default 4
    // covers decode (0.212 page-ins/request) and verify (0.465) while leaving
    // prefill (31.4) on the whole-page read, which is where striping measured a
    // +10.7% regression.
    static size_t stripe_max_pageins_from_env() {
        const char * e = std::getenv("WP_EXPERT_STRIPE_MAX_PAGEINS");
        if (e == nullptr || e[0] == '\0') {
            return 4;
        }
        const long parsed = std::strtol(e, nullptr, 10);
        return parsed >= 0 ? (size_t) parsed : (size_t) 4;
    }

    // WP_EXPERT_STRIPE_MIN_PART=<bytes>: floor below which stripe_plan will
    // not split a page further (rounded down to a 4096 multiple at use).
    //
    // WHY THIS EXISTS. stripe_plan's old floor was a hardcoded 1 MiB
    // (`kMinPart`), sized for the pre-sliced rig's ~12.75 MiB whole-expert
    // page. On THIS (DS4-Flash sliced) rig the catalog's own page-size
    // comment (ResidentExpertPool::Slot, above) records a 1671168-byte
    // (1.594 MiB) slice page, and the decode miss pages this worker actually
    // sees run ~1.5-9 MiB -- close enough to the 1 MiB floor that
    // total/read_stripes_ (4) undercuts it: 1.594 MiB / 4 = 408 KiB < 1 MiB,
    // so the old floor forced n = total/kMinPart = 1, i.e. NO STRIPING AT
    // ALL, silently, for exactly the smallest and most common decode-miss
    // pages. That defeated the striped read/H2D pipeline (read_worker /
    // drain_one_read: reader thread reads stripe k+1 on its own thread while
    // the dispatch thread uploads stripe k) for a lone small slice read --
    // the read and the H2D ran fully serially because there was only ever
    // one "stripe" to run them on.
    //
    // 256 KiB keeps stripe_plan from collapsing back to n=1 until a page is
    // itself under ~1 MiB (256 KiB * read_stripes_ default of 4), which is
    // below every sliced page size observed on this rig, while still leaving
    // each stripe's O_DIRECT pread comfortably above the 4096-byte alignment
    // floor. It does NOT touch WP_EXPERT_STRIPE_MAX_PAGEINS (still default
    // 4): prefill's ~31 page-ins/request stays on the whole-page path
    // regardless of this value, so this cannot reintroduce the +10.7% prefill
    // regression measured 2026-08-05 for striping dense batches.
    static size_t stripe_min_part_from_env() {
        const char * e = std::getenv("WP_EXPERT_STRIPE_MIN_PART");
        if (e == nullptr || e[0] == '\0') {
            return 256u << 10;   // 256 KiB
        }
        const long parsed = std::strtol(e, nullptr, 10);
        return parsed > 0 ? (size_t) parsed : (size_t) (256u << 10);
    }

    ggml_backend_t             backend_ = nullptr;
    ResourcePlan               resources_;
    StagingPool                staging_;
    const bool                 cpu_on_arrival_enabled_ = cpu_on_arrival_enabled_from_env();
    const size_t                cpu_on_arrival_cap_ = cpu_on_arrival_cap_from_env();
    size_t                     read_stripes_ = read_stripes_from_env();
    size_t                     stripe_max_pageins_ = stripe_max_pageins_from_env();
    size_t                     stripe_min_part_ = stripe_min_part_from_env();
    // 0 keeps the existing reader count; a positive value caps page/chunk
    // reads for this batch, subject to the staging pool and connection quota.
    size_t                     read_inflight_ = read_inflight_from_env();
    size_t                     read_chunk_bytes_ = read_chunk_bytes_from_env();
    const bool                 read_direct_ = read_direct_from_env();
    // WP_EXPERT_STRIPE_PARALLEL=1 -- stripes of one page are claimed by
    // MULTIPLE reader threads concurrently (QD>1 per page) instead of read
    // serially by the page's claimer. Default off: bare runs stay on the
    // measured 2026-08-05 serial-stripe pipeline.
    const bool                 stripe_parallel_ = [] {
        const char * e = std::getenv("WP_EXPERT_STRIPE_PARALLEL");
        return e != nullptr && e[0] == '1';
    }();
    // WP_READER_H2D=1 -- move a page-in's H2D off the connection thread's
    // g_worker_gpu_mutex-held critical path (drain_one_read) onto the
    // reader thread that produced the bytes, once the page's last stripe
    // has landed in the staging buffer (see stripe_read_worker /
    // read_worker and the PageIn::reader_h2d comment). Default OFF: unset
    // or any value other than "1" reproduces exactly today's behaviour, no
    // matter the backend.
    //
    // Gated to CUDA/HIP only. The copy itself does NOT go through
    // ggml_backend_tensor_set / ggml_backend_cuda_buffer_set_tensor -- on
    // gfx1201 (R9700) that takes the MAD-114 legacy-stream synchronous
    // hipMemcpy branch (ggml_cuda_device_needs_mad114_legacy_memcpy,
    // ggml/src/ggml-cuda/ggml-cuda.cu), and the legacy stream implicitly
    // synchronizes with every other blocking stream on the device --
    // including a connection thread's concurrent HIP-graph capture
    // (WP_HIP_GRAPHS), which SIGABRTs ("operation would make the legacy
    // stream depend on a capturing blocking stream": this worker DOES run
    // graph capture through ggml-cuda's generic path, not only the
    // arena path -- the 2026-08-25 live crash this gate exists to avoid).
    // Instead every reader-thread copy goes through
    // tensor_set_page_range_reader -> ggml_backend_cuda_wp_reader_copy, a
    // dedicated per-device NON-BLOCKING stream defined in ggml-cuda.cu that
    // never implicitly syncs with the legacy stream, so it cannot collide
    // with a concurrent capture. See that function's comment for the full
    // ordering argument for why a host-synchronized copy on that stream is
    // still safely visible to any kernel launched afterward (including on
    // gfx1201).
    //
    // NOT enabled for Vulkan: ggml_backend_vk_buffer_set_tensor
    // (ggml-vulkan.cpp) funnels through ggml_vk_buffer_write, which is not
    // documented or otherwise provably safe to call from multiple threads
    // concurrently without external synchronization the way the CUDA/HIP
    // path is. Absent that proof this stays off there -- and off for any
    // other/unknown backend -- regardless of the env var.
    const bool                 reader_h2d_enabled_ = [this] {
        const char * e = std::getenv("WP_READER_H2D");
        if (e == nullptr || e[0] != '1') {
            return false;
        }
        const char * const name = ggml_backend_name(backend_);
        return name != nullptr &&
               (std::strstr(name, "CUDA") != nullptr ||
                std::strstr(name, "ROCm") != nullptr);
    }();
    TestHooks *                test_hooks_ = nullptr;
    std::unique_ptr<wp::HostTier> owned_host_tier_;
    wp::HostTier *              host_tier_ = nullptr;
    bool                       host_victim_enabled_ = false;
    // WP_EXPERT_FILL_HOST_ON_READ=1 keeps decode reads in the host tier so a
    // later eviction can skip the synchronous D2H. Default off for A/B.
    const bool                 fill_host_on_read_ = [] {
        const char * e = std::getenv("WP_EXPERT_FILL_HOST_ON_READ");
        return e != nullptr && e[0] == '1';
    }();
    // WP_EXPERT_DEMOTE_D2H=1 forces the old synchronous D2H path, including
    // when fill-on-read is enabled.
    const bool                 demote_d2h_ = [] {
        const char * e = std::getenv("WP_EXPERT_DEMOTE_D2H");
        return e != nullptr && e[0] == '1';
    }();
    // *** TWO LRU BANDS, NOT ONE COUNTER. ***
    // Demand ticks start at kDemandTickBase and prefetch ticks at 0, so EVERY
    // prefetched page is strictly older than EVERY demand-touched page and is
    // chosen as a victim first. A single counter cannot express this: stamping a
    // speculative page with "the tick at read time" TIES with the demand that produced that
    // tick, and select_victim breaks a tie by slot index -- so a speculative page would
    // evict the very page that was just used, which is the pollution this is
    // here to prevent. Within each band the ordinary LRU order still holds, so
    // an older prefetch is evicted before a newer one.
    // The bands cannot meet: reaching kDemandTickBase needs 2^40 prefetched
    // pages, which at 12.75 MB each is ~14 EB of reads.
    static constexpr uint64_t  kDemandTickBase = 1ull << 40;
    uint64_t                   tick_      = kDemandTickBase;
    uint64_t                   spec_tick_ = 0;
    // Use count of the last page evicted. New pages are admitted here rather
    // than at zero. WP_EXPERT_LFU=0 falls back to pure LRU so the two policies
    // can be A/B'd on the same binary.
    uint64_t                   evict_age_ = 0;
    uint64_t                   evictions_ = 0;
    // WP_EXPERT_SPEC_LEASE -- evictions a speculative page survives before it
    // becomes an ordinary eviction candidate. 0 restores the old first-victim
    // behaviour exactly, so the lease is A/B-able on one binary.
    const uint64_t             spec_lease_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_LEASE");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 64;
        return v > 0 ? (uint64_t) v : (uint64_t) 0;
    }();
    // WP_EXPERT_SPEC_LEASE_PREDICTED -- the lease for a page fetched on a GUESS.
    //
    // A flat lease prices certainty and speculation the same, and measured, that
    // is what broke the predictor: ~1.8 predicted tokens per block displaced ~200
    // ground-truth pages, because a predicted page held its slot exactly as long
    // as a page the target was already committed to. Short by default so a guess
    // can occupy capacity nothing better wants and give it up first.
    const uint64_t             spec_lease_pred_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_LEASE_PREDICTED");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 4;
        return v > 0 ? (uint64_t) v : (uint64_t) 0;
    }();
    // WP_EXPERT_SPEC_MAX_SLOTS -- cap on how many pool slots may concurrently
    // hold an UNCONFIRMED speculative page (a page landed by spec_pagein_submit
    // that no demand hit has touched yet). Ported from the sister subsystem's
    // xlayer_max_slots_ (wp-pager.cpp): same idea, VRAM-slot pool instead of
    // wp-pager's. Speculative and demand pages share ONE pool here with no cap
    // at all -- measured 2026-08-19: 5,845 speculative page-ins against a
    // 3,350-slot pool (1.7x the pool) with monotonically decaying throughput
    // (3.016 -> 2.256 -> 1.767 t/s) versus no decay with prefetch off. DEFAULT 0
    // = UNCAPPED = today's behaviour exactly; the operator sets this explicitly.
    // A slot count, not a rate: enforced once, at submission, against a live
    // occupancy count (n_spec_pending_), never against a moving window.
    const long                 spec_max_slots_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_MAX_SLOTS");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 0;
        return v > 0 ? v : 0;
    }();
    // Live count of pool slots currently holding an unconfirmed speculative
    // page -- i.e. slots with spec_pending==true. Maintained at every site that
    // makes a slot speculative (retire_spec_batch), promotes one (the demand-hit
    // branch in ensure_batch), or evicts one (the victim-eviction branch in
    // ensure_batch). Do not read this as "leased": a lease can expire while the
    // page is still occupying the slot unconfirmed, and that must still count
    // against the budget until the slot is either hit or evicted.
    size_t                      n_spec_pending_ = 0;
    uint64_t                    n_layerahead_hits_ = 0;
    // Times spec_pagein_submit refused to submit (or had to shrink a chunk)
    // because WP_EXPERT_SPEC_MAX_SLOTS was already spent. Surfaced on the
    // WP_HINT_LOG counter line so the cap binding is visible.
    uint64_t                    spec_blocked_budget_ = 0;
    // WP_EXPERT_SPEC_MAX_INFLIGHT -- how many speculative page-in BATCHES may be
    // in flight at once, i.e. the size of spec_batches_ the pump gate allows.
    // MEASURED 2026-08-19: with the old hard cap of one, the pump logged
    // pump[.../vbusy/.../vsubmit]=2494007/.../2484065/.../1441 -- 2,484,065
    // pump calls found the single spec_batch_ occupied and could only harvest,
    // against 1,441 real submits, while spec_queue_left sat at 51 the entire
    // run. Every prefetch tuning knob (WP_HINT_ROUTER2_K/CONF/PAGES,
    // WP_HINT_REUSE_PAGES, WP_EXPERT_SPEC_MAX_SLOTS) changes how many hints get
    // QUEUED, but with the queue draining at one SPEC_CHUNK-sized batch per
    // completion, none of those knobs could move measured throughput -- K=3/7/15
    // and conf=0.4/0.75/0.8 all landed at the identical 2.686-2.697 t/s. Parsed
    // exactly like WP_EXPERT_SPEC_MAX_SLOTS: a positive integer, absent/empty/
    // <=0 means 1 -- so the default build is byte-identical in behaviour to
    // today (one batch in flight, submit-drain-submit).
    const long                 spec_max_inflight_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_MAX_INFLIGHT");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 1;
        return v > 0 ? v : 1;
    }();
    // One entry per in-flight speculative BATCH: the Batch itself (holding its
    // slot pins, which is what stops an in-flight read's slot from being
    // recycled underneath it), the pages it is reading, and their leases --
    // parallel to `batch`, same indexing as spec_pagein_submit built `cold`.
    // Capped at spec_max_inflight_ elements; spec_pagein_submit refuses to grow
    // it past that, same shape as the old "one batch at a time" refusal.
    struct SpecBatch {
        std::unique_ptr<Batch>          batch;
        std::vector<const ExpertPage *> inflight;
        std::vector<uint64_t>           leases;
        bool                            layer_ahead = false;
    };
    std::vector<SpecBatch>          spec_batches_;
    // WP_EXPERT_SPEC_HOST_THREADS -- concurrent host-landing reader threads.
    // Default 1 = today's EXACT behaviour: submit one page -> read it -> reap
    // -> submit the next page, one landing thread at a time.
    //
    // MEASURED 2026-08-19: between decode tokens each GPU worker is idle for
    // the draft phase (~38 ms) with ZERO demand reads outstanding -- the drive
    // is completely free. At ~3 ms/page (~9 MB) that window has room for ~12
    // pages per worker per token. We land ~1.7. So even the fully unblocked
    // window runs at ~14% utilisation, purely because delivery is serialized
    // to one page, one thread, at a time (spec_host_submit refused any new
    // work while host_thread_.joinable(), and reaping required host_pending_
    // to hit 0 first). That caps prefetch coverage under 2% of the demand
    // paging, which is why an ~99%-precise predictor moved throughput not at
    // all: the bottleneck was never prediction quality, it was delivery
    // concurrency. Raising this env var lets several landing reads run at
    // once inside that same idle window.
    struct HostLandingThread {
        std::thread thread;
        // Set true by the thread itself, just before it returns. Lets
        // spec_host_reap() join it without ever blocking the dispatch thread.
        std::shared_ptr<std::atomic<bool>> finished;
    };
    std::vector<HostLandingThread>  host_threads_;
    // Pages currently being read by ANY landing thread combined (not thread
    // count -- see spec_host_in_flight() vs spec_host_busy()).
    std::atomic<size_t>             host_pending_{0};
    // *** STAGING-POOL SAFETY BOUND, NOT JUST THE RAW ENV VALUE. ***
    //
    // staging_ is shared with the demand and spec-VRAM read paths: ensure_batch
    // caps THEIR concurrent readers at min(WP_EXPERT_READ_WORKERS, default 4,
    // staging_.buffer_count()), and a demand batch's readers and the one
    // in-flight spec-VRAM batch's readers (spec_pagein_submit) can be
    // outstanding at the same time -- worst case up to 2x that default, i.e. 8
    // buffers of a pool that is "a fixed 16 buffers regardless of --slots" (see
    // the WP_SELF_BENCH=3 comment in the ExpertSlotPool ctor).
    //
    // StagingPool::borrow() is a plain counting semaphore (condition_variable
    // over a free list, see StagingPool::Lease) and no borrower ever holds one
    // lease while waiting on a second, so going over the pool size CANNOT
    // deadlock -- every leased buffer is released, unconditionally, when its
    // Lease goes out of scope, so there is always a draining page ahead of any
    // blocked borrow. What going over the pool size DOES cost is latency: a
    // demand borrow can queue behind however many host-landing borrows are
    // currently holding buffers. To keep that bounded, reserve half of
    // staging_.buffer_count() for demand/spec-VRAM and cap host landings at the
    // other half (floor 1, so the pool is never host-landing-only). At the
    // documented default of 16 buffers that is a cap of 8 -- comfortably above
    // the demand path's own default of 4 concurrent readers, leaving headroom
    // even when a spec-VRAM batch is ALSO reading at the same time.
    const size_t host_thread_cap_ = [this] {
        const char * e = std::getenv("WP_EXPERT_SPEC_HOST_THREADS");
        long requested = (e != nullptr && e[0] != '\0') ? std::strtol(e, nullptr, 10) : 1;
        if (requested < 1) {
            requested = 1;
        }
        const size_t buffers    = (size_t) staging_.buffer_count();
        const size_t max_allowed = std::max<size_t>(1, buffers / 2);
        return std::min<size_t>((size_t) requested, max_allowed);
    }();
    // WP_EXPERT_SPEC_PREEMPT=1 -- host landings become PREEMPTIBLE: the landing
    // thread reads each page in WP_EXPERT_SPEC_SUBREAD-byte slices (default
    // 1 MiB) and pauses between slices whenever a demand request is being
    // served. Bounds the worst-case delay a demand read inherits from an
    // in-flight speculative read to one slice (~0.2 ms at 1 MiB on this
    // hardware) instead of a whole 12.75 MB page (~2-4 ms) -- which is what
    // makes it safe to raise hint volume enough to actually pack the drive's
    // idle 60-70% (kmbandy's framing, 2026-08-07). DEFAULT OFF.
    const bool spec_preempt_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PREEMPT");
        return e != nullptr && e[0] == '1';
    }();
    // WP_EXPERT_SPEC_PREEMPT_BEFORE_BORROW=1 keeps a speculative host landing
    // from holding a shared staging lease while demand I/O is already pending.
    // Default off for an A/B against the original preemption behaviour.
    const bool spec_preempt_before_borrow_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PREEMPT_BEFORE_BORROW");
        return e != nullptr && e[0] == '1';
    }();
    const size_t spec_subread_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_SUBREAD");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 0;
        return v > 0 ? (size_t) v : (size_t) (1u << 20);
    }();
    // WP_EXPERT_SPEC_PREEMPT_DEADLINE=1 -- bound how long a single slice's
    // yield-to-demand wait can run before proceeding anyway. Only consulted
    // when spec_preempt_ is also on. See the big comment at the yield loop in
    // spec_host_submit for the 2026-08-19 measurement that motivated this.
    // DEFAULT OFF: preserves today's unbounded-yield spec_preempt_ behaviour
    // exactly until this is A/B'd on hardware.
    const bool spec_preempt_deadline_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PREEMPT_DEADLINE");
        return e != nullptr && e[0] == '1';
    }();
    // Max microseconds a slice will yield to demand before proceeding anyway,
    // when spec_preempt_deadline_ is on. Suggested/default 2000 us.
    const uint64_t spec_preempt_max_wait_us_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PREEMPT_MAX_WAIT_US");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 0;
        return v > 0 ? (uint64_t) v : (uint64_t) 2000;
    }();
    // True while Worker::dispatch is serving a request. Written by the dispatch
    // thread, read by the host landing thread between slices.
    std::atomic<bool>               demand_serving_{false};
    // Outstanding DEMAND page reads (read_worker pages from a non-speculative
    // batch). The landing thread's pause condition: gating on demand_serving_
    // alone paused landings for a request's ENTIRE service -- including compute
    // phases when the drive is free -- and throttled landing throughput to
    // ~16 ms/page (2026-08-07 pkm* arms: promotes collapsed because pages
    // landed after their layer had passed). The drive is only contended while
    // demand reads are actually outstanding; pause exactly then.
    std::atomic<int>                demand_reads_pending_{0};
    std::atomic<uint64_t>           host_landed_{0};
    std::atomic<uint64_t>           host_bytes_{0};
    std::atomic<uint64_t>           host_errors_{0};
    uint64_t                        host_skip_bad_  = 0;
    uint64_t                        host_skip_pin_  = 0;
    uint64_t                        host_skip_vram_ = 0;
    uint64_t                        host_skip_tier_ = 0;
    // retire_spec_batch -> complete_batch -> ... never re-enters ensure_batch,
    // but the guard makes that explicit and cheap rather than assumed.
    bool                            spec_recursion_ = false;
    const bool                 lfu_ = [] {
        const char * e = std::getenv("WP_EXPERT_LFU");
        return e == nullptr || e[0] != '0';   // ON by default
    }();
    const bool                 lfu_history_enabled_ = [] {
        const char * e = std::getenv("WP_EXPERT_LFU_HISTORY");
        return e != nullptr && e[0] == '1';
    }();
    const uint64_t             lfu_history_halflife_ = [] {
        const char * e = std::getenv("WP_EXPERT_LFU_HALFLIFE");
        const long v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 4096;
        return v > 0 ? (uint64_t) v : (uint64_t) 4096;
    }();
    void note_demand_reference(const ExpertPage & page) {
        if (!lfu_history_enabled_ || page.cache_id < 0) {
            return;
        }
        const size_t index = (size_t) page.cache_id;
        if (index >= lfu_history_.size()) {
            lfu_history_.resize(index + 1, 0);
        }
        if (lfu_history_[index] != std::numeric_limits<uint64_t>::max()) {
            ++lfu_history_[index];
        }
        if (++lfu_history_references_ == lfu_history_halflife_) {
            for (uint64_t & uses : lfu_history_) {
                uses /= 2;
            }
            lfu_history_references_ = 0;
        }
    }
    uint64_t history_uses(const ExpertPage & page) const {
        const size_t index = page.cache_id < 0 ? lfu_history_.size() :
            (size_t) page.cache_id;
        return index < lfu_history_.size() ? lfu_history_[index] : 0;
    }
    uint64_t history_uses(const Slot & slot) const {
        return slot.page == nullptr ? 0 : history_uses(*slot.page);
    }
    // WP_EXPERT_HINT_SHIELD=N keeps pages named by the last N hint frames out
    // of demand eviction. 0 is the default-off control arm.
    const size_t              hint_shield_depth_ = [] {
        const char * e = std::getenv("WP_EXPERT_HINT_SHIELD");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 0;
        return v > 0 ? (size_t) v : (size_t) 0;
    }();
    const bool                 hint_shield_predicted_ = [] {
        const char * e = std::getenv("WP_EXPERT_HINT_SHIELD_PREDICTED");
        return e != nullptr && e[0] == '1';
    }();
    std::deque<std::vector<uint64_t>> hint_shield_history_;
    std::unordered_map<uint64_t, size_t> hint_shield_counts_;
    uint64_t                   n_shield_hits_ = 0;
    uint64_t                   n_shield_exhausted_ = 0;
    std::vector<uint64_t>       lfu_history_;
    uint64_t                    lfu_history_references_ = 0;
    size_t                      n_pinned_ = 0;
    uint64_t                    n_pinned_demand_hits_ = 0;
    // Victim ordering: is `a` a strictly BETTER victim than `b`? With LFU off
    // this is pure LRU, byte for byte what shipped before, so a control arm
    // needs no separate build.
    //
    // A flat three-key comparison, NOT a std::tuple built per call: this runs
    // twice per candidate inside select_victim's scan of up to 2200 slots on
    // every page-in, on the dispatch thread. The keys, most significant first:
    //   1. lease  -- a live lease is the FIRST key, so a leased page loses to
    //      every unleased one and is only taken when nothing else can be. That
    //      is the deadlock guard: the lease reorders candidates, it never
    //      removes them, so select_victim always has something to return.
    //   2. uses   -- the use-count policy (skipped entirely with LFU off).
    //   3. tick   -- LRU recency, always the final tie-break.
    bool rank_less(const Slot & a, const Slot & b) const {
        const bool a_leased = a.lease_until > evictions_;
        const bool b_leased = b.lease_until > evictions_;
        if (a_leased != b_leased) {
            return b_leased;
        }
        if (lfu_ || lfu_history_enabled_) {
            const uint64_t a_uses = lfu_history_enabled_ ? history_uses(a) : a.uses;
            const uint64_t b_uses = lfu_history_enabled_ ? history_uses(b) : b.uses;
            if (a_uses != b_uses) {
                return a_uses < b_uses;
            }
        }
        return a.tick < b.tick;
    }
    // Speculative page-in accounting. spec_pageins_/spec_bytes_ are what
    // speculation SPENT; the request stream's n_pagein is what it SAVED. Both are
    // reported, neither is a verdict -- on this rig the drive is ~78% idle during
    // decode, so extra reads there are spending capacity that would otherwise go
    // unused, and totalling bytes as if bandwidth were scarce answers a question
    // the hardware is not asking.
    uint64_t                   spec_pageins_  = 0;
    uint64_t                   spec_bytes_  = 0;
    uint64_t                   spec_errors_ = 0;
    // The large backing allocations every slot is carved from. Declared BEFORE
    // slots_ so it outlives them: Slot::buffer points in here and does not own.
    std::vector<buffer_ptr>    arenas_;
    std::vector<size_t>        arena_class_starts_;
    std::vector<Slot>          slots_;
    // Filled once in the constructor, after slots_ is built; see the
    // memoisation note on arena_layout(). Immutable thereafter, so it is
    // safe to read from any thread without a lock.
    std::optional<ArenaLayout> arena_layout_;
    // find_slot's O(1) index: slot_key(layer, expert) -> slot index, for every
    // currently-VALID slot. Maintained at every write to Slot::valid/Slot::key,
    // which is exactly three sites: the invalidate in ensure_batch's pagein loop
    // (erase, before `slot.valid = false`), and the two page-landed sites that set
    // both fields together -- the host-tier hit in ensure_batch and the disk-read
    // completion in drain_one_read. select_victim is NOT indexed here: its choice
    // depends on capacity/lease/uses/tick, not key lookup, so a hash index buys it
    // nothing -- it would need a real policy structure (e.g. a clock hand) to stop
    // scanning, which is out of scope for this change.
    std::unordered_map<uint64_t, size_t> slot_index_;
    std::vector<int>           reserve_blocks_;
    std::map<std::string, int> fds_;
};

size_t ExpertSlotPool::pin_pages(const std::vector<const ExpertPage *> & pages) {
    size_t n_pinned = 0;
    for (const ExpertPage * page : pages) {
        if (page == nullptr || page->is_resident) {
            continue;
        }
        size_t slot_index = find_slot(*page);
        if (slot_index == slots_.size()) {
            std::vector<const ExpertPage *> one{page};
            Batch batch = ensure_batch(one, false, {}, 0, -1, nullptr, false);
            batch.complete();
            slot_index = find_slot(*page);
        }
        if (slot_index == slots_.size()) {
            throw std::runtime_error("pinned expert did not land in a slot");
        }
        Slot & slot = slots_[slot_index];
        if (!slot.pinned) {
            slot.pinned = true;
            ++n_pinned_;
            ++n_pinned;
        }
    }
    return n_pinned;
}

ExpertSlotPool::Batch::Batch(Batch && other) noexcept :
    owner_(other.owner_),
    entries_(std::move(other.entries_)),
    state_(std::move(other.state_)),
    workers_(std::move(other.workers_)),
    completed_(other.completed_),
    ns_lookup_(other.ns_lookup_),
    ns_read_(other.ns_read_),
    n_resident_(other.n_resident_),
    n_pagein_(other.n_pagein_),
    n_pagein_reserved_(other.n_pagein_reserved_),
    n_pagein_general_(other.n_pagein_general_),
    bytes_read_(other.bytes_read_),
    n_host_hit_(other.n_host_hit_),
    n_host_demote_(other.n_host_demote_),
    ns_host_get_(other.ns_host_get_),
    ns_demote_(other.ns_demote_),
    ns_ensure_post_(other.ns_ensure_post_),
    host_bytes_(other.host_bytes_),
    ns_h2d_(other.ns_h2d_),
    bytes_h2d_(other.bytes_h2d_),
    n_reader_h2d_(other.n_reader_h2d_),
    n_read_inflight_max_(other.n_read_inflight_max_),
    ns_read_issue_(other.ns_read_issue_),
    ns_read_complete_(other.ns_read_complete_),
    n_cpu_on_arrival_(other.n_cpu_on_arrival_),
    n_cpu_on_arrival_fallback_(other.n_cpu_on_arrival_fallback_),
    copy_event_(other.copy_event_),
    // Drain state travels with the batch. It was omitted here originally --
    // harmless while every move ran before the first drain (NRVO covered the
    // return paths), but spec_pagein_submit now moves a live batch into a
    // unique_ptr, and losing received_/first_error_ there would silently reset
    // a partially drained batch.
    received_(other.received_),
    first_error_(std::move(other.first_error_)),
    first_read_(other.first_read_),
    last_read_(other.last_read_),
    first_read_issue_(other.first_read_issue_),
    last_read_issue_(other.last_read_issue_),
    first_read_complete_(other.first_read_complete_),
    last_read_complete_(other.last_read_complete_),
    have_read_time_(other.have_read_time_) {
    other.owner_ = nullptr;
    other.copy_event_ = nullptr;
}

ExpertSlotPool::Batch::~Batch() {
    if (copy_event_ != nullptr) {
        ggml_backend_event_free(copy_event_);
        copy_event_ = nullptr;
    }
    if (owner_ != nullptr) {
        owner_->abandon_batch(*this);
    }
}

void ExpertSlotPool::Batch::complete() {
    if (owner_ == nullptr) {
        throw std::logic_error("expert batch has no owner");
    }
    owner_->complete_batch(*this);
}

void ExpertSlotPool::Batch::complete_upto(size_t entry_end) {
    if (owner_ == nullptr) {
        throw std::logic_error("expert batch has no owner");
    }
    owner_->complete_batch_upto(*this, entry_end);
}

void attach_weight(
        ggml_tensor * tensor, ggml_backend_buffer_t buffer, void * base,
        uint64_t offset) {
    const size_t buffer_size = ggml_backend_buffer_get_size(buffer);
    const size_t allocation_size = ggml_backend_buft_get_alloc_size(
        ggml_backend_buffer_get_type(buffer), tensor);
    if (offset > buffer_size || allocation_size > buffer_size - (size_t) offset) {
        throw std::runtime_error("expert weight allocation does not fit its buffer");
    }
    tensor->buffer = buffer;
    tensor->data   = (uint8_t *) base + offset;
    // Do not init_tensor / tensor_memset the quantized pad on this tensor:
    // tensor_memset cannot write past ggml_nbytes, and init_tensor's
    // cuda/hipMemset of the tail is "invalid argument" on arena-offset
    // pointers. Padding is zeroed after H2D on slot.raw (I8, full slot).
}

class DeviceWorker {
public:
    ~DeviceWorker();

    struct split_pending {
        pipe_expert_dispatch_req request;
        std::optional<ExpertSlotPool::Batch> batch;
        bool arena_eligible = false;
        uint64_t seq_id = 0;
    };

    struct AsyncDispatchGuard {
        DeviceWorker & worker;
        int previous_conn_index;

        AsyncDispatchGuard(DeviceWorker & worker, int conn_index) :
            worker(worker), previous_conn_index(worker.begin_async_dispatch(conn_index)) {}

        ~AsyncDispatchGuard() {
            worker.end_async_dispatch(previous_conn_index);
        }
    };
    DeviceWorker(
            Catalog catalog,
            const std::string & device,
            int slots,
            uint64_t host_budget_bytes,
            uint64_t host_victim_bytes,
            TestHooks * test_hooks,
            const std::vector<int> & resident_expert_blocks,
            const std::vector<int> & expert_reserve_blocks,
            uint64_t expert_reserve_bytes,
            wp::HostTier * shared_host_tier = nullptr,
            bool load_pin_file = true,
            std::function<bool(int, int)> page_owner = {},
            WorkerLogFiles * logs = nullptr) :
        catalog_(std::move(catalog)),
        page_owner_(std::move(page_owner)),
        logs_(logs),
        backend_(init_backend(device)),
        resident_(backend_.get(),
                  layout_sliced_pages(
                      catalog_, ggml_backend_get_default_buffer_type(backend_.get())),
                  resident_expert_blocks, page_owner_),
        pool_(
            backend_.get(),
            plan_resources_for_backend(
                resource_pages(catalog_, page_owner_), slots, host_budget_bytes,
                resident_.pinned_bytes(), expert_reserve_blocks,
                expert_reserve_bytes, backend_.get()),
            host_victim_bytes,
            test_hooks, expert_reserve_blocks, catalog_.pages.size(),
            shared_host_tier, logs),
        compute_galloc_(ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_.get()))),
        slots_(pool_.resources().slot_count) {
        if (!compute_galloc_) {
            throw std::runtime_error("failed to create expert graph allocator");
        }
        // The pool logs demand and speculative page-ins into the Worker's handle
        // so every event lands in one ordered stream. The coordinator owns the
        // handle because all device workers write to the same log.
        pool_.set_spec_log(logs_ != nullptr ? logs_->hint : nullptr,
                           logs_ != nullptr ? &logs_->mutex : nullptr);
        stats_.set_staging_kind(pool_.staging_kind());
        stats_.set_device(device);
        device_name_ = device;
        for (auto & kv : catalog_.pages) {
            if (page_owner_ && !page_owner_(kv.second.layer, kv.second.expert)) {
                continue;
            }
            layer_pages_sorted_[kv.first.first].push_back(&kv.second);
        }
        for (auto & kv : layer_pages_sorted_) {
            std::sort(kv.second.begin(), kv.second.end(),
                      [](const ExpertPage * a, const ExpertPage * b) {
                          if (a->blob != b->blob) {
                              return a->blob < b->blob;
                          }
                          return a->offset < b->offset;
                      });
        }
        const char * const pin_path = std::getenv("WP_EXPERT_PIN_FILE");
        if (load_pin_file && pin_path != nullptr && pin_path[0] != '\0') {
            size_t pin_budget = pool_.resources().slot_count;
            const char * const max_env = std::getenv("WP_EXPERT_PIN_MAX_SLOTS");
            if (max_env != nullptr && max_env[0] != '\0') {
                const long long parsed = std::strtoll(max_env, nullptr, 10);
                pin_budget = parsed > 0 ? (size_t) parsed : 0;
            }
            if (pin_budget > (size_t) pool_.resources().slot_count) {
                pin_budget = (size_t) pool_.resources().slot_count;
            }
            const size_t pin_class_pct = expert_pin_class_pct_from_env();
            std::vector<size_t> pin_class_caps;
            std::vector<size_t> pin_class_pinned;
            std::vector<size_t> pin_class_skipped;
            for (const SlotClass & slot_class : pool_.resources().slot_classes) {
                pin_class_caps.push_back((size_t) slot_class.slots * pin_class_pct / 100);
                pin_class_pinned.push_back(0);
                pin_class_skipped.push_back(0);
            }
            std::ifstream pin_file(pin_path);
            if (!pin_file) {
                throw std::runtime_error(
                    "failed to open WP_EXPERT_PIN_FILE: " + std::string(pin_path));
            }
            std::vector<const ExpertPage *> pin_pages;
            std::set<std::pair<int, int>> seen_pins;
            std::string line;
            size_t line_number = 0;
            bool truncated = false;
            while (std::getline(pin_file, line)) {
                ++line_number;
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::istringstream input(line);
                int layer = -1;
                int expert = -1;
                if (!(input >> layer >> expert)) {
                    std::cerr << "WARN wp expert worker: ignoring malformed pin line "
                              << line_number << std::endl;
                    continue;
                }
                const std::pair<int, int> key = { layer, expert };
                if (!seen_pins.insert(key).second) {
                    continue;
                }
                const auto it = catalog_.pages.find(key);
                if (it == catalog_.pages.end()) {
                    std::cerr << "WARN wp expert worker: ignoring unknown pin "
                              << layer << " " << expert << std::endl;
                    continue;
                }
                const size_t class_id = expert_pin_class_index(pool_.resources(), it->second.size);
                if (class_id < pin_class_caps.size() &&
                        pin_class_pinned[class_id] >= pin_class_caps[class_id]) {
                    ++pin_class_skipped[class_id];
                    continue;
                }
                if (pin_pages.size() >= pin_budget) {
                    truncated = true;
                    continue;
                }
                pin_pages.push_back(&it->second);
                if (class_id < pin_class_pinned.size()) {
                    ++pin_class_pinned[class_id];
                }
            }
            const size_t loaded = pool_.pin_pages(pin_pages);
            std::cerr << "WARN wp expert worker: pin_file=" << pin_path
                      << " n_pinned=" << loaded
                      << " pin_budget=" << pin_budget
                      << " pin_class_pct=" << pin_class_pct
                      << " demand_hits=0" << std::endl;
            for (size_t i = 0; i < pin_class_caps.size(); ++i) {
                const SlotClass & slot_class = pool_.resources().slot_classes[i];
                std::cerr << "WARN wp expert worker: pin_class bytes=" << slot_class.size
                          << " slots=" << slot_class.slots
                          << " cap=" << pin_class_caps[i]
                          << " pinned=" << pin_class_pinned[i]
                          << " skipped=" << pin_class_skipped[i] << std::endl;
            }
            if (truncated) {
                std::cerr << "WARN wp expert worker: pin file truncated in file order"
                          << " at " << pin_budget << " slots" << std::endl;
            }
        }
        if (prefill_layer_ahead_) {
            std::cerr << "WARN wp expert worker: WP_PREFILL_LAYER_AHEAD=1 layers="
                      << catalog_.layers.size()
                      << " width>" << prefill_layer_ahead_width_
                      << std::endl;
        }
        std::cerr << "WARN wp expert worker: WP_EXPERT_SPEC_DEMAND_FIRST="
                  << (spec_demand_first_ ? 1 : 0)
                  << std::endl;
        const char * const arena_env = std::getenv("WP_EXPERT_ARENA_ID");
        std::cerr << "WARN wp expert worker: WP_EXPERT_ARENA_ID="
                  << (arena_env != nullptr && std::strtol(arena_env, nullptr, 10) == 1 ? 1 : 0)
                  << " arena_ready=" << (pool_.arena_layout().has_value() ? 1 : 0)
                  << std::endl;
        if (const char * e = std::getenv("WP_WORKER_NULL"); e != nullptr && e[0] == '1') {
            std::cerr << "WARN wp expert worker: WP_WORKER_NULL=1 (TIMING PROBE: "
                         "requests answered with zeros, no reads, no compute, "
                         "outputs are garbage)" << std::endl;
        }
        std::cerr << "WARN wp expert worker: pinned_pages="
                  << resident_.pinned_pages()
                  << " pinned_bytes=" << pool_.resources().pinned_bytes
                  << " slot_count=" << pool_.resources().slot_count
                  << " slot_budget_bytes=" << pool_.resources().slot_budget_bytes
                  << std::endl;
        if (pool_.resources().requested_reserved_bytes != 0) {
            std::cerr << "WARN wp expert worker: reserved_bytes="
                      << pool_.resources().reserved_bytes
                      << " reserved_slots=" << pool_.resources().reserved_slot_count
                      << " general_slots=" << pool_.resources().general_slot_count
                      << std::endl;
            if (pool_.resources().named_reservable_bytes < pool_.resources().requested_reserved_bytes) {
                std::cerr << "WARN wp expert worker: reservation clamped to named pageable bytes="
                          << pool_.resources().named_reservable_bytes << std::endl;
            }
        }
        // Runs only under WP_SELF_BENCH=1; no-op otherwise. Placed after the
        // slot pool is built so the backend is in the same state it will serve
        // requests in.
        // PRE-ALLOCATE THE IO BUFFER AT STARTUP. The first device-buffer
        // allocation made AFTER the slot pool exists costs 1394 ms on the RX 480
        // (vs 0.3 ms on the 1070) -- one 1 MiB host-visible allocation behind
        // 5.35 GB of device-local slots. Paid lazily on the first request it
        // amortised to 0.87 ms on EVERY request and was 95% of prepare_io.
        // It is a one-time cost, so pay it here with the rest of startup.
        {
            RequestStats warmup;
            // *** THE 1 MiB FLOOR WAS SIZED FOR DECODE AND RE-OPENS THE BUG IT FIXED. ***
            // prepare_io needs n_embd * n_tokens * sizeof(f32) TWICE (input + result).
            // At n_embd=4096 that is 8 KB per buffer for a decode step (n_tokens=1)
            // but 16.8 MB for a prefill ubatch (n_tokens=512) -- so with a 1 MiB floor
            // ANY real prompt forces the io buffer to grow WHILE SERVING, which is
            // precisely the allocation this pre-allocation exists to avoid.
            //
            // MEASURED 2026-08-03 on the RX 480, 739-token prompt, n_ubatch=512:
            //   wp io-buffer grow #2: 1048576 -> 7307264      (during serving)
            //   wp io-buffer grow #3: 7307264 -> 16777216     (during serving)
            //   submit_us_max 219828 -> 1074753 -> 1290431    <- 1.29 SECOND submits
            //   submit_hist >=8ms: 62 on the 480 vs 14 on the 1070
            // The 480 measured 2.35x the 1070's per-request submit during prefill and
            // only 1.26x during decode -- the gap tracks n_tokens because the GROWTHS
            // track n_tokens. The card is not the problem; this floor is.
            // Yesterday's parity result stands: it was taken at a 6-token prompt, where
            // the buffer never outgrows 1 MiB and no in-serving allocation ever happens.
            //
            // WP_IO_PREALLOC_TOKENS overrides the assumed max ubatch. Default 512 =
            // llama.cpp's default n_ubatch. COST IS TRIVIAL: 16.8 MB at 512, 33.6 MB at
            // 1024, against an 8 GB card -- and it must be raised ALONGSIDE n_ubatch,
            // or the n_ubatch lever re-triggers this exact stall at the larger size.
            uint32_t prealloc_tokens = 512;
            if (const char * env = std::getenv("WP_IO_PREALLOC_TOKENS")) {
                if (env[0] != '\0') {
                    prealloc_tokens = (uint32_t) std::strtoul(env, nullptr, 10);
                }
            }
            size_t want = 1u << 20;
            if (prealloc_tokens > 0) {
                // Mirror prepare_io's layout: a padded input plus an equal result,
                // with slack for buffer-type alignment.
                const size_t one =
                    (size_t) catalog_.descriptor.hparams.n_embd * prealloc_tokens * sizeof(float);
                want = std::max(want, 2 * (one + 65536));
            }
            fprintf(stderr,
                    "wp io-buffer prealloc: %zu bytes for n_tokens<=%u (n_embd=%u)\n",
                    want, prealloc_tokens, (unsigned) catalog_.descriptor.hparams.n_embd);
            grow_io_buffer(want, warmup);
        }
        alloc_io_small();
        // Pinned staging is INDEPENDENT of io-small: io-small fixes the
        // DESTINATION (Vulkan BAR), this fixes the SOURCE, and a worker can
        // want one without the other.
        if (const char * e = std::getenv("WP_IO_SRC_PINNED")) {
            if (e[0] == '1') {
                unsigned long tokens = 8;
                if (const char * t = std::getenv("WP_IO_SRC_TOKENS")) {
                    if (t[0] != '\0') { tokens = std::strtoul(t, nullptr, 10); }
                }
                const size_t one = (size_t) catalog_.descriptor.hparams.n_embd *
                                   (size_t) tokens * sizeof(float);
                alloc_io_src_pinned(std::max<size_t>(1u << 20, one + 65536));
            }
        }
        if (const char * e = std::getenv("WP_IO_SET_ASYNC")) {
            const bool want_async = e[0] == '1';
            // Name-gated: only HIP/CUDA implement set_tensor_async. See the
            // block comment in prepare_io for why the fallback is worse.
            const bool have_async =
                device_name_.rfind("ROCm", 0) == 0 ||
                device_name_.rfind("CUDA", 0) == 0;
            io_set_async_ = want_async && have_async && io_src_base_ != nullptr;
            if (want_async) {
                fprintf(stderr,
                        "wp io-set-async: %s on %s (pinned=%s, backend=%s)\n",
                        io_set_async_ ? "ENABLED" : "declined",
                        device_name_.c_str(),
                        io_src_base_ != nullptr ? "yes" : "NO",
                        have_async ? "has async" : "NO async iface");
            }
        }
        stats_.set_probe_backend(backend_.get());
        run_self_bench(backend_.get(),
                       catalog_.descriptor.hparams.n_embd,
                       catalog_.descriptor.hparams.n_ff_exp);
        build_keepalive();
    }

    // WP_KEEPALIVE_US=N (0 = off): while waiting for the next request, submit a
    // trivial graph every N microseconds instead of leaving the GPU idle.
    //
    // WHY. On the RX 480 the cost of a submit depends on how long the GPU idled
    // beforehand. Measured 2026-08-02 on this exact expert graph, clocks already
    // pinned to max via power_dpm_force_performance_level=high:
    //     idle gap      200us    1ms     3ms
    //     idle          284      512     547   us/expert
    //     keepalive     163      163     165   us/expert
    // The keepalive removes the penalty entirely and is FLAT in gap length. It
    // is not a clock effect -- sclk and mclk are both pinned at maximum while
    // this happens -- so it is gating below the clock level, and occupying the
    // GPU is the only lever available without a kernel parameter.
    //
    // Submitted from THIS thread, between poll() timeouts on the request socket.
    // Do not move it to a background thread: Vulkan command pools have thread
    // affinity, so concurrent submits risk corruption even behind a mutex.
    void keepalive_tick() {
        if (keepalive_graph_ != nullptr) {
            ggml_backend_graph_compute(backend_.get(), keepalive_graph_);
        }
    }

    bool keepalive_enabled() const { return keepalive_us_ > 0 && keepalive_graph_ != nullptr; }
    int  keepalive_us()      const { return keepalive_us_; }

    // Record a prefetch hint. READS NOTHING YET -- see the frame-loop
    // comment. Experts outside this worker's shard are counted separately rather
    // than rejected: the spine routes by the same static hash the dispatch uses,
    // so a nonzero foreign count means the two disagree, which is a spine bug
    // that would otherwise show up only as prefetch mysteriously not helping.
    void note_prefetch_hint(const pipe_expert_prefetch_hint & hint) {
        ++hint_frames_;
        hint_experts_ += hint.expert_ids.size();
        log_hint_ids(hint);
        if (!std::binary_search(catalog_.layers.begin(), catalog_.layers.end(), hint.layer)) {
            pool_.note_hint_frame(hint.provenance, {});
            hint_foreign_layer_ += hint.expert_ids.size();
            log_prefetch_hints();
            return;
        }
        std::vector<std::pair<int32_t, int32_t>> shield_pages;
        shield_pages.reserve(hint.expert_ids.size());
        size_t predicted_this_frame = 0;
        // Whole-slice layer-ahead frames are larger than the decode generation
        // cap (16). Keep them off spec_queue_ (WP_SPEC_QUEUE_MAX=64 would drop
        // most of a 256-expert layer). The catalog path fetches the slice.
        const bool whole_slice_ahead =
            prefill_layer_ahead_ &&
            hint.provenance == PIPE_HINT_CERTAIN &&
            hint.expert_ids.size() > 16;
        for (int32_t expert_id : hint.expert_ids) {
            if (expert_id < catalog_.descriptor.expert_first ||
                expert_id > catalog_.descriptor.expert_last ||
                catalog_.pages.count({ hint.layer, expert_id }) == 0) {
                ++hint_foreign_expert_;
                continue;
            }
            shield_pages.emplace_back(hint.layer, expert_id);
            if (whole_slice_ahead) {
                continue;
            }
            // Resolved here, on the frame thread, so the idle path does no map
            // lookups between reads. Ascending on the wire, so this queue is in
            // ascending page order per layer -- which is the order that lets the
            // drive read something closer to a stream than a random walk.
            if (spec_enabled_) {
                const ExpertPage * page = &catalog_.pages.at({ hint.layer, expert_id });
                // pool_.host_landing_available() is the load-bearing half of
                // this condition. Without it a predicted hint goes onto a queue
                // that has nowhere to drain to -- spec_host_submit returns 0 with
                // no host tier -- and the prediction is silently discarded rather
                // than fetched. That is worse than not predicting: the arm looks
                // like it ran and measured nothing.
                if (hint.provenance == PIPE_HINT_PREDICTED && spec_host_enabled_ &&
                    pool_.host_landing_available()) {
                    // A GUESS DOES NOT GET A VRAM SLOT. It lands in host RAM,
                    // where a wrong guess costs only the bandwidth that fetched
                    // it and a right one is promoted over PCIe instead of being
                    // re-read from NVMe.
                    if (spec_predict_topm_ != 0 && predicted_this_frame >= spec_predict_topm_) {
                        ++spec_dropped_;
                        continue;
                    }
                    ++predicted_this_frame;
                    enqueue_newest(host_queue_, page);
                } else {
                    // CERTAIN (or host-less) -> VRAM spec_queue_. Newest-wins:
                    // when full, drop the OLDEST page (already late) so the
                    // incoming hint -- the one with remaining lead -- stays.
                    //
                    // MAD-LAB 2026-08-21: the host branch's per-frame predicted
                    // cap never applied on this fallthrough, so an n-gram flood
                    // (76k frames/run) drowned the certain hash-layer hints in
                    // this shared queue and resident fraction FELL (vspec arm,
                    // §8.13 follow-up). Two rules restore the hierarchy:
                    // predicted admissions respect the per-frame cap, and a
                    // guess can never evict a certain hint -- eviction takes
                    // the oldest PREDICTED entry first, and a predicted arrival
                    // is refused outright when the queue is full of certain
                    // work. Discriminated by lease value (predicted lease
                    // defaults to 4 vs 64; equal leases degrade to old
                    // behaviour, never worse).
                    const bool predicted = hint.provenance == PIPE_HINT_PREDICTED;
                    if (predicted && spec_predict_topm_ != 0 &&
                        predicted_this_frame >= spec_predict_topm_) {
                        ++spec_dropped_;
                        continue;
                    }
                    if (predicted) {
                        ++predicted_this_frame;
                    }
                    const uint64_t lease =
                        predicted ? pool_.spec_lease_predicted() : pool_.spec_lease();
                    if (spec_queue_max_ != 0 && spec_queue_.size() >= spec_queue_max_) {
                        const uint64_t pred_lease = pool_.spec_lease_predicted();
                        const uint64_t cert_lease = pool_.spec_lease();
                        auto victim = spec_queue_.end();
                        if (pred_lease != cert_lease) {
                            victim = std::find_if(
                                spec_queue_.begin(), spec_queue_.end(),
                                [pred_lease](const std::pair<const ExpertPage *, uint64_t> & e) {
                                    return e.second == pred_lease;
                                });
                        }
                        if (victim != spec_queue_.end()) {
                            spec_queue_.erase(victim);
                            ++spec_dropped_;
                        } else if (predicted && pred_lease != cert_lease) {
                            ++spec_dropped_;
                            continue;
                        } else {
                            spec_queue_.pop_front();
                            ++spec_dropped_;
                        }
                    }
                    spec_queue_.emplace_back(page, lease);
                }
            }
        }
        pool_.note_hint_frame(hint.provenance, shield_pages);
        log_prefetch_hints();
    }

    void note_prefetch_hint_bad() {
        ++hint_bad_;
        log_prefetch_hints();
    }

    int32_t next_served_layer(int32_t layer) const {
        const auto it = std::upper_bound(catalog_.layers.begin(), catalog_.layers.end(), layer);
        return it == catalog_.layers.end() ? -1 : *it;
    }

    void note_expert_recency(int32_t expert_id) {
        if (expert_id < 0) {
            return;
        }
        if ((size_t) expert_id >= expert_recency_.size()) {
            expert_recency_.resize((size_t) expert_id + 1, 0);
        }
        expert_recency_[(size_t) expert_id] = ++recency_tick_;
    }

    uint64_t expert_recency_of(int32_t expert_id) const {
        if (expert_id < 0 || (size_t) expert_id >= expert_recency_.size()) {
            return 0;
        }
        return expert_recency_[(size_t) expert_id];
    }

    // Prefill L+1 whole-slice stream. Demand batch for L is already issued
    // (and pinned). Catalog-driven: the worker already owns its slice, so this
    // does not wait on spine hint frames. spec_pagein_submit is async and
    // filters residents; PREFILL_GATE / SPEC_CHUNK / SPEC_QUEUE_MAX are not
    // consulted -- this is not a guess. Demand-first: skip while demand reads
    // are still outstanding and retry from later call sites in dispatch().
    void submit_prefill_layer_ahead(int32_t layer, uint32_t n_tokens) {
        if (!prefill_layer_ahead_ || n_tokens <= prefill_layer_ahead_width_) {
            return;
        }
        const int32_t nxt = next_served_layer(layer);
        if (nxt < 0 || nxt == ahead_target_) {
            return;
        }
        const auto it = layer_pages_sorted_.find(nxt);
        if (it == layer_pages_sorted_.end() || it->second.empty()) {
            return;
        }
        if (pool_.demand_reads_outstanding()) {
            ++pump_vram_demand_defer_;
            return;
        }
        (void) pool_.spec_pagein_poll(false);

        std::vector<const ExpertPage *> pages = it->second;
        const size_t budget = pool_.unpinned_slots();
        if (budget == 0) {
            return;
        }
        if (pages.size() > budget) {
            std::nth_element(
                pages.begin(), pages.begin() + (ptrdiff_t) budget, pages.end(),
                [this](const ExpertPage * a, const ExpertPage * b) {
                    const uint64_t ra = expert_recency_of(a->expert);
                    const uint64_t rb = expert_recency_of(b->expert);
                    if (ra != rb) {
                        return ra > rb;
                    }
                    if (a->blob != b->blob) {
                        return a->blob < b->blob;
                    }
                    return a->offset < b->offset;
                });
            pages.resize(budget);
            std::sort(pages.begin(), pages.end(),
                      [](const ExpertPage * a, const ExpertPage * b) {
                          if (a->blob != b->blob) {
                              return a->blob < b->blob;
                          }
                          return a->offset < b->offset;
                      });
        }
        if (ahead_offered_layer_ != nxt) {
            n_layerahead_hints_ += it->second.size();
            ahead_offered_layer_ = nxt;
        }
        std::vector<uint64_t> leases(pages.size(), pool_.spec_lease());
        const size_t n = pool_.spec_pagein_submit(pages, leases, /*layer_ahead=*/ true);
        if (n > 0) {
            ahead_target_ = nxt;
            n_layerahead_pageins_ += n;
            ++ahead_submits_;
            ahead_pages_ += n;
        } else if (pool_.spec_inflight_live() < pool_.spec_inflight_cap() + 1) {
            // All filtered (resident or already in flight): nothing left to
            // fetch for this target. Still at-cap returns 0 without this mark
            // so a later retry can submit.
            ahead_target_ = nxt;
        }
    }

    // Wrapper so the layer-ahead hint shows up as its own banner field. It is
    // called from inside the `phase` lap that becomes ns_pagein_compute, i.e.
    // from inside the Vulkan ns_vk_dispatch_path envelope, so an expensive
    // hint would otherwise be indistinguishable from compute.
    void time_layer_ahead(
            const pipe_expert_dispatch_req & request, RequestStats & request_stats) {
        if (!stats_.enabled()) {
            submit_prefill_layer_ahead(request.layer, request.n_tokens);
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        submit_prefill_layer_ahead(request.layer, request.n_tokens);
        const uint64_t elapsed =
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
        if (is_vulkan_backend()) {
            request_stats.ns_vk_layer_ahead += elapsed;
        }
    }



    // Drop the queue. Called when the connection has been idle long enough that
    // the hinted layer is certainly behind us -- speculating on a layer already
    // computed is a read with no possible upside.
    void drop_spec_work() {
        spec_dropped_ += spec_queue_.size() + host_queue_.size();
        spec_queue_.clear();
        host_queue_.clear();
    }

    // After a dispatch RESPONSE: keep the speculative pipeline moving so a read
    // overlaps the following layer.
    //
    // THIS MUST HARVEST, AND THE SUBMIT-ONLY VERSION WAS A DEADLOCK. The step
    // short-circuits on `spec_in_flight()`, and the ONLY place that clears it is
    // a harvest. The idle pump was supposed to do that, but await_request()
    // returns the moment a frame is ready, and during continuous decode a frame
    // is essentially always ready -- so the pump body never ran. Net effect:
    // exactly ONE speculative read per connection, never reaped, every later
    // submit dead. Measured 2026-08-19: spec_pageins=+1 against spec_dropped
    // =+22066 over 128 tokens.
    //
    // The harvest is non-blocking (spec_pagein_poll(false) takes only what has
    // already landed), so the recv thread pays an H2D copy solely on the step
    // where a read actually completed -- which is the step that also frees the
    // pipeline to issue the next one. That cost is why this was written
    // submit-only; a permanently wedged pipeline is the higher price.
    void spec_pagein_after_dispatch() {
        if (spec_submit_interleave_enabled_ && spec_enabled_) {
            (void) spec_pagein_step(/*harvest=*/ true);
        }
    }

    // Keep the speculative pipeline moving. Called from the idle pump.
    //
    // TWO DISTINCT JOBS, and neither of them blocks:
    //   1. harvest a batch that already landed, which frees its slot pins;
    //   2. submit the next chunk if nothing is in flight.
    // Between calls the read runs on the reader threads, so it proceeds WHILE
    // the dispatch thread polls or computes. That overlap is the entire reason
    // this path exists; the previous version completed the read inline here and
    // therefore never overlapped anything.
    //
    // H2D still happens on this thread inside drain_one_read -- Vulkan command
    // pools have thread affinity, the same constraint keepalive_tick documents.
    // Only the DISK read moved off it, which is the part worth moving.
    //
    // Returns true if it did any work worth staying awake for.
    // *** PUMP CENSUS (WP_HINT_LOG). ***
    // spec_pagein_step runs on every dispatch response -- ~5500 times per 128
    // tokens -- and submits a handful. Which EXIT it takes is the difference
    // between "the queue is empty", "the reader is busy", and "everything we
    // predicted is already resident", and those need completely different fixes.
    // Counting them costs an increment on a path that is already doing I/O.
    bool spec_pagein_step(bool harvest = true) {
        ++pump_calls_;
        // Host landings run on their own reader thread and never touch the GPU,
        // so they are reaped and refilled independently of the VRAM path.
        pool_.spec_host_reap();
        if (spec_prefill_gate_active_) {
            ++pump_gated_;
            // Prefill gate: harvest what is in flight, submit nothing new.
            // spec_any_in_flight, not spec_in_flight -- gated harvest must drain
            // every live batch, not only fire once the pool is at the
            // WP_EXPERT_SPEC_MAX_INFLIGHT cap.
            if (harvest && pool_.spec_any_in_flight()) {
                return pool_.spec_pagein_poll(false);
            }
            return false;
        }
        // Capacity, not activity: with WP_EXPERT_SPEC_HOST_THREADS > 1 several
        // landing threads can be busy at once while there is STILL room for
        // another submission -- spec_host_busy() is the "may I submit" gate,
        // spec_host_in_flight() (below, in has_spec_work) is "is there work
        // still outstanding". At the default of 1 thread the two agree, so
        // this is byte-identical to today's submit-one-at-a-time behaviour.
        if (pool_.spec_host_busy()) {
            ++pump_host_busy_;
        } else if (host_queue_.empty()) {
            ++pump_host_empty_;
        }
        if (!pool_.spec_host_busy() && !host_queue_.empty()) {
            if (spec_demand_first_ && pool_.demand_reads_outstanding()) {
                ++pump_vram_demand_defer_;
                return harvest ? pool_.spec_pagein_poll(false) : false;
            }
            const size_t take = std::min(spec_chunk_, host_queue_.size());
            std::vector<const ExpertPage *> chunk(
                host_queue_.begin(), host_queue_.begin() + (ptrdiff_t) take);
            host_queue_.erase(host_queue_.begin(), host_queue_.begin() + (ptrdiff_t) take);
            if (pool_.spec_host_submit(chunk) != 0) {
                ++pump_host_submit_;
                return true;
            }
            // The chunk was consumed but nothing was read: every page in it was
            // filtered (resident, pinned, or already in the tier). This is the
            // exit that says the PREDICTIONS are wrong-but-harmless rather than
            // the pipeline being busy.
            ++pump_host_filtered_;
        }
        // spec_in_flight() is the AT-CAPACITY gate (spec_batches_.size() >=
        // WP_EXPERT_SPEC_MAX_INFLIGHT), deliberately -- with more than one
        // in-flight batch allowed, "busy" must mean "no room for another
        // submission", not "something is outstanding" (that's
        // spec_any_in_flight, used above for the gated-harvest branch and
        // below in has_spec_work). At the default cap of 1 the two conditions
        // coincide and this is byte-identical to today's pump gate --
        // pump_vram_busy_ still increments exactly once per pump call that
        // finds the pool full, whether "full" means one batch or N.
        if (pool_.spec_in_flight()) {
            ++pump_vram_busy_;
            return harvest ? pool_.spec_pagein_poll(false) : false;
        }
        if (spec_demand_first_ && pool_.demand_reads_outstanding()) {
            ++pump_vram_demand_defer_;
            return harvest ? pool_.spec_pagein_poll(false) : false;
        }
        if (spec_queue_.empty()) {
            ++pump_vram_empty_;
            return false;
        }
        ++pump_vram_submit_;
        const size_t take = std::min(spec_chunk_, spec_queue_.size());
        std::vector<const ExpertPage *> chunk;
        std::vector<uint64_t>           leases;
        chunk.reserve(take);
        leases.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            chunk.push_back(spec_queue_[i].first);
            leases.push_back(spec_queue_[i].second);
        }
        spec_queue_.erase(spec_queue_.begin(), spec_queue_.begin() + (ptrdiff_t) take);
        return pool_.spec_pagein_submit(chunk, leases) != 0;
    }

    // A speculative read in flight is work in progress, not idleness -- the pump
    // must keep spinning to harvest it even when the queue is empty.
    bool has_spec_work() const {
        return !spec_queue_.empty() || !host_queue_.empty() ||
               pool_.spec_any_in_flight() || pool_.spec_host_in_flight();
    }

    // Something to SUBMIT right now, as opposed to something in flight. Only the
    // former justifies a zero-timeout poll: when a read is already running on a
    // reader thread we are waiting on the disk, not on ourselves, and spinning
    // would burn a core for the 3-5 ms of the read while doing nothing.
    bool has_spec_submit_work() const {
        if (spec_prefill_gate_active_) {
            return false;   // gated: nothing may be submitted, so do not spin
        }
        // !pool_.spec_in_flight() here means "not at the WP_EXPERT_SPEC_MAX_
        // INFLIGHT cap", i.e. there is room to submit -- the same capacity
        // sense spec_pagein_step's pump gate uses, not spec_any_in_flight.
        return (!spec_queue_.empty() && !pool_.spec_in_flight()) ||
               (!host_queue_.empty() && !pool_.spec_host_busy());
    }

    // The counter line, built in ONE place so stderr and WP_HINT_LOG cannot
    // drift apart. spec_pageins/spec_bytes sit next to the request stream's own
    // n_pagein and bytes_read because spend and saving are only interpretable
    // together.
    std::string prefetch_hint_line() const {
        char buf[1280];
        std::snprintf(buf, sizeof(buf),
                      "frames=%llu experts=%llu "
                      "foreign_layer=%llu foreign_expert=%llu malformed=%llu "
                      "spec_pageins=%llu spec_bytes=%llu spec_errors=%llu "
                      "spec_dropped=%llu spec_queue_left=%zu host_queue_left=%zu "
                      "host_landed=%llu host_bytes=%llu host_errors=%llu "
                      "host_promoted=%llu host_wasted=%llu "
                      "host_skip[bad/pin/vram/tier]=%llu/%llu/%llu/%llu "
                      "demand_prefetch_late[queued/inflight]=%llu/%llu "
                      "spec_cap[pending/blocked_budget]=%zu/%llu "
                      "pin[n/hits]=%zu/%llu "
                      "pump[calls/gated/hbusy/hempty/hsubmit/hfiltered/vbusy/vempty/vsubmit/vdemand_defer]="
                      "%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu "
                      "spec_inflight[live/cap]=%zu/%zu "
                      "n_layerahead[hints/pageins/hits]=%llu/%llu/%llu",
                      (unsigned long long) hint_frames_,
                      (unsigned long long) hint_experts_,
                      (unsigned long long) hint_foreign_layer_,
                      (unsigned long long) hint_foreign_expert_,
                      (unsigned long long) hint_bad_,
                      (unsigned long long) pool_.spec_pageins(),
                      (unsigned long long) pool_.spec_bytes(),
                      (unsigned long long) pool_.spec_errors(),
                      (unsigned long long) spec_dropped_,
                      spec_queue_.size(),
                      host_queue_.size(),
                      (unsigned long long) pool_.host_landed(),
                      (unsigned long long) pool_.host_spec_bytes(),
                      (unsigned long long) pool_.host_spec_errors(),
                      (unsigned long long) pool_.host_spec_promotions(),
                      (unsigned long long) pool_.host_spec_wasted(),
                      (unsigned long long) pool_.host_skip_bad(),
                      (unsigned long long) pool_.host_skip_pin(),
                      (unsigned long long) pool_.host_skip_vram(),
                      (unsigned long long) pool_.host_skip_tier(),
                      (unsigned long long) n_demand_prefetch_queued_,
                      (unsigned long long) n_demand_prefetch_inflight_,
                      pool_.n_spec_pending(),
                      (unsigned long long) pool_.spec_blocked_budget(),
                      pool_.n_pinned(),
                      (unsigned long long) pool_.n_pinned_demand_hits(),
                      (unsigned long long) pump_calls_,
                      (unsigned long long) pump_gated_,
                      (unsigned long long) pump_host_busy_,
                      (unsigned long long) pump_host_empty_,
                      (unsigned long long) pump_host_submit_,
                      (unsigned long long) pump_host_filtered_,
                      (unsigned long long) pump_vram_busy_,
                      (unsigned long long) pump_vram_empty_,
                      (unsigned long long) pump_vram_submit_,
                      (unsigned long long) pump_vram_demand_defer_,
                      pool_.spec_inflight_live(),
                      pool_.spec_inflight_cap(),
                      (unsigned long long) n_layerahead_hints_,
                      (unsigned long long) n_layerahead_pageins_,
                      (unsigned long long) pool_.n_layerahead_hits());
        return buf;
    }

    // R -- GROUND TRUTH: the experts a dispatch actually asked for. Without this
    // in the same stream, a speculative page-in that was never selected cannot be
    // told apart from one that was selected but arrived too late, and the whole
    // log answers neither question. Duplicates WP_REF_LOG on purpose -- see
    // set_spec_log for why a second file will not do.
    void log_reference(int32_t layer, const std::vector<pipe_expert_assignment> & assignments) {
        if (logs_ == nullptr || logs_->hint == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(logs_->mutex);
        std::fprintf(logs_->hint, "R %d", layer);
        for (const pipe_expert_assignment & a : assignments) {
            std::fprintf(logs_->hint, " %d", a.expert_id);
        }
        std::fputc('\n', logs_->hint);
        std::fflush(logs_->hint);
    }

    void report_prefetch_hints() const {
        if (hint_frames_ == 0 && hint_bad_ == 0 && ahead_submits_ == 0) {
            return;
        }
        if (hint_frames_ > 0 || hint_bad_ > 0) {
            std::fprintf(stderr, "wp-expert-worker prefetch hints: %s\n",
                         prefetch_hint_line().c_str());
        }
        if (ahead_submits_ > 0 || n_layerahead_hints_ > 0) {
            std::fprintf(stderr,
                         "wp-expert-worker prefill layer-ahead: submits=%llu pages=%llu "
                         "n_layerahead_hints=%llu n_layerahead_pageins=%llu n_layerahead_hits=%llu\n",
                         (unsigned long long) ahead_submits_,
                         (unsigned long long) ahead_pages_,
                         (unsigned long long) n_layerahead_hints_,
                         (unsigned long long) n_layerahead_pageins_,
                         (unsigned long long) pool_.n_layerahead_hits());
        }
    }

    pipe_expert_hello hello() const {
        pipe_expert_hello hello;
        hello.role          = PIPE_EXPERT_ROLE_WORKER;
        hello.hidden_type   = PIPE_HIDDEN_F16;
        hello.n_embd        = catalog_.descriptor.hparams.n_embd;
        hello.n_ff_exp      = catalog_.descriptor.hparams.n_ff_exp;
        hello.n_expert      = catalog_.descriptor.hparams.n_expert;
        hello.n_expert_used = catalog_.descriptor.hparams.n_expert_used;
        hello.expert_first  = catalog_.descriptor.expert_first;
        hello.expert_last   = catalog_.descriptor.expert_last;
        hello.n_slots       = (uint32_t) slots_;
        hello.layers        = catalog_.layers;
        hello.model_identity = source_model_identity(
            catalog_.descriptor.input_model, catalog_.descriptor.model_files);
        hello.shard_identity =
            (catalog_.descriptor.sliced ? "slice:" : "") +
            catalog_.descriptor.identity_algorithm + ":" +
            catalog_.descriptor.identity_value;
        return hello;
    }

    void validate_dispatch(const pipe_expert_dispatch_req & request) const {
        if (!std::binary_search(catalog_.layers.begin(), catalog_.layers.end(), request.layer)) {
            throw pipe_protocol_error(
                PIPE_ERR_EXPERT_LAYER,
                "worker does not serve layer " + std::to_string(request.layer));
        }
        if (request.assignments.size() >
            (size_t) catalog_.descriptor.hparams.n_expert) {
            throw pipe_protocol_error(
                PIPE_ERR_BAD_FRAME,
                "expert dispatch has more assignments than model experts");
        }
        std::set<int32_t> seen_experts;
        for (const pipe_expert_assignment & assignment : request.assignments) {
            if (!seen_experts.insert(assignment.expert_id).second) {
                throw pipe_protocol_error(
                    PIPE_ERR_BAD_FRAME,
                    "expert dispatch repeats expert " + std::to_string(assignment.expert_id));
            }
            if (assignment.expert_id < catalog_.descriptor.expert_first ||
                assignment.expert_id > catalog_.descriptor.expert_last ||
                catalog_.pages.count({ request.layer, assignment.expert_id }) == 0) {
                throw pipe_protocol_error(
                    PIPE_ERR_EXPERT_RANGE,
                    "worker does not serve expert " + std::to_string(assignment.expert_id));
            }
        }
    }

    bool grouped_gemv_eligible(const pipe_expert_dispatch_req & request) const {
        static const bool enabled = [] {
            const char * e = std::getenv("WP_EXPERT_GROUPED_GEMV");
            return e != nullptr && std::strtol(e, nullptr, 10) == 1;
        }();
        const char * const backend_name = ggml_backend_name(backend_.get());
        const bool backend_supported = backend_name != nullptr &&
            (std::strstr(backend_name, "ROCm") != nullptr ||
             std::strstr(backend_name, "Vulkan") != nullptr);
        return enabled && backend_name != nullptr &&
            backend_supported &&
            request.n_tokens >= 1 && request.n_tokens <= 8 &&
            request.assignments.size() >= 1 &&
            request.assignments.size() <= (size_t) 16 * request.n_tokens;
    }

    bool arena_id_eligible(
            const pipe_expert_dispatch_req & request,
            const ExpertSlotPool::Batch & batch) const {
        static const bool enabled = [] {
            const char * e = std::getenv("WP_EXPERT_ARENA_ID");
            return e != nullptr && std::strtol(e, nullptr, 10) == 1;
        }();
        // *** ORDER MATTERS: the env gate is first and it short-circuits. ***
        // This used to evaluate ggml_backend_name(), the three strstr()s and
        // pool_.arena_layout() BEFORE testing `enabled`, so a worker running
        // WITHOUT WP_EXPERT_ARENA_ID still paid the whole probe -- including
        // arena_layout()'s full slot scan -- two to three times per request on
        // the decode critical path. Every condition below is side-effect free,
        // so hoisting the cheapest, most selective one is behaviour-identical.
        if (!enabled) {
            return false;
        }
        if (request.n_tokens < 1 || request.n_tokens > 8 ||
                request.assignments.empty() ||
                request.assignments.size() > (size_t) 16 * request.n_tokens) {
            return false;
        }
        const char * const backend_name = ggml_backend_name(backend_.get());
        if (wp_persistent_graphs_enabled() && backend_name != nullptr &&
                std::strstr(backend_name, "Vulkan") != nullptr) {
            return false;
        }
        // Vulkan uses its MM mul_mat_id path for the strided arena view.
        const bool backend_supported = backend_name != nullptr &&
            (std::strstr(backend_name, "ROCm") != nullptr ||
             std::strstr(backend_name, "CUDA") != nullptr ||
             std::strstr(backend_name, "Vulkan") != nullptr);
        const std::optional<ExpertSlotPool::ArenaLayout> & layout_opt = pool_.arena_layout();
        if (!backend_supported || !layout_opt.has_value()) {
            return false;
        }
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            const std::vector<float> & weights = request.assignments[i].weights;
            if (weights.size() != request.n_tokens ||
                    std::any_of(weights.begin(), weights.end(), [](float weight) {
                        return !std::isfinite(weight);
                    })) {
                return false;
            }
            if (batch.slot_index(i) == std::numeric_limits<size_t>::max()) {
                return false;
            }
        }
        const ExpertSlotPool::ArenaLayout & layout = *layout_opt;
        static const char * k_roles[3] = {"gate", "up", "down"};
        const auto & specs = catalog_.descriptor.layers.at(request.layer);
        auto stride_ok = [](uint64_t stride, ggml_type type) {
            const uint64_t type_size = ggml_type_size(type);
            const uint64_t block_size = ggml_blck_size(type);
            return type_size != 0 && block_size != 0 && stride % type_size == 0 &&
                stride / type_size <= UINT32_MAX / block_size;
        };
        const ExpertPage & first = catalog_.pages.at({
            request.layer, request.assignments[0].expert_id
        });
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            const ExpertPage & page = catalog_.pages.at({
                request.layer, request.assignments[i].expert_id
            });
            if (batch.slot_index(i) >= layout.n_slots) {
                return false;
            }
            const ExpertSlotPool::ArenaLayout::Arena * arena =
                layout.arena_for_slot(batch.slot_index(i));
            if (arena == nullptr || arena->stride == 0 ||
                    page.device_size > arena->stride) {
                return false;
            }
            for (const char * role : k_roles) {
                if (!stride_ok(arena->stride, specs.at(role).type)) {
                    return false;
                }
            }
            for (const char * role : k_roles) {
                if (page.roles.at(role).device_offset !=
                        first.roles.at(role).device_offset) {
                    return false;
                }
            }
        }
        return true;
    }

    pipe_expert_partial dispatch(
            const pipe_expert_dispatch_req & request,
            RequestStats & request_stats,
            std::optional<ExpertSlotPool::Batch> prepared = std::nullopt,
            int conn_index = -1) {
        // RAII so every exit path counts -- dispatch() returns from several
        // places and throws from more.
        struct DispatchTotalScope {
            bool on;
            RequestStats & st;
            std::chrono::steady_clock::time_point started;
            ~DispatchTotalScope() {
                if (!on) return;
                st.ns_dispatch_total +=
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count();
            }
        } dispatch_total_scope{
            stats_enabled(), request_stats,
            stats_enabled() ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point() };
        const bool owns_gate = !prepared.has_value();
        if (owns_gate) {
            spec_prefill_gate_active_ = spec_prefill_gate_enabled_ &&
                                        request.n_tokens > spec_prefill_gate_width_;
        }
        // Raise the demand gate for the whole request; preemptible landings
        // pause between slices while it is up. RAII so every exit (including
        // the protocol throws below) lowers it.
        if (owns_gate) {
            pool_.demand_serving(true);
        }
        struct DemandGate {
            ExpertSlotPool & pool;
            bool active;
            ~DemandGate() { if (active) pool.demand_serving(false); }
        } demand_gate{ pool_, owns_gate };
        validate_dispatch(request);

        // WP_WORKER_NULL=1 -- TIMING PROBE ONLY (hop-theory probe B/C,
        // 2026-08-22): decode-path requests answered with zeros, no reads,
        // no compute. Real wire + protocol, zero work -- isolates the
        // per-request protocol floor from worker-internal cost. Outputs are
        // garbage; never pair with a quality gate.
        static const bool worker_null = [] {
            const char * e = std::getenv("WP_WORKER_NULL");
            return e != nullptr && e[0] == '1';
        }();
        // Decode-path requests only: a prepared batch (split dispatch) has
        // pinned slots and in-flight reads that must be consumed, and prefill
        // rides that path -- abandoning it mid-flight killed the worker on the
        // first prefill of the 10:07 run. Prefill/split go through the real
        // path; the probe only needs decode timing.
        if (worker_null && !prepared.has_value() && request.n_tokens <= 8) {
            pipe_expert_partial out;
            out.layer    = request.layer;
            out.n_tokens = request.n_tokens;
            out.partial.assign(
                (size_t) request.n_tokens * catalog_.descriptor.hparams.n_embd, 0.0f);
            return out;
        }

        // Already f32 on the wire as of PIPE_VERSION 4; this used to widen a
        // f16 value back to f32, which recovered the storage type but NOT the
        // ~3e-4 of precision the spine had already thrown away.
        const std::vector<float> & activation = request.activations;
        const bool measure = stats_.enabled();
        const std::chrono::steady_clock::time_point lookup_started =
            measure ? std::chrono::steady_clock::now() :
                      std::chrono::steady_clock::time_point();
        std::vector<const ExpertPage *> pages;
        pages.reserve(request.assignments.size());
        for (const pipe_expert_assignment & assignment : request.assignments) {
            note_expert_recency(assignment.expert_id);
            pages.push_back(&catalog_.pages.at({
                request.layer, assignment.expert_id
            }));
        }
        if (!prepared.has_value()) {
            note_demand_prefetch_lateness(pages);
        }
        ExpertSlotPool::Batch batch = prepared.has_value()
            ? std::move(*prepared)
            : pool_.ensure_batch(pages, measure, lookup_started, request.n_tokens, conn_index);
        // Declare this after batch: its destructor synchronizes before batch
        // releases pins or permits slot reuse on an exceptional exit.
        AsyncDispatchGuard async_dispatch_guard(*this, conn_index);
        if (measure) {
            request_stats.ns_lookup  = batch.lookup_ns();
            request_stats.n_resident      = batch.n_resident();
            request_stats.n_pagein     = batch.n_pagein();
            request_stats.n_pagein_reserved = batch.n_pagein_reserved();
            request_stats.n_pagein_general = batch.n_pagein_general();
            request_stats.n_host_hit = batch.n_host_hit();
            request_stats.n_host_demote = batch.n_host_demote();
            request_stats.bytes_read = batch.bytes_read();
            request_stats.ns_host_get = batch.ns_host_get();
            request_stats.ns_demote = batch.ns_demote();
            request_stats.ns_ensure_post = batch.ns_ensure_post();
            request_stats.host_bytes = batch.host_bytes();
        }
        const bool cpu_on_arrival_request = batch.n_cpu_on_arrival() != 0;

        // WP_PREFILL_LAYER_AHEAD: after THIS layer's demand pins, try the NEXT
        // layer's catalog as one spec-VRAM batch. Demand-first defers while
        // this layer's reads are still outstanding; retries after complete_upto
        // / complete so the read overlaps remaining compute. Bypasses the
        // 64-deep hint queue, PREFILL_GATE, and SPEC_CHUNK.
        submit_prefill_layer_ahead(request.layer, request.n_tokens);
        // WP_EXPERT_DOUBLE_BUFFER: see double_buffer_reads_'s comment for the
        // full argument. Short version -- THIS request's own pages are already
        // pinned and (if cold) reading by this point, so it is now safe to let
        // the NEXT layer's already-hinted pages start reading too: submit-only,
        // reads land underneath whatever this request's compute does next.
        if (double_buffer_reads_ && spec_enabled_) {
            (void) spec_pagein_step(/*harvest=*/ false);
        }

        const std::chrono::steady_clock::time_point compute_started =
            measure ? std::chrono::steady_clock::now() :
                      std::chrono::steady_clock::time_point();
        const size_t result_size =
            (size_t) request.n_tokens * catalog_.descriptor.hparams.n_embd;
        std::vector<float> sum(result_size, 0.0f);
        bool have_hits = false;
        bool have_pageins = false;
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            have_hits |= batch.is_resident(i);
            have_pageins |= !batch.is_resident(i);
        }
        // MEASURED 2026-08-29: everything from compute_started to the `phase`
        // origin below was outside every lap(), so it landed in ns_compute and
        // in NO named field. arena_id_eligible() is the suspect -- it used to
        // materialise a full ArenaLayout (a scan of all ~6700 Slot records)
        // BEFORE checking whether WP_EXPERT_ARENA_ID is even set. ns_arena_probe
        // names that cost directly so the next round can see it go to zero.
        const std::chrono::steady_clock::time_point probe_started =
            measure ? std::chrono::steady_clock::now() :
                      std::chrono::steady_clock::time_point();
        const bool persistent_graphs = wp_persistent_graphs_enabled();
        const bool arena_request = persistent_graphs &&
            !cpu_on_arrival_request && arena_id_eligible(request, batch);
        const bool grouped_gemv_request =
            !cpu_on_arrival_request &&
            (grouped_gemv_eligible(request) ||
             (persistent_graphs ? arena_request : arena_id_eligible(request, batch)));
        if (measure) {
            const uint64_t probe_ns =
                (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - probe_started).count();
            request_stats.ns_arena_probe += probe_ns;
            if (is_vulkan_backend()) {
                request_stats.ns_vk_arena_probe += probe_ns;
            }
        }
        // PHASE TIMERS. ns_compute is the wall span of this whole section, but
        // ns_read/h2d/submit/readback only summed to 78% of it on the RX 480
        // (8.95 s of 11.53 s) -- 1.34 ms per request attributed to nothing.
        // These close the accounting: prepare_io (activation upload + io buffer
        // growth), the blocking wait in batch.complete() (thread join + I/O),
        // and the fp32->fp16 encode of the reply.
        auto phase = std::chrono::steady_clock::now();
        if (measure) {
            request_stats.ns_prologue +=
                (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                    phase - compute_started).count();
        }
        auto lap = [&measure, &phase]() -> uint64_t {
            if (!measure) return 0;
            const auto now = std::chrono::steady_clock::now();
            const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - phase).count();
            phase = now;
            return ns;
        };
        // *** DETERMINISM: ONE PASS, INDEX ORDER, AFTER complete(). ***
        // This used to compute the RESIDENT experts first (overlapping the
        // page-in I/O), then the newly-paged-in ones, chaining the two partial
        // sums. That made the floating-point ASSOCIATION depend on which pages
        // happened to be resident when the graph was built:
        //     all resident   : total = h1 + h2 + h3 + h4
        //     expert 2 a miss: total = (h1 + h3 + h4) + m2
        // Same experts, same weights, different rounding -- and membership turns
        // on whether the PREVIOUS request's page-ins had landed yet, i.e. on I/O
        // timing. FP addition is not associative, so the worker returned a
        // slightly different sum run to run. f16 KV absorbed it; turbo4 amplified
        // a last-bit delta into a 4-bit centroid flip and thence a router top-k
        // change, which is how it surfaced (2026-08-03: 3 of 3 pairs divergent
        // with the CPU DSpark worker, 0 of 12 without it).
        // Measured cost of dropping the overlap: see WP_EXPERT_OVERLAP below.
        static const bool overlap = [] {
            const char * e = std::getenv("WP_EXPERT_OVERLAP");
            return e != nullptr && e[0] == '1';   // default OFF = deterministic
        }();
        const bool effective_overlap = overlap &&
            !(persistent_graphs && arena_request);
        // WP_EXPERT_COMPUTE_CHUNKS=<n>: split the expert compute into n fixed
        // index chunks so all but the last can run while the tail of the page-in
        // reads is still in flight. 1 = the original strictly-serial path.
        // Unlike WP_EXPERT_OVERLAP this does NOT trade determinism -- see the
        // note at the compute loop below.
        // NOTE: this is clamped to the assignment count AND gated on n_pagein > 0
        // at the loop. Clamping alone was NOT enough and the earlier comment here
        // claiming "decode is unaffected" was wrong -- min(4, 2) is 2, so 2-expert
        // decode/verify requests were splitting into two submits for nothing.
        static const size_t s_compute_chunks = [] {
            const char * e = std::getenv("WP_EXPERT_COMPUTE_CHUNKS");
            if (e == nullptr || e[0] == '\0') {
                return (size_t) 4;
            }
            const long parsed = std::strtol(e, nullptr, 10);
            return parsed > 0 ? (size_t) parsed : (size_t) 1;
        }();
        // *** WP_EXPERT_RESIDENT_FIRST: overlap reads with compute WITHOUT
        // giving up determinism (default OFF -- opt-in until A/B'd on real
        // hardware, 2026-08-19). ***
        //
        // WP_EXPERT_COMPUTE_CHUNKS above already overlaps read and compute,
        // but only by waiting for INDEX-ORDERED chunks to land -- if expert 0
        // is a miss and expert 1 is a hit, chunk 0 (which covers both) still
        // blocks on expert 0's read before computing EITHER. WP_EXPERT_OVERLAP
        // computes hits immediately instead, which hides the read fully, but
        // it accumulates hits and pageins into the SAME device sum in two
        // GROUPS (all hits, then all misses) -- a different FP association
        // than the canonical index-order fold, hence off by default (see the
        // determinism note above).
        //
        // THE SEPARATION THIS PATH MAKES: "compute resident experts while
        // reads are in flight" and "accumulate in arrival order" are
        // independent choices. Determinism only cares about the SECOND one.
        // So: compute each hit into its OWN buffer slot (compute_batch's new
        // result_offset param) the moment we know it's resident -- before any
        // read has to finish -- then, after complete(), compute each miss
        // into its own slot the same way. Nothing is summed until every slot
        // is filled; fold_resident_first_partials() then does ONE sequential
        // ggml_add pass over the slots in ASSIGNMENT-INDEX order, exactly the
        // association the fully-serial path would produce. Bit-identical
        // output, read time hidden under compute time.
        //
        // MEMORY: n_embd * n_tokens floats per assignment (~128 KiB at
        // n_embd=4096, n_tokens<=8), times tens of experts -- a few MB. That
        // is why this is gated on n_tokens: at PREFILL widths (n_tokens up to
        // 2048) the same per-assignment slot is hundreds of MB, which is NOT
        // an acceptable amount of device memory to hold live for one request.
        // WP_EXPERT_RESIDENT_FIRST_MAX_TOKENS defaults to 32, the same
        // decode/spec-window threshold used elsewhere in this worker (see
        // spec_prefill_gate_width_) -- wide (prefill) requests always fall
        // through to the chunked-serial path below, unchanged.
        static const bool s_resident_first = [] {
            const char * e = std::getenv("WP_EXPERT_RESIDENT_FIRST");
            return e != nullptr && e[0] == '1';   // default OFF
        }();
        static const uint32_t s_resident_first_max_tokens = [] {
            const char * e = std::getenv("WP_EXPERT_RESIDENT_FIRST_MAX_TOKENS");
            if (e == nullptr || e[0] == '\0') {
                return (uint32_t) 32;
            }
            const long parsed = std::strtol(e, nullptr, 10);
            return parsed > 0 ? (uint32_t) parsed : (uint32_t) 32;
        }();
        // Gate on there being a read to hide under (have_pageins) same as the
        // chunked path above: if everything is already resident there is
        // nothing to overlap and the plain serial path is strictly better
        // (no extra per-expert graph submits, no fold pass).
        const bool resident_first_eligible =
            !cpu_on_arrival_request && !grouped_gemv_request && s_resident_first &&
            !request.assignments.empty() &&
            request.n_tokens <= s_resident_first_max_tokens &&
            have_pageins;
        size_t resident_first_base_offset = 0;
        size_t resident_first_slot_size   = 0;
        size_t cpu_on_arrival_base_offset = 0;
        size_t cpu_on_arrival_slot_size   = 0;
        if (resident_first_eligible) {
            // Pre-size io_buffer_ for the canonical input+result slot PLUS
            // one partial-result slot per assignment BEFORE prepare_io runs.
            // grow_io_buffer() reallocates a FRESH device buffer on growth
            // (see its comment) rather than resizing in place, so growing
            // AFTER prepare_io uploads the activation would silently drop it.
            const IoSlotLayout layout = compute_io_slot_layout(request.n_tokens);
            resident_first_slot_size = GGML_PAD(layout.result_size, layout.alignment);
            resident_first_base_offset =
                GGML_PAD(layout.result_offset + layout.result_size, layout.alignment);
            const size_t total = resident_first_base_offset +
                resident_first_slot_size * request.assignments.size();
            io_reserved_hint_ = total;   // prepare_io must not pick the small buffer under this
            grow_io_buffer(total, request_stats);
        }
        if (cpu_on_arrival_request) {
            const IoSlotLayout layout = compute_io_slot_layout(request.n_tokens);
            cpu_on_arrival_slot_size = GGML_PAD(layout.result_size, layout.alignment);
            cpu_on_arrival_base_offset =
                GGML_PAD(layout.result_offset + layout.result_size, layout.alignment);
            const size_t total = cpu_on_arrival_base_offset +
                cpu_on_arrival_slot_size * request.assignments.size();
            io_reserved_hint_ = total;   // prepare_io must not pick the small buffer under this
            grow_io_buffer(total, request_stats);
        }
        const bool measure_vk = measure && is_vulkan_backend();
        std::chrono::steady_clock::time_point vk_dispatch_started;
        if (!request.assignments.empty()) {
            prepare_io(activation, request.n_tokens, request_stats);
            request_stats.ns_prep = lap();
            if (measure_vk) {
                vk_dispatch_started = std::chrono::steady_clock::now();
            }
            if (resident_first_eligible) {
                // Compute every already-resident expert NOW, each into its
                // own slot -- this is what overlaps with the reader threads
                // still pulling the missing pages. See the fold below for
                // why each expert gets its own slot instead of one shared sum.
                for (size_t i = 0; i < request.assignments.size(); ++i) {
                    if (batch.is_resident(i)) {
                        compute_batch(
                            request, pages, batch, /* hits = */ true,
                            /* add_previous = */ false, request_stats,
                            /* all_experts = */ true, /* force_dense = */ false,
                            i, i + 1,
                            resident_first_base_offset + i * resident_first_slot_size);
                    }
                }
            } else if (effective_overlap && have_hits && !cpu_on_arrival_request) {
                compute_batch(
                    request, pages, batch, /* hits = */ true,
                    /* add_previous = */ false, request_stats);
            }
            request_stats.ns_hits = lap();
        }
        // *** READ / COMPUTE OVERLAP (WP_EXPERT_COMPUTE_CHUNKS, default 4). ***
        //
        // THE COST THIS ATTACKS. complete() blocks until EVERY expert page has
        // landed and only then runs one compute pass, so read and compute are
        // strictly serial. Measured 2026-08-05 on the RX 480 at 659-token
        // prefill: read 198.3 ms/request against compute 43.8 ms/request. An
        // expert only needs ITS OWN page, so all but the last chunk's compute
        // can hide under the reads still in flight.
        //
        // WHY CHUNK BY INDEX AND NOT BY ARRIVAL. Computing whatever happens to
        // be ready is what WP_EXPERT_OVERLAP does, and it is off by default
        // precisely because it makes the FP summation order depend on I/O
        // timing -- same experts, different association, a last-bit delta that
        // turbo4 amplified into a different token. Fixed index chunks keep the
        // accumulation order identical to the serial path no matter when reads
        // land, so this is deterministic BY CONSTRUCTION rather than by luck.
        //
        // Chunk count is a tuning knob, not a correctness one: the summation
        // order is the same at every value, so arms differ only in speed. =1
        // restores the exact serial path.
        if (!effective_overlap && !resident_first_eligible && !cpu_on_arrival_request &&
                !request.assignments.empty()) {
            const size_t n_assign = request.assignments.size();
            // *** GATE ON THERE BEING READS TO HIDE UNDER. ***
            // Clamping to the assignment count is NOT enough: min(4, 2) == 2, so a
            // 2-expert request still split into TWO graph submits. Decode averages
            // ~1.5 experts and verify ~2.0, and at 85% residency those requests
            // usually have NOTHING in flight to overlap -- so the split bought a
            // second submit (~0.45 ms on the RX 480, plus its idle-recovery gap
            // penalty) for zero benefit, on the decode critical path. verify is
            // 40.8 s of the 74.0 s decode dispatch wait, so this is not a rounding
            // error. n_pagein == 0 means every expert is already resident and the
            // serial path is strictly better.
            const size_t chunks   = grouped_gemv_request || batch.n_pagein() == 0
                ? 1
                : std::max<size_t>(1, std::min(s_compute_chunks, n_assign));
            for (size_t c = 0; c < chunks; ++c) {
                const size_t beg = n_assign * c / chunks;
                const size_t end = n_assign * (c + 1) / chunks;
                if (beg == end) {
                    continue;
                }
                batch.complete_upto(end);
                request_stats.ns_wait += lap();
                time_layer_ahead(request, request_stats);
                compute_batch(
                    request, pages, batch, /* hits = */ true,
                    /* add_previous = */ c > 0, request_stats,
                    /* all_experts = */ true, /* force_dense = */ false,
                    beg, end);
                request_stats.ns_pagein_compute += lap();
            }
        }
        // Always: joins the reader threads, finalises ns_read and rethrows the
        // first read error. Cheap and already drained after the loop above.
        batch.complete();
        request_stats.ns_wait += lap();
        time_layer_ahead(request, request_stats);
        if (measure) {
            request_stats.ns_read = batch.read_ns();
            request_stats.n_read_inflight_max = batch.n_read_inflight_max();
            request_stats.ns_read_issue = batch.ns_read_issue();
            request_stats.ns_read_complete = batch.ns_read_complete();
            request_stats.ns_h2d    = batch.ns_h2d();
            request_stats.bytes_h2d = batch.bytes_h2d();
            request_stats.n_reader_h2d = batch.n_reader_h2d();
            request_stats.n_cpu_on_arrival = batch.n_cpu_on_arrival();
            request_stats.n_cpu_on_arrival_fallback = batch.n_cpu_on_arrival_fallback();
        }
        if (cpu_on_arrival_request) {
            std::vector<std::vector<float>> cpu_partials(request.assignments.size());
            for (size_t i = 0; i < request.assignments.size(); ++i) {
                const size_t result_offset = cpu_on_arrival_base_offset +
                    i * cpu_on_arrival_slot_size;
                if (batch.is_cpu_on_arrival(i)) {
                    compute_cpu_on_arrival(
                        request, *pages[i], batch, i, cpu_partials[i], request_stats);
                } else {
                    compute_batch(
                        request, pages, batch, /* hits = */ true,
                        /* add_previous = */ false, request_stats,
                        /* all_experts = */ true, /* force_dense = */ false,
                        i, i + 1, result_offset);
                }
            }
            synchronize_async(&request_stats);
            for (size_t i = 0; i < cpu_partials.size(); ++i) {
                if (!cpu_partials[i].empty()) {
                    write_io_partial(
                        cpu_partials[i],
                        cpu_on_arrival_base_offset + i * cpu_on_arrival_slot_size);
                }
            }
            fold_resident_first_partials(
                request.n_tokens, request.assignments.size(),
                cpu_on_arrival_base_offset, cpu_on_arrival_slot_size, request_stats);
        } else if (resident_first_eligible) {
            // Reads are drained (batch.complete() above): compute every
            // formerly-missing expert now, each into its own slot, same as
            // the hits loop before prepare_io returned.
            for (size_t i = 0; i < request.assignments.size(); ++i) {
                if (!batch.is_resident(i)) {
                    compute_batch(
                        request, pages, batch, /* hits = */ false,
                        /* add_previous = */ false, request_stats,
                        /* all_experts = */ true, /* force_dense = */ false,
                        i, i + 1,
                        resident_first_base_offset + i * resident_first_slot_size);
                }
            }
            // *** THE FOLD: CANONICAL ASSIGNMENT-INDEX ORDER. *** Every slot
            // (hit or miss, computed above at whatever time it became ready)
            // now holds exactly what compute_batch would have contributed for
            // that assignment on the plain serial path. Summing them
            // left-to-right by index -- not by which finished first -- is
            // what makes this bit-identical to WP_EXPERT_RESIDENT_FIRST=0.
            fold_resident_first_partials(
                request.n_tokens, request.assignments.size(),
                resident_first_base_offset, resident_first_slot_size, request_stats);
        } else if (effective_overlap && have_pageins) {
            compute_batch(
                request, pages, batch, /* hits = */ false,
                /* add_previous = */ have_hits, request_stats);
        }
        request_stats.ns_pagein_compute += lap();
        if (measure_vk && !request.assignments.empty()) {
            request_stats.ns_vk_dispatch_path +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - vk_dispatch_started).count();
        }
        if (!request.assignments.empty()) {
            read_result(sum, request_stats);
        }
        request_stats.ns_result = lap();

        // *** WP_SELFCHECK=1: EQUIVALENCE PROBE (default OFF, diagnostic only). ***
        // THE INVARIANT UNDER TEST: an expert's contribution to a token must not
        // depend on how many OTHER tokens shared its batch. gather changes the graph
        // shape with batch width (get_rows -> matmul at compacted width ->
        // get_rows_back) while dense does not, so if the two disagree the expert path
        // is batch-width-dependent and THAT is the foundational bug.
        // WHY THIS MATTERS (2026-08-04): conf_min changes ONLY n_draft, i.e. the verify
        // batch width -- yet it changed the generated text at temperature 0. Under
        // correct speculative decoding the target's logits cannot depend on the draft,
        // so something downstream of batch width is not width-invariant. This probe
        // answers that directly instead of inferring it from throughput.
        // PRECEDENT: this exact code already had one width-dependent defect -- the
        // routing-weight orientation, "correct only at n_tokens == 1, so decode looked
        // fine while PREFILL was corrupted and poisoned the KV cache."
        static const bool s_selfcheck = [] {
            const char * e = std::getenv("WP_SELFCHECK");
            return e != nullptr && e[0] == '1';
        }();
        if (s_selfcheck && !cpu_on_arrival_request &&
                !request.assignments.empty() && !sum.empty()) {
            std::vector<float> dense(sum.size(), 0.0f);
            compute_batch(
                request, pages, batch, /* hits = */ true,
                /* add_previous = */ false, request_stats,
                /* all_experts = */ true, /* force_dense = */ true);
            read_result(dense, request_stats);
            double max_abs = 0.0, max_rel = 0.0, sum_abs = 0.0;
            size_t worst = 0;
            for (size_t i = 0; i < sum.size(); ++i) {
                const double a = (double) sum[i], b = (double) dense[i];
                const double d = std::fabs(a - b);
                sum_abs += d;
                if (d > max_abs) { max_abs = d; worst = i; }
                const double den = std::max(std::fabs(a), std::fabs(b));
                if (den > 1e-6) { max_rel = std::max(max_rel, d / den); }
            }
            std::fprintf(stderr,
                "WP_SELFCHECK layer=%d n_tokens=%u n_exp=%zu "
                "max_abs=%.6g max_rel=%.6g mean_abs=%.6g worst_i=%zu "
                "gather=%.6g dense=%.6g %s\n",
                request.layer, request.n_tokens, request.assignments.size(),
                max_abs, max_rel, sum_abs / (double) sum.size(), worst,
                (double) sum[worst], (double) dense[worst],
                max_rel > 1e-3 ? "*** MISMATCH ***" : "ok");
            // NOTE: `sum` already holds the GATHER result (read above) and is what the
            // caller ships to the spine. The dense recompute overwrote only the io
            // buffer's result region, which is not read again. Do NOT re-read into
            // `sum` here -- that would ship the dense result and silently change what
            // the probe is supposed to be observing.
        }
        if (measure) {
            request_stats.ns_compute =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - compute_started).count();
        }

        pipe_expert_partial response;
        response.layer    = request.layer;
        response.n_tokens = request.n_tokens;
        // f32 by default: this subtotal is only PART of the layer's expert sum -- the
        // spine adds the other workers' subtotals to it. Rounding a partial sum to f16
        // put a ~5e-4 relative error at the expert->worker partition boundary, so the
        // layer output depended on which worker happened to get which expert. See
        // f32 UNCONDITIONALLY since 2026-08-19 -- the f16 opt-in was removed, see
        // the tombstone above wp_expert_partial_dtype()'s former definition. The
        // tag is still stamped (not omitted) so the frame stays self-describing
        // for a spine decoding partials from mixed-vintage workers.
        response.dtype = PIPE_HIDDEN_F32;
        response.partial.assign(sum.begin(), sum.end());
        request_stats.ns_encode = lap();
        return response;
    }

    // conn_index identifies which connection's split-dispatch transaction
    // this is (see split_pending_by_conn_'s comment above the member decl).
    // Single-connection default path always passes -1.
    //
    // WP_WORKER_MULTI_CONN correctness fix (2026-08-24, ported from the
    // main-box canonical copy): this USED TO be a single
    // std::optional<split_pending> plus a bare split_seq_id_, i.e.
    // Worker-wide "the one in-flight BEGIN/ACTS split-dispatch transaction"
    // -- correct only under the single-connection assumption. BEGIN and its
    // matching ACTS/ACTS_PUBLISH/ACTS_REF frame are two SEPARATE loop
    // iterations in serve_connection, each taking g_worker_gpu_mutex
    // independently -- the lock is released between them. With two live
    // connections, connection B's BEGIN can land in that gap between
    // connection A's BEGIN and A's ACTS, and since this state was keyed by
    // nothing, B's BEGIN either threw "BEGIN arrived before ACTS" or
    // silently overwrote A's pending transaction outright -- A's
    // subsequent ACTS then failed has_split_dispatch()/seq_id-mismatch and
    // got a PIPE_ERROR frame instead of its expected ACK/PARTIAL. This is
    // the exact mechanism behind "sent an unexpected frame in place of the
    // dedup publish ack" / "died while publishing dedup activations" seen
    // on this box's 8803 leg under real two-dispatcher load.
    //
    // gpu_lock: forwarded to pool_.ensure_batch() below, unchanged -- see
    // that function's parameter comment and its "READ-ISSUE UNLOCKED" block
    // for what it does. Only serve_connection's PIPE_EXPERT_DISPATCH_BEGIN
    // branch passes non-null; every other caller (there are none today
    // besides that branch) gets the default nullptr and this function's
    // locking footprint is unchanged.
    void begin_split_dispatch(const pipe_expert_dispatch_begin & begin, uint64_t seq_id,
                              int conn_index = -1,
                              std::unique_lock<std::mutex> * gpu_lock = nullptr) {
        if (split_pending_by_conn_.count(conn_index) != 0) {
            throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                      "expert dispatch BEGIN arrived before ACTS");
        }
        pipe_expert_dispatch_req request;
        request.layer = begin.layer;
        request.n_tokens = begin.n_tokens;
        request.assignments = begin.assignments;
        request.swiglu_clamp = begin.swiglu_clamp;
        validate_dispatch(request);
        std::vector<const ExpertPage *> pages;
        pages.reserve(request.assignments.size());
        for (const pipe_expert_assignment & assignment : request.assignments) {
            note_expert_recency(assignment.expert_id);
            pages.push_back(&catalog_.pages.at({ request.layer, assignment.expert_id }));
        }
        note_demand_prefetch_lateness(pages);
        // spec_prefill_gate_active_ and pool_.demand_serving() are still
        // Worker/pool-wide (not keyed by conn_index) -- known imprecision
        // under multi-conn (one connection's finish/abandon can clear a gate
        // meant for another's still-in-flight transaction), tracked
        // separately. NOT the bug fixed here: spec_prefill_gate_active_'s
        // only readers are spec_pagein_step()/has_spec_submit_work(), both
        // reachable only from await_request()'s keepalive/spec pump, which
        // is unconditionally skipped whenever g_worker_gpu_mutex != nullptr
        // (see await_request's comment) -- so in multi-conn mode this value
        // is written but never read, hence harmless there today.
        spec_prefill_gate_active_ = spec_prefill_gate_enabled_ &&
                                        request.n_tokens > spec_prefill_gate_width_;
        pool_.demand_serving(true);
        try {
            split_pending pending;
            pending.request = std::move(request);
            pending.seq_id = seq_id;
            const auto lookup_started = stats_.enabled() ? std::chrono::steady_clock::now() :
                std::chrono::steady_clock::time_point{};
            pending.batch.emplace(pool_.ensure_batch(pages, stats_.enabled(), lookup_started,
                                                     pending.request.n_tokens, conn_index,
                                                     gpu_lock));
            pending.arena_eligible = arena_id_eligible(pending.request, *pending.batch);
            submit_prefill_layer_ahead(pending.request.layer, pending.request.n_tokens);
            split_pending_by_conn_.emplace(conn_index, std::move(pending));
        } catch (...) {
            pool_.demand_serving(false);
            spec_prefill_gate_active_ = false;
            throw;
        }
    }

    pipe_expert_partial finish_split_dispatch(
            const pipe_expert_dispatch_acts & acts, uint64_t seq_id,
            RequestStats & request_stats, int conn_index = -1) {
        const auto it = split_pending_by_conn_.find(conn_index);
        if (it == split_pending_by_conn_.end()) {
            throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                      "expert dispatch ACTS has no BEGIN");
        }
        if (seq_id != it->second.seq_id) {
            throw pipe_protocol_error(PIPE_ERR_STALE_SEQ,
                                      "expert dispatch ACTS sequence does not match BEGIN");
        }
        split_pending pending = std::move(it->second);
        split_pending_by_conn_.erase(it);
        pending.request.activations = acts.activations;
        try {
            pipe_expert_partial response = dispatch(
                pending.request, request_stats, std::move(pending.batch), conn_index);
            pool_.demand_serving(false);
            spec_prefill_gate_active_ = false;
            return response;
        } catch (...) {
            pool_.demand_serving(false);
            spec_prefill_gate_active_ = false;
            throw;
        }
    }

    void abandon_split_dispatch(int conn_index = -1) noexcept {
        split_pending_by_conn_.erase(conn_index);
        pool_.demand_serving(false);
        spec_prefill_gate_active_ = false;
    }

    bool has_split_dispatch(int conn_index = -1) const {
        return split_pending_by_conn_.count(conn_index) != 0;
    }
    uint32_t split_n_tokens(int conn_index = -1) const {
        return split_pending_by_conn_.at(conn_index).request.n_tokens;
    }
    bool split_arena_eligible(int conn_index = -1) const {
        return split_pending_by_conn_.at(conn_index).arena_eligible;
    }

    const ResourcePlan & resources() const {
        return pool_.resources();
    }

    size_t read_inflight() const { return pool_.read_inflight(); }
    size_t read_chunk_bytes() const { return pool_.read_chunk_bytes(); }
    bool read_direct() const { return pool_.read_direct(); }
    bool read_direct_fallback() const { return pool_.read_direct_fallback(); }

    // See ExpertSlotPool::set_staging_multi_conn / StagingPool::set_multi_conn
    // (the 2026-08-25 deadlock fix). run() calls this once, right after
    // parsing WP_WORKER_MULTI_CONN and before spawning any connection thread.
    void set_staging_multi_conn(int n) {
        pool_.set_staging_multi_conn(n);
    }

    int pinned_pages() const {
        return resident_.pinned_pages();
    }

    size_t pin_pages(const std::vector<std::pair<int, int>> & keys, size_t budget) {
        std::vector<const ExpertPage *> pages;
        pages.reserve(std::min(keys.size(), budget));
        for (const std::pair<int, int> key : keys) {
            const auto it = catalog_.pages.find(key);
            if (it != catalog_.pages.end()) {
                pages.push_back(&it->second);
            }
            if (pages.size() == budget) {
                break;
            }
        }
        return pool_.pin_pages(pages);
    }

    bool stats_enabled() const {
        return stats_.enabled();
    }

    bool is_vulkan_backend() const {
        const char * name = ggml_backend_name(backend_.get());
        return name != nullptr && std::strstr(name, "Vulkan") != nullptr;
    }

    ggml_backend_graph_plan_t create_persistent_plan(ggml_cgraph * graph) {
        if (!wp_persistent_graphs_enabled() || !is_vulkan_backend()) {
            return nullptr;
        }
        ggml_backend_graph_plan_t plan = nullptr;
        try {
            synchronize_async(nullptr);
            ggml_backend_synchronize(backend_.get());
            plan = ggml_backend_graph_plan_create(backend_.get(), graph);
        } catch (...) {
            plan = nullptr;
        }
        if (plan == nullptr) {
            std::fprintf(stderr,
                         "WARN wp expert worker: persistent Vulkan graph plan unavailable; "
                         "using graph compute\n");
        }
        return plan;
    }

    void release_persistent_plan(ggml_backend_graph_plan_t & plan,
                                 RequestStats * request_stats = nullptr) {
        if (plan == nullptr) {
            return;
        }
        synchronize_async(request_stats);
        ggml_backend_graph_plan_free(backend_.get(), plan);
        plan = nullptr;
    }

    void record_stats(const RequestStats & request, size_t n_experts) {
        stats_.set_shield_stats(pool_.n_shield_hits(), pool_.n_shield_exhausted());
        stats_.set_pin_stats(pool_.n_pinned(), pool_.n_pinned_demand_hits());
        stats_.set_layerahead_stats(
            n_layerahead_hints_, n_layerahead_pageins_, pool_.n_layerahead_hits());
        stats_.record(request, n_experts);
    }

    void record_wire_stats(const RequestStats & request) {
        stats_.record_wire(request);
    }

private:
    // Count demand pages for which a hint has not reached a usable slot yet.
    // The queue check includes the separate host landing queue.
    void note_demand_prefetch_lateness(const std::vector<const ExpertPage *> & pages) {
        if (!spec_enabled_) {
            return;
        }
        for (const ExpertPage * page : pages) {
            const bool queued = std::any_of(
                spec_queue_.begin(), spec_queue_.end(),
                [page](const std::pair<const ExpertPage *, uint64_t> & entry) {
                    return entry.first->layer == page->layer && entry.first->expert == page->expert;
                }) || std::any_of(
                host_queue_.begin(), host_queue_.end(),
                [page](const ExpertPage * entry) {
                    return entry->layer == page->layer && entry->expert == page->expert;
                });
            if (queued) {
                ++n_demand_prefetch_queued_;
            }
            if (pool_.spec_in_flight_for(*page)) {
                ++n_demand_prefetch_inflight_;
            }
        }
    }

    struct AsyncSubmitState {
        // Keep one host vector per upload: several submits can be in flight.
        std::vector<std::vector<uint8_t>> params;
        std::vector<std::vector<int32_t>> ids;
        std::vector<std::vector<float>> route_weights;
        bool pending = false;
        bool graph_pending = false;
    };

    // WP_SUBMIT_ASYNC is opt-in. The tensor upload and graph compute use the
    // same backend stream, so the upload is ordered before its consumer. This
    // avoids the MAD-114 gfx1201 cross-stream visibility hazard.
    const bool submit_async_ = [] {
        const char * e = std::getenv("WP_SUBMIT_ASYNC");
        return e != nullptr && e[0] == '1';
    }();

    int begin_async_dispatch(int conn_index) {
        const int previous = active_async_conn_index_;
        active_async_conn_index_ = conn_index;
        if (submit_async_) {
            AsyncSubmitState & state = async_submit_state_by_conn_[conn_index];
            state.params.clear();
            state.ids.clear();
            state.route_weights.clear();
            state.pending = false;
            state.graph_pending = false;
        }
        return previous;
    }

    void synchronize_async(RequestStats * request_stats) {
        if (!submit_async_) {
            return;
        }
        const auto it = async_submit_state_by_conn_.find(active_async_conn_index_);
        if (it == async_submit_state_by_conn_.end() || !it->second.pending) {
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_synchronize(backend_.get());
        const uint64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (request_stats != nullptr) {
            request_stats->ns_final_sync += elapsed;
            if (stats_.enabled() && is_vulkan_backend()) {
                request_stats->ns_vk_sync += elapsed;
            }
        }
        it->second.pending = false;
        it->second.graph_pending = false;
        it->second.params.clear();
        it->second.ids.clear();
        it->second.route_weights.clear();
    }

    void end_async_dispatch(int previous_conn_index) {
        synchronize_async(nullptr);
        active_async_conn_index_ = previous_conn_index;
    }

    AsyncSubmitState & async_submit_state() {
        return async_submit_state_by_conn_.at(active_async_conn_index_);
    }

    enum ggml_status submit_graph(
            ggml_cgraph * graph, RequestStats & request_stats,
            ggml_backend_graph_plan_t persistent_plan = nullptr) {
        if (submit_async_) {
            AsyncSubmitState & state = async_submit_state();
            if (is_vulkan_backend() && state.graph_pending) {
                // Vulkan reuses graph state, so drain before the next submit.
                synchronize_async(&request_stats);
            }
            state.pending = true;
            if (is_vulkan_backend()) {
                state.graph_pending = true;
            }
        }
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status;
        if (persistent_plan != nullptr) {
            status = ggml_backend_graph_plan_compute(backend_.get(), persistent_plan);
            if (!submit_async_) {
                ggml_backend_synchronize(backend_.get());
            }
        } else {
            status = submit_async_
                ? ggml_backend_graph_compute_async(backend_.get(), graph)
                : ggml_backend_graph_compute(backend_.get(), graph);
        }
        const uint64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        request_stats.ns_submit += elapsed;
        if (stats_.enabled() && is_vulkan_backend()) {
            request_stats.ns_vk_graph_compute += elapsed;
        }
        ++request_stats.n_graph_submits;
        return status;
    }

    // ---- prefetch hints (see note_prefetch_hint) ----
    //
    // WP_EXPERT_SPEC_PAGEIN=1 arms speculative page-ins. DEFAULT OFF and separate from the
    // spine.s WP_PREFETCH_HINT on purpose: with hints on and speculation off, a run
    // reads exactly what the config of record reads while still reporting what
    // was offered, so "the hint is wrong" and "the speculation is wrong" are two
    // separate experiments instead of one confounded one.
    const bool         spec_enabled_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PAGEIN");
        return e != nullptr && e[0] == '1';
    }();
    // WP_PREFILL_LAYER_AHEAD=1 -- while a prefill-shaped request for layer L
    // computes, spec-page the NEXT served layer's full catalog in disk order.
    // Default OFF. Width 8: n_tokens > 8 is prefill; decode/verify is 1..8
    // (same window as the arena path). WP_PREFILL_LAYER_AHEAD_WIDTH overrides.
    const bool         prefill_layer_ahead_ = [] {
        const char * e = std::getenv("WP_PREFILL_LAYER_AHEAD");
        return e != nullptr && e[0] == '1';
    }();
    const uint32_t     prefill_layer_ahead_width_ = [] {
        const char * e = std::getenv("WP_PREFILL_LAYER_AHEAD_WIDTH");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 8;
        return v > 0 ? (uint32_t) v : (uint32_t) 8;
    }();
    uint64_t           ahead_submits_ = 0;
    uint64_t           ahead_pages_   = 0;
    uint64_t           n_layerahead_hints_   = 0;
    uint64_t           n_layerahead_pageins_ = 0;
    int32_t            ahead_target_         = -1;
    int32_t            ahead_offered_layer_  = -1;
    std::vector<uint64_t> expert_recency_;
    uint64_t           recency_tick_ = 0;
    // WP_EXPERT_SPEC_CHUNK -- pages read per idle step. 1 by default: this is
    // the worst-case delay a real request can inherit from a speculative read already in
    // progress, about one 12.75 MB O_DIRECT read. Raising it trades that latency
    // for fewer round trips through poll().
    const size_t       spec_chunk_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_CHUNK");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 1;
        return v > 0 ? (size_t) v : (size_t) 1;
    }();
    // WP_EXPERT_SPEC_PREFILL_GATE=1 -- pause speculative SUBMISSION while the
    // last dispatch was prefill-shaped (n_tokens > 1). During the prefill sweep
    // the pool is in a guaranteed-eviction regime and spec LATE runs 84-100%:
    // every speculative read is pure drive contention against a demand stream
    // that already misses ~everything. Harvest of in-flight reads continues;
    // the gate opens on the first decode-shaped request. DEFAULT OFF until the
    // decomposition arm prices it.
    // *** WHAT COUNTS AS PREFILL. ***
    // This gate used to key on n_tokens > 1, which is NOT a prefill test on a
    // rig running speculative decode: a DSpark verify batch is 1 + n_draft
    // tokens (~8 at spec-draft-n-max=7), so EVERY decode step tripped it and
    // the speculative path was switched off permanently -- during exactly the
    // phase it exists to serve. Measured 2026-08-19: 61,803,183 of 61,845,095
    // pump calls exited here, 99.93%, while host_landed sat at 3.
    //
    // Same conflation as the dsv4 kq_mask gate (llama-kv-cache-dsv4.cpp): a
    // width that is >1 is not thereby a prompt. Prefill on this serve is
    // ubatch-wide (2048); the decode/spec window is a couple of dozen at most,
    // so 32 separates them with two orders of magnitude of margin -- and it is
    // the same 32 llm_graph_logit_row_cap uses for the same distinction.
    const uint32_t spec_prefill_gate_width_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PREFILL_WIDTH");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 32;
        return v > 0 ? (uint32_t) v : (uint32_t) 32;
    }();
    const bool spec_prefill_gate_enabled_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_PREFILL_GATE");
        return e != nullptr && e[0] == '1';
    }();
    bool spec_prefill_gate_active_ = false;
    // WP_SPEC_QUEUE_MAX -- bound each of spec_queue_/host_queue_. Newest-wins:
    // a full queue drops the OLDEST page (already the latest to land) so the
    // incoming hint keeps its lead. 64 ~= next-token L0-2 (3 x ~6) plus slack.
    // 0 = unbounded. Dropping the incoming page kept the stale backlog and
    // rejected the only useful hints.
    const size_t       spec_queue_max_ = [] {
        const char * e = std::getenv("WP_SPEC_QUEUE_MAX");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 64;
        return v >= 0 ? (size_t) v : (size_t) 64;
    }();
    // WP_SPEC_PREDICT_TOPM -- cap the PREDICTED (host_queue_) contribution per
    // frame. 0 = uncapped, AND THAT IS THE DEFAULT, deliberately.
    //
    // A frame arrives strictly ascending by expert id (the wire's dedup
    // invariant), so truncating it here keeps the M LOWEST-NUMBERED experts --
    // selection by nothing. The spine is the only side that knows how likely
    // each id was, so the real gate and cap live there
    // (WP_PREFETCH_CONF_MIN / WP_PREFETCH_TOPM); by the time a frame is on the
    // wire it has already been reduced to the ids worth reading. Set this
    // non-zero only as a blunt backstop against a spine that is not gating.
    const size_t       spec_predict_topm_ = [] {
        const char * e = std::getenv("WP_SPEC_PREDICT_TOPM");
        const long   v = (e != nullptr && e[0] != '\0') ? strtol(e, nullptr, 10) : 0;
        return v >= 0 ? (size_t) v : (size_t) 0;
    }();
    // WP_SPEC_SUBMIT_INTERLEAVE=1 (default on) -- submit one spec chunk after
    // each dispatch RESPONSE. Harvest/H2D stays on the idle pump.
    const bool         spec_submit_interleave_enabled_ = [] {
        const char * e = std::getenv("WP_SPEC_SUBMIT_INTERLEAVE");
        return e == nullptr || e[0] != '0';
    }();
    // WP_EXPERT_DOUBLE_BUFFER=1 (default OFF) -- submit the NEXT layer's
    // ALREADY-HINTED page-in reads at the moment THIS request's own compute
    // begins, instead of waiting for spec_pagein_after_dispatch (fires only
    // after THIS request's response has been sent) or the idle pump in
    // await_request (fires only when the socket has nothing waiting). Under a
    // pipelined spine the gap between "response sent" and "next frame recv'd"
    // is close to zero, so those two existing call sites almost never gave a
    // hinted read this request's whole compute span to run under -- only the
    // recv gap, which is the exact serialization this flag targets: "the next
    // layer's reads cannot start while the current layer computes."
    //
    // *** READS ONLY -- NO SECOND GPU SUBMIT. ***
    // spec_pagein_step(harvest=false) takes the submit branches only: it may
    // spawn an spec_host_submit landing thread (host RAM, no GPU at all) or,
    // if no spec-VRAM batch is already in flight, call spec_pagein_submit,
    // whose ensure_batch call spins up plain reader threads that only pread()
    // pages into staging buffers (see spec_pagein_submit's own comment: "READ
    // pages... WITHOUT building a compute batch"). It never reaches
    // pool_.spec_pagein_poll, which is the harvest half that runs
    // drain_one_read's H2D upload -- so this call issues no ggml/backend work
    // and never touches the backend off the dispatch thread. See
    // keepalive_tick's comment above: "Vulkan command pools have thread
    // affinity, so concurrent submits risk corruption even behind a mutex."
    // That upload still happens later, on this same dispatch thread, inside a
    // future batch.complete()/spec_pagein_poll -- exactly as it does today.
    //
    // *** STAGING BOUND: NOT A NEW ONE -- THE ALREADY-SHIPPED ONE. ***
    // ensure_batch clamps concurrent reader threads to
    // min(WP_EXPERT_READ_WORKERS, staging_.buffer_count()) FOR EVERY CALLER,
    // demand or spec-VRAM alike (see the s_read_workers comment ~30 lines
    // above ensure_batch's read_worker spawn loop), and spec_pagein_submit
    // refuses to start another spec-VRAM batch once spec_batches_ is already
    // at WP_EXPERT_SPEC_MAX_INFLIGHT (default 1, i.e. exactly the old "one
    // batch at a time" refusal). So the worst case this flag can produce is
    // exactly the one host_thread_cap_'s own comment already derives with
    // numbers AT THE DEFAULT CAP: one demand batch (<=4 readers by default)
    // plus one spec-VRAM batch (<=4 readers by default) concurrently = <=8 of
    // the pool's default 16 buffers, with host landings capped separately at
    // buffer_count()/2. Raising WP_EXPERT_SPEC_MAX_INFLIGHT scales the
    // spec-VRAM side of that bound linearly -- N batches of <=4 readers each
    // -- so an operator who raises it should also mind staging_.buffer_count()
    // headroom; this flag itself does not raise the per-batch reader cap, it
    // only makes batches overlap in time far more often than they used to
    // (previously the timing rarely lined up). StagingPool::borrow() was
    // already proven sound for exactly that overlap: no borrower holds a
    // lease while waiting on a second, so a blocked borrow always has a
    // draining page ahead of it and cannot deadlock, no matter how many
    // spec-VRAM batches are concurrently borrowing.
    //
    // *** WHY dispatch() AND NOT serve_connection(). ***
    // finish_split_dispatch() (the BEGIN/ACTS split path) calls this same
    // dispatch() with its batch already ensured at BEGIN time, so hooking
    // dispatch() covers both the plain PIPE_EXPERT_DISPATCH_REQ path and the
    // split path from one call site, and fires at each one's own true
    // compute-start instead of guessing a single point in serve_connection
    // that is right for neither.
    const bool         double_buffer_reads_ = [] {
        const char * e = std::getenv("WP_EXPERT_DOUBLE_BUFFER");
        return e != nullptr && e[0] == '1';
    }();

    template <typename T>
    void enqueue_newest(std::deque<T> & q, T item) {
        if (spec_queue_max_ != 0 && q.size() >= spec_queue_max_) {
            q.pop_front();
            ++spec_dropped_;
        }
        q.push_back(std::move(item));
    }
    // (page, lease) -- provenance is resolved to a lease at enqueue, so nothing
    // downstream has to know where a page came from.
    // Pump census, dispatch-thread only (spec_pagein_step is never concurrent
    // with itself), so plain counters -- no atomics needed.
    uint64_t                        pump_calls_         = 0;
    uint64_t                        pump_gated_         = 0;
    uint64_t                        pump_host_busy_     = 0;
    uint64_t                        pump_host_empty_    = 0;
    uint64_t                        pump_host_submit_   = 0;
    uint64_t                        pump_host_filtered_ = 0;
    uint64_t                        pump_vram_busy_     = 0;
    uint64_t                        pump_vram_empty_    = 0;
    uint64_t                        pump_vram_submit_   = 0;
    uint64_t                        pump_vram_demand_defer_ = 0;
    // WP_EXPERT_SPEC_DEMAND_FIRST=1 -- do not SUBMIT new speculative reads
    // (VRAM or host lane) while demand reads are outstanding. In-flight
    // speculative batches are untouched; the gate protects only the next
    // submission, so the worst case a demand read waits behind is one
    // already-submitted chunk instead of the whole spec queue. Motivated by
    // the 2026-08-22 request-anatomy measurement: demand ns_wait 6.9 ms vs a
    // ~2.8 ms clean pipeline, the gap being spec reads ahead of demand on the
    // shared readers/drive. DEFAULT OFF: byte-identical behaviour until A/B'd.
    const bool spec_demand_first_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_DEMAND_FIRST");
        return e != nullptr && e[0] == '1';
    }();
    std::deque<std::pair<const ExpertPage *, uint64_t>> spec_queue_;
    // Predicted pages bound for host RAM. Separate queue, not a flag on the
    // other one: they take a different read path to a different destination.
    std::deque<const ExpertPage *> host_queue_;
    // WP_EXPERT_SPEC_HOST=0 sends predictions back to VRAM on their short lease,
    // which is the arm this is meant to beat.
    const bool spec_host_enabled_ = [] {
        const char * e = std::getenv("WP_EXPERT_SPEC_HOST");
        return e == nullptr || e[0] != '0';
    }();
    uint64_t           spec_dropped_        = 0;
    uint64_t           hint_frames_         = 0;
    uint64_t           hint_experts_        = 0;
    uint64_t           hint_foreign_layer_  = 0;
    uint64_t           hint_foreign_expert_ = 0;
    uint64_t           hint_bad_            = 0;
    uint64_t           n_demand_prefetch_queued_ = 0;
    uint64_t           n_demand_prefetch_inflight_ = 0;

    // WP_HINT_LOG=path -- the counter line, appended after every hint frame and
    // fflushed, so `tail -1` is the final answer.
    //
    // WHY THIS EXISTS: report_prefetch_hints() writes to stderr only on a CLEAN
    // CONNECTION CLOSE, and the harness SIGKILLs workers at teardown. Arm 1
    // therefore produced NO foreign_expert number at all -- and foreign_expert
    // is the routing-agreement check, the one counter that proves spine and
    // worker resolve (layer, expert) through the same static hash. A disagreement
    // there would otherwise surface only much later, disguised as "prefetch
    // mysteriously does not help". Same failure and same fix as WP_PAGEIN_LOG's
    // per-line fflush, which was added after that flag's first run produced two
    // 0-byte logs.
    //
    // One line per hint frame rather than one at exit ON PURPOSE: it survives any
    // death, and it dates the FIRST frame at which a foreign count appears, which
    // is the next question if one ever does. ~150 bytes per frame against a frame
    // that already crossed WireGuard.
    void log_prefetch_hints() {
        if (logs_ == nullptr || logs_->hint == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(logs_->mutex);
        std::fprintf(logs_->hint, "C %s\n", prefetch_hint_line().c_str());
        std::fflush(logs_->hint);
    }

    // H -- what was PREDICTED, as received on the wire, before any shard filter.
    // Logged even for foreign ids: if spine and worker ever disagree about who
    // owns an expert, the ids are the evidence and the counters are only the
    // alarm.
    void log_hint_ids(const pipe_expert_prefetch_hint & hint) {
        if (logs_ == nullptr || logs_->hint == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(logs_->mutex);
        std::fprintf(logs_->hint, "H %d", hint.layer);
        for (int32_t expert_id : hint.expert_ids) {
            std::fprintf(logs_->hint, " %d", expert_id);
        }
        std::fputc('\n', logs_->hint);
        std::fflush(logs_->hint);
    }


    // ---- keepalive (see keepalive_tick) ----
    int                keepalive_us_    = 0;
    ggml_context     * keepalive_ctx_   = nullptr;
    ggml_gallocr_t     keepalive_alloc_ = nullptr;
    ggml_cgraph      * keepalive_graph_ = nullptr;

    void build_keepalive() {
        const char * e = std::getenv("WP_KEEPALIVE_US");
        keepalive_us_ = (e != nullptr && e[0] != '\0') ? atoi(e) : 0;
        if (keepalive_us_ <= 0) {
            return;
        }
        // Deliberately the smallest graph that still reaches the GPU: one add on
        // a 1-element tensor. It exists to occupy the device, not to compute.
        ggml_init_params p = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        keepalive_ctx_ = ggml_init(p);
        if (keepalive_ctx_ == nullptr) {
            keepalive_us_ = 0;
            return;
        }
        ggml_tensor * a   = ggml_new_tensor_1d(keepalive_ctx_, GGML_TYPE_F32, 1);
        ggml_tensor * sum = ggml_add(keepalive_ctx_, a, a);
        keepalive_graph_  = ggml_new_graph(keepalive_ctx_);
        ggml_build_forward_expand(keepalive_graph_, sum);
        keepalive_alloc_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_.get()));
        if (keepalive_alloc_ == nullptr ||
            !ggml_gallocr_alloc_graph(keepalive_alloc_, keepalive_graph_)) {
            if (keepalive_alloc_) { ggml_gallocr_free(keepalive_alloc_); keepalive_alloc_ = nullptr; }
            ggml_free(keepalive_ctx_); keepalive_ctx_ = nullptr;
            keepalive_graph_ = nullptr;
            keepalive_us_ = 0;
            return;
        }
        // Warm it once so the first real gap does not pay pipeline compilation.
        ggml_backend_graph_compute(backend_.get(), keepalive_graph_);
        fprintf(stderr, "wp keepalive enabled: every %d us while idle\n", keepalive_us_);
    }

    // *** WP_IO_SMALL_TOKENS: a small HOST-VISIBLE io buffer for decode. ***
    //
    // ns_prep is ~100% ggml_backend_tensor_set (the activation upload) and it
    // costs 30.5 us/expert on the 6900XT and 14.2 on the RX 480 during decode,
    // against 6.7 / 2.0 on the R9700 / 1070. Measured 2026-08-29: raising
    // GGML_VK_HOST_VISIBLE_VIDMEM_MAX_BYTES to 16 MiB collapsed the RX 480's
    // prep to 0.4 us/expert -- the io buffer had simply grown past the 1 MiB
    // threshold and gone device-local, making every upload a real H2D.
    //
    // That threshold CANNOT fix it: the io buffer is 10.1 MiB and the staging
    // buffers are 1.2-2.7 MiB, so any cutoff that makes io host-visible also
    // drags staging into GTT -- which cost 37% of decode. And the 512-token
    // prealloc cannot shrink, because in-serving grow_io_buffer() caused
    // 1.29-SECOND submits on the RX 480 (that is why the prealloc exists).
    //
    // So: keep the big device buffer for prefill and add a 1 MiB host-visible
    // one for decode. n_tokens=1 needs 10,240 bytes of activation, so 1 MiB
    // covers the input, the result, and the per-assignment partial slots that
    // resident-first / cpu-on-arrival pre-size. COSTS NO SLOTS -- it is host
    // memory (or BAR on Vulkan), not the expert slot budget.
    //
    // Portable entry point only. ggml_backend_dev_host_buffer_type is the same
    // one StagingPool uses; never a backend-specific host-alloc symbol (that
    // #if GGML_USE_* branching is a bug class that has recurred here). The
    // O_DIRECT hazard that makes pinned staging unsafe on Vulkan does NOT
    // apply: nothing ever read()s into the io buffer, it is only written by
    // ggml_backend_tensor_set.
    void alloc_io_small() {
        static const uint32_t small_tokens = [] {
            const char * e = std::getenv("WP_IO_SMALL_TOKENS");
            if (e == nullptr || e[0] == '\0') { return (uint32_t) 0; }
            const unsigned long v = std::strtoul(e, nullptr, 10);
            return v > 4096 ? (uint32_t) 0 : (uint32_t) v;
        }();
        if (small_tokens == 0 || backend_ == nullptr) {
            return;
        }
        // *** A PLAIN DEVICE BUFFER, DELIBERATELY -- NOT a host buffer type. ***
        // ggml_backend_dev_host_buffer_type() is NOT portable in the way that
        // matters here: on Vulkan it returns host-visible DEVICE (BAR) memory,
        // but on HIP/CUDA it returns real pinned HOST RAM, which ggml treats as
        // a CPU buffer. Attaching a graph input to one made the spine fail with
        // llama_decode ret=-3 on the first request (2026-08-29).
        //
        // The mechanism that actually works needs no special allocator. ggml-vulkan
        // keeps the host-visible BAR preference for any allocation with
        // size <= GGML_VK_HOST_VISIBLE_VIDMEM_MAX_BYTES (force_device_local is
        // `size > host_visible_max`). The io buffer is only slow because it grew
        // to 10.1 MiB and crossed that 1 MiB line, so every activation upload
        // became a real H2D. A SECOND buffer that stays under the line is an
        // ordinary device buffer -- fully graph-compatible -- that lands in the
        // BAR and takes memcpy writes.
        //
        // SCOPE: this is a VULKAN fix. On HIP/CUDA a small device buffer is
        // still device memory, so the 6900XT's 30.5 us/expert prep is NOT
        // addressed here and needs a different mechanism.
        const size_t one =
            (size_t) catalog_.descriptor.hparams.n_embd * small_tokens * sizeof(float);
        const size_t want = std::max<size_t>(1u << 20, 2 * (one + 65536));
        if (want > (1u << 20)) {
            // Above the default BAR threshold it would be forced device-local and
            // buy nothing, so do not spend the memory.
            fprintf(stderr,
                    "wp io-small: %zu bytes for n_tokens<=%u exceeds the 1 MiB "
                    "host-visible threshold, disabled\n", want, small_tokens);
            return;
        }
        buffer_ptr buf(ggml_backend_alloc_buffer(backend_.get(), want));
        if (!buf || ggml_backend_buffer_get_base(buf.get()) == nullptr) {
            fprintf(stderr, "wp io-small: allocation failed, staying on the big buffer\n");
            return;
        }
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        io_small_      = std::move(buf);
        io_small_size_ = want;
        fprintf(stderr, "wp io-small: %zu bytes device buffer for n_tokens<=%u "
                "(host-visible on Vulkan under the BAR threshold)\n", want, small_tokens);
        // SEPARATE FLAG, deliberately: the destination fix (io-small, Vulkan
        // BAR) and the source fix (pinned staging) address different things.
        // The destination fix is REAL and measured -- RX480 prep 14.2 -> 7.0.
        //
        // *** RETRACTION (2026-08-29). *** This comment used to also claim the
        // pinned-source fix "did NOTHING for the 6900XT (30.5 -> 30.4)" and
        // "made the GTX 1070 WORSE (2.1 -> 3.8)". BOTH CLAIMS ARE WITHDRAWN.
        // alloc_io_src_pinned prints "wp io-src: ..." on every outcome, success
        // or failure, and that string appears in NO log on EITHER machine. The
        // function was never called: it was nested inside alloc_io_small (which
        // main never runs, having no WP_IO_SMALL_TOKENS) and WP_IO_SRC_PINNED
        // was never actually set to 1 in those runs. Those numbers measured a
        // code path that did not execute. Pinning is UNTESTED, not refuted.
        // NOTE: pinned staging used to be allocated HERE. It is not, any more
        // -- see the alloc_io_src_pinned call in the constructor. Nesting it
        // inside this function silently tied it to WP_IO_SMALL_TOKENS, so
        // main's worker (which does not set that) had no pinned buffer at all
        // and WP_IO_SRC_PINNED=1 was a no-op there (found 2026-08-29).
    }

    // *** THE HIP/CUDA HALF OF THE FIX. ***
    // The Vulkan half above works by making the DESTINATION land in the BAR. On
    // HIP/CUDA a small device buffer is still device memory, so that does
    // nothing there -- and the destination CANNOT be host memory, because a
    // pinned host buffer is a CPU buffer to ggml and attaching a graph input to
    // one fails the spine with llama_decode ret=-3 (measured 2026-08-29).
    //
    // So fix the SOURCE instead. ggml_backend_tensor_set copies from the caller's
    // pointer, and `activation` is a std::vector -- pageable memory, which makes
    // the driver stage through its own internal bounce buffer on every upload.
    // Copying into a pinned host region first lets the H2D be a direct DMA. The
    // extra ~10 KB memcpy is ~1 us against a 30.4 us prep on the 6900XT.
    //
    // This buffer is NEVER attached to a tensor, so the CPU-buffer graph problem
    // does not arise; it is only ever the `data` argument. Applies to every
    // backend, which is the point -- the rig runs at the speed of its slowest
    // device, so a Vulkan-only fix is worth nothing.
    void alloc_io_src_pinned(size_t bytes) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend_.get());
        ggml_backend_buffer_type_t host_buft =
            dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (host_buft == nullptr) {
            fprintf(stderr, "wp io-src: no host buffer type, uploads stay pageable\n");
            return;
        }
        buffer_ptr buf(ggml_backend_buft_alloc_buffer(host_buft, bytes));
        void * base = buf ? ggml_backend_buffer_get_base(buf.get()) : nullptr;
        // ggml_backend_cuda_host_buffer_type_alloc_buffer silently falls back to
        // a plain CPU buffer when cudaHostAlloc fails and only stamps buft on the
        // success path -- so verify, or we would claim a pinned upload that is
        // actually pageable and attribute a null result to the wrong mechanism.
        if (!buf || base == nullptr ||
                ggml_backend_buffer_get_type(buf.get()) != host_buft) {
            fprintf(stderr, "wp io-src: pinned host buffer rejected, uploads stay pageable\n");
            return;
        }
        io_src_pinned_ = std::move(buf);
        io_src_base_   = base;
        io_src_size_   = bytes;
        fprintf(stderr, "wp io-src: %zu bytes pinned host staging for activation uploads\n", bytes);
    }

    void grow_io_buffer(size_t size, RequestStats & request_stats) {
        if (io_buffer_ && io_buffer_size_ >= size) {
            return;
        }
        // GEOMETRIC GROWTH. This used to allocate EXACTLY `size`, so a run whose
        // requests arrive at increasing n_tokens reallocated once per new high-
        // water mark (observed: 65536 -> 81936 -> 81952 -> 131072 -> 163920).
        // Only 5 allocations in a 64-token run -- but on Vulkan each one frees
        // the previous device buffer, and that stalls: measured 1.59 s total,
        // ~318 ms per growth, which amortised to 1.04 ms on EVERY request and
        // was 95% of prepare_io on the RX 480 (vs 0.001 ms on CUDA, where frees
        // are cheap). Doubling makes growth O(log n) instead of O(distinct
        // sizes), and the floor means the common decode case allocates once.
        static constexpr size_t IO_FLOOR = 1u << 20;   // 1 MiB: ~16 tokens of f32 activations
        size_t want = std::max(size, IO_FLOOR);
        if (io_buffer_ && want < io_buffer_size_ * 2) {
            want = io_buffer_size_ * 2;
        }
        // n_device_allocs is shared with the gallocr growth in compute_batch, so
        // it cannot say how many of these were io-buffer allocations. Time and
        // count them explicitly: ns_prep_grow is a FIXED ~1.4 s total regardless
        // of request count, so it is a few very expensive calls, not per-request.
        const auto grow_t0 = std::chrono::steady_clock::now();
        buffer_ptr buffer(ggml_backend_alloc_buffer(backend_.get(), want));
        fprintf(stderr, "wp io-buffer grow #%llu: %zu -> %zu bytes, alloc took %.1f ms\n",
                (unsigned long long) ++io_grow_count_, io_buffer_size_, want,
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - grow_t0).count() / 1000.0);
        if (!buffer) {
            throw std::runtime_error("failed to allocate persistent expert IO buffer");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        io_buffer_ = std::move(buffer);
        io_buffer_size_ = want;
        ++io_gen_;   // D2: cached graphs bound into the old buffer are stale
        ++request_stats.n_device_allocs;
    }

    // D1: persistent device span for the per-expert routing weights and gather
    // indices. Deliberately NOT part of io_buffer_: prepare_io uploads the
    // activations into io_buffer_ BEFORE compute_batch runs, and chunked calls
    // read `result` back out of it as the fold seed -- growing it mid-request
    // would silently drop both. This buffer's content never outlives one
    // compute_batch call, so growing it here is always safe.
    void grow_params_buffer(size_t size, RequestStats & request_stats) {
        if (params_buffer_ && params_buffer_size_ >= size) {
            return;
        }
        // Geometric growth for the same reason as grow_io_buffer: Vulkan frees
        // stall ~318 ms each, so reallocate O(log n) times, not per high-water
        // mark.
        static constexpr size_t PARAMS_FLOOR = 1u << 16;   // 64 KiB
        size_t want = std::max(size, PARAMS_FLOOR);
        if (params_buffer_ && want < params_buffer_size_ * 2) {
            want = params_buffer_size_ * 2;
        }
        buffer_ptr buffer(ggml_backend_alloc_buffer(backend_.get(), want));
        if (!buffer) {
            throw std::runtime_error("failed to allocate expert params buffer");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        params_buffer_ = std::move(buffer);
        params_buffer_size_ = want;
        ++params_gen_;   // D2: cached graphs bound into the old buffer are stale
        ++request_stats.n_device_allocs;
    }

    // Grouped GEMV scratch buffer for the 3N role-slice device-to-device copies
    // compute_batch_grouped issues per request. NOT gen-tracked
    // like io_/params_buffer_ -- compute_batch_grouped bypasses the D2 graph
    // cache entirely (see its header comment), so nothing outlives one call
    // that could be left dangling by a grow-triggered reallocation.
    void grow_batch_scratch(size_t size, RequestStats & request_stats) {
        if (batch_scratch_ && batch_scratch_size_ >= size) {
            return;
        }
        static constexpr size_t BATCH_FLOOR = 1u << 20;   // 1 MiB
        size_t want = std::max(size, BATCH_FLOOR);
        if (batch_scratch_ && want < batch_scratch_size_ * 2) {
            want = batch_scratch_size_ * 2;
        }
        buffer_ptr buffer(ggml_backend_alloc_buffer(backend_.get(), want));
        if (!buffer) {
            throw std::runtime_error("failed to allocate expert batch scratch buffer");
        }
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        batch_scratch_ = std::move(buffer);
        batch_scratch_size_ = want;
        ++request_stats.n_device_allocs;
    }

    ggml_tensor * make_io_tensor(
            ggml_context * ctx, uint32_t n_tokens, size_t offset) const {
        ggml_tensor * tensor = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd, n_tokens);
        // io_active_ is chosen per request in prepare_io; fall back to the big
        // buffer for any path that builds an io tensor without it.
        ggml_backend_buffer_t buf = io_active_ != nullptr ? io_active_ : io_buffer_.get();
        attach_weight(tensor, buf, ggml_backend_buffer_get_base(buf), offset);
        return tensor;
    }

    void prepare_io(
            const std::vector<float> & activation,
            uint32_t n_tokens,
            RequestStats & request_stats) {
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert IO metadata");
        }
        // Sub-timers: prepare_io measured 1.53 ms/req on the RX 480 vs 0.07 on
        // the 1070, and the io buffer is CONFIRMED host-visible (memcpy writes)
        // under the size-split policy -- so the cost is not the upload. Split
        // the function to find what it actually is.
        auto sub = std::chrono::steady_clock::now();
        auto sublap = [&request_stats, &sub](uint64_t & dst) {
            const auto now = std::chrono::steady_clock::now();
            dst += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(now - sub).count();
            sub = now;
        };
        ggml_tensor * input = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd, n_tokens);
        const ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(backend_.get());
        const size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        io_result_offset_ = GGML_PAD(input_size, alignment);
        const size_t result_size = ggml_backend_buft_get_alloc_size(buft, input);
        sublap(request_stats.ns_prep_setup);
        // Pick the buffer for THIS request. io_reserved_hint_ carries the larger
        // total that resident-first / cpu-on-arrival already reserved, so a
        // request with per-assignment partial slots cannot land in the small one.
        const size_t io_need =
            std::max(io_result_offset_ + result_size, io_reserved_hint_);
        io_reserved_hint_ = 0;
        if (io_small_ && io_need <= io_small_size_) {
            io_active_ = io_small_.get();
        } else {
            grow_io_buffer(io_result_offset_ + result_size, request_stats);
            io_active_ = io_buffer_.get();
        }
        sublap(request_stats.ns_prep_grow);
        attach_weight(
            input, io_active_, ggml_backend_buffer_get_base(io_active_), 0);
        sublap(request_stats.ns_prep_attach);
        const size_t act_bytes = activation.size() * sizeof(float);
        const void * src = activation.data();
        const bool pinned =
            io_src_base_ != nullptr && act_bytes <= io_src_size_;
        if (pinned) {
            std::memcpy(io_src_base_, activation.data(), act_bytes);
            src = io_src_base_;
        }
        // *** WP_IO_SET_ASYNC: the 6900XT's prep is a STALL, not a transfer. ***
        // ggml_backend_cuda_buffer_set_tensor is a SYNCHRONOUS cudaMemcpy (made
        // so deliberately by MAD-114, for cross-stream visibility on HIP), and
        // the sync form blocks on device-wide ordering. Measured 2026-08-29 on
        // ROCm1: 119.8 us/request with 99.6% of prepare_io inside that one call,
        // to move 10 KB -- 85 MB/s, three orders of magnitude under PCIe. Pinning
        // the SOURCE was already tried and did nothing (30.5 -> 30.4 us/expert),
        // which is what rules out staging overhead and leaves the host block.
        //
        // set_tensor_async issues cudaMemcpyAsync on cuda_ctx->stream() -- the
        // SAME stream the expert graph runs on -- so ordering is guaranteed by
        // the stream and MAD-114's cross-stream hazard does not arise. The host
        // does not block at all.
        //
        // TWO conditions, both required:
        //  * pinned source. cudaMemcpyAsync from PAGEABLE memory stages
        //    synchronously inside the driver, so async alone buys nothing --
        //    which is also why pinned alone bought nothing. They only work as a
        //    pair, and neither half was ever tested with the other.
        //  * a backend that HAS set_tensor_async. When the iface slot is null
        //    ggml_backend_tensor_set_async falls back to a full
        //    ggml_backend_synchronize + sync set, which is strictly WORSE than
        //    what we do today. Vulkan and CPU are gated out by name.
        //
        // Lifetime: the pinned buffer must not be rewritten before the copy
        // lands. Requests on a device are serialised by the blocking
        // batch.complete(), so the next prepare_io cannot run until this
        // request's compute -- and therefore this copy -- has finished.
        if (io_set_async_ && pinned) {
            ggml_backend_tensor_set_async(
                backend_.get(), input, src, 0, act_bytes);
        } else {
            ggml_backend_tensor_set(input, src, 0, act_bytes);
        }
        sublap(request_stats.ns_prep_set);
    }

    // all_experts=true builds ONE graph over every assignment in index order,
    // ignoring residency. See the determinism note on handle_request: splitting
    // by residency makes the floating-point ASSOCIATION depend on I/O timing.
    //
    // ArenaGraphKey/ArenaRoleKey/ArenaGraphEntry/ArenaGroup are defined below,
    // near compute_batch_arena_multi -- nested-class member order doesn't
    // matter in C++, and keeping the arena-cache types together with the
    // arena compute functions that use them is clearer than splitting them
    // across the file.
    void compute_batch(
            const pipe_expert_dispatch_req & request,
            const std::vector<const ExpertPage *> & pages,
            const ExpertSlotPool::Batch & batch,
            bool hits,
            bool add_previous,
            RequestStats & request_stats,
            bool all_experts = false,
            // force_dense: ignore the gather path for this call only. Used by
            // WP_SELFCHECK to compute the SAME request both ways and diff them.
            bool force_dense = false,
            // Half-open ASSIGNMENT-INDEX range this call computes. Used by the
            // read/compute overlap to run a leading chunk of experts while the
            // tail is still being read. entries_ is indexed by assignment index
            // (is_resident(i) reads entries_.at(i)), and pageins carry the same
            // index, so this range selects a matching set of page-ins.
            // Default [0, SIZE_MAX) = every expert, i.e. unchanged behaviour.
            size_t sel_begin = 0,
            size_t sel_end = std::numeric_limits<size_t>::max(),
            // Overrides where `result` (and the D2/D3 fast paths, which
            // assume io_result_offset_) land. SIZE_MAX (default) = the
            // existing single shared slot at io_result_offset_, i.e.
            // unchanged behaviour for every pre-existing caller.
            //
            // WP_EXPERT_RESIDENT_FIRST (see dispatch()) is the only caller
            // that sets this: it computes each assignment into its OWN
            // buffer slot instead of accumulating into the shared one, so a
            // later fold can re-associate them in canonical index order.
            // D2's cache key does NOT include this offset, so a cache hit
            // would happily replay a graph built for a DIFFERENT slot and
            // silently write the wrong buffer -- correctness bug, not a perf
            // one. D3 (compute_batch_grouped) is hardwired to
            // io_result_offset_ the same way. Both are disabled below
            // whenever this is overridden; see the two guards further down.
            size_t result_offset = std::numeric_limits<size_t>::max()) {
        const bool measure_vk = stats_.enabled() && is_vulkan_backend();
        const std::chrono::steady_clock::time_point vk_compute_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        const auto record_vk_compute = [&]() {
            if (measure_vk) {
                request_stats.ns_vk_compute_path +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - vk_compute_started).count();
            }
        };
        const std::chrono::steady_clock::time_point vk_wait_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        batch.wait_copy_event(backend_.get());
        if (measure_vk) {
            request_stats.ns_vk_wait +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - vk_wait_started).count();
        }
        // Everything from here to the D2 cache lookup was untimed: the
        // selection scan, the arena probe, the buffer-type alignment queries,
        // grow_params_buffer and the gather-rank pass. On the RX 480 that span
        // is the only remaining unnamed cost inside ns_vk_dispatch_path once
        // graph_compute is subtracted, so name it.
        const std::chrono::steady_clock::time_point vk_setup_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        const auto record_vk_setup = [&]() {
            if (measure_vk) {
                request_stats.ns_vk_setup +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - vk_setup_started).count();
            }
        };
        const auto selected = [&](size_t i) {
            return i >= sel_begin && i < sel_end &&
                   (all_experts || (batch.is_resident(i) == hits));
        };
        size_t n_selected = 0;
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            n_selected += selected(i) ? 1 : 0;
        }
        if (n_selected == 0) {
            record_vk_setup();
            record_vk_compute();
            return;
        }

        const std::chrono::steady_clock::time_point probe_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        const bool arena_ok = arena_id_eligible(request, batch);
        if (measure_vk) {
            const uint64_t probe_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - probe_started).count();
            request_stats.ns_arena_probe += probe_ns;
            request_stats.ns_vk_arena_probe += probe_ns;
        }
        if (arena_ok &&
                sel_begin == 0 && sel_end >= request.assignments.size() &&
                n_selected == request.assignments.size() &&
                result_offset == std::numeric_limits<size_t>::max() &&
                compute_batch_arena(request, pages, batch, request_stats)) {
            record_vk_setup();
            record_vk_compute();
            return;
        }

        // *** ROUTING-DENSITY GATHER/SCATTER (WP_EXPERT_GATHER=1, default OFF). ***
        // MEASURED 2026-08-04: at n_tokens=512 an assigned expert receives only
        // ~3.5% of the tokens, so the dense path below multiplies 96.5% of its
        // expert FLOPs by a ZERO router weight (28.7x waste in PREFILL; VERIFY is
        // 2.3x; decode is exactly 1.0x and thus unaffected). This path instead
        // gathers each expert's routed tokens with ggml_get_rows, runs the FFN at
        // the compacted width, and scatters back with scatter_compact_rows
        // (zero dest + ggml_set_rows). idx is unique per expert so overwrite
        // equals the old get_rows_back scatter-ADD. Kill switch
        // WP_EXPERT_SCATTER_SET_ROWS=0 restores get_rows_back.
        // *** CONFIG OF RECORD 2026-08-04: DEFAULT ON. *** Measured +25.8% prefill on
        // its own at n_ubatch=512, and SUPER-ADDITIVE with n_ubatch: the pair
        // (ub1024 + gather) is +67.3% against +45.3% predicted multiplicatively,
        // because a wider ubatch manufactures more zero-weight work for gather to
        // strip.
        // Set WP_EXPERT_GATHER=0 to fall back to the dense masked path.
        //
        // *** CORRECTION 2026-08-04 EVENING: "decode is unaffected" WAS WRONG. ***
        // That claim (previously on this line) rested on decode TOK/S, which sits
        // inside this harness's ~11.5% same-config noise floor and so could not have
        // resolved the effect. The spine's per-token decode dispatch wait CAN: eight
        // paired dense-vs-gather arms, alternating within one sweep, put gather at
        // +100 ms/token on that number, 8 for 8 (dense 177-253 ms -> gather 290-366 ms).
        // MECHANISM, and it is not subtle: decode routing density is EXACTLY 100%
        // (measured pre-gather; at n_tokens=1 an assigned expert has that token routed
        // by definition). So at decode ggml_get_rows compacts NOTHING and
        // scatter writes NOTHING back -- gather is pure added graph nodes
        // per expert per layer buying zero saved FLOPs. It is a prefill optimisation
        // that was billed to decode.
        // Hence WP_EXPERT_GATHER_MIN_TOKENS: gather only once a request is wide enough
        // for the compaction to pay for its own nodes. Default 2 = bypass at decode.
        // Set it to 1 to restore the always-gather behaviour (the A/B control).
        static const bool s_gather = parse_env_default_on(std::getenv("WP_EXPERT_GATHER"));
        // Default 2: decode (n_tokens==1) is 100% dense, so gather is pure
        // extra nodes. Prefill/verify (n_tokens>=2) still gather. =1 restores
        // always-gather. Static assign makes this bit-stable (the 2026-08-04
        // determinism miss was timing-dependent placement, now gone).
        static const int s_gather_min_tokens =
            parse_gather_min_tokens(std::getenv("WP_EXPERT_GATHER_MIN_TOKENS"));
        // Default ON: linear set_rows scatter. =0 restores get_rows_back.
        static const bool s_set_rows =
            parse_env_default_on(std::getenv("WP_EXPERT_SCATTER_SET_ROWS"));
        // PER-REQUEST, not static: prefill and decode requests interleave in one
        // worker, so this must be decided per request and never cached.
        const bool use_gather = use_expert_gather(
            request.n_tokens, force_dense, s_gather_min_tokens, s_gather);
        // Default ON: one tensor_set for routing weights (+ gather idx).
        // Byte-identical to the per-tensor path. Decode graph-cache requires it.
        static const bool s_params_coalesce =
            parse_env_default_on(std::getenv("WP_EXPERT_PARAMS_COALESCE")) ||
            wp_persistent_graphs_enabled();
        // D1 packing state (moved above the D2 cache: the hit path packs and
        // uploads without ever building a graph). Offsets are assigned in build
        // order; params_host mirrors the device span byte-for-byte (pad = 0).
        // The buffer is grown BEFORE any tensor attaches into it -- growing
        // reallocates, which would orphan already-attached tensors (and bumps
        // params_gen_, which invalidates every cached graph bound into it).
        const size_t params_align = s_params_coalesce
            ? ggml_backend_buft_get_alignment(
                  ggml_backend_get_default_buffer_type(backend_.get()))
            : 1;

        // *** GROUPED GEMV: GROUPED mul_mat_id ACROSS ALL SELECTED EXPERTS. ***
        // Collapses the per-expert loop's 3 ggml_mul_mat (gate/up/down) into 3
        // ggml_mul_mat_id calls total, batched over every selected expert. See
        // compute_batch_grouped for the full rationale, the mul_mat_id layout,
        // and why gate+up are NOT merged into one call (that would require
        // clamping a non-contiguous split view, which ggml_cuda_op_clamp gets
        // silently wrong for n_rows > 1 -- see the comment there).
        // The old batch flag is dense-only and whole-request-only: gather's
        // per-expert idx varies (breaks one shared ids tensor), and a partial
        // [sel_begin,sel_end) chunk covers only some assignments. The new
        // grouped GEMV arm also uses the dense route-weight representation for
        // n_tokens=1..8, so it remains one shared batched dispatch. The D2 graph cache
        // is a separate rebind model -- this path never uses it.
        static const bool s_batch_moe = [] {
            const char * e = std::getenv("WP_EXPERT_BATCH_MOE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        // WP_EXPERT_GROUPED_GEMV=1 is the explicit decode/verify arm. Keep it
        // separate from WP_EXPERT_BATCH_MOE: that older switch was not safe on
        // width-sliced pages. The helper uses the round-3 scratch layout and
        // accepts at most eight routed assignments per token.
        const bool grouped_gemv = grouped_gemv_eligible(request);
        const bool persistent_graphs = wp_persistent_graphs_enabled();
        static const bool s_worker_collapse = [] {
            const char * e = std::getenv("WP_WORKER_COLLAPSE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        const bool d3_grouped =
            !persistent_graphs &&
            (grouped_gemv || (s_batch_moe && !use_gather) || (s_worker_collapse && !use_gather)) &&
                sel_begin == 0 &&
                sel_end >= request.assignments.size() &&
                result_offset == std::numeric_limits<size_t>::max();
        if (s_worker_collapse && !d3_grouped && !request_stats.d3_counted) {
            ++request_stats.n_d3_bounce;
            request_stats.d3_counted = true;
        }
        if (d3_grouped) {
            record_vk_setup();
            compute_batch_grouped(
                request, pages, batch, selected, n_selected, add_previous, request_stats,
                s_worker_collapse, grouped_gemv);
            record_vk_compute();
            return;
        }
        std::vector<uint8_t> params_host;
        size_t params_span = 0;
        if (s_params_coalesce) {
            const size_t per =
                GGML_PAD((size_t) request.n_tokens * sizeof(float), params_align);
            grow_params_buffer(2 * per * n_selected + params_align, request_stats);
        }
        const auto place_param = [&](ggml_tensor * t, const void * data, size_t nbytes) {
            const size_t off = GGML_PAD(params_span, params_align);
            params_host.resize(off, 0);   // zero-fill the alignment gap
            params_host.insert(
                params_host.end(),
                (const uint8_t *) data, (const uint8_t *) data + nbytes);
            attach_weight(
                t, params_buffer_.get(),
                ggml_backend_buffer_get_base(params_buffer_.get()), off);
            params_span = off + nbytes;
        };

        // D2 cache lookup (see the member comment). COALESCE required so the
        // hit path can repack routing (+ gather idx) at fixed offsets.
        // Dense (decode n=1) always keys. Gather keys only when every selected
        // expert has the SAME idx rank — verify n=2-8 is almost always rank 1
        // per expert, so those graphs now hit instead of rebuilding. Mixed
        // ranks (rare; one expert got 2 tokens, another 1) skip the cache.
        static const bool s_graph_cache =
            parse_env_default_on(std::getenv("WP_EXPERT_GRAPH_CACHE")) ||
            wp_persistent_graphs_enabled();
        static const size_t s_graph_cache_max = [] {
            const char * e = std::getenv("WP_EXPERT_GRAPH_CACHE_MAX");
            const long v = (e != nullptr && e[0] != '\0') ? std::strtol(e, nullptr, 10) : 0;
            const size_t requested = v > 0 ? (size_t) v : (size_t) 16;
            return wp_persistent_graphs_enabled()
                ? std::min(requested, (size_t) 2) : requested;
        }();
        // *** WP_EXPERT_FUSE_GATE_UP=1 (DEFAULT OFF). ***
        //
        // THE COST THIS ATTACKS. The dense per-expert chain below is six graph
        // nodes -- mul_mat(gate), mul_mat(up), swiglu_split, mul_mat(down),
        // mul(route_w), add(fold) -- and every one is a separate Vulkan
        // dispatch with a pipeline barrier between dependent pairs. MEASURED
        // 2026-08-29: the 2026 leg costs ~165-175 us PER EXPERT at n_tokens=4
        // (30 experts, 5.32 ms/request) and ~200 us/expert at n_tokens=1, while
        // an expert's whole 0.9 MiB weight slice is only ~4 us of RX 480
        // bandwidth. The per-expert cost is launch and barrier overhead, so the
        // lever is NODES PER EXPERT, and it is linear in expert count -- it
        // bounds both single-stream decode and width scaling.
        //
        // THE MECHANISM. gate and up are the same type, the same shape
        // [n_embd, n_ff_slice], and layout_sliced_pages() packs an expert's
        // roles back to back in slot order, so when `up` sits exactly
        // ggml_row_size(type, ne0) * ne1 bytes after `gate` the pair is ONE
        // contiguous [n_embd, 2*n_ff_slice] matrix. One mul_mat over that
        // produces gate rows 0..ne1-1 and up rows ne1..2*ne1-1 in a single
        // dispatch, and non-split ggml_swiglu() consumes exactly that layout
        // (out[i] = silu(a[i]) * a[i + ne1]) -- which is what swiglu_split(gate,
        // up) computes. Six nodes per expert become five, the two smallest
        // dispatches (192 rows -> 48 workgroups each on Polaris' rm_kq=4)
        // become one 96-workgroup dispatch, and one barrier disappears.
        //
        // WHY IT SHOULD BE BIT-EXACT (and why it is still behind a flag). Each
        // output row of a mul_mat_vec is an independent reduction over k by the
        // same 64 lanes of the same shader; k (2560) does not change and the
        // GCN pipeline choice does not depend on the row count (dmmv_wg is
        // pinned to DMMV_WG_SIZE_SUBGROUP on AMD_GCN), so row j's dot product
        // is computed identically whether the matrix has 192 rows or 384. That
        // argument is sound but UNVERIFIED on hardware here, hence default OFF.
        //
        // THREE HARD GUARDS, all checked per request, all falling back silently:
        //  * dense only. gather rebuilds ffn_in per expert; the fusion is
        //    orthogonal but untested there.
        //  * swiglu_clamp must be off. The clamp is ASYMMETRIC -- up gets
        //    [-L, L] and gate gets [-INF, L] -- so a single clamp on the fused
        //    tensor is a different function. See the clamp note below.
        //  * every selected expert must actually have up adjacent to gate.
        //    Checked against the real device offsets, not assumed from the
        //    layout algorithm.
        static const bool s_fuse_gate_up = [] {
            const char * e = std::getenv("WP_EXPERT_FUSE_GATE_UP");
            return e != nullptr && e[0] == '1';
        }();
        const auto fuse_gate_up_ok = [&]() {
            if (!s_fuse_gate_up || use_gather || request.swiglu_clamp > 1e-6f) {
                return false;
            }
            const auto & fspecs = catalog_.descriptor.layers.at(request.layer);
            const RoleSpec & sg = fspecs.at("gate");
            const RoleSpec & su = fspecs.at("up");
            if (sg.type != su.type || sg.ne0 != su.ne0 || sg.ne1 != su.ne1) {
                return false;
            }
            const size_t gate_bytes =
                ggml_row_size(sg.type, (int64_t) sg.ne0) * (size_t) sg.ne1;
            for (size_t i = 0; i < request.assignments.size(); ++i) {
                if (!selected(i)) {
                    continue;
                }
                const ExpertPage & page = *pages[i];
                const uint64_t go = page.roles.at("gate").device_offset;
                const uint64_t uo = page.roles.at("up").device_offset;
                if (uo != go + (uint64_t) gate_bytes) {
                    return false;
                }
            }
            return true;
        };
        const bool fuse_gate_up = fuse_gate_up_ok();
        uint32_t gather_rank = 0;
        bool     gather_rank_uniform = !use_gather;
        if (use_gather) {
            gather_rank_uniform = true;
            for (size_t i = 0; i < request.assignments.size(); ++i) {
                if (!selected(i)) {
                    continue;
                }
                const uint32_t k =
                    (uint32_t) compact_routing_rows(request.assignments[i].weights).idx.size();
                if (gather_rank == 0) {
                    gather_rank = k;
                } else if (k != gather_rank) {
                    gather_rank_uniform = false;
                    gather_rank = 0;
                    break;
                }
            }
        }
        record_vk_setup();
        GraphCacheEntry * gc = nullptr;
        bool gc_hit = false;
        const std::chrono::steady_clock::time_point vk_cache_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        if (s_graph_cache && s_params_coalesce && gather_rank_uniform &&
                result_offset == std::numeric_limits<size_t>::max()) {
            GraphKey key;
            key.n_tokens     = request.n_tokens;
            key.n_selected   = (uint32_t) n_selected;
            key.idx_rank     = gather_rank;
            key.add_previous = add_previous;
            key.fused_gate_up = fuse_gate_up;
            std::memcpy(&key.clamp_bits, &request.swiglu_clamp, sizeof(key.clamp_bits));
            // See the note on GraphKey: a cache hit rebinds pointers, never
            // types, so two layers whose expert tensors are quantized
            // differently must not share an entry.
            {
                const auto & key_specs = catalog_.descriptor.layers.at(request.layer);
                key.type_gate = (uint32_t) key_specs.at("gate").type;
                key.type_up   = (uint32_t) key_specs.at("up").type;
                key.type_down = (uint32_t) key_specs.at("down").type;
                if (wp_persistent_graphs_enabled()) {
                    key.ne0_gate  = key_specs.at("gate").ne0;
                    key.ne1_gate  = key_specs.at("gate").ne1;
                    key.ne0_up    = key_specs.at("up").ne0;
                    key.ne1_up    = key_specs.at("up").ne1;
                    key.ne0_down  = key_specs.at("down").ne0;
                    key.ne1_down  = key_specs.at("down").ne1;
                }
            }
            auto it = graph_cache_.find(key);
            // graph == nullptr marks a half-built entry (an exception hit the
            // build path after insertion): stale, rebuild. Buffer-generation
            // mismatch means the device buffer it was bound into was replaced.
            if (it != graph_cache_.end() &&
                    (it->second.graph == nullptr ||
                     it->second.io_gen != io_gen_ ||
                     it->second.params_gen != params_gen_ ||
                     (wp_persistent_graphs_enabled() &&
                      it->second.io_buffer !=
                          (io_active_ != nullptr ? io_active_ : io_buffer_.get())))) {
                if (it->second.persistent_plan != nullptr) {
                    release_persistent_plan(it->second.persistent_plan, &request_stats);
                }
                graph_cache_.erase(it);
                it = graph_cache_.end();
            }
            if (it != graph_cache_.end()) {
                gc = &it->second;
                gc_hit = true;
            } else {
                if (graph_cache_.size() >= s_graph_cache_max) {
                    auto victim = graph_cache_.begin();
                    for (auto j = graph_cache_.begin(); j != graph_cache_.end(); ++j) {
                        if (j->second.last_used < victim->second.last_used) {
                            victim = j;
                        }
                    }
                    if (victim->second.persistent_plan != nullptr) {
                        release_persistent_plan(victim->second.persistent_plan, &request_stats);
                    }
                    graph_cache_.erase(victim);
                }
                gc = &graph_cache_[key];
                gc->io_gen = io_gen_;
                gc->params_gen = params_gen_;
                if (wp_persistent_graphs_enabled()) {
                    gc->io_buffer = io_active_ != nullptr ? io_active_ : io_buffer_.get();
                }
            }
            gc->last_used = ++graph_cache_tick_;
        }
        if (measure_vk) {
            request_stats.ns_vk_cache_lookup +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - vk_cache_started).count();
        }

        if (gc_hit) {
            // *** THE D2 FAST PATH: no context, no graph build, no gallocr. ***
            // Rebind this request's expert weights into the cached graph (src
            // data ptrs only -- the selected backend graph path handles it),
            // repack the routing blob at the SAME offsets (attach is a no-op
            // re-bind to the same place), upload once, submit the cached graph.
            // Iteration order matches the build below exactly: assignment index.
            ++request_stats.n_gcache_hit;
            const std::chrono::steady_clock::time_point vk_rebind_started =
                measure_vk ? std::chrono::steady_clock::now() :
                              std::chrono::steady_clock::time_point();
            static const char * kRoles[3] = {"gate", "up", "down"};
            size_t k = 0;
            for (size_t i = 0; i < request.assignments.size(); ++i) {
                if (!selected(i)) {
                    continue;
                }
                const ExpertPage & page = *pages[i];
                const ExpertSlotPool::Loaded loaded = batch.loaded(i);
                for (int j = 0; j < 3; ++j) {
                    // Fused graphs leave the `up` slot null: its rows live in
                    // the [ne0, 2*ne1] tensor already rebound at slot `gate`.
                    ggml_tensor * w = gc->expert_w[k * 3 + (size_t) j];
                    if (w == nullptr) {
                        continue;
                    }
                    attach_weight(
                        w, loaded.buffer, loaded.base,
                        page.roles.at(kRoles[j]).device_offset);
                }
                const auto & wv = request.assignments[i].weights;
                if (use_gather) {
                    const CompactRouting compact = compact_routing_rows(wv);
                    uint64_t nz = 0;
                    for (float f : compact.weights) { nz += (f != 0.0f); }
                    request_stats.n_weight_nonzero += nz;
                    request_stats.n_weight_total += compact.idx.size();
                    place_param(gc->gather_idx[k], compact.idx.data(),
                                compact.idx.size() * sizeof(int32_t));
                    place_param(gc->route_w[k], compact.weights.data(),
                                compact.weights.size() * sizeof(float));
                } else {
                    uint64_t nz = 0;
                    for (float f : wv) { nz += (f != 0.0f); }
                    request_stats.n_weight_nonzero += nz;
                    request_stats.n_weight_total += wv.size();
                    place_param(gc->route_w[k], wv.data(), wv.size() * sizeof(float));
                }
                ++k;
            }
            if (measure_vk) {
                request_stats.ns_vk_rebind +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - vk_rebind_started).count();
            }
            if (params_span > 0) {
                const auto params_started = std::chrono::steady_clock::now();
                if (submit_async_) {
                    AsyncSubmitState & state = async_submit_state();
                    state.params.emplace_back(std::move(params_host));
                    state.pending = true;
                    ggml_backend_tensor_set_async(
                        backend_.get(), gc->blob, state.params.back().data(), 0, params_span);
                } else {
                    ggml_backend_tensor_set(gc->blob, params_host.data(), 0, params_span);
                }
                const uint64_t params_elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - params_started).count();
                request_stats.ns_params_set += params_elapsed;
                if (measure_vk) {
                    request_stats.ns_vk_params_set += params_elapsed;
                }
            }
            enum ggml_status status = submit_graph(
                gc->graph, request_stats, gc->persistent_plan);
            if (status != GGML_STATUS_SUCCESS && gc->persistent_plan != nullptr) {
                release_persistent_plan(gc->persistent_plan, &request_stats);
                std::fprintf(stderr, "WARN wp expert worker: persistent Vulkan graph replay failed; retrying normal graph compute\n");
                status = submit_graph(gc->graph, request_stats);
            }
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cached expert backend graph compute failed");
            }
            record_vk_compute();
            return;
        }
        if (gc != nullptr) {
            ++request_stats.n_gcache_miss;
        }

        const auto build_started = std::chrono::steady_clock::now();
        // gather adds per expert: idx, sub_input, scattered (+ the add) -- budget
        // generously, ggml_init only reserves metadata.
        // +2 tensors / +2 nodes per expert for the SwiGLU clamp pair (up, gate)
        // added 2026-08-05. These are budgeted unconditionally even when the
        // clamp is off: under-budgeting the graph is a hard allocation failure,
        // and ggml_init only reserves metadata, so the slack is free.
        // +1 constant for the coalesced-params upload blob (D1), budgeted
        // unconditionally -- same free-slack argument as the clamp pair.
        const size_t tensor_count = (s_gather ? 20 : 14) * n_selected + 9;
        const size_t graph_nodes  = (s_gather ? 12 :  8) * n_selected + 3;
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                              ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        // D2 miss: the context (and everything built in it) must OUTLIVE this
        // call, so it lives in the cache entry; uncached calls keep the local.
        // `ctx` stays a reference so the build below is unchanged either way.
        context_ptr ctx_local;
        context_ptr & ctx = gc != nullptr ? gc->ctx : ctx_local;
        ctx.reset(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate batched expert graph metadata");
        }

        ggml_tensor * input = make_io_tensor(ctx.get(), request.n_tokens, 0);
        ggml_set_input(input);
        const size_t effective_result_offset =
            result_offset == std::numeric_limits<size_t>::max() ? io_result_offset_ : result_offset;
        ggml_tensor * result = make_io_tensor(
            ctx.get(), request.n_tokens, effective_result_offset);
        // *** SEED THE FOLD, DO NOT ADD AT THE END. ***
        // The accumulator below is a LEFT-FOLD in assignment-index order:
        //     sum = ((((e0 + e1) + e2) + ...))
        // A chunked caller (WP_EXPERT_COMPUTE_CHUNKS) must continue that exact
        // fold, so the previous chunks' running total has to be the SEED. Adding
        // it at the end instead computes
        //     (e8 + ... + e15) + (e0 + ... + e7)
        // which is the same experts in the same order but a DIFFERENT
        // ASSOCIATION, and FP addition is not associative. Measured 2026-08-05:
        // that re-association alone moved draft acceptance 0.84286 -> 0.77966,
        // because the router's top-k is discontinuous and amplifies a last-bit
        // delta into a different token. Seeding makes chunked output bit-identical
        // to the unchunked path.
        //
        // Gather+set_rows writes into `result` (the io buffer). Zero it on the
        // first chunk so scatter-add starts from 0; later chunks keep the seed.
        if (use_gather && s_set_rows && !add_previous) {
            result = ggml_scale_inplace(ctx.get(), result, 0.0f);
        }
        ggml_tensor * sum = (add_previous || (use_gather && s_set_rows)) ? result : nullptr;
        std::vector<std::pair<ggml_tensor *, const pipe_expert_assignment *>> routing_weights;
        routing_weights.reserve(n_selected);
        // Per-expert contributions in assignment order, for WP_EXPERT_FOLD_LAST.
        std::vector<ggml_tensor *> fold_terms;
        fold_terms.reserve(n_selected);
        // Parallel to routing_weights: the gathered token indices per expert, kept
        // alive until after ggml_gallocr_alloc_graph so they can be uploaded.
        std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> gather_idx;
        gather_idx.reserve(n_selected);

        for (size_t i = 0; i < request.assignments.size(); ++i) {
            if (!selected(i)) {
                continue;
            }
            const ExpertPage & page = *pages[i];
            const ExpertSlotPool::Loaded loaded = batch.loaded(i);
            const auto & specs = catalog_.descriptor.layers.at(page.layer);
            const auto make_weight = [&](const std::string & role) {
                const RoleSpec & spec = specs.at(role);
                ggml_tensor * tensor =
                    ggml_new_tensor_2d(ctx.get(), spec.type, spec.ne0, spec.ne1);
                attach_weight(
                    tensor, loaded.buffer, loaded.base, page.roles.at(role).device_offset);
                // D2 miss: record for rebinding on later hits. Creation order is
                // gate, up, down per expert -- the hit path indexes k*3+j on
                // exactly that order.
                if (gc != nullptr) {
                    gc->expert_w.push_back(tensor);
                }
                return tensor;
            };

            // GATHER: restrict this expert to the tokens actually routed to it.
            ggml_tensor * ffn_in = input;
            ggml_tensor * idx_t  = nullptr;
            std::vector<int32_t> idx;
            if (use_gather) {
                const CompactRouting compact =
                    compact_routing_rows(request.assignments[i].weights);
                idx = compact.idx;
                idx_t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t) idx.size());
                ggml_set_input(idx_t);
                if (s_params_coalesce) {
                    place_param(idx_t, idx.data(), idx.size() * sizeof(int32_t));
                }
                if (gc != nullptr) {
                    gc->gather_idx.push_back(idx_t);
                }
                ffn_in = ggml_get_rows(ctx.get(), input, idx_t);
            }
            ggml_tensor * hidden = nullptr;
            if (fuse_gate_up) {
                // ONE weight tensor spanning gate then up, attached at gate's
                // offset; adjacency was verified in fuse_gate_up_ok() above.
                // Push a null in the `up` slot so the D2 hit path's k*3+j
                // indexing (gate, up, down) is preserved unchanged.
                const RoleSpec & sg = specs.at("gate");
                ggml_tensor * gate_up = ggml_new_tensor_2d(
                    ctx.get(), sg.type, sg.ne0, 2 * (int64_t) sg.ne1);
                attach_weight(
                    gate_up, loaded.buffer, loaded.base,
                    page.roles.at("gate").device_offset);
                if (gc != nullptr) {
                    gc->expert_w.push_back(gate_up);
                    gc->expert_w.push_back(nullptr);
                }
                // [2*ne1, n_rows] -> swiglu halves it: out[i] = silu(a[i]) * a[i+ne1],
                // i.e. exactly ggml_swiglu_split(gate, up).
                hidden = ggml_swiglu(
                    ctx.get(), ggml_mul_mat(ctx.get(), gate_up, ffn_in));
            }
            ggml_tensor * gate = nullptr;
            ggml_tensor * up   = nullptr;
            if (hidden == nullptr) {
                gate = ggml_mul_mat(ctx.get(), make_weight("gate"), ffn_in);
                up   = ggml_mul_mat(ctx.get(), make_weight("up"), ffn_in);
            }
            // *** SwiGLU CLAMP. ADDED 2026-08-05 -- ITS ABSENCE WAS A CORRECTNESS BUG. ***
            // Mirrors the LLM_ARCH_DEEPSEEK4 branch of build_moe_ffn() in
            // src/llama-graph.cpp EXACTLY: up is clamped symmetrically, the GATE is
            // clamped ABOVE ONLY (-INFINITY lower bound) and fed to swiglu_split,
            // which applies silu to it. Do not "simplify" this to a symmetric gate
            // clamp or to silu-then-clamp -- those are the OTHER architectures'
            // branch and give different numbers.
            // The spine cannot do this for us: build_moe_ffn() returns at the
            // `expert_dispatch != nullptr` branch before ever reaching the clamp.
            // NOTE the asymmetry: it is WHY fuse_gate_up bails out whenever the
            // clamp is armed. A single ggml_clamp over the fused [2*ne1] tensor
            // would apply the same bounds to both halves and gate must NOT get
            // a lower bound.
            const float swiglu_limit = request.swiglu_clamp;
            if (hidden == nullptr) {
                if (swiglu_limit > 1e-6f) {
                    up   = ggml_clamp(ctx.get(), up,   -swiglu_limit, swiglu_limit);
                    gate = ggml_clamp(ctx.get(), gate, -INFINITY,     swiglu_limit);
                }
                hidden = ggml_swiglu_split(ctx.get(), gate, up);
            }
            ggml_tensor * output = ggml_mul_mat(ctx.get(), make_weight("down"), hidden);
            // SHAPE MATTERS: [1, n_tokens], NOT [n_tokens]. output is
            // [n_embd, n_tokens]; ggml_mul broadcasts src1 into src0 via
            // ggml_can_repeat, which only checks ne[i] % src1->ne[i] == 0. A
            // 1-D [n_tokens] tensor PASSES that check (6144 % 8 == 0) and then
            // scales along the EMBEDDING axis instead of the token axis --
            // silently wrong, no assert. It is correct only at n_tokens == 1,
            // where both readings coincide, so decode looked fine while PREFILL
            // was corrupted and poisoned the KV cache.
            const int64_t n_rows = use_gather ? (int64_t) idx.size() : request.n_tokens;
            ggml_tensor * weights =
                ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_rows);
            ggml_set_input(weights);
            if (s_params_coalesce) {
                // Same values the per-tensor path uploads after alloc_graph:
                // the compacted (gather) or full (dense) router weights.
                const auto & wv = request.assignments[i].weights;
                if (use_gather) {
                    std::vector<float> compact;
                    compact.reserve(idx.size());
                    for (int32_t t : idx) { compact.push_back(wv[(size_t) t]); }
                    place_param(weights, compact.data(), compact.size() * sizeof(float));
                } else {
                    place_param(weights, wv.data(), wv.size() * sizeof(float));
                }
            }
            if (gc != nullptr) {
                gc->route_w.push_back(weights);
            }
            ggml_tensor * weighted = ggml_mul(ctx.get(), output, weights);
            const std::chrono::steady_clock::time_point fold_started =
                measure_vk ? std::chrono::steady_clock::now() :
                              std::chrono::steady_clock::time_point();
            if (use_gather) {
                // Scatter compacted rows back to [n_embd, n_tokens]. Default is
                // set_rows into a zero dest (linear in n_sel). get_rows_back is
                // the O(n_tokens * n_sel) dest scan; keep it behind =0.
                if (s_set_rows) {
                    // Accumulate into the io result. Do not scale(input,0) —
                    // that is a second [n_embd, n_tokens] compute-buffer tensor
                    // and is what OOM'd the 1070 at ubatch 2048.
                    sum = scatter_add_compact_rows(ctx.get(), sum, weighted, idx_t);
                } else {
                    weighted = ggml_get_rows_back(ctx.get(), weighted, idx_t, input);
                    sum = sum ? ggml_add(ctx.get(), sum, weighted) : weighted;
                }
            } else {
                sum = sum ? ggml_add(ctx.get(), sum, weighted) : weighted;
            }
            if (measure_vk) {
                request_stats.ns_vk_fold +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - fold_started).count();
            }
            routing_weights.emplace_back(weights, &request.assignments[i]);
            gather_idx.emplace_back(idx_t, std::move(idx));
            fold_terms.push_back(weighted);
        }
        // (add_previous is folded in as the SEED above, not appended here.)
        if (sum == nullptr) {
            // Defensive: nothing contributed. Cannot happen now that empty experts
            // keep a zero-weight row, but a null here segfaults inside ggml_cpy
            // with no diagnostic, so refuse loudly rather than crash the worker.
            throw std::runtime_error("expert compute produced no contribution");
        }
        ggml_tensor * copy = ggml_cpy(ctx.get(), sum, result);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        // *** WP_EXPERT_FOLD_LAST=1 (DEFAULT OFF): EMIT THE FOLD AS ONE RUN. ***
        //
        // WHAT IT FIXES. ggml-vulkan fuses a run of GGML_OP_ADD nodes into ONE
        // multi_add dispatch (ggml_vk_fuse_multi_add), up to MAX_FUSED_ADDS
        // sources -- but ONLY when the adds are CONSECUTIVE in cgraph->nodes.
        // ggml_build_forward_expand(copy) walks the fold chain depth first, so
        // the emitted order is
        //     [expert 0 chain] [expert 1 chain] add0 [expert 2 chain] add1 ...
        // Every add is separated from the next by five nodes of the following
        // expert, so the fusion NEVER fires here and the fold costs n-1
        // separate, strictly serial dispatches -- n-1 pipeline barriers and
        // n-1 GPU round trips on top of the n-1 kernels. At n_tokens=4 with 30
        // experts that is 29 avoidable dispatches per request.
        //
        // THE CHANGE IS EMISSION ORDER ONLY. Expanding each per-expert
        // contribution first marks those subgraphs visited, so the later
        // expand of `copy` appends add0..add(n-2) back to back. The tensors,
        // the dependencies and the left-fold ASSOCIATION are untouched --
        // sum = ((e0+e1)+e2)+... exactly as before -- and ggml executes in
        // topological order either way, so with the fusion inactive this is
        // bit-identical.
        //
        // WHY IT IS STILL DEFAULT OFF: once the run IS fused, multi_add.comp
        // sums all sources in one shader invocation rather than as a chain of
        // pairwise adds. That is the same set of values in the same order, but
        // it is a different sequence of roundings, and this fold's association
        // is exactly what moved draft acceptance 0.84286 -> 0.77966 once
        // before (see the SEED THE FOLD note above). Two more preconditions
        // are outside this file's control and must be checked on the box:
        // vk_device::multi_add requires shaderRoundingModeRTEFloat16, and
        // MAX_FUSED_ADDS caps the run length.
        //
        // Dense only: the gather arm folds with scatter_add_compact_rows, not
        // ggml_add, so there is no run to make consecutive.
        static const bool s_fold_last = [] {
            const char * e = std::getenv("WP_EXPERT_FOLD_LAST");
            return e != nullptr && e[0] == '1';
        }();
        if (s_fold_last && !use_gather) {
            for (ggml_tensor * term : fold_terms) {
                ggml_build_forward_expand(graph, term);
            }
        }
        ggml_build_forward_expand(graph, copy);

        // D2 miss: the cached graph needs its OWN allocator. Sharing
        // compute_galloc_ would let the next uncached shape re-plan the buffer
        // and move this graph's intermediate tensors out from under it.
        ggml_gallocr_t galloc = compute_galloc_.get();
        if (gc != nullptr) {
            gc->galloc.reset(ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_.get())));
            if (!gc->galloc) {
                throw std::runtime_error("failed to create cached graph allocator");
            }
            galloc = gc->galloc.get();
        }
        const size_t old_compute_size =
            ggml_gallocr_get_buffer_size(galloc, 0);
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            throw std::runtime_error("failed to allocate batched expert compute graph");
        }
        if (ggml_gallocr_get_buffer_size(galloc, 0) > old_compute_size) {
            ++request_stats.n_device_allocs;
        }
        if (gc != nullptr && wp_persistent_graphs_enabled()) {
            gc->io_buffer = io_active_ != nullptr ? io_active_ : io_buffer_.get();
        }
        request_stats.ns_graph_build +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - build_started).count();

        for (size_t k = 0; k < routing_weights.size(); ++k) {
            const auto & item = routing_weights[k];
            const auto & wv = item.second->weights;
            uint64_t nz = 0;
            for (float f : wv) { nz += (f != 0.0f); }
            request_stats.n_weight_nonzero += nz;
            // n_weight_total is what was actually COMPUTED, so it must follow the
            // path taken: the compacted row count when gathering, the full ubatch
            // when not. Otherwise the density counter would report the dense
            // figure even after the fix and hide whether it worked.
            if (use_gather) {
                const auto & idx = gather_idx[k].second;
                request_stats.n_weight_total += idx.size();
                if (!s_params_coalesce) {
                    std::vector<float> compact;
                    compact.reserve(idx.size());
                    for (int32_t t : idx) { compact.push_back(wv[(size_t) t]); }
                    const auto params_started = std::chrono::steady_clock::now();
                    ggml_backend_tensor_set(
                        item.first, compact.data(), 0, compact.size() * sizeof(float));
                    ggml_backend_tensor_set(
                        gather_idx[k].first, idx.data(), 0, idx.size() * sizeof(int32_t));
                    const uint64_t params_elapsed =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - params_started).count();
                    request_stats.ns_params_set += params_elapsed;
                    if (measure_vk) {
                        request_stats.ns_vk_params_set += params_elapsed;
                    }
                }
            } else {
                request_stats.n_weight_total += wv.size();
                if (!s_params_coalesce) {
                    const auto params_started = std::chrono::steady_clock::now();
                    ggml_backend_tensor_set(
                        item.first, wv.data(), 0, wv.size() * sizeof(float));
                    const uint64_t params_elapsed =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - params_started).count();
                    request_stats.ns_params_set += params_elapsed;
                    if (measure_vk) {
                        request_stats.ns_vk_params_set += params_elapsed;
                    }
                }
            }
        }
        if (s_params_coalesce && params_span > 0) {
            // THE single upload D1 exists for: every routing weight and gather
            // idx this call packed, in one tensor_set. The blob tensor is not
            // part of the graph -- it is only a typed window over the span so
            // the backend API can address it.
            ggml_tensor * blob =
                ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) params_span);
            attach_weight(
                blob, params_buffer_.get(),
                ggml_backend_buffer_get_base(params_buffer_.get()), 0);
            if (gc != nullptr) {
                gc->blob = blob;
            }
            const auto params_started = std::chrono::steady_clock::now();
            if (submit_async_) {
                AsyncSubmitState & state = async_submit_state();
                state.params.emplace_back(std::move(params_host));
                state.pending = true;
                ggml_backend_tensor_set_async(
                    backend_.get(), blob, state.params.back().data(), 0, params_span);
            } else {
                ggml_backend_tensor_set(blob, params_host.data(), 0, params_span);
            }
            const uint64_t params_elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - params_started).count();
            request_stats.ns_params_set += params_elapsed;
            if (measure_vk) {
                request_stats.ns_vk_params_set += params_elapsed;
            }
        }
        // Warm the first request through the normal path. Vulkan may need to
        // allocate prealloc buffers while it records a graph; recording a plan
        // before that setup would leave this shape permanently on the fallback.
        enum ggml_status status = submit_graph(graph, request_stats);
        if (status != GGML_STATUS_SUCCESS) {
            if (gc != nullptr) {
                gc->graph = nullptr;
            }
            throw std::runtime_error("batched expert backend graph compute failed");
        }
        if (gc != nullptr) {
            gc->graph = graph;
            if (wp_persistent_graphs_enabled() && is_vulkan_backend()) {
                if (submit_async_) {
                    synchronize_async(&request_stats);
                } else {
                    ggml_backend_synchronize(backend_.get());
                }
                gc->persistent_plan = create_persistent_plan(graph);
            }
        }
        // D2: the entry becomes valid ONLY now -- a build that threw anywhere
        // above leaves graph == nullptr and the next lookup rebuilds it.
    }

    // WP_EXPERT_ARENA_ID=1 and persistent GPU graphs use grouped dispatch over
    // the slot arena. Cache hits only upload slot ids and router weights.

    // *** GROUPED GEMV: GROUPED mul_mat_id ACROSS EVERY SELECTED EXPERT. ***
    //
    // WHY. The per-expert loop above issues 3 ggml_mul_mat (gate, up, down) +
    // 2 ggml_clamp + 1 swiglu_split + 1 ggml_mul (router weight) + 1 ggml_add
    // (fold) PER SELECTED EXPERT -- ~8 kernel dispatches/expert, ~64 for an
    // 8-expert request. At n_tokens 2-3 (spec-verify) this worker is
    // dispatch-launch-overhead-bound (~0.15 ms/dispatch), not FLOP-bound, so
    // dispatch COUNT is the lever, not per-op efficiency.
    //
    // WHAT THIS DOES NOT DO: merge gate+up into ONE mul_mat_id call. That was
    // investigated first (concatenate the byte-contiguous up+gate weight
    // region -- confirmed by wp-repack-lib.cpp's build_expert_groups, which
    // sorts group members by role_mask ascending, i.e. UP(1) < GATE(2) <
    // DOWN(4) -- into a [n_embd, 2*n_ff_slice] weight and split the mul_mat_id
    // OUTPUT into gate/up views). That output split is a ROW-CONTIGUOUS but
    // MATRIX-STRIDED view whenever n_expert_used*n_tokens > 1 (nb1 spans the
    // full 2*n_ff_slice row, ne0 only covers half of it), and
    // ggml_cuda_op_clamp (ggml/src/ggml-cuda/clamp.cu) computes
    // `dst[i] = clamp(x[i])` over a FLAT `ggml_nelements(src0)` range with NO
    // stride awareness at all -- it is only correct when src0 is fully
    // contiguous. ggml-cuda's CLAMP supports_op() (ggml-cuda.cu, the
    // GGML_OP_CLAMP case) returns true unconditionally, so nothing catches
    // this: the merged path would silently compute wrong gate/up values for
    // every request with n_expert_used*n_tokens > 1, i.e. every request this
    // worker actually serves. (The exact same hazard exists, unaudited, in
    // src/llama-graph.cpp build_moe_ffn's own "merged gate_up path" --
    // ggml_view_3d + ggml_clamp on the split -- for any model that ships a
    // consolidated ffn_gate_up_exps tensor and runs on CUDA/HIP with
    // n_tokens > 1. Out of scope here: that file belongs to the graph
    // builder, not this worker.) Keeping gate and up as SEPARATE mul_mat_id
    // calls costs one extra matmul dispatch but means their outputs are
    // freshly-allocated, genuinely contiguous tensors -- clamp is then
    // provably safe regardless of n_tokens.
    //
    // HOW THE BATCHED WEIGHT TENSOR IS BUILT. mul_mat_id's `as` argument needs
    // one tensor with UNIFORM per-expert stride (nb2). The normal grouped path
    // copies each selected role into its page-sized scratch slot. The
    // WP_WORKER_COLLAPSE path (collapse_copies) uses one typed strided source
    // per role when the selected slots form a regular span. CUDA/HIP give
    // each role an allocated typed owner and copy through an I32/F16 byte
    // view of that owner. Vulkan keeps the role-major scratch layout.
    // Irregular spans, and callers with collapse_copies off (e.g. the plain
    // WP_EXPERT_GROUPED_GEMV / WP_EXPERT_BATCH_MOE arms), use the typed
    // per-role dense-pack fallback.
    //
    // NET: collapse mode has 3 role copies + 3 mul_mat_id + 2 clamp + 1
    // swiglu_split + 1 weight-mul + (n_selected - 1) fold-adds + 1 final cpy.
    // The fold stays a left fold to keep the legacy floating-point association;
    // reducing it with a tree would save nodes but change output bits.
    //
    // MATH: identical to the existing dense (non-gather) path. Every selected
    // expert is computed for every token (weights[i][t] is 0 for tokens not
    // routed to expert i, exactly as the per-expert dense loop already does),
    // clamp bounds and swiglu_split are byte-for-byte the same calls, and the
    // final reduction folds in ASSIGNMENT-INDEX order via sequential
    // ggml_add -- the same left-fold association documented at the top of
    // compute_batch (SEED THE FOLD, DO NOT ADD AT THE END), using the exact
    // per-expert-view + sequential-add idiom already proven in
    // src/llama-graph.cpp build_moe_ffn's own expert aggregation loop
    // (ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])
    // then ggml_add), which is stride-safe: ggml-cuda's binbcast kernel
    // (add.cu/binbcast.cu) is a proper multi-dim indexed kernel, not a flat
    // one like clamp -- unlike clamp, ADD/MUL over these per-expert-per-token
    // views is safe on CUDA/HIP for any n_tokens.
    //
    // NOT SUPPORTED (falls back to the per-expert loop): gather (variable
    // per-expert token counts break the single shared `ids` tensor), the
    // sel_begin/sel_end chunked-compute overlap, and the D2 persistent graph
    // cache (D2's rebind model reattaches weight tensors directly into live
    // slot buffers each request; this path instead copies into a private
    // scratch buffer every request, which is a different graph shape D2 was
    // never taught about). Integrating D2 here -- caching the graph and only
    // reissuing the 3N copies + two small uploads per request -- is a
    // reasonable follow-up once this path is validated on real hardware.
    void compute_batch_grouped(
            const pipe_expert_dispatch_req & request,
            const std::vector<const ExpertPage *> & pages,
            const ExpertSlotPool::Batch & batch,
            const std::function<bool(size_t)> & selected,
            size_t n_selected,
            bool add_previous,
            RequestStats & request_stats,
            bool collapse_copies,
            bool warn_grouped_gemv) {
        const bool measure_vk = stats_.enabled() && is_vulkan_backend();
        const auto build_started = std::chrono::steady_clock::now();

        // Assignment indices selected for this call, in order -- this IS the
        // fold order (see the header comment).
        std::vector<size_t> sel;
        sel.reserve(n_selected);
        for (size_t i = 0; i < request.assignments.size(); ++i) {
            if (selected(i)) {
                sel.push_back(i);
            }
        }
        const size_t n = sel.size();
        // compute_batch already returned early for n_selected == 0.

        // One layer per request (pipe_expert_dispatch_req::layer is a single
        // field), so every selected page shares one RoleSpec map and one
        // page size -- both required for the uniform nb2 stride below.
        const int layer = pages[sel[0]]->layer;
        struct GroupedInvocationLog {
            bool     enabled;
            int      layer;
            uint32_t n_tokens;
            size_t   n_experts;
            uint64_t page_device_size;
            bool     success = false;

            ~GroupedInvocationLog() {
                // ONLY ON FAILURE unless explicitly asked for. This used to log
                // on EVERY invocation because the caller passes `grouped_gemv`
                // itself as `enabled`, so turning the path on turned the logging
                // on: measured 148,287 unbuffered stderr writes in ONE decode
                // arm (2026-08-29). That is a syscall per request in the hot
                // path, and it silently taxes any measurement of the very path
                // it is reporting on. WP_WARN_GROUPED_GEMV=1 restores the
                // verbose form for debugging.
                static const bool verbose = [] {
                    const char * e = std::getenv("WP_WARN_GROUPED_GEMV");
                    return e != nullptr && e[0] == '1';
                }();
                if (!enabled || (success && !verbose)) {
                    return;
                }
                std::fprintf(
                    stderr,
                    "WARN wp grouped gemv: layer=%d n_tokens=%u n_experts=%zu "
                    "page_device_size=%llu status=%s\n",
                    layer, n_tokens, n_experts,
                    (unsigned long long) page_device_size,
                    success ? "success" : "failure");
            }
        } grouped_log{
            warn_grouped_gemv,
            layer,
            request.n_tokens,
            n,
            pages[sel[0]]->device_size,
        };

        const auto & specs = catalog_.descriptor.layers.at(layer);
        const RoleSpec & gate_spec = specs.at("gate");
        const RoleSpec & up_spec   = specs.at("up");
        const RoleSpec & down_spec = specs.at("down");
        const uint64_t page_size = pages[sel[0]]->device_size;
        const int64_t  n_embd    = catalog_.descriptor.hparams.n_embd;
        const uint32_t n_tokens  = request.n_tokens;

        // Generously budgeted -- ggml_init only reserves metadata (no device
        // memory), so slack here is free, and undersizing is a hard abort in
        // ggml_new_graph_custom / ggml_build_forward_expand. Actual usage is
        // roughly: 2n (typed role copies: src+dst) + 3 (batched role weights) + n
        // (per-expert contrib views) + (n-1) (fold adds) + ~15 fixed
        // (ids, input, gate/up/down outputs, clamps, swiglu, weighted mul,
        // result, final copy) tensors; nodes are the same set minus the
        // non-graph leaves (as_gate/as_up/as_down/ids/route_w/input/result).
        // CUDA collapse replaces each packed destination with a view. The
        // tensor count is unchanged; the three views need three extra nodes.
        // The typed fallback creates 6n copy tensors (THREE roles x src+dst
        // per expert, not the 2n the old comment claimed), and cgraph's leaf
        // table is sized by `graph_nodes` too — leaf count ~6n+6 overflowed
        // the old 5n+27 budget at n>=22 (live crash: n=23, ggml.c:7411).
        const size_t tensor_count = 14 * n + 60;
        const size_t graph_nodes  = 8 * n + 40;
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                              ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate grouped expert graph metadata");
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);

        const ExpertPage & page0 = *pages[sel[0]];
        const auto buft = ggml_backend_get_default_buffer_type(backend_.get());
        const size_t scratch_align = ggml_backend_buft_get_alignment(buft);
        if (scratch_align == 0) {
            throw std::runtime_error("invalid grouped scratch alignment");
        }
        void * scratch_base = nullptr;
        const auto role_bytes = [](const RoleSpec & spec) {
            return ggml_row_size(spec.type, spec.ne0) * (size_t) spec.ne1;
        };
        const auto role_copy_type = [&](const RoleSpec & spec) {
            const size_t bytes = role_bytes(spec);
            return bytes % sizeof(uint32_t) == 0 ? GGML_TYPE_I32 :
                   bytes % sizeof(uint16_t) == 0 ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
        };
        const auto role_copy_count = [&](const RoleSpec & spec) {
            const enum ggml_type type = role_copy_type(spec);
            return type == GGML_TYPE_COUNT ? 0 : role_bytes(spec) / ggml_type_size(type);
        };

        const char * const backend_name = ggml_backend_name(backend_.get());
        const bool cuda_collapse = collapse_copies && backend_name != nullptr &&
            (std::strstr(backend_name, "CUDA") != nullptr ||
             std::strstr(backend_name, "ROCm") != nullptr);
        bool use_batched_copies = collapse_copies;
        ggml_backend_buffer_t source_buffer = nullptr;
        void * source_base = nullptr;
        uint64_t source_offset = 0;
        uint64_t source_stride = 0;
        if (use_batched_copies) {
            const ExpertSlotPool::Loaded first_loaded = batch.loaded(sel[0]);
            source_buffer = first_loaded.buffer;
            const uintptr_t first_address = reinterpret_cast<uintptr_t>(first_loaded.base);
            source_base = source_buffer == nullptr ? nullptr :
                          ggml_backend_buffer_get_base(source_buffer);
            const uintptr_t buffer_address = reinterpret_cast<uintptr_t>(source_base);
            if (source_buffer == nullptr || source_base == nullptr ||
                    first_loaded.base == nullptr || first_address < buffer_address) {
                use_batched_copies = false;
            } else {
                source_offset = first_address - buffer_address;
                if (n == 1) {
                    source_stride = page_size;
                } else {
                    const ExpertSlotPool::Loaded second_loaded = batch.loaded(sel[1]);
                    const uintptr_t second_address =
                        reinterpret_cast<uintptr_t>(second_loaded.base);
                    if (second_loaded.buffer != source_buffer ||
                            second_address <= first_address) {
                        use_batched_copies = false;
                    } else {
                        source_stride = second_address - first_address;
                        for (size_t k = 2; k < n; ++k) {
                            const ExpertSlotPool::Loaded loaded = batch.loaded(sel[k]);
                            const uintptr_t address =
                                reinterpret_cast<uintptr_t>(loaded.base);
                            if (loaded.buffer != source_buffer ||
                                    address < first_address ||
                                    source_stride > (UINTPTR_MAX - first_address) / k ||
                                    address != first_address + k * source_stride) {
                                use_batched_copies = false;
                                break;
                            }
                        }
                    }
                }
                if (use_batched_copies) {
                    if (source_stride == 0) {
                        use_batched_copies = false;
                    }
                    for (size_t k = 0; k < n; ++k) {
                        const ExpertPage & page = *pages[sel[k]];
                        if (page.device_size != page_size ||
                                page.roles.at("gate").device_offset !=
                                    page0.roles.at("gate").device_offset ||
                                page.roles.at("up").device_offset !=
                                    page0.roles.at("up").device_offset ||
                                page.roles.at("down").device_offset !=
                                    page0.roles.at("down").device_offset) {
                            use_batched_copies = false;
                            break;
                        }
                    }
                }
                if (role_copy_type(gate_spec) == GGML_TYPE_COUNT ||
                        role_copy_type(up_spec) == GGML_TYPE_COUNT ||
                        role_copy_type(down_spec) == GGML_TYPE_COUNT) {
                    use_batched_copies = false;
                }
            }
        }

        if (collapse_copies && !request_stats.d3_counted) {
            if (use_batched_copies) {
                ++request_stats.n_d3_collapse;
            } else {
                ++request_stats.n_d3_typed;
            }
            request_stats.d3_counted = true;
        }

        size_t gate_region = 0;
        size_t up_region = 0;
        size_t down_region = 0;
        ggml_tensor * as_gate = nullptr;
        ggml_tensor * as_up = nullptr;
        ggml_tensor * as_down = nullptr;
        if (use_batched_copies) {
            if (cuda_collapse) {
                as_gate = ggml_new_tensor_3d(
                    ctx.get(), gate_spec.type, gate_spec.ne0, gate_spec.ne1, (int64_t) n);
                as_up = ggml_new_tensor_3d(
                    ctx.get(), up_spec.type, up_spec.ne0, up_spec.ne1, (int64_t) n);
                as_down = ggml_new_tensor_3d(
                    ctx.get(), down_spec.type, down_spec.ne0, down_spec.ne1, (int64_t) n);

                const auto byte_alias = [&](ggml_tensor * owner, const RoleSpec & spec) {
                    const enum ggml_type copy_type = role_copy_type(spec);
                    const size_t copy_count = role_copy_count(spec);
                    ggml_tensor * alias = ggml_view_3d(
                        ctx.get(), owner, owner->ne[0], owner->ne[1], owner->ne[2],
                        owner->nb[1], owner->nb[2], 0);
                    alias->type = copy_type;
                    alias->ne[0] = (int64_t) copy_count;
                    alias->ne[1] = 1;
                    alias->ne[2] = (int64_t) n;
                    alias->nb[0] = ggml_type_size(copy_type);
                    alias->nb[1] = ggml_row_size(copy_type, copy_count);
                    alias->nb[2] = owner->nb[2];
                    alias->nb[3] = alias->nb[2] * (size_t) n;
                    return alias;
                };
                const auto batched_copy = [&](const RoleSpec & spec, const char * role,
                                              ggml_tensor * owner) {
                    const enum ggml_type copy_type = role_copy_type(spec);
                    ggml_tensor * src = ggml_new_tensor_3d(
                        ctx.get(), copy_type, (int64_t) role_copy_count(spec), 1, (int64_t) n);
                    src->nb[2] = source_stride;
                    ggml_tensor * alias = byte_alias(owner, spec);
                    attach_weight(src, source_buffer, source_base,
                                  source_offset + page0.roles.at(role).device_offset);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), src, alias));
                };
                batched_copy(gate_spec, "gate", as_gate);
                batched_copy(up_spec, "up", as_up);
                batched_copy(down_spec, "down", as_down);
            } else {
                const auto region_size = [&](const RoleSpec & spec) {
                    ggml_tensor * probe = ggml_new_tensor_3d(
                        ctx.get(), spec.type, spec.ne0, spec.ne1, (int64_t) n);
                    return GGML_PAD(ggml_backend_buft_get_alloc_size(buft, probe), scratch_align);
                };
                gate_region = region_size(gate_spec);
                up_region = region_size(up_spec);
                down_region = region_size(down_spec);
                const size_t up_offset = gate_region;
                const size_t down_offset = up_offset + up_region;
                grow_batch_scratch(down_offset + down_region, request_stats);
                scratch_base = ggml_backend_buffer_get_base(batch_scratch_.get());
                as_gate = ggml_new_tensor_3d(
                    ctx.get(), gate_spec.type, gate_spec.ne0, gate_spec.ne1, (int64_t) n);
                as_up = ggml_new_tensor_3d(
                    ctx.get(), up_spec.type, up_spec.ne0, up_spec.ne1, (int64_t) n);
                as_down = ggml_new_tensor_3d(
                    ctx.get(), down_spec.type, down_spec.ne0, down_spec.ne1, (int64_t) n);
                attach_weight(as_gate, batch_scratch_.get(), scratch_base, 0);
                attach_weight(as_up, batch_scratch_.get(), scratch_base, up_offset);
                attach_weight(as_down, batch_scratch_.get(), scratch_base, down_offset);

                const auto batched_copy = [&](const RoleSpec & spec, const char * role,
                                              size_t region_offset) {
                    const enum ggml_type copy_type = role_copy_type(spec);
                    ggml_tensor * src = ggml_new_tensor_3d(
                        ctx.get(), copy_type, (int64_t) role_copy_count(spec), 1, (int64_t) n);
                    src->nb[2] = source_stride;
                    ggml_tensor * packed = ggml_new_tensor_3d(
                        ctx.get(), copy_type, (int64_t) role_copy_count(spec), 1, (int64_t) n);
                    attach_weight(packed, batch_scratch_.get(), scratch_base,
                                  region_offset);
                    attach_weight(src, source_buffer, source_base,
                                  source_offset + page0.roles.at(role).device_offset);
                    ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), src, packed));
                };
                batched_copy(gate_spec, "gate", 0);
                batched_copy(up_spec, "up", up_region);
                batched_copy(down_spec, "down", gate_region + up_region);
            }
        } else {
            // Dense-pack each role: [role_bytes * n] contiguous per region, so
            // the batched weight tensors below have CANONICAL strides. The old
            // page-image layout (dst offset = role_offset + k*page_size, then
            // t->nb[2] = page_size) put non-canonically-strided mxfp4 tensors
            // in a COMPUTE-usage buffer, which trips CUDA mmvq's padding-clear
            // GGML_ASSERT(ggml_is_contiguously_allocated(src0)) on every
            // request that misses the collapse-copy fast path.
            const auto role_region = [&](const RoleSpec & spec) {
                // Size by the backend's ALLOC size, not nbytes: quantized
                // types get row padding (mmvq's over-read guard), and an
                // exact-fit region fails attach_weight's bounds check.
                ggml_tensor * probe = ggml_new_tensor_3d(
                    ctx.get(), spec.type, spec.ne0, spec.ne1, (int64_t) n);
                return GGML_PAD(ggml_backend_buft_get_alloc_size(buft, probe), scratch_align);
            };
            const size_t gate_reg = role_region(gate_spec);
            const size_t up_reg = role_region(up_spec);
            const size_t down_reg = role_region(down_spec);
            const size_t up_base = gate_reg;
            const size_t down_base = gate_reg + up_reg;
            grow_batch_scratch(down_base + down_reg, request_stats);
            scratch_base = ggml_backend_buffer_get_base(batch_scratch_.get());
            const auto typed_copy = [&](const RoleSpec & spec, const char * role,
                                        size_t region_base, size_t k) {
                const enum ggml_type copy_type = role_copy_type(spec);
                const size_t copy_count = role_copy_count(spec);
                ggml_tensor * src = ggml_new_tensor_1d(
                    ctx.get(), copy_type == GGML_TYPE_COUNT ? GGML_TYPE_I8 : copy_type,
                    (int64_t) (copy_type == GGML_TYPE_COUNT ? role_bytes(spec) : copy_count));
                const ExpertSlotPool::Loaded loaded = batch.loaded(sel[k]);
                attach_weight(src, loaded.buffer, loaded.base,
                              page0.roles.at(role).device_offset);
                ggml_tensor * dst = ggml_new_tensor_1d(
                    ctx.get(), copy_type == GGML_TYPE_COUNT ? GGML_TYPE_I8 : copy_type,
                    (int64_t) (copy_type == GGML_TYPE_COUNT ? role_bytes(spec) : copy_count));
                attach_weight(dst, batch_scratch_.get(), scratch_base,
                              region_base + k * role_bytes(spec));
                if (copy_type == GGML_TYPE_COUNT) {
                    ggml_backend_tensor_copy(src, dst);
                } else {
                    ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), src, dst));
                }
            };
            for (size_t k = 0; k < n; ++k) {
                typed_copy(gate_spec, "gate", 0, k);
                typed_copy(up_spec, "up", up_base, k);
                typed_copy(down_spec, "down", down_base, k);
            }
            const auto batched_role = [&](const RoleSpec & spec, size_t region_base) {
                ggml_tensor * t = ggml_new_tensor_3d(
                    ctx.get(), spec.type, spec.ne0, spec.ne1, (int64_t) n);
                attach_weight(t, batch_scratch_.get(), scratch_base, region_base);
                return t;
            };
            as_gate = batched_role(gate_spec, 0);
            as_up = batched_role(up_spec, up_base);
            as_down = batched_role(down_spec, down_base);
        }

        // 3) `ids`: mul_mat_id's expert-selection tensor. This call computes
        // EVERY selected expert for EVERY token (the existing dense/masked
        // semantics -- router weight is 0 for tokens not actually routed to
        // an expert, applied in step 6), so ids[e,t] = e for all t: a
        // constant identity mapping, not real per-token routing. n_selected
        // is small (single digits), so building this on the host per request
        // is negligible.
        ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, (int64_t) n, (int64_t) n_tokens);
        ggml_set_input(ids);
        std::vector<int32_t> ids_host((size_t) n * n_tokens);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            for (size_t e = 0; e < n; ++e) {
                ids_host[(size_t) t * n + e] = (int32_t) e;
            }
        }

        // input: [n_embd, 1, n_tokens] -- ne1=1 broadcasts to `n` via
        // mul_mat_id's documented "b can be broadcast to match ids" rule.
        ggml_tensor * input2d = make_io_tensor(ctx.get(), n_tokens, 0);
        ggml_set_input(input2d);
        ggml_tensor * input3d = ggml_reshape_3d(ctx.get(), input2d, n_embd, 1, (int64_t) n_tokens);

        // 4) gate/up: TWO SEPARATE mul_mat_id calls (see header for why this
        // is not merged into one). Each output is a fresh, fully contiguous
        // [n_ff, n, n_tokens] tensor, so the clamp calls below are safe
        // regardless of n_tokens.
        ggml_tensor * gate_out = ggml_mul_mat_id(ctx.get(), as_gate, input3d, ids);
        ggml_tensor * up_out   = ggml_mul_mat_id(ctx.get(), as_up,   input3d, ids);

        // *** SwiGLU CLAMP -- byte-for-byte the same bounds/order as the
        // per-expert loop (see the comment there): up symmetric, gate
        // above-only. ***
        const float swiglu_limit = request.swiglu_clamp;
        if (swiglu_limit > 1e-6f) {
            up_out   = ggml_clamp(ctx.get(), up_out,   -swiglu_limit, swiglu_limit);
            gate_out = ggml_clamp(ctx.get(), gate_out, -INFINITY,     swiglu_limit);
        }
        ggml_tensor * hidden = ggml_swiglu_split(ctx.get(), gate_out, up_out);   // [n_ff, n, n_tokens]

        // 5) down: hidden already has one row per (expert, token) (b->ne[1]
        // == n == ids->ne[0]), so this call is NOT a broadcast -- each
        // token's expert-e row multiplies as_down's expert-e slice.
        ggml_tensor * down_out = ggml_mul_mat_id(ctx.get(), as_down, hidden, ids);   // [n_embd, n, n_tokens]

        // 6) Router weights, ONE ggml_mul for the whole batch (vs one per
        // expert before). Layout matches down_out: ne0=1 (broadcasts over
        // n_embd), ne1=n (expert slot), ne2=n_tokens.
        ggml_tensor * route_w =
            ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, (int64_t) n, (int64_t) n_tokens);
        ggml_set_input(route_w);
        std::vector<float> route_w_host((size_t) n * n_tokens);
        for (uint32_t t = 0; t < n_tokens; ++t) {
            for (size_t k = 0; k < n; ++k) {
                route_w_host[(size_t) t * n + k] = request.assignments[sel[k]].weights[t];
            }
        }
        ggml_tensor * weighted = ggml_mul(ctx.get(), down_out, route_w);   // [n_embd, n, n_tokens]

        // 7) Fold over experts in ASSIGNMENT-INDEX order (sel[] is already in
        // that order), seeding with the running total exactly like the
        // per-expert loop's "SEED THE FOLD, DO NOT ADD AT THE END" -- see
        // that comment for why the association matters (draft-acceptance
        // sensitivity). Per-expert views + sequential ggml_add is the exact
        // idiom src/llama-graph.cpp build_moe_ffn uses for its own expert
        // aggregation; ggml-cuda's ADD is a strided multi-dim kernel
        // (binbcast.cu), unlike CLAMP, so this is stride-safe for any
        // n_tokens.
        ggml_tensor * result = make_io_tensor(ctx.get(), n_tokens, io_result_offset_);
        ggml_tensor * sum = add_previous ? result : nullptr;
        const std::chrono::steady_clock::time_point fold_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        for (size_t k = 0; k < n; ++k) {
            ggml_tensor * contrib = ggml_view_2d(
                ctx.get(), weighted, n_embd, (int64_t) n_tokens,
                weighted->nb[2], (size_t) k * weighted->nb[1]);
            sum = sum ? ggml_add(ctx.get(), sum, contrib) : contrib;
        }
        if (measure_vk) {
            request_stats.ns_vk_fold +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - fold_started).count();
        }
        if (sum == nullptr) {
            throw std::runtime_error("grouped expert compute produced no contribution");
        }
        ggml_tensor * copy = ggml_cpy(ctx.get(), sum, result);
        ggml_build_forward_expand(graph, copy);

        ggml_gallocr_t galloc = compute_galloc_.get();
        const size_t old_compute_size = ggml_gallocr_get_buffer_size(galloc, 0);
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            throw std::runtime_error("failed to allocate grouped expert compute graph");
        }
        if (ggml_gallocr_get_buffer_size(galloc, 0) > old_compute_size) {
            ++request_stats.n_device_allocs;
        }
        request_stats.ns_graph_build +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - build_started).count();

        uint64_t params_elapsed = 0;
        if (submit_async_) {
            AsyncSubmitState & state = async_submit_state();
            state.ids.emplace_back(std::move(ids_host));
            state.route_weights.emplace_back(std::move(route_w_host));
            state.pending = true;
            const auto params_started = std::chrono::steady_clock::now();
            ggml_backend_tensor_set_async(
                backend_.get(), ids, state.ids.back().data(), 0,
                state.ids.back().size() * sizeof(int32_t));
            ggml_backend_tensor_set_async(
                backend_.get(), route_w, state.route_weights.back().data(), 0,
                state.route_weights.back().size() * sizeof(float));
            params_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - params_started).count();
        } else {
            const auto params_started = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
            ggml_backend_tensor_set(route_w, route_w_host.data(), 0, route_w_host.size() * sizeof(float));
            params_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - params_started).count();
        }
        request_stats.ns_params_set += params_elapsed;
        if (measure_vk) {
            request_stats.ns_vk_params_set += params_elapsed;
        }

        const enum ggml_status status = submit_graph(graph, request_stats);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("grouped expert backend graph compute failed");
        }
        for (size_t k = 0; k < n; ++k) {
            const auto & wv = request.assignments[sel[k]].weights;
            uint64_t nz = 0;
            for (float f : wv) { nz += (f != 0.0f); }
            request_stats.n_weight_nonzero += nz;
            request_stats.n_weight_total   += wv.size();
        }
        grouped_log.success = true;
    }

    struct ArenaGroup {
        size_t arena_index = 0;
        std::vector<size_t> assignments;
    };

    struct ArenaRoleKey {
        enum ggml_type type = GGML_TYPE_COUNT;
        int64_t        ne0 = 0;
        int64_t        ne1 = 0;
        uint64_t       offset = 0;

        bool operator==(const ArenaRoleKey & o) const {
            return type == o.type && ne0 == o.ne0 && ne1 == o.ne1 &&
                   offset == o.offset;
        }
    };

    bool compute_batch_arena_multi(
            const pipe_expert_dispatch_req & request,
            const ExpertSlotPool::Batch & batch,
            RequestStats & request_stats,
            const ExpertSlotPool::ArenaLayout & layout,
            const std::array<ArenaRoleKey, 3> & roles,
            const std::vector<ArenaGroup> & groups) {
        const bool measure_vk = stats_.enabled() && is_vulkan_backend();
        const size_t n = request.assignments.size();
        const size_t params_align = ggml_backend_buft_get_alignment(
            ggml_backend_get_default_buffer_type(backend_.get()));
        if (params_align == 0) {
            throw std::runtime_error("invalid arena parameter alignment");
        }
        std::vector<size_t> ids_offsets;
        std::vector<size_t> route_offsets;
        ids_offsets.reserve(groups.size());
        route_offsets.reserve(groups.size());
        size_t ids_bytes = 0;
        for (const ArenaGroup & group : groups) {
            ids_offsets.push_back(ids_bytes);
            ids_bytes += group.assignments.size() * (size_t) request.n_tokens *
                         sizeof(int32_t);
        }
        const size_t route_offset = GGML_PAD(ids_bytes, params_align);
        size_t route_bytes = 0;
        for (const ArenaGroup & group : groups) {
            route_offsets.push_back(route_offset + route_bytes);
            route_bytes += group.assignments.size() * (size_t) request.n_tokens *
                           sizeof(float);
        }
        const size_t params_span = route_offset + route_bytes;
        grow_params_buffer(params_span, request_stats);

        ArenaGraphKey key{request.n_tokens, (uint32_t) n};
        for (const ArenaGroup & group : groups) {
            key.group_arenas.push_back((uint32_t) group.arena_index);
            key.group_sizes.push_back((uint32_t) group.assignments.size());
        }
        uint32_t clamp_bits = 0;
        std::memcpy(&clamp_bits, &request.swiglu_clamp, sizeof(clamp_bits));
        static const size_t cache_max = [] {
            const char * e = std::getenv("WP_EXPERT_GRAPH_CACHE_MAX");
            const long v = (e != nullptr && e[0] != '\0') ? std::strtol(e, nullptr, 10) : 0;
            const size_t requested = v > 0 ? (size_t) v : (size_t) 16;
            return wp_persistent_graphs_enabled()
                ? std::min(requested, (size_t) 2) : requested;
        }();

        const std::chrono::steady_clock::time_point vk_cache_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        auto it = arena_graph_cache_.find(key);
        if (it != arena_graph_cache_.end() &&
                (it->second.graph == nullptr ||
                 it->second.io_gen != io_gen_ ||
                 it->second.params_gen != params_gen_ ||
                 (wp_persistent_graphs_enabled() &&
                  it->second.io_buffer != (io_active_ != nullptr ? io_active_ : io_buffer_.get())) ||
                 it->second.clamp_bits != clamp_bits ||
                 it->second.roles != roles)) {
            if (it->second.persistent_plan != nullptr) {
                release_persistent_plan(it->second.persistent_plan, &request_stats);
            }
            arena_graph_cache_.erase(it);
            it = arena_graph_cache_.end();
        }
        const bool hit = it != arena_graph_cache_.end();
        if (measure_vk) {
            request_stats.ns_vk_cache_lookup +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - vk_cache_started).count();
        }
        if (!hit) {
            if (arena_graph_cache_.size() >= cache_max) {
                auto victim = arena_graph_cache_.begin();
                for (auto j = arena_graph_cache_.begin(); j != arena_graph_cache_.end(); ++j) {
                    if (j->second.last_used < victim->second.last_used) {
                        victim = j;
                    }
                }
                if (victim->second.persistent_plan != nullptr) {
                    release_persistent_plan(victim->second.persistent_plan, &request_stats);
                }
                arena_graph_cache_.erase(victim);
            }
            it = arena_graph_cache_.emplace(key, ArenaGraphEntry{}).first;
            ArenaGraphEntry & entry = it->second;
            entry.roles = roles;
            entry.clamp_bits = clamp_bits;
            entry.io_gen = io_gen_;
            entry.params_gen = params_gen_;
            if (wp_persistent_graphs_enabled()) {
                entry.io_buffer = io_active_ != nullptr ? io_active_ : io_buffer_.get();
            }
            ++request_stats.n_arena_build;

            const auto build_started = std::chrono::steady_clock::now();
            // Group terms cover role views, IDs, routing weights, the three
            // matmuls, clamps, and the per-group intermediate tensors. The n
            // terms cover assignment views and the left fold. Fixed headroom
            // covers shared inputs, result, copy, and graph leaves.
            const size_t tensor_count = 12 * groups.size() + 4 * n + 64;
            const size_t graph_nodes = 8 * groups.size() + 4 * n + 32;
            entry.ctx.reset(ggml_init({
                /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                                  ggml_graph_overhead_custom(graph_nodes, false),
                /* .mem_base = */ nullptr,
                /* .no_alloc = */ true,
            }));
            if (!entry.ctx) {
                throw std::runtime_error("failed to allocate multi-arena graph metadata");
            }
            ggml_context * ctx = entry.ctx.get();
            const auto make_role = [&](const ArenaGroup & group,
                                       const ArenaRoleKey & role) {
                const ExpertSlotPool::ArenaLayout::Arena & arena =
                    layout.arenas[group.arena_index];
                ggml_tensor * tensor = ggml_new_tensor_3d(
                    ctx, role.type, role.ne0, role.ne1, (int64_t) arena.n_slots);
                tensor->nb[2] = arena.stride != 0 ? arena.stride : layout.slot_stride;
                attach_weight(tensor, arena.buffer, arena.base, role.offset);
                return tensor;
            };
            const int64_t n_embd = catalog_.descriptor.hparams.n_embd;
            ggml_tensor * input2d = make_io_tensor(ctx, request.n_tokens, 0);
            ggml_set_input(input2d);
            ggml_tensor * input3d = ggml_reshape_3d(
                ctx, input2d, n_embd, 1, request.n_tokens);
            std::vector<ggml_tensor *> contributions(n, nullptr);
            for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
                const ArenaGroup & group = groups[group_index];
                const size_t group_size = group.assignments.size();
                ggml_tensor * as_gate = make_role(group, roles[0]);
                ggml_tensor * as_up   = make_role(group, roles[1]);
                ggml_tensor * as_down = make_role(group, roles[2]);
                ggml_tensor * ids = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_I32, (int64_t) group_size, request.n_tokens);
                ggml_set_input(ids);
                attach_weight(ids, params_buffer_.get(),
                              ggml_backend_buffer_get_base(params_buffer_.get()),
                              ids_offsets[group_index]);
                ggml_tensor * route_w = ggml_new_tensor_3d(
                    ctx, GGML_TYPE_F32, 1, (int64_t) group_size, request.n_tokens);
                ggml_set_input(route_w);
                attach_weight(route_w, params_buffer_.get(),
                              ggml_backend_buffer_get_base(params_buffer_.get()),
                              route_offsets[group_index]);
                ggml_tensor * gate_out = ggml_mul_mat_id(ctx, as_gate, input3d, ids);
                ggml_tensor * up_out   = ggml_mul_mat_id(ctx, as_up, input3d, ids);
                if (request.swiglu_clamp > 1e-6f) {
                    up_out = ggml_clamp(
                        ctx, up_out, -request.swiglu_clamp, request.swiglu_clamp);
                    gate_out = ggml_clamp(
                        ctx, gate_out, -INFINITY, request.swiglu_clamp);
                }
                ggml_tensor * hidden = ggml_swiglu_split(ctx, gate_out, up_out);
                ggml_tensor * down_out = ggml_mul_mat_id(ctx, as_down, hidden, ids);
                ggml_tensor * weighted = ggml_mul(ctx, down_out, route_w);
                for (size_t local = 0; local < group_size; ++local) {
                    const size_t assignment = group.assignments[local];
                    contributions[assignment] = ggml_view_2d(
                        ctx, weighted, n_embd, request.n_tokens,
                        weighted->nb[2], local * weighted->nb[1]);
                }
            }
            ggml_tensor * sum = nullptr;
            const std::chrono::steady_clock::time_point fold_started =
                measure_vk ? std::chrono::steady_clock::now() :
                              std::chrono::steady_clock::time_point();
            for (ggml_tensor * contribution : contributions) {
                sum = sum != nullptr ? ggml_add(ctx, sum, contribution) : contribution;
            }
            if (measure_vk) {
                request_stats.ns_vk_fold +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - fold_started).count();
            }
            if (sum == nullptr) {
                throw std::runtime_error("multi-arena graph produced no contribution");
            }
            ggml_tensor * result = make_io_tensor(ctx, request.n_tokens, io_result_offset_);
            ggml_tensor * copy = ggml_cpy(ctx, sum, result);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
            ggml_build_forward_expand(graph, copy);
            entry.galloc.reset(ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_.get())));
            if (!entry.galloc || !ggml_gallocr_alloc_graph(entry.galloc.get(), graph)) {
                throw std::runtime_error("failed to allocate multi-arena expert graph");
            }
            if (ggml_gallocr_get_buffer_size(entry.galloc.get(), 0) > 0) {
                ++request_stats.n_device_allocs;
            }
            entry.blob = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) params_span);
            attach_weight(entry.blob, params_buffer_.get(),
                          ggml_backend_buffer_get_base(params_buffer_.get()), 0);
            entry.persistent_plan = create_persistent_plan(graph);
            entry.graph = graph;
            request_stats.ns_graph_build +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - build_started).count();
        }

        ArenaGraphEntry & entry = it->second;
        entry.last_used = ++arena_graph_cache_tick_;
        std::vector<uint8_t> params_host(params_span, 0);
        for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
            const ArenaGroup & group = groups[group_index];
            const ExpertSlotPool::ArenaLayout::Arena & arena =
                layout.arenas[group.arena_index];
            const size_t group_size = group.assignments.size();
            for (uint32_t t = 0; t < request.n_tokens; ++t) {
                for (size_t local = 0; local < group_size; ++local) {
                    const size_t assignment = group.assignments[local];
                    const size_t slot = batch.slot_index(assignment);
                    const int32_t local_slot = (int32_t) (slot - arena.first_slot);
                    const size_t index = (size_t) t * group_size + local;
                    std::memcpy(params_host.data() + ids_offsets[group_index] +
                                    index * sizeof(int32_t),
                                &local_slot, sizeof(local_slot));
                    const float weight = request.assignments[assignment].weights[t];
                    std::memcpy(params_host.data() + route_offsets[group_index] +
                                    index * sizeof(float),
                                &weight, sizeof(weight));
                }
            }
        }
        const auto params_started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(entry.blob, params_host.data(), 0, params_span);
        const uint64_t params_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - params_started).count();
        request_stats.ns_params_set += params_elapsed;
        if (measure_vk) {
            request_stats.ns_vk_params_set += params_elapsed;
        }
        enum ggml_status status = submit_graph(
            entry.graph, request_stats, entry.persistent_plan);
        if (status != GGML_STATUS_SUCCESS && entry.persistent_plan != nullptr) {
            release_persistent_plan(entry.persistent_plan, &request_stats);
            std::fprintf(stderr,
                         "WARN wp expert worker: persistent Vulkan graph replay failed; "
                         "retrying normal graph compute\n");
            status = submit_graph(entry.graph, request_stats);
        }
        if (status != GGML_STATUS_SUCCESS) {
            if (entry.persistent_plan != nullptr) {
                release_persistent_plan(entry.persistent_plan, &request_stats);
            }
            entry.graph = nullptr;
            throw std::runtime_error("multi-arena expert backend graph compute failed");
        }
        ++request_stats.n_arena_hit;
        request_stats.n_arena_groups += groups.size();
        for (const pipe_expert_assignment & assignment : request.assignments) {
            for (float weight : assignment.weights) {
                request_stats.n_weight_nonzero += weight != 0.0f;
                ++request_stats.n_weight_total;
            }
        }
        return true;
    }

    // WP_EXPERT_ARENA_ID=1 uses persistent grouped dispatch over the slot arena.
    bool compute_batch_arena(
            const pipe_expert_dispatch_req & request,
            const std::vector<const ExpertPage *> & pages,
            const ExpertSlotPool::Batch & batch,
            RequestStats & request_stats) {
        const bool measure_vk = stats_.enabled() && is_vulkan_backend();
        const std::optional<ExpertSlotPool::ArenaLayout> & layout_opt = pool_.arena_layout();
        if (!layout_opt.has_value()) {
            return false;
        }
        const ExpertSlotPool::ArenaLayout & layout = *layout_opt;
        const size_t n = request.assignments.size();
        if (n == 0 || n > (size_t) INT32_MAX || layout.n_slots > (size_t) INT32_MAX) {
            return false;
        }

        static const char * k_roles[3] = {"gate", "up", "down"};
        const auto & specs = catalog_.descriptor.layers.at(request.layer);
        std::array<ArenaRoleKey, 3> roles;
        for (size_t j = 0; j < roles.size(); ++j) {
            const RoleSpec & spec = specs.at(k_roles[j]);
            roles[j] = {spec.type, spec.ne0, spec.ne1,
                        pages[0]->roles.at(k_roles[j]).device_offset};
            const size_t type_size = ggml_type_size(spec.type);
            const size_t block_size = ggml_blck_size(spec.type);
            if (type_size == 0 || block_size == 0) {
                return false;
            }
        }
        for (size_t i = 0; i < pages.size(); ++i) {
            const ExpertSlotPool::ArenaLayout::Arena * arena =
                layout.arena_for_slot(batch.slot_index(i));
            const uint64_t stride = arena != nullptr && arena->stride != 0
                ? arena->stride : layout.slot_stride;
            if (stride == 0 || pages[i]->device_size > stride) {
                return false;
            }
            for (size_t j = 0; j < roles.size(); ++j) {
                const size_t type_size = ggml_type_size(roles[j].type);
                const size_t block_size = ggml_blck_size(roles[j].type);
                if (type_size == 0 || block_size == 0 ||
                        stride % type_size != 0 ||
                        stride / type_size > UINT32_MAX / block_size) {
                    return false;
                }
            }
            for (size_t j = 0; j < roles.size(); ++j) {
                if (pages[i]->roles.at(k_roles[j]).device_offset != roles[j].offset) {
                    return false;
                }
            }
            if (batch.slot_index(i) >= layout.n_slots) {
                return false;
            }
        }

        std::vector<ArenaGroup> groups;
        for (size_t arena_index = 0; arena_index < layout.arenas.size(); ++arena_index) {
            ArenaGroup group;
            group.arena_index = arena_index;
            for (size_t i = 0; i < n; ++i) {
                const ExpertSlotPool::ArenaLayout::Arena * arena =
                    layout.arena_for_slot(batch.slot_index(i));
                if (arena == nullptr) {
                    return false;
                }
                const size_t owner = (size_t) (arena - layout.arenas.data());
                if (owner == arena_index) {
                    group.assignments.push_back(i);
                }
            }
            if (!group.assignments.empty()) {
                groups.push_back(std::move(group));
            }
        }
        if (groups.empty()) {
            return false;
        }

        // Keep the one-arena graph below unchanged. Multi-arena requests use a
        // separate graph because each mul_mat_id needs a different buffer base.
        // Route by LAYOUT shape, not group count: a single-group request on a
        // multi-arena layout must still use its group's arena base — the
        // legacy builder below attaches via layout.buffer/base, which are
        // null when arenas.size() > 1 (live crash: ggml-backend.cpp:123).
        if (layout.arenas.size() > 1) {
            return compute_batch_arena_multi(
                request, batch, request_stats, layout, roles, groups);
        }

        const size_t params_align = ggml_backend_buft_get_alignment(
            ggml_backend_get_default_buffer_type(backend_.get()));
        if (params_align == 0) {
            throw std::runtime_error("invalid arena parameter alignment");
        }
        const size_t ids_bytes = n * (size_t) request.n_tokens * sizeof(int32_t);
        const size_t route_offset = GGML_PAD(ids_bytes, params_align);
        const size_t route_bytes = n * (size_t) request.n_tokens * sizeof(float);
        const size_t params_span = route_offset + route_bytes;
        grow_params_buffer(params_span, request_stats);

        ArenaGraphKey key{request.n_tokens, (uint32_t) n};
        uint32_t clamp_bits = 0;
        std::memcpy(&clamp_bits, &request.swiglu_clamp, sizeof(clamp_bits));
        static const size_t cache_max = [] {
            const char * e = std::getenv("WP_EXPERT_GRAPH_CACHE_MAX");
            const long v = (e != nullptr && e[0] != '\0') ? std::strtol(e, nullptr, 10) : 0;
            const size_t requested = v > 0 ? (size_t) v : (size_t) 16;
            return wp_persistent_graphs_enabled()
                ? std::min(requested, (size_t) 2) : requested;
        }();
        // WP_ARENA_FOLD_COLLAPSE / WP_ARENA_HIP_GRAPH -- single-arena bucket
        // only (the multi-arena path above never reaches here). fold_collapse
        // reduces the per-expert fold with one ggml_sum_rows instead of
        // (n-1) ggml_add nodes; hip_graph_replay additionally requires
        // fold_collapse and opts this bucket's graph into HIP graph
        // capture/replay (see the capture-then-replay state machine below).
        static const bool fold_collapse_env = [] {
            const char * e = std::getenv("WP_ARENA_FOLD_COLLAPSE");
            return e != nullptr && e[0] == '1';
        }();
        const bool fold_collapse = !wp_persistent_graphs_enabled() && fold_collapse_env;
        static const bool hip_graph_replay = [] {
            if (wp_persistent_graphs_enabled()) {
                return false;
            }
            const char * fold = std::getenv("WP_ARENA_FOLD_COLLAPSE");
            const char * graph = std::getenv("WP_ARENA_HIP_GRAPH");
            return fold != nullptr && fold[0] == '1' &&
                   graph != nullptr && graph[0] == '1';
        }();

        const std::chrono::steady_clock::time_point vk_cache_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        auto it = arena_graph_cache_.find(key);
        if (it != arena_graph_cache_.end() &&
                (it->second.graph == nullptr ||
                 it->second.io_gen != io_gen_ ||
                 it->second.params_gen != params_gen_ ||
                 (wp_persistent_graphs_enabled() &&
                  it->second.io_buffer != (io_active_ != nullptr ? io_active_ : io_buffer_.get())) ||
                 it->second.clamp_bits != clamp_bits ||
                 it->second.roles != roles)) {
            if (it->second.persistent_plan != nullptr) {
                release_persistent_plan(it->second.persistent_plan, &request_stats);
            }
            arena_graph_cache_.erase(it);
            it = arena_graph_cache_.end();
        }
        const bool hit = it != arena_graph_cache_.end();
        if (measure_vk) {
            request_stats.ns_vk_cache_lookup +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - vk_cache_started).count();
        }
        if (!hit) {
            if (arena_graph_cache_.size() >= cache_max) {
                auto victim = arena_graph_cache_.begin();
                for (auto j = arena_graph_cache_.begin(); j != arena_graph_cache_.end(); ++j) {
                    if (j->second.last_used < victim->second.last_used) {
                        victim = j;
                    }
                }
                if (victim->second.persistent_plan != nullptr) {
                    release_persistent_plan(victim->second.persistent_plan, &request_stats);
                }
                arena_graph_cache_.erase(victim);
            }
            it = arena_graph_cache_.emplace(key, ArenaGraphEntry{}).first;
            ArenaGraphEntry & entry = it->second;
            entry.roles = roles;
            entry.clamp_bits = clamp_bits;
            entry.io_gen = io_gen_;
            entry.params_gen = params_gen_;
            if (wp_persistent_graphs_enabled()) {
                entry.io_buffer = io_active_ != nullptr ? io_active_ : io_buffer_.get();
            }
            ++request_stats.n_arena_build;

            const auto build_started = std::chrono::steady_clock::now();
            const size_t tensor_count = 8 * n + 64;
            const size_t graph_nodes = 4 * n + 32;
            entry.ctx.reset(ggml_init({
                /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                                  ggml_graph_overhead_custom(graph_nodes, false),
                /* .mem_base = */ nullptr,
                /* .no_alloc = */ true,
            }));
            if (!entry.ctx) {
                throw std::runtime_error("failed to allocate arena graph metadata");
            }
            ggml_context * ctx = entry.ctx.get();

            const auto make_role = [&](const ArenaRoleKey & role) {
                const uint64_t stride = !layout.arenas.empty() && layout.arenas[0].stride != 0
                    ? layout.arenas[0].stride : layout.slot_stride;
                ggml_tensor * tensor = ggml_new_tensor_3d(
                    ctx, role.type, role.ne0, role.ne1, (int64_t) layout.n_slots);
                tensor->nb[2] = stride;
                attach_weight(tensor, layout.buffer, layout.base, role.offset);
                return tensor;
            };
            ggml_tensor * as_gate = make_role(roles[0]);
            ggml_tensor * as_up   = make_role(roles[1]);
            ggml_tensor * as_down = make_role(roles[2]);

            ggml_tensor * ids = ggml_new_tensor_2d(
                ctx, GGML_TYPE_I32, (int64_t) n, request.n_tokens);
            ggml_set_input(ids);
            attach_weight(ids, params_buffer_.get(),
                          ggml_backend_buffer_get_base(params_buffer_.get()), 0);
            ggml_tensor * route_w = ggml_new_tensor_3d(
                ctx, GGML_TYPE_F32, 1, (int64_t) n, request.n_tokens);
            ggml_set_input(route_w);
            attach_weight(route_w, params_buffer_.get(),
                          ggml_backend_buffer_get_base(params_buffer_.get()), route_offset);

            const int64_t n_embd = catalog_.descriptor.hparams.n_embd;
            ggml_tensor * input2d = make_io_tensor(ctx, request.n_tokens, 0);
            ggml_set_input(input2d);
            ggml_tensor * input3d = ggml_reshape_3d(
                ctx, input2d, n_embd, 1, request.n_tokens);
            ggml_tensor * gate_out = ggml_mul_mat_id(ctx, as_gate, input3d, ids);
            ggml_tensor * up_out   = ggml_mul_mat_id(ctx, as_up, input3d, ids);
            if (request.swiglu_clamp > 1e-6f) {
                up_out = ggml_clamp(
                    ctx, up_out, -request.swiglu_clamp, request.swiglu_clamp);
                gate_out = ggml_clamp(
                    ctx, gate_out, -INFINITY, request.swiglu_clamp);
            }
            ggml_tensor * hidden = ggml_swiglu_split(ctx, gate_out, up_out);
            ggml_tensor * down_out = ggml_mul_mat_id(ctx, as_down, hidden, ids);
            ggml_tensor * weighted = ggml_mul(ctx, down_out, route_w);

            ggml_tensor * sum = nullptr;
            if (fold_collapse) {
                const std::chrono::steady_clock::time_point fold_started =
                    measure_vk ? std::chrono::steady_clock::now() :
                                  std::chrono::steady_clock::time_point();
                // mul_mat_id lays out [n_embd, n_assignments, n_tokens]. Make
                // the assignment axis contiguous and reduce it in one op.
                ggml_tensor * rows = ggml_cont(
                    ctx, ggml_permute(ctx, weighted, 1, 0, 2, 3));
                sum = ggml_reshape_2d(
                    ctx, ggml_sum_rows(ctx, rows), n_embd, request.n_tokens);
                if (measure_vk) {
                    request_stats.ns_vk_fold +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - fold_started).count();
                }
            } else {
                const std::chrono::steady_clock::time_point fold_started =
                    measure_vk ? std::chrono::steady_clock::now() :
                                  std::chrono::steady_clock::time_point();
                for (size_t k = 0; k < n; ++k) {
                    ggml_tensor * contrib = ggml_view_2d(
                        ctx, weighted, n_embd, request.n_tokens,
                        weighted->nb[2], k * weighted->nb[1]);
                    sum = sum != nullptr ? ggml_add(ctx, sum, contrib) : contrib;
                }
                if (measure_vk) {
                    request_stats.ns_vk_fold +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - fold_started).count();
                }
            }
            ggml_tensor * result = make_io_tensor(ctx, request.n_tokens, io_result_offset_);
            ggml_tensor * copy = ggml_cpy(ctx, sum, result);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_nodes, false);
            ggml_build_forward_expand(graph, copy);

            entry.galloc.reset(ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_.get())));
            if (!entry.galloc || !ggml_gallocr_alloc_graph(entry.galloc.get(), graph)) {
                throw std::runtime_error("failed to allocate arena expert graph");
            }
            if (ggml_gallocr_get_buffer_size(entry.galloc.get(), 0) > 0) {
                ++request_stats.n_device_allocs;
            }
            entry.blob = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) params_span);
            attach_weight(entry.blob, params_buffer_.get(),
                          ggml_backend_buffer_get_base(params_buffer_.get()), 0);
            entry.persistent_plan = create_persistent_plan(graph);
            entry.graph = graph;
            request_stats.ns_graph_build +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - build_started).count();
        }

        ArenaGraphEntry & entry = it->second;
        entry.last_used = ++arena_graph_cache_tick_;
        std::vector<uint8_t> params_host(params_span, 0);
        std::vector<int32_t> ids_host(n * request.n_tokens);
        std::vector<float> route_host(n * request.n_tokens);
        for (uint32_t t = 0; t < request.n_tokens; ++t) {
            for (size_t k = 0; k < n; ++k) {
                ids_host[(size_t) t * n + k] = (int32_t) batch.slot_index(k);
                route_host[(size_t) t * n + k] = request.assignments[k].weights[t];
            }
        }
        std::memcpy(params_host.data(), ids_host.data(), ids_bytes);
        std::memcpy(params_host.data() + route_offset, route_host.data(), route_bytes);
        const auto params_started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(entry.blob, params_host.data(), 0, params_span);
        const uint64_t params_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - params_started).count();
        request_stats.ns_params_set += params_elapsed;
        if (measure_vk) {
            request_stats.ns_vk_params_set += params_elapsed;
        }

        enum ggml_status status;
        if (entry.persistent_plan != nullptr) {
            status = submit_graph(entry.graph, request_stats, entry.persistent_plan);
        } else {
            // *** ONLY TOUCH THE ENVIRONMENT WHEN HIP GRAPH REPLAY IS ARMED. ***
            // The eager path remains unchanged when the persistent flag is off.
            const bool hip_env_active = hip_graph_replay;
            const char * const saved_hip_graphs =
                hip_env_active ? std::getenv("WP_HIP_GRAPHS") : nullptr;
            const bool saved_hip_graphs_set = saved_hip_graphs != nullptr;
            const std::string saved_hip_graphs_value =
                saved_hip_graphs_set ? saved_hip_graphs : "";
            const char * const saved_disable_graphs =
                hip_env_active ? std::getenv("GGML_CUDA_DISABLE_GRAPHS") : nullptr;
            const bool saved_disable_graphs_set = saved_disable_graphs != nullptr;
            const std::string saved_disable_graphs_value =
                saved_disable_graphs_set ? saved_disable_graphs : "";
            const bool hip_graph_attempt = hip_graph_replay &&
                !entry.hip_graph_failed && !saved_disable_graphs_set;
            const auto restore_graph_env = [&]() {
                if (!hip_env_active) {
                    return;
                }
                if (saved_hip_graphs_set) {
                    setenv("WP_HIP_GRAPHS", saved_hip_graphs_value.c_str(), 1);
                } else {
                    unsetenv("WP_HIP_GRAPHS");
                }
                if (saved_disable_graphs_set) {
                    setenv("GGML_CUDA_DISABLE_GRAPHS", saved_disable_graphs_value.c_str(), 1);
                } else {
                    unsetenv("GGML_CUDA_DISABLE_GRAPHS");
                }
            };
            const auto graph_compute = [&](bool use_hip_graph) {
                if (hip_graph_replay) {
                    if (use_hip_graph) {
                        setenv("WP_HIP_GRAPHS", "1", 1);
                    } else {
                        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
                    }
                }
                const auto result = ggml_backend_graph_compute(backend_.get(), entry.graph);
                restore_graph_env();
                return result;
            };
            const auto submit_started = std::chrono::steady_clock::now();
            status = graph_compute(hip_graph_attempt);
            const uint64_t submit_elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - submit_started).count();
            request_stats.ns_submit += submit_elapsed;
            if (measure_vk) {
                request_stats.ns_vk_graph_compute += submit_elapsed;
            }
            ++request_stats.n_graph_submits;
            if (status != GGML_STATUS_SUCCESS && hip_graph_attempt) {
                // Graph capture is an optional optimization. Retry this bucket
                // eagerly and keep it on the eager path for the process lifetime.
                entry.hip_graph_failed = true;
                entry.hip_graph_submits = 0;
                std::fprintf(stderr,
                             "WARN wp expert worker: HIP graph disabled for arena bucket "
                             "tokens=%u assignments=%u clamp_bits=%u\n",
                             request.n_tokens, (unsigned) n, clamp_bits);
                const auto fallback_started = std::chrono::steady_clock::now();
                status = graph_compute(false);
                const uint64_t fallback_elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - fallback_started).count();
                request_stats.ns_submit += fallback_elapsed;
                if (measure_vk) {
                    request_stats.ns_vk_graph_compute += fallback_elapsed;
                }
                ++request_stats.n_graph_submits;
            }
            if (status == GGML_STATUS_SUCCESS && hip_graph_attempt) {
                if (entry.hip_graph_submits == 0) {
                    entry.hip_graph_submits = 1;
                } else if (entry.hip_graph_submits == 1) {
                    ++request_stats.n_hipgraph_capture;
                    entry.hip_graph_submits = 2;
                } else {
                    ++request_stats.n_hipgraph_replay;
                }
            }
        }
        if (status != GGML_STATUS_SUCCESS && entry.persistent_plan != nullptr) {
            release_persistent_plan(entry.persistent_plan, &request_stats);
            std::fprintf(stderr,
                         "WARN wp expert worker: persistent Vulkan graph replay failed; "
                         "retrying normal graph compute\n");
            status = submit_graph(entry.graph, request_stats);
        }
        if (status != GGML_STATUS_SUCCESS) {
            if (entry.persistent_plan != nullptr) {
                release_persistent_plan(entry.persistent_plan, &request_stats);
            }
            entry.graph = nullptr;
            throw std::runtime_error("arena expert backend graph compute failed");
        }
        ++request_stats.n_arena_hit;
        request_stats.n_arena_groups += groups.size();
        for (const pipe_expert_assignment & assignment : request.assignments) {
            for (float weight : assignment.weights) {
                request_stats.n_weight_nonzero += weight != 0.0f;
                ++request_stats.n_weight_total;
            }
        }
        return true;
    }

    void read_result(std::vector<float> & result, RequestStats & request_stats) {
        synchronize_async(&request_stats);
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert result metadata");
        }
        const uint32_t n_tokens = (uint32_t) (
            result.size() / (size_t) catalog_.descriptor.hparams.n_embd);
        ggml_tensor * output = make_io_tensor(ctx.get(), n_tokens, io_result_offset_);
        const auto readback_started = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(
            output, result.data(), 0, result.size() * sizeof(float));
        const uint64_t readback_elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - readback_started).count();
        request_stats.ns_readback += readback_elapsed;
        if (stats_.enabled() && is_vulkan_backend()) {
            request_stats.ns_vk_readback += readback_elapsed;
        }
    }

    void compute_cpu_on_arrival(
            const pipe_expert_dispatch_req & request,
            const ExpertPage & page,
            ExpertSlotPool::Batch & batch,
            size_t assignment_index,
            std::vector<float> & result,
            RequestStats & request_stats) {
        if (!cpu_backend_) {
            cpu_backend_.reset(ggml_backend_cpu_init());
            if (!cpu_backend_) {
                throw std::runtime_error("failed to initialize CPU expert backend");
            }
            // tier=false ON PURPOSE. This backend runs on a GPU DeviceWorker's
            // executor thread, and ggml applies the threadpool's priority and
            // affinity to OpenMP thread 0 -- which is the calling thread -- and
            // never restores them. Confining a GPU submission thread here would
            // recreate the very starvation WP_CPU_TIER_OVERLAP exists to avoid.
            configure_cpu_backend(cpu_backend_.get(), /* tier = */ false);
        }

        const auto started = std::chrono::steady_clock::now();
        const auto & specs = catalog_.descriptor.layers.at(page.layer);
        const size_t activation_bytes = request.activations.size() * sizeof(float);
        buffer_ptr input_buffer(ggml_backend_cpu_buffer_from_ptr(
            const_cast<float *>(request.activations.data()), activation_bytes));
        buffer_ptr weight_buffer(ggml_backend_cpu_buffer_from_ptr(
            const_cast<void *>(batch.cpu_staging(assignment_index)), (size_t) page.size));
        if (!input_buffer || !weight_buffer) {
            throw std::runtime_error("failed to wrap CPU expert inputs");
        }

        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate CPU expert graph metadata");
        }
        ggml_tensor * input = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd,
            request.n_tokens);
        attach_weight(input, input_buffer.get(),
                      ggml_backend_buffer_get_base(input_buffer.get()), 0);
        ggml_tensor * gate = ggml_new_tensor_2d(
            ctx.get(), specs.at("gate").type, specs.at("gate").ne0, specs.at("gate").ne1);
        ggml_tensor * up = ggml_new_tensor_2d(
            ctx.get(), specs.at("up").type, specs.at("up").ne0, specs.at("up").ne1);
        ggml_tensor * down = ggml_new_tensor_2d(
            ctx.get(), specs.at("down").type, specs.at("down").ne0, specs.at("down").ne1);
        attach_weight(gate, weight_buffer.get(),
                      ggml_backend_buffer_get_base(weight_buffer.get()),
                      page.roles.at("gate").offset);
        attach_weight(up, weight_buffer.get(),
                      ggml_backend_buffer_get_base(weight_buffer.get()),
                      page.roles.at("up").offset);
        attach_weight(down, weight_buffer.get(),
                      ggml_backend_buffer_get_base(weight_buffer.get()),
                      page.roles.at("down").offset);

        ggml_tensor * gate_x = ggml_mul_mat(ctx.get(), gate, input);
        ggml_tensor * up_x = ggml_mul_mat(ctx.get(), up, input);
        if (request.swiglu_clamp > 1e-6f) {
            up_x = ggml_clamp(ctx.get(), up_x, -request.swiglu_clamp, request.swiglu_clamp);
            gate_x = ggml_clamp(ctx.get(), gate_x, -INFINITY, request.swiglu_clamp);
        }
        ggml_tensor * hidden = ggml_swiglu_split(ctx.get(), gate_x, up_x);
        ggml_tensor * output = ggml_mul_mat(ctx.get(), down, hidden);
        ggml_tensor * route = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_F32, 1, request.n_tokens);
        buffer_ptr route_buffer(ggml_backend_cpu_buffer_from_ptr(
            const_cast<float *>(request.assignments[assignment_index].weights.data()),
            request.assignments[assignment_index].weights.size() * sizeof(float)));
        if (!route_buffer) {
            throw std::runtime_error("failed to wrap CPU routing weights");
        }
        attach_weight(route, route_buffer.get(),
                      ggml_backend_buffer_get_base(route_buffer.get()), 0);
        ggml_tensor * weighted = ggml_mul(ctx.get(), output, route);
        ggml_tensor * result_tensor = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd, request.n_tokens);
        ggml_tensor * copy = ggml_cpy(ctx.get(), weighted, result_tensor);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, copy);
        galloc_ptr galloc(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc.get(), graph)) {
            throw std::runtime_error("failed to allocate CPU expert graph");
        }
        if (ggml_backend_graph_compute(cpu_backend_.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CPU expert graph compute failed");
        }
        result.resize((size_t) request.n_tokens * catalog_.descriptor.hparams.n_embd);
        ggml_backend_tensor_get(
            result_tensor, result.data(), 0, result.size() * sizeof(float));
        batch.release_cpu_staging(assignment_index);
        for (float weight : request.assignments[assignment_index].weights) {
            request_stats.n_weight_nonzero += weight != 0.0f;
            ++request_stats.n_weight_total;
        }
        const uint64_t elapsed = (uint64_t) std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        request_stats.ns_cpu_on_arrival += elapsed;
        g_read_path_stats.record_cpu_on_arrival(elapsed);
    }

    void write_io_partial(const std::vector<float> & partial, size_t offset) {
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert partial metadata");
        }
        ggml_tensor * output = make_io_tensor(
            ctx.get(), (uint32_t) (partial.size() /
                (size_t) catalog_.descriptor.hparams.n_embd), offset);
        ggml_backend_tensor_set(
            output, partial.data(), 0, partial.size() * sizeof(float));
    }

    // *** WP_EXPERT_RESIDENT_FIRST support: per-slot IO layout + the final
    // fold. See the big comment on WP_EXPERT_RESIDENT_FIRST in dispatch() for
    // the overlap argument; these two are just the plumbing. ***

    struct IoSlotLayout {
        size_t result_offset;   // where the canonical (single, shared) result lands
        size_t result_size;     // bytes of one [n_embd, n_tokens] slot
        size_t alignment;
    };

    // Mirrors prepare_io's offset math EXACTLY (input at 0, result padded
    // right after it) without touching io_buffer_ -- called BEFORE prepare_io
    // so the resident-first path can pre-size the buffer for its extra
    // per-assignment slots before anything is uploaded. Keep this in sync
    // with prepare_io if that offset math ever changes.
    IoSlotLayout compute_io_slot_layout(uint32_t n_tokens) const {
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead(),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate expert IO layout metadata");
        }
        ggml_tensor * probe = ggml_new_tensor_2d(
            ctx.get(), GGML_TYPE_F32, catalog_.descriptor.hparams.n_embd, n_tokens);
        const ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(backend_.get());
        const size_t input_size = ggml_backend_buft_get_alloc_size(buft, probe);
        const size_t alignment  = ggml_backend_buft_get_alignment(buft);
        IoSlotLayout layout;
        layout.result_offset = GGML_PAD(input_size, alignment);
        layout.result_size   = input_size;   // input and result are the same [n_embd, n_tokens] shape
        layout.alignment     = alignment;
        return layout;
    }

    // Sums the per-assignment partial buffers written by the
    // WP_EXPERT_RESIDENT_FIRST path -- one ggml_add per partial, IN
    // ASSIGNMENT-INDEX ORDER -- and copies the total into the canonical
    // result slot that read_result() (and every other caller) expects.
    //
    // *** THIS IS THE ENTIRE CORRECTNESS ARGUMENT. *** Each partial i already
    // holds exactly what compute_batch would have contributed for expert i on
    // the ordinary serial path (same op, same inputs -- compute_batch's
    // result_offset override changes WHERE it lands, never the per-expert
    // math). Folding left-to-right over i = 0..n-1 reproduces the exact same
    // association as the serial path's
    // `sum = sum ? ggml_add(sum, weighted) : weighted` loop, so the total is
    // bit-identical to today's default (WP_EXPERT_RESIDENT_FIRST=0) path no
    // matter which experts were hits, which were pageins, or what order the
    // reads happened to land in. Fold in arrival or hit/miss-GROUP order
    // instead (the WP_EXPERT_OVERLAP mistake this path exists to avoid
    // repeating) and the numbers still look plausible but are NOT
    // bit-identical -- see the determinism note above WP_EXPERT_OVERLAP.
    void fold_resident_first_partials(
            uint32_t n_tokens, size_t n_assign, size_t base_offset,
            size_t slot_size, RequestStats & request_stats) {
        const bool measure_vk = stats_.enabled() && is_vulkan_backend();
        // TENSOR BUDGET -- got this wrong once and it aborted the worker on the
        // first real request (2026-08-19: "ggml_new_object: not enough space in
        // the context's memory pool (needed 4048, available 3776)", GGML_ASSERT
        // in ggml_new_tensor_impl). The fold creates MORE than one tensor per
        // assignment: n_assign part tensors, n_assign-1 ggml_add results, the
        // result tensor and the ggml_cpy -- about 2*n_assign+2 -- and
        // make_io_tensor may add a view per part on top of that. This is
        // METADATA ONLY (no_alloc = true), so headroom costs a few hundred
        // bytes and a wrong estimate costs a crash. Be generous.
        const size_t tensor_count = 4 * n_assign + 16;
        const size_t graph_nodes  = 2 * n_assign + 8;
        const ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * tensor_count +
                              ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ nullptr,
            /* .no_alloc = */ true,
        };
        context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to allocate resident-first fold metadata");
        }
        // Left-fold in assignment-index order -- see the correctness note above.
        ggml_tensor * fold = nullptr;
        const std::chrono::steady_clock::time_point fold_started =
            measure_vk ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
        for (size_t i = 0; i < n_assign; ++i) {
            ggml_tensor * part = make_io_tensor(ctx.get(), n_tokens, base_offset + i * slot_size);
            fold = fold ? ggml_add(ctx.get(), fold, part) : part;
        }
        if (measure_vk) {
            request_stats.ns_vk_fold +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - fold_started).count();
        }
        ggml_tensor * result = make_io_tensor(ctx.get(), n_tokens, io_result_offset_);
        ggml_tensor * copy = ggml_cpy(ctx.get(), fold, result);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(graph, copy);
        if (!ggml_gallocr_alloc_graph(compute_galloc_.get(), graph)) {
            throw std::runtime_error("failed to allocate resident-first fold graph");
        }
        const enum ggml_status status = submit_graph(graph, request_stats);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("resident-first fold graph compute failed");
        }
    }

    // *** D2 (WP_EXPERT_GRAPH_CACHE): SHAPE-KEYED PERSISTENT GRAPHS. ***
    // The backend's CUDA/HIP graph cache keys on cgraph->nodes[0] -- a tensor
    // address inside the per-request ggml context -- so a fresh context per
    // request churns the key and the 2-call warmup never completes: every
    // submit pays ~n_nodes raw kernel launches (live ns_submit 0.25-1.8 ms vs
    // 87 us for a static graph). Cache one context+graph+gallocr per SHAPE
    // (n_tokens, n_selected, idx_rank, add_previous, clamp); per request only
    // the expert weight tensors rebind (attach_weight = src data ptrs only)
    // and the routing-weight blob repacks at fixed offsets. With the graph flag
    // enabled the backend then takes its stable-key path instead
    // of resetting warmup. Gather caches when every selected expert has the
    // same idx rank (verify). Mixed ranks skip the cache. Entries pin their
    // gallocr VRAM (~2-4 MB each); the LRU cap bounds it.
    struct ArenaGraphKey {
        uint32_t n_tokens = 0;
        uint32_t n_assignments = 0;
        std::vector<uint32_t> group_arenas;
        std::vector<uint32_t> group_sizes;

        bool operator<(const ArenaGraphKey & o) const {
            return std::tie(n_tokens, n_assignments, group_arenas, group_sizes) <
                   std::tie(o.n_tokens, o.n_assignments, o.group_arenas, o.group_sizes);
        }
    };


    struct ArenaGraphEntry {
        context_ptr                 ctx;
        galloc_ptr                  galloc;
        ggml_cgraph *               graph = nullptr;
        ggml_backend_graph_plan_t   persistent_plan = nullptr;
        ggml_tensor *               blob = nullptr;
        ggml_backend_buffer_t       io_buffer = nullptr;
        std::array<ArenaRoleKey, 3> roles;
        uint32_t                    clamp_bits = 0;
        uint64_t                    io_gen = 0;
        uint64_t                    params_gen = 0;
        uint64_t                    last_used = 0;
        // WP_ARENA_FOLD_COLLAPSE + WP_ARENA_HIP_GRAPH (single-arena path
        // only -- see compute_batch_arena): HIP graph capture/replay state
        // for this bucket. hip_graph_submits counts consecutive successful
        // eager submits before attempting capture (0 -> 1 -> capture -> 2 =
        // steady-state replay); hip_graph_failed permanently falls back to
        // eager for this bucket once a captured replay fails.
        uint8_t                     hip_graph_submits = 0;
        bool                        hip_graph_failed = false;
    };

    struct GraphKey {
        uint32_t n_tokens = 0;
        uint32_t n_selected = 0;
        uint32_t idx_rank = 0;   // 0 = dense; else gather row count per expert
        bool     add_previous = false;
        uint32_t clamp_bits = 0;
        // MAD-LAB 2026-08-26: THE WEIGHT TYPES ARE PART OF THE KEY.
        //
        // The D2 fast path rebinds only each weight tensor's buffer+data
        // POINTER (attach_weight); the cached ggml_tensor keeps the TYPE and
        // SHAPE it was built with. This model's expert tensors are not
        // uniformly quantized -- it is an Unsloth UD-Q4_K_XL dynamic quant, so
        // e.g. layer 2 is q8_0 down + q5_K gate/up while layers 0/1/3/5 are
        // q5_1 down + q4_K gate/up. Those two shapes collide on every other
        // key field (same n_tokens, n_selected, idx_rank, add_previous, clamp),
        // so a graph built for one layer was replayed for the other and the
        // matmul decoded q8_0 bytes as q5_1: deterministic garbage, identical
        // on CUDA/Vulkan/CPU because all three faithfully misinterpret the
        // same bytes, and invisible to any check on the shard data (which is
        // byte-identical to the source GGUF).
        uint32_t type_gate = 0;
        uint32_t type_up   = 0;
        uint32_t type_down = 0;
        int64_t  ne0_gate = 0, ne1_gate = 0;
        int64_t  ne0_up = 0, ne1_up = 0;
        int64_t  ne0_down = 0, ne1_down = 0;
        // WP_EXPERT_FUSE_GATE_UP: a fused graph has ONE [ne0, 2*ne1] weight per
        // expert where an unfused one has two, so the two shapes must never
        // share a cache entry (same failure mode as the type fields above).
        bool     fused_gate_up = false;
        bool operator<(const GraphKey & o) const {
            return std::tie(n_tokens, n_selected, idx_rank, add_previous, clamp_bits,
                            type_gate, type_up, type_down,
                            ne0_gate, ne1_gate, ne0_up, ne1_up, ne0_down, ne1_down,
                            fused_gate_up) <
                   std::tie(o.n_tokens, o.n_selected, o.idx_rank, o.add_previous, o.clamp_bits,
                            o.type_gate, o.type_up, o.type_down,
                            o.ne0_gate, o.ne1_gate, o.ne0_up, o.ne1_up,
                            o.ne0_down, o.ne1_down, o.fused_gate_up);
        }
    };
    struct GraphCacheEntry {
        context_ptr   ctx;
        galloc_ptr    galloc;
        ggml_cgraph * graph = nullptr;
        ggml_backend_graph_plan_t persistent_plan = nullptr;
        ggml_tensor * blob  = nullptr;
        ggml_backend_buffer_t io_buffer = nullptr;
        // gate,up,down per expert, flat. When fused_gate_up is set the slot
        // for `up` holds nullptr and slot `gate` holds the [ne0, 2*ne1] view.
        std::vector<ggml_tensor *> expert_w;
        std::vector<ggml_tensor *> route_w;    // routing weights per expert
        std::vector<ggml_tensor *> gather_idx; // I32 idx per expert (gather only)
        uint64_t io_gen = 0, params_gen = 0, last_used = 0;
    };
    std::map<GraphKey, GraphCacheEntry> graph_cache_;
    uint64_t graph_cache_tick_ = 0;
    std::map<ArenaGraphKey, ArenaGraphEntry> arena_graph_cache_;
    uint64_t arena_graph_cache_tick_ = 0;
    // Buffer generations: a grow REPLACES the device buffer, so every cached
    // graph holding tensors bound into the old one is stale. Bumped by the
    // grow_* functions; checked at cache lookup.
    uint64_t io_gen_ = 0, params_gen_ = 0;

    Catalog        catalog_;
    std::function<bool(int, int)> page_owner_;
    WorkerLogFiles * logs_ = nullptr;
    // Per-layer catalog views, (blob, offset) sorted at load. Used by
    // WP_PREFILL_LAYER_AHEAD so the L+1 union is a sequential NVMe stream
    // rather than assignment-order random seeks.
    std::map<int, std::vector<const ExpertPage *>> layer_pages_sorted_;
    backend_ptr    backend_;
    backend_ptr    cpu_backend_;
    ResidentExpertPool resident_;
    ExpertSlotPool pool_;
    galloc_ptr     compute_galloc_;
    buffer_ptr     io_buffer_;
    size_t         io_buffer_size_ = 0;
    // WP_IO_SMALL_TOKENS: a SECOND, small, host-visible io buffer used only for
    // decode-sized requests. See the block comment on alloc_io_small().
    buffer_ptr     io_small_;
    size_t         io_small_size_ = 0;
    // Pinned HOST staging for the activation SOURCE (all backends). Never a
    // graph tensor -- only the src pointer handed to ggml_backend_tensor_set.
    buffer_ptr     io_src_pinned_;
    void *         io_src_base_ = nullptr;
    size_t         io_src_size_ = 0;
    std::string    device_name_;
    bool           io_set_async_ = false;
    size_t         io_reserved_hint_ = 0;
    ggml_backend_buffer_t io_active_ = nullptr;
    uint64_t            io_grow_count_ = 0;
    buffer_ptr     params_buffer_;
    size_t         params_buffer_size_ = 0;
    // WP_EXPERT_BATCH_MOE / WP_EXPERT_GROUPED_GEMV (D3): scratch VRAM for the
    // grouped mul_mat_id path (see compute_batch_grouped). The normal grouped
    // path holds N page-sized role slots. WP_WORKER_COLLAPSE uses role-major
    // contiguous regions when the selected slot span permits three batched
    // copies; both layouts expose [n_embd, n_ff_slice, n_selected] batched
    // weight tensors.
    // Content never outlives one compute_batch_grouped call (same invariant
    // as params_buffer_), so growing it here is always safe.
    buffer_ptr     batch_scratch_;
    size_t         batch_scratch_size_ = 0;
    size_t         io_result_offset_ = 0;
    WorkerStats    stats_;
    int            slots_ = 0;
    int            active_async_conn_index_ = -1;
    std::unordered_map<int, AsyncSubmitState> async_submit_state_by_conn_;
    // Keyed by conn_index -- see the long comment above begin_split_dispatch()
    // for why this must not be a single Worker-wide std::optional.
    std::unordered_map<int, split_pending> split_pending_by_conn_;
};

DeviceWorker::~DeviceWorker() {
    if (backend_ == nullptr) {
        return;
    }
    for (auto & kv : graph_cache_) {
        if (kv.second.persistent_plan != nullptr) {
            release_persistent_plan(kv.second.persistent_plan);
        }
    }
    for (auto & kv : arena_graph_cache_) {
        if (kv.second.persistent_plan != nullptr) {
            release_persistent_plan(kv.second.persistent_plan);
        }
    }
}

static void accumulate_request_stats(RequestStats & dst, const RequestStats & src) {
    dst.ns_lookup += src.ns_lookup;
    dst.ns_read += src.ns_read;
    dst.ns_compute += src.ns_compute;
    dst.ns_send += src.ns_send;
    dst.n_resident += src.n_resident;
    dst.n_pagein += src.n_pagein;
    dst.n_host_hit += src.n_host_hit;
    dst.n_host_demote += src.n_host_demote;
    dst.bytes_read += src.bytes_read;
    dst.n_pagein_reserved += src.n_pagein_reserved;
    dst.n_pagein_general += src.n_pagein_general;
    dst.ns_host_get += src.ns_host_get;
    dst.host_bytes += src.host_bytes;
    dst.n_graph_submits += src.n_graph_submits;
    dst.n_device_allocs += src.n_device_allocs;
    dst.ns_graph_build += src.ns_graph_build;
    dst.ns_submit += src.ns_submit;
    dst.ns_final_sync += src.ns_final_sync;
    dst.ns_vk_compute_path += src.ns_vk_compute_path;
    dst.ns_vk_dispatch_path += src.ns_vk_dispatch_path;
    dst.ns_vk_wait += src.ns_vk_wait;
    dst.ns_vk_cache_lookup += src.ns_vk_cache_lookup;
    dst.ns_vk_graph_compute += src.ns_vk_graph_compute;
    dst.ns_vk_params_set += src.ns_vk_params_set;
    dst.ns_vk_fold += src.ns_vk_fold;
    dst.ns_vk_sync += src.ns_vk_sync;
    dst.ns_vk_readback += src.ns_vk_readback;
    dst.ns_prologue += src.ns_prologue;
    dst.ns_arena_probe += src.ns_arena_probe;
    dst.ns_vk_arena_probe += src.ns_vk_arena_probe;
    dst.ns_vk_setup += src.ns_vk_setup;
    dst.ns_vk_rebind += src.ns_vk_rebind;
    dst.ns_vk_layer_ahead += src.ns_vk_layer_ahead;
    dst.ns_readback += src.ns_readback;
    dst.ns_prep += src.ns_prep;
    dst.ns_prep_setup += src.ns_prep_setup;
    dst.ns_prep_grow += src.ns_prep_grow;
    dst.ns_prep_attach += src.ns_prep_attach;
    dst.ns_prep_set += src.ns_prep_set;
    dst.ns_hits += src.ns_hits;
    dst.ns_wait += src.ns_wait;
    dst.ns_pagein_compute += src.ns_pagein_compute;
    dst.ns_result += src.ns_result;
    dst.ns_encode += src.ns_encode;
    dst.ns_h2d += src.ns_h2d;
    dst.bytes_h2d += src.bytes_h2d;
    dst.n_reader_h2d += src.n_reader_h2d;
    dst.n_weight_nonzero += src.n_weight_nonzero;
    dst.n_weight_total += src.n_weight_total;
    dst.n_gcache_hit += src.n_gcache_hit;
    dst.n_gcache_miss += src.n_gcache_miss;
    dst.n_arena_hit += src.n_arena_hit;
    dst.n_arena_groups += src.n_arena_groups;
    dst.n_arena_build += src.n_arena_build;
    dst.n_hipgraph_capture += src.n_hipgraph_capture;
    dst.n_hipgraph_replay += src.n_hipgraph_replay;
    dst.n_d3_collapse += src.n_d3_collapse;
    dst.n_d3_typed += src.n_d3_typed;
    dst.n_d3_bounce += src.n_d3_bounce;
    dst.ns_params_set += src.ns_params_set;
    dst.ns_demote += src.ns_demote;
    dst.ns_ensure_post += src.ns_ensure_post;
    dst.n_read_inflight_max = std::max(dst.n_read_inflight_max, src.n_read_inflight_max);
    dst.ns_read_issue += src.ns_read_issue;
    dst.ns_read_complete += src.ns_read_complete;
    dst.n_cpu_on_arrival += src.n_cpu_on_arrival;
    dst.ns_cpu_on_arrival += src.ns_cpu_on_arrival;
    dst.n_cpu_on_arrival_fallback += src.n_cpu_on_arrival_fallback;
    dst.d3_counted = dst.d3_counted || src.d3_counted;
}

class Worker {
public:
    Worker(
            Catalog catalog,
            const std::string & device,
            int slots,
            uint64_t host_budget_bytes,
            uint64_t host_victim_bytes,
            TestHooks * test_hooks,
            const std::vector<int> & resident_expert_blocks,
            const std::vector<int> & expert_reserve_blocks,
            uint64_t expert_reserve_bytes) :
        Worker(std::move(catalog), std::vector<std::string>{device},
               std::vector<int>{slots}, host_budget_bytes, host_victim_bytes,
               test_hooks, resident_expert_blocks, expert_reserve_blocks,
               expert_reserve_bytes) {
    }

    Worker(
            Catalog catalog,
            const std::vector<std::string> & devices,
            const std::vector<int> & device_slots,
            uint64_t host_budget_bytes,
            uint64_t host_victim_bytes,
            TestHooks * test_hooks,
            const std::vector<int> & resident_expert_blocks,
            const std::vector<int> & expert_reserve_blocks,
            uint64_t expert_reserve_bytes) :
        catalog_(std::move(catalog)),
        device_names_(devices),
        device_slots_(device_slots),
        logs_(),
        device_mutexes_(devices.size()) {
        if (device_names_.empty() || device_names_.size() != device_slots_.size()) {
            throw std::invalid_argument(
                "worker device and slot lists must have the same non-zero length");
        }
        for (const int slots : device_slots_) {
            if (slots <= 0) {
                throw std::invalid_argument("worker device slot budgets must be positive");
            }
        }
        const size_t n_pages = catalog_.pages.size();
        page_access_counts_ = std::make_unique<std::atomic<uint64_t>[]>(n_pages);
        page_current_owner_ = std::make_unique<std::atomic<uint32_t>[]>(n_pages);
        page_static_owners_.resize(n_pages);
        resident_expert_blocks_ = resident_expert_blocks;
        std::sort(resident_expert_blocks_.begin(), resident_expert_blocks_.end());
        for (const auto & item : catalog_.pages) {
            const ExpertPage & page = item.second;
            if (page.cache_id < 0 || (size_t) page.cache_id >= n_pages) {
                throw std::invalid_argument("expert page cache ids are not contiguous");
            }
            const size_t id = (size_t) page.cache_id;
            const size_t owner = static_owner_for_page(page.layer, page.expert);
            page_static_owners_[id] = owner;
            page_access_counts_[id].store(0, std::memory_order_relaxed);
            page_current_owner_[id].store((uint32_t) owner, std::memory_order_relaxed);
        }
        devices_.reserve(device_names_.size());
        const bool single_device = device_names_.size() == 1;
        for (size_t i = 0; i < device_names_.size(); ++i) {
            const std::function<bool(int, int)> page_owner = single_device
                ? std::function<bool(int, int)>()
                : [this, i](int layer, int expert) {
                    return owning_device_for_page(layer, expert) == i;
                };
            devices_.emplace_back(std::make_unique<DeviceWorker>(
                catalog_, device_names_[i], device_slots_[i], host_budget_bytes,
                host_victim_bytes, test_hooks, resident_expert_blocks,
                expert_reserve_blocks, expert_reserve_bytes,
                single_device ? nullptr : &host_tier_, single_device,
                page_owner, &logs_));
        }
        initialize_placement_policy();
        if (!single_device) {
            load_pin_file();
        }
    }

    size_t owning_device_for_page(
            int layer, int expert, size_t * migration_budget = nullptr) const {
        return owner_for_page(layer, expert, migration_budget);
    }

    size_t static_owner_for_page(int layer, int expert) const {
        (void) layer;
        const int first = catalog_.descriptor.expert_first;
        const int last = catalog_.descriptor.expert_last;
        const uint64_t count = last >= first
            ? (uint64_t) (last - first) + 1 : 0;
        const uint64_t ordinal = expert >= first ? (uint64_t) (expert - first) : 0;
        const uint64_t total = std::accumulate(
            device_slots_.begin(), device_slots_.end(), (uint64_t) 0);
        const uint64_t point = count > 0 ? ordinal * total / count : 0;
        uint64_t begin = 0;
        for (size_t i = 0; i < device_slots_.size(); ++i) {
            begin += (uint64_t) device_slots_[i];
            if (point < begin) {
                return i;
            }
        }
        return device_slots_.size() - 1;
    }

    pipe_expert_hello hello() const {
        pipe_expert_hello result = devices_.front()->hello();
        uint64_t slots = 0;
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            slots += device->hello().n_slots;
        }
        if (slots > UINT32_MAX) {
            throw std::overflow_error("expert worker slot count overflows HELLO");
        }
        result.n_slots = (uint32_t) slots;
        return result;
    }

    const ResourcePlan & resources() const {
        return devices_.front()->resources();
    }

    const ResourcePlan & device_resources(size_t index) const {
        return devices_.at(index)->resources();
    }

    const std::string & device_name(size_t index) const {
        return device_names_.at(index);
    }

    size_t device_count() const {
        return devices_.size();
    }

    bool multi_device() const {
        return devices_.size() > 1;
    }

    size_t read_inflight() const { return devices_.front()->read_inflight(); }
    size_t read_chunk_bytes() const { return devices_.front()->read_chunk_bytes(); }
    bool read_direct() const { return devices_.front()->read_direct(); }
    bool read_direct_fallback() const {
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            if (device->read_direct_fallback()) {
                return true;
            }
        }
        return false;
    }

    int pinned_pages() const {
        int result = 0;
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            result += device->pinned_pages();
        }
        return result;
    }

    bool stats_enabled() const {
        return devices_.front()->stats_enabled();
    }

    void record_stats(const RequestStats & request, size_t n_experts) {
        if (multi_device()) {
            devices_.front()->record_wire_stats(request);
        } else {
            devices_.front()->record_stats(request, n_experts);
        }
    }

    void set_staging_multi_conn(int n) {
        for (size_t i = 0; i < devices_.size(); ++i) {
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            devices_[i]->set_staging_multi_conn(n);
        }
    }

    void keepalive_tick() {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (hip_graph_executor_needed(i)) {
                ensure_device_executor(i);
                device_exec_[i]->run([this, i] {
                    std::lock_guard<std::mutex> lock(device_mutexes_[i]);
                    devices_[i]->keepalive_tick();
                });
            } else {
                std::lock_guard<std::mutex> lock(device_mutexes_[i]);
                devices_[i]->keepalive_tick();
            }
        }
    }

    bool keepalive_enabled() const {
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            if (device->keepalive_enabled()) {
                return true;
            }
        }
        return false;
    }

    int keepalive_us() const {
        int result = std::numeric_limits<int>::max();
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            if (device->keepalive_enabled()) {
                result = std::min(result, device->keepalive_us());
            }
        }
        return result == std::numeric_limits<int>::max() ? 0 : result;
    }

    bool has_spec_work() const {
        for (size_t i = 0; i < devices_.size(); ++i) {
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            if (devices_[i]->has_spec_work()) {
                return true;
            }
        }
        return false;
    }

    bool has_spec_submit_work() const {
        for (size_t i = 0; i < devices_.size(); ++i) {
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            if (devices_[i]->has_spec_submit_work()) {
                return true;
            }
        }
        return false;
    }

    void drop_spec_work() {
        for (size_t i = 0; i < devices_.size(); ++i) {
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            devices_[i]->drop_spec_work();
        }
    }

    bool spec_pagein_step(bool harvest = true) {
        bool result = false;
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (hip_graph_executor_needed(i)) {
                ensure_device_executor(i);
                bool did_work = false;
                device_exec_[i]->run([this, i, harvest, &did_work] {
                    std::lock_guard<std::mutex> lock(device_mutexes_[i]);
                    did_work = devices_[i]->spec_pagein_step(harvest);
                });
                result |= did_work;
            } else {
                std::lock_guard<std::mutex> lock(device_mutexes_[i]);
                result |= devices_[i]->spec_pagein_step(harvest);
            }
        }
        return result;
    }

    void spec_pagein_after_dispatch() {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (hip_graph_executor_needed(i)) {
                ensure_device_executor(i);
                device_exec_[i]->run([this, i] {
                    std::lock_guard<std::mutex> lock(device_mutexes_[i]);
                    devices_[i]->spec_pagein_after_dispatch();
                });
            } else {
                std::lock_guard<std::mutex> lock(device_mutexes_[i]);
                devices_[i]->spec_pagein_after_dispatch();
            }
        }
    }

    void note_prefetch_hint(const pipe_expert_prefetch_hint & hint) {
        std::vector<std::vector<int32_t>> ids(devices_.size());
        for (const int32_t expert : hint.expert_ids) {
            const size_t owner = catalog_.pages.count({ hint.layer, expert }) != 0
                ? owning_device_for_page(hint.layer, expert) : 0;
            ids[owner].push_back(expert);
        }
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (ids[i].empty()) {
                continue;
            }
            pipe_expert_prefetch_hint sub = hint;
            sub.expert_ids = std::move(ids[i]);
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            devices_[i]->note_prefetch_hint(sub);
        }
    }

    void note_prefetch_hint_bad() {
        std::lock_guard<std::mutex> lock(device_mutexes_.front());
        devices_.front()->note_prefetch_hint_bad();
    }

    void log_reference(int32_t layer,
                       const std::vector<pipe_expert_assignment> & assignments) {
        std::lock_guard<std::mutex> lock(device_mutexes_.front());
        devices_.front()->log_reference(layer, assignments);
    }

    void report_prefetch_hints() const {
        for (size_t i = 0; i < devices_.size(); ++i) {
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            devices_[i]->report_prefetch_hints();
        }
    }

    pipe_expert_partial dispatch(
            const pipe_expert_dispatch_req & request,
            RequestStats & request_stats,
            std::optional<ExpertSlotPool::Batch> prepared = std::nullopt,
            int conn_index = -1) {
        if (prepared.has_value() || !multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            return devices_.front()->dispatch(
                request, request_stats, std::move(prepared), conn_index);
        }
        validate_dispatch(request);
        note_dispatch_references(request);
        pipe_expert_partial result;
        result.layer = request.layer;
        result.n_tokens = request.n_tokens;
        result.dtype = PIPE_HIDDEN_F32;
        result.partial.assign(
            (size_t) request.n_tokens * catalog_.descriptor.hparams.n_embd, 0.0f);
        size_t migration_budget = migration_cap_;
        const std::vector<AssignmentGroup> groups =
            assignment_groups(request, &migration_budget);
        // See DeviceExecutor: the serial version of this loop made a layer cost
        // sum(devices) instead of max(devices).
        std::vector<pipe_expert_partial> partials(groups.size());
        std::vector<RequestStats>        sub_stats(groups.size());
        if (device_parallel_ && (groups.size() > 1 || hip_graph_executor_enabled())) {
            if (device_exec_.size() != devices_.size()) {
                device_exec_.resize(devices_.size());
            }
            // Groups are bucketed BY DEVICE: two groups on the same device must
            // still run one after another (one executor thread each), or they
            // would race on that device's backend.
            std::vector<std::vector<size_t>> by_device(devices_.size());
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                by_device[groups[gi].device].push_back(gi);
            }
            // *** STRICT THREAD AFFINITY: A DEVICE IS ALWAYS DRIVEN BY ITS OWN
            // EXECUTOR THREAD, NEVER BY THE CALLING THREAD. ***
            // An earlier revision ran the HEAVIEST group inline on the caller to
            // save one condition-variable handoff. That CRASHED the worker:
            //   Memory access fault by GPU node-2 on address 0x7fa429b3d000.
            //   Reason: Page not present or supervisor privilege.
            // Which device is heaviest changes per REQUEST, so a device's GPU
            // work migrated between the caller and its executor from one request
            // to the next. Vulkan command pools have thread affinity and HIP
            // streams are bound per thread (the same constraint spec_pagein_step
            // documents for drain_one_read's H2D) -- driving one device from two
            // different threads is not legal here, however cheap it looks.
            // The handoff cost is real but is paid down by join_one()'s spin,
            // NOT by moving work onto the caller.
            const auto run_device = [this, &by_device, &groups, &request,
                                     &partials, &sub_stats, conn_index](size_t d) {
                for (const size_t gi : by_device[d]) {
                    const AssignmentGroup & group = groups[gi];
                    pipe_expert_dispatch_req sub =
                        make_subrequest(request, group.begin, group.end);
                    std::lock_guard<std::mutex> lock(device_mutexes_[group.device]);
                    partials[gi] = devices_[group.device]->dispatch(
                        sub, sub_stats[gi], std::nullopt, conn_index);
                    if (devices_[group.device]->stats_enabled()) {
                        devices_[group.device]->record_stats(
                            sub_stats[gi], group.end - group.begin);
                    }
                }
            };
            // *** THE CPU EXPERT DEVICE RUNS OUTSIDE THE CONCURRENT WINDOW. ***
            // MEASURED 2026-08-29 on the box with THREE DIFFERENT BACKENDS
            // (2026: CUDA0 + Vulkan0 + CPU -- no shared runtime lock anywhere),
            // running all three concurrently:
            //     RX480 (Vulkan) ns_submit 261 -> 495 us
            //     CPU            ns_submit 125 -> 362 us   <-- makes NO GPU calls
            // The CPU device's own compute tripling proves the cost is CPU CORE
            // STARVATION, not a GPU-runtime lock: ggml's CPU backend SPIN-WAITS
            // at its thread barriers, so the expert tier's threads burn every
            // core they are given and deschedule the GPU devices' submission and
            // host-side threads. Same cause on main, different leg -- a GPU
            // submit is mostly waiting on the GPU so it absorbs descheduling,
            // and the host-side span (the untimed hole) takes the hit instead
            // (215 -> 1177 us on the R9700).
            // This is the same effect already on record here: WP_CPU_THREADS
            // 8 -> 4 cut CPU expert time 46% AND made the RX 480 19% faster for
            // free, because the CPU tier was starving the Vulkan driver.
            // So: GPUs overlap each other, and the CPU tier gets the machine to
            // itself. It is the SMALLEST group (231 us vs 833/714), so serialising
            // it costs little; letting it spin alongside the GPUs cost everything.
            // Ideal layer = max(GPUs) + CPU = 833 + 231 = 1064 us, vs 1779 serial.
            //
            // *** WP_CPU_TIER_OVERLAP=1 LIFTS THAT SERIALISATION. *** Default
            // off; unset, everything below runs exactly as described above.
            // Set, the CPU tier is launched in the SAME window as the GPUs and
            // the leg becomes max(GPUs, CPU) instead of max(GPUs) + CPU -- worth
            // a measured 0.42 ms/leg on 2026 and 0.32 ms/leg on main, ~48 legs
            // per token. What makes that safe now and unsafe before is that the
            // tier's backend is bound to a CPU-confined, SCHED_BATCH threadpool
            // (see configure_cpu_backend): its threads still spin, but they no
            // longer own cores the GPU submission threads need, and they lose
            // wakeup preemption against those threads. The knob is per-box
            // because the right CPU set is a topology question -- see
            // WP_CPU_TIER_CPUS.
            const bool cpu_overlap = cpu_tier_overlap_enabled();
            std::vector<size_t> launched;
            size_t cpu_device = SIZE_MAX;
            // The CPU tier's executor is the only one that gets pinned, and the
            // only one for which pinning is meaningful -- WP_CPU_TIER_PIN, off
            // by default. Every other device's executor is a GPU submission
            // thread and must stay wherever the scheduler wants it.
            const auto ensure_exec = [this](size_t d) {
                if (!device_exec_[d]) {
                    device_exec_[d] = std::make_unique<DeviceExecutor>();
                    device_exec_[d]->start(/* pin = */ is_cpu_device(d));
                }
            };
            for (size_t d = 0; d < by_device.size(); ++d) {
                if (by_device[d].empty()) { continue; }
                if (is_cpu_device(d)) {
                    cpu_device = d;   // scheduled separately, below
                    break;
                }
            }
            // LAUNCH THE CPU TIER FIRST when overlapping. It is the shortest
            // group, so starting it last would leave its tail hanging past the
            // GPUs for no reason; starting it first puts its whole span inside
            // the GPU window. It still gets its OWN executor thread -- device
            // work never migrates threads here, overlap or not (an earlier
            // revision ran a device on the calling thread and took a GPU
            // memory-access fault for it).
            if (cpu_overlap && cpu_device != SIZE_MAX) {
                ensure_exec(cpu_device);
                device_exec_[cpu_device]->submit(
                    [run_device, cpu_device] { run_device(cpu_device); });
                launched.push_back(cpu_device);
            }
            for (size_t d = 0; d < by_device.size(); ++d) {
                if (by_device[d].empty() || d == cpu_device) { continue; }
                ensure_exec(d);
                device_exec_[d]->submit([run_device, d] { run_device(d); });
                launched.push_back(d);
            }
            // Join ALL before rethrowing any, or a still-running executor would
            // keep writing into partials/sub_stats after this frame unwound.
            // Join the GPUs before the CPU tier even though the CPU tier was
            // submitted first: the GPU legs are the long pole, so waiting on
            // them first means the CPU join is almost always already satisfied
            // and costs no extra spin.
            std::exception_ptr first_error;
            const auto join = [this, &first_error](size_t d) {
                try {
                    device_exec_[d]->join_one();
                } catch (...) {
                    if (!first_error) { first_error = std::current_exception(); }
                }
            };
            for (const size_t d : launched) {
                if (d == cpu_device) { continue; }
                join(d);
            }
            // PHASE 2 (default builds): the CPU tier, alone, after every GPU has
            // finished. With WP_CPU_TIER_OVERLAP=1 it was already submitted
            // above and this only collects it.
            if (cpu_device != SIZE_MAX) {
                if (!cpu_overlap) {
                    ensure_exec(cpu_device);
                    device_exec_[cpu_device]->submit(
                        [run_device, cpu_device] { run_device(cpu_device); });
                }
                join(cpu_device);
            }
            if (first_error) { std::rethrow_exception(first_error); }
        } else {
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                const AssignmentGroup & group = groups[gi];
                pipe_expert_dispatch_req sub = make_subrequest(request, group.begin, group.end);
                std::lock_guard<std::mutex> lock(device_mutexes_[group.device]);
                partials[gi] = devices_[group.device]->dispatch(
                    sub, sub_stats[gi], std::nullopt, conn_index);
                if (devices_[group.device]->stats_enabled()) {
                    devices_[group.device]->record_stats(sub_stats[gi], group.end - group.begin);
                }
            }
        }
        // FOLD IN GROUP ORDER -- identical association to the serial loop, so
        // the output stays bit-for-bit what it was. Only compute is reordered.
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            accumulate_request_stats(request_stats, sub_stats[gi]);
            if (partials[gi].partial.size() != result.partial.size()) {
                throw std::runtime_error("expert device partial sizes disagree");
            }
            for (size_t i = 0; i < result.partial.size(); ++i) {
                result.partial[i] += partials[gi].partial[i];
            }
        }
        return result;
    }

    void begin_split_dispatch(const pipe_expert_dispatch_begin & begin, uint64_t seq_id,
                              int conn_index = -1,
                              std::unique_lock<std::mutex> * gpu_lock = nullptr) {
        if (!multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            devices_.front()->begin_split_dispatch(begin, seq_id, conn_index, gpu_lock);
            return;
        }
        pipe_expert_dispatch_req request;
        request.layer = begin.layer;
        request.n_tokens = begin.n_tokens;
        request.assignments = begin.assignments;
        request.swiglu_clamp = begin.swiglu_clamp;
        validate_dispatch(request);
        {
            std::lock_guard<std::mutex> lock(split_mutex_);
            if (split_pending_by_conn_.count(conn_index) != 0) {
                throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                          "expert dispatch BEGIN arrived before ACTS");
            }
        }
        note_dispatch_references(request);
        size_t migration_budget = migration_cap_;
        std::vector<AssignmentGroup> groups =
            assignment_groups(request, &migration_budget);
        if (groups.empty()) {
            groups.push_back({0, 0, 0});
        }
        {
            std::lock_guard<std::mutex> lock(split_mutex_);
            if (split_pending_by_conn_.count(conn_index) != 0) {
                throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                          "expert dispatch BEGIN arrived before ACTS");
            }
            split_pending_by_conn_.emplace(conn_index,
                                           SplitPending{begin, seq_id, groups});
        }
    }

    pipe_expert_partial finish_split_dispatch(
            const pipe_expert_dispatch_acts & acts, uint64_t seq_id,
            RequestStats & request_stats, int conn_index = -1) {
        if (!multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            return devices_.front()->finish_split_dispatch(
                acts, seq_id, request_stats, conn_index);
        }
        SplitPending pending;
        {
            std::lock_guard<std::mutex> lock(split_mutex_);
            const auto it = split_pending_by_conn_.find(conn_index);
            if (it == split_pending_by_conn_.end()) {
                throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                          "expert dispatch ACTS has no BEGIN");
            }
            if (seq_id != it->second.seq_id) {
                throw pipe_protocol_error(PIPE_ERR_STALE_SEQ,
                                          "expert dispatch ACTS sequence does not match BEGIN");
            }
            pending = std::move(it->second);
            split_pending_by_conn_.erase(it);
        }
        pipe_expert_partial result;
        result.layer = pending.begin.layer;
        result.n_tokens = pending.begin.n_tokens;
        result.dtype = PIPE_HIDDEN_F32;
        result.partial.assign(
            (size_t) result.n_tokens * catalog_.descriptor.hparams.n_embd, 0.0f);
        for (const AssignmentGroup & group : pending.groups) {
            pipe_expert_dispatch_req sub_request;
            sub_request.layer = pending.begin.layer;
            sub_request.n_tokens = pending.begin.n_tokens;
            sub_request.swiglu_clamp = pending.begin.swiglu_clamp;
            sub_request.assignments.assign(
                pending.begin.assignments.begin() + (ptrdiff_t) group.begin,
                pending.begin.assignments.begin() + (ptrdiff_t) group.end);
            sub_request.activations = acts.activations;
            RequestStats sub_stats;
            pipe_expert_partial partial;
            if (hip_graph_executor_needed(group.device)) {
                ensure_device_executor(group.device);
                device_exec_[group.device]->run([this, &sub_request, &sub_stats, &partial,
                                                 &group, conn_index] {
                    std::lock_guard<std::mutex> lock(device_mutexes_[group.device]);
                    partial = devices_[group.device]->dispatch(
                        sub_request, sub_stats, std::nullopt, conn_index);
                    if (devices_[group.device]->stats_enabled()) {
                        devices_[group.device]->record_stats(
                            sub_stats, group.end - group.begin);
                    }
                });
            } else {
                std::lock_guard<std::mutex> lock(device_mutexes_[group.device]);
                partial = devices_[group.device]->dispatch(
                    sub_request, sub_stats, std::nullopt, conn_index);
                if (devices_[group.device]->stats_enabled()) {
                    devices_[group.device]->record_stats(
                        sub_stats, group.end - group.begin);
                }
            }
            accumulate_request_stats(request_stats, sub_stats);
            if (partial.partial.size() != result.partial.size()) {
                throw std::runtime_error("expert device partial sizes disagree");
            }
            for (size_t i = 0; i < result.partial.size(); ++i) {
                result.partial[i] += partial.partial[i];
            }
        }
        return result;
    }

    void abandon_split_dispatch(int conn_index = -1) noexcept {
        if (!multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            devices_.front()->abandon_split_dispatch(conn_index);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(split_mutex_);
            const auto it = split_pending_by_conn_.find(conn_index);
            if (it == split_pending_by_conn_.end()) {
                return;
            }
            split_pending_by_conn_.erase(it);
        }
    }

    bool has_split_dispatch(int conn_index = -1) const {
        if (!multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            return devices_.front()->has_split_dispatch(conn_index);
        }
        std::lock_guard<std::mutex> lock(split_mutex_);
        return split_pending_by_conn_.count(conn_index) != 0;
    }

    uint32_t split_n_tokens(int conn_index = -1) const {
        if (!multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            return devices_.front()->split_n_tokens(conn_index);
        }
        std::lock_guard<std::mutex> lock(split_mutex_);
        return split_pending_by_conn_.at(conn_index).begin.n_tokens;
    }

    bool split_arena_eligible(int conn_index = -1) const {
        if (!multi_device()) {
            std::lock_guard<std::mutex> lock(device_mutexes_.front());
            return devices_.front()->split_arena_eligible(conn_index);
        }
        return false;
    }

private:
    bool hip_graph_executor_enabled() const {
        if (!device_parallel_ || !wp_hip_graphs_enabled()) {
            return false;
        }
        for (size_t i = 0; i < device_names_.size(); ++i) {
            if (hip_graph_executor_needed(i)) {
                return true;
            }
        }
        return false;
    }

    bool hip_graph_executor_needed(size_t device) const {
        return device_parallel_ && wp_hip_graphs_enabled() &&
               device < device_names_.size() &&
               device_names_[device].rfind("ROCm", 0) == 0;
    }

    bool is_cpu_device(size_t device) const {
        return device < device_names_.size() &&
               device_names_[device].find("CPU") != std::string::npos;
    }

    void ensure_device_executor(size_t device) {
        if (device_exec_.size() != devices_.size()) {
            device_exec_.resize(devices_.size());
        }
        if (!device_exec_[device]) {
            device_exec_[device] = std::make_unique<DeviceExecutor>();
            // Pin here too: whichever call site happens to create the CPU
            // tier's executor first must be the one that places it, because the
            // pin is applied once at thread start and this executor lives for
            // the process. See wp_cpu_tier_pin_self().
            device_exec_[device]->start(/* pin = */ is_cpu_device(device));
        }
    }

    struct PlacementSnapshot {
        std::vector<uint32_t> owner;
        std::vector<uint64_t> promotion_floor;
        std::vector<uint64_t> demotion_ceiling;
    };

    size_t page_id_for(int layer, int expert) const {
        const auto it = catalog_.pages.find({layer, expert});
        if (it == catalog_.pages.end() || it->second.cache_id < 0) {
            return page_static_owners_.size();
        }
        const size_t id = (size_t) it->second.cache_id;
        return id < page_static_owners_.size() ? id : page_static_owners_.size();
    }

    size_t owner_for_page(int layer, int expert, size_t * migration_budget) const {
        const size_t id = page_id_for(layer, expert);
        if (id == page_static_owners_.size()) {
            return static_owner_for_page(layer, expert);
        }
        if (!placement_enabled_ || !placement_ready_ || migration_budget == nullptr ||
                resident_layer(layer)) {
            return placement_enabled_ && placement_ready_
                ? (size_t) page_current_owner_[id].load(std::memory_order_relaxed)
                : page_static_owners_[id];
        }

        const std::shared_ptr<const PlacementSnapshot> snapshot =
            std::atomic_load_explicit(&placement_snapshot_, std::memory_order_acquire);
        const size_t current =
            (size_t) page_current_owner_[id].load(std::memory_order_relaxed);
        if (!snapshot || current >= device_names_.size() ||
                id >= snapshot->owner.size()) {
            return current;
        }
        const size_t desired = (size_t) snapshot->owner[id];
        if (desired == current) {
            return current;
        }

        const uint64_t count = page_access_counts_[id].load(std::memory_order_relaxed);
        const uint64_t boundary = current > desired
            ? snapshot->promotion_floor[id] : snapshot->demotion_ceiling[id];
        const uint64_t margin = std::max<uint64_t>(
            8, boundary / 100 * hysteresis_pct_ +
                boundary % 100 * hysteresis_pct_ / 100);
        const bool enough = current > desired
            ? margin <= UINT64_MAX - boundary &&
                  count >= boundary + margin
            : boundary >= margin && count <= boundary - margin;
        if (!enough) {
            placement_declined_hysteresis_.fetch_add(1, std::memory_order_relaxed);
            return current;
        }
        if (*migration_budget == 0) {
            placement_declined_cap_.fetch_add(1, std::memory_order_relaxed);
            return current;
        }
        uint32_t expected = (uint32_t) current;
        if (!page_current_owner_[id].compare_exchange_strong(
                expected, (uint32_t) desired, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return (size_t) expected;
        }
        --*migration_budget;
        placement_migrations_.fetch_add(1, std::memory_order_relaxed);
        return desired;
    }

    bool resident_layer(int layer) const {
        return std::binary_search(
            resident_expert_blocks_.begin(), resident_expert_blocks_.end(), layer);
    }

    void note_dispatch_references(
            const pipe_expert_dispatch_req & request) const {
        if (!multi_device()) {
            return;
        }
        for (const pipe_expert_assignment & assignment : request.assignments) {
            const size_t id = page_id_for(request.layer, assignment.expert_id);
            if (id == page_static_owners_.size()) {
                continue;
            }
            uint64_t count = page_access_counts_[id].load(std::memory_order_relaxed);
            while (count != UINT64_MAX &&
                   !page_access_counts_[id].compare_exchange_weak(
                       count, count + 1, std::memory_order_relaxed,
                       std::memory_order_relaxed)) {
            }
        }
        const uint64_t previous = placement_references_.fetch_add(
            (uint64_t) request.assignments.size(), std::memory_order_relaxed);
        const uint64_t added = (uint64_t) request.assignments.size();
        const uint64_t after = previous > UINT64_MAX - added
            ? UINT64_MAX : previous + added;
        if (previous / placement_refresh_period_ !=
                after / placement_refresh_period_) {
            refresh_placement_snapshot();
        }
        maybe_report_placement_stats();
    }

    void initialize_placement_policy() {
        if (!multi_device()) {
            return;
        }
        bool use_size_classes = true;
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            use_size_classes = use_size_classes && device->resources().size_classes;
        }
        if (use_size_classes) {
            for (const auto & item : catalog_.pages) {
                if (!resident_layer(item.second.layer)) {
                    placement_class_sizes_.push_back(item.second.size);
                }
            }
            std::sort(placement_class_sizes_.begin(), placement_class_sizes_.end());
            placement_class_sizes_.erase(
                std::unique(placement_class_sizes_.begin(), placement_class_sizes_.end()),
                placement_class_sizes_.end());
        } else {
            uint64_t max_page_size = 0;
            for (const auto & item : catalog_.pages) {
                if (!resident_layer(item.second.layer)) {
                    max_page_size = std::max(max_page_size, item.second.size);
                }
            }
            placement_class_sizes_.push_back(max_page_size);
        }
        placement_pages_by_class_.resize(placement_class_sizes_.size());
        for (const auto & item : catalog_.pages) {
            const ExpertPage & page = item.second;
            if (resident_layer(page.layer)) {
                continue;
            }
            const size_t id = (size_t) page.cache_id;
            const size_t class_id = use_size_classes
                ? (size_t) (std::lower_bound(
                      placement_class_sizes_.begin(), placement_class_sizes_.end(),
                      page.size) - placement_class_sizes_.begin())
                : 0;
            placement_pages_by_class_[class_id].push_back(id);
        }
        placement_capacity_.assign(
            placement_class_sizes_.size(), std::vector<size_t>(devices_.size(), 0));
        for (size_t device_id = 0; device_id < devices_.size(); ++device_id) {
            const ResourcePlan & resources = devices_[device_id]->resources();
            for (size_t class_id = 0; class_id < placement_class_sizes_.size(); ++class_id) {
                for (const SlotClass & slot_class : resources.slot_classes) {
                    if ((use_size_classes && slot_class.size == placement_class_sizes_[class_id]) ||
                            (!use_size_classes && slot_class.size >= placement_class_sizes_[class_id])) {
                        placement_capacity_[class_id][device_id] =
                            (size_t) slot_class.slots;
                        break;
                    }
                }
            }
        }
        placement_ready_ = true;
        refresh_placement_snapshot();
    }

    void refresh_placement_snapshot() const {
        if (!placement_ready_ || !placement_enabled() ||
                placement_refreshing_.test_and_set(std::memory_order_acquire)) {
            return;
        }
        struct RefreshGuard {
            std::atomic_flag & flag;
            ~RefreshGuard() { flag.clear(std::memory_order_release); }
        } refresh_guard{placement_refreshing_};
        std::shared_ptr<PlacementSnapshot> snapshot =
            std::make_shared<PlacementSnapshot>();
        const size_t n_pages = page_static_owners_.size();
        snapshot->owner.resize(n_pages);
        snapshot->promotion_floor.resize(n_pages);
        snapshot->demotion_ceiling.resize(n_pages);
        std::vector<uint64_t> counts(n_pages, 0);
        for (size_t id = 0; id < n_pages; ++id) {
            counts[id] = page_access_counts_[id].load(std::memory_order_relaxed);
        }
        for (size_t class_id = 0; class_id < placement_pages_by_class_.size(); ++class_id) {
            std::vector<size_t> pages = placement_pages_by_class_[class_id];
            std::sort(pages.begin(), pages.end(), [&](size_t a, size_t b) {
                if (counts[a] != counts[b]) {
                    return counts[a] > counts[b];
                }
                if (page_static_owners_[a] != page_static_owners_[b]) {
                    return page_static_owners_[a] < page_static_owners_[b];
                }
                return a < b;
            });
            std::vector<size_t> band_begin(device_names_.size(), pages.size());
            std::vector<size_t> band_end(device_names_.size(), pages.size());
            size_t cursor = 0;
            for (size_t device_id = 0; device_id < device_names_.size(); ++device_id) {
                band_begin[device_id] = cursor;
                const size_t take = std::min(
                    placement_capacity_[class_id][device_id], pages.size() - cursor);
                cursor += take;
                band_end[device_id] = cursor;
                for (size_t rank = band_begin[device_id]; rank < band_end[device_id]; ++rank) {
                    snapshot->owner[pages[rank]] = (uint32_t) device_id;
                }
            }
            if (cursor < pages.size()) {
                size_t device_id = device_names_.size();
                for (size_t i = device_names_.size(); i-- > 0;) {
                    if (placement_capacity_[class_id][i] != 0) {
                        device_id = i;
                        break;
                    }
                }
                if (device_id == device_names_.size()) {
                    for (size_t rank = cursor; rank < pages.size(); ++rank) {
                        snapshot->owner[pages[rank]] =
                            (uint32_t) page_static_owners_[pages[rank]];
                    }
                    continue;
                }
                if (band_begin[device_id] == pages.size()) {
                    band_begin[device_id] = cursor;
                }
                band_end[device_id] = pages.size();
                for (size_t rank = cursor; rank < pages.size(); ++rank) {
                    snapshot->owner[pages[rank]] = (uint32_t) device_id;
                }
                for (size_t i = device_id + 1; i < device_names_.size(); ++i) {
                    band_begin[i] = pages.size();
                    band_end[i] = pages.size();
                }
            }
            for (size_t device_id = 0; device_id < device_names_.size(); ++device_id) {
                if (band_begin[device_id] == pages.size()) {
                    continue;
                }
                const uint64_t first = counts[pages[band_begin[device_id]]];
                const uint64_t last = counts[pages[band_end[device_id] - 1]];
                for (size_t rank = band_begin[device_id]; rank < band_end[device_id]; ++rank) {
                    snapshot->promotion_floor[pages[rank]] = last;
                    snapshot->demotion_ceiling[pages[rank]] = first;
                }
            }
        }
        std::atomic_store_explicit(
            &placement_snapshot_, std::shared_ptr<const PlacementSnapshot>(snapshot),
            std::memory_order_release);
    }

    bool placement_enabled() const {
        return placement_enabled_;
    }

    void maybe_report_placement_stats() const {
        if (!placement_ready_ || placement_interval_ns_ == 0) {
            return;
        }
        const uint64_t now = placement_now_ns();
        uint64_t next = placement_next_report_ns_.load(std::memory_order_relaxed);
        if (now < next || !placement_next_report_ns_.compare_exchange_strong(
                next, now > UINT64_MAX - placement_interval_ns_ ? UINT64_MAX :
                    now + placement_interval_ns_, std::memory_order_relaxed)) {
            return;
        }
        std::vector<size_t> page_counts(device_names_.size(), 0);
        for (size_t id = 0; id < page_static_owners_.size(); ++id) {
            const size_t owner = (size_t) page_current_owner_[id].load(
                std::memory_order_relaxed);
            if (owner < page_counts.size()) {
                ++page_counts[owner];
            }
        }
        std::fprintf(stderr,
                     "wp expert worker placement stats enabled=%d page_counts=[",
                     placement_enabled_ ? 1 : 0);
        for (size_t i = 0; i < page_counts.size(); ++i) {
            if (i != 0) {
                std::fputc(',', stderr);
            }
            std::fprintf(stderr, "%zu", page_counts[i]);
        }
        std::fprintf(stderr,
                     "] migrations=%llu declined_cap=%llu declined_hysteresis=%llu "
                     "migration_cap=%zu hysteresis_pct=%llu references=%llu\n",
                     (unsigned long long) placement_migrations_.load(std::memory_order_relaxed),
                     (unsigned long long) placement_declined_cap_.load(std::memory_order_relaxed),
                     (unsigned long long) placement_declined_hysteresis_.load(
                         std::memory_order_relaxed),
                     migration_cap_,
                     (unsigned long long) hysteresis_pct_,
                     (unsigned long long) placement_references_.load(
                         std::memory_order_relaxed));
        std::fflush(stderr);
        dump_access_counts();
    }

    // WP_EXPERT_COUNTS_DUMP=<path>: at every placement-stats interval, write the
    // learned per-expert access counts, hottest first, as pin-file-compatible
    // "layer expert  # count" lines (tmp + atomic rename). Feed the result back
    // via WP_EXPERT_PIN_FILE so a fresh worker starts with the hot set resident
    // instead of re-learning placement from zero after every restart -- access
    // counts die with the process, which is why LFU tiering never converges on
    // a restart-heavy rig.
    void dump_access_counts() const {
        static const char * const path = std::getenv("WP_EXPERT_COUNTS_DUMP");
        if (path == nullptr || path[0] == '\0' || !placement_ready_) {
            return;
        }
        struct Row { uint64_t count; int layer; int expert; };
        std::vector<Row> rows;
        rows.reserve(page_static_owners_.size());
        for (const auto & kv : catalog_.pages) {
            const auto cid = kv.second.cache_id;
            if (cid < 0 || (size_t) cid >= page_static_owners_.size()) {
                continue;
            }
            const uint64_t c = page_access_counts_[cid].load(std::memory_order_relaxed);
            if (c == 0) {
                continue;
            }
            rows.push_back({ c, kv.first.first, kv.first.second });
        }
        std::sort(rows.begin(), rows.end(), [](const Row & a, const Row & b) {
            if (a.count != b.count) { return a.count > b.count; }
            if (a.layer != b.layer) { return a.layer < b.layer; }
            return a.expert < b.expert;
        });
        const std::string tmp = std::string(path) + ".tmp";
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            return;
        }
        for (const Row & r : rows) {
            out << r.layer << ' ' << r.expert << "  # " << r.count << '\n';
        }
        out.close();
        if (out) {
            std::rename(tmp.c_str(), path);
        }
    }

    struct AssignmentGroup {
        size_t device = 0;
        size_t begin = 0;
        size_t end = 0;
    };

    struct SplitPending {
        pipe_expert_dispatch_begin begin;
        uint64_t seq_id = 0;
        std::vector<AssignmentGroup> groups;
    };

    std::vector<AssignmentGroup> assignment_groups(
            const pipe_expert_dispatch_req & request,
            size_t * migration_budget = nullptr) const {
        std::vector<AssignmentGroup> result;
        if (request.assignments.empty()) {
            return result;
        }
        size_t begin = 0;
        size_t owner = owning_device_for_page(
            request.layer, request.assignments.front().expert_id, migration_budget);
        for (size_t i = 1; i <= request.assignments.size(); ++i) {
            const size_t next = i == request.assignments.size() ? owner :
                owning_device_for_page(request.layer, request.assignments[i].expert_id,
                                       migration_budget);
            if (i == request.assignments.size() || next != owner) {
                result.push_back({owner, begin, i});
                begin = i;
                owner = next;
            }
        }
        return result;
    }

    pipe_expert_dispatch_req make_subrequest(
            const pipe_expert_dispatch_req & request, size_t begin, size_t end) const {
        pipe_expert_dispatch_req result;
        result.layer = request.layer;
        result.n_tokens = request.n_tokens;
        result.swiglu_clamp = request.swiglu_clamp;
        result.assignments.assign(request.assignments.begin() + (ptrdiff_t) begin,
                                  request.assignments.begin() + (ptrdiff_t) end);
        result.activations = request.activations;
        return result;
    }

    void validate_dispatch(const pipe_expert_dispatch_req & request) const {
        if (!std::binary_search(catalog_.layers.begin(), catalog_.layers.end(), request.layer)) {
            throw pipe_protocol_error(
                PIPE_ERR_EXPERT_LAYER,
                "worker does not serve layer " + std::to_string(request.layer));
        }
        if (request.assignments.size() >
                (size_t) catalog_.descriptor.hparams.n_expert) {
            throw pipe_protocol_error(
                PIPE_ERR_BAD_FRAME,
                "expert dispatch has more assignments than model experts");
        }
        std::set<int32_t> seen_experts;
        for (const pipe_expert_assignment & assignment : request.assignments) {
            if (!seen_experts.insert(assignment.expert_id).second) {
                throw pipe_protocol_error(
                    PIPE_ERR_BAD_FRAME,
                    "expert dispatch repeats expert " + std::to_string(assignment.expert_id));
            }
            if (assignment.expert_id < catalog_.descriptor.expert_first ||
                    assignment.expert_id > catalog_.descriptor.expert_last ||
                    catalog_.pages.count({ request.layer, assignment.expert_id }) == 0) {
                throw pipe_protocol_error(
                    PIPE_ERR_EXPERT_RANGE,
                    "worker does not serve expert " + std::to_string(assignment.expert_id));
            }
        }
    }

    void load_pin_file() {
        const char * const pin_path = std::getenv("WP_EXPERT_PIN_FILE");
        if (pin_path == nullptr || pin_path[0] == '\0') {
            return;
        }
        size_t pin_budget = 0;
        for (const std::unique_ptr<DeviceWorker> & device : devices_) {
            pin_budget += (size_t) device->resources().slot_count;
        }
        if (const char * max_env = std::getenv("WP_EXPERT_PIN_MAX_SLOTS")) {
            const long long parsed = std::strtoll(max_env, nullptr, 10);
            pin_budget = parsed > 0 ? (size_t) parsed : 0;
        }
        const size_t pin_class_pct = expert_pin_class_pct_from_env();
        std::vector<std::vector<size_t>> pin_class_pinned(devices_.size());
        std::vector<std::vector<size_t>> pin_class_skipped(devices_.size());
        for (size_t i = 0; i < devices_.size(); ++i) {
            const ResourcePlan & resources = devices_[i]->resources();
            pin_class_pinned[i].assign(resources.slot_classes.size(), 0);
            pin_class_skipped[i].assign(resources.slot_classes.size(), 0);
        }
        std::ifstream pin_file(pin_path);
        if (!pin_file) {
            throw std::runtime_error(
                "failed to open WP_EXPERT_PIN_FILE: " + std::string(pin_path));
        }
        std::vector<std::vector<std::pair<int, int>>> pages(devices_.size());
        std::set<std::pair<int, int>> seen;
        std::string line;
        size_t line_number = 0;
        size_t selected = 0;
        bool truncated = false;
        while (std::getline(pin_file, line)) {
            ++line_number;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream input(line);
            int layer = -1;
            int expert = -1;
            if (!(input >> layer >> expert)) {
                std::cerr << "WARN wp expert worker: ignoring malformed pin line "
                          << line_number << std::endl;
                continue;
            }
            const std::pair<int, int> key = {layer, expert};
            if (!seen.insert(key).second) {
                continue;
            }
            if (catalog_.pages.count(key) == 0) {
                std::cerr << "WARN wp expert worker: ignoring unknown pin "
                          << layer << " " << expert << std::endl;
                continue;
            }
            const size_t device_id = owning_device_for_page(layer, expert);
            const ResourcePlan & resources = devices_[device_id]->resources();
            const size_t class_id = expert_pin_class_index(resources, catalog_.pages.at(key).size);
            if (class_id < pin_class_pinned[device_id].size() &&
                    pin_class_pinned[device_id][class_id] >=
                        (size_t) resources.slot_classes[class_id].slots * pin_class_pct / 100) {
                ++pin_class_skipped[device_id][class_id];
                continue;
            }
            if (selected >= pin_budget) {
                truncated = true;
                continue;
            }
            pages[device_id].push_back(key);
            if (class_id < pin_class_pinned[device_id].size()) {
                ++pin_class_pinned[device_id][class_id];
            }
            ++selected;
        }
        size_t loaded = 0;
        for (size_t i = 0; i < pages.size(); ++i) {
            std::lock_guard<std::mutex> lock(device_mutexes_[i]);
            loaded += devices_[i]->pin_pages(pages[i], pages[i].size());
        }
        std::cerr << "WARN wp expert worker: pin_file=" << pin_path
                  << " n_pinned=" << loaded
                  << " pin_budget=" << pin_budget
                  << " pin_class_pct=" << pin_class_pct
                  << " demand_hits=0" << std::endl;
        for (size_t i = 0; i < devices_.size(); ++i) {
            const ResourcePlan & resources = devices_[i]->resources();
            for (size_t j = 0; j < resources.slot_classes.size(); ++j) {
                const SlotClass & slot_class = resources.slot_classes[j];
                std::cerr << "WARN wp expert worker: pin_class device=" << device_names_[i]
                          << " bytes=" << slot_class.size
                          << " slots=" << slot_class.slots
                          << " cap=" << (size_t) slot_class.slots * pin_class_pct / 100
                          << " pinned=" << pin_class_pinned[i][j]
                          << " skipped=" << pin_class_skipped[i][j] << std::endl;
            }
        }
        if (truncated) {
            std::cerr << "WARN wp expert worker: pin file truncated in file order"
                      << " at " << pin_budget << " slots" << std::endl;
        }
    }

    Catalog catalog_;
    std::vector<std::string> device_names_;
    std::vector<int> device_slots_;
    WorkerLogFiles logs_;
    wp::HostTier host_tier_;
    const bool placement_enabled_ = lfu_placement_from_env();
    const size_t migration_cap_ = lfu_migration_cap_from_env();
    const uint64_t hysteresis_pct_ = lfu_hysteresis_pct_from_env();
    static constexpr uint64_t placement_refresh_period_ = 256;
    const uint64_t placement_interval_ms_ = read_stats_interval_ms_from_env();
    const uint64_t placement_interval_ns_ = placement_interval_ms_ * 1000000ull;
    mutable std::atomic<uint64_t> placement_next_report_ns_{
        placement_interval_ns_ == 0 ? std::numeric_limits<uint64_t>::max() :
            placement_now_ns() + placement_interval_ns_};
    mutable std::atomic<uint64_t> placement_references_{0};
    mutable std::atomic<uint64_t> placement_migrations_{0};
    mutable std::atomic<uint64_t> placement_declined_cap_{0};
    mutable std::atomic<uint64_t> placement_declined_hysteresis_{0};
    mutable std::atomic_flag placement_refreshing_ = ATOMIC_FLAG_INIT;
    std::unique_ptr<std::atomic<uint64_t>[]> page_access_counts_;
    std::unique_ptr<std::atomic<uint32_t>[]> page_current_owner_;
    std::vector<size_t> page_static_owners_;
    std::vector<int> resident_expert_blocks_;
    std::vector<uint64_t> placement_class_sizes_;
    std::vector<std::vector<size_t>> placement_pages_by_class_;
    std::vector<std::vector<size_t>> placement_capacity_;
    mutable std::shared_ptr<const PlacementSnapshot> placement_snapshot_;
    bool placement_ready_ = false;
    std::vector<std::unique_ptr<DeviceWorker>> devices_;
    mutable std::vector<std::mutex> device_mutexes_;

    // *** WP_DEVICE_PARALLEL -- RUN THE DEVICES CONCURRENTLY (DEFAULT ON). ***
    //
    // PERSISTENT THREAD PER DEVICE, not a thread per dispatch. The parallel
    // path buckets groups by device, runs GPU devices concurrently, and keeps
    // one executor thread per device. Vulkan command pools have thread
    // affinity, and HIP device work also must stay on the same thread. A
    // device must not migrate between the caller and another thread.
    //
    // BIT-EXACTNESS IS PRESERVED. Partials are collected into a group-indexed
    // vector and folded into `result` sequentially in GROUP ORDER, exactly as
    // in the serial loop. Only the compute is reordered; the floating-point
    // association is not.
    //
    // The CPU expert tier runs AFTER the GPUs. ggml's CPU backend spin-waits at
    // its thread barriers, so running it with the GPUs starves their host-side
    // submission threads. Keeping the CPU tier on its own persistent executor
    // thread, but outside the concurrent GPU window, gives the intended layer
    // cost of max(GPUs) + CPU tier.
    //
    // WP_CPU_TIER_OVERLAP=1 (default OFF) folds the CPU tier back INTO the
    // concurrent window, which is only safe because the tier's backend is then
    // bound to a CPU-confined SCHED_BATCH threadpool -- see
    // configure_cpu_backend() for the whole argument, including why ggml's
    // `poll` knob cannot help on this OpenMP build.
    //
    // POST-FIX MEASUREMENTS (2026-08-29), sliced rig, code cd787a2b7,
    // identical config, only the env var changed; eslice-measure.sh protocol,
    // 1040-tok prompts, 256-tok decodes:
    //   serial:   prefill 24.8/24.9 t/s   decode 3.58/3.57 t/s
    //   parallel: prefill 53.7/55.5 t/s   decode 5.97/6.23 t/s
    // Decode-only per-device dispatch, from deltas between stat banners:
    //   2026: CUDA0 0.63 ms, Vulkan0 0.97 ms, CPU 0.24 ms
    //   main:  ROCm0 0.94 ms, ROCm1 0.75 ms, CPU 0.27 ms
    // Layer cost = max(GPUs) + CPU tier, as designed.
    // Same-day config win: dispatching to 2026 over its LAN IP instead of
    // Tailscale changed RTT 0.94 -> 0.28 ms.
    // Same-day config win: WP_CPU_THREADS 4 -> 2 on the 4-core i7-6700K
    // changed Vulkan submit 389 -> 254 us and decode 6.47 -> 7.4 t/s.
    // Both were CPU starvation on that box.
    struct DeviceExecutor {
        std::thread             thread;
        std::mutex              mu;
        std::condition_variable cv_task;
        std::condition_variable cv_done;
        std::function<void()>   task;
        std::exception_ptr      error;
        bool                    has_task = false;
        bool                    idle     = true;
        bool                    stop     = false;
        // Lock-free mirror of `idle`, published with release ordering AFTER the
        // mutex-protected store so a spinning joiner can poll it without taking
        // `mu` on every iteration. The old spin locked and unlocked `mu` up to
        // 20000 times per join, contending with the very worker thread it was
        // waiting for -- the joiner's own polling could delay the completion
        // store it was polling for. Correctness still rests on the CV wait
        // below; this is purely a fast path.
        std::atomic<bool>       idle_flag{true};

        // `pin` is true ONLY for the CPU expert tier's executor. See
        // wp_cpu_tier_pin_self(): at WP_CPU_TIER_THREADS=1 ggml never enters an
        // OpenMP region and so never applies the threadpool's cpumask/priority,
        // and this is the only place the tier's thread can be placed at all.
        void start(bool pin = false) {
            thread = std::thread([this, pin] {
                if (pin) {
                    wp_cpu_tier_pin_self();
                }
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mu);
                        cv_task.wait(lock, [this] { return has_task || stop; });
                        if (stop) { return; }
                        job = std::move(task);
                        has_task = false;
                    }
                    std::exception_ptr caught;
                    try {
                        job();
                    } catch (...) {
                        caught = std::current_exception();
                    }
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        error = caught;
                        idle  = true;
                    }
                    idle_flag.store(true, std::memory_order_release);
                    cv_done.notify_all();
                }
            });
        }
        void submit(std::function<void()> job) {
            {
                std::lock_guard<std::mutex> lock(mu);
                task     = std::move(job);
                has_task = true;
                idle     = false;
                idle_flag.store(false, std::memory_order_relaxed);
                error    = nullptr;
            }
            cv_task.notify_one();
        }
        void run(std::function<void()> job) {
            {
                std::unique_lock<std::mutex> lock(mu);
                cv_done.wait(lock, [this] { return idle; });
                task     = std::move(job);
                has_task = true;
                idle     = false;
                idle_flag.store(false, std::memory_order_relaxed);
                error    = nullptr;
            }
            cv_task.notify_one();
            join_one();
        }
        // Rethrows on the CALLING thread, so a device fault still aborts the
        // request the same way the serial loop's exception did.
        void join_one() {
            // SPIN BRIEFLY BEFORE SLEEPING. A device group is ~200-900 us of
            // work; a condition_variable round trip is 5-50 us under load, and
            // the first build paid one per device per request (~171k of them in
            // a decode window) to parallelise work of the same order. Spin for a
            // bounded window first so the common case never sleeps at all, then
            // fall back to the CV so a long group (a cold page-in) still parks
            // the thread instead of burning a core.
            //
            // WP_EXEC_SPIN tunes that window (default 20000 = unchanged; 0
            // disables the spin and goes straight to the CV). It exists for
            // WP_CPU_TIER_OVERLAP: this spin makes the CALLING thread runnable
            // for the whole leg, and on the 4c/8t box adding an overlapped CPU
            // tier to that is one more contender than the GPU submission
            // threads may have room for. If overlap measures worse than it
            // should on 2026, try WP_EXEC_SPIN=0 before giving up on it.
            static const int spin_rounds = [] {
                const char * e = std::getenv("WP_EXEC_SPIN");
                if (e == nullptr) { return 20000; }
                const long v = std::strtol(e, nullptr, 10);
                return v < 0 ? 0 : (int) v;
            }();
            for (int i = 0; i < spin_rounds; ++i) {
                if (idle_flag.load(std::memory_order_acquire)) { break; }
                std::this_thread::yield();
            }
            std::unique_lock<std::mutex> lock(mu);
            cv_done.wait(lock, [this] { return idle; });
            if (error) {
                std::exception_ptr e = error;
                error = nullptr;
                lock.unlock();
                std::rethrow_exception(e);
            }
        }
        ~DeviceExecutor() {
            if (!thread.joinable()) { return; }
            { std::lock_guard<std::mutex> lock(mu); stop = true; }
            cv_task.notify_all();
            thread.join();
        }
    };
    mutable std::vector<std::unique_ptr<DeviceExecutor>> device_exec_;
    const bool device_parallel_ = [] {
        const char * e = std::getenv("WP_DEVICE_PARALLEL");
        return e == nullptr || std::strcmp(e, "0") != 0;
    }();
    mutable std::mutex split_mutex_;
    std::unordered_map<int, SplitPending> split_pending_by_conn_;
};

bool validate_client_hello(
        const pipe_expert_hello & client, const pipe_expert_hello & worker,
        std::string & error) {
    if (client.role != PIPE_EXPERT_ROLE_CLIENT) {
        error = "peer did not identify as an expert client";
    } else if (client.hidden_type != worker.hidden_type ||
               client.n_embd != worker.n_embd ||
               client.n_ff_exp != worker.n_ff_exp ||
               client.n_expert != worker.n_expert ||
               client.n_expert_used != worker.n_expert_used) {
        error = "expert HELLO hparams mismatch";
    } else if (client.model_identity != worker.model_identity) {
        error = "expert HELLO model identity mismatch";
    } else if (client.shard_identity != worker.shard_identity) {
        error = "expert HELLO shard identity mismatch";
    }
    return error.empty();
}

bool send_hello_ack(
        pipe_socket_t & socket, bool accepted, const std::string & reason) {
    const std::vector<uint8_t> payload =
        pipe_encode_expert_hello_ack({ accepted, reason });
    return pipe_send_frame(
        socket, PIPE_EXPERT_HELLO_ACK, 0, payload.data(), payload.size());
}

#if defined(__linux__)
// ---------------------------------------------------------------------------
// WP_DISPATCH_DEDUP_ACTIVATIONS: POSIX shm rendezvous between worker
// PROCESSES that share a machine. See the PIPE_VERSION 14 comment in
// pipe-protocol.h for the wire-level design and dedup_publish_and_ref() in
// pipe-expert-dispatcher.cpp for the ordering guarantee this relies on (a REF
// frame is never sent by the spine before the matching PUBLISH's ack is
// received there) -- that ordering is why `subscribe` below can treat its
// poll loop as defense in depth rather than the primary correctness
// mechanism: by construction the segment already exists and is READY by the
// time any ACTS_REF naming it can possibly arrive here.
namespace wp_dedup {

struct ShmHeader {
    std::atomic<uint32_t> state;      // 0 = writing, 1 = ready
    std::atomic<int32_t>  remaining;  // holders left to check in; last one unlinks
    uint32_t              n_tokens;
    uint32_t              n_embd;
};

constexpr uint32_t STATE_WRITING = 0;
constexpr uint32_t STATE_READY   = 1;
// Bounded wait for a segment to appear/become ready. Generous relative to a
// same-host mmap + memcpy, tiny relative to the tens-of-seconds prefill this
// mechanism targets -- a worker stuck here for the full window is reporting a
// genuine local fault, not slow hardware.
constexpr auto SHM_WAIT_BUDGET = std::chrono::milliseconds(500);

std::string segment_name(uint64_t seq_id, int32_t layer) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/wp_acts_%016llx_%d",
                  (unsigned long long) seq_id, layer);
    return std::string(buf);
}

// Publish `activations` so `n_subscribers` co-located siblings can read them
// without their own wire copy. Never throws: a publish failure is reported to
// the spine via PIPE_EXPERT_ACTS_PUBLISH_ACK{success=false} and the caller
// still computes its OWN partial from the activations it already holds in
// memory regardless of the return value here -- see the ACTS_PUBLISH frame
// handler below. A publish failure therefore costs a sibling's fallback
// round trip, never this worker's own correctness.
bool publish(uint64_t seq_id, int32_t layer, const std::vector<float> & activations,
            uint32_t n_tokens, uint32_t n_embd, uint32_t n_subscribers, std::string & error) {
    const std::string name = segment_name(seq_id, layer);
    // O_EXCL: a name collision means a segment from a crashed prior run
    // (seq_id/layer pairs are not expected to repeat within a spine's
    // lifetime, but "not expected" is not a guarantee across a spine
    // restart that reuses a low seq_id). One unlink+retry clears garbage
    // rather than silently reusing or corrupting someone else's segment.
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0 && errno == EEXIST) {
        shm_unlink(name.c_str());
        fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    }
    if (fd < 0) {
        error = std::string("shm_open: ") + std::strerror(errno);
        return false;
    }
    const size_t total = sizeof(ShmHeader) + (size_t) n_tokens * (size_t) n_embd * sizeof(float);
    if (ftruncate(fd, (off_t) total) != 0) {
        error = std::string("ftruncate: ") + std::strerror(errno);
        close(fd);
        shm_unlink(name.c_str());
        return false;
    }
    void * mapped = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); // the mapping keeps the segment usable regardless of the fd
    if (mapped == MAP_FAILED) {
        error = std::string("mmap: ") + std::strerror(errno);
        shm_unlink(name.c_str());
        return false;
    }
    ShmHeader * hdr = reinterpret_cast<ShmHeader *>(mapped);
    hdr->n_tokens = n_tokens;
    hdr->n_embd   = n_embd;
    // Plain writes here are fine: nothing else can see this segment's
    // contents until the release-store of state below, which is the
    // synchronization point a subscriber's acquire-load pairs with.
    hdr->remaining.store((int32_t) n_subscribers + 1, std::memory_order_relaxed);
    std::memcpy(reinterpret_cast<uint8_t *>(mapped) + sizeof(ShmHeader),
               activations.data(), activations.size() * sizeof(float));
    hdr->state.store(STATE_READY, std::memory_order_release);
    munmap(mapped, total); // the NAME, not this mapping, is what matters now
    return true;
}

// Called by whichever holder (the publisher or a subscriber) finishes with a
// segment last. Decrements the shared refcount seeded in publish() and
// unlinks once nobody is left. Best-effort: a missing segment (already
// unlinked+fully released) is not an error.
void release(uint64_t seq_id, int32_t layer) {
    const std::string name = segment_name(seq_id, layer);
    const int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        return;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || (size_t) st.st_size < sizeof(ShmHeader)) {
        close(fd);
        return;
    }
    void * mapped = mmap(nullptr, sizeof(ShmHeader), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        return;
    }
    ShmHeader * hdr = reinterpret_cast<ShmHeader *>(mapped);
    const int32_t left = hdr->remaining.fetch_sub(1, std::memory_order_acq_rel) - 1;
    munmap(mapped, sizeof(ShmHeader));
    if (left <= 0) {
        shm_unlink(name.c_str());
    }
}

// Subscribe (secondary side): bounded poll for the segment to exist and reach
// STATE_READY, then copy its payload out to a plain vector so the caller's
// compute path never touches the shm mapping again. Returns nullopt on ANY
// failure -- the caller MUST treat that as "activations unavailable" and
// never compute on a partial, stale, or default-initialised buffer; see the
// ACTS_REF frame handler below, which is the only place a nullopt here is
// allowed to turn into anything other than closing the connection.
std::optional<std::vector<float>> subscribe(uint64_t seq_id, int32_t layer, uint32_t n_tokens,
                                            uint32_t n_embd, std::string & error) {
    const std::string name = segment_name(seq_id, layer);
    const auto deadline = std::chrono::steady_clock::now() + SHM_WAIT_BUDGET;
    int fd = -1;
    for (;;) {
        fd = shm_open(name.c_str(), O_RDONLY, 0600);
        if (fd >= 0) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            error = "shm segment never appeared";
            return std::nullopt;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    struct stat st{};
    const size_t expect = sizeof(ShmHeader) + (size_t) n_tokens * (size_t) n_embd * sizeof(float);
    if (fstat(fd, &st) != 0 || (size_t) st.st_size != expect) {
        close(fd);
        error = "shm segment has the wrong size";
        return std::nullopt;
    }
    void * mapped = mmap(nullptr, expect, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        error = std::string("mmap: ") + std::strerror(errno);
        return std::nullopt;
    }
    const ShmHeader * hdr = reinterpret_cast<const ShmHeader *>(mapped);
    bool ready = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (hdr->state.load(std::memory_order_acquire) == STATE_READY) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!ready || hdr->n_tokens != n_tokens || hdr->n_embd != n_embd) {
        munmap(mapped, expect);
        error = "shm segment never became ready or has the wrong shape";
        return std::nullopt;
    }
    std::vector<float> out((size_t) n_tokens * (size_t) n_embd);
    std::memcpy(out.data(), reinterpret_cast<const uint8_t *>(mapped) + sizeof(ShmHeader),
               out.size() * sizeof(float));
    munmap(mapped, expect);
    return out;
}

} // namespace wp_dedup
#endif // __linux__

// WP_WORKER_MULTI_CONN=N (N>=2) -- production worker double-buffering,
// lever-queue item #9. A throwaway N=2, one-coarse-mutex probe (2026-08-22)
// measured 126.9 aggregate rps vs 66.9 single-stream (+90%) with per-request
// latency up only 14.6->15.4ms, fed by two replay streams under a lock that
// covered the ENTIRE per-request handling body including the network send.
// This is the real version: same coarse GPU-section lock (see the
// correctness argument below -- it is intentional, not left over), but
// narrowed to stop covering the response encode + socket send, so a
// connection's own network I/O can now overlap the OTHER connection's GPU
// work instead of serializing behind it too. Persistent serving (run()
// below runs N acceptor threads in a loop, not accept-N-then-exit) replaces
// the probe's accept-exactly-N.
//
// Unset or "1" (or any WP_WORKER_MULTI_CONN<2) is the byte-identical
// default path: single accept(), g_worker_gpu_mutex stays null, every check
// against it below is one branch on a null pointer, no lock ever taken.
//
// WHY STILL ONE COARSE LOCK AROUND THE WORKER-TOUCHING WORK (not per-
// substructure locks): worker.dispatch() (and begin/finish_split_dispatch,
// which wrap it) is ONE call that walks ExpertSlotPool pool_ lookup ->
// page-in read -> H2D -> the shared ggml_gallocr* compute_galloc_ graph
// build/alloc -> GPU submit -> readback, all as one interleaved sequence
// with no safe seam already exposed at the API boundary (pool_.ensure_batch
// starts page-in reads AND touches residency/LRU bookkeeping in the same
// call that also returns slots pinned for the compute that follows). GPU
// submission must be serialized regardless (single device, single command
// stream) and neither pool_, compute_galloc_, nor the resident_
// (ResidentExpertPool -- pinned-page tracking, see Worker::resident_ /
// pinned_pages()) residency state are documented or evidenced anywhere in
// this file as safe for concurrent access -- they are mutated throughout
// dispatch(), not just at the edges. Splitting "the NVMe read part" from
// "the GPU compute part" so they can run under separate locks would mean
// restructuring ExpertSlotPool's batch/read/compute pipeline into
// interruptible phases, which is exactly the slot-pool redesign this task
// is scoped to NOT do. So: one mutex, g_worker_gpu_mutex, covers every
// Worker-state-touching call (dispatch/begin/finish/abandon_split_dispatch,
// note_prefetch_hint(_bad), log_reference, report_prefetch_hints,
// record_stats) -- correctness over cleverness, per the campaign brief;
// the +90% already measured with a FULLY coarse lock (covering network I/O
// too) means the overlap this narrower lock adds on top (recv of the next
// request, and send of the previous response, now able to run concurrently
// with another connection's GPU-lock-holding dispatch) is a strict
// improvement over what was already a decisive win, not a bet on more.
std::mutex * g_worker_gpu_mutex = nullptr;

// *** WP_FRAME_TRACE=1 -- PER-FRAME WORKER RESIDENCY. ***
//
// THE GAP THIS EXISTS TO CLOSE (2026-08-29). The spine reports waiting
// 1.66-1.78 ms per layer on the LOOPBACK worker (127.0.0.1:8801), while that
// worker's own ns_compute is 814 us. ~890 us/layer -- ~43 ms/token -- is
// unaccounted, on the same machine, with no network in the path. Neither side
// measured that boundary: the worker's counters all start inside dispatch(),
// and the spine's stop at the socket.
//
// This brackets EVERY frame from "pipe_recv_frame returned" to "this loop
// iteration finished", by RAII, so continue/return/throw all count. It is the
// worker-side half of the spine's wait, so:
//     spine_wait_per_layer - sum(residency of that layer's frames)
//         == transport + spine-side queueing
// which is the number that decides whether to attack the wire or the worker.
//
// It also settles, without reading any more code, WHICH wire path is live:
// split (BEGIN + ACTS, two frames per layer) or single DISPATCH_REQ. ns_send
// reads 0 on every device, which implies the plain branch never runs -- this
// prints the frame mix and proves it.
//
// COST: two steady_clock reads per frame (~50 ns) against a 1.7 ms budget, and
// one mutex-guarded map update. Report every 20k frames, NOT per frame -- an
// unbuffered stderr write per request is a syscall in the hot path and would
// tax the very thing it measures (see the grouped-gemv WARN, same file).
// *** THREAD-LOCAL, NO LOCK. ***
// The first version of this took a std::mutex and updated a std::map on EVERY
// frame, and it cost ~5% of decode throughput: 6.277/6.561/6.653 -> 5.756/
// 6.193/6.331 t/s, with the low sample BELOW the established 6.14-6.63 band.
// That is a probe distorting the thing it measures. Judge an instrument by work
// COMPLETED, not by elapsed time. Fixed rather than kept-with-a-caveat, because
// the residency number it produces is the one the whole investigation turns on.
// Single connection = single serve thread, so thread-local needs no lock at all;
// under WP_WORKER_MULTI_CONN each thread simply reports its own line.
constexpr int FRAME_SLOTS = 24;

bool frame_trace_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("WP_FRAME_TRACE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

const char * frame_type_name(int t) {
    switch (t) {
        case PIPE_PING:                         return "PING";
        case PIPE_EXPERT_DISPATCH_REQ:          return "DISPATCH_REQ";
        case PIPE_EXPERT_PREFETCH_HINT:         return "PREFETCH_HINT";
        case PIPE_EXPERT_DISPATCH_BEGIN:        return "BEGIN";
        case PIPE_EXPERT_DISPATCH_ACTS:         return "ACTS";
        case PIPE_EXPERT_DISPATCH_ACTS_PUBLISH: return "ACTS_PUBLISH";
        case PIPE_EXPERT_DISPATCH_ACTS_REF:     return "ACTS_REF";
        default:                                return "other";
    }
}

void note_frame_residency(int type, uint64_t ns) {
    if (type < 0 || type >= FRAME_SLOTS) { type = 0; }
    static thread_local uint64_t n_by_type[FRAME_SLOTS]  = {};
    static thread_local uint64_t ns_by_type[FRAME_SLOTS] = {};
    // WINDOWED, not cumulative. The cumulative average is dominated by the
    // prefill frames at the head of a run and decays for the rest of it
    // (measured 3532 -> 2757 -> 2441 us on one arm, all one steady state);
    // reading it required differencing consecutive reports by hand. Report the
    // WINDOW so the printed number is the number.
    static thread_local uint64_t win_n[FRAME_SLOTS]  = {};
    static thread_local uint64_t win_ns[FRAME_SLOTS] = {};
    static thread_local uint64_t since = 0;
    ++n_by_type[type];  ns_by_type[type] += ns;
    ++win_n[type];      win_ns[type]     += ns;
    if (++since < 20000) {
        return;
    }
    since = 0;
    std::string line = "wp frame-residency (window):";
    for (int t = 0; t < FRAME_SLOTS; ++t) {
        if (win_n[t] == 0) { continue; }
        char buf[224];
        std::snprintf(buf, sizeof(buf), " %s[n=%llu avg_us=%.1f | life n=%llu avg_us=%.1f]",
                      frame_type_name(t),
                      (unsigned long long) win_n[t],
                      (double) win_ns[t] / (double) win_n[t] / 1000.0,
                      (unsigned long long) n_by_type[t],
                      (double) ns_by_type[t] / (double) n_by_type[t] / 1000.0);
        line += buf;
        win_n[t] = 0; win_ns[t] = 0;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}

int serve_connection(pipe_socket_t & socket, Worker & worker, int conn_index = -1) {
    struct PendingCleanup {
        Worker & worker;
        int      conn_index;
        ~PendingCleanup() {
            // Connection close is NOT inside the per-request gpu_lock below
            // (that lock is scoped to one loop iteration; this destructor
            // fires after the loop exits, on `return` from anywhere in this
            // function, including recv/protocol failures). Single-connection
            // default: only one thread ever exists, no race possible.
            // Multi-conn: another connection's thread can still be
            // mid-dispatch under the lock when this one closes, so take the
            // same lock here rather than leave this as an unguarded touch of
            // shared Worker state (split_pending_by_conn_, pool_.demand_
            // serving) -- and pass THIS connection's own conn_index so it
            // only ever tears down its own pending transaction, never
            // another connection's (see split_pending_by_conn_'s comment).
            // NOTE: this graft previously declared conn_index as a
            // serve_connection parameter but never actually threaded it into
            // PendingCleanup or any of the split-dispatch calls below --
            // every one of them was still hitting the single Worker-wide
            // split_pending_ this fix removes.
            if (g_worker_gpu_mutex != nullptr) {
                std::lock_guard<std::mutex> lock(*g_worker_gpu_mutex);
                worker.abandon_split_dispatch(conn_index);
            } else {
                worker.abandon_split_dispatch(conn_index);
            }
        }
    } pending_cleanup{ worker, conn_index };
    const pipe_expert_hello mine = worker.hello();
    const std::vector<uint8_t> hello_payload = pipe_encode_expert_hello(mine);
    if (!pipe_send_frame(
            socket, PIPE_HELLO, 0, hello_payload.data(), hello_payload.size())) {
        return 1;
    }

    pipe_frame_type type;
    uint64_t seq_id = 0;
    std::vector<uint8_t> payload;
    if (!pipe_recv_frame(socket, type, seq_id, payload)) {
        return 1;
    }
    if (type != PIPE_HELLO || seq_id != 0) {
        send_hello_ack(socket, false, "expected expert HELLO");
        return 1;
    }
    try {
        const pipe_expert_hello client =
            pipe_decode_expert_hello(payload.data(), payload.size());
        std::string error;
        if (!validate_client_hello(client, mine, error)) {
            send_hello_ack(socket, false, error);
            return 1;
        }
    } catch (const pipe_protocol_error & error) {
        send_hello_ack(socket, false, error.what());
        return 1;
    }
    if (!send_hello_ack(socket, true, "")) {
        return 1;
    }

    // WP_REQ_LOG=path -- one line per dispatch request. Columns:
    //   layer n_tokens n_exp n_resident n_pagein bytes_read ns_wall ns_lookup ns_prep
    //   ns_hits ns_wait ns_pagein_compute ns_result ns_read ns_h2d ns_submit
    //   ns_readback ns_encode ns_send n_weight_nonzero n_weight_total epoch_end
    //   ns_params_set n_host_hit n_host_demote ns_host_get ns_demote ns_ensure_post
    //   ns_final_sync
    // epoch_end (added 2026-08-06) is the request's wall-clock END in epoch
    // seconds; start = epoch_end - ns_wall/1e9.
    // ns_params_set (added 2026-08-07): the coalesced D1 blob upload; 0 when
    // WP_EXPERT_PARAMS_COALESCE is off. Appended AFTER epoch_end -- the format
    // is positional-from-the-left, so trailing additions never move existing
    // columns, but anything indexing epoch_end as [-1] must switch to [21].
    // ns_final_sync is the deferred backend sync when WP_SUBMIT_ASYNC=1.
    // It is appended as the final column and is zero on the default path.
    // Segment into tokens by watching request.layer wrap back to its minimum.
    //
    // n_tokens (added 2026-08-03) IS THE PREFILL/DECODE LABEL, and it is the whole
    // reason this log can now answer "where does prefill go". n_tokens > 1 is a
    // prefill ubatch, n_tokens == 1 is a decode step -- the dispatcher already
    // relies on exactly this test to disable deferral (pipe-expert-dispatcher.cpp
    // :969), so the semantics are not invented here, only recorded.
    //
    // Until now EVERY phase timer in this file was un-attributable: prefill and
    // decode requests interleave in one stream and the only way to tell them apart
    // was to guess from n_exp. That made the 18 columns below useless for the one
    // question that matters (prefill is 33.9 ms/token and decode is 6.5 tok/s --
    // WHICH of these phases owns which?). One integer fixes it.
    FILE * const req_log = [] {
        const char * p = std::getenv("WP_REQ_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();
    auto write_req_log = [req_log](int32_t layer, uint32_t n_tokens, size_t n_assignments,
                                   const RequestStats & s,
                                   std::chrono::steady_clock::time_point started) {
        if (req_log == nullptr || started == std::chrono::steady_clock::time_point{}) {
            return;
        }
        const uint64_t ns_wall = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        const double epoch_end = (double) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1e6;
        fprintf(req_log,
                "%d %u %zu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu "
                "%llu %llu %llu %llu %llu %llu %llu %llu %.6f %llu "
                "%llu %llu %llu %llu %llu %llu\n",
                layer, n_tokens, n_assignments,
                (unsigned long long) s.n_resident, (unsigned long long) s.n_pagein,
                (unsigned long long) s.bytes_read, (unsigned long long) ns_wall,
                (unsigned long long) s.ns_lookup, (unsigned long long) s.ns_prep,
                (unsigned long long) s.ns_hits, (unsigned long long) s.ns_wait,
                (unsigned long long) s.ns_pagein_compute, (unsigned long long) s.ns_result,
                (unsigned long long) s.ns_read, (unsigned long long) s.ns_h2d,
                (unsigned long long) s.ns_submit, (unsigned long long) s.ns_readback,
                (unsigned long long) s.ns_encode, (unsigned long long) s.ns_send,
                (unsigned long long) s.n_weight_nonzero,
                (unsigned long long) s.n_weight_total, epoch_end,
                (unsigned long long) s.ns_params_set,
                (unsigned long long) s.n_host_hit,
                (unsigned long long) s.n_host_demote,
                (unsigned long long) s.ns_host_get,
                (unsigned long long) s.ns_demote,
                (unsigned long long) s.ns_ensure_post,
                (unsigned long long) s.ns_final_sync);
        fflush(req_log);
    };
    pipe_expert_dispatch_begin split_log_begin;
    RequestStats split_log_stats;
    std::chrono::steady_clock::time_point split_log_started;
    bool null_split_active = false;   // §8.25 WP_WORKER_NULL split-path flag

    // WP_REF_LOG=path -- the full REFERENCE stream: "<layer> <expert> <expert> ..."
    // one line per request, every expert asked for whether it was resident or paged in.
    // WP_PAGEIN_LOG cannot substitute: which pages pagein is a function of the
    // replacement policy, so a pagein trace can only ever describe the policy that
    // produced it. The reference stream is policy-independent, which makes LRU /
    // LFU / ARC / Belady all simulatable OFFLINE from a single run, at zero GPU
    // cost per candidate -- and Belady gives the true ceiling, so we learn
    // whether a policy change is worth implementing BEFORE implementing one.
    FILE * const ref_log = [] {
        const char * p = std::getenv("WP_REF_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    // Keepalive pump: while no request is pending, occupy the GPU rather than
    // let it idle (see Worker::keepalive_tick for the measurements). Gated on
    // KEEPALIVE_IDLE_MS of recent activity so a genuinely idle worker still lets
    // the card reach runtime suspend (D3), which is where its idle power saving
    // comes from -- an unconditional pump would keep it awake forever.
    const int keepalive_fd = socket.poll_fd();
    const auto KEEPALIVE_IDLE_MS = std::chrono::milliseconds(2000);
    auto last_request_at = std::chrono::steady_clock::now();
    auto await_request = [&]() {
        // WP_WORKER_MULTI_CONN: the keepalive/speculative pump below touches
        // shared Worker state (keepalive_tick, spec_pagein_step,
        // drop_spec_work) and runs BEFORE recv, i.e. before this thread has
        // any chance to take g_worker_gpu_mutex for the iteration it's
        // priming. Two real options were considered:
        //   1. Take the GPU lock around the whole pump loop too.
        //   2. Skip the pump entirely in multi-conn mode (kept, below).
        // (1) is unsafe-by-a-different-route: the pump's inner loop can
        // block in ppoll() for up to keepalive_us per iteration while idle,
        // and holding the GPU lock across that would stall the OTHER
        // connection's dispatch for the same duration -- worse than just
        // skipping the pump, since it would turn "idle time filled by
        // another stream" (the whole premise of multi-conn) into "idle time
        // spent holding the lock nobody else can use". (2) is also not a
        // real loss: the pump exists to keep the GPU from clock-dropping
        // during single-connection idle gaps; multi-conn's entire premise is
        // that a second stream fills those idle gaps with real work, so the
        // pump's job is already substantially done by the other connection
        // in the cases that matter. Recv falls straight through to a
        // blocking wait, same as the probe.
        if (g_worker_gpu_mutex != nullptr) {
            return;
        }
        // The speculative path needs this loop even when the keepalive pump is off:
        // they share the idle window but are independent features, and gating
        // speculation on WP_KEEPALIVE_US would silently couple two experiments.
        if ((!worker.keepalive_enabled() && !worker.has_spec_work()) || keepalive_fd < 0) {
            return;
        }
        for (;;) {
            // Re-checked every iteration, not just on entry. Without this, a
            // worker with the pump OFF that finishes its spec queue would sit in
            // a zero-timeout ppoll spinning a core: nothing left to read, and a
            // keepalive_tick that is a no-op when the pump is disabled.
            if (!worker.keepalive_enabled() && !worker.has_spec_work()) {
                return;
            }
            struct pollfd pfd { keepalive_fd, POLLIN, 0 };
            // ppoll, NOT poll: poll's timeout is in whole milliseconds, so a
            // 200 us period silently became 1 ms and left the GPU idle for most
            // of every interval. That cost us half the available win -- measured
            // 0.319 ms/expert with the 1 ms pump against 0.163 in an isolated
            // bench that occupied the device continuously.
            // Zero timeout ONLY when there is something to submit -- then the
            // pump period is dead time before issuing a read. With a read already
            // in flight we are waiting on the reader thread, so poll on the
            // ordinary period and harvest when it lands. Spinning there would
            // burn a core for the whole 3-5 ms read and buy nothing.
            const long ns = worker.has_spec_submit_work() ? 0L
                                                          : (long) worker.keepalive_us() * 1000L;
            struct timespec ts { ns / 1000000000L, ns % 1000000000L };
            const int r = ::ppoll(&pfd, 1, &ts, nullptr);
            if (r != 0) {
                // A REAL REQUEST BEATS A PREFETCH, ALWAYS. Whatever is still
                // queued stays queued -- if it is still worth reading, the next
                // idle window will take it, and if the layer has moved on the
                // idle timeout below discards it.
                return;   // data ready, or an error recv_data will surface
            }
            if (std::chrono::steady_clock::now() - last_request_at > KEEPALIVE_IDLE_MS) {
                // Idle long enough that any hinted layer is far behind us.
                // Reading for a layer already computed is a read with no
                // possible upside, so drop it rather than carry it forward.
                worker.drop_spec_work();
                return;   // let the card sleep
            }
            // Warm before pumping: the pump exists to stop the card dropping
            // clocks while idle, and a page-in plus its H2D is not idle.
            if (worker.spec_pagein_step()) {
                continue;
            }
            worker.keepalive_tick();
        }
    };

    // See RequestStats::ns_recv_body. `hdr_done_ns` is stamped INSIDE
    // pipe_recv_frame the instant the 32-byte header recv returns, which is the
    // earliest point in this process that anything knows a frame is arriving --
    // every other worker clock starts after the whole body has landed. The
    // pointer is null unless a consumer of the number is enabled, and
    // pipe_recv_frame reads no clock at all in that case, so the default build
    // is unchanged.
    const bool     time_recv   = worker.stats_enabled() || frame_trace_enabled();
    uint64_t       hdr_done_ns = 0;
    // Reused response-encode buffer. One per serve_connection call, so one per
    // connection thread -- no sharing, no thread_local, and nothing to
    // synchronise. It removes a heap allocation AND a full zero-fill of the
    // payload from every single response (see the note on
    // pipe_encode_expert_partial_into); decode shape is constant across a run,
    // so after the first response resize() stops doing any work at all. The
    // encoded bytes are identical either way -- this changes where the bytes
    // live, never what they are. Safe against the unlock-for-send window
    // because the buffer is only ever read between its own encode and the send
    // that immediately follows, inside one loop iteration on this thread.
    std::vector<uint8_t> encode_buf;
    // Comma operator so the pump runs before EVERY recv, including after the
    // PING branch's `continue` -- appending it to the loop body would skip that.
    while ((await_request(),
            pipe_recv_frame(socket, type, seq_id, payload,
                            time_recv ? &hdr_done_ns : nullptr))) {
        // Bracket this whole iteration -- see FrameResidency above. Declared
        // FIRST so it spans every branch below, including the ones that
        // `continue` (PING, PREFETCH_HINT, BEGIN) and the ones that return.
        struct FrameResidencyScope {
            bool                                  on;
            int                                   type;
            std::chrono::steady_clock::time_point started;
            ~FrameResidencyScope() {
                if (!on) return;
                note_frame_residency(type,
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count());
            }
        } frame_residency_scope{
            frame_trace_enabled(), (int) type,
            frame_trace_enabled() ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point() };
        // Segment stamps. Reuse the scope's own start so the two agree exactly.
        const bool seg_trace = frame_trace_enabled();
        const std::chrono::steady_clock::time_point t_frame =
            frame_residency_scope.started;
        // "The whole frame is in hand." Identical to t_frame when the frame
        // trace is on (reuse the same reading so the two can never disagree);
        // taken independently when only WP_WORKER_STATS is on, because t_frame
        // is a default-constructed epoch value in that configuration.
        const std::chrono::steady_clock::time_point t_recv_done =
            seg_trace ? frame_residency_scope.started
                      : (time_recv ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point());
        // Header-recv-return -> body-complete, for THIS frame. Zero when the
        // instrumentation is off, and zero-guarded against a header stamp that
        // was never written.
        const uint64_t ns_recv_body_frame =
            (time_recv && hdr_done_ns != 0)
                ? (uint64_t) ((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  t_recv_done.time_since_epoch()).count() - hdr_done_ns)
                : 0;
        // WP_WORKER_MULTI_CONN: default-held for the whole per-request
        // handling below, same shape as the probe (RAII releases it on
        // every exit from this scope -- return, continue, or falling off
        // the bottom -- so a second connection's thread can proceed the
        // instant this one lets go, without touching every individual
        // return/continue site). Single-connection default path: mutex
        // pointer is null, this is one branch and no lock is ever taken --
        // byte-identical to before.
        //
        // UNLIKE the probe, the four response-bearing branches below
        // (ACTS, ACTS_PUBLISH, ACTS_REF, plain DISPATCH_REQ) explicitly
        // gpu_lock.unlock() right before their pipe_send_frame() of the
        // computed response and gpu_lock.lock() again immediately after --
        // encoding and sending an already-computed pipe_expert_partial (a
        // local value, no shared-state references) does not touch pool_,
        // compute_galloc_, or any residency structure, so it is provably
        // safe to run unlocked. That window is exactly "this connection's
        // network send" overlapping "the other connection's GPU-lock-
        // holding dispatch", which is the overlap this task asks for.
        // spec_pagein_after_dispatch() (pool_-touching) and record_stats()
        // stay locked, immediately after the relock.
        std::unique_lock<std::mutex> gpu_lock;
        if (g_worker_gpu_mutex != nullptr) {
            gpu_lock = std::unique_lock<std::mutex>(*g_worker_gpu_mutex);
        }
        const std::chrono::steady_clock::time_point t_locked =
            seg_trace ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point();
        last_request_at = std::chrono::steady_clock::now();
        if (type == PIPE_PING) {
            if (!pipe_send_frame(socket, PIPE_PONG, seq_id, nullptr, 0)) {
                return 1;
            }
            continue;
        }
        // Prefetch hint: accept, validate, count, DO NOTHING ELSE (yet).
        //
        // This half exists on its own so the wire can be proven before the pool
        // is touched. With only this, a run is observably identical to the
        // config of record -- no extra read, no eviction, no slot pinned -- and
        // the hint counters still say exactly what the spine offered and when.
        // If those counters come out wrong, the fault is on the spine side and
        // is found without a single changed page-in.
        //
        // A hint gets NO response frame, so it must not fall through to the
        // dispatch path's reply, and a malformed one must not kill the session:
        // it is advisory, and the request stream is untouched by dropping it.
        if (type == PIPE_EXPERT_PREFETCH_HINT) {
            try {
                const pipe_expert_prefetch_hint hint =
                    pipe_decode_expert_prefetch_hint(payload.data(), payload.size());
                worker.note_prefetch_hint(hint);
            } catch (const pipe_protocol_error & error) {
                worker.note_prefetch_hint_bad();
                std::fprintf(stderr, "wp-expert-worker: ignoring malformed prefetch hint: %s\n",
                             error.what());
            }
            continue;
        }
        if (type == PIPE_EXPERT_DISPATCH_BEGIN) {
            try {
                split_log_begin = pipe_decode_expert_dispatch_begin(payload.data(), payload.size());
                split_log_stats = RequestStats{};
                split_log_started = req_log != nullptr ? std::chrono::steady_clock::now() :
                    std::chrono::steady_clock::time_point{};
                // WP_WORKER_NULL=1 (TIMING PROBE, §8.25): decode-scale split
                // dispatches are nulled AT BEGIN — the batch is never prepared,
                // no slot pinned, no read issued — and ACTS below answers with
                // a zeroed partial. Prefill (n_tokens>8) stays real.
                static const bool worker_null_split = [] {
                    const char * e = std::getenv("WP_WORKER_NULL");
                    return e != nullptr && e[0] == '1';
                }();
                if (worker_null_split && split_log_begin.n_tokens <= 8) {
                    null_split_active = true;
                    continue;
                }
                // &gpu_lock, not gated here on g_worker_gpu_mutex != nullptr:
                // ensure_batch's own owns_unlock check (gpu_lock->owns_lock())
                // already collapses to a no-op whenever gpu_lock was never
                // locked in the first place -- true both for single-connection
                // mode (g_worker_gpu_mutex == nullptr, gpu_lock stays a
                // default-constructed unique_lock with no mutex) and for a
                // multi-conn PING/other frame's gpu_lock passed by mistake, so
                // there is no separate gate to keep in sync with that one.
                worker.begin_split_dispatch(split_log_begin, seq_id, conn_index, &gpu_lock);
            } catch (const pipe_protocol_error & error) {
                pipe_send_error(socket, seq_id, error.code, error.what());
                return 1;
            } catch (const std::exception & error) {
                pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
                return 1;
            }
            continue;
        }
        if (type == PIPE_EXPERT_DISPATCH_ACTS) {
            try {
                if (null_split_active) {
                    // §8.25 null path: zeroed partial, no batch ever existed.
                    null_split_active = false;
                    pipe_expert_partial znull;
                    znull.layer    = split_log_begin.layer;
                    znull.n_tokens = split_log_begin.n_tokens;
                    znull.partial.assign(
                        (size_t) split_log_begin.n_tokens * mine.n_embd, 0.0f);
                    const std::vector<uint8_t> zenc = pipe_encode_expert_partial(znull);
                    if (!pipe_send_frame(socket, PIPE_EXPERT_PARTIAL, seq_id,
                                         zenc.data(), zenc.size())) {
                        return 1;
                    }
                    continue;
                }
                if (!worker.has_split_dispatch(conn_index)) {
                    throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                              "expert dispatch ACTS has no BEGIN");
                }
                // Bracket the request decode itself. On this branch the payload
                // is the full activation block, so this is a real bulk copy,
                // not a header parse -- and ns_decode_req never covered it
                // because that counter only exists on the plain DISPATCH_REQ
                // branch, which this path never takes.
                const auto t_acts_decode_start = time_recv
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                const pipe_expert_dispatch_acts acts = pipe_decode_expert_dispatch_acts(
                    payload.data(), payload.size(), worker.split_n_tokens(conn_index), mine.n_embd);
                const uint64_t ns_req_decode_frame = time_recv
                    ? (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t_acts_decode_start).count()
                    : 0;
                // BEGIN fixes assignment index order; splitting changes only when
                // reads start, never the computation order or resulting bytes.
                const pipe_expert_partial response = worker.finish_split_dispatch(
                    acts, seq_id, split_log_stats, conn_index);
                const auto split_send_started = worker.stats_enabled()
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                pipe_encode_expert_partial_into(encode_buf, response);
                const std::vector<uint8_t> & encoded = encode_buf;
                // Unlock for the send: `encoded` aliases this connection's own
                // encode buffer (no other thread can reach it) and
                // the socket write touches no Worker state, so this
                // connection's network I/O can overlap another connection's
                // GPU-lock-holding dispatch. Relock immediately after --
                // record_stats()/spec_pagein_after_dispatch() below both
                // touch pool_/stats_ and must stay serialized.
                const bool relock1 = gpu_lock.owns_lock();
                if (relock1) gpu_lock.unlock();
                const bool sent1 = pipe_send_frame(socket, PIPE_EXPERT_PARTIAL, seq_id,
                                                   encoded.data(), encoded.size());
                if (relock1) gpu_lock.lock();
                // Stamped before the bookkeeping below so ns_resp_send is
                // exactly encode + unlock + send + relock -- the worker->spine
                // leg, and the last thing that is still inside the spine's wait.
                const auto t_resp_sent = worker.stats_enabled()
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                if (!sent1) {
                    return 1;
                }
                if (conn_index >= 0) {
                    g_worker_conn_request_counts[(size_t) conn_index].fetch_add(
                        1, std::memory_order_relaxed);
                }
                if (worker.stats_enabled()) {
                    split_log_stats.ns_send = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - split_send_started).count();
                    // Assigned AFTER finish_split_dispatch has populated
                    // split_log_stats, so a whole-struct write in there can
                    // never clobber them.
                    split_log_stats.ns_recv_body  = ns_recv_body_frame;
                    split_log_stats.ns_req_decode = ns_req_decode_frame;
                    split_log_stats.ns_resp_send  =
                        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t_resp_sent - split_send_started).count();
                    worker.record_stats(split_log_stats, split_log_begin.assignments.size());
                }
                write_req_log(split_log_begin.layer, split_log_begin.n_tokens,
                              split_log_begin.assignments.size(), split_log_stats, split_log_started);
                split_log_started = std::chrono::steady_clock::time_point{};
                worker.spec_pagein_after_dispatch();
            } catch (const pipe_protocol_error & error) {
                pipe_send_error(socket, seq_id, error.code, error.what());
                return 1;
            } catch (const std::exception & error) {
                pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
                return 1;
            }
            continue;
        }
#if defined(__linux__)
        // WP_DISPATCH_DEDUP_ACTIVATIONS: primary role. Full activations arrive
        // exactly as PIPE_EXPERT_DISPATCH_ACTS would carry them, plus a
        // request to publish them for co-located siblings. See the
        // PIPE_VERSION 14 comment in pipe-protocol.h and the wp_dedup
        // namespace above.
        if (type == PIPE_EXPERT_DISPATCH_ACTS_PUBLISH) {
            try {
                if (!worker.has_split_dispatch(conn_index)) {
                    throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                              "expert dispatch ACTS_PUBLISH has no BEGIN");
                }
                const int32_t layer_for_shm = split_log_begin.layer;
                const auto t_pub_decode_start = time_recv
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                pipe_expert_dispatch_acts_publish publish = pipe_decode_expert_dispatch_acts_publish(
                    payload.data(), payload.size(), worker.split_n_tokens(conn_index), mine.n_embd);
                // Decode only. The wp_dedup::publish() shm write below is a
                // deliberate exclusion: it is dedup work done on behalf of the
                // SIBLING workers, not the cost of materialising this request.
                const uint64_t ns_req_decode_frame = time_recv
                    ? (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t_pub_decode_start).count()
                    : 0;
                std::string publish_error;
                const bool published = wp_dedup::publish(
                    seq_id, layer_for_shm, publish.activations, worker.split_n_tokens(conn_index),
                    (uint32_t) mine.n_embd, publish.n_subscribers, publish_error);
                if (!published) {
                    // Advisory to the log only -- the spine finds out via the
                    // ack below and falls back to sending siblings the
                    // ordinary full payload. This worker's own compute below
                    // is unaffected either way.
                    std::fprintf(stderr,
                                 "wp-expert-worker: dedup publish failed (seq=%llu layer=%d): %s\n",
                                 (unsigned long long) seq_id, layer_for_shm, publish_error.c_str());
                }
                const std::vector<uint8_t> ack_payload =
                    pipe_encode_expert_acts_publish_ack({ published });
                if (!pipe_send_frame(socket, PIPE_EXPERT_ACTS_PUBLISH_ACK, seq_id,
                                     ack_payload.data(), ack_payload.size())) {
                    return 1;
                }
                // Compute from the inline bytes already in hand. This is the
                // guarantee that makes a publish failure harmless to THIS
                // worker: it never reads back its own shm segment.
                pipe_expert_dispatch_acts acts;
                acts.activations = std::move(publish.activations);
                const pipe_expert_partial response = worker.finish_split_dispatch(
                    acts, seq_id, split_log_stats, conn_index);
                const auto split_send_started = worker.stats_enabled()
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                pipe_encode_expert_partial_into(encode_buf, response);
                const std::vector<uint8_t> & encoded = encode_buf;
                // Same unlock-for-send window as the ACTS branch above.
                const bool relock2 = gpu_lock.owns_lock();
                if (relock2) gpu_lock.unlock();
                const bool sent2 = pipe_send_frame(socket, PIPE_EXPERT_PARTIAL, seq_id,
                                                   encoded.data(), encoded.size());
                if (relock2) gpu_lock.lock();
                // See the ACTS branch: encode + unlock + send + relock only.
                const auto t_resp_sent = worker.stats_enabled()
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                if (!sent2) {
                    return 1;
                }
                if (conn_index >= 0) {
                    g_worker_conn_request_counts[(size_t) conn_index].fetch_add(
                        1, std::memory_order_relaxed);
                }
                if (published) {
                    // This worker is one of the n_subscribers+1 holders
                    // publish() seeded remaining with; check in now that its
                    // own compute (the last thing that needed the bytes) is
                    // done. Skipped when publish failed -- nothing to
                    // release.
                    wp_dedup::release(seq_id, layer_for_shm);
                }
                if (worker.stats_enabled()) {
                    split_log_stats.ns_send = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - split_send_started).count();
                    split_log_stats.ns_recv_body  = ns_recv_body_frame;
                    split_log_stats.ns_req_decode = ns_req_decode_frame;
                    split_log_stats.ns_resp_send  =
                        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t_resp_sent - split_send_started).count();
                    worker.record_stats(split_log_stats, split_log_begin.assignments.size());
                }
                write_req_log(split_log_begin.layer, split_log_begin.n_tokens,
                              split_log_begin.assignments.size(), split_log_stats, split_log_started);
                split_log_started = std::chrono::steady_clock::time_point{};
                worker.spec_pagein_after_dispatch();
            } catch (const pipe_protocol_error & error) {
                pipe_send_error(socket, seq_id, error.code, error.what());
                return 1;
            } catch (const std::exception & error) {
                pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
                return 1;
            }
            continue;
        }
        // WP_DISPATCH_DEDUP_ACTIVATIONS: secondary role. No activation bytes
        // on this frame -- read them out of the primary's shm segment.
        //
        // PIPE_ERR_ACTS_UNAVAILABLE is the ONE error on this frame pair that
        // does NOT close the connection (`continue` instead of `return 1`).
        // Every other branch here keeps the existing "any protocol error on
        // BEGIN/ACTS is fatal to the connection" behaviour unchanged; this is
        // a new, narrow, fully self-contained failure mode with its own
        // spine-side recovery (see receive_partial()'s dedup fallback in
        // pipe-expert-dispatcher.cpp), not a general relaxation of that rule.
        if (type == PIPE_EXPERT_DISPATCH_ACTS_REF) {
            try {
                if (!worker.has_split_dispatch(conn_index)) {
                    throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                              "expert dispatch ACTS_REF has no BEGIN");
                }
                const auto t_ref_decode_start = time_recv
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                const pipe_expert_dispatch_acts_ref ref =
                    pipe_decode_expert_dispatch_acts_ref(payload.data(), payload.size());
                if (ref.n_tokens != worker.split_n_tokens(conn_index)) {
                    throw pipe_protocol_error(PIPE_ERR_BAD_FRAME,
                                              "expert dispatch ACTS_REF n_tokens does not match BEGIN");
                }
                const int32_t layer_for_shm = split_log_begin.layer;
                std::string   sub_error;
                std::optional<std::vector<float>> acts_vec = wp_dedup::subscribe(
                    seq_id, layer_for_shm, ref.n_tokens, (uint32_t) mine.n_embd, sub_error);
                if (!acts_vec.has_value()) {
                    std::fprintf(stderr,
                                 "wp-expert-worker: dedup subscribe failed (seq=%llu layer=%d): %s\n",
                                 (unsigned long long) seq_id, layer_for_shm, sub_error.c_str());
                    worker.abandon_split_dispatch(conn_index);
                    if (!pipe_send_error(socket, seq_id, PIPE_ERR_ACTS_UNAVAILABLE, sub_error)) {
                        return 1;
                    }
                    continue;
                }
                wp_dedup::release(seq_id, layer_for_shm);
                // On THIS branch the request's activation bytes do not arrive
                // on the frame at all -- the frame is a reference and the bytes
                // come out of the primary's shm segment. So the comparable
                // "materialise the request" cost is decode + subscribe + the
                // copy out, and that is what is measured here. Read alongside
                // ns_recv_body, which will be near-zero on this branch
                // precisely because the payload is tiny: on ACTS_REF the bulk
                // transfer moved from the socket to shm, and these two counters
                // are what make that visible instead of inferred.
                const uint64_t ns_req_decode_frame = time_recv
                    ? (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t_ref_decode_start).count()
                    : 0;
                pipe_expert_dispatch_acts acts;
                acts.activations = std::move(*acts_vec);
                const pipe_expert_partial response = worker.finish_split_dispatch(
                    acts, seq_id, split_log_stats, conn_index);
                const auto split_send_started = worker.stats_enabled()
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                pipe_encode_expert_partial_into(encode_buf, response);
                const std::vector<uint8_t> & encoded = encode_buf;
                // Same unlock-for-send window as the ACTS branch above.
                const bool relock3 = gpu_lock.owns_lock();
                if (relock3) gpu_lock.unlock();
                const bool sent3 = pipe_send_frame(socket, PIPE_EXPERT_PARTIAL, seq_id,
                                                   encoded.data(), encoded.size());
                if (relock3) gpu_lock.lock();
                // See the ACTS branch: encode + unlock + send + relock only.
                const auto t_resp_sent = worker.stats_enabled()
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
                if (!sent3) {
                    return 1;
                }
                if (conn_index >= 0) {
                    g_worker_conn_request_counts[(size_t) conn_index].fetch_add(
                        1, std::memory_order_relaxed);
                }
                if (worker.stats_enabled()) {
                    split_log_stats.ns_send = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - split_send_started).count();
                    split_log_stats.ns_recv_body  = ns_recv_body_frame;
                    split_log_stats.ns_req_decode = ns_req_decode_frame;
                    split_log_stats.ns_resp_send  =
                        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t_resp_sent - split_send_started).count();
                    worker.record_stats(split_log_stats, split_log_begin.assignments.size());
                }
                write_req_log(split_log_begin.layer, split_log_begin.n_tokens,
                              split_log_begin.assignments.size(), split_log_stats, split_log_started);
                split_log_started = std::chrono::steady_clock::time_point{};
                worker.spec_pagein_after_dispatch();
            } catch (const pipe_protocol_error & error) {
                pipe_send_error(socket, seq_id, error.code, error.what());
                return 1;
            } catch (const std::exception & error) {
                pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
                return 1;
            }
            continue;
        }
#endif // __linux__
        if (worker.has_split_dispatch(conn_index)) {
            pipe_send_error(socket, seq_id, PIPE_ERR_BAD_FRAME,
                            "frame is not legal between dispatch BEGIN and ACTS");
            return 1;
        }
        if (type != PIPE_EXPERT_DISPATCH_REQ) {
            pipe_send_error(socket, seq_id, PIPE_ERR_BAD_FRAME, "expected expert dispatch request");
            return 1;
        }
        try {
            const auto t_req_decode_start = time_recv
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            const pipe_expert_dispatch_req request =
                pipe_decode_expert_dispatch_req(
                    payload.data(), payload.size(), mine.n_embd);
            const std::chrono::steady_clock::time_point t_decoded =
                seg_trace ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point();
            // Same window ns_decode_req covers, but gated on WP_WORKER_STATS
            // rather than WP_FRAME_TRACE so the one counter name means the same
            // thing on every branch and the stats line is comparable across
            // wire paths without also enabling the frame trace.
            const uint64_t ns_req_decode_frame = time_recv
                ? (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - t_req_decode_start).count()
                : 0;
            RequestStats request_stats;
            if (seg_trace) {
                request_stats.ns_lock_wait =
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_locked - t_frame).count();
                request_stats.ns_decode_req =
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_decoded - t_locked).count();
            }
            // WP_REQ_LOG: per-request phase dump. The cumulative WorkerStats
            // totals cannot separate "every request is uniformly slow" from
            // "most are fast and a few are enormous", and on the RX 480 the
            // per-token cost distribution is p25 140 / median 170 / max 1653 ms
            // -- so the shape is the whole question. request.layer lets the
            // reader segment this stream into tokens by layer wrap WITHOUT a
            // cross-machine clock join (the spine runs on the other box); that
            // is what made WP_PAGEIN_LOG unusable for the same purpose.
            if (ref_log != nullptr) {
                fprintf(ref_log, "%d", request.layer);
                for (const pipe_expert_assignment & a : request.assignments) {
                    fprintf(ref_log, " %d", a.expert_id);
                }
                // D5 (throughput-analysis): trailing n_tokens column so the offline
                // eviction sim can tell prefill (n_tokens > 1) from decode (==1)
                // requests -- the prefill-aware / sweep-boundary policies need the
                // phase boundary. Trailing, so the expert columns never move. The
                // nt= sentinel keeps the column self-describing: a bare integer is
                // indistinguishable from an expert id, so every REF_LOG consumer
                // (sim-evict.py, probe-embed-routing.py) would silently misparse
                // legacy-vs-new captures without it.
                fprintf(ref_log, " nt=%u\n", request.n_tokens);
                fflush(ref_log);
            }
            // Ground truth into the ordered stream. Emitted BEFORE dispatch, so
            // an R always precedes the D lines that dispatch provokes -- which is
            // what makes "was this page already speculated for THIS reference"
            // answerable by a single forward pass over the file.
            worker.log_reference(request.layer, request.assignments);
            const std::chrono::steady_clock::time_point req_started =
                req_log != nullptr ? std::chrono::steady_clock::now() :
                                     std::chrono::steady_clock::time_point();
            const std::chrono::steady_clock::time_point t_pre_dispatch =
                seg_trace ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point();
            if (seg_trace) {
                request_stats.ns_pre_dispatch =
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_pre_dispatch - t_decoded).count();
            }
            const pipe_expert_partial response = worker.dispatch(
                request, request_stats, std::nullopt, conn_index);
            const std::chrono::steady_clock::time_point t_dispatched =
                seg_trace ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point();
            const bool measure = worker.stats_enabled();
            const std::chrono::steady_clock::time_point send_started =
                measure ? std::chrono::steady_clock::now() :
                          std::chrono::steady_clock::time_point();
            pipe_encode_expert_partial_into(encode_buf, response);
            const std::vector<uint8_t> & encoded = encode_buf;
            // Same unlock-for-send window as the split-dispatch branches
            // above: `response` is a local value and `encoded` aliases this
            // connection's own encode buffer, so the socket write touches no
            // Worker state and nothing another thread can observe.
            const bool relock4 = gpu_lock.owns_lock();
            if (relock4) gpu_lock.unlock();
            const bool sent4 = pipe_send_frame(
                    socket, PIPE_EXPERT_PARTIAL, seq_id,
                    encoded.data(), encoded.size());
            if (relock4) gpu_lock.lock();
            if (!sent4) {
                return 1;
            }
            if (conn_index >= 0) {
                g_worker_conn_request_counts[(size_t) conn_index].fetch_add(
                    1, std::memory_order_relaxed);
            }
            if (measure) {
                const std::chrono::steady_clock::time_point t_sent =
                    std::chrono::steady_clock::now();
                request_stats.ns_send =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_sent - send_started).count();
                request_stats.ns_recv_body  = ns_recv_body_frame;
                request_stats.ns_req_decode = ns_req_decode_frame;
                // Same window as ns_send on this branch (t_sent is taken after
                // the connection counter bump, so both carry that one relaxed
                // atomic increment). Kept as its own name so ns_resp_send has
                // identical meaning on the split branches, where ns_send is
                // stopped at a different point.
                request_stats.ns_resp_send =
                    (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_sent - send_started).count();
                if (seg_trace) {
                    // encode + unlock + send + relock. Everything between
                    // dispatch() returning and the bytes being on the wire --
                    // the last segment that can still be inside the spine's wait.
                    request_stats.ns_encode_send =
                        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            t_sent - t_dispatched).count();
                    // NOTE: this stops at record_stats, so it does NOT include
                    // spec_pagein_after_dispatch(), which runs after this block.
                    // That call is AFTER the send and therefore cannot be part of
                    // what the spine waits for; recover it as
                    //   frame residency - (lock+decode+pre+dispatch+encode_send)
                    // from the frame-residency line.
                    request_stats.ns_post_send =
                        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t_sent).count();
                }
                worker.record_stats(
                    request_stats, request.assignments.size());
            }
            // Every phase field above is populated only when WP_WORKER_STATS=1,
            // so this log is meaningless without it -- hence `measure &&`.
            // fflush per line: the harness SIGKILLs workers at teardown, and an
            // unflushed stdio buffer produced 0-byte files the first time.
            if (measure && req_log != nullptr) {
                write_req_log(request.layer, request.n_tokens, request.assignments.size(),
                              request_stats, req_started);
            }
            worker.spec_pagein_after_dispatch();
        } catch (const pipe_protocol_error & error) {
            // LOG LOCALLY as well as replying. The spine renders a dropped or
            // errored connection as "worker died while computing <experts>",
            // which names the symptom and never the cause; without this the
            // only copy of the real message is in flight on a socket that is
            // about to close.
            std::fprintf(stderr, "wp expert worker: protocol error (code %d): %s\n",
                         (int) error.code, error.what());
            if (!pipe_send_error(socket, seq_id, error.code, error.what())) {
                return 1;
            }
        } catch (const std::exception & error) {
            std::fprintf(stderr, "wp expert worker: compute error: %s\n", error.what());
            pipe_send_error(socket, seq_id, PIPE_ERR_EXPERT_COMPUTE, error.what());
            return 1;
        }
    }
    // Only on a clean close, and the harness SIGKILLs workers at teardown, so
    // this line is best effort and arm 1 never saw it. WP_HINT_LOG is the
    // durable record -- NOT WP_REQ_LOG, which carries no hint fields and was
    // wrongly named here before.
    //
    // report_prefetch_hints() reads pool_.n_layerahead_hits() and Worker's
    // own hint counters -- shared state another connection's thread can be
    // mid-dispatch-mutating, so this needs the same lock as everything else
    // that touches Worker state (see PendingCleanup above for the same
    // reasoning on abandon_split_dispatch()).
    if (g_worker_gpu_mutex != nullptr) {
        std::lock_guard<std::mutex> lock(*g_worker_gpu_mutex);
        worker.report_prefetch_hints();
    } else {
        worker.report_prefetch_hints();
    }
    return 0;
}

} // namespace

bool self_bench_stats(uint64_t & n, uint64_t & min_us, uint64_t & mean_us) {
    if (g_probe.n == 0) return false;
    n = g_probe.n;
    min_us = g_probe.min_ns / 1000;
    mean_us = g_probe.total_ns / g_probe.n / 1000;
    return true;
}

void self_bench_tick(ggml_backend_t backend) {
    if (!g_probe.ready || g_probe.backend != backend) return;
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, g_probe.graph);
    const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (ns < g_probe.min_ns) g_probe.min_ns = ns;
    g_probe.total_ns += ns;
    ++g_probe.n;
}

ResourcePlan inspect_resources(const Options & options) {
    std::vector<std::string> devices = options.devices;
    if (devices.empty() && !options.device.empty()) {
        devices.push_back(options.device);
    }
    std::vector<int> device_slots = options.device_slots;
    if (device_slots.empty() && options.slots > 0) {
        device_slots.push_back(options.slots);
    }
    if (devices.empty() || devices.size() != device_slots.size()) {
        throw std::invalid_argument(
            "worker device and slot lists must have the same non-zero length");
    }
    const fs::path manifest   = fs::canonical(options.shard_manifest);
    const fs::path descriptor = fs::canonical(options.descriptor);
    Worker worker(
        load_catalog(manifest, descriptor),
        devices,
        device_slots,
        options.host_budget_bytes,
        options.host_victim_bytes,
        options.test_hooks,
        options.resident_expert_blocks, options.expert_reserve_blocks,
        options.expert_reserve_bytes);
    return worker.resources();
}

int run(const Options & options) {
    std::vector<std::string> devices = options.devices;
    if (devices.empty() && !options.device.empty()) {
        devices.push_back(options.device);
    }
    std::vector<int> device_slots = options.device_slots;
    if (device_slots.empty() && options.slots > 0) {
        device_slots.push_back(options.slots);
    }
    if (devices.empty() || devices.size() != device_slots.size() ||
        options.listen_host.empty() ||
        options.listen_port <= 0 || options.listen_port > 65535 ||
        std::any_of(device_slots.begin(), device_slots.end(), [](int slots) {
            return slots <= 0;
        })) {
        throw std::invalid_argument(
            "worker device and slot lists must have the same non-zero length and positive budgets");
    }
    // Same as WeightPager: default graph keying recaptures every submit
    // (nodes[0] is ephemeral; this worker rebinds expert data pointers).
    // CUDA graph opt-in is separate from the Vulkan persistent-plan mode.
    {
        if (!wp_hip_graphs_enabled()) {
            if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
                setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
            }
        }
    }
    const fs::path manifest = fs::canonical(options.shard_manifest);
    const fs::path descriptor = fs::canonical(options.descriptor);
    Worker worker(
        load_catalog(manifest, descriptor),
        devices,
        device_slots,
        options.host_budget_bytes,
        options.host_victim_bytes,
        options.test_hooks,
        options.resident_expert_blocks, options.expert_reserve_blocks,
        options.expert_reserve_bytes);
    const pipe_expert_hello advertised = worker.hello();
    const ResourcePlan & resources = worker.resources();

    // *** WP_WARMUP: pay the cold-start stalls BEFORE serving. Default OFF. ***
    //
    // MEASURED 2026-08-05 (fp3-r1, both workers). ALL of the slow submits and ALL
    // device allocations happen in the first ~140 requests, then stop dead:
    //     requests:      1   74   82   91   99  107  115  140  362  646 5794
    //     >=8ms submits: 1    2   10   19   27   35   43   44   44   44   44
    //     device allocs: 1    5    7    8    9   11   11   11   11   11   11
    // ns_submit is 1.833 s at request 140 and 4.677 s at request 5794 -- 39% of all
    // submit time is spent in 2.4% of the requests. The RX 480's first n_tokens=2
    // verify block cost 6044 ms, of which 5975 ms was that one worker.
    // This is NOT a Vulkan pathology: the CUDA worker shows 14 device allocations
    // and an 83.8 ms worst submit against the RX 480's 11 and 74.7 ms.
    //
    // run_self_bench does NOT cover this: it builds a ONE-EXPERT, ONE-TOKEN graph,
    // so it never touches verify widths, the prefill width, multi-expert graphs,
    // the gather path, or the SwiGLU clamp.
    //
    // Spec: WP_WARMUP="<tokens>x<experts>[,<tokens>x<experts>...]", e.g.
    //   WP_WARMUP="1x8,2x8,5x8,659x35"
    // Each entry replays a synthetic request through the REAL dispatch path so the
    // graph build, gallocr allocation, backend pipeline specialisation and clamp
    // ops are all compiled for that shape before the first real request arrives.
    // Results are discarded. Kept default-OFF so it can be A/B'd against the
    // measured baseline rather than silently changing the config of record.
    if (const char * warm = std::getenv("WP_WARMUP")) {
        if (warm[0] != '\0' && warm[0] != '0') {
            // ALL LAYERS, not just the first (2026-08-05, second iteration).
            // Warming one layer removed all 11 device allocations but left the
            // >=8 ms submit count at 43-44 -- and the worker serves 43 layers.
            // One stall per layer says the cost is per-LAYER tensor-set/descriptor
            // setup, not per-shape, so the shape list is swept on one layer while
            // every layer is touched at the cheapest shape.
            const int32_t n_embd = advertised.n_embd;
            const int32_t layer  = advertised.layers.empty() ? -1 : advertised.layers.front();
            const int32_t e_first = advertised.expert_first;
            const int32_t e_last  = advertised.expert_last;
            const auto t0 = std::chrono::steady_clock::now();
            size_t done = 0;
            std::string spec(warm);
            size_t pos = 0;
            while (pos <= spec.size() && layer >= 0 && e_first >= 0) {
                const size_t comma = spec.find(',', pos);
                const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
                const size_t x = item.find('x');
                if (x == std::string::npos) { continue; }
                const long toks = strtol(item.substr(0, x).c_str(), nullptr, 10);
                const long nexp = strtol(item.substr(x + 1).c_str(), nullptr, 10);
                if (toks <= 0 || nexp <= 0) { continue; }
                pipe_expert_dispatch_req req;
                req.layer        = layer;
                req.n_tokens     = (uint32_t) toks;
                req.swiglu_clamp = 10.0f;   // non-zero so the clamp ops get built
                const long avail = (long) (e_last - e_first + 1);
                for (long i = 0; i < nexp && i < avail; ++i) {
                    pipe_expert_assignment a;
                    a.expert_id = (int32_t) (e_first + i);
                    a.weights.assign((size_t) toks, 0.5f);
                    req.assignments.push_back(std::move(a));
                }
                req.activations.assign((size_t) toks * (size_t) n_embd, 0.01f);
                try {
                    RequestStats throwaway;
                    (void) worker.dispatch(req, throwaway);
                    ++done;
                } catch (const std::exception & e) {
                    // Warmup must never prevent serving. A shape we cannot warm is
                    // simply a shape that pays its stall on the first real request,
                    // which is exactly the status quo.
                    std::cout << "wp warmup: skipped " << item << ": " << e.what() << std::endl;
                }
            }
            // Second pass: touch EVERY served layer at the cheapest shape.
            size_t layers_done = 0;
            for (int32_t ly : advertised.layers) {
                if (ly == layer || e_first < 0) { continue; }   // first layer already covered
                pipe_expert_dispatch_req req;
                req.layer        = ly;
                req.n_tokens     = 1;
                req.swiglu_clamp = 10.0f;
                pipe_expert_assignment a;
                a.expert_id = e_first;
                a.weights.assign(1, 0.5f);
                req.assignments.push_back(std::move(a));
                req.activations.assign((size_t) n_embd, 0.01f);
                try {
                    RequestStats throwaway;
                    (void) worker.dispatch(req, throwaway);
                    ++layers_done;
                } catch (const std::exception &) {
                    // non-fatal, same reasoning as above
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::cout << "wp warmup: " << done << " shape(s) from \"" << warm
                      << "\" + " << layers_done << " extra layer(s) in "
                      << ms << " ms (results discarded)" << std::endl;
        }
    }

    std::cerr << "wp expert worker: read_path"
              << " direct=" << (worker.read_direct() ? "on" : "off")
              << " direct_fallback=" << (worker.read_direct_fallback() ? 1 : 0)
              << " inflight=" << worker.read_inflight()
              << " alignment=" << DIRECT_ALIGNMENT
              << " stats_interval_ms=" << g_read_path_stats.interval_ms()
              << std::endl;

    pipe_socket_ptr server =
        pipe_socket_t::create_server(options.listen_host.c_str(), options.listen_port);
    if (!server) {
        throw std::runtime_error(
            "failed to listen on " + options.listen_host + ":" +
            std::to_string(options.listen_port));
    }
    std::cout << "expert worker listening on " << options.listen_host << ":"
              << options.listen_port << " device=" << options.device
              << " experts=" << advertised.expert_first << "-"
              << advertised.expert_last << " layers=" << advertised.layers.size()
              << " slots=" << advertised.n_slots
              << " requested_slots=" << resources.requested_slots
              << " pinned_pages=" << worker.pinned_pages()
              << " pinned_bytes=" << resources.pinned_bytes
              << " slot_budget_bytes=" << resources.slot_budget_bytes
              << " device_bytes=" << resources.device_bytes
              << " size_classes=" << (resources.size_classes ? 1 : 0)
              << " staging=" << resources.staging_buffers << "x"
              << resources.staging_buffer_bytes
              << " read_inflight=" << worker.read_inflight()
              << " read_chunk_bytes=" << worker.read_chunk_bytes()
              << " read_direct=" << (worker.read_direct() ? 1 : 0)
              << " read_direct_fallback=" << (worker.read_direct_fallback() ? 1 : 0)
              << " read_alignment=" << DIRECT_ALIGNMENT
              << " host_budget=" << resources.host_budget_bytes
              << " host_victim_budget=" << options.host_victim_bytes
              << " partial_dtype=f32"
              << '\n';
    for (const SlotClass & slot_class : resources.slot_classes) {
        std::cout << "expert slot class bytes=" << slot_class.size
                  << " slots=" << slot_class.slots
                  << " pin_floor=" << slot_class.pin_floor
                  << " pages=" << slot_class.pages << '\n';
    }
    if (worker.device_count() > 1) {
        for (size_t i = 0; i < worker.device_count(); ++i) {
            const ResourcePlan & device_resources = worker.device_resources(i);
            std::cout << "expert worker device=" << worker.device_name(i)
                      << " slots=" << device_resources.slot_count
                      << " requested_slots=" << device_resources.requested_slots
                      << " budget_bytes=" << device_resources.device_budget_bytes
                      << " slot_budget_bytes=" << device_resources.slot_budget_bytes
                      << " resident_bytes=" << device_resources.pinned_bytes +
                          device_resources.device_bytes
                      << " pinned_bytes=" << device_resources.pinned_bytes
                      << " device_bytes=" << device_resources.device_bytes
                      << " staging=" << device_resources.staging_buffers << "x"
                      << device_resources.staging_buffer_bytes << '\n';
        }
    }
    // FLUSH. Everything above goes to stdout; the WARN lines around it go to
    // stderr, which is unbuffered. Under a redirect that means the startup
    // report -- slot counts, per-class breakdown, per-device budgets, the one
    // place the worker says what it actually allocated -- sits in the buffer
    // while the process runs, so a log tailed during a run appears to end at
    // the read-path line and the numbers you need are invisible. Cost me a
    // diagnosis today; the banner is worth nothing if it arrives at exit.
    std::cout << std::flush;

    // WP_WORKER_MULTI_CONN=N (N>=2) -- see g_worker_gpu_mutex comment above
    // serve_connection for the lock design. Unset/absent/"1"/anything <2 is
    // the untouched default path below: strictly one connection at a time,
    // byte-identical to before this flag existed.
    //
    // PERSISTENT SERVING, not accept-exactly-N-then-exit (that was the
    // throwaway probe's shape): N slot threads each loop
    // accept() -> serve_connection() -> accept() forever. When a
    // connection closes, serve_connection() returns, that thread's slot is
    // immediately free, and its very next accept() call can pick up a new
    // client -- so up to N connections are live at once, not exactly N for
    // the process's whole lifetime. Multiple threads blocked in accept() on
    // the SAME listening socket is well-defined POSIX behaviour (the kernel
    // hands each ready connection to exactly one waiting caller); this is
    // the standard thread-pool-acceptor pattern, not something specific to
    // this socket wrapper.
    //
    // SHUTDOWN: this loop has no in-process stop condition -- an
    // orchestrator drives connect/disconnect cycles, this worker does not
    // decide when it's done. No signal handler is installed here (same as
    // the single-connection default path above), so SIGTERM's default
    // disposition (terminate) applies immediately, even with every slot
    // thread blocked in accept() or mid-request. The orchestrator-visible
    // change from the old accept-exactly-N probe is: this worker no longer
    // exits on its own once the streams close, so re-running the probe
    // comparison against this version means killing the process
    // (SIGTERM/SIGKILL, exactly as the single-connection path has always
    // required) instead of waiting on it -- see the run recipe in the task
    // report for the concrete command.
    int multi_conn_n = 1;
    if (const char * e = std::getenv("WP_WORKER_MULTI_CONN")) {
        const long v = strtol(e, nullptr, 10);
        if (v >= 2) {
            multi_conn_n = (int) v;
        }
    }

    if (multi_conn_n >= 2) {
        std::mutex serialize_mutex;
        g_worker_gpu_mutex = worker.multi_device() ? nullptr : &serialize_mutex;
        g_worker_conn_request_counts = std::vector<std::atomic<uint64_t>>((size_t) multi_conn_n);
        g_worker_staging_held = std::vector<std::atomic<int64_t>>((size_t) multi_conn_n);
        // Per-connection staging quota (see StagingPool's 2026-08-25 deadlock
        // fix comment) -- must be set before any connection thread starts so
        // no borrow() call can race the quota changing under it.
        worker.set_staging_multi_conn(multi_conn_n);
        std::vector<std::thread> threads;
        threads.reserve((size_t) multi_conn_n);
        pipe_socket_t * const server_raw = server.get();
        for (int i = 0; i < multi_conn_n; ++i) {
            threads.emplace_back([server_raw, &worker, i] {
                for (;;) {
                    pipe_socket_ptr client = server_raw->accept();
                    if (!client) {
                        // Listening socket is gone (closed/errored): this
                        // slot has nothing left to do. Other slots may
                        // still be serving live connections -- only this
                        // one thread exits.
                        return;
                    }
                    std::cout << "wp multi-conn: slot " << i
                              << " accepted a connection" << std::endl;
                    (void) serve_connection(*client, worker, i);
                    std::cout << "wp multi-conn: slot " << i
                              << " connection closed, awaiting a new one"
                              << std::endl;
                }
            });
        }
        // Runs until the process is killed (see SHUTDOWN above) -- these
        // threads never return on their own in normal operation.
        for (auto & t : threads) { t.join(); }
        g_worker_gpu_mutex = nullptr;
        return 0;
    }

    int result = 0;
    do {
        pipe_socket_ptr client = server->accept();
        if (!client) {
            return 1;
        }
        result = serve_connection(*client, worker);
    } while (!options.once);
    return result;
}

namespace {

bool trunk_inproc_enabled() {
    const char * e = std::getenv("WP_TRUNK_INPROC");
    return e != nullptr && std::strtol(e, nullptr, 10) == 1;
}

struct trunk_shard_spec {
    int                   port = 0;
    std::string           device;
    int                   slots = 0;
    std::filesystem::path manifest;
    std::filesystem::path descriptor;
    uint64_t              host_victim_bytes = 0;
};

// WP_TRUNK_INPROC_SHARDS=
//   port,device,slots,manifest,descriptor[,host_victim_bytes][;port,...]
std::vector<trunk_shard_spec> parse_trunk_shards() {
    const char * e = std::getenv("WP_TRUNK_INPROC_SHARDS");
    if (e == nullptr || e[0] == '\0') {
        throw std::runtime_error("WP_TRUNK_INPROC=1 requires WP_TRUNK_INPROC_SHARDS");
    }
    std::vector<trunk_shard_spec> specs;
    std::string text = e;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t semi = text.find(';', begin);
        const std::string item = text.substr(
            begin, semi == std::string::npos ? std::string::npos : semi - begin);
        if (!item.empty()) {
            std::vector<std::string> fields;
            size_t f0 = 0;
            while (f0 <= item.size()) {
                const size_t comma = item.find(',', f0);
                fields.push_back(item.substr(
                    f0, comma == std::string::npos ? std::string::npos : comma - f0));
                if (comma == std::string::npos) {
                    break;
                }
                f0 = comma + 1;
            }
            if (fields.size() < 5 || fields.size() > 6) {
                throw std::runtime_error(
                    "WP_TRUNK_INPROC_SHARDS entry must be "
                    "port,device,slots,manifest,descriptor[,host_victim_bytes]");
            }
            trunk_shard_spec spec;
            spec.port = std::atoi(fields[0].c_str());
            spec.device = fields[1];
            spec.slots = std::atoi(fields[2].c_str());
            spec.manifest = fields[3];
            spec.descriptor = fields[4];
            if (fields.size() == 6) {
                spec.host_victim_bytes = (uint64_t) std::strtoull(fields[5].c_str(), nullptr, 10);
            }
            if (spec.port <= 0 || spec.port > 65535 || spec.slots <= 0 ||
                    spec.device.empty() || spec.manifest.empty() || spec.descriptor.empty()) {
                throw std::runtime_error("WP_TRUNK_INPROC_SHARDS has an invalid entry: " + item);
            }
            specs.push_back(std::move(spec));
        }
        if (semi == std::string::npos) {
            break;
        }
        begin = semi + 1;
    }
    if (specs.empty()) {
        throw std::runtime_error("WP_TRUNK_INPROC_SHARDS is empty");
    }
    return specs;
}

class InProcessEngine final : public pipe_expert_dispatcher::inproc_backend {
    Worker worker_;

  public:
    explicit InProcessEngine(const trunk_shard_spec & spec) :
        worker_(
            load_catalog(fs::canonical(spec.manifest), fs::canonical(spec.descriptor)),
            spec.device,
            spec.slots,
            /*host_budget_bytes=*/0,
            spec.host_victim_bytes,
            /*test_hooks=*/nullptr,
            /*resident_expert_blocks=*/{},
            /*expert_reserve_blocks=*/{},
            /*expert_reserve_bytes=*/0) {}

    pipe_expert_hello hello() override {
        return worker_.hello();
    }

    pipe_expert_partial dispatch(const pipe_expert_dispatch_req & request) override {
        RequestStats stats;
        return worker_.dispatch(request, stats);
    }
};

std::unique_ptr<pipe_expert_dispatcher::inproc_backend>
make_inproc_backend(const pipe_expert_dispatcher::endpoint & target) {
    if (!trunk_inproc_enabled()) {
        return nullptr;
    }
    static const std::vector<trunk_shard_spec> specs = parse_trunk_shards();
    for (const trunk_shard_spec & spec : specs) {
        if (spec.port != target.port) {
            continue;
        }
        {
            if (!wp_hip_graphs_enabled()) {
                if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
                    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
                }
            }
        }
        std::fprintf(stderr,
                     "wp expert worker: in-process %s:%d device=%s slots=%d\n",
                     target.host.c_str(), target.port, spec.device.c_str(), spec.slots);
        return std::unique_ptr<pipe_expert_dispatcher::inproc_backend>(
            new InProcessEngine(spec));
    }
    return nullptr;
}

} // namespace

void install_inproc_factory() {
    pipe_expert_dispatcher::set_inproc_backend_factory(&make_inproc_backend);
}

} // namespace wp_expert_worker
