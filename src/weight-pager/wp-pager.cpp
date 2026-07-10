#include "wp-pager.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "../../ggml/src/ggml-impl.h"
#include "llama-impl.h"  // LLAMA_LOG_*
#include "wp-eval-cb.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>      // getenv, setenv, unsetenv, malloc, free
#include <exception>
#include <fstream>     // WP_ROUTE_TRACE diagnostic dump
#include <cstring>
#include <fcntl.h>
#include <limits.h>
#include <new>          // placement new
#include <utility>
#include <unistd.h>     // close(), pread, readlink

#if defined(LLAMA_HAVE_IO_URING) && defined(__linux__)
#include <liburing.h>
#endif

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>

extern "C++" void ggml_cuda_get_routed_expert_ptrs_stats(
    uint64_t * set, uint64_t * consumed, uint64_t * discarded_unconsumed);
#endif

namespace wp {

// ---------------------------------------------------------------------------
// EnvSnapshot — record current value of an env var, then restore it later.
//
// Used for GGML_CUDA_DISABLE_GRAPHS: the pager forces it to "1" during init
// unless WP_HIP_GRAPHS=1 (default off). With WP_HIP_GRAPHS unset or 0 this
// preserves today's safety path exactly: graph capture stays disabled because
// the eval callback rewrites tensor->data before execution. With
// WP_HIP_GRAPHS=1 we leave the user's graph setting alone so MAD-P1 graph
// update handling can be tested. On shutdown we restore whatever value (or
// absence) the user had before when we forced it. This fixes B-P5: the
// previous pager set the var unconditionally and never restored it, leaking
// the "graphs disabled" state into any subsequent model load in the same
// process.
// ---------------------------------------------------------------------------

namespace {

void env_snapshot(const char * var, bool & present_out, std::string & prior_out) {
    const char * v = std::getenv(var);
    if (v != nullptr) {
        present_out = true;
        prior_out   = v;
    } else {
        present_out = false;
        prior_out.clear();
    }
}

void env_restore(const char * var, bool present, const std::string & prior) {
    if (present) {
        setenv(var, prior.c_str(), /*overwrite=*/1);
    } else {
        unsetenv(var);
    }
}

constexpr const char * kEnvDisableGraphs = "GGML_CUDA_DISABLE_GRAPHS";
constexpr const char * kEnvWpHipGraphs   = "WP_HIP_GRAPHS";
constexpr const char * kEnvWpAsyncEnsure = "WP_ASYNC_ENSURE";
constexpr const char * kEnvWpHostBudgetBytes = "WP_HOST_BUDGET_BYTES";
constexpr const char * kEnvWpGraphPinMax = "WP_GRAPH_PIN_MAX";
constexpr const char * kEnvWpStickyL2 = "WP_STICKY_L2";
constexpr const char * kEnvWpStickyL2Pages = "WP_STICKY_L2_PAGES";
constexpr const char * kEnvWpStickyL2Stats = "WP_STICKY_L2_STATS";

bool env_flag_is_one(const char * var) {
    const char * v = std::getenv(var);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

size_t env_size_bytes(const char * var) {
    const char * v = std::getenv(var);
    if (v == nullptr || v[0] == '\0') {
        return 0;
    }

    errno = 0;
    char * end = nullptr;
    unsigned long long n = std::strtoull(v, &end, 10);
    if (errno != 0 || end == v || (end != nullptr && *end != '\0')) {
        LLAMA_LOG_WARN("wp::WeightPager: ignoring invalid %s=%s\n", var, v);
        return 0;
    }
    return (size_t) n;
}

int env_nonnegative_int(const char * var, int fallback) {
    const char * v = std::getenv(var);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }

    errno = 0;
    char * end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (errno != 0 || end == v || (end != nullptr && *end != '\0') || n < 0) {
        LLAMA_LOG_WARN("wp::WeightPager: ignoring invalid %s=%s\n", var, v);
        return fallback;
    }
    return (int) n;
}

int clamp_int(int n, int lo, int hi) {
    return std::max(lo, std::min(n, hi));
}

void log_graph_pin_degrade(int page_idx, int slot, int max_slots) {
    static int s_logs = 0;
    if (s_logs < 8) {
        LLAMA_LOG_WARN("wp::WeightPager: graph pin cap reached for page=%d slot=%d "
                       "(WP_GRAPH_PIN_MAX effective cap=%d); using non-graph pointer for this page\n",
                       page_idx, slot, max_slots);
    } else if (s_logs == 8) {
        LLAMA_LOG_WARN("wp::WeightPager: suppressing further graph pin cap logs\n");
    }
    ++s_logs;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool zero_device_padding(void * dst_vram, size_t payload_size, size_t slot_size) {
    if (slot_size <= payload_size) return true;
    if (dst_vram == nullptr) return false;
#if defined(GGML_USE_HIP)
    hipError_t err = hipMemset((char *) dst_vram + payload_size, 0,
                               slot_size - payload_size);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: p2p padding hipMemset failed: %s\n",
                       hipGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

}  // anonymous namespace

bool WeightPager::ensure_host_bufs_ready_(size_t n, size_t page_bytes) {
    if (n == 0 || page_bytes == 0) {
        return false;
    }
    // O_DIRECT bounce: align-down prefix (<=511) + size pad to 512.
    const size_t need = page_bytes + 512 + 512;
    const size_t alloc = (need + 4095) & ~(size_t) 4095;
    if (ensure_host_bufs_.size() >= n && ensure_host_buf_bytes_ >= alloc) {
        return true;
    }
    free_ensure_host_bufs_();
    // Cap pool so we never pin huge prefill unions (same spirit as batch cap).
    const size_t cap = std::max(n, (size_t) 32);
    ensure_host_bufs_.assign(cap, nullptr);
    ensure_host_buf_bytes_ = alloc;
    ensure_host_bufs_pinned_ = false;
#if defined(GGML_USE_HIP)
    static const int s_pageable = [](){ const char* e=std::getenv("WP_ODIRECT_PAGEABLE"); return (e&&e[0]=='1')?1:0; }();
    bool all_ok = !s_pageable;
    for (size_t i = 0; !s_pageable && i < cap; ++i) {
        void * p = nullptr;
        // hipHostMalloc is page-aligned (O_DIRECT-safe) and faster for H2D.
        if (hipHostMalloc(&p, alloc, hipHostMallocDefault) != hipSuccess || p == nullptr) {
            all_ok = false;
            break;
        }
        ensure_host_bufs_[i] = p;
    }
    if (all_ok) {
        ensure_host_bufs_pinned_ = true;
        return true;
    }
    free_ensure_host_bufs_();
#endif
    // 4 KiB-aligned malloc fallback for O_DIRECT.
    ensure_host_bufs_.assign(cap, nullptr);
    for (size_t i = 0; i < cap; ++i) {
        void * p = nullptr;
        if (posix_memalign(&p, 4096, alloc) != 0 || p == nullptr) {
            free_ensure_host_bufs_();
            return false;
        }
        ensure_host_bufs_[i] = p;
    }
    ensure_host_buf_bytes_ = alloc;
    ensure_host_bufs_pinned_ = false;
    return true;
}

void WeightPager::free_ensure_host_bufs_() {
    shutdown_ensure_odirect_workers_();
#if defined(LLAMA_HAVE_IO_URING) && defined(__linux__)
    shutdown_ensure_odirect_ring_();
#endif
    for (void * p : ensure_host_bufs_) {
        if (p == nullptr) continue;
#if defined(GGML_USE_HIP)
        if (ensure_host_bufs_pinned_) {
            hipHostFree(p);
        } else
#endif
        {
            std::free(p);
        }
    }
    ensure_host_bufs_.clear();
    ensure_host_buf_bytes_ = 0;
    ensure_host_bufs_pinned_ = false;
    for (int fd : ensure_odirect_fds_) {
        if (fd >= 0) {
            close(fd);
        }
    }
    ensure_odirect_fds_.clear();
}

int WeightPager::ensure_odirect_worker_count_(size_t n_jobs) const {
    if (n_jobs == 0) {
        return 0;
    }

    static const int s_env_workers = []() {
        const char * e = std::getenv("WP_ODIRECT_READ_WORKERS");
        if (e == nullptr || e[0] == '\0') {
            return -1;
        }
        errno = 0;
        char * end = nullptr;
        long n = std::strtol(e, &end, 10);
        if (errno != 0 || end == e || (end != nullptr && *end != '\0') || n <= 0) {
            LLAMA_LOG_WARN("wp::WeightPager: ignoring invalid WP_ODIRECT_READ_WORKERS=%s\n", e);
            return -1;
        }
        return (int) n;
    }();

    int n = s_env_workers;
    if (n <= 0) {
        n = cfg_.io_uring_depth;
        if (n <= 0) {
            n = cfg_.prefetch_depth;
        }
    }
    if (n <= 0) {
        n = 1;
    }
    n = std::min(n, (int) n_jobs);
    return clamp_int(n, 1, 64);
}

bool WeightPager::ensure_odirect_workers_ready_(int n_workers) {
    if (n_workers <= 0) {
        return false;
    }
    if ((int) ensure_odirect_workers_.size() >= n_workers) {
        return true;
    }
    try {
        while ((int) ensure_odirect_workers_.size() < n_workers) {
            ensure_odirect_workers_.emplace_back(&WeightPager::ensure_odirect_worker_loop_, this);
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("wp::WeightPager: failed to start O_DIRECT worker: %s\n", e.what());
        return false;
    } catch (...) {
        LLAMA_LOG_WARN("wp::WeightPager: failed to start O_DIRECT worker\n");
        return false;
    }
    static int s_worker_log = 0;
    if (s_worker_log < 1) {
        LLAMA_LOG_WARN("wp::WeightPager: O_DIRECT host path using %zu persistent read workers\n",
                       ensure_odirect_workers_.size());
        ++s_worker_log;
    }
    return true;
}

void WeightPager::shutdown_ensure_odirect_workers_() {
    {
        std::lock_guard<std::mutex> lock(ensure_odirect_mu_);
        ensure_odirect_workers_stop_ = true;
    }
    ensure_odirect_cv_.notify_all();
    for (std::thread & t : ensure_odirect_workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    ensure_odirect_workers_.clear();
    {
        std::lock_guard<std::mutex> lock(ensure_odirect_mu_);
        ensure_odirect_queue_.clear();
        ensure_odirect_workers_stop_ = false;
    }
}

void WeightPager::ensure_odirect_worker_loop_() {
    for (;;) {
        EnsureODirectReadJob * job = nullptr;
        {
            std::unique_lock<std::mutex> lock(ensure_odirect_mu_);
            ensure_odirect_cv_.wait(lock, [this]() {
                return ensure_odirect_workers_stop_ || !ensure_odirect_queue_.empty();
            });
            if (ensure_odirect_workers_stop_ && ensure_odirect_queue_.empty()) {
                return;
            }
            job = ensure_odirect_queue_.front();
            ensure_odirect_queue_.pop_front();
        }

        bool ok = false;
        int err = 0;
        if (job != nullptr && job->fd >= 0 && job->dst != nullptr && job->size > 0) {
            size_t total = 0;
            while (total < job->size) {
                const ssize_t n = pread(job->fd, (char *) job->dst + total,
                                        job->size - total,
                                        (off_t) (job->off + total));
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    err = -errno;
                    break;
                }
                if (n == 0) {
                    err = -EIO;
                    break;
                }
                total += (size_t) n;
            }
            ok = (total == job->size);
        } else {
            err = -EINVAL;
        }

        {
            std::lock_guard<std::mutex> lock(ensure_odirect_mu_);
            if (job != nullptr) {
                job->ok = ok;
                job->err = err;
                job->done = true;
            }
        }
        ensure_odirect_done_cv_.notify_all();
    }
}

#if defined(LLAMA_HAVE_IO_URING) && defined(__linux__)
void WeightPager::shutdown_ensure_odirect_ring_() {
    if (ensure_odirect_ring_ != nullptr) {
        if (ensure_odirect_ring_files_registered_) {
            io_uring_unregister_files(ensure_odirect_ring_);
        }
        io_uring_queue_exit(ensure_odirect_ring_);
        delete ensure_odirect_ring_;
        ensure_odirect_ring_ = nullptr;
    }
    ensure_odirect_fd_registered_.clear();
    ensure_odirect_ring_files_registered_ = false;
}

bool WeightPager::ensure_odirect_ring_ready_(size_t n_entries) {
    if (ensure_odirect_ring_ != nullptr) {
        return true;
    }

    unsigned file_cap = 1;
    for (int i = 0; i < catalog_.size(); ++i) {
        const PageMeta & m = catalog_.at(i);
        if (!m.is_pinned && (unsigned) m.file_idx >= file_cap) {
            file_cap = (unsigned) m.file_idx + 1;
        }
    }

    int queue_depth = cfg_.io_uring_depth;
    if (queue_depth <= 0) {
        queue_depth = cfg_.prefetch_depth;
    }
    if (queue_depth < (int) n_entries) {
        queue_depth = (int) n_entries;
    }
    if (queue_depth <= 0) {
        queue_depth = 1;
    }

    ensure_odirect_ring_ = new io_uring();
    std::memset(ensure_odirect_ring_, 0, sizeof(*ensure_odirect_ring_));

    int ret = io_uring_queue_init(queue_depth, ensure_odirect_ring_, 0);
    if (ret < 0) {
        LLAMA_LOG_WARN("wp::ensure_odirect_ring_ready_: queue_init failed: %s\n",
                       strerror(-ret));
        delete ensure_odirect_ring_;
        ensure_odirect_ring_ = nullptr;
        return false;
    }

    ret = io_uring_register_files_sparse(ensure_odirect_ring_, file_cap);
    if (ret < 0) {
        LLAMA_LOG_WARN("wp::ensure_odirect_ring_ready_: register_files_sparse failed: %s\n",
                       strerror(-ret));
        io_uring_queue_exit(ensure_odirect_ring_);
        delete ensure_odirect_ring_;
        ensure_odirect_ring_ = nullptr;
        return false;
    }

    ensure_odirect_ring_files_registered_ = true;
    ensure_odirect_fd_registered_.assign(file_cap, 0);
    return true;
}

bool WeightPager::ensure_odirect_fixed_fd_ready_(int file_idx, int fd, size_t n_entries) {
    if (file_idx < 0 || fd < 0) {
        return false;
    }
    if (!ensure_odirect_ring_ready_(n_entries)) {
        return false;
    }
    if ((size_t) file_idx >= ensure_odirect_fd_registered_.size()) {
        return false;
    }
    if (ensure_odirect_fd_registered_[(size_t) file_idx]) {
        return true;
    }

    int upd = fd;
    int ret = io_uring_register_files_update(ensure_odirect_ring_, (unsigned) file_idx, &upd, 1);
    if (ret < 0) {
        static int s_reg_warn = 0;
        if (s_reg_warn < 3) {
            LLAMA_LOG_WARN("wp::ensure_odirect_fixed_fd_ready_: register fd %d failed: %s\n",
                           file_idx, strerror(-ret));
            ++s_reg_warn;
        }
        return false;
    }
    if (ret != 1) {
        return false;
    }

    ensure_odirect_fd_registered_[(size_t) file_idx] = 1;
    return true;
}
#endif

int WeightPager::ensure_odirect_fd_(int file_idx) {
    if (file_io_ == nullptr || file_idx < 0) {
        return -1;
    }
    if ((size_t) file_idx >= ensure_odirect_fds_.size()) {
        ensure_odirect_fds_.resize((size_t) file_idx + 1, -1);
    }
    if (ensure_odirect_fds_[(size_t) file_idx] >= 0) {
        return ensure_odirect_fds_[(size_t) file_idx];
    }
    const int src = file_io_->fd(file_idx);
    if (src < 0) {
        return -1;
    }
#ifdef O_DIRECT
    // Resolve the path of the (buffered) model fd and re-open O_DIRECT.
    // GGUF offsets are not sector-aligned; callers must bounce-align each read.
    char link[64];
    char path[PATH_MAX];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%d", src);
    const ssize_t n = ::readlink(link, path, sizeof(path) - 1);
    if (n <= 0) {
        return -1;
    }
    path[n] = '\0';
    const int od = ::open(path, O_RDONLY | O_DIRECT);
    if (od < 0) {
        static int s_od_warn = 0;
        if (s_od_warn < 3) {
            LLAMA_LOG_WARN("wp::ensure_odirect_fd_: open(O_DIRECT) failed for %s: %s\n",
                           path, strerror(errno));
            ++s_od_warn;
        }
        return -1;
    }
    ensure_odirect_fds_[(size_t) file_idx] = od;
    return od;
#else
    (void) src;
    return -1;
#endif
}

// ---------------------------------------------------------------------------
// WeightPager
// ---------------------------------------------------------------------------

WeightPager::~WeightPager() {
    shutdown();
}

int WeightPager::register_pinned(const std::string & name, void * device_ptr, size_t bytes) {
    return catalog_.add_pinned(name, device_ptr, bytes);
}

int WeightPager::add_page(const std::string & name, uint16_t file_idx,
                          uint64_t file_offset, size_t size, int n_experts,
                          ggml_backend_buffer_type_t buft) {
    const int n_before = catalog_.size();
    int first = -1;
    // Non-MoE / per-expert tensor: add as-is.
    if (n_experts <= 1) {
        first = catalog_.add(name, file_idx, file_offset, size);
    } else {
        // Consolidated MoE tensor: register N sub-pages, one per expert.
        // Returns the index of the FIRST sub-page (subsequent experts are at
        // sequential indices). Per-expert size is the consolidated size
        // divided by n_experts; per-expert offset is base_offset + e * size_e.
        first = catalog_.add_consolidated_experts(name, file_idx, file_offset, size, n_experts);
    }
    const int n_after = catalog_.size();
    if ((int) page_buft_.size() < n_after) {
        page_buft_.resize((size_t) n_after, nullptr);
    }
    for (int i = n_before; i < n_after; ++i) {
        page_buft_[(size_t) i] = buft;
    }
    return first;
}

void build_expert_page_index(const PageCatalog & catalog,
                             std::map<std::pair<int,int>, std::vector<int>> & out) {
    out.clear();
    for (int i = 0; i < catalog.size(); ++i) {
        const PageMeta & m = catalog.at(i);
        if (!m.is_expert || m.expert_idx < 0 || m.block_idx < 0) continue;
        out[std::make_pair((int) m.block_idx, (int) m.expert_idx)].push_back(i);
    }
}

void WeightPager::expert_sister_pages(int block_idx, int expert_idx,
                                      std::vector<int> & out) const {
    auto it = expert_page_index_.find(std::make_pair(block_idx, expert_idx));
    if (it == expert_page_index_.end()) return;
    out.insert(out.end(), it->second.begin(), it->second.end());
}

void WeightPager::note_router_weight(int block_idx, const float * W, int n_expert, int n_embd) {
    predictor_.set_router(block_idx, W, n_expert, n_embd);
}

bool WeightPager::predictor_has_router(int block_idx) const {
    return predictor_.has_router(block_idx);
}

void WeightPager::submit_xlayer_prefetch(const float * h, int from_layer) {
    if (!initialized_ || !xlayer_prefetch_enabled_ || h == nullptr) return;
    std::vector<ExpertRef> refs;
    predictor_.predict(h, from_layer, xlayer_lookahead_k_, xlayer_topk_, n_layer_, refs);
    if (refs.empty()) return;
    std::vector<int> pages;
    for (const ExpertRef & r : refs) {
        expert_sister_pages(r.layer, r.expert, pages);
    }
    if (pages.empty()) return;
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    std::vector<int> fresh;
    fresh.reserve(pages.size());
    for (int p : pages) {
        if (p < 0 || p >= catalog_.size()) continue;
        if (catalog_.at(p).is_pinned) continue;
        if (page_to_slot_[p] >= 0) continue;   // resident or already in flight
        fresh.push_back(p);
    }
    if (fresh.empty()) return;
    // Speculative slot budget: never let speculation exceed the cap.
    if (xlayer_max_slots_ > 0) {
        const int budget = xlayer_max_slots_ - pool_.n_speculative();
        if (budget <= 0) return;
        if ((int) fresh.size() > budget) fresh.resize((size_t) budget);
    }
    mark_cross_layer_prefetch_candidates(fresh);
    prefetch_pages_batch(fresh, /*count_dense_prefetch=*/false, /*allow_evict=*/true, /*speculative=*/true);
}

bool WeightPager::init(const Config &             cfg,
                       ggml_backend_buffer_type_t device_buft,
                       int                        device_idx,
                       std::vector<int>           fds,
                       const std::vector<int> &   devices_used) {
    if (initialized_) {
        LLAMA_LOG_WARN("wp::WeightPager: init called twice — ignoring\n");
        return false;
    }
    if (catalog_.size() == 0) {
        LLAMA_LOG_WARN("wp::WeightPager::init: no pages registered\n");
        return false;
    }

    build_expert_page_index(catalog_, expert_page_index_);

    if (devices_used.size() > 1) {
        LLAMA_LOG_WARN(
            "wp::WeightPager::init: model uses %zu devices; pager pool remains on configured "
            "paging device %d and per-page buffer selection is enabled\n",
            devices_used.size(), device_idx);
    }
    if (!devices_used.empty() && devices_used.front() != device_idx) {
        LLAMA_LOG_WARN("wp::WeightPager::init: device mismatch (used=%d, configured=%d)\n",
                       devices_used.front(), device_idx);
    }

    stats_ = Stats{};
    cfg_ = cfg;
    if (cfg_.n_slots <= 0) cfg_.n_slots         = catalog_.size();  // pin everything if user didn't pick
    if (cfg_.prefetch_depth <= 0) cfg_.prefetch_depth = 4;
    if (cfg_.io_uring_depth <= 0) cfg_.io_uring_depth = cfg_.prefetch_depth;
    if (cfg_.io_uring_depth < cfg_.prefetch_depth) {
        LLAMA_LOG_WARN("wp::WeightPager: io_uring_depth=%d below prefetch_depth=%d; raising to %d\n",
                       cfg_.io_uring_depth, cfg_.prefetch_depth, cfg_.prefetch_depth);
        cfg_.io_uring_depth = cfg_.prefetch_depth;
    }

    // Snapshot env BEFORE we touch it.
    env_snapshot(kEnvDisableGraphs, env_was_present_, env_prior_value_);
    hip_graphs_enabled_ = env_flag_is_one(kEnvWpHipGraphs);
    async_ensure_enabled_ = env_flag_is_one(kEnvWpAsyncEnsure);
    sticky_l2_enabled_ = env_flag_is_one(kEnvWpStickyL2);
    sticky_l2_max_pages_ = clamp_int(env_nonnegative_int(kEnvWpStickyL2Pages, 32), 1, 256);
    sticky_l2_stats_ = env_flag_is_one(kEnvWpStickyL2Stats);
    env_disable_graphs_forced_ = !hip_graphs_enabled_;
    if (env_disable_graphs_forced_) {
        setenv(kEnvDisableGraphs, "1", /*overwrite=*/1);
    } else {
        LLAMA_LOG_WARN("wp::WeightPager: WP_HIP_GRAPHS=1, leaving GGML_CUDA_DISABLE_GRAPHS unchanged\n");
    }

    const size_t slot_size = catalog_.max_page_size();
    if (slot_size == 0) {
        LLAMA_LOG_WARN("wp::WeightPager::init: catalog max_page_size is 0\n");
        restore_disable_graphs_env_();
        return false;
    }

    // 1. VRAM pool. Pass device_idx so the pool can detect UMA / APU devices
    //    (Strix Halo etc.) and refuse oversized requests up front rather than
    //    silently allocating from system RAM and triggering a swap storm.
    //    MAD-234.
    if (!pool_.init(device_buft, cfg_.n_slots, slot_size, device_idx)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: pool allocation failed\n");
        restore_disable_graphs_env_();
        return false;
    }
    pool_.set_eviction_callback([this](int slot_idx) { on_pool_evict_(slot_idx); });

    // MAD-237: opt-in hot-slot protection. Off by default (threshold=0 ⇒
    // pure LRU = MAD-231 behaviour). Operators enable via env when their
    // workload has measurable expert reuse (typically MoE with skewed
    // routing distribution). Cheap to leave off; nothing else changes.
    if (const char * env = std::getenv("WP_HOT_HIT_THRESHOLD")) {
        long t = std::strtol(env, nullptr, 10);
        if (t < 0) t = 0;
        pool_.set_hot_hit_threshold((uint32_t) t);
        if (t > 0) {
            LLAMA_LOG_INFO("wp::WeightPager: WP_HOT_HIT_THRESHOLD=%ld "
                           "(slots with hit_count > %ld are skipped in LRU "
                           "eviction Pass A)\n", t, t);
        }
    }

    // Cross-layer prefetch (WP_PREFETCH_XLAYER). Off by default => default path
    // stays byte-identical. Router weights are captured lazily in the eval cb.
    {
        n_layer_ = 0;
        for (int i = 0; i < catalog_.size(); ++i) {
            const int b = catalog_.at(i).block_idx;
            if (b + 1 > n_layer_) n_layer_ = b + 1;
        }
        if (const char * e = std::getenv("WP_PREFETCH_XLAYER"))      xlayer_prefetch_enabled_ = (e[0] == '1');
        if (const char * e = std::getenv("WP_PREFETCH_LOOKAHEAD_K")) { long v = std::strtol(e,nullptr,10); if (v > 0) xlayer_lookahead_k_ = (int) v; }
        if (const char * e = std::getenv("WP_PREFETCH_TOPK"))        { long v = std::strtol(e,nullptr,10); if (v > 0) xlayer_topk_ = (int) v; }
        xlayer_max_slots_ = pool_.n_slots() / 4;
        if (const char * e = std::getenv("WP_PREFETCH_MAX_SLOTS"))   { long v = std::strtol(e,nullptr,10); if (v >= 0) xlayer_max_slots_ = (int) v; }
        if (xlayer_prefetch_enabled_) {
            LLAMA_LOG_INFO("wp::xlayer prefetch: on K=%d M=%d cap=%d n_layer=%d\n",
                           xlayer_lookahead_k_, xlayer_topk_, xlayer_max_slots_, n_layer_);
        }
    }

    // 2. Per-device transfer stream + event pool. Size events generously
    //    so prefetch never blocks waiting for an event.
    const int n_transport_events = async_ensure_enabled_
        ? cfg_.prefetch_depth * 2 + 8
        : cfg_.prefetch_depth + 2;
    if (!transport_.init(device_idx, n_transport_events, async_ensure_enabled_)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: gpu transport init failed\n");
        pool_.~PoolAllocator();   // explicit teardown via dtor (RAII'd)
        new (&pool_) PoolAllocator{};
        restore_disable_graphs_env_();
        return false;
    }

    // 3. File IO layer (io_uring or pread).
    const FileIOP2PConfig p2p_cfg{
        pool_.pool_base(),
        pool_.pool_size(),
    };
    file_io_ = create_file_io(std::move(fds), cfg_.prefer_async_io,
                              cfg_.io_uring_depth, &p2p_cfg);
    if (!file_io_) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: file IO layer init failed\n");
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        restore_disable_graphs_env_();
        return false;
    }

    // 4. Prefetch scheduler bound to the above.
    if (!prefetch_.init(file_io_.get(), &transport_, slot_size, cfg_.prefetch_depth,
                        async_ensure_enabled_)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: prefetch scheduler init failed\n");
        file_io_.reset();
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        restore_disable_graphs_env_();
        return false;
    }

    // 5. Page-to-slot map + per-page loaded flag.
    page_to_slot_.assign((size_t) catalog_.size(), -1);
    page_loaded_.assign((size_t)  catalog_.size(), false);
    cross_layer_prefetch_candidate_.assign((size_t) catalog_.size(), false);
    draft_tid2eid_mark_.assign((size_t) catalog_.size(), false);
    oracle_pred_mark_.assign((size_t) catalog_.size(), false);
    oracle_pred_hit_.assign((size_t) catalog_.size(), false);
    sticky_l2_score_.assign((size_t) catalog_.size(), 0);
    sticky_l2_mark_.assign((size_t) catalog_.size(), false);
    sticky_l2_pinned_.assign((size_t) catalog_.size(), false);
    sticky_l2_pages_.clear();
    sticky_l2_hits_since_refresh_ = 0;
    prefetch_started_at_.assign((size_t) catalog_.size(), std::chrono::steady_clock::time_point{});
    page_async_event_.assign((size_t) catalog_.size(), -1);
    slot_to_page_.assign((size_t) pool_.n_slots(), -1);

    // 6. Shared sync staging buffer (max_page_size pinned host). Allocated
    //    ONCE here so page_in_sync_ doesn't pay hipHostMalloc latency per
    //    call. For a 540 MB token-embed tensor that's ~10+ seconds saved
    //    per access.
    sync_staging_size_ = slot_size;
    sync_staging_      = nullptr;
#if defined(GGML_USE_HIP)
    if (hipHostMalloc(&sync_staging_, sync_staging_size_, hipHostMallocDefault) == hipSuccess) {
        sync_staging_pinned_ = true;
    } else {
        LLAMA_LOG_WARN("wp::WeightPager: hipHostMalloc(%zu) for shared sync staging failed; falling back to malloc\n",
                       sync_staging_size_);
        sync_staging_        = std::malloc(sync_staging_size_);
        sync_staging_pinned_ = false;
    }
#else
    sync_staging_ = std::malloc(sync_staging_size_);
#endif
    if (sync_staging_ == nullptr) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: shared sync staging allocation failed\n");
        prefetch_.shutdown();
        file_io_.reset();
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        restore_disable_graphs_env_();
        return false;
    }

    const size_t host_budget = env_size_bytes(kEnvWpHostBudgetBytes);
    if (host_budget > 0) {
        auto host_tier = std::make_unique<HostTier>();
        if (host_tier->init(host_budget, device_idx)) {
            host_tier_ = std::move(host_tier);
        } else {
            LLAMA_LOG_WARN("wp::WeightPager: WP_HOST_BUDGET_BYTES=%zu requested, but HostTier init failed; continuing disabled\n",
                           host_budget);
        }
    }

    initialized_ = true;
    if (pool_.size_class_slots_enabled()) {
        LLAMA_LOG_WARN("wp::WeightPager: %d pages, %d slots x %zu B budget (%.1f MiB), size_class_slots=1, prefetch_depth=%d, io_uring_depth=%d, sync_staging_pinned=%d, WP_ASYNC_ENSURE=%d\n",
                       catalog_.size(), cfg_.n_slots, slot_size,
                       (double) pool_.pool_size() / 1048576.0,
                       cfg_.prefetch_depth, cfg_.io_uring_depth, (int) sync_staging_pinned_,
                       (int) async_ensure_enabled_);
    } else {
        LLAMA_LOG_WARN("wp::WeightPager: %d pages, %d slots x %zu B (%.1f MiB), prefetch_depth=%d, io_uring_depth=%d, sync_staging_pinned=%d, WP_ASYNC_ENSURE=%d\n",
                       catalog_.size(), cfg_.n_slots, slot_size,
                       (double) cfg_.n_slots * (double) slot_size / 1048576.0,
                       cfg_.prefetch_depth, cfg_.io_uring_depth, (int) sync_staging_pinned_,
                       (int) async_ensure_enabled_);
    }
    return true;
}

void WeightPager::shutdown() {
    if (!initialized_) {
        // If init partially completed, there's no live state — but the env
        // snapshot may have been taken. Restore it defensively.
        restore_disable_graphs_env_();
        env_was_present_ = false;
        env_prior_value_.clear();
        hip_graphs_enabled_ = false;
        async_ensure_enabled_ = false;
        return;
    }

    log_stats_summary();
    weight_pager_eval_cb_reset(this);
    release_graph_pins_();
    release_sticky_l2_();
    for (int evt : page_async_event_) {
        if (evt >= 0) {
            transport_.synchronize(evt);
            transport_.release_event(evt);
        }
    }
    page_async_event_.clear();
    page_buft_.clear();

    // Tear down in reverse construction order.
    prefetch_.shutdown();
    file_io_.reset();
    transport_.shutdown();
    host_tier_.reset();
    if (sync_staging_ != nullptr) {
#if defined(GGML_USE_HIP)
        if (sync_staging_pinned_) {
            hipHostFree(sync_staging_);
        } else {
            std::free(sync_staging_);
        }
#else
        std::free(sync_staging_);
#endif
        sync_staging_       = nullptr;
        sync_staging_size_  = 0;
        sync_staging_pinned_ = false;
    }
    free_ensure_host_bufs_();
    // PoolAllocator dtor frees the ggml buffer.
    pool_.~PoolAllocator();
    new (&pool_) PoolAllocator{};

    page_to_slot_.clear();
    page_loaded_.clear();
    cross_layer_prefetch_candidate_.clear();
    draft_tid2eid_mark_.clear();
    oracle_pred_mark_.clear();
    oracle_pred_hit_.clear();
    draft_retain_pages_.clear();
    sticky_l2_score_.clear();
    sticky_l2_mark_.clear();
    sticky_l2_pinned_.clear();
    sticky_l2_pages_.clear();
    sticky_l2_hits_since_refresh_ = 0;
    sticky_l2_enabled_ = false;
    sticky_l2_stats_ = false;
    sticky_l2_max_pages_ = 32;
    hot_expert_history_.clear();
    tid2eid_tables_.clear();
    sample_sticky_pages_.clear(); // pins already gone with pool; just drop list
    draft_window_ = 0;
    draft_warm_streak_ = 0;
    draft_oracle_fires_ = 0;
    prefetch_started_at_.clear();
    slot_to_page_.clear();
    catalog_.clear();

    restore_disable_graphs_env_();
    env_was_present_ = false;
    env_prior_value_.clear();
    hip_graphs_enabled_ = false;
    async_ensure_enabled_ = false;

    initialized_ = false;
}

ggml_backend_buffer_t WeightPager::pool_buf(int page_idx) const {
    (void) page_idx;
    return pool_.vram_buf();
}

void WeightPager::restore_disable_graphs_env_() {
    if (!env_disable_graphs_forced_) {
        return;
    }
    env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
    env_disable_graphs_forced_ = false;
}

void WeightPager::release_graph_pins_() {
    if (graph_pin_slots_.empty()) {
        return;
    }
    for (const auto & kv : graph_pin_slots_) {
        for (int slot : kv.second) {
            pool_.unpin_slot(slot);
        }
    }
    graph_pin_slots_.clear();
}

int WeightPager::graph_pin_max_slots_() const {
    const int n_slots = pool_.n_slots() > 0 ? pool_.n_slots() : (cfg_.n_slots > 0 ? cfg_.n_slots : 0);
    const int floor = std::max(1, cfg_.prefetch_depth + 1);
    const int hard_max = std::max(0, n_slots - floor);
    const int requested = env_nonnegative_int(kEnvWpGraphPinMax, hard_max);
    return std::min(requested, hard_max);
}

int WeightPager::graph_pin_slot_count_except_(const void * graph_key) const {
    std::vector<int> slots;
    for (const auto & kv : graph_pin_slots_) {
        if (kv.first == graph_key) {
            continue;
        }
        for (int slot : kv.second) {
            bool seen = false;
            for (int s : slots) {
                if (s == slot) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                slots.push_back(slot);
            }
        }
    }
    return (int) slots.size();
}

void WeightPager::on_pool_evict_(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= (int) slot_to_page_.size()) return;
    if (pool_.is_speculative(slot_idx)) ++stats_.speculative_evicted_unused;
    int page = slot_to_page_[slot_idx];
    if (page >= 0 && page < (int) page_to_slot_.size()) {
        if (page < (int) page_async_event_.size() && page_async_event_[page] >= 0) {
            const int evt = page_async_event_[page];
            transport_.synchronize(evt);
            finish_async_transfer_event(page, evt);
            page_async_event_[page] = -1;
        }
        page_to_slot_[page] = -1;
        page_loaded_[page]  = false;
    }
    slot_to_page_[slot_idx] = -1;
    ++stats_.evictions;
}

void WeightPager::ensure_slot_map_(int slot_idx) {
    if (slot_idx < 0) return;
    if (slot_idx >= (int) slot_to_page_.size()) {
        slot_to_page_.resize((size_t) slot_idx + 1, -1);
    }
}

const WeightPager::Stats & WeightPager::stats() const {
    stats_.lru_walk_hot_skips    = pool_.lru_walk_hot_skips();
    stats_.lru_walk_pinned_skips = pool_.lru_walk_pinned_skips();
    stats_.sticky_l2_pins = 0;
    for (bool pinned : sticky_l2_pinned_) {
        if (pinned) {
            ++stats_.sticky_l2_pins;
        }
    }
#if defined(GGML_USE_HIP)
    ggml_cuda_get_routed_expert_ptrs_stats(
        &stats_.routing_ptrs_set,
        &stats_.routing_ptrs_consumed,
        &stats_.routing_ptrs_discarded_unconsumed);
#endif
    return stats_;
}

bool WeightPager::batch_safe() const {
    // Dense-only guard: WP_BATCH_EVAL_CB batching is only correct for dense models.
    // On MoE the scheduler batches a routing op as the *last* node of a range, so the
    // routed-expert TLS / pinned-slot lifetime isn't isolated -> near-null expert-pointer
    // GPU fault. Until the MoE-batching redesign lands (routing op must break the range
    // *before* it), keep MoE on the per-op sync path.
    if (wp_paged_batch_enabled()) {
        // Under WP_PAGED_BATCH, batching is governed by live pinnability and
        // routing-boundary breaks in wp-eval-cb.cpp, not a static eviction count.
        return pool_.size_class_slots_enabled();
    }
    return stats_.evictions == 0 && pool_.size_class_slots_enabled() && !catalog_.has_experts();
}

void WeightPager::mark_routing_boundaries(const struct ggml_cgraph * gf) {
    if (gf == nullptr || gf->n_nodes <= 0) { return; }
    const void * first = gf->nodes[0];
    const void * last  = gf->nodes[gf->n_nodes - 1];
    if (routing_sig_.n_nodes == gf->n_nodes && routing_sig_.first == first && routing_sig_.last == last) { return; }
    routing_break_tensors_.clear();
    for (int i = 0; i < gf->n_nodes; ++i) {
        struct ggml_tensor * node = gf->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) { continue; }
        routing_break_tensors_.insert(node);
        struct ggml_tensor * ids = node->src[2];
        while (ids != nullptr && ids->view_src != nullptr) { ids = ids->view_src; }
        if (ids != nullptr) { routing_break_tensors_.insert(ids); }
    }
    routing_sig_.n_nodes = gf->n_nodes;
    routing_sig_.first = first;
    routing_sig_.last  = last;
}

int WeightPager::loaded_pages() const {
    int n = 0;
    for (bool loaded : page_loaded_) {
        if (loaded) ++n;
    }
    return n;
}

void WeightPager::record_page_in_(size_t bytes, double seconds) {
    ++stats_.page_ins;
    stats_.io_bytes += (uint64_t) bytes;
    if (seconds > 0.0) {
        stats_.io_seconds += seconds;
    }
}

void WeightPager::log_stats_summary() {
    // WP_PROFILE_EVAL host-time breakdown (no-op unless the env flag is set).
    weight_pager_eval_cb_print_profile();

    const Stats & s = stats();
    const uint64_t prefetch_total = s.prefetch_hits + s.prefetch_misses;
    const double hit_rate = prefetch_total > 0
        ? 100.0 * (double) s.prefetch_hits / (double) prefetch_total
        : 0.0;
    const double gb_read = (double) s.io_bytes / 1000000000.0;
    const double gbps = s.io_seconds > 0.0 ? gb_read / s.io_seconds : 0.0;

    if (host_tier_ || s.host_tier_hits > 0) {
        LLAMA_LOG_WARN(
            "wp::WeightPager summary:\n"
            "  page_ins: %lu\n"
            "  evictions: %lu\n"
            "  prefetch_hits: %lu\n"
            "  prefetch_misses: %lu\n"
            "  prefetch_hit_rate: %.2f%%\n"
            "  io_gb_read: %.3f\n"
            "  io_effective_gb_s: %.3f\n"
            "  sync_fallbacks: %lu\n"
            "  lru_walk_hot_skips: %lu\n"
            "  lru_walk_pinned_skips: %lu\n"
            "  cross_layer_prefetch_submitted: %lu\n"
            "  cross_layer_hit_in_ensure: %lu\n"
        "  speculative_evicted_unused: %lu\n"
            "  speculative_evicted_unused: %lu\n"
            "  host_tier_hits: %lu\n"
            "  routing_ptrs_set: %lu\n"
            "  routing_ptrs_consumed: %lu\n"
            "  routing_ptrs_discarded_unconsumed: %lu\n"
            "  draft_prefetch_calls: %lu\n"
            "  draft_prefetch_pages_submitted: %lu\n"
            "  draft_prefetch_pages_resident: %lu\n"
            "  draft_retain_pins: %lu\n"
            "  draft_tid2eid_pages: %lu\n"
            "  draft_cold_pages: %lu\n"
            "  draft_tid2eid_cold: %lu\n"
            "  draft_tid2eid_hits_in_ensure: %lu\n"
            "  draft_oracle_skips: %lu\n"
            "  draft_window_opens: %lu\n"
            "  draft_window_closes: %lu\n"
            "  draft_prefetch_queue_blocked: %lu\n"
            "  draft_prefetch_harvested: %lu\n"
            "  draft_hot_records: %lu\n"
            "  oracle_sample_fires: %lu\n"
            "  oracle_draft_fires: %lu\n"
            "  oracle_pred_pages: %lu\n"
            "  oracle_actual_hash_pages: %lu\n"
            "  oracle_tp: %lu\n"
            "  oracle_fn: %lu\n"
            "  oracle_fp: %lu\n"
            "  oracle_pages_submitted: %lu\n"
            "  oracle_pages_free_slot: %lu\n"
            "  oracle_pages_evict_slot: %lu\n"
            "  oracle_hash_slots_freed: %lu\n"
            "  oracle_protect_pins: %lu\n"
            "  oracle_sticky_pins: %lu\n"
            "  sticky_l2_pins: %lu\n"
            "  sticky_l2_hits_in_ensure: %lu\n"
            "  sticky_l2_promotions: %lu\n"
            "  sticky_l2_demotions: %lu\n"
            "  sticky_spec_fires: %lu\n"
            "  sticky_spec_pages_submitted: %lu\n"
            "  sticky_spec_pages_resident: %lu\n"
            "  ensure_batch_calls: %lu\n"
            "  ensure_batch_pages: %lu\n"
            "  ensure_batch_max_n: %lu\n"
            "  ensure_batch_avg_n: %.2f\n"
            "  ensure_batch_gb_s: %.3f\n"
            "  ensure_batch_submit_ms: %.1f\n"
            "  ensure_batch_wait_ms: %.1f\n"
            "  ensure_batch_timeouts: %lu\n",
            (unsigned long) s.page_ins,
            (unsigned long) s.evictions,
            (unsigned long) s.prefetch_hits,
            (unsigned long) s.prefetch_misses,
            hit_rate,
            gb_read,
            gbps,
            (unsigned long) s.sync_fallbacks,
            (unsigned long) s.lru_walk_hot_skips,
            (unsigned long) s.lru_walk_pinned_skips,
            (unsigned long) s.cross_layer_prefetch_submitted,
            (unsigned long) s.cross_layer_hit_in_ensure,
        (unsigned long) s.speculative_evicted_unused,
            (unsigned long) s.speculative_evicted_unused,
            (unsigned long) s.host_tier_hits,
            (unsigned long) s.routing_ptrs_set,
            (unsigned long) s.routing_ptrs_consumed,
            (unsigned long) s.routing_ptrs_discarded_unconsumed,
            (unsigned long) s.draft_prefetch_calls,
            (unsigned long) s.draft_prefetch_pages_submitted,
            (unsigned long) s.draft_prefetch_pages_resident,
            (unsigned long) s.draft_retain_pins,
            (unsigned long) s.draft_tid2eid_pages,
            (unsigned long) s.draft_cold_pages,
            (unsigned long) s.draft_tid2eid_cold,
            (unsigned long) s.draft_tid2eid_hits_in_ensure,
            (unsigned long) s.draft_oracle_skips,
            (unsigned long) s.draft_window_opens,
            (unsigned long) s.draft_window_closes,
            (unsigned long) s.draft_prefetch_queue_blocked,
            (unsigned long) s.draft_prefetch_harvested,
            (unsigned long) s.draft_hot_records,
            (unsigned long) s.oracle_sample_fires,
            (unsigned long) s.oracle_draft_fires,
            (unsigned long) s.oracle_pred_pages,
            (unsigned long) s.oracle_actual_hash_pages,
            (unsigned long) s.oracle_tp,
            (unsigned long) s.oracle_fn,
            (unsigned long) s.oracle_fp,
            (unsigned long) s.oracle_pages_submitted,
            (unsigned long) s.oracle_pages_free_slot,
            (unsigned long) s.oracle_pages_evict_slot,
            (unsigned long) s.oracle_hash_slots_freed,
            (unsigned long) s.oracle_protect_pins,
            (unsigned long) s.oracle_sticky_pins,
            (unsigned long) s.sticky_l2_pins,
            (unsigned long) s.sticky_l2_hits_in_ensure,
            (unsigned long) s.sticky_l2_promotions,
            (unsigned long) s.sticky_l2_demotions,
            (unsigned long) s.sticky_spec_fires,
            (unsigned long) s.sticky_spec_pages_submitted,
            (unsigned long) s.sticky_spec_pages_resident,
            (unsigned long) s.ensure_batch_calls,
            (unsigned long) s.ensure_batch_pages,
            (unsigned long) s.ensure_batch_max_n,
            (s.ensure_batch_calls > 0
                 ? (double) s.ensure_batch_pages / (double) s.ensure_batch_calls
                 : 0.0),
            (s.ensure_batch_seconds > 0.0
                 ? ((double) s.ensure_batch_bytes / 1e9) / s.ensure_batch_seconds
                 : 0.0),
            s.ensure_batch_submit_seconds * 1e3,
            s.ensure_batch_wait_seconds * 1e3,
            (unsigned long) s.ensure_batch_timeouts);
        if (s.dense_prefetch_submitted > 0) {
            LLAMA_LOG_WARN("  dense_prefetch_submitted: %lu\n",
                           (unsigned long) s.dense_prefetch_submitted);
        }
        return;
    }

    LLAMA_LOG_WARN(
        "wp::WeightPager summary:\n"
        "  page_ins: %lu\n"
        "  evictions: %lu\n"
        "  prefetch_hits: %lu\n"
        "  prefetch_misses: %lu\n"
        "  prefetch_hit_rate: %.2f%%\n"
        "  io_gb_read: %.3f\n"
        "  io_effective_gb_s: %.3f\n"
        "  sync_fallbacks: %lu\n"
        "  lru_walk_hot_skips: %lu\n"
        "  lru_walk_pinned_skips: %lu\n"
        "  cross_layer_prefetch_submitted: %lu\n"
        "  cross_layer_hit_in_ensure: %lu\n"
        "  routing_ptrs_set: %lu\n"
        "  routing_ptrs_consumed: %lu\n"
        "  routing_ptrs_discarded_unconsumed: %lu\n"
        "  draft_prefetch_calls: %lu\n"
        "  draft_prefetch_pages_submitted: %lu\n"
        "  draft_prefetch_pages_resident: %lu\n"
        "  draft_retain_pins: %lu\n"
        "  draft_tid2eid_pages: %lu\n"
        "  draft_cold_pages: %lu\n"
        "  draft_tid2eid_cold: %lu\n"
        "  draft_tid2eid_hits_in_ensure: %lu\n"
        "  draft_oracle_skips: %lu\n"
        "  draft_window_opens: %lu\n"
        "  draft_window_closes: %lu\n"
        "  draft_prefetch_queue_blocked: %lu\n"
        "  draft_prefetch_harvested: %lu\n"
        "  draft_hot_records: %lu\n"
        "  oracle_sample_fires: %lu\n"
        "  oracle_draft_fires: %lu\n"
        "  oracle_pred_pages: %lu\n"
        "  oracle_actual_hash_pages: %lu\n"
        "  oracle_tp: %lu\n"
        "  oracle_fn: %lu\n"
        "  oracle_fp: %lu\n"
        "  oracle_pages_submitted: %lu\n"
        "  oracle_pages_free_slot: %lu\n"
        "  oracle_pages_evict_slot: %lu\n"
        "  oracle_hash_slots_freed: %lu\n"
        "  oracle_protect_pins: %lu\n"
        "  oracle_sticky_pins: %lu\n"
        "  sticky_l2_pins: %lu\n"
        "  sticky_l2_hits_in_ensure: %lu\n"
        "  sticky_l2_promotions: %lu\n"
        "  sticky_l2_demotions: %lu\n"
        "  sticky_spec_fires: %lu\n"
        "  sticky_spec_pages_submitted: %lu\n"
        "  sticky_spec_pages_resident: %lu\n"
        "  ensure_batch_calls: %lu\n"
        "  ensure_batch_pages: %lu\n"
        "  ensure_batch_max_n: %lu\n"
        "  ensure_batch_avg_n: %.2f\n"
        "  ensure_batch_gb_s: %.3f\n"
        "  ensure_batch_submit_ms: %.1f\n"
        "  ensure_batch_wait_ms: %.1f\n"
        "  ensure_batch_timeouts: %lu\n",
        (unsigned long) s.page_ins,
        (unsigned long) s.evictions,
        (unsigned long) s.prefetch_hits,
        (unsigned long) s.prefetch_misses,
        hit_rate,
        gb_read,
        gbps,
        (unsigned long) s.sync_fallbacks,
        (unsigned long) s.lru_walk_hot_skips,
        (unsigned long) s.lru_walk_pinned_skips,
        (unsigned long) s.cross_layer_prefetch_submitted,
        (unsigned long) s.cross_layer_hit_in_ensure,
        (unsigned long) s.routing_ptrs_set,
        (unsigned long) s.routing_ptrs_consumed,
        (unsigned long) s.routing_ptrs_discarded_unconsumed,
        (unsigned long) s.draft_prefetch_calls,
        (unsigned long) s.draft_prefetch_pages_submitted,
        (unsigned long) s.draft_prefetch_pages_resident,
        (unsigned long) s.draft_retain_pins,
        (unsigned long) s.draft_tid2eid_pages,
        (unsigned long) s.draft_cold_pages,
        (unsigned long) s.draft_tid2eid_cold,
        (unsigned long) s.draft_tid2eid_hits_in_ensure,
        (unsigned long) s.draft_oracle_skips,
        (unsigned long) s.draft_window_opens,
        (unsigned long) s.draft_window_closes,
        (unsigned long) s.draft_prefetch_queue_blocked,
        (unsigned long) s.draft_prefetch_harvested,
        (unsigned long) s.draft_hot_records,
        (unsigned long) s.oracle_sample_fires,
        (unsigned long) s.oracle_draft_fires,
        (unsigned long) s.oracle_pred_pages,
        (unsigned long) s.oracle_actual_hash_pages,
        (unsigned long) s.oracle_tp,
        (unsigned long) s.oracle_fn,
        (unsigned long) s.oracle_fp,
        (unsigned long) s.oracle_pages_submitted,
        (unsigned long) s.oracle_pages_free_slot,
        (unsigned long) s.oracle_pages_evict_slot,
        (unsigned long) s.oracle_hash_slots_freed,
        (unsigned long) s.oracle_protect_pins,
        (unsigned long) s.oracle_sticky_pins,
        (unsigned long) s.sticky_l2_pins,
        (unsigned long) s.sticky_l2_hits_in_ensure,
        (unsigned long) s.sticky_l2_promotions,
        (unsigned long) s.sticky_l2_demotions,
        (unsigned long) s.sticky_spec_fires,
        (unsigned long) s.sticky_spec_pages_submitted,
        (unsigned long) s.sticky_spec_pages_resident,
        (unsigned long) s.ensure_batch_calls,
        (unsigned long) s.ensure_batch_pages,
        (unsigned long) s.ensure_batch_max_n,
        (s.ensure_batch_calls > 0
             ? (double) s.ensure_batch_pages / (double) s.ensure_batch_calls
             : 0.0),
        (s.ensure_batch_seconds > 0.0
             ? ((double) s.ensure_batch_bytes / 1e9) / s.ensure_batch_seconds
             : 0.0),
        s.ensure_batch_submit_seconds * 1e3,
        s.ensure_batch_wait_seconds * 1e3,
        (unsigned long) s.ensure_batch_timeouts);
    if (s.dense_prefetch_submitted > 0) {
        LLAMA_LOG_WARN("  dense_prefetch_submitted: %lu\n",
                       (unsigned long) s.dense_prefetch_submitted);
    }
}

int WeightPager::slot_for_page(int page_idx) const {
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return -1;
    if (!page_loaded_[page_idx])                                 return -1;
    return page_to_slot_[page_idx];
}

void WeightPager::update_graph_pins(const void * graph_key, const std::vector<int> & page_indices) {
    if (!initialized_ || !hip_graphs_enabled_ || graph_key == nullptr) {
        return;
    }

    std::vector<int> slots;
    slots.reserve(page_indices.size());
    for (int page_idx : page_indices) {
        const int slot = slot_for_page(page_idx);
        if (slot < 0) {
            continue;
        }
        bool seen = false;
        for (int s : slots) {
            if (s == slot) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            slots.push_back(slot);
        }
    }

    const int max_slots = graph_pin_max_slots_();
    const int existing_slots = graph_pin_slot_count_except_(graph_key);
    const int room = std::max(0, max_slots - existing_slots);
    if ((int) slots.size() > room) {
        for (size_t i = (size_t) room; i < slots.size(); ++i) {
            log_graph_pin_degrade(/*page_idx=*/-1, slots[i], max_slots);
        }
        slots.resize((size_t) room);
    }

    auto it = graph_pin_slots_.find(graph_key);
    if (it != graph_pin_slots_.end() && it->second == slots) {
        return;
    }

    if (it != graph_pin_slots_.end()) {
        for (int old_slot : it->second) {
            pool_.unpin_slot(old_slot);
        }
        graph_pin_slots_.erase(it);
    }

    if (slots.empty()) {
        return;
    }

    for (int slot : slots) {
        pool_.pin_slot(slot);
    }
    graph_pin_slots_.emplace(graph_key, std::move(slots));
}

bool WeightPager::try_add_graph_pin_page(const void * graph_key, int page_idx, std::vector<int> & page_indices) const {
    if (!initialized_ || !hip_graphs_enabled_ || graph_key == nullptr) {
        return false;
    }

    const int slot = slot_for_page(page_idx);
    if (slot < 0) {
        return false;
    }

    std::vector<int> selected_slots;
    selected_slots.reserve(page_indices.size());
    for (int p : page_indices) {
        const int s = slot_for_page(p);
        if (s < 0) {
            continue;
        }
        bool seen = false;
        for (int existing : selected_slots) {
            if (existing == s) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            selected_slots.push_back(s);
        }
        if (p == page_idx || s == slot) {
            return true;
        }
    }

    const int max_slots = graph_pin_max_slots_();
    const int existing_slots = graph_pin_slot_count_except_(graph_key);
    if (existing_slots + (int) selected_slots.size() + 1 > max_slots) {
        // Full lifetime-correct graph pinning needs per-node graph arg patching
        // and release on graph recapture/destroy; that remains hardware work.
        log_graph_pin_degrade(page_idx, slot, max_slots);
        return false;
    }

    page_indices.push_back(page_idx);
    return true;
}

void WeightPager::note_draft_tid2eid_ensure_(int page_idx) {
    if (page_idx >= 0 &&
        page_idx < (int) draft_tid2eid_mark_.size() &&
        draft_tid2eid_mark_[page_idx]) {
        ++stats_.draft_tid2eid_hits_in_ensure;
        if (sticky_l2_enabled_ && page_idx < (int) sticky_l2_score_.size()) {
            if (sticky_l2_score_[page_idx] != 0xffffffffu) {
                ++sticky_l2_score_[page_idx];
            }
            ++sticky_l2_hits_since_refresh_;
            sticky_l2_refresh_if_due_("ensure");
        }
    }
    if (sticky_l2_enabled_ &&
        page_idx >= 0 &&
        page_idx < (int) sticky_l2_mark_.size() &&
        sticky_l2_mark_[page_idx]) {
        ++stats_.sticky_l2_hits_in_ensure;
    }
}

bool WeightPager::draft_retain_contains_(int page_idx) const {
    for (int p : draft_retain_pages_) {
        if (p == page_idx) {
            return true;
        }
    }
    return false;
}

void WeightPager::sticky_l2_refresh_if_due_(const char * reason) {
    if (!sticky_l2_enabled_) {
        return;
    }
    // ~once per ~2 tokens of MoE layers (n_layer~40) with per-route credits.
    const int refresh_every = std::max(64, sticky_l2_max_pages_ * 2);
    if (sticky_l2_hits_since_refresh_ >= refresh_every) {
        sticky_l2_refresh_(reason);
    }
}

void WeightPager::sticky_l2_refresh_(const char * reason) {
    if (!initialized_ || !sticky_l2_enabled_ || sticky_l2_score_.empty()) {
        return;
    }

    // Pin set: currently resident high-score pages only (eviction shield).
    // Cold high-score pages are handled by prefetch_sticky_hot_experts().
    std::vector<std::pair<uint32_t, int>> candidates;
    candidates.reserve(sticky_l2_score_.size());
    for (int p = 0; p < (int) sticky_l2_score_.size(); ++p) {
        const uint32_t score = sticky_l2_score_[p];
        if (score == 0) {
            continue;
        }
        if (p >= catalog_.size() || catalog_.at(p).is_pinned) {
            continue;
        }
        if (p >= (int) page_loaded_.size() || !page_loaded_[p]) {
            continue;
        }
        if (p >= (int) page_to_slot_.size() || page_to_slot_[p] < 0) {
            continue;
        }
        candidates.push_back({ score, p });
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<uint32_t, int> & a, const std::pair<uint32_t, int> & b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });
    if ((int) candidates.size() > sticky_l2_max_pages_) {
        candidates.resize((size_t) sticky_l2_max_pages_);
    }

    std::vector<bool> want(sticky_l2_mark_.size(), false);
    for (const auto & c : candidates) {
        want[(size_t) c.second] = true;
    }

    int promotions = 0;
    int demotions = 0;
    std::vector<int> next_pages;
    next_pages.reserve(candidates.size());

    for (int p : sticky_l2_pages_) {
        if (p < 0 || p >= (int) sticky_l2_mark_.size()) {
            continue;
        }
        if (want[(size_t) p]) {
            next_pages.push_back(p);
            continue;
        }
        if (sticky_l2_pinned_[p]) {
            unpin_page(p);
            sticky_l2_pinned_[p] = false;
        }
        sticky_l2_mark_[p] = false;
        ++demotions;
    }

    for (const auto & c : candidates) {
        const int p = c.second;
        if (!sticky_l2_mark_[p]) {
            sticky_l2_mark_[p] = true;
            ++promotions;
            next_pages.push_back(p);
        }
        if (!sticky_l2_pinned_[p] && !draft_retain_contains_(p)) {
            pin_page(p);
            sticky_l2_pinned_[p] = true;
        }
    }

    sticky_l2_pages_.swap(next_pages);
    sticky_l2_hits_since_refresh_ = 0;
    stats_.sticky_l2_promotions += (uint64_t) promotions;
    stats_.sticky_l2_demotions += (uint64_t) demotions;

    if (sticky_l2_stats_ && (promotions > 0 || demotions > 0)) {
        int pins = 0;
        for (bool pinned : sticky_l2_pinned_) {
            if (pinned) {
                ++pins;
            }
        }
        LLAMA_LOG_WARN("sticky-l2: reason=%s candidates=%d pins=%d promotions=%d demotions=%d\n",
                       reason != nullptr ? reason : "refresh",
                       (int) candidates.size(), pins, promotions, demotions);
    }
}

void WeightPager::release_sticky_l2_() {
    if (sticky_l2_pinned_.empty()) {
        return;
    }
    for (int p = 0; p < (int) sticky_l2_pinned_.size(); ++p) {
        if (sticky_l2_pinned_[p]) {
            unpin_page(p);
            sticky_l2_pinned_[p] = false;
        }
    }
    std::fill(sticky_l2_mark_.begin(), sticky_l2_mark_.end(), false);
    sticky_l2_pages_.clear();
    sticky_l2_hits_since_refresh_ = 0;
}

void * WeightPager::ensure(int page_idx) {
    if (!initialized_)                                         return nullptr;
    if (page_idx < 0 || page_idx >= catalog_.size())           return nullptr;
    if (page_idx < (int) page_async_event_.size() && page_async_event_[page_idx] >= 0) {
        const int evt = page_async_event_[page_idx];
        transport_.synchronize(evt);
        finish_async_transfer_event(page_idx, evt);
        page_async_event_[page_idx] = -1;
    }

    // MAD-236 — pinned (always-resident) pages live in caller-owned VRAM,
    // not the pool. No slot lookup, no LRU update, no IO. Just return the
    // registered device pointer. Cheap and short-circuits the whole pipeline.
    const PageMeta & m_check = catalog_.at(page_idx);
    if (m_check.is_pinned) {
        return m_check.resident_ptr;
    }

    note_draft_tid2eid_ensure_(page_idx);

    const bool cross_layer_candidate =
        page_idx >= 0 &&
        page_idx < (int) cross_layer_prefetch_candidate_.size() &&
        cross_layer_prefetch_candidate_[page_idx];

    // Already committed? Bump LRU and return.
    if (page_loaded_[page_idx]) {
        if (cross_layer_candidate) {
            ++stats_.cross_layer_hit_in_ensure;
            cross_layer_prefetch_candidate_[page_idx] = false;
        }
        const int slot = page_to_slot_[page_idx];
        pool_.mark_used(slot);
        return slot_ptr_(slot);
    }

    // Slot reserved by an in-flight prefetch? Wait for it.
    bool counted_prefetch_miss = false;
    int slot = page_to_slot_[page_idx];
    if (slot >= 0) {
        if (async_ensure_enabled_) {
            prefetch_.tick();
        }
        const bool loaded_before_wait = prefetch_.is_loaded(page_idx);
        if (loaded_before_wait) {
            ++stats_.prefetch_hits;
            if (cross_layer_candidate) {
                ++stats_.cross_layer_hit_in_ensure;
                cross_layer_prefetch_candidate_[page_idx] = false;
            }
        } else {
            ++stats_.prefetch_misses;
            counted_prefetch_miss = true;
        }
        if (async_ensure_enabled_ && !loaded_before_wait) {
            const int evt = prefetch_.take_stage2_event(page_idx);
            if (evt >= 0) {
                ++stats_.prefetch_hits;
                if (counted_prefetch_miss) {
                    --stats_.prefetch_misses;
                    counted_prefetch_miss = false;
                }
                if (cross_layer_candidate) {
                    ++stats_.cross_layer_hit_in_ensure;
                    cross_layer_prefetch_candidate_[page_idx] = false;
                }

                page_loaded_[page_idx] = true;
                pool_.mark_used(slot);
                // The async H2D still reads from PrefetchScheduler's pinned
                // staging buffer until evt signals. Reaping here would recycle
                // that buffer for another prefetch and corrupt the slot copy.
                double seconds = 0.0;
                if (page_idx < (int) prefetch_started_at_.size() &&
                    prefetch_started_at_[page_idx] != std::chrono::steady_clock::time_point{}) {
                    seconds = seconds_since(prefetch_started_at_[page_idx]);
                    prefetch_started_at_[page_idx] = std::chrono::steady_clock::time_point{};
                }
                record_page_in_(m_check.size, seconds);
                page_async_event_[page_idx] = evt;
                return slot_ptr_(slot);
            }
        }
        if (prefetch_.wait_for(page_idx, /*timeout_ms=*/-1)) {
            // Stage 2 done; commit and reap.
            page_loaded_[page_idx] = true;
            pool_.mark_used(slot);
            prefetch_.reap(page_idx);
            double seconds = 0.0;
            if (page_idx < (int) prefetch_started_at_.size() &&
                prefetch_started_at_[page_idx] != std::chrono::steady_clock::time_point{}) {
                seconds = seconds_since(prefetch_started_at_[page_idx]);
                prefetch_started_at_[page_idx] = std::chrono::steady_clock::time_point{};
            }
            record_page_in_(m_check.size, seconds);
            if (cross_layer_candidate && !loaded_before_wait) {
                cross_layer_prefetch_candidate_[page_idx] = false;
            }
            return slot_ptr_(slot);
        }
        // Prefetch failed; tear down the reservation so sync fallback can
        // start fresh.
        prefetch_.reap(page_idx);
        page_to_slot_[page_idx] = -1;
        slot_to_page_[slot]     = -1;
        pool_.release_slot(slot);
        // fall through to sync fallback
    }

    // Synchronous fallback: read directly into a slot.
    if (page_idx < (int) cross_layer_prefetch_candidate_.size()) {
        cross_layer_prefetch_candidate_[page_idx] = false;
    }
    if (!counted_prefetch_miss) {
        ++stats_.prefetch_misses;
    }
    ++stats_.sync_fallbacks;
    slot = page_in_sync_(page_idx);
    if (slot < 0) return nullptr;
    return slot_ptr_(slot);
}

void WeightPager::ensure_batch(const std::vector<int> & page_indices,
                               std::vector<void *>     & out_ptrs,
                               std::vector<int>        & out_pinned) {
    out_ptrs.assign(page_indices.size(), nullptr);
    out_pinned.clear();
    if (!initialized_) return;

    // Pass 1 — resolve hits/pinned inline; for every cold miss reserve AND PIN
    // a slot up front. Pinning before any sibling read means alloc_slot for a
    // later miss can never evict an earlier miss's in-flight slot (alloc_slot
    // skips pinned), which is what collapsed effective QD to ~1 under decode
    // eviction. Each pinned page is reported in out_pinned for the caller to
    // release next callback.
    struct Miss { int page; int slot; std::size_t out_i; };
    std::vector<Miss> misses;
    misses.reserve(page_indices.size());
    for (std::size_t i = 0; i < page_indices.size(); ++i) {
        const int p = page_indices[i];
        if (p < 0 || p >= catalog_.size()) continue;
        const PageMeta & m = catalog_.at(p);
        if (m.is_pinned) {                       // MAD-236: always-resident, no slot
            out_ptrs[i] = m.resident_ptr;
            continue;
        }
        if (page_loaded_[p]) {                   // hit — bump LRU, pin, harvest
            note_draft_tid2eid_ensure_(p);
            const int s = page_to_slot_[p];
            pool_.mark_used(s);
            pool_.pin_slot(s);
            out_pinned.push_back(p);
            out_ptrs[i] = slot_ptr_(s);
            continue;
        }
        if (page_to_slot_[p] >= 0) {
            // Reserved by an in-flight (e.g. cross-layer) prefetch. Harvest via
            // the tested ensure() path (waits on the prefetch, or syncs), then
            // pin the result. Not part of the concurrent batch.
            // ensure() notes draft tid2eid hits.
            void * ptr = ensure(p);
            if (ptr != nullptr) {
                pool_.pin_slot(page_to_slot_[p]);
                out_pinned.push_back(p);
                out_ptrs[i] = ptr;
            }
            continue;
        }
        note_draft_tid2eid_ensure_(p);
        const int s = pool_.alloc_slot(m.size);
        if (s < 0) { ++stats_.sync_fallbacks; continue; }  // pool exhausted (rare)
        ensure_slot_map_(s);
        page_to_slot_[p]  = s;
        slot_to_page_[s]  = p;
        pool_.pin_slot(s);                       // PIN before the next alloc
        out_pinned.push_back(p);
        misses.push_back({ p, s, i });
    }
    if (misses.empty()) return;

    // Optional: multi-QD O_DIRECT host bounce → H2D.
    // WP_ENSURE_BATCH_HOST=1. Cold random buffered ~1.1 GB/s; O_DIRECT ~6.2.
    // GGUF offs are not 512-aligned → align-down bounce then H2D payload slice.
    static const int s_batch_host = []() {
        const char * e = std::getenv("WP_ENSURE_BATCH_HOST");
        return (e != nullptr && e[0] == '1') ? 1 : 0;
    }();
    if (s_batch_host && ensure_host_bufs_ready_(misses.size(), catalog_.max_page_size())) {
        const auto io_t0 = std::chrono::steady_clock::now();
        struct HostJob {
            int      file_idx;
            int      fd;       // O_DIRECT fd
            uint64_t off;      // logical (possibly unaligned) file offset
            uint64_t base;     // 512-aligned O_DIRECT offset
            size_t   size;     // payload bytes
            void *   dst;      // 4k-aligned host bounce (>= size+1k)
            size_t   buf_cap;
            size_t   prefix;   // bytes of bounce pad before payload
            size_t   nbytes;   // 512-padded O_DIRECT byte count
            bool     queued;
            bool     seen;
            bool     ok;
        };
        std::vector<HostJob> jobs;
        jobs.reserve(misses.size());
        int n_od = 0;
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const PageMeta & m = catalog_.at(misses[k].page);
            const int od = ensure_odirect_fd_((int) m.file_idx);
            if (od >= 0) {
                ++n_od;
            }
            jobs.push_back(HostJob{
                (int) m.file_idx, od, m.file_offset, 0, m.size, ensure_host_bufs_[k],
                ensure_host_buf_bytes_, 0, 0, false, false, false
            });
        }
        int n_queued = 0;
        const auto tp_jobs = std::chrono::steady_clock::now();
        for (std::size_t k = 0; k < jobs.size(); ++k) {
            HostJob & j = jobs[k];
            if (j.fd < 0 || j.dst == nullptr || j.size == 0) {
                continue;
            }

            j.base   = j.off & ~511ULL;
            j.prefix = (size_t) (j.off - j.base);
            j.nbytes = (j.prefix + j.size + 511) & ~(size_t) 511;
            if (j.nbytes > j.buf_cap || j.nbytes > UINT_MAX) {
                continue;
            }
            j.queued = true;
            ++n_queued;
        }

        const auto tp_prep = std::chrono::steady_clock::now();
        std::vector<EnsureODirectReadJob> read_jobs(jobs.size());
        const int n_workers = ensure_odirect_worker_count_((size_t) n_queued);
        bool submit_failed = false;
        int n_submitted = 0;
        if (n_queued > 0 && ensure_odirect_workers_ready_(n_workers)) {
            {
                std::lock_guard<std::mutex> lock(ensure_odirect_mu_);
                for (std::size_t k = 0; k < jobs.size(); ++k) {
                    HostJob & j = jobs[k];
                    if (!j.queued) {
                        continue;
                    }
                    EnsureODirectReadJob & r = read_jobs[k];
                    r.fd   = j.fd;
                    r.off  = j.base;
                    r.size = j.nbytes;
                    r.dst  = j.dst;
                    ensure_odirect_queue_.push_back(&r);
                    ++n_submitted;
                }
            }
            ensure_odirect_cv_.notify_all();
        } else if (n_queued > 0) {
            submit_failed = true;
        }

        const auto tp_submit = std::chrono::steady_clock::now();
        int n_seen = 0;
        if (n_submitted > 0) {
            std::unique_lock<std::mutex> lock(ensure_odirect_mu_);
            ensure_odirect_done_cv_.wait(lock, [&read_jobs, n_submitted]() {
                int n_done = 0;
                for (const EnsureODirectReadJob & r : read_jobs) {
                    if (r.done) {
                        ++n_done;
                    }
                }
                return n_done >= n_submitted;
            });
            for (std::size_t k = 0; k < jobs.size(); ++k) {
                HostJob & j = jobs[k];
                EnsureODirectReadJob & r = read_jobs[k];
                if (!j.queued || !r.done) {
                    continue;
                }
                j.seen = true;
                j.ok = r.ok;
                ++n_seen;
                if (!r.ok) {
                    static int s_read_warn = 0;
                    if (s_read_warn < 3) {
                        LLAMA_LOG_WARN("wp::ensure_batch: HOST O_DIRECT pread failed fd=%d off=%llu size=%zu err=%s\n",
                                       r.fd, (unsigned long long) r.off, r.size,
                                       r.err < 0 ? strerror(-r.err) : "short read");
                        ++s_read_warn;
                    }
                }
            }
        }
        {
            const auto tp_reap = std::chrono::steady_clock::now();
            auto msd = [](auto a2, auto b2){ return std::chrono::duration<double,std::milli>(b2-a2).count(); };
            static double s_jobs=0,s_prep=0,s_sub=0,s_reap=0; static long s_n=0; static long s_pg=0;
            s_jobs += msd(io_t0, tp_jobs);
            s_prep += msd(tp_jobs, tp_prep);
            s_sub  += msd(tp_prep, tp_submit);
            s_reap += msd(tp_submit, tp_reap);
            s_pg   += n_seen;
            if (++s_n % 1000 == 0) {
                LLAMA_LOG_WARN("wp ODIRECT phase cum ms @%ld calls (%ld pages): jobs=%.0f prep=%.0f submit=%.0f reap=%.0f\n",
                               s_n, s_pg, s_jobs, s_prep, s_sub, s_reap);
            }
        }
        (void) submit_failed;
        const double read_seconds = seconds_since(io_t0);
        size_t batch_bytes = 0;
        int    batch_ok_n  = 0;
#if defined(GGML_USE_HIP)
        // Queue all H2Ds then one device sync (overlap PCIe copies).
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const Miss & mm = misses[k];
            const PageMeta & m = catalog_.at(mm.page);
            void * vram = slot_ptr_(mm.slot);
            if (!jobs[k].ok || vram == nullptr) {
                continue;
            }
            const void * src = (const char *) ensure_host_bufs_[k] + jobs[k].prefix;
            hipError_t err = hipMemcpyAsync(vram, src, m.size,
                                            hipMemcpyHostToDevice, nullptr);
            if (err != hipSuccess) {
                jobs[k].ok = false;
            }
        }
        hipDeviceSynchronize();
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const Miss & mm = misses[k];
            const PageMeta & m = catalog_.at(mm.page);
            void * vram = slot_ptr_(mm.slot);
            if (jobs[k].ok && vram != nullptr &&
                zero_device_padding(vram, m.size, pool_.slot_size(mm.slot))) {
                page_to_slot_[mm.page] = mm.slot;
                page_loaded_[mm.page]  = true;
                slot_to_page_[mm.slot] = mm.page;
                pool_.mark_used(mm.slot);
                batch_bytes += m.size;
                ++batch_ok_n;
                out_ptrs[mm.out_i] = vram;
            } else {
                const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
                out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
            }
        }
        static int s_host_path_log = 0;
        if (s_host_path_log < 1) {
            LLAMA_LOG_WARN("wp::ensure_batch: HOST path O_DIRECT=%d/%d misses first batch\n",
                           n_od, (int) misses.size());
            ++s_host_path_log;
        }
#else
        (void) n_od;
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const int s = page_in_sync_(misses[k].page, /*reuse_slot=*/misses[k].slot);
            out_ptrs[misses[k].out_i] = (s < 0) ? nullptr : slot_ptr_(s);
        }
#endif
        const double batch_seconds = seconds_since(io_t0);
        if (batch_ok_n > 0) {
            stats_.page_ins  += (uint64_t) batch_ok_n;
            stats_.io_bytes  += (uint64_t) batch_bytes;
            stats_.io_seconds += batch_seconds;
            ++stats_.ensure_batch_calls;
            stats_.ensure_batch_pages   += (uint64_t) batch_ok_n;
            stats_.ensure_batch_bytes   += (uint64_t) batch_bytes;
            stats_.ensure_batch_seconds += batch_seconds;
            stats_.ensure_batch_submit_seconds += read_seconds; // host O_DIRECT phase
            stats_.ensure_batch_wait_seconds   += (batch_seconds - read_seconds); // H2D
            stats_.ensure_batch_n_sub_sum      += (uint64_t) batch_ok_n;
            if ((uint64_t) batch_ok_n > stats_.ensure_batch_max_n) {
                stats_.ensure_batch_max_n = (uint64_t) batch_ok_n;
            }
        }
        return;
    }

    // Pass 2 — issue all cold-miss reads. On the P2P/direct-to-device path the
    // reads land straight in the VRAM slots, so one io_uring batch keeps N reads
    // in flight at once (true QD=N). Off P2P the shared staging buffer can't
    // hold N pages, so fall back to serial sync into each pinned slot (still
    // correct; the throughput case that matters for decode is P2P).
    if (file_io_->direct_to_device()) {
        std::vector<FileIOBatchRequest> reqs;
        std::vector<uint64_t>           req_ids;
        reqs.reserve(misses.size());
        req_ids.reserve(misses.size());
        // Wall-clock the whole multi-QD burst (submit → last wait), then
        // attribute that one interval once. Recording per-page after waits
        // (old) under-counted concurrent P2P BW (~0.4 vs real multi-GB/s).
        const auto io_t0 = std::chrono::steady_clock::now();
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const PageMeta & m = catalog_.at(misses[k].page);
            const uint64_t rid = next_io_req_id_++;   // high-bit-tagged; disjoint from prefetch
            req_ids.push_back(rid);
            reqs.push_back({ rid, (int) m.file_idx, m.file_offset,
                             m.size, slot_ptr_(misses[k].slot) });
        }
        const int n_sub = file_io_->submit_batch(reqs);
        file_io_->flush();
        const double submit_seconds = seconds_since(io_t0);
        const auto wait_t0 = std::chrono::steady_clock::now();
        std::vector<bool> ok(misses.size(), false);
        // Wait per req via demuxing wait_for_req (foreign CQEs buffered in
        // ready_). A multi-id wait_for_reqs was tried; it could busy-spin
        // at load (97% CPU, never reach "model loaded") when ensure_batch
        // fires during graph init. Ordered waits keep multi-QD overlap —
        // I/O is already in flight from submit_batch; we only reap.
        uint64_t n_timeout = 0;
        for (int k = 0; k < n_sub; ++k) {
            IoResult r = file_io_->wait_for_req(req_ids[(size_t) k], /*timeout_ms=*/-1);
            ok[(size_t) k] = (r.status == IoStatus::Ok &&
                              r.bytes_read == (int) catalog_.at(misses[(size_t) k].page).size);
            if (!ok[(size_t) k]) {
                ++n_timeout;
            }
        }
        const double wait_seconds = seconds_since(wait_t0);
        const double batch_seconds = seconds_since(io_t0);
        size_t batch_bytes = 0;
        int    batch_ok_n  = 0;
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const Miss & mm = misses[k];
            const PageMeta & m = catalog_.at(mm.page);
            if (ok[k] && zero_device_padding(slot_ptr_(mm.slot), m.size, pool_.slot_size(mm.slot))) {
                page_to_slot_[mm.page] = mm.slot;
                page_loaded_[mm.page]  = true;
                slot_to_page_[mm.slot] = mm.page;
                pool_.mark_used(mm.slot);
                batch_bytes += m.size;
                ++batch_ok_n;
                out_ptrs[mm.out_i] = slot_ptr_(mm.slot);
            } else {
                // read (or padding) failed — sync-fallback into the SAME pinned
                // slot so the up-front pin/out_pinned bookkeeping stays valid.
                const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
                out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
            }
        }
        if (batch_ok_n > 0 || n_sub > 0) {
            // One wall interval for the concurrent burst; page_ins += N.
            stats_.page_ins  += (uint64_t) batch_ok_n;
            stats_.io_bytes  += (uint64_t) batch_bytes;
            stats_.io_seconds += batch_seconds;
            ++stats_.ensure_batch_calls;
            stats_.ensure_batch_pages   += (uint64_t) batch_ok_n;
            stats_.ensure_batch_bytes   += (uint64_t) batch_bytes;
            stats_.ensure_batch_seconds += batch_seconds;
            stats_.ensure_batch_submit_seconds += submit_seconds;
            stats_.ensure_batch_wait_seconds   += wait_seconds;
            stats_.ensure_batch_timeouts       += n_timeout;
            stats_.ensure_batch_n_sub_sum      += (uint64_t) n_sub;
            if ((uint64_t) batch_ok_n > stats_.ensure_batch_max_n) {
                stats_.ensure_batch_max_n = (uint64_t) batch_ok_n;
            }
        }
    } else {
        for (const Miss & mm : misses) {
            const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
            out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
        }
    }
}

int WeightPager::take_async_transfer_event(int page_idx) {
    if (!initialized_) return -1;
    if (page_idx < 0 || page_idx >= (int) page_async_event_.size()) return -1;
    const int evt = page_async_event_[page_idx];
    page_async_event_[page_idx] = -1;
    return evt;
}

bool WeightPager::enqueue_async_transfer_wait(int event_handle, void * stream) {
    return transport_.wait_event_on_stream(event_handle, stream);
}

bool WeightPager::synchronize_async_transfer_event(int event_handle) {
    return transport_.synchronize(event_handle);
}

void WeightPager::finish_async_transfer_event(int page_idx, int event_handle) {
    if (!initialized_) return;
    if (page_idx >= 0) {
        prefetch_.reap(page_idx);
    }
    transport_.release_event(event_handle);
}

void WeightPager::release_async_transfer_event(int event_handle) {
    transport_.release_event(event_handle);
}

bool WeightPager::prefetch_page(int page_idx, bool count_dense_prefetch,
                                bool allow_evict) {
    if (!initialized_)                                          return false;
    if (page_idx < 0 || page_idx >= catalog_.size())            return false;
    if (catalog_.at(page_idx).is_pinned)                        return false;  // MAD-236: already resident, no slot needed
    if (page_to_slot_[page_idx] >= 0)                            return false;  // loaded or in flight

    // Capacity gate: never alloc_slot (which may LRU-evict) if the prefetch
    // scheduler cannot accept another submit. Evicting for a rejected submit
    // destroys useful residents for no I/O (Codex draft-prefetch analysis).
    if (prefetch_.free_queue_slots() <= 0) {
        return false;
    }

    const PageMeta & m = catalog_.at(page_idx);

    // Allocate a slot. Sample oracle uses no-evict so we never thrash MoE
    // working-set pages to make room for hash-layer speculation.
    const int slot = allow_evict ? pool_.alloc_slot(m.size)
                                 : pool_.alloc_slot_no_evict(m.size);
    if (slot < 0) return false;
    ensure_slot_map_(slot);
    void * dst = slot_ptr_(slot);

    // Track ownership BEFORE submitting so eviction-callbacks resolve right.
    // page_loaded_ stays false until ensure() commits after stage 2.
    page_to_slot_[page_idx]     = slot;
    page_loaded_[page_idx]      = false;
    slot_to_page_[slot]         = page_idx;

    if (!prefetch_.submit(page_idx, (int) m.file_idx, m.file_offset,
                          m.size, dst, pool_.slot_size(slot))) {
        // Rejected — likely queue full. Roll back our reservation.
        page_to_slot_[page_idx] = -1;
        slot_to_page_[slot]     = -1;
        pool_.release_slot(slot);
        return false;
    } else {
        if (page_idx < (int) prefetch_started_at_.size()) {
            prefetch_started_at_[page_idx] = std::chrono::steady_clock::now();
        }
        if (count_dense_prefetch) {
            ++stats_.dense_prefetch_submitted;
        }
        if (page_idx < (int) cross_layer_prefetch_candidate_.size() &&
            cross_layer_prefetch_candidate_[page_idx]) {
            ++stats_.cross_layer_prefetch_submitted;
        }
        return true;
    }
}

void WeightPager::tick() {
    if (!initialized_) return;
    prefetch_.tick();
}

bool WeightPager::prefetch_pages_batch(const std::vector<int> & page_indices,
                                       bool count_dense_prefetch,
                                       bool allow_evict,
                                       bool speculative) {
    if (!initialized_) return false;
    if (page_indices.empty()) return true;

    // Filter to pages that actually need prefetching (skip already-resident
    // and already-in-flight). Skipping is not a failure — common case is the
    // MoE re-uses some experts from the previous layer.
    std::vector<int> needed;
    needed.reserve(page_indices.size());
    for (int p : page_indices) {
        if (p < 0 || p >= catalog_.size()) continue;
        if (catalog_.at(p).is_pinned)   continue;  // MAD-236: pinned needs no prefetch
        if (page_to_slot_[p] >= 0)      continue;  // resident or in flight
        // Dedupe within input.
        bool dup = false;
        for (int q : needed) {
            if (q == p) { dup = true; break; }
        }
        if (!dup) needed.push_back(p);
    }
    if (needed.empty()) return true;

    // Capacity gate: only reserve pool slots for as many pages as the
    // PrefetchScheduler can accept. Truncating is a successful partial
    // prefetch (best-effort); allocating beyond free_queue_slots forces
    // LRU evictions that are rolled back when submit_batch rejects — net
    // thrash with zero I/O.
    const int free_q = prefetch_.free_queue_slots();
    if (free_q <= 0) {
        return false;
    }
    if ((int) needed.size() > free_q) {
        needed.resize((size_t) free_q);
    }
    // Sample oracle: also cap to free pool slots so we never LRU-evict.
    if (!allow_evict) {
        const int free_pool = pool_.n_free_unpinned();
        if (free_pool <= 0) {
            return false;
        }
        if ((int) needed.size() > free_pool) {
            needed.resize((size_t) free_pool);
        }
    }

    // Reserve N slots up-front. On failure (any unable to alloc, e.g. all
    // currently pinned per MAD-231), release the prefix and report.
    std::vector<int> slots;
    slots.reserve(needed.size());
    for (size_t i = 0; i < needed.size(); ++i) {
        const PageMeta & m = catalog_.at(needed[i]);
        const int s = allow_evict ? pool_.alloc_slot(m.size)
                                  : pool_.alloc_slot_no_evict(m.size);
        if (s < 0) {
            // Partial no-evict is OK (use free slots we already took).
            if (!allow_evict && !slots.empty()) {
                needed.resize(slots.size());
                break;
            }
            for (int prev : slots) {
                pool_.release_slot(prev);
            }
            return false;
        }
        ensure_slot_map_(s);
        if (speculative) pool_.set_speculative(s, true);
        slots.push_back(s);
    }
    if (slots.empty()) {
        return false;
    }

    // Wire the provisional page→slot bookkeeping (matches what prefetch_page
    // does pre-submit). The scheduler's submit_batch will see clean state —
    // it checks ITS OWN page_to_slot_ map, not ours — but populating ours
    // here prevents a concurrent caller (none today, but defensive) from
    // double-prefetching the same page.
    std::vector<PrefetchBatchRequest> reqs;
    reqs.reserve(needed.size());
    for (size_t i = 0; i < needed.size(); ++i) {
        const int page_idx = needed[i];
        const int slot     = slots[i];
        page_to_slot_[page_idx] = slot;
        page_loaded_[page_idx]  = false;
        slot_to_page_[slot]     = page_idx;

        const PageMeta & m = catalog_.at(page_idx);
        reqs.push_back(PrefetchBatchRequest{
            page_idx, (int) m.file_idx, m.file_offset, m.size,
            slot_ptr_(slot), pool_.slot_size(slot)
        });
    }

    if (!prefetch_.submit_batch(reqs)) {
        // Roll back EVERYTHING — page maps, slot_to_page maps, pool slots.
        // The all-or-nothing contract makes this clean: scheduler made no
        // promises about half-completed state, so we don't need to chase
        // partial scheduler progress.
        for (size_t i = 0; i < needed.size(); ++i) {
            page_to_slot_[needed[i]] = -1;
            slot_to_page_[slots[i]]  = -1;
            pool_.release_slot(slots[i]);
        }
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (count_dense_prefetch) {
        stats_.dense_prefetch_submitted += (uint64_t) needed.size();
    }
    for (int page_idx : needed) {
        if (page_idx < (int) prefetch_started_at_.size()) {
            prefetch_started_at_[page_idx] = now;
        }
        if (page_idx < (int) cross_layer_prefetch_candidate_.size() &&
            cross_layer_prefetch_candidate_[page_idx]) {
            ++stats_.cross_layer_prefetch_submitted;
        }
    }
    return true;
}

void WeightPager::pin_page(int page_idx) {
    if (!initialized_) return;
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return;
    const int slot = page_to_slot_[page_idx];
    if (slot < 0) return;  // page not resident; nothing to pin
    pool_.pin_slot(slot);
}

void WeightPager::unpin_page(int page_idx) {
    if (!initialized_) return;
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return;
    const int slot = page_to_slot_[page_idx];
    if (slot < 0) return;  // page evicted between pin and unpin — refcount was on a
                            // slot that's been reassigned; not our problem (and the
                            // refcount on the new owner would be wrong if we touched it).
    pool_.unpin_slot(slot);
}

std::vector<AdviseRange> compute_advise_ranges(const PageCatalog & catalog,
                                                int                 block_idx,
                                                int                 k) {
    std::vector<AdviseRange> out;
    if (k <= 0 || block_idx < 0) return out;

    // Walk forward k layers. pages_for_block already filters by block_idx;
    // we additionally drop consolidated parents — they carry the full
    // consolidated size for other consumers (eval-cb lookup), but the disk
    // bytes are already covered by the per-expert sub-pages, so advising
    // them again would double-count the file range.
    out.reserve((size_t) k * 8);  // ~3 dense / ~16 MoE per block, headroom
    for (int b = block_idx + 1; b <= block_idx + k; ++b) {
        const std::vector<int> pages = catalog.pages_for_block(b);
        for (int page_idx : pages) {
            const PageMeta & m = catalog.at(page_idx);
            if (m.is_consolidated) continue;    // parents — children cover the bytes
            if (m.size == 0)       continue;    // defence-in-depth for malformed entries
            out.push_back(AdviseRange{m.file_idx, m.file_offset, m.size});
        }
    }
    return out;
}

void WeightPager::advise_layer_lookahead(int block_idx, int k) {
    if (!initialized_ || file_io_ == nullptr) return;
    if (k <= 0 || block_idx < 0)              return;

    // Resident-aware skip. advise_prefetch issues posix_fadvise(WILLNEED)
    // NVMe readahead to warm the page cache for weights that will be paged
    // in from disk. When the working set is fully resident (size-class pool
    // holds every page in VRAM), nothing is ever re-read from NVMe, so the
    // readahead warms nothing and just burns syscalls on the decode critical
    // path — measured at ~35% of paged-resident decode time. Only advise when
    // at least one page in the [block+1, block+k] window is NOT resident,
    // i.e. real paging pressure exists. Self-adjusting: fires during the
    // initial cold load and for genuinely-paging (pool < working set) models,
    // stays out of the way once resident.
    bool any_unresident = false;
    for (int b = block_idx + 1; b <= block_idx + k && !any_unresident; ++b) {
        for (int p : catalog_.pages_for_block(b)) {
            if (p >= 0 && p < (int) page_loaded_.size() && !page_loaded_[p]) {
                any_unresident = true;
                break;
            }
        }
    }
    if (!any_unresident) return;

    const std::vector<AdviseRange> ranges = compute_advise_ranges(catalog_, block_idx, k);
    for (const auto & r : ranges) {
        file_io_->advise_prefetch((int) r.fd_idx, r.offset, r.size);
    }
}

void WeightPager::record_active_expert_pages(const std::vector<int> & page_indices) {
    if (page_indices.empty()) {
        return;
    }
    // First MoE snap past hash layers: drop sample-oracle sticky pins so MoE
    // can reclaim those slots. Hash pages remain resident and LRU-hot.
    for (int p : page_indices) {
        if (p >= 0 && p < catalog_.size()) {
            release_sample_sticky_if_past_hash(catalog_.at(p).block_idx);
            break;
        }
    }
    std::vector<int> snap;
    snap.reserve(page_indices.size() * 3);
    auto push_unique = [&](int p) {
        if (p < 0 || p >= catalog_.size()) {
            return;
        }
        for (int q : snap) {
            if (q == p) {
                return;
            }
        }
        snap.push_back(p);
    };
    for (int p : page_indices) {
        push_unique(p);
        // Expand to gate/up/down sisters for the same (block, expert).
        if (p >= 0 && p < catalog_.size()) {
            const PageMeta & m = catalog_.at(p);
            if (m.is_sub_expert && m.block_idx >= 0 && m.expert_idx >= 0) {
                for (int s : catalog_.pages_for_expert(m.block_idx, m.expert_idx)) {
                    push_unique(s);
                }
            }
        }
    }
    if (snap.empty()) {
        return;
    }
    // WP_ROUTE_TRACE=<path>: dump per-MoE-op demand (layer + sister-expanded
    // pages) for offline reuse-distance / Belady analysis. Inert when unset.
    {
        static std::ofstream * s_route_trace = []() -> std::ofstream * {
            const char * v = std::getenv("WP_ROUTE_TRACE");
            if (v == nullptr || v[0] == '\0') {
                return nullptr;
            }
            return new std::ofstream(v, std::ios::out | std::ios::trunc);
        }();
        if (s_route_trace != nullptr && s_route_trace->is_open()) {
            int blk = -1;
            if (snap[0] >= 0 && snap[0] < catalog_.size()) {
                blk = catalog_.at(snap[0]).block_idx;
            }
            (*s_route_trace) << blk;
            for (int p : snap) {
                (*s_route_trace) << ' ' << p;
            }
            (*s_route_trace) << '\n';
            s_route_trace->flush();
        }
    }
    // Real target routing is the strongest sticky signal (draft tid2eid alone
    // left sticky_l2_pins~3 with almost no score growth without a draft model).
    if (sticky_l2_enabled_) {
        for (int p : snap) {
            if (p >= 0 && p < (int) sticky_l2_score_.size() &&
                sticky_l2_score_[p] != 0xffffffffu) {
                ++sticky_l2_score_[p];
            }
        }
        // One credit per routing event (not per page) so refresh is not every
        // MoE op — rapid promote/demote thrash cost more than pin hits.
        ++sticky_l2_hits_since_refresh_;
        sticky_l2_refresh_if_due_("route");
    }
    // Oracle precision: only hash-layer pages are in the prediction set.
    for (int p : snap) {
        if (!is_tid2eid_hash_page_(p)) {
            continue;
        }
        ++stats_.oracle_actual_hash_pages;
        if (p < (int) oracle_pred_mark_.size() && oracle_pred_mark_[p]) {
            if (p < (int) oracle_pred_hit_.size() && !oracle_pred_hit_[p]) {
                oracle_pred_hit_[p] = true;
                ++stats_.oracle_tp;
            }
        } else {
            ++stats_.oracle_fn;
        }
    }
    hot_expert_history_.push_back(std::move(snap));
    if ((int) hot_expert_history_.size() > kHotHistoryMax) {
        hot_expert_history_.erase(hot_expert_history_.begin());
    }
    ++stats_.draft_hot_records;
}

void WeightPager::clear_draft_retain_() {
    for (int p : draft_retain_pages_) {
        if (sticky_l2_enabled_ &&
            p >= 0 &&
            p < (int) sticky_l2_mark_.size() &&
            sticky_l2_mark_[p] &&
            !sticky_l2_pinned_[p] &&
            p < (int) page_loaded_.size() &&
            page_loaded_[p]) {
            pin_page(p);
            sticky_l2_pinned_[p] = true;
        }
        unpin_page(p);
    }
    draft_retain_pages_.clear();
    clear_draft_tid2eid_mark_();
}

void WeightPager::clear_draft_tid2eid_mark_() {
    if (draft_tid2eid_mark_.empty()) {
        return;
    }
    std::fill(draft_tid2eid_mark_.begin(), draft_tid2eid_mark_.end(), false);
}

void WeightPager::union_push_(std::vector<int> & dst, int page_idx, int cap) const {
    if (page_idx < 0 || page_idx >= catalog_.size()) {
        return;
    }
    if ((int) dst.size() >= cap) {
        return;
    }
    for (int q : dst) {
        if (q == page_idx) {
            return;
        }
    }
    dst.push_back(page_idx);
}

void WeightPager::register_tid2eid_host(int block_idx, int n_expert_used, int n_vocab,
                                        const int32_t * host_data) {
    if (block_idx < 0 || n_expert_used <= 0 || n_vocab <= 0 || host_data == nullptr) {
        return;
    }
    const size_t n = (size_t) n_expert_used * (size_t) n_vocab;
    // Replace existing table for this block if re-registered.
    for (Tid2EidTable & t : tid2eid_tables_) {
        if (t.block_idx == block_idx) {
            t.n_expert_used = n_expert_used;
            t.n_vocab       = n_vocab;
            t.data.assign(host_data, host_data + n);
            return;
        }
    }
    Tid2EidTable t;
    t.block_idx     = block_idx;
    t.n_expert_used = n_expert_used;
    t.n_vocab       = n_vocab;
    t.data.assign(host_data, host_data + n);
    tid2eid_tables_.push_back(std::move(t));
    LLAMA_LOG_INFO("wp::WeightPager: tid2eid host table blk=%d n_used=%d n_vocab=%d (%.1f MiB)\n",
                   block_idx, n_expert_used, n_vocab,
                   (double) (n * sizeof(int32_t)) / (1024.0 * 1024.0));
}

void WeightPager::collect_tid2eid_pages_(const int32_t * tokens, int n_tokens,
                                         std::vector<int> & out, int cap) const {
    if (tokens == nullptr || n_tokens <= 0 || tid2eid_tables_.empty()) {
        return;
    }
    for (const Tid2EidTable & t : tid2eid_tables_) {
        for (int ti = 0; ti < n_tokens; ++ti) {
            const int32_t tok = tokens[ti];
            if (tok < 0 || tok >= t.n_vocab) {
                continue;
            }
            const size_t base = (size_t) tok * (size_t) t.n_expert_used;
            for (int k = 0; k < t.n_expert_used; ++k) {
                const int32_t eid = t.data[base + (size_t) k];
                if (eid < 0) {
                    continue;
                }
                for (int p : catalog_.pages_for_expert(t.block_idx, (int) eid)) {
                    union_push_(out, p, cap);
                    if ((int) out.size() >= cap) {
                        return;
                    }
                }
            }
        }
    }
}

void WeightPager::set_draft_window(int n_draft) {
    if (n_draft <= 0) {
        ++stats_.draft_window_closes;
        clear_draft_retain_();
        draft_window_ = 0;
        return;
    }
    ++stats_.draft_window_opens;
    draft_window_ = n_draft;
}

int WeightPager::harvest_ready_prefetches_() {
    if (!initialized_) {
        return 0;
    }
    prefetch_.tick();
    int n = 0;
    // 1) Commit Done prefetches still mapped in the pool.
    for (int p = 0; p < (int) page_to_slot_.size(); ++p) {
        if (page_to_slot_[p] < 0 || page_loaded_[p]) {
            continue;
        }
        if (prefetch_.is_loaded(p)) {
            page_loaded_[p] = true;
            pool_.mark_used(page_to_slot_[p]);
            double seconds = 0.0;
            if (p < (int) prefetch_started_at_.size() &&
                prefetch_started_at_[p] != std::chrono::steady_clock::time_point{}) {
                seconds = seconds_since(prefetch_started_at_[p]);
                prefetch_started_at_[p] = std::chrono::steady_clock::time_point{};
            }
            record_page_in_(catalog_.at(p).size, seconds);
            prefetch_.reap(p);
            ++n;
            continue;
        }
        if (prefetch_.is_failed(p)) {
            const int slot = page_to_slot_[p];
            prefetch_.reap(p);
            page_to_slot_[p] = -1;
            if (slot >= 0 && slot < (int) slot_to_page_.size()) {
                slot_to_page_[slot] = -1;
            }
            pool_.release_slot(slot);
            ++n;
            continue;
        }
        if (!prefetch_.has_page(p)) {
            const int slot = page_to_slot_[p];
            page_to_slot_[p] = -1;
            if (slot >= 0 && slot < (int) slot_to_page_.size()) {
                slot_to_page_[slot] = -1;
            }
            pool_.release_slot(slot);
            ++n;
        }
    }
    // 2) Free scheduler slots still holding Done/Failed after pool eviction
    //    dropped the WeightPager mapping (the free_q=0 deadlock: depth-4
    //    Done orphans never reaped because ensure never touched those pages).
    n += prefetch_.reap_finished();
    return n;
}

int WeightPager::submit_cold_waves_(const std::vector<int> & cold, bool allow_evict) {
    if (cold.empty()) {
        return 0;
    }
    // Default 1 wave: at most free_queue_slots (~4) cold submits per fire.
    // Multi-wave (64) thrash-inflated page_ins and hurt t/s when draft tokens
    // diverge from the actual sample (accept=0 strip mode).
    static int s_max_waves = -1;
    if (s_max_waves < 0) {
        const char * v = std::getenv("WP_DRAFT_PREFETCH_WAVES");
        s_max_waves = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 1;
        if (s_max_waves < 1) {
            s_max_waves = 1;
        }
    }

    int submitted = 0;
    size_t i = 0;
    for (int wave = 0; wave < s_max_waves && i < cold.size(); ++wave) {
        stats_.draft_prefetch_harvested += (uint64_t) harvest_ready_prefetches_();
        int free_q = prefetch_.free_queue_slots();
        if (free_q <= 0) {
            // Drain completions: Done slots stuck unreaped were the main
            // free_q=0 cause (depth 4, only ensure() used to reap).
            for (int d = 0; d < 64 && free_q <= 0; ++d) {
                tick();
                stats_.draft_prefetch_harvested += (uint64_t) harvest_ready_prefetches_();
                free_q = prefetch_.free_queue_slots();
            }
            if (free_q <= 0) {
                ++stats_.draft_prefetch_queue_blocked;
                break;
            }
        }
        if (!allow_evict) {
            const int free_pool = pool_.n_free_unpinned();
            if (free_pool <= 0) {
                break; // do not thrash; ensure path will load later
            }
            if (free_q > free_pool) {
                free_q = free_pool;
            }
        }

        std::vector<int> batch;
        batch.reserve((size_t) free_q);
        while (i < cold.size() && (int) batch.size() < free_q) {
            const int p = cold[i++];
            if (p < 0 || p >= (int) page_to_slot_.size()) {
                continue;
            }
            if (page_to_slot_[p] >= 0) {
                continue; // already reserved / harvested
            }
            batch.push_back(p);
        }
        if (batch.empty()) {
            continue;
        }

        if (prefetch_pages_batch(batch, /*count_dense_prefetch=*/false, allow_evict)) {
            for (int p : batch) {
                if (p >= 0 && p < (int) page_to_slot_.size() && page_to_slot_[p] >= 0) {
                    ++submitted;
                }
            }
        } else {
            for (int p : batch) {
                if (prefetch_page(p, /*count_dense_prefetch=*/false, allow_evict)) {
                    ++submitted;
                }
            }
        }
        tick(); // flush SQEs for this wave
    }
    // Final harvest of anything that completed during submit.
    stats_.draft_prefetch_harvested += (uint64_t) harvest_ready_prefetches_();
    return submitted;
}

bool WeightPager::draft_oracle_should_run() {
    if (!initialized_) {
        return false;
    }
    static int s_adaptive = -1;
    if (s_adaptive < 0) {
        const char * v = std::getenv("WP_DRAFT_ADAPTIVE");
        s_adaptive = (v != nullptr && v[0] == '0') ? 0 : 1;
    }
    if (s_adaptive == 0) {
        return true;
    }

    static int s_always_first = -1;
    if (s_always_first < 0) {
        const char * v = std::getenv("WP_DRAFT_ALWAYS_FIRST");
        s_always_first = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 4;
        if (s_always_first < 0) {
            s_always_first = 0;
        }
    }
    static int s_opportunities = 0;
    const int opp = s_opportunities++;
    if (opp < s_always_first) {
        return true;
    }

    // Update hit ratio for the window since the last draft fire.
    if (last_tid2eid_n_ > 0) {
        const uint64_t wh = stats_.draft_tid2eid_hits_in_ensure - hits_at_last_fire_;
        last_hit_ratio_ = (float) wh / (float) last_tid2eid_n_;
    }

    // Fully warm hash set: nothing cold last fire.
    static int s_warm_need = -1;
    if (s_warm_need < 0) {
        const char * v = std::getenv("WP_DRAFT_WARM_STREAK");
        s_warm_need = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 2;
        if (s_warm_need < 1) {
            s_warm_need = 1;
        }
    }
    if (draft_warm_streak_ >= s_warm_need) {
        ++stats_.draft_oracle_skips;
        sticky_l2_refresh_("adaptive-skip");
        return false;
    }

    // Explicit duty cycle (manual override).
    static int s_every = -1;
    if (s_every < 0) {
        const char * v = std::getenv("WP_DRAFT_EVERY");
        s_every = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 1;
        if (s_every < 1) {
            s_every = 1;
        }
    }

    // Hit-based duty: if last window barely used draft pages, draft less often.
    // High hit ratio => always run (oracle is paying off).
    int every = s_every;
    if (last_tid2eid_n_ >= 8 && last_hit_ratio_ < 0.10f) {
        every = std::max(every, 2);
    } else if (last_tid2eid_n_ >= 8 && last_hit_ratio_ > 0.30f) {
        every = 1;
    }

    if (every > 1 && ((opp - s_always_first) % every) != 0) {
        ++stats_.draft_oracle_skips;
        sticky_l2_refresh_("duty-skip");
        return false;
    }
    return true;
}

bool WeightPager::is_tid2eid_hash_page_(int page_idx) const {
    if (page_idx < 0 || page_idx >= catalog_.size() || tid2eid_tables_.empty()) {
        return false;
    }
    const PageMeta & m = catalog_.at(page_idx);
    if (!m.is_sub_expert || m.block_idx < 0) {
        return false;
    }
    for (const Tid2EidTable & t : tid2eid_tables_) {
        if (t.block_idx == m.block_idx) {
            return true;
        }
    }
    return false;
}

int WeightPager::free_stale_hash_slots_(int n_need, const std::vector<int> & keep_pages) {
    if (n_need <= 0 || !initialized_) {
        return 0;
    }
    std::vector<bool> keep((size_t) catalog_.size(), false);
    for (int p : keep_pages) {
        if (p >= 0 && p < (int) keep.size()) {
            keep[(size_t) p] = true;
        }
    }
    // Candidates: resident unpinned hash pages not needed for this fire.
    // Prefer low hit_count (cold hash leftovers) over recently ensure-hit ones.
    struct Cand { int page; uint32_t hits; };
    std::vector<Cand> cands;
    cands.reserve((size_t) n_need * 4);
    for (int p = 0; p < (int) catalog_.size(); ++p) {
        if (keep[(size_t) p] || !is_tid2eid_hash_page_(p)) {
            continue;
        }
        if (p >= (int) page_to_slot_.size() || page_to_slot_[p] < 0) {
            continue;
        }
        if (p >= (int) page_loaded_.size() || !page_loaded_[p]) {
            continue;
        }
        const int slot = page_to_slot_[p];
        if (pool_.is_pinned(slot)) {
            continue;
        }
        if (sticky_l2_enabled_ &&
            p < (int) sticky_l2_pinned_.size() &&
            sticky_l2_pinned_[p]) {
            continue;
        }
        // Skip pages that scored oracle TP this window — still hot for hash.
        if (p < (int) oracle_pred_hit_.size() && oracle_pred_hit_[p]) {
            continue;
        }
        cands.push_back({ p, pool_.hit_count(slot) });
        if ((int) cands.size() >= n_need * 8) {
            break;
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand & a, const Cand & b) { return a.hits < b.hits; });
    int freed = 0;
    for (const Cand & c : cands) {
        if (freed >= n_need) {
            break;
        }
        const int p = c.page;
        if (p >= (int) page_to_slot_.size() || page_to_slot_[p] < 0) {
            continue;
        }
        const int slot = page_to_slot_[p];
        if (pool_.is_pinned(slot)) {
            continue;
        }
        on_pool_evict_(slot);
        pool_.release_slot(slot);
        ++freed;
    }
    stats_.oracle_hash_slots_freed += (uint64_t) freed;
    return freed;
}

void WeightPager::clear_sample_sticky_() {
    for (int p : sample_sticky_pages_) {
        unpin_page(p);
    }
    sample_sticky_pages_.clear();
}

void WeightPager::release_sample_sticky_if_past_hash(int block_idx) {
    if (sample_sticky_pages_.empty() || block_idx < 0) {
        return;
    }
    int max_hash = -1;
    for (const Tid2EidTable & t : tid2eid_tables_) {
        if (t.block_idx > max_hash) {
            max_hash = t.block_idx;
        }
    }
    // No hash tables registered: never auto-release on layer index.
    if (max_hash < 0) {
        return;
    }
    if (block_idx > max_hash) {
        clear_sample_sticky_();
    }
}

void WeightPager::install_sample_sticky_(const std::vector<int> & tid_pages, int cap) {
    clear_sample_sticky_();
    if (cap < 1 || tid_pages.empty()) {
        return;
    }
    // Pin *in-flight only*. Do not pin or mark_used resident pages: both
    // starve MoE LRU and were measured at +~700 page_ins with submitted=0.
    sample_sticky_pages_.reserve((size_t) cap);
    for (int p : tid_pages) {
        if ((int) sample_sticky_pages_.size() >= cap) {
            break;
        }
        if (p < 0 || p >= (int) page_to_slot_.size() || page_to_slot_[p] < 0) {
            continue;
        }
        if (sticky_l2_enabled_ &&
            p < (int) sticky_l2_pinned_.size() &&
            sticky_l2_pinned_[p]) {
            continue;
        }
        const bool loaded = p < (int) page_loaded_.size() && page_loaded_[p];
        if (loaded) {
            continue;
        }
        bool dup = false;
        for (int q : sample_sticky_pages_) {
            if (q == p) { dup = true; break; }
        }
        if (dup) {
            continue;
        }
        pin_page(p);
        sample_sticky_pages_.push_back(p);
    }
    stats_.oracle_sticky_pins += (uint64_t) sample_sticky_pages_.size();
}

void WeightPager::oracle_finalize_fp_() {
    if (oracle_pred_mark_.empty()) {
        return;
    }
    for (size_t p = 0; p < oracle_pred_mark_.size(); ++p) {
        if (oracle_pred_mark_[p] && (p >= oracle_pred_hit_.size() || !oracle_pred_hit_[p])) {
            ++stats_.oracle_fp;
        }
    }
}

void WeightPager::oracle_begin_prediction_(const std::vector<int> & pages) {
    oracle_finalize_fp_();
    if (oracle_pred_mark_.size() != (size_t) catalog_.size()) {
        oracle_pred_mark_.assign((size_t) catalog_.size(), false);
        oracle_pred_hit_.assign((size_t) catalog_.size(), false);
    } else {
        std::fill(oracle_pred_mark_.begin(), oracle_pred_mark_.end(), false);
        std::fill(oracle_pred_hit_.begin(), oracle_pred_hit_.end(), false);
    }
    for (int p : pages) {
        if (p >= 0 && p < (int) oracle_pred_mark_.size()) {
            oracle_pred_mark_[p] = true;
            ++stats_.oracle_pred_pages;
        }
    }
}

int WeightPager::note_sampled_token(int32_t token) {
    if (!initialized_ || token < 0) {
        return 0;
    }
    pending_sample_token_ = token;
    pending_sample_flushed_ = false;
    // Mark prediction for precision stats immediately (even if I/O deferred).
    std::vector<int> tid_pages;
    tid_pages.reserve(64);
    collect_tid2eid_pages_(&token, 1, tid_pages, kHotExpertCap);
    oracle_begin_prediction_(tid_pages);
    ++stats_.oracle_sample_fires;

    // Default: start I/O at sample time (post-forward MoE pins are already
    // down). FA is too short alone to finish a 16-page wave before hash L0-2;
    // sampling gives the sample->next-FA gap + FA as lead time.
    // WP_SAMPLE_ORACLE_EAGER=0 defers submit to layer-0 FA only.
    static int s_eager = -1;
    if (s_eager < 0) {
        const char * v = std::getenv("WP_SAMPLE_ORACLE_EAGER");
        s_eager = (v != nullptr && v[0] == '0') ? 0 : 1;
    }
    if (s_eager == 1) {
        return flush_sample_oracle_at_fa();
    }
    return 0;
}

std::vector<int> WeightPager::pin_oracle_protect_set_() {
    // Temp-pin recent target MoE routing so sample-oracle LRU will not steal
    // those working-set pages. Sticky L2 already holds its own pins.
    static int s_hist = -1;
    if (s_hist < 0) {
        const char * v = std::getenv("WP_SAMPLE_ORACLE_PROTECT_HIST");
        s_hist = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 8;
        if (s_hist < 0) {
            s_hist = 0;
        }
        if (s_hist > kHotHistoryMax) {
            s_hist = kHotHistoryMax;
        }
    }
    std::vector<int> pinned;
    if (s_hist == 0 || hot_expert_history_.empty()) {
        return pinned;
    }
    std::vector<bool> seen((size_t) catalog_.size(), false);
    const int n_hist = (int) hot_expert_history_.size();
    const int from   = n_hist > s_hist ? n_hist - s_hist : 0;
    pinned.reserve(256);
    for (int hi = from; hi < n_hist; ++hi) {
        for (int p : hot_expert_history_[(size_t) hi]) {
            if (p < 0 || p >= (int) catalog_.size() || seen[(size_t) p]) {
                continue;
            }
            if (p >= (int) page_to_slot_.size() || page_to_slot_[p] < 0) {
                continue;
            }
            if (p >= (int) page_loaded_.size() || !page_loaded_[p]) {
                continue;
            }
            // Already sticky-pinned: skip (would double-pin).
            if (sticky_l2_enabled_ &&
                p < (int) sticky_l2_pinned_.size() &&
                sticky_l2_pinned_[p]) {
                continue;
            }
            seen[(size_t) p] = true;
            pin_page(p);
            pinned.push_back(p);
        }
    }
    stats_.oracle_protect_pins += (uint64_t) pinned.size();
    return pinned;
}

int WeightPager::flush_sample_oracle_at_fa() {
    if (!initialized_ || pending_sample_flushed_ || pending_sample_token_ < 0) {
        return 0;
    }
    pending_sample_flushed_ = true;
    // DIAG: set WP_SAMPLE_ORACLE_FLUSH=0 to measure note-only thrash.
    static int s_flush = -1;
    if (s_flush < 0) {
        const char * v = std::getenv("WP_SAMPLE_ORACLE_FLUSH");
        s_flush = (v != nullptr && v[0] == '0') ? 0 : 1;
    }
    if (s_flush == 0) {
        return 0;
    }
    const int32_t tok = pending_sample_token_;

    std::vector<int> tid_pages;
    tid_pages.reserve(64);
    collect_tid2eid_pages_(&tok, 1, tid_pages, kHotExpertCap);
    stats_.draft_tid2eid_pages += (uint64_t) tid_pages.size();
    for (int p : tid_pages) {
        if (p >= 0 && p < (int) draft_tid2eid_mark_.size()) {
            draft_tid2eid_mark_[p] = true;
        }
    }

    static int s_max_cold = -1;
    if (s_max_cold < 0) {
        const char * v = std::getenv("WP_SAMPLE_ORACLE_MAX");
        s_max_cold = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 16;
        if (s_max_cold < 1) {
            s_max_cold = 1;
        }
        if (s_max_cold > 64) {
            s_max_cold = 64;
        }
    }

    std::vector<int> cold;
    cold.reserve(tid_pages.size());
    int resident = 0;
    for (int p : tid_pages) {
        if (p < 0 || p >= catalog_.size() || catalog_.at(p).is_pinned) {
            continue;
        }
        const bool has_slot = p < (int) page_to_slot_.size() && page_to_slot_[p] >= 0;
        const bool loaded   = has_slot && p < (int) page_loaded_.size() && page_loaded_[p];
        if (loaded || has_slot) {
            ++resident;
        } else {
            cold.push_back(p);
        }
    }
    stats_.draft_prefetch_pages_resident += (uint64_t) resident;
    stats_.draft_tid2eid_cold            += (uint64_t) cold.size();
    stats_.draft_cold_pages              += (uint64_t) cold.size();

    // WP_SAMPLE_ORACLE_EVICT:
    //   0 = free pool slots only (default; page_ins-safe)
    //   1 = free_stale hash when free_pool short
    //   2 = also protected MoE LRU
    static int s_evict = -1;
    if (s_evict < 0) {
        const char * v = std::getenv("WP_SAMPLE_ORACLE_EVICT");
        if (v != nullptr && v[0] == '1') {
            s_evict = 1;
        } else if (v != nullptr && v[0] == '2') {
            s_evict = 2;
        } else {
            s_evict = 0;
        }
    }

    auto finish_sticky = [&]() {
        install_sample_sticky_(tid_pages, s_max_cold);
    };

    // No cold work, or free-only with a full pool: do not touch retain/harvest.
    // Measured: any flush side-effect with free_pool==0 raised page_ins ~+700.
    if (cold.empty()) {
        return 0;
    }
    if (s_evict == 0 && pool_.n_free_unpinned() <= 0) {
        return 0;
    }

    if ((int) cold.size() > s_max_cold) {
        cold.resize((size_t) s_max_cold);
    }

    stats_.draft_prefetch_harvested += (uint64_t) harvest_ready_prefetches_();
    // Re-check free pool after harvest (Done slots may free).
    if (s_evict == 0 && pool_.n_free_unpinned() <= 0) {
        return 0;
    }

    int submitted = 0;
    int free_sub  = 0;
    int evict_sub = 0;

    auto still_cold_of = [&](const std::vector<int> & src) {
        std::vector<int> out;
        out.reserve(src.size());
        for (int p : src) {
            if (p >= 0 && p < (int) page_to_slot_.size() && page_to_slot_[p] < 0) {
                out.push_back(p);
            }
        }
        return out;
    };

    ++stats_.draft_prefetch_calls;
    ++draft_oracle_fires_;

    // Phase 1: free pool slots only.
    {
        const int free_pool = pool_.n_free_unpinned();
        if (free_pool > 0) {
            std::vector<int> free_batch;
            free_batch.reserve((size_t) free_pool);
            std::vector<int> remain;
            remain.reserve(cold.size());
            for (int p : cold) {
                if ((int) free_batch.size() < free_pool &&
                    p >= 0 && p < (int) page_to_slot_.size() && page_to_slot_[p] < 0) {
                    free_batch.push_back(p);
                } else {
                    remain.push_back(p);
                }
            }
            if (!free_batch.empty()) {
                free_sub = submit_cold_waves_(free_batch, /*allow_evict=*/false);
                submitted += free_sub;
            }
            cold.swap(remain);
        }
    }

    // Phase 2: free_stale hash only when still no free slots (sticky from prior
    // fire should leave headroom; avoid hash churn when free_pool > 0).
    if (s_evict >= 1 && !cold.empty()) {
        std::vector<int> still = still_cold_of(cold);
        if (!still.empty()) {
            const int free_pool = pool_.n_free_unpinned();
            if (free_pool <= 0) {
                (void) free_stale_hash_slots_((int) still.size(), tid_pages);
            } else if (free_pool < (int) still.size()) {
                // Partial: free only the shortfall via stale hash.
                (void) free_stale_hash_slots_((int) still.size() - free_pool, tid_pages);
            }
            const int n = submit_cold_waves_(still, /*allow_evict=*/false);
            free_sub  += n;
            submitted += n;
            cold = still_cold_of(still);
        }
    }

    // Phase 3 (opt-in): protected MoE LRU.
    if (s_evict >= 2 && !cold.empty()) {
        std::vector<int> still = still_cold_of(cold);
        if (!still.empty()) {
            std::vector<int> protect = pin_oracle_protect_set_();
            evict_sub = submit_cold_waves_(still, /*allow_evict=*/true);
            submitted += evict_sub;
            for (int p : protect) {
                unpin_page(p);
            }
        }
    }

    stats_.draft_prefetch_pages_submitted += (uint64_t) submitted;
    stats_.oracle_pages_submitted         += (uint64_t) submitted;
    stats_.oracle_pages_free_slot         += (uint64_t) free_sub;
    stats_.oracle_pages_evict_slot        += (uint64_t) evict_sub;

    // Sticky retain: loaded + in-flight oracle pages (cap MAX) until next sample.
    finish_sticky();
    return submitted;
}

int WeightPager::prefetch_hot_experts(const int32_t * tokens, int n_tokens, int source) {
    if (!initialized_) {
        return 0;
    }
    // Free SQ slots held by unreaped Done prefetches from prior layers.
    stats_.draft_prefetch_harvested += (uint64_t) harvest_ready_prefetches_();

    // Fresh retain set for this fire (old pins drop first).
    clear_draft_retain_();

    // Hash-only oracle: tid2eid(token). source=0 sample (ground-truth next
    // input); source=1 draft (speculative — wrong under strip unless accept).
    static int s_max_tok = -1;
    if (s_max_tok < 0) {
        const char * v = std::getenv("WP_DRAFT_ORACLE_MAX_TOK");
        s_max_tok = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 1;
        if (s_max_tok < 1) {
            s_max_tok = 1;
        }
    }
    const int n_use = std::min(n_tokens, s_max_tok);

    std::vector<int> tid_pages;
    tid_pages.reserve(64);
    collect_tid2eid_pages_(tokens, n_use, tid_pages, kHotExpertCap);
    stats_.draft_tid2eid_pages += (uint64_t) tid_pages.size();
    // Sample path already called oracle_begin_prediction_ in note_sampled_token.
    if (source != 0) {
        oracle_begin_prediction_(tid_pages);
        ++stats_.oracle_draft_fires;
    }

    for (int p : tid_pages) {
        if (p >= 0 && p < (int) draft_tid2eid_mark_.size()) {
            draft_tid2eid_mark_[p] = true;
        }
    }

    if (tid_pages.empty()) {
        draft_warm_streak_ = 0;
        ++draft_oracle_fires_;
        hits_at_last_fire_ = stats_.draft_tid2eid_hits_in_ensure;
        last_tid2eid_n_ = 0;
        last_tid_cold_ = 0;
        sticky_l2_refresh_("draft-empty");
        return 0;
    }
    ++stats_.draft_prefetch_calls;
    ++draft_oracle_fires_;
    set_draft_window(std::max(1, n_use));

    int resident = 0;
    int tid_cold = 0;
    std::vector<int> cold;
    cold.reserve(tid_pages.size());
    for (int p : tid_pages) {
        if (p < 0 || p >= catalog_.size()) {
            continue;
        }
        if (catalog_.at(p).is_pinned) {
            continue;
        }
        const bool has_slot = p < (int) page_to_slot_.size() && page_to_slot_[p] >= 0;
        const bool loaded   = has_slot && p < (int) page_loaded_.size() && page_loaded_[p];
        if (loaded) {
            ++resident;
        } else if (!has_slot) {
            cold.push_back(p);
            ++tid_cold;
        } else {
            ++resident; // in-flight or reserved
        }
    }
    stats_.draft_prefetch_pages_resident += (uint64_t) resident;
    stats_.draft_cold_pages              += (uint64_t) tid_cold;
    stats_.draft_tid2eid_cold            += (uint64_t) tid_cold;

    // Sample oracle never LRU-evicts: free slots only. That stops the thrash
    // where hash speculation stole MoE working-set pages (+1k page_ins).
    const bool allow_evict = (source != 0);
    const int submitted = submit_cold_waves_(cold, allow_evict);
    stats_.draft_prefetch_pages_submitted += (uint64_t) submitted;

    // Pin only pages that need eviction protection through the next target
    // decode: cold/in-flight. Pinning the full already-resident tid2eid set
    // (~54/page) every token thrash-evicted non-hash experts (+1k page_ins).
    draft_retain_pages_.reserve(tid_pages.size());
    for (int p : tid_pages) {
        if (p < 0 || p >= (int) page_to_slot_.size()) {
            continue;
        }
        if (page_to_slot_[p] < 0) {
            continue;
        }
        const bool loaded = p < (int) page_loaded_.size() && page_loaded_[p];
        if (loaded && source == 0) {
            continue; // sample oracle: resident hash pages need no retain pin
        }
        if (sticky_l2_enabled_ &&
            p < (int) sticky_l2_pinned_.size() &&
            sticky_l2_pinned_[p]) {
            continue;
        }
        pin_page(p);
        draft_retain_pages_.push_back(p);
    }
    stats_.draft_retain_pins += (uint64_t) draft_retain_pages_.size();

    if (tid_cold == 0) {
        ++draft_warm_streak_;
    } else {
        draft_warm_streak_ = 0;
    }

    // Baseline for next window's hit-ratio (hits accrue during target ensure).
    hits_at_last_fire_ = stats_.draft_tid2eid_hits_in_ensure;
    last_tid2eid_n_    = (int) tid_pages.size();
    last_tid_cold_     = tid_cold;

    static int s_draft_stats = -1;
    if (s_draft_stats < 0) {
        const char * v = std::getenv("WP_DRAFT_STATS");
        s_draft_stats = (v != nullptr && v[0] == '1') ? 1 : 0;
    }
    if (s_draft_stats) {
        LLAMA_LOG_WARN("draft-oracle: n_tok=%d tid2eid=%d cold=%d submitted=%d retain=%d free_q=%d hit_ratio=%.2f\n",
                       n_tokens, (int) tid_pages.size(), tid_cold, submitted,
                       (int) draft_retain_pages_.size(), prefetch_.free_queue_slots(),
                       last_hit_ratio_);
    }

    sticky_l2_refresh_("draft-fire");

    return submitted;
}

int WeightPager::prefetch_sticky_hot_experts() {
    if (!initialized_ || !sticky_l2_enabled_) {
        return 0;
    }
    ++stats_.sticky_spec_fires;
    stats_.draft_prefetch_harvested += (uint64_t) harvest_ready_prefetches_();

    // Cap cold I/O so we do not thrash the pool during FA.
    static int s_max_cold = -1;
    if (s_max_cold < 0) {
        const char * v = std::getenv("WP_STICKY_SPEC_MAX");
        // Default 16: ~one multi-QD wave; higher thrash measured at 32 every FA.
        s_max_cold = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 16;
        if (s_max_cold < 1) {
            s_max_cold = 1;
        }
        if (s_max_cold > 64) {
            s_max_cold = 64;
        }
    }

    // Cold submit = recent routing snaps only (sisters already expanded in
    // record_active_expert_pages). Global score ranks thrash; pins use scores
    // via sticky_l2_refresh_. Last ~4 snaps ~ a few MoE layers of history.
    std::vector<int> cold;
    cold.reserve((size_t) s_max_cold);
    int resident = 0;
    int hist_pages = 0;
    std::vector<bool> seen_cold((size_t) catalog_.size(), false);
    const int hist_n = std::min(4, (int) hot_expert_history_.size());
    for (int h = 0; h < hist_n && (int) cold.size() < s_max_cold; ++h) {
        const auto & snap = hot_expert_history_[hot_expert_history_.size() - 1 - (size_t) h];
        hist_pages += (int) snap.size();
        for (int p : snap) {
            if ((int) cold.size() >= s_max_cold) {
                break;
            }
            if (p < 0 || p >= catalog_.size() || catalog_.at(p).is_pinned) {
                continue;
            }
            if (seen_cold[(size_t) p]) {
                continue;
            }
            seen_cold[(size_t) p] = true;
            const bool has_slot = p < (int) page_to_slot_.size() && page_to_slot_[p] >= 0;
            const bool loaded   = has_slot && p < (int) page_loaded_.size() && page_loaded_[p];
            if (loaded || has_slot) {
                ++resident;
                continue;
            }
            cold.push_back(p);
        }
    }
    stats_.sticky_spec_pages_resident += (uint64_t) resident;

    const int submitted = cold.empty() ? 0 : submit_cold_waves_(cold);
    stats_.sticky_spec_pages_submitted += (uint64_t) submitted;

    // Do not force pin refresh every FA fire — that demote/promote churn
    // raised page_ins. Route-based sticky_l2_refresh_if_due_ is enough.
    tick();

    static int s_stats = -1;
    if (s_stats < 0) {
        const char * v = std::getenv("WP_STICKY_L2_STATS");
        s_stats = (v != nullptr && v[0] == '1') ? 1 : 0;
    }
    if (s_stats) {
        int pins = 0;
        for (bool pinned : sticky_l2_pinned_) {
            if (pinned) {
                ++pins;
            }
        }
        LLAMA_LOG_WARN("sticky-spec: hist=%d cold=%d submitted=%d resident=%d free_q=%d pins=%d\n",
                       hist_pages, (int) cold.size(), submitted, resident,
                       prefetch_.free_queue_slots(), pins);
    }
    return submitted;
}

void WeightPager::mark_cross_layer_prefetch_candidates(const std::vector<int> & page_indices) {
    if (!initialized_) return;
    for (int page_idx : page_indices) {
        if (page_idx < 0 || page_idx >= (int) cross_layer_prefetch_candidate_.size()) continue;
        if (catalog_.at(page_idx).is_pinned) continue;
        cross_layer_prefetch_candidate_[page_idx] = true;
    }
}

int WeightPager::page_in_sync_(int page_idx, int reuse_slot) {
    // Synchronous read into a slot, bypassing the prefetch pipeline. Used by
    // ensure() on miss (reuse_slot < 0: alloc a fresh slot) and by ensure_batch
    // as its per-page failure fallback (reuse_slot >= 0: read into that already-
    // reserved+pinned slot; never release it here). Host transports read into
    // staging and then hand off to GpuTransport; P2P transports read into VRAM.
    const bool owns_slot = (reuse_slot < 0);

    static int s_diag_count = 0;
    const bool diag = (s_diag_count < 5);
    if (diag) {
        const PageMeta & dm = catalog_.at(page_idx);
        LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: ENTER page=%d name=%s file_idx=%u offset=%lu size=%zu\n",
                        s_diag_count, page_idx, dm.tensor_name.c_str(),
                        (unsigned) dm.file_idx, (unsigned long) dm.file_offset, dm.size);
    }

    const auto io_t0 = std::chrono::steady_clock::now();

    const PageMeta & m = catalog_.at(page_idx);

    int slot = reuse_slot;
    if (slot < 0) {
        slot = pool_.alloc_slot(m.size);
        if (slot < 0) return -1;
        ensure_slot_map_(slot);
    }
    void * dst = slot_ptr_(slot);
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: alloc_slot ok, slot=%d dst=%p\n", s_diag_count, slot, dst);

    // Use the shared pinned staging buffer allocated at init time. Pinning
    // a fresh buffer per call costs hundreds of ms for hundred-MB tensors
    // and would dominate the paging path; the shared buffer is sized to
    // max_page_size so any individual page fits.
    void * staging = sync_staging_;
    if (staging == nullptr || m.size > sync_staging_size_) {
        LLAMA_LOG_ERROR("wp::WeightPager::page_in_sync_: page %d size %zu exceeds shared staging size %zu\n",
                        page_idx, m.size, sync_staging_size_);
        if (owns_slot) pool_.release_slot(slot);
        return -1;
    }

    if (host_tier_) {
        const void * host_ptr = host_tier_->lookup(page_idx);
        if (host_ptr != nullptr) {
            int evt = transport_.stage_in(dst, host_ptr, m.size, pool_.slot_size(slot));
            if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: host tier stage_in returned evt=%d\n", s_diag_count, evt);
            if (evt < 0) {
                LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: host tier gpu stage_in failed for page %d\n",
                               page_idx);
                if (owns_slot) pool_.release_slot(slot);
                return -1;
            }
            transport_.release_event(evt);

            page_to_slot_[page_idx] = slot;
            page_loaded_[page_idx]  = true;
            slot_to_page_[slot]     = page_idx;
            ++stats_.host_tier_hits;
            ++stats_.page_ins;
            if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT host-tier slot=%d\n", s_diag_count, slot);
            ++s_diag_count;
            return slot;
        }
    }

    // Stage 1: blocking read via the file IO layer. P2P reads directly
    // into the VRAM slot; host transports read into the shared pinned
    // staging buffer and use GpuTransport for the H2D copy below.
    const bool host_store_possible = host_tier_ && m.size <= host_tier_->budget_bytes();
    bool direct_to_device = file_io_->direct_to_device() && !host_store_possible;
    void * read_dst = direct_to_device ? dst : staging;
    auto read_once = [&]() {
        const uint64_t req_id = next_io_req_id_++;  // unique, high-bit-tagged
        bool read_ok = file_io_->submit(req_id, (int) m.file_idx, m.file_offset, m.size, read_dst);
        if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: submit returned ok=%d\n", s_diag_count, (int)read_ok);
        if (read_ok) {
            file_io_->flush();
            // Wait for OUR completion by req_id. The FileIOLayer demux buffers
            // any concurrent prefetch completion reaped along the way rather
            // than discarding it, so no sibling read is lost on the shared ring.
            IoResult r = file_io_->wait_for_req(req_id, /*timeout_ms=*/-1);
            if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: wait_for_req req_id=%lu status=%d bytes=%d\n",
                                      s_diag_count, (unsigned long) r.req_id, (int) r.status, r.bytes_read);
            read_ok = (r.status == IoStatus::Ok && r.bytes_read == (int) m.size);
        }
        return read_ok;
    };
    bool ok = read_once();
    if (!ok && direct_to_device && !file_io_->direct_to_device()) {
        direct_to_device = false;
        read_dst = staging;
        ok = read_once();
    }
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: stage1 done ok=%d\n", s_diag_count, (int)ok);
    if (!ok) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: file IO failed for page %d\n", page_idx);
        if (owns_slot) pool_.release_slot(slot);
        return -1;
    }

    if (direct_to_device) {
        if (!zero_device_padding(dst, m.size, pool_.slot_size(slot))) {
            if (owns_slot) pool_.release_slot(slot);
            return -1;
        }
        page_to_slot_[page_idx] = slot;
        page_loaded_[page_idx]  = true;
        slot_to_page_[slot]     = page_idx;
        record_page_in_(m.size, seconds_since(io_t0));
        if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT p2p slot=%d\n", s_diag_count, slot);
        ++s_diag_count;
        return slot;
    }

    if (host_tier_) {
        host_tier_->store(page_idx, staging, m.size);
    }

    // Stage 2: H2D + padding zero. stage_in() preserves the synchronous
    // "resident on return" contract even when WP_ASYNC_ENSURE selects a
    // dedicated transport stream for prefetch stage 2.
    int evt = transport_.stage_in(dst, staging, m.size, pool_.slot_size(slot));
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: stage_in returned evt=%d\n", s_diag_count, evt);
    if (evt < 0) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: gpu stage_in failed for page %d\n", page_idx);
        if (owns_slot) pool_.release_slot(slot);
        return -1;
    }
    transport_.release_event(evt);

    // Shared sync_staging_ is owned by the WeightPager; no per-call free.

    page_to_slot_[page_idx] = slot;
    page_loaded_[page_idx]  = true;
    slot_to_page_[slot]     = page_idx;
    record_page_in_(m.size, seconds_since(io_t0));
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT slot=%d\n", s_diag_count, slot);
    ++s_diag_count;
    return slot;
}

}  // namespace wp
