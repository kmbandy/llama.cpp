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
#include <sys/stat.h>   // fstat() -- shard file size, cached alongside the O_DIRECT fd
#include <sys/vfs.h>    // fstatfs() -- filesystem block size, the O_DIRECT alignment authority

#if defined(LLAMA_HAVE_IO_URING) && defined(__linux__)
#include <liburing.h>
#endif

#include "wp-gpu-runtime.h"

#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)

extern "C++" void ggml_cuda_get_routed_expert_ptrs_stats(
    uint64_t * set, uint64_t * consumed, uint64_t * discarded_unconsumed);
#endif

namespace wp {

// ---------------------------------------------------------------------------
// HostBorrowGuard — releases every HostTier zero-copy borrow() taken during
// ensure_batch's HOST path, on every exit from the borrow/sync region.
//
// See docs/superpowers/specs/2026-07-25-hosttier-zerocopy-promotion-design.md
// §4: auditing exits by hand is how this class of bug ships (a borrowed page
// left un-released is stuck un-evictable forever), so ownership of the
// borrowed-page list -- and releasing all of it -- is a single RAII object
// instead. Deliberately NOT copyable/movable: exactly one guard owns exactly
// one borrow list for the duration of one ensure_batch call.
//
// Stores (page_idx, generation handle) pairs, not bare page indices: HostTier
// keys a retired-while-borrowed entry by its own generation (see
// HostTier::release()), so releasing the exact entry THIS borrow() call saw
// -- as opposed to whatever entry currently occupies the same page_idx --
// requires carrying the handle borrow() returned, not just the page_idx.
struct HostBorrowGuard {
    struct Borrowed {
        int                   page;
        HostTier::BorrowHandle handle;
    };

    HostTier *             tier = nullptr;
    std::vector<Borrowed>  pages;

    explicit HostBorrowGuard(HostTier * t) : tier(t) {}
    HostBorrowGuard(const HostBorrowGuard &)             = delete;
    HostBorrowGuard & operator=(const HostBorrowGuard &) = delete;

    ~HostBorrowGuard() {
        if (tier == nullptr) {
            return;
        }
        for (const Borrowed & b : pages) {
            tier->release(b.page, b.handle);
        }
    }
};

void WeightPager::enqueue_tier_promotions_(
        const std::vector<TierPromotionRequest> & requests,
        std::vector<TierPromotion> & queued,
        const TierPromotionEnqueue & enqueue,
        bool transport_events) {
    queued.clear();
    queued.reserve(requests.size());
    if (!host_tier_) return;

    for (const TierPromotionRequest & request : requests) {
        const PageMeta & m = catalog_.at(request.page);
        const void * src = nullptr;
        HostTier::BorrowHandle borrow = HostTier::kInvalidBorrowHandle;
        if (!host_tier_->borrow(request.page, &src, m.size, &borrow)) continue;

        // Do not consume a borrow if no completion event can be acquired. The
        // caller leaves this request on its real-read path, rather than
        // committing a slot whose promotion was silently skipped.
        if (transport_events && transport_.n_free_events() <= 0) {
            ++stats_.tier_promotion_event_pool_exhausted;
            host_tier_->release(request.page, borrow);
            continue;
        }
        int event = -1;
        if (!enqueue(slot_ptr_(request.slot), src, m.size, pool_.slot_size(request.slot), event)) {
            host_tier_->release(request.page, borrow);
            continue;
        }
        queued.push_back({request.page, borrow, event});
        if (transport_events) ++stats_.tier_promotion_async_enqueued;
        else                  ++stats_.tier_promotion_sync_enqueued;
    }
}

bool WeightPager::synchronize_tier_promotions_(const std::vector<TierPromotion> & queued) {
    // The transport stream is ordered, so its final completion event is also
    // the completion fence for every earlier promotion in this batch.
    if (queued.empty() || queued.back().event < 0) return true;
    const auto fence_t0 = std::chrono::steady_clock::now();
    const bool ok = transport_.synchronize(queued.back().event);
    stats_.tier_promotion_fence_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fence_t0).count();
    return ok;
}

void WeightPager::release_tier_promotions_(std::vector<TierPromotion> & queued) {
    for (const TierPromotion & promotion : queued) {
        if (promotion.event >= 0) transport_.release_event(promotion.event);
        if (host_tier_) host_tier_->release(promotion.page, promotion.borrow);
    }
    queued.clear();
}

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
constexpr const char * kEnvWpHostPrefetch = "WP_HOST_PREFETCH";
constexpr const char * kEnvWpHostPrefetchLookahead = "WP_HOST_PREFETCH_LOOKAHEAD";
constexpr const char * kEnvWpHostPrefetchTopM = "WP_HOST_PREFETCH_TOPM";
constexpr const char * kEnvWpHostPrefetchQueue = "WP_HOST_PREFETCH_QUEUE";
constexpr const char * kEnvWpHostPrefetchStrikes = "WP_HOST_PREFETCH_STRIKES";
constexpr const char * kEnvWpHostPrefetchBytes = "WP_HOST_PREFETCH_BYTES";
constexpr const char * kEnvWpHostPrefetchConfStep = "WP_HOST_PREFETCH_CONF_STEP";

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
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
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
    // O_DIRECT bounce buffer: align-down prefix (<= align-1) + size pad to
    // align on the tail (<= align-1), so a read needs up to size + 2*align.
    // Sized against the largest alignment resolved so far across shards
    // (ensure_odirect_align_max_); before any shard has been opened this
    // defaults to 4096, which already covers the common btrfs case. If a
    // later-resolved shard needs a larger alignment, the size grows here on
    // the next call and free_ensure_host_bufs_()+reallocation follows
    // automatically via the size check below -- was hardcoded to the O_DIRECT
    // *device* sector size (512), which under-sized the buffer once the fix
    // in compute_odirect_read_plan()/resolve_odirect_alignment() aligns to
    // the *filesystem* block size (4096 on btrfs) instead.
    const size_t align = ensure_odirect_align_max_ > 0 ? ensure_odirect_align_max_ : (size_t) 4096;
    const size_t need = page_bytes + 2 * align;
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
    ensure_host_bufs_vk_pinned_ = false;

    // Vulkan first: an arena registered with the pool's device lets stage_in copy
    // straight from it, which is both faster than the shared-staging hop AND the
    // precondition for leaving transfers in flight. Only usable if the mapped
    // pointer happens to satisfy O_DIRECT alignment — vkMapMemory promises
    // minMemoryMapAlignment, which is not required to be a filesystem block
    // size, and this project has already been burned once by assuming an
    // alignment rather than checking it. Verify, and fall through if it fails.
    if (transport_.is_vulkan()) {
        const size_t od_align = align;
        bool vk_ok = true;
        for (size_t i = 0; i < cap; ++i) {
            void * p = transport_.host_alloc(alloc);
            if (p == nullptr) {
                LLAMA_LOG_WARN("wp::ensure_host_bufs: vulkan host_alloc failed at buffer %zu/%zu "
                               "(%zu B); falling back to unregistered arena (staging hop retained)\n",
                               i, cap, alloc);
                vk_ok = false;
                break;
            }
            if (((uintptr_t) p % od_align) != 0) {
                LLAMA_LOG_WARN("wp::ensure_host_bufs: vulkan pinned arena at %p is not %zu-aligned; "
                               "falling back to unregistered arena (staging hop retained)\n",
                               p, od_align);
                transport_.host_free(p);
                vk_ok = false;
                break;
            }
            ensure_host_bufs_[i] = p;
        }
        if (vk_ok) {
            ensure_host_bufs_vk_pinned_ = true;
            LLAMA_LOG_WARN("wp::ensure_host_bufs: %zu x %zu B vulkan-registered pinned arena "
                           "(stage_in copies direct, transfers can stay in flight)\n",
                           cap, alloc);
            return true;
        }
        // Release whatever was taken before the failure.
        for (size_t i = 0; i < cap; ++i) {
            if (ensure_host_bufs_[i] != nullptr) {
                transport_.host_free(ensure_host_bufs_[i]);
                ensure_host_bufs_[i] = nullptr;
            }
        }
    }

#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
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
        if (ensure_host_bufs_vk_pinned_) {
            transport_.host_free(p);
            continue;
        }
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
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
    ensure_host_bufs_vk_pinned_ = false;
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
            // Achieved-concurrency: begin() brackets exactly the interval
            // this worker is actually reading (from the first pread() of
            // this job to the point its bytes are fully read or it fails).
            // See EnsureODirectInFlightTracker in wp-pager.h for what
            // peak()/average() mean.
            ensure_odirect_inflight_.begin();
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
            ensure_odirect_inflight_.end();
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
    if (ret >= 0) {
        wp::set_iowq_max_workers(ensure_odirect_ring_, "ensure-odirect");
    }
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

// MAD: alignment authority fix. O_DIRECT alignment must come from the
// FILESYSTEM's block size (statfs f_bsize), not the device's
// logical_block_size -- see the doc comment on the declaration in
// wp-pager.h for the measured 2.49x read-amplification this fixes.
size_t resolve_odirect_alignment(long fs_bsize) {
    constexpr size_t kLogicalBlockFloor = 512;   // known NVMe logical_block_size lower bound
    constexpr size_t kFallbackAlign     = 4096;  // btrfs's actual block size; safe fallback
    if (fs_bsize <= 0) {
        return kFallbackAlign;
    }
    size_t v = (size_t) fs_bsize;
    if (v < kLogicalBlockFloor) {
        v = kLogicalBlockFloor;
    }
    if ((v & (v - 1)) != 0) {
        // Not a power of two after flooring -- implausible statfs result.
        // The alignment math below is a bitmask op and requires pow2; fall
        // back rather than silently mis-aligning every read.
        return kFallbackAlign;
    }
    return v;
}

OdirectReadPlan compute_odirect_read_plan(uint64_t off, size_t size, size_t align, uint64_t file_size) {
    // Defensive: callers are expected to pass an `align` already validated by
    // resolve_odirect_alignment(), but never let a bad value corrupt the
    // mask arithmetic below -- degrade to the safe fallback instead.
    if (align == 0 || (align & (align - 1)) != 0) {
        align = 4096;
    }
    const uint64_t mask   = (uint64_t) align - 1;
    const uint64_t base   = off & ~mask;
    const size_t   prefix = (size_t) (off - base);

    // Bytes actually required to deliver the full payload, unpadded.
    const uint64_t payload_end = base + (uint64_t) prefix + (uint64_t) size;
    // Payload padded out to the next alignment boundary on both ends.
    const uint64_t padded_len = ((uint64_t) prefix + (uint64_t) size + mask) & ~mask;
    uint64_t clamp_end = base + padded_len;

    // Clamp the padded tail at EOF: O_DIRECT returns EIO (not a short read)
    // for a request that runs past the end of the file, and padding to a
    // coarser alignment makes the overrun bigger, not smaller. file_size==0
    // means "unresolved" -- skip clamping rather than truncate to nothing.
    if (file_size != 0 && file_size < clamp_end) {
        clamp_end = file_size;
    }
    // Never clamp below what the payload itself needs delivered, even if
    // file_size turns out to be inconsistent with off+size (that would be a
    // separate, pre-existing data bug -- not something to paper over here).
    if (clamp_end < payload_end) {
        clamp_end = payload_end;
    }
    if (clamp_end < base) {
        clamp_end = base;
    }

    OdirectReadPlan plan;
    plan.base   = base;
    plan.prefix = prefix;
    plan.nbytes = (size_t) (clamp_end - base);
    return plan;
}

int WeightPager::ensure_odirect_fd_(int file_idx) {
    if (file_io_ == nullptr || file_idx < 0) {
        return -1;
    }
    if ((size_t) file_idx >= ensure_odirect_fds_.size()) {
        ensure_odirect_fds_.resize((size_t) file_idx + 1, -1);
        ensure_odirect_align_.resize((size_t) file_idx + 1, 0);
        ensure_odirect_filesize_.resize((size_t) file_idx + 1, 0);
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

    // Resolve the O_DIRECT alignment authority (filesystem block size) and
    // cache the shard's byte size, once, alongside the fd -- both are on the
    // hot read path via ensure_batch and must not be re-queried per read.
    struct statfs sfs;
    const bool statfs_ok = (::fstatfs(od, &sfs) == 0);
    const size_t align = resolve_odirect_alignment(statfs_ok ? (long) sfs.f_bsize : -1);
    ensure_odirect_align_[(size_t) file_idx] = align;
    if (align > ensure_odirect_align_max_) {
        ensure_odirect_align_max_ = align;
    }

    struct stat st;
    ensure_odirect_filesize_[(size_t) file_idx] =
        (::fstat(od, &st) == 0 && st.st_size > 0) ? (uint64_t) st.st_size : 0;

    static int s_align_log = 0;
    if (s_align_log < 8) {  // one line per distinct shard, capped defensively
        LLAMA_LOG_WARN("wp::ensure_odirect_fd_: file_idx=%d path=%s O_DIRECT alignment resolved to "
                       "%zu bytes (fstatfs %s, f_bsize=%ld), shard size=%llu\n",
                       file_idx, path, align, statfs_ok ? "ok" : "FAILED",
                       statfs_ok ? (long) sfs.f_bsize : -1L,
                       (unsigned long long) ensure_odirect_filesize_[(size_t) file_idx]);
        ++s_align_log;
    }
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
    ++stats_.xlayer_predict_calls;
    // Precision-first decaying horizon: full topk only for L+1; deeper layers
    // get a shrinking M so VRAM speculation cannot flood the free-slot pool.
    std::vector<ExpertRef> refs;
    for (int d = 1; d <= xlayer_lookahead_k_; ++d) {
        const int M = (d == 1) ? xlayer_topk_ : std::max(1, xlayer_topk_ >> (d - 1));
        // CONFIDENCE GATE (wired 2026-07-27). This call omitted min_conf, so it
        // defaulted to 0.0f and took the whole top-M no matter how improbable
        // the tail experts were -- the VRAM path "pulled EVERYTHING in". The
        // gate was built and unit-tested on 2026-07-22 but wired only into
        // submit_host_prefetch; this is the same policy applied to the path
        // that actually spends VRAM. Measured cost of not having it (M=2/M=4,
        // laguna 3400 slots): +12-14% bytes for a 2-3.6% hit rate, and M=4
        // scoring WORSE than M=2 because widening M only reaches deeper into
        // low-probability experts.
        const float conf = std::min(
            0.99f, xlayer_min_conf_ + (float) (d - 1) * xlayer_conf_step_);
        predictor_.predict(h, from_layer + d - 1, /*K=*/1, M, n_layer_, refs, conf);
    }
    if (refs.empty()) return;
    std::vector<int> pages;
    for (const ExpertRef & r : refs) {
        expert_sister_pages(r.layer, r.expert, pages);
    }
    if (pages.empty()) return;
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    stats_.xlayer_pred_pages += (uint64_t) pages.size();
    std::vector<int> fresh;
    fresh.reserve(pages.size());
    for (int p : pages) {
        if (p < 0 || p >= catalog_.size()) continue;
        if (catalog_.at(p).is_pinned) continue;
        if (page_to_slot_[p] >= 0) { ++stats_.xlayer_resident_skips; continue; }  // resident/in-flight
        fresh.push_back(p);
    }
    if (fresh.empty()) return;
    // Speculative slot budget: never let speculation exceed the cap.
    if (xlayer_max_slots_ > 0) {
        const int budget = xlayer_max_slots_ - pool_.n_speculative();
        if (budget <= 0) { ++stats_.xlayer_blocked_budget; return; }
        if ((int) fresh.size() > budget) fresh.resize((size_t) budget);
    }
    // Done-but-unreaped slots hold queue capacity that no reservation can free.
    // WP_SPEC_RESERVE was measured to move blocked_free_queue by 2 counts out of
    // 7630 even with the reservation genuinely enforced on both demand paths, so
    // the capacity is not held by demand submissions — it is held by finished
    // work nobody has collected. Harvest first, then test the gate.
    //
    // harvest_ready_prefetches_() and NOT prefetch_.reap_finished(): reaping
    // without committing would release slots whose completed data never reached
    // page_loaded_, marking pages resident with uncommitted contents (see the
    // contract on PrefetchScheduler::reap_finished in wp-prefetch.h).
    if (spec_reap_) {
        const int harvested = harvest_ready_prefetches_();
        ++stats_.xlayer_harvest_calls;
        stats_.xlayer_harvested_pages += (uint64_t) (harvested > 0 ? harvested : 0);
    }
    // Scheduler queue starvation is the other suspect: if the async queue is
    // full (demand traffic), speculative submits get rejected. Count it.
    if (prefetch_.free_queue_slots() <= 0) { ++stats_.xlayer_blocked_free_queue; return; }
    mark_cross_layer_prefetch_candidates(fresh);
    // HARD RULE: speculative VRAM page-ins never LRU-evict the demand working
    // set. Wrong guesses may only consume free slots; thrash dies by construction.
    // Soft (HostTier) path is the place for wider / deeper lookahead.
    prefetch_pages_batch(fresh, /*count_dense_prefetch=*/false,
                         /*allow_evict=*/false, /*speculative=*/true);
}

void WeightPager::submit_host_prefetch(const float * h, int from_layer) {
    if (!initialized_ || !host_prefetcher_ || h == nullptr) return;

    // Soft prefetch into HostTier only. Mispredicts burn RAM/worker time, not
    // VRAM. Decaying horizon: L+1 gets full top-M + base conf; L+2.. get
    // fewer experts and a higher confidence gate. 2-strike holds one-off
    // noise; per-wave byte budget caps ring pressure.
    size_t used_bytes = 0;
    for (int d = 1; d <= host_prefetch_lookahead_; ++d) {
        const int M = (d == 1)
            ? host_prefetch_topm_
            : std::max(1, host_prefetch_topm_ >> (d - 1));
        const float conf = std::min(
            0.99f, host_prefetch_min_conf_ + (float) (d - 1) * host_prefetch_conf_step_);

        std::vector<ExpertRef> refs;
        // from_layer+(d-1) with K=1 predicts exactly layer from_layer+d.
        predictor_.predict(h, from_layer + d - 1, /*K=*/1, M, n_layer_, refs, conf);
        for (const ExpertRef & r : refs) {
            std::vector<int> pages;
            expert_sister_pages(r.layer, r.expert, pages);
            for (int page_idx : pages) {
                if (page_idx < 0 || page_idx >= catalog_.size()) continue;
                if (catalog_.at(page_idx).is_pinned) continue;
                // Already in VRAM / in-flight or already staged in RAM — skip.
                if (page_idx < (int) page_to_slot_.size() && page_to_slot_[page_idx] >= 0) {
                    continue;
                }
                if (host_tier_ && host_tier_->contains(page_idx)) {
                    continue;
                }
                // 2-strike (or N-strike) gate: require repeated high-conf prediction.
                if (page_idx < (int) host_prefetch_strikes_.size()) {
                    uint8_t & st = host_prefetch_strikes_[(size_t) page_idx];
                    if (st < 255) {
                        ++st;
                    }
                    if ((int) st < host_prefetch_strikes_needed_) {
                        ++stats_.host_prefetch_strike_held;
                        continue;
                    }
                }
                const size_t sz = catalog_.at(page_idx).size;
                if (host_prefetch_bytes_budget_ > 0 &&
                    used_bytes + sz > host_prefetch_bytes_budget_) {
                    ++stats_.host_prefetch_budget_trim;
                    continue;
                }
                host_prefetcher_->enqueue(page_idx);
                used_bytes += sz;
            }
        }
    }
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
    pipeline_promotions_enabled_ = wp_pipeline_promotions_enabled();
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
    //    cfg_.block_alignment additionally forces every slot offset to a
    //    multiple of the quant block size. Vulkan needs it: its matmul indexes
    //    the weight buffer as an array of quant blocks (a_offset = .. / QUANT_K),
    //    so a slot that is not block-aligned cannot be addressed at all.
    //    HIP/CUDA leave it at 1 — raw byte pointers, alignment already validated.
    if (cfg_.block_alignment > 1) {
        LLAMA_LOG_INFO("wp::WeightPager::init: forcing slot block alignment %zu B\n",
                       cfg_.block_alignment);
    }

    if (!pool_.init(device_buft, cfg_.n_slots, slot_size, device_idx, cfg_.block_alignment)) {
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
        // Confidence gate on VRAM speculation. Mirrors the host path's
        // MIN_CONF/CONF_STEP pair: a floor on the router's softmax probability
        // before an expert is worth spending VRAM and NVMe bandwidth on, with
        // the floor rising for deeper lookahead (a d=2 guess conditioned on a
        // d=1 guess is strictly less trustworthy). 0.0 = old take-everything
        // behaviour, so the default remains byte-identical.
        if (const char * e = std::getenv("WP_PREFETCH_MIN_CONF"))    { const float v = std::strtof(e,nullptr); if (v >= 0.0f) xlayer_min_conf_  = v; }
        if (const char * e = std::getenv("WP_PREFETCH_CONF_STEP"))   { const float v = std::strtof(e,nullptr); if (v >= 0.0f) xlayer_conf_step_ = v; }
        xlayer_max_slots_ = pool_.n_slots() / 4;
        if (const char * e = std::getenv("WP_PREFETCH_MAX_SLOTS"))   { long v = std::strtol(e,nullptr,10); if (v >= 0) xlayer_max_slots_ = (int) v; }
        if (const char * e = std::getenv("WP_SPEC_RESERVE"))         { long v = std::strtol(e,nullptr,10); if (v >= 0) spec_reserve_ = (int) v; }
        if (const char * e = std::getenv("WP_SPEC_REAP"))            spec_reap_ = (e[0] == '1');
        // WP_PREFETCH_FIX=1 enables the three-part fix as ONE switch, because
        // no part works alone:
        //   1. spec_reap_       -- harvest before the scheduler-queue gate at
        //                          the top of submit_xlayer_prefetch, or
        //                          free_queue_slots() is 0 and nothing submits
        //                          (measured blocked_free_queue 6207-7472).
        //   2. WP_SPEC_BOOTSTRAP -- let speculation seed the tier by eviction,
        //                          or n_free_unpinned() is 0 on a warm pool and
        //                          nothing allocates (bootstrap deadlock).
        //   3. spec_keep_tier_  -- harvest must NOT promote, or the tier never
        //                          accumulates and eviction falls onto demand.
        // Enabling 1+2 without 3 makes speculation evict the demand working
        // set, which is the exact failure the "hard rule" forbids.
        if (const char * e = std::getenv("WP_PREFETCH_FIX")) {
            if (e[0] == '1') {
                spec_reap_       = true;
                spec_keep_tier_  = true;
                setenv("WP_SPEC_BOOTSTRAP", "1", /*overwrite=*/0);
            }
        }
        if (const char * e = std::getenv("WP_SPEC_KEEP_TIER"))       spec_keep_tier_ = (e[0] == '1');
        if (spec_reap_) {
            LLAMA_LOG_WARN("wp::spec reap: harvesting finished prefetches before the speculative gate\n");
        }
        // Never let the reservation swallow the whole queue: demand must always
        // be able to make progress or we deadlock decode instead of accelerating it.
        if (spec_reserve_ > cfg_.prefetch_depth / 2) {
            LLAMA_LOG_WARN("wp::spec reserve %d too large for queue depth %d; clamping to %d\n",
                           spec_reserve_, cfg_.prefetch_depth, cfg_.prefetch_depth / 2);
            spec_reserve_ = cfg_.prefetch_depth / 2;
        }
        if (spec_reserve_ > 0) {
            LLAMA_LOG_WARN("wp::spec reserve: %d of %d queue slots withheld from demand\n",
                           spec_reserve_, cfg_.prefetch_depth);
        }
        if (xlayer_prefetch_enabled_) {
            LLAMA_LOG_WARN("wp::xlayer prefetch: min_conf=%.3f conf_step=%.3f\n",
                           xlayer_min_conf_, xlayer_conf_step_);
            LLAMA_LOG_WARN("wp::xlayer prefetch: on K=%d M=%d cap=%d n_layer=%d\n",
                           xlayer_lookahead_k_, xlayer_topk_, xlayer_max_slots_, n_layer_);
        }
    }

    // 2. Per-device transfer stream + event pool. Size events generously
    //    so prefetch never blocks waiting for an event.
    const int n_transport_events = async_ensure_enabled_
        ? cfg_.prefetch_depth * 2 + 8
        : cfg_.prefetch_depth + 2;
    if (!transport_.init(device_idx, n_transport_events, async_ensure_enabled_, pool_.vram_buf())) {
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
    {
        const char * e = std::getenv("WP_PAGE_HIST");
        page_hist_enabled_ = (e != nullptr && e[0] == '1');
        if (page_hist_enabled_) {
            page_access_.assign((size_t) catalog_.size(), 0u);
            page_pagein_.assign((size_t) catalog_.size(), 0u);
            LLAMA_LOG_WARN("wp::WeightPager: WP_PAGE_HIST=1 — per-page access histogram enabled (%zu pages)\n",
                           (size_t) catalog_.size());
        }
    }
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
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
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
            // The tier must copy out of the pool through the transport, not via
            // a raw device memcpy: on Vulkan a slot pointer is a sentinel base
            // plus an offset and cannot be dereferenced by hip*/cuda*.
            host_tier->set_device_reader(
                [this](void * dst_host, const void * src_device, size_t n) {
                    return transport_.read_to_host(dst_host, src_device, n);
                });
            host_tier_ = std::move(host_tier);
            // WP_HOST_SPEC_TIER=1: evict prefetched-but-unused entries before
            // victim pages. Off by default so the tier stays byte-identical.
            if (env_flag_is_one("WP_HOST_SPEC_TIER")) {
                host_tier_->set_speculative_tier(true);
                LLAMA_LOG_WARN("wp::HostTier: speculative sub-tier ON — mispredicted "
                               "prefetches evict before victim pages\n");
            }
            LLAMA_LOG_WARN("wp::HostTier: RAM victim tier ENABLED, budget %zu B (%.1f MiB), "
                           "backend_pinned=%d, D2H via transport (vulkan-safe)\n",
                           host_budget, (double) host_budget / (1024.0 * 1024.0),
                           (int) host_tier_->backend_pinned());
        } else {
            LLAMA_LOG_WARN("wp::WeightPager: WP_HOST_BUDGET_BYTES=%zu requested, but HostTier init failed; continuing disabled\n",
                           host_budget);
        }
    }

    if (env_flag_is_one(kEnvWpHostPrefetch)) {
        if (!host_tier_) {
            LLAMA_LOG_WARN("wp::host prefetch: WP_HOST_PREFETCH=1 requested but HostTier is disabled; no-op\n");
        } else {
            host_prefetch_lookahead_ = env_nonnegative_int(kEnvWpHostPrefetchLookahead, 2);
            host_prefetch_topm_ = env_nonnegative_int(kEnvWpHostPrefetchTopM, 16);
            const char * mc_env = std::getenv("WP_HOST_PREFETCH_MIN_CONF");
            host_prefetch_min_conf_ = (mc_env && *mc_env) ? std::strtof(mc_env, nullptr) : 0.10f;
            const char * cs_env = std::getenv(kEnvWpHostPrefetchConfStep);
            if (cs_env && *cs_env) {
                host_prefetch_conf_step_ = std::strtof(cs_env, nullptr);
                if (host_prefetch_conf_step_ < 0.0f) host_prefetch_conf_step_ = 0.0f;
            }
            host_prefetch_strikes_needed_ = env_nonnegative_int(kEnvWpHostPrefetchStrikes, 2);
            if (host_prefetch_strikes_needed_ < 1) host_prefetch_strikes_needed_ = 1;
            {
                const char * be = std::getenv(kEnvWpHostPrefetchBytes);
                if (be && *be) {
                    errno = 0;
                    char * end = nullptr;
                    unsigned long long n = std::strtoull(be, &end, 10);
                    if (errno == 0 && end != be && (end == nullptr || *end == '\0')) {
                        host_prefetch_bytes_budget_ = (size_t) n; // 0 = unlimited
                    }
                }
            }
            host_prefetch_strikes_.assign((size_t) catalog_.size(), 0);
            const int queue_depth = env_nonnegative_int(kEnvWpHostPrefetchQueue, 256);
            int max_file_idx = -1;
            for (int i = 0; i < catalog_.size(); ++i) {
                max_file_idx = std::max(max_file_idx, (int) catalog_.at(i).file_idx);
            }
            std::vector<int> prefetch_fds;
            prefetch_fds.reserve((size_t) std::max(0, max_file_idx + 1));
            for (int i = 0; i <= max_file_idx; ++i) {
                const int fd = file_io_->fd(i);
                prefetch_fds.push_back(fd >= 0 ? dup_clear_o_direct(fd) : -1);
            }
            host_prefetch_file_io_ = create_host_file_io(std::move(prefetch_fds),
                                                          /*prefer_async=*/false,
                                                          /*queue_depth=*/1);
            if (!host_prefetch_file_io_) {
                LLAMA_LOG_WARN("wp::host prefetch: dedicated SyncPread layer init failed; no-op\n");
            } else {
                host_prefetcher_ = std::make_unique<HostPrefetcher>(
                    [this](int page_idx, void * dst, size_t capacity) -> int64_t {
                        if (page_idx < 0 || page_idx >= catalog_.size()) return -1;
                        const PageMeta & m = catalog_.at(page_idx);
                        if (m.size > capacity) return -1;
                        const uint64_t req_id = next_host_prefetch_req_id_++;
                        if (!host_prefetch_file_io_->submit(req_id, (int) m.file_idx,
                                                            m.file_offset, m.size, dst)) {
                            return -1;
                        }
                        host_prefetch_file_io_->flush();
                        const IoResult result = host_prefetch_file_io_->wait_for_req(req_id);
                        return result.status == IoStatus::Ok && result.bytes_read == (int) m.size
                            ? result.bytes_read : -1;
                    },
                    [this](int page_idx, const void * bytes, size_t n) {
                        // speculative=true: this is the PREDICTION path. Pages
                        // arriving here are guesses, and must be evicted ahead
                        // of victim pages the GPU actually used. Contrast the
                        // demand-read store at page_in_sync_ and the victim
                        // store_from_device(), both of which stay non-
                        // speculative because they record real use.
                        return host_tier_->store(page_idx, bytes, n, /*speculative=*/true);
                    },
                    [this](int page_idx) {
                        if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return true;
                        // Intentional benign race: page_to_slot_ is written by the eval
                        // thread without locking. A stale read can create a transient
                        // RAM+VRAM duplicate (removed on promotion/eviction) or miss a
                        // prefetch; neither affects correctness or steady-state exclusivity.
                        return page_to_slot_[page_idx] >= 0 || host_tier_->contains(page_idx);
                    },
                    (size_t) queue_depth, catalog_.max_page_size());
                host_prefetcher_->start();
                LLAMA_LOG_WARN(
                    "wp::host prefetch: on K=%d M=%d min_conf=%.3f conf_step=%.3f "
                    "strikes=%d bytes_budget=%zu queue=%d (soft HostTier only; "
                    "VRAM xlayer uses allow_evict=0)\n",
                    host_prefetch_lookahead_, host_prefetch_topm_,
                    host_prefetch_min_conf_, host_prefetch_conf_step_,
                    host_prefetch_strikes_needed_, host_prefetch_bytes_budget_,
                    queue_depth);
            }
        }
    }

    initialized_ = true;
    if (pool_.size_class_slots_enabled()) {
        LLAMA_LOG_WARN("wp::WeightPager: %d pages, %d slots x %zu B budget (%.1f MiB), size_class_slots=1, prefetch_depth=%d, io_uring_depth=%d, sync_staging_pinned=%d, WP_ASYNC_ENSURE=%d, WP_PIPELINE_PROMOTIONS=%d (%s)\n",
                       catalog_.size(), cfg_.n_slots, slot_size,
                       (double) pool_.pool_size() / 1048576.0,
                       cfg_.prefetch_depth, cfg_.io_uring_depth, (int) sync_staging_pinned_,
                       (int) async_ensure_enabled_, (int) pipeline_promotions_enabled_,
                       pipeline_promotions_enabled_ ? "async/batched" : "synchronous");
    } else {
        LLAMA_LOG_WARN("wp::WeightPager: %d pages, %d slots x %zu B (%.1f MiB), prefetch_depth=%d, io_uring_depth=%d, sync_staging_pinned=%d, WP_ASYNC_ENSURE=%d, WP_PIPELINE_PROMOTIONS=%d (%s)\n",
                       catalog_.size(), cfg_.n_slots, slot_size,
                       (double) cfg_.n_slots * (double) slot_size / 1048576.0,
                       cfg_.prefetch_depth, cfg_.io_uring_depth, (int) sync_staging_pinned_,
                       (int) async_ensure_enabled_, (int) pipeline_promotions_enabled_,
                       pipeline_promotions_enabled_ ? "async/batched" : "synchronous");
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

    if (host_prefetcher_) {
        host_prefetcher_->stop();
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
    host_prefetcher_.reset();
    host_prefetch_file_io_.reset();
    file_io_.reset();
    transport_.shutdown();
    host_tier_.reset();
    host_prefetch_strikes_.clear();
    if (sync_staging_ != nullptr) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
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
        // Exclusive victim tier: move the evicted page's bytes to RAM (D2H)
        // before dropping the slot mapping, so a later re-reference is served
        // from Tier 1 instead of re-read from NVMe. The async transfer (if any)
        // was synchronized just above, so the slot contents are valid here.
        // Skip speculative pages (prefetched-but-unused = not real working set).
        if (host_tier_ && page_loaded_[page] && !pool_.is_speculative(slot_idx)) {
            if (host_tier_->store_from_device(page, slot_ptr_(slot_idx),
                                              catalog_.at(page).size)) {
                ++stats_.host_tier_stores;
            } else {
                ++stats_.host_tier_store_fail;
            }
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
    if (host_prefetcher_) {
        stats_.host_prefetch_enqueued = host_prefetcher_->enqueued();
        stats_.host_prefetch_dropped = host_prefetcher_->dropped();
        stats_.host_prefetch_read = host_prefetcher_->read_ok();
        stats_.host_prefetch_read_fail = host_prefetcher_->read_fail();
        stats_.host_prefetch_skipped = host_prefetcher_->skipped();
    }
    if (host_tier_) {
        stats_.host_spec_resident       = (uint64_t) host_tier_->speculative_count();
        stats_.host_spec_evicted_unused = host_tier_->speculative_evicted_unused();
        stats_.host_spec_promotions     = host_tier_->speculative_promotions();
    }
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
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


void WeightPager::log_page_histogram_() const {
    if (!page_hist_enabled_ || page_access_.empty()) return;

    // Pages-per-token is a property of the model, not the run: every token
    // touches n_expert_used * roles pages per MoE layer. Rather than plumb a
    // token counter through eval_cb, derive the denominator from the busiest
    // page -- a page that is genuinely read every step defines 100%.
    uint32_t max_access = 0;
    for (uint32_t a : page_access_) max_access = std::max(max_access, a);
    if (max_access == 0) return;

    struct Row { int page; uint32_t acc; uint32_t pin; };
    std::vector<Row> rows;
    rows.reserve(page_access_.size());
    for (size_t i = 0; i < page_access_.size(); ++i) {
        if (page_access_[i] > 0) {
            rows.push_back({(int) i, page_access_[i],
                            i < page_pagein_.size() ? page_pagein_[i] : 0u});
        }
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row & a, const Row & b) { return a.acc > b.acc; });

    // Bucket by access frequency relative to the busiest page.
    const double thresholds[] = {0.80, 0.50, 0.25, 0.10, 0.01};
    const char * labels[]     = {">=80%", ">=50%", ">=25%", ">=10%", ">=1%"};
    LLAMA_LOG_WARN("wp::PAGE HISTOGRAM: %zu pages touched of %zu, total_accesses=%llu, busiest=%u\n",
                   rows.size(), page_access_.size(),
                   (unsigned long long) page_hist_total_accesses_, max_access);
    for (int t = 0; t < 5; ++t) {
        size_t   n = 0;
        uint64_t pins = 0;
        for (const Row & r : rows) {
            if ((double) r.acc / (double) max_access >= thresholds[t]) { ++n; pins += r.pin; }
        }
        LLAMA_LOG_WARN("  accessed %s of busiest: %zu pages, %llu page_ins\n",
                       labels[t], n, (unsigned long long) pins);
    }
    const size_t topn = std::min<size_t>(rows.size(), 40);
    LLAMA_LOG_WARN("  TOP %zu pages by access (page | acc | %%busiest | page_ins | name):\n", topn);
    for (size_t i = 0; i < topn; ++i) {
        const Row & r = rows[i];
        LLAMA_LOG_WARN("    %6d %8u %6.1f%% %8u  %s\n",
                       r.page, r.acc,
                       100.0 * (double) r.acc / (double) max_access,
                       r.pin, catalog_.at(r.page).tensor_name.c_str());
    }
}

void WeightPager::log_stats_summary() {
    log_page_histogram_();
    if (xlayer_prefetch_enabled_) {
        LLAMA_LOG_WARN("  [xlayer] predict_calls=%lu pred_pages=%lu resident_skips=%lu "
                       "blocked_budget=%lu blocked_free_queue=%lu bootstrap=%lu submitted=%lu hit=%lu spec_evict_unused=%lu n_spec=%d\n",
                       (unsigned long) stats_.xlayer_predict_calls,
                       (unsigned long) stats_.xlayer_pred_pages,
                       (unsigned long) stats_.xlayer_resident_skips,
                       (unsigned long) stats_.xlayer_blocked_budget,
                       (unsigned long) stats_.xlayer_blocked_free_queue,
                       (unsigned long) stats_.xlayer_bootstrap_allocs,
                       (unsigned long) stats_.cross_layer_prefetch_submitted,
                       (unsigned long) stats_.cross_layer_hit_in_ensure,
                       (unsigned long) stats_.speculative_evicted_unused,
                       pool_.n_speculative());
    }
    if (spec_reserve_ > 0) {
        LLAMA_LOG_WARN("  [spec-reserve] slots=%d demand_trimmed=%lu\n",
                       spec_reserve_, (unsigned long) stats_.demand_trimmed_by_reserve);
    }
    if (spec_reap_) {
        LLAMA_LOG_WARN("  [spec-reap] harvest_calls=%lu harvested_pages=%lu\n",
                       (unsigned long) stats_.xlayer_harvest_calls,
                       (unsigned long) stats_.xlayer_harvested_pages);
    }
    // WP_PROFILE_EVAL host-time breakdown (no-op unless the env flag is set).
    weight_pager_eval_cb_print_profile();

    const Stats & s = stats();

    // Transport identity: one unmissable line naming which ensure_batch
    // path(s) actually served reads this run. WP_ENSURE_BATCH_HOST=1
    // bypasses the P2P path entirely and nothing else stated this
    // plainly which transport ran -- exactly the ambiguity that let a
    // long-standing confusion about it survive.
    {
        const int n_transports_used =
            (s.ensure_batch_host_path_batches   > 0 ? 1 : 0) +
            (s.ensure_batch_p2p_path_batches    > 0 ? 1 : 0) +
            (s.ensure_batch_serial_path_batches > 0 ? 1 : 0);
        const char * active = "none (no ensure_batch reads this run)";
        if (n_transports_used > 1) {
            active = "MIXED -- see per-path counts below";
        } else if (s.ensure_batch_host_path_batches > 0) {
            active = "HOST (O_DIRECT pthread pool)";
        } else if (s.ensure_batch_p2p_path_batches > 0) {
            active = "P2P (direct_to_device)";
        } else if (s.ensure_batch_serial_path_batches > 0) {
            active = "SERIAL (sync fallback)";
        }
        LLAMA_LOG_WARN(
            "wp::ensure_batch TRANSPORT: active=%s  host_batches=%lu p2p_batches=%lu serial_batches=%lu\n",
            active,
            (unsigned long) s.ensure_batch_host_path_batches,
            (unsigned long) s.ensure_batch_p2p_path_batches,
            (unsigned long) s.ensure_batch_serial_path_batches);
    }

    // Achieved concurrency on the HOST O_DIRECT path: how many pread()s
    // were genuinely in flight at once, as distinct from
    // ensure_batch_max_n / ensure_batch_avg_n below, which only count how
    // many jobs were QUEUED per batch -- queue occupancy, not reads
    // actually executing. See EnsureODirectInFlightTracker in wp-pager.h
    // for exactly what peak/avg mean. Both read 0 on runs that never used
    // the HOST path (no begin() calls were ever made).
    {
        const int64_t inflight_peak = ensure_odirect_inflight_.peak();
        const double  inflight_avg  = ensure_odirect_inflight_.average();
        LLAMA_LOG_WARN(
            "wp::ensure_batch HOST ACHIEVED CONCURRENCY (reads genuinely in "
            "flight -- NOT queued jobs, do not confuse with ensure_batch_max_n"
            "/ensure_batch_avg_n below): inflight_peak=%lld inflight_avg_at_read_start=%.2f\n",
            (long long) inflight_peak, inflight_avg);
    }
    {
        LLAMA_LOG_WARN(
            "wp::ensure_batch P2P ACHIEVED CONCURRENCY (kernel-submitted reads): "
            "inflight_peak=%lu inflight_avg_at_read_start=%.2f\n",
            (unsigned long) s.ensure_batch_n_sub_sum,
            (unsigned long) s.ensure_batch_window_pressure_fallbacks,
            (unsigned long) s.ensure_batch_p2p_inflight_peak,
            s.ensure_batch_p2p_inflight_avg_at_read_start);
    }

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
            "  batch_slot_exhaustions: %lu\n"
            "  page_ins_ensure_async: %lu\n"
            "  page_ins_ensure_sync: %lu\n"
            "  page_ins_prefetch_reap: %lu\n"
            "  page_ins_sync_direct: %lu\n"
            "  pis_from_ensure: %lu\n"
            "  pis_vk_host: %lu\n"
            "  pis_host_path: %lu\n"
            "  pis_nonhip: %lu\n"
            "  pis_tier_pre: %lu\n"
            "  pis_read_failed: %lu\n"
            "  pis_tier_promo: %lu\n"
            "  pis_serial_batch: %lu\n"
            "  lru_walk_hot_skips: %lu\n"
            "  lru_walk_pinned_skips: %lu\n"
            "  cross_layer_prefetch_submitted: %lu\n"
            "  cross_layer_hit_in_ensure: %lu\n"
            "  speculative_evicted_unused: %lu\n"
            "  host_tier_hits: %lu\n"
            "  host_tier_stores: %lu\n"
            "  host_tier_store_fail: %lu\n"
            "  ensure_batch_host_hits: %lu\n"
            "  ensure_batch_host_odirect_cap_skips: %lu\n"
            "  host_prefetch_enqueued: %lu\n"
            "  host_prefetch_dropped: %lu\n"
            "  host_prefetch_read: %lu\n"
            "  host_prefetch_read_fail: %lu\n"
            "  host_prefetch_skipped: %lu\n"
            "  host_prefetch_strike_held: %lu\n"
            "  host_prefetch_budget_trim: %lu\n"
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
            "  ensure_batch_timeouts: %lu\n"
            "  ensure_batch_host_jobs_ms: %.1f\n"
            "  ensure_batch_host_prep_ms: %.1f\n"
            "  ensure_batch_host_enqueue_ms: %.1f\n"
            "  ensure_batch_host_read_wait_ms: %.1f\n"
            "  ensure_batch_host_h2d_ms: %.1f\n"
            "  ensure_batch_p2p_jobs_ms: %.1f\n"
            "  ensure_batch_p2p_prep_ms: %.1f\n"
            "  ensure_batch_p2p_enqueue_ms: %.1f\n"
            "  ensure_batch_p2p_read_wait_ms: %.1f\n"
            "  ensure_batch_p2p_h2d_ms: %.1f\n"
            "  ensure_batch_p2p_fresh_count: %lu\n"
            "  ensure_batch_n_sub_sum: %lu\n"
            "  ensure_batch_window_pressure_fallbacks: %lu\n"
            "  ensure_batch_p2p_inflight_peak: %lu\n"
            "  ensure_batch_p2p_inflight_avg_at_read_start: %.2f\n"
            "  ensure_batch_host_promotion_count: %lu\n"
            "  ensure_batch_host_zerocopy_promotions: %lu\n"
            "  ensure_batch_host_promotion_h2d_ms: %.1f\n"
            "  ensure_batch_host_fresh_count: %lu\n"
            "  ensure_batch_host_fresh_h2d_ms: %.1f\n"
            "  page_in_sync_promotion_count: %lu\n"
            "  page_in_sync_promotion_h2d_ms: %.1f\n"
            "  page_in_sync_zerocopy_promotions: %lu\n"
            "  page_in_sync_fresh_count: %lu\n"
            "  page_in_sync_fresh_h2d_ms: %.1f\n",
            (unsigned long) s.page_ins,
            (unsigned long) s.evictions,
            (unsigned long) s.prefetch_hits,
            (unsigned long) s.prefetch_misses,
            hit_rate,
            gb_read,
            gbps,
            (unsigned long) s.sync_fallbacks,
            (unsigned long) s.batch_slot_exhaustions,
            (unsigned long) s.page_ins_ensure_async,
            (unsigned long) s.page_ins_ensure_sync,
            (unsigned long) s.page_ins_prefetch_reap,
            (unsigned long) s.page_ins_sync_direct,
            (unsigned long) s.pis_from_ensure,
            (unsigned long) s.pis_vk_host,
            (unsigned long) s.pis_host_path,
            (unsigned long) s.pis_nonhip,
            (unsigned long) s.pis_tier_pre,
            (unsigned long) s.pis_read_failed,
            (unsigned long) s.pis_tier_promo,
            (unsigned long) s.pis_serial_batch,
            (unsigned long) s.lru_walk_hot_skips,
            (unsigned long) s.lru_walk_pinned_skips,
            (unsigned long) s.cross_layer_prefetch_submitted,
            (unsigned long) s.cross_layer_hit_in_ensure,
            (unsigned long) s.speculative_evicted_unused,
            (unsigned long) s.host_tier_hits,
            (unsigned long) s.host_tier_stores,
            (unsigned long) s.host_tier_store_fail,
            (unsigned long) s.ensure_batch_host_hits,
            (unsigned long) s.ensure_batch_host_odirect_cap_skips,
            (unsigned long) s.host_prefetch_enqueued,
            (unsigned long) s.host_prefetch_dropped,
            (unsigned long) s.host_prefetch_read,
            (unsigned long) s.host_prefetch_read_fail,
            (unsigned long) s.host_prefetch_skipped,
            (unsigned long) s.host_prefetch_strike_held,
            (unsigned long) s.host_prefetch_budget_trim,
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
            (unsigned long) s.ensure_batch_timeouts,
            s.ensure_batch_host_jobs_seconds * 1e3,
            s.ensure_batch_host_prep_seconds * 1e3,
            s.ensure_batch_host_enqueue_seconds * 1e3,
            s.ensure_batch_host_read_wait_seconds * 1e3,
            s.ensure_batch_host_h2d_seconds * 1e3,
            s.ensure_batch_p2p_jobs_seconds * 1e3,
            s.ensure_batch_p2p_prep_seconds * 1e3,
            s.ensure_batch_p2p_enqueue_seconds * 1e3,
            s.ensure_batch_p2p_read_wait_seconds * 1e3,
            s.ensure_batch_p2p_h2d_seconds * 1e3,
            (unsigned long) s.ensure_batch_p2p_fresh_count,
            (unsigned long) s.ensure_batch_n_sub_sum,
            (unsigned long) s.ensure_batch_window_pressure_fallbacks,
            (unsigned long) s.ensure_batch_p2p_inflight_peak,
            s.ensure_batch_p2p_inflight_avg_at_read_start,
            (unsigned long) s.ensure_batch_host_promotion_count,
            (unsigned long) s.ensure_batch_host_zerocopy_promotions,
            s.ensure_batch_host_promotion_h2d_seconds * 1e3,
            (unsigned long) s.ensure_batch_host_fresh_count,
            s.ensure_batch_host_fresh_h2d_seconds * 1e3,
            (unsigned long) s.page_in_sync_promotion_count,
            s.page_in_sync_promotion_h2d_seconds * 1e3,
        (unsigned long) s.page_in_sync_zerocopy_promotions,
        (unsigned long) s.page_in_sync_fresh_count,
        s.page_in_sync_fresh_h2d_seconds * 1e3);
        LLAMA_LOG_WARN("  tier_promotion_async_enqueued: %lu\n"
                       "  tier_promotion_sync_enqueued: %lu\n"
                       "  tier_promotion_event_pool_exhausted: %lu\n"
                       "  tier_promotion_fence_ms: %.1f\n",
                       (unsigned long) s.tier_promotion_async_enqueued,
                       (unsigned long) s.tier_promotion_sync_enqueued,
                       (unsigned long) s.tier_promotion_event_pool_exhausted,
                       s.tier_promotion_fence_seconds * 1e3);
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
            "  batch_slot_exhaustions: %lu\n"
            "  page_ins_ensure_async: %lu\n"
            "  page_ins_ensure_sync: %lu\n"
            "  page_ins_prefetch_reap: %lu\n"
            "  page_ins_sync_direct: %lu\n"
            "  pis_from_ensure: %lu\n"
            "  pis_vk_host: %lu\n"
            "  pis_host_path: %lu\n"
            "  pis_nonhip: %lu\n"
            "  pis_tier_pre: %lu\n"
            "  pis_read_failed: %lu\n"
            "  pis_tier_promo: %lu\n"
            "  pis_serial_batch: %lu\n"
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
        "  ensure_batch_timeouts: %lu\n"
        "  ensure_batch_host_jobs_ms: %.1f\n"
        "  ensure_batch_host_prep_ms: %.1f\n"
        "  ensure_batch_host_enqueue_ms: %.1f\n"
        "  ensure_batch_host_read_wait_ms: %.1f\n"
        "  ensure_batch_host_h2d_ms: %.1f\n",
        (unsigned long) s.page_ins,
        (unsigned long) s.evictions,
        (unsigned long) s.prefetch_hits,
        (unsigned long) s.prefetch_misses,
        hit_rate,
        gb_read,
        gbps,
        (unsigned long) s.sync_fallbacks,
            (unsigned long) s.batch_slot_exhaustions,
            (unsigned long) s.page_ins_ensure_async,
            (unsigned long) s.page_ins_ensure_sync,
            (unsigned long) s.page_ins_prefetch_reap,
            (unsigned long) s.page_ins_sync_direct,
            (unsigned long) s.pis_from_ensure,
            (unsigned long) s.pis_vk_host,
            (unsigned long) s.pis_host_path,
            (unsigned long) s.pis_nonhip,
            (unsigned long) s.pis_tier_pre,
            (unsigned long) s.pis_read_failed,
            (unsigned long) s.pis_tier_promo,
            (unsigned long) s.pis_serial_batch,
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
        (unsigned long) s.ensure_batch_timeouts,
        s.ensure_batch_host_jobs_seconds * 1e3,
        s.ensure_batch_host_prep_seconds * 1e3,
        s.ensure_batch_host_enqueue_seconds * 1e3,
        s.ensure_batch_host_read_wait_seconds * 1e3,
        s.ensure_batch_host_h2d_seconds * 1e3);
    LLAMA_LOG_WARN("  tier_promotion_async_enqueued: %lu\n"
                   "  tier_promotion_sync_enqueued: %lu\n"
                   "  tier_promotion_event_pool_exhausted: %lu\n"
                   "  tier_promotion_fence_ms: %.1f\n",
                   (unsigned long) s.tier_promotion_async_enqueued,
                   (unsigned long) s.tier_promotion_sync_enqueued,
                   (unsigned long) s.tier_promotion_event_pool_exhausted,
                   s.tier_promotion_fence_seconds * 1e3);
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
    if (page_hist_enabled_ && page_idx >= 0 && page_idx < (int) page_access_.size()) {
        ++page_access_[(size_t) page_idx];
        ++page_hist_total_accesses_;
    }
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

                if (pool_.is_speculative(slot)) pool_.unpin_slot(slot);  // release in-flight speculative pin
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
                ++stats_.page_ins_ensure_async;
                record_page_in_(m_check.size, seconds);
                page_async_event_[page_idx] = evt;
                return slot_ptr_(slot);
            }
        }
        if (prefetch_.wait_for(page_idx, /*timeout_ms=*/-1)) {
            // Stage 2 done; commit and reap.
            if (pool_.is_speculative(slot)) pool_.unpin_slot(slot);  // release in-flight speculative pin
            page_loaded_[page_idx] = true;
            pool_.mark_used(slot);
            prefetch_.reap(page_idx);
            double seconds = 0.0;
            if (page_idx < (int) prefetch_started_at_.size() &&
                prefetch_started_at_[page_idx] != std::chrono::steady_clock::time_point{}) {
                seconds = seconds_since(prefetch_started_at_[page_idx]);
                prefetch_started_at_[page_idx] = std::chrono::steady_clock::time_point{};
            }
            ++stats_.page_ins_ensure_sync;
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
        if (pool_.is_speculative(slot)) pool_.unpin_slot(slot);  // release in-flight speculative pin (leak guard)
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
    ++stats_.pis_from_ensure;
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
        if (page_hist_enabled_) {
            ++page_access_[(size_t) p];
            ++page_hist_total_accesses_;
        }
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
        if (s < 0) {
            // No free slot for this miss (alloc_slot skips pinned slots, and
            // every earlier miss in THIS batch is already pinned). Falling
            // through with `continue` used to leave out_ptrs[i] NULL with no
            // fallback and no log — and wp-eval-cb.cpp hard-aborts on a NULL
            // for an active expert ("active expert page-in failed"). That is
            // the only NULL-producing path here that logs nothing, which is
            // why the abort looked like a read failure when the read path was
            // never involved.
            //
            // Serve it synchronously instead, exactly like the in-flight
            // prefetch branch above. ensure() re-runs eviction and may well
            // succeed where the batch alloc did not.
            ++stats_.sync_fallbacks;
            void * ptr = ensure(p);
            if (ptr != nullptr) {
                pool_.pin_slot(page_to_slot_[p]);
                out_pinned.push_back(p);
                out_ptrs[i] = ptr;
                continue;
            }
            // Genuinely unservable: the pool cannot hold this batch's working
            // set. Still leaves a NULL, but now it is diagnosed at the source
            // rather than surfacing as an abort a hundred frames later.
            ++stats_.batch_slot_exhaustions;
            LLAMA_LOG_ERROR("[wp::ensure_batch] pool exhausted: page=%d size=%zu "
                            "batch_misses=%zu pinned_this_batch=%zu — no slot and "
                            "sync fallback failed\n",
                            p, (size_t) m.size, misses.size(), out_pinned.size());
            continue;
        }
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
        ++stats_.ensure_batch_host_path_batches;
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
        // Tier-1 (host RAM) lookup partition.
        //
        // Zero-copy promotion (2026-07-25 design): when the tier's arena is
        // HIP-pinned, borrow() hands back a pointer straight into the arena
        // instead of lookup()'s memcpy into the per-miss bounce buffer --
        // legal because the pinned arena is itself a valid hipMemcpyAsync
        // source. Every borrow() taken here is released by host_borrow_guard
        // below, after this whole HOST-path region (including the H2D
        // enqueue and the single hipDeviceSynchronize()) is done with it.
        // Gated on GGML_USE_HIP because backend_pinned() is only ever true
        // there (a CPU-only build's arena is plain malloc, and the #else
        // branch below ignores `jobs`/`host_hit` entirely in favor of
        // page_in_sync_ -- calling borrow() there would leak the refcount
        // with nothing to release it).
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
        const bool host_zerocopy = host_tier_ != nullptr && host_tier_->backend_pinned();
#else
        const bool host_zerocopy = false;
#endif
        std::vector<bool> host_hit(misses.size(), false);
        // Source of the promotion H2D copy for each host_hit[k]: either the
        // borrowed arena pointer (zero-copy) or ensure_host_bufs_[k] (the
        // lookup() bounce buffer), populated as each hit is resolved below.
        std::vector<const void *> host_hit_src(misses.size(), nullptr);
        std::vector<bool> host_hit_zerocopy(misses.size(), false);
        std::vector<HostJob> jobs;
        jobs.reserve(misses.size());
        // The synchronous HOST route retains these borrows through its one
        // hipDeviceSynchronize(), exactly as it did before the promotion
        // helper was introduced. The pipelined route owns its borrows in
        // tier_promotions instead.
        HostBorrowGuard host_borrow_guard(host_tier_.get());
        host_borrow_guard.pages.reserve(misses.size());
        std::vector<TierPromotion> tier_promotions;
        if (host_zerocopy) {
            // Reserve these as promotion candidates now so they are not sent
            // to the O_DIRECT workers. The shared helper borrows and enqueues
            // them below, immediately before the existing H2D timing region.
            for (std::size_t k = 0; k < misses.size(); ++k) {
                if (host_tier_->contains(misses[k].page)) {
                    host_hit[k] = true;
                    host_hit_zerocopy[k] = true;
                }
            }
        }
        int n_od = 0, n_host_hit = 0;
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const PageMeta & m = catalog_.at(misses[k].page);
            if (host_tier_) {
                bool hit = host_hit[k];
                if (!hit && host_tier_->lookup(misses[k].page, ensure_host_bufs_[k], m.size)) {
                    hit = true;
                    host_hit_src[k] = ensure_host_bufs_[k];
                }
                if (hit) {
                    host_hit[k] = true;
                    ++n_host_hit;
                    // fd=-1, queued stays false -> no O_DIRECT read issued.
                    jobs.push_back(HostJob{
                        (int) m.file_idx, -1, m.file_offset, 0, m.size,
                        ensure_host_bufs_[k], ensure_host_buf_bytes_, 0, 0,
                        false, false, false
                    });
                    continue;
                }
            }
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

            // Alignment/file-size were resolved once and cached by
            // ensure_odirect_fd_() when j.fd was opened; look them up by
            // file_idx (parallel to ensure_odirect_fds_) rather than
            // hardcoding 512, which is the DEVICE's logical_block_size and
            // the wrong authority for O_DIRECT alignment -- see
            // resolve_odirect_alignment()/compute_odirect_read_plan().
            const size_t align =
                (j.file_idx >= 0 && (size_t) j.file_idx < ensure_odirect_align_.size() &&
                 ensure_odirect_align_[(size_t) j.file_idx] > 0)
                    ? ensure_odirect_align_[(size_t) j.file_idx]
                    : (size_t) 4096;
            const uint64_t file_size =
                (j.file_idx >= 0 && (size_t) j.file_idx < ensure_odirect_filesize_.size())
                    ? ensure_odirect_filesize_[(size_t) j.file_idx]
                    : 0;
            const OdirectReadPlan plan = compute_odirect_read_plan(j.off, j.size, align, file_size);
            j.base   = plan.base;
            j.prefix = plan.prefix;
            j.nbytes = plan.nbytes;
            if (j.nbytes > j.buf_cap || j.nbytes > UINT_MAX) {
                // Was a silent `continue` that degraded invisibly to the
                // slower page_in_sync_ fallback below. Make it observable:
                // a counter plus a one-shot warning so a future session can
                // see this is happening instead of just seeing lower
                // throughput.
                ++stats_.ensure_batch_host_odirect_cap_skips;
                static int s_cap_skip_warn = 0;
                if (s_cap_skip_warn < 3) {
                    LLAMA_LOG_WARN("wp::ensure_batch: HOST O_DIRECT read for page skipped -- "
                                   "aligned/padded size %zu exceeds bounce buf_cap %zu "
                                   "(align=%zu off=%llu payload=%zu); falling back to "
                                   "page_in_sync_ for this page\n",
                                   j.nbytes, j.buf_cap, align,
                                   (unsigned long long) j.off, j.size);
                    ++s_cap_skip_warn;
                }
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
        const auto tp_reap = std::chrono::steady_clock::now();
        auto msd = [](auto a2, auto b2){ return std::chrono::duration<double,std::milli>(b2-a2).count(); };
        // Named HOST-path phase breakdown (seconds), reported via Stats
        // below. Unlike ensure_batch_submit_seconds/wait_seconds -- which on
        // this path are aliases with different, legacy meanings (see
        // wp-pager.h) -- these four intervals mean exactly what they say.
        const double host_jobs_seconds      = msd(io_t0,     tp_jobs)   / 1e3;
        const double host_prep_seconds      = msd(tp_jobs,   tp_prep)   / 1e3;
        const double host_enqueue_seconds   = msd(tp_prep,   tp_submit) / 1e3;
        const double host_read_wait_seconds = msd(tp_submit, tp_reap)   / 1e3;
        {
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
        // Populated inside the GGML_USE_HIP branch below; declared here
        // (with inert defaults) because the stats-fold block after the
        // #if/#else/#endif is shared by both branches.
        bool   h2d_events_valid = false;
        double promo_h2d_ms     = 0.0;
        double fresh_h2d_ms     = 0.0;
        // Set by the Vulkan route below so the HIP/CUDA route is skipped. Note
        // this file is compiled WITH -DGGML_USE_CUDA even for Vulkan-only runs,
        // so the guard below is not enough on its own — without this flag a
        // Vulkan run would fall into raw cudaMemcpy against the pool sentinel.
        bool   vk_h2d_handled   = false;

#if defined(GGML_USE_VULKAN)
        if (transport_.is_vulkan()) {
            vk_h2d_handled = true;
            // Vulkan H2D for the HOST O_DIRECT batch. The multi-QD read above is
            // backend-neutral and has already landed the payloads in the pinned
            // bounce arena; all that differs here is the copy into the pool,
            // which must go through the transport bridge rather than a device
            // memcpy. Queue every page first, then wait once, so a batch of N
            // pages costs one fence rather than N submit-and-blocks.
            //
            // Deliberately simpler than the HIP route: no three-event device
            // timing and no pipelined tier promotions. Those are optimisations
            // on top of a working copy, and the copy is what was missing.
            std::vector<int> vk_events(misses.size(), -1);
            for (std::size_t k = 0; k < misses.size(); ++k) {
                const Miss     & mm = misses[k];
                const PageMeta & m  = catalog_.at(mm.page);
                void * vram = slot_ptr_(mm.slot);
                if (vram == nullptr) { continue; }

                const void * src = nullptr;
                if (host_hit[k]) {
                    // RAM hit. Borrow the arena region as the copy source and
                    // hold the handle until after the fence, exactly as the
                    // synchronous HIP route does.
                    if (host_hit_zerocopy[k]) {
                        const void * borrowed = nullptr;
                        HostTier::BorrowHandle handle = HostTier::kInvalidBorrowHandle;
                        if (host_tier_ == nullptr ||
                            !host_tier_->borrow(mm.page, &borrowed, m.size, &handle)) {
                            jobs[k].ok  = false;
                            host_hit[k] = false;
                            continue;
                        }
                        host_hit_src[k] = borrowed;
                        host_borrow_guard.pages.push_back({mm.page, handle});
                    }
                    src = host_hit_src[k];
                } else if (jobs[k].ok) {
                    src = (const char *) ensure_host_bufs_[k] + jobs[k].prefix;
                } else {
                    continue;
                }
                if (src == nullptr) { jobs[k].ok = false; host_hit[k] = false; continue; }

                // stage_in_async also zeroes the slot tail, so the HIP route's
                // zero_device_padding() is neither needed nor usable here (it is
                // a hipMemset and the destination is a Vulkan sentinel pointer).
                vk_events[k] = transport_.stage_in_async(
                    vram, src, m.size, pool_.slot_size(mm.slot));
                if (vk_events[k] < 0) {
                    jobs[k].ok  = false;
                    host_hit[k] = false;
                }
            }

            bool vk_all_ok = true;
            for (std::size_t k = 0; k < misses.size(); ++k) {
                if (vk_events[k] < 0) { continue; }
                if (!transport_.synchronize(vk_events[k])) {
                    vk_all_ok   = false;
                    jobs[k].ok  = false;
                    host_hit[k] = false;
                }
                transport_.release_event(vk_events[k]);
            }
            if (!vk_all_ok) {
                LLAMA_LOG_WARN("wp::ensure_batch: vulkan stage fence failed; affected pages routed to sync fallback\n");
            }

            // Commit. Same bookkeeping as the HIP route, minus the padding
            // memset. A page whose stage failed goes down page_in_sync_, which
            // re-reads and re-checks residency.
            for (std::size_t k = 0; k < misses.size(); ++k) {
                const Miss     & mm = misses[k];
                const PageMeta & m  = catalog_.at(mm.page);
                void * vram = slot_ptr_(mm.slot);
                if (vram != nullptr && vk_events[k] >= 0 && (host_hit[k] || jobs[k].ok)) {
                    page_to_slot_[mm.page] = mm.slot;
                    page_loaded_[mm.page]  = true;
                    slot_to_page_[mm.slot] = mm.page;
                    pool_.mark_used(mm.slot);
                    ++batch_ok_n;
                if (page_hist_enabled_ && mm.page >= 0 && mm.page < (int) page_pagein_.size()) {
                    ++page_pagein_[(size_t) mm.page];
                }
                    out_ptrs[mm.out_i] = vram;
                    if (host_hit[k]) {
                        ++stats_.host_tier_hits;
                        ++stats_.ensure_batch_host_hits;
                        ++stats_.ensure_batch_host_promotion_count;
                        if (host_hit_zerocopy[k]) {
                            ++stats_.ensure_batch_host_zerocopy_promotions;
                        }
                        if (host_tier_) { host_tier_->erase(mm.page); }
                    } else {
                        batch_bytes += m.size;
                        ++stats_.ensure_batch_host_fresh_count;
                    }
                    if (mm.page >= 0 && mm.page < (int) host_prefetch_strikes_.size()) {
                        host_prefetch_strikes_[(size_t) mm.page] = 0;
                    }
                } else {
                    ++stats_.pis_vk_host;
                    const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
                    out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
                }
            }
            static int s_vk_host_path_log = 0;
            if (s_vk_host_path_log < 1) {
                LLAMA_LOG_WARN("wp::ensure_batch: HOST path (vulkan) O_DIRECT=%d/%d misses, host_tier_hits=%d first batch\n",
                               n_od, (int) misses.size(), n_host_hit);
                ++s_vk_host_path_log;
            }
        }
#endif

#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
        if (!vk_h2d_handled) {
        // Queue all H2Ds then one device sync (overlap PCIe copies).
        //
        // MAD-P4 follow-up: promotion (HostTier RAM->VRAM) copies are
        // enqueued before fresh-read copies so three timing events can
        // bracket each group and measure its total device H2D time
        // directly (see Stats::ensure_batch_host_promotion_h2d_seconds /
        // ensure_batch_host_fresh_h2d_seconds). This only changes enqueue
        // ORDER -- both groups still land on the same default stream and
        // are still committed by the single hipDeviceSynchronize() below,
        // unchanged. hipEventRecord does not block the host, so no new
        // synchronization point is introduced.
        hipEvent_t h2d_ev_start = nullptr, h2d_ev_mid = nullptr, h2d_ev_end = nullptr;
        const bool h2d_timing_ok =
            hipEventCreate(&h2d_ev_start) == hipSuccess &&
            hipEventCreate(&h2d_ev_mid)   == hipSuccess &&
            hipEventCreate(&h2d_ev_end)   == hipSuccess;
        if (h2d_timing_ok) {
            const hipError_t event_err = hipEventRecord(h2d_ev_start, nullptr);
            if (event_err != hipSuccess) {
                LLAMA_LOG_ERROR("[WP_HIP_DIAG] HOST promotion start hipEventRecord failed: err=%s\n",
                                hipGetErrorString(event_err));
            }
        }
        std::vector<TierPromotionRequest> promotion_requests;
        for (std::size_t k = 0; k < misses.size(); ++k) {
            if (host_hit[k] && host_hit_zerocopy[k]) {
                promotion_requests.push_back({misses[k].page, misses[k].slot});
            }
        }
        size_t sync_promotion_count = 0;
        if (pipeline_promotions_enabled_) {
            enqueue_tier_promotions_(promotion_requests, tier_promotions,
                [](void * dst, const void * src, size_t size, size_t, int & event) {
                    event = -1;
                    return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, nullptr) == hipSuccess;
                }, false);
            for (std::size_t k = 0; k < misses.size(); ++k) {
                if (!host_hit_zerocopy[k]) continue;
                bool queued = false;
                for (const TierPromotion & promotion : tier_promotions) {
                    if (promotion.page == misses[k].page) { queued = true; break; }
                }
                if (!queued) host_hit[k] = false;
            }
        } else {
            // Exact pre-pipeline HOST promotion route: all tier hits enqueue on
            // the default stream before the existing mid event and device fence.
            for (std::size_t k = 0; k < misses.size(); ++k) {
                if (!host_hit[k]) continue;
                const PageMeta & m = catalog_.at(misses[k].page);
                // The helper owns and fills the borrowed source only when
                // pipelining is enabled. Reacquire it here for the synchronous
                // route; otherwise host_hit_src[k] is null and HIP latches
                // hipErrorInvalidValue at this copy.
                if (host_hit_zerocopy[k]) {
                    const void * src = nullptr;
                    HostTier::BorrowHandle handle = HostTier::kInvalidBorrowHandle;
                    if (!host_tier_->borrow(misses[k].page, &src, m.size, &handle)) {
                        jobs[k].ok = false;
                        host_hit[k] = false;
                        continue;
                    }
                    host_hit_src[k] = src;
                    host_borrow_guard.pages.push_back({misses[k].page, handle});
                }
                const hipError_t copy_err = hipMemcpyAsync(
                    slot_ptr_(misses[k].slot), host_hit_src[k], m.size,
                    hipMemcpyHostToDevice, nullptr);
                if (copy_err != hipSuccess) {
                    LLAMA_LOG_ERROR("[WP_HIP_DIAG] HOST promotion hipMemcpyAsync failed: page=%d slot=%d dst=%p src=%p bytes=%zu err=%s\n",
                                    misses[k].page, misses[k].slot, slot_ptr_(misses[k].slot),
                                    host_hit_src[k], m.size, hipGetErrorString(copy_err));
                    jobs[k].ok = false;
                    host_hit[k] = false;
                } else {
                    ++sync_promotion_count;
                }
            }
        }
        auto issue_h2d_copy = [&](std::size_t k) {
            const Miss & mm = misses[k];
            const PageMeta & m = catalog_.at(mm.page);
            void * vram = slot_ptr_(mm.slot);
            if (vram == nullptr) {
                return;
            }
            // A RAM hit's source is the borrowed arena pointer (zero-copy) or
            // the per-miss bounce buffer (lookup() fallback, unpinned arena);
            // O_DIRECT reads begin after their 512-align prefix.
            const void * src = nullptr;
            if (host_hit[k]) {
                src = host_hit_src[k];
            } else if (jobs[k].ok) {
                src = (const char *) ensure_host_bufs_[k] + jobs[k].prefix;
            } else {
                return;
            }
            hipError_t err = hipMemcpyAsync(vram, src, m.size,
                                            hipMemcpyHostToDevice, nullptr);
            if (err != hipSuccess) {
                jobs[k].ok = false;
                host_hit[k] = false;
            }
        };
        size_t n_promo_copy = pipeline_promotions_enabled_ ? tier_promotions.size() : sync_promotion_count;
        size_t n_fresh_copy = 0;
        if (h2d_timing_ok) {
            const hipError_t event_err = hipEventRecord(h2d_ev_mid, nullptr);
            if (event_err != hipSuccess) {
                LLAMA_LOG_ERROR("[WP_HIP_DIAG] HOST promotion mid hipEventRecord failed: err=%s\n",
                                hipGetErrorString(event_err));
            }
        }
        for (std::size_t k = 0; k < misses.size(); ++k) {
            if (!host_hit[k]) {
                issue_h2d_copy(k);
                ++n_fresh_copy;
            }
        }
        if (h2d_timing_ok) {
            const hipError_t event_err = hipEventRecord(h2d_ev_end, nullptr);
            if (event_err != hipSuccess) {
                LLAMA_LOG_ERROR("[WP_HIP_DIAG] HOST promotion end hipEventRecord failed: err=%s\n",
                                hipGetErrorString(event_err));
            }
        }
        // A copy that fails at EXECUTION time (vs enqueue) only surfaces here.
        // If the sync failed, some slot may hold garbage -- do NOT commit any of
        // them; force the whole batch down the sync fallback, which re-reads and
        // re-checks residency correctly. (Guards the wrong-expert-weights class.)
        const hipError_t sync_err = hipDeviceSynchronize();
        const bool sync_ok = (sync_err == hipSuccess);
        if (!sync_ok) {
            LLAMA_LOG_WARN("wp::ensure_batch: hipDeviceSynchronize failed (%s); routing batch to sync fallback\n",
                           hipGetErrorString(sync_err));
        }
        // Direct measurement of each group's H2D device time (see the
        // enqueue split above). Only valid if event creation succeeded and
        // the batch sync itself succeeded (a failed sync leaves the events'
        // completion state meaningless).
        h2d_events_valid = h2d_timing_ok && sync_ok;
        if (h2d_events_valid) {
            float ms = 0.0f;
            if (n_promo_copy > 0 && hipEventElapsedTime(&ms, h2d_ev_start, h2d_ev_mid) == hipSuccess) {
                promo_h2d_ms = (double) ms;
            }
            ms = 0.0f;
            if (n_fresh_copy > 0 && hipEventElapsedTime(&ms, h2d_ev_mid, h2d_ev_end) == hipSuccess) {
                fresh_h2d_ms = (double) ms;
            }
        }
        // Destroy whichever events were actually created, even if creation
        // only partially succeeded (a partial failure must not leak the
        // events that DID get created).
        if (h2d_ev_start) hipEventDestroy(h2d_ev_start);
        if (h2d_ev_mid)   hipEventDestroy(h2d_ev_mid);
        if (h2d_ev_end)   hipEventDestroy(h2d_ev_end);
        // hipDeviceSynchronize above is the observable completion fence for
        // these default-stream promotions. Only now may HostTier recycle the
        // borrowed arena regions.
        release_tier_promotions_(tier_promotions);
        for (std::size_t k = 0; k < misses.size(); ++k) {
            const Miss & mm = misses[k];
            const PageMeta & m = catalog_.at(mm.page);
            void * vram = slot_ptr_(mm.slot);
            const bool ready = sync_ok && (host_hit[k] || jobs[k].ok);
            if (ready && vram != nullptr &&
                zero_device_padding(vram, m.size, pool_.slot_size(mm.slot))) {
                page_to_slot_[mm.page] = mm.slot;
                page_loaded_[mm.page]  = true;
                slot_to_page_[mm.slot] = mm.page;
                pool_.mark_used(mm.slot);
                ++batch_ok_n;
                if (page_hist_enabled_ && mm.page >= 0 && mm.page < (int) page_pagein_.size()) {
                    ++page_pagein_[(size_t) mm.page];
                }
                out_ptrs[mm.out_i] = vram;
                if (host_hit[k]) {
                    ++stats_.host_tier_hits;   // served from RAM; no NVMe bytes
                    ++stats_.ensure_batch_host_hits;
                    ++stats_.ensure_batch_host_promotion_count;
                    if (host_hit_zerocopy[k]) {
                        // Sourced the H2D straight from the pinned HostTier
                        // arena (borrow()), not lookup()'s bounce-buffer
                        // memcpy -- see the 2026-07-25 zero-copy design.
                        ++stats_.ensure_batch_host_zerocopy_promotions;
                    }
                    // Exclusive tier: the page is back in VRAM, so drop its RAM
                    // copy. It re-enters Tier 1 only if evicted again.
                    if (host_tier_) {
                        host_tier_->erase(mm.page);
                    }
                    if (mm.page >= 0 && mm.page < (int) host_prefetch_strikes_.size()) {
                        host_prefetch_strikes_[(size_t) mm.page] = 0;
                    }
                } else {
                    batch_bytes += m.size;     // real storage read
                    ++stats_.ensure_batch_host_fresh_count;
                    // Do NOT populate Tier 1 on read: a page resident in VRAM must
                    // not also sit in RAM (that made RAM a useless duplicate of the
                    // pool). Fresh reads are VRAM-only; a page enters Tier 1 as a
                    // victim via on_pool_evict_ (D2H on eviction).
                    if (mm.page >= 0 && mm.page < (int) host_prefetch_strikes_.size()) {
                        host_prefetch_strikes_[(size_t) mm.page] = 0;
                    }
                }
            } else {
                ++stats_.pis_host_path;
                const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
                out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
            }
        }
        static int s_host_path_log = 0;
        if (s_host_path_log < 1) {
            LLAMA_LOG_WARN("wp::ensure_batch: HOST path O_DIRECT=%d/%d misses, host_tier_hits=%d first batch\n",
                           n_od, (int) misses.size(), n_host_hit);
            ++s_host_path_log;
        }
        }   // end if (!vk_h2d_handled)
#else
        (void) n_od;
        if (!vk_h2d_handled) {
            for (std::size_t k = 0; k < misses.size(); ++k) {
                ++stats_.pis_nonhip;
                const int s = page_in_sync_(misses[k].page, /*reuse_slot=*/misses[k].slot);
                out_ptrs[misses[k].out_i] = (s < 0) ? nullptr : slot_ptr_(s);
            }
        }
#endif
        (void) vk_h2d_handled;
        const double batch_seconds = seconds_since(io_t0);
        if (batch_ok_n > 0) {
            stats_.page_ins  += (uint64_t) batch_ok_n;
            stats_.io_bytes  += (uint64_t) batch_bytes;
            stats_.io_seconds += batch_seconds;
            ++stats_.ensure_batch_calls;
            stats_.ensure_batch_pages   += (uint64_t) batch_ok_n;
            stats_.ensure_batch_bytes   += (uint64_t) batch_bytes;
            // Headline gb/s denominator: storage-read-phase time only
            // (excludes H2D), and only when real storage bytes were
            // actually read -- a batch served entirely by HostTier
            // reads zero storage bytes and must not inflate the wall
            // time this ratio is divided by.
            if (batch_bytes > 0) {
                stats_.ensure_batch_seconds += read_seconds;
            }
            stats_.ensure_batch_submit_seconds += read_seconds; // host O_DIRECT phase
            stats_.ensure_batch_wait_seconds   += (batch_seconds - read_seconds); // H2D
            stats_.ensure_batch_host_jobs_seconds      += host_jobs_seconds;
            stats_.ensure_batch_host_prep_seconds      += host_prep_seconds;
            stats_.ensure_batch_host_enqueue_seconds   += host_enqueue_seconds;
            stats_.ensure_batch_host_read_wait_seconds += host_read_wait_seconds;
            stats_.ensure_batch_host_h2d_seconds       += (batch_seconds - read_seconds);
            if (h2d_events_valid) {
                stats_.ensure_batch_host_promotion_h2d_seconds += promo_h2d_ms / 1e3;
                stats_.ensure_batch_host_fresh_h2d_seconds     += fresh_h2d_ms / 1e3;
            }
            // Real storage submissions only: n_submitted counts jobs
            // actually enqueued to the O_DIRECT worker pool. HostTier
            // hits never reach the queue (their HostJob has fd=-1 and
            // queued stays false) and must not inflate this count.
            stats_.ensure_batch_n_sub_sum      += (uint64_t) n_submitted;
            if ((uint64_t) n_submitted > stats_.ensure_batch_max_n) {
                stats_.ensure_batch_max_n = (uint64_t) n_submitted;
            }
        }
        return;
    }

    // Pass 2 — issue all cold-miss reads. On the P2P/direct-to-device path the
    // reads land straight in the VRAM slots, so one io_uring batch keeps N reads
    // in flight at once (true QD=N). Off P2P the shared staging buffer can't
    // hold N pages, so fall back to serial sync into each pinned slot (still
    // correct; the throughput case that matters for decode is P2P).
    //
    // HostTier (soft prefetch / victim tier) is consulted FIRST on the P2P
    // path so predictor-filled RAM pages become H2D hits instead of NVMe
    // re-reads. Previously P2P bypassed HostTier entirely — soft prefetch
    // could never pay off under the default decode transport.
    if (file_io_->direct_to_device()) {
        ++stats_.ensure_batch_p2p_path_batches;
        const auto p2p_jobs_t0 = std::chrono::steady_clock::now();
        std::vector<Miss> cold;
        std::vector<TierPromotionRequest> promotion_requests;
        cold.reserve(misses.size());
        promotion_requests.reserve(misses.size());
        if (!pipeline_promotions_enabled_) {
            // This is the pre-a563629a5 P2P route: each HostTier hit completes
            // synchronously before the cold-read batch is submitted.
            int n_host_hit = 0;
            for (const Miss & mm : misses) {
                if (host_tier_ && host_tier_->contains(mm.page)) {
                    ++stats_.pis_tier_pre;
                    const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
                    out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
                    if (s >= 0) {
                        ++n_host_hit;
                        ++stats_.ensure_batch_host_hits;
                        if (mm.page >= 0 && mm.page < (int) host_prefetch_strikes_.size()) {
                            host_prefetch_strikes_[(size_t) mm.page] = 0;
                        }
                    } else {
                        cold.push_back(mm);
                    }
                } else {
                    cold.push_back(mm);
                }
            }
            if (cold.empty()) {
                if (n_host_hit > 0) {
                    ++stats_.ensure_batch_calls;
                    stats_.ensure_batch_pages += (uint64_t) n_host_hit;
                }
                return;
            }
        } else {
            for (const Miss & mm : misses) {
                if (host_tier_ && host_tier_->contains(mm.page)) {
                    promotion_requests.push_back({mm.page, mm.slot});
                } else {
                    cold.push_back(mm);
                }
            }
        }

        std::vector<FileIOBatchRequest> reqs;
        std::vector<uint64_t>           req_ids;
        reqs.reserve(cold.size());
        req_ids.reserve(cold.size());
        // Wall-clock the whole multi-QD burst (submit → last wait), then
        // attribute that one interval once. Recording per-page after waits
        // (old) under-counted concurrent P2P BW (~0.4 vs real multi-GB/s).
        const auto io_t0 = std::chrono::steady_clock::now();
        for (std::size_t k = 0; k < cold.size(); ++k) {
            const PageMeta & m = catalog_.at(cold[k].page);
            const uint64_t rid = next_io_req_id_++;   // high-bit-tagged; disjoint from prefetch
            req_ids.push_back(rid);
            reqs.push_back({ rid, (int) m.file_idx, m.file_offset,
                             m.size, slot_ptr_(cold[k].slot) });
        }
        const auto p2p_enqueue_t0 = std::chrono::steady_clock::now();
        const double p2p_jobs_seconds = seconds_since(p2p_jobs_t0);
        const int n_sub = reqs.empty() ? 0 : file_io_->submit_batch(reqs);
        if (n_sub > 0) file_io_->flush();
        // Submit NVMe first, then queue RAM->VRAM on the independent transport
        // stream while the read batch is in flight. Failed event acquisition is
        // intentionally left out of `promotions`, and is read synchronously
        // after the batch rather than being silently committed.
        std::vector<TierPromotion> promotions;
        enqueue_tier_promotions_(promotion_requests, promotions,
            [this](void * dst, const void * src, size_t size, size_t slot_size, int & event) {
                event = transport_.stage_in_async(dst, src, size, slot_size);
                return event >= 0;
            }, true);
        std::vector<int> promoted_pages;
        promoted_pages.reserve(promotions.size());
        for (const TierPromotion & promotion : promotions) promoted_pages.push_back(promotion.page);
        const double submit_seconds = seconds_since(io_t0);
        const double p2p_enqueue_seconds = seconds_since(p2p_enqueue_t0);
        const auto wait_t0 = std::chrono::steady_clock::now();
        std::vector<bool> ok(cold.size(), false);
        // Wait per req via demuxing wait_for_req (foreign CQEs buffered in
        // ready_). A multi-id wait_for_reqs was tried; it could busy-spin
        // at load (97% CPU, never reach "model loaded") when ensure_batch
        // fires during graph init. Ordered waits keep multi-QD overlap —
        // I/O is already in flight from submit_batch; we only reap.
        uint64_t n_timeout = 0;
        for (int k = 0; k < n_sub; ++k) {
            IoResult r = file_io_->wait_for_req(req_ids[(size_t) k], /*timeout_ms=*/-1);
            ok[(size_t) k] = (r.status == IoStatus::Ok &&
                              r.bytes_read == (int) catalog_.at(cold[(size_t) k].page).size);
            if (!ok[(size_t) k]) {
                ++n_timeout;
            }
        }
        const bool promotions_ok = synchronize_tier_promotions_(promotions);
        // Completion is observed before this release: the HostTier arena must
        // remain immutable through the event fence, not merely through enqueue.
        release_tier_promotions_(promotions);
        const double wait_seconds = seconds_since(wait_t0);
        const double batch_seconds = seconds_since(io_t0);
        size_t batch_bytes = 0;
        int    batch_ok_n  = 0;
        for (std::size_t k = 0; k < cold.size(); ++k) {
            const Miss & mm = cold[k];
            const PageMeta & m = catalog_.at(mm.page);
            if (ok[k] && zero_device_padding(slot_ptr_(mm.slot), m.size, pool_.slot_size(mm.slot))) {
                page_to_slot_[mm.page] = mm.slot;
                page_loaded_[mm.page]  = true;
                slot_to_page_[mm.slot] = mm.page;
                pool_.mark_used(mm.slot);
                batch_bytes += m.size;
                ++batch_ok_n;
                if (page_hist_enabled_ && mm.page >= 0 && mm.page < (int) page_pagein_.size()) {
                    ++page_pagein_[(size_t) mm.page];
                }
                out_ptrs[mm.out_i] = slot_ptr_(mm.slot);
                if (mm.page >= 0 && mm.page < (int) host_prefetch_strikes_.size()) {
                    host_prefetch_strikes_[(size_t) mm.page] = 0;
                }
            } else {
                // read (or padding) failed — sync-fallback into the SAME pinned
                // slot so the up-front pin/out_pinned bookkeeping stays valid.
                ++stats_.pis_read_failed;
                const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
                out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
            }
        }
        int n_host_hit = 0;
        for (const TierPromotionRequest & request : promotion_requests) {
            const bool promoted = promotions_ok &&
                std::find(promoted_pages.begin(), promoted_pages.end(), request.page) != promoted_pages.end();
            if (promoted) {
                page_to_slot_[request.page] = request.slot;
                page_loaded_[request.page] = true;
                slot_to_page_[request.slot] = request.page;
                pool_.mark_used(request.slot);
                ++n_host_hit;
                ++stats_.host_tier_hits;
                ++stats_.ensure_batch_host_hits;
                ++stats_.page_in_sync_promotion_count;
                host_tier_->erase(request.page);
                for (const Miss & mm : misses) {
                    if (mm.page == request.page) out_ptrs[mm.out_i] = slot_ptr_(request.slot);
                }
            } else {
                // The promotion could not obtain an event (or its completion
                // fence failed). Retire this tier copy before the serial path
                // so page_in_sync_ performs a genuine storage read.
                host_tier_->erase(request.page);
                ++stats_.pis_tier_promo;
                const int s = page_in_sync_(request.page, request.slot);
                for (const Miss & mm : misses) {
                    if (mm.page == request.page) out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
                }
            }
        }
        const int total_ok = batch_ok_n + n_host_hit;
        if (total_ok > 0 || n_sub > 0) {
            // One wall interval for the concurrent burst; page_ins += N.
            // (host hits already counted page_ins inside page_in_sync_)
            stats_.page_ins  += (uint64_t) batch_ok_n;
            stats_.io_bytes  += (uint64_t) batch_bytes;
            stats_.io_seconds += batch_seconds;
            ++stats_.ensure_batch_calls;
            stats_.ensure_batch_pages   += (uint64_t) total_ok;
            stats_.ensure_batch_bytes   += (uint64_t) batch_bytes;
            // Headline gb/s denominator: storage-read-phase time only
            // (submit+wait; no separate H2D stage on this path), and
            // only when real storage bytes were actually read.
            if (batch_bytes > 0) {
                stats_.ensure_batch_seconds += batch_seconds;
            }
            stats_.ensure_batch_submit_seconds += submit_seconds;
            stats_.ensure_batch_wait_seconds   += wait_seconds;
            stats_.ensure_batch_timeouts       += n_timeout;
            stats_.ensure_batch_p2p_jobs_seconds      += p2p_jobs_seconds;
            // P2P has no O_DIRECT alignment/bounce preparation phase.
            stats_.ensure_batch_p2p_enqueue_seconds   += p2p_enqueue_seconds;
            stats_.ensure_batch_p2p_read_wait_seconds += wait_seconds;
            stats_.ensure_batch_p2p_fresh_count       += (uint64_t) batch_ok_n;
            const FileIOConcurrency p2p_concurrency = file_io_->concurrency();
            if (p2p_concurrency.peak > stats_.ensure_batch_p2p_inflight_peak) {
                stats_.ensure_batch_p2p_inflight_peak = p2p_concurrency.peak;
            }
            stats_.ensure_batch_p2p_inflight_avg_at_read_start =
                p2p_concurrency.average_at_start;
            stats_.ensure_batch_window_pressure_fallbacks =
                p2p_concurrency.window_pressure_fallbacks;
            // Real storage submissions only: n_sub is submit_batch's
            // return value (jobs actually queued to io_uring). n_host_hit
            // came from HostTier and never reached submit_batch, so it
            // must not inflate the submission/concurrency counters.
            stats_.ensure_batch_n_sub_sum      += (uint64_t) n_sub;
            if ((uint64_t) n_sub > stats_.ensure_batch_max_n) {
                stats_.ensure_batch_max_n = (uint64_t) n_sub;
            }
        }
    } else {
        ++stats_.ensure_batch_serial_path_batches;
        for (const Miss & mm : misses) {
            ++stats_.pis_serial_batch;
            const int s = page_in_sync_(mm.page, /*reuse_slot=*/mm.slot);
            out_ptrs[mm.out_i] = (s < 0) ? nullptr : slot_ptr_(s);
            if (s >= 0 && mm.page >= 0 && mm.page < (int) host_prefetch_strikes_.size()) {
                host_prefetch_strikes_[(size_t) mm.page] = 0;
            }
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
    //
    // Also honour the speculative reservation. This is a DEMAND path -- the
    // speculative submitter (submit_xlayer_prefetch) goes through
    // prefetch_pages_batch, never here -- so it must leave spec_reserve_ slots
    // for speculation. Measured 2026-07-19: gating only prefetch_pages_batch
    // left blocked_free_queue bit-identical (7630) because demand drained the
    // pool through THIS path instead.
    if (prefetch_.free_queue_slots() <= spec_reserve_) {
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
    int free_q = prefetch_.free_queue_slots();
    // Withhold the speculative reservation from DEMAND. Measured 2026-07-19:
    // demand drains the shared pool to zero regardless of its size, so
    // speculative submits are rejected on 94% of calls (blocked_free_queue
    // 8119/8643, identical at depth 16/64/128). Capacity was never the
    // constraint; contention for one pool was.
    if (!speculative && spec_reserve_ > 0) {
        const int avail = free_q - spec_reserve_;
        if (avail < (int) needed.size()) {
            stats_.demand_trimmed_by_reserve += (uint64_t) ((int) needed.size() - std::max(avail, 0));
        }
        free_q = avail;
    }
    if (free_q <= 0) {
        return false;
    }
    if ((int) needed.size() > free_q) {
        needed.resize((size_t) free_q);
    }
    // Sample oracle: also cap to free pool slots so we never LRU-evict.
    //
    // BOOTSTRAP DEADLOCK (diagnosed 2026-07-27, WP_SPEC_BOOTSTRAP=1 lifts it).
    // n_free_unpinned() counts slots that are !used_ && unpinned — genuinely
    // UNUSED. On a warm pool that is permanently 0, so this returned false
    // before submitting anything and speculation never allocated. Since
    // set_speculative() is only ever called AFTER a successful allocation
    // here, no slot could ever become speculative, so pool_.n_speculative()
    // stayed 0 forever, so the xlayer_max_slots_ budget never bound
    // (blocked_budget == 0), and the pool's speculative-first eviction tier
    // — built precisely to make this safe — was unreachable dead code.
    //
    // That single gate is upstream of every prefetch knob measured "dead":
    // WP_PREFETCH_MAX_SLOTS (blocked_budget=0), WP_SPEC_RESERVE (reserves
    // QUEUE slots, not pool slots), WP_PREFETCH_DEPTH and the io-wq cap
    // (queue depth is irrelevant behind a pool gate). It is also why
    // WP_SPEC_REAP appeared to "unblock submission" (18 -> 41,778): harvesting
    // commits finished prefetches and RELEASES their slots, manufacturing the
    // free slots speculation could not otherwise obtain — bootstrapping the
    // tier as a side effect, and racing harvest against use while doing it.
    //
    // The safety the original rule wanted already exists one layer down:
    // alloc_slot() evicts the LRU SPECULATIVE slot before touching the
    // pinned/hot working set, and mark_used() promotes a speculative slot to
    // non-speculative the moment demand actually hits it. So once the tier is
    // seeded it recycles within itself. Bounding the seed to xlayer_max_slots_
    // (default n_slots/4) caps how much demand it may ever displace.
    if (!allow_evict) {
        static const int s_spec_bootstrap = []() {
            const char * e = std::getenv("WP_SPEC_BOOTSTRAP");
            return (e != nullptr && e[0] == '1') ? 1 : 0;
        }();
        const int free_pool = pool_.n_free_unpinned();
        const bool may_seed = s_spec_bootstrap && speculative &&
                              xlayer_max_slots_ > 0 &&
                              pool_.n_speculative() < xlayer_max_slots_;
        if (free_pool <= 0 && !may_seed) {
            return false;
        }
        if (may_seed) {
            // Room left under the speculative cap; allow alloc_slot's
            // speculative-first LRU to place these.
            const int headroom = xlayer_max_slots_ - pool_.n_speculative();
            if ((int) needed.size() > headroom) {
                needed.resize((size_t) headroom);
            }
            allow_evict = true;
            stats_.xlayer_bootstrap_allocs += (uint64_t) needed.size();
        } else if ((int) needed.size() > free_pool) {
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
                // Drop the intra-batch pin taken below; release_slot does not.
                if (speculative) pool_.unpin_slot(prev);
                pool_.release_slot(prev);
            }
            return false;
        }
        ensure_slot_map_(s);
        if (speculative) {
            // GATE 4 / INTRA-BATCH SELF-CANNIBALISATION (diagnosed 2026-07-27).
            // set_speculative() makes this slot a legal victim for the NEXT
            // iteration's alloc_slot(): Pass 0 of alloc_slot_fixed_ recycles the
            // LRU slot that is (speculative && pin_count_ == 0), and a slot we
            // just marked is exactly that -- on the seeding batch it is the ONLY
            // speculative slot, hence trivially the LRU of its cohort. The batch
            // then carries the same slot twice, slot_to_page_ keeps only the last
            // writer, and both reads DMA into one buffer: the first page is
            // silently mapped to the second page's weights. Wrong experts, no
            // crash -- the corruption behind the 128-token unprintable output.
            //
            // The pin used to be taken after submit_batch (which correctly
            // protects the in-flight read ACROSS calls); the hole was the window
            // between marking and pinning WITHIN one call. Pin on allocation so
            // Pass 0's pin_count_ test excludes it from the moment it is marked.
            // Unreachable before af1778211: speculative used alloc_slot_no_evict,
            // which never runs Pass 0. Bootstrap's allow_evict=true opened it.
            pool_.pin_slot(s);
            pool_.set_speculative(s, true);
        }
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
            // Drop the intra-batch pin taken in the alloc loop above.
            if (speculative) pool_.unpin_slot(slots[i]);
            pool_.release_slot(slots[i]);
        }
        return false;
    }
    // MAD-231 extension: each speculative slot is pinned for its in-flight
    // lifetime. The async read (p2p direct-to-device) lands straight in this VRAM
    // slot; without the pin, alloc_slot's Pass-0 "recycle LRU speculative" (or LRU
    // eviction) could hand the slot to another page before the read completes,
    // and the late read would corrupt the new owner. Mirrors ensure_batch's
    // in-flight miss pin. Released at each commit/teardown site below, keyed on
    // is_speculative (still set there, before mark_used()/release_slot() clears it).
    //
    // The pin is now taken in the allocation loop above rather than here, so it
    // also covers the intra-batch window -- see the gate-4 note there. Exactly
    // one pin per slot either way; the release sites are unchanged.
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
            // Release the in-flight speculative pin (see prefetch_pages_batch)
            // before mark_used clears the speculative flag. Read has landed, so
            // the slot is now safe to recycle. No-op for demand pages.
            const bool was_spec = pool_.is_speculative(page_to_slot_[p]);
            if (was_spec) pool_.unpin_slot(page_to_slot_[p]);
            page_loaded_[p] = true;
            // A landed prefetch is NOT a demand hit. mark_used() clears
            // speculative_, so calling it here promoted every prefetched page
            // to the hot working set the instant its read completed — the
            // speculative tier could never accumulate, pool_.n_speculative()
            // stayed ~0, the xlayer_max_slots_ budget could not bind, and
            // alloc_slot's speculative-first eviction fell straight through
            // onto the demand set. touch_lru bumps LRU without promoting, so
            // an unwanted prediction stays first in line to be evicted and a
            // genuinely useful one is promoted later by the demand path's own
            // mark_used(). Gated so the default remains byte-identical.
            if (was_spec && spec_keep_tier_) {
                pool_.touch_lru(page_to_slot_[p]);
            } else {
                pool_.mark_used(page_to_slot_[p]);
            }
            double seconds = 0.0;
            if (p < (int) prefetch_started_at_.size() &&
                prefetch_started_at_[p] != std::chrono::steady_clock::time_point{}) {
                seconds = seconds_since(prefetch_started_at_[p]);
                prefetch_started_at_[p] = std::chrono::steady_clock::time_point{};
            }
            ++stats_.page_ins_prefetch_reap;
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
            // release_slot does NOT clear pin_count_, so drop the in-flight
            // speculative pin first or the freed slot stays pinned (leak).
            if (pool_.is_speculative(slot)) pool_.unpin_slot(slot);
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
            // release_slot does NOT clear pin_count_; drop the speculative pin
            // first (leak guard).
            if (pool_.is_speculative(slot)) pool_.unpin_slot(slot);
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
    if (page_hist_enabled_ && page_idx >= 0 && page_idx < (int) page_pagein_.size()) {
        ++page_pagein_[(size_t) page_idx];
    }
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
        const void * tier_src = staging;
        bool zerocopy = false;
        bool tier_hit = false;
        HostBorrowGuard page_borrow_guard(host_tier_.get());
        if (host_tier_->backend_pinned()) {
            HostTier::BorrowHandle handle = HostTier::kInvalidBorrowHandle;
            const void * borrowed = nullptr;
            if (host_tier_->borrow(page_idx, &borrowed, m.size, &handle)) {
                tier_src = borrowed;
                zerocopy = true;
                tier_hit = true;
                page_borrow_guard.pages.push_back({page_idx, handle});
            }
        } else if (host_tier_->lookup(page_idx, staging, m.size)) {
            tier_src = staging;
            tier_hit = true;
        }
        if (tier_hit) {
            const auto stage_t0 = std::chrono::steady_clock::now();
            int evt = -1;
            if (!pipeline_promotions_enabled_) {
                // Pre-pipeline route: stage_in synchronizes and returns the
                // completion event, preserving the original promo_ms meaning.
                evt = transport_.stage_in(dst, tier_src, m.size, pool_.slot_size(slot));
                if (evt >= 0) transport_.release_event(evt);
            } else {
                std::vector<TierPromotion> promotions;
                std::vector<TierPromotionRequest> requests = {{page_idx, slot}};
                enqueue_tier_promotions_(requests, promotions,
                    [this](void * out, const void * in, size_t size, size_t slot_size, int & event) {
                        event = transport_.stage_in_async(out, in, size, slot_size);
                        return event >= 0;
                    }, true);
                const bool promotion_ok = promotions.size() == 1 && synchronize_tier_promotions_(promotions);
                release_tier_promotions_(promotions);
                evt = promotion_ok ? 0 : -1;
            }
            const double stage_seconds = seconds_since(stage_t0);
            if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: host tier stage_in returned evt=%d\n", s_diag_count, evt);
            if (evt < 0) {
                LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: host tier gpu stage_in failed for page %d\n",
                               page_idx);
                // An async event failure must not drop a page. Retire the
                // tier candidate and take the ordinary storage-read path.
                host_tier_->erase(page_idx);
                goto read_from_storage;
            }
            page_to_slot_[page_idx] = slot;
            page_loaded_[page_idx]  = true;
            slot_to_page_[slot]     = page_idx;
            ++stats_.host_tier_hits;
            ++stats_.page_ins;
            ++stats_.page_in_sync_promotion_count;
            if (zerocopy) {
                ++stats_.page_in_sync_zerocopy_promotions;
            }
            stats_.page_in_sync_promotion_h2d_seconds += stage_seconds;
            host_tier_->erase(page_idx);
            if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT host-tier slot=%d\n", s_diag_count, slot);
            ++s_diag_count;
            return slot;
        }
    }

read_from_storage:
    // Stage 1: blocking read via the file IO layer. P2P reads directly
    // into the VRAM slot; host transports read into the shared pinned
    // staging buffer and use GpuTransport for the H2D copy below.
    const bool host_store_possible = host_tier_ && m.size <= host_tier_->budget_bytes();
    const bool p2p_skip_tier_store = p2p_direct_to_device_with_tier();
    bool direct_to_device = file_io_->direct_to_device() &&
                            (!host_store_possible || p2p_skip_tier_store);
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
    // Retry through host staging on ANY direct-to-device read failure, not
    // only when the transport disabled itself. A P2P submit can now fail for
    // transient window pressure while P2P stays enabled (see
    // IoUringP2PFileIO::submit) — the old condition required
    // !file_io_->direct_to_device(), so that case fell through unretried and
    // returned -1, which surfaces as a null active-expert pointer and the hard
    // GGML_ABORT in wp-eval-cb. Retrying unconditionally is also correct for
    // the self-disable case: direct_to_device is already false there.
    if (!ok && direct_to_device) {
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
        ++stats_.page_ins_sync_direct;
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
    const auto fresh_stage_t0 = std::chrono::steady_clock::now();
    int evt = transport_.stage_in(dst, staging, m.size, pool_.slot_size(slot));
    const double fresh_stage_seconds = seconds_since(fresh_stage_t0);
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: stage_in returned evt=%d\n", s_diag_count, evt);
    if (evt < 0) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: gpu stage_in failed for page %d\n", page_idx);
        if (owns_slot) pool_.release_slot(slot);
        return -1;
    }
    transport_.release_event(evt);
    ++stats_.page_in_sync_fresh_count;
    stats_.page_in_sync_fresh_h2d_seconds += fresh_stage_seconds;

    // Shared sync_staging_ is owned by the WeightPager; no per-call free.

    page_to_slot_[page_idx] = slot;
    page_loaded_[page_idx]  = true;
    slot_to_page_[slot]     = page_idx;
    ++stats_.page_ins_sync_direct;
    record_page_in_(m.size, seconds_since(io_t0));
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT slot=%d\n", s_diag_count, slot);
    ++s_diag_count;
    return slot;
}

bool wp_pipeline_promotions_enabled() {
    const char * v = std::getenv("WP_PIPELINE_PROMOTIONS");
    return v == nullptr || std::strcmp(v, "0") != 0;
}

}  // namespace wp
