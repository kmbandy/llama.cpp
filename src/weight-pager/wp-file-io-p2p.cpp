#include "wp-file-io.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

#if defined(GGML_USE_HIP) && defined(LLAMA_HAVE_IO_URING) && defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#endif

namespace wp {

#if defined(GGML_USE_HIP) && defined(LLAMA_HAVE_IO_URING) && defined(__linux__)

namespace {

using HsaExportDmaBufFn = uint32_t (*)(const void *, size_t, int *, uint64_t *);
using HsaCloseDmaBufFn  = uint32_t (*)(int);

const char * hsa_status_name(uint32_t status) {
    switch (status) {
        case 0x0000: return "SUCCESS";
        case 0x1004: return "INVALID_AGENT";
        case 0x1008: return "OUT_OF_RESOURCES";
        case 0x100B: return "NOT_INITIALIZED";
        default:     return "UNKNOWN";
    }
}

bool env_force_p2p() {
    const char * env = std::getenv("LLAMA_WP_TRANSPORT_FORCE");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

std::vector<int> dup_fds(const std::vector<int> & fds) {
    std::vector<int> out;
    out.reserve(fds.size());
    for (int fd : fds) {
        out.push_back((fd >= 0) ? dup(fd) : -1);
    }
    return out;
}

class IoUringP2PFileIOLayer : public FileIOLayer {
public:
    static std::unique_ptr<IoUringP2PFileIOLayer> create(std::vector<int> fds,
                                                         bool prefer_async,
                                                         int queue_depth,
                                                         const FileIOP2PConfig & cfg) {
        queue_depth = resolve_p2p_queue_depth(queue_depth);
        auto host = create_host_file_io(dup_fds(fds), prefer_async, queue_depth);
        if (!host) {
            return nullptr;
        }

        auto layer = std::unique_ptr<IoUringP2PFileIOLayer>(
            new IoUringP2PFileIOLayer(std::move(fds), std::move(host)));
        if (!layer->init_(queue_depth, cfg)) {
            return nullptr;
        }
        return layer;
    }

    ~IoUringP2PFileIOLayer() override {
        shutdown_p2p_();
        for (int fd : fds_) {
            if (fd >= 0) close(fd);
        }
    }

    bool submit(uint64_t req_id, int fd_idx, uint64_t offset,
                size_t size, void * dst) override {
        if (!p2p_enabled_) {
            return host_->submit(req_id, fd_idx, offset, size, dst);
        }
        if (!ring_ok_ || pool_dmabuf_fd_ < 0 || fd_idx < 0 ||
            (size_t) fd_idx >= fds_.size() || dst == nullptr || size == 0) {
            return false;
        }

        // ReBAR P2P: NVMe DMA into VRAM via windowed dma_buf maps. Pool is
        // exported once; we never map the whole pool. A small refcounted
        // window cache (cap ≈ 2*QD) avoids mmap/munmap per CQE while keeping
        // peak host VA at a few dozen MiB, not 27 GiB.
        char * base = static_cast<char *>(pool_base_);
        char * d    = static_cast<char *>(dst);
        if (d < base || d + size > base + pool_size_) {
            // Not a pool destination. Since submit() can now reject a single
            // request for window pressure while P2P stays enabled, the pager
            // retries that read into its host staging buffer — which is
            // legitimately outside the pool. Treating it as a transport fault
            // killed P2P AND returned false, so the retry failed on its only
            // attempt and the caller returned -1 (null active expert -> the
            // GGML_ABORT in wp-eval-cb).
            //
            // A non-pool dst is simply a request the host layer must serve.
            // reap_raw_ drains host_ whenever it has work outstanding, so the
            // completion is not stranded.
            return host_->submit(req_id, fd_idx, offset, size, dst);
        }
        const uint64_t pool_off = (uint64_t) (d - base);

        void * mapped_dst = nullptr;
        uint64_t map_key  = 0;
        if (!acquire_window_(pool_off, size, mapped_dst, map_key)) {
            if (errno != EAGAIN) {
                switch_to_host_errno_("window mmap failed", errno);
                return false;
            }
            // Window pressure. Push queued SQEs so in-flight reads can land,
            // drain whatever has completed (each completion releases its
            // window via release_inflight_key_), then try once more. Reaped
            // results go into ready_, which the normal reap path drains, so
            // consuming CQEs here cannot lose a completion.
            flush_submissions_();
            while (reap_ready_cqe_()) { }
            if (!acquire_window_(pool_off, size, mapped_dst, map_key)) {
                // Still no window. Fail THIS request only — the caller falls
                // back for it — and leave P2P enabled for everything after.
                ++window_pressure_fallbacks_;
                return false;
            }
        }

        struct io_uring_sqe * sqe = io_uring_get_sqe(&ring_);
        if (sqe == nullptr) {
            flush_submissions_();
            sqe = io_uring_get_sqe(&ring_);
        }
        if (sqe == nullptr) {
            release_window_key_(map_key);
            switch_to_host_("SQ ring full", 0);
            return false;
        }

        io_uring_prep_read(sqe, fd_idx, mapped_dst, (unsigned) size, (off_t) offset);
        // FIXED_FILE: registered fds. IOSQE_ASYNC: do not complete inline in
        // io_uring_submit — measured ensure_batch_submit_ms >> wait_ms (I/O
        // was finishing inside submit, capping random P2P at ~2 GB/s vs
        // host O_DIRECT ~6 GB/s at QD=6). Force async so multi-QD wait path
        // can actually overlap.
        sqe->flags    |= IOSQE_FIXED_FILE | IOSQE_ASYNC;
        sqe->user_data = req_id;
        inflight_keys_[req_id] = map_key;
        pending_submit_.push_back(req_id);
        pending_reqs_.push_back(req_id);
        ++pending_;
        return true;
    }

    void flush() override {
        if (ring_ok_ && !pending_submit_.empty()) {
            flush_submissions_();
        }
        if (!p2p_enabled_ && pending_ == 0) {
            host_->flush();
            return;
        }
    }

    // Reap one raw completion for the FileIOLayer base demux. Drains the P2P
    // ring first (it may hold reads submitted before a mid-flight
    // switch_to_host_); once the ring is empty and we're in host-fallback
    // mode, pull one completion from the host layer. Returns false on
    // timeout / nothing available. req_id routing + foreign-completion
    // buffering happen in the base.
    bool reap_raw_(int timeout_ms, IoResult & out) override {
        if (ring_ok_ && pending_ > 0) {
            if (!pending_submit_.empty()) {
                flush_submissions_();
            }
            if (pending_ == 0) {
                return false;
            }

            struct io_uring_cqe * cqe = nullptr;
            int ret = 0;
            if (timeout_ms < 0) {
                // Retry on signal interruption — the reads are still in flight,
                // nothing failed. ROCm/HIP fires signals frequently; treating
                // -EINTR as a transport failure spuriously downgrades P2P.
                do {
                    ret = io_uring_wait_cqe(&ring_, &cqe);
                } while (ret == -EINTR);
            } else if (timeout_ms == 0) {
                ret = io_uring_peek_cqe(&ring_, &cqe);
                if (ret == -EAGAIN || ret == -EINTR) return false;  // nothing ready yet
            } else {
                struct __kernel_timespec ts;
                ts.tv_sec  = timeout_ms / 1000;
                ts.tv_nsec = (long) (timeout_ms % 1000) * 1000000L;
                ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
                // -EINTR: signal interrupted the bounded wait. Report "no
                // completion yet" so the caller's deadline loop retries with a
                // recomputed budget, rather than downgrading the transport.
                if (ret == -ETIME || ret == -EINTR) return false;
            }

            if (ret < 0 || cqe == nullptr) {
                out = IoResult{};
                out.status     = IoStatus::ErrorIo;
                out.bytes_read = ret;
                out.req_id     = 0;  // transport-level failure — fatal, propagate
                switch_to_host_errno_("io_uring wait failed", ret < 0 ? -ret : 0);
                fail_p2p_pending_(ret < 0 ? ret : -EIO);
                return true;
            }

            out = IoResult{};
            out.req_id = cqe->user_data;
            const int res = cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
            --pending_;
            remove_pending_req_(out.req_id);
            release_inflight_key_(out.req_id);
            finish_p2p_read_(out.req_id);

            if (res < 0) {
                out.status     = IoStatus::ErrorIo;
                out.bytes_read = res;
                switch_to_host_errno_("read failed", -res);
            } else {
                out.status     = IoStatus::Ok;
                out.bytes_read = res;
            }
            return true;
        }

        // P2P ring drained. Pull from the host layer in host-fallback mode, and
        // ALSO while P2P is still enabled if the host layer has work
        // outstanding — submit() now routes non-pool destinations there (the
        // pager's staging-buffer retry after a window-pressure rejection), so
        // the two coexist. Without this the routed read is submitted and never
        // reaped, and wait_for_req(req_id, -1) blocks forever.
        if (!p2p_enabled_ || host_->pending() > 0) {
            IoResult r = host_->wait_any(timeout_ms);
            if (r.status == IoStatus::Timeout) return false;
            out = r;
            return true;
        }
        return false;
    }

    int pending() const override {
        return pending_ + host_->pending() + (int) ready_.size();
    }

    int fd(int fd_idx) const override {
        if (!p2p_enabled_) {
            return host_->fd(fd_idx);
        }
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return -1;
        return fds_[fd_idx];
    }

    void advise_prefetch(int fd_idx, uint64_t offset, size_t size) override {
        host_->advise_prefetch(fd_idx, offset, size);
    }

    int submit_batch(const std::vector<FileIOBatchRequest> & reqs) override {
        if (!p2p_enabled_) {
            return host_->submit_batch(reqs);
        }
        int n = 0;
        for (const auto & r : reqs) {
            if (!submit(r.req_id, r.fd_idx, r.offset, r.size, r.dst)) {
                break;
            }
            ++n;
        }
        if (n > 0) {
            flush();
        }
        return n;
    }

    FileIOTransport transport() const override {
        return p2p_enabled_ ? FileIOTransport::IoUringP2P : host_->transport();
    }

    bool direct_to_device() const override {
        return p2p_enabled_;
    }

    FileIOConcurrency concurrency() const override {
        FileIOConcurrency out;
        out.starts = p2p_inflight_starts_;
        out.peak = p2p_inflight_peak_;
        out.window_pressure_fallbacks = window_pressure_fallbacks_;
        out.average_at_start = p2p_inflight_starts_ == 0 ? 0.0 :
            p2p_inflight_sum_at_start_ / (double) p2p_inflight_starts_;
        return out;
    }

private:
    IoUringP2PFileIOLayer(std::vector<int> fds, std::unique_ptr<FileIOLayer> host)
        : fds_(std::move(fds)), host_(std::move(host)) {}

    void synthesize_submit_failures_(int err) {
        while (!pending_submit_.empty()) {
            const uint64_t req_id = pending_submit_.front();
            pending_submit_.pop_front();

            IoResult r;
            r.req_id     = req_id;
            r.status     = IoStatus::ErrorNoSubmit;
            r.bytes_read = err;
            ready_[req_id] = r;
            --pending_;
            remove_pending_req_(req_id);
            release_inflight_key_(req_id);
        }
    }

    bool flush_submissions_() {
        if (!ring_ok_ || pending_submit_.empty()) {
            return true;
        }

        while (!pending_submit_.empty()) {
            int ret = 0;
            do {
                ret = io_uring_submit(&ring_);
            } while (ret == -EINTR);

            if (ret == -EBUSY || ret == -EAGAIN || ret == 0) {
                bool drained = false;
                while (reap_ready_cqe_()) {
                    drained = true;
                }
                if (drained) {
                    continue;
                }
            }
            if (ret < 0) {
                switch_to_host_errno_("io_uring submit failed", -ret);
                synthesize_submit_failures_(ret);
                return false;
            }
            if (ret == 0) {
                switch_to_host_("io_uring submit made no progress", 0);
                synthesize_submit_failures_(-EAGAIN);
                return false;
            }

            for (int i = 0; i < ret && !pending_submit_.empty(); ++i) {
                start_p2p_read_(pending_submit_.front());
                pending_submit_.pop_front();
            }
        }
        return true;
    }

    bool reap_ready_cqe_() {
        if (pending_ == 0) {
            return false;
        }

        struct io_uring_cqe * cqe = nullptr;
        int ret = io_uring_peek_cqe(&ring_, &cqe);
        if (ret == -EAGAIN || ret == -EINTR || cqe == nullptr) {
            return false;
        }
        if (ret < 0) {
            return false;
        }

        IoResult r;
        r.req_id = cqe->user_data;
        const int res = cqe->res;
        io_uring_cqe_seen(&ring_, cqe);
        --pending_;
        remove_pending_req_(r.req_id);
        release_inflight_key_(r.req_id);
        finish_p2p_read_(r.req_id);

        if (res < 0) {
            r.status     = IoStatus::ErrorIo;
            r.bytes_read = res;
            switch_to_host_errno_("read failed", -res);
        } else {
            r.status     = IoStatus::Ok;
            r.bytes_read = res;
        }
        ready_[r.req_id] = r;
        return true;
    }

    void remove_pending_req_(uint64_t req_id) {
        for (auto it = pending_reqs_.begin(); it != pending_reqs_.end(); ++it) {
            if (*it == req_id) {
                pending_reqs_.erase(it);
                return;
            }
        }
    }

    void fail_p2p_pending_(int err) {
        for (uint64_t req_id : pending_reqs_) {
            IoResult r;
            r.req_id     = req_id;
            r.status     = IoStatus::ErrorIo;
            r.bytes_read = err;
            ready_[req_id] = r;
            release_inflight_key_(req_id);
            finish_p2p_read_(req_id);
        }
        pending_reqs_.clear();
        pending_submit_.clear();
        pending_ = 0;
    }

    void start_p2p_read_(uint64_t req_id) {
        p2p_submitted_.insert(req_id);
        ++p2p_inflight_;
        p2p_inflight_sum_at_start_ += (double) p2p_inflight_;
        ++p2p_inflight_starts_;
        if ((uint64_t) p2p_inflight_ > p2p_inflight_peak_) {
            p2p_inflight_peak_ = (uint64_t) p2p_inflight_;
        }
    }

    void finish_p2p_read_(uint64_t req_id) {
        if (p2p_submitted_.erase(req_id) > 0 && p2p_inflight_ > 0) {
            --p2p_inflight_;
        }
    }

    bool init_(int queue_depth, const FileIOP2PConfig & cfg) {
        queue_depth_ = queue_depth;
        int rt_version = 0;
        if (hipRuntimeGetVersion(&rt_version) == hipSuccess) {
            const int major = rt_version / 10000000;
            const int minor = (rt_version / 100000) % 100;
            const int patch = (rt_version / 1000) % 100;
            if (major != 7 && !env_force_p2p()) {
                LLAMA_LOG_WARN("wp::IoUringP2PFileIO: ROCm runtime %d.%d.%d is unvalidated for P2P; validated version is 7.2.2. Set LLAMA_WP_TRANSPORT_FORCE=1 to silence this warning.\n",
                               major, minor, patch);
            }
        }

        if (!load_hsa_()) {
            return false;
        }
        int ret = io_uring_queue_init(queue_depth, &ring_, 0);
        if (ret >= 0) {
            set_iowq_max_workers(&ring_, "p2p");
        }
        if (ret < 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: queue_init failed: %s\n", strerror(-ret));
            return false;
        }
        ring_ok_ = true;

        if (!fds_.empty()) {
            ret = io_uring_register_files(&ring_, fds_.data(), (unsigned) fds_.size());
            if (ret < 0) {
                LLAMA_LOG_WARN("wp::IoUringP2PFileIO: register_files failed: %s\n", strerror(-ret));
                return false;
            }
            files_registered_ = true;
        }

        // Cap cached maps: enough for multi-QD bursts without 27 GiB VA.
        // Prefer 4*QD so a few ensure_batch waves reuse maps before LRU.
        max_windows_ = resolve_p2p_window_cache_max(queue_depth);
        return setup_pool_mapping_(cfg.pool_base, cfg.pool_size);
    }

    bool load_hsa_() {
        libhsa_ = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (libhsa_ == nullptr) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: dlopen libhsa-runtime64.so.1 failed: %s\n", dlerror());
            return false;
        }
        void * export_ptr = dlsym(libhsa_, "hsa_amd_portable_export_dmabuf");
        if (export_ptr == nullptr) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: dlsym hsa_amd_portable_export_dmabuf failed: %s\n", dlerror());
            return false;
        }
        void * close_ptr = dlsym(libhsa_, "hsa_amd_portable_close_dmabuf");
        if (close_ptr == nullptr) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: dlsym hsa_amd_portable_close_dmabuf failed: %s\n", dlerror());
            return false;
        }
        hsa_export_ = reinterpret_cast<HsaExportDmaBufFn>(export_ptr);
        hsa_close_  = reinterpret_cast<HsaCloseDmaBufFn>(close_ptr);
        return true;
    }

    // Export the VRAM pool as one dma_buf (ReBAR/device memory). Do NOT map
    // the whole pool into host VA — that was ~27 GiB page-tables and OOM.
    // submit() maps only the in-flight destination window (page-aligned),
    // so NVMe DMA still targets VRAM via ReBAR without a host bounce buffer.
    bool setup_pool_mapping_(void * ptr, size_t size) {
        if (ptr == nullptr || size == 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool target is empty\n");
            return false;
        }

        int dmabuf_fd = -1;
        uint64_t dmabuf_offset = 0;
        const uint32_t status = hsa_export_(ptr, size, &dmabuf_fd, &dmabuf_offset);
        if (status != 0 || dmabuf_fd < 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf export failed: %s (0x%04x), fd=%d\n",
                           hsa_status_name(status), status, dmabuf_fd);
            return false;
        }

        struct stat st {};
        if (fstat(dmabuf_fd, &st) != 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf fstat failed: %s\n", strerror(errno));
            hsa_close_(dmabuf_fd);
            return false;
        }

        long ps = sysconf(_SC_PAGESIZE);
        if (ps < 4096) {
            ps = 4096;
        }
        page_size_ = (size_t) ps;

        // Prove windowed mmap works (one page), then unmap — no full-pool map.
        void * probe = mmap(nullptr, page_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                            dmabuf_fd, (off_t) dmabuf_offset);
        if (probe == MAP_FAILED) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: windowed dmabuf mmap probe failed: %s\n",
                           strerror(errno));
            hsa_close_(dmabuf_fd);
            return false;
        }
        munmap(probe, page_size_);

        pool_base_          = ptr;
        pool_size_          = size;
        pool_dmabuf_fd_     = dmabuf_fd;
        pool_dmabuf_offset_ = dmabuf_offset;
        p2p_enabled_        = true;
        const char * iowq_workers = std::getenv("WP_IOWQ_MAX_WORKERS");
        LLAMA_LOG_WARN(
            "wp::IoUringP2PFileIO: P2P enabled — pool dma_buf exported (%.1f MiB VRAM), "
            "window cache max=%d queue_depth=%d iowq_workers=%s tier_mode=%s page=%zu B "
            "(no full-pool host map, no host bounce)\n",
            (double) size / 1048576.0, max_windows_, queue_depth_,
            (iowq_workers != nullptr && iowq_workers[0] != '\0') ? iowq_workers : "kernel-default",
            p2p_direct_to_device_with_tier() ? "direct-to-VRAM/skip-tier" : "staging/store-tier",
            page_size_);
        return true;
    }

    // Window cache: key = page-aligned dmabuf map offset. Refcounted while
    // I/O is in flight; idle entries kept up to max_windows_ then LRU-evicted.
    struct CacheEntry {
        void *   base     = nullptr;
        size_t   len      = 0;
        int      refs     = 0;
        uint64_t last_tick = 0;
    };

    void compute_map_geom_(uint64_t pool_off, size_t size,
                           uint64_t & map_off, size_t & pad, size_t & map_len) const {
        const uint64_t abs_off = pool_dmabuf_offset_ + pool_off;
        map_off = abs_off & ~((uint64_t) page_size_ - 1ull);
        pad     = (size_t) (abs_off - map_off);
        map_len = pad + size;
        map_len = (map_len + page_size_ - 1) & ~(page_size_ - 1);
    }

    void unmap_entry_(CacheEntry & e) {
        if (e.base != nullptr && e.base != MAP_FAILED && e.len > 0) {
            munmap(e.base, e.len);
        }
        e = CacheEntry{};
    }

    // Drop one idle (refs==0) LRU entry. Returns true if something was freed.
    bool evict_one_idle_() {
        uint64_t best_key  = UINT64_MAX;
        uint64_t best_tick = UINT64_MAX;
        for (auto & kv : window_cache_) {
            if (kv.second.refs != 0) {
                continue;
            }
            if (kv.second.last_tick < best_tick) {
                best_tick = kv.second.last_tick;
                best_key  = kv.first;
            }
        }
        if (best_key == UINT64_MAX) {
            return false;
        }
        unmap_entry_(window_cache_[best_key]);
        window_cache_.erase(best_key);
        return true;
    }

    void trim_idle_to_cap_() {
        while ((int) window_cache_.size() > max_windows_) {
            if (!evict_one_idle_()) {
                break; // all remaining are in-flight
            }
        }
    }

    bool acquire_window_(uint64_t pool_off, size_t size,
                         void * & mapped_dst, uint64_t & map_key) {
        uint64_t map_off = 0;
        size_t pad = 0, map_len = 0;
        compute_map_geom_(pool_off, size, map_off, pad, map_len);
        map_key = map_off;

        auto it = window_cache_.find(map_off);
        if (it != window_cache_.end()) {
            CacheEntry & e = it->second;
            if (e.base != nullptr && e.len >= map_len) {
                ++e.refs;
                e.last_tick = ++cache_tick_;
                mapped_dst = static_cast<char *>(e.base) + pad;
                return true;
            }
            // Too small or broken: only replace if idle.
            if (e.refs == 0) {
                unmap_entry_(e);
                window_cache_.erase(it);
            } else {
                // In use with smaller map — key collision; fail rare path.
                errno = EBUSY;
                return false;
            }
        }

        while ((int) window_cache_.size() >= max_windows_) {
            if (!evict_one_idle_()) {
                // TRANSIENT, not fatal: the cache is at cap and every entry is
                // still referenced by an in-flight read. Signal EAGAIN so the
                // caller drains completions and retries instead of tearing the
                // whole transport down. This used to report ENOMEM, which the
                // caller could not distinguish from a real mmap failure, so a
                // moment of window pressure permanently downgraded P2P to
                // sync-pread for the rest of the process.
                errno = EAGAIN;
                return false;
            }
        }

        void * mapped = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                             pool_dmabuf_fd_, (off_t) map_off);
        if (mapped == MAP_FAILED) {
            return false;
        }
        CacheEntry e;
        e.base      = mapped;
        e.len       = map_len;
        e.refs      = 1;
        e.last_tick = ++cache_tick_;
        window_cache_[map_off] = e;
        mapped_dst = static_cast<char *>(mapped) + pad;
        return true;
    }

    void release_window_key_(uint64_t map_key) {
        auto it = window_cache_.find(map_key);
        if (it == window_cache_.end()) {
            return;
        }
        CacheEntry & e = it->second;
        if (e.refs > 0) {
            --e.refs;
        }
        e.last_tick = ++cache_tick_;
        // Keep mapped for reuse; only unmap when over cap.
        trim_idle_to_cap_();
    }

    void release_inflight_key_(uint64_t req_id) {
        auto it = inflight_keys_.find(req_id);
        if (it == inflight_keys_.end()) {
            return;
        }
        release_window_key_(it->second);
        inflight_keys_.erase(it);
    }

    void release_all_windows_() {
        inflight_keys_.clear();
        for (auto & kv : window_cache_) {
            unmap_entry_(kv.second);
        }
        window_cache_.clear();
    }

    void switch_to_host_(const char * reason, uint32_t hsa_status) {
        if (!p2p_enabled_) return;
        p2p_enabled_ = false;
        if (hsa_status != 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: p2p failed (%s: %s 0x%04x); active transport downgraded to %s\n",
                           reason, hsa_status_name(hsa_status), hsa_status,
                           host_->transport() == FileIOTransport::IoUringHost ? "io_uring-host" : "sync-pread");
        } else {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: p2p failed (%s); active transport downgraded to %s\n",
                           reason,
                           host_->transport() == FileIOTransport::IoUringHost ? "io_uring-host" : "sync-pread");
        }
    }

    void switch_to_host_errno_(const char * reason, int saved_errno) {
        if (!p2p_enabled_) return;
        p2p_enabled_ = false;
        LLAMA_LOG_WARN("wp::IoUringP2PFileIO: p2p failed (%s: %s); active transport downgraded to %s\n",
                       reason, strerror(saved_errno),
                       host_->transport() == FileIOTransport::IoUringHost ? "io_uring-host" : "sync-pread");
    }

    void shutdown_p2p_() {
        pending_ = 0;
        release_all_windows_();
        if (ring_ok_) {
            if (files_registered_) io_uring_unregister_files(&ring_);
            io_uring_queue_exit(&ring_);
            ring_ok_ = false;
        }
        // No full-pool munmap — only window maps, already released above.
        if (pool_dmabuf_fd_ >= 0 && hsa_close_ != nullptr) {
            hsa_close_(pool_dmabuf_fd_);
            pool_dmabuf_fd_ = -1;
        }
        pool_base_ = nullptr;
        pool_size_ = 0;
        pool_dmabuf_offset_ = 0;
        if (libhsa_ != nullptr) {
            dlclose(libhsa_);
            libhsa_ = nullptr;
        }
    }

    std::vector<int> fds_;
    std::unique_ptr<FileIOLayer> host_;
    bool p2p_enabled_ = false;
    // Count of requests that fell back for window pressure alone.
    // Non-fatal; if this is large the window cache is undersized
    // relative to queue depth.
    uint64_t window_pressure_fallbacks_ = 0;
    bool ring_ok_ = false;
    bool files_registered_ = false;
    int pending_ = 0;
    std::deque<uint64_t> pending_submit_;
    std::deque<uint64_t> pending_reqs_;
    struct io_uring ring_ {};
    void * libhsa_ = nullptr;
    HsaExportDmaBufFn hsa_export_ = nullptr;
    HsaCloseDmaBufFn hsa_close_ = nullptr;

    // Pool: device pointer + one dma_buf export. CPU maps only a small window cache.
    void *   pool_base_          = nullptr;
    size_t   pool_size_          = 0;
    int      pool_dmabuf_fd_     = -1;
    uint64_t pool_dmabuf_offset_ = 0;
    size_t   page_size_          = 4096;
    int      max_windows_        = 32;
    uint64_t cache_tick_         = 0;
    std::unordered_map<uint64_t, CacheEntry> window_cache_;   // map_off -> entry
    std::unordered_map<uint64_t, uint64_t>    inflight_keys_;  // req_id -> map_off
    int      queue_depth_        = 0;
    int      p2p_inflight_       = 0;
    uint64_t p2p_inflight_starts_ = 0;
    uint64_t p2p_inflight_peak_   = 0;
    double   p2p_inflight_sum_at_start_ = 0.0;
    std::unordered_set<uint64_t> p2p_submitted_;
};

}  // anonymous namespace

std::unique_ptr<FileIOLayer> create_p2p_file_io(std::vector<int> fds,
                                                bool prefer_async,
                                                int queue_depth,
                                                const FileIOP2PConfig & cfg) {
    return IoUringP2PFileIOLayer::create(std::move(fds), prefer_async, queue_depth, cfg);
}

#else

std::unique_ptr<FileIOLayer> create_p2p_file_io(std::vector<int> fds,
                                                bool prefer_async,
                                                int queue_depth,
                                                const FileIOP2PConfig & cfg) {
    (void) fds;
    (void) prefer_async;
    (void) queue_depth;
    (void) cfg;
    return nullptr;
}

#endif

}  // namespace wp
