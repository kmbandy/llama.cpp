#include "wp-file-io.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <cstdlib>
#include <unistd.h>

#ifdef LLAMA_HAVE_IO_URING
#include <liburing.h>
#endif

namespace wp {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Issue POSIX_FADV_WILLNEED on a (fd, offset, size) range. Returns true on
// success. Failure is non-fatal — the read still works, just without the
// page-cache warm-up. Logged at most a handful of times to avoid spam.
//
// On non-Linux builds posix_fadvise isn't available; we no-op. The pager is
// Linux-only in practice (io_uring + HIP), but the guard keeps the file
// portable to test/CI on macOS dev machines.
bool fadvise_willneed(int fd, uint64_t offset, size_t size) {
#if defined(__linux__) && defined(POSIX_FADV_WILLNEED)
    if (fd < 0 || size == 0) return false;
    const int ret = posix_fadvise(fd, (off_t) offset, (off_t) size, POSIX_FADV_WILLNEED);
    if (ret != 0) {
        static int s_warn_count = 0;
        if (s_warn_count < 3) {
            LLAMA_LOG_WARN("wp::fadvise_willneed: posix_fadvise(WILLNEED) failed on fd %d "
                           "off=%llu size=%zu: %s\n",
                           fd, (unsigned long long) offset, size, strerror(ret));
            ++s_warn_count;
        }
        return false;
    }
    return true;
#else
    (void) fd; (void) offset; (void) size;
    return false;
#endif
}

// Tell the kernel that subsequent accesses to this fd will be random (i.e.
// disable the default sequential-readahead heuristic). Without this, the
// kernel speculatively reads ahead from any pread() call — useless for the
// tensor-strided MoE access pattern, and it pollutes the page cache with
// unrelated bytes that crowd out the ranges we ACTUALLY want via
// POSIX_FADV_WILLNEED. Best-effort; failure is logged once and ignored.
void fadvise_random_once(int fd) {
#if defined(__linux__) && defined(POSIX_FADV_RANDOM)
    if (fd < 0) return;
    const int ret = posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    if (ret != 0) {
        static int s_warn_count = 0;
        if (s_warn_count < 3) {
            LLAMA_LOG_WARN("wp::fadvise_random_once: posix_fadvise(RANDOM) failed on fd %d: %s\n",
                           fd, strerror(ret));
            ++s_warn_count;
        }
    }
#else
    (void) fd;
#endif
}

}  // anonymous namespace

int dup_clear_o_direct(int src_fd) {
    if (src_fd < 0) {
        return -1;
    }
    int fd = dup(src_fd);
    if (fd < 0) {
        return -1;
    }
#ifdef O_DIRECT
    int fl = fcntl(fd, F_GETFL);
    if (fl != -1 && (fl & O_DIRECT)) {
        if (fcntl(fd, F_SETFL, fl & ~O_DIRECT) != 0) {
            // Best-effort: clearing failed but we can still try to read.
            // The pager checks alignment elsewhere; log a single warning.
            LLAMA_LOG_WARN("wp::dup_clear_o_direct: failed to clear O_DIRECT on fd %d: %s\n",
                           fd, strerror(errno));
        }
    }
#endif
    return fd;
}

// ---------------------------------------------------------------------------
// SyncPread implementation
// ---------------------------------------------------------------------------
//
// pread() runs to completion inside submit(); the result is queued for
// wait_any() to drain. This keeps the FileIOLayer contract identical
// across both impls — the only difference visible to callers is whether
// submit() blocks on I/O.
//
// Useful as a fallback when liburing is unavailable, and as the reference
// implementation for tests.

namespace {

class SyncPreadFileIO : public FileIOLayer {
public:
    explicit SyncPreadFileIO(std::vector<int> fds) : fds_(std::move(fds)) {}

    ~SyncPreadFileIO() override {
        for (int fd : fds_) {
            if (fd >= 0) close(fd);
        }
    }

    bool submit(uint64_t req_id, int fd_idx, uint64_t offset,
                size_t size, void * dst) override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size() || dst == nullptr) {
            return false;
        }
        const int fd = fds_[fd_idx];
        if (fd < 0) {
            return false;
        }

        IoResult r;
        r.req_id = req_id;
        ssize_t total = 0;
        while ((size_t) total < size) {
            ssize_t n = pread(fd, (char *) dst + total, size - total,
                              (off_t) (offset + total));
            if (n < 0) {
                if (errno == EINTR) continue;
                r.status     = IoStatus::ErrorIo;
                r.bytes_read = -errno;
                results_.push_back(r);
                return true;  // submitted; failed-completion is queued
            }
            if (n == 0) {
                // EOF before requested size — short read.
                break;
            }
            total += n;
        }
        r.bytes_read = (int) total;
        r.status     = ((size_t) total == size) ? IoStatus::Ok : IoStatus::Short;
        results_.push_back(r);
        return true;
    }

    void flush() override { /* no-op for sync */ }

    // Every submit() runs the read inline and queues its result, so a
    // completion is always immediately available; timeout is irrelevant.
    bool reap_raw_(int /*timeout_ms*/, IoResult & out) override {
        if (results_.empty()) return false;
        out = results_.front();
        results_.pop_front();
        return true;
    }

    // In-flight = queued-not-yet-reaped (results_) plus reaped-not-yet-claimed
    // (ready_, held by the base demux for another consumer).
    int pending() const override { return (int) (results_.size() + ready_.size()); }

    FileIOTransport transport() const override { return FileIOTransport::SyncPread; }

    int fd(int fd_idx) const override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return -1;
        return fds_[fd_idx];
    }

    void advise_prefetch(int fd_idx, uint64_t offset, size_t size) override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return;
        fadvise_willneed(fds_[fd_idx], offset, size);
    }

private:
    std::vector<int>     fds_;
    std::deque<IoResult> results_;
};

}  // anonymous namespace

// ---------------------------------------------------------------------------
// IoUringAsync implementation
// ---------------------------------------------------------------------------
//
// One io_uring instance covering all of the model's split files (registered
// with io_uring_register_files). user_data on every SQE is the caller-
// supplied req_id; that round-trip is the contract the layer guarantees.
//
// Only compiled when liburing is available at build time. The factory
// silently falls back to SyncPread otherwise.

#ifdef LLAMA_HAVE_IO_URING

namespace {

class IoUringAsyncFileIO : public FileIOLayer {
public:
    static std::unique_ptr<IoUringAsyncFileIO> create(std::vector<int> fds, int queue_depth) {
        auto layer = std::unique_ptr<IoUringAsyncFileIO>(new IoUringAsyncFileIO(std::move(fds)));
        if (!layer->init_(queue_depth)) {
            return nullptr;
        }
        return layer;
    }

    ~IoUringAsyncFileIO() override {
        if (ring_ok_) {
            if (files_registered_) io_uring_unregister_files(&ring_);
            io_uring_queue_exit(&ring_);
        }
        for (int fd : fds_) {
            if (fd >= 0) close(fd);
        }
    }

    bool submit(uint64_t req_id, int fd_idx, uint64_t offset,
                size_t size, void * dst) override {
        if (!ring_ok_ || fd_idx < 0 || (size_t) fd_idx >= fds_.size() || dst == nullptr) {
            return false;
        }
        struct io_uring_sqe * sqe = io_uring_get_sqe(&ring_);
        if (sqe == nullptr) {
            // Ring full: flush and retry once.
            io_uring_submit(&ring_);
            sqe = io_uring_get_sqe(&ring_);
            if (sqe == nullptr) return false;
        }
        // Use registered-file index (faster than passing raw fds).
        io_uring_prep_read(sqe, fd_idx, dst, (unsigned) size, (off_t) offset);
        sqe->flags     |= IOSQE_FIXED_FILE;
        sqe->user_data  = req_id;
        ++pending_;
        return true;
    }

    void flush() override {
        if (!ring_ok_ || pending_ == 0) return;
        io_uring_submit(&ring_);
    }

    // Reap one completion from the ring. The demux/routing (buffering foreign
    // completions, matching req_ids) lives in the FileIOLayer base — this only
    // pulls the next raw CQE. Returns false on timeout / nothing ready.
    bool reap_raw_(int timeout_ms, IoResult & out) override {
        if (!ring_ok_) {
            // Unreachable on a live layer (init failure returns nullptr), but
            // signal fatal rather than a silent timeout if it ever happens.
            out = IoResult{};
            out.status = IoStatus::ErrorIo;
            out.req_id = 0;
            return true;
        }
        if (pending_ == 0) {
            return false;  // nothing in flight
        }

        struct io_uring_cqe * cqe = nullptr;
        int                   ret = 0;

        if (timeout_ms < 0) {
            ret = io_uring_wait_cqe(&ring_, &cqe);
        } else if (timeout_ms == 0) {
            ret = io_uring_peek_cqe(&ring_, &cqe);
            if (ret == -EAGAIN) {
                return false;  // no completion ready
            }
        } else {
            struct __kernel_timespec ts;
            ts.tv_sec  = timeout_ms / 1000;
            ts.tv_nsec = (long) (timeout_ms % 1000) * 1000000L;
            ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
            if (ret == -ETIME) {
                return false;  // timed out
            }
        }

        if (ret < 0 || cqe == nullptr) {
            // Transport-level failure — not tied to a request. req_id 0 marks
            // it fatal so a targeted waiter propagates rather than buffers it.
            out = IoResult{};
            out.status     = IoStatus::ErrorIo;
            out.bytes_read = ret;
            out.req_id     = 0;
            return true;
        }

        out = IoResult{};
        out.req_id = cqe->user_data;
        const int res = cqe->res;
        if (res < 0) {
            out.status     = IoStatus::ErrorIo;
            out.bytes_read = res;
        } else {
            out.bytes_read = res;
            out.status     = IoStatus::Ok;  // caller compares against requested size for Short
        }
        io_uring_cqe_seen(&ring_, cqe);
        --pending_;
        return true;
    }

    // In-flight = submitted-not-yet-reaped (pending_) plus reaped-not-yet-
    // claimed (ready_, buffered by the base demux for another consumer).
    int pending() const override { return pending_ + (int) ready_.size(); }

    FileIOTransport transport() const override { return FileIOTransport::IoUringHost; }

    int fd(int fd_idx) const override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return -1;
        return fds_[fd_idx];
    }

    void advise_prefetch(int fd_idx, uint64_t offset, size_t size) override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return;
        fadvise_willneed(fds_[fd_idx], offset, size);
    }

    int submit_batch(const std::vector<FileIOBatchRequest> & reqs) override {
        if (!ring_ok_) return 0;
        int n_queued = 0;
        // Pass 1: prep all SQEs. If the ring runs out of free SQEs midway
        // (rare but possible if a prior tick didn't fully drain), flush
        // what we have and continue. Worst case we end up doing one more
        // io_uring_submit than a single-syscall batch — still strictly
        // fewer than N syscalls.
        for (const auto & r : reqs) {
            if (r.fd_idx < 0 || (size_t) r.fd_idx >= fds_.size() || r.dst == nullptr) {
                break;
            }
            struct io_uring_sqe * sqe = io_uring_get_sqe(&ring_);
            if (sqe == nullptr) {
                // SQ ring full mid-batch — flush what we've prepped, then retry once.
                io_uring_submit(&ring_);
                sqe = io_uring_get_sqe(&ring_);
                if (sqe == nullptr) break;
            }
            io_uring_prep_read(sqe, r.fd_idx, r.dst, (unsigned) r.size, (off_t) r.offset);
            sqe->flags     |= IOSQE_FIXED_FILE;
            sqe->user_data  = r.req_id;
            ++pending_;
            ++n_queued;
        }
        // Pass 2: single io_uring_submit for the batch. This is the MAD-235
        // win — one syscall covers N expert-prefetches for the MoE layer
        // instead of N separate submits.
        if (n_queued > 0) {
            io_uring_submit(&ring_);
        }
        return n_queued;
    }

private:
    explicit IoUringAsyncFileIO(std::vector<int> fds) : fds_(std::move(fds)) {}

    bool init_(int queue_depth) {
        int ret = io_uring_queue_init(queue_depth, &ring_, 0);
        if (ret < 0) {
            LLAMA_LOG_WARN("wp::IoUringAsyncFileIO: queue_init failed: %s\n", strerror(-ret));
            return false;
        }
        ring_ok_ = true;

        // Register the model files for IOSQE_FIXED_FILE submissions.
        if (!fds_.empty()) {
            // io_uring_register_files takes a non-const int array.
            ret = io_uring_register_files(&ring_, fds_.data(), (unsigned) fds_.size());
            if (ret < 0) {
                LLAMA_LOG_WARN("wp::IoUringAsyncFileIO: register_files failed: %s — using non-fixed reads\n",
                               strerror(-ret));
                // Not fatal — submit() will fall back to non-fixed if needed.
                // But we don't currently support that fallback per-call, so
                // disable the layer to avoid silent breakage.
                io_uring_queue_exit(&ring_);
                ring_ok_ = false;
                return false;
            }
            files_registered_ = true;
        }
        return true;
    }

    std::vector<int>   fds_;
    bool               ring_ok_          = false;
    bool               files_registered_ = false;
    int                pending_          = 0;
    struct io_uring    ring_{};
};

}  // anonymous namespace

#endif  // LLAMA_HAVE_IO_URING

// ---------------------------------------------------------------------------
// FileIOLayer default submit_batch — loops singles. Subclasses with a real
// submission queue (io_uring) override this to push all SQEs first then
// submit once.
// ---------------------------------------------------------------------------

int FileIOLayer::submit_batch(const std::vector<FileIOBatchRequest> & reqs) {
    int n_queued = 0;
    for (const auto & r : reqs) {
        if (!submit(r.req_id, r.fd_idx, r.offset, r.size, r.dst)) {
            // Stop on first rejection — caller treats [n_queued, N) as failed.
            // SyncPread can't really fail at submit (it runs the read inline
            // and queues the result), so this loop almost always succeeds
            // fully here. The io_uring override gets the real batching win.
            break;
        }
        ++n_queued;
    }
    return n_queued;
}

// ---------------------------------------------------------------------------
// Completion demux — shared base implementation over reap_raw_()
// ---------------------------------------------------------------------------
//
// One FileIOLayer (io_uring ring / pread deque) is shared by several logical
// consumers: the prefetch scheduler, synchronous pager page-ins, and the
// ensure_batch expert fetch. Each waits for its OWN req_ids. A completion for
// consumer A can be reaped by consumer B first; B MUST NOT discard it. These
// methods buffer any completion whose req_id was not the one asked for, so its
// owner still claims it later. This is the fix for the shared-ring cross-drain
// that leaked prefetch slots and stalled the decode pipeline (2x regression).

IoResult FileIOLayer::wait_any(int timeout_ms) {
    // Buffered completions first — they were reaped earlier for someone who
    // hadn't asked yet; returning them here keeps them from lingering.
    if (!ready_.empty()) {
        auto it = ready_.begin();
        IoResult r = it->second;
        ready_.erase(it);
        return r;
    }
    IoResult r;
    if (reap_raw_(timeout_ms, r)) {
        return r;
    }
    r = IoResult{};
    r.status = IoStatus::Timeout;
    return r;
}

bool FileIOLayer::try_take(uint64_t req_id, IoResult & out) {
    // Drain everything immediately available into the buffer (non-blocking),
    // then take our own. This moves foreign completions into ready_ for their
    // owners rather than leaving them unreaped behind ours in the ring.
    IoResult r;
    while (reap_raw_(0, r)) {
        ready_[r.req_id] = r;
    }
    auto it = ready_.find(req_id);
    if (it == ready_.end()) {
        return false;
    }
    out = it->second;
    ready_.erase(it);
    return true;
}

IoResult FileIOLayer::wait_for_req(uint64_t req_id, int timeout_ms) {
    IoResult out;
    if (try_take(req_id, out)) {
        return out;
    }

    using clock = std::chrono::steady_clock;
    const bool        have_deadline = (timeout_ms >= 0);
    clock::time_point deadline      = have_deadline
        ? clock::now() + std::chrono::milliseconds(timeout_ms)
        : clock::time_point{};

    IoResult r;
    while (true) {
        int remaining = -1;
        if (have_deadline) {
            const auto now = clock::now();
            if (now >= deadline) break;
            remaining = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now).count();
            if (remaining <= 0) remaining = 1;  // don't degrade a live budget to a poll
        }
        if (!reap_raw_(remaining, r)) {
            // No completion within the budget. For an indefinite wait this can
            // only mean the transport is drained with nothing in flight — stop
            // rather than spin.
            break;
        }
        if (r.req_id == req_id) {
            return r;  // ours
        }
        if (r.status == IoStatus::ErrorIo && r.req_id == 0) {
            return r;  // transport-level failure — propagate, not a request result
        }
        ready_[r.req_id] = r;  // foreign completion — buffer for its owner, never drop
    }
    out = IoResult{};
    out.status = IoStatus::Timeout;
    return out;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<FileIOLayer> create_host_file_io(std::vector<int> fds,
                                                 bool             prefer_async,
                                                 int              queue_depth) {
    const size_t n_fds = fds.size();

    // Disable kernel auto-sequential-readahead on every fd before either
    // transport sees it. Without this, the kernel speculatively reads ahead
    // from each pread() (the SyncPread path is most affected, but io_uring
    // is also affected when fixed_file reads land on the same fd). That
    // speculative readahead pollutes the page cache with unrelated tensor
    // bytes and crowds out the ranges we deliberately advise via
    // POSIX_FADV_WILLNEED for the next K layers (MAD-232 / advise_prefetch).
    for (int fd : fds) {
        fadvise_random_once(fd);
    }

#ifdef LLAMA_HAVE_IO_URING
    if (prefer_async) {
        // Copy fds for the attempt; on success the layer owns them and we
        // discard the caller's vector. On failure the layer's destructor
        // closes its copy, so we re-dup them before falling back to
        // SyncPread to avoid double-close.
        std::vector<int> fds_copy;
        fds_copy.reserve(fds.size());
        for (int fd : fds) {
            int dup_fd = (fd >= 0) ? dup(fd) : -1;
            fds_copy.push_back(dup_fd);
        }
        auto layer = IoUringAsyncFileIO::create(std::move(fds_copy), queue_depth);
        if (layer) {
            // Original fds passed by caller are no longer needed by us;
            // close them — the layer owns its dup'd copies.
            for (int fd : fds) {
                if (fd >= 0) close(fd);
            }
            LLAMA_LOG_INFO("wp::create_file_io: io_uring (queue_depth=%d, fds=%zu)\n",
                           queue_depth, n_fds);
            return layer;
        }
        LLAMA_LOG_WARN("wp::create_file_io: io_uring init failed — falling back to pread\n");
        // Layer's dtor closed the dup'd copies. Caller's fds are still open;
        // we hand them to SyncPread below.
    }
#else
    (void) prefer_async;
    (void) queue_depth;
#endif
    LLAMA_LOG_INFO("wp::create_file_io: SyncPread (fds=%zu)\n", n_fds);
    return std::unique_ptr<FileIOLayer>(new SyncPreadFileIO(std::move(fds)));
}

std::unique_ptr<FileIOLayer> create_file_io(std::vector<int> fds,
                                            bool             prefer_async,
                                            int              queue_depth,
                                            const FileIOP2PConfig * p2p) {
    const char * env = std::getenv("LLAMA_WP_TRANSPORT");
    const bool want_p2p = (env != nullptr && std::strcmp(env, "p2p") == 0);

    if (!want_p2p) {
        if (env != nullptr && env[0] != '\0' && std::strcmp(env, "host") != 0) {
            LLAMA_LOG_WARN("wp::create_file_io: unknown LLAMA_WP_TRANSPORT=%s; using host ladder\n", env);
        }
        return create_host_file_io(std::move(fds), prefer_async, queue_depth);
    }

    if (p2p == nullptr || p2p->pool_base == nullptr || p2p->pool_size == 0) {
        LLAMA_LOG_WARN("wp::create_file_io: LLAMA_WP_TRANSPORT=p2p requested but pool export target is unavailable; falling back to host ladder\n");
        return create_host_file_io(std::move(fds), prefer_async, queue_depth);
    }

    std::vector<int> p2p_fds;
    p2p_fds.reserve(fds.size());
    for (int fd : fds) {
        p2p_fds.push_back((fd >= 0) ? dup(fd) : -1);
    }
    auto p2p_layer = create_p2p_file_io(std::move(p2p_fds), prefer_async, queue_depth, *p2p);
    if (p2p_layer) {
        for (int fd : fds) {
            if (fd >= 0) close(fd);
        }
        LLAMA_LOG_INFO("wp::create_file_io: active transport=p2p (fallback ladder p2p->io_uring-host->sync-pread, queue_depth=%d)\n",
                       queue_depth);
        return p2p_layer;
    }

    LLAMA_LOG_WARN("wp::create_file_io: p2p unavailable; falling back p2p->io_uring-host->sync-pread\n");
    auto host = create_host_file_io(std::move(fds), prefer_async, queue_depth);
    if (host) {
        LLAMA_LOG_INFO("wp::create_file_io: active transport=%s after p2p fallback\n",
                       host->transport() == FileIOTransport::IoUringHost ? "io_uring-host" : "sync-pread");
    }
    return host;
}

}  // namespace wp
