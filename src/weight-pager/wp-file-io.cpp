#include "wp-file-io.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <unistd.h>

#ifdef LLAMA_HAVE_IO_URING
#include <liburing.h>
#endif

namespace wp {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

    IoResult wait_any(int /*timeout_ms*/) override {
        if (results_.empty()) {
            // SyncPread cannot have a pending request that is not already
            // in results_ — every submit produces a result. Returning
            // Timeout here means "no completions queued."
            IoResult r;
            r.status = IoStatus::Timeout;
            return r;
        }
        IoResult r = results_.front();
        results_.pop_front();
        return r;
    }

    int pending() const override { return (int) results_.size(); }

    int fd(int fd_idx) const override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return -1;
        return fds_[fd_idx];
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

    IoResult wait_any(int timeout_ms) override {
        IoResult r;
        if (!ring_ok_) {
            r.status = IoStatus::ErrorNoSubmit;
            return r;
        }
        if (pending_ == 0) {
            r.status = IoStatus::Timeout;
            return r;
        }

        struct io_uring_cqe * cqe = nullptr;
        int                   ret = 0;

        if (timeout_ms < 0) {
            ret = io_uring_wait_cqe(&ring_, &cqe);
        } else if (timeout_ms == 0) {
            ret = io_uring_peek_cqe(&ring_, &cqe);
            if (ret == -EAGAIN) {
                r.status = IoStatus::Timeout;
                return r;
            }
        } else {
            struct __kernel_timespec ts;
            ts.tv_sec  = timeout_ms / 1000;
            ts.tv_nsec = (long) (timeout_ms % 1000) * 1000000L;
            ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
            if (ret == -ETIME) {
                r.status = IoStatus::Timeout;
                return r;
            }
        }

        if (ret < 0 || cqe == nullptr) {
            r.status     = IoStatus::ErrorIo;
            r.bytes_read = ret;
            return r;
        }

        r.req_id = cqe->user_data;
        const int res = cqe->res;
        if (res < 0) {
            r.status     = IoStatus::ErrorIo;
            r.bytes_read = res;
        } else {
            r.bytes_read = res;
            r.status     = IoStatus::Ok;  // caller compares against requested size for Short
        }
        io_uring_cqe_seen(&ring_, cqe);
        --pending_;
        return r;
    }

    int pending() const override { return pending_; }

    int fd(int fd_idx) const override {
        if (fd_idx < 0 || (size_t) fd_idx >= fds_.size()) return -1;
        return fds_[fd_idx];
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
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<FileIOLayer> create_file_io(std::vector<int> fds,
                                            bool             prefer_async,
                                            int              queue_depth) {
    const size_t n_fds = fds.size();
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

}  // namespace wp
