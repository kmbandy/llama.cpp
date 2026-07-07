#include "wp-file-io.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
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
        if (!ring_ok_ || pool_mapped_ == nullptr || fd_idx < 0 ||
            (size_t) fd_idx >= fds_.size() || dst == nullptr || size == 0) {
            return false;
        }

        // Read directly into the persistent pool dma_buf mapping. dst points
        // into the VRAM slot pool; its byte offset within the pool maps 1:1
        // onto the once-mmap'd dma_buf. This avoids a per-read
        // export/mmap/munmap/close (which dominated wall time and made P2P
        // slower than host staging).
        char * base = static_cast<char *>(pool_base_);
        char * d    = static_cast<char *>(dst);
        if (d < base || d + size > base + pool_size_) {
            // dst outside the exported pool region; can't P2P this read.
            switch_to_host_("dst outside pool", 0);
            return false;
        }
        void * mapped_dst = static_cast<char *>(pool_mapped_) + (d - base);

        struct io_uring_sqe * sqe = io_uring_get_sqe(&ring_);
        if (sqe == nullptr) {
            io_uring_submit(&ring_);
            sqe = io_uring_get_sqe(&ring_);
        }
        if (sqe == nullptr) {
            switch_to_host_("SQ ring full", 0);
            return false;
        }

        io_uring_prep_read(sqe, fd_idx, mapped_dst, (unsigned) size, (off_t) offset);
        sqe->flags    |= IOSQE_FIXED_FILE;
        sqe->user_data = req_id;
        ++pending_;
        return true;
    }

    void flush() override {
        if (!p2p_enabled_ && pending_ == 0) {
            host_->flush();
            return;
        }
        if (ring_ok_ && pending_ > 0) {
            io_uring_submit(&ring_);
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
                return true;
            }

            out = IoResult{};
            out.req_id = cqe->user_data;
            const int res = cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
            --pending_;

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

        // P2P ring drained. In host-fallback mode pull from the host layer;
        // in active-P2P mode with nothing pending there's nothing to reap.
        if (!p2p_enabled_) {
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

private:
    IoUringP2PFileIOLayer(std::vector<int> fds, std::unique_ptr<FileIOLayer> host)
        : fds_(std::move(fds)), host_(std::move(host)) {}

    bool init_(int queue_depth, const FileIOP2PConfig & cfg) {
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

    // Export the whole VRAM slot pool as a dma_buf ONCE and mmap it for the
    // layer's lifetime. Per-read submits then just index into this mapping.
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

        void * mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             dmabuf_fd, (off_t) dmabuf_offset);
        if (mapped == MAP_FAILED) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf mmap failed: %s\n", strerror(errno));
            hsa_close_(dmabuf_fd);
            return false;
        }

        pool_base_      = ptr;
        pool_size_      = size;
        pool_dmabuf_fd_ = dmabuf_fd;
        pool_mapped_    = mapped;
        p2p_enabled_    = true;
        LLAMA_LOG_WARN("wp::IoUringP2PFileIO: P2P enabled — pool dma_buf exported+mmap'd once (%.1f MiB, persistent)\n",
                       (double) size / 1048576.0);
        return true;
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
        if (ring_ok_) {
            if (files_registered_) io_uring_unregister_files(&ring_);
            io_uring_queue_exit(&ring_);
            ring_ok_ = false;
        }
        if (pool_mapped_ != nullptr && pool_mapped_ != MAP_FAILED) {
            munmap(pool_mapped_, pool_size_);
            pool_mapped_ = nullptr;
        }
        if (pool_dmabuf_fd_ >= 0 && hsa_close_ != nullptr) {
            hsa_close_(pool_dmabuf_fd_);
            pool_dmabuf_fd_ = -1;
        }
        if (libhsa_ != nullptr) {
            dlclose(libhsa_);
            libhsa_ = nullptr;
        }
    }

    std::vector<int> fds_;
    std::unique_ptr<FileIOLayer> host_;
    bool p2p_enabled_ = false;
    bool ring_ok_ = false;
    bool files_registered_ = false;
    int pending_ = 0;
    struct io_uring ring_ {};
    void * libhsa_ = nullptr;
    HsaExportDmaBufFn hsa_export_ = nullptr;
    HsaCloseDmaBufFn hsa_close_ = nullptr;

    // Persistent pool dma_buf mapping (exported+mmap'd once in init).
    void * pool_base_      = nullptr;
    size_t pool_size_      = 0;
    int    pool_dmabuf_fd_ = -1;
    void * pool_mapped_    = nullptr;
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
