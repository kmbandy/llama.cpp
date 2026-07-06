#include "wp-file-io.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
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
        if (!ring_ok_ || fd_idx < 0 || (size_t) fd_idx >= fds_.size() ||
            dst == nullptr || size == 0) {
            return false;
        }

        int dmabuf_fd = -1;
        uint64_t dmabuf_offset = 0;
        const uint32_t status = hsa_export_(dst, size, &dmabuf_fd, &dmabuf_offset);
        if (status != 0 || dmabuf_fd < 0) {
            switch_to_host_("export failed", status);
            return false;
        }

        void * mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             dmabuf_fd, (off_t) dmabuf_offset);
        if (mapped == MAP_FAILED) {
            const int saved_errno = errno;
            hsa_close_(dmabuf_fd);
            switch_to_host_errno_("mmap failed", saved_errno);
            return false;
        }

        struct io_uring_sqe * sqe = io_uring_get_sqe(&ring_);
        if (sqe == nullptr) {
            io_uring_submit(&ring_);
            sqe = io_uring_get_sqe(&ring_);
        }
        if (sqe == nullptr) {
            munmap(mapped, size);
            hsa_close_(dmabuf_fd);
            switch_to_host_("SQ ring full", 0);
            return false;
        }

        io_uring_prep_read(sqe, fd_idx, mapped, (unsigned) size, (off_t) offset);
        sqe->flags    |= IOSQE_FIXED_FILE;
        sqe->user_data = req_id;
        mappings_.emplace(req_id, Mapping{dmabuf_fd, mapped, size});
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

    IoResult wait_any(int timeout_ms) override {
        if (!p2p_enabled_ && pending_ == 0) {
            return host_->wait_any(timeout_ms);
        }

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
        int ret = 0;
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
            r.status = IoStatus::ErrorIo;
            r.bytes_read = ret;
            switch_to_host_("io_uring wait failed", 0);
            return r;
        }

        r.req_id = cqe->user_data;
        const int res = cqe->res;
        auto it = mappings_.find(r.req_id);
        if (it != mappings_.end()) {
            cleanup_mapping_(it->second);
            mappings_.erase(it);
        }
        io_uring_cqe_seen(&ring_, cqe);
        --pending_;

        if (res < 0) {
            r.status = IoStatus::ErrorIo;
            r.bytes_read = res;
            switch_to_host_errno_("read failed", -res);
        } else {
            r.status = IoStatus::Ok;
            r.bytes_read = res;
        }
        return r;
    }

    int pending() const override {
        return pending_ + host_->pending();
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
    struct Mapping {
        int fd = -1;
        void * addr = nullptr;
        size_t size = 0;
    };

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

        return probe_export_(cfg.pool_base, cfg.pool_size);
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

    bool probe_export_(void * ptr, size_t size) {
        if (ptr == nullptr || size == 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool probe target is empty\n");
            return false;
        }
        const size_t probe_size = size < 4096 ? size : 4096;
        int dmabuf_fd = -1;
        uint64_t dmabuf_offset = 0;
        const uint32_t status = hsa_export_(ptr, probe_size, &dmabuf_fd, &dmabuf_offset);
        if (status != 0 || dmabuf_fd < 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf probe export failed: %s (0x%04x), fd=%d\n",
                           hsa_status_name(status), status, dmabuf_fd);
            return false;
        }

        struct stat st {};
        if (fstat(dmabuf_fd, &st) != 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf probe fstat failed: %s\n", strerror(errno));
            hsa_close_(dmabuf_fd);
            return false;
        }

        void * mapped = mmap(nullptr, probe_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             dmabuf_fd, (off_t) dmabuf_offset);
        if (mapped == MAP_FAILED) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf probe mmap failed: %s\n", strerror(errno));
            hsa_close_(dmabuf_fd);
            return false;
        }
        munmap(mapped, probe_size);
        const uint32_t close_status = hsa_close_(dmabuf_fd);
        if (close_status != 0) {
            LLAMA_LOG_WARN("wp::IoUringP2PFileIO: pool dmabuf probe close failed: %s (0x%04x)\n",
                           hsa_status_name(close_status), close_status);
            return false;
        }
        p2p_enabled_ = true;
        return true;
    }

    void cleanup_mapping_(const Mapping & m) {
        if (m.addr != nullptr && m.addr != MAP_FAILED && m.size > 0) {
            munmap(m.addr, m.size);
        }
        if (m.fd >= 0 && hsa_close_ != nullptr) {
            hsa_close_(m.fd);
        }
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
        for (auto & kv : mappings_) {
            cleanup_mapping_(kv.second);
        }
        mappings_.clear();
        pending_ = 0;
        if (ring_ok_) {
            if (files_registered_) io_uring_unregister_files(&ring_);
            io_uring_queue_exit(&ring_);
            ring_ok_ = false;
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
    std::unordered_map<uint64_t, Mapping> mappings_;
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
