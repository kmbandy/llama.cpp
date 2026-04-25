#ifdef LLAMA_HAVE_IO_URING

#include "llama-io-uring.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstring>
#include <stdexcept>

llama_io_uring::llama_io_uring(int fd, int queue_depth) : fd_(fd) {
    if (fd < 0) return;
    int ret = io_uring_queue_init(queue_depth, &ring_, 0);
    if (ret < 0) {
        LLAMA_LOG_WARN("%s: io_uring_queue_init failed: %s — falling back to pread\n",
                       __func__, strerror(-ret));
        return;
    }
    ring_ok_ = true;
}

llama_io_uring::~llama_io_uring() {
    if (!ring_ok_) return;
    if (bufs_registered_) io_uring_unregister_buffers(&ring_);
    io_uring_queue_exit(&ring_);
}

bool llama_io_uring::register_buffers(struct iovec * bufs, int n_bufs) {
    if (!ring_ok_) return false;
    int ret = io_uring_register_buffers(&ring_, bufs, n_bufs);
    if (ret < 0) {
        LLAMA_LOG_WARN("%s: io_uring_register_buffers failed: %s\n",
                       __func__, strerror(-ret));
        return false;
    }
    bufs_registered_ = true;
    return true;
}

void llama_io_uring::submit_read(int buf_idx, void * buf, size_t size,
                                  uint64_t offset, uint64_t user_data) {
    struct io_uring_sqe * sqe = io_uring_get_sqe(&ring_);
    if (!sqe) {
        // Ring full — flush and retry
        io_uring_submit(&ring_);
        sqe = io_uring_get_sqe(&ring_);
        if (!sqe) return;
    }

    if (bufs_registered_) {
        io_uring_prep_read_fixed(sqe, fd_, buf, (unsigned)size,
                                 (off_t)offset, buf_idx);
    } else {
        io_uring_prep_read(sqe, fd_, buf, (unsigned)size, (off_t)offset);
    }
    sqe->user_data = user_data;
    pending_++;
}

void llama_io_uring::submit() {
    if (pending_ > 0) io_uring_submit(&ring_);
}

uint64_t llama_io_uring::wait_one(int * bytes_read) {
    struct io_uring_cqe * cqe = nullptr;
    int ret = io_uring_wait_cqe(&ring_, &cqe);
    if (ret < 0 || !cqe) {
        if (bytes_read) *bytes_read = ret;
        return UINT64_MAX;
    }
    uint64_t ud = cqe->user_data;
    if (bytes_read) *bytes_read = cqe->res;
    io_uring_cqe_seen(&ring_, cqe);
    pending_--;
    return ud;
}

#endif // LLAMA_HAVE_IO_URING
