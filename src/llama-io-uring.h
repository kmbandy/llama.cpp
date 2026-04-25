#pragma once
// io_uring-backed async NVMe reader for model weight loading.
// Drop-in replacement for the synchronous read_raw_unsafe path in
// llama_model_loader::load_all_data.  Overlaps NVMe reads with GPU DMA
// uploads to approach the SN850X sequential ceiling (~5.7 GB/s on RDNA4
// via SAM-mapped PCIe BAR).
//
// Only compiled on Linux when liburing is present (LLAMA_HAVE_IO_URING).

#ifdef LLAMA_HAVE_IO_URING

#include <cstddef>
#include <cstdint>
#include <liburing.h>

// Wraps a single io_uring instance tied to one file descriptor.
// Caller owns the staging buffers; this class only manages submissions
// and completions.
class llama_io_uring {
public:
    explicit llama_io_uring(int fd, int queue_depth = 64);
    ~llama_io_uring();

    llama_io_uring(const llama_io_uring &) = delete;
    llama_io_uring & operator=(const llama_io_uring &) = delete;

    // Register fixed buffers for zero-copy DMA into pinned host memory.
    // bufs / n_bufs must stay alive for the lifetime of this object.
    bool register_buffers(struct iovec * bufs, int n_bufs);

    // Submit a fixed-buffer read (buf_idx indexes into registered buffers).
    // offset is the absolute byte offset in the file.
    // user_data is returned verbatim in wait_one().
    void submit_read(int buf_idx, void * buf, size_t size, uint64_t offset,
                     uint64_t user_data);

    // Flush pending submissions to the kernel.
    void submit();

    // Wait for one completion.  Returns user_data; *bytes_read is set to the
    // result (negative = error code from -errno).
    uint64_t wait_one(int * bytes_read);

    bool valid() const { return ring_ok_; }

private:
    int       fd_;
    bool      ring_ok_ = false;
    bool      bufs_registered_ = false;
    int       pending_ = 0;
    struct io_uring ring_;
};

#endif // LLAMA_HAVE_IO_URING
