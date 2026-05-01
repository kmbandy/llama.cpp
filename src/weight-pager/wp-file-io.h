#pragma once

// FileIOLayer — abstract NVMe→host read interface for the weight pager.
//
// Two implementations:
//   - SyncPread     : pread, always available, no liburing dependency.
//   - IoUringAsync  : io_uring with registered files, available on Linux
//                     when LLAMA_HAVE_IO_URING is defined.
//
// Single API across both. The caller submits reads keyed by an opaque
// monotonic `req_id` of its choosing; the layer guarantees `req_id` is
// returned verbatim on completion. **req_id is the source of truth for
// completion routing.** The previous pager kept this state via the
// page index in io_uring's user_data field, which broke when in-flight
// requests completed out of order — bug B-P6 in
// docs/dev/memory-tier-bug-catalog.md.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace wp {

// Status flags for a completed request.
enum class IoStatus {
    Ok,                  // read completed, bytes_read == requested size
    Short,               // read completed but bytes_read < requested size
    ErrorIo,             // -errno from kernel
    ErrorNoSubmit,       // request was rejected at submit (queue full, fd invalid)
    Timeout,             // wait_any timed out; no request reaped, queue unchanged
};

struct IoResult {
    uint64_t req_id     = 0;
    IoStatus status     = IoStatus::ErrorNoSubmit;
    int      bytes_read = 0;   // negative on ErrorIo (== -errno)
};

class FileIOLayer {
public:
    virtual ~FileIOLayer() = default;

    // Queue a read of `size` bytes from file `fd_idx` starting at `offset`,
    // landing in `dst` (host-side, pinned for IoUringAsync). The req_id is
    // returned verbatim by wait_any() on completion. Caller owns dst.
    //
    // Returns false if the request could not be queued (queue full, fd_idx
    // out of range, dst null). In that case the layer never produces a
    // matching completion for req_id — the caller MUST treat unqueued
    // requests as terminal failures.
    virtual bool submit(uint64_t req_id,
                        int      fd_idx,
                        uint64_t offset,
                        size_t   size,
                        void *   dst) = 0;

    // Push any pending submissions to the kernel. SyncPread is a no-op.
    virtual void flush() = 0;

    // Wait for the next completion. timeout_ms < 0 = wait indefinitely;
    // 0 = poll non-blocking. On timeout returns IoStatus::Timeout and the
    // request stays in flight. On indefinite wait this only returns when a
    // completion arrives (or on internal error).
    virtual IoResult wait_any(int timeout_ms = -1) = 0;

    // How many requests are currently in flight (submitted, not yet reaped).
    virtual int pending() const = 0;

    // Return the lower-level fd for a given index, or -1 if out of range.
    // Exposed for callers that need direct pread fallback (e.g. tests or
    // failure recovery). The layer retains ownership.
    virtual int fd(int fd_idx) const = 0;
};

// Factory. `fds` is a list of pre-prepared file descriptors (typically dup'd
// from the model loader's fds with O_DIRECT cleared via
// dup_clear_o_direct). The layer takes ownership of the fds (closes them on
// destruction). `prefer_async` requests the io_uring path; if liburing is
// unavailable or initialization fails, falls back to SyncPread silently.
std::unique_ptr<FileIOLayer> create_file_io(std::vector<int> fds,
                                            bool             prefer_async,
                                            int              queue_depth = 8);

// Helper: dup `src_fd` and clear O_DIRECT on the result. Returns the new fd
// or -1 on failure. Callers should use this when handing fds to FileIOLayer
// because GGUF tensor offsets are not sector-aligned and O_DIRECT would
// silently round reads to the prior 512-byte boundary on some filesystems
// (bug B-P3 in docs/dev/memory-tier-bug-catalog.md).
int dup_clear_o_direct(int src_fd);

}  // namespace wp
