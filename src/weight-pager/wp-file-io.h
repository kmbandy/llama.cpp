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
#include <unordered_map>
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

enum class FileIOTransport {
    SyncPread,
    IoUringHost,
    IoUringP2P,
};

struct IoResult {
    uint64_t req_id     = 0;
    IoStatus status     = IoStatus::ErrorNoSubmit;
    int      bytes_read = 0;   // negative on ErrorIo (== -errno)
};

// MAD-235 — one request descriptor in a batch submission. Same fields as
// the single-shot submit() call. Caller fills a vector of these and the
// FileIOLayer prepares all SQEs in one pass before flushing.
struct FileIOBatchRequest {
    uint64_t req_id;
    int      fd_idx;
    uint64_t offset;
    size_t   size;
    void *   dst;
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

    // Wait for the next completion of ANY in-flight request. timeout_ms < 0 =
    // wait indefinitely; 0 = poll non-blocking. On timeout returns
    // IoStatus::Timeout and the request stays in flight.
    //
    // Completion demux: this layer buffers reaped-but-unclaimed completions
    // in `ready_` keyed by req_id. wait_any drains that buffer first, then
    // reaps one fresh completion from the transport. Concrete (not virtual) —
    // subclasses supply the raw reap via reap_raw_().
    IoResult wait_any(int timeout_ms = -1);

    // Wait for the completion of ONE specific req_id. Foreign completions
    // reaped while waiting are BUFFERED (never discarded), so their owner can
    // still claim them via a later wait_for_req()/try_take()/wait_any(). This
    // is the safe primitive when multiple logical consumers share one ring:
    // it is the fix for the shared-ring cross-drain (a targeted waiter used
    // to cqe_seen and drop another consumer's completion, hanging its slot).
    // On timeout returns IoStatus::Timeout; the request stays in flight.
    IoResult wait_for_req(uint64_t req_id, int timeout_ms = -1);

    // Non-blocking claim of a specific req_id. Drains any immediately-ready
    // completions from the transport into the buffer, then removes and
    // returns the one matching `req_id` if present. Returns false (leaving
    // `out` untouched) when that req_id has not completed yet. Never blocks,
    // never discards other consumers' completions.
    bool try_take(uint64_t req_id, IoResult & out);

    // Wait until every req_id in `ids` has completed. Completions may arrive
    // out of order; foreign req_ids (other consumers on the shared ring) are
    // buffered in ready_ and never dropped. Results are written into `outs`
    // in the same order as `ids`. Returns false only on transport-level
    // failure (req_id==0 ErrorIo); individual short/error results are still
    // placed in outs.
    bool wait_for_reqs(const std::vector<uint64_t> & ids,
                       std::vector<IoResult> &       outs,
                       int                           timeout_ms = -1);

    // How many requests are currently in flight (submitted, not yet reaped).
    virtual int pending() const = 0;

    // Return the lower-level fd for a given index, or -1 if out of range.
    // Exposed for callers that need direct pread fallback (e.g. tests or
    // failure recovery). The layer retains ownership.
    virtual int fd(int fd_idx) const = 0;

    // Hint to the kernel that we will need [offset, offset+size) from
    // `fd_idx` soon (POSIX_FADV_WILLNEED on Linux). Cheap, idempotent at
    // the kernel level, no completion is produced. Use to warm page cache
    // ahead of the next K layers' tensor reads — subsequent submit()s
    // against advised ranges hit warm page cache (~10 GB/s memcpy) instead
    // of cold NVMe (~500 MB/s QD=1 ceiling).
    //
    // RAM cost: each advised range becomes page cache pressure of `size`
    // bytes. Operators on RAM-tight systems should keep WP_FADVISE_LOOKAHEAD
    // low (1) or 0 to disable.
    //
    // Default no-op so non-file-backed FileIOLayer impls (future: in-memory,
    // network) are forward-compatible without forcing a stub.
    virtual void advise_prefetch(int /*fd_idx*/, uint64_t /*offset*/, size_t /*size*/) {}

    // MAD-235 — batch-submit a vector of reads in one shot. Returns the
    // number successfully queued (0..reqs.size()). On partial success the
    // SUCCEEDED prefix [0, returned) will produce completions normally;
    // the remainder [returned, N) was rejected and the caller MUST treat
    // them as terminal failures (no completion will arrive).
    //
    // The io_uring override prepares all SQEs first then calls
    // io_uring_submit once — collapses N syscalls to 1 for a single
    // MoE layer's expert-set fetch.
    //
    // Default impl loops calling submit() — semantically equivalent for
    // non-io_uring backends; the SQE-batching syscall savings only apply
    // when the underlying transport is async with a submission queue.
    virtual int submit_batch(const std::vector<FileIOBatchRequest> & reqs);

    // Which transport is currently active. P2P layers may downgrade at
    // runtime after a transport failure; callers should query this at the
    // point where they choose the read destination.
    virtual FileIOTransport transport() const = 0;
    virtual bool direct_to_device() const { return false; }

protected:
    // Reaped-but-unclaimed completions, keyed by req_id. Populated by the
    // demux methods above when they encounter a completion whose owner has
    // not yet asked for it. Subclasses count these in pending().
    std::unordered_map<uint64_t, IoResult> ready_;

    // Reap exactly ONE raw completion from the underlying transport, blocking
    // up to timeout_ms (<0 = block indefinitely, 0 = non-blocking poll).
    // Returns true and fills `out` when a completion (including a per-request
    // error) is reaped; returns false on timeout / no completion available.
    // A transport-level failure (not tied to a request) is reported by
    // returning true with out.status == ErrorIo and out.req_id == 0.
    // Implementations MUST decrement their own in-flight counter here.
    virtual bool reap_raw_(int timeout_ms, IoResult & out) = 0;
};

struct FileIOP2PConfig {
    void * pool_base = nullptr;
    size_t pool_size = 0;
};

// Factory. `fds` is a list of pre-prepared file descriptors (typically dup'd
// from the model loader's fds with O_DIRECT cleared via
// dup_clear_o_direct). The layer takes ownership of the fds (closes them on
// destruction). `prefer_async` requests the io_uring path; if liburing is
// unavailable or initialization fails, falls back to SyncPread silently.
std::unique_ptr<FileIOLayer> create_file_io(std::vector<int> fds,
                                            bool             prefer_async,
                                            int              queue_depth = 8,
                                            const FileIOP2PConfig * p2p = nullptr);

// Host-only factory used by the P2P layer after runtime downgrade. It follows
// the existing io_uring-host -> SyncPread ladder and never tries P2P.
std::unique_ptr<FileIOLayer> create_host_file_io(std::vector<int> fds,
                                                 bool             prefer_async,
                                                 int              queue_depth = 8);

// P2P factory implemented in wp-file-io-p2p.cpp. Non-HIP or non-io_uring
// builds compile a stub that returns nullptr.
std::unique_ptr<FileIOLayer> create_p2p_file_io(std::vector<int> fds,
                                                bool             prefer_async,
                                                int              queue_depth,
                                                const FileIOP2PConfig & cfg);

// Helper: dup `src_fd` and clear O_DIRECT on the result. Returns the new fd
// or -1 on failure. Callers should use this when handing fds to FileIOLayer
// because GGUF tensor offsets are not sector-aligned and O_DIRECT would
// silently round reads to the prior 512-byte boundary on some filesystems
// (bug B-P3 in docs/dev/memory-tier-bug-catalog.md).
int dup_clear_o_direct(int src_fd);

}  // namespace wp
