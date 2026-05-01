#pragma once

// KvtcStore — cold-tier SSD storage for evicted KV / recurrent state.
//
// One file per pager-process lifetime, append-only. An in-memory index
// maps (kind, layer, key) -> { file_offset, on_disk_bytes,
// original_bytes, compression } so reads can find their bytes.
//
// File format (KVTC v2):
//   [file_header]                      24 bytes
//     magic            uint32  = 0x4B565443 ("KVTC")
//     version          uint32  = 2
//     reserved[4]      uint64
//   [section payloads]                 contiguous, append-only
//
// The on-disk format does NOT include section metadata interleaved
// with payloads; the index is the source of truth and lives in memory.
// A crash loses the index and effectively the file (we'd just re-evict
// from the model). Phase 2 follow-up may persist the index in a footer
// for clean restarts.
//
// Compression policy:
//   - Attention K and V: per `cfg.compression` (None / Int4 / Int8).
//     INT4 is the default for cold storage; ~4x compression vs FP16
//     with quality acceptable for tier restoration.
//   - Recurrent r/s: always stored as raw FP. Recurrent state's
//     information density makes quantization unsafe.
//
// Threadsafe via internal mutex. All read/write operations block on the
// mutex; concurrency is not yet exploited (Phase 2 follow-up).
//
// Stateless w.r.t. the GPU: the store sees only host-side bytes.
// Movers do device <-> host before calling write_*, and host -> device
// after read_*.

#include "mt-config.h"
#include "llama.h"      // llama_seq_id, llama_pos

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mt {

class KvtcStore {
public:
    static constexpr uint32_t kMagic   = 0x4B565443;  // "KVTC"
    static constexpr uint32_t kVersion = 2;

    KvtcStore() = default;
    ~KvtcStore();

    KvtcStore(const KvtcStore &)             = delete;
    KvtcStore & operator=(const KvtcStore &) = delete;

    // Open or create the file at `path`. The file is truncated on init
    // (we never re-use an existing cold tier from a previous run).
    // Returns false on I/O failure.
    bool init(const std::string & path, Compression compression);

    // Close the file and clear the index. Safe to call multiple times.
    void shutdown();

    // ---- attention K/V ----
    //
    // `bytes` is the size of the FP source. Compression is applied
    // automatically per cfg. Reads decompress into the caller's buffer
    // (which must be `bytes` long).

    bool write_attn_k(int layer, llama_pos pos, const void * src, size_t bytes);
    bool write_attn_v(int layer, llama_pos pos, const void * src, size_t bytes);

    bool read_attn_k(int layer, llama_pos pos, void * dst, size_t bytes) const;
    bool read_attn_v(int layer, llama_pos pos, void * dst, size_t bytes) const;

    // Drop a stored entry. Frees the index slot but does NOT reclaim
    // file space (append-only; reuse comes via Phase 2 follow-up
    // compaction). Returns true if an entry existed and was removed.
    bool erase_attn(int layer, llama_pos pos);

    // ---- recurrent state ----
    //
    // is_s = true: SSM state ("s"); false: convolution state ("r").
    // Always stored uncompressed (raw FP).

    bool write_recurrent(int layer, llama_seq_id seq, bool is_s,
                          const void * src, size_t bytes);

    bool read_recurrent(int layer, llama_seq_id seq, bool is_s,
                         void * dst, size_t bytes) const;

    bool erase_recurrent(int layer, llama_seq_id seq, bool is_s);

    // ---- inspection ----

    bool         is_initialized() const;
    size_t       n_entries()      const;
    uint64_t     bytes_written()  const;  // total payload bytes appended (excludes header)
    Compression  compression()    const;
    const std::string & path()    const { return path_; }

private:
    enum class Kind : uint8_t {
        AttnK     = 0,
        AttnV     = 1,
        RecurR    = 2,
        RecurS    = 3,
    };

    struct Key {
        Kind        kind   = Kind::AttnK;
        int         layer  = 0;
        int64_t     pos_or_seq = 0;  // pos for Attn*, seq_id for Recur*

        bool operator==(const Key & o) const {
            return kind == o.kind && layer == o.layer && pos_or_seq == o.pos_or_seq;
        }
    };
    struct KeyHash {
        size_t operator()(const Key & k) const noexcept {
            return std::hash<int>{}((int) k.kind)
                 ^ (std::hash<int>{}(k.layer) << 1)
                 ^ (std::hash<int64_t>{}(k.pos_or_seq) << 2);
        }
    };

    struct Entry {
        uint64_t    file_offset    = 0;
        uint32_t    on_disk_bytes  = 0;
        uint32_t    original_bytes = 0;
        Compression compression    = Compression::None;
    };

    bool write_entry_(Kind kind, int layer, int64_t key,
                       const void * src, size_t bytes,
                       Compression comp_for_payload);
    bool read_entry_(Kind kind, int layer, int64_t key,
                      void * dst, size_t bytes) const;
    bool erase_entry_(Kind kind, int layer, int64_t key);

    mutable std::mutex                              mu_;
    std::unordered_map<Key, Entry, KeyHash>         index_;
    std::string                                      path_;
    int                                              fd_      = -1;
    uint64_t                                         eof_     = 0;  // current end-of-file
    Compression                                      cfg_compression_ = Compression::Int4;
    bool                                             initialized_     = false;
};

}  // namespace mt
