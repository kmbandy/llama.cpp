#include "mt-kvtc-store.h"

#include "mt-quant.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace mt {

namespace {

struct FileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t reserved0;
    uint64_t reserved1;
};
static_assert(sizeof(FileHeader) == 24, "FileHeader must be 24 bytes");

// pwrite-all loop: write `n` bytes from `src` at file offset `off`, retrying
// on partial writes / EINTR. Returns true on full success.
bool pwrite_all(int fd, const void * src, size_t n, uint64_t off) {
    const uint8_t * p = (const uint8_t *) src;
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t w = pwrite(fd, p, remaining, (off_t) off);
        if (w < 0) {
            if (errno == EINTR) continue;
            LLAMA_LOG_WARN("mt::KvtcStore: pwrite failed at off %lu: %s\n",
                           (unsigned long) off, std::strerror(errno));
            return false;
        }
        if (w == 0) {
            LLAMA_LOG_WARN("mt::KvtcStore: pwrite returned 0 (no progress) at off %lu\n",
                           (unsigned long) off);
            return false;
        }
        p         += w;
        remaining -= (size_t) w;
        off       += (uint64_t) w;
    }
    return true;
}

bool pread_all(int fd, void * dst, size_t n, uint64_t off) {
    uint8_t * p = (uint8_t *) dst;
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t r = pread(fd, p, remaining, (off_t) off);
        if (r < 0) {
            if (errno == EINTR) continue;
            LLAMA_LOG_WARN("mt::KvtcStore: pread failed at off %lu: %s\n",
                           (unsigned long) off, std::strerror(errno));
            return false;
        }
        if (r == 0) {
            LLAMA_LOG_WARN("mt::KvtcStore: pread short-read at off %lu (wanted %zu)\n",
                           (unsigned long) off, n);
            return false;
        }
        p         += r;
        remaining -= (size_t) r;
        off       += (uint64_t) r;
    }
    return true;
}

}  // namespace

KvtcStore::~KvtcStore() {
    shutdown();
}

bool KvtcStore::init(const std::string & path, Compression compression) {
    if (initialized_) {
        LLAMA_LOG_WARN("mt::KvtcStore: init called twice\n");
        return false;
    }
    path_            = path;
    cfg_compression_ = compression;

    // Truncate-or-create. The cold tier is per-run; we don't read any
    // pre-existing file.
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd_ < 0) {
        LLAMA_LOG_WARN("mt::KvtcStore: open(%s) failed: %s\n",
                       path.c_str(), std::strerror(errno));
        path_.clear();
        return false;
    }

    FileHeader h{};
    h.magic   = kMagic;
    h.version = kVersion;
    if (!pwrite_all(fd_, &h, sizeof(h), 0)) {
        ::close(fd_);
        fd_ = -1;
        path_.clear();
        return false;
    }
    eof_         = sizeof(h);
    initialized_ = true;

    LLAMA_LOG_INFO("mt::KvtcStore: opened %s (compression=%d)\n",
                   path.c_str(), (int) compression);
    return true;
}

void KvtcStore::shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    index_.clear();
    eof_         = 0;
    initialized_ = false;
}

bool KvtcStore::is_initialized() const {
    std::lock_guard<std::mutex> lk(mu_);
    return initialized_;
}

size_t KvtcStore::n_entries() const {
    std::lock_guard<std::mutex> lk(mu_);
    return index_.size();
}

uint64_t KvtcStore::bytes_written() const {
    std::lock_guard<std::mutex> lk(mu_);
    return eof_ > sizeof(FileHeader) ? eof_ - sizeof(FileHeader) : 0;
}

Compression KvtcStore::compression() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cfg_compression_;
}

// ---------------------------------------------------------------------------
// Internal write/read/erase
// ---------------------------------------------------------------------------

bool KvtcStore::write_entry_(Kind kind, int layer, int64_t key,
                              const void * src, size_t bytes,
                              Compression comp) {
    if (src == nullptr || bytes == 0) return false;

    std::lock_guard<std::mutex> lk(mu_);
    if (!initialized_ || fd_ < 0) return false;

    // Compress if requested. Source is interpreted as float for INT4/INT8.
    std::vector<uint8_t> compressed;
    const uint8_t * payload      = (const uint8_t *) src;
    size_t          payload_size = bytes;
    Compression     used_comp    = Compression::None;

    if (comp == Compression::Int4) {
        if (bytes % sizeof(float) != 0) {
            LLAMA_LOG_WARN("mt::KvtcStore::write: int4 path expects float-multiple, got %zu\n", bytes);
            return false;
        }
        const size_t n = bytes / sizeof(float);
        compressed     = quantize_int4((const float *) src, n);
        payload        = compressed.data();
        payload_size   = compressed.size();
        used_comp      = Compression::Int4;
    } else if (comp == Compression::Int8) {
        if (bytes % sizeof(float) != 0) {
            LLAMA_LOG_WARN("mt::KvtcStore::write: int8 path expects float-multiple, got %zu\n", bytes);
            return false;
        }
        const size_t n = bytes / sizeof(float);
        compressed     = quantize_int8((const float *) src, n);
        payload        = compressed.data();
        payload_size   = compressed.size();
        used_comp      = Compression::Int8;
    } else {
        used_comp = Compression::None;
    }

    const uint64_t off = eof_;
    if (!pwrite_all(fd_, payload, payload_size, off)) {
        return false;
    }
    eof_ += payload_size;

    Entry e;
    e.file_offset    = off;
    e.on_disk_bytes  = (uint32_t) payload_size;
    e.original_bytes = (uint32_t) bytes;
    e.compression    = used_comp;

    Key k{ kind, layer, key };
    index_[k] = e;
    return true;
}

bool KvtcStore::read_entry_(Kind kind, int layer, int64_t key,
                             void * dst, size_t bytes) const {
    if (dst == nullptr || bytes == 0) return false;

    std::lock_guard<std::mutex> lk(mu_);
    if (!initialized_ || fd_ < 0) return false;

    const Key k{ kind, layer, key };
    const auto it = index_.find(k);
    if (it == index_.end()) return false;
    const Entry & e = it->second;
    if (e.original_bytes != bytes) {
        LLAMA_LOG_WARN("mt::KvtcStore::read: size mismatch (have %u, requested %zu)\n",
                       e.original_bytes, bytes);
        return false;
    }

    if (e.compression == Compression::None) {
        return pread_all(fd_, dst, e.on_disk_bytes, e.file_offset);
    }

    // Need to read compressed bytes into a scratch buffer, then decompress.
    std::vector<uint8_t> scratch(e.on_disk_bytes);
    if (!pread_all(fd_, scratch.data(), scratch.size(), e.file_offset)) {
        return false;
    }
    const size_t n = bytes / sizeof(float);
    if (e.compression == Compression::Int4) {
        return dequantize_int4(scratch.data(), (float *) dst, n);
    }
    if (e.compression == Compression::Int8) {
        return dequantize_int8(scratch.data(), (float *) dst, n);
    }
    LLAMA_LOG_WARN("mt::KvtcStore::read: unsupported compression %d\n", (int) e.compression);
    return false;
}

bool KvtcStore::erase_entry_(Kind kind, int layer, int64_t key) {
    std::lock_guard<std::mutex> lk(mu_);
    return index_.erase(Key{ kind, layer, key }) > 0;
}

// ---------------------------------------------------------------------------
// Public attn API
// ---------------------------------------------------------------------------

bool KvtcStore::write_attn_k(int layer, llama_pos pos, const void * src, size_t bytes) {
    return write_entry_(Kind::AttnK, layer, (int64_t) pos, src, bytes, cfg_compression_);
}
bool KvtcStore::write_attn_v(int layer, llama_pos pos, const void * src, size_t bytes) {
    return write_entry_(Kind::AttnV, layer, (int64_t) pos, src, bytes, cfg_compression_);
}
bool KvtcStore::read_attn_k(int layer, llama_pos pos, void * dst, size_t bytes) const {
    return read_entry_(Kind::AttnK, layer, (int64_t) pos, dst, bytes);
}
bool KvtcStore::read_attn_v(int layer, llama_pos pos, void * dst, size_t bytes) const {
    return read_entry_(Kind::AttnV, layer, (int64_t) pos, dst, bytes);
}
bool KvtcStore::erase_attn(int layer, llama_pos pos) {
    // Erase both K and V entries (no short-circuit). Returns true if at
    // least one was present.
    const bool k = erase_entry_(Kind::AttnK, layer, (int64_t) pos);
    const bool v = erase_entry_(Kind::AttnV, layer, (int64_t) pos);
    return k || v;
}

// ---------------------------------------------------------------------------
// Public recurrent API (always uncompressed)
// ---------------------------------------------------------------------------

bool KvtcStore::write_recurrent(int layer, llama_seq_id seq, bool is_s,
                                 const void * src, size_t bytes) {
    return write_entry_(is_s ? Kind::RecurS : Kind::RecurR, layer,
                        (int64_t) seq, src, bytes, Compression::None);
}
bool KvtcStore::read_recurrent(int layer, llama_seq_id seq, bool is_s,
                                void * dst, size_t bytes) const {
    return read_entry_(is_s ? Kind::RecurS : Kind::RecurR, layer,
                       (int64_t) seq, dst, bytes);
}
bool KvtcStore::erase_recurrent(int layer, llama_seq_id seq, bool is_s) {
    return erase_entry_(is_s ? Kind::RecurS : Kind::RecurR, layer, (int64_t) seq);
}

}  // namespace mt
