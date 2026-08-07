#include "wp-host-tier.h"

#include "wp-gpu-transport.h"
#include "wp-pool.h"      // is_uma_device
#include "llama-impl.h"  // LLAMA_LOG_*

#include <cerrno>
#include <iterator>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "wp-gpu-runtime.h"

namespace wp {

namespace {

bool env_flag_is_one(const char * var) {
    const char * v = std::getenv(var);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

}  // anonymous namespace

HostTier::~HostTier() {
    shutdown();
}

bool HostTier::init(size_t budget_bytes, int device_idx, GpuTransport * transport) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ != nullptr && budget_bytes_ > 0) {
        LLAMA_LOG_WARN("wp::HostTier: init called twice\n");
        return false;
    }
    if (budget_bytes == 0) {
        return false;
    }

    budget_bytes_ = budget_bytes;
    transport_ = transport;

    if (transport_ != nullptr && transport_->is_vulkan()) {
        arena_ = (uint8_t *) transport_->host_alloc(budget_bytes_);
        backend_pinned_ = arena_ != nullptr;
        transport_pinned_ = backend_pinned_;
    }
    if (arena_ == nullptr && transport_ != nullptr && transport_->is_vulkan()) {
        arena_ = (uint8_t *) std::malloc(budget_bytes_);
    }
    if (arena_ == nullptr) {
// Compile-time: the HIP/CUDA host-allocation API is absent from other builds.
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
        void * p = nullptr;
        hipError_t err = hipHostMalloc(&p, budget_bytes_, hipHostMallocDefault);
        if (err == hipSuccess) {
            arena_ = (uint8_t *) p;
            backend_pinned_ = true;
        } else {
            LLAMA_LOG_WARN("wp::HostTier: hipHostMalloc(%zu) failed: %s; falling back to malloc\n",
                           budget_bytes_, hipGetErrorString(err));
            arena_ = (uint8_t *) std::malloc(budget_bytes_);
            backend_pinned_ = false;
        }
#else
        (void) device_idx;
        arena_ = (uint8_t *) std::malloc(budget_bytes_);
        backend_pinned_ = false;
#endif
    }

    if (arena_ == nullptr) {
        LLAMA_LOG_WARN("wp::HostTier::init: allocation of %zu bytes failed\n", budget_bytes_);
        budget_bytes_ = 0;
        return false;
    }

    if (env_flag_is_one("WP_PIN_HOST")) {
        if (is_uma_device(device_idx)) {
            LLAMA_LOG_WARN("wp::HostTier: WP_PIN_HOST=1 ignored on UMA/APU device; host RAM and VRAM share physical memory\n");
        } else {
#if defined(__linux__)
            if (mlock(arena_, budget_bytes_) == 0) {
                mlocked_ = true;
            } else {
                const int e = errno;
                if (e == EPERM || e == ENOMEM) {
                    LLAMA_LOG_WARN("wp::HostTier: mlock(%zu) denied (%s); continuing without mlock. Try `ulimit -l unlimited` if pinning is required.\n",
                                   budget_bytes_, std::strerror(e));
                } else {
                    LLAMA_LOG_WARN("wp::HostTier: mlock(%zu) failed (%s); continuing without mlock\n",
                                   budget_bytes_, std::strerror(e));
                }
            }
#else
            LLAMA_LOG_WARN("wp::HostTier: WP_PIN_HOST=1 requested, but mlock is unavailable on this platform\n");
#endif
        }
    }

    LLAMA_LOG_INFO("wp::HostTier: enabled, budget=%zu bytes (%.1f MiB), backend_pinned=%d, mlocked=%d\n",
                   budget_bytes_, (double) budget_bytes_ / 1048576.0,
                   (int) backend_pinned_, (int) mlocked_);
    return true;
}

void HostTier::shutdown() {
    std::lock_guard<std::mutex> lock(mu_);
    resident_.clear();
    pending_.clear();
    free_lists_.clear();
    lru_.clear();
    lru_pos_.clear();
    used_bytes_ = 0;
    high_water_ = 0;

    if (arena_ != nullptr) {
#if defined(__linux__)
        if (mlocked_) {
            munlock(arena_, budget_bytes_);
        }
#endif
        if (transport_pinned_ && transport_ != nullptr) {
            transport_->host_free(arena_);
        } else {
// Compile-time: the HIP/CUDA host-free API is absent from other builds.
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
        if (backend_pinned_) {
            (void) hipHostFree(arena_);
        } else {
            std::free(arena_);
        }
#else
        std::free(arena_);
#endif
        }
    }

    arena_ = nullptr;
    budget_bytes_ = 0;
    backend_pinned_ = false;
    transport_ = nullptr;
    transport_pinned_ = false;
    mlocked_ = false;
}

bool HostTier::contains(int page_idx) const {
    std::lock_guard<std::mutex> lock(mu_);
    return resident_.find(page_idx) != resident_.end();
}

bool HostTier::store(int page_idx, const void * src_bytes, size_t n, bool speculative) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ == nullptr || budget_bytes_ == 0 || page_idx < 0 || src_bytes == nullptr || n == 0) {
        return false;
    }
    if (n > budget_bytes_) {
        erase_resident_(page_idx);
        return false;
    }

    // Never DOWNGRADE a page that is already here as a victim (a page the GPU
    // actually used) into a speculative one. The prefetch call site skips
    // pages the tier already contains, so this should not arise -- but if it
    // ever does, a re-store must not cost a confirmed page its priority.
    if (speculative) {
        auto prev = resident_.find(page_idx);
        if (prev != resident_.end() && !prev->second.speculative) {
            speculative = false;
        }
    }

    erase_resident_(page_idx);

    size_t offset = 0;
    if (!acquire_slot_(page_idx, n, offset)) {
        return false;
    }

    std::memcpy(arena_ + offset, src_bytes, n);
    resident_[page_idx] = Resident{offset, n, /*borrow_count=*/0, next_gen_++, speculative};
    if (speculative) ++spec_count_;
    used_bytes_ += n;
    lru_.push_back(page_idx);
    lru_pos_[page_idx] = std::prev(lru_.end());
    return true;
}

void HostTier::erase(int page_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    if (page_idx < 0) {
        return;
    }
    erase_resident_(page_idx);
}

bool HostTier::store_from_device(int page_idx, const void * device_bytes, size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ == nullptr || budget_bytes_ == 0 || page_idx < 0 || device_bytes == nullptr || n == 0) {
        return false;
    }
    if (n > budget_bytes_) {
        erase_resident_(page_idx);
        return false;
    }
    // Refuse rather than guess. Without a reader the only thing available is a
    // raw hipMemcpy, and that is WRONG for any pool whose pointers are not real
    // device addresses -- notably Vulkan, where they are a sentinel base plus an
    // offset. Note this file is compiled with -DGGML_USE_CUDA even in
    // Vulkan-only configurations, so the preprocessor cannot be used to decide
    // this; only the owner knows what the pool actually is.
    if (!device_reader_) {
// Compile-time: the raw HIP/CUDA fallback API is absent from other builds.
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
        // Fall through to the built-in copy below (raw-addressable pool).
#else
        (void) device_bytes;
        return false;
#endif
    }

    erase_resident_(page_idx);
    size_t offset = 0;
    if (!acquire_slot_(page_idx, n, offset)) {
        return false;
    }

    // Synchronous D2H: the caller (on_pool_evict_) has already synchronized any
    // in-flight transfer for this page, so the device slot is settled here.
    bool ok;
    if (device_reader_) {
        ok = device_reader_(arena_ + offset, device_bytes, n, page_idx);
        if (!ok) {
            LLAMA_LOG_WARN("wp::HostTier::store_from_device: device read D2H(%zu) page %d failed\n",
                           n, page_idx);
        }
    } else {
// Compile-time: the raw HIP/CUDA fallback API is absent from other builds.
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
        hipError_t err = hipMemcpy(arena_ + offset, device_bytes, n, hipMemcpyDeviceToHost);
        ok = err == hipSuccess;
        if (!ok) {
            LLAMA_LOG_WARN("wp::HostTier::store_from_device: hipMemcpy D2H(%zu) page %d failed: %s\n",
                           n, page_idx, hipGetErrorString(err));
        }
#else
        ok = false;
#endif
    }
    if (!ok) {
        free_lists_[n].push_back(offset);  // return the acquired slot
        return false;
    }

    resident_[page_idx] = Resident{offset, n, /*borrow_count=*/0, next_gen_++};
    used_bytes_ += n;
    lru_.push_back(page_idx);
    lru_pos_[page_idx] = std::prev(lru_.end());
    return true;
}

bool HostTier::lookup(int page_idx, void * dst_bytes, size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ == nullptr || budget_bytes_ == 0 || page_idx < 0 || dst_bytes == nullptr || n == 0) {
        return false;
    }
    auto it = resident_.find(page_idx);
    if (it == resident_.end() || it->second.bytes != n) {
        return false;
    }
    touch_lru_(page_idx);
    promote_(it->second);   // a demand hit confirms the prediction
    std::memcpy(dst_bytes, arena_ + it->second.offset, n);
    return true;
}

bool HostTier::borrow(int page_idx, const void ** src_out, size_t n, BorrowHandle * handle_out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ == nullptr || budget_bytes_ == 0 || page_idx < 0 || src_out == nullptr ||
        handle_out == nullptr || n == 0) {
        return false;
    }
    auto it = resident_.find(page_idx);
    if (it == resident_.end() || it->second.bytes != n) {
        return false;
    }
    touch_lru_(page_idx);
    promote_(it->second);   // a demand hit confirms the prediction
    it->second.borrow_count++;
    *src_out    = arena_ + it->second.offset;
    *handle_out = it->second.gen;
    return true;
}

void HostTier::release(int page_idx, BorrowHandle handle) {
    std::lock_guard<std::mutex> lock(mu_);
    if (page_idx < 0 || handle == kInvalidBorrowHandle) {
        return;
    }

    // The handle names the EXACT entry generation borrow() saw, so there is
    // no ambiguity even if this page_idx has since been erase()'d and
    // re-store()'d (or evicted) any number of times: check whether the
    // CURRENT resident_[page_idx] entry (if any) is that generation first,
    // then fall back to pending_ (entries retired-while-borrowed live there,
    // keyed by their own generation, never by page_idx).
    auto it = resident_.find(page_idx);
    if (it != resident_.end() && it->second.gen == handle) {
        if (it->second.borrow_count > 0) {
            it->second.borrow_count--;
        }
        return;
    }

    auto pit = pending_.find(handle);
    if (pit != pending_.end()) {
        Resident & r = pit->second;
        if (r.borrow_count > 0) {
            r.borrow_count--;
        }
        if (r.borrow_count == 0) {
            reclaim_(r);
            pending_.erase(pit);
        }
        return;
    }

    // Neither: a stale/double-released handle. No-op.
}

bool HostTier::is_initialized() const {
    std::lock_guard<std::mutex> lock(mu_);
    return arena_ != nullptr && budget_bytes_ > 0;
}

size_t HostTier::budget_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return budget_bytes_;
}

size_t HostTier::used_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return used_bytes_;
}

size_t HostTier::high_water() const {
    std::lock_guard<std::mutex> lock(mu_);
    return high_water_;
}

size_t HostTier::resident_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return resident_.size();
}

bool HostTier::backend_pinned() const {
    std::lock_guard<std::mutex> lock(mu_);
    return backend_pinned_;
}

bool HostTier::mlocked() const {
    std::lock_guard<std::mutex> lock(mu_);
    return mlocked_;
}

bool HostTier::acquire_slot_(int page_idx, size_t n, size_t & offset_out) {
    if (n > budget_bytes_) {
        return false;
    }

    for (;;) {
        auto & slots = free_lists_[n];
        if (!slots.empty()) {
            offset_out = slots.back();
            slots.pop_back();
            return true;
        }

        if (high_water_ <= budget_bytes_ && n <= budget_bytes_ - high_water_) {
            offset_out = high_water_;
            high_water_ += n;
            return true;
        }

        if (!evict_one_lru_()) {
            LLAMA_LOG_WARN("wp::HostTier::store: could not acquire %zu-byte slot for page %d; arena is saturated by other size classes\n",
                           n, page_idx);
            return false;
        }
    }
}

bool HostTier::evict_one_lru_() {
    // Pass 0 (only when the speculative tier is enabled): drain the LRU
    // SPECULATIVE entry before touching anything the GPU actually used.
    //
    // A speculative entry is an unconfirmed prediction; a victim entry is a
    // page VRAM demonstrably touched. On one flat LRU the prediction lands at
    // the MRU end and outranks the victim, so a wrong guess evicts a known-good
    // page -- prefetch actively degrading the tier it is meant to fill. This
    // pass makes a mispredict cost only the bandwidth that fetched it, which is
    // the same guarantee PoolAllocator::alloc_slot's Pass 0 gives in VRAM.
    //
    // Borrowed entries are skipped here for exactly the reason they are skipped
    // below: their arena bytes are in flight to a caller.
    if (spec_tier_ && spec_count_ > 0) {
        for (auto lit = lru_.begin(); lit != lru_.end(); ++lit) {
            auto it = resident_.find(*lit);
            if (it == resident_.end() || it->second.borrow_count > 0 ||
                !it->second.speculative) {
                continue;
            }
            const Resident r = it->second;
            resident_.erase(it);
            lru_pos_.erase(*lit);
            lru_.erase(lit);
            reclaim_(r);
            if (spec_count_ > 0) --spec_count_;
            ++spec_evicted_unused_;
            return true;
        }
    }

    // Walk from the LRU front (least recently used) to the first entry with
    // no outstanding borrows and evict that one. A borrowed entry is skipped,
    // not removed -- its arena bytes are in flight to/from a caller and must
    // stay valid until release(). If every resident entry is borrowed, no
    // victim exists and the store this was called for fails cleanly (a
    // failed soft-prefetch store is a non-event by design).
    for (auto lit = lru_.begin(); lit != lru_.end(); ++lit) {
        const int page_idx = *lit;
        auto it = resident_.find(page_idx);
        if (it == resident_.end() || it->second.borrow_count > 0) {
            continue;
        }

        const Resident r = it->second;
        resident_.erase(it);
        lru_pos_.erase(page_idx);
        lru_.erase(lit);
        free_lists_[r.bytes].push_back(r.offset);
        used_bytes_ = used_bytes_ >= r.bytes ? used_bytes_ - r.bytes : 0;
        if (r.speculative) {
            if (spec_count_ > 0) --spec_count_;
            ++spec_evicted_unused_;
        }
        return true;
    }
    return false;
}

void HostTier::promote_(Resident & r) {
    // Landing is NOT use. A prefetch that merely completed stays speculative;
    // only a genuine demand hit clears the flag. Getting this backwards is
    // exactly VRAM gate 3, where harvest called mark_used() and promoted every
    // prefetched page the instant its read completed, so the tier could never
    // accumulate and eviction fell straight onto the demand set.
    if (!r.speculative) {
        return;
    }
    r.speculative = false;
    if (spec_count_ > 0) --spec_count_;
    ++spec_promotions_;
}

void HostTier::set_speculative_tier(bool on) {
    std::lock_guard<std::mutex> lock(mu_);
    spec_tier_ = on;
}

size_t HostTier::speculative_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return spec_count_;
}

uint64_t HostTier::speculative_evicted_unused() const {
    std::lock_guard<std::mutex> lock(mu_);
    return spec_evicted_unused_;
}

uint64_t HostTier::speculative_promotions() const {
    std::lock_guard<std::mutex> lock(mu_);
    return spec_promotions_;
}

void HostTier::reclaim_(const Resident & r) {
    free_lists_[r.bytes].push_back(r.offset);
    used_bytes_ = used_bytes_ >= r.bytes ? used_bytes_ - r.bytes : 0;
}

void HostTier::erase_resident_(int page_idx) {
    auto it = resident_.find(page_idx);
    if (it == resident_.end()) {
        return;
    }

    const Resident r = it->second;
    resident_.erase(it);
    // Leaving resident_ by ANY route must keep spec_count_ honest, or the
    // Pass-0 `spec_count_ > 0` guard drifts and starts scanning for entries
    // that no longer exist. Not counted as evicted-unused: this is erase() /
    // promotion-back-to-VRAM / displacement by re-store, not eviction.
    if (r.speculative && spec_count_ > 0) {
        --spec_count_;
    }

    auto pos_it = lru_pos_.find(page_idx);
    if (pos_it != lru_pos_.end()) {
        lru_.erase(pos_it->second);
        lru_pos_.erase(pos_it);
    }

    if (r.borrow_count > 0) {
        // Deferred retirement: gone from resident_/lru_ (contains() is
        // false, no aliasing on a re-store) but the slot itself is withheld
        // from free_lists_ until release() drains the last outstanding
        // borrow on it. Keyed by this entry's OWN generation handle so a
        // later borrow()/release() pair on a NEW entry for the same
        // page_idx can never collide with it.
        pending_[r.gen] = r;
        return;
    }

    reclaim_(r);
}

void HostTier::touch_lru_(int page_idx) {
    auto pos_it = lru_pos_.find(page_idx);
    if (pos_it == lru_pos_.end()) {
        return;
    }
    // Move the node to the back (MRU) in O(1) without invalidating any
    // other iterator, using splice on the same list instance.
    lru_.splice(lru_.end(), lru_, pos_it->second);
}

}  // namespace wp
