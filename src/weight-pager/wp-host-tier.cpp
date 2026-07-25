#include "wp-host-tier.h"

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

bool HostTier::init(size_t budget_bytes, int device_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ != nullptr && budget_bytes_ > 0) {
        LLAMA_LOG_WARN("wp::HostTier: init called twice\n");
        return false;
    }
    if (budget_bytes == 0) {
        return false;
    }

    budget_bytes_ = budget_bytes;

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

    arena_ = nullptr;
    budget_bytes_ = 0;
    backend_pinned_ = false;
    mlocked_ = false;
}

bool HostTier::contains(int page_idx) const {
    std::lock_guard<std::mutex> lock(mu_);
    return resident_.find(page_idx) != resident_.end();
}

bool HostTier::store(int page_idx, const void * src_bytes, size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    if (arena_ == nullptr || budget_bytes_ == 0 || page_idx < 0 || src_bytes == nullptr || n == 0) {
        return false;
    }
    if (n > budget_bytes_) {
        erase_resident_(page_idx);
        return false;
    }

    erase_resident_(page_idx);

    size_t offset = 0;
    if (!acquire_slot_(page_idx, n, offset)) {
        return false;
    }

    std::memcpy(arena_ + offset, src_bytes, n);
    resident_[page_idx] = Resident{offset, n, /*borrow_count=*/0, next_gen_++};
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
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
    erase_resident_(page_idx);
    size_t offset = 0;
    if (!acquire_slot_(page_idx, n, offset)) {
        return false;
    }
    // Synchronous D2H: the caller (on_pool_evict_) has already synchronized any
    // in-flight transfer for this page, so the device slot is settled here.
    hipError_t err = hipMemcpy(arena_ + offset, device_bytes, n, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::HostTier::store_from_device: hipMemcpy D2H(%zu) page %d failed: %s\n",
                       n, page_idx, hipGetErrorString(err));
        free_lists_[n].push_back(offset);  // return the acquired slot
        return false;
    }
    resident_[page_idx] = Resident{offset, n, /*borrow_count=*/0, next_gen_++};
    used_bytes_ += n;
    lru_.push_back(page_idx);
    lru_pos_[page_idx] = std::prev(lru_.end());
    return true;
#else
    (void) device_bytes;
    return false;
#endif
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
        return true;
    }
    return false;
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
