#include "wp-pool.h"

#include "ggml-backend.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <limits>

namespace wp {

PoolAllocator::~PoolAllocator() {
    if (buf_ != nullptr) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    base_ = nullptr;
}

bool PoolAllocator::init(ggml_backend_buffer_type_t buft,
                         int                        n_slots,
                         size_t                     slot_size) {
    if (buf_ != nullptr) {
        LLAMA_LOG_WARN("wp::PoolAllocator: init called twice — ignoring second call\n");
        return false;
    }
    if (buft == nullptr || n_slots <= 0 || slot_size == 0) {
        LLAMA_LOG_WARN("wp::PoolAllocator::init: invalid args (buft=%p, n_slots=%d, slot_size=%zu)\n",
                       (void *) buft, n_slots, slot_size);
        return false;
    }

    const size_t total = (size_t) n_slots * slot_size;
    buf_ = ggml_backend_buft_alloc_buffer(buft, total);
    if (buf_ == nullptr) {
        LLAMA_LOG_WARN("wp::PoolAllocator::init: ggml_backend_buft_alloc_buffer(%zu B) failed\n", total);
        return false;
    }
    base_ = ggml_backend_buffer_get_base(buf_);
    if (base_ == nullptr) {
        LLAMA_LOG_WARN("wp::PoolAllocator::init: ggml_backend_buffer_get_base returned null\n");
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
        return false;
    }

    n_slots_   = n_slots;
    slot_size_ = slot_size;
    tick_      = 0;
    used_.assign(n_slots, false);
    last_used_.assign(n_slots, 0);

    LLAMA_LOG_INFO("wp::PoolAllocator: allocated %d slots x %zu B (%.1f MiB)\n",
                   n_slots, slot_size, total / 1048576.0);
    return true;
}

int PoolAllocator::alloc_slot() {
    if (n_slots_ == 0 || base_ == nullptr) {
        return -1;
    }
    // First pass: any free slot.
    for (int i = 0; i < n_slots_; ++i) {
        if (!used_[i]) {
            used_[i]      = true;
            last_used_[i] = ++tick_;
            return i;
        }
    }
    // All in use: evict LRU.
    int      lru     = 0;
    uint64_t lru_t   = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < n_slots_; ++i) {
        if (last_used_[i] < lru_t) {
            lru_t = last_used_[i];
            lru   = i;
        }
    }
    if (on_evict_) {
        on_evict_(lru);
    }
    last_used_[lru] = ++tick_;
    // used_ stays true: slot transitions directly from old owner to new one.
    return lru;
}

void PoolAllocator::mark_used(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) return;
    last_used_[slot_idx] = ++tick_;
}

void PoolAllocator::release_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) return;
    used_[slot_idx] = false;
}

void * PoolAllocator::slot_ptr(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_ || base_ == nullptr) return nullptr;
    return (uint8_t *) base_ + (size_t) slot_idx * slot_size_;
}

int PoolAllocator::lru_slot() const {
    if (n_slots_ == 0) return -1;
    int      lru   = 0;
    uint64_t lru_t = last_used_[0];
    for (int i = 1; i < n_slots_; ++i) {
        if (last_used_[i] < lru_t) {
            lru_t = last_used_[i];
            lru   = i;
        }
    }
    return lru;
}

}  // namespace wp
