#include "wp-pool.h"

#include "ggml-backend.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace wp {

// ---------------------------------------------------------------------------
// MAD-234 — UMA / APU safety helpers
// ---------------------------------------------------------------------------

bool is_uma_archname(const char * arch_name) {
    if (arch_name == nullptr) return false;
    // Known UMA / APU prefixes. Match by prefix because HIP appends
    // sub-version suffixes (gfx1151:sramecc+:xnack-, gfx1103:..., etc.).
    static const char * const kUmaPrefixes[] = {
        "gfx115",   // Strix Halo / Strix Point family (gfx1150, gfx1151, ...)
        "gfx1103",  // Phoenix / Hawk Point
        "gfx1102",  // Phoenix variant
        "gfx1100",  // Some integrated RDNA3 SKUs report this — conservative include
        "gfx90c",   // Renoir / Cezanne (Vega-based APU)
        "gfx940",   // MI300A (server APU; UMA in unified-mem mode)
    };
    for (const char * pfx : kUmaPrefixes) {
        const size_t plen = std::strlen(pfx);
        if (std::strncmp(arch_name, pfx, plen) == 0) {
            return true;
        }
    }
    return false;
}

bool is_uma_device(int device_idx) {
#if defined(GGML_USE_HIP)
    if (device_idx < 0) return false;

    // Primary signal: hipDeviceGetAttribute(hipDeviceAttributeIntegrated).
    // Returns 1 for integrated (UMA) devices, 0 for discrete. This is the
    // canonical HIP API and should be honoured first.
    int integrated = 0;
    hipError_t err = hipDeviceGetAttribute(&integrated,
                                           hipDeviceAttributeIntegrated,
                                           device_idx);
    if (err == hipSuccess && integrated != 0) {
        return true;
    }

    // Fallback: parse hipDeviceProp_t::gcnArchName for known UMA prefixes.
    // Older HIP runtimes can report Integrated=0 on real APUs; the gfx-string
    // is authoritative. Costs one hipGetDeviceProperties call (cheap).
    hipDeviceProp_t prop{};
    err = hipGetDeviceProperties(&prop, device_idx);
    if (err != hipSuccess) {
        return false;
    }
    return is_uma_archname(prop.gcnArchName);
#else
    (void) device_idx;
    return false;
#endif
}

size_t read_mem_available_bytes() {
#if defined(__linux__)
    // Parse /proc/meminfo for the "MemAvailable:" line. Format is fixed
    // per Documentation/filesystems/proc.rst — key, value, unit (always "kB").
    std::FILE * f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char line[256];
    size_t out_bytes = 0;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            unsigned long long kb = 0;
            // Format: "MemAvailable:    <kb> kB"
            if (std::sscanf(line + 13, "%llu", &kb) == 1) {
                out_bytes = (size_t) kb * 1024ULL;
            }
            break;
        }
    }
    std::fclose(f);
    return out_bytes;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// PoolAllocator
// ---------------------------------------------------------------------------


PoolAllocator::~PoolAllocator() {
    if (buf_ != nullptr) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    base_ = nullptr;
}

bool PoolAllocator::init(ggml_backend_buffer_type_t buft,
                         int                        n_slots,
                         size_t                     slot_size,
                         int                        device_idx) {
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

    // MAD-234: UMA / APU pre-flight safety check.
    //
    // On integrated GPUs the buffer-type allocator pulls from system RAM via
    // an IOMMU-mapped region. ggml_backend_buft_alloc_buffer (and the
    // underlying hipMalloc / hipExtMallocWithFlags) silently succeeds even
    // when there isn't actually that much RAM available — the pages are
    // swap-eligible. The moment we touch each page during a fault, the OS
    // starts swapping out everything else; the box becomes unresponsive
    // with multi-second per-step stalls. Refuse here with an actionable
    // error instead of letting the allocator take the system down.
    //
    // Discrete GPUs (device_idx < 0 sentinel OR is_uma_device returns
    // false) skip this check entirely — there the kernel has its own
    // VRAM budget and an oversized request just fails at hipMalloc, which
    // we report cleanly below.
    if (device_idx >= 0 && is_uma_device(device_idx)) {
        const size_t avail   = read_mem_available_bytes();
        const size_t safety  = (size_t) 2 * 1024 * 1024 * 1024;  // 2 GiB margin
        LLAMA_LOG_WARN(
            "wp::PoolAllocator: UMA / APU device detected (device_idx=%d) — "
            "VRAM and host RAM share physical memory; this pool's %.2f GiB "
            "(%d slots x %zu B) reduces system RAM headroom 1:1.\n",
            device_idx, (double) total / (1024.0 * 1024.0 * 1024.0),
            n_slots, slot_size);
        if (avail > 0 && total + safety > avail) {
            LLAMA_LOG_ERROR(
                "wp::PoolAllocator::init: refusing pool allocation on UMA device. "
                "Requested %.2f GiB + %.2f GiB safety margin exceeds MemAvailable "
                "(%.2f GiB). An IOMMU-mapped allocation this size would silently "
                "succeed then swap out the system on first fault. Reduce n_slots "
                "(currently %d, slot_size=%zu B) so total <= %.2f GiB, or run on a "
                "discrete GPU.\n",
                (double) total  / (1024.0 * 1024.0 * 1024.0),
                (double) safety / (1024.0 * 1024.0 * 1024.0),
                (double) avail  / (1024.0 * 1024.0 * 1024.0),
                n_slots, slot_size,
                (avail > safety ? (double) (avail - safety) / (1024.0 * 1024.0 * 1024.0) : 0.0));
            return false;
        }
        if (avail == 0) {
            // MemAvailable read failed (non-Linux or /proc not mounted).
            // Log the warning but proceed — operator is on their own.
            LLAMA_LOG_WARN("wp::PoolAllocator: could not read MemAvailable; "
                           "UMA safety clamp skipped. Proceed at your own risk.\n");
        }
    }

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
    pin_count_.assign(n_slots, 0);              // MAD-231: refcount starts at 0
    lru_walk_pinned_skips_ = 0;                 // MAD-231: telemetry reset
    hit_count_.assign(n_slots, 0);              // MAD-237: popularity counter
    lru_walk_hot_skips_      = 0;
    n_evictions_since_decay_ = 0;
    n_decays_                = 0;

    LLAMA_LOG_INFO("wp::PoolAllocator: allocated %d slots x %zu B (%.1f MiB)\n",
                   n_slots, slot_size, total / 1048576.0);
    return true;
}

int PoolAllocator::alloc_slot() {
    if (n_slots_ == 0 || base_ == nullptr) {
        return -1;
    }
    // First pass: any free AND unpinned slot. Pinned-free is a transient
    // state (release_slot before unpin_slot) but skip them anyway — using
    // a pinned slot would mean the caller doesn't know it's still in use.
    for (int i = 0; i < n_slots_; ++i) {
        if (!used_[i] && pin_count_[i] == 0) {
            used_[i]      = true;
            last_used_[i] = ++tick_;
            return i;
        }
    }
    // All used (or remaining free are pinned): evict LRU among UNPINNED.
    // Pinned slots are skipped in the LRU walk — they're referenced by an
    // in-flight op and evicting them would corrupt the read (MAD-231).
    //
    // MAD-237 two-pass eviction:
    //   Pass A (only if hot_hit_threshold_ > 0): LRU among
    //          (unpinned AND hit_count <= hot_hit_threshold_). Protects
    //          the popular set from being evicted under bursty pressure.
    //   Pass B (fallback): LRU among unpinned regardless of hit_count.
    //          Always evictable so the pool can make forward progress.
    int      lru   = -1;
    uint64_t lru_t = std::numeric_limits<uint64_t>::max();
    int      n_pinned_skipped_this_walk = 0;
    int      n_hot_skipped_this_walk    = 0;
    const bool hot_enabled = (hot_hit_threshold_ > 0);

    // Pass A: cold + unpinned only (if hot protection enabled).
    if (hot_enabled) {
        for (int i = 0; i < n_slots_; ++i) {
            if (pin_count_[i] > 0) {
                ++n_pinned_skipped_this_walk;
                continue;
            }
            if (hit_count_[i] > hot_hit_threshold_) {
                ++n_hot_skipped_this_walk;
                continue;
            }
            if (last_used_[i] < lru_t) {
                lru_t = last_used_[i];
                lru   = i;
            }
        }
    }

    // Pass B: ignore hot status; pick LRU among unpinned. Runs unconditionally
    // when hot protection is disabled, OR when Pass A found nothing (all
    // unpinned slots are hot — extreme pressure where every slot is loved).
    if (lru < 0) {
        // Reset walk-local pinned counter so we don't double-count if
        // Pass A also did a walk. Hot skips DO accumulate across passes
        // — they represent slots we'd have liked to skip but couldn't.
        n_pinned_skipped_this_walk = 0;
        for (int i = 0; i < n_slots_; ++i) {
            if (pin_count_[i] > 0) {
                ++n_pinned_skipped_this_walk;
                continue;
            }
            if (last_used_[i] < lru_t) {
                lru_t = last_used_[i];
                lru   = i;
            }
        }
    }

    lru_walk_pinned_skips_ += (uint64_t) n_pinned_skipped_this_walk;
    lru_walk_hot_skips_    += (uint64_t) n_hot_skipped_this_walk;

    if (lru < 0) {
        // Every slot is pinned. Transient by design (an op will complete
        // and unpin); caller should treat as backpressure and retry.
        LLAMA_LOG_WARN("wp::PoolAllocator::alloc_slot: every slot is pinned "
                       "(%d/%d). Pool is too small for the working set, or an "
                       "unpin is missing. Caller should fall back to sync ensure.\n",
                       n_slots_, n_slots_);
        return -1;
    }
    if (on_evict_) {
        on_evict_(lru);
    }
    last_used_[lru] = ++tick_;
    // MAD-237: the evicted slot starts fresh — new owner has no hit history.
    hit_count_[lru] = 0;
    // used_ stays true: slot transitions directly from old owner to new one.

    // MAD-237 periodic decay: halve every hit count every kDecayEvery
    // evictions. Prevents long-running counters from monopolizing the
    // "hot" set after a workload shift. Cheap — one pass over n_slots_
    // every 1024 evictions = O(n/1024) amortized per eviction.
    ++n_evictions_since_decay_;
    if (n_evictions_since_decay_ >= kDecayEvery) {
        for (int i = 0; i < n_slots_; ++i) {
            hit_count_[i] >>= 1;
        }
        n_evictions_since_decay_ = 0;
        ++n_decays_;
    }
    return lru;
}

void PoolAllocator::mark_used(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) return;
    last_used_[slot_idx] = ++tick_;
    // MAD-237: bump popularity. Saturates rather than wraps — once a slot
    // is "very hot" we don't need more precision.
    if (hit_count_[slot_idx] < std::numeric_limits<uint32_t>::max()) {
        ++hit_count_[slot_idx];
    }
}

uint32_t PoolAllocator::hit_count(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_) return 0;
    return hit_count_[slot_idx];
}

void PoolAllocator::pin_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) {
        LLAMA_LOG_WARN("wp::PoolAllocator::pin_slot: out-of-range slot=%d (n=%d)\n",
                       slot_idx, n_slots_);
        return;
    }
    if (pin_count_[slot_idx] == std::numeric_limits<uint16_t>::max()) {
        LLAMA_LOG_WARN("wp::PoolAllocator::pin_slot: refcount overflow on slot=%d "
                       "(already at max %u). Likely a missing unpin somewhere.\n",
                       slot_idx, (unsigned) std::numeric_limits<uint16_t>::max());
        return;
    }
    ++pin_count_[slot_idx];
}

void PoolAllocator::unpin_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) {
        LLAMA_LOG_WARN("wp::PoolAllocator::unpin_slot: out-of-range slot=%d (n=%d)\n",
                       slot_idx, n_slots_);
        return;
    }
    if (pin_count_[slot_idx] == 0) {
        LLAMA_LOG_WARN("wp::PoolAllocator::unpin_slot: refcount underflow on slot=%d "
                       "(unpin called more times than pin). Bug in caller.\n", slot_idx);
        return;
    }
    --pin_count_[slot_idx];
}

bool PoolAllocator::is_pinned(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_) return false;
    return pin_count_[slot_idx] > 0;
}

int PoolAllocator::pin_count(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_) return 0;
    return (int) pin_count_[slot_idx];
}

int PoolAllocator::n_pinned() const {
    int n = 0;
    for (int i = 0; i < n_slots_; ++i) {
        if (pin_count_[i] > 0) ++n;
    }
    return n;
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
