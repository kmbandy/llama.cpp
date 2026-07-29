#include "wp-pool.h"

#include "ggml-backend.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <cmath>      // std::llround
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>   // std::gcd
#include <string>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace wp {

namespace {

bool env_flag_is_one(const char * var) {
    const char * v = std::getenv(var);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

size_t align_up(size_t n, size_t align) {
    if (align <= 1) return n;
    const size_t rem = n % align;
    if (rem == 0) return n;
    if (n > std::numeric_limits<size_t>::max() - (align - rem)) {
        return 0;
    }
    return n + (align - rem);
}

}  // anonymous namespace

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
    arena_size_ = 0;
}

bool PoolAllocator::init(ggml_backend_buffer_type_t buft,
                         int                        n_slots,
                         size_t                     slot_size,
                         int                        device_idx,
                         size_t                     extra_alignment,
                         const std::map<size_t, int> * page_size_hist,
                         const std::map<size_t, std::map<int, int>> * page_size_layer_counts) {
    if (buf_ != nullptr) {
        LLAMA_LOG_WARN("wp::PoolAllocator: init called twice — ignoring second call\n");
        return false;
    }
    if (buft == nullptr || n_slots <= 0 || slot_size == 0) {
        LLAMA_LOG_WARN("wp::PoolAllocator::init: invalid args (buft=%p, n_slots=%d, slot_size=%zu)\n",
                       (void *) buft, n_slots, slot_size);
        return false;
    }

    // Effective slot alignment, and therefore the slot STRIDE. This must be
    // settled before `total`: slot_ptr() is base + idx*slot_size_, so it is the
    // stride that decides whether a slot offset is legal, not a recorded
    // alignment value. lcm() so the result satisfies the buffer type and the
    // caller's extra constraint at once. See the note on init() in wp-pool.h.
    const size_t buft_align_ = std::max<size_t>(1, ggml_backend_buft_get_alignment(buft));
    const size_t extra_      = std::max<size_t>(1, extra_alignment);
    const size_t align_eff   = buft_align_ / std::gcd(buft_align_, extra_) * extra_;

    const size_t slot_size_eff = (slot_size + align_eff - 1) / align_eff * align_eff;
    if (slot_size_eff != slot_size) {
        LLAMA_LOG_INFO("wp::PoolAllocator: slot stride padded %zu -> %zu B for alignment %zu "
                       "(+%.2f%%)\n",
                       slot_size, slot_size_eff, align_eff,
                       100.0 * (double) (slot_size_eff - slot_size) / (double) slot_size);
    }

    const size_t total = (size_t) n_slots * slot_size_eff;
    const bool use_size_classes = env_flag_is_one("WP_SIZE_CLASS_SLOTS");

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

    n_slots_          = use_size_classes ? 0 : n_slots;
    slot_size_        = slot_size_eff;
    arena_size_       = total;
    slot_alignment_   = align_eff;
    size_class_slots_ = use_size_classes;
    precarved_        = false;
    high_water_       = 0;
    tick_             = 0;
    used_.assign((size_t) n_slots_, false);
    last_used_.assign((size_t) n_slots_, 0);
    pin_count_.assign((size_t) n_slots_, 0);              // MAD-231: refcount starts at 0
    slot_offset_.clear();
    slot_bytes_.clear();
    slot_class_.clear();
    free_by_class_.clear();
    if (size_class_slots_) {
        used_.reserve((size_t) n_slots);
        last_used_.reserve((size_t) n_slots);
        pin_count_.reserve((size_t) n_slots);
        hit_count_.reserve((size_t) n_slots);
        slot_offset_.reserve((size_t) n_slots);
        slot_bytes_.reserve((size_t) n_slots);
        slot_class_.reserve((size_t) n_slots);
    } else {
        slot_offset_.assign((size_t) n_slots, 0);
        slot_bytes_.assign((size_t) n_slots, slot_size);
        slot_class_.assign((size_t) n_slots, slot_size);
    }
    lru_walk_pinned_skips_ = 0;                 // MAD-231: telemetry reset
    hit_count_.assign((size_t) n_slots_, 0);              // MAD-237: popularity counter
    speculative_.assign((size_t) n_slots_, 0);            // cross-layer prefetch tier
    lru_walk_hot_skips_      = 0;
    n_evictions_since_decay_ = 0;
    n_decays_                = 0;

    // MAD-420 — pre-carve the arena per class from the catalog histogram.
    // Only when size classes are enabled AND the caller supplied a
    // histogram. On success this sets n_slots_, populates the slot vectors
    // and free_by_class_, and marks precarved_=true so alloc_slot_size_class_
    // never carves. On failure (empty histogram) we keep the legacy on-
    // demand carve path (n_slots_ stays 0, high_water_ grows on alloc).
    if (size_class_slots_ && page_size_hist != nullptr) {
        if (!carve_size_classes_(*page_size_hist, page_size_layer_counts)) {
            LLAMA_LOG_WARN("wp::PoolAllocator::init: size-class pre-carve skipped "
                           "(empty histogram); falling back to on-demand carve\n");
        }
    }

    if (size_class_slots_ && precarved_) {
        LLAMA_LOG_INFO("wp::PoolAllocator: allocated size-class arena budget %zu B (%.1f MiB), max_page_size=%zu, alignment=%zu, pre-carved %d slots in %zu class(es)\n",
                       total, total / 1048576.0, slot_size, slot_alignment_,
                       n_slots_, slot_class_table_.size());
    } else if (size_class_slots_) {
        LLAMA_LOG_INFO("wp::PoolAllocator: allocated size-class arena budget %zu B (%.1f MiB), max_page_size=%zu, alignment=%zu (on-demand carve)\n",
                       total, total / 1048576.0, slot_size, slot_alignment_);
    } else {
        LLAMA_LOG_INFO("wp::PoolAllocator: allocated %d slots x %zu B (%.1f MiB)\n",
                       n_slots, slot_size, total / 1048576.0);
    }
    return true;
}

int PoolAllocator::alloc_slot(size_t requested_size) {
    if (size_class_slots_) {
        return alloc_slot_size_class_(requested_size == 0 ? slot_size_ : requested_size);
    }
    return alloc_slot_fixed_();
}

int PoolAllocator::n_free_unpinned() const {
    if (n_slots_ == 0 || base_ == nullptr) {
        return 0;
    }
    int n = 0;
    if (size_class_slots_) {
        for (const auto & kv : free_by_class_) {
            for (int s : kv.second) {
                if (s >= 0 && s < n_slots_ && pin_count_[s] == 0) {
                    ++n;
                }
            }
        }
        return n;
    }
    for (int i = 0; i < n_slots_; ++i) {
        if (!used_[i] && pin_count_[i] == 0) {
            ++n;
        }
    }
    return n;
}

int PoolAllocator::alloc_slot_no_evict(size_t requested_size) {
    if (n_slots_ == 0 || base_ == nullptr) {
        return -1;
    }
    if (size_class_slots_) {
        const size_t need = requested_size == 0 ? slot_size_ : requested_size;
        // Any free class that fits; never steal used slots.
        for (auto & kv : free_by_class_) {
            if (kv.first < need) {
                continue;
            }
            while (!kv.second.empty()) {
                const int s = kv.second.back();
                kv.second.pop_back();
                if (s < 0 || s >= n_slots_ || used_[s] || pin_count_[s] > 0) {
                    continue;
                }
                used_[s]        = true;
                last_used_[s]   = ++tick_;
                speculative_[s] = 0;
                return s;
            }
        }
        return -1;
    }
    for (int i = 0; i < n_slots_; ++i) {
        if (!used_[i] && pin_count_[i] == 0) {
            used_[i]        = true;
            last_used_[i]   = ++tick_;
            speculative_[i] = 0;
            return i;
        }
    }
    return -1;
}

void PoolAllocator::set_speculative(int slot_idx, bool spec) {
    if (slot_idx < 0 || slot_idx >= n_slots_) return;
    speculative_[(size_t) slot_idx] = spec ? 1 : 0;
}

bool PoolAllocator::is_speculative(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_) return false;
    return speculative_[(size_t) slot_idx] != 0;
}

int PoolAllocator::n_speculative() const {
    int n = 0;
    for (int i = 0; i < n_slots_; ++i) if (speculative_[(size_t) i]) ++n;
    return n;
}

int PoolAllocator::alloc_slot_fixed_() {
    if (n_slots_ == 0 || base_ == nullptr) {
        return -1;
    }
    // First pass: any free AND unpinned slot. Pinned-free is a transient
    // state (release_slot before unpin_slot) but skip them anyway — using
    // a pinned slot would mean the caller doesn't know it's still in use.
    for (int i = 0; i < n_slots_; ++i) {
        if (!used_[i] && pin_count_[i] == 0) {
            used_[i]        = true;
            last_used_[i]   = ++tick_;
            speculative_[i] = 0;
            return i;
        }
    }
    // Pass 0: recycle the LRU *speculative* slot before evicting the working
    // set. Speculation must never evict a pinned/hot live page (the footgun).
    {
        int      spec_lru   = -1;
        uint64_t spec_lru_t = std::numeric_limits<uint64_t>::max();
        for (int i = 0; i < n_slots_; ++i) {
            if (pin_count_[i] > 0 || !speculative_[i]) continue;
            if (last_used_[i] < spec_lru_t) { spec_lru_t = last_used_[i]; spec_lru = i; }
        }
        if (spec_lru >= 0) {
            if (on_evict_) on_evict_(spec_lru);   // callback sees is_speculative()==true
            speculative_[spec_lru] = 0;
            last_used_[spec_lru]   = ++tick_;
            hit_count_[spec_lru]   = 0;
            decay_after_eviction_();
            return spec_lru;
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
    speculative_[lru] = 0;
    // used_ stays true: slot transitions directly from old owner to new one.

    decay_after_eviction_();
    return lru;
}

int PoolAllocator::alloc_slot_size_class_(size_t requested_size) {
    if (base_ == nullptr || requested_size == 0 || requested_size > slot_size_) {
        return -1;
    }

    const size_t requested_class = size_class_for_(requested_size);
    if (requested_class == 0 || requested_class > slot_size_) {
        return -1;
    }

    int slot = take_free_size_class_slot_(requested_class);
    if (slot >= 0) {
        used_[slot]        = true;
        last_used_[slot]   = ++tick_;
        hit_count_[slot]   = 0;
        speculative_[slot] = 0;
        return slot;
    }

    // MAD-420 — when the arena was pre-carved at init, NEVER carve on demand.
    // The class mix is fixed by the catalog histogram; carving here would
    // re-introduce the original bug (arena fills with small slots, large
    // page has no adequate class). The legacy on-demand path is retained
    // only for the non-precarved size-class mode (unit tests).
    if (!precarved_ && high_water_ <= arena_size_ && requested_class <= arena_size_ - high_water_) {
        slot = n_slots_++;
        used_.push_back(true);
        last_used_.push_back(++tick_);
        pin_count_.push_back(0);
        hit_count_.push_back(0);
        speculative_.push_back(0);
        slot_offset_.push_back(high_water_);
        slot_bytes_.push_back(requested_class);
        slot_class_.push_back(requested_class);
        high_water_ += requested_class;
        return slot;
    }

    // Pass 0: recycle the LRU *speculative* slot of an adequate class before
    // evicting the working set. Speculation never evicts a live page.
    {
        int      spec_lru   = -1;
        uint64_t spec_lru_t = std::numeric_limits<uint64_t>::max();
        for (int i = 0; i < n_slots_; ++i) {
            if (!used_[i] || pin_count_[i] > 0 || !speculative_[i]) continue;
            if (slot_class_[i] < requested_class) continue;
            if (last_used_[i] < spec_lru_t) { spec_lru_t = last_used_[i]; spec_lru = i; }
        }
        if (spec_lru >= 0) {
            if (on_evict_) on_evict_(spec_lru);
            speculative_[spec_lru] = 0;
            last_used_[spec_lru]   = ++tick_;
            hit_count_[spec_lru]   = 0;
            decay_after_eviction_();
            return spec_lru;
        }
    }

    int n_pinned_skipped_this_walk = 0;
    int n_hot_skipped_this_walk    = 0;
    const int lru = pick_size_class_victim_(requested_class,
                                            n_pinned_skipped_this_walk,
                                            n_hot_skipped_this_walk);
    lru_walk_pinned_skips_ += (uint64_t) n_pinned_skipped_this_walk;
    lru_walk_hot_skips_    += (uint64_t) n_hot_skipped_this_walk;

    if (lru < 0) {
        LLAMA_LOG_WARN("wp::PoolAllocator::alloc_slot: no unpinned size-class slot can fit %zu B "
                       "(class=%zu, allocated_slots=%d, budget=%zu B, high_water=%zu B, precarved=%d)\n",
                       requested_size, requested_class, n_slots_, arena_size_, high_water_,
                       precarved_ ? 1 : 0);
        return -1;
    }

    if (on_evict_) {
        on_evict_(lru);
    }
    last_used_[lru]   = ++tick_;
    hit_count_[lru]   = 0;
    speculative_[lru] = 0;
    decay_after_eviction_();
    return lru;
}

size_t PoolAllocator::size_class_for_(size_t requested_size) const {
    if (requested_size == 0) {
        return 0;
    }
    size_t out = align_up(requested_size, slot_alignment_);
    if (out == 0) {
        return 0;
    }
    if (out > slot_size_ && requested_size <= slot_size_) {
        out = slot_size_;
    }
    return out;
}

int PoolAllocator::take_free_size_class_slot_(size_t requested_class) {
    size_t best_class = 0;
    int    best_slot  = -1;

    for (const auto & kv : free_by_class_) {
        const size_t cls = kv.first;
        if (cls < requested_class) {
            continue;
        }
        if (best_class != 0 && cls >= best_class) {
            continue;
        }
        for (int slot : kv.second) {
            if (slot < 0 || slot >= n_slots_) {
                continue;
            }
            if (used_[slot] || pin_count_[slot] > 0 || slot_class_[slot] != cls) {
                continue;
            }
            best_class = cls;
            best_slot  = slot;
            break;
        }
    }

    if (best_slot < 0) {
        return -1;
    }

    auto & slots = free_by_class_[best_class];
    for (auto it = slots.begin(); it != slots.end(); ++it) {
        if (*it == best_slot) {
            slots.erase(it);
            break;
        }
    }
    return best_slot;
}

int PoolAllocator::pick_size_class_victim_(size_t requested_class,
                                           int & n_pinned_skipped,
                                           int & n_hot_skipped) const {
    n_pinned_skipped = 0;
    n_hot_skipped    = 0;

    std::vector<size_t> classes;
    classes.reserve(slot_class_.size());
    for (int i = 0; i < n_slots_; ++i) {
        if (!used_[i] || slot_class_[i] < requested_class) {
            continue;
        }
        if (std::find(classes.begin(), classes.end(), slot_class_[i]) == classes.end()) {
            classes.push_back(slot_class_[i]);
        }
    }
    std::sort(classes.begin(), classes.end());

    const bool hot_enabled = (hot_hit_threshold_ > 0);
    for (size_t cls : classes) {
        int      lru   = -1;
        uint64_t lru_t = std::numeric_limits<uint64_t>::max();
        int      class_pinned_skips = 0;
        int      class_hot_skips    = 0;

        if (hot_enabled) {
            for (int i = 0; i < n_slots_; ++i) {
                if (!used_[i] || slot_class_[i] != cls) {
                    continue;
                }
                if (pin_count_[i] > 0) {
                    ++class_pinned_skips;
                    continue;
                }
                if (hit_count_[i] > hot_hit_threshold_) {
                    ++class_hot_skips;
                    continue;
                }
                if (last_used_[i] < lru_t) {
                    lru_t = last_used_[i];
                    lru   = i;
                }
            }
            if (lru >= 0) {
                n_pinned_skipped = class_pinned_skips;
                n_hot_skipped    = class_hot_skips;
                return lru;
            }
        }

        class_pinned_skips = 0;
        lru_t = std::numeric_limits<uint64_t>::max();
        for (int i = 0; i < n_slots_; ++i) {
            if (!used_[i] || slot_class_[i] != cls) {
                continue;
            }
            if (pin_count_[i] > 0) {
                ++class_pinned_skips;
                continue;
            }
            if (last_used_[i] < lru_t) {
                lru_t = last_used_[i];
                lru   = i;
            }
        }
        if (lru >= 0) {
            n_pinned_skipped = class_pinned_skips;
            n_hot_skipped    = class_hot_skips;
            return lru;
        }
    }

    return -1;
}

void PoolAllocator::decay_after_eviction_() {
    ++n_evictions_since_decay_;
    if (n_evictions_since_decay_ >= kDecayEvery) {
        for (int i = 0; i < n_slots_; ++i) {
            hit_count_[i] >>= 1;
        }
        n_evictions_since_decay_ = 0;
        ++n_decays_;
    }
}

// MAD-420 — pre-carve the arena into per-class slots from the catalog
// histogram. See docs/superpowers/specs/2026-07-28-size-class-slots.md.
//
// Solver (spec section 2):
//   f_c      = n_c / total_pages           (class c demand fraction)
//   avg      = sum(f_c * s_c) = (sum n_c*s_c) / total_pages
//   K        = floor(A / avg)              (total slots that fit in arena A)
//   k_c      = max(1, round(f_c * K))      (each class gets >= 1 slot)
// then trim k_c down so sum(k_c * s_c) <= A (rounding can overshoot).
//
// Classes are aligned up to slot_alignment_ exactly the way
// size_class_for_ does at alloc time, and clamped to slot_size_ (the max
// page size), so a request for a given page size resolves to the same class
// we carved. A single-class histogram reproduces the uniform layout:
// avg == s == slot_size_, K = A/s = n_slots, one class with k = K slots at
// stride slot_size_ -> identical to the fixed-slot path.
bool PoolAllocator::carve_size_classes_(const std::map<size_t, int> & hist,
                                       const std::map<size_t, std::map<int, int>> * layer_counts) {
    if (hist.empty() || arena_size_ == 0 || slot_alignment_ == 0) {
        return false;
    }

    // 1. Build the class table: align each raw page size up to slot_alignment_,
    //    clamp to slot_size_, and coalesce any sizes that land on the same
    //    class (e.g. two raw sizes within one alignment granule). Sorted asc.
    struct ClassPlan {
        size_t class_size;   // stride in bytes (aligned, <= slot_size_)
        int    n_pages;      // demand count (sum of coalesced raw sizes)
    };
    std::vector<ClassPlan> plan;
    for (const auto & kv : hist) {
        const size_t raw  = kv.first;
        const int    cnt  = kv.second;
        size_t cls = align_up(raw, slot_alignment_);
        if (cls == 0)                    continue;   // overflow guard
        if (cls > slot_size_)            cls = slot_size_;  // clamp to max
        if (!plan.empty() && plan.back().class_size == cls) {
            plan.back().n_pages += cnt;
        } else {
            plan.push_back({cls, cnt});
        }
    }
    if (plan.empty()) {
        return false;
    }

    // 2. Demand fractions and average slot size.
    long long total_pages = 0;
    long long weighted    = 0;   // sum n_c * s_c, in bytes
    for (const auto & p : plan) {
        total_pages += p.n_pages;
        weighted    += (long long) p.n_pages * (long long) p.class_size;
    }
    if (total_pages <= 0 || weighted <= 0) {
        return false;
    }
    const double avg = (double) weighted / (double) total_pages;
    if (avg <= 0.0) {
        return false;
    }

    // 3. Per-class PIN FLOOR.
    //
    //    This is the correction to the first cut of this solver, which sized
    //    every class purely by demand share. Demand share is the wrong axis.
    //    A whole ensure_batch is pinned at once (wp-pager.cpp pins each page
    //    as it is allocated, before allocating the next), and alloc_slot
    //    refuses to evict a pinned slot -- so the binding constraint is the
    //    peak number of pages of a class pinned SIMULTANEOUSLY, not how often
    //    the class is requested.
    //
    //    That peak is bounded by one block's expert union: an ensure_batch
    //    covers one tensor of one block, and in the worst case (a wide prefill
    //    batch) the union is every expert in the block. So the floor for a
    //    class is the largest number of its pages that any single block owns.
    //
    //    Measured on GLM-5.2: the 6.375 MiB class is 1.75% of all pages
    //    (-> ~72 slots by demand share) but lives in exactly 4 blocks at 256
    //    pages each. Every run aborted the moment a wide batch reached block 8.
    //
    //    layer_counts == nullptr (unit tests, callers that only have a plain
    //    histogram) leaves every floor at 0, reproducing the previous purely
    //    proportional behaviour exactly.
    std::vector<int> floor_c(plan.size(), 0);
    if (layer_counts != nullptr) {
        // Merge per-block counts into ALIGNED classes first, then take the max
        // over blocks. Two raw sizes can coalesce into one class; taking the
        // max before merging would understate a block that owns both.
        std::map<int, std::map<size_t, int>> per_block;   // block -> class -> count
        for (const auto & kv : *layer_counts) {
            size_t cls = align_up(kv.first, slot_alignment_);
            if (cls == 0)         continue;
            if (cls > slot_size_) cls = slot_size_;
            for (const auto & bc : kv.second) {
                per_block[bc.first][cls] += bc.second;
            }
        }
        std::map<size_t, int> floor_by_class;
        for (const auto & b : per_block) {
            for (const auto & c : b.second) {
                int & f = floor_by_class[c.first];
                if (c.second > f) f = c.second;
            }
        }
        for (size_t i = 0; i < plan.size(); ++i) {
            auto it = floor_by_class.find(plan[i].class_size);
            if (it != floor_by_class.end()) {
                floor_c[i] = it->second;
                // A block cannot own more pages of a class than exist.
                if (floor_c[i] > plan[i].n_pages) floor_c[i] = plan[i].n_pages;
            }
        }
    }

    // 4. The floors must fit. If they do not, this arena cannot serve a wide
    //    batch under size classes at all -- carving anyway would just move the
    //    abort to run time, on the GPU, minutes in. Abandon the pre-carve and
    //    let the caller fall back to the uniform path, which is what works
    //    today (uniform slots are all max-size, so any page fits any slot).
    long long floor_bytes = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        floor_bytes += (long long) floor_c[i] * (long long) plan[i].class_size;
    }
    if (floor_bytes > (long long) arena_size_) {
        LLAMA_LOG_WARN("wp::PoolAllocator::carve_size_classes_: per-class pin floors need "
                       "%.1f MiB but the arena is %.1f MiB; abandoning pre-carve and falling "
                       "back to uniform slots. Raise the paging budget to use size classes.\n",
                       (double) floor_bytes / 1048576.0, (double) arena_size_ / 1048576.0);
        return false;
    }

    // 5. Distribute what is left of the arena by demand share, on top of the
    //    floors. Cap each class at its own page count -- more slots than pages
    //    of that class is pure waste (those bytes could serve another class).
    const double R     = (double) arena_size_ - (double) floor_bytes;
    const double K_rem = (R > 0.0) ? (R / avg) : 0.0;

    std::vector<int> kc(plan.size(), 0);
    long long total_k = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const double f = (double) plan[i].n_pages / (double) total_pages;
        long long k = (long long) floor_c[i] + (long long) llround(f * K_rem);
        if (k > plan[i].n_pages) k = plan[i].n_pages;   // no more slots than pages
        if (k < floor_c[i])      k = floor_c[i];        // the floor outranks the cap
        if (k < 1)               k = 1;                 // every class gets >= 1 slot
        kc[i] = (int) k;
        total_k += k;
    }

    // 6. Trim if rounding overshoots the arena. Pick the class whose removal
    //    of one slot reclaims the most bytes and still has a slot to spare
    //    above its floor (and above the >= 1 invariant). Repeat until it fits.
    //    Dropping a class below its floor would reintroduce the abort.
    auto total_bytes = [&]() -> long long {
        long long b = 0;
        for (size_t i = 0; i < plan.size(); ++i) b += (long long) kc[i] * (long long) plan[i].class_size;
        return b;
    };
    while (total_bytes() > (long long) arena_size_) {
        int    best_i  = -1;
        size_t best_sz = 0;
        for (size_t i = 0; i < plan.size(); ++i) {
            const int keep = floor_c[i] > 1 ? floor_c[i] : 1;
            if (kc[i] <= keep) continue;         // protect floor / >= 1 invariant
            if (plan[i].class_size > best_sz) {
                best_sz = plan[i].class_size;
                best_i  = (int) i;
            }
        }
        if (best_i < 0) break;                  // everything at its floor; can't trim
        --kc[best_i];
        --total_k;
    }
    if (total_k <= 0) {
        return false;
    }

    // 7. Carve contiguously. Slots are laid out class-by-class in ascending
    //    class order; within a class they are back-to-back at class_size
    //    stride. All start free and un-pinned, recorded in free_by_class_.
    const int n_carved = (int) total_k;
    used_.assign((size_t) n_carved, false);
    last_used_.assign((size_t) n_carved, 0);
    pin_count_.assign((size_t) n_carved, 0);
    hit_count_.assign((size_t) n_carved, 0);
    speculative_.assign((size_t) n_carved, 0);
    slot_offset_.resize((size_t) n_carved);
    slot_bytes_.resize((size_t) n_carved);
    slot_class_.resize((size_t) n_carved);
    free_by_class_.clear();
    slot_class_table_.clear();

    size_t off = 0;
    int    idx = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const size_t cls = plan[i].class_size;
        const int    n   = kc[i];
        slot_class_table_.emplace_back(cls, n);
        auto & free_list = free_by_class_[cls];
        free_list.reserve((size_t) n);
        for (int j = 0; j < n; ++j) {
            slot_offset_[(size_t) idx] = off;
            slot_bytes_[(size_t) idx]  = cls;
            slot_class_[(size_t) idx]  = cls;
            free_list.push_back(idx);
            off += cls;
            ++idx;
        }
    }
    // off is the carved bytes; it must not exceed the arena. Trimming above
    // guarantees this in normal cases; the remaining case is a pathological
    // arena too small for one slot per class (sum of per-class min sizes > A),
    // where trim bottoms out with every class at 1 and still overshoots.
    if (off > arena_size_) {
        LLAMA_LOG_WARN("wp::PoolAllocator::carve_size_classes_: carved %zu B exceeds arena %zu B "
                       "after trim (arena too small for one slot per class); "
                       "abandoning pre-carve, falling back to on-demand carve\n",
                       off, arena_size_);
        // Roll back so the caller can fall back to on-demand carve.
        n_slots_ = 0;
        slot_offset_.clear();
        slot_bytes_.clear();
        slot_class_.clear();
        free_by_class_.clear();
        slot_class_table_.clear();
        return false;
    }

    n_slots_    = n_carved;
    high_water_ = off;     // arena consumed by the carve; no further carving
    precarved_  = true;

    // 8. Human-readable per-class summary. A human sanity-checks the mix
    //    here: total slots, total bytes, each class stride / count / share.
    LLAMA_LOG_WARN("wp::PoolAllocator: pre-carved %d slots (%.1f MiB) from %zu class(es), "
                   "arena=%.1f MiB, avg_page=%.1f KiB, pin_floor=%.1f MiB:\n",
                   n_carved, (double) off / 1048576.0, plan.size(),
                   (double) arena_size_ / 1048576.0, avg / 1024.0,
                   (double) floor_bytes / 1048576.0);
    for (size_t i = 0; i < plan.size(); ++i) {
        const size_t cls = plan[i].class_size;
        const int    n   = kc[i];
        const double share = 100.0 * (double) n / (double) n_carved;
        LLAMA_LOG_WARN("  class %6.3f MiB (%zu B): %4d slots (%5.1f%% of pool), "
                       "pin_floor %d, demand %d pages (%.1f%% of %lld)\n",
                       (double) cls / 1048576.0, cls, n, share,
                       floor_c[i],
                       plan[i].n_pages,
                       100.0 * (double) plan[i].n_pages / (double) total_pages,
                       total_pages);
    }
    const long long tb = total_bytes();
    LLAMA_LOG_WARN("  total: %lld slots, %lld B (%.1f MiB) carved, %.1f MiB arena headroom\n",
                   total_k, tb, (double) tb / 1048576.0,
                   (double) ((long long) arena_size_ - tb) / 1048576.0);
    return true;
}

void PoolAllocator::mark_used(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) return;
    last_used_[slot_idx] = ++tick_;
    speculative_[slot_idx] = 0;   // demand hit promotes a speculative slot
    // MAD-237: bump popularity. Saturates rather than wraps — once a slot
    // is "very hot" we don't need more precision.
    if (hit_count_[slot_idx] < std::numeric_limits<uint32_t>::max()) {
        ++hit_count_[slot_idx];
    }
}

void PoolAllocator::touch_lru(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= n_slots_) return;
    last_used_[slot_idx] = ++tick_;
    // Deliberately does NOT clear speculative_ and does NOT bump hit_count_:
    // a landed prefetch is not evidence that the page is wanted.
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
    speculative_[slot_idx] = 0;
    if (size_class_slots_) {
        if (!used_[slot_idx]) return;
        used_[slot_idx] = false;
        free_by_class_[slot_class_[slot_idx]].push_back(slot_idx);
        return;
    }
    used_[slot_idx] = false;
}

void * PoolAllocator::slot_ptr(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_ || base_ == nullptr) return nullptr;
    if (size_class_slots_) {
        return (uint8_t *) base_ + slot_offset_[slot_idx];
    }
    return (uint8_t *) base_ + (size_t) slot_idx * slot_size_;
}

size_t PoolAllocator::slot_size(int slot_idx) const {
    if (slot_idx < 0 || slot_idx >= n_slots_) return 0;
    if (size_class_slots_) {
        return slot_bytes_[slot_idx];
    }
    return slot_size_;
}

int PoolAllocator::lru_slot() const {
    if (n_slots_ == 0) return -1;
    if (size_class_slots_) {
        int      lru   = -1;
        uint64_t lru_t = std::numeric_limits<uint64_t>::max();
        for (int i = 0; i < n_slots_; ++i) {
            if (!used_[i]) continue;
            if (last_used_[i] < lru_t) {
                lru_t = last_used_[i];
                lru   = i;
            }
        }
        return lru;
    }
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
