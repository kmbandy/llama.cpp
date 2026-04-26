#include "llama-weight-pager.h"
#include "ggml.h"
#include "llama-impl.h"

#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif
#include <algorithm>
#include <limits>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

// ============================================================================
// Weight Pager Eval Callback (Phase 3 + Phase 4)
// ============================================================================

/// Track the last processed page index for prefetch submission across callbacks
static int s_last_processed_page = -1;

/// Eval callback for weight pager. Called before each graph node is executed.
/// Phase 3: Ensures weight tensors are paged in to VRAM before use.
/// Phase 4: Completes in-flight prefetches and submits prefetch for next layer.
bool weight_pager_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true; // yes, we want the callback
    }

    auto * pager = (llama_weight_pager *) user_data;
    if (!pager) {
        return true;
    }

    // Collect page indices for all weight tensors in this tensor's sources
    std::vector<int> page_indices;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!t->src[i]) {
            break;
        }

        struct ggml_tensor * src = t->src[i];
        if (!src) {
            continue;
        }

        const char * tensor_name = ggml_get_name(src);
        int page_idx = pager->find_page(tensor_name);
        if (page_idx < 0) {
            continue; // not a weight tensor tracked by pager
        }

        page_indices.push_back(page_idx);
    }

    if (page_indices.empty()) {
        return true;
    }

    // Phase 4: Complete any in-flight prefetches for the pages we need
#ifdef LLAMA_HAVE_IO_URING
    if (pager->async_prefetch && pager->io_reader != nullptr) {
        for (int page_idx : page_indices) {
            pager->complete_prefetch(page_idx);
        }
    }
#endif

    // Phase 3: Ensure each tensor is in VRAM and redirect data pointers
    for (int page_idx : page_indices) {
        llama_weight_page & page = pager->pages[page_idx];
        void * vram = pager->ensure(page.tensor_name);
        if (vram) {
            // Redirect tensor data pointer to current VRAM slot
            // Find the source tensor and update its data pointer
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (!t->src[i]) {
                    break;
                }
                struct ggml_tensor * src = t->src[i];
                if (src && strcmp(ggml_get_name(src), page.tensor_name.c_str()) == 0) {
                    src->data = vram;
                }
            }
        }
    }

    // Phase 4: Submit prefetch for the next page (pipeline NVMe reads with GPU compute)
#ifdef LLAMA_HAVE_IO_URING
    if (pager->async_prefetch && pager->io_reader != nullptr) {
        // Find the highest page index we just processed
        int max_page = -1;
        for (int page_idx : page_indices) {
            if (page_idx > max_page) {
                max_page = page_idx;
            }
        }

        // Submit prefetch for the next page if it exists
        if (max_page >= 0 && max_page + 1 < (int)pager->pages.size()) {
            pager->submit_prefetch(max_page + 1);
        }
    }
#endif

    return true;
}

// ============================================================================
// llama_vram_pool implementation
// ============================================================================

int llama_vram_pool::alloc_slot() {
    // First, try to find a free slot
    for (int i = 0; i < n_slots; i++) {
        if (!used[i]) {
            used[i] = true;
            return i;
        }
    }

    // All slots are in use - find and evict the LRU slot
    int lru_idx = 0;
    uint64_t min_tick = lru_tick[0];

    for (int i = 1; i < n_slots; i++) {
        if (lru_tick[i] < min_tick) {
            min_tick = lru_tick[i];
            lru_idx = i;
        }
    }

    // Evict the LRU slot — caller is responsible for updating page state
    // (slot remains marked used; caller overwrites it)
    used[lru_idx] = true;
    return lru_idx;
}

void llama_vram_pool::free_slot(int idx) {
    if (idx < 0 || idx >= n_slots) {
        return;  // invalid index
    }
    used[idx] = false;
}

// ============================================================================
// llama_weight_pager implementation
// ============================================================================

void llama_weight_pager::init_pool(size_t slot_size, int n_slots) {
    if (n_slots <= 0) {
        return;  // invalid configuration
    }

    pool.slot_size = slot_size;
    pool.n_slots = n_slots;
    pool.used.resize(n_slots, false);
    pool.lru_tick.resize(n_slots, 0);

#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
    hipError_t err = hipMalloc(&pool.base, (size_t)n_slots * slot_size);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("llama_weight_pager: hipMalloc failed (%s), weight paging disabled\n", hipGetErrorString(err));
        pool.base = nullptr;
    }
#else
    pool.base = nullptr;
#endif
}

void* llama_weight_pager::ensure(const std::string & name) {
    int page_idx = find_page(name);
    if (page_idx < 0) {
        // Tensor not tracked by pager
        return nullptr;
    }

    llama_weight_page & page = pages[page_idx];

    // If already in VRAM, update tick and return pointer
    if (page.slot_idx >= 0 && page.vram_ptr != nullptr) {
        // Update LRU tick for this page
        page.last_used = ++tick;
        // Also update the slot's lru_tick
        if (page.slot_idx < pool.n_slots) {
            pool.lru_tick[page.slot_idx] = tick;
        }
        return page.vram_ptr;
    }

    // Not in VRAM - need to page in
    // Phase 1 stub: page_in() returns nullptr, so we return nullptr
    // In Phase 2, this will actually load from NVMe
    page_in(page);

    if (page.vram_ptr != nullptr) {
        page.last_used = ++tick;
        if (page.slot_idx >= 0 && page.slot_idx < pool.n_slots) {
            pool.lru_tick[page.slot_idx] = tick;
        }
    }

    return page.vram_ptr;
}

void llama_weight_pager::evict_lru() {
    // Find the LRU page that is currently loaded
    int lru_page_idx = -1;
    uint64_t min_tick = std::numeric_limits<uint64_t>::max();

    for (size_t i = 0; i < pages.size(); i++) {
        const llama_weight_page & page = pages[i];
        if (page.slot_idx >= 0 && page.last_used < min_tick) {
            min_tick = page.last_used;
            lru_page_idx = (int)i;
        }
    }

    if (lru_page_idx < 0) {
        return;
    }

    llama_weight_page & page = pages[lru_page_idx];
    pool.free_slot(page.slot_idx);
    page.slot_idx = -1;
    page.vram_ptr = nullptr;
}

// When pool.alloc_slot() evicts a slot, we must also clear the page that owned it.
// Called internally after alloc_slot() returns an in-use slot index.
static void pager_invalidate_slot(llama_weight_pager * pager, int slot_idx) {
    for (auto & page : pager->pages) {
        if (page.slot_idx == slot_idx) {
            page.slot_idx = -1;
            page.vram_ptr = nullptr;
            return;
        }
    }
}

int llama_weight_pager::find_page(const std::string & name) const {
    auto it = name_to_page.find(name);
    if (it != name_to_page.end()) {
        return it->second;
    }
    return -1;
}

void llama_weight_pager::page_in(llama_weight_page & page) {
    // Phase 2: actual page-in from NVMe to VRAM
    // Check if we have a valid pool and file descriptor
    if (!pool.is_valid() || fd < 0) {
        return;  // cannot page in without pool or fd
    }

    // Allocate a slot (may evict LRU if full — invalidate the displaced page)
    int slot = pool.alloc_slot();
    pager_invalidate_slot(this, slot);
    void * dst = pool.slot_ptr(slot);

    // Read from file directly into VRAM via pread
    // For SAM/ReBAR, this goes NVMe → BAR1 → VRAM directly
    ssize_t n = pread(fd, dst, page.size, page.file_offset);
    if (n != (ssize_t)page.size) {
        LLAMA_LOG_WARN("page_in: failed to read tensor (read %zd/%zu bytes)\n", n, page.size);
        // On failure, free the slot and return
        pool.free_slot(slot);
        return;
    }

    // Update page state
    page.slot_idx = slot;
    page.vram_ptr = (uint8_t*)dst;
    pool.used[slot] = true;
    pool.lru_tick[slot] = ++tick;
}

int llama_weight_pager::get_lru_slot() const {
    if (pool.n_slots <= 0) {
        return -1;
    }

    int lru_idx = 0;
    uint64_t min_tick = pool.lru_tick[0];

    for (int i = 1; i < pool.n_slots; i++) {
        if (pool.lru_tick[i] < min_tick) {
            min_tick = pool.lru_tick[i];
            lru_idx = i;
        }
    }

    return lru_idx;
}

// ============================================================================
// Async Prefetch via io_uring (Phase 4)
// ============================================================================

#ifdef LLAMA_HAVE_IO_URING

#include "llama-io-uring.h"

bool llama_weight_pager::init_io_uring(int queue_depth) {
    if (fd < 0) {
        return false;
    }
    io_reader = new llama_io_uring(fd, queue_depth);
    if (!io_reader->valid()) {
        delete io_reader;
        io_reader = nullptr;
        return false;
    }
    return true;
}

void llama_weight_pager::submit_prefetch(int page_idx) {
    if (!io_reader || !async_prefetch) {
        return;
    }

    if (page_idx < 0 || page_idx >= (int)pages.size()) {
        return;
    }

    auto & page = pages[page_idx];
    if (page.slot_idx >= 0) {
        return; // already in VRAM
    }

    // Check if already in-flight
    if (in_flight.count(page_idx)) {
        return; // already being prefetched
    }

    // Allocate a slot (may evict LRU if needed — invalidate the displaced page)
    int slot = pool.alloc_slot();
    pager_invalidate_slot(this, slot);
    void * dst = pool.slot_ptr(slot);

    // Submit async read via io_uring
    // Use page_idx as user_data to identify completion
    io_reader->submit_read(0, dst, page.size, page.file_offset, (uint64_t)page_idx);

    in_flight[page_idx] = { page_idx, slot, dst };
}

bool llama_weight_pager::complete_prefetch(int page_idx) {
    if (page_idx < 0 || page_idx >= (int)pages.size()) {
        return false;
    }

    auto & page = pages[page_idx];

    // If already loaded (via ensure or previous prefetch), return true
    if (page.slot_idx >= 0 && page.vram_ptr != nullptr) {
        return true;
    }

    // Check if this page is in-flight
    auto it = in_flight.find(page_idx);
    if (it == in_flight.end()) {
        return false; // not being prefetched
    }

    // Wait for completion and process all completions
    bool loaded = false;
    while (!in_flight.empty()) {
        uint64_t user_data = io_reader->wait_one(nullptr);
        if (user_data == UINT64_MAX) {
            break; // error or timeout
        }

        int completed_idx = (int)user_data;
        if (completed_idx < 0 || completed_idx >= (int)pages.size()) {
            continue; // invalid page index
        }

        auto & completed_page = pages[completed_idx];
        auto fit = in_flight.find(completed_idx);
        if (fit == in_flight.end()) {
            continue; // already processed
        }

        auto & req = fit->second;

        // Update page state with the loaded data
        completed_page.slot_idx = req.slot_idx;
        completed_page.vram_ptr = (uint8_t*)req.dst;
        pool.used[req.slot_idx] = true;
        pool.lru_tick[req.slot_idx] = ++tick;

        in_flight.erase(fit);
        if (completed_idx == page_idx) {
            loaded = true;
        }
    }

    return loaded;
}

void llama_weight_pager::drain_prefetches() {
    while (!in_flight.empty()) {
        io_reader->wait_one(nullptr);
    }
}

#endif // LLAMA_HAVE_IO_URING
