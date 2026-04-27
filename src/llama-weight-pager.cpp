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
#include <errno.h>
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

    // DIAGNOSTIC: Log callback invocation
    LLAMA_LOG_INFO("weight_pager_eval_cb: tensor=%s ask=%s\n",
                   ggml_get_name(t), ask ? "true" : "false");
    LLAMA_LOG_INFO("weight_pager_eval_cb: pager=%p pool.base=%p pool.n_slots=%d\n",
                   pager, pager->pool.base, pager->pool.n_slots);

    // Collect page indices for all weight tensors in this tensor's sources.
    // Also detect view tensors of tracked weights (ggml_gallocr_alloc_graph initialises
    // their data to (char*)1 + view_offs -- garbage -- so we must overwrite it
    // with the real paged-in VRAM address before every op that uses them).
    std::vector<int> page_indices;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * src = t->src[i];
        if (!src) {
            break;
        }

        // Direct weight tensor match
        int page_idx = pager->find_page(ggml_get_name(src));

        // View tensor whose source is a tracked weight
        if (page_idx < 0 && src->view_src != nullptr) {
            page_idx = pager->find_page(ggml_get_name(src->view_src));
        }

        if (page_idx < 0) {
            continue;
        }

        // Deduplicate (e.g. two views of the same weight in one op)
        bool already = false;
        for (int idx : page_indices) { if (idx == page_idx) { already = true; break; } }
        if (!already) {
            page_indices.push_back(page_idx);
        }
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

    // Phase 3: Ensure each tensor is in VRAM and redirect data pointers.
    // Handles both direct weight sources and view tensors of those weights.
    for (int page_idx : page_indices) {
        llama_weight_page & page = pager->pages[page_idx];
        void * vram = pager->ensure(page.tensor_name);
        if (vram) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * src = t->src[i];
                if (!src) {
                    break;
                }
                // Direct weight tensor
                if (strcmp(ggml_get_name(src), page.tensor_name.c_str()) == 0) {
                    src->data = vram;
                }
                // View of this weight: gallocr sets data=(char*)sentinel+view_offs;
                // overwrite with the real paged-in slot address + offset.
                else if (src->view_src != nullptr &&
                         strcmp(ggml_get_name(src->view_src), page.tensor_name.c_str()) == 0) {
                    src->data = (char *)vram + src->view_offs;
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

llama_weight_pager::~llama_weight_pager() {
    for (int fd : fds) {
        if (fd >= 0) {
            close(fd);
        }
    }
#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
    if (pool.pinned_staging != nullptr) {
        hipHostFree(pool.pinned_staging);
    }
    if (pool.base != nullptr) {
        hipFree(pool.base);
    }
    if (transfer_stream != nullptr) {
        hipStreamDestroy((hipStream_t)transfer_stream);
    }
#endif
}

bool llama_weight_pager::init_pool(size_t slot_size, int n_slots) {
    if (n_slots <= 0) {
        LLAMA_LOG_WARN("init_pool: invalid n_slots=%d\n", n_slots);
        return false;
    }

    pool.slot_size = slot_size;
    pool.n_slots = n_slots;
    pool.used.resize(n_slots, false);
    pool.lru_tick.resize(n_slots, 0);

    LLAMA_LOG_INFO("init_pool: allocating %d slots of %zu bytes each\n", n_slots, slot_size);

#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
    // Allocate the VRAM pool using hipMalloc
    hipError_t err = hipMalloc(&pool.base, (size_t)n_slots * slot_size);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("llama_weight_pager: hipMalloc failed: %s\n", hipGetErrorString(err));
        pool.base = nullptr;
        return false;
    }

    // Allocate the pinned host staging buffer
    err = hipHostMalloc(&pool.pinned_staging, slot_size, hipHostMallocDefault);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("llama_weight_pager: hipHostMalloc for pinned_staging failed: %s\n", hipGetErrorString(err));
        hipFree(pool.base);
        pool.base = nullptr;
        pool.pinned_staging = nullptr;
        return false;
    }

    // Create the async transfer stream
    err = hipStreamCreate((hipStream_t*)&transfer_stream);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("llama_weight_pager: hipStreamCreate failed: %s\n", hipGetErrorString(err));
        hipHostFree(pool.pinned_staging);
        hipFree(pool.base);
        pool.base = nullptr;
        pool.pinned_staging = nullptr;
        transfer_stream = nullptr;
        return false;
    }

    LLAMA_LOG_INFO("init_pool: allocated pool.base=%p pinned_staging=%p transfer_stream=%p (slot_size=%zu, n_slots=%d)\n",
                   pool.base, pool.pinned_staging, transfer_stream, slot_size, n_slots);
    return true;
#else
    LLAMA_LOG_WARN("init_pool: HIP not available, falling back\n");
    pool.base = nullptr;
    pool.pinned_staging = nullptr;
    transfer_stream = nullptr;
    return false;
#endif
}

void* llama_weight_pager::ensure(const std::string & name) {
    int page_idx = find_page(name);
    if (page_idx < 0) {
        // Tensor not tracked by pager
        return nullptr;
    }

    llama_weight_page & page = pages[page_idx];

    // DIAGNOSTIC: Log ensure call
    LLAMA_LOG_INFO("ensure: tensor=%s page_idx=%d slot_idx=%d vram_ptr=%p\n",
                   name.c_str(), page_idx, page.slot_idx, page.vram_ptr);

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
    LLAMA_LOG_INFO("ensure: calling page_in for tensor %s\n", name.c_str());
    page_in(page);
    LLAMA_LOG_INFO("ensure: page_in returned, page.slot_idx=%d page.vram_ptr=%p\n",
                   page.slot_idx, page.vram_ptr);

    if (page.vram_ptr != nullptr) {
        page.last_used = ++tick;
        if (page.slot_idx >= 0 && page.slot_idx < pool.n_slots) {
            pool.lru_tick[page.slot_idx] = tick;
        }
    } else {
        LLAMA_LOG_WARN("ensure: page_in returned nullptr for tensor %s\n", name.c_str());
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
            LLAMA_LOG_INFO("pager_invalidate_slot: invalidating slot=%d tensor=%s\n",
                           slot_idx, page.tensor_name.c_str());
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
    LLAMA_LOG_INFO("page_in: loading tensor %s (file_idx=%u, offset=%zu, size=%zu)\n",
                   page.tensor_name.c_str(), (unsigned)page.file_idx, page.file_offset, page.size);
    // Phase 3: two-step path with async stream - NVMe → pinned staging → VRAM
    // Check if we have a valid pool and file descriptor
    if (!pool.is_valid() || pool.pinned_staging == nullptr || fds.empty() || fds[0] < 0) {
        LLAMA_LOG_WARN("page_in: no valid pool or fd (pool.valid=%s pool.pinned_staging=%p fds.empty=%s fds[0]=%d)\n",
                       pool.is_valid() ? "true" : "false",
                       pool.pinned_staging,
                       fds.empty() ? "true" : "false",
                       fds.empty() ? -1 : fds[0]);
        return;  // cannot page in without pool, pinned_staging, or fd
    }

    // Step 1: Read from NVMe into pinned host staging buffer
    int read_fd = (page.file_idx < (uint16_t)fds.size()) ? fds[page.file_idx] : (fds.empty() ? -1 : fds[0]);
    LLAMA_LOG_INFO("page_in: pread fd=%d pinned_staging=%p size=%zu offset=%zu\n",
                   read_fd, pool.pinned_staging, page.size, page.file_offset);
    ssize_t n = pread(read_fd, pool.pinned_staging, page.size, page.file_offset);
    LLAMA_LOG_INFO("page_in: pread returned %zd (errno=%d)\n", n, errno);
    if (n != (ssize_t)page.size) {
        LLAMA_LOG_WARN("page_in: failed to read tensor (read %zd/%zu bytes)\n", n, page.size);
        return;
    }

    // Step 2: Allocate a slot (may evict LRU if full — invalidate the displaced page)
    int slot = pool.alloc_slot();
    pager_invalidate_slot(this, slot);
    void * dst = pool.slot_ptr(slot);

    // Step 3: Copy from pinned staging to VRAM slot using hipMemcpyAsync on transfer_stream
#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
    LLAMA_LOG_INFO("page_in: hipMemcpyAsync dst=%p pinned_staging=%p size=%zu stream=%p\n",
                   dst, pool.pinned_staging, page.size, transfer_stream);
    hipError_t err = hipMemcpyAsync(dst, pool.pinned_staging, page.size, hipMemcpyHostToDevice, (hipStream_t)transfer_stream);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("page_in: hipMemcpyAsync failed: %s\n", hipGetErrorString(err));
        pool.free_slot(slot);
        return;
    }

    // Synchronize to ensure transfer completes before returning (synchronous case)
    err = hipStreamSynchronize((hipStream_t)transfer_stream);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("page_in: hipStreamSynchronize failed: %s\n", hipGetErrorString(err));
        pool.free_slot(slot);
        return;
    }
#endif

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
    if (fds.empty() || fds[0] < 0) {
        return false;
    }
    io_reader = new llama_io_uring(fds.empty() ? -1 : fds[0], queue_depth);
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

    // Submit async read via io_uring to pinned staging buffer
    // Use page_idx as user_data to identify completion
    // The read goes into pinned_staging, not directly into VRAM
    void * staging_dst = pool.pinned_staging;
    io_reader->submit_read(0, staging_dst, page.size, page.file_offset, (uint64_t)page_idx);

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

        // Issue async GPU transfer from pinned_staging to VRAM slot
#if defined(GGML_USE_CUDA) && defined(__HIP_PLATFORM_AMD__)
        if (pool.pinned_staging != nullptr && transfer_stream != nullptr) {
            hipError_t err = hipMemcpyAsync(req.dst, pool.pinned_staging, page.size, hipMemcpyHostToDevice, (hipStream_t)transfer_stream);
            if (err != hipSuccess) {
                LLAMA_LOG_WARN("complete_prefetch: hipMemcpyAsync failed: %s\n", hipGetErrorString(err));
                in_flight.erase(fit);
                continue;
            }
        }
#endif

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
