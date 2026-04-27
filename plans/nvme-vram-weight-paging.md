# NVMe→VRAM Demand Paging for Oversized Models

## Goal

Enable running models larger than available VRAM by treating NVMe SSD as the
backing store for model weights. Layers not currently needed live on NVMe.
Before each layer is computed, its weights are paged in NVMe→VRAM directly
via SAM/ReBAR (bypassing RAM entirely). When VRAM is full, the LRU layer is
evicted (weights are read-only so no write-back needed — just mark the slot
free). RAM is never involved.

**Hardware context:** AMD Radeon AI Pro R9700 (gfx1201, 32GB VRAM) + NVMe SSD
with SAM/ReBAR enabled. Measured NVMe→VRAM bandwidth via SAM: ~5.7 GB/s.

## Architecture Overview

```
NVMe (GGUF file)
     |
     | io_uring O_DIRECT reads
     | into SAM-mapped BAR1 addresses
     v
VRAM Slot Pool  ←→  Weight Page Table  ←→  LRU Eviction
     |
     | ggml eval callback hook
     v
Compute Graph (forward pass)
```

Key insight: model weights are **read-only**. Eviction = mark slot free.
No write-back. This makes the pager much simpler than a general VM system.

## Phases

---

### Phase 1: Weight Page Table + VRAM Slot Pool
**File:** `src/llama-weight-pager.h` / `src/llama-weight-pager.cpp`
**Depends on:** nothing (pure data structures + hipMalloc)
**Testable standalone:** yes — unit test alloc/evict/LRU without IO

#### Data structures

```cpp
struct llama_weight_page {
    std::string  tensor_name;   // ggml tensor name (e.g. "blk.0.attn_q.weight")
    size_t       file_offset;   // byte offset in GGUF file
    size_t       size;          // tensor size in bytes
    int          slot_idx;      // VRAM slot index, -1 = not in VRAM
    uint64_t     last_used;     // monotonic counter for LRU
    void       * vram_ptr;      // pointer into slot pool, null if not loaded
};

struct llama_vram_pool {
    void   * base;              // hipMalloc'd contiguous VRAM block
    size_t   slot_size;         // bytes per slot (= max single-layer weight size)
    int      n_slots;           // total slots = floor(pool_bytes / slot_size)
    std::vector<bool>      used;
    std::vector<uint64_t>  lru_tick; // per-slot last-used tick

    int  alloc_slot();          // returns slot idx, evicts LRU if full
    void free_slot(int idx);
    void * slot_ptr(int idx) { return (uint8_t*)base + (size_t)idx * slot_size; }
};

struct llama_weight_pager {
    std::string              model_path;
    int                      fd;           // open fd for io_uring reads
    llama_vram_pool          pool;
    std::vector<llama_weight_page> pages;  // one per weight tensor
    std::unordered_map<std::string, int>   name_to_page;
    uint64_t                 tick = 0;

    // Ensure tensor is in VRAM. Returns VRAM pointer.
    void * ensure(const std::string & name);
    void   evict_lru();
    int    find_page(const std::string & name);
};
```

#### What to implement
- `llama_vram_pool::alloc_slot()` — scan `used[]`, if all full find min `lru_tick` and evict
- `llama_weight_pager::ensure()` — if already in VRAM update tick and return; else page_in()
- `llama_weight_pager::evict_lru()` — free slot, set page.slot_idx = -1, page.vram_ptr = null

**No IO in this phase.** `page_in()` is a stub that returns nullptr. Pool is
allocated with `hipMalloc`. Slot size is set externally (Phase 2 sets it).

---

### Phase 2: GGUF Offset Extraction + SAM Page-In
**File:** `src/llama-weight-pager.cpp` (page_in impl) + changes to `src/llama-model-loader.cpp`
**Depends on:** Phase 1, liburing, HIP

#### 2a: Extract tensor file offsets during model load

In `llama_model_loader::load_all_data`, before the main load loop, for each
weight tensor record its `(name, file_offset, size)` into a
`std::vector<llama_weight_page_info>` that gets stored on the model.

Key: `weight->offs` is already the byte offset into the GGUF file for each
tensor. This just needs to be saved instead of discarded after loading.

Add to `llama_model` (in `llama-model.h`):
```cpp
std::unique_ptr<llama_weight_pager> weight_pager; // null if paging disabled
```

Add to `common_params` (in `common.h`):
```cpp
bool    weight_paging       = false;   // --weight-paging flag
int     weight_paging_slots = -1;      // VRAM slots (-1 = auto from -ngl)
```

CLI flag: `--weight-paging` (enables the system)

#### 2b: SAM page_in implementation

```cpp
void llama_weight_pager::page_in(llama_weight_page & page) {
    int slot = pool.alloc_slot();       // may evict LRU
    void * dst = pool.slot_ptr(slot);   // fine-grained VRAM address via SAM

    // O_DIRECT aligned read directly into VRAM via BAR1
    // Use pread with O_DIRECT — simpler than io_uring for synchronous case
    size_t aligned_off  = page.file_offset & ~(size_t)(DIRECT_IO_ALIGN - 1);
    size_t prefix       = page.file_offset - aligned_off;
    // read into staging, copy to slot  (fallback path)
    // OR: pread directly into dst (SAM path — dst is BAR1-mapped VRAM)
    ssize_t n = pread(fd, dst, page.size, page.file_offset);

    page.slot_idx  = slot;
    page.vram_ptr  = (uint8_t*)dst;
    pool.used[slot] = true;
    pool.lru_tick[slot] = ++tick;
}
```

**Important:** For `pread` into a `hipExtMallocWithFlags(hipDeviceMallocFinegrained)`
address to work, the write must reach VRAM via BAR1. Test this with a small
synthetic benchmark before wiring into inference.

If `pread` into fine-grained VRAM works → full SAM path.
If not → use `hipHostMalloc` staging + `hipMemcpyAsync` as fallback. Still
avoids persistent RAM residency (staging buffer is temporary, reused).

#### Slot sizing
`slot_size = max weight tensor size across all layers`. Compute during offset
extraction pass. Typical value for 100B Q3 model: ~500MB per layer.

---

### Phase 3: ggml Eval Callback Hook
**File:** `src/llama.cpp` (context init) + `tools/server/server-context.cpp`
**Depends on:** Phase 1 + 2

ggml provides `ggml_backend_sched_set_eval_callback(sched, cb, userdata)`.
The callback fires **before** each graph node is executed.

```cpp
static bool weight_pager_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true; // yes we want the callback
    auto * pager = (llama_weight_pager *)user_data;

    // For each src tensor of t that is a weight tensor (has a page entry):
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!t->src[i]) break;
        auto it = pager->name_to_page.find(ggml_get_name(t->src[i]));
        if (it != pager->name_to_page.end()) {
            auto & page = pager->pages[it->second];
            void * vram = pager->ensure(page);
            // Redirect tensor data pointer to current VRAM slot
            t->src[i]->data = vram;
        }
    }
    return true;
}
```

Wire up in `llama_new_context_with_model` when `weight_pager` is set:
```cpp
if (model.weight_pager) {
    ggml_backend_sched_set_eval_callback(ctx->sched,
        weight_pager_eval_cb, model.weight_pager.get());
}
```

**Critical:** `t->src[i]->data` redirection must be safe to do mid-graph.
ggml backends read `data` at op execution time, not at graph build time,
so this is safe as long as the VRAM slot isn't evicted before the op runs.
Eviction is prevented because `ensure()` updates `last_used` and eviction
only touches the LRU slot.

---

### Phase 4: Async Prefetch via io_uring
**File:** `src/llama-weight-pager.cpp`
**Depends on:** Phase 1-3

The forward pass visits layers in order 0, 1, 2, ... N. While layer K is
being computed on GPU, prefetch layer K+1's weights from NVMe.

Replace synchronous `pread` in `page_in` with io_uring async submission:

```cpp
struct prefetch_req {
    int          page_idx;
    int          slot_idx;
    void       * dst;
};

void llama_weight_pager::submit_prefetch(int page_idx) {
    auto & page = pages[page_idx];
    if (page.slot_idx >= 0) return; // already in VRAM
    int slot = pool.alloc_slot();
    void * dst = pool.slot_ptr(slot);
    io_uring_submit_read(page.file_offset, dst, page.size, page_idx);
    in_flight[page_idx] = { page_idx, slot, dst };
}

void llama_weight_pager::complete_prefetch(int page_idx) {
    // wait for io_uring completion for page_idx
    // mark page as loaded
}
```

The eval callback becomes two-phase:
1. On layer K entry: `complete_prefetch(K)` (wait if still in flight)
2. After layer K dispatch: `submit_prefetch(K+1)`

This pipelines NVMe reads with GPU compute, hiding the ~5.7 GB/s transfer
latency behind computation.

---

### Phase 5: Integration, Flags, and Metrics
**File:** `common/arg.cpp`, `tools/server/server-context.cpp`, `tools/server/server-http.cpp`
**Depends on:** Phase 1-4

#### New CLI flags
```
--weight-paging              Enable NVMe→VRAM demand paging for model weights
--weight-paging-slots N      Number of VRAM layer slots (default: -ngl value)
--weight-paging-prefetch     Enable async prefetch of next layer (default: on)
```

#### Metrics endpoint additions (`/metrics`)
```
llama_weight_pager_page_ins_total
llama_weight_pager_evictions_total
llama_weight_pager_prefetch_hits_total   (layer was ready before needed)
llama_weight_pager_prefetch_misses_total (had to wait for IO)
llama_weight_pager_io_bytes_total
llama_weight_pager_io_seconds_total      (compute bandwidth: bytes/seconds)
```

#### Interaction with tiered KV cache
Weight paging and KV tiering are independent systems on the same hardware:
- Weight pager uses VRAM for layer weights, NVMe as backing
- KV tiered cache uses VRAM for hot KV, RAM for warm, NVMe for cold
- They share NVMe bandwidth — io_uring submission queues should be separate
  fds (model file vs KV slab files) to avoid head-of-line blocking

---

## File List (new/modified)

| File | Change |
|------|--------|
| `src/llama-weight-pager.h` | NEW — page table + pool declarations |
| `src/llama-weight-pager.cpp` | NEW — implementation |
| `src/llama-model-loader.cpp` | save tensor file offsets, init pager |
| `src/llama-model.h` | add `weight_pager` field |
| `src/llama.cpp` | register eval callback when pager present |
| `common/common.h` | add `weight_paging` params |
| `common/arg.cpp` | add `--weight-paging` flags |
| `tools/server/server-context.cpp` | pass pager config to init |
| `src/CMakeLists.txt` | add llama-weight-pager.cpp |

---

## Implementation Order for Handoff

Each phase can be handed to a separate model session:

1. **Session A** → Phase 1 only. Deliverable: `llama-weight-pager.h/cpp` with
   pool alloc/evict/LRU working, unit-testable without IO or HIP.

2. **Session B** → Phase 2. Deliverable: offset extraction wired into model
   loader + `page_in()` with SAM pread. Requires reading Phase 1 output.
   Test: load a model with `--weight-paging`, confirm pread hits VRAM slot.

3. **Session C** → Phase 3. Deliverable: eval callback hook wired, inference
   works correctly (outputs match non-paged run). Requires Phase 1+2 output.

4. **Session D** → Phase 4. Deliverable: async io_uring prefetch, pipeline
   bench showing >80% prefetch hit rate. Requires Phase 1-3 output.

5. **Session E** → Phase 5. Deliverable: flags, metrics, integration tests.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `pread` into fine-grained VRAM fails on kernel | Benchmark first (Phase 2); fall back to staging+hipMemcpyAsync |
| Slot eviction races with GPU compute | LRU never evicts a slot whose `last_used == tick` (current tick); GPU runs synchronously per-layer |
| VRAM pool too large, OOM | `--weight-paging-slots` cap; default to `-ngl` value |
| io_uring and KV tier contend on NVMe | Separate submission queues; weight pager gets priority |
| Tensor data pointer redirect unsafe | Test with small model first; ggml reads `data` at execution not graph build |

