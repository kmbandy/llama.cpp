// MAD-420 — unit tests for the pre-carved size-class slot allocator.
//
// Standalone, CPU-only, no test framework. Exercises:
//   1. PageCatalog::page_size_histogram (sizes/counts, non-slottable excluded).
//   2. The pre-carve solver (proportional counts, within budget, every class
//      >= 1 slot, single-size input reproduces the uniform layout).
//   3. THE CRASH: a request for the LARGEST class succeeds after the arena is
//      full of small slots. This is exactly the GLM-5.2 abort: with on-demand
//      carving the arena filled with small slots and a large page had no
//      adequate class to take or evict. Pre-carving guarantees a slot per
//      class so this can never return -1 while the arena has space.
//   4. Eviction stays within class and respects pins.
//
// Build (in-tree, no GPU needed):
//   g++ -std=c++17 -I include -I ggml/include -I src -I . \
//       tests/test-wp-size-class-slots.cpp \
//       src/weight-pager/wp-pool.cpp src/weight-pager/wp-page-catalog.cpp \
//       -L build/bin -Wl,-rpath,$(pwd)/build/bin \
//       -lllama -lggml-base -lggml-cpu -o /tmp/t-sizeclass

#include "weight-pager/wp-page-catalog.h"
#include "weight-pager/wp-pool.h"

#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct ScopedEnv {
    explicit ScopedEnv(const char * name_) : name(name_) {
        const char * v = std::getenv(name);
        if (v != nullptr) { had = true; old = v; }
    }
    ~ScopedEnv() {
        if (had) setenv(name, old.c_str(), 1);
        else     unsetenv(name);
    }
    const char * name;
    bool        had = false;
    std::string old;
};

#define EXPECT(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, (msg)); ++fails; } \
} while (0)
#define EXPECT_EQ_INT(got, want, msg) do { \
    auto _g = (got); auto _w = (want); \
    if (!(_g == _w)) { std::fprintf(stderr, "  FAIL: %s (line %d): %s -- got=%lld want=%lld\n", \
        __func__, __LINE__, (msg), (long long)(_g), (long long)(_w)); ++fails; } \
} while (0)

// ---------------------------------------------------------------------------
// 1. PageCatalog::page_size_histogram
// ---------------------------------------------------------------------------
int test_histogram_excludes_non_slottable() {
    int fails = 0;
    wp::PageCatalog cat;

    // Consolidated expert tensor: parent (NOT slottable) + N sub-experts
    // (slottable, all of per_expert_size).
    const size_t per_expert = 1000;
    const int    n_experts  = 4;
    cat.add_consolidated_experts("blk.0.ffn_gate_exps.weight", /*file_idx=*/0,
                                 /*file_offset=*/0, per_expert * (size_t) n_experts,
                                 n_experts);

    // A second consolidated tensor with a different per-expert size.
    const size_t per_expert_b = 4000;
    cat.add_consolidated_experts("blk.0.ffn_down_exps.weight", 0, 0,
                                 per_expert_b * 2, 2);

    // Pinned (always-resident) entry: must be EXCLUDED even though its name
    // matches an expert role pattern.
    cat.add_pinned("blk.0.ffn_up.weight", /*device_ptr=*/(void *) 0x1, /*bytes=*/7777);

    // Non-expert dense tensor: is_expert stays false -> excluded.
    cat.add("blk.0.attn_q.weight", 0, 0, 555);

    // Per-expert standalone (not consolidated): is_expert && !is_consolidated
    // -> slottable, counted at its own size.
    cat.add("blk.0.ffn_up.2.weight", 0, 0, 2000);

    std::map<size_t, int> h = cat.page_size_histogram();

    // per_expert (1000) x 4, per_expert_b (4000) x 2, standalone 2000 x 1.
    EXPECT_EQ_INT((int) h.size(), 3, "histogram has 3 distinct slottable sizes");
    EXPECT_EQ_INT(h[1000], 4, "four 1000-byte sub-experts");
    EXPECT_EQ_INT(h[2000], 1, "one 2000-byte per-expert standalone");
    EXPECT_EQ_INT(h[4000], 2, "two 4000-byte sub-experts");

    // The pinned 7777, the parent metadata, and the dense 555 must NOT appear.
    auto it_pin = h.find(7777);
    EXPECT(it_pin == h.end(), "pinned page excluded from histogram");
    auto it_dense = h.find(555);
    EXPECT(it_dense == h.end(), "non-expert dense page excluded");

    // Empty catalog -> empty histogram.
    wp::PageCatalog empty;
    std::map<size_t, int> he = empty.page_size_histogram();
    EXPECT(he.empty(), "empty catalog yields empty histogram");
    return fails;
}

// ---------------------------------------------------------------------------
// 2. Pre-carve solver: proportional counts, within budget, every class >= 1.
// ---------------------------------------------------------------------------
int test_precarve_solver_proportional() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    // Arena budget = n_slots * slot_size = 2 * 256 = 512 B.
    // Histogram: 4 small (64 B) + 1 large (256 B). total_pages=5,
    // weighted = 4*64 + 1*256 = 512, avg = 102.4. K = floor(512/102.4)=5.
    // f_small=0.8 -> k_small=round(4)=4; f_large=0.2 -> k_large=round(1)=1.
    // total = 4*64 + 1*256 = 512 <= 512. Exact fit.
    std::map<size_t, int> hist;
    hist[64]  = 4;
    hist[256] = 1;

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/2, /*slot_size=*/256,
                     /*device_idx=*/-1, /*extra_alignment=*/1, &hist),
           "precarve init");
    EXPECT(pool.size_class_slots_enabled(), "size-class mode on");
    EXPECT(pool.size_class_precarved(),     "precarved flag set");

    // 5 slots total: 4 small (class 64) + 1 large (class 256).
    EXPECT_EQ_INT(pool.n_slots(), 5, "K = 5 slots");
    EXPECT(pool.pool_size() >= 4 * 64 + 256, "arena >= carved bytes");
    EXPECT(pool.pool_size() == 512u, "arena budget preserved");

    // Class sizes: count slots per class.
    std::map<size_t, int> got;
    for (int i = 0; i < pool.n_slots(); ++i) got[pool.slot_size(i)] += 1;
    EXPECT_EQ_INT(got[64],  4, "four small slots");
    EXPECT_EQ_INT(got[256], 1, "one large slot");

    // Every class got >= 1 slot (guaranteed even with a tiny minority class).
    EXPECT(got[256] >= 1, "large class has >= 1 slot");
    EXPECT(got[64]  >= 1, "small class has >= 1 slot");

    // Offsets are contiguous and within the arena; slots of the same class
    // are back-to-back.
    std::vector<size_t> offs;
    for (int i = 0; i < pool.n_slots(); ++i) offs.push_back((size_t)((uint8_t*)pool.slot_ptr(i) - (uint8_t*)pool.pool_base()));
    for (size_t o : offs) EXPECT(o + 64 <= pool.pool_size() || o + 256 <= pool.pool_size(), "offset in arena");
    return fails;
}

// Pre-carve must trim rounding overshoot and never exceed the arena, while
// keeping every class >= 1.
int test_precarve_solver_trims_overshoot() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    // Budget = 1 * 256 = 256 B. Histogram 50/50 small(64)/large(256):
    // total=2, weighted=320, avg=160. K=floor(256/160)=1 (capped >=1).
    // f_small=0.5 -> round(0.5)=1 (round half away from zero); same for large.
    // total_k=2, bytes=64+256=320 > 256 -> trim. Largest class (256) drops to
    // 0? No -- protected: kc<=1 skipped. So 256 stays at 1, small 64 stays at
    // 1. bytes=320 > 256 still. Every class is at 1 -> can't trim -> carve
    // would overshoot. carve_size_classes_ must detect this and either still
    // fit (it can't) -> it returns false (abandon pre-carve) OR the init
    // falls back. The invariant is: never exceed the arena. Verify n_slots
    // and arena consistency hold whichever way it resolved.
    std::map<size_t, int> hist;
    hist[64]  = 1;
    hist[256] = 1;

    wp::PoolAllocator pool;
    // n_slots=1, slot_size=256 -> arena 256.
    bool ok = pool.init(buft, 1, 256, -1, 1, &hist);
    EXPECT(ok, "init succeeds (carve or fallback)");
    // Carved bytes must never exceed the arena.
    size_t carved = 0;
    for (int i = 0; i < pool.n_slots(); ++i) carved += pool.slot_size(i);
    EXPECT(carved <= pool.pool_size(), "carved bytes never exceed arena");
    // Either it pre-carved (>=1 slot per class, total <= arena) or it fell
    // back to on-demand carve (precarved false, n_slots 0). Both are valid;
    // the hard rule is just "no abort, no overshoot".
    if (pool.size_class_precarved()) {
        std::map<size_t, int> got;
        for (int i = 0; i < pool.n_slots(); ++i) got[pool.slot_size(i)] += 1;
        for (const auto & kv : got) EXPECT(kv.second >= 1, "every precarved class >= 1 slot");
    }
    return fails;
}

// A single-class histogram must reproduce the uniform fixed-slot layout:
// K = arena / slot_size slots, all at stride slot_size, offsets 0, s, 2s, ...
int test_precarve_single_size_matches_uniform() {
    int fails = 0;

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    // Uniform path (env unset): 4 slots x 256 B.
    {
        ScopedEnv env("WP_SIZE_CLASS_SLOTS");  // unsets on destruction
        wp::PoolAllocator uni;
        EXPECT(uni.init(buft, 4, 256), "uniform init");
        EXPECT_EQ_INT(uni.n_slots(), 4, "uniform n_slots");
        EXPECT_EQ_INT((size_t)((uint8_t*)uni.slot_ptr(2) - (uint8_t*)uni.pool_base()),
                      2u * 256u, "uniform slot 2 offset");
        EXPECT_EQ_INT(uni.slot_size(0), 256u, "uniform slot size");
    }

    // Pre-carve path with a single class: histogram {256: 8}, arena 4*256=1024.
    // K = 1024/256 = 4 slots of 256. Same layout.
    {
        ScopedEnv env("WP_SIZE_CLASS_SLOTS");
        setenv("WP_SIZE_CLASS_SLOTS", "1", 1);
        std::map<size_t, int> hist;
        hist[256] = 8;
        wp::PoolAllocator pc;
        EXPECT(pc.init(buft, 4, 256, -1, 1, &hist), "precarve init single class");
        EXPECT(pc.size_class_precarved(), "precarved");
        EXPECT_EQ_INT(pc.n_slots(), 4, "K = 4 slots (matches uniform)");
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ_INT(pc.slot_size(i), 256u, "single-class slot size = max");
            EXPECT_EQ_INT((size_t)((uint8_t*)pc.slot_ptr(i) - (uint8_t*)pc.pool_base()),
                          (size_t) i * 256u, "single-class offset matches uniform");
        }
        // Behaviour matches: alloc 4, then 5th evicts LRU slot 0.
        int ev = -1;
        pc.set_eviction_callback([&](int s) { ev = s; });
        int s0 = pc.alloc_slot(256);
        int s1 = pc.alloc_slot(256);
        int s2 = pc.alloc_slot(256);
        int s3 = pc.alloc_slot(256);
        EXPECT(s0 >= 0 && s1 >= 0 && s2 >= 0 && s3 >= 0, "four allocs succeed");
        int s4 = pc.alloc_slot(256);
        EXPECT(s4 == s0, "5th alloc evicts LRU slot 0 (matches uniform)");
        EXPECT_EQ_INT(ev, s0, "eviction callback fired for slot 0");
    }
    return fails;
}

// ---------------------------------------------------------------------------
// 3. THE CRASH: large request succeeds after the arena is full of small slots.
//    With on-demand carving this returned -1 and aborted. Pre-carving reserves
//    a large slot up front, so the request takes it.
// ---------------------------------------------------------------------------
int test_precarve_large_request_after_small_fill() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    std::map<size_t, int> hist;
    hist[64]  = 4;
    hist[256] = 1;

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, 2, 256, -1, 1, &hist), "precarve init");
    EXPECT_EQ_INT(pool.n_slots(), 5, "4 small + 1 large");

    int ev = -1;
    pool.set_eviction_callback([&](int s) { ev = s; });

    // Fill the small class completely.
    int s0 = pool.alloc_slot(64);
    int s1 = pool.alloc_slot(64);
    int s2 = pool.alloc_slot(64);
    int s3 = pool.alloc_slot(64);
    EXPECT(s0 >= 0 && s1 >= 0 && s2 >= 0 && s3 >= 0, "four small allocs fill small class");

    // THE CRASH REPRO: request the large class. With on-demand carving this
    // returned -1 (arena full of small slots, no adequate class to evict).
    // With pre-carving the large slot is reserved and free -> succeeds.
    int s4 = pool.alloc_slot(256);
    EXPECT(s4 >= 0, "LARGE request succeeds after small fill (the GLM crash)");
    EXPECT_EQ_INT(pool.slot_size(s4), 256u, "large slot has large class size");
    EXPECT_EQ_INT(ev, -1, "no eviction needed -- large slot was free");

    // A second large request must evict within the large class (the only
    // adequate slot is s4, which is unpinned) -- NOT abort.
    int s5 = pool.alloc_slot(256);
    EXPECT(s5 == s4, "second large reuses the large slot via within-class eviction");
    EXPECT_EQ_INT(ev, s4, "eviction callback fired for the large slot");

    // Pin the only large slot and request large again. Every adequate slot
    // (class >= 256) is pinned -- genuine backpressure, returns -1. This is
    // NOT the crash (the crash was: arena had usable space but no adequate
    // class). Here the adequate class exists but is pinned; -1 is correct and
    // matches the uniform path's "all pinned" semantics.
    pool.pin_slot(s4);
    ev = -1;
    int s6 = pool.alloc_slot(256);
    EXPECT_EQ_INT(s6, -1, "all adequate slots pinned -> -1 (backpressure, not crash)");
    EXPECT_EQ_INT(ev, -1, "no eviction while the only adequate slot is pinned");
    return fails;
}

// ---------------------------------------------------------------------------
// 4. Eviction stays within class and respects pins.
// ---------------------------------------------------------------------------
int test_precarve_eviction_within_class_and_pins() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    // Two classes: small (64) and large (128). Histogram 50/50 by pages.
    // Budget = 3 * 128 = 384 B. total=4, weighted=64*2+128*2=384, avg=96.
    // K=floor(384/96)=4. f_small=0.5->round(2)=2; f_large=0.5->round(2)=2.
    // bytes=2*64+2*128=384 == budget. Exact fit: 2 small + 2 large.
    std::map<size_t, int> hist;
    hist[64]  = 2;
    hist[128] = 2;

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, 3, 128, -1, 1, &hist), "precarve init");
    {
        std::map<size_t, int> got;
        for (int i = 0; i < pool.n_slots(); ++i) got[pool.slot_size(i)] += 1;
        EXPECT_EQ_INT(got[64],  2, "two small slots");
        EXPECT_EQ_INT(got[128], 2, "two large slots");
    }

    int ev = -1;
    pool.set_eviction_callback([&](int s) { ev = s; });

    // Fill BOTH classes completely so no free slot remains -- only then does
    // alloc fall through to pick_size_class_victim_, which evicts within the
    // SMALLEST adequate class (within-class eviction, no cross-class waste).
    int a = pool.alloc_slot(64);    // small slot
    int b = pool.alloc_slot(64);    // small slot (small class full)
    int lg0 = pool.alloc_slot(128); // large slot
    int lg1 = pool.alloc_slot(128); // large slot (large class full)
    EXPECT(a >= 0 && b >= 0 && lg0 >= 0 && lg1 >= 0, "fill both classes");
    EXPECT_EQ_INT(pool.slot_size(a), 64u,  "a is small");
    EXPECT_EQ_INT(pool.slot_size(lg0), 128u, "lg0 is large");

    // Make `a` recently used across all slots, so `b` is the global LRU AND
    // the LRU within the small class. A small request must evict `b` (within
    // the small class), NOT a large slot.
    pool.mark_used(a);
    pool.mark_used(lg0);
    pool.mark_used(lg1);
    ev = -1;
    int c = pool.alloc_slot(64);
    EXPECT(c == b, "small eviction reuses LRU small slot (within class)");
    EXPECT_EQ_INT(pool.slot_size(c), 64u, "evicted slot is small class");
    EXPECT_EQ_INT(ev, b, "eviction callback fired for the small LRU");

    // Pin both small slots. A small request now has no free small slot and no
    // evictable small slot -- the existing cross-class upward fallback takes a
    // free larger slot if one exists. Free a large slot, then request small:
    // it must take the free large slot (no abort) rather than evict.
    pool.pin_slot(a);
    pool.pin_slot(c);
    pool.release_slot(lg1);     // free one large slot
    ev = -1;
    int d = pool.alloc_slot(64);
    EXPECT(d == lg1, "small request with small class pinned takes the free large slot");
    EXPECT_EQ_INT(pool.slot_size(d), 128u, "fallback slot is the larger class");
    EXPECT_EQ_INT(ev, -1, "no eviction -- a free larger slot was available");

    // Now pin every slot of class >= 128 and request large -- all adequate
    // slots pinned -> -1. This is backpressure, not the crash.
    pool.pin_slot(d);
    pool.pin_slot(lg0);
    ev = -1;
    int e = pool.alloc_slot(128);
    EXPECT_EQ_INT(e, -1, "all adequate slots pinned -> -1 (no abort, just backpressure)");
    EXPECT_EQ_INT(ev, -1, "no eviction while all adequate slots pinned");
    return fails;
}

// GLM-5.2-shaped distribution (5 distinct sizes, 65% small) at toy scale.
// Confirms the largest-minority class (1.8% of pages) still gets >= 1 slot,
// the small class gets the bulk, and the largest request succeeds after the
// small slots are filled -- the exact shape of the production crash.
int test_precarve_glm_shaped_distribution() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    // Sizes proportional to GLM-5.2 UD-Q2_K_XL sub-page MiB sizes; counts
    // proportional to the measured page counts (1024/256/18688/512/37888),
    // scaled down ~100x to keep the arena small. max page = 6375 B.
    std::map<size_t, int> hist;
    hist[3469] = 379;   // 64.9% of pages (gate/up)
    hist[3938] =   5;   //  0.9%
    hist[4594] = 186;   // 32.0%
    hist[5156] =   3;   //  0.4%
    hist[6375] =  10;   // 1.8% (ffn_down, largest)

    long long total_pages = 0, weighted = 0;
    for (const auto & kv : hist) { total_pages += kv.second; weighted += (long long)kv.second * (long long)kv.first; }
    const double avg = (double) weighted / (double) total_pages;
    const size_t slot_size = 6375;
    const int n_slots = (int) ((double) total_pages * avg / (double) slot_size);
    EXPECT(n_slots >= 1, "arena big enough for at least one slot");

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, n_slots, slot_size, -1, 1, &hist), "precarve init glm-shaped");
    EXPECT(pool.size_class_precarved(), "precarved");

    // Every distinct class must have >= 1 slot (the 1.8% largest class too).
    // Classes are aligned up to the buffer alignment, so compare by the
    // pool's actual class sizes, not the raw histogram keys.
    std::map<size_t, int> got;
    for (int i = 0; i < pool.n_slots(); ++i) got[pool.slot_size(i)] += 1;
    EXPECT_EQ_INT((int) got.size(), 5, "all 5 classes carved");
    for (const auto & kv : got) {
        EXPECT(kv.second >= 1, "every carved class gets >= 1 slot");
    }

    // The smallest class (aligned from 3469) must hold the bulk of slots;
    // the largest class (aligned from 6375, 1.8% demand) the fewest -- but
    // still >= 1. This is the headline: a 1.8% minority class is no longer
    // starved to zero (the crash), it is reserved up front.
    size_t smallest_cls = got.begin()->first;
    size_t largest_cls  = got.rbegin()->first;
    EXPECT(got[smallest_cls] > got[largest_cls], "small class has more slots than largest class");
    EXPECT(got[largest_cls] >= 1, "largest minority class still gets >= 1 slot");

    size_t carved = 0;
    for (int i = 0; i < pool.n_slots(); ++i) carved += pool.slot_size(i);
    EXPECT(carved <= pool.pool_size(), "carved <= arena");

    // THE CRASH REPRO for this shape: fill the small class, then the largest
    // request must succeed (a largest-class slot is reserved).
    int small_n = got[smallest_cls];
    for (int i = 0; i < small_n; ++i) {
        int s = pool.alloc_slot(3469);
        EXPECT(s >= 0, "small alloc fills small class");
    }
    int big = pool.alloc_slot(6375);
    EXPECT(big >= 0, "largest request succeeds after small fill (GLM crash repro)");
    EXPECT(pool.slot_size(big) >= 6375u, "largest slot can fit the largest request");
    EXPECT_EQ_INT(pool.slot_size(big), largest_cls, "largest request lands in the largest class");
    return fails;
}


// ---------------------------------------------------------------------------
// 5. THE PIN FLOOR -- the case every earlier test in this file missed.
//
// The existing GLM-shaped test fills the small class and then makes ONE large
// request. That passes with a single large slot, so it could never see the real
// failure: a whole ensure_batch is PINNED at once and alloc_slot will not evict
// a pinned slot, so the class must hold a whole block's expert union
// simultaneously.
//
// Shape below is GLM-5.2 scaled down 16x, preserving the ratio that matters:
//   large class (256 B): 64 pages, concentrated in 4 blocks at 16 each
//   small class ( 64 B): 3584 pages, spread over 56 blocks at 64 each
// The large class is 1.75% of all pages -- exactly GLM's share -- so demand
// share alone buys it ~4 slots while one block needs 16 pinned at once.
// ---------------------------------------------------------------------------
namespace {

// hist + per-block counts for the scaled-down GLM shape.
void make_concentrated_shape(std::map<size_t, int> & hist,
                             std::map<size_t, std::map<int, int>> & layers) {
    hist.clear();
    layers.clear();
    // Large class: 4 blocks x 16 pages.
    for (int b = 0; b < 4; ++b) {
        layers[256][b] = 16;
        hist[256]     += 16;
    }
    // Small class: 56 blocks x 64 pages.
    for (int b = 4; b < 60; ++b) {
        layers[64][b] = 64;
        hist[64]     += 64;
    }
}

const int kBlockUnion = 16;   // pages of the large class owned by one block

}  // namespace

int test_pin_floor_holds_a_whole_block() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    std::map<size_t, int> hist;
    std::map<size_t, std::map<int, int>> layers;
    make_concentrated_shape(hist, layers);

    // Arena = 68 * 256 = 17408 B.
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/68, /*slot_size=*/256,
                     /*device_idx=*/-1, /*extra_alignment=*/1, &hist, &layers),
           "precarve init with layer counts");
    EXPECT(pool.size_class_precarved(), "precarved flag set");

    std::map<size_t, int> got;
    for (int i = 0; i < pool.n_slots(); ++i) got[pool.slot_size(i)] += 1;

    // The floor: the large class must hold one block's whole union.
    EXPECT(got[256] >= kBlockUnion, "large class has at least one block's union of slots");
    // The small class must still get the bulk of the pool -- a floor that ate
    // the arena would be its own regression.
    EXPECT(got[64] > got[256], "small class still gets the majority of slots");

    size_t carved = 0;
    for (int i = 0; i < pool.n_slots(); ++i) carved += pool.slot_size(i);
    EXPECT(carved <= pool.pool_size(), "carved <= arena");

    // THE ACTUAL CRASH: allocate a whole block's worth of the large class and
    // PIN each one before taking the next, exactly as ensure_batch does. With
    // demand-share sizing this returns -1 partway through and the caller
    // aborts. Warm the small class first so the pool is under real pressure.
    for (int i = 0; i < got[64]; ++i) {
        (void) pool.alloc_slot(64);
    }
    int pinned_ok = 0;
    for (int i = 0; i < kBlockUnion; ++i) {
        int s = pool.alloc_slot(256);
        if (s < 0) break;
        pool.pin_slot(s);
        ++pinned_ok;
    }
    EXPECT_EQ_INT(pinned_ok, kBlockUnion,
                  "a whole block's union of the large class allocates while pinned");
    EXPECT_EQ_INT(pool.n_pinned(), kBlockUnion, "all of them stayed pinned");
    return fails;
}

// Same shape, layer counts withheld: documents that demand-share sizing alone
// under-provisions the concentrated class. This is the witness for the bug --
// if it ever starts passing, the floor is being applied by some other path and
// this test has stopped proving anything.
int test_without_layer_counts_underprovisions() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    std::map<size_t, int> hist;
    std::map<size_t, std::map<int, int>> layers;
    make_concentrated_shape(hist, layers);

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, 68, 256, -1, 1, &hist, /*layer_counts=*/nullptr),
           "precarve init without layer counts");

    std::map<size_t, int> got;
    for (int i = 0; i < pool.n_slots(); ++i) got[pool.slot_size(i)] += 1;
    EXPECT(got[256] < kBlockUnion,
           "without layer counts the concentrated class is under-provisioned "
           "(this is the bug; the floor test above is the fix)");

    // And it fails exactly the way GLM did: -1 partway through a pinned batch.
    for (int i = 0; i < got[64]; ++i) (void) pool.alloc_slot(64);
    int pinned_ok = 0;
    for (int i = 0; i < kBlockUnion; ++i) {
        int s = pool.alloc_slot(256);
        if (s < 0) break;
        pool.pin_slot(s);
        ++pinned_ok;
    }
    EXPECT(pinned_ok < kBlockUnion, "pinned batch runs out of the class (the GLM abort)");
    return fails;
}

// Floors that cannot fit must abandon the pre-carve, not carve a pool that will
// abort on the GPU minutes later. The uniform path still works in that case.
int test_pin_floor_too_big_falls_back() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    std::map<size_t, int> hist;
    std::map<size_t, std::map<int, int>> layers;
    make_concentrated_shape(hist, layers);

    // Arena = 4 * 256 = 1024 B; the large class floor alone needs 16*256 = 4096.
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, 4, 256, -1, 1, &hist, &layers),
           "init succeeds by falling back");
    EXPECT(!pool.size_class_precarved(), "pre-carve abandoned when floors do not fit");

    // The fallback must still serve a large request rather than return -1.
    int s = pool.alloc_slot(256);
    EXPECT(s >= 0, "fallback path still serves the largest request");
    return fails;
}

}  // namespace

int main() {
    int total = 0;
    struct Named { const char * name; int (*fn)(); };
    Named tests[] = {
        { "histogram_excludes_non_slottable",      test_histogram_excludes_non_slottable      },
        { "precarve_solver_proportional",          test_precarve_solver_proportional          },
        { "precarve_solver_trims_overshoot",       test_precarve_solver_trims_overshoot       },
        { "precarve_single_size_matches_uniform",  test_precarve_single_size_matches_uniform  },
        { "precarve_large_request_after_small_fill", test_precarve_large_request_after_small_fill },
        { "precarve_eviction_within_class_and_pins", test_precarve_eviction_within_class_and_pins },
        { "precarve_glm_shaped_distribution",      test_precarve_glm_shaped_distribution      },
        { "pin_floor_holds_a_whole_block",         test_pin_floor_holds_a_whole_block         },
        { "without_layer_counts_underprovisions",  test_without_layer_counts_underprovisions  },
        { "pin_floor_too_big_falls_back",          test_pin_floor_too_big_falls_back          },
    };
    for (const auto & t : tests) {
        std::fprintf(stderr, "RUN  test_%s\n", t.name);
        int f = t.fn();
        std::fprintf(stderr, "%s test_%s (%d failure%s)\n",
                     f == 0 ? "PASS" : "FAIL", t.name, f, f == 1 ? "" : "s");
        total += f;
    }
    std::fprintf(stderr, "\n=== %s: %d total failures ===\n",
                 total == 0 ? "PASS" : "FAIL", total);
    return total == 0 ? 0 : 1;
}
