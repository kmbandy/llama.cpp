#include "../src/weight-pager/wp-host-arena.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using wp::HostArena;

static const size_t ENTRY = 4096 * 4;   // 16 KiB fake page

// build-hip compiles tests as RelWithDebInfo (NDEBUG defined), under which
// assert()'s argument -- including any side-effecting call inside it -- is
// not evaluated at all, not just unchecked. Every sibling test-wp-*.cpp
// (e.g. test-wp-expert-worker.cpp) avoids assert() for exactly this reason
// and uses this require()-throws idiom instead; follow the same convention
// here so the checks actually run.
static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct CountingAlloc {
    size_t calls = 0, fail_after = SIZE_MAX, freed = 0;
    HostArena::Allocator alloc() {
        return [this](size_t n) -> void * {
            if (calls++ >= fail_after) return nullptr;
            void * p = nullptr;
            if (posix_memalign(&p, 4096, n) != 0) return nullptr;
            return p;
        };
    }
    HostArena::Deallocator dealloc() {
        return [this](void * p, size_t) { ++freed; std::free(p); };
    }
};

static HostArena::Config cfg(size_t entries, size_t chunk_entries = 4) {
    HostArena::Config c;
    c.budget_bytes      = entries * ENTRY;
    c.entry_bytes       = ENTRY;
    c.chunk_bytes       = chunk_entries * ENTRY;
    c.spec_frac_pct     = 25;
    c.pinned_cap_pct    = 90;
    c.read_inflight_max = 16;
    // No retention cap: every existing test here predates tier_bytes and
    // exercises the raw state machine, not the trim policy (see
    // test_tier_cap_* below for that). Setting the cap to the whole budget
    // means resident-unpinned bytes can never exceed it before the ordinary
    // begin_read()-driven eviction path would have kicked in anyway.
    c.tier_bytes        = c.budget_bytes;
    return c;
}

static void test_init_chunked_and_shrink_on_failure() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(8), a.alloc(), a.dealloc()), "init 8 entries");
    require(arena.entry_count() == 8, "entry_count 8");
    require(arena.chunk_count() == 2, "chunk_count 2");
    arena.shutdown();
    require(a.freed == 2, "shutdown frees both chunks");

    CountingAlloc b;
    b.fail_after = 1;                         // second chunk fails
    HostArena small;
    require(small.init(cfg(8), b.alloc(), b.dealloc()), "init shrinks on chunk failure");
    require(small.entry_count() == 4, "shrank, did not fall back");
    require(small.chunk_count() == 1, "only the first chunk landed");

    CountingAlloc c;
    c.fail_after = 0;
    HostArena none;
    require(!none.init(cfg(8), c.alloc(), c.dealloc()), "zero entries fit -> init fails");
    require(!none.is_initialized(), "not initialized after a failed init");
}

static void test_read_state_machine() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(4), a.alloc(), a.dealloc()), "init");
    void * data = nullptr; HostArena::Handle h = 0;
    require(arena.begin_read(7, false, &data, &h), "begin_read 7");
    require(data != nullptr && h != HostArena::kInvalidHandle, "begin_read yields data+handle");
    require(arena.state_of(7) == HostArena::State::Reading, "7 is Reading");
    require(arena.reading_count() == 1, "reading_count 1");
    // cannot borrow while Reading, cannot begin a second read of the same page
    const void * src = nullptr; HostArena::Handle bh = 0;
    require(!arena.borrow(7, &src, &bh), "cannot borrow while Reading");
    void * d2 = nullptr; HostArena::Handle h2 = 0;
    require(!arena.begin_read(7, false, &d2, &h2), "cannot double begin_read");
    std::memset(data, 0xAB, ENTRY);
    arena.finish_read(7, h, true);
    require(arena.state_of(7) == HostArena::State::Resident, "7 is Resident");
    require(arena.resident_bytes() == ENTRY, "resident_bytes ENTRY");
    require(arena.borrow(7, &src, &bh), "borrow 7");
    require(((const unsigned char *) src)[0] == 0xAB, "borrowed bytes match what was written");
    arena.release(7, bh);
    // failed read frees the entry
    require(arena.begin_read(8, false, &data, &h), "begin_read 8");
    arena.finish_read(8, h, false);
    require(arena.state_of(8) == HostArena::State::Free, "failed read frees the entry");
    require(arena.resident_count() == 1, "resident_count still 1");
}

static void test_lru_never_evicts_reading_borrowed_or_pinned() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(4), a.alloc(), a.dealloc()), "init");
    void * data; HostArena::Handle h[5];
    for (int i = 0; i < 4; ++i) {
        require(arena.begin_read(i, false, &data, &h[i]), "read");
        if (i != 2) arena.finish_read(i, h[i], true);     // page 2 stays Reading
    }
    // page 0: borrowed; page 1: pinned; page 2: Reading; page 3: plain
    const void * src; HostArena::Handle b0;
    require(arena.borrow(0, &src, &b0), "borrow 0");
    require(arena.pin(1), "pin 1");
    // arena full: page 4 must evict page 3 (the only evictable)
    require(arena.begin_read(4, false, &data, &h[4]), "read 4");
    require(arena.state_of(3) == HostArena::State::Free, "3 evicted");
    require(arena.evictions() == 1, "evictions 1");
    arena.finish_read(4, h[4], true);
    // borrow 4 as well: now 0 and 4 are borrowed, 1 pinned, 2 Reading -> refusal
    HostArena::Handle b4;
    require(arena.borrow(4, &src, &b4), "borrow 4");
    require(!arena.begin_read(5, false, &data, &h[0]), "refused");
    require(arena.begin_read_refusals() == 1, "refusals 1");
    // release page 0 (the LRU-older borrowed one, but it sits BEHIND 4 in
    // the list only if touched later -- it was touched first, so it is at
    // the front): eviction must find it by scanning past nothing
    arena.release(0, b0);
    require(arena.begin_read(5, false, &data, &h[0]), "read 5");
    require(arena.state_of(0) == HostArena::State::Free, "0 evicted");
    arena.finish_read(5, h[0], true);
    // now: 1 pinned, 2 Reading, 4 borrowed (LRU-older than 5), 5 plain (MRU).
    // The scan must SKIP the borrowed entry 4 at the front and evict 5.
    require(arena.begin_read(6, false, &data, &h[1]), "read 6 skips borrowed front");
    require(arena.state_of(5) == HostArena::State::Free, "5 evicted, not 4");
    require(arena.state_of(4) == HostArena::State::Resident, "4 survives (borrowed)");
    arena.release(4, b4);
    arena.finish_read(2, h[2], true);
}

static void test_speculative_drained_first_and_promoted_on_demand_hit() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(8), a.alloc(), a.dealloc()), "init");   // spec cap = 2 entries
    void * data; HostArena::Handle h;
    // 6 demand pages, MRU order 0..5
    for (int i = 0; i < 6; ++i) { require(arena.begin_read(i, false, &data, &h), "demand read"); arena.finish_read(i, h, true); }
    // 2 speculative pages fill the spec cap
    require(arena.begin_read(10, true, &data, &h), "spec read 10"); arena.finish_read(10, h, true);
    require(arena.begin_read(11, true, &data, &h), "spec read 11"); arena.finish_read(11, h, true);
    require(arena.spec_bytes() == 2 * ENTRY, "spec cap filled");
    // a third speculative read must evict a speculative entry, not page 0
    require(arena.begin_read(12, true, &data, &h), "spec read 12"); arena.finish_read(12, h, true);
    require(arena.state_of(10) == HostArena::State::Free, "10 evicted");
    require(arena.state_of(0)  == HostArena::State::Resident, "0 survives");
    require(arena.spec_evicted_unused() == 1, "10 counted as unused");
    // demand hit promotes 11 out of the speculative side
    const void * src; HostArena::Handle b;
    require(arena.borrow(11, &src, &b), "borrow 11");
    arena.release(11, b);
    require(arena.spec_bytes() == ENTRY, "spec_bytes drops after promotion");
    require(arena.spec_promotions() == 1, "promotion counted");
    // arena full with 6 demand + 2 spec: a DEMAND read evicts the remaining
    // speculative entry (12) before the oldest demand page (0)
    require(arena.begin_read(13, false, &data, &h), "demand read 13");
    require(arena.state_of(12) == HostArena::State::Free, "12 evicted");
    require(arena.state_of(0)  == HostArena::State::Resident, "0 still survives");
}

static void test_inflight_cap_and_pinned_cap() {
    CountingAlloc a;
    HostArena::Config c = cfg(8);
    c.read_inflight_max = 2;
    c.pinned_cap_pct    = 50;   // 4 of 8
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");
    void * data; HostArena::Handle h[3];
    require(arena.begin_read(0, false, &data, &h[0]), "read 0");
    require(arena.begin_read(1, false, &data, &h[1]), "read 1");
    require(!arena.begin_read(2, false, &data, &h[2]), "inflight cap");   // cap
    arena.finish_read(0, h[0], true);
    require(arena.begin_read(2, false, &data, &h[2]), "read 2 after a slot frees");
    arena.finish_read(1, h[1], true);
    arena.finish_read(2, h[2], true);
    for (int i = 3; i < 8; ++i) {
        const bool speculative = (i == 4);   // page 4 speculative: covers pin() promoting it below
        require(arena.begin_read(i, speculative, &data, &h[0]), "read i");
        arena.finish_read(i, h[0], true);
    }
    require(arena.spec_bytes() == ENTRY, "page 4 resident on the spec side before pin");
    require(arena.pin(0) && arena.pin(1) && arena.pin(2) && arena.pin(3), "pin 0-3");
    require(!arena.pin(4), "pinned cap");                                // pinned cap
    require(arena.pinned_bytes() == 4 * ENTRY, "pinned_bytes at cap");
    arena.unpin(3);
    require(arena.pin(4), "pin 4 after unpin 3");
    require(arena.spec_bytes() == 0, "pin promotes a speculative entry out of the spec side");
    require(arena.spec_promotions() == 1, "pin-promotion counted");
}

static void test_stale_handle_is_noop() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(2), a.alloc(), a.dealloc()), "init");
    void * data; HostArena::Handle h1, h2;
    require(arena.begin_read(1, false, &data, &h1), "read 1"); arena.finish_read(1, h1, true);
    const void * src; HostArena::Handle b;
    require(arena.borrow(1, &src, &b), "borrow 1");
    // evict 1 is impossible while borrowed; fill the other entry, then
    // release and force eviction of page 1, then release the stale handle
    require(arena.begin_read(2, false, &data, &h2), "read 2"); arena.finish_read(2, h2, true);
    arena.release(1, b);
    require(arena.begin_read(3, false, &data, &h1), "read 3 evicts 1 (LRU)");
    require(arena.state_of(1) == HostArena::State::Free, "1 evicted");
    arena.release(1, b);                               // stale: no crash, no change
    arena.finish_read(1, h1, true);                    // wrong page for handle: no-op
    require(arena.state_of(3) == HostArena::State::Reading, "3 still Reading");
    arena.finish_read(3, h1, true);
    require(arena.state_of(3) == HostArena::State::Resident, "3 now Resident");
}

// finish_read(..., ok=true, keep_borrowed=true) hands the caller a borrow
// atomically with the landing: the entry is Resident but, unlike an ordinary
// finish_read, unevictable until the caller's own release() drops it. This
// is the mechanism the expert worker uses to keep an arena entry alive for
// the lifetime of an in-flight async H2D copy without a second borrow() call
// racing the read landing.
static void test_finish_read_keep_borrowed() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(1), a.alloc(), a.dealloc()), "init 1 entry");
    void * data; HostArena::Handle h;
    require(arena.begin_read(1, false, &data, &h), "read 1");
    arena.finish_read(1, h, true, /*keep_borrowed=*/true);
    require(arena.state_of(1) == HostArena::State::Resident,
            "1 resident with an outstanding hold");
    // The only entry in a 1-entry arena is held: nothing is evictable, so a
    // second page cannot get a slot.
    void * data2; HostArena::Handle h2;
    require(!arena.begin_read(2, false, &data2, &h2),
            "held entry is unevictable");
    arena.release(1, h);
    require(arena.begin_read(2, false, &data2, &h2),
            "evictable once the hold's release() runs");
    require(arena.state_of(1) == HostArena::State::Free,
            "1 evicted to make room for 2");
}

// tier_bytes=0: an entry is freed the instant nothing holds it -- identical
// IO behaviour to the pre-arena StagingPool, which never retained a buffer
// past its lease.
static void test_tier_cap_zero_frees_on_release() {
    CountingAlloc a;
    HostArena::Config c = cfg(4);
    c.tier_bytes = 0;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");
    void * data; HostArena::Handle h;
    require(arena.begin_read(1, false, &data, &h), "read 1");
    arena.finish_read(1, h, true, /*keep_borrowed=*/true);
    require(arena.state_of(1) == HostArena::State::Resident, "1 resident while held");
    arena.release(1, h);
    require(arena.state_of(1) == HostArena::State::Free,
            "tier_bytes=0 frees the entry the moment its hold is released");
}

// tier_bytes < what is currently resident: once entries actually become
// evictable (unborrowed, unpinned), release() trims LRU-first down to the
// cap in the same locked call -- no separate GC pass needed.
static void test_tier_cap_trims_lru_on_release() {
    CountingAlloc a;
    HostArena::Config c = cfg(4);
    c.tier_bytes = 2 * ENTRY;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");
    void * data; HostArena::Handle h[3];
    for (int i = 0; i < 3; ++i) {
        require(arena.begin_read(i, false, &data, &h[i]), "read i");
        arena.finish_read(i, h[i], true, /*keep_borrowed=*/true);
    }
    require(arena.resident_count() == 3,
            "all three resident while every hold is still outstanding (cap cannot evict a held entry)");
    // Page 0 is the LRU entry (finished first, never re-touched). Releasing
    // it is what finally makes something evictable, and the trim takes the
    // LRU entry -- itself -- straight back down to the cap.
    arena.release(0, h[0]);
    require(arena.state_of(0) == HostArena::State::Free &&
                arena.resident_count() == 2,
            "release trimmed the LRU entry down to the 2-entry cap");
    // Releasing 1 and 2 does not push further: resident bytes are already at
    // the cap, so nothing more is evicted just because it becomes releasable.
    arena.release(1, h[1]);
    arena.release(2, h[2]);
    require(arena.resident_count() == 2,
            "at-cap resident set is left alone once nothing exceeds it");
}

// begin_read_wait(): unlike begin_read(), a refusal for capacity reasons
// (inflight cap here) blocks instead of failing, and wakes on the very next
// release()/finish_read() from ANY thread. This is what lets a single
// caller process more distinct pages than read_inflight_max allows at once
// -- reserve, read, release, repeat -- instead of needing every page's
// entry reserved simultaneously (the pre-fix deadlock: a batch bigger than
// the inflight cap could never get its later pages reserved at all).
static void test_begin_read_wait_unblocks_on_release() {
    CountingAlloc a;
    HostArena::Config c = cfg(1);
    c.read_inflight_max = 1;   // force real contention with only one thread
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init 1 entry, inflight=1");

    void * data0; HostArena::Handle h0;
    require(arena.begin_read(0, false, &data0, &h0), "read 0 takes the only inflight slot");

    std::atomic<bool> waiter_started{false};
    std::atomic<bool> waiter_done{false};
    bool waiter_result = false;
    std::thread waiter([&]() {
        waiter_started.store(true);
        void * data1; HostArena::Handle h1;
        waiter_result = arena.begin_read_wait(1, false, &data1, &h1, /*timeout_ms=*/2000);
        waiter_done.store(true);
    });
    while (!waiter_started.load()) {
        std::this_thread::yield();
    }
    // Give the waiter a real chance to reach the blocking wait before we
    // free the slot -- if it raced ahead and returned early (a bug), this
    // sleep does not create a false pass: waiter_done would already be true.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(!waiter_done.load(), "waiter must still be blocked: inflight cap is 1 and page 0 has not finished");

    arena.finish_read(0, h0, true);   // frees the inflight slot, notifies cv_
    waiter.join();
    require(waiter_result, "begin_read_wait must succeed once finish_read() frees the slot");
    require(waiter_done.load(), "waiter completed");
}

// End-to-end shape of the deadlock this fixes: a "batch" of 2x
// read_inflight_max distinct pages, processed by read_inflight_max
// concurrent "reader threads" pulling from a shared work queue -- each
// reserves, "reads" (no-op), lands, and releases before pulling the next.
// tier_bytes=0 means every entry is freed the instant it is released, so
// this also exercises the retention cap interacting with active contention.
static void test_batch_larger_than_inflight_cap_completes() {
    CountingAlloc a;
    HostArena::Config c = cfg(4);
    c.read_inflight_max = 4;
    c.tier_bytes = 0;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init 4 entries, inflight=4");

    static constexpr int kPages = 8;   // 2x read_inflight_max
    std::atomic<int> next_page{0};
    std::atomic<int> completed{0};
    const auto worker = [&]() {
        while (true) {
            const int page = next_page.fetch_add(1);
            if (page >= kPages) {
                return;
            }
            void * data; HostArena::Handle h;
            if (!arena.begin_read_wait(page, false, &data, &h, /*timeout_ms=*/5000)) {
                continue;   // let require() below catch the shortfall
            }
            arena.finish_read(page, h, true, /*keep_borrowed=*/true);
            arena.release(page, h);
            completed.fetch_add(1);
        }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) {
        workers.emplace_back(worker);
    }
    for (std::thread & w : workers) {
        w.join();
    }
    require(completed.load() == kPages,
            "a page count twice the inflight cap must still fully complete");
}


// C1: a demand caller that finds its page being READ by someone else (a
// speculative landing, another connection) must wait for that read to land
// and then BORROW it -- never time out, never read the page itself.
static void test_reserve_wait_present_after_concurrent_read() {
    CountingAlloc a;
    HostArena arena;
    require(arena.init(cfg(4), a.alloc(), a.dealloc()), "init");
    void * data_a; HostArena::Handle h_a;
    require(arena.begin_read(7, false, &data_a, &h_a), "A reserves 7");
    std::atomic<bool> b_done{false};
    HostArena::Reserve b_result = HostArena::Reserve::Timeout;
    bool b_borrowed = false;
    std::thread b([&]() {
        void * data_b = nullptr; HostArena::Handle h_b = HostArena::kInvalidHandle;
        b_result = arena.reserve_wait(7, false, &data_b, &h_b, 2000);
        if (b_result == HostArena::Reserve::Present) {
            const void * src = nullptr;
            b_borrowed = arena.borrow(7, &src, &h_b);
            if (b_borrowed) arena.release(7, h_b);
        }
        b_done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    require(!b_done.load(), "B must block while A is still Reading 7");
    arena.finish_read(7, h_a, true);
    b.join();
    require(b_result == HostArena::Reserve::Present, "B sees 7 Present once A lands it");
    require(b_borrowed, "B borrows the page A read");
    require(arena.resident_count() == 1, "7 was read exactly once");

    // Already Resident: Present immediately, no wait.
    void * d; HostArena::Handle h;
    require(arena.reserve_wait(7, false, &d, &h, 0) == HostArena::Reserve::Present,
            "Resident page is Present with a zero timeout");
    // Not present at all: Reserved, as begin_read.
    require(arena.reserve_wait(8, false, &d, &h, 0) == HostArena::Reserve::Reserved,
            "unknown page is Reserved");
    arena.finish_read(8, h, true);

    // A concurrent read that FAILS frees the entry and the waiter gets it
    // Reserved instead of Present.
    require(arena.begin_read(9, false, &data_a, &h_a), "A reserves 9");
    std::thread c([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        arena.finish_read(9, h_a, false);
    });
    const HostArena::Reserve r9 = arena.reserve_wait(9, false, &d, &h, 2000);
    c.join();
    require(r9 == HostArena::Reserve::Reserved, "failed concurrent read hands the waiter the reservation");
    arena.finish_read(9, h, true);
}

// I3: the pinned cap counts TIER entries only; in-flight entries are never
// pinnable, so a pin set cannot take the entries readers need.
static void test_pinned_cap_excludes_inflight_entries() {
    CountingAlloc a;
    HostArena::Config c = cfg(6);        // budget = 6 entries
    c.tier_bytes        = 2 * ENTRY;     // 2 tier + 4 in-flight
    c.read_inflight_max = 4;
    c.pinned_cap_pct    = 100;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");
    void * d; HostArena::Handle h[3];
    for (int i = 0; i < 3; ++i) {
        require(arena.begin_read(i, false, &d, &h[i]), "read i");
        arena.finish_read(i, h[i], true, /*keep_borrowed=*/true);   // held: no trim
    }
    require(arena.pin(0) && arena.pin(1), "two tier entries pinnable");
    require(!arena.pin(2), "third pin refused: only tier_bytes/entry_bytes entries are pinnable");
    require(arena.pinned_bytes() == 2 * ENTRY, "pinned bytes at the tier cap");
    for (int i = 0; i < 3; ++i) arena.release(i, h[i]);
}

// --- freq_admission (WP_HOST_TIER_POLICY=freq_admit) -----------------------

static void read_page(HostArena & arena, int page, bool prefill = true) {
    void * data; HostArena::Handle h;
    require(arena.begin_read(page, false, &data, &h), "begin_read (miss)");
    arena.finish_read(page, h, true, /*keep_borrowed=*/false, prefill);
}

// true (and releases) iff the page was resident -- a "hit" in the cache
// sense, mirroring reserve_arena_for_pagein's own borrow()-first sequence.
static bool hit_page(HostArena & arena, int page) {
    const void * src; HostArena::Handle h;
    if (!arena.borrow(page, &src, &h)) return false;
    arena.release(page, h);
    return true;
}

// The motivating bug: prefill sweeps every expert of every layer in the same
// cyclic order, a cycle far bigger than the tier. Plain LRU (cfg()'s default,
// freq_admission left false) evicts every page before its next reference:
// hit rate 0 forever, exactly the ram_hit_rate=0 the worker was measuring.
static void test_plain_lru_gets_zero_hits_on_oversized_cyclic_sweep() {
    CountingAlloc a;
    HostArena::Config c = cfg(4);
    c.tier_bytes = 2 * ENTRY;               // tier holds 2, sweep touches 4
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");
    int hits = 0;
    for (int pass = 0; pass < 4; ++pass) {
        for (int page = 0; page < 4; ++page) {
            if (hit_page(arena, page)) ++hits; else read_page(arena, page);
        }
    }
    require(hits == 0, "plain LRU: a 4-page cyclic sweep through a 2-page tier never hits");
}

// freq_admission=true on the SAME trace: it must settle on a STABLE 2-page
// subset instead of thrashing, and hit rate must converge to tier_size /
// sweep_size (2/4 = 50%) from the second pass on -- see the block comment
// above HostArena::admit_landed_locked_ for why tie-favors-incumbent with an
// unrecorded landing is what produces exactly this convergence.
static void test_freq_admission_locks_a_stable_subset_on_cyclic_sweep() {
    CountingAlloc a;
    HostArena::Config c = cfg(4);
    c.tier_bytes      = 2 * ENTRY;
    c.freq_admission  = true;
    c.sketch_width    = 64;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    // Pass 1: cold cache, nothing to compare against yet (every landing is
    // either under the cap or ties 0-vs-0 and self-evicts) -- 0 hits, same
    // as plain LRU has to be on a cold start.
    int hits_pass1 = 0;
    for (int page = 0; page < 4; ++page) {
        if (hit_page(arena, page)) ++hits_pass1; else read_page(arena, page);
    }
    require(hits_pass1 == 0, "pass 1 is necessarily cold");
    require(arena.resident_count() == 2, "tier holds exactly 2 after pass 1");

    bool resident_after_pass1[4];
    for (int page = 0; page < 4; ++page) resident_after_pass1[page] = arena.is_resident(page);

    // Pass 2 and 3: the resident pair from pass 1 must now be HIT (they
    // never left), and the other pair must keep losing admission and
    // self-evict without disturbing the resident pair.
    for (int pass = 2; pass <= 3; ++pass) {
        int hits = 0;
        for (int page = 0; page < 4; ++page) {
            if (hit_page(arena, page)) ++hits; else read_page(arena, page);
        }
        require(hits == 2, "hit rate settles at tier_size/sweep_size (2/4) once locked in");
        for (int page = 0; page < 4; ++page) {
            require(arena.is_resident(page) == resident_after_pass1[page],
                    "the resident subset is STABLE across passes, not re-shuffled");
        }
    }
    require(arena.admission_cold_landed() > 0,
            "the losing pair actually went through the cold-landing/self-evict path");
}

// Decode-shaped scan resistance: a single hot page is borrow()'d repeatedly
// (building real sketch frequency) while a long stream of one-shot cold
// pages -- each never referenced again, exactly like the low-popularity
// tail of a Zipf routing distribution -- tries to take its slot. None of
// them may succeed: every one of them starts at estimate 0, which can never
// beat the hot page's real (>0) estimate under the <= tie-break.
static void test_freq_admission_hot_page_resists_a_cold_scan() {
    CountingAlloc a;
    HostArena::Config c = cfg(8);
    c.tier_bytes      = 1 * ENTRY;          // one slot: everything contends for it
    c.freq_admission  = true;
    c.sketch_width    = 64;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    read_page(arena, /*hot=*/0);
    for (int i = 0; i < 3; ++i) {
        require(hit_page(arena, 0), "hot page re-referenced (builds real sketch frequency)");
    }
    for (int cold = 100; cold < 164; ++cold) {
        read_page(arena, cold);   // miss: never resident before, admission-gated
        require(arena.state_of(0) == HostArena::State::Resident,
                "a first-time cold page must never evict the proven-hot incumbent");
        require(arena.state_of(cold) == HostArena::State::Free,
                "the losing cold page self-evicts on this same call instead of squatting");
    }
    require(arena.admission_cold_landed() >= 64, "every cold-scan page went through the loss path");
}

// freq_admission=false (the default, unset in cfg()) must be exactly the
// pre-existing behaviour: admission_cold_landed() stays 0 even under the
// same pressure that exercises it above.
static void test_freq_admission_off_never_touches_admission_counters() {
    CountingAlloc a;
    HostArena::Config c = cfg(8);
    c.tier_bytes = 1 * ENTRY;
    HostArena arena;   // freq_admission left at its default: false
    require(arena.init(c, a.alloc(), a.dealloc()), "init");
    read_page(arena, 0);
    for (int i = 0; i < 3; ++i) require(hit_page(arena, 0), "hit");
    for (int cold = 100; cold < 110; ++cold) read_page(arena, cold);
    require(arena.admission_cold_landed() == 0,
            "policy off: the counter never increments, byte-for-byte legacy behaviour");
    require(arena.state_of(0) == HostArena::State::Free,
            "policy off: plain LRU still evicts the old page like every pre-existing test expects");
}

// prefill_hint=false (decode) must NEVER be gated, even under exactly the
// scan pressure that gates prefill above -- this is what keeps
// freq_admit's phase split from regressing decode below plain LRU (see the
// docs/dev/sim-host-tier.py freq_admit vs freq_admit_phase measurements the
// block comment above HostArena::admit_landed_locked_ cites).
static void test_freq_admission_decode_hint_never_gates() {
    CountingAlloc a;
    HostArena::Config c = cfg(8);
    c.tier_bytes      = 1 * ENTRY;
    c.freq_admission  = true;
    c.sketch_width    = 64;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    read_page(arena, 0, /*prefill=*/false);
    for (int i = 0; i < 3; ++i) require(hit_page(arena, 0), "hit");
    // A stream of decode (prefill=false) one-shot pages: plain LRU DOES
    // evict the incumbent here (unlike the prefill-hint scan-resistance
    // test above) because decode landings are never gated.
    bool ever_evicted = false;
    for (int cold = 100; cold < 110; ++cold) {
        read_page(arena, cold, /*prefill=*/false);
        if (arena.state_of(0) != HostArena::State::Resident) ever_evicted = true;
    }
    require(ever_evicted, "decode-hinted landings are ungated: they behave like plain LRU, "
                          "not like the prefill scan-resistance path");
    require(arena.admission_cold_landed() == 0,
            "the gate never even evaluates a decode-hinted landing");
}

// Reproduces the live 2026-09-25 finding: WP_HOST_TIER_POLICY=freq_admit
// measured n_host_hit=0 in production even with the tier sized to hold the
// whole working set. admit_landed_locked_ (tested above) only gates WHERE a
// DEMAND page lands; it says nothing about a SPECULATIVE begin_read's own
// eviction choice inside begin_read_locked_. Before the fix, when spec was
// under its cap but the arena had no free entries AND spec_lru_ happened to
// be empty at that instant, begin_read_locked_'s speculative branch fell
// back to EvictScope::Any, which evicts from lru_ (the DEMAND list) same as
// plain LRU -- completely bypassing the frequency gate, since that gate is
// never even consulted on this path. Under continuous production prefetch
// (WP_PREFILL_LAYER_AHEAD=1 + WP_EXPERT_SPEC_PAGEIN=1, not exercised by any
// test in this file before this one) spec_lru_ churns constantly and
// routinely empties for a moment, so an unconfirmed guess kept evicting
// confirmed demand pages -- matching the live symptom of near-0 hits and
// ram_evictions far exceeding the distinct page count.
static void test_freq_admission_speculative_never_evicts_demand() {
    CountingAlloc a;
    HostArena::Config c = cfg(3);       // exactly demand(1) + spec(2): no free slack
    c.tier_bytes     = 3 * ENTRY;       // no retention pressure: isolates begin_read's own bug
    c.spec_frac_pct  = 100;             // spec cap == the whole arena: never the limiting factor
    c.freq_admission = true;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    // 1. One demand (prefill) page lands and is never touched again.
    read_page(arena, /*page=*/1, /*prefill=*/true);

    // 2. Two speculative reads fill the rest of the arena.
    void * data; HostArena::Handle h2, h3;
    require(arena.begin_read(2, /*speculative=*/true, &data, &h2), "spec read 2");
    arena.finish_read(2, h2, true);
    require(arena.begin_read(3, /*speculative=*/true, &data, &h3), "spec read 3");
    arena.finish_read(3, h3, true);
    require(arena.entry_count() == 3 && arena.resident_count() == 3, "arena is completely full");

    // 3. Demand hits on 2 and 3 promote them out of spec_lru_ into the
    //    demand list, emptying spec_lru_ while the arena is still full.
    const void * src; HostArena::Handle b2, b3;
    require(arena.borrow(2, &src, &b2), "promote 2 to demand");
    arena.release(2, b2);
    require(arena.borrow(3, &src, &b3), "promote 3 to demand");
    arena.release(3, b3);
    require(arena.spec_bytes() == 0, "spec_lru_ is now empty -- everything is confirmed demand");

    // 4. A NEW speculative guess: free_ is empty, spec_lru_ has nothing to
    //    give up, and page 1 (confirmed demand, untouched since landing)
    //    sits at the front of lru_ looking like the cheapest LRU victim.
    //    It must refuse rather than evict page 1.
    void * data4; HostArena::Handle h4;
    const uint64_t refusals_before = arena.begin_read_refusals();
    require(!arena.begin_read(4, /*speculative=*/true, &data4, &h4),
            "a speculative guess must not be able to evict a confirmed demand page");
    require(arena.begin_read_refusals() == refusals_before + 1,
            "refused and counted, not silently starved");
    require(arena.state_of(1) == HostArena::State::Resident,
            "page 1 (demand, never re-touched) must have survived");
}

// Same exact scenario with freq_admission left at its default (false): the
// fix must be opt-in only. This documents (does not newly introduce) that a
// speculative guess CAN evict a confirmed demand page under plain LRU --
// proving the fix above is additive behind the knob, not a default change.
static void test_freq_admission_off_speculative_can_still_evict_demand() {
    CountingAlloc a;
    HostArena::Config c = cfg(3);
    c.tier_bytes     = 3 * ENTRY;
    c.spec_frac_pct  = 100;
    // freq_admission left false (default).
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    read_page(arena, /*page=*/1);
    void * data; HostArena::Handle h2, h3;
    require(arena.begin_read(2, true, &data, &h2), "spec read 2");
    arena.finish_read(2, h2, true);
    require(arena.begin_read(3, true, &data, &h3), "spec read 3");
    arena.finish_read(3, h3, true);
    const void * src; HostArena::Handle b2, b3;
    require(arena.borrow(2, &src, &b2), "promote 2");
    arena.release(2, b2);
    require(arena.borrow(3, &src, &b3), "promote 3");
    arena.release(3, b3);

    void * data4; HostArena::Handle h4;
    require(arena.begin_read(4, true, &data4, &h4),
            "plain LRU (policy off): the speculative guess DOES get admitted");
    require(arena.state_of(1) == HostArena::State::Free,
            "plain LRU (policy off): page 1 gets evicted to seat it -- the pre-existing behaviour, unchanged");
}

// Larger-scale version of test_freq_admission_locks_a_stable_subset_on_
// cyclic_sweep, shaped like the live production symptom this whole change
// exists to fix: a sweep much bigger than the tier, run for MANY passes (not
// just 2-3), with a realistic margin (budget = tier + margin, same as the
// worker's own cfg.budget_bytes = host_tier_bytes + entry_bytes *
// read_inflight_max) so reservation-time eviction has real slack to work
// with instead of the tiny 4-page test's edge-of-capacity behaviour.
//
// This specifically guards against a regression class the tiny test above
// cannot catch: admit_landed_locked_'s gate comparing against the wrong
// victim (or not firing at all) once reject_lru_ has been populated by an
// earlier landing -- on a small trace, trim_to_tier_cap_locked_ tends to
// evict a freshly-cold-landed page in the SAME call that landed it (reject_
// lru_ empty before, so the new entry is its own front), so reject_lru_
// rarely stays populated across separate page-in events and a gate bug tied
// to "reject_lru_ non-empty" would not show up. At this larger scale it can
// (and, during development of this change, briefly did in the standalone
// Python simulator, scripts/sim-host-tier.py -- FREQ_ADMIT_PHASE_REJECT on
// the pure-prefill trace measured an exact 0% hit rate with a first, buggy
// version of this gate condition, instead of converging to tier_size /
// sweep_size like every other prefill trial): if the gate stops firing once
// reject_lru_ has anything in it, every later prefill sweep page gets
// admitted hot unconditionally, evicting main's real residents on a rolling
// basis, and the second-pass-onward hit rate degrades toward 0 as the
// window slides instead of stabilising.
static void test_freq_admission_large_sweep_stays_stable_many_passes() {
    CountingAlloc a;
    const size_t tier_pages   = 20;
    const size_t margin_pages = 20;   // production shape: budget = tier + margin
    const size_t sweep_pages  = 100;
    HostArena::Config c = cfg(tier_pages + margin_pages);
    c.tier_bytes      = tier_pages * ENTRY;
    c.freq_admission  = true;
    c.sketch_width    = 256;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    const int n_passes = 20;
    double last_hit_rate = -1.0;
    for (int pass = 0; pass < n_passes; ++pass) {
        int hits = 0;
        for (size_t page = 0; page < sweep_pages; ++page) {
            if (hit_page(arena, (int) page)) ++hits; else read_page(arena, (int) page);
        }
        const double hit_rate = (double) hits / (double) sweep_pages;
        if (pass >= 2) {
            // Converged by pass 2 (pass 0 is cold, pass 1 may still be
            // settling); from here the resident subset must be STABLE, not
            // degrade pass over pass -- a wide tolerance (not an exact
            // tier/sweep check, which is a stronger property already
            // covered by the smaller test above) because this test's job is
            // catching regression to near-0, not verifying the exact number.
            require(hit_rate > 0.10,
                    "hit rate degraded well below tier_pages/sweep_pages on a "
                    "large many-pass sweep -- the admission gate likely "
                    "stopped firing once reject_lru_ was populated");
            if (last_hit_rate >= 0.0) {
                require(hit_rate >= last_hit_rate - 0.02,
                        "hit rate is still trending down pass over pass instead "
                        "of having stabilised -- the resident set is not locking in");
            }
        }
        last_hit_rate = hit_rate;
    }
    // The metric this whole change exists to move: once stable, evictions
    // that actually displaced an admitted (lru_) resident should be a small
    // fraction of total evictions -- most churn should have landed on
    // reject_lru_ instead.
    require(arena.evictions_lru() < arena.evictions() / 4,
            "too many evictions are landing on real (lru_) residents instead "
            "of being absorbed by reject_lru_");
    require(arena.admission_cold_landed() > 0, "the gate never actually rejected anything");
}

// Reproduces the 2026-09-25 LIVE finding (three back-to-back 24.5k-token
// prefills, freq_admit engaged, main: ram_lookups=61641 ram_lookup_hits=34
// ram_evictions_lru=59483 ram_admission_cold_landed=754
// ram_reject_promotions=0): live traffic is ~96% layer-ahead SPECULATIVE
// reads (WP_PREFILL_LAYER_AHEAD), which land Resident-but-speculative via
// begin_read/finish_read and only ever become genuine demand (lru_)
// residents through borrow()'s spec->demand promotion on their first real
// reference -- a transition admit_landed_locked_ (finish_read's landing
// gate) never sees, because the page was already Resident by the time any
// demand access happens. Before this test's fix, that promotion was
// unconditional: every promoted page landed hot in lru_ regardless of
// frequency, so ram_evictions_lru tracked ram_lookups almost 1:1 and
// ram_reject_promotions stayed 0 even with admission_cold_landed proving the
// OTHER gate (fresh demand misses) was, in isolation, working -- exactly
// the live numbers above (cold_landed=754 nonzero, reject_promotions=0).
//
// This test builds a real (frequency-bearing) demand incumbent, fills the
// tier to its cap with it plus a speculative page, then demand-borrows the
// speculative page with prefill_hint=true: the promotion must be gated the
// same way a fresh landing is, and a losing promotion must land in
// reject_lru_ without disturbing the incumbent.
static void test_freq_admission_gates_spec_to_demand_promotion() {
    CountingAlloc a;
    HostArena::Config c = cfg(4);      // 2 tier + 2 margin
    c.tier_bytes      = 2 * ENTRY;
    c.freq_admission  = true;
    c.spec_frac_pct   = 100;
    c.sketch_width    = 64;
    HostArena arena;
    require(arena.init(c, a.alloc(), a.dealloc()), "init");

    // page 1: real demand incumbent with genuine (nonzero, repeat-earned)
    // frequency -- the same shape as a decode expert's own resident set that
    // a prefill sweep must not casually displace.
    read_page(arena, /*page=*/1, /*prefill=*/true);
    require(hit_page(arena, 1), "build page 1's real frequency (hit 1/2)");
    require(hit_page(arena, 1), "build page 1's real frequency (hit 2/2)");

    // page 2: a layer-ahead speculative landing -- Resident, speculative,
    // never yet demand-referenced. Together with page 1 this exactly fills
    // the 2-page tier (tier_full via >=, not >: see the tier_full comment in
    // HostArena::borrow()), with no trim needed (not OVER cap), so the
    // scenario is stable across separate calls instead of self-correcting.
    void * data; HostArena::Handle h2;
    require(arena.begin_read(2, /*speculative=*/true, &data, &h2), "spec read 2");
    arena.finish_read(2, h2, /*ok=*/true);
    require(arena.resident_count() == 2, "tier exactly full: page 1 (demand) + page 2 (spec)");
    require(arena.spec_promotions() == 0 && arena.spec_promotions_rejected() == 0,
            "nothing promoted yet");

    // The promotion under test: a prefill-hinted demand reference to page 2.
    // Its own estimate (freshly recorded by this very call, per "record on
    // every access") cannot beat page 1's real, multiply-earned estimate --
    // it must lose and land in reject_lru_, not lru_, and page 1 must be
    // completely undisturbed.
    const void * src; HostArena::Handle b2;
    require(arena.borrow(2, &src, &b2, /*demand=*/true, /*prefill_hint=*/true),
            "demand borrow of the speculative page");
    arena.release(2, b2);

    require(arena.spec_promotions() == 1, "page 2 was confirmed out of spec_lru_");
    require(arena.spec_promotions_rejected() == 1,
            "the promotion lost the admission comparison -- THE counter a live run "
            "measuring 0 here (despite admission_cold_landed > 0 elsewhere) is the "
            "signature of this exact bug");
    require(arena.state_of(1) == HostArena::State::Resident,
            "page 1 (the real incumbent) must survive an ungated-would-have-evicted promotion");
    require(arena.is_resident(2), "page 2 is still resident (in reject_lru_), just not admitted");

    // A demand hit on page 2 now (it is resident, just in reject_lru_) must
    // still promote it the rest of the way on real re-reference -- losing
    // once is not permanent exile, same guarantee reject_lru_ gives a
    // rejected fresh landing.
    const void * src2; HostArena::Handle b2b;
    require(arena.borrow(2, &src2, &b2b), "page 2 is still borrowable after losing admission");
    arena.release(2, b2b);
    require(arena.reject_promotions() == 1, "a real re-reference promotes it out of reject_lru_");

    // Sanity: the SAME scenario with prefill_hint=false (a decode access)
    // must never gate -- decode promotions stay unconditional, same phase
    // split as admit_landed_locked_'s prefill_hint.
    void * data3; HostArena::Handle h3;
    require(arena.begin_read(3, /*speculative=*/true, &data3, &h3), "spec read 3");
    arena.finish_read(3, h3, /*ok=*/true, /*keep_borrowed=*/true);   // hold it so trim can't evict it below
    arena.release(3, h3);
    const void * src3; HostArena::Handle b3;
    require(arena.borrow(3, &src3, &b3, /*demand=*/true, /*prefill_hint=*/false),
            "decode-hinted promotion");
    arena.release(3, b3);
    require(arena.spec_promotions_rejected() == 1,
            "a decode-hinted (prefill_hint=false) promotion must never be gated -- "
            "the rejected count must not have grown");
}

int main() {
    try {
        test_init_chunked_and_shrink_on_failure();
        test_read_state_machine();
        test_lru_never_evicts_reading_borrowed_or_pinned();
        test_speculative_drained_first_and_promoted_on_demand_hit();
        test_inflight_cap_and_pinned_cap();
        test_stale_handle_is_noop();
        test_finish_read_keep_borrowed();
        test_tier_cap_zero_frees_on_release();
        test_tier_cap_trims_lru_on_release();
        test_begin_read_wait_unblocks_on_release();
        test_batch_larger_than_inflight_cap_completes();
        test_reserve_wait_present_after_concurrent_read();
        test_pinned_cap_excludes_inflight_entries();
        test_plain_lru_gets_zero_hits_on_oversized_cyclic_sweep();
        test_freq_admission_locks_a_stable_subset_on_cyclic_sweep();
        test_freq_admission_large_sweep_stays_stable_many_passes();
        test_freq_admission_gates_spec_to_demand_promotion();
        test_freq_admission_hot_page_resists_a_cold_scan();
        test_freq_admission_decode_hint_never_gates();
        test_freq_admission_speculative_never_evicts_demand();
        test_freq_admission_off_speculative_can_still_evict_demand();
        test_freq_admission_off_never_touches_admission_counters();
        std::cout << "test-wp-host-arena: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-host-arena: " << error.what() << '\n';
        return 1;
    }
}
