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
        std::cout << "test-wp-host-arena: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-host-arena: " << error.what() << '\n';
        return 1;
    }
}
