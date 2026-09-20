#include "../src/weight-pager/wp-host-arena.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
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

int main() {
    try {
        test_init_chunked_and_shrink_on_failure();
        test_read_state_machine();
        test_lru_never_evicts_reading_borrowed_or_pinned();
        test_speculative_drained_first_and_promoted_on_demand_hit();
        test_inflight_cap_and_pinned_cap();
        test_stale_handle_is_noop();
        std::cout << "test-wp-host-arena: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-host-arena: " << error.what() << '\n';
        return 1;
    }
}
