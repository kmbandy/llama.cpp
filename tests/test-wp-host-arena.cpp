#include "../src/weight-pager/wp-host-arena.h"

// build-hip compiles tests as RelWithDebInfo, which defines NDEBUG -- under
// NDEBUG the standard assert() macro discards its argument's evaluation
// entirely (not just the check), so every meaningful call in this file would
// silently never run. Force assertions on for this translation unit only;
// <cassert> re-reads NDEBUG on every inclusion (unlike normal headers, it is
// explicitly required to have no include-guard-style behavior), so this is
// sufficient regardless of what any earlier header already pulled in.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using wp::HostArena;

static const size_t ENTRY = 4096 * 4;   // 16 KiB fake page

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
    assert(arena.init(cfg(8), a.alloc(), a.dealloc()));
    assert(arena.entry_count() == 8);
    assert(arena.chunk_count() == 2);
    arena.shutdown();
    assert(a.freed == 2);

    CountingAlloc b;
    b.fail_after = 1;                         // second chunk fails
    HostArena small;
    assert(small.init(cfg(8), b.alloc(), b.dealloc()));
    assert(small.entry_count() == 4);         // shrank, did not fall back
    assert(small.chunk_count() == 1);

    CountingAlloc c;
    c.fail_after = 0;
    HostArena none;
    assert(!none.init(cfg(8), c.alloc(), c.dealloc()));
    assert(!none.is_initialized());
}

static void test_read_state_machine() {
    CountingAlloc a;
    HostArena arena;
    assert(arena.init(cfg(4), a.alloc(), a.dealloc()));
    void * data = nullptr; HostArena::Handle h = 0;
    assert(arena.begin_read(7, false, &data, &h));
    assert(data != nullptr && h != HostArena::kInvalidHandle);
    assert(arena.state_of(7) == HostArena::State::Reading);
    assert(arena.reading_count() == 1);
    // cannot borrow while Reading, cannot begin a second read of the same page
    const void * src = nullptr; HostArena::Handle bh = 0;
    assert(!arena.borrow(7, &src, &bh));
    void * d2 = nullptr; HostArena::Handle h2 = 0;
    assert(!arena.begin_read(7, false, &d2, &h2));
    std::memset(data, 0xAB, ENTRY);
    arena.finish_read(7, h, true);
    assert(arena.state_of(7) == HostArena::State::Resident);
    assert(arena.resident_bytes() == ENTRY);
    assert(arena.borrow(7, &src, &bh));
    assert(((const unsigned char *) src)[0] == 0xAB);
    arena.release(7, bh);
    // failed read frees the entry
    assert(arena.begin_read(8, false, &data, &h));
    arena.finish_read(8, h, false);
    assert(arena.state_of(8) == HostArena::State::Free);
    assert(arena.resident_count() == 1);
}

static void test_lru_never_evicts_reading_borrowed_or_pinned() {
    CountingAlloc a;
    HostArena arena;
    assert(arena.init(cfg(4), a.alloc(), a.dealloc()));
    void * data; HostArena::Handle h[4];
    for (int i = 0; i < 4; ++i) {
        assert(arena.begin_read(i, false, &data, &h[i]));
        arena.finish_read(i, h[i], true);
    }
    // page 0: borrowed; page 1: pinned; page 2: re-read in flight; page 3: plain
    const void * src; HostArena::Handle b0;
    assert(arena.borrow(0, &src, &b0));
    assert(arena.pin(1));
    void * d2; HostArena::Handle h2;
    assert(!arena.begin_read(2, false, &d2, &h2));   // already Resident
    // arena full: page 4 must evict page 3 (the only evictable)
    assert(arena.begin_read(4, false, &data, &h[0]));
    assert(arena.state_of(3) == HostArena::State::Free);
    assert(arena.evictions() == 1);
    arena.finish_read(4, h[0], true);
    // now nothing evictable -> refusal
    assert(!arena.begin_read(5, false, &data, &h[0]));
    assert(arena.begin_read_refusals() == 1);
    arena.release(0, b0);
    assert(arena.begin_read(5, false, &data, &h[0]));  // page 0 evicted (LRU, unborrowed)
    assert(arena.state_of(0) == HostArena::State::Free);
}

static void test_speculative_drained_first_and_promoted_on_demand_hit() {
    CountingAlloc a;
    HostArena arena;
    assert(arena.init(cfg(8), a.alloc(), a.dealloc()));   // spec cap = 2 entries
    void * data; HostArena::Handle h;
    // 6 demand pages, MRU order 0..5
    for (int i = 0; i < 6; ++i) { assert(arena.begin_read(i, false, &data, &h)); arena.finish_read(i, h, true); }
    // 2 speculative pages fill the spec cap
    assert(arena.begin_read(10, true, &data, &h)); arena.finish_read(10, h, true);
    assert(arena.begin_read(11, true, &data, &h)); arena.finish_read(11, h, true);
    assert(arena.spec_bytes() == 2 * ENTRY);
    // a third speculative read must evict a speculative entry, not page 0
    assert(arena.begin_read(12, true, &data, &h)); arena.finish_read(12, h, true);
    assert(arena.state_of(10) == HostArena::State::Free);
    assert(arena.state_of(0)  == HostArena::State::Resident);
    assert(arena.spec_evicted_unused() == 1);
    // demand hit promotes 11 out of the speculative side
    const void * src; HostArena::Handle b;
    assert(arena.borrow(11, &src, &b));
    arena.release(11, b);
    assert(arena.spec_bytes() == ENTRY);
    assert(arena.spec_promotions() == 1);
    // arena full with 6 demand + 2 spec: a DEMAND read evicts the remaining
    // speculative entry (12) before the oldest demand page (0)
    assert(arena.begin_read(13, false, &data, &h));
    assert(arena.state_of(12) == HostArena::State::Free);
    assert(arena.state_of(0)  == HostArena::State::Resident);
}

static void test_inflight_cap_and_pinned_cap() {
    CountingAlloc a;
    HostArena::Config c = cfg(8);
    c.read_inflight_max = 2;
    c.pinned_cap_pct    = 50;   // 4 of 8
    HostArena arena;
    assert(arena.init(c, a.alloc(), a.dealloc()));
    void * data; HostArena::Handle h[3];
    assert(arena.begin_read(0, false, &data, &h[0]));
    assert(arena.begin_read(1, false, &data, &h[1]));
    assert(!arena.begin_read(2, false, &data, &h[2]));   // cap
    arena.finish_read(0, h[0], true);
    assert(arena.begin_read(2, false, &data, &h[2]));
    arena.finish_read(1, h[1], true);
    arena.finish_read(2, h[2], true);
    for (int i = 3; i < 8; ++i) { assert(arena.begin_read(i, false, &data, &h[0])); arena.finish_read(i, h[0], true); }
    assert(arena.pin(0) && arena.pin(1) && arena.pin(2) && arena.pin(3));
    assert(!arena.pin(4));                                // pinned cap
    assert(arena.pinned_bytes() == 4 * ENTRY);
    arena.unpin(3);
    assert(arena.pin(4));
}

static void test_stale_handle_is_noop() {
    CountingAlloc a;
    HostArena arena;
    assert(arena.init(cfg(2), a.alloc(), a.dealloc()));
    void * data; HostArena::Handle h1, h2;
    assert(arena.begin_read(1, false, &data, &h1)); arena.finish_read(1, h1, true);
    const void * src; HostArena::Handle b;
    assert(arena.borrow(1, &src, &b));
    // evict 1 is impossible while borrowed; fill the other entry, then
    // release and force eviction of page 1, then release the stale handle
    assert(arena.begin_read(2, false, &data, &h2)); arena.finish_read(2, h2, true);
    arena.release(1, b);
    assert(arena.begin_read(3, false, &data, &h1));   // evicts 1 (LRU)
    assert(arena.state_of(1) == HostArena::State::Free);
    arena.release(1, b);                               // stale: no crash, no change
    arena.finish_read(1, h1, true);                    // wrong page for handle: no-op
    assert(arena.state_of(3) == HostArena::State::Reading);
    arena.finish_read(3, h1, true);
    assert(arena.state_of(3) == HostArena::State::Resident);
}

int main() {
    test_init_chunked_and_shrink_on_failure();
    test_read_state_machine();
    test_lru_never_evicts_reading_borrowed_or_pinned();
    test_speculative_drained_first_and_promoted_on_demand_hit();
    test_inflight_cap_and_pinned_cap();
    test_stale_handle_is_noop();
    std::printf("test-wp-host-arena: all tests passed\n");
    return 0;
}
