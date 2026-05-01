// Unit tests for the wp::* modules under src/weight-pager/.
//
// Lightweight, no test framework. Each TEST_FN runs subtests and returns
// the number of failures. main() sums them and exits non-zero if any
// failed. Tests that require GPU (hip*) are gated on GGML_USE_HIP at
// runtime — they no-op compile-out under non-HIP builds.

#include "weight-pager/wp-page-catalog.h"
#include "weight-pager/wp-file-io.h"
#include "weight-pager/wp-pool.h"

#include "ggml-backend.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, (msg)); \
        ++fails;                                                                    \
    } \
} while (0)

#define EXPECT_EQ_INT(actual, expected, msg) do { \
    if ((actual) != (expected)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d): %s — got %lld, expected %lld\n", \
                     __func__, __LINE__, (msg), (long long)(actual), (long long)(expected)); \
        ++fails; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// PageCatalog
// ---------------------------------------------------------------------------

static int test_page_catalog() {
    int fails = 0;
    wp::PageCatalog cat;

    EXPECT_EQ_INT(cat.size(), 0, "empty catalog size");
    EXPECT_EQ_INT(cat.find("nope"), -1, "find on empty");
    EXPECT_EQ_INT(cat.max_page_size(), 0u, "empty max_page_size");

    int i0 = cat.add("blk.0.attn_q.weight",  0, 1024, 4096);
    int i1 = cat.add("blk.0.attn_k.weight",  0, 5120, 8192);
    int i2 = cat.add("blk.1.ffn_down.weight", 1, 16384, 65536);

    EXPECT_EQ_INT(i0, 0, "first page index");
    EXPECT_EQ_INT(i1, 1, "second page index");
    EXPECT_EQ_INT(i2, 2, "third page index");
    EXPECT_EQ_INT(cat.size(), 3, "post-insert size");
    EXPECT_EQ_INT(cat.max_page_size(), 65536u, "max_page_size tracks largest");

    EXPECT_EQ_INT(cat.find("blk.0.attn_q.weight"), 0, "lookup first");
    EXPECT_EQ_INT(cat.find("blk.0.attn_k.weight"), 1, "lookup second");
    EXPECT_EQ_INT(cat.find("blk.1.ffn_down.weight"), 2, "lookup third");
    EXPECT_EQ_INT(cat.find("missing"), -1, "lookup missing");

    const wp::PageMeta & m = cat.at(2);
    EXPECT(m.tensor_name == "blk.1.ffn_down.weight", "metadata: name");
    EXPECT_EQ_INT(m.file_idx, 1, "metadata: file_idx");
    EXPECT_EQ_INT(m.file_offset, 16384u, "metadata: offset");
    EXPECT_EQ_INT(m.size, 65536u, "metadata: size");

    cat.clear();
    EXPECT_EQ_INT(cat.size(), 0, "post-clear size");
    EXPECT_EQ_INT(cat.find("blk.0.attn_q.weight"), -1, "post-clear lookup");
    EXPECT_EQ_INT(cat.max_page_size(), 0u, "post-clear max_page_size");

    return fails;
}

// ---------------------------------------------------------------------------
// dup_clear_o_direct
// ---------------------------------------------------------------------------

static int test_dup_clear_o_direct() {
    int fails = 0;

    // Create a tmp file. We can't reliably set O_DIRECT on it (filesystem
    // dependent), but we CAN at least verify the helper returns a usable
    // dup'd fd and doesn't error.
    char path[] = "/tmp/wp-test-fd-XXXXXX";
    int fd = mkstemp(path);
    EXPECT(fd >= 0, "mkstemp succeeded");
    if (fd < 0) return fails;

    int dup_fd = wp::dup_clear_o_direct(fd);
    EXPECT(dup_fd >= 0, "dup_clear_o_direct returned a valid fd");
    EXPECT(dup_fd != fd, "dup'd fd is distinct from source");

    // The dup'd fd must NOT have O_DIRECT set, regardless of source.
    int fl = fcntl(dup_fd, F_GETFL);
    EXPECT(fl != -1, "fcntl F_GETFL on dup'd fd");
#ifdef O_DIRECT
    EXPECT((fl & O_DIRECT) == 0, "O_DIRECT cleared on dup'd fd");
#endif

    // Write something via the dup'd fd, read via the original — proves
    // both fds point at the same file.
    const char * msg = "hi\n";
    ssize_t w = write(dup_fd, msg, 3);
    EXPECT_EQ_INT(w, 3, "write to dup'd fd");

    char buf[4] = {};
    lseek(fd, 0, SEEK_SET);
    ssize_t r = read(fd, buf, 3);
    EXPECT_EQ_INT(r, 3, "read from original fd");
    EXPECT(std::strncmp(buf, msg, 3) == 0, "round-trip data");

    close(dup_fd);
    close(fd);
    unlink(path);

    // Negative: dup_clear_o_direct(-1) returns -1.
    EXPECT_EQ_INT(wp::dup_clear_o_direct(-1), -1, "invalid fd returns -1");

    return fails;
}

// ---------------------------------------------------------------------------
// FileIOLayer (SyncPread)
// ---------------------------------------------------------------------------

static int test_file_io_sync_pread() {
    int fails = 0;

    // Write a known pattern to a temp file.
    char path[] = "/tmp/wp-test-io-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr, "  FAIL: %s: mkstemp failed: %s\n", __func__, std::strerror(errno));
        return 1;
    }
    constexpr size_t N = 4096;
    std::vector<uint8_t> pattern(N);
    for (size_t i = 0; i < N; ++i) pattern[i] = (uint8_t) (i * 7 + 13);
    ssize_t w = write(fd, pattern.data(), N);
    EXPECT_EQ_INT((size_t) w, N, "wrote pattern");

    // Hand the fd to FileIOLayer (sync path, no io_uring).
    std::vector<int> fds = { fd };
    auto layer = wp::create_file_io(std::move(fds), /*prefer_async=*/false, 4);
    EXPECT(layer != nullptr, "create_file_io (sync) returns non-null");
    if (!layer) {
        unlink(path);
        return fails;
    }

    // Issue 3 reads of different ranges with distinct req_ids.
    std::vector<uint8_t> dst1(1024);
    std::vector<uint8_t> dst2(2048);
    std::vector<uint8_t> dst3(512);
    bool ok1 = layer->submit(/*req=*/100, /*fd_idx=*/0,    0, 1024, dst1.data());
    bool ok2 = layer->submit(/*req=*/200, /*fd_idx=*/0, 1024, 2048, dst2.data());
    bool ok3 = layer->submit(/*req=*/300, /*fd_idx=*/0, 3072,  512, dst3.data());
    EXPECT(ok1, "submit #1");
    EXPECT(ok2, "submit #2");
    EXPECT(ok3, "submit #3");

    // Drain completions; verify req_id round-trips and bytes.
    int n_completed = 0;
    bool seen_100 = false, seen_200 = false, seen_300 = false;
    for (int i = 0; i < 10 && n_completed < 3; ++i) {
        wp::IoResult r = layer->wait_any(/*timeout_ms=*/0);
        if (r.status == wp::IoStatus::Timeout) break;
        ++n_completed;
        if (r.req_id == 100)      { seen_100 = true; EXPECT_EQ_INT(r.bytes_read, 1024, "req 100 bytes"); EXPECT(r.status == wp::IoStatus::Ok, "req 100 status"); }
        else if (r.req_id == 200) { seen_200 = true; EXPECT_EQ_INT(r.bytes_read, 2048, "req 200 bytes"); EXPECT(r.status == wp::IoStatus::Ok, "req 200 status"); }
        else if (r.req_id == 300) { seen_300 = true; EXPECT_EQ_INT(r.bytes_read,  512, "req 300 bytes"); EXPECT(r.status == wp::IoStatus::Ok, "req 300 status"); }
        else                      { EXPECT(false, "unknown req_id in completion"); }
    }
    EXPECT_EQ_INT(n_completed, 3, "all completions drained");
    EXPECT(seen_100 && seen_200 && seen_300, "all three req_ids seen");

    // Verify content: bytes match the pattern.
    bool match1 = (std::memcmp(dst1.data(), pattern.data() + 0,    1024) == 0);
    bool match2 = (std::memcmp(dst2.data(), pattern.data() + 1024, 2048) == 0);
    bool match3 = (std::memcmp(dst3.data(), pattern.data() + 3072,  512) == 0);
    EXPECT(match1, "data 1 matches pattern");
    EXPECT(match2, "data 2 matches pattern");
    EXPECT(match3, "data 3 matches pattern");

    // Out-of-range fd_idx -> submit returns false (not queued).
    std::vector<uint8_t> ignore(64);
    bool bad = layer->submit(999, /*fd_idx=*/5, 0, 64, ignore.data());
    EXPECT(!bad, "submit with bad fd_idx returns false");

    // Layer takes ownership of fds (closes them on destruction).
    layer.reset();
    unlink(path);

    return fails;
}

// ---------------------------------------------------------------------------
// PoolAllocator
// ---------------------------------------------------------------------------

static int test_pool_allocator() {
    int fails = 0;

    // Use the CPU backend's buffer type — pool allocator only depends on
    // the ggml buffer-type interface, not on a specific device.
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    EXPECT(buft != nullptr, "cpu buffer_type available");
    if (!buft) return fails;

    wp::PoolAllocator pool;
    bool ok = pool.init(buft, /*n_slots=*/4, /*slot_size=*/256);
    EXPECT(ok, "pool init");
    EXPECT_EQ_INT(pool.n_slots(), 4, "n_slots");
    EXPECT_EQ_INT((int)pool.slot_size(), 256, "slot_size");
    EXPECT(pool.vram_buf() != nullptr, "vram_buf valid");

    // Slot pointers are distinct and stride matches slot_size.
    void * p0 = pool.slot_ptr(0);
    void * p1 = pool.slot_ptr(1);
    void * p3 = pool.slot_ptr(3);
    EXPECT(p0 != nullptr, "slot 0 ptr non-null");
    EXPECT_EQ_INT((intptr_t)p1 - (intptr_t)p0, 256, "slot 1 - slot 0 stride");
    EXPECT_EQ_INT((intptr_t)p3 - (intptr_t)p0, 768, "slot 3 - slot 0 stride");
    EXPECT(pool.slot_ptr(4) == nullptr, "slot OOB returns null");
    EXPECT(pool.slot_ptr(-1) == nullptr, "negative slot returns null");

    // Allocation: first 4 slots come back free, then eviction starts.
    int evict_called = 0;
    int last_evicted = -1;
    pool.set_eviction_callback([&](int slot) {
        ++evict_called;
        last_evicted = slot;
    });

    int s0 = pool.alloc_slot();
    int s1 = pool.alloc_slot();
    int s2 = pool.alloc_slot();
    int s3 = pool.alloc_slot();
    EXPECT_EQ_INT(s0, 0, "first alloc");
    EXPECT_EQ_INT(s1, 1, "second alloc");
    EXPECT_EQ_INT(s2, 2, "third alloc");
    EXPECT_EQ_INT(s3, 3, "fourth alloc");
    EXPECT_EQ_INT(evict_called, 0, "no eviction yet");

    // Fifth alloc triggers eviction. LRU is slot 0 (lowest tick).
    int s4 = pool.alloc_slot();
    EXPECT_EQ_INT(s4, 0, "evicted LRU = slot 0");
    EXPECT_EQ_INT(evict_called, 1, "eviction callback fired once");
    EXPECT_EQ_INT(last_evicted, 0, "callback said slot 0 evicted");

    // mark_used bumps LRU. After bumping slot 1, the new LRU is slot 2.
    pool.mark_used(1);
    int s5 = pool.alloc_slot();
    EXPECT_EQ_INT(s5, 2, "evicted next LRU after mark_used(1) = slot 2");
    EXPECT_EQ_INT(evict_called, 2, "eviction callback fired twice");
    EXPECT_EQ_INT(last_evicted, 2, "callback said slot 2 evicted");

    // release_slot makes a slot free without eviction.
    pool.release_slot(3);
    int s6 = pool.alloc_slot();
    EXPECT_EQ_INT(s6, 3, "alloc after release returns released slot");
    EXPECT_EQ_INT(evict_called, 2, "no extra eviction after release");

    // lru_slot returns the current LRU (read-only inspection).
    int lru = pool.lru_slot();
    EXPECT(lru >= 0 && lru < 4, "lru_slot in range");

    return fails;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int total_fails = 0;

    struct named_test {
        const char * name;
        int (*fn)();
    };
    named_test tests[] = {
        { "page_catalog",       test_page_catalog       },
        { "dup_clear_o_direct", test_dup_clear_o_direct },
        { "file_io_sync_pread", test_file_io_sync_pread },
        { "pool_allocator",     test_pool_allocator     },
    };

    for (const auto & t : tests) {
        std::fprintf(stderr, "RUN  test_%s\n", t.name);
        int f = t.fn();
        std::fprintf(stderr, "%s test_%s (%d failure%s)\n",
                     f == 0 ? "PASS" : "FAIL", t.name, f, f == 1 ? "" : "s");
        total_fails += f;
    }

    std::fprintf(stderr, "\n=== %s: %d total failures ===\n",
                 total_fails == 0 ? "PASS" : "FAIL", total_fails);
    return total_fails == 0 ? 0 : 1;
}
