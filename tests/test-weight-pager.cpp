// Unit tests for the wp::* modules under src/weight-pager/.
//
// Lightweight, no test framework. Each TEST_FN runs subtests and returns
// the number of failures. main() sums them and exits non-zero if any
// failed. Tests that require GPU (hip*) are gated on GGML_USE_HIP at
// runtime — they no-op compile-out under non-HIP builds.

#include "weight-pager/wp-page-catalog.h"
#include "weight-pager/wp-eval-cb.h"
#include "weight-pager/wp-file-io.h"
#include "weight-pager/wp-host-tier.h"
#include "weight-pager/wp-pager.h"   // compute_advise_ranges / AdviseRange
#include "weight-pager/wp-pool.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

struct ScopedEnv {
    explicit ScopedEnv(const char * name_) : name(name_) {
        const char * v = std::getenv(name);
        if (v != nullptr) {
            had = true;
            old = v;
        }
    }
    ~ScopedEnv() {
        if (had) {
            setenv(name, old.c_str(), 1);
        } else {
            unsetenv(name);
        }
    }

    const char * name;
    bool had = false;
    std::string old;
};

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
// PageCatalog — MoE / block classification (Phase 1 of MAD-88)
// ---------------------------------------------------------------------------

static int test_page_catalog_moe_classification() {
    int fails = 0;
    wp::PageCatalog cat;

    // Non-block tensor — no block prefix at all.
    int i_embed = cat.add("token_embd.weight", 0, 0, 1024);

    // Block-scoped non-expert — attention.
    int i_attnq = cat.add("blk.0.attn_q.weight", 0, 1024, 4096);

    // Block-scoped non-expert — dense FFN (no _exps suffix, no expert idx).
    int i_dense_ffn = cat.add("blk.1.ffn_down.weight", 0, 5120, 8192);

    // Consolidated MoE expert — Qwen3-MoE style. One tensor packs all
    // experts of one role.
    int i_cons_gate = cat.add("blk.5.ffn_gate_exps.weight", 0, 10000, 65536);
    int i_cons_up   = cat.add("blk.5.ffn_up_exps.weight",   0, 75536, 65536);
    int i_cons_down = cat.add("blk.5.ffn_down_exps.weight", 0, 141072, 65536);

    // Per-expert MoE — Mixtral style. One tensor per (role, expert).
    int i_pe_up_7   = cat.add("blk.10.ffn_up.7.weight",   0, 200000, 4096);
    int i_pe_gate_3 = cat.add("blk.10.ffn_gate.3.weight", 0, 204096, 4096);
    int i_pe_down_0 = cat.add("blk.12.ffn_down.0.weight", 0, 208192, 4096);

    // 1. Non-block tensor: all block/expert fields default.
    {
        const auto & m = cat.at(i_embed);
        EXPECT_EQ_INT(m.block_idx, -1, "embed: block_idx defaults to -1");
        EXPECT_EQ_INT(m.expert_idx, -1, "embed: expert_idx defaults to -1");
        EXPECT(!m.is_expert, "embed: not an expert");
    }

    // 2. Block-scoped non-expert: block_idx parsed, expert fields default.
    {
        const auto & m = cat.at(i_attnq);
        EXPECT_EQ_INT(m.block_idx, 0, "attn_q: block_idx parsed");
        EXPECT_EQ_INT(m.expert_idx, -1, "attn_q: expert_idx default");
        EXPECT(!m.is_expert, "attn_q: not an expert");
        EXPECT_EQ_INT(m.expert_role_mask, 0, "attn_q: no role bits");
    }

    // 3. Dense FFN: looks like role-prefixed but no _exps and no expert idx —
    //    must NOT be classified as expert.
    {
        const auto & m = cat.at(i_dense_ffn);
        EXPECT_EQ_INT(m.block_idx, 1, "dense ffn: block_idx parsed");
        EXPECT(!m.is_expert, "dense ffn: not classified as expert");
        EXPECT_EQ_INT(m.expert_role_mask, 0, "dense ffn: no role bits");
    }

    // 4. Consolidated experts: is_expert=true, is_consolidated=true,
    //    role mask set, expert_idx stays -1 (tensor packs all experts).
    {
        const auto & g = cat.at(i_cons_gate);
        EXPECT_EQ_INT(g.block_idx, 5, "cons gate: block");
        EXPECT(g.is_expert, "cons gate: is_expert");
        EXPECT(g.is_consolidated, "cons gate: is_consolidated");
        EXPECT_EQ_INT(g.expert_idx, -1, "cons gate: expert_idx -1");
        EXPECT_EQ_INT(g.expert_role_mask, wp::ROLE_GATE, "cons gate: role bit");

        const auto & u = cat.at(i_cons_up);
        EXPECT(u.is_expert && u.is_consolidated, "cons up: is_expert + cons");
        EXPECT_EQ_INT(u.expert_role_mask, wp::ROLE_UP, "cons up: role bit");

        const auto & d = cat.at(i_cons_down);
        EXPECT(d.is_expert && d.is_consolidated, "cons down: is_expert + cons");
        EXPECT_EQ_INT(d.expert_role_mask, wp::ROLE_DOWN, "cons down: role bit");
    }

    // 5. Per-expert: is_expert=true, is_consolidated=false, expert_idx set.
    {
        const auto & u7 = cat.at(i_pe_up_7);
        EXPECT_EQ_INT(u7.block_idx, 10, "pe up7: block");
        EXPECT(u7.is_expert, "pe up7: is_expert");
        EXPECT(!u7.is_consolidated, "pe up7: not consolidated");
        EXPECT_EQ_INT(u7.expert_idx, 7, "pe up7: expert idx");
        EXPECT_EQ_INT(u7.expert_role_mask, wp::ROLE_UP, "pe up7: role bit");

        const auto & g3 = cat.at(i_pe_gate_3);
        EXPECT_EQ_INT(g3.expert_idx, 3, "pe gate3: expert idx");
        EXPECT_EQ_INT(g3.expert_role_mask, wp::ROLE_GATE, "pe gate3: role bit");

        const auto & d0 = cat.at(i_pe_down_0);
        EXPECT_EQ_INT(d0.block_idx, 12, "pe down0: block");
        EXPECT_EQ_INT(d0.expert_idx, 0, "pe down0: expert idx");
        EXPECT_EQ_INT(d0.expert_role_mask, wp::ROLE_DOWN, "pe down0: role bit");
    }

    // 6. has_experts() / n_expert_pages() summary.
    EXPECT(cat.has_experts(), "catalog has experts");
    EXPECT_EQ_INT(cat.n_expert_pages(), 6, "n_expert_pages: 3 cons + 3 per-expert");

    // 7. pages_for_block lookup.
    {
        auto blk5 = cat.pages_for_block(5);
        EXPECT_EQ_INT(blk5.size(), 3, "blk 5 has 3 consolidated experts");

        auto blk10 = cat.pages_for_block(10);
        EXPECT_EQ_INT(blk10.size(), 2, "blk 10 has 2 per-expert tensors");

        auto blk_none = cat.pages_for_block(99);
        EXPECT_EQ_INT(blk_none.size(), 0, "non-existent block returns empty");
    }

    // 8. pages_for_expert lookup — per-expert path.
    {
        auto blk10_e7 = cat.pages_for_expert(10, 7);
        EXPECT_EQ_INT(blk10_e7.size(), 1, "blk 10 expert 7: just up.7");
        if (!blk10_e7.empty()) {
            EXPECT(cat.at(blk10_e7[0]).expert_role_mask == wp::ROLE_UP,
                   "blk 10 expert 7 is the up tensor");
        }

        auto blk12_e0 = cat.pages_for_expert(12, 0);
        EXPECT_EQ_INT(blk12_e0.size(), 1, "blk 12 expert 0: just down.0");

        // Consolidated experts have expert_idx=-1; pass -1 to retrieve them.
        auto blk5_cons = cat.pages_for_expert(5, -1);
        EXPECT_EQ_INT(blk5_cons.size(), 3, "blk 5 consolidated: 3 role tensors");
    }

    // 9. Bad input: block prefix with non-numeric idx must not classify.
    {
        wp::PageCatalog c2;
        int idx = c2.add("blk.bad.attn_q.weight", 0, 0, 100);
        EXPECT_EQ_INT(c2.at(idx).block_idx, -1, "non-numeric block idx not parsed");
        EXPECT(!c2.at(idx).is_expert, "no expert classification on bad block");
    }

    // 10. Bad input: per-expert with non-numeric expert idx must not classify.
    {
        wp::PageCatalog c2;
        int idx = c2.add("blk.0.ffn_up.bad.weight", 0, 0, 100);
        EXPECT_EQ_INT(c2.at(idx).block_idx, 0, "block parsed despite bad expert");
        EXPECT(!c2.at(idx).is_expert, "bad expert idx not classified");
    }

    return fails;
}

// ---------------------------------------------------------------------------
// PageCatalog — consolidated MoE expert splitting (Phase 2 of MAD-88)
// ---------------------------------------------------------------------------

static int test_page_catalog_consolidated_split() {
    int fails = 0;
    wp::PageCatalog cat;

    // Register a consolidated MoE tensor: 4 experts, total 4096 bytes.
    // Per-expert size = 4096 / 4 = 1024 bytes.
    const std::string parent_name = "blk.5.ffn_gate_exps.weight";
    constexpr int     n_experts   = 4;
    constexpr size_t  total_size  = 4096;
    constexpr size_t  per_expert  = total_size / n_experts;
    constexpr uint64_t base_off   = 100000;

    int first_sub = cat.add_consolidated_experts(parent_name, 0, base_off, total_size, n_experts);

    // The catalog should now have 1 parent + N sub-pages = 5 entries.
    EXPECT_EQ_INT(cat.size(), 1 + n_experts, "size after consolidated add");

    // First sub-expert is at index 1 (parent at 0).
    EXPECT_EQ_INT(first_sub, 1, "first sub-expert index");

    // 1. Parent meta — pure metadata, is_consolidated, is_expert=false
    //    (parent isn't slottable; its children are).
    {
        const auto & p = cat.at(0);
        EXPECT(p.tensor_name == parent_name, "parent name");
        EXPECT_EQ_INT(p.size, total_size, "parent size = full consolidated");
        EXPECT(p.is_consolidated, "parent is_consolidated");
        EXPECT(!p.is_expert, "parent NOT counted as expert (children are)");
        EXPECT(!p.is_sub_expert, "parent is NOT a sub-expert");
        EXPECT_EQ_INT(p.block_idx, 5, "parent block parsed");
        EXPECT_EQ_INT(p.expert_role_mask, wp::ROLE_GATE, "parent role parsed");
        EXPECT_EQ_INT(p.parent_page_idx, -1, "parent has no parent");
    }

    // 2. Sub-experts — N entries with synthetic names + per-expert offsets.
    for (int e = 0; e < n_experts; ++e) {
        const int  sub_idx = first_sub + e;
        const auto & s     = cat.at(sub_idx);

        const std::string expected_name = parent_name + "#expert." + std::to_string(e);
        EXPECT(s.tensor_name == expected_name, "sub: synthetic name");
        EXPECT_EQ_INT(s.file_offset, base_off + (uint64_t)e * per_expert, "sub: offset");
        EXPECT_EQ_INT(s.size, per_expert, "sub: per-expert size");
        EXPECT(s.is_expert, "sub: is_expert");
        EXPECT(s.is_sub_expert, "sub: is_sub_expert");
        EXPECT(!s.is_consolidated, "sub: NOT consolidated itself");
        EXPECT_EQ_INT(s.block_idx, 5, "sub: inherited block_idx");
        EXPECT_EQ_INT(s.expert_idx, e, "sub: expert_idx");
        EXPECT_EQ_INT(s.expert_role_mask, wp::ROLE_GATE, "sub: inherited role");
        EXPECT_EQ_INT(s.parent_page_idx, 0, "sub: parent_page_idx");
    }

    // 3. Synthetic names are findable via the standard find() lookup.
    {
        EXPECT_EQ_INT(cat.find(parent_name), 0, "find: parent by original name");
        EXPECT_EQ_INT(cat.find(parent_name + "#expert.0"), 1, "find: sub by synthetic name");
        EXPECT_EQ_INT(cat.find(parent_name + "#expert.3"), 4, "find: last sub");
        EXPECT_EQ_INT(cat.find(parent_name + "#expert.4"), -1, "find: out-of-range expert");
    }

    // 4. has_experts / n_expert_pages — only sub-experts count.
    EXPECT(cat.has_experts(), "has_experts after consolidated add");
    EXPECT_EQ_INT(cat.n_expert_pages(), n_experts, "n_expert_pages = sub-experts only");

    // 5. max_page_size tracks per-expert size (not the consolidated total),
    //    since only sub-experts allocate slots.
    EXPECT_EQ_INT(cat.max_page_size(), per_expert, "max_page_size = per-expert");

    // 6. pages_for_block(5) returns parent + all sub-experts.
    {
        auto blk5 = cat.pages_for_block(5);
        EXPECT_EQ_INT(blk5.size(), 1 + n_experts, "blk 5 includes parent + N subs");
    }

    // 7. pages_for_expert(5, 2) returns just the e=2 sub-expert.
    {
        auto e2 = cat.pages_for_expert(5, 2);
        EXPECT_EQ_INT(e2.size(), 1, "blk 5 expert 2: one sub");
        if (!e2.empty()) {
            EXPECT_EQ_INT(cat.at(e2[0]).expert_idx, 2, "found the e=2 sub");
        }
    }

    // 8. Non-uniform sizes: total not divisible by n_experts → falls back
    //    to single-page registration (no children registered, no sub-experts).
    //    The name still classifies as consolidated by string pattern, but
    //    no slottable per-expert children exist.
    {
        wp::PageCatalog c2;
        int idx = c2.add_consolidated_experts("blk.0.ffn_up_exps.weight", 0, 0,
                                              /*total_size=*/100, /*n_experts=*/3);
        EXPECT_EQ_INT(c2.size(), 1, "non-uniform: single-page fallback");
        EXPECT_EQ_INT(idx, 0, "non-uniform: returned single-page index");
        EXPECT(!c2.at(0).is_sub_expert, "non-uniform: not a sub-expert");
        // n_expert_pages counts entries with is_expert=true. The fallback
        // registers ONE entry which the parser sees as a consolidated
        // expert tensor (by name) — so n_expert_pages = 1 is consistent
        // (the unsplittable parent IS itself an expert page in this case).
        EXPECT_EQ_INT(c2.n_expert_pages(), 1, "non-uniform: parent counted as expert");
    }

    // 9. n_experts <= 1 → falls back to plain add().
    {
        wp::PageCatalog c2;
        int idx = c2.add_consolidated_experts("blk.0.ffn_up_exps.weight", 0, 0, 1024, 1);
        EXPECT_EQ_INT(c2.size(), 1, "n_experts=1: single-page");
        EXPECT_EQ_INT(idx, 0, "n_experts=1: returned single-page index");
        EXPECT(!c2.at(0).is_sub_expert, "n_experts=1: not a sub-expert");
    }

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

static int test_pool_size_class_packs_small_pages() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/2, /*slot_size=*/256), "pool init");
    EXPECT(pool.size_class_slots_enabled(), "size-class mode enabled");
    EXPECT_EQ_INT(pool.slot_size(), 256u, "max slot_size remains max page size");
    EXPECT_EQ_INT(pool.pool_size(), 512u, "arena budget matches fixed-slot bytes");
    EXPECT(pool.pool_base() != nullptr, "pool base valid before first slot");

    int evicted = -1;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });

    int s0 = pool.alloc_slot(64);
    int s1 = pool.alloc_slot(64);
    int s2 = pool.alloc_slot(64);
    int s3 = pool.alloc_slot(64);
    (void) s2;
    EXPECT_EQ_INT(s0, 0, "small alloc 0");
    EXPECT_EQ_INT(s3, 3, "small alloc 3");
    EXPECT_EQ_INT(pool.n_slots(), 4, "four small slots fit in one old 256-byte slot");
    EXPECT_EQ_INT(pool.slot_size(s0), 64u, "slot 0 class size");
    EXPECT_EQ_INT((intptr_t) pool.slot_ptr(s1) - (intptr_t) pool.slot_ptr(s0),
                  64, "small slot stride is class size");

    void * p1 = pool.slot_ptr(s1);
    int s4 = pool.alloc_slot(256);
    EXPECT_EQ_INT(s4, 4, "large slot allocated after small slots");
    EXPECT_EQ_INT(pool.n_slots(), 5, "dynamic slot id added inside same arena budget");
    EXPECT_EQ_INT(pool.slot_size(s4), 256u, "large slot class size");
    EXPECT_EQ_INT((intptr_t) pool.slot_ptr(s4) - (intptr_t) pool.pool_base(),
                  256, "large slot starts after four 64-byte slots");
    EXPECT(pool.slot_ptr(s1) == p1, "existing slot address is stable");
    EXPECT(pool.slot_base_for_capture(s1) == p1, "capture base matches stable slot ptr");

    int s5 = pool.alloc_slot(64);
    EXPECT_EQ_INT(s5, s0, "full arena evicts LRU within requested size class");
    EXPECT_EQ_INT(evicted, s0, "eviction callback reports reused slot");
    EXPECT_EQ_INT(pool.n_slots(), 5, "eviction reuses existing slot id");
    return fails;
}

static int test_pool_size_class_pin_skip() {
    int fails = 0;
    ScopedEnv env("WP_SIZE_CLASS_SLOTS");
    setenv("WP_SIZE_CLASS_SLOTS", "1", 1);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/2, /*slot_size=*/64), "pool init");

    int s0 = pool.alloc_slot(64);
    int s1 = pool.alloc_slot(64);
    EXPECT_EQ_INT(s0, 0, "alloc 0");
    EXPECT_EQ_INT(s1, 1, "alloc 1");

    pool.pin_slot(s0);
    pool.pin_slot(s1);
    int evicted = -42;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });

    int s2 = pool.alloc_slot(64);
    EXPECT_EQ_INT(s2, -1, "all pinned size-class slots return -1");
    EXPECT_EQ_INT(evicted, -42, "eviction callback not fired when all pinned");

    pool.unpin_slot(s1);
    int s3 = pool.alloc_slot(64);
    EXPECT_EQ_INT(s3, s1, "unpinned slot becomes evictable");
    EXPECT_EQ_INT(evicted, s1, "eviction callback fired for unpinned slot");
    pool.unpin_slot(s0);
    return fails;
}

// ---------------------------------------------------------------------------
// PoolAllocator — popularity counter + hot-slot protection (MAD-237)
// ---------------------------------------------------------------------------
//
// Tests cover:
//   - mark_used increments hit_count_ alongside LRU bump
//   - Default threshold 0 ⇒ identical behavior to MAD-231 pure LRU
//     (re-check existing pool_allocator test still passes — covered by
//      that test's continued existence)
//   - With threshold > 0, alloc_slot Pass A skips hot slots and picks the
//     LRU among cold; Pass B falls back to LRU-among-unpinned when all are
//     hot (so the pool always makes forward progress)
//   - Evicted slot's hit_count resets to 0 (new owner has no history)
//   - lru_walk_hot_skips_ counter increments correctly
//   - Periodic decay halves all counters every kDecayEvery evictions

static int test_pool_hit_count_basic() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buft unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");

    // alloc, mark_used 3 times → hit_count should be 3.
    int s = pool.alloc_slot();
    EXPECT_EQ_INT(pool.hit_count(s), 0u, "fresh alloc has hit_count 0");
    pool.mark_used(s);
    pool.mark_used(s);
    pool.mark_used(s);
    EXPECT_EQ_INT(pool.hit_count(s), 3u, "hit_count == 3 after 3 mark_used");

    // OOB hit_count returns 0 safely.
    EXPECT_EQ_INT(pool.hit_count(-1), 0u, "OOB negative hit_count returns 0");
    EXPECT_EQ_INT(pool.hit_count(99), 0u, "OOB past-end hit_count returns 0");
    return fails;
}

static int test_pool_hot_threshold_protects_in_eviction() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buft unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");

    // Set threshold = 2: hit_count > 2 is "hot".
    pool.set_hot_hit_threshold(2);
    EXPECT_EQ_INT(pool.hot_hit_threshold(), 2u, "threshold set");

    // Allocate all 4 slots in order; LRU order is 0,1,2,3.
    int s0 = pool.alloc_slot();
    int s1 = pool.alloc_slot();
    int s2 = pool.alloc_slot();
    int s3 = pool.alloc_slot();
    EXPECT_EQ_INT(s0, 0, "alloc 0");
    EXPECT_EQ_INT(s3, 3, "alloc 3");

    // Slot 0 is LRU. Bump its hit_count above the threshold to make it hot.
    pool.mark_used(0);  // 1
    pool.mark_used(0);  // 2
    pool.mark_used(0);  // 3 (now > threshold of 2 → HOT)
    EXPECT_EQ_INT(pool.hit_count(0), 3u, "slot 0 hot with count 3");

    // BUT mark_used also bumps LRU tick. Re-LRU is now slot 1.
    // We want slot 0 to BE THE LRU FRONT but skipped because hot.
    // Force the LRU back to 0 by bumping the others harder.
    pool.mark_used(1); pool.mark_used(1);  // s1 hit_count=2 (NOT hot at threshold=2)
    pool.mark_used(2);                     // s2 hit_count=1
    pool.mark_used(3); pool.mark_used(3); pool.mark_used(3);  // s3 hit_count=3 HOT too
    // Now LRU tick order (oldest first): 0 has the oldest tick because
    // we bumped 1,2,3 after 0's last bump.
    // hit_counts: s0=3 (hot), s1=2 (cold), s2=1 (cold), s3=3 (hot).
    // Pass A should skip 0 and 3, pick LRU among {1, 2}. Tick ordering
    // had 1 first then 2 → s1 is LRU.

    int evicted = -1;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });
    int s4 = pool.alloc_slot();
    EXPECT_EQ_INT(s4, 1, "Pass A picked LRU among cold (s1), not the hot s0");
    EXPECT_EQ_INT(evicted, 1, "eviction CB fired on slot 1");
    EXPECT_EQ_INT(pool.hit_count(1), 0u, "evicted slot's hit_count reset");
    EXPECT(pool.lru_walk_hot_skips() >= 2, "telemetry: skipped >= 2 hot slots");
    return fails;
}

static int test_pool_hot_fallback_when_all_hot() {
    // If every unpinned slot is hot, Pass B picks pure-LRU regardless.
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buft unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/3, /*slot_size=*/64), "pool init");
    pool.set_hot_hit_threshold(1);

    int s0 = pool.alloc_slot();
    int s1 = pool.alloc_slot();
    int s2 = pool.alloc_slot();
    (void) s2;

    // Make all 3 slots hot.
    pool.mark_used(0); pool.mark_used(0);
    pool.mark_used(1); pool.mark_used(1);
    pool.mark_used(2); pool.mark_used(2);
    // hit_counts all = 2 (> threshold 1) → all hot. Bumped most recently was s2.
    // LRU now is s0 (oldest tick).

    int evicted = -1;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });
    int s3 = pool.alloc_slot();
    EXPECT_EQ_INT(s3, s0, "Pass B fallback: evicted LRU even though all hot (slot 0)");
    EXPECT_EQ_INT(evicted, s0, "eviction CB fired on the LRU");
    return fails;
}

static int test_pool_default_threshold_zero_is_pure_lru() {
    // Threshold = 0 (the default) means hot protection is disabled.
    // Behavior must be identical to MAD-231-only.
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buft unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");
    EXPECT_EQ_INT(pool.hot_hit_threshold(), 0u, "default threshold == 0");

    // Fill, make slot 0 very hot, then alloc. With threshold=0, hot
    // protection is OFF — slot 0 should still be evicted as the LRU.
    int s0 = pool.alloc_slot();
    int s1 = pool.alloc_slot();
    int s2 = pool.alloc_slot();
    int s3 = pool.alloc_slot();
    (void) s1; (void) s2; (void) s3;
    pool.mark_used(0); pool.mark_used(0); pool.mark_used(0);  // very hot
    // mark_used also bumps LRU tick → s0 is now MRU, not LRU. Bump others
    // to push s0 back to LRU.
    pool.mark_used(1); pool.mark_used(2); pool.mark_used(3);

    int evicted = -1;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });
    int s4 = pool.alloc_slot();
    EXPECT_EQ_INT(s4, 0, "threshold=0: pure LRU evicts slot 0 despite hit_count");
    EXPECT_EQ_INT(evicted, 0, "eviction CB on slot 0");
    EXPECT_EQ_INT(pool.lru_walk_hot_skips(), 0u, "no hot skips when disabled");
    return fails;
}

// ---------------------------------------------------------------------------
// PageCatalog — always-resident pinning (MAD-236)
// ---------------------------------------------------------------------------
//
// Tests cover:
//   - add_pinned registers correctly: is_pinned=true, resident_ptr matches,
//     size tracked, file_idx/file_offset zeroed
//   - Name-based lookup works for pinned entries (find returns the index)
//   - n_pinned_pages + pinned_bytes telemetry increments correctly
//   - Pinned entries do NOT inflate max_page_size (must not break slot stride)
//   - Pinned entries do NOT count toward n_expert_pages even if name pattern
//     looks expert-shaped (they live outside the slot pool)
//   - block_idx is still parsed from the name (useful for per-layer telemetry)
//   - clear() resets the pinned counters

static int test_catalog_add_pinned_basic() {
    int fails = 0;
    wp::PageCatalog cat;

    // Sentinel pointers — we don't dereference, just check round-trip.
    void * fake_embed_ptr  = reinterpret_cast<void *>(0x1000);
    void * fake_norm_ptr   = reinterpret_cast<void *>(0x2000);
    void * fake_router_ptr = reinterpret_cast<void *>(0x3000);

    int i_embed  = cat.add_pinned("token_embd.weight",   fake_embed_ptr,  1024 * 1024);
    int i_norm   = cat.add_pinned("output_norm.weight",  fake_norm_ptr,   8 * 1024);
    int i_router = cat.add_pinned("blk.0.ffn_gate_inp.weight", fake_router_ptr, 32 * 1024);

    EXPECT_EQ_INT(cat.size(), 3, "3 pinned pages");
    EXPECT_EQ_INT(cat.n_pinned_pages(), 3, "n_pinned_pages == 3");
    EXPECT(cat.has_pinned(), "has_pinned == true");
    EXPECT_EQ_INT(cat.pinned_bytes(),
                  (size_t) (1024 * 1024 + 8 * 1024 + 32 * 1024),
                  "pinned_bytes sums correctly");

    // Per-entry round-trip.
    const auto & e = cat.at(i_embed);
    EXPECT(e.is_pinned, "embed is_pinned");
    EXPECT(e.resident_ptr == fake_embed_ptr, "embed resident_ptr matches");
    EXPECT_EQ_INT(e.size, (size_t) (1024 * 1024), "embed size");
    EXPECT_EQ_INT(e.file_idx, 0, "pinned file_idx zeroed");
    EXPECT_EQ_INT(e.file_offset, 0u, "pinned file_offset zeroed");

    // Name lookup.
    EXPECT_EQ_INT(cat.find("token_embd.weight"), i_embed, "find embed");
    EXPECT_EQ_INT(cat.find("output_norm.weight"), i_norm, "find norm");
    EXPECT_EQ_INT(cat.find("blk.0.ffn_gate_inp.weight"), i_router, "find router");

    // block_idx still parsed from name for the router (telemetry useful).
    EXPECT_EQ_INT(cat.at(i_router).block_idx, 0, "router block_idx parsed");

    // Pinned entries MUST NOT inflate max_page_size — it dictates slot
    // stride for the pool, and pinned tensors don't use slots.
    EXPECT_EQ_INT(cat.max_page_size(), 0u, "pinned does not inflate max_page_size");

    // Pinned entries MUST NOT count as expert pages even if the name pattern
    // matches (router weight has "ffn_" prefix but is single-tensor pinned).
    EXPECT_EQ_INT(cat.n_expert_pages(), 0, "pinned not counted as expert");
    EXPECT(!cat.at(i_router).is_expert, "pinned router not classified as expert");
    return fails;
}

static int test_catalog_add_pinned_mixed_with_paged() {
    int fails = 0;
    wp::PageCatalog cat;

    // Mix: 2 paged + 2 pinned + 1 paged.
    int p0 = cat.add("blk.0.ffn_gate.weight", 0, 1024, 65536);
    int p1 = cat.add("blk.0.ffn_up.weight",   0, 66560, 65536);
    int n0 = cat.add_pinned("token_embd.weight", reinterpret_cast<void *>(0x1000), 1024);
    int n1 = cat.add_pinned("output_norm.weight", reinterpret_cast<void *>(0x2000), 512);
    int p2 = cat.add("blk.0.ffn_down.weight", 0, 132096, 65536);
    (void) p0; (void) p1; (void) n0; (void) n1; (void) p2;

    EXPECT_EQ_INT(cat.size(), 5, "5 total");
    EXPECT_EQ_INT(cat.n_pinned_pages(), 2, "2 pinned");
    EXPECT_EQ_INT(cat.pinned_bytes(), 1536u, "pinned bytes 1024+512");

    // max_page_size only reflects PAGED entries (65536), not pinned.
    EXPECT_EQ_INT(cat.max_page_size(), 65536u, "max_page_size from paged only");

    // is_pinned flag distinguishes correctly.
    EXPECT(!cat.at(p0).is_pinned, "p0 not pinned");
    EXPECT(cat.at(n0).is_pinned,  "n0 pinned");
    EXPECT(cat.at(n1).is_pinned,  "n1 pinned");
    EXPECT(!cat.at(p2).is_pinned, "p2 not pinned");
    return fails;
}

static int test_catalog_clear_resets_pinned_counters() {
    int fails = 0;
    wp::PageCatalog cat;
    cat.add_pinned("a", reinterpret_cast<void *>(0x100), 256);
    cat.add_pinned("b", reinterpret_cast<void *>(0x200), 512);
    EXPECT_EQ_INT(cat.n_pinned_pages(), 2, "before clear: 2 pinned");
    EXPECT_EQ_INT(cat.pinned_bytes(), 768u, "before clear: bytes");

    cat.clear();
    EXPECT_EQ_INT(cat.n_pinned_pages(), 0, "after clear: 0 pinned");
    EXPECT_EQ_INT(cat.pinned_bytes(), 0u, "after clear: bytes 0");
    EXPECT(!cat.has_pinned(), "after clear: has_pinned false");
    return fails;
}

// ---------------------------------------------------------------------------
// PoolAllocator — slot pin / refcount (MAD-231)
// ---------------------------------------------------------------------------
//
// Tests cover:
//   - Basic pin/unpin lifecycle + is_pinned + pin_count + n_pinned accessors
//   - Refcount semantics: pin twice, unpin once → still pinned; second unpin clears
//   - alloc_slot skips pinned slots in the eviction LRU walk
//   - alloc_slot returns -1 (no crash, no eviction of pinned) when all are pinned
//   - lru_walk_pinned_skips_ telemetry counter increments correctly
//   - OOB pin/unpin is a logged no-op (does not corrupt state)
//   - Underflow unpin is a logged no-op (does not wrap pin_count to 65535)

static int test_pool_pin_basic() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }

    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");

    EXPECT_EQ_INT(pool.n_pinned(), 0, "initially no pins");
    EXPECT(!pool.is_pinned(0), "slot 0 starts unpinned");
    EXPECT_EQ_INT(pool.pin_count(0), 0, "slot 0 count 0");

    pool.pin_slot(0);
    EXPECT(pool.is_pinned(0), "slot 0 pinned after pin_slot");
    EXPECT_EQ_INT(pool.pin_count(0), 1, "slot 0 refcount 1");
    EXPECT_EQ_INT(pool.n_pinned(), 1, "n_pinned == 1");
    EXPECT(!pool.is_pinned(1), "other slots unaffected");

    pool.unpin_slot(0);
    EXPECT(!pool.is_pinned(0), "slot 0 unpinned");
    EXPECT_EQ_INT(pool.pin_count(0), 0, "slot 0 refcount back to 0");
    EXPECT_EQ_INT(pool.n_pinned(), 0, "n_pinned back to 0");
    return fails;
}

static int test_pool_pin_refcount() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");

    pool.pin_slot(2);
    pool.pin_slot(2);
    pool.pin_slot(2);
    EXPECT_EQ_INT(pool.pin_count(2), 3, "triple-pinned refcount 3");
    EXPECT(pool.is_pinned(2), "still pinned");

    pool.unpin_slot(2);
    EXPECT_EQ_INT(pool.pin_count(2), 2, "refcount 2 after one unpin");
    EXPECT(pool.is_pinned(2), "still pinned after one unpin");
    pool.unpin_slot(2);
    EXPECT(pool.is_pinned(2), "still pinned after two unpins");
    pool.unpin_slot(2);
    EXPECT(!pool.is_pinned(2), "fully unpinned after three unpins");

    // Underflow: extra unpin is a no-op (logged WARN), does NOT wrap.
    pool.unpin_slot(2);
    EXPECT_EQ_INT(pool.pin_count(2), 0, "underflow unpin stays at 0 (no wrap)");
    return fails;
}

static int test_pool_pin_oob_safe() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");

    // Out-of-range pin/unpin should not crash and should not affect state.
    pool.pin_slot(-1);
    pool.pin_slot(4);
    pool.pin_slot(999);
    pool.unpin_slot(-1);
    pool.unpin_slot(4);
    EXPECT_EQ_INT(pool.n_pinned(), 0, "OOB pin/unpin leaves state unchanged");
    EXPECT(!pool.is_pinned(-1), "is_pinned OOB returns false");
    EXPECT(!pool.is_pinned(4),  "is_pinned past end returns false");
    EXPECT_EQ_INT(pool.pin_count(-1), 0, "pin_count OOB returns 0");
    return fails;
}

static int test_pool_alloc_skips_pinned_in_eviction() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/4, /*slot_size=*/64), "pool init");

    // Fill the pool: slots 0..3 alloc'd in order, so LRU order is 0,1,2,3.
    int s0 = pool.alloc_slot();
    int s1 = pool.alloc_slot();
    int s2 = pool.alloc_slot();
    int s3 = pool.alloc_slot();
    EXPECT_EQ_INT(s0, 0, "alloc 0");
    EXPECT_EQ_INT(s1, 1, "alloc 1");
    EXPECT_EQ_INT(s2, 2, "alloc 2");
    EXPECT_EQ_INT(s3, 3, "alloc 3");

    // Pin slot 0 (the LRU front). Next eviction must skip it and pick slot 1.
    pool.pin_slot(0);
    int evicted = -1;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });
    EXPECT_EQ_INT(pool.lru_walk_pinned_skips(), 0u, "skip counter starts at 0");

    int s4 = pool.alloc_slot();
    EXPECT_EQ_INT(s4, 1, "evicted slot is the LRU among UNPINNED (slot 1, not 0)");
    EXPECT_EQ_INT(evicted, 1, "eviction callback fired on slot 1, not 0");
    EXPECT(pool.is_pinned(0), "pinned slot 0 unchanged");
    EXPECT_EQ_INT(pool.lru_walk_pinned_skips(), 1u, "telemetry: skipped slot 0 once");

    // Pin slot 2 too. Next eviction now skips 0 and 2; picks 3 (slot 1 was
    // just re-used so it's MRU, leaving 3 as next-LRU among unpinned).
    pool.pin_slot(2);
    evicted = -1;
    int s5 = pool.alloc_slot();
    EXPECT_EQ_INT(s5, 3, "skips pinned 0 and 2; picks LRU-unpinned (slot 3)");
    EXPECT_EQ_INT(evicted, 3, "eviction callback fired on slot 3");
    // Skip counter: this walk skipped both 0 and 2 = +2 → cumulative 3.
    EXPECT_EQ_INT(pool.lru_walk_pinned_skips(), 3u, "telemetry: cumulative skips = 3");

    pool.unpin_slot(0);
    pool.unpin_slot(2);
    return fails;
}

static int test_pool_alloc_returns_neg1_when_all_pinned() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type unavailable"); return fails; }
    wp::PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/3, /*slot_size=*/64), "pool init");

    // Fill and pin every slot.
    int s0 = pool.alloc_slot();
    int s1 = pool.alloc_slot();
    int s2 = pool.alloc_slot();
    EXPECT_EQ_INT(s0, 0, "alloc 0");
    EXPECT_EQ_INT(s1, 1, "alloc 1");
    EXPECT_EQ_INT(s2, 2, "alloc 2");
    pool.pin_slot(0);
    pool.pin_slot(1);
    pool.pin_slot(2);
    EXPECT_EQ_INT(pool.n_pinned(), 3, "all pinned");

    // alloc_slot should refuse cleanly with -1 — NEVER evict a pinned slot.
    int evicted = -42;
    pool.set_eviction_callback([&](int slot) { evicted = slot; });
    int s3 = pool.alloc_slot();
    EXPECT_EQ_INT(s3, -1, "alloc_slot returns -1 when all pinned");
    EXPECT_EQ_INT(evicted, -42, "eviction callback NOT fired");
    EXPECT(pool.is_pinned(0) && pool.is_pinned(1) && pool.is_pinned(2),
           "pinned state unchanged after refused alloc");

    // After unpinning one, allocation succeeds and reuses that slot.
    pool.unpin_slot(1);
    int s4 = pool.alloc_slot();
    EXPECT_EQ_INT(s4, 1, "after one unpin, alloc reuses that slot");
    EXPECT_EQ_INT(evicted, 1, "eviction callback fired on the newly-unpinned slot");

    pool.unpin_slot(0);
    pool.unpin_slot(2);
    return fails;
}

// ---------------------------------------------------------------------------
// HostTier - pinned/pageable host slab + LRU bookkeeping (MAD-P4)
// ---------------------------------------------------------------------------

static int test_host_tier_store_lookup() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/128, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> src(32);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t) (i * 3 + 1);

    EXPECT(tier.store(/*page_idx=*/7, src.data(), src.size()), "store page");
    const void * p = tier.lookup(7);
    EXPECT(p != nullptr, "lookup returns pointer");
    EXPECT(std::memcmp(p, src.data(), src.size()) == 0, "lookup bytes match");
    EXPECT_EQ_INT(tier.used_bytes(), src.size(), "used bytes after store");
    EXPECT_EQ_INT(tier.resident_count(), 1u, "one resident page");

    return fails;
}

static int test_host_tier_size_class_reuse() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> a(32, 0xA1);
    std::vector<uint8_t> b(32, 0xB2);
    std::vector<uint8_t> c(32, 0xC3);

    EXPECT(tier.store(1, a.data(), a.size()), "store page 1");
    const void * p1 = tier.lookup(1);
    EXPECT(tier.store(2, b.data(), b.size()), "store page 2");
    EXPECT(tier.store(3, c.data(), c.size()), "store page 3 evicts page 1");
    const void * p3 = tier.lookup(3);

    EXPECT(p1 != nullptr && p3 != nullptr, "pointers valid");
    EXPECT(p3 == p1, "same-size store reuses evicted slot");
    EXPECT(tier.lookup(1) == nullptr, "evicted page 1 missing");
    EXPECT(std::memcmp(p3, c.data(), c.size()) == 0, "reused slot has new bytes");

    return fails;
}

static int test_host_tier_lru_eviction_order() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x11);
    EXPECT(tier.store(10, bytes.data(), bytes.size()), "store 10");
    EXPECT(tier.store(11, bytes.data(), bytes.size()), "store 11");
    EXPECT(tier.store(12, bytes.data(), bytes.size()), "store 12");
    EXPECT(tier.store(13, bytes.data(), bytes.size()), "store 13 evicts oldest");

    EXPECT(tier.lookup(10) == nullptr, "oldest page evicted first");
    EXPECT(tier.lookup(11) != nullptr, "page 11 still resident");
    EXPECT(tier.lookup(12) != nullptr, "page 12 still resident");
    EXPECT(tier.lookup(13) != nullptr, "new page resident");
    EXPECT_EQ_INT(tier.resident_count(), 3u, "resident count stays at capacity");

    return fails;
}

static int test_host_tier_lookup_touch_keeps_mru() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x22);
    EXPECT(tier.store(20, bytes.data(), bytes.size()), "store 20");
    EXPECT(tier.store(21, bytes.data(), bytes.size()), "store 21");
    EXPECT(tier.store(22, bytes.data(), bytes.size()), "store 22");

    EXPECT(tier.lookup(20) != nullptr, "touch page 20");
    EXPECT(tier.store(23, bytes.data(), bytes.size()), "store 23 evicts LRU after touch");

    EXPECT(tier.lookup(20) != nullptr, "touched page kept as MRU");
    EXPECT(tier.lookup(21) == nullptr, "untouched oldest page evicted");
    EXPECT(tier.lookup(22) != nullptr, "page 22 still resident");
    EXPECT(tier.lookup(23) != nullptr, "page 23 resident");

    return fails;
}

static int test_host_tier_over_budget_evict() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x33);
    EXPECT(tier.store(30, bytes.data(), bytes.size()), "store 30");
    EXPECT(tier.store(31, bytes.data(), bytes.size()), "store 31");
    EXPECT_EQ_INT(tier.used_bytes(), 64u, "budget full");

    EXPECT(tier.store(32, bytes.data(), bytes.size()), "store beyond used budget evicts and succeeds");
    EXPECT_EQ_INT(tier.used_bytes(), 64u, "used bytes remains capped");
    EXPECT_EQ_INT(tier.resident_count(), 2u, "resident count remains capped");
    EXPECT(tier.lookup(30) == nullptr, "oldest page evicted under pressure");
    EXPECT(tier.lookup(31) != nullptr, "page 31 still resident");
    EXPECT(tier.lookup(32) != nullptr, "new page resident");

    return fails;
}

static int test_host_tier_lookup_miss() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(16, 0x44);
    EXPECT(tier.lookup(99) == nullptr, "empty lookup returns nullptr");
    EXPECT(tier.lookup(-1) == nullptr, "negative lookup returns nullptr");
    EXPECT(tier.store(40, bytes.data(), bytes.size()), "store 40");
    EXPECT(tier.lookup(41) == nullptr, "different page lookup misses");

    return fails;
}

// ---------------------------------------------------------------------------
// compute_advise_ranges — MAD-232 posix_fadvise lookahead
// ---------------------------------------------------------------------------
//
// Pure-function test: catalog walk produces the right (fd, offset, size) set
// for [block_idx+1, block_idx+k]. No I/O, no GPU. Validates:
//   - Returns empty for k <= 0 or block_idx < 0
//   - Walks the requested window only (no off-by-one)
//   - Skips consolidated parents (they have size 0 by construction)
//   - Includes consolidated sub-experts (which DO have size)
//   - Preserves (fd_idx, offset, size) fidelity from catalog

static int test_compute_advise_ranges() {
    int fails = 0;
    wp::PageCatalog cat;

    // Build a representative catalog:
    //   blk.0: dense (3 MLP linears)
    //   blk.1: MoE consolidated (1 parent + 4 sub-experts each gate/up/down)
    //   blk.2: dense (3 MLP linears)
    cat.add("blk.0.ffn_gate.weight", /*file_idx=*/0,   1000,  4096);
    cat.add("blk.0.ffn_up.weight",   /*file_idx=*/0,   5096,  4096);
    cat.add("blk.0.ffn_down.weight", /*file_idx=*/0,   9192,  4096);
    cat.add_consolidated_experts("blk.1.ffn_gate_exps.weight", /*file_idx=*/0, 100000, 16384, /*n_experts=*/4);
    cat.add_consolidated_experts("blk.1.ffn_up_exps.weight",   /*file_idx=*/0, 120000, 16384, /*n_experts=*/4);
    cat.add_consolidated_experts("blk.1.ffn_down_exps.weight", /*file_idx=*/0, 140000, 16384, /*n_experts=*/4);
    cat.add("blk.2.ffn_gate.weight", /*file_idx=*/0, 200000,  4096);
    cat.add("blk.2.ffn_up.weight",   /*file_idx=*/0, 204096,  4096);
    cat.add("blk.2.ffn_down.weight", /*file_idx=*/0, 208192,  4096);

    // k=0 → empty
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/0, /*k=*/0);
        EXPECT_EQ_INT(r.size(), 0u, "k=0 returns empty");
    }
    // negative block_idx → empty
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/-1, /*k=*/2);
        EXPECT_EQ_INT(r.size(), 0u, "block_idx<0 returns empty");
    }
    // Block 0 → advise block 1 (k=1). Block 1 has 3 parents + 12 sub-experts.
    // Parents have size 0 (per add_consolidated_experts contract), so only the
    // 12 sub-experts make it through the size>0 filter.
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/0, /*k=*/1);
        EXPECT_EQ_INT(r.size(), 12u, "block 0, k=1: 12 sub-experts from block 1");
        // Each sub-expert size = consolidated/n_experts = 16384/4 = 4096
        for (const auto & rr : r) {
            EXPECT_EQ_INT(rr.size, 4096u, "sub-expert size correct");
            EXPECT_EQ_INT(rr.fd_idx, 0, "sub-expert fd_idx correct");
        }
    }
    // Block 0 → advise blocks 1+2 (k=2). 12 sub-experts + 3 dense linears = 15.
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/0, /*k=*/2);
        EXPECT_EQ_INT(r.size(), 15u, "block 0, k=2: 12+3");
    }
    // Block 1 → advise block 2 only (k=1). 3 dense.
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/1, /*k=*/1);
        EXPECT_EQ_INT(r.size(), 3u, "block 1, k=1: 3 dense from block 2");
        // Verify exact (offset, size) from catalog
        bool found_gate = false, found_up = false, found_down = false;
        for (const auto & rr : r) {
            EXPECT_EQ_INT(rr.size, 4096u, "dense linear size");
            if (rr.offset == 200000) found_gate = true;
            if (rr.offset == 204096) found_up   = true;
            if (rr.offset == 208192) found_down = true;
        }
        EXPECT(found_gate, "block 2 gate offset");
        EXPECT(found_up,   "block 2 up offset");
        EXPECT(found_down, "block 2 down offset");
    }
    // Block 2 → advise blocks 3+4 (k=2). Catalog has no blocks > 2, so empty.
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/2, /*k=*/2);
        EXPECT_EQ_INT(r.size(), 0u, "past end of catalog returns empty");
    }
    // Large k overshooting end is harmless — only available blocks are walked.
    {
        auto r = wp::compute_advise_ranges(cat, /*block_idx=*/0, /*k=*/100);
        EXPECT_EQ_INT(r.size(), 15u, "k=100 caps at available blocks");
    }
    return fails;
}

// ---------------------------------------------------------------------------
// FileIOLayer::submit_batch — MAD-235 batched io_uring submission
// ---------------------------------------------------------------------------
//
// Verifies the batch API delivers the same results as a sequence of singles:
//   - returns reqs.size() on full success
//   - completions arrive with the right req_ids and bytes
//   - bytes match the source file
//   - out-of-range fd_idx mid-batch aborts cleanly (returns the prefix count)
//
// We can't directly assert "one io_uring_submit syscall" from inside the
// process without strace; the semantic guarantees + the obvious single
// io_uring_submit call site in the override are enough for a unit test.

static int test_file_io_submit_batch() {
    int fails = 0;

    char path[] = "/tmp/wp-test-batch-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr, "  FAIL: %s: mkstemp failed: %s\n", __func__, std::strerror(errno));
        return 1;
    }
    constexpr size_t N = 8192;
    std::vector<uint8_t> pattern(N);
    for (size_t i = 0; i < N; ++i) pattern[i] = (uint8_t) ((i * 13 + 5) & 0xff);
    ssize_t w = write(fd, pattern.data(), N);
    EXPECT_EQ_INT((size_t) w, N, "wrote pattern");

    std::vector<int> fds = { fd };
    auto layer = wp::create_file_io(std::move(fds), /*prefer_async=*/false, 8);
    EXPECT(layer != nullptr, "create_file_io non-null");
    if (!layer) { unlink(path); return fails; }

    // Batch of 4 reads at distinct offsets.
    std::vector<uint8_t> dst[4] = {
        std::vector<uint8_t>(1024),
        std::vector<uint8_t>(1024),
        std::vector<uint8_t>(1024),
        std::vector<uint8_t>(1024),
    };
    std::vector<wp::FileIOBatchRequest> reqs = {
        { /*req_id=*/10, /*fd_idx=*/0, /*offset=*/   0, /*size=*/1024, dst[0].data() },
        { /*req_id=*/20, /*fd_idx=*/0, /*offset=*/2048, /*size=*/1024, dst[1].data() },
        { /*req_id=*/30, /*fd_idx=*/0, /*offset=*/4096, /*size=*/1024, dst[2].data() },
        { /*req_id=*/40, /*fd_idx=*/0, /*offset=*/6144, /*size=*/1024, dst[3].data() },
    };
    int n_ok = layer->submit_batch(reqs);
    EXPECT_EQ_INT(n_ok, 4, "all 4 queued");
    layer->flush();

    // Drain completions, dedup by req_id.
    std::vector<bool> seen(50, false);
    for (int i = 0; i < 20; ++i) {
        if (layer->pending() == 0) break;
        wp::IoResult r = layer->wait_any(/*timeout_ms=*/0);
        if (r.status == wp::IoStatus::Timeout) break;
        EXPECT(r.status == wp::IoStatus::Ok, "completion OK");
        EXPECT(r.req_id >= 10 && r.req_id <= 40, "req_id in expected set");
        if (r.req_id < seen.size()) seen[r.req_id] = true;
    }
    EXPECT(seen[10] && seen[20] && seen[30] && seen[40], "all 4 req_ids seen");

    // Verify content for each batch req.
    EXPECT(std::memcmp(dst[0].data(), pattern.data() +    0, 1024) == 0, "req 10 bytes");
    EXPECT(std::memcmp(dst[1].data(), pattern.data() + 2048, 1024) == 0, "req 20 bytes");
    EXPECT(std::memcmp(dst[2].data(), pattern.data() + 4096, 1024) == 0, "req 30 bytes");
    EXPECT(std::memcmp(dst[3].data(), pattern.data() + 6144, 1024) == 0, "req 40 bytes");

    layer.reset();
    unlink(path);
    return fails;
}

static int test_file_io_submit_batch_partial_failure() {
    // Mid-batch invalid fd_idx aborts cleanly at the prefix that succeeded.
    int fails = 0;
    char path[] = "/tmp/wp-test-batchp-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    constexpr size_t N = 4096;
    std::vector<uint8_t> pattern(N, 0xAB);
    write(fd, pattern.data(), N);

    std::vector<int> fds = { fd };
    auto layer = wp::create_file_io(std::move(fds), /*prefer_async=*/false, 4);
    if (!layer) { unlink(path); return fails; }

    std::vector<uint8_t> d0(512), d1(512), d2(512);
    std::vector<wp::FileIOBatchRequest> reqs = {
        { /*req_id=*/1, /*fd_idx=*/0,    0,  512, d0.data() },
        { /*req_id=*/2, /*fd_idx=*/99,  // BAD fd — batch stops here
                                          512,  512, d1.data() },
        { /*req_id=*/3, /*fd_idx=*/0, 1024,  512, d2.data() },  // never queued
    };
    int n_ok = layer->submit_batch(reqs);
    EXPECT_EQ_INT(n_ok, 1, "batch stopped at bad fd; only first queued");

    // Drain the single queued completion.
    layer->flush();
    int n_completed = 0;
    for (int i = 0; i < 5; ++i) {
        if (layer->pending() == 0) break;
        wp::IoResult r = layer->wait_any(/*timeout_ms=*/0);
        if (r.status == wp::IoStatus::Timeout) break;
        ++n_completed;
        EXPECT_EQ_INT(r.req_id, 1, "only req 1 completes");
    }
    EXPECT_EQ_INT(n_completed, 1, "exactly one completion");

    layer.reset();
    unlink(path);
    return fails;
}

// ---------------------------------------------------------------------------
// Completion demux — targeted waits must never drop a sibling's completion
// ---------------------------------------------------------------------------
//
// Regression guard for the shared-ring cross-drain bug: when several logical
// consumers (prefetch scheduler + synchronous pager page-ins + ensure_batch)
// submit reads on ONE FileIOLayer, a caller that waits for its OWN req_id may
// reap a DIFFERENT consumer's completion first. The old code discarded that
// foreign completion (io_uring_cqe_seen without routing it), permanently
// losing it — the owner's slot then hung forever and the prefetch pool leaked
// slots until the pipeline stalled (2x decode regression / depth-8 load hang).
//
// The demux contract fixes this: wait_for_req(id) reaps and BUFFERS any
// foreign completion it encounters, so a later wait_for_req/try_take for that
// id still finds it. This test drives three reads and claims them strictly
// out of submit order, asserting none are lost.
static int test_file_io_demux_no_cross_drain() {
    int fails = 0;
    char path[] = "/tmp/wp-test-demux-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr, "  FAIL: %s: mkstemp failed: %s\n", __func__, std::strerror(errno));
        return 1;
    }
    constexpr size_t N = 8192;
    std::vector<uint8_t> pattern(N);
    for (size_t i = 0; i < N; ++i) pattern[i] = (uint8_t) ((i * 17 + 3) & 0xff);
    ssize_t w = write(fd, pattern.data(), N);
    EXPECT_EQ_INT((size_t) w, N, "wrote pattern");

    std::vector<int> fds = { fd };
    auto layer = wp::create_file_io(std::move(fds), /*prefer_async=*/true, 8);
    EXPECT(layer != nullptr, "create_file_io non-null");
    if (!layer) { unlink(path); return fails; }

    // Three in-flight reads with distinct req_ids, distinct offsets.
    std::vector<uint8_t> d1(1024), d2(2048), d3(512);
    EXPECT(layer->submit(/*req=*/100, 0,    0, 1024, d1.data()), "submit 100");
    EXPECT(layer->submit(/*req=*/200, 0, 1024, 2048, d2.data()), "submit 200");
    EXPECT(layer->submit(/*req=*/300, 0, 4096,  512, d3.data()), "submit 300");
    layer->flush();

    // Claim strictly OUT of submit order. Each targeted wait must return its
    // own completion; siblings reaped along the way must NOT be lost.
    wp::IoResult r3 = layer->wait_for_req(300, /*timeout_ms=*/-1);
    EXPECT(r3.status == wp::IoStatus::Ok, "req 300 status Ok");
    EXPECT_EQ_INT(r3.req_id, 300, "req 300 round-trips");
    EXPECT_EQ_INT(r3.bytes_read, 512, "req 300 bytes");

    wp::IoResult r1 = layer->wait_for_req(100, /*timeout_ms=*/-1);
    EXPECT(r1.status == wp::IoStatus::Ok, "req 100 status Ok (not lost by 300's wait)");
    EXPECT_EQ_INT(r1.req_id, 100, "req 100 round-trips");
    EXPECT_EQ_INT(r1.bytes_read, 1024, "req 100 bytes");

    // The remaining one is claimable non-blocking via try_take.
    wp::IoResult r2{};
    bool took2 = layer->try_take(200, r2);
    EXPECT(took2, "try_take 200 succeeds");
    EXPECT_EQ_INT(r2.req_id, 200, "req 200 round-trips");
    EXPECT_EQ_INT(r2.bytes_read, 2048, "req 200 bytes");

    // Unknown / already-claimed ids are a clean miss, never a hang.
    wp::IoResult miss{};
    EXPECT(!layer->try_take(999, miss), "try_take unknown id -> false");
    EXPECT(!layer->try_take(300, miss), "try_take already-claimed id -> false");

    // Content integrity: every buffered read landed in the right dst.
    EXPECT(std::memcmp(d1.data(), pattern.data() +    0, 1024) == 0, "d1 content");
    EXPECT(std::memcmp(d2.data(), pattern.data() + 1024, 2048) == 0, "d2 content");
    EXPECT(std::memcmp(d3.data(), pattern.data() + 4096,  512) == 0, "d3 content");

    // Nothing left outstanding.
    EXPECT_EQ_INT(layer->pending(), 0, "no pending after all claimed");

    layer.reset();
    unlink(path);
    return fails;
}

// ---------------------------------------------------------------------------
// FileIOLayer::advise_prefetch — MAD-232 integration with a real fd
// ---------------------------------------------------------------------------
//
// Verifies posix_fadvise(WILLNEED) doesn't trash the fd: bytes remain
// readable after advise (which would catch a bad off/size getting clamped
// into an unreadable state). Out-of-range fd_idx is a silent no-op (matches
// the safe default for all "hint" APIs — never break correctness on bad
// inputs). On non-Linux builds the impl is a no-op and we just verify
// reads still work afterward.

static int test_file_io_advise_prefetch() {
    int fails = 0;

    char path[] = "/tmp/wp-test-advise-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr, "  FAIL: %s: mkstemp failed: %s\n", __func__, std::strerror(errno));
        return 1;
    }
    constexpr size_t N = 8192;
    std::vector<uint8_t> pattern(N);
    for (size_t i = 0; i < N; ++i) pattern[i] = (uint8_t) ((i * 31 + 7) & 0xff);
    ssize_t w = write(fd, pattern.data(), N);
    EXPECT_EQ_INT((size_t) w, N, "wrote pattern");

    std::vector<int> fds = { fd };
    auto layer = wp::create_file_io(std::move(fds), /*prefer_async=*/false, 4);
    EXPECT(layer != nullptr, "create_file_io returns non-null");
    if (!layer) { unlink(path); return fails; }

    // Valid advise — should not crash, should not affect subsequent reads.
    layer->advise_prefetch(/*fd_idx=*/0, /*offset=*/0,    /*size=*/4096);
    layer->advise_prefetch(/*fd_idx=*/0, /*offset=*/4096, /*size=*/4096);

    // Out-of-range fd_idx is a silent no-op.
    layer->advise_prefetch(/*fd_idx=*/99, /*offset=*/0,   /*size=*/4096);
    layer->advise_prefetch(/*fd_idx=*/-1, /*offset=*/0,   /*size=*/4096);

    // size=0 is a no-op (advising an empty range is meaningless; some kernels
    // treat it as "whole file" via the same syscall, which we don't want).
    layer->advise_prefetch(/*fd_idx=*/0, /*offset=*/0, /*size=*/0);

    // Bytes remain readable after advise.
    std::vector<uint8_t> dst(N);
    bool ok = layer->submit(/*req=*/1, /*fd_idx=*/0, 0, N, dst.data());
    EXPECT(ok, "submit after advise");
    wp::IoResult r = layer->wait_any(/*timeout_ms=*/0);
    EXPECT(r.status == wp::IoStatus::Ok, "read OK after advise");
    EXPECT_EQ_INT(r.bytes_read, (int) N, "full bytes after advise");
    EXPECT(std::memcmp(dst.data(), pattern.data(), N) == 0, "content matches after advise");

    layer.reset();
    unlink(path);
    return fails;
}

// ---------------------------------------------------------------------------
// MAD-234 — UMA detection + MemAvailable parse
// ---------------------------------------------------------------------------
//
// Pure-function tests for is_uma_archname() (no HIP needed) and a runtime
// sanity check on read_mem_available_bytes(). is_uma_device() needs a real
// HIP device so it's only smoke-tested under GGML_USE_HIP — and even then
// it can't fail-fast because the actual UMA-vs-discrete state depends on
// the host machine. We log what it sees and move on.

static int test_is_uma_archname() {
    int fails = 0;

    // Known UMA prefixes — match by prefix (HIP appends sram/xnack suffixes).
    EXPECT(wp::is_uma_archname("gfx1151"),                    "gfx1151 (Strix Halo)");
    EXPECT(wp::is_uma_archname("gfx1150:sramecc+:xnack-"),    "gfx1150 with suffix");
    EXPECT(wp::is_uma_archname("gfx1152"),                    "gfx1152");
    EXPECT(wp::is_uma_archname("gfx1103"),                    "gfx1103 (Phoenix)");
    EXPECT(wp::is_uma_archname("gfx1103:sramecc-:xnack-"),    "gfx1103 with suffix");
    EXPECT(wp::is_uma_archname("gfx90c"),                     "gfx90c (Renoir/Cezanne)");
    EXPECT(wp::is_uma_archname("gfx940"),                     "gfx940 (MI300A)");

    // Discrete GPUs (RDNA2/3/4 desktop) — must be reported as non-UMA.
    EXPECT(!wp::is_uma_archname("gfx1030"),                   "gfx1030 (6900 XT) NOT UMA");
    EXPECT(!wp::is_uma_archname("gfx1031"),                   "gfx1031 (6800)    NOT UMA");
    EXPECT(!wp::is_uma_archname("gfx1101"),                   "gfx1101 (7800)    NOT UMA");
    EXPECT(!wp::is_uma_archname("gfx1200"),                   "gfx1200 (9070)    NOT UMA");
    EXPECT(!wp::is_uma_archname("gfx1201"),                   "gfx1201 (R9700)   NOT UMA");
    EXPECT(!wp::is_uma_archname("gfx906"),                    "gfx906 (MI50)     NOT UMA");
    EXPECT(!wp::is_uma_archname("gfx908"),                    "gfx908 (MI100)    NOT UMA");

    // Edge cases.
    EXPECT(!wp::is_uma_archname(nullptr),                     "nullptr returns false");
    EXPECT(!wp::is_uma_archname(""),                          "empty string returns false");
    EXPECT(!wp::is_uma_archname("gfx"),                       "too-short prefix returns false");
    EXPECT(!wp::is_uma_archname("not-a-gfx-string"),          "non-gfx prefix returns false");

    return fails;
}

static int test_read_mem_available_bytes() {
    int fails = 0;
    const size_t mem = wp::read_mem_available_bytes();
#if defined(__linux__)
    // On Linux /proc/meminfo always reports MemAvailable. Anything > 16 MiB
    // is plausible on any machine that can run this test binary.
    EXPECT(mem > 16ULL * 1024 * 1024, "Linux: MemAvailable > 16 MiB");
    std::fprintf(stderr, "  INFO: MemAvailable = %.2f GiB\n",
                 (double) mem / (1024.0 * 1024.0 * 1024.0));
#else
    // Non-Linux: helper returns 0 by contract.
    EXPECT_EQ_INT(mem, 0u, "non-Linux: MemAvailable returns 0");
#endif
    return fails;
}

static int test_is_uma_device_smoke() {
    // Cannot assert true/false without knowing the host's GPU topology.
    // The point of this smoke test is to make sure is_uma_device(0) doesn't
    // crash and returns SOMETHING when HIP is available. On non-HIP builds
    // it must return false.
    int fails = 0;
    const bool result = wp::is_uma_device(0);
    std::fprintf(stderr, "  INFO: is_uma_device(0) = %s (informational; "
                          "expected true on Strix Halo / Phoenix, false on dGPU)\n",
                 result ? "true" : "false");

    // Negative device_idx is always false (sentinel for "skip the check").
    EXPECT(!wp::is_uma_device(-1), "is_uma_device(-1) returns false");
    EXPECT(!wp::is_uma_device(-99), "is_uma_device(very negative) returns false");
    return fails;
}

static int test_routing_boundary_prepass() {
    int fails = 0;
    struct ggml_init_params ip = { /*.mem_size=*/ 16*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * ids_producer = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
    ggml_set_name(ids_producer, "ids_producer");
    struct ggml_tensor * ids_view = ggml_view_1d(ctx, ids_producer, 8, 0);
    struct ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 2);
    struct ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    struct ggml_tensor * mmid = ggml_mul_mat_id(ctx, as, b, ids_view);
    ggml_set_name(mmid, "mmid");
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mmid);
    wp::WeightPager pager;
    pager.mark_routing_boundaries(gf);
    if (!pager.is_routing_break(mmid))         { fprintf(stderr, "FAIL: mmid not marked\n"); fails++; }
    if (!pager.is_routing_break(ids_producer)) { fprintf(stderr, "FAIL: ids producer (view root) not marked\n"); fails++; }
    if (pager.is_routing_break(b))             { fprintf(stderr, "FAIL: unrelated tensor marked\n"); fails++; }
    ggml_free(ctx);
    return fails;
}

static int test_wp_paged_batch_flag_default_off() {
    int fails = 0;
    ScopedEnv guard("WP_PAGED_BATCH");
    unsetenv("WP_PAGED_BATCH");
    if (wp::wp_paged_batch_enabled()) { fprintf(stderr, "FAIL: WP_PAGED_BATCH must default OFF\n"); fails++; }
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
        { "page_catalog",                test_page_catalog                },
        { "page_catalog_moe_classify",   test_page_catalog_moe_classification },
        { "page_catalog_consolidated",   test_page_catalog_consolidated_split },
        { "dup_clear_o_direct", test_dup_clear_o_direct },
        { "file_io_sync_pread", test_file_io_sync_pread },
        { "file_io_advise_prefetch",  test_file_io_advise_prefetch  },
        { "file_io_submit_batch",            test_file_io_submit_batch            },
        { "file_io_submit_batch_partial",    test_file_io_submit_batch_partial_failure },
        { "file_io_demux_no_cross_drain",    test_file_io_demux_no_cross_drain    },
        { "compute_advise_ranges",    test_compute_advise_ranges    },
        { "is_uma_archname",          test_is_uma_archname          },
        { "read_mem_available_bytes", test_read_mem_available_bytes },
        { "is_uma_device_smoke",      test_is_uma_device_smoke      },
        { "pool_allocator",     test_pool_allocator     },
        { "pool_size_class_packs_small_pages", test_pool_size_class_packs_small_pages },
        { "pool_size_class_pin_skip",          test_pool_size_class_pin_skip          },
        { "pool_pin_basic",                       test_pool_pin_basic                       },
        { "pool_pin_refcount",                    test_pool_pin_refcount                    },
        { "pool_pin_oob_safe",                    test_pool_pin_oob_safe                    },
        { "pool_alloc_skips_pinned_in_eviction",  test_pool_alloc_skips_pinned_in_eviction  },
        { "pool_alloc_returns_neg1_when_all_pinned", test_pool_alloc_returns_neg1_when_all_pinned },
        { "host_tier_store_lookup",              test_host_tier_store_lookup              },
        { "host_tier_size_class_reuse",          test_host_tier_size_class_reuse          },
        { "host_tier_lru_eviction_order",        test_host_tier_lru_eviction_order        },
        { "host_tier_lookup_touch_keeps_mru",    test_host_tier_lookup_touch_keeps_mru    },
        { "host_tier_over_budget_evict",         test_host_tier_over_budget_evict         },
        { "host_tier_lookup_miss",               test_host_tier_lookup_miss               },
        { "catalog_add_pinned_basic",            test_catalog_add_pinned_basic            },
        { "catalog_add_pinned_mixed_with_paged", test_catalog_add_pinned_mixed_with_paged },
        { "catalog_clear_resets_pinned",         test_catalog_clear_resets_pinned_counters },
        { "pool_hit_count_basic",                test_pool_hit_count_basic                },
        { "pool_hot_threshold_protects",         test_pool_hot_threshold_protects_in_eviction },
        { "pool_hot_fallback_when_all_hot",      test_pool_hot_fallback_when_all_hot      },
        { "pool_default_threshold_zero_lru",     test_pool_default_threshold_zero_is_pure_lru },
        { "routing_boundary_prepass",            test_routing_boundary_prepass            },
        { "wp_paged_batch_flag_default_off",     test_wp_paged_batch_flag_default_off     },
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
