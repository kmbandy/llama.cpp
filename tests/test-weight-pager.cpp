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
