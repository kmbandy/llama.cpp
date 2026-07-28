// Unit tests for the wp::* modules under src/weight-pager/.
//
// Lightweight, no test framework. Each TEST_FN runs subtests and returns
// the number of failures. main() sums them and exits non-zero if any
// failed. Tests that require GPU (hip*) are gated on GGML_USE_HIP at
// runtime — they no-op compile-out under non-HIP builds.

#include "weight-pager/wp-page-catalog.h"
#include "weight-pager/wp-eval-cb.h"
#include "weight-pager/wp-file-io.h"
#include "weight-pager/wp-gpu-transport.h"
#include "weight-pager/wp-host-tier.h"
#include "weight-pager/wp-host-prefetch.h"
#include "weight-pager/wp-pager.h"   // compute_advise_ranges / AdviseRange
#include "weight-pager/wp-pool.h"
#include "weight-pager/wp-prefetch.h"
#include "weight-pager/wp-router.h"
#include "weight-pager/wp-router-predictor.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <thread>
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

class TransportErrorReqIdZeroFileIO : public wp::FileIOLayer {
public:
    bool submit(uint64_t req_id, int /*fd_idx*/, uint64_t /*offset*/,
                size_t size, void * dst) override {
        if (req_id == 0 || size == 0 || dst == nullptr || pending_ != 0) {
            return false;
        }
        pending_ = 1;
        return true;
    }

    void flush() override {}
    int pending() const override { return pending_; }
    int fd(int /*fd_idx*/) const override { return -1; }
    wp::FileIOTransport transport() const override { return wp::FileIOTransport::SyncPread; }

protected:
    bool reap_raw_(int timeout_ms, wp::IoResult & out) override {
        if (timeout_ms == 0 || emitted_) {
            return false;
        }
        emitted_ = true;
        pending_ = 0;
        out = wp::IoResult{};
        out.req_id = 0;
        out.status = wp::IoStatus::ErrorIo;
        out.bytes_read = -EIO;
        return true;
    }

private:
    int pending_ = 0;
    bool emitted_ = false;
};

static int test_p2p_tunable_resolution() {
    int fails = 0;
    ScopedEnv queue_guard("WP_P2P_QUEUE_DEPTH");
    ScopedEnv window_guard("WP_P2P_WINDOW_CACHE_MAX");
    ScopedEnv tier_guard("WP_P2P_DIRECT_TO_DEVICE");

    unsetenv("WP_P2P_QUEUE_DEPTH");
    unsetenv("WP_P2P_WINDOW_CACHE_MAX");
    unsetenv("WP_P2P_DIRECT_TO_DEVICE");
    EXPECT_EQ_INT(wp::resolve_p2p_queue_depth(16), 16, "queue default preserves configured depth");
    // Default clamp widened from [64,256] to [256,1024] on 2026-07-27: the old
    // floor starved the P2P window cache under batch-width pressure (measured
    // 2.13x prefill from raising it). 4*16=64 therefore clamps UP to 256.
    EXPECT_EQ_INT(wp::resolve_p2p_window_cache_max(16), 256, "window default preserves clamp(4*QD,256,1024)");
    EXPECT(!wp::p2p_direct_to_device_with_tier(), "tier-direct default preserves staging/store behavior");

    setenv("WP_P2P_QUEUE_DEPTH", "32", 1);
    setenv("WP_P2P_WINDOW_CACHE_MAX", "99", 1);
    setenv("WP_P2P_DIRECT_TO_DEVICE", "1", 1);
    EXPECT_EQ_INT(wp::resolve_p2p_queue_depth(16), 32, "queue env override");
    EXPECT_EQ_INT(wp::resolve_p2p_window_cache_max(16), 99, "independent window env override");
    EXPECT(wp::p2p_direct_to_device_with_tier(), "tier-direct explicit opt-in");

    setenv("WP_P2P_QUEUE_DEPTH", "0", 1);
    setenv("WP_P2P_WINDOW_CACHE_MAX", "99999", 1);
    EXPECT_EQ_INT(wp::resolve_p2p_queue_depth(16), 1, "queue lower clamp");
    EXPECT_EQ_INT(wp::resolve_p2p_window_cache_max(16), 4096, "window upper clamp");
    return fails;
}

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
// PrefetchScheduler
// ---------------------------------------------------------------------------

static int test_prefetch_wait_transport_error_req_id_zero() {
    int fails = 0;

    TransportErrorReqIdZeroFileIO file_io;
    wp::GpuTransport gpu;
    wp::PrefetchScheduler prefetch;
    std::vector<uint8_t> dst(64, 0);

    EXPECT(prefetch.init(&file_io, &gpu, /*max_page_size=*/64, /*queue_depth=*/1),
           "prefetch init");
    EXPECT(prefetch.submit(/*page_idx=*/7, /*fd_idx=*/0, /*file_offset=*/0,
                           /*payload_size=*/32, dst.data(), /*slot_size=*/64),
           "prefetch submit");

    bool ok = prefetch.wait_for(/*page_idx=*/7, /*timeout_ms=*/-1);
    EXPECT(!ok, "transport ErrorIo req_id 0 fails waited slot");

    prefetch.reap(/*page_idx=*/7);
    EXPECT_EQ_INT(prefetch.pending(), 0, "failed slot can be reaped");

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

static int test_pool_allocator_two_pools_independent() {
    int fails = 0;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    EXPECT(buft != nullptr, "cpu buffer_type available");
    if (!buft) return fails;

    wp::PoolAllocator pool_a;
    wp::PoolAllocator pool_b;
    EXPECT(pool_a.init(buft, /*n_slots=*/2, /*slot_size=*/128), "pool A init");
    EXPECT(pool_b.init(buft, /*n_slots=*/3, /*slot_size=*/256), "pool B init");

    int evict_a = -1;
    int evict_b = -1;
    pool_a.set_eviction_callback([&](int slot) { evict_a = slot; });
    pool_b.set_eviction_callback([&](int slot) { evict_b = slot; });

    EXPECT_EQ_INT(pool_a.alloc_slot(), 0, "pool A first slot");
    EXPECT_EQ_INT(pool_a.alloc_slot(), 1, "pool A second slot");
    EXPECT_EQ_INT(pool_b.alloc_slot(), 0, "pool B first slot");
    EXPECT_EQ_INT(pool_b.alloc_slot(), 1, "pool B second slot");
    EXPECT_EQ_INT(pool_b.alloc_slot(), 2, "pool B third slot");

    pool_a.pin_slot(0);
    EXPECT_EQ_INT(pool_a.alloc_slot(), 1, "pool A evicts only its unpinned slot");
    EXPECT_EQ_INT(evict_a, 1, "pool A eviction callback");
    EXPECT_EQ_INT(evict_b, -1, "pool B not evicted by pool A pressure");
    EXPECT(pool_a.is_pinned(0), "pool A pin remains local");
    EXPECT(!pool_b.is_pinned(0), "pool B pin state remains independent");

    EXPECT_EQ_INT(pool_b.alloc_slot(), 0, "pool B evicts its own LRU slot");
    EXPECT_EQ_INT(evict_b, 0, "pool B eviction callback");
    pool_a.unpin_slot(0);
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
    std::vector<uint8_t> out(src.size());
    EXPECT(tier.lookup(7, out.data(), out.size()), "lookup copies page");
    EXPECT(std::memcmp(out.data(), src.data(), src.size()) == 0, "lookup bytes match");
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
    EXPECT(tier.store(2, b.data(), b.size()), "store page 2");
    EXPECT(tier.store(3, c.data(), c.size()), "store page 3 evicts page 1");
    std::vector<uint8_t> out(c.size());

    EXPECT(tier.lookup(3, out.data(), out.size()), "lookup reused page");
    EXPECT(!tier.lookup(1, out.data(), out.size()), "evicted page 1 missing");
    EXPECT(std::memcmp(out.data(), c.data(), c.size()) == 0, "reused slot has new bytes");

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

    std::vector<uint8_t> out(bytes.size());
    EXPECT(!tier.lookup(10, out.data(), out.size()), "oldest page evicted first");
    EXPECT(tier.lookup(11, out.data(), out.size()), "page 11 still resident");
    EXPECT(tier.lookup(12, out.data(), out.size()), "page 12 still resident");
    EXPECT(tier.lookup(13, out.data(), out.size()), "new page resident");
    EXPECT_EQ_INT(tier.resident_count(), 3u, "resident count stays at capacity");

    return fails;
}

// HostTier speculative sub-tier (WP_HOST_SPEC_TIER): prefetch and eviction
// sharing the RAM tier without prefetch degrading it.
//
// The invariant under test is the one that matters operationally: a
// MISPREDICTED PREFETCH MUST NEVER COST A VICTIM PAGE. On one flat LRU the
// prediction is the most-recently-stored entry, so it outranks the victim and
// the victim dies -- prefetch actively making the tier worse. This is the same
// shape as VRAM gate 3, one tier down.
static int test_host_tier_speculative_evicts_before_victim() {
    int fails = 0;
    std::vector<uint8_t> bytes(32, 0x11);
    std::vector<uint8_t> out(bytes.size());

    // --- Gate ON: the speculative entry dies, the victim survives. ---
    {
        wp::HostTier tier;
        EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "spec tier init");
        tier.set_speculative_tier(true);

        // Victim page 10 is the OLDEST entry -- pure LRU would take it first.
        EXPECT(tier.store(10, bytes.data(), bytes.size(), /*speculative=*/false), "victim 10");
        EXPECT(tier.store(11, bytes.data(), bytes.size(), /*speculative=*/true),  "prediction 11");
        EXPECT(tier.store(12, bytes.data(), bytes.size(), /*speculative=*/true),  "prediction 12");
        EXPECT_EQ_INT((int) tier.speculative_count(), 2, "two predictions resident");

        // Capacity is 3; this store must evict. Pure LRU picks 10 (oldest);
        // speculative-first must pick 11 (LRU *among predictions*).
        EXPECT(tier.store(13, bytes.data(), bytes.size(), /*speculative=*/false), "victim 13 forces evict");
        EXPECT(tier.lookup(10, out.data(), out.size()),
               "VICTIM SURVIVES even though it is the oldest entry");
        EXPECT(!tier.lookup(11, out.data(), out.size()),
               "LRU prediction evicted instead of the victim");
        EXPECT_EQ_INT((int) tier.speculative_evicted_unused(), 1, "one prediction wasted");
    }

    // --- Promotion: landing is not use; only a demand hit confirms. ---
    {
        wp::HostTier tier;
        EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "promote tier init");
        tier.set_speculative_tier(true);

        EXPECT(tier.store(20, bytes.data(), bytes.size(), /*speculative=*/true), "prediction 20");
        EXPECT_EQ_INT((int) tier.speculative_count(), 1, "still speculative after landing");
        EXPECT(tier.lookup(20, out.data(), out.size()), "demand hit on 20");
        EXPECT_EQ_INT((int) tier.speculative_count(), 0, "demand hit promotes");
        EXPECT_EQ_INT((int) tier.speculative_promotions(), 1, "promotion counted");

        // Now confirmed, 20 must be protected exactly like any victim page.
        EXPECT(tier.store(21, bytes.data(), bytes.size(), /*speculative=*/true), "prediction 21");
        EXPECT(tier.store(22, bytes.data(), bytes.size(), /*speculative=*/true), "prediction 22");
        EXPECT(tier.store(23, bytes.data(), bytes.size(), /*speculative=*/false), "forces evict");
        EXPECT(tier.lookup(20, out.data(), out.size()),
               "promoted page protected like a victim, despite being oldest");
    }

    // --- Gate OFF: unchanged flat-LRU behaviour (the victim dies). ---
    {
        wp::HostTier tier;
        EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "default tier init");
        // set_speculative_tier NOT called.
        EXPECT(tier.store(30, bytes.data(), bytes.size(), /*speculative=*/false), "victim 30");
        EXPECT(tier.store(31, bytes.data(), bytes.size(), /*speculative=*/true),  "prediction 31");
        EXPECT(tier.store(32, bytes.data(), bytes.size(), /*speculative=*/true),  "prediction 32");
        EXPECT(tier.store(33, bytes.data(), bytes.size(), /*speculative=*/false), "forces evict");
        EXPECT(!tier.lookup(30, out.data(), out.size()),
               "gate off: oldest victim still dies first (documents what the gate changes)");
    }

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

    std::vector<uint8_t> out(bytes.size());
    EXPECT(tier.lookup(20, out.data(), out.size()), "touch page 20");
    EXPECT(tier.store(23, bytes.data(), bytes.size()), "store 23 evicts LRU after touch");

    EXPECT(tier.lookup(20, out.data(), out.size()), "touched page kept as MRU");
    EXPECT(!tier.lookup(21, out.data(), out.size()), "untouched oldest page evicted");
    EXPECT(tier.lookup(22, out.data(), out.size()), "page 22 still resident");
    EXPECT(tier.lookup(23, out.data(), out.size()), "page 23 resident");

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
    std::vector<uint8_t> out(bytes.size());
    EXPECT(!tier.lookup(30, out.data(), out.size()), "oldest page evicted under pressure");
    EXPECT(tier.lookup(31, out.data(), out.size()), "page 31 still resident");
    EXPECT(tier.lookup(32, out.data(), out.size()), "new page resident");

    return fails;
}

static int test_host_tier_lookup_miss() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(16, 0x44);
    std::vector<uint8_t> out(bytes.size());
    EXPECT(!tier.lookup(99, out.data(), out.size()), "empty lookup misses");
    EXPECT(!tier.lookup(-1, out.data(), out.size()), "negative lookup misses");
    EXPECT(tier.store(40, bytes.data(), bytes.size()), "store 40");
    EXPECT(!tier.lookup(41, out.data(), out.size()), "different page lookup misses");

    return fails;
}

static int test_host_tier_concurrency() {
    int fails = 0;
    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/512, /*device_idx=*/-1), "host tier init");

    constexpr int n_threads = 4;
    constexpr int n_pages = 8;
    constexpr int n_iters = 25000;
    constexpr size_t page_size = 32;
    std::atomic<int> bad_lookups{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&tier, &bad_lookups, t]() {
            std::vector<uint8_t> expected(page_size);
            std::vector<uint8_t> out(page_size);
            for (int i = 0; i < n_iters; ++i) {
                const int page = (i + t) % n_pages;
                std::fill(expected.begin(), expected.end(), (uint8_t) page);
                if ((i % 7) == 0) {
                    tier.erase(page);
                } else {
                    tier.store(page, expected.data(), expected.size());
                }
                if (tier.lookup(page, out.data(), out.size()) &&
                    std::memcmp(out.data(), expected.data(), out.size()) != 0) {
                    ++bad_lookups;
                }
            }
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

    EXPECT_EQ_INT(bad_lookups.load(), 0, "lookup bytes remain page-consistent");
    EXPECT(tier.used_bytes() <= tier.budget_bytes(), "used bytes stay within budget");
    EXPECT(tier.resident_count() <= n_pages, "resident pages stay within page set");
    return fails;
}

static int test_host_tier_repeated_touches_eviction_order() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/128, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x55);
    EXPECT(tier.store(50, bytes.data(), bytes.size()), "store 50");
    EXPECT(tier.store(51, bytes.data(), bytes.size()), "store 51");
    EXPECT(tier.store(52, bytes.data(), bytes.size()), "store 52");
    EXPECT(tier.store(53, bytes.data(), bytes.size()), "store 53");

    std::vector<uint8_t> out(bytes.size());

    // Repeatedly touch 50 and 51 so 52 and 53 remain the least-recently-used,
    // in that order, regardless of how many times the MRU pages are re-touched.
    for (int i = 0; i < 5; ++i) {
        EXPECT(tier.lookup(51, out.data(), out.size()), "repeated touch of 51");
        EXPECT(tier.lookup(50, out.data(), out.size()), "repeated touch of 50");
    }

    // Budget for 4 pages of 32 bytes; storing a 5th must evict 52 (now LRU).
    EXPECT(tier.store(54, bytes.data(), bytes.size()), "store 54 evicts current LRU (52)");
    EXPECT(!tier.lookup(52, out.data(), out.size()), "52 evicted first despite earlier insertion order");
    EXPECT(tier.lookup(53, out.data(), out.size()), "53 still resident");
    EXPECT(tier.lookup(50, out.data(), out.size()), "50 still resident (touched)");
    EXPECT(tier.lookup(51, out.data(), out.size()), "51 still resident (touched)");
    EXPECT(tier.lookup(54, out.data(), out.size()), "54 resident");

    // Next eviction should now take 53, the next-oldest untouched page.
    EXPECT(tier.store(55, bytes.data(), bytes.size()), "store 55 evicts next LRU (53)");
    EXPECT(!tier.lookup(53, out.data(), out.size()), "53 evicted second");
    EXPECT(tier.lookup(55, out.data(), out.size()), "55 resident");

    return fails;
}

static int test_host_tier_erase_middle_preserves_order() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/128, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x66);
    EXPECT(tier.store(60, bytes.data(), bytes.size()), "store 60");
    EXPECT(tier.store(61, bytes.data(), bytes.size()), "store 61");
    EXPECT(tier.store(62, bytes.data(), bytes.size()), "store 62");
    EXPECT(tier.store(63, bytes.data(), bytes.size()), "store 63");

    // Erase the middle element (61); the relative recency order of the
    // remaining pages (60, 62, 63) must be unaffected.
    tier.erase(61);
    EXPECT_EQ_INT(tier.resident_count(), 3u, "erase drops resident count by one");

    std::vector<uint8_t> out(bytes.size());
    EXPECT(!tier.lookup(61, out.data(), out.size()), "erased page gone");

    // Budget holds 4 pages; after freeing 61's slot there is room for two more
    // stores before an eviction is forced, and the LRU order among (60,62,63)
    // must still be 60 first.
    EXPECT(tier.store(64, bytes.data(), bytes.size()), "store 64 into freed slot");
    EXPECT(tier.store(65, bytes.data(), bytes.size()), "store 65 forces eviction of oldest (60)");
    EXPECT(!tier.lookup(60, out.data(), out.size()), "60 was oldest remaining and is evicted first");
    EXPECT(tier.lookup(62, out.data(), out.size()), "62 still resident");
    EXPECT(tier.lookup(63, out.data(), out.size()), "63 still resident");
    EXPECT(tier.lookup(64, out.data(), out.size()), "64 still resident");
    EXPECT(tier.lookup(65, out.data(), out.size()), "65 still resident");

    return fails;
}

static int test_host_tier_touch_absent_is_noop() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x77);
    EXPECT(tier.store(70, bytes.data(), bytes.size()), "store 70");
    EXPECT(tier.store(71, bytes.data(), bytes.size()), "store 71");
    EXPECT(tier.store(72, bytes.data(), bytes.size()), "store 72");

    std::vector<uint8_t> out(bytes.size());

    // Looking up (and thus attempting to touch) a page that was never stored,
    // and erasing a page that was never stored, must not disturb existing
    // residency, byte accounting, or LRU order.
    EXPECT(!tier.lookup(999, out.data(), out.size()), "lookup of absent page misses");
    tier.erase(998);
    EXPECT_EQ_INT(tier.resident_count(), 3u, "resident count unaffected by no-op touch/erase");
    EXPECT_EQ_INT(tier.used_bytes(), 96u, "used bytes unaffected by no-op touch/erase");

    // LRU order should still be 70, 71, 72 (oldest first) since the no-op
    // lookups/erases above must not have touched anything.
    EXPECT(tier.store(73, bytes.data(), bytes.size()), "store 73 evicts true LRU (70)");
    EXPECT(!tier.lookup(70, out.data(), out.size()), "70 evicted as expected, unaffected by no-op calls");
    EXPECT(tier.lookup(71, out.data(), out.size()), "71 still resident");
    EXPECT(tier.lookup(72, out.data(), out.size()), "72 still resident");
    EXPECT(tier.lookup(73, out.data(), out.size()), "73 resident");

    return fails;
}

// Regression test for the O(n) std::find LRU scan (measured ~2.95s for
// ~20000 lookups against ~1880 resident pages in production). Reproduces
// the same shape with small synthetic pages so the cost measured here is
// the bookkeeping (touch/erase) itself, not the memcpy. Lookups are issued
// in a fixed pseudo-random order (not insertion order) so each touch must
// actually locate an arbitrary page in the recency list rather than always
// finding the next-oldest page sitting at the front — a purely sequential
// access pattern would let an O(n) std::find degenerate to O(1) per call
// and hide the defect. An O(1) LRU (list + index map) finishes this well
// within the deadline; the O(n) std::find-based version does not.
static int test_host_tier_lru_touch_is_not_linear_scan() {
    int fails = 0;

    constexpr int n_pages = 200000;
    constexpr size_t page_bytes = 16;
    constexpr int n_iters = 8000;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/(size_t) n_pages * page_bytes, /*device_idx=*/-1),
           "host tier init");

    std::vector<uint8_t> bytes(page_bytes, 0x99);
    for (int i = 0; i < n_pages; ++i) {
        EXPECT(tier.store(i, bytes.data(), bytes.size()), "prime resident set");
    }
    EXPECT_EQ_INT(tier.resident_count(), (size_t) n_pages, "all pages resident before timing");

    // Deterministic pseudo-random page order (xorshift32), independent of
    // insertion order, so lookups land throughout the recency list instead
    // of only ever hitting whichever page is currently at the front.
    std::vector<int> order(n_iters);
    uint32_t rng_state = 0x9e3779b9u;
    for (int i = 0; i < n_iters; ++i) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        order[i] = (int) (rng_state % (uint32_t) n_pages);
    }

    std::vector<uint8_t> out(page_bytes);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_iters; ++i) {
        (void) tier.lookup(order[i], out.data(), out.size());
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::fprintf(stderr, "  [perf] %d random-order touches over %d resident pages took %.1f ms\n",
                 n_iters, n_pages, elapsed_ms);

    constexpr double deadline_ms = 500.0;
    EXPECT(elapsed_ms < deadline_ms, "LRU touch must be O(1), not a linear scan of the resident set");

    return fails;
}

// ---------------------------------------------------------------------------
// HostTier borrow/release -- zero-copy promotion (2026-07-25 design)
// ---------------------------------------------------------------------------

// 1. Borrow returns the arena address, not a copy: contents match, and the
// pointer is stable across two successive borrow/release cycles for an
// untouched entry.
static int test_host_tier_borrow_returns_arena_address() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/128, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> src(32);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t) (i * 7 + 3);
    EXPECT(tier.store(80, src.data(), src.size()), "store page 80");

    const void * p1 = nullptr;
    wp::HostTier::BorrowHandle h1 = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(80, &p1, src.size(), &h1), "first borrow hits");
    EXPECT(p1 != nullptr, "first borrow returns non-null");
    EXPECT(h1 != wp::HostTier::kInvalidBorrowHandle, "first borrow returns a valid handle");
    EXPECT(std::memcmp(p1, src.data(), src.size()) == 0, "first borrow bytes match");
    tier.release(80, h1);

    const void * p2 = nullptr;
    wp::HostTier::BorrowHandle h2 = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(80, &p2, src.size(), &h2), "second borrow hits");
    EXPECT(std::memcmp(p2, src.data(), src.size()) == 0, "second borrow bytes match");
    EXPECT(p1 == p2, "borrow pointer stable across borrow/release cycles for an untouched entry");
    EXPECT(h1 == h2, "same entry generation across borrow/release cycles for an untouched entry");
    tier.release(80, h2);

    return fails;
}

// 2. Borrow misses for an absent page, and for a resident page requested
// with the wrong size -- both must return false and leave the out-pointer
// untouched.
static int test_host_tier_borrow_miss() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0x81);
    EXPECT(tier.store(81, bytes.data(), bytes.size()), "store page 81");

    const void * sentinel = (const void *) 0x1;
    const wp::HostTier::BorrowHandle handle_sentinel = (wp::HostTier::BorrowHandle) 0xDEAD;
    const void * out = sentinel;
    wp::HostTier::BorrowHandle handle = handle_sentinel;
    EXPECT(!tier.borrow(999, &out, bytes.size(), &handle), "borrow of absent page misses");
    EXPECT(out == sentinel, "absent-page borrow leaves out-pointer untouched");
    EXPECT(handle == handle_sentinel, "absent-page borrow leaves handle-out untouched");

    out = sentinel;
    handle = handle_sentinel;
    EXPECT(!tier.borrow(81, &out, bytes.size() - 1, &handle), "borrow with wrong size misses");
    EXPECT(out == sentinel, "wrong-size borrow leaves out-pointer untouched");
    EXPECT(handle == handle_sentinel, "wrong-size borrow leaves handle-out untouched");

    return fails;
}

// 3. A borrowed page is not evicted: fill the arena to capacity, borrow the
// LRU-front page, store a new page forcing an eviction, and confirm the
// borrowed page's bytes are intact and the *second*-oldest page was the
// victim instead.
static int test_host_tier_borrow_blocks_eviction() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> b90(32, 0x90), b91(32, 0x91), b92(32, 0x92), b93(32, 0x93);
    EXPECT(tier.store(90, b90.data(), b90.size()), "store 90");
    EXPECT(tier.store(91, b91.data(), b91.size()), "store 91");
    EXPECT(tier.store(92, b92.data(), b92.size()), "store 92");
    EXPECT_EQ_INT(tier.resident_count(), 3u, "arena full at capacity");

    const void * borrowed = nullptr;
    wp::HostTier::BorrowHandle handle = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(90, &borrowed, b90.size(), &handle), "borrow LRU-front page 90");

    EXPECT(tier.store(93, b93.data(), b93.size()), "store 93 forces an eviction");

    EXPECT(std::memcmp(borrowed, b90.data(), b90.size()) == 0, "borrowed page 90 bytes intact");
    EXPECT(tier.contains(90), "borrowed page 90 still resident (protected from eviction)");
    EXPECT(!tier.contains(91), "second-oldest page 91 was evicted instead");
    EXPECT(tier.contains(92), "page 92 untouched");
    EXPECT(tier.contains(93), "new page 93 resident");

    tier.release(90, handle);
    return fails;
}

// 4. All-borrowed saturation fails the store cleanly: borrow every resident
// entry, attempt a store, confirm it returns false and no borrowed content
// changed.
static int test_host_tier_all_borrowed_store_fails_cleanly() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> a(32, 0xA0), b(32, 0xB0);
    EXPECT(tier.store(100, a.data(), a.size()), "store 100");
    EXPECT(tier.store(101, b.data(), b.size()), "store 101");

    const void * pa = nullptr;
    const void * pb = nullptr;
    wp::HostTier::BorrowHandle ha = wp::HostTier::kInvalidBorrowHandle;
    wp::HostTier::BorrowHandle hb = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(100, &pa, a.size(), &ha), "borrow 100");
    EXPECT(tier.borrow(101, &pb, b.size(), &hb), "borrow 101");

    std::vector<uint8_t> c(32, 0xC0);
    EXPECT(!tier.store(102, c.data(), c.size()), "store fails when every resident entry is borrowed");

    EXPECT(std::memcmp(pa, a.data(), a.size()) == 0, "borrowed page 100 unchanged");
    EXPECT(std::memcmp(pb, b.data(), b.size()) == 0, "borrowed page 101 unchanged");
    EXPECT(tier.contains(100), "page 100 still resident");
    EXPECT(tier.contains(101), "page 101 still resident");
    EXPECT(!tier.contains(102), "failed store did not create page 102");

    tier.release(100, ha);
    tier.release(101, hb);
    return fails;
}

// 5. Deferred retirement: borrow a page, erase() it, confirm contains() is
// immediately false and the borrowed bytes are still readable; then
// release() and confirm the slot is reused by the next same-size store.
static int test_host_tier_deferred_retirement() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> bytes(32, 0xD0);
    EXPECT(tier.store(110, bytes.data(), bytes.size()), "store 110");

    const void * borrowed = nullptr;
    wp::HostTier::BorrowHandle handle = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(110, &borrowed, bytes.size(), &handle), "borrow 110");

    tier.erase(110);
    EXPECT(!tier.contains(110), "erase() while borrowed makes contains() false immediately");
    EXPECT(std::memcmp(borrowed, bytes.data(), bytes.size()) == 0,
           "borrowed bytes still readable after erase() while retirement is deferred");

    tier.release(110, handle);

    // The slot freed by 110's deferred reclamation, plus the still-free
    // second slot, gives room for two more same-size stores without a
    // forced eviction of anything -- if the slot were leaked, capacity
    // would silently shrink to one page instead of two.
    std::vector<uint8_t> other1(32, 0xD1), other2(32, 0xD2);
    EXPECT(tier.store(111, other1.data(), other1.size()), "slot reused after release drains retirement");
    EXPECT(tier.store(112, other2.data(), other2.size()), "second slot also available (no leak)");
    EXPECT_EQ_INT(tier.resident_count(), 2u, "exactly two pages resident, no leaked/duplicated slot");

    return fails;
}

// 6. Re-store while borrowed does not alias: borrow page A, store() page A
// again with different bytes, confirm the borrowed pointer still yields the
// *original* bytes (it must have been given a different slot).
static int test_host_tier_restore_while_borrowed_no_alias() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/128, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> original(32, 0xE0);
    std::vector<uint8_t> replacement(32, 0xE1);
    EXPECT(tier.store(120, original.data(), original.size()), "store 120 (original)");

    const void * borrowed = nullptr;
    wp::HostTier::BorrowHandle old_handle = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(120, &borrowed, original.size(), &old_handle), "borrow 120");

    EXPECT(tier.store(120, replacement.data(), replacement.size()), "re-store 120 while borrowed");

    EXPECT(std::memcmp(borrowed, original.data(), original.size()) == 0,
           "borrowed pointer still yields the ORIGINAL bytes -- re-store used a different slot");

    std::vector<uint8_t> out(replacement.size());
    EXPECT(tier.lookup(120, out.data(), out.size()), "lookup sees the NEW resident entry for 120");
    EXPECT(std::memcmp(out.data(), replacement.data(), replacement.size()) == 0,
           "new resident entry for 120 has the replacement bytes");

    // Borrowing the NEW entry must yield a DIFFERENT generation handle from
    // the one still held on the retired original -- this is exactly the
    // disambiguation release() relies on to route each release() call to
    // the correct physical entry.
    const void * new_borrowed = nullptr;
    wp::HostTier::BorrowHandle new_handle = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(120, &new_borrowed, replacement.size(), &new_handle),
           "borrow the new entry for 120");
    EXPECT(old_handle != new_handle,
           "re-stored entry has a distinguishably different generation handle");
    EXPECT(new_borrowed != borrowed, "new entry occupies a different arena slot");

    // Release each generation's handle; each must free/decrement the entry
    // it actually belongs to, not whichever one currently occupies page 120.
    tier.release(120, new_handle);
    tier.release(120, old_handle);
    return fails;
}

// 7. Refcount, not a flag: borrow the same page twice, release once and
// confirm it is still protected from eviction; release again and confirm it
// becomes evictable.
static int test_host_tier_borrow_is_refcounted() {
    int fails = 0;

    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    std::vector<uint8_t> b130(32, 0x30), b131(32, 0x31), b132(32, 0x32);
    EXPECT(tier.store(130, b130.data(), b130.size()), "store 130");
    EXPECT(tier.store(131, b131.data(), b131.size()), "store 131 fills arena to capacity");

    const void * p1 = nullptr;
    const void * p2 = nullptr;
    wp::HostTier::BorrowHandle h1 = wp::HostTier::kInvalidBorrowHandle;
    wp::HostTier::BorrowHandle h2 = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(130, &p1, b130.size(), &h1), "first borrow of 130");
    EXPECT(tier.borrow(130, &p2, b130.size(), &h2), "second borrow of 130");
    EXPECT(p1 == p2, "both borrows of the same page return the same address");
    EXPECT(h1 == h2, "both borrows of the same page return the same generation handle");

    tier.release(130, h1);
    EXPECT(tier.store(132, b132.data(), b132.size()),
           "store after ONE release still succeeds (131 is the only evictable victim)");
    EXPECT(tier.contains(130), "130 still protected -- one borrow remains outstanding");
    EXPECT(!tier.contains(131), "131 evicted as the only unborrowed resident entry");

    tier.release(130, h2);
    // Now 130 has zero outstanding borrows and is the sole resident page
    // (alongside 132); a further store must be able to evict it.
    std::vector<uint8_t> b133(32, 0x33);
    EXPECT(tier.store(133, b133.data(), b133.size()), "store after SECOND release evicts 130");
    EXPECT(!tier.contains(130), "130 evictable once its refcount reached zero");
    EXPECT(tier.contains(132), "132 untouched");
    EXPECT(tier.contains(133), "133 resident");

    return fails;
}

// Promotion lifetime seam: this is the CPU analogue of an async H2D. The
// borrow stays live while the completion is deferred, so erase/re-store cannot
// recycle or overwrite the source region before the caller observes completion.
static int test_tier_promotion_borrow_held_until_deferred_completion() {
    int fails = 0;
    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/128, /*device_idx=*/-1), "host tier init");
    std::vector<uint8_t> original(32, 0x41), replacement(32, 0x42);
    EXPECT(tier.store(200, original.data(), original.size()), "store promotion source");

    const void * source = nullptr;
    wp::HostTier::BorrowHandle handle = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(200, &source, original.size(), &handle), "borrow promotion source");
    tier.erase(200); // concurrent retirement while transfer completion is deferred
    EXPECT(!tier.contains(200), "retired source is no longer resident during deferred completion");
    EXPECT(std::memcmp(source, original.data(), original.size()) == 0,
           "borrowed promotion source remains intact before completion fence");
    EXPECT(tier.store(200, replacement.data(), replacement.size()), "re-store uses a distinct slot while borrowed");
    EXPECT(std::memcmp(source, original.data(), original.size()) == 0,
           "re-store cannot overwrite source before deferred completion");
    tier.release(200, handle); // model release after observing completion
    return fails;
}

// Event acquisition failure must leave no outstanding HostTier borrow. The
// actual HIP event pool cannot be initialized without a GPU workload, so this
// covers the CPU-owned lifetime edge that enqueue_tier_promotions_ takes before
// returning its real-read fallback.
static int test_tier_promotion_event_exhaustion_releases_borrow() {
    int fails = 0;
    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/32, /*device_idx=*/-1), "host tier init");
    std::vector<uint8_t> bytes(32, 0x51), other(32, 0x52);
    EXPECT(tier.store(201, bytes.data(), bytes.size()), "store source");
    const void * source = nullptr;
    wp::HostTier::BorrowHandle handle = wp::HostTier::kInvalidBorrowHandle;
    EXPECT(tier.borrow(201, &source, bytes.size(), &handle), "borrow before failed event acquisition");
    tier.release(201, handle); // event unavailable: helper must not retain it
    EXPECT(tier.store(202, other.data(), other.size()), "released borrow leaves a real-read fallback page evictable");
    EXPECT(!tier.contains(201), "source can be evicted after failed event acquisition releases borrow");
    EXPECT(tier.contains(202), "fallback/read replacement is resident");
    return fails;
}


// 8. Concurrency: borrow/release racing with store/erase must never mutate a
// borrowed region while it is held, and used-bytes accounting must return to
// a consistent state at the end. Shaped like test_host_tier_concurrency.
//
// This is the ORIGINAL version: independent threads race borrow/release/
// store/erase on the SAME page_idx, with no coordination between them. Before
// borrow()/release() carried a generation handle, release(page_idx) alone
// could not tell apart two independently-outstanding borrows of the same
// key -- it always guessed "the pending (retired) entry", which was wrong
// whenever the release actually belonged to whatever fresh entry currently
// occupies resident_[page_idx]. That produced real corruption under this
// exact test (24/80000 mismatches observed). Each borrow() now returns the
// handle of the EXACT generation it saw, and release(page_idx, handle)
// decrements that generation specifically -- resident_[page_idx] first if
// its `gen` matches, else the matching entry in pending_ (keyed by handle,
// not by page_idx) -- so two overlapping generations of the same page_idx
// can never be confused with each other.
static int test_host_tier_borrow_release_concurrency_same_key() {
    int fails = 0;
    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/512, /*device_idx=*/-1), "host tier init");

    constexpr int n_threads = 4;
    constexpr int n_pages = 8;
    constexpr int n_iters = 20000;
    constexpr size_t page_size = 32;
    std::atomic<int> bad_borrows{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&tier, &bad_borrows, t]() {
            std::vector<uint8_t> expected(page_size);
            for (int i = 0; i < n_iters; ++i) {
                const int page = (i + t) % n_pages;
                std::fill(expected.begin(), expected.end(), (uint8_t) page);
                if ((i % 7) == 0) {
                    tier.erase(page);
                } else {
                    tier.store(page, expected.data(), expected.size());
                }
                const void * borrowed = nullptr;
                wp::HostTier::BorrowHandle handle = wp::HostTier::kInvalidBorrowHandle;
                if (tier.borrow(page, &borrowed, page_size, &handle)) {
                    // Hold the borrow across a bit of concurrent churn from
                    // other threads, then confirm the region is still
                    // exactly what a resident page of `page` should read as
                    // (all bytes == page), never a torn/aliased mix.
                    std::this_thread::yield();
                    bool consistent = true;
                    const uint8_t * p = (const uint8_t *) borrowed;
                    for (size_t j = 0; j < page_size; ++j) {
                        if (p[j] != (uint8_t) page) {
                            consistent = false;
                            break;
                        }
                    }
                    if (!consistent) {
                        ++bad_borrows;
                    }
                    tier.release(page, handle);
                }
            }
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

    EXPECT_EQ_INT(bad_borrows.load(), 0, "no borrowed region was ever mutated/aliased while held");
    EXPECT(tier.used_bytes() <= tier.budget_bytes(), "used bytes stay within budget");
    EXPECT(tier.resident_count() <= n_pages, "resident pages stay within the page set");
    return fails;
}

// Additional coverage kept alongside the same-key race above: each thread
// owns a disjoint page_idx (no cross-thread key collisions at all), while
// still exercising real cross-thread arena contention -- shared mutex,
// eviction pressure, deferred-retirement churn on other threads' pages.
static int test_host_tier_borrow_release_concurrency_disjoint_pages() {
    int fails = 0;
    wp::HostTier tier;
    // Only 2 32-byte slots for 4 threads' pages -- deliberately undersized so
    // the 2 pages not currently held by their owning thread are under
    // constant eviction pressure from each other while this thread's own
    // borrow is outstanding.
    EXPECT(tier.init(/*budget_bytes=*/64, /*device_idx=*/-1), "host tier init");

    constexpr int n_threads = 4;
    constexpr int n_iters = 20000;
    constexpr size_t page_size = 32;
    std::atomic<int> bad_borrows{0};
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&tier, &bad_borrows, t]() {
            const int page = t;
            std::vector<uint8_t> expected(page_size, (uint8_t) page);
            for (int i = 0; i < n_iters; ++i) {
                if ((i % 7) == 0) {
                    tier.erase(page);
                } else {
                    tier.store(page, expected.data(), expected.size());
                }
                const void * borrowed = nullptr;
                wp::HostTier::BorrowHandle handle = wp::HostTier::kInvalidBorrowHandle;
                if (tier.borrow(page, &borrowed, page_size, &handle)) {
                    // Hold the borrow across a bit of concurrent churn from
                    // the other threads (store/erase/evict on THEIR pages),
                    // then confirm this page's region is still exactly what
                    // it should read as -- never torn, aliased, or reclaimed
                    // out from under the borrow by that unrelated churn.
                    std::this_thread::yield();
                    bool consistent = true;
                    const uint8_t * p = (const uint8_t *) borrowed;
                    for (size_t j = 0; j < page_size; ++j) {
                        if (p[j] != (uint8_t) page) {
                            consistent = false;
                            break;
                        }
                    }
                    if (!consistent) {
                        ++bad_borrows;
                    }
                    tier.release(page, handle);
                }
            }
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

    EXPECT_EQ_INT(bad_borrows.load(), 0, "no borrowed region was ever mutated/aliased while held");
    EXPECT(tier.used_bytes() <= tier.budget_bytes(), "used bytes stay within budget");
    EXPECT(tier.resident_count() <= n_threads, "resident pages stay within the thread-owned page set");
    return fails;
}

static int test_host_prefetcher() {
    int fails = 0;
    wp::HostTier tier;
    EXPECT(tier.init(/*budget_bytes=*/96, /*device_idx=*/-1), "host tier init");

    std::map<int, std::vector<uint8_t>> pages;
    pages.emplace(1, std::vector<uint8_t>(32, 0x11));
    pages.emplace(2, std::vector<uint8_t>(32, 0x22));
    pages.emplace(3, std::vector<uint8_t>(32, 0x33));
    wp::HostPrefetcher prefetcher(
        [&pages](int page, void * dst, size_t capacity) -> int64_t {
            const auto it = pages.find(page);
            if (it == pages.end() || it->second.size() > capacity) {
                return -1;
            }
            std::memcpy(dst, it->second.data(), it->second.size());
            return (int64_t) it->second.size();
        },
        [&tier](int page, const void * bytes, size_t n) {
            return tier.store(page, bytes, n);
        },
        [](int page) { return page == 2; },
        /*max_queue_depth=*/2, /*max_page_size=*/32);

    prefetcher.enqueue(1);
    prefetcher.enqueue(2);
    prefetcher.enqueue(3);
    prefetcher.start();
    prefetcher.stop();

    std::vector<uint8_t> out(32);
    EXPECT(tier.lookup(1, out.data(), out.size()), "non-skipped page stored");
    EXPECT(std::memcmp(out.data(), pages[1].data(), out.size()) == 0, "stored bytes match");
    EXPECT(!tier.contains(2), "skipped page not stored");
    EXPECT(!tier.contains(3), "dropped page not stored");
    EXPECT_EQ_INT(prefetcher.enqueued(), 2u, "two pages accepted into queue");
    EXPECT_EQ_INT(prefetcher.dropped(), 1u, "queue oversubscription drops newest page");
    EXPECT_EQ_INT(prefetcher.read_ok(), 1u, "one page read successfully");
    EXPECT_EQ_INT(prefetcher.read_fail(), 0u, "no read failures");
    EXPECT_EQ_INT(prefetcher.skipped(), 1u, "one page skipped");
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
// resolve_odirect_alignment / compute_odirect_read_plan — O_DIRECT alignment
// authority fix. The pager previously hardcoded 512 (the NVMe device's
// logical_block_size) as the O_DIRECT alignment. That's the wrong authority:
// alignment must come from the FILESYSTEM's block size (statfs f_bsize),
// e.g. btrfs = 4096. Using 512 on a 4096-block filesystem measured 2.49x
// read amplification (221.9 GB delivered vs 82.7 GB buffered for the exact
// same 89.24 GB of requested pages). This also couples in a fix for a
// pre-existing bug: the padded tail of the last page of a shard can run
// past EOF, and O_DIRECT returns EIO (not a short read) rather than
// truncating — padding to a coarser alignment makes the overrun worse, not
// better, so the clamp has to move with the alignment fix.
// ---------------------------------------------------------------------------

static int test_resolve_odirect_alignment() {
    int fails = 0;

    EXPECT_EQ_INT(wp::resolve_odirect_alignment(4096), 4096, "btrfs f_bsize=4096 -> 4096");
    EXPECT_EQ_INT(wp::resolve_odirect_alignment(512),  512,  "f_bsize=512 -> 512 (already >= floor, pow2)");
    EXPECT_EQ_INT(wp::resolve_odirect_alignment(8192), 8192, "larger pow2 f_bsize is honored, not clamped to 4096");
    EXPECT_EQ_INT(wp::resolve_odirect_alignment(-1),   4096, "fstatfs failure sentinel (<=0) -> 4096 fallback");
    EXPECT_EQ_INT(wp::resolve_odirect_alignment(0),    4096, "f_bsize=0 -> 4096 fallback");
    EXPECT_EQ_INT(wp::resolve_odirect_alignment(300),  512,  "below device logical-block floor -> floored to 512, still pow2");
    EXPECT_EQ_INT(wp::resolve_odirect_alignment(600),  4096, "not a power of two after flooring -> 4096 fallback");

    return fails;
}

static int test_compute_odirect_read_plan_aligned_offset() {
    int fails = 0;

    // Already-aligned offset: zero prefix, and the total never grows beyond
    // one pad past the payload (align_up(size, align)).
    for (size_t align : { (size_t) 512, (size_t) 4096 }) {
        const uint64_t off  = align * 10;   // exactly aligned
        const size_t   size = 1000;
        const auto plan = wp::compute_odirect_read_plan(off, size, align, /*file_size=*/0);
        EXPECT_EQ_INT(plan.base, off, "aligned offset: base == off");
        EXPECT_EQ_INT(plan.prefix, 0u, "aligned offset: zero prefix");
        const size_t expect_nbytes = (size + align - 1) & ~(align - 1);
        EXPECT_EQ_INT(plan.nbytes, expect_nbytes, "aligned offset: nbytes is size padded up to align, no extra growth");
        EXPECT(plan.nbytes % align == 0, "aligned offset: nbytes is a multiple of align");
    }
    return fails;
}

static int test_compute_odirect_read_plan_unaligned_offset() {
    int fails = 0;

    for (size_t align : { (size_t) 512, (size_t) 4096 }) {
        const uint64_t base_off = align * 7;
        const size_t   prefix_in = align / 4;              // partial-block unaligned offset
        const uint64_t off  = base_off + prefix_in;
        const size_t   size = align + 17;                  // spans more than one block
        const auto plan = wp::compute_odirect_read_plan(off, size, align, /*file_size=*/0);
        EXPECT_EQ_INT(plan.base, base_off, "unaligned offset: base is align-down of off");
        EXPECT_EQ_INT(plan.prefix, prefix_in, "unaligned offset: prefix is off - base");
        EXPECT(plan.nbytes % align == 0, "unaligned offset: total is a multiple of align");
        EXPECT(plan.prefix + size <= plan.nbytes, "unaligned offset: padded window fully covers the payload");
    }
    return fails;
}

static int test_compute_odirect_read_plan_never_exceeds_buf_cap() {
    int fails = 0;

    // Buffers are sized as page_bytes + 2*align (see ensure_host_bufs_ready_).
    // The worst case (max prefix, max tail pad) must still fit.
    for (size_t align : { (size_t) 512, (size_t) 4096 }) {
        const size_t page_bytes = 16384;
        const size_t buf_cap = page_bytes + 2 * align;
        // Worst-case prefix: align-1.
        const uint64_t off = (align * 3) + (align - 1);
        const auto plan = wp::compute_odirect_read_plan(off, page_bytes, align, /*file_size=*/0);
        EXPECT(plan.nbytes <= buf_cap, "worst-case prefix/pad never exceeds the sized buffer capacity");
    }
    return fails;
}

static int test_compute_odirect_read_plan_eof_clamp() {
    int fails = 0;

    // Mirrors the measured production case: a shard's last page overruns
    // EOF once padded. At align=512 this is the pre-existing bug (fires 3x
    // per run in production); at align=4096 the same offset overruns by
    // MORE, which is exactly why the clamp has to move with the alignment
    // fix rather than staying a 512-only patch.
    const uint64_t shard_size = 46774881376ULL;
    const uint64_t off        = 46770424832ULL;
    const size_t   size       = (size_t) (shard_size - off);  // last page's real payload size

    for (size_t align : { (size_t) 512, (size_t) 4096 }) {
        const auto plan = wp::compute_odirect_read_plan(off, size, align, shard_size);
        EXPECT(plan.base + plan.nbytes <= shard_size,
               "padded end is clamped at EOF, never reads past shard_size");
        EXPECT(plan.prefix + size <= plan.nbytes,
               "clamped window still fully covers the payload bytes");
    }

    // file_size == 0 means "unresolved" -- no clamping should be applied,
    // i.e. the plan pads out fully even though that would run past a
    // (currently unknown) real EOF.
    {
        const auto plan = wp::compute_odirect_read_plan(off, size, /*align=*/4096, /*file_size=*/0);
        const size_t expect_nbytes = (size_t) (((off - (off & ~4095ULL)) + size + 4095) & ~(size_t) 4095);
        EXPECT_EQ_INT(plan.nbytes, expect_nbytes, "file_size=0 (unresolved) disables EOF clamping");
    }

    // A second measured overrun case, at the smaller shard.
    {
        const uint64_t shard_size2 = 46789437824ULL;
        const uint64_t off2        = 46784980992ULL;
        const size_t   size2       = (size_t) (shard_size2 - off2);
        for (size_t align : { (size_t) 512, (size_t) 4096 }) {
            const auto plan = wp::compute_odirect_read_plan(off2, size2, align, shard_size2);
            EXPECT(plan.base + plan.nbytes <= shard_size2, "second shard: clamped at EOF");
            EXPECT(plan.prefix + size2 <= plan.nbytes, "second shard: payload still fully covered");
        }
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

static int test_file_io_submit_batch_depth_one_targeted_waits() {
    int fails = 0;

    char path[] = "/tmp/wp-test-batch-depth1-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr, "  FAIL: %s: mkstemp failed: %s\n", __func__, std::strerror(errno));
        return 1;
    }

    constexpr size_t N = 16384;
    std::vector<uint8_t> pattern(N);
    for (size_t i = 0; i < N; ++i) pattern[i] = (uint8_t) ((i * 19 + 11) & 0xff);
    ssize_t w = write(fd, pattern.data(), N);
    EXPECT_EQ_INT((size_t) w, N, "wrote pattern");

    std::vector<int> fds = { fd };
    auto layer = wp::create_file_io(std::move(fds), /*prefer_async=*/true, 1);
    EXPECT(layer != nullptr, "create_file_io non-null");
    if (!layer) { unlink(path); return fails; }

    std::vector<std::vector<uint8_t>> dst(8, std::vector<uint8_t>(1024));
    std::vector<wp::FileIOBatchRequest> reqs;
    reqs.reserve(dst.size());
    for (size_t i = 0; i < dst.size(); ++i) {
        reqs.push_back({ 1000 + i, 0, (uint64_t) (i * 1536), 1024, dst[i].data() });
    }

    int n_ok = layer->submit_batch(reqs);
    EXPECT_EQ_INT(n_ok, (int) reqs.size(), "all depth-1 batch entries accepted");

    for (size_t i = 0; i < reqs.size(); ++i) {
        wp::IoResult r = layer->wait_for_req(reqs[i].req_id, /*timeout_ms=*/5000);
        EXPECT(r.status == wp::IoStatus::Ok, "targeted wait returns Ok");
        EXPECT_EQ_INT(r.req_id, reqs[i].req_id, "targeted wait req_id");
        EXPECT_EQ_INT(r.bytes_read, 1024, "targeted wait bytes");
        EXPECT(std::memcmp(dst[i].data(), pattern.data() + i * 1536, 1024) == 0,
               "targeted wait content");
    }
    EXPECT_EQ_INT(layer->pending(), 0, "no pending after targeted waits");

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

static int test_router_overrides_expert_only() {
    int fails = 0;

    auto paging   = (ggml_backend_buffer_type_t) 0x1;
    auto resident = (ggml_backend_buffer_type_t) 0x2;
    auto cpu      = (ggml_backend_buffer_type_t) 0x4;
    auto ov = wp::build_router_overrides(paging, resident, cpu, nullptr);
    // expert + shexp + ffn_island + token_embd + dense + terminator
    EXPECT_EQ_INT((int) ov.size(), 6, "expert+shexp+ffn_island+embd+dense+term");
    EXPECT(std::string(ov[0].pattern) == std::string(wp::ROUTER_EXPERT_PATTERN), "expert pattern first");
    EXPECT(ov[0].buft == paging, "expert routed to paging buft");
    EXPECT(std::string(ov[1].pattern) == std::string(wp::ROUTER_SHEXP_PATTERN), "shexp second");
    EXPECT(ov[1].buft == paging, "shexp on paging GPU (resident, not paged)");
    EXPECT(std::string(ov[2].pattern) == std::string(wp::ROUTER_FFN_ISLAND_PATTERN), "ffn island third");
    EXPECT(ov[2].buft == paging, "ffn island on paging GPU");
    EXPECT(std::string(ov[3].pattern) == std::string(wp::ROUTER_TOKEN_EMBD_PATTERN), "token_embd fourth");
    EXPECT(ov[3].buft == cpu, "token_embd on CPU");
    EXPECT(std::string(ov[4].pattern) == std::string(wp::ROUTER_DENSE_PATTERN), "dense catch-all");
    EXPECT(ov[4].buft == resident, "dense catch-all to resident buft");
    EXPECT(ov[5].pattern == nullptr, "list is terminated");
    return fails;
}

static int test_router_overrides_preserve_user() {
    int fails = 0;

    auto paging   = (ggml_backend_buffer_type_t) 0x1;
    auto resident = (ggml_backend_buffer_type_t) 0x2;
    auto userbuft = (ggml_backend_buffer_type_t) 0x3;
    auto cpu      = (ggml_backend_buffer_type_t) 0x4;
    llama_model_tensor_buft_override user[] = {
        { "attn_q\\.", userbuft },
        { nullptr, nullptr },
    };
    auto ov = wp::build_router_overrides(paging, resident, cpu, user);
    // expert + shexp + ffn_island + embd + user + dense + term
    EXPECT_EQ_INT((int) ov.size(), 7, "expert+shexp+ffn_island+embd+user+dense+term");
    EXPECT(std::string(ov[0].pattern) == std::string(wp::ROUTER_EXPERT_PATTERN), "expert pattern first");
    EXPECT(std::string(ov[4].pattern) == std::string("attn_q\\."), "user override BEFORE dense catch-all");
    EXPECT(ov[4].buft == userbuft, "user override buft preserved");
    EXPECT(std::string(ov[5].pattern) == std::string(wp::ROUTER_DENSE_PATTERN), "dense catch-all after user");
    EXPECT(ov[6].pattern == nullptr, "list is terminated");
    return fails;
}

static int test_router_overrides_island_null_matches_default() {
    int fails = 0;

    auto paging   = (ggml_backend_buffer_type_t) 0x1;
    auto resident = (ggml_backend_buffer_type_t) 0x2;
    auto cpu      = (ggml_backend_buffer_type_t) 0x4;
    auto baseline = wp::build_router_overrides(paging, resident, cpu, nullptr);
    auto ov       = wp::build_router_overrides(paging, resident, cpu, nullptr, true, nullptr);
    EXPECT_EQ_INT((int) ov.size(), (int) baseline.size(), "island=nullptr does not change entry count");
    for (size_t i = 0; i < baseline.size(); i++) {
        bool same_pattern = (baseline[i].pattern == nullptr && ov[i].pattern == nullptr) ||
                             (baseline[i].pattern != nullptr && ov[i].pattern != nullptr &&
                              std::string(baseline[i].pattern) == std::string(ov[i].pattern));
        EXPECT(same_pattern, "island=nullptr: pattern matches baseline entry-for-entry");
        EXPECT(baseline[i].buft == ov[i].buft, "island=nullptr: buft matches baseline entry-for-entry");
    }
    return fails;
}

static int test_router_overrides_island_routes_shexp_and_ffn() {
    int fails = 0;

    auto paging   = (ggml_backend_buffer_type_t) 0x1;
    auto resident = (ggml_backend_buffer_type_t) 0x2;
    auto cpu      = (ggml_backend_buffer_type_t) 0x4;
    auto island   = (ggml_backend_buffer_type_t) 0x8;
    auto ov = wp::build_router_overrides(paging, resident, cpu, nullptr, true, island);
    // expert + shexp + ffn_island + token_embd + dense + terminator
    EXPECT_EQ_INT((int) ov.size(), 6, "expert+shexp+ffn_island+embd+dense+term");
    EXPECT(std::string(ov[0].pattern) == std::string(wp::ROUTER_EXPERT_PATTERN), "expert pattern first");
    EXPECT(ov[0].buft == paging, "routed experts still on paging buft");
    EXPECT(std::string(ov[1].pattern) == std::string(wp::ROUTER_SHEXP_PATTERN), "shexp second");
    EXPECT(ov[1].buft == island, "shexp routed to island buft");
    EXPECT(std::string(ov[2].pattern) == std::string(wp::ROUTER_FFN_ISLAND_PATTERN), "ffn island third");
    EXPECT(ov[2].buft == island, "ffn island routed to island buft");
    EXPECT(std::string(ov[3].pattern) == std::string(wp::ROUTER_TOKEN_EMBD_PATTERN), "token_embd fourth");
    EXPECT(ov[3].buft == cpu, "token_embd on CPU");
    EXPECT(std::string(ov[4].pattern) == std::string(wp::ROUTER_DENSE_PATTERN), "dense catch-all");
    EXPECT(ov[4].buft == resident, "dense catch-all still on resident buft");
    EXPECT(ov[5].pattern == nullptr, "list is terminated");
    return fails;
}

static int test_wp_paged_batch_flag_default_off() {
    int fails = 0;
    ScopedEnv guard("WP_PAGED_BATCH");
    unsetenv("WP_PAGED_BATCH");
    if (wp::wp_paged_batch_enabled()) { fprintf(stderr, "FAIL: WP_PAGED_BATCH must default OFF\n"); fails++; }
    return fails;
}

static int test_wp_pipeline_promotions_flag_default_on() {
    int fails = 0;
    ScopedEnv guard("WP_PIPELINE_PROMOTIONS");
    unsetenv("WP_PIPELINE_PROMOTIONS");
    EXPECT(wp::wp_pipeline_promotions_enabled(), "pipeline promotions must default ON");
    setenv("WP_PIPELINE_PROMOTIONS", "0", 1);
    EXPECT(!wp::wp_pipeline_promotions_enabled(), "literal 0 disables pipeline promotions");
    setenv("WP_PIPELINE_PROMOTIONS", "1", 1);
    EXPECT(wp::wp_pipeline_promotions_enabled(), "literal 1 enables pipeline promotions");
    return fails;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// WP_RESIDENT_DENSE relies on the catalog's expert classification agreeing
// with the loader's is_consolidated detection (llama-model.cpp) and the
// is_paged_weight predicate. Lock in the invariant: routed-expert (_exps)
// tensors are experts; dense tensors — attention, embeddings, and the SHARED
// expert (ffn_*_shexp, which matches an ffn_ role prefix but is NOT _exps) —
// are not. A mismatch here would let a dense tensor slip past one filter.
static int test_catalog_is_expert_classification() {
    int fails = 0;
    wp::PageCatalog cat;
    // Dense tensors — must NOT be experts.
    int p_attn = cat.add("blk.0.attn_q.weight",         0, 0,   4096);
    int p_emb  = cat.add("token_embd.weight",           0, 0, 100000);
    int p_shex = cat.add("blk.0.ffn_down_shexp.weight", 0, 0,   8192);
    // Consolidated routed experts — MUST be experts (parent + N sub-pages).
    int first_sub = cat.add_consolidated_experts("blk.0.ffn_gate_exps.weight",
                                                 0, 0, 256 * 8192, 256);
    EXPECT(!cat.at(p_attn).is_expert, "attn_q is dense");
    EXPECT(!cat.at(p_emb).is_expert,  "token_embd is dense");
    EXPECT(!cat.at(p_shex).is_expert, "ffn_down_shexp is dense (shared expert)");
    EXPECT(cat.at(first_sub).is_expert, "ffn_gate_exps sub-page is expert");
    EXPECT(cat.at(first_sub).is_sub_expert, "ffn_gate_exps sub-page is a sub-expert");
    EXPECT(cat.at(first_sub - 1).is_consolidated, "parent is consolidated");
    EXPECT(!cat.at(first_sub - 1).is_expert, "consolidated parent is not itself an expert page");
    EXPECT(cat.has_experts(), "catalog reports experts present");
    EXPECT_EQ_INT(cat.n_expert_pages(), 256, "256 expert sub-pages counted (parent excluded)");
    return fails;
}

static int test_router_predictor() {
    int fails = 0;
    using namespace wp;
    RouterPredictor rp;
    const int n_expert = 4, n_embd = 3;
    // layer 1 router: expert 2 aligns with h=(1,0,0); expert 0 second.
    float W1[n_expert*n_embd] = {
        0.5f,0,0,   // e0
        0,1,0,      // e1
        1,0,0,      // e2 (max dot with h)
        0,0,1 };    // e3
    rp.set_router(/*layer=*/1, W1, n_expert, n_embd);
    EXPECT(rp.has_router(1), "router present after set");
    EXPECT(!rp.has_router(2), "router absent for unset layer");
    float h[n_embd] = {1.0f, 0.0f, 0.0f};
    std::vector<ExpertRef> out;
    rp.predict(h, /*from_layer=*/0, /*K=*/1, /*M=*/2, /*n_layer=*/43, out);
    EXPECT_EQ_INT((int)out.size(), 2, "K=1,M=2 -> 2 refs");
    EXPECT_EQ_INT(out[0].layer, 1, "predicted target layer");
    EXPECT_EQ_INT(out[0].expert, 2, "top-1 expert is e2");
    EXPECT_EQ_INT(out[1].expert, 0, "top-2 expert is e0");
    // K beyond n_layer or unset router -> no refs
    out.clear();
    rp.predict(h, /*from_layer=*/1, /*K=*/1, /*M=*/2, /*n_layer=*/43, out); // target 2 unset
    EXPECT_EQ_INT((int)out.size(), 0, "unset target router -> no refs");
    return fails;
}

// Regression: predict() runs on the async host-prefetch worker while the EVAL
// thread keeps calling set_router() as it first sees each layer. Two hazards:
// routers_ RESIZING under a reader (dangling W), and a router's W being
// REWRITTEN mid-GEMV (torn read). Both are undefined behaviour; the shared_mutex
// serialises them.
//
// DETECTION IS THE HARD PART. A first version of this test checked only that
// returned ExpertRefs were in range -- but `expert` is assigned from the loop
// counter, so it is in range BY CONSTRUCTION and the check passed even with the
// locks deleted (verified: 3/3 pass unlocked). That test could not fail.
//
// This version makes corruption OBSERVABLE. The writer alternates a layer
// between two weight matrices that disagree about which expert wins:
//   Wa: logit(e) grows with e   -> argmax is n_expert-1
//   Wb: logit(e) shrinks with e -> argmax is 0
// Under the lock a reader sees one matrix or the other, so the top expert is
// ALWAYS 0 or n_expert-1. A torn read blends the two and yields a middle
// expert, which no consistent snapshot can produce.
static int test_router_predictor_concurrent_set_and_predict() {
    int fails = 0;
    using namespace wp;

    const int n_expert = 64, n_embd = 64, n_layer = 32;
    std::vector<float> h((size_t) n_embd, 1.0f);
    std::vector<float> Wa((size_t) n_expert * n_embd), Wb((size_t) n_expert * n_embd);
    for (int e = 0; e < n_expert; ++e) {
        for (int j = 0; j < n_embd; ++j) {
            Wa[(size_t) e * n_embd + j] =  (float) e / (float) n_expert;   // argmax = n_expert-1
            Wb[(size_t) e * n_embd + j] = -(float) e / (float) n_expert;   // argmax = 0
        }
    }

    RouterPredictor pred;
    for (int L = 0; L < n_layer; ++L) pred.set_router(L, Wa.data(), n_expert, n_embd);

    std::atomic<bool> torn{false};
    std::atomic<int>  seen{0};
    std::atomic<bool> stop{false};

    std::thread writer([&] {
        for (int rep = 0; rep < 4000 && !stop.load(); ++rep) {
            const float * W = (rep & 1) ? Wb.data() : Wa.data();
            for (int L = 0; L < n_layer; ++L) pred.set_router(L, W, n_expert, n_embd);
        }
    });

    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t) {
        readers.emplace_back([&] {
            std::vector<ExpertRef> out;
            for (int rep = 0; rep < 4000 && !stop.load(); ++rep) {
                out.clear();
                pred.predict(h.data(), 0, /*K=*/1, /*M=*/1, n_layer, out, 0.0f);
                if (!out.empty()) {
                    const int top = out[0].expert;
                    seen.fetch_add(1, std::memory_order_relaxed);
                    if (top != 0 && top != n_expert - 1) {
                        torn.store(true);      // impossible from any single snapshot
                        stop.store(true);
                    }
                }
            }
        });
    }
    writer.join();
    stop.store(true);
    for (auto & r : readers) r.join();

    EXPECT(seen.load() > 0, "readers actually observed predictions");
    EXPECT(!torn.load(), "top expert is always 0 or n_expert-1 (no torn read of W)");
    return fails;
}

static int test_router_predictor_confidence() {
    int fails = 0;
    using namespace wp;
    const int n_expert = 4, n_embd = 1;
    float h[n_embd] = { 1.0f };
    std::vector<ExpertRef> out;

    // PEAKED: e0 logit 10, others 0 -> softmax(e0) ~ 0.9999.
    RouterPredictor rp;
    float Wpk[n_expert*n_embd] = { 10.0f, 0.0f, 0.0f, 0.0f };
    rp.set_router(1, Wpk, n_expert, n_embd);
    rp.predict(h, 0, 1, 4, 43, out, 0.5f);
    EXPECT_EQ_INT((int)out.size(), 1, "peaked+min_conf0.5 -> only e0");
    if (!out.empty()) EXPECT_EQ_INT(out[0].expert, 0, "peaked survivor is e0");
    out.clear();
    rp.predict(h, 0, 1, 4, 43, out, 0.0f);
    EXPECT_EQ_INT((int)out.size(), 4, "min_conf0 gate off -> all M pass");

    // FLAT: all logits 0 -> each softmax prob 0.25.
    RouterPredictor rf;
    float Wfl[n_expert*n_embd] = { 0.0f, 0.0f, 0.0f, 0.0f };
    rf.set_router(1, Wfl, n_expert, n_embd);
    out.clear();
    rf.predict(h, 0, 1, 4, 43, out, 0.5f);
    EXPECT_EQ_INT((int)out.size(), 0, "flat+min_conf0.5 -> none pass");
    out.clear();
    rf.predict(h, 0, 1, 4, 43, out, 0.2f);
    EXPECT_EQ_INT((int)out.size(), 4, "flat+min_conf0.2 -> all pass, p=0.25");
    return fails;
}

static int test_expert_page_index() {
    int fails = 0;
    using namespace wp;
    PageCatalog cat;
    // block 5: three consolidated MoE roles, 4 experts each.
    cat.add_consolidated_experts("blk.5.ffn_gate_exps.weight", 0, 0,      4*4096, 4);
    cat.add_consolidated_experts("blk.5.ffn_up_exps.weight",   0, 100000, 4*4096, 4);
    cat.add_consolidated_experts("blk.5.ffn_down_exps.weight", 0, 200000, 4*4096, 4);
    std::map<std::pair<int,int>, std::vector<int>> idx;
    build_expert_page_index(cat, idx);
    // (block 5, expert 3) -> gate.3 + up.3 + down.3 = 3 sister pages
    auto it = idx.find(std::make_pair(5,3));
    EXPECT(it != idx.end(), "(5,3) present in index");
    if (it != idx.end()) {
        EXPECT_EQ_INT((int) it->second.size(), 3, "(5,3) has 3 sister pages");
        for (int pg : it->second) {
            EXPECT_EQ_INT(cat.at(pg).block_idx, 5, "sister page block 5");
            EXPECT_EQ_INT(cat.at(pg).expert_idx, 3, "sister page expert 3");
        }
    }
    // absent (block,expert)
    EXPECT(idx.find(std::make_pair(99,0)) == idx.end(), "(99,0) absent -> not in index");
    // cross-check: index result matches the linear pages_for_expert() scan.
    auto scan = cat.pages_for_expert(5, 3);
    EXPECT_EQ_INT((int) scan.size(), 3, "pages_for_expert(5,3) also returns 3");
    return fails;
}

static int test_ensure_odirect_inflight_serial_peak_one() {
    int fails = 0;
    using namespace wp;
    // Strictly serial: each read finishes before the next begins. This is
    // the case that would actually diagnose the suspected "9 queued but
    // effectively serialized" problem -- if reads never overlap, peak must
    // be 1 no matter how many jobs were queued.
    EnsureODirectInFlightTracker t;
    for (int i = 0; i < 5; ++i) {
        const int64_t n = t.begin();
        EXPECT_EQ_INT(n, 1, "serial begin() always observes in-flight==1");
        t.end();
    }
    EXPECT_EQ_INT(t.peak(), 1, "serial sequence: peak in-flight == 1");
    EXPECT_EQ_INT(t.current(), 0, "serial sequence: counter back to zero");
    const double avg = t.average();
    EXPECT(avg > 0.999 && avg < 1.001, "serial sequence: avg in-flight == 1.0");
    return fails;
}

static int test_ensure_odirect_inflight_overlap_peak_and_average() {
    int fails = 0;
    using namespace wp;
    EnsureODirectInFlightTracker t;
    // b1 b2 b3 e1 b4 e2 e3 e4 -> in-flight samples at each begin(): 1,2,3,3
    // (b1=1, b2=2, b3=3 [peak], e1 drops to 2, b4 samples 3 again).
    t.begin();               // sample 1
    t.begin();               // sample 2
    const int64_t n3 = t.begin(); // sample 3 (peak)
    EXPECT_EQ_INT(n3, 3, "third overlapping begin() observes in-flight==3");
    t.end();                 // one read completes; current drops to 2
    const int64_t n4 = t.begin(); // sample 3 again (2 in flight + this one)
    EXPECT_EQ_INT(n4, 3, "begin() after one completion observes in-flight==3 again");
    t.end();
    t.end();
    t.end();
    EXPECT_EQ_INT(t.peak(), 3, "overlap sequence: peak in-flight == 3");
    EXPECT_EQ_INT(t.current(), 0, "overlap sequence: counter back to zero (no leak)");
    // samples = {1, 2, 3, 3} -> sum 9, avg 2.25.
    const double avg = t.average();
    EXPECT(avg > 2.249 && avg < 2.251, "overlap sequence: avg in-flight == 2.25 per documented definition");
    return fails;
}

static int test_ensure_odirect_inflight_no_leak_on_pairing() {
    int fails = 0;
    using namespace wp;
    EnsureODirectInFlightTracker t;
    // A larger mixed begin/end interleaving; regardless of the pattern,
    // every begin() must be matched by exactly one end(), so the counter
    // must return to zero once all reads complete -- no leak in the
    // increment/decrement pairing.
    for (int i = 0; i < 8; ++i) {
        t.begin();
    }
    for (int i = 0; i < 3; ++i) {
        t.end();
    }
    for (int i = 0; i < 4; ++i) {
        t.begin();
    }
    for (int i = 0; i < 9; ++i) {
        t.end();
    }
    EXPECT_EQ_INT(t.current(), 0, "mixed begin/end sequence: counter back to zero, no leak");
    EXPECT(t.peak() >= 8, "mixed sequence: peak reflects the highest overlap actually reached");
    return fails;
}

static int test_pool_speculative() {
    int fails = 0;
    using namespace wp;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type available"); return fails; }

    PoolAllocator pool;
    EXPECT(pool.init(buft, /*n_slots=*/3, /*slot_size=*/256), "pool init 3 slots");
    int a = pool.alloc_slot(); int b = pool.alloc_slot(); int c = pool.alloc_slot();
    EXPECT_EQ_INT(a, 0, "alloc a=0"); EXPECT_EQ_INT(b, 1, "alloc b=1"); EXPECT_EQ_INT(c, 2, "alloc c=2");
    EXPECT(!pool.is_speculative(a), "fresh alloc is non-speculative");
    EXPECT_EQ_INT(pool.n_speculative(), 0, "no speculative yet");

    pool.set_speculative(b, true);
    EXPECT(pool.is_speculative(b), "b marked speculative");
    EXPECT_EQ_INT(pool.n_speculative(), 1, "one speculative");

    pool.pin_slot(a); pool.pin_slot(c);            // a,c are the pinned working set
    int d = pool.alloc_slot();                     // must evict b (speculative), not a/c
    EXPECT_EQ_INT(d, b, "alloc evicts the speculative slot first");
    EXPECT(!pool.is_speculative(d), "reused slot is non-speculative");
    EXPECT_EQ_INT(pool.n_speculative(), 0, "speculative cleared after reuse");

    // promotion: a speculative slot that gets mark_used is no longer speculative
    pool.set_speculative(d, true);
    pool.mark_used(d);
    EXPECT(!pool.is_speculative(d), "mark_used promotes (clears speculative)");
    EXPECT_EQ_INT(pool.n_speculative(), 0, "promoted slot not counted");
    pool.unpin_slot(a); pool.unpin_slot(c);

    // Discriminating case: Pass-0 evicts a NEWER speculative slot before an
    // OLDER non-speculative one (proves speculative-first beats pure LRU).
    PoolAllocator p2;
    EXPECT(p2.init(buft, /*n_slots=*/3, /*slot_size=*/256), "p2 init");
    int x = p2.alloc_slot(); (void) x;             // oldest, non-speculative
    int y = p2.alloc_slot();
    int z = p2.alloc_slot();
    p2.mark_used(z);                                // z newest by recency
    p2.set_speculative(z, true);                    // z speculative AND newest
    p2.pin_slot(y);                                 // isolate x (old) vs z (new,spec)
    int w = p2.alloc_slot();                        // pure LRU->x; speculative-first->z
    EXPECT_EQ_INT(w, z, "speculative slot evicted before older non-speculative (Pass 0 > LRU)");
    return fails;
}

// Regression: intra-batch self-cannibalisation (gate 4, diagnosed 2026-07-27).
//
// prefetch_pages_batch reserves N slots in a loop. Marking a slot speculative
// makes it a legal Pass-0 victim for the very next iteration -- and on a seeding
// batch it is the ONLY speculative slot, so it is trivially the LRU of its
// cohort and is handed straight back. The batch then carries the same slot
// twice, two reads DMA into one buffer, and the first page is silently mapped to
// the second page's bytes. Wrong weights, no crash.
//
// The invariant that closes it: pin BEFORE marking, so Pass 0's pin_count_ test
// excludes the slot from the moment it becomes speculative. Both halves are
// asserted -- the second documents why the ordering is load-bearing.
static int test_pool_speculative_batch_no_self_evict() {
    int fails = 0;
    using namespace wp;
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (!buft) { EXPECT(false, "cpu buffer_type available"); return fails; }

    const int N = 4;   // warm pool: every slot used, unpinned, non-speculative

    // --- Correct order: pin, then mark. A batch must return distinct slots. ---
    {
        PoolAllocator pool;
        EXPECT(pool.init(buft, N, /*slot_size=*/256), "batch pool init");
        for (int i = 0; i < N; ++i) { int s = pool.alloc_slot(); pool.mark_used(s); }

        std::vector<int> slots;
        bool dup = false;
        for (int i = 0; i < N; ++i) {
            const int s = pool.alloc_slot();          // allow_evict path
            EXPECT(s >= 0, "batch alloc succeeds");
            if (s < 0) break;
            for (int prev : slots) { if (prev == s) dup = true; }
            slots.push_back(s);
            pool.pin_slot(s);                          // pin FIRST ...
            pool.set_speculative(s, true);             // ... then mark
        }
        EXPECT(!dup, "batch alloc returns distinct slots when pinned before marking");
        EXPECT_EQ_INT((int) slots.size(), N, "whole batch allocated");
        EXPECT_EQ_INT(pool.n_speculative(), N, "all batch slots speculative");
        for (int s : slots) pool.unpin_slot(s);
    }

    // --- The hazard itself: mark without pinning and the next alloc reclaims it. ---
    {
        PoolAllocator pool;
        EXPECT(pool.init(buft, N, /*slot_size=*/256), "hazard pool init");
        for (int i = 0; i < N; ++i) { int s = pool.alloc_slot(); pool.mark_used(s); }

        const int first = pool.alloc_slot();
        pool.set_speculative(first, true);             // marked but NOT pinned
        const int second = pool.alloc_slot();
        EXPECT_EQ_INT(second, first,
                      "unpinned speculative slot is recycled by the next alloc "
                      "(this is the corruption the pin-before-mark order prevents)");
    }
    return fails;
}

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
        { "prefetch_wait_transport_error_req_id_zero", test_prefetch_wait_transport_error_req_id_zero },
        { "p2p_tunable_resolution", test_p2p_tunable_resolution },
        { "dup_clear_o_direct", test_dup_clear_o_direct },
        { "file_io_sync_pread", test_file_io_sync_pread },
        { "file_io_advise_prefetch",  test_file_io_advise_prefetch  },
        { "file_io_submit_batch",            test_file_io_submit_batch            },
        { "file_io_submit_batch_partial",    test_file_io_submit_batch_partial_failure },
        { "file_io_submit_batch_depth_one_targeted_waits", test_file_io_submit_batch_depth_one_targeted_waits },
        { "file_io_demux_no_cross_drain",    test_file_io_demux_no_cross_drain    },
        { "compute_advise_ranges",    test_compute_advise_ranges    },
        { "resolve_odirect_alignment", test_resolve_odirect_alignment },
        { "compute_odirect_read_plan_aligned_offset", test_compute_odirect_read_plan_aligned_offset },
        { "compute_odirect_read_plan_unaligned_offset", test_compute_odirect_read_plan_unaligned_offset },
        { "compute_odirect_read_plan_never_exceeds_buf_cap", test_compute_odirect_read_plan_never_exceeds_buf_cap },
        { "compute_odirect_read_plan_eof_clamp", test_compute_odirect_read_plan_eof_clamp },
        { "is_uma_archname",          test_is_uma_archname          },
        { "read_mem_available_bytes", test_read_mem_available_bytes },
        { "is_uma_device_smoke",      test_is_uma_device_smoke      },
        { "pool_allocator",     test_pool_allocator     },
        { "pool_allocator_two_pools_independent", test_pool_allocator_two_pools_independent },
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
        { "host_tier_speculative_evicts_before_victim", test_host_tier_speculative_evicts_before_victim },
        { "host_tier_lookup_touch_keeps_mru",    test_host_tier_lookup_touch_keeps_mru    },
        { "host_tier_over_budget_evict",         test_host_tier_over_budget_evict         },
        { "host_tier_lookup_miss",               test_host_tier_lookup_miss               },
        { "host_tier_concurrency",               test_host_tier_concurrency               },
        { "host_tier_repeated_touches_eviction_order", test_host_tier_repeated_touches_eviction_order },
        { "host_tier_erase_middle_preserves_order",    test_host_tier_erase_middle_preserves_order    },
        { "host_tier_touch_absent_is_noop",            test_host_tier_touch_absent_is_noop            },
        { "host_tier_lru_touch_is_not_linear_scan",    test_host_tier_lru_touch_is_not_linear_scan    },
        { "host_tier_borrow_returns_arena_address",     test_host_tier_borrow_returns_arena_address     },
        { "host_tier_borrow_miss",                      test_host_tier_borrow_miss                      },
        { "host_tier_borrow_blocks_eviction",            test_host_tier_borrow_blocks_eviction            },
        { "host_tier_all_borrowed_store_fails_cleanly",  test_host_tier_all_borrowed_store_fails_cleanly  },
        { "host_tier_deferred_retirement",               test_host_tier_deferred_retirement               },
        { "host_tier_restore_while_borrowed_no_alias",   test_host_tier_restore_while_borrowed_no_alias   },
        { "host_tier_borrow_is_refcounted",              test_host_tier_borrow_is_refcounted              },
        { "tier_promotion_borrow_held_until_deferred_completion", test_tier_promotion_borrow_held_until_deferred_completion },
        { "tier_promotion_event_exhaustion_releases_borrow",      test_tier_promotion_event_exhaustion_releases_borrow },
        { "host_tier_borrow_release_concurrency_same_key",      test_host_tier_borrow_release_concurrency_same_key      },
        { "host_tier_borrow_release_concurrency_disjoint_pages", test_host_tier_borrow_release_concurrency_disjoint_pages },
        { "host_prefetcher",                     test_host_prefetcher                     },
        { "catalog_add_pinned_basic",            test_catalog_add_pinned_basic            },
        { "catalog_add_pinned_mixed_with_paged", test_catalog_add_pinned_mixed_with_paged },
        { "catalog_clear_resets_pinned",         test_catalog_clear_resets_pinned_counters },
        { "catalog_is_expert_classification",    test_catalog_is_expert_classification    },
        { "pool_hit_count_basic",                test_pool_hit_count_basic                },
        { "pool_hot_threshold_protects",         test_pool_hot_threshold_protects_in_eviction },
        { "pool_hot_fallback_when_all_hot",      test_pool_hot_fallback_when_all_hot      },
        { "pool_default_threshold_zero_lru",     test_pool_default_threshold_zero_is_pure_lru },
        { "routing_boundary_prepass",            test_routing_boundary_prepass            },
        { "router_overrides_expert_only",        test_router_overrides_expert_only        },
        { "router_overrides_preserve_user",      test_router_overrides_preserve_user      },
        { "router_overrides_island_null_matches_default", test_router_overrides_island_null_matches_default },
        { "router_overrides_island_routes_shexp_and_ffn", test_router_overrides_island_routes_shexp_and_ffn },
        { "wp_paged_batch_flag_default_off",     test_wp_paged_batch_flag_default_off     },
        { "wp_pipeline_promotions_flag_default_on",  test_wp_pipeline_promotions_flag_default_on },
        { "router_predictor",                    test_router_predictor                    },
        { "router_predictor_confidence",         test_router_predictor_confidence         },
        { "router_predictor_concurrent",         test_router_predictor_concurrent_set_and_predict },
        { "expert_page_index",                   test_expert_page_index                   },
        { "pool_speculative",                    test_pool_speculative                    },
        { "pool_speculative_batch_no_self_evict", test_pool_speculative_batch_no_self_evict },
        { "ensure_odirect_inflight_serial_peak_one", test_ensure_odirect_inflight_serial_peak_one },
        { "ensure_odirect_inflight_overlap_peak_and_average", test_ensure_odirect_inflight_overlap_peak_and_average },
        { "ensure_odirect_inflight_no_leak_on_pairing", test_ensure_odirect_inflight_no_leak_on_pairing },
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
