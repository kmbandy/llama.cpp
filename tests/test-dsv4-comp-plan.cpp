// dsv4 compressor-plan ranks must be stable for a given ubatch width.
// A live plan at pos=3 (no HCA commit) and pos=127 (HCA commit) used to
// differ in state_write_idxs / n_kv, which fails can_reuse and recaptures
// a new HIP graph every time the rank changes.

#include "llama-kv-cache-dsv4.h"
#include "llama-batch.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct UbatchHolder {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_storage;
    std::vector<llama_seq_id *> seq_id_ptrs;
    llama_seq_id                seq_unq = 0;
    llama_ubatch                ub{};
};

UbatchHolder make_decode_ubatch(llama_pos pos) {
    UbatchHolder h;
    h.pos = { pos };
    h.n_seq_id = { 1 };
    h.seq_id_storage = { 0 };
    h.seq_id_ptrs = { &h.seq_id_storage[0] };
    h.seq_unq = 0;
    h.ub.n_tokens     = 1;
    h.ub.n_seq_tokens = 1;
    h.ub.n_seqs       = 1;
    h.ub.n_seqs_unq   = 1;
    h.ub.n_pos        = 1;
    h.ub.pos          = h.pos.data();
    h.ub.n_seq_id     = h.n_seq_id.data();
    h.ub.seq_id       = h.seq_id_ptrs.data();
    h.ub.seq_id_unq   = &h.seq_unq;
    return h;
}

void require(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

void require_same_rank(
        const llama_kv_cache_dsv4_context::comp_plan & a,
        const llama_kv_cache_dsv4_context::comp_plan & b,
        const char * tag) {
    auto chk = [&](size_t x, size_t y, const char * name) {
        if (x != y) {
            std::fprintf(stderr, "FAIL: %s %s rank %zu vs %zu\n", tag, name, x, y);
            std::exit(1);
        }
    };
    chk(a.state_pos.size(), b.state_pos.size(), "state_pos");
    chk(a.state_persist_src_idxs.size(), b.state_persist_src_idxs.size(), "persist_src");
    chk(a.state_persist_dst_idxs.size(), b.state_persist_dst_idxs.size(), "persist_dst");
    chk(a.state_restore_src_idxs.size(), b.state_restore_src_idxs.size(), "restore_src");
    chk(a.state_restore_dst_idxs.size(), b.state_restore_dst_idxs.size(), "restore_dst");
    chk(a.state_snapshot_src_idxs.size(), b.state_snapshot_src_idxs.size(), "snapshot_src");
    chk(a.state_snapshot_dst_idxs.size(), b.state_snapshot_dst_idxs.size(), "snapshot_dst");
    chk(a.state_read_idxs.size(), b.state_read_idxs.size(), "read_idxs");
    chk(a.state_write_idxs.size(), b.state_write_idxs.size(), "write_idxs");
    chk(a.state_write_pos.size(), b.state_write_pos.size(), "write_pos");
    chk((size_t) a.n_kv, (size_t) b.n_kv, "n_kv");
    chk((size_t) a.n_stream, (size_t) b.n_stream, "n_stream");
}

} // namespace

int main() {
    constexpr uint32_t k_state = 8;
    constexpr uint32_t k_kv    = 1024;
    constexpr uint32_t k_stream = 1;
    constexpr uint32_t k_rs     = 0;
    const std::vector<uint32_t> rs_idx;

    auto early = make_decode_ubatch(3);    // (pos+1) % 128 != 0 — no HCA commit
    auto bound = make_decode_ubatch(127);  // (pos+1) % 128 == 0 — HCA commit
    auto late  = make_decode_ubatch(900);  // n_visible has grown

    const auto hca_early = llama_dsv4_build_comp_plan(
            early.ub, /*ratio=*/128, /*overlap=*/false, k_state, k_kv, k_stream, k_rs, rs_idx);
    const auto hca_bound = llama_dsv4_build_comp_plan(
            bound.ub, /*ratio=*/128, /*overlap=*/false, k_state, k_kv, k_stream, k_rs, rs_idx);
    const auto hca_late = llama_dsv4_build_comp_plan(
            late.ub, /*ratio=*/128, /*overlap=*/false, k_state, k_kv, k_stream, k_rs, rs_idx);

    require_same_rank(hca_early, hca_bound, "hca early vs boundary");
    require_same_rank(hca_early, hca_late,  "hca early vs late");
    require(hca_early.n_kv == (int64_t) k_kv, "hca decode n_kv is the allocated cache size");
    require(!hca_early.state_write_idxs.empty(), "hca decode always has a write slot (scratch if no commit)");

    const auto csa_early = llama_dsv4_build_comp_plan(
            early.ub, /*ratio=*/4, /*overlap=*/true, k_state, k_kv, k_stream, k_rs, rs_idx);
    const auto csa_late = llama_dsv4_build_comp_plan(
            late.ub, /*ratio=*/4, /*overlap=*/true, k_state, k_kv, k_stream, k_rs, rs_idx);
    require_same_rank(csa_early, csa_late, "csa early vs late");
    require(csa_early.n_kv == (int64_t) k_kv, "csa decode n_kv is the allocated cache size");

    std::printf("ok: dsv4 decode plan ranks are stable\n");
    return 0;
}
