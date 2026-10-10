// T1: WP_SPINE_ROW_GROUPS compressor-plan split. CPU only, no model.
//
// Integer-tagged simulation. Each state row carries a tag that names the
// position it came from. Applying the whole padded plan (oracle) must give
// the same ring, rollback planes and cache rows as applying the group A
// plan then the group B plan, with B's commits reading the ring after A's
// persist. Scratch rows of group B are poisoned until B is computed.

#include "llama-batch.h"
#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

using Tag = int64_t;
using comp_plan = llama_kv_cache_dsv4_context::comp_plan;

static const Tag ZERO_TAG   = -2;
static const Tag STALE_TAG  = -1;
static const Tag POISON_TAG = -999;

struct sim {
    int32_t S = 0;
    int32_t n_rs = 0;
    int32_t R = 0;
    std::vector<Tag> X;                       // [ring S | planes n_rs*S | scratch R]
    std::vector<std::vector<Tag>> cache;      // one tuple per cache row

    int64_t scratch_base() const { return (int64_t) S*(1 + n_rs); }
};

static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

static int64_t mod_pos(int64_t p, int32_t S) {
    return ((p % S) + S) % S;
}

static sim make_sim(int32_t S, int32_t n_rs, int32_t R, int32_t kv_size, int64_t p0, bool poison_from_k, int32_t k) {
    sim s;
    s.S = S;
    s.n_rs = n_rs;
    s.R = R;
    s.X.assign((size_t) S*(1 + n_rs) + R, 0);

    // Ring: slot r holds the latest pre-ubatch position with pos % S == r.
    for (int32_t r = 0; r < S; ++r) {
        Tag t = 900000 + r;
        for (int64_t q = p0 - 1; q >= 0 && q >= p0 - S; --q) {
            if (mod_pos(q, S) == r) {
                t = 1000000 + q;
                break;
            }
        }
        s.X[r] = t;
    }
    for (int32_t d = 1; d <= n_rs; ++d) {
        for (int32_t r = 0; r < S; ++r) {
            s.X[(size_t) S*d + r] = 3000000 + (Tag) d*100000 + r;
        }
    }
    for (int32_t j = 0; j < R; ++j) {
        const bool poisoned = poison_from_k && j >= k;
        s.X[s.scratch_base() + j] = poisoned ? POISON_TAG : 2000000 + p0 + j;
    }
    s.cache.assign((size_t) kv_size, std::vector<Tag>{ STALE_TAG });
    return s;
}

// Value of one source row. zero_idx is the group's zero sentinel.
static Tag src_val(const sim & s, int32_t idx, int32_t zero_idx, int32_t n_scr) {
    if (idx == zero_idx) {
        return ZERO_TAG;
    }
    if (idx >= 0 && idx < s.S) {
        return s.X[idx];
    }
    if (idx >= s.S && idx < s.S + n_scr) {
        const Tag t = s.X[s.scratch_base() + (idx - s.S)];
        if (t == POISON_TAG) {
            fail("read of a scratch row that is not computed yet");
        }
        return t;
    }
    fail("source index out of the group's view");
    return 0;
}

static void apply_restore(sim & s, const comp_plan & P) {
    for (size_t j = 0; j < P.state_restore_dst_idxs.size(); ++j) {
        s.X[P.state_restore_dst_idxs[j]] = s.X[P.state_restore_src_idxs[j]];
    }
}

static void apply_commits(sim & s, const comp_plan & P, uint32_t ratio, bool overlap, int32_t n_scr, int32_t zero_idx) {
    const size_t nb = P.state_write_idxs.size();
    const size_t half = overlap ? P.state_read_idxs.size()/2 : 0;
    for (size_t b = 0; b < nb; ++b) {
        std::vector<Tag> tuple;
        for (uint32_t j = 0; j < ratio; ++j) {
            tuple.push_back(src_val(s, P.state_read_idxs[b*ratio + j], zero_idx, n_scr));
        }
        if (overlap) {
            for (uint32_t j = 0; j < ratio; ++j) {
                tuple.push_back(src_val(s, P.state_read_idxs[half + b*ratio + j], zero_idx, n_scr));
            }
        }
        s.cache[P.state_write_idxs[b]] = tuple;
    }
}

// only_src_ok: when non-null, snapshot entries with src >= scratch_off are
// skipped unless the predicate allows them (used by the phase split).
static void apply_snapshot(sim & s, const comp_plan & P, bool (*take)(int32_t src, int64_t scratch_off, int32_t k), int32_t k) {
    for (size_t j = 0; j < P.state_snapshot_dst_idxs.size(); ++j) {
        const int32_t src = P.state_snapshot_src_idxs[j];
        if (take != nullptr && !take(src, s.scratch_base(), k)) {
            continue;
        }
        s.X[P.state_snapshot_dst_idxs[j]] = s.X[src];
    }
}

static void apply_persist(sim & s, const comp_plan & P, int32_t n_scr_base_tokens) {
    (void) n_scr_base_tokens;
    for (size_t j = 0; j < P.state_persist_dst_idxs.size(); ++j) {
        s.X[P.state_persist_dst_idxs[j]] = s.X[s.scratch_base() + P.state_persist_src_idxs[j]];
    }
}

static void apply_whole(sim & s, const comp_plan & P, uint32_t ratio, bool overlap) {
    apply_restore(s, P);
    apply_commits(s, P, ratio, overlap, s.R, s.S + s.R);
    apply_snapshot(s, P, nullptr, 0);
    apply_persist(s, P, s.R);
}

static bool take_a(int32_t src, int64_t scratch_off, int32_t k) {
    return src < scratch_off || src - scratch_off < k;
}
static bool take_b(int32_t src, int64_t scratch_off, int32_t k) {
    return src >= scratch_off && src - scratch_off >= k;
}
static bool take_all(int32_t, int64_t, int32_t) {
    return true;
}

// Group A: rows [0,k) computed. Order: restore, commits, snapshot, persist.
// bad_order = 1: persist before snapshot. bad_order = 2: snapshot takes all entries.
static void apply_group_a(sim & s, const comp_plan & A, uint32_t ratio, bool overlap, int32_t k, int bad_order) {
    apply_restore(s, A);
    apply_commits(s, A, ratio, overlap, k, s.S + k);
    if (bad_order == 1) {
        apply_persist(s, A, s.R);
        apply_snapshot(s, A, take_a, k);
        return;
    }
    apply_snapshot(s, A, bad_order == 2 ? take_all : take_a, k);
    apply_persist(s, A, s.R);
}

// Group B: rows [k,R) are computed by set_b_rows before this runs.
static void apply_group_b(sim & s, const comp_plan & B, uint32_t ratio, bool overlap, int32_t k) {
    apply_commits(s, B, ratio, overlap, s.R, s.S + s.R);
    apply_snapshot(s, B, take_b, k);
    apply_persist(s, B, s.R);
}

static void set_b_rows(sim & s, int64_t p0, int32_t k) {
    for (int32_t j = k; j < s.R; ++j) {
        s.X[s.scratch_base() + j] = 2000000 + p0 + j;
    }
}

static void compare(const sim & want, const sim & got, const char * what) {
    for (int32_t r = 0; r < want.S*(1 + want.n_rs); ++r) {
        if (want.X[r] != got.X[r]) {
            std::fprintf(stderr, "%s: state index %d differs (want %lld got %lld)\n", what, r,
                         (long long) want.X[r], (long long) got.X[r]);
            std::exit(1);
        }
    }
    for (size_t c = 0; c + 1 < want.cache.size(); ++c) {  // last slot is the masked dummy
        if (want.cache[c] != got.cache[c]) {
            std::fprintf(stderr, "%s: cache row %zu differs\n", what, c);
            std::exit(1);
        }
    }
}

static llama_ubatch make_ubatch(std::shared_ptr<llama_ubatch::data_t> & data, int32_t R, int64_t p0) {
    data = std::make_shared<llama_ubatch::data_t>();
    data->pos.resize(R);
    data->n_seq_id.assign(R, 1);
    data->seq_id_data.assign(R, 0);
    data->seq_id.resize(R);
    for (int32_t i = 0; i < R; ++i) {
        data->pos[i] = (llama_pos) (p0 + i);
        data->seq_id[i] = &data->seq_id_data[i];
    }
    data->seq_id_unq = { 0 };
    data->seq_idx.assign(1, 0);
    data->output.assign(R, 0);

    llama_ubatch ub = {};
    ub.b_equal_seqs = 1;
    ub.n_tokens = R;
    ub.n_seq_tokens = R;
    ub.n_seqs = 1;
    ub.n_seqs_unq = 1;
    ub.n_pos = 1;
    ub.pos = data->pos.data();
    ub.n_seq_id = data->n_seq_id.data();
    ub.seq_id = data->seq_id.data();
    ub.seq_id_unq = data->seq_id_unq.data();
    ub.seq_idx = data->seq_idx.data();
    ub.output = data->output.data();
    ub.data = data;
    return ub;
}

// llama_dsv4_split_supported: accept/reject table. Returns the number of checks run.
static long run_supported_checks() {
    long n = 0;
    const char * why = nullptr;

    auto expect = [&](bool want, uint32_t R, uint32_t ratio, bool overlap, uint32_t S,
                      uint32_t n_stream, uint32_t k, const char * label, const std::function<void(llama_ubatch &, std::shared_ptr<llama_ubatch::data_t> &)> & mut) {
        std::shared_ptr<llama_ubatch::data_t> data;
        llama_ubatch ub = make_ubatch(data, (int32_t) R, 100);
        if (mut) {
            mut(ub, data);
        }
        const bool got = llama_dsv4_split_supported(ub, ratio, overlap, S, n_stream, k, &why);
        if (got != want) {
            std::fprintf(stderr, "split_supported(%s): want %d got %d (why=%s)\n", label, (int) want, (int) got,
                         why ? why : "none");
            std::exit(1);
        }
        if (!got && (why == nullptr || why[0] == '\0')) {
            fail("split_supported rejected without a reason");
        }
        ++n;
    };
    const std::function<void(llama_ubatch &, std::shared_ptr<llama_ubatch::data_t> &)> none = nullptr;

    // v41 configs, every R and k = ceil(R/2).
    for (uint32_t R = 4; R <= 8; ++R) {
        expect(true, R, 4,   false, 4,   1, (R + 1)/2, "v41 ratio4 S4", none);
        expect(true, R, 128, false, 128, 1, (R + 1)/2, "v41 ratio128 S128", none);
    }
    // DS4.1 real ratios 1 and 2 with state_size == ratio, every R and k = ceil(R/2).
    for (uint32_t R = 4; R <= 8; ++R) {
        expect(true, R, 1, false, 1, 1, (R + 1)/2, "v41 ratio1 S1", none);
        expect(true, R, 2, false, 2, 1, (R + 1)/2, "v41 ratio2 S2", none);
    }
    // Existing overlap configs.
    expect(true, 8, 4,   true,  8,   1, 4, "overlap ratio4 S8", none);
    expect(true, 8, 4,   true,  16,  1, 4, "overlap ratio4 S16", none);
    expect(true, 6, 128, false, 256, 1, 3, "non-overlap ratio128 S256", none);
    // Non-overlap S < k and S < R-k are accepted: each group persists min(rows, S).
    expect(true, 8, 2, false, 3, 1, 4, "non-overlap S < k", none);
    expect(true, 8, 2, false, 4, 1, 2, "non-overlap S < R-k", none);

    // Rejections.
    expect(false, 8, 4, false, 2, 1, 4, "S < ratio", none);
    expect(false, 8, 2, false, 1, 1, 4, "non-overlap S < ratio (ratio2 S1)", none);
    expect(false, 8, 4, true,  4, 1, 4, "overlap S < 2*ratio", none);
    expect(false, 8, 2, true,  4, 1, 4, "overlap S < R", none);
    expect(false, 8, 4, false, 4, 2, 4, "n_stream != 1", none);
    expect(false, 8, 4, false, 4, 1, 0, "k == 0", none);
    expect(false, 8, 4, false, 4, 1, 8, "k == R", none);
    expect(false, 8, 0, false, 4, 1, 4, "ratio == 0", none);
    expect(false, 8, 4, false, 4, 1, 4, "multi-seq",
            [](llama_ubatch & ub, std::shared_ptr<llama_ubatch::data_t> & d) {
                d->seq_id_unq = { 0, 1 };
                ub.n_seqs_unq = 2;
                ub.seq_id_unq = d->seq_id_unq.data();
            });
    expect(false, 8, 4, false, 4, 1, 4, "coupled rows",
            [](llama_ubatch & ub, std::shared_ptr<llama_ubatch::data_t> & d) {
                d->n_seq_id[3] = 2;
                (void) ub;
            });
    expect(false, 8, 4, false, 4, 1, 4, "non-consecutive pos",
            [](llama_ubatch &, std::shared_ptr<llama_ubatch::data_t> & d) {
                d->pos[5] += 2;
            });
    expect(false, 8, 4, false, 4, 1, 4, "negative pos",
            [](llama_ubatch &, std::shared_ptr<llama_ubatch::data_t> & d) {
                for (auto & p : d->pos) p -= 200;
            });
    return n;
}

int main() {
    struct cfg {
        uint32_t ratio;
        bool     overlap;
        int32_t  S;
        int32_t  kv;
        std::vector<int64_t> p0s;
    };
    const std::vector<cfg> cfgs = {
        { 4,   true,  8,   128, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100, 101, 102, 103, 104, 205 } },
        { 4,   true,  16,  128, { 0, 3, 7, 101, 203 } },
        { 128, false, 256, 8,   { 0, 1, 125, 126, 127, 128, 129, 130, 253, 254, 255, 256, 257, 300 } },
        // DS4.1 real configs: state_size == ratio, overlap=false.
        { 4,   false, 4,   128, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 } },
        { 128, false, 128, 8,   { 0, 1, 2, 124, 125, 126, 127, 128, 129, 130, 252, 253, 254, 255, 256, 257, 383 } },
        // DS4.1 real ratios 1 and 2: state_size == ratio, overlap=false.
        { 1,   false, 1,   128, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } },
        { 2,   false, 2,   128, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 } },
    };

    long n_cases = 0;
    long n_order_neg = 0;
    std::vector<long> cfg_cases(cfgs.size(), 0);
    std::vector<long> cfg_neg(cfgs.size(), 0);

    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        const cfg & c = cfgs[ci];
        for (int32_t R = 4; R <= 8; ++R) {
            const int32_t k = (R + 1) / 2;
            // Persist-before-snapshot only shows up when d > R-k, so n_rs goes to 4.
            for (int32_t n_rs : { 0, 2, 4 }) {
                for (int32_t rb : { 0, 1, 2 }) {
                    if (n_rs == 0 && rb > 0) {
                        continue;
                    }
                    for (int64_t p0 : c.p0s) {
                        std::shared_ptr<llama_ubatch::data_t> data;
                        llama_ubatch ub = make_ubatch(data, R, p0);
                        const std::vector<uint32_t> rs_idx = { (uint32_t) rb };

                        const comp_plan whole = llama_dsv4_build_comp_plan(
                                ub, c.ratio, c.overlap, c.S, c.kv, 1, n_rs, rs_idx);
                        const llama_dsv4_comp_plan_split sp = llama_dsv4_split_comp_plan(
                                ub, c.ratio, c.overlap, c.S, c.kv, 1, n_rs, rs_idx, k);

                        sim want = make_sim(c.S, n_rs, R, c.kv, p0, false, k);
                        apply_whole(want, whole, c.ratio, c.overlap);

                        sim got = make_sim(c.S, n_rs, R, c.kv, p0, true, k);
                        apply_group_a(got, sp.a, c.ratio, c.overlap, k, 0);
                        set_b_rows(got, p0, k);
                        apply_group_b(got, sp.b, c.ratio, c.overlap, k);
                        compare(want, got, "partitioned vs whole");

                        // Negative check: persist before snapshot must change the
                        // planes for some case (aggregated below).
                        if (n_rs > 0) {
                            sim bad = make_sim(c.S, n_rs, R, c.kv, p0, true, k);
                            apply_group_a(bad, sp.a, c.ratio, c.overlap, k, 1);
                            set_b_rows(bad, p0, k);
                            apply_group_b(bad, sp.b, c.ratio, c.overlap, k);
                            for (int32_t r = 0; r < c.S*(1 + n_rs); ++r) {
                                if (bad.X[r] != want.X[r]) {
                                    ++n_order_neg;
                                    ++cfg_neg[ci];
                                    break;
                                }
                            }
                        }

                        // Fixed rank per group: A has ceil(k/ratio) write blocks and
                        // min(k,S) persist rows; B has ceil((R-k)/ratio) and min(R-k,S).
                        const size_t rank_a = (k + c.ratio - 1) / c.ratio;
                        const size_t rank_b = (R - k + c.ratio - 1) / c.ratio;
                        if (sp.a.state_write_idxs.size() != rank_a || sp.b.state_write_idxs.size() != rank_b) {
                            fail("write rank is not the per-group reserve rank");
                        }
                        if (sp.a.state_persist_dst_idxs.size() != std::min<size_t>(k, c.S) ||
                                sp.b.state_persist_dst_idxs.size() != std::min<size_t>(R - k, c.S)) {
                            fail("persist size is not min(group rows, S)");
                        }
                        if (sp.a.state_pos.size() != (size_t) k || sp.b.state_pos.size() != (size_t) (R - k)) {
                            fail("state_pos slice has the wrong size");
                        }
                        ++n_cases;
                        ++cfg_cases[ci];
                    }
                }
            }
        }
    }

    if (n_cases == 0) {
        fail("no cases ran");
    }
    if (n_order_neg == 0) {
        fail("persist-before-snapshot never changed the planes: the negative check is blind");
    }
    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        std::printf("  cfg %zu (ratio=%u overlap=%d S=%d): %ld cases, persist-before-snapshot detected in %ld\n",
                    ci, cfgs[ci].ratio, (int) cfgs[ci].overlap, cfgs[ci].S, cfg_cases[ci], cfg_neg[ci]);
        if (cfg_cases[ci] == 0) {
            fail("config ran no cases");
        }
    }
    // The DS4.1 real configs (the last four entries) must each be covered by the negative check too.
    for (size_t ci = cfgs.size() - 4; ci < cfgs.size(); ++ci) {
        if (cfg_neg[ci] == 0) {
            fail("negative check is blind for a DS4.1 config");
        }
    }
    const long n_supported = run_supported_checks();
    std::printf("PASS: %ld partitioned-vs-whole cases; persist-before-snapshot detected in %ld cases; "
                "%ld split_supported checks\n",
                n_cases, n_order_neg, n_supported);
    return 0;
}
