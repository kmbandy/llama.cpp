// T5: WP_SPINE_ROW_GROUPS (S4) compressor state hook. CPU only, no model.
//
// Group A stashes its state_kv/state_score rows for a layer. Group B builds its
// compressor source as concat(ring, concat(stash_A, state_B), zero) and its
// snapshot source as concat(all rows, concat(stash_A, state_B)), the same
// expressions build_attention_v41 emits. The S2 split plans index those
// concatenations with whole-ubatch row ids. Checked on the ggml CPU backend:
//   - the row-sliced projections, concatenated, equal the whole projection;
//   - B's read rows (prev and cur halves) gathered through the S4 source equal
//     the whole-ubatch plan's gathered rows for the same blocks;
//   - A's read rows, with the zero sentinel remapped to S+k, equal the whole
//     plan's rows for the same blocks;
//   - B's snapshot (src,dst) pairs are pairs of the whole plan and gather equal rows;
//   - persist pairs of A and B partition the whole persist list.

#include "ggml.h"
#include "ggml-cpu.h"
#include "llama-batch.h"
#include "llama-kv-cache-dsv4.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using comp_plan = llama_kv_cache_dsv4_context::comp_plan;

static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
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

// ggml [ne0, ne1] f32 tensor from row-major host data (ne0 is the row width).
static ggml_tensor * f32_t(ggml_context * ctx, int64_t ne0, int64_t ne1, const std::vector<float> & v) {
    if ((int64_t) v.size() != ne0*ne1) fail("f32_t size");
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    std::memcpy(t->data, v.data(), v.size()*sizeof(float));
    return t;
}

static ggml_tensor * i32_t(ggml_context * ctx, const std::vector<int32_t> & v) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) v.size());
    std::memcpy(t->data, v.data(), v.size()*sizeof(int32_t));
    return t;
}

static std::vector<float> data_of(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    std::memcpy(v.data(), t->data, v.size()*sizeof(float));
    return v;
}

// Row r (width d) of a row-major [d, rows] buffer.
static std::vector<float> row_of(const std::vector<float> & buf, int64_t d, int64_t r) {
    return std::vector<float>(buf.begin() + r*d, buf.begin() + (r + 1)*d);
}

static bool same_row(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i])) return false; // exact, -inf included
    }
    return true;
}

int main() {
    const int64_t d   = 8;   // state width per row
    const int64_t din = 8;   // projection input width

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    auto rnd = [&](size_t n) {
        std::vector<float> v(n);
        for (auto & x : v) x = U(rng);
        return v;
    };

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
    };

    long n_cases = 0;
    long n_b_blocks = 0;
    long n_b_snap = 0;

    for (const cfg & c : cfgs) {
        for (int32_t R = 4; R <= 8; ++R) {
            const int32_t k = (R + 1) / 2;
            for (int32_t n_rs : { 0, 2 }) {
                for (int64_t p0 : c.p0s) {
                    std::shared_ptr<llama_ubatch::data_t> data;
                    llama_ubatch ub = make_ubatch(data, R, p0);
                    const std::vector<uint32_t> rs_idx = { 0 };

                    const comp_plan whole = llama_dsv4_build_comp_plan(
                            ub, c.ratio, c.overlap, c.S, c.kv, 1, n_rs, rs_idx);
                    const llama_dsv4_comp_plan_split sp = llama_dsv4_split_comp_plan(
                            ub, c.ratio, c.overlap, c.S, c.kv, 1, n_rs, rs_idx, k);

                    const int32_t S = c.S;
                    const int64_t n_all = (int64_t) S*(1 + n_rs);

                    const std::vector<float> ring_kv = rnd(d*S), ring_sc = rnd(d*S);
                    const std::vector<float> all_kv  = rnd(d*n_all), all_sc = rnd(d*n_all);
                    const std::vector<float> x       = rnd(din*R);
                    const std::vector<float> Wkv     = rnd(d*din), Wsc = rnd(d*din);
                    const std::vector<float> zero_kv(d, 0.0f);
                    const std::vector<float> zero_sc(d, -std::numeric_limits<float>::infinity());

                    ggml_init_params ip = {};
                    ip.mem_size   = 128u*1024u*1024u;
                    ip.mem_buffer = nullptr;
                    ip.no_alloc   = false;
                    ggml_context * ctx = ggml_init(ip);
                    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

                    // Projections: whole rows, and the A (rows [0,k)) and B (rows [k,R)) slices.
                    ggml_tensor * Mkv = f32_t(ctx, din, d, Wkv);
                    ggml_tensor * Msc = f32_t(ctx, din, d, Wsc);
                    ggml_tensor * xw  = f32_t(ctx, din, R, x);
                    ggml_tensor * xa  = f32_t(ctx, din, k, std::vector<float>(x.begin(), x.begin() + din*k));
                    ggml_tensor * xb  = f32_t(ctx, din, R - k, std::vector<float>(x.begin() + din*k, x.end()));
                    ggml_tensor * kv_w = ggml_mul_mat(ctx, Mkv, xw), * sc_w = ggml_mul_mat(ctx, Msc, xw);
                    ggml_tensor * kv_a = ggml_mul_mat(ctx, Mkv, xa), * sc_a = ggml_mul_mat(ctx, Msc, xa);
                    ggml_tensor * kv_b = ggml_mul_mat(ctx, Mkv, xb), * sc_b = ggml_mul_mat(ctx, Msc, xb);

                    // S4 expressions, as build_attention_v41 emits them for group B.
                    ggml_tensor * kv_b_full = ggml_concat(ctx, kv_a, kv_b, 1);
                    ggml_tensor * sc_b_full = ggml_concat(ctx, sc_a, sc_b, 1);

                    // Compressor source: [ring S | state rows | zero row]. The overlap builder
                    // appends the zero row itself, at index S + rows.
                    auto src_of = [&](ggml_tensor * ring, ggml_tensor * st, const std::vector<float> & zero) {
                        ggml_tensor * z = f32_t(ctx, d, 1, zero);
                        return ggml_concat(ctx, ggml_concat(ctx, ring, st, 1), z, 1);
                    };
                    ggml_tensor * ring_kv_t = f32_t(ctx, d, S, ring_kv);
                    ggml_tensor * ring_sc_t = f32_t(ctx, d, S, ring_sc);
                    ggml_tensor * src_w_kv = src_of(ring_kv_t, kv_w, zero_kv);
                    ggml_tensor * src_w_sc = src_of(ring_sc_t, sc_w, zero_sc);
                    ggml_tensor * src_a_kv = src_of(ring_kv_t, kv_a, zero_kv);
                    ggml_tensor * src_a_sc = src_of(ring_sc_t, sc_a, zero_sc);
                    ggml_tensor * src_b_kv = src_of(ring_kv_t, kv_b_full, zero_kv);
                    ggml_tensor * src_b_sc = src_of(ring_sc_t, sc_b_full, zero_sc);

                    // Snapshot sources: [all rows | state rows].
                    ggml_tensor * all_kv_t = f32_t(ctx, d, n_all, all_kv);
                    ggml_tensor * all_sc_t = f32_t(ctx, d, n_all, all_sc);
                    ggml_tensor * snap_w_kv = ggml_concat(ctx, all_kv_t, kv_w, 1);
                    ggml_tensor * snap_w_sc = ggml_concat(ctx, all_sc_t, sc_w, 1);
                    ggml_tensor * snap_b_kv = ggml_concat(ctx, all_kv_t, kv_b_full, 1);
                    ggml_tensor * snap_b_sc = ggml_concat(ctx, all_sc_t, sc_b_full, 1);

                    // Gathers through the plans' own index lists.
                    ggml_tensor * g_w_kv = ggml_get_rows(ctx, src_w_kv, i32_t(ctx, whole.state_read_idxs));
                    ggml_tensor * g_w_sc = ggml_get_rows(ctx, src_w_sc, i32_t(ctx, whole.state_read_idxs));
                    ggml_tensor * g_a_kv = ggml_get_rows(ctx, src_a_kv, i32_t(ctx, sp.a.state_read_idxs));
                    ggml_tensor * g_a_sc = ggml_get_rows(ctx, src_a_sc, i32_t(ctx, sp.a.state_read_idxs));
                    ggml_tensor * g_b_kv = ggml_get_rows(ctx, src_b_kv, i32_t(ctx, sp.b.state_read_idxs));
                    ggml_tensor * g_b_sc = ggml_get_rows(ctx, src_b_sc, i32_t(ctx, sp.b.state_read_idxs));
                    ggml_tensor * g_snap_w = ggml_get_rows(ctx, snap_w_kv, i32_t(ctx, whole.state_snapshot_src_idxs));
                    ggml_tensor * g_snap_b = ggml_get_rows(ctx, snap_b_kv, i32_t(ctx, sp.b.state_snapshot_src_idxs));
                    ggml_tensor * g_snap_wsc = ggml_get_rows(ctx, snap_w_sc, i32_t(ctx, whole.state_snapshot_src_idxs));
                    ggml_tensor * g_snap_bsc = ggml_get_rows(ctx, snap_b_sc, i32_t(ctx, sp.b.state_snapshot_src_idxs));

                    const ggml_tensor * outs[] = {
                        kv_w, sc_w, kv_a, sc_a, kv_b, sc_b, kv_b_full, sc_b_full,
                        g_w_kv, g_w_sc, g_a_kv, g_a_sc, g_b_kv, g_b_sc, g_snap_w, g_snap_b, g_snap_wsc, g_snap_bsc,
                    };
                    for (const ggml_tensor * t : outs) {
                        ggml_build_forward_expand(gf, const_cast<ggml_tensor *>(t));
                    }
                    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
                        fail("graph compute");
                    }

                    // Projection premise: the B-side concat equals the whole projection (rows are independent).
                    const std::vector<float> wkv_full = data_of(kv_w), wsc_full = data_of(sc_w);
                    const std::vector<float> cat_kv = data_of(kv_b_full), cat_sc = data_of(sc_b_full);
                    for (size_t i = 0; i < cat_kv.size(); ++i) {
                        if (std::fabs(cat_kv[i] - wkv_full[i]) > 1e-5f || std::fabs(cat_sc[i] - wsc_full[i]) > 1e-5f) {
                            fail("row-sliced projection differs from whole projection");
                        }
                    }

                    const std::vector<float> o_w_kv = data_of(g_w_kv), o_w_sc = data_of(g_w_sc);
                    const std::vector<float> o_a_kv = data_of(g_a_kv), o_a_sc = data_of(g_a_sc);
                    const std::vector<float> o_b_kv = data_of(g_b_kv), o_b_sc = data_of(g_b_sc);
                    const std::vector<float> o_sw   = data_of(g_snap_w), o_swsc = data_of(g_snap_wsc);
                    const std::vector<float> o_sb   = data_of(g_snap_b), o_sbsc = data_of(g_snap_bsc);

                    // Real blocks: block b completes at row block_row[b]; the whole plan lists them in
                    // the same order the split plans do, per group.
                    std::vector<int32_t> block_row;
                    for (int32_t i = 0; i < R; ++i) {
                        if ((ub.pos[i] + 1) % (int64_t) c.ratio == 0) block_row.push_back(i);
                    }
                    const size_t nb = block_row.size();
                    const size_t n_half_w = c.overlap ? whole.state_read_idxs.size()/2 : 0;
                    const size_t n_half_a = c.overlap ? sp.a.state_read_idxs.size()/2 : 0;
                    const size_t n_half_b = c.overlap ? sp.b.state_read_idxs.size()/2 : 0;

                    // Read-row comparison: entry (block, j) of a group's half h.
                    auto cmp_read = [&](const std::vector<float> & gk_want, const std::vector<float> & gs_want, size_t iw,
                                        const std::vector<float> & gk_got, const std::vector<float> & gs_got, size_t ig,
                                        const char * what) {
                        if (!same_row(row_of(gk_got, d, ig), row_of(gk_want, d, iw)) ||
                            !same_row(row_of(gs_got, d, ig), row_of(gs_want, d, iw))) {
                            std::fprintf(stderr, "%s\n", what);
                            fail("read rows differ from whole plan");
                        }
                    };

                    size_t ai = 0, bi = 0;
                    for (size_t b = 0; b < nb; ++b) {
                        const bool is_a = block_row[b] < k;
                        size_t & gi = is_a ? ai : bi;
                        const comp_plan & P = is_a ? sp.a : sp.b;
                        const size_t n_half_g = is_a ? n_half_a : n_half_b;
                        if (gi >= P.state_write_idxs.size()) fail("group block count");
                        if (P.state_write_idxs[gi] != whole.state_write_idxs[b]) fail("write index mismatch");
                        if (!is_a && P.state_write_pos[gi] != whole.state_write_pos[b]) fail("B write pos mismatch");

                        for (uint32_t j = 0; j < c.ratio; ++j) {
                            const size_t iw = (size_t) b*c.ratio + j;
                            const size_t ig = (size_t) gi*c.ratio + j;
                            if (is_a) {
                                cmp_read(o_w_kv, o_w_sc, iw, o_a_kv, o_a_sc, ig, "A prev");
                            } else {
                                cmp_read(o_w_kv, o_w_sc, iw, o_b_kv, o_b_sc, ig, "B prev");
                            }
                            if (c.overlap) {
                                const size_t iw2 = n_half_w + (size_t) b*c.ratio + j;
                                const size_t ig2 = n_half_g + (size_t) gi*c.ratio + j;
                                if (is_a) {
                                    cmp_read(o_w_kv, o_w_sc, iw2, o_a_kv, o_a_sc, ig2, "A cur");
                                } else {
                                    cmp_read(o_w_kv, o_w_sc, iw2, o_b_kv, o_b_sc, ig2, "B cur");
                                }
                            }
                        }
                        gi++;
                        if (!is_a) n_b_blocks++;
                    }
                    if (bi != nb - ai) fail("B real block count != whole blocks >= k");

                    // Snapshot: B's (src,dst) pairs are pairs of the whole plan, rows equal.
                    for (size_t j = 0; j < sp.b.state_snapshot_src_idxs.size(); ++j) {
                        const int32_t src = sp.b.state_snapshot_src_idxs[j];
                        const int32_t dst = sp.b.state_snapshot_dst_idxs[j];
                        if (src < n_all + k) {
                            continue; // ring/A-scratch sources, and self-copy padding (A phase / no-op)
                        }
                        bool found = false;
                        for (size_t q = 0; q < whole.state_snapshot_src_idxs.size(); ++q) {
                            if (whole.state_snapshot_dst_idxs[q] == dst && whole.state_snapshot_src_idxs[q] == src) {
                                found = true;
                                // rows gathered through the S4 source equal the whole source rows
                                if (!same_row(row_of(o_sb, d, j), row_of(o_sw, d, q)) ||
                                    !same_row(row_of(o_sbsc, d, j), row_of(o_swsc, d, q))) {
                                    fail("B snapshot row differs from whole");
                                }
                                break;
                            }
                        }
                        if (!found) fail("B snapshot (src,dst) not in whole plan");
                        n_b_snap++;
                    }

                    // Persist: A and B pairs partition the whole persist list.
                    std::map<std::pair<int32_t,int32_t>, int> left;
                    for (size_t j = 0; j < whole.state_persist_src_idxs.size(); ++j) {
                        left[{ whole.state_persist_src_idxs[j], whole.state_persist_dst_idxs[j] }]++;
                    }
                    size_t n_persist = 0;
                    for (const comp_plan * P : { &sp.a, &sp.b }) {
                        for (size_t j = 0; j < P->state_persist_src_idxs.size(); ++j) {
                            auto key = std::make_pair(P->state_persist_src_idxs[j], P->state_persist_dst_idxs[j]);
                            if (left[key]-- <= 0) fail("persist pair not in whole plan");
                            n_persist++;
                        }
                    }
                    if (n_persist != whole.state_persist_src_idxs.size()) fail("persist partition size");

                    ggml_free(ctx);
                    n_cases++;
                }
            }
        }
    }

    if (n_cases == 0 || n_b_blocks == 0) {
        fail("no cases ran");
    }
    std::printf("PASS: %ld cases, %ld B blocks, %ld B snapshot entries match the whole plan\n",
            n_cases, n_b_blocks, n_b_snap);
    return 0;
}
