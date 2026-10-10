// T2: WP_SPINE_ROW_GROUPS per-group DSv4 graph inputs. CPU only, no model.
//
// For single-sequence verify ubatches (R in 4..8, k = ceil(R/2)) the group
// inputs must equal the corresponding slices of the whole-ubatch fill:
//   - comp kq_mask rows: group A = whole rows [0,k), group B = whole rows [k,R)
//   - comp state_pos:    group A = whole state_pos[0,k), group B = [k,R)
//   - raw kq_mask rows:  group rows = whole rows [r0, r0+width) (byte copy)
// and every group tensor must be a dedicated leaf, never a view.
// Raw k_idxs are filled by the KV-cache fixture-free path only; no KV fixture
// exists, so that fill is covered by the code path but not by this test.

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache-dsv4.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Defined in src/llama-graph.cpp (test hooks).
void llm_graph_dsv4_build_comp_input(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const std::string & tag,
        const llama_cparams & cparams,
        int64_t n_stream);
void llm_graph_dsv4_fill_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream);
void llm_graph_dsv4_copy_kq_mask_rows(ggml_tensor * full, ggml_tensor * rows, uint32_t r0);

using comp_plan = llama_kv_cache_dsv4_context::comp_plan;
using comp_input = llm_graph_input_dsv4::comp_input;

[[noreturn]] static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

static void expect_i32_slice(const ggml_tensor * t, const std::vector<int32_t> & want, size_t off, size_t n, const char * what) {
    if (!t) {
        if (n != 0) {
            fail(what);
        }
        return;
    }
    if (t->ne[0] != (int64_t) n || (size_t) want.size() < off + n) {
        fail(what);
    }
    if (std::memcmp(t->data, want.data() + off, n*sizeof(int32_t)) != 0) {
        fail(what);
    }
}

// rows of `grp` (width rows) equal rows [r0, r0+width) of `full`
static void expect_rows_eq(const ggml_tensor * full, const ggml_tensor * grp, uint32_t r0, const char * what) {
    if (!full || !grp) {
        if (full != grp) {
            fail(what);
        }
        return;
    }
    if (full->type != grp->type || full->ne[0] != grp->ne[0] || full->ne[2] != grp->ne[2] || full->ne[3] != grp->ne[3]) {
        fail(what);
    }
    const size_t row_bytes = ggml_row_size(full->type, full->ne[0]);
    for (int64_t i = 0; i < grp->ne[1]; ++i) {
        const char * a = (const char *) full->data + (size_t) (r0 + i) * full->nb[1];
        const char * b = (const char *) grp->data  + (size_t) i * grp->nb[1];
        if (std::memcmp(a, b, row_bytes) != 0) {
            fail(what);
        }
    }
}

static void expect_dedicated(const ggml_tensor * grp, const ggml_tensor * whole, const char * what) {
    if (!grp) {
        return;
    }
    if (grp->view_src != nullptr || grp->data == whole->data || grp->buffer == nullptr) {
        fail(what);
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

struct tensors_holder {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    tensors_holder() {
        ggml_init_params p = {};
        p.mem_size = ggml_tensor_overhead()*4096 + ggml_graph_overhead();
        p.no_alloc = true;
        ctx = ggml_init(p);
        backend = ggml_backend_cpu_init();
    }
    void alloc() {
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fail("alloc failed");
        }
    }
    ~tensors_holder() {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
};

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
    };

    long n_cases = 0;
    long n_mask_rows = 0;

    for (const cfg & c : cfgs) {
        for (int flash = 0; flash < 2; ++flash) {
            llama_cparams cparams = {};
            cparams.flash_attn = flash != 0;

            for (int32_t R = 4; R <= 8; ++R) {
                const uint32_t k = (R + 1) / 2;
                for (int64_t p0 : c.p0s) {
                    for (int32_t n_rs : { 0, 2 }) {
                        std::vector<uint32_t> rs_idx(1, n_rs > 0 ? 1u : 0u);

                        std::shared_ptr<llama_ubatch::data_t> data;
                        const llama_ubatch ub = make_ubatch(data, R, p0);

                        const comp_plan whole = llama_dsv4_build_comp_plan(
                                ub, c.ratio, c.overlap, c.S, c.kv, 1, n_rs, rs_idx);
                        const llama_dsv4_comp_plan_split sp = llama_dsv4_split_comp_plan(
                                ub, c.ratio, c.overlap, c.S, c.kv, 1, n_rs, rs_idx, k);

                        if (whole.state_pos.size() != (size_t) R || whole.n_visible.size() != (size_t) R) {
                            fail("whole plan rank is not R");
                        }

                        tensors_holder th;
                        comp_input w_in, a_in, b_in;
                        llm_graph_dsv4_build_comp_input(th.ctx, w_in, whole, "csa", "", cparams, 1);
                        llm_graph_dsv4_build_comp_input(th.ctx, a_in, sp.a, "csa", "_gA", cparams, 1);
                        llm_graph_dsv4_build_comp_input(th.ctx, b_in, sp.b, "csa", "_gB", cparams, 1);

                        // raw kq_mask: whole [n_kv_raw, R] pattern, groups copy rows
                        const int64_t n_kv_raw = 24;
                        ggml_tensor * w_raw = ggml_new_tensor_4d(th.ctx, GGML_TYPE_F32, n_kv_raw, R, 1, 1);
                        ggml_tensor * a_raw = ggml_new_tensor_4d(th.ctx, GGML_TYPE_F32, n_kv_raw, k, 1, 1);
                        ggml_tensor * b_raw = ggml_new_tensor_4d(th.ctx, GGML_TYPE_F32, n_kv_raw, R - k, 1, 1);

                        th.alloc();

                        llm_graph_dsv4_fill_comp_input(w_in, whole, R, 1);
                        llm_graph_dsv4_fill_comp_input(a_in, sp.a, k, 1);
                        llm_graph_dsv4_fill_comp_input(b_in, sp.b, R - k, 1);

                        float * wd = (float *) w_raw->data;
                        for (int64_t i = 0; i < R; ++i) {
                            for (int64_t j = 0; j < n_kv_raw; ++j) {
                                wd[i*n_kv_raw + j] = (float) (1000*i + j);
                            }
                        }
                        llm_graph_dsv4_copy_kq_mask_rows(w_raw, a_raw, 0);
                        llm_graph_dsv4_copy_kq_mask_rows(w_raw, b_raw, k);

                        // comp kq_mask: group rows equal whole rows (A: [0,k), B: [k,R))
                        if (whole.n_kv > 0) {
                            expect_rows_eq(w_in.kq_mask, a_in.kq_mask, 0, "csa A kq_mask rows != whole rows [0,k)");
                            expect_rows_eq(w_in.kq_mask, b_in.kq_mask, k, "csa B kq_mask rows != whole rows [k,R)");
                            n_mask_rows += R;
                        }

                        // state_pos: split groups are the whole plan's row slices
                        expect_i32_slice(a_in.state_pos, whole.state_pos, 0, k, "csa A state_pos != whole[0,k)");
                        expect_i32_slice(b_in.state_pos, whole.state_pos, k, R - k, "csa B state_pos != whole[k,R)");

                        // group n_visible drives the mask; check it is the same slice
                        for (int32_t i = 0; i < R; ++i) {
                            const int32_t wv = whole.n_visible[i];
                            const int32_t gv = (uint32_t) i < k ? sp.a.n_visible[i] : sp.b.n_visible[i - k];
                            if (wv != gv) {
                                fail("group n_visible != whole n_visible slice");
                            }
                        }

                        // raw rows
                        expect_rows_eq(w_raw, a_raw, 0, "raw A kq_mask rows != whole rows [0,k)");
                        expect_rows_eq(w_raw, b_raw, k, "raw B kq_mask rows != whole rows [k,R)");

                        // dedicated leaves, never views of the whole-ubatch leaves
                        expect_dedicated(a_in.kq_mask, w_in.kq_mask, "csa A kq_mask is a view");
                        expect_dedicated(b_in.kq_mask, w_in.kq_mask, "csa B kq_mask is a view");
                        expect_dedicated(a_raw, w_raw, "raw A is a view");
                        expect_dedicated(b_raw, w_raw, "raw B is a view");

                        n_cases++;
                    }
                }
            }
        }
    }

    if (n_cases == 0 || n_mask_rows == 0) {
        fail("no cases ran");
    }
    std::printf("PASS: %ld group-vs-whole input cases, %ld comp mask rows compared\n", n_cases, n_mask_rows);
    return 0;
}
