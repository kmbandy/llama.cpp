// T6: WP_SPINE_ROW_GROUPS (S7) engram row slicing. CPU only, no model.
//
// The engram lookup is gathered once for all R rows (get_rows over the table, reshaped to
// [key_len*n_cols, R]). Each row group then takes a row slice of that gather, runs its own wkv
// projection on it, and mixes into its own row slice of the stream. This test runs the production
// mix (dsv41-engram-mix.h, the same helper build_engram calls) on the ggml CPU backend and checks:
//   - the gathered rows equal the table rows named by the hash indices (exact);
//   - the per-group slices of the gather and of the wkv projection equal the whole-batch ones (exact);
//   - the per-group mix output, concatenated, equals the whole-batch mix output (max abs diff tiny);
//   - the engram actually changes the stream (so the check is not vacuous).
// The stream and projection inputs are computed tensors (ggml_scale of a leaf) before they are
// sliced, mirroring inpL in the trunk; slicing an input leaf is not what the model does.

#include "ggml.h"
#include "ggml-cpu.h"

#include "models/dsv41-engram-mix.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

static ggml_tensor * f32_2d(ggml_context * ctx, int64_t ne0, int64_t ne1, std::mt19937 & rng, float amp) {
    std::uniform_real_distribution<float> U(-amp, amp);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ne0*ne1; ++i) d[i] = U(rng);
    return t;
}

static ggml_tensor * f32_3d(ggml_context * ctx, int64_t ne0, int64_t ne1, int64_t ne2, std::mt19937 & rng, float amp) {
    std::uniform_real_distribution<float> U(-amp, amp);
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ne0*ne1*ne2; ++i) d[i] = U(rng);
    return t;
}

int main() {
    const int64_t n_embd   = 16;
    const int64_t hc       = 4;
    const int64_t key_len  = 8;   // engram head dim
    const int64_t n_cols   = 3;   // hash columns per token
    const int64_t E        = key_len * n_cols;
    const int64_t n_rows   = 50;  // engram table rows
    const float   eps      = 1e-6f;
    const float   tol      = 1e-5f;

    long n_cases = 0;
    double worst = 0.0;
    double max_effect = 0.0;

    for (int64_t R = 4; R <= 16; ++R) {
        const int64_t k = (R + 1) / 2;
        for (int seed = 0; seed < 4; ++seed) {
            std::mt19937 rng(1000 + 31*R + seed);

            const size_t mem = 64u * 1024u * 1024u;
            ggml_init_params ip = { mem, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) fail("ggml_init");

            ggml_tensor * table = f32_2d(ctx, key_len, n_rows, rng, 0.5f);
            ggml_tensor * wkv   = f32_2d(ctx, E, n_embd * (hc + 1), rng, 0.2f);
            ggml_tensor * w_k   = f32_2d(ctx, n_embd, hc, rng, 1.0f);
            ggml_tensor * w_q   = f32_2d(ctx, n_embd, hc, rng, 1.0f);

            // hash indices for all R rows, n_cols per row, laid out as build_inp_engram does
            ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cols * R);
            std::vector<int32_t> idx(n_cols * R);
            std::uniform_int_distribution<int32_t> I(0, (int32_t) n_rows - 1);
            for (auto & v : idx) v = I(rng);
            std::memcpy(rows->data, idx.data(), idx.size() * sizeof(int32_t));

            // stream leaf -> computed stream, as inpL is in the trunk
            ggml_tensor * x_leaf = f32_3d(ctx, n_embd, hc, R, rng, 1.0f);
            ggml_tensor * x_full = ggml_scale(ctx, x_leaf, 1.0f);

            // one gather for all rows: [key_len, n_cols*R] -> [key_len*n_cols, R]
            ggml_tensor * emb_full = ggml_reshape_2d(ctx, ggml_get_rows(ctx, table, rows), E, R);

            ggml_cgraph * gf = ggml_new_graph(ctx);

            // whole batch
            ggml_tensor * kv_full  = ggml_mul_mat(ctx, wkv, emb_full);
            ggml_tensor * gate_full = nullptr;
            ggml_tensor * out_full = dsv41_engram_mix(ctx, gf, x_full, kv_full, w_k, w_q, n_embd, hc, eps, &gate_full);
            ggml_build_forward_expand(gf, out_full);

            // per group: slice the gather, project, slice the stream, mix
            struct grp { int64_t r0, w; };
            const grp groups[2] = { { 0, k }, { k, R - k } };
            ggml_tensor * emb_g[2], * kv_g[2], * out_g[2];
            for (int g = 0; g < 2; ++g) {
                const int64_t r0 = groups[g].r0, w = groups[g].w;
                emb_g[g] = ggml_view_2d(ctx, emb_full, E, w, emb_full->nb[1], r0 * emb_full->nb[1]);
                kv_g[g]  = ggml_mul_mat(ctx, wkv, emb_g[g]);
                ggml_tensor * x_g = ggml_view_3d(ctx, x_full, n_embd, hc, w, x_full->nb[1], x_full->nb[2], r0 * x_full->nb[2]);
                out_g[g] = dsv41_engram_mix(ctx, gf, x_g, kv_g[g], w_k, w_q, n_embd, hc, eps, nullptr);
                ggml_build_forward_expand(gf, emb_g[g]);
                ggml_build_forward_expand(gf, kv_g[g]);
                ggml_build_forward_expand(gf, out_g[g]);
            }

            if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
                fail("graph compute");
            }

            // gather equals the table rows named by the indices
            const float * tb = (const float *) table->data;
            const float * eb = (const float *) emb_full->data;
            for (int64_t t = 0; t < R; ++t) {
                for (int64_t c = 0; c < n_cols; ++c) {
                    const int32_t r = idx[t * n_cols + c];
                    for (int64_t j = 0; j < key_len; ++j) {
                        if (eb[t * E + c * key_len + j] != tb[(int64_t) r * key_len + j]) fail("gather != table rows");
                    }
                }
            }

            // per-group slices of the gather, the projection and the mix equal the whole-batch rows
            const float * xf = (const float *) out_full->data;
            const int64_t row_out = n_embd * hc;
            for (int g = 0; g < 2; ++g) {
                const int64_t r0 = groups[g].r0, w = groups[g].w;
                const float * eg = (const float *) emb_g[g]->data;
                for (int64_t i = 0; i < w * E; ++i) {
                    if (eg[i] != eb[r0 * E + i]) fail("group gather slice != whole gather slice");
                }
                const float * kg = (const float *) kv_g[g]->data;
                const float * kf = (const float *) kv_full->data;
                const int64_t kw = n_embd * (hc + 1);
                for (int64_t i = 0; i < w * kw; ++i) {
                    if (kg[i] != kf[r0 * kw + i]) fail("group wkv slice != whole wkv slice");
                }
                const float * og = (const float *) out_g[g]->data;
                for (int64_t i = 0; i < w * row_out; ++i) {
                    const double d = std::fabs((double) og[i] - (double) xf[r0 * row_out + i]);
                    if (d > worst) worst = d;
                    if (d > tol) fail("group mix slice != whole mix slice");
                }
            }

            // the engram changes the stream
            const float * xs = (const float *) x_full->data;
            for (int64_t i = 0; i < R * row_out; ++i) {
                const double d = std::fabs((double) xf[i] - (double) xs[i]);
                if (d > max_effect) max_effect = d;
            }

            ggml_free(ctx);
            n_cases++;
        }
    }

    if (n_cases == 0) fail("no cases ran");
    if (!(max_effect > 1e-3)) fail("engram had no effect on the stream (vacuous test)");
    std::printf("PASS: %ld cases (R=4..16, 2 groups): gather and wkv slices exact, mix slices within %.3g (worst %.3g), engram effect %.3g\n",
            n_cases, (double) tol, worst, max_effect);
    return 0;
}
