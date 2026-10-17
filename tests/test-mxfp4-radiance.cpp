// test-mxfp4-radiance.cpp -- MXFP4 MUL_MAT / MUL_MAT_ID on rdna4 radiance-packed weights (ggml-cuda, gfx1201).
//
// Runs on the R9700 only (selected by device description; the 9070 XT is also gfx1201 and is never used).
// Each case builds a flagged weight (GGML_TENSOR_FLAG_MXFP4_RADIANCE, packed in place via the backend's
// repack) and an unflagged stock MXFP4 copy of the same bytes, runs both on the device, and reports the
// relative RMS error of each against an fp64 CPU reference built from the dequantized MXFP4 weights and
// the fp32 activations. PASS iff radiance rel RMS <= stock rel RMS and <= 0.002.
//
// Reference sampling: the fp64 reference is evaluated on at most 64 output rows (MUL_MAT: activation rows;
// MUL_MAT_ID: routed (slot, token) entries), evenly spaced; every output column of a sampled row is checked.
//
// The radiance entry points live in libggml-hip, which ggml_backend_load_all() loads dynamically, so they
// are looked up by name through the backend registry.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kBound = 0.002;
constexpr int64_t kMaxRefSamples = 64;

using repack_fn       = bool (*)(ggml_backend_t, const void *, void *, int64_t, int64_t);
using packed_bytes_fn = size_t (*)(int64_t, int64_t);
using unpack_fn       = bool (*)(ggml_backend_t, const void *, void *, int64_t, int64_t);
using stage_reader_fn = bool (*)(ggml_backend_t, void *, size_t, int64_t, int64_t);
using unstage_fn      = bool (*)(ggml_backend_t, const void *, void *, size_t, int64_t, int64_t, bool);

struct env_t {
    ggml_backend_dev_t        dev    = nullptr;
    ggml_backend_t            be     = nullptr;
    ggml_backend_buffer_type_t buft  = nullptr;
    repack_fn                 repack = nullptr;
    packed_bytes_fn           packed = nullptr;
    unpack_fn                 unpack = nullptr;
    stage_reader_fn           stage_reader = nullptr;
    unstage_fn                unstage = nullptr;
    std::mt19937              rng{ 1234 };
};
env_t E;

struct err_acc {
    double e2 = 0, r2 = 0;
    void add(double got, double ref) { const double d = got - ref; e2 += d * d; r2 += ref * ref; }
    double rel() const { return std::sqrt(e2 / (r2 + 1e-300)); }
};

struct row_result {
    std::string desc;
    double rel_rad   = NAN;
    double rel_stock = NAN;
    bool   skip      = false;
    bool   pass      = false;
    std::string why;
};

// rows of N-wide weights with a per-row scale spread (as in gemm_mxfp4_radiance_bench)
void fill_weights(std::vector<float> & w, int64_t N, int64_t K) {
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int64_t n = 0; n < N; ++n) {
        const float rs = 0.02f * std::exp2((float) (n % 7) - 3.f);
        for (int64_t k = 0; k < K; ++k) {
            w[(size_t) (n * K + k)] = nd(E.rng) * rs;
        }
    }
}

// quantize [N, K] floats to ggml MXFP4 and dequantize back (the fp64 reference uses the dequantized values)
void quantize_mxfp4(const float * w, int64_t N, int64_t K, std::vector<uint8_t> & q) {
    q.resize((size_t) (ggml_row_size(GGML_TYPE_MXFP4, K) * N));
    ggml_quantize_chunk(GGML_TYPE_MXFP4, w, q.data(), 0, N, K, nullptr);
}

void dequantize_mxfp4(const uint8_t * q, int64_t N, int64_t K, std::vector<float> & d) {
    d.resize((size_t) (N * K));
    ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(q, d.data(), N * K);
}

bool init_device() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            strstr(ggml_backend_dev_description(d), "R9700") != nullptr) {
            E.dev = d;
            break;
        }
    }
    if (!E.dev) {
        fprintf(stderr, "no R9700 GPU device found; refusing to run on any other device\n");
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(E.dev);
    E.repack = (repack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mxfp4_radiance_repack");
    E.packed = (packed_bytes_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mxfp4_radiance_packed_bytes");
    E.unpack = (unpack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mxfp4_radiance_unpack");
    E.stage_reader = (stage_reader_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_wp_mxfp4_radiance_stage_reader");
    E.unstage = (unstage_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_wp_mxfp4_radiance_unstage");
    if (!E.repack || !E.packed || !E.unpack || !E.stage_reader || !E.unstage) {
        fprintf(stderr, "radiance MXFP4 entry points not exported by the CUDA/HIP backend\n");
        return false;
    }
    E.be   = ggml_backend_dev_init(E.dev, nullptr);
    E.buft = ggml_backend_dev_buffer_type(E.dev);
    if (!E.be || !E.buft) {
        fprintf(stderr, "failed to init R9700 backend\n");
        return false;
    }
    printf("device: %s\n", ggml_backend_dev_description(E.dev));
    return true;
}

// Allocates the flagged weight's storage (pbytes, device) for tensor t and returns the buffer.
ggml_backend_buffer_t alloc_flagged(ggml_tensor * t, size_t pbytes) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(E.buft, pbytes);
    if (buf) {
        ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf));
    }
    return buf;
}

// ---- MUL_MAT: Y[M, N] = X[M, K] . W[N, K]^T ----
row_result mul_mat_case(int64_t N, int64_t K, int64_t M) {
    row_result r;
    char buf[128];
    snprintf(buf, sizeof buf, "MUL_MAT  N=%lld K=%lld M=%lld", (long long) N, (long long) K, (long long) M);
    r.desc = buf;

    const size_t pbytes = E.packed(N, K);
    if (!pbytes) { r.skip = true; r.why = "shape not supported"; return r; }

    std::vector<float> wf((size_t) (N * K)), x((size_t) (M * K));
    fill_weights(wf, N, K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(E.rng);
    // TEST_OUTLIER=<mult>: 8 massive-activation channels per row (real transformer hidden states have them)
    if (const char * om = std::getenv("TEST_OUTLIER")) {
        const float mult = (float) std::atof(om);
        for (int64_t m = 0; m < M; ++m)
            for (int c = 0; c < 8; ++c) x[(size_t) (m * K + (c * 641 + 17) % K)] *= mult;
    }
    std::vector<uint8_t> wq;
    quantize_mxfp4(wf.data(), N, K, wq);
    std::vector<float> wd;
    dequantize_mxfp4(wq.data(), N, K, wd);
    const size_t qsz = wq.size();

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * Ws = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, K, N);   // stock block_mxfp4 bytes
    ggml_tensor * Wf = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, K, N);   // radiance-packed bytes
    Wf->flags |= GGML_TENSOR_FLAG_MXFP4_RADIANCE;
    ggml_tensor * X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * Ys = ggml_mul_mat(ctx, Ws, X);
    ggml_tensor * Yf = ggml_mul_mat(ctx, Wf, X);

    ggml_backend_buffer_t buf_f = alloc_flagged(Wf, pbytes);
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx, E.be);
    if (!buf_f || !buf_c) {
        r.why = "device allocation failed";
        ggml_free(ctx);
        if (buf_f) ggml_backend_buffer_free(buf_f);
        if (buf_c) ggml_backend_buffer_free(buf_c);
        return r;
    }

    ggml_backend_tensor_set(Ws, wq.data(), 0, qsz);
    ggml_backend_tensor_set(X, x.data(), 0, x.size() * sizeof(float));
    const bool packed_ok = E.repack(E.be, Ws->data, Wf->data, N, K);
    ggml_backend_synchronize(E.be);
    if (!packed_ok) {
        r.why = "repack declined";
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Ys);
    ggml_build_forward_expand(gf, Yf);
    const ggml_status st = ggml_backend_graph_compute(E.be, gf);
    ggml_backend_synchronize(E.be);

    std::vector<float> ys((size_t) (M * N)), yf((size_t) (M * N));
    ggml_backend_tensor_get(Ys, ys.data(), 0, ys.size() * sizeof(float));
    ggml_backend_tensor_get(Yf, yf.data(), 0, yf.size() * sizeof(float));

    err_acc ea_rad, ea_stock;
    const int64_t step = std::max<int64_t>(1, (M + kMaxRefSamples - 1) / kMaxRefSamples);
    for (int64_t m = 0; m < M; m += step) {
        for (int64_t n = 0; n < N; ++n) {
            double ref = 0;
            const float * wr = &wd[(size_t) (n * K)];
            const float * xr = &x[(size_t) (m * K)];
            for (int64_t k = 0; k < K; ++k) ref += (double) wr[k] * (double) xr[k];
            ea_rad.add(yf[(size_t) (m * N + n)], ref);
            ea_stock.add(ys[(size_t) (m * N + n)], ref);
        }
    }
    r.rel_rad = ea_rad.rel();
    r.rel_stock = ea_stock.rel();

    ggml_free(ctx);
    ggml_backend_buffer_free(buf_f);
    ggml_backend_buffer_free(buf_c);

    if (st != GGML_STATUS_SUCCESS) {
        r.why = "graph compute failed";
        return r;
    }
    r.pass = packed_ok && r.rel_rad <= r.rel_stock && r.rel_rad <= kBound;
    if (!r.pass && r.why.empty()) {
        r.why = r.rel_rad > kBound ? "radiance rel RMS above bound" : "radiance worse than stock";
    }
    return r;
}

// ---- MUL_MAT_ID: Y[N, n_used, n_tok] with expert e = ids[s, t]; n_as experts ----
row_result mul_mat_id_case(int64_t N, int64_t K, int64_t n_as, int64_t n_used, int64_t n_tok, bool bcast) {
    row_result r;
    char buf[160];
    snprintf(buf, sizeof buf, "MUL_MAT_ID N=%lld K=%lld n_as=%lld n_used=%lld n_tok=%lld src1_ne1=%s",
             (long long) N, (long long) K, (long long) n_as, (long long) n_used, (long long) n_tok,
             bcast ? "1" : "n_used");
    r.desc = buf;

    const size_t pbytes = E.packed(N, K);
    if (!pbytes) { r.skip = true; r.why = "shape not supported"; return r; }
    const size_t qsz    = (size_t) ggml_row_size(GGML_TYPE_MXFP4, K) * (size_t) N;
    const size_t stride = (pbytes + 255) & ~(size_t) 255;   // expert stride for the packed weights

    // expert weights, stock bytes for all experts
    std::vector<uint8_t> wq_all((size_t) n_as * qsz);
    std::vector<float> wf((size_t) (N * K));
    for (int64_t e = 0; e < n_as; ++e) {
        fill_weights(wf, N, K);
        std::vector<uint8_t> q;
        quantize_mxfp4(wf.data(), N, K, q);
        std::memcpy(wq_all.data() + (size_t) e * qsz, q.data(), qsz);
    }

    // routing: each token picks n_used distinct experts
    std::vector<int32_t> ids_h((size_t) (n_used * n_tok));
    std::vector<int32_t> perm((size_t) n_as);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t t = 0; t < n_tok; ++t) {
        std::shuffle(perm.begin(), perm.end(), E.rng);
        for (int64_t s = 0; s < n_used; ++s) ids_h[(size_t) (t * n_used + s)] = perm[(size_t) s];
    }

    const int64_t ne11 = bcast ? 1 : n_used;
    std::vector<float> x((size_t) (K * ne11 * n_tok));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(E.rng);

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * Ws  = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, N, n_as);  // stock, contiguous
    ggml_tensor * Wf  = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, N, n_as);  // radiance, stride nb[2]
    Wf->nb[2] = (size_t) stride;
    Wf->nb[3] = (size_t) n_as * stride;
    Wf->flags |= GGML_TENSOR_FLAG_MXFP4_RADIANCE;
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * X   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, n_tok);
    ggml_tensor * Ys  = ggml_mul_mat_id(ctx, Ws, X, ids);
    ggml_tensor * Yf  = ggml_mul_mat_id(ctx, Wf, X, ids);

    ggml_backend_buffer_t buf_f = alloc_flagged(Wf, (size_t) n_as * stride);
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx, E.be);
    if (!buf_f || !buf_c) {
        r.why = "device allocation failed";
        ggml_free(ctx);
        if (buf_f) ggml_backend_buffer_free(buf_f);
        if (buf_c) ggml_backend_buffer_free(buf_c);
        return r;
    }

    ggml_backend_tensor_set(Ws, wq_all.data(), 0, wq_all.size());
    ggml_backend_tensor_set(ids, ids_h.data(), 0, ids_h.size() * sizeof(int32_t));
    ggml_backend_tensor_set(X, x.data(), 0, x.size() * sizeof(float));
    bool packed_ok = true;
    for (int64_t e = 0; e < n_as && packed_ok; ++e) {
        packed_ok = E.repack(E.be, (const char *) Ws->data + (size_t) e * qsz,
                             (char *) Wf->data + (size_t) e * stride, N, K);
    }
    ggml_backend_synchronize(E.be);
    if (!packed_ok) {
        r.why = "repack declined";
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Ys);
    ggml_build_forward_expand(gf, Yf);
    const ggml_status st = ggml_backend_graph_compute(E.be, gf);
    ggml_backend_synchronize(E.be);

    const int64_t n_out = N * n_used * n_tok;
    std::vector<float> ys((size_t) n_out), yf((size_t) n_out);
    ggml_backend_tensor_get(Ys, ys.data(), 0, ys.size() * sizeof(float));
    ggml_backend_tensor_get(Yf, yf.data(), 0, yf.size() * sizeof(float));

    // fp64 reference on evenly spaced routed (slot, token) entries
    const int64_t n_entries = n_used * n_tok;
    const int64_t step = std::max<int64_t>(1, (n_entries + kMaxRefSamples - 1) / kMaxRefSamples);
    err_acc ea_rad, ea_stock;
    std::vector<float> wd;
    for (int64_t idx = 0; idx < n_entries; idx += step) {
        const int64_t s = idx % n_used;
        const int64_t t = idx / n_used;
        const int32_t e = ids_h[(size_t) (t * n_used + s)];
        const int64_t i11 = bcast ? 0 : s;
        dequantize_mxfp4(wq_all.data() + (size_t) e * qsz, N, K, wd);
        const float * xr = &x[(size_t) (K * (i11 + ne11 * t))];
        for (int64_t n = 0; n < N; ++n) {
            double ref = 0;
            const float * wr = &wd[(size_t) (n * K)];
            for (int64_t k = 0; k < K; ++k) ref += (double) wr[k] * (double) xr[k];
            const size_t o = (size_t) (n + N * (s + n_used * t));
            ea_rad.add(yf[o], ref);
            ea_stock.add(ys[o], ref);
        }
    }
    r.rel_rad = ea_rad.rel();
    r.rel_stock = ea_stock.rel();

    ggml_free(ctx);
    ggml_backend_buffer_free(buf_f);
    ggml_backend_buffer_free(buf_c);

    if (st != GGML_STATUS_SUCCESS) {
        r.why = "graph compute failed";
        return r;
    }
    r.pass = packed_ok && r.rel_rad <= r.rel_stock && r.rel_rad <= kBound;
    if (!r.pass && r.why.empty()) {
        r.why = r.rel_rad > kBound ? "radiance rel RMS above bound" : "radiance worse than stock";
    }
    return r;
}


// ---- exact round trip: stock bytes -> repack -> unpack == stock bytes (arbitrary 17-byte blocks) ----
row_result roundtrip_case(int64_t N, int64_t K) {
    row_result r;
    char buf[128];
    snprintf(buf, sizeof buf, "ROUNDTRIP repack->unpack N=%lld K=%lld", (long long) N, (long long) K);
    r.desc = buf;
    const size_t pbytes = E.packed(N, K);
    if (!pbytes) { r.skip = true; r.why = "shape not supported"; return r; }
    const size_t qsz = (size_t) ggml_row_size(GGML_TYPE_MXFP4, K) * (size_t) N;
    std::vector<uint8_t> src(qsz), back(qsz, 0xAB);
    std::uniform_int_distribution<int> bd(0, 255);
    for (auto & b : src) b = (uint8_t) bd(E.rng);

    ggml_backend_buffer_t bs = ggml_backend_buft_alloc_buffer(E.buft, qsz);
    ggml_backend_buffer_t bp = ggml_backend_buft_alloc_buffer(E.buft, pbytes);
    ggml_backend_buffer_t bd2 = ggml_backend_buft_alloc_buffer(E.buft, qsz);
    if (!bs || !bp || !bd2) { r.why = "device allocation failed"; return r; }
    void * ps = ggml_backend_buffer_get_base(bs);
    void * pp = ggml_backend_buffer_get_base(bp);
    void * pd = ggml_backend_buffer_get_base(bd2);
    // ggml_backend_tensor_set needs a tensor; use a throwaway 1-D i8 view per buffer
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ts = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) qsz);
    ggml_tensor * td = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) qsz);
    ggml_backend_tensor_alloc(bs, ts, ps);
    ggml_backend_tensor_alloc(bd2, td, pd);
    ggml_backend_tensor_set(ts, src.data(), 0, qsz);
    const bool ok = E.repack(E.be, ps, pp, N, K) && E.unpack(E.be, pp, pd, N, K);
    ggml_backend_synchronize(E.be);
    ggml_backend_tensor_get(td, back.data(), 0, qsz);
    r.pass = ok && src == back;
    size_t ndiff = 0;
    for (size_t i = 0; i < qsz; ++i) ndiff += src[i] != back[i];
    r.rel_rad = r.rel_stock = (double) ndiff;   // reported as the number of differing bytes
    if (!r.pass) r.why = ok ? "bytes differ after round trip" : "repack/unpack declined";
    ggml_free(ctx);
    ggml_backend_buffer_free(bs);
    ggml_backend_buffer_free(bp);
    ggml_backend_buffer_free(bd2);
    return r;
}

// ---- worker-like MUL_MAT_ID: n_as=256 experts resident in ONE 3D tensor, shared across n_tok cases ----
struct moe_fixture {
    int64_t N = 0, K = 0, n_as = 0;
    size_t  qsz = 0, stride = 0;
    static constexpr int kDistinct = 32;                // distinct expert weight sets, expert e uses e % kDistinct
    std::vector<std::vector<uint8_t>> wq;               // stock bytes of the distinct sets
    ggml_context *        ctx = nullptr;
    ggml_tensor *         Ws = nullptr, * Wf = nullptr;
    ggml_backend_buffer_t buf_s = nullptr, buf_f = nullptr;
    bool ok = false;
    // use_ptrs: expert e's packed matrix lives at a scrambled, irregularly spaced slot and is located through the weight
    // pager's expert pointer table (ggml_mul_mat_id_set_expert_ptrs) instead of base + e*nb[2] (the worker's batch-mmid).
    bool use_ptrs = false;
    ggml_context *        ctx_p = nullptr;
    ggml_tensor *         ptr_t = nullptr;
    ggml_backend_buffer_t buf_p = nullptr;
    std::vector<size_t>   slot_off;

    bool init(int64_t N_, int64_t K_, int64_t n_as_, bool use_ptrs_ = false) {
        use_ptrs = use_ptrs_;
        N = N_; K = K_; n_as = n_as_;
        const size_t pbytes = E.packed(N, K);
        if (!pbytes) return false;
        qsz    = (size_t) ggml_row_size(GGML_TYPE_MXFP4, K) * (size_t) N;
        stride = (pbytes + 255) & ~(size_t) 255;
        std::vector<float> wf((size_t) (N * K));
        wq.resize(kDistinct);
        for (int d = 0; d < kDistinct; ++d) {
            fill_weights(wf, N, K);
            quantize_mxfp4(wf.data(), N, K, wq[(size_t) d]);
        }
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        Ws = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, N, n_as);
        Wf = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, N, n_as);
        Wf->nb[2] = stride;
        Wf->nb[3] = (size_t) n_as * stride;
        Wf->flags |= GGML_TENSOR_FLAG_MXFP4_RADIANCE;
        buf_s = ggml_backend_buft_alloc_buffer(E.buft, (size_t) n_as * qsz + 65536);
        const size_t slot_stride = use_ptrs ? stride + 8192 : stride;     // gaps between slots in ptr mode
        buf_f = ggml_backend_buft_alloc_buffer(E.buft, (size_t) n_as * slot_stride + 65536);
        if (!buf_s || !buf_f) return false;
        slot_off.resize((size_t) n_as);
        {
            std::vector<int64_t> perm((size_t) n_as);
            std::iota(perm.begin(), perm.end(), 0);
            if (use_ptrs) std::shuffle(perm.begin(), perm.end(), E.rng);
            for (int64_t e = 0; e < n_as; ++e) slot_off[(size_t) e] = (size_t) perm[(size_t) e] * slot_stride;
        }
        ggml_backend_tensor_alloc(buf_s, Ws, ggml_backend_buffer_get_base(buf_s));
        ggml_backend_tensor_alloc(buf_f, Wf, ggml_backend_buffer_get_base(buf_f));
        bool packed_ok = true;
        for (int64_t e = 0; e < n_as; ++e) {
            ggml_backend_tensor_set(Ws, wq[(size_t) (e % kDistinct)].data(), (size_t) e * qsz, qsz);
        }
        for (int64_t e = 0; e < n_as && packed_ok; ++e) {
            packed_ok = E.repack(E.be, (const char *) Ws->data + (size_t) e * qsz,
                                 (char *) Wf->data + slot_off[(size_t) e], N, K);
        }
        ggml_backend_synchronize(E.be);
        if (use_ptrs) {
            ctx_p = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
            ptr_t = ggml_new_tensor_1d(ctx_p, GGML_TYPE_I64, n_as);
            buf_p = ggml_backend_alloc_ctx_tensors(ctx_p, E.be);
            std::vector<int64_t> ptrs((size_t) n_as);
            for (int64_t e = 0; e < n_as; ++e) ptrs[(size_t) e] = (int64_t) (uintptr_t) ((char *) Wf->data + slot_off[(size_t) e]);
            ggml_backend_tensor_set(ptr_t, ptrs.data(), 0, ptrs.size() * sizeof(int64_t));
            Wf->nb[2] = stride;      // placeholder stride: still validated by supports_op, the table is what locates experts
        }
        ok = packed_ok;
        return ok;
    }
    ~moe_fixture() {
        if (ctx_p) ggml_free(ctx_p);
        if (buf_p) ggml_backend_buffer_free(buf_p);
        if (ctx) ggml_free(ctx);
        if (buf_s) ggml_backend_buffer_free(buf_s);
        if (buf_f) ggml_backend_buffer_free(buf_f);
    }
};

struct moe_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * ids = nullptr, * X = nullptr, * Ys = nullptr, * Yf = nullptr;
    ggml_cgraph * gs = nullptr, * gf = nullptr;
    std::vector<int32_t> ids_h;
    std::vector<float>   x;
    ~moe_graph() {
        if (ctx) ggml_free(ctx);
        if (buf) ggml_backend_buffer_free(buf);
    }
};

bool build_moe_graph(moe_graph & g, moe_fixture & fx, int64_t n_used, int64_t n_tok, bool bcast) {
    g.ids_h.resize((size_t) (n_used * n_tok));
    std::vector<int32_t> perm((size_t) fx.n_as);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t t = 0; t < n_tok; ++t) {
        // partial Fisher-Yates: n_used distinct experts per token
        for (int64_t s = 0; s < n_used; ++s) {
            std::uniform_int_distribution<int64_t> d(s, fx.n_as - 1);
            std::swap(perm[(size_t) s], perm[(size_t) d(E.rng)]);
            g.ids_h[(size_t) (t * n_used + s)] = perm[(size_t) s];
        }
    }
    const int64_t ne11 = bcast ? 1 : n_used;
    g.x.resize((size_t) (fx.K * ne11 * n_tok));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : g.x) v = nd(E.rng);

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, n_tok);
    g.X   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, fx.K, ne11, n_tok);
    g.Ys  = ggml_mul_mat_id(g.ctx, fx.Ws, g.X, g.ids);
    g.Yf  = ggml_mul_mat_id(g.ctx, fx.Wf, g.X, g.ids);
    if (fx.use_ptrs) ggml_mul_mat_id_set_expert_ptrs(g.Yf, fx.ptr_t->data, (int32_t) fx.n_as);
    g.gs  = ggml_new_graph(g.ctx);
    g.gf  = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gs, g.Ys);
    ggml_build_forward_expand(g.gf, g.Yf);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, E.be);
    if (!g.buf) return false;
    ggml_backend_tensor_set(g.ids, g.ids_h.data(), 0, g.ids_h.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.X, g.x.data(), 0, g.x.size() * sizeof(float));
    return true;
}

row_result moe_worker_case(moe_fixture & fx, int64_t n_used, int64_t n_tok, bool bcast) {
    row_result r;
    char buf[160];
    snprintf(buf, sizeof buf, "MUL_MAT_ID(%s) N=%lld K=%lld n_as=%lld n_used=%lld n_tok=%lld src1_ne1=%s",
             fx.use_ptrs ? "expert_ptrs" : "grouped", (long long) fx.N, (long long) fx.K, (long long) fx.n_as,
             (long long) n_used, (long long) n_tok, bcast ? "1" : "n_used");
    r.desc = buf;
    moe_graph g;
    if (!build_moe_graph(g, fx, n_used, n_tok, bcast)) { r.why = "device allocation failed"; return r; }
    const ggml_status st1 = ggml_backend_graph_compute(E.be, g.gs);
    const ggml_status st2 = ggml_backend_graph_compute(E.be, g.gf);
    ggml_backend_synchronize(E.be);

    const int64_t N = fx.N, K = fx.K, ne11 = bcast ? 1 : n_used;
    const int64_t n_out = N * n_used * n_tok;
    std::vector<float> ys((size_t) n_out), yf((size_t) n_out);
    ggml_backend_tensor_get(g.Ys, ys.data(), 0, ys.size() * sizeof(float));
    ggml_backend_tensor_get(g.Yf, yf.data(), 0, yf.size() * sizeof(float));
    size_t nonfinite = 0;
    for (float v : yf) nonfinite += !std::isfinite(v);

    const int64_t n_entries = n_used * n_tok;
    const int64_t step = std::max<int64_t>(1, (n_entries + kMaxRefSamples - 1) / kMaxRefSamples);
    err_acc ea_rad, ea_stock;
    std::vector<float> wd;
    for (int64_t idx = 0; idx < n_entries; idx += step) {
        const int64_t s = idx % n_used;
        const int64_t t = idx / n_used;
        const int32_t e = g.ids_h[(size_t) (t * n_used + s)];
        const int64_t i11 = bcast ? 0 : s;
        dequantize_mxfp4(fx.wq[(size_t) (e % moe_fixture::kDistinct)].data(), N, K, wd);
        const float * xr = &g.x[(size_t) (K * (i11 + ne11 * t))];
        for (int64_t n = 0; n < N; ++n) {
            double ref = 0;
            const float * wr = &wd[(size_t) (n * K)];
            for (int64_t k = 0; k < K; ++k) ref += (double) wr[k] * (double) xr[k];
            const size_t o = (size_t) (n + N * (s + n_used * t));
            ea_rad.add(yf[o], ref);
            ea_stock.add(ys[o], ref);
        }
    }
    r.rel_rad = ea_rad.rel();
    r.rel_stock = ea_stock.rel();
    if (st1 != GGML_STATUS_SUCCESS || st2 != GGML_STATUS_SUCCESS) { r.why = "graph compute failed"; return r; }
    // every routed (slot, token) row is written exactly once, so the whole output must be finite
    r.pass = nonfinite == 0 && r.rel_rad <= r.rel_stock && r.rel_rad <= kBound;
    if (!r.pass) {
        r.why = nonfinite ? "non-finite output" : (r.rel_rad > kBound ? "radiance rel RMS above bound" : "radiance worse than stock");
    }
    return r;
}


// ---- worker helpers: reader-thread stage (concurrent) and unstage, byte-exact against the compute-stream repack ----
row_result worker_stage_case(int64_t N, int64_t K, int n_threads) {
    row_result r;
    char buf[160];
    snprintf(buf, sizeof buf, "WORKER stage_reader x%d threads + unstage N=%lld K=%lld", n_threads, (long long) N, (long long) K);
    r.desc = buf;
    const size_t pbytes = E.packed(N, K);
    if (!pbytes) { r.skip = true; r.why = "shape not supported"; return r; }
    const size_t qsz = (size_t) ggml_row_size(GGML_TYPE_MXFP4, K) * (size_t) N;
    const size_t slot = (pbytes + 255) & ~(size_t) 255;
    std::vector<uint8_t> src(qsz);
    std::uniform_int_distribution<int> bd(0, 255);
    for (auto & b : src) b = (uint8_t) bd(E.rng);

    // n_threads roles in one buffer, each holding the stock bytes at its base (as after the H2D), plus a reference
    ggml_backend_buffer_t bslots = ggml_backend_buft_alloc_buffer(E.buft, slot * (size_t) n_threads);
    ggml_backend_buffer_t bsrc   = ggml_backend_buft_alloc_buffer(E.buft, qsz);
    ggml_backend_buffer_t bref   = ggml_backend_buft_alloc_buffer(E.buft, pbytes);
    ggml_backend_buffer_t bback  = ggml_backend_buft_alloc_buffer(E.buft, qsz);
    if (!bslots || !bsrc || !bref || !bback) { r.why = "device allocation failed"; return r; }
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tsl  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) (slot * (size_t) n_threads));
    ggml_tensor * tsrc = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) qsz);
    ggml_tensor * tref = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) pbytes);
    ggml_tensor * tbk  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) qsz);
    ggml_backend_tensor_alloc(bslots, tsl, ggml_backend_buffer_get_base(bslots));
    ggml_backend_tensor_alloc(bsrc, tsrc, ggml_backend_buffer_get_base(bsrc));
    ggml_backend_tensor_alloc(bref, tref, ggml_backend_buffer_get_base(bref));
    ggml_backend_tensor_alloc(bback, tbk, ggml_backend_buffer_get_base(bback));
    ggml_backend_tensor_set(tsrc, src.data(), 0, qsz);
    for (int t = 0; t < n_threads; ++t) {
        ggml_backend_tensor_set(tsl, src.data(), (size_t) t * slot, qsz);
    }
    bool ok = E.repack(E.be, tsrc->data, tref->data, N, K);
    ggml_backend_synchronize(E.be);

    std::vector<int> th_ok((size_t) n_threads, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        th.emplace_back([&, t] {
            bool all = true;
            for (int rep = 0; rep < 3 && all; ++rep) {     // repeated, so the per-thread scratch/stream are reused
                if (rep > 0) {
                    ggml_backend_tensor_set(tsl, src.data(), (size_t) t * slot, qsz);   // back to stock bytes
                }
                all = E.stage_reader(E.be, (char *) tsl->data + (size_t) t * slot, qsz, N, K);
            }
            th_ok[(size_t) t] = all ? 1 : 0;
        });
    }
    for (auto & x : th) x.join();
    for (int t = 0; t < n_threads; ++t) ok = ok && th_ok[(size_t) t];

    std::vector<uint8_t> ref(pbytes), got(pbytes);
    ggml_backend_tensor_get(tref, ref.data(), 0, pbytes);
    size_t ndiff = 0;
    for (int t = 0; t < n_threads; ++t) {
        ggml_backend_tensor_get(tsl, got.data(), (size_t) t * slot, pbytes);
        for (size_t i = 0; i < pbytes; ++i) ndiff += got[i] != ref[i];
    }
    // unstage back to stock bytes from thread 0's packed role
    std::vector<uint8_t> back(qsz, 0xAB);
    const bool un_ok = E.unstage(E.be, tsl->data, tbk->data, qsz, N, K, true);
    ggml_backend_tensor_get(tbk, back.data(), 0, qsz);
    size_t nback = 0;
    for (size_t i = 0; i < qsz; ++i) nback += back[i] != src[i];
    r.rel_rad = (double) ndiff;
    r.rel_stock = (double) nback;
    r.pass = ok && un_ok && ndiff == 0 && nback == 0;
    if (!r.pass) r.why = !ok ? "stage_reader declined" : (!un_ok ? "unstage declined" : "bytes differ");
    ggml_free(ctx);
    ggml_backend_buffer_free(bslots);
    ggml_backend_buffer_free(bsrc);
    ggml_backend_buffer_free(bref);
    ggml_backend_buffer_free(bback);
    return r;
}

// ---- fused gate;up pair (GGML_TENSOR_FLAG_MXFP4_RADIANCE_PAIR): one flagged [2N, K] weight vs a stock [gate;up] ----
row_result pair_case(int64_t N, int64_t K, int64_t n_as, int64_t n_used, int64_t n_tok, bool id, bool bcast) {
    row_result r;
    char buf[180];
    if (id) {
        snprintf(buf, sizeof buf, "PAIR MUL_MAT_ID N=2x%lld K=%lld n_as=%lld n_used=%lld n_tok=%lld src1_ne1=%s",
                 (long long) N, (long long) K, (long long) n_as, (long long) n_used, (long long) n_tok, bcast ? "1" : "n_used");
    } else {
        snprintf(buf, sizeof buf, "PAIR MUL_MAT    N=2x%lld K=%lld M=%lld", (long long) N, (long long) K, (long long) n_tok);
    }
    r.desc = buf;
    const size_t pbytes = E.packed(N, K);
    if (!pbytes) { r.skip = true; r.why = "shape not supported"; return r; }
    const size_t qsz   = (size_t) ggml_row_size(GGML_TYPE_MXFP4, K) * (size_t) N;
    const size_t moff  = (pbytes + 255) & ~(size_t) 255;          // gate packed, then up packed
    const size_t stride = 2 * moff;                                // per-expert pair stride
    const int64_t E_n  = id ? n_as : 1;

    // stock bytes per (expert, half); halves are independent quantizations
    std::vector<std::vector<uint8_t>> wq((size_t) (2 * E_n));
    std::vector<float> wf((size_t) (N * K));
    for (auto & q : wq) { fill_weights(wf, N, K); quantize_mxfp4(wf.data(), N, K, q); }

    const int64_t ne11 = bcast ? 1 : n_used;
    const int64_t M = id ? n_tok : n_tok;
    std::vector<float> x((size_t) (id ? K * ne11 * n_tok : K * M));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(E.rng);
    std::vector<int32_t> ids_h;
    if (id) {
        ids_h.resize((size_t) (n_used * n_tok));
        std::vector<int32_t> perm((size_t) n_as);
        std::iota(perm.begin(), perm.end(), 0);
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t s = 0; s < n_used; ++s) {
                std::uniform_int_distribution<int64_t> d(s, n_as - 1);
                std::swap(perm[(size_t) s], perm[(size_t) d(E.rng)]);
                ids_h[(size_t) (t * n_used + s)] = perm[(size_t) s];
            }
        }
    }

    ggml_init_params ip = { 24 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * Ws = id ? ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, 2 * N, n_as) : ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, K, 2 * N);
    ggml_tensor * Wf = id ? ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, 2 * N, n_as) : ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, K, 2 * N);
    Wf->flags |= GGML_TENSOR_FLAG_MXFP4_RADIANCE | GGML_TENSOR_FLAG_MXFP4_RADIANCE_PAIR;
    Wf->op_params[0] = (int32_t) moff;
    if (id) { Wf->nb[2] = stride; Wf->nb[3] = (size_t) n_as * stride; }
    ggml_tensor * ids = id ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok) : nullptr;
    ggml_tensor * X = id ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, n_tok) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * Ys = id ? ggml_mul_mat_id(ctx, Ws, X, ids) : ggml_mul_mat(ctx, Ws, X);
    ggml_tensor * Yf = id ? ggml_mul_mat_id(ctx, Wf, X, ids) : ggml_mul_mat(ctx, Wf, X);

    ggml_backend_buffer_t buf_f = ggml_backend_buft_alloc_buffer(E.buft, (size_t) E_n * stride + 65536);
    if (!buf_f) { r.why = "device allocation failed"; ggml_free(ctx); return r; }
    ggml_backend_tensor_alloc(buf_f, Wf, ggml_backend_buffer_get_base(buf_f));
    ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx, E.be);
    if (!buf_c) { r.why = "device allocation failed"; ggml_free(ctx); ggml_backend_buffer_free(buf_f); return r; }

    // stock: expert e's [gate;up] = gate bytes then up bytes (contiguous rows 0..N-1 then N..2N-1)
    bool packed_ok = true;
    ggml_backend_buffer_t buf_t = ggml_backend_buft_alloc_buffer(E.buft, qsz + 4096);   // staging for repack source
    ggml_context * ctx_t = ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * tstage = ggml_new_tensor_1d(ctx_t, GGML_TYPE_I8, (int64_t) qsz);
    ggml_backend_tensor_alloc(buf_t, tstage, ggml_backend_buffer_get_base(buf_t));
    for (int64_t e = 0; e < E_n; ++e) {
        for (int h = 0; h < 2; ++h) {
            const auto & q = wq[(size_t) (2 * e + h)];
            ggml_backend_tensor_set(Ws, q.data(), (size_t) (e * 2 + h) * qsz, qsz);
            ggml_backend_tensor_set(tstage, q.data(), 0, qsz);
            packed_ok = packed_ok && E.repack(E.be, tstage->data, (char *) Wf->data + (size_t) e * stride + (size_t) h * moff, N, K);
        }
    }
    if (id) ggml_backend_tensor_set(ids, ids_h.data(), 0, ids_h.size() * sizeof(int32_t));
    ggml_backend_tensor_set(X, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_synchronize(E.be);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Ys);
    ggml_build_forward_expand(gf, Yf);
    const ggml_status st = ggml_backend_graph_compute(E.be, gf);
    ggml_backend_synchronize(E.be);

    const int64_t n_rows = id ? n_used * n_tok : M;
    std::vector<float> ys((size_t) (2 * N * n_rows)), yf((size_t) (2 * N * n_rows));
    ggml_backend_tensor_get(Ys, ys.data(), 0, ys.size() * sizeof(float));
    ggml_backend_tensor_get(Yf, yf.data(), 0, yf.size() * sizeof(float));
    size_t nonfinite = 0;
    for (float v : yf) nonfinite += !std::isfinite(v);

    err_acc ea_rad, ea_stock;
    const int64_t step = std::max<int64_t>(1, (n_rows + kMaxRefSamples - 1) / kMaxRefSamples);
    std::vector<float> wd;
    for (int64_t idx = 0; idx < n_rows; idx += step) {
        int64_t e = 0, i11 = 0, t = 0, row = idx;
        if (id) {
            const int64_t s = idx % n_used;
            t = idx / n_used;
            e = ids_h[(size_t) (t * n_used + s)];
            i11 = bcast ? 0 : s;
            row = s + n_used * t;
        }
        const float * xr = id ? &x[(size_t) (K * (i11 + ne11 * t))] : &x[(size_t) (K * idx)];
        for (int h = 0; h < 2; ++h) {
            dequantize_mxfp4(wq[(size_t) (2 * e + h)].data(), N, K, wd);
            for (int64_t n = 0; n < N; ++n) {
                double ref = 0;
                const float * wr = &wd[(size_t) (n * K)];
                for (int64_t k = 0; k < K; ++k) ref += (double) wr[k] * (double) xr[k];
                const size_t o = (size_t) (h * N + n + 2 * N * row);
                ea_rad.add(yf[o], ref);
                ea_stock.add(ys[o], ref);
            }
        }
    }
    r.rel_rad = ea_rad.rel();
    r.rel_stock = ea_stock.rel();
    ggml_free(ctx);
    ggml_free(ctx_t);
    ggml_backend_buffer_free(buf_f);
    ggml_backend_buffer_free(buf_c);
    ggml_backend_buffer_free(buf_t);
    if (st != GGML_STATUS_SUCCESS) { r.why = "graph compute failed"; return r; }
    r.pass = packed_ok && nonfinite == 0 && r.rel_rad <= r.rel_stock && r.rel_rad <= kBound;
    if (!r.pass) r.why = nonfinite ? "non-finite output" : (r.rel_rad > kBound ? "radiance rel RMS above bound" : "radiance worse than stock");
    return r;
}

// ---- perf: grouped radiance MUL_MAT_ID vs stock MXFP4 MUL_MAT_ID (host wall of graph_compute + sync) ----
double median3(double a, double b, double c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }

// PERF_COLD=1: before every timed call a 256 MB buffer is cleared (untimed) so the 64 MB MALL / L2 no longer hold the
// expert weights, which is what a real decode step sees.
ggml_backend_buffer_t g_flush_buf = nullptr;
void flush_caches() {
    static const bool cold = std::getenv("PERF_COLD") != nullptr;
    if (!cold) return;
    if (!g_flush_buf) g_flush_buf = ggml_backend_buft_alloc_buffer(E.buft, (size_t) 256 << 20);
    if (g_flush_buf) { ggml_backend_buffer_clear(g_flush_buf, 0); ggml_backend_synchronize(E.be); }
}

double time_graph_ms(ggml_cgraph * gr, int iters) {
    ggml_backend_graph_compute(E.be, gr);   // warmup
    ggml_backend_graph_compute(E.be, gr);
    ggml_backend_synchronize(E.be);
    double rep[3];
    for (int r = 0; r < 3; ++r) {
        double total = 0;
        for (int i = 0; i < iters; ++i) {
            flush_caches();
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(E.be, gr);
            ggml_backend_synchronize(E.be);
            total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        rep[r] = total / iters;
    }
    return median3(rep[0], rep[1], rep[2]);
}

void perf_moe(moe_fixture & fx, int64_t n_used, int64_t n_tok, bool bcast) {
    moe_graph g;
    if (!build_moe_graph(g, fx, n_used, n_tok, bcast)) { printf("| alloc failed\n"); return; }
    const int iters = n_tok >= 512 ? 3 : 10;
    const double ts = time_graph_ms(g.gs, iters);
    const double tf = time_graph_ms(g.gf, iters);
    printf("| N=%-5lld K=%-5lld n_as=%lld n_used=%lld n_tok=%-5lld ne1=%-6s | stock %9.3f ms | radiance %9.3f ms | speedup %5.2fx\n",
           (long long) fx.N, (long long) fx.K, (long long) fx.n_as, (long long) n_used, (long long) n_tok,
           bcast ? "1" : "n_used", ts, tf, ts / tf);
    fflush(stdout);
}

void print_row(const row_result & r) {
    if (r.skip) {
        printf("| %-74s | %12s | %12s | SKIP (%s)\n", r.desc.c_str(), "-", "-", r.why.c_str());
        return;
    }
    printf("| %-74s | %12.3e | %12.3e | %s%s%s\n", r.desc.c_str(), r.rel_rad, r.rel_stock,
           r.pass ? "PASS" : "FAIL", r.why.empty() ? "" : " (", r.why.empty() ? "" : (r.why + ")").c_str());
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_time_init();
    const bool perf = argc > 1 && std::string(argv[1]) == "--perf";
    if (!init_device()) {
        return 2;
    }

    // worker-like MUL_MAT_ID (n_as=256, n_used=6): gate/up = (N=2304, K=5120) src1_ne1=1, down = (5120, 2304) src1_ne1=n_used
    const int64_t wl_ntok[] = { 1, 3, 8, 64, 512, 2048 };
    if (perf) {
        printf("\nperf: grouped radiance MUL_MAT_ID vs stock MXFP4 MUL_MAT_ID (host wall, median of 3 reps)\n");
        const int64_t shapes[][2] = { { 2304, 5120 }, { 5120, 2304 } };
        for (auto & sh : shapes) {
            moe_fixture fx;
            if (!fx.init(sh[0], sh[1], 256)) { printf("fixture init failed\n"); return 1; }
            for (int64_t nt : wl_ntok) {
                if (const char * only = std::getenv("PERF_NTOK")) { if (std::atoll(only) != nt) continue; }
                perf_moe(fx, 6, nt, sh[0] == 2304);
            }
        }
        return 0;
    }

    std::vector<row_result> rows;
    printf("\nrel RMS vs fp64 reference (flagged radiance | stock MXFP4; ROUNDTRIP rows: differing bytes), bound %.4f\n", kBound);
    printf("| %-74s | %12s | %12s | result\n", "case", "radiance", "stock");

    const int64_t shapes[][2] = { { 2304, 5120 }, { 5120, 2304 }, { 256, 256 } };
    const int64_t Ms[] = { 1, 7, 64, 65, 200, 512 };
    for (auto & sh : shapes) {
        for (int64_t M : Ms) {
            rows.push_back(mul_mat_case(sh[0], sh[1], M));
            print_row(rows.back());
            fflush(stdout);
        }
    }

    const int64_t n_used_list[] = { 2, 6 };
    const int64_t n_tok_list[]  = { 1, 5, 64, 200 };
    for (int64_t n_used : n_used_list) {
        for (int64_t n_tok : n_tok_list) {
            for (bool bcast : { true, false }) {
                rows.push_back(mul_mat_id_case(2304, 5120, 8, n_used, n_tok, bcast));
                print_row(rows.back());
                fflush(stdout);
            }
        }
    }

    for (auto & sh : shapes) {
        rows.push_back(roundtrip_case(sh[0], sh[1]));
        print_row(rows.back());
        fflush(stdout);
    }

    for (int th : { 1, 4 }) {
        rows.push_back(worker_stage_case(2304, 5120, th));
        print_row(rows.back());
        fflush(stdout);
    }
    for (int64_t M : { 1, 65, 512 }) {
        rows.push_back(pair_case(2304, 5120, 1, 1, M, false, false));
        print_row(rows.back());
        fflush(stdout);
    }
    for (int64_t nt : { 1, 8, 200 }) {
        rows.push_back(pair_case(2304, 5120, 32, 6, nt, true, true));
        print_row(rows.back());
        fflush(stdout);
    }
    {
        const int64_t wshapes[][2] = { { 2304, 5120 }, { 5120, 2304 } };
        for (auto & sh : wshapes) {
            moe_fixture fx;
            if (!fx.init(sh[0], sh[1], 256)) {
                row_result r; r.desc = "MUL_MAT_ID(grouped) fixture"; r.why = "fixture init failed";
                rows.push_back(r); print_row(rows.back());
                continue;
            }
            for (int64_t nt : wl_ntok) {
                rows.push_back(moe_worker_case(fx, 6, nt, sh[0] == 2304));
                print_row(rows.back());
                fflush(stdout);
            }
        }
    }

    {   // expert_ptrs (worker batch-mmid): scrambled, irregularly spaced slots located through the pointer table
        moe_fixture fx;
        if (!fx.init(2304, 5120, 64, true)) {
            row_result r; r.desc = "MUL_MAT_ID(expert_ptrs) fixture"; r.why = "fixture init failed";
            rows.push_back(r); print_row(rows.back());
        } else {
            for (int64_t nt : { 1, 3, 8, 64, 512 }) {
                rows.push_back(moe_worker_case(fx, 6, nt, true));
                print_row(rows.back());
                fflush(stdout);
            }
        }
    }

    int n_pass = 0, n_fail = 0, n_skip = 0;
    for (const auto & r : rows) {
        if (r.skip) n_skip++; else if (r.pass) n_pass++; else n_fail++;
    }
    printf("\n%d PASS, %d FAIL, %d SKIP (of %zu cases)\n", n_pass, n_fail, n_skip, rows.size());
    return n_fail == 0 ? 0 : 1;
}
