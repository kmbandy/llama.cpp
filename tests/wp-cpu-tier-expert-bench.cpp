// CPU-tier expert microbench: mirrors wp-expert-worker cpu_tier_compute_one (stage 0).
// One full expert = MXFP4 gate[5120->2304] + up[5120->2304] -> clamp -> swiglu_split ->
// down[2304->5120] -> * router weight, run through the ggml CPU backend (q8_0 activation
// quantization happens inside ggml mul_mat). Weights are wrapped in place via
// ggml_backend_cpu_buffer_from_ptr over a file-layout page (gate|up|down, 3 x 6,266,880 B).
// usage: wp-cpu-tier-expert-bench [--threads N=8] [--experts N=48] [--thp] [--min-sec S=2]
//                                 [--mode a|b|ab=ab] [--ms 1,2,4,6,8] [--clamp 10]
// env: WP_BENCH_EXPERTS, WP_BENCH_THREADS as defaults.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/mman.h>
#endif

static constexpr int64_t N_EMBD = 5120, N_FF = 2304;
static constexpr size_t  ROLE_BYTES = 6266880;           // 2304 * (5120/32) * 17
static constexpr size_t  PAGE_BYTES = 3 * ROLE_BYTES;    // 18,800,640
static constexpr size_t  OFF_GATE = 0, OFF_UP = ROLE_BYTES, OFF_DOWN = 2 * ROLE_BYTES;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void log_openmp() {
#if defined(__linux__)
    void * kmp  = dlsym(RTLD_DEFAULT, "__kmpc_fork_call");
    void * kmp2 = kmp ? kmp : dlsym(RTLD_DEFAULT, "kmp_set_blocktime");
    void * gomp = dlsym(RTLD_DEFAULT, "GOMP_parallel");
    const char * kind = kmp2 && gomp ? "libomp+libgomp" : kmp2 ? "libomp" : gomp ? "libgomp" : "none";
    auto e = [](const char * k) { const char * v = std::getenv(k); return (v && *v) ? v : "<unset>"; };
    fprintf(stderr, "OpenMP runtime=%s KMP_BLOCKTIME=%s OMP_WAIT_POLICY=%s GOMP_SPINCOUNT=%s\n",
            kind, e("KMP_BLOCKTIME"), e("OMP_WAIT_POLICY"), e("GOMP_SPINCOUNT"));
#endif
}

struct Bench {
    ggml_backend_t cpu;
    ggml_gallocr_t galloc;
    uint8_t * base;
    ggml_backend_buffer_t whole;   // persistent wrap of the full arena (mode b only)
    float clamp;
};

// Mode (a): exactly cpu_tier_compute_one stage 0: per-call wrap, ggml_init, graph, alloc, compute.
static void run_one_fresh(Bench & b, const uint8_t * page, int64_t m, const std::vector<float> & xs,
                          const std::vector<float> & ws, std::vector<float> & out) {
    std::vector<float> x(xs.begin(), xs.begin() + (size_t) m * N_EMBD);
    std::vector<float> w(ws.begin(), ws.begin() + m);
    ggml_backend_buffer_t wb = ggml_backend_cpu_buffer_from_ptr(const_cast<uint8_t *>(page), PAGE_BYTES);
    ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    auto role = [&](int64_t ne0, int64_t ne1, size_t off) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, ne0, ne1);
        t->buffer = wb;
        t->data = (uint8_t *) ggml_backend_buffer_get_base(wb) + off;
        return t;
    };
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, m);
    ggml_set_input(input);
    ggml_tensor * gate_x = ggml_mul_mat(ctx, role(N_EMBD, N_FF, OFF_GATE), input);
    ggml_tensor * up_x   = ggml_mul_mat(ctx, role(N_EMBD, N_FF, OFF_UP), input);
    if (b.clamp > 1e-6f) {
        up_x   = ggml_clamp(ctx, up_x, -b.clamp, b.clamp);
        gate_x = ggml_clamp(ctx, gate_x, -INFINITY, b.clamp);
    }
    ggml_tensor * hidden = ggml_swiglu_split(ctx, gate_x, up_x);
    ggml_tensor * output = ggml_mul_mat(ctx, role(N_FF, N_EMBD, OFF_DOWN), hidden);
    ggml_tensor * route = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, m);
    ggml_set_input(route);
    ggml_tensor * weighted = ggml_mul(ctx, output, route);
    ggml_set_output(weighted);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(g, weighted);
    if (!ggml_gallocr_alloc_graph(b.galloc, g)) { fprintf(stderr, "alloc failed\n"); exit(1); }
    ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(route, w.data(), 0, ggml_nbytes(route));
    if (ggml_backend_graph_compute(b.cpu, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    ggml_backend_tensor_get(weighted, out.data(), 0, (size_t) m * N_EMBD * sizeof(float));
    ggml_free(ctx);
    ggml_backend_buffer_free(wb);
}

// Mode (b): graph built once per m; per expert only rebind the 3 weight data pointers,
// copy activations/route in, compute, copy out.
struct Persistent {
    ggml_context * ctx = nullptr;
    ggml_cgraph * g = nullptr;
    ggml_tensor *input, *route, *weighted, *gate, *up, *down;
    int64_t m;
    Persistent(Bench & b, int64_t m_) : m(m_) {
        ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true };
        ctx = ggml_init(p);
        auto role = [&](int64_t ne0, int64_t ne1) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, ne0, ne1);
            t->buffer = b.whole;
            return t;
        };
        gate = role(N_EMBD, N_FF); up = role(N_EMBD, N_FF); down = role(N_FF, N_EMBD);
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, m);
        ggml_set_input(input);
        ggml_tensor * gx = ggml_mul_mat(ctx, gate, input);
        ggml_tensor * ux = ggml_mul_mat(ctx, up, input);
        if (b.clamp > 1e-6f) {
            ux = ggml_clamp(ctx, ux, -b.clamp, b.clamp);
            gx = ggml_clamp(ctx, gx, -INFINITY, b.clamp);
        }
        ggml_tensor * hidden = ggml_swiglu_split(ctx, gx, ux);
        ggml_tensor * output = ggml_mul_mat(ctx, down, hidden);
        route = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, m);
        ggml_set_input(route);
        weighted = ggml_mul(ctx, output, route);
        ggml_set_output(weighted);
        g = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(g, weighted);
        if (!ggml_gallocr_alloc_graph(b.galloc, g)) { fprintf(stderr, "alloc failed\n"); exit(1); }
    }
    ~Persistent() { ggml_free(ctx); }
    void run(Bench & b, const uint8_t * page, const std::vector<float> & xs, const std::vector<float> & ws,
             std::vector<float> & out) {
        gate->data = const_cast<uint8_t *>(page) + OFF_GATE;
        up->data   = const_cast<uint8_t *>(page) + OFF_UP;
        down->data = const_cast<uint8_t *>(page) + OFF_DOWN;
        ggml_backend_tensor_set(input, xs.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(route, ws.data(), 0, ggml_nbytes(route));
        if (ggml_backend_graph_compute(b.cpu, g) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        ggml_backend_tensor_get(weighted, out.data(), 0, (size_t) m * N_EMBD * sizeof(float));
    }
};

static void report(const char * mode, int64_t m, std::vector<double> & t) {
    std::sort(t.begin(), t.end());
    double sum = 0; for (double v : t) sum += v;
    double mean = sum / t.size(), p50 = t[t.size() / 2], p99 = t[std::min(t.size() - 1, (size_t)(t.size() * 0.99))];
    printf("%-10s m=%d  n=%-6zu mean %7.3f ms  p50 %7.3f ms  p99 %7.3f ms  min %7.3f ms | %6.2f GB/s (mean)  %6.2f GB/s (p50)\n",
           mode, (int) m, t.size(), mean * 1e3, p50 * 1e3, p99 * 1e3, t.front() * 1e3,
           PAGE_BYTES / mean / 1e9, PAGE_BYTES / p50 / 1e9);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    int nth = getenv("WP_BENCH_THREADS") ? atoi(getenv("WP_BENCH_THREADS")) : 8;
    int ne  = getenv("WP_BENCH_EXPERTS") ? atoi(getenv("WP_BENCH_EXPERTS")) : 48;
    double min_sec = 2.0; bool thp = false; float clamp = 10.0f;
    std::string mode = "ab";
    std::vector<int> ms = {1, 2, 4, 6, 8};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--threads") nth = atoi(nx());
        else if (a == "--experts") ne = atoi(nx());
        else if (a == "--thp") thp = true;
        else if (a == "--min-sec") min_sec = atof(nx());
        else if (a == "--mode") mode = nx();
        else if (a == "--clamp") clamp = (float) atof(nx());
        else if (a == "--ms") { ms.clear(); std::string s = nx(); size_t p = 0; while (p < s.size()) { ms.push_back(atoi(s.c_str() + p)); p = s.find(',', p); if (p == std::string::npos) break; ++p; } }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    log_openmp();

    const size_t total = (size_t) ne * PAGE_BYTES;
    const size_t align = 2u << 20;
    uint8_t * arena = (uint8_t *) aligned_alloc(align, (total + align - 1) / align * align);
    if (!arena) { fprintf(stderr, "alloc failed\n"); return 1; }
#if defined(__linux__)
    if (thp) {
        int rc = madvise(arena, (total + align - 1) / align * align, MADV_HUGEPAGE);
        fprintf(stderr, "madvise(MADV_HUGEPAGE) rc=%d\n", rc);
    }
#endif
    {   // random but valid MXFP4 blocks: block = {u8 e8m0 scale; u8 qs[16]} (17 B); scale exp 120..126
        std::mt19937_64 rng(1234);
        for (size_t o = 0; o < total; o += 17) {
            uint64_t r0 = rng(), r1 = rng();
            arena[o] = (uint8_t)(120 + (r0 >> 56) % 7);
            std::memcpy(arena + o + 1, &r0, 8);
            std::memcpy(arena + o + 9, &r1, 8);
        }
    }
    if (thp) {
        FILE * f = fopen("/proc/self/smaps_rollup", "r");
        char line[256];
        while (f && fgets(line, sizeof line, f)) if (strstr(line, "AnonHugePages")) fputs(line, stderr);
        if (f) fclose(f);
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.f, 1.f), uw(0.1f, 1.f);
    std::vector<float> xs((size_t) 8 * N_EMBD), ws(8);
    for (auto & v : xs) v = ud(rng);
    for (auto & v : ws) v = uw(rng);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, nth);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    Bench b{cpu, galloc, arena, nullptr, clamp};
    b.whole = ggml_backend_cpu_buffer_from_ptr(arena, total);
    printf("experts=%d (%.1f MB arena, %.2f MB/expert) threads=%d thp=%d clamp=%g min_sec=%.1f\n",
           ne, total / 1e6, PAGE_BYTES / 1e6, nth, (int) thp, clamp, min_sec);

    std::vector<float> out((size_t) 8 * N_EMBD);
    for (int m : ms) {
        if (m < 1 || m > 8) continue;
        for (char md : std::string("ab")) {
            if (mode.find(md) == std::string::npos) continue;
            std::unique_ptr<Persistent> pg;
            if (md == 'b') pg.reset(new Persistent(b, m));
            auto call = [&](int e) {
                const uint8_t * page = arena + (size_t) e * PAGE_BYTES;
                if (md == 'a') run_one_fresh(b, page, m, xs, ws, out);
                else pg->run(b, page, xs, ws, out);
            };
            for (int e = 0; e < std::min(ne, 8); ++e) call(e);   // warmup
            std::vector<double> t;
            double start = now_s();
            for (int e = 0; ; e = (e + 1) % ne) {
                double t0 = now_s();
                call(e);
                t.push_back(now_s() - t0);
                if (now_s() - start >= min_sec && t.size() >= (size_t) ne) break;
            }
            report(md == 'a' ? "fresh(a)" : "persist(b)", m, t);
        }
    }
    ggml_backend_buffer_free(b.whole);
    ggml_gallocr_free(galloc);
    ggml_backend_free(cpu);
    free(arena);
    return 0;
}
