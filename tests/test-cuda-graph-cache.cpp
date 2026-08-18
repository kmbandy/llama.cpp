#include "ggml-cuda-graph-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>

namespace {

struct FakeGraph {
    int64_t last_used_time = 0;
};

using Map = std::unordered_map<const void *, std::unique_ptr<FakeGraph>>;

void put(Map & m, const void * k, int64_t used) {
    auto g = std::make_unique<FakeGraph>();
    g->last_used_time = used;
    m[k] = std::move(g);
}

void require(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

ggml_tensor make_node(const char * name, ggml_op op, int64_t ne0, int64_t ne1 = 1) {
    ggml_tensor t;
    std::memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.op = op;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = sizeof(float) * (size_t) ne0;
    t.nb[2] = t.nb[1] * (size_t) ne1;
    t.nb[3] = t.nb[2];
    std::snprintf(t.name, sizeof(t.name), "%s", name);
    return t;
}

} // namespace

int main() {
    const void * a = (const void *) 0x1;
    const void * b = (const void *) 0x2;
    const void * c = (const void *) 0x3;
    const void * d = (const void *) 0x4;

    {
        Map m;
        put(m, a, 1'000'000);
        put(m, b, 9'000'000);
        const size_t n = ggml_cuda_graph_cache_evict_ttl(m, 11'000'000, 10'000'000);
        require(n == 1 && m.size() == 1 && m.count(b) == 1, "ttl drops only unused >= 10s");
    }

    {
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        put(m, c, 3);
        const size_t n = ggml_cuda_graph_cache_evict_lru(m, /*cap=*/2, /*keep=*/nullptr);
        require(n == 2 && m.size() == 1 && m.count(c) == 1,
                "lru evicts until size < cap so the next insert fits");
        require(m.count(a) == 0 && m.count(b) == 0, "two oldest are gone");
    }

    {
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        const size_t n = ggml_cuda_graph_cache_evict_lru(m, /*cap=*/1, /*keep=*/a);
        require(n == 1 && m.size() == 1 && m.count(a) == 1, "lru never evicts keep");
    }

    {
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        require(ggml_cuda_graph_cache_evict_lru(m, /*cap=*/0, nullptr) == 0 && m.size() == 2,
                "cap 0 is no cap");
        require(ggml_cuda_graph_cache_evict_lru(m, /*cap=*/8, nullptr) == 0 && m.size() == 2,
                "under cap is a no-op");
    }

    {
        // insert path: evict to cap-1 then add
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        put(m, c, 3);
        ggml_cuda_graph_cache_evict_lru(m, /*cap=*/3, nullptr);
        put(m, d, 4);
        require(m.size() == 3 && m.count(d) == 1 && m.count(a) == 0,
                "insert after lru-to-cap keeps the new key and drops LRU");
    }

    {
        require(ggml_cuda_graph_cache_policy{}.cap == 0,
                "default cap is 0 so a split decode working set is not evicted");
    }

    {
        ggml_tensor x = make_node("attn_out", GGML_OP_MUL_MAT, 4096, 1);
        ggml_tensor y = x;
        y.data = (void *) 0x1000;
        y.src[0] = (ggml_tensor *) 0x2000;
        y.buffer = (ggml_backend_buffer *) 0x3000;
        y.extra = (void *) 0x4000;
        y.view_src = (ggml_tensor *) 0x5000;
        require(ggml_cuda_graph_tensor_topo_equal(x, y),
                "topo equal ignores object and device pointers");

        y.ne[1] = 4;
        require(!ggml_cuda_graph_tensor_topo_equal(x, y),
                "topo unequal when ne changes");

        ggml_tensor v1 = make_node("k_view", GGML_OP_VIEW, 128, 4);
        ggml_tensor v2 = v1;
        size_t off1 = 1024;
        size_t off2 = 2048;
        std::memcpy(v1.op_params, &off1, sizeof(off1));
        std::memcpy(v2.op_params, &off2, sizeof(off2));
        v1.view_offs = off1;
        v2.view_offs = off2;
        require(ggml_cuda_graph_tensor_topo_equal(v1, v2),
                "VIEW offset is not topology");

        ggml_tensor s1 = make_node("rms", GGML_OP_RMS_NORM, 4096, 1);
        ggml_tensor s2 = s1;
        float eps = 1e-5f;
        float eps2 = 1e-6f;
        std::memcpy(s1.op_params, &eps, sizeof(eps));
        std::memcpy(s2.op_params, &eps2, sizeof(eps2));
        require(!ggml_cuda_graph_tensor_topo_equal(s1, s2),
                "non-VIEW op_params are topology");
    }

    {
        ggml_tensor x = make_node("attn_out", GGML_OP_MUL_MAT, 4096, 1);
        ggml_tensor y = x;
        y.data = (void *) 0x1000;
        y.src[0] = (ggml_tensor *) 0x2000;
        const uint64_t hx = ggml_cuda_graph_mix_tensor_topo(1469598103934665603ULL, &x);
        const uint64_t hy = ggml_cuda_graph_mix_tensor_topo(1469598103934665603ULL, &y);
        require(hx == hy, "fingerprint ignores pointers");

        y.ne[1] = 8;
        const uint64_t hy2 = ggml_cuda_graph_mix_tensor_topo(1469598103934665603ULL, &y);
        require(hx != hy2, "fingerprint changes with ne");

        ggml_tensor z = x;
        std::snprintf(z.name, sizeof(z.name), "ffn_out");
        const uint64_t hz = ggml_cuda_graph_mix_tensor_topo(1469598103934665603ULL, &z);
        require(hx != hz, "fingerprint changes with name");
    }

    std::printf("ok: cuda graph cache ttl + lru cap + topo identity\n");
    return 0;
}
