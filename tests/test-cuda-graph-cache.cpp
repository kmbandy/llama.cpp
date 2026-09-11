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
    // evict_* now hand the evicted graph to a retire callback (deferred
    // destroy on its last-launch event) instead of destroying in place.
    auto retire = [](std::unique_ptr<FakeGraph>) {};

    const void * a = (const void *) 0x1;
    const void * b = (const void *) 0x2;
    const void * c = (const void *) 0x3;
    const void * d = (const void *) 0x4;

    {
        Map m;
        put(m, a, 1'000'000);
        put(m, b, 9'000'000);
        const size_t n = ggml_cuda_graph_cache_evict_ttl(m, 11'000'000, 10'000'000, retire);
        require(n == 1 && m.size() == 1 && m.count(b) == 1, "ttl drops only unused >= 10s");
    }

    {
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        put(m, c, 3);
        const size_t n = ggml_cuda_graph_cache_evict_lru(m, /*cap=*/2, /*keep=*/nullptr, retire);
        require(n == 2 && m.size() == 1 && m.count(c) == 1,
                "lru evicts until size < cap so the next insert fits");
        require(m.count(a) == 0 && m.count(b) == 0, "two oldest are gone");
    }

    {
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        const size_t n = ggml_cuda_graph_cache_evict_lru(m, /*cap=*/1, /*keep=*/a, retire);
        require(n == 1 && m.size() == 1 && m.count(a) == 1, "lru never evicts keep");
    }

    {
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        require(ggml_cuda_graph_cache_evict_lru(m, /*cap=*/0, nullptr, retire) == 0 && m.size() == 2,
                "cap 0 is no cap");
        require(ggml_cuda_graph_cache_evict_lru(m, /*cap=*/8, nullptr, retire) == 0 && m.size() == 2,
                "under cap is a no-op");
    }

    {
        // insert path: evict to cap-1 then add
        Map m;
        put(m, a, 1);
        put(m, b, 2);
        put(m, c, 3);
        ggml_cuda_graph_cache_evict_lru(m, /*cap=*/3, nullptr, retire);
        put(m, d, 4);
        require(m.size() == 3 && m.count(d) == 1 && m.count(a) == 0,
                "insert after lru-to-cap keeps the new key and drops LRU");
    }

    {
        require(ggml_cuda_graph_cache_policy{}.cap == 256,
                "default cap covers split-decode working set and bounds prefill misses");
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

    {
        ggml_tensor x = make_node("ffn_down", GGML_OP_MUL_MAT, 1408, 1);
        ggml_tensor y = x;
        y.data = (void *) 0x1000;
        ggml_tensor w1 = make_node("w", GGML_OP_NONE, 1408, 2048);
        ggml_tensor w2 = w1;
        w1.data = (void *) 0xaaa000;
        w2.data = (void *) 0xbbb000;
        x.src[0] = &w1;
        y.src[0] = &w2;
        y.data = x.data;
        const uint64_t seed = 1469598103934665603ULL;
        require(ggml_cuda_graph_mix_tensor_topo(seed, &x) ==
                    ggml_cuda_graph_mix_tensor_topo(seed, &y),
                "same shape different expert slots is the same topology");
        require(ggml_cuda_graph_mix_tensor_addrs(seed, &x) !=
                    ggml_cuda_graph_mix_tensor_addrs(seed, &y),
                "different expert slots hash as different addr identity");
        y.src[0] = &w1;
        require(ggml_cuda_graph_mix_tensor_addrs(seed, &x) ==
                    ggml_cuda_graph_mix_tensor_addrs(seed, &y),
                "same slot replays");
    }

    {
        // 2026-09-05 regression: a fragment whose only wide-token node is
        // GGML_OP_PAGED_ATTN_MT (dst mirrors q: [head_dim, n_heads,
        // sum(q_lens), 1], token count in ne[2]) must be classified
        // prefill-shaped so the WP HIP-graph executor does not capture it --
        // capturing it crashed prefill under WP_HIP_GRAPHS=1 ("operation
        // failed due to a previous error during capture").
        ggml_tensor wide = make_node("paged_attn_wide", GGML_OP_PAGED_ATTN_MT, 256, 24);
        wide.ne[2] = 512; // sum(q_lens): a 512-token prefill
        wide.ne[3] = 1;
        ggml_tensor * nodes[] = { &wide };
        ggml_cgraph g;
        std::memset(&g, 0, sizeof(g));
        g.n_nodes = 1;
        g.nodes   = nodes;
        require(ggml_cuda_graph_is_prefill_shaped(&g),
                "PAGED_ATTN_MT with ne[2] (token count) > 32 is prefill-shaped");

        ggml_tensor narrow = wide;
        narrow.ne[2] = 5; // MTP decode: a handful of speculative queries
        ggml_tensor * nodes_decode[] = { &narrow };
        ggml_cgraph gd;
        std::memset(&gd, 0, sizeof(gd));
        gd.n_nodes = 1;
        gd.nodes   = nodes_decode;
        require(!ggml_cuda_graph_is_prefill_shaped(&gd),
                "PAGED_ATTN_MT with ne[2] <= 32 stays capturable (decode/spec)");
    }

    {
        // GATED_DELTA_NET's dst concatenates recurrent state onto the token
        // rows (ggml_gated_delta_net, ggml.c), so dst->ne[1] alone cannot
        // tell prefill from decode -- the true per-call token count is
        // src[2] (v)'s ne[2]. A fragment with a small dst but a wide v must
        // still be classified prefill-shaped.
        ggml_tensor v = make_node("gdn_v", GGML_OP_NONE, 128, 4);
        v.ne[2] = 512; // n_tokens
        v.ne[3] = 1;   // n_seqs
        ggml_tensor gdn = make_node("gdn_out", GGML_OP_GATED_DELTA_NET, 512, 8);
        gdn.src[2] = &v;
        ggml_tensor * nodes[] = { &gdn };
        ggml_cgraph g;
        std::memset(&g, 0, sizeof(g));
        g.n_nodes = 1;
        g.nodes   = nodes;
        require(ggml_cuda_graph_is_prefill_shaped(&g),
                "GATED_DELTA_NET reads token count from src[2] (v), not dst->ne[1]");

        v.ne[2] = 3; // decode: a few tokens
        require(!ggml_cuda_graph_is_prefill_shaped(&g),
                "GATED_DELTA_NET stays capturable when v's token count is small");
    }

    {
        // SSM_SCAN's dst is a flat 1-D [nelements(x) + K*state] buffer
        // (ggml_ssm_scan, ggml.c); the real token count is src[1] (x)'s
        // ne[2].
        ggml_tensor x = make_node("ssm_x", GGML_OP_NONE, 128, 4);
        x.ne[2] = 512;
        x.ne[3] = 1;
        ggml_tensor scan = make_node("ssm_scan_out", GGML_OP_SSM_SCAN, 1000000, 1);
        scan.src[1] = &x;
        ggml_tensor * nodes[] = { &scan };
        ggml_cgraph g;
        std::memset(&g, 0, sizeof(g));
        g.n_nodes = 1;
        g.nodes   = nodes;
        require(ggml_cuda_graph_is_prefill_shaped(&g),
                "SSM_SCAN reads token count from src[1] (x), not its flat dst");
    }

    std::printf("ok: cuda graph cache ttl + lru cap + topo identity + prefill-shaped classifier\n");
    return 0;
}
