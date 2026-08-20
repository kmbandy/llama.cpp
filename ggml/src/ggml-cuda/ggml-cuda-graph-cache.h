// HIP/CUDA graph cache helpers. Header-only so they can be unit-tested
// without a GPU.
//
// Production map is keyed by a structural fingerprint (see
// ggml_cuda_graph_mix_tensor_topo), not by cgraph->nodes[0]. Values expose
// last_used_time (ggml_time_us).
//
// Eviction:
//   1. TTL: drop entries unused for >= ttl_us (default 10s sweep).
//   2. Cap: if GGML_CUDA_GRAPH_MAX is set, drop LRU until size < cap
//      (never `keep`). cap == 0 (the default) means no cap — a split
//      decode graph has ~40-90 GPU segments and a tiny default cap
//      evicts every segment before its second visit, so warmup can
//      never complete.
//
// Replay identity:
//   Object pointers (src[], buffer, extra, view_src) are NOT topology.
//   The backend scheduler rebuilds split subgraphs every compute, so
//   those never repeat. Topology is (op, type, flags, name, ne, nb).
//
//   Device addresses (dst->data and src data) ARE part of the WP_HIP_GRAPHS
//   cache key. HIP's hipGraphExecUpdate is not a safe pointer-patch: s0
//   SIGSEGV'd SEGV_MAPERR in it (2026-08-20, compute_batch of a 17-expert
//   verify union) after we destroyed the source graph the exec was built
//   from. One shape-key + Update-per-expert was also why worker graphs
//   never replayed (every layer's experts looked like "same graph, new
//   ptrs"). Keying by (topo, addrs) makes a resident expert in the same
//   slot a pure Launch; a new slot is a new capture. CUDA without
//   WP_HIP_GRAPHS still keys on nodes[0] and uses ExecUpdate.

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ggml_cuda_graph_cache_policy {
    size_t  cap      = 256;
    int64_t ttl_us   = 10'000'000;
    int64_t sweep_us =  5'000'000;
};

template <typename Map>
size_t ggml_cuda_graph_cache_evict_ttl(Map & graphs, int64_t time_now, int64_t ttl_us) {
    size_t n = 0;
    for (auto it = graphs.begin(); it != graphs.end(); ) {
        if (time_now - it->second->last_used_time >= ttl_us) {
            it = graphs.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

// Evict LRU until graphs.size() < cap. `keep` is never removed (the key
// about to be returned). cap == 0 means no cap.
template <typename Map>
size_t ggml_cuda_graph_cache_evict_lru(Map & graphs, size_t cap, typename Map::key_type keep) {
    if (cap == 0) {
        return 0;
    }
    size_t n = 0;
    while (graphs.size() >= cap) {
        auto victim = graphs.end();
        int64_t oldest = 0;
        for (auto it = graphs.begin(); it != graphs.end(); ++it) {
            if (it->first == keep) {
                continue;
            }
            if (victim == graphs.end() || it->second->last_used_time < oldest) {
                victim = it;
                oldest = it->second->last_used_time;
            }
        }
        if (victim == graphs.end()) {
            break;
        }
        graphs.erase(victim);
        ++n;
    }
    return n;
}

inline uint64_t ggml_cuda_graph_fnv1a_mix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
    return h;
}

inline uint64_t ggml_cuda_graph_fnv1a_bytes(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Mix the fields that identify a node's captured kernel shape. Pointers
// and VIEW offsets are excluded so ephemeral split rebuilds and a
// moving KV write position hash to the same slot when the op/shape match.
inline uint64_t ggml_cuda_graph_mix_tensor_topo(uint64_t h, const ggml_tensor * t) {
    if (t == nullptr) {
        return ggml_cuda_graph_fnv1a_mix(h, 0);
    }
    h = ggml_cuda_graph_fnv1a_bytes(h, t->name, strnlen(t->name, GGML_MAX_NAME));
    h = ggml_cuda_graph_fnv1a_mix(h, (uint64_t) t->op);
    h = ggml_cuda_graph_fnv1a_mix(h, (uint64_t) t->type);
    h = ggml_cuda_graph_fnv1a_mix(h, (uint64_t) t->flags);
    h = ggml_cuda_graph_fnv1a_bytes(h, t->ne, sizeof(t->ne));
    h = ggml_cuda_graph_fnv1a_bytes(h, t->nb, sizeof(t->nb));
    return h;
}

// Device addresses only. Combined with mix_tensor_topo under WP_HIP_GRAPHS
// so a resident expert in the same slot hashes to the same cache entry.
inline uint64_t ggml_cuda_graph_mix_tensor_addrs(uint64_t h, const ggml_tensor * t) {
    if (t == nullptr) {
        return ggml_cuda_graph_fnv1a_mix(h, 0);
    }
    h = ggml_cuda_graph_fnv1a_mix(h, (uint64_t) (uintptr_t) t->data);
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * s = t->src[j];
        h = ggml_cuda_graph_fnv1a_mix(h, s ? (uint64_t) (uintptr_t) s->data : 0);
    }
    return h;
}

inline bool ggml_cuda_graph_tensor_is_view_or_noop(const ggml_tensor * t) {
    return t == nullptr ||
           t->op == GGML_OP_NONE ||
           t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE;
}

// True when two nodes describe the same kernel topology. Ignores object
// identity (src[], buffer, extra, view_src) and resolved device pointers.
// VIEW op_params hold the byte offset of the view — that is a pointer
// equivalent, not a kernel-shape change.
inline bool ggml_cuda_graph_tensor_topo_equal(const ggml_tensor & a, const ggml_tensor & b) {
    if (a.type != b.type || a.op != b.op || a.flags != b.flags) {
        return false;
    }
    if (memcmp(a.ne, b.ne, sizeof(a.ne)) != 0) {
        return false;
    }
    if (memcmp(a.nb, b.nb, sizeof(a.nb)) != 0) {
        return false;
    }
    if (strncmp(a.name, b.name, GGML_MAX_NAME) != 0) {
        return false;
    }
    if (a.op != GGML_OP_VIEW &&
        memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0) {
        return false;
    }
    return true;
}
