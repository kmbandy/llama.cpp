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
//   Device addresses are compared AFTER lookup, not mixed into the key.
//   HIP cannot ExecUpdate (s0 SIGSEGV SEGV_MAPERR, 2026-08-20). Mixing
//   addrs into the key made every gallocr/activation pointer a unique
//   graph so we recaptured 100% of computes — slower than eager. Equal
//   addrs -> Launch; unequal -> eager (next stable pair recaptures).
//   CUDA without WP_HIP_GRAPHS still keys on nodes[0] and uses ExecUpdate.

#pragma once

#include "ggml.h"
#include "ggml-impl.h" // full ggml_cgraph definition (ggml_cuda_graph_is_prefill_shaped)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

struct ggml_cuda_graph_cache_policy {
    size_t  cap      = 256;
    int64_t ttl_us   = 10'000'000;
    int64_t sweep_us =  5'000'000;
    bool    track_ttl = false;
};

template <typename Map>
size_t ggml_cuda_graph_cache_evict_ttl(
        Map & graphs, int64_t time_now, int64_t ttl_us,
        std::unordered_set<typename Map::key_type> * ttl_evicted_keys = nullptr) {
    size_t n = 0;
    for (auto it = graphs.begin(); it != graphs.end(); ) {
        if (time_now - it->second->last_used_time >= ttl_us) {
            if (ttl_evicted_keys != nullptr) {
                ttl_evicted_keys->insert(it->first);
            }
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

// WP_HIP_GRAPH_KEY_FAST=0 restores the byte-at-a-time FNV mixing below.
// Default is the word-at-a-time variant: byte-wise FNV is a fully serial
// xor->imul dependency chain, and mix_tensor_topo hashes ~84 bytes per tensor
// across (node + up to GGML_MAX_SRC srcs) for every node of every split, every
// token. Word-wise mixes the same *information* 8x fewer rounds.
//
// This is safe irrespective of hash quality: the graph key only selects a cache
// slot. After lookup, ggml_cuda_graph_update_required() compares the stored
// node_props against the live nodes, so a collision shows up as a topology
// change and forces a recapture. It can cost a capture; it cannot launch the
// wrong graph.
inline bool ggml_cuda_graph_key_fast() {
    static const bool fast = [] {
        const char * e = std::getenv("WP_HIP_GRAPH_KEY_FAST");
        return !(e != nullptr && e[0] == '0');
    }();
    return fast;
}

// n must be a multiple of 8 and p suitably aligned (ne/nb arrays are).
inline uint64_t ggml_cuda_graph_fnv1a_words(uint64_t h, const void * p, size_t n) {
    const uint64_t * w = static_cast<const uint64_t *>(p);
    for (size_t i = 0; i < n / 8; ++i) {
        h = ggml_cuda_graph_fnv1a_mix(h, w[i]);
    }
    return h;
}

// Length-prefixed word-wise hash of an arbitrary byte range.
inline uint64_t ggml_cuda_graph_fnv1a_str(uint64_t h, const char * s, size_t n) {
    h = ggml_cuda_graph_fnv1a_mix(h, (uint64_t) n);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, s + i, sizeof(w));
        h = ggml_cuda_graph_fnv1a_mix(h, w);
    }
    if (i < n) {
        uint64_t w = 0;
        memcpy(&w, s + i, n - i);
        h = ggml_cuda_graph_fnv1a_mix(h, w);
    }
    return h;
}

// Mix the fields that identify a node's captured kernel shape. Pointers
// and VIEW offsets are excluded so ephemeral split rebuilds and a
// moving KV write position hash to the same slot when the op/shape match.
// ggml auto-names anonymous tensors "node_<N>"/"leaf_<N>" with a GLOBAL
// monotonically increasing counter, so a fragment containing one gets a
// different name on EVERY graph build — its key can never repeat and capture
// churns forever (measured 2026-08-23: 35% permanent fallback, node_1734 vs
// node_2173 the only differing topo field). Canonicalize: for such names,
// hash/compare only the prefix and any non-digit suffix.
inline size_t ggml_cuda_graph_canon_name_len(const char * name) {
    size_t n = strnlen(name, GGML_MAX_NAME);
    const char * p = nullptr;
    if (n > 5 && strncmp(name, "node_", 5) == 0) { p = name + 5; }
    if (n > 5 && strncmp(name, "leaf_", 5) == 0) { p = name + 5; }
    if (p == nullptr) { return n; }
    const char * q = p;
    while (*q >= '0' && *q <= '9') { ++q; }
    if (q == p) { return n; }          // no digits: not an auto-name
    return (size_t)(p - name);         // keep "node_"/"leaf_" prefix only
}

inline uint64_t ggml_cuda_graph_mix_tensor_topo(uint64_t h, const ggml_tensor * t) {
    if (t == nullptr) {
        return ggml_cuda_graph_fnv1a_mix(h, 0);
    }
    const size_t name_len = ggml_cuda_graph_canon_name_len(t->name);
    // op/type/flags all fit in one word, so fold them into a single mix.
    const uint64_t meta = (uint64_t) t->op
                        | ((uint64_t) t->type  << 16)
                        | ((uint64_t) t->flags << 32);
    if (ggml_cuda_graph_key_fast()) {
        h = ggml_cuda_graph_fnv1a_str(h, t->name, name_len);
        h = ggml_cuda_graph_fnv1a_mix(h, meta);
        h = ggml_cuda_graph_fnv1a_words(h, t->ne, sizeof(t->ne));
        h = ggml_cuda_graph_fnv1a_words(h, t->nb, sizeof(t->nb));
        return h;
    }
    h = ggml_cuda_graph_fnv1a_bytes(h, t->name, name_len);
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
    {
        const size_t la = ggml_cuda_graph_canon_name_len(a.name);
        const size_t lb = ggml_cuda_graph_canon_name_len(b.name);
        if (la != lb || strncmp(a.name, b.name, la) != 0) {
            return false;
        }
    }
    if (false) {
        return false;
    }
    if (a.op != GGML_OP_VIEW &&
        memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0) {
        return false;
    }
    return true;
}

// Prefill-shaped graphs (activation width > decode/spec window) must not be
// captured. Each distinct prompt length would otherwise park a HIP graph and
// its host-visible scratch for the process lifetime — the 2057/1136/1010 MiB
// /dev/zero set on the sliced spine. Decode/spec stay capturable (ne <= 32).
//
// 2026-09-03: the MUL_MAT-only test missed most prefill fragments on the sliced
// spine (a split of ~86 fragments per ubatch holds norms / rope / GDN / mul_mat_id
// and often no MUL_MAT at all), so every prefill fragment was captured and
// instantiated: ~258 captures per 1372-token prompt, ~430 MiB of runtime scratch
// on the R9700, which on a card within 1 GB of full evicts 3-6 GB to GTT and
// OOMs the 15 GB host. Any op node whose output is a plain 2-D activation
// [n, n_tokens] wider than the decode/spec window is prefill-shaped, and so is
// a MUL_MAT_ID whose activations carry more than 32 tokens.
//
// 2026-09-05: GGML_OP_PAGED_ATTN_MT (mt:: paged attention, MAD-114) and the
// other fork-only ops were never added to this switch, so a prefill fragment
// whose only wide node is one of them was misclassified as decode-shaped and
// captured -- the pagedattn tile/decode dispatch inside it then runs for the
// first time *while the stream is mid-capture*, and whatever in that path
// isn't capture-safe poisons the capture ("operation failed due to a previous
// error during capture" on the next kernel launch, e.g. WP_HIP_GRAPHS=1 on the
// qwen38-27b paged-attn preset). Each op's real token-count source is used
// (not always the dst ne[], since GATED_DELTA_NET/SSM_SCAN concatenate
// recurrent state onto the dst and TURBO_WHT mirrors whatever shape its input
// happens to have) rather than folding them into the generic 2-D/ROPE-shaped
// cases above, matching how MUL_MAT_ID already reads its own src instead of
// dst.
inline bool ggml_cuda_graph_is_prefill_shaped(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[1] != nullptr) {
            const int64_t n_act = node->src[1]->ne[1];
            if (n_act > 32 && node->src[1]->ne[0] <= 16384) {
                return true;
            }
        }
        if (node->op == GGML_OP_MUL_MAT_ID && node->src[1] != nullptr) {
            if (node->src[1]->ne[2] > 32) {
                return true;
            }
        }
        // GATED_DELTA_NET's dst is a flat [S_v*H, n_tokens*n_seqs + state_rows]
        // buffer (ggml_gated_delta_net, ggml.c) -- the recurrent-state rows
        // tacked on the end make dst->ne[1] useless as a token count. Read the
        // true per-call token count from src[2] (v: [S_v, H, n_tokens, n_seqs]).
        if (node->op == GGML_OP_GATED_DELTA_NET && node->src[2] != nullptr) {
            if (node->src[2]->ne[2] > 32) {
                return true;
            }
        }
        // SSM_SCAN's dst is a flat 1-D [nelements(x) + K*state] buffer
        // (ggml_ssm_scan, ggml.c) for the same reason -- read the token count
        // from src[1] (x: [head_dim, n_head, n_seq_tokens, n_seqs]).
        if (node->op == GGML_OP_SSM_SCAN && node->src[1] != nullptr) {
            if (node->src[1]->ne[2] > 32) {
                return true;
            }
        }
        // Token-width rules per op. A blanket "any 2-D node wider than 32"
        // test also caught decode fragments (recurrent-state copies are 2-D
        // and wide), which halved graph replays on decode; scope it to ops
        // whose 2-D output is [n, n_tokens].
        switch (node->op) {
            case GGML_OP_RMS_NORM:
            case GGML_OP_NORM:
            case GGML_OP_GET_ROWS:
            case GGML_OP_GLU:
            case GGML_OP_ADD:
            case GGML_OP_MUL:
            case GGML_OP_SCALE:
                if (node->ne[2] == 1 && node->ne[3] == 1 && node->ne[1] > 32) {
                    return true;
                }
                break;
            case GGML_OP_ROPE:
                if (node->ne[2] > 32) {
                    return true;
                }
                break;
            case GGML_OP_SSM_CONV:
                if (node->ne[1] > 32) {
                    return true;
                }
                break;
            // dst mirrors q: [head_dim, n_heads, sum(q_lens), 1]
            // (ggml_paged_attn_mt, ggml.c) -- token count is ne[2], same
            // convention as ROPE above.
            case GGML_OP_PAGED_ATTN_MT:
                if (node->ne[2] > 32) {
                    return true;
                }
                break;
            // dst is [n_embd, n_tokens] (ggml_dsv4_hc_pre, ggml.c) -- fits the
            // plain 2-D convention above but the op isn't in that op list.
            case GGML_OP_DSV4_HC_PRE:
                if (node->ne[2] == 1 && node->ne[3] == 1 && node->ne[1] > 32) {
                    return true;
                }
                break;
            // dst is [hc, hc, n_tokens] / [n_embd, hc, n_tokens]
            // (ggml_dsv4_hc_comb / ggml_dsv4_hc_post, ggml.c) -- token count
            // is ne[2], same convention as ROPE.
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_POST:
                if (node->ne[2] > 32) {
                    return true;
                }
                break;
            // dst is [k->ne[2] (n_kv), q->ne[2] (n_tokens), 1, q->ne[3]]
            // (ggml_lightning_indexer, ggml.c) -- ne[3] (n_seqs) need not be 1,
            // so this can't share the plain-2-D case above; token count is
            // ne[1].
            case GGML_OP_LIGHTNING_INDEXER:
                if (node->ne[1] > 32) {
                    return true;
                }
                break;
            // dst shape mirrors src[0] exactly (ggml_turbo_wht, ggml.c), and
            // callers apply it to both 4-D per-head activations
            // ([head_dim, n_head, n_tokens, n_seqs], e.g. q) and flattened 2-D
            // ones ([n_embd, n_tokens], e.g. kqv) -- ne[2] is the token axis
            // in the former, ne[1] in the latter (ne[2]==1 there).
            case GGML_OP_TURBO_WHT: {
                const int64_t width = node->ne[2] > 1 ? node->ne[2] : node->ne[1];
                if (width > 32) {
                    return true;
                }
                break;
            }
            default:
                break;
        }
    }
    return false;
}
