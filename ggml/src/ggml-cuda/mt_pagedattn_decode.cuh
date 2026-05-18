// mt_pagedattn_decode — flash-decode paged attention kernel for q_len <= 8.
//
// The WMMA tile kernel (mt_pagedattn_tile.cu) is great for batched prefill
// but wastes ~15/16 of WMMA throughput at q_len=1 AND launches only
// (n_heads, n_seqs) blocks — no parallelism along the KV dimension. At
// long context (we hit ~400K) the scalar kernel collapses to 0.19 t/s
// because each block iterates all KV tokens serially with a per-token
// block_reduce_sum. See MAD-185.
//
// This kernel splits KV into chunks of CHUNK_KV tokens. Each block scans
// one (head, seq, chunk), producing a partial (running max, sum, V@logits).
// A second kernel reduces partials per (head, seq) via online-softmax
// recurrence.
//
// Targets HEAD_SIZE=128, BLOCK_SIZE=16, CACHE_TYPE in {F16, TURBO4_0}.
// Dispatch gate (in mt_pagedattn.cu): q_len <= 8 AND valid_ctx >= 8192
// AND amd_wmma_available(cc) (actually we don't need WMMA here, but the
// gate runs after the tile gate). Tile kernel handles q_len >= 16
// prefill; scalar handles short-ctx decode.

#pragma once

#include "mt_pagedattn.cuh"

#include <cmath>

namespace mt {

// Pass 1: per-chunk partial computation. Each block handles one
// (head, seq, kv_chunk). Writes a per-chunk partial:
//
//   partials[head, seq, chunk] = (V_acc[HEAD_SIZE], max, sum)   fp32
//
// Layout in memory:
//   partials[((h * n_seqs + s) * num_chunks + c) * (HEAD_SIZE + 2) + d]
//     d in [0, HEAD_SIZE):   V_acc element
//     d == HEAD_SIZE:        running max
//     d == HEAD_SIZE+1:      running sum
template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
void launch_paged_attn_decode(
    __half         * out,
    const __half   * q,
    const void     * k_cache,
    const void     * v_cache,
    const int32_t  * block_tables,
    const int32_t  * context_lens,
    const int32_t  * q_lens,
    float          * partials_scratch,   // [n_heads * n_seqs * num_chunks * max_q_len * (HEAD_SIZE+2)]
    int             num_seqs,
    int             n_heads,
    int             n_kv_heads,
    int             max_blocks_per_seq,
    int             max_ctx_len,
    int             max_q_len,           // upper bound on per-seq q_len in this batch
    float           scale,
    cudaStream_t    stream);

// Helper: number of KV chunks for a given ctx, given the kernel's CHUNK_KV.
// Exposed so the dispatch site can size the partials scratch.
int paged_attn_decode_num_chunks(int max_ctx_len);

} // namespace mt
