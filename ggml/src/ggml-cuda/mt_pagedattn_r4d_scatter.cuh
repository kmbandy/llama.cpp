#pragma once

// mt_pagedattn_r4d_scatter — device-side scatter for the libr4d paged fp8 KV
// cache (GGML_TYPE_R4D_FP8_KV).
//
// libr4d reads a paged KV cache laid out as ONE tensor per layer:
//   kv[num_blocks][kv_heads][block_size][2*head_dim]
// in raw (descale=1.0) OCP e4m3fn bytes, where each 512-byte slot is
// K[0..255] followed by V[0..255]. This scatter converts F16 k_cur/v_cur
// (as produced by the graph, [head_dim, n_kv_heads, n_tokens]) to that
// layout in one pass, indexed directly through slot_mapping (no per-seq
// prefix-sum walk needed — slot_mapping already carries -1 for padding).
//
// Only compiled when ggml-hip is built with -DGGML_HIP_R4D=ON (gfx1201).
// Non-HIP / non-R4D builds get the inline no-op stub below.

#include "common.cuh"

#include <cstdint>

namespace mt {

#if defined(GGML_HIP_R4D) && defined(GGML_USE_HIP)

// Scatter K_cur/V_cur (F16, [head_dim, n_kv_heads, n_tokens]) into the
// libr4d paged fp8 KV cache `kv` ([num_blocks][kv_heads][block_size]
// [2*head_dim] e4m3fn bytes, K then V per slot).
//
//   k_cur, v_cur   : [head_dim, n_kv_heads, n_tokens] F16, contiguous;
//                    element (d, h, t) at (t*n_kv_heads + h)*head_dim + d.
//   kv             : device buffer for one layer's combined K|V cache
//                    (this IS ggml's k_cache tensor for GGML_TYPE_R4D_FP8_KV;
//                    v_cache of the same type is allocated but unused).
//   slot_mapping   : [n_tokens] destination slot per token, or -1 to skip
//                    (padding). slot = block_idx*block_size + slot_in_block.
//   q_lens         : [num_seqs] tokens per sequence. Not needed for indexing
//                    here (slot_mapping already addresses tokens directly)
//                    but kept in the signature for parity with the other
//                    paged scatter launchers and possible future per-seq
//                    bookkeeping (e.g. debug dumps).
//   num_seqs       : number of live sequences (informational only here).
//   n_tokens       : number of tokens in k_cur/v_cur.
//   n_kv_heads     : number of KV heads.
//   head_dim       : per-head dimension (element count of K, and of V).
//   block_size     : tokens per paged-cache block.
void mt_r4d_scatter_kv(const half * k_cur, const half * v_cur, uint8_t * kv,
                       const int32_t * slot_mapping, const int32_t * q_lens,
                       int num_seqs, int n_tokens, int n_kv_heads, int head_dim,
                       int block_size, cudaStream_t stream);

#else  // GGML_HIP_R4D undefined, or non-HIP build — stub out

inline void mt_r4d_scatter_kv(const half *, const half *, uint8_t *,
                              const int32_t *, const int32_t *,
                              int, int, int, int,
                              int, cudaStream_t) {
    // Unreachable: the ggml_cuda op-support predicate for GGML_TYPE_R4D_FP8_KV
    // only accepts the type when the backend was built with GGML_HIP_R4D.
}

#endif

}  // namespace mt
