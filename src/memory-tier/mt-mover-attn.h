#pragma once

// AttentionMover — synchronous K/V copy primitives for one token at a
// position in a per-layer K/V tensor.
//
// Four primitives:
//   evict_k   VRAM[layer.k @ pos]         -> host buf
//   restore_k host buf                     -> VRAM[layer.k @ pos]
//   evict_v   VRAM[layer.v @ pos]         -> host buf  (gather if v_trans)
//   restore_v host buf                     -> VRAM[layer.v @ pos]  (scatter if v_trans)
//
// Synchronous via hipMemcpy / hipMemcpy2D — same pattern as the legacy
// pager and the legacy llama_kv_cache_tiered. Phase 2 follow-up will
// add async overlap with proper stream-event ordering once correctness
// is locked.
//
// Layouts (ggml semantics):
//   K is always contiguous: token p starts at byte offset p * k_row_bytes.
//   V when !v_trans: same as K.
//   V when v_trans: stored transposed [kv_size, n_embd_v]. Token p's
//     element j is at flat offset (j * kv_size + p) * v_elem_bytes.
//     Gather/scatter via hipMemcpy2D handles this as a 2D column copy.
//
// Row-stride bytes use AttentionLayerView::{k,v}_row_bytes which the
// caller must populate from tensor->nb[1] — using ne[0]*element_size
// is wrong for block-quantized types (B-K1 in the bug catalog).
//
// HIP-only in Phase 2. Non-HIP builds get a stub that returns false
// from every method (matches GpuTransport's approach).

#include "mt-inner-access.h"

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace mt {

class AttentionMover {
public:
    AttentionMover() = default;

    // K, contiguous: hipMemcpy(dst, k->data + pos*k_row_bytes, k_row_bytes, D2H).
    bool evict_k(const AttentionLayerView & layer,
                 int64_t                    token_pos,
                 void *                     dst_host) const;

    bool restore_k(const AttentionLayerView & layer,
                   const void *               src_host,
                   int64_t                    token_pos) const;

    // V — gather (v_trans) or contiguous (!v_trans). Caller's buffer is
    // ALWAYS contiguous of size layer.v_row_bytes; this class handles the
    // transposed case internally via hipMemcpy2D.
    bool evict_v(const AttentionLayerView & layer,
                 int64_t                    token_pos,
                 void *                     dst_host) const;

    bool restore_v(const AttentionLayerView & layer,
                   const void *               src_host,
                   int64_t                    token_pos) const;
};

}  // namespace mt
