#pragma once

// RecurrentStateMover — synchronous copy of one sequence's full
// recurrent state (r + s across all recurrent layers) to/from a
// contiguous host buffer.
//
// Migration unit is the whole sequence's recurrent state. This is
// fundamentally different from the attention path:
//   - Attention: per-token K/V, ~1 KB per token per layer.
//   - Recurrent: per-sequence r/s, several MB per sequence (independent
//     of token count). For Qwen3.5-REAP-97B with 36 recurrent layers
//     this is ~149 MiB per seq.
//
// New work in Phase 2 — the legacy llama_kv_cache_tiered did not tier
// recurrent state at all (B-K5 in the bug catalog).
//
// HIP-only. Stub for non-HIP builds (return false everywhere).

#include "mt-inner-access.h"

#include <cstddef>
#include <cstdint>

namespace mt {

class RecurrentStateMover {
public:
    RecurrentStateMover() = default;

    // Evict a sequence's recurrent state to a host buffer.
    //
    // Layout in `dst_host`:
    //     [r tensors of all layers concatenated][s tensors of all layers concatenated]
    //
    // Total bytes written = seq.r_bytes_total + seq.s_bytes_total. The
    // caller provides at least that many bytes. Each tensor is copied
    // verbatim (its full payload, not per-token) because recurrent state
    // is per-sequence as a whole.
    bool evict_seq(const RecurrentStateView & seq, void * dst_host) const;

    // Restore: dst_host's contents must match the layout produced by
    // evict_seq. Bytes are written back into the same per-layer r/s
    // tensors.
    bool restore_seq(const RecurrentStateView & seq, const void * src_host) const;
};

}  // namespace mt
