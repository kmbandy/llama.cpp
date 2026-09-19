#pragma once

// mt_pagedattn_r4d — R4D-backed path for the mt:: paged attention op.
//
// Active only when ggml-hip is built with GGML_HIP_R4D=ON (gfx1201 offload arch requested; see
// ggml/src/ggml-hip/CMakeLists.txt) AND the runtime flag MAD_USE_R4D=1 is set AND the current HIP
// device is actually gfx1201 at runtime (ggml_cuda_r4d_available(), ggml-cuda/r4d/ggml-r4d.h).
//
// R4D (ggml-cuda/r4d/, vendored from radiance-libr4d commit b9e42ab-rx6) supplies hand-tuned
// fp8-KV paged causal attention at exactly head_dim=256, GQA-6, block_size=16 — see r4d.h for the
// full contract. This path only ever claims that one shape; every other (head_dim, GQA, block
// size, cache dtype) combination falls through unchanged to the tile/decode/scalar/AITER paths
// this same op already supports. The KV cache layout R4D expects (GGML_TYPE_R4D_FP8_KV, a
// combined K|V fp8-e4m3 paged buffer) is different from every other cache type this op knows
// about, so — same as the AITER path above it — the choice is mutually exclusive with those per
// tensor: a k_cache of type R4D_FP8_KV can only ever be served here.
//
// MAD-406 (R4D integration).

#include "common.cuh"

namespace mt {

#ifdef GGML_HIP_R4D

// Runtime gate. Reads env var MAD_USE_R4D once (cached) and ANDs it with ggml_cuda_r4d_available()
// (compiled in AND current device is gfx1201). Compiled out entirely when GGML_HIP_R4D is
// undefined.
bool r4d_backend_enabled();

// R4D-path dispatch entry. Same tensor/op_params contract as ggml_cuda_op_paged_attn_mt.
//
// Returns true iff this call fully handled the op (scatter + attention already launched into
// dst->data) — the caller must `return` immediately in that case. Returns false to signal that
// this call did NOT touch the cache or dst at all (a pure eligibility/geometry miss, or a
// once-per-graph capture-safety deferral) and the caller should fall through to its other paths
// exactly as if this function had never been called.
//
// IMPORTANT: once this function's KV scatter has run (i.e. once it has committed to the R4D
// layout for this call), it no longer returns false — a return-code rejection from the R4D
// entry point itself past that point is a GGML_ABORT, not a fallback, because the cache is now in
// R4D's fp8 layout and the other paths' scatter/attention kernels do not understand it.
bool ggml_cuda_op_paged_attn_mt_r4d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else  // GGML_HIP_R4D undefined — stub out

inline bool r4d_backend_enabled() { return false; }
inline bool ggml_cuda_op_paged_attn_mt_r4d(ggml_backend_cuda_context &, ggml_tensor *) {
    // Unreachable: caller must check r4d_backend_enabled() first, and it is always false here.
    return false;
}

#endif

}  // namespace mt
