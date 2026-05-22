// MAD-214 Phase 1G-A: runtime registry for turbo-FP8 centroid LUTs.
//
// Architecture: per-model, per-layer, per-direction (K vs V) centroid LUTs
// are loaded lazily from disk on first attention call. If the model's LUTs
// don't exist yet, the registry triggers an in-process calibration warmup
// (Phase 1G-E) that captures K/V samples from a built-in corpus, fits
// constrained Lloyd-Max centroids (Phase 1G-D), and writes the LUT files
// before serving any user prompts.
//
// File layout: ~/.cache/llama.cpp/turbo-fp8/<model_fingerprint>/
//                 l<layer_idx>_k.bin   (16 E4M3 bytes — centroid LUT for K)
//                 l<layer_idx>_v.bin   (16 E4M3 bytes — centroid LUT for V)
//                 manifest.json        (n_layers, head_dim, fingerprint metadata)
//
// Per-process lifetime: registry holds device buffers and pointer table
// keyed by (layer, k_or_v); the table is built on first lookup and reused
// for the rest of the process. Thread-safe (mutex around init).

#pragma once

#include <cstdint>
#include <string>

namespace mt_turbo_fp8 {

enum kv_dir { KV_K = 0, KV_V = 1 };

// Stable identity for a model. Computed from architecture-shape identifiers
// (arch_name + n_layer + n_embd + head_dim + n_kv_heads) — NOT the full
// GGUF hash, which would be too slow on a 35B model. Same arch+shape gets
// the same LUTs regardless of the specific weights file path.
struct model_fingerprint {
    std::string arch;        // e.g. "qwen3moe", "llama"
    int         n_layer;     // total layers (including non-attention for hybrid arch)
    int         n_embd;      // hidden dim
    int         head_dim;    // per-head dim
    int         n_kv_heads;  // KV head count

    // Hex digest used as the on-disk cache key.
    std::string digest() const;
};

// Initialize the registry for a model. Sets the active fingerprint and the
// cache directory under ~/.cache/llama.cpp/turbo-fp8/<digest>/. Safe to
// call multiple times for the same fingerprint; no-op after first.
//
// If `auto_calibrate_if_missing` is true and any LUT file is absent for
// this fingerprint, the registry will (in Phase 1G-E) request the host
// to perform calibration warmup; until that lands, the registry falls
// back to the embedded qwen3.5-4B LUTs as a stop-gap and logs a WARN.
//
// Returns true on success, false on I/O or device alloc failure.
bool init(const model_fingerprint & fp, bool auto_calibrate_if_missing);

// Returns the device pointer to the 16-byte E4M3 centroid LUT for the
// given (layer, kv_dir). Lazy-loads + uploads on first call. Subsequent
// calls return the cached device pointer.
//
// Returns nullptr if init() hasn't been called or the LUT can't be
// resolved (file missing AND auto-calibration disabled).
const uint8_t * get_lut_device_ptr(int layer, kv_dir dir);

// Returns true iff every (layer, k|v) LUT file exists on disk for the
// active fingerprint. Used by Phase 1G-E to decide whether to trigger
// calibration warmup.
bool all_luts_cached();

// Frees device buffers and clears the in-memory cache. Called at process
// shutdown — usually unnecessary since the process is about to exit.
void shutdown();

}  // namespace mt_turbo_fp8
