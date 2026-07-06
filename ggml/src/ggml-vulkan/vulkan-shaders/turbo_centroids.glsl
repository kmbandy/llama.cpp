// Centroid and midpoint tables for turbo4_0 4-bit PolarQuant.
// Verbatim from ggml/src/ggml-cuda/turbo-quant.cuh:297-311.
// Included by cpy_f32_turbo4_0.comp; Task 3 dequant shader may also include it.

#ifndef TURBO_CENTROIDS_GLSL
#define TURBO_CENTROIDS_GLSL

// Lloyd-Max optimal centroids for N(0, 1/128) — 16 levels
const float TURBO_CENTROIDS_4BIT[16] = float[](
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
);

// Midpoints between adjacent centroids — for nearest-centroid lookup
const float TURBO_MID_4BIT[15] = float[](
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
);

// Centroids/midpoints for turbo4_64 (64-element blocks), calibrated from real
// LFM2.5-8B-A1B K/V activation statistics (2026-07-01 investigation), NOT the
// N(0, 1/128) Gaussian assumption above. A 64-element group's L2-normalized
// value has ~sqrt(2)x the typical magnitude of a 128-element group's, AND
// real K/V activations are heavy-tailed (occasional outlier channels), not
// Gaussian — the shared TURBO_CENTROIDS_4BIT table (designed for turbo4_0's
// 128-element blocks) hard-clips ~13% of real turbo4_64 values and has 2.8x
// worse RMSE than this table on real data. Used ONLY by turbo4_64 (native
// head_dim-64 paged KV cache); does NOT affect turbo4_0 (128-element blocks,
// e.g. Qwen3.5-4B's hd256 path), which keeps its existing table unchanged.
// Verbatim from ggml/src/ggml-cuda/turbo-quant.cuh's TURBO_CENTROIDS_4BIT_N64.
const float TURBO_CENTROIDS_4BIT_N64[16] = float[](
    -0.489086f, -0.332636f, -0.244498f, -0.182456f,
    -0.132429f, -0.089625f, -0.051251f, -0.016052f,
     0.016052f,  0.051251f,  0.089625f,  0.132429f,
     0.182456f,  0.244498f,  0.332636f,  0.489086f
);

const float TURBO_MID_4BIT_N64[15] = float[](
    -0.410861f, -0.288567f, -0.213477f, -0.157443f,
    -0.111027f, -0.070438f, -0.033652f,  0.000000f,
     0.033652f,  0.070438f,  0.111027f,  0.157443f,
     0.213477f,  0.288567f,  0.410861f
);

#endif // TURBO_CENTROIDS_GLSL
