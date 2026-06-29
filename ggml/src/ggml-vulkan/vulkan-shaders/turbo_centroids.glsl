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

#endif // TURBO_CENTROIDS_GLSL
