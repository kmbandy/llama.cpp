// ml8-k AllReduce wire codebooks.
// Fitted by Lloyd-Max on 2M normalized values pooled from REAL captured
// AllReduce partials across 16 layers, then snapped to the E4M3 lattice.
// SIGNED codebook: k bits per element, no separate sign bit.
// Block = 32 elements + one fp16 scale, MATCHING the ggml q-family block so
// bit budgets are directly comparable (k + 0.5 bits/element).
// Baked constants => both ranks encode identically by construction.
#pragma once

#define ML8_4_N_CENT 16   // 4.50 bits/element
static __device__ __constant__ float ML8_4_CENTROIDS[16] = {
    -0.93750000f, -0.75000000f, -0.56250000f, -0.43750000f, -0.31250000f, -0.21875000f, -0.12500000f, -0.03906250f,
    +0.04296875f, +0.12500000f, +0.21875000f, +0.31250000f, +0.43750000f, +0.56250000f, +0.75000000f, +0.93750000f,
};

#define ML8_5_N_CENT 32   // 5.50 bits/element
static __device__ __constant__ float ML8_5_CENTROIDS[32] = {
    -1.00000000f, -0.81250000f, -0.68750000f, -0.62500000f, -0.50000000f, -0.43750000f, -0.37500000f, -0.31250000f,
    -0.28125000f, -0.23437500f, -0.18750000f, -0.15625000f, -0.11718750f, -0.07812500f, -0.04687500f, -0.01367188f,
    +0.01757812f, +0.05078125f, +0.08593750f, +0.11718750f, +0.15625000f, +0.18750000f, +0.23437500f, +0.28125000f,
    +0.31250000f, +0.37500000f, +0.43750000f, +0.50000000f, +0.62500000f, +0.68750000f, +0.81250000f, +1.00000000f,
};

// ML8_6_CENTROIDS: derived the same way as ML8_4/ML8_5 -- Lloyd-Max on
// normalized AllReduce-partial samples, then snapped to the E4M3 lattice --
// EXCEPT the real 2M-sample pooled captured-partials dump that ML8_4/ML8_5
// were fit on is not present in this checkout, so this table is fit on a
// Laplace(0, b=0.2500) proxy distribution instead (b chosen to minimize RMSE
// against the checked-in ML8_4 table). Validation against the real tables:
//   k=16 vs checked-in ML8_4_CENTROIDS RMSE = 0.00655
//   k=32 vs checked-in ML8_5_CENTROIDS RMSE = 0.03459
// (nonzero but small RMSE is expected: this is a Laplace proxy for the real
// captured-partials dump the original tables were fit on, not a
// reproduction of it.) See the derivation script and its docstring (not
// checked into this repo -- delivered alongside this change; move it under
// scripts/calibration/ if it should be kept):
//   /tmp/claude-1000/-home-kmbandy/275b3362-366e-4ff9-80e5-cc265704ba5b/scratchpad/ml8_6_centroids.py
//   (run: python3 ml8_6_centroids.py)
// Re-fit against the real captured-partials dump instead of the Laplace
// proxy if/when that dump (or its generating script) can be located.
//
// KNOWN CAVEAT (do not silently assume otherwise): the two sides are fit
// independently on asymmetric random draws and then E4M3-snapped, so this
// table is NOT exactly symmetric about zero (e.g. -0.03906250f vs.
// +0.03515625f at the 4th step from center) and is NOT strictly increasing
// -- it has two duplicate-value plateaus where independent bins snapped to
// the same E4M3 lattice point (-0.28125000f repeated at indices 13-14,
// +0.28125000f repeated at indices 49-50), so only 62 of the 64 entries are
// distinct. ml8_nearest() (linear nearest-centroid scan) is still correct
// with duplicates/plateaus -- it just means 2 of the 64 codes are
// unreachable (ties break to the lower index), costing a negligible
// fraction of a bit of effective codebook size. Re-fit with a larger
// N_SAMPLES / different seed in the script if an exactly monotone, exactly
// symmetric table is required.
#define ML8_6_N_CENT 64   // 6.50 bits/element
static __device__ __constant__ float ML8_6_CENTROIDS[64] = {
    -1.00000000f, -0.87500000f, -0.81250000f, -0.75000000f, -0.68750000f, -0.62500000f, -0.56250000f, -0.50000000f,
    -0.46875000f, -0.43750000f, -0.37500000f, -0.34375000f, -0.31250000f, -0.28125000f, -0.28125000f, -0.23437500f,
    -0.21875000f, -0.20312500f, -0.17187500f, -0.15625000f, -0.14062500f, -0.12500000f, -0.10937500f, -0.10156250f,
    -0.08593750f, -0.07031250f, -0.05859375f, -0.04687500f, -0.03906250f, -0.02539062f, -0.01367188f, -0.00585938f,
    +0.00585938f, +0.01367188f, +0.02539062f, +0.03515625f, +0.04687500f, +0.05859375f, +0.07031250f, +0.08593750f,
    +0.10156250f, +0.10937500f, +0.12500000f, +0.14062500f, +0.15625000f, +0.17187500f, +0.20312500f, +0.21875000f,
    +0.23437500f, +0.28125000f, +0.28125000f, +0.31250000f, +0.34375000f, +0.37500000f, +0.43750000f, +0.46875000f,
    +0.50000000f, +0.56250000f, +0.62500000f, +0.68750000f, +0.75000000f, +0.81250000f, +0.87500000f, +1.00000000f,
};
