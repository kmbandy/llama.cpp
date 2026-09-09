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
