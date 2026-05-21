// MAD-214 Phase 1E: Walsh-Hadamard rotation kernel for the turbo-FP8 KV
// cache path. Shared across all backends (AITER+RDNA4, paged-tile+RDNA4,
// and future NVIDIA Hopper/Blackwell) — pure HIP source that ggml-cuda's
// build infrastructure compiles for both HIP and CUDA via macros.
//
// Why Hadamard:
//   Per Phase 0 calibration, applying Walsh-Hadamard rotation along the
//   head_dim axis before quantizing K to turbo-FP8 improves MSE by ~12% on
//   Qwen3.5-4B. The rotation spreads outlier channels across all dimensions
//   so per-block scaling can fit non-uniform centroid LUTs more tightly.
//
// Math identity preserved:
//   For attention scores QK^T, applying H to both Q and K leaves the result
//   unchanged: (QH)·(KH)^T = QH·H^T·K^T = Q·(HH^T)·K^T = Q·I·K^T = QK^T.
//   We rotate K at KV-cache write time and rotate Q at inference time.
//
// We rotate K only (not V):
//   Phase 0 showed K-only = 12% MSE gain; K+V = 15% (only +3% extra).
//   K-only requires no inverse rotation on the attention output (V flows
//   through unchanged). +3% MSE isn't worth the inverse-rotation kernel
//   complexity for Phase 1. (Future MAD ticket if needed.)
//
// FWHT (Fast Walsh-Hadamard Transform):
//   In-place butterfly, O(d log d) per vector vs O(d^2) naive matmul.
//   d must be power of 2; we support d ∈ {16, 32, 64, 128, 256, 512, 1024}.
//   Each stage: pair elements at stride 1, 2, 4, ..., d/2 and replace
//   (a, b) → (a+b, a-b). Final normalize by 1/sqrt(d).
//
// Build: included by ggml-cuda translation units; no separate compile.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────
// Device kernel: one block per row, head_dim threads.
// Applies FWHT in-place to a single row of length D (D = blockDim.x, must
// be a power of 2 ≤ 1024).
// ─────────────────────────────────────────────────────────────────────────
template <int D>
__global__ void mt_turbo_fp8_fwht_kernel(float * __restrict__ data, int n_rows, int row_stride) {
    static_assert((D & (D - 1)) == 0, "D must be a power of 2");
    static_assert(D >= 2 && D <= 1024, "D must be in [2, 1024]");

    const int row = blockIdx.x;
    if (row >= n_rows) return;

    __shared__ float smem[D];

    const int tid = threadIdx.x;
    float * row_ptr = data + row * row_stride;

    // Load row into shared memory
    smem[tid] = row_ptr[tid];
    __syncthreads();

    // FWHT in-place: log2(D) stages, butterflies at stride 1, 2, 4, ..., D/2
    #pragma unroll
    for (int stage = 0; (1 << stage) < D; ++stage) {
        const int stride = 1 << stage;
        // Pair element index for this stage:
        //   For each thread t, its pair partner is t XOR stride.
        //   Threads where (t & stride) == 0 compute (a + b), partners compute (a - b).
        const int partner = tid ^ stride;
        const float a = smem[tid];
        const float b = smem[partner];
        __syncthreads();
        // The "lower" partner of the pair gets (a + partner_value); the "upper" gets (lower_value - partner_value).
        // Since both threads ran the same load, both have (a, b). We disambiguate by bit:
        if ((tid & stride) == 0) {
            smem[tid] = a + b;
        } else {
            smem[tid] = b - a;  // here b is OUR value, a is partner's lower; we want (partner_value - our_value)... wait
        }
        __syncthreads();
    }

    // Normalize by 1/sqrt(D)
    constexpr float inv_sqrt_d = 1.0f / 16.0f;  // placeholder — replaced per D below
    // Use a constexpr-correct value:
    //   sqrt(16) = 4, sqrt(32) = 4√2, etc. — compute at compile time.
    // For correctness we just use rsqrt:
    const float scale = rsqrtf((float) D);
    row_ptr[tid] = smem[tid] * scale;
}


// ─────────────────────────────────────────────────────────────────────────
// Host wrapper. Dispatches on head_dim to the right template instantiation.
// data is (n_rows, head_dim) fp32, row-major.
// Applies FWHT in-place along the last dim and normalizes by 1/sqrt(head_dim).
//
// row_stride is in fp32 elements (typically == head_dim for tightly packed rows).
// ─────────────────────────────────────────────────────────────────────────
static inline hipError_t mt_turbo_fp8_fwht(
    hipStream_t stream,
    float *     data_device,
    int         n_rows,
    int         head_dim,
    int         row_stride
) {
    if (n_rows <= 0) return hipSuccess;
    dim3 grid(n_rows, 1, 1);

    switch (head_dim) {
        case 16:
            mt_turbo_fp8_fwht_kernel<16>  <<<grid, dim3(16),  0, stream>>>(data_device, n_rows, row_stride);
            break;
        case 32:
            mt_turbo_fp8_fwht_kernel<32>  <<<grid, dim3(32),  0, stream>>>(data_device, n_rows, row_stride);
            break;
        case 64:
            mt_turbo_fp8_fwht_kernel<64>  <<<grid, dim3(64),  0, stream>>>(data_device, n_rows, row_stride);
            break;
        case 128:
            mt_turbo_fp8_fwht_kernel<128> <<<grid, dim3(128), 0, stream>>>(data_device, n_rows, row_stride);
            break;
        case 256:
            mt_turbo_fp8_fwht_kernel<256> <<<grid, dim3(256), 0, stream>>>(data_device, n_rows, row_stride);
            break;
        case 512:
            mt_turbo_fp8_fwht_kernel<512> <<<grid, dim3(512), 0, stream>>>(data_device, n_rows, row_stride);
            break;
        case 1024:
            mt_turbo_fp8_fwht_kernel<1024><<<grid, dim3(1024),0, stream>>>(data_device, n_rows, row_stride);
            break;
        default:
            return hipErrorInvalidValue;
    }
    return hipGetLastError();
}


// ─────────────────────────────────────────────────────────────────────────
// Scalar reference (CPU-side, fp32). Used by tests/test_turbo_fp8_hadamard.cu
// to validate the GPU kernel.
// ─────────────────────────────────────────────────────────────────────────
static inline void mt_turbo_fp8_fwht_reference_cpu(
    float * data,
    int     n_rows,
    int     head_dim,
    int     row_stride
) {
    for (int r = 0; r < n_rows; ++r) {
        float * row = data + r * row_stride;
        // In-place FWHT butterfly
        for (int stride = 1; stride < head_dim; stride *= 2) {
            for (int i = 0; i < head_dim; i += 2 * stride) {
                for (int j = 0; j < stride; ++j) {
                    float a = row[i + j];
                    float b = row[i + j + stride];
                    row[i + j]          = a + b;
                    row[i + j + stride] = a - b;
                }
            }
        }
        // Normalize by 1/sqrt(head_dim)
        const float scale = 1.0f / std::sqrt((float) head_dim);
        for (int i = 0; i < head_dim; ++i) row[i] *= scale;
    }
}
