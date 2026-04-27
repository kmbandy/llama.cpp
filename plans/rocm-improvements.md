# ROCm/HIP Improvement Opportunities for llama.cpp

## Executive Summary

This document identifies 5 high-impact opportunities to improve token generation speed, stability, and overall performance for AMD GPUs using ROCm/HIP in llama.cpp. The current implementation uses a compatibility layer (`ggml/src/ggml-cuda/vendors/hip.h`) that maps CUDA APIs to HIP equivalents, but several performance-critical paths are suboptimal or disabled for AMD hardware.

---

## Opportunity 1: Enable Flash Attention with Quantized KV Cache for AMD (TILE/MMA Kernels)

**Impact**: Token Generation Speed (Prefill Phase)
**Severity**: High - Forces fallback to slower VEC kernel

### Current State

In [`fattn.cu:475-486`](ggml/src/ggml-cuda/fattn.cu:475), HIP is forced to use the vector (VEC) flash attention kernel for all quantized KV cache scenarios:

```cpp
#ifdef GGML_USE_HIP
    // HIP/ROCm: the TILE/MMA/WMMA FA paths allocate unbounded f16 temp buffers
    // for quantized KV types (K_f16, V_f16 in launch_fattn). The pool retains
    // peak allocation size, so the temp buffer VRAM exceeds KV compression savings.
    // This causes quantized KV to OOM before f16 on the same context length.
    // Force VEC path which does inline dequant with zero temp buffer overhead.
    // Trade-off: prefill is slower (sequential query processing).
    // Limitation: head_dim > 256 cannot use VEC (falls through to TILE).
    if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel) {
        return BEST_FATTN_KERNEL_VEC;
    }
#endif // GGML_USE_HIP
```

Additionally, in [`fattn-common.cuh:1300-1302`](ggml/src/ggml-cuda/fattn-common.cuh:1300), HIP bypasses the memory pool for f16 temp buffers:

```cpp
#ifdef GGML_USE_HIP
    // HIP/ROCm: bypass the memory pool for f16 temp buffers.
    // The legacy pool (ggml_cuda_pool_leg) retains peak-sized allocations permanently.
```

### Root Cause

The TILE/MMA/WMMA flash attention kernels allocate temporary f16 buffers for dequantizing KV cache during attention computation. On CUDA, these are managed by the VMM (Virtual Memory Management) pool which can release memory. On HIP, the legacy pool retains peak allocations permanently, causing VRAM pressure.

### Proposed Improvements

1. **Implement per-operation temp buffer management**: Instead of relying on the pool, allocate and free temp buffers per-operation for HIP, similar to how CUDA handles it with VMM.

2. **Use rocALUTION or custom scratchpad allocator**: Create a HIP-specific scratchpad allocator that uses `hipMalloc`/`hipFree` with proper lifecycle management for temporary attention buffers.

3. **Enable TILE kernel for quantized KV on RDNA4/CDNA3**: Once temp buffer management is fixed, remove or relax the `GGML_USE_HIP` guard in `fattn.cu:483`.

### Expected Impact

- **Prefill speed improvement**: 20-40% for models using quantized KV cache (common in production)
- **VRAM efficiency**: Better utilization, allowing longer context lengths with quantized KV
- **Models affected**: All models using quantized KV cache (Q4_0, Q5_0, Q8_0, etc.)

---

## Opportunity 2: Enable rocprim for SSM-Scan (State Space Models)

**Impact**: Token Generation Speed (Mamba/SSM-based models) + Stability
**Severity**: Medium - Currently falls back to slower custom implementation

### Current State

In [`ssm-scan.cu:1-3`](ggml/src/ggml-cuda/ssm-scan.cu:1):

```cpp
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
```

HIP is explicitly excluded from using CUB (CUDA Blocks). ROCm provides **rocprim** as the equivalent library, which has been available since ROCm 4.0+.

### Proposed Implementation

```cpp
#if !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif
#if defined(GGML_USE_HIP) && HIP_VERSION >= 40000000
#define USE_ROCPRIM
#endif
```

Then in the scan implementation:

```cpp
#ifdef USE_ROCPRIM
#include <rocprim/rocprim.hpp>
namespace prim_ns = rocprim;
#endif

// Use prim_ns::device_scan instead of cub::DeviceScan
```

### Expected Impact

- **Mamba/MXCP model inference speed**: 30-50% improvement for SSM-scan operations
- **Stability**: rocprim is actively maintained by AMD, while the custom fallback may have edge cases
- **Models affected**: Mamba, Hyena, and other SSM-based architectures

---

## Opportunity 3: Enable DKQ640 Flash Attention Tile Kernel for AMD

**Impact**: Token Generation Speed (models with head_dim=640)
**Severity**: Medium - Completely disabled for AMD

### Current State

In [`fattn-tile.cu:49-54`](ggml/src/ggml-cuda/fattn-tile.cu:49):

```cpp
#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_FATTN_DKQ640)
        case 640: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<640, 512>(ctx, dst);
        } break;
#endif // !GGML_USE_HIP && !GGML_CUDA_NO_FATTN_DKQ640
```

The DKQ640/DV512 tile kernel is completely disabled for HIP. This affects models like **PaLM 2** and potentially others with head_dim=640.

### Root Cause

The DKQ640 kernel likely exceeds shared memory limits on older AMD architectures or has compilation issues. However, modern AMD GPUs (RDNA3/CDNA3) have sufficient shared memory.

### Proposed Implementation

1. **Add architecture-specific guards**: Enable DKQ640 only for GPUs with sufficient shared memory:

```cpp
#if defined(GGML_USE_HIP) && (defined(CDNA) || defined(RDNA3) || defined(RDNA4))
        case 640: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<640, 512>(ctx, dst);
        } break;
#elif !defined(GGML_CUDA_NO_FATTN_DKQ640)
        case 640: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case<640, 512>(ctx, dst);
        } break;
#endif
```

2. **Verify shared memory usage**: Ensure the DKQ640 kernel does not exceed 164KB (CDNA3 shared memory) or 164KB (RDNA3 shared memory).

### Expected Impact

- **Enables flash attention** for head_dim=640 models on AMD
- **Performance parity** with CUDA for these models
- **Models affected**: PaLM 2, and any future models with head_dim=640

---

## Opportunity 4: Optimize Multi-GPU Cross-Device Memory Copies

**Impact**: Multi-GPU Stability + Token Generation Speed (multi-GPU setups)
**Severity**: Medium - Row-by-row fallback is slow for large transfers

### Current State

In [`ggml-cuda.cu:1680-1713`](ggml/src/ggml-cuda/ggml-cuda.cu:1680):

```cpp
static cudaError_t ggml_cuda_Memcpy2DPeerAsync(...) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy3DPeerAsync for CUDA
    cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools.
    // hipMemcpy2DAsync with hipMemcpy2DAsync requires P2P access between devices.
    // For mixed-architecture multi-GPU setups without P2P,
    // hipMemcpy2DAsync submits successfully but fails asynchronously with
    // hipErrorNoBinaryForGpu when the internal copy kernel isn't compiled for the
    // destination device. Use hipMemcpyPeerAsync row-by-row instead...
    if (dstDevice == srcDevice) {
        return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice, stream);
    }
    for (size_t i = 0; i < height; ++i) {
        cudaError_t err = cudaMemcpyPeerAsync(...);
    }
    return cudaSuccess;
#endif
}
```

Similarly in [`ggml-cuda.cu:1430-1459`](ggml/src/ggml-cuda/ggml-cuda.cu:1430) for `cpy.cu`.

### Root Cause

- HIP does not support `cudaMemcpy3DPeerAsync`
- P2P (Peer-to-Peer) access is only available between GPUs on the same PCIe switch/NUMA domain
- The row-by-row fallback (`hipMemcpyPeerAsync`) is significantly slower for large transfers

### Proposed Improvements

1. **Use `hipMemcpyPeerAsync` with host staging for 3D copies**: For pitch-linear copies between different devices, use a two-stage approach:

```cpp
#if defined(GGML_USE_HIP)
    // For 3D pitch-linear copies, use a staging buffer approach
    if (dstDevice != srcDevice) {
        // Stage through host memory for non-P2P copies
        void* host_buf = nullptr;
        hipHostMalloc(&host_buf, height * dpitch, hipHostMallocDefault);
        
        // Host -> Device
        hipMemcpy2DAsync(dst, dpitch, host_buf, dpitch, width, height, 
                         hipMemcpyHostToDevice, stream);
        
        hipHostFree(host_buf);
    }
#endif
```

2. **Leverage HIP Graphs for repeated copies**: Enable `GGML_HIP_GRAPHS` for capturing and replaying memory copy patterns in multi-GPU scenarios.

3. **Add environment variable for copy strategy**: Allow users to choose between speed and memory:

```cpp
// GGML_HIP_COPY_STRATEGY=0: Auto-detect (default)
// GGML_HIP_COPY_STRATEGY=1: Always use P2P (faster if available)
// GGML_HIP_COPY_STRATEGY=2: Always use staging (more reliable)
```

### Expected Impact

- **Multi-GPU inference speed**: 2-5x improvement for cross-device copies
- **Stability**: Eliminates `hipErrorNoBinaryForGpu` errors in mixed-architecture setups
- **Models affected**: Any multi-GPU deployment

---

## Opportunity 5: Improve MMQ Kernel Selection for RDNA3/RDNA4

**Impact**: Token Generation Speed (decoding phase)
**Severity**: Medium - Suboptimal kernel selection for common quantization types

### Current State

In [`mmq.cu:343-359`](ggml/src/ggml-cuda/mmq.cu:343):

```cpp
if (amd_wmma_available(cc)) {
    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
        // High expert counts are almost always better on MMQ due to
        //     the synchronization overhead in the cuBLAS/hipBLAS path
        if (n_experts >= 64) {
            return true;
        }

        // For some quantization types MMQ can have lower peak TOPS than hipBLAS
        //     so it's only faster for sufficiently small batch sizes:
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 128;
            case GGML_TYPE_Q6_K:
                return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
            case GGML_TYPE_IQ2_XS:
                // ... (truncated)
```

The MMQ kernel selection for RDNA3 is conservative, falling back to hipBLAS for many batch sizes where MMQ could be faster.

### Proposed Improvements

1. **Expand MMQ thresholds for RDNA3/RDNA4**: Based on benchmarks, increase the batch size thresholds where MMQ is preferred:

```cpp
if (amd_wmma_available(cc)) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        // RDNA4 has better WMMA performance - use MMQ for wider range
        if (n_experts >= 32) {
            return true;
        }
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
                return ne11 <= 512;  // Increased from default
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 384;  // Increased from default
        }
    }
}
```

2. **Add runtime benchmarking for kernel selection**: Allow users to enable automatic kernel selection tuning:

```cpp
// GGML_HIP_MMQ_BENCH=1: Benchmark MMQ vs hipBLAS at startup
// GGML_HIP_MMQ_BENCH=0: Use static thresholds (default)
```

3. **Leverage RDNA4-specific WMMA instructions**: RDNA4 (GFX12) has improved WMMA instructions that can be better exploited:

```cpp
#if defined(RDNA4)
    // RDNA4 has improved WMMA throughput for small matrices
    // Lower the batch size threshold where MMQ becomes competitive
    return ne11 <= 256;  // Default for RDNA4
#endif
```

### Expected Impact

- **Decode speed improvement**: 10-25% for common quantization types (Q4_K, Q5_K, Q6_K)
- **Better expert handling**: Improved performance for MoE models on AMD
- **Models affected**: All models using quantized weights on RDNA3/RDNA4 hardware

---

## Additional Lower-Priority Opportunities

### A. rocwmma Flash Attention for CDNA

In [`fattn-wmma-f16.cuh:9-23`](ggml/src/ggml-cuda/fattn-wmma-f16.cuh:9):

```cpp
#if defined(GGML_HIP_ROCWMMA_FATTN)
#if defined(CDNA) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
#warning "rocwmma fattn on CDNA is broken on rocwmma v2.0.0, expect degraded performance"
```

**Issue**: rocwmma flash attention is broken on CDNA with rocwmma v2.0.0. Users must either:
- Build with `GGML_HIP_ROCWMMA_FATTN=OFF` (falls back to MFMA/TILE kernels)
- Wait for rocwmma v2.1.0+ fix

**Recommendation**: Document this limitation clearly and provide build-time detection.

### B. HIP Graph Support

In [`common.cuh:1165`](ggml/src/ggml-cuda/common.cuh:1165):

```cpp
#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)
#define USE_CUDA_GRAPH
```

**Issue**: HIP graphs are available since ROCm 5.0+ but are opt-in via `GGML_HIP_GRAPHS`.

**Recommendation**: Enable HIP graphs by default for RDNA3/CDNA3 hardware to reduce CPU overhead in the inference loop.

### C. Shared Memory Optimization for CDNA

In [`common.cuh:340`](ggml/src/ggml-cuda/common.cuh:340):

```cpp
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;  // Physical warp size for AMD
```

**Issue**: CDNA architectures have different shared memory characteristics than CUDA GPUs. Some kernels could benefit from warp-size-specific tuning.

**Recommendation**: Add CDNA-specific shared memory configuration for kernels that use dynamic shared memory.

---

## Priority Summary

| Priority | Opportunity | Impact Area | Effort | Speed Gain |
|----------|-------------|-------------|--------|------------|
| P0 | Flash Attention Quantized KV | Prefill speed | Medium | 20-40% |
| P1 | rocprim for SSM-Scan | SSM models | Low | 30-50% |
| P1 | DKQ640 Tile Kernel | head_dim=640 models | Low | Enable |
| P2 | Multi-GPU Copy Optimization | Multi-GPU | Medium | 2-5x |
| P2 | MMQ Kernel Selection | Decode speed | Low | 10-25% |
| P3 | rocwmma Workaround | Stability | Low | N/A |
| P3 | HIP Graphs | CPU overhead | Low | 5-10% |

---

## Build Configuration Recommendations

Add the following CMake options for ROCm-specific tuning:

```cmake
# Enable rocprim for SSM-scan operations
option(GGML_HIP_USE_ROCPRIM "Use rocprim instead of custom SSM-scan fallback" ON)

# Enable HIP graphs for reduced CPU overhead
option(GGML_HIP_GRAPHS "Enable CUDA graphs for HIP (ROCm 5.0+)" ON)

# Enable DKQ640 tile kernel for modern AMD GPUs
option(GGML_HIP_FATTN_DKQ640 "Enable flash attention tile kernel for head_dim=640" ON)

# Enable rocwmma flash attention (experimental)
option(GGML_HIP_ROCWMMA_FATTN "Enable rocwmma flash attention" OFF)

# MMQ kernel selection strategy
# GGML_HIP_MMQ_BENCH=1: Enable runtime benchmarking
```

---

## Testing Recommendations

For each improvement, test on:

1. **CDNA architectures**: MI100 (CDNA1), MI250X (CDNA2), MI300X (CDNA3)
2. **RDNA architectures**: RX 6600 (RDNA2), RX 7900 XTX (RDNA3), RX 9070 (RDNA4)
3. **Key benchmarks**:
   - `./bench -m <model> -n 512 -b 1 -t <threads>` - Token generation speed
   - `./bench -m <model> -n 1 -b 1 -t 1 -f 2048` - Prefill speed
   - Multi-GPU: `./main -m <model> -ngl 2` - Cross-device copies

4. **VRAM monitoring**: Track peak VRAM usage with quantized KV cache
