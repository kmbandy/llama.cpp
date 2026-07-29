#pragma once

// Private HIP/CUDA runtime compatibility layer for the weight pager.
// Keep pager implementation sources written against the original HIP names so
// their HIP preprocessed output remains unchanged.

// Compile-time: select the runtime header and type namespace present in this build.
#if defined(GGML_USE_HIP)

#include <hip/hip_runtime.h>

// Compile-time: CUDA compatibility aliases require the CUDA runtime header.
#elif defined(GGML_USE_CUDA)

#include <cuda_runtime.h>

#define hipDeviceAttributeIntegrated cudaDevAttrIntegrated
#define hipDeviceGetAttribute         cudaDeviceGetAttribute
#define hipDeviceProp_t               cudaDeviceProp
#define hipDeviceSynchronize          cudaDeviceSynchronize
#define hipDevice_t                   int
#define hipErrorInvalidDevice         cudaErrorInvalidDevice
#define hipErrorInvalidValue          cudaErrorInvalidValue
#define hipErrorNotReady              cudaErrorNotReady
#define hipError_t                    cudaError_t
#define hipEventCreate                cudaEventCreate
#define hipEventCreateWithFlags       cudaEventCreateWithFlags
#define hipEventDestroy               cudaEventDestroy
#define hipEventDisableTiming         cudaEventDisableTiming
#define hipEventElapsedTime           cudaEventElapsedTime
#define hipEventQuery                 cudaEventQuery
#define hipEventRecord                cudaEventRecord
#define hipEventSynchronize           cudaEventSynchronize
#define hipEvent_t                    cudaEvent_t
#define hipGetDevice                  cudaGetDevice
#define hipGetDeviceProperties        cudaGetDeviceProperties
#define hipGetErrorString             cudaGetErrorString
#define hipHostFree                   cudaFreeHost
#define hipHostMalloc                 cudaHostAlloc
#define hipHostMallocDefault          cudaHostAllocDefault
#define hipMalloc                     cudaMalloc
#define hipMemcpy                     cudaMemcpy
#define hipMemcpyAsync                cudaMemcpyAsync
#define hipMemcpyDeviceToHost         cudaMemcpyDeviceToHost
#define hipMemcpyHostToDevice         cudaMemcpyHostToDevice
#define hipMemset                     cudaMemset
#define hipMemsetAsync                cudaMemsetAsync
#define hipSetDevice                  cudaSetDevice
#define hipStreamCreateWithFlags      cudaStreamCreateWithFlags
#define hipStreamDestroy              cudaStreamDestroy
// NOTE: hipStreamGetDevice is deliberately NOT mapped. CUDA gained
// cudaStreamGetDevice only in 12.8 (this box builds against 12.0), and the
// obvious substitute -- cudaGetDevice -- answers a DIFFERENT question: the
// currently-active device rather than the stream's. At the one call site
// (wp-eval-cb.cpp) the current device has already been set to target_device by
// ScopedHipDevice, so such a substitution would make the guard tautologically
// true and silently inert. A safety check that can never fire is worse than an
// absent one, so the call site skips the check explicitly on CUDA instead.
#define hipStreamNonBlocking          cudaStreamNonBlocking
#define hipStreamPerThread            cudaStreamPerThread
#define hipStreamSynchronize          cudaStreamSynchronize
#define hipStreamWaitEvent            cudaStreamWaitEvent
#define hipStream_t                   cudaStream_t
#define hipSuccess                    cudaSuccess

#endif
