#pragma once

// Private HIP/CUDA runtime compatibility layer for the weight pager.
// Keep pager implementation sources written against the original HIP names so
// their HIP preprocessed output remains unchanged.

#if defined(GGML_USE_HIP)

#include <hip/hip_runtime.h>

#elif defined(GGML_USE_CUDA)

#include <cuda_runtime.h>

static inline cudaError_t wp_cuda_stream_get_device(cudaStream_t /*stream*/, int * device) {
    return cudaGetDevice(device);
}

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
#define hipStreamGetDevice            wp_cuda_stream_get_device
#define hipStreamNonBlocking          cudaStreamNonBlocking
#define hipStreamPerThread            cudaStreamPerThread
#define hipStreamSynchronize          cudaStreamSynchronize
#define hipStreamWaitEvent            cudaStreamWaitEvent
#define hipStream_t                   cudaStream_t
#define hipSuccess                    cudaSuccess

#endif
