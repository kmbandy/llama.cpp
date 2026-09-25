#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Fork-local declarations below must sit inside the extern "C" block: their
// call sites bind them as weak extern "C" symbols (unmangled names).
#ifdef  __cplusplus
extern "C" {
#endif

// Fork-local weight-pager copy-stream hooks. These return false when the
// feature is unavailable or has permanently disarmed after a runtime error.
GGML_BACKEND_API bool ggml_backend_cuda_wp_copy_stream_enabled(ggml_backend_t backend);
GGML_BACKEND_API bool ggml_backend_cuda_wp_copy_tensor_async(ggml_backend_t backend, ggml_tensor * tensor,
                                                             const void * data, size_t offset, size_t size);
GGML_BACKEND_API bool ggml_backend_cuda_wp_copy_stream_record_event(ggml_backend_t backend,
                                                                    ggml_backend_event_t event);
// Live hipGraph/cudaGraph counters for this backend's CUDA device. False if
// backend is not CUDA/HIP. Used by the worker 5s banner; atexit is SIGKILL'd.
GGML_BACKEND_API bool ggml_backend_cuda_wp_graph_counts(
        ggml_backend_t backend,
        uint64_t * captures, uint64_t * replays, uint64_t * fallbacks,
        uint64_t * cap_newkey, uint64_t * cap_lru);
// PRINT-ONLY diagnostic: dumps the internal-AllReduce pipeline's host-side
// state (call_count, pool slot/token, spin-watchdog fields, etc.) to stderr.
// No-op if no internal-AllReduce pipeline was ever created in this process
// (e.g. non-TP runs, or TP running the meta/NCCL backend instead). Defined
// in ggml-cuda.cu, thin wrapper over allreduce.cu's ggml_cuda_ar_dump_state.
GGML_BACKEND_API void ggml_backend_cuda_ar_dump_state(const char * reason);

// Pack host f32 into the ml8-4 AllReduce wire layout on the current device.
// ne must be a multiple of 32. False if the device pack cannot run.
GGML_BACKEND_API bool ggml_cuda_expert_wire_pack_ml8_4(const float * src, void * dst, int64_t ne);
// Same pack, but src is already on the current device. Downloads only the
// 18-byte blocks. False if the pointer is not device memory.
GGML_BACKEND_API bool ggml_cuda_expert_wire_pack_ml8_4_device(const float * src, void * dst, int64_t ne);

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// HIP: flush deferred multi-input cross-device stages (no-op on CUDA/non-batch).
// Call before WP eval_cb / any consumer that must see staged activations.
GGML_BACKEND_API void ggml_backend_cuda_xdev_batch_flush(void);

// Wait only for work queued on the backend compute stream.
GGML_BACKEND_API bool ggml_backend_cuda_synchronize_compute(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
