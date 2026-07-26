#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

// Weight-paging bridge. The destination pointer is a Vulkan buffer offset
// biased by the backend's fixed pointer base; it must not be dereferenced.
GGML_BACKEND_API bool ggml_backend_vk_wp_stage_in(ggml_backend_buffer_t buffer,
                                                   void * dst, const void * src,
                                                   size_t payload_size, size_t slot_size,
                                                   void ** event);
GGML_BACKEND_API bool ggml_backend_vk_wp_event_query(void * event);
GGML_BACKEND_API bool ggml_backend_vk_wp_event_wait(void * event);
GGML_BACKEND_API void ggml_backend_vk_wp_event_free(void * event);

// Host staging memory registered with the same Vulkan device as `pool_buffer`.
// This matters for more than speed: a stage_in whose source is registered here
// is copied straight from it, while an unregistered source has to go through the
// single device-wide staging buffer, which forces the transfer to be fenced
// before stage_in returns (two overlapping unpinned transfers would clobber that
// shared region). So allocating the pager's bounce arena here is what allows
// transfers to actually stay in flight.
//
// Returns memory suitable for O_DIRECT only if the returned pointer is
// sufficiently aligned — the caller must check, since the alignment comes from
// vkMapMemory and is not guaranteed to be a filesystem block size. Returns null
// if the buffer is not a Vulkan buffer or the allocation fails.
GGML_BACKEND_API void * ggml_backend_vk_wp_host_alloc(ggml_backend_buffer_t pool_buffer, size_t size);
GGML_BACKEND_API void   ggml_backend_vk_wp_host_free(ggml_backend_buffer_t pool_buffer, void * ptr);

// Device-to-host read out of the pool, for the RAM victim tier: an evicted
// slot's bytes are copied to host RAM so a later reference is served from RAM
// instead of re-read from NVMe. `src` is a pool "pointer" (the backend's fixed
// base plus an offset) and must not be dereferenced. Synchronous.
GGML_BACKEND_API bool ggml_backend_vk_wp_read(ggml_backend_buffer_t pool_buffer,
                                               const void * src, void * dst, size_t n);

// Weight-paging consumption bridge — the counterpart of the CUDA backend's
// ggml_cuda_set_routed_expert_ptrs. Publishes, for the NEXT mul_mat_id node
// only, where each active expert's weights actually live: `pool_buffer` is the
// pager's slot pool and `block_offsets[i]` is the offset of the expert selected
// at active slot i, expressed in QUANT BLOCKS (not bytes) because the shaders
// index the weight buffer as an array of quant blocks.
//
// Vulkan needs offsets rather than CUDA's raw pointers because every slot lives
// in one VkBuffer. The value is consumed (and cleared) by the next mul_mat_id
// dispatch, so a stale publication cannot leak into an unrelated node.
// `block_offsets` is indexed by EXPERT ID and must cover every expert of the
// node, not just the active ones: one publication serves the whole batch, and
// different tokens route to different experts.
GGML_BACKEND_API void ggml_backend_vk_wp_set_expert_offsets(ggml_backend_buffer_t pool_buffer,
                                                            const uint32_t * block_offsets,
                                                            int n_experts);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
