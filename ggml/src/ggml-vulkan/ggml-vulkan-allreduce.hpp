#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// Vulkan AllReduce for tensor-parallel inference across two Vulkan devices.
//
// This file is #include'd near the end of ggml-vulkan.cpp -- it is NOT a
// standalone translation unit.  Everything in ggml-vulkan.cpp has internal
// linkage (vk_device_struct, ggml_backend_vk_context, ggml_vk_create_buffer,
// ggml_vk_create_pipeline_func, ...), and there is no shared internal header,
// so textual inclusion is the only way to reuse that machinery without a large
// refactor.  Include it after ggml_backend_is_vk() and before the reg iface.
//
// It implements the meta-backend comm contract:
//
//     void * ggml_backend_comm_init(ggml_backend_t * backends, size_t n)
//     void   ggml_backend_comm_free(void * comm_ctx)
//     bool   ggml_backend_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors)
//
// resolved by ggml-backend-meta.cpp through ggml_backend_reg_get_proc_address()
// on the reg of backends[0]'s device.  Returning nullptr from init, or false
// from allreduce_tensor, makes the meta backend fall back to its generic
// butterfly reduction.
//
// ---------------------------------------------------------------------------
// Data plane
//
// Two devices, in-place F32 sum, latency-bound (~128 reductions per decoded
// token, ~20 KB each at n_embd=5120).  Per rank we allocate one plain host
// allocation and import it into BOTH vk::Devices via VK_EXT_external_memory_host
// (already used by this backend for the weight pager, see
// ggml_vk_buffer_from_host_ptr).  Both devices then address the same physical
// pages through their own VkBuffer handle, so the accumulate shader reads the
// peer's contribution straight out of host memory over PCIe.
//
// Work recorded per reduction, per device (peer = 1 - i):
//
//   pack  : zero the shard if inactive (no GGML_TENSOR_FLAG_COMPUTE, so it
//           contributes 0 -- matching NCCL / the meta backend), copy or
//           convert it into send[i][slot], then release that range to
//           VK_QUEUE_FAMILY_EXTERNAL.
//   add   : acquire send[peer][slot] from VK_QUEUE_FAMILY_EXTERNAL and run
//           allreduce_add_{f32,bf16}: tensor[i] += unpack(send[peer][slot]).
//
// The pack of every rank must complete before the add of any rank starts.
//
// ---------------------------------------------------------------------------
// Control plane -- two transports, because the obvious one measured badly
//
// v1 used cross-device timeline semaphores exclusively, in two separate
// vkQueueSubmit calls per device.  Measured on an R9700 + 6900XT RADV pair it
// cost ~390 us per reduction and lost to the generic fallback (6.37 vs 9.28
// t/s decode).  Two independent causes, addressed separately:
//
//   * 4 vkQueueSubmit calls per reduction, plus a vkAllocateDescriptorSets
//     growth step per dispatch.  Fixed for both transports below.
//   * The cross-device semaphore hop itself.  On amdgpu a shared-syncobj
//     signal has to travel through the kernel's dma-fence machinery and
//     re-schedule the waiting job; that is the term that does not shrink by
//     submitting less.
//
// So the transport is selectable, and the default is the one the measurement
// argues for:
//
//   GGML_VK_ALLREDUCE=fence (default)
//     Host-mediated.  Both packs are submitted (one vkQueueSubmit each), the
//     host spin-waits both completion fences, then both adds are submitted.
//     4 submits, 2 host fence waits, zero cross-device syncobj hops.  The
//     generic fallback is also host-mediated and reaches 9.28 t/s on this
//     pair, which is direct evidence that a host round-trip here is cheap --
//     and this path does strictly less work than the fallback (one small
//     staged copy each way plus one add dispatch, versus full tensor copies
//     through the backend's copy machinery).
//
//   GGML_VK_ALLREDUCE=semaphore
//     Device-mediated.  Pack and add are recorded as two submissions inside
//     ONE vk_context, so ggml_vk_submit issues them as a single vkQueueSubmit
//     carrying two VkSubmitInfos: 2 submits total per reduction and no host
//     stall at all, at the price of one cross-device syncobj hop.  Because the
//     add submission is already queued and waiting when the peer's pack
//     signals, the kernel only has to satisfy a fence rather than ingest a
//     fresh submission -- the part of v1's cost that was self-inflicted.
//
// Timeline values are shared by both transports: pack signals 2n, add signals
// 2n+1, for reduction number n.  In fence mode nothing waits on them
// cross-device, but they are still the cheapest way to answer "has the work
// that used descriptor set / staging slot X retired?" from the host.
//
// Numerics: the BF16 wire path rounds the local value through BF16 before
// adding the peer's already-rounded value, so both devices produce
// bit-identical sums.  Accumulation is F32 throughout.
// ---------------------------------------------------------------------------

// Cross-device timeline semaphores need an external handle type.  Only the
// POSIX fd flavour is wired up; on other platforms comm_init declines and the
// meta backend uses its generic path.
#if defined(__linux__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#   define GGML_VK_AR_EXTERNAL_SEMAPHORE_FD 1
#endif

// Staging slots per rank.  Slot reuse is gated on the peer's add from one lap
// earlier having retired; four laps of slack makes that gate a formality
// rather than a stall.  In semaphore mode that gate sits on the same
// submission as the current subgraph's graph work, so keeping it trivially
// satisfiable matters.
static constexpr size_t GGML_VK_AR_SLOTS = 4;

// Descriptor sets pre-allocated per rank at init, consumed as a ring.  Sized
// well above the number of reductions the host can record ahead of the GPU
// within one token (~128 x 2), so the wrap-around wait below is effectively
// never taken.  A set is only rewritten once the reduction that bound it has
// retired -- Vulkan forbids updating a set referenced by a pending command
// buffer.
static constexpr uint32_t GGML_VK_AR_DESC_RING = 512;

// Default bytes per staging slot (per rank).  Covers a 2 MB F32 prefill
// reduction (ubatch 128 x 5120 x 4B) in one chunk; larger ones are chunked.
// Override with GGML_VK_AR_SLOT_BYTES.
static constexpr size_t GGML_VK_AR_SLOT_BYTES_DEFAULT = 4u << 20; // 4 MB

// Reductions of at least this many bytes use the BF16 wire.  Below it the
// transfer is latency-bound rather than bandwidth-bound, so halving the bytes
// buys nothing and costs precision.  0 disables BF16 entirely.
// Override with GGML_VK_AR_BF16_THRESHOLD.
static constexpr size_t GGML_VK_AR_BF16_THRESHOLD_DEFAULT = 256u << 10; // 256 KB

enum ggml_vk_ar_mode {
    GGML_VK_AR_MODE_FENCE,      // host-mediated (default)
    GGML_VK_AR_MODE_SEMAPHORE,  // cross-device timeline semaphores
};

struct vk_op_allreduce_push_constants {
    uint32_t ne;     // F32 elements covered by this dispatch
    uint32_t off_a;  // element offset into binding 0's view
    uint32_t off_d;  // element offset into binding 1's view
};

// A tensor resolved to something the shaders can index: a storage-aligned
// buffer binding plus the leftover misalignment expressed in F32 elements.
struct ggml_vk_ar_view {
    vk_buffer buffer;
    size_t    offset      = 0;  // storage-aligned byte offset of the binding
    uint32_t  elem_offset = 0;  // F32 elements from `offset` to the tensor
};

struct ggml_vk_ar_rank {
    ggml_backend_t            backend = nullptr;
    ggml_backend_vk_context * ctx     = nullptr;
    vk_device                 device;

    // Host staging owned by this rank: GGML_VK_AR_SLOTS * slot_bytes.
    void * host_ptr   = nullptr;
    size_t host_bytes = 0;

    // The same host pages seen from each device.  send_local is written by
    // this rank; send_peer is this device's view of the other rank's staging.
    vk_buffer send_local;
    vk_buffer send_peer;

    // Timeline semaphore owned by this rank (exported) and this device's
    // imported handle onto the peer's timeline.  Both refer to a shared
    // payload, so values written by one device are observed by the other.
    vk::Semaphore sem_own;
    vk::Semaphore sem_peer;

    // Completion fence for the pack submission (fence mode only).
    vk::Fence fence_pack;

    // Private descriptor pool + ring, so the steady state never calls
    // vkAllocateDescriptorSets and never grows the backend's own pool.
    vk::DescriptorPool             desc_pool;
    std::vector<vk::DescriptorSet> desc_sets;
    // Reduction number that last bound desc_sets[k]; 0 means never used.
    std::vector<uint64_t>          desc_token;
    size_t                         desc_next = 0;

    vk_pipeline pipeline_pack_bf16;
    vk_pipeline pipeline_add_f32;
    vk_pipeline pipeline_add_bf16;
};

struct ggml_vk_ar_comm {
    ggml_vk_ar_rank  ranks[2];
    ggml_vk_ar_mode  mode           = GGML_VK_AR_MODE_FENCE;
    size_t           slot_bytes     = 0;
    size_t           bf16_threshold = 0;
    uint64_t         call_count     = 0;
};

static uint64_t ggml_vk_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

// ---------------------------------------------------------------------------
// Pipeline creation
//
// ggml_vk_create_pipeline is a lambda local to ggml_vk_load_shaders, and the
// lazy-compile path there only knows about pipelines in its own table.  These
// three are tiny, so create them eagerly against ggml_vk_create_pipeline_func.
// ---------------------------------------------------------------------------
static vk_pipeline ggml_vk_ar_create_pipeline(
        vk_device & device, const char * name, size_t spv_size, const void * spv_data,
        uint32_t parameter_count) {

    vk_pipeline pipeline = std::make_shared<vk_pipeline_struct>();
    pipeline->name               = name;
    pipeline->parameter_count    = parameter_count;
    pipeline->push_constant_size = sizeof(vk_op_allreduce_push_constants);
    pipeline->wg_denoms          = { 256, 1, 1 };
    pipeline->align              = 1;
    pipeline->initialized        = true;

    // No subgroup-size requirement and no full-subgroup requirement: the
    // shaders are subgroup-agnostic, which is what RADV wave64 parts need.
    ggml_vk_create_pipeline_func(device, pipeline, spv_size, spv_data, "main",
                                 parameter_count, pipeline->wg_denoms, {},
                                 /* disable_robustness      = */ false,
                                 /* require_full_subgroups  = */ false,
                                 /* required_subgroup_size  = */ 0);
    return pipeline;
}

static void ggml_vk_ar_destroy_pipeline(vk_device & device, vk_pipeline & pipeline) {
    if (!pipeline) {
        return;
    }
    ggml_vk_destroy_pipeline(device->device, pipeline);
    // Drop the last strong ref; device->all_pipelines holds only weak refs, so
    // the device teardown loop will skip this (now expired) entry.
    pipeline.reset();
}

// ---------------------------------------------------------------------------
// External timeline semaphores
// ---------------------------------------------------------------------------
#ifdef GGML_VK_AR_EXTERNAL_SEMAPHORE_FD

static bool ggml_vk_ar_external_semaphore_supported(vk_device & device) {
    if (!device->external_semaphore_fd) {
        return false;
    }

    vk::SemaphoreTypeCreateInfo type_info{ vk::SemaphoreType::eTimeline, 0 };
    vk::PhysicalDeviceExternalSemaphoreInfo info{};
    info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
    info.pNext      = &type_info;

    vk::ExternalSemaphoreProperties props;
    try {
        props = device->physical_device.getExternalSemaphoreProperties(info);
    } catch (const vk::SystemError & e) {
        GGML_LOG_DEBUG("%s: getExternalSemaphoreProperties failed (%s)\n", __func__, e.what());
        return false;
    }

    const auto required = vk::ExternalSemaphoreFeatureFlagBits::eExportable |
                          vk::ExternalSemaphoreFeatureFlagBits::eImportable;
    if ((props.externalSemaphoreFeatures & required) != required) {
        return false;
    }
    if (!(props.compatibleHandleTypes & vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd)) {
        return false;
    }
    return true;
}

// Create an exportable timeline semaphore on `owner`, then import its payload
// into a fresh timeline semaphore on `importer`.  Both handles then track the
// same counter.
static bool ggml_vk_ar_create_shared_timeline(
        vk_device & owner, vk_device & importer,
        vk::Semaphore & out_owner_sem, vk::Semaphore & out_importer_sem) {

    out_owner_sem    = VK_NULL_HANDLE;
    out_importer_sem = VK_NULL_HANDLE;

    try {
        vk::SemaphoreTypeCreateInfo   type_info{ vk::SemaphoreType::eTimeline, 0 };
        vk::ExportSemaphoreCreateInfo export_info{ vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
        export_info.pNext = &type_info;

        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&export_info);
        out_owner_sem = owner->device.createSemaphore(ci);

        vk::SemaphoreTypeCreateInfo import_type_info{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo import_ci{};
        import_ci.setPNext(&import_type_info);
        out_importer_sem = importer->device.createSemaphore(import_ci);

        // Permanent import: the fd is consumed by vkImportSemaphoreFdKHR and
        // must not be closed by us afterwards.
        vk::SemaphoreGetFdInfoKHR get_info{ out_owner_sem,
                                            vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
        const int fd = owner->device.getSemaphoreFdKHR(get_info);

        vk::ImportSemaphoreFdInfoKHR import_info{};
        import_info.semaphore  = out_importer_sem;
        import_info.flags      = vk::SemaphoreImportFlags{};
        import_info.handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
        import_info.fd         = fd;
        importer->device.importSemaphoreFdKHR(import_info);
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: AllReduce timeline semaphore sharing failed (%s)\n", e.what());
        if (out_owner_sem) {
            owner->device.destroySemaphore(out_owner_sem);
            out_owner_sem = VK_NULL_HANDLE;
        }
        if (out_importer_sem) {
            importer->device.destroySemaphore(out_importer_sem);
            out_importer_sem = VK_NULL_HANDLE;
        }
        return false;
    }

    return true;
}

#endif // GGML_VK_AR_EXTERNAL_SEMAPHORE_FD

// Block until this rank's own timeline has reached `value`.  Used only for
// resource recycling, where the value is normally long since signalled.
static void ggml_vk_ar_wait_own_timeline(ggml_vk_ar_rank & r, uint64_t value) {
    vk::SemaphoreWaitInfo wi{ vk::SemaphoreWaitFlags{}, r.sem_own, value };
    VK_CHECK(r.device->device.waitSemaphores(wi, UINT64_MAX), "AllReduce own timeline wait", r.device);
}

// Block until the peer's timeline has reached `value`, observed through this
// device's imported handle.
static void ggml_vk_ar_wait_peer_timeline(ggml_vk_ar_rank & r, uint64_t value) {
    vk::SemaphoreWaitInfo wi{ vk::SemaphoreWaitFlags{}, r.sem_peer, value };
    VK_CHECK(r.device->device.waitSemaphores(wi, UINT64_MAX), "AllReduce peer timeline wait", r.device);
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

static void ggml_vk_ar_free(ggml_vk_ar_comm * comm);

static ggml_vk_ar_comm * ggml_vk_ar_init(ggml_backend_t * backends, size_t n_backends, ggml_vk_ar_mode mode) {
#ifndef GGML_VK_AR_EXTERNAL_SEMAPHORE_FD
    GGML_UNUSED(backends);
    GGML_UNUSED(n_backends);
    GGML_UNUSED(mode);
    GGML_LOG_DEBUG("%s: Vulkan AllReduce needs POSIX external semaphore fds; falling back\n", __func__);
    return nullptr;
#else
    if (n_backends != 2) {
        GGML_LOG_DEBUG("%s: Vulkan AllReduce only supports 2 devices (got %zu); falling back\n",
                       __func__, n_backends);
        return nullptr;
    }

    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_vk(backends[i])) {
            return nullptr;
        }
    }

    auto * ctx0 = (ggml_backend_vk_context *) backends[0]->context;
    auto * ctx1 = (ggml_backend_vk_context *) backends[1]->context;
    if (ctx0->device == ctx1->device) {
        GGML_LOG_DEBUG("%s: both backends share one vk::Device; falling back\n", __func__);
        return nullptr;
    }

    ggml_backend_vk_context * ctxs[2] = { ctx0, ctx1 };

    for (size_t i = 0; i < 2; i++) {
        if (!ctxs[i]->device->external_memory_host) {
            GGML_LOG_WARN("ggml_vulkan: %s lacks VK_EXT_external_memory_host; "
                          "AllReduce falls back to the generic path\n", ctxs[i]->device->name.c_str());
            return nullptr;
        }
        if (!ggml_vk_ar_external_semaphore_supported(ctxs[i]->device)) {
            GGML_LOG_WARN("ggml_vulkan: %s cannot share timeline semaphores over opaque fds; "
                          "AllReduce falls back to the generic path\n", ctxs[i]->device->name.c_str());
            return nullptr;
        }
    }

    auto * comm = new ggml_vk_ar_comm{};
    comm->mode           = mode;
    comm->slot_bytes     = ggml_vk_ar_env_u64("GGML_VK_AR_SLOT_BYTES",     GGML_VK_AR_SLOT_BYTES_DEFAULT);
    comm->bf16_threshold = ggml_vk_ar_env_u64("GGML_VK_AR_BF16_THRESHOLD", GGML_VK_AR_BF16_THRESHOLD_DEFAULT);

    // The same host pages are imported into both devices, so the allocation
    // has to satisfy both devices' import alignment.  Round the slot up too,
    // so slot boundaries stay aligned.
    size_t alignment = 4096;
    for (size_t i = 0; i < 2; i++) {
        alignment = std::max<size_t>(alignment, ctxs[i]->device->min_imported_host_pointer_alignment);
    }
    comm->slot_bytes = ((comm->slot_bytes + alignment - 1) / alignment) * alignment;
    if (comm->slot_bytes == 0) {
        comm->slot_bytes = alignment;
    }
    const size_t host_bytes = comm->slot_bytes * GGML_VK_AR_SLOTS;

    for (size_t i = 0; i < 2; i++) {
        ggml_vk_ar_rank & r = comm->ranks[i];
        r.backend = backends[i];
        r.ctx     = ctxs[i];
        r.device  = ctxs[i]->device;

        r.host_bytes = host_bytes;
        if (posix_memalign(&r.host_ptr, alignment, host_bytes) != 0 || r.host_ptr == nullptr) {
            GGML_LOG_ERROR("ggml_vulkan: AllReduce staging allocation failed (%zu bytes)\n", host_bytes);
            r.host_ptr = nullptr;
            ggml_vk_ar_free(comm);
            return nullptr;
        }
        memset(r.host_ptr, 0, host_bytes);
    }

    // Import each rank's staging into both devices.
    for (size_t i = 0; i < 2; i++) {
        const size_t peer = 1 - i;
        ggml_vk_ar_rank & r = comm->ranks[i];

        r.send_local = ggml_vk_buffer_from_host_ptr(r.device, r.host_ptr, host_bytes);
        r.send_peer  = ggml_vk_buffer_from_host_ptr(r.device, comm->ranks[peer].host_ptr, host_bytes);

        if (!r.send_local || !r.send_local->buffer || !r.send_peer || !r.send_peer->buffer) {
            GGML_LOG_WARN("ggml_vulkan: AllReduce could not import host staging into %s; "
                          "falling back to the generic path\n", r.device->name.c_str());
            ggml_vk_ar_free(comm);
            return nullptr;
        }
    }

    // Shared timelines: rank i signals sem_own, rank peer observes the same
    // payload through its imported sem_peer.
    for (size_t i = 0; i < 2; i++) {
        const size_t peer = 1 - i;
        if (!ggml_vk_ar_create_shared_timeline(comm->ranks[i].device, comm->ranks[peer].device,
                                               comm->ranks[i].sem_own, comm->ranks[peer].sem_peer)) {
            ggml_vk_ar_free(comm);
            return nullptr;
        }
    }

    try {
        for (size_t i = 0; i < 2; i++) {
            ggml_vk_ar_rank & r = comm->ranks[i];

            r.fence_pack = r.device->device.createFence({});

            // Private descriptor pool: one ring of two-binding sets, allocated
            // once.  Keeps vkAllocateDescriptorSets and the backend's own
            // 50%-growth pool logic out of the per-reduction path entirely.
            vk::DescriptorPoolSize pool_size(vk::DescriptorType::eStorageBuffer,
                                             (uint32_t) MAX_PARAMETER_COUNT * GGML_VK_AR_DESC_RING);
            vk::DescriptorPoolCreateInfo pool_ci({}, GGML_VK_AR_DESC_RING, pool_size);
            r.desc_pool = r.device->device.createDescriptorPool(pool_ci);

            std::vector<vk::DescriptorSetLayout> layouts(GGML_VK_AR_DESC_RING, r.device->dsl);
            vk::DescriptorSetAllocateInfo set_ai(r.desc_pool, GGML_VK_AR_DESC_RING, layouts.data());
            r.desc_sets  = r.device->device.allocateDescriptorSets(set_ai);
            r.desc_token.assign(GGML_VK_AR_DESC_RING, 0);

            r.pipeline_pack_bf16 = ggml_vk_ar_create_pipeline(
                r.device, "allreduce_pack_bf16", allreduce_pack_bf16_len, allreduce_pack_bf16_data, 2);
            r.pipeline_add_f32 = ggml_vk_ar_create_pipeline(
                r.device, "allreduce_add_f32", allreduce_add_f32_len, allreduce_add_f32_data, 2);
            r.pipeline_add_bf16 = ggml_vk_ar_create_pipeline(
                r.device, "allreduce_add_bf16", allreduce_add_bf16_len, allreduce_add_bf16_data, 2);
        }
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: AllReduce resource creation failed (%s)\n", e.what());
        ggml_vk_ar_free(comm);
        return nullptr;
    }

    GGML_LOG_INFO("ggml_vulkan: AllReduce initialized for %s + %s "
                  "(transport=%s, %zu KB x %zu staging slots per device, BF16 wire >= %zu KB)\n",
                  comm->ranks[0].device->name.c_str(), comm->ranks[1].device->name.c_str(),
                  comm->mode == GGML_VK_AR_MODE_FENCE ? "fence" : "semaphore",
                  comm->slot_bytes >> 10, GGML_VK_AR_SLOTS, comm->bf16_threshold >> 10);

    return comm;
#endif // GGML_VK_AR_EXTERNAL_SEMAPHORE_FD
}

static void ggml_vk_ar_free(ggml_vk_ar_comm * comm) {
    if (comm == nullptr) {
        return;
    }

    // Drain both devices before tearing down anything they might still be
    // reading: the steady state deliberately leaves submissions in flight.
    for (size_t i = 0; i < 2; i++) {
        if (comm->ranks[i].device) {
            comm->ranks[i].device->device.waitIdle();
        }
    }

    for (size_t i = 0; i < 2; i++) {
        ggml_vk_ar_rank & r = comm->ranks[i];
        if (r.device) {
            ggml_vk_ar_destroy_pipeline(r.device, r.pipeline_pack_bf16);
            ggml_vk_ar_destroy_pipeline(r.device, r.pipeline_add_f32);
            ggml_vk_ar_destroy_pipeline(r.device, r.pipeline_add_bf16);

            if (r.desc_pool) {
                // Frees every set allocated from it.
                r.device->device.destroyDescriptorPool(r.desc_pool);
                r.desc_pool = VK_NULL_HANDLE;
            }
            r.desc_sets.clear();
            r.desc_token.clear();

            if (r.fence_pack) {
                r.device->device.destroyFence(r.fence_pack);
                r.fence_pack = VK_NULL_HANDLE;
            }
            if (r.sem_own) {
                r.device->device.destroySemaphore(r.sem_own);
                r.sem_own = VK_NULL_HANDLE;
            }
            if (r.sem_peer) {
                r.device->device.destroySemaphore(r.sem_peer);
                r.sem_peer = VK_NULL_HANDLE;
            }
        }
        // Buffers must go before the host pages they alias.
        ggml_vk_destroy_buffer(r.send_local);
        ggml_vk_destroy_buffer(r.send_peer);
    }

    for (size_t i = 0; i < 2; i++) {
        if (comm->ranks[i].host_ptr) {
            free(comm->ranks[i].host_ptr);
            comm->ranks[i].host_ptr = nullptr;
        }
    }

    delete comm;
}

// ---------------------------------------------------------------------------
// Recording helpers
// ---------------------------------------------------------------------------

// Resolve a tensor to a storage-aligned binding plus an F32 element offset.
// Fails if the tensor is not 4-byte aligned, which would make the element
// offset meaningless.
static bool ggml_vk_ar_tensor_view(
        const ggml_backend_vk_context * ctx, const ggml_tensor * tensor, ggml_vk_ar_view & out) {

    vk_buffer buffer = nullptr;
    size_t    offset = 0;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->data, buffer, offset);
    }
    if (!buffer) {
        auto * buf_ctx = (ggml_backend_vk_buffer_context *) tensor->buffer->context;
        buffer = buf_ctx->dev_buffer;
        offset = vk_tensor_offset(tensor) + tensor->view_offs;
    }
    if (buffer == nullptr || (offset & 3) != 0) {
        return false;
    }

    const size_t align       = ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const size_t aligned_off = align > 1 ? (offset & ~(align - 1)) : offset;

    out.buffer      = buffer;
    out.offset      = aligned_off;
    out.elem_offset = (uint32_t) ((offset - aligned_off) / sizeof(float));
    return true;
}

// Claim the next descriptor set from this rank's ring, waiting only if the
// reduction that last bound it has not retired yet.
static vk::DescriptorSet ggml_vk_ar_next_descriptor_set(ggml_vk_ar_rank & r, uint64_t token) {
    const size_t idx = r.desc_next;
    r.desc_next = (r.desc_next + 1) % r.desc_sets.size();

    const uint64_t prev = r.desc_token[idx];
    if (prev != 0) {
        // The add of reduction `prev` signals 2*prev+1.  Normally already past.
        ggml_vk_ar_wait_own_timeline(r, 2 * prev + 1);
    }
    r.desc_token[idx] = token;
    return r.desc_sets[idx];
}

// Bind and dispatch one of our pipelines against an explicitly supplied
// descriptor set.  Deliberately bypasses ggml_vk_dispatch_pipeline, which
// draws from the backend context's ever-growing set pool.
static void ggml_vk_ar_dispatch(
        ggml_vk_ar_rank & r, vk_context & subctx, vk_pipeline & pipeline, vk::DescriptorSet set,
        const vk::DescriptorBufferInfo & binding0, const vk::DescriptorBufferInfo & binding1,
        const vk_op_allreduce_push_constants & pc, uint32_t n_invocations) {

    const vk::DescriptorBufferInfo infos[2] = { binding0, binding1 };
    vk::WriteDescriptorSet write{ set, 0, 0, 2, vk::DescriptorType::eStorageBuffer, nullptr, infos };
    r.device->device.updateDescriptorSets({ write }, {});

    const uint32_t wg = CEIL_DIV(n_invocations, pipeline->wg_denoms[0]);
    GGML_ASSERT(wg <= r.device->properties.limits.maxComputeWorkGroupCount[0]);

    vk::CommandBuffer & cmd = subctx->s->buffer->buf;
    // Use the same helpers ggml_vk_dispatch_pipeline uses, so this resolves to
    // whichever pushConstants overload the vendored vulkan.hpp exposes.
    cmd.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0,
                      push_constant_size(pc), push_constant_data(pc));
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->layout, 0, { set }, {});
    cmd.dispatch(wg, 1, 1);
}

// Record: zero-if-inactive, pack the local shard into our staging slot, and
// release that range to the external queue family.
static void ggml_vk_ar_record_pack(
        ggml_vk_ar_comm * comm, size_t i, vk_context & subctx, ggml_tensor ** tensors,
        ggml_vk_ar_view & view, uint64_t token,
        uint32_t chunk_start, uint32_t chunk_ne, bool use_bf16,
        size_t slot_off, size_t wire_bytes) {

    ggml_vk_ar_rank &         r   = comm->ranks[i];
    ggml_backend_vk_context * ctx = r.ctx;

    // Orders against the graph work already submitted on this queue.
    ggml_vk_sync_buffers(ctx, subctx);

    // the async helpers take vk_buffer& (non-const); the view is const here,
    // so pass a local shared_ptr copy
    vk_buffer view_buf = view.buffer;

    if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ggml_vk_buffer_memset_async(subctx, view_buf,
                                    view.offset + (size_t) (view.elem_offset + chunk_start) * sizeof(float),
                                    0, (size_t) chunk_ne * sizeof(float));
        ggml_vk_sync_buffers(ctx, subctx);
    }

    if (use_bf16) {
        const vk_op_allreduce_push_constants pc = {
            /* .ne    = */ chunk_ne,
            /* .off_a = */ view.elem_offset + chunk_start,
            /* .off_d = */ (uint32_t) (slot_off / sizeof(uint32_t)),
        };
        ggml_vk_ar_dispatch(r, subctx, r.pipeline_pack_bf16,
                            ggml_vk_ar_next_descriptor_set(r, token),
                            ggml_vk_subbuffer(ctx, view.buffer, view.offset),
                            ggml_vk_subbuffer(ctx, r.send_local, 0),
                            pc, (chunk_ne + 1u) / 2u);
    } else {
        ggml_vk_buffer_copy_async(subctx, r.send_local, slot_off, view_buf,
                                  view.offset + (size_t) (view.elem_offset + chunk_start) * sizeof(float),
                                  wire_bytes);
    }

    // Release the slot to the external (peer-device) queue family.  This is
    // what makes our writes available outside this device; the peer pairs it
    // with the matching acquire before its add.
    vk::BufferMemoryBarrier release{};
    release.srcAccessMask       = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
    release.dstAccessMask       = vk::AccessFlags{};
    release.srcQueueFamilyIndex = subctx->p->q->queue_family_index;
    release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    release.buffer              = r.send_local->buffer;
    release.offset              = slot_off;
    release.size                = wire_bytes;

    subctx->s->buffer->buf.pipelineBarrier(
        subctx->p->q->stage_flags, vk::PipelineStageFlagBits::eBottomOfPipe,
        {}, {}, { release }, {});
}

// Record: acquire the peer's staging slot and accumulate it into our tensor.
static void ggml_vk_ar_record_add(
        ggml_vk_ar_comm * comm, size_t i, vk_context & subctx,
        ggml_vk_ar_view & view, uint64_t token,
        uint32_t chunk_start, uint32_t chunk_ne, bool use_bf16,
        size_t slot_off, size_t wire_bytes) {

    ggml_vk_ar_rank &         r   = comm->ranks[i];
    ggml_backend_vk_context * ctx = r.ctx;

    // Orders the accumulate against everything already submitted on this queue
    // -- in particular the pack's read of this same tensor.
    ggml_vk_sync_buffers(ctx, subctx);

    vk::BufferMemoryBarrier acquire{};
    acquire.srcAccessMask       = vk::AccessFlags{};
    acquire.dstAccessMask       = vk::AccessFlagBits::eShaderRead;
    acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    acquire.dstQueueFamilyIndex = subctx->p->q->queue_family_index;
    acquire.buffer              = r.send_peer->buffer;
    acquire.offset              = slot_off;
    acquire.size                = wire_bytes;

    subctx->s->buffer->buf.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, subctx->p->q->stage_flags,
        {}, {}, { acquire }, {});

    vk_pipeline & add = use_bf16 ? r.pipeline_add_bf16 : r.pipeline_add_f32;

    const vk_op_allreduce_push_constants pc = {
        /* .ne    = */ chunk_ne,
        /* .off_a = */ (uint32_t) (slot_off / sizeof(uint32_t)),
        /* .off_d = */ view.elem_offset + chunk_start,
    };
    ggml_vk_ar_dispatch(r, subctx, add,
                        ggml_vk_ar_next_descriptor_set(r, token),
                        ggml_vk_subbuffer(ctx, view.buffer, view.offset),
                        ggml_vk_subbuffer(ctx, r.send_peer, 0),
                        pc, use_bf16 ? (chunk_ne + 1u) / 2u : chunk_ne);

    // The reduced tensor is consumed by the next subgraph, which has no record
    // of this write in the backend's unsynced-node tracking.
    ggml_vk_sync_buffers(ctx, subctx);
}

// Close out a context and hand it to the queue.  Staging writes the context
// accumulated (e.g. a set_tensor_async that landed in this same compute
// context) have to be replayed before the command buffer that reads them runs.
static void ggml_vk_ar_end_and_submit(ggml_backend_vk_context * ctx, vk_context & subctx, vk::Fence fence) {
    ggml_vk_ctx_end(subctx);

    for (auto & cpy : subctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }
    subctx->in_memcpys.clear();
    for (auto & mset : subctx->memsets) {
        memset(mset.dst, mset.val, mset.n);
    }
    subctx->memsets.clear();

    ggml_vk_submit(subctx, fence);
    ctx->submit_pending = true;
    ctx->compute_ctx.reset();
}

// Wait for both packs to complete, then reset their fences.
//
// Spin on getFenceStatus rather than calling waitForFences: a 20 KB pack
// completes in microseconds, so a blocking wait's wake-up latency would be the
// dominant term -- the exact mistake that made the semaphore transport slow.
// Same reasoning (and same shape) as ggml_vk_wait_for_fence.  Both packs are
// already submitted when we get here, so waiting on them in order costs the
// max of the two, not the sum.
static void ggml_vk_ar_wait_pack_fences(ggml_vk_ar_comm * comm) {
    for (size_t i = 0; i < 2; i++) {
        ggml_vk_ar_rank & r = comm->ranks[i];

        for (;;) {
            vk::Result result;
            try {
                result = r.device->device.getFenceStatus(r.fence_pack);
            } catch (vk::DeviceLostError &) {
                ggml_vk_print_device_lost_info(r.device);
                GGML_LOG_ERROR("ggml_vulkan: AllReduce getFenceStatus at %s:%d\n", __FILE__, __LINE__);
                throw;
            }
            if (result == vk::Result::eSuccess) {
                break;
            }
            if (result != vk::Result::eNotReady) {
                GGML_LOG_ERROR("ggml_vulkan: AllReduce fence error %s at %s:%d\n",
                               to_string(result).c_str(), __FILE__, __LINE__);
                throw vk::SystemError(vk::make_error_code(result), "ggml_vulkan: AllReduce getFenceStatus");
            }
            for (uint32_t s = 0; s < 64; ++s) {
                YIELD();
            }
        }

        r.device->device.resetFences({ r.fence_pack });
    }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

// One chunk: a full pack/add cycle across both devices.
static void ggml_vk_ar_reduce_chunk(
        ggml_vk_ar_comm * comm, ggml_tensor ** tensors, ggml_vk_ar_view views[2],
        uint32_t chunk_start, uint32_t chunk_ne, bool use_bf16) {

    const uint64_t token    = ++comm->call_count;
    const size_t   slot     = (size_t) (token % GGML_VK_AR_SLOTS);
    const size_t   slot_off = slot * comm->slot_bytes;

    const uint32_t wire_units = use_bf16 ? (chunk_ne + 1u) / 2u : chunk_ne;
    const size_t   wire_bytes = (size_t) wire_units * sizeof(uint32_t);

    if (comm->mode == GGML_VK_AR_MODE_SEMAPHORE) {
        // One vkQueueSubmit per device, carrying two VkSubmitInfos: the pack
        // signals 2n, and the add -- already queued and waiting by the time
        // the peer's pack signals -- waits on the peer at 2n and signals 2n+1.
        for (size_t i = 0; i < 2; i++) {
            ggml_vk_ar_rank &         r   = comm->ranks[i];
            ggml_backend_vk_context * ctx = r.ctx;

            ggml_vk_submit_transfer_ctx(ctx);
            vk_context subctx = ggml_vk_get_compute_ctx(ctx);

            ggml_vk_ar_record_pack(comm, i, subctx, tensors, views[i], token,
                                   chunk_start, chunk_ne, use_bf16, slot_off, wire_bytes);

            // Don't overwrite a slot the peer might still be reading: gate on
            // the peer's add from one lap earlier.
            if (token > GGML_VK_AR_SLOTS) {
                subctx->s->wait_semaphores.push_back({ r.sem_peer, 2 * (token - GGML_VK_AR_SLOTS) + 1 });
            }
            subctx->s->signal_semaphores.push_back({ r.sem_own, 2 * token });

            // Second submission in the SAME context -> same vkQueueSubmit.
            ggml_vk_ctx_begin(r.device, subctx);

            ggml_vk_ar_record_add(comm, i, subctx, views[i], token,
                                  chunk_start, chunk_ne, use_bf16, slot_off, wire_bytes);

            subctx->s->wait_semaphores.push_back({ r.sem_peer, 2 * token });
            subctx->s->signal_semaphores.push_back({ r.sem_own, 2 * token + 1 });

            ggml_vk_ar_end_and_submit(ctx, subctx, {});
        }
        return;
    }

    // Fence transport.  No cross-device syncobj hop: the host observes both
    // packs completing and then releases both adds.
    for (size_t i = 0; i < 2; i++) {
        ggml_vk_ar_rank &         r   = comm->ranks[i];
        ggml_backend_vk_context * ctx = r.ctx;

        // Gate slot reuse on the peer's add from one lap earlier.  Long since
        // signalled in the steady state, so this is a status check, not a stall.
        if (token > GGML_VK_AR_SLOTS) {
            ggml_vk_ar_wait_peer_timeline(r, 2 * (token - GGML_VK_AR_SLOTS) + 1);
        }

        ggml_vk_submit_transfer_ctx(ctx);
        vk_context subctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_ar_record_pack(comm, i, subctx, tensors, views[i], token,
                               chunk_start, chunk_ne, use_bf16, slot_off, wire_bytes);

        subctx->s->signal_semaphores.push_back({ r.sem_own, 2 * token });

        // Both packs are in flight before either is waited on, so they overlap.
        ggml_vk_ar_end_and_submit(ctx, subctx, r.fence_pack);
    }

    ggml_vk_ar_wait_pack_fences(comm);

    for (size_t i = 0; i < 2; i++) {
        ggml_vk_ar_rank &         r   = comm->ranks[i];
        ggml_backend_vk_context * ctx = r.ctx;

        vk_context subctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_ar_record_add(comm, i, subctx, views[i], token,
                              chunk_start, chunk_ne, use_bf16, slot_off, wire_bytes);

        subctx->s->signal_semaphores.push_back({ r.sem_own, 2 * token + 1 });

        ggml_vk_ar_end_and_submit(ctx, subctx, {});
    }
}

static bool ggml_vk_ar_allreduce(ggml_vk_ar_comm * comm, ggml_tensor ** tensors) {
    GGML_ASSERT(comm != nullptr);

    if (tensors[0] == nullptr || tensors[1] == nullptr) {
        return false;
    }

    // build_in_out_ids can produce a zero-element tensor when n_outputs == 0.
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }

    // F32 only.  F16/BF16 tensors would need their own pack/add variants and
    // don't occur on the tensor-parallel reduction path today.
    if (tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (ne > (int64_t) std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    // Validate and resolve everything before issuing any work: a late `false`
    // would leave half a reduction enqueued for the fallback to trip over.
    ggml_vk_ar_view views[2];
    for (size_t i = 0; i < 2; i++) {
        if (tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne) {
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i]) || tensors[i]->buffer == nullptr) {
            return false;
        }
        if (!ggml_vk_ar_tensor_view(comm->ranks[i].ctx, tensors[i], views[i])) {
            return false;
        }
        if (views[i].buffer->device != comm->ranks[i].device) {
            return false;
        }
    }

    const bool use_bf16 = comm->bf16_threshold > 0 &&
                          (size_t) ne * sizeof(float) >= comm->bf16_threshold;

    // Elements that fit in one staging slot at the chosen wire width.  BF16
    // packs two elements per uint32; keep chunks even so a chunk boundary
    // never splits a pair.
    size_t max_chunk_ne = use_bf16 ? (comm->slot_bytes / sizeof(uint32_t)) * 2
                                   :  comm->slot_bytes / sizeof(float);
    if (use_bf16) {
        max_chunk_ne &= ~size_t{1};
    }
    if (max_chunk_ne == 0) {
        return false;
    }

    for (int64_t start = 0; start < ne; start += (int64_t) max_chunk_ne) {
        const uint32_t chunk_ne = (uint32_t) std::min<int64_t>((int64_t) max_chunk_ne, ne - start);
        ggml_vk_ar_reduce_chunk(comm, tensors, views, (uint32_t) start, chunk_ne, use_bf16);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Meta-backend comm contract
// ---------------------------------------------------------------------------

static void * ggml_backend_vk_comm_init(ggml_backend_t * backends, size_t n_backends) {
    // Default OFF: measured 2026-08-15 on the RADV R9700+6900XT pair, both
    // internal transports LOSE to the meta backend's generic fallback
    // (fence 3.53 t/s, semaphore 6.91 vs fallback 9.28, Q8 27B decode).
    // Keep the machinery as an opt-in experiment for other device pairs.
    const char * env = getenv("GGML_VK_ALLREDUCE");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "none") == 0) {
        return nullptr;
    }

    ggml_vk_ar_mode mode;
    if (strcmp(env, "fence") == 0) {
        mode = GGML_VK_AR_MODE_FENCE;
    } else if (strcmp(env, "semaphore") == 0) {
        mode = GGML_VK_AR_MODE_SEMAPHORE;
    } else {
        GGML_LOG_WARN("ggml_vulkan: unknown GGML_VK_ALLREDUCE value '%s' "
                      "(expected none|fence|semaphore); AllReduce disabled\n", env);
        return nullptr;
    }

    return ggml_vk_ar_init(backends, n_backends, mode);
}

static void ggml_backend_vk_comm_free(void * comm_ctx) {
    ggml_vk_ar_free(static_cast<ggml_vk_ar_comm *>(comm_ctx));
}

static bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
    if (comm_ctx == nullptr) {
        return false;
    }
    return ggml_vk_ar_allreduce(static_cast<ggml_vk_ar_comm *>(comm_ctx), tensors);
}
