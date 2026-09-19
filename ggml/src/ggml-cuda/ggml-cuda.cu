#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/wp-tp-trace.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/mt_pagedattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/ml8.cuh"
#include "ggml-ml8.h"  // FP8_B128 phase 2: GGML_FP8_QUANT_ROT_KIND_* constants
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/sinkhorn.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/turbo-wht.cuh"
#include "ggml-cuda/wp-node-trace.cuh"
#include "ggml-cuda/wp-op-profile.cuh"
#include "ggml-cuda/mmvq-tq.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// NOTE(fork): upstream early-returns here when the physical device already matches
// the current one ("faster on Windows"). That optimization is unsafe on ROCm
// multi-GPU with uninitialized thread contexts (ggml-org/llama.cpp#21140), so we
// keep upstream's virtual->physical translation but always call cudaSetDevice.
void ggml_cuda_set_device(int device) {
    const int physical_device = ggml_cuda_get_physical_device(device);
    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

// WP_ALLOC_LOG=1: log every device allocation (size, device, wall clock) to
// stderr so a VRAM->GTT eviction spike on a nearly-full card can be attributed
// to the allocation that triggered it. Off unless set; one getenv at first use.
static bool wp_alloc_log_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("WP_ALLOC_LOG");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}
static void wp_alloc_log(const char * what, int device, size_t size, size_t extra) {
    if (!wp_alloc_log_enabled()) {
        return;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv;
    localtime_r(&ts.tv_sec, &tmv);
    fprintf(stderr, "wp alloc-log %02d:%02d:%02d.%03ld %s device=%d size=%.1fMiB extra=%.1fMiB\n",
            tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ts.tv_nsec / 1000000, what, device,
            size / 1048576.0, extra / 1048576.0);
}

// MAD-LAB 2026-09-13 (VRAM->GTT eviction incident): amdgpu/KFD does not fail an
// over-budget device allocation -- it evicts the WHOLE process's resident VRAM
// into GTT (host RAM) to satisfy it. On a 16 GB host that took the box down twice
// (18.7 GB evicted in one second on the display GPU). So the runtime must never
// ask the driver for memory the device does not have: every device allocation
// after model load goes through ggml_cuda_device_malloc (pool misses, backend
// buffers, gallocr regrowth) or the VMM pool's cuMemCreate, and both now check
// live free VRAM against GGML_CUDA_VRAM_RESERVE_MB (default 512 MiB) first. A
// refused allocation fails the request (ggml_cuda_pool_oom below ->
// GGML_STATUS_ALLOC_FAILED), it does not abort and it never reaches the driver.
// Every post-load device allocation is also logged at INFO with its size and
// the free VRAM before it, so the server log attributes growth to a call site.
struct ggml_cuda_pool_oom : public std::runtime_error {
    int    device;
    size_t requested;
    size_t free_before;
    ggml_cuda_pool_oom(int device, size_t requested, size_t free_before)
        : std::runtime_error("ggml_cuda: device allocation refused (VRAM reserve)"),
          device(device), requested(requested), free_before(free_before) {}
};

static size_t ggml_cuda_vram_reserve_bytes() {
    static const size_t reserve = [] {
        // Default 0 = LOG ONLY, never refuse: behaviour identical to before this
        // change except that every post-load device allocation is attributed in
        // the log. Set GGML_CUDA_VRAM_RESERVE_MB>0 to turn on refusal.
        const char * e = getenv("GGML_CUDA_VRAM_RESERVE_MB");
        const long long mb = e ? atoll(e) : 0;
        return (size_t) (mb < 0 ? 0 : mb) * 1024ull * 1024ull;
    }();
    return reserve;
}

// returns true if `size` bytes may be allocated on `device` without dipping under
// the reserve; logs the allocation either way. `what` names the call site.
static bool ggml_cuda_vram_budget_check(const char * what, int device, size_t size, size_t * free_out) {
    ggml_cuda_set_device(device);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        if (free_out) { *free_out = 0; }
        return true; // cannot measure: keep the old behaviour rather than block loading
    }
    if (free_out) { *free_out = free_b; }
    const size_t reserve = ggml_cuda_vram_reserve_bytes();
    if (reserve > 0 && size + reserve > free_b) {
        GGML_LOG_ERROR(GGML_CUDA_NAME " vram-budget: REFUSING %s of %.1f MiB on device %d: free %.1f MiB of %.1f MiB, reserve %.1f MiB (GGML_CUDA_VRAM_RESERVE_MB) -- the driver would have evicted this process to GTT\n",
                       what, size / 1048576.0, device, free_b / 1048576.0, total_b / 1048576.0, reserve / 1048576.0);
        return false;
    }
    // raw stderr like the "wp hip-graphs" prints: common_log demotes GGML_LOG_INFO to trace (verbosity 4)
    // WP_VRAM_LOG=1 to enable; silent by default (fires on every allocation).
    if (ggml_cuda_wp_vram_log_enabled()) {
        fprintf(stderr, GGML_CUDA_NAME " vram-budget: %s %.1f MiB on device %d (free before %.1f MiB of %.1f MiB)\n",
                what, size / 1048576.0, device, free_b / 1048576.0, total_b / 1048576.0);
        fflush(stderr);
    }
    return true;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    wp_alloc_log("device_malloc", device, size, 0);
    if (!ggml_cuda_vram_budget_check("device_malloc", device, size, nullptr)) {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// MAD-LAB 2026-09-17: the HIP runtime's default hardware-queue pool is 4 per
// device and KFD reports num_cp_queues=4 for gfx12 (R9700, 9070 XT). A process
// that lands streams on all four queues takes the device's whole HQD budget
// and the HWS starts time-slicing them (wave save/restore on every switch):
// every kernel on that device pays a fixed latency, small kernels most. On
// qwen38-27b-q8-tp (-sm tensor, R9700 + 9070 XT) prefill @8k measured
// 660 t/s with 4 queues, 1225 with 3, 1318 with 2, 1034 with 1; decode was
// unchanged. The runtime reads GPU_MAX_HW_QUEUES once, lazily, on the first
// HIP API call, so this must run at library load: cap the pool at 2 when any
// GPU node's num_cp_queues is 4 or fewer. An explicit GPU_MAX_HW_QUEUES in
// the environment always wins (the same knob also fixes the RDNA2 wedge).
static int ggml_hip_kfd_min_cp_queues() {
    int min_q = -1;
    for (int node = 0; node < 64; ++node) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/properties", node);
        FILE * f = fopen(path, "r");
        if (f == nullptr) {
            break;
        }
        int  cp_queues = -1;
        long simd      = 0;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "num_cp_queues ", 14) == 0) {
                cp_queues = atoi(line + 14);
            } else if (strncmp(line, "simd_count ", 11) == 0) {
                simd = atol(line + 11);
            }
        }
        fclose(f);
        if (simd > 0 && cp_queues >= 0 && (min_q < 0 || cp_queues < min_q)) {
            min_q = cp_queues;
        }
    }
    return min_q;
}

static bool g_ggml_hip_hw_queues_capped = false;
static int  g_ggml_hip_kfd_cp_queues    = -1;

__attribute__((constructor)) static void ggml_hip_cap_hw_queues() {
    if (getenv("GPU_MAX_HW_QUEUES") != nullptr) {
        return;
    }
    g_ggml_hip_kfd_cp_queues = ggml_hip_kfd_min_cp_queues();
    if (g_ggml_hip_kfd_cp_queues > 0 && g_ggml_hip_kfd_cp_queues <= 4) {
        setenv("GPU_MAX_HW_QUEUES", "2", 0);
        g_ggml_hip_hw_queues_capped = true;
    }
}
#endif

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
#if defined(GGML_USE_HIP)
        info.devices[id].integrated = prop.integrated;
#else
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#endif
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

#if defined(GGML_USE_HIP)
    // MAD-LAB 2026-09-16: an RDNA2 card in a multi-GPU process stops draining
    // a stream while another of its HIP hardware queues holds a barrier on a
    // cross-device signal (the internal AllReduce), with every dependency of
    // the stalled stream satisfied -- the "16k wedge". GPU_MAX_HW_QUEUES=2
    // (read by the HIP runtime at init, so it must be in the environment
    // before the first HIP call) is what makes it drain; =8 hangs earlier.
    // This cannot be set from here (the runtime is already initialised), so
    // say so loudly instead.
    if (g_ggml_hip_hw_queues_capped) {
        GGML_LOG_INFO("%s: GPU_MAX_HW_QUEUES=2 (KFD num_cp_queues=%d; 4 queues time-slice, see ggml_hip_cap_hw_queues)\n",
                      __func__, g_ggml_hip_kfd_cp_queues);
    }
    if (info.device_count > 1) {
        bool has_rdna2 = false;
        for (int id = 0; id < info.device_count; ++id) {
            has_rdna2 = has_rdna2 || GGML_CUDA_CC_IS_RDNA2(info.devices[id].cc);
        }
        const char * hwq = getenv("GPU_MAX_HW_QUEUES");
        if (has_rdna2 && (hwq == nullptr || atoi(hwq) > 2)) {
            GGML_LOG_WARN("%s: RDNA2 device in a %d-GPU process without GPU_MAX_HW_QUEUES<=2 -- "
                          "tensor-parallel AllReduce can wedge the RDNA2 card (set GPU_MAX_HW_QUEUES=2 "
                          "in the process environment)\n", __func__, info.device_count);
        }
    }
#endif

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        // MAD-XXX diag (2026-09-10): PHYSICAL free of pooled memory. A HIP graph
        // captured earlier may still hold these pointers and replay against them
        // later -- cudaDeviceSynchronize() before this only drains IN-FLIGHT work,
        // not a FUTURE replay. This path fires only under memory pressure, so the
        // print costs nothing on the hot path and cannot mask a timing-sensitive
        // fault the way per-call instrumentation did.
        if (std::getenv("MAD_POOL_FREE_TRACE")) {
            int n = 0; size_t tot = 0;
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                if (buffer_pool[i].ptr) { ++n; tot += buffer_pool[i].size; }
            }
            std::fprintf(stderr, "[pool-free] dev=%d CLEAR_POOL freeing %d buffers, %.2f MiB (alloc failure)\n",
                         device, n, tot/1048576.0);
            for (int i = 0; i < MAX_BUFFERS; ++i) {
                if (buffer_pool[i].ptr) {
                    std::fprintf(stderr, "[pool-free] dev=%d   cudaFree %p +%zu\n",
                                 device, buffer_pool[i].ptr, buffer_pool[i].size);
                }
            }
        }
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        wp_alloc_log("pool_miss", device, look_ahead_size, pool_size);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        if (err == cudaErrorMemoryAllocation) {
            // MAD-LAB: refused by the VRAM budget (or a genuine driver OOM) even after
            // flushing the pool. Fail this graph compute, do not abort the process.
            (void)cudaGetLastError();
            size_t free_b = 0, total_b = 0;
            (void)cudaMemGetInfo(&free_b, &total_b);
            GGML_LOG_ERROR(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB refused after flush (pool cached %.2f MiB, device free %.2f MiB) -- failing the request\n",
                           device, look_ahead_size/1024.0/1024.0, pool_size/1024.0/1024.0, free_b/1024.0/1024.0);
            throw ggml_cuda_pool_oom(device, look_ahead_size, free_b);
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        // MAD-XXX diag: second physical-free path. Same hazard as clear_pool().
        if (std::getenv("MAD_POOL_FREE_TRACE")) {
            std::fprintf(stderr, "[pool-free] dev=%d POOL_FULL cudaFree %p +%zu\n", device, ptr, size);
        }
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // MAD-LAB: same VRAM reserve as ggml_cuda_device_malloc; see ggml_cuda_pool_oom.
            {
                size_t free_b = 0;
                if (!ggml_cuda_vram_budget_check("pool_vmm_grow", device, reserve_size, &free_b)) {
                    throw ggml_cuda_pool_oom(device, reserve_size, free_b);
                }
            }

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

#ifdef USE_CUDA_GRAPH
    // Drain point 3/3: force-sync whatever's left in the retired list. A
    // stream sync is acceptable here (teardown), and necessary -- nothing
    // else will drive these events to completion after this point.
    ggml_cuda_graph_drain_retired(true);
#endif

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (wp_copy_latest_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(wp_copy_latest_event));
    }
    if (wp_copy_stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(wp_copy_stream));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}

#if defined(GGML_USE_HIP) && defined(USE_CUDA_GRAPH)
static void ggml_cuda_hipblaslt_warmup(ggml_backend_cuda_context & ctx) {
    if (const char * env = std::getenv("WP_HIPBLASLT_WARMUP"); env != nullptr && std::strcmp(env, "0") == 0) {
        return;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!fp16_mma_hardware_available(cc)) {
        GGML_LOG_DEBUG("%s: skipping hipBLASLt warm-up on device %d (cc 0x%x)\n", __func__, ctx.device, cc & 0xffff);
        return;
    }

    // hipBLASLt may initialize through the legacy stream on its first GEMM.
    // Do that before any HIP graph capture can start.
    ggml_cuda_set_device(ctx.device);
    cublasHandle_t handle = ctx.cublas_handle();

    float * data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, 3*sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(data, 0, 3*sizeof(float), ctx.stream()));

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            1, 1, 1,
            &alpha, data, CUDA_R_32F, 1,
                    data + 1, CUDA_R_32F, 1,
            &beta,  data + 2, CUDA_R_32F, 1,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        GGML_LOG_WARN("%s: hipBLASLt warm-up failed on device %d: %s\n",
                      __func__, ctx.device, cublas_get_error_str(status));
        (void) cudaFree(data);
        return;
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaFree(data));
}
#endif

static bool ggml_cuda_wp_copy_requested(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    // Default ON for real CUDA/HIP backends (opt OUT with WP_EXPERT_COPY_STREAM=0).
    // The dedicated copy stream hides the paged-in expert's H2D under the
    // compute-stream graph. The 2026-08-07 null that kept this opt-in was the
    // CLASSIC whole-expert rig (few page-ins); the SLICED rig pages ~0.95/req,
    // a regime where hiding the H2D should pay. Unreachable for Vulkan (the
    // caller gates on ggml_backend_is_cuda first); any stream/event/async-copy
    // failure self-disarms to the sync path for the process (ggml_cuda_wp_copy_disarm).
    const char * env = std::getenv("WP_EXPERT_COPY_STREAM");
    if (env != nullptr && std::strcmp(env, "0") == 0) {
        return false;
    }
    return true;
}

static cudaError_t ggml_cuda_wp_copy_stream_create(cudaStream_t * stream) {
#if defined(GGML_USE_HIP)
    return hipStreamCreateWithFlags(stream, hipStreamNonBlocking);
#else
    return cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);
#endif
}

static cudaError_t ggml_cuda_wp_copy_async(void * dst, const void * src, size_t size, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
#else
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
#endif
}

static void ggml_cuda_wp_copy_disarm(ggml_backend_cuda_context * ctx) {
    if (ctx->wp_copy_enabled) {
        cudaDeviceSynchronize();
    }
    ctx->wp_copy_enabled = false;
}

bool ggml_backend_cuda_wp_copy_stream_enabled(ggml_backend_t backend) {
    // Check capability before reading backend->context; backend names are not type checks.
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    if (ctx->wp_copy_initialized) {
        return ctx->wp_copy_enabled;
    }
    ctx->wp_copy_initialized = true;
    if (!ggml_cuda_wp_copy_requested(backend)) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    const cudaError_t stream_status =
        ggml_cuda_wp_copy_stream_create(&ctx->wp_copy_stream);
    const cudaError_t event_status = stream_status == cudaSuccess
        ? cudaEventCreateWithFlags(&ctx->wp_copy_latest_event, cudaEventDisableTiming)
        : stream_status;
    if (stream_status != cudaSuccess || event_status != cudaSuccess) {
        if (ctx->wp_copy_latest_event != nullptr) {
            cudaEventDestroy(ctx->wp_copy_latest_event);
            ctx->wp_copy_latest_event = nullptr;
        }
        if (ctx->wp_copy_stream != nullptr) {
            cudaStreamDestroy(ctx->wp_copy_stream);
            ctx->wp_copy_stream = nullptr;
        }
        cudaDeviceSynchronize();
        return false;
    }
    ctx->wp_copy_enabled = true;
    return true;
}

bool ggml_backend_cuda_wp_copy_tensor_async(ggml_backend_t backend, ggml_tensor * tensor,
                                            const void * data, size_t offset, size_t size) {
    // Check capability before reading backend->context; backend names are not type checks.
    if (!ggml_backend_is_cuda(backend) || !ggml_backend_cuda_wp_copy_stream_enabled(backend)) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    const cudaError_t copy_status = ggml_cuda_wp_copy_async(
        (char *) tensor->data + offset, data, size, ctx->wp_copy_stream);
    if (copy_status != cudaSuccess ||
        cudaEventRecord(ctx->wp_copy_latest_event, ctx->wp_copy_stream) != cudaSuccess) {
        GGML_LOG_WARN("wp copy stream: async H2D failed (%s), disabling\n", cudaGetErrorString(copy_status));
        ggml_cuda_wp_copy_disarm(ctx);
        return false;
    }
    ctx->wp_copy_pending = true;
    return true;
}

bool ggml_backend_cuda_wp_copy_stream_record_event(ggml_backend_t backend,
                                                   ggml_backend_event_t event) {
    // Check capability before reading backend->context; backend names are not type checks.
    if (!ggml_backend_is_cuda(backend) || !ggml_backend_cuda_wp_copy_stream_enabled(backend) || event == nullptr) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    return cudaEventRecord((cudaEvent_t) event->context, ctx->wp_copy_stream) == cudaSuccess;
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

// ggml_backend_cuda_device_context (defined later in this file, alongside the
// other device-management code) tracks how many buffers/backends are active
// per device so ggml_backend_cuda_device_get_memory can safely cudaDeviceReset
// a device nothing is using instead of leaking a lazily-created CUDA context.
// These sites run before that struct's definition, so use forward-declared
// helpers rather than accessing dev_ctx members directly here.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static void ggml_backend_cuda_device_active_count_inc(ggml_backend_dev_t dev);
static void ggml_backend_cuda_device_active_count_dec(ggml_backend_dev_t dev);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    // The ml8 / ml8-fp8 weight-repack caches key on a weight's device pointer.
    // Freeing this buffer invalidates any pointers into it, so drop the repack
    // entries (and their separately-allocated repack buffers) to avoid serving
    // a stale repack if the allocator later hands the same address to a
    // different weight. Cheap no-op when no ml8 weights are in use.
    ggml_cuda_ml8_clear_cache();
    // In-place ML8_FP8 weights live inside this buffer: drop their registry
    // entries (there is no separate allocation to free).
    ggml_cuda_ml8_inplace_forget_range(ctx->dev_ptr, buffer->size);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_device_active_count_dec(buffer->buft->device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);

    // MAD-114: use synchronous cudaMemcpy instead of cudaMemcpyAsync+sync.
    // The previous code copied on cudaStreamPerThread and synced that stream,
    // but the graph's compute kernels run on cuda_ctx->stream() — a different
    // stream. On HIP/RDNA (gfx1201, ROCm 7.2.x) cross-stream visibility isn't
    // reliable even after host-side sync of the source stream, which lets
    // graph kernels read stale input data (see ROCm/hip#3882, #3887).
    // Synchronous cudaMemcpy provides device-wide ordering before returning
    // to the caller. Cost is not negligible for large transfers (e.g. 12.2 MB
    // expert pages) — that was true only for the KB-sized activations this
    // comment originally described.
    if (ggml_cuda_ml8_inplace_eligible(tensor)) {
        ggml_cuda_ml8_inplace_set(cudaStreamPerThread, tensor, data, offset, size, 1, size, size);
        return;
    }
    CUDA_CHECK(cudaMemcpy((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    static const bool trace = getenv("WP_GET_TENSOR_TRACE") != nullptr;
    if (trace) {
        fprintf(stderr, "wp get_tensor: dev=%d name=%s op=%s size=%zu offset=%zu ne=[%lld,%lld,%lld,%lld]\n",
                ctx->device, tensor->name, ggml_op_name(tensor->op), size, offset,
                (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3]);
    }
    if (ggml_cuda_ml8_inplace_eligible(tensor)) {
        ggml_cuda_ml8_inplace_get(cudaStreamPerThread, tensor, data, offset, size);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_ml8_inplace_eligible(tensor)) {
        ggml_cuda_ml8_inplace_set(cudaStreamPerThread, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (ggml_cuda_ml8_inplace_eligible(tensor)) {
        // Strided read of an in-place tensor: unpack per row (load-time / test only).
        for (size_t i = 0; i < n_copies; i++) {
            ggml_cuda_ml8_inplace_get(cudaStreamPerThread, tensor, (char *) data + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream);

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            if (ggml_cuda_ml8_inplace_is_packed(src->data) && ggml_cuda_ml8_inplace_eligible(dst)) {
                CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_cuda_ml8_inplace_alloc_size(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                ggml_cuda_ml8_inplace_alias(src->data, dst->data);
                return true;
            }
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#elif defined(GGML_USE_HIP)
            {
                const size_t nb = ggml_nbytes(src);
                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(dst->data, dst_ctx->device, nb,
                    const_cast<void *>(src->data), src_ctx->device, nb, nb, 1, cudaStreamPerThread));
            }
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_device_active_count_inc(buft->device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // In-place ML8_FP8: allocate the FP8-WMMA kernel layout (e4m3 [K,N] +
    // fp32 scales) instead of the on-disk block layout. No mmvq row padding
    // is needed: these weights never touch the generic quantized kernels.
    if (ggml_cuda_ml8_inplace_eligible(tensor)) {
        return std::max(size, ggml_cuda_ml8_inplace_alloc_size(tensor));
    }

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    // NOTE(fork): this used to take a function-local static std::mutex on EVERY
    // call, although the lock only ever guarded the one-time initialisation
    // below. That made a process-wide serialisation point out of a pure getter.
    // It is called from the hot path (ggml_backend_buft_get_alloc_size /
    // ggml_backend_buffer_get_type, ~18x per expert-dispatch request per
    // device), and because the mutex is a function-local static it is SHARED BY
    // EVERY DEVICE -- so two devices doing host-side work concurrently in one
    // process serialised on it. Profiled 2026-08-29 on the sliced expert rig:
    // 22% of dispatch-path samples in pthread_mutex_lock/unlock and 10% in this
    // function and its callers; two concurrent devices each inflated ~2.7x with
    // ns_submit (the GPU work) untouched. call_once keeps the init exactly as
    // safe and makes the steady state lock-free.
    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];
    static std::once_flag ggml_backend_cuda_buffer_type_once;

    std::call_once(ggml_backend_cuda_buffer_type_once, [] {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
    });

    // Bounds check AFTER init (device_count is stable once the runtime is up)
    // so the common path is a single compare plus a return.
    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    return &ggml_backend_cuda_buffer_types[device];
}

// NOTE(fork): the CUDA split-buffer implementation (-sm row) that lived here was
// removed upstream in 74976e1ae "CUDA: remove -sm row, refactor cuBLAS" (#24216).
// It contained no fork-specific code, and ggml_backend_cuda_split_buffer_type is
// gone from ggml-cuda.h, so it is dropped here rather than carried forward.

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

    // Split AllReduce op slots for the meta backend's two-ubatch overlap
    // (ggml_backend_comm_allreduce_begin / _end).  Internal transport only.
    ggml_cuda_ar_op             ar_ops[GGML_BACKEND_COMM_MAX_OPS];

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
// Input validation shared by the blocking and the split (begin/end) internal
// paths.  Returns false for inputs the pipeline cannot take; the caller then
// falls back.  A zero-element reduce is accepted and the caller skips it.
static bool ggml_backend_cuda_comm_allreduce_internal_check(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return true;
}

static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    if (!ggml_backend_cuda_comm_allreduce_internal_check(comm_ctx, tensors)) {
        return false;
    }
    if (ggml_nelements(tensors[0]) == 0) {
        return true;
    }
    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// Split AllReduce (ggml_backend_comm_allreduce_begin_t / _end_t).  Only the
// internal transport has a split form; NCCL and the butterfly return false
// from begin() so the meta backend uses the blocking call for that reduce.
// The op slots and the two-in-flight limit are the ones of allreduce.cu
// (GGML_CUDA_AR_DX_SLOTS == GGML_BACKEND_COMM_MAX_OPS).
static bool ggml_backend_cuda_comm_allreduce_begin(void * comm_ctx_v, struct ggml_tensor ** tensors, int i_op) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    GGML_ASSERT(i_op >= 0 && i_op < GGML_BACKEND_COMM_MAX_OPS);
    ggml_cuda_ar_op & op = comm_ctx->ar_ops[i_op];
    GGML_ASSERT(!op.pending && "AllReduce op slot begun again before end()");
    // WP_TP_TRACE_FILE only: i_op IS the rolling-loop i_slot in this codebase
    // (every call site passes i_op == (int) slot.i_slot -- see
    // ggml_backend_meta_graph_runner::begin_reduce/end_reduce and their
    // callers). Stashed on the op so allreduce.cu's begin()/end(), which do
    // not otherwise see i_op, can tag AR_PACK_DONE/AR_SENT/AR_RECVD/
    // AR_UNPACK_DONE rows with the right slot (and, via
    // wp_tp_trace_current_ubatch/_subgraph, ubatch_idx/subgraph_idx).
    op.trace_slot = i_op;

    if (comm_ctx->try_allreduce != ggml_backend_cuda_comm_try_allreduce_internal) {
        return false;
    }
    if (!ggml_backend_cuda_comm_allreduce_internal_check(comm_ctx, tensors)) {
        return false;
    }
    if (ggml_nelements(tensors[0]) == 0) {
        return true;
    }
    const bool tracing = wp_tp_trace_enabled();
    if (tracing) {
        wp_tp_trace_log_host(WP_TPT_AR_BEGIN_HOST, wp_tp_trace_current_ubatch(i_op), i_op,
                              wp_tp_trace_current_subgraph(i_op), -1, "phase=enter");
    }
    const bool ok = ggml_cuda_ar_allreduce_begin(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors, &op);
    if (tracing) {
        char extra[64];
        snprintf(extra, sizeof(extra), "phase=exit;op_id=%llu;ok=%d;pending=%d",
                 (unsigned long long) op.op_id, (int) ok, (int) op.pending);
        wp_tp_trace_log_host(WP_TPT_AR_BEGIN_HOST, wp_tp_trace_current_ubatch(i_op), i_op,
                              wp_tp_trace_current_subgraph(i_op), -1, extra);
    }
    // WP_AR_TRACE=N: print the first N AllReduces (tensor, shape, bytes, path) so the
    // per-layer reduce structure can be read off the log.
    static const int trace_n = [] { const char * e = getenv("WP_AR_TRACE"); return e ? atoi(e) : 0; }();
    static int traced = 0;
    if (traced < trace_n) {
        traced++;
        const ggml_tensor * t = tensors[0];
        fprintf(stderr, "wp ar-trace: #%d name=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] bytes=%zu path=%s\n",
                traced, t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                ggml_nbytes(t), ok ? (op.pending ? "duplex" : "sync/chunked") : "fallback");
    }
    return ok;
}

static bool ggml_backend_cuda_comm_allreduce_end(void * comm_ctx_v, int i_op) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    GGML_ASSERT(i_op >= 0 && i_op < GGML_BACKEND_COMM_MAX_OPS);
    ggml_cuda_ar_op & op = comm_ctx->ar_ops[i_op];
    if (!op.pending) {
        return true;
    }
    const bool tracing = wp_tp_trace_enabled();
    if (tracing) {
        wp_tp_trace_log_host(WP_TPT_AR_END_HOST, wp_tp_trace_current_ubatch(i_op), i_op,
                              wp_tp_trace_current_subgraph(i_op), -1, "phase=enter");
    }
    const bool ok = ggml_cuda_ar_allreduce_end(comm_ctx->ar_pipeline, comm_ctx->backends.data(), &op);
    if (tracing) {
        wp_tp_trace_log_host(WP_TPT_AR_END_HOST, wp_tp_trace_current_ubatch(i_op), i_op,
                              wp_tp_trace_current_subgraph(i_op), -1, "phase=exit");
    }
    return ok;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_device_active_count_dec(buffer->buft->device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_device_active_count_inc(buft->device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

#if defined(GGML_USE_HIP)
// Cross-device host staging for mixed-ISA / no-P2P pairs (e.g. gfx1201 + gfx1030 TB3).
//
// Env:
//   GGML_HIP_COPY_STATS=1       count + wall time + direction + atexit summary
//   GGML_HIP_COPY_STATS_EVERY=N also dump every N stage copies
//   GGML_HIP_COPY_STRATEGY=...  peer/stage (peer unsafe on mixed-ISA TB3)
//   GGML_HIP_STAGE_HOST=wc|default|mapped   host slab flags (default: wc)
//   GGML_HIP_STAGE_1D=1         opt-in contiguous 1D hipMemcpy (default off)
//   GGML_HIP_STAGE_BATCH=1      enable multi-input stage batching (default off)
//
// Peer (hipMemcpyPeer / cross-device D2D) segfaults on gfx1201+gfx1030 — always stage.
// Multi-input batch: cpy_tensor_async queues cross-device stages; flush before
// graph_compute does one sync + one setDevice(src) D2H wave + one setDevice(dst) H2D wave.

struct hip_xdev_copy_stats {
    std::atomic<uint64_t> n_stage{0};
    std::atomic<uint64_t> n_stage_2d{0};
    std::atomic<uint64_t> n_stage_1d{0};
    std::atomic<uint64_t> n_stage_fast1d{0}; // contiguous hipMemcpy path
    std::atomic<uint64_t> n_peer{0};
    std::atomic<uint64_t> bytes_stage{0};
    std::atomic<uint64_t> ns_stage{0};
    std::atomic<uint64_t> ns_sync{0};
    std::atomic<uint64_t> n_slab_grow{0};
    std::atomic<uint64_t> n_b_le_4k{0};
    std::atomic<uint64_t> n_b_le_64k{0};
    std::atomic<uint64_t> n_b_le_1m{0};
    std::atomic<uint64_t> n_b_gt_1m{0};
    std::atomic<uint64_t> max_bytes{0};
    // Direction counters (HIP ordinals): useful because TB3 stage is asymmetric.
    std::atomic<uint64_t> n_dir_0_to_1{0};
    std::atomic<uint64_t> n_dir_1_to_0{0};
    std::atomic<uint64_t> n_dir_other{0};
    std::atomic<uint64_t> ns_dir_0_to_1{0};
    std::atomic<uint64_t> ns_dir_1_to_0{0};
    // Multi-input stage batch (sched split inputs amortized)
    std::atomic<uint64_t> n_batch_flushes{0};
    std::atomic<uint64_t> n_batch_items{0};
    std::atomic<uint64_t> n_batch_groups{0}; // (src_dev,dst_dev) groups flushed
    // Call-site tags (where stage traffic comes from)
    std::atomic<uint64_t> n_from_sched{0};   // cpy_tensor_async (split inputs)
    std::atomic<uint64_t> n_from_mul_mat{0}; // ggml_cuda_op_mul_mat peer path
    std::atomic<uint64_t> n_from_cpy2d{0};   // ggml_cuda_cpy_tensor_2d
    std::atomic<uint64_t> n_from_other{0};
    std::atomic<uint64_t> bytes_from_sched{0};
    std::atomic<uint64_t> bytes_from_mul_mat{0};
    std::atomic<uint64_t> bytes_from_cpy2d{0};
    std::atomic<uint64_t> n_unnamed{0}; // sched stages with empty tensor name
    std::atomic<uint64_t> bytes_unnamed{0};
};

static hip_xdev_copy_stats g_hip_xdev_stats;

static bool hip_xdev_stats_enabled() {
    static const bool on = []() {
        const char * e = getenv("GGML_HIP_COPY_STATS");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}

// Coarse name buckets for "what is crossing" (Codex SAFE TRY #1).
struct hip_xdev_name_bucket {
    char     key[48];
    uint64_t n;
    uint64_t bytes;
};
static constexpr int kHipXdevNameBuckets = 96;
static hip_xdev_name_bucket g_hip_xdev_names[kHipXdevNameBuckets];
static std::mutex g_hip_xdev_names_mtx;

// Collapse layer indices and stream views: "hc_attn_post-12 (view)" -> "hc_attn_post* (view)"
static void hip_xdev_family_key(const char * name, char * out, size_t out_n) {
    if (strncmp(name, "blk.", 4) == 0) {
        const char * dot = strchr(name + 4, '.');
        if (dot && dot[1]) {
            name = dot + 1;
        }
    }
    size_t o = 0;
    for (size_t i = 0; name[i] && o + 1 < out_n; ) {
        if (name[i] >= '0' && name[i] <= '9') {
            if (o == 0 || out[o - 1] != '*') {
                out[o++] = '*';
            }
            while (name[i] >= '0' && name[i] <= '9') {
                i++;
            }
            continue;
        }
        out[o++] = name[i++];
    }
    out[o] = '\0';
}

static void hip_xdev_note_name(const char * name, size_t bytes) {
    if (!hip_xdev_stats_enabled()) {
        return;
    }
    if (name == nullptr || name[0] == '\0') {
        g_hip_xdev_stats.n_unnamed.fetch_add(1, std::memory_order_relaxed);
        g_hip_xdev_stats.bytes_unnamed.fetch_add((uint64_t) bytes, std::memory_order_relaxed);
        return;
    }
    char key[48];
    hip_xdev_family_key(name, key, sizeof(key));
    std::lock_guard<std::mutex> lk(g_hip_xdev_names_mtx);
    int free_i = -1;
    for (int i = 0; i < kHipXdevNameBuckets; ++i) {
        if (g_hip_xdev_names[i].key[0] == '\0') {
            if (free_i < 0) {
                free_i = i;
            }
            continue;
        }
        if (strncmp(g_hip_xdev_names[i].key, key, sizeof(g_hip_xdev_names[i].key) - 1) == 0) {
            g_hip_xdev_names[i].n++;
            g_hip_xdev_names[i].bytes += (uint64_t) bytes;
            return;
        }
    }
    if (free_i >= 0) {
        strncpy(g_hip_xdev_names[free_i].key, key, sizeof(g_hip_xdev_names[free_i].key) - 1);
        g_hip_xdev_names[free_i].key[sizeof(g_hip_xdev_names[free_i].key) - 1] = '\0';
        g_hip_xdev_names[free_i].n = 1;
        g_hip_xdev_names[free_i].bytes = (uint64_t) bytes;
    }
}

static uint64_t hip_xdev_stats_every() {
    static const uint64_t every = []() -> uint64_t {
        const char * e = getenv("GGML_HIP_COPY_STATS_EVERY");
        if (e == nullptr || e[0] == '\0') {
            return 0;
        }
        return (uint64_t) strtoull(e, nullptr, 10);
    }();
    return every;
}

static void hip_xdev_stats_print(FILE * f) {
    const uint64_t n   = g_hip_xdev_stats.n_stage.load(std::memory_order_relaxed);
    const uint64_t n2  = g_hip_xdev_stats.n_stage_2d.load(std::memory_order_relaxed);
    const uint64_t n1  = g_hip_xdev_stats.n_stage_1d.load(std::memory_order_relaxed);
    const uint64_t nf  = g_hip_xdev_stats.n_stage_fast1d.load(std::memory_order_relaxed);
    const uint64_t np  = g_hip_xdev_stats.n_peer.load(std::memory_order_relaxed);
    const uint64_t by  = g_hip_xdev_stats.bytes_stage.load(std::memory_order_relaxed);
    const uint64_t ns  = g_hip_xdev_stats.ns_stage.load(std::memory_order_relaxed);
    const uint64_t nss = g_hip_xdev_stats.ns_sync.load(std::memory_order_relaxed);
    const uint64_t mb  = g_hip_xdev_stats.max_bytes.load(std::memory_order_relaxed);
    const uint64_t gr  = g_hip_xdev_stats.n_slab_grow.load(std::memory_order_relaxed);
    const uint64_t d01 = g_hip_xdev_stats.n_dir_0_to_1.load(std::memory_order_relaxed);
    const uint64_t d10 = g_hip_xdev_stats.n_dir_1_to_0.load(std::memory_order_relaxed);
    const uint64_t ns01 = g_hip_xdev_stats.ns_dir_0_to_1.load(std::memory_order_relaxed);
    const uint64_t ns10 = g_hip_xdev_stats.ns_dir_1_to_0.load(std::memory_order_relaxed);
    const uint64_t nbf = g_hip_xdev_stats.n_batch_flushes.load(std::memory_order_relaxed);
    const uint64_t nbi = g_hip_xdev_stats.n_batch_items.load(std::memory_order_relaxed);
    const uint64_t nbg = g_hip_xdev_stats.n_batch_groups.load(std::memory_order_relaxed);
    const double ms    = (double) ns / 1e6;
    const double sync_ms = (double) nss / 1e6;
    const double avg_us = n > 0 ? (double) ns / (double) n / 1e3 : 0.0;
    const double avg01 = d01 > 0 ? (double) ns01 / (double) d01 / 1e3 : 0.0;
    const double avg10 = d10 > 0 ? (double) ns10 / (double) d10 / 1e3 : 0.0;
    fprintf(f,
            "ggml-hip xdev-copy: stage=%" PRIu64 " (2d=%" PRIu64 " 1d=%" PRIu64 " fast1d=%" PRIu64 ") peer=%" PRIu64
            " bytes=%" PRIu64 " max_b=%" PRIu64
            " wall_ms=%.3f avg_us=%.1f sync_ms=%.3f slab_grow=%" PRIu64
            " buckets(<=4k/64k/1M/>1M)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " dir0to1=%" PRIu64 " (avg_us=%.1f) dir1to0=%" PRIu64 " (avg_us=%.1f)"
            " batch_flush=%" PRIu64 " batch_items=%" PRIu64 " batch_groups=%" PRIu64
            " from(sched/mm/cpy2d/other)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " bytes(sched/mm/cpy2d)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            n, n2, n1, nf, np, by, mb, ms, avg_us, sync_ms, gr,
            g_hip_xdev_stats.n_b_le_4k.load(std::memory_order_relaxed),
            g_hip_xdev_stats.n_b_le_64k.load(std::memory_order_relaxed),
            g_hip_xdev_stats.n_b_le_1m.load(std::memory_order_relaxed),
            g_hip_xdev_stats.n_b_gt_1m.load(std::memory_order_relaxed),
            d01, avg01, d10, avg10, nbf, nbi, nbg,
            g_hip_xdev_stats.n_from_sched.load(std::memory_order_relaxed),
            g_hip_xdev_stats.n_from_mul_mat.load(std::memory_order_relaxed),
            g_hip_xdev_stats.n_from_cpy2d.load(std::memory_order_relaxed),
            g_hip_xdev_stats.n_from_other.load(std::memory_order_relaxed),
            g_hip_xdev_stats.bytes_from_sched.load(std::memory_order_relaxed),
            g_hip_xdev_stats.bytes_from_mul_mat.load(std::memory_order_relaxed),
            g_hip_xdev_stats.bytes_from_cpy2d.load(std::memory_order_relaxed));
    // Top name buckets by count
    {
        std::lock_guard<std::mutex> lk(g_hip_xdev_names_mtx);
        int order[kHipXdevNameBuckets];
        int n_ord = 0;
        for (int i = 0; i < kHipXdevNameBuckets; ++i) {
            if (g_hip_xdev_names[i].key[0]) {
                order[n_ord++] = i;
            }
        }
        std::sort(order, order + n_ord, [](int a, int b) {
            return g_hip_xdev_names[a].n > g_hip_xdev_names[b].n;
        });
        const int show = n_ord < 20 ? n_ord : 20;
        if (show > 0) {
            fprintf(f, "ggml-hip xdev-copy names (top %d by count):", show);
            for (int i = 0; i < show; ++i) {
                const auto & b = g_hip_xdev_names[order[i]];
                fprintf(f, " %s:n=%" PRIu64 "/b=%" PRIu64, b.key, b.n, b.bytes);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "ggml-hip xdev-copy unnamed: n=%" PRIu64 " bytes=%" PRIu64 "\n",
                g_hip_xdev_stats.n_unnamed.load(std::memory_order_relaxed),
                g_hip_xdev_stats.bytes_unnamed.load(std::memory_order_relaxed));
    }
    fflush(f);
}

// Optional call-site tag for the next stage record (thread-local).
enum class hip_xdev_src_tag { other, sched, mul_mat, cpy2d };
static thread_local hip_xdev_src_tag g_hip_xdev_src_tag = hip_xdev_src_tag::other;

static void hip_xdev_stats_record(bool is_2d, bool fast1d, size_t bytes, uint64_t ns_total, uint64_t ns_sync,
                                  int src_dev, int dst_dev) {
    if (!hip_xdev_stats_enabled()) {
        return;
    }
    static std::once_flag atexit_once;
    std::call_once(atexit_once, []() {
        atexit([]() { hip_xdev_stats_print(stderr); });
    });

    switch (g_hip_xdev_src_tag) {
        case hip_xdev_src_tag::sched:
            g_hip_xdev_stats.n_from_sched.fetch_add(1, std::memory_order_relaxed);
            g_hip_xdev_stats.bytes_from_sched.fetch_add((uint64_t) bytes, std::memory_order_relaxed);
            break;
        case hip_xdev_src_tag::mul_mat:
            g_hip_xdev_stats.n_from_mul_mat.fetch_add(1, std::memory_order_relaxed);
            g_hip_xdev_stats.bytes_from_mul_mat.fetch_add((uint64_t) bytes, std::memory_order_relaxed);
            break;
        case hip_xdev_src_tag::cpy2d:
            g_hip_xdev_stats.n_from_cpy2d.fetch_add(1, std::memory_order_relaxed);
            g_hip_xdev_stats.bytes_from_cpy2d.fetch_add((uint64_t) bytes, std::memory_order_relaxed);
            break;
        default:
            g_hip_xdev_stats.n_from_other.fetch_add(1, std::memory_order_relaxed);
            break;
    }

    g_hip_xdev_stats.n_stage.fetch_add(1, std::memory_order_relaxed);
    if (is_2d) {
        g_hip_xdev_stats.n_stage_2d.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_hip_xdev_stats.n_stage_1d.fetch_add(1, std::memory_order_relaxed);
    }
    if (fast1d) {
        g_hip_xdev_stats.n_stage_fast1d.fetch_add(1, std::memory_order_relaxed);
    }
    g_hip_xdev_stats.bytes_stage.fetch_add((uint64_t) bytes, std::memory_order_relaxed);
    g_hip_xdev_stats.ns_stage.fetch_add(ns_total, std::memory_order_relaxed);
    g_hip_xdev_stats.ns_sync.fetch_add(ns_sync, std::memory_order_relaxed);

    if (src_dev == 0 && dst_dev == 1) {
        g_hip_xdev_stats.n_dir_0_to_1.fetch_add(1, std::memory_order_relaxed);
        g_hip_xdev_stats.ns_dir_0_to_1.fetch_add(ns_total, std::memory_order_relaxed);
    } else if (src_dev == 1 && dst_dev == 0) {
        g_hip_xdev_stats.n_dir_1_to_0.fetch_add(1, std::memory_order_relaxed);
        g_hip_xdev_stats.ns_dir_1_to_0.fetch_add(ns_total, std::memory_order_relaxed);
    } else {
        g_hip_xdev_stats.n_dir_other.fetch_add(1, std::memory_order_relaxed);
    }

    if (bytes <= 4096) {
        g_hip_xdev_stats.n_b_le_4k.fetch_add(1, std::memory_order_relaxed);
    } else if (bytes <= 65536) {
        g_hip_xdev_stats.n_b_le_64k.fetch_add(1, std::memory_order_relaxed);
    } else if (bytes <= (1u << 20)) {
        g_hip_xdev_stats.n_b_le_1m.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_hip_xdev_stats.n_b_gt_1m.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t prev = g_hip_xdev_stats.max_bytes.load(std::memory_order_relaxed);
    while (bytes > prev &&
           !g_hip_xdev_stats.max_bytes.compare_exchange_weak(prev, (uint64_t) bytes,
                                                             std::memory_order_relaxed)) {
    }

    const uint64_t every = hip_xdev_stats_every();
    if (every > 0) {
        const uint64_t n = g_hip_xdev_stats.n_stage.load(std::memory_order_relaxed);
        if (n % every == 0) {
            hip_xdev_stats_print(stderr);
        }
    }
}

static void hip_xdev_stats_record_peer() {
    if (!hip_xdev_stats_enabled()) {
        return;
    }
    g_hip_xdev_stats.n_peer.fetch_add(1, std::memory_order_relaxed);
}

// Sticky pinned host slabs for residual-sized D2H+H2D (T1). Floor growth at
// 256 KiB so decode residuals do not thrash host malloc.
//
// Host flags (env GGML_HIP_STAGE_HOST):
//   wc      (default) - hipHostMallocWriteCombined; microbench ~best for pure DMA bounce
//   default           - hipHostMallocDefault
//   mapped            - hipHostMallocMapped
constexpr size_t kHipXdevStagingSlabs = 8;
constexpr size_t kHipXdevStagingFloor = 256 * 1024;

struct hip_xdev_staging_slab {
    std::mutex mtx;
    void *     buf = nullptr;
    size_t     cap = 0;
};

static hip_xdev_staging_slab g_hip_xdev_slabs[kHipXdevStagingSlabs];
static std::atomic<uint32_t> g_hip_xdev_next_slab{0};

static unsigned hip_xdev_host_malloc_flags() {
    static const unsigned flags = []() -> unsigned {
        const char * e = getenv("GGML_HIP_STAGE_HOST");
        if (e && !strcmp(e, "default")) {
            return hipHostMallocDefault;
        }
        if (e && !strcmp(e, "mapped")) {
            return hipHostMallocMapped;
        }
        // default: WriteCombined pure-DMA bounce (no CPU touch of payload)
        return hipHostMallocWriteCombined;
    }();
    return flags;
}

// Always hipSetDevice: ambient-device caching is unsafe on ROCm multi-GPU
// (uninitialized thread contexts; see ggml_cuda_set_device).
static void hip_xdev_set_device(int dev) {
    hipSetDevice(dev);
}

// Acquire a sticky slab (lock held until hip_xdev_slab_unlock). *out_buf is pinned host.
static hipError_t hip_xdev_slab_lock(size_t need, void ** out_buf, hip_xdev_staging_slab ** out_slab) {
    const uint32_t idx = g_hip_xdev_next_slab.fetch_add(1, std::memory_order_relaxed) % kHipXdevStagingSlabs;
    hip_xdev_staging_slab & slab = g_hip_xdev_slabs[idx];
    slab.mtx.lock();

    size_t want = need;
    if (want < kHipXdevStagingFloor) {
        want = kHipXdevStagingFloor;
    }
    // Round up to a power of two before comparing against capacity.
    //
    // Slabs are chosen ROUND-ROBIN, so without quantisation every distinct
    // staging size that lands on a slab smaller than it forces a
    // hipHostFree + hipHostMalloc. Pinned-host allocation costs 150-370 ms on
    // this stack, and a server submits many distinct shapes (a full prompt
    // chunk, a short tail, then 1-token decodes) where a benchmark submits one.
    // Measured on llama-server: ~3 reallocations INSIDE every prefill,
    // ~0.7-1.2 s of a ~9.7 s request; llama-bench never reallocated at all
    // because its ubatch shape is uniform.
    // Quantising collapses the size space so the 8 slabs converge after a few
    // requests and then stop allocating, at the cost of at most 2x slack.
    {
        size_t pow2 = kHipXdevStagingFloor;
        while (pow2 < want) {
            pow2 <<= 1;
        }
        want = pow2;
    }
    if (want > slab.cap) {
        if (slab.buf) {
            hipHostFree(slab.buf);
            slab.buf = nullptr;
            slab.cap = 0;
        }
        const unsigned flags = hip_xdev_host_malloc_flags();
        hipError_t err = hipHostMalloc(&slab.buf, want, flags);
        // WriteCombined/Mapped can fail on some drivers; fall back.
        if (err != hipSuccess && flags != hipHostMallocDefault) {
            err = hipHostMalloc(&slab.buf, want, hipHostMallocDefault);
        }
        if (err != hipSuccess) {
            slab.mtx.unlock();
            return hipErrorOutOfMemory;
        }
        slab.cap = want;
        if (hip_xdev_stats_enabled()) {
            g_hip_xdev_stats.n_slab_grow.fetch_add(1, std::memory_order_relaxed);
        }
    }
    *out_buf  = slab.buf;
    *out_slab = &slab;
    return hipSuccess;
}

static void hip_xdev_slab_unlock(hip_xdev_staging_slab * slab) {
    if (slab) {
        slab->mtx.unlock();
    }
}

// Core host-stage copy used by Memcpy2DPeerAsync / cpy_tensor_2d.
// Contiguous payloads use 1D hipMemcpy (faster); pitched use hipMemcpy2D.
static hipError_t hip_xdev_stage_impl(
        void * dst, int dst_dev, size_t dpitch,
        const void * src, int src_dev, size_t spitch,
        size_t width, size_t height,
        hipStream_t producer_stream,
        bool skip_producer_sync = false) {
    if (width == 0 || height == 0) {
        return hipSuccess;
    }
    if (src_dev == dst_dev) {
        hip_xdev_set_device(dst_dev);
        if (spitch == width && dpitch == width) {
            return hipMemcpyAsync(dst, src, width * height, hipMemcpyDeviceToDevice, producer_stream);
        }
        return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice, producer_stream);
    }

    using clock = std::chrono::steady_clock;
    const bool stats = hip_xdev_stats_enabled();
    const auto t0 = stats ? clock::now() : clock::time_point{};

    const auto t_sync0 = stats ? clock::now() : clock::time_point{};
    // Always sync producer unless batch already did (null stream is valid).
    if (!skip_producer_sync) {
        hipStreamSynchronize(producer_stream);
    }
    const uint64_t ns_sync = stats && !skip_producer_sync
        ? (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t_sync0).count()
        : 0;

    int saved_dev = 0;
    hipGetDevice(&saved_dev);

    const size_t staging_size = width * height;
    // 1D only if env opt-in: previously faulted under hetero load without
    // stream-sync; still gated until more soak. WC host is the main win.
    static const bool allow_1d = []() {
        const char * e = getenv("GGML_HIP_STAGE_1D");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    const bool use_1d = allow_1d && (spitch == width && dpitch == width);

    void * host_buf = nullptr;
    hip_xdev_staging_slab * slab = nullptr;
    hipError_t err = hip_xdev_slab_lock(staging_size, &host_buf, &slab);
    if (err != hipSuccess) {
        hipSetDevice(saved_dev);
        return err;
    }

    hip_xdev_set_device(src_dev);
    if (use_1d) {
        err = hipMemcpy(host_buf, src, staging_size, hipMemcpyDeviceToHost);
    } else {
        err = hipMemcpy2D(host_buf, width, src, spitch, width, height, hipMemcpyDeviceToHost);
    }
    if (err == hipSuccess) {
        hip_xdev_set_device(dst_dev);
        if (use_1d) {
            err = hipMemcpy(dst, host_buf, staging_size, hipMemcpyHostToDevice);
        } else {
            err = hipMemcpy2D(dst, dpitch, host_buf, width, width, height, hipMemcpyHostToDevice);
        }
    }
    hip_xdev_slab_unlock(slab);
    hipSetDevice(saved_dev);

    if (stats) {
        const uint64_t ns_total =
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count();
        hip_xdev_stats_record(/*is_2d=*/true, /*fast1d=*/use_1d, staging_size, ns_total, ns_sync,
                              src_dev, dst_dev);
    }
    return err;
}

// ---- Multi-input stage batch (sched split inputs) ----
// cpy_tensor_async queues; graph_compute / synchronize flush with two-phase
// D2H-all / H2D-all per (src_dev,dst_dev) group.

static bool hip_xdev_batch_enabled() {
    static const bool on = []() {
        const char * e = getenv("GGML_HIP_STAGE_BATCH");
        // default OFF: deferred queue proved fragile under WP eval_cb ordering
        // on mixed-ISA; set 1 to enable (flush after split inputs via sched hook).
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

struct hip_xdev_batch_item {
    void *       dst;
    const void * src;
    size_t       nbytes;
    int          src_dev;
    int          dst_dev;
    hipStream_t  src_stream;
};

static constexpr int kHipXdevBatchMax = 32;

struct hip_xdev_batch_state {
    int n = 0;
    hip_xdev_batch_item items[kHipXdevBatchMax];
};

static thread_local hip_xdev_batch_state g_hip_xdev_batch;

static void hip_xdev_batch_flush() {
    hip_xdev_batch_state & b = g_hip_xdev_batch;
    if (b.n <= 0) {
        return;
    }

    const bool stats = hip_xdev_stats_enabled();

    // Sync each unique producer stream once, then stage each item with
    // skip_producer_sync. Safer than packing views into one host blob.
    hipStream_t seen_streams[kHipXdevBatchMax];
    int n_seen = 0;
    for (int i = 0; i < b.n; ++i) {
        hipStream_t s = b.items[i].src_stream;
        bool found = false;
        for (int j = 0; j < n_seen; ++j) {
            if (seen_streams[j] == s) {
                found = true;
                break;
            }
        }
        if (!found) {
            seen_streams[n_seen++] = s;
            hipStreamSynchronize(s);
        }
    }

    int n_groups = 0;
    bool seen_pair[kHipXdevBatchMax] = {};
    for (int i = 0; i < b.n; ++i) {
        if (seen_pair[i]) {
            continue;
        }
        n_groups++;
        for (int j = i; j < b.n; ++j) {
            if (b.items[j].src_dev == b.items[i].src_dev &&
                b.items[j].dst_dev == b.items[i].dst_dev) {
                seen_pair[j] = true;
            }
        }
    }

    for (int i = 0; i < b.n; ++i) {
        const auto & it = b.items[i];
        (void) hip_xdev_stage_impl(it.dst, it.dst_dev, it.nbytes,
                                   it.src, it.src_dev, it.nbytes,
                                   it.nbytes, 1, it.src_stream,
                                   /*skip_producer_sync=*/true);
    }

    if (stats) {
        g_hip_xdev_stats.n_batch_flushes.fetch_add(1, std::memory_order_relaxed);
        g_hip_xdev_stats.n_batch_items.fetch_add((uint64_t) b.n, std::memory_order_relaxed);
        g_hip_xdev_stats.n_batch_groups.fetch_add((uint64_t) n_groups, std::memory_order_relaxed);
    }

    b.n = 0;
}

// Public flush entry (also used from WP eval_cb before ensure).
void ggml_backend_cuda_xdev_batch_flush(void) {
#if defined(GGML_USE_HIP)
    hip_xdev_batch_flush();
#endif
}

// Queue a contiguous cross-device stage. Returns true if handled (possibly deferred).
// Items are flushed in hip_xdev_batch_flush() before graph_compute / synchronize / eval_cb.
static bool hip_xdev_batch_queue(
        void * dst, int dst_dev,
        const void * src, int src_dev,
        size_t nbytes,
        hipStream_t src_stream) {
    if (!hip_xdev_batch_enabled() || nbytes == 0 || dst == nullptr || src == nullptr) {
        return false;
    }
    hip_xdev_batch_state & b = g_hip_xdev_batch;
    // Flush if full
    if (b.n >= kHipXdevBatchMax) {
        hip_xdev_batch_flush();
    }
    b.items[b.n++] = hip_xdev_batch_item{ dst, src, nbytes, src_dev, dst_dev, src_stream };
    return true;
}

#endif // GGML_USE_HIP

// NOTE(fork): ggml_cuda_cpy_tensor_2d, cublas_force_compute_type,
// ggml_cuda_op_mul_mat_cublas and ggml_cuda_op_mul_mat used to follow here. All were
// removed upstream in 74976e1ae (-sm row removal + cuBLAS refactor) and replaced by
// ggml_cuda_mul_mat_cublas below; none of them were fork code.
//
// The hip_xdev cross-device staging above IS ours and is retained -- it is still
// reached via ggml_backend_cuda_cpy_tensor_async / hip_xdev_batch_flush. So is
// ggml_cuda_Memcpy2DPeerAsync below, which carries our no-P2P host-staging path and
// is still forward-declared and called by the buffer cpy_tensor helpers.

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools.
    if (dstDevice == srcDevice) {
        return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice, stream);
    }

    // Strategy selection (env-overridable):
    //   GGML_HIP_COPY_STRATEGY=auto (default): P2P if GGML_CUDA_P2P set, else staging
    //   GGML_HIP_COPY_STRATEGY=peer:   force peer-async row-by-row (requires P2P-capable pair)
    //   GGML_HIP_COPY_STRATEGY=stage:  force staging-buffer path
    //
    // Only attempt P2P if GGML_CUDA_P2P is set, which also calls hipDeviceEnablePeerAccess
    // in ggml_cuda_init(). Without explicit enablement, hipMemcpyPeerAsync (and its internal
    // fallback path) uses a GPU copy kernel that may not be compiled for all ISAs in a
    // mixed-architecture build (e.g. gfx1201 + gfx1030), causing async GPU page faults.
    enum copy_strategy { COPY_AUTO, COPY_PEER, COPY_STAGE };
    static const copy_strategy s_strategy = []() {
        const char * env = getenv("GGML_HIP_COPY_STRATEGY");
        if (env) {
            if (!strcmp(env, "peer"))  { return COPY_PEER;  }
            if (!strcmp(env, "stage")) { return COPY_STAGE; }
        }
        return COPY_AUTO;
    }();

    // Cache P2P capability per (dst, src) pair — hipDeviceCanAccessPeer is a host call but
    // not free, and ggml_cuda_op_mul_mat hits this every multi-GPU matmul.
    static int s_p2p_cache[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {{-1}};
    static std::once_flag s_p2p_cache_init;
    std::call_once(s_p2p_cache_init, []() {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
            for (int j = 0; j < GGML_CUDA_MAX_DEVICES; ++j) {
                s_p2p_cache[i][j] = -1;
            }
        }
    });

    int can_access_peer = 0;
    if (s_strategy != COPY_STAGE) {
        const bool p2p_env_set = (s_strategy == COPY_PEER) || (getenv("GGML_CUDA_P2P") != nullptr);
        if (p2p_env_set) {
            int & cached = s_p2p_cache[dstDevice][srcDevice];
            if (cached < 0) {
                hipDeviceCanAccessPeer(&cached, dstDevice, srcDevice);
            }
            can_access_peer = cached;
        }
    }

    if (can_access_peer) {
        // P2P explicitly enabled: async row-by-row peer copies.
        for (size_t i = 0; i < height; ++i) {
            cudaError_t err = cudaMemcpyPeerAsync(
                (char *) dst + i*dpitch, dstDevice,
                (const char *) src + i*spitch, srcDevice,
                width, stream);
            if (err != cudaSuccess) { return err; }
        }
        hip_xdev_stats_record_peer();
        return cudaSuccess;
    }

    // No P2P: host stage via hip_xdev_stage_impl (WC sticky slabs).
    // Caller must set g_hip_xdev_src_tag if not already (default: other).
    hipError_t err = hip_xdev_stage_impl(dst, dstDevice, dpitch, src, srcDevice, spitch,
                                         width, height, stream);
    g_hip_xdev_src_tag = hip_xdev_src_tag::other;
    return (err == hipSuccess) ? cudaSuccess : (cudaError_t) err;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // FP8_B128 phase 2: the CUDA buffer interface opportunistically converts
    // any in-place-eligible FP8_B128 weight into the AITER preshuffle layout
    // as soon as it's uploaded (ggml_cuda_ml8_inplace_eligible + set_tensor),
    // regardless of which op ends up consuming it. There is no dedicated
    // FP8_B128 GEMM wired into this generic cuBLAS path (that's
    // GGML_OP_FP8_MUL_MAT's job) -- this is only the plain-MUL_MAT dequant
    // fallback (test-backend-ops' FP8_B128 correctness sweep, and any other
    // caller that mat-muls an FP8_B128 tensor without going through the
    // dedicated op). If the weight was already packed on upload, its bytes
    // are the AITER-shuffled layout, not the on-disk block_fp8_b128 layout
    // ggml_get_to_fp32/fp16_cuda's generic dequantizer expects. Mirror what
    // getrows.cu already does for the identical reason: unpack once into a
    // scratch on-disk-layout buffer and read that instead of the live
    // (possibly packed) src0->data.
    ggml_tensor src0_unpacked_storage;
    void * fp8_b128_unpack_scratch = nullptr;
    if (src0->type == GGML_TYPE_FP8_B128) {
        fp8_b128_unpack_scratch = ggml_cuda_ml8_inplace_fp8_b128_unpack_to_device(ctx.stream(), src0);
        if (fp8_b128_unpack_scratch != nullptr) {
            src0_unpacked_storage = *src0;
            src0_unpacked_storage.data = fp8_b128_unpack_scratch;
            src0 = &src0_unpacked_storage;
        }
    }

    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !bf16_mma_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
        // Mirrors the F16 arm above. Without BF16 tensor cores cuBLAS emulates the BF16 compute
        // type extremely slowly; F32 (not F16) is the fallback because the BF16 -> compute-type
        // conversion must stay exact - F16 would round the exponent range away.
        compute_type = GGML_TYPE_F32;
    }
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }

    if (fp8_b128_unpack_scratch != nullptr) {
        // The dispatch above enqueued its dequant/convert + GEMM work on
        // ctx.stream(); make sure it's done reading the scratch buffer
        // before freeing it.
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaFree(fp8_b128_unpack_scratch));
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

// GGML_HINT_MUL_MAT_PIN on a MUL_MAT_ID op pins it to the matrix kernel regardless of
// token count (the worker's grouped prefill sets it); the env below is the older
// token-count-driven form of the same pin
static bool ggml_cuda_mul_mat_id_hint_pinned(const ggml_tensor * dst) {
    return dst->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(dst, 1) == GGML_HINT_MUL_MAT_PIN;
}

static void ggml_cuda_trim_ascii(std::string & s) {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        s.clear();
        return;
    }
    const size_t end = s.find_last_not_of(" \t\r\n");
    s = s.substr(begin, end - begin + 1);
}

// GGML_MUL_MAT_PIN_KERNEL / WP_EXPERT_MM_PIN_KERNEL, per device. Default MMQ.
// "mmvq" / "1" -> mmvq on every device. "ROCm0:mmvq,CUDA0:mmvq" is a map.
// "ROCm0,CUDA0" / "!CPU" is the same allow-list as WP_EXPERT_ARENA_PREFILL.
static bool ggml_cuda_parse_pin_kernel_mmvq(const char * env, const char * device_name) {
    if (env == nullptr || device_name == nullptr) {
        return false;
    }
    std::string value(env);
    ggml_cuda_trim_ascii(value);
    if (value.empty() || value == "0" || value == "mmq") {
        return false;
    }
    if (value == "mmvq" || value == "1") {
        return true;
    }
    if (value.compare(0, 5, "mmvq:") == 0) {
        value.erase(0, 5);
        env = value.c_str();
    } else if (value.compare(0, 4, "mmq:") == 0) {
        return false;
    }
    const bool mapped = value.find(':') != std::string::npos;
    if (mapped) {
        bool all_mapped = true;
        bool found = false;
        bool hit = false;
        size_t start = 0;
        while (start <= value.size()) {
            const size_t comma = value.find(',', start);
            std::string item = (comma == std::string::npos) ?
                value.substr(start) : value.substr(start, comma - start);
            ggml_cuda_trim_ascii(item);
            const size_t colon = item.find(':');
            if (colon == std::string::npos) {
                all_mapped = false;
                break;
            }
            std::string name = item.substr(0, colon);
            std::string kernel = item.substr(colon + 1);
            ggml_cuda_trim_ascii(name);
            ggml_cuda_trim_ascii(kernel);
            if (!name.empty() && name == device_name) {
                hit = true;
                found = (kernel == "mmvq" || kernel == "1");
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        if (all_mapped) {
            return hit && found;
        }
    }
    bool negate = false;
    if (!value.empty() && value[0] == '!') {
        negate = true;
        value.erase(0, 1);
    }
    bool listed = false;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        std::string name = (comma == std::string::npos) ?
            value.substr(start) : value.substr(start, comma - start);
        ggml_cuda_trim_ascii(name);
        if (!name.empty() && name == device_name) {
            listed = true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return negate ? !listed : listed;
}

static bool ggml_cuda_pin_kernel_mmvq(ggml_backend_cuda_context & ctx) {
    if (ctx.wp_pin_mmvq < 0) {
        const char * env = std::getenv("GGML_MUL_MAT_PIN_KERNEL");
        if (env != nullptr && env[0] != '\0') {
            ctx.wp_pin_mmvq = ggml_cuda_parse_pin_kernel_mmvq(env, ctx.name.c_str()) ? 1 : 0;
        } else {
            ctx.wp_pin_mmvq = ggml_cuda_parse_pin_kernel_mmvq(
                std::getenv("WP_EXPERT_MM_PIN_KERNEL"), ctx.name.c_str()) ? 1 : 0;
        }
    }
    return ctx.wp_pin_mmvq == 1;
}

static bool ggml_cuda_mul_mat_id_force_mm(const int64_t total_tokens) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MUL_MAT_ID_FORCE_MM");
        return env != nullptr && std::strtol(env, nullptr, 10) != 0;
    }();
    static const int64_t min_tokens = [] {
        const char * env = std::getenv("GGML_MUL_MAT_ID_FORCE_MM_MIN_TOKENS");
        if (env == nullptr) {
            return int64_t(64);
        }
        char * end = nullptr;
        const long value = std::strtol(env, &end, 10);
        return end != env && *end == '\0' && value >= 0 ? int64_t(value) : int64_t(64);
    }();
    return enabled && total_tokens >= min_tokens;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    if (tensor->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(tensor, 1) == GGML_HINT_MUL_MAT_PIN) {
        return false;
    }

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (is_mul_mat_id && (ggml_cuda_mul_mat_id_force_mm(src1->ne[2]) || ggml_cuda_mul_mat_id_hint_pinned(tensor))) {
        return false;
    }
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;
    // PIN stays unfused. Fusion launches only at ncols_dst=1 (mmvq.cu asserts
    // that). The mmvq pin pads to MMVQ_MAX_BATCH_SIZE=8 so nwarps/rows_per_block
    // do not depend on ne11; fused ncols=1 uses a different nwarps (RDNA2 Q4_K:
    // 2 vs 1) so the last bits would depend on draft length. Same K order at
    // equal ncols_dst, but equal ncols is 8 here, which fusion cannot take.
    if (tensor->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(tensor, 1) == GGML_HINT_MUL_MAT_PIN) {
        return false;
    }
    if (is_mul_mat_id && (ggml_cuda_mul_mat_id_force_mm(src1->ne[2]) || ggml_cuda_mul_mat_id_hint_pinned(tensor))) {
        return false;
    }

    const bool is_tq_weight = (src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && !is_tq_weight &&
                             src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }
    const bool force_mm = hint == GGML_HINT_MUL_MAT_PIN;

    // MAD Task 11: scaled-fp8 (ml8-fp8) weights stay a plain GGML_OP_MUL_MAT
    // (no centroid sidecar, no load-time op-swap). Route them to the no-LUT
    // FP8-WMMA path (WEIGHT_FORMAT=0) before any of the generic mul_mat
    // kernels — none of which understand the ML8_FP8 block layout.
    if (src0->type == GGML_TYPE_ML8_FP8) {
        // ggml_cuda_mul_mat_id() decomposes a MoE op into per-expert slices and calls
        // us with a synthetic dst (memset to 0), so dst->src[] is null on that path --
        // it passes the operands as arguments instead. ggml_cuda_op_ml8_fp8_mul_mat()
        // reads dst->src[0]/[1], so hand it a copy with those wired up. When we are
        // called normally from ggml_cuda_compute_forward these already match dst->src[].
        ggml_tensor dst_ml8 = *dst;
        dst_ml8.src[0] = const_cast<ggml_tensor *>(src0);
        dst_ml8.src[1] = const_cast<ggml_tensor *>(src1);
        ggml_cuda_op_ml8_fp8_mul_mat(ctx, &dst_ml8);
        return;
    }

    // ml8-4 sidecar guard (MAD-223): F8_E4M3 is claimed in supports_op so the
    // centroid sidecars stay on the HIP buffer alongside their ml8_4 weights,
    // but the centroids are consumed via GGML_OP_ML8_MUL_MAT, never as a real
    // MUL_MAT weight. Hitting this path means someone built a graph that
    // matmuls F8_E4M3 directly — none of the CUDA mul_mat kernels know that
    // type, so fail loudly rather than silently corrupting outputs.
    GGML_ASSERT(src0->type != GGML_TYPE_F8_E4M3 &&
        "GGML_TYPE_F8_E4M3 is not a real MUL_MAT weight type on the HIP backend "
        "(use GGML_OP_ML8_MUL_MAT for ml8 dispatch)");

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (force_mm) {
        GGML_ASSERT(!bad_padding_clear);
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        // Default pin = MMQ: at 1-8 columns it costs the same as the MMVQ groups and at
        // 64-512 columns it is 2-12x cheaper on R9700, 6900XT and GTX1070
        // (test-wp-mul-mat-pin WP_PIN_TEST_BENCH, 2026-09-02). GGML_MUL_MAT_PIN_KERNEL=mmvq
        // (or WP_EXPERT_MM_PIN_KERNEL, per device) runs the vector kernel in fixed
        // groups of MMVQ_MAX_BATCH_SIZE columns instead (tail zero-padded); both keep
        // a column's arithmetic independent of ne11. If ne11 is already a multiple of
        // the group, skip the per-mul_mat memset/memcpy -- the worker pads once.
        const bool pin_mmvq = ggml_cuda_pin_kernel_mmvq(ctx);
        const bool mmvq_ok = pin_mmvq && ggml_is_quantized(src0->type) && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
        if (!mmvq_ok) {
            GGML_ASSERT(ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0, true));
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst, true);
            return;
        }
        constexpr int64_t group = MMVQ_MAX_BATCH_SIZE;
        auto mmvq_group_view = [&](const ggml_tensor * src1_v, ggml_tensor * dst_v, int64_t c0) {
            ggml_tensor y = *src1_v;
            ggml_tensor d = *dst_v;
            y.ne[1] = group; y.ne[2] = 1; y.ne[3] = 1;
            y.nb[1] = ne10 * sizeof(float); y.nb[2] = y.nb[1] * group; y.nb[3] = y.nb[2];
            d.ne[1] = group; d.ne[2] = 1; d.ne[3] = 1;
            d.nb[1] = ne0 * sizeof(float); d.nb[2] = d.nb[1] * group; d.nb[3] = d.nb[2];
            y.data = (char *) src1_v->data + c0 * nb11;
            d.data = (char *) dst_v->data + c0 * nb1;
            y.src[0] = nullptr; y.view_src = nullptr; y.op = GGML_OP_NONE;
            d.src[0] = nullptr; d.src[1] = nullptr; d.view_src = nullptr;
            ggml_cuda_mul_mat_vec_q(ctx, src0, &y, nullptr, &d);
        };
        if (ne11 == group) {
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            return;
        }
        if (ne11 % group == 0) {
            for (int64_t c0 = 0; c0 < ne11; c0 += group) {
                mmvq_group_view(src1, dst, c0);
            }
            return;
        }
        cudaStream_t stream = ctx.stream();
        ggml_cuda_pool_alloc<float> y_pad(ctx.pool());
        ggml_cuda_pool_alloc<float> d_pad(ctx.pool());
        for (int64_t c0 = 0; c0 < ne11; c0 += group) {
            const int64_t nc = std::min(group, ne11 - c0);
            if (nc == group) {
                mmvq_group_view(src1, dst, c0);
                continue;
            }
            ggml_tensor y = *src1;
            ggml_tensor d = *dst;
            y.ne[1] = group; y.ne[2] = 1; y.ne[3] = 1;
            y.nb[1] = ne10 * sizeof(float); y.nb[2] = y.nb[1] * group; y.nb[3] = y.nb[2];
            d.ne[1] = group; d.ne[2] = 1; d.ne[3] = 1;
            d.nb[1] = ne0 * sizeof(float); d.nb[2] = d.nb[1] * group; d.nb[3] = d.nb[2];
            y_pad.alloc((size_t) (ne10 * group));
            d_pad.alloc((size_t) (ne0 * group));
            CUDA_CHECK(cudaMemsetAsync(y_pad.get(), 0, (size_t) (ne10 * group) * sizeof(float), stream));
            CUDA_CHECK(cudaMemcpyAsync(y_pad.get(), (char *) src1->data + c0 * nb11,
                (size_t) (ne10 * nc) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            y.data = y_pad.get();
            d.data = d_pad.get();
            y.src[0] = nullptr; y.view_src = nullptr; y.op = GGML_OP_NONE;
            d.src[0] = nullptr; d.src[1] = nullptr; d.view_src = nullptr;
            ggml_cuda_mul_mat_vec_q(ctx, src0, &y, nullptr, &d);
            CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + c0 * nb1, d_pad.get(),
                (size_t) (ne0 * nc) * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    // fork: TQ weight types (TQ4_1S/TQ3_1S) have bespoke kernels and are not
    // understood by the mmvq/mmq paths, so they are dispatched here rather than
    // being excluded from each generic predicate as before the 2026-07-31 sync.
    if (src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S) {
        if (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE) {
            // Fused TQ weight mul_mat: handles decode (ne[1]=1) and speculative (ne[1]<=8)
            ggml_cuda_mul_mat_tq(ctx, src0, src1, dst);
        } else if (src0->type == GGML_TYPE_TQ4_1S) {
            // Large prefill: runtime TQ4_1S → fp16 dequant + cuBLAS tensor cores
            ggml_cuda_mul_mat_tq4_1s_cublas(ctx, src0, src1, dst);
        } else {
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        }
        return;
    }

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    // NOTE: the threshold here is deliberately MMVQ_MAX_BATCH_SIZE (8) and not
    // MMVF_MAX_BATCH_SIZE: MMVF was widened to 16 for BF16 on pre-Ampere/pre-RDNA3, but this
    // transposed-vector fallback is an F32 heuristic whose tuning point did not move.
    if (ne01 == 1 && ne11 > MMVQ_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const bool force_mm = ggml_cuda_mul_mat_id_force_mm(src1->ne[2]) || ggml_cuda_mul_mat_id_hint_pinned(dst);

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    // TQ weight types have no mmvq/mmq kernels: they always take the
    // dequant-to-f16 cuBLAS fallback, which synchronizes the stream.
    if (src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S) {
        return true;
    }

    if (!force_mm && dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2], force_mm)) {
        return false;
    }

    if (!force_mm && ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool force_mm = ggml_cuda_mul_mat_id_force_mm(ne12) || ggml_cuda_mul_mat_id_hint_pinned(dst);

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    // TQ weight types use dequant-to-f16 cuBLAS path only (no mmvq/mmq kernels)
    const bool is_tq_weight_id = (src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);

    // MAD-88 Phase 2 (A2): when the weight-pager eval callback armed
    // routing-aware expert pointers for this op, both MMVQ and MMQ honor
    // them (MMVQ via Phase A2, MMQ via Phase A1). MMVF does not — gate
    // it off when routing is active. The consolidated parent's
    // src0->data is a placeholder; kernels that ignore expert_ptrs and
    // dereference it with the per-expert stride will fault.
    const bool routing_active = ggml_cuda_has_routed_expert_ptrs() ||
        ggml_mul_mat_id_get_expert_ptrs_n_as(dst) > 0;

    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        // MMVF is instantiated at least as wide as MMVQ, so the single MMVQ-sized gate below
        // bounds both the quantized (MMVQ) and non-quantized (MMVF) MUL_MAT_ID branches.
        // MUL_MAT_ID routing intentionally stays at the MMVQ width.
        static_assert(MMVF_MAX_BATCH_SIZE >= MMVQ_MAX_BATCH_SIZE);
        if (!force_mm && ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type) && !is_tq_weight_id) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else if (!ggml_is_quantized(src0->type)) {
                if (!routing_active && GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        // routing_active forces MMQ: it is the only mul_mat_id path that honors the
        // weight-pager expert pointers here (MMVQ handled above for small batch). On
        // gfx80x ggml_cuda_should_use_mmq() now returns false for large batches (no
        // hardware dp4a -> route to dequant+hipBLAS), so without this OR the routing
        // case would fall through to the GGML_ABORT below.
        if (routing_active || ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02, force_mm)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst, force_mm);
            return;
        }

        if (!force_mm && !routing_active && ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }

        if (routing_active) {
            // Routing was armed but no kernel path honors expert_ptrs for
            // this op shape. Falling through to the dequant path below
            // would read the consolidated placeholder and fault. Abort
            // with a clear message — at this point only the cuBLAS
            // dequant path is left, which would need its own hook.
            GGML_ABORT("weight-pager routing armed but no routing-aware kernel "
                       "matched (src0 type %s, ne12=%lld, ne02=%lld).",
                       ggml_type_name(src0->type), (long long) ne12, (long long) ne02);
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // MAD-230 DIAG: host-side log of every op being dispatched, including
    // src data pointers. Gated by env GGML_CUDA_OP_TRACE=1. Purely passive
    // (CPU fprintf only — no sync, no DMA, no GPU scheduling impact). When
    // the GPU faults, the LAST printed op identifies the kernel-in-flight
    // (or the one launched immediately before). Src data ptrs in low
    // address range (< 0x100000) flag a tensor whose data wasn't patched
    // by the weight pager (would manifest as a near-null GPU fault).
    static const bool s_op_trace = []() {
        const char * env = std::getenv("GGML_CUDA_OP_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    if (s_op_trace) {
        static int s_op_idx = 0;
        ++s_op_idx;
        const bool routing_set = ggml_cuda_has_routed_expert_ptrs();
        const char * route_tag = routing_set ? " ROUTING_SET" : "";
        fprintf(stderr, "[op #%d] %s name=%s%s",
                s_op_idx, ggml_op_name(dst->op), dst->name, route_tag);
        for (int i = 0; i < 3; ++i) {
            struct ggml_tensor * s = dst->src[i];
            if (s == nullptr) break;
            const uintptr_t a = (uintptr_t) s->data;
            const char * warn = (a != 0 && a < 0x100000ULL) ? " <-- LOW_ADDR" : "";
            fprintf(stderr, "  src[%d]=%s data=0x%lx%s",
                    i, s->name, (unsigned long) a, warn);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_TURBO_WHT:
            ggml_cuda_turbo_wht(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_SINKHORN_NORM:
            ggml_cuda_op_sinkhorn_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_ML8_MUL_MAT:
            ggml_cuda_op_ml8_mul_mat(ctx, dst);
            break;
        case GGML_OP_ML8_APPLY_ROTATION:
            ggml_cuda_op_ml8_apply_rotation(ctx, dst);
            break;
        case GGML_OP_ML8_MUL_MAT_ID:
            ggml_cuda_op_ml8_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_ML8_GET_ROWS:
            ggml_cuda_op_ml8_get_rows(ctx, dst);
            break;
        case GGML_OP_FP8_QUANT_ROT:
            ggml_cuda_op_fp8_quant_rot(ctx, dst);
            break;
        case GGML_OP_FP8_MUL_MAT:
            ggml_cuda_op_fp8_mul_mat(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_PAGED_ATTN_MT:
            mt::ggml_cuda_op_paged_attn_mt(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_device_active_count_dec(backend->device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_set_device(cuda_ctx->device);
    if (ggml_cuda_ml8_inplace_eligible(tensor)) {
        ggml_cuda_ml8_inplace_set(cuda_ctx->stream(), tensor, data, offset, size, 1, size, size);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    // A degenerate 2D copy (one row, or rows that are contiguous on both
    // sides) is a plain 1D copy. On ROCm this matters: CLR runs every 2D host
    // copy through its "unpinned rect" path, which is synchronous and queues
    // behind all in-flight work even from pinned memory (the meta backend's
    // per-device splice of a row-split input is exactly this shape).
    if (n_copies == 1 || (stride_tensor == size && stride_data == size)) {
        CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size * n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    if (n_copies == 1 || (stride_tensor == size && stride_data == size)) {
        CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size * n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#elif defined(GGML_USE_HIP)
            {
                const size_t nb = ggml_nbytes(dst);
                // For anonymous graph nodes, key by op+shape so we can fix sources.
                if (src->name[0] && strncmp(src->name, "node_", 5) == 0) {
                    char key[64];
                    snprintf(key, sizeof(key), "node/%s/%ldx%ldx%ld%s",
                             ggml_op_name(src->op),
                             (long) src->ne[0], (long) src->ne[1], (long) src->ne[2],
                             src->view_src ? "/v" : "");
                    hip_xdev_note_name(key, nb);
                } else if (src->name[0]) {
                    hip_xdev_note_name(src->name, nb);
                } else if (dst->name[0]) {
                    hip_xdev_note_name(dst->name, nb);
                } else {
                    char key[64];
                    snprintf(key, sizeof(key), "node/%s/%ldx%ldx%ld",
                             ggml_op_name(src->op),
                             (long) src->ne[0], (long) src->ne[1], (long) src->ne[2]);
                    hip_xdev_note_name(key, nb);
                }
                g_hip_xdev_src_tag = hip_xdev_src_tag::sched;
                // Queue for multi-input batch flush before graph_compute.
                // Falls back to immediate stage if batching disabled.
                if (hip_xdev_batch_queue(dst->data, cuda_ctx_dst->device,
                                         src->data, cuda_ctx_src->device,
                                         nb, cuda_ctx_src->stream())) {
                    // Defer host-stage until graph_compute/synchronize flush.
                    // Do NOT skip events when batch is empty after a forced
                    // path; only skip when items are still pending.
                    if (g_hip_xdev_batch.n > 0) {
                        g_hip_xdev_src_tag = hip_xdev_src_tag::other;
                        return true;
                    }
                    // Flushed inside queue (batch was full) - fall through to
                    // events so dst stream is ordered for subsequent work.
                } else {
                    CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(dst->data, cuda_ctx_dst->device, nb,
                        const_cast<void *>(src->data), cuda_ctx_src->device, nb, nb, 1, cuda_ctx_src->stream()));
                }
                g_hip_xdev_src_tag = hip_xdev_src_tag::other;
            }
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        {
            char tag[80];
            snprintf(tag, sizeof(tag), "cpy<-dev%d:%s", cuda_ctx_src->device, dst->name);
            ggml_cuda_ar_wait_logged(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, tag, 0, cuda_ctx_dst->device);
        }
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
#if defined(GGML_USE_HIP)
    hip_xdev_batch_flush();
#endif
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    if (cuda_ctx->wp_copy_enabled && cuda_ctx->wp_copy_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->wp_copy_stream));
    }

    GGML_UNUSED(backend);
}

bool ggml_backend_cuda_synchronize_compute(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    return true;
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_wp_persistent_graphs_enabled() {
    static const bool enabled = [] {
        // Deliberately a SEPARATE knob from WP_PERSISTENT_GRAPHS (vk plans):
        // this one overrides the sub-Volta CUDA-graph arch guard, which has
        // historically shielded capture bugs — measure it as its own arm.
        const char * env = getenv("WP_PERSISTENT_CUDA_GRAPHS");
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled;
}

static bool ggml_cuda_wp_hip_graphs_enabled() {
    // Cached: this is called ~6x per backend graph_compute, and the split
    // decode forward makes ~86 of those per token, so a raw getenv() here was
    // ~500 environ scans/token for a value that cannot change mid-process.
    static const bool enabled = [] {
        const char * env = getenv("WP_HIP_GRAPHS");
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled || ggml_cuda_wp_persistent_graphs_enabled();
}

struct ggml_cuda_wp_graph_counters {
    std::atomic<uint64_t> captures{0};
    std::atomic<uint64_t> replays{0};
    std::atomic<uint64_t> fallbacks{0};
    // WHY a capture happened. capture_reason was already recorded per graph but
    // never read, so a collapsing capture:replay ratio was indistinguishable
    // between "new shape" (benign warmup) and "cache too small" (thrash).
    std::atomic<uint64_t> cap_newkey{0};
    std::atomic<uint64_t> cap_lru{0};
    std::atomic<uint64_t> cap_ttl{0};
    std::atomic<uint64_t> cap_recapture{0};
    // wp hip-graphs churn diagnostic: recapture cause/flip breakdown, updated
    // only from ggml_cuda_graph_update_required's real-recapture path (see
    // that function) -- these cost nothing on the no-change hot path.
    // recap_total/recap_topo/recap_addr classify every real recapture by
    // cause (topo-only vs addr-only vs both, where "both" is
    // recap_total - recap_topo - recap_addr); recap_flip counts how many of
    // those recaptures were an A->B->A flip back to a value seen two
    // recaptures ago at the same key (see ggml_cuda_graph::last_recap_data0 /
    // recap_prev2_data0 in common.cuh).
    std::atomic<uint64_t> recap_total{0};
    std::atomic<uint64_t> recap_flip{0};
    std::atomic<uint64_t> recap_topo{0};
    std::atomic<uint64_t> recap_addr{0};
    std::atomic<uint64_t> live_graphs{0};
    // 2026-09-11: deferred-destroy retired-graph list (common.cuh
    // ggml_cuda_graph_retire()/drain_retired()) -- see hipgraph-ttl-race.patch.
    // retired = current size of the per-device retired list (gauge);
    // retired_freed = cumulative graphs actually destroyed via that list;
    // retired_synced = of those, how many needed the bounded-list fallback sync.
    std::atomic<uint64_t> retired{0};
    std::atomic<uint64_t> retired_freed{0};
    std::atomic<uint64_t> retired_synced{0};
    // vram-budget: sum of measured hipGraphInstantiate deltas (ggml_cuda_graph::
    // exec_bytes) for execs currently alive on this device -- added at
    // instantiate, subtracted when the exec is actually destroyed (either the
    // HIP recapture destroy site below or ~ggml_cuda_graph()). budget_evicted
    // counts LRU evictions triggered specifically because exec_bytes_live
    // exceeded GGML_CUDA_GRAPH_VRAM_BUDGET_MB, as opposed to the cap/TTL
    // evictions already tracked above.
    std::atomic<size_t>   exec_bytes_live{0};
    std::atomic<uint64_t> budget_evicted{0};
};

static ggml_cuda_wp_graph_counters ggml_cuda_wp_graph_counts[GGML_CUDA_MAX_DEVICES];

#ifdef USE_CUDA_GRAPH
// vram-budget: accessors declared in common.cuh so ggml_cuda_graph's
// destructor and ggml_backend_cuda_context::cuda_graph()/ggml_cuda_graph_retire()
// (both header-only, included before this struct exists) can reach the
// per-device counters above without common.cuh needing to know about
// ggml_cuda_wp_graph_counters itself.
void ggml_cuda_graph_wp_live_inc(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    ggml_cuda_wp_graph_counts[device].live_graphs.fetch_add(1, std::memory_order_relaxed);
}

void ggml_cuda_graph_wp_live_dec(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    ggml_cuda_wp_graph_counts[device].live_graphs.fetch_sub(1, std::memory_order_relaxed);
}

void ggml_cuda_graph_wp_exec_bytes_sub(int device, size_t bytes) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || bytes == 0) {
        return;
    }
    ggml_cuda_wp_graph_counts[device].exec_bytes_live.fetch_sub(bytes, std::memory_order_relaxed);
}
#endif // USE_CUDA_GRAPH
static std::once_flag ggml_cuda_wp_graph_atexit_once;

static void ggml_cuda_wp_graph_print_counts() {
    uint64_t captures = 0, replays = 0, fallbacks = 0;
    uint64_t newkey = 0, lru = 0, ttl = 0, live = 0, recap = 0;
    uint64_t retired = 0, retired_freed = 0, retired_synced = 0;
    uint64_t budget_evicted = 0;
    uint64_t recap_total = 0, recap_flip = 0, recap_topo = 0, recap_addr = 0;
    size_t   exec_bytes_live_total = 0;
    // vram-budget: per-device exec_mb, appended to the line below so a
    // per-device VRAM skew (e.g. two contexts sharing device 0) is visible
    // without cross-referencing the "wp vram:" line.
    char exec_mb_per_dev[256];
    size_t exec_mb_per_dev_len = 0;
    exec_mb_per_dev[0] = '\0';
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        captures += ggml_cuda_wp_graph_counts[i].captures.load(std::memory_order_relaxed);
        replays += ggml_cuda_wp_graph_counts[i].replays.load(std::memory_order_relaxed);
        fallbacks += ggml_cuda_wp_graph_counts[i].fallbacks.load(std::memory_order_relaxed);
        newkey += ggml_cuda_wp_graph_counts[i].cap_newkey.load(std::memory_order_relaxed);
        lru += ggml_cuda_wp_graph_counts[i].cap_lru.load(std::memory_order_relaxed);
        ttl += ggml_cuda_wp_graph_counts[i].cap_ttl.load(std::memory_order_relaxed);
        recap += ggml_cuda_wp_graph_counts[i].cap_recapture.load(std::memory_order_relaxed);
        live += ggml_cuda_wp_graph_counts[i].live_graphs.load(std::memory_order_relaxed);
        retired += ggml_cuda_wp_graph_counts[i].retired.load(std::memory_order_relaxed);
        retired_freed += ggml_cuda_wp_graph_counts[i].retired_freed.load(std::memory_order_relaxed);
        retired_synced += ggml_cuda_wp_graph_counts[i].retired_synced.load(std::memory_order_relaxed);
        budget_evicted += ggml_cuda_wp_graph_counts[i].budget_evicted.load(std::memory_order_relaxed);
        recap_total += ggml_cuda_wp_graph_counts[i].recap_total.load(std::memory_order_relaxed);
        recap_flip += ggml_cuda_wp_graph_counts[i].recap_flip.load(std::memory_order_relaxed);
        recap_topo += ggml_cuda_wp_graph_counts[i].recap_topo.load(std::memory_order_relaxed);
        recap_addr += ggml_cuda_wp_graph_counts[i].recap_addr.load(std::memory_order_relaxed);
        const size_t dev_exec_bytes = ggml_cuda_wp_graph_counts[i].exec_bytes_live.load(std::memory_order_relaxed);
        exec_bytes_live_total += dev_exec_bytes;
        if (dev_exec_bytes != 0 && exec_mb_per_dev_len < sizeof(exec_mb_per_dev) - 32) {
            const int n = snprintf(exec_mb_per_dev + exec_mb_per_dev_len, sizeof(exec_mb_per_dev) - exec_mb_per_dev_len,
                                    " dev%d=%.1f", i, dev_exec_bytes / 1048576.0);
            if (n > 0) {
                exec_mb_per_dev_len += (size_t) n;
            }
        }
    }
    static std::atomic<uint64_t> last_captures{0};
    static std::atomic<uint64_t> last_replays{0};
    static std::atomic<uint64_t> last_fallbacks{0};
    const uint64_t interval_captures = captures - last_captures.exchange(captures, std::memory_order_relaxed);
    const uint64_t interval_replays  = replays  - last_replays.exchange(replays, std::memory_order_relaxed);
    const uint64_t interval_fallbacks = fallbacks - last_fallbacks.exchange(fallbacks, std::memory_order_relaxed);
    // WP_HIP_GRAPHS_LOG=1 to print; counters above are still maintained either way.
    if (!ggml_cuda_wp_hip_graphs_log_enabled()) {
        return;
    }
    fprintf(stderr, "wp hip-graphs: hits=%llu captures=%llu fallbacks=%llu "
            "interval(hits=%llu captures=%llu fallbacks=%llu) "
            "(newkey=%llu lru_evicted=%llu ttl_evicted=%llu recapture=%llu live=%llu "
            "retired=%llu retired_freed=%llu retired_synced=%llu budget_evicted=%llu "
            "recap_total=%llu recap_flip=%llu recap_topo=%llu recap_addr=%llu "
            "exec_mb=%.1f exec_mb_per_dev=[%s])\n",
            (unsigned long long) replays, (unsigned long long) captures,
            (unsigned long long) fallbacks,
            (unsigned long long) interval_replays, (unsigned long long) interval_captures,
            (unsigned long long) interval_fallbacks,
            (unsigned long long) newkey,
            (unsigned long long) lru, (unsigned long long) ttl,
            (unsigned long long) recap, (unsigned long long) live,
            (unsigned long long) retired, (unsigned long long) retired_freed,
            (unsigned long long) retired_synced, (unsigned long long) budget_evicted,
            (unsigned long long) recap_total, (unsigned long long) recap_flip,
            (unsigned long long) recap_topo, (unsigned long long) recap_addr,
            exec_bytes_live_total / 1048576.0, exec_mb_per_dev);
    // MAD-LAB 2026-09-13: VRAM trace in the same periodic line -- free/total per
    // device from the driver (includes every other process on the card), so the
    // server log shows when device memory moves and next to which graph stats.
    {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) == cudaSuccess) {
            int cur = 0;
            (void)cudaGetDevice(&cur);
            std::string line = "wp vram:";
            for (int d = 0; d < ndev && d < GGML_CUDA_MAX_DEVICES; ++d) {
                size_t free_b = 0, total_b = 0;
                if (cudaSetDevice(d) == cudaSuccess && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
                    char buf[96];
                    snprintf(buf, sizeof(buf), " dev%d free=%.0fMiB used=%.0fMiB", d,
                             free_b / 1048576.0, (total_b - free_b) / 1048576.0);
                    line += buf;
                }
            }
            (void)cudaSetDevice(cur);
            (void)cudaGetLastError();
            fprintf(stderr, "%s\n", line.c_str());
        }
    }
}

bool ggml_backend_cuda_wp_graph_counts(
        ggml_backend_t backend,
        uint64_t * captures, uint64_t * replays, uint64_t * fallbacks,
        uint64_t * cap_newkey, uint64_t * cap_lru) {
    if (!ggml_backend_is_cuda(backend) || backend->context == nullptr) {
        return false;
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    if (ctx->device < 0 || ctx->device >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    const ggml_cuda_wp_graph_counters & c = ggml_cuda_wp_graph_counts[ctx->device];
    if (captures) {
        *captures = c.captures.load(std::memory_order_relaxed);
    }
    if (replays) {
        *replays = c.replays.load(std::memory_order_relaxed);
    }
    if (fallbacks) {
        *fallbacks = c.fallbacks.load(std::memory_order_relaxed);
    }
    if (cap_newkey) {
        *cap_newkey = c.cap_newkey.load(std::memory_order_relaxed);
    }
    if (cap_lru) {
        *cap_lru = c.cap_lru.load(std::memory_order_relaxed);
    }
    return true;
}

void ggml_backend_cuda_ar_dump_state(const char * reason) {
    ggml_cuda_ar_dump_state(reason);
}

static void ggml_cuda_wp_graph_count_init() {
    std::call_once(ggml_cuda_wp_graph_atexit_once, []() { atexit(ggml_cuda_wp_graph_print_counts); });
}

// Periodic dump, because the atexit() above is NOT reachable in this harness:
// the router SIGKILLs its model children, so atexit handlers never run and these
// counters were silently unobservable for the whole life of the process. (Logged
// in the KG after it ate a hip-graphs engagement measurement once already --
// exit-time prints are incompatible with a SIGKILL teardown.)
//
// Fires every WP_HIP_GRAPHS_LOG_EVERY graph_compute calls (default 256, 0 = off)
// so "are graphs actually replaying, or capturing/falling back every token" is
// answerable from a live process.
static void ggml_cuda_wp_graph_count_tick() {
    static const uint64_t every = [] {
        const char * e = std::getenv("WP_HIP_GRAPHS_LOG_EVERY");
        if (e == nullptr) {
            return (uint64_t) 256;
        }
        const long v = std::atol(e);
        return v >= 0 ? (uint64_t) v : (uint64_t) 256;
    }();
    if (every == 0) {
        return;
    }
    static std::atomic<uint64_t> calls{0};
    if ((calls.fetch_add(1, std::memory_order_relaxed) + 1) % every == 0) {
        ggml_cuda_wp_graph_print_counts();
    }
}

// 2026-09-07: `blocker`/`why` (both optional) report WHICH node vetoed capture.
// Without them the wp hip-graphs counter could say "64 fallbacks, 0 captures"
// forever with no way to attribute it: a graph that fails this test never
// reaches ggml_cuda_graph_update_required(), so the churn diagnostic -- the
// only other per-graph log -- can never see it.
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph,
        const ggml_tensor ** blocker = nullptr, const char ** why = nullptr) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // (TQ weight types included -- see the helper)
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (blocker) { *blocker = node; }
                if (why)     { *why = "MUL_MAT_ID needs a stream sync"; }
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
            // expert_ptrs live on the node. Replay skips the host dispatcher, so
            // take() never runs and the fail-safe (n_as set, TLS miss -> op_params)
            // never sees the node. Keep these graphs eager.
            if (ggml_mul_mat_id_get_expert_ptrs_n_as(node) > 0) {
                use_cuda_graph = false;
                if (blocker) { *blocker = node; }
                if (why)     { *why = "MUL_MAT_ID carries host expert_ptrs"; }
            }
        }
        // MAD-244: ml8 MoE dispatch downloads ids host-side to bin by expert
        // (see ggml_cuda_op_ml8_mul_mat_id in ml8.cu) — incompatible with
        // graph capture, which forbids hipStreamSynchronize. Always disable
        // capture when an ml8 MoE op is present.
        if (node->op == GGML_OP_ML8_MUL_MAT_ID) {
            use_cuda_graph = false;
            if (blocker) { *blocker = node; }
            if (why)     { *why = "ml8 MoE host-side routing"; }
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to ml8 MoE host-side routing\n", __func__);
#endif
        }

        // MAD-288 (2026-09-13): capture is disabled for any graph holding a
        // GGML_OP_PAGED_ATTN_MT node. Original reason: the op's workspace came
        // from ctx.pool() per call, so a replayed graph could alias recycled
        // pool memory. That half is fixed (mt_pagedattn_aiter.cu,
        // mt_aiter_persist_get, 2026-09-16: persistent scratch that never
        // moves). The exclusion STAYS ON because with capture enabled the
        // production TP alias wedges deterministically at 16k prefill
        // (2026-09-16 chains 22/24/25/26: both GPUs 99-100% busy, duplex AR
        // ops never complete, spin trap not fired) -- the AITER launch still
        // bakes grid dims / kernel-handle selection derived from tensor DATA
        // (context lens) into the captured node, invisible to the topology
        // compare. Cost of the exclusion, measured: -12% prefill, -20% decode
        // on qwen38-27b-q8-tp. WP_HIP_GRAPHS_PAGED_ATTN=1 re-enables capture
        // for A/B only.
        if (node->op == GGML_OP_PAGED_ATTN_MT) {
            static const bool allow_paged_attn_mt_graphs = [] {
                const char * e = std::getenv("WP_HIP_GRAPHS_PAGED_ATTN");
                return e != nullptr && e[0] == '1';
            }();
            if (!allow_paged_attn_mt_graphs) {
                use_cuda_graph = false;
                if (blocker) { *blocker = node; }
                if (why)     { *why = "PAGED_ATTN_MT: launch geometry not replay-safe (MAD-288)"; }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    if (cgraph->n_nodes == 0) {
        return cgraph;
    }
    if (!ggml_cuda_wp_hip_graphs_enabled()) {
        // Upstream default: key on the first node's address.
        return cgraph->nodes[0];
    }
    // WP_HIP_GRAPHS: the backend scheduler splits the decode forward on every
    // in-graph CPU dispatch op, then throws the split cgraph away. Keying on
    // nodes[0] (a fresh pointer) makes the capture cache a 100% miss.
    //
    // Key on the structural fingerprint; the post-lookup property check compares resolved device addresses before replay.
    // Including pointers here turns moving activation or KV views into new entries, so the recurring split is never found.
    //
    // WP_HIP_GRAPH_KEY_ADDRS=1 restores address-keyed entries for a diagnostic
    // run. Persistent worker graphs keep their existing address-keyed behavior.
    static const bool key_addrs = [] {
        if (ggml_cuda_wp_persistent_graphs_enabled()) {
            return true;
        }
        const char * e = std::getenv("WP_HIP_GRAPH_KEY_ADDRS");
        return e != nullptr && e[0] == '1';
    }();

    uint64_t h = 1469598103934665603ULL;
    h = ggml_cuda_graph_fnv1a_mix(h, (uint64_t) (unsigned) cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        h = ggml_cuda_graph_mix_tensor_topo(h, cgraph->nodes[i]);
        h = ggml_cuda_graph_mix_rcache_offset(h, cgraph->nodes[i]);
        if (key_addrs) {
            h = ggml_cuda_graph_mix_tensor_addrs(h, cgraph->nodes[i]);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                h = ggml_cuda_graph_mix_tensor_topo(h, cgraph->nodes[i]->src[j]);
                h = ggml_cuda_graph_mix_rcache_offset(h, cgraph->nodes[i]->src[j]);
                if (key_addrs) {
                    h = ggml_cuda_graph_mix_tensor_addrs(h, cgraph->nodes[i]->src[j]);
                }
            } else {
                h = ggml_cuda_graph_fnv1a_mix(h, 0);
            }
        }
    }
    return (const void *) (uintptr_t) h;
}

static bool ggml_cuda_graph_node_topo_equal(
        const ggml_cuda_graph::node_properties & a,
        const ggml_cuda_graph::node_properties & b) {
    if (!ggml_cuda_graph_tensor_topo_equal(a.node, b.node)) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (memcmp(a.node_src_ne[j], b.node_src_ne[j], sizeof(a.node_src_ne[j])) != 0) {
            return false;
        }
        if (memcmp(a.node_src_nb[j], b.node_src_nb[j], sizeof(a.node_src_nb[j])) != 0) {
            return false;
        }
    }
    return true;
}

static bool ggml_cuda_graph_node_addrs_equal(
        const ggml_cuda_graph::node_properties & a,
        const ggml_cuda_graph::node_properties & b) {
    // View/noop nodes do not launch. Their dest pointer is only relevant as
    // a later kernel's src, which is compared via node_src_data_ptrs.
    if (!ggml_cuda_graph_tensor_is_view_or_noop(&a.node) &&
        a.node.data != b.node.data) {
        return false;
    }
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (a.node_src_data_ptrs[j] != b.node_src_data_ptrs[j]) {
            return false;
        }
    }
    return true;
}

// WP_HIP_GRAPH_FAST_PROPS=1 (default off) enables this allocation-free
// unchanged-node check.
//
// The stock loop builds a fully zeroed ggml_cuda_graph::node_properties on the
// stack for EVERY node (memset ~1 KiB), memcpy's the whole ggml_tensor plus up
// to GGML_MAX_SRC ne/nb pairs into it, then memcmp's the lot. That is ~2.5 KiB
// of memory traffic per node, on a graph that is re-walked for every split of
// every token, purely to answer "did anything change?" — which in warm steady
// state is always "no".
//
// This predicate answers the same question by reading the live tensors, and
// short-circuits on the first difference. It is deliberately conservative: it
// returns true only when the node is already stored AND matches on exactly the
// fields the stock path would have compared (topology, src ne/nb, and resolved
// device addresses). When it returns true the stock path would have set neither
// `res` nor cleared `only_src_data_ptrs_changed`; the only thing skipped is
// refreshing object pointers (src[], buffer, extra, view_src, and VIEW
// op_params) inside node_props, and nothing ever reads those back.
static bool ggml_cuda_graph_props_unchanged(
        const ggml_cuda_graph::node_properties & p, const ggml_tensor * n) {
    const bool stored = p.node.op != GGML_OP_NONE ||
                        p.node.ne[0] != 0 ||
                        p.node.data != nullptr;
    if (!stored) {
        return false;
    }
    if (!ggml_cuda_graph_tensor_topo_equal(p.node, *n)) {
        return false;
    }
    if (!ggml_cuda_graph_tensor_is_view_or_noop(&p.node) && p.node.data != n->data) {
        return false;
    }
    static const int64_t zero_ne[GGML_MAX_DIMS] = {};
    static const size_t  zero_nb[GGML_MAX_DIMS] = {};
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * s = n->src[j];
        if (p.node_src_data_ptrs[j] != (s ? s->data : nullptr)) {
            return false;
        }
        const void * ne = s ? (const void *) s->ne : (const void *) zero_ne;
        const void * nb = s ? (const void *) s->nb : (const void *) zero_nb;
        if (memcmp(p.node_src_ne[j], ne, sizeof(p.node_src_ne[j])) != 0 ||
            memcmp(p.node_src_nb[j], nb, sizeof(p.node_src_nb[j])) != 0) {
            return false;
        }
    }
    return true;
}

static bool ggml_cuda_graph_fast_props_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("WP_HIP_GRAPH_FAST_PROPS");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

// graph_key MUST be the key the caller already obtained for this exact cgraph.
// It used to be recomputed here, which meant the O(n_nodes * n_src) structural
// hash ran a second time per split per token for a provably identical result.
static bool ggml_cuda_graph_update_required(
        ggml_backend_cuda_context * cuda_ctx,
        ggml_cgraph * cgraph,
        const void * graph_key,
        bool * src_data_ptrs_only) {
    bool res = false;
    bool only_src_data_ptrs_changed = true;
    if (src_data_ptrs_only != nullptr) {
        *src_data_ptrs_only = false;
    }

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    // WP_HIP_GRAPHS churn diagnostic.
    //
    // churn_log_budget (WP_HIP_GRAPHS_CHURN_BUDGET, default 400) now gates a
    // single compact line per REAL recapture only -- an entry that already
    // had node_props from a prior capture (not a first_snapshot) and for
    // which ggml_cuda_graph_update_required is about to return true because
    // of a topology or resolved-address change. Whether a given call is a
    // real recapture can only be known after the per-node comparison loop
    // below, so the budget is spent (fetch_sub) only after the loop, and
    // only on that classification -- first-snapshot captures (a brand new
    // key, or the SIZE-changed resize below) and no-change lookups never
    // touch the budget. That is the fix for the budget being exhausted by
    // startup captures before any steady-state churn could be logged.
    static std::atomic<int> churn_log_budget{[] {
        const char * e = std::getenv("WP_HIP_GRAPHS_CHURN_BUDGET");
        if (e == nullptr) { return 400; }
        const long v = std::strtol(e, nullptr, 10);
        return (v > 0 && v < 10000000) ? (int) v : 400;
    }()};
    // WP_HIP_GRAPHS_CHURN_VERBOSE=1 restores the old per-node dump (the first
    // few differing nodes, unthrottled by the budget). Off by default: the
    // one-line recap summary below is what the default run relies on.
    static const bool churn_verbose = [] {
        const char * e = std::getenv("WP_HIP_GRAPHS_CHURN_VERBOSE");
        return e != nullptr && e[0] == '1';
    }();
    const bool wp_hip_graphs = ggml_cuda_wp_hip_graphs_enabled();

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        only_src_data_ptrs_changed = false;
        graph->node_props.resize(cgraph->n_nodes);
    }

    // Only safe to skip work when the verbose per-node dump is not
    // classifying this pass (it needs the full memcmp to tell
    // object_ptr_only from addr_ptr_only for every differing node).
    const bool fast_props = !churn_verbose && ggml_cuda_graph_fast_props_enabled();

    // Diagnostic-only accounting for a real recapture on this call (see the
    // "wp hip-graphs recap:" line below and the recap_* counters in
    // ggml_cuda_wp_graph_counters). Cheap: only touched for nodes that a) are
    // not skipped by fast_props (which only skips provably-unchanged nodes)
    // and b) already had stored props and topo/addr-changed. Gated on
    // wp_hip_graphs so a non-WP_HIP_GRAPHS run pays nothing extra.
    int        recap_addr_nodes   = 0;
    int        recap_src_nodes    = 0;
    bool       recap_any_topo     = false;
    bool       recap_any_addr     = false;
    int        recap_first_idx    = -1;
    ggml_op    recap_first_op     = GGML_OP_NONE;
    const void * recap_first_old_data = nullptr;
    const void * recap_first_new_data = nullptr;
    const void * recap_first_old_src0 = nullptr;
    const void * recap_first_new_src0 = nullptr;

    int churn_nodes_logged = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (fast_props && ggml_cuda_graph_props_unchanged(graph->node_props[i], cgraph->nodes[i])) {
            continue;
        }
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        const bool stored = graph->node_props[i].node.op != GGML_OP_NONE ||
                            graph->node_props[i].node.ne[0] != 0 ||
                            graph->node_props[i].node.data != nullptr;
        if (memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            const bool topo_changed  = stored && !ggml_cuda_graph_node_topo_equal(graph->node_props[i], prop);
            const bool addrs_changed = stored && !ggml_cuda_graph_node_addrs_equal(graph->node_props[i], prop);
            if (topo_changed) {
                only_src_data_ptrs_changed = false;
            }
            // Real-recapture accounting: only nodes that already had stored
            // props AND changed topology or resolved addresses count -- a
            // first_snapshot node (stored == false) never contributes.
            if (wp_hip_graphs && stored && (topo_changed || addrs_changed)) {
                const bool node_addr_diff = graph->node_props[i].node.data != prop.node.data;
                bool node_src_diff = false;
                for (int j = 0; j < GGML_MAX_SRC; ++j) {
                    if (graph->node_props[i].node_src_data_ptrs[j] != prop.node_src_data_ptrs[j]) {
                        node_src_diff = true;
                        break;
                    }
                }
                if (node_addr_diff) { ++recap_addr_nodes; }
                if (node_src_diff)  { ++recap_src_nodes; }
                if (topo_changed)   { recap_any_topo = true; }
                if (addrs_changed)  { recap_any_addr = true; }
                if (recap_first_idx < 0) {
                    recap_first_idx     = i;
                    recap_first_op      = cgraph->nodes[i]->op;
                    recap_first_old_data = graph->node_props[i].node.data;
                    recap_first_new_data = prop.node.data;
                    recap_first_old_src0 = graph->node_props[i].node_src_data_ptrs[0];
                    recap_first_new_src0 = prop.node_src_data_ptrs[0];
                }
            }
            // WP_HIP_GRAPHS_CHURN_VERBOSE=1: log the first few differing nodes.
            if (churn_verbose && churn_nodes_logged < 6) {
                ++churn_nodes_logged;
                const char * kind = !stored ? "first_snapshot" :
                                   (topo_changed ? "TOPOLOGY(ne/nb/op)" :
                                   (addrs_changed ? "addr_ptr_only" : "object_ptr_only"));
                fprintf(stderr, "wp hip-graphs churn: node[%d] op=%s name='%s' %s\n",
                        i, ggml_op_name(cgraph->nodes[i]->op), cgraph->nodes[i]->name, kind);
            }
            graph->node_props[i] = prop;
            // Object-pointer churn (src[] / buffer / extra) is expected on
            // ephemeral split rebuilds and is not a capture miss. Only topology
            // or resolved device addresses require an update.
            if (!stored || topo_changed || addrs_changed) {
                res = true;
            }
        }
    }

    // This call was a real recapture iff some node already had stored props
    // and its topology or a resolved address changed (recap_first_idx set).
    if (wp_hip_graphs && recap_first_idx >= 0) {
        ggml_cuda_wp_graph_counters & dc = ggml_cuda_wp_graph_counts[cuda_ctx->device];
        dc.recap_total.fetch_add(1, std::memory_order_relaxed);
        // Pure topology vs addr-only classification (task 4): "pure topology"
        // is a topology change with no node-owned address movement at all;
        // "addr-only" is a resolved-address change with no topology change.
        // A recapture that is neither (both topo and addr moved) falls into
        // neither bucket -- recap_total - recap_topo - recap_addr recovers it.
        if (recap_addr_nodes == 0 && recap_any_topo) {
            dc.recap_topo.fetch_add(1, std::memory_order_relaxed);
        }
        if (recap_any_addr && !recap_any_topo) {
            dc.recap_addr.fetch_add(1, std::memory_order_relaxed);
        }

        // Always-on A/B/A flip detector (no budget cost): does the NEW data
        // pointer of the first differing node match what that same slot held
        // two recaptures ago at this key? See ggml_cuda_graph::
        // last_recap_data0 / recap_prev2_data0 (common.cuh).
        if (graph->recap_prev2_data0 != nullptr && recap_first_new_data == graph->recap_prev2_data0) {
            graph->recap_flips++;
            dc.recap_flip.fetch_add(1, std::memory_order_relaxed);
        }
        graph->recap_prev2_data0 = graph->last_recap_data0;
        graph->last_recap_data0  = recap_first_new_data;

        // Budget-gated one-line summary. Spent only here -- real recaptures --
        // never on first_snapshot captures or no-change lookups.
        // Refill: 20 recap lines per 5 s of wall clock, forever, so a long-lived
        // router child keeps sampling steady-state churn instead of going
        // silent after the first burst (and so journald's rate limit keeps
        // most of them).
        {
            static std::atomic<int64_t> recap_refill_us{0};
            const int64_t now_us = ggml_time_us();
            int64_t last = recap_refill_us.load(std::memory_order_relaxed);
            if (now_us - last > 5000000 && recap_refill_us.compare_exchange_strong(last, now_us, std::memory_order_relaxed)) {
                churn_log_budget.store(20, std::memory_order_relaxed);
            }
        }
        if (ggml_cuda_wp_hip_graphs_log_enabled() && churn_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            const char * cause = (recap_any_topo && recap_any_addr) ? "both" :
                                 (recap_any_topo ? "topo" : "addr");
            fprintf(stderr,
                    "wp hip-graphs recap: key=%016llx n_nodes=%d cause=%s topo_changed=%d addr_nodes=%d src_nodes=%d "
                    "first=[%d] op=%s name='%s' data %p->%p src0 %p->%p\n",
                    (unsigned long long) (uintptr_t) graph_key, cgraph->n_nodes, cause,
                    (int) recap_any_topo, recap_addr_nodes, recap_src_nodes,
                    recap_first_idx, ggml_op_name(recap_first_op), cgraph->nodes[recap_first_idx]->name,
                    recap_first_old_data, recap_first_new_data,
                    recap_first_old_src0, recap_first_new_src0);
            fflush(stderr);
        }
    }

    if (src_data_ptrs_only != nullptr) {
        *src_data_ptrs_only = res && only_src_data_ptrs_changed;
    }
    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (is_equal(mul_mat_id_bias_glu_ops, ops) &&
                (ggml_cuda_has_routed_expert_ptrs() ||
                 ggml_mul_mat_id_get_expert_ptrs_n_as(ffn_gate) > 0 ||
                 ggml_mul_mat_id_get_expert_ptrs_n_as(ffn_up) > 0)) {
            return false;
        }

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (is_equal(mul_mat_id_glu_ops, ops) &&
                (ggml_cuda_has_routed_expert_ptrs() ||
                 ggml_mul_mat_id_get_expert_ptrs_n_as(ffn_gate) > 0 ||
                 ggml_mul_mat_id_get_expert_ptrs_n_as(ffn_up) > 0)) {
            return false;
        }

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // RMS_NORM+MUL+FP8_QUANT_ROT cannot go through ggml_can_fuse: that helper
    // requires identical shapes, and per-row QUANT_ROT is [K+4] I8 vs MUL [K].
    std::initializer_list<enum ggml_op> rms_norm_mul_qrot_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_FP8_QUANT_ROT };
    if (is_equal(rms_norm_mul_qrot_ops, ops)) {
        if (node_idx + 3 > cgraph->n_nodes) {
            return false;
        }
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }
        const ggml_tensor * rms  = cgraph->nodes[node_idx];
        const ggml_tensor * mul  = cgraph->nodes[node_idx + 1];
        const ggml_tensor * qrot = cgraph->nodes[node_idx + 2];
        if (qrot->src[0] != mul) {
            return false;
        }
        if (mul->src[0] != rms && mul->src[1] != rms) {
            return false;
        }
        int out_nodes[] = { node_idx + 2 };
        return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    // scale + unary [+ scale], generalization of the softcap pattern above to other unary ops.
    // The fused kernel carries the scale bias through, so no bias check is needed here.
    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY
     && (ops.size() == 2 || ops.begin()[2] == GGML_OP_SCALE) && unary_ops.size() == 1) {
        const enum ggml_unary_op uop = unary_ops.begin()[0];

        if (uop != GGML_UNARY_OP_SILU && uop != GGML_UNARY_OP_SIGMOID && uop != GGML_UNARY_OP_TANH) {
            return false;
        }

        const ggml_tensor * scale  = cgraph->nodes[node_idx];
        const ggml_tensor * unary  = cgraph->nodes[node_idx+1];
        const ggml_tensor * scale2 = ops.size() == 3 ? cgraph->nodes[node_idx+2] : nullptr;
        const ggml_tensor * dst    = scale2 ? scale2 : unary;

        if (ggml_get_unary_op(unary) != uop) {
            return false;
        }

        // the fused kernel is f32-only and indexes both ends linearly
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
            unary->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            return false;
        }

        if (!ggml_is_contiguous(scale->src[0]) || !ggml_is_contiguous(dst)) {
            return false;
        }

        return true;
    }

    return false;
}

// WP_FUSE_BCAST_MUL_ADD: 0 = off, 1 = {REPEAT,MUL,ADD} only, 2 = also {MUL,ADD}. Default 2.
static int ggml_cuda_bcast_mul_add_level() {
    static const int level = []() {
        const char * e = getenv("WP_FUSE_BCAST_MUL_ADD");
        return e == nullptr ? 2 : std::atoi(e);
    }();
    return level;
}

// Fuse {REPEAT(_4D), MUL, ADD} -- and the {MUL, ADD} pair where ggml's implicit
// binary broadcast already avoids the repeat -- into a single dst = a*b + c pass.
//
// The win is memory traffic, not flops: the repeat's materialized intermediate
// and the mul's intermediate both disappear, and dst is written once instead of
// three times. The kernel keeps the multiply and the add as two separately
// rounded F32 ops in the original operand order and never contracts them into
// an FMA, so every output element is bit-identical to the unfused chain.
//
// Returns the number of *extra* nodes consumed (2 or 1), or 0 if not applicable.
static int ggml_cuda_try_fuse_bcast_mul_add(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int level = ggml_cuda_bcast_mul_add_level();
    if (level <= 0) {
        return 0;
    }

    const bool with_repeat = cgraph->nodes[i]->op == GGML_OP_REPEAT;

    if (with_repeat) {
        if (!ggml_can_fuse(cgraph, i, { GGML_OP_REPEAT, GGML_OP_MUL, GGML_OP_ADD })) {
            return 0;
        }
    } else {
        if (level < 2 || !ggml_can_fuse(cgraph, i, { GGML_OP_MUL, GGML_OP_ADD })) {
            return 0;
        }
    }

    // ggml_can_fuse already guarantees: ops match, the intermediates have exactly
    // one use and are neither views nor graph outputs, each node consumes the
    // previous one, and all three nodes have the same shape.
    ggml_tensor * repeat = with_repeat ? cgraph->nodes[i] : nullptr;
    ggml_tensor * mul    = cgraph->nodes[i + (with_repeat ? 1 : 0)];
    ggml_tensor * add    = cgraph->nodes[i + (with_repeat ? 2 : 1)];

    // F32 only. This is what makes the fusion value-preserving: the intermediate
    // that the unfused chain round-tripped through memory was already a float, so
    // eliding the store/load pair drops no rounding step. With an F16 dst the
    // unfused mul would have rounded to half first -- not bit-exact, so bail.
    if (mul->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
        return 0;
    }
    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }
    if (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }
    if (with_repeat && (repeat->type != GGML_TYPE_F32 || repeat->src[0]->type != GGML_TYPE_F32)) {
        return 0;
    }

    // resolve the three real operands, replacing the repeat by its source
    const ggml_tensor * a = mul->src[0];
    const ggml_tensor * b = mul->src[1];

    if (with_repeat) {
        if (a == repeat) {
            a = repeat->src[0];
        }
        if (b == repeat) {
            b = repeat->src[0];
        }
        if (a == repeat || b == repeat) {
            return 0;
        }
    }

    const bool mul_first = add->src[0] == mul;
    if (!mul_first && add->src[1] != mul) {
        return 0;
    }

    const ggml_tensor * c = mul_first ? add->src[1] : add->src[0];

    // no operand may be one of the elided intermediates (e.g. add(m, m))
    if (a == mul || b == mul || c == mul || (with_repeat && c == repeat)) {
        return 0;
    }

    // The kernel indexes every operand with a plain modulo over its own extents.
    // That is exactly ggml_repeat's tiling and ggml's implicit binary broadcast,
    // but it needs the divisibility that ggml_can_repeat asserts, and it assumes
    // contiguous operands so the strides are the canonical ones.
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(c)) {
        return 0;
    }
    if (!ggml_can_repeat(a, add) || !ggml_can_repeat(b, add) || !ggml_can_repeat(c, add)) {
        return 0;
    }

    // index space must fit the kernel's uint32 flat index / strides
    if (ggml_nelements(add) <= 0 || ggml_nelements(add) > (int64_t) std::numeric_limits<int32_t>::max()) {
        return 0;
    }

    // Aliasing: an operand that shares dst's shape, layout and base pointer is
    // read and written at the identical index by the same thread, which is safe
    // in place (this is the common residual-add case). Any other overlap with dst
    // could be clobbered by one thread before another thread reads it.
    auto overlaps_dst = [&](const ggml_tensor * x) {
        const char * xs = (const char *) x->data;
        const char * xe = xs + ggml_nbytes(x);
        const char * ds = (const char *) add->data;
        const char * de = ds + ggml_nbytes(add);
        return xs < de && ds < xe;
    };
    auto operand_ok = [&](const ggml_tensor * x) {
        return !overlaps_dst(x) || (x->data == add->data && ggml_are_same_shape(x, add));
    };

    if (!operand_ok(a) || !operand_ok(b) || !operand_ok(c)) {
        return 0;
    }

    ggml_cuda_op_fused_bcast_mul_add(*cuda_ctx, a, b, c, add, mul_first);

    return with_repeat ? 2 : 1;
}

// QUANT_ROT nodes fused early (RMS+MUL consumer is not the next graph node).
// Cleared at the start of each host-side graph walk. thread_local: two GPUs
// can evaluate on two host threads.
static thread_local std::unordered_set<const ggml_tensor *> g_fused_qrot_skip;

static ggml_tensor * ggml_cuda_find_qrot_consumer(const ggml_cgraph * cgraph, int mul_idx) {
    if (!ggml_node_has_n_uses(cgraph, mul_idx, 1)) {
        return nullptr;
    }
    const ggml_tensor * mul = cgraph->nodes[mul_idx];
    for (int j = mul_idx + 1; j < cgraph->n_nodes; ++j) {
        ggml_tensor * cand = cgraph->nodes[j];
        if (cand->op == GGML_OP_FP8_QUANT_ROT && cand->src[0] == mul) {
            return cand;
        }
    }
    return nullptr;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    // escape hatch for the scale + unary [+ scale] fusion only, for A/B without a rebuild
    static const bool disable_scale_unary_fusion =
        getenv("GGML_CUDA_DISABLE_SCALE_UNARY_FUSION") != nullptr &&
        std::atoi(getenv("GGML_CUDA_DISABLE_SCALE_UNARY_FUSION"));

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    // ml8: rotation → mul_mat (G.6.d). The FWHT + H_a^T fold into the GEMM's
    // activation-quantize prologue, eliding the rotation node entirely.
    if (node->op == GGML_OP_ML8_APPLY_ROTATION &&
        ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ML8_APPLY_ROTATION, GGML_OP_ML8_MUL_MAT }, { i + 1 }) &&
        ggml_cuda_ml8_can_fuse_rot_mm(cgraph->nodes[i], cgraph->nodes[i + 1])) {
        ggml_cuda_op_ml8_mul_mat_fused(*cuda_ctx, cgraph->nodes[i], cgraph->nodes[i + 1]);
        return 1;
    }

    // ml8: FFN {gate mul_mat, up mul_mat, swiglu} -> one fused GEMM whose
    // epilogue computes silu(gate)*up and stores it once (MAD-305 fused-FFN
    // task). Only the pattern shape is matched here (both node[i]/node[i+1]
    // being ML8_MUL_MAT feeding node[i+2]'s GLU, with no other consumer of
    // either mul_mat's output -- ggml_can_fuse_subgraph enforces that);
    // every actual eligibility check (same shared x, same N/K, RDNA4_TRFEED
    // layout, M>32 prefill-only, SWIGLU not swapped, matching lut_group_off,
    // etc.) lives in ggml_cuda_ml8_can_fuse_ffn_swiglu (ml8.cu) so it stays
    // in one place. glu->src[0] is always the SiLU'd operand and src[1] the
    // multiplied one (ggml_swiglu(a,b) never swaps them), but the GRAPH may
    // emit the gate/up ML8_MUL_MAT nodes in either index order, so try both
    // assignments of {node[i], node[i+1]} to {gate, up} rather than assuming
    // node[i] is gate.
    if (node->op == GGML_OP_ML8_MUL_MAT && i + 2 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_ML8_MUL_MAT &&
        cgraph->nodes[i + 2]->op == GGML_OP_GLU &&
        ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ML8_MUL_MAT, GGML_OP_ML8_MUL_MAT, GGML_OP_GLU }, { i + 2 })) {
        ggml_tensor * mm0 = cgraph->nodes[i];
        ggml_tensor * mm1 = cgraph->nodes[i + 1];
        ggml_tensor * glu = cgraph->nodes[i + 2];
        ggml_tensor * mm_gate = (glu->src[0] == mm0) ? mm0 : mm1;
        ggml_tensor * mm_up   = (glu->src[0] == mm0) ? mm1 : mm0;
        if (ggml_cuda_ml8_can_fuse_ffn_swiglu(mm_gate, mm_up, glu)) {
            ggml_cuda_op_ml8_ffn_gate_up_swiglu(*cuda_ctx, mm_gate, mm_up, glu);
            return 2;
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // broadcast multiply-add: {REPEAT(_4D), MUL, ADD} and {MUL, ADD}
    if (node->op == GGML_OP_REPEAT || node->op == GGML_OP_MUL) {
        const int n_skip = ggml_cuda_try_fuse_bcast_mul_add(cuda_ctx, cgraph, i);
        if (n_skip > 0) {
            return n_skip;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    // MAD-88 (A2): the fused-mmvq paths below bundle gate + up into a
    // single mmvq call with fusion.gate set. When routing is active, the
    // gate and up consolidated parents have placeholder data; honoring
    // routing for both vx AND vgate would require a parallel expert_ptrs
    // array for the gate weight. Out of scope for tonight — let the op
    // fall through to the unfused dispatcher path which already routes
    // correctly via Phase A2.
    const bool routing_active_for_fusion = ggml_cuda_has_routed_expert_ptrs() ||
        (node->op == GGML_OP_MUL_MAT_ID && ggml_mul_mat_id_get_expert_ptrs_n_as(node) > 0);

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        if (routing_active_for_fusion && op == GGML_OP_MUL_MAT_ID) continue;
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        if (routing_active_for_fusion && op == GGML_OP_MUL_MAT_ID) continue;
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_FP8_QUANT_ROT }, {}) &&
        ggml_cuda_op_fp8_quant_rot_fused_norm(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        return 2;
    }

    // qwen35 expands attn_norm (RMS+MUL) before QKV creates QUANT_ROT, so the
    // 3-op pattern is often not consecutive. MUL still has a single QUANT consumer.
    if (ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        ggml_tensor * qrot = ggml_cuda_find_qrot_consumer(cgraph, i + 1);
        const bool already_tried_consecutive = (i + 2 < cgraph->n_nodes && qrot == cgraph->nodes[i + 2]);
        if (qrot && !already_tried_consecutive &&
            ggml_cuda_op_fp8_quant_rot_fused_norm(*cuda_ctx, node, cgraph->nodes[i + 1], qrot)) {
            g_fused_qrot_skip.insert(qrot);
            return 1;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // scale + unary [+ scale] for the remaining unary ops (softcap above keeps the tanh 3-node case).
    // GGML_CUDA_DISABLE_SCALE_UNARY_FUSION=1 turns this off without a rebuild.
    if (node->op == GGML_OP_SCALE && !disable_scale_unary_fusion) {
        for (ggml_unary_op uop : { GGML_UNARY_OP_SILU, GGML_UNARY_OP_SIGMOID, GGML_UNARY_OP_TANH }) {
            if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { uop })) {
                ggml_cuda_op_scale_unary_scale(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
                return 2;
            }

            if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY }, { uop })) {
                ggml_cuda_op_scale_unary_scale(*cuda_ctx, node, cgraph->nodes[i + 1], nullptr);
                return 1;
            }
        }
    }

    return 0;
}

#ifdef USE_CUDA_GRAPH
// vram-budget: per-device cap on ggml_cuda_wp_graph_counters::exec_bytes_live
// (the sum of measured hipGraphInstantiate deltas for execs currently alive
// on this device). MB, read once via getenv (called from the hot instantiate
// path -- see ggml_cuda_wp_hip_graphs_enabled() above for why a live getenv()
// per call is unacceptable here). 0 disables the budget (pre-existing
// unbounded behavior). Default 512 MiB.
static size_t ggml_cuda_graph_vram_budget_bytes() {
    static const size_t budget_bytes = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_VRAM_BUDGET_MB");
        long mb = 512;
        if (e != nullptr) {
            mb = atol(e);
            if (mb < 0) {
                mb = 512;
            }
        }
        return (size_t) mb * 1024ull * 1024ull;
    }();
    return budget_bytes;
}

// vram-budget: called right after a successful hipGraphInstantiate for
// `keep_key`'s graph. While this device's exec_bytes_live total exceeds the
// configured budget and the cache holds more than just the entry we just
// captured/are about to replay, evict THIS context's single LRU entry
// through the existing retire_fn path (ggml_backend_cuda_context::
// ggml_cuda_graph_retire(), common.cuh) -- the same deferred-destroy
// machinery TTL/cap eviction already use, so there is no second erase path
// and no extra synchronization beyond what retire already does. Reusing
// ggml_cuda_graph_cache_evict_lru() with cap == current size makes it evict
// exactly one entry per call: the loop's `size() >= cap` is true once, before
// the eviction, and false immediately after size drops by one.
static void ggml_cuda_graph_enforce_vram_budget(ggml_backend_cuda_context * cuda_ctx, const void * keep_key) {
    const size_t budget_bytes = ggml_cuda_graph_vram_budget_bytes();
    if (budget_bytes == 0) {
        return; // 0 = disabled
    }
    auto & counts = ggml_cuda_wp_graph_counts[cuda_ctx->device];
    auto retire_fn = [cuda_ctx](std::unique_ptr<ggml_cuda_graph> g) { cuda_ctx->ggml_cuda_graph_retire(std::move(g)); };
    while (counts.exec_bytes_live.load(std::memory_order_relaxed) > budget_bytes &&
           cuda_ctx->cuda_graphs.size() > 1) {
        const size_t n_evicted = ggml_cuda_graph_cache_evict_lru(cuda_ctx->cuda_graphs, cuda_ctx->cuda_graphs.size(), keep_key, retire_fn);
        if (n_evicted == 0) {
            break; // nothing left to evict besides keep_key
        }
        counts.budget_evicted.fetch_add(1, std::memory_order_relaxed);
    }
}
#endif // USE_CUDA_GRAPH

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;
            g_fused_qrot_skip.clear();

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (g_fused_qrot_skip.count(node)) {
                    prev_i = i;
                    continue;
                }
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    if (wp_node_trace_enabled()) {
                        // Log skipped nodes too (no event) so the ring shows the full node sequence.
                        wp_node_trace_record(cuda_ctx->device, cuda_ctx->stream(), node, i, /*in_capture=*/true);
                    }
                    continue;
                }

                const bool wp_prof_capture = use_cuda_graph && cuda_graph_update_required;
                wp_op_profile_begin_node(cuda_ctx->device, cuda_ctx->stream(), wp_prof_capture);

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    wp_op_profile_end_node(cuda_ctx->device, cuda_ctx->stream(), node, nodes_to_skip + 1, wp_prof_capture);
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                // WP_HOST_STALL_LOG=<us>: report any node whose HOST-side launch
                // (not GPU time) exceeds the threshold -- a launch that spins in
                // the runtime shows up here with the op that triggered it.
                static const int64_t wp_host_stall_us = [] {
                    const char * e = getenv("WP_HOST_STALL_LOG");
                    return e ? (int64_t) atoll(e) : (int64_t) 0;
                }();
                const int64_t wp_hs_t0 = wp_host_stall_us > 0 ? ggml_time_us() : 0;
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                if (wp_host_stall_us > 0) {
                    const int64_t dt = ggml_time_us() - wp_hs_t0;
                    if (dt > wp_host_stall_us) {
                        fprintf(stderr, "wp host-stall dev=%d %lld us op=%s name=%s ne=[%lld,%lld,%lld] src0=%s\n",
                                cuda_ctx->device, (long long) dt, ggml_op_name(node->op), node->name,
                                (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2],
                                node->src[0] ? ggml_op_name(node->src[0]->op) : "-");
                    }
                }
                wp_op_profile_end_node(cuda_ctx->device, cuda_ctx->stream(), node, 1, wp_prof_capture);

                if (wp_node_trace_enabled()) {
                    // In capture iff this pass is being captured into a
                    // CUDA/HIP graph -- see ggml_backend_cuda_graph_compute()
                    // (cudaStreamBeginCapture is called there exactly when
                    // use_cuda_graph && cuda_graph_update_required, right
                    // before this function runs the same node loop).
                    const bool in_capture = use_cuda_graph && cuda_graph_update_required;
                    wp_node_trace_record(cuda_ctx->device, cuda_ctx->stream(), node, i, in_capture);
                }

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            // EndCapture first. Destroying the live graph/exec while the
            // stream is still capturing, then hipGraphExecUpdate on the stale
            // instance, is the 2026-08-20 s0 SIGSEGV (SEGV_MAPERR inside
            // hipGraphExecUpdate <- compute_batch). HIP's GraphExec is not
            // independent of the Graph it was instantiated from.
            cudaGraph_t captured = nullptr;
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &captured));
#if defined(GGML_USE_HIP)
            if (graph->instance != nullptr) {
                // vram-budget: this destroy bypasses ~ggml_cuda_graph; measure it here.
                size_t wp_fb = 0, wp_fa = 0, wp_tot = 0;
                (void) cudaMemGetInfo(&wp_fb, &wp_tot);
                CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
                graph->instance = nullptr;
                (void) cudaMemGetInfo(&wp_fa, &wp_tot);
                if (ggml_cuda_wp_hip_graphs_log_enabled()) {
                    fprintf(stderr, GGML_CUDA_NAME " vram-budget: graph_exec_destroy(recapture) freed %.1f MiB on device %d (free after %.1f MiB)\n",
                            (wp_fa > wp_fb ? wp_fa - wp_fb : 0) / 1048576.0, cuda_ctx->device, wp_fa / 1048576.0);
                    fflush(stderr);
                }
                // vram-budget: this exec is gone -- return its tracked share of
                // exec_bytes_live now, and zero exec_bytes so ~ggml_cuda_graph()
                // (which will run later, whenever this cache entry is eventually
                // evicted/replaced) does not subtract it a second time.
                if (graph->exec_bytes != 0) {
                    ggml_cuda_graph_wp_exec_bytes_sub(cuda_ctx->device, graph->exec_bytes);
                    graph->exec_bytes = 0;
                }
            }
#endif
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
            }
            graph->graph = captured;
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            if (ggml_cuda_wp_hip_graphs_enabled()) {
                ggml_cuda_wp_graph_count_init();
                auto & c = ggml_cuda_wp_graph_counts[cuda_ctx->device];
                c.captures.fetch_add(1, std::memory_order_relaxed);
                if (wp_alloc_log_enabled()) {
                    int64_t max_ne1_2d = 0; int n_mm = 0, n_mmid = 0; const ggml_tensor * widest = nullptr;
                    int64_t max_ne1_2d_op = 0; const ggml_tensor * widest_op = nullptr;
                    for (int i = 0; i < cgraph->n_nodes; ++i) {
                        const ggml_tensor * t = cgraph->nodes[i];
                        if (t->op == GGML_OP_MUL_MAT) { ++n_mm; }
                        if (t->op == GGML_OP_MUL_MAT_ID) { ++n_mmid; }
                        if (t->ne[2] == 1 && t->ne[3] == 1 && t->ne[1] > max_ne1_2d) { max_ne1_2d = t->ne[1]; widest = t; }
                        if (!ggml_cuda_is_view_or_noop(t) && t->ne[2] == 1 && t->ne[3] == 1 && t->ne[1] > max_ne1_2d_op) { max_ne1_2d_op = t->ne[1]; widest_op = t; }
                    }
                    wp_alloc_log("graph_capture", cuda_ctx->device, (size_t) cgraph->n_nodes, (size_t) max_ne1_2d);
                    fprintf(stderr, "wp alloc-log   capture detail: n_nodes=%d n_mul_mat=%d n_mul_mat_id=%d max_ne1_2d=%lld op=%s name=%s | widest non-view: %lld op=%s name=%s ne0=%lld\n",
                            cgraph->n_nodes, n_mm, n_mmid, (long long) max_ne1_2d,
                            widest ? ggml_op_name(widest->op) : "-", widest ? widest->name : "-",
                            (long long) max_ne1_2d_op, widest_op ? ggml_op_name(widest_op->op) : "-",
                            widest_op ? widest_op->name : "-", widest_op ? (long long) widest_op->ne[0] : 0);
                }
                // Consume the reason: it is assigned on INSERT and must be
                // counted exactly once, by the capture that the insert caused.
                const auto reason = graph->capture_reason;
                graph->capture_reason = ggml_cuda_graph::CAPTURE_OTHER;
                switch (reason) {
                    case ggml_cuda_graph::CAPTURE_LRU:
                        c.cap_lru.fetch_add(1, std::memory_order_relaxed); break;
                    case ggml_cuda_graph::CAPTURE_TTL:
                        c.cap_ttl.fetch_add(1, std::memory_order_relaxed); break;
                    case ggml_cuda_graph::CAPTURE_OTHER:
                        c.cap_recapture.fetch_add(1, std::memory_order_relaxed); break;
                    default:
                        c.cap_newkey.fetch_add(1, std::memory_order_relaxed); break;
                }
                // vram-budget: live_graphs is no longer set here. Two contexts
                // (e.g. main model + DFlash draft) can share device 0, and this
                // per-context cuda_graph_count() used to clobber whatever the
                // other context's capture had just stored. It is now true
                // per-device accounting: fetch_add on cache insert
                // (ggml_backend_cuda_context::cuda_graph(), common.cuh) and
                // fetch_sub on every erase (ggml_cuda_graph_retire(), common.cuh).
            }

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            // vram-budget: hipGraphInstantiate consumes device memory outside ggml's
            // pools (kernel-arg buffers); measure every instantiate unconditionally.
            size_t wp_free_before = 0, wp_total_dummy = 0;
            CUDA_CHECK(cudaMemGetInfo(&wp_free_before, &wp_total_dummy));
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
            {
                size_t wp_free_after = 0;
                CUDA_CHECK(cudaMemGetInfo(&wp_free_after, &wp_total_dummy));
                const size_t wp_used = wp_free_before > wp_free_after ? wp_free_before - wp_free_after : 0;
                // journald rate-limits the router child (10000 msgs / 30 s): a
                // two-slot rep instantiates ~150k graphs and floods everything
                // else out. Print the first 300 only; exec_mb in the stats line
                // carries the running total.
                static std::atomic<int> wp_inst_print_budget{300};
                if (ggml_cuda_wp_hip_graphs_log_enabled() && wp_used > 0 &&
                    wp_inst_print_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    fprintf(stderr, GGML_CUDA_NAME " vram-budget: graph_instantiate %.1f MiB on device %d (free after %.1f MiB of %.1f MiB, n_nodes=%d)\n",
                            wp_used / 1048576.0, cuda_ctx->device, wp_free_after / 1048576.0, wp_total_dummy / 1048576.0, (int) cgraph->n_nodes);
                    fflush(stderr);
                }
                if (wp_alloc_log_enabled()) {
                    wp_alloc_log("graph_instantiate", cuda_ctx->device, wp_used, wp_free_after);
                }

                // vram-budget: record this exec's measured cost and enforce the
                // per-device budget (GGML_CUDA_GRAPH_VRAM_BUDGET_MB) by evicting
                // this context's coldest cache entries, if configured. graph is
                // the entry we just instantiated -- never evict it here.
                // The delta is a free-memory difference around the instantiate
                // call, so an unrelated pool/device allocation on another
                // thread that lands inside the window gets attributed here
                // (seen: 94 MiB on gfx1030, where an exec otherwise costs 0).
                // Real exec costs measured so far are 2.0 MiB (gfx1201); cap
                // the charge so a coincidence cannot poison the budget.
                const size_t wp_exec_charge_cap = 32u * 1024u * 1024u;
                size_t wp_charge = wp_used;
                if (wp_charge > wp_exec_charge_cap) {
                    if (ggml_cuda_wp_hip_graphs_log_enabled()) {
                        fprintf(stderr, GGML_CUDA_NAME " vram-budget: graph_instantiate delta %.1f MiB on device %d exceeds exec charge cap; not charged to graph budget\n",
                                wp_used / 1048576.0, cuda_ctx->device);
                    }
                    wp_charge = 0;
                }
                graph->exec_bytes = wp_charge;
                if (wp_charge != 0) {
                    ggml_cuda_wp_graph_counts[cuda_ctx->device].exec_bytes_live.fetch_add(wp_charge, std::memory_order_relaxed);
                }
                ggml_cuda_graph_enforce_vram_budget(cuda_ctx, graph_key);
            }
        }
        // CUDA: ExecUpdate patches kernel params in the existing exec.
        // HIP: never. hipGraphExecUpdate SIGSEGV'd on this path (s0 2026-08-20);
        // recapture already replaced graph->graph and dropped the exec above,
        // so Instantiate is the whole update.
#if !defined(GGML_USE_HIP)
        if (cuda_graph_update_required) {
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
#endif
        // Launch graph
        wp_op_profile_begin_replay(cuda_ctx->device, cuda_ctx->stream());
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        wp_op_profile_end_replay(cuda_ctx->device, cuda_ctx->stream());
        // Record right after launch (async, no host sync) so a later TTL/LRU
        // eviction of THIS graph can tell whether this replay has finished
        // before it retires/frees the graph -- see ggml_cuda_graph_retire()/
        // drain_retired() in common.cuh and the race they close.
        if (graph->last_launch_event == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&graph->last_launch_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(graph->last_launch_event, cuda_ctx->stream()));
        if (ggml_cuda_wp_hip_graphs_enabled() && !cuda_graph_update_required) {
            ggml_cuda_wp_graph_count_init();
            auto & c = ggml_cuda_wp_graph_counts[cuda_ctx->device];
            c.replays.fetch_add(1, std::memory_order_relaxed);
            c.retired.store(cuda_ctx->retired_graphs.size(), std::memory_order_relaxed);
            c.retired_freed.store(cuda_ctx->retired_freed, std::memory_order_relaxed);
            c.retired_synced.store(cuda_ctx->retired_synced, std::memory_order_relaxed);
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
// 2026-09-06: PAGED_ATTN_MT's decode branch (mt_pagedattn.cu,
// ggml_cuda_op_paged_attn_mt) allocates its partials scratch from
// ctx.pool(), and ggml_cuda_pool_leg::alloc() does a raw cudaMalloc on a
// cache miss -- illegal while the stream is mid HIP/CUDA-graph capture.
// That's normally safe because the same bucketed size is already warm in
// the pool from earlier eager calls before a key is ever captured -- but
// under WP_HIP_GRAPHS the FIRST visit of a graph key is captured directly
// (see the warmup_complete handling below), with no prior eager pass. If
// that first visit is also the very first time this process has ever
// evaluated this exact decode shape (e.g. a brand-new turbo4/draft-MTP
// shape hit at server startup), the pool has nothing cached for that
// bucket and the in-capture alloc is a guaranteed miss, which poisons the
// capture ("operation failed due to a previous error during capture").
// Used to force one eager warm-up visit (below) for graphs containing this
// op, same as the pre-WP two-visit warmup, without touching the WP fast
// path for every other op.
static bool ggml_cuda_graph_has_paged_attn_mt(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op == GGML_OP_PAGED_ATTN_MT) {
            return true;
        }
    }
    return false;
}

// 2026-09-07 (MAD-LAB): attribute an eager fallback.
//
// The wp hip-graphs counter reports "interval(hits=192 captures=0 fallbacks=64)"
// -- a permanent, per-decode-step eager split -- with no way to tell WHICH split
// or WHY. The churn diagnostic cannot help: a graph vetoed by
// ggml_cuda_graph_check_compability() / is_prefill_shaped() never reaches
// ggml_cuda_graph_update_required(), which is where churn logging lives. So log
// it here instead, once per distinct (key, reason), bounded by
// WP_HIP_GRAPHS_FALLBACK_BUDGET (default 32; 0 disables).
static void ggml_cuda_wp_graph_log_fallback(
        const void * graph_key, const ggml_cgraph * cgraph,
        const ggml_tensor * blocker, const char * reason,
        const ggml_cuda_graph * graph) {
    if (!ggml_cuda_wp_hip_graphs_log_enabled()) {
        return;
    }
    static const int budget_max = [] {
        const char * e = std::getenv("WP_HIP_GRAPHS_FALLBACK_BUDGET");
        if (e == nullptr) { return 32; }
        const long v = std::strtol(e, nullptr, 10);
        return (v >= 0 && v < 1000000) ? (int) v : 32;
    }();
    static std::atomic<int> budget{budget_max};

    if (budget.load(std::memory_order_relaxed) <= 0) {
        return;
    }

    // Only the FIRST fallback of each distinct key is interesting: a permanent
    // fallback repeats the same reason every step and would drown the journal.
    static std::mutex               seen_mutex;
    static std::set<const void *>   seen;
    {
        std::lock_guard<std::mutex> lock(seen_mutex);
        if (!seen.insert(graph_key).second) {
            return;
        }
    }
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    if (reason == nullptr) {
        // Compatible graph whose capture was declined for a non-structural
        // reason (upstream warmup pass, properties still churning).
        reason = "capture declined (warmup / properties churning)";
    }
    GGML_UNUSED(graph);

    if (blocker != nullptr) {
        fprintf(stderr,
                "wp hip-graphs fallback: key=%p n_nodes=%d reason='%s' "
                "node op=%s name='%s' ne=[%lld,%lld,%lld,%lld]\n",
                graph_key, cgraph->n_nodes, reason,
                ggml_op_name(blocker->op), blocker->name,
                (long long) blocker->ne[0], (long long) blocker->ne[1],
                (long long) blocker->ne[2], (long long) blocker->ne[3]);
    } else {
        fprintf(stderr, "wp hip-graphs fallback: key=%p n_nodes=%d reason='%s'\n",
                graph_key, cgraph->n_nodes, reason);
    }
}

static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA &&
                !ggml_cuda_wp_persistent_graphs_enabled()) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
#if defined(GGML_USE_HIP)
    // Complete any deferred multi-input cross-device stages before kernels run.
    hip_xdev_batch_flush();
#endif
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    wp_op_profile_begin_graph(cuda_ctx->device, cgraph);
    if (wp_node_trace_enabled() && cgraph->n_nodes > 0) {
        // Marker entry: an event recorded on the compute stream BEFORE any node of
        // this graph, so the ring can tell "stream never reached this graph" from
        // "stream is stuck inside it".
        wp_node_trace_record(cuda_ctx->device, cuda_ctx->stream(), cgraph->nodes[0], -1, false);
    }

    // See ggml_cuda_wp_graph_count_tick(): the atexit dump never runs under this
    // harness's SIGKILL teardown, so the counters are reported periodically from
    // here (once per backend graph_compute call) instead.
    if (ggml_cuda_wp_hip_graphs_enabled()) {
        ggml_cuda_wp_graph_count_tick();
    }

    if (cuda_ctx->wp_copy_enabled && cuda_ctx->wp_copy_pending &&
        cudaStreamWaitEvent(cuda_ctx->stream(), cuda_ctx->wp_copy_latest_event, 0) != cudaSuccess) {
        ggml_cuda_wp_copy_disarm(cuda_ctx);
    }
    cuda_ctx->wp_copy_pending = false;

    // MAD-230: publish the compute stream to the weight-pager eval_cb
    // side channel so its hipMemcpyAsync(stream) for the expert-pointer
    // array is correctly stream-ordered with the MMQ kernels that read
    // it. The eval_cb fires before each op compute, so the stream must
    // be set before we enter the per-op loop below.
    ggml_cuda_set_wp_compute_stream(cuda_ctx->device, (void *) cuda_ctx->stream());

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        // HIP graph capture is per-thread/per-stream. The worker's one-node
        // keepalive graph is built before its device executor exists, so keep
        // it eager instead of replaying it from a different thread.
        const ggml_tensor * fb_node   = nullptr;
        const char        * fb_reason = nullptr;
        const bool node_compatible = ggml_cuda_graph_check_compability(cgraph, &fb_node, &fb_reason);
        if (node_compatible) {
            if (const ggml_tensor * pf = ggml_cuda_graph_prefill_shaped_node(cgraph)) {
                fb_node   = pf;
                fb_reason = "prefill-shaped node (token width > 32)";
            } else if (ggml_cuda_wp_hip_graphs_enabled() && cgraph->n_nodes < 2) {
                fb_node   = cgraph->n_nodes > 0 ? cgraph->nodes[0] : nullptr;
                fb_reason = "fragment has < 2 nodes";
            }
        }
        const bool graph_compatible = node_compatible && fb_reason == nullptr;
        if (graph_compatible) {
            bool properties_src_data_ptrs_only = false;
            const bool properties_changed = ggml_cuda_graph_update_required(
                cuda_ctx, cgraph, graph_key, &properties_src_data_ptrs_only);

            if (ggml_cuda_wp_hip_graphs_enabled()) {
                // Key is (topo, device addrs). First visit of that identity is
                // enough to capture — the 2-call warmup existed because the
                // key was an ephemeral nodes[0] pointer that never repeated.
                // Do NOT take the src-ptrs-only ExecUpdate path: that is the
                // crash (hipGraphExecUpdate SEGV_MAPERR) and the decode
                // regression (recapture+Update every expert instead of replay).
                //
                // Exception: a graph containing PAGED_ATTN_MT still needs one
                // eager visit before its first capture, to warm the decode
                // partials pool bucket outside of capture (see the comment on
                // ggml_cuda_graph_has_paged_attn_mt above). Scoped to graphs
                // carrying that op so every other op keeps the first-visit
                // capture fast path.
                if (!graph->warmup_complete && ggml_cuda_graph_has_paged_attn_mt(cgraph)) {
                    graph->warmup_complete = true; // next visit captures
                    use_cuda_graph = false;
                    fb_reason      = "PAGED_ATTN_MT warm-up (one eager visit, then captures)";
                } else {
                    graph->warmup_complete = true;
                    use_cuda_graph = true;
                    cuda_graph_update_required =
                        graph->instance == nullptr || properties_changed;
                }
            } else if (!graph->warmup_complete) {
                // Warmup: the first visit of a key always looks like a size
                // change (empty -> N). The second visit of the SAME structural
                // key is enough to arm capture — including the case where only
                // resolved device addresses moved. Requiring a bitwise-identical
                // ggml_tensor (the old rule) can never complete on scheduler
                // split subgraphs, because src[] object pointers are rebuilt
                // every compute.
                const bool stable_enough = !properties_changed || properties_src_data_ptrs_only;
                if (stable_enough && graph->node_props.size() == (size_t) cgraph->n_nodes) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
        if (!use_cuda_graph && ggml_cuda_wp_hip_graphs_enabled()) {
            ggml_cuda_wp_graph_count_init();
            ggml_cuda_wp_graph_counts[cuda_ctx->device].fallbacks.fetch_add(1, std::memory_order_relaxed);
            ggml_cuda_wp_graph_log_fallback(graph_key, cgraph, fb_node, fb_reason, graph);
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    try {
        ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    } catch (const ggml_cuda_pool_oom & e) {
        // MAD-LAB: a device allocation was refused by the VRAM reserve mid-graph.
        // Unwind an in-progress stream capture (discard the partial graph) and the
        // capture lock the begin above took, then fail this compute so llama_decode
        // returns an error and the server fails the request instead of the process
        // dying or the driver evicting it.
        if (use_cuda_graph && cuda_graph_update_required) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(cuda_ctx->stream(), &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone) {
                cudaGraph_t discarded = nullptr;
                (void)cudaStreamEndCapture(cuda_ctx->stream(), &discarded);
                if (discarded != nullptr) {
                    (void)cudaGraphDestroy(discarded);
                }
            }
            (void)cudaGetLastError();
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        }
        GGML_LOG_ERROR("%s: device %d: allocation of %.1f MiB refused (free was %.1f MiB) -- returning GGML_STATUS_ALLOC_FAILED\n",
                       __func__, e.device, e.requested / 1048576.0, e.free_before / 1048576.0);
        return GGML_STATUS_ALLOC_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        ggml_cuda_ar_wait_logged(cuda_ctx->stream(), (cudaEvent_t)event->context, "backend_event_wait", 0, cuda_ctx->device);
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    // upstream: record allocation dependencies for the fused MoE weighted reduction
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    // Bail BEFORE hashing. ggml_cuda_graph_get_key() is an O(n_nodes * n_src)
    // structural hash of the whole subgraph, and the scheduler calls
    // graph_optimize once per split per token (ggml-backend.cpp:1587). With
    // GGML_CUDA_GRAPH_OPT unset (the default) every byte of that hash was
    // thrown away. ggml_cuda_graph_set_enabled() is redundant here too: it only
    // latches disable_due_to_gpu_arch, and graph_compute() calls it on the same
    // key immediately afterwards.
    if (!enable_graph_optimization) {
        return;
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    std::mutex device_mutex;
    int active_count = 0;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
};

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static void ggml_backend_cuda_device_active_count_inc(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    std::lock_guard<std::mutex> lock(dev_ctx->device_mutex);
    dev_ctx->active_count++;
}

static void ggml_backend_cuda_device_active_count_dec(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    std::lock_guard<std::mutex> lock(dev_ctx->device_mutex);
    dev_ctx->active_count--;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    std::lock_guard<std::mutex> lock(ctx->device_mutex);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // If no backends or buffers are active, the cudaMemGetInfo call above lazily created a CUDA
    // context that permanently consumes VRAM. Reset the device to free it.
    if (ctx->active_count == 0) {
        CUDA_CHECK(cudaDeviceReset());
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_TQ4_1S:
                    case GGML_TYPE_TQ3_1S:
                        return true;
                    // FP8_B128 phase 2: no dedicated GEMM kernel yet (that lands
                    // with the HIP AITER preshuffle work) -- covered here only so
                    // the generic dequant(to_fp32)+cuBLAS fallback in
                    // ggml_cuda_op_mul_mat can serve it as a correctness path,
                    // same as the dequant fallback exercised by test-backend-ops'
                    // MUL_MAT/GET_ROWS "all types" sweep.
                    case GGML_TYPE_FP8_B128:
                        return b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
                    // MAD Task 11: scaled-fp8 (ml8-fp8) weights are a real
                    // MUL_MAT weight type — routed to the no-LUT FP8-WMMA path
                    // (WEIGHT_FORMAT=0) in ggml_cuda_mul_mat. Requires fp32
                    // activations + fp32 output, K a multiple of QK_ML8_FP8 (32)
                    // and N a multiple of MT_ML8_BLOCK_SIZE_N (16). The dispatch
                    // aborts loudly if ggml-hip was built without AITER.
                    case GGML_TYPE_ML8_FP8: {
                        // The no-LUT FP8-WMMA GEMM runs on the AITER Triton
                        // kernel, which targets RDNA4 (gfx1201). On a mixed-arch
                        // build (e.g. gfx1201 + gfx1030) only the RDNA4 device
                        // can serve it — gate here so non-RDNA4 devices report
                        // "not supported" instead of aborting at dispatch.
                        const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                        return GGML_CUDA_CC_IS_RDNA4(cc)
                            && b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32
                            && a->ne[0] % 32 == 0   // QK_ML8_FP8
                            && a->ne[1] % 16 == 0;  // MT_ML8_BLOCK_SIZE_N
                    }
                    // ml8-4 sidecar (MAD-223): F8_E4M3 centroids ride along with
                    // ml8_4 weight tensors but are consumed as src[1] of
                    // GGML_OP_ML8_MUL_MAT, never as a real MUL_MAT weight.
                    // Claiming support here keeps the centroid sidecar on the
                    // same HIP buffer as the ml8_4 weight (the loader probes
                    // every "blk.X.ffn_gate.*" tensor with MUL_MAT via the
                    // LLM_TENSOR_INFOS table). ggml_cuda_mul_mat aborts
                    // defensively if anyone actually tries to MUL_MAT an
                    // F8_E4M3 tensor.
                    case GGML_TYPE_F8_E4M3:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_ML8_MUL_MAT:
            {
                // MAD-223 G.4.f: ml8-4 dense GEMM via mt_ml8_gemm. Requires
                // ml8_4 weights + f8_e4m3 centroid LUT + fp32 output.
                // mt_ml8_gemm wraps the kernel for any shape where N is a
                // multiple of MT_ML8_BLOCK_SIZE_N (16) and K is a multiple of
                // QK_ML8 (64). M is padded internally.
                //
                // MAD-3xx activation fusion: x is EITHER GGML_TYPE_F32 [K, M]
                // (legacy, the GEMM quantizes internally) OR GGML_TYPE_I8
                // [K+4, M] — pre-quantized per-row, the packed output of
                // ggml_fp8_quant_rot(..., G=0) (see ggml-ml8.h). The GEMM
                // consumes a_fp8/a_scale straight from that buffer, skipping
                // its own quantize pass.
                const ggml_tensor * w    = op->src[0];
                const ggml_tensor * cent = op->src[1];
                const ggml_tensor * x    = op->src[2];
                if (!w || !cent || !x) return false;
                if (w->type    != GGML_TYPE_ML8_4)    return false;
                if (cent->type != GGML_TYPE_F8_E4M3)  return false;
                if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_I8) return false;
                if (x->type == GGML_TYPE_I8 && x->ne[0] != w->ne[0] + 4) return false;
                // LLAMA_ACT_BF16 (2026-09-18 phase 2): bf16 dst is only wired
                // for the ML8_4 RDNA4_TRFEED prefill tile (M>32); see
                // ggml_cuda_ml8_4_mul_mat_supports_bf16_out (ml8.cu) — the
                // M<=32 decode split-K kernel writes fp32 only.
                if (op->type != GGML_TYPE_F32 &&
                    !(op->type == GGML_TYPE_BF16 &&
                      ggml_cuda_ml8_4_mul_mat_supports_bf16_out(w->ne[1], x->ne[1]))) {
                    return false;
                }
                if (w->ne[0] % 64 != 0)               return false;
                if (w->ne[1] % 16 != 0)               return false;
                // lut_group_off (op_params[0]): under tensor parallelism w
                // holds only a K-slice while cent is mirrored in full, so
                // cent->ne[1] may exceed w->ne[0]/QK_ML8 — see ggml.h.
                const int32_t off = ggml_get_op_params_i32(op, 0);
                if (cent->ne[0] != 16 || off < 0 || off + w->ne[0] / 64 > cent->ne[1]) return false;
                return true;
            } break;
        case GGML_OP_ML8_APPLY_ROTATION:
            {
                // MAD-223 G.4.g: per-token Kronecker rotation H_a^T @ X @ H_b.
                // MAD-266: h_a == NULL selects block_hadamard (Q = I_a ⊗ H_b,
                // no H_a leg) — no a_dim limit in that case, unlike kronecker
                // whose a_dim must fit ml8_h_a_left_multiply_kernel's register
                // array (see ml8.cu). One CUDA block per token; blockDim.x =
                // b_dim, so b_dim must fit the device block-size limit.
                const ggml_tensor * x   = op->src[0];
                const ggml_tensor * h_a = op->src[1];
                if (!x) return false;
                if (x->type   != GGML_TYPE_F32) return false;
                if (h_a != nullptr && h_a->type != GGML_TYPE_F32) return false;
                if (op->type  != GGML_TYPE_F32) return false;
                const int32_t * pp    = (const int32_t *) op->op_params;
                const int32_t   a_dim = pp[0];
                const int32_t   b_dim = pp[1];
                if (h_a != nullptr && (a_dim <= 0 || a_dim > 16)) return false;
                if (b_dim <= 0 || (b_dim & (b_dim - 1)) != 0) return false;
                if (b_dim < 16 || b_dim > 1024) return false;
                return true;
            } break;
        case GGML_OP_ML8_MUL_MAT_ID:
            {
                // MAD-223 G.7: ml8-4 MoE GEMM via mt_ml8_moe_gemm. Same type
                // contract as the dense ML8_MUL_MAT (ml8_4 weights, f8_e4m3
                // centroid LUT, fp32 activation, fp32 output) plus i32 ids,
                // with an extra n_experts dim on w/centroids.
                const ggml_tensor * w    = op->src[0];
                const ggml_tensor * cent = op->src[1];
                const ggml_tensor * x    = op->src[2];
                const ggml_tensor * ids  = op->src[3];
                if (!w || !cent || !x || !ids) return false;
                if (w->type != GGML_TYPE_ML8_4 && w->type != GGML_TYPE_ML8_4_SOA) return false;
                if (cent->type != GGML_TYPE_F8_E4M3)  return false;
                if (x->type    != GGML_TYPE_F32)      return false;
                if (ids->type  != GGML_TYPE_I32)      return false;
                if (op->type   != GGML_TYPE_F32)      return false;
                if (w->ne[0] % 64 != 0)               return false;
                if (w->ne[1] % 16 != 0)               return false;
                if (w->ne[2] <= 0)                    return false;
                return true;
            } break;
        case GGML_OP_ML8_GET_ROWS:
            {
                // MAD-256 ml8-4 native token-embedding gather. Pure dequant
                // gather (no AITER GEMM) — ml8_4 weight, f8_e4m3 centroid LUT,
                // i32 ids, fp32 output. K must be a multiple of QK_ML8=64.
                const ggml_tensor * w    = op->src[0];
                const ggml_tensor * cent = op->src[1];
                const ggml_tensor * ids  = op->src[2];
                if (!w || !cent || !ids) return false;
                if (w->type    != GGML_TYPE_ML8_4)   return false;
                if (cent->type != GGML_TYPE_F8_E4M3) return false;
                if (ids->type  != GGML_TYPE_I32)     return false;
                if (op->type   != GGML_TYPE_F32)     return false;
                if (w->ne[0] % 64 != 0)              return false;
                if (cent->ne[0] != 16)               return false;
                if (cent->ne[1] != w->ne[0] / 64)    return false;
                return true;
            } break;
        case GGML_OP_FP8_QUANT_ROT:
            {
                // FP8_B128 phase 2 design section 3(a)/4(c): fused rotate +
                // block-128 quantize. Runs on any HIP device (the FWHT/H_a^T
                // primitives it reuses from ML8_FP8/ML8_4 rotation are not
                // RDNA4-specific); only the paired FP8_MUL_MAT is gated to
                // gfx1201+AITER, so a mixed-arch build can still run
                // FP8_QUANT_ROT everywhere but only dispatch the GEMM on the
                // RDNA4 device (falls back to CPU mul_mat elsewhere).
                const ggml_tensor * x   = op->src[0];
                const ggml_tensor * h_a = op->src[1];
                if (!x) return false;
                // LLAMA_ACT_BF16 (2026-09-18 phase 2): the V3/V4 per-row
                // kernels are templated on the input element type (see
                // ml8_fp8_qrot_v3_kernel / ml8_fp8_qrot_v4_kernel in ml8.cu),
                // so a bf16 residual/activation row can be quantized directly
                // without an f32 round-trip.
                if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_BF16) return false;
                if (op->type  != GGML_TYPE_I8)  return false;
                if (!ggml_is_contiguous(x))     return false;
                if (x->nb[1] != (size_t) x->ne[0] * ggml_type_size(x->type)) return false;
                {
                    // MAD-305 Phase 5 (round 3): op_params[3] == 0 is now a
                    // genuine "per-row" mode (a single scale for the whole
                    // row, not an alias of G=128) -- only K%32==0 is needed,
                    // there is no scale-group alignment requirement.
                    const int32_t G_raw = ((const int32_t *) op->op_params)[3];
                    if (G_raw == 0) {
                        if (x->ne[0] % 32 != 0) return false;
                    } else {
                        if (G_raw != 32 && G_raw != 128) return false;
                        if (x->ne[0] % G_raw != 0)       return false;
                    }
                }
                const int32_t * pp    = (const int32_t *) op->op_params;
                const int32_t   a_dim = pp[0];
                const int32_t   b_dim = pp[1];
                const int32_t   kind  = pp[2];
                if (kind == GGML_FP8_QUANT_ROT_KIND_NONE) {
                    return h_a == nullptr && x->type == GGML_TYPE_F32;
                }
                if (b_dim < 16 || b_dim > 1024 || (b_dim & (b_dim - 1)) != 0) return false;
                if (a_dim <= 0 || (int64_t) a_dim * (int64_t) b_dim != x->ne[0]) return false;
                // LLAMA_ACT_BF16 (2026-09-18 phase 2): a bf16 src is only
                // dispatched through ml8_launch_qrot_v4/v3 (ml8.cu), which are
                // the only kernels templated on the input type. Rather than
                // duplicate their full internal shape-coverage tables (V3's
                // b_dim==128 path in particular chooses among several
                // NW/per_wave instantiations at runtime, incl. via the
                // MT_FP8_QROT_V3_NW env override), gate bf16 to the exact
                // shapes qwen35's activation stream uses (KRONECKER a<=5,
                // b_dim=1024; BLOCK_HADAMARD b_dim=128) plus a generous a_dim
                // margin under V3's b=128 register-array bound; ml8.cu's
                // ggml_cuda_op_fp8_quant_rot GGML_ABORTs defensively if a
                // shape ever slips through without a matching instantiation.
                if (x->type == GGML_TYPE_BF16) {
                    if (((const int32_t *) op->op_params)[3] != 0) return false;  // per-row (G=0) only
                    if (kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER) {
                        if (b_dim != 1024 || a_dim > 5) return false;
                    } else if (kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD) {
                        if (b_dim != 128 || a_dim > 160) return false;
                    } else {
                        return false;
                    }
                }
                if (kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER) {
                    if (h_a == nullptr || h_a->type != GGML_TYPE_F32) return false;
                    if (a_dim > 16) return false;  // ml8_h_a_left_multiply_kernel register array
                    return h_a->ne[0] == a_dim && h_a->ne[1] == a_dim;
                }
                if (kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD) {
                    return h_a == nullptr;
                }
                return false;
            } break;
        case GGML_OP_FP8_MUL_MAT:
            {
                // FP8_B128 phase 2 design section 4(b): AITER preshuffle
                // GEMM, RDNA4 + AITER only. N must be a multiple of 128 —
                // tighter than the design note's "N%16==0": the packed
                // weight's [K/128, N/128] scale table is only meaningful in
                // whole 128-row N tiles (the design's own CONVERTER
                // INVARIANT for FP8_B128 storage), so an N that isn't
                // %128 has nowhere consistent to source a tile scale from.
                // Such shapes (e.g. test-backend-ops' N=16 case) fall back
                // to the CPU reference instead of aborting.
#ifdef GGML_HIP_AITER
                const ggml_tensor * w = op->src[0];
                const ggml_tensor * a = op->src[1];
                if (!w || !a) return false;
                if (a->type   != GGML_TYPE_I8)       return false;
                if (w->ne[2] != 1 || w->ne[3] != 1)  return false;
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (w->type == GGML_TYPE_FP8_B128) {
                    // K%32==0 covers both activation contracts below; the
                    // block-packing one additionally self-asserts K%128==0
                    // in ggml_cuda_op_fp8_mul_mat (needed for its n_groups=
                    // K/128 to be exact) -- true for every production shape.
                    if (w->ne[0] % 32 != 0)              return false;
                    if (w->ne[1] % 128 != 0)             return false;
                    // MAD-305 Phase 5 (round 3, default): src1->ne[0] ==
                    // K + 4 is the per-row (G=0) contract, dispatched to the
                    // frozen rdna4 trfeed kernel. K + K/32 is the legacy
                    // block-packing (G=128, K/128 groups * 4 bytes) contract
                    // the Triton generic/preshuffle/rdna4-tile layouts serve.
                    const bool per_row = (a->ne[0] == w->ne[0] + 4);
                    const bool block   = (a->ne[0] == w->ne[0] + w->ne[0] / 32);
                    if (!per_row && !block)              return false;
                    // LLAMA_ACT_BF16 (2026-09-18 phase 2): bf16 dst is only
                    // wired for the per-row trfeed epilogue, and only for
                    // M>32 (prefill) — see rdna4_gemm_fp8_trfeed_bf16's
                    // caller in ggml_cuda_op_fp8_mul_mat.
                    if (op->type != GGML_TYPE_F32 &&
                        !(op->type == GGML_TYPE_BF16 && per_row && a->ne[1] * a->ne[2] * a->ne[3] > 32)) {
                        return false;
                    }
                    // The packed weight layout is fixed at load (MT_FP8_B128_LAYOUT):
                    // the trfeed layout carries one scale per row and serves only
                    // the per-row activation contract; the Triton layouts carry the
                    // [K/128,N/128] table and serve only the block contract. A
                    // mismatch (e.g. test-backend-ops running both) goes to CPU.
                    // The trfeed pack also needs K >= 256 to fit in place (N*K+4N
                    // <= 130*N*K/128) -- the cache-copy path covers smaller K, but
                    // keep it out of the kernel path entirely.
                    if (ggml_cuda_fp8_b128_layout_is_per_row()) {
                        if (!per_row)                    return false;
                        if (w->ne[0] < 256)              return false;
                    } else {
                        if (!block)                      return false;
                    }
                    return GGML_CUDA_CC_IS_RDNA4(cc);
                }
                // ML8_FP8 weight (G=32 scale groups): same gating as the
                // existing ML8_FP8 MUL_MAT path (RDNA4 + AITER, K%32==0,
                // N%16==0) -- see deliverable (2) in the FP8_B128 phase 2 design.
                // src1 (the FP8_QUANT_ROT output) must carry exactly K raw
                // e4m3 bytes plus K/32 fp32 per-group scales per row (MAD-305
                // Phase 5 production-integration contract).
                if (w->type == GGML_TYPE_ML8_FP8) {
                    // LLAMA_ACT_BF16 does not extend to this weight type.
                    if (op->type != GGML_TYPE_F32)       return false;
                    if (w->ne[0] % 32 != 0)              return false;
                    if (w->ne[1] % 16 != 0)              return false;
                    if (a->ne[0] != w->ne[0] + w->ne[0] / 8) return false;
                    return GGML_CUDA_CC_IS_RDNA4(cc);
                }
                return false;
#else
                return false;
#endif
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_TQ4_1S:
                    case GGML_TYPE_TQ3_1S:
                    case GGML_TYPE_ML8_FP8:
                    case GGML_TYPE_FP8_B128:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    case GGML_TYPE_TURBO4_0:
                        // 128-element blocks; a partial trailing block would read past it
                        return op->src[0]->ne[0] % QK_TURBO4 == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                // turbo types require head_dim divisible by appropriate group size
                if ((op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO2_0) && op->src[0]->ne[0] % 64 != 0) {
                    return false;
                }
                // turbo4 block size is 128, so head_dim must be divisible by 128
                if (op->type == GGML_TYPE_TURBO4_0 && op->src[0]->ne[0] % 128 != 0) {
                    return false;
                }
                // MAD-214: turbo-FP8 BS=256 requires head_dim divisible by 256
                if (op->type == GGML_TYPE_TURBO4_FP8_BS256 && op->src[0]->ne[0] % 256 != 0) {
                    return false;
                }
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL ||
                               // fork: turbo KV dst types are f32-source only
                               op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO2_0 ||
                               op->type == GGML_TYPE_TURBO4_0 || op->type == GGML_TYPE_TURBO4_FP8_BS256) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SINKHORN_NORM:
            // stricter than the norms above: the kernel holds the whole n*n
            // matrix per thread, so it needs FULL contiguity (not just rows),
            // a square leading pair, and n within the instantiated set.
            return op->src[0]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op) &&
                   op->src[0]->ne[0] == op->src[0]->ne[1] &&
                   (op->src[0]->ne[0] == 2 || op->src[0]->ne[0] == 4 ||
                    op->src[0]->ne[0] == 8);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_TURBO_WHT:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] % 32 == 0;  // supports 32, 64, and 128 WHT groups
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            // BF16 activation coverage (2026-09-18)
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_BF16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16 || op->type         == GGML_TYPE_BF16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_PAGED_ATTN_MT:
            // K/V cache may be quantized; mt::ggml_cuda_op_paged_attn_mt
            // dispatch checks for a specialized paged_cache_ops<TYPE> and
            // aborts with a clear message if unsupported. Currently
            // F16 + Q8_0 + TURBO4_0 + TURBO3_0 are wired (MAD-116).
            return op->type == GGML_TYPE_F16
                && op->src[0]->type == GGML_TYPE_F16   // q
                && op->src[1]->type == op->src[2]->type  // k/v cache same type
                && (op->src[1]->type == GGML_TYPE_F16
                    || op->src[1]->type == GGML_TYPE_Q8_0
                    || op->src[1]->type == GGML_TYPE_TURBO4_0
                    || op->src[1]->type == GGML_TYPE_TURBO3_0
                    || op->src[1]->type == GGML_TYPE_TURBO4_FP8_BS256   // MAD-214
                    || op->src[1]->type == GGML_TYPE_TURBO4_64          // MAD-301C Lever B
                    || op->src[1]->type == GGML_TYPE_TURBO4_64_OL      // SP2.5 fixed-outlier-channel
                    || op->src[1]->type == GGML_TYPE_TURBO4_64_OL8     // outlier-matrix sweep
                    || op->src[1]->type == GGML_TYPE_TURBO4_64_OL12    // outlier-matrix sweep
                    || op->src[1]->type == GGML_TYPE_R4D_FP8_KV)       // libr4d paged fp8 KV
                && op->src[6]                          // k_cur (fused scatter)
                && op->src[6]->type == GGML_TYPE_F16
                && op->src[7]                          // v_cur (fused scatter)
                && op->src[7]->type == GGML_TYPE_F16
                && op->src[8]                          // slot_mapping (fused scatter)
                && op->src[8]->type == GGML_TYPE_I32;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_ALL_QUANTS
        features.push_back({ "FA_ALL_QUANTS", "1" });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// WP_META_SLOT_STREAMS: select which of this device's compute streams (see
// ggml_backend_cuda_context::stream(device, curr_stream_no) in common.cuh) subsequent work on
// `backend` is dispatched to -- cublas handle, cuda_pool()/scratch, and cuda_ctx->stream() (which
// is what AllReduce's `cs` and ggml_backend_cuda_event_record/_wait resolve to) all key off
// curr_stream_no already, per-[device][stream], so flipping this one field is sufficient; nothing
// else needs to change per stream. Used exclusively by the meta backend
// (ggml-backend-meta.cpp) to give each rolling tensor-parallel overlap slot (0 = "sched", 1 =
// "sched_overlap" in llama_context) its own stream per device instead of both slots funneling
// through stream 0, which is the actual cause of the AR_END_WAIT stall documented at the
// WP_META_SLOT_STREAMS call sites: the two slots' AllReduce end-wait/unpack used to be FIFO-
// ordered against the OTHER slot's kernels purely because they shared one stream, not because of
// a real dependency.
static void ggml_backend_cuda_set_stream_no(ggml_backend_t backend, int stream_no) {
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    GGML_ASSERT(stream_no >= 0 && stream_no < GGML_CUDA_MAX_STREAMS);
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    cuda_ctx->curr_stream_no = stream_no;
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_begin") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_begin;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_end") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_end;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_set_stream_no") == 0) {
        return (void *)ggml_backend_cuda_set_stream_no;
    }
    // WP_TP_TRACE_FILE: proc-address bridges into wp-tp-trace.cu for
    // ggml-backend-meta.cpp, which is backend-agnostic and cannot link CUDA
    // directly -- resolved once per backend_config, same pattern as
    // ggml_backend_set_stream_no above.
    if (strcmp(name, "wp_tp_trace_mark") == 0) {
        return (void *)wp_tp_trace_mark;
    }
    if (strcmp(name, "wp_tp_trace_mark_global") == 0) {
        return (void *)wp_tp_trace_mark_global;
    }
    if (strcmp(name, "wp_tp_trace_gpu_mark") == 0) {
        return (void *)wp_tp_trace_gpu_mark;
    }
    // mad-lab: MAD_META_GPUTIME -- ggml-backend-meta.cpp is backend-agnostic
    // (see the header comment on mad_meta_gputime_mark in allreduce.cuh) and
    // reaches this file's implementation (allreduce.cu) purely through this
    // proc-address, the same mechanism already used for
    // ggml_backend_set_stream_no and wp_tp_trace_mark/_gpu_mark above.
    if (strcmp(name, "mad_meta_gputime_mark") == 0) {
        return (void *)mad_meta_gputime_mark;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

#if defined(GGML_USE_HIP) && defined(USE_CUDA_GRAPH)
    ggml_cuda_hipblaslt_warmup(*ctx);
#endif

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    ggml_backend_cuda_device_active_count_inc(cuda_backend->device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
