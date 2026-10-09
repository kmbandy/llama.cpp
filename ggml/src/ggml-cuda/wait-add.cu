#include "wait-add.cuh"

#if defined(GGML_USE_HIP)

// Device-side consume of a remote expert-dispatch partial.
//
// Handshake (validated by /tmp/flagprobe on the 9070XT: 40k iterations, 0
// mismatches): the host writes the slot data, issues a release fence and then
// release-stores ctrl->flag = ctrl->expect. This kernel is enqueued BEFORE the
// data exist, spins with a system-scope acquire load on ctrl->flag, and then
// reads the slot with non-temporal loads (never a stale cache line).
//
// `expect` is read from host-pinned memory at RUN time, not passed as a kernel
// argument, so a captured HIP graph replays correctly: the only things baked
// into the graph are the (persistent) pointers.
//
// Arithmetic: one fp32 add per element, dst = a + slot -- identical to
// ggml_add(a, slot), no contraction possible.
static __global__ void k_wait_add(
        const float * __restrict__ a, const float * slot, float * __restrict__ dst,
        const int64_t n, ggml_wait_ctrl * ctrl, const long long timeout_ticks) {
    __shared__ int ok;
    if (ctrl != nullptr) {
        if (threadIdx.x == 0) {
            const uint64_t expect = __hip_atomic_load(&ctrl->expect, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
            const long long t0 = wall_clock64();
            ok = 1;
            while (__hip_atomic_load(&ctrl->flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != expect) {
                if (wall_clock64() - t0 > timeout_ticks) {
                    __hip_atomic_store(&ctrl->fail, (uint64_t) 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                    ok = 0;
                    break;
                }
            }
        }
        __syncthreads();
    } else {
        ok = 1;
    }
    const bool good = ok != 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = good ? __builtin_nontemporal_load(slot + i) : 0.0f;
        dst[i] = a[i] + v;
    }
}

static long long wait_add_ticks_per_ms(int device) {
    static long long cache[16] = {};
    if (device >= 0 && device < 16 && cache[device] != 0) {
        return cache[device];
    }
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) != hipSuccess || khz <= 0) {
        (void) hipGetLastError();
        khz = 100000; // 100 MHz constant wall clock (gfx9/gfx12)
    }
    if (device >= 0 && device < 16) {
        cache[device] = khz;
    }
    return khz; // kHz == ticks per millisecond
}

void ggml_cuda_op_wait_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));

    const float *    slot = nullptr;
    ggml_wait_ctrl * ctrl = nullptr;
    int32_t timeout_ms = 0;
    memcpy(&slot, (const char *) dst->op_params, sizeof(void *));
    memcpy(&ctrl, (const char *) dst->op_params + sizeof(void *), sizeof(void *));
    memcpy(&timeout_ms, (const char *) dst->op_params + 2*sizeof(void *), sizeof(int32_t));

    const int64_t n = ggml_nelements(dst);
    const int block = 256;
    const int grid  = (int) std::min<int64_t>((n + block - 1) / block, 64);
    const long long ticks = wait_add_ticks_per_ms(ctx.device) * (long long) timeout_ms;
    k_wait_add<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data, slot, (float *) dst->data, n, ctrl, ticks);
}

#else  // !GGML_USE_HIP

void ggml_cuda_op_wait_add(ggml_backend_cuda_context &, ggml_tensor *) {
    GGML_ABORT("GGML_OP_WAIT_ADD is HIP-only");
}

#endif
