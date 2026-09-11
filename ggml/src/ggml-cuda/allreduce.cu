#include "allreduce.cuh"

#include <vector>

#if !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "allreduce-ml8.cuh"
#include "cpy-utils.cuh"
#include "dequantize.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <limits>
#include <string>
#include <thread>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif // defined(__linux__)

// ---------------------------------------------------------------------------
// CUDA / HIP AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA (or,
// under GGML_USE_HIP, ROCm) devices in the same process.  Used by the
// tensor-split path alongside NCCL/RCCL; targets setups without NVLink/XGMI,
// where data is exchanged between the GPUs by staging it through pinned host
// memory over PCIe.
//
// HIP notes (see the individual sites for detail):
//   * Host staging is allocated with hipHostMalloc(Portable|Mapped|Coherent)
//     where the device polls/writes it, so the mapping is fine-grained and
//     device stores land in host memory rather than in the GPU's L2.
//   * The in-kernel spin uses __builtin_amdgcn_s_sleep instead of __nanosleep.
//   * The arrival handshake uses __hip_atomic_load/store at
//     __HIP_MEMORY_SCOPE_SYSTEM instead of volatile accesses.
//   * MUSA still gets the nullptr stub at the bottom of this file.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
//
// HIP: volatile is explicitly *not* a cross-agent ordering primitive in the HIP
// memory model -- it only stops the compiler from eliding the access, it does
// not control the sc0/sc1 (glc/dlc) cache bits on the generated load/store.  We
// therefore use __hip_atomic_load / __hip_atomic_store at
// __HIP_MEMORY_SCOPE_SYSTEM, which lower to relaxed-size (4 B) accesses that
// bypass/invalidate the device caches and carry the acquire/release waitcnt +
// cache-maintenance instructions.  These are plain load/store atomics (no
// read-modify-write), so they do *not* need PCIe atomics / native host atomic
// support -- unlike atomicAdd_system.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
#if defined(GGML_USE_HIP)
    __hip_atomic_store(p, token, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    *(volatile int *)p = token;
#endif // defined(GGML_USE_HIP)
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
#if defined(GGML_USE_HIP)
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *(const volatile int *)p;
#endif // defined(GGML_USE_HIP)
}

// Backoff inside the arrival spin loop.
//
// NVIDIA: __nanosleep(100), sm70+.
// AMD:    s_sleep, whose operand is an immediate bounded to a small range
//         (only the low bits are honoured by the hardware); s_sleep(2) parks
//         the wave for ~128 core clocks, i.e. the same tens-of-nanoseconds
//         order as __nanosleep(100).  Called from a loop, so the total wait is
//         unbounded by design -- see the note at the call site.
//
// NOTE: vendors/hip.h #defines __CUDA_ARCH__ to 1300 unconditionally (host pass
// included), so the GGML_USE_HIP branch MUST come first; a bare
// "__CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA" test is always true under HIP.
static __device__ __forceinline__ void ggml_cuda_ar_spin_pause() {
#if defined(GGML_USE_HIP)
    __builtin_amdgcn_s_sleep(2);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(100);
#else
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP)
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    // Vector unit for the wire type, sized to the arch's widest single-instruction
    // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
    // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid       = threadIdx.x;
    const int nt        = blockDim.x;
    const int bid       = blockIdx.x;
    const int gtid      = bid * nt + tid;
    const int gnt       = gridDim.x * nt;
    const int count_vec = count / ELEMS_PER_VEC;
    const int tail      = count_vec * ELEMS_PER_VEC;

    // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        if (bid == 0 && tid < count - tail) {
            host_mine[tail + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
        }
    }

    // Commit this block's host writes before signalling.
    __threadfence_system();
    __syncthreads();

    // Phase 2: thread 0 of each block signals on its own arrival slot, then
    // spins for the matching slot from peer.  Per-block tokens mean blocks
    // proceed independently -- no inter-block barrier needed.
    //
    // Wave-size independence: the handshake is expressed purely in terms of
    // threadIdx.x == 0 plus __syncthreads(), with no warp-level primitives, no
    // warpSize/32 arithmetic and no ballot/shfl.  It is therefore correct for
    // both wave32 and wave64 (RDNA defaults to wave64 under HIP).  The spin
    // sits in divergent control flow inside wave 0; the remaining lanes of that
    // wave fall through to the __syncthreads() below, which is the
    // reconvergence point on both vendors.  Do NOT reintroduce any
    // warp-granular assumption here.
    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system(); // make our signal visible system-wide

        // Deliberately unbounded.  Bailing out after N iterations would let the
        // kernel proceed to phase 3 and read a stale/partial host_other, i.e.
        // silently wrong numerics; a hang is at least diagnosable.  The
        // fail-safe for this pipeline is at init time (nullptr -> generic
        // AllReduce), not mid-kernel.  On ROCm a genuine deadlock here trips
        // the HSA queue watchdog and surfaces as a GPU fault.
        while (ggml_cuda_ar_signal_get(other_slot) != token) {
            ggml_cuda_ar_spin_pause();
        }
    }

    __syncthreads();

    // Acquire peer's host_other writes (this block's stripe of them).
    __threadfence_system();

    // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
    // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
    {
        for (int i = gtid; i < count_vec; i += gnt) {
            const int off = i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (bid == 0 && tid < count - tail) {
            const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail + tid]);
            recvbuf[tail + tid] = ggml_cuda_cast<T_dst>(
                ggml_cuda_cast<float>(d_low) +
                ggml_cuda_cast<float>(host_other[tail + tid]));
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

struct ggml_cuda_ar_codec;

template <typename T_dst, typename T_src>
static void ggml_cuda_ar_dx_add(T_dst * dst, const T_src * src, int64_t ne, cudaStream_t stream);

using ggml_cuda_ar_pack_fn = void (*) (
        const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream);
using ggml_cuda_ar_unpack_accumulate_fn = void (*) (
        void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream);

struct ggml_cuda_ar_codec {
    const char *                         name;
    ggml_type                            wire_type;
    size_t                               elements_per_block;
    size_t                               bytes_per_block;
    ggml_cuda_ar_pack_fn                 pack_fn;
    ggml_cuda_ar_unpack_accumulate_fn    unpack_accumulate_fn;
};

template <typename T_src>
static __global__ void ggml_cuda_ar_codec_pack_q8_0_kernel(
        const T_src * src, block_q8_0 * dst, int n_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int ib = tid; ib < n_blocks; ib += nt) {
        float values[QK8_0];
        for (int j = 0; j < QK8_0; ++j) {
            values[j] = ggml_cuda_cast<float>(src[ib * QK8_0 + j]);
        }
        quantize_f32_q8_0_block(values, &dst[ib]);
    }
}

template <typename T_dst>
static __global__ void ggml_cuda_ar_codec_unpack_q8_0_kernel(
        T_dst * dst, const block_q8_0 * src, int n_blocks) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int ib = tid; ib < n_blocks; ib += nt) {
        float values[QK8_0];
        for (int j = 0; j < QK8_0; ++j) {
            values[j] = ggml_cuda_cast<float>(dst[ib * QK8_0 + j]);
        }

        block_q8_0 local;
        quantize_f32_q8_0_block(values, &local);
        for (int j = 0; j < QK8_0; j += 2) {
            float2 local_value;
            float2 peer_value;
            dequantize_q8_0(&local, 0, j, local_value);
            dequantize_q8_0(src, ib, j, peer_value);
            dst[ib * QK8_0 + j + 0] = ggml_cuda_cast<T_dst>(local_value.x + peer_value.x);
            dst[ib * QK8_0 + j + 1] = ggml_cuda_cast<T_dst>(local_value.y + peer_value.y);
        }
    }
}

static int ggml_cuda_ar_codec_grid(int64_t n) {
    const int blocks = (int) ((n + 255) / 256);
    return std::max(1, std::min(blocks, 1024));
}

template <typename T_src>
static void ggml_cuda_ar_codec_pack_q8_0(
        const void * src, void * dst, int64_t ne, cudaStream_t stream) {
    const int n_blocks = (int) (ne / QK8_0);
    ggml_cuda_ar_codec_pack_q8_0_kernel<T_src><<<ggml_cuda_ar_codec_grid(n_blocks), 256, 0, stream>>>(
        static_cast<const T_src *>(src), static_cast<block_q8_0 *>(dst), n_blocks);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T_dst>
static void ggml_cuda_ar_codec_unpack_q8_0(
        void * dst, const void * src, int64_t ne, cudaStream_t stream) {
    const int n_blocks = (int) (ne / QK8_0);
    ggml_cuda_ar_codec_unpack_q8_0_kernel<T_dst><<<ggml_cuda_ar_codec_grid(n_blocks), 256, 0, stream>>>(
        static_cast<T_dst *>(dst), static_cast<const block_q8_0 *>(src), n_blocks);
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_ar_codec_pack_bf16(
        const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream) {
    if (src_type == GGML_TYPE_BF16) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, (size_t) ne * sizeof(nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    } else {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(src_type);
        to_bf16(src, static_cast<nv_bfloat16 *>(dst), ne, stream);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void ggml_cuda_ar_codec_pack_f16(
        const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream) {
    if (src_type == GGML_TYPE_F16) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, (size_t) ne * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    } else {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(src_type);
        to_fp16(src, static_cast<half *>(dst), ne, stream);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void ggml_cuda_ar_codec_pack_f32(
        const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream) {
    if (src_type == GGML_TYPE_F32) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, (size_t) ne * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    } else {
        to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(src_type);
        to_fp32(src, static_cast<float *>(dst), ne, stream);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void ggml_cuda_ar_codec_pack_q8_0_dispatch(
        const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream) {
    switch (src_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_codec_pack_q8_0<float>(src, dst, ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_codec_pack_q8_0<half>(src, dst, ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_codec_pack_q8_0<nv_bfloat16>(src, dst, ne, stream); break;
        default: GGML_ABORT("AllReduce q8_0 pack: unsupported source type %d", (int) src_type);
    }
}

static void ggml_cuda_ar_codec_unpack_bf16(
        void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_dx_add<float, nv_bfloat16>(static_cast<float *>(dst), static_cast<const nv_bfloat16 *>(src), ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_dx_add<half, nv_bfloat16>(static_cast<half *>(dst), static_cast<const nv_bfloat16 *>(src), ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_dx_add<nv_bfloat16, nv_bfloat16>(static_cast<nv_bfloat16 *>(dst), static_cast<const nv_bfloat16 *>(src), ne, stream); break;
        default: GGML_ABORT("AllReduce bf16 unpack: unsupported destination type %d", (int) dst_type);
    }
}

static void ggml_cuda_ar_codec_unpack_f16(
        void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_dx_add<float, half>(static_cast<float *>(dst), static_cast<const half *>(src), ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_dx_add<half, half>(static_cast<half *>(dst), static_cast<const half *>(src), ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_dx_add<nv_bfloat16, half>(static_cast<nv_bfloat16 *>(dst), static_cast<const half *>(src), ne, stream); break;
        default: GGML_ABORT("AllReduce f16 unpack: unsupported destination type %d", (int) dst_type);
    }
}

static void ggml_cuda_ar_codec_unpack_f32(
        void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_dx_add<float, float>(static_cast<float *>(dst), static_cast<const float *>(src), ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_dx_add<half, float>(static_cast<half *>(dst), static_cast<const float *>(src), ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_dx_add<nv_bfloat16, float>(static_cast<nv_bfloat16 *>(dst), static_cast<const float *>(src), ne, stream); break;
        default: GGML_ABORT("AllReduce f32 unpack: unsupported destination type %d", (int) dst_type);
    }
}

static void ggml_cuda_ar_codec_unpack_q8_0_dispatch(
        void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_codec_unpack_q8_0<float>(dst, src, ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_codec_unpack_q8_0<half>(dst, src, ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_codec_unpack_q8_0<nv_bfloat16>(dst, src, ne, stream); break;
        default: GGML_ABORT("AllReduce q8_0 unpack: unsupported destination type %d", (int) dst_type);
    }
}


// ---- ml8-k pack / unpack-accumulate ---------------------------------------
// Same rank-symmetric contract as q8_0: the LOCAL partial is re-quantized here
// before summing, so both ranks add two rounded values and produce bit-identical
// sums. Leaving the local side unrounded would be more accurate per rank and
// would make the two ranks disagree.
#define GGML_CUDA_AR_ML8_DEF(K)                                                                   \
template <typename T_src>                                                                          \
static __global__ void ggml_cuda_ar_codec_pack_ml8_##K##_kernel(                                   \
        const T_src * src, block_ml8_##K##_wire * dst, int n_blocks) {                             \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    const int nt  = gridDim.x * blockDim.x;                                                        \
    for (int ib = tid; ib < n_blocks; ib += nt) {                                                  \
        float v[QK_ML8_WIRE];                                                                      \
        for (int j = 0; j < QK_ML8_WIRE; ++j) {                                                    \
            v[j] = ggml_cuda_cast<float>(src[ib * QK_ML8_WIRE + j]);                               \
        }                                                                                          \
        ml8_##K##_quantize(v, &dst[ib]);                                                           \
    }                                                                                              \
}                                                                                                  \
template <typename T_dst>                                                                          \
static __global__ void ggml_cuda_ar_codec_unpack_ml8_##K##_kernel(                                 \
        T_dst * dst, const block_ml8_##K##_wire * src, int n_blocks) {                             \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;                                         \
    const int nt  = gridDim.x * blockDim.x;                                                        \
    for (int ib = tid; ib < n_blocks; ib += nt) {                                                  \
        float v[QK_ML8_WIRE];                                                                      \
        for (int j = 0; j < QK_ML8_WIRE; ++j) {                                                    \
            v[j] = ggml_cuda_cast<float>(dst[ib * QK_ML8_WIRE + j]);                               \
        }                                                                                          \
        block_ml8_##K##_wire local;                                                                \
        ml8_##K##_quantize(v, &local);                                                             \
        float lv[QK_ML8_WIRE];                                                                     \
        float pv[QK_ML8_WIRE];                                                                     \
        ml8_##K##_dequantize(&local,   lv);                                                        \
        ml8_##K##_dequantize(&src[ib], pv);                                                        \
        for (int j = 0; j < QK_ML8_WIRE; ++j) {                                                    \
            dst[ib * QK_ML8_WIRE + j] = ggml_cuda_cast<T_dst>(lv[j] + pv[j]);                      \
        }                                                                                          \
    }                                                                                              \
}                                                                                                  \
template <typename T_src>                                                                          \
static void ggml_cuda_ar_codec_pack_ml8_##K(                                                       \
        const void * src, void * dst, int64_t ne, cudaStream_t stream) {                           \
    const int nb = (int) (ne / QK_ML8_WIRE);                                                       \
    ggml_cuda_ar_codec_pack_ml8_##K##_kernel<T_src>                                                \
        <<<ggml_cuda_ar_codec_grid(nb), 256, 0, stream>>>(                                          \
            static_cast<const T_src *>(src), static_cast<block_ml8_##K##_wire *>(dst), nb);        \
    CUDA_CHECK(cudaGetLastError());                                                                \
}                                                                                                  \
template <typename T_dst>                                                                          \
static void ggml_cuda_ar_codec_unpack_ml8_##K(                                                     \
        void * dst, const void * src, int64_t ne, cudaStream_t stream) {                           \
    const int nb = (int) (ne / QK_ML8_WIRE);                                                       \
    ggml_cuda_ar_codec_unpack_ml8_##K##_kernel<T_dst>                                              \
        <<<ggml_cuda_ar_codec_grid(nb), 256, 0, stream>>>(                                          \
            static_cast<T_dst *>(dst), static_cast<const block_ml8_##K##_wire *>(src), nb);        \
    CUDA_CHECK(cudaGetLastError());                                                                \
}                                                                                                  \
static void ggml_cuda_ar_codec_pack_ml8_##K##_dispatch(                                            \
        const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream) {       \
    switch (src_type) {                                                                            \
        case GGML_TYPE_F32:  ggml_cuda_ar_codec_pack_ml8_##K<float>(src, dst, ne, stream); break;  \
        case GGML_TYPE_F16:  ggml_cuda_ar_codec_pack_ml8_##K<half>(src, dst, ne, stream); break;   \
        case GGML_TYPE_BF16: ggml_cuda_ar_codec_pack_ml8_##K<nv_bfloat16>(src, dst, ne, stream); break; \
        default: GGML_ABORT("AllReduce ml8-" #K " pack: unsupported source type %d", (int) src_type); \
    }                                                                                              \
}                                                                                                  \
static void ggml_cuda_ar_codec_unpack_ml8_##K##_dispatch(                                          \
        void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream) {       \
    switch (dst_type) {                                                                            \
        case GGML_TYPE_F32:  ggml_cuda_ar_codec_unpack_ml8_##K<float>(dst, src, ne, stream); break; \
        case GGML_TYPE_F16:  ggml_cuda_ar_codec_unpack_ml8_##K<half>(dst, src, ne, stream); break;  \
        case GGML_TYPE_BF16: ggml_cuda_ar_codec_unpack_ml8_##K<nv_bfloat16>(dst, src, ne, stream); break; \
        default: GGML_ABORT("AllReduce ml8-" #K " unpack: unsupported destination type %d", (int) dst_type); \
    }                                                                                              \
}

GGML_CUDA_AR_ML8_DEF(4)
GGML_CUDA_AR_ML8_DEF(5)
GGML_CUDA_AR_ML8_DEF(8)


// ml8-8r: scale-free E4M3. Elementwise, so it needs no block machinery -- and
// rank symmetry still holds because both partials are rounded identically.
template <typename T_src>
static __global__ void ggml_cuda_ar_codec_pack_ml8_8r_kernel(const T_src * src, uint8_t * dst, int64_t ne) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nt  = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < ne; i += nt) {
        ml8_8r_quantize_elem(ggml_cuda_cast<float>(src[i]), &dst[i]);
    }
}
template <typename T_dst>
static __global__ void ggml_cuda_ar_codec_unpack_ml8_8r_kernel(T_dst * dst, const uint8_t * src, int64_t ne) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nt  = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < ne; i += nt) {
        const float local = ml8_8r_dequantize_elem(ml8_f32_to_e4m3(ggml_cuda_cast<float>(dst[i])));
        dst[i] = ggml_cuda_cast<T_dst>(local + ml8_8r_dequantize_elem(src[i]));
    }
}
template <typename T_src>
static void ggml_cuda_ar_codec_pack_ml8_8r(const void * src, void * dst, int64_t ne, cudaStream_t stream) {
    ggml_cuda_ar_codec_pack_ml8_8r_kernel<T_src><<<ggml_cuda_ar_codec_grid(ne), 256, 0, stream>>>(
        static_cast<const T_src *>(src), static_cast<uint8_t *>(dst), ne);
    CUDA_CHECK(cudaGetLastError());
}
template <typename T_dst>
static void ggml_cuda_ar_codec_unpack_ml8_8r(void * dst, const void * src, int64_t ne, cudaStream_t stream) {
    ggml_cuda_ar_codec_unpack_ml8_8r_kernel<T_dst><<<ggml_cuda_ar_codec_grid(ne), 256, 0, stream>>>(
        static_cast<T_dst *>(dst), static_cast<const uint8_t *>(src), ne);
    CUDA_CHECK(cudaGetLastError());
}
static void ggml_cuda_ar_codec_pack_ml8_8r_dispatch(const void * src, ggml_type src_type, void * dst, int64_t ne, cudaStream_t stream) {
    switch (src_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_codec_pack_ml8_8r<float>(src, dst, ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_codec_pack_ml8_8r<half>(src, dst, ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_codec_pack_ml8_8r<nv_bfloat16>(src, dst, ne, stream); break;
        default: GGML_ABORT("AllReduce ml8-8r pack: unsupported source type %d", (int) src_type);
    }
}
static void ggml_cuda_ar_codec_unpack_ml8_8r_dispatch(void * dst, ggml_type dst_type, const void * src, int64_t ne, cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:  ggml_cuda_ar_codec_unpack_ml8_8r<float>(dst, src, ne, stream); break;
        case GGML_TYPE_F16:  ggml_cuda_ar_codec_unpack_ml8_8r<half>(dst, src, ne, stream); break;
        case GGML_TYPE_BF16: ggml_cuda_ar_codec_unpack_ml8_8r<nv_bfloat16>(dst, src, ne, stream); break;
        default: GGML_ABORT("AllReduce ml8-8r unpack: unsupported destination type %d", (int) dst_type);
    }
}

static const ggml_cuda_ar_codec GGML_CUDA_AR_CODECS[] = {
    { "bf16",  GGML_TYPE_BF16,  1, sizeof(nv_bfloat16),  ggml_cuda_ar_codec_pack_bf16,           ggml_cuda_ar_codec_unpack_bf16 },
    { "f16",   GGML_TYPE_F16,   1, sizeof(half),          ggml_cuda_ar_codec_pack_f16,            ggml_cuda_ar_codec_unpack_f16  },
    { "f32",   GGML_TYPE_F32,   1, sizeof(float),         ggml_cuda_ar_codec_pack_f32,            ggml_cuda_ar_codec_unpack_f32  },
    { "q8_0",  GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0), ggml_cuda_ar_codec_pack_q8_0_dispatch, ggml_cuda_ar_codec_unpack_q8_0_dispatch },
    // ml8 wire tiers are NOT ggml types -- they exist only on this link and are
    // never serialised. Q4_0/Q5_0 ids are reused as private registry keys so the
    // end() lookup by op->wire_type works; nothing else may pack those ids here.
    { "ml8_4", GGML_TYPE_Q4_0, QK_ML8_WIRE, sizeof(block_ml8_4_wire), ggml_cuda_ar_codec_pack_ml8_4_dispatch, ggml_cuda_ar_codec_unpack_ml8_4_dispatch },
    { "ml8_5", GGML_TYPE_Q5_0, QK_ML8_WIRE, sizeof(block_ml8_5_wire), ggml_cuda_ar_codec_pack_ml8_5_dispatch, ggml_cuda_ar_codec_unpack_ml8_5_dispatch },
    { "ml8_8", GGML_TYPE_Q4_1, QK_ML8_WIRE, sizeof(block_ml8_8_wire), ggml_cuda_ar_codec_pack_ml8_8_dispatch, ggml_cuda_ar_codec_unpack_ml8_8_dispatch },
    { "ml8_8r", GGML_TYPE_Q5_1, 1, 1, ggml_cuda_ar_codec_pack_ml8_8r_dispatch, ggml_cuda_ar_codec_unpack_ml8_8r_dispatch },
};

static_assert(sizeof(block_q8_0) == 34, "unexpected q8_0 wire block size");

static const ggml_cuda_ar_codec * ggml_cuda_ar_codec_from_name(const char * name) {
    for (const auto & codec : GGML_CUDA_AR_CODECS) {
        if (std::strcmp(codec.name, name) == 0) {
            return &codec;
        }
    }
    return nullptr;
}

static const ggml_cuda_ar_codec * ggml_cuda_ar_codec_from_type(ggml_type type) {
    for (const auto & codec : GGML_CUDA_AR_CODECS) {
        if (codec.wire_type == type) {
            return &codec;
        }
    }
    return nullptr;
}

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t cpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {};  // copy-engine D2H chunks complete
    cudaEvent_t h2d = nullptr;  // copy-engine H2Ds complete (handoff AR stream -> compute stream)
    cudaEvent_t ker = nullptr;  // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
//
// HIP: hipHostMalloc's *coherence* is not part of the CUDA API surface and, for
// hipHostMallocDefault, has historically been switchable by environment
// (HIP_HOST_COHERENT).  Buffers that the device polls or writes while the peer
// device concurrently reads them MUST be fine-grained, so we request
// hipHostMallocCoherent explicitly rather than trusting the default.  Buffers
// touched only by the copy engine (SDMA) are left at the default coherence --
// their ordering comes from stream/event handshakes, not from cache behaviour,
// and fine-grained mappings can cost D2H/H2D bandwidth.
enum ggml_cuda_ar_host_coherence {
    GGML_CUDA_AR_HOST_DEFAULT,   // copy-engine staging: SDMA only
    GGML_CUDA_AR_HOST_COHERENT,  // kernel-visible: device polls / stores here
};

struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes, ggml_cuda_ar_host_coherence coherence = GGML_CUDA_AR_HOST_DEFAULT) {
        unsigned int flags = cudaHostAllocPortable | cudaHostAllocMapped;
#if defined(GGML_USE_HIP)
        if (coherence == GGML_CUDA_AR_HOST_COHERENT) {
            flags |= hipHostMallocCoherent;
        }
#else
        GGML_UNUSED(coherence);
#endif // defined(GGML_USE_HIP)
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes, flags);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// Duplex ("dx") transport -- asynchronous, event-driven, no host syncs.
//
// Ranks r0 = devices[0], r1 = devices[1].  Per reduce, each rank produces its
// wire-typed partial in a device-resident send buffer on its compute stream,
// then:
//
//   p2p:  r1's out-stream pushes send1 straight into a receive buffer that
//         lives in r0's memory (peer access r1 -> r0 enabled at init; the
//         reverse direction is not required and is not attempted).
//         r0's out-stream D2H-copies send0 into pinned host staging; r1's
//         in-stream waits that (cross-device event) and H2D-copies staging
//         into r1's receive buffer.
//   host: both ranks D2H into their own staging; each rank's in-stream waits
//         the peer's D2H event and H2D-copies into its receive buffer.
//
// Both directions are issued back-to-back on separate streams so the link
// runs duplex.  end() makes the compute stream wait for "own send done" and
// "peer data landed" and runs the same add kernel as the copy-engine path,
// i.e. identical numerics: both partials rounded through the wire type, F32
// accumulate, own + peer order.
//
// Buffers are double-buffered by slot (DX_SLOTS).  Reuse of a slot is fenced
// by three events per (rank, slot): sent (out-stream), recvd (in-stream) and
// freed (compute stream, after the add kernel).  The fences assume the
// caller ends op N before it begins op N+DX_SLOTS -- asserted at begin().
// ---------------------------------------------------------------------------
enum ggml_cuda_ar_transport {
    GGML_CUDA_AR_TRANSPORT_COPY, // legacy synchronous copy-engine host bounce (+ chunked kernel)
    GGML_CUDA_AR_TRANSPORT_P2P,  // duplex: r1 pushes into r0 memory, r1 pulls via host staging
    GGML_CUDA_AR_TRANSPORT_HOST, // duplex: both directions via host staging
};

static constexpr int GGML_CUDA_AR_DX_SLOTS = 2;

struct ggml_cuda_ar_dx_slot {
    cudaEvent_t app   = nullptr; // compute stream: send buffer ready
    cudaEvent_t sent  = nullptr; // out-stream: outbound transfer done (send + staging/peer-recv written)
    cudaEvent_t recvd = nullptr; // in-stream:  inbound H2D into recv done (pull ranks only)
    cudaEvent_t freed = nullptr; // compute stream: add kernel done, recv slot reusable
    bool sent_valid  = false;
    bool recvd_valid = false;
    bool freed_valid = false;
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    const struct ggml_cuda_ar_codec * wire_codec;
    bool     wire_codec_explicit;
    uint64_t call_count;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copy-engine path
    cudaStream_t             streams[GGML_CUDA_MAX_DEVICES];   // non-blocking
    ggml_cuda_ar_event_slot  ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copy-engine: per-device "I finished reading my peer's host_large"
    // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
    // after stage 2's last H2D from host_large[peer].  Waited cross-device
    // by peer's stage-1 stream before the next AR overwrites host_large[peer].
    cudaEvent_t              host_large_read_done[GGML_CUDA_MAX_DEVICES];
    bool                     host_large_read_done_valid;

    // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
    // Recorded on the compute stream after each add_kernel; the AR stream
    // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
    // single-buffer dev_tmp despite add_kernel running on a separate stream.
    cudaEvent_t              dev_tmp_kernel_done[GGML_CUDA_MAX_DEVICES];
    bool                     dev_tmp_kernel_done_valid;

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;

    // Duplex transport state (unused when transport == COPY).
    ggml_cuda_ar_transport    transport;
    size_t                    dx_bytes;      // bytes per slot
    uint64_t                  dx_call;       // begin() counter -> slot = dx_call % DX_SLOTS
    int                       dx_in_flight;  // begun-but-not-ended ops
    cudaStream_t              streams_in[GGML_CUDA_MAX_DEVICES];  // inbound H2D (non-blocking)
    char *                    dx_send[GGML_CUDA_MAX_DEVICES];     // device: DX_SLOTS * dx_bytes, wire-typed partial
    char *                    dx_recv[GGML_CUDA_MAX_DEVICES];     // device: DX_SLOTS * dx_bytes, peer's partial
    ggml_cuda_ar_host_mapping dx_staging[GGML_CUDA_MAX_DEVICES];  // pinned host: DX_SLOTS * dx_bytes
    ggml_cuda_ar_dx_slot      dx_ev[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_DX_SLOTS];

    // -----------------------------------------------------------------
    // Stall watchdog (GGML_CUDA_AR_WATCHDOG_S). All of the below is
    // written with relaxed atomics from the existing hot-path call sites
    // (begin/end/acquire_slot) -- no syscalls, no extra synchronization,
    // zero cost when the watchdog thread isn't running. See
    // ggml_cuda_ar_watchdog_touch() / ggml_cuda_ar_watchdog_main().
    // -----------------------------------------------------------------
    std::atomic<uint64_t> wd_progress_tick{0};   // bumped once per begin()/end()/acquire_slot()
    std::atomic<uint64_t> wd_last_caller_tid{0}; // pthread/gettid of the last thread to touch the pipeline
    std::atomic<size_t>   wd_last_nbytes{0};     // payload size of the most recent dx begin()
    std::atomic<int>      wd_dx_phase[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_DX_SLOTS] = {}; // 0 idle,1 begun,2 sent,3 recvd,4 ended
    std::atomic<uint64_t> wd_dx_slot_call[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_DX_SLOTS] = {}; // dx_call occupying the slot
    std::thread            wd_thread;
    std::atomic<bool>      wd_stop{false};
    uint64_t                wd_seconds = 0;      // 0 = disabled
    bool                     wd_abort   = false;   // GGML_CUDA_AR_WATCHDOG_ACTION=abort
};


// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

// Cheap "we're alive" marker for the stall watchdog: a relaxed store of an
// already-cached value, no clock read and no syscall.  Called from the
// existing hot-path sites (chunked-kernel acquire_slot, dx begin/end) --
// the watchdog thread (off-path, 1 Hz) is the only reader, and it tolerates
// a torn/stale read since this is diagnostics-only.
static __forceinline__ void ggml_cuda_ar_watchdog_touch(ggml_cuda_ar_pipeline * p) {
    p->wd_progress_tick.fetch_add(1, std::memory_order_relaxed);
#if defined(__linux__)
    p->wd_last_caller_tid.store((uint64_t) syscall(SYS_gettid), std::memory_order_relaxed);
#endif // defined(__linux__)
}

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;
    ggml_cuda_ar_watchdog_touch(p);

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Stall watchdog (GGML_CUDA_AR_WATCHDOG_S / GGML_CUDA_AR_WATCHDOG_ACTION).
//
// Off by default: no thread is started and the only steady-state cost is the
// handful of relaxed atomic stores added at the existing begin()/end()/
// acquire_slot() call sites above, which is noise next to the HIP calls they
// sit beside. Reads below are best-effort diagnostics -- they intentionally
// tolerate a torn/stale value rather than adding any synchronization to the
// hot path.
// ---------------------------------------------------------------------------

// gpu_busy_percent, straight from sysfs, no privileges required.  Returns -1
// if the card doesn't exist or the attribute can't be read (older/unloaded
// driver, non-AMD GPU, etc).
static int ggml_cuda_ar_watchdog_gpu_busy(int card) {
#if defined(__linux__)
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/gpu_busy_percent", card);
    FILE * f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    int pct = -1;
    if (fscanf(f, "%d", &pct) != 1) {
        pct = -1;
    }
    fclose(f);
    return pct;
#else
    (void) card;
    return -1;
#endif // defined(__linux__)
}

static const char * ggml_cuda_ar_watchdog_phase_name(int phase) {
    switch (phase) {
        case 0: return "idle";
        case 1: return "begun";
        case 2: return "sent";
        case 3: return "recvd";
        case 4: return "ended";
        default: return "?";
    }
}

// One self-describing dump of everything cheap we know about the pipeline's
// state, prefixed so journalctl grep for "wp ar-watchdog:" finds it. Called
// from the watchdog thread only -- never on the hot path.
static void ggml_cuda_ar_watchdog_dump(ggml_cuda_ar_pipeline * p, uint64_t stalled_s) {
    fprintf(stderr,
        "wp ar-watchdog: STALL detected, no progress for %llu s (threshold %llu s)\n",
        (unsigned long long) stalled_s, (unsigned long long) p->wd_seconds);
    fprintf(stderr,
        "wp ar-watchdog: transport=%s wire_codec=%s last_dx_nbytes=%zu "
        "dx_call=%llu dx_in_flight=%d call_count=%llu progress_tick=%llu "
        "last_caller_tid=%llu watchdog_tid=%llu\n",
        p->transport == GGML_CUDA_AR_TRANSPORT_P2P  ? "p2p"  :
        p->transport == GGML_CUDA_AR_TRANSPORT_HOST ? "host" : "copy",
        p->wire_codec ? p->wire_codec->name : "?",
        p->wd_last_nbytes.load(std::memory_order_relaxed),
        (unsigned long long) p->dx_call,
        p->dx_in_flight,
        (unsigned long long) p->call_count,
        (unsigned long long) p->wd_progress_tick.load(std::memory_order_relaxed),
        (unsigned long long) p->wd_last_caller_tid.load(std::memory_order_relaxed),
#if defined(__linux__)
        (unsigned long long) syscall(SYS_gettid));
#else
        0ull);
#endif // defined(__linux__)

    for (int i = 0; i < p->n_devices; ++i) {
        for (int s = 0; s < GGML_CUDA_AR_DX_SLOTS; ++s) {
            const int phase = p->wd_dx_phase[i][s].load(std::memory_order_relaxed);
            fprintf(stderr,
                "wp ar-watchdog: dx_slot rank=%d slot=%d op=%llu phase=%s "
                "sent_valid=%d recvd_valid=%d freed_valid=%d\n",
                i, s,
                (unsigned long long) p->wd_dx_slot_call[i][s].load(std::memory_order_relaxed),
                ggml_cuda_ar_watchdog_phase_name(phase),
                (int) p->dx_ev[i][s].sent_valid,
                (int) p->dx_ev[i][s].recvd_valid,
                (int) p->dx_ev[i][s].freed_valid);
        }
    }

    // Chunked-kernel arrival ring: host-mapped, readable without a device
    // sync (see the GGML_CUDA_AR_HOST_COHERENT comment at alloc()). Dump the
    // most recently issued slot plus its immediate predecessor for both
    // ranks -- covers the in-flight chunked AR, if any.
    if (p->arrival.host && p->call_count > 0) {
        for (int back = 0; back < 2 && (int64_t) p->call_count - 1 - back >= 0; ++back) {
            const int slot = (int) ((p->call_count - 1 - (uint64_t) back) % GGML_CUDA_AR_POOL_SIZE);
            for (int i = 0; i < p->n_devices; ++i) {
                fprintf(stderr, "wp ar-watchdog: arrival rank=%d slot=%d blocks=[",
                        i, slot);
                // ggml_cuda_ar_arrival_ptr(p, slot, rank) returns the base for
                // block 0; each block's token lives ARRIVAL_STRIDE bytes later.
                const int * base = ggml_cuda_ar_arrival_ptr(p, slot, i);
                for (int b = 0; b < GGML_CUDA_AR_KERNEL_BLOCKS; ++b) {
                    const int * word = base + (size_t) b * (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));
                    fprintf(stderr, "%d%s", *word, b + 1 < GGML_CUDA_AR_KERNEL_BLOCKS ? "," : "");
                }
                fprintf(stderr, "]\n");
            }
        }
    }

    for (int card = 0; card < 8; ++card) {
        const int busy = ggml_cuda_ar_watchdog_gpu_busy(card);
        if (busy < 0) {
            continue;
        }
        fprintf(stderr, "wp ar-watchdog: gpu_busy_percent card=%d value=%d\n", card, busy);
    }
}

static void ggml_cuda_ar_watchdog_main(ggml_cuda_ar_pipeline * p) {
    uint64_t last_tick     = p->wd_progress_tick.load(std::memory_order_relaxed);
    uint64_t stalled_since = 0; // seconds; 0 == not currently stalled
    bool     dumped_this_stall = false;

    while (!p->wd_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (p->wd_stop.load(std::memory_order_relaxed)) {
            break;
        }

        const uint64_t tick = p->wd_progress_tick.load(std::memory_order_relaxed);
        const bool     in_flight = p->dx_in_flight > 0;

        if (tick != last_tick) {
            // Progress since last check -- re-arm.
            last_tick          = tick;
            stalled_since       = 0;
            dumped_this_stall   = false;
            continue;
        }

        if (!in_flight) {
            // Nothing outstanding -- idle, not stalled.
            stalled_since = 0;
            dumped_this_stall = false;
            continue;
        }

        stalled_since++;
        if (stalled_since >= p->wd_seconds && !dumped_this_stall) {
            dumped_this_stall = true;
            ggml_cuda_ar_watchdog_dump(p, stalled_since);
            if (p->wd_abort) {
                fprintf(stderr, "wp ar-watchdog: GGML_CUDA_AR_WATCHDOG_ACTION=abort, aborting\n");
                fflush(stderr);
                abort();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

#if defined(GGML_USE_HIP)
    // HIP: there is no cc gate equivalent to the Volta/__nanosleep one below --
    // s_sleep and system-scope load/store atomics exist on every supported
    // gfx target, and the wire path only ever *converts* through bf16 (via
    // ggml_cuda_cast), never does bf16 arithmetic, so bf16-arithmetic-less
    // parts like gfx1030 (RDNA2) are fine.  What we do need is host-memory
    // mapping, so query it directly and fall back if it is unavailable.
    for (size_t i = 0; i < n_devices; ++i) {
        int can_map = 0;
        const cudaError_t rc = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, devices[i]);
        if (rc != cudaSuccess || !can_map) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires host-memory mapping "
                           "(device %d: rc=%d can_map=%d); falling back\n",
                           __func__, devices[i], (int) rc, can_map);
            return nullptr;
        }
    }
#else
    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }
#endif // defined(GGML_USE_HIP)

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    const char * wire_env = getenv("GGML_CUDA_AR_WIRE_TYPE");
    p->wire_codec_explicit = wire_env && wire_env[0];
    p->wire_codec = ggml_cuda_ar_codec_from_name(wire_env && wire_env[0] ? wire_env : "bf16");
    if (p->wire_codec == nullptr) {
        GGML_ABORT("%s: unknown GGML_CUDA_AR_WIRE_TYPE='%s' (expected bf16, f16, f32, q8_0, ml8_4, ml8_5, ml8_8, or ml8_8r)",
                   __func__, wire_env ? wire_env : "");
    }
    if (p->wire_codec->elements_per_block == 0 || p->wire_codec->bytes_per_block == 0 ||
        p->wire_codec->pack_fn == nullptr || p->wire_codec->unpack_accumulate_fn == nullptr) {
        GGML_ABORT("%s: invalid AllReduce codec '%s'", __func__, p->wire_codec->name);
    }
    p->dx_bytes = (GGML_CUDA_AR_COPY_MAX_BYTES / p->wire_codec->bytes_per_block) *
                  p->wire_codec->bytes_per_block;
    if (p->dx_bytes == 0 || p->dx_bytes % p->wire_codec->bytes_per_block != 0) {
        GGML_ABORT("%s: codec '%s' does not fit the duplex slot", __func__, p->wire_codec->name);
    }
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        p->streams[i] = stream;

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].h2d, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            for (int c = 0; ok && c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                ok = cudaEventCreateWithFlags(&p->ev_pool[i][s].cpy[c], cudaEventDisableTiming) == cudaSuccess;
            }
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        if (cudaEventCreateWithFlags(&p->host_large_read_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaEventCreateWithFlags(&p->dev_tmp_kernel_done[i], cudaEventDisableTiming) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    // Device-coherent: polled from inside the chunked kernel by both GPUs.
    if (p->arrival.alloc(arrival_bytes, GGML_CUDA_AR_HOST_COHERENT) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
#if defined(GGML_USE_HIP)
    // HIP: zero the ring through the host handle.  This is init time -- no
    // device work references the ring yet -- and the mapping is fine-grained,
    // so plain host stores are the cheapest and most portable way to do it.
    // (hipMemset on a host-mapped pointer is accepted but relies on the
    // pointer-attribute lookup classifying it as device-accessible.)
    std::memset(p->arrival.host, 0, arrival_bytes);
#else
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
#endif // defined(GGML_USE_HIP)

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        // Device-coherent: written and read directly by the chunked kernel on
        // both GPUs (no copy engine involved on this path).
        if (p->host_buf[i].alloc(host_buf_total, GGML_CUDA_AR_HOST_COHERENT) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copy-engine path: pinned host staging + device scratch, sized for the
    // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
    // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
    // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // The chunked kernel running on device i dereferences host_buf[peer].dev and
    // the arrival ring, i.e. mapped-host device pointers obtained while some
    // *other* device was current.  That is only sound if the mapping is a single
    // process-wide virtual address (unified addressing), which both CUDA UVA and
    // ROCm provide -- in which case cudaHostGetDevicePointer returns the host
    // pointer unchanged.  If a runtime ever hands back a per-device alias,
    // cross-device use would silently read the wrong memory, so bail out to the
    // generic AllReduce instead.
    {
        const ggml_cuda_ar_host_mapping * shared[] = { &p->arrival, &p->host_buf[0], &p->host_buf[1] };
        for (const ggml_cuda_ar_host_mapping * m : shared) {
            if (m->dev != m->host) {
                GGML_LOG_WARN("%s: mapped host pointer is not device-uniform (host=%p dev=%p); "
                              "cross-device access would be unsafe -- falling back\n",
                              __func__, (void *) m->host, (void *) m->dev);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }
    }

    // Duplex transport selection.  GGML_CUDA_AR_TRANSPORT = p2p | host | copy.
    // Default: p2p when device 1 can enable peer access to device 0's memory,
    // else host.  "copy" is the legacy synchronous copy-engine path, kept
    // selectable so a baseline can be re-run.
    {
        bool peer_ok = false;
        {
            int can = 0;
            ggml_cuda_set_device(p->devices[1]);
            if (cudaDeviceCanAccessPeer(&can, p->devices[1], p->devices[0]) == cudaSuccess && can) {
                cudaError_t rc = cudaDeviceEnablePeerAccess(p->devices[0], 0);
                if (rc == cudaErrorPeerAccessAlreadyEnabled) {
                    rc = cudaSuccess;
                }
                (void) cudaGetLastError();
                peer_ok = rc == cudaSuccess;
            } else {
                (void) cudaGetLastError();
            }
        }

        const char * env = getenv("GGML_CUDA_AR_TRANSPORT");
        std::string  env_str = env ? env : "";
        if (env_str == "copy") {
            p->transport = GGML_CUDA_AR_TRANSPORT_COPY;
        } else if (env_str == "host") {
            p->transport = GGML_CUDA_AR_TRANSPORT_HOST;
        } else if (env_str == "p2p") {
            if (!peer_ok) {
                GGML_LOG_WARN("%s: GGML_CUDA_AR_TRANSPORT=p2p but device %d cannot access device %d memory; using host\n",
                              __func__, p->devices[1], p->devices[0]);
            }
            p->transport = peer_ok ? GGML_CUDA_AR_TRANSPORT_P2P : GGML_CUDA_AR_TRANSPORT_HOST;
        } else {
            if (!env_str.empty()) {
                GGML_LOG_WARN("%s: unknown GGML_CUDA_AR_TRANSPORT value '%s'; using default\n", __func__, env);
            }
            p->transport = peer_ok ? GGML_CUDA_AR_TRANSPORT_P2P : GGML_CUDA_AR_TRANSPORT_HOST;
        }

        // The duplex transports keep the same copy_threshold as the legacy
        // path: below it, small reductions still take the chunked kernel.
        //
        // This was previously forced to 0 here, routing every size class
        // through the async path, on the theory that the duplex transports
        // made the chunked kernel redundant.  Measured on 2026-09-08
        // (mad-lab-main, R9700 + RX 6900 XT over TB3, Qwen3.8-27B Q8,
        // -sm tensor -ts 68/32), that cost decode and bought nothing:
        //
        //   arm                      pp512 t/s   tg128 t/s
        //   copy (threshold 1 MiB)      612        20.66
        //   p2p  (threshold 0)          730        17.85
        //   p2p  (threshold 1 MiB)      729        20.76
        //   host (threshold 0)          727        14.26
        //
        // The decode reduction is one token x n_embd (10 KB on the bf16 wire
        // at n_embd=5120), ~100x below the threshold, so zeroing it moved
        // decode off the chunked kernel -- whose in-kernel spin exists
        // precisely for that latency-sensitive shape -- and onto the async
        // path, for a flat ~7.4 ms/token penalty at every KV depth.  Prefill
        // is unaffected either way: at n_ubatch 512 the reduction is 5.2 MB,
        // above the threshold, so it takes the duplex path regardless.  The
        // crossover sits at ~102 tokens.
    }

    p->dx_call      = 0;
    p->dx_in_flight = 0;
    if (p->transport != GGML_CUDA_AR_TRANSPORT_COPY) {
        const size_t dx_total = (size_t) GGML_CUDA_AR_DX_SLOTS * p->dx_bytes;
        for (size_t i = 0; i < n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            if (cudaStreamCreateWithFlags(&p->streams_in[i], cudaStreamNonBlocking) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags (in) failed for device %d\n", __func__, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            if (cudaMalloc(reinterpret_cast<void **>(&p->dx_send[i]), dx_total) != cudaSuccess ||
                cudaMalloc(reinterpret_cast<void **>(&p->dx_recv[i]), dx_total) != cudaSuccess) {
                GGML_LOG_ERROR("%s: cudaMalloc for duplex send/recv failed (%zu bytes) on device %d\n",
                               __func__, dx_total, p->devices[i]);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            // SDMA-only staging: default coherence, ordering comes from events.
            if (p->dx_staging[i].alloc(dx_total) != cudaSuccess) {
                GGML_LOG_ERROR("%s: alloc for duplex staging failed (%zu bytes)\n", __func__, dx_total);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
            for (int s = 0; s < GGML_CUDA_AR_DX_SLOTS; ++s) {
                ggml_cuda_ar_dx_slot & ev = p->dx_ev[i][s];
                const bool ok =
                    cudaEventCreateWithFlags(&ev.app,   cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&ev.sent,  cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&ev.recvd, cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventCreateWithFlags(&ev.freed, cudaEventDisableTiming) == cudaSuccess;
                if (!ok) {
                    GGML_LOG_ERROR("%s: cudaEventCreate (duplex) failed for device %d slot %d\n",
                                   __func__, p->devices[i], s);
                    ggml_cuda_ar_pipeline_free(p);
                    return nullptr;
                }
            }
        }
    }

    const char * transport_name =
        p->transport == GGML_CUDA_AR_TRANSPORT_P2P  ? "p2p (r1 pushes into r0, r1 pulls via host staging)" :
        p->transport == GGML_CUDA_AR_TRANSPORT_HOST ? "host (duplex via host staging)" :
                                                      "copy (legacy copy-engine bounce)";
    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU, "
                  "transport=%s, copy_threshold=%zu, wire_codec=%s (%.1f bits/element)\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20,
                  transport_name, p->copy_threshold, p->wire_codec->name,
                  8.0 * p->wire_codec->bytes_per_block / p->wire_codec->elements_per_block);

    // Stall watchdog: off unless GGML_CUDA_AR_WATCHDOG_S is set. Started
    // last, after everything the dump reads from is already initialized.
    p->wd_seconds = ggml_cuda_ar_env_u64("GGML_CUDA_AR_WATCHDOG_S", 0);
    if (p->wd_seconds > 0) {
        const char * action = getenv("GGML_CUDA_AR_WATCHDOG_ACTION");
        p->wd_abort = action && std::string(action) == "abort";
        p->wd_stop.store(false, std::memory_order_relaxed);
        p->wd_thread = std::thread(ggml_cuda_ar_watchdog_main, p);
        // GGML_LOG_INFO is dropped at llama-server's default verbosity --
        // common_log_get_verbosity() maps GGML_LOG_LEVEL_INFO to
        // LOG_LEVEL_TRACE (4), which is above the LOG_DEFAULT_LLAMA (3)
        // threshold, so common_log_default_callback() never queues it. Use
        // the same raw fprintf(stderr, ...) the "wp hip-graphs" stats line
        // uses so this reaches the journal unconditionally, like that line
        // does.
        fprintf(stderr, "wp ar-watchdog: armed, N=%llu s, action=%s\n",
                      (unsigned long long) p->wd_seconds, p->wd_abort ? "abort" : "dump");
    }

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    if (p->wd_thread.joinable()) {
        p->wd_stop.store(true, std::memory_order_relaxed);
        p->wd_thread.join();
    }

    // Drain all in-flight kernels before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
        if (p->streams_in[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams_in[i]);
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->dx_send[i]) { cudaFree(p->dx_send[i]); }
        if (p->dx_recv[i]) { cudaFree(p->dx_recv[i]); }
        p->dx_staging[i].free();
        for (int s = 0; s < GGML_CUDA_AR_DX_SLOTS; ++s) {
            ggml_cuda_ar_dx_slot & ev = p->dx_ev[i][s];
            if (ev.app)   { cudaEventDestroy(ev.app); }
            if (ev.sent)  { cudaEventDestroy(ev.sent); }
            if (ev.recvd) { cudaEventDestroy(ev.recvd); }
            if (ev.freed) { cudaEventDestroy(ev.freed); }
        }
        if (p->streams_in[i]) { cudaStreamDestroy(p->streams_in[i]); }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            for (int c = 0; c < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++c) {
                if (p->ev_pool[i][s].cpy[c]) { cudaEventDestroy(p->ev_pool[i][s].cpy[c]); }
            }
            if (p->ev_pool[i][s].h2d) { cudaEventDestroy(p->ev_pool[i][s].h2d); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        if (p->host_large_read_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->host_large_read_done[i]);
        }
        if (p->dev_tmp_kernel_done[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaEventDestroy(p->dev_tmp_kernel_done[i]);
        }
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamDestroy(p->streams[i]);
        }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    // Stage 1: both GPUs copy their local contribution to pinned host memory.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        // Wait for peer's H2D from our host_large[i] (recorded in the
        // previous AR's stage 2) to complete before we overwrite host_large[i].
        // host_large_read_done[peer] = peer finished reading host_large[i].
        // No-op on the first AR -- no prior record exists.
        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        if (!compute[i]) {
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
    // local device scratch (dev_tmp), then performs one device-local add over
    // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
    // and the add_kernel runs on the caller's compute stream, so the AR stream
    // stays pure-copy and avoids an in-stream copy->compute engine switch every
    // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
    // prior AR's add_kernel-done event before overwriting it.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        // Wait for the previous AR's add_kernel (on the compute stream) to
        // finish reading dev_tmp before our H2D overwrites it.  No-op on the
        // first copy_impl call.
        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = (nbytes - offset) < chunk_bytes ?
                (nbytes - offset) : chunk_bytes;

            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        // Mark our reads of host_large[peer] complete so peer's next AR can
        // safely overwrite it.
        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));

        // Hand off from AR stream (copy engine) to compute stream: compute
        // stream waits for all H2Ds to finish, then runs the add_kernel.
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i],
            reinterpret_cast<const T_src *>(p->dev_tmp[i]),
            (int) ne);
        CUDA_CHECK(cudaGetLastError());

        // Record dev_tmp-released on the compute stream so the next copy_impl
        // can wait for the kernel to finish before overwriting dev_tmp.  Also
        // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }
    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;

    return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    const int64_t outer_max_elems = (int64_t) (p->copy_bytes / sizeof(T_src));
    GGML_ASSERT(outer_max_elems > 0);

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne     = std::min(outer_max_elems, ne - outer_start);
        const size_t  outer_nbytes = (size_t) outer_ne * sizeof(T_src);

        T_src * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>(
            p, backends, src, dst, compute, outer_ne, outer_nbytes);
    }
    return ok;
}

// Legacy synchronous entry: chunked spin kernel (small) or copy-engine host
// bounce (large), selected by use_copy_engine.  Everything is ordered on the
// caller's compute streams before this returns (host event syncs inside
// acquire_slot), so it may also serve as the fallback for calls the duplex
// transport cannot take asynchronously.
static bool ggml_cuda_ar_allreduce_sync(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const bool              use_copy_engine) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne);
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<float, float>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_outer<half, half>(
                        p, backends, buf, buf, inner_compute, ne);
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // host_buf carries T_wire-typed data; max_chunk_elems is the count that
        // fits in one host_buf at the wire size.
        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
        // since AR is a barrier here, same-stream ordering subsumes any
        // cross-stream event handshake that the copy-engine path needs, and
        // skips the cross-stream scheduling overhead that was hurting the
        // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
        // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
        for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
            const size_t remaining_elems = (size_t) (ne - chunk_start);
            const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();

                char * data = static_cast<char *>(tensors[i]->data) + chunk_start * (int64_t) input_type_size;

                // Match NCCL/meta-backend semantics: inactive shards contribute
                // zeros.  On the BF16 path the F32 tensor data was already
                // zeroed up-front (above), so per-chunk zeroing isn't needed.
                if (!compute_flag[i] && !use_bf16) {
                    CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
                }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                    reinterpret_cast<const T_dst *>(data), \
                    reinterpret_cast<T_dst *>(data), \
                    reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes), \
                    reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                    static_cast<int>(chunk_elems), \
                    ggml_cuda_ar_arrival_ptr(p, slot, i), \
                    ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                    token)

                if (use_bf16) {
                    GGML_ASSERT(input_type == GGML_TYPE_F32);
                    LAUNCH_AR_KERNEL(float, nv_bfloat16);
                } else {
                    switch (input_type) {
                        case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,       float);       break;
                        case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,        half);        break;
                        case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                        default: GGML_ASSERT(false);
                    }
                }

#undef LAUNCH_AR_KERNEL
                CUDA_CHECK(cudaGetLastError());

                if (last_chunk) {
                    CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
                }
            }
        }
    }

    return ok;
}

// ---------------------------------------------------------------------------
// Duplex transport: begin / end
// ---------------------------------------------------------------------------

template <typename T_dst, typename T_src>
static void ggml_cuda_ar_dx_add(T_dst * dst, const T_src * src, int64_t ne, cudaStream_t stream) {
    const int block_size = 256;
    int n_blocks = (int) ((ne + block_size - 1) / block_size);
    if (n_blocks > 1024) {
        n_blocks = 1024;
    }
    ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, stream>>>(dst, src, (int) ne);
    CUDA_CHECK(cudaGetLastError());
}

// ── PHASE 0 CEILING MEASUREMENT ONLY — DELIBERATELY MATH-INVALID ──────────
//
// GGML_CUDA_AR_WIRE_SHRINK=N transfers only nbytes/N across the link while
// leaving the send buffers, the BF16 conversion and the add kernel at full
// size, so the tail of every payload is whatever the receive slot held last.
// OUTPUT IS GARBAGE BY CONSTRUCTION.
//
// Purpose: measure the throughput ceiling of ANY wire-compression scheme
// before writing an encoder.  Encode+decode cost would land on the
// TB3-attached 6900XT, which is already the contended card, so "fewer bytes
// is faster" is a hypothesis, not a given.  If a 4x byte cut does not buy
// meaningful throughput here, no encoder is worth writing.
//
// Never set this outside a benchmark.
static size_t ggml_cuda_ar_wire_shrink() {
    static const size_t shrink = []() -> size_t {
        const uint64_t v = ggml_cuda_ar_env_u64("GGML_CUDA_AR_WIRE_SHRINK", 1);
        return v < 1 ? 1 : (size_t) v;
    }();
    return shrink;
}

// ── FIDELITY CAPTURE (debug) ─────────────────────────────────────────────
//
// GGML_CUDA_AR_DUMP_PARTIALS=<dir> writes the F32 partials from BOTH ranks for
// the first GGML_CUDA_AR_DUMP_N reduces (default 8) as raw little-endian f32:
//   <dir>/ar<call>_r<rank>.f32
// plus a one-line manifest per reduce recording ne and the element count kept.
//
// These are the exact tensors a wire codec would have to encode.  The point of
// capturing REAL partials rather than synthesising activations is that the
// error a block codec makes depends on the value distribution, and the whole
// question is whether a fitted or outlier-preserving codec beats a flat one on
// THIS data.  It is also the only way to measure the error entering TWICE
// (both partials rounded before summing) which is what rank symmetry costs.
//
// Cost when unset: one getenv-backed static read per call. Off by default.
static const char * ggml_cuda_ar_dump_dir() {
    static const char * dir = getenv("GGML_CUDA_AR_DUMP_PARTIALS");
    return (dir != nullptr && dir[0] != '\0') ? dir : nullptr;
}

static void ggml_cuda_ar_dump_partials(
        ggml_cuda_ar_pipeline * p, ggml_tensor ** tensors, int n, int64_t ne) {
    const char * dir = ggml_cuda_ar_dump_dir();
    if (dir == nullptr) {
        return;
    }
    static const uint64_t max_calls = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DUMP_N", 8);
    // Cap per-reduce volume; block codecs need CONTIGUOUS elements, so keep a
    // prefix rather than a stride sample.
    static const uint64_t max_elts  = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DUMP_ELTS", 1u << 20);
    // Sample ACROSS DEPTH, not the first N. Reduces arrive in layer order, so
    // dumping the first N captures only early layers -- and the outlier
    // structure these codecs exploit is layer-dependent (the OL channel sets
    // were calibrated on activations POOLED ACROSS ALL LAYERS). Ranking codecs
    // on layers 0-8 could invert the answer. Stride so the sample spans the
    // model: with 2 reduces/layer x 64 layers x 8 ubatches, stride 37 walks the
    // whole prefill.
    static const uint64_t stride = ggml_cuda_ar_env_u64("GGML_CUDA_AR_DUMP_STRIDE", 1);
    static uint64_t seen = 0, call = 0;
    const uint64_t this_seen = seen++;
    if (stride > 1 && (this_seen % stride) != 0) {
        return;
    }
    if (call >= max_calls) {
        return;
    }
    const uint64_t idx  = call++;
    const size_t   keep = (size_t) std::min<uint64_t>((uint64_t) ne, max_elts);

    std::vector<float> host(keep);
    for (int i = 0; i < n; ++i) {
        if (tensors[i]->type != GGML_TYPE_F32) {
            return;  // only the F32 partial case is interesting here
        }
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host.data(), tensors[i]->data, keep * sizeof(float),
                              cudaMemcpyDeviceToHost));
        char path[1024];
        snprintf(path, sizeof(path), "%s/ar%04llu_seen%06llu_r%d.f32", dir,
                 (unsigned long long) idx, (unsigned long long) this_seen, i);
        FILE * f = fopen(path, "wb");
        if (f == nullptr) {
            GGML_LOG_WARN("ggml_cuda_ar: cannot open %s for partial dump\n", path);
            return;
        }
        fwrite(host.data(), sizeof(float), keep, f);
        fclose(f);
    }
    GGML_LOG_INFO("ggml_cuda_ar: dumped partials %llu of reduce #%llu (ne=%lld, kept=%zu/rank)\n",
                  (unsigned long long) idx, (unsigned long long) this_seen, (long long) ne, keep);
}

bool ggml_cuda_ar_allreduce_begin(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        ggml_cuda_ar_op       * op) {
    GGML_ASSERT(p != nullptr);
    GGML_ASSERT(op != nullptr);
    op->pending = false;

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t input_nbytes = ggml_nbytes(tensors[0]);

    // Keep path selection identical to the synchronous path.  Codec packing is
    // only used by the duplex path; decode-sized reductions stay unchanged.
    const bool legacy_use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;
    const ggml_type legacy_wire_type = legacy_use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    legacy_nbytes = (size_t) ne * ggml_type_size(legacy_wire_type);

    const bool chunked = p->copy_threshold > 0 && legacy_nbytes < p->copy_threshold;

    if (p->transport == GGML_CUDA_AR_TRANSPORT_COPY) {
        // Legacy semantics, unchanged: copy engine at/above the threshold, chunked kernel below.
        return ggml_cuda_ar_allreduce_sync(p, backends, tensors, !chunked);
    }
    if (chunked) {
        // Chunked kernel by explicit request.  It does not use the duplex codec.
        return ggml_cuda_ar_allreduce_sync(p, backends, tensors, !chunked);
    }

    const ggml_cuda_ar_codec * codec = p->wire_codec;
    if (!p->wire_codec_explicit) {
        if (input_type != GGML_TYPE_F32) {
            codec = ggml_cuda_ar_codec_from_type(input_type);
        } else if (!legacy_use_bf16) {
            codec = ggml_cuda_ar_codec_from_type(GGML_TYPE_F32);
        }
    }
    GGML_ASSERT(codec != nullptr);
    if (ne % (int64_t) codec->elements_per_block != 0) {
        GGML_ABORT("%s: codec '%s' requires ne divisible by %zu (got %lld)",
                   __func__, codec->name, codec->elements_per_block, (long long) ne);
    }
    const size_t nbytes = (size_t) ne / codec->elements_per_block * codec->bytes_per_block;
    if (nbytes > p->dx_bytes) {
        if (!p->wire_codec_explicit) {
            // Preserve the default oversized-payload fallback.
            return ggml_cuda_ar_allreduce_sync(p, backends, tensors, true);
        }
        GGML_ABORT("%s: codec '%s' payload %zu exceeds duplex slot %zu",
                   __func__, codec->name, nbytes, p->dx_bytes);
    }

    // Phase 0 only: bytes actually put on the wire.  Equals nbytes normally.
    const size_t shrink      = ggml_cuda_ar_wire_shrink();
    const size_t xfer_nbytes = shrink > 1 ? std::max<size_t>(nbytes / shrink, 1) : nbytes;
    if (shrink > 1) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("ggml_cuda_ar: GGML_CUDA_AR_WIRE_SHRINK=%zu ACTIVE - "
                          "transferring %zu of %zu bytes per reduce. OUTPUT IS INVALID. "
                          "Benchmark use only.\n", shrink, xfer_nbytes, nbytes);
        }
    }

    GGML_ASSERT(p->dx_in_flight < GGML_CUDA_AR_DX_SLOTS && "too many in-flight AllReduce ops (end() the oldest first)");

    const int slot = (int) (p->dx_call % GGML_CUDA_AR_DX_SLOTS);
    p->dx_call++;
    p->dx_in_flight++;
    p->wd_last_nbytes.store(nbytes, std::memory_order_relaxed);
    ggml_cuda_ar_watchdog_touch(p);

    const size_t slot_off = (size_t) slot * p->dx_bytes;
    const bool   p2p      = p->transport == GGML_CUDA_AR_TRANSPORT_P2P;

    ggml_backend_cuda_context * cuda_ctx[2] = {};
    bool compute_flag[2] = {};
    for (int i = 0; i < n; ++i) {
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Phase A (compute streams): materialise the wire-typed partial in
    // dx_send[i][slot].  Inactive shards contribute zeros, and their F32
    // accumulator is zeroed too since the add kernel accumulates in place.
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t cs   = cuda_ctx[i]->stream();
        char *       send = p->dx_send[i] + slot_off;
        if (!compute_flag[i]) {
            CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, input_nbytes, cs));
            CUDA_CHECK(cudaMemsetAsync(send, 0, nbytes, cs));
        } else {
            codec->pack_fn(tensors[i]->data, input_type, send, ne, cs);
        }
        CUDA_CHECK(cudaEventRecord(p->dx_ev[i][slot].app, cs));
        p->wd_dx_slot_call[i][slot].store(p->dx_call, std::memory_order_relaxed);
        p->wd_dx_phase[i][slot].store(1 /* begun */, std::memory_order_relaxed);
    }

    ggml_cuda_ar_dump_partials(p, tensors, n, ne);

    // Phase B (out-streams): outbound transfers, both ranks back-to-back.
    for (int i = 0; i < n; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t          out  = p->streams[i];
        ggml_cuda_ar_dx_slot & ev  = p->dx_ev[i][slot];
        const char *          send = p->dx_send[i] + slot_off;

        CUDA_CHECK(cudaStreamWaitEvent(out, ev.app));
        if (p2p && i == 1) {
            // Push straight into r0's receive slot.  r0's add kernel from the
            // op that last used this slot must be done reading it.
            if (p->dx_ev[0][slot].freed_valid) {
                CUDA_CHECK(cudaStreamWaitEvent(out, p->dx_ev[0][slot].freed));
            }
            CUDA_CHECK(cudaMemcpyPeerAsync(
                p->dx_recv[0] + slot_off, p->devices[0], send, p->devices[1], xfer_nbytes, out));
        } else {
            // D2H into our staging slot.  The peer's in-stream must be done
            // pulling the previous contents of this staging slot.
            if (p->dx_ev[peer][slot].recvd_valid) {
                CUDA_CHECK(cudaStreamWaitEvent(out, p->dx_ev[peer][slot].recvd));
            }
            CUDA_CHECK(cudaMemcpyAsync(
                p->dx_staging[i].host + slot_off, send, xfer_nbytes, cudaMemcpyDeviceToHost, out));
        }
        CUDA_CHECK(cudaEventRecord(ev.sent, out));
        ev.sent_valid = true;
        p->wd_dx_phase[i][slot].store(2 /* sent */, std::memory_order_relaxed);
    }

    // Phase C (in-streams): pulls from the peer's staging for the ranks that
    // are not pushed into (r1 on p2p; both on host).
    for (int i = 0; i < n; ++i) {
        if (p2p && i == 0) {
            continue;
        }
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        cudaStream_t          in = p->streams_in[i];
        ggml_cuda_ar_dx_slot & ev = p->dx_ev[i][slot];

        CUDA_CHECK(cudaStreamWaitEvent(in, p->dx_ev[peer][slot].sent)); // cross-device
        if (ev.freed_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(in, ev.freed));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            p->dx_recv[i] + slot_off, p->dx_staging[peer].host + slot_off, xfer_nbytes, cudaMemcpyHostToDevice, in));
        CUDA_CHECK(cudaEventRecord(ev.recvd, in));
        ev.recvd_valid = true;
        p->wd_dx_phase[i][slot].store(3 /* recvd */, std::memory_order_relaxed);
    }

    op->pending   = true;
    op->slot      = slot;
    op->ne        = ne;
    op->dst_type  = input_type;
    op->wire_type = codec->wire_type;
    for (int i = 0; i < n; ++i) {
        op->dst[i] = tensors[i]->data;
    }
    return true;
}

bool ggml_cuda_ar_allreduce_end(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_cuda_ar_op       * op) {
    GGML_ASSERT(p != nullptr);
    GGML_ASSERT(op != nullptr);
    if (!op->pending) {
        return true;
    }
    GGML_ASSERT(p->transport != GGML_CUDA_AR_TRANSPORT_COPY);
    GGML_ASSERT(p->dx_in_flight > 0);

    const int    n        = p->n_devices;
    const int    slot     = op->slot;
    const size_t slot_off = (size_t) slot * p->dx_bytes;
    const bool   p2p      = p->transport == GGML_CUDA_AR_TRANSPORT_P2P;

    for (int i = 0; i < n; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx->device == p->devices[i]);
        cudaStream_t          cs = cuda_ctx->stream();
        ggml_cuda_ar_dx_slot & ev = p->dx_ev[i][slot];

        // Own outbound done (send slot reusable, and for r0/p2p its staging
        // read is what the peer's pull ordered on) + peer data landed.
        CUDA_CHECK(cudaStreamWaitEvent(cs, ev.sent));
        if (p2p && i == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(cs, p->dx_ev[peer][slot].sent)); // r1's push
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(cs, ev.recvd));
        }

        const char * recv = p->dx_recv[i] + slot_off;
        const ggml_cuda_ar_codec * codec = ggml_cuda_ar_codec_from_type(op->wire_type);
        if (codec == nullptr) {
            GGML_ABORT("%s: no codec registered for wire type %d", __func__, (int) op->wire_type);
        }
        codec->unpack_accumulate_fn(op->dst[i], op->dst_type, recv, op->ne, cs);

        CUDA_CHECK(cudaEventRecord(ev.freed, cs));
        ev.freed_valid = true;
        p->wd_dx_phase[i][slot].store(4 /* ended */, std::memory_order_relaxed);
    }

    p->dx_in_flight--;
    op->pending = false;
    ggml_cuda_ar_watchdog_touch(p);
    return true;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    ggml_cuda_ar_op op;
    if (!ggml_cuda_ar_allreduce_begin(p, backends, tensors, &op)) {
        return false;
    }
    return ggml_cuda_ar_allreduce_end(p, backends, &op);
}

#else // defined(GGML_USE_MUSA)

// MUSA has not been audited for the host-mapped pinned-memory APIs
// (cudaHostAllocPortable / cudaHostAllocMapped / cudaHostGetDevicePointer),
// the in-kernel sleep primitive, or the system-scope atomics this
// implementation relies on, so the internal AllReduce is disabled there.
// (HIP/ROCm *is* supported -- see the main implementation above.)
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}
bool ggml_cuda_ar_allreduce_begin(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **, ggml_cuda_ar_op *) {
    return false;
}
bool ggml_cuda_ar_allreduce_end(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_cuda_ar_op *) {
    return false;
}

#endif // !defined(GGML_USE_MUSA)
