// mt_pagedattn_r4d_scatter.cu — device-side scatter for the libr4d paged fp8
// KV cache (GGML_TYPE_R4D_FP8_KV). See the header for the layout contract.
//
// Only compiled when ggml-hip is built with -DGGML_HIP_R4D=ON (gfx1201).

#include "mt_pagedattn_r4d_scatter.cuh"

#if defined(GGML_HIP_R4D) && defined(GGML_USE_HIP)

#include <cstring>

namespace mt {

// Elements converted per thread. 8 F16 elements = 16 bytes (one uint4 load,
// naturally aligned since head_dim and block strides are all multiples of
// 8 in every shape this type is used for); 8 e4m3fn bytes = 8 bytes (one
// uint64_t store).
static constexpr int R4D_SCATTER_VEC = 8;

// Software fp32 -> OCP e4m3fn (round-to-nearest-even, saturate to +-448, no
// infinities; NaN input -> S.1111.111). Same bit-level algorithm as the CPU
// reference (ggml-turbo-quant.c:quantize_row_f8_e4m3_ref) and the existing
// ml8.cu device codec -- kept as an independent copy here since this file
// must not depend on ml8.cu (owned by another workstream).
static __device__ __forceinline__ uint8_t r4d_f32_to_e4m3fn_sw(float xv) {
    uint32_t bits;
    memcpy(&bits, &xv, 4);
    const uint32_t sign  = (bits >> 31) & 1u;
    const uint32_t exp_b = (bits >> 23) & 0xFFu;
    const uint32_t mant  = bits & 0x7FFFFFu;

    // NaN or Inf input -> e4m3 NaN (S.1111.111).
    if (exp_b == 0xFFu) {
        return (uint8_t) ((sign << 7) | 0x7Fu);
    }
    // Zero (fp32 subnormals underflow to e4m3 zero too).
    if (exp_b == 0) {
        return (uint8_t) (sign << 7);
    }

    const int32_t e_un = (int32_t) exp_b - 127;

    // Saturate to +-448 = e=15, m=6.
    if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u)) {
        return (uint8_t) ((sign << 7) | (0xFu << 3) | 0x6u);
    }

    if (e_un >= -6) {
        // Normal e4m3: e in {1..14 or 15 w/ m<=6}, m in {0..7}.
        const uint32_t e_e4m3 = (uint32_t) (e_un + 7);
        const uint32_t guard  = (mant >> 19) & 1u;
        const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
        const uint32_t lsb    = (mant >> 20) & 1u;
        uint32_t       m_e4m3 = (mant >> 20) & 0x7u;
        if (guard && (sticky || lsb)) m_e4m3 += 1;
        uint32_t e_out = e_e4m3;
        if (m_e4m3 == 8) {
            m_e4m3 = 0;
            e_out += 1;
            if (e_out > 15) {
                return (uint8_t) ((sign << 7) | (0xFu << 3) | 0x6u);
            }
        }
        // Don't accidentally synthesize the NaN pattern (e=15, m=7).
        if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
        return (uint8_t) ((sign << 7) | (e_out << 3) | m_e4m3);
    }

    // Subnormal e4m3: |x| < 2^-6. m = round(|x| * 2^9) in {0..7}.
    const int32_t shift = 23 - (e_un + 9);
    if (shift > 31) {
        return (uint8_t) (sign << 7);
    }
    const uint32_t implicit = (1u << 23) | mant;
    const uint32_t guard    = (implicit >> (shift - 1)) & 1u;
    const uint32_t sticky   = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
    uint32_t       m_e4m3   = implicit >> shift;
    const uint32_t lsb      = m_e4m3 & 1u;
    if (guard && (sticky || lsb)) m_e4m3 += 1;

    if (m_e4m3 >= 8) {
        return (uint8_t) ((sign << 7) | (1u << 3)); // rounds up into smallest normal
    }
    return (uint8_t) ((sign << 7) | m_e4m3); // m=0 means +-0
}

static __device__ __forceinline__ uint8_t r4d_f16_to_e4m3fn(half hx) {
    return r4d_f32_to_e4m3fn_sw(__half2float(hx));
}

// Convert a pair of F16 values to two e4m3fn bytes. Uses the gfx12 packed
// hardware converter (validated RNE, saturating) when compiled for gfx1201;
// falls back to the software codec otherwise. Inputs are clamped to +-448
// before the hardware path since the intrinsic itself does not saturate.
static __device__ __forceinline__ void r4d_f16x2_to_e4m3fn(half a, half b, uint8_t & oa, uint8_t & ob) {
#if defined(__gfx1201__)
    float fa = __half2float(a);
    float fb = __half2float(b);
    fa = fminf(fmaxf(fa, -448.0f), 448.0f);
    fb = fminf(fmaxf(fb, -448.0f), 448.0f);
    const uint32_t packed = (uint32_t) __builtin_amdgcn_cvt_pk_fp8_f32(fa, fb, 0, false);
    oa = (uint8_t) (packed & 0xFFu);
    ob = (uint8_t) ((packed >> 8) & 0xFFu);
#else
    oa = r4d_f16_to_e4m3fn(a);
    ob = r4d_f16_to_e4m3fn(b);
#endif
}

// One block per (token, kv_head). Threads split the head_dim into
// R4D_SCATTER_VEC-wide chunks: 16-byte F16 load (uint4 == 8 halfs), convert,
// 8-byte e4m3fn store (uint64_t == 8 bytes). Padding tokens (slot < 0) are
// skipped entirely -- slot_mapping already encodes which tokens are real, so
// there is no need to walk q_lens/prefix-sums the way the AITER scatter does.
__global__ void mt_r4d_scatter_kv_kernel(
        const half    * __restrict__ k_cur,   // [head_dim, n_kv_heads, n_tokens]
        const half    * __restrict__ v_cur,   // [head_dim, n_kv_heads, n_tokens]
        uint8_t       * __restrict__ kv,      // [num_blocks, kv_heads, block_size, 2*head_dim] e4m3fn
        const int32_t * __restrict__ slot_mapping, // [n_tokens], -1 = skip
        int n_kv_heads, int head_dim, int block_size) {

    const int t = blockIdx.x; // token index
    const int h = blockIdx.y; // kv head index

    const int slot = slot_mapping[t];
    if (slot < 0) {
        return; // padding token
    }

    const int block_idx     = slot / block_size;
    const int slot_in_block = slot % block_size;

    const size_t src_base = (size_t) (t * n_kv_heads + h) * (size_t) head_dim;
    const size_t dst_base = ((size_t) (block_idx * n_kv_heads + h) * (size_t) block_size
                              + (size_t) slot_in_block) * (size_t) (2 * head_dim);

    const int d0 = threadIdx.x * R4D_SCATTER_VEC;
    if (d0 >= head_dim) {
        return;
    }

    const int n_left = head_dim - d0;
    if (n_left >= R4D_SCATTER_VEC) {
        // Fast, fully vectorized path.
        const uint4 kvec = *reinterpret_cast<const uint4 *>(&k_cur[src_base + d0]);
        const uint4 vvec = *reinterpret_cast<const uint4 *>(&v_cur[src_base + d0]);
        const half * kh = reinterpret_cast<const half *>(&kvec);
        const half * vh = reinterpret_cast<const half *>(&vvec);

        uint64_t kbytes = 0;
        uint64_t vbytes = 0;
        #pragma unroll
        for (int i = 0; i < R4D_SCATTER_VEC; i += 2) {
            uint8_t k0, k1, v0, v1;
            r4d_f16x2_to_e4m3fn(kh[i], kh[i + 1], k0, k1);
            r4d_f16x2_to_e4m3fn(vh[i], vh[i + 1], v0, v1);
            kbytes |= ((uint64_t) k0 << (8 * i)) | ((uint64_t) k1 << (8 * (i + 1)));
            vbytes |= ((uint64_t) v0 << (8 * i)) | ((uint64_t) v1 << (8 * (i + 1)));
        }

        *reinterpret_cast<uint64_t *>(&kv[dst_base + d0])             = kbytes;
        *reinterpret_cast<uint64_t *>(&kv[dst_base + head_dim + d0])  = vbytes;
    } else {
        // Tail: head_dim not a multiple of R4D_SCATTER_VEC. Scalar cleanup.
        for (int i = 0; i < n_left; ++i) {
            const int d = d0 + i;
            kv[dst_base + d]             = r4d_f16_to_e4m3fn(k_cur[src_base + d]);
            kv[dst_base + head_dim + d]  = r4d_f16_to_e4m3fn(v_cur[src_base + d]);
        }
    }
}

void mt_r4d_scatter_kv(const half * k_cur, const half * v_cur, uint8_t * kv,
                       const int32_t * slot_mapping, const int32_t * q_lens,
                       int num_seqs, int n_tokens, int n_kv_heads, int head_dim,
                       int block_size, cudaStream_t stream) {
    GGML_UNUSED(q_lens);   // slot_mapping already addresses tokens directly
    GGML_UNUSED(num_seqs); // informational only; no per-seq walk needed here

    if (n_tokens <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        return;
    }

    const int threads = (head_dim + R4D_SCATTER_VEC - 1) / R4D_SCATTER_VEC;
    const dim3 grid(n_tokens, n_kv_heads);
    const dim3 block(threads);

    mt_r4d_scatter_kv_kernel<<<grid, block, 0, stream>>>(
        k_cur, v_cur, kv, slot_mapping, n_kv_heads, head_dim, block_size);
}

}  // namespace mt

#endif // GGML_HIP_R4D && GGML_USE_HIP
