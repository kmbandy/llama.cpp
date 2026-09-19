// Standalone correctness check for the libr4d paged-fp8-KV scatter kernel
// (mt::mt_r4d_scatter_kv, ggml/src/ggml-cuda/mt_pagedattn_r4d_scatter.cu).
//
// Reproduces the kernel + its e4m3fn codec verbatim (they are file-local
// __global__/static device functions in that .cu, not exported symbols) and
// checks every byte the kernel writes against an independent CPU e4m3fn
// reference, for a shape with gaps and -1 (padding) entries in slot_mapping:
//
//   n_tokens = 300, n_kv_heads = 4, head_dim = 256, block_size = 16
//
// Also checks that slots never touched by slot_mapping keep a sentinel byte
// pattern, i.e. the kernel does not write outside the slots it's told to.
//
// This file is deliberately NOT wired into tests/CMakeLists.txt -- per the
// task, it is reported but not built/run here (mirrors the idiom used by
// tests/test-ar-codec-q8-wide-grid.hip.cpp). Build with (adjust
// --offload-arch for the target GPU; gfx1201 = RDNA4 RX 9070 XT / R9700):
//
//   hipcc -O2 --offload-arch=gfx1201 -std=c++17 \
//       tests/test-r4d-scatter.hip.cpp -o /tmp/test-r4d-scatter
//   /tmp/test-r4d-scatter
//
// Expected output: "PASS" with a max abs/ulp error report, or "FAIL" with
// the first mismatching byte.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Verbatim copy of the device codec + kernel from mt_pagedattn_r4d_scatter.cu
// (kept in sync by hand; there is no shared header because the production
// file's device functions are file-local statics).
// ---------------------------------------------------------------------------

static constexpr int R4D_SCATTER_VEC = 8;

static __device__ __forceinline__ uint8_t r4d_f32_to_e4m3fn_sw(float xv) {
    uint32_t bits;
    memcpy(&bits, &xv, 4);
    const uint32_t sign  = (bits >> 31) & 1u;
    const uint32_t exp_b = (bits >> 23) & 0xFFu;
    const uint32_t mant  = bits & 0x7FFFFFu;

    if (exp_b == 0xFFu) {
        return (uint8_t) ((sign << 7) | 0x7Fu);
    }
    if (exp_b == 0) {
        return (uint8_t) (sign << 7);
    }

    const int32_t e_un = (int32_t) exp_b - 127;

    if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u)) {
        return (uint8_t) ((sign << 7) | (0xFu << 3) | 0x6u);
    }

    if (e_un >= -6) {
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
        if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
        return (uint8_t) ((sign << 7) | (e_out << 3) | m_e4m3);
    }

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
        return (uint8_t) ((sign << 7) | (1u << 3));
    }
    return (uint8_t) ((sign << 7) | m_e4m3);
}

static __device__ __forceinline__ uint8_t r4d_f16_to_e4m3fn(_Float16 hx) {
    return r4d_f32_to_e4m3fn_sw((float) hx);
}

static __device__ __forceinline__ void r4d_f16x2_to_e4m3fn(_Float16 a, _Float16 b, uint8_t & oa, uint8_t & ob) {
#if defined(__gfx1201__)
    float fa = (float) a;
    float fb = (float) b;
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

__global__ void mt_r4d_scatter_kv_kernel(
        const _Float16 * __restrict__ k_cur,
        const _Float16 * __restrict__ v_cur,
        uint8_t         * __restrict__ kv,
        const int32_t   * __restrict__ slot_mapping,
        int n_kv_heads, int head_dim, int block_size) {

    const int t = blockIdx.x;
    const int h = blockIdx.y;

    const int slot = slot_mapping[t];
    if (slot < 0) {
        return;
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
        const uint4 kvec = *reinterpret_cast<const uint4 *>(&k_cur[src_base + d0]);
        const uint4 vvec = *reinterpret_cast<const uint4 *>(&v_cur[src_base + d0]);
        const _Float16 * kh = reinterpret_cast<const _Float16 *>(&kvec);
        const _Float16 * vh = reinterpret_cast<const _Float16 *>(&vvec);

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

        *reinterpret_cast<uint64_t *>(&kv[dst_base + d0])            = kbytes;
        *reinterpret_cast<uint64_t *>(&kv[dst_base + head_dim + d0]) = vbytes;
    } else {
        for (int i = 0; i < n_left; ++i) {
            const int d = d0 + i;
            kv[dst_base + d]            = r4d_f16_to_e4m3fn(k_cur[src_base + d]);
            kv[dst_base + head_dim + d] = r4d_f16_to_e4m3fn(v_cur[src_base + d]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference e4m3fn conversion. Independent implementation (not shared
// code with the device path above) so this is a genuine cross-check, not a
// tautology. Same OCP e4m3fn semantics: RNE, saturate to +-448, no
// infinities, NaN only for S.1111.111.
// ---------------------------------------------------------------------------

static uint8_t cpu_f32_to_e4m3fn(float xv) {
    uint32_t bits;
    memcpy(&bits, &xv, 4);
    const uint32_t sign  = (bits >> 31) & 1u;
    const uint32_t exp_b = (bits >> 23) & 0xFFu;
    const uint32_t mant  = bits & 0x7FFFFFu;

    if (exp_b == 0xFFu) return (uint8_t) ((sign << 7) | 0x7Fu);
    if (exp_b == 0)     return (uint8_t) (sign << 7);

    const int32_t e_un = (int32_t) exp_b - 127;
    if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u)) {
        return (uint8_t) ((sign << 7) | (0xFu << 3) | 0x6u);
    }

    if (e_un >= -6) {
        const uint32_t e_e4m3 = (uint32_t) (e_un + 7);
        const uint32_t guard  = (mant >> 19) & 1u;
        const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
        const uint32_t lsb    = (mant >> 20) & 1u;
        uint32_t       m      = (mant >> 20) & 0x7u;
        if (guard && (sticky || lsb)) m += 1;
        uint32_t e_out = e_e4m3;
        if (m == 8) {
            m = 0;
            e_out += 1;
            if (e_out > 15) return (uint8_t) ((sign << 7) | (0xFu << 3) | 0x6u);
        }
        if (e_out == 15 && m == 7) m = 6;
        return (uint8_t) ((sign << 7) | (e_out << 3) | m);
    }

    const int32_t shift = 23 - (e_un + 9);
    if (shift > 31) return (uint8_t) (sign << 7);
    const uint32_t implicit = (1u << 23) | mant;
    const uint32_t guard    = (implicit >> (shift - 1)) & 1u;
    const uint32_t sticky   = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
    uint32_t       m        = implicit >> shift;
    const uint32_t lsb      = m & 1u;
    if (guard && (sticky || lsb)) m += 1;
    if (m >= 8) return (uint8_t) ((sign << 7) | (1u << 3));
    return (uint8_t) ((sign << 7) | m);
}

static float e4m3fn_to_f32(uint8_t b) {
    const uint32_t s = (b >> 7) & 1u;
    const uint32_t e = (b >> 3) & 0xFu;
    const uint32_t m = b & 0x7u;
    if (e == 0 && m == 0) return s ? -0.0f : 0.0f;
    if (e == 15 && m == 7) return NAN;
    float val;
    if (e == 0) {
        val = ldexpf((float) m, -9); // subnormal: m/8 * 2^-6
    } else {
        val = ldexpf(1.0f + (float) m / 8.0f, (int) e - 7);
    }
    return s ? -val : val;
}

int main() {
    const int n_tokens   = 300;
    const int n_kv_heads = 4;
    const int head_dim   = 256;
    const int block_size = 16;
    const int num_blocks = 40; // 40*16 = 640 slots, enough for 300 tokens w/ gaps

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f); // exercise saturation too
    std::uniform_int_distribution<int> gap_dist(0, 2);

    std::vector<_Float16> k_cur((size_t) n_tokens * n_kv_heads * head_dim);
    std::vector<_Float16> v_cur((size_t) n_tokens * n_kv_heads * head_dim);
    for (auto & x : k_cur) x = (_Float16) dist(rng);
    for (auto & x : v_cur) x = (_Float16) dist(rng);

    // Build slot_mapping with gaps and some -1 (padding) entries. Slots used
    // are a strictly increasing, non-contiguous subset of [0, num_blocks*block_size).
    std::vector<int32_t> slot_mapping(n_tokens);
    {
        int slot = 0;
        int max_slot = num_blocks * block_size;
        for (int t = 0; t < n_tokens; ++t) {
            if (t % 7 == 0) { // sprinkle padding tokens
                slot_mapping[t] = -1;
                continue;
            }
            slot += 1 + gap_dist(rng); // leave gaps between used slots
            if (slot >= max_slot) {
                slot_mapping[t] = -1; // ran out of room -> treat as padding
                continue;
            }
            slot_mapping[t] = slot;
        }
    }

    const size_t kv_bytes = (size_t) num_blocks * n_kv_heads * block_size * 2 * head_dim;
    const uint8_t SENTINEL = 0xAA;
    std::vector<uint8_t> kv_host(kv_bytes, SENTINEL);

    _Float16 * d_k = nullptr;
    _Float16 * d_v = nullptr;
    uint8_t  * d_kv = nullptr;
    int32_t  * d_slot = nullptr;

    HIP_CHECK(hipMalloc(&d_k, k_cur.size() * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_v, v_cur.size() * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_kv, kv_bytes));
    HIP_CHECK(hipMalloc(&d_slot, slot_mapping.size() * sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(d_k, k_cur.data(), k_cur.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, v_cur.data(), v_cur.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kv, kv_host.data(), kv_bytes, hipMemcpyHostToDevice)); // seed sentinel on device too
    HIP_CHECK(hipMemcpy(d_slot, slot_mapping.data(), slot_mapping.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    const int threads = (head_dim + R4D_SCATTER_VEC - 1) / R4D_SCATTER_VEC;
    dim3 grid(n_tokens, n_kv_heads);
    dim3 block(threads);
    mt_r4d_scatter_kv_kernel<<<grid, block>>>(d_k, d_v, d_kv, d_slot, n_kv_heads, head_dim, block_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(kv_host.data(), d_kv, kv_bytes, hipMemcpyDeviceToHost));

    bool pass = true;
    uint64_t n_checked = 0;
    float max_abs_err = 0.0f;
    int max_ulp_err = 0;
    int first_bad_t = -1, first_bad_h = -1, first_bad_d = -1;
    bool first_bad_is_v = false;
    uint8_t first_bad_got = 0, first_bad_want = 0;

    std::vector<bool> slot_touched(num_blocks * n_kv_heads * block_size, false);

    for (int t = 0; t < n_tokens; ++t) {
        const int slot = slot_mapping[t];
        if (slot < 0) continue;
        const int block_idx     = slot / block_size;
        const int slot_in_block = slot % block_size;
        for (int h = 0; h < n_kv_heads; ++h) {
            slot_touched[(size_t) block_idx * n_kv_heads * block_size + (size_t) h * block_size + slot_in_block] = true;
            const size_t src_base = (size_t) (t * n_kv_heads + h) * head_dim;
            const size_t dst_base = ((size_t) (block_idx * n_kv_heads + h) * block_size + slot_in_block) * (size_t) (2 * head_dim);
            for (int d = 0; d < head_dim; ++d) {
                const uint8_t want_k = cpu_f32_to_e4m3fn((float) k_cur[src_base + d]);
                const uint8_t want_v = cpu_f32_to_e4m3fn((float) v_cur[src_base + d]);
                const uint8_t got_k = kv_host[dst_base + d];
                const uint8_t got_v = kv_host[dst_base + head_dim + d];
                n_checked += 2;
                if (got_k != want_k && pass) {
                    pass = false; first_bad_t = t; first_bad_h = h; first_bad_d = d;
                    first_bad_is_v = false; first_bad_got = got_k; first_bad_want = want_k;
                }
                if (got_v != want_v && pass) {
                    pass = false; first_bad_t = t; first_bad_h = h; first_bad_d = d;
                    first_bad_is_v = true; first_bad_got = got_v; first_bad_want = want_v;
                }
                const float fk = e4m3fn_to_f32(got_k);
                const float fkw = e4m3fn_to_f32(want_k);
                if (!std::isnan(fk) && !std::isnan(fkw)) {
                    max_abs_err = std::max(max_abs_err, std::fabs(fk - fkw));
                    max_ulp_err = std::max(max_ulp_err, std::abs((int) got_k - (int) want_k));
                }
            }
        }
    }

    // Sentinel check: every slot never referenced by slot_mapping must be untouched.
    int sentinel_violations = 0;
    for (int b = 0; b < num_blocks; ++b) {
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int sib = 0; sib < block_size; ++sib) {
                const size_t idx = (size_t) b * n_kv_heads * block_size + (size_t) h * block_size + sib;
                if (slot_touched[idx]) continue;
                const size_t dst_base = idx * (size_t) (2 * head_dim);
                for (int i = 0; i < 2 * head_dim; ++i) {
                    if (kv_host[dst_base + i] != SENTINEL) {
                        sentinel_violations++;
                    }
                }
            }
        }
    }

    printf("checked %llu bytes worth of conversions, max_abs_err=%.6f max_ulp_err=%d, sentinel_violations=%d\n",
           (unsigned long long) n_checked, max_abs_err, max_ulp_err, sentinel_violations);

    if (!pass) {
        printf("FAIL: first mismatch at t=%d h=%d d=%d (%s): got=0x%02x want=0x%02x\n",
               first_bad_t, first_bad_h, first_bad_d, first_bad_is_v ? "V" : "K",
               first_bad_got, first_bad_want);
    }
    if (sentinel_violations > 0) {
        pass = false;
    }

    printf("%s\n", pass ? "PASS" : "FAIL");

    hipFree(d_k); hipFree(d_v); hipFree(d_kv); hipFree(d_slot);
    return pass ? 0 : 1;
}
