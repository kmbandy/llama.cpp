// gemm_ml84_prod.hip — MAD-305 ML8_4 (4.5 bpw) production entry points.
//
// Three independent pieces, matching gemm_capi.h's declarations:
//   1. rdna4_pack_ml84_trfeed      — on-disk block_ml8_4[N][K/64] -> the
//      ML84_TRFEED B_nib + b_scale_g packed layout (ml84_trfeed_layout.h).
//   2. rdna4_gemm_ml84_trfeed_decode — launches the SEPARATE-COPY decode
//      kernel gemm_ml84_trfeed<32,1> (bench/trfeed_ml84_kernels.h) straight
//      against B_nib/lut/b_scale_g -- no fp8 expansion, no B_shuf.
//   3. rdna4_expand_ml84_to_trfeed — re-expands B_nib+lut+b_scale_g into the
//      frozen fp8 kernel's B_shuf + a single per-column b_scale, so the
//      UNCHANGED frozen `gemm_fp8_trfeed<128,2>` (gemm_trfeed_prod.hip) can
//      run the prefill GEMM at its full measured 131+ TF.
//
// None of these touch the frozen fp8 trfeed kernel bodies. Like
// gemm_trfeed_prod.hip, the byte-shuffle/pack kernels here are NOT
// gfx12-gated (pure integer/byte arithmetic + software e4m3 conversion,
// same convention as gemm_blockscale.hip's bs_trperm preshuffle kernels);
// only the decode kernel body (bench/trfeed_ml84_kernels.h) uses gfx12-only
// intrinsics, and that header carries its own `#if defined(__gfx1201__)`
// guard exactly like trfeed_kernels.h.
#include <hip/hip_runtime.h>
#include "bench/trfeed_ml84_kernels.h"   // gemm_ml84_trfeed<TBM,TWAVES_M>; pulls in trfeed_kernels.h + ml84_trfeed_layout.h
#include "gemm_capi.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────
// 1. Device packer: on-disk block_ml8_4[N][n_groups_k] -> B_nib + b_scale_g.
//
// One thread per (n, g) block (mirrors ml8.cu's ml8_repack_kernel grid
// shape). Each thread owns exactly QK_ML8=64 K-positions of ONE output
// column n within ONE 64-wide K-group g; ml84_trfeed_nk_to_pos(n,k,...) is
// a bijection, so distinct (n,g) threads never write the same B_nib byte
// (different g -> disjoint 16x16 tiles' K-range; different n -> disjoint
// lane L, since L encodes nl directly) -- no atomics needed for the nibble
// read-modify-write.
//
// OUT-OF-PLACE ONLY: `w_blocks` (source) and `B_nib`/`b_scale_g`
// (destination) must not alias. This kernel needs no additional device
// scratch beyond the caller-provided B_nib (N*K/2 bytes) and b_scale_g
// ((K/64)*N floats) -- unlike an in-place repack (overwriting the same
// bytes the source occupies), which would need its own scratch copy of the
// source block first; that in-place variant is left to the wiring task per
// the assignment's brief, exactly as ml8.cu's WP_ML8_INPLACE path handles
// its own in-place eligibility checks for the existing ml8-4 repack.
// ─────────────────────────────────────────────────────────────────────────
__global__ void ml84_pack_kernel(const uint8_t* __restrict__ src,
                                 uint8_t* __restrict__ B_nib, float* __restrict__ b_scale_g,
                                 int N, int K, int n_groups_k) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (n >= N || g >= n_groups_k) return;

    const block_ml8_4* blocks = reinterpret_cast<const block_ml8_4*>(src);
    const block_ml8_4& blk = blocks[(size_t) n * (size_t) n_groups_k + (size_t) g];
    b_scale_g[(size_t) g * (size_t) N + (size_t) n] = blk.scale;

    #pragma unroll
    for (int i = 0; i < QK_ML8 / 2; ++i) {
        const uint8_t packed = blk.qs[i];
        const uint8_t lo_idx = packed & 0x0Fu;
        const uint8_t hi_idx = (packed >> 4) & 0x0Fu;
        const int k_lo = g * QK_ML8 + 2 * i;
        const int k_hi = k_lo + 1;
        ml84_set_nibble(B_nib, ml84_trfeed_nk_to_pos(n, k_lo, N, K), lo_idx);
        ml84_set_nibble(B_nib, ml84_trfeed_nk_to_pos(n, k_hi, N, K), hi_idx);
    }
}

extern "C" hipError_t rdna4_pack_ml84_trfeed(const void* w_blocks, int N, int K,
                                             uint8_t* B_nib, float* b_scale_g, hipStream_t stream) {
    if (N <= 0 || K <= 0 || (K % QK_ML8) != 0 || (N % 16) != 0) {
        return hipErrorInvalidValue;
    }
    const int n_groups_k = K / QK_ML8;
    // Zero B_nib first: every (n,k) position IS covered exactly once by the
    // grid below, so this is defensive determinism (matching
    // ml84_trfeed_pack_host's memset), not a correctness requirement.
    hipError_t zc = hipMemsetAsync(B_nib, 0, (size_t) N * (size_t) K / 2, stream);
    if (zc != hipSuccess) {
        return zc;
    }
    constexpr int BLOCK_N = 64;
    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, n_groups_k, 1);
    const dim3 block(BLOCK_N, 1, 1);
    ml84_pack_kernel<<<grid, block, 0, stream>>>((const uint8_t*) w_blocks, B_nib, b_scale_g, N, K, n_groups_k);
    return hipGetLastError();
}

// ─────────────────────────────────────────────────────────────────────────
// 2. Decode launcher: the SAME (32,1) tile shape as the frozen fp8 kernel's
// decode/verify instantiation (gemm_trfeed_prod.hip's TBM_DECODE), so this
// entry point's grid math is copied from there verbatim.
//
// ROUND 3: this used to launch a separate LDS-staged `gemm_ml84_trfeed`
// kernel; the coordinator's bench measured NO improvement from that staging
// (0.300-0.305 ms vs round 1's 0.31 ms, both at N=17408) while the unstaged
// split-K kernel at n_splits=1 measured 0.194 ms -- already within 10% of
// the frozen fp8 kernel. So that kernel is gone (trfeed_ml84_kernels.h's
// HISTORY comment) and decode now simply IS the split-K entry point called
// with n_splits=1 -- one workgroup per (n-tile), walking every K-group in a
// single blockIdx.z=0 pass, storing via the split-K kernel's ATOMIC=false
// path (see rdna4_gemm_ml84_trfeed_decode_splitk below): no memset, no
// atomics, functionally identical to a dedicated non-split kernel.
// ─────────────────────────────────────────────────────────────────────────
constexpr int TBM_ML84_DECODE = 32, TWAVES_M_ML84_DECODE = 1;

extern "C" hipError_t rdna4_gemm_ml84_trfeed_decode(const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                                    float* C_f32, const float* a_scale, const float* b_scale_g,
                                                    int M_pad, int N, int K, hipStream_t stream) {
    return rdna4_gemm_ml84_trfeed_decode_splitk(A, B_nib, lut, C_f32, a_scale, b_scale_g,
                                                M_pad, N, K, /*n_splits=*/1, stream);
}

// ─────────────────────────────────────────────────────────────────────────
// 3. Prefill expander: ML84_TRFEED (B_nib/lut/b_scale_g) -> frozen fp8
// kernel's B_shuf + a single per-column fp32 b_scale.
//
// SCALE CONVENTION (matched to the frozen path exactly -- see ml8.cu's
// documented `amax/448` convention, e.g. the a_scale/b_scale quant kernels'
// comments "Scale: absmax / 448" / "amax/448.0f: real division"): the
// frozen kernel's epilogue computes `C[gr*N+gc] = ws[t]*a_scale[gr]*b_scale[gc]`
// where `ws[t]` is the raw fp8xfp8 WMMA accumulation and the STORED fp8
// byte for weight w was `round(w / b_scale)`, i.e. b_scale = amax/448 (the
// value that maps the column's max magnitude to ~448, e4m3's max normal).
// We reproduce that exactly for the re-expanded column:
//   b_scale_out[n] := (max_g b_scale_g[g,n]) / 448
// (LUT entries have |c| <= 1 by construction -- an invariant of the
// calibration pipeline, ggml-turbo-quant.c's dequantize_row_ml8_4_with_lut
// -- so |w| = |centroid * scale_g| <= scale_g for every element of group g,
// hence |w| <= max_g(scale_g) over the whole column: b_scale_out as defined
// above is a valid "amax/448"-style scale for the re-quantized fp8 byte),
// and each fp8 byte becomes:
//   byte = e4m3_round(w / b_scale_out[n])   where w = centroid * scale_g
// which is algebraically e4m3_round(centroid * scale_g / max_g(scale_g) * 448)
// -- the task brief's formula, just with the "* 448" folded into
// b_scale_out's own definition so the encode step and the value handed back
// to the frozen kernel's epilogue use the exact same number (no separate
// "*448-but-not-in-b_scale_out" reading, which would desync the encode from
// the decode by a factor of 448 -- flagged as the one place this spec was
// ambiguous; see the task report).
// ─────────────────────────────────────────────────────────────────────────

// Per-column fp8 scale for the expanded B_shuf: the column's true absmax
// over all K = max_g (b_scale_g[g][n] * max_i |lut[g][i]|), divided by 448
// (the frozen path's amax/448 convention). Using the LUT's actual per-group
// absmax instead of assuming |c| <= 1 (i) is exact for any LUT -- a LUT with
// entries > 1 (test-backend-ops' random F8_E4M3 init goes to 1.875) would
// otherwise push bytes past e4m3's 448 and saturate, and (ii) tightens the
// scale when a group's LUT tops out below 1.
__global__ void ml84_scale_colmax_kernel(const float* __restrict__ b_scale_g, const uint8_t* __restrict__ lut,
                                         float* __restrict__ b_scale_out, int N, int n_groups_k) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float mx = 0.f;
    for (int g = 0; g < n_groups_k; ++g) {
        float lut_amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            lut_amax = fmaxf(lut_amax, fabsf(ml84_decode_e4m3(lut[(size_t) g * 16 + i])));
        }
        mx = fmaxf(mx, b_scale_g[(size_t) g * (size_t) N + (size_t) n] * lut_amax);
    }
    b_scale_out[n] = fmaxf(mx / 448.0f, 1e-12f);
}

// REWRITTEN (was per-(n,k)-element, iterating (n,k) then inverting to
// (L,s); the coordinator's bench run found the old version numerically
// wrong AND ~6x slower than bandwidth). This version iterates one thread
// per (kt,nt,L) triple instead -- i.e. exactly the same domain
// trfeed_ml84_kernels.h's decode kernel walks per-lane, and exactly the
// iteration ORDER trfeed_common.h's preshuffle_B / gemm_blockscale.hip's
// rdna4_ml8fp8_preshuffle_kernel use (outer loop over (L,s), not over the
// logical (kl,nl) domain) -- so this is now provably built the same way as
// two already-proven-correct pieces of code instead of relying on an
// (kl,nl)->(L,s) matrix inversion I could not get independently verified in
// this sandbox. Per thread: ONE coalesced 4-byte B_nib load (the SAME
// nib_base+L*4 address the decode kernel reads for this exact (kt,nt,L) --
// so this is reading the identical byte decode reads, byte for byte) serves
// all 8 elements s=0..7 the lane owns, instead of the old code's 8
// independent single-byte re-derivations (also removing 8x redundant
// kt/nt/kl/nl integer-division work). The group `g` is a TILE-level
// constant here (QK_ML8=64 spans exactly 4 consecutive 16-wide kt tiles, so
// every k in tile kt maps to the same g = kt*16/QK_ML8), computed once per
// thread instead of once per element.
__global__ void ml84_expand_kernel(const uint8_t* __restrict__ B_nib, const uint8_t* __restrict__ lut,
                                   const float* __restrict__ b_scale_g, const float* __restrict__ b_scale_out,
                                   int K, int N, uint8_t* __restrict__ B_shuf_out) {
    const int KT = K / 16, NT = N / 16;
    const size_t total_threads = (size_t) KT * (size_t) NT * 32;   // one thread per (kt,nt,L)
    for (size_t tix = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         tix < total_threads; tix += (size_t) gridDim.x * blockDim.x) {
        const int L = (int) (tix % 32);
        const size_t tile_idx = tix / 32;
        const int nt = (int) (tile_idx % (size_t) NT);
        const int kt = (int) (tile_idx / (size_t) NT);

        // ---- ONE coalesced 4-byte load, same address the decode kernel reads ----
        const size_t nib_base = ml84_tile_nib_base(kt, nt, NT);
        const uint32_t w = *reinterpret_cast<const uint32_t*>(B_nib + nib_base + (size_t) L * 4);

        // g is constant across this whole tile (4 kt-tiles per 64-wide group).
        const int g = (kt * 16) / QK_ML8;
        const uint8_t* lut_g = lut + (size_t) g * 16;
        const size_t tile_off = b_tile_offset(kt, nt, NT);
        const int nl = L & 15;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            // low nibbles of w's 4 bytes = elements 0-3, high nibbles = elements 4-7
            // (identical byte/nibble convention to ml84_expand_frag / B_nib's contract).
            const uint8_t idx4 = (s < 4) ? (uint8_t) ((w >> (s * 8)) & 0x0Fu)
                                          : (uint8_t) ((w >> ((s - 4) * 8 + 4)) & 0x0Fu);
            const int kl = ((L >> 4) & 1) * 8 + s;
            const int n  = nt * 16 + nl;
            (void) kl;   // only needed for documentation/debug -- k itself is never used below (g already known)

            const float centroid = ml84_decode_e4m3(lut_g[idx4]);
            const float w_val = centroid * b_scale_g[(size_t) g * (size_t) N + (size_t) n];
            const float denom = b_scale_out[n] > 0.f ? b_scale_out[n] : 1.f;   // degenerate all-zero column guard

            // Same (kt,nt,L,s) -> trperm(L,s) mapping as trfeed_common.h's
            // preshuffle_B (identical iteration structure, not just identical
            // formula), so this is provably the same permutation as the fp8 path.
            B_shuf_out[tile_off + (size_t) trperm(L, s)] = ml84_encode_e4m3(w_val / denom);
        }
    }
}

extern "C" hipError_t rdna4_expand_ml84_to_trfeed(const uint8_t* B_nib, const uint8_t* lut,
                                                  const float* b_scale_g, int N, int K,
                                                  uint8_t* B_shuf_out, float* b_scale_out, hipStream_t stream) {
    if (N <= 0 || K <= 0 || (K % QK_ML8) != 0 || (N % 16) != 0 || (K % 16) != 0) {
        return hipErrorInvalidValue;
    }
    const int n_groups_k = K / QK_ML8;
    constexpr int TPB = 256;

    const int grid_n = (N + TPB - 1) / TPB;
    ml84_scale_colmax_kernel<<<grid_n, TPB, 0, stream>>>(b_scale_g, lut, b_scale_out, N, n_groups_k);
    hipError_t rc = hipGetLastError();
    if (rc != hipSuccess) {
        return rc;
    }

    const int KT = K / 16, NT = N / 16;
    const size_t total_threads = (size_t) KT * (size_t) NT * 32;
    const size_t grid64 = (total_threads + TPB - 1) / TPB;
    const int grid_e = (int) (grid64 > 65535u ? 65535u : grid64);
    ml84_expand_kernel<<<grid_e, TPB, 0, stream>>>(B_nib, lut, b_scale_g, b_scale_out, K, N, B_shuf_out);
    return hipGetLastError();
}

// ─────────────────────────────────────────────────────────────────────────
// 4. Split-K decode: for workgroup-starved N (N/128 workgroups << the GPU's
// CU count -- e.g. the (8,5120,17408) shape's N/128=40 WGs measured 51 GB/s
// vs the (…,17408,…) shapes' 136 WGs at 161 GB/s), split each (n-tile)'s
// K-group range across blockIdx.z the same way trfeed_splitk_kernels.h does
// for the frozen fp8 kernel, but on 64-K GROUP boundaries (not raw K-tiles)
// so the per-group scale fold in bench/trfeed_ml84_kernels.h's
// gemm_ml84_trfeed_splitk<32,1> always sees a whole group.
// ─────────────────────────────────────────────────────────────────────────

extern "C" hipError_t rdna4_gemm_ml84_trfeed_decode_splitk(const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                                           float* C_f32, const float* a_scale, const float* b_scale_g,
                                                           int M_pad, int N, int K, int n_splits, hipStream_t stream) {
    if (M_pad != TBM_ML84_DECODE || N <= 0 || K <= 0 || (N % BN) != 0 || (K % QK_ML8) != 0) {
        return hipErrorInvalidValue;
    }
    const int n_groups_k = K / QK_ML8;
    if (n_splits < 1) {
        n_splits = 1;
    }
    if (n_splits > n_groups_k) {
        n_splits = n_groups_k;   // never split finer than one group per workgroup
    }
    const int groups_per_split = (n_groups_k + n_splits - 1) / n_splits;
    constexpr int TBLOCK = TWAVES_M_ML84_DECODE * WAVES_N * WAVE_SIZE;

    if (n_splits == 1) {
        // ATOMIC=false path: one workgroup per (n-tile) walks the WHOLE K
        // range (groups_per_split==n_groups_k, grid.z==1) and plain-stores
        // once -- no memset, no atomics. This IS what
        // rdna4_gemm_ml84_trfeed_decode calls; kept as its own n_splits==1
        // branch here (rather than a separate function) so there is exactly
        // one launch path to keep correct.
        const dim3 grid(N / BN, M_pad / TBM_ML84_DECODE, 1);
        gemm_ml84_trfeed_splitk<TBM_ML84_DECODE, TWAVES_M_ML84_DECODE, /*ATOMIC=*/false>
            <<<grid, TBLOCK, 0, stream>>>(
                (const float8_t*) A, B_nib, lut, C_f32, a_scale, b_scale_g, M_pad, N, K, groups_per_split);
        return hipGetLastError();
    }

    // ATOMIC=true path (n_splits > 1): atomics accumulate onto zero -- zero
    // the WHOLE M_pad*N output first, same rationale as
    // rdna4_gemm_fp8_trfeed_splitk (the padded decode rows the A-tile fill
    // still reads/writes must never contain garbage).
    const hipError_t zc = hipMemsetAsync(C_f32, 0, (size_t) M_pad * (size_t) N * sizeof(float), stream);
    if (zc != hipSuccess) {
        return zc;
    }
    const dim3 grid(N / BN, M_pad / TBM_ML84_DECODE, n_splits);
    gemm_ml84_trfeed_splitk<TBM_ML84_DECODE, TWAVES_M_ML84_DECODE, /*ATOMIC=*/true>
        <<<grid, TBLOCK, 0, stream>>>(
            (const float8_t*) A, B_nib, lut, C_f32, a_scale, b_scale_g, M_pad, N, K, groups_per_split);
    return hipGetLastError();
}

// Heuristic default split count -- round 3, re-derived from the coordinator's
// measured numbers (9070 XT): N=17408 (N/128=136 WGs) n_splits=1 0.194 ms
// (258 GB/s) was BEST (2: 0.198, 4: 0.296, 8: 0.535 -- all worse, this shape
// is already well-supplied with workgroups); N=5120/K=17408 (N/128=40 WGs)
// n_splits=4 0.177 ms (283 GB/s) was BEST (1: 0.364, 2: 0.223, 8: 0.211 --
// n=8 WORSE than n=4, more workgroups than the K range usefully absorbs
// once each split's own A-fill/WMMA loop overhead dominates). Rule: no split
// once N/128 already gives a generous workgroup count (>=128, comfortably
// above a 9070 XT's CU count); otherwise aim for a fixed total-workgroup
// target of 160 (160/40=4, matching the measured-best n=4 at N=5120 exactly)
// capped at both n_groups_k/2 (a split needs >=2 groups to amortize its own
// loop overhead) and a hard 8 (n=8 measured worse than n=4 at the only
// workgroup-starved shape tested -- more splits than that were never a win).
extern "C" int rdna4_ml84_trfeed_splitk_default_splits(int N, int K) {
    if (N <= 0 || K <= 0) {
        return 1;
    }
    const int n_wg_base = N / BN;
    const int n_groups_k = K / QK_ML8;
    if (n_wg_base <= 0 || n_groups_k <= 0) {
        return 1;
    }
    if (n_wg_base >= 128) {
        return 1;
    }
    const int want = (160 + n_wg_base / 2) / n_wg_base;   // round(160 / n_wg_base)
    int max_splits = std::max(1, n_groups_k / 2);
    if (max_splits > 8) {
        max_splits = 8;
    }
    return std::min(std::max(want, 1), max_splits);
}

// ─────────────────────────────────────────────────────────────────────────
// 5. Prefill EXPERIMENT: in-kernel LUT expansion at the frozen prefill tile
// geometry (TBM=128, TWAVES_M=2 -- acc[4][4], 4 waves), instead of
// expander-once-then-frozen-fp8-kernel (rdna4_expand_ml84_to_trfeed +
// rdna4_gemm_fp8_trfeed). Reuses gemm_ml84_trfeed_splitk<128,2,false> with
// groups_per_split = n_groups_k and grid.z = 1 -- i.e. exactly the decode
// launcher's "n_splits=1" trick, just at the bigger tile -- so this is not a
// new kernel, only a new launcher/instantiation.
//
// Hypothesis under test (coordinator, round 3): in-kernel LUT expansion is
// VALU work sharing the issue port with WMMA (ml84_lut_gather4's
// __builtin_amdgcn_perm calls), so this should LOSE to expander+frozen at
// M=2048 (where the expander's ~0.4-0.6ms fixed cost is amortized over a
// huge GEMM) but may WIN at M=128-512 where that fixed cost dominates.
// gemm_ml84_bench.hip benches both paths at M in {128,512,2048} to decide
// per-M rather than asserting an answer here.
// ─────────────────────────────────────────────────────────────────────────
extern "C" hipError_t rdna4_gemm_ml84_trfeed_prefill(const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                                     float* C_f32, const float* a_scale, const float* b_scale_g,
                                                     int M_pad, int N, int K, hipStream_t stream) {
    constexpr int TBM_PREFILL = BM, TWAVES_M_PREFILL = WAVES_M;   // the frozen 128x128/4-wave tile geometry
    if (M_pad <= 0 || (M_pad % TBM_PREFILL) != 0 || N <= 0 || K <= 0 ||
        (N % BN) != 0 || (K % QK_ML8) != 0) {
        return hipErrorInvalidValue;
    }
    const int n_groups_k = K / QK_ML8;
    constexpr int TBLOCK = TWAVES_M_PREFILL * WAVES_N * WAVE_SIZE;
    const dim3 grid(N / BN, M_pad / TBM_PREFILL, 1);
    gemm_ml84_trfeed_splitk<TBM_PREFILL, TWAVES_M_PREFILL, /*ATOMIC=*/false><<<grid, TBLOCK, 0, stream>>>(
        (const float8_t*) A, B_nib, lut, C_f32, a_scale, b_scale_g, M_pad, N, K, n_groups_k);
    return hipGetLastError();
}
