#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include "mma.cuh"

#include <cstdlib>

namespace mma = ggml_cuda_mma;

// Longest token block the chunked (UT-transform) kernel accepts. Sized for the MTP
// verify-block regime; longer blocks keep the autoregressive kernel.
#define GGML_CUDA_GDN_CHUNK_MAX 16

// exp() argument clamp. The gate is -exp(A_log)*softplus(...) <= 0 so every exponent the
// chunked kernel evaluates is <= 0 in exact arithmetic; the clamp only guards against a
// pathological/NaN-free model producing a positive gate and overflowing f32.
#define GGML_CUDA_GDN_EXP_CLAMP 30.0f

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int64_t       tok_offset,
                                     int64_t       dst_seq_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    // dst_seq_stride is the ORIGINAL (un-split) n_tokens for this op; tok_offset is this
    // launch's starting token within that full sequence (0 unless this is a composed
    // prefill+tail launch, see ggml_cuda_op_gated_delta_net_impl).
    attn_data += (sequence * dst_seq_stride * H + h_idx) * S_v + tok_offset * S_v * H;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // Software pipeline over the token loop.
    //
    // The recurrence is serial in t, and its critical path is
    //   s_shard -> kv partial -> CROSS-LANE REDUCTION -> delta_col -> s_shard'
    // The original kernel put *two* cross-lane reductions in the loop body (kv and attn).
    // Only the kv one is a true dependency for the next token, but both sit in the same
    // issue stream, so on architectures with expensive cross-lane ops (Pascal __shfl,
    // GCN ds_swizzle/DPP) the loop paid ~2x reduction latency per token.
    //
    // Here token t's inputs are loaded and its kv partial computed *before* the reduction,
    // and token t-1's attention partial rides along in the same shuffle tree as a float2.
    // Both component sums keep their original operand order, so results are bit-identical.
    //
    // The loop runs one extra iteration (t == n_tokens) purely to drain the last token's
    // attention reduction.
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    float g_reg[rows_per_lane];             // KDA: per-row exp(g); GDA: unused
    float g_scalar    = 0.0f;               // GDA: exp(g), broadcast over rows
    float beta_val    = 0.0f;
    float v_val       = 0.0f;
    float attn_partial = 0.0f;

    for (int t = 0; t <= (int) n_tokens; t++) {
        // ---- load token t and form this lane's kv partial against the current state ----
        // kv[col] = sum_i (KDA ? g[i] : 1) * S[i][col] * k[i].
        // Multiply association matches the original: KDA -> (g*s)*k, GDA -> s*k with the
        // scalar g applied to the reduced kv_col afterwards.
        float kv_partial = 0.0f;
        if (t < (int) n_tokens) {
            const int64_t tg  = tok_offset + t; // global token index (this launch's window)
            const float * q_t = q + iq3 * sq3 + tg * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + tg * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + tg * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + tg * sb2 + h_idx * sb1;
            const float * g_t       = g + gb_offset * (KDA ? S_v : 1);

            beta_val = beta[gb_offset];
            v_val    = v_t[col];

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            }

            if constexpr (!KDA) {
                g_scalar = expf(*g_t);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_partial += s_shard[r] * k_reg[r];
                }
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    g_reg[r] = expf(g_t[i]);
                }
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_partial += g_reg[r] * s_shard[r] * k_reg[r];
                }
            }
        }

        // ---- one fused cross-lane reduction: kv[t] and attn[t-1] ----
        const float2 red = warp_reduce_sum<warp_size>(make_float2(kv_partial, attn_partial));

        // token t-1's attention output (at t == 0 this reduces a zero, which is discarded)
        if (t > 0 && lane == 0) {
            attn_data[(int64_t) (t - 1) * S_v * H + col] = red.y * scale;
        }

        if (t == (int) n_tokens) {
            break;
        }

        // delta[col] = (v[col] - (GDA ? g * kv[col] : kv[col])) * beta
        float delta_col;
        if constexpr (!KDA) {
            delta_col = (v_val - g_scalar * red.x) * beta_val;
        } else {
            delta_col = (v_val - red.x) * beta_val;
        }

        // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
        //        attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
        float ap = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const float g_r = KDA ? g_reg[r] : g_scalar;
            s_shard[r] = g_r * s_shard[r] + k_reg[r] * delta_col;
            ap += s_shard[r] * q_reg[r];
        }
        attn_partial = ap;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * snap = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    snap[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Chunked (UT-transform) gated delta net, for short token blocks (2 <= T <= 16), GDA only.
//
// The autoregressive kernel is serial in t with a cross-lane reduction on the critical
// path of every step, so a T-token block costs ~T reduction latencies. This kernel breaks
// that dependency chain algebraically.
//
// With a scalar gate, write gl_t for the raw log-decay of token t, G_t = sum_{u<=t} gl_u
// (inclusive prefix sum) and c_t = exp(G_t). Unrolling
//     S_t = g_t * S_{t-1} + k_t d_t^T,   d_t = beta_t (v_t - S_{t-1}^T (g_t k_t))
// over the block gives
//     S_t = c_t S_0 + sum_{j<=t} (c_t/c_j) k_j d_j^T
// and therefore
//     d_t = u_t - sum_{j<t} A[t][j] d_j,   u_t   = beta_t (v_t - c_t * (S_0^T k_t))
//                                          A[t][j] = beta_t * exp(G_t-G_j) * (k_t . k_j)
//     o_t = c_t * (S_0^T q_t) + sum_{j<=t} P[t][j] d_j,
//                                          P[t][j] = exp(G_t-G_j) * (k_j . q_t)
//
// Phases, in order:
//   (0) stage k, q, beta and the gate prefix sum in shared memory
//   (1) Gram blocks A and P    -- T(T+1)/2 mutually INDEPENDENT cross-lane reductions
//   (2) S_0^T k_t and S_0^T q_t for all t -- 2T mutually INDEPENDENT cross-lane reductions
//   (3) forward substitution for d -- T serial steps, but each is a scalar FMA chain with
//                                     NO cross-lane op, so the whole T-loop costs ~T*FMA
//   (4) outputs o_t, then the T state updates + snapshots -- per-lane FMAs, no reductions
//
// Net effect: the number of *serialized* cross-lane reductions drops from 2T to O(1).
// ---------------------------------------------------------------------------------------
// TM is the compile-time block length: the smallest supported chunk width that still
// covers n_tokens (see gdn_chunk_width()).  It is NOT always GGML_CUDA_GDN_CHUNK_MAX --
// phases (2), (3) and (4) below are unrolled over TM, not over n_tokens, so a T=8 verify
// block compiled at TM=16 paid for 16 cross-lane reductions, a 120-FMA substitution chain
// and 2x the shared memory it needed.  Everything is still exact for any TM >= n_tokens:
// the (t,j) arithmetic is identical, TM only changes the sh_A/sh_P row stride and how many
// provably-inert t >= n_tokens slots the unrolled loops carry.
template <int S_v, int TM, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_chunked_cuda(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             float *       state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       n_seqs,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale,
                             int64_t       state_slot_stride,
                             int           K) {
    constexpr int lanes         = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % lanes == 0, "S_v must be a multiple of the reduction width");
    constexpr int rows_per_lane = S_v / lanes;
    constexpr int nwarps        = 4;
    static_assert(TM >= 2 && TM <= GGML_CUDA_GDN_CHUNK_MAX, "chunk width out of range");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;              // 0 .. lanes-1
    const int      wid      = threadIdx.y;              // logical warp == one output column
    const int      col      = blockIdx.z * nwarps + wid;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int nthreads = lanes * nwarps;
    const int tid      = wid * lanes + lane;
    const int nt       = (int) n_tokens;

    __shared__ float sh_k [TM * S_v];
    __shared__ float sh_q [TM * S_v];
    __shared__ float sh_gl[TM];        // per-token raw log decay
    __shared__ float sh_G [TM];        // inclusive prefix sum of sh_gl
    __shared__ float sh_bt[TM];        // beta
    __shared__ float sh_A [TM * TM];   // beta_t * exp(G_t-G_j) * (k_t . k_j), j <  t
    __shared__ float sh_P [TM * TM];   // exp(G_t-G_j) * (k_j . q_t),          j <= t

    ggml_cuda_pdl_sync();

    // ---- phase 0: stage k / q / beta / gate ------------------------------------------
    for (int t = 0; t < nt; t++) {
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        for (int i = tid; i < S_v; i += nthreads) {
            sh_k[t * S_v + i] = k_t[i];
            sh_q[t * S_v + i] = q_t[i];
        }
    }
    // zero the Gram blocks so unused (t,j) entries can never feed a NaN into the
    // fully-unrolled substitution loops below
    for (int i = tid; i < TM * TM; i += nthreads) {
        sh_A[i] = 0.0f;
        sh_P[i] = 0.0f;
    }
    // strided, not "if (tid < nt)": keeps this correct if the block is ever narrower than nt
    for (int t = tid; t < nt; t += nthreads) {
        const int64_t gb = sequence * sb3 + (int64_t) t * sb2 + h_idx * sb1;
        sh_gl[t] = g[gb];
        sh_bt[t] = beta[gb];
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int t = 0; t < nt; t++) {
            acc      += sh_gl[t];
            sh_G[t]   = acc;
        }
    }
    __syncthreads();

    // ---- phase 1: Gram blocks (independent reductions) --------------------------------
    // pair p enumerates (t,j) with j <= t; the iteration count is block-uniform so every
    // lane reaches every warp_reduce_sum (the shuffles use a full mask)
    const int npairs = (nt * (nt + 1)) / 2;
    const int niter  = (npairs + nwarps - 1) / nwarps;

    for (int it = 0; it < niter; it++) {
        const int  p     = it * nwarps + wid;
        const bool valid = p < npairs;

        int t = 0;
        int j = valid ? p : 0;
        while (j > t) {
            j -= t + 1;
            t++;
        }

        float pkk = 0.0f;
        float pkq = 0.0f;
        if (valid) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int   i  = r * lanes + lane;
                const float kt = sh_k[t * S_v + i];
                const float kj = sh_k[j * S_v + i];
                const float qt = sh_q[t * S_v + i];
                pkk += kt * kj;
                pkq += kj * qt;
            }
        }

        const float2 red = warp_reduce_sum<lanes>(make_float2(pkk, pkq));

        if (valid && lane == 0) {
            const float e = expf(fminf(sh_G[t] - sh_G[j], GGML_CUDA_GDN_EXP_CLAMP));
            if (j < t) {
                sh_A[t * TM + j] = sh_bt[t] * e * red.x;
            }
            sh_P[t * TM + j] = e * red.y;
        }
    }
    __syncthreads();

    // ---- state into registers ---------------------------------------------------------
    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;

    const float * s_in   = curr_state + state_in_offset + (int64_t) col * S_v;
    float *       st_out = state + state_out_offset;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = s_in[r * lanes + lane];
    }

    // ---- phase 2: u_t and the S_0 part of o_t (independent reductions) -----------------
    // The loop bound is the compile-time TM (not n_tokens) so nvcc fully unrolls it and
    // keeps u[]/ob[]/d[] in registers instead of spilling them to local memory.
    float u [TM];
    float ob[TM];
#pragma unroll
    for (int t = 0; t < TM; t++) {
        float pu = 0.0f;
        float pw = 0.0f;
        if (t < nt) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes + lane;
                pu += s_shard[r] * sh_k[t * S_v + i];
                pw += s_shard[r] * sh_q[t * S_v + i];
            }
        }

        const float2 red = warp_reduce_sum<lanes>(make_float2(pu, pw));

        if (t < nt) {
            const float ct = expf(fminf(sh_G[t], GGML_CUDA_GDN_EXP_CLAMP));
            const float vt = v[sequence * sv3 + (int64_t) t * sv2 + h_idx * sv1 + col];
            u [t] = sh_bt[t] * (vt - ct * red.x);
            ob[t] = ct * red.y;
        } else {
            u [t] = 0.0f;
            ob[t] = 0.0f;
        }
    }

    // ---- phase 3: forward substitution -- serial in t, but no cross-lane op ------------
    float d[TM];
#pragma unroll
    for (int t = 0; t < TM; t++) {
        float acc = u[t];
#pragma unroll
        for (int j = 0; j < TM; j++) {
            if (j < t) {
                acc -= sh_A[t * TM + j] * d[j];
            }
        }
        d[t] = acc;
    }

    // ---- phase 4a: outputs ------------------------------------------------------------
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
#pragma unroll
    for (int t = 0; t < TM; t++) {
        if (t < nt) {
            float o = ob[t];
#pragma unroll
            for (int j = 0; j < TM; j++) {
                if (j <= t) {
                    o += sh_P[t * TM + j] * d[j];
                }
            }
            if (lane == 0) {
                attn_data[(int64_t) t * S_v * H + col] = o * scale;
            }
        }
    }

    // ---- phase 4b: state updates and snapshots ----------------------------------------
#pragma unroll
    for (int t = 0; t < TM; t++) {
        if (t < nt) {
            const float gam = expf(fminf(sh_gl[t], GGML_CUDA_GDN_EXP_CLAMP));
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes + lane;
                s_shard[r] = gam * s_shard[r] + sh_k[t * S_v + i] * d[t];
            }

            if constexpr (keep_rs_t) {
                const int target_slot = nt - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * snap = st_out + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * lanes + lane;
                        snap[(int64_t) col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes + lane;
            st_out[(int64_t) col * S_v + i] = s_shard[r];
        }
    }
}

// Smallest supported chunk width that covers n_tokens. The chunked kernel unrolls its
// substitution / output / state phases over this width, so a T=8 verify block must not be
// compiled at 16: the fixed part of the kernel (TM cross-lane reductions in phase 2, a
// TM(TM-1)/2 FMA chain in phase 3, TM*S_v*2 floats of staging LDS) is what made T=8 cost
// 105 us/layer against 8.6 us for the T=1 autoregressive kernel. Widths are powers of two
// to bound template instantiations to 4 per (S_v, keep_rs_t).
static int gdn_chunk_width(int64_t n_tokens) {
    if (n_tokens <= 2) { return 2; }
    if (n_tokens <= 4) { return 4; }
    if (n_tokens <= 8) { return 8; }
    return GGML_CUDA_GDN_CHUNK_MAX;
}

template <bool keep_rs_t>
static void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const int tm = gdn_chunk_width(n_tokens);
    GGML_ASSERT(n_tokens <= tm);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#define GGML_CUDA_GDN_LAUNCH_CHUNKED(SV, TMV)                                                \
    ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<(SV), (TMV), keep_rs_t>,            \
        launch_params, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,                      \
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                      \
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K)

#define GGML_CUDA_GDN_LAUNCH_CHUNKED_SV(SV)                                                  \
    switch (tm) {                                                                            \
        case  2: GGML_CUDA_GDN_LAUNCH_CHUNKED((SV),  2); break;                              \
        case  4: GGML_CUDA_GDN_LAUNCH_CHUNKED((SV),  4); break;                              \
        case  8: GGML_CUDA_GDN_LAUNCH_CHUNKED((SV),  8); break;                              \
        case 16: GGML_CUDA_GDN_LAUNCH_CHUNKED((SV), 16); break;                              \
        default: GGML_ABORT("fatal error");                                                  \
    }

    switch (S_v) {
        case  16: GGML_CUDA_GDN_LAUNCH_CHUNKED_SV( 16); break;
        case  32: GGML_CUDA_GDN_LAUNCH_CHUNKED_SV( 32); break;
        case  64: GGML_CUDA_GDN_LAUNCH_CHUNKED_SV( 64); break;
        case 128: GGML_CUDA_GDN_LAUNCH_CHUNKED_SV(128); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

#undef GGML_CUDA_GDN_LAUNCH_CHUNKED_SV
#undef GGML_CUDA_GDN_LAUNCH_CHUNKED
}

// Chunk width for the long-sequence prefill kernel below. Independent of
// GGML_CUDA_GDN_CHUNK_MAX (that one bounds the MTP-verify chunked kernel's *compiled*
// unroll width and stays 16). 64 is the standard FLA chunk_gated_delta_rule width.
#define GGML_CUDA_GDN_PREFILL_CHUNK 64

// ---------------------------------------------------------------------------------------
// Chunked GDA kernel for LONG prefill token blocks (n_tokens > GGML_CUDA_GDN_CHUNK_MAX,
// scalar gate only). Same UT-transform algebra as gated_delta_net_chunked_cuda above (read
// that comment first for the derivation of A, P, u, d, o) but looped over sequential
// GGML_CUDA_GDN_PREFILL_CHUNK-token chunks instead of compiled for one fixed-width block:
// the running state S is carried across chunks in registers (s_shard), the same way the
// per-token autoregressive kernel carries s_shard across tokens.
//
// Precision: the K/Q staging tiles and the P (inclusive) Gram block are kept in shared
// memory as fp16 so the running-state registers plus two C x S_v chunk tiles fit inside a
// 64 KB workgroup LDS budget at S_v = 128 (see the LDS accounting at the launcher). The
// running state, the A (strict) Gram block that feeds the serial forward-substitution
// recursion, and all elementwise decay math (g, beta, prefix sums) stay fp32. Every dot
// product still accumulates in fp32 registers -- only the stored operands are rounded to
// fp16 -- which is the accuracy tradeoff flagged in test_gated_delta_net's max_nmse_err()
// override for n_seq_tokens > GGML_CUDA_GDN_CHUNK_MAX.
//
// This kernel NEVER emits K-snapshot slots. It always produces exactly the plain final
// state after its own n_tokens tokens, in the SAME layout gated_delta_net_cuda's
// !keep_rs_t branch writes (state + (sequence*H+h_idx)*S_v*S_v). When K > 1,
// ggml_cuda_op_gated_delta_net_impl composes this kernel over the token prefix [0, T0)
// with a short tail of the LAST K tokens re-run by the existing autoregressive kernel
// (which alone knows how to emit snapshot slots), feeding this kernel's output state to
// the tail launch as curr_state. Ragged final chunks (n_tokens % C != 0) are handled by
// the same "if (t < nt)" masking gated_delta_net_chunked_cuda already uses: rows t >= nt
// are zeroed (beta = 0 -> d = 0, so they cannot perturb the running state) and never
// written to dst or folded into the state update.
// ---------------------------------------------------------------------------------------
template <int S_v, int C>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 1)
gated_delta_net_prefill_cuda(const float * q,
                              const float * k,
                              const float * v,
                              const float * g,
                              const float * beta,
                              const float * curr_state,
                              float *       dst,
                              float *       state,
                              int64_t       H,
                              int64_t       n_tokens, // T0: this launch's own token count
                              int64_t       n_seqs,
                              int64_t       sq1,
                              int64_t       sq2,
                              int64_t       sq3,
                              int64_t       sv1,
                              int64_t       sv2,
                              int64_t       sv3,
                              int64_t       sb1,
                              int64_t       sb2,
                              int64_t       sb3,
                              const uint3   neqk1_magic,
                              const uint3   rq3_magic,
                              float         scale,
                              int64_t       dst_seq_stride) {
    constexpr int lanes         = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % lanes == 0, "S_v must be a multiple of the reduction width");
    constexpr int rows_per_lane = S_v / lanes;
    constexpr int nwarps        = 4;

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      wid      = threadIdx.y;
    const int      col      = blockIdx.z * nwarps + wid;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int     nthreads = lanes * nwarps;
    const int     tid      = wid * lanes + lane;
    const int64_t T0       = n_tokens;

    __shared__ half  sh_k [C * S_v];
    __shared__ half  sh_q [C * S_v];
    __shared__ float sh_gl[C];        // per-token raw log decay
    __shared__ float sh_G [C];        // inclusive prefix sum of sh_gl
    __shared__ float sh_bt[C];        // beta
    __shared__ float sh_A [C * C];    // beta_t * exp(G_t-G_j) * (k_t . k_j), j <  t   (fp32)
    __shared__ half  sh_P [C * C];    // exp(G_t-G_j) * (k_j . q_t),          j <= t   (fp16)

    ggml_cuda_pdl_sync();

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    const float * s_in             = curr_state + state_in_offset + (int64_t) col * S_v;
    float *       st_out           = state + state_out_offset;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = s_in[r * lanes + lane];
    }

    float * attn_data = dst + (sequence * dst_seq_stride * H + h_idx) * S_v;

    for (int64_t cbase = 0; cbase < T0; cbase += C) {
        // guard the previous iteration's phase-4 shared reads before phase 0 overwrites them
        __syncthreads();

        const int64_t remain = T0 - cbase;
        const int     nt     = (int) (remain < (int64_t) C ? remain : (int64_t) C);

        // ---- phase 0: stage k / q / beta / gate for this chunk -------------------------
        for (int t = 0; t < nt; t++) {
            const int64_t tg  = cbase + t;
            const float * k_t = k + iq3 * sq3 + tg * sq2 + iq1 * sq1;
            const float * q_t = q + iq3 * sq3 + tg * sq2 + iq1 * sq1;
            for (int i = tid; i < S_v; i += nthreads) {
                sh_k[t * S_v + i] = __float2half(k_t[i]);
                sh_q[t * S_v + i] = __float2half(q_t[i]);
            }
        }
        for (int i = tid; i < C * C; i += nthreads) {
            sh_A[i] = 0.0f;
            sh_P[i] = __float2half(0.0f);
        }
        for (int t = tid; t < nt; t += nthreads) {
            const int64_t tg = cbase + t;
            const int64_t gb = sequence * sb3 + tg * sb2 + h_idx * sb1;
            sh_gl[t] = g[gb];
            sh_bt[t] = beta[gb];
        }
        __syncthreads();

        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < nt; t++) {
                acc     += sh_gl[t];
                sh_G[t]  = acc;
            }
        }
        __syncthreads();

        // ---- phase 1: Gram blocks (independent reductions) -----------------------------
        const int npairs = (nt * (nt + 1)) / 2;
        const int niter   = (npairs + nwarps - 1) / nwarps;

        for (int it = 0; it < niter; it++) {
            const int  p     = it * nwarps + wid;
            const bool valid = p < npairs;

            int t = 0;
            int j = valid ? p : 0;
            while (j > t) {
                j -= t + 1;
                t++;
            }

            float pkk = 0.0f;
            float pkq = 0.0f;
            if (valid) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int   i  = r * lanes + lane;
                    const float kt = __half2float(sh_k[t * S_v + i]);
                    const float kj = __half2float(sh_k[j * S_v + i]);
                    const float qt = __half2float(sh_q[t * S_v + i]);
                    pkk += kt * kj;
                    pkq += kj * qt;
                }
            }

            const float2 red = warp_reduce_sum<lanes>(make_float2(pkk, pkq));

            if (valid && lane == 0) {
                const float e = expf(fminf(sh_G[t] - sh_G[j], GGML_CUDA_GDN_EXP_CLAMP));
                if (j < t) {
                    sh_A[t * C + j] = sh_bt[t] * e * red.x;
                }
                sh_P[t * C + j] = __float2half(e * red.y);
            }
        }
        __syncthreads();

        // ---- phase 2: u_t and the S_0 part of o_t (independent reductions) -------------
        float u [C];
        float ob[C];
        for (int t = 0; t < C; t++) {
            float pu = 0.0f;
            float pw = 0.0f;
            if (t < nt) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes + lane;
                    pu += s_shard[r] * __half2float(sh_k[t * S_v + i]);
                    pw += s_shard[r] * __half2float(sh_q[t * S_v + i]);
                }
            }

            const float2 red = warp_reduce_sum<lanes>(make_float2(pu, pw));

            if (t < nt) {
                const int64_t tg = cbase + t;
                const float   ct = expf(fminf(sh_G[t], GGML_CUDA_GDN_EXP_CLAMP));
                const float   vt = v[sequence * sv3 + tg * sv2 + h_idx * sv1 + col];
                u [t] = sh_bt[t] * (vt - ct * red.x);
                ob[t] = ct * red.y;
            } else {
                u [t] = 0.0f;
                ob[t] = 0.0f;
            }
        }

        // ---- phase 3: forward substitution -- serial in t, no cross-lane op ------------
        float d[C];
        for (int t = 0; t < C; t++) {
            float acc = u[t];
            for (int j = 0; j < t && j < C; j++) {
                acc -= sh_A[t * C + j] * d[j];
            }
            d[t] = acc;
        }

        // ---- phase 4a: outputs -----------------------------------------------------------
        for (int t = 0; t < C; t++) {
            if (t < nt) {
                float o = ob[t];
                for (int j = 0; j <= t; j++) {
                    o += __half2float(sh_P[t * C + j]) * d[j];
                }
                if (lane == 0) {
                    attn_data[(cbase + t) * S_v * H + col] = o * scale;
                }
            }
        }

        // ---- phase 4b: state update (no snapshots -- see kernel comment) ---------------
        for (int t = 0; t < C; t++) {
            if (t < nt) {
                const float gam = expf(fminf(sh_gl[t], GGML_CUDA_GDN_EXP_CLAMP));
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes + lane;
                    s_shard[r]  = gam * s_shard[r] + __half2float(sh_k[t * S_v + i]) * d[t];
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * lanes + lane;
        st_out[col * S_v + i] = s_shard[r];
    }
}

static void launch_gated_delta_net_prefill(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t dst_seq_stride, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#define GGML_CUDA_GDN_LAUNCH_PREFILL(SV)                                                    \
    ggml_cuda_kernel_launch(gated_delta_net_prefill_cuda<(SV), GGML_CUDA_GDN_PREFILL_CHUNK>, \
        launch_params, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,                      \
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                      \
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, dst_seq_stride)

    switch (S_v) {
        case  16: GGML_CUDA_GDN_LAUNCH_PREFILL( 16); break;
        case  32: GGML_CUDA_GDN_LAUNCH_PREFILL( 32); break;
        case  64: GGML_CUDA_GDN_LAUNCH_PREFILL( 64); break;
        case 128: GGML_CUDA_GDN_LAUNCH_PREFILL(128); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

#undef GGML_CUDA_GDN_LAUNCH_PREFILL
}

// =========================================================================================
// WMMA long-prefill kernel (RDNA4 matrix cores). Same chunked GDA algebra as
// gated_delta_net_prefill_cuda above, but every per-chunk matmul (A = K K^T, P = Q K^T,
// v_new = u - w.S, o_inter = (c*q).S, S += (k*decay)^T v_new) runs on the WMMA units via
// ggml_cuda_mma's tile<> abstraction (ggml/src/ggml-cuda/mma.cuh), instead of the scalar
// warp-reduction dot products the fallback kernel above uses. Only the O(C^2) forward
// substitution for u/w stays scalar (it is inherently serial and small).
//
// This uses the FLA-standard decomposition (distinct from the fallback kernel's folded
// per-token recurrence): u = T.(beta*v), w = T.(beta*c*k) are computed WITHOUT the running
// state (T = (I+A)^-1 applied via forward substitution), then v_new = u - w.S and
// o_inter = (c*q).S are separate matmuls against the state carried from the previous chunk.
// u, w, v_new and the elementwise decay/beta/gate math all stay fp32; only WMMA operands
// (K, Q, w, c*q, k*decay, v_new-as-operand, S-as-operand) are rounded to fp16.
//
// This template (and its host-side launcher below) is defined UNCONDITIONALLY, not behind
// an `#if defined(AMD_WMMA_AVAILABLE)` guard: HIP compiles a .cu file's host and device code
// in separate passes, and RDNA4/RDNA3/AMD_WMMA_AVAILABLE are only guaranteed visible in the
// device-code pass -- wrapping the launcher (host code) in that guard silently compiled it
// out on the host pass in testing (caught as an "unused variable" warning on the dispatcher's
// `cc`, not a runtime symptom -- worth knowing about since it is an easy mistake to repeat
// elsewhere in this file). mma.cuh's own tile<>/mma() specializations already fall back to
// NO_DEVICE_CODE (or a different, CUDA/RDNA3-shaped lane mapping this kernel does not intend
// to use) on architectures this kernel isn't validated for, so the body below compiles
// everywhere; ggml_cuda_op_gated_delta_net_impl's `wmma_ok` runtime check (amd_wmma_available
// && GGML_CUDA_CC_IS_RDNA4) is what actually keeps this kernel from being LAUNCHED anywhere
// but RDNA4 (gfx12) hardware, where get_i/get_j (mma.cuh) are validated for this fragment
// layout.
template <int S_v, int C, bool DebugCheck>
__global__ void __launch_bounds__(128, 1)
gated_delta_net_prefill_wmma_cuda(const float * q,
                                   const float * k,
                                   const float * v,
                                   const float * g,
                                   const float * beta,
                                   const float * curr_state,
                                   float *       dst,
                                   float *       state,
                                   int64_t       H,
                                   int64_t       n_tokens, // T0: this launch's own token count
                                   int64_t       n_seqs,
                                   int64_t       sq1,
                                   int64_t       sq2,
                                   int64_t       sq3,
                                   int64_t       sv1,
                                   int64_t       sv2,
                                   int64_t       sv3,
                                   int64_t       sb1,
                                   int64_t       sb2,
                                   int64_t       sb3,
                                   const uint3   neqk1_magic,
                                   const uint3   rq3_magic,
                                   float         scale,
                                   int64_t       dst_seq_stride) {
    // ---- geometry ------------------------------------------------------------------------
    // Workgroup = (v-head h, sequence, 32-column v-STRIPE). The gated delta rule is
    // independent per value column: S[:,n] only depends on q/k/beta/g and S[:,n] itself, so
    // splitting the whole S_v range across separate workgroups (instead of across waves
    // within ONE workgroup, as the previous version did) means each workgroup only ever
    // holds a 32-column slice of the running state in registers -- no cross-wave "owner"
    // rotation, no giant persistent accumulator. The Gram matrices A, P and the
    // forward-substitution matrix T are recomputed redundantly per stripe (they are 64x64,
    // "tiny" per the design review that asked for this).
    constexpr int WT   = 16;   // WMMA tile width (M=N=K=16)
    constexpr int SW   = 32;   // stripe width (fixed, independent of S_v)
    constexpr int nwarps = 4;
    constexpr int CT   = C / WT;         // token tiles (4 for C=64)
    constexpr int ST   = S_v / WT;       // total S_v row-tiles (8 @128, 4 @64)
    constexpr int SNT  = SW / WT;        // stripe col-tiles (2, fixed)
    constexpr int RPW  = ST / nwarps;    // S row-tiles owned by each wave (2 @128, 1 @64)
    static_assert(C % WT == 0 && S_v % WT == 0 && SW % WT == 0, "S_v/C/SW must be multiples of 16");
    static_assert(ST % nwarps == 0, "S_v/16 must split evenly across 4 waves");
    static_assert(CT == nwarps, "Gram-phase wave->row-tile assignment assumes C/16 == 4");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const uint32_t sidx     = blockIdx.z;             // stripe index: this WG's columns are [sidx*SW, sidx*SW+SW)
    const int      lane     = threadIdx.x;            // 0..31 (one full wave32 per warp, required by WMMA)
    const int      wid      = threadIdx.y;            // 0..3
    const int      tid      = wid * 32 + lane;
    constexpr int  nthreads = 32 * nwarps;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const int64_t  T0  = n_tokens;
    const int      col0 = (int) sidx * SW;            // this WG's first output/state column

    // ---- LDS: distinct, non-aliased buffers (no phase-to-phase byte reuse this time --
    // budget allows it and it removes a whole class of aliasing bugs). ~59 KB at S_v=128.
    __shared__ half2 sh_k[C * S_v / 2];    // -> becomes w  in place after phase 2
    __shared__ half2 sh_q[C * S_v / 2];    // -> becomes cq in place after phase 3a, then k*decay after phase 3b
    __shared__ half2 sh_v[C * SW / 2];     // this stripe's V only
    __shared__ half  sh_A[C * C];          // -> becomes S-stage (transposed, full S_v x SW) after phase 2
    __shared__ half  sh_P[C * C];
    __shared__ half2 sh_u[C * SW / 2];
    __shared__ half2 sh_vnT[SW * C / 2];   // v_new, transposed: vnT[n][t]
    __shared__ float sh_G[C];              // inclusive prefix sum of the raw log-decay
    __shared__ float sh_Gc;                // sh_G[nt-1]: this chunk's total log-decay

    ggml_cuda_pdl_sync();

    // ---- running state: THIS stripe only, held as WMMA accumulator tiles in registers for
    // the whole kernel. Wave wid owns S row-tiles [wid*RPW, wid*RPW+RPW), all SNT col-tiles
    // (the full stripe): RPW*SNT tiles/wave = 4 tiles (32 VGPRs/lane) at S_v=128.
    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    const float * s_in   = curr_state + state_in_offset;
    float *       st_out = state + state_out_offset;

    mma::tile<16, 16, float> S_tiles[RPW][SNT];
#pragma unroll
    for (int m = 0; m < RPW; m++) {
#pragma unroll
        for (int n = 0; n < SNT; n++) {
            mma::load_generic(S_tiles[m][n], s_in + (int64_t) (wid * RPW + m) * WT * S_v + col0 + n * WT, S_v);
        }
    }

    float * attn_data = dst + (sequence * dst_seq_stride * H + h_idx) * S_v;

    for (int64_t cbase = 0; cbase < T0; cbase += C) {
        __syncthreads(); // guard previous chunk's LDS reads before this chunk overwrites them

        const int64_t remain = T0 - cbase;
        const int     nt     = (int) (remain < (int64_t) C ? remain : (int64_t) C);
        const bool    dbg    = DebugCheck && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && cbase == 0;

        // ---- phase 0: stage K, Q (full S_v) and V (this stripe only) as fp16; prefix-sum
        // the raw log-decay. beta is re-read from global memory wherever needed (a cheap
        // per-token scalar) instead of staged, same as before.
        for (int t = 0; t < nt; t++) {
            const int64_t tg  = cbase + t;
            const float * k_t = k + iq3 * sq3 + tg * sq2 + iq1 * sq1;
            const float * q_t = q + iq3 * sq3 + tg * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + tg * sv2 + h_idx * sv1 + col0;
            for (int i = tid; i < S_v; i += nthreads) {
                ((half *) sh_k)[t * S_v + i] = __float2half(k_t[i]);
                ((half *) sh_q)[t * S_v + i] = __float2half(q_t[i]);
            }
            for (int i = tid; i < SW; i += nthreads) {
                ((half *) sh_v)[t * SW + i] = __float2half(v_t[i]);
            }
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < nt; t++) {
                acc     += g[sequence * sb3 + (cbase + t) * sb2 + h_idx * sb1];
                sh_G[t]  = acc;
            }
            for (int t = nt; t < C; t++) {
                sh_G[t] = acc; // padding rows: never actually read, kept finite defensively
            }
            sh_Gc = sh_G[nt - 1];
        }
        __syncthreads();

        // ---- phase 1 (WMMA): Gram blocks A = K K^T (masked j<t), P = Q K^T (masked j<=t).
        // Wave wid owns output row-tile wid (CT == nwarps == 4), looping all n-tiles and all
        // S_v/16 contraction tiles.
        {
            mma::tile<16, 16, float> A_acc[CT];
            mma::tile<16, 16, float> P_acc[CT];
            for (int nti = 0; nti < CT; nti++) {
#pragma unroll
                for (int kk = 0; kk < ST; kk++) {
                    mma::tile<16, 8, half2> K_m, K_n, Q_m;
                    mma::load_ldmatrix(K_m, sh_k + (int64_t) wid * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::load_ldmatrix(K_n, sh_k + (int64_t) nti * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::load_ldmatrix(Q_m, sh_q + (int64_t) wid * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::mma(A_acc[nti], K_m, K_n);
                    mma::mma(P_acc[nti], Q_m, K_n);
                }
            }
            for (int nti = 0; nti < CT; nti++) {
#pragma unroll
                for (int l = 0; l < A_acc[nti].ne; l++) {
                    const int t = wid * WT + A_acc[nti].get_i(l);
                    const int j = nti * WT + A_acc[nti].get_j(l);
                    const float e = expf(fminf(sh_G[t] - sh_G[j], GGML_CUDA_GDN_EXP_CLAMP));
                    float a_val = 0.0f;
                    if (j < t && t < nt && j < nt) {
                        const float beta_t = beta[sequence * sb3 + (cbase + t) * sb2 + h_idx * sb1];
                        a_val = beta_t * e * A_acc[nti].x[l];
                    }
                    float p_val = (j <= t) ? e * P_acc[nti].x[l] : 0.0f;
                    sh_A[t * C + j] = __float2half(a_val);
                    sh_P[t * C + j] = __float2half(p_val);
                }
            }
        }
        __syncthreads();

        if (dbg && tid == 0) {
            float max_diff = 0.0f;
            for (int t = 0; t < nt; t++) {
                for (int j = 0; j <= t; j++) {
                    float kk_dot = 0.0f, qk_dot = 0.0f;
                    for (int i = 0; i < S_v; i++) {
                        const float kt = __half2float(((half *) sh_k)[t * S_v + i]);
                        const float kj = __half2float(((half *) sh_k)[j * S_v + i]);
                        const float qt = __half2float(((half *) sh_q)[t * S_v + i]);
                        kk_dot += kt * kj;
                        qk_dot += kj * qt;
                    }
                    const float e = expf(fminf(sh_G[t] - sh_G[j], GGML_CUDA_GDN_EXP_CLAMP));
                    const float beta_t = beta[sequence * sb3 + (cbase + t) * sb2 + h_idx * sb1];
                    const float a_ref = (j < t) ? beta_t * e * kk_dot : 0.0f;
                    const float p_ref = e * qk_dot;
                    max_diff = fmaxf(max_diff, fabsf(a_ref - __half2float(sh_A[t * C + j])));
                    max_diff = fmaxf(max_diff, fabsf(p_ref - __half2float(sh_P[t * C + j])));
                }
            }
            printf("[GGML_GDN_WMMA_CHECK] chunk0 A/P max_abs_diff=%g\n", (double) max_diff);
        }

        // ---- phase 2 (scalar): forward substitution for u = T.(beta*v_stripe) [C,SW] and
        // w = T.(beta*c*k) [C,S_v]. Per-column, serial in t, writing straight into LDS (no
        // per-thread float[64] array -- that spilled in an earlier version of this kernel and
        // was the dominant cost of the SCALAR fallback path this WMMA kernel replaces). w is
        // written in place over sh_k (already fully consumed by phase 1 above).
        for (int col = tid; col < S_v; col += nthreads) {
            for (int t = 0; t < C; t++) {
                float bck = 0.0f;
                if (t < nt) {
                    const int64_t gb     = sequence * sb3 + (cbase + t) * sb2 + h_idx * sb1;
                    const float   beta_t = beta[gb];
                    const float   kt     = __half2float(((half *) sh_k)[t * S_v + col]);
                    const float   ct     = expf(fminf(sh_G[t], GGML_CUDA_GDN_EXP_CLAMP));
                    bck = beta_t * ct * kt;
                }
                float acc = bck;
                for (int j = 0; j < t; j++) {
                    const float a  = __half2float(sh_A[t * C + j]);
                    const float wj = __half2float(((half *) sh_k)[j * S_v + col]); // already-written w[j]
                    acc -= a * wj;
                }
                ((half *) sh_k)[t * S_v + col] = __float2half(acc); // now w[t][col]
            }
        }
        for (int col = tid; col < SW; col += nthreads) {
            for (int t = 0; t < C; t++) {
                float bv = 0.0f;
                if (t < nt) {
                    const int64_t gb     = sequence * sb3 + (cbase + t) * sb2 + h_idx * sb1;
                    const float   beta_t = beta[gb];
                    const float   vt     = __half2float(((half *) sh_v)[t * SW + col]);
                    bv = beta_t * vt;
                }
                float acc = bv;
                for (int j = 0; j < t; j++) {
                    const float a  = __half2float(sh_A[t * C + j]);
                    const float uj = __half2float(((half *) sh_u)[j * SW + col]); // already-written u[j]
                    acc -= a * uj;
                }
                ((half *) sh_u)[t * SW + col] = __float2half(acc); // now u[t][col]
            }
        }
        // turn sh_q (fully consumed by phase 1) into c*q, in place.
        for (int t = 0; t < C; t++) {
            const float ct = expf(fminf(sh_G[t], GGML_CUDA_GDN_EXP_CLAMP));
            for (int i = tid; i < S_v; i += nthreads) {
                const float qv = __half2float(((half *) sh_q)[t * S_v + i]);
                ((half *) sh_q)[t * S_v + i] = __float2half(ct * qv);
            }
        }
        __syncthreads();

        // ---- stage S into LDS (transposed: Sstage[n][r] = S[r][n]), ALL WAVES IN PARALLEL --
        // each wave writes only its own row-tile range, so there is no cross-wave contention
        // and exactly one barrier is needed. Reuses sh_A's now-dead bytes (same C*C footprint
        // as SW*S_v: 64*64 == 32*128).
        half * sh_Sstage = (half *) sh_A; // [SW][S_v]
#pragma unroll
        for (int m = 0; m < RPW; m++) {
#pragma unroll
            for (int n = 0; n < SNT; n++) {
#pragma unroll
                for (int l = 0; l < S_tiles[m][n].ne; l++) {
                    const int r       = (wid * RPW + m) * WT + S_tiles[m][n].get_i(l);
                    const int n_local = n * WT + S_tiles[m][n].get_j(l);
                    sh_Sstage[n_local * S_v + r] = __float2half(S_tiles[m][n].x[l]);
                }
            }
        }
        __syncthreads();

        if (dbg && tid == 0) {
            float max_diff = 0.0f;
            for (int r = 0; r < S_v; r++) {
                for (int n = 0; n < SW; n++) {
                    const float sref = s_in[r * S_v + col0 + n];
                    max_diff = fmaxf(max_diff, fabsf(sref - __half2float(sh_Sstage[n * S_v + r])));
                }
            }
            printf("[GGML_GDN_WMMA_CHECK] chunk0 S-stage max_abs_diff=%g (vs curr_state, fp16 rounding only)\n",
                   (double) max_diff);
        }

        // ---- v_new = u - w.S (WMMA): wave wid owns token M-tile wid (CT == nwarps), both
        // N-tiles (the full stripe) -- 8 total output tiles / 4 waves = 2 tiles/wave.
        // Contraction is over the FULL S_v range via the just-staged Sstage, so every wave
        // sees the complete state regardless of which row-tiles it itself owns.
        {
            mma::tile<16, 16, float> D[SNT];
#pragma unroll
            for (int nlt = 0; nlt < SNT; nlt++) {
#pragma unroll
                for (int kk = 0; kk < ST; kk++) {
                    mma::tile<16, 8, half2> W_m, S_n;
                    mma::load_ldmatrix(W_m, sh_k + (int64_t) wid * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::load_ldmatrix(S_n, (half2 *) sh_Sstage + (int64_t) nlt * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::mma(D[nlt], W_m, S_n);
                }
            }
#pragma unroll
            for (int nlt = 0; nlt < SNT; nlt++) {
#pragma unroll
                for (int l = 0; l < D[nlt].ne; l++) {
                    const int t       = wid * WT + D[nlt].get_i(l);
                    const int n_local = nlt * WT + D[nlt].get_j(l);
                    const float u_val = __half2float(((half *) sh_u)[t * SW + n_local]);
                    const float vnew  = (t < nt) ? (u_val - D[nlt].x[l]) : 0.0f;
                    ((half *) sh_vnT)[n_local * C + t] = __float2half(vnew);
                }
            }
        }
        __syncthreads();

        if (dbg && tid == 0) {
            float max_diff = 0.0f;
            for (int t = 0; t < nt; t++) {
                for (int n = 0; n < SW; n++) {
                    float wS = 0.0f;
                    for (int r = 0; r < S_v; r++) {
                        wS += __half2float(((half *) sh_k)[t * S_v + r]) * __half2float(sh_Sstage[n * S_v + r]);
                    }
                    const float uv  = __half2float(((half *) sh_u)[t * SW + n]);
                    const float ref = uv - wS;
                    max_diff = fmaxf(max_diff, fabsf(ref - __half2float(((half *) sh_vnT)[n * C + t])));
                }
            }
            printf("[GGML_GDN_WMMA_CHECK] chunk0 v_new max_abs_diff=%g\n", (double) max_diff);
        }

        // ---- o_inter = (c*q).S (WMMA), same tile assignment as v_new; combine with the
        // scalar intra-chunk term o_intra = P . v_new (P is already zero above the diagonal
        // from phase 1's masking) and write the final output for this stripe.
        {
            mma::tile<16, 16, float> D[SNT];
#pragma unroll
            for (int nlt = 0; nlt < SNT; nlt++) {
#pragma unroll
                for (int kk = 0; kk < ST; kk++) {
                    mma::tile<16, 8, half2> CQ_m, S_n;
                    mma::load_ldmatrix(CQ_m, sh_q + (int64_t) wid * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::load_ldmatrix(S_n, (half2 *) sh_Sstage + (int64_t) nlt * WT * (S_v / 2) + kk * 8, S_v / 2);
                    mma::mma(D[nlt], CQ_m, S_n);
                }
            }
#pragma unroll
            for (int nlt = 0; nlt < SNT; nlt++) {
#pragma unroll
                for (int l = 0; l < D[nlt].ne; l++) {
                    const int t = wid * WT + D[nlt].get_i(l);
                    if (t >= nt) {
                        continue;
                    }
                    const int n_local = nlt * WT + D[nlt].get_j(l);
                    const int n_col   = col0 + n_local;

                    float o_intra = 0.0f;
                    for (int j = 0; j <= t; j++) {
                        const float p_val = __half2float(sh_P[t * C + j]);
                        const float vn_j  = __half2float(((half *) sh_vnT)[n_local * C + j]);
                        o_intra += p_val * vn_j;
                    }
                    const float o = scale * (o_intra + D[nlt].x[l]);
                    attn_data[(cbase + t) * S_v * H + n_col] = o;

                    if (dbg && lane == 0 && wid == 0 && t < 2) {
                        float cqS = 0.0f;
                        for (int r = 0; r < S_v; r++) {
                            cqS += __half2float(((half *) sh_q)[t * S_v + r]) * __half2float(sh_Sstage[n_local * S_v + r]);
                        }
                        printf("[GGML_GDN_WMMA_CHECK] chunk0 o_inter[t=%d,n=%d] wmma=%g scalar=%g diff=%g\n",
                               t, n_local, (double) D[nlt].x[l], (double) cqS, (double) fabsf(D[nlt].x[l] - cqS));
                    }
                }
            }
        }
        __syncthreads();

        // ---- build k*decay (transposed [S_v][C]) for the S update, reloading k from global
        // (sh_k's fp16 copy became w above) -- reuses sh_q's (now dead) bytes.
        half * sh_kdecay = (half *) sh_q; // [S_v][C]
        for (int r = tid; r < S_v; r += nthreads) {
            for (int t = 0; t < C; t++) {
                float val = 0.0f;
                if (t < nt) {
                    const float kt  = k[iq3 * sq3 + (int64_t) (cbase + t) * sq2 + iq1 * sq1 + r];
                    const float dec = expf(fminf(sh_Gc - sh_G[t], GGML_CUDA_GDN_EXP_CLAMP));
                    val = kt * dec;
                }
                sh_kdecay[r * C + t] = __float2half(val);
            }
        }
        __syncthreads();

        // ---- S <- exp(Gc)*S + (k*decay)^T . v_new (WMMA): each wave updates only its own
        // RPW*SNT tiles (no cross-wave traffic at all for this step -- S is the OUTPUT here,
        // and wave ownership of S's row-tiles already matches the output tiling exactly).
        {
            const float c_C = expf(fminf(sh_Gc, GGML_CUDA_GDN_EXP_CLAMP));
#pragma unroll
            for (int m = 0; m < RPW; m++) {
#pragma unroll
                for (int n = 0; n < SNT; n++) {
#pragma unroll
                    for (int l = 0; l < S_tiles[m][n].ne; l++) {
                        S_tiles[m][n].x[l] *= c_C;
                    }
                }
            }
            // Sized 1x1 (not RPW x SNT) when DebugCheck is off so this costs ~0 registers on
            // the default fast path -- DebugCheck is a template param specifically so this
            // and the printf blocks below fold away entirely at compile time when unused.
            constexpr int DBG_RPW = DebugCheck ? RPW : 1;
            constexpr int DBG_SNT = DebugCheck ? SNT : 1;
            mma::tile<16, 16, float> S_before[DBG_RPW][DBG_SNT];
            if constexpr (DebugCheck) {
                if (dbg) {
#pragma unroll
                    for (int m = 0; m < RPW; m++) {
#pragma unroll
                        for (int n = 0; n < SNT; n++) {
                            S_before[m][n] = S_tiles[m][n];
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < RPW; m++) {
#pragma unroll
                for (int n = 0; n < SNT; n++) {
#pragma unroll
                    for (int kk = 0; kk < CT; kk++) {
                        mma::tile<16, 8, half2> Kd_m, Vn_n;
                        mma::load_ldmatrix(Kd_m, (half2 *) sh_kdecay + (int64_t) (wid * RPW + m) * WT * (C / 2) + kk * 8, C / 2);
                        mma::load_ldmatrix(Vn_n, sh_vnT + (int64_t) n * WT * (C / 2) + kk * 8, C / 2);
                        mma::mma(S_tiles[m][n], Kd_m, Vn_n);
                    }
                }
            }
            if constexpr (DebugCheck) {
            if (dbg && lane == 0 && wid == 0) {
                float max_diff = 0.0f;
                for (int m = 0; m < RPW; m++) {
                    for (int n = 0; n < SNT; n++) {
                        for (int l = 0; l < S_tiles[m][n].ne; l++) {
                            const int r       = (wid * RPW + m) * WT + S_tiles[m][n].get_i(l);
                            const int n_local = n * WT + S_tiles[m][n].get_j(l);
                            float outer = 0.0f;
                            for (int t = 0; t < C; t++) {
                                outer += __half2float(sh_kdecay[r * C + t]) * __half2float(((half *) sh_vnT)[n_local * C + t]);
                            }
                            const float ref = S_before[m][n].x[l] + outer;
                            max_diff = fmaxf(max_diff, fabsf(ref - S_tiles[m][n].x[l]));
                        }
                    }
                }
                printf("[GGML_GDN_WMMA_CHECK] chunk0 S-update max_abs_diff=%g\n", (double) max_diff);
            }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < RPW; m++) {
#pragma unroll
        for (int n = 0; n < SNT; n++) {
#pragma unroll
            for (int l = 0; l < S_tiles[m][n].ne; l++) {
                const int r = (wid * RPW + m) * WT + S_tiles[m][n].get_i(l);
                const int c = col0 + n * WT + S_tiles[m][n].get_j(l);
                st_out[(int64_t) r * S_v + c] = S_tiles[m][n].x[l];
            }
        }
    }
}

static void launch_gated_delta_net_prefill_wmma(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t dst_seq_stride, cudaStream_t stream) {
    // GGML_GDN_WMMA_CHECK=1: for block (0,0,0) and the first chunk only, recompute A, P,
    // v_new, o_inter and the S update via scalar loops from the SAME staged LDS operands and
    // printf the max abs diff per product -- run with `HIP_VISIBLE_DEVICES` narrowed and
    // small shapes (e.g. head_count=4,head_size=64,n_seq_tokens=64) to keep the log short.
    static const bool debug_check = []() {
        const char * s = std::getenv("GGML_GDN_WMMA_CHECK");
        return s != nullptr && std::atoi(s) != 0;
    }();

    constexpr int stripe_w = 32;
    GGML_ASSERT(S_v % stripe_w == 0);
    dim3 grid_dims(H, n_seqs, S_v / stripe_w);
    dim3 block_dims(32, 4, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    // DebugCheck is a TEMPLATE parameter, not a runtime flag passed into an otherwise-identical
    // kernel body: the debug blocks compare against register-sized scratch (S_before) that
    // must not exist on the default fast path. An earlier version threaded this through as a
    // plain bool and it inflated VGPR usage (and spilling) even with GGML_GDN_WMMA_CHECK
    // unset, because the "dead" branch still had to be compiled and register-allocated.
#define GGML_CUDA_GDN_LAUNCH_WMMA(SV, DBG)                                                   \
    ggml_cuda_kernel_launch(gated_delta_net_prefill_wmma_cuda<(SV), GGML_CUDA_GDN_PREFILL_CHUNK, (DBG)>, \
        launch_params, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,                      \
        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                      \
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, dst_seq_stride)

    if (debug_check) {
        switch (S_v) {
            case  64: GGML_CUDA_GDN_LAUNCH_WMMA( 64, true); break;
            case 128: GGML_CUDA_GDN_LAUNCH_WMMA(128, true); break;
            default:
                GGML_ABORT("fatal error"); // S_v in {16,32}: caller must use the scalar kernel
                break;
        }
    } else {
        switch (S_v) {
            case  64: GGML_CUDA_GDN_LAUNCH_WMMA( 64, false); break;
            case 128: GGML_CUDA_GDN_LAUNCH_WMMA(128, false); break;
            default:
                GGML_ABORT("fatal error"); // S_v in {16,32}: caller must use the scalar kernel
                break;
        }
    }

#undef GGML_CUDA_GDN_LAUNCH_WMMA
}


// GGML_GDN_PREFILL_WMMA=1 opts into the WMMA chunked-prefill kernel. Default OFF
// (2026-09-18): as written it FAILS test-backend-ops (NMSE ~1.5 on every n_tokens>16 case)
// and measures 21.7 ms at (16 heads, S_v=128, 2048 tokens, v_repeat 3) vs 4.2 ms for the
// autoregressive kernel on the R9700. Kept for the rewrite; the autoregressive kernel is
// the shipping prefill path until a chunk kernel passes the suite AND beats it.
static bool ggml_cuda_gdn_prefill_wmma_enabled() {
    static const bool enabled = []() {
        const char * s = std::getenv("GGML_GDN_PREFILL_WMMA");
        return s != nullptr && std::atoi(s) != 0;
    }();
    return enabled;
}

// GGML_GDN_PREFILL_CHUNKED=1 opts into the SCALAR chunked-prefill kernel
// (gated_delta_net_prefill_cuda) as a fallback for long sequences when WMMA is unavailable
// or disabled. Default OFF: measured on an R9700 (test-backend-ops perf,
// head_count=16,head_size=128,n_seq_tokens=2048,v_repeat=3) at 671,607 us/call vs 4,197
// us/call for the plain autoregressive kernel it was meant to replace -- 160x SLOWER, not a
// safe default. It exists for A/B comparison and as a WMMA-unavailable fallback that a user
// explicitly opts into; it is not used unless asked for. The WMMA kernel
// (gated_delta_net_prefill_wmma_cuda, GGML_GDN_PREFILL_WMMA, default ON) is what actually
// ships enabled by default -- see use_prefill_chunked in ggml_cuda_op_gated_delta_net_impl.
static bool ggml_cuda_gdn_prefill_scalar_chunked_enabled() {
    static const bool enabled = []() {
        const char * s = std::getenv("GGML_GDN_PREFILL_CHUNKED");
        return s != nullptr && std::atoi(s) != 0;
    }();
    return enabled;
}

// GGML_GDN_CHUNKED=0 forces the autoregressive kernel for every block length (A/B switch).
static bool ggml_cuda_gdn_chunked_enabled() {
    static const bool enabled = []() {
        const char * s = std::getenv("GGML_GDN_CHUNKED");
        return s == nullptr || std::atoi(s) != 0;
    }();
    return enabled;
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream,
        int64_t tok_offset = 0, int64_t dst_seq_stride = -1) {
    if (dst_seq_stride < 0) {
        dst_seq_stride = n_tokens;
    }
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                tok_offset, dst_seq_stride);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                tok_offset, dst_seq_stride);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                tok_offset, dst_seq_stride);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                tok_offset, dst_seq_stride);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // Chunked (UT-transform) path: short multi-token blocks with a scalar gate. This is the
    // MTP verify-block regime, where the autoregressive kernel costs ~n serialized cross-lane
    // reductions per layer. KDA, n_tokens == 1 and long blocks keep the autoregressive kernel.
    const bool use_chunked = !kda &&
                             n_tokens >= 2 && n_tokens <= GGML_CUDA_GDN_CHUNK_MAX &&
                             ggml_cuda_gdn_chunked_enabled();

    // Long-prefill chunked path: n_tokens beyond the MTP-verify regime, scalar gate only.
    // K <= 1 excluded is handled below via `tail`; a sequence that's entirely inside the
    // K-tail window (n_tokens <= K) falls through to the plain autoregressive path instead.
    //
    // Default-enabled implementation is the WMMA kernel (RDNA4 matrix cores, S_v in
    // {64,128}); the scalar chunked kernel is opt-in only (GGML_GDN_PREFILL_CHUNKED=1) --
    // measured 160x SLOWER than the plain autoregressive kernel it was meant to replace
    // (671,607 us vs 4,197 us/call, R9700, head_count=16,head_size=128,n_seq_tokens=2048,
    // v_repeat=3; see ggml_cuda_gdn_prefill_scalar_chunked_enabled's comment), so it must
    // never be reached unless a user explicitly opts in for A/B testing or as a
    // WMMA-unavailable fallback.
    const int  cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool wmma_ok = amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) &&
                          (S_v == 64 || S_v == 128) && ggml_cuda_gdn_prefill_wmma_enabled();
    const bool use_prefill_chunked = !kda &&
                             n_tokens > GGML_CUDA_GDN_CHUNK_MAX &&
                             (n_tokens - (keep_rs ? (int64_t) K : 0)) > 0 &&
                             (wmma_ok || ggml_cuda_gdn_prefill_scalar_chunked_enabled());

    if (use_chunked) {
        if (keep_rs) {
            launch_gated_delta_net_chunked<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net_chunked<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else if (use_prefill_chunked) {
        // Split the block: tokens [0, T0) are handled by the chunked prefill kernel; when
        // K > 1 a tail of exactly the LAST K tokens is re-run by the existing autoregressive
        // kernel, which alone knows how to emit K-snapshot slots. tail must be >= K (not
        // K-1): the autoregressive kernel's `target_slot = tail_n_tokens - 1 - t` needs t to
        // range over tail-1..0 to touch every slot 0..K-1, which requires tail == K exactly
        // (a smaller tail only reaches slots 0..tail-1 and leaves older slots un-derivable
        // from curr_state + the tail's own tokens). K <= 1 needs no tail at all: slot 0 IS
        // the plain final state, which the prefill kernel already writes in that same layout.
        const int64_t tail = keep_rs ? (int64_t) K : 0;
        const int64_t T0   = n_tokens - tail;

        ggml_cuda_pool_alloc<float> inter_state(ctx.pool());
        float * prefill_state_out = state_d; // K<=1: write straight to the real output slot
        if (tail > 0) {
            // K>1: use scratch rather than aliasing state_d. The tail launch below needs to
            // read the T0-token state as curr_state while writing snapshot slots into state_d
            // itself; slot 0 of state_d isn't the T0-token state (it's the caller's output
            // buffer, uninitialized at this point), so there is no correct in-place aliasing
            // here -- a scratch buffer avoids having to prove otherwise.
            inter_state.alloc(S_v * S_v * H * n_seqs);
            prefill_state_out = inter_state.get();
        }

        if (wmma_ok) {
            launch_gated_delta_net_prefill_wmma(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, prefill_state_out,
                S_v, H, T0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, n_tokens, stream);
        } else {
            launch_gated_delta_net_prefill(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, prefill_state_out,
                S_v, H, T0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, n_tokens, stream);
        }

        if (tail > 0) {
            // prefill_state_out is laid out exactly like this op's own `state` input
            // ([S_v, S_v, H, n_seqs]): it's written through the same state_out_offset formula
            // as the plain (!keep_rs_t) autoregressive path, which is what curr_state expects.
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, prefill_state_out, dst_d, state_d,
                S_v, H, tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream,
                /*tok_offset=*/T0, /*dst_seq_stride=*/n_tokens);
        }
    } else if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
