#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdlib>

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
                                     int           K) {
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
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

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
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
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
template <int S_v, bool keep_rs_t>
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
    constexpr int TM            = GGML_CUDA_GDN_CHUNK_MAX;

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

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<16, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<32, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64:
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<64, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 128:
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<128, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
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
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
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
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
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
