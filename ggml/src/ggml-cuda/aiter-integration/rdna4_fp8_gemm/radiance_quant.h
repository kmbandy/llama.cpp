// radiance_quant.h — fused ml8-4 activation epilogue: [optional residual add
// or split-gate/up silu] -> [optional RMSNorm*w] -> Kronecker/block-Hadamard
// rotation -> per-row e4m3 quant, storing straight into radiance's
// fragment-TILED A layout that rdna4_gemm_ml84_radiance (gemm_ml84_radiance.h)
// consumes. This is OUR OWN kernel (ml8.cu's ml8_fp8_qrot_v3_kernel copied
// verbatim as the numerics oracle; the fused/tiled variants below are a
// bandwidth-oriented restructuring of it, not a verbatim port) -- see
// radiance_quant.hip's header comment for the full rationale and the
// occupancy/register-pressure argument behind the restructuring.
//
// HISTORY: this header used to also declare rdna4_radiance_add_rms_quant /
// rdna4_radiance_silu_mul_quant / rdna4_radiance_gdn_norm_quant, direct ports
// of radiance's OWN bf16 rmsnorm/silu/gdn epilogues (no rotation). Those are
// wrong for this repo's ml8-4 activation path (MXFP4, radiance's actual
// target, has no rotation step; ml8-4 always does) and are removed --
// nothing calls them, they are not exercised by any bench, and keeping dead,
// numerically-inapplicable code around here just invites someone wiring them
// into the wrong path. The rot_kind/a_dim/b_dim family below is the only
// thing this file offers now.
//
// TILED LAYOUT (A_tiled / a_scale): identical to gemm_ml84_radiance.h's AT
// operand for rdna4_gemm_ml84_radiance (and, before that, to
// radiance_mxfp4_fp8_gemm_atiled's own AT read). Byte b of 8-element group
// g (g = k >> 3, k in [0,K)) of real row r lands at:
//   mt = r >> 4, lrow = r & 15, kt = g >> 1, half = g & 1, KT = K >> 4
//   byte_offset = ((mt*KT + kt) * 256) + lrow*8 + half*128 + b
// i.e. AT is sized ceil(M/16)*16*K bytes, rows padded up to a multiple of
// 16; pad rows are zero-filled (every launcher below hipMemsetAsync's the
// buffer before launch) and are never read back by rdna4_gemm_ml84_radiance's
// own M-tile clamp. a_scale is indexed by the REAL (untiled) row only:
// a_scale[M], not a_scale[ceil(M/16)*16].
//
// rot_kind: matches ggml-ml8.h's GGML_FP8_QUANT_ROT_KIND_* (duplicated here
// as plain ints so this header does not have to pull in ggml.h):
//   1 = KRONECKER      — h_a required, fp32 [a_dim, a_dim], K == a_dim*b_dim
//   2 = BLOCK_HADAMARD — h_a must be null, K == a_dim*b_dim
// (0 = NONE is not handled by ml8_fp8_qrot_v3_kernel in ml8.cu either --
// that shape falls back to a different, non-rotating quant kernel there --
// so every entry point below declines rot_kind==0.)
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RDNA4_ML8_QROT_KIND_KRONECKER      1
#define RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD 2

// y = (norm_w ? rmsnorm(x,eps)*norm_w : x); A_tiled/a_scale = per-row e4m3
// quant of rotate(y) via h_a (Kronecker) or the block Hadamard transform
// (rot_kind), in the TILED layout documented above. norm_w may be null (no
// norm, matching ml8_fp8_qrot_v3_kernel's own norm_w==nullptr path).
// x: fp32 [M,K]. norm_w: fp32[K] or null. h_a: fp32[a_dim,a_dim] (Kronecker)
// or null (block Hadamard). K == a_dim*b_dim.
//
// y_out (optional, nullable): if non-null, the fp32 [M,K] value of `y`
// above (post add/norm, BEFORE rotation) is also written there -- the value
// the graph's separate RMS_NORM*MUL node would have produced, for a second
// consumer of the normalized-but-unrotated activation.
//
// Declines (returns false, nothing launched) if M<=0, K<=0, K%16!=0 (TILED
// requirement), rot_kind not in {KRONECKER, BLOCK_HADAMARD}, a_dim<=0,
// b_dim not a power of two >= 32, or (a_dim, b_dim) outside the set of
// instantiations radiance_quant.hip compiles in (see its dispatch table,
// which mirrors ml8_launch_qrot_v3's own shape coverage in ml8.cu).
bool rdna4_ml8_qrot_tiled(const float* x, const void* h_a, int rot_kind, int a_dim, int b_dim,
                           const float* norm_w, float norm_eps, int M, int K, uint8_t* A_tiled,
                           float* a_scale, hipStream_t stream, float* y_out = nullptr);

// qrot_tiled with a fused residual add at the load: r = x + residual
// (residual null => r = x); r optionally written to residual_out in fp32
// (captured BEFORE norm/rotation, i.e. the raw sum, for the next residual
// add in the chain); then the same norm(optional)+rotate+TILED-quant as
// rdna4_ml8_qrot_tiled. y_out (optional, nullable): same meaning as above --
// the post-add, post-norm, pre-rotation row.
bool rdna4_ml8_qrot_add_tiled(const float* x, const float* residual, float* residual_out,
                               const void* h_a, int rot_kind, int a_dim, int b_dim,
                               const float* norm_w, float norm_eps, int M, int K,
                               uint8_t* A_tiled, float* a_scale, hipStream_t stream,
                               float* y_out = nullptr);

// qrot_tiled with silu(gate)*up computed at the load instead of a direct row
// read, for ggml's FUSED gate_up GLU layout (gate_up: fp32 [M, 2*N],
// concatenated along the last dim). gate_first=1 means columns [0,N) are
// gate, [N,2N) are up (matches ggml_swiglu_split()/swapped=false);
// gate_first=0 swaps them. NO norm_w (ggml never renormalizes a GLU output
// before this quant). N plays the other entry points' K.
bool rdna4_ml8_qrot_silu_mul_tiled(const float* gate_up, int gate_first, const void* h_a,
                                    int rot_kind, int a_dim, int b_dim, int M, int N,
                                    uint8_t* A_tiled, float* a_scale, hipStream_t stream);

// Same as rdna4_ml8_qrot_silu_mul_tiled, but for ggml_glu_split's SEPARATE
// gate/up tensors (two independent [M,N] fp32 buffers) instead of one fused
// [M,2N] tensor -- this is the actual GLU shape ml8-4's swiglu uses in this
// repo (ggml_glu_split, glu->src[1] != nullptr), which the fused-gate_up
// entry point above cannot represent. Output: silu(gate)*up, rotated and
// per-row e4m3 quantized into the TILED layout, exactly like the fused form.
bool rdna4_ml8_qrot_silu_mul_split_tiled(const float* gate, const float* up, const void* h_a,
                                          int rot_kind, int a_dim, int b_dim, int M, int N,
                                          uint8_t* A_tiled, float* a_scale, hipStream_t stream);

// bf16-INPUT variants (2026-09-19): LLAMA_ACT_BF16 prefill's ffn_up/ffn_gate
// GEMM outputs (src/models/qwen35.cpp build_layer_ffn) and the residual
// stream are bf16, not fp32 -- reading them as fp32 (the entry points
// above) costs 2x the bytes these kernels are bandwidth-bound on
// (production rocprof: rdna4_ml8_qrot_silu_mul_split_tiled at ~475 GB/s
// reading fp32 gate+up, 570 MB/call at M=4096,N=17408). Same math as the
// fp32 entry points -- every bf16 value is upcast to fp32 immediately
// after the load (exact, no rounding: bf16 bits sit in the upper 16 of an
// fp32) -- so silu*mul/add -> rotate -> amax -> e4m3 pack is byte-for-byte
// identical to the fp32 variants; only the load's byte count changes.
// gate_bf16/up_bf16/x_bf16/residual_bf16 are raw bf16 bit patterns (`void*`
// so callers can pass __hip_bfloat16*, nv_bfloat16*, or uint16_t* --
// bit-identical for this purpose). Same shape dispatch / decline rule as
// their fp32 counterparts.
bool rdna4_ml8_qrot_silu_mul_split_tiled_bf16(const void* gate_bf16, const void* up_bf16,
                                               const void* h_a, int rot_kind, int a_dim, int b_dim,
                                               int M, int N, uint8_t* A_tiled, float* a_scale,
                                               hipStream_t stream);

// bf16-INPUT rdna4_ml8_qrot_add_tiled: x and residual are bf16; residual_out
// (if non-null) is still written fp32 -- keeping the residual stream itself
// bf16 end-to-end is a later step, not this kernel's concern (the task's
// "keep it simple" call). norm_w/norm_eps/y_out share the exact same
// optional-fused-RMSNorm / post-norm-capture semantics as
// rdna4_ml8_qrot_add_tiled.
bool rdna4_ml8_qrot_add_tiled_bf16(const void* x_bf16, const void* residual_bf16,
                                    float* residual_out, const void* h_a, int rot_kind, int a_dim,
                                    int b_dim, const float* norm_w, float norm_eps, int M, int K,
                                    uint8_t* A_tiled, float* a_scale, hipStream_t stream,
                                    float* y_out = nullptr);

// qrot_tiled with a GATED PER-HEAD RMSNORM prologue, for the GDN output site
// (qwen35 build_norm_gated -> ssm_out): the row [K = n_heads*head_dim] is
// treated as n_heads groups of head_dim; each group is rms-normalized over
// head_dim (eps), multiplied by norm_w[head_dim], then by silu(z) where z is
// fp32 with the same [head_dim, n_heads, M] logical shape but STRIDED
// (z_nb1 = bytes between heads, z_nb2 = bytes between tokens; head_dim is
// contiguous). Then the same rotate + per-row TILED quant as
// rdna4_ml8_qrot_tiled over the full K. y_out (optional): the post-gate,
// pre-rotation row (what the graph's final MUL would have produced).
// Attention output gating (2026-09-20, ggml_fp8_quant_rot_gated): A = quant(rot(o * sigmoid(z)))
// with z a strided [head_dim, n_heads, M] view (z_nb1/z_nb2 in bytes; head_dim's own stride
// must be 4). No norm. Same dispatch coverage / decline rules as rdna4_ml8_qrot_gated_norm_tiled.
bool rdna4_ml8_qrot_gate_sigmoid_tiled(const float* o, const float* z, size_t z_nb1, size_t z_nb2,
                                        int head_dim, int n_heads,
                                        const void* h_a, int rot_kind, int a_dim, int b_dim, int M,
                                        uint8_t* A_tiled, float* a_scale, hipStream_t stream);
bool rdna4_ml8_qrot_gated_norm_tiled(const float* o, const float* z, size_t z_nb1, size_t z_nb2,
                                      const float* norm_w, float norm_eps, int head_dim, int n_heads,
                                      const void* h_a, int rot_kind, int a_dim, int b_dim,
                                      int M, uint8_t* A_tiled, float* a_scale, hipStream_t stream,
                                      float* y_out = nullptr);

// Same math as rdna4_ml8_qrot_gated_norm_tiled (per-head rmsnorm * w *
// silu(z) -> rotate -> quant), but `o` is read bf16 (upcast exactly, same
// as this file's other bf16 variants) through a head-index indirection
// instead of packed fp32 -- lets the GDN gated-norm fusion consume
// libr4d chunk_scan's OUTPUT directly (bf16, [T,H,head_dim], heads in
// libr4d's own GQA-blocked order), dropping the adapter's separate
// "output cast+unpermute" kernel (mt_gdn_r4d.cu's
// r4d_gdn_cast_o_unpermute_kernel).
//
//   o_bf16:    bf16 [n_rows, H_src, head_dim] where H_src is however many
//              source (libr4d-order) heads exist -- addressed via strides,
//              not assumed packed: element (row, src_head, d) is at byte
//              offset row*o_nb_tok + src_head*o_nb_head + d*2. head_dim is
//              assumed contiguous within a head (d*2, not a third stride).
//   head_src:  int32[n_heads]; for LOGICAL head h (0..n_heads-1), the
//              source head index to read from o_bf16 -- i.e.
//              src_head = head_src ? head_src[h] : h. This is exactly
//              mt_gdn_r4d.cu's r4d_gdn_perm_head(h, Hg, R) = R*(h%Hg) +
//              h/Hg (R = H/Hg), precomputed by the caller; nullptr means
//              identity (R=1 / Hg=H, no permutation needed).
//   z, norm_w, a_scale, A_tiled, y_out: all stay in LOGICAL head order,
//              unaffected by head_src -- only the `o` load is redirected.
//              y_out (if non-null) is the post-gate, pre-rotation row in
//              LOGICAL order, i.e. exactly what the graph's MUL would have
//              produced (same contract as the fp32 entry point's y_out).
//
// Declines under the exact same conditions as rdna4_ml8_qrot_gated_norm_tiled
// (head_dim/dispatch-table coverage) -- see that entry point's doc comment.
bool rdna4_ml8_qrot_gated_norm_tiled_r4d(const void* o_bf16, size_t o_nb_head, size_t o_nb_tok,
                                          const int32_t* head_src, const float* z, size_t z_nb1,
                                          size_t z_nb2, const float* norm_w, float norm_eps,
                                          int head_dim, int n_heads, const void* h_a, int rot_kind,
                                          int a_dim, int b_dim, int M, uint8_t* A_tiled,
                                          float* a_scale, hipStream_t stream,
                                          float* y_out = nullptr);

// BENCH-ONLY oracle: runs the UNMODIFIED, verbatim-copied
// ml8_fp8_qrot_v3_kernel (ml8.cu:5208-5392) over the same shape dispatch
// table as the entry points above, writing ROW-MAJOR fp8 (its native output
// format, untouched) instead of TILED. The bench retiles this with
// rdna4_gemm_ml84_radiance_retile_a (gemm_ml84_radiance.h) and diffs the
// bytes against rdna4_ml8_qrot_tiled's direct TILED output -- plain and add
// must match bit-for-bit (same math, different store point); same shape
// coverage / decline rule as rdna4_ml8_qrot_tiled.
bool rdna4_ml8_qrot_oracle_rowmajor(const float* x, const void* h_a, int rot_kind, int a_dim,
                                     int b_dim, const float* norm_w, float norm_eps, int M, int K,
                                     uint8_t* a_fp8_rowmajor, float* a_scale, hipStream_t stream);

#ifdef __cplusplus
}
#endif
