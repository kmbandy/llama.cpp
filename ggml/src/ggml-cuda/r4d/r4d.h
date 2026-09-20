// R4D -- the RDNA4 (gfx1201) kernel library: attention, gated delta net, all-reduce and a skinny
// bf16 GEMM, compiled into one shared object. Plain C ABI: every entry point takes raw device
// pointers and a stream, so the driver can be a pybind module or ctypes with no torch/C++ ABI
// coupling. Nothing on the launch paths allocates or synchronises, so any of them can be recorded
// into a HIP graph.
//
// NAMING. An entry point is
//
//     r4d_<family>_<op>_<geometry it is compiled for>
//
// and the geometry suffix is not decoration: these are specialised kernels, and every dimension in
// the name is a compile-time constant that the entry point REJECTS a mismatch on rather than
// running. `attn_decode_h256_gqa6_fp8kv` runs for head_dim 256 with 6 queries per KV head and an
// fp8-e4m3 paged cache, and for nothing else. A model with a different head size needs a new
// instantiation, which will sit beside this one under its own name; nothing has to be renamed to
// make room for it. Dimensions that are fixed for the whole library, and so discriminate nothing
// between entry points, stay out of the names and are reported by r4d_*_dims() and the kernel
// registry instead: the paged block size (16), the query dtype (bf16 everywhere -- see
// r4d_attn_paged_h256_gqa6.hip) and the target architecture.
//
// SOURCE FILES CARRY THE SAME NAME. A translation unit is named for the entry point it provides,
// geometry included -- r4d_gdn_kkt_solve_k128_c64_bf16.hip holds r4d_gdn_kkt_solve_k128_c64_bf16
// and nothing else. Where one unit provides a family that differs only in a suffix, the file name
// stops at the shared part: r4d_gdn_conv_w4_h128_bf16.hip provides the prep and update pair, and
// r4d_attn_paged_h256_gqa6.hip provides all four paged variants by including the two kernel
// templates that sit beside it under their own names. Only the shared machinery -- r4d_common.h,
// r4d_dt16.h, r4d_gdn_wmma.h -- is named for what it is rather than for a kernel, because it
// provides none.
//
// Where a shape is unsupported the attention and GDN entry points return a negative code; the
// all-reduce and GEMM throw std::runtime_error (they are only reachable from the pybind surface,
// which turns that into a Python exception at the call site).
#pragma once
#include <hip/hip_runtime.h>

// Library version. The module exposes it as r4d.__version__, and the git tag it was built from is
// expected to match -- which is what lets a consumer assert it linked the sources it pinned rather
// than whatever a stale clone happened to hold.
#define R4D_VERSION "0.4.0"

struct R4DArgs {
    const void*  q;             // (num_seqs*q_len, q_heads, head_dim)  bf16
    const void*  kv;            // (num_blocks, kv_heads, block_size, 2*head_dim)  fp8 e4m3 or bf16
                                //   strides below are in ELEMENTS of that dtype; K then V per slot
    const int*   block_table;   // (num_seqs, max_blocks)            int32
    const int*   seqused_k;     // (num_seqs,)                       int32
    void*        out;           // (total_q, q_heads, head_dim)      bf16
    const float* k_descale;     // (num_seqs, kv_heads)
    const float* v_descale;     // (num_seqs, kv_heads)
    const float* q_descale;     // unused: the query is bf16
    const void*  v_cache;       // KVP=2 (turbo4) only: V records; `kv` then holds the K records
    const unsigned char* k_lut; // KVP=2 only: 16 e4m3 magnitude centroids for K
    const unsigned char* v_lut; // KVP=2 only: same for V
    long kv_slot_stride;        // KVP=2 only: BYTES between consecutive slots inside a block (= kv_heads*162); kv_block_stride and kv_head_stride are then also in BYTES
    void*        scratch;       // split-KV partials (decode only), or null
    int num_seqs, q_len, q_heads, kv_heads, head_dim, block_size, max_blocks;
    long kv_block_stride;       // elements between consecutive blocks
    long kv_head_stride;        // elements between kv heads inside a block
    float scale;
    int  splits;                // decode only; 0 = let the split law choose
    int  max_ctx;               // host-visible context bound (seqused_k is device-side)
};

extern "C" {
// ---- attention: paged, causal, varlen ------------------------------------------------------
// Compiled for head_dim 256, 6 queries per KV head, paged block size 16, bf16 query. The prefill
// kernel tiles the query; the decode kernel splits the KV and takes at most 64 query rows
// (q_len * gqa), the band a speculative-decode verify step falls in.
// KVP=2 (turbo4_fp8_bs256): K and V live in SEPARATE caches of 162-byte records per (block, slot,
// kv head) -- fp16 per-vector scale, 128 bytes of 4-bit centroid indices, 32 sign bytes -- with a
// 16-entry e4m3 magnitude LUT per (layer, K/V) passed via k_lut/v_lut. Step A: prefill and decode.
// Return 0 on success, negative on a shape this instantiation does not serve.
int  r4d_attn_prefill_h256_gqa6_fp8kv (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_prefill_h256_gqa6_turbo4kv(const R4DArgs* a, hipStream_t stream);
int  r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_fp8kv  (const R4DArgs* a, hipStream_t stream);
int  r4d_attn_decode_h256_gqa6_bf16kv (const R4DArgs* a, hipStream_t stream);
// KVP=2 (turbo4_fp8_bs256) decode: K read straight from the paged cache (no sK LDS staging).
int  r4d_attn_decode_h256_gqa6_turbo4kv(const R4DArgs* a, hipStream_t stream);
// Bytes of split-KV partial buffer one decode launch of this shape needs. Independent of the cache
// dtype: the partials are f16 either way.
long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a);
// The geometry the attention kernels above are compiled for, so a caller can test a model against
// it instead of discovering the mismatch at the first launch.
void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows);

// ---- attention: vision encoder -------------------------------------------------------------
// Dense, non-causal, multi-head attention at the vision tower's head_dim of 72, bf16 throughout.
// Nothing is paged and nothing is quantised: q, k, v and o are [total_tokens, heads, 72] and
// contiguous, and cu_seqlens [num_seqs + 1] bounds one image (or one attention window) per entry,
// so a whole batch is ONE launch rather than a per-segment loop and a concatenate. `max_seqlen` is
// the host-side bound on those lengths and selects the query-block height; the kernel reads the
// device-side cu_seqlens itself. head_dim 72 is carried natively (5 k-tiles of 16, the last half
// structurally zero) rather than padded to 128. Returns -1 for any other head size.
int  r4d_attn_vit_h72_bf16(const void* q, const void* k, const void* v, void* o,
                           const void* cu_seqlens, int num_seqs, int max_seqlen,
                           int heads, int head_dim, float scale, void* stream);
// head_dim, the two query-block heights, and the segment length at which the launcher switches.
void r4d_attn_vit_dims(int* head_dim, int* rows_large, int* rows_small, int* split);

// ---- gated delta net -----------------------------------------------------------------------
// One chunked scan over N variable-length sequences: the WY recompute, the recurrent state scan and
// the output in one kernel (FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o).
// Layouts, all contiguous and bf16 unless stated:
//   q,k [T, Hg, K]   v,o [T, H, V]   A [T, H, bt]   g,beta [T, H] fp32
//   h0,ht [N, H, V, K] fp32          cu [N+1] int32
// K, V and bt are the compile-time 128/128/64 in the name; a mismatch returns -1 rather than
// running. Named for the algorithm, not the serving phase: this is the chunked scan, which is what
// prefill and chunked prefill both run. Decode uses the recurrent update, a different kernel.
int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, void* stream);
void r4d_gdn_dims(int* head_k, int* head_v, int* chunk);

// The chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, which is
// FLA's chunk_scaled_dot_kkt_fwd followed by solve_tril with the fp32 gram in between never
// reaching HBM. k [T,Hg,K] bf16; beta, g [T,H] fp32 (g already summed along the chunk);
// A [T,H,64] bf16 out; cu [N+1] int32. Returns -1 for a geometry it was not compiled for.
int r4d_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g, void* A,
                                    const void* cu, int N, int T, int H, int Hg, int K, int bt,
                                    void* stream);

// Everything between the qkv projection and the chunked scan, in one kernel: the depthwise causal
// convolution (width 4, silu) with its state cache, the q/k/v split, the l2 norm on q and k, the
// gate g = -exp(A_log).softplus(a + dt_bias) with its per-chunk cumsum, and beta = sigmoid(b).
// Replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum; the conv output never
// reaches HBM. Strides are in ELEMENTS; x may be a padded view (the qkvz split the layer hands it).
int r4d_gdn_conv_prep_w4_h128_bf16(
        const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
        long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
        const void* has_init, const void* a, const void* b, long ab_stride, int ab_is_bf16,
        const void* A_log, const void* dt_bias, void* q, void* k, void* v, void* g, void* beta,
        const void* cu, int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr,
        void* stream);

// The same convolution for a decode step: the tokens are a speculative window, the state cache is
// a rolling buffer of width-1 + num_spec entries read at the slot the last ACCEPTED token left,
// and q / k / v are written straight into their own layouts. Replaces causal_conv1d_update and
// the cat that made its output contiguous.
int r4d_gdn_conv_update_w4_h128_bf16(
        const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
        long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
        long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
        int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream);

// The gated RMS norm the layer applies to its own output: out = rms(x) . w . act(z), one row per
// (token, head). Replaces FLA's rmsnorm_fn. act: 0 = silu/swish, 1 = sigmoid. Only the prefill
// path needs it -- the decode kernel below folds the same arithmetic into its epilogue, because
// its workgroup owns the whole row.
int r4d_gdn_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w, void* o,
                                    long rows, long xrow, long zrow, long orow, int width,
                                    float eps, int act, void* stream);

// The recurrent delta-rule update decode runs where prefill runs the chunked scan: gating, the qk
// l2 norm, the state update and the output, against the paged state cache. The state is fp32
// (this model's config asks for it) and one is written per candidate token, which is the whole
// cost of the kernel. Replaces fused_sigmoid_gating_delta_rule_update.
int r4d_gdn_fused_update_w4k128v128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, const void* cu, int N, int H, int Hg,
    int K, int V, int width, int max_query_len, void* q, void* k, void* v,
    const void* a, const void* b, long ab_stride, int ab_is_bf16,
    const void* A_log, const void* dt_bias, void* state, long st_slot, long st_head,
    void* o, const void* sidx, long sidx_stride, float scale, float softplus_thr,
    void* barrier_cnt, int o_rows, void* stream);
int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
        const void* q, const void* k, const void* v, const void* a, const void* b,
        long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
        long state_slot_stride, long state_head_stride, void* o, const void* cu,
        const void* ssm_state_indices, long indices_stride, const void* num_accepted,
        const void* z_gate, const void* norm_weight, float norm_eps, int norm_act,
        int N, int H, int Hg, int K, int V, float scale, float softplus_thr, void* stream);

// ---- all-reduce: one-shot, push, 2 ranks over P2P ------------------------------------------
// One-shot means each rank pushes its whole input into the peer's IPC scratch and then reduces
// locally -- no ring, no two-shot reduce-scatter, so the scratch is sized by the full message.
// Exactly 2 ranks: the handshake is a single peer flag, not a tree.
enum { R4D_AR_HANDLE_BYTES = 64 };
// IPC scratch helpers. Startup-only, and not kernels: alloc returns the device pointer and writes
// the IPC handle (up to HANDLE_BYTES) into out_handle.
long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len);
long r4d_ar_ipc_open(const char* handle, int len);
void r4d_ar_ipc_free(long p);
void r4d_ar_ipc_memzero(long p, long size);
void r4d_ar_ipc_enable_peer(long peer);
// Exact sum, fp32 accumulate, bf16 / fp16 / fp32 payload (dtype 0 / 1 / 2).
void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                                long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                                long dtype, long stream, long nblocks, long nthreads, long drain,
                                long acq);

// radiance extras: exact all-reduce with the decoder layer's fused post-AR epilogue
// (residual add + Gemma rms_norm + per-token e4m3 quant), one block per row. bf16 only.
void r4d_ar_oneshot_2rank_exact_nq(long peer_scratch, long my_scratch, long peer_flags,
                                   long my_flags, long seq_ctrs, long slot_stride16,
                                   long inp, long residual, long norm_w, long q_out,
                                   long scale_out, long res_out, long m_rows, long k_cols,
                                   double eps, long stream_i, long drain, long acq);
// Same topology, but the wire payload is Walsh-Hadamard rotated and quantised to 6 bits per element
// over groups of 64 (plus a bf16 scale per group). Lossy, and bf16 / fp16 payload only. Takes this
// rank's own packed copy (loc_pack) as well, so the reduce folds exactly the bytes it sent.
void r4d_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq);
int  r4d_ar_max_blocks(void);                                   // both kernels

// radiance extras: the same one-shot push all-reduce for EXACTLY THREE ranks (TP=3). Scratch per
// rank is 2 receive regions x 2 slots x (slot_stride16 * 16) bytes and flags are 2 x max_blocks
// uints: region 0 holds the lower-ranked peer's message, region 1 the higher. peer_lo / peer_hi
// are the peers below / above `rank`. fp32 accumulate in canonical rank order ((x0 + x1) + x2),
// so the three ranks hold bit-identical results. Exact only (no compressed 3-rank payload).
void r4d_ar_oneshot_3rank_exact(long my_scratch, long peer_lo_scratch, long peer_hi_scratch,
                                long my_flags, long peer_lo_flags, long peer_hi_flags,
                                long seq_ctrs, long slot_stride16, long rank,
                                long inp, long out, long n_elem, long dtype, long stream,
                                long nblocks, long nthreads, long drain, long acq);
int  r4d_ar_3rank_max_blocks(void);                             // flags per region
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems); // rotated-6-bit payload only

// ---- GEMM ----------------------------------------------------------------------------------
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 throughout: a torch Linear with the weight stored (N,K), which
// is the "nt" in the name. Specialised for skinny M -- M <= 16, the band a small projection such
// as an MoE router gate produces -- where the weight read dominates and rocBLAS does poorly.
// WV columns per block x SK k-splits, reduced in LDS.
void r4d_gemm_bf16_nt_m16(long a, long w, long c, int M, int K, int N, int WV, int SK,
                          long stream);
int  r4d_gemm_bf16_nt_m16_max_m(void);

// ---- registry ------------------------------------------------------------------------------
// Every kernel in the library, with the constraints its name encodes spelled out. A caller that
// wants to know whether R4D covers a model can read this instead of hardcoding what it remembers.
//
// Each row carries the geometry twice: `shape` for a human, and `constraints` for a caller that
// wants to TEST a model against it. The second form is why the table exists -- without it every
// integration ends up restating the constraints in its own gate, and the copies drift.
//
// A constraint is one predicate on one named parameter. It is a NECESSARY condition, not a
// sufficient one: it settles whether a kernel exists for a geometry, never whether a particular
// call is safe. Strides, contiguity, alignment and buffer sizes are per-call and stay with the
// entry point, which still rejects anything it cannot run.
enum {
    R4D_C_EQ  = 0,  // param == ival
    R4D_C_LE  = 1,  // param <= ival
    R4D_C_GE  = 2,  // param >= ival
    R4D_C_DIV = 3,  // param % ival == 0
    R4D_C_IN  = 4,  // param is one of the space-separated words in sval
};
struct R4DConstraint {
    const char* key;   // parameter name, e.g. "head_dim"
    int         op;    // one of R4D_C_*
    long long   ival;  // numeric operand (EQ / LE / GE / DIV)
    const char* sval;  // string-set operand (IN)
};
struct R4DKernelInfo {
    const char* name;      // entry point, without the r4d_ prefix
    const char* family;    // attn | gdn | ar | gemm
    const char* op;        // the operation it implements -- the key callers select on, and the
                           // one thing several kernels may share (fp8 and bf16 KV, exact and
                           // lossy all-reduce)
    const char* computes;  // what it computes, for a human
    const char* shape;     // the geometry it is compiled for, for a human
    const char* dtypes;    // operand dtypes
    const struct R4DConstraint* constraints;  // the same geometry, testable
    int n_constraints;
};
int r4d_kernel_count(void);
const struct R4DKernelInfo* r4d_kernel_at(int i);
}
