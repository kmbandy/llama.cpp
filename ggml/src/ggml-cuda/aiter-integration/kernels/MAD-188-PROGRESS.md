# MAD-188 pre-validation progress — 2026-05-17

**Status:** Architectural validation complete. AITER's
`kernel_unified_attention_2d` AOT-compiles cleanly for gfx1201 from a
near-verbatim vendor of `aiter/ops/triton/_triton_kernels/attention/
unified_attention.py`. Real integration work is still scoped to MAD-188
proper; this note captures what was verified tonight so we don't lose it.

## What was verified

End-to-end Triton AOT compile of AITER's production 2D attention kernel:

```bash
python -m triton.tools.compile \
    /tmp/aiter-vendoring/unified_attention.py \
    --kernel-name kernel_unified_attention_2d \
    --target hip:gfx1201:32 \
    --signature "*fp16:16, *fp16:16, *fp16:16, *fp16:16, *fp32, *i32, *i32, *fp32, *fp32, 0.0883883, *fp32, *fp32, *fp32, *fp32, fp32, 16, 8, i64, i64, 128, i64, 128, i64, 16, 32, 128, 128, 0, 0, 0, 0, 0, i64, i64, i64, 1, i64, i64, i64, 1, *i32, 2, i32, 16, -448.0, 448.0, 1" \
    --grid "num_kv_heads, num_seqs / 2 + num_seqs, 1" \
    --num-warps 4 \
    --num-stages 1 \
    --out-name uattn_2d \
    --out-path /tmp/aiter-vendoring/uattn_2d
```

Produces:
- `uattn_2d.7902ab7d_0d1d2d3d.c` — 128 KB, embedded HSACO + ~26 runtime arg
  launcher
- `uattn_2d.7902ab7d_0d1d2d3d.h` — declaration of the launcher

Generated launcher signature:
```c
hipError_t uattn_2d_7902ab7d_0d1d2d3d(
    hipStream_t stream,
    hipDeviceptr_t output_ptr, hipDeviceptr_t query_ptr,
    hipDeviceptr_t key_cache_ptr, hipDeviceptr_t value_cache_ptr,
    hipDeviceptr_t sink_ptr, hipDeviceptr_t block_tables_ptr,
    hipDeviceptr_t seq_lens_ptr, hipDeviceptr_t alibi_slopes_ptr,
    hipDeviceptr_t qq_bias_ptr,
    hipDeviceptr_t q_descale_ptr, hipDeviceptr_t k_descale_ptr,
    hipDeviceptr_t v_descale_ptr, hipDeviceptr_t out_scale_ptr,
    double softcap,
    int64_t block_table_stride,
    int64_t query_stride_0, int64_t output_stride_0,
    int64_t qq_bias_stride_0,
    int64_t stride_k_cache_0, int64_t stride_k_cache_1, int64_t stride_k_cache_2,
    int64_t stride_v_cache_0, int64_t stride_v_cache_1, int64_t stride_v_cache_2,
    hipDeviceptr_t query_start_len_ptr, int32_t num_seqs);
```

The kernel's other ~21 args are folded as `tl.constexpr` at AOT compile time:
scale, num_query_heads=16, num_queries_per_kv=8 (GQA ratio for Qwen3.6),
BLOCK_SIZE=16, TILE_SIZE=32 (gfx1201-specific tuning from AITER!),
HEAD_SIZE=128, USE_ALIBI_SLOPES=0, USE_QQ_BIAS=0, USE_SOFTCAP=0, USE_SINKS=0,
SLIDING_WINDOW=0, BLOCK_M=16, BLOCK_Q=2, FP8_MIN=-448.0, FP8_MAX=448.0,
ALL_DECODE=1.

## Patches required to vendor the kernel

One line. Original `unified_attention.py` line 6:
```python
from aiter.ops.triton.utils.types import e4m3_dtype
```
That import resolves to `torch.float8_e4m3fn` for non-MI300 targets. Replace
with:
```python
# Vendored patch: AITER's e4m3_dtype resolves to torch.float8_e4m3fn on
# gfx1201/gfx1100/Hopper+ and to torch.float8_e4m3fnuz on gfx942/MI300.
# We hardcode E4M3FN here; for an MI300 build, branch on target arch.
e4m3_dtype = torch.float8_e4m3fn
```

Everything else compiles as-is.

## Open issues blocking actual usage

Both fixable, neither architectural:

1. **Grid expression must reference kernel args only.** vLLM/AITER's
   dispatcher computes `total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs`
   host-side; the AOT launcher can't see `q.shape[0]`. Workarounds:
   - **Pure decode** (q_len=1 per seq) → `total_q_blocks = num_seqs / BLOCK_Q + num_seqs`,
     which IS expressible. Above invocation uses this form.
   - **General case** → patch the kernel to accept `total_q_blocks` as an
     explicit `tl.int32` arg, compute host-side, pass in.

2. **Triton AOT signature parser doesn't accept Python `True`/`False`.**
   Use `0` and `1` for boolean constexprs. Filed-able upstream.

## Path to actual MAD-188 deliverable

This validates the AOT pipeline against AITER's production kernel. The full
integration story (MAD-188) still requires:

1. Vendor `unified_attention.py` into `aiter-integration/kernels/` with
   proper license header preservation and the e4m3 patch.
2. Vendor `kernel_unified_attention_3d` + `reduce_segments` and verify each
   AOT-compiles (same pattern, larger spec matrix).
3. Decide grid-arg approach (patch kernel for `total_q_blocks` vs special-
   case decode formula).
4. Write the C++ wrapper that translates from ggml-cuda's tensor types into
   the launcher's `hipDeviceptr_t` + stride arguments.
5. Migrate our KV cache to the vLLM/AITER unified layout
   (`[num_blocks, block_size, n_kv_heads, head_size]`).
6. Dispatch gate in `mt_pagedattn.cu` to route F16/BF16 KV through the new
   path while preserving the scalar fallback.

None of these are blocked by today's findings. Toolchain is proven.

## Files

- `/tmp/aiter-vendoring/unified_attention.py` — vendored + patched source
- `/tmp/aiter-vendoring/uattn_2d.7902ab7d_0d1d2d3d.{c,h}` — AOT artifacts
  (gfx1201)

Not committed yet — the actual MAD-188 vendoring should bring the full
AITER source into `aiter-integration/kernels/` with proper attribution.
