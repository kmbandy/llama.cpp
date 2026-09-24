# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM ROCm/aiter
#
#   Source:   aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py
#   Upstream: https://github.com/ROCm/aiter
#   Repo HEAD at vendoring time: c1d1476cf36f13b00b430e4fcd331f1979f30461 (2026-09-23)
#   Last commit touching this file: e44c1d0bc2c883108c58b16538c81f7198863e93 (2026-09-17)
#
# License: MIT (matches upstream aiter LICENSE — see
#          https://github.com/ROCm/aiter/blob/main/LICENSE).
#
# Vendored for: DS4.1 native sparse attention on gfx1201 (RX 9070 XT),
# prefill half. Only `_sparse_attn_prefill_kernel` is carried over — the
# upstream file's ragged-index-packing kernels
# (_pack_dense_prefix_to_ragged_kernel, _compute_topk_lens_kernel,
# _pack_global_topk_ragged_kernel, _compute_combined_lens_kernel,
# _combine_topk_swa_indices_ragged_kernel) are NOT vendored: this
# integration builds its per-query (kv_indices, kv_indptr) ragged pair on
# the ggml graph side, from tensors the graph already has (the existing
# kq_mask's window cells + build_indexer_top_k's output), with a fixed
# stride-N indptr (N = window + top_k, no host-side packing kernel needed) —
# see src/models/deepseek41.cpp build_attention_v41.
#
# ─────────────────────────────────────────────────────────────────────────
# LOCAL PATCHES APPLIED (deviations from upstream) — import/adapter and a
# fixed autotune config ONLY, per project policy of vendoring kernel bodies
# verbatim:
#
# 1. Dropped the `@triton.autotune(...)` decorator (and its supporting
#    `_get_prefill_autotune_configs` / `_prefill_prune_configs` /
#    `autotune_configs(...)` call, which pulled in
#    `aiter.ops.triton.utils.tuned_config_utils`). Our runtime-compile path
#    (aiter-integration's `compile_aiter_kernel.py`, called from
#    `wrappers/aiter_runtime_compiler.cpp`) calls `triton.compile()` on the
#    bare kernel object directly — it needs a `JITFunction`, not an
#    `Autotuner` (autotune wraps the kernel in a benchmark-and-select
#    harness that has no meaning for a single ahead-of-time compile with a
#    config already chosen offline). The kernel BODY is unchanged.
#
#    Config used (chosen by running the still-autotune-decorated upstream
#    kernel once against DS4.1's real pp4096 shape — T=4096, num_kv=11520,
#    H=64, D=512, N_IDX=640 — on gfx1201 and reading
#    `_sparse_attn_prefill_kernel.best_config`; see
#    scratchpad/sparse-attn/bench.py Stage 1 run):
#        BLOCK_H=32, BLOCK_K=16, waves_per_eu=0 (Triton default),
#        matrix_instr_nonkdim=16, num_warps=4, num_stages=1
#    This is passed at AOT-compile time via the C++ wrapper's `KernelSpec`
#    (mt_dsv4_sparse_attn.cpp), not baked into this file.
#
# 2. Dropped `_sparse_attn_prefill_kernel_repr` / `make_kernel_repr` import
#    (`aiter.ops.triton.utils._triton.kernel_repr`) — cosmetic, used only
#    for autotune-benchmark logging output, which no longer applies once
#    autotune is removed. `@triton.jit` (bare, no `repr=`) replaces
#    `@triton.jit(repr=_sparse_attn_prefill_kernel_repr)`.
#
# 3. Dropped the unused `torch` import and `_get_lds_limit()` /
#    `_LDS_LIMIT` (only consumed by the now-removed `_prefill_prune_configs`
#    autotune config pruner) — dead code once (1) is applied, not a
#    behavioral change to anything this file still exports.
#
# No other line of `_sparse_attn_prefill_kernel`'s body was touched.
# ─────────────────────────────────────────────────────────────────────────

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Sparse attention kernel (prefill) — upstream body, verbatim.
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,  # [num_queries, num_heads, head_dim]
    kv_ptr,  # [num_kv, head_dim]
    kv_indices_ptr,  # [nnz]
    kv_indptr_ptr,  # [num_queries + 1]
    attn_sink_ptr,  # [num_heads]
    out_ptr,  # [num_queries, num_heads, head_dim]
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 64-bit before the multiply, same reasoning as `slot_off` below: the
    # program id fits 32 bits, but `query_idx * q_stride_t` does not once
    # num_queries passes 32K, because q_stride_t is num_heads * head_dim
    # (128 * 512 = 65536) in the V4 layout. Unlike the pool read, the wrapped
    # offset lands outside the q/out allocations, so it page-faults.
    query_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    # Prefetch first tile's slot indices so the indirect int32 load can overlap
    # the next iteration's QK MFMA latency.
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)

        # 64-bit before the multiply, same as the decode kernel: a slot index
        # fits 32 bits (the index buffer is int32 by ABI) but `slot *
        # kv_stride_n` does not once the unified V4 pool runs to ~150M rows.
        # The wrapped offset still lands inside the same allocation, so the
        # bad read is silent.
        slot_off = slot.to(tl.int64)

        kv = tl.load(
            kv_ptr
            + slot_off[:, None] * kv_stride_n
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        # Prefetch next tile's indices before heavy compute on current tile.
        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos,
            mask=next_k_pos < kv_len,
            other=-1,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(
            m_new[:, None] == float("-inf"), 0.0, tl.exp(scores - m_new[:, None])
        )
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=float("-inf")
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.where(m_final == float("-inf"), 0.0, tl.exp(m_i - m_final))
        exp_sink = tl.where(sink == float("-inf"), 0.0, tl.exp(sink - m_final))
        l_final = l_i * alpha + exp_sink
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )
