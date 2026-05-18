# AITER + vLLM TurboQuant Integration Architecture

**Status:** Design captured 2026-05-17. Implementation not started.
**Parent Jira epic:** [MAD-186](https://mad-lab-ai.atlassian.net/browse/MAD-186) — [mad-lab-cpp] AMD-focused llama.cpp fork — strategic productization
**Author:** Kurt + Claude (design session)

---

## 1. Executive Summary

This document describes the architecture for integrating two upstream Triton-based
attention/quantization backends into our llama.cpp fork:

- **ROCm/aiter `unified_attention`** — vLLM's paged-attention kernel adapted by AMD
  with explicit gfx1201 tuning. Handles prefill + decode + sliding-window in a
  single kernel for F16/BF16/FP8 KV cache types.

- **vllm-project/vllm TurboQuant backend** — PolarQuant + Walsh-Hadamard rotation +
  Lloyd-Max quantization for sub-FP8 KV cache. The published TurboQuant paper
  (Zandieh et al, ICLR 2026) is the same algorithm family as our existing
  `GGML_TYPE_TURBO4_0`.

Both kernel sets are **pure Triton, self-contained**, and Apache-2.0 licensed.
They have **zero dependency on `torch.Tensor` at execution** — they only need
device pointers + stride descriptors + scalar compile-time constants. We can
vendor the kernels under their original licenses, AOT-compile them to HSACO via
the Triton AOT toolchain, and load them from C++ via `hipModuleLoad`.

The integration replaces our hand-rolled `mt_paged_attention_tile_kernel` (MAD-180
family), our hand-rolled `mt_paged_attention_decode_kernel` (MAD-185), and our
scalar `mt_paged_attention_kernel` for the F16/BF16/FP8/TurboQuant KV paths.
A scalar fallback remains for unusual configurations.

**Why this matters:** Our hand-rolled flash-decode hit 15 t/s @ 82K context, ~18×
off the memory-bandwidth floor (~280 t/s at that ctx). AITER's tuned kernel +
TurboQuant's optimized store + WorkspaceManager-shared dequant buffers should put
us in the 60-150 t/s range at 400K decode — competitive with vLLM on AMD
hardware. Long-term, this is the foundation for tracking AMD's silicon-specific
tuning automatically rather than re-deriving it.

---

## 2. Strategic Context

### 2.1 Fork positioning

This fork is positioned as the **AMD-focused fork of llama.cpp**. We are explicit
that ggml/llama.cpp is the foundation; our value-add is additive AMD-specific
features that vanilla llama.cpp does not have. We do not pull an Ollama on
attribution — `CREDITS.md` lists every upstream we depend on.

Existing AMD-specific differentiators in the fork (pre-this-work):

- **Tiered KV cache** — hot/warm/cold tiers with SSD spillover (1M ctx on 32 GB
  VRAM, [MAD-179](https://mad-lab-ai.atlassian.net/browse/MAD-179))
- **Turbo4 KV quantization** — 4-bit, AMD-friendly ([MAD-116](https://mad-lab-ai.atlassian.net/browse/MAD-116))
- **Paged blocks + scatter-attention** — vLLM pattern, inside llama.cpp
  ([MAD-120/121/122](https://mad-lab-ai.atlassian.net/browse/MAD-120))
- **Multi-warp WMMA tile FA kernel** — RDNA3/4 specific
  ([MAD-180 family](https://mad-lab-ai.atlassian.net/browse/MAD-180))
- **Semantic prefetch** via bge-small/granite embedders
  ([MAD-122/125/129](https://mad-lab-ai.atlassian.net/browse/MAD-122))
- **MTP spec-decode plumbing** ([MAD-174/176](https://mad-lab-ai.atlassian.net/browse/MAD-174))
- **LockedBuffer** (`--kv-tier-warm-mlock`) — production hygiene for the warm tier

This work adds the next layer.

### 2.2 The bootstrap cascade

The work in this document is on the critical path of a larger plan. Long-context
inference on AMD hardware is the *enabler* for coding agents that can hold real
codebases in context. Those agents then take over downstream engineering work.

```
                  Manual work (claude + human design)
                  ───────────────────────────────────
   Step 1:        AITER attention + TurboQuant on R9700
                  Qwen3.6 27B at long ctx → FAST
                              │
                              ▼
                  Coding agent fleet unlocks (1 GPU)
                  ──────────────────────────────────
   Step 2:        Qwen3.6 handles → FSR4 INT8 research
                                  → AITER port to RDNA2/3
                              │
                              ▼
                  Coding agent fleet doubles (2 GPUs)
                  ───────────────────────────────────
   Step 3:        gpt-oss-20b on 6900 XT at long ctx
                              │
                              ▼
                  Hand off everything else
                  ────────────────────────
   Step 4:        Quark integration, MXFP4/MXFP8 types,
                  MoE optimization, 250M / 1B / MoE model
                  training pipeline — all parallel agent work
```

Step 1 — what this document specifies — is the only thing on the critical path
that must come out of our heads. After it lands, leverage multiplies.

### 2.3 What this work is NOT

- Not a re-skin of llama.cpp; ggml/llama.cpp foundation is preserved.
- Not a vLLM competitor at the scheduler / multi-host layer.
- Not a Python-runtime engine; the runtime remains C++ via the llama.cpp backend.
- Not abandoning AMD-only focus; upstream sync gives us NVIDIA improvements for
  free.

---

## 3. Current State of the Fork

### 3.1 Existing paged-attention dispatch

`ggml/src/ggml-cuda/mt_pagedattn.cu` currently dispatches:

```
ggml_cuda_op_paged_attn_mt:
├── KV type F16/TURBO4_0, q_len >= 16, gfx1100/gfx1201:
│   → launch_paged_attn_tile{_mw}  (MAD-180 family, multi-warp WMMA)
├── KV type F16/TURBO4_0, q_len == 1, max_ctx_len >= 8192:
│   → launch_paged_attn_decode  (MAD-185 hand-rolled flash-decode)
└── otherwise:
    → mt_paged_attention_kernel  (scalar fallback)
```

KV cache layout is split:

- K cache: `[num_blocks, n_kv_heads, head_size/x, block_size, x]` (vLLM-style, x=8 for fp16)
- V cache: `[num_blocks, n_kv_heads, head_size, block_size]` (head-dim-major)

The split layouts exist because each kernel was hand-tuned for its specific
access pattern. They are NOT aligned with vLLM/AITER's modern unified layout.

### 3.2 Measured baseline (MAD-185 hand-rolled flash-decode)

Qwen3.6-35B-A3B-UD-Q4_K_XL, turbo4 KV, gfx1201:

| ctx tokens (actual) | flash-decode | scalar (baseline) | speedup |
|---|---|---|---|
| 8K (8.5K)   | 42.0 t/s   | ~50-60 t/s    | (slower at small ctx; flash-decode has fixed launch overhead) |
| 16K (31K)   | 27.5 t/s   | ~40 t/s       | (still slower) |
| 24K (59K)   | 19.7 t/s   | 1.40 t/s      | **14×** |
| 32K (82K)   | 15.3 t/s   | ~0.4 t/s (extrapolated) | ~40× |
| 400K (proj) | ~5 t/s     | 0.19 t/s      | ~25× |

Memory-bandwidth floor at 400K decode (on R9700 ~720 GB/s, n_heads=16,
n_kv_heads=2, head_dim=256, turbo4=0.5 B/elem, 16 full-attention layers):
~60 t/s. We are ~18× off the floor at 82K. Real headroom.

### 3.3 What the integration replaces

| Path | Replaced by |
|---|---|
| Scalar `mt_paged_attention_kernel` (all KV types) | Kept as ultimate fallback |
| `launch_paged_attn_tile{_mw}` (F16/turbo4 prefill, MAD-180 family) | AITER `unified_attention` 2D path (F16/BF16); vLLM TurboQuant continuation prefill (turbo4) |
| `launch_paged_attn_decode` (MAD-185, F16/turbo4 decode) | AITER `unified_attention` 3D path (F16/BF16); vLLM TurboQuant decode kernel (turbo4) |

Our hand-rolled kernels were architecturally correct (3D split-K + reduce) but
under-tuned. The vendor-blessed versions inherit gfx1201-specific tuning
constants, GQA fanout, and feature breadth (SWA, sinks, softcap, ALiBi, FP8 KV)
we would otherwise build over months.

---

## 4. The Architectural Choice

### 4.1 Two-backend dispatch

```
ggml_cuda_op_paged_attn_mt (post-integration):
├── KV type F16/BF16/Q8_0:
│   → AITER unified_attention (vendored)
│     · Cache layout: (2, num_blocks, block_size, n_kv_heads, head_size)
│     · Single entry handles prefill + decode + SWA
│     · 2D kernel for short ctx / SWA / busy GPUs
│     · 3D split-K + reduce_segments for long-ctx decode
├── KV type FP8 (future MXFP8_E4M3 / MXFP8_E5M2):
│   → AITER unified_attention + q/k/v_descale tensors (same kernel, FP8 path)
├── KV type TURBO4_0 (and future TQ variants — k8v4, 4bit_nc):
│   → vLLM TurboQuant backend (vendored)
│     · Cache layout: (num_blocks, block_size, n_kv_heads, slot_size_aligned)
│     · Pre-attention: triton_turboquant_store (separate custom op)
│     · Decode: inline dequant in attention kernel
│     · Continuation prefill: ≤128 tokens uses decode kernel; >128 dequants + flash_attn
└── Fallback:
    → mt_paged_attention_kernel (scalar) for unusual head sizes / cache types
```

The dispatch gate is **KV cache type**, not q_len. Both AITER and TurboQuant
backends internally route between prefill-shaped and decode-shaped paths.

### 4.2 Why two backends, not one

- AITER's `unified_attention` does **not know about turbo4**. The kernel reads
  fp16/bf16/fp8 elements directly from cache; it has no per-block dequant logic
  for our 4-bit codebook format.
- vLLM TurboQuant's kernels are **format-specific** (k8v4, 4bit_nc, etc.); each
  variant has a different slot layout and inline dequant routine. Generalizing
  TurboQuant kernels to handle native fp16/bf16 would require either a new
  preset or runtime branching that defeats the kernel's compile-time
  specialization.
- Two backends with **clearly separated domains** is the same pattern vLLM uses
  internally. It maps cleanly onto our dispatch.

### 4.3 Why Triton AOT, not something else

Alternatives considered:

| Option | Pro | Con |
|---|---|---|
| Triton AOT (chosen) | Vendor source `.py`, ahead-of-time compile to HSACO, no Python runtime dep | Build dep on Triton toolchain |
| Hand-port to HIP/CUDA C++ | Pure HIP, no new deps | Years of tuning replicated; loses upstream tracking |
| Run Triton kernels via Python bridge | Use upstream as-is | Python runtime in llama.cpp; rejected on principle |
| Composable Kernel (CK) | AMD-blessed tiling library | CK is C++ template-heavy; we'd still need our own tiling decisions |

Triton AOT preserves upstream tracking (we re-pull kernel sources when they
improve) while keeping the runtime pure C++. The AOT toolchain runs only at
build time.

### 4.4 RDNA4 readiness of Triton

Verified via search of `triton-lang/triton` PRs (April-May 2026):

- PR #10109 — GFX12 cache modifiers for buffer/async ops (merged 2026-04-23)
- PR #10178 — gfx1250 fp8 fix and `is_rdna4` predicate (merged 2026-04-30)
- PR #10185 — IN_THREAD_TRANSPOSE on gfx1201 by default; **FA2 at 79.10 TFLOPS,
  GEMM at 120.89 TFLOPS measured on gfx1201** (merged 2026-05-01)
- PR #10304 — translator test skips for RDNA4 (merged 2026-05-13)

Gluon experimental layouts file declares `AMDWMMALayout.version=2` for
RDNA4 (gfx1200/gfx1201) explicitly.

**Triton 3.6.0 (current install in `~/venvs/agents312`) is too old.** The bulk of
gfx1201 enablement and tuning landed after that release. Required: Triton 3.7.0
or mainline build.

---

## 5. Backend A: AITER `unified_attention`

### 5.1 Source

`ROCm/aiter:aiter/ops/triton/attention/unified_attention.py` (dispatcher,
~290 lines) and `ROCm/aiter:aiter/ops/triton/_triton_kernels/attention/unified_attention.py`
(kernel `@jit` bodies, ~700 lines). Apache-2.0.

Adapted from vLLM upstream (`vllm/attention/ops/triton_unified_attention.py`)
with AMD-specific tuning added.

### 5.2 Public API

```python
unified_attention(
    q,                # [total_q_tokens, num_query_heads, head_size]
    k,                # [num_blocks, block_size, num_kv_heads, head_size]
    v,                # [num_blocks, block_size, num_kv_heads, head_size]
    out,              # same shape as q
    cu_seqlens_q,     # [batch+1] ragged seq starts
    max_seqlen_q,     # scalar
    seqused_k,        # [batch] kv len per seq
    max_seqlen_k,     # scalar
    softmax_scale,
    causal,           # must be True
    window_size,      # (left, right) — SLIDING_WINDOW = 1 + left
    block_table,      # [batch, max_blocks_per_seq]
    softcap,
    q_descale,        # optional FP8 dequant scale
    k_descale,        # optional FP8 dequant scale
    v_descale,        # optional FP8 dequant scale
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,       # StreamingLLM attention sinks
)
```

### 5.3 KV cache layout

```
key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
value_cache: [num_blocks, block_size, num_kv_heads, head_size]
```

**This is the standard vLLM v1 layout** (after `kv_cache.unbind(0)`). Both K and
V use the *same* shape. K and V are stored in a single contiguous tensor of
shape `(2, num_blocks, block_size, num_kv_heads, head_size)` with the leading
`2` separating them.

### 5.4 Dispatch logic (the "unified" part)

```python
ALL_DECODE = (max_seqlen_q == 1)
target_num_prgms = num_cus * 4
num_2d_prgms = total_num_q_blocks * num_kv_heads

# 2D kernel chosen when:
#   sliding_window > 0          OR
#   max_seqlen_k <= 512         OR
#   num_2d_prgms > target_num_prgms  (already enough work)
# Otherwise → 3D split-K + reduce_segments
```

The same entry point handles prefill, decode, and SWA. The runtime config
function chooses between the 2D kernel (`kernel_unified_attention_2d`) and the
3D split-K kernel (`kernel_unified_attention_3d` + `reduce_segments`).

### 5.5 gfx1201-specific tuning

```python
BLOCK_M = 16 if num_queries_per_kv <= 16 else next_pow2(num_queries_per_kv)
TILE_SIZE = 32 if arch.name == "gfx1201" else 16 if arch.is_rdna else 64
waves_per_eu = 8 if arch.name == "gfx1151" else 6 if arch.is_rdna else 2

# Pure decode on RDNA:
num_stages_2d, num_warps = 1, 4
# Large prefill (max_seqlen_q >= 256) on RDNA:
BLOCK_M = 64    # vs 128 on CDNA
num_stages_2d, num_warps = 1, 4
```

**`gfx1201` is treated specifically**, not as a generic RDNA family member.
The K-tile is doubled (32 vs 16) — tells us they tuned on actual R9700/R9600D
silicon.

### 5.6 Implementation tricks worth understanding

1. **`exp2` instead of `exp`.** Everything prescaled by `RCP_LN2 =
   1.4426950408889634`, then `tl.math.exp2` (→ `rocdl.exp2` intrinsic, one
   instruction on RDNA).
2. **`.cg` cache modifier on decode.** Q is cached-global (bypasses L1) since
   it's small + read-once. KV uses `.cg` when `ALL_DECODE`.
3. **GQA fanout via BLOCK_M.** Multiple query heads sharing a kv_head pack into
   the same BLOCK_M and load the same KV tile once. For Qwen3.6 (n_kv_heads=2,
   num_queries_per_kv=8): BLOCK_M=16 means 2 query positions × 8 heads load the
   SAME KV tile. **8× memory bandwidth efficiency.**
4. **Sliding-window TILE pruning.** Tiles outside the window are skipped at the
   tile-index level (not just masked at logit level). SWA layer with window=4K
   at ctx=400K does 4K/TILE_SIZE iterations, not 400K. This is the feature
   that would fix qwen35-hybrid SWA layers.
5. **FP8 path via descale tensors.** `qk_scale = scale * RCP_LN2 * q_descale *
   k_descale`, then `acc = acc * v_descale` at end. Whole FP8 KV cache works by
   passing 3 scalar tensors.
6. **Sinks (StreamingLLM).** Pre-initialize the running max `M` to
   `sink * RCP_LN2`. Adds a virtual attention sink to the softmax denominator.
7. **Softcap (Gemma-2).** Tanh-shaped soft logit clipping done in-kernel via
   two `exp2` calls.

### 5.7 Features inherited for free

- Sliding window attention
- Attention sinks (StreamingLLM)
- ALiBi
- Softcap (Gemma-2)
- FP8 KV via descale tensors
- QQ-bias (custom attention pattern)
- Configurable causal vs. non-causal (`causal=True` enforced today, easy to relax)

---

## 6. Backend B: vLLM TurboQuant

### 6.1 Source

```
vllm/v1/attention/ops/triton_turboquant_store.py        (~441 LoC, store kernels)
vllm/v1/attention/ops/triton_turboquant_decode.py       (~617 LoC, decode + dequant kernels)
vllm/model_executor/layers/quantization/turboquant/    (Lloyd-Max solver,
    centroids.py, config.py                             centroid tables, config)
```

All Apache-2.0. Source PRs:

- [#38479](https://github.com/vllm-project/vllm/pull/38479) — base TurboQuant
  backend (vibhavagarwal5)
- [#40092](https://github.com/vllm-project/vllm/pull/40092) — FA3/FA4 prefill
  wire-up (huangzhilin-hzl)
- [#40941](https://github.com/vllm-project/vllm/pull/40941) — WorkspaceManager
  sharing, fp16 Hadamard, kernel reduction (bhoomit)
- [#39931](https://github.com/vllm-project/vllm/pull/39931) — hybrid models +
  ROCm compatibility (JartX)

### 6.2 Algorithm

PolarQuant + Walsh-Hadamard rotation + Lloyd-Max scalar quantization. Reference:
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al, ICLR 2026). This is the **same algorithm family** as our
existing `GGML_TYPE_TURBO4_0` (Turbo4_0 specialization in
`mt_pagedattn_ops.cuh`).

Online quantization at store time (no offline calibration, no model
modifications). Per key vector:

1. Compute `||k||`, normalize to `x_hat = k / ||k||`.
2. Apply WHT rotation: `y = x_hat @ PiT` where `PiT = signs * H` (Hadamard with
   random sign flips, per-layer seed).
3. For each element of `y`: binary-search bucketize against sorted centroid
   midpoints; emit the centroid index.
4. Pack indices (3-bit or 4-bit) into the cache slot. Store `||k||` (fp16)
   alongside the indices.

Per value vector: uniform scalar quantization (`val = (v - v_min) * scale`).

### 6.3 Slot format presets

Per slot for head_dim=128:

| Preset | K bits | V bits | Slot bytes | Compression | GSM8K (baseline 0.900) | PPL Δ |
|---|---|---|---|---|---|---|
| `turboquant_k8v4` | **FP8 (E4M3)** | 4 | 196 | 2.6× | **0.860** (-4.4%) | +1.17% |
| `turboquant_4bit_nc` | 4 + NC | 4 + NC | 136 | 3.8× | 0.840 (-6.6%) | +2.71% |
| `turboquant_k3v4_nc` | 3 + NC | 4 + NC | 120 | 4.3× | 0.780 (-13%) | +10.63% |
| `turboquant_3bit_nc` | 3 + NC | 3 + NC | 104 | 4.9× | 0.720 (-20%) | +20.59% |

For our HEAD_SIZE=256 model (Qwen3.6-35B-A3B): scale each row by 2×. e.g.
`k8v4` → ~388 byte slots; `4bit_nc` → ~262 byte slots.

**Production-recommended presets for our fork: `k8v4` and `4bit_nc`**. The
3-bit variants have >10% accuracy loss and are research-only.

### 6.4 Why `k8v4` is the right default once FP8 hardware works

- Keys are quality-critical (softmax weights → errors amplify through `exp()`).
- Values are resilient (weighted average → noise tolerant).
- FP8 keys + 4-bit values balances precision exactly where it matters.
- RDNA4 has native FP8 hardware (E4M3FN), so the cast is essentially free.
- Quality data: +1.17% PPL on Qwen3-4B, vs +2.71% for 4bit_nc.

Our current `GGML_TYPE_TURBO4_0` is 4-bit symmetric. Adding the `k8v4` preset
is the right evolution.

### 6.5 Store kernel (`triton_turboquant_store`)

```python
@triton.jit
def _tq_fused_store_mse(
    Y_ptr,              # WHT-rotated x_hat: [N, D] fp32 (already y = x_hat @ PiT)
    Norms_ptr,          # ||k||: [N] fp16
    Value_ptr,          # raw V: [N, D]
    Midpoints_ptr,      # sorted centroid midpoints: [2^MSE_BITS - 1] fp32
    KV_cache_ptr,       # paged cache: [num_blocks, block_size, n_kv_heads, slot_size]
    Slot_mapping_ptr,   # [N] int32 — destination slot per row
    ...,
    MSE_BITS: tl.constexpr,        # 3 or 4
    N_CENTROIDS: tl.constexpr,     # 2^MSE_BITS
    ...
)
```

**Per row (1 thread block):**

1. Load WHT-rotated key + norm.
2. Binary-search bucketize each element against sorted midpoints → centroid index.
3. Pack indices: 4-bit nibble pairs, OR 3-bit triple-byte (8 indices in 24 bits).
4. Store `||k||` fp16 alongside packed indices.
5. Uniform quantize value: `v_quant = round((v - v_min) / v_scale)`, store
   indices + (v_min, v_scale) as fp16.

**WHT rotation is done OUTSIDE the kernel** as a `cuBLAS`/`rocBLAS` GEMM
(`y = x_hat @ PiT`). The kernel-author's note explicitly chose this over an
in-kernel butterfly:
> "single cuBLAS GEMM instead of log2(D) butterfly kernel launches. 64KB for D=128"

For our integration: this is a `hipBLASLt` call before the Triton kernel.

### 6.6 Decode kernel (`triton_turboquant_decode_attention`)

```python
@triton.jit
def _tq_decode_stage1(
    Q_rot_ptr,          # WHT-rotated query: [B, Hq, D]
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,      # [2^MSE_BITS, D] fp32 — per-layer codebook
    Mid_o_ptr,          # split-K partial outputs (workspace)
    ...,
    NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
    NUM_KV_SPLITS,      # split-K factor (configurable, default 32)
    KV_GROUP_SIZE,      # GQA fanout (num_queries_per_kv)
    MSE_BITS,           # 3 or 4
    KPS,                # key_packed_size (slot key bytes)
    VQB,                # value_quant_bits
    ATTN_SCALE,
    BLOCK_D, BLOCK_KV,  # tile sizes
    KEY_FP8: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    FP8_E4B15: tl.constexpr,
)
```

Grid: `(B, Hq, NUM_KV_SPLITS)`. Each program owns one (batch, q_head, kv_split).
Inside, tile the assigned KV range in `BLOCK_KV=4` token chunks:

1. Load packed key bytes from slot.
2. Bitshift-unpack MSE indices.
3. Gather centroid floats: `c_vals[i] = Centroids[idx[i]]`.
4. **Norm-correct** (if `NORM_CORRECTION=True`):
   ```python
   c_norm_sq  = sum(c_vals * c_vals)
   c_inv_norm = 1.0 / sqrt(c_norm_sq + 1e-16)
   c_vals     = c_vals * c_inv_norm
   ```
5. Compute `term1 = sum(q_rot * c_vals)` (dot product with rotated query).
6. Score: `scores = vec_norm * term1 * ATTN_SCALE`.
7. Online softmax: `m_prev`, `l_prev`, `re_scale = exp(m_prev - n_e_max)`.
8. Unpack values inline (3-bit triple-byte or 4-bit nibble pair).
9. Apply value dequant: `v_vals = v_idx * v_scale + v_zero`.
10. Accumulate: `acc = acc * re_scale + sum(p[:, None] * v_vals, 0)`.

Stage 2 reuses vLLM's existing `_fwd_kernel_stage2` for log-sum-exp reduction
across splits.

### 6.7 Norm Correction (the ~0.8% PPL free win)

When you Lloyd-Max-quantize a unit-norm vector coordinate-wise, the
*reconstructed* vector's norm drifts off 1.0 because each coordinate is rounded
independently. Without NC you reconstruct `||k|| * centroid`. With NC you
reconstruct `||k|| * (centroid / ||centroid||)`, which restores the original
normalization.

NC happens at decode time only (5 lines of Triton). Store kernel is unchanged.
NC is on for every preset except `turboquant_k8v4` (FP8 keys don't need it).

**This is a free addition to our existing `GGML_TYPE_TURBO4_0` dequant** — adds
~5 lines, gives ~0.8% PPL improvement.

### 6.8 Continuation prefill — two sub-paths

When `q_len > 1` AND we have cached KV (continuation of a previous request):

- **`q_len <= 128`** (default threshold): call the decode kernel per-row, with
  synthesized `seq_lens = arange(cached_len + 1, seq_len + 1)`. Reuses the
  decode path; no extra dequant.
- **`q_len > 128`**: fully dequant cached K/V to fp16 via `_tq_full_dequant_kv`
  kernel; inverse-rotate K via `k_flat @ Pi` (the inverse Hadamard, which is
  just H itself since H is self-inverse); concatenate cached and new tokens;
  call `flash_attn_varlen_func` on the fp16 tensor.

The threshold avoids `O(N² / chunk)` cost of the full dequant for short
continuations.

### 6.9 WorkspaceManager pattern

The continuation prefill path and the decode stage-1/stage-2 path allocate scratch
buffers. Per-layer allocation grows linearly with layer count: at 60 attention
layers × 2 buffers, a 1M-context request needed 58 GB of dequant scratch.

vLLM PR #40941 fixed this with `WorkspaceManager`: **one shared bump-pointer
allocator per ubatch**, allocated up front during memory profiling, reused
across layers. Size: 976 MB at 1M context (60× reduction).

`WorkspaceManager` interface (Python):

```python
class WorkspaceManager:
    def get_simultaneous(self, *shapes_and_dtypes) -> list[Tensor]:
        # Returns N views into a shared buffer, each aligned to 256 bytes
        ...
    def lock(self): ...    # Post-warmup; further growth raises
    def unlock(self): ...  # For elastic EP scaling
```

For our C++ port: ~100 LoC. Maps onto `ggml_cuda_pool_alloc` semantics —
one large pool allocation per step, with bump-pointer offsets handed out as
sub-views per kernel call.

---

## 7. Common Infrastructure

### 7.1 Triton AOT toolchain

**Build-time pipeline:**

1. CMake invokes `triton.compile()` (or the `triton.tools.compile` AOT entry
   point) on each `.py` kernel source.
2. Triton emits HSACO binary + a small C++ header with kernel metadata
   (entry-point symbol, num args, shared-memory size).
3. CMake compiles the C++ wrapper that loads HSACO via `hipModuleLoad` and
   dispatches via `hipModuleLaunchKernel`.
4. Resulting `.o` files link into `libggml-hip.a` as part of the standard build.

**Runtime:** zero Python dependency. The HSACO files are either embedded as
binary blobs (`objcopy --add-section`) or shipped as separate files loaded at
init time.

**Triton version requirements:**

- Triton **3.7.0 or newer** — required for explicit gfx1201 support
  (PR #10109/#10185/#10304 landed in mainline April-May 2026)
- Triton 3.6.0 (currently installed in `~/venvs/agents312`) is too old: AMD
  backend knows about gfx10/11/908/90a/942/94plus/950 but not gfx12
- Mainline build may be needed if 3.7.0 lacks some PRs we depend on

**Build-only dependency:** developers building from source need Triton installed.
End users running pre-built binaries do not. Document this in `BUILD.md`.

### 7.1.1 Verified behavior (POC 2026-05-17, Triton 3.7.0+git4768da5e)

End-to-end vector_add AOT proof-of-concept landed and ran on the R9700
(gfx1201) on 2026-05-17. The pipeline is solid; the following details are
empirically confirmed, not theoretical:

**Verified `triton.tools.compile` CLI:**

```bash
python -m triton.tools.compile <kernel.py> \
    --kernel-name add_kernel \
    --target hip:gfx1201:32 \
    --signature "*fp32:16, *fp32:16, *fp32:16, i32, 1024" \
    --grid "n_elements / 1024, 1, 1" \
    --num-warps 4 \
    --num-stages 1 \
    --out-name vector_add \
    --out-path /tmp/out/vector_add
```

Produces:

- `vector_add.<spec-hash>.c` — embedded HSACO blob + load/unload helpers + a
  `hipError_t <kernel>_<spec-hash>(hipStream_t, hipDeviceptr_t..., int32_t...)`
  launcher. `static hipModule_t _mod = NULL` and `_func = NULL` globals do
  lazy `hipModuleLoadData` on first call; subsequent calls just dispatch.
  License header: `SPDX-License-Identifier: MIT`, `Copyright AMD`.
- `vector_add.<spec-hash>.h` — declaration of the above. **Missing `extern "C"`
  guards** — C++ consumers must wrap their `#include` in `extern "C" { ... }`.

**Verified compile + link from C++:**

```bash
hipcc test.cpp vector_add.<spec-hash>.c -o test \
    -I/opt/rocm/include -I. \
    --offload-arch=gfx1201
```

`--offload-arch=gfx1201` is required on the final link, not just at AOT
compile time. Without it hipcc emits for some default arch and the HSACO
mismatch breaks at module load.

**Gotchas (worth filing upstream):**

1. **Python-syntax operators in `--grid` produce broken C.** Triton inlines
   the grid expression as-is into the C launcher (no operator translation).
   The natural Python integer-division `//` becomes a C line comment.
   Workaround: use `/` for int division. **File upstream**: either translate
   or document.

2. **No `extern "C"` guards in the generated `.h`.** C++ consumers hit
   linker errors unless they wrap the include manually. Trivial nit;
   worth a small upstream PR adding `#ifdef __cplusplus extern "C" {`.

3. **Spec-hash suffix is target-arch-independent.** Same kernel +
   signature + grid produces the same `<spec-hash>` for gfx1201 AND gfx1030
   builds (verified 2026-05-17). The *binary content* differs (different
   HSACO), but the C symbol names are identical: `vector_add_<spec-hash>`,
   `_hsaco`, `_mod`, `_func` all collide if you naively link both archs into
   one library. **Per-arch static libs** (TritonAOT.cmake already plans
   this) or **`triton.tools.link`** for a unified dispatched wrapper are
   the two viable workarounds.

**Lazy module load behavior:**

The generated launcher does:
```c
if (mod_func == NULL)
    load_kernel();  // calls hipModuleLoadData on first invocation
return hipModuleLaunchKernel(mod_func, gX, gY, gZ, ...);
```

So the per-process state is one `hipModule_t` per launcher per process,
shared across all callers, freed only via the explicit `unload_*` function.

### 7.2 KV cache layout convergence

**Migration target:**

Non-TurboQuant cache (F16/BF16/FP8):
```
(2, num_blocks, block_size, num_kv_heads, head_size)
```
This is the standard vLLM v1 layout. K and V are stored together with the
leading `2` separating them. `kv_cache.unbind(0)` produces K-slice and V-slice
both of shape `(num_blocks, block_size, num_kv_heads, head_size)` matching
AITER's expectations.

TurboQuant cache (all TQ variants):
```
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```
No leading-2 split. K and V are interleaved into a single uint8 slot per
(block, position, head). Per-preset `slot_size_aligned` rounded to next even
number.

**Migration impact in our codebase:**

- `mt_pagedattn_ops.cuh` — `paged_cache_ops<F16>` and `paged_cache_ops<TURBO4_0>`
  specializations rewrite K/V offset math
- `mt_pagedattn.cu` — scatter kernel `launch_scatter_kv<>` rewrites
- Tiered KV (hot/warm/cold) — eviction granularity is now (block, all-heads,
  all-data), simpler than before
- `paged-block`/semantic prefetch — reads from cache, needs new offset math
- `LockedBuffer` integration — no change, just operates on the new shape

**Migration strategy:** Add new layout in parallel with old. Migrate one cache
type at a time. F16 first (smallest blast radius), then turbo4 (which becomes
the new TurboQuant slot format). Drop old layout after both work.

### 7.3 WorkspaceManager C++ port

Approximate ~100 LoC class in `ggml/src/ggml-cuda/mt_workspace_manager.{h,cpp}`:

```cpp
class WorkspaceManager {
public:
    explicit WorkspaceManager(ggml_cuda_pool * pool, int num_ubatches = 1);

    struct View {
        void * ptr;
        size_t bytes;
    };

    // Returns N views into the underlying buffer, each 256-byte aligned.
    std::vector<View> get_simultaneous(
        std::initializer_list<size_t> sizes_bytes);

    // Resize the underlying buffer to at least the high-water-mark.
    void grow_if_needed(size_t total_bytes);

    void lock();   // After warmup; further grow_if_needed raises
    void unlock();

private:
    ggml_cuda_pool * pool_;
    int num_ubatches_;
    std::vector<size_t> current_sizes_;     // per ubatch slot
    std::vector<ggml_cuda_pool_alloc<uint8_t>> current_workspaces_;
    bool locked_ = false;
};
```

Used by the TurboQuant decode kernel (stage-1 `mid_o_buf`, stage-2
`output_buf`, `lse_buf`) and the continuation prefill (`k_dequant_buf`,
`v_dequant_buf`).

---

## 8. KV Layout Migration

### 8.1 Current state

Per `mt_pagedattn_ops.cuh`:

```cpp
// F16
K: [num_blocks, n_kv_heads, head_size/8, block_size, 8]   // x=8 coalesce stride
V: [num_blocks, n_kv_heads, head_size, block_size]        // head_dim-major

// TURBO4_0
K/V: [num_blocks, n_kv_heads, block_size, head_size / QK_TURBO4]
     of block_turbo4_0 (one half-precision norm + 64 packed-nibble bytes per 128-elem qblock)
```

### 8.2 Target state

```cpp
// F16/BF16/Q8_0  (AITER-compatible)
KV: [2, num_blocks, block_size, n_kv_heads, head_size]    // unbind(0) → K, V

// TurboQuant (all TQ presets)
KV: [num_blocks, block_size, n_kv_heads, slot_size_aligned]   // uint8 slots
```

### 8.3 Migration plan

**Phase 1 — additive.** Add new layout types `GGML_TYPE_F16_VLLM` (or similar) and
`GGML_TYPE_TURBO_VLLM` alongside existing `F16`/`TURBO4_0`. Implement
specializations of `paged_cache_ops<>`. Adapt scatter kernel.

**Phase 2 — switch new code.** AITER unified_attention and vLLM TurboQuant
backends only support the new layouts. Dispatch routes new KV types to new
backends.

**Phase 3 — migrate defaults.** Server defaults switch to new types. Old layouts
remain runnable for backward compatibility.

**Phase 4 — deprecate.** Mark old types deprecated. Remove after one minor
release cycle.

### 8.4 Tiered KV interaction

The tiered KV system (`mt-tiered`) moves blocks between hot (VRAM) / warm
(pinned host) / cold (SSD) tiers. Block granularity is unchanged — a paged
block remains the unit of tier movement.

Per-block byte size changes slightly with the new layout:

- Old F16: `n_kv_heads * head_size * block_size * 2` (K) + same (V)
- New F16: `2 * block_size * n_kv_heads * head_size * 2`   (same total, just contiguous)

No code change in `mt-tiered`; the block size in bytes is recomputed from the
new layout.

For TurboQuant: per-block size is `block_size * n_kv_heads * slot_size_aligned`.
This is smaller per-token than F16 (the compression win). Tiering becomes
proportionally less aggressive — more context fits hot.

---

## 9. Phased Rollout

Each phase is one or more Jira stories under [MAD-186](https://mad-lab-ai.atlassian.net/browse/MAD-186).

### Phase 1 — Triton AOT foundation ([MAD-187](https://mad-lab-ai.atlassian.net/browse/MAD-187))

**Goal:** prove the AOT toolchain works end-to-end on gfx1201.

- Set up a Python 3.12 venv with Triton 3.7+ and ROCm PyTorch
- Write a trivial vector-add `@triton.jit` kernel
- AOT-compile it for gfx1201, gfx1100, gfx1030
- Load via `hipModuleLoad`, launch, verify result
- Add CMake step: `find_program(TRITON_AOT triton)` + custom command per kernel
- Define env-var gate (`GGML_AITER_TRITON_BACKEND`) for runtime selection

**Acceptance:** `cmake --build` produces and links a Triton-generated HSACO
into `libggml-hip.a`. Vector-add test passes on R9700 (gfx1201) and 6900XT
(gfx1030).

### Phase 2 — AITER unified_attention integration ([MAD-188](https://mad-lab-ai.atlassian.net/browse/MAD-188))

**Depends on:** Phase 1.

- Vendor `unified_attention.py` + underlying `@jit` kernels under
  `ggml/src/ggml-cuda/aiter-integration/triton-kernels/`. License headers preserved.
- AOT-compile per `(HEAD_SIZE, KV_TYPE)` combination needed at runtime
- C++ launcher `mt_aiter_unified_attn.cu` extracts strides/shapes, dispatches
  via `hipModuleLaunchKernel`
- KV layout migration (Phase 1 of layout migration; F16/BF16 only)
- Dispatch gate in `mt_pagedattn.cu`: route F16/BF16 KV → AITER backend
- Bench against existing tile + flash-decode kernels at 8K, 32K, 128K, 400K

**Acceptance:**
- Decode rate at 400K ≥ 60 t/s (3-4× over hand-rolled MAD-185)
- Correctness vs scalar baseline within fp16 tolerance at all test ctx values
- No regression at short ctx (≤ 4K) — keep hand-rolled or scalar fallback

### Phase 3 — vLLM TurboQuant backend integration

**Depends on:** Phase 1, Phase 2.

New Jira story (TBD number) — to be filed.

- Vendor `triton_turboquant_decode.py`, `triton_turboquant_store.py`,
  `centroids.py`, `config.py` from vLLM under `aiter-integration/triton-kernels/`
- C++ port of `WorkspaceManager` (~100 LoC)
- AOT-compile TurboQuant kernels per `(HEAD_SIZE, MSE_BITS, KEY_FP8)` matrix
- C++ launcher `mt_turboquant_backend.cu` for store + decode + continuation
  prefill paths
- KV layout migration (Phase 2; TurboQuant slot layout)
- Dispatch gate: route TURBO4_0 / k8v4 / 4bit_nc → TurboQuant backend
- New GGML types: `GGML_TYPE_TURBOQUANT_K8V4`, `GGML_TYPE_TURBOQUANT_4BIT_NC`
- Convert `GGML_TYPE_TURBO4_0` to be a synonym for `4BIT_NC` (or deprecate)

**Acceptance:**
- TurboQuant backend produces correct output vs scalar on Qwen3.6-35B
- Decode rate at 400K ≥ 60 t/s on `k8v4` preset
- Memory profile shows shared-buffer pattern (1M ctx within VRAM budget)
- PPL within published vLLM bounds (+1.17% for k8v4, +2.71% for 4bit_nc)

### Phase 4 — Hybrid model + skip-layers ([NEW STORY])

**Depends on:** Phase 3.

- Implement `kv_cache_dtype_skip_layers` equivalent: first/last 2 layers stay
  at native precision via `--kv-tq-skip-layers` CLI
- Detect hybrid models (qwen35moe) via `full_attention_interval`; route
  non-attention layers to non-TQ backend automatically
- Mirror vLLM PR #39931 ROCm fixes (notably `flash_attn_varlen_func out=`
  incompatibility wrapper)

**Acceptance:** Qwen3.6-35B-A3B (hybrid) runs end-to-end with TurboQuant on
attention layers, native fp16 KV on the rest. Needle test passes at 200K+ ctx.

### Phase 5 — Cross-arch (gfx1030 / RDNA2 via FSR4-style INT8)

**Depends on:** Phases 1-4 stable.

New Jira story. Researches whether AITER attention can run on RDNA2/3 via INT8
emulation (the FSR4 pattern). Out of scope for this document.

### Phase 6 — Public positioning ([MAD-193](https://mad-lab-ai.atlassian.net/browse/MAD-193))

Gated on Phases 1-4 demonstrable. README rewrite, CREDITS.md, benchmark page,
fork naming.

---

## 10. Open Design Questions

### 10.1 Triton kernel deployment

Two options for shipping the AOT artifacts:

- **Embedded:** Use `objcopy --add-section` to embed HSACO into the linker
  binary. End users get one self-contained `llama-server` with no external
  files. Build complexity higher.
- **External:** Ship HSACO files alongside the binary in a `share/` directory.
  Loaded by path at init time. Simpler build, but users must keep them next to
  the binary.

Recommendation: embedded for releases, external during development.

### 10.2 Multi-(HEAD_SIZE, KV_TYPE, ARCH) AOT matrix + symbol collision

Each kernel `@jit` instance must be AOT-compiled for each
`(HEAD_SIZE, KV_TYPE, MSE_BITS, ARCH, ...)` combination it's used with.
The matrix for our target models:

- AITER: HEAD_SIZE ∈ {128, 256}, KV_TYPE ∈ {F16, BF16, FP8}, ARCH ∈ {gfx1201, gfx1030, …}
- TurboQuant: HEAD_SIZE ∈ {128, 256}, presets ∈ {k8v4, 4bit_nc}, ARCH ∈ {…}
- Total: ~16-24 AOT compilations per build (8-12 specs × 2-3 arches)

At ~5-15 seconds per compile, total build-time impact is ~2-6 minutes. Cache
artifacts in a `build/triton-cache/` directory keyed on kernel hash.

**Symbol collision across arches (verified 2026-05-17):** the `<spec-hash>`
suffix in generated launcher names is target-arch-independent. So gfx1201
and gfx1030 builds of the same kernel produce launchers with the *same C
symbol name* and the *same internal `_hsaco`/`_mod`/`_func` globals*.
Naive link of both into one library = duplicate-symbol error.

Three options (pick at MAD-187 milestone 2):

1. **Per-arch static libs.** Build a separate `aiter_triton_aot_gfx1201.a`
   and `aiter_triton_aot_gfx1030.a`; runtime dispatcher in ggml-hip selects
   one based on detected GPU. Two libraries per shipped binary, but each
   self-contained. **TritonAOT.cmake's current per-arch-subdir design
   naturally produces this — just need one CMake target per arch.**
2. **`triton.tools.link`** to merge per-arch specializations into a unified
   dispatched wrapper. Triton's intended approach; cleaner end product but
   another tool dependency at build time.
3. **`objcopy --redefine-sym`** to rename symbols per arch. Hacky;
   error-prone. Use only as a last resort.

Recommendation: **option 1 now**, evaluate option 2 as a follow-up.

### 10.3 Continuation prefill threshold

vLLM uses `_CONTINUATION_DECODE_THRESHOLD = 128`. We may want this configurable
via env var (`GGML_AITER_TQ_CONTINUATION_DECODE_THRESHOLD`). The right value
depends on q_len distribution in real workloads.

### 10.4 Skip-layers UX

How does the user configure boundary skip-layers?

- Auto (recommend): detect first/last 2 attention layers, skip TQ on those
- Explicit CLI: `--kv-tq-skip-layers 0,1,N-2,N-1`
- Both: auto by default, override via CLI

### 10.5 Sliding-window dispatch

AITER's `unified_attention` handles SWA natively. For hybrid models (qwen35moe),
should we:

- Route ALL layers through AITER (it handles both full and SWA correctly), OR
- Route only full-attention layers to AITER, send SWA layers through a different
  backend?

Recommendation: route all through AITER. SWA is handled correctly and TILE
pruning makes it nearly free.

### 10.6 Format convergence for `GGML_TYPE_TURBO4_0`

Our current `block_turbo4_0` stores a half-precision `norm` per 128-element
qblock. vLLM TurboQuant stores `||k||` per-row (per-token, not per-qblock) plus
a per-layer centroid table. Conversion options:

- **A:** Keep our format for backward compat, add new `TURBOQUANT_4BIT_NC` type
  with vLLM format. Migrate over time. Models requantize on load.
- **B:** Migrate `TURBO4_0` semantics to match vLLM. One-time format break;
  models must be requantized.

Recommendation: **A**. We have existing TURBO4_0 models in active use.

### 10.7 Centroid table source

vLLM uses Lloyd-Max-trained centroids embedded in `centroids.py`. Per-layer
seed is `tq_config.seed + layer_idx * 1337`. Should we:

- Use vLLM's centroids verbatim (compatible with vLLM-quantized models)
- Train our own centroids on a different calibration set
- Both — support multiple centroid sources via config

Recommendation: use vLLM's verbatim initially. Cross-tool compatibility is more
valuable than micro-quality wins from custom calibration.

---

## 11. Attribution

The integration described in this document builds on substantial upstream work.
We commit to maintaining accurate attribution in `CREDITS.md` and
prominently in our fork's README.

Current accumulated list (work in progress):

```
# Foundations
- ggml-org / llama.cpp                                 MIT — the foundation engine

# Attention kernel sources
- ROCm / aiter (ROCm/aiter)                            MIT — unified_attention.py with
                                                       explicit gfx1201 tuning
- vllm-project / vllm                                  Apache-2.0 — base TurboQuant
                                                       attention backend, store + decode
                                                       kernels, WorkspaceManager pattern

# vLLM TurboQuant PRs (all external contributors)
- vibhavagarwal5  (vllm-project/vllm#38479)            base TurboQuant integration
- huangzhilin-hzl (vllm-project/vllm#40092)            FA3/FA4 prefill wire-up
- bhoomit         (vllm-project/vllm#40941)            WorkspaceManager sharing,
                                                       fp16 Hadamard, kernel reduction
- JartX           (vllm-project/vllm#39931)            hybrid model + ROCm compat

# AITER PRs we depend on
- sunway513       (ROCm/aiter#2969)                    FlyDSL flash_attn_func for
                                                       gfx1201 (prefill, separate path)
- 0xDELUXA        (ROCm/aiter#2621)                    FP8 unlock for gfx1200/1201

# Triton enablement
- triton-lang / triton                                 MIT — Triton compiler
- alefimov-amd    (triton-lang/triton#10109)           GFX12 cache modifiers
- skysnow2001     (triton-lang/triton#10185)           IN_THREAD_TRANSPOSE gfx1201 tuning
- saeid-rostami   (triton-lang/triton#10304)           RDNA4 test fixes

# Algorithm
- Zandieh et al, ICLR 2026                             "TurboQuant: Online Vector
                                                       Quantization with Near-optimal
                                                       Distortion Rate"

# Tooling (future)
- AMD / Quark                                          MIT — quantization framework
```

This list is **not exhaustive** — every upstream contributor in the dependency
graph deserves credit. The fork's `CREDITS.md` is the canonical record and is
maintained alongside each upstream sync.

---

## 12. References

### Source files (external)

- ROCm/aiter `aiter/ops/triton/attention/unified_attention.py` (dispatcher)
- ROCm/aiter `aiter/ops/triton/_triton_kernels/attention/unified_attention.py`
  (kernels)
- vllm-project/vllm `vllm/v1/attention/backends/turboquant_attn.py`
- vllm-project/vllm `vllm/v1/attention/ops/triton_turboquant_store.py`
- vllm-project/vllm `vllm/v1/attention/ops/triton_turboquant_decode.py`
- vllm-project/vllm `vllm/model_executor/layers/quantization/turboquant/`
- vllm-project/vllm `vllm/v1/worker/workspace.py` (`WorkspaceManager`)
- vllm-project/vllm `vllm/v1/attention/backend.py` (abstract interfaces)
- vllm-project/vllm `vllm/v1/kv_cache_interface.py`
  (`FullAttentionSpec`, `TQFullAttentionSpec`)

### Pull requests

- [ROCm/aiter PR #2969](https://github.com/ROCm/aiter/pull/2969) — FlyDSL flash_attn_func for gfx1201
- [ROCm/aiter PR #2621](https://github.com/ROCm/aiter/pull/2621) — FP8 unlock for gfx1200/1201
- [vllm-project/vllm PR #38479](https://github.com/vllm-project/vllm/pull/38479) — TurboQuant base
- [vllm-project/vllm PR #40092](https://github.com/vllm-project/vllm/pull/40092) — FA3/FA4 prefill
- [vllm-project/vllm PR #40941](https://github.com/vllm-project/vllm/pull/40941) — WorkspaceManager
- [vllm-project/vllm PR #39931](https://github.com/vllm-project/vllm/pull/39931) — hybrid + ROCm
- [triton-lang/triton PR #10109](https://github.com/triton-lang/triton/pull/10109) — GFX12 cache modifiers
- [triton-lang/triton PR #10185](https://github.com/triton-lang/triton/pull/10185) — gfx1201 IN_THREAD_TRANSPOSE

### Papers

- Zandieh et al, "TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate," ICLR 2026.

### Internal

- [MAD-186](https://mad-lab-ai.atlassian.net/browse/MAD-186) — parent epic
- [MAD-187](https://mad-lab-ai.atlassian.net/browse/MAD-187) — Triton AOT integration backend
- [MAD-188](https://mad-lab-ai.atlassian.net/browse/MAD-188) — AITER unified_attention integration
- [MAD-189](https://mad-lab-ai.atlassian.net/browse/MAD-189) — AITER MoE kernel
- [MAD-190](https://mad-lab-ai.atlassian.net/browse/MAD-190) — Quark → GGUF MXFP4 converter
- [MAD-191](https://mad-lab-ai.atlassian.net/browse/MAD-191) — GGML_TYPE_MXFP8 weight types
- [MAD-192](https://mad-lab-ai.atlassian.net/browse/MAD-192) — FP8/MXFP8 KV cache + AITER FP8 wire-up
- [MAD-193](https://mad-lab-ai.atlassian.net/browse/MAD-193) — Fork public positioning

---

## 13. Glossary

- **AITER:** AMD's AI Tensor Engine for ROCm. Open-source kernel library.
  GitHub: ROCm/aiter.
- **AOT:** Ahead-of-time compilation. Producing a binary at build time rather
  than JIT-compiling at runtime.
- **CK:** Composable Kernel. AMD's templated C++ kernel library, alternative to
  Triton.
- **FlyDSL:** AMD's emerging in-house DSL for high-performance kernels. PR
  #2969 (FlyDSL flash_attn_func for gfx1201) is the first RDNA4 attention
  backend in AITER but is prefill-only.
- **FP8 E4M3FN:** IEEE 754 8-bit float, 4 exponent + 3 mantissa bits, FN ==
  "finite-only" (no ±∞, only NaN). Used on RDNA4 + Hopper+.
- **FP8 E4M3FNUZ:** AMD-specific FP8 variant, used on MI300 (gfx942). NOT the
  same as E4M3FN.
- **GQA:** Grouped-Query Attention. Multiple query heads share one KV head.
  `num_queries_per_kv = num_query_heads / num_kv_heads`.
- **HSACO:** HSA Code Object. Compiled GPU binary format for AMD ROCm.
- **MAD-NNN:** Internal Jira ticket numbers in our `MAD` project.
- **MFMA:** Matrix Fused Multiply-Add. CDNA instruction class (MI300/MI350).
- **MXFP4/MXFP8:** Microscaling FP4/FP8 with block-shared exponent. OCP
  standard.
- **NC (Norm Correction):** TurboQuant's read-time normalization correction.
- **PolarQuant:** A rotation-based quantization scheme. Decomposes a vector
  into magnitude + unit direction, quantizes them separately. Underlies
  TurboQuant.
- **RDNA4:** AMD's 4th-gen Radeon DNA architecture. R9700, R9600D, RX 9070 XT
  (gfx1200/1201).
- **TurboQuant:** The published algorithm (Zandieh et al ICLR'26) and the vLLM
  backend that implements it.
- **WHT:** Walsh-Hadamard Transform. An orthonormal, self-inverse rotation
  (`H = H^T = H^{-1}`) constructed by recursive Kronecker products of
  `[[1,1],[1,-1]]` / √2.
- **WMMA:** Wave Matrix Multiply-Accumulate. RDNA3/4 matrix instructions.
- **WMMA Layout v1/v2/v3:** Triton's enum. v1=RDNA3 (gfx1100/1101), v2=RDNA4
  (gfx1200/1201), v3=gfx1250.
