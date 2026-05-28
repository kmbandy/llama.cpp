# ML8 MoE Storage-as-Repacked (SOA) — Design

**MAD-223 / MAD-244 follow-up.** Eliminates the runtime MoE repack cache
that OOM'd a 24 GB Qwen3.6-35B-A3B model on the 32 GB R9700.

## Problem

The current ML8 MoE path requires a **runtime layout transformation**
before each kernel call. The GGUF stores ML8_4 weights in block-AOS
layout (per-group `[scale_fp32, nibbles_32B]` blocks). The AITER MoE
kernel (`_moe_gemm_a8w8_blockscale`) expects SOA layout (separated
`b_packed[K/2, N]` nibbles array + `b_scale[n_groups_k, N]` scales
array). `ggml_cuda_ml8_get_or_repack_moe` does the transform and caches
the result keyed by `w->data`.

This is fine for the **dense** ML8 path — each weight tensor is one
matrix, repack memory is small relative to model size, cache is
amortized once at load.

For **MoE** the repack overhead is fatal:

- One ML8 MoE tensor stores `n_experts` matrices stacked along axis 2:
  shape `[K, N, n_experts]`. For Qwen3.6-A3B: `[2048, 512, 256]`
  (gate/up) or `[512, 2048, 256]` (down).
- Per-tensor repack memory:
  `n_experts × (K/2 × N) + n_experts × (n_groups_k × N) × 4 bytes`
  ≈ **150 MB**.
- Qwen3.6-A3B has 40 layers × 3 MoE-expert tensors = **120 weight
  tensors**. Cache working set = **120 × 150 MB = 18 GB**.
- 24 GB model + 18 GB repack + KV cache + compile scratch exceeds the
  R9700's 32 GB VRAM. Warmup OOMs at layer ~17.

The total bytes the repack produces are **identical** to the stored ML8
bytes (`K × 0.5625 × N × n_experts`). The repack is a pure layout
transform, not a quantity inflation. Storing the SOA layout directly in
the GGUF makes the runtime transform unnecessary.

## Goal

For MoE expert tensors, store ML8 weights in the AITER kernel's expected
SOA layout, so the kernel reads `b_packed` and `b_scale` directly from
the tensor buffer with no per-call repack and no cache.

Non-goals:

- The dense ML8 path is unchanged. The dense repack cache is correct
  for its working set.
- No change to centroids, AWQ scale, rotation sidecars, or kernel
  signature.
- No change to ML8 quantization math (4-bit indices into a 16-entry
  fp8 LUT, group_size=64).

## Design

### A new ggml type: `GGML_TYPE_ML8_4_SOA`

A second type with the same total bytes as `ML8_4` but a different byte
layout within the tensor:

| | ML8_4 (existing, dense AOS) | ML8_4_SOA (new, MoE SOA) |
|---|---|---|
| Per-row bytes | `n_groups_k × 36` (block-interleaved) | `K/2 + n_groups_k × 4` (split) |
| Row layout | `[blk_g0(36B), blk_g1(36B), …]` | `[nibbles(K/2 B), scales(n_groups_k × 4 B)]` |
| Tensor dim | 2D `[K, N]` or 3D `[K, N, n_experts]` | 3D `[K, N, n_experts]` only |
| Total bytes per tensor | `n_experts × N × n_groups_k × 36` | `n_experts × N × (K/2 + n_groups_k × 4)` |

These two formulas evaluate to the same number (`n_experts × N × K × 9/16`).
Picking SOA in the file does **not** change the GGUF file size.

The new type lives alongside `ML8_4` in the ggml type table. Both
remain supported. Dense models continue to use `ML8_4`. The MoE writer
emits `ML8_4_SOA`.

### File format

For each MoE-expert tensor (e.g. `blk.0.ffn_gate_exps.weight`):

```
shape:    [K, N, n_experts]              ← unchanged
dtype:    ML8_4_SOA                       ← new
bytes:    [n_experts][N][K/2 + n_groups_k × 4]
```

Within each expert × N row, the bytes are laid out so the kernel can
take two views of the tensor:

```
b_packed view:  ptr + 0,                          stride = K/2 per row
b_scale  view:  ptr + n_experts × N × K/2,        stride = n_groups_k × 4 per row
```

i.e. the entire `b_packed` block precedes the entire `b_scale` block in
each expert chunk. This matches the kernel's expected pointer layout
exactly — no per-row deinterleaving at runtime.

### `ml8_to_gguf.py` (writer changes)

In `convert_to_ml8_gguf`, when writing MoE-expert tensors:

1. For each expert blob, derive `b_packed_e` (uint8 `[K/2, N]`) and
   `b_scale_e` (fp32 `[n_groups_k, N]`) using the **same** byte-level
   transform that `ml8_repack_kernel` does today.
2. Concatenate per-expert: `b_packed_e.tobytes() + b_scale_e.tobytes()`.
3. Stack across experts on axis 0.
4. `writer.add_tensor(name, stacked, raw_dtype=ML8_4_SOA)`.

The existing `pack_ml8_blocks` (block-AOS packing) is retained for the
dense path. A new `pack_ml8_blocks_soa(indices, scales)` is added.

### `ml8.cu` (runtime changes)

`ggml_cuda_op_ml8_mul_mat_id`:

1. Check `w->type == GGML_TYPE_ML8_4 || w->type == GGML_TYPE_ML8_4_SOA`.
2. If SOA: compute `b_packed_ptr` and `b_scale_ptr` directly from
   `w->data` (offsets fixed by the file layout above). Skip the call to
   `ggml_cuda_ml8_get_or_repack_moe` entirely.
3. If AOS (legacy fallback): existing repack path — preserved so models
   written before this change still load.

The `ml8_weight_repack_moe_t` struct and `g_ml8_moe_cache` map remain
in place but are unused by SOA models. They can be removed in a later
cleanup once all in-use MoE GGUFs are SOA.

### Loader (`qwen35moe.cpp`)

No change. The loader assigns `layer.ffn_gate_exps`, `ffn_up_exps`,
`ffn_down_exps` from whatever type is in the file. The forward path
(`build_moe_ffn_ml8`) hands them to `ggml_ml8_mul_mat_id`, which
dispatches by type in `ml8.cu`.

### Centroids / AWQ / rotation sidecars

No change. These tensors are independent of the main weight layout.

## Conversion

Re-running `ml8_to_gguf.py` on the existing 35B-A3B calibration blobs
takes the same wall time as the original conversion: a few minutes for
the MoE tensors plus the pass-through of the base GGUF. The
memory-pressure mitigations added yesterday (madvise `SEQUENTIAL`,
`posix_fadvise(DONTNEED)` per pass-through tensor, `use_temp_file=True`)
all apply unchanged.

No re-calibration is required. The blob format on disk (centroids,
indices, per-group scales) is unchanged.

## Validation plan

1. **Unit-level**: byte-equivalence between
   `pack_ml8_blocks_soa(indices, scales)` and running
   `pack_ml8_blocks(indices, scales)` then applying the
   `ml8_repack_kernel` transform offline. Confirms the writer produces
   bit-identical output to what the runtime would have computed.
2. **End-to-end smoke**: load the converted 35B-A3B GGUF, run a
   single-token decode, confirm the MoE kernel dispatches without
   touching `ggml_cuda_ml8_get_or_repack_moe`.
3. **PPL eval**: wikitext-2 perplexity run. Confirms numerical
   equivalence to a dense ml8-4 reference (or, if no dense reference
   exists at this size, confirms the value falls within expected
   bounds for an ml8-4 model).
4. **VRAM accounting**: confirm steady-state VRAM after warmup is
   within ~500 MB of the model size (24 GB + KV cache + AITER
   scratch).

## Out of scope

- Migrating the dense path to SOA. Defer until a dense user hits a
  memory-budget issue we can't otherwise address.
- Removing the `ml8_weight_repack_moe_t` machinery. Keep it for one
  release as a fallback for any GGUFs in the wild that pre-date this
  change.
- Multi-GPU repack distribution. Not needed once SOA lands; the design
  removes the pressure.
