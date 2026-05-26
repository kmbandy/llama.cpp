# ml8-4 → GGUF / llama.cpp Production Integration Design

**Status:** DRAFT — design only, no code yet. Companion to `ML8_WMMA_KERNEL_DESIGN.md`
(the kernel-side design) and `ML8_PHASE_STATUS.md` (phase status tracker).

**Goal:** make ml8-4 a first-class GGML quantization type. Once shipped, an
`.gguf` file with ml8-typed tensors loads transparently in `llama-cli`,
`llama-server`, `llama-perplexity`, and the rest of the llama.cpp tool family
— no special flags, no Python prototypes, no shims. Drop a `.gguf` in front of
`llama-server`, it auto-detects the type, and dispatches into the
`mt_ml8_gemm` / `mt_ml8_moe_gemm` wrappers we already shipped (MAD-243 / 244)
via the standard ggml backend mechanism. Production-grade,
upstream-merge-shaped, no shortcuts.

This is the C++ counterpart to MAD-245 (the Python Ml8Linear). It carries the
forward-time rotation + AWQ math identity into the ggml graph so the same
calibrated artifacts work in both paths.

---

## 0. Scope + non-scope

**In scope:**
- New `GGML_TYPE_ML8_4` quantization type registered in ggml core
- On-disk GGUF format (main tensor + sidecar conventions)
- CPU dequant + vec_dot fallback (correctness reference; slow but portable)
- HIP backend dispatch into `mt_ml8_gemm` for dense linears
- HIP backend dispatch into `mt_ml8_moe_gemm` for MoE expert linears
- llama.cpp graph build: when the loader sees an ml8 tensor, the matmul
  graph node carries the rotation + AWQ sidecar bindings and applies them
  to activations (mirrors Ml8Linear.forward exactly)
- Conversion tool: calibration `.pt` blobs + base GGUF → ml8-typed GGUF
- Auto-detection: zero llama.cpp CLI changes; GGUF type ID drives dispatch

**Out of scope (future work):**
- CUDA NVIDIA backend (only HIP for now; AITER is HIP-only)
- Vulkan backend
- CPU vec_dot performance (CPU is fallback only)
- Activation 4-bit quantization (W4A4) — separate research direction
- llama.cpp upstream PR (file the design here first; upstream after we've
  battle-tested it on R9700 for a few weeks)

**Non-goals:**
- Don't change existing quant type IDs or block layouts
- Don't touch the Triton kernel — it's already shipped, the GGML side
  builds AROUND it via the C++ wrappers
- Don't introduce a new file format. GGUF + standard tensor naming covers
  everything we need

---

## 1. On-disk format

Anchor decision: **one main tensor (typed `GGML_TYPE_ML8_4`) plus four
sidecar tensors per Linear layer.** Sidecars use a tensor-name convention
that mirrors the existing turbo-FP8 KV path (which also stores per-(kv-layer)
runtime LUTs as sidecar tensors and decodes via the LUT at kernel time — see
`block_turbo3_fp8_bs256` and `mt_aiter_unified_attn.cpp`'s `centroids_k_ptr`
plumbing).

### 1.1 Main tensor: `GGML_TYPE_ML8_4`

| Field | Value |
|---|---|
| Type ID | **`GGML_TYPE_ML8_4 = 48`** (next slot after the turbo-FP8 family at 47) |
| Block element count `QK_ML8` | **64** (matches calibration's `group_size=64`) |
| Block struct | `block_ml8_4 { fp32 scale; uint8 indices[QK_ML8 / 2]; }` — 4 bytes scale + 32 bytes packed 4-bit indices = **36 bytes / 64 elements** |
| In-block bpv | 36 × 8 / 64 = **4.5 bpv** for indices+scales |
| Sidecar bpv contribution | centroids tensor adds (`n_groups_k × 16 fp8`) / (`N × K`) ≈ **0.001 bpv** for typical Qwen MLP shape (negligible) |
| **Effective bpv** | **~4.5 bpv** for indices+scales; the calibration's "4.25 bpv" comes from sharing LUT across all N rows per K-group — preserved here via the centroids sidecar |

Block layout (C struct, declared in `ggml/src/ggml-common.h`):

```c
#define QK_ML8 64
typedef struct {
    float   scale;                  // per-block fp32 scale (== row's b_scale[k_group])
    uint8_t qs[QK_ML8 / 2];         // 64 × 4-bit indices, lo-nibble first
} block_ml8_4;
static_assert(sizeof(block_ml8_4) == 4 + 32, "wrong block_ml8_4 size");
```

The main tensor in GGUF stores `[N × n_groups_k]` blocks back-to-back, in
row-major (N varies fastest within a K-group, then K-group). This matches the
kernel-expected layout (centroids per K-group, scales per (K-group, N)).

### 1.2 Sidecar tensors (per Linear layer)

For a calibrated weight at GGUF name `blk.{L}.{kind}.weight` (e.g.,
`blk.0.ffn_gate.weight`), four sidecars follow the same prefix:

| Sidecar name | dtype | shape | semantics |
|---|---|---|---|
| `blk.{L}.{kind}.weight.centroids` | `GGML_TYPE_F8_E4M3` (NEW — see §1.3) | `[n_groups_k, 16]` | Per-K-group LUT, 16 fp8 centroid values; indexed by main tensor's 4-bit indices |
| `blk.{L}.{kind}.weight.rotation_h_a` | `GGML_TYPE_F32` | `[a_dim, a_dim]` | KroneckerRotation's `h_a` factor (the only thing not deterministic; `h_b` is regenerated as Sylvester Hadamard at load time) |
| `blk.{L}.{kind}.weight.rotation_meta` | `GGML_TYPE_I32` (4 elements) | `[4]` | `{a_dim, b_dim, in_features, kind_id}` where `kind_id = 1` for `kronecker_orth_sylvester`. Tiny — fits in a 16-byte tensor |
| `blk.{L}.{kind}.weight.awq_scale` | `GGML_TYPE_F32` | `[in_features]` | Per-input-channel AWQ scale, applied to activations at forward time |

Any sidecar can be absent. Loader treats absent rotation/AWQ sidecars as
identity transforms (no-op). Centroids are required (the main tensor is
useless without them).

### 1.3 New auxiliary type: `GGML_TYPE_F8_E4M3`

The centroids sidecar uses fp8 e4m3 — already the native format the
calibration produces (after the `--snap-centroids e4m3` step) and what the
HIP kernel expects. No existing GGML type covers this exactly:
- `GGML_TYPE_F16` is wider than necessary
- `GGML_TYPE_Q8_0` is int8 with a block scale, not the e4m3 lattice

Add `GGML_TYPE_F8_E4M3 = 49`, `blck_size = 1`, `type_size = 1`. No
dequant function needed for the centroid use case — the GPU kernel
reads e4m3 bytes directly via `tl.load → tl.fp8_e4m3` and the CPU
fallback can convert on the fly inline.

(Aside: this is independent enough from ml8 that future quant types can
also use F8_E4M3 sidecars — kv-side turbo paths might benefit too.)

### 1.4 GGUF metadata

No new top-level GGUF keys required for auto-detection. Standard tensor
type-ID dispatch already handles everything once `GGML_TYPE_ML8_4` is
registered. Optional metadata for diagnostics / debugging:

| Key | Value |
|---|---|
| `ml8.calibration.recipe` | string, e.g., `"cell-e"` or `"cell-c"` — informational only |
| `ml8.calibration.seed` | uint32 |
| `ml8.calibration.git_sha` | string |

Tooling-only. Not read by the inference path.

---

## 2. Dispatch flow

### 2.1 CPU fallback (reference + verification)

`ggml/src/ggml-quants.c`:

- `dequantize_row_ml8_4(const block_ml8_4 *x, float *y, int64_t k, const void *aux)`
  - `aux` carries: pointer to centroid LUT for the current K-group, K-group index
  - For each block of 64 values: scale = x->scale; for each pair of 4-bit
    indices, look up fp8 centroid in LUT, convert to fp32, multiply by scale,
    write to y. No vectorization required — fallback only.
- `ggml_vec_dot_ml8_4_q8_K(...)` — standard pattern: dequant left, dot with
  q8_K right (activations get quantized to q8_K elsewhere in the graph).
- `from_float` is unsupported for ml8 (calibration is offline). Function
  pointer is null in `ggml_type_traits`; loader rejects any attempt to call it.

Rotation + AWQ are NOT applied here — they're handled by the matmul graph
node wrapping the dequant call (see §2.3). Keeping dequant pure simplifies
testing.

### 2.2 HIP backend (production path on R9700)

`ggml/src/ggml-cuda/`:

- New file `ggml/src/ggml-cuda/ml8.cu` (or `.hip.cpp` — match repo
  convention) registers ML8_4 with the cuda backend's matmul dispatcher.
- The dispatcher receives a ggml node `MUL_MAT` whose `src[0]` is an
  ml8-typed tensor. It:
  1. Extracts strides + dims from the ggml tensors
  2. Calls into `mt_ml8_gemm` (or `mt_ml8_moe_gemm` for MoE — selected by
     whether `src[0]` is a 3-D expert-stacked tensor)
  3. Passes the rotation + AWQ sidecar buffers through `mt_ml8_gemm_args_t`
     and lets the wrapper handle the kernel call

Sidecars are bound to the matmul node at graph build time (§2.3), so by
the time dispatch runs, all four sidecar pointers are available on the
node's `op_params` or via a small extension to the cuda context state.

### 2.3 Graph build (where rotation + AWQ get wired)

`src/llama-model.cpp` — when constructing the Linear/MLP graph for a layer
whose weight is `GGML_TYPE_ML8_4`:

1. Load main tensor as `W`
2. Load sidecars `W.centroids`, `W.rotation_h_a` (if present),
   `W.rotation_meta`, `W.awq_scale` (if present)
3. Construct the matmul node `y = ml8_matmul(x, W, centroids, h_a, awq_scale)`
   — a custom op or `MUL_MAT` with extended `op_params`
4. The backend dispatcher (CPU or HIP) sees this node and runs:
   - `x → x * awq_scale` (if awq present)
   - `x → rotate.forward(x)` (if rotation present; Kronecker math in CPU,
     fused into the kernel call on HIP via the wrapper's existing path)
   - GEMM
5. The resulting `y` is a normal fp16/fp32 tensor and continues into the
   rest of the graph (residual, layer norm, etc.) unchanged

For HIP, the rotation matrix multiplication is currently a separate
PyTorch op in Ml8Linear; in the ggml path it can either:
- (a) Be a separate ggml `MUL_MAT` node (x @ Q_dense reconstructed) —
  simplest but materializes Q
- (b) Stay factored as Kronecker (x reshape → small Q @ X @ small Q.T →
  reshape) — saves memory, takes 2 small ggml matmuls
- (c) Get fused inside the Triton kernel (Phase H+)

**Decision: start with (b)** — the math is already in `kronecker_rotation.py`
and the small Q matrices live in the rotation_h_a sidecar. Phase H can
fuse into the kernel later for perf.

### 2.4 MoE dispatch

Mirrors §2.2. The MoE GGUF layout already stacks experts (e.g.,
`blk.0.ffn_gate_exps.weight` is a 3-D `[n_experts, N, K]` tensor). For
ml8, the same applies plus per-expert sidecars:
- `blk.0.ffn_gate_exps.weight.centroids` is `[n_experts, n_groups_k, 16]`
- Other sidecars dimensioned similarly

`mt_ml8_moe_gemm` already takes the full feature surface (PATCH #6,
MAD-244), so this is a wiring exercise once the dense path is done.

---

## 3. Conversion pipeline

Rewrite `scripts/calibration/ml8_to_gguf.py`. Current behavior: dequant + embed as
fp16 (patcher). New behavior: write actual ml8-typed tensors + sidecars.

```
Inputs:
  --base-gguf <path>          # base f16 or bf16 GGUF (non-MLP tensors come from here)
  --calibration-dir <path>    # dir of .pt blobs from calibrate_ml8.py
  --out <path>                # destination .gguf
  --include-sidecars rotation,awq,centroids   # default all present

For each .pt blob in calibration-dir:
  1. Read indices, centroids, scales, rotation, awq from blob
  2. Pack indices into block_ml8_4 layout (per-row, per-K-group)
  3. Write main tensor `blk.{L}.{kind}.weight` as GGML_TYPE_ML8_4
  4. Write centroids sidecar (transpose if needed to [n_groups_k, 16])
  5. If rotation present: write rotation_h_a (a×a fp32) + rotation_meta (4×i32)
  6. If awq present: write awq_scale (K fp32)

For all other tensors in base-gguf: pass through unchanged.

Emit `ml8.calibration.*` GGUF metadata (informational).
```

End state: single `.gguf` file consumable by any llama.cpp tool. The
existing `gguf-py` Python library handles the on-disk write — we extend it
with `GGMLQuantizationType.ML8_4 = 48` and the matching block struct.

---

## 4. Acceptance / verification

A round-trip is the gate. For each phase landing:

1. **Round-trip-CPU:** convert Cell E → ml8.gguf, load via CPU backend,
   run `llama-cli -p "Hello"` for 10 tokens, compare output tokens with
   the Python `reconstruct_model.py --use-ml8-kernel` path output. **Match
   token-for-token** (deterministic CPU). This validates the format and
   the dequant path independent of the GPU kernel.
2. **Round-trip-HIP:** same convert, load via HIP backend, dispatcher
   calls `mt_ml8_gemm`. Generate 10 tokens, compare to CPU round-trip:
   **expect near-match (top-K identical for most tokens, may diverge after
   ~5 tokens due to fp8-act-quant noise — same as our PPL +0.019
   measurement).**
3. **llama-perplexity Cell E:** `llama-perplexity -m cell-e.gguf` on
   wikitext-2 — produces a PPL we can cross-reference. Expect ~8.62 ±
   noise (per the known HF↔llama-perplexity framework offset of +0.30).
   Headline: kernel-path PPL in the llama-perplexity world.
4. **llama-server chat:** boot, chat-via-curl. Interactive responses
   coherent.
5. **MoE round-trip:** repeat 1-3 on Qwen3.6-35B-A3B once Phase E
   calibration is done.

---

## 5. Phase decomposition

Companion entries to add in `ML8_PHASE_STATUS.md`:

| Phase | Subject | Estimated sessions | Gate |
|---|---|---|---|
| G.1 | Format design (this doc) + GGML type registration skeleton + builds clean | 1 | `make -j8 llama-cli` builds; ml8_4 type appears in `llama-quantize --help` listing (informational) |
| G.2 | CPU dequant + vec_dot fallback + first round-trip test (no rotation/AWQ) | 1-2 | CPU round-trip test #1 above passes on a 1-layer toy model |
| G.3 | Sidecar tensor loading + graph build with rotation + AWQ application (CPU only first) | 1-2 | CPU round-trip on rotated/AWQ Cell E artifact matches Python within fp32 precision |
| G.4 | HIP backend dispatch into `mt_ml8_gemm` (dense only) | 1-2 | Round-trip-HIP test #2 above passes |
| G.5 | Conversion tool (`ml8_to_gguf.py` v2) + Cell E → `.gguf` artifact | 1 | Produces a valid .gguf with all sidecars present |
| G.6 | llama-perplexity Cell E end-to-end + llama-server chat smoke | 1 | Tests #3 and #4 above pass |
| G.7 | MoE: `mt_ml8_moe_gemm` HIP dispatch + Phase E end-to-end | 2 | Gated on Phase E calibration |
| G.8 | (Optional) CPU vec_dot perf — vectorize the fallback | 1 | Out of critical path |

**G.1-G.6 is the production-grade dense ml8-in-llama-server release.**
G.7 adds MoE for the 35B-A3B story. G.8 is polish.

---

## 6. Open design questions (decide as we go)

- **Q1:** `block_ml8_4` size 64 — should it be 64 or some larger super-block
  (matches Q4_K's 256 with 8 sub-blocks of 32)? 64 keeps the calibration's
  group_size alignment trivial; larger blocks may improve memory bandwidth
  on CPU. Start with 64; revisit in G.8.
- **Q2:** rotation matrix application — graph-node (b) vs kernel-fused (c)
  per §2.3? Start with (b), measure cost, decide if (c) is worth the kernel
  surgery. The kernel-fused version requires plumbing rotation tensors
  through the Triton signature and adding tl.constexpr branches (cf. PATCH
  #6 pattern) — non-trivial.
- **Q3:** activation quantization — currently per-row max-abs fp8 in
  Ml8Linear. Same in the C++ path? Yes — keeps PPL bit-equivalent to Python.
  Plumbing: the graph node also emits the per-row scale as a small fp32
  tensor that flows into the kernel arg.
- **Q4:** does any of this need to land in upstream llama.cpp? Eventually
  yes (so other AMD users can consume our format), but not until we've
  shipped + dogfooded for 2-4 weeks. Filing the design here first.

---

## 7. References

- `ML8_WMMA_KERNEL_DESIGN.md` — kernel side (already shipped Phase A-C.3)
- `ML8_PHASE_STATUS.md` — phase status; G.x entries will be added at G.1 land
- `scripts/calibration/ml8_io.py` — current Python format reader (mirror
  the math in the C++ loader)
- `scripts/calibration/ml8_runtime.py::Ml8Linear` — Python forward, the
  authoritative reference for the graph-build math
- `wrappers/mt_ml8_gemm.{h,cpp}` — dense kernel wrapper, dispatch target
- `wrappers/mt_ml8_moe_gemm.{h,cpp}` — MoE kernel wrapper
- `ggml/src/ggml-common.h::block_turbo3_fp8_bs256` — the closest existing
  precedent in this tree (LUT-based block quant with runtime LUT sidecar)
- KG `[[ml8-forward-time-rotation-awq-2026-05-26]]` — the math identity
- KG `[[triton-aot-three-patches-2026-05-26]]` + companion patches —
  AOT-compat patches the wrappers depend on
