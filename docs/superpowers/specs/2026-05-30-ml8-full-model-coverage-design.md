# ml8 Full-Model Coverage (Design A) — Design Spec

**Date:** 2026-05-30
**Status:** Approved for planning
**Jira:** MAD-223 (ml8 weight-quant epic) — extends FFN-only to full-model
**Branch base:** `feat/upstream-merge-2026-05-27`

## Problem

ml8 calibration today is **FFN-only**: `find_target_linears` matches only
`mlp.{gate,up,down}_proj` (`calibrate_ml8_paged.py:112,:531`). Every other matmul
weight — attention, the SSM/linear-attn projections, `lm_head`, and the embedding —
is resident-loaded **bf16, un-quantized** (`:840`).

- **MoE (qwen35moe):** experts ARE the FFN and dominate the weight, so FFN-only is
  competitive.
- **Dense hybrid (qwen35, e.g. the 9B/27B):** FFN is a **minority** of the weight, so
  FFN-only leaves **~76 % bf16** → the 11.46 GB "4-bit" 9B that triggered this work.

A produced GGUF that is mostly bf16 must never again be silently labelled 4-bit
(the coverage guardrail in `ml8_to_gguf.py`, landed 2026-05-30, now refuses it). This
spec makes dense ml8 a **real ~4-bit product** by extending ml8 — and a companion
8-bit FP8 tier — to the whole model, full-stack: calibration target set → converter →
GGUF type → C++ inference (loader + graph dispatch + kernels).

## Goal

One sentence: **bring every model weight onto the FP8 lattice** — 4-bit ml8 for the
rotatable GEMMs, 8-bit scaled-FP8 for the few tensors outside that class, f32 only for
the irreducible SSM recurrence core — for both qwen35 (dense) and qwen35moe (MoE), with
dense 9B as the first end-to-end validated target.

## Design ethos alignment

This is **not** dynamic per-tensor bit allocation (the UD tier-jumping move the ml8
ethos rejects). The bit-width tracks **where incoherence processing makes low-bit
lossless**:

- **4-bit** where rotation/AWQ can be applied-and-undone through a linear GEMM
  (`y = (WR)(R⁻¹x)`) → outliers spread → a uniform 4-bit grid fits.
- **8-bit FP8** where that trick structurally cannot apply: a **gather** (`token_embd`,
  no matmul) or a **nonlinear recurrence gate** (`ssm_alpha`/`ssm_beta`, whose output
  enters the SSM state update through a nonlinearity, so there is no downstream matmul
  to cancel a rotation against). 8-bit's exponent range + per-group scale handles these
  natively, no rotation required.
- **f32** for the recurrence core (`A`, `dt`, `conv1d`) and norms — a few hundred KB of
  irreducible state machinery; quantizing them saves nothing and risks compounding
  error through the recurrence.

Honest one-liner: *"every weight is on the FP8 lattice except the SSM recurrence core
and norms."* No INT, no bf16, fully labelled at true bpv.

---

## The precision map (authoritative)

Derived from the actual Qwen3.5-9B GGUF (33 layers, 3 layer signatures, vocab 248,320).

| Tier | Tensors | Calibration | Inference consumer |
|---|---|---|---|
| **4-bit ml8** (`ML8_4`: 4-bit index → FP8 centroid LUT + per-group scale + optional rotation/AWQ) | `attn_q`, `attn_k`, `attn_v`, `attn_output` (9 full-attn layers); `attn_qkv`, `attn_gate`, `ssm_out` (24 gated-deltanet layers); `ffn_gate/up/down`; `nextn.eh_proj`; `output` (lm_head) | Full GPTQ → Lloyd-Max(16) → E4M3-snap → rotation/AWQ (unchanged, matrix-agnostic) | `ggml_ml8_mul_mat` (existing) |
| **8-bit scaled FP8** (new type: e4m3 values + per-group fp16 scale, no rotation) | `token_embd`; `ssm_alpha`; `ssm_beta` | Direct per-group scale + e4m3 cast (no Hessian) | `token_embd` → new FP8 `get_rows` case; `ssm_alpha/beta` → **real FP8-WMMA `mul_mat`** (ml8 kernel minus the centroid-LUT indirection) |
| **f32 (recurrence/structural core)** | `ssm_a` (A), `ssm_dt.bias` (dt), `ssm_conv1d`, all norms (`*_norm`, q/k norms), `nextn.*norm` | none | unchanged |

Notes:
- `token_embd` and `output` are **untied** (separate tensors) — no shared-storage
  complication. `output` (lm_head, ~1.0B params) is the single biggest ml8 win.
- `token_embd` (~1.0B params) is a **gather**, not a matmul → 8-bit FP8 (scaled),
  consumed by `get_rows`. It is the only tensor whose 8-bit choice is brand-vs-Q8: we
  choose FP8 to keep the all-FP8 story; perf-neutral because a gather never hits the
  matrix cores. (lm_head, the per-token output GEMM, carries the real FP8 matmul perf.)
- `ssm_alpha/beta` are tiny ([4096×32]) but **real matmuls** → real FP8-WMMA at runtime.
- `conv1d` stays f32: ~2.3 MB to FP8, expands to f32 at runtime anyway (conv, not on
  the FP8-WMMA path), and adds error to a recurrence-adjacent weight — bad trade.

---

## Architecture (how the pieces fit)

Existing facts the design leans on (verified 2026-05-30):
- **Op dispatch is automatic by type:** `llama-model-loader.cpp:910` rewrites
  `MUL_MAT` with an `ML8_4` weight to `ML8_MUL_MAT` (and `MUL_MAT_ID` →
  `ML8_MUL_MAT_ID` for `ML8_4`/`ML8_4_SOA`). So once a weight is ml8-typed, op
  selection is free — the gap is **sidecar wiring** (the matmul node needs its centroid
  /rotation/awq tensors) and **calibration coverage**, not op dispatch.
- **Sidecars are wired per-call-site, not generically:** dense ml8 lives in the
  `qwen35.cpp` sidecar; MoE in `build_moe_ffn_ml8` (`llama-graph.cpp:1804`). Sidecars
  are explicit named fields on the layer struct (`ffn_*_centroids/rotation_h_a/awq_scale`,
  `llama-model.h:497+`).
- **`get_rows` type table** (`getrows.cu:181-217`): F16/F32/I32/BF16/Q*_0/**Q8_0** —
  **no `F8_E4M3`, no `ML8_4`.** So a scaled-FP8 embed needs a new `get_rows` case.
- **ml8 FP8 dequant lives in the matmul kernel, not `get_rows`/`to_float`**
  (`ggml.c:815`): "real dequant lives in the matmul graph node." This is why ml8 FP8
  works without a `get_rows` entry — and why the scaled-FP8 small-matmul path is "the
  ml8 kernel minus the LUT."
- **FP8→float primitive already exists and is shared:** `ggml_cuda_ue4m3_to_fp32`
  (`common.cuh:830`, handles both AMD `_fnuz` and NV encodings). Reused by the new
  scaled-FP8 paths.

### Component map

```
calibration (Python)            converter (Python)         inference (C++)
─────────────────────           ──────────────────         ───────────────
find_target_linears  ──roles──> ml8_to_gguf.py    ──GGUF──> llama-model-loader (op swap, exists)
  (role-tagged set)               ml8 tier: existing         llama-model.h  (sidecar fields / registry)
per-matrix GPTQ/ml8               sidecar write              llama-graph    (ml8-aware mul_mat helper)
  (unchanged, hooked)             scaled-FP8 tier: new       ggml-cuda:
scaled-FP8 quantizer  ─new─       type + sidecar               ml8.cu        (real FP8 mul_mat, no-LUT mode)
  (embed, α, β)                   coverage metric              getrows.cu    (F8_E4M3 case)
sensitivity instrument          + native .ml8 output        ggml.c         (scaled-FP8 type traits + to_float)
```

---

## Section-by-section design

### S1 — Calibration (`scripts/calibration/`)

**S1.1 Role-tagged target set.** Generalize `find_target_linears` from the FFN regex to
a role-tagged matcher returning `(module, gguf_name, role)` for the ml8 tier:
`attn_q/k/v/output`, `attn_qkv`, `attn_gate`, `ssm_out`, `nextn.eh_proj`, `output`.
The per-matrix ml8 routine (GPTQ Cholesky-of-H⁻¹ → Lloyd-Max(16) → E4M3-snap →
optional Kronecker rotation + AWQ) is **matrix-agnostic and unchanged** — the only new
work is **hooking these modules to capture their input activations** for the Hessian
(today only FFN linears are hooked).

**S1.2 Scaled-FP8 quantizer (new).** A small module: per-group (group_size configurable,
default 32 per MAD-214) compute `scale = max(|w_group|)/e4m3_max`, round `w/scale` to the
e4m3 lattice (reuse the calibration-side e4m3 snap). No Hessian, no rotation. Applied to
`token_embd`, `ssm_alpha`, `ssm_beta`. Emits `{e4m3_bytes, scale}` blobs.

**S1.3 SSM sensitivity instrument (mlambaformer dividend).** During calibration, record
per-channel + per-token kurtosis (the `down_proj_rig` gauge) for `ssm_alpha/beta/A/dt`,
plus a quant-sensitivity probe (Δ measured cost of 8-bit FP8 on the gates). This is the
first characterization of SSM-gate quant dynamics (quant-native-architecture lever #5).

**Budget note:** calibrating ~7 more linears/layer × 33 layers + the two vocab tensors
makes the dense 9B calibration heavier than FFN-only. Local/resident is fine; the 35B MoE
add is modest on top of the experts.

### S2 — Format + converter (`ml8_to_gguf.py`, `gguf-py`, `ggml.c`)

**S2.1 New GGML type — scaled F8_E4M3 weight.** Register a `GGML_TYPE_ML8_FP8` (working
name) = e4m3 values + per-group fp16 scale. Add `gguf-py` `GGMLQuantizationType` entry +
block layout, and `ggml.c` type-traits with a `to_float` (used by the conv-style
expand-at-load fallback and the CPU mul_mat dequant path). Disk-side it is the canonical
8-bit-FP8 representation shared by `token_embd`, `ssm_alpha`, `ssm_beta`.

**S2.2 Converter.** `ml8_to_gguf.py` already writes ml8 tensors + sidecars; extend the
blob discovery to the new ml8 roles (reuse the exact sidecar path), and add a writer for
the scaled-FP8 tier (new type). Produce **both** the GGUF-wrapped artifact **and** the
native `.ml8` artifact (the two-formats rule).

**S2.3 Coverage metric refinement.** The guardrail currently credits only `ML8_4` as
"quantized" and counts every 2-D `.weight` passthrough as bf16. Update
`evaluate_coverage`/the byte accounting to credit the scaled-FP8 tier as
**quantized-8bit** (not bf16), and print a separate "8-bit FP8 %" line so coverage reads
honestly (e.g. "92 % 4-bit ml8 + 8 % 8-bit FP8 + <0.2 % f32 core").

### S3 — C++ inference (`ggml-cuda`, `llama-graph`, `llama-model`)

**S3.1 Generic `ml8-aware mul_mat` helper + sidecar registry.** Add one helper —
`build_ml8_or_mul_mat(ctx, weight, x)` — that looks up `weight`'s sidecars
(centroids/rotation/awq) in a load-time registry (weight tensor → sidecar set) and
dispatches the ml8 path when present, else falls back to plain `ggml_mul_mat`. The
new matmul call sites (full-attn q/k/v/o, gated-deltanet qkv/gate/ssm_out, `eh_proj`,
`lm_head`) route through it. **Fallback on sidecar-absence = zero behaviour change for
every non-ml8 model and for any tensor we leave native.** Do **not** refactor the
working FFN/MoE explicit-field path in this pass (migrate later as cleanup).

**S3.2 Scaled-FP8 type support.**
- **`get_rows` case** for `GGML_TYPE_ML8_FP8` in `getrows.cu` — a float-style gather
  applying the per-group scale via the existing `ggml_cuda_ue4m3_to_fp32`. CPU mirror +
  type-traits registration.
- **Real FP8-WMMA `mul_mat`** for `ssm_alpha/beta`: the existing ml8 FP8-WMMA kernel
  **minus the centroid-LUT indirection** (the e4m3 values ARE the weights; apply
  per-group scale). Implement as a "no-LUT" mode sharing the ml8 accumulation machinery.
- **`to_float`** for the expand-at-load consumers (none required for the shipped map,
  since conv1d stays f32 — kept for completeness/CPU mul_mat).

**S3.3 Layer struct.** Prefer the registry (S3.1) over adding ~5 roles × 3 sidecar
fields to `llama_layer` (avoids struct bloat). If a minimal field set is cleaner for the
loader, scope it to the new roles only.

**S3.4 Two graph paths.** Both the full-attention build (q/k/v/output) and the
gated-deltanet build (qkv/gate + ssm_out, with α/β/conv feeding the SSM op) route their
GEMMs through the S3.1 helper; α/β use the S3.2 FP8 mul_mat; embed uses the S3.2
`get_rows`.

### S4 — Validation & sequencing

**Dense 9B first** (local, free, fast):
1. **Python↔C++ bit-equivalence** — reproduce the Python `Ml8Linear` reference PPL in
   the C++ graph path to ≥4 decimals on a fixed artifact+seed (the 8.2990-style
   determinism gate). >0.005 drift = a wiring/dispatch bug, not FP noise.
2. **Near-lossless gate** — Δ_PPL vs the bf16 9B ≤ **+0.08–0.10** (matches the 4B ml8
   precedent of +0.0834).
3. **Size** — total < UD-Q4_K_XL 9B (projected ~4.7 bpv / ~5.4 GB with FP8 embed).
4. **Coverage** — guardrail clears (≥85 % ml8; report 4-bit/8-bit/f32 split).
5. **Long-context probe** — a long-ctx eval (needle/long-ppl) specifically to catch
   SSM-gate compounding that short-ctx PPL averages out (the "equal PPL ≠ equal
   behaviour" risk).

**Then 35B MoE** — the same machinery adds non-expert (attn/ssm/lm_head/embed) coverage
on top of the already-working expert path; gate = MAD-256 both-axes (smaller AND lower
PPL than UD-Q4_K_XL).

### S5 — Testing

- **Unit:** role classifier; scaled-FP8 quant round-trip SNR; coverage metric incl the
  FP8 tier (4-bit/8-bit/f32 split + the FFN-only-dense refuse case still holds); FP8
  `mul_mat` kernel vs a dequant reference (bit-equivalence to the no-LUT math); FP8
  `get_rows` case vs reference gather.
- **Integration:** 4B/9B Python-`Ml8Linear` ↔ C++-graph equivalence smoke; the dense-9B
  PPL gate; the long-context probe.

---

## Out of scope (explicit)

- **ml8 (4-bit) embedding** — deferred lever (~0.4 bpv / ~540 MB more) needing a LUT-gather
  `get_rows` kernel + a no-Hessian embed codebook. Scaled-FP8 embed first.
- **Refactoring the existing FFN/MoE explicit-field wiring** to the generic helper —
  later cleanup; don't destabilize the validated 35B path now.
- **mlambaformer SSM-by-construction design** (state-norm / bounded gate parameterization
  so the gates go 4-bit) — separate architecture work; this spec only *characterizes* the
  retrofit's SSM-gate quant dynamics to feed it.
- **`requantize_nonexpert.py` wiring** — superseded for dense by this spec; remains a
  separate MoE-side size item only if A's MoE coverage is deferred.

## Risks

| Risk | Mitigation |
|---|---|
| Attention/lm_head at 4-bit costs PPL beyond the gate | Existing rotation/AWQ/GPTQ machinery (the reason 4-bit attn is reasonable); per-role `group_size` knob to bump just the offending role (esp. lm_head) without re-architecting; measure, don't assume |
| SSM-gate 8-bit error compounds over long context | 8-bit error ≈ 1/16th of 4-bit; explicit long-context probe gate; α/β/A/dt sensitivity instrument |
| New helper touches shared `build_attn` (all models) | Fallback keyed on sidecar-absence → provably zero change when no ml8 sidecars present; covered by a non-ml8-model regression test |
| Scaled-FP8 `mul_mat` (no-LUT) kernel bug | Bit-equivalence unit test vs dequant reference; reuses validated ml8 accumulation path |
| Coverage metric mislabels 8-bit FP8 as bf16 | S2.3 explicit fix + test |
```
