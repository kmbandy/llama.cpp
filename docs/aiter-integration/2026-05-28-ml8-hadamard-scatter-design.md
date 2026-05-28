# ML8 QuaRot-R1 Hadamard Scatter — Design

**MAD-244 follow-up (task #104).** Close the +0.046 PPL gap between
ml8_4_soa and Q4_K_XL on Qwen3.6-35B-A3B by rotating the residual stream
into a basis that disperses outliers before ml8 quantization runs.

## Problem

The 2026-05-28 ml8_4_soa run on Qwen3.6-35B-A3B landed at
PPL = 5.7968 ± 0.0360 @ ctx=4096, vs Q4_K_XL 5.7507. The +0.046 gap
points at calibration-recipe headroom, not at the storage layout (SOA
matches AOS bit-for-bit) and not at the runtime (rotation 3D bug fixed,
gather semantics verified).

Per-kind Y_SNR on the same run: gate 24.28 dB / up 23.93 dB / down 21.69
dB. The down-proj output dim is the worst layer by ~2.3 dB. That
signature — SwiGLU pushing outliers into the residual basis through
down_proj — is exactly what QuaRot's R1 (residual-stream rotation) was
designed to fix.

The existing per-linear input rotation (`--rotation kronecker`) only
randomizes the basis _inside_ a single linear's input dim; it does not
align across the residual stream. The outliers that down_proj produces
land back in the residual basis and propagate, where they hurt the next
layer's gate/up quantization.

## Goal

Apply a single model-global random Hadamard rotation `R_resid` of size
`d_model` to the residual stream, folded entirely into weights at
calibration time. Every linear that reads or writes the residual gets
modified so that the model is mathematically equivalent before
quantization, but the weights present a less-coherent (more
incoherent-with-outliers) target to the ml8 quantizer.

Non-goals:

- Runtime R1 application — R1 is fully absorbed into weights, kernel
  sees nothing new.
- AWQ scan, act_order, R3/R4 online Hadamard, KV-side R2 — separate
  levers, separate specs.
- GGUF schema changes — same tensor names, same shapes, same dtype.
- Architectures other than Qwen3.5 (dense) and Qwen3.6 (MoE+Mamba
  hybrid). Pattern table extends per-arch as needed.

## Architecture

```
bf16 GGUF (source)        rotate_model_quarot.py             rotated bf16 GGUF        calibrate_ml8_paged.py
  Qwen3.6-35B-A3B-bf16  ──► stream tensors one at a time  ──►  same shapes,         ──► unchanged
  (or 4B-bf16)              rotate in CPU/GPU buffer,         rotated weights         (paged ingest of
                            write to output GGUF,              + γ-absorbed norms       rotated GGUF)
                            free buffer
```

R1 is a pure preprocessing pass: in → out, both bf16 GGUFs of identical
size. Calibration is rotation-blind.

### Inputs

- `--source` path to source bf16 GGUF
- `--output` path for rotated bf16 GGUF
- `--seed` (default 42)
- `--device cuda:N` for matmul speed; CPU fallback works for the 4B size

### Outputs

- Rotated bf16 GGUF — drop-in for `calibrate_ml8_paged.py --gguf`. Same
  tensor names, shapes, dtype. Total file size unchanged.
- `<output>.quarot_r1.json` sidecar — `{seed, d_model, sign_flip_b64,
  rotated_tensor_names[], absorbed_norm_names[], arch}`. Calibration
  ignores it. Kept for audit + future GGUF embedding.

### Memory pattern (paged-equivalent)

- `GGUFReader` mmaps the source; per-tensor `data` view is mmap-backed.
- Pass 1 pulls only the RMSNorm γ vectors (`d_model × 2 bytes × ~120 γs
  ≈ 480 KB`) into RAM, builds the role roster.
- Pass 2 streams tensors one at a time: load → rotate in fp32 on the
  chosen device → cast back to bf16 → write to `GGUFWriter` → free.
- Peak host RAM ≈ one MoE expert tensor (~150 MB for 35B-A3B's largest)
  + `R_resid` (16 MB at d_model=2048).
- Matmul cost is negligible vs the file I/O; one R-multiply per residual
  linear, batched along the expert axis for MoE tensors.

## Math

### R_resid construction

`R_resid = D ⊙ H_sylvester(d_model)`:

- `H_sylvester(d_model)` is the deterministic normalized Sylvester
  Hadamard. For `d_model = 2048` (Qwen3.6-35B-A3B) it's a pure power of
  2. For `d_model = 2560` (Qwen3.5-4B) it factors as
  `H_5_random ⊗ H_512_sylvester` using the existing
  `kronecker_rotation.factor_for_dim(2560, max_b=1024)` machinery.
- `D` is `diag(±1)` chosen from `Bernoulli(0.5)` seeded by `--seed`.
  Breaks the structural axis-alignment of pure Sylvester, gives
  incoherence with high probability against any fixed outlier
  direction.
- `R_resid` is orthogonal → RMSNorm scale is invariant → equivalence
  holds.

### Rotation table

Row-vector convention throughout. Shapes below are PyTorch `[out, in]`
weight shapes (forward is `y = x @ W.T`). GGUF stores tensor dims as
`[K, N, ...]` with K=input, N=output, which is the transpose of the
shapes shown here — the implementation maps the rotation axis via the
role enum, not the shape, so the convention difference is non-load-bearing
for the math but matters for the reshape calls in code.

| Tensor | Operation | Reason |
|---|---|---|
| `token_embd.weight` `[vocab, d_model]` | output rot: `W ← W @ R_resid` | embed output IS the residual |
| `blk.L.attn_norm.weight` (γ) | absorb into Q/K/V, then `γ ← 1` | γ elementwise, doesn't commute with R |
| `blk.L.attn_q.weight` `[head*d_head, d_model]` | input rot + γ absorption | reads residual |
| `blk.L.attn_k.weight` `[kv*d_head, d_model]` | input rot + γ absorption | reads residual |
| `blk.L.attn_v.weight` `[kv*d_head, d_model]` | input rot + γ absorption | reads residual |
| `blk.L.attn_output.weight` `[d_model, head*d_head]` | output rot: rotate output axis | writes residual |
| `blk.L.ffn_norm.weight` (γ) | absorb into gate/up + router, then `γ ← 1` | same as attn_norm |
| `blk.L.ffn_gate_inp.weight` (router) `[n_experts, d_model]` | input rot + γ absorption | reads pre-norm residual |
| `blk.L.ffn_gate_exps.weight` `[d_ffn, d_model, n_exp]` | per-expert input rot + γ absorption | MoE reads residual |
| `blk.L.ffn_up_exps.weight` `[d_ffn, d_model, n_exp]` | per-expert input rot + γ absorption | MoE reads residual |
| `blk.L.ffn_down_exps.weight` `[d_model, d_ffn, n_exp]` | per-expert output rot | MoE writes residual |
| `blk.L.ssm_norm.weight` (γ, Mamba) | absorb into Mamba in_proj | same pattern |
| `blk.L.ssm_in.weight` (Mamba in_proj) | input rot + γ absorption | reads residual |
| `blk.L.ssm_out.weight` (Mamba out_proj) | output rot | writes residual |
| `output_norm.weight` (γ) | absorb into LM head | final norm |
| `output.weight` (LM head) `[vocab, d_model]` | input rot + γ absorption | reads final residual |

Mamba block internals (SSM state, conv1d, dt_proj, A_log, D) operate in
a basis _internal_ to the Mamba block. They don't read or write the
residual directly, so they don't rotate.

### Input-side rotation

For a linear with `W` shape `[N, d_model]` (in_features = d_model):

```python
# γ absorption (γ shape: [d_model])
W = W * gamma.unsqueeze(0)         # column-wise scale
# R_resid input rotation
W = W @ R_resid.T                  # rotate input axis
```

### Output-side rotation

For a linear with `W` shape `[d_model, in_features]`:

```python
# R_resid output rotation
W = R_resid @ W                    # rotate output axis
```

Output-side rotation has no γ to absorb (the γ that gates _into_ this
linear was absorbed at the input side; the γ that gates _the residual
this linear writes into_ lives in the next block's norm and gets
absorbed there).

### MoE batching

Per-expert tensors are 3D `[*, *, n_experts]`. Reshape to 2D for one
batched matmul:

- Input-side: reshape `[d_ffn, d_model, n_exp] → [d_ffn * n_exp, d_model]`,
  apply `W @ R_resid.T`, reshape back.
- Output-side: reshape `[d_model, d_ffn, n_exp] → [d_model, d_ffn * n_exp]`,
  apply `R_resid @ W`, reshape back.

One matmul per tensor, not 256.

### Equivalence invariant

For any input token IDs:

```python
logits(rotated_gguf, x) ≈ logits(source_gguf, x)
```

up to fp32 round-trip noise (∼1e-3 max relative diff). This is the
testable gate before any calibration begins.

## Calibration interaction

The existing per-linear `--rotation kronecker` flag composes with R1 on
the input side. Three options:

1. **Disable per-linear rotation when input is rotated.** Pass
   `--rotation none`. Simplest, lets us measure R1's effect cleanly.
2. **Compose at calibration time** — pre-multiply `Q_effective =
   R_resid @ Q_layer`, store in blob.
3. **Independent layers** — both rotations stored, runtime applies in
   sequence. Requires GGUF schema work.

Pick option 1 for the prototype. Fall back to option 2 if PPL needs
extra rotation diversity. Option 3 is overkill for the gap we're
closing.

## Implementation

### `rotate_model_quarot.py` (new, ~300 lines)

Standalone CLI. Pass 1 indexes tensors and pulls γ vectors. Pass 2
streams source → rotate → write. Role mapping via per-arch regex
patterns in `_ROLE_PATTERNS[arch]` dict; unknown tensor name → fail
loudly with the name printed. No HF dependency.

Sub-helpers:

- `_build_R_resid(d_model, seed, device) → torch.Tensor` (uses
  `kronecker_rotation.sylvester` + `kronecker_rotation.factor_for_dim`)
- `_rotate_input_side(W, gamma, R_resid)` and
  `_rotate_output_side(W, R_resid)` (pure functions, unit-testable)
- `_classify_tensor(name, arch) → Role` (regex lookup, unit-testable)

### `test_rotate_model_quarot.py` (new, ~150 lines)

- Property: orthogonality `R @ R.T = I` to fp32 tol.
- Property: input/output rotation cancellation —
  `x @ R @ R.T @ W.T == x @ W.T`.
- Property: γ absorption equivalence —
  `(γ * x) @ W.T == x @ (W * γ.unsqueeze(0)).T`.
- End-to-end: rotate a 2-layer toy GGUF, run forward, assert equivalence
  to source.

### Equivalence gate script (new, ~80 lines)

Reuses `llama-cli` (subprocess) with `--no-mmap` (mad-lab standing
rule). Encodes 512-token prompt from wikitext-2 val, dumps final logits
on both GGUFs, compares max relative diff. Also runs a 4K-token
perplexity slice on both, asserts equal to ±0.005.

### `calibrate_ml8_paged.py` (unchanged)

Source GGUF arg points at rotated file. `--rotation none` for option 1
behavior.

## Validation

**Phase 1: equivalence gate** — must pass on both 4B and 35B-A3B before
proceeding. Max relative logit diff < 1e-3; 4K PPL slice matches ±0.005.

**Phase 2: 4B calibration** — fast iteration loop. Recipe matches the
2026-05-24 TRUE-ML8 Cell C run. Baseline Δ_PPL = +0.0834. Gate
Δ_PPL ≤ +0.04 (~50% gap close). Hard fail Δ_PPL > +0.0834. Wall time
~15 min on R9700. Time-based Monitor on `Y_SNR per kind` and
`perplexity:.*\[` lines.

**Phase 3: 35B-A3B calibration** — only after Phase 2 hits the gate.
Recipe matches the 5.7968 run, rotated GGUF as source. Baseline PPL =
5.7968. Goal PPL ≤ 5.770 (closes most of +0.046 gap vs Q4_K_XL 5.7507).
Wall time ~3 h calibration + ~30 min PPL. Time-based Monitor with
failure-signature regex `Traceback|Error|FAILED|Killed|OOM`.

**Phase 4: diagnostic readout** — per-kind Y_SNR. If down-proj jumped
significantly from 21.69 dB, R1 worked structurally; remaining gap is a
different lever. If 4B improved but 35B regressed, document MoE-specific
interaction and file followup.

## Out of scope

- AWQ scan (`n_awq > 0`)
- `act_order` GPTQ flag
- Online R3/R4 rotations (SwiGLU intermediate, would need runtime FWHT
  kernel)
- KV-side R2 rotation
- Per-linear rotation composition with R_resid (Section 4 option 2)
  unless option 1 underperforms
- GGUF schema changes — R1 absorbed into weights, runtime untouched
- Architectures beyond Qwen3.5 / Qwen3.6

## Related

- [docs/aiter-integration/2026-05-28-ml8-moe-soa-design.md] — yesterday's
  SOA design, this work runs on its output
- KG `30636e13` — rotation 3D bug fix (the n_tokens correction that
  unblocked 35B-A3B PPL)
- KG `a293ba94` — followup priorities project; this spec implements
  priority 1
- KG `132334f5` — full session summary including the 5.7968 baseline
