# MAD-281 Phase E — Full-`H` GPTQ-Owned Axis B Reassignment (Design)

**Status:** design, approved by kmbandy 2026-06-12
**Supersedes:** the diagonal-curvature `pv_vstep` reassignment (Phase B/D) for Axis B
**Epic:** MAD-281 (fp8-substrate QAT + lattice-pluggable ml-k precision ladder)

---

## 1. Goal

Replace the diagonal-curvature, parallel index-flip Axis B (`pv_vstep`) with index
reassignment that models the **actual correlated shape** of the weight space — the
full activation Hessian `H = XᵀX` — by reusing mad-lab's hardened GPTQ implementation.
One sentence: **Axis A (gradient/QAT) owns the centroids; Axis B (full-`H` GPTQ) owns
the index assignment.**

## 2. Why (the falsified approach)

The Phase B/D Axis B used only the **diagonal** curvature `h_k = E[x_k²] = H_kk` in a
quadratic flip criterion `ΔL(j) ≈ g·Δw + ½·h·Δw²`, applied to millions of weights **in
parallel**. Measured on Qwen3.5-0.8B (smoke `smoke_fp8_qat.py`, bf16 teacher):

| frac | original linear pv (step 10) | curvature + trust-region pv (D.2v3) |
|------|------------------------------|--------------------------------------|
| 1e-1 | 15.9                         | 4.30                                 |
| 1e-3 | 12.8                         | 0.92 (final 3.49)                    |

The curvature + value-rank trust region cut divergence ~14× but **never reached the
0.0514 Axis-A (frozen) floor at any frac** — KL ∝ flip count, stable only as flips → 0
(i.e. inert). Root cause (real-model diagnostic `diag_pv_curvature.py`): the diagonal
throws away the **off-diagonal `H_jk`** = input-channel correlations. Parallel flips of
correlated weights under an axis-aligned model let the neglected cross-terms accumulate
→ divergence. The correct fix is **sequential, full-`H`, `H⁻¹`-compensated** assignment
— i.e. OBQ/GPTQ, which the near-lossless frontier (QuIP#/AQLM/OmniQuant) all use and
which mad-lab already ships in `batched_gptq.py`.

## 3. Architecture

Phase E is **block-coordinate descent** over two coupled proxies:

- **Axis A (continuous, gradient/KL):** Adam-tune the fp32 centroids (+scales) against
  the top-K KL distillation loss. Existing, working, the proven win.
- **Axis B (discrete, full-`H` GPTQ):** given the *current* (Axis-A-tuned) centroids as a
  **fixed** quantization grid, re-solve the per-element index assignment to minimize the
  `H`-weighted reconstruction of the original weight `W_orig`, with `H⁻¹` compensation.

These optimize different proxies (B: `H`-weighted reconstruction of `W_orig`; A: end-to-end
KL). Reconstruction is the classic second-order proxy for layer output error — it
correlates with KL but is not identical. Phase E is therefore alternation over two coupled
proxies, **not** one exact objective; the empirical gate decides whether the alternation
compounds.

The base `Qwen3.5-0.8B-ml8.gguf` indices are **already** full-`H` GPTQ — but assigned
against the *original* Lloyd-Max centroids. Once Axis A moves the centroids, those indices
are **stale** (GPTQ-optimal for a codebook that no longer exists). Re-solving them against
the tuned codebook is the value-add.

## 4. Core component — `batched_gptq_reassign`

The only substantial new code. A focused carve-out of `batched_gptq_quantize`
(`batched_gptq.py`) that **skips the Lloyd-Max centroid fit** and treats the centroids as
a fixed grid:

```
batched_gptq_reassign(
    W_stack,        # [E, N, K] fp32 — the reconstruction target (W_orig)
    H_stack,        # [E, K, K] fp32 — activation Hessian
    centroids,      # [E, n_groups_k, n_centroids] fp32 — FIXED (Axis-A-tuned) grid, sorted
    scales,         # [E, N, n_groups_k] fp32 — FIXED per-(row,group) scale
    *, group_size, percdamp=0.05, act_order=False,
) -> indices        # [E, N, K] int (0..n_centroids-1)
```

- Reuses `_cholesky_inv_upper` (percdamp escalation, the hardened Cholesky-`H⁻¹` path) and
  the existing batched per-column GPTQ update (outer-product compensation over the
  not-yet-processed columns).
- Per column: pick the nearest **fixed** centroid (in the per-group grid) for `W_q`, then
  propagate the error `(W − W_q)` to the remaining columns via the `H⁻¹` row — identical to
  the existing loop, only the "choose `W_q`" step changes from "Lloyd-Max grid" to "this
  fixed grid".
- For the dense (single-expert) case `E = 1`. The MoE batched axis is preserved for free.

**Equivalence anchor (test):** with `centroids/scales` set to the values `batched_gptq_quantize`
itself fits, `batched_gptq_reassign` must return **bit-identical** indices to
`batched_gptq_quantize`'s index output. This proves the carve-out didn't change the GPTQ math.

## 5. Hessian source

Reuse the **static offline `H = XᵀX`** infrastructure that already made the GGUF
(`compute_hessian` / `_collect_hessians_layer_moe`). Activations shift only slightly under
centroid tuning, so a static `H` (computed once on the calib corpus) is a sound, cheap,
proven choice. **Online `xᵀx` accumulation in the fp8 forward is explicitly deferred** to a
later rung if and only if rung B shows the static `H` is the limiter. (YAGNI.)

Practical: if the 0.8B `H` is not cached, recompute it once (one calib forward) and cache it
keyed to model/gguf/corpus, matching the existing cache discipline.

## 6. The two rungs (sequenced A → B)

### Rung A — single post-Axis-A re-solve (cheap, de-risks the whole idea)
1. Run **frozen** Axis A to convergence (centroids → ~0.0514, indices frozen).
2. Do **one** `batched_gptq_reassign(W_orig, H, tuned_centroids, scales)` per target.
3. Copy the new indices in; re-measure holdout KL.
- **Question answered:** does re-solving the now-stale indices against the tuned codebook
  beat the 0.0514 Axis-A floor?
- Reuses the `smoke_fp8_qat.py` harness; adds a `gptq` arm (one reassign at the end of the
  Axis-A run).

### Rung B — interleaved block-coordinate descent (only if A earns it)
- Alternate [Axis A: tune centroids `N` steps] ↔ [Axis B: `batched_gptq_reassign`], for the
  duration of training. `N` (the cadence) is a hyperparameter, swept empirically.
- Same primitive as A, called periodically. No new core code beyond a cadence knob.

## 7. Data flow

```
W_orig  (AttachedTarget buffer, present)        ─┐
H = XᵀX (static, offline infra)                  ├─► batched_gptq_reassign ─► new indices ─► AttachedTarget.indices
centroids/scales (Axis-A-tuned, live)           ─┘
```

No change to the fp8 forward/backward for rung A (the diagonal-`h` stash from D.1 becomes
unused for Axis B and may be left in place or removed in cleanup). `Ml8Fp8Fn.last_dLdW` /
`last_h` are no longer consumed by Axis B.

## 8. Testing

- **Unit (CPU/TDD):** `batched_gptq_reassign` bit-identical to `batched_gptq_quantize`'s
  index output when fed that function's own fitted centroids/scales (the equivalence anchor).
- **Unit (CPU/TDD):** a constructed small case where a stale assignment (optimal for old
  centroids) is measurably improved by a re-solve against shifted centroids — assert
  reconstruction error `‖W_orig − Wq‖²_H` strictly decreases.
- **Integration (GPU, supervised):** the `gptq` arm in `smoke_fp8_qat.py` (rung A). Gate:
  stable AND ideally final KL **< 0.0514** (the Axis-A floor) → Axis B finally earns its keep.
- All runs under the RAM-safe SOP (oom_score_adj=600, systemd-run MemoryMax=11G), time-based
  Monitor, single model resident.

## 9. Scope / non-goals

- **In:** `batched_gptq_reassign` primitive; rung A arm in the smoke; static-`H` reuse;
  rung B interleave + cadence knob.
- **Out (deferred):** online `xᵀx` accumulation; re-emitting a QAT'd GGUF + full-vocab
  `llama-perplexity --kl-divergence` vs the bf16 parent (the real UD comparison); throughput
  (C.4); applying Phase E to larger / different-arch models (4B, 35B-A3B MoE) — that is the
  separate "does this scale" experiment, tracked apart from this build.

## 10. Success criteria

1. `batched_gptq_reassign` proven equivalent to the hardened GPTQ index math (bit-identical anchor).
2. Rung A produces a **stable** holdout-KL number on 0.8B (no divergence — the diagonal-pv
   failure mode is gone by construction, since GPTQ is sequential + `H⁻¹`-compensated).
3. A clear verdict: does full-`H` reassignment beat the 0.0514 Axis-A floor? Either outcome is
   a publishable result — a win validates the two-axis QAT product; a null result rigorously
   bounds how much discrete reassignment can add over centroid tuning on this codebook.
