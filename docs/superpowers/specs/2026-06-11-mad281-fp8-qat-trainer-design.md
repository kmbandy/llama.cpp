# MAD-281 — fp8 + joint-discrete QAT trainer (design)

**Date:** 2026-06-11
**Status:** design (brainstorm output; pre-plan)
**Supersedes for forward direction:** the act-replay light codebook-FT trainer (MAD-283) — that
trainer becomes the bf16/frozen-index baseline this method must beat.

## Goal

Build the **real** quantization-aware training (QAT) method for ml8: a trainer whose student matmuls
run in **fp8 in both the forward and backward GEMMs**, and which jointly optimizes **both** the
continuous codebook (centroids + scales) **and** the discrete index assignment (PV-tuning-style),
against a KL-to-bf16-teacher objective. This is the frontier-grade lever (the family QuIP#/AQLM/
PV-tuning belong to), not the light frozen-index codebook FT we already have.

This realizes the committed strategic direction "fp8 as the universal substrate" (KG `38e7ff7d`):
the training forward **is** the deployed inference kernel, so we train in deployed numerics by
construction.

## Why (one paragraph)

The light lever (frozen-index codebook FT, bf16 backward) underwhelmed on a 4B hint run — but that is
the *weakest* rung of the ladder, and a weak signal from the weak lever argues *for* the heavy lever,
not against it. The near-lossless low-bit frontier is reached only by **jointly** moving the codebook
*and* the discrete assignment. Doing this in fp8 (a) is the product's portability/segmentation moat
(fp8 runs on all pre-Blackwell HW + NVIDIA's own H100/H200 fleet; fp4 is Blackwell-only), and (b)
makes the QAT forward byte-identical to the inference kernel, collapsing train/infer skew. The
capability probe is green on the R9700/gfx1201 (`torch._scaled_mm` + Triton `tl.dot` work for the fp8
combos, no rebuilds — KG `417a2165`), and the forward LUT GEMM already exists (`gemm_ml8.py`
`WEIGHT_FORMAT=1`). The genuinely new build is the **backward**.

## Non-goals / scope boundaries

- **Teacher stays the bf16 version of the *same* model.** The objective is lossless *compression*
  (be indistinguishable from bf16-self), so the teacher is definitionally bf16-of-this-model. A
  larger/smarter teacher (capability transfer) is a different project with a different,
  no-longer-"lossless" success criterion and is explicitly out of scope.
- **No C++/HIP kernel changes.** The forward Triton kernel (`gemm_ml8.py`) already carries the ml8
  LUT branch and is launchable from PyTorch; the backward GEMMs reuse the same Triton `a8w8` path
  (`WEIGHT_FORMAT=0`) or `torch._scaled_mm`. This build is Python-side.
- **Validation is on small models (0.8B, then 4B).** Scaling the *same* method to 27B/35B is a
  downstream consumer ("runs next week"), not part of this method build. (Running it at scale is
  running it — not a deferred-correctness fast-follow.)
- **Master weights + Adam moments stay fp32.** Accumulating a tiny update into ~3-mantissa-bit fp8
  rounds to zero → divergence. This high-precision spine is universal physics (NVIDIA's fp4 training
  keeps the same spine), not a shortfall.

## Architecture — two orthogonal axes

```
                      ┌─────────────────────────── Axis A: fp8 engine ───────────────────────────┐
ids → student forward: per ml8 linear  x(bf16) ─[fp8_quant e4m3, per-row]→ x_fp8
                                       Ml8Fp8Fn.forward = deployed LUT kernel (WEIGHT_FORMAT=1) → y(bf16)
                       … rest of model (attn/SSM scan) in bf16 …                → logits
   KL(logits, teacher_topK) × loss_scale → backward:
                       per ml8 linear  Ml8Fp8Fn.backward:
                           dy ─[/loss_scale, fp8_quant e5m2]→ dy_fp8
                           dx     = (dy_fp8 @ W_e4m3ᵀ)·scale      ← a8w8, exact e4m3 weight operand
                           dW_raw = (dy_fp8ᵀ @ x_fp8)             ← a8w8           ┐ produces dL/dW_raw
                           dcent  = scatter_add(dW_raw by index); dscale = Σ…     ┘ (Axis B taps this)
   optimizer (fp32 master centroids/scales).step()
                      └──────────────────────────────────────────────────────────────────────────┘

                      ┌──────────────── Axis B: joint discrete optimization ─────────────────┐
   every N opt-steps: index_reassign(mode):
       mse_estep : re-solve each index vs ORIGINAL bf16 weight (Hessian/MSE).  baseline + init.
       pv_vstep  : per-element linearized flip, ΔL ≈ (dL/dW_raw)·ΔW, ΔW=(cent[j]-cent[i])·scale;
                   apply top-K flips by predicted improvement (trust region). consumes Axis A's dL/dW_raw.
       indices live as a MUTABLE BUFFER the fp8 Fn consumes as input → reassignment never touches GEMMs.
                      └──────────────────────────────────────────────────────────────────────┘
```

**The two axes are orthogonal but composable:** Axis A produces `dL/dW_raw` as a backward
by-product; Axis B's `pv_vstep` is a natural consumer of exactly that signal. Neither needs the other
to exist (A can run with indices frozen; B can run with bf16 numerics), which is what makes them
independently testable.

## Components

### Axis A — fp8 engine

1. **`fp8_quant(t, fmt, axis) -> (t_fp8, scale)`** — per-row (activations/grads) or per-group
   (handled by caller) amax → `scale = amax / FP8_MAX` → cast. `e4m3` for activations/weights,
   `e5m2` for gradients. Just-in-time amax (delayed-scaling is a later optimization — YAGNI).
   Guards `amax == 0` (→ scale 1, zero tensor passes through).

2. **`Ml8Fp8Fn(torch.autograd.Function)`** — the heart.
   - `forward(ctx, x_fp8, x_scale, centroids, scales, packed_idx, gidx)`: the kernel's centroid
     operand is `snap_to_e4m3(centroids)` — i.e. the existing `weight()` STE boundary still applies:
     the **master `centroids` are fp32** (what the optimizer updates), but the value fed to the LUT
     kernel (and reused in backward) is the **snapped e4m3** lattice point, with the gradient flowing
     straight through to the fp32 master. Launches the deployed LUT kernel (`WEIGHT_FORMAT=1`),
     returns bf16. Saves `x_fp8`, the snapped-e4m3 centroids, `scales`, `packed_idx`, `gidx` for
     backward. **This forward is the inference forward** (bit-equivalent to the deployed path).
   - `backward(ctx, dy)`: `dy_fp8, s_dy = fp8_quant(dy/loss_scale, e5m2)`; `dx = (dy_fp8 @ W_e4m3ᵀ)`
     with per-group `scale` applied outside the dot (exact: centroids *are* e4m3 lattice points);
     `dW_raw = (dy_fp8ᵀ @ x_fp8)` (also a8w8). Then chain `dW_raw → dcentroids` (scatter-add over
     index) and `→ dscales`. Returns grads for `centroids`, `scales`; `dx` for the upstream; `None`
     for `packed_idx`/`gidx` (non-differentiable). **Exposes `dL/dW_raw`** to the trainer (saved on
     `ctx` or returned via a side-channel) for Axis B's `pv_vstep`.

3. **`Ml8Fp8RefFn`** — `torch._scaled_mm`-based fwd+bwd reference (Approach 2), **test-only**: an
   independent fp8 ground truth to gradcheck `Ml8Fp8Fn`'s hand-written backward against (tight rel
   tol, same fp8 numerics). Not the product (its forward is dequant→`_scaled_mm`, not the LUT kernel).

### Axis B — joint discrete optimization

4. **`index_reassign(target, mode, dLdW=None, frac=F)`** — one interface, two implementations,
   mutates the index buffer in place:
   - **`mse_estep`**: re-solve each weight element's index = `argmin_j ‖W_orig − cent[j]·scale‖²`
     (optionally Hessian-weighted), against the **original bf16 weight** kept resident as the anchor.
     The deterministic baseline + the initializer the gradient step must beat.
   - **`pv_vstep`** (adapted PV-tuning): per-element predicted loss change for flipping index `i→j`
     is `ΔL(j) ≈ dLdW · (cent[j] − cent[i]) · scale`. Because the term is *linear* in `cent[j]`, the
     best `j` is an `argmin` over the codebook of a linear function (cheap, vectorizable). Apply only
     the **top-`frac` flips by predicted improvement** per call (trust region); re-linearize next call.
     Consumes `dL/dW_raw` from Axis A's backward.
   - Indices are a **non-differentiable mutable buffer** on the `AttachedTarget`; the fp8 Fn reads it
     each forward, so a reassignment between optimizer steps changes the next forward with no GEMM
     change.

### Integration

5. **`act_replay_student.attach_to_linear(..., fp8=True)`** — routes the forward through `Ml8Fp8Fn`
   (replacing the current `F.linear(x, STE_weight)` path). The existing bf16-STE path stays as the
   `fp8=False` fallback/reference. Keeps the original bf16 weight available (anchor for `mse_estep`).

6. **`act_replay.py`** new flags: `--fp8`, `--loss-scale`, `--grad-fmt {e5m2,e4m3}`,
   `--reassign {none,mse,pv}`, `--reassign-interval N`, `--reassign-frac F`,
   `--lr-warmup-steps` + cosine decay (carried over from the MAD-283 trainer-fix work — the QAT loop
   still needs a sane lr schedule). Teacher / KL loss / holdout / GGUF export are **unchanged**.

## Data flow (one optimizer step, both axes on)

1. For each ml8 linear: `x → fp8_quant(e4m3) → Ml8Fp8Fn.forward` (LUT kernel) → `y`. Rest of model bf16.
2. `loss = KL(logits, teacher_topK) · loss_scale`; `loss.backward()`.
3. Per ml8 linear backward: produces `dx`, `dcentroids`, `dscales`, and stashes `dL/dW_raw`.
4. `optimizer.step()` updates fp32 master centroids/scales; lr per warmup+cosine schedule.
5. If `step % reassign_interval == 0`: `index_reassign(mode)` mutates index buffers
   (`pv_vstep` uses the stashed `dL/dW_raw`; `mse_estep` uses the resident bf16 anchor).
6. Periodic holdout KL eval + checkpoint (existing machinery).

## Error handling / numerical concerns

- **fp8 range:** per-row amax scaling for x and dy; static **loss-scale** keeps the small KL gradient
  in e5m2 range (KL loss is O(0.1) → grads can underflow e5m2). Static scale to start (a CLI flag);
  dynamic loss scaling is a later optimization. Guard `amax==0`.
- **Backward exactness:** the weight operand in both backward GEMMs is the **e4m3 centroid** gathered
  by index with per-group `scale` applied outside the dot — lossless, because the centroids already
  *are* e4m3 lattice points (this is why the backward is "plain a8w8").
- **`scatter_add` non-determinism** (GPU atomics) in `dW_raw → dcentroids`: acceptable for training;
  for the gradcheck tests use deterministic mode / CPU so the oracle comparison is exact-ish.
- **Trust region for `pv_vstep`:** the linearization is only locally valid; capping flips per call +
  validating against held-out KL each reassign round is the guardrail. A V-step round that does not
  reduce held-out KL is rejected/rolled back (or `frac` annealed).
- **Master-weight spine:** centroids/scales fp32, Adam moments fp32 — never fp8.

## Testing strategy (TDD; each axis test-isolated, then composed)

**Axis A (indices held fixed):**
- `fp8_quant` roundtrip + edge cases (zero, near-`FP8_MAX` clamp, e4m3 vs e5m2).
- `Ml8Fp8Fn.forward` == deployed LUT-kernel output on a tiny matmul (bit-equivalence to inference).
- `Ml8Fp8Fn.backward` (`dx`, `dcentroids`, `dscales`) vs `Ml8Fp8RefFn` oracle (tight rel tol) **and**
  vs `torch.autograd.gradcheck` on an fp32 shadow (loose/structural).
- `dW_raw → dcentroids/dscales` scatter chain vs autograd through the existing `weight()` STE.
- Tiny `StubLM` fp8-mode: a few steps → KL descends, grads finite, no NaN.

**Axis B (numerics held at bf16):**
- `mse_estep` reduces `‖W_orig − W_q‖²` (assignment is optimal per element).
- `pv_vstep` predicted ΔL matches measured ΔL on a tiny case (linearization correctness);
  a V-step round reduces a synthetic loss it was told to reduce.
- Index buffer mutation does not change tensor shapes / breaks nothing downstream.

**Composed (GPU smoke):**
- 0.8B fp8-mode + `--reassign pv`: step-0 KL sanity, short run, holdout KL trajectory, throughput vs
  bf16 backward.

## Quality gates / success criteria (the three rungs)

On a small model's **held-out KL** (apples-to-apples fixed holdout), each rung must beat the prior to
justify its complexity:

1. `frozen` (fp8 engine, indices frozen) — control.
2. `+ mse_estep` — does cheap reassignment beat frozen?
3. `+ pv_vstep` — does gradient-aware reassignment beat `mse`?

Ultimately measured against **UD-Q4_K_XL** and the **nvfp4** reference (the real "gunfight"). Note the
payoff is **rung-dependent**: index reassignment buys most at **ml8-3** (8 centroids, coarse Voronoi)
and shrinks toward ml8-5/8 — good, because the low rungs are exactly where we fight nvfp4.

## File structure

- **Create** `scripts/calibration/fp8_qat.py` — `fp8_quant`, `Ml8Fp8Fn`, `Ml8Fp8RefFn`. Axis A.
- **Create** `scripts/calibration/index_reassign.py` — `mse_estep`, `pv_vstep`, `index_reassign`. Axis B.
- **Create** `scripts/calibration/test_fp8_qat.py` — Axis A unit tests + oracle gradcheck.
- **Create** `scripts/calibration/test_index_reassign.py` — Axis B unit tests.
- **Modify** `scripts/calibration/act_replay_student.py` — `attach_to_linear(..., fp8=True)`; index
  buffer + original-weight anchor on `AttachedTarget`.
- **Modify** `scripts/calibration/act_replay.py` — new CLI flags; warmup+cosine schedule; wire fp8 +
  reassign into the loop; stash `dL/dW_raw` for `pv_vstep`.
- **Reuse** `gemm_ml8.py` (forward LUT kernel + a8w8 backward), `centroid_quantizer.snap_to_e4m3`,
  `kl_loss.{kl_topk,topk_teacher}`, `teacher_source` (bf16 teacher, cache), holdout/ckpt/export.

## Open questions / risks (carry into the plan)

- **fp8-precision `dL/dW` for the V-step:** e5m2 grads are noisy; is the flip *ranking* robust to
  that noise? (Likely yes — argmin is robust — but validate; `mse_estep` is the fallback if not.)
- **Loss-scale tuning:** static value vs dynamic; pick via the small-model smoke.
- **Reassignment cadence/`frac` schedule:** tuning, not architecture; start conservative.
- **Adapting PV-tuning to a KL (not reconstruction) objective** is an adaptation, not a replication —
  the `mse`-baseline rung exists precisely to prove the gradient V-step earns its keep.
- **Throughput:** fp8 backward speedup vs bf16 is expected but unmeasured on gfx1201 — the smoke
  measures it; not a correctness gate.
