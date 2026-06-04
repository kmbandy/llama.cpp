# Calibration Pipeline Speedup — Phase 1 Findings (measured)

**Date:** 2026-06-04 (MAD-256). **Source:** one instrumented 21k-token dense
calibration on the R9700 (gfx1201), `--phase-timing --forward-dtype-probe 8`.
Run dir: `/home/kmbandy/models/phase1/smoke_b20966/`. Branch `calib-pipeline-speedup`.
Model: Qwen3.5-0.8B dense, `--dense-coverage full --faithful-acts --faithful-weights`,
`ML8_DETERMINISTIC=1`, A3 tier override.

This is the **Step 1 gate** of the spec
(`docs/superpowers/specs/2026-06-03-calibration-pipeline-speedup-design.md`).
No Phase-2 lever was committed until these numbers returned.

> **CORRECTION (2026-06-04, post code-review).** The phase split, per-target
> seconds, and dtype ratio below are all MEASURED and stand. But the *interpretation*
> "102× redundant re-forward → collapse to 1, bit-identical" was WRONG. The dense
> per-target loop is **true-sequential GPTQ**: after quantizing target *k* it writes
> the quantized weight back (`weight_override` on the PagedLinear / `weight.data`
> resident) so target *k+1*'s Hessian sees the **quantized upstream** — GPTQ
> cross-layer error propagation (`calibrate_ml8_paged.py:1860`, `:2050`; the resume
> logic trusts only a contiguous prefix *because* of this). So the 102 forwards are
> NOT redundant (each sees a more-quantized model), and the wasted work is only the
> *downstream-of-target* portion of each forward. Collapsing to one static pass
> (`--hessian-mode single`) is **static-Hessian GPTQ**, a DIFFERENT algorithm —
> equivalence is an empirical PPL question, not bit-identity. If PPL holds within
> noise, we keep the ~100× win; if not, the exact fix is **block-sequential GPTQ**
> (forward each layer once, propagate its quantized output forward — ~N× faster AND
> bit-identical). The fp32 (1.07×) and NVMe (0.0 s) findings are unaffected.

---

## 1. Phase split (measured)

| phase | seconds | share | calls |
|---|---:|---:|---:|
| `hessian_forward` | 1384.3 | **93.6%** | **102** |
| `gptq_quantize` | 94.7 | 6.4% | 102 |
| `corpus_load` | 0.0 | 0.0% | 1 |
| **TOTAL** | **1479.0** | | |

Consistent with the macro fit from the sweep (`calib_s ≈ 383 + 0.0666·tokens`):
the forward dominates, the quantize is the small tail.

## 2. The pig: N× redundant re-forwarding — CONFIRMED

- **102 target linears**, each triggering a **full corpus forward** via
  `compute_hessian` (`calibrate_ml8.py:127` runs `for ids in calib: model(ids)`,
  called once per target at `calibrate_ml8_paged.py:1844` loop).
- Per-target forward: **min 13.1 / median 13.4 / max 14.6 s** — uniform, because
  every target re-forwards the **same** 21,578 tokens through the **whole** model
  (the hook only captures one layer; the other 101 layers' compute is discarded).
- Aggregate: **1384 s = 102 re-forwards** of a 13.6 s pass.
- **The MoE path already solved this** (`calibrate_ml8_paged.py:1509–1553`: set
  collect flags on all layers, do ONE forward). The dense path never got it.

**Single-pass projection (21k):** forward 14 s (one pass) + quant 95 s = **108 s
vs 1479 s → 13.7×**. Extrapolated to 256k (forward is the token-linear term):
forward ~167 s + fixed tail ~383 s ≈ **~9–12 min vs 4h50m**. That is **far past
the 1–2 h target** — the single-pass fix alone over-delivers the entire spec goal.
(Projection is optimistic: 102 simultaneous XtX hooks add some per-step cost and
VRAM; Phase 2's real 256k run confirms the absolute number.)

## 3. fp32-vs-WMMA tax — DEAD (1.07×)

Warmup-corrected dtype probe (8 samples): `allow_tf32=False` 3.52 s vs
`allow_tf32=True` 3.29 s → **ratio 1.07×**. The determinism path costs ~7% of the
forward, **not** a hero lever. Reconciles the de-risk's 26× *isolated-GEMM* finding:
the real 0.8B forward is dominated by the delta-net **SSM scan + paging + dispatch**,
not big matmuls, so matmul precision barely moves it. **Keep `ML8_DETERMINISTIC`
/ `allow_tf32=False` for free** — no `--fast` forward mode needed.

## 4. NVMe corpus staging — ALREADY DONE

`corpus_load = 0.0 s` (cache hit). The NVMe pre-sample cache
(`calib_corpus.collect_calibration` → `/home/kmbandy/models/.calib_cache`) already
eliminates the HDD random-seek tax. **Cross this lever off** — it's shipped.

---

## 5. Phase-2 lever ranking (the gate output)

| lever | verdict | why |
|---|---|---|
| **Single-pass dense Hessian** (port MoE `:1509–1553` to dense) | **DO — the only lever needed** | 93.6% of the clock; ~102× on the forward; alone hits ~9–12 min at 256k |
| fp32→tf32/bf16 forward mode | **DROP** | 1.07× measured; keep determinism for free |
| NVMe corpus staging | **DONE** | `corpus_load` already 0.0 s (cache) |
| Dual-GPU data-parallel forward (`--devices`, n_tok-weighted merge) | **DEFER — not needed for 0.8B 256k** | single-pass (~100×) dwarfs dual-GPU (~2×); the n_tok-weighted shard-merge + equivalence-gate machinery is **unnecessary here**. Keep for the 35B MoE / throughput case where the forward is genuinely paging-bound and can't collapse |
| Route dense GPTQ through batched path | **ALREADY DONE** | the dense loop already calls `batched_gptq_quantize` (not the scalar `gptq_quantize_linear`); quantize is 6.4% and already batched |

**The measure-first discipline paid off twice:** we were about to build the
dual-GPU shard-merge (with its n_tok-weighted correctness landmine + `--max-layers 1`
equivalence gate) and a `--fast` precision mode — both now provably unnecessary for
this target. Phase 2 collapses to **one change**.

## 6. Phase-2 correctness story (single-pass) — clean

In `--faithful-acts`, all 102 `FaithfulActHook` pre-hooks are **already installed
and active on every forward** (`:1839`, before the loop) — they transform each
layer's activations (`x_eff = e4m3(x@Q)@Qᵀ`) deployment-faithfully. The *only*
per-target thing today is which hook **accumulates** H (`set_hessian_target`).
So collecting all 102 in one forward = enable accumulation on all hooks during a
single pass. The activations each target sees are **identical** to its current
per-target pass (transformations are always all-active), so under
`ML8_DETERMINISTIC` the single-pass Hessians should be **bit-identical** (not just
fp-noise-equivalent) to the sequential ones — a trivial equivalence gate.

**Equivalence gate for Phase 2:** single-pass vs sequential per-target Hessians
match (bit-identical under determinism; else within fp noise) on a `--max-layers`
subset; then full 256k reproduces **wiki 19.5470 / held-out 12.2391** in the
**1–2 h** band (expected ~10–15 min). VRAM check: 102 simultaneous fp32 Hessians
(in_feat² each; up to 3584²·4 = 51 MB) ≈ 1–2 GB — fits the R9700's 32 GB.

---

## 6b. EMPIRICAL VERDICT (2026-06-04) — static single-pass: 62× faster, ~+0.1 PPL

Ran the real 256k `--hessian-mode single` calibration + apples-to-apples PPL
(498 MB artifact, same A3 tier, embed quantized — first attempt shipped embed at
bf16 because `ML8_TIER_OVERRIDE` wasn't exported to the converter; that confounded
run was discarded).

| | static single-pass | baseline true-sequential | Δ |
|---|---|---|---|
| wiki PPL | 19.6793 ± 0.159 | 19.5470 | **+0.132 (~0.83σ)** |
| held-out PPL | 12.3032 ± 0.081 | 12.2391 | **+0.064 (~0.79σ)** |
| calib time | **279.6 s (4.7 min)** | 17,430 s (4h50m) | **62× faster** |
| forward | 184.9 s, **1 pass** | (102 passes) | — |

Each Δ is <1σ but BOTH worse → a small, probably-real regression. Cross-layer
error propagation (true-sequential) was worth ~+0.1 wiki PPL. Static single-pass
is **62× faster at a ~0.1 PPL cost** — a cheap lunch, not a free one.

**Strategic wrinkle:** the **MoE path is ALREADY static** (`:1509–1553` collects
all H in one forward, no cross-layer propagation). So the 35B-A3B *product* never
had this dense-only edge — static single-pass is actually the more
product-representative method for the dense bed. The exact-and-fast option is
**block-sequential GPTQ** (forward each layer once, propagate its quantized output
forward — recovers the ~0.1 PPL AND keeps ~N× speed).

## 7. Next action

Re-enter `superpowers:writing-plans` for the **Phase-2 plan**: the single-pass
dense Hessian collection (one all-hooks forward), its equivalence gate, and the
256k acceptance run (leave `--phase-timing` on so it doubles as the production-scale
confirmation — folding in the deferred Task 5). The hackable method core
(`Codebook`/`ErrorProp` seams) rides on top per the spec, but the *speed* goal is
essentially won by this one lever.
