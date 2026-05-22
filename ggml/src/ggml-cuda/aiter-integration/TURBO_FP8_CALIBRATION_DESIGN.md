# turbo-FP8 Calibration Design Space

**Status:** living reference doc, started 2026-05-21 wrap-up. Captures the
full set of levers in the calibration pipeline so we can iterate without
re-deriving the design each time. Pair with `TURBO_FP8_KERNEL_DESIGN.md`
for the kernel-side context.

## Pipeline recap

```
captured fp16 K/V tensors  →  optional transforms  →  per-block normalize
                                     ↓
                       pooled normalized magnitudes
                                     ↓
                        Lloyd-Max fit (16 centroids)
                                     ↓
                     E4M3 lattice snap → 16-byte LUT
                                     ↓
              ~/.cache/llama.cpp/turbo-fp8/<fingerprint>/l<N>_<k|v>.bin
                                     ↓
              registry lazy-loads → kernel decodes at attention time
```

Calibration produces one 16-byte LUT per (layer, kv-dir). The kernel
treats it as a small constant table: 4-bit qs index → centroid byte →
E4M3 value → multiply by per-block fp16 scale → final magnitude → XOR
sign bit.

## The 7 levers

Levers are listed in **roughly decreasing impact** based on Phase 0
findings + Option F empirical results on Qwen3.6-35B-A3B.

---

### 1. Hadamard rotation at scatter time

**Current state:** NOT applied. Phase 1E built the rotation kernel
(`ggml/src/ggml-cuda/turbo_fp8_hadamard.cuh`) but it's not connected
to the FP8 scatter path.

**Why it matters:** K_cur on transformer attention typically has
channel-aligned outliers — a few attention dims dominate the magnitude
range while most are near-zero. Lloyd-Max on this raw distribution
places too many centroids near zero (where data is dense but signal
is weak) and too few in the moderate-magnitude range (where attention
actually weights things).

Walsh-Hadamard rotation along head_dim spreads channel-aligned outliers
uniformly across all positions. After rotation:
- The distribution becomes more uniform → centroids spread more usefully
- Same kernel decode logic; the only difference is the K bytes stored

**Phase 0 measurement:** ~12% MSE improvement from Hadamard on K.

**Hard constraint:** the kernel decoder must apply the SAME rotation as
calibration. They have to be paired — calibrate-without + kernel-without
is consistent (current state). Calibrate-with + kernel-with is the goal.
Crossing them breaks correctness.

**Why this is probably the #1 lever:** the fallback Qwen3.5-4B canonical
LUT slightly beat our Qwen3.6-A3B-fitted LUTs in the Option F runs
(5.7796 vs 5.7984). The fallback was fit WITH Hadamard, on a different
model. Even with that model-mismatch handicap, the Hadamard-fitted
centroid distribution wins on attention-relevance.

**Cost:** kernel work + AOT spec rework. Captured in MAD-224 acceptance.

---

### 2. Fitting loss function

**Current state:** Lloyd-Max minimizes squared error on raw magnitudes
(MSE-optimal).

**Why this is suboptimal for attention:** attention output =
`softmax(Q·K^T / √d) · V`. Small K values contribute almost nothing
(their dot-product is small → softmax weight is small → V contribution
is negligible). Large K values dominate the softmax. So **encoding
fidelity matters more for large magnitudes than small**.

Lloyd-Max with MSE loss treats all magnitudes equally → places centroids
where samples are dense → packs centroids near zero (where attention
doesn't need them).

**Alternatives:**
| Loss | Effect |
|---|---|
| Magnitude-weighted MSE: `Σ |x|^p · (x - c)^2` | Biases centroids toward larger values. `p=1` gives moderate bias; `p=2` heavy. |
| Log-space fit: `Lloyd-Max(log(|x|))` | Natural geometric spacing. Centroids cover orders of magnitude evenly. |
| Attention-aware: capture not just K but the actual attention output | Fit to minimize output divergence. Most accurate, most expensive. |
| Reverse KL on softmax distribution | Even tighter coupling to the actual loss surface. Research-grade. |

**Quick win:** magnitude-weighted MSE is a ~10-line change to the Python
fitter. Worth trying before any more expensive ideas.

---

### 3. Data corpus (what we capture)

**Current state:** Option F captures 4096 tokens from 1 chunk of
wikitext.test via the `MT_TURBO_FP8_DUMP_DIR` env hook.

**Levers:**
- **Size**: more tokens = better convergence. Diminishing returns past
  a few thousand per (layer, dir).
- **Diversity**: code + prose + chat + math + non-English. The
  distribution of K/V values differs by content type. A pure-prose
  calibration may underweight code or math distributions.
- **Capture phase**: prefill-only vs prefill+decode. Distributions
  can differ slightly because decode's Q is single-token. Probably
  small effect but worth measuring.
- **Per-prompt vs cross-prompt pool**: pooling across many prompts
  averages out prompt-specific biases.

**Built-in corpus for full A (MAD-224):** plan to embed ~2KB varied
text in the binary. Mix:
- ~500 bytes English technical prose
- ~500 bytes code (Python or C)
- ~500 bytes numerical / table-like content
- ~500 bytes natural conversation

---

### 4. Granularity (per-what LUT)

**Current state:** per-(layer, kv-dir) — 16 bytes × n_layers × 2 dirs.

**Trade-offs by granularity:**
| Granularity | Bytes (Qwen3.6-A3B, 8 attn layers × 2) | Quality |
|---|---|---|
| One global LUT | 16 | Worst |
| Per kv-dir only | 32 | Slightly better |
| Per (layer, kv-dir) (current) | 256 | Phase 0 sweet spot |
| Per (layer, kv-dir, kv-head) | 512 (for 2 kv-heads) | Marginal for Qwen3.5-4B; untested on 3.6 |
| Per (layer, kv-dir, kv-head, channel) | Explodes | Overkill |

**Phase 0 finding:** per-(kv, layer) was the sweet spot for Qwen3.5-4B.
Per-head granularity didn't help meaningfully. Worth re-checking on
Qwen3.6-A3B since the architecture differs slightly (different
attention-to-DeltaNet ratio).

---

### 5. Centroid count (turbo3 vs turbo4 vs turbo5)

**Family:**
| Variant | # centroids | Index bits | Bytes per block (BS=256) |
|---|---|---|---|
| turbo3-FP8 | 8 | 3 | 130 (2 scale + 96 qs + 32 signs) |
| turbo4-FP8 (current production) | 16 | 4 | 162 (2 + 128 + 32) |
| turbo5-FP8 | 32 | 5 | 194 (2 + 160 + 32) |

**Phase 0 quality (MSE relative to fp8_raw, lower = better):**
- turbo3-FP8: 11.21× — 1.5× better than q4_0
- turbo4-FP8: 3.21× — 5× better than q4_0
- turbo5-FP8: 1.30× — near-lossless vs fp8_raw

**Lever:** if quality is critical and you have the bits, jump to turbo5.
Each step costs +0.5 bpv but gives ~3× MSE improvement.

---

### 6. Normalization unit (currently per-block max-abs)

**Current state:** each 256-element (token, kv-head) row divided by its
own max-abs, scale stored as fp16 in the block header.

**Why this MUST match the kernel:** calibration fits on normalized
magnitudes in [0, 1]. The kernel reconstructs by `lut[idx] * scale`.
If calibration uses per-row L2-norm but the kernel uses per-row max-abs,
the LUT decodes the wrong magnitudes.

**Alternatives (must update kernel + calibration together):**
- **Per-block L2-norm**: more robust to single-outlier rows. Slightly
  better fit precision near the outlier but trickier scale recovery.
- **Per-tensor max-abs**: one scale per layer-dir per call. Simpler
  storage but loses dynamic range across rows.
- **Channel-wise**: per-position normalization. Probably overkill.

Sticking with per-block max-abs for now is the right call — it's the
classic min-overhead choice that almost every FP8 implementation uses.

---

### 7. E4M3 snap strategy

**Current state:** distinct-snap (no duplicate E4M3 bytes). Earlier
first-fit collapsed 6/16 centroids onto 0x00.

**Strategies:**
| Strategy | Behavior |
|---|---|
| First-fit (initial) | For each centroid, snap to nearest E4M3 byte. Duplicates allowed. Wastes capacity on peaked distributions. |
| Distinct-snap (current) | Force all 16 bytes unique. Walks outward to next unused E4M3 byte. Marginal improvement. |
| Greedy-coverage | Maximize the magnitude range covered. Gives up local precision near dense regions. |
| Forced anchors | Always include byte 0x00 (=0.0) for "true zero" + byte 0x38 (=1.5) for "max" + 14 fitted in between. Useful when known structural values exist. |

**Open question:** is the snap loss meaningful compared to the underlying
Lloyd-Max placement? Phase 0 measured "E4M3 lattice constraint costs only
4-5% MSE" vs unconstrained fp16-scale fitting — so the snap is not the
dominant error source. Focus elsewhere.

---

## Priority ranking for tomorrow (and beyond)

| # | Lever | Effort | Expected impact |
|---|---|---|---|
| 1 | Hadamard at scatter time (lever #1) | kernel work, finite scope | Probably gets us past the fallback. Phase 0 said ~12%. |
| 2 | Magnitude-weighted fit loss (lever #2) | ~10-line Python change | Tests the "MSE vs attention-relevance" hypothesis. Fast experiment. |
| 3 | Try turbo5-FP8 end-to-end (lever #5) | New kernel instantiation + LUT format | Phase 0 predicted near-lossless. Worth confirming on real model. |
| 4 | Bigger / diverse calibration corpus (lever #3) | Update fitter + embed corpus | Modest improvement; mostly affects how well per-prompt distributions are handled. |
| 5 | Per-head granularity (lever #4) | Storage layout change | Phase 0 said marginal; re-check on Qwen3.6. |

## What we KNOW vs what we GUESS

**Known from measurement:**
- Hadamard helps ~12% MSE (Phase 0)
- E4M3 snap loss is small, ~4-5% (Phase 0)
- Per-(layer, kv-dir) granularity is the sweet spot for Qwen3.5-4B (Phase 0)
- turbo5-FP8 is near-lossless vs fp8_raw (Phase 0)
- Magnitude-weighted Lloyd-Max loss with p=5 is the production-best fitter recipe — see "Matrix sweep results" below (2026-05-22 empirical)
- The remaining ~0.012 PPL gap to f16 at ctx=4096 is Hadamard-shaped, not fitter-shaped (matrix sweep saturation)

**Hypothesized but unmeasured:**
- Hadamard wiring closes the residual fitter gap (theory + Phase 0 MSE proxy)
- Attention-aware loss would be the best (intuitive, expensive to test)
- Hadamard's benefit may also shrink at long ctx where the calibration-vs-fallback delta already does

## Matrix sweep results (2026-05-22 empirical, Qwen3.6-35B-A3B, ctx=4096 unless noted)

Four-tier sweep across the calibration lever space (`tests/perf-baseline/calibration-sweep/`).
All cells on the paged AITER FP8 WMMA path.

| Stage | Best recipe | PPL | Δ vs f16 (5.7486) |
|---|---|---|---|
| Baseline (mse, distinct, default corpus) | T1_baseline | 5.8014 | +0.053 |
| Fallback (Qwen3.5-4B canonical + Hadamard) | T1_ctrl_fallback | 5.7796 | +0.031 |
| Tier 1 winner | mag_weighted p=1 × bigger | 5.7763 | +0.028 |
| Tier 2 winner | mag_weighted p=2 × bigger | 5.7703 | +0.022 |
| Tier 3 winner | mag_weighted p=3 × bigger | 5.7655 | +0.017 |
| **Tier 4 winner** | **mag_weighted p=5 × bigger** | **5.7601** | **+0.012** |

**Headline:** fitter tuning alone closes ~78% of the original 0.053-PPL gap to f16.

**Per-lever findings:**
- **mag_weighted is the dominant lever** — went from p=1 to p=5; deltas grow with p across this range, plateauing into noise past p≈4.
- **bigger corpus** consistently adds ~0.001-0.005 PPL — tiny but free.
- **forced_anchors** synergizes with mag_p1 (+0.008 interaction term) but the absolute synergy is dominated by the higher-p effect.
- **per_dir granularity** is noise-level on quality, but collapses to a single LUT per dir — useful for the storage-conscious user (single 16-byte LUT per kv-dir vs n_attention_layers × 16 bytes).
- **log_space loss** is catastrophic (+0.79 PPL) — geometric spacing on normalized [0,1] dumps too many centroids near zero.
- **Mixed corpus (prose+C+++Python)** is meaningfully worse than pure prose (+0.035) — calibration-domain matching matters.
- **Triple combos** added nothing over the best pair.

**Recipe validation at production contexts** (`*-validate.json`):
| ctx | recipe (p=3×bigger) | fallback | Δ |
|---|---|---|---|
| 4096 | 5.7655 | 5.7797 | **-0.014** |
| 8192 | 5.7109 | 5.7195 | -0.009 |
| 16384 | 5.3746 | 5.3788 | -0.004 (within noise) |

The win **narrows at long ctx** — calibration matters most when the LUT is the dominant approximation source (short ctx); at long ctx other error sources dominate. Hadamard's expected benefit may have similar ctx-shape; worth specifically validating as part of the Hadamard wiring ticket.

**Production recipe:** `--fit-loss mag_weighted --mag-weight-p 5.0 --granularity per_layer_dir --snap-strategy distinct` on a ~16k-token capture from in-domain prose.

## Related files / tickets

- `scripts/calibration/fit_centroids.py` — original Phase 0 fitter (Hadamard-aware, fits offline from HF transformers KV dumps)
- `scripts/calibration/fit_centroids_from_dump.py` — Option F fitter (reads device dumps from MT_TURBO_FP8_DUMP_DIR). Now supports `--fit-loss`, `--granularity`, `--snap-strategy`, `--forced-anchors` for the matrix sweep.
- `scripts/perf/turbo_fp8_calibration_sweep.sh` — matrix sweep driver (`TIER=1..4`)
- `scripts/perf/turbo_fp8_validate_recipe.sh` — recipe validation across multi-ctx
- `tests/perf-baseline/calibration-sweep/` — per-tier results JSONs
- `ggml/src/ggml-cuda/turbo_fp8_hadamard.cuh` — rotation kernel (built, not yet wired into scatter)
- `ggml/src/ggml-cuda/mt_turbo_fp8_lut_registry.{h,cu}` — LUT registry (load path only)
- MAD-214 — parent epic
- MAD-216 — multi-model calibration
- MAD-217 — runtime LUT loading (GGUF metadata or sidecar)
- MAD-224 — in-process auto-calibration (Phase 1G-C/D/E + Hadamard wiring acceptance criterion)
- TURBO_FP8_KERNEL_DESIGN.md — kernel-side companion to this doc
