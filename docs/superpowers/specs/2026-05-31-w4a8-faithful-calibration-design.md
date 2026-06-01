# W4A8 Activation-e4m3-Aware (Deployment-Faithful) Calibration — Design

**Date:** 2026-05-31
**Status:** Design approved (brainstorming) — pending spec review → implementation plan
**Authoritative KG node:** f4ffce4b (W4A8 correction). Supersedes the "fp8 weight-tier"
framing of Part B in `docs/superpowers/2026-05-31-calibration-fidelity-fla-rdna.md`.

---

## Goal

Make ml8 calibration **deployment-faithful**: collect Hessians on the rotated,
e4m3-quantized activation the hardware actually feeds, and propagate that quantized
activation through the sequential per-layer loop — so both GPTQ assignment *and* the
heavy fine-tune optimize against the real FP8 lattice instead of a cleaner-than-reality
bf16 target. This is a **bpv-neutral** lever (zero added bits), which is the correct
axis for the MAD-256 both-axes thesis (smaller *and* lower PPL than UD-Q4_K_XL).

## Why this exists (the leak)

ml8 is **W4A8**, not W4A16. The inference kernel (`ggml/src/ggml-cuda/ml8.cu:430–518`,
`ml8_quantize_activations_kernel ~771`) quantizes **activations to e4m3** at every ml8-4
GEMM: per-row dynamic scale `row_absmax/448`, `a_fp8 = e4m3(x/scale)`, computed on the
**post-rotation** activation (kernel input is documented `a : [K] fp32 post-rotation
activation`, ml8.cu:537), then FP8×FP8 against e4m3 centroid weights.

Calibration today (`compute_hessian`, calibrate_ml8.py:139–166) builds `H = XᵀX` on the
**un-rotated bf16** activation, then rotates it algebraically (`rotate_hessian`,
kronecker_rotation.py:132–146). That rotation is the *exact* Hessian of the clean rotated
activation `x'` — but it models **zero** e4m3 rounding. GPTQ and heavy-FT therefore solve
against activations cleaner than the hardware feeds, leaving recoverable PPL on the table.

**Honest framing (carried verbatim from the approved brief):** this does **not** remove
the e4m3 rounding — that rounding is deployed reality, already inside the measured gap.
It lets weight-quant + heavy-FT **compensate** for it. The recoverable PPL is bounded by
how *structured* the e4m3 noise is; per-token post-rotation e4m3 is the structured kind
GPTQ/heavy-FT can partially cancel. **Magnitude is unknown until measured.** A null or
negative result is a legitimate, reportable finding.

## Scope boundary — what the 0.8B bed is and is not

The 0.8B dense-hybrid is a **mechanism + correctness + sign** bed. It is **NOT** a
"does ml8 beat UD-Q4_K_XL" verdict. Per the 2026-05-31 correction (KG): dense models are
UD's *strongest* regime (smart mixed precision, no fat target), and the only full-coverage
data point we have (0.8B = +1.03 PPL vs bf16) loses to UD on both axes. The both-axes win
is **MoE-shaped** — it lives on the 35B where UD wastes bits keeping routed experts at Q8.
So 0.8B proves the faithful forward is *correct* and that the e4m3-aware H + heavy-FT
*recovers PPL versus its own clean baseline*; the sign and rough magnitude then carry up
the ladder 0.8B → 2B → 4B → 8×MI300X pod, where the expert-side gap actually lives.
**Do not cite a 0.8B (or the old 4B FFN-only +0.08) number as the product verdict.**

---

## Architecture — the propagation identity

For an ml8-4 linear with orthogonal rotation `Q`, row-token convention (`x:[N,K]`,
`W:[out,K]`), the stored weight is `W' = WQ` and deployment computes
`y ≈ e4m3(x@Q) @ W'ᵀ`. Expand:

```
y_faithful = e4m3(x@Q) @ W'ᵀ = e4m3(x@Q) @ (Qᵀ Wᵀ) = [ e4m3(x@Q) @ Qᵀ ] @ Wᵀ
           = x_eff @ Wᵀ ,        x_eff = e4m3(x@Q) @ Qᵀ
```

Because `Q` is orthogonal, `rotate ∘ rotate_back` is identity **except** for the e4m3
rounding baked into the middle. So feeding `x_eff = rotate_back(e4m3(rotate(x)))` into the
**unchanged** linear reproduces the faithful output exactly. The entire design turns on
this identity.

### The single forward-pre-hook (installed on every ml8-4 target linear)

1. `a'_q = e4m3(x @ Q)` — rotated, quantized activation, **per-row (per-token) scale
   `row_absmax / 448`**, `ML8_ACT_SCALE_EPS = 1e-12` for all-zero rows (kernel-matched).
2. **Hessian** (only when this layer is the active target): `H += a'_qᵀ @ a'_q` — already
   in rotated space ⟹ **`rotate_hessian` is dropped** (the double-rotation guard).
3. **Propagation** (always): return `x_eff = a'_q @ Qᵀ` as the replacement input ⟹ the
   e4m3 + weight-quant error compounds downstream exactly as in deployment.

### What stays byte-for-byte unchanged

The weight-rotation + GPTQ + dequant + storage path (rotated quantized weight written,
`rotation_blob` written). We change **only** where `H`'s rotation comes from (the forward,
not the algebra) and add the activation rounding. Rotations **precompute from dims + seeds
up front** — they never depended on `H`'s values (paged loop 1390–1397 builds `Q` from
`factor_for_dim(K)` + seed; `H` is only *consumed* by `rotate_hessian`), which breaks the
ordering circularity that "rotate in the forward" would otherwise create.

### fp8 weight tiers (embed, ssm α/β)

Simpler: one-time quant→dequant weight overrides via the scaled-FP8 quantizer (Exec T2),
installed for the whole calibration so they propagate faithfully. Gated by
`--faithful-weights`; the activation e4m3 gated by `--faithful-acts`. v1 models everything
by default; the toggles exist to ablate one tier for clean attribution.

---

## Components

| Unit | File | Responsibility |
|---|---|---|
| e4m3 sim | `ml8_e4m3_sim.py` (new) | Pure-Python e4m3 **bit-matching** `ml8_fp32_to_e4m3` (ml8.cu:440): `fp32_to_e4m3_bits(x)→u8`, `e4m3_decode(u8)→f32`, fused `quantize_act_per_row(x)→f32` (round-trip, `row_absmax/448`, eps). The unit under the gate. |
| faithful pre-hook installer | `calibrate_ml8.py` | `install_faithful_hooks(model, role_map, rotations, accum, flags)`→handles. Per ml8-4 target: compute `a'_q`, accumulate `H` only for the active target, return `x_eff`. Restructures the current `compute_hessian` XtX hook (H now comes from `a'_q`, not raw `inputs[0]`). |
| rotation precompute | `calibrate_ml8_paged.py` | Lift the per-(layer,kind) rotation build (now at 1390) into `build_rotations(dims, seeds)` run **before** Hessian collection. |
| fp8 weight-tier overrides | reuse `scaled_fp8.py` (Exec T2) | Quant→dequant embed + ssm α/β, install as `weight_override` for the whole run. |
| CLI flags | both drivers + `method_gauntlet.py` | `--faithful-acts`, `--faithful-weights` (gauntlet cell pass-through). |

### Which linears get e4m3

Driven by the **same `role_targets.py` classifier the converter uses**, so calibration and
deployment agree on tiers. e4m3 activation injection on ml8-4 targets only; bf16-shipping
linears stay bf16.

---

## Data flow (dense 0.8B path, sequential per-layer loop)

```
build_rotations(dims, seeds)                          # up front, from dims only
install fp8 weight overrides            (if --faithful-weights)
install faithful pre-hooks on all ml8-4 linears   (if --faithful-acts)
for target_linear in order:
    accum.target = target_linear
    model(calib_ids)            # full forward; x_eff replacement propagates faithfully
    H = accum.H                 # already rotated+quantized space
    assert not rotate_hessian_called   # double-rotation guard
    gptq_quantize_linear(target, H, ...)             # weight-rotation + store UNCHANGED
    install dequant(W_q) as weight_override on target   # weight propagation, as today
```

`compute_hessian` already runs a full forward per target linear (O(L²) in the slow scan,
mitigated by the fla-fp32 shim). Injecting e4m3 at every ml8-4 input during that forward
makes propagation faithful at no extra forward cost beyond the rotate+quant ops.

---

## Equivalence gate (must pass A+B before trusting any faithful PPL)

- **Gate A — kernel bit-match.** Sweep a fp32 battery through C++ `ml8_fp32_to_e4m3` vs
  the Python sim ⟹ assert **bitwise-identical uint8**. Battery must hit the kernel's own
  flagged edges: ±448 saturation; the `e=15, m=0..6` fix (ml8.cu:474–481, the bug that
  cost +0.33 PPL); subnormals `< 2⁻⁶`; NaN/Inf; round-to-nearest-**even** ties. Dovetails
  with pending **Exec T15** (Python↔C++ bit-equivalence harness).
- **Gate B — per-row + round-trip.** Real captured activation → Python
  `quantize_act_per_row` vs a dump from `ml8_quantize_activations_kernel` ⟹ assert
  **max-abs-diff 0** on dequantized values (per-token scale included).
- **Gate C — refactor neutrality.** With `--faithful-acts OFF`, the new path must
  reproduce **19.2678 exactly** — proves the restructure did not move the baseline.

**Ground-truth rule:** if Gate A fails, the **kernel is truth** (it is what ships) — fix
the Python sim; never bend the kernel to match Python.

---

## Measurement protocol

Paired-toggle on the 0.8B bed: fixed corpus seed `S`, identical sample order, **only the
`--faithful-*` flags differ** so shared run-to-run noise cancels and a sub-0.05 effect is
resolvable from one pair. (±0.05 is the documented single-run PPL noise floor.)

| # | Config | Isolates |
|---|---|---|
| 1 | faithful OFF | Gate C — must reproduce 19.2678 exactly |
| 2 | acts ON, weights OFF, heavy OFF | activation-e4m3 alone (Δ vs #1) |
| 3 | acts+weights ON, heavy OFF | full faithful forward, heavy off |
| 4 | full faithful + **heavy ON** | the product config |

Then a **3-seed average on the winner (#4) + its OFF control** for a defensible number
before scaling to 2B. The toggles recover clean per-lever attribution even though v1
models everything.

## Success criteria (mechanism validation on 0.8B)

- **Correctness:** Gates A+B bit-exact; Gate C reproduces 19.2678.
- **No-regression:** faithful-ON (heavy off, config #3) must not regress beyond the noise
  floor — the e4m3 rounding is deployed reality already in the gap, so modeling it should
  not *hurt*; if it does, the restructure is wrong.
- **Recovery:** config #4 (faithful + heavy) must clear the **+0.05 floor as a real
  improvement** over config #1 — the signal that heavy-FT finally has a faithful target.
- **Carry-up:** validated sign + rough magnitude justify scaling 0.8B → 2B → 4B → pod,
  where the MoE expert-side gap lives. The 0.8B number is not the product verdict.

---

## Testing

- **Unit (pytest):** Gate A bit-match battery; Gate B per-row round-trip; `x_eff` identity
  test (e4m3 disabled ⟹ `x_eff == x` to fp32); double-rotation assert (faithful-acts on ⟹
  `rotate_hessian` never called).
- **Integration:** Gate C neutrality on a small/fast slice; existing `test_dense_resume`
  stays green.

## Compute note

The O(L²) re-forward is already present; faithful adds rotate+e4m3 per linear per forward —
tolerable on 0.8B with the fla-fp32 path (~1572s baseline). Budget this when scaling to
2B/4B. The standing O(L²) all-targets-in-one-forward optimization remains a separate
follow-up and is intentionally out of scope here.

## Out of scope (v1)

- The O(L²) `compute_hessian` re-forward optimization (separate follow-up).
- AWQ / act_order / centroid-init and other bpv-neutral quantizer levers (stack *after*
  the faithful forward lands; they will now optimize against the faithful H for free).
- MoE paged-path parallelization specifics beyond reusing the precomputed rotations
  (the paged loop already collects all H then task-queues GPTQ; the faithful forward slots
  into the H-collection phase the same way — detailed in the implementation plan).
