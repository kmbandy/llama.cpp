# ml8 FP8 Kernel-Substrate Unification — Design Spec

**Date:** 2026-06-14
**Status:** Spec (design) — approved scope: full 4-phase, all phases in buildable detail.
**Predecessor notes:** `docs/superpowers/2026-06-14-ml8-fp8-substrate-unification-notes.md` (capture/recon).
**Recon sources:** `docs/superpowers/recon/2026-06-14-{triton,aiter,rocm-therock}-rdna4-fp8.md`.
**Profile source:** `docs/superpowers/2026-06-13-ml8-trainer-dispatch-profile.md`.

---

## 1. Goal

Unify the three parallel implementations of the same fp8 / e4m3 / Hadamard math onto **one
Triton kernel source** — JIT-compiled for training, AOT-compiled for inference, **bit-exact by
construction rather than by testing** — and in doing so dismantle the host-bound dispatch wall
(≈80K kernel launches, ~1744 ms CPU dispatch, GPU idle ~89%) that now dominates the 4B trainer's
~4.3s micro-step after the MAD-290 fp8 backward landed.

Two outcomes, one mechanism:
- **Performance:** drop the micro-step from ~4.3s toward the GPU-bound floor (~2.8–3.2s) by
  cutting launch *count* (fused prologue) and launch *overhead* (graph capture).
- **Faithfulness:** collapse the training-vs-inference numerics duplication that has repeatedly
  produced drift (the act-replay work burned on exactly this), so a single kernel source is the
  ground truth for both paths.

---

## 2. Background — how we got here

MAD-290 replaced the placeholder fp8 backward (which computed fp8 `dy8/sdy` then discarded them
and ran fp32 `torch.@` GEMMs on bad rocBLAS skinny tiles) with the real `ml8_backward_gemms` via
`torch._scaled_mm`. That took the 4B micro-step **10.6s → ~4.3s** (committed `8e3fe3a3a`).

Profiling the residual flipped the bottleneck's character: the step is now **host-bound**.
~498 ms GPU sync (11% of wall) versus **1744 ms CPU dispatch over ~80K kernel launches**, GPU
idle ~89%. A `TRACE_ELEM` instrumentation pass attributed the launch storm to the **ml8 activation
prologue**, dominated by:
- `ml8_e4m3_sim.e4m3_roundtrip` — a ~40-op pure-torch **software emulation of e4m3 round-to-nearest-even**
  (subnormal / saturation / NaN-slot handling), called ~400×/step (`scripts/calibration/ml8_e4m3_sim.py:69`);
- the **FWHT rotation** butterfly (`kronecker_rotation.fwht_raw`, `scripts/calibration/kronecker_rotation.py:31`),
  log₂(b) stages of slice/cat/add per call;
- the per-call **`layer_from_components` rebuild** (`scripts/calibration/ml8_runtime.py:413`).

The deeper finding: we maintain **three parallel kernel implementations** of the same fp8/e4m3/Hadamard
math —
1. **Training** — torch + Triton (`fp8_qat.py`, `ml8_e4m3_sim.py`, `kronecker_rotation.py`, `ml8_runtime.py`);
2. **Inference** — HIP C++ (`ggml/src/ggml-cuda/ml8.cu`) + AOT-Triton GEMM (`gemm_ml8.py`);
3. **KV** — HIP + Triton (`ml8-4-kv`, 5bpv sign+mag) — a **separate, third** bit-layout, out of scope here.

That triplication is a faithfulness-debt generator. **Decision (made, not re-litigated here):
unify the kernel substrate on Triton.** Triton is already the cross-runtime through-line — the
inference GEMM `gemm_ml8.py` is AOT-compiled Triton vendored from aiter's `gemm_a8w8_blockscale`,
the same a8w8-blockscale source the trainer's `ml8_gemm` calls. The GEMM is already unified; the
duplicated surface is the **prologue/quant**.

---

## 3. Hardware & toolchain context (load-bearing constraints)

- **R9700 = RDNA4 = gfx1201** — dev/train box, fp8 via **WMMA**.
- **MI300X = CDNA3 = gfx942** — datacenter, fp8 via **MFMA**.
- **gfx1250 is a newer arch we do NOT own** — gfx1250-only upstream work is irrelevant to us.

**Triton:** local clone `~/GitHub/triton` @ `4768da5` (2026-05-16), 150 commits behind
`origin/main` @ `007ef1530`. main is `3.8.0-dev` with **no 3.8.0 tag** — any bump pins a main SHA;
latest real tag is v3.7.0.

**aiter:** local clone `~/GitHub/aiter` (@ `69cbe3ff8`); vendored into llama.cpp @ `9c79a5b59`
(2026-05-25); latest release v0.1.15.post1 (2026-06-08). aiter requires Triton **≥ 3.6.0**; we run
3.7.0, so re-vendoring newer aiter does **not** force a Triton downgrade, and the #10458 bump stays
compatible.

**ROCm libraries are not a viable RDNA4 fp8 path today** (so there is no library escape hatch to
reconsider mid-stream): hipBLASLt #6365 (gfx1200/1201 fp8 arch list) **open**; Tensile #7192 —
gfx1201 resolves to a nonexistent `gfx1200.dat` with no tuned configs (**exactly** the skinny-tile
failure that pushed us off rocBLAS) **open**; Composable Kernel now *has* WMMA fp8 GEMM instances
(`device_gemm_wmma_universal_f8_*`) but **no gfx1201 tuning/validation**. (ROCm has consolidated
hipBLASLt/Tensile/rocBLAS/MIOpen/CK into the `ROCm/rocm-libraries` monorepo.) CK is a *future*
MI300X escape hatch, unproven on gfx1201 now.

---

## 4. Architecture — the unification model

**One `@triton.jit` source per kernel, consumed two ways:**

```
                    ┌─────────────────────────────┐
                    │   Triton kernel source (.py) │   ← single ground truth
                    │   fused rot+quant prologue   │
                    └───────────────┬──────────────┘
                  JIT (training)    │    AOT (inference)
            ┌───────────────────────┘    └───────────────────────┐
            ▼                                                      ▼
  @triton.jit + autograd wrapper                     add_triton_aot_kernel (CMake)
  called from fp8_qat / act_replay                   → aiter_triton_aot static lib
  (R9700 dev, JIT-compiled)                          → linked into llama.cpp, replaces
                                                       ml8.cu::ml8_fused_rot_quant_kernel
```

- **Already unified:** the **GEMM**. `gemm_ml8.py` (AOT-Triton) ⇄ trainer `ml8_gemm`
  (`ml8_runtime.py`), same a8w8-blockscale source. No work needed; this is the proof the model works.
- **To unify:** the **prologue/quant**. Inference HIP `ml8_fused_rot_quant_kernel` (`ml8.cu:881`)
  and `ml8_quantize_activations_kernel` (`ml8.cu:771`) vs training torch `e4m3_roundtrip` /
  `quantize_act_per_row` / `fwht_raw`. **Component A** collapses these onto one Triton source.
- **Stays hand-HIP (pragmatic, not dogmatic):** `gemv`/decode (`ml8_gemv_tpl`, `ml8.cu:589`,
  batch=1) where Triton is weak and inference latency is non-negotiable.
- **Out of scope:** the **KV kernel** (`ml8-4-kv`, 5bpv sign+mag) — different bit-layout, separate
  system, not touched by this work.

**Component → phase map:**

| # | Component | Lang | New/reuse | Phase |
|---|---|---|---|---|
| **E** | Triton bump → main SHA ≥ #10458 + carry AOT `--target` patch + **reproducible rebuild** | toolchain | enabling | 0 |
| **D** | gfx1201 tuned a8w8-blockscale configs for our shapes | JSON (data) | net-new data | 0 |
| **A** | "Stack" kernel: fused FWHT-rotation + per-row e4m3 quant, **fwd + bwd** | Triton | net-new | 1 |
| **B** | e4m3 quant primitive (per-row + per-tensor, RNE) | Triton | net-new (crib aiter `fused_fp8_quant.py`) | 1 |
| **F** | `layer_from_components` per-call rebuild caching | Python | cheap | 1 |
| **C** | CUDA/HIP graph capture of the training micro-step | torch/HIP | net-new | 2 |
| **(A→AOT)** | AOT the Phase-1 prologue into inference, replace HIP kernel | CMake/Triton | net-new wiring | 3 |

---

## 5. Phase 0 — currency + microbench (measured gate)

**Purpose:** get the substrate current and *measure* it before committing to Phases 1–3. Output is
data: substrate confirmed, free perf banked, numbers that re-confirm the scope of 1–3.

### 5.1 E — Triton bump + AOT patch + reproducible rebuild

- **Pin** a `~/GitHub/triton` main SHA **≥ `bb5acbe59` (#10458, 2026-06-04, "Fix fp8 conversions on
  archs not natively supporting fnuz")**. This is the OCP-e4m3-vs-fnuz correctness fix that landed
  *after* our HEAD; without it Triton *can* emit wrong fp8 on RDNA4. Record the exact pinned SHA in
  the build script and in this repo's vendored-version manifest.
- **Re-apply our `tools/compile.py` AOT `--target` patch** on the new SHA. Upstream tracking issue
  **#4219 is open with no fix**, and main's `compile.py` is byte-identical to ours — it still forces
  GPU init during C-stub emission. We carry this patch regardless of which SHA we pin. The patch
  must be a tracked diff (not an ad-hoc edit) so it re-applies cleanly across SHAs.
- **Reproducible rebuild (the "fragility is a bug" item):** produce a **pinned, documented,
  idempotent build script** for the Triton toolchain — exact SHA, exact patch, exact build flags,
  exact gfx target list (gfx1201 for R9700; gfx942 for MI300X), and a post-build smoke that imports
  Triton and compiles a trivial gfx1201 kernel. The acceptance bar: a clean checkout + one script
  invocation yields a working Triton with our AOT patch, no manual steps, no GPU-init failure during
  AOT stub emission. This replaces the current "fragile rebuild we avoid."

### 5.2 D — gfx1201 tuned a8w8 configs

- Of our **16 (N,K) combos** from {2560, 4096, 8192, 9216}, **exactly one — (8192,8192)** — has a
  tuned per-shape `gfx1201-GEMM-A8W8_BLOCKSCALE-N=..-K=..json`. The other 15 fall back to the
  generic `gfx1201-GEMM-A8W8_BLOCKSCALE.json`, whose large-M path collapses onto a single `"any"`
  config. There are **zero** files for N=2560, N=9216, K=2560, K=9216.
- **Generate/vendor tuned JSONs** for our shapes using aiter's tuning harness. **Data-only — no
  kernel or Triton change.** Vendor the JSONs alongside the existing config tree.

### 5.3 Verify fp8 correctness (don't assume either way)

- turbo4_fp8 PPL/NIAH validation currently *passes* (f16-equivalent), so we may not be hitting the
  broken fnuz path. **Measure** e4m3 round-trip semantics on gfx1201 **before and after** the #10458
  bump; confirm OCP e4m3 (not fnuz) codes. Gate: turbo4 PPL/NIAH parity holds post-bump.
- Oracle for the bit check: `ml8_e4m3_sim.fp32_to_e4m3_bits` / `e4m3_roundtrip` (the existing scalar
  + vectorized references), which mirror `ml8.cu:440 ml8_fp32_to_e4m3`.

### 5.4 Microbench (the substrate-confirming measurement)

- TFLOPS at our **real shapes on R9700**, four cells:
  1. old-vendored Triton-a8w8 (baseline, pre-bump),
  2. Triton ≥ #10458,
  3. Triton ≥ #10458 **+ our tuned gfx1201 configs (D)**,
  4. `torch._scaled_mm` (the backward's current GEMM).
- This tells us (a) substrate is confirmed on data, and (b) whether D closes the ~20% perf gap. The
  recon already established the ~20% is **not** addressed by any in-range Triton commit (fp8 WMMA
  instruction-selection for gfx12 and the RDNA4 wide-LDS/InThreadTranspose lever both predate our
  HEAD), so D + tuning is the realistic lever; the microbench proves or disproves it.

**Phase 0 exit gate:** pinned reproducible Triton build green; fp8 correctness verified (turbo4
parity); microbench table produced; **re-confirm scope of Phases 1–3 against the numbers** before
proceeding.

---

## 6. Phase 1 — the stack kernel (A / B / F)

**Purpose:** replace the launch-storm prologue (FWHT rotation + e4m3 emulation) with one fused
Triton kernel that is **bit-exact to the current torch oracle** and **AOT-able to inference**
(Phase 3). This is where the unification asset is created.

### 6.1 The prologue being replaced (exact current numerics)

The faithful W4A8 activation transform is `x_eff = e4m3(x @ Q)` — **rotate, then per-row e4m3
quantize**, no inverse rotation (weights are rehydrated in the rotated basis). Current code path
(`act_replay_student.py::apply_acts`, `:155`):

```python
if self.col_perm is not None:               # input-column reorder FIRST (GGUF order)
    x = x.index_select(-1, self.col_perm)
if self.rotation is None:
    return x
flat  = x.reshape(-1, K).float()
x_rot = self.rotation.forward(flat)          # KroneckerRotation.forward: h_a.T @ (fwht_raw(X)*1/√b)
# optional Hessian collection on ROTATED, PRE-QUANT activations (see 6.4):
#   if self._collecting_h: self._h_acc += x_rot.detach().T @ x_rot.detach()
a_q = x_rot + (quantize_act_per_row(x_rot) - x_rot).detach()   # STE: fwd=quant, bwd=identity
x_eff = a_q.reshape(x.shape).to(orig_dtype)
```

with `quantize_act_per_row` (`ml8_e4m3_sim.py:136`): `scale = rowabsmax/448` (eps-floored
`ACT_SCALE_EPS=1e-12`), `q = e4m3_roundtrip(x_rot/scale) * scale`, and `KroneckerRotation.forward`
(`kronecker_rotation.py:85`): `Y = h_a.T @ (fwht_raw(X) * 1/√b)` over `X = x.reshape(..., a, b)`.

The deployed inference kernel that this mirrors is `ml8.cu::ml8_fused_rot_quant_kernel` (`:881`),
which fuses the FWHT **H_b leg** + per-row e4m3 quant (the `turbo_fp8_hadamard.cuh` FWHT, capped at
b ≤ 1024 — see `factor_for_dim`). **Phase-1 task 1 reads `ml8.cu:881` to confirm the kernel's exact
fused scope** (whether the small `h_a` a×a leg is folded or left as a cheap dense matmul) and matches
it; default assumption = fuse FWHT(H_b) + e4m3 quant, keep the O(a²)-per-row `h_a` leg as a small
dense matmul as the kernel does.

### 6.2 A — fused rotation+quant Triton kernel

- **Forward:** one Triton kernel computing `a_q = e4m3_per_row(rotate(x))`, fusing the FWHT H_b leg
  and the per-row e4m3 RNE quantization (and the `h_a` leg per 6.1). Output identical in value to
  the torch composition above.
- **Backward:** straight-through identity through the (piecewise-constant) e4m3 quant, then the exact
  linear VJP through the orthogonal rotation. Since `rotation.forward(x) = x @ Q`, the VJP is
  `dx = dy @ Qᵀ = rotation.inverse(dy)` (`kronecker_rotation.py:102`), followed by the inverse
  `col_perm` scatter when a column reorder is attached. This preserves the existing STE semantics:
  the act-quant contributes identity gradient, the rotation contributes its transpose — matching the
  `x_rot + (q - x_rot).detach()` forward value exactly while restoring the gradient path to upstream
  centroids/scales that a naive detach would sever.
- **Numerics ground truth:** `e4m3_roundtrip` (RNE, subnormal/saturation/NaN-slot per `ml8.cu:440`)
  for the quant; `fwht_raw` for the rotation. The kernel must reproduce the **e4m3 codes** exactly.

### 6.3 B — e4m3 quant primitive

- A single Triton primitive: per-row (per-token) and per-tensor e4m3 RNE quant. Crib the structure
  from aiter `quant/fused_fp8_quant.py` (arch-generic, exists upstream), but the **rotation-fused**
  variant and any **LUT-weight** path are net-new (confirmed absent upstream — all aiter "rotat*"
  hits are RoPE; no LUT-weight fp8 anywhere).
- Reused by: A's forward, the backward's tensorwise quant, and standalone activation quant. **May be
  folded into A** if profiling shows the separate call isn't worth the boundary — decided in Phase 1
  on the re-profile, not pre-committed.

### 6.4 Constraint — keep the Hessian hook

`apply_acts` exposes `x_rot` (rotated, **pre-quant**) to accumulate a rotated-space Hessian when
`_collecting_h` is True (`act_replay_student.py:196`, used by the GPTQ/pv reassignment paths). The
fused kernel **must not hide `x_rot`** — it must optionally emit the rotated pre-quant activation (or
provide a mode that returns it) so Hessian collection still works. This is a hard interface
constraint, not an optimization.

### 6.5 F — `layer_from_components` rebuild caching

`layer_from_components` (`ml8_runtime.py:413`) rebuilds the kernel layer every forward. Within a
step's forward + activation-recompute the centroids are constant, so **cache the built layer per
step** (keyed by step + tensor identity/version, invalidated when centroids update). Small; folds
into the A/C work. Note the existing `_packed_indices_cached` already caches frozen indices by
id+version+device — extend the caching to the full built layer.

### 6.6 Testing (TDD, bit-exact)

- **Forward:** assert the kernel's e4m3 **codes** are identical to `quantize_act_per_row(rotation.forward(x))`
  across a battery of inputs spanning normal / subnormal / saturation / NaN-slot / zero and the real
  trainer shapes (M × {2560,4096,8192,9216}). Code-level equality, not just value tolerance.
- **Backward:** assert the kernel's VJP matches autograd of the STE composition (identity through
  quant, `rotation.inverse` through rotation) to fp32 tolerance, including the `col_perm` case.
- **Wiring regression:** after wiring into `fp8_qat.py` / `act_replay_student.py` / `ml8_e4m3_sim.py`,
  a one-step trainer run reproduces the pre-change KL/loss to tolerance.

### 6.7 Wiring points

- `scripts/calibration/act_replay_student.py::apply_acts` (`:155`) — swap the
  `rotation.forward` + `quantize_act_per_row` STE composition for the fused kernel (keeping the
  `col_perm`, Hessian-hook, and STE-value semantics).
- `scripts/calibration/ml8_e4m3_sim.py` — `e4m3_roundtrip` / `quantize_act_per_row` are retained as
  the **test oracle** (not deleted), and as a runtime fallback behind a flag (see §9 risk/rollback).
- `scripts/calibration/fp8_qat.py::Ml8Fp8Fn.forward` (`:60`) — the activation `fp8_quant`/prologue
  path consumes the fused kernel where it overlaps.

**Phase 1 exit gate:** bit-exact forward + backward tests green; one-step trainer KL/loss parity;
re-profile shows launch count down in the quant region. **Expected:** ~4.3s → ~3.8–4.0s (drops
~20–25K of ~80K launches) plus the AOT-able faithful asset for Phase 3.

---

## 7. Phase 2 — graph capture (C)

**Purpose:** remove dispatch *overhead* for the whole step (orthogonal to A/B, which remove launch
*count* in one region). CUDA/HIP-graph capture collapses dispatch for **all** ~80K launches toward
the GPU-bound floor.

### 7.1 Prereq inventory (the real work)

Enumerate and remove every host sync / dynamic allocation that blocks capture inside the micro-step:
- `.item()` / `.cpu()` calls (any remaining host reads on the hot path);
- the `layer_from_components` rebuild (addressed by **F** — must be hoisted/cached out of the captured
  region);
- `_validate_gidx_once` and the weakref pack cache (`ml8_runtime.py`);
- any data-dependent control flow (e.g. `if n_pad:` in `Ml8Fp8Fn.forward:87`) — must become static or
  be lifted out of capture.

Static shapes already hold (the micro-step is fixed-shape), which is the precondition graphs need.

### 7.2 Implementation

- Hoist all CPU-side control flow out of the captured region (build the layer once per step, before
  capture; reuse inside).
- Allocate a **static memory pool** for the captured region's intermediates (graph capture requires
  stable addresses across replays).
- Capture the forward + backward of the micro-step; replay per step.

### 7.3 Testing

- Captured-vs-eager **parity**: one captured step reproduces the eager step's KL/loss and gradients
  to tolerance.
- **Re-profile** confirms dispatch time collapses and GPU idle drops.

**Phase 2 exit gate:** captured/eager parity; re-profile. **Expected:** ~4.3s → ~2.8–3.2s.

---

## 8. Phase 3 — AOT to inference (the payoff)

**Purpose:** make the Phase-1 kernel the single source for inference too — faithfulness by
construction, the duplication retired.

### 8.1 Mechanism

- AOT-compile the Phase-1 fused rot+quant Triton kernel into llama.cpp via the existing
  `add_triton_aot_kernel` CMake path → `aiter_triton_aot` static lib (the **same mechanism** already
  proven by `gemm_ml8.py`). Requires the Phase-0 `compile.py` `--target` patch (AOT stub emission
  without GPU init).
- **Replace** the HIP `ml8.cu::ml8_fused_rot_quant_kernel` (`:881`) call site with the AOT'd Triton
  kernel. Leave `ml8_gemv_tpl` (decode/batch=1) on hand-HIP per §4.

### 8.2 Equivalence gate

- **PPL / NIAH parity** vs the current HIP path on the validation models (the same gate used for the
  rotation/turbo4 work). Bit-exactness of the prologue is already guaranteed by Phase 1's TDD against
  the shared oracle; this gate confirms end-to-end inference parity and catches integration faults.

**Phase 3 exit gate:** PPL/NIAH parity vs HIP baseline; training and inference now share one kernel
source. **This is where unification pays off.**

---

## 9. Cross-cutting concerns

### 9.1 Execution discipline — measured, gated

All four phases are specced in detail, but **execution stays gated**: Phase 0's microbench +
correctness numbers re-confirm the substrate and the scope of 1–3 before we commit build effort to
them. Writing the spec for a phase is not a commitment to build it before its predecessor's data
lands. Re-decide at each exit gate.

### 9.2 Testing strategy

- **Kernels:** TDD, **bit-exact** against the existing torch oracles (`e4m3_roundtrip`,
  `fwht_raw`, `quantize_act_per_row`, `ml8_ref_linear`'s `_scaled_mm` reference). Code-level equality
  for quant, fp32 tolerance for VJPs.
- **Inference:** PPL / NIAH equivalence gates vs the HIP baseline.
- **Perf:** microbench (Phase 0) and trainer re-profile (Phases 1, 2).

### 9.3 Risk & rollback

- **Triton bump:** old SHA `4768da5` stays pinnable; revert is a one-line SHA change + rebuild.
- **Tuned configs (D):** additive data; removing the JSONs restores the generic fallback.
- **Stack kernel (A):** gated behind a flag with `e4m3_roundtrip`/`quantize_act_per_row` retained as
  the runtime fallback **and** the permanent test oracle — never deleted.
- **Graph capture (C):** behind a flag; eager path remains the default until parity is proven.
- **AOT swap (Phase 3):** behind a build/runtime flag; HIP `ml8_fused_rot_quant_kernel` retained until
  PPL/NIAH parity is signed off.

### 9.4 Out of scope

- The **KV kernel** (`ml8-4-kv`, 5bpv sign+mag) — separate bit-layout, untouched.
- **aiter-CK on MI300X** — a *future* absolute-peak fp8 escape hatch, gated on a microbench proving
  it; CK is untuned for gfx1201 today. Not part of this epic.
- **Backward GEMM in aiter** — does not exist upstream (confirmed across main/branches/PRs); our
  hand-written `_scaled_mm` backward (MAD-290) stays the path. No dependency taken on upstream here.

---

## 10. Open questions / to-verify (measure, don't assume)

1. **Does our current Triton (`4768da5`) actually miscompute fp8 on RDNA4?** #10458 says it *can*
   (fnuz vs OCP); turbo4 PPL/NIAH passes today, so we may not hit the broken path. **Verify
   before/after the bump** (Phase 0 §5.3) rather than assume.
2. **Where do the TFLOPS come from?** Microbench at our real shapes (Phase 0 §5.4) decides whether D
   closes the ~20% gap or whether the gap is structural.
3. **CUDA-graph feasibility:** the host-sync / dynamic-alloc inventory (Phase 2 §7.1) is itself a
   deliverable — it confirms whether full-step capture is reachable or whether we capture sub-regions.
4. **`ml8_fused_rot_quant_kernel` exact fused scope** (Phase 1 §6.1) — read `ml8.cu:881` to confirm
   whether the `h_a` leg is folded; match it so the AOT swap in Phase 3 is a true drop-in.

---

## 11. Jira structure (to file when this spec is approved)

- **Epic:** ml8 FP8 kernel-substrate unification (Triton JIT+AOT; kill the host-bound dispatch wall;
  retire training/inference prologue duplication). Links: this spec, the notes doc, the three recon
  docs, MAD-290 (predecessor).
- **Story — Phase 0:** Triton bump ≥ #10458 + AOT `--target` patch + reproducible rebuild; gfx1201
  tuned a8w8 configs; fp8 correctness verify; TFLOPS microbench. *Exit: substrate confirmed on data.*
- **Story — Phase 1:** fused rot+quant stack kernel (fwd+bwd, bit-exact TDD), e4m3 primitive,
  `layer_from_components` caching, Hessian-hook preserved, trainer re-profile. *Exit: ~3.8–4.0s + AOT-able asset.*
- **Story — Phase 2:** host-sync inventory, control-flow hoist, static mem pool, micro-step graph
  capture, captured/eager parity. *Exit: ~2.8–3.2s.*
- **Story — Phase 3:** AOT the prologue into llama.cpp, replace HIP `ml8_fused_rot_quant_kernel`,
  PPL/NIAH equivalence gate. *Exit: one kernel source for train + inference.*

Unblocks **MAD-281 #222** (4B Axis-B verdict / LR tuning), which the trainer speedup was gating.
