# MAD-281 fp8 + joint-discrete QAT trainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real ml8 QAT trainer — fp8 forward+backward GEMMs (the deployed LUT kernel as the training forward) plus joint optimization of the continuous codebook *and* the discrete index assignment — against a KL-to-bf16-teacher objective, validated on small models.

**Architecture:** Two orthogonal, composable axes. **Axis A** (fp8 engine): an `autograd.Function` whose forward is the deployed `ml8_gemm` (`WEIGHT_FORMAT=1`) and whose backward runs fp8 dgrad/wgrad, producing `dL/dW_raw`. **Axis B** (joint discrete): an index-reassignment step (`mse_estep` baseline, `pv_vstep` linearized flip) that consumes Axis A's `dL/dW_raw`; indices are a mutable buffer the forward reads. Master codebook + Adam stay fp32. A gated **Phase 0** refreshes the upstream toolchain (pytorch/triton/aiter) and folds in keeping-them-current.

**Tech stack:** Python, PyTorch (custom multi-arch ROCm build, gfx1030+gfx1201), Triton (`ml8_runtime.ml8_gemm` / `gemm_ml8.py`), `torch._scaled_mm` (fp8), pytest. Spec: `docs/superpowers/specs/2026-06-11-mad281-fp8-qat-trainer-design.md`.

**Working dir for all commands:** `/home/kmbandy/GitHub/llama.cpp/scripts/calibration` unless noted. Run tests with `PYTHONPATH=../../gguf-py python -m pytest`.

---

## File structure

- **Create** `scripts/calibration/fp8_qat.py` — Axis A: `fp8_quant`, `Ml8Fp8Fn`, `Ml8Fp8RefFn`, `pad_to_multiple`.
- **Create** `scripts/calibration/index_reassign.py` — Axis B: `mse_estep`, `pv_vstep`, `index_reassign`.
- **Create** `scripts/calibration/test_fp8_qat.py` — Axis A unit tests.
- **Create** `scripts/calibration/test_index_reassign.py` — Axis B unit tests.
- **Modify** `scripts/calibration/act_replay_student.py` — `AttachedTarget` gains an fp8 forward path + original-weight anchor; `attach_to_linear(..., fp8=)`.
- **Modify** `scripts/calibration/act_replay.py` — CLI flags, warmup+cosine schedule, fp8 + reassign wiring, `dL/dW_raw` stash.
- **Reuse** `ml8_runtime.ml8_gemm`, `centroid_quantizer.snap_to_e4m3`, `ml8_e4m3_sim.quantize_act_per_row`, `kl_loss.{kl_topk,topk_teacher}`, `teacher_source`, holdout/ckpt/export in `act_replay.py`.

**Key reuse facts (verified):**
- `ml8_runtime.ml8_gemm(a_fp8, layer, a_scale=None, out_dtype=bf16, block_size_m=16, block_size_n=16) -> C[M,N]` computes `C = A @ W` via `WEIGHT_FORMAT=1`. Constraints: `M % block_size_m == 0`, `N % block_size_n == 0`, `K % layer.group_size == 0`. No padding handled — caller pads M.
- `Ml8Layer` (from `ml8_runtime.load_ml8_layer`) holds `centroids_fp8` (e4m3 LUT), packed indices, scales, `n_cols`/`n_rows`/`group_size`. The trainer must rebuild the e4m3 LUT from the *current trainable centroids* each forward.
- `AttachedTarget` (act_replay_student.py): `centroids` `[G,16]` fp32 Param, `scales` `[N,G]` fp32 Param, `indices` `[N,K]` uint8 buffer, `gidx` `[K]` long buffer, `weight()` returns STE dequant `[N,K]`.
- `snap_to_e4m3(t)` → e4m3-snapped fp32 tensor (lattice points). `FP8_E4M3_MAX = 448.0`, `FP8_E5M2_MAX = 57344.0`.

---

## Phase 0 — Upstream toolchain refresh (gated; does NOT block Phases A–C)

**Rationale:** aiter is already current (`69cbe3ff8`). pytorch/triton are behind, but the trainer builds fine on the *proven current* env (the fp8 probe passes). So Phase 0 runs in parallel / before the GPU phases, never blocking CPU TDD. The pytorch dirty tree is **build-generated** (hipify + AOTriton gfx120x images), so refresh = clean rebuild, not a rebase. Treat as deliberate + rollback-protected (a prior clobber cost a 3–4h rebuild).

### Task 0.1: Capture the build recipe + rollback point (do FIRST, before touching anything)

**Files:** Create `~/models/act_replay/ENV_REFRESH_RECIPE.md` (notes, not code).

- [ ] **Step 1:** Record current working state:
```bash
{ echo "## Frozen-good state $(date -I)";
  echo "torch: $(python -c 'import torch;print(torch.__version__)')";
  echo "triton: $(python -c 'import triton;print(triton.__version__)')";
  for r in pytorch triton aiter; do echo "$r HEAD: $(git -C ~/GitHub/$r rev-parse HEAD)"; done;
  echo "ROCM: $(cat /opt/rocm/.info/version 2>/dev/null)";
  echo "PYTORCH_ROCM_ARCH (from build): $(python -c 'import torch;print(torch.cuda.get_arch_list())')";
} | tee ~/models/act_replay/ENV_REFRESH_RECIPE.md
```
Expected: prints torch `2.13.0a0+gitdbae54c`, triton `3.7.0`, the three HEADs, and an arch list containing `gfx1030` and `gfx1201`.

- [ ] **Step 2:** Snapshot the installed torch so we can roll back without a rebuild. Record the site-packages torch location and back up the compiled `.so` payload:
```bash
TORCH_DIR=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')
echo "torch dir: $TORCH_DIR" >> ~/models/act_replay/ENV_REFRESH_RECIPE.md
tar czf ~/models/act_replay/torch_frozen_dbae54c.tgz -C "$(dirname "$TORCH_DIR")" "$(basename "$TORCH_DIR")" 2>&1 | tail -1
ls -lh ~/models/act_replay/torch_frozen_dbae54c.tgz
```
Expected: a multi-GB tarball exists. **This is the rollback: untar to restore the proven torch without rebuilding.**

- [ ] **Step 3:** Find and record the original build command. Search shell history + any build scripts:
```bash
grep -rhiE "PYTORCH_ROCM_ARCH|setup.py (install|develop)|python -m build" ~/.zsh_history ~/.bash_history ~/GitHub/pytorch/*.sh 2>/dev/null | sort -u | tee -a ~/models/act_replay/ENV_REFRESH_RECIPE.md
```
Expected: at least one line revealing the arch list + install mode. If empty, STOP and ask the human for the build command before proceeding — do not guess a multi-hour build invocation.

- [ ] **Step 4: Commit the recipe.**
```bash
git -C ~/GitHub/llama.cpp add docs/superpowers/plans/2026-06-11-mad281-fp8-qat-trainer.md
# (recipe lives under ~/models, not the repo; this commit is the plan itself)
git -C ~/GitHub/llama.cpp commit -m "docs: MAD-281 fp8 QAT trainer implementation plan" || true
```

### Task 0.2: Refresh aiter (already pulled — verify + pin)

- [ ] **Step 1:** Confirm aiter is at the pulled tip and the forward kernel still imports:
```bash
git -C ~/GitHub/aiter rev-parse HEAD   # expect 69cbe3ff8...
PYTHONPATH=../../gguf-py python -c "import sys; sys.path.insert(0,'.'); import ml8_runtime; print('ml8_runtime OK')"
```
Expected: `69cbe3ff8...` and `ml8_runtime OK`.

- [ ] **Step 2:** Re-run the capability probe to confirm fp8 still works post-aiter-pull:
```bash
python ~/models/act_replay/fp8_probe.py
```
Expected: `_scaled_mm` rows `OK rel_err=0.0x`, `tl.dot` rows `OK rel_err=0.000000`. If any FAIL, STOP.

### Task 0.3: pytorch + triton refresh — DEFERRED decision gate

**This task is a documented decision point, not an immediate rebuild.** The trainer build (Phases A–C) does not need it. Execute only after Phase C proves the trainer works on the current env, OR if a measured RDNA4 limitation appears.

- [ ] **Step 1:** Document the go/no-go criteria in `ENV_REFRESH_RECIPE.md`:
  - Go only if: (a) recipe + rollback tarball exist (Task 0.1), (b) a measured need (perf/correctness gap on current torch), (c) a maintenance window (not mid-build).
  - Procedure when go: `cd ~/GitHub/pytorch && git checkout -- . && git clean -fdx torch/lib/aotriton.images && git pull` → re-run the captured multi-arch build → `python ~/models/act_replay/fp8_probe.py` + a `torch.compile` smoke → on any failure, restore from `torch_frozen_dbae54c.tgz`.
  - triton: only alongside a torch rebuild (inductor coupling); same probe + rollback.

- [ ] **Step 2:** Add a recurring reminder (keep-current cadence): note in the recipe "re-evaluate upstream refresh every 2–3 weeks or when a needed RDNA4 feature lands." (No automation — a human-gated cadence, because each refresh is a supervised rebuild.)

---

## Phase A — fp8 engine (Axis A). All CPU/tiny-GPU unit-testable.

### Task A.1: `fp8_quant` — per-row amax scaling to e4m3/e5m2

**Files:** Create `fp8_qat.py`; Test `test_fp8_qat.py`.

- [ ] **Step 1: Write the failing test.**
```python
# test_fp8_qat.py
import torch
from fp8_qat import fp8_quant, FP8_E4M3_MAX, FP8_E5M2_MAX

def test_fp8_quant_e4m3_roundtrip_per_row():
    x = torch.tensor([[1.0, 2.0, 4.0], [100.0, 200.0, 400.0]])
    q, scale = fp8_quant(x, fmt="e4m3")
    assert q.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 1)                       # per-row
    recon = q.float() * scale
    assert torch.allclose(recon, x, rtol=0.1)          # fp8 rounding only
    # row amax maps to <= FP8_E4M3_MAX after scaling
    assert q.float().abs().max() <= FP8_E4M3_MAX + 1e-3

def test_fp8_quant_zero_row_guard():
    x = torch.zeros(1, 4)
    q, scale = fp8_quant(x, fmt="e4m3")
    assert scale.item() == 1.0                          # no div-by-zero
    assert q.float().abs().max() == 0.0

def test_fp8_quant_e5m2_wider_range():
    x = torch.full((1, 2), 20000.0)
    q, scale = fp8_quant(x, fmt="e5m2")
    assert q.dtype == torch.float8_e5m2
    assert torch.allclose(q.float() * scale, x, rtol=0.1)
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=../../gguf-py python -m pytest test_fp8_qat.py -q` → FAIL (`fp8_qat` missing).

- [ ] **Step 3: Implement.**
```python
# fp8_qat.py
import torch

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
_FMT = {"e4m3": (torch.float8_e4m3fn, FP8_E4M3_MAX),
        "e5m2": (torch.float8_e5m2, FP8_E5M2_MAX)}

def fp8_quant(x: torch.Tensor, fmt: str = "e4m3"):
    """Per-row (last-dim) amax scaling → fp8. Returns (x_fp8, scale[*,1])."""
    dt, fmax = _FMT[fmt]
    amax = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = (amax / fmax).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    x_fp8 = (x / scale).to(dt)
    return x_fp8, scale
```

- [ ] **Step 4: Run, verify PASS.** Expected: 3 passed.

- [ ] **Step 5: Commit.**
```bash
git add fp8_qat.py test_fp8_qat.py && git commit -m "feat(fp8): per-row amax fp8_quant (e4m3/e5m2)"
```

### Task A.2: `pad_to_multiple` — M-axis padding for the kernel's tile constraint

**Files:** `fp8_qat.py`; `test_fp8_qat.py`.

- [ ] **Step 1: Write the failing test.**
```python
from fp8_qat import pad_to_multiple

def test_pad_to_multiple_pads_and_unpads():
    x = torch.randn(20, 8)
    xp, n_pad = pad_to_multiple(x, 16, dim=0)
    assert xp.shape[0] == 32 and n_pad == 12
    assert torch.equal(xp[:20], x) and xp[20:].abs().sum() == 0
    assert torch.equal(xp[: xp.shape[0] - n_pad], x)

def test_pad_to_multiple_noop_when_aligned():
    x = torch.randn(16, 8)
    xp, n_pad = pad_to_multiple(x, 16, dim=0)
    assert n_pad == 0 and torch.equal(xp, x)
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement.**
```python
def pad_to_multiple(x: torch.Tensor, m: int, dim: int = 0):
    """Zero-pad `x` along `dim` up to a multiple of `m`. Returns (padded, n_pad)."""
    n = x.shape[dim]
    n_pad = (-n) % m
    if n_pad == 0:
        return x, 0
    shape = list(x.shape); shape[dim] = n_pad
    pad = x.new_zeros(shape)
    return torch.cat([x, pad], dim=dim), n_pad
```

- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(fp8): pad_to_multiple for kernel tile constraint"`.

### Task A.3: `Ml8Fp8RefFn` — the `_scaled_mm` reference oracle (test ground truth)

**Files:** `fp8_qat.py`; `test_fp8_qat.py`. **Note:** GPU-only (`_scaled_mm` needs CUDA/ROCm). Guard with `@pytest.mark.skipif(not torch.cuda.is_available())`.

- [ ] **Step 1: Write the failing test.**
```python
import pytest
from fp8_qat import ml8_ref_linear

@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 GEMM needs GPU")
def test_ref_linear_matches_dequant_matmul():
    dev = "cuda"
    x = torch.randn(16, 64, device=dev) * 0.3
    W = torch.randn(32, 64, device=dev) * 0.1          # [N, K]
    y = ml8_ref_linear(x, W)                            # fp8 fwd
    y_ref = x @ W.t()
    rel = (y.float() - y_ref).norm() / y_ref.norm()
    assert rel < 0.1                                    # fp8 rounding band
```

- [ ] **Step 2: Run, verify FAIL** (run on the R9700 host).

- [ ] **Step 3: Implement** (the oracle: fp8 both operands via `_scaled_mm`).
```python
def ml8_ref_linear(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Reference fp8 linear y = x @ W.T via torch._scaled_mm (test oracle)."""
    x8, sx = fp8_quant(x, "e4m3")                       # [M,K], [M,1]
    w8, sw = fp8_quant(W, "e4m3")                       # [N,K], [N,1]
    # _scaled_mm wants column-major B; compute (x @ W.T) = scaled_mm(x8, w8.T)
    out = torch._scaled_mm(x8, w8.t().contiguous().t(),
                           scale_a=sx, scale_b=sw.t(),
                           out_dtype=torch.bfloat16)
    return out
```

- [ ] **Step 4: Run, verify PASS** on the R9700.
- [ ] **Step 5: Commit** `git commit -am "feat(fp8): _scaled_mm reference oracle ml8_ref_linear"`.

### Task A.4: `Ml8Fp8Fn.forward` — deployed LUT kernel as the training forward

**Files:** `fp8_qat.py`; `test_fp8_qat.py`. GPU-only.

**Contract:** forward consumes the *trainable* fp32 `centroids[G,16]`, `scales[N,G]`, `indices[N,K]` (uint8), `gidx[K]`. It (1) snaps centroids → e4m3 (STE boundary), (2) builds the per-K-group e4m3 LUT, (3) fp8-quantizes x per-row, (4) pads M to 16, (5) calls `ml8_runtime.ml8_gemm`, (6) unpads. Saves tensors for backward.

- [ ] **Step 1: Write the failing test** (forward ≈ STE-dequant matmul, i.e. matches the existing bf16 `weight()` path within fp8 tolerance):
```python
from fp8_qat import Ml8Fp8Fn
from act_replay_student import AttachedTarget
from test_act_replay_cli import _mk_state   # reuse the tiny ml8 target builder

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fp8fn_forward_matches_ste_weight():
    dev = "cuda"
    at = AttachedTarget(_mk_state(N=32, K=128, G=2)).to(dev)
    x = torch.randn(16, 128, device=dev) * 0.3
    y = Ml8Fp8Fn.apply(x, at.centroids, at.scales, at.indices, at.gidx)
    y_ref = x @ at.weight().t()                         # bf16 STE dequant path
    rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
    assert rel < 0.12                                   # fp8 vs bf16 rounding
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement forward** (backward added in A.5 — for now a `backward` that raises is fine to keep the test forward-only; use `torch.no_grad()` in the test if needed, but prefer implementing the autograd skeleton now):
```python
from centroid_quantizer import snap_to_e4m3
import ml8_runtime

def _build_lut_layer(centroids_e4m3, scales, indices, gidx):
    """Assemble an ml8_runtime.Ml8Layer-shaped view from live trainer tensors.
    centroids_e4m3: [G,16] e4m3-as-fp32; scales: [N,G]; indices: [N,K] uint8."""
    # Reuse ml8_runtime's packing helpers; see ml8_to_packed + Ml8Layer fields.
    return ml8_runtime.layer_from_components(            # add this thin helper in ml8_runtime
        centroids_fp8=centroids_e4m3.to(torch.float8_e4m3fn),
        indices=indices, scales=scales, gidx=gidx)

class Ml8Fp8Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, centroids, scales, indices, gidx):
        cent_e4m3 = centroids + (snap_to_e4m3(centroids) - centroids).detach()
        layer = _build_lut_layer(cent_e4m3, scales, indices, gidx)
        M, K = x.reshape(-1, x.shape[-1]).shape
        xf = x.reshape(-1, K)
        x8, sx = fp8_quant(xf, "e4m3")
        x8p, n_pad = pad_to_multiple(x8, 16, dim=0)
        sxp, _ = pad_to_multiple(sx, 16, dim=0)
        y = ml8_runtime.ml8_gemm(x8p, layer, a_scale=sxp.squeeze(-1))
        y = y[: y.shape[0] - n_pad] if n_pad else y
        ctx.save_for_backward(x8, sx, cent_e4m3, scales, indices, gidx)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Task A.5")
```
**Note:** `ml8_runtime.layer_from_components` is a thin constructor to ADD in `ml8_runtime.py` (it already has `Ml8Layer`, `load_ml8_layer`, `ml8_layer_from_blob` — factor the field assembly into a from-tensors path). Add it with its own micro-test in this task.

- [ ] **Step 4: Run, verify PASS** (forward only; use `torch.no_grad()` around the call in the test if backward raises).
- [ ] **Step 5: Commit** `git commit -am "feat(fp8): Ml8Fp8Fn.forward via deployed LUT kernel + layer_from_components"`.

### Task A.5: `Ml8Fp8Fn.backward` — fp8 dgrad/wgrad + scatter to codebook

**Files:** `fp8_qat.py`; `test_fp8_qat.py`. GPU-only.

**Contract:** `backward(dy)` → `dx`, `dcentroids`, `dscales`. Quantize `dy/loss_scale` → e5m2; `dx = scaled_mm(dy8, W_e4m3)`; `dW_raw = scaled_mm(dy8.T, x8)`; stash `dL/dW_raw` on the Function's module (side-channel for Axis B); chain `dW_raw → dcentroids` (scatter-add by index) and `→ dscales`. Validate against the oracle + autograd `gradcheck` on an fp32 shadow.

- [ ] **Step 1: Write the failing test** (grads match the bf16 STE `weight()` autograd within fp8 tolerance):
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_fp8fn_backward_matches_ste_grads():
    dev = "cuda"
    s = _mk_state(N=32, K=128, G=2)
    at_a = AttachedTarget(s).to(dev); at_b = AttachedTarget(s).to(dev)
    x = torch.randn(16, 128, device=dev) * 0.3
    g = torch.randn(16, 32, device=dev)
    # fp8 path
    y_a = Ml8Fp8Fn.apply(x, at_a.centroids, at_a.scales, at_a.indices, at_a.gidx)
    y_a.backward(g)
    # bf16 STE reference path
    y_b = x @ at_b.weight().t(); y_b.backward(g)
    # cosine of centroid grads should be high (same descent direction)
    ca, cb = at_a.centroids.grad.flatten(), at_b.centroids.grad.flatten()
    cos = torch.nn.functional.cosine_similarity(ca, cb, dim=0)
    assert cos > 0.95, f"centroid grad cosine {cos:.3f}"
    assert at_a.scales.grad is not None and torch.isfinite(at_a.scales.grad).all()
```

- [ ] **Step 2: Run, verify FAIL** (`NotImplementedError`).

- [ ] **Step 3: Implement backward.**
```python
class Ml8Fp8Fn(torch.autograd.Function):
    loss_scale = 1.0           # set by the trainer
    last_dLdW = {}             # side-channel: id(centroids) -> dL/dW_raw (for Axis B)

    # ... forward as A.4 ...

    @staticmethod
    def backward(ctx, dy):
        x8, sx, cent_e4m3, scales, indices, gidx = ctx.saved_tensors
        N, K = indices.shape
        dyf = dy.reshape(-1, dy.shape[-1]) / Ml8Fp8Fn.loss_scale     # [M,N]
        dy8, sdy = fp8_quant(dyf, "e5m2")
        # reconstruct raw e4m3 weight W[N,K] = cent_e4m3[gidx, indices] * scales[:,gidx]
        cent_per_col = cent_e4m3[gidx]                                # [K,16]
        W = cent_per_col.unsqueeze(0).expand(N, -1, -1).gather(
            2, indices.long().unsqueeze(-1)).squeeze(-1) * scales[:, gidx]   # [N,K]
        x = (x8.float() * sx)                                         # dequant acts [M,K]
        dx = (dy8.float() * sdy) @ W                                  # [M,K]
        dW_raw = (dy8.float() * sdy).t() @ x                          # [N,K]
        Ml8Fp8Fn.last_dLdW[id(ctx)] = dW_raw                          # Axis B taps this
        # chain dW_raw -> dcentroids (scatter-add over (group, index)) and -> dscales
        dW_scaled = dW_raw * scales[:, gidx]                          # ∂W/∂cent path
        dcent = torch.zeros_like(cent_e4m3)                          # [G,16]
        flat_g = gidx.unsqueeze(0).expand(N, -1).reshape(-1)         # [N*K]
        flat_i = indices.long().reshape(-1)
        dcent.index_put_((flat_g, flat_i), dW_scaled.reshape(-1), accumulate=True)
        dscales = torch.zeros_like(scales)                          # [N,G]
        contrib = (dW_raw * W / scales[:, gidx].clamp_min(1e-12))   # ∂(W)/∂scale ≈ W/scale
        dscales.index_add_(1, gidx, contrib)                        # sum cols per group
        return (dx.reshape(dy.shape[:-1] + (K,)), dcent, dscales, None, None)
```
**Note:** the `dL/dW_raw` side-channel keyed by `id(ctx)` is replaced in Task C.2 by a cleaner per-target stash on the `AttachedTarget`; this keeps A.5 self-contained and testable.

- [ ] **Step 4: Run, verify PASS** (centroid grad cosine > 0.95 vs bf16 STE).
- [ ] **Step 5: Commit** `git commit -am "feat(fp8): Ml8Fp8Fn.backward fp8 dgrad/wgrad + codebook scatter"`.

---

## Phase B — joint discrete optimization (Axis B). CPU-unit-testable.

### Task B.1: `mse_estep` — re-solve indices vs the original bf16 weight

**Files:** Create `index_reassign.py`; Test `test_index_reassign.py`.

- [ ] **Step 1: Write the failing test.**
```python
# test_index_reassign.py
import torch
from index_reassign import mse_estep

def test_mse_estep_assigns_nearest_centroid():
    # 2 groups, 16 centroids; W_orig exactly equals centroid[g, j]*scale for a known j
    G, NC, N, K = 2, 16, 4, 8
    centroids = torch.randn(G, NC)
    scales = torch.rand(N, G) + 0.5
    gidx = torch.tensor([0,0,0,0,1,1,1,1])
    true_idx = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    # build W_orig from the true assignment
    cent_per_col = centroids[gidx]                                  # [K,NC]
    W = cent_per_col.unsqueeze(0).expand(N,-1,-1).gather(
        2, true_idx.long().unsqueeze(-1)).squeeze(-1) * scales[:, gidx]
    new_idx = mse_estep(W, centroids, scales, gidx)
    assert torch.equal(new_idx, true_idx)                          # recovers exact assignment
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement.**
```python
# index_reassign.py
import torch

def mse_estep(W_orig, centroids, scales, gidx):
    """Per-element argmin_j ||W_orig - centroids[g,j]*scale||^2. Returns uint8 [N,K]."""
    N, K = W_orig.shape
    cand = centroids[gidx].unsqueeze(0) * scales[:, gidx].unsqueeze(-1)   # [N,K,NC]
    err = (cand - W_orig.unsqueeze(-1)) ** 2                              # [N,K,NC]
    return err.argmin(dim=-1).to(torch.uint8)
```

- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `git add index_reassign.py test_index_reassign.py && git commit -m "feat(reassign): mse_estep nearest-centroid index solve"`.

### Task B.2: `pv_vstep` — linearized flip using dL/dW (trust-region top-K)

**Files:** `index_reassign.py`; `test_index_reassign.py`.

- [ ] **Step 1: Write the failing test** (a flip that the linearization says reduces loss is applied; predicted ΔL is negative for applied flips):
```python
from index_reassign import pv_vstep

def test_pv_vstep_applies_loss_reducing_flips():
    G, NC, N, K = 1, 4, 2, 4
    centroids = torch.tensor([[-1.0, -0.3, 0.3, 1.0]])              # [G,NC]
    scales = torch.ones(N, G)
    gidx = torch.zeros(K, dtype=torch.long)
    idx = torch.zeros(N, K, dtype=torch.uint8)                     # all point at centroid 0 (-1.0)
    # dL/dW positive everywhere → loss decreases if W decreases → want most-negative centroid (already 0)
    dLdW = torch.ones(N, K)
    new_idx, n_flips = pv_vstep(idx, dLdW, centroids, scales, gidx, frac=1.0)
    assert n_flips == 0                                            # already optimal direction
    # now dL/dW negative → loss decreases if W increases → want centroid 3 (+1.0)
    new_idx2, n2 = pv_vstep(idx, -torch.ones(N, K), centroids, scales, gidx, frac=1.0)
    assert (new_idx2 == 3).all() and n2 == N * K
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement.**
```python
def pv_vstep(indices, dLdW, centroids, scales, gidx, frac=0.1):
    """PV-tuning-style discrete step. For each element, predicted ΔL(j) =
    dLdW * (centroids[g,j]-centroids[g,cur]) * scale. Flip the top-`frac` elements
    (by predicted improvement) to their argmin-ΔL centroid. Returns (new_idx, n_flips)."""
    N, K = indices.shape
    g = gidx                                                       # [K]
    scale_col = scales[:, g]                                       # [N,K]
    cent_cols = centroids[g]                                       # [K,NC]
    cur = cent_cols.gather(1, indices.long().t()).t() if False else \
          cent_cols.unsqueeze(0).expand(N,-1,-1).gather(
              2, indices.long().unsqueeze(-1)).squeeze(-1)         # [N,K] current centroid val
    # ΔL(j) = dLdW * (cent_j - cur) * scale  → minimize over j
    dW = (cent_cols.unsqueeze(0) - cur.unsqueeze(-1)) * scale_col.unsqueeze(-1)  # [N,K,NC]
    dL = dLdW.unsqueeze(-1) * dW                                   # [N,K,NC]
    best_dL, best_j = dL.min(dim=-1)                              # [N,K]
    improve = (-best_dL).clamp_min(0.0)                           # >0 where a flip helps
    n_candidates = int(improve.numel() * frac)
    if n_candidates == 0 or improve.max() == 0:
        return indices.clone(), 0
    thresh = torch.topk(improve.reshape(-1), n_candidates).values.min()
    do_flip = (improve >= thresh) & (improve > 0)
    new_idx = torch.where(do_flip, best_j.to(torch.uint8), indices)
    return new_idx, int(do_flip.sum())
```

- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `git commit -am "feat(reassign): pv_vstep linearized trust-region index flips"`.

### Task B.3: `index_reassign` dispatcher

**Files:** `index_reassign.py`; `test_index_reassign.py`.

- [ ] **Step 1: Write the failing test.**
```python
from index_reassign import index_reassign

def test_index_reassign_dispatch_none_is_noop():
    idx = torch.randint(0,16,(4,8),dtype=torch.uint8)
    out, n = index_reassign(idx, "none", None, None, None, None, None)
    assert torch.equal(out, idx) and n == 0
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement.**
```python
def index_reassign(indices, mode, W_orig, dLdW, centroids, scales, gidx, frac=0.1):
    if mode == "none":
        return indices.clone(), 0
    if mode == "mse":
        return mse_estep(W_orig, centroids, scales, gidx), -1
    if mode == "pv":
        return pv_vstep(indices, dLdW, centroids, scales, gidx, frac=frac)
    raise ValueError(f"unknown reassign mode {mode}")
```

- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `git commit -am "feat(reassign): index_reassign dispatcher"`.

---

## Phase C — integration + GPU validation

### Task C.1: `attach_to_linear(..., fp8=True)` routes forward through `Ml8Fp8Fn`

**Files:** Modify `act_replay_student.py`; Test `test_act_replay_student.py` (reuse its harness).

- [ ] **Step 1: Write the failing test** (fp8 forward path produces finite output and reaches centroids in backward):
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_attach_fp8_forward_backprops_to_centroids(...):   # build a 1-linear stub on cuda
    at = attach_to_linear(lin, target, fp8=True)
    y = lin(x); y.sum().backward()
    assert at.centroids.grad is not None and torch.isfinite(at.centroids.grad).all()
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** — add an `fp8` flag to `AttachedTarget`/`attach_to_linear`; in the monkeypatched `_forward`, when `fp8` use `Ml8Fp8Fn.apply(x, self.centroids, self.scales, self.indices, self.gidx)` instead of `F.linear(x, self.weight())`. Keep the original bf16 weight on the target as `self.W_orig` (anchor for `mse_estep`).
- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `git commit -am "feat(student): fp8 forward path + W_orig anchor"`.

### Task C.2: `act_replay.py` — `--fp8`, `--loss-scale`, `--reassign*`, warmup+cosine

**Files:** Modify `act_replay.py`; Test `test_act_replay_cli.py`.

- [ ] **Step 1: Write the failing tests** — (a) `parse_args` exposes `fp8=False`, `loss_scale`, `reassign="none"`, `reassign_interval`, `reassign_frac`, `lr_warmup_steps=0`, and the lowered defaults `lr_cent==2e-4`, `lr_scale==2e-5`; (b) `lr_warmup_cosine(step,w,total)` ramps then decays (port the unit from the MAD-283 fix work: linear to step w, cosine to 0 at total).
```python
def test_parse_args_fp8_defaults():
    a = parse_args(["--gguf","g","--base-gguf","b","--model","m","--out-dir","o"])
    assert a.fp8 is False and a.reassign == "none" and a.lr_warmup_steps == 0
    assert a.lr_cent == 2e-4 and a.lr_scale == 2e-5

def test_lr_warmup_cosine_shape():
    from act_replay import lr_warmup_cosine
    assert lr_warmup_cosine(1,2,10) == 0.5 and lr_warmup_cosine(2,2,10) == 1.0
    assert lr_warmup_cosine(10,2,10) == 0.0 and lr_warmup_cosine(6,2,10) < 1.0
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** — add `lr_warmup_cosine(step, warmup, total)` (linear `step/warmup` to 1.0, then `0.5*(1+cos(pi*progress))` to 0); add the CLI flags + lowered lr defaults; in `train()` add `warmup_steps=0, total_steps=None` params (default = old constant-lr behavior; existing tests stay green) and scale param-group lr per step; when `args.reassign != "none"` call `index_reassign` every `reassign_interval` steps using the per-target `W_orig`, `Ml8Fp8Fn.last_dLdW` (rehomed to a per-target attr), `centroids/scales/gidx`; set `Ml8Fp8Fn.loss_scale = args.loss_scale`.
- [ ] **Step 4: Run, verify PASS** + full suite green (`python -m pytest -q`).
- [ ] **Step 5: Commit** `git commit -am "feat(trainer): fp8 + reassign + warmup/cosine CLI wiring"`.

### Task C.3: GPU smoke — 0.8B fp8 step-0 sanity + short run (the composed gate)

**Files:** Create `scripts/calibration/smoke_fp8_qat.py` (a script, not a unit test — GPU, model load). Mirrors `diag_realrun_holdout.py` structure.

- [ ] **Step 1:** Write the smoke: load Qwen3.5-0.8B bf16 + its A-cell ml8 GGUF, attach fp8 targets, in-memory bf16 teacher top-K, fixed 8-window holdout. Run three arms on the SAME data/seed: `--reassign none`, `--reassign mse`, `--reassign pv` (lr 2e-4 + warmup 5, ~60 steps). Print per-step holdout KL + per-arm final.
- [ ] **Step 2: Run under the RAM-safe SOP** (single model, `oom_score_adj=600`, `systemd-run --scope -p MemoryMax=11G`). Expected: step-0 KL ≈ the PTQ baseline; no NaN; throughput logged.
- [ ] **Step 3:** Record the three-rung result (`frozen` vs `mse` vs `pv` final holdout KL) to `~/models/act_replay/MAD281_RUNG_RESULTS.md`. **Gate:** `pv ≤ mse ≤ frozen` (each rung earns its keep) — if not, that's a finding to investigate, not a code failure.
- [ ] **Step 4: Commit** `git add smoke_fp8_qat.py && git commit -m "feat(fp8): 0.8B three-rung QAT smoke harness"`.

### Task C.4: throughput check — fp8 backward vs bf16 backward

- [ ] **Step 1:** In `smoke_fp8_qat.py`, add a `--bf16-backward` arm (existing STE path) and time N steps of each. Print steps/sec ratio.
- [ ] **Step 2: Run.** Expected: fp8 backward ≥ bf16 backward steps/sec (or document the gap). Not a correctness gate.
- [ ] **Step 3: Commit.**

---

## Self-review notes (gaps to resolve during execution, not placeholders)

- **`ml8_runtime.layer_from_components`** (Task A.4) is a new thin constructor to factor out of the existing `Ml8Layer`/`ml8_layer_from_blob` field assembly — add it with a micro-test in A.4.
- **`dscales` gradient** (A.5) uses the `W/scale` approximation; the A.5 cosine-vs-STE test is the gate. If scale-grad cosine is low, derive the exact `∂(cent·scale)/∂scale = cent` form (gather centroid value, sum per group) — the test will catch it.
- **`last_dLdW` side-channel** (A.5) is rehomed to a per-`AttachedTarget` attribute in C.2 (cleaner than `id(ctx)` keying); both forms satisfy the same Axis-B contract.
- **Phase 0 Task 0.3** is a deliberate decision gate, not an unconditional rebuild — it executes only post-Phase-C or on a measured need, with the rollback tarball from 0.1.
```
