# MAD-281 Phase E — Full-`H` GPTQ-Owned Axis B Reassignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the falsified diagonal-curvature parallel `pv_vstep` Axis B with sequential, `H⁻¹`-compensated GPTQ index reassignment against a *fixed* (QAT-tuned) codebook, reusing the hardened `batched_gptq.py`.

**Architecture:** Block-coordinate descent — Axis A (gradient/KL) owns the centroids; Axis B (full-`H` GPTQ) owns the index assignment. The core component, `batched_gptq_reassign`, is an *extraction* of the already-validated `_reassign` closure inside `batched_gptq_quantize` (it runs the GPTQ column loop quantizing each column to the nearest **fixed** centroid with `H⁻¹` error propagation, skipping the Lloyd-Max re-fit). Rung A does one re-solve after Axis A converges; rung B interleaves re-solves during training. Static offline `H = XᵀX` (per target, via the existing `compute_hessian`); online accumulation is deferred.

**Tech Stack:** PyTorch (ROCm/gfx1201 R9700), pytest. All files under `scripts/calibration/`. Spec: `docs/superpowers/specs/2026-06-12-mad281-phase-e-gptq-reassign-design.md`.

**Conventions (read before starting):**
- RAM-safe SOP for any GPU run: script self-sets `oom_score_adj=600`; launch under `systemd-run --user --scope -p MemoryHigh=9G -p MemoryMax=11G`. Single model resident. Use a time-based Monitor for jobs > 2 min, never wait-and-see.
- Targeted `git add <files>` only — never `-am`/`-A` (avoids sweeping unrelated in-tree work).
- The R9700 is `torch cuda:0 = gfx1201`; the smoke's `pick_gfx1201()` finds it by name.
- End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- **Modify `batched_gptq.py`** — lift the inner `_reassign` + its `Hinv` setup (currently lines ~409–460 inside `batched_gptq_quantize`) into a module-level `batched_gptq_reassign(W_stack, H_stack, centroids, scales, *, group_size, percdamp, act_order)`. Refactor `batched_gptq_quantize`'s act_order path to call it (DRY; equivalence by construction).
- **Modify `act_replay.py`** — add `gptq_reassign_targets(targets, H_by_name, *, percdamp, act_order)`: per `AttachedTarget`, build `E=1` stacks from `W_orig`/`centroids`/`scales`, call `batched_gptq_reassign`, copy new indices into `at.indices`. Add `collect_target_hessians(targets, calib_ids, model, dev)` wrapping the existing `compute_hessian` per target linear.
- **Modify `smoke_fp8_qat.py`** — add a `gptq` arm (rung A): run frozen Axis A to convergence, collect `H` once, one `gptq_reassign_targets`, re-measure holdout KL.
- **Modify `test_index_reassign.py` / new `test_gptq_reassign.py`** — equivalence anchor + stale-assignment-improves test.

---

## Task 1: Extract `batched_gptq_reassign` (equivalence-anchored)

**Files:**
- Modify: `scripts/calibration/batched_gptq.py` (lift `_reassign` + Hinv setup to module scope)
- Test: `scripts/calibration/test_gptq_reassign.py` (new)

- [ ] **Step 1: Write the failing equivalence test**

```python
# test_gptq_reassign.py
import torch
from batched_gptq import batched_gptq_quantize, batched_gptq_reassign

def test_reassign_matches_quantize_indices_with_fitted_centroids():
    # GPTQ with act_order produces indices via its internal fixed-centroid reassign.
    # batched_gptq_reassign fed those SAME fitted centroids/scales must reproduce them bit-for-bit.
    torch.manual_seed(0)
    E, N, K, GS = 1, 16, 64, 32
    W = torch.randn(E, N, K)
    X = torch.randn(256, K)
    H = (X.t() @ X).unsqueeze(0)                       # [E,K,K] SPD
    out = batched_gptq_quantize(W, H, n_centroids=16, group_size=GS,
                                snap_centroids="none", act_order=True)
    idx = batched_gptq_reassign(W, H, out["centroids_per_group"], out["scales"],
                                group_size=GS, act_order=True)
    assert torch.equal(idx, out["indices"]), "reassign must reproduce GPTQ act_order indices"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_gptq_reassign.py::test_reassign_matches_quantize_indices_with_fitted_centroids -x -q`
Expected: FAIL with `ImportError: cannot import name 'batched_gptq_reassign'`.

- [ ] **Step 3: Lift `_reassign` to a module-level function**

In `batched_gptq.py`, add a module-level function that generalizes the existing inner `_reassign` closure and its `Hinv` setup (the block currently at ~lines 409–460). It takes the centroids/scales as arguments instead of closing over the fitted ones:

```python
def batched_gptq_reassign(W_stack, H_stack, centroids, scales, *,
                          group_size, percdamp=0.05, act_order=True,
                          chunk_E=8):
    """GPTQ index reassignment against a FIXED codebook (no Lloyd-Max refit).

    W_stack   [E, N, K] fp32  — reconstruction target (e.g. W_orig)
    H_stack   [E, K, K] fp32  — activation Hessian
    centroids [E, n_groups_k, n_centroids] fp32 — FIXED grid (sorted), Axis-A-tuned
    scales    [E, N, n_groups_k] fp32 — FIXED per-(row,group) scale
    Returns indices [E, N, K] int8. Sequential column sweep, H^-1 compensation;
    optionally Hessian-importance (act_order) permuted. This is the exact loop
    the validated `batched_gptq_quantize(..., act_order=True)` runs internally.
    """
    dev = W_stack.device
    E, N, K = W_stack.shape
    W_stack = W_stack.float(); H = H_stack.float()
    eye_K = torch.eye(K, device=dev)
    diag_means = H.diagonal(dim1=-2, dim2=-1).mean(dim=1)
    damp = (percdamp * diag_means).view(E, 1, 1)
    gidx_orig = torch.arange(K, device=dev) // group_size
    if act_order:
        importance = H.diagonal(dim1=-2, dim2=-1).mean(0)          # [K]
        perm = torch.argsort(importance, descending=True)
    else:
        perm = torch.arange(K, device=dev)
    Hp = H[:, perm][:, :, perm]
    Hinv_p = torch.empty((E, K, K), device=dev, dtype=torch.float32)
    for cs in range(0, E, chunk_E):
        ce = min(cs + chunk_E, E)
        chol_chunk, _ = _cholesky_inv_upper(Hp[cs:ce], damp[cs:ce], eye_K.unsqueeze(0))
        Hinv_p[cs:ce] = chol_chunk
        del chol_chunk
    del Hp
    Wp = W_stack[:, :, perm].clone()
    idx_p = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
    for c in range(K):
        g = int(gidx_orig[perm[c]])
        sc = scales[:, :, g]; cg = centroids[:, g, :]
        di = (Wp[:, :, c].div(sc).unsqueeze(-1) - cg.unsqueeze(1)).abs().argmin(-1)
        q = cg.gather(1, di) * sc
        idx_p[:, :, c] = di.to(torch.int8)
        err = (Wp[:, :, c] - q) / Hinv_p[:, c, c].clamp_min(1e-30).unsqueeze(1)
        if c + 1 < K:
            Wp[:, :, c + 1:].sub_(err.unsqueeze(2) * Hinv_p[:, c, c + 1:].unsqueeze(1))
    idx_full = torch.zeros((E, N, K), dtype=torch.int8, device=dev)
    idx_full.index_copy_(2, perm, idx_p)
    return idx_full
```

- [ ] **Step 4: Refactor `batched_gptq_quantize`'s `_reassign` to delegate (DRY)**

In `batched_gptq_quantize`, replace the body of the inner `_reassign(cents, scls)` so it returns `batched_gptq_reassign(W_stack, H_stack, cents, scls, group_size=group_size, percdamp=percdamp, act_order=True, chunk_E=chunk_E)` for the index output (keep any local `Q` reconstruction the heavy loop still needs, or recompute it from the returned indices). This guarantees the extracted function and the in-place path run identical math.

- [ ] **Step 5: Run the equivalence test to verify it passes**

Run: `python -m pytest test_gptq_reassign.py -x -q` and the existing `python -m pytest test_batched_gptq.py -q` (if present) to confirm no regression.
Expected: PASS; existing GPTQ tests still green.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibration/batched_gptq.py scripts/calibration/test_gptq_reassign.py
git commit -m "feat(mad281): E.1 extract batched_gptq_reassign (fixed-codebook GPTQ)"
```

---

## Task 2: `gptq_reassign` reduces `H`-weighted reconstruction on a stale assignment

**Files:**
- Test: `scripts/calibration/test_gptq_reassign.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_reassign_lowers_H_reconstruction_after_centroid_shift():
    # Indices optimal for OLD centroids become stale when centroids move; a GPTQ
    # re-solve against the NEW centroids must lower H-weighted reconstruction error.
    torch.manual_seed(1)
    E, N, K, GS, NC = 1, 16, 64, 32, 16
    W = torch.randn(E, N, K)
    X = torch.randn(256, K); H = (X.t() @ X).unsqueeze(0)
    base = batched_gptq_quantize(W, H, n_centroids=NC, group_size=GS, act_order=True)
    cents, scales, stale_idx = base["centroids_per_group"], base["scales"], base["indices"]
    # shift centroids (simulate Axis-A tuning) — keep sorted
    new_cents = (cents * 1.15).sort(dim=-1).values
    def recon_err(idx):
        ng = K // GS
        cg = new_cents[:, torch.arange(K, device=W.device) // GS, :]    # [E,K,NC]
        sc = scales[:, :, torch.arange(K, device=W.device) // GS]       # [E,N,K]
        Wq = cg.gather(2, idx.long().unsqueeze(-1)).squeeze(-1) * sc     # [E,N,K]
        d = (W - Wq).float()
        return torch.einsum("eij,ejk,eik->e", d, H, d).sum().item()
    new_idx = batched_gptq_reassign(W, H, new_cents, scales, group_size=GS, act_order=True)
    assert recon_err(new_idx) < recon_err(stale_idx), "re-solve must lower H-reconstruction"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python -m pytest test_gptq_reassign.py::test_reassign_lowers_H_reconstruction_after_centroid_shift -x -q`
Expected: PASS (the function from Task 1 already implements this; this test pins the *behavioral contract* that motivates Phase E). If it FAILS, the extraction has a bug — fix Task 1 before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibration/test_gptq_reassign.py
git commit -m "test(mad281): E.2 pin GPTQ re-solve lowers H-reconstruction on stale indices"
```

---

## Task 3: `collect_target_hessians` — per-target static `H`

**Files:**
- Modify: `scripts/calibration/act_replay.py` (new helper)
- Test: `scripts/calibration/test_act_replay_cli.py` (extend) — CPU, tiny model

- [ ] **Step 1: Write the failing test**

```python
def test_collect_target_hessians_returns_spd_per_target():
    import torch
    from act_replay import collect_target_hessians
    from act_replay_student import AttachedTarget
    from test_act_replay_cli import _mk_state
    # one attached target wrapping a Linear-like module; 2 calib windows
    at = AttachedTarget(_mk_state(N=8, K=16, G=2))
    targets = {"blk.0.x": at}
    calib = [torch.randint(0, 5, (1, 4)), torch.randint(0, 5, (1, 4))]
    H = collect_target_hessians(targets, calib, model=_TinyModel(at), dev="cpu")
    assert "blk.0.x" in H and H["blk.0.x"].shape == (16, 16)        # [K,K]
    Hk = H["blk.0.x"]
    assert torch.allclose(Hk, Hk.t(), atol=1e-5)                    # symmetric
    assert (torch.linalg.eigvalsh(Hk) >= -1e-4).all()              # PSD
```

(Define a minimal `_TinyModel` in the test that forwards `calib` through `at`'s linear so the hook sees real input activations. If wiring a forward is awkward for the stub, implement `collect_target_hessians` to accept an explicit `{name: input_activations}` map and feed it `X` directly — choose the simpler grounded interface and make the test match.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_act_replay_cli.py::test_collect_target_hessians_returns_spd_per_target -x -q`
Expected: FAIL with `ImportError: cannot import name 'collect_target_hessians'`.

> **⚠️ CRITICAL CORRECTNESS REFINEMENT (verified 2026-06-12 — ALL 102 ml8 targets of the 0.8B carry a Kronecker rotation `Q`).** The ml8 weights are rehydrated in the **rotated** space and the forward computes `x_eff = x @ Q` before the GEMM (`act_replay_student.py`, `AttachedTarget._forward`). The GPTQ reconstruction target `W_orig` and the centroids ALSO live in that rotated space. Therefore the Hessian MUST be `XᵀX` of the **post-rotation, pre-quant** activations `x_eff_unquant = x @ Q` — **NOT** the raw linear input `x`. A naive `register_forward_pre_hook` on the host linear captures raw `x` and produces a silently wrong-space `H` (the exact "looks fine, is broken" trap). Two correct options — pick one and TDD it:
> 1. **Reuse the faithful collector** `faithful_forward.collect_hessians_single_pass` / `FaithfulActHook`, which is purpose-built to capture the rotated faithful Hessian in one pass (preferred — it's already validated for this).
> 2. **Capture inside `AttachedTarget._forward`** at the post-rotation point: add an opt-in accumulation buffer on `AttachedTarget` that, when enabled, accumulates `x_eff_unquant.T @ x_eff_unquant` for the rotated activation the weights actually consume. (`x_eff_unquant` is the rotated value BEFORE the `quantize_act_per_row` STE — match the space `W_orig` was quantized against.)
>
> The CPU TDD test must construct a target WITH a non-identity rotation and assert the collected `H` equals `(x @ Q)ᵀ(x @ Q)` (rotated), and is NOT equal to `xᵀx` (raw) — so the test actually pins the rotated space. The reference code below is the WRONG (raw) version, kept only to show the shape/structure; replace the `x = inp[0]...` capture with the rotated-activation capture per the option chosen.

- [ ] **Step 3: Implement `collect_target_hessians` (rotated-Hessian — see refinement above)**

```python
# REFERENCE STRUCTURE ONLY — the `x` capture below is RAW (wrong space). Replace it
# with the post-rotation activation x_eff_unquant = x @ Q per the refinement note.
def collect_target_hessians(targets, calib, model, dev):
    """Per-target static activation Hessian H = (1/N) sum Xeff^T Xeff over calib
    windows, where Xeff = x @ Q is the ROTATED (faithful) activation the ml8
    weights consume. Returns {name: H[K,K] fp32}. One forward pass, all targets.
    """
    import torch
    acc = {n: None for n in targets}
    cnt = {n: 0 for n in targets}
    handles = []
    def mk(n):
        def hook(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()  # RAW — WRONG; rotate first
            xtx = x.t() @ x
            acc[n] = xtx if acc[n] is None else acc[n] + xtx
            cnt[n] += x.shape[0]
        return hook
    for n, at in targets.items():
        handles.append(at.host_linear.register_forward_pre_hook(mk(n)))
    model.eval()
    with torch.no_grad():
        for ids in calib:
            model(ids.to(dev))
    for h in handles:
        h.remove()
    return {n: (acc[n] / max(1, cnt[n])).to(dev) for n in targets}
```

Note: confirm the attribute name for the host linear on `AttachedTarget` (it stores the module it replaced). If it is not `host_linear`, use the actual attribute; if `AttachedTarget` does not retain a handle to its host module, add one in `AttachedTarget.__init__` (a single `self.host_linear = linear` assignment) as part of this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_act_replay_cli.py::test_collect_target_hessians_returns_spd_per_target -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/act_replay.py scripts/calibration/act_replay_student.py scripts/calibration/test_act_replay_cli.py
git commit -m "feat(mad281): E.3 collect_target_hessians (per-target static H)"
```

---

## Task 4: `gptq_reassign_targets` integration helper

**Files:**
- Modify: `scripts/calibration/act_replay.py`
- Test: `scripts/calibration/test_act_replay_cli.py` (extend) — CPU

- [ ] **Step 1: Write the failing test**

```python
def test_gptq_reassign_targets_updates_indices_in_place():
    import torch
    from act_replay import gptq_reassign_targets
    from act_replay_student import AttachedTarget
    from test_act_replay_cli import _mk_state
    at = AttachedTarget(_mk_state(N=16, K=32, G=1))     # G=1 group → group_size=K
    targets = {"blk.0.x": at}
    before = at.indices.clone()
    H = {"blk.0.x": torch.eye(32) + 0.1 * torch.randn(32, 32) @ torch.randn(32, 32).t()}
    n = gptq_reassign_targets(targets, H, percdamp=0.05, act_order=True)
    assert at.indices.dtype == before.dtype and at.indices.shape == before.shape
    assert isinstance(n, int)                            # count of changed entries
    # with H=identity-ish and untuned centroids the assignment is near-stable; just
    # assert it ran and produced valid indices in range
    assert int(at.indices.max()) < at.centroids.shape[-1] and int(at.indices.min()) >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_act_replay_cli.py::test_gptq_reassign_targets_updates_indices_in_place -x -q`
Expected: FAIL with `ImportError: cannot import name 'gptq_reassign_targets'`.

- [ ] **Step 3: Implement `gptq_reassign_targets`**

```python
def gptq_reassign_targets(targets, H_by_name, *, percdamp=0.05, act_order=True):
    """Axis B (full-H GPTQ): re-solve each target's indices vs its CURRENT (Axis-A
    -tuned) centroids using the static Hessian. Returns total #entries changed.

    Per target: build E=1 stacks (W_orig, H, snapped centroids, scales) in the
    [E,N,K]/[E,n_groups,nc]/[E,N,n_groups] layout batched_gptq_reassign expects,
    call it, copy the new indices into at.indices.
    """
    import torch
    from batched_gptq import batched_gptq_reassign
    from centroid_quantizer import snap_to_e4m3
    total = 0
    for name, at in targets.items():
        H = H_by_name.get(name)
        if H is None:
            continue
        K = at.indices.shape[1]
        group_size = K // at.centroids.shape[0]                # n_groups = at.centroids.shape[0]
        W = at.W_orig.unsqueeze(0).float()                     # [1,N,K]
        Hs = H.unsqueeze(0).float()                            # [1,K,K]
        cents = snap_to_e4m3(at.centroids).detach().unsqueeze(0)   # [1,n_groups,nc] (sorted by build)
        scl = at.scales.detach().unsqueeze(0)                  # [1,N,n_groups]
        new_idx = batched_gptq_reassign(W, Hs, cents, scl,
                                        group_size=group_size, percdamp=percdamp,
                                        act_order=act_order)[0]    # [N,K] int8
        changed = int((new_idx.to(at.indices.dtype) != at.indices).sum().item())
        at.indices.copy_(new_idx.to(at.indices.dtype))
        total += changed
    return total
```

Note: `batched_gptq_reassign` assumes per-group centroids sorted ascending (it does nearest-centroid by absolute distance, order-independent for the argmin, but `act_order` and grid semantics match the GGUF build which stores sorted centroids). Confirm `at.centroids` rows are sorted; if not, sort + remap indices once at attach time, or sort here and remap. Keep the snap-then-use consistent with how `reassign_targets` already snaps (`snap_to_e4m3(at.centroids).detach()`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_act_replay_cli.py::test_gptq_reassign_targets_updates_indices_in_place -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/act_replay.py scripts/calibration/test_act_replay_cli.py
git commit -m "feat(mad281): E.4 gptq_reassign_targets (E=1 per-target GPTQ re-solve)"
```

---

## Task 5: Rung A — `gptq` arm in the smoke (supervised GPU)

**Files:**
- Modify: `scripts/calibration/smoke_fp8_qat.py`

- [ ] **Step 1: Add the `gptq` arm to the CLI + arm dispatch**

In `smoke_fp8_qat.py`: extend `--arms` to accept `gptq`. Add an arm spec branch:

```python
elif a == "gptq":
    arm_specs.append(("gptq", "gptq", 0.0))
```

- [ ] **Step 2: Implement the rung-A behavior in `run_arm`**

For `reassign_mode == "gptq"`: run the frozen Axis-A loop exactly as the `frozen` arm (no per-step reassign), and after the final optimizer step do ONE GPTQ re-solve, then a final holdout eval. Collect the static `H` once before the loop:

```python
# before the arm loop, once (reuse across gptq runs):
H_by_name = None  # lazy
...
# inside run_arm, when reassign_mode == "gptq":
#   (run frozen Axis A to max_steps), then:
nonlocal H_by_name
if H_by_name is None:
    from act_replay import collect_target_hessians
    H_by_name = collect_target_hessians(targets, train_w, model, dev)
from act_replay import gptq_reassign_targets
nflip = gptq_reassign_targets(targets, H_by_name, percdamp=0.05, act_order=True)
kf = holdout_kl()
print(f"[arm gptq] post-reassign KL {kf:.4f}  ({nflip} indices changed)", flush=True)
```

(Mirror the existing `restore()`/snapshot discipline so the `gptq` arm starts from the same init as the other arms.)

- [ ] **Step 3: Smoke-run the arm (supervised, RAM-safe, time-monitored)**

Run (R9700, single model resident):

```bash
systemd-run --user --scope -p MemoryHigh=9G -p MemoryMax=11G --unit mad281-e-gptq \
  python smoke_fp8_qat.py --arms frozen,gptq --eval-interval 5 --steps 30 \
  > /home/kmbandy/models/act_replay/MAD281_E_gptq.log 2>&1 &
```

Arm with a time-based Monitor on the log (`=== ARM|^[[:space:]]*[0-9]|\[arm |Traceback|Error|Killed|OOM`).
Expected: the `gptq` arm reports a **stable** post-reassign KL (no divergence — sequential `H⁻¹`-compensated GPTQ cannot blow up the way the parallel diagonal flip did).

- [ ] **Step 4: Record the verdict**

Append the `frozen` vs `gptq` numbers to `/home/kmbandy/models/act_replay/MAD281_RUNG_RESULTS.md`.
- **Gate:** `gptq` stable AND ideally final KL **< 0.0514** (the Axis-A floor) → Axis B earns its keep.
- Either outcome is a result: a win validates the two-axis product; a null bounds discrete reassignment's headroom over centroid tuning. Bank the verdict to KG and comment MAD-281.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/smoke_fp8_qat.py
git commit -m "feat(mad281): E.5 rung-A gptq arm (one post-Axis-A full-H re-solve)"
```

---

## Task 6 (conditional on Task 5 showing headroom): Rung B — interleaved re-solve

**Files:**
- Modify: `scripts/calibration/act_replay.py` (train loop), `scripts/calibration/smoke_fp8_qat.py`

> **Gate to start this task:** only build rung B if rung A (Task 5) shows the GPTQ re-solve *moves* KL meaningfully (either a clear win, or a clear-but-incomplete gain that interleaving could compound). If rung A is inert (≈ frozen), stop here and record that Axis B adds nothing over Axis A on this codebook — that is the finding, not a reason to build more.

- [ ] **Step 1: Add an interleaved-`gptq` cadence to the arm**

Add a `gptq-interleave` arm that, every `REASSIGN_INTERVAL` optimizer steps, calls `gptq_reassign_targets(targets, H_by_name, ...)` (reusing the `H` collected once). The cadence `REASSIGN_INTERVAL` is the swept hyperparameter. Re-measure holdout KL at the existing eval cadence.

- [ ] **Step 2: Smoke-run the interleave (supervised)**

```bash
systemd-run --user --scope -p MemoryHigh=9G -p MemoryMax=11G --unit mad281-e-gptqi \
  python smoke_fp8_qat.py --arms frozen,gptq,gptq-interleave --eval-interval 5 --steps 60 \
  > /home/kmbandy/models/act_replay/MAD281_E_gptq_interleave.log 2>&1 &
```

Expected: stable; compare `gptq-interleave` final KL vs `gptq` (single re-solve) vs `frozen`.

- [ ] **Step 3: Record + commit**

Append results to `MAD281_RUNG_RESULTS.md`; bank verdict; commit:

```bash
git add scripts/calibration/act_replay.py scripts/calibration/smoke_fp8_qat.py
git commit -m "feat(mad281): E.6 rung-B interleaved gptq reassignment + cadence sweep"
```

---

## Task 7 (cleanup): retire the diagonal `pv` path

**Files:**
- Modify: `scripts/calibration/index_reassign.py`, `scripts/calibration/act_replay.py`, `scripts/calibration/fp8_qat.py`

- [ ] **Step 1: Mark `pv_vstep` superseded**

Add a module docstring note to `index_reassign.py` that `pv_vstep` (diagonal parallel flip) is **superseded by Phase E `batched_gptq_reassign`** and retained only for the historical negative-result record; it is no longer wired into training. Keep `mse_estep` (still a useful nearest-centroid baseline arm).

- [ ] **Step 2: Stop consuming the diagonal stash**

Remove the now-unused `Ml8Fp8Fn.last_h` accumulation in `fp8_qat.py:backward` (the `(x*x).mean(dim=0)` stash from D.1) **only if** no arm still references it. Leave `last_dLdW` if `mse`/diagnostics use it. Run the full suite to confirm nothing breaks.

- [ ] **Step 3: Run full suite + commit**

Run: `HIP_VISIBLE_DEVICES=0 python -m pytest test_fp8_qat.py test_index_reassign.py test_act_replay_cli.py test_gptq_reassign.py -q`
Expected: all green.

```bash
git add scripts/calibration/index_reassign.py scripts/calibration/fp8_qat.py scripts/calibration/act_replay.py
git commit -m "chore(mad281): E.7 retire diagonal pv path (superseded by GPTQ reassignment)"
```

---

## Self-Review

- **Spec coverage:** §4 core component → Task 1 (+anchor) & Task 2; §5 static `H` → Task 3; §6 rung A → Task 5, rung B → Task 6; §8 tests → Tasks 1/2/3/4 (CPU TDD) + Task 5 (GPU integration); §9 deferrals (online `H`, UD comparison, scale-up) → explicitly out; §10 success criteria → Task 1 (equivalence), Task 5 (stable + gate).
- **Type consistency:** `batched_gptq_reassign(W_stack, H_stack, centroids, scales, *, group_size, percdamp, act_order, chunk_E)` returns `[E,N,K]` int8; `gptq_reassign_targets(targets, H_by_name, *, percdamp, act_order) -> int`; `collect_target_hessians(targets, calib, model, dev) -> {name: H[K,K]}`. Names used consistently across Tasks 1/3/4/5.
- **Open implementation confirmations flagged inline (not placeholders):** the `AttachedTarget` host-linear attribute name (Task 3) and centroid-sortedness (Task 4) are called out with a concrete fallback in each task.

---

## Status (2026-06-12 night) + revised next steps

**Built (E.1–E.6), committed, on `origin/master`:** `batched_gptq_reassign` (bit-identical anchor), `collect_target_hessians` (rotated faithful `H`), `gptq_reassign_targets`, rung-A `gptq` + rung-B `gptq-interleave` smoke arms, smoke parametrized `--model/--gguf`.

**0.8B verdict:** full-`H` GPTQ reassignment is **stable + correct** (0.0540 vs 0.0531 frozen floor — never diverged, unlike every pv arm) but **inert** here (9457/250M indices changed). Reason: the 0.8B is near-lossless, so Axis A barely moves the centroids → indices stay optimal → nothing to re-solve. **Not a valid test of Axis B's value** (no gap to close).

**4B test (running overnight):** `run_4b_phaseE_chain.sh` = calibrate (`--dense-coverage full --faithful-acts/weights --rotation kronecker`, 200 ml8 + 49 fp8) → `ml8_to_gguf` → smoke `frozen,gptq,gptq-interleave`. AM: read `~/models/act_replay/MAD281_4B_chain.log`.

**E.7 (retire pv): DROPPED.** Keep the working, tested pv path for future use — don't delete working code.

### THE GAP that reframes everything (kmbandy, 2026-06-12)
The QAT **product loop has never been closed on any model.** Every result is **holdout KL — a training-internal metric.** We have never re-emitted a tuned GGUF, loaded it in llama.cpp, or PPL-gated it. The product is a QAT trainer that outputs an **improved, deployable, PPL-verified** ml8 quant; the loop is **open at the re-emit boundary** everywhere.

### Revised next steps (Phase F — close the product loop)
1. **Build re-emit:** checkpoint the smoke's tuned centroids/indices for the winning arm → write a deployable GGUF (reuse `ml8_to_gguf` machinery). TDD, supervised.
2. **Real PPL gate:** `llama-perplexity --kl-divergence` of the re-emitted GGUF vs the bf16 parent (verify ml8 inference dispatch first).
3. **Close the loop cheaply on the already-calibrated 0.8B first** (fast iteration), then run the genuine end-to-end on the 4B (and eventually 35B-A3B) so the result is a **shippable PPL-gated artifact, not just a KL number.**
4. **Calibrator bug to fix:** `calibrate_ml8_paged` dense meta-init path doesn't re-tie `lm_head` for tied-embedding models → `--eval-ppl` is garbage (4B baseline printed 248320 not ~8.3). Non-blocking (per-target Y_SNR 27–28 dB proves the blobs are valid); ignore `--eval-ppl`, use `llama-perplexity`.

---

## Status (2026-06-13) — 4B attach unblocked + trainer memory teardown; backward must be rewritten

**4B verdict: still NOT achieved.** Three attach/OOM/fault blockers got fixed (below); the verdict run finally trained but only reached `frozen` step 5 (KL 0.2090→0.2131) before being stopped — **no `gptq` comparison, so Axis B's value on the 4B remains unanswered.**

**Fixes landed (all TDD, committed on `sync/upstream-2026-06-09`):**
- `3f1cf520c` — **ml8 input-column V-head reorder.** The 4B smoke crashed at attach: `ssm_out` (linear-attn out_proj) is ml8-tiered (`role_targets` BASE) but needs an axis-1 (input-column) V-head reorder, which `_apply_perm_to_ml8_entry` rejected (Phase E assumed out_proj was always fp8-frozen). The 0.8B masked it (symmetric heads → identity perm). Fix: carry the inverse perm on `AttachedTarget.col_perm` and reorder the **input activation** hf→tiled at the front of `apply_acts`, leaving W/Q/scales/centroids in GGUF order — exact (per-row act-quant commutes with a column perm) and never conjugates the Kronecker rotation. Keeps the Hessian in the GGUF-rotated basis so Axis-B GPTQ on `ssm_out` stays consistent.
- `a3ffb0941` — skip the `W_orig` mse/pv anchor unless an mse/pv arm runs.
- `e821168f3` — gate `Ml8Fp8Fn.capture_dLdW` (stop hoarding a dense fp32 `dL/dW` per layer every backward).
- `f54201910` — free the **dead bf16 host weights** (`attach_to_linear(free_host_weight=True)`; the patched forward never reads `lin.weight`) + move the arm-reset `init_idx` snapshot off VRAM. **Measured 4B QAT peak 20.4GB → 9.68GB torch-allocated** (rocm-smi 26→16; the residual ~6GB on rocm-smi is torch allocator reserve ~4GB + HIP/kernel context ~2GB, NOT the model). The 4B QAT fits in **<10GB of 32GB** — the card was never the constraint, the trainer was double-storing the model.

**GPU-safety note:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` caused a hard GPU memory-access fault ("page not present") on ROCm/RDNA4 (required a machine restart). **Never set it on this hardware** — the HIP virtual-memory path is flaky.

### THE decision that supersedes Phase F (kmbandy, 2026-06-13): rewrite the fp8 BACKWARD on-substrate
mad-lab is documented **aiter/Triton/AMD-first for fp8**, and the ml8 **forward is already a fused aiter kernel** — but `Ml8Fp8Fn.backward` (`fp8_qat.py`) is **Python-orchestrated dense fp32 PyTorch**: per layer per micro it rebuilds dense `W`, `dW_raw`, `dW_scaled`, `contrib` `[N,K]` then scatters into a 16-entry codebook (~2400 tiny ops/micro). Result: **dispatch-bound** (GPU 30–55%, CPU pegged, ~16s/micro → ~4hr for a 3-arm verdict) **and** memory-wasteful. The memory waste and the speed waste are the **same off-substrate bug**. A correct reference is fine as throwaway scaffolding, but it shipped as the trainer.

**NEXT (after compact) — brainstorm → spec → plan, then build:**
1. **Fused aiter/Triton ml8-QAT backward kernel** — compute the codebook gradient (`dcent`, `dscales`, `dx`) directly from `(dy, x, indices, gidx, scales)` with **no dense weight reconstruction** and no Python per-layer op loop. Mirror the fused forward. This dissolves both the dispatch stall and the dense-fp32 memory cost.
2. **Streaming memory model** — `init_idx` re-read from the GGUF on NVMe at arm boundaries (not hoarded in VRAM or host RAM; my host-snapshot fix was backwards — it relieved abundant 32GB VRAM by loading scarce 15GB host RAM). Weights can page like `calibrate_ml8_paged`'s `WeightPager` if scale demands.
3. Then resume the **4B Axis-B verdict** (frozen vs gptq vs gptq-interleave) — fast and lean on the fused trainer — and only after that, Phase F re-emit + PPL gate.
