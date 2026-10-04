# Global Calibration Checkpoint/Resume Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `calibrate_ml8_paged.py` checkpoint/resume work correctly for BOTH the dense
and MoE strategies — not just MoE — so a crashed or killed calibration resumes from the last
completed unit instead of restarting from layer 0.

**Why now:** Resume exists today ONLY for the MoE per-expert path (skip-if-blob-exists at the
worker dispatch). The dense path has no resume at all, and naively adding "skip existing blobs"
to dense would be *silently incorrect* (see Architecture). On the MI300X gauntlet these runs are
long and on paid credits, so a mid-run crash without resume wastes money.

## Architecture — read this before writing code

The two strategies compute Hessians differently, and that dictates how resume must work:

- **MoE** precomputes ALL Hessians upfront from the ORIGINAL (bf16) model in one bulk forward
  pass (cached: `H_gate_up_per_layer` / `H_down_per_layer`, persisted to a hessian cache file).
  Experts are then quantized independently. No expert's Hessian depends on another's quantized
  weight → "skip blobs that already exist" is already correct. **Leave MoE's correctness as-is.**

- **Dense** is INTERLEAVED: for each linear it computes the Hessian against the *running,
  partially-quantized* model (resident mode copies each quantized weight back into `layer.weight`
  precisely so the next layer's Hessian sees quantized upstream — GPTQ cross-layer error
  propagation; see the comment near the resident `weight.data.copy_`). Therefore:
  - To resume dense correctly you MUST reload the *quantized* weights of every completed layer
    into the model BEFORE computing the next layer's Hessian. Skipping without reloading would
    calibrate downstream layers against the wrong (original) upstream weights.
  - Resume must trust only a **contiguous completed prefix**. If blob for unit *k* is missing but
    *k+1* exists, *k+1* was computed against a different upstream state and is STALE — discard it
    and resume at *k*. (Implement prefix detection; do not trust blobs after the first gap.)

**Reconstruction already exists — reuse it, do not reimplement:**
- `ml8_io.py` has a function (~line 69) that dequantizes a blob and absorbs BOTH the Kronecker
  rotation (`rotation.inverse`) and the AWQ scale (`absorb_awq_in_reconstruction`) back into the
  weight — i.e. it reproduces exactly what the live dense path writes to `weight.data`
  (`out["Q"]` → inverse rotation → absorb AWQ). Read `ml8_io.load_ml8_layer` and that helper.
- `reconstruct_model.py::overlay_ml8_weights` (~line 55) already walks a calibration dir, loads
  each blob, dequant+absorbs, and copies the result into the matching module's weight. The dense
  resume reload is essentially "overlay the completed-prefix blobs," so reuse this logic/pattern.

**The fidelity invariant (this is the correctness gate):**
`reconstruct_weight_from_blob(blob)` MUST equal the tensor the live loop copies into
`layer.weight.data` for that same linear, to within bf16 round-trip tolerance. `out["Q"]` is
literally `dequant(indices, centroids_per_group, scale_per_group)` (batched_gptq.py ~line 307),
so reconstructing dequant→inverse-rotation→absorb-AWQ reproduces it exactly.

## Files

- Modify: `scripts/calibration/calibrate_ml8_paged.py` — dense resume scan + prefix reload + skip
- Reuse (do not duplicate): `scripts/calibration/ml8_io.py`, `scripts/calibration/reconstruct_model.py`
- Test: `scripts/calibration/test_dense_resume.py` (new)

## Flag semantics (keep consistent across both strategies)

`--no-resume` already exists and means "ignore checkpoints, start fresh, overwrite." Keep it.
Default (resume ON) must now do the right thing for dense too. Do NOT add a new flag.

---

### Task 1: `reconstruct_weight_from_blob` helper (fidelity-tested)

**Files:**
- Modify: `scripts/calibration/ml8_io.py` (or confirm the existing ~line 69 helper already does
  this and just export/name it clearly)
- Test: `scripts/calibration/test_dense_resume.py`

- [ ] **Step 1: Write the failing test.** Build a tiny weight `W [N=32, K=64]`, a random SPD
      Hessian, run `batched_gptq_quantize(W_stack=W[None], H_stack=H[None], n_centroids=16,
      group_size=32, snap_centroids="e4m3", act_order=True, heavy_rounds=0)`. Construct a blob
      dict exactly like the dense writer does (`indices`, `centroids_per_group`, `scale_per_group`,
      no rotation/awq for this case). Assert
      `torch.allclose(reconstruct_weight_from_blob(blob), out["Q"][0], atol=1e-2)`.

- [ ] **Step 2: Run it, watch it fail** (`python -m pytest scripts/calibration/test_dense_resume.py -k reconstruct -v`).

- [ ] **Step 3: Implement** `reconstruct_weight_from_blob(blob, device="cpu", dtype=torch.float32)`
      reusing the existing ml8_io dequant+absorb path. It must: dequant `indices`/`centroids`/
      `scale` → if `blob.get("rotation")` apply `rotation.inverse` → if `blob.get("awq")` apply
      `absorb_awq_in_reconstruction`. Mirror the live order in calibrate_ml8_paged.py.

- [ ] **Step 4: Add the rotation+AWQ case to the test.** Repeat Step 1 but with a Kronecker
      rotation and an AWQ scale, building the blob the way the live loop does, and assert the
      reconstruction equals the live `weight_override`-after-inverse-rotation-and-absorb value.
      (You can capture the live value by replicating lines ~1392–1417 inline in the test.)

- [ ] **Step 5: Run tests green. Commit** (only if the user has authorized commits; otherwise stop
      at green and report).

### Task 2: Dense resume scan — completed contiguous prefix

**Files:** Modify `scripts/calibration/calibrate_ml8_paged.py` (dense loop, just before it starts
iterating linears — find where `target` linears are enumerated and the per-layer loop begins).

- [ ] **Step 1: Write the failing test.** Unit-test a new pure function
      `dense_completed_prefix(target_names, output_dir) -> int` that returns the count of leading
      linears whose blob `.pt` exists with NO gap. Given names `[a,b,c,d]` and existing blobs for
      `a,b,d` (gap at c), it must return `2` (not 3). Test in `test_dense_resume.py`.

- [ ] **Step 2: Run, fail.**

- [ ] **Step 3: Implement** `dense_completed_prefix`. Blob path = the same
      `name.replace('.', '_').replace('/', '_') + ".pt"` the writer uses. Stop at the first
      missing file.

- [ ] **Step 4: Run green.**

### Task 3: Dense resume reload + skip in the calibration loop

**Files:** Modify `scripts/calibration/calibrate_ml8_paged.py` dense path.

- [ ] **Step 1:** When `not args.no_resume`, before the per-linear loop, call
      `dense_completed_prefix`. For each linear in that prefix: load its blob, call
      `reconstruct_weight_from_blob`, and load into the model — resident: `layer.weight.data.copy_`;
      paged: set `layer.weight_override` (match how the live loop leaves it for that mode). Append
      the blob's metrics to `manifest["results"]`. Print `[resume] dense: restored N completed
      linears, resuming at <name>`.

- [ ] **Step 2:** In the per-linear loop, skip (`continue`) any linear whose index is within the
      completed prefix (its weight is already loaded). Do NOT recompute its Hessian.

- [ ] **Step 3: Integration test** (mark `@pytest.mark.slow`, guard behind an env flag / a tiny
      model so CI-light runs skip it): run a 2-layer dense calibration on the smallest available
      model to completion, record the per-linear `y_snr_db` from the manifest. Delete the last
      layer's blobs, rerun WITHOUT `--no-resume`, and assert (a) the log shows it restored the
      prefix and resumed, (b) the final manifest `y_snr_db` values for the early layers are
      identical, (c) the late layers re-quantized to the same `y_snr_db` (deterministic).

- [ ] **Step 4:** Run it. If a real small model isn't wired for tests, instead assert correctness
      structurally: a 2-linear toy `nn.Sequential` quantized via the same code path, killed after
      linear 0, resumed, produces a final state whose linear-0 weight equals the uninterrupted
      run's linear-0 weight (allclose).

### Task 4: Consistency pass + docs

- [ ] **Step 1:** Confirm MoE resume still works (don't change its skip logic). If trivial, make
      the MoE skip path also append restored metrics to the manifest for parity, but do NOT alter
      its correctness model.
- [ ] **Step 2:** Update the `--no-resume` help text to state resume now covers dense + MoE.
- [ ] **Step 3:** Add a short note to `scripts/calibration/ML8_README.md` documenting resume
      behavior and the dense "contiguous prefix only" rule.
- [ ] **Step 4:** Run the full `test_dense_resume.py` green. Report results. Do not commit unless
      the user authorizes it.

## Constraints (from the repo owner)

- Build it COMPLETELY for both strategies; never silently scope-reduce. Flag anything skipped.
- Do NOT commit unless explicitly told to.
- The running 9B calibration (`--strategy dense --resident`) reads this file but already loaded it
  into memory; your edits won't affect that process. Do not kill it.
- Never write models/GGUFs to /tmp. Tests must not leave large artifacts behind.
