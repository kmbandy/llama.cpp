# Calibration Pipeline Speedup — Phase 2 (Single-Pass Dense Hessian) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the dense calibration's N redundant full-corpus forwards (one per target linear) into a SINGLE forward that populates all target Hessians at once — taking a real 256k dense calibration from ~4h50m to the 1–2 h band (measured projection: ~10 min), with a bit-identical equivalence gate.

**Architecture:** The faithful-acts path already installs a persistent `FaithfulActHook` pre-hook on every target (`calibrate_ml8_paged.py:1833–1841`); each hook ALWAYS transforms activations when `enabled`, and its `_is_target` flag only gates whether it also accumulates `H += a_qᵀa_q`. Today the per-target loop calls `compute_hessian` once per target (N full forwards). We add a single-pass collector that resets + targets ALL hooks, runs ONE forward over the corpus, then reads each hook's `.H`. The old per-target path stays behind `--hessian-mode per-target` as the reference.

> **CORRECTION (2026-06-04, post code-review) — read before implementing.** The
> per-target path is **true-sequential GPTQ**: after quantizing target *k* it writes
> the quantized weight back (`weight_override` / `weight.data`) so target *k+1*'s H
> sees the **quantized upstream** (cross-layer error propagation; `:1860`, `:2050`).
> The single-pass collector builds every H against the **original** model — that's
> **static-Hessian GPTQ**, a DIFFERENT algorithm, **NOT bit-identical**. So the
> equivalence gate is **PPL-within-noise (empirical), not byte-diff.** Task 1's toy
> test proves only the collector's mechanics on independent layers (no propagation),
> not production equivalence. If Task 4's PPL holds within noise of 19.5470/12.2391,
> static single-pass wins (~100×). If it degrades, the exact fallback is
> **block-sequential GPTQ** (forward each layer once, propagate its quantized output
> forward — ~N× faster AND bit-identical), planned separately.

**Tech Stack:** Python 3, PyTorch (ROCm), pytest, the `calibrate_ml8_paged.py --strategy dense --faithful-acts` pipeline on the R9700 (gfx1201).

**Spec:** `docs/superpowers/specs/2026-06-03-calibration-pipeline-speedup-design.md`. **Phase-1 findings (the gate that ranked this):** `docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md`. Branch: `calib-pipeline-speedup`.

---

## Scope (what this plan is and is NOT)

- **IS:** the single-pass dense Hessian speed win — the one lever Phase 1 ranked. Faithful-acts path only (the production/256k path). `--awq none` (production setting; AWQ is structurally dead for Qwen per MAD-256 #1).
- **IS NOT (deferred):**
  - The hackable method core (`Codebook`/`ErrorProp` seams) → **Phase 3**, a separate refactor on `batched_gptq.py`. The dense loop already calls `batched_gptq_quantize`, so the seams are carved there; independent of this speed fix.
  - Dual-GPU `--devices` forward → not needed for the 0.8B 256k target (single-pass ~100× dwarfs dual-GPU ~2×); kept for the 35B MoE / throughput case.
  - The non-faithful (`--faithful-acts` off) dense path → legacy; keeps today's per-target behavior. Single-pass requires faithful-acts.

## File Structure

- **Modify** `scripts/calibration/faithful_forward.py` — add `collect_hessians_single_pass(hooks_by_index, calib, model, device)`: one responsibility (reset+target all hooks, one forward, return per-index `(H, n_tokens)`). ~20 lines.
- **Create** `scripts/calibration/test_faithful_single_pass.py` — pytest proving single-pass `H` == sequential `H` bit-identical on a toy model.
- **Modify** `scripts/calibration/calibrate_ml8_paged.py` — add `--hessian-mode {single,per-target}` (default `single`); call the collector once before the per-target loop in `single` mode; have the loop read precollected `H` instead of re-forwarding. Guard `single` ⇒ requires `--faithful-acts` + `--awq none`.

---

### Task 1: Single-pass collector + bit-identical equivalence unit test (CPU, TDD)

**Files:**
- Modify: `scripts/calibration/faithful_forward.py`
- Test: `scripts/calibration/test_faithful_single_pass.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/calibration/test_faithful_single_pass.py`:

```python
# scripts/calibration/test_faithful_single_pass.py
"""Single-pass dense Hessian collection must be BIT-IDENTICAL to the per-target
sequential approach, because every FaithfulActHook sees the same all-transforms-
active deterministic forward in both. Toy-model proof (CPU, fast)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
from faithful_forward import FaithfulActHook, collect_hessians_single_pass


def _toy():
    torch.manual_seed(0)
    # three linears of differing in-features; a tiny stack so a forward touches all
    d = 16
    model = nn.Sequential(nn.Linear(d, d, bias=False),
                          nn.Linear(d, d, bias=False),
                          nn.Linear(d, d, bias=False))
    targets = [(f"l{i}", model[i]) for i in range(3)]
    hooks = {}
    for i, (_n, lyr) in enumerate(targets):
        a, b = factor_for_dim(d, max_b=1024)
        rot = KroneckerRotation(h_a=random_orthogonal(a, seed=100 + i), b_dim=b)
        hk = FaithfulActHook(rot, enabled=True)
        lyr.register_forward_pre_hook(hk)
        hooks[i] = hk
    # toy "calib": a list of [tokens, d] batches
    calib = [torch.randn(5, d) for _ in range(4)]
    return model, hooks, calib


def _sequential(model, hooks, calib):
    """Replicate today's per-target path: target ONE hook at a time, one forward
    over calib per target, read that hook's H."""
    out = {}
    for i, hk in hooks.items():
        for h in hooks.values():
            h.reset_hessian(); h.set_hessian_target(False)
        hk.reset_hessian(); hk.set_hessian_target(True)
        with torch.no_grad():
            for ids in calib:
                model(ids)
        hk.set_hessian_target(False)
        out[i] = (hk.H, hk.n_tokens)
    return out


def test_single_pass_hessians_bit_identical_to_sequential():
    model, hooks, calib = _toy()
    seq = _sequential(model, hooks, calib)
    one = collect_hessians_single_pass(hooks, calib, model, device="cpu")
    assert set(one) == set(seq)
    for i in seq:
        H_seq, n_seq = seq[i]
        H_one, n_one = one[i]
        assert n_one == n_seq
        assert torch.equal(H_one, H_seq), f"target {i} Hessian not bit-identical"


def test_single_pass_resets_stale_state():
    # a prior accumulation must not leak into the single pass
    model, hooks, calib = _toy()
    for hk in hooks.values():
        hk.set_hessian_target(True)
        with torch.no_grad():
            model(calib[0])          # dirty the hooks
        hk.set_hessian_target(False)
    one = collect_hessians_single_pass(hooks, calib, model, device="cpu")
    # recompute clean sequential for comparison
    model2, hooks2, calib2 = _toy()
    seq = _sequential(model2, hooks2, calib2)
    for i in seq:
        assert torch.equal(one[i][0], seq[i][0]), f"target {i} leaked stale H"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/calibration && python3 -m pytest test_faithful_single_pass.py -v`
Expected: FAIL — `ImportError: cannot import name 'collect_hessians_single_pass'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/calibration/faithful_forward.py`:

```python
@torch.no_grad()
def collect_hessians_single_pass(hooks_by_index, calib, model, device):
    """Collect every target's faithful Hessian in ONE forward pass.

    `hooks_by_index`: {target_index: FaithfulActHook}. All hooks are already
    installed as forward pre-hooks and `enabled` (they transform activations on
    every forward regardless of target state). We reset + target ALL of them,
    run one forward over `calib`, then untarget and return each hook's H.

    Bit-identical to the per-target sequential path: in both, each hook
    accumulates a_qᵀa_q over the same deterministic all-transforms-active forward,
    over `calib` in the same order. Returns {index: (H, n_tokens)}.
    """
    for hk in hooks_by_index.values():
        hk.reset_hessian()
        hk.set_hessian_target(True)
    for ids in calib:
        model(ids.to(device))
    for hk in hooks_by_index.values():
        hk.set_hessian_target(False)
    return {i: (hk.H, hk.n_tokens) for i, hk in hooks_by_index.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/calibration && python3 -m pytest test_faithful_single_pass.py -v`
Expected: PASS — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/faithful_forward.py scripts/calibration/test_faithful_single_pass.py
git commit --no-verify -m "feat(calib): single-pass dense Hessian collector + bit-identical equivalence test"
```

---

### Task 2: Wire `--hessian-mode single` into the dense path

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py`

Locate anchors by content (line numbers drift). Make minimal additions; keep the per-target path intact as the reference.

- [ ] **Step 1: Add the CLI flag**

After the `--phase-timing` / `--forward-dtype-probe` args (added in Phase 1, near the `--strategy` block), add:

```python
    p.add_argument("--hessian-mode", choices=("single", "per-target"), default="single",
                   help="Dense Hessian collection. 'single' (default): ONE forward "
                        "populates all target Hessians (requires --faithful-acts and "
                        "--awq none); ~Nx faster, bit-identical. 'per-target': legacy "
                        "one-forward-per-target reference path.")
```

- [ ] **Step 2: Import the collector**

In the `from faithful_forward import (...)` line (currently `FaithfulActHook, assert_not_double_rotated, fp8_weight_override`), add `collect_hessians_single_pass`:

```python
from faithful_forward import FaithfulActHook, assert_not_double_rotated, fp8_weight_override, collect_hessians_single_pass  # noqa: E402  (W4A8 faithful tiers)
```

- [ ] **Step 3: Collect all Hessians in one pass before the per-target loop**

Find where the faithful pre-hooks finish installing — the line:
```python
        print(f"[faithful-acts] installed {len(faithful_hooks)} activation-e4m3 pre-hooks")
```
Immediately AFTER it (still inside `main()`, at the same indentation as that block — note it sits under `if args.faithful_acts:`, so place this at the outer `main()` indentation after that block closes; i.e. just before `# ─── Per-layer calibration loop`), add:

```python
    _precollected_H = None
    if args.hessian_mode == "single":
        if not args.faithful_acts or args.awq != "none":
            raise SystemExit("[hessian-mode] 'single' requires --faithful-acts and "
                             "--awq none; use --hessian-mode per-target otherwise.")
        print(f"\n[hessian-single] collecting H for all {len(targets)} targets in ONE "
              f"forward pass over {len(calib)} samples...")
        _t_sp = time.time()
        with TIMER.phase("hessian_forward"):
            _precollected_H = collect_hessians_single_pass(
                {i: faithful_hooks[i][0] for i in range(len(targets))},
                calib, model, args.device)
        print(f"[hessian-single] done in {time.time()-_t_sp:.1f}s "
              f"({len(_precollected_H)} Hessians, 1 forward)")
```

(`targets`, `calib`, `faithful_hooks`, `TIMER`, `model`, `args` are all in scope here.)

- [ ] **Step 4: Make the per-target loop read precollected H in single mode**

In the per-target loop, replace the Hessian block (the Phase-1-wired block that begins `collect_awq = args.awq != "none"` and contains the `_t_hess0`, the `if args.faithful_acts:` / `else:` `compute_hessian` branches, and the per-target event append) with:

```python
        collect_awq = args.awq != "none"
        _t_hess0 = time.time()
        if args.hessian_mode == "single":
            H, n_tok = _precollected_H[i]
            sum_abs = None
        elif args.faithful_acts:
            hk_i, _frot_i = faithful_hooks[i]
            hk_i.reset_hessian(); hk_i.set_hessian_target(True)
            with TIMER.phase("hessian_forward"):
                _H_discard, n_tok, sum_abs = compute_hessian(
                    layer, calib, model, args.device, collect_awq=collect_awq)
            hk_i.set_hessian_target(False)
            H = hk_i.H
        else:
            with TIMER.phase("hessian_forward"):
                H, n_tok, sum_abs = compute_hessian(
                    layer, calib, model, args.device, collect_awq=collect_awq)
        if args.phase_timing:
            TIMER._events.append({
                "label": "hessian_forward_target", "target": name,
                "n_tok": int(n_tok), "seconds": time.time() - _t_hess0,
                "shape": [int(rows), int(in_feat)]})
```

(In `single` mode the read is ~0 s, so the per-target event records the cache read; the real forward cost is the single `TIMER.phase("hessian_forward")` from Step 3, which will show `calls=1`.)

- [ ] **Step 5: Syntax + flag + Task-1 tests**

Run: `cd scripts/calibration && python3 -c "import ast; ast.parse(open('calibrate_ml8_paged.py').read()); print('OK')"` → `OK`
Run: `cd /home/kmbandy/GitHub/llama.cpp && PYTHONPATH=gguf-py python3 scripts/calibration/calibrate_ml8_paged.py --help 2>&1 | grep -E "hessian-mode"` → flag present.
Run: `cd scripts/calibration && python3 -m pytest test_faithful_single_pass.py test_calib_timing.py -q` → all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py
git commit --no-verify -m "feat(calib): --hessian-mode single (one forward, all dense targets)"
```

---

### Task 3: Real-model bit-identical equivalence gate [GPU CHECKPOINT, ~2–5 min]

**Files:** none (run + diff). **STOP — human checkpoint.** Real GPU run; flag the human before dispatching.

Runs the SAME tiny calibration twice — `per-target` (reference) and `single` — at `--max-layers 2 --token-budget 4000`, and confirms the saved per-target output blobs are **byte-identical**. (If H is bit-identical and the deterministic downstream quantize is unchanged, the `.pt` blobs are byte-identical.)

- [ ] **Step 1: Run the per-target reference**

```bash
cd /home/kmbandy/GitHub/llama.cpp
PYTHONPATH=gguf-py ML8_DETERMINISTIC=1 \
ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8" \
python3 scripts/calibration/calibrate_ml8_paged.py \
  --model /home/kmbandy/models/Qwen3.5-0.8B-hf --gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --arch qwen35 --device cuda:0 --strategy dense --output-dir /home/kmbandy/models/phase2/gate_ref \
  --rotation kronecker --group-size 64 --n-centroids 16 --percdamp 0.01 --fit-loss mse \
  --dense-coverage full --faithful-acts --faithful-weights --awq none \
  --corpus mix --seq-len 2048 --corpus-seed 0 --token-budget 4000 --no-resume \
  --max-layers 2 --hessian-mode per-target 2>&1 | tail -5
```

- [ ] **Step 2: Run single-pass to a separate dir**

Same command with `--output-dir /home/kmbandy/models/phase2/gate_single` and `--hessian-mode single`.

- [ ] **Step 3: Diff the blobs byte-for-byte**

```bash
cd /home/kmbandy/models/phase2
for f in gate_ref/*.pt; do
  b=$(basename "$f")
  if cmp -s "gate_ref/$b" "gate_single/$b"; then echo "OK  $b"; else echo "DIFF $b"; fi
done
```

Expected: every blob `OK` (byte-identical). Any `DIFF` ⇒ the single-pass Hessians are NOT bit-identical — STOP and investigate before proceeding (do not skip — this is the "rushed and broken" trap). If blob filenames differ trivially, fall back to comparing the Hessian tensors directly via a small `torch.equal` script over matching target names.

- [ ] **Step 4: Confirm the speedup shape in the gate logs**

In `gate_single`'s phase output, `hessian_forward` should show `calls=1`; in `gate_ref`, `calls=2`. Note the wall-time ratio.

---

### Task 4: Full 256k single-pass acceptance run [GPU CHECKPOINT, ~10–20 min]

**Files:** none (run + inspect). **STOP — human checkpoint.** Flag the human. This is the real acceptance: reproduce the known-good 256k PPL in the 1–2 h band (expected ~10 min), with `--phase-timing` on so it doubles as the production-scale confirmation (folds in Phase-1's deferred Task 5).

- [ ] **Step 1: Run the full 256k single-pass calibration**

```bash
cd /home/kmbandy/GitHub/llama.cpp
PYTHONPATH=gguf-py ML8_DETERMINISTIC=1 \
ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8" \
python3 scripts/calibration/calibrate_ml8_paged.py \
  --model /home/kmbandy/models/Qwen3.5-0.8B-hf --gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --arch qwen35 --device cuda:0 --strategy dense --output-dir /home/kmbandy/models/phase2/full_b256000 \
  --rotation kronecker --group-size 64 --n-centroids 16 --percdamp 0.01 --fit-loss mse \
  --dense-coverage full --faithful-acts --faithful-weights --awq none \
  --corpus mix --seq-len 2048 --corpus-seed 0 --token-budget 256000 --no-resume \
  --hessian-mode single --phase-timing \
  2>&1 | tee /home/kmbandy/models/phase2/full_b256000.log
```

- [ ] **Step 2: Convert + PPL (the equivalence acceptance)**

```bash
cd /home/kmbandy/GitHub/llama.cpp
PYTHONPATH=gguf-py python3 scripts/calibration/ml8_to_gguf.py \
  --base-gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --calib-dir /home/kmbandy/models/phase2/full_b256000 \
  --out-gguf /home/kmbandy/models/phase2/full_b256000.gguf --allow-partial
build-hip/bin/llama-perplexity --no-mmap -m /home/kmbandy/models/phase2/full_b256000.gguf \
  -ngl 99 --device ROCm0 -f wikitext-2-raw/wiki.test.raw -c 512 2>&1 | tail -3
build-hip/bin/llama-perplexity --no-mmap -m /home/kmbandy/models/phase2/full_b256000.gguf \
  -ngl 99 --device ROCm0 -f /home/kmbandy/models/hessian-sweep/quant_so_eval.txt -c 512 2>&1 | tail -3
```

Expected: wiki `PPL = 19.54xx` (within noise of 19.5470) and held-out `12.23xx` (within noise of 12.2391). This is the acceptance gate.

- [ ] **Step 3: Confirm the speedup**

Run: `cd scripts/calibration && python3 analyze_phase_timing.py /home/kmbandy/models/phase2/full_b256000`
Expected: `hessian_forward` `calls=1`, total wall time in the 1–2 h band (expected ~10 min). Compare against the 17,430 s baseline → report the realized speedup.

---

## Self-Review

**Spec/Phase-1-findings coverage:**
- Findings §5 "single-pass dense Hessian — the only lever needed" → Tasks 1–2 implement, Tasks 3–4 gate. ✔
- Findings §6 "bit-identical under determinism" → Task 1 toy proof + Task 3 real-model byte-diff. ✔
- Spec §4 "256k acceptance reproduces wiki 19.5470 / held-out 12.2391 in 1–2 h" → Task 4. ✔
- Spec §4 dual-GPU + §6 fp32 → deliberately NOT here (Phase-1 findings retired them for this target); method core → Phase 3. Flagged in Scope. ✔

**Placeholder scan:** none. Task 4's PPL numbers are acceptance targets to match, not placeholders.

**Type/name consistency:** `collect_hessians_single_pass(hooks_by_index, calib, model, device)` returning `{i: (H, n_tokens)}` is defined in Task 1 and consumed identically in Task 2 Steps 3–4. `--hessian-mode {single,per-target}` used consistently in Tasks 2–4. `FaithfulActHook` attributes (`reset_hessian`, `set_hessian_target`, `.H`, `.n_tokens`) match `faithful_forward.py:26–54`. ✔
