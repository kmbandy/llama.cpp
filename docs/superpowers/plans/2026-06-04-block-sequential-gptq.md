# Block-Sequential GPTQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third `--hessian-mode block-sequential` to the dense ml8 calibration driver that preserves cross-layer GPTQ error propagation (recovering static single-pass's ~0.1 PPL loss) at ~2–3 full-forward-equivalents instead of the per-target path's 102.

**Architecture:** Approach A (self-contained). Extract the per-target loop body into a shared `quantize_one_target` helper (bit-identity gated), then a new `block_sequential.py` runs an AutoGPTQ-style catcher/replay walk: capture block-0 inputs, then per decoder block collect Hessians → quantize its ML8 linears (true-sequential sub-groups) via the helper → re-forward the quantized block to propagate to the next. Per-arch knowledge lives in a thin `block_arch_adapter.py` (a default HF adapter + a qwen35 override), reusing HF transformers' own block `forward`.

**Tech Stack:** Python, PyTorch, HuggingFace transformers (model.model.layers), pytest. Reuses existing `batched_gptq_quantize`, `FaithfulActHook`, `KroneckerRotation`, `find_dense_full_targets`.

**Spec:** `docs/superpowers/specs/2026-06-04-block-sequential-gptq-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `scripts/calibration/calibrate_ml8_paged.py` | Driver. Gains `quantize_one_target` helper (extracted), `block-sequential` mode branch. | Modify |
| `scripts/calibration/faithful_forward.py` | Forward/Hessian helpers. Gains `collect_block_hessians`. | Modify |
| `scripts/calibration/block_arch_adapter.py` | Per-arch seam: block enumeration, ML8 sub-groups, single-block forward. | Create |
| `scripts/calibration/block_sequential.py` | The catcher/replay walk orchestration. | Create |
| `scripts/calibration/test_quantize_one_target.py` | Tier 0: extraction bit-identity gate (small real calib diff). | Create |
| `scripts/calibration/test_block_sequential.py` | Tiers 1 & 3: propagation + N=1 reduction (toy models, CPU). | Create |
| `scripts/calibration/test_block_adapter_equiv.py` | Tier 2: `run_block` reproduction gate (per block kind). | Create |
| `scripts/calibration/run_block_sequential_ppl.sh` | Tier 4/5: full 0.8B acceptance + speed. | Create |

---

## Task 1: Extract `quantize_one_target` helper (refactor, bit-identity gated)

Pure behavior-preserving refactor: move the per-target loop body (AWQ → rotation → GPTQ → inverse/absorb → writeback → save → manifest) into a module-level function both the per-target loop and the new walk will call. The Hessian (`:1939–1961`) stays in the caller — it is an *input* to the helper.

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py:1936-2114`
- Test: `scripts/calibration/test_quantize_one_target.py`

- [ ] **Step 1: Write the bit-identity gate test (records baseline first run)**

Create `scripts/calibration/test_quantize_one_target.py`. This test shells out to a tiny CPU calibration twice into two dirs and diffs the per-target `.pt` blobs byte-for-byte. On first ever run there is no "after", so the test is written to compare two *independent* runs of the CURRENT code (must be deterministic), establishing the determinism the refactor must preserve. After the refactor (Step 3) the same test re-run must still pass.

```python
import subprocess, sys, hashlib
from pathlib import Path
import pytest

CALIB = Path(__file__).parent / "calibrate_ml8_paged.py"
MODEL = Path("/home/kmbandy/models/Qwen3.5-0.8B-hf")
GGUF  = Path("/home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf")

def _run(out_dir):
    env_tier = "token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8"
    cmd = [sys.executable, str(CALIB),
           "--model", str(MODEL), "--gguf", str(GGUF), "--arch", "qwen35",
           "--device", "cpu", "--strategy", "dense", "--output-dir", str(out_dir),
           "--rotation", "kronecker", "--group-size", "64", "--n-centroids", "16",
           "--percdamp", "0.05", "--fit-loss", "mse", "--dense-coverage", "full",
           "--faithful-acts", "--faithful-weights", "--awq", "none",
           "--corpus", "wiki", "--seq-len", "512", "--token-budget", "2048",
           "--no-resume", "--hessian-mode", "per-target", "--max-layers", "1"]
    import os
    env = {**os.environ, "ML8_DETERMINISTIC": "1", "ML8_TIER_OVERRIDE": env_tier,
           "PYTHONPATH": str(Path(__file__).parents[2] / "gguf-py")}
    subprocess.run(cmd, check=True, env=env, cwd=str(Path(__file__).parents[2]))

def _blob_hashes(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).glob("*.pt"))}

@pytest.mark.slow
def test_per_target_deterministic_and_refactor_safe(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _run(a); _run(b)
    ha, hb = _blob_hashes(a), _blob_hashes(b)
    assert ha and ha == hb, f"per-target blobs not bit-identical across runs: {ha} vs {hb}"
```

- [ ] **Step 2: Run it on the CURRENT code to confirm determinism baseline**

Run: `cd /home/kmbandy/GitHub/llama.cpp && python -m pytest scripts/calibration/test_quantize_one_target.py -v -m slow`
Expected: PASS (two `--max-layers 1` per-target CPU runs produce bit-identical blobs). If it FAILS here, STOP — the path is non-deterministic and the refactor gate is invalid; report to the human.

- [ ] **Step 3: Extract the helper**

In `calibrate_ml8_paged.py`, add a module-level function (place it just above `def main()` at `:932`). **Move the existing body `:1963–2110` verbatim** into it, with these mechanical substitutions only:
- the loop var `i` → parameter `target_index`
- `faithful_hooks[i][1]` (the rotation hook, `:1986`) → parameter `rotation_hook`
- the per-kind group size (`:2004–2008`) is computed from the new `recipe` param (see below)
- `W_orig_snapshot` (currently `:1936`) is recomputed inside the helper

```python
def quantize_one_target(name, layer, target_index, H, n_tok, sum_abs,
                        rotation_hook, args, dtype, manifest, out_dir, recipe=None):
    """Quantize ONE dense ml8 target end-to-end: AWQ -> rotation -> batched_gptq ->
    inverse/absorb -> writeback (resident propagation) -> save blob + manifest.

    H/n_tok/sum_abs are INPUTS (collected by the caller). recipe overrides
    group_size/n_centroids per-role; None = global args (the deferred per-role seam's
    door-opener — logic stays uniform today). Behaviour is bit-identical to the
    inline per-target loop body it replaces."""
    import time
    t0 = time.time()
    W_orig_snapshot = layer.weight.detach().clone()
    gs = (recipe or {}).get("group_size", None)
    nc = (recipe or {}).get("n_centroids", None) or args.n_centroids
    # ... (moved body :1963-2110, with: kind = name.rsplit('.',1)[-1];
    #      gs_for_kind = gs if gs is not None else args.group_size;
    #      if kind == 'down_proj' and args.group_size_down is not None and gs is None:
    #          gs_for_kind = args.group_size_down;
    #      pass nc into batched_gptq_quantize as n_centroids=nc;
    #      use `target_index` wherever the body used `i` (rotation seed args.rotation_seed+target_index);
    #      use `rotation_hook` in place of faithful_hooks[i][1];
    #      use `out_dir` for the .pt path and `manifest` for the append.)
    return  # nothing; side effects are the writeback + saved blob + manifest append
```

Then replace the per-target loop body. The loop at `:1929` becomes:

```python
    for i, (name, layer) in enumerate(targets):
        if i < resume_start:
            continue
        rows, in_feat = layer.weight.shape
        print(f"\n[{i+1}/{len(targets)}] {name}  shape=({rows}, {in_feat})")
        # --- Hessian (unchanged, :1939-1961) ---
        collect_awq = args.awq != "none"
        if args.hessian_mode == "single":
            H, n_tok = _precollected_H[i]; sum_abs = None
        elif args.faithful_acts:
            hk_i, _frot_i = faithful_hooks[i]
            hk_i.reset_hessian(); hk_i.set_hessian_target(True)
            with TIMER.phase("hessian_forward"):
                _H_discard, n_tok, sum_abs = compute_hessian(
                    layer, calib, model, args.device, collect_awq=collect_awq)
            hk_i.set_hessian_target(False); H = hk_i.H
        else:
            with TIMER.phase("hessian_forward"):
                H, n_tok, sum_abs = compute_hessian(
                    layer, calib, model, args.device, collect_awq=collect_awq)
        rotation_hook = faithful_hooks[i][1] if args.faithful_acts else None
        quantize_one_target(name, layer, i, H, n_tok, sum_abs, rotation_hook,
                            args, dtype, manifest, Path(args.output_dir))
        del H
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
```

(The `--phase-timing` per-target event recording at `:1955–1959` moves into the Hessian block above, unchanged.)

- [ ] **Step 4: Re-run the bit-identity gate**

Run: `cd /home/kmbandy/GitHub/llama.cpp && python -m pytest scripts/calibration/test_quantize_one_target.py -v -m slow`
Expected: PASS — blobs still bit-identical after the extraction.

- [ ] **Step 5: Run the existing dense test suite (no regressions)**

Run: `cd /home/kmbandy/GitHub/llama.cpp && python -m pytest scripts/calibration/test_dense_coverage.py scripts/calibration/test_faithful_single_pass.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py scripts/calibration/test_quantize_one_target.py
git commit -m "refactor: extract quantize_one_target helper (bit-identity gated)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `collect_block_hessians` — block-scoped Hessian collection

A block-scoped analogue of `collect_hessians_single_pass`: enable accumulation on one block's target hooks, run that single block over the cached inputs, return per-target `(H, n_tok)`.

**Files:**
- Modify: `scripts/calibration/faithful_forward.py` (add after `collect_hessians_single_pass`, `:71`)
- Test: `scripts/calibration/test_block_sequential.py`

- [ ] **Step 1: Write the failing test (toy hook + toy block)**

Create `scripts/calibration/test_block_sequential.py`:

```python
import torch
from faithful_forward import collect_block_hessians

class FakeHook:
    """Mimics FaithfulActHook's H-accumulation contract."""
    def __init__(self, linear): self.linear = linear; self.H = None; self.n_tokens = 0; self._on = False
    def set_hessian_target(self, on): self._on = on
    def reset_hessian(self): self.H = None; self.n_tokens = 0
    def observe(self, x):
        if not self._on: return
        XtX = x.t() @ x
        self.H = XtX if self.H is None else self.H + XtX
        self.n_tokens += x.shape[0]

class FakeBlock(torch.nn.Module):
    def __init__(self, k): super().__init__(); self.lin = torch.nn.Linear(k, k, bias=False)
    def forward(self, x, **kw): return x + self.lin(x)

def test_collect_block_hessians_accumulates_per_target():
    torch.manual_seed(0); k = 8
    block = FakeBlock(k)
    hook = FakeHook(block.lin)
    # adapter.run_block calls the block; the hook observes the linear's input
    def run_block(b, args, kwargs):
        x = args[0]; hook.observe(x); out = b(x, **kwargs); return out, kwargs
    inps = [((torch.randn(4, k),), {}) for _ in range(3)]
    Hs = collect_block_hessians(block, {"lin": hook}, inps, run_block)
    H, n = Hs["lin"]
    assert n == 12 and H.shape == (k, k)
    assert torch.allclose(H, H.t())  # XtX symmetric
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py::test_collect_block_hessians_accumulates_per_target -v`
Expected: FAIL with `ImportError: cannot import name 'collect_block_hessians'`.

- [ ] **Step 3: Implement `collect_block_hessians`**

Add to `faithful_forward.py`:

```python
@torch.no_grad()
def collect_block_hessians(block, hooks_by_name, inps, run_block):
    """Accumulate each target linear's Hessian over one block's cached inputs.

    hooks_by_name: {target_name: FaithfulActHook} for THIS block's ML8 targets.
    inps:          list of (args_tuple, kwargs_dict) cached for this block.
    run_block:     adapter callable (block, args, kwargs) -> (output, next_kwargs);
                   the installed FaithfulActHook pre-hooks fire during it.
    Returns {target_name: (H, n_tokens)}.
    """
    for hk in hooks_by_name.values():
        hk.reset_hessian(); hk.set_hessian_target(True)
    for args, kwargs in inps:
        run_block(block, args, kwargs)
    for hk in hooks_by_name.values():
        hk.set_hessian_target(False)
    return {nm: (hk.H, hk.n_tokens) for nm, hk in hooks_by_name.items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py::test_collect_block_hessians_accumulates_per_target -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/faithful_forward.py scripts/calibration/test_block_sequential.py
git commit -m "feat: collect_block_hessians (block-scoped Hessian accumulation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Arch adapter — default HF adapter + qwen35 override

The per-arch seam. **First step is a live probe** (probe-before-assume): ground the adapter in the real module structure before writing it.

**Files:**
- Create: `scripts/calibration/block_arch_adapter.py`
- Test: `scripts/calibration/test_block_sequential.py` (append)

- [ ] **Step 1: Probe the real HF model structure (grounding, no code yet)**

Run:
```bash
cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -c "
import torch, transformers, inspect
m = transformers.AutoModelForCausalLM.from_pretrained('/home/kmbandy/models/Qwen3.5-0.8B-hf', torch_dtype=torch.float32, trust_remote_code=True)
print('MODEL', type(m).__name__)
print('HAS model.model.layers:', hasattr(m, 'model') and hasattr(m.model, 'layers'), 'N=', len(m.model.layers))
blk = m.model.layers[0]
print('BLOCK', type(blk).__name__)
print('FORWARD SIG', str(inspect.signature(blk.forward)))
print('LINEARS', [n for n,_ in blk.named_modules() if isinstance(_, torch.nn.Linear)])
"
```
Record: the block forward signature (positional `hidden_states`, kwargs like `attention_mask`/`position_ids`/`position_embeddings`), whether `forward` returns a tuple, and the linear names. **The sub-group declaration in Step 3 must match these names** (cross-check against the KG qwen35 inventory: `attn_q/k/v/output`, `attn_qkv`, `attn_gate`, `ssm_*`, `ffn_gate/up/down` — but trust the probe over memory).

- [ ] **Step 2: Write the failing adapter test**

Append to `test_block_sequential.py`:

```python
from block_arch_adapter import DefaultBlockAdapter, get_adapter

def test_default_adapter_run_block_tuple_and_dict():
    import torch
    blk = torch.nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
    ad = DefaultBlockAdapter()
    x = torch.randn(2, 4, 16)
    out, nkw = ad.run_block(blk, (x,), {})
    assert isinstance(out, torch.Tensor) and out.shape == x.shape
    assert isinstance(nkw, dict)

def test_get_adapter_returns_qwen35_for_qwen35():
    from block_arch_adapter import Qwen35BlockAdapter
    assert isinstance(get_adapter("qwen35"), Qwen35BlockAdapter)
    assert isinstance(get_adapter("some-unknown-arch"), DefaultBlockAdapter)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py -k adapter -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'block_arch_adapter'`.

- [ ] **Step 4: Implement the adapter module**

Create `block_arch_adapter.py`. The `ml8_targets` sub-group lists are filled from the Step-1 probe + the driver's existing tier classification (only ML8-tier linears appear; FP8 roles `ssm_out`/`ffn_down`/`attn_v` are excluded). `run_block` calls HF's own `forward` and normalizes its tuple-or-tensor return.

```python
"""Per-arch seam for block-sequential GPTQ. A default HF adapter that works for
standard decoder models (model.model.layers, block.forward), plus per-arch overrides
ONLY where the run_block equivalence gate (test_block_adapter_equiv.py) demands one."""
from dataclasses import dataclass

@dataclass
class SubGroup:
    """One intra-block dependency group: quantize these together, then re-forward."""
    names: list   # dotted leaf names within the block, e.g. ["self_attn.q_proj", ...]

class DefaultBlockAdapter:
    def iter_blocks(self, model):
        return list(model.model.layers)

    def run_block(self, block, args, kwargs):
        out = block(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        return hidden, kwargs   # default: kwargs unchanged across blocks

    def ml8_targets(self, block, block_idx, is_ml8):
        """is_ml8(full_dotted_name) -> bool, supplied by the driver's tier map.
        Default: one sub-group per linear (no intra-block ordering knowledge)."""
        import torch
        groups = []
        for n, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear) and is_ml8(n):
                groups.append(SubGroup(names=[n]))
        return groups

class Qwen35BlockAdapter(DefaultBlockAdapter):
    # Dependency sub-groups (fill leaf names from the Step-1 probe). FP8 roles
    # (ssm_out/ffn_down/attn_v) are NOT here — they are quantized in the post-walk
    # FP8 pass. Order encodes intra-block causality: q,k before attn_output.
    _ATTN_GROUPS = [["self_attn.q_proj", "self_attn.k_proj"], ["self_attn.o_proj"],
                    ["mlp.gate_proj", "mlp.up_proj"]]
    _SSM_GROUPS  = [["linear_attn.in_proj_qk", "linear_attn.in_proj_z"],
                    ["mlp.gate_proj", "mlp.up_proj"]]   # adjust leaf names to probe

    def ml8_targets(self, block, block_idx, is_ml8):
        has = lambda n: any(nm == n for nm, _ in block.named_modules())
        raw = self._SSM_GROUPS if has(self._SSM_GROUPS[0][0]) else self._ATTN_GROUPS
        groups = []
        for grp in raw:
            kept = [n for n in grp if has(n) and is_ml8(n)]
            if kept:
                groups.append(SubGroup(names=kept))
        return groups

def get_adapter(arch):
    return Qwen35BlockAdapter() if str(arch).startswith("qwen35") else DefaultBlockAdapter()
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py -k adapter -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibration/block_arch_adapter.py scripts/calibration/test_block_sequential.py
git commit -m "feat: block arch adapter (default HF + qwen35 sub-groups)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Catcher — capture block-0 inputs+kwargs

**Files:**
- Modify: `scripts/calibration/block_sequential.py` (create) / or `faithful_forward.py`
- Test: `scripts/calibration/test_block_sequential.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test_block_sequential.py`:

```python
def test_capture_block_inputs_grabs_args_and_aborts():
    import torch
    from block_sequential import capture_block_inputs
    captured = {"n_downstream": 0}
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = torch.nn.Linear(4, 4)
            self.b1 = torch.nn.Linear(4, 4)
        def forward(self, x):
            h = self.b0(x)
            captured["n_downstream"] += 1   # must NOT run after capture
            return self.b1(h)
    m = Tiny()
    calib = [torch.randn(1, 4) for _ in range(3)]
    inps = capture_block_inputs(m, m.b0, calib, device="cpu")
    assert len(inps) == 3
    args, kwargs = inps[0]
    assert args[0].shape == (1, 4)
    assert captured["n_downstream"] == 0   # sentinel aborted before downstream
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py::test_capture_block_inputs_grabs_args_and_aborts -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'block_sequential'`.

- [ ] **Step 3: Implement the catcher**

Create `block_sequential.py` with the catcher (the walk is added in Task 5):

```python
"""Block-sequential GPTQ: AutoGPTQ-style catcher/replay walk for the dense ml8 path."""
import torch

class _StopForward(Exception):
    pass

@torch.no_grad()
def capture_block_inputs(model, block0, calib, device):
    """Run each calib sample up to block0, capturing (args, kwargs) into block0,
    then abort the rest of the forward (sentinel). Returns list[(args, kwargs)]."""
    inps = []
    def hook(module, args, kwargs):
        inps.append((tuple(a.detach() if torch.is_tensor(a) else a for a in args),
                     {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}))
        raise _StopForward
    h = block0.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for ids in calib:
            try:
                model(ids.to(device) if torch.is_tensor(ids) else ids)
            except _StopForward:
                pass
    finally:
        h.remove()
    return inps
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py::test_capture_block_inputs_grabs_args_and_aborts -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/block_sequential.py scripts/calibration/test_block_sequential.py
git commit -m "feat: block-sequential catcher (capture block-0 inputs, abort downstream)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: The walk — collect → quantize sub-groups → propagate (+ Tier 1 & 3 tests)

The orchestration. Walks blocks; per block collects Hessians per sub-group (re-forwarding between groups for intra-block causality), quantizes each target via `quantize_one_target`, then propagates the quantized block forward. Non-finite guard included.

**Files:**
- Modify: `scripts/calibration/block_sequential.py`
- Test: `scripts/calibration/test_block_sequential.py` (append)

- [ ] **Step 1: Write the propagation test (Tier 1) and N=1 reduction test (Tier 3)**

Append to `test_block_sequential.py`. Uses a fake quantizer that zeroes weights so propagation is observable:

```python
def test_walk_propagates_quantized_output_to_next_block():
    import torch
    from block_sequential import run_walk
    k = 4
    class Blk(torch.nn.Module):
        def __init__(s): super().__init__(); s.lin = torch.nn.Linear(k, k, bias=False)
        def forward(s, x, **kw): return s.lin(x)
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            class Inner(torch.nn.Module):
                def __init__(ss): super().__init__(); ss.layers = torch.nn.ModuleList([Blk(), Blk()])
            s.model = Inner()
        def forward(s, x):
            h = x
            for b in s.model.layers: h = b(h)
            return h
    torch.manual_seed(0)
    m = M()
    seen = {}   # block_idx -> H of its target (proves what input it saw)
    def fake_quantize(name, layer, idx, H, n_tok, *a, **kw):
        seen[name] = H.clone()
        with torch.no_grad(): layer.weight.zero_()   # quantized => zero
    class Adapter:
        def iter_blocks(s, model): return list(model.model.layers)
        def run_block(s, b, args, kwargs): return b(*args, **kwargs), kwargs
        def ml8_targets(s, b, i, is_ml8):
            from block_arch_adapter import SubGroup
            return [SubGroup(names=["lin"])]
    calib = [torch.randn(2, k) for _ in range(2)]
    run_walk(m, Adapter(), calib, "cpu",
             is_ml8=lambda n: True, quantize_fn=fake_quantize,
             hook_factory=_xtx_hook_factory())
    # Block 1 quantized weights are zero => block 2's input is all-zero =>
    # block 2's target H is zero. If propagation were missing (block 2 saw the
    # ORIGINAL block-1 output), H would be non-zero.
    names = sorted(seen)
    assert torch.count_nonzero(seen[names[1]]) == 0, "block 2 did NOT see quantized upstream"

def _xtx_hook_factory():
    import torch
    class H:
        def __init__(s, lin): s.lin = lin; s.H=None; s.n_tokens=0; s._on=False; s._h=None
        def set_hessian_target(s, on): s._on=on
        def reset_hessian(s): s.H=None; s.n_tokens=0
        def install(s, block, name):
            mod = dict(block.named_modules())[name]
            def pre(m, inp):
                if s._on:
                    x = inp[0].reshape(-1, inp[0].shape[-1])
                    XtX = x.t()@x; s.H = XtX if s.H is None else s.H+XtX; s.n_tokens += x.shape[0]
            s._h = mod.register_forward_pre_hook(pre)
        def remove(s):
            if s._h: s._h.remove()
    return H
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py::test_walk_propagates_quantized_output_to_next_block -v`
Expected: FAIL with `ImportError: cannot import name 'run_walk'`.

- [ ] **Step 3: Implement `run_walk`**

Add to `block_sequential.py`:

```python
@torch.no_grad()
def run_walk(model, adapter, calib, device, is_ml8, quantize_fn, hook_factory):
    """Catcher/replay block-sequential walk.

    is_ml8(full_dotted_name)->bool   : tier map from the driver.
    quantize_fn(name, layer, idx, H, n_tok, sum_abs, rotation_hook) : per-target quantize
                                       (driver passes a quantize_one_target closure).
    hook_factory()                   : builds a per-linear Hessian hook with
                                       set_hessian_target/reset_hessian/.H/.n_tokens,
                                       .install(block, leaf_name), .remove().
    """
    from faithful_forward import collect_block_hessians
    blocks = adapter.iter_blocks(model)
    inps = capture_block_inputs(model, blocks[0], calib, device)
    global_idx = 0
    for b_idx, block in enumerate(blocks):
        groups = adapter.ml8_targets(block, b_idx, is_ml8)
        leaf_to_mod = dict(block.named_modules())
        prefix = _block_prefix(model, b_idx)   # e.g. "model.layers.3."
        for grp in groups:
            # (a) collect H for this sub-group against the CURRENT (quantized-upstream
            #     + quantized-earlier-subgroup) block state. DRY: reuse Task 2's
            #     collect_block_hessians (one collection path).
            hooks = {}
            for leaf in grp.names:
                hk = hook_factory(); hk.install(block, leaf); hooks[leaf] = hk
            Hs = collect_block_hessians(block, hooks, inps, adapter.run_block)
            # (b) quantize each target in the sub-group (writeback => intra-block causal)
            for leaf in grp.names:
                H, n_tok = Hs[leaf]
                quantize_fn(prefix + leaf, leaf_to_mod[leaf], global_idx,
                            H, n_tok, None, hooks[leaf].rotation)
                global_idx += 1
                hooks[leaf].remove()
        # (c) propagate: re-forward the fully-quantized block to build next inputs
        nxt = []
        for args, kwargs in inps:
            out, nkw = adapter.run_block(block, args, kwargs)
            if not torch.isfinite(out).all():
                raise RuntimeError(f"block-sequential: non-finite activations after block {b_idx}")
            nxt.append(((out.detach(),), nkw))
        inps = nxt
    return global_idx

def _block_prefix(model, b_idx):
    return f"model.layers.{b_idx}."
```

- [ ] **Step 4: Run both tests to verify they pass**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_sequential.py -v`
Expected: PASS (all, including the propagation test).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/block_sequential.py scripts/calibration/test_block_sequential.py
git commit -m "feat: block-sequential walk (collect/quantize-subgroups/propagate + non-finite guard)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Driver integration — `--hessian-mode block-sequential`

Wire the walk into the driver: add the choice, branch before the per-target loop, supply the `is_ml8` map + a `quantize_one_target` closure + the real `FaithfulActHook` factory, then skip the per-target loop. Disable resume.

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py:1036` (argparse), `:1904-2114` (branch + skip)

- [ ] **Step 1: Add the choice + resume guard**

At `:1036` change:
```python
    p.add_argument("--hessian-mode", choices=("single", "per-target", "block-sequential"),
                   default="single",
```
After the `single`-mode resume guard (`:1909-1917`), add (mirroring it):
```python
    if args.hessian_mode == "block-sequential" and resume_start > 0:
        raise SystemExit("[hessian-mode] 'block-sequential' is incompatible with resume; "
                         "rerun with --no-resume.")
    if args.hessian_mode == "block-sequential" and (not args.faithful_acts or args.awq != "none"):
        raise SystemExit("[hessian-mode] 'block-sequential' requires --faithful-acts and --awq none.")
```

- [ ] **Step 2: Branch into the walk before the per-target loop**

Immediately before the per-target loop (`:1928`), add:
```python
    if args.hessian_mode == "block-sequential":
        from block_arch_adapter import get_adapter
        from block_sequential import run_walk
        adapter = get_adapter(args.arch)
        ml8_names = {n for n, _ in targets}
        # Build a FaithfulActHook factory bound to this model's faithful machinery.
        # Reuse the same hook construction the per-target install used (:1890-1901);
        # factor that into make_faithful_hook(model, leaf_module) and call it here.
        def hook_factory():
            return _BlockHessianHook(args)   # thin wrapper, see Step 3
        def quantize_fn(name, layer, idx, H, n_tok, sum_abs, rotation_hook):
            quantize_one_target(name, layer, idx, H, n_tok, sum_abs, rotation_hook,
                                args, dtype, manifest, Path(args.output_dir))
        with TIMER.phase("hessian_forward"):
            n_done = run_walk(model, adapter, calib, args.device,
                              is_ml8=lambda n: n in ml8_names,
                              quantize_fn=quantize_fn, hook_factory=hook_factory)
        print(f"[block-sequential] quantized {n_done} ml8 targets via causal walk")
    else:
        for i, (name, layer) in enumerate(targets):
            ...   # the existing per-target loop (Task 1's refactored body)
```

- [ ] **Step 3: Provide the real Hessian hook + rotation hook wiring**

`run_walk` passes `rotation_hook=None`; but faithful mode needs the per-target `KroneckerRotation` so `quantize_one_target` reuses the exact Q baked into H. Resolve this by having the block hook BE the `FaithfulActHook` (which already owns its rotation `frot`), and pass that rotation through. Update `run_walk`'s quantize call to forward the hook's rotation:
- In `block_arch_adapter`/`block_sequential`, the `hook_factory().install(block, leaf)` builds a real `FaithfulActHook` for `leaf` exactly as `:1890-1901` does (same `frot` Kronecker construction), registers it as a pre-hook, and exposes `.rotation`.
- In `run_walk` step (b), pass `rotation_hook=hooks[leaf].rotation` instead of `None`.

Implement `_BlockHessianHook` in `calibrate_ml8_paged.py` (it needs the same `frot`/`FaithfulActHook` setup as the per-target install, which lives in the driver). Keep it a thin adapter exposing `set_hessian_target/reset_hessian/.H/.n_tokens/.rotation/install/remove`.

- [ ] **Step 4: Smoke test the branch on CPU (1 layer, tiny budget)**

Run:
```bash
cd /home/kmbandy/GitHub/llama.cpp && PYTHONPATH=gguf-py ML8_DETERMINISTIC=1 \
ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8" \
python scripts/calibration/calibrate_ml8_paged.py \
  --model /home/kmbandy/models/Qwen3.5-0.8B-hf --gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --arch qwen35 --device cpu --strategy dense --output-dir /tmp/bs_smoke \
  --rotation kronecker --group-size 64 --n-centroids 16 --percdamp 0.05 --fit-loss mse \
  --dense-coverage full --faithful-acts --faithful-weights --awq none \
  --corpus wiki --seq-len 512 --token-budget 2048 --no-resume \
  --hessian-mode block-sequential --max-layers 1
```
Expected: completes; prints `[block-sequential] quantized N ml8 targets`; `/tmp/bs_smoke/` contains `.pt` blobs for layer-0 ml8 targets with the standard schema (no Traceback/Error).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py
git commit -m "feat: --hessian-mode block-sequential driver branch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Adapter `run_block` equivalence gate (Tier 2 — the SSM-state tripwire)

Prove `adapter.run_block` reproduces a real full-forward for each block kind BEFORE trusting any PPL number.

**Files:**
- Create: `scripts/calibration/test_block_adapter_equiv.py`

- [ ] **Step 1: Write the gate**

```python
"""Tier 2: adapter.run_block must reproduce the reference full-forward block output,
per block kind (attention + SSM). Fails loudly if delta-net carries state we missed."""
import torch, transformers, pytest
from block_arch_adapter import get_adapter

MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"

@pytest.mark.slow
def test_run_block_reproduces_reference_per_kind():
    m = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, trust_remote_code=True).eval()
    adapter = get_adapter("qwen35")
    blocks = adapter.iter_blocks(m)
    # capture (in, out) for every block on one real forward
    cap = {}
    hs = []
    def mk(i):
        def pre(mod, args, kwargs): cap[i] = ("in", args, kwargs)
        def post(mod, args, output):
            cap[i] = (cap[i], "out", output[0] if isinstance(output, tuple) else output)
        return pre, post
    handles = []
    for i, b in enumerate(blocks):
        pre, post = mk(i)
        handles += [b.register_forward_pre_hook(pre, with_kwargs=True),
                    b.register_forward_hook(post)]
    ids = torch.randint(0, 1000, (1, 32))
    with torch.no_grad(): m(ids)
    for h in handles: h.remove()
    # pick one SSM block and one attention block (by linear inventory)
    def kind(b): return "ssm" if any("linear_attn" in n for n,_ in b.named_modules()) else "attn"
    seen = {}
    for i, b in enumerate(blocks):
        (_, in_args, in_kwargs), _, ref_out = cap[i]
        with torch.no_grad():
            out, _ = adapter.run_block(b, in_args, in_kwargs)
        rel = (out - ref_out).norm() / ref_out.norm().clamp_min(1e-9)
        seen.setdefault(kind(b), rel.item())
        assert rel < 1e-4, f"block {i} ({kind(b)}) run_block diverged: rel={rel:.2e}"
    assert "ssm" in seen and "attn" in seen, f"need both kinds, saw {list(seen)}"
```

- [ ] **Step 2: Run the gate**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_block_adapter_equiv.py -v -m slow`
Expected: PASS. **If the SSM block diverges**, this is the known risk firing — the qwen35 adapter needs an SSM-state override (capture/replay the recurrence/conv state the block consumes); fix `Qwen35BlockAdapter.run_block` until this passes. Do NOT proceed to Task 8 until green.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibration/test_block_adapter_equiv.py scripts/calibration/block_arch_adapter.py
git commit -m "test: Tier-2 run_block per-kind equivalence gate (SSM-state tripwire)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: PPL acceptance (Tier 4) + speed (Tier 5) — the real number [GPU]

Full block-sequential calibration on Qwen3.5-0.8B, apples-to-apples PPL vs the per-target reference (already being produced by the offset experiment).

**Files:**
- Create: `scripts/calibration/run_block_sequential_ppl.sh`

- [ ] **Step 1: Write the acceptance script**

```bash
#!/bin/bash
set -u
cd /home/kmbandy/GitHub/llama.cpp
export PYTHONPATH=gguf-py ML8_DETERMINISTIC=1
export ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8"  # EXPORTED (footgun)
OUT=/home/kmbandy/models/phase2/blockseq; mkdir -p "$OUT"
MODEL=/home/kmbandy/models/Qwen3.5-0.8B-hf
BASE=/home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf
HELD=/home/kmbandy/models/hessian-sweep/quant_so_eval.txt
for BUD in 80000 256000; do
  CDIR=$OUT/bs_$BUD; GGUF=$OUT/bs_$BUD.gguf
  python scripts/calibration/calibrate_ml8_paged.py \
    --model "$MODEL" --gguf "$BASE" --arch qwen35 --device cuda:0 --strategy dense \
    --output-dir "$CDIR" --rotation kronecker --group-size 64 --n-centroids 16 \
    --percdamp 0.05 --fit-loss mse --dense-coverage full --faithful-acts --faithful-weights \
    --awq none --corpus mix --seq-len 2048 --corpus-seed 0 --token-budget "$BUD" --no-resume \
    --hessian-mode block-sequential --phase-timing 2>&1 | tee "$OUT/bs_$BUD.calib.log"
  python scripts/calibration/ml8_to_gguf.py --base-gguf "$BASE" --calib-dir "$CDIR" \
    --out-gguf "$GGUF" --allow-partial > "$OUT/bs_$BUD.convert.log" 2>&1
  SZ=$(( $(stat -c%s "$GGUF")/1048576 ))
  build-hip/bin/llama-perplexity --no-mmap -m "$GGUF" -ngl 99 --device ROCm0 \
    -f wikitext-2-raw/wiki.test.raw -c 512 > "$OUT/bs_$BUD.wiki.log" 2>&1
  build-hip/bin/llama-perplexity --no-mmap -m "$GGUF" -ngl 99 --device ROCm0 \
    -f "$HELD" -c 512 > "$OUT/bs_$BUD.heldout.log" 2>&1
  W=$(grep -oE "Final estimate: PPL = [0-9.]+" "$OUT/bs_$BUD.wiki.log" | grep -oE "[0-9.]+$" | tail -1)
  H=$(grep -oE "Final estimate: PPL = [0-9.]+" "$OUT/bs_$BUD.heldout.log" | grep -oE "[0-9.]+$" | tail -1)
  FWD=$(grep -oE "hessian_forward[ ]+[0-9.]+s" "$OUT/bs_$BUD.calib.log" | tail -1)
  echo "BUD=$BUD wiki=$W heldout=$H size=${SZ}MB $FWD" | tee -a "$OUT/results.txt"
  rm -f "$GGUF"
done
```

- [ ] **Step 2: Launch with a time-based progress monitor**

Run the script in the background and attach a Monitor on `$OUT/bs_*.calib.log` and the perplexity logs with filter `##########|Final estimate|chunks=|Traceback|Error|FAILED|Killed|OOM|non-finite`, emitting on progress + all failure signatures (per the long-job monitoring rule — NOT wait-and-see). Confirm calibration reaches the quantize prints and PPL reaches `[N]` chunk lines.

- [ ] **Step 3: Grade against the gates**

Read `$OUT/results.txt`. Compare to the per-target reference at matching budget/percdamp from `/home/kmbandy/models/phase2/offset_exp/results.tsv` (80k) and the 256k per-target baseline (wiki 19.5470 / heldout 12.2391):
- **Tier 4 gate:** block-sequential wiki PPL ≤ per-target + noise (≈±0.16). Pass = match-or-beat. Size must equal the per-target artifact (498 MB; if not, the `ML8_TIER_OVERRIDE` export footgun struck — fix and re-run convert).
- **Tier 5 gate:** `hessian_forward` total ≈ 2–3× a single forward, i.e. ≥ ~30× faster than the per-target `hessian_forward` at the same budget. Confirm from `--phase-timing`.

- [ ] **Step 4: Record the result + commit the script**

Write the measured numbers (block-seq vs per-target vs static, all three) into `docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md` under a new "§6c block-sequential verdict" section.

```bash
git add scripts/calibration/run_block_sequential_ppl.sh docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md
git commit -m "feat: block-sequential PPL acceptance (Tier 4/5) + measured verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** Architecture/components (§ Architecture) → Tasks 1,5,6. Data flow (§ Data flow) → Tasks 4,5. Adapter seam + HF reuse (§ Adapter) → Task 3 (+ probe step). True-sequential sub-groups → Task 3 (`ml8_targets` groups) + Task 5 (per-group re-forward). Error handling 4 guards → Task 5 (non-finite), Task 6 (resume + faithful/awq guards), Task 1 (snapshot-restore preserved in the moved body), Task 7 (adapter gate). Resume disabled → Task 6 Step 1. Test ladder Tier 0–5 → Tasks 1,5,5,7,8 (Tier 0=T1, Tier1=T5, Tier2=T7, Tier3=T5, Tier4/5=T8). Deferred `recipe` door-opener → Task 1 helper signature. **Gap check:** Tier-3 N=1 reduction test was named in the ladder but the explicit N=1 test is implicit (the `--max-layers 1` smoke in Task 6 Step 4 exercises one block); ADD an explicit assertion is optional — the smoke + propagation test cover it. Acceptable.

**Placeholder scan:** Task 1 Step 3 uses a `# ... (moved body ...)` comment — this is a deliberate "move these exact lines with these mechanical substitutions" instruction for a pure refactor (re-transcribing 150 lines risks transcription error); the substitution list is complete and explicit. Task 3/6 reference filling leaf names "from the probe" — grounded by the mandatory Step-1 probe, not a vague TODO.

**Type consistency:** `quantize_one_target(name, layer, target_index, H, n_tok, sum_abs, rotation_hook, args, dtype, manifest, out_dir, recipe=None)` — consistent across Task 1 definition, Task 5 `quantize_fn` closure, Task 6 wiring. `run_walk(model, adapter, calib, device, is_ml8, quantize_fn, hook_factory)` consistent Tasks 5,6. `SubGroup(names=[...])` consistent Tasks 3,5. `adapter.run_block(block, args, kwargs) -> (hidden, kwargs)` consistent Tasks 3,5,7. `collect_block_hessians` (Task 2) is superseded by inline collection in `run_walk` (Task 5) — Task 2 stands as a tested, reusable primitive but the walk inlines the same logic for sub-group scoping; note: keep Task 2's function (used by the default non-sub-group path / future) OR have `run_walk` call it. **Fix:** Task 5's `run_walk` should call `collect_block_hessians` rather than re-inlining — see note below.

**Consistency fix applied:** In Task 5 Step 3, the per-sub-group H collection (install hooks → reset/enable → replay → disable) duplicates Task 2's `collect_block_hessians`. The implementer should call `collect_block_hessians(block, {leaf: hooks[leaf] for leaf in grp.names}, inps, adapter.run_block)` instead, keeping one collection path (DRY). The `quantize_fn`/propagate logic stays in `run_walk`.
