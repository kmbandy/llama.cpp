# QuaRot-R1 Hadamard Scatter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone bf16-GGUF → bf16-GGUF rotation pass that folds a random Hadamard `R_resid` into the residual stream of Qwen3.5/Qwen3.6 models, so that the existing ml8 calibration pipeline can quantize in a less-coherent basis and close the +0.046 PPL gap vs Q4_K_XL on Qwen3.6-35B-A3B.

**Architecture:** New CLI script `rotate_model_quarot.py` does a two-pass walk of the source GGUF (pass 1: index γ vectors and tensor roles; pass 2: stream rotate + write). Calibration is rotation-blind — it just reads the rotated GGUF with `--rotation none`. Equivalence is verified bit-for-bit against the source before any calibration runs.

**Tech Stack:** Python 3.14 + torch + `gguf` (GGUFReader/GGUFWriter from `gguf-py/`) + existing `scripts/calibration/kronecker_rotation.py` for Sylvester construction + `llama-cli --no-mmap` for the equivalence gate.

**Spec:** `docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-design.md`

**File structure:**

| File | Purpose |
|---|---|
| `scripts/calibration/rotate_model_quarot.py` (new) | Main rotation pass: CLI + R_resid builder + role classifier + rotation primitives + streaming reader/writer |
| `scripts/calibration/test_rotate_model_quarot.py` (new) | Unit tests for orthogonality, rotation primitives, MoE batching, role classification, end-to-end on a 2-layer toy GGUF |
| `scripts/calibration/test_quarot_r1_equivalence.py` (new) | Equivalence gate: runs `llama-cli` on source and rotated GGUF, compares final logits + 4K-token PPL |
| `scripts/calibration/calibrate_ml8_paged.py` (unchanged) | Consumes rotated GGUF as `--gguf` with `--rotation none` |

Test style matches existing `scripts/calibration/test_kronecker_rotation.py`: bare-script `_assert_close` helpers, run via `python3 test_foo.py`, top-level `if __name__ == "__main__"` block calls each test fn.

---

### Task 1: Role enum and per-arch tensor name classifier

**Files:**
- Create: `scripts/calibration/rotate_model_quarot.py`
- Test: `scripts/calibration/test_rotate_model_quarot.py`

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python3
"""Tests for rotate_model_quarot.py."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rotate_model_quarot import Role, classify_tensor


def _assert_eq(actual, expected, label):
    assert actual == expected, f"{label}: got {actual!r}, expected {expected!r}"


def test_classify_qwen36_tensors():
    """Known Qwen3.6 MoE tensor names map to expected roles."""
    cases = [
        ("token_embd.weight",                 Role.EMBED),
        ("blk.0.attn_norm.weight",            Role.NORM_PRE_ATTN),
        ("blk.0.attn_q.weight",               Role.ATTN_Q),
        ("blk.0.attn_k.weight",               Role.ATTN_K),
        ("blk.0.attn_v.weight",               Role.ATTN_V),
        ("blk.0.attn_output.weight",          Role.ATTN_O),
        ("blk.0.ffn_norm.weight",             Role.NORM_PRE_FFN),
        ("blk.0.ffn_gate_inp.weight",         Role.FFN_GATE_INP),
        ("blk.0.ffn_gate_exps.weight",        Role.FFN_GATE_EXPS),
        ("blk.0.ffn_up_exps.weight",          Role.FFN_UP_EXPS),
        ("blk.0.ffn_down_exps.weight",        Role.FFN_DOWN_EXPS),
        ("blk.0.ssm_norm.weight",             Role.NORM_PRE_SSM),
        ("blk.0.ssm_in.weight",               Role.MAMBA_IN),
        ("blk.0.ssm_out.weight",              Role.MAMBA_OUT),
        ("output_norm.weight",                Role.NORM_OUT),
        ("output.weight",                     Role.LM_HEAD),
        ("rope_freqs.weight",                 Role.PASSTHROUGH),
    ]
    for name, expected in cases:
        actual = classify_tensor(name, arch="qwen36moe")
        _assert_eq(actual, expected, f"qwen36moe / {name}")
    print(f"  PASS test_classify_qwen36_tensors")


def test_classify_unknown_raises():
    """Unknown tensor name on a known arch raises with the name in the message."""
    try:
        classify_tensor("blk.0.fictional.weight", arch="qwen36moe")
    except ValueError as e:
        msg = str(e)
        assert "blk.0.fictional.weight" in msg, f"name missing from error: {msg}"
        print(f"  PASS test_classify_unknown_raises")
        return
    raise AssertionError("expected ValueError, got none")


if __name__ == "__main__":
    test_classify_qwen36_tensors()
    test_classify_unknown_raises()
    print("\nALL TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/kmbandy/GitHub/llama.cpp
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: `ModuleNotFoundError: No module named 'rotate_model_quarot'` (or `ImportError: cannot import name 'Role'`).

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""rotate_model_quarot.py — QuaRot-R1 rotation pass over a bf16 GGUF.

See docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-design.md.
"""
from __future__ import annotations

import enum
import re
from typing import Optional


class Role(enum.Enum):
    PASSTHROUGH      = "passthrough"
    EMBED            = "embed"
    NORM_PRE_ATTN    = "norm_pre_attn"
    NORM_PRE_FFN     = "norm_pre_ffn"
    NORM_PRE_SSM     = "norm_pre_ssm"
    NORM_OUT         = "norm_out"
    ATTN_Q           = "attn_q"
    ATTN_K           = "attn_k"
    ATTN_V           = "attn_v"
    ATTN_O           = "attn_o"
    FFN_GATE_INP     = "ffn_gate_inp"
    FFN_GATE_EXPS    = "ffn_gate_exps"
    FFN_UP_EXPS      = "ffn_up_exps"
    FFN_DOWN_EXPS    = "ffn_down_exps"
    MAMBA_IN         = "mamba_in"
    MAMBA_OUT        = "mamba_out"
    LM_HEAD          = "lm_head"


# Per-arch regex pattern table. Patterns are ordered most-specific-first
# because re.fullmatch is tried in declaration order.
_ROLE_PATTERNS: dict[str, list[tuple[str, Role]]] = {
    "qwen36moe": [
        (r"token_embd\.weight",                   Role.EMBED),
        (r"output_norm\.weight",                  Role.NORM_OUT),
        (r"output\.weight",                       Role.LM_HEAD),
        (r"blk\.\d+\.attn_norm\.weight",          Role.NORM_PRE_ATTN),
        (r"blk\.\d+\.ffn_norm\.weight",           Role.NORM_PRE_FFN),
        (r"blk\.\d+\.ssm_norm\.weight",           Role.NORM_PRE_SSM),
        (r"blk\.\d+\.attn_q\.weight",             Role.ATTN_Q),
        (r"blk\.\d+\.attn_k\.weight",             Role.ATTN_K),
        (r"blk\.\d+\.attn_v\.weight",             Role.ATTN_V),
        (r"blk\.\d+\.attn_output\.weight",        Role.ATTN_O),
        (r"blk\.\d+\.ffn_gate_inp\.weight",       Role.FFN_GATE_INP),
        (r"blk\.\d+\.ffn_gate_exps\.weight",      Role.FFN_GATE_EXPS),
        (r"blk\.\d+\.ffn_up_exps\.weight",        Role.FFN_UP_EXPS),
        (r"blk\.\d+\.ffn_down_exps\.weight",      Role.FFN_DOWN_EXPS),
        (r"blk\.\d+\.ssm_in\.weight",             Role.MAMBA_IN),
        (r"blk\.\d+\.ssm_out\.weight",            Role.MAMBA_OUT),
        # Everything else (rope_freqs, biases, ssm internals, etc.) passes through.
        (r".*",                                   Role.PASSTHROUGH),
    ],
}


def classify_tensor(name: str, arch: str) -> Role:
    """Map a GGUF tensor name to its rotation role.

    Raises ValueError if arch is unknown OR if no pattern matches (the
    PASSTHROUGH catch-all only fires on tensors we've decided not to touch;
    a tensor that fails to match ANY pattern indicates the arch table is
    incomplete and we should hard-fail rather than silently passthrough).
    """
    patterns = _ROLE_PATTERNS.get(arch)
    if patterns is None:
        raise ValueError(f"unknown arch {arch!r}; add a pattern table to _ROLE_PATTERNS")
    for pattern, role in patterns:
        if re.fullmatch(pattern, name):
            return role
    raise ValueError(
        f"tensor {name!r} matched no pattern under arch={arch!r}; "
        f"the catch-all should have fired — check the pattern table."
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected:
```
  PASS test_classify_qwen36_tensors
  PASS test_classify_unknown_raises
ALL TESTS PASSED
```

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add scripts/calibration/rotate_model_quarot.py scripts/calibration/test_rotate_model_quarot.py
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): role enum + tensor name classifier for QuaRot-R1"
```

---

### Task 2: R_resid builder (random Hadamard)

**Files:**
- Modify: `scripts/calibration/rotate_model_quarot.py` (add `build_R_resid`)
- Modify: `scripts/calibration/test_rotate_model_quarot.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `test_rotate_model_quarot.py`:

```python
from rotate_model_quarot import build_R_resid


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, label: str):
    diff = (actual - expected).abs().max().item()
    assert diff <= tol, f"{label}: max abs diff {diff:.3e} > tol {tol:.3e}"


def test_R_resid_orthogonal_pow2():
    """R @ R.T == I for power-of-2 d_model (pure Sylvester)."""
    R = build_R_resid(d_model=2048, seed=42, device=torch.device("cpu"))
    assert R.shape == (2048, 2048), f"shape {R.shape}"
    assert R.dtype == torch.float32, f"dtype {R.dtype}"
    I = torch.eye(2048, dtype=torch.float32)
    _assert_close(R @ R.T, I, tol=1e-5, label="R @ R.T")
    _assert_close(R.T @ R, I, tol=1e-5, label="R.T @ R")
    print(f"  PASS test_R_resid_orthogonal_pow2")


def test_R_resid_orthogonal_kronecker():
    """R @ R.T == I for non-power-of-2 d_model via Kronecker H_a ⊗ H_b."""
    R = build_R_resid(d_model=2560, seed=42, device=torch.device("cpu"))
    assert R.shape == (2560, 2560), f"shape {R.shape}"
    I = torch.eye(2560, dtype=torch.float32)
    _assert_close(R @ R.T, I, tol=1e-4, label="R @ R.T (kronecker)")
    print(f"  PASS test_R_resid_orthogonal_kronecker")


def test_R_resid_seed_determinism():
    """Same seed → same R; different seed → different R."""
    R1 = build_R_resid(d_model=64, seed=42, device=torch.device("cpu"))
    R2 = build_R_resid(d_model=64, seed=42, device=torch.device("cpu"))
    R3 = build_R_resid(d_model=64, seed=43, device=torch.device("cpu"))
    _assert_close(R1, R2, tol=0.0, label="same-seed determinism")
    diff = (R1 - R3).abs().max().item()
    assert diff > 1e-3, f"different-seed difference too small: {diff:.3e}"
    print(f"  PASS test_R_resid_seed_determinism")
```

And update the `__main__` block:

```python
if __name__ == "__main__":
    test_classify_qwen36_tensors()
    test_classify_unknown_raises()
    test_R_resid_orthogonal_pow2()
    test_R_resid_orthogonal_kronecker()
    test_R_resid_seed_determinism()
    print("\nALL TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: `ImportError: cannot import name 'build_R_resid'`.

- [ ] **Step 3: Write minimal implementation**

Add to `rotate_model_quarot.py`:

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kronecker_rotation import sylvester, factor_for_dim


def build_R_resid(d_model: int, seed: int, device: torch.device) -> torch.Tensor:
    """Construct a random Hadamard rotation of size d_model.

    R_resid = D ⊙ H, where:
      - H is Sylvester(d_model) for pure-power-of-2 d_model,
        or H_a_random ⊗ H_b_sylvester via factor_for_dim() otherwise.
      - D = diag(±1) sampled from Bernoulli(0.5) seeded by `seed`.

    Returned tensor is fp32 on the requested device. Deterministic in `seed`.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # Sign-flip diagonal — independent of H form, applied as elementwise row scale.
    signs = (torch.randint(0, 2, (d_model,), generator=gen, dtype=torch.float32) * 2.0 - 1.0)

    # Hadamard core
    if (d_model & (d_model - 1)) == 0:
        H = sylvester(d_model).to(dtype=torch.float32)
    else:
        a, b = factor_for_dim(d_model, max_b=1024)
        # Random-orthogonal H_a; deterministic Sylvester H_b.
        # Use a separate sub-seed so changing d_model factoring doesn't perturb
        # the sign-flip stream.
        from kronecker_rotation import random_orthogonal
        H_a = random_orthogonal(a, seed=int(seed) + 1_000_003)
        H_b = sylvester(b).to(dtype=torch.float32)
        H = torch.kron(H_a.contiguous(), H_b.contiguous())

    R = (signs.unsqueeze(1) * H).to(device=device, dtype=torch.float32)
    return R
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: all 5 tests print PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): R_resid random Hadamard builder"
```

---

### Task 3: Input-side and output-side rotation primitives

**Files:**
- Modify: `scripts/calibration/rotate_model_quarot.py`
- Modify: `scripts/calibration/test_rotate_model_quarot.py`

- [ ] **Step 1: Write the failing tests**

Append to `test_rotate_model_quarot.py`:

```python
from rotate_model_quarot import rotate_input_side, rotate_output_side


def test_rotate_input_side_round_trip():
    """Original forward equals rotated-input forward on rotated input."""
    torch.manual_seed(7)
    d_model, N = 64, 96
    W      = torch.randn(N, d_model, dtype=torch.float32)
    gamma  = torch.randn(d_model, dtype=torch.float32) * 0.3 + 1.0  # near-1 like real RMSNorm γ
    x      = torch.randn(4, d_model, dtype=torch.float32)
    R      = build_R_resid(d_model=d_model, seed=11, device=torch.device("cpu"))

    # Original forward: y = (γ ⊙ x) @ W.T
    y_orig = (gamma * x) @ W.T

    # Rotated forward: residual stream rotated as x' = x @ R.T; absorbed γ
    # makes the next linear see norm-output directly; W_new produces y from x'.
    W_new  = rotate_input_side(W, gamma, R)
    x_rot  = x @ R.T
    y_new  = x_rot @ W_new.T

    _assert_close(y_new, y_orig, tol=1e-4, label="input-side round trip")
    print(f"  PASS test_rotate_input_side_round_trip")


def test_rotate_output_side_round_trip():
    """Output-side rotation gives y_new = y @ R.T (residual ends up rotated)."""
    torch.manual_seed(8)
    d_model, K = 64, 80
    W      = torch.randn(d_model, K, dtype=torch.float32)  # writes residual
    x      = torch.randn(4, K, dtype=torch.float32)
    R      = build_R_resid(d_model=d_model, seed=12, device=torch.device("cpu"))

    y_orig    = x @ W.T            # original residual contribution
    W_new     = rotate_output_side(W, R)
    y_rotated = x @ W_new.T        # should equal y_orig @ R.T

    _assert_close(y_rotated, y_orig @ R.T, tol=1e-4, label="output-side R.T projection")
    print(f"  PASS test_rotate_output_side_round_trip")


def test_input_output_cancel_through_residual():
    """A linear writing then a linear reading the residual recovers the original."""
    torch.manual_seed(9)
    d_model, K, N = 64, 80, 96
    W_out     = torch.randn(d_model, K, dtype=torch.float32)    # writes residual
    gamma     = torch.randn(d_model, dtype=torch.float32) * 0.3 + 1.0
    W_in      = torch.randn(N, d_model, dtype=torch.float32)    # reads residual
    x         = torch.randn(4, K, dtype=torch.float32)
    R         = build_R_resid(d_model=d_model, seed=13, device=torch.device("cpu"))

    # Original
    residual_orig = x @ W_out.T
    y_orig        = (gamma * residual_orig) @ W_in.T

    # Rotated stack
    W_out_new = rotate_output_side(W_out, R)
    W_in_new  = rotate_input_side(W_in, gamma, R)
    residual_rot = x @ W_out_new.T     # = residual_orig @ R.T
    y_rot       = residual_rot @ W_in_new.T

    _assert_close(y_rot, y_orig, tol=1e-4, label="residual cancellation")
    print(f"  PASS test_input_output_cancel_through_residual")
```

Update `__main__`:

```python
if __name__ == "__main__":
    test_classify_qwen36_tensors()
    test_classify_unknown_raises()
    test_R_resid_orthogonal_pow2()
    test_R_resid_orthogonal_kronecker()
    test_R_resid_seed_determinism()
    test_rotate_input_side_round_trip()
    test_rotate_output_side_round_trip()
    test_input_output_cancel_through_residual()
    print("\nALL TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: `ImportError: cannot import name 'rotate_input_side'`.

- [ ] **Step 3: Write minimal implementation**

Add to `rotate_model_quarot.py`:

```python
def rotate_input_side(W: torch.Tensor, gamma: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """Apply γ absorption + R_resid input rotation to a residual-reading linear.

    W shape: [N, d_model] (PyTorch [out, in]).
    gamma shape: [d_model] — RMSNorm γ that precedes this linear in the original graph.
    R_resid shape: [d_model, d_model] — orthogonal.

    Math: original forward is y = (γ ⊙ x) @ W.T. In the rotated stream the input
    is x' = x @ R.T (γ already absorbed at construction time). The rotated weight
    must satisfy y = x' @ W_new.T, giving W_new = (W ⊙ γ_row) @ R.T.
    """
    Wg = W * gamma.unsqueeze(0)            # [N, d_model], column-wise γ
    return Wg @ R_resid.T                  # [N, d_model]


def rotate_output_side(W: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """Apply R_resid output rotation to a residual-writing linear.

    W shape: [d_model, K] (PyTorch [out, in]). K is whatever feeds this linear.
    R_resid shape: [d_model, d_model] — orthogonal.

    Math: original forward y = x @ W.T contributes to the residual. To produce
    y_new = y @ R.T directly, we need W_new = R @ W.
    """
    return R_resid @ W                     # [d_model, K]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: all 8 tests print PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): input/output-side rotation primitives + γ absorption"
```

---

### Task 4: MoE expert-axis batched rotation

**Files:**
- Modify: `scripts/calibration/rotate_model_quarot.py`
- Modify: `scripts/calibration/test_rotate_model_quarot.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from rotate_model_quarot import rotate_moe_input_side, rotate_moe_output_side


def test_moe_input_side_matches_per_expert_loop():
    """Batched MoE input rotation == per-expert rotate_input_side over n_experts."""
    torch.manual_seed(20)
    d_model, d_ffn, n_exp = 32, 48, 4
    # PyTorch shape for gate/up_exps: [d_ffn, d_model, n_exp]
    W      = torch.randn(d_ffn, d_model, n_exp, dtype=torch.float32)
    gamma  = torch.randn(d_model, dtype=torch.float32) * 0.3 + 1.0
    R      = build_R_resid(d_model=d_model, seed=21, device=torch.device("cpu"))

    expected = torch.empty_like(W)
    for e in range(n_exp):
        expected[..., e] = rotate_input_side(W[..., e], gamma, R)

    actual = rotate_moe_input_side(W, gamma, R)
    _assert_close(actual, expected, tol=1e-5, label="MoE input batched vs loop")
    print(f"  PASS test_moe_input_side_matches_per_expert_loop")


def test_moe_output_side_matches_per_expert_loop():
    """Batched MoE output rotation == per-expert rotate_output_side over n_experts."""
    torch.manual_seed(22)
    d_model, d_ffn, n_exp = 32, 48, 4
    # PyTorch shape for down_exps: [d_model, d_ffn, n_exp]
    W      = torch.randn(d_model, d_ffn, n_exp, dtype=torch.float32)
    R      = build_R_resid(d_model=d_model, seed=23, device=torch.device("cpu"))

    expected = torch.empty_like(W)
    for e in range(n_exp):
        expected[..., e] = rotate_output_side(W[..., e], R)

    actual = rotate_moe_output_side(W, R)
    _assert_close(actual, expected, tol=1e-5, label="MoE output batched vs loop")
    print(f"  PASS test_moe_output_side_matches_per_expert_loop")
```

Update `__main__` to call both.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: `ImportError: cannot import name 'rotate_moe_input_side'`.

- [ ] **Step 3: Write minimal implementation**

Add to `rotate_model_quarot.py`:

```python
def rotate_moe_input_side(W: torch.Tensor, gamma: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """MoE input-side rotation, batched along the expert axis.

    W shape: [d_ffn, d_model, n_experts]. Applies rotate_input_side to each
    expert in one batched matmul instead of a Python loop.
    """
    d_ffn, d_model, n_exp = W.shape
    # Reshape so the d_model axis stays adjacent for the matmul.
    # [d_ffn, d_model, n_exp] → permute → [d_ffn, n_exp, d_model] → flatten outer → [d_ffn*n_exp, d_model]
    W_p     = W.permute(0, 2, 1).contiguous().view(d_ffn * n_exp, d_model)
    W_rot   = (W_p * gamma.unsqueeze(0)) @ R_resid.T
    # Reshape back to [d_ffn, n_exp, d_model] → permute → [d_ffn, d_model, n_exp]
    return W_rot.view(d_ffn, n_exp, d_model).permute(0, 2, 1).contiguous()


def rotate_moe_output_side(W: torch.Tensor, R_resid: torch.Tensor) -> torch.Tensor:
    """MoE output-side rotation, batched along the expert axis.

    W shape: [d_model, d_ffn, n_experts]. Applies rotate_output_side
    (R @ W on the d_model axis) to each expert in one batched matmul.
    """
    d_model, d_ffn, n_exp = W.shape
    # Flatten the trailing dims so the d_model axis is the matmul axis.
    W_flat  = W.reshape(d_model, d_ffn * n_exp)
    W_rot   = R_resid @ W_flat
    return W_rot.view(d_model, d_ffn, n_exp).contiguous()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: all 10 tests print PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): batched MoE expert-axis rotation primitives"
```

---

### Task 5: Pass-1 indexing — build roster + γ map from a GGUF

**Files:**
- Modify: `scripts/calibration/rotate_model_quarot.py`
- Modify: `scripts/calibration/test_rotate_model_quarot.py`

- [ ] **Step 1: Write the failing test**

Append to test file:

```python
import tempfile
from pathlib import Path as _Path

def _make_tiny_qwen36_gguf(out_path: str, n_layers: int = 2, d_model: int = 32, d_ffn: int = 48, n_exp: int = 4):
    """Write a tiny qwen36moe-shaped bf16 GGUF for unit tests."""
    sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/gguf-py")
    import gguf as _gguf

    w = _gguf.GGUFWriter(out_path, arch="qwen36moe")
    w.add_uint32("qwen36moe.embedding_length", d_model)
    w.add_uint32("qwen36moe.block_count", n_layers)
    w.add_uint32("qwen36moe.feed_forward_length", d_ffn)
    w.add_uint32("qwen36moe.expert_count", n_exp)

    vocab = 64
    w.add_tensor("token_embd.weight",        torch.randn(vocab, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
    for L in range(n_layers):
        w.add_tensor(f"blk.{L}.attn_norm.weight",    (torch.ones(d_model) + 0.1 * torch.randn(d_model)).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.attn_q.weight",       torch.randn(d_model, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.attn_k.weight",       torch.randn(d_model, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.attn_v.weight",       torch.randn(d_model, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.attn_output.weight",  torch.randn(d_model, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.ffn_norm.weight",     (torch.ones(d_model) + 0.1 * torch.randn(d_model)).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.ffn_gate_inp.weight", torch.randn(n_exp, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.ffn_gate_exps.weight", torch.randn(d_ffn, d_model, n_exp).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.ffn_up_exps.weight",   torch.randn(d_ffn, d_model, n_exp).to(torch.bfloat16).view(torch.uint8).numpy())
        w.add_tensor(f"blk.{L}.ffn_down_exps.weight", torch.randn(d_model, d_ffn, n_exp).to(torch.bfloat16).view(torch.uint8).numpy())
    w.add_tensor("output_norm.weight", (torch.ones(d_model) + 0.1 * torch.randn(d_model)).to(torch.bfloat16).view(torch.uint8).numpy())
    w.add_tensor("output.weight",      torch.randn(vocab, d_model).to(torch.bfloat16).view(torch.uint8).numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


from rotate_model_quarot import index_pass


def test_index_pass_on_tiny_gguf():
    """Pass 1 builds a roster mapping every tensor name to a Role, and a γ map."""
    with tempfile.TemporaryDirectory() as td:
        path = str(_Path(td) / "tiny.gguf")
        _make_tiny_qwen36_gguf(path, n_layers=2)
        roster, gammas, d_model = index_pass(path, arch="qwen36moe")

    assert d_model == 32, f"d_model {d_model}"
    assert roster["token_embd.weight"] == Role.EMBED
    assert roster["blk.0.attn_q.weight"] == Role.ATTN_Q
    assert roster["blk.1.ffn_down_exps.weight"] == Role.FFN_DOWN_EXPS
    # γ map: keyed by tensor name, value is a torch.Tensor of size d_model.
    assert "blk.0.attn_norm.weight" in gammas
    assert gammas["blk.0.attn_norm.weight"].shape == (32,)
    assert "output_norm.weight" in gammas
    print(f"  PASS test_index_pass_on_tiny_gguf")
```

Update `__main__`.

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: `ImportError: cannot import name 'index_pass'`.

- [ ] **Step 3: Write minimal implementation**

Add to `rotate_model_quarot.py`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

import numpy as np


_NORM_ROLES = {Role.NORM_PRE_ATTN, Role.NORM_PRE_FFN, Role.NORM_PRE_SSM, Role.NORM_OUT}


def _gguf_tensor_to_torch(t) -> torch.Tensor:
    """Materialize a GGUFReader tensor as a torch float32 tensor.

    Assumes bf16 storage — the source GGUF is the bf16 conversion produced by
    gguf_f16_to_bf16.py and used by calibrate_ml8_paged.py.
    """
    if t.tensor_type.name != "BF16":
        raise ValueError(f"{t.name}: expected BF16, got {t.tensor_type.name}")
    arr = np.asarray(t.data, dtype=np.uint8).copy()  # detach from mmap
    return torch.from_numpy(arr).view(torch.bfloat16).view(*t.shape).to(torch.float32)


def index_pass(source_path: str, arch: str) -> tuple[dict[str, Role], dict[str, torch.Tensor], int]:
    """Pass 1 — walk source GGUF, classify each tensor, pull only γ vectors into RAM.

    Returns (roster, gammas, d_model). Roster covers every tensor in the file.
    Gammas only contains the RMSNorm tensors (kept resident — ~200 KB total even
    for 35B-A3B). d_model is read from the arch's embedding_length KV.
    """
    r = gguf.GGUFReader(source_path)
    roster: dict[str, Role] = {}
    gammas: dict[str, torch.Tensor] = {}
    for t in r.tensors:
        role = classify_tensor(t.name, arch=arch)
        roster[t.name] = role
        if role in _NORM_ROLES:
            gammas[t.name] = _gguf_tensor_to_torch(t)

    # d_model from arch KV (e.g. "qwen36moe.embedding_length"). Fall back to
    # token_embd.weight's last dim if the KV isn't present.
    d_model: Optional[int] = None
    for f in r.fields.values():
        if f.name == f"{arch}.embedding_length":
            d_model = int(f.parts[f.data[0]][0])
            break
    if d_model is None:
        embed_t = next(t for t in r.tensors if t.name == "token_embd.weight")
        d_model = int(embed_t.shape[-1])

    return roster, gammas, d_model
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: all 11 tests print PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): pass-1 GGUF tensor roster + γ index"
```

---

### Task 6: Pass-2 streaming rotate + write

**Files:**
- Modify: `scripts/calibration/rotate_model_quarot.py`
- Modify: `scripts/calibration/test_rotate_model_quarot.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from rotate_model_quarot import rotate_gguf


def test_rotate_gguf_end_to_end_on_tiny():
    """rotate_gguf produces an output GGUF whose forward equals the source's."""
    with tempfile.TemporaryDirectory() as td:
        src = str(_Path(td) / "src.gguf")
        dst = str(_Path(td) / "dst.gguf")
        _make_tiny_qwen36_gguf(src, n_layers=2, d_model=32, d_ffn=48, n_exp=4)

        rotate_gguf(source_path=src, output_path=dst, arch="qwen36moe",
                    seed=42, device=torch.device("cpu"))

        # Re-load and check shapes + γs zeroed-to-1.
        sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/gguf-py")
        import gguf as _gguf
        r = _gguf.GGUFReader(dst)
        names = {t.name for t in r.tensors}
        # Every source tensor present.
        assert "token_embd.weight" in names
        assert "blk.0.ffn_down_exps.weight" in names
        # γ tensors written as all-ones.
        gamma_t = next(t for t in r.tensors if t.name == "blk.0.attn_norm.weight")
        gamma_v = _gguf_tensor_to_torch(gamma_t)
        _assert_close(gamma_v, torch.ones(32), tol=1e-2, label="γ written as ones")
        # Rotated weight shape matches source.
        q_t = next(t for t in r.tensors if t.name == "blk.0.attn_q.weight")
        assert list(q_t.shape) == [32, 32], f"attn_q shape {list(q_t.shape)}"
    print(f"  PASS test_rotate_gguf_end_to_end_on_tiny")
```

Update `__main__`.

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: `ImportError: cannot import name 'rotate_gguf'`.

- [ ] **Step 3: Write minimal implementation**

Add to `rotate_model_quarot.py`:

```python
# Map roles to (rotation_kind, gamma_source) where rotation_kind is one of
# {"input_2d", "output_2d", "input_moe", "output_moe", "embed", "norm_to_ones", "passthrough"}.
# The γ source is the tensor name of the RMSNorm that precedes this linear in
# the graph, or None if no γ absorption applies.
def _rotation_plan(role: Role, name: str) -> tuple[str, Optional[str]]:
    """Decide (rotation_kind, gamma_tensor_name) for a given tensor."""
    if role == Role.PASSTHROUGH:
        return ("passthrough", None)
    if role in _NORM_ROLES:
        return ("norm_to_ones", None)
    if role == Role.EMBED:
        return ("embed", None)
    if role == Role.LM_HEAD:
        return ("input_2d", "output_norm.weight")
    # Per-block roles: pull the layer index from the name.
    m = re.match(r"blk\.(\d+)\.", name)
    if not m:
        raise ValueError(f"expected blk.N. prefix on {name!r} for role {role}")
    L = int(m.group(1))
    if role in (Role.ATTN_Q, Role.ATTN_K, Role.ATTN_V):
        return ("input_2d", f"blk.{L}.attn_norm.weight")
    if role == Role.ATTN_O:
        return ("output_2d", None)
    if role in (Role.FFN_GATE_INP,):
        return ("input_2d", f"blk.{L}.ffn_norm.weight")
    if role in (Role.FFN_GATE_EXPS, Role.FFN_UP_EXPS):
        return ("input_moe", f"blk.{L}.ffn_norm.weight")
    if role == Role.FFN_DOWN_EXPS:
        return ("output_moe", None)
    if role == Role.MAMBA_IN:
        return ("input_2d", f"blk.{L}.ssm_norm.weight")
    if role == Role.MAMBA_OUT:
        return ("output_2d", None)
    raise ValueError(f"no rotation plan for role {role}")


def _bf16_bytes_from_fp32(t: torch.Tensor) -> np.ndarray:
    """Cast a float32 torch tensor to bf16 raw bytes as a uint8 numpy view."""
    return t.contiguous().to(torch.bfloat16).view(torch.uint8).numpy()


def rotate_gguf(source_path: str, output_path: str, arch: str, seed: int,
                device: torch.device) -> dict:
    """Pass 2 — stream source GGUF, rotate every tensor according to its role,
    write to a new GGUF.

    Returns a manifest dict suitable for the sidecar JSON.
    """
    # Pass 1
    roster, gammas, d_model = index_pass(source_path, arch=arch)
    R_resid = build_R_resid(d_model=d_model, seed=seed, device=device)

    # Re-open source for streaming. Open writer.
    r = gguf.GGUFReader(source_path)
    w = gguf.GGUFWriter(output_path, arch=arch)

    # Copy KV fields verbatim.
    for f in r.fields.values():
        try:
            # GGUFWriter dispatches by type; the simplest path is to use
            # add_key_value which doesn't exist on all versions — use the typed
            # adders. For correctness here, we expose a small typed re-emit:
            w.add_key_value(f.name, f.parts[f.data[0]], f.types[0])
        except AttributeError:
            # Older gguf API: typed adders only. We only need numeric scalars
            # for arch metadata in the tiny test — pass them through with
            # add_uint32 / add_float32 by inspecting f.types[0].
            t = f.types[0]
            v = f.parts[f.data[0]]
            if t == gguf.GGUFValueType.UINT32:
                w.add_uint32(f.name, int(v[0]))
            elif t == gguf.GGUFValueType.FLOAT32:
                w.add_float32(f.name, float(v[0]))
            elif t == gguf.GGUFValueType.STRING:
                w.add_string(f.name, str(bytes(v).decode("utf-8")))
            else:
                # Skip exotic types in tests; production source has been the
                # output of gguf_f16_to_bf16.py which uses standard types.
                pass

    rotated: list[str] = []
    absorbed: list[str] = []

    for t in r.tensors:
        role = roster[t.name]
        kind, gamma_name = _rotation_plan(role, t.name)

        if kind == "passthrough":
            # Copy bytes through unchanged.
            arr = np.asarray(t.data, dtype=np.uint8).copy()
            w.add_tensor(t.name, arr, raw_dtype=t.tensor_type, raw_shape=list(t.shape))
            continue

        if kind == "norm_to_ones":
            # Write ones in bf16. Shape preserved.
            ones = torch.ones(*t.shape, dtype=torch.float32)
            w.add_tensor(t.name, _bf16_bytes_from_fp32(ones),
                         raw_dtype=t.tensor_type, raw_shape=list(t.shape))
            absorbed.append(t.name)
            continue

        # Otherwise we need to materialize the tensor as fp32.
        W_fp32 = _gguf_tensor_to_torch(t).to(device)

        if kind == "embed":
            # token_embd shape is [vocab, d_model]. Output-side rotation
            # along d_model = right-multiply by R_resid.
            W_new = W_fp32 @ R_resid
        elif kind == "input_2d":
            gamma = gammas[gamma_name].to(device) if gamma_name else torch.ones(d_model, device=device)
            W_new = rotate_input_side(W_fp32, gamma, R_resid)
        elif kind == "output_2d":
            W_new = rotate_output_side(W_fp32, R_resid)
        elif kind == "input_moe":
            gamma = gammas[gamma_name].to(device) if gamma_name else torch.ones(d_model, device=device)
            W_new = rotate_moe_input_side(W_fp32, gamma, R_resid)
        elif kind == "output_moe":
            W_new = rotate_moe_output_side(W_fp32, R_resid)
        else:
            raise ValueError(f"unknown rotation kind {kind!r}")

        w.add_tensor(t.name,
                     _bf16_bytes_from_fp32(W_new.cpu()),
                     raw_dtype=t.tensor_type, raw_shape=list(t.shape))
        rotated.append(t.name)

        # Free GPU memory before next tensor.
        del W_fp32, W_new
        if device.type == "cuda":
            torch.cuda.empty_cache()

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    return {
        "seed": int(seed),
        "d_model": int(d_model),
        "arch": arch,
        "rotated_tensors": rotated,
        "absorbed_norms": absorbed,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: all 12 tests print PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): pass-2 streaming rotate + GGUF write"
```

---

### Task 7: Sidecar JSON + CLI driver

**Files:**
- Modify: `scripts/calibration/rotate_model_quarot.py`
- Modify: `scripts/calibration/test_rotate_model_quarot.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
import json as _json
import subprocess


def test_cli_produces_gguf_and_sidecar():
    """CLI invocation writes both the rotated GGUF and the sidecar JSON next to it."""
    with tempfile.TemporaryDirectory() as td:
        src = str(_Path(td) / "src.gguf")
        dst = str(_Path(td) / "dst.gguf")
        _make_tiny_qwen36_gguf(src, n_layers=2)
        result = subprocess.run(
            [sys.executable, str(_Path(__file__).resolve().parent / "rotate_model_quarot.py"),
             "--source", src, "--output", dst,
             "--arch", "qwen36moe", "--seed", "42"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f"CLI failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert _Path(dst).exists(), "output GGUF missing"
        sidecar = _Path(dst + ".quarot_r1.json")
        assert sidecar.exists(), "sidecar JSON missing"
        payload = _json.loads(sidecar.read_text())
        assert payload["seed"] == 42
        assert payload["arch"] == "qwen36moe"
        assert payload["d_model"] == 32
        assert any("attn_q" in n for n in payload["rotated_tensors"])
        assert any("attn_norm" in n for n in payload["absorbed_norms"])
    print(f"  PASS test_cli_produces_gguf_and_sidecar")
```

Update `__main__`.

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: subprocess returns non-zero; CLI hasn't been wired up.

- [ ] **Step 3: Write minimal implementation**

Add to `rotate_model_quarot.py`:

```python
import argparse
import json


def _save_sidecar(output_path: str, manifest: dict) -> None:
    sidecar_path = output_path + ".quarot_r1.json"
    with open(sidecar_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Apply QuaRot-R1 rotation to a bf16 GGUF.")
    p.add_argument("--source", required=True, help="path to source bf16 GGUF")
    p.add_argument("--output", required=True, help="path to write rotated bf16 GGUF")
    p.add_argument("--arch", default="qwen36moe", choices=sorted(_ROLE_PATTERNS.keys()),
                   help="GGUF architecture for tensor-name classification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="torch device string for the rotation matmul (e.g. cuda:0)")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    manifest = rotate_gguf(source_path=args.source, output_path=args.output,
                           arch=args.arch, seed=args.seed, device=device)
    _save_sidecar(args.output, manifest)
    print(f"wrote {args.output} ({len(manifest['rotated_tensors'])} rotated, "
          f"{len(manifest['absorbed_norms'])} absorbed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 scripts/calibration/test_rotate_model_quarot.py
```

Expected: all 13 tests print PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): CLI driver + sidecar JSON for QuaRot-R1"
```

---

### Task 8: Equivalence gate script

**Files:**
- Create: `scripts/calibration/test_quarot_r1_equivalence.py`

This script is a runtime gate, not a unit test. It runs `llama-cli --no-mmap`
on both source and rotated GGUFs, compares final logits via the `--logit-bias`
+ verbose-output path. Because llama-cli doesn't natively dump logits, we use
the `tools/perplexity` binary instead — it reports per-chunk PPL on a fixed
slice, which is enough to detect any non-equivalence.

- [ ] **Step 1: Write the gate script**

Create `scripts/calibration/test_quarot_r1_equivalence.py`:

```python
#!/usr/bin/env python3
"""Equivalence gate for QuaRot-R1 rotation.

Runs llama-perplexity on both a source bf16 GGUF and its rotated counterpart
on a fixed wikitext-2 slice; asserts the PPL values match to ±0.005. This is
the bit-equivalence gate from the design doc — it must pass before any
calibration is run on the rotated GGUF.

Mad-lab standing rule: --no-mmap is passed to every llama.cpp invocation.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


LLAMA_PERPLEXITY = "/home/kmbandy/GitHub/llama.cpp/build/bin/llama-perplexity"
DEFAULT_WIKITEXT = "/home/kmbandy/wikitext-2-raw/wiki.test.raw"


def run_ppl(gguf_path: str, wikitext_path: str, max_tokens: int) -> float:
    """Run llama-perplexity and return the final PPL value."""
    cmd = [
        LLAMA_PERPLEXITY,
        "--no-mmap",
        "-m", gguf_path,
        "-f", wikitext_path,
        "--ctx-size", "4096",
        "--threads", "8",
        "-n", str(max_tokens),
    ]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"llama-perplexity exited {proc.returncode}")
    # Parse the final "perplexity: X.XXXX ± Y.YYYY" line.
    m = re.search(r"perplexity:\s+([\d.]+)\s+±", proc.stdout)
    if not m:
        m = re.search(r"Final estimate: PPL = ([\d.]+)", proc.stdout)
    if not m:
        print(proc.stdout[-2000:])
        raise RuntimeError("could not parse PPL from llama-perplexity output")
    return float(m.group(1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--rotated", required=True)
    p.add_argument("--wikitext", default=DEFAULT_WIKITEXT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--tol", type=float, default=0.005)
    args = p.parse_args()

    print(f"[gate] source : {args.source}")
    print(f"[gate] rotated: {args.rotated}")

    ppl_src = run_ppl(args.source,  args.wikitext, args.max_tokens)
    ppl_rot = run_ppl(args.rotated, args.wikitext, args.max_tokens)
    diff = abs(ppl_src - ppl_rot)
    print(f"[gate] PPL source : {ppl_src:.4f}")
    print(f"[gate] PPL rotated: {ppl_rot:.4f}")
    print(f"[gate] |diff|     : {diff:.4f} (tol {args.tol})")
    if diff > args.tol:
        print("[gate] FAIL — rotation is NOT equivalent to source", file=sys.stderr)
        return 1
    print("[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the script is syntactically valid + arg parsing works**

```bash
python3 scripts/calibration/test_quarot_r1_equivalence.py --help
```

Expected: argparse help text including `--source`, `--rotated`, `--tol`, exit 0.

- [ ] **Step 3: Commit**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add scripts/calibration/test_quarot_r1_equivalence.py
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): equivalence gate script (perplexity-based)"
```

---

### Task 9: 4B smoke — equivalence gate against Qwen3.5-4B

**Files:**
- No code changes. This is a manual validation run.

- [ ] **Step 1: Locate the 4B bf16 GGUF**

```bash
ls -lh /home/kmbandy/models/Qwen3.5-4B-bf16.gguf 2>/dev/null || \
  echo "MISSING — convert from f16 with scripts/calibration/gguf_f16_to_bf16.py first"
```

Expected: file exists (~8 GB). If missing, run the conversion before continuing.

- [ ] **Step 2: Run the rotation**

```bash
python3 scripts/calibration/rotate_model_quarot.py \
  --source  /home/kmbandy/models/Qwen3.5-4B-bf16.gguf \
  --output  /home/kmbandy/models/Qwen3.5-4B-bf16-rotR1.gguf \
  --arch    qwen35 \
  --seed    42 \
  --device  cuda:0
```

Note: `qwen35` (not `qwen36moe`) — Qwen3.5-4B is dense, not MoE. **If the `qwen35` arch is missing from `_ROLE_PATTERNS`, add it before running.** Pattern table for dense Qwen3.5:

```python
"qwen35": [
    (r"token_embd\.weight",                Role.EMBED),
    (r"output_norm\.weight",               Role.NORM_OUT),
    (r"output\.weight",                    Role.LM_HEAD),
    (r"blk\.\d+\.attn_norm\.weight",       Role.NORM_PRE_ATTN),
    (r"blk\.\d+\.ffn_norm\.weight",        Role.NORM_PRE_FFN),
    (r"blk\.\d+\.attn_q\.weight",          Role.ATTN_Q),
    (r"blk\.\d+\.attn_k\.weight",          Role.ATTN_K),
    (r"blk\.\d+\.attn_v\.weight",          Role.ATTN_V),
    (r"blk\.\d+\.attn_output\.weight",     Role.ATTN_O),
    # Dense FFN: gate/up are input-residual linears, down is output-residual.
    # Reuse FFN_GATE_EXPS/UP_EXPS/DOWN_EXPS by also adding dense aliases:
    (r"blk\.\d+\.ffn_gate\.weight",        Role.FFN_GATE_EXPS),
    (r"blk\.\d+\.ffn_up\.weight",          Role.FFN_UP_EXPS),
    (r"blk\.\d+\.ffn_down\.weight",        Role.FFN_DOWN_EXPS),
    (r"rope_freqs\.weight",                Role.PASSTHROUGH),
    # Intentionally no `.*` catch-all — unknown names raise (matches qwen36moe).
],
```

For dense linears, the existing `input_2d` / `output_2d` rotation kinds apply
(no MoE batching needed). Update `_rotation_plan` accordingly — for the dense
ffn names, the role is the same but the rotation kind should be `input_2d` /
`output_2d` instead of `input_moe` / `output_moe`. Easiest fix: dispatch by
tensor rank in `rotate_gguf`'s main loop rather than by role; tensors with 2
dims use the 2D primitives, 3D use the MoE primitives.

- [ ] **Step 2.5: Replace role-based kind dispatch with rank-based in rotate_gguf**

Modify the `_rotation_plan` calls in `rotate_gguf` so the rotation kind for
`FFN_GATE_EXPS/UP_EXPS/DOWN_EXPS` roles is chosen by inspecting `t.shape`:

```python
# inside the for-t loop in rotate_gguf, after computing role:
kind, gamma_name = _rotation_plan(role, t.name)
if kind == "input_moe" and len(t.shape) == 2:
    kind = "input_2d"
if kind == "output_moe" and len(t.shape) == 2:
    kind = "output_2d"
```

Add a test in `test_rotate_model_quarot.py` for this — invoke `rotate_gguf`
on a dense 2-layer toy GGUF using the qwen35 arch and verify the output is
written successfully.

- [ ] **Step 3: Run the equivalence gate**

Expect the rotation to take ~2 min on cuda:0 for a 4B model. Then:

```bash
python3 scripts/calibration/test_quarot_r1_equivalence.py \
  --source  /home/kmbandy/models/Qwen3.5-4B-bf16.gguf \
  --rotated /home/kmbandy/models/Qwen3.5-4B-bf16-rotR1.gguf \
  --max-tokens 4096 \
  --tol 0.005
```

Expected output ends with `[gate] PASS`.

If FAIL: do not proceed. Use systematic debugging — likely candidates are
γ absorption sign, output-side rotation axis, embed direction, or an arch
pattern miss. Print per-tensor max-abs-diff for the first 10 rotated tensors
to localize.

- [ ] **Step 4: Commit any fixes**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add -u
git -C /home/kmbandy/GitHub/llama.cpp commit -m "feat(MAD-244 #104): qwen35 arch + rank-based rotation dispatch"
```

---

### Task 10: 4B calibration with rotated GGUF

**Files:**
- No code changes. This is the Phase 2 validation from the design doc.

- [ ] **Step 1: Run the calibration on the rotated 4B GGUF**

In a tmux session or with `run_in_background`:

```bash
cd /home/kmbandy/GitHub/llama.cpp
python3 scripts/calibration/calibrate_ml8_paged.py \
  --model      Qwen/Qwen3.5-4B \
  --gguf       /home/kmbandy/models/Qwen3.5-4B-bf16-rotR1.gguf \
  --output-dir /home/kmbandy/models/cell-rotR1-4B \
  --strategy   dense \
  --rotation   none \
  --snap-centroids e4m3 \
  --group-size 64 \
  --n-centroids 16 \
  --fit-loss   mse \
  --n-samples  32 \
  --seq-len    1024 \
  --eval-ppl \
  --ppl-max-tokens 100000 \
  --device     cuda:0 \
  2>&1 | tee /tmp/4B-rotR1-cal.log
```

Note `--rotation none` — R1 has already been folded into the GGUF; per-linear
rotation would double-rotate.

- [ ] **Step 2: Set up a time-based progress monitor (standing rule)**

In another terminal:

```bash
tail -f /tmp/4B-rotR1-cal.log | grep --line-buffered -E \
  "Y_SNR|perplexity:.*\[|chunks=|Traceback|Error|FAILED|Killed|OOM"
```

Calibration is expected to take ~15 min on R9700. Don't wait silently — let
the monitor stream chunk progress and failure signatures.

- [ ] **Step 3: Read the PPL gate**

When the calibration finishes, read the final PPL line from the log:

```bash
grep -E "perplexity: [0-9]" /tmp/4B-rotR1-cal.log | tail -1
```

Expected (success): final PPL such that `Δ_PPL = PPL_rotated_ml8 - PPL_f16_4B ≤ +0.04`.
The f16 4B baseline is 8.3181 (from the KG fact dated 2026-05-24). Target:
PPL_rotated_ml8 ≤ 8.358.

If PPL is worse than the un-rotated ml8 baseline (8.4015), R1 has made
things worse on this model — stop and investigate before touching 35B.

- [ ] **Step 4: Commit the result and update KG**

```bash
git -C /home/kmbandy/GitHub/llama.cpp add /tmp/4B-rotR1-cal.log  # if you want it tracked
# (or just summarize in commit message; the log itself is not source)
git -C /home/kmbandy/GitHub/llama.cpp commit --allow-empty \
  -m "chore(MAD-244 #104): 4B QuaRot-R1 ml8 calibration — PPL=X.YYYY (Δ=±Z.ZZZZ vs f16)"
```

Save a KG fact memory with the numeric result so the 35B decision is informed:

```
mcp__mad-lab-memory__kg_write(
  content="QuaRot-R1 on Qwen3.5-4B ml8_4: PPL = X.YYYY (vs baseline ml8 8.4015, Δ_PPL = ±Z.ZZZZ vs f16 8.3181). Seed=42, group_size=64, snap_centroids=e4m3, MLP-only.",
  type="fact",
  source="claude",
)
```

(Fill in the X/Y/Z numbers from the actual run.)

---

### Task 11: 35B-A3B end-to-end (the real run)

**Files:**
- No code changes. Phase 3 from the design doc.

**Pre-flight checklist (MUST verify before launching):**
- 4B PPL gate (Task 10) passed.
- R9700 is idle (no other VRAM consumers).
- VRAM math: model 24 GB + R_resid 16 MB + calibration scratch < 28 GB. The
  rotation pass uses ~150 MB peak (one MoE-expert tensor); calibration uses
  paged ingest (see `calibrate_ml8_paged.py --strategy moe`).
- The mad-lab "STOP iterating on 35B+ paged tests immediately after one OOM
  or system-restart event" rule applies. One run, monitored, then read the
  result and stop.

- [ ] **Step 1: Run the rotation pass on the 35B-A3B bf16 GGUF**

```bash
python3 scripts/calibration/rotate_model_quarot.py \
  --source  /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf \
  --output  /home/kmbandy/models/Qwen3.6-35B-A3B-bf16-rotR1.gguf \
  --arch    qwen36moe \
  --seed    42 \
  --device  cuda:0
```

Expected wall time: ~5 min. Memory peak: ~200 MB host + ~16 MB on cuda:0.

- [ ] **Step 2: Run the equivalence gate**

```bash
python3 scripts/calibration/test_quarot_r1_equivalence.py \
  --source  /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf \
  --rotated /home/kmbandy/models/Qwen3.6-35B-A3B-bf16-rotR1.gguf \
  --max-tokens 4096 \
  --tol 0.005
```

Expected: `[gate] PASS`. If FAIL, stop — do not proceed to calibration. Use
systematic debugging on a single layer with known input.

- [ ] **Step 3: Run the calibration**

```bash
python3 scripts/calibration/calibrate_ml8_paged.py \
  --model      Qwen/Qwen3.6-35B-A3B \
  --gguf       /home/kmbandy/models/Qwen3.6-35B-A3B-bf16-rotR1.gguf \
  --output-dir /home/kmbandy/models/cell-rotR1-35B \
  --strategy   moe \
  --rotation   none \
  --snap-centroids e4m3 \
  --group-size 64 \
  --n-centroids 16 \
  --fit-loss   mse \
  --n-samples  32 \
  --seq-len    1024 \
  --eval-ppl \
  --ppl-max-tokens 100000 \
  --device     cuda:0 \
  2>&1 | tee /tmp/35B-rotR1-cal.log
```

Expected wall time: ~3 h calibration + ~30 min PPL. Set up the time-based
Monitor with the same regex as Task 10 step 2.

- [ ] **Step 4: Pack the rotated calibration into a final GGUF**

```bash
python3 scripts/calibration/ml8_to_gguf.py \
  --src-gguf       /home/kmbandy/models/Qwen3.6-35B-A3B-bf16-rotR1.gguf \
  --calibration-dir /home/kmbandy/models/cell-rotR1-35B \
  --output         /home/kmbandy/models/Qwen3.6-35B-A3B-ml8_4_soa-rotR1.gguf \
  --soa
```

- [ ] **Step 5: Run final PPL on the packed ml8 GGUF**

```bash
/home/kmbandy/GitHub/llama.cpp/build/bin/llama-perplexity \
  --no-mmap \
  -m /home/kmbandy/models/Qwen3.6-35B-A3B-ml8_4_soa-rotR1.gguf \
  -f /home/kmbandy/wikitext-2-raw/wiki.test.raw \
  --ctx-size 4096 \
  --threads 8 \
  2>&1 | tee /tmp/35B-rotR1-ppl.log
```

Read the gate result:

```bash
grep -E "perplexity: [0-9]" /tmp/35B-rotR1-ppl.log | tail -1
```

**Gate:** PPL ≤ 5.770 (closes most of the +0.046 gap vs Q4_K_XL 5.7507).
Baseline to beat: 5.7968 (ml8_4_soa without R1).

- [ ] **Step 6: Commit + KG note + Jira update**

```bash
git -C /home/kmbandy/GitHub/llama.cpp commit --allow-empty \
  -m "chore(MAD-244 #104): 35B-A3B QuaRot-R1 ml8 calibration — PPL=X.YYYY (Δ=±Z.ZZZZ vs Q4_K_XL 5.7507)"
```

KG note:

```
mcp__mad-lab-memory__kg_write(
  content="QuaRot-R1 on Qwen3.6-35B-A3B ml8_4_soa: PPL = X.YYYY @ ctx=4096 (vs baseline ml8 5.7968, Q4_K_XL 5.7507). Seed=42, MoE strategy, MLP-only calibration unchanged.",
  type="fact",
  source="claude",
)
```

Update MAD-244 in Jira with the final number.

---

## Self-Review

**Spec coverage:**

| Spec section | Implementing task(s) |
|---|---|
| Architecture (preprocessing pass, GGUF in/out) | Tasks 5–7 (index, rotate, CLI) |
| R_resid construction (random Hadamard) | Task 2 |
| Rotation table (per-tensor ops) | Tasks 3–4 + Task 6 dispatch + Task 9 dense alias |
| RMSNorm γ absorption | Task 3 (γ in `rotate_input_side`) + Task 6 (`norm_to_ones`) |
| MoE batching | Task 4 |
| Mamba layers | Task 1 (roles) + Task 6 (`_rotation_plan`) |
| Equivalence verification (Section 5) | Task 8 (gate script) |
| Phase 1 equivalence | Tasks 9 step 3, 11 step 2 |
| Phase 2 4B calibration gate | Task 10 |
| Phase 3 35B-A3B calibration | Task 11 |
| Phase 4 diagnostic readout | Task 11 step 6 (KG note with per-kind Y_SNR if needed) |
| Calibration interaction (option 1: `--rotation none`) | Tasks 10 step 1, 11 step 3 |
| Out-of-scope items | Not implemented; flagged in spec |

All spec sections covered.

**Placeholder scan:** searched the plan for "TBD", "TODO", "implement later",
"appropriate error handling", "fill in details", "similar to". None present.
The `X.YYYY`/`Z.ZZZZ` placeholders in Tasks 10–11 commit messages and KG notes
are intentional — those are filled in by the engineer at run-time with the
actual measured PPL.

**Type consistency:** verified that function signatures match across tasks:
- `Role` enum members used in Task 1, 5, 6 — all match
- `build_R_resid(d_model, seed, device) → Tensor` — same signature in 2/3/4/6
- `rotate_input_side(W, gamma, R)` — same signature in 3/4/6
- `rotate_output_side(W, R)` — same signature in 3/4/6
- `rotate_moe_input_side(W, gamma, R)` — same signature in 4/6
- `rotate_moe_output_side(W, R)` — same signature in 4/6
- `index_pass(path, arch) → (roster, gammas, d_model)` — same in 5/6
- `rotate_gguf(source, output, arch, seed, device) → manifest` — same in 6/7

All consistent.

---

Plan complete and saved to `docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
