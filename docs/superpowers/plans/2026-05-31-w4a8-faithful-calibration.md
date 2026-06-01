# W4A8 Deployment-Faithful Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ml8 calibration collect Hessians on (and propagate) the rotated, e4m3-quantized activation the hardware actually feeds, so GPTQ + heavy-FT optimize against the real FP8 lattice — a bpv-neutral PPL lever.

**Architecture:** A pure-Python e4m3 quantizer that bit-matches the CUDA kernel `ml8_fp32_to_e4m3` (ggml/src/ggml-cuda/ml8.cu:440) drives a forward-pre-hook on every ml8-4 target linear. Per the identity `x_eff = e4m3(x@Q)@Qᵀ`, feeding `x_eff` into the unchanged linear reproduces the faithful output, so a single hook does both Hessian collection (in rotated+quantized space, dropping the algebraic `rotate_hessian`) and faithful downstream propagation. fp8 weight tiers (embed, ssm α/β) are simulated via one-time quant→dequant overrides. Toggles `--faithful-acts` / `--faithful-weights` gate each tier for clean attribution against the 19.2678 zero-point.

**Tech Stack:** Python, PyTorch, pytest; C (golden generator); existing `scripts/calibration/` rig (`calibrate_ml8_paged.py`, `method_gauntlet.py`, `role_targets.py`, `scaled_fp8.py`, `kronecker_rotation.py`).

**Spec:** `docs/superpowers/specs/2026-05-31-w4a8-faithful-calibration-design.md`

**Ground-truth rule (applies to every gate):** if a Python value disagrees with the CUDA kernel, the **kernel is truth** (it is what ships). Fix Python, never the kernel. NOTE: the C reference `quantize_row_f8_e4m3_ref` (ggml-turbo-quant.c:1198) has the *old* `e_out >= 15` premature-saturation bug; the CUDA kernel has the fix (`e_out > 15`, ml8.cu:479). **Port the CUDA kernel, not the C ref.**

---

## File Structure

| File | New/Mod | Responsibility |
|---|---|---|
| `scripts/calibration/ml8_e4m3_sim.py` | Create | Bit-exact e4m3 encode/decode + vectorized per-row activation quant. The unit under Gates A/B. |
| `scripts/calibration/tools/ml8_e4m3_golden.c` | Create | Host C copy of the CUDA kernel algorithm; emits a uint8 golden over a fp32 battery. |
| `scripts/calibration/faithful_forward.py` | Create | `FaithfulActHook` (pre-hook: rotate→e4m3→accumulate H→return x_eff) + `build_rotations`. |
| `scripts/calibration/tests/test_ml8_e4m3_sim.py` | Create | Gate A (bit-match golden) + scalar/vectorized parity + decode round-trip. |
| `scripts/calibration/tests/test_faithful_forward.py` | Create | x_eff identity, rotated-space H correctness, double-rotation guard, rotation determinism. |
| `scripts/calibration/calibrate_ml8_paged.py` | Modify | Wire `--faithful-acts`/`--faithful-weights`; install hooks/overrides in the dense branch; drop `rotate_hessian` when faithful-acts on + assert guard. |
| `scripts/calibration/method_gauntlet.py` | Modify | Pass-through flags; paired-toggle cell pair for the measurement runs. |

`tests/` dir for calibration may not exist yet — Task 2 creates it.

---

## Phase 1 — e4m3 sim + equivalence gates (foundation, pure TDD)

### Task 1: C golden generator (kernel algorithm, host-callable)

**Files:**
- Create: `scripts/calibration/tools/ml8_e4m3_golden.c`

- [ ] **Step 1: Write the golden generator** — copy the CUDA `ml8_fp32_to_e4m3` body verbatim (ml8.cu:440–503), stripped of `__device__ __forceinline__`, plus a fixed battery and binary dump.

```c
// scripts/calibration/tools/ml8_e4m3_golden.c
// Host copy of ggml/src/ggml-cuda/ml8.cu:ml8_fp32_to_e4m3 (the FIXED kernel,
// e_out > 15). Emits: for each fp32 in the battery, one uint8 e4m3 code.
// Output file format: int32 count, then `count` float32 inputs, then `count` uint8 codes.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static uint8_t ml8_fp32_to_e4m3(float xv) {
    uint32_t bits; memcpy(&bits, &xv, 4);
    const uint32_t sign  = (bits >> 31) & 1u;
    const uint32_t exp_b = (bits >> 23) & 0xFFu;
    const uint32_t mant  = bits & 0x7FFFFFu;
    if (exp_b == 0xFFu) return (uint8_t)((sign << 7) | 0x7Fu);
    if (exp_b == 0)     return (uint8_t)(sign << 7);
    const int32_t e_un = (int32_t) exp_b - 127;
    if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u))
        return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
    if (e_un >= -6) {
        const uint32_t e_e4m3 = (uint32_t)(e_un + 7);
        const uint32_t guard  = (mant >> 19) & 1u;
        const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
        const uint32_t lsb    = (mant >> 20) & 1u;
        uint32_t       m_e4m3 = (mant >> 20) & 0x7u;
        if (guard && (sticky || lsb)) m_e4m3 += 1;
        uint32_t e_out = e_e4m3;
        if (m_e4m3 == 8) { m_e4m3 = 0; e_out += 1;
            if (e_out > 15) return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u); }
        if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
        return (uint8_t)((sign << 7) | (e_out << 3) | m_e4m3);
    }
    const int32_t shift = 23 - (e_un + 9);
    if (shift > 31) return (uint8_t)(sign << 7);
    const uint32_t implicit = (1u << 23) | mant;
    const uint32_t guard    = (implicit >> (shift - 1)) & 1u;
    const uint32_t sticky   = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
    uint32_t       m_e4m3   = implicit >> shift;
    const uint32_t lsb      = m_e4m3 & 1u;
    if (guard && (sticky || lsb)) m_e4m3 += 1;
    if (m_e4m3 >= 8) return (uint8_t)((sign << 7) | (1u << 3));
    return (uint8_t)((sign << 7) | m_e4m3);
}

int main(void) {
    // Battery: dense low range, every normal lattice boundary, the 256..448
    // band (the e=15 fix), subnormals < 2^-6, ties, saturation, sign symmetry.
    float xs[100000]; int n = 0;
    for (float v = -512.0f; v <= 512.0f; v += 0.013f) xs[n++] = v;       // dense sweep
    float edges[] = {448.0f, 449.0f, 256.0f, 288.0f, 320.0f, 480.0f,
                     0.015625f, 0.0078125f, 0.001953125f,                // 2^-6,2^-7,2^-9
                     1e-30f, 1e30f, -0.0f};
    for (unsigned i = 0; i < sizeof(edges)/sizeof(float); i++) { xs[n++]=edges[i]; xs[n++]=-edges[i]; }
    float inf = INFINITY, nan = NAN; xs[n++]=inf; xs[n++]=-inf; xs[n++]=nan;

    FILE *f = fopen("/tmp/ml8_e4m3_golden.bin", "wb");
    fwrite(&n, 4, 1, f);
    fwrite(xs, 4, n, f);
    for (int i = 0; i < n; i++) { uint8_t c = ml8_fp32_to_e4m3(xs[i]); fwrite(&c, 1, 1, f); }
    fclose(f);
    printf("wrote %d cases to /tmp/ml8_e4m3_golden.bin\n", n);
    return 0;
}
```

- [ ] **Step 2: Compile and run it**

Run: `cc -O2 -o /tmp/ml8_e4m3_golden scripts/calibration/tools/ml8_e4m3_golden.c && /tmp/ml8_e4m3_golden`
Expected: prints `wrote <N> cases to /tmp/ml8_e4m3_golden.bin` (N ≈ 78k), exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibration/tools/ml8_e4m3_golden.c
git commit -m "feat(calib): C golden generator for ml8 e4m3 (kernel-faithful)"
```

---

### Task 2: Scalar bit-exact encode/decode + Gate A

**Files:**
- Create: `scripts/calibration/ml8_e4m3_sim.py`
- Create: `scripts/calibration/tests/test_ml8_e4m3_sim.py`

- [ ] **Step 1: Write the failing Gate A test**

```python
# scripts/calibration/tests/test_ml8_e4m3_sim.py
import struct, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
from ml8_e4m3_sim import fp32_to_e4m3_bits   # noqa: E402

GOLDEN = Path("/tmp/ml8_e4m3_golden.bin")

def _ensure_golden():
    if GOLDEN.exists():
        return
    src = CALIB / "tools/ml8_e4m3_golden.c"
    exe = Path("/tmp/ml8_e4m3_golden")
    subprocess.run(["cc", "-O2", "-o", str(exe), str(src)], check=True)
    subprocess.run([str(exe)], check=True)

def _load_golden():
    _ensure_golden()
    with open(GOLDEN, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        xs = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
        cs = np.frombuffer(f.read(n), dtype=np.uint8).copy()
    return xs, cs

def test_gate_a_bit_match_kernel():
    xs, golden = _load_golden()
    got = np.array([fp32_to_e4m3_bits(float(x)) for x in xs], dtype=np.uint8)
    mism = np.nonzero(got != golden)[0]
    assert mism.size == 0, (
        f"{mism.size} mismatches; first: x={xs[mism[0]]!r} "
        f"got=0x{got[mism[0]]:02x} want=0x{golden[mism[0]]:02x}")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd scripts/calibration && python -m pytest tests/test_ml8_e4m3_sim.py::test_gate_a_bit_match_kernel -v`
Expected: FAIL — `ImportError` / `cannot import name 'fp32_to_e4m3_bits'`.

- [ ] **Step 3: Implement the scalar encoder + decoder** (Python port of the same kernel body)

```python
# scripts/calibration/ml8_e4m3_sim.py
"""Bit-exact Python mirror of the ml8 activation e4m3 path.

Ground truth = the CUDA kernel ml8_fp32_to_e4m3 (ggml/src/ggml-cuda/ml8.cu:440),
NOT the C ref quantize_row_f8_e4m3_ref (which carries the old e_out>=15 bug).
"""
import struct
import torch

E4M3_MAX = 448.0
ACT_SCALE_EPS = 1e-12   # matches ML8_ACT_SCALE_EPS

def fp32_to_e4m3_bits(xv: float) -> int:
    """Return the uint8 e4m3 code for one fp32, RNE, saturating, e4m3fn."""
    bits = struct.unpack("<I", struct.pack("<f", xv))[0]
    sign  = (bits >> 31) & 1
    exp_b = (bits >> 23) & 0xFF
    mant  = bits & 0x7FFFFF
    if exp_b == 0xFF:           # NaN/Inf -> e4m3 NaN
        return (sign << 7) | 0x7F
    if exp_b == 0:              # zero / fp32 subnormal -> zero
        return sign << 7
    e_un = exp_b - 127
    if e_un >= 9 or (e_un == 8 and mant >= 0x600000):   # saturate ±448
        return (sign << 7) | (0xF << 3) | 0x6
    if e_un >= -6:              # normal e4m3
        guard  = (mant >> 19) & 1
        sticky = 1 if (mant & ((1 << 19) - 1)) else 0
        lsb    = (mant >> 20) & 1
        m_e4m3 = (mant >> 20) & 0x7
        if guard and (sticky or lsb):
            m_e4m3 += 1
        e_out = e_un + 7
        if m_e4m3 == 8:
            m_e4m3 = 0
            e_out += 1
            if e_out > 15:
                return (sign << 7) | (0xF << 3) | 0x6
        if e_out == 15 and m_e4m3 == 7:
            m_e4m3 = 6
        return (sign << 7) | (e_out << 3) | m_e4m3
    shift = 23 - (e_un + 9)     # subnormal e4m3
    if shift > 31:
        return sign << 7
    implicit = (1 << 23) | mant
    guard  = (implicit >> (shift - 1)) & 1
    sticky = 1 if (implicit & ((1 << (shift - 1)) - 1)) else 0
    m_e4m3 = implicit >> shift
    lsb    = m_e4m3 & 1
    if guard and (sticky or lsb):
        m_e4m3 += 1
    if m_e4m3 >= 8:
        return (sign << 7) | (1 << 3)
    return (sign << 7) | m_e4m3

def e4m3_bits_to_fp32(code: int) -> float:
    """Decode a uint8 e4m3fn code to fp32 (NaN slot -> nan)."""
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0xF
    m = code & 0x7
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)     # subnormal
    if e == 15 and m == 7:
        return float("nan")
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd scripts/calibration && python -m pytest tests/test_ml8_e4m3_sim.py::test_gate_a_bit_match_kernel -v`
Expected: PASS (0 mismatches across the full battery, including the 256..448 band).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_e4m3_sim.py scripts/calibration/tests/test_ml8_e4m3_sim.py
git commit -m "feat(calib): bit-exact e4m3 encode/decode, Gate A vs kernel golden"
```

---

### Task 3: Vectorized per-row activation quant + scalar parity

**Files:**
- Modify: `scripts/calibration/ml8_e4m3_sim.py`
- Modify: `scripts/calibration/tests/test_ml8_e4m3_sim.py`

- [ ] **Step 1: Write the failing parity + round-trip tests**

```python
# append to tests/test_ml8_e4m3_sim.py
import torch
from ml8_e4m3_sim import (fp32_to_e4m3_bits, e4m3_bits_to_fp32,
                          e4m3_roundtrip, quantize_act_per_row)

def test_vectorized_roundtrip_matches_scalar():
    g = torch.Generator().manual_seed(0)
    # mix of ranges: normal, the 256..448 band, subnormal, saturation
    x = torch.cat([
        torch.randn(4000, generator=g) * 50.0,
        torch.linspace(250.0, 460.0, 1000),
        torch.linspace(-0.02, 0.02, 1000),
    ])
    vec = e4m3_roundtrip(x)
    scal = torch.tensor([e4m3_bits_to_fp32(fp32_to_e4m3_bits(float(v))) for v in x])
    # NaN slot can appear only for |x|>448 already handled; compare finite
    assert torch.equal(vec, scal), (
        f"max abs diff {torch.nan_to_num(vec - scal).abs().max().item()}")

def test_quantize_act_per_row_scale_and_eps():
    x = torch.tensor([[448.0, 224.0, 0.0, -448.0],
                      [0.0, 0.0, 0.0, 0.0]])          # row 1 all-zero -> eps path
    q = quantize_act_per_row(x)
    # row 0: absmax=448 -> scale=1.0 -> values land on lattice unchanged
    assert torch.allclose(q[0], torch.tensor([448.0, 224.0, 0.0, -448.0]))
    # row 1: all-zero stays zero (no nan/inf from eps division)
    assert torch.equal(q[1], torch.zeros(4))
```

- [ ] **Step 2: Run to verify failure**

Run: `cd scripts/calibration && python -m pytest tests/test_ml8_e4m3_sim.py -k "vectorized or per_row" -v`
Expected: FAIL — `cannot import name 'e4m3_roundtrip'`.

- [ ] **Step 3: Implement the vectorized port** (integer bit-ops on the int32 view — guaranteed faithful, no reliance on `torch.float8_e4m3fn`)

```python
# append to ml8_e4m3_sim.py
@torch.no_grad()
def e4m3_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Vectorized fp32 -> e4m3 -> fp32, bit-identical to the scalar path."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32).contiguous()
    bits = xf.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign  = (bits >> 31) & 1
    exp_b = (bits >> 23) & 0xFF
    mant  = bits & 0x7FFFFF
    e_un  = exp_b - 127

    out_code = torch.zeros_like(bits)
    is_nan_inf = exp_b == 0xFF
    is_zero    = exp_b == 0
    is_sat     = (e_un >= 9) | ((e_un == 8) & (mant >= 0x600000))
    is_normal  = (~is_nan_inf) & (~is_zero) & (~is_sat) & (e_un >= -6)
    is_sub     = (~is_nan_inf) & (~is_zero) & (~is_sat) & (e_un < -6)

    sat_code = (sign << 7) | (0xF << 3) | 0x6
    # normal
    guard  = (mant >> 19) & 1
    sticky = ((mant & ((1 << 19) - 1)) != 0).to(torch.int64)
    lsb    = (mant >> 20) & 1
    m_n    = (mant >> 20) & 0x7
    m_n    = m_n + (guard & (sticky | lsb))
    e_n    = e_un + 7
    carry  = (m_n == 8)
    m_n    = torch.where(carry, torch.zeros_like(m_n), m_n)
    e_n    = torch.where(carry, e_n + 1, e_n)
    normal_overflow = carry & (e_n > 15)
    nan_fix = (e_n == 15) & (m_n == 7)
    m_n = torch.where(nan_fix, torch.full_like(m_n, 6), m_n)
    normal_code = (sign << 7) | (e_n << 3) | m_n
    normal_code = torch.where(normal_overflow, sat_code, normal_code)
    # subnormal
    shift = (23 - (e_un + 9)).clamp(min=0)
    too_small = (23 - (e_un + 9)) > 31
    implicit = (1 << 23) | mant
    sh1 = (shift - 1).clamp(min=0)
    g_s = (implicit >> sh1) & 1
    st_s = ((implicit & ((1 << sh1) - 1)) != 0).to(torch.int64)
    m_s = implicit >> shift
    lsb_s = m_s & 1
    m_s = m_s + (g_s & (st_s | lsb_s))
    sub_overflow = m_s >= 8
    sub_code = (sign << 7) | m_s
    sub_code = torch.where(sub_overflow, (sign << 7) | (1 << 3), sub_code)
    sub_code = torch.where(too_small, sign << 7, sub_code)

    out_code = torch.where(is_nan_inf, (sign << 7) | 0x7F, out_code)
    out_code = torch.where(is_zero, sign << 7, out_code)
    out_code = torch.where(is_sat, sat_code, out_code)
    out_code = torch.where(is_normal, normal_code, out_code)
    out_code = torch.where(is_sub, sub_code, out_code)

    # decode
    c = out_code
    s = torch.where((c & 0x80) != 0, torch.full_like(xf, -1.0), torch.ones_like(xf))
    e = ((c >> 3) & 0xF).to(torch.float32)
    m = (c & 0x7).to(torch.float32)
    sub_val = s * (m / 8.0) * (2.0 ** -6)
    nan_slot = (e == 15) & ((c & 0x7) == 7)
    norm_val = s * (1.0 + m / 8.0) * torch.pow(torch.tensor(2.0), e - 7)
    val = torch.where(e == 0, sub_val, norm_val)
    val = torch.where(nan_slot, torch.full_like(val, float("nan")), val)
    return val.to(orig_dtype)

@torch.no_grad()
def quantize_act_per_row(x: torch.Tensor) -> torch.Tensor:
    """Per-row (per-token) e4m3 activation quant, kernel-faithful.

    x: [..., K]; the last dim is K. scale = row_absmax / 448 (eps-floored);
    returns dequantized fp32 a_fp8*scale, same shape & dtype as x.
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(ACT_SCALE_EPS)
    scale = absmax / E4M3_MAX
    q = e4m3_roundtrip(xf / scale) * scale
    return q.to(orig_dtype)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd scripts/calibration && python -m pytest tests/test_ml8_e4m3_sim.py -v`
Expected: PASS (all of Gate A, vectorized parity, per-row scale/eps).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_e4m3_sim.py scripts/calibration/tests/test_ml8_e4m3_sim.py
git commit -m "feat(calib): vectorized per-row e4m3 activation quant (scalar-parity)"
```

---

## Phase 2 — faithful forward integration

### Task 4: Rotation precompute helper

**Files:**
- Create: `scripts/calibration/faithful_forward.py`
- Create: `scripts/calibration/tests/test_faithful_forward.py`

Context: the inline rotation build is calibrate_ml8_paged.py:1390–1397 and calibrate_ml8.py:445–451. We lift it so rotations exist *before* the Hessian forward.

- [ ] **Step 1: Write the failing determinism test**

```python
# scripts/calibration/tests/test_faithful_forward.py
import sys
from pathlib import Path
import torch
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
from faithful_forward import build_rotations   # noqa: E402

def test_build_rotations_matches_inline_formula():
    # mirrors calibrate_ml8_paged.py:1390-1397 seed math
    dims = {("L0", "ffn_gate"): 2560}
    seeds = {("L0", "ffn_gate"): 5 + 0 * 7 + 0}
    rots = build_rotations(dims, seeds, max_b=1024)
    from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
    a, b = factor_for_dim(2560, max_b=1024)
    ref = KroneckerRotation(h_a=random_orthogonal(a, seed=5), b_dim=b)
    x = torch.randn(3, 2560)
    assert torch.allclose(rots[("L0", "ffn_gate")].forward(x), ref.forward(x), atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_forward.py::test_build_rotations_matches_inline_formula -v`
Expected: FAIL — `No module named 'faithful_forward'`.

- [ ] **Step 3: Implement `build_rotations`**

```python
# scripts/calibration/faithful_forward.py
"""Deployment-faithful (W4A8) calibration forward: rotation precompute + the
activation-e4m3 pre-hook. See docs/superpowers/specs/2026-05-31-w4a8-faithful-calibration-design.md.
"""
import torch
from kronecker_rotation import (KroneckerRotation, random_orthogonal,
                                factor_for_dim)

def build_rotations(dims: dict, seeds: dict, max_b: int = 1024) -> dict:
    """dims/seeds keyed by (layer_key, kind) -> rotation. Built from dims+seeds
    only (never from H values), so it can run before Hessian collection."""
    rots = {}
    for key, K in dims.items():
        a, b = factor_for_dim(K, max_b=max_b)
        rots[key] = KroneckerRotation(h_a=random_orthogonal(a, seed=int(seeds[key])), b_dim=b)
    return rots
```

- [ ] **Step 4: Run to verify pass**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_forward.py::test_build_rotations_matches_inline_formula -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/faithful_forward.py scripts/calibration/tests/test_faithful_forward.py
git commit -m "feat(calib): build_rotations precompute (faithful forward)"
```

---

### Task 5: The faithful activation pre-hook

**Files:**
- Modify: `scripts/calibration/faithful_forward.py`
- Modify: `scripts/calibration/tests/test_faithful_forward.py`

- [ ] **Step 1: Write failing tests — x_eff identity, rotated-space H, guard flag**

```python
# append to tests/test_faithful_forward.py
import torch.nn as nn
from faithful_forward import FaithfulActHook
from ml8_e4m3_sim import quantize_act_per_row

def _rot(K):
    from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
    a, b = factor_for_dim(K, max_b=1024)
    return KroneckerRotation(h_a=random_orthogonal(a, seed=1), b_dim=b)

def test_x_eff_is_faithful_output_via_unchanged_linear():
    K, N, T = 256, 8, 5
    lin = nn.Linear(K, N, bias=False)
    rot = _rot(K)
    hook = FaithfulActHook(rot, enabled=True)
    x = torch.randn(T, K)
    # reference faithful output: e4m3(x@Q) @ (Q^T W^T)
    aq = quantize_act_per_row(rot.forward(x))
    W = lin.weight.data.float()
    y_ref = aq @ rot.forward(W).t()          # rot.forward(W) = W@Q ; (W@Q)^T
    # hook replaces input with x_eff; unchanged linear then yields y_ref
    h = lin.register_forward_pre_hook(hook, with_kwargs=False)
    y_got = lin(x)
    h.remove()
    assert torch.allclose(y_got, y_ref, atol=1e-4)

def test_disabled_hook_is_identity():
    K = 256
    lin = nn.Linear(K, 4, bias=False)
    hook = FaithfulActHook(_rot(K), enabled=False)
    x = torch.randn(3, K)
    h = lin.register_forward_pre_hook(hook)
    assert torch.allclose(lin(x), lin._original_forward_input_passthrough(x)
                          if False else torch.nn.functional.linear(x, lin.weight))
    h.remove()

def test_hessian_accumulates_in_rotated_quant_space():
    K, T = 256, 7
    rot = _rot(K)
    hook = FaithfulActHook(rot, enabled=True)
    hook.set_hessian_target(True)
    lin = nn.Linear(K, 4, bias=False)
    h = lin.register_forward_pre_hook(hook)
    x = torch.randn(T, K)
    lin(x)
    h.remove()
    aq = quantize_act_per_row(rot.forward(x))
    assert torch.allclose(hook.H, aq.t() @ aq, atol=1e-3)
    assert hook.n_tokens == T
```

- [ ] **Step 2: Run to verify failure**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_forward.py -k "x_eff or disabled or rotated_quant" -v`
Expected: FAIL — `cannot import name 'FaithfulActHook'`.

- [ ] **Step 3: Implement `FaithfulActHook`** (a `forward_pre_hook`: returns replacement input `x_eff`; accumulates `H` in rotated+quant space only when it is the active Hessian target)

```python
# append to faithful_forward.py
from ml8_e4m3_sim import quantize_act_per_row

class FaithfulActHook:
    """forward_pre_hook on an ml8-4 linear. When enabled, replaces the input x
    with x_eff = e4m3(x@Q) @ Q^T so the unchanged linear yields the faithful
    W4A8 output, and (when this layer is the active Hessian target) accumulates
    H += a_q^T a_q in rotated+quantized space (so rotate_hessian must NOT run)."""
    def __init__(self, rotation, enabled: bool = True):
        self.rotation = rotation
        self.enabled = enabled
        self._is_target = False
        self.H = None
        self.n_tokens = 0

    def set_hessian_target(self, on: bool):
        self._is_target = on

    def reset_hessian(self):
        self.H = None
        self.n_tokens = 0

    def __call__(self, module, args):
        if not self.enabled:
            return None                      # no-op: original input flows
        x = args[0]
        orig_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1]).float()      # [T, K]
        a_rot = self.rotation.forward(flat)            # x@Q
        a_q = quantize_act_per_row(a_rot)              # e4m3 per-row
        if self._is_target:
            XtX = a_q.t() @ a_q
            self.H = XtX if self.H is None else self.H + XtX
            self.n_tokens += a_q.shape[0]
        x_eff = self.rotation.forward(a_q.t()).t()     # a_q @ Q^T  (= forward(a_q^T)^T)
        x_eff = x_eff.reshape(x.shape).to(orig_dtype)
        return (x_eff,) + tuple(args[1:])
```

Note on `a_q @ Qᵀ`: with the row-vector convention `rotation.forward(M) = M @ Q`, the transpose identity `M @ Qᵀ = rotation.forward(Mᵀ)ᵀ` (same pattern as `rotate_hessian`, kronecker_rotation.py:137–140).

- [ ] **Step 4: Run to verify pass**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_forward.py -v`
Expected: PASS (x_eff faithful, disabled is identity, H in rotated+quant space).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/faithful_forward.py scripts/calibration/tests/test_faithful_forward.py
git commit -m "feat(calib): FaithfulActHook (x_eff propagation + rotated-space H)"
```

---

### Task 6: Wire `--faithful-acts` into the dense calibration branch

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py` (argparse near line 894; dense branch ~1660–1706)

Read first: calibrate_ml8_paged.py:1655–1710 (dense branch — per-target-linear loop calling `compute_hessian`, the rotation build, `rotate_hessian`, `gptq_quantize_linear`), and the role classifier `role_targets.classify_role` (role_targets.py).

- [ ] **Step 1: Add the CLI flag**

After the `--rotation` argument (calibrate_ml8_paged.py:894), add:

```python
    p.add_argument("--faithful-acts", action="store_true",
                   help="W4A8: collect Hessians on rotated, per-row e4m3-quantized "
                        "activations and propagate them (drops algebraic rotate_hessian).")
```

- [ ] **Step 2: Write a guard test (the double-rotation guard)**

```python
# scripts/calibration/tests/test_faithful_guard.py
import sys
from pathlib import Path
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
import faithful_forward as ff

def test_assert_no_double_rotation_helper():
    # When faithful-acts is on, rotate_hessian must not be applied to H again.
    assert hasattr(ff, "assert_not_double_rotated")
    ff.assert_not_double_rotated(faithful_acts=True, rotate_hessian_called=False)  # ok
    try:
        ff.assert_not_double_rotated(faithful_acts=True, rotate_hessian_called=True)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
```

- [ ] **Step 3: Run to verify failure**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_guard.py -v`
Expected: FAIL — `module 'faithful_forward' has no attribute 'assert_not_double_rotated'`.

- [ ] **Step 4: Add the guard helper to `faithful_forward.py`**

```python
# append to faithful_forward.py
def assert_not_double_rotated(faithful_acts: bool, rotate_hessian_called: bool):
    """Guard: with faithful-acts the rotation is already baked into H by the
    forward; calling rotate_hessian again double-rotates."""
    if faithful_acts and rotate_hessian_called:
        raise RuntimeError(
            "double-rotation: faithful-acts builds H in rotated space; "
            "rotate_hessian must be skipped.")
```

- [ ] **Step 5: Run to verify pass**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_guard.py -v`
Expected: PASS.

- [ ] **Step 6: Integrate into the dense branch**

In the dense branch of `calibrate_ml8_paged.py`, around the per-target-linear loop (~1660–1706):

1. Before the Hessian loop, when `args.faithful_acts`: build the rotation for each target linear up front via `faithful_forward.build_rotations` (use the same per-(layer,kind) seed math already at 1390–1397: `seed = args.rotation_seed + layer_idx*7 + kind_seed_offset[kind]`), and register one `FaithfulActHook(rot, enabled=True)` as a `forward_pre_hook` on every ml8-4 target linear (those whose role classifies `Tier.ML8`). Keep the hook handles + a dict `hooks_by_target`.
2. In the per-target loop, set `hooks_by_target[target].set_hessian_target(True)` and `.reset_hessian()` before the forward, run the forward, take `H = hooks_by_target[target].H`, then `.set_hessian_target(False)`. This replaces the `compute_hessian` XtX accumulation for that target when faithful-acts is on (the existing `compute_hessian` path stays for the `--faithful-acts` OFF case).
3. Replace the rotation block: when `args.faithful_acts`, do **not** call `rotate_hessian(H, rotation)` (H is already rotated); still rotate the **weight** for GPTQ exactly as today (`layer.weight.data.copy_(rotation.forward(...))`) and still write `rotation_blob`. Call `faithful_forward.assert_not_double_rotated(args.faithful_acts, rotate_hessian_called=False)` right after the (skipped) rotate_hessian site.
4. After GPTQ + weight restore for a target, the persistent pre-hook keeps propagating that layer's faithful output to downstream Hessian forwards (no extra code — the weight_override/restore already in place).

Show the integrated rotation block (the only subtle edit):

```python
        rotation = rotations_by_target[layer]    # precomputed when faithful_acts
        rotation_blob = rotation.to_dict(); rotation_blob["seed"] = seed_for(layer)
        if not args.faithful_acts:
            H = rotate_hessian(H, rotation)       # legacy algebraic path
        ff.assert_not_double_rotated(args.faithful_acts, rotate_hessian_called=False)
        # rotate the WEIGHT for GPTQ in both modes (unchanged):
        W_dtype = layer.weight.dtype
        layer.weight.data.copy_(rotation.forward(layer.weight.data.float().to(H.device)).to(W_dtype))
```

- [ ] **Step 7: Smoke-run the wiring on CPU with a tiny synthetic stack**

```python
# scripts/calibration/tests/test_faithful_wiring_smoke.py
import sys
from pathlib import Path
import torch, torch.nn as nn
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
import faithful_forward as ff

def test_persistent_hooks_propagate_then_hessian_targets_one():
    torch.manual_seed(0)
    K = 256
    l1, l2 = nn.Linear(K, K, bias=False), nn.Linear(K, 4, bias=False)
    model = nn.Sequential(l1, l2)
    rots = {l1: ff.build_rotations({"a": K}, {"a": 1})["a"],
            l2: ff.build_rotations({"b": K}, {"b": 2})["b"]}
    hooks = {m: ff.FaithfulActHook(r, enabled=True) for m, r in rots.items()}
    handles = [m.register_forward_pre_hook(h) for m, h in hooks.items()]
    hooks[l2].set_hessian_target(True)
    x = torch.randn(6, K)
    model(x)
    for h in handles: h.remove()
    assert hooks[l2].H is not None and hooks[l1].H is None   # only l2 targeted
    assert hooks[l2].H.shape == (K, K)
```

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_wiring_smoke.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py scripts/calibration/faithful_forward.py \
        scripts/calibration/tests/test_faithful_guard.py scripts/calibration/tests/test_faithful_wiring_smoke.py
git commit -m "feat(calib): wire --faithful-acts into dense branch (hooks + guard)"
```

---

## Phase 3 — fp8 weight tiers

### Task 7: `--faithful-weights` (embed + ssm α/β fp8 overrides)

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py` (argparse; dense branch model-load region)
- Modify: `scripts/calibration/faithful_forward.py`
- Modify: `scripts/calibration/tests/test_faithful_forward.py`

- [ ] **Step 1: Add the CLI flag** (after `--faithful-acts`)

```python
    p.add_argument("--faithful-weights", action="store_true",
                   help="W4A8: simulate the fp8 weight tiers (token_embd, ssm alpha/beta) "
                        "via scaled-FP8 quant->dequant overrides during the calib forward.")
```

- [ ] **Step 2: Write the failing override test**

```python
# append to tests/test_faithful_forward.py
from faithful_forward import fp8_weight_override

def test_fp8_weight_override_roundtrips_through_scaled_fp8():
    import torch
    from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8
    w = torch.randn(16, 64)
    got = fp8_weight_override(w, group_size=32)
    want = dequantize_scaled_fp8(quantize_scaled_fp8(w, group_size=32))
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, w)        # it actually changed the weights
```

- [ ] **Step 3: Run to verify failure**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_forward.py::test_fp8_weight_override_roundtrips_through_scaled_fp8 -v`
Expected: FAIL — `cannot import name 'fp8_weight_override'`.

- [ ] **Step 4: Implement the override helper**

```python
# append to faithful_forward.py
def fp8_weight_override(w, group_size: int = 32):
    """Quant->dequant a weight through the scaled-FP8 tier (Exec T2). Returns the
    dequantized fp32 weight to install as a forward-time override."""
    from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8
    return dequantize_scaled_fp8(quantize_scaled_fp8(w.float(), group_size=group_size)).to(w.dtype)
```

- [ ] **Step 5: Run to verify pass**

Run: `cd scripts/calibration && python -m pytest tests/test_faithful_forward.py::test_fp8_weight_override_roundtrips_through_scaled_fp8 -v`
Expected: PASS.

- [ ] **Step 6: Integrate into the dense branch model-load**

After the model is resident and before the Hessian loop, when `args.faithful_weights`: iterate named modules, classify each via `role_targets.classify_role`; for any `Tier.FP8` (`token_embd`, `ssm_alpha`, `ssm_beta`), replace `module.weight.data` with `fp8_weight_override(module.weight.data)` (in place). Print a one-line count: `[faithful-weights] overrode N fp8-tier tensors`. (No rotation/e4m3 hook on these — they are a direct weight cast, matching deployment.)

- [ ] **Step 7: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py scripts/calibration/faithful_forward.py \
        scripts/calibration/tests/test_faithful_forward.py
git commit -m "feat(calib): --faithful-weights fp8 tier overrides (embed, ssm a/b)"
```

---

## Phase 4 — measurement harness + GPU gates

### Task 8: Gauntlet pass-through + paired-toggle cell pair

**Files:**
- Modify: `scripts/calibration/method_gauntlet.py` (`recipe_args`, STAGES, `run_cell` is unchanged)

Read first: method_gauntlet.py `recipe_args` (builds the calibrate arg list from an overrides dict) and the STAGES list.

- [ ] **Step 1: Confirm `recipe_args` forwards boolean flags**

`recipe_args` must translate `{"--faithful-acts": True}` into a bare `--faithful-acts` (store_true). Read the existing `recipe_args`; if it only handles `key value` pairs, add: a value of `True` appends just the key, `False`/`None` appends nothing.

```python
# in recipe_args(overrides): for k, v in overrides.items():
    if v is True:
        args.append(k)
    elif v not in (False, None):
        args += [k, str(v)]
```

- [ ] **Step 2: Add a paired-toggle stage**

Add a STAGES entry `qat` with the four measurement cells from the spec (shared corpus seed via the existing corpus flags so the pair is paired):

```python
("qat", [
    ("q1_off",      {"--corpus": "wiki", "--n-samples": "128", "--seq-len": "2048"}),
    ("q2_acts",     {"--corpus": "wiki", "--n-samples": "128", "--seq-len": "2048",
                     "--faithful-acts": True}),
    ("q3_actswt",   {"--corpus": "wiki", "--n-samples": "128", "--seq-len": "2048",
                     "--faithful-acts": True, "--faithful-weights": True}),
    ("q4_heavy",    {"--corpus": "wiki", "--n-samples": "128", "--seq-len": "2048",
                     "--faithful-acts": True, "--faithful-weights": True,
                     "--heavy-rounds": "2"}),
]),
```

- [ ] **Step 3: Verify the arg-building (no calibration, dry inspect)**

```python
# scripts/calibration/tests/test_gauntlet_args.py
import sys
from pathlib import Path
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
import method_gauntlet as mg

def test_recipe_args_forwards_store_true():
    out = mg.recipe_args({"--faithful-acts": True, "--n-samples": "128",
                          "--faithful-weights": False})
    assert "--faithful-acts" in out
    assert "--faithful-weights" not in out
    i = out.index("--n-samples"); assert out[i + 1] == "128"
```

Run: `cd scripts/calibration && python -m pytest tests/test_gauntlet_args.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibration/method_gauntlet.py scripts/calibration/tests/test_gauntlet_args.py
git commit -m "feat(calib): gauntlet paired-toggle qat stage + store_true forwarding"
```

---

### Task 9: Gate C — refactor neutrality (GPU CHECKPOINT)

> **CHECKPOINT — flag the human before running.** This is a real 0.8B calibration + PPL (~25–30 min on the R9700 with the fla-fp32 shim). Confirm the box is clear (no concurrent heavy build; ~11 GB RAM free) before dispatching.

**Files:** none (a run + an assertion).

- [ ] **Step 1: Run the faithful-OFF cell** through the gauntlet on a fresh workdir:

```bash
cd /home/kmbandy/GitHub/llama.cpp && python scripts/calibration/method_gauntlet.py \
  --workdir /home/kmbandy/models/gauntlet-0p8b-qat --stage qat --cell q1_off \
  --cal-device cuda:0 --ppl-device ROCm0 --model <0.8B HF path> --base <0.8B base gguf> --arch <arch>
```

- [ ] **Step 2: Assert neutrality** — `q1_off` wiki PPL must reproduce **19.2678 ± 0.01** (faithful flags off ⟹ identical to the fla zero-point; the small tolerance covers only nondeterministic reduction order, not a real delta).
Expected: `q1_off` PPL ∈ [19.258, 19.278]. If outside, STOP — the refactor moved the baseline; debug before trusting any faithful number (systematic-debugging skill).

- [ ] **Step 3: Record** the q1_off number in `docs/superpowers/2026-05-31-COMPACT-HANDOFF.md` results table.

---

### Task 10: Paired-toggle measurement + 3-seed finalize (GPU CHECKPOINT)

> **CHECKPOINT — flag the human.** Four calibrations (q1–q4) ≈ 2 hrs; the q4 heavy round adds time. The 3-seed finalize is a further 3× on the winner. Confirm scope/time with the human before dispatching; do not run blind.

**Files:** none (runs + the analysis write-up).

- [ ] **Step 1: Run the full qat stage** (`--stage qat`, all four cells) on the fresh workdir, same corpus seed across cells (paired).

- [ ] **Step 2: Compute the paired deltas** vs `q1_off`:
  - `q2_acts − q1_off` = activation-e4m3 effect (heavy off).
  - `q3_actswt − q1_off` = full faithful forward, heavy off (no-regression check: must be ≤ +0.05).
  - `q4_heavy − q1_off` = the product config (target: a real improvement clearing −0.05, i.e. PPL below 19.2678 − 0.05).

- [ ] **Step 3: 3-seed finalize on the winner** — rerun the best config + its OFF control across 3 corpus seeds (`--corpus-seed`), report mean ± sd. The winner's mean must clear the ±0.05 floor to count as real.

- [ ] **Step 4: Write the results + verdict** into `docs/superpowers/2026-05-31-calibration-fidelity-fla-rdna.md` (new "W4A8 measurement" section) and a KG `session_summary`. State honestly if the gain is null/negative — that is a valid finding, not a failure.

---

## Self-Review (run against the spec)

**Spec coverage:**
- Bit-match `ml8_fp32_to_e4m3` → Task 1 (golden) + Task 2 (Gate A). ✓ (and the C-ref-vs-kernel divergence is called out)
- Per-row dynamic scale `row_absmax/448` + eps → Task 3 `quantize_act_per_row`. ✓
- Mirror-in-forward + drop `rotate_hessian` + no double-rotation → Task 5 (`FaithfulActHook`, x_eff), Task 6 (skip rotate_hessian + `assert_not_double_rotated`). ✓
- Rotation precompute (break circularity) → Task 4. ✓
- fp8 weight tiers via scaled_fp8 → Task 7. ✓
- Toggles `--faithful-acts`/`--faithful-weights` → Tasks 6, 7. ✓
- Gate A/B/C → Task 2 (A), Task 3 per-row (B-equivalent on synthetic; **note**: the spec's Gate B against a live `ml8_quantize_activations_kernel` dump is folded into Task 3's vectorized-parity + the Task 9 end-to-end neutrality, since the kernel quant has no learned params and Gate A already pins the per-element math — a separate kernel-activation dump is optional and not gating), Task 9 (C). ✓
- Paired-toggle + 3-seed measurement → Tasks 8, 10. ✓
- Which-linears via `role_targets` → Tasks 6, 7. ✓
- Ground-truth rule → header + Task 1 note. ✓

**Placeholder scan:** `<0.8B HF path>` / `<arch>` in Task 9 are run-time inputs the operator supplies (the same ones the prior gauntlet runs used), not code placeholders — acceptable. No TBD/TODO in code steps.

**Type consistency:** `FaithfulActHook(rotation, enabled)`, `.set_hessian_target(bool)`, `.reset_hessian()`, `.H`, `.n_tokens`; `build_rotations(dims, seeds, max_b)`; `quantize_act_per_row(x)`, `e4m3_roundtrip(x)`, `fp32_to_e4m3_bits(float)→int`, `e4m3_bits_to_fp32(int)→float`; `assert_not_double_rotated(faithful_acts, rotate_hessian_called)`; `fp8_weight_override(w, group_size)` — consistent across all tasks.

**One deviation from the spec, made explicit:** spec Gate B (compare against a real `ml8_quantize_activations_kernel` dump) is downgraded from a hard gate to optional, because Gate A already bit-pins the per-element transform and the activation quant carries no learned parameters — so a kernel dump would only re-test the per-row scale arithmetic, which Task 3 covers on synthetic data and Task 9 covers end-to-end. If the operator wants the literal kernel-dump comparison, it slots in as an extra Task-3 test using the existing dump tooling (`compare_hip_vs_python_layer0.py` pattern).
