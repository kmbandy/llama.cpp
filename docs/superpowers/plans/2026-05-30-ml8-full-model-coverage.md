# ml8 Full-Model Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ml8 from FFN-only to the whole model — 4-bit ml8 for every rotatable GEMM, a new 8-bit scaled-FP8 tier for the embedding + SSM gate projections, f32 for the recurrence core — so dense qwen35 is a real ~4-bit product, validated end-to-end on the 9B before the 35B MoE.

**Architecture:** A new `GGML_TYPE_ML8_FP8` (e4m3 values + per-group fp16 scale) is the foundation; a role-tagged calibration target set feeds the existing per-matrix ml8 pipeline for GEMMs and a new direct-cast quantizer for the FP8 tier; the converter writes both tiers + a refined coverage report; C++ inference gains a generic `ml8-aware mul_mat` helper + sidecar registry, an FP8 `get_rows` case, and a no-LUT FP8-WMMA `mul_mat` for α/β. Dependency-chained, so phases land bottom-up.

**Tech Stack:** Python (torch, gguf-py) calibration/converter; C/C++ ggml + llama.cpp graph; HIP/CUDA (ml8.cu, getrows.cu) kernels; gfx1201/gfx1030 multi-arch HIP build.

**Spec:** `docs/superpowers/specs/2026-05-30-ml8-full-model-coverage-design.md`

**Branch:** create `feat/ml8-full-model-coverage` off `feat/upstream-merge-2026-05-27`.

**Conventions for every task:** run Python tests from `scripts/calibration/` with `python3 <test_file>.py`. Build HIP with the multi-arch invocation (`-DAMDGPU_TARGETS="gfx1201;gfx1030"`). Pass `--no-mmap` to all llama.cpp tools. Never write GGUFs to `/tmp` — use `/home/kmbandy/models/`. Commit only the files a task names.

---

## File Structure

**Calibration (Python, `scripts/calibration/`)**
- Modify `calibrate_ml8_paged.py` — add the role-tagged name map + capture hooks for the new linears + the FP8-tier dispatch.
- Create `role_targets.py` — the role classifier: `{HF module path → (gguf_name, role, tier)}`. One responsibility: name→role mapping.
- Create `scaled_fp8.py` — the scaled-FP8 quantizer (per-group scale + e4m3 cast) + dequant. Pure, no torch-model deps.
- Create `ssm_sensitivity.py` — kurtosis + quant-sensitivity instrument for α/β/A/dt.
- Create tests: `test_role_targets.py`, `test_scaled_fp8.py`, `test_ssm_sensitivity.py`.

**Format/converter (Python)**
- Modify `gguf-py/gguf/constants.py` — register `ML8_FP8` quant type + block size.
- Modify `scripts/calibration/ml8_to_gguf.py` — discover the new ml8 roles; write the FP8 tier; refine `evaluate_coverage`.
- Modify `scripts/calibration/test_ml8_to_gguf.py` — coverage-with-FP8 tests.

**C type traits (C/C++)**
- Modify `ggml/src/ggml.c` — `GGML_TYPE_ML8_FP8` traits + `dequantize_row_ml8_fp8` (`to_float`).
- Modify `ggml/include/ggml.h` / `ggml/src/ggml-common.h` — the `block_ml8_fp8` struct + enum.

**CUDA kernels (HIP/CUDA, `ggml/src/ggml-cuda/`)**
- Modify `getrows.cu` — `GGML_TYPE_ML8_FP8` gather case.
- Modify `ml8.cu` — a no-LUT FP8-WMMA `mul_mat` path (the α/β consumer).
- Modify `ggml-cuda.cu` — op/type dispatch for the FP8 mul_mat where needed.

**Graph wiring (C++, `src/`)**
- Create `src/llama-ml8-registry.{h,cpp}` — the load-time weight→sidecar registry + `build_ml8_or_mul_mat` helper.
- Modify `src/llama-graph.cpp` — route qwen35 attn/ssm/lm_head/eh_proj GEMMs through the helper; wire embed + α/β.
- Modify `src/llama-model.cpp` / `llama-model-loader.cpp` — load the new sidecars + FP8 tensors into the registry.

**Validation (Python/scripts)**
- Create `scripts/calibration/check_bit_equivalence.py` — Python `Ml8Linear` ↔ C++ graph PPL parity.
- Create `scripts/calibration/longctx_probe.py` — long-context eval for SSM-gate compounding.

---

## Phase 1 — Scaled-FP8 representation + role-tagged calibration (Python foundation)

### Task 1: Role classifier (`role_targets.py`)

**Files:**
- Create: `scripts/calibration/role_targets.py`
- Test: `scripts/calibration/test_role_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# test_role_targets.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from role_targets import classify_role, Tier

def test_ml8_gemm_roles():
    cases = {
        "model.layers.0.self_attn.q_proj":   ("blk.0.attn_q.weight",     "attn_q",   Tier.ML8),
        "model.layers.3.self_attn.k_proj":   ("blk.3.attn_k.weight",     "attn_k",   Tier.ML8),
        "model.layers.3.self_attn.v_proj":   ("blk.3.attn_v.weight",     "attn_v",   Tier.ML8),
        "model.layers.3.self_attn.o_proj":   ("blk.3.attn_output.weight","attn_out", Tier.ML8),
        "model.layers.5.linear_attn.out_proj":("blk.5.ssm_out.weight",   "ssm_out",  Tier.ML8),
        "lm_head":                            ("output.weight",          "lm_head",  Tier.ML8),
    }
    for hf, (gguf, role, tier) in cases.items():
        assert classify_role(hf) == (gguf, role, tier), f"{hf} -> {classify_role(hf)}"

def test_scaled_fp8_roles():
    assert classify_role("model.embed_tokens")[2] is Tier.FP8
    assert classify_role("model.layers.2.linear_attn.alpha_proj")[2] is Tier.FP8
    assert classify_role("model.layers.2.linear_attn.beta_proj")[2] is Tier.FP8

def test_native_left_alone():
    # A/dt/conv/norms must classify as NATIVE (skip), never ML8/FP8.
    assert classify_role("model.layers.0.linear_attn.conv1d")[2] is Tier.NATIVE
    assert classify_role("model.layers.0.input_layernorm")[2] is Tier.NATIVE

if __name__ == "__main__":
    test_ml8_gemm_roles(); test_scaled_fp8_roles(); test_native_left_alone()
    print("ALL ROLE TESTS PASSED")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 test_role_targets.py`
Expected: `ModuleNotFoundError: No module named 'role_targets'`

- [ ] **Step 3: Implement `role_targets.py`**

> NOTE: the exact HF submodule names for the gated-deltanet layer (`alpha_proj`/`beta_proj`/`out_proj` under `linear_attn`) must be confirmed against the transformers Qwen3.5 modelling file during implementation — the GGUF names (`ssm_out`, `ssm_alpha`, `ssm_beta`) are authoritative (verified from the 9B GGUF); map whatever HF names produce them. Build the map by iterating `model.named_modules()` the way `_qwen_mlp_name_map` does (`calibrate_ml8_paged.py:95`).

```python
# role_targets.py
import enum, re

class Tier(enum.Enum):
    ML8 = "ml8"        # 4-bit ml8 GEMM, full GPTQ pipeline
    FP8 = "fp8"        # 8-bit scaled-FP8, direct cast
    NATIVE = "native"  # leave as-is (A/dt/conv/norms/embed-not-matmul handled by caller)

# suffix -> (gguf_suffix, role)
_ML8 = {
    "q_proj": ("attn_q", "attn_q"), "k_proj": ("attn_k", "attn_k"),
    "v_proj": ("attn_v", "attn_v"), "o_proj": ("attn_output", "attn_out"),
    "qkv_proj": ("attn_qkv", "attn_qkv"), "gate_proj_attn": ("attn_gate", "attn_gate"),
    "out_proj": ("ssm_out", "ssm_out"),
    "gate_proj": ("ffn_gate", "ffn_gate"), "up_proj": ("ffn_up", "ffn_up"),
    "down_proj": ("ffn_down", "ffn_down"),
}
_FP8 = {"alpha_proj": ("ssm_alpha", "ssm_alpha"), "beta_proj": ("ssm_beta", "ssm_beta")}

def _layer_idx(name):
    parts = name.split(".")
    try: return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError): return None

def classify_role(hf_name: str):
    """Return (gguf_name, role, Tier). NATIVE for anything we don't quantize."""
    if hf_name in ("lm_head", "model.lm_head"):
        return ("output.weight", "lm_head", Tier.ML8)
    if hf_name.endswith("eh_proj"):
        L = _layer_idx(hf_name)
        return (f"blk.{L}.nextn.eh_proj.weight" if L is not None else "nextn.eh_proj.weight",
                "eh_proj", Tier.ML8)
    if hf_name in ("model.embed_tokens", "model.embed_tokens.weight"):
        return ("token_embd.weight", "token_embd", Tier.FP8)
    L = _layer_idx(hf_name)
    suffix = hf_name.split(".")[-1]
    if suffix in _ML8 and L is not None:
        g, role = _ML8[suffix]; return (f"blk.{L}.{g}.weight", role, Tier.ML8)
    if suffix in _FP8 and L is not None:
        g, role = _FP8[suffix]; return (f"blk.{L}.{g}.weight", role, Tier.FP8)
    return (hf_name, "native", Tier.NATIVE)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 test_role_targets.py` → Expected: `ALL ROLE TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/role_targets.py scripts/calibration/test_role_targets.py
git commit -m "feat(ml8): role classifier for full-model target set"
```

### Task 2: Scaled-FP8 quantizer (`scaled_fp8.py`)

**Files:**
- Create: `scripts/calibration/scaled_fp8.py`
- Test: `scripts/calibration/test_scaled_fp8.py`

- [ ] **Step 1: Write the failing test** — round-trip SNR + shape/scale-grouping contract.

```python
# test_scaled_fp8.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8

def _snr_db(w, wq):
    return 10*torch.log10((w.pow(2).sum()/(w-wq).pow(2).sum()).clamp_min(1e-12)).item()

def test_roundtrip_snr():
    torch.manual_seed(0)
    w = torch.randn(256, 128)                       # [N, K]
    packed = quantize_scaled_fp8(w, group_size=32)  # per-group along K
    wq = dequantize_scaled_fp8(packed)
    assert wq.shape == w.shape
    # 8-bit e4m3 + per-group scale on gaussian weights: expect comfortably > 30 dB.
    assert _snr_db(w, wq) > 30.0, f"SNR too low: {_snr_db(w, wq):.1f} dB"

def test_scale_grouping_shape():
    w = torch.randn(64, 256)
    packed = quantize_scaled_fp8(w, group_size=32)
    assert packed["scale"].shape == (64, 256 // 32)   # one fp16 scale per (row, K-group)
    assert packed["e4m3"].shape == (64, 256)

def test_zero_group_no_nan():
    w = torch.zeros(8, 32); w[0, :] = 1.0
    wq = dequantize_scaled_fp8(quantize_scaled_fp8(w, group_size=32))
    assert not torch.isnan(wq).any()

if __name__ == "__main__":
    test_roundtrip_snr(); test_scale_grouping_shape(); test_zero_group_no_nan()
    print("ALL SCALED-FP8 TESTS PASSED")
```

- [ ] **Step 2: Run to verify it fails** — `python3 test_scaled_fp8.py` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scaled_fp8.py`** — reuse the existing e4m3 snap from `centroid_quantizer.py` (the calibration-side `snap_to_e4m3`) so disk encoding matches the C++ `ggml_cuda_ue4m3_to_fp32` decode.

```python
# scaled_fp8.py
import torch
from centroid_quantizer import snap_to_e4m3   # existing calibration e4m3 round (E4M3 lattice)

E4M3_MAX = 448.0

def quantize_scaled_fp8(w: torch.Tensor, group_size: int = 32) -> dict:
    """Per-group (along K, the last dim) scale + e4m3 cast. w: [N, K]."""
    N, K = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    g = K // group_size
    wg = w.reshape(N, g, group_size)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / E4M3_MAX   # [N, g, 1]
    e4m3 = snap_to_e4m3(wg / scale).reshape(N, K)                             # e4m3 lattice
    return {"e4m3": e4m3.to(torch.float32), "scale": scale.reshape(N, g).to(torch.float16),
            "group_size": group_size, "shape": (N, K)}

def dequantize_scaled_fp8(packed: dict) -> torch.Tensor:
    N, K = packed["shape"]; gs = packed["group_size"]; g = K // gs
    e4m3 = packed["e4m3"].reshape(N, g, gs)
    scale = packed["scale"].to(torch.float32).reshape(N, g, 1)
    return (e4m3 * scale).reshape(N, K)
```

- [ ] **Step 4: Run to verify it passes** — `python3 test_scaled_fp8.py` → `ALL SCALED-FP8 TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/scaled_fp8.py scripts/calibration/test_scaled_fp8.py
git commit -m "feat(ml8): scaled-FP8 quantizer (e4m3 + per-group scale)"
```

### Task 3: SSM sensitivity instrument (`ssm_sensitivity.py`)

**Files:**
- Create: `scripts/calibration/ssm_sensitivity.py`
- Test: `scripts/calibration/test_ssm_sensitivity.py`

- [ ] **Step 1: Write the failing test**

```python
# test_ssm_sensitivity.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from ssm_sensitivity import kurtosis, fp8_sensitivity_db

def test_kurtosis_gaussian_near_3():
    torch.manual_seed(0)
    k = kurtosis(torch.randn(100000))
    assert 2.5 < k < 3.5, f"gaussian kurtosis {k}"

def test_fp8_sensitivity_positive_db():
    torch.manual_seed(1)
    w = torch.randn(64, 128)
    db = fp8_sensitivity_db(w, group_size=32)   # SNR of scaled-fp8 reconstruction
    assert db > 25.0

if __name__ == "__main__":
    test_kurtosis_gaussian_near_3(); test_fp8_sensitivity_positive_db()
    print("ALL SSM-SENSITIVITY TESTS PASSED")
```

- [ ] **Step 2: Run to verify it fails** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ssm_sensitivity.py`**

```python
# ssm_sensitivity.py
import torch
from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8

def kurtosis(x: torch.Tensor) -> float:
    x = x.flatten().float(); x = x - x.mean()
    return (x.pow(4).mean() / x.pow(2).mean().pow(2).clamp_min(1e-12)).item()

def fp8_sensitivity_db(w: torch.Tensor, group_size: int = 32) -> float:
    wq = dequantize_scaled_fp8(quantize_scaled_fp8(w, group_size))
    return 10*torch.log10((w.pow(2).sum()/(w-wq).pow(2).sum()).clamp_min(1e-12)).item()

def report(name: str, w: torch.Tensor, group_size: int = 32) -> dict:
    """One-line record per α/β/A/dt tensor for the mlambaformer SSM characterization."""
    return {"name": name, "per_channel_kurtosis": kurtosis(w.float().abs().amax(0)) if w.ndim>1 else None,
            "per_token_kurtosis": kurtosis(w), "fp8_snr_db": fp8_sensitivity_db(w, group_size)}
```

- [ ] **Step 4: Run to verify it passes** — `ALL SSM-SENSITIVITY TESTS PASSED`.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ssm_sensitivity.py scripts/calibration/test_ssm_sensitivity.py
git commit -m "feat(ml8): SSM gate quant-sensitivity instrument"
```

### Task 4: Wire role-tagged targets + FP8 tier into the calibration driver

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py` (the `_qwen_mlp_name_map` call sites + the per-linear loop + blob save)

- [ ] **Step 1: Read** `calibrate_ml8_paged.py` around the `_qwen_mlp_name_map` usage and the per-linear capture/quantize/save loop (lines ~95–260, ~660–710) to find where targets are enumerated, where input activations are hooked, and where `.pt` blobs are saved (`name.replace('.', '_')...pt`).

- [ ] **Step 2: Write the failing test** — a small integration test that builds a tiny stub model with attn/ssm/embed submodules and asserts the driver's target enumeration includes the ml8 + fp8 roles and excludes native ones.

```python
# add to a new scripts/calibration/test_calib_targets.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch.nn as nn
from calibrate_ml8_paged import enumerate_quant_targets   # new function this task adds

def test_enumerate_targets_covers_attn_ssm_embed():
    class L(nn.Module): pass
    # build a stub with the HF names role_targets expects; assert tiers
    targets = enumerate_quant_targets_for_names([
        "model.layers.0.self_attn.q_proj", "model.layers.0.linear_attn.alpha_proj",
        "model.embed_tokens", "model.layers.0.linear_attn.conv1d", "lm_head"])
    tiers = {t.role: t.tier_name for t in targets}
    assert tiers["attn_q"] == "ml8" and tiers["ssm_alpha"] == "fp8" and tiers["lm_head"] == "ml8"
    assert "native" not in tiers.values()   # conv1d excluded entirely
```

> The exact helper name (`enumerate_quant_targets` / `enumerate_quant_targets_for_names`) is defined in this task — keep it consistent with what Step 3 implements.

- [ ] **Step 3: Implement** — replace the FFN-only enumeration with a `role_targets.classify_role` sweep over `named_modules()`, building a list of `QuantTarget(module, gguf_name, role, tier)`; route `Tier.ML8` through the existing GPTQ/Lloyd-Max path (hook its input for the Hessian), `Tier.FP8` through `scaled_fp8.quantize_scaled_fp8` (no Hessian), skip `Tier.NATIVE`. Save FP8 blobs with a distinct marker key (`{"kind": "scaled_fp8", "e4m3":..., "scale":..., "group_size":...}`) so the converter can tell tiers apart. Emit `ssm_sensitivity.report(...)` for α/β/A/dt into `manifest.json`.

- [ ] **Step 4: Run** — `python3 test_calib_targets.py` (PASS), then a 1-layer dry-run of the driver on the local 4B to confirm it enumerates attn/ssm/embed/lm_head and writes both blob kinds without error.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py scripts/calibration/test_calib_targets.py
git commit -m "feat(ml8): drive full-model role-tagged calibration (ml8 + fp8 tiers)"
```

---

## Phase 2 — Format, converter, coverage (Python)

### Task 5: Register `ML8_FP8` in gguf-py

**Files:**
- Modify: `gguf-py/gguf/constants.py` (the `GGMLQuantizationType` enum + `GGML_QUANT_SIZES`)
- Test: `scripts/calibration/test_ml8_fp8_gguf_type.py`

- [ ] **Step 1: Read** `gguf-py/gguf/constants.py` for the existing `ML8_4` / `F8_E4M3` enum entries and the block-size table format.

- [ ] **Step 2: Write the failing test** — assert the new enum value exists and its block size matches the scaled-fp8 layout (group_size e4m3 bytes + 1 fp16 scale).

```python
# test_ml8_fp8_gguf_type.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "gguf-py"))
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
def test_ml8_fp8_registered():
    t = GGMLQuantizationType.ML8_FP8
    block, size = GGML_QUANT_SIZES[t]
    assert block == 32                      # group_size
    assert size == 32 * 1 + 2               # 32 e4m3 bytes + one fp16 scale
if __name__ == "__main__":
    test_ml8_fp8_registered(); print("ML8_FP8 GGUF TYPE OK")
```

- [ ] **Step 3: Implement** — add `ML8_FP8 = <next free id>` to `GGMLQuantizationType` (match the C enum id chosen in Task 9) and `GGMLQuantizationType.ML8_FP8: (32, 34)` to `GGML_QUANT_SIZES`.

- [ ] **Step 4: Run** — `python3 test_ml8_fp8_gguf_type.py` → `ML8_FP8 GGUF TYPE OK`.

- [ ] **Step 5: Commit**

```bash
git add gguf-py/gguf/constants.py scripts/calibration/test_ml8_fp8_gguf_type.py
git commit -m "feat(gguf): register ML8_FP8 scaled-fp8 quant type"
```

### Task 6: Converter writes new ml8 roles + FP8 tier

**Files:**
- Modify: `scripts/calibration/ml8_to_gguf.py` (`_build_blob_map`, the per-tensor write loop, a new `_write_scaled_fp8`)

- [ ] **Step 1: Read** `ml8_to_gguf.py` `_build_blob_map` (~264) and the write loop (~374–520) to see how FFN blobs map to GGUF tensors and how sidecars are written.

- [ ] **Step 2: Write the failing test** — a synthetic blob dir with one ml8 attn blob + one scaled-fp8 α blob; assert the writer emits an `ML8_4` `attn_q` tensor (+ centroids sidecar) and an `ML8_FP8` `ssm_alpha` tensor (+ its scale), via a `classify_blob_kind(blob)` helper this task adds.

- [ ] **Step 3: Implement** — extend blob discovery to the new ml8 role gguf-names; branch on the blob `kind`: ml8 blobs reuse the existing pack/sidecar path; `scaled_fp8` blobs write the e4m3 payload as `raw_dtype=GGMLQuantizationType.ML8_FP8` with the fp16 scale interleaved per the block layout from Task 5. Keep the per-tensor coverage accounting calls (Task 7).

- [ ] **Step 4: Run** — the synthetic-blob test (PASS).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_to_gguf.py scripts/calibration/test_ml8_to_gguf.py
git commit -m "feat(ml8): converter writes full-model ml8 roles + scaled-fp8 tier"
```

### Task 7: Coverage-metric refinement (credit the FP8 tier)

**Files:**
- Modify: `scripts/calibration/ml8_to_gguf.py` (`evaluate_coverage` + the byte accounting + the report)
- Modify: `scripts/calibration/test_ml8_to_gguf.py`

- [ ] **Step 1: Write the failing test** — extend `evaluate_coverage` semantics so an 8-bit FP8 tensor counts as *quantized* (not bf16), with the function returning a tier breakdown.

```python
# add to test_ml8_to_gguf.py
from ml8_to_gguf import evaluate_coverage
def test_coverage_credits_fp8_tier():
    # 88% ml8, 11% fp8, 1% bf16 leftover -> total quantized 99% (clears), ml8 share 88%
    cov, below, breakdown = evaluate_coverage(params_ml8=88, params_fp8=11,
                                              params_passthrough_weight=1, min_coverage=0.85)
    assert abs(cov - 0.99) < 1e-9 and below is False
    assert abs(breakdown["ml8"] - 0.88) < 1e-9 and abs(breakdown["fp8"] - 0.11) < 1e-9
def test_coverage_ffn_only_still_refuses():
    # 24% ml8, 0 fp8, 76% bf16 -> 24% quantized -> still flagged (the original regression)
    cov, below, _ = evaluate_coverage(params_ml8=24, params_fp8=0,
                                      params_passthrough_weight=76, min_coverage=0.85)
    assert abs(cov - 0.24) < 1e-9 and below is True
```

- [ ] **Step 2: Run to verify it fails** — current `evaluate_coverage` has a 2-arg signature → `TypeError`.

- [ ] **Step 3: Implement** — widen `evaluate_coverage(params_ml8, params_fp8, params_passthrough_weight, min_coverage)` returning `(coverage, below_threshold, breakdown)` where `coverage = (ml8+fp8)/total` and `breakdown = {"ml8":.., "fp8":.., "bf16":..}`. Update the converter's accounting to track `params_ml8` and `params_fp8` separately and print `"[coverage] 88.0% 4-bit ml8 + 11.0% 8-bit FP8 + 1.0% bf16 (total quantized 99.0%)"`. Write `ml8.weight_coverage` (total) + new `ml8.ml8_fraction` / `ml8.fp8_fraction` keys.

- [ ] **Step 4: Run** — `python3 test_ml8_to_gguf.py` (all pass, incl. the still-refuses case).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/ml8_to_gguf.py scripts/calibration/test_ml8_to_gguf.py
git commit -m "feat(ml8): coverage metric credits the 8-bit FP8 tier"
```

### Task 8: Native `.ml8` artifact for the new tiers

**Files:**
- Modify: the native `.ml8` writer (locate via `grep -rn "\.ml8" scripts/calibration/*.py`); if none exists yet, scope is the GGUF path only and this task records that finding and is closed.

- [ ] **Step 1: Investigate** whether a native `.ml8` writer exists. If yes, extend it to emit the new ml8 roles + FP8 tier symmetrically with the GGUF path (the two-formats rule). If no native writer exists in-tree, **stop**: note in the plan that native `.ml8` is out of current scope and the GGUF-wrapped artifact is the deliverable, then mark the task done.

- [ ] **Step 2: Commit (only if code changed)**

```bash
git add -A && git commit -m "feat(ml8): native .ml8 emits full-model tiers"
```

---

## Phase 3 — C type traits

### Task 9: `GGML_TYPE_ML8_FP8` traits + `to_float` in ggml.c

**Files:**
- Modify: `ggml/include/ggml.h` (enum) , `ggml/src/ggml-common.h` (`block_ml8_fp8`), `ggml/src/ggml.c` (traits + `dequantize_row_ml8_fp8`)
- Test: `tests/test-ml8-fp8-dequant.cpp` (or extend an existing ggml quant test)

- [ ] **Step 1: Read** the `GGML_TYPE_ML8_4` / `GGML_TYPE_F8_E4M3` enum entries (`ggml.h`), the `block_ml8_4` struct (`ggml-common.h`), and the `[GGML_TYPE_ML8_4]` traits block (`ggml.c:803`) for the exact pattern (blck_size, type_size, to_float, from_float_ref).

- [ ] **Step 2: Write the failing test** — quantize a known row with the Python `scaled_fp8`, write the bytes, and assert the C `dequantize_row_ml8_fp8` reproduces them to fp32 within e4m3 tolerance (cross-checks Python writer ↔ C reader byte layout).

- [ ] **Step 3: Implement** — add `GGML_TYPE_ML8_FP8` enum; `block_ml8_fp8 { ggml_fp16_t scale; uint8_t qs[32]; }` (matching Task 5's 34-byte block); traits with `blck_size=32`, `type_size=sizeof(block_ml8_fp8)`, `to_float = dequantize_row_ml8_fp8` (decode e4m3 via the same lattice as `ggml_cuda_ue4m3_to_fp32`, multiply by `scale`), `from_float_ref` stub (calibration writes blobs, not via ggml). Ensure the enum id matches Task 5's gguf-py id.

- [ ] **Step 4: Run** — build `ggml` host + run the dequant test (PASS, ≤ e4m3 round tolerance).

- [ ] **Step 5: Commit**

```bash
git add ggml/include/ggml.h ggml/src/ggml-common.h ggml/src/ggml.c tests/test-ml8-fp8-dequant.cpp
git commit -m "feat(ggml): ML8_FP8 type traits + to_float dequant"
```

---

## Phase 4 — CUDA kernels

### Task 10: FP8 `get_rows` case

**Files:**
- Modify: `ggml/src/ggml-cuda/getrows.cu` (the `ggml_cuda_get_rows_switch_src0_type` switch + a `get_rows_cuda_ml8_fp8` template)

- [ ] **Step 1: Read** `getrows.cu:170-225` (the float/quant case structure) and `common.cuh:830` (`ggml_cuda_ue4m3_to_fp32`).

- [ ] **Step 2: Write the failing test** — add an `ML8_FP8` case to `tests/test-backend-ops.cpp` GET_ROWS coverage (or a focused harness): gather known rows from an `ML8_FP8` tensor, compare to the `to_float`-then-gather reference within e4m3 tolerance.

- [ ] **Step 3: Implement** — a `get_rows_cuda_ml8_fp8` that, per gathered row, reads each block's `scale` + e4m3 `qs` and writes `ue4m3_to_fp32(qs[i]) * scale` to `dst`; add `case GGML_TYPE_ML8_FP8:` calling it. CPU mirror already covered by the Task 9 `to_float` (ggml's CPU get_rows uses traits).

- [ ] **Step 4: Run** — build HIP multi-arch + run the GET_ROWS test for `ML8_FP8` (PASS).

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/getrows.cu tests/test-backend-ops.cpp
git commit -m "feat(cuda): ML8_FP8 get_rows gather case"
```

### Task 11: No-LUT FP8-WMMA `mul_mat` for α/β

**Files:**
- Modify: `ggml/src/ggml-cuda/ml8.cu` (a no-LUT variant of `ggml_cuda_op_ml8_mul_mat`), `ggml/src/ggml-cuda/ggml-cuda.cu` (dispatch for `ML8_FP8` mul_mat)

- [ ] **Step 1: Read** `ml8.cu` `ggml_cuda_op_ml8_mul_mat` end-to-end (the centroid-LUT load + the FP8-WMMA accumulation, ~around the gemv/gemm bodies) to identify exactly where the 4-bit index → centroid lookup happens, so the no-LUT path can substitute "the e4m3 byte *is* the weight value × per-group scale."

- [ ] **Step 2: Write the failing test** — bit-equivalence: build an `ML8_FP8` weight + input, run the new mul_mat, compare to a reference `dequantize_scaled_fp8` (host) @ x within FP8-WMMA accumulation tolerance (reuse the Y_SNR/≥ ~40 dB style check; the ml8 path is deterministic to 4 decimals per the kernel-determinism fact).

- [ ] **Step 3: Implement** — a templated no-LUT mode (compile-time flag or a sibling entry) that skips the centroid lookup: load the e4m3 weight bytes directly into the FP8-WMMA fragment, scale by the per-group fp16 scale in the fp32 epilogue. Reuse the existing two-level fp32 accumulation. Wire `ggml-cuda.cu` so an `ML8_FP8` weight in `MUL_MAT` routes here (mirror the `ML8_4` op-swap at `llama-model-loader.cpp:910` for the type check).

- [ ] **Step 4: Run** — the bit-equivalence test (PASS) + `rocprofv3 --hip-trace` sanity that the no-LUT kernel launches (gfx1201).

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/ml8.cu ggml/src/ggml-cuda/ggml-cuda.cu tests/test-backend-ops.cpp
git commit -m "feat(cuda): no-LUT FP8-WMMA mul_mat for scaled-fp8 weights"
```

---

## Phase 5 — C++ graph wiring

### Task 12: Sidecar registry + `build_ml8_or_mul_mat` helper

**Files:**
- Create: `src/llama-ml8-registry.h`, `src/llama-ml8-registry.cpp`
- Modify: `src/CMakeLists.txt` (add the new source)
- Test: a focused unit (`tests/test-ml8-registry.cpp`) for the fallback contract

- [ ] **Step 1: Write the failing test** — a registry with no sidecars for a weight returns "plain mul_mat selected" (a sentinel/flag), and a registry *with* sidecars returns "ml8 path selected." This proves the zero-impact fallback.

- [ ] **Step 2: Implement** — `ml8_registry` mapping `const ggml_tensor* weight → ml8_sidecars{centroids, rotation_h_a, rotation_meta, awq_scale}` (all nullable). `build_ml8_or_mul_mat(ctx, reg, weight, x)`: if `weight->type` is `ML8_4` and the registry has centroids → `ggml_ml8_mul_mat(...)` (+ rotation/awq apply if present); elif `weight->type` is `ML8_FP8` → the no-LUT FP8 mul_mat; else → `ggml_mul_mat(ctx, weight, x)`. Pure function over the registry; no global state.

- [ ] **Step 3: Run** — build + run `test-ml8-registry` (PASS).

- [ ] **Step 4: Commit**

```bash
git add src/llama-ml8-registry.h src/llama-ml8-registry.cpp src/CMakeLists.txt tests/test-ml8-registry.cpp
git commit -m "feat(llama): ml8 sidecar registry + ml8-aware mul_mat helper"
```

### Task 13: Load new sidecars + route qwen35 GEMM call sites through the helper

**Files:**
- Modify: `src/llama-model-loader.cpp` / `src/llama-model.cpp` (load the new ml8 roles' sidecars into the registry), `src/llama-graph.cpp` (the qwen35 dense build: attn q/k/v/output, qkv/gate, ssm_out, eh_proj, lm_head)

- [ ] **Step 1: Read** the qwen35 dense graph build in `llama-graph.cpp` (both the full-attention and gated-deltanet branches) + the existing FFN sidecar load to mirror the pattern for the new roles.

- [ ] **Step 2: Write the failing test** — load the existing FFN-only 9B ml8 GGUF and assert a non-ml8 reference model's graph is byte-identical with the helper in place (the regression guard), plus assert the registry is populated for the new roles when their sidecars are present.

- [ ] **Step 3: Implement** — populate `ml8_registry` from the loaded sidecar tensors for every ml8 role; replace the `ggml_mul_mat`/`ggml_mul_mat_aux` calls for attn q/k/v/output, qkv/gate, ssm_out, eh_proj, and lm_head with `build_ml8_or_mul_mat(...)`. Leave FFN/MoE explicit paths untouched (per spec out-of-scope).

- [ ] **Step 4: Run** — the regression test (non-ml8 model unchanged) + a smoke `llama-cli --no-mmap` load of the FFN-only 9B (still runs, helper falls through for un-quantized attn).

- [ ] **Step 5: Commit**

```bash
git add src/llama-model-loader.cpp src/llama-model.cpp src/llama-graph.cpp
git commit -m "feat(llama): route qwen35 attn/ssm/lm_head GEMMs through ml8 helper"
```

### Task 14: Wire scaled-FP8 embed + α/β into the graph

**Files:**
- Modify: `src/llama-model.cpp` (load `token_embd` as `ML8_FP8`; load α/β as `ML8_FP8`), `src/llama-graph.cpp` (α/β consumed by `build_ml8_or_mul_mat`; embed via `get_rows` which now supports `ML8_FP8`)

- [ ] **Step 1: Write the failing test** — load a small synthetic GGUF with an `ML8_FP8` `token_embd` + α/β and assert (a) embedding lookup produces correct rows (vs `to_float` reference) and (b) α/β matmuls route to the no-LUT FP8 path.

- [ ] **Step 2: Implement** — allow `token_embd`/α/β to load as `ML8_FP8`; the embed `get_rows` call needs no change (Task 10 added the case); α/β call sites use the helper (Task 11/12 cover the dispatch).

- [ ] **Step 3: Run** — the synthetic test (PASS).

- [ ] **Step 4: Commit**

```bash
git add src/llama-model.cpp src/llama-graph.cpp
git commit -m "feat(llama): scaled-fp8 embed + ssm alpha/beta in the graph"
```

---

## Phase 6 — Validation (dense 9B first, then 35B MoE)

### Task 15: Python↔C++ bit-equivalence harness

**Files:**
- Create: `scripts/calibration/check_bit_equivalence.py`

- [ ] **Step 1: Implement** — run the Python `Ml8Linear` reference forward on a fixed calibration artifact and the C++ graph path (`llama-perplexity --no-mmap` on the same tiny eval + seed); assert PPL matches to ≥4 decimals (>0.005 drift ⇒ wiring bug, per the kernel-determinism fact). Start on the **4B** (fast).

- [ ] **Step 2: Run** — on the 4B artifact; record the matched PPL.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibration/check_bit_equivalence.py
git commit -m "test(ml8): python<->C++ bit-equivalence harness"
```

### Task 16: Dense 9B end-to-end — calibrate, convert, gate

**Files:** none new (driving the pipeline)

- [ ] **Step 1: Calibrate** the 9B full-model (resident, local) → blobs for all ml8 roles + FP8 tier. Honor the GPU rules (calculate VRAM first; 95% ceiling; stop-after-restart).
- [ ] **Step 2: Convert** → `/home/kmbandy/models/Qwen3.5-9B-ml8-fullcov.gguf`; confirm `[coverage]` prints ≥85% ml8 + ~11% FP8 and the guardrail **passes** (no `--allow-partial`).
- [ ] **Step 3: Gate** — `llama-perplexity --no-mmap` vs the bf16 9B: **Δ_PPL ≤ +0.08–0.10**; record size **< UD-Q4_K_XL 9B**. If the gate fails, the per-role `group_size` knob (esp. lm_head) is the lever — adjust and re-run that role only.
- [ ] **Step 4: Record** results (PPL, size, coverage split) in the spec's validation section + a KG fact.

### Task 17: Long-context probe

**Files:**
- Create: `scripts/calibration/longctx_probe.py`

- [ ] **Step 1: Implement** — a long-context eval (needle-in-haystack and/or long-ctx PPL) comparing the full-coverage 9B vs bf16, specifically to catch SSM-gate compounding that short-ctx PPL averages out.
- [ ] **Step 2: Run** — on the 9B; assert no disproportionate long-ctx degradation vs the short-ctx Δ. Cross-reference the α/β sensitivity instrument output.
- [ ] **Step 3: Commit**

```bash
git add scripts/calibration/longctx_probe.py
git commit -m "test(ml8): long-context probe for SSM-gate compounding"
```

### Task 18: 35B MoE non-expert coverage + both-axes gate

**Files:** none new (driving the pipeline on MoE)

- [ ] **Step 1:** Run the same full-model calibration on the 35B-A3B — the experts already calibrate as today; this adds attn/ssm/lm_head/embed coverage. Honor the 35B paged-test safety rule (no repeated OOM iterations; smaller-model repro for any bug).
- [ ] **Step 2:** Convert; confirm coverage now reports the non-expert tiers (no longer bf16).
- [ ] **Step 3:** Gate — **smaller AND lower PPL than UD-Q4_K_XL** (MAD-256 both-axes). Record vs the 5.7507 reference.
- [ ] **Step 4:** Record results + update MAD-256 / KG.

---

## Plan-level self-review

- **Spec coverage:** S1 calibration → Tasks 1–4; S2 format/converter/coverage → Tasks 5–8; S3 C++ inference (helper/registry, get_rows, no-LUT mul_mat) → Tasks 9–14; S4 validation (bit-equiv, near-lossless, size, coverage, long-ctx, then MoE) → Tasks 15–18; S5 testing → embedded per task. All spec sections mapped.
- **Type/name consistency:** `Tier.{ML8,FP8,NATIVE}`, `classify_role`, `quantize_scaled_fp8`/`dequantize_scaled_fp8`, `GGML_TYPE_ML8_FP8`/`block_ml8_fp8` (34-byte block, group 32), `evaluate_coverage(params_ml8, params_fp8, params_passthrough_weight, min_coverage)`, `build_ml8_or_mul_mat`, `ml8_registry` — used consistently across tasks.
- **Known derive-from-source points (not placeholders):** exact HF submodule names for the gated-deltanet projections (Task 1/4), the ml8.cu LUT-lookup location for the no-LUT variant (Task 11), and whether a native `.ml8` writer exists (Task 8). Each is an explicit read/investigate step with a concrete fallback.
- **Sequencing:** type (5,9) before consumers (10,11); kernels before graph wiring (12–14); dense 9B (15–17) before 35B MoE (18).
```
