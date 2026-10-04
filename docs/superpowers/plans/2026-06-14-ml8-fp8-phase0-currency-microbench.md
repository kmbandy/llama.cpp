# ml8 FP8 Phase 0 — Currency + Microbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the Triton/aiter fp8 substrate current and *measure* it on the R9700 (gfx1201) — bump Triton past #10458, make the rebuild reproducible, generate the missing gfx1201 a8w8 configs, and produce a 4-cell TFLOPS + fp8-correctness table — so the scope of Phases 1–3 is re-confirmed on data, not assumption.

**Architecture:** Phase 0 is toolchain + measurement, so tasks split into (a) **code deliverables** built TDD-first — a 4-cell a8w8 fp8 microbench and a bit-level e4m3 correctness harness — and (b) **verify-gated ops** — a reproducible pinned-SHA Triton build, the actual bump, and aiter config generation. The microbench and correctness harness are written and baselined on the *current* Triton FIRST, so the same tools measure before/after the bump (the tool is the constant; the toolchain is the variable).

**Tech Stack:** Python 3 / PyTorch (ROCm), Triton (editable install from `~/GitHub/triton`), aiter (`~/GitHub/aiter`), pytest, bash. Target arch gfx1201 (R9700 / RDNA4). Spec: `docs/superpowers/specs/2026-06-14-ml8-fp8-substrate-unification-design.md` (§5). Epic MAD-293, story **MAD-294**.

**Environment facts (verified 2026-06-14):**
- Triton clone `~/GitHub/triton` HEAD `4768da5e` (`3.7.0+git4768da5e`), **editable install** at `~/GitHub/triton/python`.
- Pin target `007ef1530` (origin/main, 2026-06-14) **contains** `bb5acbe59` (#10458, OCP-e4m3 fix) — verified ancestor.
- AOT driverless patch: `ggml/src/ggml-cuda/aiter-integration/cmake/patch_triton_driverless_aot.py` (idempotent; **exits non-zero if a bump removes its anchors** — the guardrail).
- aiter Triton a8w8 configs: `~/GitHub/aiter/aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE-N=<N>-K=<K>.json`; lookup via `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale._get_config(M,N,K)`. Of our 16 (N,K) from {2560,4096,8192,9216}, **only (8192,8192) exists**; the rest fall to the generic `gfx1201-GEMM-A8W8_BLOCKSCALE.json`.
- Reusable patterns: `scripts/calibration/tune_gemm_ml8.py` (LUT-path tuner — median-ms + `_theoretical_us` TFLOPS pattern), `scripts/calibration/bench_ml8_wgrad.py`.
- e4m3 oracle: `scripts/calibration/ml8_e4m3_sim.py::fp32_to_e4m3_bits` / `e4m3_roundtrip` (≡ `ml8.cu:440`), tests in `scripts/calibration/tests/test_ml8_e4m3_sim.py`.
- 4B Qwen3.5 linear shapes (N=out, K=in): gate/up `(9216, 2560)`, down `(2560, 9216)`, o_proj `(2560, 2560)`. Trainer micro-step M≈2048 (1×2048); prefill M=512; decode M=16.

**[GPU] tasks require the R9700.** Pure-Python TDD tasks run on CPU.

---

## File Structure

- **Create** `scripts/calibration/microbench_a8w8_fp8.py` — 4-cell fp8 GEMM microbench (aiter-Triton-a8w8 vs `torch._scaled_mm`) at our real shapes; emits JSON. One responsibility: measure GEMM throughput.
- **Create** `scripts/calibration/test_microbench_a8w8_fp8.py` — CPU TDD for the shape-enum + TFLOPS math.
- **Create** `scripts/calibration/verify_e4m3_triton.py` — bit-level fp32→e4m3 code comparison: Triton cast vs the `ml8_e4m3_sim` oracle; emits JSON of mismatched code-points.
- **Create** `scripts/calibration/test_verify_e4m3_triton.py` — CPU TDD for the code-comparison logic (oracle-vs-oracle and stubbed-cast).
- **Create** `ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh` — reproducible pinned-SHA Triton build (checkout + driverless patch + editable rebuild + post-build gfx1201 smoke).
- **Create** `ggml/src/ggml-cuda/aiter-integration/tools/TRITON_PIN.txt` — the pinned SHA + provenance (one line: `007ef1530  # contains #10458 bb5acbe59`).
- **Create** `ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py` — CPU TDD for the patch-anchor verification helper.
- **Create** `scripts/calibration/gen_gfx1201_a8w8_configs.py` — driver that runs aiter's a8w8 autotuner for our (N,K) grid and vendors the JSONs.
- **Create** `scripts/calibration/test_gfx1201_a8w8_coverage.py` — asserts `_get_config(M,N,K)` returns a per-shape (not generic) config for each target (N,K).
- **Create** `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md` — the 4-cell TFLOPS + correctness table and the Phase-0 exit-gate scope re-confirmation.
- **Modify** `~/GitHub/aiter/aiter/ops/triton/configs/gemm/` (add generated `gfx1201-...-N=..K=..json`) — data only.

---

## Task 1: a8w8 fp8 microbench harness (CPU-testable core)

**Files:**
- Create: `scripts/calibration/microbench_a8w8_fp8.py`
- Test: `scripts/calibration/test_microbench_a8w8_fp8.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/calibration/test_microbench_a8w8_fp8.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microbench_a8w8_fp8 import gemm_tflops, default_shapes


def test_gemm_tflops_matches_2mnk_over_seconds():
    # 2*M*N*K flops; at M=N=K=1024 and 1 ms -> 2*1024^3 / 1e-3 / 1e12 TFLOPS
    tflops = gemm_tflops(M=1024, N=1024, K=1024, seconds=1e-3)
    assert abs(tflops - (2 * 1024**3 / 1e-3 / 1e12)) < 1e-6


def test_default_shapes_cover_4b_mlp_and_oproj():
    shapes = {(n, k) for (_name, n, k) in default_shapes()}
    assert (9216, 2560) in shapes   # gate/up
    assert (2560, 9216) in shapes   # down
    assert (2560, 2560) in shapes   # o_proj
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_microbench_a8w8_fp8.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'microbench_a8w8_fp8'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/calibration/microbench_a8w8_fp8.py
"""4-cell fp8 GEMM microbench (gfx1201): aiter Triton a8w8-blockscale vs
torch._scaled_mm, at the real 4B linear shapes. Emits JSON. The toolchain is the
variable across runs (pre-bump / post-bump / +configs); this script is constant."""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch


def gemm_tflops(M: int, N: int, K: int, seconds: float) -> float:
    return (2.0 * M * N * K) / seconds / 1e12


def default_shapes():
    # (name, N=out, K=in) — Qwen3.5-4B (hidden=2560, intermediate=9216)
    return [("gate", 9216, 2560), ("up", 9216, 2560),
            ("down", 2560, 9216), ("o_proj", 2560, 2560)]


def _median_seconds(fn, *, warmup=5, iters=30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_microbench_a8w8_fp8.py -v`
Expected: PASS (2 passed). No GPU needed for these two tests.

- [ ] **Step 5: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add scripts/calibration/microbench_a8w8_fp8.py scripts/calibration/test_microbench_a8w8_fp8.py
git commit -m "MAD-294: a8w8 fp8 microbench core (tflops + shapes, TDD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Wire the two GEMM cells + JSON output into the microbench [GPU]

**Files:**
- Modify: `scripts/calibration/microbench_a8w8_fp8.py`

- [ ] **Step 1: Add the aiter-Triton and `_scaled_mm` cells + `main()`**

Append to `microbench_a8w8_fp8.py`. The aiter cell calls the same kernel the trainer uses; the torch cell is the backward's current GEMM (the comparison baseline).

```python
def _run_aiter_a8w8(M, N, K, dev):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    xs = torch.ones((M, (K + 127) // 128), device=dev, dtype=torch.float32)
    ws = torch.ones((N, (K + 127) // 128), device=dev, dtype=torch.float32)
    return lambda: gemm_a8w8_blockscale(x, w, xs, ws, dtype=torch.bfloat16)


def _run_scaled_mm(M, N, K, dev):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    sa = torch.ones((M, 1), device=dev, dtype=torch.float32)
    sb = torch.ones((1, N), device=dev, dtype=torch.float32)
    return lambda: torch._scaled_mm(x, w, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. pre-bump / post-bump / post-bump+configs")
    ap.add_argument("--m-tiers", type=int, nargs="+", default=[16, 512, 2048])
    ap.add_argument("--out", type=Path, default=Path("/tmp/phase0_microbench.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    for name, N, K in default_shapes():
        for M in args.m_tiers:
            for cell, mk in (("aiter_a8w8", _run_aiter_a8w8), ("scaled_mm", _run_scaled_mm)):
                try:
                    sec = _median_seconds(mk(M, N, K, dev))
                    rows.append(dict(shape=name, M=M, N=N, K=K, cell=cell,
                                     ms=sec * 1e3, tflops=gemm_tflops(M, N, K, sec)))
                except Exception as e:  # noqa: BLE001 — record, don't abort the sweep
                    rows.append(dict(shape=name, M=M, N=N, K=K, cell=cell, error=str(e)[:200]))
    out = dict(label=args.label, triton_version=__import__("triton").__version__, rows=rows)
    args.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        if "tflops" in r:
            print(f"{r['shape']:7s} M={r['M']:5d} {r['cell']:11s} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TFLOPS")
        else:
            print(f"{r['shape']:7s} M={r['M']:5d} {r['cell']:11s} ERROR {r['error']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on the R9700**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python microbench_a8w8_fp8.py --label smoke --m-tiers 512 --out /tmp/phase0_smoke.json`
Expected: prints one line per (shape, cell) with ms + TFLOPS (or a recorded ERROR for an unsupported cell), and writes `/tmp/phase0_smoke.json`. Non-zero TFLOPS for at least the `scaled_mm` cell.

- [ ] **Step 3: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add scripts/calibration/microbench_a8w8_fp8.py
git commit -m "MAD-294: microbench GEMM cells (aiter-a8w8 + _scaled_mm) + JSON out

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Bit-level e4m3 correctness harness (CPU-testable core)

**Files:**
- Create: `scripts/calibration/verify_e4m3_triton.py`
- Test: `scripts/calibration/test_verify_e4m3_triton.py`

The #10458 fix changes Triton's fp32→e4m3 *codes* on non-fnuz (OCP) archs. This harness compares a Triton cast against the `ml8_e4m3_sim` oracle over a code-point sweep and reports mismatches. Core logic (sweep generation + code comparison) is CPU-testable; the Triton cast is injected.

- [ ] **Step 1: Write the failing test**

```python
# scripts/calibration/test_verify_e4m3_triton.py
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_e4m3_triton import sweep_inputs, compare_codes
from ml8_e4m3_sim import e4m3_roundtrip


def test_oracle_against_itself_zero_mismatches():
    x = sweep_inputs()
    # A "cast" that is exactly the oracle must produce zero mismatches.
    mism = compare_codes(x, cast_fn=e4m3_roundtrip)
    assert mism["n_mismatch"] == 0


def test_sweep_covers_subnormal_and_saturation():
    x = sweep_inputs()
    assert (x.abs() < 2.0 ** -6).any()    # subnormal e4m3 region
    assert (x.abs() > 448.0).any()         # saturation region
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_verify_e4m3_triton.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_e4m3_triton'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/calibration/verify_e4m3_triton.py
"""Compare a fp32->e4m3 cast (Triton, or any callable) against the ml8_e4m3_sim
oracle, bit-exactly via decoded values. Used to verify Triton's OCP e4m3 codes
before/after #10458 on gfx1201."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from ml8_e4m3_sim import e4m3_roundtrip


def sweep_inputs() -> torch.Tensor:
    # Dense small-magnitude sweep (subnormal + normal) plus saturation probes.
    lin = torch.linspace(-512.0, 512.0, steps=200001)
    sub = torch.linspace(-(2.0 ** -6), 2.0 ** -6, steps=50001)
    sat = torch.tensor([-1e4, -448.0, -447.9, 447.9, 448.0, 1e4])
    return torch.cat([lin, sub, sat]).contiguous()


def compare_codes(x: torch.Tensor, cast_fn) -> dict:
    ref = e4m3_roundtrip(x.float())          # oracle dequantized values
    got = cast_fn(x.float()).float()
    # Compare decoded values; NaN slots compare equal-as-NaN.
    both_nan = torch.isnan(ref) & torch.isnan(got)
    mism = (~both_nan) & (ref != got)
    idx = torch.nonzero(mism).flatten()[:20].tolist()
    return dict(n_total=int(x.numel()), n_mismatch=int(mism.sum()),
                sample_mismatch_inputs=[float(x[i]) for i in idx])


def _triton_cast(x: torch.Tensor) -> torch.Tensor:
    # Reference Triton OCP e4m3 cast for the on-device check. The exact tl dtype
    # name (float8e4nv = OCP e4m3) is confirmed against the installed Triton.
    import triton, triton.language as tl
    x = x.to("cuda")
    out = torch.empty_like(x)

    @triton.jit
    def _k(xp, op, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        v = tl.load(xp + off, mask=m)
        tl.store(op + off, v.to(tl.float8e4nv).to(tl.float32), mask=m)

    n = x.numel(); BLOCK = 1024
    _k[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/phase0_e4m3.json"))
    args = ap.parse_args()
    x = sweep_inputs()
    res = compare_codes(x, cast_fn=_triton_cast)
    res.update(label=args.label, triton_version=__import__("triton").__version__)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_verify_e4m3_triton.py -v`
Expected: PASS (2 passed). CPU only (the `_triton_cast` path is not exercised by these tests).

- [ ] **Step 5: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add scripts/calibration/verify_e4m3_triton.py scripts/calibration/test_verify_e4m3_triton.py
git commit -m "MAD-294: bit-level e4m3 correctness harness vs ml8_e4m3_sim oracle (TDD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Capture PRE-bump baseline (current Triton 4768da5e) [GPU]

**Files:** none created — produces baseline JSON snapshots committed under `docs/superpowers/results/`.

- [ ] **Step 1: Confirm the current Triton SHA**

Run: `python -c "import triton; print(triton.__version__)"`
Expected: `3.7.0+git4768da5e` (the pre-bump baseline).

- [ ] **Step 2: Run the microbench (baseline)**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python microbench_a8w8_fp8.py --label pre-bump-4768da5 --out /tmp/phase0_microbench_prebump.json`
Expected: full table printed; JSON written. Record whether `aiter_a8w8` cells error (the generic-fallback config may underperform but should run).

- [ ] **Step 3: Run the e4m3 verify (baseline)**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python verify_e4m3_triton.py --label pre-bump-4768da5 --out /tmp/phase0_e4m3_prebump.json`
Expected: prints `n_mismatch`. **Record the number** — this is the pre-#10458 datapoint (nonzero would confirm the latent fnuz/OCP bug; zero means we were not hitting it).

- [ ] **Step 4: Persist the baselines**

```bash
cd /home/kmbandy/GitHub/llama.cpp
mkdir -p docs/superpowers/results
cp /tmp/phase0_microbench_prebump.json /tmp/phase0_e4m3_prebump.json docs/superpowers/results/
git add docs/superpowers/results/phase0_microbench_prebump.json docs/superpowers/results/phase0_e4m3_prebump.json
git commit -m "MAD-294: Phase 0 pre-bump baseline (microbench + e4m3 codes @ 4768da5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Patch-anchor verification helper (CPU-testable core of the build script)

**Files:**
- Create: `ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py`
- Create: `ggml/src/ggml-cuda/aiter-integration/tools/TRITON_PIN.txt`

The reproducible build's load-bearing guarantee is that the driverless AOT patch still applies after the bump. `patch_triton_driverless_aot.py` already exits non-zero if its anchors are gone; we add a thin pre-flight that asserts this *before* a rebuild, with a clear message, so a bad bump fails fast and legibly.

- [ ] **Step 1: Write `TRITON_PIN.txt`**

```
007ef1530aa1c9d1a78d206417fb7721fbd53211  # pinned 2026-06-14; contains #10458 bb5acbe59 (OCP e4m3 fix)
```

- [ ] **Step 2: Write the failing test**

```python
# ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_pin_file_is_a_40hex_sha():
    sha = (HERE / "TRITON_PIN.txt").read_text().split()[0]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_preflight_passes_on_current_driverless_patch_anchors():
    # The driverless patch script is the source of truth for anchors; running it
    # in --check mode against the installed Triton must succeed (anchors present
    # or already patched). Exit 0 = ok; non-zero = a bump broke the patch.
    import triton
    triton_root = Path(triton.__file__).resolve().parent.parent  # .../python
    r = subprocess.run([sys.executable, str(HERE.parent / "cmake" / "patch_triton_driverless_aot.py"),
                        "--check", str(triton_root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp && python -m pytest ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py -v`
Expected: FAIL — `test_preflight...` errors because `patch_triton_driverless_aot.py` has no `--check` mode yet (or the path/anchor check is absent).

- [ ] **Step 4: Add a `--check` (dry-run) mode to `patch_triton_driverless_aot.py`**

Read `ggml/src/ggml-cuda/aiter-integration/cmake/patch_triton_driverless_aot.py`. Add an argument parse so `--check <triton_root>` runs the existing anchor detection **without writing** — returning 0 if anchors are present or the file is already patched, non-zero otherwise. Keep the default (no `--check`) behavior byte-identical.

```python
# in patch_triton_driverless_aot.py main(), before applying edits:
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="verify anchors only; do not write")
ap.add_argument("triton_root", nargs="?", default="/opt/triton")
ns = ap.parse_args()
# ... locate compile.py under ns.triton_root, detect anchors / patched-form ...
if ns.check:
    sys.exit(0 if (anchors_present or already_patched) else 2)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp && python -m pytest ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py -v`
Expected: PASS (2 passed) — confirms the patch anchors are present on the *current* Triton (pre-bump), so the helper is correct before we use it as a bump gate.

- [ ] **Step 6: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/tools/TRITON_PIN.txt \
        ggml/src/ggml-cuda/aiter-integration/tools/test_build_triton_pinned.py \
        ggml/src/ggml-cuda/aiter-integration/cmake/patch_triton_driverless_aot.py
git commit -m "MAD-294: TRITON_PIN + driverless-patch --check preflight (TDD)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Reproducible pinned Triton build script

**Files:**
- Create: `ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh`

The "fragility is a bug" deliverable: one idempotent script that takes a clean Triton clone to a working, AOT-patched, gfx1201-capable editable install — no manual steps.

- [ ] **Step 1: Write the build script**

```bash
#!/usr/bin/env bash
# Reproducible pinned Triton build for the R9700 (gfx1201) dev box.
# Idempotent: re-running is safe. Fails fast (set -e) on any step.
set -euo pipefail

TRITON_DIR="${TRITON_DIR:-$HOME/GitHub/triton}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$(awk 'NR==1{print $1}' "$HERE/TRITON_PIN.txt")"
PATCH="$HERE/../cmake/patch_triton_driverless_aot.py"

echo "== [1/5] checkout $PIN in $TRITON_DIR =="
git -C "$TRITON_DIR" fetch --quiet origin
git -C "$TRITON_DIR" checkout --quiet "$PIN"

echo "== [2/5] pre-flight: driverless AOT patch anchors present? =="
python3 "$PATCH" --check "$TRITON_DIR/python"

echo "== [3/5] apply driverless AOT patch (idempotent) =="
python3 "$PATCH" "$TRITON_DIR/python"

echo "== [4/5] rebuild editable install =="
pip install -e "$TRITON_DIR/python" --no-build-isolation

echo "== [5/5] post-build smoke: import + compile a gfx1201 kernel =="
python3 - <<'PY'
import triton, triton.language as tl, torch
print("triton", triton.__version__)
@triton.jit
def _k(xp, op, n, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK); m = off < n
    tl.store(op+off, tl.load(xp+off, mask=m).to(tl.float8e4nv).to(tl.float32), mask=m)
x = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
_k[(triton.cdiv(x.numel(),1024),)](x, o, x.numel(), BLOCK=1024)
torch.cuda.synchronize(); print("gfx1201 fp8 kernel OK")
PY
echo "== DONE: pinned Triton $PIN built, patched, smoke-passed =="
```

- [ ] **Step 2: Make it executable + shellcheck**

Run: `cd /home/kmbandy/GitHub/llama.cpp && chmod +x ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh && bash -n ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh`
Expected: no syntax errors (exit 0). (Do NOT run the full build yet — that's Task 7.)

- [ ] **Step 3: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh
git commit -m "MAD-294: reproducible pinned-SHA Triton build script (gfx1201)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Execute the bump + post-bump correctness gate [GPU/build]

**Files:** none — runs the build script and the correctness harness; records results.

- [ ] **Step 1: Run the reproducible build**

Run: `bash /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh`
Expected: all 5 steps pass; final line `DONE: pinned Triton 007ef1530 built, patched, smoke-passed`. If step [2/5] (anchor pre-flight) fails, STOP — the bump broke the AOT patch; the patch must be re-anchored before proceeding (escalate; do not force-build).

- [ ] **Step 2: Confirm the new SHA is live**

Run: `python -c "import triton; print(triton.__version__)"`
Expected: `3.7.0+git007ef1530` (or the 3.8.0-dev string for that SHA) — i.e. no longer `4768da5e`.

- [ ] **Step 3: Re-run the e4m3 verify (post-bump) and gate**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python verify_e4m3_triton.py --label post-bump-007ef1530 --out /tmp/phase0_e4m3_postbump.json`
Expected: `n_mismatch == 0` against the OCP-e4m3 oracle. **Gate:** post-bump mismatches MUST be 0 (the oracle is OCP e4m3, which is what #10458 makes Triton emit). If pre-bump was nonzero and post-bump is 0, that is the #10458 fix landing. If post-bump is nonzero, STOP and investigate (the cast dtype or the oracle, not a "move on").

- [ ] **Step 4: turbo4 PPL/NIAH parity gate**

Run the existing turbo4_fp8 validation (the same gate used for the rotation work) on the post-bump Triton.
Run: `cd /home/kmbandy/GitHub/llama.cpp && ls scripts/ | grep -iE "turbo4|ppl|niah"` to locate the existing validation entrypoint, then run it.
Expected: PPL/NIAH within the established tolerance of the pre-bump numbers. **Gate:** parity holds, else STOP — a correctness regression from the bump blocks Phase 0.

- [ ] **Step 5: Persist post-bump correctness**

```bash
cd /home/kmbandy/GitHub/llama.cpp
cp /tmp/phase0_e4m3_postbump.json docs/superpowers/results/
git add docs/superpowers/results/phase0_e4m3_postbump.json
git commit -m "MAD-294: post-bump e4m3 correctness (007ef1530, OCP exact-match) + turbo4 gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Generate the missing gfx1201 a8w8 configs (D) [GPU]

**Files:**
- Create: `scripts/calibration/gen_gfx1201_a8w8_configs.py`
- Create: `scripts/calibration/test_gfx1201_a8w8_coverage.py`
- Modify (data): `~/GitHub/aiter/aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE-N=*-K=*.json`

- [ ] **Step 1: Write the failing coverage test**

```python
# scripts/calibration/test_gfx1201_a8w8_coverage.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_gfx1201_a8w8_configs import target_shapes, config_is_per_shape


def test_target_grid_is_16_combos_from_the_recon_set():
    shapes = target_shapes()
    dims = {2560, 4096, 8192, 9216}
    assert shapes == [(n, k) for n in sorted(dims) for k in sorted(dims)]
    assert len(shapes) == 16


def test_8192_8192_already_per_shape_others_initially_not():
    # (8192,8192) ships tuned; this guards that our per-shape detector is real.
    assert config_is_per_shape(8192, 8192) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_gfx1201_a8w8_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gen_gfx1201_a8w8_configs'`.

- [ ] **Step 3: Write the generator + per-shape detector**

```python
# scripts/calibration/gen_gfx1201_a8w8_configs.py
"""Generate gfx1201 a8w8-blockscale tuned configs for our (N,K) grid so
aiter's _get_config hits per-shape instead of the generic fallback. Data-only."""
from __future__ import annotations
import argparse
from pathlib import Path

AITER_CFG = Path.home() / "GitHub/aiter/aiter/ops/triton/configs/gemm"


def target_shapes():
    dims = sorted({2560, 4096, 8192, 9216})
    return [(n, k) for n in dims for k in dims]


def config_is_per_shape(N: int, K: int) -> bool:
    return (AITER_CFG / f"gfx1201-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json").exists()


def tune_one(N: int, K: int, m_tiers) -> None:
    # Drive aiter's a8w8 blockscale autotuner for (N,K) across M tiers and write
    # the per-shape JSON into AITER_CFG. Uses aiter's tuning utility:
    #   aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm_blockscale.py
    # which profiles the config_list from get_input_shape_and_config_list and
    # persists the winning config keyed by (gfx1201, N, K).
    from aiter.ops.triton.utils._triton.tunning import ut_a8w8_gemm_blockscale as ut  # noqa
    ut.tune_and_write(N=N, K=K, m_tiers=list(m_tiers), out_dir=str(AITER_CFG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-tiers", type=int, nargs="+", default=[16, 512, 2048])
    ap.add_argument("--only-missing", action="store_true", default=True)
    args = ap.parse_args()
    for N, K in target_shapes():
        if args.only_missing and config_is_per_shape(N, K):
            print(f"skip N={N} K={K} (already per-shape)"); continue
        print(f"tune N={N} K={K} ...")
        tune_one(N, K, args.m_tiers)
    print("done")


if __name__ == "__main__":
    main()
```

> **Implementer note:** `ut_a8w8_gemm_blockscale` exposes a profiling flow, not a one-call `tune_and_write`. Step 3's `tune_one` is the intended interface — wire it to that module's actual `run_profile` / `get_input_shape_and_config_list` entrypoints (read the file), persisting the winning config to the `gfx1201-...-N=..K=..json` name `_get_config` expects. If the aiter util can't be driven headlessly, fall back to aiter's `csrc/.../gemm_a8w8_blockscale_tune.py` tune path documented in `op_tests/test_gemm_a8w8_blockscale.py`. Keep `target_shapes`/`config_is_per_shape` exactly as tested.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python -m pytest test_gfx1201_a8w8_coverage.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Generate configs for the 4B-critical shapes first [GPU]**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python gen_gfx1201_a8w8_configs.py --m-tiers 16 512 2048`
Expected: tunes the missing combos (incl. the 4B-active (9216,2560), (2560,9216), (2560,2560), (9216,9216)), writes per-shape JSONs into the aiter config dir; skips (8192,8192).

- [ ] **Step 6: Coverage gate — `_get_config` now hits per-shape**

```python
# run as a one-off check (record output in the results doc):
python - <<'PY'
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import _get_config
for N,K in [(9216,2560),(2560,9216),(2560,2560),(9216,9216)]:
    cfg,_ = _get_config(2048, N, K)
    print(N, K, "BLOCK_SIZE_M" in cfg, cfg.get("_source","?"))
PY
```
Expected: each target (N,K) resolves to a real config dict. **Gate:** the four 4B-active shapes no longer fall on the generic `"any"` fallback.

- [ ] **Step 7: Commit (configs are data; vendored in the aiter clone)**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add scripts/calibration/gen_gfx1201_a8w8_configs.py scripts/calibration/test_gfx1201_a8w8_coverage.py
git commit -m "MAD-294: gfx1201 a8w8 config generator + coverage gate (D)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
# Vendor the generated JSONs into the aiter clone's own git (separate repo):
git -C ~/GitHub/aiter add aiter/ops/triton/configs/gemm/ && \
git -C ~/GitHub/aiter commit -m "ml8: gfx1201 a8w8-blockscale tuned configs for 4B shapes (MAD-294)"
```

---

## Task 9: Post-bump + post-config microbench, assemble the table, Phase 0 exit gate [GPU]

**Files:**
- Create: `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md`

- [ ] **Step 1: Microbench — post-bump (no new configs effect on aiter path resolution differs from baseline)**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python microbench_a8w8_fp8.py --label post-bump --out /tmp/phase0_microbench_postbump.json`

- [ ] **Step 2: Microbench — post-bump + tuned configs**

Run: `cd /home/kmbandy/GitHub/llama.cpp/scripts/calibration && python microbench_a8w8_fp8.py --label post-bump+configs --out /tmp/phase0_microbench_postconfigs.json`
Expected: the `aiter_a8w8` cells at the 4B-active shapes improve vs `post-bump` (now hitting per-shape configs).

- [ ] **Step 3: Assemble the 4-cell table + exit-gate writeup**

Write `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md` with, per (shape, M):
the TFLOPS for the 4 cells — (1) pre-bump aiter-a8w8, (2) post-bump aiter-a8w8, (3) post-bump+configs aiter-a8w8, (4) `torch._scaled_mm` — plus the e4m3 mismatch counts (pre/post). Then the **Phase 0 exit-gate verdict**:
- Substrate confirmed? (does any aiter-a8w8 cell beat `_scaled_mm`, and by how much)
- Did D (configs) close the ~20% gap? (cell 3 vs cell 2)
- Did the bump change fp8 correctness? (e4m3 pre vs post)
- **Scope re-confirmation for Phases 1–3** given the numbers (proceed / re-scope).

- [ ] **Step 4: Persist + commit results**

```bash
cd /home/kmbandy/GitHub/llama.cpp
cp /tmp/phase0_microbench_postbump.json /tmp/phase0_microbench_postconfigs.json docs/superpowers/results/
git add docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md \
        docs/superpowers/results/phase0_microbench_postbump.json \
        docs/superpowers/results/phase0_microbench_postconfigs.json
git commit -m "MAD-294: Phase 0 microbench table + exit-gate verdict (substrate confirmed on data)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Close MAD-294 / report to user**

Update MAD-294 with the microbench table and the exit-gate verdict; report the scope re-confirmation for Phases 1–3 to the user before any Phase 1 work begins (the spec's gate).

---

## Self-Review

**Spec coverage (§5):**
- §5.1 E (Triton bump + AOT patch + reproducible rebuild) → Tasks 5, 6, 7. ✓
- §5.2 D (gfx1201 tuned configs) → Task 8. ✓
- §5.3 fp8 correctness verify (before/after #10458) → Tasks 3, 4(baseline), 7(post + turbo4 gate). ✓
- §5.4 microbench (4 cells at real shapes) → Tasks 1, 2, 4(baseline), 9. ✓
- §5 exit gate (re-confirm scope) → Task 9. ✓

**Placeholder scan:** the one soft spot is Task 8 Step 3 (`tune_one` → aiter's autotuner), flagged with an explicit implementer note + a named fallback path, because aiter's tuning util interface must be read at implementation time; `target_shapes`/`config_is_per_shape` are fully specified and tested. Task 7 Step 4 locates the turbo4 entrypoint by grep rather than hard-coding a path I have not verified. These are honest "confirm-at-exec" points, not vague TODOs.

**Type/name consistency:** `gemm_tflops`, `default_shapes`, `sweep_inputs`, `compare_codes`, `target_shapes`, `config_is_per_shape` are referenced identically in tests and impl. JSON `--label`/`--out` contract is uniform across both bench tools.

**Ordering:** baseline (Task 4) is captured BEFORE the bump (Task 7), so the pre/post cells are real. The patch-anchor preflight (Task 5) gates the build (Task 7) so a bad bump fails fast.

**Scope:** Phase 0 only. Phases 1–3 plans are deliberately deferred to their own gated plans per the spec.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-14-ml8-fp8-phase0-currency-microbench.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec then quality) between tasks, fast iteration. Note: Tasks 2, 4, 7, 8, 9 are **[GPU]** and run on the R9700; Tasks 1, 3, 5, 6 are CPU TDD.
2. **Inline Execution** — execute tasks in this session via executing-plans, batched with checkpoints (natural checkpoints before the GPU build in Task 7 and the tuning in Task 8).

Which approach?
