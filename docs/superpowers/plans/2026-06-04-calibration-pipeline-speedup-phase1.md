# Calibration Pipeline Speedup — Phase 1 (Instrument + Measure + Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument one real dense calibration so the within-forward cost split is measured (not assumed), confirm/quantify the per-target redundant re-forwarding, attribute the fp32-vs-WMMA (`allow_tf32=False`) tax, and produce the findings that gate Phase 2 lever selection.

**Architecture:** A tiny pure-Python phase-timer module (`calib_timing.py`) accumulates wall time per labelled phase. It is wired into the dense path of `calibrate_ml8_paged.py` behind a `--phase-timing` flag (off by default → zero behavior change to production runs). A second opt-in flag `--forward-dtype-probe` runs a small fp32-vs-tf32 micro-A/B on the live model to isolate the matmul-precision contribution. We validate the instrument on a cheap 21k-token run first, then take the real 256k numbers, then analyze and hand back to `writing-plans` for Phase 2.

**Tech Stack:** Python 3, PyTorch (ROCm), pytest, the existing `calibrate_ml8_paged.py --strategy dense` pipeline on the R9700 (gfx1201).

**Spec:** `docs/superpowers/specs/2026-06-03-calibration-pipeline-speedup-design.md` (branch `calib-pipeline-speedup`).

---

## Why Phase 1 is the whole plan (read first)

The spec's hard rule: **"No design decision past Step 1 is committed until Step 1 returns."** §6 explicitly flags that redundant per-layer re-forwarding may beat even the dual-GPU lever. Reading the code confirms the suspect is real:

- `calibrate_ml8_paged.py:1844` — dense per-target loop `for i, (name, layer) in enumerate(targets):`
- inside it, `:1861`/`:1866` call `compute_hessian(layer, calib, model, ...)`
- `calibrate_ml8.py:127` `compute_hessian` runs `for ids in calibration_ids: model(ids)` — **a full corpus forward, once per target.**
- the MoE path (`:1509`–`:1553`) already avoids this with one forward + simultaneous per-layer hooks.

So the prime Phase-2 candidate is "give the dense path the MoE single-pass treatment." But we **measure first** to (a) get real per-phase seconds to predict the post-fix runtime, (b) rank the *remaining* levers (dual-GPU, NVMe, fp32) for whatever the re-forward fix doesn't eliminate, and (c) honor the "ship the real number" discipline. After Task 6, this plan **stops and re-enters `writing-plans` for Phase 2.**

## File Structure

- **Create** `scripts/calibration/calib_timing.py` — `PhaseTimer` accumulator (one responsibility: wall-time-per-phase + JSON dump). ~60 lines.
- **Create** `scripts/calibration/test_calib_timing.py` — pytest for the timer.
- **Create** `scripts/calibration/analyze_phase_timing.py` — parse `phase_timing.json` → human breakdown table + forward-share computation. ~70 lines.
- **Modify** `scripts/calibration/calibrate_ml8_paged.py` — add `--phase-timing` / `--forward-dtype-probe` args; wrap corpus load, per-target `compute_hessian`, per-target quantize, and emit JSON. Dense path only.
- **Create** `docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md` — the measured breakdown + the Phase-2 lever ranking (Task 6 output).

---

### Task 1: `PhaseTimer` accumulator (pure-Python, TDD)

**Files:**
- Create: `scripts/calibration/calib_timing.py`
- Test: `scripts/calibration/test_calib_timing.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/calibration/test_calib_timing.py
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from calib_timing import PhaseTimer


def test_accumulates_seconds_and_calls():
    t = PhaseTimer()
    for _ in range(3):
        with t.phase("hessian_forward"):
            time.sleep(0.01)
    s = t.summary()
    assert s["phases"]["hessian_forward"]["calls"] == 3
    assert s["phases"]["hessian_forward"]["seconds"] >= 0.025
    assert s["total_seconds"] >= 0.025


def test_records_per_call_events_with_metadata():
    t = PhaseTimer()
    with t.phase("hessian_forward", target="blk.0.ffn_down", n_tok=2048):
        time.sleep(0.005)
    s = t.summary()
    ev = s["events"]
    assert len(ev) == 1
    assert ev[0]["label"] == "hessian_forward"
    assert ev[0]["target"] == "blk.0.ffn_down"
    assert ev[0]["n_tok"] == 2048
    assert ev[0]["seconds"] >= 0.004


def test_multiple_labels_kept_separate():
    t = PhaseTimer()
    with t.phase("corpus_load"):
        time.sleep(0.005)
    with t.phase("gptq_quantize"):
        time.sleep(0.005)
    s = t.summary()
    assert set(s["phases"]) == {"corpus_load", "gptq_quantize"}


def test_dump_json_roundtrips(tmp_path):
    t = PhaseTimer()
    with t.phase("corpus_load"):
        time.sleep(0.001)
    out = tmp_path / "phase_timing.json"
    t.dump_json(out)
    loaded = json.loads(Path(out).read_text())
    assert "corpus_load" in loaded["phases"]
    assert loaded["total_seconds"] >= 0.0


def test_exception_in_phase_still_records():
    t = PhaseTimer()
    try:
        with t.phase("gptq_quantize"):
            raise ValueError("boom")
    except ValueError:
        pass
    s = t.summary()
    assert s["phases"]["gptq_quantize"]["calls"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/calibration && python3 -m pytest test_calib_timing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'calib_timing'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/calibration/calib_timing.py
"""Lightweight phase-timer for calibration instrumentation (MAD-256 Phase 1).

One responsibility: accumulate wall time per labelled phase + optional per-call
events, dump to JSON. Zero deps beyond stdlib so it never perturbs the run.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class PhaseTimer:
    def __init__(self) -> None:
        self._phases: dict[str, dict] = {}
        self._events: list[dict] = []

    @contextmanager
    def phase(self, label: str, **meta):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            p = self._phases.setdefault(label, {"seconds": 0.0, "calls": 0})
            p["seconds"] += dt
            p["calls"] += 1
            if meta:
                self._events.append({"label": label, "seconds": dt, **meta})

    def summary(self) -> dict:
        total = sum(p["seconds"] for p in self._phases.values())
        return {
            "phases": self._phases,
            "total_seconds": total,
            "events": self._events,
        }

    def dump_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/calibration && python3 -m pytest test_calib_timing.py -v`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/calib_timing.py scripts/calibration/test_calib_timing.py
git commit -m "feat(calib): PhaseTimer accumulator for Phase-1 instrumentation"
```

---

### Task 2: Wire phase timers into the dense calibration path

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py` (arg parsing near `:1021`; corpus load near `:1295`; per-target loop `:1844`–end of loop; dump after the loop)

This instruments the four phases the spec names: corpus load, per-target Hessian forward (the suspected pig — recorded per target with `n_tok` so we can see N× re-forward directly), per-target quantize (GPTQ + Lloyd-Max + rotation/AWQ), and a final dump. It is gated behind `--phase-timing` so production runs are untouched. There is no unit test here (it edits a multi-hour GPU driver); Task 4's smoke run is the verification.

- [ ] **Step 1: Add the CLI flags**

In the argparse block (alongside `--strategy` at `:1021`), add:

```python
    p.add_argument("--phase-timing", action="store_true",
                   help="Phase-1 instrumentation: accumulate wall time per phase "
                        "(corpus/hessian_forward/gptq_quantize) and write "
                        "phase_timing.json into --output-dir. No effect on results.")
    p.add_argument("--forward-dtype-probe", type=int, default=0, metavar="K",
                   help="Phase-1: before the main loop, time K calib samples through "
                        "model() with allow_tf32 False (current/deterministic) vs True, "
                        "to isolate the fp32-vs-WMMA matmul tax. 0 = off.")
```

- [ ] **Step 2: Instantiate the timer next to the imports**

After the `from calibrate_ml8 import (...)` block (`:94`–`:101`), add:

```python
from calib_timing import PhaseTimer  # noqa: E402  (MAD-256 Phase-1 instrumentation)
```

And inside `main()`, immediately after `args = p.parse_args()`, add:

```python
    TIMER = PhaseTimer()  # accumulates only when --phase-timing; cheap regardless
```

- [ ] **Step 3: Time the corpus load**

Wrap the corpus collection call (near `:1295`, `collect_calibration(...)`). Change:

```python
    calib = collect_calibration(model, tok, n_samples=args.n_samples,
                                seq_len=args.seq_len, composition=args.corpus,
                                seed=args.corpus_seed, token_budget=args.token_budget)
```

to:

```python
    with TIMER.phase("corpus_load"):
        calib = collect_calibration(model, tok, n_samples=args.n_samples,
                                    seq_len=args.seq_len, composition=args.corpus,
                                    seed=args.corpus_seed, token_budget=args.token_budget)
```

- [ ] **Step 4: Time the per-target Hessian forward (record n_tok per target)**

In the dense per-target loop, wrap both `compute_hessian` call sites (`:1861` and `:1866`). The phase context cannot see `n_tok` until the call returns, so wrap the call and append the event afterward. Replace the `if args.faithful_acts: ... else: ...` Hessian block (`:1854`–`:1867`) with:

```python
        collect_awq = args.awq != "none"
        _t_hess0 = time.time()
        if args.faithful_acts:
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

(The bare `TIMER.phase("hessian_forward")` already accumulates the aggregate; the extra event row gives the per-target breakdown that exposes N× re-forwarding.)

- [ ] **Step 5: Time the per-target quantize (GPTQ + Lloyd-Max + rotation/AWQ)**

Find the quantize call in the loop (the `gptq_quantize_linear(...)` invocation after `effective_percdamp` is set near `:1909`). Wrap it:

```python
        with TIMER.phase("gptq_quantize", target=name):
            # ... existing gptq_quantize_linear(...) call and its surrounding
            #     rotation/AWQ application stay exactly as-is, just indented ...
```

Indent the existing quantize statement(s) under the `with`. Do not change any argument or order — only wrap.

- [ ] **Step 6: Dump the JSON after the per-target loop**

Immediately after the `for i, (name, layer) in enumerate(targets):` loop ends (before the convert/embed/fp8 tail near `:2032`), add:

```python
    if args.phase_timing:
        _pt_path = Path(args.output_dir) / "phase_timing.json"
        TIMER.dump_json(_pt_path)
        _s = TIMER.summary()
        print("\n=== [phase-timing] aggregate ===")
        for _lbl, _d in sorted(_s["phases"].items(),
                               key=lambda kv: -kv[1]["seconds"]):
            print(f"  {_lbl:18s} {_d['seconds']:9.1f}s  "
                  f"({100*_d['seconds']/max(_s['total_seconds'],1e-9):5.1f}%)  "
                  f"calls={_d['calls']}")
        print(f"  {'TOTAL':18s} {_s['total_seconds']:9.1f}s")
        print(f"[phase-timing] wrote {_pt_path}")
```

- [ ] **Step 7: Syntax check + import check**

Run: `cd scripts/calibration && python3 -c "import ast; ast.parse(open('calibrate_ml8_paged.py').read()); print('OK')"`
Expected: `OK`

Run: `cd /home/kmbandy/GitHub/llama.cpp && PYTHONPATH=gguf-py python3 scripts/calibration/calibrate_ml8_paged.py --help 2>&1 | grep -E "phase-timing|forward-dtype-probe"`
Expected: both flags appear in the help output.

- [ ] **Step 8: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py
git commit -m "feat(calib): wire PhaseTimer into dense path behind --phase-timing"
```

---

### Task 3: fp32-vs-WMMA forward-dtype probe

**Files:**
- Modify: `scripts/calibration/calibrate_ml8_paged.py` (after `calib` is built, before the per-target loop / strategy branch)

Isolates the matmul-precision contribution: the de-risk measured R9700 fp32 GEMM at 3.8 TFLOPS vs bf16 at 99.8 (≈26× on the matmul, because fp32 misses RDNA4 WMMA). `ML8_DETERMINISTIC=1` sets `allow_tf32=False` at import (`:84`). This probe times K samples through the live model with `allow_tf32` False vs True so we learn whether the determinism path is a hero lever or a rounding error *on top of* the re-forward cost. tf32 only affects matmuls (not the SSM scan), so the ratio isolates exactly OQ#3.

- [ ] **Step 1: Add the probe block**

After `calib` is assigned (Task 2 Step 3) and before `full_targets = list(find_dense_full_targets(...))` (`:1313`), add:

```python
    if args.forward_dtype_probe > 0:
        import torch.backends.cuda as _bcuda
        import torch.backends.cudnn as _bcudnn
        K = min(args.forward_dtype_probe, len(calib))
        probe = calib[:K]

        def _time_forward(tag):
            torch.cuda.synchronize() if args.device.startswith("cuda") else None
            t0 = time.time()
            with torch.no_grad():
                for ids in probe:
                    model(ids.to(args.device))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"[dtype-probe] {tag:18s} {dt:7.2f}s for {K} samples "
                  f"({dt/K*1000:7.1f} ms/sample)")
            return dt

        _saved = (_bcuda.matmul.allow_tf32, _bcudnn.allow_tf32)
        _bcuda.matmul.allow_tf32 = False
        _bcudnn.allow_tf32 = False
        dt_fp32 = _time_forward("allow_tf32=False")
        _bcuda.matmul.allow_tf32 = True
        _bcudnn.allow_tf32 = True
        dt_tf32 = _time_forward("allow_tf32=True")
        _bcuda.matmul.allow_tf32, _bcudnn.allow_tf32 = _saved
        print(f"[dtype-probe] fp32/tf32 forward ratio = {dt_fp32/max(dt_tf32,1e-9):.2f}x "
              f"(matmul-precision tax on this card)")
        _pp = Path(args.output_dir); _pp.mkdir(parents=True, exist_ok=True)
        (_pp / "dtype_probe.json").write_text(json.dumps(
            {"k_samples": K, "fp32_s": dt_fp32, "tf32_s": dt_tf32,
             "ratio": dt_fp32 / max(dt_tf32, 1e-9)}, indent=2))
```

- [ ] **Step 2: Confirm `json` and `time` are imported at module top**

Run: `cd scripts/calibration && grep -nE "^import json|^import time" calibrate_ml8_paged.py`
Expected: both present. If `json` is missing, add `import json` near the other stdlib imports and re-run.

- [ ] **Step 3: Syntax check**

Run: `cd scripts/calibration && python3 -c "import ast; ast.parse(open('calibrate_ml8_paged.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/calibration/calibrate_ml8_paged.py
git commit -m "feat(calib): --forward-dtype-probe to isolate fp32-vs-WMMA tax"
```

---

### Task 4: 21k instrumented smoke run — validate instrument + reveal re-forward shape [GPU CHECKPOINT]

**Files:** none (run + inspect)

**STOP — human checkpoint.** This launches a real GPU calibration (~30 min). Flag the human before dispatching. A 21k run is budget-independent for the *count* of targets (N× re-forward shows up regardless of token budget), so it validates the instrument AND likely answers the structural question at 1/12th the cost of the full 256k.

- [ ] **Step 1: Run the 21k instrumented calibration with the dtype probe**

```bash
cd /home/kmbandy/GitHub/llama.cpp
PYTHONPATH=gguf-py ML8_DETERMINISTIC=1 \
ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8" \
python3 scripts/calibration/calibrate_ml8_paged.py \
  --model /home/kmbandy/models/Qwen3.5-0.8B-hf \
  --gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --arch qwen35 --device cuda:0 --strategy dense \
  --output-dir /home/kmbandy/models/phase1/smoke_b20966 \
  --rotation kronecker --group-size 64 --n-centroids 16 --percdamp 0.01 \
  --fit-loss mse --dense-coverage full --faithful-acts --faithful-weights \
  --awq none --corpus mix --seq-len 2048 --corpus-seed 0 \
  --token-budget 20966 --no-resume \
  --phase-timing --forward-dtype-probe 8 \
  2>&1 | tee /home/kmbandy/models/phase1/smoke_b20966.log
```

- [ ] **Step 2: Verify the instrument produced output**

Run: `cat /home/kmbandy/models/phase1/smoke_b20966/phase_timing.json | python3 -m json.tool | head -30`
Expected: `phases` has `corpus_load`, `hessian_forward`, `gptq_quantize`; `events` contains many `hessian_forward_target` rows (one per target linear) each with `n_tok` and `target`.

Run: `cat /home/kmbandy/models/phase1/smoke_b20966/dtype_probe.json`
Expected: a `ratio` field. (A ratio ≫ 1 means fp32 is expensive on this card; ≈ 1 means it is not the lever.)

- [ ] **Step 3: Eyeball the re-forward shape**

Run: `grep -c hessian_forward_target /dev/stdin < <(python3 -c "import json,sys;[print(e['label']) for e in json.load(open('/home/kmbandy/models/phase1/smoke_b20966/phase_timing.json'))['events']]")`
Expected: a count = number of target linears (N). Confirm aggregate `hessian_forward` seconds ≈ N × (single corpus forward time). This is the N× re-forward, measured.

- [ ] **Step 4: Decide whether the full 256k run is needed**

If the 21k breakdown cleanly shows (a) `hessian_forward` ≫ `gptq_quantize` and (b) the per-target events confirm N× re-forwarding, the structural finding is settled and Task 5 becomes a *confirmation* we can fold into the next real calibration rather than a standalone 4.7h burn. Note the decision in the run log. **Surface the call to the human** (do not silently skip Task 5).

---

### Task 5: Full 256k instrumented run — real numbers [GPU CHECKPOINT, conditional on Task 4]

**Files:** none (run + inspect)

**STOP — human checkpoint.** ~4.7h GPU run. Only needed if Task 4 was inconclusive, or if the human wants the production-scale per-phase seconds on record (the spec's acceptance baseline is the 256k run). Reproduces the known-good `wiki 19.5470 / held-out 12.2391` while timing.

- [ ] **Step 1: Run the 256k instrumented calibration**

```bash
cd /home/kmbandy/GitHub/llama.cpp
PYTHONPATH=gguf-py ML8_DETERMINISTIC=1 \
ML8_TIER_OVERRIDE="token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8" \
python3 scripts/calibration/calibrate_ml8_paged.py \
  --model /home/kmbandy/models/Qwen3.5-0.8B-hf \
  --gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --arch qwen35 --device cuda:0 --strategy dense \
  --output-dir /home/kmbandy/models/phase1/full_b256000 \
  --rotation kronecker --group-size 64 --n-centroids 16 --percdamp 0.01 \
  --fit-loss mse --dense-coverage full --faithful-acts --faithful-weights \
  --awq none --corpus mix --seq-len 2048 --corpus-seed 0 \
  --token-budget 256000 --no-resume \
  --phase-timing \
  2>&1 | tee /home/kmbandy/models/phase1/full_b256000.log
```

- [ ] **Step 2: Convert + PPL to confirm equivalence (the baseline must still reproduce)**

```bash
cd /home/kmbandy/GitHub/llama.cpp
PYTHONPATH=gguf-py python3 scripts/calibration/ml8_to_gguf.py \
  --base-gguf /home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf \
  --calib-dir /home/kmbandy/models/phase1/full_b256000 \
  --out-gguf /home/kmbandy/models/phase1/full_b256000.gguf --allow-partial
build-hip/bin/llama-perplexity --no-mmap -m /home/kmbandy/models/phase1/full_b256000.gguf \
  -ngl 99 --device ROCm0 -f wikitext-2-raw/wiki.test.raw -c 512 2>&1 | tail -3
```

Expected: `Final estimate: PPL = 19.54xx` (within the noise band of 19.5470). This proves the instrumentation did not perturb the result.

- [ ] **Step 3: Verify the phase JSON**

Run: `cat /home/kmbandy/models/phase1/full_b256000/phase_timing.json | python3 -m json.tool | head -20`
Expected: aggregate `hessian_forward` ≈ 97% of `total_seconds`, consistent with the spec's `calib_s ≈ 383 + 0.0666·tokens` fit.

---

### Task 6: Analyze + write findings + GATE to Phase 2

**Files:**
- Create: `scripts/calibration/analyze_phase_timing.py`
- Create: `docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md`

- [ ] **Step 1: Write the analysis script**

```python
# scripts/calibration/analyze_phase_timing.py
"""Summarize a phase_timing.json from a Phase-1 instrumented calibration.

Prints the phase breakdown, the forward share, and the per-target Hessian-forward
distribution that exposes N-times redundant re-forwarding. Optionally folds in a
dtype_probe.json for the fp32-vs-WMMA ratio.

Usage:
  python3 analyze_phase_timing.py <output_dir>
"""
import json
import sys
from pathlib import Path


def main(out_dir: str) -> None:
    d = Path(out_dir)
    s = json.loads((d / "phase_timing.json").read_text())
    phases = s["phases"]
    total = s["total_seconds"]
    print(f"# Phase breakdown ({d.name}) — total {total:.1f}s")
    for lbl, p in sorted(phases.items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"  {lbl:20s} {p['seconds']:9.1f}s  "
              f"{100*p['seconds']/max(total,1e-9):5.1f}%  calls={p['calls']}")

    tgt = [e for e in s["events"] if e.get("label") == "hessian_forward_target"]
    if tgt:
        n = len(tgt)
        secs = sorted(e["seconds"] for e in tgt)
        agg = phases.get("hessian_forward", {}).get("seconds", 0.0)
        per = agg / max(n, 1)
        print(f"\n# Hessian-forward re-forwarding")
        print(f"  targets (N)            {n}")
        print(f"  aggregate forward      {agg:.1f}s")
        print(f"  mean per-target        {per:.1f}s  (= one full corpus forward)")
        print(f"  implied 1-pass forward {per:.1f}s  vs  N-pass {agg:.1f}s "
              f"→ up to {n:.0f}x headroom if collapsed to a single pass")
        print(f"  per-target min/median/max  "
              f"{secs[0]:.1f} / {secs[n//2]:.1f} / {secs[-1]:.1f}s")

    probe = d / "dtype_probe.json"
    if probe.exists():
        pj = json.loads(probe.read_text())
        print(f"\n# fp32-vs-WMMA matmul tax")
        print(f"  allow_tf32=False  {pj['fp32_s']:.2f}s / {pj['k_samples']} samp")
        print(f"  allow_tf32=True   {pj['tf32_s']:.2f}s / {pj['k_samples']} samp")
        print(f"  ratio             {pj['ratio']:.2f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
```

- [ ] **Step 2: Run it on the smoke (and full, if present)**

Run: `cd scripts/calibration && python3 analyze_phase_timing.py /home/kmbandy/models/phase1/smoke_b20966`
Expected: a breakdown table + the N× re-forwarding block + the dtype ratio.

- [ ] **Step 3: Write the findings doc**

Create `docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md` containing the measured numbers from Step 2, structured as:
- **Phase split** (corpus / hessian_forward / gptq_quantize / total) with percentages.
- **Re-forwarding verdict**: N (target count), aggregate forward seconds, per-target seconds, the implied single-pass headroom.
- **fp32 tax verdict**: the probe ratio + whether fp32 is a hero lever or a rounding error.
- **Phase-2 lever ranking** (the gate output): order the spec's §2 candidates by measured payoff — e.g. "(1) collapse dense to a single all-targets forward (port the MoE `:1509`–`:1553` pattern), (2) fp32→tf32/bf16 forward mode behind `--deterministic` vs `--fast` [if ratio warranted it], (3) dual-GPU `--devices`, (4) NVMe corpus staging" — with each lever's predicted contribution from the measured seconds.

Use real numbers from the run; no placeholders.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibration/analyze_phase_timing.py \
        docs/superpowers/notes/2026-06-04-calibration-phase1-findings.md
git commit -m "docs(calib): Phase-1 timing findings + Phase-2 lever ranking (gate)"
```

- [ ] **Step 5: GATE — re-enter writing-plans for Phase 2**

Phase 1 is complete. **Do not start building Phase-2 levers from this plan.** Re-invoke `superpowers:writing-plans` with the findings doc as input to produce the Phase-2 implementation plan (the single-pass dense forward + whichever remaining levers the measured ranking justifies, each with the equivalence gate the spec requires: 256k reproduces `wiki 19.5470 / held-out 12.2391` within the noise band, and — for dual-GPU — the `--max-layers 1` single-vs-merged Hessian fp-noise gate).

---

## Self-Review

**Spec coverage (Phase 1 scope only):**
- Spec §3 Step 1 "instrument the real 256k forward, ~20 lines, do FIRST, no decision committed until it returns" → Tasks 1–6 (timer + wiring + run + gate). ✔
- Spec §3 Step 1 phases {corpus load, per-layer forward, XtX accumulate, GPTQ, Lloyd-Max, convert} → Task 2 times corpus_load, hessian_forward (forward+XtX together, since the XtX runs inside the hook during the forward — they are not separable without a second probe, which §1 does not require), gptq_quantize (GPTQ+Lloyd-Max+rotation/AWQ). Convert/PPL are timed by the surrounding shell (Task 5 Step 2) and are the spec's fixed ~383s tail, so not re-instrumented. ✔
- Spec §6 "faithful re-forwarding — does the forward re-run per layer?" → Task 4 Step 3 + Task 6 re-forwarding block answer this directly. ✔
- Spec §6 + §1 "fp32 Hessian / allow_tf32" suspect → Task 3 probe + Task 6 fp32 verdict. ✔
- Spec §3 Steps 2–3 (levers, method core) → **deliberately deferred** to the Phase-2 plan via the Task 6 gate, per the spec's own "no decision past Step 1" rule. Flagged, not dropped. ✔

**Placeholder scan:** No TBD/TODO. The only "fill in real numbers" is Task 6 Step 3 (the findings doc), which by nature records measured output and is explicitly instructed to use real numbers. ✔

**Type/name consistency:** `PhaseTimer.phase(label, **meta)`, `.summary()`, `.dump_json(path)` used identically in Tasks 1, 2, 3, 6. JSON keys (`phases`, `total_seconds`, `events`, `hessian_forward_target`, `n_tok`, `ratio`) match between the writer (Task 2/3) and reader (Task 6). Output dirs (`/home/kmbandy/models/phase1/...`) consistent across Tasks 4–6. ✔
