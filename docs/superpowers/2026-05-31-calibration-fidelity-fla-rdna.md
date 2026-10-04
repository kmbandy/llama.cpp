# Calibration Fidelity + fla-on-RDNA — Findings & Plan (2026-05-31)

Two threads from the 0.8B gauntlet session: (A) why calibration was CPU-bound and
how to fix it, and (B) the deployment-faithful / QAT insight that came out of it.

---

## A. Why calibration was hammering CPU (root cause + fix)

**Symptom:** "GPU" calibration pegged ~960% CPU/proc (~10 cores each), cells crawling;
"+70m since last cell" when cells should take ~35 min uncontended (c_wiki ran in 2122s
BEFORE the MI300X build started competing).

**Root cause (py-spy, decisive):** the MainThread sat in
`torch_chunk_gated_delta_rule` (HF `modeling_qwen3_5.py:286`) — the **pure-pytorch
reference** Gated-DeltaNet (linear-attention SSM) scan. HF falls back to it because
`flash-linear-attention` (fla) was **not installed** (`modeling_qwen3_5.py:407`:
`self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule`).
The reference scan is a sequential many-tiny-ops recurrence → host-dispatch-bound →
CPU pegged, GPU mostly idle between micro-launches.

**Key reframe:** the cost driver is the SSM scan, which scales with
`seq_len × n_samples × n_layers` — **NOT parameter count.** So a 0.8B *hybrid* model
is just as slow as a big one. (Earlier "switch to --resident" hypothesis was WRONG —
py-spy proved paging is a cheap device_memcpy; the bottleneck is the scan.)

**Second problem (open):** `compute_hessian` (calibrate_ml8.py:160) registers a hook
on ONE target linear, then runs a **full-model forward** over the whole calib set —
and it's called **once per target linear** (~188/cell). O(L²) in the slow scan.
The all-target-layers-in-one-forward optimization is missed. NOT fixed yet (interacts
with the QAT/sequential-propagation design below). Tracked as a follow-up.

### fla on RDNA4 (gfx1201) — the fix and the hard constraint

fla's `chunk_gated_delta_rule` Triton kernel emits the AMD **bf16 dot intrinsic
`llvm.amdgcn.fdot2.bf16.bf16`** (a CDNA/GFX9-class instruction). RDNA's LLVM backend
**cannot lower it** → `LLVM ERROR: Cannot select` → process core-dump (SIGABRT).
Building fla from source does NOT help — it's a GPU-ISA gap, not packaging.

**Per-dtype probe on cuda:0 (gfx1201 / R9700), isolated subprocesses:**
| dtype | result | note |
|---|---|---|
| fp16 | **OK** exit 0, finite | 30s cold JIT, cached after; sub-ms per-call |
| fp32 | **OK** exit 0, finite | 119s cold JIT; matches deployed f32 recurrence |
| bf16 | crashes cold (`Cannot select`); "passed" in 638ms ONLY with a warm cache from the fp16/fp32 runs → **fragile, do NOT rely on it** |

**Decision: run the calibration recurrence scan in fp32** (matches the deployed f32
recurrence core — ssm_a/dt/conv1d/norms are all f32 — AND reliably compiles on RDNA,
no bf16 fdot2). fla source kept at `~/GitHub/flash-linear-attention` (editable install
0.5.1). On **CDNA3 (gfx942, the MI300X)** bf16 fla works natively → arch-aware: bf16
on CDNA3, fp32 on RDNA. The MI300X Dockerfile should `pip install` fla too (free
speedup on the pod).

**Fix = arch-aware fla dtype shim:** after model load, walk modules, find each
linear-attn layer's `chunk_gated_delta_rule` attribute, wrap it to cast q/k/v/beta →
fp32 (RDNA) and the output back, leaving `g` fp32. Bottleneck (the scan) becomes a
fused GPU kernel instead of the CPU reference. Validate: re-run c_wiki, confirm PPL ≈
banked **19.2347** (fla-fp32 is quant-NEUTRAL — proves we changed speed, not the
answer) AND that it's fast.

### Resume device bug (separate, FIXED)
`load_dense_prefix_into_model` (calibrate_ml8_paged.py:815, paged branch) set
`weight_override = W.to(dtype)` — W is on CPU (blob loaded `map_location="cpu"`) and
the move-to-device was missing (the resident branch right above DOES move). On resume
from a partial dir, the next layer's Hessian forward pushed GPU activations through a
CPU override → `mat2 is on cpu, different from cuda:0`. Killed c_mix + s1_n128.
**Fix:** added `device=None` param, paged branch now `W.to(dtype=dtype, device=device)`,
call site passes `args.device`. Needs the paged-branch regression test (the existing
test only covers resident).

---

## B. Deployment-faithful calibration = QAT-flavored PPL lever (THE insight)

**The leak:** today the ml8-4 GEMMs DO propagate their own quant error (per-layer loop
sets `weight_override = dequant(quantized W)`, so layer i+1's Hessian sees quantized
upstream ml8-4). BUT everything that isn't ml8-4 runs at the model's **bf16** during
the calibration forward, when it ships at lower precision:
- token_embd → ships **ML8_FP8**, runs bf16. Pervasive: it's layer-0 input, so its fp8
  error should color EVERY downstream activation distribution — and currently doesn't.
- ssm α/β → ship **ML8_FP8**, run bf16.
- recurrence/conv/norms → ship **f32**, run at bf16.

So every Hessian `H` (which GPTQ assignment AND the heavy fine-tune both minimize
against, `tr((W−Wq)·H·(W−Wq)ᵀ)`) is built on activations **cleaner than inference**.
We're solving for the wrong target distribution → leaving PPL on the table. Closing it
is exactly what quantization-aware calibration / QAT does — a known PPL win, at
identical bits (a both-axes lever for MAD-256).

**Why this is the missing link for the heavy-FT lever (kmbandy insisted on keeping):**
heavy-FT tunes centroids+scales against `H`. If `H` is bf16-clean, heavy-FT optimizes
for a model that doesn't exist — which is the literature's dismissive "+0.03". Run it
against a deployment-faithful forward and it tunes for the REAL target — exactly where
that +0.03 should grow.

**Build (reuses existing pieces, cheap — the fp8 sims are one-time weight edits, no
extra O(L²) forward cost):**
1. Before the per-layer loop, quant→dequant the fp8 tiers (embed, α/β) through the
   existing scaled-FP8 quantizer (Exec T2) and install as overrides → forward now
   simulates the fp8 tiers.
2. Recurrence scan → fp32 fla (the section-A fix; faithful + compiles).
3. ml8-4 propagation stays as-is.
4. Heavy-FT then runs against that forward = real quantization-aware tuning.

**Attribution sequencing (measure each gain separately, all on the cheap 0.8B bed):**
- Step 1: fla-fp32 on GPU, NO fp8 sim, heavy OFF → c_wiki must ≈ 19.2347 (proves the
  fla integration is correct + quant-neutral; the win here is pure SPEED).
- Step 2: + fp8 embed/α/β sim, heavy OFF → measure ΔPPL vs 19.2347 = the
  deployment-faithful (QAT-calibration) gain.
- Step 3: + heavy-FT on top → measure the QAT fine-tune gain.
Then scale 0.8B → 2B → 4B → pod.

---

## Current state (end of 2026-05-31 session)
- **Killed:** content gauntlet (was on c_code), crank (was on s1_n512) — both stopped
  per kmbandy. Banked on disk: bf16 18.37, UD 18.50, s3_heavy0 19.3317, c_wiki 19.2347
  / held-out 13.0022, s1_n32 CALIB_OK. c_mix + s1_n128 dirs gone (the resume bug).
- **fla:** 0.5.1 editable-installed; HF gate True. ⚠ Until the fp32 shim is wired, a
  cold calibration run will core-dump on bf16 — wiring the shim is what closes this.
- **MI300X build:** still running (in-image Triton AMD-backend compile, MEM=5g cap).
- **Code edits uncommitted:** device-bug fix (calibrate_ml8_paged.py 794/815/1617);
  pre-compact: corpus loader (calib_corpus.py), heavy-LR wiring, method_gauntlet stages.
- **Host:** 15 GB RAM — never run heavy builds concurrent with calibration; build cap
  ≤6g. `sudo systemctl restart user@1000.service` recovers the desktop in place.

---

## UPDATE 2026-05-31 PM — W4A8 correction + fla validation (supersedes Part B framing)

### ⚠️ ml8 is W4A8, NOT W4A16 (the big correction)
Part B above framed "deployment-faithful calibration" as fp8 **weight** tiers (embed, α/β).
That's secondary. The dominant gap is **activations**. The ml8 GEMM
(`ggml/src/ggml-cuda/ml8.cu:430-518`, `ml8_quantize_activations_kernel ~771`)
quantizes **activations to e4m3** at every GEMM: per-row dynamic scale `row_absmax/448`,
`a_fp8 = e4m3(x/scale)`, on the **post-rotation** activation, then FP8×FP8 on the matrix
cores vs e4m3 centroid weights. The all-FP8 lattice IS the point.

**The real calibration gap:** `compute_hessian` (calibrate_ml8.py:160) builds `H = XᵀX`
on **bf16** activations; e4m3 activation rounding is **never modeled**. So GPTQ + heavy-FT
solve against activations cleaner than the hardware feeds. "Calibrate at the actual quant"
= **W4A8-aware**: collect `H` on the rotated, e4m3-quantized activation and propagate e4m3
activations. KG node **f4ffce4b** is authoritative.

**Honest framing (no overhype):** this does NOT remove the e4m3 rounding (it's the deployed
reality, already in the measured gap) — it lets weight-quant + heavy-FT **compensate** for it.
Recoverable PPL is bounded by how *structured* the noise is; per-token post-rotation e4m3 IS
the structured kind GPTQ/heavy-FT can partially cancel. Magnitude UNKNOWN until measured.

**Why you can't "quantize then calibrate":** GPTQ needs the **original** `W` + a faithful `H`
to solve `Wq`; pre-quantizing deletes the `W` it optimizes from. The sequential layer order
already gives each layer an optimally-quantized upstream input — so we **calibrate while
simulating the quantized forward** (keep current layer's original W, run e4m3 acts +
quantized upstream). The activation e4m3 has NO learned params (just absmax/448) → simulating
it in the Python forward IS "running the quantized activation path."

### fla validation RESULT (CPU/GPU thread closed)
Fresh `c_wiki` with the fla-fp32 shim (`gauntlet-0p8b-fla/`):
| | banked no-fla | fla-fp32 | Δ |
|---|---|---|---|
| wiki PPL | 19.2347 | **19.2678** | +0.033 (within ±0.05 noise floor) |
| held-out | 13.0022 | **13.0159** | +0.014 |
| wall-time | 2122s | **1572s** | **−26%** |
| CPU | ~960% pegged | **~242%** | off the peg |
**Quant-neutral ✅** (Δ inside the documented ±0.05 single-run noise floor). Speed is −26%
(NOT 10× — the scan is ~half the cell; the earlier *crawl* was build contention, now killed).
**New clean zero-point for QAT attribution = 19.2678.**

### Measurement caveat for QAT attribution
±0.05 PPL single-run noise floor on 0.8B means a sub-0.05 gain (e.g. the literature's +0.03
heavy-FT) **drowns in noise** in one run. Design the attribution with **paired/averaged runs**
or read the **composite** (all faithful tiers + heavy-FT) where signal clears the floor.

### State at compact
- CPU/GPU thread: CLOSED + validated. fla 0.5.1 editable-installed (`~/GitHub/flash-linear-attention`).
- MI300X build: **KILLED** (off critical path; rebuild later WITH fla + final calib code; on
  CDNA3 bf16 fla works natively → add `pip install fla` to the Dockerfile then).
- Uncommitted: `fla_compat.py` (new), `calibrate_ml8_paged.py` (device fix 794/815/1617 + fla
  shim import+call), `method_gauntlet.py` (`--cell` filter), `test_dense_resume.py` (paged
  regression test). Pre-compact: `calib_corpus.py`, heavy-LR wiring, stage defs.
- **NEXT: open the activation-e4m3 design pass** (brainstorm → spec → equivalence gate that
  bit-matches `ml8_fp32_to_e4m3` + rotation order vs `rotate_hessian` + per-row dynamic scale +
  no double-rotation → measure recovered PPL on 0.8B, heavy off then on).
