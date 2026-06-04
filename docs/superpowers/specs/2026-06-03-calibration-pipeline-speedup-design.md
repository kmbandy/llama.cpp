# Calibration Pipeline Speedup + Hackable Method Core — Design

**Status:** design (brainstormed 2026-06-03, MAD-256). Pending user review → writing-plans.

**Goal (one sentence):** Cut a real, full small-model calibration at 256k tokens from
~4h50m to **1–2h**, by building a clean, legible calibration *core* whose method is
separable and hackable — so we iterate fast *and* experiment fast — without weakening the
project's equivalence-gate discipline.

**Why now:** the calibration loop is the gate on every dense experiment ahead (re-test
battery #169, 512k/1M Hessian sweep #168, 9B #148, 27B-dense #128) and on the longer-term
push toward 3-bit-lossless (the NVFP4-beating play). Faster *real* runs = more shots at the
winning formula. The user's call: run the **real** thing (full calibration + full PPL),
never a fast-but-misleading proxy — so the substrate must make truth cheap, not fake it.

---

## 1. The measured target (data, not priors)

The 2026-06-03 Hessian-size sweep logged `calib_s` at four token budgets
(`/home/kmbandy/models/hessian-sweep/sweep.log`). Because we have four token counts, the
**token-scaling is itself a profile**:

| tokens | calib_s |
|---:|---:|
| 20,966 | 1,779 |
| 80,000 | 5,720 |
| 168,000 | 11,510 |
| 256,000 | 17,430 |

Linear fit (predicts all four within seconds): **`calib_s ≈ 383 + 0.0666 × tokens`.**

- **Fixed ≈ 383 s** (token-independent) = GPTQ sweep + Lloyd-Max + rotation + GGUF convert
  + model load. On the 256k run that's **~6.4 min — 2.2% of the 4h50m.**
- **Linear 0.0666 s/token** = the **Hessian-collection forward.** On 256k that's
  **~4h44m — 97.8% of the run.**

**Conclusion that reframes the whole effort:** tonight's GPTQ-sweep finding (≈98% of
*per-linear* compute in isolation) is a **2% rounding error on a real 256k run.** The
4.7 hours is the **Hessian-collection forward**. The engine must make *that* fast.

**Smoking gun for feasibility:** 0.0666 s/token = **~15 tokens/sec** for a 0.8B model — a
0.8B should forward *thousands* of tok/s. The Hessian collection is badly inefficient;
2–4h is very reachable, possibly better. Prime suspects (to be confirmed in Step 1):
faithful **per-layer re-forwarding** (×n_layers), **no/low batching**, the sequential
**SSM scan**, **fp32**, and **HDD corpus seek** interleaved.

---

## 2. Architecture (Approach B: clean method core on trusted primitives)

Separate three concerns that today are tangled across `calibrate_ml8.py`,
`calibrate_ml8_paged.py`, `batched_gptq.py`, `centroid_quantizer.py`:

- **Plumbing** (stable): corpus load, the model forward / Hessian collection harness,
  GGUF I/O, tier routing, equivalence gates.
- **Engine** (fast, batched): the vectorized/batched executors — Hessian forward
  (incl. multi-GPU), batched Lloyd-Max, Cholesky-escalation, the batched GPTQ sweep.
  **This is where almost all the 256k speedup comes from (the forward).**
- **Method core** (legible, hackable): explicit, swappable seams for the algorithm —
  `Objective`, `Codebook`, `Rotation`, `ErrorProp` — built *on top of* the engine's
  trusted primitives, never rewriting the hard-won correctness pieces (Cholesky escalation
  #151–153, the e4m3 sim, rotation reconciliation, batched Lloyd-Max 127 dB-equivalence).

**Scope discipline (YAGNI):** this is *not* a speculative platform. Build it as the fast
256k pipeline first (deliverable b); only **two seams are real on day one** — `Codebook`
and `ErrorProp` (the ones we'll hack toward 3-bit-lossless). `Rotation` and `Objective`
stay fixed trusted defaults until an experiment demands them.

**The c-payoff:** a method experiment ("throw a −1 in the error term", "try a lattice
codebook") becomes a one-function swap that (a) runs fast on the shared engine and (b)
cannot silently corrupt an invariant, because the swap is isolated behind the seam.

---

## 3. Plan of attack

### Step 1 — Instrument the real 256k Hessian forward (~20 lines, do FIRST)
Add lightweight phase timers around {corpus load, per-layer forward, XtX accumulate,
GPTQ, Lloyd-Max, convert} and run **one** 256k calibration. Output: the within-Hessian
breakdown, so we aim the forward work with numbers, not the §1 priors. **No design
decision past this point is committed until Step 1 returns.**

### Step 2 — Engine: make the Hessian forward fast (the hero work)
Pull the levers Step 1 ranks. Strong candidates (hypotheses):
1. **Dual-GPU data-parallel Hessian forward** (R9700 cuda:0 + 6900 XT cuda:1). A design
   already exists (shard-and-merge, **n_tok-weighted** combine `H=(n_A·H_A+n_B·H_B)/(n_A+n_B)`,
   stage-2 GPTQ stays 1 worker/GPU). ~2× on the dominant cost. Likely the single biggest lever.
   **Device selection is an explicit opt-in arg** (llama-server style, e.g.
   `--devices ROCm0,ROCm1` / `--devices ROCm0`), so a run picks single- vs dual-GPU at the
   command line. **Single-GPU is the default**; dual is opt-in. The shard-and-merge plumbing
   is gated behind this flag — one device = today's single-process path untouched; two
   devices = corpus-shard + n_tok-weighted merge. Mixed-arch caveat stays in force (the
   de-risk's `HIP_VISIBLE_DEVICES` discipline; gfx1201 + gfx1030 must not clash in dispatch).
2. **NVMe corpus staging** — pre-sample a small (~few-hundred-MB) doc cache to NVMe to kill
   the HDD random-seek tax (the diverse-corpus long pole).
3. **Forward efficiency** — batching / avoid redundant per-layer re-forwarding / sequence
   length — guided by Step 1.

### Step 3 — Method core + batched GPTQ cleanup (the c-bones, rides along)
Route dense GPTQ through the batched engine (vectorized Lloyd-Max + blocked updates),
behind the `Codebook`/`ErrorProp` seams. Cheap (2% of the clock) but it's where the
hackable surface lives — so we do it now while the core is being shaped.

---

## 4. Correctness & equivalence discipline

- **Feedback signal = the real thing:** full calibration + full PPL (wiki.test + held-out).
  Y_SNR is an **in-run triage glance** ("is a layer pathological?"), never the experiment gate.
- **Reference:** the current scalar/batched path is the equivalence reference. The optimized
  path must land **quality-equivalent within the existing noise floor** (the ~127 dB-SNR /
  fp32-reduction-order tolerance the batched MoE path already ships with), validated by
  Y_SNR + PPL — *not* bit-identical (the batched kernel isn't bit-identical to scalar; that's
  the accepted standard already in use for MoE).
- **Dual-GPU Hessian landmine (gate before trusting):** the cache stores *normalized*
  `H = XᵀX/n_tok`; a naive two-shard average is WRONG — must be **n_tok-weighted**. Persist
  per-(layer,kind) `n_tok` (currently not saved). Equivalence gate: `--max-layers 1`
  single-GPU vs dual-shard-merged Hessians match within fp noise. Do **not** skip.
- **256k acceptance:** the new pipeline reproduces the known-good 256k result
  (**wiki 19.5470 / held-out 12.2391**) within the PPL noise band, in the **1–2h** target
  band (stretch ≤1h). Equivalently: bring the forward from ~0.0666 s/token down to
  **~0.013–0.027 s/token** (≈ a **2.5–5× forward speedup**; ~15 tok/s → ~37–79 tok/s). The
  *same* per-token rate puts a 512k run at ~2–4h — i.e. the 256k 1–2h and the original 512k
  2–4h are one goal expressed at two token counts.

---

## 5. Success criteria

1. A real 256k calibration completes in **1–2h** (stretch: ≤1h) on the local R9700(+6900 XT)
   — i.e. a ~2.5–5× speedup on the Hessian forward.
2. PPL within the noise band of the current 256k baseline (19.5470 / 12.2391).
3. The dual-GPU Hessian equivalence gate passes (`--max-layers 1` merge == single-GPU).
4. `Codebook` and `ErrorProp` exist as clean seams; a trivial experiment (e.g. swap the
   codebook init) is a single-function change that runs end-to-end without touching plumbing.
5. No regression to the MoE/batched path or the existing equivalence gates.

---

## 6. Open questions (resolve during build, not blocking the spec)

- **Step 1 result** decides the Step 2 lever order; §1 priors are hypotheses.
- **Faithful re-forwarding:** does the faithful path re-run the forward per layer? If so,
  collapsing redundant forwards may beat even the dual-GPU lever. Step 1 reveals this.
- **fp32 Hessian:** the forward/XtX is fp32 for determinism/accuracy. Whether a bf16/tf32
  Hessian (free Y_SNR per tonight's bench) is acceptable for *production* (non-A/B) runs is a
  separate lever — flagged, not committed.

---

*Design v1, 2026-06-03. Next: user review → `writing-plans` for the implementation plan,
starting with the Step 1 instrumentation pass.*
