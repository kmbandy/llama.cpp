# ml8 Method Gauntlet — corpus + heavy-fine-tune sweep on dense small models

**Date:** 2026-05-31
**Owner decision (kmbandy):** Dense small models ARE the target (Qwen3.6-27B-class dense is
the most popular OSS model; "small dense for us = large frontier MoE for others"). The
per-tier precision *scheme* is correct (today's 0.8B breakdown proved `ssm_out`/`embed`
aren't the leak); **rotation is largely spent**; the lever to lean into is the **heavy
per-layer fine-tune + the calibration corpus** (size, contents, sequence length). The
per-layer codebook fine-tune is the *only* lever that has beaten one-shot PTQ on held-out
eval. Nail the corpus + the method → pull ahead.

## Where we stand (the only full-coverage data point that exists)

Qwen3.5-0.8B, llama-perplexity, `--no-mmap -c 512 --chunks 8`, wiki.test.raw, R9700 (ROCm0):

| Model | PPL | size | Δ vs bf16 |
|---|---|---|---|
| bf16 | 18.37 | 1485 MB | — |
| **ml8 full-coverage (baseline A0)** | **19.40** | 558 MB | **+1.03** |
| UD-Q4_K_XL (the target) | **18.50** | 546 MB | +0.13 |

On the 0.8B, UD currently wins BOTH axes. The +0.90-vs-UD gap is spread across the **bulk
4-bit attention + FFN** (uniform 4.25 bpv vs UD's mixed Q4/Q5/Q6) — NOT `ssm_out` (+0.14)
or embed (+0.02). The 4B "+0.08 vs f16" number is **FFN-only** (≈75% of the model was
unquantized) and is IRRELEVANT to full coverage — do not cite it.

**Test bed:** the 0.8B full-coverage pipeline works end-to-end (after the 3D `ssm_out`
mul_mat fix). Calibrate ≈ 5 min, convert ≈ 30 s, PPL ≈ 1 min on the R9700. Cheap enough
to sweep dozens of recipes locally before spending the $16/hr MI300X pod. Scale-up path
after a direction is found on 0.8B: **0.8B → 2B → 4B** (all local), then the pod for 9B/35B.

## The mechanism (why corpus is foundational)

One forward pass over the calibration corpus builds the per-(layer,kind) Hessian
`H = XᵀX` (activation covariance). **Both** the GPTQ index assignment **and** the heavy
fine-tune optimize the *same* objective: `tr((W−Wq)·H·(W−Wq)ᵀ)` = the error in the layer's
**output**, weighted by which input directions carry activation energy. So corpus →
Hessian → the objective everything optimizes. Better/bigger/more-representative corpus →
better target.

## Q1 levers — Corpus (`collect_wikitext_calibration` in calibrate_ml8.py)

Current sampler: takes the **first** `n_samples` rows of `wikitext-2-raw-v1` train (NOT
shuffled), one row/sample, truncated to `seq_len`. Default 32 × 1024 ≈ 32k tokens.

1. **Token count** — `--n-samples` (exposed). 32k is small (GPTQ/AQLM use 256k–1M+). The
   Hessian is a covariance estimate; more tokens = lower variance. Sweep 32k→512k, find
   saturation. *This is the "dump tokens" axis.*
2. **Sequence length** — `--seq-len` (exposed). **Underrated for this hybrid arch:** the
   delta-net SSM state evolves across the sequence, so a 1024-token calib under-samples the
   long-context activation distribution the recurrent gates operate in. Test seq_len as a
   SEPARATE axis, token-matched (128×1024 vs 64×2048 vs 32×4096 = 131k each) to isolate
   "longer context" from "more tokens".
3. **Contents** — NOT exposed (hardcoded wikitext-2, first-N, unshuffled). Biggest gap.
   wikitext-2 is tiny + encyclopedic; the model is used on code/chat/reasoning/diverse web.
   **Build needed:** (a) thread `dataset_name`/`config` to the CLI (function already takes
   them); (b) shuffle/random-offset sampling. Variations: wikitext-103 → C4/RedPajama →
   a deliberate mix (wiki+code+math+chat).

**Eval confound to design around:** PPL is *evaluated* on wikitext. Calibrating on wikitext
optimistically matches the eval. Honest test = calibrate on diverse content, eval on a
held-out set that ISN'T the calib distribution (a wikitext hold-out AND a non-wikitext eval).

## Q2 levers — Heavy fine-tune (`batched_gptq.py` heavy loop)

**Analysis:** AQLM/PV-tuning-style. Alternates `heavy_rounds` times: (a) gradient-tune
continuous centroids + per-group scales (Adam, `heavy_steps` iters) to minimize the
Hessian-weighted output error with 4-bit indices FROZEN; (b) re-assign each weight's index
Hessian-awarely (act_order GPTQ + error-prop) against the tuned codebook. All bit-free
(true 4.25 bpv). Learns ~17 floats/group.

**CLI-exposed levers:** `--heavy-rounds` (0=off; baseline gauntlet used 4), `--heavy-steps`
(60), `--heavy-dtype` (fp32|bf16), `--n-iter` (initial Lloyd-Max iters, 25), `--percdamp`
(0.01), `--fit-loss mse|mag_weighted` + `--mag-weight-p`, `--group-size` / `--group-size-down`,
`--n-centroids`, `--act-order`, `--snap-centroids none|e4m3`.

**Exists but HARDCODED (needs ~1-line wiring to sweep):** Adam LRs `heavy_lr_cent=1e-2`,
`heavy_lr_scale=1e-3` (batched_gptq.py:161-162).

**Refuted — do NOT re-try:** straight-through-e4m3 (`snap_ste`) HURTS (overfits the lattice).

**NOT built (the real near-lossless ceiling):** current heavy is **per-layer / block-wise**.
The frontier wins (QuIP#/AQLM/PV-tuning that beat Q4_K) come from **end-to-end** (whole-model
distillation) fine-tuning + **full index search** (beam/coordinate-descent over assignments,
not one GPTQ pass). Bigger build; only pursue if the cheap levers stall.

**Measurement caveat (bit us before):** the offline rig reports per-matrix Y_SNR, which
UNDERPREDICTS model ΔPPL (errors compound across layers). Prior heavy result was modest
(~+0.3 dB on down at matrix level); whether it compounds to a model-level PPL win is
UNMEASURED. The 0.8B test bed answers this at model level — measure PPL, not Y_SNR.

## The matrix — staged main-effects, then combine (run via `method_gauntlet.py`)

Baseline A0 = the command that produced 19.40 (rotation=kronecker, gs=64, nc=16,
percdamp=0.01, fit=mse, snap=none[convert snaps], heavy=0, n_samples=32, seq_len=1024).
Target = UD **18.50** (and bf16 floor **18.37**). Each cell: calibrate → convert (`--mtp-fp8`)
→ PPL(8 chunks) → log PPL + size + Δ-vs-UD; GGUF deleted after measuring.

**Stage 1 — token count** (seq_len=1024, heavy=0): n_samples ∈ {32, 128, 512} → 32k/128k/512k.
**Stage 2 — seq_len, token-matched ~131k** (heavy=0): {128×1024, 64×2048, 32×4096}.
**Stage 3 — heavy fine-tune** (best tokens+seq_len from 1&2): heavy_rounds ∈ {0, 4, 8}
  (heavy enables act_order). Also probe `--heavy-steps {60,120}` on the winner.
**Stage 4 — secondary levers** (on the stage-3 winner): `--snap-centroids e4m3` (calib-side),
  `--fit-loss mag_weighted` (down only via group-size-down semantics), `--group-size-down 32`.
**Stage 5 — combine** best-of-each → candidate config; report vs UD 18.50.

Decision rule: if Stages 1–4 don't move materially toward 18.50 on the 0.8B, that's the
signal the win needs the **end-to-end build** (Q2 "NOT built"), not more knobs — escalate
that decision to kmbandy before building it.

## Scale-up confirmation (after a 0.8B direction is found — all LOCAL)

Re-run the winning recipe on 2B then 4B, compare to each UD-Q4_K_XL, confirm the direction
holds / improves with scale (it should — the uniform-4-bit tax shrinks on bigger models).

**Model acquisition prerequisites (flag — needed before the scale step):**
- 2B: HF dir + UD-Q4_K_XL GGUF — MISSING (download).
- 4B: HF dir — MISSING (have bf16 GGUF); UD-Q4_K_XL GGUF — MISSING (download).
- 0.8B: HF + bf16 + UD — all present. ✓
(kmbandy grabbed the 0.8B UD manually; same for 2B/4B, or `hf download` the bases.)

## References / what's already known (do not re-derive)
- Rotation spent (kronecker near codebook-optimal); AWQ dead (per-channel, wrong axis for
  Qwen per-token outliers); mag_weighted mostly hurt on 35B; light codebook-FT down-only
  +0.1–0.26 dB. See KG: MAD-256 scorecard.
- ssm_out 4-bit costs +0.14 PPL, embed 8-bit costs +0.02 (today's 0.8B breakdown) — map is sound.
- 3D `ssm_out` mul_mat fix (ml8.cu:871/1041, M=ne[1]*ne[2]*ne[3]) — required for any
  full-coverage run to not explode.
