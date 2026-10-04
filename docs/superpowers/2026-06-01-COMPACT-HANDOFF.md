# ml8 Calibration — Compact Handoff (2026-06-01 PM)

Read this first on resume. Supersedes the live-state parts of `2026-05-31-COMPACT-HANDOFF.md`.
Authoritative KG nodes this session: **fb2749b3** (q-ladder results + q4 overfit), **5a087498**
(corpus-on-NVMe next-time), **61e96841** + Jira **MAD-259** (KV-as-warehouse, separate thread).

## ⭐ HEADLINE
1. **q3_actswt is the W4A8-faithful WINNER: wiki.test PPL 19.1392, −0.099 vs the q1_off
   baseline (19.2378), BPV-NEUTRAL (558 MB).** The two faithful levers stack monotonically &
   free: q2_acts (faithful e4m3 activations) −0.052, q3 adds faithful fp8 weights −0.046.
   Refs: bf16 18.37, UD-Q4_K_XL 18.50. 0.8B dense = UD's strongest regime (still +0.64 over UD)
   → MECHANISM validation, not a beat-UD claim. **This is the result to bank.**
2. **q4_heavy REGRESSED (overfit, CONFIRMED — not a bug): 19.743 (+0.50 PAST baseline).**
   Decisive evidence: calib Y_SNR went UP (median 24.21 vs q3 23.91) while wiki.test PPL went
   DOWN. Ruled out via cheap CPU diagnostic: sim↔kernel bit-mismatch (Gate A 9/9 pass),
   STE-e4m3 footgun (calib uses snap=none = the GOOD path), composition bug (heavy loop is clean
   Hessian-weighted opt, batched_gptq.py:394). Cause = high-capacity AQLM/PV heavy-FT
   (centroids+scales tune + index-reassign ×4 rounds) overfitting the small wiki calib set.
   NOTE heavy_rounds>0 IMPLIES act_order (calibrate_ml8_paged.py:1522) — they can't be isolated.

### The q-ladder (workdir /home/kmbandy/models/gauntlet-0p8b-qat, stage 6)
| cell | wiki.test PPL | Δ vs q1_off | secs | note |
|---|---|---|---|---|
| q1_off | 19.2378 | — | 1448 | W4A16-style baseline |
| q2_acts | 19.1855 | −0.052 | 3838 | + faithful e4m3 activations |
| **q3_actswt** | **19.1392** | **−0.099** ✅ | 4456 | + faithful fp8 weights — **WINNER** |
| q4_heavy | 19.743 | +0.505 ❌ | 13134 | + heavy-FT ×4 + act_order — OVERFIT |

## 🔑 THE BIG DISCOVERY (drove the redesign): token accounting was wrong
Measured with the REAL code paths (seed=0, CPU): wiki samples avg **601 tok**, mix samples avg
**1265 tok** (mix draws whole documents; wiki-2 qualifying rows are short paragraphs; both filter
≥512 and truncate at 2048).
- **The "262k token" labels everywhere are `n_samples × seq_len` UPPER BOUNDS, NOT actual.**
  q3's ACTUAL calib budget ≈ 128 × 601 ≈ **~77k tokens** (not 262k). All prior token-budget
  reasoning (incl. "32k→262k = −0.165") is inflated ~3×. **Re-baseline by ACTUAL tokens.**
- **q5 (mix) was confounded AND slow** → KILLED. Slow: 2.1× more tokens × O(seq²) attention =
  ~7× wall-time (was ~20% done at 102 min, projecting ~9 hr). Confounded: q5 used ~162k calib
  tokens vs q3's ~77k, so a q5-vs-q3 win couldn't separate CONTENT from MORE-TOKENS.

## NEXT ACTION (resume here) — the PRECISE lever-map
**Control variable MUST be total ACTUAL tokens, not n_samples** (sample length varies by corpus).

0. **Clean up:** `rm -rf /home/kmbandy/models/gauntlet-0p8b-qat/q5_actswt_mix` (confounded
   partial, still on disk — the last rm got interrupted by the restart; also dodges the
   resume-from-partial-loads-to-CPU bug).
1. **(Recommended) Add a token-budget mode** to calib_corpus.py / calibrate driver: draw samples
   until a target TOTAL token count, and have calibrate PRINT actual total tokens. Makes every
   corpus comparison token-matched by construction. (Quick alternative: just set n_samples per
   corpus to hit the budget — token-matched MIX ≈ **n=60** for ~77k tokens, since mix mean=1265.)
2. **Q1 — CONTENT sweep at FIXED ~77k tokens**, seq_len 2048, q3 recipe (faithful acts+weights,
   heavy OFF), `--holdout-eval`: wiki (control = 19.139) vs mix vs code vs math vs chat. Read =
   best AVERAGE of wiki.test + the never-train held-out (quant_so). Isolates CONTENT cleanly.
3. **Q2 — SIZE sweep** at fixed (winning) content: vary actual tokens (e.g. ~64k / ~128k / ~256k)
   to find saturation. (This is where the real "more tokens helps" number gets re-measured.)
4. **Q3 — heavy-FT LAST**, on the winning corpus+budget, WITH a held-out-gated early stop +
   fewer rounds / lower LR. Heavy is the real frontier lever (AQLM/PV) but must be fed
   diverse+sufficient data and stopped on held-out — naive ×4 on 77k wiki tokens overfit.

Resume launch pattern (token-matched mix content cell, once partial removed):
```
PYTHONPATH=$PWD/scripts/calibration:$PWD/gguf-py /usr/bin/python3 \
  scripts/calibration/method_gauntlet.py --stage 6 --cell q5_actswt_mix \
  --workdir /home/kmbandy/models/gauntlet-0p8b-qat --cal-device cuda:0 --ppl-device ROCm0 --holdout-eval
```
…BUT first edit the q5_actswt_mix cell's `--n-samples` from 128 → ~60 (token-match), OR implement
the token-budget mode (cleaner). Always pass `--no-mmap` to llama tools (perplexity already does).

## CODE CHANGED (uncommitted working tree — survives restart on disk; NOT yet committed)
- `scripts/calibration/calib_corpus.py` — **stackoverflow source repointed** from the raw 24.8 GB
  HTML dump (`/mnt/hdd/corpus/raw/stackexchange/stackoverflow.jsonl`, page-chrome/CSS boilerplate
  per record) to the cleaned `[HUMAN]:`-format `existing/stackoverflow_raw.jsonl` (4.6 GB).
  **Implication: yesterday's c_code used the DIRTY file → its number is suspect; re-run c_code.**
- `scripts/calibration/method_gauntlet.py` — added `q5_actswt_mix` cell to STAGES[6] (q3 recipe +
  `--corpus mix`). **Edit its `--n-samples` to ~60 before the real token-matched run.**
- Pre-existing uncommitted (from 2026-05-31): calib_corpus.py itself, `--corpus`/`--heavy-lr`
  wiring, method_gauntlet content stage 5 + dual-GPU flags. Consider committing the lot.

## HARDWARE / ENV STATE
- **All jobs killed** (q5 gauntlet TaskStop'd, monitors stopped). Box about to RESTART (user).
- HW MAP (verified this session): rocm-smi **GPU[0] = RX 6900 XT (16 GB)**, **GPU[1] = R9700
  (32 GB)**. HIP `cuda:0` → R9700 (rocm-smi GPU1) — HIP and rocm-smi enumerate in OPPOSITE order.
  So calibration runs on the R9700; the 6900 XT sits idle (could be the cuda:1 stage-1 helper).
- Corpus on SPINNING HDD `/mnt/hdd/corpus/raw` (~65 GB). Random-seek sampling is slow but NOT the
  q5 slowdown (corpus loaded in ~3 min). Still: pre-stage to NVMe / pre-sample cache next time
  (KG 5a087498). 15 GB host RAM — never raise build caps / no heavy builds concurrent with calib.
- Dual-GPU data-parallel Hessian = DESIGN ONLY, unimplemented, has an n_tok-weighted-merge
  correctness landmine + needs an equivalence gate. NOT a quick flag. Don't bolt on untested.

## ALSO PENDING (deferred, lower priority than the content sweep)
- FP4 frontier comparison (KG ea1a50a1): wire `llama-quantize NVFP4` CLI string → MXFP4 (iso-bpv
  4.25) + NVFP4 (4.5) GGUFs through the same perplexity harness vs ml8/UD/bf16.
- Commit today's fixes; optionally a Jira Story for the W4A8 0.8B faithful-calibration line.

## INTERPRETER (cost a failed launch before)
Calibration + tests MUST run under **/usr/bin/python3** (tf 5.6.1 native qwen3_5 + editable fla +
custom torch 2.13). The mlambaformer venv CANNOT load qwen3_5. pytest is installed in /usr/bin/python3.
