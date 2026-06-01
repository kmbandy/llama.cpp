# ML8 QAT/Calibration — Compact Handoff (2026-05-31 PM)

Read this first on resume. Detailed technical doc:
`docs/superpowers/2026-05-31-calibration-fidelity-fla-rdna.md` (has the W4A8 correction +
fla validation in its "UPDATE 2026-05-31 PM" section). KG authoritative node: **f4ffce4b**.

## ⭐ THE HEADLINE (where we are)
1. **CPU/GPU thread = CLOSED + validated.** Calibration was CPU-bound because HF qwen3.5 fell
   back to the slow pure-pytorch `torch_chunk_gated_delta_rule` SSM scan (fla wasn't installed).
   Fixed with an **arch-aware fla fp32 shim** (`fla_compat.py`): on RDNA the gated-delta kernel
   can't lower bf16 `fdot2` (core-dumps) but runs fine in **fp32**, which also matches the
   deployed f32 recurrence. Validated: c_wiki **19.2678** vs banked 19.2347 = **quant-neutral**
   (within ±0.05 noise), **−26%** wall-time, CPU **960%→242%**.
2. **THE BIG INSIGHT (next sub-project): ml8 is W4A8, not W4A16.** The ml8 GEMM quantizes
   **activations to e4m3** (per-row absmax/448, post-rotation) at every GEMM. Our calibration
   models **zero** activation quant (`H = XᵀX` on bf16 acts). So we've been calibrating against
   activations cleaner than the hardware feeds → PPL left on the table. Closing it =
   **W4A8-aware / QAT-flavored calibration** = the path to 4.25-bpv near-lossless (kmbandy's
   "this is our ticket"). Honest: it lets weight-quant + heavy-FT **compensate** for the e4m3
   activation noise (doesn't remove it); magnitude UNKNOWN until measured.

## NEXT ACTION (resume here) — UPDATED 2026-06-01 (impl in progress)
Design + plan are DONE and the implementation is mostly built:
- **Spec:** `docs/superpowers/specs/2026-05-31-w4a8-faithful-calibration-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-31-w4a8-faithful-calibration.md` (10 tasks, 4 phases)
- **Tasks 1–8 DONE + committed** (8 commits `2ddd8495d`..`7536d4ef6`), 12 pytest pass:
  `ml8_e4m3_sim.py` (Gate A bit-exact vs kernel golden over 78,794 cases incl 256–448 band;
  vectorized per-row quant scalar-parity), `faithful_forward.py` (`build_rotations`,
  `FaithfulActHook` x_eff=rotation.inverse(a_q) identity verified, `assert_not_double_rotated`,
  `fp8_weight_override`), `--faithful-acts`/`--faithful-weights` wired into the dense branch of
  `calibrate_ml8_paged.py` (rotate_hessian skipped+guarded; legacy path byte-identical),
  `method_gauntlet.py` STAGE **6** paired-toggle (q1_off≡c_wiki, q2_acts, q3_actswt, q4_heavy).
- **Task 9 Gate C DONE — PASS:** `q1_off = 19.2378` (558 MB, 1448s). −0.030 vs banked fla
  19.2678, +0.003 vs banked no-fla 19.2347 — within the documented ±0.05 floor; fla active
  (1448s « 2122s no-fla path); legacy path byte-identical by inspection. Refactor-neutral.
  (Plan's ±0.01 gate was over-tight vs the project's own ±0.05 floor — corrected.)
- **Task 10 — q2 DONE, ★ THE RESULT ★:** `q2_acts` (faithful-acts ON, weights/heavy OFF) =
  **19.1855** → **Δ −0.0523 vs q1_off 19.2378**. Activation-e4m3-faithful calibration ALONE,
  bit-free. CLEAN signal (shared `--corpus-seed 0` → deterministic pairing; Δ ≈ pure effect,
  not noise). kmbandy: braced for +0.02-0.03 sign, got −0.052 = "the show." KG c61814f4.
  Honest: 0.8B dense = UD's strongest regime, still +0.686 over UD — MECHANISM validation,
  not a beat-UD claim. q2 cost 3838s (~2.6× q1_off — faithful fwd overhead; budget for scale).
- **NEXT (deferred to 2026-06-01 eve — getting late):** run `q3_actswt` (+fp8 weight tiers)
  and `q4_heavy` (+heavy-FT on the faithful FP8 lattice — the lever the reframe unlocks).
  Resume command (box idle, GPU free):
  ```
  PYTHONPATH=$PWD/scripts/calibration:$PWD/gguf-py /usr/bin/python3 \
    scripts/calibration/method_gauntlet.py --stage 6 --cell q3_actswt,q4_heavy \
    --workdir /home/kmbandy/models/gauntlet-0p8b-qat --cal-device cuda:0 --ppl-device ROCm0
  ```
  Success bar: q3 must not regress >+0.05 vs q1_off; q4 should clear a real gain. Then 3-seed
  finalize on the winner (q4 has heavy-FT RNG → most worth averaging). Banked: q1_off 19.2378,
  q2_acts 19.1855 in `/home/kmbandy/models/gauntlet-0p8b-qat/*/results.md`.
- **pytest now in /usr/bin/python3** (KG ac0fb4e3) — use it for BOTH calibration AND tests;
  the mlambaformer venv is retired for this work. Custom torch (editable @ ~/GitHub/pytorch)
  verified untouched.

**⚠ INTERPRETER (cost me one failed launch — KG 53a5dd94):** calibration MUST run under
**/usr/bin/python3** (tf 5.6.1 native qwen3_5 + editable fla + torch 2.13). The mlambaformer venv
(`/home/kmbandy/GitHub/mlambaformer/.venv/bin/python`) is ONLY for pytest — its tf 4.57.1 can't
load qwen3_5 (→ CALIB_FAIL at model load). Gate C / measurement command:
```
PYTHONPATH=$PWD/scripts/calibration:$PWD/gguf-py /usr/bin/python3 \
  scripts/calibration/method_gauntlet.py --stage 6 --cell q1_off \
  --workdir /home/kmbandy/models/gauntlet-0p8b-qat --cal-device cuda:0 --ppl-device ROCm0
```

**NEXT after Gate C passes (q1_off ∈ [19.258,19.278]):** Task 10 — run the full stage 6
(q2_acts, q3_actswt, q4_heavy) paired vs q1_off; heavy uses `--heavy-rounds 4 --act-order`;
then 3-seed average on the winner. Write results into the fidelity doc + a KG session_summary.
If Gate C is OFF, STOP and debug (systematic-debugging) before trusting any faithful number.

**Measurement caveat:** ±0.05 single-run PPL noise floor on 0.8B → sub-0.05 gains (e.g. the
+0.03 heavy-FT) drown in one run. Use paired/averaged runs OR read the composite effect.

## CODE CHANGED (uncommitted working tree, branch as-is)
- `scripts/calibration/fla_compat.py` (NEW) — `apply_fla_arch_shim(model, device)`: wraps each
  linear-attn layer's `chunk_gated_delta_rule` to fp32 on RDNA (CDNA/NVIDIA/CPU = no-op; torch
  fallback = skipped). Unit-verified (gfx1201→fp32, cpu→no-op).
- `scripts/calibration/calibrate_ml8_paged.py` — (a) resume **device bug fix**: `load_dense_prefix_into_model`
  now takes `device=` and the paged branch does `W.to(dtype=dtype, device=device)` (was leaving
  the reconstructed weight on CPU → `mat2 is on cpu` on resume); call site (~1617) passes
  `args.device`. (b) fla shim import (after calib_corpus import) + `apply_fla_arch_shim(model,
  args.device)` call right before the calib-corpus block (~line 1112).
- `scripts/calibration/method_gauntlet.py` — `--cell <name>` filter (run one named cell within a
  stage; use with a fresh `--workdir` to force clean re-calibrate). Used for the fla validation
  and the upcoming QAT attribution runs.
- `scripts/calibration/test_dense_resume.py` — NEW `test_dense_resume_paged_override_on_device`
  (meta-device sentinel, GPU-free; guards the resume device bug). All 3 resume tests pass.
- Pre-compact (still uncommitted): `calib_corpus.py` (corpus content loader), heavy-LR wiring
  (`--heavy-lr-cent/scale`), method_gauntlet stage defs (content stage 5, heavy-LR cells).

## ENV STATE
- **fla 0.5.1 editable-installed** at `~/GitHub/flash-linear-attention` (source build; the 0.5.0
  pip wheel was missing `fla/ops` — DON'T use the wheel). HF gate True. Safe because the shim
  routes RDNA→fp32. On CDNA3 (MI300X) bf16 fla works natively.
- **MI300X build KILLED** (was a 3h in-image Triton compile; off critical path — pod is after
  0.8B/2B/4B local). Rebuild later WITH fla added to the Dockerfile + final calib code.
- Box clean: no stray procs, both GPUs idle, ~11 GB RAM free, swap ~950 MB. Monitor bx1jx1h2m ended.
- **Host RAM = 15 GB**: never run heavy builds concurrent with calibration; build cap ≤6g;
  `sudo systemctl restart user@1000.service` recovers desktop in place.

## RESULTS BANKED (`gauntlet-0p8b/results.md` + `gauntlet-0p8b-fla/results.md`)
refs: **bf16 18.37 | UD-Q4_K_XL 18.50** (win BOTH axes = the MAD-256 goal). A0 (32k wiki)=19.40.
| cell | wiki PPL | held-out | note |
|---|---|---|---|
| s3_heavy0 | 19.3317 | — | heavy-OFF baseline, 262k wiki |
| c_wiki (no-fla) | 19.2347 | 13.0022 | original 262k wiki control |
| **c_wiki (fla-fp32)** | **19.2678** | 13.0159 | **the QAT zero-point** (quant-neutral, −26%) |
| s1_n32 | (calib only) | — | 32k blob |
KEY FINDING (pre-fla): 32k→262k tokens = −0.165 PPL (tokens alone, 8×) > literature's 0.03.

## KG NODES THIS SESSION
- **f4ffce4b** — W4A8 correction + activation-aware calibration (AUTHORITATIVE).
- e5cc0263 — QAT decision (weight-tier framing; PARTIALLY SUPERSEDED by f4ffce4b).
- f6d087c7 — fla-on-RDNA hardware fact (fdot2/fp32).
- 93f3739c — monitoring feedback (watch resource-shape + throughput, not just errors).
- 4c02f178 — OOM rule (15 GB host); cc9a9f65 — harness/device naming.

## STRATEGY CONTEXT (unchanged)
MAD-256: dense-hybrid Qwen3.5 → ~4.25bpv ml8 that beats Unsloth UD-Q4_K_XL on BOTH axes
(smaller AND lower PPL). Cheap 0.8B bed → 2B → 4B local → 8×MI300X pod. Heavy fine-tune KEPT as
first-class ("tuned correctly"). Sequence now: **activation-e4m3-aware calibration** (the W4A8
fix) is THE keystone lever; stacks with heavy-FT + rotation + bpv-neutral levers (mag_weighted,
group_size). Corpus content/size sweeps (c_code/c_math/c_chat, 512k/1M grid) were paused mid-run
and remain TODO but are lower priority than the QAT keystone.
