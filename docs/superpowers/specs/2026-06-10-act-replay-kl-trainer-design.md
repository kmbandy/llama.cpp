# Act-Replay fp8-Native KL Trainer (MAD-283 Stage 1) — Design

**Date:** 2026-06-10 · **Branch:** sync/upstream-2026-06-09 · **GPU:** R9700 (HIP_VISIBLE_DEVICES=0, ≤28GB)

## Problem

KL-to-parent pilot (PART 11) showed PPL ≠ identity: A3 beats UD on PPL 5/5 yet UD is ~2×
closer to the parent in KL everywhere (wiki 0.052 vs A3 0.115; agentic 0.114 vs 0.207).
Every closed-form one-shot lever is exhausted. The act-replay trainer is the frozen-index
special case of full QAT: gradient-descend the CONTINUOUS quant params against the bf16
teacher with KL as the loss — the metric that actually measures identity — through an
fp8-faithful forward. It gates the MI300X full-QAT spend.

## Goal

Take a deployed ml8 GGUF (first target: cell_A0_anchor_A3.gguf, 3187MB), train its 4-bit
codebooks + group scales under frozen indices to minimize held-out KL-to-bf16, and re-emit
a deployable GGUF. Success = KL(wiki) meaningfully toward UD's 0.052 from A3's 0.115 with
PPL not regressing past UD; floor gate = loss-down + KL improves at all.

## Non-goals

fp8-tensor codes/scales training (frozen), index re-assignment (Stage 2), fp8 KV cache
(round two), decode perf, 35B (until configs prove out).

## Architecture

New files in `scripts/calibration/`, reusing the existing stack.

### 1. `gguf_state.py` — GGUF → trainer state (the rehydrator)

No calib blob dirs survive; the GGUF is the canonical init. Reads from an ml8 GGUF:
- ML8_4 tensors → `indices` (uint8→long [N,K]), `centroids` [G,nc] (e4m3-decoded fp32),
  `scales` [N,G] — exact inverse of `ml8_to_gguf.py`'s packing, incl. token_embd sidecar.
- fp8/scaled-fp8 + bf16 tensors → frozen dequantized fp32 weights (act as constants).

**Gate (test):** for every ML8_4 tensor, `dequant(indices, centroids, scales)` bit-equals
the GGUF dequant. Reuses `ml8_e4m3_sim` for e4m3 decode. Also exposes the rotation
seeds/dims (kronecker sidecar metadata) so the faithful hooks match deployment.

### 2. `act_replay_student.py` — student model wrapper

Wraps the HF bf16 model (`Qwen3.5-4B-hf`). Per ML8_4 target:
- master `centroids`/`scales` fp32 leaf tensors (requires_grad),
- forward override `W = dequant(indices, cent_ste, scl)` recomputed under e4m3-STE
  (`codebook_finetune_rig.dequant`, snap_ste=True; token_embd via embedding+head override),
- `FaithfulActHook` (faithful_forward.py) for kernel-faithful e4m3 acts (Hessian capture OFF).
Non-targets keep the frozen rehydrated weights. fp8-NATIVE = STE weights + faithful acts.
Memory: gradient checkpointing per block; micro-batch 1×2048 + grad-accum; ≤28GB asserted.

### 3. `teacher_source.py` — `TeacherSource` interface (configurable strategy)

`get(batch) → (topk_idx [T,K], topk_logits [T,K], tail_logsumexp [T])`, K=256.
- `--teacher live` — bf16 teacher in-process, no_grad, same device (default; 4B local + MI300X).
- `--teacher cache` — one-time teacher pass writes per-token top-K + tail to disk
  (~10GB/512k; keyed by gguf+corpus+K, persisted under `~/models/act_replay/teacher_cache/`,
  not /tmp); training reads cache, no teacher resident.
- `--teacher device:N` — teacher on second device, overlapped (big-model local once 6900XT frees).
KL is exact w.r.t. tail-bucketing; eval shards always use the same source for self-consistency.

### 4. `act_replay.py` — trainer CLI

```
python3 act_replay.py --gguf cell_A0_anchor_A3.gguf --model Qwen3.5-4B-hf
  --corpus mix --token-budget 512000 --seq-len 2048 --corpus-seed 0      # calib_corpus, chat fmt
  --tensors-train ml8 --tensors-skip ''  (role/glob filter; ml8 = all ML8_4 incl. embed)
  --teacher live|cache|device:N --topk 256
  --lr-cent 1e-2 --lr-scale 1e-3 --steps N|--epochs E --grad-accum 8
  --out-dir ~/models/act_replay/A3 (ckpt+resume; blobs KEPT)
```
Loss = KL(teacher‖student) over response tokens, fp32, chunked over seq; LM head in fp32.
Holdout split (seed-stable 90/10) → train-KL/holdout-KL logged.
Eval: wiki/agentic held-out shards every N steps (top-K KL, same teacher source).
Output: updated centroids (e4m3-snapped)/scales → `ml8_to_gguf.py` re-emit → GGUF.
Determinism: ML8_DETERMINISTIC=1; time-based monitor logging; OOM guard.

## Smoke gates (before overnight)

1. Rehydration bit-equality (per-tensor).
2. Step-0 sanity: student==GGUF; train KL ≈ pilot magnitude.
3. 1-layer overfit on 1 batch → KL → ~0.
4. loss-down @200 steps real config, VRAM ≤28GB, throughput stable.
Overnight ~10–20k steps; morning re-score via llama-perplexity KL pipeline.

## Risks

GGUF reverse-read drift (gate 1 bit-equality protects); FLA RDNA bf16 crash → arch-aware
fp32 fla shim (existing); throughput ~teacher+student+bwd ≈ steps in low-thousands —
acceptable for codebook-only params (~tens of M).
