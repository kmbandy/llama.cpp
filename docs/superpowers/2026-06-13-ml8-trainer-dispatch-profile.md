# ml8 fp8-QAT trainer — dispatch profile + speedup plan (2026-06-13)

**Context:** the 4B fp8-QAT trainer runs ~2 min/optimizer-step on the R9700 (gfx1201),
GPU rarely above ~50% util. kmbandy (data engineer) flagged it: a 12c/24t ~5GHz CPU
should never leave the GPU starved. We profiled instead of theorizing. This doc is the
post-compact source of truth.

## What was WRONG (my earlier guesses, all disproven by profiling)
- "SSM scan is a slow sequential torch fallback / structural / move to MI300X" — FALSE.
  `fla_compat.py` shows fla's `chunk_gated_delta_rule` **Triton kernel runs on RDNA4**
  (cast to fp32 to dodge the bf16 `fdot2` intrinsic). The "[fla-shim] wrapped 48 ... to
  torch.float32 scan" log line = the fla Triton kernel run in fp32, NOT a torch loop.
- The SSM is **7%** of a real step. The fused-SSM-kernel idea would have bought ~7%.
- A plain-model fwd+bwd profile (no ml8 attach) is GPU-bound on GEMMs (rocBLAS picks a
  skinny `MT64x1x64` tile, ~42ms/call) — a RED HERRING. The real step with the ml8 op
  attached flips entirely to host-bound.

## The MEASURED bottleneck (real ml8 micro-step, 4B, 1×1024, grad-ckpt)
Profiled via the `PROFILE_STEPS` env hook in `smoke_fp8_qat.py` (env-gated block before
the train loop) — profiles the actual `wrapped(ids) → kl_topk → backward` micro-step.

```
wall:                          19,800 ms/micro  (~20s)
CPU blocked-on-GPU (sync):      1,329 ms  =  7% of wall   → GPU idle ~93%
CPU dispatch (LaunchKernel):    2,093 ms  → 77,012 launches / micro
top HOST self-time:
  Ml8Fp8Fn               12,863 ms (65%)   ← our fp8 custom op's host code
  hipMemcpyWithStream     3,694 ms          ← 2,000 DtoH copies
  hipLaunchKernel         1,968 ms          ← 73,808 launches
  hipDeviceSynchronize    1,291 ms          ← 400 device syncs (≈1 per ml8 layer fwd+bwd)
```
GPU is idle 93% of the time. It's host-bound, dominated by `Ml8Fp8Fn`.

## ROOT CAUSE (exact)
`fp8_qat.py:Ml8Fp8Fn.forward` calls `ml8_runtime.layer_from_components(...)` **every
forward, per ml8 layer (~200/micro)**. That function rebuilds the kernel layout from
scratch each call (`ml8_runtime.py`):
- **line 382** `idx_np = indices.cpu().contiguous().numpy()` → packs indices to 4-bit
  nibbles **in numpy on the CPU**, then `from_numpy(...).to(device)` back. The index
  buffers are FROZEN during Axis-A training (only change at reassign steps), yet every
  micro-step re-downloads ~all of them (down_proj ≈ 23MB), repacks on host, re-uploads.
  → the 2,000 memcpys + most of the 12.86s Ml8Fp8Fn host time.
- **line 374** `if not torch.equal(gidx.cpu().long(), expected_gidx)` → a DtoH sync
  validation of an invariant layout, every call → ~200 of the 400 syncs.

## TOP 3 SPEEDUP OPPORTUNITIES (sequence #1 → #2 → #3; each unblocks the next)

### #1 — Pack indices on-GPU + cache the kernel layer  (the 65%; biggest)
- Replace the `indices.cpu().numpy()` nibble-pack with torch bit-ops ON GPU:
  `(lo & 0xF) | ((hi & 0xF) << 4)` works on uint8 CUDA tensors (no numpy, no `.cpu()`).
- Cache the packed `Ml8Layer` (esp. `indices_packed`); invalidate only when indices
  actually change (reassign steps mutate `indices` in-place — use a version/dirty flag,
  NOT id() alone). Per-step rebuild then only re-casts centroids→fp8 + transposes scales
  (cheap, on-device).
- Bit-exact equivalence test vs current packing required (TDD).
- Impact: removes the bulk of 12.86s + most of 2,000 memcpys + 400 syncs → turns
  ~20s/micro into a few seconds.

### #2 — Kill the per-call `gidx.cpu()` + `torch.equal` sync-validation  (cheap)
- Validate uniform-contiguous grouping ONCE at attach (or check on-device without
  `.cpu()`), never per forward.
- Impact: ~200 syncs + 200 memcpys/micro gone. ~10 min of work.

### #3 — Cut the 77K launches/micro + feed the GPU  (last mile)
- Fuse the small elementwise (`snap_to_e4m3` e4m3-sim shows ~10,800 `bitwise_and` +
  4,400 `bitwise_or` + thousands of `where`/`copy_`; fp8-quant; pack) into the Triton
  path; drop the dense `[N,K]` W rebuild in `Ml8Fp8Fn.backward`.
- THEN the multithreaded producer/consumer dispatch queue (kmbandy's idea) to keep the
  GPU fed across remaining launches. MUST come after #1/#2 — a queue can't pipeline
  across a forced device sync.
- Impact: raises sustained GPU util toward saturation.

## Reproduce the profile
```
cd scripts/calibration
PROFILE_STEPS=3 python3 smoke_fp8_qat.py \
  --model ~/models/Qwen3.5-4B-hf --gguf ~/models/Qwen3.5-4B-ml8.gguf \
  --arms frozen --n-win 12 --steps 1 --eval-interval 1
```
(`PROFILE_STEPS` block prints CPU-blocked-on-GPU %, launches, syncs, and top device/host
ops, then exits. `profile_dispatch.py` is the plain-model variant.) RAM-safe: run under
`systemd-run --user --scope -p MemoryMax=11G` + `oom_score_adj=600` + the ram_watchdog.

## State at compact
- Streaming-anchor W_orig fix committed: **a7fc63fe** (fixed the gptq-arm crash).
- Axis-B verdict run was KILLED to free the GPU for profiling (never got the post-GPTQ
  KL). Axis-A reproduces 0.2090→0.1707 by step20; Axis-B number still pending.
- Profiler tooling (`profile_dispatch.py`, `PROFILE_STEPS` block in `smoke_fp8_qat.py`)
  — commit alongside this doc.
- mneme daemon (127.0.0.1:8810) was 500-ing on writes through this session.
