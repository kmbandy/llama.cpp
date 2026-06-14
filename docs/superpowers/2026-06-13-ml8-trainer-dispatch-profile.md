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

---

## UPDATE 2026-06-13 (evening) — #223+#224 landed, re-profiled, new bottleneck

### Result: 1.87× faster, root cause confirmed
Re-ran the identical 4B / 1×1024 profiled micro-step after committing #223
(`192e7da8c`, pack indices on-GPU + cache packed layer) and #224 (`da0e0ca3b`,
validate gidx once per buffer).

| Metric | Before | After | |
|---|---|---|---|
| wall | 19,800 ms | **10,603 ms** | **1.87×** |
| CPU blocked-on-GPU | 1,329 ms (7%) | 2,387 ms (23%) | flipped toward GPU-bound |
| **Ml8Fp8Fn host self** | **12,863 ms (65%)** | **207 ms** | the host mountain is gone |
| DtoH memcpys | 2,000 | 800 | index repack + gidx copy removed |
| kernel launches | 77,012 | 76,964 | unchanged (still to do) |

The `indices.cpu().numpy()` repack collapsed 12.86s → 0.21s exactly as predicted.
The step is no longer host-bound on our custom op. (Both numbers are *profiled*
wall — profiler inflates CPU — so they're apples-to-apples; unprofiled wall is
lower.) RAM-safe SOP held: watchdog re-asserted oom_adj=600, RAM dipped to 873MB
but stayed above the 600MB floor, clean exit.

### NEW dominant cost: bf16 GEMM on the *frozen* weights (only ml8 targets are fp8)
After-profile top GPU cost is **`aten::mm` 4,549 ms (1,195 calls)** — bf16 linears,
NOT fp8. Root cause found in the rehydrate path (`smoke_fp8_qat.py:181-189`):
- **ml8 targets (200)** → `_attach_one(fp8=True)` → real `Ml8Fp8Fn` a8w8 fp8 kernel.
  These are the only genuine fp8-compute layers.
- **"fp8-tier" tensors (49)** → the `else` branch → `_install_one(..., torch.bfloat16,
  ...)` → `mod.weight.copy_(w.to(dtype=bfloat16))` (`act_replay.py:888`). "fp8-tier"
  is their *storage/deployment* tier; in this PyTorch trainer they are dequantized
  to **bf16** and run as ordinary `nn.Linear` → bf16 `aten::mm`.
- **Untiered residual linears** (anything not in the GGUF stream: SSM gates/projections,
  head, whatever calibration left un-quantized) keep their bf16 parent weights → also
  `aten::mm`.

So the frozen, non-trained weights took a bf16 shortcut (numerically fine — frozen,
carrying fp8-faithful dequant values — but a big chunk of GEMM FLOPs runs bf16). In
*deployment* (llama.cpp C++) those same weights run `ML8_FP8` WMMA (fp8). Why it's
slow specifically: rocBLAS mis-selected **`MT16`/`MT8` skinny tiles (~7 ms/call)** for
M≈1024 GEMMs that should use `MT128` (~1 ms/call). rocBLAS tile autotune is
non-deterministic — the *before* run happened to pick MT128, this run picked MT16.

### Remaining speedup backlog (re-scoped from fresh numbers)
Ranked by lever size on the 10.6s step:
1. **Route frozen fp8-tier + untiered linears through the fp8 a8w8 path** (the one
   the ml8 targets already use). Matches deployment numerics AND sidesteps the bad
   rocBLAS tiles. Biggest lever: ~4.5s of bf16 `aten::mm` → est ~1.5s. **(was #225's
   surprise — promote to #225-A.)**
2. **Trace + kill the remaining 800 DtoH memcpys** (`hipMemcpyWithStream` 3,852 ms) —
   ~4 per ml8 layer, likely the backward dense-`W` rebuild / fp8-quant amax path.
3. **Cut the 77K launches** (`hipLaunchKernel` 2,378 ms) — fuse the small elementwise
   (`snap_to_e4m3`: ~10.8K `bitwise_and` + 4.4K `bitwise_or` + `where`/`copy_`;
   fp8-quant; index pack) into the Triton path; drop the dense `[N,K]` W rebuild in
   `Ml8Fp8Fn.backward`. The original #225 framing.
4. **Multithreaded producer/consumer dispatch queue** (kmbandy's idea) — keep the GPU
   fed across remaining launches. Only pays off after the forced syncs/host-serial
   points are gone (which #223/#224 mostly did).

### Open / not-yet-pinned
- **Confirm the rocBLAS tile non-determinism** with one more profile before committing
  to lever #1 — verify the 4.5s `aten::mm` is stable, not a one-off bad autotune.
- **Exact split** of the 4.5s between the 49 fp8-tier-as-bf16 and the untiered residual:
  enumerate the student's Linear modules, classify each as ml8 / fp8-tier-bf16 /
  untiered-bf16 (one 4B load).
- **VRAM note from #223:** packed indices are now resident per layer (cached, not
  rebuilt-and-freed each forward) — a bounded +~half-the-index-bytes on GPU. R9700 is
  32GB (not 16 as earlier assumed: torch reports cuda:0 = AMD Radeon AI PRO R9700 34.2GB),
  step peaked ~11GB VRAM, so headroom is fine.
- The R9700 is torch **cuda:0**; the RX 6900 XT is cuda:1 (now hosts the E2Rank
  embedding server — moved off host RAM). Pin trainer with `HIP_VISIBLE_DEVICES=0`.

### Jira / KG
Backlog filed as a MAD story (fp8-QAT trainer speedup follow-ups). KG updated
(mneme back up after the embedding migration that caused the all-session 500s).

---

## UPDATE 2026-06-14 — old lever #1 DISPROVEN; fp8 backward substrate finished

### The evening backlog's lever #1 was wrong (measured, not theorized)
"Route frozen fp8-tier + untiered linears to fp8" chased ~50 ms. Two measurements
killed it:
- **Module enumeration** (`ENUMERATE_MODULES=1`): the only large bf16 *Linear
  weight* is `lm_head` (2560×248320) = 99.4% of bf16 Linear params — but **41 ms**
  of time (one efficient GEMM). fp8-tier = 48 tiny SSM `in_proj` (0.6%).
- **Shape-attributed profile** (`record_shapes`, GEMM-by-operand-shape): the 5.16s
  `aten::mm` is **~3.4s ml8 trained-path bf16** (ffn/attn fwd-recompute + backward
  dgrad/wgrad) + **~1.7s SSM fp32** (fla gated-delta) + ~0.05s frozen Linears.

### Root cause: the fp8 backward was a torch placeholder
`Ml8Fp8Fn.backward` computed `dy8` then **threw the fp8 away** — `(dy8.float()*sdy) @ W`
and `.t() @ x` ran **fp32 torch matmuls** on the dense-reconstructed W, landing on
the bad rocBLAS MT16/MT8 tiles. Only the forward ran genuine fp8 (a8w8). The
"completed" #200/#220 fp8 backward was never finished.

### Fix (MAD-290, TDD): both backward GEMMs on fp8 tensor cores
`ml8_backward_kernels.ml8_wgrad → +ml8_backward_gemms(dy, W, x)` runs `dx` and
`dW_raw` via `torch._scaled_mm` (e5m2 grad, e4m3 W/x; per-tensor scale so one
quant serves both contraction axes; contraction-dim padded to 16; operands
normalized to row-major×col-major — a non-contiguous live-graph crash caught by
integration profiling, now regression-tested). Wired into `Ml8Fp8Fn.backward`
replacing the two fp32 `@`. Deployment-faithful: W is fp8 in the gradient too.
29 tests green; centroid-grad cosine vs bf16 STE > 0.95 (dynamics preserved).

### Re-profile (identical 4B micro-step) — validated on hardware
| Metric | step 0 | fp8 backward | |
|---|---|---|---|
| wall | 10,547 ms | **8,199 ms** | 1.29× (2.4× cumulative from 19.8s) |
| `aten::mm` | 5,159 ms / 1195 | **1,562 ms / 795** | ml8 bf16 GEMMs gone |

The ml8 ffn/attn bf16 shapes vanished from `aten::mm`. What remains is **SSM fp32**
(`[7362,1024]²` 748ms, `[4090,512]²` 422ms, `[3272,1024]²` 336ms ≈ **1.5s**) — the
next frontier. lm_head 9–30 ms (noise, as predicted). RAM floor 769 MB.

### Source of truth
MAD-290 (reframed + reparented under MAD-281) holds the consolidated story.
Diagnostics added to `smoke_fp8_qat.py`: `ENUMERATE_MODULES=1`, `record_shapes`
GEMM-by-shape in the `PROFILE_STEPS` block. **Next: SSM fp32 scan (fla
gated-delta, RDNA4 bf16-`fdot2` workaround) — separate subsystem, ~1.5s.**

---

## UPDATE 2026-06-14 (afternoon) — the "SSM 1.5s" was the QuaRot rotation; FWHT built

### Three measurement-driven corrections of the residual ~1.5s `aten::mm`
1. **Not the SSM scan.** fla's `chunk_gated_delta_rule` is **pure Triton** (no
   `torch.matmul`) — ~415ms (`ChunkGatedDeltaRuleFunction` fwd 46 + bwd 369), and
   its fp32 is **deployment-faithful** (ml8.cu recurrence is f32). fla can't even
   consume fp8. So the SSM was never the lever.
2. **Not full-attention / not a Python `torch.matmul`.** A `TRACE_MM=1` tracer
   (patches `torch.matmul`/`mm`/`__matmul__`, one fwd+bwd, groups big matmuls by
   call-site) pinned all 800 big matmuls to **`kronecker_rotation.py:72`** — the
   QuaRot incoherence rotation `Q = H_a ⊗ H_b` applied to ml8 layer inputs
   (`act_replay_student.py:192`).
3. **Precision is a red herring.** Production (`ml8.cu` `ml8_fused_rot_quant_kernel`,
   line 537/882) rotates in **fp32** (FWHT + H_a in registers) then e4m3-quantizes
   — the trainer already matches (fp32 rotate → `quantize_act_per_row`). So fp32 IS
   the production number; bf16/fp8 would *diverge*. The trainer is slow only because
   it uses a **dense `X @ h_b` matmul** where production uses a **fused FWHT**
   (`O(b·log b)` vs `O(b²)`).

### Fix (committed-pending): FWHT rotation, fp32, deployment-faithful
`kronecker_rotation.py`: added `fwht_raw` (butterfly Walsh-Hadamard; `fwht_raw(x)/√n
== x @ sylvester(n)` to ~5e-7). `KroneckerRotation.forward`/`inverse` now do
`h_a.T @ (fwht_raw(X) * 1/√b)` instead of `h_a.T @ X @ h_b` (Sylvester H symmetric →
inverse b-leg is the same FWHT). Same fp32 math, `O(b·log b)`. **TDD: 8 rotation +
17 student/consumer tests green** (1 pre-existing unrelated `test_one_size` fixture
error, not mine). Micro-bench (real shapes): **FWHT 1.4–2.6× fwd+bwd** vs dense,
scaling with b. Mirrors the deployed ml8 fused FWHT prologue.

### RESOLVED 2026-06-14 — FWHT rotation is a WASH at the step wall (measured)
Clean `WALL_ONLY=1 PROFILE_STEPS=3` runs (R9700, frozen arm, identical harness,
profiler bypassed):

| rotation forward | step wall (ms/micro) |
|---|---|
| **FWHT** butterfly | **4232, 4422** (two runs) |
| **dense** `h_a.T @ X @ h_b` (temp revert, apples-to-apples) | **4428** |

**Verdict: the rotation method does not move the step wall.** Dense vs FWHT differ
by ~200 ms — *less than the run-to-run noise between the two FWHT runs (190 ms)*.
The earlier hypothesis (that the residual ~1.5 s was the rotation and FWHT would
claw it back) was **wrong at the wall level**: the rotation is too small a slice of
the full step to matter. The micro-bench win (FWHT 1.4–2.6× **in isolation**,
relerr 5e-7) is real but irrelevant end-to-end. The TRACE_MM "800 big matmuls at
kronecker_rotation.py:72" were real call counts but cheap wall-time / overlapped.

**Kept FWHT anyway** — it is mathematically equivalent (5e-7), strictly not-slower,
and *more deployment-faithful* (mirrors the ml8.cu fused FWHT prologue). Just don't
bill it as the speedup.

**Correction to the trajectory:** the "8.2 s" figure was a **profiler-context**
number, not a clean wall. The real clean wall after the **fp8 backward** is
**~4.3 s** — i.e. the fp8 backward did the entire MAD-290 win (10.6 → 4.3 s);
the rotation work was a faithfulness improvement, not a perf lever.

### FINAL PROFILE 2026-06-14 — the 4.3s step is HOST-BOUND (dispatch wall)
Full `PROFILE_STEPS=3` (dense rotation temp-swapped so the profiler survives —
wall-equivalent to FWHT, just sane op count). Frozen arm, R9700:

```
wall                              4400 ms/micro
CPU blocked-on-GPU (Synchronize)   498 ms  =  11% of wall   → GPU idle ~89%
CPU dispatch (LaunchKernel self)  1744 ms  over 80,128 launches  → THIS is the wall
```

GPU device kernels are all proportionate/healthy — `aten::mm` 687ms, fp8 gemm
(`_gemm_a8w8_blockscale`) 490ms ×400, `aten::copy_` 411ms, SSM
(`ChunkGatedDeltaRuleFunctionBackward`) only 178ms ×24. No runaway kernel, VRAM
stable (peak 11GB), no leak.

**The 80K launches are dominated by thousands of tiny elementwise ops** (~49K of
80K), per-ml8-layer (200 layers):

| op | count | ~/layer |
|---|---|---|
| `aten::copy_` | 10,945 | 55 |
| `_to_copy` | 9,084 | 45 |
| `bitwise_and` | 10,800 | 54 |
| `aten::where` | 6,004 | 30 |
| `bitwise_or` | 4,400 | 22 |
| `vectorized_elementwise` | 4,400 | 22 |

That signature (bitwise_and/or + where + copy, all per-ml8-layer) = the **fp8 quant
/ ml8 index path running as unfused pure-torch tensor ops** instead of one kernel.

**Interpretation / close-out:** the fp8 backward removed the GPU-time bottleneck
(10.6→4.3s) and **flipped the step from GPU-bound to host-bound**. We've hit the
dispatch wall. Below 4.3s the lever is no longer math — it's **launch reduction**
(fuse the fp8-quant elementwise chain into a kernel, and/or CUDA-graph the step).
That is the original MAD-290 "cut 77K launches" goal and a **separate campaign**.

### Cumulative trainer state — MAD-290 perf phase CLOSED at ~4.3s
19.8s → 10.6s (#223/#224) → **~4.3s clean wall (fp8 backward — the real & only win)**.
FWHT rotation kept for faithfulness (wall-neutral). **Next campaign (NEW work):**
the host-bound 80K-launch dispatch wall — fuse the per-ml8-layer fp8-quant/index
elementwise ops (bitwise_and/or + where + copy, ~49K launches) into a kernel; CUDA
graphs as a fallback. Separate ticket still worth filing: "fp16 rotation in the
`ml8.cu` kernel" — a real *inference* micro-opt, needs a PPL gate.
