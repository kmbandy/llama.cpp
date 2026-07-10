# DS4 Flash — Strategy, Compute-Ceiling Reopening & Morning Start Plan (2026-07-09)

Companion to: 2026-07-09-ds4flash-decode-bandwidth-state.md, -dflash-predictive-streaming-plan.md,
-weight-paging-review-codex.md. Branch feat/wp-dflash-ds4. All hooks/P0 fixes UNCOMMITTED (read-only, env-gated).

## THE UNIFYING INSIGHT (kmbandy, end of session) — start here
Once kernel overhead is removed, we are NEVER compute-bound — ALWAYS I/O-bound.
Why: "compute" at batch-1 decode = reading the RESIDENT active weights from VRAM (~640 GB/s); "I/O" =
reading COLD experts from NVMe (~2.6 GB/s). NVMe << VRAM, so I/O dominates the moment compute overhead
is stripped. => The entire decode-speed problem reduces to MINIMIZING I/O TIME. Compute de-overhead is
necessary only to get compute OFF the critical path (below I/O); after that, every remaining lever is I/O:
  (a) raise NVMe bandwidth (software 2.6->6, then RAID-0 Gen4 ~12),
  (b) reduce cold bytes (RAM victim tier + prediction so recurring/predicted experts don't hit NVMe),
  (c) overlap I/O under compute (prediction fills the idle SSD during the compute window).

## 2026-07-10 UPDATE — EXPERT CONCENTRATION MEASURED; CAPACITY IS THE CONSTRAINT (supersedes the priority order at the bottom)

### 2026-07-10 PM CORRECTION — "O_DIRECT DEAD" WAS WRONG; O_DIRECT IS THE BANDWIDTH LEVER
Overturns the "O_DIRECT = DEAD" claims below (MEASURED GROUND TRUTH + the O_DIRECT-amplification lines). Clean standalone benchmark on the UNCOMPRESSED / NOCOW copy (random 4MB O_DIRECT, device bytes via diskstats):
- O_DIRECT per-read amplification = 1.00x (NOT 2.3-2.8x). NOCOW concurrency scaling: 1t 3.87, 2t 5.25, 4t 5.84, 8t 5.50, 12t 5.76 GB/s — saturates ~5.8 GB/s by QD4, vs the pager's current 2.6 (BUFFERED) = ~2.2x.
- NOCOW (chattr +C = disables btrfs data checksums) is ~38% faster than plain uncompressed single-thread; ~7% at saturation. Plain uncompressed 8t = 5.45.
- The "2.79x amplification" that got O_DIRECT killed was EXPERT RE-READ CHURN (capacity: hold ~20% -> ~63% churn) + reads of the COMPRESSED Downloads copy — MISATTRIBUTED to btrfs O_DIRECT. Per-read O_DIRECT is clean (1.0x).
- The pager's own O_DIRECT path got 1.39 t/s ONLY because it always read the checksummed/compressed copy, never a NOCOW file (kmbandy).
IMPLEMENTATION (lever #1, decided 2026-07-10 PM): model becomes a NOCOW file (chattr +C, at /home/kmbandy/models/ds4-nocow/) + pager uses its O_DIRECT host-bounce path (WP_ENSURE_BATCH_HOST=1, WP_ODIRECT_READ_WORKERS>=4) -> target ~5.8 GB/s (~2.2x on every page-in). Compounds with the capacity levers (which cut the churn / number of page-ins). /home is btrfs (compress=zstd:1) — there is NO ext4/xfs partition; NOCOW is how we get clean O_DIRECT without a filesystem migration.

Trigger: reviewed kacper-daftcode/vLLM-Moet (same two models: GLM-5.2 753B, DS4-Flash 159B) + measured our own routing concentration on a length-corrected 1026-token deterministic DS4 trace (~/wp_logs/accounting/concentration.py on routing_capture.bin).

### vLLM-Moet — what ports and what doesn't
- Their win is a CAPACITY story via 2-bit sign-symmetric quant: the whole base fits pinned host RAM (51.6 GB/s), GPU holds an N-GiB hot-expert cache. Kernels are Blackwell SM120 SASS (QMMA fp4) — NOT portable to RDNA4 (no fp4 matrix cores). We don't quantize further, so we cannot copy the capacity trick directly.
- PORTABLE ideas: (a) MISS-TOLERANCE knob (zero a missing expert's contribution, bump an in-graph miss counter, batch the H2D correction, skip the re-run when <=k of ~600 routings miss); (b) PASSIVE convergence (background promote/evict, no predictor); (c) their concentration framing.
- Their "19%->96% (DS4)" leans on CONTINUOUS BATCHING (~600 routings/step ~= 100 tokens) reinforcing hot experts + a long live trace. Our single-stream (--parallel 1) regime is structurally flatter.

### Our measured concentration (1026-tok trace, single-stream, temp0)
- Pool = 11,008 (layer,expert) slots x 13.37 MB = 147 GB. R9700 holds 28.9 GB ~= 20% of the pool.
- STATIC: 40% of the pool (~4,449 slots ~= 59 GB) covers 96% of routings; per-layer ~41% of 256 for 96%.
- DEPTH GRADIENT (robust, actionable): layer 0 near-uniform (156/256 for 96%), layer 42 concentrated (95/256) — deep layers concentrate ~1.6x harder.
- PASSIVE LRU hit-rate (length-corrected): 19%->70%, 30%->83%, 51%->96.7%, 75%->97.1%. (The earlier 130-tok trace's "84% plateau" was cold-start bias — WRONG; ignore it.)
- KEY REFRAME: the binding constraint is VRAM CAPACITY, not routing predictability. We hold ~20% -> ~70% hit (~78 misses/tok); the cache converges to 96.7% at 51% (~8.6 misses/tok) but that is ~2x our VRAM. The cache demonstrably WORKS once big enough — no predictor required to reach it.
- CAVEAT: single deterministic prompt; varied overnight-agent workloads sit somewhat lower (cross-prompt hot-set shift). A varied-prompt trace is the honest next measurement.

### Consequences for the levers
- CAPACITY-EXTENSION is now the high-leverage path (elevated): RAM victim tier + depth-aware residency (spend the 28.9 GB where concentration is highest; pin deep layers hard, let shallow near-uniform layers page). Every point of effective coverage added pays steeply in the 30->51% band (83->97%).
- PREDICTION (DFlash adapter) is DEMOTED to escape-hatch: only if capacity-extension cannot reach effective ~51%.
- MISS-TOLERANCE: build the non-blocking substrate (k=0 = always-correct, just instrument the miss counter); open k>0 only once the cache is hot (at 20%/70% it would zero ~30% of experts — too aggressive). Bias tolerated misses to shallow near-uniform layers. Correction = targeted FFN recompute (NO graphs needed).

### Decisions logged today
- NO fp8/safetensors: routed experts are ALREADY MXFP4 (4-bit) on disk -> native fp8/fp4 gives ZERO I/O reduction. fp8 is a prefill-compute/quality play only, pursuable LATER from the existing GGUF via the ml8 fp8-LUT kernel.
- Model relocated to /home/kmbandy/models/ds4-uncompressed/ (uncompressed reads ~25% faster: 1.78 vs 1.42 GB/s single-thread buffered; btrfs compressed the incompressible weights ~0 but still paid the decompress path). Downloads copy DELETED, 151 GB reclaimed. Buffered reads show NO amplification either copy (~0.95x); O_DIRECT amplification (2.3-2.8x) confirmed DEAD.
- GRAPHS are force-disabled in the paged path BY DESIGN (not broken; MAD-288 fixed them for KV-attention). Filed MAD-344 to complete it. Graphs = kernel-launch overhead only (~single-digit %), NOT a miss-handling mechanism — off the critical path.
- BANDWIDTH gap 2.6->6 is QUEUE-DEPTH/CONCURRENCY, not compression (single-thread buffered ~1.4-1.8; pager 2.6 @ QD16; SSD ceiling 6.2 @ QD>=6). ensure_batch under-drives the SSD's parallelism.

### REWEIGHTED PRIORITY ORDER (supersedes the numbered list at the bottom)
1. I/O bandwidth 2.6->6 (unchanged #1): find why ensure_batch delivers 2.6 of 6 at QD16 -> drive higher effective queue depth. Uncompressed base already set.
2. Depth-aware residency (NEW, free): reallocate the 28.9 GB toward concentrated deep layers -> higher effective coverage at fixed VRAM.
3. RAM victim tier (elevated): capacity extension now PROVEN to pay (30->51% = 83->97% hit).
4. Decode fused grouped-GEMV (unchanged rationale): compute permanently below I/O.
5. DFlash->routing adapter — DEMOTED to escape-hatch (only if 1-3 cannot reach effective ~51%).
6. Miss-tolerance substrate — build mechanism at k=0 (non-blocking infra + miss instrumentation); open k>0 only once cache hot.
7. Later: varied-prompt concentration trace, overnight-agent test, NVMe RAID-0, ml8/DSWS fp8 prefill kernel, Ornith eval.

## MEASURED GROUND TRUTH (this session)
- Best transport: p2p BAR io_uring 1.6 t/s, 1:1 device bytes. O_DIRECT = DEAD (btrfs below-app 2.3x
  amplification, app issues clean 4.45MB reads, device delivers 2.3x). host buffered 1.38. QD 16=32=64 INERT.
- Per token: total 626ms = ~268ms "compute" + ~358ms exposed I/O-wait (predictors off, ~serial).
- SSD: 2.6 GB/s ACTIVE (ensure_batch_gb_s) but only ~40% duty (io_effective 1.06) -> idle during compute.
- Per pass: 258 routed groups (6 used x 43 layers) x 13.37MB (MXFP4 3x4.456MB sisters) = 3.45 GB/pass.
  Pool R9700 32GB = 6500 slots = 28.9GB = ~8 passes. ~63% churn/token -> ~1.02 GB/token cold NVMe.

## COMPUTE CEILING WAS WRONG — REOPENED (MAD-299 lesson, again)
My "3.7 t/s compute-bound ceiling" was a SOFTWARE-overhead number, not hardware.
- Hardware floor: ~6GB active weights / ~640 GB/s VRAM = ~10ms/token memory-bound GEMV floor. 268ms = ~25x.
- GPU-busy measurement (rocm-smi 10Hz over decode): card0(R9700 experts) mean 34% / p90 62% / max 93%;
  card1(6900XT dense) mean 17%. => mostly idle over the window (I/O-bound), but ~80% busy DURING compute
  bursts => the 268ms is GPU-bound-but-INEFFICIENT: 774 tiny batch-1 GEMV ops (258 experts x 3 sisters),
  ~215ms GPU work for a ~10-60ms bandwidth floor. NOT CPU overhead. NOT TB3 (kmbandy+Grok measured <20ms).
- Fix = FUSION (one grouped-GEMV that saturates VRAM bandwidth + few launches), NOT peak TFLOPS.

## KERNEL DATATYPE (checked in ggml + AITER recon)
- MoE experts run int8 MMVQ (decode) / int8 MMQ (prefill): vec_dot_mxfp4_q8_1, LUT-decodes fp4->int8,
  int8 DP4A. NOT dequant-to-fp16 (good), but NOT native fp8/fp4.
- RDNA4/gfx1201 has NO fp4 matrix cores. WMMA = fp16/bf16/fp8/int8/int4. MXFP4 WMMA = CDNA4/MI350X only.
  ggml native-fp4 MMQ path is blackwell-gated (NVIDIA-only) -> falls to int8 on RDNA4.
- TRANSLATION TARGET = fp8, NOT int8: fp4->fp8 is LOSSLESS (e2m1->e4m3); fp8 activations hold the
  per-token outliers (kurtosis 121-183) that int8 clips; hardware-aligned (fp8 WMMA). int8/fp8 split is FALSE.
- fp4->int4 is a LOSSY requant (uniform vs non-uniform) -> AITER standard a8w4/int4 kernels OFF the table
  (no more quantizing). AITER mxfp4 kernels (gemm_afp4wfp4, moe a4w4/a8w4) are fp4-native = CDNA-only, no RDNA4.

## THE FUSED fp8 MoE KERNEL — mostly EXISTS but compute-weak
- kernels/moe_op_gemm_ml8.py + gemm_ml8.py have WEIGHT_FORMAT tl.constexpr branch: =1 does
  4-bit-index -> per-K-group fp8 centroid LUT -> fp8 WMMA MoE, BIT-EXACT on R9700 (MAD-223). Adding a
  WEIGHT_FORMAT=2 mxfp4 branch (e2m1 nibble x per-32-block scale -> e4m3) is SMALL and simpler than the LUT.
- BUT the ml8 kernel is COMPUTE-INEFFICIENT: ~165 TF square (52% of 307 TF ceiling), ~113 TF on real ml8
  dims; wall = VALU issue port (~31 non-WMMA issues per 32 WMMAs). Worse on skinny Qwen MoE shapes (N=512,
  small M) it was never tuned for (tuned on square/K=16384). DSWS is the effort to break this; NOT done.
- REGIME SPLIT (key): ml8/DSWS = COMPUTE-BOUND prefill GEMM. DS4 DECODE = MEMORY-BOUND -> wants a DIFFERENT,
  more tractable fused grouped-GEMV (saturate bandwidth + fewer launches; mediocre peak-TFLOPS is FINE).
  Decode grouped-GEMV for paged experts is NOT built ("decode M=16 deferred"; hand-HIP job).
- Chokepoint for the Triton path: MAD-223 Phase C (Triton-AOT -> llama.cpp, aiter_triton_aot.a / Registry /
  hipModuleLaunchKernel) was ~80% with a launch SIGSEGV; MoE wrapper (C.3) pending. That segfault is the real
  blocker between the vendored kernel and running in the inference loop.

## DFlash PREDICTION (earlier this session)
- Cross-layer routing signal REAL in TARGET residual: 0.64@top6, 0.82@top16 (1 layer ahead), strong mid/late.
- DFlash DIRECT projection through target routers = NEGATIVE (~random 0.017). DFlash hidden is its own space.
  Fix = small learned ADAPTER (DFlash inp_g[4096] -> target routing, linear then MLP; per-tap [3,13,23,32,42]).
  Fine-tune DFlash = last resort only if even MLP adapter fails.
- 3-tier residency: VRAM pool + RAM victim (4-5GB slow-drain, ~28GB/s, catches mispredicts) + NVMe.
  Confidence-graded progressive prefetch: priority ~ router_prob x token_acceptance x cross_pass_frequency.
  DFlash lookahead: block_size 8, n_max 4 in runs, 76% accept, ~4 reliable tokens (~1s lead).

## STRATEGY / MODEL CHOICE
- REAL GOAL: a test-gated, mneme-continuous OVERNIGHT AGENT that executes handoffs Claude preps (last
  20-30min of a session), while user sleeps / Claude usage resets. DS4 is the smaller end capable of that
  caliber of work; mneme gives night-to-night continuity; 1M ctx = headroom. Handoff quality lowers the
  knowledge bar (execute, not invent); risk = self-recovery from surprises the handoff didn't anticipate;
  harness (tests-as-guardrails, graceful bail, progress ledger -> mneme resume) is what makes it trustworthy.
- MODELS (paging t/s governed by ACTIVE params, not total size):
  * DS4 Flash: fastest paging, tech-proving, ~1.6->3+ t/s. Already loaded.
  * Ornith 397B A17B (Qwen35MoE): near-frontier, ~2.2x DS4 I/O -> ~2-3 t/s, Q4_K_M 242GB (fits after cleanup),
    coding bench ~ recent Sonnet / near Claude prev iters. BEST interactive upgrade.
  * GLM 5.2 754B A40B (glm-dsa): top quality but ~4x DS4 I/O -> ~0.5-1.3 t/s. TOKEN MATH: ~22k out/night <
    one session's output -> NOT a "second you". Overnight-oracle only; needs storage.
- IP is NOT a concern (provider TOS covers it) -> cloud (Claude) is fine for deep interactive work; local
  model is for OVERNIGHT CONTINUATION, not IP.
- HARDWARE: not VRAM (frontier VRAM = datacenter). Lever is RAM/mem-bandwidth (high-RAM EPYC/TR, ~5-8 t/s
  frontier MoE) BUT RAM is expensive now (~$1-1.5k/128GB DDR4). CHEAP near-term = NVMe: software gap (free)
  + RAID-0 two Gen4 (~$200, AM4=PCIe4 so RAID not single Gen5) -> ~12 GB/s. hipfire is DROPPED; all in llama.cpp fork.

## MORNING START — PRIORITY ORDER (given: always I/O-bound)
1. **I/O bandwidth software gap 2.6->6 GB/s** (free, biggest tractable lever, applies to every model). This is
   THE lever since we're always I/O-bound. Instrument ensure_batch to find why it delivers 2.6 of the 6 the SSD does.
2. **RAM victim tier** (WP_HOST_BUDGET_BYTES demote-on-evict + check-on-miss; measured -11.5% page_ins;
   verify free RAM before pinning 4-5GB on the 15GB box). Model-agnostic, cheap.
3. **Decode fused grouped-GEMV** (memory-bound, NOT DSWS-gated): kills the 268ms compute overhead ->
   ~40ms; if compute is serial with I/O that alone is ~626->~400ms = ~2.5 t/s BEFORE bandwidth work.
   Net-new hand-HIP kernel consuming paged expert pointers. Gets compute permanently below I/O.
4. **DFlash->routing ADAPTER** (the prediction make-or-break): generate (DFlash inp_g, target routing) pairs
   from captures, train linear/MLP adapter, measure cold-page recall at lead. Enables overlap (fill idle SSD).
5. **Overnight-agent workflow TEST on DS4-as-is**: prep a real test-gated handoff for a paging-work slice,
   run overnight, isolate model-vs-harness bottleneck. $0, validates the whole direction.
6. Later: NVMe RAID-0 (bandwidth), ml8/DSWS fp8 prefill kernel (agent prefill-heavy), Ornith eval.

## ARTIFACTS / HARNESS (~/wp_logs/accounting/)
capture-run.sh (target routing), capture-spec-run.sh (DFlash hidden), matrix-run.sh, decode-gpubusy.sh,
analyze-routing.py, dflash-align.py, analyze-io.py, routing_capture.bin, dflash_capture.bin.
Uncommitted read-only env-gated hooks: wp-eval-cb.cpp WP_CAPTURE_ROUTING; speculative.cpp WP_CAPTURE_DFLASH
(DFlash class ::process, inp_g, ~line 1108; backups .bak). P0 fixes uncommitted: #5 DFlash-enable unified,
#8 no silent wrong-expert substitution. NOTHING committed (no-git rule) — decide commit in morning.

---

## 2026-07-10 SESSION LOG — PREFETCH BUILD LAUNCHED + FULL FORWARD ROADMAP
Detailed notes of everything decided this session, so nothing is lost across compaction. Read with the 2026-07-10 UPDATE + PM CORRECTION blocks above.

### Decisions made this session (chronological)
1. **vLLM-Moet (kacper-daftcode) reviewed** — same 2 models (GLM-5.2 753B, DS4-Flash 159B). Their win = CAPACITY via 2-bit quant (base in RAM); kernels Blackwell SM120 SASS = NOT portable. Portable ideas: miss-tolerance knob, passive convergence, concentration framing. Their 19%->96% leans on continuous BATCHING (~100 tok/step); our single-stream is flatter.
2. **Concentration measured** (1026-tok trace): binding constraint is VRAM CAPACITY not predictability. Cache 96.7% hit at 51% coverage; we hold ~20% (~70%). Depth gradient real (L0 156/256, L42 95/256 for 96%). Full detail in the UPDATE block.
3. **Graphs = tech-debt side-quest, NOT critical path.** Force-disabled BY DESIGN in the paged path (wp-pager.cpp forces GGML_CUDA_DISABLE_GRAPHS=1 unless WP_HIP_GRAPHS=1); MAD-288 fixed them for KV-attention only. Filed **MAD-344** to complete (two-incompatible-designs root cause documented there). Graphs = kernel-launch overhead only (~single-digit %), NOT a miss-handling mechanism.
4. **fp8/safetensors = NO** (this session). Routed experts already MXFP4 (4-bit) on disk -> native fp8/fp4 gives ZERO I/O reduction. fp8 is a prefill-compute/quality play only, pursuable later from the existing GGUF via the ml8 fp8-LUT kernel.
5. **Model housekeeping.** Deleted the compressed ~/Downloads copy (151 GB reclaimed). Model is now NOCOW at /home/kmbandy/models/ds4-nocow/ (chattr +C, no btrfs checksums). /home is btrfs compress=zstd:1; NO ext4/xfs partition exists. Harness .sh/.py repointed to ds4-nocow.
6. **Bandwidth lever, resolved:**
   - **NOCOW buffered p2p = 1.81 t/s (+12% over 1.61 compressed) — BANKED.** Checksum removal cut ensure_batch wait time.
   - **O_DIRECT drop-in = RULED OUT.** Pager reads in demand-driven ~6-page bursts with a per-batch barrier -> the disk queue DRAINS between batches. O_DIRECT (1.37 t/s, 1.61 GB/s) is WORSE than buffered (2.6) because it loses kernel readahead in the bursty pattern. Standalone O_DIRECT hits ~5.8 GB/s only at SUSTAINED QD4.
   - **CONVERGENCE:** the 2.6->5.8 headroom needs a SUSTAINED disk queue = prefetch the next work AHEAD of demand during compute. The bandwidth lever and the prefetch lever are the SAME lever. This also fixes the ~40% idle duty cycle.

### The prefetch build (current work)
- **Approach 1** (chosen): host-side cross-layer ROUTER predictor + existing PrefetchScheduler + a new pool speculative eviction tier. Zero GPU-graph changes.
- **Predictor:** run layer L+k's ffn_gate_inp router on layer L's LIVE residual -> top-M experts (measured 0.64@top6 / 0.82@top16, NO training). This is CROSS-LAYER (works). CROSS-TOKEN prefetch is ~0 locality (dead).
- **DFlash disposition (important):** cross-layer router REPLACES the DFlash->routing adapter for PREFETCH (adapter shelved — unbuilt, projects at chance). DFlash-as-SPECULATIVE-DECODE STAYS ENABLED as a co-resident lever on the 6900XT (ROCm1) — it does NOT compete with R9700 paging, and it COMPOUNDS: draft-verification passes need the UNION of the draft tokens' experts = bigger ensure_batch = better queue saturation. Not parked.
- **Footgun fix:** speculative eviction tier — prefetched pages evicted FIRST, NEVER evict a pinned working-set page, promote-on-demand-hit. Cap WP_PREFETCH_MAX_SLOTS = the VRAM-split knob (a redistribution WITHIN the 6500-slot pool, no new VRAM).
- **SPEC (approved):** docs/superpowers/specs/2026-07-10-cross-layer-prefetch-overlap-design.md
- **PLAN (approved):** docs/superpowers/plans/2026-07-10-cross-layer-prefetch-overlap.md — 6 TDD tasks. Execution = INLINE by Claude via ssh, task-by-task with checkpoints (Codex was the alt).
- **Tasks:** T1 RouterPredictor (new wp-router-predictor.{h,cpp}, CPU, unit-tested) -> T2 (block,expert)->sister-pages reverse index -> T3 pool speculative eviction tier (wp-pool) -> T4 eval-cb wiring (lazy ffn_gate_inp capture + predict + submit, gated WP_PREFETCH_XLAYER, DFlash union) -> T5 config+stats (WP_PREFETCH_XLAYER/LOOKAHEAD_K/TOPK/MAX_SLOTS + online recall + speculative_evicted_unused) -> T6 GPU sweep (user-gated: finds K/M/MAX_SLOTS + the VRAM split). RESUME AT T1.

### STEPS AFTER PREFETCH (the forward roadmap, in order)
1. **VRAM-split decision** — falls out of the T6 sweep: the MAX_SLOTS that maximizes t/s tells us how much of the 28.9 GB is pinned working set vs prefetch reservation. (This is WHY prefetch is built before residency — kmbandy's sequencing.)
2. **Depth-aware residency** — pin the concentrated DEEP-layer hot sets harder (L42 ~37% of experts for 96%), let shallow near-uniform layers (L0 ~61%) page. Free reallocation of the pinned budget -> higher effective coverage at fixed VRAM.
3. **RAM victim tier** — extend fast-memory coverage past VRAM at ~28 GB/s (measured -11.5% NVMe page_ins). Now PROVEN to pay steeply (cache 30->51% = 83->97% hit). WP_HOST_BUDGET_BYTES demote-on-evict + check-on-miss; verify free RAM first.
4. **DFlash spec-decode + prefetch combined** — measure the compounded t/s (bigger batches + saturated queue + fewer forward passes).
5. **Miss-tolerance substrate** (vLLM-Moet import) — build the non-blocking mechanism at k=0 (instrument the miss counter, always-correct), open k>0 only once the cache is hot; bias tolerated misses to shallow near-uniform layers. Correction = targeted FFN recompute (NO graphs needed).
6. **Overnight-agent workflow test on DS4-as-is** — $0, isolates model-vs-harness bottleneck; validates the whole direction.
7. **Later:** NVMe RAID-0 (bandwidth compounds), ml8/DSWS fp8 prefill kernel, Ornith 397B-A17B eval (best interactive upgrade, ~2.2x DS4 I/O), complete graphs (MAD-344).

### Uncommitted state (nothing git-committed, per no-git rule)
- Spec + plan + this doc update — written, uncommitted.
- Harness repointed to ds4-nocow; odirect-validate.sh + buffered-nocow-validate.sh staged.
- All prior session hooks/P0 fixes still uncommitted on feat/wp-dflash-ds4.
- MAD-344 filed (graphs). KG: nocow/O_DIRECT fact 92428344; earlier session anchors b2ebb724 / a346cdb3.
