# DeepSeek V4 Flash — Paged Decode Acceleration Levers

**Date:** 2026-07-08
**Target:** single-stream (`--parallel 1`) interactive decode speed for DS4 Flash on mad-lab-main. Not multi-user throughput.
**Branch:** `feat/wp-attention-island`

## The bottleneck (measured, not assumed)

DS4 Flash: 284B total / 13B active per token, 256 routed experts (6 used + 1 shared), 43 layers, Q8_K_XL GGUF ~151 GB. Experts don't fit VRAM → demand-paged from NVMe.

The GPU spends most of each token **idle, waiting for experts to arrive from NVMe**: `page_in_sync`, prefetch hit-rate **0.03%**. This is **not** a bandwidth wall (single-card hits 1.038 t/s, so ≥1 t/s is reachable through the same NVMe). It is **un-overlapped I/O latency plus no forward knowledge of which experts are coming** — every cache miss is a synchronous stall. Moving *compute* between cards never touched this; the whole game is cutting/hiding the paging.

Measured decode: single-card dense-resident **1.038 t/s** (best); cross-device attention/expert split **0.48–0.65 t/s** (loses — see hardware below).

## Hardware constraints (these shape every lever)

- **R9700** (ROCm0, paging device): PCIe 4.0 x16, CPU-direct, **28.3 GB/s**, SAM active. → NVMe→VRAM (P2P) and RAM→VRAM are both fast. This is the primary target for everything paging-related.
- **6900XT** (ROCm1, resident/attention): Razer Core X eGPU over **TB3, ~2.7 GB/s**, and **NO P2P to the R9700** (X570 root complex blocks it). → any cross-device tensor crossing is **host-staged over the slow TB3 link**. Architectures with *few* crossings win; per-layer ping-pong (86 crossings/token) is what sank the naive cross-device split.
- **15 GB system RAM (~11 GB free).** A full RAM tier for 137 GB of experts is impossible; a small *selective* RAM tier (~4–6 GB) is viable and useful.

## Target memory hierarchy

```
VRAM  (MTP-pinned, high-confidence experts)
  ↑ 28.3 GB/s PCIe
RAM   (plausible / recently-hot victim experts)   ~4–6 GB selective tier
  ↑ NVMe read
NVMe  (cold experts, P2P direct-to-VRAM)
```

All three tiers fed by the **MTP draft signal**: sure-things pinned in VRAM, plausible staged to RAM, cold left on NVMe. The RAM tier is the safety net for imperfect MTP acceptance — a wrong/uncertain prediction resolves at RAM speed instead of NVMe speed.

## Levers by implementation status

### A. Done / config-only — usable right now

- **Bigger VRAM expert cache** — `--weight-paging-slots`. VRAM was freed by offloading dense (attention-island: `token_embd`/`output` → paging card). Just tune slots to available VRAM. *Confirmed helps: page-ins/token 587→279 going 2000→5000 slots.*
- **P2P / dma_buf paging (NVMe→R9700)** — implemented and is the fast path (the 1.038 baseline used it); R9700's x16 slot makes it effective. (Inter-GPU P2P is hardware-blocked but irrelevant to paging.)
- **io_uring queue-depth infra** (`WP_IOURING_DEPTH`) + the QD>4 hang fix — done and stable.

### B. Little additional work — enabling / wiring existing code

- **Hot-expert retention (backward)** — the pool already has LRU + a hot-count. Needs the frequency-biased eviction policy enabled/tuned so recurring experts survive eviction. (Forward-looking pin is in C.)
- **RAM victim tier** — `HostTier` (LRU + `WP_HOST_BUDGET_BYTES`) **already exists but is disabled** (this box never had RAM to spare). Enable with a ~4–6 GB budget; confirm/allow on-evict population (demote hot victims to RAM). PCIe 28.3 GB/s makes a RAM hit ~10× cheaper than an NVMe miss. (MTP pre-staging into it is in C.)
- **Queue-depth re-evaluation in the MTP-batched regime** — infra is done; only a re-measure. Rationale: single-token decode needs ~6 experts (no parallelism, so depth-8 ≈ depth-4, QD was correctly ruled out). MTP *verify* needs the union of experts over K tokens (≈K×6 pages demanded at once) → now there's parallelism for higher QD to saturate the NVMe. QD flips from irrelevant to potentially significant, *only* once MTP batching exists.

### C. New implementations

- **MTP head — the spine (LINCHPIN).** Drive DS4 Flash's multi-token-prediction / draft head for speculative decode. Delivers two compounding wins from one draft: (1) verify K tokens per forward pass → pay the expert-paging cost once per K tokens (amortize); (2) forward knowledge of which experts the next tokens need (the signal every other new lever consumes). Everything below hangs off its acceptance rate. **Open questions:** is `output_hc_*` the draft head (it's a small projection next to `output.weight`, not the `nextn.*` tensors llama.cpp's speculative path expects)? Can `--spec-type draft-mtp` drive it, or does it need wiring? Does the draft run expert-free (cheap on the resident card) or does it also need experts?
- **MTP-driven prefetch** — feed MTP-predicted experts into the existing `PrefetchScheduler` (which works mechanically but is blind → 0.03%). Load predicted experts ahead of need → overlap NVMe reads with compute so the GPU stops stalling.
- **MTP-driven forward retention** — pin experts the draft says recur across the next K tokens (forward-looking) instead of, or on top of, the backward hot-count.
- **MTP-confidence RAM pre-staging** — graded placement using the draft's confidence: high → VRAM pin, medium → RAM tier, low → leave on NVMe. This is what makes the whole hierarchy robust to <100% MTP acceptance.
- **Cross-device draft/verify** — repurpose the attention-island split: draft cheap on the resident 6900XT, verify the K tokens batched on the paging R9700. The placement machinery (dense resident, experts paged) is already built; the draft/verify orchestration is new. Key benefit given the hardware: far fewer TB3 crossings than the per-layer split, so the slow/no-P2P TB3 link stops dominating.

## Deferred / dead

- **Lower-bit expert quant** — deferred. DS4 Flash is already QAT'd for a 4/8-bit mix; re-quantizing isn't our fight now. Revisit when ml8 is ready.
- **Faster / striped NVMe & full RAM tier** — dead on this box. 15 GB RAM can't buffer 137 GB of experts; the only other drive is a slow HDD. NVMe is the floor.

## Ruled out (measured)

- **QD as a single-token bottleneck** — depth-8 ≈ depth-4 (revived only for the MTP-batched regime).
- **Inter-GPU P2P** — hardware-blocked (X570 root complex).
- **Per-layer attention/expert cross-device split for single-stream** — loses to single-card; the handoff tax (86 host-staged TB3 crossings/token) exceeds the bigger-cache gain. Repurposed into draft/verify.

## The linchpin

DS4 Flash's MTP head viability — acceptance rate, and how concentrated expert demand is over the next K tokens. Every new lever (prefetch, forward retention, RAM pre-staging, draft/verify) compounds off that one draft signal. **Next action: investigate the head.**
