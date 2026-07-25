# Distributed Weight Paging — direction and work items

**Date:** 2026-07-24
**Status:** direction agreed, nothing below is built unless marked DONE.
**Decision log:** `repo__kmbandy__llama.cpp` decision `cbb7417d`.
**Related:** `docs/dev/2026-07-24-rpc-mesh-weight-pager-integration.md`,
`docs/dev/2026-07-21-tiered-dual-gpu-expert-feeding-design.md`,
`docs/superpowers/specs/2026-07-24-ffn-island-device-role-design.md`.

---

## The direction

Split the model by **contiguous layer range** across both machines. Each machine
runs a full llama.cpp instance that owns its own shard and pages it from its own
NVMe, through its own RAM tier, into its own VRAM. The two are joined at a single
layer boundary.

Why this and not the current RPC arrangement: `ggml-rpc-server` has no model,
loader, or catalog, so it can only hold *resident* layer homes. All paging happens
on main, and the second machine contributes VRAM but no NVMe→RAM→VRAM tier — we
forfeit an entire independent prefetch path. It also cannot reach GLM-5.2 (239 GB)
on ~65 GB of aggregate VRAM.

A contiguous split keeps the interconnect trivial: **one ~12–16 KB activation
crossing per token per direction** at 0.5 ms RTT, and **zero cross-machine expert
traffic**. It also lets 2026 prefetch its own layers' experts while main computes
its own — two independent I/O paths working simultaneously, which is the
throughput lever that actually exists while decode is storage-bound.

**Split by hot storage (VRAM + RAM), not VRAM alone**, because what you equalize
is per-layer stall time, not layer count:

| machine | VRAM | RAM tier | hot storage | share |
|---|---|---|---|---|
| mad-lab-main | 48 GB (R9700 32 + 6900XT 16) | 8 GB | 56 GB | ~70% |
| mad-lab-2026 | 16 GB (1070 8 + RX480 8) | 8 GB | 24 GB | ~30% |

≈ 13 of 44 DS4 layers on 2026 → ~45 GB shard there; ~72 GB for GLM-5.2 at IQ2.

---

## P0 — Pager read concurrency  ← do this first

The single largest measured win, the smallest build, and it benefits both sides of
the split whether or not the split ever ships.

- [ ] **Issue a MoE op's N active-expert reads concurrently** so the io_uring ring
      actually fills (QD1 → QD N). Work lives in the `wp-eval-cb.cpp` Step-2 ensure
      loop, the "fire async prefetch for every active expert" pass, and
      `WeightPager::ensure` / `page_in_sync_`.
- [ ] **Fix the QD instrument first.** `ensure_batch_n_sub_sum` currently folds
      HostTier hits into the submission count and `ensure_batch_max_n` uses
      `total_ok` — so the queue-depth counter inflates exactly when we start
      measuring queue depth. Keep `n_sub` pure and report host hits separately.
- [ ] Re-measure against `iostat -x` and `ensure_batch_gb_s`.

**Evidence (2026-07-24, pager's exact 4456448 B page, O_DIRECT, random offsets,
fresh files):**

| drive | QD1 | QD4 | QD16 |
|---|---|---|---|
| main WD_BLACK SN850X 1000GB | 0.74–0.91 GB/s | 2.38–2.62 | 2.84–2.95 |
| 2026 WD Black SN750 250GB (WDS250G3X0C) | 2.13–2.20 GB/s | 2.86–2.91 | 2.82–2.89 |

The pager achieves ~0.8 GB/s in production — i.e. it is effectively at **queue
depth 1** despite a 16-deep ring. Same drive does 2.84–2.95 GB/s at QD16, so this
is 3–4× left on the table **in software, not hardware**.

Note the SN750 is *equal at depth and ~2.5× faster at QD1* — the regime the pager
actually occupies. Storage is no reason to give 2026 a smaller share.

*Probe gotcha:* the measurement script uses fixed per-thread seeds, so re-running
it on the same file re-reads the same offsets and the drive cache serves them
(produced a physically impossible 17 GB/s). Always point it at a fresh file.

---

## P1 — Correctness fixes (small; fold in alongside P0)

- [ ] **O_DIRECT read past end-of-shard → EIO.** The padded read of each shard's
      last page overruns EOF (shard 2/3 by 416 B, shard 1 by 128 B; read size is
      page 4456448 + 512). Fires **3× per run** in every run measured, masked by the
      sync fallback. Not an alignment problem — the device is 512/512 and all reads
      are 512-aligned. Clamp the read at EOF and handle the tail.
- [ ] **RPC `supports_op` forwarding.** The stub always returns `true`, so the
      scheduler hands ops to remotes that cannot execute them. Cherry-pick the
      upstream PR ("query remote backend op support", fixes #24177). Today this
      **silently corrupts output** on the Vulkan card — it only surfaced because the
      chat parser choked; a raw completion path would return fluent garbage.
- [ ] **(Optional) Vulkan `SINKHORN_NORM`.** DS4 calls `ggml_sinkhorn_norm`, which
      has CUDA/HIP but *no* Vulkan implementation — this is why the RX 480 produces
      garbage. Without the kernel, `supports_op` forwarding only makes the card
      *safe*, not *useful*. (`ML8_*` also lacks Vulkan but DS4 doesn't use it;
      `PAGED_ATTN_MT` / `TURBO_WHT` have partial Vulkan from the SP1 work.)
      Any Vulkan build must go through the capped `build-vk.sh` (CUDA-off, `-j1`) —
      uncapped builds have OOM'd that box before.

---

## P2 — The split itself (pipeline parallelism)

Needs its own spec before any code. `llama.cpp` has no pipeline parallelism today;
RPC is remote-tensor-ops and cannot express this.

- [ ] Spec: layer-boundary transport, shard format, per-side pager configuration.
- [ ] Sharding: split a GGUF by layer range, or load a layer subset from full shards.
- [ ] **Boundary ownership decisions:** which side owns `token_embd` and `lm_head`,
      and where the MTP / draft layers land.
- [ ] Activation transport: one exchange per token per direction. Decide whether to
      reuse the RPC channel or add a thin dedicated one.
- [ ] Per-side pager: each instance pages only its own layer range.
- [ ] Cross-machine prefetch overlap: 2026 prefetches its layers while main computes
      its own. This is the payoff — verify it actually materializes.

---

## P3 — Capacity and logistics

- [ ] **Clear NVMe on 2026** — needs ~72 GB free for GLM's 30% share (~45 GB if we
      prototype on DS4 first). Currently **26 GB free of a 229.8 GB drive**; the
      drive size is the hard cap on what can live there. *(kmbandy owns.)*
- [ ] **Clear NVMe on main for GLM** — 171 GB free vs ~239 GB needed for UD-IQ2.
      `/mnt/hdd` has space but is 7200 rpm and unusable for paging. *(kmbandy owns.)*
- [ ] **Mixed-quant size-class pool fragmentation.** GLM's UD-IQ2 experts are mixed
      precision; `WP_SIZE_CLASS_SLOTS=0` is the known stopgap, a real allocator fix
      is wanted.

---

## Open questions

- [x] **RAM tier stays at 8 GB per machine — SETTLED by kmbandy (2026-07-24).**
      He will kill superfluous services to make room when these runs happen; that
      is his call and it is not to be re-litigated. Sizing therefore stands as
      written: main 48 GB VRAM + 8 = 56 GB, 2026 16 GB VRAM + 8 = 24 GB, ~70/30,
      ~13 of 44 DS4 layers on 2026 (~45 GB shard; ~72 GB for GLM IQ2).
      For the record, the measured constraint that prompted the question: an 8 GB
      pinned arena alongside the model process left 1.1 GB available on the 16 GB
      box during an unattended sweep. That is a reason to free RAM first, not a
      reason to shrink the tier.
- [x] **Hot-set coverage curve: NO re-derivation needed.** An earlier revision of
      this roadmap claimed the 2026-07-21 design underestimated per-expert size by
      2x (25.2 MiB vs its ~13.4 MB). That claim was WRONG and is retracted: the
      283.5 GB routed-expert total it rested on exceeds the whole 160.0 GB model
      file. The pager's own figures settle it -- 33792 expert sub-pages
      (44 x 256 x 3) at the 4456448 B slot = 150.6 GB, i.e. **12.75 MiB per
      expert**, cross-checking to ~12.9 MiB from file size minus non-expert
      tensors. The design's ~13.4 MB was right; its coverage table stands. The
      OTHER caveat on that table is untouched and still open: coverage was
      measured on a single non-representative prompt.
- [ ] **The dual-GPU overlap question (tiered design §7) is still unanswered** and
      cannot be answered while decode is storage-bound. Needs a profiler/stream
      timeline showing concurrent kernels, or an A/B at compute-bound residency.

---

## Done (2026-07-24) — context, not work

- FFN-island device role built, measured, landed (`09221a91c`, `a0460ed53`,
  `447744bb5`, results in the spec §11). PPL-neutral (1.9007 → 1.9035), **−6.4%
  NVMe** reproducibly, throughput indistinguishable from noise.
- Multi-device resident works — verified with the CUDA remote (`ROCm1,RPC0`
  coherent). It was *not* the cause of the 4-GPU garbage; the Vulkan card was.
- **RAM victim tier and host prefetch worker verified working** for the first time
  (`host_tier_hits`, `host_prefetch_read`, strike gate active). Everything measured
  before this was Tier-0 only.
- Best currently-working config: 3 GPUs —
  `paging=ROCm0, --weight-paging-resident-device ROCm1,RPC0`, 8 GB HostTier +
  prefetch, coherent output.
