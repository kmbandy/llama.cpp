# 2026-07-27 morning pickup

Written at the end of 2026-07-26. Read §1 and §4 before touching anything.
§4 is the only open engineering question; everything above it is settled and
measured, and re-deriving it will waste a morning.

---

## 1. TL;DR

| work | state |
|---|---|
| Vulkan weight paging (RX 480) | **DONE** — PPL-equivalent, root cause was not the suspect in the old brief |
| Vulkan `ggml_sinkhorn_norm` | **DONE**, shipped with a real bug found |
| Pool auto-sizing + block-aligned slots | **DONE** |
| HIP regression for all of the above | **PASSED** on the 6900 XT |
| Fleet on one master | **DONE** — both boxes at the same tip, `build-army` + `build-hip` rebuilt |
| RAM victim tier on Vulkan | **DONE** — was a live footgun, now works and is observable |
| 4 GPUs / 3 backends / 2 machines, one model | **ACHIEVED**, numerically clean |
| laguna decode baseline | **MEASURED** — 1.1–1.2 t/s, see §3 |
| **P2P transport defect** | **FOUND, fix demonstrated (+33%), REVERTED — §4** |

**Start at §4.** Everything else is done or measured.

---

## 2. Machine and repo state

- Both boxes on **master, same tip `89b6491f0`**, trees clean on 2026,
  60 dirty files on main belonging to the **DSWS session — do not touch**.
- **91 commits ahead of `origin/master`. Nothing pushed to GitHub.**
- Build dirs current: 2026 `build-army` (CUDA sm_61 + Vulkan + **RPC now ON**),
  main `build-hip` (multi-arch `gfx1201;gfx1030`).
- No board claims held. All GPUs idle at shutdown.
- `llama-router.service` live on both boxes and **spawns from these build dirs** —
  never `pkill` by pattern.

**Disk, after kmbandy's cleanup:** 2026 `/` went 39 GB → **82 GB free**.
`/home/kmbandy/llama/kv-cold` shows ~358 GB in QDirStat and is **1.3 MB real**
(sparse holes) — QDirStat reports *apparent* size, so it will always mislead here.
The 112 GB ADATA SP550 (`sda`) is a **Windows install**, unmounted, and is the
biggest single reclaimable win on that box. `/mnt/storage2` is **USB**, not SATA —
unusable for paging.

---

## 3. Measured numbers — do not re-derive

### laguna decode, the shipped Phase 3 topology

`paging=ROCm0 (R9700)`, `residents=ROCm1 (6900 XT)`, 9000 slots (23.2 GiB),
128 tokens, output verified coherent (non-degenerate — this matters, degenerate
decode repeats routing and flatters every paging measurement).

| config | gen t/s | page_ins | io_gb_read |
|---|---|---|---|
| tier off | **1.1–1.2** | 88,276 | 168.5 GB |
| tier on (4 GB) | 1.0 | 88,276 | 167.3 GB |

- **1.316 GB/token.** All-miss footprint computed from the GGUF is **2.68 GB/token**
  (per expert 2.163 + 1.769 + 1.769 = **5.70 MB**, × 10 active × 47 MoE layers),
  so pool hit rate ≈ **51%** — cross-checked independently by page counts
  (690 misses of 1410 accesses/token).
- **DS4-Flash reads 0.64 GB/token and decodes 3.57 t/s.** laguna is structurally
  ~2× heavier per token despite being 4-bit and half the size. But see §4 —
  the DS4 figure was measured **with P2P working**, so any comparison taken while
  P2P was dead is not like-for-like.

### Pool coverage is the dominant lever (confirmed)

12.5 GiB → 23.2 GiB pool: **−24% page-ins, −24% bytes, +33% t/s.**
Cleanest causal result of the day; consistent across every metric.

### 4-GPU mesh (laguna, 69 GB, RPC)

PPL **5.9193 ± 0.74641** vs local-only control **5.9700 ± 0.75429**.
Placement matrix, identical pool, PPL deterministic:

| residents | PPL |
|---|---|
| ROCm1 (local 6900 XT) | 5.9700 |
| RPC1 (480, Vulkan remote) | 5.9965 |
| ROCm1 + RPC0 + RPC1 (all four) | 5.9193 |
| RPC0 + RPC1 | 7.2728 |
| RPC0 (1070) alone | 8.4878 |

Open, unexplained: degradation is **dose-dependent on the 1070's share** of dense
(all → 8.49, half → 7.27, a third → clean). Not capacity (it held 4000 of 8192 MiB)
and not a broken kernel (`test-backend-ops` 13033/13033 on CUDA0). The mesh was
**14% slower** than staying local — it buys capacity, not throughput.

---

## 4. START HERE — the P2P defect

### The bug

P2P dies after **exactly 142 batches** — identical on an 8-token run and a
128-token run, so deterministic, not load — and permanently downgrades to
**sync-pread**, the slowest transport available, below even the io_uring host
ladder it could have fallen back to.

The log lies. It says `window mmap failed: Cannot allocate memory`, and
**the mmap is never called**:

```c
// acquire_window_(), wp-file-io-p2p.cpp
while ((int) window_cache_.size() >= max_windows_) {
    if (!evict_one_idle_()) { errno = ENOMEM; return false; }   // set BY HAND
}
```

and `submit()` routes **any** acquire failure through `switch_to_host_errno_()`.
A transient backpressure condition is indistinguishable from a real mmap failure.

### Fixing it works — this is measured, not projected

EAGAIN + `flush_submissions_()` + drain `reap_ready_cqe_()` + retry:

| | before | after |
|---|---|---|
| transport | 142 P2P / 16,549 serial | **16,691 P2P / 0 serial** |
| decode | 1.2 t/s | **1.6 t/s (+33%)** |
| io_effective | 0.971 GB/s | 1.249 GB/s |
| io_gb_read | 168.495 GB | **168.495 GB (identical)** |

Byte-identical is the correct signature for a transport-only change and proves it
did not perturb caching.

### Why it is reverted

With the **RAM tier also enabled** it produced a reproducible
`GGML_ABORT` — `wp::eval_cb active expert page-in failed`, `layer=1 weight_page=0
expert=205 sub_page=206`. Tier-off stable across 3 runs; tier-on aborted twice.
A second fix did **not** resolve it. Both commits reverted (`89b6491f0`,
`aac57ee3d`); main rebuilt and verified back to known-good (tier-on runs again,
P2P back to 142).

### Two suspects, neither verified — resume here

1. **Window refcount.** With `queue_depth=4` at most 4 windows should be pinned,
   yet **all 64 were**. Either refs leak on some completion path, or the cache cap
   and QD are mismatched. `release_inflight_key_` is called at 4 sites; find the
   path that skips it.
2. **`ensure_batch` Pass 1 has a silent null:**
   ```c
   const int s = pool_.alloc_slot(m.size);
   if (s < 0) { ++stats_.sync_fallbacks; continue; }   // leaves out_ptrs[i] NULL
   ```
   That `continue` has **no fallback**, which matches the abort's shape exactly,
   and the tier pins more slots so `alloc_slot` failure gets likelier. **Strong
   next suspect. Check `sync_fallbacks` in a tier-on run first — it is a one-line
   read and it either confirms or kills this immediately.**

### What the failure log did NOT show

No `file IO failed`, no `stage_in failed`. The read path was not the failing link.
Do not start there.

> **RESOLVED 2026-07-27 — and this paragraph was wrong.** The read path *was*
> the failing link. `page_in_sync_` logs `file IO failed for page %d` through
> `LLAMA_LOG_WARN`, which §6 of this very brief records as **suppressed without
> `-v`**. The log fired; we could not see it. Absence of a log is not absence of
> execution — we wrote that rule down and then reasoned straight past it.
>
> Both suspects below were also wrong in their framing. **Suspect 1 is not a
> refcount leak** — nothing leaks, every release path is balanced. `submit_batch`
> pushes the whole batch through `submit()` with no reaping in between and each
> pins a window until completion, so concurrently-pinned windows equal the
> **batch width**, never `queue_depth`; `max(queue_depth*4, 64)` is sized off the
> wrong quantity. That is also why it dies at exactly 142 batches on both run
> lengths: the first batch whose window spread crosses the cap is a property of
> the model's layout, not of load.
>
> **The abort had a third and fourth cause**, which is why last night's two
> attempts each failed despite both being individually correct:
> `submit()` delegated to `host_` only when P2P was *already disabled*, so the
> staging retry arrived with P2P live, failed the pool bounds check, and hit
> `switch_to_host_("dst outside pool") + return false` — failing on its only
> attempt. And `reap_raw_` drained `host_` only when `!p2p_enabled_`, so simply
> routing the retry would have **hung** instead of aborting.
>
> Fixed in `f82a6dbfb` (ensure_batch fallback) and `4f9cdc32f` (all four P2P
> pieces). **Compiles; NOT yet verified on hardware** — the tier-on run that
> aborted still needs a rerun on the R9700.

### Also worth knowing

`FileIOLayer::submit_batch` **stops at the first rejection** and the caller treats
`[n_queued, N)` as failed, so one rejected request costs the rest of that batch a
sync fallback. Correct but slow; a `submit_batch` that skipped rather than stopped
would be better.

---

## 4b. 2026-07-27 validation on mad-lab-2026 (measured — do not re-derive)

Vehicle: LFM2.5-8B-A1B Q6_K (lfm2moe, 24 blocks, **32 experts / 4 used**, 2 leading
dense), `llama-perplexity`, 2 chunks, `-c 512`. Not gpt-oss.

| run | PPL | page_ins | sync_fallbacks | exhaustions | exit |
|---|---|---|---|---|---|
| CUDA0, 128 slots, tier off | 30.5792 ± 5.28405 | 4436 | 308 | 0 | 0 |
| CUDA0, 128 slots, tier ON 2 GB | **30.5792 ± 5.28405** | 4436 | 308 | 0 | 0 |
| CUDA0, 12 slots | — | — | — | 20 | **134 (SIGABRT)** |
| CUDA0, 6 slots | — | — | — | 26 | **134 (SIGABRT)** |
| Vulkan0 (RX 480), 128 slots | 30.7924 ± 5.32553 | — | — | 0 | 0 |

**Correction to `f82a6dbfb`'s commit message — it overclaims.** The `ensure()`
fallback **diagnoses but cannot recover**. Every failing run shows
`pinned_this_batch == pool size`: `ensure_batch` pins every cold miss *before*
issuing reads, so whenever `alloc_slot` fails it is because earlier misses **in
the same batch** hold the pins — and `ensure()` contends for those same pins.
It can essentially never succeed. The real value delivered is the diagnostic
(page, size, batch width) replacing a silent NULL. Do not call it a recovery.

**Pool minimum is set by TOTAL experts, not `n_expert_used`.** Under prefill,
`ensure_batch`'s page set is every expert any token in the chunk routes to — at
32 experts × 512 tokens that is nearly all of them. A pool below that cannot
work, and chunking cannot fix it, because the MoE kernel needs all active
experts resident simultaneously. **The right fix is a startup validation** that
fails with an actionable message instead of aborting mid-run. Not yet written.

**Negative result that matters: the RAM tier does NOT starve `alloc_slot`.**
Tier on vs off is byte-identical PPL with 0 exhaustions and
`host_tier_stores: 4308`. That **eliminates suspect 2** as the cause of the
2026-07-26 tier-on abort and leaves the P2P chain (§4), which is what the code
reading concluded independently. It also confirms the tier is numerically
transparent.

**Q6_K on Vulkan is fine** — 0 block-alignment errors, so `PoolAllocator::init`'s
padding does produce block-multiple offsets for 210-byte blocks. This closes the
open question raised in `3b90dd346`'s commit message.

**Still NOT verified:** the four-part P2P fix (`4f9cdc32f`). It is ROCm/HSA
`dma_buf` only and cannot be exercised on 2026 at all. Needs the R9700.

### Tooling gotchas hit today

- `llama-cli` no longer accepts `-no-cnv` ("use llama-completion"), but
  `llama-completion` does **not** register `--weight-paging` (SERVER/CLI/PERPLEXITY
  only). Use `llama-perplexity` for paging work.
- Piping a run through `| tail` buffers everything; killing it yields an empty log.
  Redirect to a file instead.

---

## 4c. VERIFIED ON HARDWARE 2026-07-27 — §4 IS CLOSED

R9700 (paging) + 6900 XT (residents), laguna-S-2.1 UD-Q4_K_XL, 9000 slots,
`WP_RESIDENT_DENSE=1`, 128 tokens requested (EOS at 22). Commit `4f9cdc32f`.

| arm | transport | batches | page_ins | io_gb_read | io_eff GB/s | decode t/s | exit |
|---|---|---|---|---|---|---|---|
| A tier off | P2P | 3175 / 0 serial | 52278 | 99.619 | 1.024 | 1.48 | 0 |
| B tier ON 4 GB | P2P | 3175 / 0 serial | 51191 | 97.553 | 1.096 | 1.46 | 0 |
| C repeat of A | P2P | 3175 / 0 serial | 52278 | **99.619** | 1.131 | 1.51 | 0 |
| D control | **SERIAL** | 0 / 3174 serial | 52275 | **99.612** | 0.789 | **0.93** | 0 |

**The gate passed.** Arm B is the exact configuration that `GGML_ABORT`ed twice on
07-26. It completes. `host_tier_hits: 1086`, `host_tier_stores: 42986`.

**P2P survives backpressure:** 3175 / 0 serial, against the broken 142 / 16,549.

**P2P is worth +59–62% on decode** (0.93 → 1.48–1.51), A-vs-C control spread 2%.
This supersedes the 07-26 "+33%", which compared against a partially-P2P baseline.

**Bytes identical across transports** (99.612 vs 99.619, 3 pages of 52,278) — the
correct signature for a transport-only change. Judge transport work on bytes and
batch counts; t/s is secondary on this box.

**Suspect 2 is dead, measured twice.** `sync_fallbacks: 0` and
`batch_slot_exhaustions: 0` in every arm including tier-on; and on 2026 the tier
gave byte-identical PPL with 0 exhaustions. `f82a6dbfb` fixes a real latent
defect but was never the cause of the abort.

**Caveats.** `LLAMA_WP_TRANSPORT=host` resolves to SERIAL, not the io_uring host
ladder — D is P2P vs sync-pread. And 1.13 GB/s is still only ~39% of the 2.9 GB/s
these drives give at QD16: **the fix did not touch concurrency.** `avg_n` ~5.4 of
a 16-deep ring is unchanged and is now the top remaining lever.

---

## 5. Corrections to the record made today

- **The old brief's §4.3 suspect was wrong.** Vulkan paging's garbage output was
  not slot lifetime. `ggml_vk_mul_mat_id` forks on `src2->ne[1] <= 8` and only the
  **vec** path had paged addressing; a 10-token prompt (chat template) sent the whole
  **prefill** through `mul_mm.comp` with uniform-stride expert addressing against a
  `src0` pointing at one pool slot. Prefill poisoned the KV cache and decode emitted
  `6666…` forever despite the decode path being correct.
- **"WP_FORCE on a non-paged model gives coherent output" proved nothing** — a shader
  that never received `p.paged` produces exactly the same pass. A test a broken system
  also passes is not evidence.
- **PPL in this harness IS deterministic** (5.9700 three times, four decimals). An
  earlier retraction of that was itself wrong: the 8.4878/8.2067 pair differed because
  the `--device` lists differed. **Listing an extra RPC device changes results even
  when it holds nothing.**
- **I measured a full decode baseline on the slowest transport, twice**, because
  `LLAMA_WP_TRANSPORT` was unset. Always read the `TRANSPORT: active=` line.
- **The RAM tier's "0 hits" was measured on prefill**, where a 4 GB tier holds ~4% of
  a 36k-page model against a streaming pattern. On decode it does hit (1,268) but
  serves ~1%, and the tier thrashes: 109,033 stores for 1,268 hits.

---

## 6. Standing gotchas learned today

- **A `GGML_USE_*` macro says what was COMPILED, never what is RUNNING.** Three
  separate bugs today from this: `WP_ENSURE_BATCH_HOST`, the routing-index
  `hipMemcpy`, and `HostTier::store_from_device`. `build-army` defines
  `-DGGML_USE_CUDA` even on Vulkan-only runs. Decide backend at runtime
  (`GpuTransport::is_vulkan()`).
- **`LLAMA_LOG_WARN` during model load / pager init is SUPPRESSED without `-v`.**
  Cost real time twice. Absence of a log is not absence of execution.
- **`ssh host bash -s < script.sh` feeds the script on stdin** — any inference binary
  in that script eats the remaining script text as its prompt. Always `< /dev/null`.
  `ssh host bash -c '...'` is unreliable under fish; use the script-pipe form.
- **`load_tensors: ... model buffer size = 0.00 MiB` is a lie** in this fork. Verify
  placement by sampling VRAM mid-run.
- **`--device` does not drive placement when paging is on.** The `WP_RESIDENT_DENSE`
  router takes the paging device from the first `--device` entry and residents from
  `--weight-paging-resident-device`; `auto`/a single name selects **exactly one**.
  Always read the `WP_RESIDENT_DENSE router:` line.
- **Board queue slots can silently fail to convert.** I queued #2 for the R9700, the
  holder released, `claims` went empty, and I was never promoted. Re-check rather than
  wait.
- **On the display-attached R9700, pass explicit `--weight-paging-slots`.**
  Auto-sizing took a 28.7 GiB pool and tripped the 95.2% VRAM alert; the 3 GiB
  reserve does not hold once KV + compute allocate after the pool.
- **Merging onto main's dirty tree:** prove the incoming file set is disjoint from the
  dirty set, use `git merge --ff-only`, then diff `git status --porcelain` before/after
  as proof. Worked four times today.

---

## 7. Roadmap context — GLM

The plan is the **two-process layer split**, not the RPC mesh.

- **§4a layer-range graph construction is the gate.** No layer-subset concept exists
  for the model today (`il_start`/`il_end` are control-vector only). Needs: range
  parameter, graph input = boundary activation instead of embedding lookup, graph
  output = activation instead of logits, KV sized to owned layers, `graph_reserve`
  and compute-buffer sizing following the reduced range.
- **The tap exists, the inject does not.** `llama_get_embeddings_layer_inp` is
  production code feeding the DFlash drafter — the output side of a boundary already
  works. There is no counterpart to *start* a pass from a supplied residual.
- **Scope risk, unresolved:** the layer loop lives in **115 of 139 model builders**.
  Settle whether the boundary can live generically in `llm_graph_context` before
  writing code. Phase 0 needs it in **one** builder only.
- **Phase 0 first** — one machine, small model, localhost, no paging, no network.
  Success = token-for-token identical vs unsplit, greedy, same seed. The spec says
  explicitly: do not skip it, and do not start it on DS4.
- **Weights on 2026 are a Phase 2 prerequisite, not a blocker now.** 2026 has no
  laguna and no DS4. laguna (69 GB) now fits in its 82 GB free; DS4 (151 GB) does not
  without moving CachyOS to the SATA SSD.
- GLM is in **neither** `llm_arch_is_recurrent` nor `llm_arch_is_hybrid`, so it dodges
  the RPC hang that kills ornith-35B. `glm-dsa` is implemented here and is MoE.
  **We have no GLM weights on either box** — that gates any GLM work.

### On the 10–20 t/s target

At laguna's 1.316 GB/token, 10 t/s needs 13 GB/s and both drives together give
~5.8 GB/s. At DS4's 0.64 GB/token, 10 t/s needs 6.4 GB/s — borderline reachable with
working P2P plus the split, and MTP (~0.64× bytes at depth 4) or expert pruning
(keep 128 of 256 retains 96.4% of activations) would close it. **The number that
decides this is GLM's per-token expert footprint, which varies 2× between the two
models we have.** Any projection before that is a guess.

---

## 8. Suggested order for the morning

1. **`sync_fallbacks` in a tier-on run** — one read, confirms or kills suspect 2 in §4.
2. Fix P2P properly and re-land it. It is worth **+33% decode** and it is the
   cheapest measured win available.
3. Re-run the tier arms on honest transport — the "tier is worth ~1%" conclusion was
   measured on sync-pread and is suspect.
4. Then §4a scoping: generic vs per-builder.

Do **not** start §4a before P2P is settled — it is a multi-week build and the
transport bug distorts every throughput number that would justify it.
