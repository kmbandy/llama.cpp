# DeepSeek V4 Flash — Paged Decode Acceleration Levers

**Date:** 2026-07-08 (updated after hetero decode smoke)
**Target:** single-stream (`--parallel 1`) interactive decode speed for DS4 Flash on mad-lab-main. Not multi-user throughput.
**Branch:** `feat/wp-dflash-ds4`

Related:

- [2026-07-08-wp-hetero-dflash-oracle-plan.md](./2026-07-08-wp-hetero-dflash-oracle-plan.md) — hetero placement + DFlash oracle design
- [weight-paging-batch-eval-continuation.md](./weight-paging-batch-eval-continuation.md) — WP_RESIDENT_DENSE history
- `~/wp_logs/het-*.log` / `het-matrix.log` — latest hetero vs single-card smoke

---

## The bottleneck (measured, not assumed)

DS4 Flash: 284B total / 13B active per token, 256 routed experts (6 used + 1 shared), 43 layers, Q8_K_XL GGUF ~151-162 GB. Experts don't fit VRAM -> demand-paged from NVMe.

The GPU still spends much of each token **idle waiting for experts** (`page_in_sync`, low prefetch hit-rate). This is **not** primarily a raw NVMe bandwidth wall on single-card (baseline ~1.1 t/s is reachable). It is **un-overlapped I/O latency plus incomplete forward knowledge of which experts come next**.

### Current measured decode (same prompt, WP strip, 2000 slots)

| Config | n | t/s | page_ins | notes |
|--------|---|-----|----------|--------|
| single-card nodraft (dense on R9700) | 16 | **1.11** | ~12738 | baseline |
| single-card dflash oracle (prior lv3) | 16 / 64 | **1.22** | ~12.8k / ~29k | draft = paging oracle under strip |
| **hetero** dflash Q8 (attn on 6900XT, experts R9700) | 16 | **0.78** | 12783 | FA on; no 94 GiB PP path |
| **hetero** dflash Q8 | 64 | **0.81** | 29247 | page_ins shape ~ single-token |

Hetero is coherent and load-healthy, but **~30% slower** than single-card at the same slot budget. Growing the R9700 expert pool (+~15-16 GB now that dense left that card) is the first recovery bet. TB3 residual transport is a separate lever (below) — do not confuse the two.

---

## Hardware constraints (these shape every lever)

- **R9700** (ROCm0, paging device): PCIe 4.0 x16, CPU-direct, **~28 GB/s**, SAM active. -> NVMe->VRAM (P2P) and RAM->VRAM are both fast. Primary target for **expert weights** and the page pool.
- **6900XT** (ROCm1, resident / attention island): Razer Core X eGPU over **TB3 ~2.7 GB/s**. **No useful inter-GPU P2P fabric** for this pair (measured `canAccessPeer 0->1: 0`; `1->0: 1` asymmetric and not trusted as real P2P). Residual crossings are **host-staged**.
- **Host RAM:** selective HostTier for demoted *routed* experts later (~4 GB); **not** for embd/shexp under the locked layout. `token_embd` already on CPU.

### Locked placement (hetero)

```
6900 XT (TB3)     attention + FA + hot KV + draft Q8 + lm_head/...
R9700 (x16)       expert page pool + shexp resident; NO dense attn weights
Host              token_embd; optional HostTier demoted experts later
```

Never put expert **weights** on TB3. Residual / activation crossings are acceptable **if** transport cost is controlled.

---

## Target memory hierarchy

```
VRAM pool on R9700  (pinned / hot experts; grow past 8.5 GB once dense is off-card)
  ^ 28 GB/s PCIe
HostTier (optional) (demoted routed experts only)   ~4 GB selective
  ^ NVMe read / demote
NVMe                (cold experts, P2P direct-to-VRAM on R9700)
```

Draft signal (DFlash + host tid2eid hash layers 0-2) is the paging oracle under WP. Multi-row target verify is stripped by default (`WP_SPEC_VERIFY_MAX` auto) so the active set stays near top-k (~6), not the multi-token union thrash.

---

## Levers by implementation status

### A. Done / usable

- **WP_RESIDENT_DENSE** — page only routed `ffn_*_exps`; dense resident. Single-card win measured earlier (~0.02 -> ~1.0+ t/s era).
- **Hetero router** — experts/shexp -> ROCm0, embd -> CPU, rest -> ROCm1; layer-home/KV/FA on resident. Load + decode smoke pass.
- **FA + 94 GiB reserve fix** — mark `has_tensor_overrides` when WP injects overrides; force pipeline PP off under WP multi-device. No more ~94 GiB ROCm0 graph reserve / PP retry.
- **DFlash draft-as-paging-oracle** — tid2eid pin, cold wave submit (WAVES=1, MAX_TOK=1), harvest/reap_finished, adaptive skip. Single-card ~1.22 vs ~1.11 nodraft.
- **Draft Q8_0** — `/home/kmbandy/models/dflash-speculator-DS4-Q8_0.gguf` (~1.9 GB) on resident.
- **P2P / dma_buf paging (NVMe->R9700)** + io_uring + QD>4 hang fix.
- **Prefill / cold-start oracle (P0)** — server-context fires tid2eid before first decode.

### B. Next — high expected ROI, little design risk

- **Grow R9700 expert pool** — **measured 2026-07-09.** Slot size 4456448 B. Matrix (hetero nodraft, n=32 cold+warm same prompt):

| slots | pool | R9700 used (approx) | page_ins (2 reqs) | cold t/s | warm t/s |
|------:|------|---------------------|------------------:|---------:|---------:|
| 2000 | 8.5 GiB | ~12.7 GiB | 34245 | 0.985 | 1.017 |
| 4000 | 17.0 GiB | ~21.1 GiB | 27013 | 1.365 | 1.444 |
| 5500 | 23.4 GiB | ~27.3 GiB | 24880 | 1.412 | 1.401 |
| **6500** | **27.6 GiB** | **~31.4 GiB** | **22973** | **1.512** | **1.532** |
| 7000 | 29.8 GiB | OOM | - | - | compute buf 671 MiB fail |

**Recommend `--weight-paging-slots 6500`** (or 6000 if longer ctx needs compute headroom). 7000 dies on ROCm0 compute reserve.

- **Mid-graph residual (T5–T7, 2026-07-09)** — view-parent coalesce, FFN act pin, hc_pre RMS pin. Hetero nodraft n=16 ~**1.28 t/s** before pool growth; residual stage ~0.14 GB/run (island edges only). TB3 transport further investment is **low ROI**.

### B2. Ordered next levers (post 1.5 t/s stack) — **GOING DOWN THIS LIST**

Bottleneck is no longer residual transport. It is **NVMe expert miss latency + low sustained P2P BW + weak forward knowledge of which experts stay hot**.

| # | Lever | Why | Status |
|---|--------|-----|--------|
| 1 | **DFlash oracle @ 6500 slots** | Fat pool needs better expert prediction; nodraft still ~23k page_ins / 2×32 tok | next measure |
| 2 | **Sticky L2** (`WP_STICKY_L2=1`, `WP_STICKY_L2_PAGES=32..128`) | Path exists but pins=0 in logs; use pool capacity | next |
| 3 | **NVMe/P2P I/O bandwidth** | Peak P2P ~**6.5 GB/s**; live `io_effective_gb_s` only **~0.7–1.9 GB/s** (see B3). Raise QD / batch / async ensure | **in progress** |
| 4 | **Prefetch hit-rate** | `prefetch_hit_rate ~0.1%`; `sync_fallbacks ≈ page_ins` | with 3 |
| 5 | **Draft cost cut under strip** | `spec-draft-n-max=1`, skip when sticky hot | after 1 |
| 6 | **Conditional multi-row verify** | Only if hit ratio high + free pool healthy | later |
| 7 | **HostTier (~4 GB)** | Re-measure on fat pool; earlier locality weak | later |
| 8 | **eGPU FA profile** | Secondary once I/O overlapped | later |

```
1. DFlash + 6500 A/B vs nodraft 1.53
2. Sticky L2 enable/tune
3. I/O BW: QD / io_uring depth / batch submit / async ensure  ← user priority; peak 6.5 vs ~1
4. Prefetch hit-rate + draft-only QD
5. Draft cost cut (n-max=1 / skip when hot)
6. Conditional multi-row verify
7. HostTier re-smoke
8. eGPU FA only if I/O no longer dominates
```

### B3. NVMe / P2P I/O bandwidth gap (measured)

| signal | value | notes |
|--------|------:|-------|
| Microbench P2P peak (historical) | **~6.5 GB/s** | NVMe → R9700 dma_buf |
| Expected real-world target | **~5–6 GB/s** | sustained under multi-QD |
| Live `io_effective_gb_s` @ 2000 slots | **1.86** | n=32×2; still << peak |
| Live `io_effective_gb_s` @ 6500 slots | **0.68** | more wait/overhead in timer; not faster DMA |
| Single sync page_in (DIAG) | **~1.3 GB/s** | 4.45 MiB / ~3.5 ms serial |
| `sync_fallbacks` @ 6500 | **22438 / 22973** | almost all ensures sync-path |
| `prefetch_hit_rate` | **~0.1%** | prefetch not hiding latency |
| Depths in use | **prefetch=4, io_uring=4** | `WP_PREFETCH_DEPTH` / `WP_IOURING_DEPTH` |

**Diagnosis:** not an NVMe hardware wall — **queue depth + sync ensure serialization** leave the bus idle. Each expert is ~4.45 MiB; serial ensure → one I/O at a time → ~1 GB/s class. Peak needs **many in-flight** P2P reads (batch submit + higher QD + prefetch that completes before ensure).

**I/O levers (try in order):**
1. Raise `WP_PREFETCH_DEPTH` + `WP_IOURING_DEPTH` (8/16/32) — measure `io_effective_gb_s` + t/s (QD>4 hung historically without demux; test carefully).
2. Keep / force **batched** `prefetch_pages_batch` for MoE actives + cross-layer.
3. **`WP_ENSURE_BATCH=1`** — multi-QD P2P for MoE actives (see measured table below). **Default OFF for nodraft; ON when draft window active.**
4. `WP_ASYNC_ENSURE=1` — **measured not a win on P2P+ensure_batch** (see "Compute overlap" below).
5. Oracle/sticky so fewer page_ins (BW × fewer bytes wins more than BW alone).
6. Avoid multi-row verify thrash that multiplies unique pages.

#### Compute overlap (2026-07-09) — what is foundational vs what is dead

**Scheduler shape (today):** `eval_cb` ensure → return → launch GPU → sync. Host blocks in
`ensure_batch` wait before the MoE kernel of *this* op. GPU is idle during that wait.

**`WP_ASYNC_ENSURE=1` (current stack, P2P, ensure_batch ON):** warm ~1.91 t/s vs ~2.03 baseline;
`ensure_batch_gb_s` still ~3.56. Path is largely **inert for P2P**: `ensure_batch` already waits
on NVMe→VRAM; there is no stage-2 H2D event to hide. Slight overhead only.

**Sister co-wait vs async sisters:** co-wait gate+up+down in one `ensure_batch` (cap 18) is the
winner. Async-only sisters (wait this weight, prefetch others) dropped warm **2.03 → 1.68 t/s**
(`avg_n` 8.6 → 3.4, more ensure_batch calls). Multi-QD efficiency of one larger wait beats
partial hide of sister I/O under gate compute.

**Where real overlap lives (foundational, still open):**

| Window | Who computes | What I/O can hide | Needs |
|--------|--------------|-------------------|-------|
| Same MoE layer, later weights | gate/up kernel | already paid by sister co-wait | done |
| Next MoE layer | this layer MoE | `WP_NEXT_LAYER_PREFETCH_K` (submitted 68, hits ~0–20) | deeper queue / less pin pressure |
| **eGPU FA (hetero)** | 6900XT attention | R9700 expert page-in for *this* layer | expert IDs **before** FA ends → draft oracle / sticky speculation |
| Host during GPU sync | — | reap/tick prefetch CQ | background reaper or sync hook |

**Implication:** do not expect `WP_ASYNC_ENSURE` to raise ensure_batch GB/s. Prefer (1) draft/sticky
speculation so page-in starts during FA, (2) make cross-layer prefetch actually hit, (3) optional
I/O reaper during long GPU windows.

#### Measured @ 6500 slots (2026-07-09 evening) — n=32 cold+warm

| config | warm t/s | page_ins | sync_fb | io_eff GB/s | notes |
|--------|---------:|---------:|--------:|------------:|-------|
| baseline QD=4 | 1.529 | 22973 | 22438 | 0.68 | serial ensure path |
| QD=16 only | 1.454 | 22990 | 22424 | 0.32 | **no help** without ensure_batch |
| QD=32 only | 1.511 | 23001 | 22407 | 0.26 | same |
| sticky+QD16 | 1.680 | 22990 | 22424 | 0.34 | sticky pins≈3 (nodraft weak) |
| **ENSURE_BATCH=1 +QD16** | **1.655** | 22991 | **0** | *timer* | multi-issue MoE; **sync_fb=0** |
| ebatch+sticky | **1.694** | 22991 | **0** | *timer* | best nodraft so far |
| dflash+ebatch+sticky | 1.605 | 23137 | 0 | *timer* | draft tax; accept=0; sticky pins=27 |

**Finding:** depth alone does nothing because **~98% of ensures were `sync_fallbacks`** (serial `page_in_sync_`). `WP_ENSURE_BATCH` (now **default ON**; `=0` disables) routes MoE through concurrent P2P.

**Burst meter (after timer fix):** `ensure_batch_gb_s` ≈ **1.9–2.0 GB/s** with max concurrent miss set ~6 (one weight’s actives). Peak P2P 6.5 still not sustained — random 4.45 MiB NVMe reads + small burst size. Sister co-batch (gate+up+down together, max_n=72 on prefill) **hurt t/s**; kept actives-only + async sister prefetch.

| config | warm t/s | ensure_batch_gb_s | max_n | notes |
|--------|---------:|------------------:|------:|-------|
| ebatch actives only | **~1.69** | ~2.0 | ~6 | best nodraft |
| ebatch + sisters uncapped | ~1.60 | ~1.94 | 72 | thrash |
| ebatch + sisters cap 24 | ~1.57 | ~1.97 | 24 | no win |

**Recommended env for now (best measured ~2.05 warm t/s, ensure_batch ~3.6 GB/s):**
```bash
# WP_ENSURE_BATCH default ON
export WP_PREFETCH_DEPTH=16
export WP_IOURING_DEPTH=16
# Sticky L2 / FA-spec: leave OFF on this hetero stack (see sticky section).
# export WP_STICKY_L2=1
# export WP_STICKY_SPEC=1   # FA-window cold submit; needs WP_STICKY_L2=1
# --weight-paging-slots 6500
```

#### Sticky / draft expert speculation (2026-07-09)

**Goal:** start expert page-in during eGPU FA using previous-token routing.

**Shipped infrastructure:**
- Score sticky from **real** `record_active_expert_pages` (+ sister expand).
- `prefetch_sticky_hot_experts()` — recent history cold submit + tick.
- FA hook on **layer-0** `FLASH_ATTN_EXT` only, gated by **`WP_STICKY_SPEC=1`**.
- Slower sticky pin refresh (less promote/demote thrash).

**Measured (n=32 cold+warm, 6500, P2P, ensure_batch):**

| config | warm t/s | page_ins | sticky pins | notes |
|--------|---------:|---------:|------------:|-------|
| **sticky off (recommended)** | **~2.05** | 22989 | 0 | best |
| sticky on + FA spec (early) | ~1.35 | 79k | 64 | every-layer FA thrash |
| sticky on + layer0 + hist cold | ~1.93–2.02 | 24k | 64 | pins cost; cold submit often 0 |
| sticky on, no FA force-refresh | ~2.01 | 24k | 64 | still +1k page_ins vs off |

**Why no win yet:** MoE experts for DS4 under single-token decode have **weak cross-token reuse** for a 64-page pin set; pins steal pool slots and raise page_ins. FA-window cold submit finds history already resident (`submitted=0`) once ensure_batch+sisters paid. **DFlash tid2eid oracle** (not loaded in these smokes: `draft_prefetch_pages_submitted: 0`) is the next signal to pair with this hook.

**Keep:** code + counters (`sticky_spec_*`). **Default:** sticky/spec off until draft model A/B.

#### DFlash as paging oracle (2026-07-09) — loaded + measured

**CLI (oracle-only; target still single-token under WP strip):**
```bash
--model-draft /home/kmbandy/models/dflash-speculator-DS4-Q8_0.gguf \
--spec-type draft-dflash \
--spec-draft-n-max 1 \
--spec-draft-device ROCm1 \
--spec-draft-ngl 99
# WP_SPEC_VERIFY_MAX unset => draft 1 -> 0 in target batch (accept=0 by design)
```

**Fair A/B (same session, 6500, P2P, ensure_batch, QD16, sticky off, n=32 cold+warm):**

| config | cold t/s | warm t/s | page_ins | draft_sub | tid2eid hits | ensure GB/s |
|--------|---------:|---------:|---------:|----------:|-------------:|------------:|
| nodraft | 1.90 | 1.74 | 22989 | 0 | 18 | 3.51 |
| **dflash n_max=1** | 1.80 | **1.90** | 23058 | **112** | **63** | 3.43 |
| dflash + sticky+FA-spec | 1.70 | 1.64 | 24132 | 112 | 63 | 3.51 |
| dflash waves=4 depth=32 | 1.70 | 1.79 | 24379 | 938 | 342 | 3.38 |

**Findings:**
- DFlash path is live: `draft-dflash` loads, `llama_wp_on_draft_tokens` submits tid2eid cold pages, strip keeps accept=0.
- Warm **~+9%** vs nodraft in-session (1.74 → 1.90); cold slightly worse (draft GPU tax on ROCm1).
- tid2eid covers **hash layers 0–2 only** (~54 pages/fire); free_q caps one wave at QD=16 so early fires submit 16 of 54.
- More waves/depth raise hits (342) but also thrash/page_ins and **hurt** t/s.
- Sticky+FA-spec on top of DFlash still loses (pin cost).

**Recommended for decode:** DFlash Q8_0 oracle, `n_max=1`, adaptive on (default), waves=1, sticky off.

#### Sample-token oracle fix (2026-07-09) — root cause of “more page_ins”

**Bug (conceptual):** under WP strip, DFlash drafts a *future* token. Hash-layer experts for the
*next target forward* depend on the token that forward will **consume as input**, which is the
token we **just sampled** (ground truth). Draft-token tid2eid is the wrong prior → false loads.

**Fix:**
- `llama_wp_on_sampled_token(ctx, id)` after each sample → note `tid2eid(id)` (default ON:
  `WP_SAMPLE_ORACLE=1`).
- I/O deferred to layer-0 FA (`flush_sample_oracle_at_fa`): free pool slots first, then
  capped protected LRU (`WP_SAMPLE_ORACLE_MAX` default 16; `WP_SAMPLE_ORACLE_EVICT=0`
  forces free-only). Recent MoE history is temp-pinned during LRU
  (`WP_SAMPLE_ORACLE_PROTECT_HIST` default 8 snaps).
- Draft cold submit default **OFF** (`WP_DRAFT_PREFETCH=1` to re-enable).
- Prefill uses sample path on last prompt tokens.
- Stats: `oracle_tp/fn/fp`, `oracle_sample_fires`, `oracle_pages_submitted`,
  `oracle_pages_free_slot`, `oracle_pages_evict_slot`, `oracle_protect_pins`.

**Precision (proves the fix):**

| prior | tp | fp | fn | precision tp/(tp+fp) | page_ins | warm t/s |
|-------|---:|---:|---:|---------------------:|---------:|---------:|
| draft token (old) | 126 | 522 | 11889 | **~19%** | 23048 | ~1.65 |
| sample free-only (lean) | 3402 | 108 | 2475 | **~97%** | 23695 | ~1.83 |
| no oracle | 0 | 0 | - | - | **22970** | **~1.92** |

Sample prior is **correct** (almost no FP). Free-only lean mode often has
`oracle_pages_submitted=0` (pool full) so precision never becomes I/O - protected
LRU path is required to convert that into FA-overlapped loads without thrashing
the hot MoE set.

**Still open for 5–6 GB/s:** larger *useful* bursts without pin thrash; overlap ensure with eGPU attn (`WP_ASYNC_ENSURE`); close software gap vs device (below).

#### fio-like proof (2026-07-09) — SN850X + same GGUF shard

Tool: `/tmp/wp_io_bench` (pread + liburing). File: `...-00002-of-00005.gguf` on **btrfs zstd** `/home` (`/dev/nvme0n1p2`). PAGE=4456448. Log: `~/wp_logs/fio-like-wp-io-bench.txt`.

| pattern | QD | BW |
|---------|---:|---:|
| seq 256 MiB **O_DIRECT** | 1 | **7.04 GB/s** |
| seq 256 MiB buffered (cold-ish) | 1 | 1.33 GB/s |
| random PAGE **O_DIRECT** pread | 1 | **4.00 GB/s** |
| random PAGE **O_DIRECT** pread | **6** | **6.17 GB/s** |
| random PAGE **O_DIRECT** pread | 16 | **6.21 GB/s** |
| random PAGE **io_uring O_DIRECT** | 6 | **6.08 GB/s** |
| random PAGE **io_uring O_DIRECT** | 16 | 5.87 GB/s |
| random PAGE buffered pread | 1 | 0.87 GB/s |
| random PAGE buffered pread (warm) | 6 | cache-inflated 10–20 GB/s (ignore) |
| **WP ensure_batch (live)** | ~6 | **~1.9–2.0 GB/s** |
| **WP serial page_in DIAG** | 1 | **~1.3 GB/s** |

**Verdict:** Device+fs can deliver **~6 GB/s** on the *same* expert page size at QD≥6 (O_DIRECT/io_uring). WP’s **~2 GB/s** burst is a **software path gap** (~3×), not an SN850X wall. Sequential O_DIRECT ~7 GB/s matches the historical P2P peak story.

**Implication:** Prefer fixing ensure/P2P pipeline (true in-flight overlap with compute, fewer sync points, avoid extra copies) over buying faster NVMe.

#### Windowed dma_buf P2P (2026-07-09) — kill full-pool host map

**Incident:** full-pool `mmap` of the VRAM slot dma_buf (~27.6 GiB) for io_uring
destinations ballooned process VA (~84 GiB total_vm, ~57 MiB page tables) and
OOM-killed a 15 GiB host session. That map was **not** needed for ReBAR DMA.

**Fix (still P2P / still dma_buf / no host bounce for the hot path):**
- Export the VRAM pool as **one** dma_buf once (device memory / ReBAR).
- **Never** map the whole pool into host VA.
- Per in-flight read: page-aligned **window mmap** of only the destination
  (≈ one expert page); `munmap` on CQE. Peak host VA ≈ `QD × page_size`.
- NVMe → VRAM still direct via the windowed dma_buf mapping (same physical
  pages as the HIP pool). No host staging round-trip on the P2P path.
- Weight-paging still forces GGUF `use_mmap=false` (model file on NVMe, not
  mapped into RAM).

**A/B hygiene on 16G hosts:** ≤2 full server reloads per suite; check `free -h`
between runs; `ulimit -c 0` to avoid drkonqi dump thrash.

#### IOSQE_ASYNC fix (same day) — submit was the wall

Phase timers on `ensure_batch` showed **submit_ms ≫ wait_ms** (e.g. 52.6 s vs 1.0 s): P2P `io_uring_submit` was completing I/O **inline**, so multi-QD never overlapped.

| | before `IOSQE_ASYNC` | after |
|--|---------------------:|------:|
| ensure_batch_submit_ms | ~52600 | **~27** |
| ensure_batch_wait_ms | ~1000 | **~28000** |
| ensure_batch_gb_s | ~1.9 | **~3.65** |
| warm t/s (n=32, 6500, sticky) | ~1.74 | **~1.93** |

Force `IOSQE_ASYNC | IOSQE_FIXED_FILE` on P2P (+ host uring) SQEs. Still below fio O_DIRECT ~6.2 at QD=6 (avg batch ~8.6) — remaining gap is wait-side / PCIe-to-VRAM random, not inline submit.

#### DFlash @ 6500 (same stack)

`accept=0` under default strip is **policy**, not draft quality: logs show `target-verify draft N -> 0 (WP_SPEC_VERIFY_MAX)`. Draft still runs as **paging oracle** only.

| config | warm t/s | page_ins | accept | notes |
|--------|---------:|---------:|-------:|-------|
| nodraft + ebatch | ~1.49–1.69 | ~23.0k | - | variance across runs |
| dflash n-max=1 strip | **~1.64** | ~23.1k | 0 (strip) | cheaper oracle; sticky hits up |
| dflash n-max=4 VERIFY_MAX=1 | ~1.38 | **24.6k** | 25% | free tokens but more thrash |

**Oracle strip is correct default** until page_ins drop further. VERIFY_MAX=1 not free on this workload.

### C. TB3 residual transport (parked — diminishing)

**Status:** mid-graph fixed (T5–T7). Further TB3 transport investment is low priority.

#### Diagnosis (why ~+380 ms/tok is not "TB3 is too slow for residuals")

Per-layer intended cut:

```
ROCm1 (attn island)  --residual-->  ROCm0 (MoE + shexp)
ROCm1 (next attn)    <--residual--  ROCm0 (ffn_out)
```

For decode `n=1`, residual is **tens of KB** (n_embd * hc streams). Pure BW at 2.7 GB/s for all layer crossings is **<< 1 ms/token**. Measured hetero excess is ~**380 ms/tok** vs single-card nodraft — so the tax is **not residual payload size**.

What the stack does today:

1. **No usable peer path** for mixed gfx1201 + gfx1030 (peer-async historically page-faulted; default is CPU stage).
2. **Synchronous host staging** in HIP (`ggml_cuda_Memcpy2DPeerAsync` no-P2P path): stream sync -> D2H into pinned slab -> H2D. Comment in tree: async D2H/H2D was tried and made things worse without full pipelining (`GGML_HIP_COPY_STRATEGY=stage`).
3. **Scheduler serializes devices** (`ggml_backend_sched_compute_splits`): compute split A, copy inputs, compute split B. Full pipeline parallelism disabled under WP hetero (correctly — it tried ~94 GiB multi-device compute buffers).
4. **WP eval_cb full backend sync** after ensure-needed ranges — correct for paging, multiplies walls with every device split.
5. **Not all hetero tax is TB3** — attention/FA on gfx1030 eGPU and draft on ROCm1 are slower than dense-on-R9700 single-card. Measure before over-investing in copy path.

Mental model:

```
Today (serialized):
  [attn ROCm1] --sync-- [stage D2H|H2D] --sync-- [page+MoE ROCm0] --sync-- [stage] -- [attn]

Wanted (streamed residual):
  [attn ROCm1] ----async residual----> [page+MoE ROCm0]
       ^                                      |
       +--------- async residual <------------+
  (compute on one card overlaps DMA of the other)
```

"Stream" means **overlap + permanent bounce buffers + fewer boundaries**, not chunking already-tiny residuals into smaller pieces.

#### TB3 / activation levers (ordered)

| ID | Lever | Idea | Expected gain | Risk | Status |
|----|--------|------|---------------|------|--------|
| T0 | **Instrument** | Count stage/peer copies, bytes, wall + stream-sync + **direction** (`GGML_HIP_COPY_STATS=1`) | Know if bounce is 10 vs 200 ms of the ~380 ms | low | **landed** |
| T1 | **Sticky residual bounce buffer** | Shared 8-slab pinned host ring; 256 KiB floor | Cut host malloc thrash | low | **landed** |
| T2 | **Async stage + events** | D2H/H2D async + events | Hide bounce under compute | medium | deferred (WP eval_cb full-sync walls; stream-sync only ~0.5% of stage wall) |
| T3 | **Activation pipeline (not full PP)** | Double-buffer split inputs only | Overlap devices | medium-high | deferred (same reason) |
| T4 | **Fewer boundaries / FFN island** | `ffn_norm`, `ffn_gate_inp`, `tid2eid`, `hc_ffn_*` + shexp + experts on **paging** GPU | Keep MoE block intra-device | medium | **landed** (`wp-router`) |
| T5 | **Peer copy** | `hipMemcpyPeer` when `canAccessPeer` | Skip host stage | high risk | **ruled out** (segfaults on gfx1201+gfx1030 even when canAccessPeer=1) |

**T0 usage:** `GGML_HIP_COPY_STATS=1` (atexit dump); optional `GGML_HIP_COPY_STATS_EVERY=N`.

### Measured hetero nodraft n=16 (2000 slots)

| | pre-T4 (`het-xdev-stats5`) | **T4 FFN island** (`het-t4-ffn-island`) | first T0 (`het-xdev-stats`) |
|--|---------------------------|----------------------------------------|------------------------------|
| t/s | 0.709 | **0.822** | 0.756 |
| stage wall | 6.74 s | **5.19 s** | 5.58 s |
| avg us/copy | 172 | **131** | 142 |
| stage count | 39186 | 39528 | 39186 |
| dir0->1 avg | 204 us | **153 us** | - |
| dir1->0 avg | 144 us | **108 us** | - |
| page_ins | 12710 | 12662 | 12710 |

Single-card nodraft baseline remains ~**1.11** t/s. Hetero closed part of the gap (0.71 -> **0.82**).

### Hard constraints (this box)

- **No usable peer** for residual: `hipMemcpyPeer` crashes; always host-stage.
- Stage is **asymmetric** by direction (~1.5x slower one way).
- Stage is **many ~40 KiB copies** (~1600/token-step), not 86 residual blobs. T4 improved **latency/locality** more than raw copy count (count stayed flat).
- Contiguous `hipMemcpy` 1D microbench was ~15% faster than `hipMemcpy2D` but **faulted** under live hetero load; left disabled.

### Second pass (2026-07-08 evening) — Codex analysis + re-try

**Codex report:** `~/wp_logs/codex-tb3-analyze-report.md` (analysis only).

| Idea | Verdict | Action |
|------|---------|--------|
| Peer / cross-device D2D | WON'T WORK (segfault) | confirmed again |
| Zero-copy mapped host | WON'T WORK (Codex + cost) | skip |
| dma_buf bounce for activations | WON'T WORK without P2P | skip |
| hipSetDevice ambient cache | RISKY on ROCm multi-GPU | not used |
| WC host bounce (`GGML_HIP_STAGE_HOST=wc`) | SAFE TRY | **default on**; stage wall 5.19→**4.89 s**, t/s ~flat (0.82) |
| Name stats for crossings | SAFE TRY | **landed** — top names are `hc_attn_post` / `l_out` views |
| Batch split-input staging in sched | SAFE TRY tried | **landed but default OFF** — see below |
| Async chunk pipeline | RISKY (WP sync walls) | deferred |
| FFN island edge placement | SAFE TRY | already T4; names confirm residuals are layer edges |

**WC smoke** (`het-wc-stage.log`, nodraft n=16): t/s **0.811**, stage wall **4.89 s**, avg **124 us**, top copies: `hc_attn_post` / `l_out` (layer-boundary residual views).

**Batching microbench (host):** 8×40 KiB stages ~766 us vs 1×320 KiB ~178 us (~4×) when truly multi-copy.

**Sched multi-input batch (2026-07-09):** implemented (`GGML_HIP_STAGE_BATCH=1`):
- Queue in `cpy_tensor_async`, flush after split inputs in `ggml-backend.cpp` (before eval_cb/compute).
- Safe flush = one producer-stream sync wave + per-item host stage (no packed-blob; packing faulted).
- **Measured:** `batch_flush ≈ batch_items ≈ stage` → almost every split has **one** cross-device input. No multi-input amortization on DS4 hetero residual graph.
- t/s ~0.80, stage wall ~6.0 s (slightly **worse** than no-batch WC due to queue/flush overhead).
- **Default OFF.** Keep code for future multi-input splits; not a win on current layout.

### Mid-graph + eGPU pass (2026-07-09)

**Diagnosis (instrumented):**
- 100% of stages are **sched split inputs** (`from=sched/mm/cpy2d/other=N/0/0/0`), not mul_mat.
- Unnamed = 0. Top names were `hc_attn_post-* (view)` / `l_out-* (view)`.
- Root cause: DS4 `build_hc_post` does **hc x hc residual stream views** (4x4=16) per post; each view was a separate TB3 stage. Theory: ~2*43*16 ≈ 1376 view stages/tok vs measured ~1647.

**T5 view-parent coalesce** (`GGML_SCHED_VIEW_COALESCE=1` default ON; `=0` to disable; `GGML_SCHED_VIEW_COALESCE_MAX` bytes cap, default 4MiB):
- Stage **parent residual once**, rebind views into the local parent copy.

**T6 FFN-island activation pin** (`graph_get_cb` under WP hetero):
- Root cause: default `"norm" -> layer-home` pin forced `ffn_norm` onto resident/eGPU, then restaged the whole MoE chain across TB3.
- Fix: under WP multi-device overrides, **skip** the generic norm pin; pin `ffn_moe_*` / `hc_ffn*` / `ffn_norm` / `ffn_shexp` / `ffn_out` / `l_out` to the **paging** backend (resolved from `ffn_norm` weight buft).

Measured nodraft n=16 progression:

| | t/s | stages | stage wall | stage bytes |
|--|-----|--------|------------|-------------|
| pre-T4 | 0.709 | 39186 | 6.74 s | - |
| T4 FFN island weights | 0.822 | 39528 | 5.19 s | - |
| T5 view coalesce (`het-view-coalesce`) | 0.914 | 9682 | 2.62 s | 0.58 GB |
| T6 FFN act pin (`het-ffn-locality`) | 0.960 | 4534 | 0.81 s | 0.28 GB |
| **T7 hc_pre RMS pin** (`het-hc-rms-pin`) | **1.283** | **2987** | **0.85 s** | **0.14 GB** |
| single-card ref (earlier) | ~1.11 | 0 | 0 | 0 |

**T7 `node_*` root cause:** anonymous `RMS_NORM` in `build_hc_pre` (2/layer). Residual-sized `flat_norm` ran on residual's GPU then restaged for `mul_mat(hc_fn)` — double residual-sized traffic. Fix: `dsv4_pin_to_weight(sched, flat_norm, hc_fn)` (+ name `hc_flat_norm`).

Post-T7 families: residual edges only `hc_attn_post`/`l_out` (~70 MB each), tiny `ffn_moe_probs*`, `GET_ROWS`. **Mid-MoE + RMS double-copy gone.** Hetero now **beats** the earlier single-card ~1.11 t/s smoke.

**eGPU / next:**
1. Layout is healthy — residual edges are the structural minimum for dual-island.
2. **Pool growth on R9700** is the next capacity lever.
3. gfx1030 FA micro-opts optional; not blocking.

### Remaining TB3 / mid-graph upside

1. **Pool growth (paging)** — primary next step; residual tax ~0.14 GB/run.
2. Residual edge fusion (unlikely big win; already 1 parent/boundary).
3. True multi-input batch only if several same-dir copies per split.
4. T2/T3 async still blocked by WP eval_cb full sync.

#### Explicit non-goals on the TB3 path

- Putting **expert weights** on TB3.
- Multi-token target verify "to hide TB3" under WP (unions MoE actives -> thrash; already measured).
- Chunking residual into smaller TB3 pieces (already tiny; overhead would rise).
- Re-enabling full `pipeline_parallel` without a residual-only / single-device compute-buffer plan (94 GiB path).

#### Dive-in order (when we leave pool growth)

1. Grow pool + hetero nodraft A/B (B above).
2. **T0 instrument** — if bounce << 50 ms/tok, TB3 is a distraction; focus eGPU attn + paging.
3. Only then T1 -> T2 -> T3/T4 as timers justify.

### D. Draft / oracle polish (after or beside pool)

- **spec-draft-n-max=1 under WP strip** — only first draft token feeds oracle; cut draft GPU cost.
- **Conditional multi-row verify** — `WP_SPEC_VERIFY_MAX=1` only when hit_ratio high and free pool healthy.
- **Draft-only QD bump** — isolated higher QD for oracle waves (global QD>4 historically hung without demux).
- **Softmax prior** — weak without multi-row thrash or learned router cache; after hash oracle is maxed.

### E. Deferred / dead

- **Lower-bit expert quant** — deferred. DS4 Flash already QAT mix; revisit with ml8 later.
- **Faster / striped NVMe & full RAM expert tier** — dead on this box for 147 GB experts.
- **Native DS4 MTP head as linchpin** — earlier plan; **DFlash external draft** is the live spine for oracle + strip policy. Revisit native MTP only if DFlash plateau + model head is clearly available.

## Ruled out (measured) — with nuance

- **QD as single-token bottleneck** — depth-8 ~= depth-4; revive only for batched multi-token demand.
- **Inter-GPU P2P as free residual path** — not available as a real fabric on this pair; HIP stages.
- **Naive per-layer split without FA island / PP fix** — historically FA disable + ~40 GiB non-FA path, or ~94 GiB PP reserve. **Current hetero** (attention island + PP off + overrides) **loads and runs**; it is slower at 2000 slots, not broken. Pool growth + TB3 transport levers are how it may beat single-card.
- **Multi-wave cold draft submit / multi-row verify under WP** — thrash (page_ins 15k vs 12.7k); strip is default.

---

## The spine now

1. **Expert cache capacity on R9700** (grow slots once dense is off-card).  
2. **DFlash as paging oracle** (tid2eid; strip multi-row target).  
3. **TB3 residual transport quality** (overlap/stage, not payload size) — only after T0 proves bounce is on the critical path.

**Next action before TB3 code:** grow pool; optional hetero nodraft A/B; then T0 timers if hetero still loses.
